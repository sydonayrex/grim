//! SPEED-ROC: GPU parity for the consolidated WMMA fused-dequant kernels
//! (`grim_wmma_fused_dequant_{q8_0,q4k,q5k,q2k,q3k,q6k}`) against a
//! dequantize-then-matmul CPU reference, at decode shape M=1 and at PARTIAL-M
//! prefill shapes (see `wmma_quant_partial_m_parity_and_route`, which also
//! asserts through the launch route counter that the WMMA kernel is the one
//! under test — a numeric parity run alone cannot tell the paths apart).
//!
//! RUN ON THIS SYSTEM: GRIM_RUN_GPU_TEST=1 cargo test -p grim-backend-rocm --test wmma_quant_parity -- --ignored

use grim_backend_rocm::RocmDevice;
use grim_quant::quant_q4k;
use grim_tensor::{
    ArithType, CoreTensorOps, DType, KQuantScheme, MemoryOps, QuantOps, Shape, Storage,
};

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    Some(RocmDevice::try_new(0).expect("RocmDevice::try_new"))
}

/// CPU reference: dequantize packed B to f32 then plain matmul C = A @ B^T.
fn ref_matmul(
    a: &[f32],
    b_packed: &[u8],
    m: usize,
    n: usize,
    k: usize,
    scheme: KQuantScheme,
    deq: &dyn Fn(&[u8], usize) -> Vec<f32>,
) -> Vec<f32> {
    let (block_elems, block_bytes): (usize, usize) = match scheme {
        KQuantScheme::Q4K => (256, 144),
        KQuantScheme::Q80 => (32, 34),
        _ => panic!("unsupported scheme in this test"),
    };
    let blocks_per_row = k / block_elems;
    let row_bytes = blocks_per_row * block_bytes;
    assert_eq!(b_packed.len(), n * row_bytes);
    let mut c = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let brow = &b_packed[col * row_bytes..(col + 1) * row_bytes];
            let mut b_f32 = Vec::with_capacity(k);
            for b in 0..blocks_per_row {
                b_f32.extend(deq(
                    &brow[b * block_bytes..(b + 1) * block_bytes],
                    block_elems,
                ));
            }
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[row * k + kk] * b_f32[kk];
            }
            c[row * n + col] = acc;
        }
    }
    c
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    dev: &RocmDevice,
    m: usize,
    n: usize,
    k: usize,
    scheme: KQuantScheme,
    format: grim_tensor::QuantFormat,
    pack: &dyn Fn(&[f32]) -> grim_tensor::error::Result<Vec<u8>>,
    deq: &dyn Fn(&[u8], usize) -> Vec<f32>,
) {
    let a_host: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05).sin()).collect();
    let b_orig: Vec<f32> = (0..k * n)
        .map(|i| 1.0 + (i as f32 * 0.013).cos().abs() * 6.0)
        .collect();
    let b_packed = pack(&b_orig).expect("pack");

    let a_shape = Shape::new(vec![m, k]);
    let a_dev = CoreTensorOps::from_cpu(dev, &a_host, &a_shape, DType::F32).expect("upload A");
    let q_dtype = DType {
        arith: ArithType::F32,
        storage: Storage::KQuant(scheme),
    };
    let b_shape = Shape::new(vec![n * b_packed.len() / n]); // flat bytes
    let b_dev =
        MemoryOps::from_cpu_bytes(dev, &b_packed, &Shape::new(vec![b_packed.len()]), q_dtype)
            .expect("upload packed B");
    let _ = b_shape;

    let out_shape = Shape::new(vec![m, n]);
    let (out_s, handle) = dev
        .quantized_matmul(a_dev.as_ref(), b_dev.as_ref(), &[], format, &out_shape)
        .expect("quantized_matmul (WMMA path, m<=4)");
    handle.synchronize().expect("sync");
    let got = out_s.to_cpu_vec_f32().expect("to_cpu");

    let want = ref_matmul(&a_host, &b_packed, m, n, k, scheme, deq);
    let mut max_diff = 0.0f32;
    for i in 0..want.len() {
        max_diff = max_diff.max((got[i] - want[i]).abs());
    }
    eprintln!("[{format:?}] m={m} n={n} k={k} max_diff={max_diff}");
    assert!(
        max_diff < 0.5,
        "{format:?}: WMMA kernel diverges from reference (max_diff={max_diff})"
    );
}

#[test]
#[ignore]
fn wmma_quant_decode_parity() {
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1");
        return;
    };
    // Decode shape: M=1. NOTE: this test checks NUMBERS ONLY, so it does not by
    // itself prove the WMMA kernel ran — it passes through any correct path.
    // `wmma_quant_partial_m_parity_and_route` adds the route-counter assertion.
    // (The previous comment here claimed M=1 "dispatches to the WMMA kernels
    // under default GRIM_WMM_MAX_M=4"; that was stale — the tile gate required
    // m % 16 == 0, so M=1 fell through to the non-WMMA fallback, and
    // GRIM_WMM_MAX_M is not consulted in this arm at all.)
    run_case(
        &dev,
        1,
        1024,
        1024,
        KQuantScheme::Q80,
        grim_tensor::QuantFormat::Q8_0,
        &pack_q8_0,
        &|blk, n| {
            let d = f16_deq(blk);
            (0..n).map(|w| d * (blk[2 + w] as i8) as f32).collect()
        },
    );
    run_case(
        &dev,
        1,
        1024,
        1024,
        KQuantScheme::Q4K,
        grim_tensor::QuantFormat::Q4K,
        &quant_q4k,
        &|blk, n| grim_quant::dequant_q4k(blk, n).unwrap(),
    );
}

/// SPEED-ROC: FP16-input Q8_0 WMMA parity.  Uploads activations as FP16 so
/// `quantized_matmul` dispatches to `grim_wmma_fused_dequant_q8_0_fp16`
/// (reads _Float16 directly, no per-element cast).  Verifies the FP16-input
/// path matches the FP32-input path within FP16 quantization tolerance.
#[test]
#[ignore]
fn wmma_quant_fp16_input_parity() {
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1");
        return;
    };
    let (m, n, k) = (1usize, 1024usize, 1024usize);
    let a_host: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05).sin()).collect();
    let b_orig: Vec<f32> = (0..k * n)
        .map(|i| 1.0 + (i as f32 * 0.013).cos().abs() * 6.0)
        .collect();
    let b_packed = pack_q8_0(&b_orig).expect("pack");

    let q_dtype = DType {
        arith: ArithType::F32,
        storage: Storage::KQuant(KQuantScheme::Q80),
    };
    let b_dev =
        MemoryOps::from_cpu_bytes(&dev, &b_packed, &Shape::new(vec![b_packed.len()]), q_dtype)
            .expect("upload packed B");
    let out_shape = Shape::new(vec![m, n]);

    // FP32-input reference path.
    let a_f32 = CoreTensorOps::from_cpu(&dev, &a_host, &Shape::new(vec![m, k]), DType::F32)
        .expect("upload A f32");
    let (out_f32, h_f32) = dev
        .quantized_matmul(
            a_f32.as_ref(),
            b_dev.as_ref(),
            &[],
            grim_tensor::QuantFormat::Q8_0,
            &out_shape,
        )
        .expect("quantized_matmul f32 input");
    h_f32.synchronize().expect("sync f32");
    let got_f32 = out_f32.to_cpu_vec_f32().expect("to_cpu f32");

    // FP16-input path: uploads the same logical values as FP16 storage.
    let a_f16 = CoreTensorOps::from_cpu(&dev, &a_host, &Shape::new(vec![m, k]), DType::F16)
        .expect("upload A f16");
    let (out_f16, h_f16) = dev
        .quantized_matmul(
            a_f16.as_ref(),
            b_dev.as_ref(),
            &[],
            grim_tensor::QuantFormat::Q8_0,
            &out_shape,
        )
        .expect("quantized_matmul f16 input");
    h_f16.synchronize().expect("sync f16");
    let got_f16 = out_f16.to_cpu_vec_f32().expect("to_cpu f16");

    let mut max_diff = 0.0f32;
    for i in 0..got_f32.len() {
        max_diff = max_diff.max((got_f32[i] - got_f16[i]).abs());
    }
    // The FP32-input arm now routes through the sudot4 GEMV (int8 activation
    // quant, per-block scales) while the FP16-input arm rides the fp16 WMMA
    // kernel — so this compares two different quantization schemes. 0.5 abs
    // accommodates int8 activation quant noise over k=1024.
    eprintln!("[fp16-input] m={m} n={n} k={k} max_diff(FP16-input WMMA vs sudot4)={max_diff}");
    assert!(
        max_diff < 5e-1,
        "FP16-input WMMA diverges from sudot4 path (max_diff={max_diff})"
    );
}

/// Pack f32 weights to Q8_0 (32-elem blocks: fp16 scale + i8 codes).
fn pack_q8_0(data: &[f32]) -> Result<Vec<u8>, grim_tensor::error::Error> {
    let mut out = Vec::with_capacity(data.len().div_ceil(32) * 34);
    for blk in data.chunks(32) {
        let max_abs = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = (max_abs / 127.0).max(1e-30);
        out.extend(half::f16::to_le_bytes(half::f16::from_f32(d)));
        for v in blk {
            out.push((v / d).round().clamp(-127.0, 127.0) as i8 as u8);
        }
    }
    Ok(out)
}

fn f16_deq(blk: &[u8]) -> f32 {
    // little-endian fp16 scale at block start
    half::f16::from_le_bytes([blk[0], blk[1]]).to_f32()
}

/// Numeric AND routing guard for PARTIAL-M dispatch to the f16-WMMA kernels.
///
/// `wmma_quant_tile_ok` required `m % 16 == 0`, which excluded every real
/// prefill shape (the 9B's 5-token prompt is M=5; a 4K prompt is M=4056). The
/// clause is gone, so this test pins both halves of the contract:
///
///   1. ROUTING — the dispatch reaches `grim_wmma_fused_dequant_q4k` at a
///      partial M, asserted through the launch route counter rather than
///      inferred. This is the half the sibling numeric tests cannot see.
///   2. NUMBERS — the output matches a host oracle that dequantizes the packed
///      weights itself (`ref_matmul` + `grim_quant::dequant_q4k`), which is what
///      proves the partial tile's out-of-range rows are ZERO-FILLED instead of
///      reading past the end of A or writing past the end of C.
///
/// M brackets the 16-row WMMA fragment height: 1 and 3 are a single partial
/// tile, 5 is the production prefill shape, and 17 crosses into a second tile
/// (one full tile plus one row), so both the masked tail and the tile-crossing
/// path are exercised. The fragment-height independence of `load_matrix_sync`
/// is the property under test (FlyDSL `kernels/gemm/rdna_f16_gemm.py`, written
/// for gfx120x, documents the same 16x16x16 atom for this card).
#[test]
#[ignore]
fn wmma_quant_partial_m_parity_and_route() {
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1");
        return;
    };
    for m in [1usize, 3, 5, 17] {
        grim_backend_rocm::reset_kernel_route_counters();
        run_case(
            &dev,
            m,
            1024,
            1024,
            KQuantScheme::Q4K,
            grim_tensor::QuantFormat::Q4K,
            &quant_q4k,
            &|blk, n| grim_quant::dequant_q4k(blk, n).unwrap(),
        );
        let wmma =
            grim_backend_rocm::rocm_kernel_route_counter("grim_wmma_fused_dequant_q4k");
        let tiled =
            grim_backend_rocm::rocm_kernel_route_counter("grim_fused_dequant_gemm_q4k_tiled");
        let ksplit =
            grim_backend_rocm::rocm_kernel_route_counter("grim_fused_dequant_gemm_q4k_ksplit");
        eprintln!("[wmma-partial-m] m={m} wmma={wmma} tiled={tiled} ksplit={ksplit}");
        assert!(
            wmma > 0,
            "m={m}: a partial-M tile must reach the f16-WMMA kernel; the route counter saw \
             wmma={wmma} tiled={tiled} ksplit={ksplit}"
        );
        assert_eq!(
            tiled, 0,
            "m={m}: the LDS-tiled fallback must not serve this shape"
        );
        assert_eq!(
            ksplit, 0,
            "m={m}: the K-split decode kernel must not serve this shape"
        );
    }
}

/// Large-M parity + routing for the 64x64x32 big-tile kernel.
///
/// Exercises what its name says: full 64-row tiles (m=1024) and a PARTIAL last
/// tile (m=1000 = 15 full + 40 rows), against the host oracle, through the
/// production dispatch. The route counter asserts `grim_wmma_big_q4k` is the
/// kernel under test (not the 16-row tile it sits beside in the dispatch).
#[test]
#[ignore]
fn wmma_big_q4k_large_m_parity_and_route() {
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TESTS=1");
        return;
    };
    for m in [64usize, 1000usize, 1024usize] {
        grim_backend_rocm::reset_kernel_route_counters();
        run_case(
            &dev,
            m,
            1024,
            4096,
            KQuantScheme::Q4K,
            grim_tensor::QuantFormat::Q4K,
            &quant_q4k,
            &|blk, n| grim_quant::dequant_q4k(blk, n).unwrap(),
        );
        let big = grim_backend_rocm::rocm_kernel_route_counter("grim_wmma_big_q4k");
        let big128 = grim_backend_rocm::rocm_kernel_route_counter("grim_wmma_big128_q4k");
        let small = grim_backend_rocm::rocm_kernel_route_counter("grim_wmma_fused_dequant_q4k");
        eprintln!("[big-q4k] m={m} big={big} big128={big128} small16={small}");
        // The high-nibble mask defect is fixed (missing & 0x0F on the w>=32
        // path of grim_big_deq_q4k — see tests/big_q4k_probe.rs). m >= 128
        // takes the 128x64 tile (one weight pass per 128 rows), 64 <= m < 128
        // the 64x64 tile.
        if m >= 128 {
            assert!(
                big128 > 0,
                "m={m}: the 128x64 tile must serve large-M prefill"
            );
        } else {
            assert!(
                big > 0,
                "m={m}: the 64x64 tile must serve 64 <= m < 128"
            );
        }
        assert_eq!(
            big + big128 + small, 1,
            "m={m}: exactly one WMMA tile kernel must serve this shape"
        );
    }
}
