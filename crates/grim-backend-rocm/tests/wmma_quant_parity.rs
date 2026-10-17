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
use grim_tensor::provider::TensorProvider;
use grim_tensor::{
    ArithType, CoreTensorOps, DType, KQuantScheme, MemoryOps, QuantOps, Shape, Storage,
};

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    Some(RocmDevice::try_new(0).expect("RocmDevice::try_new"))
}

/// The kernel route counters are process-global, so two tests that launch the
/// same WMMA kernel concurrently both increment the counter the other is
/// asserting on. Under the default parallel `cargo test` this made
/// `wmma_big_q4k_large_m_parity_and_route` fail with `small16=1` — a stray
/// launch from a sibling test, not a routing defect (it passes with
/// `--test-threads=1`). Every test in this file that launches or reads the
/// counters takes this lock.
static ROUTE_COUNTER_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn route_counter_lock() -> std::sync::MutexGuard<'static, ()> {
    ROUTE_COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Resolve a workspace-relative checkpoint path, or `None` if absent.
///
/// `cargo test` sets the CWD to the *package* root
/// (`crates/grim-backend-rocm`), not the workspace root, so a bare
/// `models/...` path never resolves. Try the CWD first, then walk up from the
/// crate manifest.
fn find_checkpoint(rel: &str) -> Option<std::path::PathBuf> {
    let direct = std::path::PathBuf::from(rel);
    if direct.exists() {
        return Some(direct);
    }
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for up in manifest.ancestors() {
        let cand = up.join(rel);
        if cand.exists() {
            return Some(cand);
        }
    }
    None
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
    // NaN-aware: `f32::max` IGNORES NaN, so the previous
    // `max_diff = max_diff.max((got - want).abs())` reported `max_diff=0` for a
    // kernel that emitted all-NaN. That is exactly how a broken Q4_K dequant
    // (NaN on every real weight) rode this gate green. Count non-finite
    // differences explicitly.
    let mut max_diff = 0.0f32;
    let mut nonfinite = 0usize;
    for i in 0..want.len() {
        let d = (got[i] - want[i]).abs();
        if d.is_finite() {
            if d > max_diff {
                max_diff = d;
            }
        } else {
            nonfinite += 1;
        }
    }
    eprintln!("[{format:?}] m={m} n={n} k={k} max_diff={max_diff} nonfinite={nonfinite}");
    assert_eq!(
        nonfinite, 0,
        "{format:?}: WMMA kernel produced {nonfinite} non-finite outputs (NaN/Inf)"
    );
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
    let _route = route_counter_lock();
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
/// SPEED-ROC: **IQ3_XXS decode-path parity** through `linear_decode_into`.
///
/// This file's coverage was `Q4K` and `Q80` only, so the `IQ3XXS` leg had no
/// parity gate anywhere in the tree. That matters because `IQ3_XXS` is the
/// single largest K-quant family in `models/qwen38-27b/Swift-1.5-Qwen3.8-27B-
/// GSQ-RCO-IQ3_XXS-mtp.gguf` (83 of 866 tensors), and that checkpoint produced
/// **all-NaN logits from step 0** on the same two cards — decoded as `!!!!`
/// forever, because token 0 is `!`.
///
/// WHICH KERNEL: on RDNA4 `is_dot4_arch` is true (`roc_device.rs:906`), and
/// with `k % 256 == 0` and `GRIM_DOT_GEMV` unset the guard at
/// `gemm_launchers.rs:2782` is FALSE, so the WMMA branch is skipped and
/// dispatch reaches `launch_fused_dequant_gemm_iq3xxs` at `:2960` — the
/// fused-dequant GEMM reading A as f32, the sibling of
/// `launch_fused_dequant_gemm_iq3s`, NOT `launch_wmma_fused_dequant_iq3xxs`
/// (that one is only reached when dot4 is unavailable or k is unaligned).
/// The test goes through the public `linear_decode_into` entry so it tracks
/// whatever the real dispatcher picks, rather than hardcoding one arm.
///
/// `linear_decode_into` is used rather than `quantized_matmul` because at M=1
/// the latter falls through to a non-WMMA fallback (see the note in
/// `wmma_quant_decode_parity`) and is not the 27B's decode entry.
///
/// Set `GRIM_DOT_GEMV=0` to force the WMMA branch and reach
/// `launch_wmma_fused_dequant_iq3xxs` instead.
///
/// GREEN on the default arm after the device-dequant fix (measured 2026-10-16
/// on gfx1200, real `blk.61.ffn_gate.weight` bytes): max_diff 1.8e-7,
/// nonfinite 0. It was 14/256 non-finite with max_diff ~1.0e6 before.
///
/// STILL RED on the `GRIM_DOT_GEMV=0` arm: 256/256 non-finite. That arm
/// (`grim_wmma_fused_dequant_iq3xxs`, wmma_iq_gemm.rs) shares the corrected
/// `dequant_iq3xxs` and the 98-byte stride, so the residual NaN is in that
/// kernel's own tile/writeback logic, not in the dequant. It is not the default
/// dispatch, so it is left tracked rather than silently masked: run it with
/// `GRIM_DOT_GEMV=0` and this test to reproduce.
#[test]
fn wmma_iq3xxs_decode_path_parity() {
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1");
        return;
    };
    let _route = route_counter_lock();

    // IQ3_XXS geometry: 256 weights per 98-byte super-block (ggml-common.h;
    // mirrored in `grim-format/src/gguf.rs` at the type-id 18 and byte-size
    // tables). `linear_decode_into` requires k % 256 == 0.
    const BLOCK_ELEMS: usize = 256;
    const BLOCK_BYTES: usize = 98;
    let (m, n, k) = (1usize, 256usize, 512usize);
    assert_eq!(k % BLOCK_ELEMS, 0, "k must be a multiple of the IQ3_XXS block");

    let a_host: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05).sin()).collect();

    // Real packed bytes, not synthetic: `dequant_iq3xxs` on arbitrary garbage
    // can produce non-finite scales, and the resulting NaN would be a
    // property of the fixture rather than of the kernel. Sourcing from the
    // actual checkpoint decodes bytes a real producer wrote.
    let Some(ckpt) = find_checkpoint(
        "models/qwen38-27b/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf",
    ) else {
        panic!("checkpoint missing — this gate must not silently skip; it is the \
                 only parity coverage for launch_wmma_fused_dequant_iq3xxs");
    };
    let ckpt = ckpt.to_string_lossy().to_string();
    let f = std::fs::File::open(&ckpt).expect("open checkpoint");
    let gguf = grim_format::gguf::read_gguf(std::io::BufReader::new(f)).expect("parse gguf header");
    let provider = grim_format::tprov::GgufProvider::open(&ckpt).expect("open provider");
    // First IQ3_XXS tensor large enough for n x k weights.
    let need_blocks = (n * k) / BLOCK_ELEMS;
    let t = gguf
        .tensors
        .iter()
        .filter(|t| {
            t.dtype == grim_format::gguf::GgufDType::IQ3_XXS
                && (t.size_bytes as usize) / BLOCK_BYTES >= need_blocks
        })
        .max_by_key(|t| t.size_bytes)
        .expect("an IQ3_XXS tensor at least n*k/256 blocks long must exist");
    let raw = provider
        .get_packed(&t.name)
        .expect("packed IQ3_XXS bytes from the real checkpoint");
    let b_packed = raw.bytes[..need_blocks * BLOCK_BYTES].to_vec();

    // CPU reference: decode the SAME bytes this kernel is handed.
    let mut b_f32: Vec<f32> = Vec::with_capacity(n * k);
    for b in 0..need_blocks {
        b_f32.extend(
            grim_quant::dequant_iq3xxs(
                &b_packed[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES],
                BLOCK_ELEMS,
            )
            .expect("host IQ3_XXS dequant"),
        );
    }
    assert!(
        b_f32.iter().all(|v| v.is_finite()),
        "reference decode of {} produced non-finite weights — the fixture is bad",
        t.name
    );

    let a_dev = CoreTensorOps::from_cpu(&dev, &a_host, &Shape::new(vec![m, k]), DType::F32)
        .expect("upload A");
    let q_dtype = DType {
        arith: ArithType::F32,
        storage: Storage::KQuant(KQuantScheme::IQ3XXS),
    };
    let b_dev = MemoryOps::from_cpu_bytes(
        &dev,
        &b_packed,
        &Shape::new(vec![n * k]),
        q_dtype,
    )
    .expect("upload packed IQ3_XXS B");
    let w_rocm = b_dev
        .as_any()
        .downcast_ref::<grim_backend_rocm::memory::storage::RocmStorage>()
        .expect("B is RocmStorage");

    let out = dev
        .from_cpu(
        &vec![0.0f32; n],
        &Shape::new(vec![n]),
        DType::F32,
    )
    .expect("out buffer");
    let out_rocm = out
        .as_any()
        .downcast_ref::<grim_backend_rocm::memory::storage::RocmStorage>()
        .expect("out is RocmStorage");

    // Scratch: the same sizing the Q8_0 decode path uses, generously rounded up.
    let act_q81 = dev
        .from_cpu(
        &vec![0.0f32; (k / 32) * 64],
        &Shape::new(vec![(k / 32) * 64]),
        DType::F32,
    )
    .expect("act_q81 scratch");
    let act_rocm = act_q81
        .as_any()
        .downcast_ref::<grim_backend_rocm::memory::storage::RocmStorage>()
        .expect("act_q81 is RocmStorage");

    let handle = dev
        .linear_decode_into(a_dev.as_ref(), w_rocm, out_rocm, act_rocm)
        .expect("linear_decode_into must accept IQ3XXS (the 27B decode path)");
    handle.synchronize().expect("sync");
    let got = out.to_cpu_vec_f32().expect("readback");

    // want = a @ B^T, computed in f32 from the decoded weights.
    let want: Vec<f32> = (0..n)
        .map(|col| {
            (0..k).map(|kk| a_host[kk] * b_f32[col * k + kk]).sum::<f32>()
        })
        .collect();

    // NaN-aware, same reasoning as `run_case`: `f32::max` ignores NaN, so an
    // all-NaN kernel would otherwise report max_diff=0 and pass. This is the
    // failure mode under investigation.
    let mut max_diff = 0.0f32;
    let mut nonfinite = 0usize;
    let mut first_bad = None;
    for i in 0..want.len() {
        let d = (got[i] - want[i]).abs();
        if d.is_finite() {
            if d > max_diff {
                max_diff = d;
            }
        } else {
            nonfinite += 1;
            if first_bad.is_none() {
                first_bad = Some((i, got[i], want[i]));
            }
        }
    }
    eprintln!(
        "[iq3xxs-decode-parity] {} m={m} n={n} k={k} max_diff={max_diff} \
         nonfinite={nonfinite}",
        t.name
    );
    assert_eq!(
        nonfinite, 0,
        "the IQ3XXS decode leg produced {nonfinite} non-finite outputs; \
         first bad index {:?} got={:?} want={:?}",
        first_bad.map(|x| x.0),
        first_bad.map(|x| x.1),
        first_bad.map(|x| x.2)
    );
    assert!(
        max_diff < 0.5,
        "IQ3XXS fused-dequant diverges from the reference (max_diff={max_diff}); \
         activations are quantized to q8_1 on the device and B is dequantized on \
         device, so this is a loose bound by design"
    );
}

#[test]
#[ignore]
fn wmma_quant_fp16_input_parity() {
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1");
        return;
    };
    let _route = route_counter_lock();
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
    let _route = route_counter_lock();
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
        let wmma = grim_backend_rocm::rocm_kernel_route_counter("grim_wmma_fused_dequant_q4k");
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
    let _route = route_counter_lock();
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
        let big64x128 = grim_backend_rocm::rocm_kernel_route_counter("grim_wmma_big64x128_q4k");
        let small = grim_backend_rocm::rocm_kernel_route_counter("grim_wmma_fused_dequant_q4k");
        eprintln!(
            "[big-q4k] m={m} big={big} big128={big128} big64x128={big64x128} small16={small}"
        );
        // The 64x128 tile (12.89 ms vs 128x64's 19.88 at m=4056 n=4096 k=4096)
        // owns every m >= 64 whose n is a multiple of 128 — the test's n=1024
        // qualifies — because halving the A-side passes beats everything else
        // now that the fill is the limiter.
        assert!(
            big64x128 > 0,
            "m={m}: the 64x128 tile must serve large-M prefill at n%128==0"
        );
        assert_eq!(
            big + big128 + big64x128 + small,
            1,
            "m={m}: exactly one WMMA tile kernel must serve this shape"
        );
    }
}

/// Pack `a` into llama.cpp Q8_1 super-blocks and return both the packed bytes
/// and the activations those bytes actually represent.
///
/// This exists because the two families of decode kernels do NOT consume the
/// same activations:
///
///   * The IQ family has explicit f32-act fused-dequant legs. They read `a`
///     directly and never look at the `act_q81` scratch.
///   * Everything else (IQ4_XS, Q2_K, Q4_K, Q6_K) routes to
///     `launch_dot4_*_q81_gemv`, which consumes Q8_1-quantized activations and
///     ignores `a`.
///
/// So the scratch must hold real Q8_1 blocks, and the CPU reference must be
/// computed from the SAME dequantized activations the kernel sees — otherwise
/// the reference is measuring activation quantization, not kernel correctness.
/// `layout::block_q8_1` is `fp16 d | fp16 s | int8 qs[32]`, 36 bytes, with
/// `d = amax/127` and `s = d * x_at_argmax`.
fn q8_1_pack(a: &[f32]) -> (Vec<u8>, Vec<f32>) {
    const G: usize = 32;
    let mut bytes = Vec::with_capacity(a.len() / G * 36);
    let mut eff = Vec::with_capacity(a.len());
    for ch in a.chunks(G) {
        let (amax, x_at_amax) = ch
            .iter()
            .fold((0f32, 0f32), |(m, x), &v| {
                if v.abs() > m {
                    (v.abs(), v)
                } else {
                    (m, x)
                }
            });
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let s = d * x_at_amax;
        bytes.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
        bytes.extend_from_slice(&half::f16::from_f32(s).to_le_bytes());
        for &v in ch {
            let q = (v * id).round().clamp(-128.0, 127.0) as i32;
            bytes.push(q as i8 as u8);
            // The kernels' low-bit correction adds `s` back on the sign planes,
            // so this is the value the kernel actually multiplies by.
            eff.push(q as f32 * d + if q < 0 { s } else { 0.0 });
        }
    }
    (bytes, eff)
}

/// Every K-quant scheme the 27B checkpoint actually contains, run through the
/// LIVE decode dispatch against a host dequant of the SAME real bytes.
///
/// Why this exists: IQ3_XXS shipped with a fabricated device dequant
/// (`(grid_idx + sub_idx*17) % 7 - 3.0f`) and a 96-byte stride across 83
/// tensors, giving all-NaN logits at step 0. With that fixed the model decodes
/// FINITE but INCOHERENT text, which points at a sibling scheme that is
/// wrong-but-finite — the same failure WhiteRaven already documented once in
/// this codebase ("same bytes percolated differently decode to finite,
/// plausible, wrong weights"). Guessing which scheme is the wrong move;
/// measuring all ten in one run is cheap.
///
/// Reports every scheme rather than stopping at the first failure, so a single
/// execution gives the whole picture.
///
/// After the IQ2_XXS / IQ2_XS / IQ4_XS fixes, on real checkpoint bytes at
/// m=1 n=256 k=512, gfx1200:
///
///   ok  IQ1_S      4 tensors   2.2e-7
///   ok  IQ2_XXS   30 tensors   1.5e-7   (was 1.0e+0 — entirely wrong values)
///   ok  IQ2_XS    32 tensors   2.5e-7   (was 1.0e+0 — entirely wrong values)
///   ok  IQ2_S     72 tensors   1.3e-7
///   ok  IQ3_XXS   83 tensors   2.3e-7
///   ok  IQ3_S     97 tensors   2.2e-7
///   ok  IQ4_XS    33 tensors   2.4e-7   (was 8/256 non-finite, max_diff 1.3e8)
///   ??  Q2_K      28 tensors   7.5e-3   UNINVESTIGATED
///   ??  Q4_K      19 tensors   2.2e-1   UNINVESTIGATED
///   ??  Q6_K       8 tensors   1.0e-1   UNINVESTIGATED
///
/// The whole IQ family is now clean at fp32 epsilon. The Q*_K trio is a
/// separate open question: they ride the dot4/q81 legs and may be a reference
/// or accumulation convention rather than a kernel defect. That has NOT been
/// established in either direction, which is why they are labelled `??` rather
/// than FAIL. Left #[ignore]d for the same reason.
///
///   cargo test -p grim-backend-rocm --test wmma_quant_parity \
///     kquant_scheme_sweep -- --ignored --nocapture --test-threads=1
#[test]
#[ignore = "IQ family clean; Q2_K/Q4_K/Q6_K still uninvestigated (see doc comment)"]
fn kquant_scheme_sweep() {
    use grim_format::gguf::GgufDType;
    let Some(dev) = gpu_device() else {
        return;
    };
    let _route = route_counter_lock();

    // (label, gguf dtype, KQuantScheme, bytes per 256-weight super-block,
    //  consumes act_q81). Block elems is 256 for every scheme here
    //  (llama.cpp QK_K) and the byte sizes come from grim-format's gguf.rs
    //  table, NOT from memory — the 96-vs-98 class of bug lived in exactly
    //  that assumption. IQ1_M is absent because grim-quant has no host
    //  `dequant_iq1m` to compare against (4 tensors; needs a reference first).
    let schemes: &[(&str, GgufDType, KQuantScheme, usize, bool)] = &[
        ("IQ1_S", GgufDType::IQ1_S, KQuantScheme::IQ1S, 50, false),
        ("IQ2_XXS", GgufDType::IQ2_XXS, KQuantScheme::IQ2XXS, 66, false),
        ("IQ2_XS", GgufDType::IQ2_XS, KQuantScheme::IQ2XS, 74, false),
        ("IQ2_S", GgufDType::IQ2_S, KQuantScheme::IQ2S, 82, false),
        ("IQ3_XXS", GgufDType::IQ3_XXS, KQuantScheme::IQ3XXS, 98, false),
        ("IQ3_S", GgufDType::IQ3_S, KQuantScheme::IQ3S, 110, false),
        // IQ4_XS used to be the `uses_q81 = true` case here (no f32-act leg, so
        // it fell to the dot4 catch-all and returned non-finite). It now has its
        // own fused-dequant leg, which reads A as raw f32 and ignores the
        // act_q81 scratch — so the reference must use raw `a_host` too. Leaving
        // this `true` charges the kernel for Q8_1 quantization it does not
        // perform, which showed up as a stubborn ~7.9e-3.
        ("IQ4_XS", GgufDType::IQ4_XS, KQuantScheme::IQ4XS, 136, false),
        ("Q2_K", GgufDType::Q2K, KQuantScheme::Q2K, 84, true),
        ("Q4_K", GgufDType::Q4K, KQuantScheme::Q4K, 144, true),
        ("Q6_K", GgufDType::Q6K, KQuantScheme::Q6K, 210, true),
    ];

    const BLOCK_ELEMS: usize = 256;
    let (m, n, k) = (1usize, 256usize, 512usize);
    let need_blocks = (n * k) / BLOCK_ELEMS;
    let a_host: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05).sin()).collect();
    let (q81_bytes, a_eff_q81) = q8_1_pack(&a_host);

    let ckpt = find_checkpoint(
        "models/qwen38-27b/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf",
    )
    .unwrap_or_else(|| {
        panic!(
            "checkpoint missing — this gate must not silently skip; it is the \
             only parity coverage for the IQ fused-dequant decode legs"
        )
    })
    .to_string_lossy()
    .to_string();
    let f = std::fs::File::open(&ckpt).expect("open checkpoint");
    let gguf = grim_format::gguf::read_gguf(std::io::BufReader::new(f)).expect("parse gguf");
    let provider = grim_format::tprov::GgufProvider::open(&ckpt).expect("open provider");

    let mut failures: Vec<String> = Vec::new();
    println!(
        "{:<9} {:>8} {:>8} {:>12} {:>10}  {}",
        "scheme", "tensors", "bytes", "max_diff", "nonfinite", "verdict"
    );

    for (label, gdt, scheme, block_bytes, uses_q81) in schemes {
        let tensors = gguf.tensors.iter().filter(|t| t.dtype == *gdt).count();
        let Some(t) = gguf
            .tensors
            .iter()
            .filter(|t| t.dtype == *gdt && (t.size_bytes as usize) / block_bytes >= need_blocks)
            .max_by_key(|t| t.size_bytes)
        else {
            println!("{label:<9} {tensors:>8} {block_bytes:>8} {:>12} {:>10}  NO-TENSOR", "", "");
            failures.push(format!("{label}: no tensor >= n*k blocks"));
            continue;
        };
        let raw = provider
            .get_packed(&t.name)
            .unwrap_or_else(|e| panic!("{label}: packed bytes for {}: {e}", t.name));
        let packed = raw.bytes[..need_blocks * block_bytes].to_vec();

        // CPU reference weights: decode the SAME bytes the kernel is handed.
        let mut w_f32: Vec<f32> = Vec::with_capacity(n * k);
        for b in 0..need_blocks {
            let blk = &packed[b * block_bytes..(b + 1) * block_bytes];
            let v = match *label {
                "IQ1_S" => grim_quant::dequant_iq1s(blk, BLOCK_ELEMS),
                "IQ2_XXS" => grim_quant::dequant_iq2xxs(blk, BLOCK_ELEMS),
                "IQ2_XS" => grim_quant::dequant_iq2xs(blk, BLOCK_ELEMS),
                "IQ2_S" => grim_quant::dequant_iq2s(blk, BLOCK_ELEMS),
                "IQ3_XXS" => grim_quant::dequant_iq3xxs(blk, BLOCK_ELEMS),
                "IQ3_S" => grim_quant::dequant_iq3s(blk, BLOCK_ELEMS),
                "IQ4_XS" => grim_quant::dequant_iq4xs(blk, BLOCK_ELEMS),
                "Q2_K" => grim_quant::dequant_q2k(blk, BLOCK_ELEMS),
                "Q4_K" => grim_quant::dequant_q4k(blk, BLOCK_ELEMS),
                "Q6_K" => grim_quant::dequant_q6k(blk, BLOCK_ELEMS),
                other => panic!("{other}: no host reference"),
            }
            .unwrap_or_else(|e| panic!("{label}: host dequant: {e}"));
            w_f32.extend(v);
        }
        if !w_f32.iter().all(|v| v.is_finite()) {
            println!("{label:<9} {tensors:>8} {block_bytes:>8} {:>12} {:>10}  BAD-FIXTURE", "", "");
            failures.push(format!("{label}: host reference non-finite (bad fixture)"));
            continue;
        }

        // Reference activations: raw f32 for the f32-act legs, the Q8_1
        // round-trip for the dot4 legs. Using the wrong one here would charge
        // the kernel for activation quantization it does not perform.
        let a_ref: &[f32] = if *uses_q81 { &a_eff_q81 } else { &a_host };

        let a_dev = CoreTensorOps::from_cpu(&dev, &a_host, &Shape::new(vec![m, k]), DType::F32)
            .expect("upload A");
        let b_dev = MemoryOps::from_cpu_bytes(
            &dev,
            &packed,
            &Shape::new(vec![n * k]),
            DType { arith: ArithType::F32, storage: Storage::KQuant(*scheme) },
        )
        .expect("upload packed B");
        let w = b_dev
            .as_any()
            .downcast_ref::<grim_backend_rocm::memory::storage::RocmStorage>()
            .expect("B is RocmStorage");
        let out = dev
            .from_cpu(&vec![0.0f32; n], &Shape::new(vec![n]), DType::F32)
            .expect("out");
        let out_r = out
            .as_any()
            .downcast_ref::<grim_backend_rocm::memory::storage::RocmStorage>()
            .expect("out is RocmStorage");
        // Scratch: must be >= need_q81 = m*(k/32)*36 BYTES and hold real Q8_1
        // blocks for the dot4 legs.
        let act_scratch = if *uses_q81 { q81_bytes.clone() } else { vec![0u8; (k / 32) * 64] };
        let act = MemoryOps::from_cpu_bytes(
            &dev,
            &act_scratch,
            &Shape::new(vec![act_scratch.len()]),
            DType::U8,
        )
        .expect("act_q81 scratch");
        let act_r = act
            .as_any()
            .downcast_ref::<grim_backend_rocm::memory::storage::RocmStorage>()
            .expect("act is RocmStorage");

        match dev.linear_decode_into(a_dev.as_ref(), w, out_r, act_r) {
            Err(e) => {
                println!(
                    "{label:<9} {tensors:>8} {block_bytes:>8} {:>12} {:>10}  REJECTED: {e}",
                    "", ""
                );
                failures.push(format!("{label}: dispatch rejected: {e}"));
                continue;
            }
            Ok(h) => {
                h.synchronize().expect("sync");
            }
        }
        let got = out.to_cpu_vec_f32().expect("readback");

        let mut want = vec![0.0f32; n];
        for col in 0..n {
            want[col] = (0..k).map(|kk| a_ref[kk] * w_f32[col * k + kk]).sum();
        }
        let scale = want.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-6);
        let mut max_diff = 0.0f32;
        let mut nonfinite = 0usize;
        for (g, wv) in got.iter().zip(want.iter()) {
            if !g.is_finite() || !wv.is_finite() {
                nonfinite += 1;
                continue;
            }
            max_diff = max_diff.max((g - wv).abs() / scale);
        }
        let ok = nonfinite == 0 && max_diff <= 1e-3;
        println!(
            "{label:<9} {tensors:>8} {block_bytes:>8} {max_diff:>12.3e} {nonfinite:>10}  {}",
            if ok { "ok" } else { "FAIL" }
        );
        if !ok {
            failures.push(format!(
                "{label} ({}): nonfinite={nonfinite} max_diff={max_diff:.3e}",
                t.name
            ));
        }
    }

    println!("\n=== {} failing scheme(s) ===", failures.len());
    for f in &failures {
        println!("  {f}");
    }
    assert!(
        failures.is_empty(),
        "k-quant decode parity failures:\n  {}",
        failures.join("\n  ")
    );
}
