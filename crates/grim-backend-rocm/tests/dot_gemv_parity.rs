//! SPEED-DOT (space-balls.md): GPU parity for the VOP3 dot-product GEMV
//! kernels (`grim_dot2_q80_gemv) against the
//! dequantize-then-matmul CPU reference, at decode shape M=1.
//!
//! Also cross-checks dot2 vs the WMMA path on the same inputs — both are
//! approximations of the same GEMM, so outputs must agree within combined
//! quantization tolerance.
//!
//! RUN ON THIS SYSTEM: GRIM_RUN_GPU_TEST=1 cargo test -p grim-backend-rocm --test dot_gemv_parity

use grim_backend_rocm::RocmDevice;
use grim_backend_rocm::AttentionOps;
use grim_tensor::{
    ArithType, BlockDtype, CoreTensorOps, DType, KQuantScheme, MemoryOps, QuantOps, Shape,
    Storage,
};
use std::sync::Mutex;

lazy_static::lazy_static! {
    static ref DOT_MUTEX: Mutex<()> = Mutex::new(());
}

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    std::panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new")).ok()
}

/// Build FP32 activation A[1,k] and pack B[n,k] to Q8_0 on the host
/// (identical layout to the WMMA tests: fp16 scale + 32 i8 codes per block).
fn build_q80_case(m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<u8>) {
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
    let b_f32: Vec<f32> = (0..n * k).map(|i| ((i % 13) as f32 - 6.0) * 0.04).collect();

    let blocks = k / 32;
    let row_bytes = blocks * 34;
    let mut packed = vec![0u8; n * row_bytes];
    for col in 0..n {
        let row_start = col * k;
        for blk in 0..blocks {
            let base = blk * 32;
            let mut max_abs = 0.0f32;
            for j in 0..32 {
                max_abs = max_abs.max(b_f32[row_start + base + j].abs());
            }
            let d = (max_abs / 127.0).max(1e-30);
            let bits = half::f16::from_f32(d).to_bits();
            let out = &mut packed[col * row_bytes + blk * 34..col * row_bytes + blk * 34 + 34];
            out[0] = (bits & 0xFF) as u8;
            out[1] = ((bits >> 8) & 0xFF) as u8;
            for j in 0..32 {
                let code =
                    ((b_f32[row_start + base + j] / d).round().clamp(-127.0, 127.0) as i8) as u8;
                out[2 + j] = code;
            }
        }
    }
    (a, packed)
}

/// CPU reference: C[1,n] = sum_k A[0,k] * (d_b[n] * code) in fp32.
fn reference_q80(a: &[f32], b_packed: &[u8], m: usize, n: usize, k: usize) -> Vec<f32> {
    let blocks = k / 32;
    let row_bytes = blocks * 34;
    let mut c = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let brow = &b_packed[col * row_bytes..(col + 1) * row_bytes];
            let mut acc = 0.0f32;
            for blk in 0..blocks {
                let base: &[u8] = &brow[blk * 34..blk * 34 + 34];
                let scale_bits = u16::from_le_bytes([base[0], base[1]]);
                let d = half::f16::from_bits(scale_bits).to_f32();
                for j in 0..32 {
                    let code = base[2 + j] as i8 as f32;
                    acc += a[row * k + blk * 32 + j] * d * code;
                }
            }
            c[row * n + col] = acc;
        }
    }
    c
}

fn max_diff(want: &[f32], got: &[f32]) -> f32 {
    assert_eq!(want.len(), got.len(), "length mismatch");
    want.iter()
        .zip(got.iter())
        .map(|(w, g)| (w - g).abs())
        .fold(0.0f32, f32::max)
}

fn upload_q80_case(
    dev: &RocmDevice,
    a: &[f32],
    b_packed: &[u8],
    m: usize,
    k: usize,
) -> Result<(Box<dyn grim_tensor::BackendStorage>, Box<dyn grim_tensor::BackendStorage>), Box<dyn std::error::Error>>
{
    let a_dev = CoreTensorOps::from_cpu(dev, a, &Shape::new(vec![m, k]), DType::F32)?;
    let q_dtype = DType {
        arith: ArithType::F32,
        storage: Storage::KQuant(KQuantScheme::Q80),
    };
    let b_dev = MemoryOps::from_cpu_bytes(dev, b_packed, &Shape::new(vec![b_packed.len()]), q_dtype)?;
    Ok((a_dev, b_dev))
}

fn run_quant_matmul(
    dev: &RocmDevice,
    a_dev: &dyn grim_tensor::BackendStorage,
    b_dev: &dyn grim_tensor::BackendStorage,
    n: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let out_shape = Shape::new(vec![1, n]);
    let (out_s, h) = dev.quantized_matmul(
        a_dev,
        b_dev,
        &[],
        grim_tensor::QuantFormat::Q8_0,
        &out_shape,
    )?;
    h.synchronize()?;
    Ok(out_s.to_cpu_vec_f32()?)
}

/// PASSED: dot2 GEMV matches the fp32 CPU reference within fp16-activation
/// quantization tolerance at the LFM2.5 decode shapes.
#[test]
fn dot2_q80_parity_vs_cpu() {
    let _lock = DOT_MUTEX.lock().unwrap();
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
        return;
    };
    unsafe {
        std::env::set_var("GRIM_DOT_GEMV", "1");
    }

    for (n, k) in [(1024usize, 1024usize), (3072, 1024), (4608, 1024)] {
        let (a, b_packed) = build_q80_case(1, n, k);
        let want = reference_q80(&a, &b_packed, 1, n, k);
        let (a_dev, b_dev) = upload_q80_case(&dev, &a, &b_packed, 1, k).expect("upload");
        let got = run_quant_matmul(&dev, a_dev.as_ref(), b_dev.as_ref(), n).expect("dot2 run");

        let diff = max_diff(&want, &got);
        eprintln!("[dot2-vs-cpu] n={n} k={k} max_diff={diff}");
        // int8 activation quantization adds ~scale/2 error per element over
        // K=1024 terms on top of Q8_0 weight noise — 0.5 absolute is generous.
        assert!(
            diff < 0.5,
            "dot2 GEMV diverges from CPU reference at n={n}: max_diff={diff}"
        );
    }
    unsafe {
        std::env::set_var("GRIM_DOT_GEMV", "0");
    }
}

/// Host-side q8_1 quantization matching `grim_quantize_q8_1` (36 bytes/block:
/// fp16 scale LE at [0..2], unused sum placeholder at [2..4], 32 i8 codes at
/// [4..36]). Only the scale + codes are read by the dot4 GEMV kernel.
fn host_quantize_q8_1(act: &[f32], m: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % 32, 0);
    let n_q_blocks = k / 32;
    let mut out = vec![0u8; m * n_q_blocks * 36];
    for row in 0..m {
        for blk in 0..n_q_blocks {
            let base = blk * 32;
            let mut amax = 0.0f32;
            for j in 0..32 {
                amax = amax.max(act[row * k + base + j].abs());
            }
            let inv_d = if amax > 1e-9 { 127.0 / amax } else { 0.0 };
            let d = (amax / 127.0).max(1e-30);
            let d_bits = half::f16::from_f32(d).to_bits();
            let off = row * n_q_blocks * 36 + blk * 36;
            out[off] = (d_bits & 0xFF) as u8;
            out[off + 1] = ((d_bits >> 8) & 0xFF) as u8;
            // bytes [2..4] are the unused sum field — left as zero.
            for j in 0..32 {
                let code = (act[row * k + base + j] * inv_d)
                    .round()
                    .clamp(-127.0, 127.0) as i8;
                out[off + 4 + j] = code as u8;
            }
        }
    }
    out
}

/// Pack an arbitrary f32 weight slice `[rows, k]` to Q8_0 bytes (34 bytes/block:
/// fp16 scale LE + 32 i8 codes). Mirrors `build_q80_case` but takes real data.
fn pack_q80(w: &[f32], rows: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % 32, 0);
    let blocks = k / 32;
    let row_bytes = blocks * 34;
    let mut packed = vec![0u8; rows * row_bytes];
    for col in 0..rows {
        let row_start = col * k;
        for blk in 0..blocks {
            let base = blk * 32;
            let mut max_abs = 0.0f32;
            for j in 0..32 {
                max_abs = max_abs.max(w[row_start + base + j].abs());
            }
            let d = (max_abs / 127.0).max(1e-30);
            let bits = half::f16::from_f32(d).to_bits();
            let out = &mut packed[col * row_bytes + blk * 34..col * row_bytes + blk * 34 + 34];
            out[0] = (bits & 0xFF) as u8;
            out[1] = ((bits >> 8) & 0xFF) as u8;
            for j in 0..32 {
                let code = ((w[row_start + base + j] / d).round().clamp(-127.0, 127.0) as i8) as u8;
                out[2 + j] = code;
            }
        }
    }
    packed
}

/// RED/GREEN for Item 1 TDD #2: one fused dot4 GEMV over the concatenated
/// QKV weight blob must reproduce the three separate GEMV outputs (sliced back
/// into q/k/v) within 1e-4 — same kernel, same math, only launch grouping differs.
#[test]
fn fused_qkv_gemv_parity() {
    let _lock = DOT_MUTEX.lock().unwrap();
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
        return;
    };
    unsafe {
        std::env::set_var("GRIM_DOT_GEMV", "1");
    }

    // LFM2.5-350M dense-attn shape: n_q=1024, n_kv=1024, hidden=1024.
    let hidden = 1024usize;
    let n_q = 1024usize;
    let n_kv = 1024usize;

    let mut seed = 0xF00Du64;
    let mut rand = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let act: Vec<f32> = (0..hidden).map(|_| rand()).collect();
    let wq_f32: Vec<f32> = (0..n_q * hidden).map(|_| rand()).collect();
    let wk_f32: Vec<f32> = (0..n_kv * hidden).map(|_| rand()).collect();
    let wv_f32: Vec<f32> = (0..n_kv * hidden).map(|_| rand()).collect();

    let q80 = DType {
        arith: ArithType::F32,
        storage: Storage::KQuant(KQuantScheme::Q80),
    };
    // Shared q8_1 activation (U8 prequant) — identical bytes for both paths.
    let q81_dev = {
        let q81_bytes = host_quantize_q8_1(&act, 1, hidden);
        MemoryOps::from_cpu_bytes(
            &dev,
            &q81_bytes,
            &Shape::new(vec![q81_bytes.len()]),
            DType {
                arith: ArithType::U8,
                storage: Storage::Native,
            },
        )
        .expect("upload q81 act")
    };
    let wq_dev =
        MemoryOps::from_cpu_bytes(&dev, &pack_q80(&wq_f32, n_q, hidden), &Shape::new(vec![n_q, hidden]), q80.clone())
            .expect("upload wq");
    let wk_dev =
        MemoryOps::from_cpu_bytes(&dev, &pack_q80(&wk_f32, n_kv, hidden), &Shape::new(vec![n_kv, hidden]), q80.clone())
            .expect("upload wk");
    let wv_dev =
        MemoryOps::from_cpu_bytes(&dev, &pack_q80(&wv_f32, n_kv, hidden), &Shape::new(vec![n_kv, hidden]), q80.clone())
            .expect("upload wv");

    // Independent CPU reference for each projection (dequantize-then-matmul in
    // fp32). This is the same reference the passing `dot2_q80_parity_vs_cpu`
    // test uses, so it anchors the fused path to the known-correct scalar math.
    let q_cpu = reference_q80(&act, &pack_q80(&wq_f32, n_q, hidden), 1, n_q, hidden);
    let k_cpu = reference_q80(&act, &pack_q80(&wk_f32, n_kv, hidden), 1, n_kv, hidden);
    let v_cpu = reference_q80(&act, &pack_q80(&wv_f32, n_kv, hidden), 1, n_kv, hidden);

    // Fused path: build the concatenated blob, launch ONE dot4 GEMV, slice the
    // output back into q/k/v. Same kernel + same q8_1 bytes as the separate
    // path → must match the CPU reference within fp16-accum + act-quant noise.
    let fused = dev
        .build_fused_qkv_q80(wq_dev.as_ref(), wk_dev.as_ref(), wv_dev.as_ref())
        .expect("build fused qkv");
    let q81_rocm = q81_dev
        .as_ref()
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .expect("q81 is RocmStorage");
    let fused_out = dev
        .launch_fused_qkv_dot4(q81_rocm, &fused.storage, fused.n_q, fused.n_k, fused.hidden)
        .expect("fused gemv");
    let fused_vec = fused_out.to_cpu_vec_f32().expect("fused d2h");
    let n_total = fused.n_total();
    assert_eq!(fused_vec.len(), n_total, "fused output length");
    let q_fused = &fused_vec[0..n_q];
    let k_fused = &fused_vec[n_q..n_q + n_kv];
    let v_fused = &fused_vec[n_q + n_kv..n_total];

    // fp16 scale quantization of the q8_1 activation plus the int8 weight
    // quantization leaves ~0.05-0.1 absolute error per output element.
    let tol = 0.2;
    let q_diff = max_diff(&q_cpu, q_fused);
    let k_diff = max_diff(&k_cpu, k_fused);
    let v_diff = max_diff(&v_cpu, v_fused);
    eprintln!(
        "[fused-qkv-parity] n_q={n_q} n_kv={n_kv} hidden={hidden} fused-vs-cpu q={q_diff:.4} k={k_diff:.4} v={v_diff:.4}"
    );
    assert!(q_diff < tol, "q diverges from CPU reference: {q_diff}");
    assert!(k_diff < tol, "k diverges from CPU reference: {k_diff}");
    assert!(v_diff < tol, "v diverges from CPU reference: {v_diff}");

    unsafe {
        std::env::set_var("GRIM_DOT_GEMV", "0");
    };
}

/// PASSED: dot2 and WMMA paths compute the same GEMM — outputs agree within
/// the combined quantization tolerance.
#[test]
fn dot2_matches_wmma() {
    let _lock = DOT_MUTEX.lock().unwrap();
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
        return;
    };
    let (n, k) = (1024usize, 1024usize);
    let (a, b_packed) = build_q80_case(1, n, k);
    let (a_dev, b_dev) = upload_q80_case(&dev, &a, &b_packed, 1, k).expect("upload");

    unsafe {
        std::env::set_var("GRIM_DOT_GEMV", "1");
    }
    let dot2 = run_quant_matmul(&dev, a_dev.as_ref(), b_dev.as_ref(), n).expect("dot2");
    unsafe {
        std::env::set_var("GRIM_DOT_GEMV", "0");
    }
    let wmma = run_quant_matmul(&dev, a_dev.as_ref(), b_dev.as_ref(), n).expect("wmma");

    let diff = max_diff(&dot2, &wmma);
    eprintln!("[dot2-vs-wmma] n={n} k={k} max_diff={diff}");
    assert!(
        diff < 0.15,
        "dot2 and WMMA disagree: max_diff={diff} (int8 act quant + fp16 accum noise)"
    );
}

/// Item 2 TDD #1: `rope_dev_base` (device base + internal step offset) must match
/// the existing host-position `grim_rope` kernel at max_diff ≤ 1e-6 (fp32 identical
/// math). Random q [1, heads*steps, head_dim] at base positions B ∈ {0, 17, 500}.
#[test]
fn rope_dev_base_parity() {
    let _lock = DOT_MUTEX.lock().unwrap();
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
        return;
    };

    let heads = 16usize;
    let steps = 1usize;
    let head_dim = 64usize; // LFM2.5 head_dim
    let elems = heads * steps * head_dim;

    let mut seed = 0xBABEu64;
    let mut rand = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let q: Vec<f32> = (0..elems).map(|_| rand()).collect();
    let rope_cfg = grim_tensor::RopeConfig::new(head_dim, 1_000_000.0);

    let out_shape = Shape::new(vec![1, steps * heads, head_dim]);
    let q_dev = CoreTensorOps::from_cpu(&dev, &q, &out_shape, DType::F32).expect("q upload");

    for base in [0u32, 17, 500] {
        // Reference: existing host-position rope. The middle dim is heads*steps,
        // so positions has one entry per (head,step); all equal to `base` for
        // decode (every head shares the base position).
        let positions = vec![base; heads * steps];
        let (ref_st, _rh) = dev.rope(q_dev.as_ref(), &positions, &rope_cfg, &out_shape).expect("ref rope");
        let ref_v = ref_st.to_cpu_vec_f32().expect("ref d2h");

        // Device-base path: upload one u32 (the base) to device memory. The kernel
        // reads it as an integer, so we bit-cast the u32 through f32 upload.
        let pos_bits = f32::from_bits(base);
        let pos_dev = dev.from_cpu(&[pos_bits], &Shape::new(vec![1]), DType::U32).expect("pos upload");
        let (dev_st, _dh) = dev
            .rope_dev_base(q_dev.as_ref(), pos_dev.as_ref(), &rope_cfg, &out_shape, heads, steps)
            .expect("rope_dev_base");
        let dev_v = dev_st.to_cpu_vec_f32().expect("dev d2h");

        let diff = max_diff(&ref_v, &dev_v);
        eprintln!("[rope-dev-base-parity] base={base} max_diff={diff:.8}");
        assert!(diff < 1e-6, "rope_dev_base diverges at base={base}: {diff}");
    }
}

/// SPEED-DOT (Phase 4.5a): `dot4_q4k_q81_gemv` (sudot4 + nibble unpack + two-dot decomposition)
/// matches the CPU reference within quantization tolerance.
#[test]
fn dot4_q4k_gemv_parity() {
    let _lock = DOT_MUTEX.lock().unwrap();
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
        return;
    };
    unsafe {
        std::env::set_var("GRIM_DOT_GEMV", "1");
    }

    let m = 1usize;
    let n = 128usize;
    let k = 256usize; // 1 Q4_K superblock per row

    let mut seed = 0x1337u64;
    let mut rand = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };

    let a_f32: Vec<f32> = (0..m * k).map(|_| rand()).collect();
    let b_f32: Vec<f32> = (0..n * k).map(|_| rand()).collect();
    let b_bytes = grim_quant::quant_q4k(&b_f32).expect("quant_q4k");

    let q4k_dtype = DType {
        arith: ArithType::F32,
        storage: Storage::KQuant(KQuantScheme::Q4K),
    };
    let b_dev = MemoryOps::from_cpu_bytes(
        &dev,
        &b_bytes,
        &Shape::new(vec![b_bytes.len()]),
        q4k_dtype,
    )
    .expect("upload q4k weights");

    let a_dev = CoreTensorOps::from_cpu(&dev, &a_f32, &Shape::new(vec![m, k]), DType::F32).unwrap();
    let out_shape = Shape::new(vec![m, n]);

    // Call through quantized_matmul with M=1 on ROCm (which routes through dot4_q4k_q81_gemv)
    let (c_storage, handle) = dev
        .quantized_matmul(
            a_dev.as_ref(),
            b_dev.as_ref(),
            &[],
            grim_tensor::QuantFormat::Q4K,
            &out_shape,
        )
        .expect("quantized_matmul dot4 q4k");
    handle.synchronize().expect("sync handle");
    let c_dev = c_storage.to_cpu_vec_f32().expect("d2h c");
    assert_eq!(c_dev.len(), m * n);

    // Compute CPU reference: dequantize Q4_K weights then matmul
    let mut c_cpu = vec![0.0f32; m * n];
    let row_bytes = (k / 256) * 144;
    for col in 0..n {
        let brow = &b_bytes[col * row_bytes..(col + 1) * row_bytes];
        let b_deq = grim_quant::dequant_q4k(brow, k).expect("dequant_q4k");
        let mut acc = 0.0f32;
        for kk in 0..k {
            acc += a_f32[kk] * b_deq[kk];
        }
        c_cpu[col] = acc;
    }

    let diff = max_diff(&c_cpu, &c_dev);
    eprintln!("[dot4-q4k-gemv-parity] n={n} k={k} max_diff={diff:.6}");
    assert!(
        diff < 0.5,
        "dot4 Q4_K GEMV diverges from CPU reference: {diff}"
    );

    unsafe {
        std::env::set_var("GRIM_DOT_GEMV", "0");
    }
}

/// SPEED-DOT: Q5_K dot4 GEMV execution and parity test at M=1 decode.
#[test]
fn dot4_q5k_gemv_parity() {
    let _lock = DOT_MUTEX.lock().unwrap();
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
        return;
    };
    unsafe {
        std::env::set_var("GRIM_DOT_GEMV", "1");
    }

    let m = 1usize;
    let n = 256usize;
    let k = 512usize;

    let mut seed = 0x5555u64;
    let mut rand = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };

    let a_f32: Vec<f32> = (0..m * k).map(|_| rand()).collect();
    let b_f32: Vec<f32> = (0..n * k).map(|_| rand()).collect();
    let b_bytes = grim_quant::quant_q5k(&b_f32).expect("quant_q5k");

    let q5k_dtype = DType {
        arith: ArithType::F32,
        storage: Storage::KQuant(KQuantScheme::Q5K),
    };
    let b_dev = MemoryOps::from_cpu_bytes(
        &dev,
        &b_bytes,
        &Shape::new(vec![b_bytes.len()]),
        q5k_dtype,
    )
    .expect("upload q5k weights");

    let a_dev = CoreTensorOps::from_cpu(&dev, &a_f32, &Shape::new(vec![m, k]), DType::F32).unwrap();
    let out_shape = Shape::new(vec![m, n]);

    let (c_storage, handle) = dev
        .quantized_matmul(
            a_dev.as_ref(),
            b_dev.as_ref(),
            &[],
            grim_tensor::QuantFormat::Q5K,
            &out_shape,
        )
        .expect("quantized_matmul dot4 q5k");
    handle.synchronize().expect("sync handle");
    let c_dev = c_storage.to_cpu_vec_f32().expect("d2h c");
    assert_eq!(c_dev.len(), m * n);

    // Compute CPU reference: dequantize Q5_K weights then matmul
    let mut c_cpu = vec![0.0f32; m * n];
    let row_bytes = (k / 256) * 176;
    for col in 0..n {
        let brow = &b_bytes[col * row_bytes..(col + 1) * row_bytes];
        let b_deq = grim_quant::dequant_q5k(brow, k).expect("dequant_q5k");
        let mut acc = 0.0f32;
        for kk in 0..k {
            acc += a_f32[kk] * b_deq[kk];
        }
        c_cpu[col] = acc;
    }

    let diff = max_diff(&c_cpu, &c_dev);
    eprintln!("[dot4-q5k-gemv-parity] n={n} k={k} max_diff={diff:.6}");
    assert!(
        diff < 0.5,
        "dot4 Q5_K GEMV diverges from CPU reference: {diff}"
    );

    unsafe {
        std::env::set_var("GRIM_DOT_GEMV", "0");
    }
}

/// SPEED-DOT: Q6_K dot4 GEMV execution and parity test at M=1 decode.
#[test]
fn dot4_q6k_gemv_parity() {
    let _lock = DOT_MUTEX.lock().unwrap();
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
        return;
    };
    unsafe {
        std::env::set_var("GRIM_DOT_GEMV", "1");
    }

    let m = 1usize;
    let n = 256usize;
    let k = 512usize;

    let mut seed = 0x6666u64;
    let mut rand = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };

    let a_f32: Vec<f32> = (0..m * k).map(|_| rand()).collect();
    let b_f32: Vec<f32> = (0..n * k).map(|_| rand()).collect();
    let b_bytes = grim_quant::quant_q6k(&b_f32).expect("quant_q6k");

    let q6k_dtype = DType {
        arith: ArithType::F32,
        storage: Storage::KQuant(KQuantScheme::Q6K),
    };
    let b_dev = MemoryOps::from_cpu_bytes(
        &dev,
        &b_bytes,
        &Shape::new(vec![b_bytes.len()]),
        q6k_dtype,
    )
    .expect("upload q6k weights");

    let a_dev = CoreTensorOps::from_cpu(&dev, &a_f32, &Shape::new(vec![m, k]), DType::F32).unwrap();
    let out_shape = Shape::new(vec![m, n]);

    let (c_storage, handle) = dev
        .quantized_matmul(
            a_dev.as_ref(),
            b_dev.as_ref(),
            &[],
            grim_tensor::QuantFormat::Q6K,
            &out_shape,
        )
        .expect("quantized_matmul dot4 q6k");
    handle.synchronize().expect("sync handle");
    let c_dev = c_storage.to_cpu_vec_f32().expect("d2h c");
    assert_eq!(c_dev.len(), m * n);

    // Compute CPU reference: dequantize Q6_K weights then matmul
    let mut c_cpu = vec![0.0f32; m * n];
    let row_bytes = (k / 256) * 210;
    for col in 0..n {
        let brow = &b_bytes[col * row_bytes..(col + 1) * row_bytes];
        let b_deq = grim_quant::dequant_q6k(brow, k).expect("dequant_q6k");
        let mut acc = 0.0f32;
        for kk in 0..k {
            acc += a_f32[kk] * b_deq[kk];
        }
        c_cpu[col] = acc;
    }

    let diff = max_diff(&c_cpu, &c_dev);
    eprintln!("[dot4-q6k-gemv-parity] n={n} k={k} max_diff={diff:.6}");
    assert!(
        diff < 0.5,
        "dot4 Q6_K GEMV diverges from CPU reference: {diff}"
    );

    unsafe {
        std::env::set_var("GRIM_DOT_GEMV", "0");
    }
}

/// SPEED-DOT-FUSED (Phase 4c): `launch_fused_gate_up_dot4` matches CPU reference.
#[test]
fn fused_gate_up_gemv_parity() {
    let _lock = DOT_MUTEX.lock().unwrap();
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
        return;
    };
    unsafe {
        std::env::set_var("GRIM_DOT_GEMV", "1");
    }

    let hidden = 1024usize;
    let n_gate = 1024usize;
    let n_up = 1024usize;

    let mut seed = 0xCAFEu64;
    let mut rand = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let act: Vec<f32> = (0..hidden).map(|_| rand()).collect();
    let w_gate_f32: Vec<f32> = (0..n_gate * hidden).map(|_| rand()).collect();
    let w_up_f32: Vec<f32> = (0..n_up * hidden).map(|_| rand()).collect();

    let q80 = DType {
        arith: ArithType::F32,
        storage: Storage::KQuant(KQuantScheme::Q80),
    };
    let q81_dev = {
        let q81_bytes = host_quantize_q8_1(&act, 1, hidden);
        MemoryOps::from_cpu_bytes(
            &dev,
            &q81_bytes,
            &Shape::new(vec![q81_bytes.len()]),
            DType {
                arith: ArithType::U8,
                storage: Storage::Native,
            },
        )
        .expect("upload q81 act")
    };
    let wg_dev = MemoryOps::from_cpu_bytes(
        &dev,
        &pack_q80(&w_gate_f32, n_gate, hidden),
        &Shape::new(vec![n_gate, hidden]),
        q80.clone(),
    )
    .expect("upload wg");
    let wu_dev = MemoryOps::from_cpu_bytes(
        &dev,
        &pack_q80(&w_up_f32, n_up, hidden),
        &Shape::new(vec![n_up, hidden]),
        q80.clone(),
    )
    .expect("upload wu");

    let gate_cpu = reference_q80(&act, &pack_q80(&w_gate_f32, n_gate, hidden), 1, n_gate, hidden);
    let up_cpu = reference_q80(&act, &pack_q80(&w_up_f32, n_up, hidden), 1, n_up, hidden);

    let fused = dev
        .build_fused_gate_up_q80(wg_dev.as_ref(), wu_dev.as_ref())
        .expect("build fused gate up");
    let q81_rocm = q81_dev
        .as_ref()
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .expect("q81 is RocmStorage");
    let fused_out = dev
        .launch_fused_gate_up_dot4(q81_rocm, &fused.storage, fused.n_gate, fused.n_up, fused.hidden)
        .expect("fused gate up gemv");
    let fused_vec = fused_out.to_cpu_vec_f32().expect("fused d2h");
    let n_total = fused.n_total();
    assert_eq!(fused_vec.len(), n_total, "fused output length");

    let gate_fused = &fused_vec[0..n_gate];
    let up_fused = &fused_vec[n_gate..n_total];

    let tol = 0.2;
    let gate_diff = max_diff(&gate_cpu, gate_fused);
    let up_diff = max_diff(&up_cpu, up_fused);
    eprintln!(
        "[fused-gate-up-parity] n_gate={n_gate} n_up={n_up} hidden={hidden} gate_diff={gate_diff:.4} up_diff={up_diff:.4}"
    );
    assert!(gate_diff < tol, "gate diverges from CPU reference: {gate_diff}");
    assert!(up_diff < tol, "up diverges from CPU reference: {up_diff}");

    unsafe {
        std::env::set_var("GRIM_DOT_GEMV", "0");
    }
}


/// SPEED-DOT-OPFUSE (Phase 4a): `rmsnorm_rope` matches unfused RMSNorm then RoPE.
#[test]
fn rmsnorm_rope_parity() {
    let _lock = DOT_MUTEX.lock().unwrap();
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
        return;
    };

    let heads = 4usize;
    let steps = 1usize;
    let head_dim = 64usize;
    let out_shape = Shape::new(vec![1, steps * heads, head_dim]);
    let total_elems = heads * steps * head_dim;

    let mut seed = 0x5EEDu64;
    let mut rand = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };

    let x: Vec<f32> = (0..total_elems).map(|_| rand()).collect();
    let w: Vec<f32> = (0..head_dim).map(|_| rand().abs() + 0.5).collect();
    let eps = 1e-5f32;
    let rope_cfg = grim_tensor::RopeConfig::new(head_dim, 10000.0);

    let x_dev = CoreTensorOps::from_cpu(&dev, &x, &out_shape, DType::F32).unwrap();
    let w_dev = CoreTensorOps::from_cpu(&dev, &w, &Shape::new(vec![head_dim]), DType::F32).unwrap();

    let positions = vec![42u32; steps * heads];

    // Reference: run on CPU / unfused math
    let mut ref_out = vec![0.0f32; total_elems];
    for h in 0..(heads * steps) {
        let base = h * head_dim;
        let mut ss = 0.0f32;
        for i in 0..head_dim {
            let v = x[base + i];
            ss += v * v;
        }
        let inv_rms = 1.0f32 / (ss / head_dim as f32 + eps).sqrt();
        let mut normed = vec![0.0f32; head_dim];
        for i in 0..head_dim {
            normed[i] = x[base + i] * inv_rms * w[i];
        }
        // apply RoPE
        let half = head_dim / 2;
        let pos = positions[h] as f32;
        for i in 0..half {
            let freq = 1.0f32 / 10000.0f32.powf((2.0 * i as f32) / head_dim as f32);
            let val = pos * freq;
            let sin_val = val.sin();
            let cos_val = val.cos();
            let a_idx = base + 2 * i;
            let b_idx = base + 2 * i + 1;
            let x1 = normed[2 * i];
            let x2 = normed[2 * i + 1];
            ref_out[a_idx] = x1 * cos_val - x2 * sin_val;
            ref_out[b_idx] = x2 * cos_val + x1 * sin_val;
        }
    }

    // Fused kernel on device
    let (fused_st, handle) = dev
        .rmsnorm_rope(
            x_dev.as_ref(),
            Some(w_dev.as_ref()),
            &positions,
            &rope_cfg,
            &out_shape,
            eps,
        )
        .expect("rmsnorm_rope launch");
    handle.synchronize().expect("sync handle");
    let dev_out = fused_st.to_cpu_vec_f32().expect("d2h");

    let diff = max_diff(&ref_out, &dev_out);
    eprintln!("[rmsnorm-rope-parity] max_diff={diff:.8}");
    assert!(diff < 1e-5, "rmsnorm_rope diverges: {diff}");
}

/// SPEED-DOT-OPFUSE (Phase 4d): `grim_silu_mul_quant_q8_1` parity test.
#[test]
fn silu_mul_quant_q81_parity() {
    let _lock = DOT_MUTEX.lock().unwrap();
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
        return;
    };

    let k = 256usize; // 8 blocks of 32
    let mut seed = 0xDEADC0DEu64;
    let mut rand = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32 / u32::MAX as f32) * 6.0 - 3.0
    };

    let gate_host: Vec<f32> = (0..k).map(|_| rand()).collect();
    let up_host: Vec<f32> = (0..k).map(|_| rand()).collect();

    let shape = Shape::new(vec![1, k]);
    let gate_dev = CoreTensorOps::from_cpu(&dev, &gate_host, &shape, DType::F32).expect("h2d gate");
    let up_dev = CoreTensorOps::from_cpu(&dev, &up_host, &shape, DType::F32).expect("h2d up");

    let n_blocks = k / 32;
    let q81_bytes = n_blocks * 36;
    let dst_storage = MemoryOps::alloc_storage(
        &dev,
        &Shape::new(vec![q81_bytes]),
        DType {
            arith: ArithType::U8,
            storage: Storage::Native,
        },
    )
    .expect("alloc q81 dst");

    let g_rocm = gate_dev
        .as_ref()
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .unwrap();
    let u_rocm = up_dev
        .as_ref()
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .unwrap();
    let dst_rocm = dst_storage
        .as_ref()
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .unwrap();

    dev.launch_silu_mul_quant_q8_1(g_rocm, u_rocm, dst_rocm, k)
        .expect("launch silu_mul_quant_q8_1");
    dev.synchronize();

    let gpu_bytes = dst_rocm.copy_to_host().expect("d2h");

    // Reference CPU calculation
    for blk in 0..n_blocks {
        let base = blk * 32;
        let mut act = [0.0f32; 32];
        let mut amax = 0.0f32;
        for j in 0..32 {
            let g = gate_host[base + j];
            let u = up_host[base + j];
            let silu = g / (1.0f32 + (-g).exp());
            let val = silu * u;
            act[j] = val;
            amax = amax.max(val.abs());
        }

        let d = amax / 127.0f32;
        let inv_d = if amax > 1e-9f32 { 127.0f32 / amax } else { 0.0f32 };
        let mut fsum = 0.0f32;
        let mut ref_q = [0i8; 32];
        for j in 0..32 {
            let mut q = (act[j] * inv_d).round() as i32;
            if q > 127 { q = 127; }
            if q < -127 { q = -127; }
            ref_q[j] = q as i8;
            fsum += q as f32;
        }

        let blk_offset = blk * 36;
        let d_raw = u16::from_le_bytes([gpu_bytes[blk_offset], gpu_bytes[blk_offset + 1]]);
        let s_raw = u16::from_le_bytes([gpu_bytes[blk_offset + 2], gpu_bytes[blk_offset + 3]]);
        let d_gpu = half::f16::from_bits(d_raw).to_f32();
        let s_gpu = half::f16::from_bits(s_raw).to_f32();

        let d_ref_f16 = half::f16::from_f32(d).to_f32();
        let s_ref_f16 = half::f16::from_f32(fsum * d).to_f32();

        assert!((d_gpu - d_ref_f16).abs() < 1e-4, "scale d mismatch at blk {blk}: gpu {d_gpu} vs ref {d_ref_f16}");
        assert!((s_gpu - s_ref_f16).abs() < 1e-3, "sum s mismatch at blk {blk}: gpu {s_gpu} vs ref {s_ref_f16}");

        for j in 0..32 {
            let q_gpu = gpu_bytes[blk_offset + 4 + j] as i8;
            let q_ref = ref_q[j];
            assert!((q_gpu - q_ref).abs() <= 1, "quant mismatch at blk {blk} j {j}: gpu {q_gpu} vs ref {q_ref}");
        }
    }
}

/// Phase 4.5c: f32 -> E4M3 with RNE — mirrors `grim_f32_to_fp8_e4m3` in
/// dot_gemv.rs bit-for-bit so the CPU reference uses identical activations.
fn f32_to_fp8_e4m3_host(f: f32) -> u8 {
    if f.is_nan() {
        return 0x7F;
    }
    let sign: u8 = if f.is_sign_negative() { 0x80 } else { 0x00 };
    let a = f.abs();
    if a >= 480.0 {
        return sign | 0x7E; // saturate to 448
    }
    let bits = a.to_bits();
    let m = bits & 0x7F_FFFF;
    let e = ((bits >> 23) & 0xFF) as i32;
    if e == 0 {
        return sign; // f32 subnormals are below E4M3 min
    }
    let big_e = e - 120; // E4M3 exponent (bias 7)
    if big_e >= 1 {
        let mut q = m >> 20;
        let r = m & 0xF_FFFF;
        if r > 0x8_0000 || (r == 0x8_0000 && (q & 1) == 1) {
            q += 1;
        }
        let mut big_e = big_e;
        if q == 8 {
            q = 0;
            big_e += 1;
        }
        if big_e > 15 {
            return sign | 0x7E; // saturate to 448
        }
        return sign | ((big_e as u8) << 3) | q as u8;
    }
    // Subnormal: value * 512 = 1.m * 2^(E+2), round to 3-bit code.
    let sh = (21 - big_e) as u32;
    let mant = 0x80_0000u32 | m;
    let mut q = mant >> sh;
    let r = mant & ((1u32 << sh) - 1);
    let half = 1u32 << (sh - 1);
    if r > half || (r == half && (q & 1) == 1) {
        q += 1;
    }
    if q == 0 {
        return sign;
    }
    sign | q as u8
}

/// Mirrors `fp8_e4m3_to_float_hip` (shared_device_fns.rs).
fn fp8_e4m3_to_f32_host(val: u8) -> f32 {
    let sign = (val >> 7) & 1 == 1;
    let exp = ((val >> 3) & 0x0F) as i32;
    let mant = (val & 0x07) as i32;
    if exp == 0xF {
        if mant == 7 {
            return f32::NAN;
        }
        return if sign { -448.0 } else { 448.0 };
    }
    let res = if exp != 0 {
        (1.0 + mant as f32 / 8.0) * (2.0f32).powi(exp - 7)
    } else {
        mant as f32 / 512.0
    };
    if sign {
        -res
    } else {
        res
    }
}

/// Phase 4.5c: `dot4_fp8_gemv` (V_DOT4_F32_FP8_FP8, activations quantized to
/// E4M3 in-register) must match a CPU reference that performs the identical
/// E4M3 quantization + dequantization.
#[test]
fn dot4_fp8_gemv_parity() {
    let _lock = DOT_MUTEX.lock().unwrap();
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
        return;
    };
    unsafe {
        std::env::set_var("GRIM_DOT_GEMV", "1");
    }

    let m = 1usize;
    let n = 128usize;
    let k = 256usize;

    let mut seed = 0xC0FFEEu64;
    let mut rand = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };

    // Activations in the E4M3-normal range; weights encoded from f32 values.
    let a_f32: Vec<f32> = (0..m * k).map(|_| rand() * 4.0).collect();
    let b_vals: Vec<f32> = (0..n * k).map(|_| rand() * 16.0).collect();
    let b_bytes: Vec<u8> = b_vals.iter().map(|&v| f32_to_fp8_e4m3_host(v)).collect();

    let fp8_dtype = DType {
        arith: ArithType::F32,
        storage: Storage::Block(BlockDtype::Fp8),
    };
    let b_dev = MemoryOps::from_cpu_bytes(
        &dev,
        &b_bytes,
        &Shape::new(vec![b_bytes.len()]),
        fp8_dtype,
    )
    .expect("upload fp8 weights");

    let a_dev = CoreTensorOps::from_cpu(&dev, &a_f32, &Shape::new(vec![m, k]), DType::F32).unwrap();
    let out_shape = Shape::new(vec![m, n]);

    let (c_storage, handle) = dev
        .quantized_matmul(
            a_dev.as_ref(),
            b_dev.as_ref(),
            &[],
            grim_tensor::QuantFormat::Fp8,
            &out_shape,
        )
        .expect("quantized_matmul dot4 fp8");
    handle.synchronize().expect("sync handle");
    let c_dev = c_storage.to_cpu_vec_f32().expect("d2h c");
    assert_eq!(c_dev.len(), m * n);

    // CPU reference: decode E4M3 weights, quantize activations identically, dot.
    let mut c_cpu = vec![0.0f32; m * n];
    for col in 0..n {
        let mut acc = 0.0f32;
        for kk in 0..k {
            let b = fp8_e4m3_to_f32_host(b_bytes[col * k + kk]);
            let a = fp8_e4m3_to_f32_host(f32_to_fp8_e4m3_host(a_f32[kk]));
            acc += a * b;
        }
        c_cpu[col] = acc;
    }

    let diff = max_diff(&c_cpu, &c_dev);
    eprintln!("[dot4-fp8-gemv-parity] n={n} k={k} max_diff={diff:.6}");
    assert!(
        diff < 1e-2,
        "dot4 FP8 GEMV diverges from CPU reference: {diff}"
    );

    unsafe {
        std::env::set_var("GRIM_DOT_GEMV", "0");
    }
}

/// Phase 4.5f: `dot4_q2k_q81_gemv` (sudot4 + two-dot decomposition) must match
/// the CPU dequant reference. Fixture encodes grim's Q2_K layout (84 bytes /
/// 256 weights) and the CPU side mirrors the kernel's Q8_1 activation quant.
#[test]
fn dot4_q2k_gemv_parity() {
    let _lock = DOT_MUTEX.lock().unwrap();
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
        return;
    };
    unsafe { std::env::set_var("GRIM_DOT_GEMV", "1"); }

    let m = 1usize;
    let n = 32usize;
    let k = 256usize;

    let mut seed = 0x5152u64;
    let mut rand = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };

    let a_f32: Vec<f32> = (0..m * k).map(|_| rand() * 3.0).collect();

    let f16_bits = |v: f32| -> [u8; 2] { half::f16::from_f32(v).to_le_bytes() };
    let mut b_bytes = vec![0u8; n * 84];
    for col in 0..n {
        let blk = &mut b_bytes[col * 84..(col + 1) * 84];
        let d = 0.02 + rand().abs() * 0.02;
        let dmin = 0.01 + rand().abs() * 0.01;
        blk[80..82].copy_from_slice(&f16_bits(d));
        blk[82..84].copy_from_slice(&f16_bits(dmin));
        for sub in 0..16 {
            let sc = (rand() as u8) % 16;
            let mi = (rand() as u8) % 16;
            blk[sub] = sc | (mi << 4);
            for w in 0..16 {
                let q = (rand() as u8) % 4;
                blk[16 + sub * 4 + w / 4] |= q << ((w % 4) * 2);
            }
        }
    }

    let q2k_dtype = DType { arith: ArithType::F32, storage: Storage::KQuant(KQuantScheme::Q2K) };
    let b_dev = MemoryOps::from_cpu_bytes(&dev, &b_bytes, &Shape::new(vec![b_bytes.len()]), q2k_dtype)
        .expect("upload q2k weights");
    let a_dev = CoreTensorOps::from_cpu(&dev, &a_f32, &Shape::new(vec![m, k]), DType::F32).unwrap();
    let out_shape = Shape::new(vec![m, n]);

    let (c_storage, handle) = dev
        .quantized_matmul(a_dev.as_ref(), b_dev.as_ref(), &[], grim_tensor::QuantFormat::Q8_0, &out_shape)
        .expect("quantized_matmul dot4 q2k");
    handle.synchronize().expect("sync handle");
    let c_dev = c_storage.to_cpu_vec_f32().expect("d2h c");

    // CPU reference: mirror the GPU dot4 path exactly. The GPU reconstructs each
    // activation as code_i * d_a (from grim_quantize_q8_1) and dots against the
    // dequantized weight. Quantize the CPU activations identically so the only
    // difference is GPU-vs-CPU float summation order.
    let q81_block = |vals: &[f32]| -> (f32, Vec<i8>) {
        let amax = vals.iter().fold(0.0f32, |mx, v| mx.max(v.abs()));
        let d = amax / 127.0;
        let inv = if amax > 1e-9 { 127.0 / amax } else { 0.0 };
        let codes = vals.iter().map(|&v| (v * inv).round() as i8).collect::<Vec<_>>();
        (d, codes)
    };
    let mut c_cpu = vec![0.0f32; m * n];
    for col in 0..n {
        let b_deq = grim_quant::dequant_q2k(&b_bytes[col * 84..(col + 1) * 84], k).expect("dequant_q2k");
        let mut acc = 0.0f32;
        for blk in 0..k / 32 {
            let vals = &a_f32[blk * 32..(blk + 1) * 32];
            let (d_a, codes) = q81_block(vals);
            for i in 0..32 {
                acc += (codes[i] as f32) * d_a * b_deq[blk * 32 + i];
            }
        }
        c_cpu[col] = acc;
    }

    let diff = max_diff(&c_cpu, &c_dev);
    eprintln!("[dot4-q2k-gemv-parity] n={n} k={k} max_diff={diff:.6}");
    assert!(diff < 0.5, "dot4 Q2_K GEMV diverges from CPU reference: {diff}");
    unsafe { std::env::set_var("GRIM_DOT_GEMV", "0"); }
}

/// Phase 4.5f: `dot4_q3k_q81_gemv` (sudot4 + two-dot + sign-plane correction)
/// must match the CPU dequant reference. Fixture encodes grim's Q3_K layout
/// (110 bytes / 256 weights, llama.cpp spec).
#[test]
fn dot4_q3k_gemv_parity() {
    let _lock = DOT_MUTEX.lock().unwrap();
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
        return;
    };
    unsafe { std::env::set_var("GRIM_DOT_GEMV", "1"); }

    let m = 1usize;
    let n = 32usize;
    let k = 256usize;

    let mut seed = 0x3353_0000u64;
    let mut rand = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };

    let a_f32: Vec<f32> = (0..m * k).map(|_| rand() * 3.0).collect();
    let f16_bits = |v: f32| -> [u8; 2] { half::f16::from_f32(v).to_le_bytes() };

    let mut b_bytes = vec![0u8; n * 110];
    for col in 0..n {
        let blk = &mut b_bytes[col * 110..(col + 1) * 110];
        for i in 0..32 { blk[i] = rand().to_bits() as u8; }
        for i in 0..64 { blk[32 + i] = rand().to_bits() as u8; }
        for i in 0..12 { blk[96 + i] = rand().to_bits() as u8; }
        blk[108..110].copy_from_slice(&f16_bits(0.01));
    }

    let q3k_dtype = DType { arith: ArithType::F32, storage: Storage::KQuant(KQuantScheme::Q3K) };
    let b_dev = MemoryOps::from_cpu_bytes(&dev, &b_bytes, &Shape::new(vec![b_bytes.len()]), q3k_dtype)
        .expect("upload q3k weights");
    let a_dev = CoreTensorOps::from_cpu(&dev, &a_f32, &Shape::new(vec![m, k]), DType::F32).unwrap();
    let out_shape = Shape::new(vec![m, n]);

    let (c_storage, handle) = dev
        .quantized_matmul(a_dev.as_ref(), b_dev.as_ref(), &[], grim_tensor::QuantFormat::Q8_0, &out_shape)
        .expect("quantized_matmul dot4 q3k");
    handle.synchronize().expect("sync handle");
    let c_dev = c_storage.to_cpu_vec_f32().expect("d2h c");

    // CPU reference: mirror the GPU dot4 path exactly (activation = code * d_a).
    let q81_block = |vals: &[f32]| -> (f32, Vec<i8>) {
        let amax = vals.iter().fold(0.0f32, |mx, v| mx.max(v.abs()));
        let d = amax / 127.0;
        let inv = if amax > 1e-9 { 127.0 / amax } else { 0.0 };
        let codes = vals.iter().map(|&v| (v * inv).round() as i8).collect::<Vec<_>>();
        (d, codes)
    };
    let mut c_cpu = vec![0.0f32; m * n];
    for col in 0..n {
        let b_deq = grim_quant::dequant_q3k(&b_bytes[col * 110..(col + 1) * 110], k).expect("dequant_q3k");
        let mut acc = 0.0f32;
        for blk in 0..k / 32 {
            let vals = &a_f32[blk * 32..(blk + 1) * 32];
            let (d_a, codes) = q81_block(vals);
            for i in 0..32 {
                acc += (codes[i] as f32) * d_a * b_deq[blk * 32 + i];
            }
        }
        c_cpu[col] = acc;
    }

    let diff = max_diff(&c_cpu, &c_dev);
    eprintln!("[dot4-q3k-gemv-parity] n={n} k={k} max_diff={diff:.6}");
    assert!(diff < 0.5, "dot4 Q3_K GEMV diverges from CPU reference: {diff}");
    unsafe { std::env::set_var("GRIM_DOT_GEMV", "0"); }
}
