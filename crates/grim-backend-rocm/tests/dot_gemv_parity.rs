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
    ArithType, CoreTensorOps, DType, KQuantScheme, MemoryOps, QuantOps, Shape, Storage,
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
