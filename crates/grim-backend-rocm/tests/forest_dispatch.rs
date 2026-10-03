//! ForestRaven dispatch parity: GPU vs a host W8A8 reference that mirrors the
//! kernel's in-register activation quantization bit-for-bit.
//!
//! The kernel quantizes each 32-element activation block with
//! inv = 127/amax (0 for a zero block) and full-fp32 d_a = amax/127 -- no
//! fp16 rounding, because the A scale is ephemeral. The reference below uses
//! the SAME formulas (same ops, same order) so activation codes agree
//! exactly; the residual is pure float-association noise from the lane
//! shuffle reduction vs sequential summation, plus the B side which is
//! bit-identical (stored codes, exact int32 dots).
//!
//! The B weights are genuinely int8: compare against the quantized model,
//! not the f32 source.

use grim_backend_rocm::RocmDevice;
use grim_tensor::{ArithType, DType, MemoryOps, QuantOps, Shape, Storage};
use std::panic;

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new")).ok()
}

/// Host W8A8 reference mirroring `grim_dot4_forest_gemv`:
///
/// - A: per-32-block absmax, inv-multiply codes, full-fp32 d_a (same formulas
///   as the kernel, so codes agree bit-for-bit).
/// - B: stored codes + per-row fp32 scales (exact).
/// - Accumulate per-block float partials sequentially (the kernel
///   lane-distributes then shuffle-sums: same terms, different association).
fn w8a8_reference(a: &[f32], codes: &[u8], scales: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    assert_eq!(scales.len(), n);
    assert_eq!(codes.len(), n * k);
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for blk in 0..k.div_ceil(32) {
                let base = blk * 32;
                let len = (k - base).min(32);
                // Block absmax over the REAL length (tail blocks are short;
                // the kernel never sees tails because k % 32 == 0 is gated).
                let mut amax = 0.0f32;
                for e in 0..len {
                    amax = amax.max(a[row * k + base + e].abs());
                }
                let d_a = amax / 127.0;
                let inv_a = if amax == 0.0 { 0.0 } else { 127.0 / amax };
                let mut iacc = 0i32;
                for e in 0..len {
                    let qa = ((a[row * k + base + e] * inv_a).round().clamp(-128.0, 127.0)) as i8;
                    let qb = codes[col * k + base + e] as i8;
                    iacc += (qa as i32) * (qb as i32);
                }
                acc += (iacc as f32) * d_a * scales[col];
            }
            out[row * n + col] = acc;
        }
    }
    out
}

fn forest_case(m: usize, n: usize, k: usize, seed: u64) -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };
    let arch = dev.gcn_arch().to_string();
    if !(arch.starts_with("gfx11") || arch.starts_with("gfx12")) {
        return Ok(());
    }
    assert_eq!(k % 32, 0, "kernel path needs K % 32 == 0");

    let mut s = seed;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    let a: Vec<f32> = (0..m * k)
        .map(|_| ((next() & 0xffff) as f32 / 32768.0) - 1.0)
        .collect();
    let w: Vec<f32> = (0..n * k)
        .map(|_| ((next() & 0xffff) as f32 / 32768.0) - 1.0)
        .collect();

    let (codes, scales_bytes) =
        grim_quant::quant_forest_per_channel(&w, n, k).map_err(|e| format!("quant: {e}"))?;
    let scales: Vec<f32> = scales_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let want = w8a8_reference(&a, &codes, &scales, m, n, k);

    let a_t = MemoryOps::from_cpu_bytes(
        &dev,
        &a.iter().flat_map(|v| v.to_le_bytes().to_vec()).collect::<Vec<u8>>(),
        &Shape::new(vec![m, k]),
        DType {
            arith: ArithType::F32,
            storage: Storage::Native,
        },
    )
    .map_err(|e| format!("act h2d: {e}"))?;

    let mut blob = Vec::with_capacity(16 + codes.len() + scales_bytes.len());
    blob.extend_from_slice(&(codes.len() as u64).to_le_bytes());
    blob.extend_from_slice(&codes);
    blob.extend_from_slice(&(scales_bytes.len() as u64).to_le_bytes());
    blob.extend_from_slice(&scales_bytes);
    let b_t = MemoryOps::from_cpu_bytes(
        &dev,
        &blob,
        &Shape::new(vec![n, k]),
        DType {
            arith: ArithType::F32,
            storage: Storage::Block(grim_tensor::BlockDtype::Int8PerChannel),
        },
    )
    .map_err(|e| format!("b h2d: {e}"))?;

    let (c_t, _handle) = dev.quantized_matmul(
        &*a_t,
        &*b_t,
        &[],
        grim_tensor::QuantFormat::Int8PerChannel,
        &Shape::new(vec![m, n]),
    )?;
    dev.synchronize();
    let got = read_f32(&c_t);

    assert_eq!(got.len(), m * n);
    // Scale-immune statistics. Per-element relative error is meaningless on
    // near-zero outputs (the worst "3.2e-4" observed during bring-up was
    // abs err 2.5e-6 on an output of 7.8e-3 -- pure float-association noise
    // amplified by a tiny denominator). Cosine similarity and max absolute
    // error catch what this test exists to catch -- a layout bug permutes
    // weights and collapses the cosine to ~0.3 with O(1) absolute errors --
    // without blowing up on cancellation outputs.
    let mut dot = 0.0f64;
    let mut gg = 0.0f64;
    let mut xx = 0.0f64;
    let mut max_abs = 0.0f32;
    for (&g, &x) in got.iter().zip(&want) {
        dot += g as f64 * x as f64;
        gg += (g as f64) * (g as f64);
        xx += (x as f64) * (x as f64);
        max_abs = max_abs.max((g - x).abs());
    }
    let cosine = dot / (gg.sqrt() * xx.sqrt()).max(f64::MIN_POSITIVE);
    eprintln!("FOREST_CASE m={m} n={n} k={k} seed={seed:#x} cosine={cosine:.8} max_abs={max_abs:e}");
    assert!(
        cosine >= 0.999,
        "m={m} n={n} k={k}: cosine {cosine:.6} -- a row-major read of the framed blob permutes weights and collapses this"
    );
    assert!(
        max_abs <= 1e-4,
        "m={m} n={n} k={k}: max abs err {max_abs:e} (noise floor ~3e-6)"
    );
    Ok(())
}

fn read_f32(t: &Box<dyn grim_tensor::BackendStorage>) -> Vec<f32> {
    grim_backend_rocm::as_rocm(t.as_ref())
        .expect("rocm storage")
        .copy_to_host()
        .expect("d2h")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn forest_decode_m1_k128() -> TestResult {
    forest_case(1, 256, 128, 0xF02E57)
}

#[test]
fn forest_single_block_k32_is_exact() -> TestResult {
    // One block, one active lane, shuffle adds 31 zeros: no association
    // noise is possible, so any error here is a code/scale divergence.
    forest_case(1, 64, 32, 0x777)
}

#[test]
fn forest_decode_m1_k1024_odd_n() -> TestResult {
    // N = 260: exercises the active_cols tail (4-col groups with 0 < tail < 4).
    forest_case(1, 260, 1024, 0x12345678)
}

#[test]
fn forest_prefill_m16() -> TestResult {
    forest_case(16, 256, 128, 0xABCDEF)
}

#[test]
fn forest_k_tail_falls_back_correctly() -> TestResult {
    // k % 32 != 0 cannot run the block kernel; the host fallback must still
    // produce the quantized model, not refuse and not miscompute.
    let Some(dev) = gpu_device() else {
        return Ok(());
    };
    let (m, n, k) = (1usize, 64usize, 100usize);
    assert_ne!(k % 32, 0);

    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01) - 0.5).collect();
    let w: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.003) - 0.4).collect();
    let (codes, scales_bytes) =
        grim_quant::quant_forest_per_channel(&w, n, k).map_err(|e| format!("quant: {e}"))?;

    let a_t = MemoryOps::from_cpu_bytes(
        &dev,
        &a.iter().flat_map(|v| v.to_le_bytes().to_vec()).collect::<Vec<u8>>(),
        &Shape::new(vec![m, k]),
        DType {
            arith: ArithType::F32,
            storage: Storage::Native,
        },
    )
    .map_err(|e| format!("act h2d: {e}"))?;
    let mut blob = Vec::with_capacity(16 + codes.len() + scales_bytes.len());
    blob.extend_from_slice(&(codes.len() as u64).to_le_bytes());
    blob.extend_from_slice(&codes);
    blob.extend_from_slice(&(scales_bytes.len() as u64).to_le_bytes());
    blob.extend_from_slice(&scales_bytes);
    let b_t = MemoryOps::from_cpu_bytes(
        &dev,
        &blob,
        &Shape::new(vec![n, k]),
        DType {
            arith: ArithType::F32,
            storage: Storage::Block(grim_tensor::BlockDtype::Int8PerChannel),
        },
    )
    .map_err(|e| format!("b h2d: {e}"))?;

    // The reference here is the EXACT-A model, not the W8A8 one: the fallback
    // dequants B on the host and runs F32 GEMM without touching A, so it is
    // more accurate than the quantized reference by the A-quantization gap.
    // Comparing against W8A8 would measure that gap (~2e-4) instead of the
    // fallback's correctness.
    let wq: Vec<f32> = grim_quant::dequant_forest(&blob, n, k).map_err(|e| format!("deq: {e}"))?;
    let want: Vec<f32> = (0..m * n)
        .map(|o| {
            let (row, col) = (o / n, o % n);
            (0..k).map(|j| a[row * k + j] * wq[col * k + j]).sum()
        })
        .collect();

    let (c_t, _handle) = dev.quantized_matmul(
        &*a_t,
        &*b_t,
        &[],
        grim_tensor::QuantFormat::Int8PerChannel,
        &Shape::new(vec![m, n]),
    )?;
    dev.synchronize();
    let got = read_f32(&c_t);
    let mut dot = 0.0f64;
    let mut gg = 0.0f64;
    let mut xx = 0.0f64;
    let mut max_abs = 0.0f32;
    for (&g, &x) in got.iter().zip(&want) {
        dot += g as f64 * x as f64;
        gg += (g as f64) * (g as f64);
        xx += (x as f64) * (x as f64);
        max_abs = max_abs.max((g - x).abs());
    }
    let cosine = dot / (gg.sqrt() * xx.sqrt()).max(f64::MIN_POSITIVE);
    eprintln!("FOREST_FALLBACK cosine={cosine:.8} max_abs={max_abs:e}");
    assert!(cosine >= 0.9999, "fallback cosine {cosine:.6}");
    assert!(max_abs <= 1e-3, "fallback max abs err {max_abs:e}");
    Ok(())
}
