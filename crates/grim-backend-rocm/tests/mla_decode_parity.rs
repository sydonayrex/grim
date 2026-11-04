//! GPU parity for `kernels/mla_decode.rs` — `grim_mla_absorbed_decode`
//! (`launch_mla_absorbed_decode` / `_scaled`) vs a CPU oracle.
//!
//! Audit §3.6: the file was covered only via `production_kernels_tests`
//! (`test_mla_absorbed_decode_parity` at 1e-3) and `mla_prefill_parity`
//! (the *prefill* sibling, a different kernel). This adds the dedicated
//! decode case at the tighter ≤1e-4 rel bar, reusing the production shape
//! (2 heads, rank 32, rope 16, seq 64) but with a mixed-sign deterministic
//! fixture so a sign-dropping accumulation cannot pass vacuously, and a
//! non-trivial `inv_sqrt_d` (the scaled launcher's whole point).

use grim_backend_rocm::RocmDevice;
use grim_tensor::{CoreTensorOps, Shape, dtype::DType};

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    grim_backend_rocm::require_gpu_device()
}

/// Deterministic mixed-sign fill in [-1, 1).
fn fill(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 9) as f32 / 8_388_608.0) - 0.5
        })
        .collect()
}

/// Direct transcription of the kernel's no-W_UV path: latent-space softmax
/// attention over the packed `[seq, rank + rope_d]` cache.
fn ref_decode(
    q_nope: &[f32],
    q_pe: &[f32],
    packed_kv: &[f32],
    num_heads: usize,
    rank: usize,
    rope_d: usize,
    seq_len: usize,
    inv_sqrt_d: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; num_heads * rank];
    for h in 0..num_heads {
        let mut running_max = -1e20f32;
        let mut running_sum = 0.0f32;
        let mut acc = vec![0.0f32; rank];
        for j in 0..seq_len {
            let base = j * (rank + rope_d);
            let mut dot = 0.0f32;
            for d in 0..rank {
                dot += q_nope[h * rank + d] * packed_kv[base + d];
            }
            for d in 0..rope_d {
                dot += q_pe[h * rope_d + d] * packed_kv[base + rank + d];
            }
            let score = dot * inv_sqrt_d;
            let new_max = running_max.max(score);
            let alpha = (running_max - new_max).exp();
            let beta = (score - new_max).exp();
            running_sum = running_sum * alpha + beta;
            running_max = new_max;
            for d in 0..rank {
                acc[d] = acc[d] * alpha + beta * packed_kv[base + d];
            }
        }
        let inv = if running_sum > 0.0 { 1.0 / running_sum } else { 0.0 };
        for d in 0..rank {
            out[h * rank + d] = acc[d] * inv;
        }
    }
    out
}

#[test]
#[ignore = "requires real ROCm device; run with GRIM_RUN_GPU_TESTS=1 ... -- --ignored"]
fn mla_absorbed_decode_matches_cpu_oracle() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };
    // Production shape (production_kernels_tests): 2 heads, rank 32, rope 16.
    let (num_heads, rank, rope_d, seq_len) = (2usize, 32usize, 16usize, 64usize);
    // Non-default denominator (exercises `_scaled`): 1/sqrt(rank) not 1/sqrt(rank+rope).
    let inv_sqrt_d = 1.0f32 / (rank as f32).sqrt();

    let q_nope = fill(num_heads * rank, 11);
    let q_pe = fill(num_heads * rope_d, 23);
    let packed_kv = fill(seq_len * (rank + rope_d), 37);

    let q_nope_t = CoreTensorOps::from_cpu(&dev, &q_nope, &Shape::new(vec![num_heads, rank]), DType::F32)?;
    let q_pe_t = CoreTensorOps::from_cpu(&dev, &q_pe, &Shape::new(vec![num_heads, rope_d]), DType::F32)?;
    let kv_t = CoreTensorOps::from_cpu(&dev, &packed_kv, &Shape::new(vec![seq_len, rank + rope_d]), DType::F32)?;
    let out_t = CoreTensorOps::zeros(&dev, &Shape::new(vec![num_heads, rank]), DType::F32)?;

    let q_s = grim_backend_rocm::device::util::as_rocm(q_nope_t.as_ref())?;
    let r_s = grim_backend_rocm::device::util::as_rocm(q_pe_t.as_ref())?;
    let kv_s = grim_backend_rocm::device::util::as_rocm(kv_t.as_ref())?;
    let o_s = grim_backend_rocm::device::util::as_rocm(out_t.as_ref())?;

    dev.launch_mla_absorbed_decode_scaled(
        q_s, r_s, kv_s, None, o_s,
        num_heads, rank, rope_d, rank /* v_head_dim unused without W_UV */, seq_len,
        0, 0, None, inv_sqrt_d,
    )?;
    dev.synchronize();

    let got = out_t.to_cpu_vec_f32()?;
    let want = ref_decode(&q_nope, &q_pe, &packed_kv, num_heads, rank, rope_d, seq_len, inv_sqrt_d);
    assert_eq!(got.len(), want.len(), "mla decode: length mismatch");
    let mut worst = 0.0f32;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let rel = (g - w).abs() / w.abs().max(1.0);
        if rel > worst {
            worst = rel;
        }
        assert!(
            rel <= 1e-4,
            "mla_absorbed_decode[{i}]: got={g} want={w} rel={rel} (tol 1e-4)"
        );
    }
    println!("mla_absorbed_decode: worst rel error {worst:.3e} (tol 1e-4)");
    Ok(())
}
