//! GPU parity for `kernels/log_softmax_vjp.rs`
//! (`grim_log_softmax_vjp_kernel`) vs a CPU f32 reference.
//!
//! Audit §3.6: the file had only a `contains` assertion and zero `tests/`
//! references. The public launcher is `RocmDevice::launch_log_softmax_vjp`
//! (compute/autograd_ops.rs) — this launches it directly against a host
//! transcription of `dL/dz = dL/dlogp * (delta - P(v))` with
//! `P = softmax(logits)` computed in f32. Tolerance ≤1e-4 rel per the task.
//! Also covers the `ignore_index` and out-of-range-target early-outs
//! (the kernel zeroes those rows).

use grim_backend_rocm::RocmDevice;
use grim_backend_rocm::memory::allocator::RocmCachingAllocator;
use grim_backend_rocm::memory::storage::RocmStorage;
use grim_tensor::{BackendStorage, DType, Shape};
use std::sync::Arc;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    grim_backend_rocm::require_gpu_device()
}

/// CPU reference: row-wise softmax, then `dL/dz = f * (delta - P)`.
/// Rows whose target is `ignore_index` or out of range yield zeros (matches
/// the kernel's early-out).
fn ref_grad(logits: &[f32], targets: &[u32], d_loss: f32, ignore_index: u32, vocab: usize, seq_len: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; seq_len * vocab];
    for row in 0..seq_len {
        let t = targets[row];
        if t == ignore_index || (t as usize) >= vocab {
            continue; // zero row (kernel early-out)
        }
        let row_logits = &logits[row * vocab..(row + 1) * vocab];
        let max = row_logits.iter().copied().fold(f32::MIN, f32::max);
        let exps: Vec<f32> = row_logits.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        for v in 0..vocab {
            let delta = if v == t as usize { 1.0 } else { 0.0 };
            out[row * vocab + v] = d_loss * (delta - exps[v] * inv);
        }
    }
    out
}

fn check_rel(what: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    let mut worst = 0.0f32;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let denom = w.abs().max(1.0);
        let rel = (g - w).abs() / denom;
        if rel > worst {
            worst = rel;
        }
        assert!(
            rel <= tol,
            "{what}[{i}]: got={g} want={w} rel={rel} tol={tol}"
        );
    }
    println!("{what}: worst rel error {worst:.3e} (tol {tol:.0e})");
}

fn run(dev: &RocmDevice, alloc: &Arc<RocmCachingAllocator>, logits: &[f32], targets: &[u32], d_loss: f32, ignore_index: u32, vocab: usize, seq_len: usize) -> Vec<f32> {
    let logits_s = RocmStorage::copy_from_host(
        logits,
        &Shape::new(vec![seq_len, vocab]),
        DType::F32,
        alloc,
        0,
    )
    .expect("logits h2d");
    let ids_bytes: Vec<u8> = targets.iter().flat_map(|t| t.to_le_bytes()).collect();
    let targets_s = RocmStorage::copy_from_host_raw_bytes(
        &ids_bytes,
        &Shape::new(vec![seq_len]),
        DType::U32,
        alloc,
        0,
    )
    .expect("targets h2d");
    let grad_s = RocmStorage::alloc_gpu(
        &Shape::new(vec![seq_len, vocab]),
        DType::F32,
        alloc,
        0,
    )
    .expect("grad alloc");
    dev.launch_log_softmax_vjp(
        &logits_s,
        &targets_s,
        &grad_s,
        d_loss,
        ignore_index,
        vocab,
        seq_len,
    )
    .expect("launch_log_softmax_vjp");
    dev.synchronize();
    grad_s.to_cpu_vec_f32().expect("download grad")
}

#[test]
#[ignore = "requires real ROCm device; run with GRIM_RUN_GPU_TESTS=1 ... -- --ignored"]
fn log_softmax_vjp_matches_cpu_reference() {
    let Some(dev) = gpu_device() else { return };
    let alloc = Arc::new(RocmCachingAllocator::new(0, 1 << 30));
    let (seq_len, vocab) = (4usize, 33usize);
    // Mixed-sign logits with a large-magnitude row (max-reduction stability).
    let logits: Vec<f32> = (0..seq_len * vocab)
        .map(|i| {
            let row = i / vocab;
            ((i % 13) as f32 * 0.37 - 2.22) + (row as f32) * 3.1
        })
        .collect();
    let targets: Vec<u32> = vec![0, 7, 32, 16];
    let (d_loss, ignore_index) = (0.73f32, 999u32);

    let got = run(&dev, &alloc, &logits, &targets, d_loss, ignore_index, vocab, seq_len);
    let want = ref_grad(&logits, &targets, d_loss, ignore_index, vocab, seq_len);
    check_rel("log_softmax_vjp", &got, &want, 1e-4);
}

#[test]
#[ignore = "requires real ROCm device; run with GRIM_RUN_GPU_TESTS=1 ... -- --ignored"]
fn log_softmax_vjp_ignored_rows_are_zero() {
    let Some(dev) = gpu_device() else { return };
    let alloc = Arc::new(RocmCachingAllocator::new(0, 1 << 30));
    let (seq_len, vocab) = (4usize, 17usize);
    let logits: Vec<f32> = (0..seq_len * vocab)
        .map(|i| ((i % 7) as f32 * 0.53) - 1.59)
        .collect();
    // Row 0: ignore_index; row 1: out-of-range target; row 2: valid.
    let targets: Vec<u32> = vec![999, 17, 5, 999];
    let ignore_index = 999u32;

    let got = run(&dev, &alloc, &logits, &targets, 1.0, ignore_index, vocab, seq_len);
    // Ignored / out-of-range rows must be exactly zero.
    for row in [0usize, 1, 3] {
        for v in 0..vocab {
            assert_eq!(
                got[row * vocab + v],
                0.0,
                "ignored row {row}[{v}] must be zero, got {}",
                got[row * vocab + v]
            );
        }
    }
    let want = ref_grad(&logits, &targets, 1.0, ignore_index, vocab, seq_len);
    check_rel("log_softmax_vjp_ignored", &got, &want, 1e-4);
}
