//! P3 wiring-contract tests for the ROCm cross-entropy path.
//!
//! Implement.md §P3: the cost is a full D2H + H2D of the logits plus a CPU
//! softmax over batch×vocab per step; the on-device fused CE kernel already
//! exists in-file and is not wired into the training path. These tests encode
//! the intended semantics *before* any dispatch/integration change:
//!
//! 1. **device fused CE is reachable and computes the training numerics
//!    contract** — for the supported shape, the ROCm device impl drives
//!    `fused_linear_cross_entropy_forward` + `backward` and returns loss +
//!    grad on device within tolerance of a CPU oracle. This is the contract a
//!    future training path relies on to replace the CPU `cross_entropy_gpu`
//!    path. (Device-gated: skips when no ROCm device.)
//!
//! 2. **CPU cross_entropy_gpu fallback contract still holds** — the CPU
//!    `cross_entropy_gpu` path remains callable and matches a CPU oracle
//!    (so the fallback the training path would keep is real). (Host-side;
//!    no device needed for this branch.)

use grim_backend_rocm::RocmDevice;
use grim_tensor::dtype::{ArithType, DType, Storage};
use grim_tensor::{BackendDevice, Shape};

fn cpu_fused_ce_oracle(hidden: &[f32], lm_head: &[f32], targets: &[usize], batch: usize, hidden_dim: usize, vocab: usize) -> (f32, Vec<f32>) {
    let mut loss = 0.0f32;
    let mut grad_hidden = vec![0.0f32; hidden.len()];
    let inv_batch = 1.0 / (batch as f32);
    for b in 0..batch {
        let h = &hidden[b * hidden_dim..(b + 1) * hidden_dim];
        let mut logits = vec![0.0f32; vocab];
        for v in 0..vocab {
            let mut s = 0.0f32;
            for d in 0..hidden_dim {
                s += h[d] * lm_head[v * hidden_dim + d];
            }
            logits[v] = s;
        }
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        let mut exp_logits = vec![0.0f32; vocab];
        for v in 0..vocab {
            let e = (logits[v] - max_logit).exp();
            exp_logits[v] = e;
            sum_exp += e;
        }
        let log_sum_exp = max_logit + sum_exp.ln();
        loss -= (logits[targets[b]] - log_sum_exp) * inv_batch;
        let mut grad_logits = vec![0.0f32; vocab];
        for v in 0..vocab {
            grad_logits[v] = exp_logits[v] / sum_exp;
        }
        grad_logits[targets[b]] -= 1.0;
        for d in 0..hidden_dim {
            for v in 0..vocab {
                grad_hidden[b * hidden_dim + d] += grad_logits[v] * lm_head[v * hidden_dim + d] * inv_batch;
            }
        }
    }
    (loss, grad_hidden)
}

#[test]
fn p3_device_fused_ce_is_reachable_and_matches_cpu_oracle_for_supported_shape() {
    if !grim_backend_rocm::gpu_test_enabled() {
        return;
    }
    let Ok(dev) = RocmDevice::try_new(0) else {
        return;
    };
    let hidden_dim = 4;
    let vocab = 3;
    let batch = 2;
    let hidden = vec![0.2f32, -0.4, 0.7, 0.1, -0.3, 0.5, 0.9, -0.2];
    let lm_head = vec![
        0.1f32, 0.3, -0.2, 0.4, -0.5, 0.2, 0.6, -0.1, 0.7, -0.2, 0.1, 0.3,
    ];
    let targets = [1usize, 2usize];

    let hs = dev
        .from_cpu(&hidden, &Shape::new(vec![batch, hidden_dim]), DType::F32)
        .unwrap();
    let ws = dev
        .from_cpu(&lm_head, &Shape::new(vec![vocab, hidden_dim]), DType::F32)
        .unwrap();
    let target_bytes: Vec<u8> = targets.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let ts = dev
        .from_cpu_bytes(
            &target_bytes,
            &Shape::new(vec![batch]),
            DType {
                arith: ArithType::U32,
                storage: Storage::Native,
            },
        )
        .unwrap();

    let (loss, lse, fh) = dev
        .fused_linear_cross_entropy_forward(hs.as_ref(), ws.as_ref(), ts.as_ref(), 2)
        .unwrap();
    fh.synchronize().unwrap();
    let (grad, gh) = dev
        .fused_linear_cross_entropy_backward(hs.as_ref(), ws.as_ref(), ts.as_ref(), lse.as_ref(), 2, 0.0)
        .unwrap();
    gh.synchronize().unwrap();

    let got_loss = loss.to_cpu_vec_f32().unwrap();
    let got_grad = grad.to_cpu_vec_f32().unwrap();

    let (expected_loss, expected_grad) = cpu_fused_ce_oracle(&hidden, &lm_head, &targets, batch, hidden_dim, vocab);
    assert!(
        (got_loss[0] - expected_loss).abs() < 1e-4,
        "P3 device fused CE loss mismatch: got {got_loss_0}, want {expected_loss}",
        got_loss_0 = got_loss[0],
        expected_loss = expected_loss,
    );
    assert_eq!(got_grad.len(), expected_grad.len());
    for i in 0..got_grad.len() {
        assert!(
            (got_grad[i] - expected_grad[i]).abs() < 1e-4,
            "P3 device fused CE grad mismatch at [{i}]: got {got_grad_i}, want {expected_grad_i}",
            got_grad_i = got_grad[i],
            expected_grad_i = expected_grad[i],
        );
    }
}

#[test]
fn p3_cpu_cross_entropy_gpu_fallback_is_callable_and_matches_cpu_oracle_for_unsupported_shape() {
    let dev = RocmDevice::try_new(0).expect("RocmDevice::try_new should succeed on ROCm");

    let batch = 2;
    let vocab = 3;
    let logits = vec![0.2f32, -0.4, 0.7, -0.3, 0.5, 0.9];
    let targets = [1usize, 2usize];

    let logits_storage = dev
        .from_cpu(&logits, &Shape::new(vec![batch, vocab]), DType::F32)
        .unwrap();
    let (loss, grad_storage) = dev.cross_entropy_gpu(logits_storage.as_ref(), &targets, None).unwrap();
    let got_loss = loss;
    let grad = grad_storage.to_cpu_vec_f32().unwrap();

    let mut expected_loss = 0.0f32;
    let mut expected_grad = vec![0.0f32; logits.len()];
    let inv_batch = 1.0 / (batch as f32);
    for b in 0..batch {
        let row_start = b * vocab;
        let row = &logits[row_start..row_start + vocab];
        let max_logit = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        let mut exp_logits = vec![0.0f32; vocab];
        for v in 0..vocab {
            let e = (row[v] - max_logit).exp();
            exp_logits[v] = e;
            sum_exp += e;
        }
        let log_sum_exp = max_logit + sum_exp.ln();
        expected_loss -= (row[targets[b]] - log_sum_exp) * inv_batch;
        let mut grad_logits = vec![0.0f32; vocab];
        for v in 0..vocab {
            grad_logits[v] = exp_logits[v] / sum_exp;
        }
        grad_logits[targets[b]] -= 1.0;
        for v in 0..vocab {
            expected_grad[b * vocab + v] = grad_logits[v] * inv_batch;
        }
    }

    assert!(
        (got_loss - expected_loss).abs() < 1e-4,
        "P3 CPU cross_entropy_gpu fallback loss mismatch: got {got_loss}, want {expected_loss}",
    );
    assert_eq!(grad.len(), expected_grad.len());
    for i in 0..grad.len() {
        assert!(
            (grad[i] - expected_grad[i]).abs() < 1e-4,
            "P3 CPU cross_entropy_gpu fallback grad mismatch at index {i}: got {grad_i}, want {expected_grad_i}",
            grad_i = grad[i],
            expected_grad_i = expected_grad[i],
        );
    }
}
