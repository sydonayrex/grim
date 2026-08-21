//! RUN ON THIS SYSTEM: GRIM_GPU_TEST=1 cargo test -p grim-backend-rocm --test fused_linear_ce_parity_tests

use grim_backend_rocm::RocmDevice;
use grim_tensor::dtype::{ArithType, DType, Storage};
use grim_tensor::{BackendDevice, Shape};

// PASSED: 2026-08-20 on gfx1036 (ROCm)
#[test]
fn fused_linear_ce_matches_cpu_oracle() {
    if !grim_backend_rocm::gpu_test_enabled() {
        return;
    }
    let Ok(dev) = RocmDevice::try_new(0) else {
        return;
    };
    let hidden = vec![0.2f32, -0.4, 0.7, 0.1, -0.3, 0.5, 0.9, -0.2];
    let lm_head = vec![
        0.1f32, 0.3, -0.2, 0.4, -0.5, 0.2, 0.6, -0.1, 0.7, -0.2, 0.1, 0.3,
    ];
    let targets = [1u32, 2u32];
    let hs = dev
        .from_cpu(&hidden, &Shape::new(vec![2, 4]), DType::F32)
        .unwrap();
    let ws = dev
        .from_cpu(&lm_head, &Shape::new(vec![3, 4]), DType::F32)
        .unwrap();
    let target_bytes: Vec<u8> = targets.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let ts = dev
        .from_cpu_bytes(
            &target_bytes,
            &Shape::new(vec![2]),
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
        .fused_linear_cross_entropy_backward(
            hs.as_ref(),
            ws.as_ref(),
            ts.as_ref(),
            lse.as_ref(),
            2,
            0.5,
        )
        .unwrap();
    gh.synchronize().unwrap();
    let got_loss = loss.to_cpu_vec_f32().unwrap();
    let got_grad = grad.to_cpu_vec_f32().unwrap();

    let mut expected_loss = 0.0f32;
    let mut expected_grad = vec![0.0f32; hidden.len()];
    for b in 0..2 {
        let h = &hidden[b * 4..(b + 1) * 4];
        let logits: Vec<f32> = (0..3)
            .map(|v| (0..4).map(|d| h[d] * lm_head[v * 4 + d]).sum())
            .collect();
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = logits.iter().map(|x| (x - max).exp()).sum();
        expected_loss += max + sum.ln() - logits[targets[b] as usize];
        for v in 0..3 {
            let dl = ((logits[v] - max).exp() / sum
                - if v == targets[b] as usize { 1.0 } else { 0.0 })
                * 0.5;
            for d in 0..4 {
                expected_grad[b * 4 + d] += dl * lm_head[v * 4 + d];
            }
        }
    }
    assert!((got_loss.iter().sum::<f32>() - expected_loss).abs() < 5e-4, "got: {}, expected: {}", got_loss.iter().sum::<f32>(), expected_loss);
    for (got, expected) in got_grad.iter().zip(expected_grad) {
        assert!((got - expected).abs() < 5e-4, "{got} != {expected}");
    }
}
