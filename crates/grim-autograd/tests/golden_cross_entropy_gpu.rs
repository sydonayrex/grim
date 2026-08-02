//! Golden parity test: GPU cross-entropy forward + backward vs CPU reference.

use grim_autograd::cross_entropy_loss;
use grim_backend_cpu::cpu_tensor;
use grim_tensor::{BackendDevice, DType, Shape, Tensor};

fn cpu_ce_ref(logits: &[f32], targets: &[usize], vocab: usize) -> (f32, Vec<f32>) {
    let b = targets.len();
    let mut loss = 0.0f32;
    let mut grad = vec![0.0f32; logits.len()];
    for r in 0..b {
        let row = &logits[r * vocab..(r + 1) * vocab];
        let maxv = row.iter().cloned().fold(-1e30f32, f32::max);
        let sum: f32 = row.iter().map(|v| (v - maxv).exp()).sum();
        let lse = maxv + sum.ln();
        loss += lse - row[targets[r]];
        for v in 0..vocab {
            grad[r * vocab + v] = (row[v] - lse).exp() / b as f32;
        }
        grad[r * vocab + targets[r]] -= 1.0 / b as f32;
    }
    (loss / b as f32, grad)
}

#[test]
fn cpu_cross_entropy_matches_reference() {
    let b = 4;
    let vocab = 100;
    let logits_vec: Vec<f32> = (0..b * vocab).map(|i| ((i as f32) * 0.01).sin()).collect();
    let targets: Vec<usize> = (0..b).map(|r| r * 17 % vocab).collect();

    let logits_tensor = cpu_tensor(logits_vec.clone(), Shape::new(vec![b, vocab]));
    let (loss, grad_tensor) = cross_entropy_loss(&logits_tensor, &targets).unwrap();
    let (loss_ref, grad_ref) = cpu_ce_ref(&logits_vec, &targets, vocab);

    assert!(
        (loss - loss_ref).abs() < 1e-4,
        "CPU CE loss {loss} vs ref {loss_ref}"
    );
    let grad_vec = grad_tensor.to_vec_f32().unwrap();
    let max_grad_diff = grad_vec
        .iter()
        .zip(grad_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_grad_diff < 1e-4, "CPU CE grad diff {max_grad_diff}");
}

#[test]
fn gpu_cross_entropy_matches_reference_if_rocm_available() {
    let dev = match grim_backend_rocm::RocmDevice::try_new(0).ok() {
        Some(d) => d,
        None => {
            eprintln!("no ROCm device; skipping GPU test");
            return;
        }
    };
    let b = 8;
    let vocab = 1000;
    let logits_vec: Vec<f32> = (0..b * vocab).map(|i| ((i as f32) * 0.001).sin()).collect();
    let targets: Vec<usize> = (0..b).map(|r| r * 17 % vocab).collect();

    let shape = Shape::new(vec![b, vocab]);
    let storage = dev.from_cpu(&logits_vec, &shape, DType::F32).unwrap();
    let logits_tensor = Tensor::new(
        storage.into(),
        shape,
        DType::F32,
        grim_tensor::dtype::QuantProvenance::default(),
        grim_tensor::Device::Rocm(0),
    );

    let (loss, grad_tensor) = cross_entropy_loss(&logits_tensor, &targets).unwrap();
    assert_eq!(grad_tensor.device(), &grim_tensor::Device::Rocm(0));

    let (loss_ref, grad_ref) = cpu_ce_ref(&logits_vec, &targets, vocab);

    assert!(
        (loss - loss_ref).abs() < 1e-4,
        "GPU CE loss {loss} vs ref {loss_ref}"
    );

    let grad_vec = grad_tensor.to_vec_f32().unwrap();
    let max_grad_diff = grad_vec
        .iter()
        .zip(grad_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_grad_diff < 1e-4, "GPU CE grad diff {max_grad_diff}");
}
