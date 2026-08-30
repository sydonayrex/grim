//! Parity test: Vulkan embedding_backward vs CPU reference.
//!
//! Run with `GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-vulkan --test embedding_backward_parity`.

use grim_tensor::backend::AutogradOps;
use grim_tensor::{CoreTensorOps, DType, Shape};
use grim_backend_vulkan::VulkanDevice;

#[test]
fn embedding_backward_matches_cpu_reference() {
    if std::env::var("GRIM_RUN_GPU_TESTS").unwrap_or_default() != "1" {
        eprintln!("Skipping GPU test (set GRIM_RUN_GPU_TESTS=1)");
        return;
    }
    let dev = VulkanDevice::new();
    let num_tokens = 4usize;
    let vocab_size = 8usize;
    let hidden_dim = 4usize;
    let grad = vec![
        0.1f32, 0.2, 0.3, 0.4, // token 0
        0.5, 0.6, 0.7, 0.8, // token 1
        0.9, 1.0, 1.1, 1.2, // token 2
        1.3, 1.4, 1.5, 1.6, // token 3
    ];
    let token_ids = vec![3u32, 5, 3, 0]; // tokens 0,2 both hit vocab 3
    let grad_shape = Shape::new(vec![num_tokens, hidden_dim]);
    let g_s = dev.from_cpu(&grad, &grad_shape, DType::F32).unwrap();

    let (dw, _handle) =
        AutogradOps::embedding_backward(&dev, &*g_s, &token_ids, vocab_size, hidden_dim).unwrap();
    let dw_v = dw.to_cpu_vec_f32().unwrap();

    let mut expected = vec![0.0f32; vocab_size * hidden_dim];
    for (t, &tok) in token_ids.iter().enumerate() {
        let tok = tok as usize;
        for d in 0..hidden_dim {
            expected[tok * hidden_dim + d] += grad[t * hidden_dim + d];
        }
    }
    for i in 0..(vocab_size * hidden_dim) {
        assert!(
            (dw_v[i] - expected[i]).abs() < 2.5e-7,
            "dw[{}]: {} vs {}",
            i,
            dw_v[i],
            expected[i]
        );
    }
}
