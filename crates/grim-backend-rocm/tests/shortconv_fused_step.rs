//! PLAN-kernel-launch-reduction Phase C regression: `grim_short_conv1d_fused_step`
//! (reads b|x|c from the in_proj output, computes bx, convolves with in-place
//! state update, applies the c gate) must match the split sequence it replaces:
//! bx = b*x -> short_conv1d_causal_step -> y = sum * c, including the state ring.
//!
//! Gated: GRIM_GPU_TEST=1 + ROCm device.

use grim_backend_rocm::{RocmDevice, RocmStorage, as_rocm, gpu_test_enabled};
use grim_tensor::{CoreTensorOps, DType, Shape};

fn gpu_device() -> Option<RocmDevice> {
    if !gpu_test_enabled() {
        return None;
    }
    std::panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new(0)")).ok()
}

fn f32_tensor(
    dev: &RocmDevice,
    data: &[f32],
    shape: &Shape,
) -> Box<dyn grim_tensor::BackendStorage> {
    CoreTensorOps::from_cpu(dev, data, shape, DType::F32).unwrap()
}

fn rand_f32(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((seed + i * 7) % 113) as f32 / 113.0 - 0.5)
        .collect()
}

#[test]
#[ignore]
fn fused_shortconv_step_matches_split() {
    let Some(dev) = gpu_device() else {
        eprintln!("skipping: GPU test gate off");
        return;
    };
    let channels = 1024usize;
    let ks = 3usize; // l_cache
    let proj: Vec<f32> = rand_f32(3 * channels, 1);
    let weight: Vec<f32> = rand_f32(channels * ks, 2);
    let state: Vec<f32> = rand_f32(channels * (ks - 1), 3);

    let proj_s = f32_tensor(&dev, &proj, &Shape::new(vec![3 * channels]));
    let w_s = f32_tensor(&dev, &weight, &Shape::new(vec![channels, ks]));

    // --- Split reference: bx = b*x; conv step; y = sum*c ---
    let b = &proj[..channels];
    let c = &proj[channels..2 * channels];
    let x = &proj[2 * channels..];
    let bx: Vec<f32> = b.iter().zip(x).map(|(a, b)| a * b).collect();
    let bx_s = f32_tensor(&dev, &bx, &Shape::new(vec![channels]));
    let state_ref = f32_tensor(&dev, &state, &Shape::new(vec![channels * (ks - 1)]));
    let sum_s = f32_tensor(&dev, &vec![0.0f32; channels], &Shape::new(vec![channels]));
    dev.short_conv1d_causal_step_into(
        bx_s.as_ref(),
        w_s.as_ref(),
        None,
        state_ref.as_ref(),
        rocm(&sum_s),
    )
    .unwrap();
    let sum = sum_s.to_cpu_vec_f32().unwrap();
    let y_ref: Vec<f32> = sum.iter().zip(c).map(|(s, c)| s * c).collect();
    let state_ref_v = state_ref.to_cpu_vec_f32().unwrap();

    // --- Fused: one launch, in place on a fresh state copy ---
    let state_f = f32_tensor(&dev, &state, &Shape::new(vec![channels * (ks - 1)]));
    let y_f = f32_tensor(&dev, &vec![0.0f32; channels], &Shape::new(vec![channels]));
    dev.short_conv1d_fused_step_into(
        proj_s.as_ref(),
        w_s.as_ref(),
        state_f.as_ref(),
        rocm(&y_f),
        1, // batch
        channels,
        ks,
    )
    .unwrap();

    let got_y = y_f.to_cpu_vec_f32().unwrap();
    let got_state = state_f.to_cpu_vec_f32().unwrap();
    for i in 0..channels {
        assert!(
            (got_y[i] - y_ref[i]).abs() <= 1e-5,
            "y[{i}]: {} vs {}",
            got_y[i],
            y_ref[i]
        );
    }
    for i in 0..state_ref_v.len() {
        assert!(
            (got_state[i] - state_ref_v[i]).abs() <= 1e-6,
            "state[{i}]: {} vs {}",
            got_state[i],
            state_ref_v[i]
        );
    }
}

fn rocm<'a>(s: &'a Box<dyn grim_tensor::BackendStorage>) -> &'a RocmStorage {
    as_rocm(s.as_ref()).unwrap()
}
