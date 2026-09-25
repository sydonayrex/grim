//! PLAN-decode-throughput-restore regression: `grim_qk_rope_dev_base`
//! (fused QK-norm + device-base RoPE, one launch) must match the split path
//! (`rms_norm_into` then `rope_dev_base_into`) on the same inputs — the
//! decode-graph capture swaps the pair for the fused kernel.

use grim_backend_rocm::{RocmDevice, as_rocm, gpu_test_enabled};
use grim_tensor::{CoreTensorOps, DType, Shape};

fn gpu_device() -> Option<RocmDevice> {
    if !gpu_test_enabled() {
        return None;
    }
    std::panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new(0)")).ok()
}

#[test]
#[ignore]
fn fused_qk_rope_matches_split_path() {
    let Some(dev) = gpu_device() else {
        eprintln!("skipping: GPU test gate off");
        return;
    };
    let batch = 1usize;
    let steps = 1usize;
    for (heads, hd) in [(16usize, 64usize), (8, 64)] {
        let n = batch * heads * steps * hd;
        let shape = Shape::new(vec![batch, heads * steps, hd]);
        let x: Vec<f32> = (0..n)
            .map(|i| ((i * 11 % 83) as f32 - 41.0) / 41.0)
            .collect();
        let gamma: Vec<f32> = (0..hd).map(|i| 0.5 + (i % 7) as f32 * 0.2).collect();
        let eps = 1e-5f32;
        let rope_cfg = {
            let mut c = grim_tensor::RopeConfig::new(hd, 1_000_000.0);
            c.interleaved = false; // LFM2: NeoX half-split
            c
        };

        let g_s = dev
            .from_cpu(&gamma, &Shape::new(vec![hd]), DType::F32)
            .unwrap();
        let pos_f32: Vec<f32> = vec![0.0];
        let pos_s = dev
            .from_cpu(&pos_f32, &Shape::new(vec![1]), DType::F32)
            .unwrap();

        // Split reference: rms_norm_into (in place) then rope_dev_base_into.
        let split = dev.from_cpu(&x, &shape, DType::F32).unwrap();
        let split_r = as_rocm(split.as_ref()).unwrap();
        dev.rms_norm_into(split.as_ref(), g_s.as_ref(), eps, split_r, &shape)
            .unwrap();
        dev.rope_dev_base_into(
            split.as_ref(),
            pos_s.as_ref(),
            split_r,
            &rope_cfg,
            &shape,
            heads,
            steps,
        )
        .unwrap();

        // Fused: norm + rope in one launch, in place on a fresh copy.
        let fused = dev.from_cpu(&x, &shape, DType::F32).unwrap();
        let fused_r = as_rocm(fused.as_ref()).unwrap();
        dev.qk_rope_dev_base_into(
            fused.as_ref(),
            pos_s.as_ref(),
            g_s.as_ref(),
            eps,
            fused_r,
            &rope_cfg,
            &shape,
            heads,
            steps,
        )
        .unwrap();

        let want = split.to_cpu_vec_f32().unwrap();
        let got = fused.to_cpu_vec_f32().unwrap();
        for i in 0..n {
            assert!(
                (got[i] - want[i]).abs() <= 1e-5,
                "heads={heads} mismatch at {i}: {} vs {}",
                got[i],
                want[i]
            );
        }
    }
}
