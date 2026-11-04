//! Task 5B.3 gate: `grim-graph`'s YaRN rope args, once they reach the
//! device, must rotate by the CORRECT ANGLES. This is the Xing4.0 defect
//! class (5db54031): a signature-identical rope that drops the YaRN
//! ramp/mscale preserves every pair's norm and silently permutes values.
//! A norm-preservation assertion cannot see it; only parity of the actual
//! output at real positions can. Positions 0 and 41 are the Xing4.0 dump
//! positions (0 is the all-identity upper half, 41 is a non-trivial angle).
//!
//! Parity target: the eager CPU rope math (`Rope::forward` convention) fed
//! the SAME `yarn_inv_freq_and_mscale` table the kernel consumes — so the
//! shared ramp definition is exercised end to end through
//! `rope_dev_yarn_pos0_into`, the graph capture path.
//!
//! RUN: GRIM_RUN_GPU_TEST=1 cargo test -p grim-backend-rocm \
//!        --test grim_graph_rope_yarn_parity -- --ignored --nocapture

use grim_backend_rocm::{as_rocm, gpu_test_enabled, RocmDevice};
use grim_tensor::{BackendStorage, CoreTensorOps, DType, MemoryOps, Shape, YaRNParams};

fn gpu_device() -> Option<RocmDevice> {
    if !gpu_test_enabled() {
        return None;
    }
    grim_backend_rocm::require_gpu_device()
}

/// Xing4.0-shaped config: NeoX half-split pairing, rotary_dim == dim,
/// YaRN with the reciprocal attn/mscale pair the metadata produces.
fn yarn_cfg() -> grim_tensor::RopeConfig {
    let mut cfg = grim_tensor::RopeConfig::new(64, 10_000.0);
    cfg.interleaved = false; // NeoX half-split (x[i], x[half+i])
    cfg.yarn = Some(YaRNParams {
        factor: 32.0,
        original_max_pos: 4096,
        beta_fast: 32.0,
        beta_slow: 1.0,
        attention_factor: 1.41589,
        rope_mscale: Some(0.70626),
    });
    cfg
}

/// CPU reference with the SAME ramp table the kernel consumes.
fn cpu_yarn_rope(x: &[f32], pos: u32, inv_freq: &[f32], mscale: f32) -> Vec<f32> {
    let half = inv_freq.len();
    let d = half * 2;
    let mut out = vec![0f32; d];
    for i in 0..half {
        let val = pos as f32 * inv_freq[i];
        let (sin_v, cos_v) = (val.sin() * mscale, val.cos() * mscale);
        let (x1, x2) = (x[i], x[half + i]);
        out[i] = x1 * cos_v - x2 * sin_v;
        out[half + i] = x2 * cos_v + x1 * sin_v;
    }
    out
}

fn run_pos(dev: &RocmDevice, pos: u32) -> (Vec<f32>, Vec<f32>) {
    let cfg = yarn_cfg();
    let (inv_freq, mscale) = RocmDevice::yarn_inv_freq_and_mscale(&cfg);
    let d = cfg.dim;
    let half = d / 2;
    assert_eq!(inv_freq.len(), half);

    let x: Vec<f32> = (0..d)
        .map(|i| (((i * 37 + 13) % 71) as f32 - 35.0) / 35.0)
        .collect();

    let x_dev = CoreTensorOps::from_cpu(dev, &x, &Shape::new(vec![1, 1, d]), DType::F32)
        .expect("upload x");
    let pos_bytes = pos.to_le_bytes();
    let pos_dev = MemoryOps::from_cpu_bytes(
        dev,
        &pos_bytes,
        &Shape::new(vec![4]),
        DType::U8,
    )
    .expect("upload pos");
    let freq_bytes: Vec<u8> = inv_freq.iter().flat_map(|f| f.to_le_bytes()).collect();
    let freq_dev = MemoryOps::from_cpu_bytes(
        dev,
        &freq_bytes,
        &Shape::new(vec![freq_bytes.len()]),
        DType::U8,
    )
    .expect("upload inv_freq");
    let out_dev = CoreTensorOps::from_cpu(
        dev,
        &vec![0f32; d],
        &Shape::new(vec![1, 1, d]),
        DType::F32,
    )
    .expect("alloc out");
    let out_r = as_rocm(out_dev.as_ref()).expect("out RocmStorage");

    dev.rope_dev_yarn_pos0_into(
        x_dev.as_ref(),
        pos_dev.as_ref(),
        as_rocm(freq_dev.as_ref()).expect("freq RocmStorage"),
        &out_r,
        &cfg,
        &Shape::new(vec![1, 1, d]),
    )
    .expect("rope_dev_yarn_pos0_into");
    dev.synchronize();

    let gpu = out_r.to_cpu_vec_f32().expect("d2h out");
    let reference = cpu_yarn_rope(&x, pos, &inv_freq, mscale);
    (gpu, reference)
}

#[test]
#[ignore = "needs a ROCm device; run with GRIM_RUN_GPU_TEST=1"]
fn yarn_rope_matches_cpu_reference_at_pos0_and_pos41() {
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
        return;
    };

    for pos in [0u32, 41u32] {
        let (gpu, reference) = run_pos(&dev, pos);
        let mscale = RocmDevice::yarn_inv_freq_and_mscale(&yarn_cfg()).1;
        if pos == 0 {
            // pos 0 rotates by angle 0: out must be x * mscale. The CPU
            // reference IS x*mscale here by construction, so hold the GPU
            // to a tighter bar than the pos-41 envelope.
            for (i, (&g, &r)) in gpu.iter().zip(&reference).enumerate() {
                assert!(
                    (g - r).abs() < 1e-6,
                    "pos0 [{i}]: gpu={g} expected x*mscale={r} — upper half not identity-scaled"
                );
            }
            let _ = mscale;
        }
        let mut worst = 0f32;
        for (i, (&g, &r)) in gpu.iter().zip(&reference).enumerate() {
            let diff = (g - r).abs();
            worst = worst.max(diff);
            assert!(
                diff < 1e-5,
                "pos{pos} [{i}]: gpu={g} cpu={r} (diff {diff}) — WRONG-ANGLE defect class"
            );
        }
        eprintln!("[yarn-rope-parity] pos={pos}: worst |gpu-cpu| = {worst:.3e}");
    }
}
