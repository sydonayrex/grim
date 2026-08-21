//! WI-F2 — Fused attention output-projection parity tests.
//!
//! RED-first per the fusion-boundary plan: the reference is *computed* by
//! running the existing unfused sequence (fused QKV attention kernel, then a
//! separate O-projection GEMM) on device — never a hand-copied constant.
//! `fused_attn_o_proj` must fail to resolve until the epilogue fusion exists.

use grim_backend_rocm::RocmDevice;
use grim_tensor::{BackendDevice, DType, Shape};

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    std::panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new(0)")).ok()
}

fn fill(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((seed + i as f32 * 0.61).sin() * 0.5 - 0.1) as f32)
        .collect()
}

#[test]
fn fused_attn_output_proj_matches_unfused() {
    let Some(dev) = gpu_device() else {
        eprintln!("skipping: GPU test gate off");
        return;
    };

    let seq_len = 5;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 16;
    let kv_len = 7;
    let hidden = num_heads * head_dim; // 64

    let q_data = fill(seq_len * num_heads * head_dim, 0.3);
    let k_data = fill(kv_len * num_kv_heads * head_dim, 0.5);
    let v_data = fill(kv_len * num_kv_heads * head_dim, 0.9);
    let o_w = fill(hidden * hidden, 0.2);

    let q = dev
        .from_cpu(
            &q_data,
            &Shape::from_slice(&[seq_len, num_heads, head_dim]),
            DType::F32,
        )
        .unwrap();
    let k = dev
        .from_cpu(
            &k_data,
            &Shape::from_slice(&[kv_len, num_kv_heads, head_dim]),
            DType::F32,
        )
        .unwrap();
    let v = dev
        .from_cpu(
            &v_data,
            &Shape::from_slice(&[kv_len, num_kv_heads, head_dim]),
            DType::F32,
        )
        .unwrap();
    let o = dev
        .from_cpu(&o_w, &Shape::from_slice(&[hidden, hidden]), DType::F32)
        .unwrap();

    // Reference: existing unfused path — attention kernel, then separate GEMM.
    let attn_shape = Shape::from_slice(&[seq_len, num_heads, head_dim]);
    let (attn, h) = dev
        .qkv_attention(
            q.as_ref(),
            k.as_ref(),
            v.as_ref(),
            num_kv_heads,
            kv_len,
            0,
            None,
            &attn_shape,
            None,
            None,
        )
        .unwrap();
    h.synchronize().unwrap();
    let attn_flat = attn.to_cpu_vec_f32().unwrap();
    let attn2d = dev
        .from_cpu(
            &attn_flat,
            &Shape::from_slice(&[seq_len, hidden]),
            DType::F32,
        )
        .unwrap();
    let (proj, h) = BackendDevice::matmul(
        &dev,
        attn2d.as_ref(),
        o.as_ref(),
        &Shape::from_slice(&[seq_len, hidden]),
    )
    .unwrap();
    h.synchronize().unwrap();
    let want = proj.to_cpu_vec_f32().unwrap();

    // Fused path: O-projection applied in the attention kernel epilogue.
    let (fused, h) = dev
        .fused_attn_o_proj(
            q.as_ref(),
            k.as_ref(),
            v.as_ref(),
            o.as_ref(),
            num_kv_heads,
            kv_len,
            0,
            &Shape::from_slice(&[seq_len, hidden]),
        )
        .expect("fused_attn_o_proj should launch");
    h.synchronize().unwrap();
    let got = fused.to_cpu_vec_f32().unwrap();

    assert_eq!(got.len(), want.len(), "fused output length mismatch");
    let tol = 2e-3f32;
    let mut worst = 0.0f32;
    for i in 0..want.len() {
        let denom = want[i].abs().max(got[i].abs()).max(1e-6);
        let err = ((got[i] - want[i]) / denom).abs();
        worst = worst.max(err);
        assert!(
            err <= tol,
            "fused O-proj mismatch at {i} (token {} col {}): fused {} vs unfused {} (rel err {err})",
            i / hidden,
            i % hidden,
            got[i],
            want[i]
        );
    }
    // Guard against vacuous pass: reference must be non-trivial.
    assert!(
        want.iter().any(|x| x.abs() > 1e-4),
        "reference output is all ~zero; test proves nothing"
    );
    eprintln!(
        "fused_attn_o_proj worst rel err: {worst:.3e} over {} elems",
        want.len()
    );
}

/// WI-F2 gate 3 — occupancy regression harness. Asserts the fused-epilogue
/// `grim_qkv_attention` keeps at least `OCCUPANCY_FLOOR` resident blocks per
/// CU at the real launch block size; a register/LDS blow-up from the fused
/// epilogue drives the measured value below the floor and fails here, rather
/// than surfacing later as an unexplained decode-latency regression.
#[test]
fn fused_attention_occupancy_no_regression() {
    let Some(dev) = gpu_device() else {
        eprintln!("skipping: GPU test gate off");
        return;
    };

    // Launch once so the kernel entry resolves into the device's kernel cache.
    let seq_len = 1;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 16;
    let kv_len = 3;
    let hidden = num_heads * head_dim;
    let q = dev
        .from_cpu(
            &fill(seq_len * num_heads * head_dim, 0.3),
            &Shape::from_slice(&[seq_len, num_heads, head_dim]),
            DType::F32,
        )
        .unwrap();
    let k = dev
        .from_cpu(
            &fill(kv_len * num_kv_heads * head_dim, 0.5),
            &Shape::from_slice(&[kv_len, num_kv_heads, head_dim]),
            DType::F32,
        )
        .unwrap();
    let v = dev
        .from_cpu(
            &fill(kv_len * num_kv_heads * head_dim, 0.9),
            &Shape::from_slice(&[kv_len, num_kv_heads, head_dim]),
            DType::F32,
        )
        .unwrap();
    let o = dev
        .from_cpu(
            &fill(hidden * hidden, 0.2),
            &Shape::from_slice(&[hidden, hidden]),
            DType::F32,
        )
        .unwrap();
    let (_, h) = dev
        .fused_attn_o_proj(
            q.as_ref(),
            k.as_ref(),
            v.as_ref(),
            o.as_ref(),
            num_kv_heads,
            kv_len,
            0,
            &Shape::from_slice(&[seq_len, hidden]),
        )
        .unwrap();
    h.synchronize().unwrap();

    let block = dev.wavefront_size() as u32 * 4; // real launch block (fusion.rs hip_launch_params: wf*4)
    let Some(blocks) = dev.kernel_max_blocks_per_cu("grim_qkv_attention", block) else {
        panic!("grim_qkv_attention not resolved in kernel cache; occupancy harness cannot run");
    };
    const OCCUPANCY_FLOOR: i32 = 2;
    eprintln!("grim_qkv_attention max blocks/CU at block={block}: {blocks}");
    assert!(
        blocks >= OCCUPANCY_FLOOR,
        "occupancy regression: fused epilogue dropped max resident blocks/CU to {blocks} (floor {OCCUPANCY_FLOOR})"
    );
}
