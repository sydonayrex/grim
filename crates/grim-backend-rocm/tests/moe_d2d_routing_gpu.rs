//! GPU parity test for the Phase-3aD device-side MoE routing (D2D) path.
//!
//! Verifies `grim_moe_route_topk` (device top-k + softmax / sqrt-softplus gating)
//! and `moe_fused_dispatch_resident_routing` (sortless Charon dispatch fed wholly
//! from device-resident routing buffers) against the host reference math, on the
//! RDNA4 discrete parts (gfx1201 / gfx1200).
//!
//! Env-gated by the repo convention: GRIM_RUN_GPU_TESTS=1 (and HIP_VISIBLE_DEVICES
//! to select the target ordinal). No-ops otherwise.

use std::panic;

use grim_backend_rocm::RocmDevice;
use grim_tensor::backend::{BackendStorage, CoreTensorOps};
use grim_tensor::dtype::{DType, Storage};
use grim_tensor::shape::Shape;

const SEQ: usize = 3;
const NUM_EXPERTS: usize = 8;
const TOP_K: usize = 2;
const HIDDEN: usize = 8;
const INTER: usize = 8;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::new should succeed on ROCm"))
        .ok()
}

/// Decode a device u32 buffer's raw bytes into host `Vec<u32>` (little-endian).
fn decode_u32(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Host reference: softmax top-k (matches `shared_moe::route_topk`).
fn host_softmax_topk(logits: &[f32]) -> Vec<(usize, usize, f32)> {
    let mut out = Vec::new();
    for s in 0..SEQ {
        let row = &logits[s * NUM_EXPERTS..(s + 1) * NUM_EXPERTS];
        let mut idx: Vec<(usize, f32)> = row.iter().cloned().enumerate().collect();
        idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let max_l = idx.iter().map(|(_, l)| *l).fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = idx
            .iter()
            .map(|(_, l)| (l - max_l).exp())
            .collect();
        let sum: f32 = exps.iter().sum();
        for k in 0..TOP_K.min(NUM_EXPERTS) {
            let (e, _) = idx[k];
            let w = exps[idx.iter().position(|&(i, _)| i == e).unwrap()] / (sum + 1e-12);
            out.push((s, e, w));
        }
    }
    out
}

/// Host reference: sqrt-softplus top-k (DeepSeek-V4).
fn host_sqrtsoftplus_topk(logits: &[f32]) -> Vec<(usize, usize, f32)> {
    let mut out = Vec::new();
    for s in 0..SEQ {
        let row = &logits[s * NUM_EXPERTS..(s + 1) * NUM_EXPERTS];
        let scores: Vec<f32> = row
            .iter()
            .map(|&l| {
                let sp = if l > 20.0 { l } else { (1.0 + l.exp()).ln() };
                sp.sqrt()
            })
            .collect();
        let sum: f32 = scores.iter().sum();
        let mut idx: Vec<(usize, f32)> = scores.iter().cloned().enumerate().collect();
        idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for k in 0..TOP_K.min(NUM_EXPERTS) {
            let (e, sc) = idx[k];
            out.push((s, e, sc / (sum + 1e-12)));
        }
    }
    out
}

fn compare_routing(got: &[f32], experts: &[u32], want: &[(usize, usize, f32)]) {
    for (i, &(s, e, w)) in want.iter().enumerate() {
        let g_w = got[i];
        let g_e = experts[i];
        assert_eq!(g_e as usize, e, "expert mismatch at slot {i} (token {s})");
        assert!(
            (g_w - w).abs() < 1e-4,
            "weight mismatch at slot {i}: got {g_w}, want {w}"
        );
    }
}

#[test]
fn route_topk_softmax_matches_host() {
    let Some(dev) = gpu_device() else { return };
    // Deterministic logits: [SEQ, NUM_EXPERTS].
    let mut logits = vec![0.0f32; SEQ * NUM_EXPERTS];
    for s in 0..SEQ {
        for e in 0..NUM_EXPERTS {
            logits[s * NUM_EXPERTS + e] = ((s as f32 + 1.0) * 0.7 + (e as f32 + 1.0) * 1.3).sin() * 3.0
                + (e as f32) * 0.2;
        }
    }

    let num_pairs = SEQ * TOP_K;
    let l_st = dev.from_cpu(&logits, &Shape::new(vec![SEQ, NUM_EXPERTS]), DType::F32).unwrap();
    let l_rocm = l_st
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .unwrap();
    let tok_st = dev.zeros(
        &Shape::new(vec![num_pairs]),
        DType { arith: grim_tensor::ArithType::U32, storage: Storage::Native },
    ).unwrap();
    let exp_st = dev.zeros(
        &Shape::new(vec![num_pairs]),
        DType { arith: grim_tensor::ArithType::U32, storage: Storage::Native },
    ).unwrap();
    let w_st = dev.zeros(&Shape::new(vec![num_pairs]), DType::F32).unwrap();
    let tok_rocm = tok_st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
    let exp_rocm = exp_st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
    let w_rocm = w_st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();

    dev.moe_route_topk_on_device(l_rocm, None, tok_rocm, exp_rocm, w_rocm, SEQ, NUM_EXPERTS, TOP_K, 0)
        .unwrap();
    dev.synchronize();

    let experts = decode_u32(&exp_rocm.copy_to_host().unwrap());
    let weights = w_rocm.to_cpu_vec_f32().unwrap();
    let want = host_softmax_topk(&logits);
    compare_routing(&weights, &experts, &want);
}

#[test]
fn route_topk_sqrtsoftplus_matches_host() {
    let Some(dev) = gpu_device() else { return };
    let mut logits = vec![0.0f32; SEQ * NUM_EXPERTS];
    for s in 0..SEQ {
        for e in 0..NUM_EXPERTS {
            logits[s * NUM_EXPERTS + e] = ((s as f32 + 1.0) * 0.5 + (e as f32) * 0.9).sin() * 2.0;
        }
    }

    let num_pairs = SEQ * TOP_K;
    let l_st = dev.from_cpu(&logits, &Shape::new(vec![SEQ, NUM_EXPERTS]), DType::F32).unwrap();
    let l_rocm = l_st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
    let tok_st = dev.zeros(
        &Shape::new(vec![num_pairs]),
        DType { arith: grim_tensor::ArithType::U32, storage: Storage::Native },
    ).unwrap();
    let exp_st = dev.zeros(
        &Shape::new(vec![num_pairs]),
        DType { arith: grim_tensor::ArithType::U32, storage: Storage::Native },
    ).unwrap();
    let w_st = dev.zeros(&Shape::new(vec![num_pairs]), DType::F32).unwrap();
    let tok_rocm = tok_st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
    let exp_rocm = exp_st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
    let w_rocm = w_st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();

    dev.moe_route_topk_on_device(l_rocm, None, tok_rocm, exp_rocm, w_rocm, SEQ, NUM_EXPERTS, TOP_K, 1)
        .unwrap();
    dev.synchronize();

    let experts = decode_u32(&exp_rocm.copy_to_host().unwrap());
    let weights = w_rocm.to_cpu_vec_f32().unwrap();
    let want = host_sqrtsoftplus_topk(&logits);
    compare_routing(&weights, &experts, &want);
}

#[test]
fn device_route_topk_and_dispatch_matches_cpu_oracle() {
    let Some(dev) = gpu_device() else { return };

    // Deterministic logits + expert weights (f32) for a full dispatch.
    let mut logits = vec![0.0f32; SEQ * NUM_EXPERTS];
    for s in 0..SEQ {
        for e in 0..NUM_EXPERTS {
            logits[s * NUM_EXPERTS + e] = ((s as f32 + 1.0) * 0.6 + (e as f32) * 1.1).sin() * 2.5;
        }
    }

    // Expert weights: [NUM_EXPERTS, INTER*HIDDEN] gate/up, [NUM_EXPERTS, HIDDEN*INTER] down.
    let mut gate_flat = vec![0.0f32; NUM_EXPERTS * INTER * HIDDEN];
    let mut up_flat = vec![0.0f32; NUM_EXPERTS * INTER * HIDDEN];
    let mut down_flat = vec![0.0f32; NUM_EXPERTS * HIDDEN * INTER];
    for e in 0..NUM_EXPERTS {
        for j in 0..INTER {
            for i in 0..HIDDEN {
                let v = ((e + 1) as f32 * 0.3 + (j as f32 + 1.0) * 0.1 + (i as f32 + 1.0) * 0.05).sin();
                gate_flat[e * INTER * HIDDEN + j * HIDDEN + i] = v;
                up_flat[e * INTER * HIDDEN + j * HIDDEN + i] = v * 0.7;
            }
        }
        for h in 0..HIDDEN {
            for j in 0..INTER {
                down_flat[e * HIDDEN * INTER + h * INTER + j] =
                    1.0 / (1.0 + h as f32 + j as f32 + e as f32);
            }
        }
    }

    // Activations [SEQ, HIDDEN].
    let mut act = vec![0.0f32; SEQ * HIDDEN];
    for s in 0..SEQ {
        for h in 0..HIDDEN {
            act[s * HIDDEN + h] = ((s as f32 + 1.0) * 0.4 + (h as f32 + 1.0) * 0.2).cos();
        }
    }

    let act_st = dev.from_cpu(&act, &Shape::new(vec![SEQ, HIDDEN]), DType::F32).unwrap();
    let act_rocm = act_st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
    let l_st = dev.from_cpu(&logits, &Shape::new(vec![SEQ, NUM_EXPERTS]), DType::F32).unwrap();
    let l_rocm = l_st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
    let gate_buf = dev.from_cpu(&gate_flat, &Shape::new(vec![gate_flat.len()]), DType::F32).unwrap();
    let up_buf = dev.from_cpu(&up_flat, &Shape::new(vec![up_flat.len()]), DType::F32).unwrap();
    let down_buf = dev.from_cpu(&down_flat, &Shape::new(vec![down_flat.len()]), DType::F32).unwrap();

    let num_pairs = SEQ * TOP_K;
    let tok_st = dev.zeros(
        &Shape::new(vec![num_pairs]),
        DType { arith: grim_tensor::ArithType::U32, storage: Storage::Native },
    ).unwrap();
    let exp_st = dev.zeros(
        &Shape::new(vec![num_pairs]),
        DType { arith: grim_tensor::ArithType::U32, storage: Storage::Native },
    ).unwrap();
    let w_st = dev.zeros(&Shape::new(vec![num_pairs]), DType::F32).unwrap();
    let tok_rocm = tok_st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
    let exp_rocm = exp_st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
    let w_rocm = w_st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();

    dev.moe_route_topk_on_device(l_rocm, None, tok_rocm, exp_rocm, w_rocm, SEQ, NUM_EXPERTS, TOP_K, 0)
        .unwrap();

    let out_shape = Shape::new(vec![SEQ, HIDDEN]);
    let (out_storage, _h) = dev
        .moe_fused_dispatch_resident_routing(
            act_rocm,
            &*gate_buf,
            &*up_buf,
            &*down_buf,
            tok_rocm,
            exp_rocm,
            w_rocm,
            num_pairs,
            &out_shape,
            HIDDEN,
            INTER,
            1.0,
        )
        .unwrap();
    dev.synchronize();

    let got = out_storage.to_cpu_vec_f32().unwrap();

    // CPU oracle: replicate the sortless dispatch math.
    let experts = decode_u32(&exp_rocm.copy_to_host().unwrap());
    let weights = w_rocm.to_cpu_vec_f32().unwrap();
    let mut oracle = vec![0.0f32; SEQ * HIDDEN];
    for p in 0..num_pairs {
        let tok = decode_u32(&tok_rocm.copy_to_host().unwrap())[p] as usize;
        let e = experts[p] as usize;
        let w = weights[p];
        // gate + up + SiLU, then down.
        for h in 0..HIDDEN {
            let mut acc = 0.0f32;
            for j in 0..INTER {
                let mut g = 0.0f32;
                let mut u = 0.0f32;
                for i in 0..HIDDEN {
                    g += gate_flat[e * INTER * HIDDEN + j * HIDDEN + i] * act[tok * HIDDEN + i];
                    u += up_flat[e * INTER * HIDDEN + j * HIDDEN + i] * act[tok * HIDDEN + i];
                }
                let silu = g / (1.0 + (-g).exp());
                let a = silu * u;
                acc += down_flat[e * HIDDEN * INTER + h * INTER + j] * a;
            }
            oracle[tok * HIDDEN + h] += w * acc;
        }
    }

    for (i, (g, o)) in got.iter().zip(oracle.iter()).enumerate() {
        assert!(
            (g - o).abs() < 1e-3,
            "dispatch mismatch at {i}: device {g}, cpu {o}"
        );
    }
}