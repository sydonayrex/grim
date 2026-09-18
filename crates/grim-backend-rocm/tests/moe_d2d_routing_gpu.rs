//! GPU parity test for the Phase-3aD device-side MoE routing (D2D) path.
//!
//! Verifies `grim_moe_route_topk` (device top-k + softmax / sqrt-softplus gating)
//! and `moe_fused_dispatch_resident_routing` (sortless Charon dispatch fed wholly
//! from device-resident routing buffers) against the host reference math, on the
//! RDNA4 discrete parts (gfx1201 / gfx1200).
//!
//! WI-gpu-native-moe Phase 2 adds `grim_moe_fused_dispatch_w8a8_int8` +
//! `moe_fused_dispatch_resident_routing_w8a8_int8` (native int8 dispatch from
//! packed blobs, same D2D routing) with an exact-roundtrip parity test below.
//!
//! Env-gated by the repo convention: GRIM_RUN_GPU_TESTS=1 (and HIP_VISIBLE_DEVICES
//! to select the target ordinal). No-ops otherwise.

use std::panic;

use grim_backend_rocm::RocmDevice;
use grim_tensor::backend::{BackendStorage, CoreTensorOps, MemoryOps};
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

/// Serializes GPU tests in this binary (one device; concurrent
/// `moe_route_topk_on_device` launches contend and give false failures
/// under default `--test-threads=N`). See `gpu_test_lock` docs.
fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    grim_backend_rocm::device::util::gpu_test_lock()
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

/// Host reference: top-k softmax renormalized over top-k (route_mode == 3, GLM/Qwen style `normalize_weights`).
fn host_renorm_topk(logits: &[f32]) -> Vec<(usize, usize, f32)> {
    let mut out = Vec::new();
    for s in 0..SEQ {
        let row = &logits[s * NUM_EXPERTS..(s + 1) * NUM_EXPERTS];
        let mut idx: Vec<(usize, f32)> = row.iter().cloned().enumerate().collect();
        idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let k = TOP_K.min(NUM_EXPERTS);
        let topk = &idx[..k];
        let max_l = topk.iter().map(|(_, l)| *l).fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = topk.iter().map(|(_, l)| (l - max_l).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for i in 0..k {
            let (e, _) = topk[i];
            let w = exps[i] / (sum + 1e-12);
            out.push((s, e, w));
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
    let _guard = gpu_lock();
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
    let _guard = gpu_lock();
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
fn route_topk_renorm_matches_host() {
    let _guard = gpu_lock();
    let Some(dev) = gpu_device() else { return };
    let mut logits = vec![0.0f32; SEQ * NUM_EXPERTS];
    for s in 0..SEQ {
        for e in 0..NUM_EXPERTS {
            logits[s * NUM_EXPERTS + e] = ((s as f32 + 1.0) * 0.8 + (e as f32) * 1.2).cos() * 3.0;
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

    dev.moe_route_topk_on_device(l_rocm, None, tok_rocm, exp_rocm, w_rocm, SEQ, NUM_EXPERTS, TOP_K, 3)
        .unwrap();
    dev.synchronize();

    let experts = decode_u32(&exp_rocm.copy_to_host().unwrap());
    let weights = w_rocm.to_cpu_vec_f32().unwrap();
    let want = host_renorm_topk(&logits);
    compare_routing(&weights, &experts, &want);
}

#[test]
fn device_route_topk_and_dispatch_matches_cpu_oracle() {
    let _guard = gpu_lock();
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

// ── WI-gpu-native-moe Phase 2: native W8A8-int8 sortless dispatch ──────────

/// Exact-roundtrip int8 row-quantize: `w` rewritten in place to
/// `code * scale` so parity is not limited by quantization error.
/// Returns `(codes, scales)`; packed blob = `[u64 len | codes | scales]`.
fn w8a8_int8_quantize_rowmajor(w: &mut [f32], rows: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    let mut codes = vec![0u8; rows * k];
    let mut scales = vec![0.0f32; rows];
    for r in 0..rows {
        let max_abs = (0..k).map(|c| w[r * k + c].abs()).fold(0.0f32, f32::max);
        let scale = if max_abs == 0.0 { 1e-12 } else { max_abs / 127.0 };
        scales[r] = scale;
        for c in 0..k {
            let q = ((w[r * k + c] / scale).round() as i32).clamp(-128, 127);
            codes[r * k + c] = q as i8 as u8;
            w[r * k + c] = q as f32 * scale;
        }
    }
    (codes, scales)
}

fn w8a8_pack_blob(codes: &[u8], scales: &[f32]) -> Vec<u8> {
    let mut blob = Vec::with_capacity(8 + codes.len() + scales.len() * 4);
    blob.extend_from_slice(&(codes.len() as u64).to_le_bytes());
    blob.extend_from_slice(codes);
    for s in scales {
        blob.extend_from_slice(&s.to_le_bytes());
    }
    blob
}

/// Native W8A8-int8 D2D dispatch vs exact-dequant CPU oracle.
/// Routing runs on-device (mode 0); packed stacks are uploaded once;
/// per-token `a_scale` is ones (quant error lives in the codes, exactly
/// like the dequant path — see `shared_moe` v1 policy).
#[test]
fn device_w8a8_dispatch_matches_exact_dequant_oracle() {
    let _guard = gpu_lock();
    let Some(dev) = gpu_device() else { return };

    // O(1) magnitudes: tiny weights hide routing/scale bugs under tolerance.
    let mut gate_w: Vec<f32> = (0..NUM_EXPERTS * INTER * HIDDEN)
        .map(|i| ((i as f32 + 1.0) * 0.37).sin() + ((i as f32 + 1.0) * 0.11).cos() * 0.5)
        .collect();
    let mut up_w: Vec<f32> = (0..NUM_EXPERTS * INTER * HIDDEN)
        .map(|i| ((i as f32 + 1.0) * 0.53 + 2.0).sin())
        .collect();
    let mut down_w: Vec<f32> = (0..NUM_EXPERTS * HIDDEN * INTER)
        .map(|i| 1.0 / (1.0 + (i as f32) * 0.05))
        .collect();
    let mut logits = vec![0.0f32; SEQ * NUM_EXPERTS];
    for s in 0..SEQ {
        for e in 0..NUM_EXPERTS {
            logits[s * NUM_EXPERTS + e] = ((s as f32 + 1.0) * 0.6 + (e as f32) * 1.1).sin() * 2.5;
        }
    }
    let mut act = vec![0.0f32; SEQ * HIDDEN];
    for s in 0..SEQ {
        for h in 0..HIDDEN {
            act[s * HIDDEN + h] = ((s as f32 + 1.0) * 0.4 + (h as f32 + 1.0) * 0.2).cos() * 1.5;
        }
    }

    // Pack per-expert blobs (gate/up: [INTER, HIDDEN] rows; down: [HIDDEN, INTER]).
    let pack_dtype = || DType {
        arith: grim_tensor::ArithType::F32,
        storage: Storage::CompressedTensorsW8A8Int8,
    };
    let mut gate_stack = Vec::new();
    let mut up_stack = Vec::new();
    let mut down_stack = Vec::new();
    for e in 0..NUM_EXPERTS {
        let gs = e * INTER * HIDDEN;
        let ds = e * HIDDEN * INTER;
        let (gc, gs_sc) = w8a8_int8_quantize_rowmajor(
            &mut gate_w[gs..gs + INTER * HIDDEN],
            INTER,
            HIDDEN,
        );
        let (uc, us_sc) = w8a8_int8_quantize_rowmajor(
            &mut up_w[gs..gs + INTER * HIDDEN],
            INTER,
            HIDDEN,
        );
        let (dc, ds_sc) = w8a8_int8_quantize_rowmajor(
            &mut down_w[ds..ds + HIDDEN * INTER],
            HIDDEN,
            INTER,
        );
        gate_stack.extend_from_slice(&w8a8_pack_blob(&gc, &gs_sc));
        up_stack.extend_from_slice(&w8a8_pack_blob(&uc, &us_sc));
        down_stack.extend_from_slice(&w8a8_pack_blob(&dc, &ds_sc));
    }
    // gate_w/up_w/down_w are now exact-dequant references (rewritten in place).

    let pack_shape = |len: usize| Shape::new(vec![len]);
    let gate_buf = MemoryOps::from_cpu_bytes(&dev, &gate_stack, &pack_shape(gate_stack.len()), pack_dtype())
        .unwrap();
    let up_buf =
        MemoryOps::from_cpu_bytes(&dev, &up_stack, &pack_shape(up_stack.len()), pack_dtype()).unwrap();
    let down_buf = MemoryOps::from_cpu_bytes(&dev, &down_stack, &pack_shape(down_stack.len()), pack_dtype())
        .unwrap();

    let act_st = dev.from_cpu(&act, &Shape::new(vec![SEQ, HIDDEN]), DType::F32).unwrap();
    let act_rocm = act_st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
    let l_st = dev.from_cpu(&logits, &Shape::new(vec![SEQ, NUM_EXPERTS]), DType::F32).unwrap();
    let l_rocm = l_st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();

    let num_pairs = SEQ * TOP_K;
    let u32_dtype = || DType { arith: grim_tensor::ArithType::U32, storage: Storage::Native };
    let tok_st = dev.zeros(&Shape::new(vec![num_pairs]), u32_dtype()).unwrap();
    let exp_st = dev.zeros(&Shape::new(vec![num_pairs]), u32_dtype()).unwrap();
    let w_st = dev.zeros(&Shape::new(vec![num_pairs]), DType::F32).unwrap();
    let tok_rocm = tok_st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
    let exp_rocm = exp_st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
    let w_rocm = w_st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();

    dev.moe_route_topk_on_device(l_rocm, None, tok_rocm, exp_rocm, w_rocm, SEQ, NUM_EXPERTS, TOP_K, 0)
        .unwrap();

    let ascale = vec![1.0f32; SEQ];
    let ascale_st = dev.from_cpu(&ascale, &Shape::new(vec![SEQ]), DType::F32).unwrap();
    let ascale_rocm = ascale_st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();

    let out_shape = Shape::new(vec![SEQ, HIDDEN]);
    let (out_storage, _h) = dev
        .moe_fused_dispatch_resident_routing_w8a8_int8(
            act_rocm,
            &*gate_buf,
            &*up_buf,
            &*down_buf,
            ascale_rocm,
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

    // CPU oracle over the exact-dequant weights with the DEVICE routing.
    let experts = decode_u32(&exp_rocm.copy_to_host().unwrap());
    let weights = w_rocm.to_cpu_vec_f32().unwrap();
    let tokens = decode_u32(&tok_rocm.copy_to_host().unwrap());
    let mut oracle = vec![0.0f32; SEQ * HIDDEN];
    for p in 0..num_pairs {
        let tok = tokens[p] as usize;
        let e = experts[p] as usize;
        let w = weights[p];
        for h in 0..HIDDEN {
            let mut acc = 0.0f32;
            for j in 0..INTER {
                let mut g = 0.0f32;
                let mut u = 0.0f32;
                for i in 0..HIDDEN {
                    g += gate_w[e * INTER * HIDDEN + j * HIDDEN + i] * act[tok * HIDDEN + i];
                    u += up_w[e * INTER * HIDDEN + j * HIDDEN + i] * act[tok * HIDDEN + i];
                }
                let silu = g / (1.0 + (-g).exp());
                let a = silu * u;
                acc += down_w[e * HIDDEN * INTER + h * INTER + j] * a;
            }
            oracle[tok * HIDDEN + h] += w * acc;
        }
    }

    assert_eq!(got.len(), oracle.len());
    let max_diff = got
        .iter()
        .zip(oracle.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("w8a8 D2D vs exact-dequant oracle max_diff={max_diff:.2e}");
    assert!(
        max_diff < 1e-3,
        "w8a8 dispatch mismatch: max diff {max_diff:.6} exceeds 1e-3"
    );
}

// ── WI-gpu-native-moe Phase 2: native W8A8-fp8 / AWQ / MXFP4 dispatch ─────

/// f32 ↔ f16 bit conversions for AWQ scale handling (round-to-nearest-even).
fn f32_to_f16_bits(v: f32) -> u16 {
    const INF32: f32 = f32::INFINITY;
    if v.is_nan() {
        return 0x7E00;
    }
    let sign = if v.is_sign_negative() { 0x8000u32 } else { 0 };
    let a = v.abs();
    if a >= INF32 {
        return (sign | 0x7BFF) as u16;
    }
    if a < 5.960464477539063e-8 {
        return sign as u16;
    }
    let b = (a * 4096.0 + 0.5) as u32;
    let exp = ((b >> 23) as i32) - 112;
    if exp >= 31 {
        return (sign | 0x7BFF) as u16;
    }
    if exp <= 0 {
        return (sign | (b >> 14)) as u16;
    }
    (sign | ((exp as u32) << 10) | ((b >> 13) & 0x3FF)) as u16
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut e = 127 - 14;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            (sign << 31) | (e << 23) | (m << 13)
        }
    } else if exp == 31 {
        (sign << 31) | (0xFF << 23) | (mant << 13)
    } else {
        (sign << 31) | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// Exact fp8-E4M3 quantize: nearest of the VALID codes under a per-tensor
/// scale; `w` rewritten to `decode(code) * scale`.
/// Codes 0x78-0x7F/0xF8-0xFF (exp 15) are EXCLUDED: per OCP they are NaN,
/// and the two decoders (`grim_quant` vs kernel) assign them different
/// finite stand-ins — a real quantizer never emits them either.
fn fp8_pack_tensor(w: &mut [f32]) -> (Vec<u8>, f32) {
    let max_abs = w.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let scale = if max_abs == 0.0 { 1e-12 } else { max_abs / 240.0 };
    let mut codes = vec![0u8; w.len()];
    for (i, v) in w.iter_mut().enumerate() {
        let target = *v / scale;
        let mut best = 0u8;
        let mut best_d = f32::INFINITY;
        for c in 0..256u16 {
            if (c & 0x7F) >= 0x78 {
                continue;
            }
            let d = (grim_quant::fp8_e4m3_to_f32(c as u8) - target).abs();
            if d < best_d {
                best_d = d;
                best = c as u8;
            }
        }
        codes[i] = best;
        *v = grim_quant::fp8_e4m3_to_f32(best) * scale;
    }
    (codes, scale)
}

fn fp8_pack_blob(codes: &[u8], scale: f32) -> Vec<u8> {
    let mut blob = Vec::with_capacity(8 + codes.len() + 4);
    blob.extend_from_slice(&(codes.len() as u64).to_le_bytes());
    blob.extend_from_slice(codes);
    blob.extend_from_slice(&scale.to_le_bytes());
    blob
}

fn route_on_device(
    dev: &RocmDevice,
    logits: &[f32],
    seq: usize,
    num_experts: usize,
    top_k: usize,
    mode: i32,
) -> (
    Box<dyn grim_tensor::backend::BackendStorage>,
    Box<dyn grim_tensor::backend::BackendStorage>,
    Box<dyn grim_tensor::backend::BackendStorage>,
) {
    let num_pairs = seq * top_k;
    let u32d = || DType { arith: grim_tensor::ArithType::U32, storage: Storage::Native };
    let l_st = dev.from_cpu(logits, &Shape::new(vec![seq, num_experts]), DType::F32).unwrap();
    let tok = dev.zeros(&Shape::new(vec![num_pairs]), u32d()).unwrap();
    let exp = dev.zeros(&Shape::new(vec![num_pairs]), u32d()).unwrap();
    let wth = dev.zeros(&Shape::new(vec![num_pairs]), DType::F32).unwrap();
    {
        let l_rocm = l_st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
        let tok_rocm = tok.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
        let exp_rocm = exp.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
        let w_rocm = wth.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
        dev.moe_route_topk_on_device(
            l_rocm, None, tok_rocm, exp_rocm, w_rocm, seq, num_experts, top_k, mode,
        )
        .unwrap();
    }
    (tok, exp, wth)
}

/// Native W8A8-fp8 D2D dispatch vs exact fp8-dequant CPU oracle.
#[test]
fn device_w8a8_fp8_dispatch_matches_exact_oracle() {
    let _guard = gpu_lock();
    let Some(dev) = gpu_device() else { return };

    let mut gate_w: Vec<f32> = (0..NUM_EXPERTS * INTER * HIDDEN)
        .map(|i| ((i as f32 + 1.0) * 0.37).sin() * 1.5)
        .collect();
    let mut up_w: Vec<f32> = (0..NUM_EXPERTS * INTER * HIDDEN)
        .map(|i| ((i as f32 + 1.0) * 0.53 + 2.0).sin() * 1.5)
        .collect();
    let mut down_w: Vec<f32> = (0..NUM_EXPERTS * HIDDEN * INTER)
        .map(|i| 1.5 / (1.0 + (i as f32) * 0.05))
        .collect();
    let mut logits = vec![0.0f32; SEQ * NUM_EXPERTS];
    for s in 0..SEQ {
        for e in 0..NUM_EXPERTS {
            logits[s * NUM_EXPERTS + e] = ((s as f32 + 1.0) * 0.6 + (e as f32) * 1.1).sin() * 2.5;
        }
    }
    let act: Vec<f32> = (0..SEQ * HIDDEN)
        .map(|i| ((i as f32 + 1.0) * 0.2).cos() * 1.5)
        .collect();

    let pack_dtype = || DType {
        arith: grim_tensor::ArithType::F32,
        storage: Storage::CompressedTensorsW8A8Fp8,
    };
    let mut gate_stack = Vec::new();
    let mut up_stack = Vec::new();
    let mut down_stack = Vec::new();
    for e in 0..NUM_EXPERTS {
        let gs = e * INTER * HIDDEN;
        let ds = e * HIDDEN * INTER;
        let (gc, gsc) = {
            let s = &mut gate_w[gs..gs + INTER * HIDDEN];
            let (c, sc) = fp8_pack_tensor(s);
            (c, sc)
        };
        let (uc, usc) = {
            let s = &mut up_w[gs..gs + INTER * HIDDEN];
            let (c, sc) = fp8_pack_tensor(s);
            (c, sc)
        };
        let (dc, dsc) = {
            let s = &mut down_w[ds..ds + HIDDEN * INTER];
            let (c, sc) = fp8_pack_tensor(s);
            (c, sc)
        };
        gate_stack.extend_from_slice(&fp8_pack_blob(&gc, gsc));
        up_stack.extend_from_slice(&fp8_pack_blob(&uc, usc));
        down_stack.extend_from_slice(&fp8_pack_blob(&dc, dsc));
    }

    let gate_buf = MemoryOps::from_cpu_bytes(&dev, &gate_stack, &Shape::new(vec![gate_stack.len()]), pack_dtype()).unwrap();
    let up_buf = MemoryOps::from_cpu_bytes(&dev, &up_stack, &Shape::new(vec![up_stack.len()]), pack_dtype()).unwrap();
    let down_buf = MemoryOps::from_cpu_bytes(&dev, &down_stack, &Shape::new(vec![down_stack.len()]), pack_dtype()).unwrap();

    let act_st = dev.from_cpu(&act, &Shape::new(vec![SEQ, HIDDEN]), DType::F32).unwrap();
    let act_rocm = act_st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
    let (tok_b, exp_b, w_b) = route_on_device(&dev, &logits, SEQ, NUM_EXPERTS, TOP_K, 0);
    let tok_rocm = tok_b.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
    let exp_rocm = exp_b.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
    let w_rocm = w_b.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();

    let ascale = vec![1.0f32; SEQ];
    let ascale_st = dev.from_cpu(&ascale, &Shape::new(vec![SEQ]), DType::F32).unwrap();
    let ascale_rocm = ascale_st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();

    let out_shape = Shape::new(vec![SEQ, HIDDEN]);
    let (out_storage, _h) = dev
        .moe_fused_dispatch_resident_routing_w8a8_fp8(
            act_rocm, &*gate_buf, &*up_buf, &*down_buf, ascale_rocm,
            tok_rocm, exp_rocm, w_rocm, SEQ * TOP_K, &out_shape, HIDDEN, INTER, 1.0,
        )
        .unwrap();
    dev.synchronize();
    let got = out_storage.to_cpu_vec_f32().unwrap();

    let experts = decode_u32(&exp_rocm.copy_to_host().unwrap());
    let weights = w_rocm.to_cpu_vec_f32().unwrap();
    let tokens = decode_u32(&tok_rocm.copy_to_host().unwrap());
    let mut oracle = vec![0.0f32; SEQ * HIDDEN];
    for p in 0..SEQ * TOP_K {
        let tok = tokens[p] as usize;
        let e = experts[p] as usize;
        let w = weights[p];
        for h in 0..HIDDEN {
            let mut acc = 0.0f32;
            for j in 0..INTER {
                let mut g = 0.0f32;
                let mut u = 0.0f32;
                for i in 0..HIDDEN {
                    g += gate_w[e * INTER * HIDDEN + j * HIDDEN + i] * act[tok * HIDDEN + i];
                    u += up_w[e * INTER * HIDDEN + j * HIDDEN + i] * act[tok * HIDDEN + i];
                }
                let silu = g / (1.0 + (-g).exp());
                let a = silu * u;
                acc += down_w[e * HIDDEN * INTER + h * INTER + j] * a;
            }
            oracle[tok * HIDDEN + h] += w * acc;
        }
    }

    let max_diff = got.iter().zip(oracle.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("w8a8_fp8 D2D vs exact oracle max_diff={max_diff:.2e}");
    assert!(max_diff < 1e-3, "w8a8_fp8 dispatch mismatch: {max_diff:.6} exceeds 1e-3");
}

/// AWQ 4-bit exact fixture: per-group (zero=8, scale=max/7) with f16 scales;
/// `w` rewritten to `(code-8) * f16(scale)`. Returns packed per-expert blob
/// `[u64 qwlen | qw | u64 qzlen | qzeros | u64 sclen | f16 scales]`.
fn awq4_pack_rows(w: &mut [f32], rows: usize, k: usize, group_size: usize) -> Vec<u8> {
    const BITS: usize = 4;
    const VPW: usize = 8;
    let groups = k.div_ceil(group_size);
    let qw_len = k.div_ceil(VPW) * rows * 4;
    let qz_len = groups * rows.div_ceil(VPW) * 4;
    let mut qw = vec![0u32; qw_len / 4];
    let mut sc = vec![0u16; groups * rows];
    for r in 0..rows {
        for g in 0..groups {
            let lo = g * group_size;
            let hi = (lo + group_size).min(k);
            let max_abs = (lo..hi).map(|c| w[r * k + c].abs()).fold(0.0f32, f32::max);
            let fscale = f32_to_f16_bits(if max_abs == 0.0 { 1e-12 } else { max_abs / 7.0 });
            sc[g * rows + r] = fscale;
            let fsc = f16_bits_to_f32(fscale);
            for c in lo..hi {
                // Clamp in float domain BEFORE int conversion: an unbounded
                // ratio saturates `as i32` to i32::MAX and the `+ 8` then
                // overflows (debug panic). Bounded here, clamped again after.
                let q = ((w[r * k + c] / fsc).round().clamp(-8.0, 7.0) as i32 + 8).clamp(0, 15) as u32;
                let wi = (c / VPW) * rows + r;
                qw[wi] |= q << ((c % VPW) * BITS);
                w[r * k + c] = (q as f32 - 8.0) * fsc;
            }
        }
    }
    // qzeros: zero point 8 for every column, packed VPW nibbles per word.
    let mut qz = vec![0u32; qz_len / 4];
    for g in 0..groups {
        for c in 0..rows {
            let word = g * rows.div_ceil(VPW) + c / VPW;
            qz[word] |= 8 << ((c % VPW) * BITS);
        }
    }
    let mut blob = Vec::new();
    blob.extend_from_slice(&(qw_len as u64).to_le_bytes());
    for x in &qw {
        blob.extend_from_slice(&x.to_le_bytes());
    }
    blob.extend_from_slice(&(qz_len as u64).to_le_bytes());
    for x in &qz {
        blob.extend_from_slice(&x.to_le_bytes());
    }
    let sc_len = groups * rows * 2;
    blob.extend_from_slice(&(sc_len as u64).to_le_bytes());
    for x in &sc {
        blob.extend_from_slice(&x.to_le_bytes());
    }
    blob
}

/// Native AWQ D2D dispatch vs blob-exact CPU oracle (the oracle decodes the
/// packed blobs with the same (code-zero)*f16(scale) math as the kernel).
#[test]
fn device_awq_dispatch_matches_blob_exact_oracle() {
    let _guard = gpu_lock();
    let Some(dev) = gpu_device() else { return };

    const BITS: u8 = 4;
    const GROUP: usize = 8;

    let gate_w: Vec<f32> = (0..NUM_EXPERTS * INTER * HIDDEN)
        .map(|i| ((i as f32 + 1.0) * 0.37).sin() * 1.5)
        .collect();
    let up_w: Vec<f32> = (0..NUM_EXPERTS * INTER * HIDDEN)
        .map(|i| ((i as f32 + 1.0) * 0.53 + 2.0).sin() * 1.5)
        .collect();
    let down_w: Vec<f32> = (0..NUM_EXPERTS * HIDDEN * INTER)
        .map(|i| 1.5 / (1.0 + (i as f32) * 0.05))
        .collect();
    let logits: Vec<f32> = (0..SEQ * NUM_EXPERTS)
        .map(|i| ((i as f32 + 1.0) * 0.6).sin() * 2.5)
        .collect();
    let act: Vec<f32> = (0..SEQ * HIDDEN)
        .map(|i| ((i as f32 + 1.0) * 0.2).cos() * 1.5)
        .collect();

    // Pack + keep decoded f32 twins for the oracle.
    let mut gate_stack = Vec::new();
    let mut up_stack = Vec::new();
    let mut down_stack = Vec::new();
    let mut gate_ref = Vec::new();
    let mut up_ref = Vec::new();
    let mut down_ref = Vec::new();
    for e in 0..NUM_EXPERTS {
        let gs = e * INTER * HIDDEN;
        let ds = e * HIDDEN * INTER;
        let mut gslab = gate_w[gs..gs + INTER * HIDDEN].to_vec();
        let mut uslab = up_w[gs..gs + INTER * HIDDEN].to_vec();
        let mut dslab = down_w[ds..ds + HIDDEN * INTER].to_vec();
        gate_stack.extend_from_slice(&awq4_pack_rows(&mut gslab, INTER, HIDDEN, GROUP));
        up_stack.extend_from_slice(&awq4_pack_rows(&mut uslab, INTER, HIDDEN, GROUP));
        down_stack.extend_from_slice(&awq4_pack_rows(&mut dslab, HIDDEN, INTER, GROUP));
        gate_ref.extend_from_slice(&gslab);
        up_ref.extend_from_slice(&uslab);
        down_ref.extend_from_slice(&dslab);
    }

    let awq_dtype = || DType {
        arith: grim_tensor::ArithType::F32,
        storage: Storage::Awq(grim_tensor::dtype::AwqStorageConfig { bits: BITS, group_size: GROUP }),
    };
    let gate_buf = MemoryOps::from_cpu_bytes(&dev, &gate_stack, &Shape::new(vec![gate_stack.len()]), awq_dtype()).unwrap();
    let up_buf = MemoryOps::from_cpu_bytes(&dev, &up_stack, &Shape::new(vec![up_stack.len()]), awq_dtype()).unwrap();
    let down_buf = MemoryOps::from_cpu_bytes(&dev, &down_stack, &Shape::new(vec![down_stack.len()]), awq_dtype()).unwrap();

    let act_st = dev.from_cpu(&act, &Shape::new(vec![SEQ, HIDDEN]), DType::F32).unwrap();
    let act_rocm = act_st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
    let (tok_b, exp_b, w_b) = route_on_device(&dev, &logits, SEQ, NUM_EXPERTS, TOP_K, 0);
    let tok_rocm = tok_b.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
    let exp_rocm = exp_b.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
    let w_rocm = w_b.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();

    let ascale = vec![1.0f32; SEQ];
    let ascale_st = dev.from_cpu(&ascale, &Shape::new(vec![SEQ]), DType::F32).unwrap();
    let ascale_rocm = ascale_st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();

    let out_shape = Shape::new(vec![SEQ, HIDDEN]);
    let (out_storage, _h) = dev
        .moe_fused_dispatch_resident_routing_awq(
            act_rocm, &*gate_buf, &*up_buf, &*down_buf, ascale_rocm,
            tok_rocm, exp_rocm, w_rocm, SEQ * TOP_K, &out_shape,
            HIDDEN, INTER, BITS, GROUP, 1.0,
        )
        .unwrap();
    dev.synchronize();
    let got = out_storage.to_cpu_vec_f32().unwrap();

    let experts = decode_u32(&exp_rocm.copy_to_host().unwrap());
    let weights = w_rocm.to_cpu_vec_f32().unwrap();
    let tokens = decode_u32(&tok_rocm.copy_to_host().unwrap());
    let mut oracle = vec![0.0f32; SEQ * HIDDEN];
    for p in 0..SEQ * TOP_K {
        let tok = tokens[p] as usize;
        let e = experts[p] as usize;
        let w = weights[p];
        for h in 0..HIDDEN {
            let mut acc = 0.0f32;
            for j in 0..INTER {
                let mut g = 0.0f32;
                let mut u = 0.0f32;
                for i in 0..HIDDEN {
                    g += gate_ref[e * INTER * HIDDEN + j * HIDDEN + i] * act[tok * HIDDEN + i];
                    u += up_ref[e * INTER * HIDDEN + j * HIDDEN + i] * act[tok * HIDDEN + i];
                }
                let silu = g / (1.0 + (-g).exp());
                let a = silu * u;
                acc += down_ref[e * HIDDEN * INTER + h * INTER + j] * a;
            }
            oracle[tok * HIDDEN + h] += w * acc;
        }
    }

    let max_diff = got.iter().zip(oracle.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("awq D2D vs blob-exact oracle max_diff={max_diff:.2e}");
    assert!(max_diff < 1e-3, "awq dispatch mismatch: {max_diff:.6} exceeds 1e-3");
}

/// Exact MXFP4 quantize: shared exp 127 (scale 1.0) per 32-group, nearest of
/// the 16 E2M1 codes; `w` rewritten to the decoded value. Returns
/// `(packed_code_bytes, exp_bytes)`, low nibble = even index (matches
/// `mxfp4_code_at` and `grim_quant::dequant_mxfp4`).
fn mxfp4_pack_tensor(w: &mut [f32]) -> (Vec<u8>, Vec<u8>) {
    let groups = w.len().div_ceil(32);
    let mut codes = vec![0u8; w.len().div_ceil(2)];
    let exps = vec![127u8; groups];
    for (i, v) in w.iter_mut().enumerate() {
        let mut best = 0u8;
        let mut best_d = f32::INFINITY;
        for c in 0..16u8 {
            let d = (grim_quant::mxfp4_e2m1_to_f32(c, 127) - *v).abs();
            if d < best_d {
                best_d = d;
                best = c;
            }
        }
        if i % 2 == 0 {
            codes[i / 2] = (codes[i / 2] & 0xF0) | best;
        } else {
            codes[i / 2] = (codes[i / 2] & 0x0F) | (best << 4);
        }
        *v = grim_quant::mxfp4_e2m1_to_f32(best, 127);
    }
    (codes, exps)
}

/// Native MXFP4 D2D dispatch vs exact-dequant CPU oracle.
#[test]
fn device_mxfp4_dispatch_matches_exact_oracle() {
    let _guard = gpu_lock();
    let Some(dev) = gpu_device() else { return };

    let mut gate_w: Vec<f32> = (0..NUM_EXPERTS * INTER * HIDDEN)
        .map(|i| ((i as f32 + 1.0) * 0.37).sin())
        .collect();
    let mut up_w: Vec<f32> = (0..NUM_EXPERTS * INTER * HIDDEN)
        .map(|i| ((i as f32 + 1.0) * 0.53 + 2.0).sin() * 0.5)
        .collect();
    let mut down_w: Vec<f32> = (0..NUM_EXPERTS * HIDDEN * INTER)
        .map(|i| 1.0 / (1.0 + (i as f32) * 0.05) - 0.5)
        .collect();
    let logits: Vec<f32> = (0..SEQ * NUM_EXPERTS)
        .map(|i| ((i as f32 + 1.0) * 0.6).sin() * 2.5)
        .collect();
    let act: Vec<f32> = (0..SEQ * HIDDEN)
        .map(|i| ((i as f32 + 1.0) * 0.2).cos() * 1.5)
        .collect();

    let mxfp4_dtype = || DType {
        arith: grim_tensor::ArithType::F32,
        storage: grim_tensor::Storage::FloatPack(grim_tensor::FloatPackScheme::MxFp4),
    };
    let mut cg = Vec::new();
    let mut cu = Vec::new();
    let mut cd = Vec::new();
    let mut eg = Vec::new();
    let mut eu = Vec::new();
    let mut ed = Vec::new();
    for e in 0..NUM_EXPERTS {
        let gs = e * INTER * HIDDEN;
        let ds = e * HIDDEN * INTER;
        let (gc, ge) = mxfp4_pack_tensor(&mut gate_w[gs..gs + INTER * HIDDEN]);
        let (uc, ue) = mxfp4_pack_tensor(&mut up_w[gs..gs + INTER * HIDDEN]);
        let (dc, de) = mxfp4_pack_tensor(&mut down_w[ds..ds + HIDDEN * INTER]);
        cg.extend_from_slice(&gc);
        cu.extend_from_slice(&uc);
        cd.extend_from_slice(&dc);
        eg.extend_from_slice(&ge);
        eu.extend_from_slice(&ue);
        ed.extend_from_slice(&de);
    }
    let up_stack = |v: &[u8]| {
        MemoryOps::from_cpu_bytes(&dev, v, &Shape::new(vec![v.len()]), mxfp4_dtype()).unwrap()
    };
    let (gate_c, up_c, down_c, gate_e, up_e, down_e) =
        (up_stack(&cg), up_stack(&cu), up_stack(&cd), up_stack(&eg), up_stack(&eu), up_stack(&ed));

    let act_st = dev.from_cpu(&act, &Shape::new(vec![SEQ, HIDDEN]), DType::F32).unwrap();
    let act_rocm = act_st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
    let (tok_b, exp_b, w_b) = route_on_device(&dev, &logits, SEQ, NUM_EXPERTS, TOP_K, 0);
    let tok_rocm = tok_b.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
    let exp_rocm = exp_b.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
    let w_rocm = w_b.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();

    let ascale = vec![1.0f32; SEQ];
    let ascale_st = dev.from_cpu(&ascale, &Shape::new(vec![SEQ]), DType::F32).unwrap();
    let ascale_rocm = ascale_st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();

    let out_shape = Shape::new(vec![SEQ, HIDDEN]);
    let (out_storage, _h) = dev
        .moe_fused_dispatch_resident_routing_mxfp4(
            act_rocm, &*gate_c, &*up_c, &*down_c, &*gate_e, &*up_e, &*down_e,
            ascale_rocm, tok_rocm, exp_rocm, w_rocm, SEQ * TOP_K,
            &out_shape, HIDDEN, INTER, 1.0,
        )
        .unwrap();
    dev.synchronize();
    let got = out_storage.to_cpu_vec_f32().unwrap();

    let experts = decode_u32(&exp_rocm.copy_to_host().unwrap());
    let weights = w_rocm.to_cpu_vec_f32().unwrap();
    let tokens = decode_u32(&tok_rocm.copy_to_host().unwrap());
    let mut oracle = vec![0.0f32; SEQ * HIDDEN];
    for p in 0..SEQ * TOP_K {
        let tok = tokens[p] as usize;
        let e = experts[p] as usize;
        let w = weights[p];
        for h in 0..HIDDEN {
            let mut acc = 0.0f32;
            for j in 0..INTER {
                let mut g = 0.0f32;
                let mut u = 0.0f32;
                for i in 0..HIDDEN {
                    g += gate_w[e * INTER * HIDDEN + j * HIDDEN + i] * act[tok * HIDDEN + i];
                    u += up_w[e * INTER * HIDDEN + j * HIDDEN + i] * act[tok * HIDDEN + i];
                }
                let silu = g / (1.0 + (-g).exp());
                let a = silu * u;
                acc += down_w[e * HIDDEN * INTER + h * INTER + j] * a;
            }
            oracle[tok * HIDDEN + h] += w * acc;
        }
    }

    let max_diff = got.iter().zip(oracle.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("mxfp4 D2D vs exact oracle max_diff={max_diff:.2e}");
    assert!(max_diff < 1e-3, "mxfp4 dispatch mismatch: {max_diff:.6} exceeds 1e-3");
}

/// Native W8A8-int8 DOT4 dispatch vs exact-dequant CPU oracle.
/// hidden % 32 == 0 is required (per-32 activation blocks); other shapes
/// stay on the scalar kernel (asserted by the launcher's loud refusal —
/// covered implicitly: this test's dims satisfy it).
#[test]
fn device_w8a8_dot4_dispatch_matches_exact_oracle() {
    let _guard = gpu_lock();
    let Some(dev) = gpu_device() else { return };
    if !grim_backend_rocm::kernels::charon::dot4_supported(
        grim_backend_rocm::RocmDevice::shared(0).gcn_arch(),
    ) {
        eprintln!("[skip: no dot4 on this arch]");
        return;
    }

    const H: usize = 64;
    const I: usize = 64;
    const E: usize = 4;
    const SEQ: usize = 2;
    const TOPK: usize = 2;

    let mut gate_w: Vec<f32> = (0..E * I * H)
        .map(|i| ((i as f32 + 1.0) * 0.37).sin())
        .collect();
    let mut up_w: Vec<f32> = (0..E * I * H)
        .map(|i| ((i as f32 + 1.0) * 0.53 + 2.0).sin() * 0.5)
        .collect();
    let mut down_w: Vec<f32> = (0..E * H * I)
        .map(|i| 1.0 / (1.0 + (i as f32) * 0.05) - 0.5)
        .collect();
    let logits: Vec<f32> = (0..SEQ * E)
        .map(|i| ((i as f32 + 1.0) * 0.6).sin() * 2.5)
        .collect();
    let act: Vec<f32> = (0..SEQ * H)
        .map(|i| ((i as f32 + 1.0) * 0.2).cos() * 1.5)
        .collect();

    let pack_dtype = || DType {
        arith: grim_tensor::ArithType::F32,
        storage: Storage::CompressedTensorsW8A8Int8,
    };
    let mut gate_stack = Vec::new();
    let mut up_stack = Vec::new();
    let mut down_stack = Vec::new();
    for e in 0..E {
        let gs = e * I * H;
        let ds = e * H * I;
        let (gc, gsc) = w8a8_int8_quantize_rowmajor(&mut gate_w[gs..gs + I * H], I, H);
        let (uc, usc) = w8a8_int8_quantize_rowmajor(&mut up_w[gs..gs + I * H], I, H);
        let (dc, dsc) = w8a8_int8_quantize_rowmajor(&mut down_w[ds..ds + H * I], H, I);
        gate_stack.extend_from_slice(&w8a8_pack_blob(&gc, &gsc));
        up_stack.extend_from_slice(&w8a8_pack_blob(&uc, &usc));
        down_stack.extend_from_slice(&w8a8_pack_blob(&dc, &dsc));
    }
    let gate_buf = MemoryOps::from_cpu_bytes(&dev, &gate_stack, &Shape::new(vec![gate_stack.len()]), pack_dtype()).unwrap();
    let up_buf = MemoryOps::from_cpu_bytes(&dev, &up_stack, &Shape::new(vec![up_stack.len()]), pack_dtype()).unwrap();
    let down_buf = MemoryOps::from_cpu_bytes(&dev, &down_stack, &Shape::new(vec![down_stack.len()]), pack_dtype()).unwrap();

    let act_st = dev.from_cpu(&act, &Shape::new(vec![SEQ, H]), DType::F32).unwrap();
    let act_rocm = act_st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
    let (tok_b, exp_b, w_b) = route_on_device(&dev, &logits, SEQ, E, TOPK, 0);
    let tok_rocm = tok_b.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
    let exp_rocm = exp_b.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
    let w_rocm = w_b.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();

    let ascale = vec![1.0f32; SEQ];
    let ascale_st = dev.from_cpu(&ascale, &Shape::new(vec![SEQ]), DType::F32).unwrap();
    let ascale_rocm = ascale_st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();

    let out_shape = Shape::new(vec![SEQ, H]);
    let (out_storage, _h) = dev
        .moe_fused_dispatch_resident_routing_w8a8_int8_dot4(
            act_rocm, &*gate_buf, &*up_buf, &*down_buf, ascale_rocm,
            tok_rocm, exp_rocm, w_rocm, SEQ * TOPK, &out_shape, H, I, 1.0,
        )
        .unwrap();
    dev.synchronize();
    let got = out_storage.to_cpu_vec_f32().unwrap();

    let experts = decode_u32(&exp_rocm.copy_to_host().unwrap());
    let weights = w_rocm.to_cpu_vec_f32().unwrap();
    let tokens = decode_u32(&tok_rocm.copy_to_host().unwrap());
    let mut oracle = vec![0.0f32; SEQ * H];
    for p in 0..SEQ * TOPK {
        let tok = tokens[p] as usize;
        let e = experts[p] as usize;
        let w = weights[p];
        for h in 0..H {
            let mut acc = 0.0f32;
            for j in 0..I {
                let mut g = 0.0f32;
                let mut u = 0.0f32;
                for i in 0..H {
                    g += gate_w[e * I * H + j * H + i] * act[tok * H + i];
                    u += up_w[e * I * H + j * H + i] * act[tok * H + i];
                }
                let silu = g / (1.0 + (-g).exp());
                let a = silu * u;
                acc += down_w[e * H * I + h * I + j] * a;
            }
            oracle[tok * H + h] += w * acc;
        }
    }

    let max_diff = got.iter().zip(oracle.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("w8a8_dot4 D2D vs exact oracle max_diff={max_diff:.2e}");
    // NOTE: tolerance here is the Q8_1 ACTIVATION-quantization noise floor,
    // not kernel slop — the dot4 contraction quantizes activations to int8
    // per 32-block (da = amax/127), unlike the scalar kernel which keeps
    // them float. Measured 1.24e-2 at O(1) magnitudes. Implementation bugs
    // are caught by the grouped-twin cross-check below (same quantization
    // on both sides), which must agree tightly.
    assert!(max_diff < 5e-2, "w8a8_dot4 dispatch blew past quantization noise: {max_diff:.6}");

    // Cross-check: sortless-dot4 vs grouped-dot4 (host-sorted routing).
    // Both quantize the same 32-blocks identically; only summation order
    // differs, so agreement must be near-exact.
    use grim_backend_rocm::kernels::charon::{RoutingAssignment, moe_align_block_size};
    let mut per_tok_idx: Vec<Vec<usize>> = vec![Vec::new(); SEQ];
    let mut per_tok_w: Vec<Vec<f32>> = vec![Vec::new(); SEQ];
    for p in 0..SEQ * TOPK {
        per_tok_idx[tokens[p] as usize].push(experts[p] as usize);
        per_tok_w[tokens[p] as usize].push(weights[p]);
    }
    let assignment = RoutingAssignment::from_route(&per_tok_idx, &per_tok_w).unwrap();
    let sorted = moe_align_block_size(&assignment, 64, E);
    let (grouped_out, _h2) = dev
        .moe_fused_grouped_dispatch_w8a8_int8(
            act_rocm, &*gate_buf, &*up_buf, &*down_buf, ascale_rocm,
            &sorted, &out_shape, H, I, E, 1.0,
        )
        .unwrap();
    dev.synchronize();
    let grouped_v = grouped_out.to_cpu_vec_f32().unwrap();
    let twin_diff = got
        .iter()
        .zip(grouped_v.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("w8a8_dot4 sortless vs grouped max_diff={twin_diff:.2e}");
    assert!(
        twin_diff < 5e-4,
        "sortless-dot4 vs grouped-dot4 diverged: {twin_diff:.6} (implementation bug, not quantization)"
    );
}

// ── WI-gpu-native-moe Phase 2: native-vs-dequant head-to-head ─────────────
// Device-gated + ignored (timing-sensitive; run explicitly on gfx1201).
// Compares the native W8A8-int8 D2D dispatch against the f32-dequant D2D
// dispatch at a decode-like shape. Print-only: numbers go to the WI perf
// gate, no hard assertion (shared-CI timing is not gateable).
#[ignore = "timing bench: GRIM_RUN_GPU_TESTS=1 cargo test --test moe_d2d_routing_gpu -- --ignored w8a8_head_to_head"]
#[test]
fn w8a8_head_to_head_vs_dequant_decode_shape() {
    let _guard = gpu_lock();
    let Some(dev) = gpu_device() else { return };

    const SEQ: usize = 8;
    const HIDDEN: usize = 2048;
    const INTER: usize = 4096;
    const EXPERTS: usize = 32;
    const TOPK: usize = 8;
    const ITERS: usize = 20;

    // Regular (non-exact) symmetric int8 quantize; perf only, no oracle.
    let mut gate_f32: Vec<f32> = (0..EXPERTS * INTER * HIDDEN)
        .map(|i| ((i as f32 + 1.0) * 0.37).sin())
        .collect();
    let mut up_f32 = gate_f32.clone();
    let mut down_f32: Vec<f32> = (0..EXPERTS * HIDDEN * INTER)
        .map(|i| ((i as f32 + 1.0) * 0.53).cos() * 0.5)
        .collect();
    // Build packed stacks per expert (gate/up [INTER,HIDDEN], down [HIDDEN,INTER]).
    let mut gate_stack = Vec::new();
    let mut up_stack = Vec::new();
    let mut down_stack = Vec::new();
    for e in 0..EXPERTS {
        for (stack, base, rows, k, src) in [
            (&mut gate_stack, e * INTER * HIDDEN, INTER, HIDDEN, &mut gate_f32),
            (&mut up_stack, e * INTER * HIDDEN, INTER, HIDDEN, &mut up_f32),
            (&mut down_stack, e * HIDDEN * INTER, HIDDEN, INTER, &mut down_f32),
        ] {
            let slab = &mut src[base..base + rows * k];
            let mut codes = vec![0u8; rows * k];
            let mut scales = vec![0.0f32; rows];
            for r in 0..rows {
                let max_abs = (0..k).map(|c| slab[r * k + c].abs()).fold(0.0f32, f32::max);
                let sc = if max_abs == 0.0 { 1e-12 } else { max_abs / 127.0 };
                scales[r] = sc;
                for c in 0..k {
                    let q = ((slab[r * k + c] / sc).round() as i32).clamp(-128, 127);
                    codes[r * k + c] = q as i8 as u8;
                    slab[r * k + c] = q as f32 * sc;
                }
            }
            stack.extend_from_slice(&(codes.len() as u64).to_le_bytes());
            stack.extend_from_slice(&codes);
            for s in &scales {
                stack.extend_from_slice(&s.to_le_bytes());
            }
        }
    }

    let pack_dtype = || DType {
        arith: grim_tensor::ArithType::F32,
        storage: Storage::CompressedTensorsW8A8Int8,
    };
    let w8_gate = MemoryOps::from_cpu_bytes(&dev, &gate_stack, &Shape::new(vec![gate_stack.len()]), pack_dtype()).unwrap();
    let w8_up = MemoryOps::from_cpu_bytes(&dev, &up_stack, &Shape::new(vec![up_stack.len()]), pack_dtype()).unwrap();
    let w8_down = MemoryOps::from_cpu_bytes(&dev, &down_stack, &Shape::new(vec![down_stack.len()]), pack_dtype()).unwrap();
    // gate_f32/up_f32/down_f32 now hold exact-dequant values: stack f32 twin.
    let f32_of = |v: &[f32]| dev.from_cpu(v, &Shape::new(vec![v.len()]), DType::F32).unwrap();
    let f_gate = f32_of(&gate_f32);
    let f_up = f32_of(&up_f32);
    let f_down = f32_of(&down_f32);

    let mut logits = vec![0.0f32; SEQ * EXPERTS];
    for s in 0..SEQ {
        for e in 0..EXPERTS {
            logits[s * EXPERTS + e] = ((s as f32 + 1.0) * 0.6 + (e as f32) * 1.1).sin() * 2.5;
        }
    }
    let act: Vec<f32> = (0..SEQ * HIDDEN)
        .map(|i| ((i as f32 + 1.0) * 0.2).cos())
        .collect();
    let act_st = dev.from_cpu(&act, &Shape::new(vec![SEQ, HIDDEN]), DType::F32).unwrap();
    let act_rocm = act_st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
    let l_st = dev.from_cpu(&logits, &Shape::new(vec![SEQ, EXPERTS]), DType::F32).unwrap();
    let l_rocm = l_st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();

    let num_pairs = SEQ * TOPK;
    let u32d = || DType { arith: grim_tensor::ArithType::U32, storage: Storage::Native };
    let tok_st = dev.zeros(&Shape::new(vec![num_pairs]), u32d()).unwrap();
    let exp_st = dev.zeros(&Shape::new(vec![num_pairs]), u32d()).unwrap();
    let w_st = dev.zeros(&Shape::new(vec![num_pairs]), DType::F32).unwrap();
    let tok_rocm = tok_st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
    let exp_rocm = exp_st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
    let w_rocm = w_st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
    dev.moe_route_topk_on_device(l_rocm, None, tok_rocm, exp_rocm, w_rocm, SEQ, EXPERTS, TOPK, 0)
        .unwrap();

    let ascale = vec![1.0f32; SEQ];
    let ascale_st = dev.from_cpu(&ascale, &Shape::new(vec![SEQ]), DType::F32).unwrap();
    let ascale_rocm = ascale_st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
    let out_shape = Shape::new(vec![SEQ, HIDDEN]);

    // Warmup (JIT + caches), then timed iters with device sync.
    let time_arm = |native: bool| -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..ITERS {
            let start = std::time::Instant::now();
            if native {
                let _ = dev
                    .moe_fused_dispatch_resident_routing_w8a8_int8(
                        act_rocm, &*w8_gate, &*w8_up, &*w8_down, ascale_rocm,
                        tok_rocm, exp_rocm, w_rocm, num_pairs, &out_shape,
                        HIDDEN, INTER, 1.0,
                    )
                    .unwrap();
            } else {
                let _ = dev
                    .moe_fused_dispatch_resident_routing(
                        act_rocm, &*f_gate, &*f_up, &*f_down,
                        tok_rocm, exp_rocm, w_rocm, num_pairs, &out_shape,
                        HIDDEN, INTER, 1.0,
                    )
                    .unwrap();
            }
            dev.synchronize();
            let dt = start.elapsed().as_secs_f64() * 1e3;
            if dt < best {
                best = dt;
            }
        }
        best
    };

    let t_dequant = time_arm(false);
    let t_native = time_arm(true);
    eprintln!(
        "[bench] decode-like h={HIDDEN} i={INTER} e={EXPERTS} topk={TOPK} seq={SEQ}: f32-dequant best={t_dequant:.3}ms w8a8-native best={t_native:.3}ms speedup={:.2}x",
        t_dequant / t_native.max(1e-9),
    );
}

// ── WI-gpu-native-moe Phase 2: all-native-arms head-to-head ────────────────
// Device-gated + ignored (timing-sensitive). Times every wired native arm
// against the f32-dequant arm at one decode-like shape. Print-only.
#[ignore = "timing bench: GRIM_RUN_GPU_TESTS=1 cargo test --test moe_d2d_routing_gpu -- --ignored quant_natives_head_to_head"]
#[test]
fn quant_natives_head_to_head() {
    let _guard = gpu_lock();
    let Some(dev) = gpu_device() else { return };

    const SEQ: usize = 4;
    const HIDDEN: usize = 1024;
    const INTER: usize = 2048;
    const EXPERTS: usize = 16;
    const TOPK: usize = 4;
    const ITERS: usize = 8;

    // Base f32 weights (also the exact-dequant reference after packing).
    // Base f32 weights (unquantized f32 twin for the traffic comparison).
    let gate_f: Vec<f32> = (0..EXPERTS * INTER * HIDDEN)
        .map(|i| ((i as f32 + 1.0) * 0.37).sin())
        .collect();
    let up_f: Vec<f32> = (0..EXPERTS * INTER * HIDDEN)
        .map(|i| ((i as f32 + 1.0) * 0.53).cos() * 0.5)
        .collect();
    let down_f: Vec<f32> = (0..EXPERTS * HIDDEN * INTER)
        .map(|i| 1.0 / (1.0 + (i as f32) * 0.02) - 0.5)
        .collect();

    // Pack per-expert stacks for int8 / fp8 / awq4-g8 / mxfp4.
    let mut i8g = Vec::new();
    let mut i8u = Vec::new();
    let mut i8d = Vec::new();
    let mut f8g = Vec::new();
    let mut f8u = Vec::new();
    let mut f8d = Vec::new();
    let mut aqg = Vec::new();
    let mut aqu = Vec::new();
    let mut aqd = Vec::new();
    let mut m4cg = Vec::new();
    let mut m4cu = Vec::new();
    let mut m4cd = Vec::new();
    let mut m4eg = Vec::new();
    let mut m4eu = Vec::new();
    let mut m4ed = Vec::new();
    for e in 0..EXPERTS {
        let gs = e * INTER * HIDDEN;
        let ds = e * HIDDEN * INTER;
        // int8 + fp8 + awq share the exact-rewritten slabs via gate_f etc.;
        // pack each format from a FRESH copy so rewrites don't interact.
        let mut gi8 = gate_f[gs..gs + INTER * HIDDEN].to_vec();
        let mut ui8 = up_f[gs..gs + INTER * HIDDEN].to_vec();
        let mut di8 = down_f[ds..ds + HIDDEN * INTER].to_vec();
        let (gc, gs_s) = {
            let (c, s) = w8a8_int8_quantize_rowmajor(&mut gi8, INTER, HIDDEN);
            (c, s)
        };
        i8g.extend_from_slice(&w8a8_pack_blob(&gc, &gs_s));
        let (uc, us_s) = w8a8_int8_quantize_rowmajor(&mut ui8, INTER, HIDDEN);
        i8u.extend_from_slice(&w8a8_pack_blob(&uc, &us_s));
        let (dc, ds_s) = w8a8_int8_quantize_rowmajor(&mut di8, HIDDEN, INTER);
        i8d.extend_from_slice(&w8a8_pack_blob(&dc, &ds_s));

        let mut gf8 = gate_f[gs..gs + INTER * HIDDEN].to_vec();
        let (gc8, gs8) = fp8_pack_tensor(&mut gf8);
        f8g.extend_from_slice(&{
            let mut b = Vec::with_capacity(8 + gc8.len() + 4);
            b.extend_from_slice(&(gc8.len() as u64).to_le_bytes());
            b.extend_from_slice(&gc8);
            b.extend_from_slice(&gs8.to_le_bytes());
            b
        });
        let mut uf8 = up_f[gs..gs + INTER * HIDDEN].to_vec();
        let (uc8, us8) = fp8_pack_tensor(&mut uf8);
        f8u.extend_from_slice(&{
            let mut b = Vec::with_capacity(8 + uc8.len() + 4);
            b.extend_from_slice(&(uc8.len() as u64).to_le_bytes());
            b.extend_from_slice(&uc8);
            b.extend_from_slice(&us8.to_le_bytes());
            b
        });
        let mut df8 = down_f[ds..ds + HIDDEN * INTER].to_vec();
        let (dc8, ds8) = fp8_pack_tensor(&mut df8);
        f8d.extend_from_slice(&{
            let mut b = Vec::with_capacity(8 + dc8.len() + 4);
            b.extend_from_slice(&(dc8.len() as u64).to_le_bytes());
            b.extend_from_slice(&dc8);
            b.extend_from_slice(&ds8.to_le_bytes());
            b
        });

        let mut ga = gate_f[gs..gs + INTER * HIDDEN].to_vec();
        aqg.extend_from_slice(&awq4_pack_rows(&mut ga, INTER, HIDDEN, 8));
        let mut ua = up_f[gs..gs + INTER * HIDDEN].to_vec();
        aqu.extend_from_slice(&awq4_pack_rows(&mut ua, INTER, HIDDEN, 8));
        let mut da = down_f[ds..ds + HIDDEN * INTER].to_vec();
        aqd.extend_from_slice(&awq4_pack_rows(&mut da, HIDDEN, INTER, 8));

        let mut gm = gate_f[gs..gs + INTER * HIDDEN].to_vec();
        let (gc4, ge4) = mxfp4_pack_tensor(&mut gm);
        m4cg.extend_from_slice(&gc4);
        m4eg.extend_from_slice(&ge4);
        let mut um = up_f[gs..gs + INTER * HIDDEN].to_vec();
        let (uc4, ue4) = mxfp4_pack_tensor(&mut um);
        m4cu.extend_from_slice(&uc4);
        m4eu.extend_from_slice(&ue4);
        let mut dm = down_f[ds..ds + HIDDEN * INTER].to_vec();
        let (dc4, de4) = mxfp4_pack_tensor(&mut dm);
        m4cd.extend_from_slice(&dc4);
        m4ed.extend_from_slice(&de4);
    }

    let pack_i8 = || DType { arith: grim_tensor::ArithType::F32, storage: Storage::CompressedTensorsW8A8Int8 };
    let pack_f8 = || DType { arith: grim_tensor::ArithType::F32, storage: Storage::CompressedTensorsW8A8Fp8 };
    let pack_aq = || DType {
        arith: grim_tensor::ArithType::F32,
        storage: Storage::Awq(grim_tensor::dtype::AwqStorageConfig { bits: 4, group_size: 8 }),
    };
    let pack_m4 = || DType {
        arith: grim_tensor::ArithType::F32,
        storage: Storage::FloatPack(grim_tensor::FloatPackScheme::MxFp4),
    };
    let up_b = |v: &[u8], dt: DType| {
        MemoryOps::from_cpu_bytes(&dev, v, &Shape::new(vec![v.len()]), dt).unwrap()
    };
    let (bi8g, bi8u, bi8d) = (up_b(&i8g, pack_i8()), up_b(&i8u, pack_i8()), up_b(&i8d, pack_i8()));
    let (bf8g, bf8u, bf8d) = (up_b(&f8g, pack_f8()), up_b(&f8u, pack_f8()), up_b(&f8d, pack_f8()));
    let (baqg, baqu, baqd) = (up_b(&aqg, pack_aq()), up_b(&aqu, pack_aq()), up_b(&aqd, pack_aq()));
    let (bm4cg, bm4cu, bm4cd) = (up_b(&m4cg, pack_m4()), up_b(&m4cu, pack_m4()), up_b(&m4cd, pack_m4()));
    let (bm4eg, bm4eu, bm4ed) = (up_b(&m4eg, pack_m4()), up_b(&m4eu, pack_m4()), up_b(&m4ed, pack_m4()));
    // f32-dequant twin stacks (exact values already in gate_f/up_f/down_f? No:
    // per-format rewrites touched only copies. Build f32 stacks from gate_f
    // as-is (unquantized) — bench compares traffic, not numerics.
    let f32_of = |v: &[f32]| dev.from_cpu(v, &Shape::new(vec![v.len()]), DType::F32).unwrap();
    let (ffg, ffu, ffd) = (f32_of(&gate_f), f32_of(&up_f), f32_of(&down_f));

    let logits: Vec<f32> = (0..SEQ * EXPERTS)
        .map(|i| ((i as f32 + 1.0) * 0.6).sin() * 2.5)
        .collect();
    let act: Vec<f32> = (0..SEQ * HIDDEN).map(|i| ((i as f32 + 1.0) * 0.2).cos()).collect();
    let act_st = dev.from_cpu(&act, &Shape::new(vec![SEQ, HIDDEN]), DType::F32).unwrap();
    let act_rocm = act_st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
    let (tok_b, exp_b, w_b) = route_on_device(&dev, &logits, SEQ, EXPERTS, TOPK, 0);
    let tok_rocm = tok_b.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
    let exp_rocm = exp_b.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
    let w_rocm = w_b.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
    let ascale = vec![1.0f32; SEQ];
    let ascale_st = dev.from_cpu(&ascale, &Shape::new(vec![SEQ]), DType::F32).unwrap();
    let ascale_rocm = ascale_st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>().unwrap();
    let out_shape = Shape::new(vec![SEQ, HIDDEN]);
    let num_pairs = SEQ * TOPK;

    let time_it = |f: &mut dyn FnMut()| -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..ITERS {
            let start = std::time::Instant::now();
            f();
            dev.synchronize();
            best = best.min(start.elapsed().as_secs_f64() * 1e3);
        }
        best
    };
    let t_f32 = time_it(&mut || {
        dev.moe_fused_dispatch_resident_routing(
            act_rocm, &*ffg, &*ffu, &*ffd, tok_rocm, exp_rocm, w_rocm,
            num_pairs, &out_shape, HIDDEN, INTER, 1.0,
        )
        .unwrap();
    });
    let t_i8 = time_it(&mut || {
        dev.moe_fused_dispatch_resident_routing_w8a8_int8(
            act_rocm, &*bi8g, &*bi8u, &*bi8d, ascale_rocm, tok_rocm, exp_rocm, w_rocm,
            num_pairs, &out_shape, HIDDEN, INTER, 1.0,
        )
        .unwrap();
    });
    let t_f8 = time_it(&mut || {
        dev.moe_fused_dispatch_resident_routing_w8a8_fp8(
            act_rocm, &*bf8g, &*bf8u, &*bf8d, ascale_rocm, tok_rocm, exp_rocm, w_rocm,
            num_pairs, &out_shape, HIDDEN, INTER, 1.0,
        )
        .unwrap();
    });
    let t_awq = time_it(&mut || {
        dev.moe_fused_dispatch_resident_routing_awq(
            act_rocm, &*baqg, &*baqu, &*baqd, ascale_rocm, tok_rocm, exp_rocm, w_rocm,
            num_pairs, &out_shape, HIDDEN, INTER, 4, 8, 1.0,
        )
        .unwrap();
    });
    let t_mx = time_it(&mut || {
        dev.moe_fused_dispatch_resident_routing_mxfp4(
            act_rocm, &*bm4cg, &*bm4cu, &*bm4cd, &*bm4eg, &*bm4eu, &*bm4ed,
            ascale_rocm, tok_rocm, exp_rocm, w_rocm, num_pairs, &out_shape,
            HIDDEN, INTER, 1.0,
        )
        .unwrap();
    });
    let t_dot4 = time_it(&mut || {
        dev.moe_fused_dispatch_resident_routing_w8a8_int8_dot4(
            act_rocm, &*bi8g, &*bi8u, &*bi8d, ascale_rocm,
            tok_rocm, exp_rocm, w_rocm, num_pairs, &out_shape,
            HIDDEN, INTER, 1.0,
        )
        .unwrap();
    });
    eprintln!(
        "[bench] h={HIDDEN} i={INTER} e={EXPERTS} topk={TOPK} seq={SEQ} best-of-{ITERS} (ms): f32={t_f32:.2} int8={t_i8:.2} fp8={t_f8:.2} awq={t_awq:.2} mxfp4={t_mx:.2} int8_dot4={t_dot4:.2}"
    );
}