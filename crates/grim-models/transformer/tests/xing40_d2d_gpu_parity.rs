//! CPU-host vs ROCm-device parity for Xing4.0's device-resident paths.
//!
//! The hyper-connection is the part of Xing4.0 that carries a *multi-stream*
//! residual state, and the part with the most ways to accidentally round-trip
//! through the host: the gate projection, the fused sigmoid + Sinkhorn gate
//! math, the stream collapse, the `post ⊗ y + comb @ streams` write-back, plus
//! the model's stream seeding and mean collapse at the two ends.
//!
//! The host reference and the D2D path are separate code, so this test builds
//! one tiny synthetic model, runs it on `Device::Cpu` (host reference) and on
//! `Device::Rocm(0)` (D2D), and compares the logits.

use grim_core::model::CausalLm;
use grim_models_transformer::{Xing40, Xing40Config};
use grim_backend_cpu::cpu_tensor;
use grim_format::tprov::SafetensorsProvider;
use grim_nn::WeightSource;
use grim_tensor::{Device, Shape};
use std::collections::BTreeMap;
use std::path::Path;

const HIDDEN: usize = 64;
const VOCAB: usize = 128;
const HEADS: usize = 4;
const KV_RANK: usize = 16;
const Q_RANK: usize = 16;
const NOPE: usize = 8;
const ROPE_D: usize = 8;
const V_HEAD: usize = 8;
const INTER: usize = 32;
const HC: usize = 4;
const MIX: usize = (2 + HC) * HC;
const LAYERS: usize = 2;

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Deterministic pseudo-random weights — the MHC gates are only interesting if
/// the projection is not a constant, so every tensor gets its own values.
fn pseudo_random(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            // [-0.5, 0.5) — small enough to keep a 2-layer forward finite.
            ((s >> 9) as f32 / 8_388_608.0) - 0.5
        })
        .collect()
}

fn write_safetensors(path: &Path, t: &BTreeMap<String, (Vec<usize>, Vec<f32>)>) {
    let mut header = String::from("{");
    let mut body: Vec<u8> = Vec::new();
    let mut first = true;
    for (name, (shape, vals)) in t {
        if !first {
            header.push(',');
        }
        first = false;
        let start = body.len();
        body.extend_from_slice(&f32_bytes(vals));
        let end = body.len();
        let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{start},{end}]}}",
            dims.join(",")
        ));
    }
    header.push('}');
    let mut out = Vec::new();
    out.extend_from_slice(&(header.as_bytes().len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&body);
    std::fs::write(path, out).expect("write synthetic safetensors");
}

fn config() -> Xing40Config {
    Xing40Config {
        vocab_size: VOCAB,
        hidden_size: HIDDEN,
        num_heads: HEADS,
        num_kv_heads: HEADS,
        head_dim: ROPE_D,
        num_layers: LAYERS,
        intermediate_size: INTER,
        kv_lora_rank: KV_RANK,
        q_lora_rank: Some(Q_RANK),
        qk_nope_head_dim: NOPE,
        qk_rope_head_dim: ROPE_D,
        v_head_dim: V_HEAD,
        moe_intermediate_size: INTER,
        n_routed_experts: 4,
        n_shared_experts: 1,
        num_experts_per_tok: 2,
        // All-dense: this test targets the hyper-connection and the model ends,
        // so keep the MoE router out of the comparison.
        first_k_dense_replace: LAYERS,
        routed_scaling_factor: 2.0,
        noaux_tc_routing: true,
        hc_mult: HC,
        ..Xing40Config::default()
    }
}

fn build(dir: &Path) -> Xing40Config {
    let cfg = config();
    let mut t: BTreeMap<String, (Vec<usize>, Vec<f32>)> = BTreeMap::new();
    let mut seed = 7u32;
    let mut add = |t: &mut BTreeMap<String, (Vec<usize>, Vec<f32>)>,
                   name: String,
                   shape: Vec<usize>,
                   fill: Option<f32>| {
        let n: usize = shape.iter().product();
        let vals = match fill {
            Some(f) => vec![f; n],
            None => {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                pseudo_random(n, seed)
            }
        };
        t.insert(name, (shape, vals));
    };

    add(&mut t, "model.embed_tokens.weight".into(), vec![VOCAB, HIDDEN], None);
    add(&mut t, "lm_head.weight".into(), vec![VOCAB, HIDDEN], None);
    add(&mut t, "model.norm.weight".into(), vec![HIDDEN], Some(1.0));

    for l in 0..cfg.num_layers {
        let p = format!("model.layers.{l}");
        add(&mut t, format!("{p}.input_layernorm.weight"), vec![HIDDEN], Some(1.0));
        add(
            &mut t,
            format!("{p}.post_attention_layernorm.weight"),
            vec![HIDDEN],
            Some(1.0),
        );
        add(
            &mut t,
            format!("{p}.self_attn.q_a_layernorm.weight"),
            vec![Q_RANK],
            Some(1.0),
        );
        add(
            &mut t,
            format!("{p}.self_attn.kv_a_layernorm.weight"),
            vec![KV_RANK],
            Some(1.0),
        );
        // The hyper-connection parameters are excluded from quantization in the
        // real export; keep them random and unquantized so the gate math is
        // actually exercised rather than collapsing to a constant.
        for hc in ["attn_hc", "ffn_hc"] {
            add(&mut t, format!("{p}.{hc}.hc_fn"), vec![MIX, HIDDEN * HC], None);
            add(&mut t, format!("{p}.{hc}.hc_base"), vec![MIX], None);
            add(&mut t, format!("{p}.{hc}.hc_scale"), vec![3], None);
        }

        add(&mut t, format!("{p}.self_attn.q_a_proj.weight"), vec![Q_RANK, HIDDEN], None);
        add(
            &mut t,
            format!("{p}.self_attn.q_b_proj.weight"),
            vec![HEADS * (NOPE + ROPE_D), Q_RANK],
            None,
        );
        add(
            &mut t,
            format!("{p}.self_attn.kv_a_proj_with_mqa.weight"),
            vec![KV_RANK + ROPE_D, HIDDEN],
            None,
        );
        add(
            &mut t,
            format!("{p}.self_attn.kv_b_proj.weight"),
            vec![HEADS * (NOPE + V_HEAD), KV_RANK],
            None,
        );
        add(
            &mut t,
            format!("{p}.self_attn.o_proj.weight"),
            vec![HIDDEN, HEADS * V_HEAD],
            None,
        );

        add(&mut t, format!("{p}.mlp.gate_proj.weight"), vec![INTER, HIDDEN], None);
        add(&mut t, format!("{p}.mlp.up_proj.weight"), vec![INTER, HIDDEN], None);
        add(&mut t, format!("{p}.mlp.down_proj.weight"), vec![HIDDEN, INTER], None);
    }

    write_safetensors(&dir.join("model.safetensors"), &t);
    cfg
}

fn load(dir: &Path, cfg: Xing40Config, device: Device) -> Xing40 {
    let provider = SafetensorsProvider::open(
        dir.join("model.safetensors").to_str().unwrap(),
    )
    .expect("open safetensors");
    let ws = WeightSource::root(&provider, device.clone());
    Xing40::load(device, &ws, cfg).expect("load Xing4.0")
}

fn forward_logits(model: &Xing40, ids: &[f32], device: &Device) -> Vec<f32> {
    let n = ids.len();
    let dev = grim_nn::pick_device_for_storage_device(device);
    let mk = |v: &[f32], d: &[usize]| {
        grim_tensor::Tensor::new(
            std::sync::Arc::from(dev.from_cpu(v, &Shape::new(d.to_vec()), grim_tensor::DType::F32).unwrap()),
            Shape::new(d.to_vec()),
            grim_tensor::DType::F32,
            grim_tensor::QuantProvenance::default(),
            device.clone(),
        )
    };
    let input = mk(ids, &[n]);
    let pos: Vec<f32> = (0..n as u32).map(|p| p as f32).collect();
    let pos_t = mk(&pos, &[n]);
    let mut session = model.new_session();
    model
        .forward(session.as_mut(), &input, &pos_t, &[])
        .expect("Xing4.0 forward")
        .to_vec_f32()
        .expect("logits to f32")
}

/// The D2D hyper-connection path must reproduce the host reference.
///
/// The whole prefill forward: multi-stream hyper-connection, latent-absorbed
/// MLA prefill, dense SwiGLU, and the model's stream seed / mean collapse — all
/// device-resident — must reproduce the host reference.
///
/// Sequence lengths 1, 3 and 7 cover the decode fast path, a short prefill, and
/// a prefill whose length is not a multiple of the launch block.
#[test]
#[ignore = "requires a visible ROCm device"]
fn device_hyper_connection_matches_host_reference() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cfg = build(tmp.path());

    // Several sequence lengths: 1 exercises the decode shape, longer ones the
    // prefill shape, and a non-256-multiple length proves the grid is derived
    // from the actual row count.
    for seq in [1usize, 3, 7, 33] {
        let ids: Vec<f32> = (0..seq).map(|i| ((i * 13 + 5) % VOCAB) as f32).collect();

        let cpu = load(tmp.path(), cfg.clone(), Device::Cpu);
        let host = forward_logits(&cpu, &ids, &Device::Cpu);
        let gpu_model = load(tmp.path(), cfg.clone(), Device::Rocm(0));
        let device = forward_logits(&gpu_model, &ids, &Device::Rocm(0));

        assert_eq!(host.len(), device.len(), "seq={seq}: logit count");
        let mut worst = 0.0f32;
        let mut worst_at = 0usize;
        for (i, (h, d)) in host.iter().zip(device.iter()).enumerate() {
            let delta = (h - d).abs();
            if delta > worst {
                worst = delta;
                worst_at = i;
            }
        }
        println!(
            "seq={seq}: {} logits, max |host - device| = {worst:e} (at {worst_at}: {} vs {})",
            host.len(),
            host[worst_at],
            device[worst_at]
        );
        assert!(
            worst < 2e-3,
            "seq={seq}: D2D hyper-connection diverges from the host reference by {worst:e}"
        );
    }
}

/// `Device::Rocm(0)` must resolve to a device that implements the
/// hyper-connection primitives through the `Arc<dyn BackendDevice>` blanket impl.
///
/// This guards a real trap: `impl ElementwiseOps for Arc<T>` forwards each method
/// explicitly, so a new trait method that is added to the trait and to a backend
/// but not to that forwarding block silently resolves to the trait's
/// `Unimplemented` default even though the concrete device implements it.
#[test]
#[ignore = "requires a visible ROCm device"]
fn rocm_device_exposes_hc_primitives() {
    use grim_tensor::Shape;
    let dev = grim_nn::pick_device_for_storage_device(&Device::Rocm(0));
    let src = dev.from_cpu(&[1.0f32, 2.0, 3.0, 4.0], &Shape::new(vec![2, 2]), grim_tensor::DType::F32).unwrap();
    let mut dst = dev.from_cpu(&[9.0f32; 4], &Shape::new(vec![2, 2]), grim_tensor::DType::F32).unwrap();
    let h = dev
        .write_cols(dst.as_mut(), 2, 0, src.as_ref(), 2, 2)
        .expect("write_cols on the picked ROCm device");
    h.synchronize().unwrap();
    println!("picked device: {:?}", std::any::type_name::<dyn grim_tensor::BackendDevice>());
    println!("write_cols ok -> {:?}", dst.to_cpu_vec_f32().unwrap());
}

/// The D2D per-head W_VC up-projection loop must equal the host reference.
///
/// `matmul` on ROCm takes its right operand in natural weight `[N, K]` layout and
/// computes `A @ Bᵀ`, so a per-head `[v_head, rank]` block feeds a
/// `[seq, rank]` latent directly. This is the prefill-only half of the attention
/// step (decode uses the in-kernel projection), so it needs its own check.
#[test]
#[ignore = "requires a visible ROCm device"]
fn per_head_v_up_projection_loop_matches_host() {
    use grim_tensor::{CoreTensorOps, DType, ElementwiseOps, Shape, Tensor};
    let (nh, vd, rank, seq) = (4usize, 8usize, 16usize, 5usize);
    let dev = grim_nn::pick_device_for_storage_device(&Device::Rocm(0));

    let latent: Vec<f32> = (0..seq * nh * rank)
        .map(|i| ((i % 31) as f32) * 0.01 - 0.15)
        .collect();
    let w_vc: Vec<f32> = (0..nh * vd * rank)
        .map(|i| ((i % 17) as f32) * 0.01 - 0.08)
        .collect();

    // Host reference: attn[s, h, d] = sum_c latent[s, h, c] * w_vc[h, d, c]
    let mut expected = vec![0.0f32; seq * nh * vd];
    for s in 0..seq {
        for h in 0..nh {
            for d in 0..vd {
                let mut acc = 0.0f32;
                for c in 0..rank {
                    acc += latent[(s * nh + h) * rank + c] * w_vc[(h * vd + d) * rank + c];
                }
                expected[(s * nh + h) * vd + d] = acc;
            }
        }
    }

    let latent_t = Tensor::new(
        std::sync::Arc::from(
            dev.from_cpu(&latent, &Shape::new(vec![seq, nh * rank]), DType::F32).unwrap(),
        ),
        Shape::new(vec![seq, nh * rank]),
        DType::F32,
        grim_tensor::QuantProvenance::default(),
        Device::Rocm(0),
    );
    let w_t = dev
        .from_cpu(&w_vc, &Shape::new(vec![nh, vd, rank]), DType::F32)
        .unwrap();

    let attn_shape = Shape::new(vec![seq, nh * vd]);
    let mut attn = dev.zeros(&attn_shape, DType::F32).unwrap();
    for h in 0..nh {
        let latent_h = dev
            .narrow_cols(
                latent_t.storage().as_ref(),
                nh * rank,
                h * rank,
                seq,
                rank,
                &Shape::new(vec![seq, rank]),
            )
            .unwrap()
            .0;
        let w_h = dev
            .narrow_rows(
                w_t.as_ref(),
                h * vd,
                vd,
                rank,
                &Shape::new(vec![vd, rank]),
            )
            .unwrap()
            .0;
        let projected = dev
            .matmul(latent_h.as_ref(), w_h.as_ref(), &Shape::new(vec![seq, vd]))
            .unwrap()
            .0;
        dev.write_cols(attn.as_mut(), nh * vd, h * vd, projected.as_ref(), seq, vd)
            .unwrap();
    }
    grim_backend_rocm::RocmDevice::shared(0).synchronize();

    let got = attn.to_cpu_vec_f32().unwrap();
    let mut worst = 0.0f32;
    for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
        worst = worst.max((a - b).abs());
        assert!((a - b).abs() < 1e-5, "V-projection[{i}]: {a} vs {b}");
    }
    println!("per-head W_VC loop: max abs delta = {worst:e}");
}

/// The multi-stream hyper-connection must match its host reference for
/// **multi-token** sequences, not just decode.
///
/// This was the one stage never validated above seq == 1: the MHC carries a
/// `[seq, hc * hidden]` stream state, and every stage of it (gate projection,
/// collapse, `post ⊗ y + comb @ streams` write-back) is shaped on `seq`. A
/// seq-dependent indexing mistake there would pass a single-token test and
/// corrupt every prefill.
#[test]
#[ignore = "requires a visible ROCm device"]
fn hyper_connection_matches_host_for_multi_token() {
    use grim_models_transformer::Xing40HyperConnection;
    use grim_tensor::{CoreTensorOps, DType, QuantProvenance, Shape, Tensor};

    let tmp = tempfile::tempdir().expect("tempdir");
    let (hidden, hc) = (64usize, 4usize);
    let mix = (2 + hc) * hc;
    let flat = hidden * hc;
    let cfg = Xing40Config {
        hidden_size: hidden,
        hc_mult: hc,
        hc_sinkhorn_iters: 20,
        hc_eps: 1e-6,
        mhc_h_res_clamp_min: -30.0,
        mhc_h_res_clamp_max: 30.0,
        ..Xing40Config::default()
    };

    // Build a tiny safetensors carrying only the two HC parameter sets.
    let mut t: BTreeMap<String, (Vec<usize>, Vec<f32>)> = BTreeMap::new();
    let mut seed = 991u32;
    for hc_tag in ["attn", "ffn"] {
        for (name, shape) in [
            (format!("{hc_tag}_hc.hc_fn"), vec![mix, flat]),
            (format!("{hc_tag}_hc.hc_base"), vec![mix]),
            (format!("{hc_tag}_hc.hc_scale"), vec![3]),
        ] {
            let n: usize = shape.iter().product();
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            t.insert(
                name,
                (shape, pseudo_random(n, seed).iter().map(|v| v * 0.5).collect()),
            );
        }
    }
    write_safetensors(&tmp.path().join("hc.safetensors"), &t);

    for seq in [1usize, 2, 3, 7, 8] {
        let provider = SafetensorsProvider::open(
            tmp.path().join("hc.safetensors").to_str().unwrap(),
        )
        .expect("open hc safetensors");

        let ws_cpu = WeightSource::root(&provider, Device::Cpu);
        let cpu = Xing40HyperConnection::load(&ws_cpu, &cfg, "attn").expect("hc cpu");
        let ws_gpu = WeightSource::root(&provider, Device::Rocm(0));
        let gpu = Xing40HyperConnection::load(&ws_gpu, &cfg, "attn").expect("hc gpu");

        // Random token-major stream state [seq, hc * hidden] and block output.
        let mut sseed = 4242u32;
        let streams_v = pseudo_random(seq * flat, sseed);
        sseed = 99;
        let y_v = pseudo_random(seq * hidden, sseed);

        // --- host reference ---
        let streams_cpu = cpu_tensor(streams_v.clone(), Shape::new(vec![seq, flat]));
        let (gates, collapsed) = cpu.forward(&streams_cpu, seq).expect("host forward");
        let next = cpu.write_back(&streams_v, &y_v, seq, &gates);

        // --- device ---
        let dev = grim_nn::pick_device_for_storage_device(&Device::Rocm(0));
        let streams = Tensor::new(
            std::sync::Arc::from(
                dev.from_cpu(&streams_v, &Shape::new(vec![seq, flat]), DType::F32).unwrap(),
            ),
            Shape::new(vec![seq, flat]),
            DType::F32,
            QuantProvenance::default(),
            Device::Rocm(0),
        );
        let y = Tensor::new(
            std::sync::Arc::from(
                dev.from_cpu(&y_v, &Shape::new(vec![seq, hidden]), DType::F32).unwrap(),
            ),
            Shape::new(vec![seq, hidden]),
            DType::F32,
            QuantProvenance::default(),
            Device::Rocm(0),
        );

        let g = gpu.gates_d2d(&streams).expect("gates_d2d");
        let collapsed_d2d = gpu.collapse_d2d(&streams, &g).expect("collapse_d2d");
        let next_d2d = gpu.update_d2d(&streams, &y, &g).expect("update_d2d");
        grim_backend_rocm::RocmDevice::shared(0).synchronize();

        // gates: compare the device tensors against the host gate vectors
        let pre = g.pre.to_cpu_vec_f32().unwrap(); // [hc, seq]
        let post = g.post.to_cpu_vec_f32().unwrap();
        let comb = g.comb.to_cpu_vec_f32().unwrap(); // [hc*hc, seq]
        let mut worst_gate = 0.0f32;
        for h in 0..hc {
            for s in 0..seq {
                worst_gate = worst_gate.max((pre[h * seq + s] - gates.pre[s * hc + h]).abs());
                worst_gate = worst_gate.max((post[h * seq + s] - gates.post[s * hc + h]).abs());
            }
        }
        for h in 0..hc * hc {
            for s in 0..seq {
                worst_gate =
                    worst_gate.max((comb[h * seq + s] - gates.comb[s * hc * hc + h]).abs());
            }
        }

        let got_c = collapsed_d2d.to_vec_f32().unwrap();
        let got_n = next_d2d.to_vec_f32().unwrap();
        let mut worst_c = 0.0f32;
        for (a, b) in got_c.iter().zip(collapsed.iter()) {
            worst_c = worst_c.max((a - b).abs());
        }
        let mut worst_n = 0.0f32;
        for (a, b) in got_n.iter().zip(next.iter()) {
            worst_n = worst_n.max((a - b).abs());
        }
        println!(
            "seq={seq}: gates {worst_gate:e}, collapse {worst_c:e}, write-back {worst_n:e}"
        );
        assert!(worst_gate < 1e-5, "seq={seq}: gates diverge by {worst_gate:e}");
        assert!(worst_c < 1e-5, "seq={seq}: collapse diverges by {worst_c:e}");
        assert!(worst_n < 1e-5, "seq={seq}: write-back diverges by {worst_n:e}");
    }
}
