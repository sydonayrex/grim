//! End-to-end verification that the Xing4.0 model type loads from a
//! block-FP8 safetensors checkpoint and runs a forward pass.
//!
//! The synthetic checkpoint mirrors the real `Xing4.0-29B-A4B-FP8` export
//! layout exactly: F8_E4M3 projection weights paired with `weight_scale_inv`
//! F32 siblings (DeepSeek-style block scales), plus the *unquantized* tensors
//! listed in that export's `modules_to_not_convert`: both RMSNorms, the two
//! hyper-connection modules (`attn_hc` / `ffn_hc`), the MoE router gate and
//! its `e_score_correction_bias`.
//!
//! Runs on `Device::Cpu`, so the test needs no GPU.

use grim_format::tprov::SafetensorsProvider;
use grim_models_transformer::{Xing40, Xing40Config};
use grim_nn::WeightSource;
use grim_tensor::{Device, Shape};
use std::collections::BTreeMap;
use std::path::Path;

/// A synthetic block-FP8 projection: E4M3 codes plus its `weight_scale_inv` grid.
struct Fp8Tensor {
    codes: Vec<u8>,
    scales: Vec<f32>,
    rows: usize,
    cols: usize,
}

/// 0x38 = +1.0, 0xB8 = -1.0 in E4M3 (sign | exp 0111 | mant 000).
fn code_sign(v: f32) -> u8 {
    if v >= 0.0 {
        0x38
    } else {
        0xB8
    }
}

fn fp8_tensor(rows: usize, cols: usize, seed: u32) -> Fp8Tensor {
    let mut codes = Vec::with_capacity(rows * cols);
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(12_345);
    for _ in 0..rows * cols {
        // xorshift: varied but deterministic sign pattern.
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let unit = ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
        codes.push(code_sign(unit));
    }
    let scale_rows = rows.div_ceil(128).max(1);
    let scale_cols = cols.div_ceil(128).max(1);
    // Non-unit scales: a dropped or misapplied block scale changes the
    // dequantized weights and would show up in the forward assertions.
    let scales = (0..scale_rows * scale_cols)
        .map(|i| 0.5 + (i as f32) * 0.25)
        .collect();
    Fp8Tensor {
        codes,
        scales,
        rows,
        cols,
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn write_safetensors(
    path: &Path,
    fp8: &BTreeMap<String, Fp8Tensor>,
    native: &BTreeMap<String, (Vec<usize>, Vec<f32>)>,
) {
    let mut header = String::from("{");
    let mut body: Vec<u8> = Vec::new();
    let mut first = true;
    for (name, t) in fp8 {
        if !first {
            header.push(',');
        }
        first = false;
        let start = body.len();
        body.extend_from_slice(&t.codes);
        let end = body.len();
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"F8_E4M3\",\"shape\":[{},{}],\"data_offsets\":[{start},{end}]}}",
            t.rows, t.cols
        ));
        // DeepSeek-style: `<base>.weight` pairs with `<base>.weight_scale_inv`.
        let scale_name = format!("{name}_scale_inv");
        let s0 = body.len();
        body.extend_from_slice(&f32_bytes(&t.scales));
        let s1 = body.len();
        let grid_rows = t.rows.div_ceil(128).max(1);
        let grid_cols = t.cols.div_ceil(128).max(1);
        header.push_str(&format!(
            ",\"{scale_name}\":{{\"dtype\":\"F32\",\"shape\":[{grid_rows},{grid_cols}],\"data_offsets\":[{s0},{s1}]}}"
        ));
    }
    for (name, (shape, vals)) in native {
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

    let header_bytes = header.as_bytes();
    let mut out = Vec::new();
    out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(header_bytes);
    out.extend_from_slice(&body);
    std::fs::write(path, out).expect("write synthetic safetensors");
}

/// Tiny Xing4.0 shape — structurally identical to the 29B-A4B release, scaled down.
const HIDDEN: usize = 64;
const VOCAB: usize = 128;
const HEADS: usize = 4;
const KV_RANK: usize = 16;
const Q_RANK: usize = 16;
const NOPE: usize = 8;
const ROPE_D: usize = 8;
const V_HEAD: usize = 8;
const INTER: usize = 32;
const MOE_INTER: usize = 16;
const EXPERTS: usize = 4;
const TOP_K: usize = 2;
const HC: usize = 4;
const MIX: usize = (2 + HC) * HC;

fn tiny_config(first_k_dense_replace: usize) -> Xing40Config {
    Xing40Config {
        vocab_size: VOCAB,
        hidden_size: HIDDEN,
        num_heads: HEADS,
        num_kv_heads: HEADS,
        head_dim: ROPE_D,
        num_layers: first_k_dense_replace + 1,
        intermediate_size: INTER,
        kv_lora_rank: KV_RANK,
        q_lora_rank: Some(Q_RANK),
        qk_nope_head_dim: NOPE,
        qk_rope_head_dim: ROPE_D,
        v_head_dim: V_HEAD,
        moe_intermediate_size: MOE_INTER,
        n_routed_experts: EXPERTS,
        n_shared_experts: 1,
        num_experts_per_tok: TOP_K,
        first_k_dense_replace,
        routed_scaling_factor: 2.0,
        noaux_tc_routing: true,
        hc_mult: HC,
        ..Xing40Config::default()
    }
}

fn build(dir: &Path, first_k_dense_replace: usize) -> Xing40Config {
    let cfg = tiny_config(first_k_dense_replace);
    let mut fp8: BTreeMap<String, Fp8Tensor> = BTreeMap::new();
    let mut native: BTreeMap<String, (Vec<usize>, Vec<f32>)> = BTreeMap::new();
    let mut seed = 7u32;
    let mut add_fp8 = |fp8: &mut BTreeMap<String, Fp8Tensor>, name: &str, rows: usize, cols: usize| {
        seed = seed.wrapping_add(101);
        fp8.insert(name.to_string(), fp8_tensor(rows, cols, seed));
    };
    let add_f32 = |native: &mut BTreeMap<String, (Vec<usize>, Vec<f32>)>,
                   name: &str,
                   shape: Vec<usize>,
                   fill: f32| {
        let n: usize = shape.iter().product();
        native.insert(name.to_string(), (shape, vec![fill; n]));
    };

    add_f32(&mut native, "model.embed_tokens.weight", vec![VOCAB, HIDDEN], 0.05);
    // `lm_head.weight` is `[vocab, hidden]` in both containers.
    add_f32(&mut native, "lm_head.weight", vec![VOCAB, HIDDEN], 0.05);
    add_f32(&mut native, "model.norm.weight", vec![HIDDEN], 1.0);

    for l in 0..cfg.num_layers {
        let p = format!("model.layers.{l}");
        add_f32(&mut native, &format!("{p}.input_layernorm.weight"), vec![HIDDEN], 1.0);
        add_f32(
            &mut native,
            &format!("{p}.post_attention_layernorm.weight"),
            vec![HIDDEN],
            1.0,
        );
        add_f32(
            &mut native,
            &format!("{p}.self_attn.q_a_layernorm.weight"),
            vec![Q_RANK],
            1.0,
        );
        add_f32(
            &mut native,
            &format!("{p}.self_attn.kv_a_layernorm.weight"),
            vec![KV_RANK],
            1.0,
        );
        for hc in ["attn_hc", "ffn_hc"] {
            add_f32(
                &mut native,
                &format!("{p}.{hc}.hc_fn"),
                vec![MIX, HIDDEN * HC],
                0.02,
            );
            add_f32(&mut native, &format!("{p}.{hc}.hc_base"), vec![MIX], 0.1);
            add_f32(&mut native, &format!("{p}.{hc}.hc_scale"), vec![3], 1.0);
        }

        add_fp8(&mut fp8, &format!("{p}.self_attn.q_a_proj.weight"), Q_RANK, HIDDEN);
        add_fp8(
            &mut fp8,
            &format!("{p}.self_attn.q_b_proj.weight"),
            HEADS * (NOPE + ROPE_D),
            Q_RANK,
        );
        add_fp8(
            &mut fp8,
            &format!("{p}.self_attn.kv_a_proj_with_mqa.weight"),
            KV_RANK + ROPE_D,
            HIDDEN,
        );
        add_fp8(
            &mut fp8,
            &format!("{p}.self_attn.kv_b_proj.weight"),
            HEADS * (NOPE + V_HEAD),
            KV_RANK,
        );
        add_fp8(
            &mut fp8,
            &format!("{p}.self_attn.o_proj.weight"),
            HIDDEN,
            HEADS * V_HEAD,
        );

        if l < first_k_dense_replace {
            add_fp8(&mut fp8, &format!("{p}.mlp.gate_proj.weight"), INTER, HIDDEN);
            add_fp8(&mut fp8, &format!("{p}.mlp.up_proj.weight"), INTER, HIDDEN);
            add_fp8(&mut fp8, &format!("{p}.mlp.down_proj.weight"), HIDDEN, INTER);
        } else {
            // Router gate + noaux_tc bias stay unquantized.
            add_f32(
                &mut native,
                &format!("{p}.mlp.gate.weight"),
                vec![EXPERTS, HIDDEN],
                0.03,
            );
            add_f32(
                &mut native,
                &format!("{p}.mlp.gate.e_score_correction_bias"),
                vec![EXPERTS],
                0.0,
            );
            for e in 0..EXPERTS {
                add_fp8(
                    &mut fp8,
                    &format!("{p}.mlp.experts.{e}.gate_proj.weight"),
                    MOE_INTER,
                    HIDDEN,
                );
                add_fp8(
                    &mut fp8,
                    &format!("{p}.mlp.experts.{e}.up_proj.weight"),
                    MOE_INTER,
                    HIDDEN,
                );
                add_fp8(
                    &mut fp8,
                    &format!("{p}.mlp.experts.{e}.down_proj.weight"),
                    HIDDEN,
                    MOE_INTER,
                );
            }
            add_fp8(
                &mut fp8,
                &format!("{p}.mlp.shared_experts.gate_proj.weight"),
                MOE_INTER,
                HIDDEN,
            );
            add_fp8(
                &mut fp8,
                &format!("{p}.mlp.shared_experts.up_proj.weight"),
                MOE_INTER,
                HIDDEN,
            );
            add_fp8(
                &mut fp8,
                &format!("{p}.mlp.shared_experts.down_proj.weight"),
                HIDDEN,
                MOE_INTER,
            );
        }
    }

    write_safetensors(&dir.join("model.safetensors"), &fp8, &native);
    cfg
}

fn load(dir: &Path, cfg: Xing40Config) -> Xing40 {
    let provider = SafetensorsProvider::open(dir.join("model.safetensors").to_str().unwrap())
        .expect("open safetensors");
    let ws = WeightSource::root(&provider, Device::Cpu);
    Xing40::load(Device::Cpu, &ws, cfg).expect("load Xing4.0 block-FP8 safetensors")
}

fn forward_logits(model: &Xing40, ids: &[f32], positions: &[f32]) -> Vec<f32> {
    use grim_core::model::CausalLm;
    let n = ids.len();
    let input = grim_backend_cpu::cpu_tensor(ids.to_vec(), Shape::new(vec![n]));
    let pos = grim_backend_cpu::cpu_tensor(positions.to_vec(), Shape::new(vec![n]));
    let mut session = model.new_session();
    model
        .forward(session.as_mut(), &input, &pos, &[])
        .expect("Xing4.0 forward")
        .to_vec_f32()
        .expect("logits to f32")
}

/// The FP8 arm: a block-scaled `F8_E4M3` safetensors export must load and
/// produce finite, non-constant logits — including a sparse MoE layer, so the
/// router, `e_score_correction_bias` read, experts, and shared expert all run.
#[test]
fn loads_block_fp8_safetensors_and_forwards() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();
    let cfg = build(dir, 2);
    let model = load(dir, cfg);

    let logits = forward_logits(&model, &[1.0, 2.0, 3.0], &[0.0, 1.0, 2.0]);
    assert_eq!(logits.len(), 3 * VOCAB);
    assert!(
        logits.iter().all(|x| x.is_finite()),
        "logits must be finite"
    );
    assert!(
        logits.iter().any(|x| *x != 0.0),
        "logits must not be all-zero (weights failed to load or dequantize)"
    );
    // Distinct positions must yield distinct distributions — a collapsed
    // residual stream or dropped MHC stream mixing would flatten these.
    assert_ne!(
        &logits[0..VOCAB],
        &logits[VOCAB..2 * VOCAB],
        "per-position logits collapsed to a constant"
    );
}

/// The block scale must actually be applied. A checkpoint whose codes are
/// identical but whose `weight_scale_inv` values are doubled must produce
/// different logits — proof the FP8 sibling is read, not ignored.
#[test]
fn fp8_block_scale_is_applied() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();
    build(dir, 2);
    let a = load(dir, tiny_config(2));
    let logits_a = forward_logits(&a, &[1.0, 2.0], &[0.0, 1.0]);

    // Re-emit the whole file with every `weight_scale_inv` value doubled and the
    // codes untouched. (Rewriting in place would shift the byte offsets that
    // later tensors point at, since F32 scales are the same width — but the
    // header must stay consistent, so a full re-serialize is the safe form.)
    let raw = std::fs::read(dir.join("model.safetensors")).expect("read");
    let header_len = u64::from_le_bytes(raw[0..8].try_into().unwrap()) as usize;
    let header: serde_json::Value =
        serde_json::from_slice(&raw[8..8 + header_len]).expect("parse header");
    let body_start = 8 + header_len;
    let body = &raw[body_start..];

    let mut new_body: Vec<u8> = Vec::with_capacity(body.len());
    let mut new_header = String::from("{");
    let mut first = true;
    let mut doubled = 0usize;
    // Walk tensors in offset order and copy their regions, doubling scale values.
    let mut entries: Vec<(String, usize, usize, serde_json::Value)> = Vec::new();
    for (name, e) in header.as_object().expect("header obj") {
        let offs = e["data_offsets"].as_array().expect("offsets");
        entries.push((
            name.clone(),
            offs[0].as_u64().expect("start") as usize,
            offs[1].as_u64().expect("end") as usize,
            e.clone(),
        ));
    }
    entries.sort_by_key(|(_, s, _, _)| *s);
    for (name, s, e, meta) in entries {
        if !first {
            new_header.push(',');
        }
        first = false;
        let start = new_body.len();
        if name.ends_with("_scale_inv") {
            for c in body[s..e].chunks_exact(4) {
                let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]) * 2.0;
                new_body.extend_from_slice(&v.to_le_bytes());
                doubled += 1;
            }
        } else {
            new_body.extend_from_slice(&body[s..e]);
        }
        let end = new_body.len();
        let mut m = meta.as_object().expect("meta obj").clone();
        m.insert("data_offsets".into(), serde_json::json!([start, end]));
        new_header.push_str(&format!("\"{name}\":{}", serde_json::Value::Object(m)));
    }
    new_header.push('}');
    assert!(doubled > 0, "expected at least one weight_scale_inv tensor");

    let hdr = new_header.as_bytes();
    let mut out = Vec::new();
    out.extend_from_slice(&(hdr.len() as u64).to_le_bytes());
    out.extend_from_slice(hdr);
    out.extend_from_slice(&new_body);
    std::fs::write(dir.join("model.safetensors"), &out).expect("write doubled");

    let b = load(dir, tiny_config(2));
    let logits_b = forward_logits(&b, &[1.0, 2.0], &[0.0, 1.0]);
    assert_ne!(
        logits_a, logits_b,
        "doubling every block scale must change the forward pass"
    );
}

/// Two forward passes over fresh sessions must agree: the MHC stream state is
/// rebuilt from the embedding each call, never carried across invocations.
#[test]
fn forward_is_deterministic_across_sessions() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();
    build(dir, 2);
    let model = load(dir, tiny_config(2));

    let a = forward_logits(&model, &[4.0, 5.0], &[0.0, 1.0]);
    let b = forward_logits(&model, &[4.0, 5.0], &[0.0, 1.0]);
    assert_eq!(a, b, "forward must be deterministic across sessions");
}
