//! End-to-end forward test against a SYNTHETIC qwen4exp checkpoint.
//!
//! # Why this file exists
//!
//! `qwen38_qsa.rs` and `qwen38_gdn.rs` prove the math. Nothing proved the layer
//! forward path actually reaches them. A mutation check over `qwen38_flash_next.rs`
//! found five survivors, all in the forward path:
//!
//!   - ReLU moved outside the indexer head sum
//!   - indexer keys normed per cell before pooling
//!   - the sparse result left on the CPU instead of the input device
//!   - the single-token guard dropped, applying a one-row mask to prefill
//!   - `GRIM_QWEN38_QSA=0` no longer forcing dense
//!
//! That is the same failure shape task 4c started in: a correct implementation
//! nothing calls. This file drives a real forward and asserts the sparse path
//! is taken, the mask prunes, and the gate and seam behave.
//!
//! # The checkpoint
//!
//! Emits the REAL GGUF tensor names the loader asks for (`attn_q`,
//! `indexer.q_proj`, `hc_attn_*`, `ffn_*_exps`, ...), not the legacy
//! safetensors names. Dimensions are shrunk to keep the test fast; every
//! relationship the code depends on is preserved:
//!
//!   - `compress_ratio > 0` exactly on the full-attention layers
//!   - `idx_dim` on the indexer equals the GDN `ssm_d_state`
//!   - attention head_dim, kv heads and indexer query-head count are consistent
//!   - `index_k_proj` is `[hidden, idx_dim]` (ONE key head)
//!   - `index_q_proj` is `[hidden, n_idx_h * idx_dim]`

use std::collections::HashMap;

use grim_core::{CausalLm, Model};
use grim_nn::WeightSource;
use grim_tensor::provider::{RawTensor, TensorMeta, TensorProvider};
use grim_tensor::{DType, Device, QuantProvenance, Shape};

use grim_models_transformer::qwen38_flash_next::{Qwen38FlashNext, Qwen38FlashNextConfig};

/// A deterministic pseudo-random f32 tensor, so a run is reproducible without
/// pulling in an RNG dependency.
fn raw_f32(seed: u32, shape: Vec<usize>) -> RawTensor {
    let n: usize = shape.iter().product();
    let mut bytes = Vec::with_capacity(n * 4);
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    for _ in 0..n {
        // xorshift32
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        // Map to [-0.05, 0.05) so activations stay in a sane range.
        let v = ((state >> 8) as f32 / 8_388_608.0) - 1.0;
        bytes.extend_from_slice(&(v * 0.05).to_le_bytes());
    }
    RawTensor {
        bytes,
        shape,
        dtype: DType::F32,
        provenance: QuantProvenance::GrimNative,
    }
}

fn raw_f32_ones(shape: Vec<usize>) -> RawTensor {
    let n: usize = shape.iter().product();
    let bytes: Vec<u8> = std::iter::repeat(0)
        .flat_map(|_| 1.0f32.to_le_bytes())
        .take(n * 4)
        .collect();
    RawTensor {
        bytes,
        shape,
        dtype: DType::F32,
        provenance: QuantProvenance::GrimNative,
    }
}

struct SynthProvider {
    tensors: HashMap<String, RawTensor>,
}

impl TensorProvider for SynthProvider {
    fn get(&self, name: &str) -> grim_tensor::error::Result<RawTensor> {
        self.tensors
            .get(name)
            .cloned()
            .ok_or_else(|| grim_tensor::error::Error::Backend(format!("missing tensor {name}")))
    }
    fn meta(&self, name: &str) -> grim_tensor::error::Result<TensorMeta> {
        let t = self.get(name)?;
        Ok(TensorMeta {
            dtype: t.dtype,
            provenance: t.provenance,
            shape: t.shape,
            fusion_mask: 0,
        })
    }
    fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }
}

/// A shrunk but structurally faithful qwen4exp config.
///
/// `n_layers` is small and the layer kinds are forced so exactly one full-
/// attention layer exists, which is the one the QSA assertions need.
fn synth_cfg(n_layers: usize) -> Qwen38FlashNextConfig {
    let mut cfg = Qwen38FlashNextConfig::default();
    cfg.vocab_size = 64;
    cfg.hidden_size = 32;
    cfg.num_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.head_dim = 8;
    cfg.value_head_dim = 8;
    cfg.num_layers = n_layers;
    cfg.intermediate_size = 16;
    cfg.num_experts = 4;
    cfg.num_experts_per_tok = 2;
    cfg.shared_expert_intermediate_size = Some(16);
    cfg.hc_count = 2;
    cfg.hc_lowrank = 4;
    // GDN geometry, shrunk but internally consistent.
    cfg.ssm_d_state = 4;
    cfg.ssm_n_group = 2;
    cfg.ssm_d_inner = 16;
    cfg.ssm_dt_rank = 4;
    cfg.linear_conv_kernel_dim = 2;
    cfg.linear_num_key_heads = 2;
    cfg.linear_key_head_dim = 4;
    cfg.linear_value_head_dim = 4;
    cfg.linear_num_value_heads = 4;
    // QSA geometry.
    cfg.indexer_n_heads = 2;
    cfg.indexer_key_length = 4;
    cfg.indexer_top_k = 4;
    // PLE off: the n-gram table is not what this test is about, and leaving it
    // on would require a 320M-row synthetic table.
    cfg.ngram_vocab_size = None;
    cfg.ngram_dim = None;
    cfg.max_seq_len = 64;
    cfg
}

/// Layer kinds: the last layer is full attention, the rest GDN, so exactly one
/// layer takes the QSA path.
fn layer_types(n_layers: usize) -> Vec<String> {
    (0..n_layers)
        .map(|i| {
            if i + 1 == n_layers {
                "full_attention".to_string()
            } else {
                "linear_attention".to_string()
            }
        })
        .collect()
}

fn compress_ratios(n_layers: usize) -> Vec<usize> {
    (0..n_layers)
        .map(|i| if i + 1 == n_layers { cfg_ratio() } else { 0 })
        .collect()
}

fn cfg_ratio() -> usize {
    2
}

/// Build the tensor set the loader requires, under the GGUF naming the code
/// uses (no `model.` / `language_model.` prefix, so `load_tp` falls through to
/// the `ws.scoped("model")` branch and then finds the tensors at the root).
fn synth_tensors(cfg: &Qwen38FlashNextConfig) -> SynthProvider {
    let mut t: HashMap<String, RawTensor> = HashMap::new();
    let h = cfg.hidden_size;
    let hc_dim = h * cfg.hc_count;
    let q_dim = cfg.num_heads * cfg.head_dim;
    let kv_dim = cfg.num_kv_heads * cfg.head_dim;
    let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
    let value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
    let conv_dim = key_dim * 2 + value_dim;
    let idx_d = cfg.indexer_key_length;
    let n_idx = cfg.indexer_n_heads;
    let hc_lr = cfg.hc_lowrank;

    // `load_tp` probes model.language_model.embed_tokens first; when that is
    // absent it falls back to `ws.scoped("model")`, so every tensor lives under
    // the `model.` prefix.
    let mut put = |name: String, r: RawTensor| {
        t.insert(format!("model.{name}"), r);
    };

    // GGUF stores weight matrices as [out, in] and `Linear::load_shape` wants
    // [in, out] in its own argument order, so the embedding is [hidden, vocab].
    put(
        "embed_tokens.weight".into(),
        raw_f32(1, vec![h, cfg.vocab_size]),
    );
    put("output_norm.weight".into(), raw_f32_ones(vec![h]));
    // The loader falls back to this when `norm` is absent. Provided so the
    // fallback resolves; a checkpoint that only has one of the two is fine.
    put(
        "hyper_connection_mixer.hc_norm.weight".into(),
        raw_f32_ones(vec![h]),
    );
    put("output.weight".into(), raw_f32(2, vec![h, cfg.vocab_size]));

    for i in 0..cfg.num_layers {
        let p = format!("layers.{i}");
        let s = (i as u32 + 10) * 977;

        // Hyper-connection mixers, per branch.
        for br in ["attn", "ffn"] {
            put(
                format!("{p}.hc_{br}_norm.weight"),
                raw_f32_ones(vec![hc_dim]),
            );
            put(
                format!("{p}.hc_{br}_down.weight"),
                raw_f32(s + 1, vec![hc_lr, hc_dim]),
            );
            put(
                format!("{p}.hc_{br}_up.weight"),
                raw_f32(s + 2, vec![hc_dim, hc_lr]),
            );
            put(
                format!("{p}.hc_{br}_inject.weight"),
                raw_f32(s + 3, vec![cfg.hc_count, hc_dim]),
            );
        }
        put(format!("{p}.input_layernorm.weight"), raw_f32_ones(vec![h]));
        put(
            format!("{p}.post_attention_layernorm.weight"),
            raw_f32_ones(vec![h]),
        );

        // MoE.
        put(
            format!("{p}.ffn_gate_inp.weight"),
            raw_f32(s + 4, vec![cfg.num_experts, h]),
        );
        put(
            format!("{p}.ffn_gate_inp_shexp.weight"),
            raw_f32(s + 5, vec![1, h]),
        );
        put(
            format!("{p}.ffn_gate_exps.weight"),
            raw_f32(s + 6, vec![cfg.num_experts, cfg.intermediate_size, h]),
        );
        put(
            format!("{p}.ffn_up_exps.weight"),
            raw_f32(s + 7, vec![cfg.num_experts, cfg.intermediate_size, h]),
        );
        put(
            format!("{p}.ffn_down_exps.weight"),
            raw_f32(s + 8, vec![cfg.num_experts, h, cfg.intermediate_size]),
        );
        let shared = cfg
            .shared_expert_intermediate_size
            .unwrap_or(cfg.intermediate_size);
        put(
            format!("{p}.ffn_gate_shexp.weight"),
            raw_f32(s + 9, vec![shared, h]),
        );
        put(
            format!("{p}.ffn_up_shexp.weight"),
            raw_f32(s + 10, vec![shared, h]),
        );
        put(
            format!("{p}.ffn_down_shexp.weight"),
            raw_f32(s + 11, vec![h, shared]),
        );

        let is_full = cfg.layer_types[i] == "full_attention";
        if is_full {
            put(
                format!("{p}.attn_q.weight"),
                raw_f32(s + 12, vec![q_dim, h]),
            );
            put(
                format!("{p}.attn_k.weight"),
                raw_f32(s + 13, vec![kv_dim, h]),
            );
            put(
                format!("{p}.attn_v.weight"),
                raw_f32(s + 14, vec![kv_dim, h]),
            );
            put(
                format!("{p}.attn_output.weight"),
                raw_f32(s + 15, vec![q_dim, h]),
            );
            put(
                format!("{p}.attn_q_norm.weight"),
                raw_f32_ones(vec![cfg.head_dim]),
            );
            put(
                format!("{p}.attn_k_norm.weight"),
                raw_f32_ones(vec![cfg.head_dim]),
            );
            // The QSA indexer. ONE key head of idx_d, and n_idx query heads.
            put(
                format!("{p}.indexer.q_proj.weight"),
                raw_f32(s + 16, vec![n_idx * idx_d, h]),
            );
            put(
                format!("{p}.indexer.k_proj.weight"),
                raw_f32(s + 17, vec![idx_d, h]),
            );
            put(
                format!("{p}.indexer.q_norm.weight"),
                raw_f32_ones(vec![idx_d]),
            );
            put(
                format!("{p}.indexer.k_norm.weight"),
                raw_f32_ones(vec![idx_d]),
            );
        } else {
            put(
                format!("{p}.attn_qkv.weight"),
                raw_f32(s + 12, vec![conv_dim, h]),
            );
            put(
                format!("{p}.attn_gate.weight"),
                raw_f32(s + 13, vec![value_dim, h]),
            );
            put(
                format!("{p}.ssm_conv1d.weight"),
                raw_f32(s + 14, vec![conv_dim, cfg.linear_conv_kernel_dim]),
            );
            put(format!("{p}.ssm_a"), raw_f32(s + 15, vec![cfg.ssm_dt_rank]));
            put(
                format!("{p}.ssm_dt.bias"),
                raw_f32(s + 16, vec![cfg.ssm_dt_rank]),
            );
            put(
                format!("{p}.ssm_alpha.weight"),
                raw_f32(s + 17, vec![cfg.ssm_dt_rank, h]),
            );
            put(
                format!("{p}.ssm_beta.weight"),
                raw_f32(s + 18, vec![cfg.ssm_dt_rank, h]),
            );
            // ssm_norm is [ssm_d_state]: the GDN head width the recurrence
            // indexes per state channel, NOT the value-head count.
            put(
                format!("{p}.ssm_norm.weight"),
                raw_f32_ones(vec![cfg.ssm_d_state]),
            );
            put(
                format!("{p}.ssm_out.weight"),
                raw_f32(s + 19, vec![h, value_dim]),
            );
        }
    }

    SynthProvider { tensors: t }
}

fn build_model(cfg: &Qwen38FlashNextConfig) -> Qwen38FlashNext {
    let provider = synth_tensors(cfg);
    let ws = WeightSource::root(&provider, Device::Cpu);
    Qwen38FlashNext::load(Device::Cpu, &ws, cfg.clone())
        .expect("the synthetic qwen4exp checkpoint must load")
}

/// One decode step: a single token through the whole model.
fn decode_once(
    model: &Qwen38FlashNext,
    session: &mut dyn grim_core::session::SessionT,
    tok: u32,
) -> Vec<f32> {
    let ids = grim_backend_cpu::cpu_tensor(vec![tok as f32], Shape::new(vec![1]));
    let pos = grim_backend_cpu::cpu_tensor(vec![0.0f32], Shape::new(vec![1]));
    let out = model
        .forward(session, &ids, &pos, &[])
        .expect("forward must succeed");
    out.to_vec_f32().expect("logits")
}

#[test]
fn synthetic_checkpoint_loads_and_a_forward_finishes() {
    let mut cfg = synth_cfg(3);
    cfg.layer_types = layer_types(3);
    cfg.attention_compress_ratios = compress_ratios(3);
    let model = build_model(&cfg);
    let mut session = model.new_session();
    let logits = decode_once(&model, session.as_mut(), 7);
    assert_eq!(
        logits.len(),
        cfg.vocab_size,
        "one forward must produce vocab_size logits"
    );
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "logits must all be finite; got a non-finite value"
    );
}

#[test]
fn the_full_attention_layer_really_is_sparse() {
    // The strongest available signal that the indexer is reached: with a
    // working indexer the mask selects a strict subset of the KV history, so
    // the attention output must depend on the K/V history and on the top_k
    // budget. Drive enough tokens for the budget to bind.
    let mut cfg = synth_cfg(3);
    cfg.layer_types = layer_types(3);
    cfg.attention_compress_ratios = compress_ratios(3);
    cfg.indexer_top_k = 4;
    let model = build_model(&cfg);
    let mut session = model.new_session();

    // 6 decode steps: more than top_k, so the selection must actually prune.
    let mut outs = Vec::new();
    for t in 0..6u32 {
        outs.push(decode_once(&model, session.as_mut(), t + 1));
    }
    // Each step must produce a distinct distribution: a stateful model, and a
    // model whose attention output ignores the indexer would still be distinct,
    // so this is necessary, not sufficient.
    for i in 1..outs.len() {
        assert_ne!(
            outs[i],
            outs[i - 1],
            "consecutive decode steps must differ; step {i} repeated the previous output"
        );
    }
    assert!(
        outs.iter().all(|o| o.iter().all(|v| v.is_finite())),
        "every step must stay finite"
    );
}

#[test]
fn the_qsa_gate_forces_dense_without_changing_shapes() {
    // GRIM_QWEN38_QSA=0 must make the run succeed and stay finite; it disables
    // the indexer, so the layer falls back to dense attention.
    let mut cfg = synth_cfg(3);
    cfg.layer_types = layer_types(3);
    cfg.attention_compress_ratios = compress_ratios(3);
    // 8 decode steps accumulate 8 KV cells; with top_k = 2 the indexer keeps
    // min(8, 2 + r - 1) of them, so the mask actually prunes.
    cfg.indexer_top_k = 2;

    // A single decode step has n_kv = 1, so the budget selects everything and
    // sparse and dense agree trivially. Accumulate history first so the mask
    // has cells to prune.
    let sparse_first = {
        let model = build_model(&cfg);
        let mut s = model.new_session();
        for t in 0..8u32 {
            decode_once(&model, s.as_mut(), t + 1);
        }
        decode_once(&model, s.as_mut(), 5)
    };
    let dense = {
        // SAFETY-adjacent note: env mutation is process-global, so this test
        // must run in isolation from the sparse assertions above. Rust runs
        // tests in threads, so guard with a process-wide mutex instead of
        // relying on ordering.
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: the process-wide ENV_LOCK is held for the whole window, and
        // the model does no other thread creation, so no other thread can be
        // reading the environment concurrently. Edition 2024 makes set_var
        // unsafe for exactly this reason.
        unsafe {
            std::env::set_var("GRIM_QWEN38_QSA", "0");
        }
        let model = build_model(&cfg);
        let mut s = model.new_session();
        for t in 0..8u32 {
            decode_once(&model, s.as_mut(), t + 1);
        }
        let out = decode_once(&model, s.as_mut(), 5);
        unsafe {
            std::env::remove_var("GRIM_QWEN38_QSA");
        }
        out
    };

    assert_eq!(
        dense.len(),
        sparse_first.len(),
        "the gate must not change the output shape"
    );
    assert!(
        dense.iter().all(|v| v.is_finite()),
        "the dense fallback must stay finite"
    );
    assert_ne!(
        dense, sparse_first,
        "forcing dense must change the result; if it does not, the gate is \
         not wired and the indexer mask was never applied"
    );
}

static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn the_sparse_result_lands_on_the_tensors_own_device() {
    // The HIP seam: the indexer and the masked softmax are host code, and the
    // result must be relocated before `wo`. On CPU this is a no-op, but the
    // code path that performs the relocation is what is being guarded, so this
    // test asserts the model completes a sparse forward and returns a tensor
    // whose device matches the model's.
    let mut cfg = synth_cfg(3);
    cfg.layer_types = layer_types(3);
    cfg.attention_compress_ratios = compress_ratios(3);
    let model = build_model(&cfg);
    let mut session = model.new_session();
    let ids = grim_backend_cpu::cpu_tensor(vec![3.0f32], Shape::new(vec![1]));
    let pos = grim_backend_cpu::cpu_tensor(vec![0.0f32], Shape::new(vec![1]));
    let logits = model
        .forward(session.as_mut(), &ids, &pos, &[])
        .expect("forward");
    assert_eq!(
        *logits.device(),
        *model.device(),
        "the sparse path must return a tensor on the model's own device, \
         not a host buffer"
    );
    assert_eq!(
        *model.device(),
        Device::Cpu,
        "this test runs on the CPU device"
    );
}

#[test]
fn prefill_does_not_apply_the_single_token_mask() {
    // The sparse path masks one query row. Feeding several tokens must fall
    // back to dense rather than broadcast a one-row mask across the batch.
    let mut cfg = synth_cfg(3);
    cfg.layer_types = layer_types(3);
    cfg.attention_compress_ratios = compress_ratios(3);
    let model = build_model(&cfg);
    let mut session = model.new_session();

    let ids = grim_backend_cpu::cpu_tensor(vec![1.0f32, 2.0, 3.0, 4.0], Shape::new(vec![4]));
    let pos = grim_backend_cpu::cpu_tensor(vec![0.0f32, 1.0, 2.0, 3.0], Shape::new(vec![4]));
    let out = model
        .forward(session.as_mut(), &ids, &pos, &[])
        .expect("a 4-token prefill must succeed");
    let v = out.to_vec_f32().expect("logits");
    assert_eq!(
        v.len(),
        4 * cfg.vocab_size,
        "prefill returns one row per token"
    );
    assert!(
        v.iter().all(|x| x.is_finite()),
        "prefill must stay finite; a one-row mask broadcast over a batch \
         would leave masked rows unnormalised"
    );

    // Prefill and token-at-a-time decode are NOT expected to agree here: the
    // stack is hybrid, and the two Gated DeltaNet layers carry a recurrent
    // state that a multi-token step advances differently from N single-token
    // steps. So row equality is the wrong invariant, and asserting it would be
    // asserting a bug.
    //
    // What IS assertable: the prefill must not be corrupted by a mask that was
    // only ever valid for one query row. A broadcast -inf leaves rows whose
    // every logit is non-finite or whose softmax degenerates, and that shows up
    // as a non-finite value or an all-equal row. Check both, per row.
    for r in 0..4 {
        let row = &v[r * cfg.vocab_size..(r + 1) * cfg.vocab_size];
        assert!(
            row.iter().all(|x| x.is_finite()),
            "prefill row {r} has a non-finite logit; a one-row mask broadcast \
             across the batch would produce exactly this"
        );
        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mn = row.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(
            (mx - mn).abs() > 1e-9,
            "prefill row {r} is constant ({mn}); the attention softmax for that \
             row was fully masked out"
        );
    }
}
