//! Test for the model-adoption seam of plan item 5 (2026-09-29).
//!
//! `LlamaBlock::load_tp` builds its own `LayerAttentionSpec` via
//! `LayerAttentionSpec::default_full`, so every one of the 77 passthrough
//! model files -- all of which funnel through `Llama::load_tp` -> this --
//! silently gets `NormKind::Rms` with no way to say otherwise. The spec fields
//! landed in commit `847c7222`; this is the seam that makes them reachable, and
//! without it the 11 LayerNorm models and 15 post-norm models stay wrong no
//! matter what a model file writes.
//!
//! The default must stay `Rms` after this change. If the seam defaulted to
//! anything else, the entire existing corpus would change arithmetic silently.

use grim_models_transformer::block::{LayerAttentionSpec, LlamaBlock};
use grim_models_transformer::LlamaConfig;
use grim_nn::NormKind;
use grim_tensor::dtype::{DType, QuantProvenance};
use grim_tensor::{Device, RawTensor, TensorMeta, TensorProvider};

struct StubProvider {
    cfg: LlamaConfig,
}

fn cfg() -> LlamaConfig {
    LlamaConfig {
        vocab_size: 64,
        hidden_size: 8,
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 4,
        num_layers: 1,
        intermediate_size: 16,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_seq_len: 32,
        partial_rotary_factor: 1.0,
        yarn: None,
    }
}

/// A provider that omits every `*_b` tensor, so a spec that declares
/// `has_norm_bias` is asking for something the checkpoint does not have.
struct NoBiasProvider {
    cfg: LlamaConfig,
}

impl TensorProvider for NoBiasProvider {
    fn get(&self, name: &str) -> grim_tensor::error::Result<RawTensor> {
        if name.ends_with("bias") {
            return Err(grim_tensor::error::Error::Backend(format!(
                "tensor '{name}' not found"
            )));
        }
        StubProvider { cfg: self.cfg.clone() }.get(name)
    }

    fn meta(&self, name: &str) -> grim_tensor::error::Result<TensorMeta> {
        StubProvider { cfg: self.cfg.clone() }.meta(name)
    }
}

impl TensorProvider for StubProvider {
    fn get(&self, name: &str) -> grim_tensor::error::Result<RawTensor> {
        let c = &self.cfg;
        // Norms are 1-D [hidden]; the sharded projections must be 2-D, since
        // `shard_raw_tensor` requires it.
        let (rows, cols) = if name.contains("attn_q_norm") || name.contains("attn_k_norm") {
            (c.head_dim, 1)
        } else if name.contains("attn_norm") || name.contains("ffn_norm") {
            (c.hidden_size, 1)
        } else if name.contains("w_gate") || name.contains("w_up") {
            (c.intermediate_size, c.hidden_size)
        } else if name.contains("w_down") {
            (c.hidden_size, c.intermediate_size)
        } else if name.contains("wq") {
            (c.num_heads * c.head_dim, c.hidden_size)
        } else if name.contains("wo") {
            (c.hidden_size, c.num_heads * c.head_dim)
        } else if name.contains("wk") || name.contains("wv") {
            (c.num_kv_heads * c.head_dim, c.hidden_size)
        } else if name.contains("tok_embeddings") || name.contains("output") {
            (c.vocab_size, c.hidden_size)
        } else {
            (c.hidden_size, 1)
        };
        let shape = if cols == 1 {
            vec![rows]
        } else {
            vec![rows, cols]
        };
        let n: usize = shape.iter().product();
        Ok(RawTensor {
            bytes: vec![0u8; n * 4],
            shape,
            dtype: DType::F32,
            provenance: QuantProvenance::GrimNative,
        })
    }

    fn meta(&self, _name: &str) -> grim_tensor::error::Result<TensorMeta> {
        Ok(TensorMeta {
            dtype: DType::F32,
            provenance: QuantProvenance::GrimNative,
            shape: vec![],
            fusion_mask: 0,
        })
    }
}

fn load() -> LlamaBlock {
    let c = cfg();
    let provider = StubProvider { cfg: c.clone() };
    let ws = grim_nn::WeightSource::root(&provider, Device::Cpu);
    LlamaBlock::load_tp(&ws, &c, Default::default()).expect("block loads")
}

#[test]
fn the_default_seam_still_means_rms() {
    // This is the guarantee that keeps the existing corpus unchanged: a passthrough
    // that never opts in must behave exactly as it did before 847c7222.
    let spec = LayerAttentionSpec::default_full(2, 1, 4, 10000.0);
    assert_eq!(spec.norm_kind, NormKind::Rms);
    assert!(!spec.has_norm_bias);
    assert!(!spec.has_attn_post_norm);
}

#[test]
fn a_loaded_block_without_opt_in_is_rms() {
    let b = load();
    assert_eq!(
        b.attn_norm.kind,
        NormKind::Rms,
        "a block loaded via the no-spec path must be RMS"
    );
    assert_eq!(b.ffn_norm.kind, NormKind::Rms);
    assert!(b.attn_post_norm.is_none());
}

#[test]
fn a_loaded_block_honours_an_explicit_layer_norm_spec() {
    // The point of the seam: a model file that asks for LayerNorm gets
    // LayerNorm, without any change to LlamaBlock's own loading code.
    let c = cfg();
    let provider = StubProvider { cfg: c.clone() };
    let ws = grim_nn::WeightSource::root(&provider, Device::Cpu);
    let mut spec = LayerAttentionSpec::default_full(2, 1, 4, 10000.0);
    spec.norm_kind = NormKind::LayerNorm;
    let b = LlamaBlock::load_tp_spec(&ws, &c, &spec, Default::default()).expect("block loads");
    assert_eq!(
        b.attn_norm.kind,
        NormKind::LayerNorm,
        "an explicit spec must reach the block; this is the adoption seam"
    );
    assert_eq!(b.ffn_norm.kind, NormKind::LayerNorm);
}

#[test]
fn a_weightless_layer_norm_still_loads() {
    // `olmo.cpp:65-67` passes NULL, NULL. A spec asking for LayerNorm with no
    // bias must load rather than erroring on a missing *_b tensor.
    let c = cfg();
    let provider = StubProvider { cfg: c.clone() };
    let ws = grim_nn::WeightSource::root(&provider, Device::Cpu);
    let mut spec = LayerAttentionSpec::default_full(2, 1, 4, 10000.0);
    spec.norm_kind = NormKind::LayerNorm;
    spec.has_norm_bias = false;
    let b = LlamaBlock::load_tp_spec(&ws, &c, &spec, Default::default()).expect("block loads");
    assert!(!b.attn_norm.has_bias());
}

#[test]
fn a_declared_bias_that_the_checkpoint_lacks_is_an_error() {
    // `Linear::load` treats a declared-but-absent bias as a hard error
    // (modules.rs:809 uses `?`). `Norm` must match that: a spec claiming a
    // norm bias the file does not have is a spec bug or a wrong checkpoint,
    // and silently continuing would run the model without a normalisation it
    // was configured for -- wrong numbers with no error.
    //
    // This case had no coverage: a mutation that kept `has_bias` honest but
    // swallowed a missing bias compiled and passed every test.
    let c = cfg();
    let provider = NoBiasProvider { cfg: c.clone() };
    let ws = grim_nn::WeightSource::root(&provider, Device::Cpu);
    let mut spec = LayerAttentionSpec::default_full(2, 1, 4, 10000.0);
    spec.has_norm_bias = true;
    let r = LlamaBlock::load_tp_spec(&ws, &c, &spec, Default::default());
    assert!(
        r.is_err(),
        "a declared norm bias that the checkpoint lacks must fail the load"
    );
}

/// `Llama::load_tp_moe` with an all-`None` moe_spec must load a dense model,
/// and must carry the spec's norm kind into every layer.
///
/// This path had no dense caller before: the only three `load_tp_moe_specs`
/// call sites are laguna, maple and mellum, all MoE. The eleven LayerNorm
/// models cannot adopt `LayerNorm` through `Llama::load_tp`, which builds its
/// own spec via `default_full`, so this all-`None` path is the only way in --
/// and it was untested.
#[test]
fn a_dense_model_can_adopt_layer_norm_through_the_moe_entry_point() {
    let c = cfg();
    let provider = StubProvider { cfg: c.clone() };
    let ws = grim_nn::WeightSource::root(&provider, Device::Cpu);
    let none: Vec<Option<grim_models_transformer::moe_block::MoESpec>> =
        vec![None; c.num_layers];
    let m = grim_models_transformer::Llama::load_tp_moe(
        Device::Cpu,
        &ws,
        c.clone(),
        &none,
        Default::default(),
    )
    .expect("a dense model must load through load_tp_moe");
    assert_eq!(m.layers.len(), c.num_layers);
    // The default spec is RMS; this asserts the dense path is reachable, not
    // that it adopts LayerNorm, which needs a per-layer spec.
    assert!(m.layers.iter().all(|l| l.attn_norm.kind == NormKind::Rms));
}

#[test]
fn a_dense_layer_norm_model_loads_through_load_tp_moe_specs() {
    let c = cfg();
    let provider = StubProvider { cfg: c.clone() };
    let ws = grim_nn::WeightSource::root(&provider, Device::Cpu);
    let none: Vec<Option<grim_models_transformer::moe_block::MoESpec>> =
        vec![None; c.num_layers];
    let specs: Vec<LayerAttentionSpec> = (0..c.num_layers)
        .map(|_| {
            let mut s = LayerAttentionSpec::default_full(
                c.num_heads,
                c.num_kv_heads,
                c.head_dim,
                c.rope_theta,
            );
            s.norm_kind = NormKind::LayerNorm;
            s
        })
        .collect();
    let m = grim_models_transformer::Llama::load_tp_moe_specs(
        Device::Cpu,
        &ws,
        c.clone(),
        &none,
        &specs,
        Default::default(),
    )
    .expect("a dense LayerNorm model must load");
    assert!(
        m.layers.iter().all(|l| l.attn_norm.kind == NormKind::LayerNorm),
        "every layer must adopt the spec's norm kind"
    );
}
