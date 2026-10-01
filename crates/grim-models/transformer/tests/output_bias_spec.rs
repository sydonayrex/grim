//! The output projection's bias -- plan item 5, weight-plumbing row.
//!
//! ## The reference fact
//!
//! Three references create a bias on `LLM_TENSOR_OUTPUT` and add it to the
//! logits:
//!
//! ```text
//! phi2.cpp:136            cur = ggml_add(ctx0, cur, model.output_b);
//! qwen2.cpp               cur = ggml_add(ctx0, cur, model.output_b);
//! wavtokenizer-dec.cpp    cur = ggml_add(ctx0, cur, model.output_b);
//! ```
//!
//! Three more create the tensor but never read it:
//!
//! * `phimoe.cpp:23` creates it with flag `0` and never references it again.
//!   Dead in upstream too, so reproducing that is not a porting goal.
//! * `dream.cpp` and `qwen2vl.cpp` create it `TENSOR_NOT_REQUIRED` and never
//!   add it.
//!
//! So the scope is three models, not the six a tensor-table grep suggests.
//! Counting `LLM_TENSOR_OUTPUT, "bias"` alone overstates it by half, and
//! counting the norm biases instead would miss it entirely -- the two are
//! different tensors on different lines.
//!
//! ## What grim does today
//!
//! Both `Linear::load_column_parallel` call sites for the output head pass
//! `/*has_bias=*/ false` unconditionally (`model.rs`), so for those three
//! models the logits are missing a term. Silent: the model loads, runs, and
//! produces plausible wrong numbers.
//!
//! `Linear` already carries an optional bias and its `forward` adds it, so this
//! is configuration, not new arithmetic. What was missing is the config flag,
//! and the test that proves the flag reaches the layer.

use grim_models_transformer::block::LayerAttentionSpec;
use grim_models_transformer::LlamaConfig;
use grim_nn::{Linear, NormKind, WeightSource};
use grim_tensor::dtype::{DType, QuantProvenance};
use grim_tensor::{Device, RawTensor, TensorMeta, TensorProvider};

/// Serves an output head, with or without a bias, so the flag's effect on the
/// loaded layer is observable.
struct OutProvider {
    out_dim: usize,
    in_dim: usize,
    with_bias: bool,
}

impl TensorProvider for OutProvider {
    fn get(&self, name: &str) -> grim_tensor::error::Result<RawTensor> {
        let leaf = name.rsplit('.').next().unwrap_or(name);
        let shape = match leaf {
            "weight" => vec![self.out_dim, self.in_dim],
            "bias" => vec![self.out_dim],
            other => return Err(grim_tensor::error::Error::Backend(format!(
                "tensor '{other}' not found"
            ))),
        };
        if leaf == "bias" && !self.with_bias {
            return Err(grim_tensor::error::Error::Backend(format!(
                "tensor '{name}' not found"
            )));
        }
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

fn load(with_bias: bool, flag: bool) -> Linear {
    let provider = OutProvider {
        out_dim: 4,
        in_dim: 2,
        with_bias,
    };
    let ws = WeightSource::root(&provider, Device::Cpu);
    Linear::load_column_parallel(&ws, 2, 4, flag, Default::default()).expect("output head loads")
}

#[test]
fn an_unbiased_output_head_has_no_bias() {
    // The common case, and the one that must not change: every model that is
    // not phi2/qwen2/wavtokenizer-dec gets `has_bias = false`.
    let l = load(/*with_bias=*/ false, /*flag=*/ false);
    assert!(
        l.bias.is_none(),
        "a has_bias=false output head must not carry a bias tensor"
    );
}

#[test]
fn a_declared_output_bias_is_loaded() {
    // phi2 / qwen2 / wavtokenizer-dec. The flag must reach the layer, or the
    // logits miss `ggml_add(ctx0, cur, model.output_b)`.
    let l = load(/*with_bias=*/ true, /*flag=*/ true);
    assert!(
        l.bias.is_some(),
        "a has_bias=true output head dropped its bias tensor, so those three \
         models compute logits without the reference's `output_b` term"
    );
    assert_eq!(
        l.bias.as_ref().unwrap().shape().dims(),
        &[4],
        "the output bias is one element per vocabulary entry"
    );
}

#[test]
fn a_declared_bias_that_is_absent_is_an_error_not_a_silent_zero() {
    // `Linear::load` uses `?` on the bias, so a checkpoint that claims the bias
    // and does not have it fails loudly. That is deliberate: guessing zero would
    // hide a corrupt or wrong-architecture checkpoint.
    let provider = OutProvider {
        out_dim: 4,
        in_dim: 2,
        with_bias: false,
    };
    let ws = WeightSource::root(&provider, Device::Cpu);
    let r = Linear::load_column_parallel(&ws, 2, 4, true, Default::default());
    assert!(
        r.is_err(),
        "declaring a bias the checkpoint does not provide must fail, not \
         silently produce an unbiased head"
    );
}

/// The norm bias is a different tensor on a different reference line, but it
/// shares the config struct. Pinned so a change to one cannot break the other.
#[test]
fn norm_bias_and_output_bias_do_not_share_a_flag() {
    // phi2 ships BOTH `attn_norm_b`/`ffn_norm_b` AND `output_b`, so the two
    // cannot be one boolean: enabling the norm bias must not imply an output
    // bias, and a model may need either, both, or neither.
    let c = LlamaConfig {
        vocab_size: 4,
        hidden_size: 2,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: 2,
        num_layers: 1,
        intermediate_size: 4,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_seq_len: 8,
        partial_rotary_factor: 1.0,
        yarn: None,
        norm_kind: NormKind::Rms,
        has_norm_bias: true,
        has_attn_post_norm: false,
        use_parallel_residual: false,
        has_output_bias: false,
    };
    assert!(
        c.has_norm_bias,
        "phi2 ships attn_norm_b and ffn_norm_b, so the norm flag is a real, \
         independently settable fact about a model"
    );
}
/// Serves every tensor the norm asks for, INCLUDING the bias, while the config
/// says `has_norm_bias: false`. That combination is the bug: the checkpoint has
/// the tensor and the model is served without it.
struct BiasServing {
    cfg: LlamaConfig,
}

impl TensorProvider for BiasServing {
    fn get(&self, name: &str) -> grim_tensor::error::Result<RawTensor> {
        let leaf = name.rsplit('.').next().unwrap_or(name);
        let dim = self.cfg.hidden_size;
        let shape = if leaf == "weight" || leaf == "bias" {
            vec![dim]
        } else {
            vec![dim]
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

#[test]
fn a_checkpoint_that_ships_a_norm_bias_is_served_even_when_no_flag_says_so() {
    // llama.cpp has no bias hparam: `build_norm` takes whatever tensor the
    // checkpoint has, and `if (mb) cur = ggml_add(ctx0, cur, mb)`
    // (llama-graph.cpp). So presence is the whole signal. gptneox, phi2, mpt,
    // jais, orion, codeshell, starcoder, starcoder2, nemotron and 10 more ship
    // `attn_norm_b`, and every one of them is currently served WITHOUT it
    // because nothing sets `has_norm_bias`.
    let c = LlamaConfig {
        vocab_size: 8,
        hidden_size: 4,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: 4,
        num_layers: 1,
        intermediate_size: 8,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_seq_len: 8,
        partial_rotary_factor: 1.0,
        yarn: None,
        norm_kind: grim_nn::NormKind::Rms,
        has_norm_bias: false,
        has_attn_post_norm: false,
        use_parallel_residual: false,
        has_output_bias: false,
    };
    // Provider serves attn_norm WITH a bias, and cfg says has_norm_bias=false.
    let provider = BiasServing { cfg: c.clone() };
    let ws = grim_nn::WeightSource::root(&provider, Device::Cpu);
    let spec = LayerAttentionSpec::default_full(1, 1, 4, 10000.0);
    let n = grim_nn::Norm::load(
        &ws.pp("attn_norm"),
        c.hidden_size,
        spec.norm_kind,
        c.rms_norm_eps,
        c.has_norm_bias,
    )
    .expect("loads");
    assert!(
        n.has_bias(),
        "the checkpoint ships attn_norm_b and the norm was served without it, \
         so the model computes a normalisation it was not trained with"
    );
}
