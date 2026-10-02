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
    ..Default::default()
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
        // A `*bias` leaf is 1-D [out] whatever the weight's rank. Must come
        // first: "output.bias" contains "output", so the weight arm below would
        // otherwise answer it with a 2-D [vocab, hidden].
        if name.ends_with("bias") {
            let n = if name.contains("output") {
                c.vocab_size
            } else if name.contains("wq") || name.contains("wk") || name.contains("wv") {
                c.num_heads * c.head_dim
            } else if name.contains("wo") {
                c.hidden_size
            } else {
                c.hidden_size
            };
            return Ok(raw_from_shape(vec![n]));
        }
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
        Ok(raw_from_shape(shape))
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

/// A deterministic non-zero F32 tensor of `shape`.
///
/// Non-zero because `Llama::load_tp` rejects a structurally-broken model whose
/// weights are all zero (`weights_look_broken`), which is a correct guard and
/// not something a stub should trip.
fn raw_from_shape(shape: Vec<usize>) -> RawTensor {
    let n: usize = shape.iter().product();
    let bytes: Vec<u8> = (0..n)
        .flat_map(|i| ((i % 7) as f32 * 0.125 + 0.0625).to_le_bytes())
        .collect();
    RawTensor {
        bytes,
        shape,
        dtype: DType::F32,
        provenance: QuantProvenance::GrimNative,
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
    //
    // This uses `NoBiasProvider`, because the bias is now gated on PRESENCE:
    // `StubProvider` answers every name, so presence-gating finds a
    // `attn_norm.bias` there and correctly attaches it. That is the behaviour
    // `output_bias_spec.rs` pins; here the point is the absent case.
    let c = cfg();
    let provider = NoBiasProvider { cfg: c.clone() };
    let ws = grim_nn::WeightSource::root(&provider, Device::Cpu);
    let mut spec = LayerAttentionSpec::default_full(2, 1, 4, 10000.0);
    spec.norm_kind = NormKind::LayerNorm;
    spec.has_norm_bias = false;
    let b = LlamaBlock::load_tp_spec(&ws, &c, &spec, Default::default()).expect("block loads");
    assert!(!b.attn_norm.has_bias());
}

#[test]
fn a_checkpoint_that_ships_a_bias_gets_it_without_a_flag() {
    // The companion to the test above: `StubProvider` answers every name, so
    // `attn_norm.bias` exists, and the block must attach it even though
    // `has_norm_bias` is false. llama.cpp has no bias hparam -- `build_norm`
    // applies whatever tensor the loader created, so presence is the signal.
    let c = cfg();
    let provider = StubProvider { cfg: c.clone() };
    let ws = grim_nn::WeightSource::root(&provider, Device::Cpu);
    let mut spec = LayerAttentionSpec::default_full(2, 1, 4, 10000.0);
    spec.norm_kind = NormKind::Rms;
    spec.has_norm_bias = false;
    let b = LlamaBlock::load_tp_spec(&ws, &c, &spec, Default::default()).expect("block loads");
    assert!(
        b.attn_norm.has_bias(),
        "the checkpoint has attn_norm.bias and the block dropped it"
    );
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

#[test]
fn use_parallel_residual_reaches_the_block_through_load_tp() {
    // The branch is only reachable if `LlamaConfig::use_parallel_residual`
    // survives `LlamaBlock::load_tp`. A test that sets the field on an
    // already-built block proves the branch works but NOT that the flag is
    // plumbed, and hardcoding `false` at the load site survived the whole 283
    // test library suite. This closes that: load a block through the real
    // `load_tp` with the flag set, and read it back off the block.
    let mut c = cfg();
    assert!(
        !c.use_parallel_residual,
        "the test fixture must start sequential, or this proves nothing"
    );
    let provider = StubProvider { cfg: c.clone() };
    let ws = grim_nn::WeightSource::root(&provider, Device::Cpu);
    let sequential = LlamaBlock::load_tp(&ws, &c, Default::default()).expect("loads");
    assert!(
        !sequential.use_parallel_residual,
        "load_tp must default a sequential config to sequential"
    );

    c.use_parallel_residual = true;
    let provider = StubProvider { cfg: c.clone() };
    let ws = grim_nn::WeightSource::root(&provider, Device::Cpu);
    let parallel = LlamaBlock::load_tp(&ws, &c, Default::default()).expect("loads");
    assert!(
        parallel.use_parallel_residual,
        "load_tp dropped use_parallel_residual, so the GGUF key cannot reach the \
         block and every gptneox checkpoint computes ffn(attn(x) + x)"
    );
}

/// The output head must honour `cfg.has_output_bias`.
///
/// Reverting both `Linear::load_column_parallel` call sites in `model.rs` to
/// `/*has_bias=*/ false` passes every test that inspects a `Linear` directly,
/// because the flag is read in `model.rs` and nowhere else. Only a load through
/// `Llama` can see it.
///
/// phi2, qwen2 and wavtokenizer-dec each do
/// `ggml_add(ctx0, cur, model.output_b)` on the logits, so a silent `false`
/// here drops a term from the output of three models.
#[test]
fn the_output_head_honours_has_output_bias() {
    let mut c = cfg();
    c.has_output_bias = true;
    let provider = StubProvider { cfg: c.clone() };
    let ws = grim_nn::WeightSource::root(&provider, Device::Cpu);
    let m = grim_models_transformer::Llama::load_tp(
        Device::Cpu,
        &ws,
        c.clone(),
        Default::default(),
    )
    .expect("Llama loads");
    assert_eq!(c.has_output_bias, m.cfg.has_output_bias);
    assert!(
        m.output.bias.is_some(),
        "has_output_bias = true but Llama.output has no bias, so phi2, qwen2 \
         and wavtokenizer-dec drop `output_b` from the logits"
    );
}

/// The FINAL norm's bias must reach `Llama.norm`.
///
/// Twenty references create `LLM_TENSOR_OUTPUT_NORM, "bias"` and pass it to
/// `build_norm`:
///
/// ```text
/// phi2.cpp:127          build_norm(cur, model.output_norm, model.output_norm_b, ...)
/// wavtokenizer-dec.cpp:252
/// rwkv6qwen2.cpp:156    cur = build_norm(cur, model.output_norm, model.output_norm_b, LLM_NORM, ...);
/// ```
///
/// `StubProvider` answers every name, so `norm.bias` exists here, and
/// `Norm::load` is presence-gated, so it must be attached without a flag.
/// `Llama.norm` is the only norm whose bias nothing asserts: `attn_norm`'s
/// half is covered by the two tests above, this is the final norm's.
#[test]
fn the_final_norm_gets_its_bias_when_the_checkpoint_ships_one() {
    let c = cfg();
    let provider = StubProvider { cfg: c.clone() };
    let ws = grim_nn::WeightSource::root(&provider, Device::Cpu);
    let m = grim_models_transformer::Llama::load_tp(
        Device::Cpu,
        &ws,
        c.clone(),
        Default::default(),
    )
    .expect("Llama loads");
    assert!(
        m.norm.has_bias(),
        "the checkpoint ships norm.bias and the final norm dropped it, so the \
         logits are missing `ggml_add(ctx0, cur, model.output_norm_b)` for \
         twenty references"
    );
}

/// The final norm's bias and the layer norms' bias are independent facts.
///
/// `rwkv6qwen2` and `wavtokenizer-dec` create `output_norm_b` with NO
/// `attn_norm_b` / `ffn_norm_b`, so one flag cannot express both. Presence
/// gating is what makes that work; a flag would have to be told about the
/// difference for each of the twenty.
#[test]
fn a_model_with_an_output_norm_bias_and_no_layer_bias_loads() {
    // A provider that serves `norm.bias` but rejects `attn_norm.bias` and
    // `ffn_norm.bias`. That is the rwkv6qwen2 / wavtokenizer-dec shape.
    struct OutputBiasOnly {
        cfg: LlamaConfig,
    }
    impl TensorProvider for OutputBiasOnly {
        fn get(&self, name: &str) -> grim_tensor::error::Result<RawTensor> {
            if is_layer_norm_bias(name) {
                return Err(grim_tensor::error::Error::Backend(format!(
                    "tensor '{name}' not found"
                )));
            }
            StubProvider { cfg: self.cfg.clone() }.get(name)
        }
        fn meta(&self, name: &str) -> grim_tensor::error::Result<TensorMeta> {
            if is_layer_norm_bias(name) {
                return Err(grim_tensor::error::Error::Backend(format!(
                    "tensor '{name}' not found"
                )));
            }
            StubProvider { cfg: self.cfg.clone() }.meta(name)
        }
    }

    /// True for `attn_norm.bias` / `ffn_norm.bias`, but NOT `norm.bias` --
    /// the final norm's bias is exactly what this provider keeps.
    fn is_layer_norm_bias(name: &str) -> bool {
        name.ends_with("bias") && (name.contains("attn_norm") || name.contains("ffn_norm"))
    }

    let c = cfg();
    let provider = OutputBiasOnly { cfg: c.clone() };
    let ws = grim_nn::WeightSource::root(&provider, Device::Cpu);
    let m = grim_models_transformer::Llama::load_tp(
        Device::Cpu,
        &ws,
        c.clone(),
        Default::default(),
    )
    .expect("a checkpoint with output_norm_b and no layer *_norm_b must load");
    assert!(m.norm.has_bias(), "the final norm lost its bias");
    assert!(
        !m.layers[0].attn_norm.has_bias(),
        "the layer norm picked up a bias the checkpoint does not have"
    );
}

/// WhiteRaven blocked FP8 must survive the load path and map to its own
/// `QuantFormat`.
///
/// The failure this pins: `QuantFormat::try_from(&Storage)` had no arm for
/// `FloatPackScheme::Fp8Blocked16`, so `Linear::forward` returned
/// Unimplemented("no QuantFormat mapping") at the first forward -- AFTER the
/// load had succeeded and looked fine. A load-only assertion passes; only the
/// mapping (which the forward calls) exercises the real gap.
#[test]
fn blocked_fp8_maps_to_its_own_format_and_dequantizes_row_major() {
    use grim_tensor::dtype::{FloatPackScheme, Storage};

    let (out, inn) = (16usize, 16usize);
    let w: Vec<f32> = (0..out * inn).map(|i| (i as f32) * 0.03125 - 0.5).collect();
    let codes: Vec<u8> = w.iter().map(|&v| grim_quant::f32_to_fp8_e4m3(v)).collect();
    let blocked = grim_quant::block_fp8_16x16(&codes, out, inn).expect("block");
    let storage = Storage::FloatPack(FloatPackScheme::Fp8Blocked16);

    let fmt = grim_tensor::QuantFormat::try_from(&storage)
        .expect("blocked fp8 must map to a QuantFormat");
    assert_eq!(
        fmt,
        grim_tensor::QuantFormat::Fp8Blocked16,
        "blocked fp8 must map to its own format, not Fp8 -- a shared tag would \
         route a blocked tensor into the row-major kernel"
    );
    assert_ne!(
        fmt,
        grim_tensor::QuantFormat::Fp8,
        "blocked and row-major fp8 must not share a QuantFormat"
    );

    // The host decoder the loader/CPU/Vulkan fallback uses must return the
    // original codes in row-major order.
    let back = grim_quant::dequant_fp8_blocked16(&blocked, out, inn).expect("dequant");
    assert_eq!(back.len(), out * inn);
    for (i, (&b, &v)) in codes.iter().zip(&back).enumerate() {
        assert_eq!(
            v.to_bits(),
            grim_quant::fp8_e4m3_to_f32(b).to_bits(),
            "element {i}: blocked round-trip diverged"
        );
    }
}
