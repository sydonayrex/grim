//! The loader must derive the norm kind from the architecture.
//!
//! `ModelArchitecture::uses_layernorm` is tested against the references in
//! `grim-core/tests/norm_kind_by_architecture.rs`. That proves the TABLE. It
//! says nothing about whether the loader reads it, and it does not: until
//! `norm_kind_for` was introduced and called at the five `LlamaConfig` sites,
//! every model was built with `Default::default()` -- RMS -- so all twenty
//! LayerNorm architectures computed the wrong normalisation while the table
//! sat there correct and unused.
//!
//! This covers the mapping itself. Replacing `norm_kind_for(model_arch)` with
//! `NormKind::Rms` passes every test in `grim-core`.
//!
//! LayerNorm subtracts the mean; RMSNorm does not. On any input with a
//! non-zero mean the two disagree, so this is silent wrong numerics rather
//! than a rounding difference.

use grim_core::ModelArchitecture;
use grim_engine::model_loader::norm_kind_for;
use grim_nn::NormKind;

/// The twenty architectures whose reference passes `LLM_NORM` to `build_norm`
/// and which exist as grim variants. Kept as a literal here rather than read
/// from the reference so the test fails loudly if grim-core's table and this
/// list drift; `norm_kind_by_architecture.rs` is what proves the table.
const LAYERNORM: &[ModelArchitecture] = &[
    ModelArchitecture::Bert,
    ModelArchitecture::Bloom,
    ModelArchitecture::Codeshell,
    ModelArchitecture::Cohere2,
    ModelArchitecture::Dbrx,
    ModelArchitecture::Falcon,
    ModelArchitecture::Gpt2,
    ModelArchitecture::GptNeoX,
    ModelArchitecture::Jais,
    ModelArchitecture::Jais2,
    ModelArchitecture::Mpt,
    ModelArchitecture::Nemotron,
    ModelArchitecture::Olmo,
    ModelArchitecture::Orion,
    ModelArchitecture::Phi2,
    ModelArchitecture::Rwkv6,
    ModelArchitecture::Rwkv7,
    ModelArchitecture::StableLm,
    ModelArchitecture::Starcoder,
    ModelArchitecture::Starcoder2,
];

#[test]
fn a_layernorm_architecture_maps_to_layernorm() {
    let wrong: Vec<&str> = LAYERNORM
        .iter()
        .filter(|a| norm_kind_for(**a) != NormKind::LayerNorm)
        .map(|a| a.as_str())
        .collect();
    assert!(
        wrong.is_empty(),
        "the loader serves these as RMSNorm: {wrong:?}"
    );
}

#[test]
fn an_rms_architecture_maps_to_rms() {
    // The other direction. A false positive is as wrong as a miss: a model
    // normalised with the wrong kind computes the wrong thing just as
    // reliably.
    for a in [
        ModelArchitecture::Llama,
        ModelArchitecture::Qwen2,
        ModelArchitecture::Mistral3,
        ModelArchitecture::DeepSeek2,
        ModelArchitecture::Gemma3,
        ModelArchitecture::Qwen3,
    ] {
        if !LAYERNORM.contains(&a) {
            assert_eq!(
                norm_kind_for(a),
                NormKind::Rms,
                "{} is not a LayerNorm model upstream",
                a.as_str()
            );
        }
    }
}

#[test]
fn every_layernorm_entry_agrees_with_grim_core() {
    // `uses_layernorm` and this list must not drift: the loader trusts the
    // former, and the list is what a reader checks by eye.
    for a in LAYERNORM {
        assert!(
            a.uses_layernorm(),
            "{} is in LAYERNORM but grim-core says otherwise",
            a.as_str()
        );
    }
}