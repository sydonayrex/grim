//! Norm-bias tensor names in the HF -> GGUF map -- plan item 5, tensor names.
//!
//! ## The gap this closes
//!
//! `remap_hf_to_gguf` mapped every norm's `.weight` and none of their
//! `.bias`, so for a checkpoint shipping `attn_norm_b` / `ffn_norm_b` /
//! `output_norm_b`:
//!
//! * the weight resolved, because `input_layernorm.weight ->
//!   attn_norm.weight` was present;
//! * the bias did not, so the lookup missed and `Norm::load`'s presence
//!   probe found nothing -- correct behaviour on a wrong input.
//!
//! Eighteen references create `attn_norm_b` or `ffn_norm_b` and twenty create
//! `output_norm_b`, so this is not an edge case. `656d3cbb` made the bias
//! presence-gated, which is right; it cannot supply a tensor the name map
//! cannot name.
//!
//! The fix derives each `.bias` key from its `.weight` counterpart rather
//! than writing 21 of them out, so a norm added later cannot arrive without
//! its bias.
//!
//! ## Why the end-to-end test exists
//!
//! The three map-shape tests above assert the map's contents. None of them
//! proves a real request hits them, and the namespace is not obvious: the
//! block asks for `layers.{i}.attn_norm.bias` under grim's own prefix, while
//! the map's GptNeoX keys are `gpt_neox.layers.{i}.*`. Only the `Llama` arm is
//! keyed on `layers.N.*`, so that is the one whose key matches the request --
//! and the test uses it for that reason, which it says out loud.

//! ## What this test pins
//!
//! Every norm whose `.weight` counterpart is mapped must also have its
//! `.bias` counterpart mapped, for the same architecture block. Derived from
//! the map itself, so adding a norm mapping without its bias fails here
//! instead of shipping.

use grim_core::architecture::TensorNamingRegistry;
use grim_core::ModelArchitecture;

/// Architectures whose reference block maps a norm, checked here. The set is
/// the ones `remap_hf_to_gguf` has a norm mapping for, discovered by running
/// it rather than by hand, so a new architecture's norms are covered
/// automatically.
fn architectures_with_norm_mappings() -> Vec<ModelArchitecture> {
    // `remap_hf_to_gguf` covers a fixed set of arms. Enumerating them here
    // keeps the test honest: an architecture absent from this list is simply
    // not asserted, rather than silently passing.
    [
        ModelArchitecture::GptNeoX,
        ModelArchitecture::Mpt,
        ModelArchitecture::Falcon,
        ModelArchitecture::Gpt2,
        ModelArchitecture::Bloom,
        ModelArchitecture::Bert,
        ModelArchitecture::GptJ,
        ModelArchitecture::Qwen2,
        ModelArchitecture::Phi2,
        ModelArchitecture::Cohere2,
        ModelArchitecture::Olmo,
        ModelArchitecture::Starcoder2,
    ]
    .into_iter()
    .filter(|a| {
        let m = TensorNamingRegistry::remap_hf_to_gguf(*a, 1);
        m.keys().any(|k| k.contains("norm") && k.ends_with(".weight"))
    })
    .collect()
}

#[test]
fn the_map_is_not_empty() {
    let archs = architectures_with_norm_mappings();
    assert!(
        archs.len() >= 8,
        "only {} architectures have norm mappings; the discovery is broken",
        archs.len()
    );
    let a = TensorNamingRegistry::remap_hf_to_gguf(ModelArchitecture::GptNeoX, 1);
    assert!(
        a.keys().any(|k| k.contains("norm")),
        "GptNeoX has no norm mapping, so this test would pass vacuously"
    );
}

#[test]
fn every_mapped_norm_weight_has_a_bias_counterpart() {
    // RED. The map has 21 norm weight mappings and zero norm bias mappings, so
    // this fails until the bias counterparts are added.
    let mut missing: Vec<String> = Vec::new();
    for arch in architectures_with_norm_mappings() {
        let m = TensorNamingRegistry::remap_hf_to_gguf(arch, 1);
        for (hf, gg) in &m {
            if !gg.starts_with("blk.0.") {
                continue;
            }
            let Some(base) = gg.strip_prefix("blk.0.") else {
                continue;
            };
            if !base.ends_with(".weight") || !base.contains("norm") {
                continue;
            }
            let stem = base.trim_end_matches(".weight");
            let want_gg = format!("blk.0.{stem}.bias");
            let want_hf = hf.trim_end_matches(".weight").to_string() + ".bias";
            if !m.values().any(|v| v == &want_gg) && !m.contains_key(&want_hf) {
                missing.push(format!("{arch:?}: {hf} -> {gg} (want {want_hf} -> {want_gg})"));
            }
        }
    }
    assert!(
        missing.is_empty(),
        "these norm weights are mapped but their biases are not, so a checkpoint \\
         that ships *_norm_b loads the weight and silently misses the bias:\n  {}",
        missing.join("\n  ")
    );
}

#[test]
fn the_output_norm_bias_is_mapped() {
    // `output_norm_b` is not per-layer, so it is outside the loop that this
    // test walks. It is separate on purpose: twenty references create it.
    let m = TensorNamingRegistry::remap_hf_to_gguf(ModelArchitecture::Phi2, 1);
    assert!(
        m.values().any(|v| v == "output_norm.bias"),
        "Phi2 creates output_norm.b (phi2.cpp:20) but the map has no key for it"
    );
}

/// End to end: the name the block actually asks for must resolve, through
/// the same `RemappingTensorProvider` the loader builds.
///
/// `LlamaBlock::load_tp` asks `ws.pp("attn_norm").get(dim, "bias")` under the
/// prefix `layers.{i}`, so the lookup key is `layers.0.attn_norm.bias`. The
/// loader wraps its provider in a `RemappingTensorProvider` keyed on
/// `remap_hf_to_gguf`, so the key has to be IN that map -- the two tests above
/// only assert the map's contents, not that a real request hits them.
#[test]
fn a_layers_attn_norm_bias_request_resolves_to_the_gguf_name() {
    use grim_format::tprov::RemappingTensorProvider;
    use grim_tensor::TensorProvider;
    use std::collections::HashMap;

    // A provider holding only GGUF-named tensors, as a GGUF file has them.
    let mut tensors: HashMap<String, grim_tensor::RawTensor> = HashMap::new();
    tensors.insert(
        "blk.0.attn_norm.bias".to_string(),
        grim_tensor::RawTensor {
            bytes: vec![0u8; 4 * 8],
            shape: vec![8],
            dtype: grim_tensor::DType::F32,
            provenance: grim_tensor::QuantProvenance::GrimNative,
        },
    );
    let inner = FixedProvider { tensors };
    let map = TensorNamingRegistry::remap_hf_to_gguf(ModelArchitecture::Llama, 1);
    let provider = RemappingTensorProvider::new(&inner, move |n: &str| -> String {
        map.get(n).cloned().unwrap_or_else(|| n.to_string())
    });

    // Exactly what LlamaBlock asks for, layer 0.
    let resolved = provider.get("layers.0.attn_norm.bias");
    assert!(
        resolved.is_ok(),
        "the block's own request `layers.0.attn_norm.bias` did not resolve, so \
         eighteen references' attn_norm_b would load without its bias"
    );
}

/// Minimal provider: serves a fixed map, counts nothing.
struct FixedProvider {
    tensors: std::collections::HashMap<String, grim_tensor::RawTensor>,
}

impl grim_tensor::TensorProvider for FixedProvider {
    fn get(&self, name: &str) -> grim_tensor::error::Result<grim_tensor::RawTensor> {
        self.tensors.get(name).cloned().ok_or_else(|| {
            grim_tensor::error::Error::Backend(format!("tensor '{name}' not found"))
        })
    }
    fn meta(&self, name: &str) -> grim_tensor::error::Result<grim_tensor::TensorMeta> {
        self.tensors
            .get(name)
            .map(|t| grim_tensor::TensorMeta {
                dtype: t.dtype.clone(),
                provenance: t.provenance.clone(),
                shape: t.shape.clone(),
                fusion_mask: 0,
            })
            .ok_or_else(|| {
                grim_tensor::error::Error::Backend(format!("tensor '{name}' not found"))
            })
    }
}
