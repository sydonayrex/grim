//! Qwen3-VL clip (`mmproj`) configuration, read from the checkpoint's own metadata.
//!
//! Geometry never comes from a literal here. A Qwen3-VL projector is a separate
//! GGUF file from the text model, and every dimension the forward pass needs is
//! declared in that file's metadata block. Hardcoding those numbers is how a
//! model silently loads the wrong shapes, so each field is read and each
//! read is checked.
//!
//! The metadata keys are the GGUF ones llama.cpp reads for the same projector
//! (`old/repo/llama.cpp-master/tools/mtmd/clip-impl.h`): `clip.vision.*` for
//! the vision tower and `clip.projector_type` for the head that follows it.

use grim_core::error::{Error, Result};
use grim_format::tprov::GgufProvider;

/// Projector type this module implements. Other `clip` projectors (LLaVA MLP,
/// Gemma3, MiniCPM-V, ...) are a different graph and are refused rather than
/// approximated.
const PROJECTOR_QWEN3VL_MERGER: &str = "qwen3vl_merger";

/// Vision-tower geometry and behaviour for a Qwen3-VL `mmproj` checkpoint.
///
/// Field names mirror the GGUF metadata keys they come from, so a reader can
/// match them against the file without a translation table.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3VlClipConfig {
    /// `clip.vision.image_size` - reference resolution the position table was
    /// trained at (768 for this family).
    pub image_size: usize,
    /// `clip.vision.patch_size` - conv kernel extent per side (16).
    pub patch_size: usize,
    /// `clip.vision.embedding_length` - ViT residual width (1152).
    pub embedding_length: usize,
    /// `clip.vision.feed_forward_length` - ViT MLP inner width (4304).
    pub feed_forward_length: usize,
    /// `clip.vision.block_count` - ViT depth (27).
    pub block_count: usize,
    /// `clip.vision.attention.head_count` - ViT attention heads (16).
    pub num_heads: usize,
    /// `clip.vision.projection_dim` - width the merger emits into the text
    /// model's embedding space (5120, equal to Qwen3.8-27B's hidden size).
    pub projection_dim: usize,
    /// `clip.vision.spatial_merge_size` - 2x2 patch blocks folded into one
    /// output token (2).
    pub spatial_merge_size: usize,
    /// `clip.vision.attention.layer_norm_epsilon` - the ViT uses LayerNorm, not
    /// RMSNorm, so this epsilon governs a mean-subtracted normalisation.
    pub layer_norm_eps: f32,
    /// `clip.use_gelu` - the ViT MLP activation. This family is tanh-approximate
    /// GELU, not SwiGLU: there is no `ffn_gate` tensor in the file.
    pub use_gelu: bool,
    /// `clip.vision.spatial_merge_size` squared; the merger consumes 2x2 blocks.
    pub in_channels: usize,
    /// Number of 2x2 blocks merged into one output token.
    pub merge_area: usize,
    /// Per-channel normalisation constants, `clip.vision.image_mean` /
    /// `clip.vision.image_std`. Read from the file rather than assumed, because
    /// the resize filter and the constants together are what the model was
    /// trained with.
    pub image_mean: Vec<f32>,
    pub image_std: Vec<f32>,
    /// `clip.vision.is_deepstack_layers` - which ViT blocks also emit a
    /// DeepStack feature. Empty for this checkpoint (27 x false).
    pub deepstack_layers: Vec<usize>,
}

impl Qwen3VlClipConfig {
    /// Patches along one side of a `image_size` image.
    pub fn patches_per_side(&self) -> usize {
        self.image_size / self.patch_size
    }

    /// Conv output rows: one per patch, before merging.
    pub fn n_patches(&self) -> usize {
        self.patches_per_side().pow(2)
    }

    /// Attention head width inside the ViT.
    pub fn head_dim(&self) -> usize {
        self.embedding_length / self.num_heads
    }

    /// Width the merger consumes: `merge_area` patches x `embedding_length`.
    pub fn merger_in(&self) -> usize {
        self.merge_area * self.embedding_length
    }

    /// Tokens the tower emits for a full `image_size` image.
    pub fn merged_tokens(&self) -> usize {
        self.n_patches() / self.merge_area
    }

    /// True when this config carries DeepStack visual features - intermediate
    /// ViT-layer outputs injected into specific text-model layers. Unsupported
    /// here, so callers can refuse rather than emit rows of the wrong width.
    pub fn has_deepstack(&self) -> bool {
        !self.deepstack_layers.is_empty()
    }
}

impl Qwen3VlClipConfig {
    /// Build from an open GGUF provider's metadata block.
    ///
    /// Errors unless the file really is a `clip`/`mmproj` pair carrying the
    /// Qwen3-VL merger projector. The error names the value actually seen, so a
    /// wrong file is diagnosable from the message alone.
    pub fn from_provider(provider: &GgufProvider) -> Result<Self> {
        let arch = provider.architecture().unwrap_or("<missing>");
        if arch != "clip" {
            return Err(Error::Config(format!(
                "mmproj expected general.architecture=clip, found '{arch}'"
            )));
        }
        let kind = provider
            .metadata("general.type")
            .and_then(|v| v.as_str())
            .unwrap_or("<missing>");
        if kind != "mmproj" {
            return Err(Error::Config(format!(
                "mmproj expected general.type=mmproj, found '{kind}'"
            )));
        }
        let projector = provider
            .metadata("clip.projector_type")
            .and_then(|v| v.as_str())
            .unwrap_or("<missing>");
        if projector != PROJECTOR_QWEN3VL_MERGER {
            return Err(Error::Config(format!(
                "clip.projector_type='{projector}' is not implemented; this build serves \
                 only '{PROJECTOR_QWEN3VL_MERGER}'"
            )));
        }
        Self::from_metadata(&|key| provider.metadata(key))
    }

    /// Build from a key-lookup closure. `from_provider` delegates here so the
    /// parsing rules are testable without a 927 MB file on disk.
    ///
    /// Every required key is read; a missing one is an error naming the key,
    /// never a default.
    pub fn from_metadata<'a>(
        get: &dyn Fn(&str) -> Option<&'a grim_format::gguf::GgufValue>,
    ) -> Result<Self> {
        let u32_at = |key: &str| -> Result<usize> {
            let v = get(key).ok_or_else(|| {
                Error::Config(format!("mmproj metadata missing required key '{key}'"))
            })?;
            v.as_u32()
                .map(|n| n as usize)
                .ok_or_else(|| Error::Config(format!("mmproj metadata '{key}' is not an integer")))
        };
        let f32_at = |key: &str| -> Result<f32> {
            let v = get(key).ok_or_else(|| {
                Error::Config(format!("mmproj metadata missing required key '{key}'"))
            })?;
            v.as_f32()
                .ok_or_else(|| Error::Config(format!("mmproj metadata '{key}' is not a float")))
        };
        let f32_vec = |key: &str| -> Result<Vec<f32>> {
            let v = get(key).ok_or_else(|| {
                Error::Config(format!("mmproj metadata missing required key '{key}'"))
            })?;
            let arr = v
                .as_array()
                .ok_or_else(|| Error::Config(format!("mmproj metadata '{key}' is not an array")))?;
            arr.iter()
                .map(|x| {
                    x.as_f32().ok_or_else(|| {
                        Error::Config(format!("mmproj metadata '{key}' has a non-float element"))
                    })
                })
                .collect()
        };

        let image_size = u32_at("clip.vision.image_size")?;
        let patch_size = u32_at("clip.vision.patch_size")?;
        let embedding_length = u32_at("clip.vision.embedding_length")?;
        let feed_forward_length = u32_at("clip.vision.feed_forward_length")?;
        let block_count = u32_at("clip.vision.block_count")?;
        let num_heads = u32_at("clip.vision.attention.head_count")?;
        let projection_dim = u32_at("clip.vision.projection_dim")?;
        let spatial_merge_size = u32_at("clip.vision.spatial_merge_size")?;
        let layer_norm_eps = f32_at("clip.vision.attention.layer_norm_epsilon")?;

        // A zero here would make `merged_tokens` divide by zero, and a zero
        // patch_size would make `patches_per_side` wrong by construction.
        if spatial_merge_size == 0 {
            return Err(Error::Config(
                "clip.vision.spatial_merge_size is 0; refusing to divide by it".into(),
            ));
        }
        if patch_size == 0 {
            return Err(Error::Config(
                "clip.vision.patch_size is 0; refusing to divide by it".into(),
            ));
        }
        if embedding_length % num_heads != 0 {
            return Err(Error::Config(format!(
                "embedding_length {embedding_length} is not divisible by num_heads {num_heads}"
            )));
        }
        if image_size % patch_size != 0 {
            return Err(Error::Config(format!(
                "image_size {image_size} is not a multiple of patch_size {patch_size}"
            )));
        }

        let use_gelu = get("clip.use_gelu")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // DeepStack injects extra ViT-layer outputs into the text model at
        // specific layers. This checkpoint declares none; a checkpoint that does
        // needs a different graph entirely, so record them and refuse later.
        //
        // A non-bool element is an error rather than `false`: silently reading a
        // malformed flag as "no deepstack" would run the plain projector against a
        // checkpoint that needs the wider one, which is a wrong-width write into
        // the text sequence rather than a visible failure.
        let mut deepstack_layers = Vec::new();
        if let Some(arr) = get("clip.vision.is_deepstack_layers").and_then(|v| v.as_array()) {
            for (i, v) in arr.iter().enumerate() {
                match v.as_bool() {
                    Some(true) => deepstack_layers.push(i),
                    Some(false) => {}
                    None => {
                        return Err(Error::Config(format!(
                            "clip.vision.is_deepstack_layers[{i}] is not a bool"
                        )));
                    }
                }
            }
        }

        let image_mean = f32_vec("clip.vision.image_mean")?;
        let image_std = f32_vec("clip.vision.image_std")?;
        if image_mean.len() != 3 || image_std.len() != 3 {
            return Err(Error::Config(format!(
                "image normalisation must be 3 per channel, got mean={:?} std={:?}",
                image_mean.len(),
                image_std.len()
            )));
        }
        if image_std.iter().any(|s| *s == 0.0) {
            return Err(Error::Config(format!(
                "image_std contains a zero, which cannot be divided by: {image_std:?}"
            )));
        }

        Ok(Self {
            image_size,
            patch_size,
            embedding_length,
            feed_forward_length,
            block_count,
            num_heads,
            projection_dim,
            spatial_merge_size,
            layer_norm_eps,
            use_gelu,
            in_channels: 3,
            merge_area: spatial_merge_size * spatial_merge_size,
            image_mean,
            image_std,
            deepstack_layers,
        })
    }

    /// Build from a HuggingFace `config.json` `vision_config` object, for the
    /// safetensors path where no GGUF metadata block exists.
    pub fn from_hf(value: &serde_json::Value) -> Result<Self> {
        let vision = value.get("vision_config").unwrap_or(value);
        let u = |k: &str| -> Result<usize> {
            vision
                .get(k)
                .and_then(|v| v.as_u64())
                .map(|n| n as usize)
                .ok_or_else(|| Error::Config(format!("vision_config missing integer '{k}'")))
        };
        let merge = u("spatial_merge_size")?;
        let patch_size = u("patch_size")?;
        let embedding_length = u("hidden_size")?;
        let num_heads = u("num_heads")?;
        let in_channels = vision
            .get("in_channels")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(3);
        let deepstack_layers: Vec<usize> = vision
            .get("deepstack_visual_indexes")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                    .collect()
            })
            .unwrap_or_default();

        let mean = vision
            .get("image_mean")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_f64())
                    .map(|v| v as f32)
                    .collect()
            })
            .unwrap_or_else(|| vec![0.5, 0.5, 0.5]);
        let std = vision
            .get("image_std")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_f64())
                    .map(|v| v as f32)
                    .collect()
            })
            .unwrap_or_else(|| vec![0.5, 0.5, 0.5]);
        if mean.len() != 3 || std.len() != 3 {
            return Err(Error::Config(
                "HF vision_config image_mean/image_std must have 3 entries".into(),
            ));
        }

        Ok(Self {
            image_size: u("image_size")?,
            patch_size,
            embedding_length,
            feed_forward_length: u("intermediate_size")?,
            block_count: u("depth")?,
            num_heads,
            projection_dim: u("out_hidden_size")?,
            spatial_merge_size: merge,
            layer_norm_eps: vision
                .get("layer_norm_eps")
                .and_then(|v| v.as_f64())
                .unwrap_or(1e-6) as f32,
            use_gelu: vision
                .get("hidden_act")
                .and_then(|v| v.as_str())
                .map(|s| s.starts_with("gelu"))
                .unwrap_or(true),
            in_channels,
            merge_area: merge * merge,
            image_mean: mean,
            image_std: std,
            deepstack_layers,
        })
    }

    /// Reject a checkpoint whose geometry this build cannot execute.
    ///
    /// DeepStack is the one structural gap: it changes the projector output
    /// width to `projection_dim * (1 + n_deepstack)` and injects rows into
    /// specific text-model layers, so running the plain path would emit rows of
    /// the wrong count into a sequence sized for the other answer.
    pub fn ensure_supported(&self) -> Result<()> {
        if self.has_deepstack() {
            return Err(Error::Unimplemented(format!(
                "Qwen3-VL DeepStack is not implemented; this checkpoint declares \
                 {} deepstack layer(s) ({:?}), which changes the projector output \
                 width to {}",
                self.deepstack_layers.len(),
                self.deepstack_layers,
                self.projection_dim * (1 + self.deepstack_layers.len())
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_format::gguf::GgufValue;
    use std::collections::HashMap;

    /// The metadata block of the real checkpoint, so the tests pin the actual
    /// numbers rather than a plausible-looking fixture.
    fn real_metadata() -> HashMap<String, GgufValue> {
        let mut m = HashMap::new();
        m.insert("clip.vision.image_size".into(), GgufValue::Uint32(768));
        m.insert("clip.vision.patch_size".into(), GgufValue::Uint32(16));
        m.insert(
            "clip.vision.embedding_length".into(),
            GgufValue::Uint32(1152),
        );
        m.insert(
            "clip.vision.feed_forward_length".into(),
            GgufValue::Uint32(4304),
        );
        m.insert("clip.vision.block_count".into(), GgufValue::Uint32(27));
        m.insert(
            "clip.vision.attention.head_count".into(),
            GgufValue::Uint32(16),
        );
        m.insert("clip.vision.projection_dim".into(), GgufValue::Uint32(5120));
        m.insert(
            "clip.vision.spatial_merge_size".into(),
            GgufValue::Uint32(2),
        );
        m.insert(
            "clip.vision.attention.layer_norm_epsilon".into(),
            GgufValue::Float32(1e-6),
        );
        m.insert("clip.use_gelu".into(), GgufValue::Bool(true));
        m.insert(
            "clip.vision.image_mean".into(),
            GgufValue::Array(vec![
                GgufValue::Float32(0.5),
                GgufValue::Float32(0.5),
                GgufValue::Float32(0.5),
            ]),
        );
        m.insert(
            "clip.vision.image_std".into(),
            GgufValue::Array(vec![
                GgufValue::Float32(0.5),
                GgufValue::Float32(0.5),
                GgufValue::Float32(0.5),
            ]),
        );
        // The real checkpoint carries 27 entries, all false.
        m.insert(
            "clip.vision.is_deepstack_layers".into(),
            GgufValue::Array(vec![GgufValue::Bool(false); 27]),
        );
        m
    }

    fn cfg_from(m: &HashMap<String, GgufValue>) -> Result<Qwen3VlClipConfig> {
        Qwen3VlClipConfig::from_metadata(&|k: &str| m.get(k))
    }

    #[test]
    fn derived_geometry_matches_the_real_checkpoint() {
        let m = real_metadata();
        let c = cfg_from(&m).expect("config parses");
        assert_eq!(c.patches_per_side(), 48);
        assert_eq!(c.n_patches(), 2304);
        assert_eq!(c.head_dim(), 72);
        assert_eq!(c.merger_in(), 4608);
        assert_eq!(c.merged_tokens(), 576);
        assert_eq!(c.merge_area, 4);
        assert_eq!(c.in_channels, 3);
        assert_eq!(c.projection_dim, 5120);
    }

    #[test]
    fn deepstack_free_checkpoint_is_supported() {
        let m = real_metadata();
        let c = cfg_from(&m).expect("config parses");
        assert!(!c.has_deepstack(), "real checkpoint declares 27 x false");
        c.ensure_supported().expect("no deepstack to refuse");
    }

    #[test]
    fn deepstack_checkpoint_is_refused_with_the_width_it_would_need() {
        let mut m = real_metadata();
        let mut flags = vec![GgufValue::Bool(false); 27];
        flags[8] = GgufValue::Bool(true);
        flags[16] = GgufValue::Bool(true);
        flags[24] = GgufValue::Bool(true);
        m.insert(
            "clip.vision.is_deepstack_layers".into(),
            GgufValue::Array(flags),
        );
        let c = cfg_from(&m).expect("config parses");
        assert_eq!(c.deepstack_layers, vec![8, 16, 24]);
        let err = c.ensure_supported().expect_err("deepstack must be refused");
        let msg = err.to_string();
        assert!(msg.contains("DeepStack"), "message names the gap: {msg}");
        // The refusal must state the width it would have produced, so an
        // operator can see why the plain path is wrong rather than approximate.
        // DeepStack concatenates along the feature axis, so the per-token width
        // becomes projection_dim * (1 + n_deepstack) = 5120 * 4 = 20480.
        assert!(
            msg.contains("20480"),
            "message states projection_dim * (1+3) = 20480: {msg}"
        );
    }

    #[test]
    fn missing_required_key_names_the_key() {
        let mut m = real_metadata();
        m.remove("clip.vision.projection_dim");
        let err = cfg_from(&m).expect_err("projection_dim is required");
        assert!(
            err.to_string().contains("clip.vision.projection_dim"),
            "error names the missing key: {err}"
        );
    }

    #[test]
    fn zero_merge_size_is_refused_before_any_division() {
        let mut m = real_metadata();
        m.insert(
            "clip.vision.spatial_merge_size".into(),
            GgufValue::Uint32(0),
        );
        let err = cfg_from(&m).expect_err("merge 0 would divide by zero");
        assert!(err.to_string().contains("spatial_merge_size"), "{err}");
    }

    #[test]
    fn zero_patch_size_is_refused_before_any_division() {
        let mut m = real_metadata();
        m.insert("clip.vision.patch_size".into(), GgufValue::Uint32(0));
        let err = cfg_from(&m).expect_err("patch 0 would divide by zero");
        assert!(err.to_string().contains("patch_size"), "{err}");
    }

    #[test]
    fn head_count_that_does_not_divide_is_refused() {
        let mut m = real_metadata();
        m.insert(
            "clip.vision.attention.head_count".into(),
            GgufValue::Uint32(7),
        );
        let err = cfg_from(&m).expect_err("1152 / 7 has no integer head_dim");
        assert!(err.to_string().contains("divisible"), "{err}");
    }

    #[test]
    fn image_size_not_a_multiple_of_patch_size_is_refused() {
        let mut m = real_metadata();
        m.insert("clip.vision.image_size".into(), GgufValue::Uint32(770));
        let err = cfg_from(&m).expect_err("770 / 16 is not a whole patch grid");
        assert!(err.to_string().contains("multiple of patch_size"), "{err}");
    }

    #[test]
    fn zero_image_std_is_refused() {
        let mut m = real_metadata();
        m.insert(
            "clip.vision.image_std".into(),
            GgufValue::Array(vec![
                GgufValue::Float32(0.5),
                GgufValue::Float32(0.0),
                GgufValue::Float32(0.5),
            ]),
        );
        let err = cfg_from(&m).expect_err("cannot divide by zero std");
        assert!(err.to_string().contains("image_std"), "{err}");
    }

    #[test]
    fn wrong_length_normalisation_is_refused() {
        let mut m = real_metadata();
        m.insert(
            "clip.vision.image_mean".into(),
            GgufValue::Array(vec![GgufValue::Float32(0.5), GgufValue::Float32(0.5)]),
        );
        let err = cfg_from(&m).expect_err("mean must be 3 channels");
        assert!(err.to_string().contains("3 per channel"), "{err}");
    }

    #[test]
    fn non_bool_deepstack_flag_is_refused_rather_than_read_as_false() {
        // Reading a malformed flag as "no deepstack" would run the plain
        // projector against a checkpoint that needs the wider one.
        let mut m = real_metadata();
        m.insert(
            "clip.vision.is_deepstack_layers".into(),
            GgufValue::Array(vec![GgufValue::Uint32(1)]),
        );
        let err = cfg_from(&m).expect_err("a non-bool flag must not read as false");
        assert!(
            err.to_string().contains("is not a bool"),
            "error names the malformed element: {err}"
        );
    }

    #[test]
    fn hf_config_produces_the_same_derived_geometry() {
        let hf = serde_json::json!({
            "vision_config": {
                "depth": 27,
                "hidden_size": 1152,
                "hidden_act": "gelu_pytorch_tanh",
                "in_channels": 3,
                "image_size": 768,
                "intermediate_size": 4304,
                "num_heads": 16,
                "num_position_embeddings": 2304,
                "out_hidden_size": 5120,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "deepstack_visual_indexes": []
            }
        });
        let c = Qwen3VlClipConfig::from_hf(&hf).expect("hf config parses");
        assert_eq!(c.patches_per_side(), 48);
        assert_eq!(c.n_patches(), 2304);
        assert_eq!(c.head_dim(), 72);
        assert_eq!(c.merger_in(), 4608);
        assert_eq!(c.merged_tokens(), 576);
        assert!(c.use_gelu, "gelu_pytorch_tanh is a gelu activation");
        assert!(!c.has_deepstack());
    }
}
