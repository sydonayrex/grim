//! LoRA injection point configuration (WI-T1 item 1).
//!
//! Forward-side prerequisite: extend LoRA application from logits-only to
//! standard QLoRA injection points (attention Q/K/V/O + MLP Gate/Up/Down).
//! The backward graph needs to match wherever forward adapters get applied,
//! so the injection-point enumeration lives here.

use crate::ParamId;
use serde::{Deserialize, Serialize};

/// Standard LoRA injection points for QLoRA parity with Unsloth.
///
/// Unsloth applies LoRA to all attention projections (Q/K/V/O) and MLP
/// projections (Gate/Up/Down) — 7 injection points per layer. The legacy
/// `Logits` injection point (the only one wired in `lora.rs` today) is kept
/// for backwards compatibility but is **not sufficient for real QLoRA parity**
/// per the plan's WI-T1 note.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LoRAInjectionPoint {
    /// Query projection in attention (W_q).
    QProj,
    /// Key projection in attention (W_k).
    KProj,
    /// Value projection in attention (W_v).
    VProj,
    /// Output projection in attention (W_o).
    OProj,
    /// Gate projection in MLP (SwiGLU gate).
    GateProj,
    /// Up projection in MLP (SwiGLU up).
    UpProj,
    /// Down projection in MLP (SwiGLU down).
    DownProj,
    /// Output logits projection — legacy/logits-only LoRA (the only path
    /// wired in `lora.rs::apply_adapters_to_logits` at the time of this plan).
    Logits,
}

impl LoRAInjectionPoint {
    /// All standard QLoRA injection points (7 total, matching Unsloth).
    /// `Logits` is intentionally excluded — it is not a standard QLoRA site.
    pub fn all_standard_qlora() -> &'static [Self] {
        &[
            Self::QProj,
            Self::KProj,
            Self::VProj,
            Self::OProj,
            Self::GateProj,
            Self::UpProj,
            Self::DownProj,
        ]
    }

    /// Attention projections only (Q/K/V/O — 4 points).
    pub fn attention_only() -> &'static [Self] {
        &[Self::QProj, Self::KProj, Self::VProj, Self::OProj]
    }

    /// MLP projections only (Gate/Up/Down — 3 points).
    pub fn mlp_only() -> &'static [Self] {
        &[Self::GateProj, Self::UpProj, Self::DownProj]
    }

    /// Weight tensor name suffix at this injection point (matches the block.rs naming).
    pub fn weight_suffix(&self) -> &'static str {
        match self {
            Self::QProj => "attn_q",
            Self::KProj => "attn_k",
            Self::VProj => "attn_v",
            Self::OProj => "attn_o",
            Self::GateProj => "ffn_gate",
            Self::UpProj => "ffn_up",
            Self::DownProj => "ffn_down",
            Self::Logits => "output",
        }
    }

    /// Adapter name prefix under which A/B weights live in a checkpoint, e.g. `blk.0.attn_q.lora`.
    pub fn adapter_prefix(&self, layer_idx: usize) -> String {
        format!("blk.{}.{}", layer_idx, self.weight_suffix())
    }

    /// `true` for Q/K/V/O.
    pub fn is_attention(&self) -> bool {
        matches!(self, Self::QProj | Self::KProj | Self::VProj | Self::OProj)
    }

    /// `true` for Gate/Up/Down.
    pub fn is_mlp(&self) -> bool {
        matches!(self, Self::GateProj | Self::UpProj | Self::DownProj)
    }

    /// Expected base-weight shape `(out_features, in_features)` for this
    /// injection point given the model geometry.
    pub fn base_weight_shape(
        &self,
        cfg: &InjectionConfig,
    ) -> (usize, usize) {
        match self {
            Self::QProj => (cfg.num_heads * cfg.head_dim, cfg.hidden_size),
            Self::KProj => (cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size),
            Self::VProj => (cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size),
            Self::OProj => (cfg.hidden_size, cfg.num_heads * cfg.head_dim),
            Self::GateProj => (cfg.intermediate_size, cfg.hidden_size),
            Self::UpProj => (cfg.intermediate_size, cfg.hidden_size),
            Self::DownProj => (cfg.hidden_size, cfg.intermediate_size),
            Self::Logits => (cfg.vocab_size, cfg.hidden_size),
        }
    }

    /// LoRA A shape `[rank, in_features]`.
    pub fn lora_a_shape(&self, cfg: &InjectionConfig, rank: usize) -> (usize, usize) {
        let (_, in_features) = self.base_weight_shape(cfg);
        (rank, in_features)
    }

    /// LoRA B shape `[out_features, rank]`.
    pub fn lora_b_shape(&self, cfg: &InjectionConfig, rank: usize) -> (usize, usize) {
        let (out_features, _) = self.base_weight_shape(cfg);
        (out_features, rank)
    }
}

/// Model geometry needed to size LoRA adapters per injection point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectionConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
}

/// Configuration for one LoRA adapter at a specific injection point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAInjectionConfig {
    pub injection_point: LoRAInjectionPoint,
    pub layer_idx: usize,
    pub adapter_id: u32,
    pub rank: usize,
    pub alpha: f32,
    pub enabled: bool,
}

impl LoRAInjectionConfig {
    pub fn new(
        injection_point: LoRAInjectionPoint,
        layer_idx: usize,
        adapter_id: u32,
        rank: usize,
        alpha: f32,
    ) -> Self {
        Self {
            injection_point,
            layer_idx,
            adapter_id,
            rank,
            alpha,
            enabled: true,
        }
    }

    /// Scaling factor `alpha / rank`.
    pub fn scale(&self) -> f32 {
        self.alpha / self.rank as f32
    }

    /// `ParamId` for this adapter's A matrix.
    pub fn param_id_a(&self) -> ParamId {
        ParamId::a(self.layer_idx, self.adapter_id)
    }

    /// `ParamId` for this adapter's B matrix.
    pub fn param_id_b(&self) -> ParamId {
        ParamId::b(self.layer_idx, self.adapter_id)
    }
}

/// Registry of all LoRA injection configs for a model.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LoRAInjectionRegistry {
    pub configs: std::collections::HashMap<(usize, LoRAInjectionPoint), LoRAInjectionConfig>,
}

impl LoRAInjectionRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, config: LoRAInjectionConfig) {
        self.configs.insert((config.layer_idx, config.injection_point), config);
    }

    pub fn get(&self, layer_idx: usize, point: LoRAInjectionPoint) -> Option<&LoRAInjectionConfig> {
        self.configs.get(&(layer_idx, point))
    }

    /// Build the standard 7-point-per-layer QLoRA registry.
    pub fn standard_qlora(num_layers: usize, rank: usize, alpha: f32, adapter_id: u32) -> Self {
        let mut r = Self::new();
        for layer_idx in 0..num_layers {
            for &point in LoRAInjectionPoint::all_standard_qlora() {
                r.add(LoRAInjectionConfig::new(point, layer_idx, adapter_id, rank, alpha));
            }
        }
        r
    }

    /// Build the attention-only (4-point) registry.
    pub fn attention_only(num_layers: usize, rank: usize, alpha: f32, adapter_id: u32) -> Self {
        let mut r = Self::new();
        for layer_idx in 0..num_layers {
            for &point in LoRAInjectionPoint::attention_only() {
                r.add(LoRAInjectionConfig::new(point, layer_idx, adapter_id, rank, alpha));
            }
        }
        r
    }

    /// Build the MLP-only (3-point) registry.
    pub fn mlp_only(num_layers: usize, rank: usize, alpha: f32, adapter_id: u32) -> Self {
        let mut r = Self::new();
        for layer_idx in 0..num_layers {
            for &point in LoRAInjectionPoint::mlp_only() {
                r.add(LoRAInjectionConfig::new(point, layer_idx, adapter_id, rank, alpha));
            }
        }
        r
    }

    /// All enabled configs.
    pub fn enabled(&self) -> Vec<&LoRAInjectionConfig> {
        self.configs.values().filter(|c| c.enabled).collect()
    }

    /// All configs for one layer.
    pub fn layer_configs(&self, layer_idx: usize) -> Vec<&LoRAInjectionConfig> {
        self.configs
            .iter()
            .filter(|((idx, _), _)| *idx == layer_idx)
            .map(|(_, c)| c)
            .collect()
    }

    /// Total trainable parameter count = Σ (|A| + |B|) over enabled configs.
    pub fn num_trainable_params(&self, cfg: &InjectionConfig) -> usize {
        self.configs
            .values()
            .filter(|c| c.enabled)
            .map(|c| {
                let (ar, ac) = c.injection_point.lora_a_shape(cfg, c.rank);
                let (br, bc) = c.injection_point.lora_b_shape(cfg, c.rank);
                ar * ac + br * bc
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> InjectionConfig {
        InjectionConfig {
            hidden_size: 4096,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 11008,
            vocab_size: 32000,
        }
    }

    #[test]
    fn standard_qlora_points_seven_attention_plus_mlp() {
        let pts = LoRAInjectionPoint::all_standard_qlora();
        assert_eq!(pts.len(), 7);
        assert!(pts.contains(&LoRAInjectionPoint::QProj));
        assert!(pts.contains(&LoRAInjectionPoint::DownProj));
        assert!(!pts.contains(&LoRAInjectionPoint::Logits));
    }

    #[test]
    fn attention_vs_mlp_classification() {
        assert!(LoRAInjectionPoint::QProj.is_attention());
        assert!(!LoRAInjectionPoint::QProj.is_mlp());
        assert!(LoRAInjectionPoint::GateProj.is_mlp());
        assert!(!LoRAInjectionPoint::GateProj.is_attention());
        assert!(!LoRAInjectionPoint::Logits.is_attention());
        assert!(!LoRAInjectionPoint::Logits.is_mlp());
    }

    #[test]
    fn base_weight_shapes_for_attention_and_mlp() {
        let c = cfg();
        assert_eq!(LoRAInjectionPoint::QProj.base_weight_shape(&c), (4096, 4096));
        assert_eq!(LoRAInjectionPoint::KProj.base_weight_shape(&c), (1024, 4096));
        assert_eq!(LoRAInjectionPoint::DownProj.base_weight_shape(&c), (4096, 11008));
        assert_eq!(LoRAInjectionPoint::Logits.base_weight_shape(&c), (32000, 4096));
    }

    #[test]
    fn lora_a_b_shape_inner_dimensions_match() {
        let c = cfg();
        let rank = 16;
        for point in LoRAInjectionPoint::all_standard_qlora() {
            let (ar, ac) = point.lora_a_shape(&c, rank); // [rank, in]
            let (br, bc) = point.lora_b_shape(&c, rank); // [out, rank]
            let (out_features, in_features) = point.base_weight_shape(&c);
            assert_eq!(ar, rank);
            assert_eq!(bc, rank);
            assert_eq!(ac, in_features);
            assert_eq!(br, out_features);
        }
    }

    #[test]
    fn registry_standard_qlora_covers_all_layers_and_points() {
        let r = LoRAInjectionRegistry::standard_qlora(4, 16, 32.0, 1);
        assert_eq!(r.configs.len(), 4 * 7);
        for layer in 0..4 {
            assert_eq!(r.layer_configs(layer).len(), 7);
        }
    }

    #[test]
    fn scale_is_alpha_over_rank() {
        let c = LoRAInjectionConfig::new(LoRAInjectionPoint::QProj, 0, 1, 16, 32.0);
        assert_eq!(c.scale(), 2.0);
    }

    #[test]
    fn num_trainable_params_matches_reference() {
        let r = LoRAInjectionRegistry::standard_qlora(1, 16, 32.0, 1);
        let c = cfg();
        // Per layer with 7 injection points, rank 16, hidden=4096, head_dim=128,
        //   num_heads=32, num_kv_heads=8, intermediate=11008:
        //   Q: (16*4096) + (4096*16)    = 131072
        //   K: (16*4096) + (1024*16)    = 81920
        //   V: same as K                = 81920
        //   O: (16*4096) + (4096*16)    = 131072
        //   Gate: (16*4096) + (11008*16)= 241664
        //   Up:   same as Gate           = 241664
        //   Down: (16*11008)+ (4096*16) = 241664
        //   total = 131072+81920*2+131072+241664*3 = 1,150,976
        let n = r.num_trainable_params(&c);
        assert_eq!(n, 1_150_976);
    }
}
