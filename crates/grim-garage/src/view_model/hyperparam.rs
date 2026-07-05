//! Hyperparam form — the data the hyperparameters panel binds to, in
//! the exact shape the CVKG `Form`/`Input` widgets consume.

use crate::ui_state::UiTrainingConfig;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HyperparamFormV1 {
    /// Display name of the form (used as the panel title).
    pub form_id: String,

    /// The training mode the user picked in the dropdown.
    /// `"LoRA"`, `"QLoRA"`, or `"Bf16-Full"`.
    pub training_mode: String,

    /// Quantization format the model will be dequantized to during training.
    /// Only meaningful when `training_mode == QLoRA`; UI hides the picker otherwise.
    pub quant_format: String,

    /// One of `[8, 16, 32, 64]`. Snapped to the nearest valid value by `normalized()`.
    pub lora_rank: u32,

    /// Initial learning rate.
    pub learning_rate: f64,

    /// Number of epochs to train (display only; backend maps to max_steps).
    pub epochs: u32,

    pub rocm_fusion_rmsnorm_matmul: bool,
    pub rocm_fusion_qkv_attention: bool,
    pub auto_wavefront: bool,
    pub xnack_enabled: bool,
}

impl Default for HyperparamFormV1 {
    fn default() -> Self {
        Self {
            form_id: "hyperparameters".into(),
            training_mode: "LoRA".into(),
            quant_format: "Q4_K".into(),
            lora_rank: 16,
            learning_rate: 2e-5,
            epochs: 1,
            rocm_fusion_rmsnorm_matmul: true,
            rocm_fusion_qkv_attention: false,
            auto_wavefront: true,
            xnack_enabled: false,
        }
    }
}

impl HyperparamFormV1 {
    /// Construct the form from the live UI state.
    pub fn from_training_config(c: UiTrainingConfig) -> Self {
        Self {
            form_id: "hyperparameters".into(),
            training_mode: c.training_mode,
            quant_format: c.quant_format,
            lora_rank: c.lora_rank,
            learning_rate: c.learning_rate,
            epochs: c.epochs,
            rocm_fusion_rmsnorm_matmul: c.rocm_fusion_rmsnorm_matmul,
            rocm_fusion_qkv_attention: c.rocm_fusion_qkv_attention,
            auto_wavefront: c.auto_wavefront,
            xnack_enabled: c.xnack_enabled,
        }
    }

    /// Allowed LoRA ranks; the form picker offers exactly these values.
    /// Doubling is intentional: each tier roughly doubles VRAM overhead.
    pub const VALID_LORA_RANKS: &'static [u32] = &[8, 16, 32, 64];

    /// Snap `lora_rank` to the nearest allowed tier. Used when the
    /// user types a custom value or moves between modes.
    pub fn normalized(mut self) -> Self {
        self.lora_rank = pick_closest(self.lora_rank, Self::VALID_LORA_RANKS);
        self
    }
}

fn pick_closest(value: u32, choices: &[u32]) -> u32 {
    *choices
        .iter()
        .min_by_key(|&&c| (c as i64 - value as i64).abs())
        .expect("non-empty choices")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalized_with_rank(rank: u32) -> HyperparamFormV1 {
        let mut f = HyperparamFormV1::default();
        f.lora_rank = rank;
        f.normalized()
    }

    #[test]
    fn form_default_matches_v1_layout() {
        let f = HyperparamFormV1::default();
        assert_eq!(f.form_id, "hyperparameters");
        assert_eq!(f.training_mode, "LoRA");
        assert_eq!(f.quant_format, "Q4_K");
        assert_eq!(f.lora_rank, 16);
    }

    #[test]
    fn valid_lora_ranks_are_exactly_8_16_32_64() {
        assert_eq!(HyperparamFormV1::VALID_LORA_RANKS, &[8, 16, 32, 64]);
    }

    #[test]
    fn normalize_snap_to_nearest_rank_below() {
        let f = normalized_with_rank(10); // closer to 8 than to 16? |16-10|=6, |8-10|=2 -> 8
        assert_eq!(f.lora_rank, 8);
    }

    #[test]
    fn normalize_snap_to_nearest_rank_above() {
        let f = normalized_with_rank(20); // |20-16|=4, |20-32|=12 -> 16
        assert_eq!(f.lora_rank, 16);
    }

    #[test]
    fn normalize_keeps_exact_match_unchanged() {
        let f = normalized_with_rank(32);
        assert_eq!(f.lora_rank, 32);
    }
}
