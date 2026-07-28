//! Training mode panel — derives visibility + options for the
//! mode dropdown AND the conditional quantization-form picker.

use super::hyperparam::HyperparamFormV1;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingPanelV1 {
    /// All training modes the dropdown exposes, in display order.
    pub mode_options: Vec<String>,
    /// Whether the quantization-format picker should be shown. Always
    /// hidden for Bf16-Full; always shown for QLoRA; defaults to hidden
    /// for LoRA but the user can still see it.
    pub show_quant_format_picker: bool,
    /// Quantization format options when the picker is shown.
    pub quant_format_options: Vec<String>,
    /// Quantization options to highlight when QLoRA is selected
    /// (more aggressive with VRAM, less with loss).
    pub qlora_recommended: Vec<String>,
    /// Card title the CVKG `Card` widget uses.
    pub panel_title: String,
    /// Optional helper text shown beneath the dropdown.
    pub help_text: String,
}

impl TrainingPanelV1 {
    pub fn from_form(form: &HyperparamFormV1) -> Self {
        let mode_options = vec![
            "LoRA".into(),
            "QLoRA".into(),
            "Bf16-Full".into(),
            "GRPO".into(),
            "DPO".into(),
            "ORPO".into(),
        ];
        // Conditional picker: shown for any mode that consumes quantized storage.
        let show_quant_format_picker = matches!(form.training_mode.as_str(), "QLoRA" | "LoRA");
        let quant_format_options = vec!["Q4_K".into(), "Q5_K".into(), "Q8_0".into()];
        let qlora_recommended = vec!["Q4_K".into(), "Q5_K".into()];
        let (panel_title, help_text) = match form.training_mode.as_str() {
            "LoRA" => (
                "Training mode: LoRA".into(),
                "Adapter rank is updated via the rank picker below; merge after training.".into(),
            ),
            "QLoRA" => (
                "Training mode: QLoRA (quantized)".into(),
                "Wider quant formats (Q5_K, Q8_0) preserve more signal but consume more VRAM.".into(),
            ),
            "Bf16-Full" => (
                "Training mode: BF16 full fine-tune".into(),
                "Materialization target is whatever the model ships as; no LoRA adapter.".into(),
            ),
            "GRPO" => (
                "Training mode: GRPO (RL)".into(),
                "Group Relative Policy Optimization for reinforcement learning.".into(),
            ),
            "DPO" => (
                "Training mode: DPO (RL)".into(),
                "Direct Preference Optimization for preference alignment.".into(),
            ),
            "ORPO" => (
                "Training mode: ORPO (RL)".into(),
                "Odds-Ratio Preference Optimization for joint SFT and alignment.".into(),
            ),
            // L4: previously this arm silently masked typos / stale
            // configs as BF16. There is no fallback family for "unknown"
            // — the panel surfaces the unrecognized string verbatim and
            // tells the operator to fix the config. Pre-fix, the
            // wildcards arm quietly picked BF16 and the quant picker
            // hid defective modes.
            other => (
                format!("Training mode: unknown ({other})"),
                "Selected training_mode is not recognized; pick one from the dropdown. Quantization picker hidden until fixed.".into(),
            ),
        };
        Self {
            mode_options,
            show_quant_format_picker,
            quant_format_options,
            qlora_recommended,
            panel_title,
            help_text,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::hyperparam::HyperparamFormV1;
    use super::*;

    #[test]
    fn lora_panel_excludes_quant_picker_by_default() {
        let mut form = HyperparamFormV1::default();
        form.training_mode = "LoRA".into();
        let p = TrainingPanelV1::from_form(&form);
        // LoRA *can* benefit from quant-aware materialization, so we expose
        // the picker; users can leave it at Q4_K. (panel always shows for LoRA/QLoRA.)
        assert!(p.show_quant_format_picker);
        assert!(!p.qlora_recommended.is_empty());
    }

    #[test]
    fn qlora_panel_lists_three_quant_options() {
        let mut form = HyperparamFormV1::default();
        form.training_mode = "QLoRA".into();
        let p = TrainingPanelV1::from_form(&form);
        assert_eq!(p.quant_format_options.len(), 3);
        assert!(p.qlora_recommended.contains(&"Q4_K".to_string()));
    }

    #[test]
    fn bf16_panel_hides_quant_picker() {
        let mut form = HyperparamFormV1::default();
        form.training_mode = "Bf16-Full".into();
        let p = TrainingPanelV1::from_form(&form);
        assert!(!p.show_quant_format_picker);
        assert!(p.help_text.contains("Materialization"));
    }

    #[test]
    fn reinforcement_learning_panels_exist_and_hide_quant_picker() {
        let mut form = HyperparamFormV1::default();

        form.training_mode = "GRPO".into();
        let p_grpo = TrainingPanelV1::from_form(&form);
        assert!(!p_grpo.show_quant_format_picker);
        assert!(p_grpo.help_text.contains("Group Relative Policy"));

        form.training_mode = "DPO".into();
        let p_dpo = TrainingPanelV1::from_form(&form);
        assert!(!p_dpo.show_quant_format_picker);
        assert!(p_dpo.help_text.contains("Direct Preference"));

        form.training_mode = "ORPO".into();
        let p_orpo = TrainingPanelV1::from_form(&form);
        assert!(!p_orpo.show_quant_format_picker);
        assert!(p_orpo.help_text.contains("Odds-Ratio"));
    }

    #[test]
    fn known_training_modes_have_specific_mode_options_strings() {
        // L4: the canonical six modes the panel supports. Any drift
        // from this list intentionally fails this assertion so the
        // dropdown, the from_form match, the poller wire labels, and
        // `jobs::TrainingMode` stay in lock-step.
        let form = HyperparamFormV1::default();
        let p = TrainingPanelV1::from_form(&form);
        assert_eq!(
            p.mode_options,
            vec![
                "LoRA".to_string(),
                "QLoRA".to_string(),
                "Bf16-Full".to_string(),
                "GRPO".to_string(),
                "DPO".to_string(),
                "ORPO".to_string(),
            ]
        );
    }

    #[test]
    fn unknown_training_mode_renders_unknown_panel_label() {
        // L4: a stale config carrying "Lo-RA" (typo) silently trained as
        // BF16 pre-fix (quantization picker hidden). The expected
        // post-fix behavior is the panel rendering a recognizable
        // `unknown (...)` title and help text that doesn't claim the
        // mode *is* BF16.
        let mut form = HyperparamFormV1::default();
        form.training_mode = "Lo-RA".into();
        let p = TrainingPanelV1::from_form(&form);
        assert!(p.panel_title.contains("unknown"));
        assert!(p.panel_title.contains("Lo-RA"));
        assert!(p.help_text.contains("not recognized"));
        assert!(
            !p.help_text.contains("Materialization"),
            "unknown mode must NOT inherit BF16 help text: got {p:?}"
        );
    }

    #[test]
    fn unknown_rl_mode_is_explicitly_labeled() {
        // Slot between well-known RL modes for deprecation / typo space.
        let mut form = HyperparamFormV1::default();
        form.training_mode = "PPO".into();
        let p = TrainingPanelV1::from_form(&form);
        assert!(p.panel_title.contains("unknown"));
        // Quant picker must remain hidden: an unknown mode shouldn't
        // appear like a quant-aware mode either way.
        assert!(!p.show_quant_format_picker);
    }
}
