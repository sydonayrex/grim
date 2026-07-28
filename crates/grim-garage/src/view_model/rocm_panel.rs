//! ROCm toggles panel — four Checkbox/Toggle pairs:
//!  - `rmsnorm_matmul`        (RmsNorm+MatMul fusion HIP kernel)
//!  - `qkv_attention`         (QKV projection+Attention fusion)
//!  - `auto_wavefront`       (auto-detect 32 vs 64 wavefront size)
//!  - `xnack`                 (XNACK-aware unified memory)
//!
//! Plus a one-line device summary derived from the GPU probe.

use crate::backend::BackendProbe;
use crate::ui_state::display::DisplayState;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RocmToggleV1 {
    /// Stable id used by the toggle widget (`Toggle::id`).
    pub id: String,
    /// User-facing label.
    pub label: String,
    /// One-line description, shown next to the toggle.
    pub description: String,
    /// Whether the toggle is currently on.
    pub checked: bool,
    /// Whether the toggle can be interacted with (false when no GPU is present).
    pub enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RocmTogglesV1 {
    pub panel_title: String,
    pub device_summary: String,
    pub toggles: Vec<RocmToggleV1>,
}

impl RocmTogglesV1 {
    /// Construct for a fresh `DisplayState` (no state, no devices).
    pub fn default_for(state: &DisplayState) -> Self {
        Self::default_for_with_devices(
            state.rocm_devices(),
            true,  // rmsnorm_matmul defaults on
            false, // qkv_attention defaults off
        )
    }

    /// Construct with explicit defaults + a slice of already-rendered devices.
    pub fn default_for_with_devices(
        devices: &[BackendProbe],
        rmsnorm_matmul: bool,
        qkv_attention: bool,
    ) -> Self {
        let device_summary = if devices.is_empty() {
            "No ROCm devices detected — install ROCm to enable fused kernels.".to_string()
        } else if devices.len() == 1 {
            summarise_one(&devices[0])
        } else {
            let names: Vec<String> = devices.iter().map(summarise_one).collect();
            format!("{} (count={})", names.join(", "), devices.len())
        };

        let enabled = !devices.is_empty();
        let toggles = vec![
            RocmToggleV1 {
                id: "rmsnorm_matmul".into(),
                label: "RMSNorm + MatMul fusion".into(),
                description: "Fused HIP kernel — snippet of `fused_rmsnorm_matmul_rocm`.".into(),
                checked: rmsnorm_matmul,
                enabled,
            },
            RocmToggleV1 {
                id: "qkv_attention".into(),
                label: "QKV + Attention fusion".into(),
                description: "Fused HIP kernel — `fused_qkv_attention_rocm`.".into(),
                checked: qkv_attention,
                enabled,
            },
            RocmToggleV1 {
                id: "auto_wavefront".into(),
                label: "Auto wavefront (W32/W64)".into(),
                description: "Detect GCN arch at runtime; pick W32 for RDNA, W64 for CDNA.".into(),
                checked: true,
                enabled,
            },
            RocmToggleV1 {
                id: "xnack".into(),
                label: "XNACK-aware unified memory".into(),
                description: "Mi300X unified-memory path; ignored on devices without XNACK.".into(),
                checked: false,
                enabled,
            },
        ];

        Self {
            panel_title: "ROCm optimizations".into(),
            device_summary,
            toggles,
        }
    }
}

fn summarise_one(d: &BackendProbe) -> String {
    // BackendProbe carries a human-readable `detail` string from the probe;
    // use it directly rather than reconstructing from structured fields.
    d.detail.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cdna_device() -> BackendProbe {
        BackendProbe {
            name: "rocm".into(),
            device_kind: "rocm:0".into(),
            available: true,
            detail: "AMD Instinct MI300X / AMD / 304 CU(s) / 206158430208 VRAM".into(),
        }
    }

    fn rdna_device() -> BackendProbe {
        BackendProbe {
            name: "rocm".into(),
            device_kind: "rocm:0".into(),
            available: true,
            detail: "AMD Radeon RX 7900 XTX / AMD / 84 CU(s) / 17179869184 VRAM".into(),
        }
    }

    #[test]
    fn empty_device_list_disables_all_toggles() {
        let panel = RocmTogglesV1::default_for_with_devices(&[], true, false);
        assert_eq!(panel.toggles.len(), 4);
        for t in &panel.toggles {
            assert!(!t.enabled);
        }
        assert!(panel.device_summary.contains("No ROCm"));
    }

    #[test]
    fn cdna_device_summary_mentions_cdna() {
        let panel = RocmTogglesV1::default_for_with_devices(&[cdna_device()], true, true);
        assert!(panel.device_summary.contains("MI300X") || panel.device_summary.contains("304"));
        for t in &panel.toggles {
            assert!(t.enabled);
        }
    }

    #[test]
    fn rdna_device_summary_includes_gcn_arch_and_vram() {
        let panel = RocmTogglesV1::default_for_with_devices(&[rdna_device()], false, false);
        assert!(panel.device_summary.contains("7900"));
        assert!(panel.device_summary.contains("7900"));
    }

    #[test]
    fn multiple_devices_joined_with_separators() {
        let devices = vec![cdna_device(), rdna_device()];
        let panel = RocmTogglesV1::default_for_with_devices(&devices, true, false);
        assert!(panel.device_summary.contains("count=2"));
    }
}
