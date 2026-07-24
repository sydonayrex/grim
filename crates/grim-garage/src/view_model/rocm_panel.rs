//! ROCm toggles panel — four Checkbox/Toggle pairs:
//!  - `rmsnorm_matmul`        (RmsNorm+MatMul fusion HIP kernel)
//!  - `qkv_attention`         (QKV projection+Attention fusion)
//!  - `auto_wavefront`       (auto-detect 32 vs 64 wavefront size)
//!  - `xnack`                 (XNACK-aware unified memory)
//!
//! Plus a one-line device summary derived from the GPU probe.

use crate::rocm::RocmDeviceInfo;
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
            true, // rmsnorm_matmul defaults on
            false, // qkv_attention defaults off
        )
    }

    /// Construct with explicit defaults + a slice of already-rendered devices.
    pub fn default_for_with_devices(
        devices: &[RocmDeviceInfo],
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

fn summarise_one(d: &RocmDeviceInfo) -> String {
    let arch = arch_label(d.ordinal, &d.gcn_arch, d.wavefront_size);
    let vram_gb = d.vram_bytes / (1024 * 1024 * 1024);
    format!("{arch} (ordinal {ord}, {wf}-wide, {vram} GiB VRAM)",
            arch = arch,
            ord = d.ordinal,
            wf = d.wavefront_size,
            vram = vram_gb)
}

fn arch_label(ordinal: u32, gcn_arch: &str, wavefront_size: u32) -> String {
    let family = match (gcn_arch, wavefront_size) {
        (a, _) if a.starts_with("gfx94") || a.starts_with("gfx90") => "CDNA",
        (a, 32) if a.starts_with("gfx11") || a.starts_with("gfx10") => "RDNA",
        _ => "unknown",
    };
    format!("{family} {gcn_arch} (ordinal {ordinal})")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cdna_device() -> RocmDeviceInfo {
        RocmDeviceInfo {
            ordinal: 0,
            name: "AMD Instinct MI300X".into(),
            vendor: "AMD".into(),
            backend: "ROCm".into(),
            is_rocm_compliant: true,
            gcn_arch: "gfx942".into(),
            vram_bytes: 192 * 1024 * 1024 * 1024,
            wavefront_size: 64,
            wmma_supported: true,
            mfma_supported: true,
            xnack_enabled: true,
            compute_units: 304,
            max_threads_per_block: 1024,
        }
    }

    fn rdna_device() -> RocmDeviceInfo {
        RocmDeviceInfo {
            ordinal: 0,
            name: "AMD Radeon RX 7900 XTX".into(),
            vendor: "AMD".into(),
            backend: "ROCm".into(),
            is_rocm_compliant: true,
            gcn_arch: "gfx1100".into(),
            vram_bytes: 16 * 1024 * 1024 * 1024,
            wavefront_size: 32,
            wmma_supported: true,
            mfma_supported: false,
            xnack_enabled: false,
            compute_units: 84,
            max_threads_per_block: 1024,
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
        assert!(panel.device_summary.contains("CDNA") || panel.device_summary.contains("W64"));
        for t in &panel.toggles {
            assert!(t.enabled);
        }
    }

    #[test]
    fn rdna_device_summary_includes_gcn_arch_and_vram() {
        let panel = RocmTogglesV1::default_for_with_devices(&[rdna_device()], false, false);
        assert!(panel.device_summary.contains("RDNA"));
        assert!(panel.device_summary.contains("16"));
    }

    #[test]
    fn multiple_devices_joined_with_separators() {
        let devices = vec![cdna_device(), rdna_device()];
        let panel = RocmTogglesV1::default_for_with_devices(&devices, true, false);
        assert!(panel.device_summary.contains("count=2"));
    }
}
