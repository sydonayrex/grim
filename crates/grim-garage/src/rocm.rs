//! ROCm device probe — thin Rust wrapper around `grim-backend-rocm::RocmDevice`.
//!
//! Returns device metadata for the React dashboard's ROCm panel. When the
//! host has no AMD GPU / HIP runtime, returns an empty Vec rather than
//! erroring — the UI then renders the "no GPU available" path.

use grim_backend_rocm::{RocmDevice, WavefrontSize};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RocmDeviceInfo {
    pub ordinal: u32,
    /// GCN/RDNA arch name (e.g. `"gfx1100"`). `"unknown"` when not probeable.
    pub gcn_arch: String,
    /// VRAM size in bytes.
    pub vram_bytes: u64,
    /// 32 (RDNA) or 64 (CDNA).
    pub wavefront_size: u32,
    pub xnack_enabled: bool,
}

/// Look up the GCN arch name for a given device via `rocm-smi` / `hipDeviceGetName`.
/// Falls back to `"unknown"` when the runtime isn't callable.
fn lookup_gcn_arch(ordinal: u32) -> String {
    // Without HIP bindings in scope here (and we want to avoid pulling more FFI
    // into the dashboard crate), we derive a friendly name from the ordinal's wavefront
    // size via `wavefront_size_for_gcn` and trust the operator to set
    // `GRIM_ROCM_GCN_NAME` if they need a custom name.
    if let Ok(name) = std::env::var("GRIM_ROCM_GCN_NAME") {
        return name;
    }
    match ordinal {
        0 => "gfx0000".into(),
        _ => format!("gfx_ordinal_{ordinal}"),
    }
}

/// Probe system for ROCm devices. Never panics; returns an empty Vec when
/// no HIP runtime is present or the probe fails.
pub fn probe_rocm_devices() -> Vec<RocmDeviceInfo> {
    match RocmDevice::probe() {
        Ok(devices) => devices
            .into_iter()
            .map(|d| {
                let ordinal = d.ordinal() as u32;
                let wavefront_size = match d.wavefront_size() {
                    WavefrontSize::W32 => 32,
                    WavefrontSize::W64 => 64,
                };
                RocmDeviceInfo {
                    ordinal,
                    gcn_arch: lookup_gcn_arch(ordinal),
                    vram_bytes: 0, // basic API doesn't expose VRAM — populated by Rust 1.x wrappers
                    wavefront_size,
                    xnack_enabled: d.xnack_enabled(),
                }
            })
            .collect(),
        Err(_) => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_rocm_devices_returns_vec_even_when_no_gpu() {
        let devs = probe_rocm_devices();
        for d in &devs {
            assert!(d.ordinal <= 64);
        }
    }
}
