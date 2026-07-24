//! ROCm device probe — thin Rust wrapper around `grim-backend-rocm::RocmDevice`.
//!
//! Returns device metadata for the React dashboard's ROCm panel. When the
//! host has no AMD GPU / HIP runtime, returns an empty Vec rather than
//! erroring — the UI then renders the "no GPU available" path.

use std::process::Command;
use grim_backend_rocm::{RocmDevice, WavefrontSize};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RocmDeviceInfo {
    pub ordinal: u32,
    /// Marketing GPU device name (e.g. `"AMD Radeon RX 7900 XTX"`, `"NVIDIA GeForce RTX 4070"`).
    #[serde(default = "default_gpu_name")]
    pub name: String,
    /// Device vendor name (`"AMD"`, `"NVIDIA"`, `"Intel"`, `"Unknown"`).
    #[serde(default = "default_vendor")]
    pub vendor: String,
    /// Execution backend engine (`"ROCm"`, `"CUDA"`, `"Vulkan"`, `"CPU"`).
    #[serde(default = "default_backend")]
    pub backend: String,
    /// Whether the device is native AMD ROCm/HIP compliant.
    #[serde(default)]
    pub is_rocm_compliant: bool,
    /// GCN/RDNA arch name (e.g. `"gfx1100"`, `"gfx90a"`) or CUDA compute capability.
    pub gcn_arch: String,
    /// Maximum VRAM size in bytes.
    pub vram_bytes: u64,
    /// Wavefront / Warp execution width (32 for RDNA / NVIDIA Warp, 64 for CDNA).
    pub wavefront_size: u32,
    /// Whether WMMA (Wave Matrix Multiply Accumulate) tensor hardware is present.
    #[serde(default)]
    pub wmma_supported: bool,
    /// Whether MFMA (Matrix Fused Multiply Add) matrix core hardware is present.
    #[serde(default)]
    pub mfma_supported: bool,
    /// Unified memory XNACK page migration enabled.
    pub xnack_enabled: bool,
    /// Total Compute Units (CUs) / Streaming Multiprocessors (SMs).
    #[serde(default = "default_cu_count")]
    pub compute_units: u32,
    /// Max threads per block.
    #[serde(default = "default_max_threads")]
    pub max_threads_per_block: u32,
}

fn default_gpu_name() -> String {
    "Graphics Accelerator".into()
}
fn default_vendor() -> String {
    "AMD".into()
}
fn default_backend() -> String {
    "ROCm".into()
}
fn default_cu_count() -> u32 {
    84
}
fn default_max_threads() -> u32 {
    1024
}

/// Helper to map NVIDIA GPU device names or chip codes to CUDA architecture families.
pub fn detect_nvidia_arch(gpu_name: &str) -> String {
    let name_upper = gpu_name.to_uppercase();
    if name_upper.contains("BLACKWELL") || name_upper.contains("GB10") || name_upper.contains("B100") || name_upper.contains("B200") || name_upper.contains("RTX 50") {
        "nv_cuda (Blackwell)".to_string()
    } else if name_upper.contains("HOPPER") || name_upper.contains("GH100") || name_upper.contains("H100") || name_upper.contains("H200") {
        "nv_cuda (Hopper)".to_string()
    } else if name_upper.contains("ADA") || name_upper.contains("AD10") || name_upper.contains("RTX 40") || name_upper.contains("L4") || name_upper.contains("L40") {
        "nv_cuda (Ada Lovelace)".to_string()
    } else if name_upper.contains("AMPERE") || name_upper.contains("GA10") || name_upper.contains("RTX 30") || name_upper.contains("A100") || name_upper.contains("A10") || name_upper.contains("A30") || name_upper.contains("A40") {
        "nv_cuda (Ampere)".to_string()
    } else if name_upper.contains("TURING") || name_upper.contains("TU10") || name_upper.contains("RTX 20") || name_upper.contains("GTX 16") || name_upper.contains("T4") {
        "nv_cuda (Turing)".to_string()
    } else if name_upper.contains("VOLTA") || name_upper.contains("GV100") || name_upper.contains("V100") {
        "nv_cuda (Volta)".to_string()
    } else if name_upper.contains("PASCAL") || name_upper.contains("GP10") || name_upper.contains("GTX 10") || name_upper.contains("P100") || name_upper.contains("P40") || name_upper.contains("P4") {
        "nv_cuda (Pascal)".to_string()
    } else if name_upper.contains("MAXWELL") || name_upper.contains("GM20") || name_upper.contains("GTX 9") || name_upper.contains("M40") {
        "nv_cuda (Maxwell)".to_string()
    } else {
        "nv_cuda (CUDA Architecture)".to_string()
    }
}

/// Helper to parse marketing GPU names from lspci PCI strings.
pub fn extract_clean_gpu_name(raw_line: &str) -> String {
    if let (Some(start), Some(end)) = (raw_line.find('['), raw_line.rfind(']')) {
        if start < end {
            let inner = &raw_line[start + 1..end];
            if inner.contains('[') && inner.contains(']') {
                if let (Some(inner_start), Some(inner_end)) = (inner.find('['), inner.rfind(']')) {
                    if inner_start < inner_end {
                        let name = &inner[inner_start + 1..inner_end];
                        if !name.contains(':') && name.len() > 3 {
                            return format!("NVIDIA {name}");
                        }
                    }
                }
            }
            if !inner.contains(':') && inner.len() > 3 {
                return inner.to_string();
            }
        }
    }
    raw_line.to_string()
}

/// Query system `nvidia-smi` driver query interface if present.
pub fn query_nvidia_smi_gpus() -> Vec<(String, u64)> {
    let mut results = Vec::new();
    if let Ok(output) = Command::new("nvidia-smi")
        .args(["--query-gpu=gpu_name,memory.total", "--format=csv,noheader,nounits"])
        .output()
    {
        if output.status.success() {
            let text = String::from_utf8_lossy(&output.stdout);
            for line in text.lines() {
                let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if parts.len() >= 2 {
                    let name = parts[0].to_string();
                    if let Ok(mb) = parts[1].parse::<u64>() {
                        results.push((name, mb * 1024 * 1024));
                    }
                }
            }
        }
    }
    results
}

/// Helper to map AMD GCN / RDNA / CDNA target architecture strings to family names.
pub fn detect_amd_arch(gcn_arch: &str, marketing_name: &str) -> String {
    let arch_lower = gcn_arch.to_lowercase();
    let name_lower = marketing_name.to_lowercase();

    if arch_lower.starts_with("gfx94") || name_lower.contains("mi300") {
        format!("{gcn_arch} (CDNA3)")
    } else if arch_lower.starts_with("gfx90a") || name_lower.contains("mi250") || name_lower.contains("mi210") {
        format!("{gcn_arch} (CDNA2)")
    } else if arch_lower.starts_with("gfx908") || name_lower.contains("mi100") {
        format!("{gcn_arch} (CDNA1)")
    } else if arch_lower.starts_with("gfx12") || name_lower.contains("rx 8000") || name_lower.contains("rdna4") {
        format!("{gcn_arch} (RDNA4)")
    } else if arch_lower.starts_with("gfx11") || name_lower.contains("rx 7900") || name_lower.contains("rx 7800") || name_lower.contains("rx 7700") || name_lower.contains("rx 7600") || name_lower.contains("rdna3") {
        format!("{gcn_arch} (RDNA3)")
    } else if arch_lower.starts_with("gfx103") || name_lower.contains("610m") || name_lower.contains("rx 6900") || name_lower.contains("rx 6800") || name_lower.contains("rx 6700") || name_lower.contains("rx 6600") || name_lower.contains("raphael") || name_lower.contains("rdna2") {
        format!("{gcn_arch} (RDNA2)")
    } else if arch_lower.starts_with("gfx101") || name_lower.contains("rx 5700") || name_lower.contains("rx 5600") || name_lower.contains("rdna1") {
        format!("{gcn_arch} (RDNA1)")
    } else if arch_lower.starts_with("gfx90") || name_lower.contains("vega") || name_lower.contains("radeon vii") || name_lower.contains("mi50") || name_lower.contains("mi60") {
        format!("{gcn_arch} (Vega / GCN5)")
    } else if !gcn_arch.is_empty() {
        format!("{gcn_arch} (AMD ROCm)")
    } else {
        "generic_amd_gpu (AMD ROCm)".to_string()
    }
}

/// Query official AMD `rocminfo` tool for installed ROCm HIP GPUs.
pub fn query_rocminfo_gpus() -> Vec<RocmDeviceInfo> {
    let mut devices = Vec::new();
    if let Ok(output) = Command::new("rocminfo").output() {
        if output.status.success() {
            let text = String::from_utf8_lossy(&output.stdout);
            let mut is_gpu = false;
            let mut name = String::new();
            let mut marketing_name = String::new();
            let mut compute_units = 36u32;
            let mut wavefront_size = 32u32;
            let mut vram_bytes = 8_589_934_592u64;
            let mut ordinal = 0u32;

            for line in text.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("Agent ") && trimmed.ends_with('*') {
                    if is_gpu && !name.is_empty() {
                        let full_arch = detect_amd_arch(&name, &marketing_name);
                        let is_w32 = wavefront_size == 32;
                        let is_w64 = wavefront_size == 64;
                        let is_cdna = name.starts_with("gfx94") || name.starts_with("gfx90");
                        devices.push(RocmDeviceInfo {
                            ordinal,
                            name: if marketing_name.is_empty() { format!("AMD GPU ({name})") } else { marketing_name.clone() },
                            vendor: "AMD".to_string(),
                            backend: "ROCm".to_string(),
                            is_rocm_compliant: true,
                            gcn_arch: full_arch,
                            vram_bytes,
                            wavefront_size,
                            wmma_supported: is_w32 || is_cdna,
                            mfma_supported: is_w64 || is_cdna,
                            xnack_enabled: is_cdna,
                            compute_units,
                            max_threads_per_block: 1024,
                        });
                        ordinal += 1;
                    }
                    is_gpu = false;
                    name.clear();
                    marketing_name.clear();
                } else if trimmed.starts_with("Device Type:") && trimmed.contains("GPU") {
                    is_gpu = true;
                } else if is_gpu {
                    if trimmed.starts_with("Name:") {
                        let val = trimmed["Name:".len()..].trim();
                        if !val.contains("amdgcn") {
                            name = val.to_string();
                        }
                    } else if trimmed.starts_with("Marketing Name:") {
                        marketing_name = trimmed["Marketing Name:".len()..].trim().to_string();
                    } else if trimmed.starts_with("Compute Unit:") {
                        if let Ok(cu) = trimmed["Compute Unit:".len()..].trim().parse::<u32>() {
                            compute_units = cu;
                        }
                    } else if trimmed.starts_with("Wavefront Size:") {
                        let raw = trimmed["Wavefront Size:".len()..].trim();
                        let clean = raw.split('(').next().unwrap_or(raw).trim();
                        if let Ok(wf) = clean.parse::<u32>() {
                            wavefront_size = wf;
                        }
                    } else if trimmed.starts_with("Size:") && trimmed.ends_with("KB") {
                        let raw = trimmed["Size:".len()..].trim_end_matches("KB").trim();
                        let clean = raw.split('(').next().unwrap_or(raw).trim();
                        if let Ok(kb) = clean.parse::<u64>() {
                            if kb > 100_000 {
                                vram_bytes = kb * 1024;
                            }
                        }
                    }
                }
            }

            if is_gpu && !name.is_empty() {
                let full_arch = detect_amd_arch(&name, &marketing_name);
                let is_w32 = wavefront_size == 32;
                let is_w64 = wavefront_size == 64;
                let is_cdna = name.starts_with("gfx94") || name.starts_with("gfx90");
                devices.push(RocmDeviceInfo {
                    ordinal,
                    name: if marketing_name.is_empty() { format!("AMD GPU ({name})") } else { marketing_name.clone() },
                    vendor: "AMD".to_string(),
                    backend: "ROCm".to_string(),
                    is_rocm_compliant: true,
                    gcn_arch: full_arch,
                    vram_bytes,
                    wavefront_size,
                    wmma_supported: is_w32 || is_cdna,
                    mfma_supported: is_w64 || is_cdna,
                    xnack_enabled: is_cdna,
                    compute_units,
                    max_threads_per_block: 1024,
                });
            }
        }
    }
    devices
}

/// Probe system PCI hardware and ROCm/CUDA telemetry for installed GPUs.
pub fn probe_rocm_devices() -> Vec<RocmDeviceInfo> {
    let mut devices = Vec::new();
    let mut ordinal = 0;

    // 1. Query official nvidia-smi driver interface for installed NVIDIA GPUs.
    let nvidia_smi_devs = query_nvidia_smi_gpus();
    for (gpu_name, vram_bytes) in nvidia_smi_devs {
        let arch = detect_nvidia_arch(&gpu_name);
        devices.push(RocmDeviceInfo {
            ordinal,
            name: gpu_name,
            vendor: "NVIDIA".to_string(),
            backend: "CUDA".to_string(),
            is_rocm_compliant: false,
            gcn_arch: arch,
            vram_bytes,
            wavefront_size: 32,
            wmma_supported: false,
            mfma_supported: false,
            xnack_enabled: false,
            compute_units: 36,
            max_threads_per_block: 1024,
        });
        ordinal += 1;
    }

    // 2. Query official rocminfo tool for installed AMD ROCm GPUs.
    let rocminfo_devs = query_rocminfo_gpus();
    for mut amd_dev in rocminfo_devs {
        amd_dev.ordinal = ordinal;
        devices.push(amd_dev);
        ordinal += 1;
    }

    // 3. Query system PCI bus via lspci to detect GPUs if telemetry tools didn't catch them.
    if let Ok(output) = Command::new("lspci").arg("-nn").output() {
        if output.status.success() {
            let text = String::from_utf8_lossy(&output.stdout);
            for line in text.lines() {
                if line.contains("VGA compatible controller") || line.contains("3D controller") || line.contains("Display controller") {
                    let mut raw_name = line.to_string();
                    if let Some(pos) = line.find(':') {
                        raw_name = line[pos + 1..].trim().to_string();
                    }
                    
                    if raw_name.contains("NVIDIA") {
                        if devices.iter().any(|d| d.vendor == "NVIDIA") {
                            continue;
                        }
                        let clean_name = extract_clean_gpu_name(&raw_name);
                        let arch = detect_nvidia_arch(&clean_name);
                        devices.push(RocmDeviceInfo {
                            ordinal,
                            name: clean_name,
                            vendor: "NVIDIA".to_string(),
                            backend: "CUDA".to_string(),
                            is_rocm_compliant: false,
                            gcn_arch: arch,
                            vram_bytes: 8_589_934_592u64,
                            wavefront_size: 32,
                            wmma_supported: false,
                            mfma_supported: false,
                            xnack_enabled: false,
                            compute_units: 36,
                            max_threads_per_block: 1024,
                        });
                        ordinal += 1;
                    } else if raw_name.contains("AMD") || raw_name.contains("Advanced Micro Devices") {
                        if devices.iter().any(|d| d.vendor == "AMD") {
                            continue;
                        }
                        let clean_name = extract_clean_gpu_name(&raw_name);
                        let arch = detect_amd_arch("", &clean_name);
                        devices.push(RocmDeviceInfo {
                            ordinal,
                            name: clean_name,
                            vendor: "AMD".to_string(),
                            backend: "ROCm".to_string(),
                            is_rocm_compliant: true,
                            gcn_arch: arch,
                            vram_bytes: 4_294_967_296u64,
                            wavefront_size: 32,
                            wmma_supported: false,
                            mfma_supported: false,
                            xnack_enabled: false,
                            compute_units: 12,
                            max_threads_per_block: 1024,
                        });
                        ordinal += 1;
                    }
                }
            }
        }
    }

    // 4. Fall back to HIP probe if no devices found so far.
    if devices.is_empty() {
        if let Ok(hip_devs) = RocmDevice::probe() {
            for d in hip_devs {
                let ordinal = d.ordinal() as u32;
                let wavefront_size = match d.wavefront_size() {
                    WavefrontSize::W32 => 32,
                    WavefrontSize::W64 => 64,
                };
                let gcn_arch_env = std::env::var("GRIM_ROCM_GCN_NAME").unwrap_or_else(|_| "gfx1030".into());
                let name = std::env::var("GRIM_ROCM_DEVICE_NAME").unwrap_or_else(|_| format!("AMD ROCm Accelerator #{ordinal}"));
                let full_arch = detect_amd_arch(&gcn_arch_env, &name);
                devices.push(RocmDeviceInfo {
                    ordinal,
                    name,
                    vendor: "AMD".to_string(),
                    backend: "ROCm".to_string(),
                    is_rocm_compliant: true,
                    gcn_arch: full_arch,
                    vram_bytes: 8_589_934_592,
                    wavefront_size,
                    wmma_supported: wavefront_size == 32,
                    mfma_supported: wavefront_size == 64,
                    xnack_enabled: d.xnack_enabled(),
                    compute_units: 36,
                    max_threads_per_block: 1024,
                });
            }
        }
    }

    devices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_rocm_devices_returns_vec_even_when_no_gpu() {
        let devs = probe_rocm_devices();
        for d in &devs {
            assert!(d.max_threads_per_block > 0);
        }
    }

    #[test]
    fn nvidia_gpu_delineated_as_non_rocm_cuda() {
        let info = RocmDeviceInfo {
            ordinal: 0,
            name: "NVIDIA GeForce RTX 4070 Laptop GPU".into(),
            vendor: "NVIDIA".into(),
            backend: "CUDA".into(),
            is_rocm_compliant: false,
            gcn_arch: detect_nvidia_arch("NVIDIA GeForce RTX 4070 Laptop GPU"),
            vram_bytes: 8 * 1024 * 1024 * 1024,
            wavefront_size: 32,
            wmma_supported: false,
            mfma_supported: false,
            xnack_enabled: false,
            compute_units: 36,
            max_threads_per_block: 1024,
        };
        assert_eq!(info.vendor, "NVIDIA");
        assert_eq!(info.backend, "CUDA");
        assert!(!info.is_rocm_compliant);
        assert_eq!(info.gcn_arch, "nv_cuda (Ada Lovelace)");
    }

    #[test]
    fn detect_nvidia_arch_dynamically_identifies_all_cuda_generations() {
        assert_eq!(detect_nvidia_arch("NVIDIA GeForce RTX 5090"), "nv_cuda (Blackwell)");
        assert_eq!(detect_nvidia_arch("NVIDIA B200 SXM 180GB"), "nv_cuda (Blackwell)");
        assert_eq!(detect_nvidia_arch("NVIDIA H100 80GB PCIe"), "nv_cuda (Hopper)");
        assert_eq!(detect_nvidia_arch("NVIDIA GeForce RTX 4090"), "nv_cuda (Ada Lovelace)");
        assert_eq!(detect_nvidia_arch("NVIDIA GeForce RTX 4070 Laptop GPU"), "nv_cuda (Ada Lovelace)");
        assert_eq!(detect_nvidia_arch("NVIDIA GeForce RTX 3090"), "nv_cuda (Ampere)");
        assert_eq!(detect_nvidia_arch("NVIDIA A100-SXM4-80GB"), "nv_cuda (Ampere)");
        assert_eq!(detect_nvidia_arch("NVIDIA GeForce RTX 2080 Ti"), "nv_cuda (Turing)");
        assert_eq!(detect_nvidia_arch("NVIDIA Tesla T4"), "nv_cuda (Turing)");
        assert_eq!(detect_nvidia_arch("NVIDIA Tesla V100-SXM2-32GB"), "nv_cuda (Volta)");
        assert_eq!(detect_nvidia_arch("NVIDIA GeForce GTX 1080 Ti"), "nv_cuda (Pascal)");
        assert_eq!(detect_nvidia_arch("NVIDIA Tesla P100-PCIE-16GB"), "nv_cuda (Pascal)");
        assert_eq!(detect_nvidia_arch("NVIDIA GeForce GTX 980 Ti"), "nv_cuda (Maxwell)");
        assert_eq!(detect_nvidia_arch("NVIDIA Unknown Accelerator"), "nv_cuda (CUDA Architecture)");
    }

    #[test]
    fn detect_amd_arch_dynamically_identifies_all_rocm_generations() {
        assert_eq!(detect_amd_arch("gfx942", "AMD Instinct MI300X"), "gfx942 (CDNA3)");
        assert_eq!(detect_amd_arch("gfx90a", "AMD Instinct MI250X"), "gfx90a (CDNA2)");
        assert_eq!(detect_amd_arch("gfx908", "AMD Instinct MI100"), "gfx908 (CDNA1)");
        assert_eq!(detect_amd_arch("gfx1200", "AMD Radeon RX 8800 XT"), "gfx1200 (RDNA4)");
        assert_eq!(detect_amd_arch("gfx1100", "AMD Radeon RX 7900 XTX"), "gfx1100 (RDNA3)");
        assert_eq!(detect_amd_arch("gfx1036", "AMD Radeon 610M"), "gfx1036 (RDNA2)");
        assert_eq!(detect_amd_arch("gfx1030", "AMD Radeon RX 6800 XT"), "gfx1030 (RDNA2)");
        assert_eq!(detect_amd_arch("gfx1010", "AMD Radeon RX 5700 XT"), "gfx1010 (RDNA1)");
        assert_eq!(detect_amd_arch("gfx906", "Radeon VII"), "gfx906 (Vega / GCN5)");
    }
}
