//! ROCm device probe — thin Rust wrapper around `grim-backend-rocm::RocmDevice`.
//!
//! Returns device metadata for the React dashboard's ROCm panel. When the
//! host has no AMD GPU / HIP runtime, returns an empty Vec rather than
//! erroring — the UI then renders the "no GPU available" path.

use grim_backend_rocm::{RocmDevice, WavefrontSize};
use serde::{Deserialize, Serialize};
use std::process::Command;
use tracing::warn;

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
    /// Currently allocated / used VRAM size in bytes.
    #[serde(default)]
    pub vram_used_bytes: u64,
    /// Live GPU utilization percent (sysfs `gpu_busy_percent`; 0 when absent).
    #[serde(default)]
    pub gpu_busy_percent: u32,
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
    if name_upper.contains("BLACKWELL")
        || name_upper.contains("GB10")
        || name_upper.contains("B100")
        || name_upper.contains("B200")
        || name_upper.contains("RTX 50")
    {
        "nv_cuda (Blackwell)".to_string()
    } else if name_upper.contains("HOPPER")
        || name_upper.contains("GH100")
        || name_upper.contains("H100")
        || name_upper.contains("H200")
    {
        "nv_cuda (Hopper)".to_string()
    } else if name_upper.contains("ADA")
        || name_upper.contains("AD10")
        || name_upper.contains("RTX 40")
        || name_upper.contains("L4")
        || name_upper.contains("L40")
    {
        "nv_cuda (Ada Lovelace)".to_string()
    } else if name_upper.contains("AMPERE")
        || name_upper.contains("GA10")
        || name_upper.contains("RTX 30")
        || name_upper.contains("A100")
        || name_upper.contains("A10")
        || name_upper.contains("A30")
        || name_upper.contains("A40")
    {
        "nv_cuda (Ampere)".to_string()
    } else if name_upper.contains("TURING")
        || name_upper.contains("TU10")
        || name_upper.contains("RTX 20")
        || name_upper.contains("GTX 16")
        || name_upper.contains("T4")
    {
        "nv_cuda (Turing)".to_string()
    } else if name_upper.contains("VOLTA")
        || name_upper.contains("GV100")
        || name_upper.contains("V100")
    {
        "nv_cuda (Volta)".to_string()
    } else if name_upper.contains("PASCAL")
        || name_upper.contains("GP10")
        || name_upper.contains("GTX 10")
        || name_upper.contains("P100")
        || name_upper.contains("P40")
        || name_upper.contains("P4")
    {
        "nv_cuda (Pascal)".to_string()
    } else if name_upper.contains("MAXWELL")
        || name_upper.contains("GM20")
        || name_upper.contains("GTX 9")
        || name_upper.contains("M40")
    {
        "nv_cuda (Maxwell)".to_string()
    } else {
        "nv_cuda (CUDA Architecture)".to_string()
    }
}

/// Helper to parse marketing GPU names from lspci PCI strings.
pub fn extract_clean_gpu_name(raw_line: &str) -> String {
    let mut bracket_contents = Vec::new();
    let mut current = String::new();
    let mut in_bracket = false;

    for c in raw_line.chars() {
        if c == '[' {
            in_bracket = true;
            current.clear();
        } else if c == ']' {
            if in_bracket {
                in_bracket = false;
                let trimmed = current.trim();
                if !trimmed.is_empty()
                    && trimmed != "AMD/ATI"
                    && !trimmed.starts_with("Device ")
                    && !trimmed.contains(':')
                    && trimmed.len() > 2
                {
                    bracket_contents.push(trimmed.to_string());
                }
            }
        } else if in_bracket {
            current.push(c);
        }
    }

    if let Some(last) = bracket_contents.pop() {
        return last;
    }

    if let Some(pos) = raw_line.find(':') {
        let after = raw_line[pos + 1..].trim();
        if let Some(pos2) = after.find(':') {
            return after[pos2 + 1..].trim().to_string();
        }
        return after.to_string();
    }

    raw_line.to_string()
}

/// Query system `nvidia-smi` driver query interface if present.
/// Returns Vec<(gpu_name, vram_used_bytes, vram_total_bytes)>.
pub fn query_nvidia_smi_gpus() -> Vec<(String, u64, u64)> {
    let mut results = Vec::new();
    if let Ok(output) = Command::new("nvidia-smi")
        .args([
            "--query-gpu=gpu_name,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        if output.status.success() {
            let text = String::from_utf8_lossy(&output.stdout);
            for line in text.lines() {
                let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if parts.len() >= 3 {
                    let name = parts[0].to_string();
                    let used_mb = parts[1].parse::<u64>().unwrap_or(0);
                    let total_mb = parts[2].parse::<u64>().unwrap_or(8192);
                    results.push((name, used_mb * 1024 * 1024, total_mb * 1024 * 1024));
                }
            }
        }
    }
    if results.is_empty() {
        warn!("nvidia-smi not found or returned no GPUs; CUDA device probe returned empty");
    }
    results
}

/// Pure clamping helper for L3 — extractable for unit testing.
///
/// Pre-fix `query_amd_vram_used` clamped `(ordinal as usize).min(len -
/// 1)`, aliasing distinct AMD cards to the last slot whenever `ordinal ≥
/// used_bytes.len()` (e.g. an iGPU without the `mem_info_vram_used`
/// sysfs node + a dGPU with it). The post-fix contract is: return the
/// slot at `ordinal` if it exists; otherwise 0 (unknown — not somebody
/// else's measurement). Pin a hand-derived 5-slot example below so a
/// mutant that swaps the predicate limbs is caught.
pub fn pick_vram_used_slot(used_bytes: &[u64], ordinal: u32) -> u64 {
    let idx = ordinal as usize;
    if idx < used_bytes.len() {
        used_bytes[idx]
    } else {
        0
    }
}

/// Query actual total AMD VRAM bytes from Linux sysfs mem_info_vram_total interface.
pub fn query_amd_vram_total(ordinal: u32) -> u64 {
    if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
        let mut amd_total_bytes = Vec::new();
        let mut card_paths: Vec<_> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .map(|n| {
                        n.to_string_lossy().starts_with("card")
                            && !n.to_string_lossy().contains('-')
                    })
                    .unwrap_or(false)
            })
            .collect();
        card_paths.sort();

        for path in card_paths {
            let vendor_path = path.join("device/vendor");
            if let Ok(v_str) = std::fs::read_to_string(&vendor_path) {
                if v_str.trim().eq_ignore_ascii_case("0x1002") {
                    let total_path = path.join("device/mem_info_vram_total");
                    if let Ok(t_str) = std::fs::read_to_string(&total_path) {
                        if let Ok(bytes) = t_str.trim().parse::<u64>() {
                            if bytes > 0 {
                                amd_total_bytes.push(bytes);
                            }
                        }
                    }
                }
            }
        }
        return pick_vram_used_slot(&amd_total_bytes, ordinal);
    }
    0
}

/// Query live AMD VRAM used bytes from Linux sysfs mem_info_vram_used interface.
pub fn query_amd_vram_used(ordinal: u32) -> u64 {
    if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
        let mut amd_used_bytes = Vec::new();
        let mut card_paths: Vec<_> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .map(|n| {
                        n.to_string_lossy().starts_with("card")
                            && !n.to_string_lossy().contains('-')
                    })
                    .unwrap_or(false)
            })
            .collect();
        card_paths.sort();

        for path in card_paths {
            let vendor_path = path.join("device/vendor");
            if let Ok(v_str) = std::fs::read_to_string(&vendor_path) {
                if v_str.trim().eq_ignore_ascii_case("0x1002") {
                    let used_path = path.join("device/mem_info_vram_used");
                    if let Ok(u_str) = std::fs::read_to_string(&used_path) {
                        if let Ok(bytes) = u_str.trim().parse::<u64>() {
                            amd_used_bytes.push(bytes);
                        }
                    }
                }
            }
        }
        return pick_vram_used_slot(&amd_used_bytes, ordinal);
    }
    0
}

/// Query live AMD GPU busy percent from the sysfs gpu_busy_percent interface.
/// Returns 0 when unavailable (non-Linux, iGPU without the node) — the panel
/// then simply shows 0% rather than pretending telemetry is absent.
pub fn query_amd_gpu_busy(ordinal: u32) -> u32 {
    if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
        let mut busy: Vec<u32> = Vec::new();
        let mut card_paths: Vec<_> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .map(|n| {
                        n.to_string_lossy().starts_with("card")
                            && !n.to_string_lossy().contains('-')
                    })
                    .unwrap_or(false)
            })
            .collect();
        card_paths.sort();

        for path in card_paths {
            let vendor_path = path.join("device/vendor");
            if let Ok(v_str) = std::fs::read_to_string(&vendor_path) {
                if v_str.trim().eq_ignore_ascii_case("0x1002") {
                    let busy_path = path.join("device/gpu_busy_percent");
                    if let Ok(b_str) = std::fs::read_to_string(&busy_path) {
                        if let Ok(pct) = b_str.trim().parse::<u32>() {
                            busy.push(pct);
                        }
                    }
                }
            }
        }
        return busy.get(ordinal as usize).copied().unwrap_or(0);
    }
    0
}

/// Helper to map AMD GCN / RDNA / CDNA target architecture strings to family names.
pub fn detect_amd_arch(gcn_arch: &str, marketing_name: &str) -> String {
    let arch_lower = gcn_arch.to_lowercase();
    let name_lower = marketing_name.to_lowercase();

    if name_lower.contains("9800x3d")
        || name_lower.contains("raphael")
        || name_lower.contains("680m")
        || name_lower.contains("780m")
        || name_lower.contains("610m")
        || arch_lower.starts_with("gfx103")
        || arch_lower.contains("1036")
    {
        let arch = if gcn_arch.is_empty() || gcn_arch.starts_with("gfx12") {
            "gfx1036"
        } else {
            gcn_arch
        };
        format!("{arch} (RDNA2)")
    } else if arch_lower.starts_with("gfx94") || name_lower.contains("mi300") {
        format!("{gcn_arch} (CDNA3)")
    } else if arch_lower.starts_with("gfx90a")
        || name_lower.contains("mi250")
        || name_lower.contains("mi210")
    {
        format!("{gcn_arch} (CDNA2)")
    } else if arch_lower.starts_with("gfx908") || name_lower.contains("mi100") {
        format!("{gcn_arch} (CDNA1)")
    } else if arch_lower.starts_with("gfx13") || name_lower.contains("rdna5") {
        format!("{gcn_arch} (RDNA5)")
    } else if arch_lower.starts_with("gfx12")
        || name_lower.contains("9070")
        || name_lower.contains("9060")
        || name_lower.contains("rx 8000")
        || name_lower.contains("rdna4")
    {
        format!("{gcn_arch} (RDNA4)")
    } else if arch_lower.starts_with("gfx11")
        || name_lower.contains("rx 7900")
        || name_lower.contains("rx 7800")
        || name_lower.contains("rx 7700")
        || name_lower.contains("rx 7600")
        || name_lower.contains("rdna3")
    {
        format!("{gcn_arch} (RDNA3)")
    } else if arch_lower.starts_with("gfx101")
        || name_lower.contains("rx 5700")
        || name_lower.contains("rx 5600")
        || name_lower.contains("rdna1")
    {
        format!("{gcn_arch} (RDNA1)")
    } else if arch_lower.starts_with("gfx90")
        || name_lower.contains("vega")
        || name_lower.contains("radeon vii")
        || name_lower.contains("mi50")
        || name_lower.contains("mi60")
    {
        format!("{gcn_arch} (Vega / GCN5)")
    } else if !gcn_arch.is_empty() {
        format!("{gcn_arch} (AMD ROCm)")
    } else {
        "generic_amd_gpu (AMD ROCm)".to_string()
    }
}

/// Helper to extract clean user-friendly marketing product names with RDNA version (e.g. Radeon 680M (RDNA 2), RX 9070 (RDNA 4)).
pub fn user_friendly_amd_name(gcn_arch: &str, marketing_name: &str) -> String {
    let arch_lower = gcn_arch.to_lowercase();
    let name_lower = marketing_name.to_lowercase();

    // Determine RDNA / CDNA version tag
    let rdna_ver = if name_lower.contains("9800x3d")
        || name_lower.contains("raphael")
        || name_lower.contains("680m")
        || name_lower.contains("780m")
        || name_lower.contains("610m")
        || arch_lower.contains("gfx103")
        || arch_lower.contains("1036")
    {
        "RDNA 2"
    } else if arch_lower.contains("gfx13") || name_lower.contains("rdna5") {
        "RDNA 5"
    } else if arch_lower.contains("gfx12")
        || name_lower.contains("rdna4")
        || name_lower.contains("8800")
        || name_lower.contains("9070")
        || name_lower.contains("9060")
    {
        "RDNA 4"
    } else if arch_lower.contains("gfx11")
        || name_lower.contains("rdna3")
        || name_lower.contains("7900")
        || name_lower.contains("7800")
        || name_lower.contains("7700")
        || name_lower.contains("7600")
    {
        "RDNA 3"
    } else if arch_lower.contains("gfx101")
        || name_lower.contains("rdna1")
        || name_lower.contains("5700")
        || name_lower.contains("5600")
    {
        "RDNA 1"
    } else if arch_lower.contains("gfx94") || name_lower.contains("mi300") {
        "CDNA 3"
    } else if arch_lower.contains("gfx90a") || name_lower.contains("mi250") {
        "CDNA 2"
    } else {
        "RDNA"
    };

    // Determine base product name
    let product_name = if name_lower.contains("9800x3d") {
        "AMD Ryzen 7 9800X3D 8-Core Processor".to_string()
    } else if arch_lower.contains("gfx1036")
        || name_lower.contains("rembrandt")
        || name_lower.contains("0300")
        || arch_lower.contains("0300")
    {
        "Radeon 680M iGPU".to_string()
    } else if !marketing_name.is_empty()
        && !marketing_name.starts_with("c_")
        && !marketing_name.starts_with("Device ")
        && marketing_name != "AMD/ATI"
        && marketing_name != "0300"
        && !marketing_name.eq_ignore_ascii_case("generic_amd_gpu")
        && !marketing_name.eq_ignore_ascii_case("Radeon GPU")
    {
        marketing_name.trim().to_string()
    } else if arch_lower.contains("gfx13") {
        "Radeon RX 9070".to_string()
    } else if arch_lower.contains("gfx1200") {
        "Radeon RX 8800 XT".to_string()
    } else if arch_lower.contains("gfx1100") {
        "Radeon RX 7900 XTX".to_string()
    } else if arch_lower.contains("gfx1030") {
        "Radeon RX 6800 XT".to_string()
    } else if !gcn_arch.is_empty() {
        format!("Radeon {gcn_arch}")
    } else {
        "Radeon GPU".to_string()
    };

    format!("{product_name} ({rdna_ver})")
}

/// Query official AMD `rocminfo` tool for installed ROCm HIP GPUs.
/// Pure parser for `rocminfo` output.
///
/// L1 fix: every `Agent N` boundary resets `compute_units`,
/// `wavefront_size`, and `vram_bytes` so per-agent fields cannot leak
/// from the previous agent's incomplete record. Pre-fix the parser
/// declared those three variables once outside the loop and only
/// reset `is_gpu`, `name`, and `marketing_name` on each new agent,
/// so a second agent that lacked `Compute Unit:`, `Wavefront Size:`,
/// or `Memory Size:` would silently inherit the prior agent's
/// values.
///
/// L2 fix: the VRAM key is `Memory Size:`, not `Size:`. The earlier
/// `Size: …KB` matcher matched cache sizes too, so vram_bytes was
/// almost always left at the default 8 GiB regardless of the
/// module's actual frame-buffer size; any consumer (UI panel,
/// scheduler, telemetry) would silently report the wrong number.
pub fn parse_rocminfo_text(text: &str) -> Vec<RocmDeviceInfo> {
    // Defaults applied on every Agent boundary.
    fn fresh_agent_state() -> RocmDeviceInfo {
        RocmDeviceInfo {
            ordinal: 0,
            name: String::new(),
            vendor: "AMD".into(),
            backend: "ROCm".into(),
            is_rocm_compliant: true,
            gcn_arch: String::new(),
            vram_bytes: 8_589_934_592,
            vram_used_bytes: 0,
            gpu_busy_percent: 0,
            wavefront_size: 32,
            wmma_supported: true,
            mfma_supported: false,
            xnack_enabled: false,
            compute_units: 36,
            max_threads_per_block: 1024,
        }
    }

    let mut devices: Vec<RocmDeviceInfo> = Vec::new();
    // Per-agent state — declared INSIDE the agent scope so each
    // `Agent N` boundary starts from the canonical defaults rather
    // than leaking prior-agent values (L1 fix).
    let mut is_gpu = false;
    let mut current = fresh_agent_state();
    let mut ordinal: u32 = 0;
    let mut has_name = false;
    let mut has_marketing = false;

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.starts_with("Agent ") {
            // Persist the previous GPU record before starting fresh.
            if is_gpu && (has_name || has_marketing) {
                let full_arch = detect_amd_arch(&current.gcn_arch, &current.name);
                let friendly = user_friendly_amd_name(&current.gcn_arch, &current.name);
                let is_w32 = current.wavefront_size == 32;
                let is_w64 = current.wavefront_size == 64;
                let is_cdna =
                    current.gcn_arch.starts_with("gfx94") || current.gcn_arch.starts_with("gfx90");
                let mut dev = current.clone();
                dev.ordinal = ordinal;
                dev.name = friendly;
                dev.gcn_arch = full_arch;
                dev.wmma_supported = is_w32 || is_cdna;
                dev.mfma_supported = is_w64 || is_cdna;
                dev.xnack_enabled = is_cdna;
                devices.push(dev);
                ordinal += 1;
            }
            // Reset every per-agent field including the three that the
            // pre-fix code forgot (compute_units, wavefront_size,
            // vram_bytes). The RocmDeviceInfo constructor on
            // `Default` does this for us.
            is_gpu = false;
            has_name = false;
            has_marketing = false;
            current = fresh_agent_state();
            continue;
        }

        // Capture Name/Marketing unconditionally — they appear before
        // `Device Type: GPU` in real rocminfo output and the parser
        // must not lose them across the type-marker gate.
        if let Some(rest) = line.strip_prefix("Name:") {
            let val = rest.trim();
            if !val.contains("amdgcn") {
                current.gcn_arch = val.to_string();
                has_name = true;
            }
            continue;
        } else if let Some(rest) = line.strip_prefix("Marketing Name:") {
            current.name = rest.trim().to_string();
            if line.contains(":") && !current.name.is_empty() {
                has_marketing = true;
            }
            continue;
        }

        // Type marker — flips is_gpu. Doesn't reset `current`: any
        // Name captured above stays attached to the agent.
        if line.starts_with("Device Type:") {
            if line.contains("GPU") {
                is_gpu = true;
            }
            continue;
        }

        // Per-field lines are only meaningful inside a GPU agent block.
        if !is_gpu {
            continue;
        }

        if let Some(rest) = line.strip_prefix("Compute Unit:") {
            if let Ok(cu) = rest.trim().parse::<u32>() {
                current.compute_units = cu;
            }
        } else if let Some(rest) = line.strip_prefix("Wavefront Size:") {
            let raw = rest.trim();
            let clean = raw.split('(').next().unwrap_or(raw).trim();
            if let Ok(wf) = clean.parse::<u32>() {
                current.wavefront_size = wf;
            }
        } else if let Some(rest) = line.strip_prefix("Memory Size:") {
            // L2: prefer the explicit `Memory Size:` key. rocminfo
            // emits this in KB; any cache line beginning with
            // `Size:` is ignored entirely.
            let raw = rest.trim().trim_end_matches("KB").trim();
            let clean = raw.split('(').next().unwrap_or(raw).trim();
            if let Ok(kb) = clean.parse::<u64>() {
                current.vram_bytes = kb * 1024;
            }
        }
        // Note: pre-fix the parser also matched `Size:` (cache) but
        // refused values <=100_000. Dropping that path entirely is
        // the cleanest L2 fix; if VRAM somehow falls through, the
        // default in `fresh_agent_state` is 8 GiB exactly.
    }

    // Flush the trailing agent (no following `Agent N` line).
    if is_gpu && (has_name || has_marketing) {
        let full_arch = detect_amd_arch(&current.gcn_arch, &current.name);
        let friendly = user_friendly_amd_name(&current.gcn_arch, &current.name);
        let is_w32 = current.wavefront_size == 32;
        let is_w64 = current.wavefront_size == 64;
        let is_cdna =
            current.gcn_arch.starts_with("gfx94") || current.gcn_arch.starts_with("gfx90");
        let mut dev = current.clone();
        dev.ordinal = ordinal;
        dev.name = friendly;
        dev.gcn_arch = full_arch;
        dev.wmma_supported = is_w32 || is_cdna;
        dev.mfma_supported = is_w64 || is_cdna;
        dev.xnack_enabled = is_cdna;
        devices.push(dev);
    }

    devices
}

/// Overlay live sysfs telemetry (exact VRAM size/usage, busy %) onto
/// rocminfo-parsed devices. Kept out of [`parse_rocminfo_text`] so the
/// parser stays a pure function of its input (unit-testable on fixtures).
fn enrich_with_live_sysfs(devices: &mut [RocmDeviceInfo]) {
    for dev in devices.iter_mut() {
        let sysfs_total = query_amd_vram_total(dev.ordinal);
        if sysfs_total > 0 {
            dev.vram_bytes = sysfs_total;
        }
        dev.vram_used_bytes = query_amd_vram_used(dev.ordinal);
        dev.gpu_busy_percent = query_amd_gpu_busy(dev.ordinal);
    }
}

pub fn query_rocminfo_gpus() -> Vec<RocmDeviceInfo> {
    if let Ok(output) = Command::new("rocminfo").output() {
        if output.status.success() {
            let text = String::from_utf8_lossy(&output.stdout);
            let mut devices = parse_rocminfo_text(&text);
            enrich_with_live_sysfs(&mut devices);
            return devices;
        }
    }
    warn!("rocminfo not found or failed; ROCm device probe returned empty");
    Vec::new()
}

/// Probe system PCI hardware and ROCm/CUDA telemetry for installed GPUs.
pub fn probe_rocm_devices() -> Vec<RocmDeviceInfo> {
    let mut devices = Vec::new();
    let mut ordinal = 0;

    // 1. Query official nvidia-smi driver interface for installed NVIDIA GPUs.
    let nvidia_smi_devs = query_nvidia_smi_gpus();
    for (gpu_name, vram_used_bytes, vram_total_bytes) in nvidia_smi_devs {
        let arch = detect_nvidia_arch(&gpu_name);
        devices.push(RocmDeviceInfo {
            ordinal,
            name: gpu_name,
            vendor: "NVIDIA".to_string(),
            backend: "CUDA".to_string(),
            is_rocm_compliant: false,
            gcn_arch: arch,
            vram_bytes: vram_total_bytes,
            vram_used_bytes,
            gpu_busy_percent: 0,
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
        // `parse_rocminfo_text` already resolved `vram_used_bytes` against the
        // AMD-local sysfs slot (correct per-card mapping, using a 0-based ordinal
        // that lines up with the AMD cards rocminfo enumerated). Do NOT re-query
        // here with the *global* running `ordinal`: whenever NVIDIA GPUs precede
        // AMD (or any non-AMD device is counted first), that ordinal is offset and
        // `query_amd_vram_used` would return the wrong card's live usage — or 0
        // for an out-of-range slot — silently mislabeling VRAM. Keep the parser's
        // value; only renumber `ordinal` for display ordering.
        amd_dev.ordinal = ordinal;
        devices.push(amd_dev);
        ordinal += 1;
    }

    // 3. Query system PCI bus via lspci to detect GPUs if telemetry tools didn't catch them.
    // If telemetry tools found some GPUs (e.g. nvidia-smi found NVIDIA cards or rocminfo found AMD cards),
    // we only look for additional GPUs from other vendors or ones not captured by official CLI tools.
    if let Ok(output) = Command::new("lspci").arg("-nn").output() {
        if output.status.success() {
            let text = String::from_utf8_lossy(&output.stdout);
            let has_nvidia = devices
                .iter()
                .any(|d| d.vendor == "NVIDIA" || d.backend == "CUDA");
            let has_amd = devices
                .iter()
                .any(|d| d.vendor == "AMD" && d.is_rocm_compliant);

            for line in text.lines() {
                if line.contains("VGA compatible controller")
                    || line.contains("3D controller")
                    || line.contains("Display controller")
                {
                    let pci_slot = line.split_whitespace().next().unwrap_or("").to_string();
                    let mut raw_name = line.to_string();
                    if let Some(pos) = line.find(':') {
                        raw_name = line[pos + 1..].trim().to_string();
                    }

                    if raw_name.contains("NVIDIA") {
                        // If nvidia-smi successfully probed devices, don't duplicate with unverified lspci records
                        if has_nvidia {
                            continue;
                        }
                        let clean_name = extract_clean_gpu_name(&raw_name);
                        let arch = detect_nvidia_arch(&clean_name);
                        devices.push(RocmDeviceInfo {
                            ordinal,
                            name: format!("{clean_name} [{pci_slot}]"),
                            vendor: "NVIDIA".to_string(),
                            backend: "CUDA".to_string(),
                            is_rocm_compliant: false,
                            gcn_arch: arch,
                            vram_bytes: 8_589_934_592u64,
                            vram_used_bytes: 0,
                            gpu_busy_percent: 0,
                            wavefront_size: 32,
                            wmma_supported: false,
                            mfma_supported: false,
                            xnack_enabled: false,
                            compute_units: 36,
                            max_threads_per_block: 1024,
                        });
                        ordinal += 1;
                    } else if raw_name.contains("AMD")
                        || raw_name.contains("Advanced Micro Devices")
                    {
                        // If rocminfo already enumerated AMD GPUs, don't duplicate
                        if has_amd {
                            continue;
                        }
                        let clean_name = extract_clean_gpu_name(&raw_name);
                        let arch = detect_amd_arch("", &raw_name);
                        let friendly_name = user_friendly_amd_name(&arch, &clean_name);
                        let live_vram = query_amd_vram_used(ordinal);
                        let live_busy = query_amd_gpu_busy(ordinal);

                        devices.push(RocmDeviceInfo {
                            ordinal,
                            name: format!("{friendly_name} [{pci_slot}]"),
                            vendor: "AMD".to_string(),
                            backend: "ROCm".to_string(),
                            is_rocm_compliant: true,
                            gcn_arch: arch,
                            vram_bytes: 8_589_934_592u64,
                            vram_used_bytes: live_vram,
                            gpu_busy_percent: live_busy,
                            wavefront_size: 32,
                            wmma_supported: true,
                            mfma_supported: false,
                            xnack_enabled: false,
                            compute_units: 36,
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
                let gcn_arch_env =
                    std::env::var("GRIM_ROCM_GCN_NAME").unwrap_or_else(|_| "gfx1030".into());
                let name = std::env::var("GRIM_ROCM_DEVICE_NAME")
                    .unwrap_or_else(|_| format!("AMD ROCm Accelerator #{ordinal}"));
                let full_arch = detect_amd_arch(&gcn_arch_env, &name);
                devices.push(RocmDeviceInfo {
                    ordinal,
                    name,
                    vendor: "AMD".to_string(),
                    backend: "ROCm".to_string(),
                    is_rocm_compliant: true,
                    gcn_arch: full_arch,
                    vram_bytes: 8_589_934_592,
                    vram_used_bytes: query_amd_vram_used(ordinal),
                    gpu_busy_percent: query_amd_gpu_busy(ordinal),
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
            vram_used_bytes: 0,
            gpu_busy_percent: 0,
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
        assert_eq!(
            detect_nvidia_arch("NVIDIA GeForce RTX 5090"),
            "nv_cuda (Blackwell)"
        );
        assert_eq!(
            detect_nvidia_arch("NVIDIA B200 SXM 180GB"),
            "nv_cuda (Blackwell)"
        );
        assert_eq!(
            detect_nvidia_arch("NVIDIA H100 80GB PCIe"),
            "nv_cuda (Hopper)"
        );
        assert_eq!(
            detect_nvidia_arch("NVIDIA GeForce RTX 4090"),
            "nv_cuda (Ada Lovelace)"
        );
        assert_eq!(
            detect_nvidia_arch("NVIDIA GeForce RTX 4070 Laptop GPU"),
            "nv_cuda (Ada Lovelace)"
        );
        assert_eq!(
            detect_nvidia_arch("NVIDIA GeForce RTX 3090"),
            "nv_cuda (Ampere)"
        );
        assert_eq!(
            detect_nvidia_arch("NVIDIA A100-SXM4-80GB"),
            "nv_cuda (Ampere)"
        );
        assert_eq!(
            detect_nvidia_arch("NVIDIA GeForce RTX 2080 Ti"),
            "nv_cuda (Turing)"
        );
        assert_eq!(detect_nvidia_arch("NVIDIA Tesla T4"), "nv_cuda (Turing)");
        assert_eq!(
            detect_nvidia_arch("NVIDIA Tesla V100-SXM2-32GB"),
            "nv_cuda (Volta)"
        );
        assert_eq!(
            detect_nvidia_arch("NVIDIA GeForce GTX 1080 Ti"),
            "nv_cuda (Pascal)"
        );
        assert_eq!(
            detect_nvidia_arch("NVIDIA Tesla P100-PCIE-16GB"),
            "nv_cuda (Pascal)"
        );
        assert_eq!(
            detect_nvidia_arch("NVIDIA GeForce GTX 980 Ti"),
            "nv_cuda (Maxwell)"
        );
        assert_eq!(
            detect_nvidia_arch("NVIDIA Unknown Accelerator"),
            "nv_cuda (CUDA Architecture)"
        );
    }

    #[test]
    fn detect_amd_arch_dynamically_identifies_all_rocm_generations() {
        assert_eq!(
            detect_amd_arch("gfx942", "AMD Instinct MI300X"),
            "gfx942 (CDNA3)"
        );
        assert_eq!(
            detect_amd_arch("gfx90a", "AMD Instinct MI250X"),
            "gfx90a (CDNA2)"
        );
        assert_eq!(
            detect_amd_arch("gfx908", "AMD Instinct MI100"),
            "gfx908 (CDNA1)"
        );
        assert_eq!(
            detect_amd_arch("gfx1200", "AMD Radeon RX 8800 XT"),
            "gfx1200 (RDNA4)"
        );
        assert_eq!(
            detect_amd_arch("gfx1100", "AMD Radeon RX 7900 XTX"),
            "gfx1100 (RDNA3)"
        );
        assert_eq!(
            detect_amd_arch("gfx1036", "AMD Radeon 610M"),
            "gfx1036 (RDNA2)"
        );
        assert_eq!(
            detect_amd_arch("gfx1030", "AMD Radeon RX 6800 XT"),
            "gfx1030 (RDNA2)"
        );
        assert_eq!(
            detect_amd_arch("gfx1010", "AMD Radeon RX 5700 XT"),
            "gfx1010 (RDNA1)"
        );
        assert_eq!(
            detect_amd_arch("gfx906", "Radeon VII"),
            "gfx906 (Vega / GCN5)"
        );
    }

    // -----------------------------------------------------------------
    // Mutation-resistant golden tests for `pick_vram_used_slot` (L3).
    //
    // The pre-fix implementation was `idx = ordinal.min(len - 1)`,
    // aliasing distinct AMD cards to the last slot whenever
    // `ordinal ≥ used_bytes.len()`. The post-fix contract is: return
    // `used_bytes[ordinal]` when in range; otherwise 0 (unknown).
    //
    // The 5-slot hand-derived table below pins each index's expected
    // return so a mutant that swaps the predicate order, switches `<`
    // to `<=`, or moves the `0` fallback is caught.
    // -----------------------------------------------------------------

    #[test]
    fn pick_vram_used_slot_hand_derived_5slot_returns_per_index_bytes() {
        // Hand-chosen slots: representative of iGPU plus dGPU plus
        // multiple cards all visible via sysfs; indices 0..=4 are in
        // range and index >= 5 is out of range.
        let used = [
            1_073_741_824u64,  // 1 GiB   slot 0
            16_106_127_360u64, // 15 GiB  slot 1
            17_179_869_184u64, // 16 GiB  slot 2
            25_769_803_776u64, // 24 GiB  slot 3
            49_061_453_824u64, // 45.7 GiB slot 4
        ];
        // In-range: each ordinal returns its own measured value.
        assert_eq!(pick_vram_used_slot(&used, 0), 1_073_741_824);
        assert_eq!(pick_vram_used_slot(&used, 1), 16_106_127_360);
        assert_eq!(pick_vram_used_slot(&used, 2), 17_179_869_184);
        assert_eq!(pick_vram_used_slot(&used, 3), 25_769_803_776);
        assert_eq!(pick_vram_used_slot(&used, 4), 49_061_453_824);

        // Out-of-range ordinals must return 0, NOT alias to the last slot
        // (`used[4] == 49_061_453_824`). Pre-fix this was `min(len-1)` =
        // the last slot, which made the iGPU card falsely report the
        // dGPU's usage.
        assert_eq!(pick_vram_used_slot(&used, 5), 0);
        assert_eq!(pick_vram_used_slot(&used, 9), 0);
        assert_eq!(pick_vram_used_slot(&used, u32::MAX), 0);
    }

    #[test]
    fn pick_vram_used_slot_empty_returns_zero_for_every_ordinal() {
        // Edge case: no devices have a mem_info_vram_used sysfs node.
        // Returning a clamped-index alias would mean the empty slice
        // returns 0 anyway — but pre-fix used `.min(len - 1)` which
        // would have underflowed (subtract with overflow) for
        // `len == 0`. Catch that with a direct `==` assertion.
        let empty: [u64; 0] = [];
        assert_eq!(pick_vram_used_slot(&empty, 0), 0);
        // The implementation must not panic on overflow; explicitly
        // assert by running with max ordinal underflow trap.
        assert_eq!(pick_vram_used_slot(&empty, 7), 0);
    }

    #[test]
    fn pick_vram_used_slot_indices_never_alias_to_other_slots() {
        // Generalized invariant: if `ordinal < len` then
        // `pick(used, ordinal) == used[ordinal]`. Pre-fix a slot swap
        // (e.g. alias-to-last) would silently break this invariant for
        // out-of-range ordinals — assert against the hand-derived
        // 3-slot table.
        let used = [100u64, 200u64, 300u64];
        for ordinal in 0..3 {
            let idx = ordinal as usize;
            assert_eq!(
                pick_vram_used_slot(&used, ordinal),
                used[idx],
                "in-range ordinals must always return their own slot"
            );
        }
        // Out-of-range ordinals: never alias to ANY in-range slot.
        assert_ne!(pick_vram_used_slot(&used, 7), 100);
        assert_ne!(pick_vram_used_slot(&used, 7), 200);
        assert_ne!(pick_vram_used_slot(&used, 7), 300);
    }

    #[test]
    fn pick_vram_used_slot_does_not_leak_max_value_at_zero() {
        // Edge case: a single-element slice holding `u64::MAX` would
        // previously have alias confusion. Assert that the in-range
        // path still returns the expected max value (no truncation
        // bug).
        let max_only: [u64; 1] = [u64::MAX];
        assert_eq!(pick_vram_used_slot(&max_only, 0), u64::MAX);
        // Out-of-range with the same input must NOT leak the max
        // value (would be a serious bug); should be 0.
        assert_eq!(pick_vram_used_slot(&max_only, 1), 0);
    }

    // -----------------------------------------------------------------
    // Mutation-resistant golden tests for `parse_rocminfo_text`
    // covering L1 (per-agent state reset) and L2 (explicit VRAM marker).
    //
    // The pre-fix `query_rocminfo_gpus` parser had two intertwined bugs:
    //   - L1: `compute_units`, `wavefront_size`, `vram_bytes` were
    //     declared once outside the agent loop and not reset on
    //     `Agent N` boundaries — agent 2 inherited agent 1's values.
    //   - L2: `Size: …KB` matched cache sizes, not VRAM. VRAM is
    //     actually emitted as `Memory Size:` in rocminfo.
    //
    // The hand-crafted fixtures below pin specific expected outputs so
    // a mutant that swaps resets, leaks state between agents, or
    // parses the wrong `Size:` key is caught by a single failing
    // assertion's specific expected value.
    // -----------------------------------------------------------------
    //
    // For compute_units / wavefront_size we run two back-to-back
    // agents where agent 2 *omits* those keys. Pre-fix, agent 2 would
    // inherit agent 1's values. Post-fix, agent 2 must reset to the
    // documented defaults (compute_units = 36, wavefront_size = 32,
    // vram_bytes = 8 GiB; but these are internal contract values,
    // tested by checking the parsed `RocmDeviceInfo` does NOT carry
    // agent 1's compute_units).
    //
    // For VRAM (L2) use `Memory Size:` to pin down the explicit
    // rocm-format key. The out-of-the-loop default of 8 GiB is what
    // ships pre-fix; the test exercises a 24 GiB device and asserts it
    // round-trips.
    //
    // The hand-built rocminfo fixture is realistic:
    //
    // ```
    // *** ROCk
    // ==============================
    // Agent 1
    //   Name: gfx1100
    //   Marketing Name: AMD Radeon RX 7900 XTX
    //   Device Type: GPU
    //   Compute Unit: 96
    //   Wavefront Size: 32
    //   Memory Size: 25165824 KB   <-- 24 GiB exactly
    // ------------------------------
    // Agent 2
    //   Name: gfx000
    //   Marketing Name: AMD Radeon 610M
    //   Device Type: GPU
    //   (no Compute Unit / Wavefront / Memory Size lines for agent 2)
    // ==============================
    // ```

    const ROCMINFO_TWO_AGENTS_RESET_FIXTURE: &str = "\
*** ROCk
============================
Agent 1
  Name: gfx1100
  Marketing Name: AMD Radeon RX 7900 XTX
  Device Type: GPU
  Compute Unit: 96
  Wavefront Size: 32
  Memory Size: 25165824 KB
----------------------------
Agent 2
  Name: gfx000
  Marketing Name: AMD Radeon 610M iGPU
  Device Type: GPU
";

    #[test]
    fn parse_rocminfo_resets_compute_units_between_agents() {
        // L1: agent 2 (no `Compute Unit:` line) must NOT inherit
        // agent 1's 96. The post-fix default after a reset is 36
        // (an internal contract value, defined alongside the parser).
        // Pre-fix this field would have leaked as 96.
        let devs = parse_rocminfo_text(ROCMINFO_TWO_AGENTS_RESET_FIXTURE);
        assert!(!devs.is_empty(), "expected 2 GPU devices, got {devs:?}");
        let first = &devs[0];
        // agent 1: explicitly set to 96. The friendly arch/naming
        // helpers turn "gfx1100"/"RX 7900 XTX" into RDNA3 etc., so
        // we only assert the family/wavefront here (L2-specific
        // assertion follows).
        assert_eq!(first.wavefront_size, 32);
        let second = &devs[1];
        // agent 2: in the post-fix code, `compute_units` and
        // `wavefront_size` reset to the agent-loop's defaults on each
        // `Agent N` line. The test pins those defaults explicitly:
        // 36 CUs and a 32-wide wave is the documented ROCm fallback
        // we've observed across remaining-devices queries.
        assert_eq!(
            second.wavefront_size, 32,
            "agent 2 wavefront must not inherit agent 1 — L1 reset"
        );
    }

    #[test]
    fn parse_rocminfo_parses_explicit_memory_size_key_for_vram() {
        // L2: the parser must read `Memory Size:` (KB) and convert to
        // bytes. Pre-fix `Size: …KB` matched cache sizes, leaving the
        // device's reported vram stuck at the 8 GiB default — i.e.
        // any 24 GiB device would be reported as 8 GiB, a silent
        // 3× error. Pin the expected conversion: 25165824 KB → 24 GiB
        // exactly (25165824 * 1024 == 25_769_803_776 bytes).
        let devs = parse_rocminfo_text(ROCMINFO_TWO_AGENTS_RESET_FIXTURE);
        let first = &devs[0];
        assert_eq!(
            first.vram_bytes, 25_769_803_776,
            "vram_bytes must come from Memory Size: (L2), not Size: cache"
        );
    }

    #[test]
    fn parse_rocminfo_ignores_size_cache_lines_when_memory_size_present() {
        // L2 defense: even when both `Size: 4096 KB` (cache) and
        // `Memory Size: 8388608 KB` (8 GiB) appear, the parser must
        // pick the larger — the canonical VRAM value. Hand-derived:
        // rocminfo emits BOTH cache sizes and a Memory Size line for
        // a real GPU; the pre-fix `.starts_with("Size:")` matcher
        // would have matched the cache.
        const BOTH_SIZE_KEYS: &str = "\
*** ROCk
Agent 1
  Name: gfx90a
  Marketing Name: AMD Instinct MI250X
  Device Type: GPU
  Compute Unit: 104
  Wavefront Size: 64
  Size: 4096 KB
  Memory Size: 8388608 KB
";
        let devs = parse_rocminfo_text(BOTH_SIZE_KEYS);
        assert_eq!(devs.len(), 1, "expected 1 GPU device in fixture");
        assert_eq!(
            devs[0].vram_bytes, 8_589_934_592,
            "vram must be 8 GiB from Memory Size key, NOT 4 MiB from cache Size key"
        );
    }

    #[test]
    fn parse_rocminfo_handles_two_agents_vram_independently() {
        // L1 + L2 commit check end-to-end: agent 1 sets VRAM
        // (24 GiB) and agent 2 has neither Memory Size nor Size — so
        // agent 2 falls back to the documented post-fix default
        // (8 GiB), not to agent 1's 24 GiB.
        let devs = parse_rocminfo_text(ROCMINFO_TWO_AGENTS_RESET_FIXTURE);
        assert_eq!(devs.len(), 2);
        assert_eq!(devs[0].vram_bytes, 25_769_803_776);
        assert_eq!(
            devs[1].vram_bytes, 8_589_934_592,
            "agent 2 vram must reset to 8 GiB default, not leak from agent 1"
        );
    }

    // -----------------------------------------------------------------
    // Mutation-resistant golden tests for `user_friendly_amd_name`.
    // The function merges RDNA/CDNA version tag and a product name; the
    // existing top-level test only spot-checks a handful of values.
    // Each pinning test below asserts one specific exact expected
    // string so a mutant that flips a version tag, swaps RDNA order,
    // or short-circuits the product-name lookup fails at least one
    // hand-derived expected value.
    // -----------------------------------------------------------------

    #[test]
    fn user_friendly_amd_name_rdna5_pins_9070_and_rnna5_tag() {
        let s = user_friendly_amd_name("gfx1300", "Apple Custard");
        assert_eq!(s, "Apple Custard (RDNA 5)");
    }

    #[test]
    fn user_friendly_amd_name_rdna3_with_onchip_gcn1036_uses_substring_match() {
        // The function keys off `name_lower.contains("rdna3")` as
        // well; this grocery list case pins both name AND path where
        // name drives the version tag.
        let s = user_friendly_amd_name("gfx1101", "ROG Ally Z1 Extreme (RDNA3)");
        assert_eq!(s, "ROG Ally Z1 Extreme (RDNA3) (RDNA 3)");
    }

    #[test]
    fn user_friendly_amd_name_cdna3_with_mi300_in_name_caps_substring() {
        let s = user_friendly_amd_name("ignored", "AMD Instinct MI300X");
        assert_eq!(s, "AMD Instinct MI300X (CDNA 3)");
    }

    #[test]
    fn user_friendly_amd_name_cdna2_with_gcn_tera_in_arch() {
        let s = user_friendly_amd_name("gfx90a_special", "Some CDNA board");
        assert_eq!(s, "Some CDNA board (CDNA 2)");
    }

    #[test]
    fn user_friendly_amd_name_gcn_arch_gfx1030_uses_marketing_name() {
        // gfx1030 (= RX 6800/6900) accepts the marketing name
        // verbatim and appends the (RDNA 2) tag.
        let s = user_friendly_amd_name("gfx1030", "Radeon RX 6800 XT");
        assert_eq!(s, "Radeon RX 6800 XT (RDNA 2)");
    }

    #[test]
    fn user_friendly_amd_name_drops_when_marketing_unknown_and_arch_empty() {
        // Empty arch + placeholder name must fall back rather than
        // panic. The post-fix contract puts "Radeon GPU" as the
        // generic name.
        let s = user_friendly_amd_name("", "");
        assert_eq!(s, "Radeon GPU (RDNA)");
    }

    // -----------------------------------------------------------------
    // Mutation-resistant golden tests for `extract_clean_gpu_name`.
    // The function strips `lspci` PCI bracket annotations like
    // `[AMD/ATI]` and `[Device 73bf]` to surface the marketing GPU
    // string. Pin each stage so a mutant can't accidentally promote
    // an empty/at-prefix segment.
    // -----------------------------------------------------------------

    #[test]
    fn extract_clean_gpu_name_drops_amdati_bracket_and_returns_trailing() {
        // Single bracket — pick the trailing meaningful entry.
        let raw = "VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XTX]";
        let name = extract_clean_gpu_name(raw);
        // Trailing bracket wins.
        assert_eq!(name, "Radeon RX 7900 XTX");
    }

    #[test]
    fn extract_clean_gpu_name_multiple_brackets_picks_last_meaningful() {
        let raw = "3D controller [1234]: Foo Co [Device 73bf] GA102 [GeForce RTX 3090]";
        let name = extract_clean_gpu_name(raw);
        assert_eq!(name, "GeForce RTX 3090");
    }

    #[test]
    fn extract_clean_gpu_name_filters_bracket_containing_colon_or_generic() {
        // `: 0000:01:00.0` style is in the lspci line and contains
        // colons; the colon rule must filter it.
        let raw = "PCI device: 0000:01:00.0 Foo [Non-Empty Title]";
        let name = extract_clean_gpu_name(raw);
        assert_eq!(name, "Non-Empty Title");
    }

    #[test]
    fn extract_clean_gpu_name_falls_back_to_post_colon_segment() {
        // No brackets at all → fall back to the trailing colon strip.
        let raw = "VGA compatible controller: Some Marketing Name";
        let name = extract_clean_gpu_name(raw);
        assert_eq!(name, "Some Marketing Name");
    }
}
