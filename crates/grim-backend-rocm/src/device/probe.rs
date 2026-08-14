//! Device-feature probes — small functions that ask HIP runtime a [see: `hipDeviceGetAttribute`, `unsafe`]

use crate::device::util::detect_gpu_arch;
use grim_tensor::error::{Error, Result};
use std::fs;
use std::path::PathBuf;

use super::handles::{
    HIP_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    HIP_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS, HIP_DEVICE_ATTRIBUTE_WARP_SIZE,
    hipDeviceGetAttribute, hipSetDevice,
};

/// HIP attribute ID for maximum shared memory (LDS) per block.
pub const HIP_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: i32 = 3;

/// Query maximum threads per block for a specific device ordinal.
pub fn max_threads_per_block(device_ordinal: usize) -> u32 {
    let mut val: i32 = 0;
    unsafe {
        let _ = hipSetDevice(device_ordinal as i32);
        let status = hipDeviceGetAttribute(
            &mut val,
            HIP_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            device_ordinal as i32,
        );
        if status == 0 && val > 0 {
            val as u32
        } else {
            1024
        }
    }
}

/// Query active compute unit (multiprocessor) count for a specific device ordinal.
pub fn active_cu_count(device_ordinal: usize) -> u32 {
    let mut val: i32 = 0;
    unsafe {
        let _ = hipSetDevice(device_ordinal as i32);
        let status = hipDeviceGetAttribute(
            &mut val,
            HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            device_ordinal as i32,
        );
        if status == 0 && val > 0 {
            val as u32
        } else {
            64
        }
    }
}

/// Query wavefront execution size (32 or 64) for a specific device ordinal.
pub fn wavefront_size(device_ordinal: usize) -> u32 {
    let mut val: i32 = 0;
    unsafe {
        let _ = hipSetDevice(device_ordinal as i32);
        let status = hipDeviceGetAttribute(
            &mut val,
            HIP_DEVICE_ATTRIBUTE_WARP_SIZE,
            device_ordinal as i32,
        );
        if status == 0 && val > 0 {
            val as u32
        } else {
            32
        }
    }
}

/// Query maximum shared memory (LDS) in bytes per block for a specific device ordinal.
pub fn max_shared_mem(device_ordinal: usize) -> u32 {
    let mut val: i32 = 0;
    unsafe {
        let _ = hipSetDevice(device_ordinal as i32);
        let status = hipDeviceGetAttribute(
            &mut val,
            HIP_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
            device_ordinal as i32,
        );
        if status == 0 && val > 0 {
            val as u32
        } else {
            384 * 1024
        }
    }
}

/// XNACK probe for unified memory availability. Returns true if the [see: `hipMemAdvise`]
pub fn probe_xnack(device_ordinal: usize) -> bool {
    let mut val: i32 = 0;
    unsafe {
        let _ = hipSetDevice(device_ordinal as i32);
        let status = hipDeviceGetAttribute(
            &mut val,
            HIP_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
            device_ordinal as i32,
        );
        status == 0 && val == 1
    }
}

/// System ROCm installation info resolved dynamically.
#[derive(Debug, Clone)]
pub struct SystemRocmInfo {
    /// Resolved root path of the ROCm installation.
    pub path: PathBuf,
    /// Read version string (e.g. "6.1.0"), or "unknown".
    pub version: String,
}

/// Dynamic ROCm runtime discovery. Queries environment variables and fallbacks
pub fn probe_system_rocm() -> Result<SystemRocmInfo> {
    let paths_to_check = [
        std::env::var("ROCM_PATH").ok(),
        std::env::var("HIP_PATH").ok(),
        Some("/opt/rocm".to_string()),
        Some("/usr/local/rocm".to_string()),
    ];

    for path_opt in &paths_to_check {
        if let Some(path_str) = path_opt {
            let path = PathBuf::from(path_str);
            if path.exists() {
                // Try reading .info/version file
                let version_file = path.join(".info/version");
                let version = if version_file.exists() {
                    fs::read_to_string(&version_file)
                        .map(|s| s.trim().to_string())
                        .unwrap_or_else(|_| "unknown".to_string())
                } else {
                    "unknown".to_string();
                    // Try parsing version from bin/hipcc --version or fallback
                    "unknown".to_string()
                };

                return Ok(SystemRocmInfo { path, version });
            }
        }
    }

    Err(Error::Backend(
        "ROCm installation not found on the system. Ensure ROCM_PATH or HIP_PATH is set.".into(),
    ))
}

/// GPU hardware capability snapshot queried dynamically from the host active device.
#[derive(Debug, Clone)]
pub struct HostGpuCapabilities {
    /// Detected GCN target architecture (e.g., "gfx1100").
    pub gcn: String,
    /// Detected wavefront execution size (32 or 64).
    pub wavefront_size: u32,
    /// Maximum Local Data Share (LDS) shared memory size in bytes.
    pub lds_size_bytes: u32,
}

/// Query host GPU capabilities for a specific ordinal using the active system HIP FFI.
pub fn probe_host_gpu(device_ordinal: usize) -> Result<HostGpuCapabilities> {
    let gcn = detect_gpu_arch(device_ordinal as i32);
    let mut warp_val: i32 = 0;
    let mut lds_val: i32 = 0;

    unsafe {
        let set_status = hipSetDevice(device_ordinal as i32);
        if set_status != 0 {
            return Err(Error::Backend(format!(
                "hipSetDevice failed for ordinal {device_ordinal}: error code {set_status}"
            )));
        }

        let warp_status = hipDeviceGetAttribute(
            &mut warp_val,
            HIP_DEVICE_ATTRIBUTE_WARP_SIZE,
            device_ordinal as i32,
        );
        if warp_status != 0 {
            return Err(Error::Backend(format!(
                "hipDeviceGetAttribute WARP_SIZE failed: error code {warp_status}"
            )));
        }

        let lds_status = hipDeviceGetAttribute(
            &mut lds_val,
            HIP_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
            device_ordinal as i32,
        );
        if lds_status != 0 {
            return Err(Error::Backend(format!(
                "hipDeviceGetAttribute MAX_SHARED_MEMORY failed: error code {lds_status}"
            )));
        }
    }

    Ok(HostGpuCapabilities {
        gcn,
        wavefront_size: warp_val as u32,
        lds_size_bytes: lds_val as u32,
    })
}

#[cfg(test)]
mod probe_self_tests {
    use super::*;

    #[test]
    fn probe_xnack_returns_bool() {
        let _: bool = probe_xnack(0);
    }
}
