//! JIT compilation, caching, and kernel module loading via NVCC and the CUDA driver API.

use std::collections::HashMap;
use std::ffi::c_void;
use std::fs;
use std::process::Command;
use std::sync::LazyLock;
use std::sync::Mutex;

use grim_tensor::error::{Error, Result};
use crate::device::handles::{cuInit, cuModuleLoadData, CUmodule};

#[derive(Debug, Clone, Copy)]
pub struct SendCmodule(pub CUmodule);

// SAFETY: `CUmodule` is owned and managed by the CUDA driver. Concurrent
// `cuModuleLoadData` / `cuLaunchKernel` calls on the same module are
// serialized by the driver; the JIT cache (`JIT_CACHE`) is additionally
// protected by a `Mutex`. `Send` is safe because the driver tracks the
// module independently of the creating thread. `Sync` is safe because
// the driver itself serializes concurrent launches on the same module.
unsafe impl Send for SendCmodule {}
unsafe impl Sync for SendCmodule {}

static JIT_CACHE: LazyLock<Mutex<HashMap<u64, SendCmodule>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Compiles and loads a CUDA kernel module, caching by source hash.
pub fn compile_and_load_kernel(src: &str, device_ordinal: usize) -> Result<CUmodule> {
    let hash = seahash::hash(src.as_bytes());
    let mut cache = JIT_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&module) = cache.get(&hash) {
        return Ok(module.0);
    }

    // SAFETY: `cuInit(0)` initializes the CUDA driver API. It is a no-op if
    // already initialized, and must be called before any other driver API call.
    unsafe {
        let res = cuInit(0);
        if res != 0 {
            return Err(Error::Backend(format!("cuInit failed with status {}", res)));
        }
    }

    let cache_dir = std::env::var("CARGO_TARGET_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::env::current_dir().unwrap_or_default().join("target"))
        .join("grim_cuda_cache");
    fs::create_dir_all(&cache_dir).ok();

    let cu_path = cache_dir.join(format!("{}.cu", hash));
    let ptx_path = cache_dir.join(format!("{}.ptx", hash));

    fs::write(&cu_path, src)
        .map_err(|e| Error::Backend(format!("Failed to write CUDA source: {e}")))?;

    /// Resolves the path to the `nvcc` executable.
    fn resolve_nvcc_path() -> std::path::PathBuf {
        if let Ok(env_nvcc) = std::env::var("NVCC") {
            let p = std::path::PathBuf::from(env_nvcc);
            if p.exists() {
                return p;
            }
        }
        if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
            let p = std::path::PathBuf::from(cuda_path).join("bin").join("nvcc");
            if p.exists() {
                return p;
            }
        }

        let candidate_paths = [
            "/opt/cuda/bin/nvcc",
            "/usr/local/cuda/bin/nvcc",
            "/usr/bin/nvcc",
        ];

        for path_str in candidate_paths {
            let p = std::path::PathBuf::from(path_str);
            if p.exists() {
                return p;
            }
        }

        std::path::PathBuf::from("nvcc")
    }

    let nvcc = resolve_nvcc_path();

    let status = Command::new(&nvcc)
        .arg("-ptx")
        .arg("-O3")
        .arg("--gpu-architecture=sm_80")
        .arg(&cu_path)
        .arg("-o")
        .arg(&ptx_path)
        .status();

    let success = match status {
        Ok(s) => s.success(),
        Err(_) => false,
    };

    if !success {
        let status2 = Command::new(&nvcc)
            .arg("-ptx")
            .arg("-O3")
            .arg(&cu_path)
            .arg("-o")
            .arg(&ptx_path)
            .status();
        let success2 = match status2 {
            Ok(s) => s.success(),
            Err(_) => false,
        };
        if !success2 {
            return Err(Error::Backend(format!(
                "nvcc failed to compile CUDA kernel for device {}",
                device_ordinal
            )));
        }
    }

    let ptx_code = fs::read_to_string(&ptx_path)
        .map_err(|e| Error::Backend(format!("Failed to read compiled PTX: {e}")))?;
    let ptx_c_str = std::ffi::CString::new(ptx_code)
        .map_err(|e| Error::Backend(format!("Failed to convert PTX to CString: {e}")))?;

    let mut module: CUmodule = std::ptr::null_mut();
    // SAFETY: `cuModuleLoadData` parses PTX text from memory and compiles it
    // into a device-specific module. `ptx_c_str` is a valid null-terminated
    // string; `module` is initialized to null and checked against 0.
    unsafe {
        let res = cuModuleLoadData(&mut module, ptx_c_str.as_ptr() as *const c_void);
        if res != 0 {
            return Err(Error::Backend(format!(
                "cuModuleLoadData failed with status {}",
                res
            )));
        }
    }

    cache.insert(hash, SendCmodule(module));
    Ok(module)
}
