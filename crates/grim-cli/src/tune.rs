//! Hardware JIT kernel tuning and autotune persistence command.
//!
//! Evaluates empirical FCP tile selection and JIT compilation across canonical GEMM workload shapes
//! on the host ROCm GPU to pre-populate `{gpu_arch}.json` autotuner maps and compiled `.hsaco` files.

use grim_tensor::error::Result;
use std::path::PathBuf;

/// Run hardware-tuned JIT compilation and empirical FCP tile search for canonical shapes.
///
/// Sweeps inference GEMM shapes on the selected ROCm device and saves the resulting
/// autotune configs and `.hsaco` binaries to `output_dir`.
pub fn cmd_tune(device_ordinal: usize, output_dir: Option<String>) -> Result<()> {
    println!("=== Grim Hardware-Adaptive Kernel Tuner ===");
    println!("Targeting GPU device ordinal: {}", device_ordinal);

    let hsaco_dir = match output_dir {
        Some(d) => PathBuf::from(d),
        None => std::env::var("GRIM_HSACO_CACHE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                let mut p = dirs_next().unwrap_or_else(std::env::temp_dir);
                p.push("grim_hsaco_cache");
                p
            }),
    };

    println!("Output cache directory: {}", hsaco_dir.display());

    #[cfg(feature = "rocm")]
    {
        println!("[grim-tune] Initializing ROCm device {}...", device_ordinal);
        let device = match grim_backend_rocm::RocmDevice::try_new(device_ordinal) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("[grim-tune] ERROR: Failed to initialize ROCm device: {}", e);
                eprintln!("[grim-tune] Skipping tuning. Grim will fallback to runtime JIT.");
                return Ok(());
            }
        };

        let spec = device.hardware_spec();
        println!("[grim-tune] Detected GPU Architecture: {}", spec.gcn_arch);
        println!("[grim-tune] Wavefront Size: {}", spec.wavefront_size);
        println!("[grim-tune] Compute Units: {}", spec.cu_count);
        println!(
            "[grim-tune] LDS per CU: {} bytes",
            spec.max_shared_mem_per_block
        );
        println!(
            "[grim-tune] Max Threads per Block: {}",
            spec.max_threads_per_block
        );
        println!(
            "[grim-tune] Memory Bandwidth: {:.1} GB/s",
            spec.mem_bandwidth_gb_s
        );
        println!();
        println!(
            "[grim-tune] Running empirical FCP search and JIT compilation on canonical shapes..."
        );

        let tuned_count = grim_backend_rocm::run_install_tune(&device, &hsaco_dir)?;

        println!();
        println!(
            "[grim-tune] Successfully tuned and compiled {} workload shapes!",
            tuned_count
        );
        println!(
            "[grim-tune] Persisted autotune cache: {}/{}.json",
            hsaco_dir.display(),
            spec.gcn_arch
        );
        println!("=== Kernel Tuning Complete ===");
        Ok(())
    }

    #[cfg(not(feature = "rocm"))]
    {
        println!("[grim-tune] ROCm feature is not enabled in this build. Skipping GPU JIT tuning.");
        Ok(())
    }
}

fn dirs_next() -> Option<PathBuf> {
    std::env::var("HOME")
        .ok()
        .map(|h| PathBuf::from(h).join(".cache").join("grim").join("hsaco"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dirs_next_returns_path_or_none() {
        let _ = dirs_next();
    }
}
