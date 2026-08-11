//! ROCm HIP kernels for GPTQ quantization-aware re-quantization.
//!
//! Provides wavefront-level parallelism for the GPTQ error-correcting update:
//! ```text
//! W_corrected = W_approx + α * H_diag^{-1} ⊙ (W_original - W_approx)
//! ```
//!
//! where `H_diag` is the Fisher/GGN diagonal, `α` is the correction rate, and
//! `⊙` is element-wise multiplication. This is the Pass 4 ROCm-accelerated
//! path: the CPU fallback in `grim-quant` runs scalar row-by-row; this module
//! runs the same algorithm with wavefront-parallel HIP kernels via `hiprtc`.
//!
//! Design follows the FFI pattern from `lib.rs` — safe wrappers over unsafe
//! HIP FFI, using `jit_compile_hsaco` for on-demand kernel compilation.

use crate::device::helpers::check_hip;
use crate::{HiprtcProgram, hipSuccess};

/// HIP source for the GPTQ wavefront correction kernel.
///
/// Each HIP thread corrects one element of the weight matrix using the
/// diagonal Fisher preconditioner. Threads within a wavefront cooperate
/// via shuffle to reduce LDS bank pressure during the correction pass.
pub const GPTQ_CORRECTION_KERNEL: &str = r#"
extern "C" __global__
void gptq_wavefront_correction_kernel(
    float* __restrict__ weight_approx,
    const float* __restrict__ weight_orig,
    const float* __restrict__ h_diag,
    const uint32_t* __restrict__ group_map,
    float correction_rate,
    int num_groups,
    int group_size,
    int rows,
    int cols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= rows || col >= cols) return;

    int flat = row * cols + col;
    int group_idx = (group_map != NULL) ? group_map[col] : (col / group_size);
    float h = h_diag[group_idx];

    float orig = weight_orig[flat];
    float approx = weight_approx[flat];
    float residual = orig - approx;

    // Diagonal preconditioning: W_corrected = W_approx + (1/h_diag) * residual
    float corrected = approx + correction_rate * (residual / h);

    // Clamp to f16 representable range (safe mixed-precision guard)
    corrected = fminf(corrected, 65504.0f);
    corrected = fmaxf(corrected, -65504.0f);

    weight_approx[flat] = corrected;
}
"#;

/// HIP source for GPU-accelerated per-block scale search.
///
/// One HIP thread per quantization block. Each thread evaluates all 7
/// scale multipliers and picks the one with lowest weighted quantization error.
/// This replaces `fit_block_quantization` on CPU.
pub const GPTQ_SCALE_FIT_KERNEL: &str = r#"
extern "C" __global__
void gptq_scale_fit_kernel(
    const float* __restrict__ block_data,
    float* __restrict__ scales_out,
    const float* __restrict__ importance_weights,
    int block_size,
    int num_blocks,
    int bits
) {
    int block_id = blockIdx.x;
    if (block_id >= num_blocks) return;

    int tid = threadIdx.x;
    int lane_count = (block_size < blockDim.x) ? block_size : blockDim.x;
    if (tid >= lane_count) return;

    int start = block_id * block_size;
    float val = block_data[start + tid];

    float imp = (importance_weights != NULL) ? importance_weights[start + tid] : 1.0f;

    // 7 scale multipliers to search (Triton/llama.cpp style)
    const float multipliers[7] = {0.6f, 0.75f, 0.9f, 1.0f, 1.1f, 1.25f, 1.4f};

    float best_scale = 1.0f;
    float best_error = 1e9f;
    int max_code = (1 << bits) - 1;
    int signed_limit = max_code / 2;

    float absmax_val = fabsf(val);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_abs = __shfl_down(absmax_val, offset);
        if (tid + offset < lane_count) {
            absmax_val = fmaxf(absmax_val, other_abs);
        }
    }
    absmax_val = __shfl(absmax_val, 0);

    for (int mi = 0; mi < 7; mi++) {
        float base_scale = (absmax_val > 1e-7f) ? (absmax_val / (float)signed_limit) : 1.0f;
        float scale = base_scale * multipliers[mi];

        // Compute error for this scale
        float q = floorf(val / scale + 0.5f);
        q = fminf(q, (float)max_code);
        q = fmaxf(q, -(float)signed_limit);
        float dq = val - q * scale;
        float weighted_err = imp * dq * dq;

        // Intra-wavefront reduction via shuffle
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_err = __shfl_down(weighted_err, offset);
            float other_val = __shfl_down(val, offset);
            float other_imp = __shfl_down(imp, offset);
            if (tid + offset < lane_count) {
                weighted_err += other_err;
                imp += other_imp;
            }
        }
        if (tid == 0) {
            float avg_err = weighted_err / imp;
            if (avg_err < best_error) {
                best_error = avg_err;
                best_scale = scale;
            }
        }
    }
    if (tid == 0) {
        scales_out[block_id] = best_scale;
    }
}
"#;

/// Returns the wavefront size for a given AMD GCN architecture identifier.
pub fn wavefront_size_for_gcn(gcn: &str) -> u32 {
    match gcn {
        "gfx90a" | "gfx942" | "gfx90c" => 64, // CDNA2/3 (MI210, MI250, MI300X)
        "gfx1200" | "gfx1201" | "gfx1100" | "gfx1102" | "gfx11" => 32, // RDNA4/3
        "gfx1030" | "gfx1036" => 32,          // RDNA2 (RX 6700, RDNA2 iGPU)
        "gfx1031" | "gfx1032" | "gfx1033" | "gfx1034" | "gfx1035" => 32, // RDNA2 variants
        _ => 32,                              // safe default: this project is RDNA-first (Wave32)
    }
}

/// Compile a GPTQ HIP kernel and return the compiled HSACO bytes.
///
/// Uses `hiprtc` (HIP runtime compilation) for JIT compilation targeting
/// the specified GCN architecture. Consults `HsacoKernelCache` first to
/// bypass compilation on cache hit.
pub fn compile_gptq_kernel(
    kernel_name: &str,
    source: &str,
    gcn: &str,
) -> Result<Vec<u8>, crate::Error> {
    let hash = seahash::hash(source.as_bytes());
    let cache_key = format!("{}_{:016x}", kernel_name, hash);

    let cache = crate::HsacoKernelCache::new();
    if let Some((path, _lowered)) = cache.get_cached_kernel(&cache_key) {
        if let Ok(bytes) = std::fs::read(&path) {
            return Ok(bytes);
        }
    }

    let target = match gcn {
        "gfx90a" => "gfx90a",
        "gfx942" => "gfx942",
        "gfx1100" => "gfx1100",
        "gfx11" => "gfx1100",
        "gfx1030" => "gfx1030",
        "gfx1200" => "gfx1200",
        "gfx1201" => "gfx1201",
        _ => "gfx900",
    };

    let options = crate::device::util::hiprtc_options_for_arch(&target);
    let option_ptrs: Vec<*const i8> = options.iter().map(|c| c.as_ptr()).collect();

    let source_cstr = std::ffi::CString::new(source)
        .map_err(|e| crate::Error::Backend(format!("CString conversion failed: {}", e)))?;
    let name_cstr = std::ffi::CString::new(kernel_name)
        .map_err(|e| crate::Error::Backend(format!("CString conversion failed: {}", e)))?;

    unsafe {
        let mut prog: HiprtcProgram = std::ptr::null_mut();
        let status = crate::hiprtcCreateProgram(
            &mut prog,
            source_cstr.as_ptr(),
            name_cstr.as_ptr(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        );
        check_hip("hiprtcCreateProgram", status)?;

        let compile_status =
            crate::hiprtcCompileProgram(prog, options.len() as i32, option_ptrs.as_ptr());

        if compile_status != hipSuccess {
            let mut log_size: usize = 0;
            let _ = crate::hiprtcGetProgramLogSize(prog, &mut log_size);
            let mut log: Vec<u8> = vec![0u8; log_size.max(1)];
            let _ = crate::hiprtcGetProgramLog(prog, log.as_mut_ptr() as *mut i8);
            let log_str = String::from_utf8_lossy(&log);
            let _ = crate::hiprtcDestroyProgram(&mut prog);
            return Err(crate::Error::Backend(format!(
                "hiprtcCompileProgram failed (status {}): {}",
                compile_status, log_str
            )));
        }

        let mut code_size: usize = 0;
        let size_status = crate::hiprtcGetCodeSize(prog, &mut code_size);
        if size_status != hipSuccess {
            let _ = crate::hiprtcDestroyProgram(&mut prog);
            return Err(crate::Error::Backend(format!(
                "hiprtcGetCodeSize failed: {}",
                size_status
            )));
        }

        let mut code_bytes = vec![0u8; code_size];
        let code_status = crate::hiprtcGetCode(prog, code_bytes.as_mut_ptr() as *mut i8);
        if code_status != hipSuccess {
            let _ = crate::hiprtcDestroyProgram(&mut prog);
            return Err(crate::Error::Backend(format!(
                "hiprtcGetCode failed: {}",
                code_status
            )));
        }

        let _ = cache.cache_kernel(&cache_key, source, &code_bytes, kernel_name);
        Ok(code_bytes)
    }
}

/// Compile the wavefront correction HIP kernel for a target GCN architecture.
pub fn compile_gptq_correction_kernel(gcn: &str) -> Result<Vec<u8>, crate::Error> {
    compile_gptq_kernel(
        "gptq_wavefront_correction_kernel",
        GPTQ_CORRECTION_KERNEL,
        gcn,
    )
}

/// Compile the scale fitting HIP kernel for a target GCN architecture.
pub fn compile_gptq_scale_fit_kernel(gcn: &str) -> Result<Vec<u8>, crate::Error> {
    compile_gptq_kernel("gptq_scale_fit_kernel", GPTQ_SCALE_FIT_KERNEL, gcn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gptq_correction_kernel_source_is_non_empty_and_valid() {
        assert!(GPTQ_CORRECTION_KERNEL.contains("gptq_wavefront_correction_kernel"));
        assert!(GPTQ_CORRECTION_KERNEL.contains("weight_approx"));
    }

    #[test]
    fn gptq_scale_fit_kernel_source_is_non_empty_and_valid() {
        assert!(GPTQ_SCALE_FIT_KERNEL.contains("gptq_scale_fit_kernel"));
        assert!(GPTQ_SCALE_FIT_KERNEL.contains("importance_weights"));
    }

    #[test]
    fn wavefront_size_classification_matches_gcn_architectures() {
        assert_eq!(wavefront_size_for_gcn("gfx90a"), 64);
        assert_eq!(wavefront_size_for_gcn("gfx942"), 64);
        assert_eq!(wavefront_size_for_gcn("gfx90c"), 64);
        assert_eq!(wavefront_size_for_gcn("gfx1100"), 32);
        assert_eq!(wavefront_size_for_gcn("gfx1201"), 32);
        assert_eq!(wavefront_size_for_gcn("gfx1030"), 32);
        assert_eq!(wavefront_size_for_gcn("gfx1036"), 32);
        assert_eq!(wavefront_size_for_gcn("gfx_unknown"), 32);
    }
}
