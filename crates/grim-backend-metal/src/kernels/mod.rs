//! Metal Shader kernel modules and dynamic assembly loader.

pub const MATH_MSL: &str = include_str!("math.msl");

/// Unified MSL shader bundle combining all modular kernels.
pub fn load_unified_msl() -> &'static str {
    include_str!("../kernels.msl")
}

pub const GEMM_MSL: &str = include_str!("gemm.msl");
pub const ATTENTION_MSL: &str = include_str!("attention.msl");

pub const QUANTIZATION_MSL: &str = include_str!("quantization.msl");

pub const OPTIMIZER_MSL: &str = include_str!("optimizer.msl");
