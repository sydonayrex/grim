//! Metal Shader kernel modules and dynamic assembly loader.

pub const MATH_MSL: &str = include_str!("math.msl");

/// Unified MSL shader bundle combining all modular kernels.
pub fn load_unified_msl() -> &'static str {
    include_str!("../kernels.msl")
}
