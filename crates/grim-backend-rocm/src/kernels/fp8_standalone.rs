//! Standalone FP8 dequantization HIP kernel for ROCm.

/// HIP source for `grim_dequant_fp8`.
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

    /// Dequantize FP8 bytes to F32. Each weight is 1 byte.
    __global__ void grim_dequant_fp8(
        const unsigned char* __restrict__ packed,
        float* __restrict__ out,
        int n_weights)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_weights) return;

        out[idx] = fp8_e4m3_to_float_hip(packed[idx]);
    }

}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_standalone_source_contains_entry() {
        assert!(KERNEL_SOURCE.contains("grim_dequant_fp8"));
        assert!(KERNEL_SOURCE.contains("fp8_e4m3_to_float_hip"));
    }
}
