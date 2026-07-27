//! Standalone FP8 dequantization HIP kernel for ROCm.
//!
//! Converts FP8 E4M3 packed bytes to full F32 values.
//! Each weight is 1 byte (FP8 E4M3 format).
//! Useful for materializing FP8 weights to F32 on-device before standard GEMM.

/// HIP source for `grim_dequant_fp8`.
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

    __device__ inline float fp8_e4m3_to_float_hip(unsigned char val) {
        if (val == 0x7F) return 0.0f / 0.0f; // NaN
        if (val == 0xFF) return -0.0f / 0.0f;
        int sign = (val >> 7) & 1;
        int exp = (val >> 3) & 0x0F;
        int mant = val & 0x07;
        if (exp == 0) {
            float res = (float)mant / 8.0f * 0.000015258789f; // 2^-16
            return sign ? -res : res;
        }
        float res = (1.0f + (float)mant / 8.0f) * powf(2.0f, (float)exp - 7.0f);
        return sign ? -res : res;
    }

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
