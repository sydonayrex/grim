//! Standalone MXFP4 / MXFP8 dequantization HIP kernels for ROCm.
//!
//! Decompresses MXFP4 (Jay) and MXFP8 (Magpie) format weights to full F32 values.
//! Each MXFP4 group of 32 elements shares a single FP8 exponent byte
//! and stores 4-bit codes. Each MXFP8 element stores an 8-bit FP8 code
//! and shares a per-group exponent.

/// HIP source for `grim_dequant_mxfp4` and `grim_dequant_mxfp8`.
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

    __device__ inline float mxfp4_to_float_hip(unsigned char code, unsigned char shared_exp) {
        int sign = (code >> 3) & 1;
        int exp = (code >> 1) & 3;
        int mant = code & 1;
        float base_val = 0.0f;
        if (exp == 0) {
            base_val = (float)mant * 0.5f;
        } else {
            base_val = (1.0f + (float)mant * 0.5f) * powf(2.0f, (float)exp - 1.0f);
        }
        if (sign) base_val = -base_val;
        float scale = powf(2.0f, (float)shared_exp - 127.0f);
        return base_val * scale;
    }

    /// Dequantize MXFP4 codes + shared exponents to F32.
    /// Layout: codes[] (2 4-bit values per byte) + exps[] (1 FP8 per 32 elements).
    __global__ void grim_dequant_mxfp4(
        const unsigned char* __restrict__ codes,
        const unsigned char* __restrict__ exps,
        float* __restrict__ out,
        int n_weights)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_weights) return;

        int group_idx = idx / 32;
        unsigned char shared_exp = exps[group_idx];
        int code_byte_idx = idx / 2;
        unsigned char packed_byte = codes[code_byte_idx];
        unsigned char code = (idx % 2 == 0) ? (packed_byte & 0x0F) : ((packed_byte >> 4) & 0x0F);

        out[idx] = mxfp4_to_float_hip(code, shared_exp);
    }

    /// Dequantize MXFP8 codes + shared exponents to F32.
    /// Layout: codes[] (1 FP8 per element) + exps[] (1 FP8 per 32 elements).
    __global__ void grim_dequant_mxfp8(
        const unsigned char* __restrict__ codes,
        const unsigned char* __restrict__ exps,
        float* __restrict__ out,
        int n_weights)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_weights) return;

        int group_idx = idx / 32;
        unsigned char shared_exp = exps[group_idx];
        unsigned char fp8_code = codes[idx];
        float fp8_val = fp8_e4m3_to_float_hip(fp8_code);
        float exp_scale = powf(2.0f, (float)shared_exp - 127.0f);

        out[idx] = fp8_val * exp_scale;
    }

}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mxfp_standalone_source_contains_entries() {
        assert!(KERNEL_SOURCE.contains("grim_dequant_mxfp4"));
        assert!(KERNEL_SOURCE.contains("grim_dequant_mxfp8"));
        assert!(KERNEL_SOURCE.contains("mxfp4_to_float_hip"));
    }
}
