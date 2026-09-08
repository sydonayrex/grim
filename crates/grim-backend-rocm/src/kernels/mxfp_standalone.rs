//! Standalone MXFP4 / MXFP8 dequantization HIP kernels for ROCm.

/// HIP source for `grim_dequant_mxfp4` and `grim_dequant_mxfp8`.
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

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
        float exp_scale = exp2f((float)shared_exp - 127.0f);

        out[idx] = fp8_val * exp_scale;
    }

    /// Dequantize NVFP4 codes + shared exponents to F32.
    /// Interleaved layout: 9 bytes per 16 weights (1 E8M0 scale byte + 8 packed code bytes).
    __global__ void grim_dequant_nvfp4(
        const unsigned char* __restrict__ packed,
        float* __restrict__ out,
        int n_weights)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_weights) return;

        int sub_block_idx = idx / 16;
        int in_sub_block = idx % 16;
        int blk_offset = sub_block_idx * 9;
        unsigned char shared_exp = packed[blk_offset];
        unsigned char code_byte = packed[blk_offset + 1 + (in_sub_block / 2)];
        unsigned char code = (in_sub_block % 2 == 0) ? (code_byte & 0x0F) : ((code_byte >> 4) & 0x0F);

        out[idx] = mxfp4_to_float_hip(code, shared_exp);
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
        assert!(KERNEL_SOURCE.contains("grim_dequant_nvfp4"));
        assert!(KERNEL_SOURCE.contains("mxfp4_to_float_hip"));
    }
}
