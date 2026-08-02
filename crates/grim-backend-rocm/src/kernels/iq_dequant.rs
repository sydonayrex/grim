//! Standalone IQ2/IQ3/IQ4 dequantization HIP kernels for ROCm.

/// HIP source for standalone IQ-family dequant kernels: [see: `grim_dequant_iq2xxs`, `grim_dequant_iq2xs`]
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

    // ─── IQ dequant helpers (mirrors iq_gemm.rs device functions) ──

    __device__ inline float dequant_iq2xxs_device(const unsigned char* blk, int in_sb) {
        float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
        const unsigned char* signs = blk + 2;
        const unsigned char* qs = blk + 32;
        int group = in_sb / 8;
        int in_group = in_sb % 8;
        unsigned char sign_byte = signs[group];
        float sign_val = ((sign_byte >> (in_group % 8)) & 1) ? -1.0f : 1.0f;
        unsigned char idx = qs[group];
        float scale = (float)idx;
        return d * scale * sign_val;
    }

    __device__ inline float dequant_iq2xs_device(const unsigned char* blk, int in_sb) {
        float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
        const unsigned char* scales = blk + 2;
        const unsigned char* signs = blk + 10;
        const unsigned char* qs = blk + 42;
        int group = in_sb / 8;
        int in_group = in_sb % 8;
        float sc = (float)(scales[group] & 0x3F);
        float sign_val = ((signs[group] >> (in_group % 8)) & 1) ? -1.0f : 1.0f;
        unsigned char idx = qs[group];
        float scale = (float)idx;
        return d * sc * scale * sign_val;
    }

    __device__ inline float dequant_iq2s_device(const unsigned char* blk, int in_sb) {
        float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
        const unsigned char* scales = blk + 2;
        const unsigned char* signs = blk + 10;
        const unsigned char* qs = blk + 42;
        int group = in_sb / 8;
        int in_group = in_sb % 8;
        float sc = (float)(scales[group] & 0x3F);
        float sign_val = ((signs[group] >> (in_group % 8)) & 1) ? -1.0f : 1.0f;
        unsigned char idx = qs[group];
        float scale = (float)(idx & 0x3F);
        return d * sc * scale * sign_val;
    }

    __device__ inline float dequant_iq3xxs_device(const unsigned char* blk, int in_sb) {
        float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
        const unsigned char* signs = blk + 2;
        const unsigned char* qs = blk + 32;
        int group = in_sb / 8;
        int in_group = in_sb % 8;
        unsigned char sign_byte = signs[group];
        float sign_val = ((sign_byte >> (in_group % 8)) & 1) ? -1.0f : 1.0f;
        unsigned char idx = qs[group];
        float scale = (float)(idx & 7);
        return d * scale * sign_val;
    }

    __device__ inline float dequant_iq3s_device(const unsigned char* blk, int in_sb) {
        float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
        const unsigned char* scales = blk + 2;
        const unsigned char* signs = blk + 14;
        const unsigned char* qs = blk + 46;
        int group = in_sb / 8;
        int in_group = in_sb % 8;
        float sc = (float)(scales[group] & 0x3F);
        float sign_val = ((signs[group] >> (in_group % 8)) & 1) ? -1.0f : 1.0f;
        unsigned char idx = qs[group];
        float scale = (float)(idx & 7);
        return d * sc * scale * sign_val;
    }

    __device__ inline float dequant_iq4nl_device(const unsigned char* blk, int in_sb) {
        float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
        const unsigned char* signs = blk + 2;
        const unsigned char* qs = blk + 34;
        const unsigned char* sc = blk + 162;
        int group = in_sb / 16;
        float group_scale = (float)(sc[group] & 3);
        group_scale = 1.0f + 0.125f * group_scale;
        int sign_byte_idx = (in_sb / 8);
        int sign_bit = in_sb % 8;
        float sign_val = ((signs[sign_byte_idx] >> sign_bit) & 1) ? -1.0f : 1.0f;
        int q_byte = in_sb / 2;
        unsigned char q_code = (in_sb % 2 == 0) ? (qs[q_byte] & 0x0F) : ((qs[q_byte] >> 4) & 0x0F);
        return d * group_scale * (float)q_code * sign_val;
    }

    __device__ inline float dequant_iq4xs_device(const unsigned char* blk, int in_sb) {
        float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
        const unsigned char* sc = blk + 2;
        const unsigned char* qs = blk + 8;
        int group = in_sb / 32;
        int sc_byte_idx = (group * 6) / 8;
        int sc_bit_offset = (group * 6) % 8;
        unsigned int sc_val = 0;
        sc_val = sc[sc_byte_idx] >> sc_bit_offset;
        if (sc_bit_offset > 2) {
            sc_val |= (unsigned int)sc[sc_byte_idx + 1] << (8 - sc_bit_offset);
        }
        sc_val &= 0x3F;
        int q_byte = in_sb / 2;
        unsigned char q_code = (in_sb % 2 == 0) ? (qs[q_byte] & 0x0F) : ((qs[q_byte] >> 4) & 0x0F);
        return d * (float)sc_val * (float)q_code;
    }

    // ─── Standalone global kernels ──────────────────────────────────

    /// Dequantize IQ2_XXS packed bytes to F32.
    __global__ void grim_dequant_iq2xxs(
        const unsigned char* __restrict__ packed,
        float* __restrict__ out,
        int n_blocks)
    {
        int b = blockIdx.x * blockDim.x + threadIdx.x;
        if (b >= n_blocks) return;
        const unsigned char* blk = packed + b * 66;
        for (int i = 0; i < 256; ++i) {
            out[b * 256 + i] = dequant_iq2xxs_device(blk, i);
        }
    }

    /// Dequantize IQ2_XS packed bytes to F32.
    __global__ void grim_dequant_iq2xs(
        const unsigned char* __restrict__ packed,
        float* __restrict__ out,
        int n_blocks)
    {
        int b = blockIdx.x * blockDim.x + threadIdx.x;
        if (b >= n_blocks) return;
        const unsigned char* blk = packed + b * 74;
        for (int i = 0; i < 256; ++i) {
            out[b * 256 + i] = dequant_iq2xs_device(blk, i);
        }
    }

    /// Dequantize IQ2_S packed bytes to F32.
    __global__ void grim_dequant_iq2s(
        const unsigned char* __restrict__ packed,
        float* __restrict__ out,
        int n_blocks)
    {
        int b = blockIdx.x * blockDim.x + threadIdx.x;
        if (b >= n_blocks) return;
        const unsigned char* blk = packed + b * 82;
        for (int i = 0; i < 256; ++i) {
            out[b * 256 + i] = dequant_iq2s_device(blk, i);
        }
    }

    /// Dequantize IQ3_XXS packed bytes to F32.
    __global__ void grim_dequant_iq3xxs(
        const unsigned char* __restrict__ packed,
        float* __restrict__ out,
        int n_blocks)
    {
        int b = blockIdx.x * blockDim.x + threadIdx.x;
        if (b >= n_blocks) return;
        const unsigned char* blk = packed + b * 96;
        for (int i = 0; i < 256; ++i) {
            out[b * 256 + i] = dequant_iq3xxs_device(blk, i);
        }
    }

    /// Dequantize IQ3_S packed bytes to F32.
    __global__ void grim_dequant_iq3s(
        const unsigned char* __restrict__ packed,
        float* __restrict__ out,
        int n_blocks)
    {
        int b = blockIdx.x * blockDim.x + threadIdx.x;
        if (b >= n_blocks) return;
        const unsigned char* blk = packed + b * 110;
        for (int i = 0; i < 256; ++i) {
            out[b * 256 + i] = dequant_iq3s_device(blk, i);
        }
    }

    /// Dequantize IQ4_NL packed bytes to F32.
    __global__ void grim_dequant_iq4nl(
        const unsigned char* __restrict__ packed,
        float* __restrict__ out,
        int n_blocks)
    {
        int b = blockIdx.x * blockDim.x + threadIdx.x;
        if (b >= n_blocks) return;
        const unsigned char* blk = packed + b * 170;
        for (int i = 0; i < 256; ++i) {
            out[b * 256 + i] = dequant_iq4nl_device(blk, i);
        }
    }

    /// Dequantize IQ4_XS packed bytes to F32.
    __global__ void grim_dequant_iq4xs(
        const unsigned char* __restrict__ packed,
        float* __restrict__ out,
        int n_blocks)
    {
        int b = blockIdx.x * blockDim.x + threadIdx.x;
        if (b >= n_blocks) return;
        const unsigned char* blk = packed + b * 136;
        for (int i = 0; i < 256; ++i) {
            out[b * 256 + i] = dequant_iq4xs_device(blk, i);
        }
    }

}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iq_dequant_source_contains_all_kernels() {
        assert!(KERNEL_SOURCE.contains("grim_dequant_iq2xxs"));
        assert!(KERNEL_SOURCE.contains("grim_dequant_iq2xs"));
        assert!(KERNEL_SOURCE.contains("grim_dequant_iq2s"));
        assert!(KERNEL_SOURCE.contains("grim_dequant_iq3xxs"));
        assert!(KERNEL_SOURCE.contains("grim_dequant_iq3s"));
        assert!(KERNEL_SOURCE.contains("grim_dequant_iq4nl"));
        assert!(KERNEL_SOURCE.contains("grim_dequant_iq4xs"));
    }
}
