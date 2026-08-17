//! Standalone IQ2/IQ3/IQ4 dequantization HIP kernels for ROCm.

/// HIP source for standalone IQ-family dequant kernels: [see: `grim_dequant_iq2xxs`, `grim_dequant_iq2xs`]
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

    // ─── IQ dequant helpers (mirrors iq_gemm.rs device functions) ──

    __device__ inline float dequant_iq2xxs_device(const unsigned char* blk, int in_sb) {
        float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
        const unsigned char* qs = blk + 2;
        const unsigned char* signs = blk + 34;
        int grid_idx = qs[in_sb / 8];
        float val = (float)((grid_idx + (in_sb % 8)) % 4) - 1.5f;
        float sign_val = ((signs[in_sb / 8] >> (in_sb % 8)) & 1) ? -1.0f : 1.0f;
        return d * val * sign_val;
    }

    __device__ inline float dequant_iq2xs_device(const unsigned char* blk, int in_sb) {
        float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
        const unsigned char* qs = blk + 2;
        const unsigned char* scales = blk + 34;
        const unsigned char* signs = blk + 42;
        int sb = in_sb / 16;
        float sc = ((float)((scales[sb / 2] >> ((sb % 2) * 4)) & 0x0F)) * 0.125f + 0.5f;
        float scale = d * sc;
        int grid_idx = qs[in_sb / 8];
        float val = (float)((grid_idx + (in_sb % 8)) % 4) - 1.5f;
        float sign_val = ((signs[in_sb / 8] >> (in_sb % 8)) & 1) ? -1.0f : 1.0f;
        return scale * val * sign_val;
    }

    __device__ inline float dequant_iq2s_device(const unsigned char* blk, int in_sb) {
        float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
        const unsigned char* qs = blk + 2;
        const unsigned char* scales = blk + 50;
        const unsigned char* signs = blk + 58;
        int sb = in_sb / 16;
        float sc = ((float)((scales[sb / 2] >> ((sb % 2) * 4)) & 0x0F)) * 0.125f + 0.5f;
        float scale = d * sc;
        int grid_idx = qs[in_sb / 8];
        float val = (float)((grid_idx + (in_sb % 8)) % 4) - 1.5f;
        float sign_val = ((signs[in_sb / 8] >> (in_sb % 8)) & 1) ? -1.0f : 1.0f;
        return scale * val * sign_val;
    }

    __device__ inline float dequant_iq3xxs_device(const unsigned char* blk, int in_sb) {
        float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
        const unsigned char* qs = blk + 2;
        const unsigned char* signs = blk + 66;
        int grid_idx = qs[in_sb / 8];
        int sub_idx = in_sb % 8;
        float base_val = (float)((grid_idx + sub_idx * 17) % 7) - 3.0f;
        int sign_byte_idx = (in_sb / 8);
        if (sign_byte_idx >= 30) sign_byte_idx = 29;
        float sign_val = ((signs[sign_byte_idx] >> (in_sb % 8)) & 1) ? -1.0f : 1.0f;
        return d * base_val * 0.25f * sign_val;
    }

    __device__ inline float dequant_iq3s_device(const unsigned char* blk, int in_sb) {
        float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
        const unsigned char* qs = blk + 2;
        const unsigned char* scales = blk + 66;
        const unsigned char* signs = blk + 78;
        int sb = in_sb / 32;
        float sc = ((float)(scales[sb * 12 / 8]) + 1.0f) * 0.125f;
        float scale = d * sc;
        float grid_val = (float)((qs[in_sb / 8] + in_sb) % 7) - 3.0f;
        float sign_val = ((signs[in_sb / 8] >> (in_sb % 8)) & 1) ? -1.0f : 1.0f;
        return scale * grid_val * sign_val;
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
        // Canonical signed codebook (ggml kvalues_iq4nl); CPU decoder takes abs()
        // then applies the q8 sign bit — replicate exactly.
        static const float kvalues_iq4nl[16] = {
            -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
            1.0f, 13.0f, 25.0f, 38.0f, 53.0f, 69.0f, 87.0f, 107.0f
        };
        float code_abs = kvalues_iq4nl[q_code];
        code_abs = code_abs < 0.0f ? -code_abs : code_abs;
        return d * group_scale * code_abs * sign_val;
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
    ///
    /// One 64-thread block per 256-element quant block; each thread decodes
    /// four consecutive elements with vectorized float4 stores (the previous
    /// one-thread-per-block form serialized 256 dependent dequants and wrote
    /// scalars).
    __global__ void __launch_bounds__(64)
    grim_dequant_iq2xxs(
        const unsigned char* __restrict__ packed,
        float* __restrict__ out,
        int n_blocks)
    {
        const int b = blockIdx.x;
        if (b >= n_blocks) return;
        const unsigned char* blk = packed + (size_t)b * 66;
        float* dst = out + (size_t)b * 256;
        const int base = threadIdx.x * 4;
        float4 v;
        v.x = dequant_iq2xxs_device(blk, base + 0);
        v.y = dequant_iq2xxs_device(blk, base + 1);
        v.z = dequant_iq2xxs_device(blk, base + 2);
        v.w = dequant_iq2xxs_device(blk, base + 3);
        *reinterpret_cast<float4*>(dst + base) = v;
    }

    /// Dequantize IQ2_XS packed bytes to F32.
    ///
    /// One 64-thread block per 256-element quant block; each thread decodes
    /// four consecutive elements with vectorized float4 stores (the previous
    /// one-thread-per-block form serialized 256 dependent dequants and wrote
    /// scalars).
    __global__ void __launch_bounds__(64)
    grim_dequant_iq2xs(
        const unsigned char* __restrict__ packed,
        float* __restrict__ out,
        int n_blocks)
    {
        const int b = blockIdx.x;
        if (b >= n_blocks) return;
        const unsigned char* blk = packed + (size_t)b * 74;
        float* dst = out + (size_t)b * 256;
        const int base = threadIdx.x * 4;
        float4 v;
        v.x = dequant_iq2xs_device(blk, base + 0);
        v.y = dequant_iq2xs_device(blk, base + 1);
        v.z = dequant_iq2xs_device(blk, base + 2);
        v.w = dequant_iq2xs_device(blk, base + 3);
        *reinterpret_cast<float4*>(dst + base) = v;
    }

    /// Dequantize IQ2_S packed bytes to F32.
    ///
    /// One 64-thread block per 256-element quant block; each thread decodes
    /// four consecutive elements with vectorized float4 stores (the previous
    /// one-thread-per-block form serialized 256 dependent dequants and wrote
    /// scalars).
    __global__ void __launch_bounds__(64)
    grim_dequant_iq2s(
        const unsigned char* __restrict__ packed,
        float* __restrict__ out,
        int n_blocks)
    {
        const int b = blockIdx.x;
        if (b >= n_blocks) return;
        const unsigned char* blk = packed + (size_t)b * 82;
        float* dst = out + (size_t)b * 256;
        const int base = threadIdx.x * 4;
        float4 v;
        v.x = dequant_iq2s_device(blk, base + 0);
        v.y = dequant_iq2s_device(blk, base + 1);
        v.z = dequant_iq2s_device(blk, base + 2);
        v.w = dequant_iq2s_device(blk, base + 3);
        *reinterpret_cast<float4*>(dst + base) = v;
    }

    /// Dequantize IQ3_XXS packed bytes to F32.
    ///
    /// One 64-thread block per 256-element quant block; each thread decodes
    /// four consecutive elements with vectorized float4 stores (the previous
    /// one-thread-per-block form serialized 256 dependent dequants and wrote
    /// scalars).
    __global__ void __launch_bounds__(64)
    grim_dequant_iq3xxs(
        const unsigned char* __restrict__ packed,
        float* __restrict__ out,
        int n_blocks)
    {
        const int b = blockIdx.x;
        if (b >= n_blocks) return;
        const unsigned char* blk = packed + (size_t)b * 96;
        float* dst = out + (size_t)b * 256;
        const int base = threadIdx.x * 4;
        float4 v;
        v.x = dequant_iq3xxs_device(blk, base + 0);
        v.y = dequant_iq3xxs_device(blk, base + 1);
        v.z = dequant_iq3xxs_device(blk, base + 2);
        v.w = dequant_iq3xxs_device(blk, base + 3);
        *reinterpret_cast<float4*>(dst + base) = v;
    }

    /// Dequantize IQ3_S packed bytes to F32.
    ///
    /// One 64-thread block per 256-element quant block; each thread decodes
    /// four consecutive elements with vectorized float4 stores (the previous
    /// one-thread-per-block form serialized 256 dependent dequants and wrote
    /// scalars).
    __global__ void __launch_bounds__(64)
    grim_dequant_iq3s(
        const unsigned char* __restrict__ packed,
        float* __restrict__ out,
        int n_blocks)
    {
        const int b = blockIdx.x;
        if (b >= n_blocks) return;
        const unsigned char* blk = packed + (size_t)b * 110;
        float* dst = out + (size_t)b * 256;
        const int base = threadIdx.x * 4;
        float4 v;
        v.x = dequant_iq3s_device(blk, base + 0);
        v.y = dequant_iq3s_device(blk, base + 1);
        v.z = dequant_iq3s_device(blk, base + 2);
        v.w = dequant_iq3s_device(blk, base + 3);
        *reinterpret_cast<float4*>(dst + base) = v;
    }

    /// Dequantize IQ4_NL packed bytes to F32.
    ///
    /// One 64-thread block per 256-element quant block; each thread decodes
    /// four consecutive elements with vectorized float4 stores (the previous
    /// one-thread-per-block form serialized 256 dependent dequants and wrote
    /// scalars).
    __global__ void __launch_bounds__(64)
    grim_dequant_iq4nl(
        const unsigned char* __restrict__ packed,
        float* __restrict__ out,
        int n_blocks)
    {
        const int b = blockIdx.x;
        if (b >= n_blocks) return;
        const unsigned char* blk = packed + (size_t)b * 170;
        float* dst = out + (size_t)b * 256;
        const int base = threadIdx.x * 4;
        float4 v;
        v.x = dequant_iq4nl_device(blk, base + 0);
        v.y = dequant_iq4nl_device(blk, base + 1);
        v.z = dequant_iq4nl_device(blk, base + 2);
        v.w = dequant_iq4nl_device(blk, base + 3);
        *reinterpret_cast<float4*>(dst + base) = v;
    }

    /// Dequantize IQ4_XS packed bytes to F32.
    ///
    /// One 64-thread block per 256-element quant block; each thread decodes
    /// four consecutive elements with vectorized float4 stores (the previous
    /// one-thread-per-block form serialized 256 dependent dequants and wrote
    /// scalars).
    __global__ void __launch_bounds__(64)
    grim_dequant_iq4xs(
        const unsigned char* __restrict__ packed,
        float* __restrict__ out,
        int n_blocks)
    {
        const int b = blockIdx.x;
        if (b >= n_blocks) return;
        const unsigned char* blk = packed + (size_t)b * 136;
        float* dst = out + (size_t)b * 256;
        const int base = threadIdx.x * 4;
        float4 v;
        v.x = dequant_iq4xs_device(blk, base + 0);
        v.y = dequant_iq4xs_device(blk, base + 1);
        v.z = dequant_iq4xs_device(blk, base + 2);
        v.w = dequant_iq4xs_device(blk, base + 3);
        *reinterpret_cast<float4*>(dst + base) = v;
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
