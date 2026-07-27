//! Standalone Q4_K dequantization HIP kernel for ROCm.
//!
//! Dequantizes llama.cpp `block_q4_K` super-blocks (256 weights,
//! 6-bit scales, 4-bit codes) to full F32.  Useful for materializing
//! Q4_K weights to full precision on-device before standard GEMM.
//!
//! Layout matches llama.cpp block_q4_K (144 bytes per 256 weights):
//! - d (f16): super-block scale (2 bytes)
//! - dmin (f16): super-block minimum (2 bytes)
//! - scales (12 bytes): packed 6-bit sc and m values for 8 sub-blocks
//! - qs (128 bytes): packed 4-bit codes for 256 weights

/// HIP source for `grim_dequant_q4k`.
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

    __device__ inline float fp16_to_float_device(unsigned short h) {
        unsigned int sign = (h >> 15) & 1;
        unsigned int exp  = (h >> 10) & 0x1f;
        unsigned int mant = h & 0x3ff;
        if (exp == 0) {
            if (mant == 0) return sign ? -0.0f : 0.0f;
            float res = (float)mant / 1024.0f * 0.00006103515625f;
            return sign ? -res : res;
        } else if (exp == 31) {
            return sign ? -1.0f/0.0f : 1.0f/0.0f;
        }
        float res = (1.0f + (float)mant / 1024.0f) * powf(2.0f, (float)exp - 15.0f);
        return sign ? -res : res;
    }

    /// Dequantize one Q4_K element from a 144-byte super-block.
    __device__ inline float dequant_q4k_element(const unsigned char* block_ptr, int in_sb) {
        const unsigned short* h_ptr = (const unsigned short*)block_ptr;
        float d = fp16_to_float_device(h_ptr[0]);
        float dmin = fp16_to_float_device(h_ptr[1]);

        const unsigned char* scales = block_ptr + 4;
        const unsigned char* qs = block_ptr + 16;

        int is = in_sb / 32;
        unsigned char sc, m;
        if (is < 4) {
            sc = scales[is] & 63;
            m  = scales[is + 4] & 63;
        } else {
            sc = (scales[is + 4] & 0xF) | ((scales[is - 4] >> 6) << 4);
            m  = (scales[is + 4] >> 4)  | ((scales[is] >> 6) << 4);
        }

        int q_idx = in_sb / 2;
        unsigned char packed = qs[q_idx];
        unsigned char q_code = (in_sb % 2 == 0) ? (packed & 0x0F) : (packed >> 4);

        return d * (float)sc * (float)q_code - dmin * (float)m;
    }

    /// Dequantize packed Q4_K bytes to F32. Each thread handles one group
    /// of 32 weights (one sub-block).
    __global__ void grim_dequant_q4k(
        const unsigned char* __restrict__ packed,
        float* __restrict__ out,
        int n_blocks)
    {
        int b = blockIdx.x * blockDim.x + threadIdx.x;
        if (b >= n_blocks) return;

        const unsigned char* blk = packed + b * 144;

        for (int i = 0; i < 32; ++i) {
            out[b * 32 + i] = dequant_q4k_element(blk, i);
        }
    }

}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4k_dequant_source_contains_entry() {
        assert!(KERNEL_SOURCE.contains("grim_dequant_q4k"));
        assert!(KERNEL_SOURCE.contains("dequant_q4k_element"));
    }
}
