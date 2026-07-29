//! Q8_0 dequantization HIP kernel for ROCm.
//!
//! Each Q8_0 block is 34 bytes on disk: 2-byte FP16 delta (scale) followed
//! by 32 signed int8 codes. After dequantization each block produces 32 F32
//! values = `delta * code`.
//!
//! Layout matches llama.cpp's `block_q8_0` exactly (see ggml-common.h):
//! ```c
//! typedef struct { ggml_half d; int8_t qs[QK8_0]; } block_q8_0;
//! ```

/// HIP source for `grim_dequant_q8_0` — flat Q8_0 → F32 dequant.
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

    #define QK8_0 32

    /// Dequantize one Q8_0 block (34 bytes: 2-byte f16 delta + 32x int8 codes)
    /// into 32 F32 values. Each output thread handles one block.
    __global__ void grim_dequant_q8_0(
        const unsigned char* __restrict__ packed,
        float* __restrict__ out,
        int n_blocks)
    {
        int b = blockIdx.x * blockDim.x + threadIdx.x;
        if (b >= n_blocks) return;

        const unsigned char* blk = packed + b * (QK8_0 + 2);
        unsigned short d_bits = *((const unsigned short*)blk);
        float d = fp16_to_float_device(d_bits);

        const signed char* qs = (const signed char*)(blk + 2);
        for (int j = 0; j < QK8_0; ++j) {
            out[b * QK8_0 + j] = d * (float)qs[j];
        }
    }

}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_kernel_source_contains_entry() {
        assert!(KERNEL_SOURCE.contains("grim_dequant_q8_0"));
        assert!(KERNEL_SOURCE.contains("QK8_0"));
    }
}
