//! Standalone Q4_K dequantization HIP kernel for ROCm. [see: `block_q4_K`]

/// HIP source for `grim_dequant_q4k`.
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

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
