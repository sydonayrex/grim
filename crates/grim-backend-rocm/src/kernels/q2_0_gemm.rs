//! Q2_0 fused-dequant **backward** GEMM. [see: `block_q2_0`, 18 B per 64]
//!
//! Separate from the Q2_0 *forward* GEMV in `dot_gemv.rs`
//! (`grim_dot4_q2_0_q80_gemv`) rather than appended to it, for two reasons.
//! First, the forward kernel pairs Q2_0 against **Q8_0 activations** — a dot
//! instruction problem — while the backward needs no activation quantization
//! at all: `dX = dY @ W^T` consumes an f32 `dY` and dequantizes the weight
//! inline. They share a block layout and nothing else. Second, that forward
//! kernel sits inside a wave32 preprocessor guard; this one is scalar integer
//! MAC and is correct on any wavefront size, so it must not inherit the guard.
//!
//! Not feature-gated, matching the forward: Q2_0 is a core GGUF type (tag 42),
//! not an optional quant tier. The `q2k`/`q3k` gates exist because those are
//! tier formats a build can legitimately omit; gating this one would make the
//! backward arm compile away while the forward stayed, which is the same
//! half-wired state this kernel exists to remove.

/// HIP source for `grim_fused_dequant_backward_gemm_q2_0`.
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

    /// Dequantize one Q2_0 element from an 18-byte block.
    /// Layout: fp16 `d` at @0..2, then 16 bytes = 64 two-bit codes @2..18,
    /// four per byte, low bits first. `in_blk` is the weight index within the
    /// 64-weight block (0..63).
    ///
    /// The codebook is `{-1, 0, +1, +2} * d`, i.e. `(q - 1) * d`, matching both
    /// `quantize_q2_0_block` (`t = v/d + 1`) and the forward kernel's
    /// `((byte >> shift) & 3) - 1`. GSQ-RCO (tag 81) shares this geometry with
    /// a `(q - 2)` codebook, so this offset is the whole difference between the
    /// two formats and getting it wrong is silent: same bytes, finite,
    /// plausible, off-by-one-level weights. That is why the forward kernel
    /// transcribes llama.cpp's `ggml_vec_dot_q2_0_q8_0_generic` verbatim
    /// rather than sharing a helper with the quant path.
    __device__ inline float dequant_q2_0_element(const unsigned char* block_ptr, int in_blk) {
        float d = fp16_to_float_device(((const unsigned short*)(block_ptr))[0]);
        int q_byte  = in_blk >> 2;          // 4 codes per byte
        int q_shift = (in_blk & 3) * 2;     // low bits first
        int q_code  = (block_ptr[2 + q_byte] >> q_shift) & 0x03;
        return d * (float)(q_code - 1);
    }

    /// dX[M, K] = dY[M, N] @ W[N, K]^T, dequantizing W from its packed
    /// Q2_0 blocks inline so the weight is never expanded to f32.
    ///
    /// One thread per (row, k) output element. The weight row stride is
    /// `(K / 64) * 18` bytes, so a caller must have K % 64 == 0; the launcher
    /// refuses otherwise rather than letting `K / 64` truncate and read a
    /// partial trailing block as if it were whole.
    ///
    /// The N-stride multiply is done in 64-bit: a 14.6 GB expert bank has
    /// `n * row_bytes` well past 2^31, and an `int` overflow there would wrap
    /// to a small positive offset and read another expert's weights.
    __global__ void grim_fused_dequant_backward_gemm_q2_0(
        const float* __restrict__ dY,
        const unsigned char* __restrict__ B_q20,
        float* __restrict__ dX,
        int M, int N, int K)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * K;
        if (idx >= total) return;

        const int row = (int)(idx / K);
        const int k_idx = (int)(idx % K);

        const int blocks_per_row = K / 64;
        const int row_bytes = blocks_per_row * 18;

        const int blk   = k_idx / 64;
        const int in_blk = k_idx % 64;

        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            float dy_val = dY[(long long)row * N + n];
            const unsigned char* block_ptr =
                B_q20 + (long long)n * row_bytes + blk * 18;
            acc += dy_val * dequant_q2_0_element(block_ptr, in_blk);
        }

        dX[(long long)row * K + k_idx] = acc;
    }

}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// The kernel name is the dispatch key: `launch_compute_kernel` resolves it
    /// by string, and `launch_fused_deq_backward_gemm_simple` strips
    /// `grim_fused_dequant_backward_gemm_` off it to probe the tiled table.
    /// Renaming the kernel without renaming the launcher's string would fail at
    /// module lookup, not at compile time.
    #[test]
    fn q2_0_backward_kernel_is_named_for_dispatch() {
        assert!(KERNEL_SOURCE.contains("grim_fused_dequant_backward_gemm_q2_0"));
        assert!(KERNEL_SOURCE.contains("dequant_q2_0_element"));
    }

    /// The codebook offset is the one thing that distinguishes Q2_0 from
    /// GSQ-RCO, and it is silent when wrong. Pin it in the source text so a
    /// well-meaning "simplification" to `(q - 2)` fails here.
    #[test]
    fn q2_0_element_uses_the_q_minus_one_codebook() {
        assert!(
            KERNEL_SOURCE.contains("(float)(q_code - 1)"),
            "Q2_0 is (q-1)*d; GSQ-RCO's (q-2)*d would be silently wrong here"
        );
        assert!(
            !KERNEL_SOURCE.contains("q_code - 2"),
            "no (q-2) codebook belongs in the Q2_0 reader"
        );
    }

    /// The 18-byte geometry is load-bearing for the row stride and for
    /// `q_byte = in_blk >> 2`.
    #[test]
    fn q2_0_block_geometry_is_18_bytes_per_64() {
        assert!(KERNEL_SOURCE.contains("blocks_per_row * 18"));
        assert!(KERNEL_SOURCE.contains("blk * 18"));
        assert!(KERNEL_SOURCE.contains("in_blk >> 2"));
    }
}
