//! GPTQ / EfficientQAT GroupInt fused dequant-GEMM CUDA kernels.
//!
//! Ported from grim-backend-rocm `kernels/gptq_gemm.rs`.
//!
//! Consumes the length-prefixed four-segment packed layout:
//!   [u64 LE: qweight_len][qweight][u64 LE: qzeros_len][qzeros]
//!   [u64 LE: scales_len][scales][u64 LE: g_idx_len][g_idx]
//!
//! Dequant: asymmetric (code - (zero+1)) * scale, matching grim_quant::dequant_gptq_group_int.
//! Supports 2-bit, 3-bit, 4-bit, 8-bit codes and optional act-order g_idx permutation.
//! Inference-only: forward GEMM only (no backward weight-gradient kernel).

pub const GPTQ_GEMM_SOURCE: &str = r#"
extern "C" {

// ---------------------------------------------------------------------------
// Helpers — GPTQ packed-weight read utilities
// ---------------------------------------------------------------------------

__device__ __forceinline__ unsigned int grim_gptq_read_u32(
    const unsigned char* __restrict__ base, long long word_idx)
{
    return *(const unsigned int*)(base + word_idx * 4);
}

// 3-bit code packing: values 0-31 span three consecutive u32 words.
__device__ __forceinline__ unsigned int grim_gptq_read_code3(
    const unsigned char* qweight, long long base_word, int lane)
{
    unsigned int w0 = grim_gptq_read_u32(qweight, base_word);
    unsigned int w1 = grim_gptq_read_u32(qweight, base_word + 1);
    unsigned int w2 = grim_gptq_read_u32(qweight, base_word + 2);
    int bit = lane * 3;
    if (bit < 32)
        return (unsigned int)(((unsigned long long)w0 | ((unsigned long long)w1 << 32)) >> bit) & 0x7u;
    return (w2 >> (bit - 32)) & 0x7u;
}

__device__ __forceinline__ unsigned int grim_gptq_read_code(
    const unsigned char* qweight, int in_idx, int col, int N,
    int bits, int values_per_word)
{
    if (bits == 3) {
        long long base = (long long)(in_idx / 32) * 3 * N + col;
        return grim_gptq_read_code3(qweight, base, in_idx % 32);
    }
    long long word_idx = (long long)(in_idx / values_per_word) * N + col;
    unsigned int word = grim_gptq_read_u32(qweight, word_idx);
    return (word >> ((in_idx % values_per_word) * bits)) & ((1u << bits) - 1u);
}

__device__ __forceinline__ float grim_gptq_read_zero(
    const unsigned char* qzeros, int group, int col, int N,
    int bits, int values_per_word, int zeros_words_per_row)
{
    if (bits == 3) {
        long long base = (long long)group * (3 * ((N + 31) / 32)) + col / 32 * 3;
        return (float)(grim_gptq_read_code3(qzeros, base, col % 32) + 1u);
    }
    long long word_idx = (long long)group * zeros_words_per_row + col / values_per_word;
    unsigned int word = grim_gptq_read_u32(qzeros, word_idx);
    return (float)((word >> ((col % values_per_word) * bits)) & ((1u << bits) - 1u)) + 1.0f;
}

// ---------------------------------------------------------------------------
// grim_gptq_dequant_gemm — forward inference GEMM for GPTQ/EfficientQAT.
//
// C[M,N] = A[M,K] @ dequant(B)^T  where B packs a [K,N] weight.
// Grid: ceil(M*N / 256) blocks, 256 threads. One thread per output cell.
//
// Contract:
//   B_packed layout: [qweight | qzeros | scales | g_idx] (offsets passed as args)
//   scale format: fp32 per (group, out_col) stored contiguously.
// ---------------------------------------------------------------------------
__global__ void grim_gptq_dequant_gemm(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    float* __restrict__ C,
    int M, int N, int K,
    int bits, int group_size, int values_per_word,
    int has_g_idx,
    long long qw_off, long long qz_off, long long sc_off, long long gi_off)
{
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (unsigned long long)M * N) return;

    const int row = (int)(idx / N);
    const int col = (int)(idx % N);

    const unsigned char* qweight = B_packed + qw_off;
    const unsigned char* qzeros  = B_packed + qz_off;
    const unsigned char* scales  = B_packed + sc_off;
    const unsigned char* g_idx   = B_packed + gi_off;

    const int zpw = (bits == 3) ? 0 : (N + values_per_word - 1) / values_per_word;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        int group = has_g_idx ? (int)grim_gptq_read_u32(g_idx, k) : k / group_size;
        unsigned int code = grim_gptq_read_code(qweight, k, col, N, bits, values_per_word);
        float zero  = grim_gptq_read_zero(qzeros, group, col, N, bits, values_per_word, zpw);
        float scale = *(const float*)(scales + ((long long)group * N + col) * 4);
        acc += A[(long long)row * K + k] * ((float)code - zero) * scale;
    }
    C[(long long)row * N + col] = acc;
}

} // extern "C"
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_contains_gptq_gemm_entry() {
        assert!(GPTQ_GEMM_SOURCE.contains("grim_gptq_dequant_gemm"));
        assert!(GPTQ_GEMM_SOURCE.contains("grim_gptq_read_code3"));
        assert!(GPTQ_GEMM_SOURCE.contains("values_per_word"));
    }
}
