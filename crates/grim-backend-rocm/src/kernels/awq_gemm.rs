//! AWQ GroupInt fused dequant-GEMM HIP kernels.
//!
//! Consumes the length-prefixed three-segment packed layout documented on
//! [`grim_tensor::dtype::AwqStorageConfig`]:
//!
//! ```text
//! [u64 LE: qweight_len][qweight][u64 LE: qzeros_len][qzeros][u64 LE: scales_len][scales (f16)]
//! ```
//!
//! AWQ conventions:
//! - `qweight`: column-packed uint32 words `[K / values_per_word, N]`.
//! - `qzeros`: packed uint32 words `[K / group_size, N / values_per_word]`, with RAW stored zero points (no +1 offset).
//! - `scales`: f16 per-(group, output-column) half floats.
//! - `g_idx`: absent (AWQ uses sequential grouping `k / group_size`).

pub const AWQ_GEMM_KERNEL_SOURCE: &str = r#"
// ---- AWQ dequant helpers (device-only, unique symbol prefix) ----
static inline __device__ unsigned int grim_awq_read_u32(
    const unsigned char* __restrict__ base, long long word_idx)
{
    return *(const unsigned int*)(base + word_idx * 4);
}

static inline __device__ float grim_awq_f16_to_f32(unsigned short h) {
    unsigned int s = ((unsigned int)(h & 0x8000u)) << 16;
    unsigned int e = ((unsigned int)(h & 0x7C00u)) << 13;
    unsigned int m = ((unsigned int)(h & 0x03FFu)) << 13;
    if (e == 0x7C00u) {
        return __uint_as_float(s | 0x7F800000u | (m ? 0x00400000u : 0u));
    }
    if (e == 0) {
        if (m == 0) return __uint_as_float(s);
        float v = (float)m * (1.0f / 16777216.0f);
        return (h & 0x8000u) ? -v : v;
    }
    return __uint_as_float(s | e | m | 0x38000000u);
}

// Read AWQ code for 2, 4, or 8 bits. `in_idx` indexes K, `col` indexes N.
static inline __device__ unsigned int grim_awq_read_code(
    const unsigned char* qweight, int in_idx, int col, int N,
    int bits, int values_per_word)
{
    long long word_idx = (long long)(in_idx / values_per_word) * N + col;
    unsigned int word = grim_awq_read_u32(qweight, word_idx);
    return (word >> ((in_idx % values_per_word) * bits)) & ((1u << bits) - 1u);
}

// Read raw AWQ zero point (no +1 offset).
static inline __device__ float grim_awq_read_zero(
    const unsigned char* qzeros, int group, int col,
    int bits, int values_per_word, int zeros_words_per_row)
{
    long long word_idx = (long long)group * zeros_words_per_row + col / values_per_word;
    unsigned int word = grim_awq_read_u32(qzeros, word_idx);
    return (float)((word >> ((col % values_per_word) * bits)) & ((1u << bits) - 1u));
}

// Forward: C[M, N] = A[M, K] @ dequant(B)^T where B packs a [K, N] AWQ weight.
extern "C" __global__ void grim_awq_dequant_gemm(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    float* __restrict__ C,
    int M, int N, int K,
    int bits, int group_size, int values_per_word,
    long long qw_off, long long qz_off, long long sc_off)
{
    const unsigned long long idx =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long total = (unsigned long long)M * N;
    if (idx >= total) return;

    const int row = (int)(idx / N);
    const int col = (int)(idx % N);

    const unsigned char* qweight = B_packed + qw_off;
    const unsigned char* qzeros  = B_packed + qz_off;
    const unsigned char* scales  = B_packed + sc_off;

    const int zeros_words_per_row = (N + values_per_word - 1) / values_per_word;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        int group = k / group_size;

        unsigned int code = grim_awq_read_code(qweight, k, col, N, bits, values_per_word);
        float zero = grim_awq_read_zero(qzeros, group, col, bits, values_per_word, zeros_words_per_row);
        
        long long scale_idx = (long long)group * N + col;
        unsigned short scale_h = *(const unsigned short*)(scales + scale_idx * 2);
        float scale = grim_awq_f16_to_f32(scale_h);

        float w = ((float)code - zero) * scale;
        acc += A[(long long)row * K + k] * w;
    }
    C[(long long)row * N + col] = acc;
}

// Backward: dX[M, K] = dY[M, N] @ dequant(B), same packed B ([K, N] weight).
extern "C" __global__ void grim_awq_dequant_backward_gemm(
    const float* __restrict__ dY,
    const unsigned char* __restrict__ B_packed,
    float* __restrict__ dX,
    int M, int N, int K,
    int bits, int group_size, int values_per_word,
    long long qw_off, long long qz_off, long long sc_off)
{
    const unsigned long long idx =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long total = (unsigned long long)M * K;
    if (idx >= total) return;

    const int row = (int)(idx / K);
    const int k = (int)(idx % K);

    const unsigned char* qweight = B_packed + qw_off;
    const unsigned char* qzeros  = B_packed + qz_off;
    const unsigned char* scales  = B_packed + sc_off;

    const int zeros_words_per_row = (N + values_per_word - 1) / values_per_word;
    const int group = k / group_size;

    float acc = 0.0f;
    for (int j = 0; j < N; ++j) {
        unsigned int code = grim_awq_read_code(qweight, k, j, N, bits, values_per_word);
        float zero = grim_awq_read_zero(qzeros, group, j, bits, values_per_word, zeros_words_per_row);

        long long scale_idx = (long long)group * N + j;
        unsigned short scale_h = *(const unsigned short*)(scales + scale_idx * 2);
        float scale = grim_awq_f16_to_f32(scale_h);

        float w = ((float)code - zero) * scale;
        acc += dY[(long long)row * N + j] * w;
    }
    dX[(long long)row * K + k] = acc;
}
"#;
