//! ROCm HIP kernel sources for the compressed tensor types.
//!
//! - W8A8MXFP8 weights: reuse the existing `mxfp_standalone` MxFp8 kernel
//!   (`grim_dequant_mxfp8`) — codes/E8M0 exps framing is identical.
//! - WNA16 weights: 256-weight blocks, per-block f16 scale, per-tensor f32
//!   scale; N-bit codes packed MSB-first across bytes.
//! - EmbeddingWNA16Int: same N-bit decode but per-tensor f32 scale only
//!   (no per-block scales), row-major packed codes.

/// MSB-first N-bit decoder: from a byte stream, extract the N-bit code for
/// element `lane_in_block` (0-based) within its block. Codes are packed MSB
/// first, crossing byte boundaries as needed. `bytes_total` = ceil(256*n_bit/8)
/// for the WNA16 path or ceil(embedding_dim*n_bit/8) per row for the embedding
/// path.
#[inline]
#[allow(dead_code)]
fn decode_msb_nbit(code_bytes: &[u8], block_offset_bytes: usize, lane_in_block: usize, n_bit: u8) -> u32 {
    let n = n_bit as usize;
    let start_bit = lane_in_block * n;
    let mut code: u32 = 0;
    for b in 0..n {
        let pos = start_bit + b;
        let byte_idx = block_offset_bytes + pos / 8;
        let bit_in_byte = pos % 8; // 0 = the byte's MSB
        let bit = (code_bytes[byte_idx] >> (7 - bit_in_byte)) & 1;
        code = (code << 1) | bit as u32;
    }
    code
}


/// Simple variant used for embedding (row-major, one row at a time): same MSB-first
/// across the row's packed bytes.
#[inline]
#[allow(dead_code)]
fn decode_msb_nbit_row(code_bytes: &[u8], row_byte_offset: usize, col: usize, n_bit: u8) -> u32 {
    let n = n_bit as usize;
    let start_bit = col * n;
    let mut code: u32 = 0;
    for b in 0..n {
        let pos = start_bit + b;
        let byte_idx = row_byte_offset + pos / 8;
        let bit_in_byte = pos % 8; // 0 = the byte's MSB
        let bit = (code_bytes[byte_idx] >> (7 - bit_in_byte)) & 1;
        code = (code << 1) | bit as u32;
    }
    code
}


pub const WEIGHT_NA16_KERNEL: &str = r#"
// Device helpers must precede first use in a single TU (audit fix: the
// previous revision defined them AFTER grim_dequant_wna16 and used Rust
// f32::from_bits syntax, which HIP C cannot parse — every kernel in the
// aggregate JIT unit failed with "undeclared identifier").
static inline __device__ float grim_f16_to_f32_hip(unsigned short h)
{
    unsigned int s = ((unsigned int)(h & 0x8000u)) << 16;
    unsigned int e = ((unsigned int)(h & 0x7C00u)) << 13;
    unsigned int m = ((unsigned int)(h & 0x03FFu)) << 13;
    if (e == 0x7C00u) {
        // Inf/NaN: preserve sign, set mantissa LSBs for NaN payload.
        return __uint_as_float(s | 0x7F800000u | (m ? 0x00400000u : 0u));
    }
    if (e == 0) {
        if (m == 0) return __uint_as_float(s); // signed zero
        // Subnormal half → normalized f32: value = m * 2^-24, sign applied.
        float v = (float)m * (1.0f / 16777216.0f);
        return (h & 0x8000u) ? -v : v;
    }
    // e = exponent field already shifted into f32 position (E<<23); the
    // f16→f32 bias adjustment adds (127-15)=112 to the exponent field,
    // i.e. OR with 112<<23 = 0x38000000. The previous 0x38800000 constant
    // (113<<23) doubled every normal-range scale — caught by the WNA16
    // fused-GEMM golden gate (its host reference decodes correctly).
    return __uint_as_float(s | e | m | 0x38000000u);
}

// Audit rewrite: the previous loop computed a NEGATIVE shift on the first
// byte (8 - 8 - take), which is UB in C/CUDA and silently extracted the
// wrong bits. MSB-first means: stream bit p lives at byte p/8, bit 7-(p%8);
// the code assembles MSB→LSB across those stream bits.
static inline __device__ unsigned int grim_decode_msb_nbit(
    const unsigned char* code_bytes,
    int block_offset_bytes,
    int lane_in_block,
    int n_bit)
{
    int start_bit = lane_in_block * n_bit;
    unsigned int code = 0;
    for (int b = 0; b < n_bit; ++b) {
        int pos = start_bit + b;
        int byte_idx = pos / 8;
        int bit_in_byte = pos % 8; // 0 = the byte's MSB
        unsigned int bit =
            (code_bytes[block_offset_bytes + byte_idx] >> (7 - bit_in_byte)) & 1u;
        code = (code << 1) | bit;
    }
    return code;
}

extern "C" __global__ void grim_dequant_wna16(
    const unsigned char* __restrict__ packed,
    float* __restrict__ out,
    int num_weights,
    int n_bit,
    int num_blocks)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_weights) return;

    int block_idx = idx / 256;
    int lane_in_block = idx % 256;

    // Layout: [u32 n_bit][u32 num_blocks][packed_codes...][f16 per-block scales...][f32 tensor_scale]
    const int code_bytes_per_block = ((256 * n_bit) + 7) / 8;
    const int code_start = 8;
    const int scales_start = code_start + code_bytes_per_block * num_blocks;
    const int tensor_scale_off = scales_start + num_blocks * 2;

    const unsigned char* block_codes = packed + code_start + block_idx * code_bytes_per_block;
    unsigned short block_scale_short = ((unsigned short)packed[scales_start + block_idx * 2])
        | ((unsigned short)packed[scales_start + block_idx * 2 + 1] << 8);
    float block_scale = grim_f16_to_f32_hip(block_scale_short);
    // Use volatile to prevent compiler from caching the blob read.
    float ts = __uint_as_float(*(volatile const unsigned int*)(packed + tensor_scale_off));

    unsigned int code = grim_decode_msb_nbit(block_codes, 0, lane_in_block, n_bit);
    out[idx] = (float)code * block_scale * ts;
}
"#;

/// EmbeddingWNA16Int: row-major packed N-bit codes, one f32 tensor_scale, no per-block scale.
pub const EMBEDDING_NA16_INT_KERNEL: &str = r#"
extern "C" __global__ void grim_dequant_embedding_wna16_int(
    const unsigned char* __restrict__ packed,
    float* __restrict__ out,
    int total_elements,
    int n_bit,
    int embedding_dim,
    float tensor_scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int row = idx / embedding_dim;
    int col = idx % embedding_dim;
    // layout: [u32 n_bit][u32 embedding_dim][u32 num_rows][packed_codes...]
    // plus tail f32 tensor_scale (4 bytes) — host marshaller provides pointer to it.
    const int header_bytes = 12; // 3 u32
    const int codes_per_row = ((embedding_dim * n_bit) + 7) / 8;
    const unsigned char* row_codes = packed + header_bytes + row * codes_per_row;

    // reuse shared decoder with block_offset=0, lane=col
    unsigned int code = grim_decode_msb_nbit(row_codes, 0, col, n_bit);
    out[idx] = (float)code * tensor_scale;
}
"#;

// ── Quant workstream: fused dequant-GEMMs for the compressed-tensors
// W8A8 formats and WNA16. Family contract (matches grim_gptq_dequant_gemm /
// marlin): C[M, N] = A[M, K] @ deq(B)\u1d40, one 256-thread block per 256
// outputs, B stored in each format's documented blob layout.

/// CompressedTensors W8A8 INT8 (SmoothQuant): blob =
/// [u64 scales_len][int8 codes (N*K, row-major over output channels)]
/// [f32 per-output-channel scales (N)]. Activations arrive F32 (the int8
/// activation quantization is applied upstream per-token and is out of
/// scope for the weight-side dequant GEMM).
pub const W8A8_GEMM_KERNEL: &str = r#"
extern "C" __global__ void grim_w8a8_int8_dequant_gemm(
    const float* __restrict__ A,
    const unsigned char* __restrict__ blob,
    float* __restrict__ C,
    int M, int N, int K)
{
    const unsigned long long idx =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long total = (unsigned long long)M * N;
    if (idx >= total) return;

    const int row = (int)(idx / N);
    const int col = (int)(idx % N);

    // codes at offset 8 (after the u64 scales_len prefix), scales after.
    const unsigned char* codes = blob + 8;
    const float* scales = (const float*)(blob + 8 + (long long)N * K);

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        // int8 code, row-major [N, K]: code(col, k).
        signed char c = (signed char)codes[(long long)col * K + k];
        // Use volatile to prevent compiler from caching the scale read.
        float w = (float)c * ((volatile const float*)scales)[col];
        acc += A[row * K + k] * w;
    }
    C[row * N + col] = acc;
}

extern "C" __global__ void grim_w8a8_fp8_dequant_gemm(
    const float* __restrict__ A,
    const unsigned char* __restrict__ blob,
    float* __restrict__ C,
    int M, int N, int K)
{
    const unsigned long long idx =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long total = (unsigned long long)M * N;
    if (idx >= total) return;

    const int row = (int)(idx / N);
    const int col = (int)(idx % N);

    // [u64 scale_len][fp8 codes (N*K)][f32 per-tensor scale]
    const unsigned char* codes = blob + 8;
    // Use volatile to prevent compiler from caching the scale read.
    const float tensor_scale = *(volatile const float*)(blob + 8 + (long long)N * K);

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        float w = fp8_e4m3_to_float_hip(codes[(long long)col * K + k]) * tensor_scale;
        acc += A[row * K + k] * w;
    }
    C[row * N + col] = acc;
}

extern "C" __global__ void grim_wna16_dequant_gemm(
    const float* __restrict__ A,
    const unsigned char* __restrict__ blob,
    float* __restrict__ C,
    int M, int N, int K,
    int n_bit,
    int num_blocks)
{
    const unsigned long long idx =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long total = (unsigned long long)M * N;
    if (idx >= total) return;

    const int row = (int)(idx / N);
    const int col = (int)(idx % N);

    // [u32 n_bit][u32 num_blocks][codes][f16 block scales][f32 ts] —
    // flat 256-weight blocks over the row-major [N, K] weight.
    const int code_bytes_per_block = ((256 * (int)n_bit) + 7) / 8;
    const int code_start = 8;
    const int scales_start = code_start + code_bytes_per_block * num_blocks;
    const int ts_off = scales_start + num_blocks * 2;
    // Use volatile to prevent compiler from caching the blob read in registers.
    // Without volatile, the HIP compiler may hoist the read and reuse a stale
    // value across loop iterations, producing near-zero outputs.
    const float ts = __uint_as_float(*(volatile const unsigned int*)(blob + ts_off));

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        long long flat = (long long)col * K + k;
        int block_idx = (int)(flat >> 8);        // flat / 256
        int lane = (int)(flat & 255);            // flat % 256
        const unsigned char* block_codes =
            blob + code_start + block_idx * code_bytes_per_block;
        unsigned short h = ((unsigned short)blob[scales_start + block_idx * 2])
            | ((unsigned short)blob[scales_start + block_idx * 2 + 1] << 8);
        unsigned int code = grim_decode_msb_nbit(block_codes, 0, lane, n_bit);
        acc += A[row * K + k] * ((float)code * grim_f16_to_f32_hip(h) * ts);
    }
    C[row * N + col] = acc;
}
"#;
