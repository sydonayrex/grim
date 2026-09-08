//! NVFP4 (NVIDIA Blackwell 4-bit float with interleaved E8M0 scale) fused GEMM / GEMV kernels.

pub const NVFP4_GEMM_SOURCE: &str = r#"
// Sub-block is 16 elements:
// 1 byte E8M0 scale + 8 bytes E2M1 packed codes (low nibble = even index, high nibble = odd index)
// Total 9 bytes per 16 elements.
__device__ __forceinline__ float dequant_nvfp4_val(
    const unsigned char* __restrict__ col_bytes,
    int k_idx
) {
    const float lut[16] = {
        0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
       -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };

    int sb_idx = k_idx / 16;
    int in_sb = k_idx % 16;
    const unsigned char* blk = col_bytes + sb_idx * 9;

    unsigned char shared_exp = blk[0];
    float scale = exp2f((float)(int)shared_exp - 127.0f);

    unsigned char code_byte = blk[1 + (in_sb / 2)];
    unsigned char code = (in_sb % 2 == 0) ? (code_byte & 0x0F) : ((code_byte >> 4) & 0x0F);

    return lut[code] * scale;
}

// Fused NVFP4 GEMM: A [M, K] * B [K, N] -> C [M, N]
// B is column-major: N columns, each column has (K/16)*9 bytes.
extern "C" __global__ void grim_nvfp4_gemm(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_bytes,
    float* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    int blocks_per_col = K / 16;
    int col_stride_bytes = blocks_per_col * 9;
    const unsigned char* col_bytes = B_bytes + col * col_stride_bytes;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        float b_val = dequant_nvfp4_val(col_bytes, k);
        acc += A[row * K + k] * b_val;
    }

    C[row * N + col] = acc;
}

// Fused NVFP4 GEMV for M=1 decode: 1 warp per column
extern "C" __global__ void grim_nvfp4_gemv(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_bytes,
    float* __restrict__ C,
    int N, int K
) {
    int col = blockIdx.x;
    if (col >= N) return;

    int lane = threadIdx.x;
    int blocks_per_col = K / 16;
    int col_stride_bytes = blocks_per_col * 9;
    const unsigned char* col_bytes = B_bytes + col * col_stride_bytes;

    float sum = 0.0f;
    for (int k = lane; k < K; k += blockDim.x) {
        float b_val = dequant_nvfp4_val(col_bytes, k);
        sum += A[k] * b_val;
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (lane == 0) {
        C[col] = sum;
    }
}
"#;
