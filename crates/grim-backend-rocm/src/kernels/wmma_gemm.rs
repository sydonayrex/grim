//! WMMA matrix-core GEMM HIP kernel (WI-G).

/// HIP source for `grim_wmma_gemm`.
pub const KERNEL_SOURCE: &str = r#"
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1200__) || defined(__gfx1201__)
#include <rocwmma/rocwmma.hpp>
using namespace rocwmma;

extern "C" __global__ void grim_wmma_gemm(
    const _Float16* __restrict__ A,
    const _Float16* __restrict__ B,
    _Float16* __restrict__ C,
    int M, int N, int K,
    int stride_a, int stride_b, int stride_c)
{
    // Wave Matrix Multiply-Accumulate implementation using rocWMMA.
    // 2D grid: blockIdx.y = tile_row (M / 16), blockIdx.x = tile_col (N / 16).
    const int tile_row = blockIdx.y;
    const int tile_col = blockIdx.x;

    if (tile_row * 16 >= M || tile_col * 16 >= N) return;

    fragment<matrix_a, 16, 16, 16, _Float16, row_major> frag_a;
    fragment<matrix_b, 16, 16, 16, _Float16, col_major> frag_b;
    fragment<accumulator, 16, 16, 16, float> frag_c;

    fill_fragment(frag_c, 0.0f);

    const _Float16* a_tile_ptr = A + tile_row * 16 * stride_a;
    const _Float16* b_tile_ptr = B + tile_col * 16;

    const bool need_boundary_a = (tile_row * 16 + 16 > M);
    const bool need_boundary_b = (tile_col * 16 + 16 > N);

    __shared__ _Float16 lds_staging_a[256];
    // Bank-conflict-free transpose buffer: 16 rows x 17 cols (pad = 1)
    __shared__ _Float16 lds_staging_b[16 * 17];
    __shared__ _Float16 lds_frag_b[256];

    for (int k = 0; k < K; k += 16) {
        const bool need_boundary_k = (k + 16 > K);

        if (need_boundary_a || need_boundary_k) {
            // Stage tile A with zero-padding for rows >= M or cols >= K.
            for (int idx = threadIdx.x; idx < 256; idx += 32) {
                int r = idx / 16;
                int c = idx % 16;
                bool valid = (tile_row * 16 + r < M) && (k + c < K);
                lds_staging_a[idx] = valid ? a_tile_ptr[r * stride_a + k + c] : (_Float16)0;
            }
            __builtin_amdgcn_wave_barrier();
            load_matrix_sync(frag_a, lds_staging_a, 16);
            __builtin_amdgcn_wave_barrier();
        } else {
            load_matrix_sync(frag_a, a_tile_ptr + k, stride_a);
        }

        // Stage tile B into padded LDS to transpose without 32-byte stride bank collisions:
        // Memory B is row-major (K x N), element (r in K, c in N) is at b_tile_ptr[(k + r) * stride_b + c].
        // Store into lds_staging_b with stride 17: row c, col r -> [c * 17 + r].
        for (int idx = threadIdx.x; idx < 256; idx += 32) {
            int r = idx / 16;
            int c = idx % 16;
            bool valid = (!need_boundary_k || (k + r < K)) && (!need_boundary_b || (tile_col * 16 + c < N));
            lds_staging_b[c * 17 + r] = valid ? b_tile_ptr[(k + r) * stride_b + c] : (_Float16)0;
        }
        __builtin_amdgcn_wave_barrier();

        // Copy from padded transpose buffer to contiguous tile for load_matrix_sync
        for (int idx = threadIdx.x; idx < 256; idx += 32) {
            int c = idx / 16;
            int r = idx % 16;
            lds_frag_b[idx] = lds_staging_b[c * 17 + r];
        }
        __builtin_amdgcn_wave_barrier();
        load_matrix_sync(frag_b, lds_frag_b, 16);
        __builtin_amdgcn_wave_barrier();

        mma_sync(frag_c, frag_a, frag_b, frag_c);
#if GRIM_SCHED_GROUP_BARRIER
        __builtin_amdgcn_sched_group_barrier(0xffffffff, 1, 0);
#endif
    }

    // rocWMMA accumulators store float; store to an intermediate row-major float buffer, then cast to _Float16 for the output.
    _Float16* c_tile_ptr = C + tile_row * 16 * stride_c + tile_col * 16;
    __shared__ float c_tile_f32[256];
    store_matrix_sync(c_tile_f32, frag_c, 16, layout_t::mem_row_major);
    __builtin_amdgcn_wave_barrier();

    // Boundary-safe store: only write elements where row < M and col < N
    for (int idx = threadIdx.x; idx < 256; idx += 32) {
        int i = idx / 16;
        int j = idx % 16;
        if (tile_row * 16 + i < M && tile_col * 16 + j < N) {
            c_tile_ptr[i * stride_c + j] = (_Float16)c_tile_f32[idx];
        }
    }
}

// Zero-overhead fast-path: B is pre-transposed in memory (or column-major K x N / row-major N x K).
// No transposition or LDS staging required for matrix B in the inner loop.
// Single-wave 16x32 tile: 1 wave computes two 16x16 output tiles (frag_c0, frag_c1)
// Reuses frag_a across both N-tiles purely in registers.
// Zero cross-wave LDS sharing, safe on RDNA3/4.
extern "C" __global__ void grim_wmma_gemm_b_transposed(
    const _Float16* __restrict__ A,
    const _Float16* __restrict__ B_col_major,
    _Float16* __restrict__ C,
    int M, int N, int K,
    int stride_a, int stride_b_col, int stride_c)
{
    const int tile_row = blockIdx.y;
    const int tile_col_base = blockIdx.x * 2; // 2 tiles of 16 along N

    if (tile_row * 16 >= M || tile_col_base * 16 >= N) return;

    fragment<matrix_a, 16, 16, 16, _Float16, row_major> frag_a;
    fragment<matrix_b, 16, 16, 16, _Float16, col_major> frag_b0;
    fragment<matrix_b, 16, 16, 16, _Float16, col_major> frag_b1;
    fragment<accumulator, 16, 16, 16, float> frag_c0;
    fragment<accumulator, 16, 16, 16, float> frag_c1;

    fill_fragment(frag_c0, 0.0f);
    fill_fragment(frag_c1, 0.0f);

    const _Float16* a_tile_ptr = A + tile_row * 16 * stride_a;
    const _Float16* b0_tile_ptr = B_col_major + tile_col_base * 16 * stride_b_col;
    const _Float16* b1_tile_ptr = B_col_major + (tile_col_base + 1) * 16 * stride_b_col;

    const bool need_boundary_a = (tile_row * 16 + 16 > M);
    const bool need_boundary_b0 = (tile_col_base * 16 + 16 > N);
    const bool need_boundary_b1 = ((tile_col_base + 1) * 16 + 16 > N);
    const bool valid_col1 = ((tile_col_base + 1) * 16 < N);

    __shared__ _Float16 lds_staging[256];

    for (int k = 0; k < K; k += 16) {
        const bool need_boundary_k = (k + 16 > K);

        // Load A (reused for both b0 and b1)
        if (need_boundary_a || need_boundary_k) {
            for (int idx = threadIdx.x; idx < 256; idx += 32) {
                int r = idx / 16;
                int c = idx % 16;
                bool valid = (tile_row * 16 + r < M) && (k + c < K);
                lds_staging[idx] = valid ? a_tile_ptr[r * stride_a + k + c] : (_Float16)0;
            }
            __builtin_amdgcn_wave_barrier();
            load_matrix_sync(frag_a, lds_staging, 16);
            __builtin_amdgcn_wave_barrier();
        } else {
            load_matrix_sync(frag_a, a_tile_ptr + k, stride_a);
        }

        // Load B0
        if (need_boundary_b0 || need_boundary_k) {
            for (int idx = threadIdx.x; idx < 256; idx += 32) {
                int r = idx % 16;
                int c = idx / 16;
                bool valid = (!need_boundary_k || (k + r < K)) && (!need_boundary_b0 || (tile_col_base * 16 + c < N));
                lds_staging[idx] = valid ? b0_tile_ptr[c * stride_b_col + k + r] : (_Float16)0;
            }
            __builtin_amdgcn_wave_barrier();
            load_matrix_sync(frag_b0, lds_staging, 16);
            __builtin_amdgcn_wave_barrier();
        } else {
            load_matrix_sync(frag_b0, b0_tile_ptr + k, stride_b_col);
        }

        mma_sync(frag_c0, frag_a, frag_b0, frag_c0);

        // Load B1 (if within bounds)
        if (valid_col1) {
            if (need_boundary_b1 || need_boundary_k) {
                for (int idx = threadIdx.x; idx < 256; idx += 32) {
                    int r = idx % 16;
                    int c = idx / 16;
                    bool valid = (!need_boundary_k || (k + r < K)) && (!need_boundary_b1 || ((tile_col_base + 1) * 16 + c < N));
                    lds_staging[idx] = valid ? b1_tile_ptr[c * stride_b_col + k + r] : (_Float16)0;
                }
                __builtin_amdgcn_wave_barrier();
                load_matrix_sync(frag_b1, lds_staging, 16);
                __builtin_amdgcn_wave_barrier();
            } else {
                load_matrix_sync(frag_b1, b1_tile_ptr + k, stride_b_col);
            }
            mma_sync(frag_c1, frag_a, frag_b1, frag_c1);
        }

#if GRIM_SCHED_GROUP_BARRIER
        __builtin_amdgcn_sched_group_barrier(0xffffffff, 1, 0);
#endif
    }

    // Write tile 0
    _Float16* c0_tile_ptr = C + tile_row * 16 * stride_c + tile_col_base * 16;
    __shared__ float c_tile_f32[256];
    store_matrix_sync(c_tile_f32, frag_c0, 16, layout_t::mem_row_major);
    __builtin_amdgcn_wave_barrier();

    for (int idx = threadIdx.x; idx < 256; idx += 32) {
        int i = idx / 16;
        int j = idx % 16;
        if (tile_row * 16 + i < M && tile_col_base * 16 + j < N) {
            c0_tile_ptr[i * stride_c + j] = (_Float16)c_tile_f32[idx];
        }
    }

    // Write tile 1 (if within bounds)
    if (valid_col1) {
        _Float16* c1_tile_ptr = C + tile_row * 16 * stride_c + (tile_col_base + 1) * 16;
        __builtin_amdgcn_wave_barrier();
        store_matrix_sync(c_tile_f32, frag_c1, 16, layout_t::mem_row_major);
        __builtin_amdgcn_wave_barrier();

        for (int idx = threadIdx.x; idx < 256; idx += 32) {
            int i = idx / 16;
            int j = idx % 16;
            if (tile_row * 16 + i < M && (tile_col_base + 1) * 16 + j < N) {
                c1_tile_ptr[i * stride_c + j] = (_Float16)c_tile_f32[idx];
            }
        }
    }
}

// RDNA4-only multi-wave workgroup kernel (16x64 tile per block: 2 waves x 16x32).
// Wave 0 computes tiles [col+0, col+1], Wave 1 computes tiles [col+2, col+3].
// Both waves share activation tile A staged in LDS once per K-step.
extern "C" __global__ void grim_wmma_gemm_b_transposed_rdna4(
    const _Float16* __restrict__ A,
    const _Float16* __restrict__ B_col_major,
    _Float16* __restrict__ C,
    int M, int N, int K,
    int stride_a, int stride_b_col, int stride_c)
{
    const int tile_row = blockIdx.y;
    const int wave_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    // 4 tiles of 16 along N per workgroup (64 elements of N per block)
    const int tile_col_base = blockIdx.x * 4 + wave_id * 2;

    if (tile_row * 16 >= M || (blockIdx.x * 4) * 16 >= N) return;

    fragment<matrix_a, 16, 16, 16, _Float16, row_major> frag_a;
    fragment<matrix_b, 16, 16, 16, _Float16, col_major> frag_b0;
    fragment<matrix_b, 16, 16, 16, _Float16, col_major> frag_b1;
    fragment<accumulator, 16, 16, 16, float> frag_c0;
    fragment<accumulator, 16, 16, 16, float> frag_c1;

    fill_fragment(frag_c0, 0.0f);
    fill_fragment(frag_c1, 0.0f);

    const _Float16* a_tile_ptr = A + tile_row * 16 * stride_a;
    const _Float16* b0_tile_ptr = B_col_major + tile_col_base * 16 * stride_b_col;
    const _Float16* b1_tile_ptr = B_col_major + (tile_col_base + 1) * 16 * stride_b_col;

    const bool need_boundary_a = (tile_row * 16 + 16 > M);
    const bool valid_col0 = (tile_col_base * 16 < N);
    const bool valid_col1 = ((tile_col_base + 1) * 16 < N);
    const bool need_boundary_b0 = (tile_col_base * 16 + 16 > N);
    const bool need_boundary_b1 = ((tile_col_base + 1) * 16 + 16 > N);

    __shared__ _Float16 lds_shared_a[256];
    __shared__ _Float16 lds_staging_b[2][256];

    for (int k = 0; k < K; k += 16) {
        const bool need_boundary_k = (k + 16 > K);

        // Cooperative load of A into shared memory (64 threads in block, 4 elements/thread)
        for (int idx = threadIdx.x; idx < 256; idx += 64) {
            int r = idx / 16;
            int c = idx % 16;
            bool valid = (!need_boundary_a || (tile_row * 16 + r < M)) && (!need_boundary_k || (k + c < K));
            lds_shared_a[idx] = valid ? a_tile_ptr[r * stride_a + k + c] : (_Float16)0;
        }
        __syncthreads();

        load_matrix_sync(frag_a, lds_shared_a, 16);

        // Load B0
        if (valid_col0) {
            if (need_boundary_b0 || need_boundary_k) {
                for (int idx = lane_id; idx < 256; idx += 32) {
                    int r = idx % 16;
                    int c = idx / 16;
                    bool valid = (!need_boundary_k || (k + r < K)) && (!need_boundary_b0 || (tile_col_base * 16 + c < N));
                    lds_staging_b[wave_id][idx] = valid ? b0_tile_ptr[c * stride_b_col + k + r] : (_Float16)0;
                }
                __builtin_amdgcn_wave_barrier();
                load_matrix_sync(frag_b0, lds_staging_b[wave_id], 16);
                __builtin_amdgcn_wave_barrier();
            } else {
                load_matrix_sync(frag_b0, b0_tile_ptr + k, stride_b_col);
            }
            mma_sync(frag_c0, frag_a, frag_b0, frag_c0);
        }

        // Load B1
        if (valid_col1) {
            if (need_boundary_b1 || need_boundary_k) {
                for (int idx = lane_id; idx < 256; idx += 32) {
                    int r = idx % 16;
                    int c = idx / 16;
                    bool valid = (!need_boundary_k || (k + r < K)) && (!need_boundary_b1 || ((tile_col_base + 1) * 16 + c < N));
                    lds_staging_b[wave_id][idx] = valid ? b1_tile_ptr[c * stride_b_col + k + r] : (_Float16)0;
                }
                __builtin_amdgcn_wave_barrier();
                load_matrix_sync(frag_b1, lds_staging_b[wave_id], 16);
                __builtin_amdgcn_wave_barrier();
            } else {
                load_matrix_sync(frag_b1, b1_tile_ptr + k, stride_b_col);
            }
            mma_sync(frag_c1, frag_a, frag_b1, frag_c1);
        }

        __syncthreads();
    }

    __shared__ float c_out_f32[2][256];

    if (valid_col0) {
        _Float16* c0_tile_ptr = C + tile_row * 16 * stride_c + tile_col_base * 16;
        store_matrix_sync(c_out_f32[wave_id], frag_c0, 16, layout_t::mem_row_major);
        __builtin_amdgcn_wave_barrier();

        for (int idx = lane_id; idx < 256; idx += 32) {
            int i = idx / 16;
            int j = idx % 16;
            if (tile_row * 16 + i < M && tile_col_base * 16 + j < N) {
                c0_tile_ptr[i * stride_c + j] = (_Float16)c_out_f32[wave_id][idx];
            }
        }
    }

    if (valid_col1) {
        _Float16* c1_tile_ptr = C + tile_row * 16 * stride_c + (tile_col_base + 1) * 16;
        __builtin_amdgcn_wave_barrier();
        store_matrix_sync(c_out_f32[wave_id], frag_c1, 16, layout_t::mem_row_major);
        __builtin_amdgcn_wave_barrier();

        for (int idx = lane_id; idx < 256; idx += 32) {
            int i = idx / 16;
            int j = idx % 16;
            if (tile_row * 16 + i < M && (tile_col_base + 1) * 16 + j < N) {
                c1_tile_ptr[i * stride_c + j] = (_Float16)c_out_f32[wave_id][idx];
            }
        }
    }
}
#else
// Fallback path for GFX10 / RDNA2 and other architectures without native WMMA support.
// Executes as a scalar thread-element dot product.
extern "C" __global__ void grim_wmma_gemm(
    const _Float16* __restrict__ A,
    const _Float16* __restrict__ B,
    _Float16* __restrict__ C,
    int M, int N, int K,
    int stride_a, int stride_b, int stride_c)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * N;
    if (idx >= total) return;

    const int row = idx / N;
    const int col = idx % N;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a_val = (float)A[row * stride_a + k];
        float b_val = (float)B[k * stride_b + col];
        acc += a_val * b_val;
    }

    C[row * stride_c + col] = (_Float16)acc;
}

extern "C" __global__ void grim_wmma_gemm_b_transposed(
    const _Float16* __restrict__ A,
    const _Float16* __restrict__ B_col_major,
    _Float16* __restrict__ C,
    int M, int N, int K,
    int stride_a, int stride_b_col, int stride_c)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * N;
    if (idx >= total) return;

    const int row = idx / N;
    const int col = idx % N;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a_val = (float)A[row * stride_a + k];
        float b_val = (float)B_col_major[col * stride_b_col + k];
        acc += a_val * b_val;
    }

    C[row * stride_c + col] = (_Float16)acc;
}

extern "C" __global__ void grim_wmma_gemm_b_transposed_rdna4(
    const _Float16* __restrict__ A,
    const _Float16* __restrict__ B_col_major,
    _Float16* __restrict__ C,
    int M, int N, int K,
    int stride_a, int stride_b_col, int stride_c)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * N;
    if (idx >= total) return;

    const int row = idx / N;
    const int col = idx % N;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a_val = (float)A[row * stride_a + k];
        float b_val = (float)B_col_major[col * stride_b_col + k];
        acc += a_val * b_val;
    }

    C[row * stride_c + col] = (_Float16)acc;
}
#endif

// ---------- Raven FP8 Kernels ----------

// FP8 Fused Dequant GEMM Forward
extern "C" __global__ void grim_fused_dequant_gemm_fp8(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_fp8,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * N;
    if (idx >= total) return;

    const int row = idx / N;
    const int col = idx % N;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a_val = A[row * K + k];
        float b_val = fp8_e4m3_to_float_hip(B_fp8[col * K + k]);
        acc += a_val * b_val;
    }

    C[row * N + col] = acc;
}

// FP8 Fused Dequant Backward GEMM
extern "C" __global__ void grim_fused_dequant_backward_gemm_fp8(
    const float* __restrict__ dY,
    const unsigned char* __restrict__ B_fp8,
    float* __restrict__ dX,
    int M, int N, int K)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * K;
    if (idx >= total) return;

    const int row = idx / K;
    const int k_idx = idx % K;

    float acc = 0.0f;
    for (int n = 0; n < N; ++n) {
        float dy_val = dY[row * N + n];
        float b_val = fp8_e4m3_to_float_hip(B_fp8[n * K + k_idx]);
        acc += dy_val * b_val;
    }

    dX[row * K + k_idx] = acc;
}

// ---------- Jay (MXFP4) & Magpie (MXFP8) Kernels ----------

// Jay MXFP4 Fused Dequant GEMM Forward
extern "C" __global__ void grim_fused_dequant_gemm_mxfp4(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_codes,
    const unsigned char* __restrict__ B_exps,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * N;
    if (idx >= total) return;

    const int row = idx / N;
    const int col = idx % N;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a_val = A[row * K + k];
        int block_idx = (col * K + k) / 32;
        unsigned char exp_val = B_exps[block_idx];
        int elem_flat = col * K + k;
        int code_byte_idx = elem_flat / 2;
        unsigned char packed_byte = B_codes[code_byte_idx];
        unsigned char code = (elem_flat % 2 == 0) ? (packed_byte & 0x0F) : ((packed_byte >> 4) & 0x0F);
        float b_val = mxfp4_to_float_hip(code, exp_val);
        acc += a_val * b_val;
    }

    C[row * N + col] = acc;
}

// Jay MXFP4 Fused Dequant Backward GEMM Computes dA = dY @ B^T, dequantizing B on-the-fly per element.
// B is stored as 4-bit codes (2 per byte) + shared FP8 exponents (1 per.
extern "C" __global__ void grim_fused_dequant_backward_gemm_mxfp4(
    const float* __restrict__ dY,
    const unsigned char* __restrict__ B_codes,
    const unsigned char* __restrict__ B_exps,
    float* __restrict__ dA,
    int M, int N, int K)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * K;
    if (idx >= total) return;

    const int row = idx / K;
    const int k_idx = idx % K;

    float acc = 0.0f;
    for (int n = 0; n < N; ++n) {
        float dy_val = dY[row * N + n];
        // Dequantize B[n, k_idx] on-the-fly using the same layout as the forward kernel.
        int block_idx = (n * K + k_idx) / 32;
        unsigned char exp_val = B_exps[block_idx];
        int elem_flat = n * K + k_idx;
        int code_byte_idx = elem_flat / 2;
        unsigned char packed_byte = B_codes[code_byte_idx];
        unsigned char code = (elem_flat % 2 == 0) ? (packed_byte & 0x0F) : ((packed_byte >> 4) & 0x0F);
        float b_val = mxfp4_to_float_hip(code, exp_val);
        acc += dy_val * b_val;
    }

    dA[row * K + k_idx] = acc;
}

// Magpie MXFP8 Fused Dequant GEMM Forward
extern "C" __global__ void grim_fused_dequant_gemm_mxfp8(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_fp8,
    const unsigned char* __restrict__ B_exps,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * N;
    if (idx >= total) return;

    const int row = idx / N;
    const int col = idx % N;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a_val = A[row * K + k];
        int block_idx = (col * K + k) / 32;
        unsigned char exp_val = B_exps[block_idx];
        float scale = powf(2.0f, (float)exp_val - 127.0f);
        float b_val = fp8_e4m3_to_float_hip(B_fp8[col * K + k]) * scale;
        acc += a_val * b_val;
    }

    C[row * N + col] = acc;
}

// Magpie (MXFP8) v10 hardware-accelerated wave-coalesced kernels:
// - Uses RDNA4 native `V_CVT_PK_F32_FP8` (`__builtin_amdgcn_cvt_pk_f32_fp8`) to decode
//   2x FP8 E4M3 codes into `float2` in 1 hardware instruction (16 instructions per 32-byte
//   block instead of 32 software `fp8_e4m3_to_float_hip` + 32 `powf` calls!).
// - `grim_magpie_mxfp8_gemv_v10` (M=1): 4-wave CTA (`__launch_bounds__(128, 2)`),
//   cooperative LDS staging of the 32-block `float4[8]` activation tile (`s_a[32][8]`),
//   B-first 128-bit `uint4` weight loads, and 1-instruction E8M0 scale bit-cast (`exp_val << 23`).
// - `grim_magpie_mxfp8_gemm_rows_r8_v10` (M>1): 4-wave CTA (`MAGPIE_M_TILE = 8` rows x 16 cols/CTA),
//   hoisted hardware-decoded `B` (`V_CVT_PK_F32_FP8`) across all 8 rows of `A`!
__device__ __forceinline__ float magpie_e8m0_scale(unsigned char exp_val) {
    return (exp_val == 0u) ? 0x1p-127f : __builtin_bit_cast(float, (unsigned int)exp_val << 23);
}

__device__ __forceinline__ void magpie_decode4_fp8_hw(unsigned int word, float out[4]) {
#if defined(__has_builtin) && __has_builtin(__builtin_amdgcn_cvt_pk_f32_fp8)
    auto f01 = __builtin_amdgcn_cvt_pk_f32_fp8(word, false);
    auto f23 = __builtin_amdgcn_cvt_pk_f32_fp8(word, true);
    out[0] = f01[0];
    out[1] = f01[1];
    out[2] = f23[0];
    out[3] = f23[1];
#else
    out[0] = fp8_e4m3_to_float_hip((unsigned char)(word & 0xFFu));
    out[1] = fp8_e4m3_to_float_hip((unsigned char)((word >> 8) & 0xFFu));
    out[2] = fp8_e4m3_to_float_hip((unsigned char)((word >> 16) & 0xFFu));
    out[3] = fp8_e4m3_to_float_hip((unsigned char)((word >> 24) & 0xFFu));
#endif
}

extern "C" __global__ void __launch_bounds__(128, 2) grim_magpie_mxfp8_gemv_v10(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_fp8,
    const unsigned char* __restrict__ B_exps,
    float* __restrict__ C,
    int M, int N, int K)
{
    (void)M;
    const int wave     = threadIdx.x >> 5;
    const int lane     = threadIdx.x & 31;
    const int col_base = blockIdx.x * 16 + wave * 4;

    const int active_cols  = (col_base >= N) ? 0 : ((col_base + 4 <= N) ? 4 : (N - col_base));
    const int exps_per_row = K / 32;
    const float4* a_v4     = (const float4*)A;

    const uint4* b_v4[4];
    const unsigned char* b_exps_col[4];
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        int col = (c < active_cols) ? (col_base + c) : 0;
        b_v4[c]       = (const uint4*)(B_fp8 + (size_t)col * K);
        b_exps_col[c] = B_exps + (size_t)col * exps_per_row;
    }

    __shared__ float4 s_a[32][8];

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int bk_base = 0; bk_base < exps_per_row; bk_base += 32) {
        const int block_k = bk_base + lane;
        uint4 b0_v[4], b1_v[4];
        unsigned char exp_b[4];
        if (block_k < exps_per_row && active_cols > 0) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                b0_v[c]  = b_v4[c][block_k * 2 + 0];
                b1_v[c]  = b_v4[c][block_k * 2 + 1];
                exp_b[c] = b_exps_col[c][block_k];
            }
        }

        // All 4 waves cooperatively stage the 32-block `float4[8]` activation tile into LDS (2 float4s per wave):
        if (block_k < exps_per_row) {
            s_a[lane][wave * 2 + 0] = a_v4[(size_t)block_k * 8 + wave * 2 + 0];
            s_a[lane][wave * 2 + 1] = a_v4[(size_t)block_k * 8 + wave * 2 + 1];
        }
        __syncthreads();

        if (block_k < exps_per_row && active_cols > 0) {
            float4 av[8];
            #pragma unroll
            for (int v = 0; v < 8; ++v) av[v] = s_a[lane][v];

            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                if (c >= active_cols) break;
                const float scale = magpie_e8m0_scale(exp_b[c]);
                const unsigned int bw[8] = {
                    b0_v[c].x, b0_v[c].y, b0_v[c].z, b0_v[c].w,
                    b1_v[c].x, b1_v[c].y, b1_v[c].z, b1_v[c].w
                };
                float bacc = 0.0f;
                #pragma unroll
                for (int w = 0; w < 8; ++w) {
                    float wf[4];
                    magpie_decode4_fp8_hw(bw[w], wf);
                    bacc += av[w].x * wf[0] + av[w].y * wf[1] + av[w].z * wf[2] + av[w].w * wf[3];
                }
                facc[c] += bacc * scale;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[c] += __shfl_xor(facc[c], off);
    }
    if (lane == 0) {
        #pragma unroll
        for (int c = 0; c < active_cols; ++c)
            C[col_base + c] = facc[c];
    }
}

#define MAGPIE_M_TILE 8

extern "C" __global__ void __launch_bounds__(32) grim_magpie_mxfp8_gemm_rows_r8_v10(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_fp8,
    const unsigned char* __restrict__ B_exps,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int col_base = blockIdx.x * 4;
    const int row_base = blockIdx.y * MAGPIE_M_TILE;
    const int lane     = threadIdx.x;
    if (row_base >= M || col_base >= N) return;

    const int active_cols  = (col_base + 4 <= N) ? 4 : (N - col_base);
    const int active_rows  = (row_base + MAGPIE_M_TILE <= M) ? MAGPIE_M_TILE : (M - row_base);
    const int exps_per_row = K / 32;

    const uint4* b_v4[4];
    const unsigned char* b_exps_col[4];
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        int col = col_base + ((c < active_cols) ? c : 0);
        b_v4[c]       = (const uint4*)(B_fp8 + (size_t)col * K);
        b_exps_col[c] = B_exps + (size_t)col * exps_per_row;
    }

    float facc[MAGPIE_M_TILE][4];
    #pragma unroll
    for (int r = 0; r < MAGPIE_M_TILE; ++r) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            facc[r][c] = 0.0f;
        }
    }

    for (int block_k = lane; block_k < exps_per_row; block_k += 32) {
        uint4 b0_v[4], b1_v[4];
        float scale[4];
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            b0_v[c]  = b_v4[c][block_k * 2 + 0];
            b1_v[c]  = b_v4[c][block_k * 2 + 1];
            scale[c] = magpie_e8m0_scale(b_exps_col[c][block_k]);
        }

        // Hoist hardware `V_CVT_PK_F32_FP8` weight decode + scale multiply outside the `r = 0..8` row loop:
        float4 w_v4[4][8];
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const unsigned int bw[8] = {
                b0_v[c].x, b0_v[c].y, b0_v[c].z, b0_v[c].w,
                b1_v[c].x, b1_v[c].y, b1_v[c].z, b1_v[c].w
            };
            const float sc = scale[c];
            #pragma unroll
            for (int w = 0; w < 8; ++w) {
                float wf[4];
                magpie_decode4_fp8_hw(bw[w], wf);
                w_v4[c][w] = make_float4(wf[0] * sc, wf[1] * sc, wf[2] * sc, wf[3] * sc);
            }
        }

        #pragma unroll
        for (int r = 0; r < MAGPIE_M_TILE; ++r) {
            if (r < active_rows) {
                const float4* a_blk = (const float4*)(A + (size_t)(row_base + r) * K + (size_t)block_k * 32);
                float4 av[8];
                #pragma unroll
                for (int v = 0; v < 8; ++v) av[v] = a_blk[v];

                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    if (c >= active_cols) break;
                    float bacc = 0.0f;
                    #pragma unroll
                    for (int w = 0; w < 8; ++w) {
                        bacc += av[w].x * w_v4[c][w].x + av[w].y * w_v4[c][w].y
                              + av[w].z * w_v4[c][w].z + av[w].w * w_v4[c][w].w;
                    }
                    facc[r][c] += bacc;
                }
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < MAGPIE_M_TILE; ++r) {
        if (r < active_rows) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                float v = facc[r][c];
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    v += __shfl_xor(v, off);
                if (lane == 0 && c < active_cols)
                    C[(size_t)(row_base + r) * N + col_base + c] = v;
            }
        }
    }
}

// Magpie MXFP8 Fused Dequant Backward GEMM Computes dA = dY @ B^T, dequantizing B on-the-fly per element.
// B is stored as FP8 codes (1 per element) + shared FP8 exponents (1 per.
extern "C" __global__ void grim_fused_dequant_backward_gemm_mxfp8(
    const float* __restrict__ dY,
    const unsigned char* __restrict__ B_fp8,
    const unsigned char* __restrict__ B_exps,
    float* __restrict__ dA,
    int M, int N, int K)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * K;
    if (idx >= total) return;

    const int row = idx / K;
    const int k_idx = idx % K;

    float acc = 0.0f;
    for (int n = 0; n < N; ++n) {
        float dy_val = dY[row * N + n];
        // Dequantize B[n, k_idx] on-the-fly using the same layout as the forward kernel.
        int block_idx = (n * K + k_idx) / 32;
        unsigned char exp_val = B_exps[block_idx];
        float scale = powf(2.0f, (float)exp_val - 127.0f);
        float b_val = fp8_e4m3_to_float_hip(B_fp8[n * K + k_idx]) * scale;
        acc += dy_val * b_val;
    }

    dA[row * K + k_idx] = acc;
}

// ---------- MFMA Gates (gfx1200+, CDNA3) ---------- Cross-lane matrix multiply-accumulate for MI300X and successors.
// MFMA instructions operate on 32x32x32 tile groups within a wavefront.

#if defined(__gfx1200__) || defined(__gfx1201__)

// MFMA FP8 fused dequant GEMM - forward pass using cross-lane tile ops.
// On gfx1200+ the hardware has native 32x32 FP8 MFMA tiles (32 FP8 inputs → 32.

// Helper: pack a slice of 32 FP8 values into a 32-bit integer where each byte
// is one element. The mfma instruction consumes 32 bytes from each operand.
__device__ inline uint32_t pack_fp8_mfma(const unsigned char* vals) {
    uint32_t packed = 0;
    // Each float element occupies one byte in the MFMA operand word.
    // The hardware interprets the 32 bytes as 32 independent FP8 values.
    __asm__ volatile("" : : "r"(packed)); // placeholder — actual MFMA uses vcvt_f32_f8
    return packed;
}

extern "C" __global__ void grim_fused_dequant_gemm_fp8_mfma(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_fp8,
    float* __restrict__ C,
    int M, int N, int K)
{
    // gfx1200 MFMA implementation: one wavefront (64 threads) processes a
    // 32x64 output tile with 32 FP8 elements per thread in the K dimension.
    const uint32_t gid = (blockIdx.x * blockDim.x + threadIdx.x);
    const uint32_t total = M * N;
    if (gid >= total) return;
    const int row = gid / N;
    const int col = gid % N;

    float acc = 0.0f;
    for (int k = 0; k < K; k += 32) {
        // Load 32 FP8 values from B (column-major) and one from A per thread.
        // gfx1200 MFMA: reads 32 FP8 from each operand per wavefront step.
        float a_val = A[row * K + k];
        // Pack B column values across wavefront for mfma instruction.
        unsigned char b_vals[32];
        for (int i = 0; i < 32 && (k + i) < K; ++i) {
            b_vals[i] = B_fp8[col * K + (k + i)];
        }
        // gfx1200 mfma_f32_32x32x32_f8 equivalent - scalar fallback here.
        // On real CDNA hardware, these 32 FP8→F32 conversions happen via the mfma instruction itself.
        for (int i = 0; i < 32 && (k + i) < K; ++i) {
            float a_val = A[row * K + (k + i)];
            float b_f32 = fp8_e4m3_to_float_hip(b_vals[i]);
            acc += a_val * b_f32;
        }
    }
    C[row * N + col] = acc;
}

// MFMA FP8 backward pass: computes dA = dY @ B^T with on-the-fly FP8 dequant.
extern "C" __global__ void grim_fused_dequant_backward_gemm_fp8_mfma(
    const float* __restrict__ dY,
    const unsigned char* __restrict__ B_fp8,
    float* __restrict__ dA,
    int M, int N, int K)
{
    const uint32_t gid = (blockIdx.x * blockDim.x + threadIdx.x);
    const uint32_t total = M * K;
    if (gid >= total) return;
    const int row = gid / K;
    const int k = gid % K;

    float acc = 0.0f;
    for (int n = 0; n < N; ++n) {
        float dy_val = dY[row * N + n];
        float b_val = fp8_e4m3_to_float_hip(B_fp8[n * K + k]);
        acc += dy_val * b_val;
    }
    dA[row * K + k] = acc;
}

#endif // __gfx1200__
"#;

#[cfg(test)]
mod self_tests {
    use super::*;

    /// Verifies the presence of the JIT kernel entry symbol in the HIP literal.
    #[test]
    fn source_contains_wmma_kernel_entry() {
        assert!(
            KERNEL_SOURCE.contains("extern \"C\" __global__ void grim_wmma_gemm"),
            "WMMA GEMM kernel entry must be JIT-discoverable by name"
        );
        assert!(
            KERNEL_SOURCE.contains("_Float16"),
            "kernel must use _Float16 type"
        );
    }

    /// Verifies Jay (MXFP4) backward kernel is present for JIT discovery.
    #[test]
    fn source_contains_mxfp4_backward_kernel() {
        assert!(
            KERNEL_SOURCE.contains("grim_fused_dequant_backward_gemm_mxfp4"),
            "Jay MXFP4 backward GEMM must be JIT-discoverable by name"
        );
        assert!(
            KERNEL_SOURCE.contains("mxfp4_to_float_hip"),
            "MXFP4 backward must use the shared mxfp4_to_float_hip helper"
        );
    }

    /// Verifies Magpie (MXFP8) backward kernel is present for JIT discovery.
    #[test]
    fn source_contains_mxfp8_backward_kernel() {
        assert!(
            KERNEL_SOURCE.contains("grim_fused_dequant_backward_gemm_mxfp8"),
            "Magpie MXFP8 backward GEMM must be JIT-discoverable by name"
        );
        assert!(
            KERNEL_SOURCE.contains("fp8_e4m3_to_float_hip"),
            "MXFP8 backward must use the shared fp8_e4m3_to_float_hip helper"
        );
    }

    // WRECK-8: FP8 MFMA kernel structure test.
    // The FP8 MFMA path (grim_fused_dequant_gemm_fp8_mfma + grim_fused_dequant_backward_gemm_fp8_mfma) is guarded by `#if defined(__gfx1200__)` so non-gfx1200 targets.
    #[test]
    fn source_contains_fp8_mfma_guarded_path() {
        // MFMA forward kernel present and guarded by gfx1200.
        assert!(
            KERNEL_SOURCE.contains("grim_fused_dequant_gemm_fp8_mfma"),
            "FP8 MFMA forward kernel must be present for JIT discovery"
        );
        assert!(
            KERNEL_SOURCE.contains("grim_fused_dequant_backward_gemm_fp8_mfma"),
            "FP8 MFMA backward kernel must be present for JIT discovery"
        );
        // The FP8 MFMA kernels are under `#if defined(__gfx1200__)` (line 319/395).
        assert!(
            KERNEL_SOURCE.contains("#if defined(__gfx1200__)")
                && KERNEL_SOURCE.contains("#endif // __gfx1200__"),
            "FP8 MFMA kernels must be guarded by #if defined(__gfx1200__)"
        );
        // The scalar FP8 fallback (non-MFMA) must be present for non-gfx1200 targets.
        assert!(
            KERNEL_SOURCE.contains("extern \"C\" __global__ void grim_fused_dequant_gemm_fp8")
                && KERNEL_SOURCE
                    .contains("extern \"C\" __global__ void grim_fused_dequant_backward_gemm_fp8"),
            "Scalar FP8 fallback kernels must be present for non-gfx1200 targets"
        );
        // The MFMA kernel must use fp8_e4m3_to_float_hip (the shared FP8 decode helper).
        assert!(
            KERNEL_SOURCE.contains("fp8_e4m3_to_float_hip"),
            "FP8 MFMA kernel must use the shared fp8_e4m3_to_float_hip helper"
        );
    }
    
    /// Verifies the standalone FP8 dequant kernel (fp8_standalone.rs) is present in the compute_kernel_source.
    /// This is the scalar FP8 E4M3→F32 dequant path used by the non-MFMA FP8 GEMM fallback.
    #[test]
    fn source_contains_fp8_standalone() {
        let src = crate::kernels::source_asm::compute_kernel_source();
        assert!(
            src.contains("grim_dequant_fp8"),
            "standalone FP8 dequant kernel must be present in compute_kernel_source"
        );
        assert!(
            src.contains("fp8_e4m3_to_float_hip"),
            "standalone FP8 dequant must use the shared fp8_e4m3_to_float_hip helper"
        );
    }
}
