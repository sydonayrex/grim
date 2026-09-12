//! WMMA fused-dequant IQ-family GEMM HIP kernel (SPEED-ROC).
//!
//! Computes C[M,N] = A[M,K] @ B^T where B is IQ-family packed.
//! Uses rocWMMA 16×16×16 tensor-core tiles with inline IQ dequantization.
//! Compatible with RDNA3 (gfx11xx) and RDNA4 (gfx12xx).
//!
//! This kernel reuses the dequant_iqXX device functions from kernels::iq_gemm.
//! It must be appended to the JIT aggregate AFTER iq_gemm.

/// HIP source for all 7 IQ-family WMMA fused-dequant GEMM kernels.
/// Each kernel is written explicitly (not macro-stamped) for HIP preprocessor compatibility.
pub const KERNEL_SOURCE: &str = r#"
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1200__) || defined(__gfx1201__)
#include <rocwmma/rocwmma.hpp>
using namespace rocwmma;

#define QK_K 256

// --- IQ2_XXS (66 bytes / 256 weights) ---
__device__ __forceinline__ void dequant_iq_iq2xxs_chunk_fp16(
    const unsigned char* block, int k_start, _Float16* out) {
    for (int i = 0; i < 16; ++i) {
        out[i] = (_Float16)dequant_iq2xxs(block, k_start + i);
    }
}

extern "C" __global__ void grim_wmma_fused_dequant_iq2xxs(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_iq,
    float* __restrict__ C,
    int M, int N, int K) {
    const int tile_row = blockIdx.y;
    const int tile_col = blockIdx.x;
    if (tile_row * 16 >= M || tile_col * 16 >= N) return;
    fragment<matrix_a, 16, 16, 16, _Float16, row_major> frag_a;
    fragment<matrix_b, 16, 16, 16, _Float16, col_major> frag_b;
    fragment<accumulator, 16, 16, 16, float> frag_c;
    fill_fragment(frag_c, 0.0f);
    const int row_base = tile_row * 16;
    const int col_base = tile_col * 16;
    const int n_blocks_per_row = K / QK_K;
    _Float16 a_tile[16 * 16];
    _Float16 b_tile[16 * 16];
    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int i = 0; i < 16; ++i) {
            int row = row_base + i;
            for (int j = 0; j < 16; ++j) {
                int kk = k0 + j;
                a_tile[i * 16 + j] = (row < M && kk < K) ? (_Float16)A[row * K + kk] : (_Float16)0;
            }
        }
        load_matrix_sync(frag_a, a_tile, 16);
        for (int j = 0; j < 16; ++j) {
            int col = col_base + j;
            if (col < N) {
                const unsigned char* row_ptr = B_iq + (long long)col * n_blocks_per_row * 66;
                int block_idx = k0 / QK_K;
                int k_in_block = k0 % QK_K;
                dequant_iq_iq2xxs_chunk_fp16(row_ptr + (long long)block_idx * 66, k_in_block, &b_tile[j * 16]);
            } else {
                for (int k = 0; k < 16; ++k) b_tile[j * 16 + k] = (_Float16)0;
            }
        }
        load_matrix_sync(frag_b, b_tile, 16);
        mma_sync(frag_c, frag_a, frag_b, frag_c);
    }
    float* c_tile_ptr = C + tile_row * 16 * N + tile_col * 16;
    __shared__ float c_tile_f32[16 * 16];
    store_matrix_sync(c_tile_f32, frag_c, 16, layout_t::mem_row_major);
    __builtin_amdgcn_wave_barrier();
    for (int idx = threadIdx.x; idx < 256; idx += 32) {
        int i = idx / 16;
        int j = idx % 16;
        if (tile_row * 16 + i < M && tile_col * 16 + j < N) {
            c_tile_ptr[i * N + j] = c_tile_f32[idx];
        }
    }
}

// --- IQ2_XS (74 bytes / 256 weights) ---
__device__ __forceinline__ void dequant_iq_iq2xs_chunk_fp16(
    const unsigned char* block, int k_start, _Float16* out) {
    for (int i = 0; i < 16; ++i) {
        out[i] = (_Float16)dequant_iq2xs(block, k_start + i);
    }
}

extern "C" __global__ void grim_wmma_fused_dequant_iq2xs(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_iq,
    float* __restrict__ C,
    int M, int N, int K) {
    const int tile_row = blockIdx.y;
    const int tile_col = blockIdx.x;
    if (tile_row * 16 >= M || tile_col * 16 >= N) return;
    fragment<matrix_a, 16, 16, 16, _Float16, row_major> frag_a;
    fragment<matrix_b, 16, 16, 16, _Float16, col_major> frag_b;
    fragment<accumulator, 16, 16, 16, float> frag_c;
    fill_fragment(frag_c, 0.0f);
    const int row_base = tile_row * 16;
    const int col_base = tile_col * 16;
    const int n_blocks_per_row = K / QK_K;
    _Float16 a_tile[16 * 16];
    _Float16 b_tile[16 * 16];
    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int i = 0; i < 16; ++i) {
            int row = row_base + i;
            for (int j = 0; j < 16; ++j) {
                int kk = k0 + j;
                a_tile[i * 16 + j] = (row < M && kk < K) ? (_Float16)A[row * K + kk] : (_Float16)0;
            }
        }
        load_matrix_sync(frag_a, a_tile, 16);
        for (int j = 0; j < 16; ++j) {
            int col = col_base + j;
            if (col < N) {
                const unsigned char* row_ptr = B_iq + (long long)col * n_blocks_per_row * 74;
                int block_idx = k0 / QK_K;
                int k_in_block = k0 % QK_K;
                dequant_iq_iq2xs_chunk_fp16(row_ptr + (long long)block_idx * 74, k_in_block, &b_tile[j * 16]);
            } else {
                for (int k = 0; k < 16; ++k) b_tile[j * 16 + k] = (_Float16)0;
            }
        }
        load_matrix_sync(frag_b, b_tile, 16);
        mma_sync(frag_c, frag_a, frag_b, frag_c);
    }
    float* c_tile_ptr = C + tile_row * 16 * N + tile_col * 16;
    __shared__ float c_tile_f32[16 * 16];
    store_matrix_sync(c_tile_f32, frag_c, 16, layout_t::mem_row_major);
    __builtin_amdgcn_wave_barrier();
    for (int idx = threadIdx.x; idx < 256; idx += 32) {
        int i = idx / 16;
        int j = idx % 16;
        if (tile_row * 16 + i < M && tile_col * 16 + j < N) {
            c_tile_ptr[i * N + j] = c_tile_f32[idx];
        }
    }
}

// --- IQ2_S (82 bytes / 256 weights) ---
__device__ __forceinline__ void dequant_iq_iq2s_chunk_fp16(
    const unsigned char* block, int k_start, _Float16* out) {
    for (int i = 0; i < 16; ++i) {
        out[i] = (_Float16)dequant_iq2s(block, k_start + i);
    }
}

extern "C" __global__ void grim_wmma_fused_dequant_iq2s(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_iq,
    float* __restrict__ C,
    int M, int N, int K) {
    const int tile_row = blockIdx.y;
    const int tile_col = blockIdx.x;
    if (tile_row * 16 >= M || tile_col * 16 >= N) return;
    fragment<matrix_a, 16, 16, 16, _Float16, row_major> frag_a;
    fragment<matrix_b, 16, 16, 16, _Float16, col_major> frag_b;
    fragment<accumulator, 16, 16, 16, float> frag_c;
    fill_fragment(frag_c, 0.0f);
    const int row_base = tile_row * 16;
    const int col_base = tile_col * 16;
    const int n_blocks_per_row = K / QK_K;
    _Float16 a_tile[16 * 16];
    _Float16 b_tile[16 * 16];
    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int i = 0; i < 16; ++i) {
            int row = row_base + i;
            for (int j = 0; j < 16; ++j) {
                int kk = k0 + j;
                a_tile[i * 16 + j] = (row < M && kk < K) ? (_Float16)A[row * K + kk] : (_Float16)0;
            }
        }
        load_matrix_sync(frag_a, a_tile, 16);
        for (int j = 0; j < 16; ++j) {
            int col = col_base + j;
            if (col < N) {
                const unsigned char* row_ptr = B_iq + (long long)col * n_blocks_per_row * 82;
                int block_idx = k0 / QK_K;
                int k_in_block = k0 % QK_K;
                dequant_iq_iq2s_chunk_fp16(row_ptr + (long long)block_idx * 82, k_in_block, &b_tile[j * 16]);
            } else {
                for (int k = 0; k < 16; ++k) b_tile[j * 16 + k] = (_Float16)0;
            }
        }
        load_matrix_sync(frag_b, b_tile, 16);
        mma_sync(frag_c, frag_a, frag_b, frag_c);
    }
    float* c_tile_ptr = C + tile_row * 16 * N + tile_col * 16;
    __shared__ float c_tile_f32[16 * 16];
    store_matrix_sync(c_tile_f32, frag_c, 16, layout_t::mem_row_major);
    __builtin_amdgcn_wave_barrier();
    for (int idx = threadIdx.x; idx < 256; idx += 32) {
        int i = idx / 16;
        int j = idx % 16;
        if (tile_row * 16 + i < M && tile_col * 16 + j < N) {
            c_tile_ptr[i * N + j] = c_tile_f32[idx];
        }
    }
}

// --- IQ3_XXS (96 bytes / 256 weights) ---
__device__ __forceinline__ void dequant_iq_iq3xxs_chunk_fp16(
    const unsigned char* block, int k_start, _Float16* out) {
    for (int i = 0; i < 16; ++i) {
        out[i] = (_Float16)dequant_iq3xxs(block, k_start + i);
    }
}

extern "C" __global__ void grim_wmma_fused_dequant_iq3xxs(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_iq,
    float* __restrict__ C,
    int M, int N, int K) {
    const int tile_row = blockIdx.y;
    const int tile_col = blockIdx.x;
    if (tile_row * 16 >= M || tile_col * 16 >= N) return;
    fragment<matrix_a, 16, 16, 16, _Float16, row_major> frag_a;
    fragment<matrix_b, 16, 16, 16, _Float16, col_major> frag_b;
    fragment<accumulator, 16, 16, 16, float> frag_c;
    fill_fragment(frag_c, 0.0f);
    const int row_base = tile_row * 16;
    const int col_base = tile_col * 16;
    const int n_blocks_per_row = K / QK_K;
    _Float16 a_tile[16 * 16];
    _Float16 b_tile[16 * 16];
    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int i = 0; i < 16; ++i) {
            int row = row_base + i;
            for (int j = 0; j < 16; ++j) {
                int kk = k0 + j;
                a_tile[i * 16 + j] = (row < M && kk < K) ? (_Float16)A[row * K + kk] : (_Float16)0;
            }
        }
        load_matrix_sync(frag_a, a_tile, 16);
        for (int j = 0; j < 16; ++j) {
            int col = col_base + j;
            if (col < N) {
                const unsigned char* row_ptr = B_iq + (long long)col * n_blocks_per_row * 96;
                int block_idx = k0 / QK_K;
                int k_in_block = k0 % QK_K;
                dequant_iq_iq3xxs_chunk_fp16(row_ptr + (long long)block_idx * 96, k_in_block, &b_tile[j * 16]);
            } else {
                for (int k = 0; k < 16; ++k) b_tile[j * 16 + k] = (_Float16)0;
            }
        }
        load_matrix_sync(frag_b, b_tile, 16);
        mma_sync(frag_c, frag_a, frag_b, frag_c);
    }
    float* c_tile_ptr = C + tile_row * 16 * N + tile_col * 16;
    __shared__ float c_tile_f32[16 * 16];
    store_matrix_sync(c_tile_f32, frag_c, 16, layout_t::mem_row_major);
    __builtin_amdgcn_wave_barrier();
    for (int idx = threadIdx.x; idx < 256; idx += 32) {
        int i = idx / 16;
        int j = idx % 16;
        if (tile_row * 16 + i < M && tile_col * 16 + j < N) {
            c_tile_ptr[i * N + j] = c_tile_f32[idx];
        }
    }
}

// --- IQ3_S (110 bytes / 256 weights) ---
__device__ __forceinline__ void dequant_iq_iq3s_chunk_fp16(
    const unsigned char* block, int k_start, _Float16* out) {
    for (int i = 0; i < 16; ++i) {
        out[i] = (_Float16)dequant_iq3s(block, k_start + i);
    }
}

extern "C" __global__ void grim_wmma_fused_dequant_iq3s(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_iq,
    float* __restrict__ C,
    int M, int N, int K) {
    const int tile_row = blockIdx.y;
    const int tile_col = blockIdx.x;
    if (tile_row * 16 >= M || tile_col * 16 >= N) return;
    fragment<matrix_a, 16, 16, 16, _Float16, row_major> frag_a;
    fragment<matrix_b, 16, 16, 16, _Float16, col_major> frag_b;
    fragment<accumulator, 16, 16, 16, float> frag_c;
    fill_fragment(frag_c, 0.0f);
    const int row_base = tile_row * 16;
    const int col_base = tile_col * 16;
    const int n_blocks_per_row = K / QK_K;
    _Float16 a_tile[16 * 16];
    _Float16 b_tile[16 * 16];
    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int i = 0; i < 16; ++i) {
            int row = row_base + i;
            for (int j = 0; j < 16; ++j) {
                int kk = k0 + j;
                a_tile[i * 16 + j] = (row < M && kk < K) ? (_Float16)A[row * K + kk] : (_Float16)0;
            }
        }
        load_matrix_sync(frag_a, a_tile, 16);
        for (int j = 0; j < 16; ++j) {
            int col = col_base + j;
            if (col < N) {
                const unsigned char* row_ptr = B_iq + (long long)col * n_blocks_per_row * 110;
                int block_idx = k0 / QK_K;
                int k_in_block = k0 % QK_K;
                dequant_iq_iq3s_chunk_fp16(row_ptr + (long long)block_idx * 110, k_in_block, &b_tile[j * 16]);
            } else {
                for (int k = 0; k < 16; ++k) b_tile[j * 16 + k] = (_Float16)0;
            }
        }
        load_matrix_sync(frag_b, b_tile, 16);
        mma_sync(frag_c, frag_a, frag_b, frag_c);
    }
    float* c_tile_ptr = C + tile_row * 16 * N + tile_col * 16;
    __shared__ float c_tile_f32[16 * 16];
    store_matrix_sync(c_tile_f32, frag_c, 16, layout_t::mem_row_major);
    __builtin_amdgcn_wave_barrier();
    for (int idx = threadIdx.x; idx < 256; idx += 32) {
        int i = idx / 16;
        int j = idx % 16;
        if (tile_row * 16 + i < M && tile_col * 16 + j < N) {
            c_tile_ptr[i * N + j] = c_tile_f32[idx];
        }
    }
}

// --- IQ4_NL (170 bytes / 256 weights) ---
__device__ __forceinline__ void dequant_iq_iq4nl_chunk_fp16(
    const unsigned char* block, int k_start, _Float16* out) {
    for (int i = 0; i < 16; ++i) {
        out[i] = (_Float16)dequant_iq4nl(block, k_start + i);
    }
}

extern "C" __global__ void grim_wmma_fused_dequant_iq4nl(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_iq,
    float* __restrict__ C,
    int M, int N, int K) {
    const int tile_row = blockIdx.y;
    const int tile_col = blockIdx.x;
    if (tile_row * 16 >= M || tile_col * 16 >= N) return;
    fragment<matrix_a, 16, 16, 16, _Float16, row_major> frag_a;
    fragment<matrix_b, 16, 16, 16, _Float16, col_major> frag_b;
    fragment<accumulator, 16, 16, 16, float> frag_c;
    fill_fragment(frag_c, 0.0f);
    const int row_base = tile_row * 16;
    const int col_base = tile_col * 16;
    const int n_blocks_per_row = K / QK_K;
    _Float16 a_tile[16 * 16];
    _Float16 b_tile[16 * 16];
    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int i = 0; i < 16; ++i) {
            int row = row_base + i;
            for (int j = 0; j < 16; ++j) {
                int kk = k0 + j;
                a_tile[i * 16 + j] = (row < M && kk < K) ? (_Float16)A[row * K + kk] : (_Float16)0;
            }
        }
        load_matrix_sync(frag_a, a_tile, 16);
        for (int j = 0; j < 16; ++j) {
            int col = col_base + j;
            if (col < N) {
                const unsigned char* row_ptr = B_iq + (long long)col * n_blocks_per_row * 170;
                int block_idx = k0 / QK_K;
                int k_in_block = k0 % QK_K;
                dequant_iq_iq4nl_chunk_fp16(row_ptr + (long long)block_idx * 170, k_in_block, &b_tile[j * 16]);
            } else {
                for (int k = 0; k < 16; ++k) b_tile[j * 16 + k] = (_Float16)0;
            }
        }
        load_matrix_sync(frag_b, b_tile, 16);
        mma_sync(frag_c, frag_a, frag_b, frag_c);
    }
    float* c_tile_ptr = C + tile_row * 16 * N + tile_col * 16;
    __shared__ float c_tile_f32[16 * 16];
    store_matrix_sync(c_tile_f32, frag_c, 16, layout_t::mem_row_major);
    __builtin_amdgcn_wave_barrier();
    for (int idx = threadIdx.x; idx < 256; idx += 32) {
        int i = idx / 16;
        int j = idx % 16;
        if (tile_row * 16 + i < M && tile_col * 16 + j < N) {
            c_tile_ptr[i * N + j] = c_tile_f32[idx];
        }
    }
}

// --- IQ4_XS (136 bytes / 256 weights) ---
__device__ __forceinline__ void dequant_iq_iq4xs_chunk_fp16(
    const unsigned char* block, int k_start, _Float16* out) {
    for (int i = 0; i < 16; ++i) {
        out[i] = (_Float16)dequant_iq4xs(block, k_start + i);
    }
}

extern "C" __global__ void grim_wmma_fused_dequant_iq4xs(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_iq,
    float* __restrict__ C,
    int M, int N, int K) {
    const int tile_row = blockIdx.y;
    const int tile_col = blockIdx.x;
    if (tile_row * 16 >= M || tile_col * 16 >= N) return;
    fragment<matrix_a, 16, 16, 16, _Float16, row_major> frag_a;
    fragment<matrix_b, 16, 16, 16, _Float16, col_major> frag_b;
    fragment<accumulator, 16, 16, 16, float> frag_c;
    fill_fragment(frag_c, 0.0f);
    const int row_base = tile_row * 16;
    const int col_base = tile_col * 16;
    const int n_blocks_per_row = K / QK_K;
    _Float16 a_tile[16 * 16];
    _Float16 b_tile[16 * 16];
    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int i = 0; i < 16; ++i) {
            int row = row_base + i;
            for (int j = 0; j < 16; ++j) {
                int kk = k0 + j;
                a_tile[i * 16 + j] = (row < M && kk < K) ? (_Float16)A[row * K + kk] : (_Float16)0;
            }
        }
        load_matrix_sync(frag_a, a_tile, 16);
        for (int j = 0; j < 16; ++j) {
            int col = col_base + j;
            if (col < N) {
                const unsigned char* row_ptr = B_iq + (long long)col * n_blocks_per_row * 136;
                int block_idx = k0 / QK_K;
                int k_in_block = k0 % QK_K;
                dequant_iq_iq4xs_chunk_fp16(row_ptr + (long long)block_idx * 136, k_in_block, &b_tile[j * 16]);
            } else {
                for (int k = 0; k < 16; ++k) b_tile[j * 16 + k] = (_Float16)0;
            }
        }
        load_matrix_sync(frag_b, b_tile, 16);
        mma_sync(frag_c, frag_a, frag_b, frag_c);
    }
    float* c_tile_ptr = C + tile_row * 16 * N + tile_col * 16;
    __shared__ float c_tile_f32[16 * 16];
    store_matrix_sync(c_tile_f32, frag_c, 16, layout_t::mem_row_major);
    __builtin_amdgcn_wave_barrier();
    for (int idx = threadIdx.x; idx < 256; idx += 32) {
        int i = idx / 16;
        int j = idx % 16;
        if (tile_row * 16 + i < M && tile_col * 16 + j < N) {
            c_tile_ptr[i * N + j] = c_tile_f32[idx];
        }
    }
}

#endif // RDNA3/RDNA4
"#;

#[cfg(test)]
mod self_tests {
    use super::*;

    /// All 7 IQ-family kernel entries must be JIT-discoverable.
    #[test]
    fn source_contains_all_iq_kernel_entries() {
        for fmt in ["iq2xxs", "iq2xs", "iq2s", "iq3xxs", "iq3s", "iq4nl", "iq4xs"] {
            assert!(
                KERNEL_SOURCE.contains(&format!(
                    "grim_wmma_fused_dequant_{fmt}"
                )),
                "missing WMMA IQ kernel entry for {fmt}"
            );
        }
    }

    #[test]
    fn source_is_rdna3_rdna4_guarded() {
        assert!(
            KERNEL_SOURCE.contains("defined(__gfx1100__)")
                && KERNEL_SOURCE.contains("defined(__gfx1200__)"),
            "kernel must be guarded for RDNA3 and RDNA4 targets"
        );
    }
}
