//! res0-T1-3: INT8 WMMA probe (RDNA4, rocwmma int8 fragments).
//!
//! Why: grim's prefill dequantizes K-weights to f16 and runs f16 WMMA; RDNA4's
//! I8 matrix-core rate is 2048 FLOP/clk/CU = 2x the FP16 rate (GPUOpen matrix
//! cores table), and INTEGER accumulation is exactly associative, so any
//! K-split order gives the same i32 result — the bit-identity gate is free.
//!
//! Skeleton copied from `wmma_fp8_gemm.rs` (the proven rocwmma-on-gfx12 path:
//! `fragment`/`load_matrix_sync`/`mma_sync`, two 16-wide N tiles per block).
//! rocwmma on gfx12 rejects the signed x unsigned mix (the SOA_INT register
//! layouts of int8_t and uint8_t differ — measured: static-assert
//! "Input fragment register layouts do not match"), so BOTH operands are
//! int8_t. Grim's unsigned weight codes map to signed at dequant-once time
//! via (code - 8) with the -8*sum(a) correction folded into the per-256-block
//! scale epilogue — the same two-dot decomposition the decode dot4 kernels
//! already use for Q2_K's dmin term.
//!
//! Gate: D (i32) must be BIT-IDENTICAL to the host reference — enforced by
//! tests/wmma_i8_probe.rs. NOT wired into the dispatch; probe launcher only.

pub const WMMA_I8_KERNEL_SOURCE: &str = r#"
#if defined(__gfx1200__) || defined(__gfx1201__)
#include <rocwmma/rocwmma.hpp>
using namespace rocwmma;

// INT8 x UINT8 -> INT32, 16x32 output tile per wavefront (two 16-wide N
// tiles), same structure as grim_wmma_gemm_fp8_e4m3.
//   A [M, K] int8, row-major (Q8_1-style activation codes)
//   B [N, K] uint8, row-major (unsigned weight codes; B^T is the GEMM's B)
//   D [M, N] int32, tightly packed
extern "C" __global__ __launch_bounds__(32) void grim_wmma_gemm_iu8(
    const int8_t* __restrict__ A,   // [M, K] row-major
    const int8_t* __restrict__ B,   // [N, K] row-major (both operands signed:
    int* __restrict__ D,            //  rocwmma gfx12 rejects int8 x uint8)
    int M, int N, int K) {

    const int tile_row = blockIdx.y;
    const int tile_col_base = blockIdx.x * 2;

    if (tile_row * 16 >= M || tile_col_base * 16 >= N) return;

    const int row_base = tile_row * 16;
    const int col_base = tile_col_base * 16;
    const bool valid_col1 = ((tile_col_base + 1) * 16 < N);

    fragment<matrix_a, 16, 16, 16, int8_t, row_major> frag_a;
    fragment<matrix_b, 16, 16, 16, int8_t, col_major> frag_b0;
    fragment<matrix_b, 16, 16, 16, int8_t, col_major> frag_b1;
    fragment<accumulator, 16, 16, 16, int32_t> frag_c0;
    fragment<accumulator, 16, 16, 16, int32_t> frag_c1;
    fill_fragment(frag_c0, 0);
    fill_fragment(frag_c1, 0);

    __shared__ int d_out[16 * 16];

    for (int k0 = 0; k0 < K; k0 += 16) {
        load_matrix_sync(frag_a, A + (long long)row_base * K + k0, K);
        load_matrix_sync(frag_b0, B + (long long)col_base * K + k0, K);
        mma_sync(frag_c0, frag_a, frag_b0, frag_c0);
        if (valid_col1) {
            load_matrix_sync(frag_b1, B + (long long)(col_base + 16) * K + k0, K);
            mma_sync(frag_c1, frag_a, frag_b1, frag_c1);
        }
    }

    int* d0_ptr = D + (long long)row_base * N + col_base;
    store_matrix_sync(d_out, frag_c0, 16, layout_t::mem_row_major);
    __builtin_amdgcn_wave_barrier();
    for (int idx = threadIdx.x; idx < 256; idx += 32) {
        int i = idx / 16;
        int j = idx % 16;
        if (row_base + i < M && col_base + j < N) d0_ptr[(long long)i * N + j] = d_out[idx];
    }

    if (valid_col1) {
        int* d1_ptr = D + (long long)row_base * N + (col_base + 16);
        store_matrix_sync(d_out, frag_c1, 16, layout_t::mem_row_major);
        __builtin_amdgcn_wave_barrier();
        for (int idx = threadIdx.x; idx < 256; idx += 32) {
            int i = idx / 16;
            int j = idx % 16;
            if (row_base + i < M && col_base + 16 + j < N)
                d1_ptr[(long long)i * N + j] = d_out[idx];
        }
    }
}

// MTILE variant: one block computes M_TILES(4) x 16 = 64 rows x 32 cols,
// loading each B fragment ONCE per k-step and reusing it across all 4 A
// fragments — the transpose of the fp8 kernel's A-reuse. The naive tile's
// measured failure was B re-read traffic ((M/16) x N x K bytes); this
// divides it by M_TILES.
extern "C" __global__ __launch_bounds__(32) void grim_wmma_gemm_iu8_mtile(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int* __restrict__ D,
    int M, int N, int K) {

    const int MT = 4;
    const int tile_m_base = blockIdx.y * (MT * 16);
    const int tile_col_base = blockIdx.x * 2;

    if (tile_m_base >= M || tile_col_base * 16 >= N) return;

    const int col_base = tile_col_base * 16;
    const bool valid_col1 = ((tile_col_base + 1) * 16 < N);

    fragment<matrix_a, 16, 16, 16, int8_t, row_major> frag_a[MT];
    fragment<matrix_b, 16, 16, 16, int8_t, col_major> frag_b0;
    fragment<matrix_b, 16, 16, 16, int8_t, col_major> frag_b1;
    fragment<accumulator, 16, 16, 16, int32_t> frag_c0[MT];
    fragment<accumulator, 16, 16, 16, int32_t> frag_c1[MT];
    #pragma unroll
    for (int t = 0; t < MT; ++t) {
        fill_fragment(frag_c0[t], 0);
        fill_fragment(frag_c1[t], 0);
    }

    __shared__ int d_out[16 * 16];

    for (int k0 = 0; k0 < K; k0 += 16) {
        #pragma unroll
        for (int t = 0; t < MT; ++t) {
            const int row_base = tile_m_base + t * 16;
            if (row_base < M) load_matrix_sync(frag_a[t], A + (long long)row_base * K + k0, K);
        }
        load_matrix_sync(frag_b0, B + (long long)col_base * K + k0, K);
        #pragma unroll
        for (int t = 0; t < MT; ++t) {
            const int row_base = tile_m_base + t * 16;
            if (row_base < M) mma_sync(frag_c0[t], frag_a[t], frag_b0, frag_c0[t]);
        }
        if (valid_col1) {
            load_matrix_sync(frag_b1, B + (long long)(col_base + 16) * K + k0, K);
            #pragma unroll
            for (int t = 0; t < MT; ++t) {
                const int row_base = tile_m_base + t * 16;
                if (row_base < M) mma_sync(frag_c1[t], frag_a[t], frag_b1, frag_c1[t]);
            }
        }
    }

    #pragma unroll
    for (int t = 0; t < MT; ++t) {
        const int row_base = tile_m_base + t * 16;
        if (row_base >= M) continue;
        int* d0_ptr = D + (long long)row_base * N + col_base;
        store_matrix_sync(d_out, frag_c0[t], 16, layout_t::mem_row_major);
        __builtin_amdgcn_wave_barrier();
        for (int idx = threadIdx.x; idx < 256; idx += 32) {
            int i = idx / 16;
            int j = idx % 16;
            if (col_base + j < N) d0_ptr[(long long)i * N + j] = d_out[idx];
        }
        if (valid_col1) {
            int* d1_ptr = D + (long long)row_base * N + (col_base + 16);
            store_matrix_sync(d_out, frag_c1[t], 16, layout_t::mem_row_major);
            __builtin_amdgcn_wave_barrier();
            for (int idx = threadIdx.x; idx < 256; idx += 32) {
                int i = idx / 16;
                int j = idx % 16;
                if (col_base + 16 + j < N) d1_ptr[(long long)i * N + j] = d_out[idx];
            }
        }
    }
}

typedef int grim_v8i __attribute__((ext_vector_type(8)));
typedef int grim_v2i __attribute__((ext_vector_type(2)));

// Direct RDNA4 AMDGCN iu8 WMMA (signed int8 A x unsigned uint4*16 B -> int32 C).
// Bypasses rocwmma's SOA_INT static assert on mixed signedness and unpacks 8 4-bit
// weights per u32 in 3 ALU instructions: lo = (w << 4) & 0xF0F0F0F0, hi = w & 0xF0F0F0F0.
// Divide accumulator by 16 at epilogue to remove the high-nibble 16x shift factor.
extern "C" __global__ __launch_bounds__(32) void grim_wmma_w4a8_iu8_gfx12(
    const signed char* __restrict__ A_frag,   // pre-packed [M/16, K/16, 32 lanes, 8 i8]
    const unsigned int* __restrict__ B_u4,    // pre-packed [N/16, K/16, 32 lanes, 1 u32 (8 nibbles)]
    int* __restrict__ D,                      // [M, N] row-major int32 (unscaled dot, /16 applied)
    int M, int N, int K)
{
    const int tile_m = blockIdx.y;
    const int tile_n = blockIdx.x;
    const int lane = threadIdx.x & 31;
    if (tile_m * 16 >= M || tile_n * 16 >= N) return;

    const int k_steps = K / 16;
    grim_v8i acc = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int ks = 0; ks < k_steps; ++ks) {
        const int* a_ptr = (const int*)(A_frag + (((long long)tile_m * k_steps + ks) * 32 + lane) * 8);
        grim_v2i a_reg = {a_ptr[0], a_ptr[1]};

        unsigned int w = B_u4[((long long)tile_n * k_steps + ks) * 32 + lane];
        grim_v2i b_reg;
        b_reg[0] = (int)((w << 4) & 0xF0F0F0F0u);
        b_reg[1] = (int)(w & 0xF0F0F0F0u);

        acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
            true, a_reg, false, b_reg, acc, false);
    }

    const int col = tile_n * 16 + (lane & 15);
    const int row_half = (lane >> 4) * 4;
    if (col < N) {
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int row = tile_m * 16 + ((e >> 2) * 8) + row_half + (e & 3);
            if (row < M) {
                D[(long long)row * N + col] = acc[e] >> 4;
            }
        }
    }
}

// Phase 1C: INT8 WMMA Double-K (MMA_K=32) GEMM for Q8_1 activations x Q8_0 weights on RDNA4.
// Each 64x64 CTA (4 waves = 2x2 warps, each warp 2x2 16x16 WMMA tiles) steps K by 32
// (exact 1:1 match with Q8_1 and Q8_0 32-element blocks), runs 2x wmma_i32_16x16x16_iu8_w32_gfx12
// per block with exact i32 inner-block accumulation, then scales by d_a * d_b into F32.
extern "C" __global__ __launch_bounds__(128) void grim_wmma_i8_q80_q81_doublek_gfx12(
    const unsigned char* __restrict__ A_q81, // [M, (K/32)*36] Q8_1 activations
    const unsigned char* __restrict__ B_q80, // [N, (K/32)*34] Q8_0 weights (transposed: N rows of K)
    float* __restrict__ C,                   // [M, N] row-major F32
    int M, int N, int K)
{
    const int cta_m = blockIdx.y * 64;
    const int cta_n = blockIdx.x * 64;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int lane_r = lane & 15;
    const int lane_h = lane >> 4;
    const int wm = (wave >> 1) * 32; // 0 or 32
    const int wn = (wave & 1) * 32;  // 0 or 32

    const int n_blocks = K >> 5; // K / 32
    const long long a_row_bytes = (long long)n_blocks * 36;
    const long long b_row_bytes = (long long)n_blocks * 34;

    __shared__ signed char sA[64][32];
    __shared__ signed char sB[64][32];
    __shared__ float sDa[64];
    __shared__ float sDb[64];

    float f_acc[2][2][8];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            #pragma unroll
            for (int e = 0; e < 8; ++e)
                f_acc[mt][nt][e] = 0.0f;

    for (int kb = 0; kb < n_blocks; ++kb) {
        // Cooperative load of 64 rows x 32 bytes (128 threads -> 16 bytes = uint4 per thread!)
        const int r_idx = tid >> 1;      // 0..63
        const int half_idx = tid & 1;    // 0 or 1 (16 bytes each)
        const int gm = cta_m + r_idx;
        const int gn = cta_n + r_idx;

        if (gm < M) {
            const unsigned char* ap = A_q81 + (long long)gm * a_row_bytes + (long long)kb * 36;
            if (half_idx == 0) {
                sDa[r_idx] = __half2float(*reinterpret_cast<const __half*>(ap));
            }
            // Copy 16 bytes (unaligned-safe via 4x u32)
            const unsigned char* qs = ap + 4 + half_idx * 16;
            #pragma unroll
            for (int b = 0; b < 16; ++b) {
                sA[r_idx][half_idx * 16 + b] = (signed char)qs[b];
            }
        } else {
            if (half_idx == 0) sDa[r_idx] = 0.0f;
            #pragma unroll
            for (int b = 0; b < 16; ++b) sA[r_idx][half_idx * 16 + b] = 0;
        }

        if (gn < N) {
            const unsigned char* bp = B_q80 + (long long)gn * b_row_bytes + (long long)kb * 34;
            if (half_idx == 0) {
                sDb[r_idx] = __half2float(*reinterpret_cast<const __half*>(bp));
            }
            const unsigned char* qs = bp + 2 + half_idx * 16;
            #pragma unroll
            for (int b = 0; b < 16; ++b) {
                sB[r_idx][half_idx * 16 + b] = (signed char)qs[b];
            }
        } else {
            if (half_idx == 0) sDb[r_idx] = 0.0f;
            #pragma unroll
            for (int b = 0; b < 16; ++b) sB[r_idx][half_idx * 16 + b] = 0;
        }
        __syncthreads();

        // Double-K (2 x 16 = 32) INT8 WMMA per 16x16 tile
        #pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            const int m_off = wm + mt * 16 + lane_r;
            const int* a0_p = reinterpret_cast<const int*>(&sA[m_off][lane_h * 8]);
            const int* a1_p = reinterpret_cast<const int*>(&sA[m_off][16 + lane_h * 8]);
            grim_v2i a0 = {a0_p[0], a0_p[1]};
            grim_v2i a1 = {a1_p[0], a1_p[1]};

            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int n_off = wn + nt * 16 + lane_r;
                const int* b0_p = reinterpret_cast<const int*>(&sB[n_off][lane_h * 8]);
                const int* b1_p = reinterpret_cast<const int*>(&sB[n_off][16 + lane_h * 8]);
                grim_v2i b0 = {b0_p[0], b0_p[1]};
                grim_v2i b1 = {b1_p[0], b1_p[1]};

                grim_v8i i_acc = {0, 0, 0, 0, 0, 0, 0, 0};
                i_acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a0, true, b0, i_acc, false);
                i_acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a1, true, b1, i_acc, false);

                const float db = sDb[wn + nt * 16 + lane_r];
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int local_r = wm + mt * 16 + lane_h * 8 + e;
                    f_acc[mt][nt][e] += (float)i_acc[e] * (sDa[local_r] * db);
                }
            }
        }
        __syncthreads();
    }

    // Store C[M, N]
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int col = cta_n + wn + nt * 16 + lane_r;
            if (col < N) {
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int row = cta_m + wm + mt * 16 + lane_h * 8 + e;
                    if (row < M) {
                        C[(long long)row * N + col] = f_acc[mt][nt][e];
                    }
                }
            }
        }
    }
}

// res12 Step 2 + HipKittens 8-wave ping-pong: Exact W4A8 (Q4_K weights x Q8_1 activations)
// using RDNA4 native signed-i8 x unsigned-u8 WMMA (wmma_i32_16x16x16_iu8_w32_gfx12).
// 8 waves (256 threads, 4x2 warp grid -> 128x64 CTA tile), double-buffered LDS (15 KB),
// 1 barrier per K=32 sub-block, exact i32 inner-block accumulation + dmin*s_a term.
__device__ __forceinline__ void q4k_get_scale_min(
    int sb, const unsigned char* __restrict__ sc, unsigned char& d_sc, unsigned char& m_sc)
{
    if (sb < 4) {
        d_sc = sc[sb] & 63u;
        m_sc = sc[sb + 4] & 63u;
    } else {
        d_sc = (sc[sb + 4] & 0x0Fu) | ((sc[sb - 4] >> 6) << 4);
        m_sc = (sc[sb + 4] >> 4)    | ((sc[sb]     >> 6) << 4);
    }
}

extern "C" __global__ __launch_bounds__(256) void grim_wmma_w4a8_q4k_q81_pingpong_gfx12(
    const unsigned char* __restrict__ A_q81, // [M, (K/32)*36] Q8_1 activations
    const unsigned char* __restrict__ B_q4k, // [N, (K/256)*144] Q4_K weights (N rows of K)
    float* __restrict__ C,                   // [M, N] row-major F32
    int M, int N, int K)
{
    const int cta_m = blockIdx.y * 128;
    const int cta_n = blockIdx.x * 64;
    const int tid = threadIdx.x; // 0..255 (8 waves)
    const int wave = tid >> 5;   // 0..7
    const int lane = tid & 31;
    const int lane_r = lane & 15;
    const int lane_h = lane >> 4;
    const int wm = (wave >> 1) * 32; // 0, 32, 64, 96 (4 warp rows)
    const int wn = (wave & 1) * 32;  // 0, 32 (2 warp cols)

    const int n_blocks = K >> 5;      // K / 32
    const int n_super = K >> 8;       // K / 256
    const long long a_row_bytes = (long long)n_blocks * 36;
    const long long b_row_bytes = (long long)n_super * 144;

    // Double-buffered LDS: 15 KB total
    __shared__ signed char   sA[2][128][32];
    __shared__ float         sDa[2][128];
    __shared__ float         sSa[2][128];
    __shared__ unsigned char sB[2][64][32];
    __shared__ float         sDb[2][64];
    __shared__ float         sMb[2][64];

    float f_acc[2][2][8];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            #pragma unroll
            for (int e = 0; e < 8; ++e)
                f_acc[mt][nt][e] = 0.0f;

    auto load_tile = [&](int kb, int buf) {
        // 1. Load 128 rows of Q8_1 (256 threads -> 2 threads/row, 16 bytes via 4x u32 + uint4 LDS store)
        const int a_row = tid >> 1;
        const int a_half = tid & 1;
        const int gm = cta_m + a_row;
        if (gm < M) {
            const unsigned char* ap = A_q81 + (long long)gm * a_row_bytes + (long long)kb * 36;
            if (a_half == 0) {
                const unsigned int ds = *reinterpret_cast<const unsigned int*>(ap);
                sDa[buf][a_row] = __half2float(*reinterpret_cast<const __half*>(&ds));
                sSa[buf][a_row] = __half2float(*(reinterpret_cast<const __half*>(&ds) + 1));
            }
            const unsigned int* qsp = reinterpret_cast<const unsigned int*>(ap + 4 + a_half * 16);
            uint4 qv = make_uint4(qsp[0], qsp[1], qsp[2], qsp[3]);
            *reinterpret_cast<uint4*>(&sA[buf][a_row][a_half * 16]) = qv;
        } else {
            if (a_half == 0) {
                sDa[buf][a_row] = 0.0f;
                sSa[buf][a_row] = 0.0f;
            }
            *reinterpret_cast<uint4*>(&sA[buf][a_row][a_half * 16]) = make_uint4(0u, 0u, 0u, 0u);
        }

        // 2. Load & unpack 64 rows of Q4_K sub-block `sb = kb & 7` (256 threads -> 4 threads/row, uint2 load + SWAR nibble unpack)
        const int b_row = tid >> 2;
        const int b_qtr = tid & 3; // 0..3 -> 8 elements each
        const int gn = cta_n + b_row;
        const int super_idx = kb >> 3;
        const int sb = kb & 7;
        const int grp = sb >> 1;   // 0..3 (each group = 32 packed bytes = 64 nibbles)
        const int is_hi = sb & 1;  // 0 = low nibble, 1 = high nibble
        if (gn < N) {
            const unsigned char* bp = B_q4k + (long long)gn * b_row_bytes + (long long)super_idx * 144;
            if (b_qtr == 0) {
                const unsigned int dd = *reinterpret_cast<const unsigned int*>(bp);
                const float d = __half2float(*reinterpret_cast<const __half*>(&dd));
                const float dmin = __half2float(*(reinterpret_cast<const __half*>(&dd) + 1));
                unsigned char sc_u, m_u;
                q4k_get_scale_min(sb, bp + 4, sc_u, m_u);
                sDb[buf][b_row] = d * (float)sc_u;
                sMb[buf][b_row] = dmin * (float)m_u;
            }
            const uint2 packed = *reinterpret_cast<const uint2*>(bp + 16 + grp * 32 + b_qtr * 8);
            const unsigned int u0 = is_hi ? ((packed.x >> 4) & 0x0F0F0F0Fu) : (packed.x & 0x0F0F0F0Fu);
            const unsigned int u1 = is_hi ? ((packed.y >> 4) & 0x0F0F0F0Fu) : (packed.y & 0x0F0F0F0Fu);
            *reinterpret_cast<uint2*>(&sB[buf][b_row][b_qtr * 8]) = make_uint2(u0, u1);
        } else {
            if (b_qtr == 0) {
                sDb[buf][b_row] = 0.0f;
                sMb[buf][b_row] = 0.0f;
            }
            *reinterpret_cast<uint2*>(&sB[buf][b_row][b_qtr * 8]) = make_uint2(0u, 0u);
        }
    };

    if (n_blocks > 0) {
        load_tile(0, 0);
        __syncthreads();
    }

    for (int kb = 0; kb < n_blocks; ++kb) {
        const int cur = kb & 1;
        const int nxt = cur ^ 1;

        // Prefetch next tile into ping-pong buffer BEFORE WMMA so global loads overlap compute
        if (kb + 1 < n_blocks) {
            load_tile(kb + 1, nxt);
        }

        // Compute current tile from sA[cur] / sB[cur]
        #pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            const int m_off = wm + mt * 16 + lane_r;
            const int* a0_p = reinterpret_cast<const int*>(&sA[cur][m_off][lane_h * 8]);
            const int* a1_p = reinterpret_cast<const int*>(&sA[cur][m_off][16 + lane_h * 8]);
            grim_v2i a0 = {a0_p[0], a0_p[1]};
            grim_v2i a1 = {a1_p[0], a1_p[1]};

            float da_r[8], sa_r[8];
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int local_r = wm + mt * 16 + lane_h * 8 + e;
                da_r[e] = sDa[cur][local_r];
                sa_r[e] = sSa[cur][local_r];
            }

            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int n_off = wn + nt * 16 + lane_r;
                const int* b0_p = reinterpret_cast<const int*>(&sB[cur][n_off][lane_h * 8]);
                const int* b1_p = reinterpret_cast<const int*>(&sB[cur][n_off][16 + lane_h * 8]);
                grim_v2i b0 = {b0_p[0], b0_p[1]};
                grim_v2i b1 = {b1_p[0], b1_p[1]};

                grim_v8i i_acc = {0, 0, 0, 0, 0, 0, 0, 0};
                // Signed i8 A x Unsigned u8 (4-bit) B -> exact i32
                i_acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a0, false, b0, i_acc, false);
                i_acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a1, false, b1, i_acc, false);

                const float db = sDb[cur][n_off];
                const float mb = sMb[cur][n_off];
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    f_acc[mt][nt][e] += (float)i_acc[e] * (da_r[e] * db) - sa_r[e] * mb;
                }
            }
        }

        __syncthreads();
    }

    // Store C[M, N]
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int col = cta_n + wn + nt * 16 + lane_r;
            if (col < N) {
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int row = cta_m + wm + mt * 16 + lane_h * 8 + e;
                    if (row < M) {
                        C[(long long)row * N + col] = f_acc[mt][nt][e];
                    }
                }
            }
        }
    }
}
#endif
"#;

#[cfg(test)]
mod tests {
    #[test]
    fn wmma_i8_source_contains_entries() {
        assert!(super::WMMA_I8_KERNEL_SOURCE.contains("grim_wmma_gemm_iu8"));
        assert!(super::WMMA_I8_KERNEL_SOURCE.contains("rocwmma"));
    }
}
