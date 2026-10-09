//! 64x64x32 fused-dequant WMMA GEMM for Q4_K — large-M prefill.
//!
//! Why (measured, 2026-10-08): the shipped 16-row-tile WMMA path re-reads the
//! whole weight set ceil(M/16) times — 254 passes for a 4K prompt, 1.4 TB of
//! weight traffic at ~26 GB/s effective = ~47 s of GEMM. A 64-row tile cuts
//! that to 64 passes (~4x). A 128-row tile would cut it 8x but needs 16
//! accumulator fragments per thread (128 VGPRs of C alone); measured as the
//! follow-up, not the first cut.
//!
//! Structure from FlyDSL `kernels/gemm/rdna_f16_gemm.py` (written for gfx120x,
//! this card): double-buffered LDS ping-pong, `a_k_pad=8` bank-conflict
//! padding, wave-partitioned tile. Differences: 64x64 tile (not 128x128) and
//! the B-tile LDS fill is the Q4_K DEQUANT FUSED — packed bytes are read with
//! run-contiguous wide loads (the eight qs bytes of an 8-weight call are one
//! contiguous 8-byte run sharing one 6-bit scale pair, so two 4-byte loads
//! replace ~10 scalar 1-byte loads; each lane serves a different run of the
//! same column, so a warp reads 4 contiguous 32-byte runs).
//!
//! Geometry: `grid = (ceil(N/64), ceil(M/64))`, `block = 128` threads = 4
//! waves in a 2x2 layout. Wave (wm, wn) owns the 32x32 output quadrant at
//! rows [32*wm, +32) x cols [32*wn, +32): 2 M-frags x 2 N-frags x 2 K-halves
//! per 32-wide K step = 8 MMAs/wave/step, 32 per block-step. RDNA4 WMMA is
//! wave32 + 16x16x16 only (rocWMMA config.hpp static_asserts) — respected.
//!
//! Requires K % 32 == 0 (dispatch gates K % 256 == 0 for Q4_K, so every
//! 256-weight block is whole). Partial M/N tiles are masked: A rows >= M and
//! columns >= N are zero-filled, and the C store guards `row < M && col < N`.

pub const KERNEL_SOURCE: &str = r#"
#define BIG_BM 64
#define BIG_BN 64
#define BIG_BK 32
#define BIG_LD (BIG_BK + 8)

// f16 -> f32, bit-exact (standalone TU: no shared header helpers).
__device__ __forceinline__ float grim_big_f16(unsigned short v) {
    return (float)(*((_Float16 const*)&v));
}

// Vectorized Q4_K dequant. The 8 weights of one call are 8 contiguous qs bytes
// (qsb = 32*(w/64) + (w%64), a multiple of 8) sharing one 6-bit scale pair, so
// two 4-byte loads replace ~10 scalar 1-byte loads. Nibble selection is a
// register shift; multiplication order matches the scalar form.
__device__ __forceinline__ void grim_big_deq_q4k(
    const unsigned char* blk, int w, float* out)
{
    const float d = grim_big_f16(((const unsigned short*)blk)[0]);
    const float dmin = grim_big_f16(((const unsigned short*)blk)[1]);
    const unsigned char* scales = blk + 4;
    const unsigned char* qs = blk + 16;

    const int k = w / 64;
    const int off0 = w % 64;
    const int s = 2 * k + (off0 >= 32 ? 1 : 0);
    const int qsb = 32 * k + (off0 & 31);
    unsigned char sc, m;
    if (s < 4) { sc = scales[s] & 63; m = scales[s + 4] & 63; }
    else {
        sc = (unsigned char)((scales[s + 4] & 0x0F) | ((scales[s - 4] >> 6) << 4));
        m = (unsigned char)((scales[s + 4] >> 4) | ((scales[s] >> 6) << 4));
    }
    const float ds = d * (float)sc;
    const float dm = dmin * (float)m;
    const bool low = off0 < 32;

    unsigned int qw_lo, qw_hi;
    __builtin_memcpy(&qw_lo, qs + qsb, 4);
    __builtin_memcpy(&qw_hi, qs + qsb + 4, 4);
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        const unsigned int byte = (e < 4) ? (qw_lo >> (8 * e)) : (qw_hi >> (8 * (e - 4)));
        const unsigned int q = low ? (byte & 0x0Fu) : (byte >> 4);
        out[e] = ds * (float)q - dm;
    }
}

// Large-M fused-dequant GEMM: C[M,N] = A[M,K] @ dequant(B_q[N,K])^T, Q4_K.

// Fill one 32-wide K slice of both LDS tiles. Explicit parameters, no captures.
__device__ __forceinline__ void grim_big_fill_stage(
    _Float16* sA, _Float16* sB,
    const float* __restrict__ A, const unsigned char* __restrict__ B_q,
    int tid, int row_base, int col_base, int k0,
    int M, int N, int K, long long row_bytes)
{
    const int w_base = (k0 / 32) % 8 * 32;
    for (int idx = tid; idx < BIG_BM * BIG_BK; idx += 128) {
        const int rr = idx / BIG_BK;
        const int kk = idx % BIG_BK;
        const int row = row_base + rr;
        const int kg = k0 + kk;
        sA[rr * BIG_LD + kk] = (row < M && kg < K)
            ? (_Float16)A[(long long)row * K + kg] : (_Float16)0;
    }
    for (int idx = tid; idx < BIG_BN * (BIG_BK / 8); idx += 128) {
        const int cc = idx / (BIG_BK / 8);
        const int rr = idx % (BIG_BK / 8);
        const int col = col_base + cc;
        const int w_in = w_base + rr * 8;
        float acc[8];
        if (col < N) {
            const unsigned char* blkptr = B_q
                + (long long)col * row_bytes + (long long)(k0 / 256) * 144;
            grim_big_deq_q4k(blkptr, w_in, acc);
        } else {
            #pragma unroll
            for (int e = 0; e < 8; ++e) acc[e] = 0.0f;
        }
        _Float16* dst = sB + cc * BIG_LD + rr * 8;
        #pragma unroll
        for (int e = 0; e < 8; ++e) dst[e] = (_Float16)acc[e];
    }
}

extern "C" __global__ void grim_wmma_big_q4k(
    const float* __restrict__ A, const unsigned char* __restrict__ B_q,
    float* __restrict__ C, int M, int N, int K)
{
    const int col_base = blockIdx.x * BIG_BN;
    const int row_base = blockIdx.y * BIG_BM;
    const int tid = threadIdx.x;          // 0..127, 4 waves
    const int wave = tid >> 5;            // 0..3
    const int wm = wave >> 1;             // row half 0..1
    const int wn = wave & 1;              // col half 0..1

    const int n_blocks = K / 256;
    const long long row_bytes = (long long)n_blocks * 144;

    __shared__ _Float16 sA[2][BIG_BM * BIG_LD];   // 2 x 5 KB
    __shared__ _Float16 sB[2][BIG_BN * BIG_LD];   // 2 x 5 KB
    __shared__ float cq[4][32 * 32];              // per-wave quadrant, 4 KB

    using namespace rocwmma;
    fragment<matrix_a, 16, 16, 16, _Float16, row_major> fa[2][2];   // [mi][kh]
    fragment<matrix_b, 16, 16, 16, _Float16, col_major> fb[2][2];   // [ni][kh]
    fragment<accumulator, 16, 16, 16, float> fc[2][2];              // [mi][ni]
    #pragma unroll
    for (int mi = 0; mi < 2; ++mi)
        #pragma unroll
        for (int ni = 0; ni < 2; ++ni)
            fill_fragment(fc[mi][ni], 0.0f);

    grim_big_fill_stage(sA[0], sB[0], A, B_q, tid, row_base, col_base,
                        0, M, N, K, row_bytes);
    __syncthreads();

    const int n_k_steps = K / BIG_BK;
    for (int it = 0; it < n_k_steps; ++it) {
        const int buf = it & 1;
        // Prefetch the next 32-wide K slice into the other stage: the dequant's
        // global loads overlap this step's MMAs.
        if (it + 1 < n_k_steps)
            grim_big_fill_stage(sA[buf ^ 1], sB[buf ^ 1], A, B_q, tid,
                                row_base, col_base, (it + 1) * BIG_BK,
                                M, N, K, row_bytes);
        #pragma unroll
        for (int kh = 0; kh < 2; ++kh) {
            #pragma unroll
            for (int mi = 0; mi < 2; ++mi)
                load_matrix_sync(fa[mi][kh],
                                 sA[buf] + (wm * 32 + mi * 16) * BIG_LD + kh * 16, BIG_LD);
            #pragma unroll
            for (int ni = 0; ni < 2; ++ni)
                load_matrix_sync(fb[ni][kh],
                                 sB[buf] + (wn * 32 + ni * 16) * BIG_LD + kh * 16, BIG_LD);
            #pragma unroll
            for (int mi = 0; mi < 2; ++mi)
                #pragma unroll
                for (int ni = 0; ni < 2; ++ni)
                    mma_sync(fc[mi][ni], fa[mi][kh], fb[ni][kh], fc[mi][ni]);
        }
        // Orders this step's prefetch writes against the next iteration's reads
        // of that stage (last read two iterations ago: no hazard on the stage
        // written now).
        __syncthreads();
    }

    // Per-wave 32x32 quadrant to LDS, then one masked cooperative global write.
    #pragma unroll
    for (int mi = 0; mi < 2; ++mi)
        #pragma unroll
        for (int ni = 0; ni < 2; ++ni)
            store_matrix_sync(cq[wave] + (mi * 16) * 32 + ni * 16, fc[mi][ni], 32,
                              layout_t::mem_row_major);
    __syncthreads();
    // Per-WAVE lane iteration: each wave writes ALL 1024 slots of its own
    // 32x32 quadrant. `idx = tid + k*128` would cover only the quarter of the
    // quadrant whose slots are congruent to the wave's lanes (mod 128) — the
    // other 3/4 never written and read back as fresh-LDS zeros.
    const int lane = tid & 31;
    for (int idx = lane; idx < 32 * 32; idx += 32) {
        const int i = idx / 32;
        const int j = idx % 32;
        const int row = row_base + wm * 32 + i;
        const int col = col_base + wn * 32 + j;
        if (row < M && col < N)
            C[(long long)row * N + col] = cq[wave][i * 32 + j];
    }
}
"#;
