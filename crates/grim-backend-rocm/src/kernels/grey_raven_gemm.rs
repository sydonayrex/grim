//! GreyRaven production GEMM: 2:4-sparse FP8 SWMMAC over tiled HW-order weights.
//!
//! Computes Y[M,N] = X[M,K] @ W[N,K]^T with W in [`BlockDtype::Fp8Sparse24Hw`]
//! hardware order (per-(16-row N-tile, 32-col K-window) 256 B A-fragment in
//! lane order + u32 sidx) and X dense F32 row-major.
//!
//! One wave (32 threads) computes one 16x16 output tile, looping over
//! K-windows and accumulating the 8-f32 C fragment in registers. Each
//! K-window runs ONE wave-wide `V_SWMMAC_F32_16X16X32_FP8_FP8` with:
//!   A: lane l's 8 bytes from the payload fragment (row l%16, half l>=16,
//!      verified fragment order -- see grey_raven_verify).
//!   B: lane l's 16 bytes gathered transposed from X: byte j = E4M3 of
//!      X[Mtile*16 + (l%16)][Kbase + 16*(l>=16) + j] (verified order).
//!   sidx: the window's word from the payload (verified (fa,fb) rule).
//!   C: running accumulator, written once as
//!      Y[Mtile*16 + (l%16)][Ntile*16 + 8*(l>=16) + s] (verified layout).
//!
//! Tails are zero-padded, never branched around: M-tail X reads and C writes
//! are bounds-checked, K-tail windows pad A (payload zero-padded at pack)
//! and B (gather pads), N-tail tiles read zero-padded A rows. Zero inputs
//! contribute exactly 0 through the integer dot product, so padding is exact.
//!
//! Graph-capture safe (caller-owned buffers only, single launch, no scratch).
//!
//! Measured (RX 9070 XT, gfx1201, n=k=4096, `grey_raven_dispatch_bench`):
//!   m=1 : 782.9 us,  21.4 GB/s -- 0.07x vs Raven dot4 GEMV (56.3 us).
//!         Structurally disadvantaged: 15/16 activation columns pad to zero
//!         and every N-tile re-gathers/re-encodes the same B window.
//!   m=16: 802.3 us,  20.9 GB/s -- 17.1x vs Raven fused-dequant MFMA
//!         (13745.6 us; that arm looks independently pathological --
//!         orders of magnitude slower than the arithmetic justifies).
//! Absolute numbers are far from HBM limits in both cases: the B
//! transpose-gather + E4M3 encode runs per (N-tile, window) but depends only
//! on (M-tile, window) -- 256x redundant on square shapes. A B-prologue
//! kernel (WhiteRaven act-prologue playbook: quantize once per (M-tile,
//! window) into fragment order) is the documented next step, not a
//! correctness issue: parity is bit-exact (see grey_raven_dispatch).
//!
//! Source placement: aggregated AFTER quant_standalone (uses
//! `float_to_fp8_e4m3_hip`) and shared_device_fns. Vector typedefs are
//! file-local (`gg_` prefix) to avoid colliding with the probe's.

use grim_tensor::dtype::BlockDtype;

pub const GEMM_SOURCE: &str = r#"
#if defined(__gfx1200__) || defined(__gfx1201__)

typedef int   gg_v2i __attribute__((vector_size(8)));
typedef int   gg_v4i __attribute__((vector_size(16)));
typedef float gg_v8f __attribute__((vector_size(32)));

extern "C" __global__ void grim_grey_raven_gemm(
    const unsigned char* __restrict__ W,  // HW payload: per (N-tile, window) [256 B frag][u32 sidx]
    const float*         __restrict__ X,  // dense activations [M, K] row-major
    float*               __restrict__ Y,  // output [M, N] row-major
    int M, int N, int K,
    int n_tiles,      // ceil(N / 16): N-tiles in the payload
    int n_windows)    // ceil(K / 32): K-windows in the payload
{
    const int tileN = blockIdx.x;   // 16-row N tile
    const int tileM = blockIdx.y;   // 16-row M tile
    const int lane  = threadIdx.x;  // 0..31, exactly one wave per block
    if (lane >= 32) return;

    const int half  = (lane >= 16) ? 1 : 0;
    const int lmod  = lane & 15;

    // Running C fragment: Y[8*half + s][lmod] within the tile.
    float cacc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int w = 0; w < n_windows; w++) {
        const unsigned char* slot =
            W + ((long long)tileN * n_windows + w) * 260;

        // A piece: lane l owns bytes [l*8, l*8+8) of the frag.
        gg_v2i a_frag;
        __builtin_memcpy(&a_frag, slot + lane * 8, 8);
        unsigned int sidx;
        __builtin_memcpy(&sidx, slot + 256, 4);

        // B piece: 16 bytes gathered transposed. Byte j = E4M3 of
        // X[Mtile*16 + lmod][w*32 + 16*half + j]; out-of-bounds reads
        // (M/K tails) encode as +0.0, contributing exactly 0.
        const int m = tileM * 16 + lmod;
        const int kbase = w * 32 + half * 16;
        unsigned int words[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            unsigned int word = 0u;
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const int jj = j * 4 + e;
                const int kk = kbase + jj;
                float v = 0.0f;
                if (m < M && kk < K)
                    v = X[(long long)m * K + kk];
                unsigned char code = float_to_fp8_e4m3_hip(v);
                word |= ((unsigned int)code) << (8 * e);
            }
            words[j] = word;
        }
        gg_v4i b_frag;
        __builtin_memcpy(&b_frag, words, 16);

        gg_v8f c_in;
        __builtin_memcpy(&c_in, cacc, 32);
        gg_v8f d = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(
            a_frag, b_frag, c_in, sidx);
        __builtin_memcpy(cacc, &d, 32);
    }

    // Write Y[Mtile*16 + lmod][Ntile*16 + 8*half + s], bounds-checked
    // (M/N tails). Lanes l, l+16 write the same Y row's two col-octaves.
    #pragma unroll
    for (int s = 0; s < 8; s++) {
        const int m = tileM * 16 + lmod;
        const int n = tileN * 16 + half * 8 + s;
        if (m < M && n < N)
            Y[(long long)m * N + n] = cacc[s];
    }
}
#endif // __gfx1200__ || __gfx1201__
"#;

/// Decode helper shared with tests: bytes of one sidx word for the given
/// per-group patterns (ascending slots), pair p at bits [4p+3:4p].
pub fn sidx_word_for_patterns(pats: &[[u8; 2]; 4]) -> u32 {
    let mut w = 0u32;
    for (p, pat) in pats.iter().enumerate() {
        let (s0, s1) = (pat[0] as u32, pat[1] as u32);
        w |= (s0 | (s1 << 2)) << (4 * p);
    }
    w
}

#[allow(dead_code)]
fn _storage_anchor() {
    let _ = BlockDtype::Fp8Sparse24Hw;
}
