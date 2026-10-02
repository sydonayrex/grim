//! **WhiteRaven** — RDNA4 FP8 E4M3 WMMA GEMM (`V_WMMA_F32_16X16X16_FP8_FP8`).
//!
//! Computes `C[M,N] = A[M,K] @ B^T` with A and B both FP8 E4M3, accumulating in
//! FP32. One wavefront owns a 16x32 output tile (two 16x16 tiles) and stages A
//! once, reusing it across both `mma_sync` calls from registers.
//!
//! # This kernel is the WhiteRaven format and nothing else
//!
//! Per §0 of `plans/PLAN-corvid-precision.md`, a kernel carries a codename only
//! if it contains that format and no other. An earlier revision of this file
//! declared `_Float16` fragments on `_Float16*` inputs while its doc header
//! claimed the FP8 instruction — an FP16 WMMA wearing the FP8 name. It was dead
//! code (its only launcher is `#[allow(dead_code)]` with no callers), so nothing
//! caught it. There is deliberately **no FP16 fallback branch here**: RDNA3 has
//! no FP8 WMMA, so dispatch selects the FP16 WMMA kernels instead, and mixing the
//! two into one translation unit would make this a composite that the §6 A/B
//! could not interpret.
//!
//! # Instruction
//!
//! ROCm 7.2.4's rocWMMA carries FP8 as `rocwmma::float8_t` (= `hip_fp8_e4m3`)
//! and lowers `fragment<..., float8_t, ...>` to
//! `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`, verified by
//! disassembly on gfx1201. The `_w32` and `_gfx12` suffixes are both required —
//! the unsuffixed spelling is not a builtin.
//!
//! The "2x the throughput of FP16" figure is a vendor claim, not a grim
//! measurement. It is the subject of step D5, not an assumption here.

/// HIP source for the WhiteRaven FP8 E4M3 WMMA GEMM kernel.
pub const KERNEL_SOURCE: &str = r#"
#if defined(__gfx1200__) || defined(__gfx1201__)
#include <rocwmma/rocwmma.hpp>
using namespace rocwmma;

// WhiteRaven: FP8 E4M3 x FP8 E4M3 -> FP32, 16x32 output tile per wavefront.
//
// __launch_bounds__(32) matches the launcher's block_dim = (32,1,1): one
// wavefront, two 16x16 WMMA fragments. The bound is not decoration -- A is staged
// in registers and reused across both mma_sync calls, so the register budget is
// the difference between fitting and spilling to scratch, the same failure class
// that cost 60x at wmma_quantized_gemm.rs:8 ("384 VGPRs, over the 256 limit").
// RDNA4's 96-VGPR-per-wave budget is 3x the 32 registers a wavefront holds at
// full occupancy, so there is headroom for the A fragment here.
extern "C" __global__ __launch_bounds__(32) void grim_wmma_gemm_fp8_e4m3(
    const float8_t* __restrict__ A,  // [M, K] FP8 E4M3, row-major
    const float8_t* __restrict__ B,  // [N, K] FP8 E4M3 (B^T is [K, N])
    float* __restrict__ C,           // [M, N] FP32, tightly packed
    int M, int N, int K) {

    const int tile_row = blockIdx.y;
    const int tile_col_base = blockIdx.x * 2;  // two 16-wide N tiles per block

    if (tile_row * 16 >= M || tile_col_base * 16 >= N) return;

    const int row_base = tile_row * 16;
    const int col_base = tile_col_base * 16;
    const bool valid_col1 = ((tile_col_base + 1) * 16 < N);

    fragment<matrix_a, 16, 16, 16, float8_t, row_major> frag_a;
    fragment<matrix_b, 16, 16, 16, float8_t, col_major> frag_b0;
    fragment<matrix_b, 16, 16, 16, float8_t, col_major> frag_b1;
    fragment<accumulator, 16, 16, 16, float> frag_c0;
    fragment<accumulator, 16, 16, 16, float> frag_c1;
    fill_fragment(frag_c0, 0.0f);
    fill_fragment(frag_c1, 0.0f);

    __shared__ float c_out[16 * 16];

    for (int k0 = 0; k0 < K; k0 += 16) {
        // A is reused by both N tiles: one load serves both mma_sync calls.
        load_matrix_sync(frag_a, A + (long long)row_base * K + k0, K);
        load_matrix_sync(frag_b0, B + (long long)col_base * K + k0, K);
        mma_sync(frag_c0, frag_a, frag_b0, frag_c0);
#if GRIM_SCHED_GROUP_BARRIER
        __builtin_amdgcn_sched_group_barrier(0xffffffff, 1, 0);
#endif
        if (valid_col1) {
            load_matrix_sync(frag_b1, B + (long long)(col_base + 16) * K + k0, K);
            mma_sync(frag_c1, frag_a, frag_b1, frag_c1);
        }
#if GRIM_SCHED_GROUP_BARRIER
        __builtin_amdgcn_sched_group_barrier(0xffffffff, 1, 0);
#endif
    }

    float* c0_ptr = C + (long long)row_base * N + col_base;
    store_matrix_sync(c_out, frag_c0, 16, layout_t::mem_row_major);
    __builtin_amdgcn_wave_barrier();
    for (int idx = threadIdx.x; idx < 256; idx += 32) {
        int i = idx / 16;
        int j = idx % 16;
        if (row_base + i < M && col_base + j < N) c0_ptr[(long long)i * N + j] = c_out[idx];
    }

    if (valid_col1) {
        float* c1_ptr = C + (long long)row_base * N + (col_base + 16);
        store_matrix_sync(c_out, frag_c1, 16, layout_t::mem_row_major);
        __builtin_amdgcn_wave_barrier();
        for (int idx = threadIdx.x; idx < 256; idx += 32) {
            int i = idx / 16;
            int j = idx % 16;
            if (row_base + i < M && col_base + 16 + j < N)
                c1_ptr[(long long)i * N + j] = c_out[idx];
        }
    }
}

// WhiteRaven-blocked: same math as grim_wmma_gemm_fp8_e4m3, but B arrives in
// 16x16-blocked order (grim_quant::block_fp8_16x16): tile (nt, kb) holds B rows
// nt*16..+16, cols kb*16..+16 contiguously, row-major within the block. The old
// entry's col_major fragment load with ldm=K pulled 16 K-strided 16B segments
// (~25% DRAM efficiency, measured 149 GB/s on a 640 GB/s card); with ldm=16
// the same fragment is ONE contiguous 256B read (4x64B transactions, all
// useful). A stays row-major: at decode shapes it is L2-resident, so its
// strided fragment load costs cache bandwidth, not DRAM. N%16==0, K%16==0
// enforced by the launcher; the epilogue store masking is unchanged.
extern "C" __global__ __launch_bounds__(32) void grim_wmma_gemm_fp8_e4m3_blocked(
    const float8_t* __restrict__ A,  // [M, K] FP8 E4M3, row-major (unchanged)
    const float8_t* __restrict__ Bb, // blocked B: [N/16][K/16][16][16] FP8
    float* __restrict__ C,           // [M, N] FP32, tightly packed
    int M, int N, int K) {

    const int tile_row = blockIdx.y;
    const int tile_col_base = blockIdx.x * 2;  // two 16-wide N tiles per block

    if (tile_row * 16 >= M || tile_col_base * 16 >= N) return;

    const int row_base = tile_row * 16;
    const int col_base = tile_col_base * 16;
    const bool valid_col1 = ((tile_col_base + 1) * 16 < N);
    const int KB = K >> 4;  // K-blocks per N-tile; K%16==0 by launcher contract

    fragment<matrix_a, 16, 16, 16, float8_t, row_major> frag_a;
    fragment<matrix_b, 16, 16, 16, float8_t, col_major> frag_b0;
    fragment<matrix_b, 16, 16, 16, float8_t, col_major> frag_b1;
    fragment<accumulator, 16, 16, 16, float> frag_c0;
    fragment<accumulator, 16, 16, 16, float> frag_c1;
    fill_fragment(frag_c0, 0.0f);
    fill_fragment(frag_c1, 0.0f);

    __shared__ float c_out[16 * 16];

    for (int kb = 0, k0 = 0; k0 < K; ++kb, k0 += 16) {
        // A is reused by both N tiles: one load serves both mma_sync calls.
        load_matrix_sync(frag_a, A + (long long)row_base * K + k0, K);
        // Contiguous 256B tile (N-tile tile_col_base, K-block kb), ldm 16.
        load_matrix_sync(frag_b0, Bb + ((long long)tile_col_base * KB + kb) * 256, 16);
        mma_sync(frag_c0, frag_a, frag_b0, frag_c0);
#if GRIM_SCHED_GROUP_BARRIER
        __builtin_amdgcn_sched_group_barrier(0xffffffff, 1, 0);
#endif
        if (valid_col1) {
            load_matrix_sync(frag_b1, Bb + ((long long)(tile_col_base + 1) * KB + kb) * 256, 16);
            mma_sync(frag_c1, frag_a, frag_b1, frag_c1);
        }
#if GRIM_SCHED_GROUP_BARRIER
        __builtin_amdgcn_sched_group_barrier(0xffffffff, 1, 0);
#endif
    }

    float* c0_ptr = C + (long long)row_base * N + col_base;
    store_matrix_sync(c_out, frag_c0, 16, layout_t::mem_row_major);
    __builtin_amdgcn_wave_barrier();
    for (int idx = threadIdx.x; idx < 256; idx += 32) {
        int i = idx / 16;
        int j = idx % 16;
        if (row_base + i < M && col_base + j < N) c0_ptr[(long long)i * N + j] = c_out[idx];
    }

    if (valid_col1) {
        float* c1_ptr = C + (long long)row_base * N + (col_base + 16);
        store_matrix_sync(c_out, frag_c1, 16, layout_t::mem_row_major);
        __builtin_amdgcn_wave_barrier();
        for (int idx = threadIdx.x; idx < 256; idx += 32) {
            int i = idx / 16;
            int j = idx % 16;
            if (row_base + i < M && col_base + 16 + j < N)
                c1_ptr[(long long)i * N + j] = c_out[idx];
        }
    }
}
// WhiteRaven act prologue: quantize an F32 activation [M, K] to FP8 E4M3
// codes padded to whole 16-row tiles ([16*ceil(M/16), K], pad rows = +0.0).
// This is what makes the blocked GEMM capture-safe: the host-side conversion
// used by the eager dispatch does a D2H readback plus an allocation, both of
// which poison a HIP graph capture (alloc) or stall it (sync). This kernel
// writes into a pre-allocated scratch, so graph decode can stage A on device.
//
// Encoding is RNE, bit-identical to the host `f32_to_fp8_e4m3` and to
// dot_gemv.rs::grim_f32_to_fp8_e4m3 -- the same rule all three must hold to,
// because the A operand the kernel accumulates is whatever this produces.
extern "C" __global__ __launch_bounds__(256) void grim_quant_fp8_pad16(
    const float* __restrict__ x,   // [M, K] F32 activations, or [M, K] fp8
                                   // codes reinterpreted when src_is_fp8 != 0
    unsigned char* __restrict__ out,// [16*ceil(M/16), K] FP8 E4M3 codes
    int M, int K, int src_is_fp8) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * K;
    // Tail rows must be zeroed: the fragment load reads whole 16-row tiles, and
    // stale scratch would be accumulated as phantom activations (15 of them at
    // decode, where M=1).
    int padded_total = ((M + 15) / 16) * 16 * K;
    if (idx >= total && idx < padded_total) {
        out[idx] = 0;
    }
    if (idx < total) {
        int row = idx / K;
        int col = idx - row * K;
        if (src_is_fp8) {
            out[row * K + col] = ((const unsigned char*)x)[idx];
            return;
        }
        float v = x[idx];
        unsigned sign = __builtin_signbit(v) ? 0x80u : 0x00u;
        float a = __builtin_fabsf(v);
        unsigned char code;
        if (__builtin_isnan(v)) {
            code = 0x7F;
        } else if (__builtin_isinf(a) || a >= 448.0f) {
            code = (unsigned char)(sign | 0x7E);  // saturate to 448
        } else if (a == 0.0f) {
            code = (unsigned char)sign;
        } else {
            unsigned bits;
            __builtin_memcpy(&bits, &a, 4);
            unsigned mm = bits & 0x7FFFFFu;
            int e = (int)((bits >> 23) & 0xFFu);
            if (e == 0) {
                code = (unsigned char)sign;      // f32 subnormal: below E4M3 min
            } else {
                int E = e - 120;                 // E4M3 exponent, bias 7
                if (E >= 1) {
                    unsigned q = mm >> 20;
                    unsigned r = mm & 0xFFFFFu;
                    if (r > 0x80000u || (r == 0x80000u && (q & 1u))) q++;
                    if (q == 8u) { q = 0; E++; }
                    code = (E > 15) ? (unsigned char)(sign | 0x7E)
                                    : (unsigned char)(sign | ((unsigned)E << 3) | q);
                } else {
                    int sh = 21 - E;
                    unsigned mant = 0x800000u | mm;
                    unsigned q = mant >> sh;
                    unsigned r = mant & ((1u << sh) - 1u);
                    unsigned half = 1u << (sh - 1);
                    if (r > half || (r == half && (q & 1u))) q++;
                    code = (q == 0) ? (unsigned char)sign : (unsigned char)(sign | q);
                }
            }
        }
        out[row * K + col] = code;
    }
}
#endif // gfx1200/gfx1201 — deliberately no FP16 fallback; see module docs
"#;

#[cfg(test)]
mod self_tests {
    use super::*;

    #[test]
    fn source_contains_fp8_kernel_entry() {
        assert!(
            KERNEL_SOURCE.contains("grim_wmma_gemm_fp8_e4m3"),
            "WhiteRaven kernel entry must be JIT-discoverable"
        );
    }

    #[test]
    fn source_is_gfx12_guarded() {
        // FP8 WMMA is only available on gfx1200/gfx1201 (RDNA4).
        assert!(
            KERNEL_SOURCE.contains("defined(__gfx1200__)"),
            "WhiteRaven must be guarded for RDNA4 targets"
        );
        // Should NOT include gfx11xx (RDNA3) — FP8 WMMA is RDNA4-only, and an
        // FP16 fallback branch here would make this a mixed kernel.
        assert!(
            !KERNEL_SOURCE.contains("defined(__gfx1100__)"),
            "FP8 WMMA is RDNA4-only, should not include RDNA3 guard"
        );
    }

    #[test]
    fn source_uses_dual_issue() {
        assert!(
            KERNEL_SOURCE.matches("mma_sync").count() >= 2,
            "WhiteRaven should dual-issue: frag_a reused across two N tiles"
        );
    }

    #[test]
    fn source_uses_fp8_element_type() {
        assert!(
            KERNEL_SOURCE.contains("float8_t"),
            "fragments must be FP8 (rocwmma::float8_t), not _Float16"
        );
        assert!(
            !KERNEL_SOURCE.contains("_Float16"),
            "no _Float16 may remain in the WhiteRaven kernel"
        );
    }

    #[test]
    fn a_fragment_is_loaded_once_for_both_n_tiles() {
        // The win over the old kernel is that A is staged once and reused; two
        // mma_sync calls must share one A load -- per entry point. Scoped to
        // each entry because the blocked twin repeats the same structure.
        let (old, blocked) = KERNEL_SOURCE
            .split_once("grim_wmma_gemm_fp8_e4m3_blocked")
            .expect("blocked entry must exist");
        for (name, src) in [("row-major", old), ("blocked", blocked)] {
            assert_eq!(
                src.matches("load_matrix_sync(frag_a").count(),
                1,
                "{name} entry: frag_a must be loaded once per k-step and reused across both N tiles"
            );
            assert_eq!(src.matches("mma_sync(frag_c").count(), 2);
        }
    }

    #[test]
    fn act_prologue_exists_and_encodes_rne_without_a_scale_header() {
        // The prologue writes raw E4M3 codes: a 4-byte f32 scale prefix (what
        // the standalone `grim_quant_fp8` kernel emits) would make the GEMM's
        // A fragment load read codes shifted by four.
        assert!(
            KERNEL_SOURCE.contains("grim_quant_fp8_pad16"),
            "act prologue must be JIT-discoverable"
        );
        let proto = KERNEL_SOURCE
            .split_once("grim_quant_fp8_pad16")
            .expect("prologue entry")
            .1;
        assert!(
            !proto.contains("scale_bits"),
            "act prologue must not write a scale prefix; codes are bare E4M3"
        );
        // RNE ties-to-even: the same rule as the host converter.
        assert!(
            proto.contains("r == 0x80000u && (q & 1u)"),
            "act prologue must round-to-nearest-even to match f32_to_fp8_e4m3"
        );
        assert!(
            proto.contains("a >= 448.0f"),
            "act prologue must saturate at 448, not 480 (the 0x7F NaN slot)"
        );
        // Tail rows must be zeroed in the SAME launch: a separate memset would
        // be a second stream operation, and at decode 15 of 16 rows are tail.
        assert!(
            proto.contains("padded_total"),
            "act prologue must zero the pad rows it does not write"
        );
        assert!(
            proto.contains("src_is_fp8"),
            "act prologue must accept pre-encoded codes without a second pass"
        );
    }

    #[test]
    fn blocked_entry_loads_b_tiles_contiguously() {
        // Pins the reason the blocked entry exists: B fragment loads with
        // ldm=16 from 256B tile bases, not K-strided gathers.
        let blocked = KERNEL_SOURCE
            .split_once("grim_wmma_gemm_fp8_e4m3_blocked")
            .expect("blocked entry must exist")
            .1;
        assert!(
            blocked.contains("* 256, 16)"),
            "blocked B loads must be contiguous 256B tiles with ldm=16"
        );
        // A keeps its row-major ldm=K load (L2-resident at decode shapes); B
        // must not.
        for line in blocked
            .lines()
            .filter(|l| l.contains("load_matrix_sync(frag_b"))
        {
            assert!(
                !line.contains(", K)"),
                "blocked B load must not use the row-major ldm=K stride: {line}"
            );
        }
    }
}
