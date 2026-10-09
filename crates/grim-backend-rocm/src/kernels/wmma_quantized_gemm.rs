//! Consolidated RDNA3/RDMA4 WMMA fused-dequant GEMM kernels.
//!
//! All six block-quantized kernels (Q8_0, Q4_K, Q5_K, Q2_K, Q3_K, Q6_K) share
//! one structure: dual-N-tile per wave (16×32 effective), cooperative LDS
//! staging, wave barriers. The six bodies are stamped from one Rust-side
//! template (plain-text substitution — HIP `##` pasting is not used).
//!
//! PERF-CRITICAL: tiles stage through __shared__ LDS, NOT per-thread register
//! arrays. A previous version declared a_tile[256] + b0_tile[256] + b1_tile[256]
//! per thread = 384 VGPRs (over the 256 limit) → scratch spills → 60x slowdown
//! (GPU 99% busy, 1% memory activity). LDS staging also lets the 32 threads
//! cooperate: each fills 8 of 256 elements instead of every thread redundantly
//! computing all 256.

/// Prologue: rocwmma includes + per-format scalar dequant device functions.
pub const PROLOGUE_SOURCE: &str = r#"
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1200__) || defined(__gfx1201__)
#include <rocwmma/rocwmma.hpp>
using namespace rocwmma;

__device__ __forceinline__ void grim_deq_q80(const unsigned char* blk, int w, float* out) {
    // 8 consecutive codes share one fp16 scale; w is 8-aligned within block.
    float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
    const signed char* codes = (const signed char*)(blk + 2);
    #pragma unroll
    for (int e = 0; e < 8; ++e) out[e] = d * (float)codes[w + e];
}

__device__ __forceinline__ void grim_deq_q4k(const unsigned char* blk, int w, float* out) {
    const unsigned short* h = (const unsigned short*)blk;
    float d = fp16_to_float_device(h[0]);
    float dmin = fp16_to_float_device(h[1]);
    const unsigned char* scales = blk + 4;
    const unsigned char* qs = blk + 16;
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        int wv = w + e;
        int k = wv / 64, off = wv % 64;
        int s = 2 * k + (off >= 32 ? 1 : 0);
        int j = off & 31;
        unsigned char sc, m;
        if (s < 4) { sc = scales[s] & 63; m = scales[s + 4] & 63; }
        else {
            sc = (scales[s + 4] & 0x0F) | ((scales[s - 4] >> 6) << 4);
            m = (scales[s + 4] >> 4) | ((scales[s] >> 6) << 4);
        }
        int qsb = 32 * k + j;
        unsigned char q = (off < 32) ? (qs[qsb] & 0x0F) : (qs[qsb] >> 4);
        out[e] = d * (float)sc * (float)q - dmin * (float)m;
    }
}

__device__ __forceinline__ void grim_deq_q5k(const unsigned char* blk, int w, float* out) {
    const unsigned short* h = (const unsigned short*)blk;
    float d = fp16_to_float_device(h[0]);
    float dmin = fp16_to_float_device(h[1]);
    const unsigned char* scales = blk + 4;
    const unsigned char* qh = blk + 16;
    const unsigned char* qs = blk + 48;
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        int wv = w + e;
        int k = wv / 64, off = wv % 64;
        int s = 2 * k + (off >= 32 ? 1 : 0);
        int j = off & 31;
        unsigned char sc, m;
        if (s < 4) { sc = scales[s] & 63; m = scales[s + 4] & 63; }
        else {
            sc = (scales[s + 4] & 0x0F) | ((scales[s - 4] >> 6) << 4);
            m = (scales[s + 4] >> 4) | ((scales[s] >> 6) << 4);
        }
        int qsb = 32 * k + j;
        unsigned char hi = (unsigned char)((qh[j] >> (2 * k + (off >= 32 ? 1 : 0))) & 1) << 4;
        unsigned char q = (off < 32) ? ((qs[qsb] & 0x0F) | hi) : (((qs[qsb] >> 4) & 0x0F) | hi);
        out[e] = d * (float)sc * (float)q - dmin * (float)m;
    }
}

__device__ __forceinline__ void grim_deq_q2k(const unsigned char* blk, int w, float* out) {
    const unsigned char* scales = blk;
    const unsigned char* qs = blk + 16;
    float d    = fp16_to_float_device(((const unsigned short*)(blk + 80))[0]);
    float dmin = fp16_to_float_device(((const unsigned short*)(blk + 82))[0]);
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        int wv = w + e;
        int sub = wv / 16;
        int in_sub = wv % 16;
        unsigned char sc_byte = scales[sub];
        float sc = (float)(sc_byte & 0x0F);
        float m  = (float)(sc_byte >> 4);
        // llama.cpp interleaved codes (ggml-quants.c:959): field
        // 2*((sub%8)/2) of byte qs[(sub/8)*32 + in_sub + (sub%2)*16].
        int q_byte = (sub / 8) * 32 + in_sub + (sub % 2) * 16;
        int q_shift = 2 * ((sub % 8) / 2);
        unsigned char q = (qs[q_byte] >> q_shift) & 0x03;
        out[e] = d * sc * (float)q - dmin * m;
    }
}

__device__ __forceinline__ void grim_deq_q3k(const unsigned char* blk, int w, float* out) {
    // Canonical block_q3_K layout (110 bytes / 256 weights), matching
    // grim_quant::dequant_q3k byte-for-byte: hmask[32]@0, qs[64]@32,
    // scales[12]@96 (ggml bit-shuffle -> 16 i8), d f16@108.
    // value_i = d * (sc - 32) * (q - hmask_bit * 4).
    const unsigned char* hmask = blk;
    const unsigned char* qs = blk + 32;
    const unsigned char* scales_raw = blk + 96;
    float d = fp16_to_float_device(((const unsigned short*)blk)[54]); // byte 108

    // ggml 12-byte scale shuffle -> 16 i8 (dequantize_row_q3_K).
    unsigned aux0 = (unsigned)scales_raw[0] | ((unsigned)scales_raw[1] << 8)
                  | ((unsigned)scales_raw[2] << 16) | ((unsigned)scales_raw[3] << 24);
    unsigned aux1 = (unsigned)scales_raw[4] | ((unsigned)scales_raw[5] << 8)
                  | ((unsigned)scales_raw[6] << 16) | ((unsigned)scales_raw[7] << 24);
    unsigned tmp  = (unsigned)scales_raw[8] | ((unsigned)scales_raw[9] << 8)
                  | ((unsigned)scales_raw[10] << 16) | ((unsigned)scales_raw[11] << 24);
    unsigned kmask1 = 0x03030303u;
    unsigned kmask2 = 0x0F0F0F0Fu;
    int sc16[16];
    unsigned a0 = (aux0 & kmask2) | ((tmp & kmask1) << 4);
    unsigned a1 = (aux1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
    unsigned a2 = ((aux0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    unsigned a3 = ((aux1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    unsigned auxs[4] = { a0, a1, a2, a3 };
    int _si = 0;
    for (int j = 0; j < 4; ++j)
        for (int b2 = 0; b2 < 4; ++b2) {
            unsigned v = (auxs[j] >> (8 * b2)) & 0xFF;
            sc16[j * 4 + b2] = (int)(v < 128u ? (int)v : (int)v - 256);
        }

    for (int e = 0; e < 8; ++e) {
        int idx = w + e;
        int half = idx >> 7;
        int sub = (idx & 0x7F) >> 5;
        int p = idx & 0x1F;
        int sc_idx = (half << 3) + (sub << 1) + (p >> 4);
        int shift = sub << 1;
        int q_code = ((qs[(half << 5) + p]) >> shift) & 3;
        int hm_bit = (hmask[p] & (1u << (half * 4 + sub))) != 0 ? 0 : 4;
        float dl = d * (float)(sc16[sc_idx] - 32);
        out[e] = dl * (float)(q_code - hm_bit);
    }
}

__device__ __forceinline__ void grim_deq_q6k(const unsigned char* blk, int w, float* out) {
    // Canonical block_q6_K layout (210 bytes / 256 weights), matching
    // grim_quant::dequant_q6k and q6k_gemm.rs::dequant_q6k_element.
    const unsigned char* ql = blk;
    const unsigned char* qh = blk + 128;
    const signed char* scales = (const signed char*)(blk + 192);
    float d = fp16_to_float_device(((const unsigned short*)blk)[104]); // byte 208
    for (int e = 0; e < 8; ++e) {
        int idx = w + e;
        int n = idx >> 7;
        int pos = idx & 0x7F;
        int quarter = pos >> 5;
        int l = pos & 0x1F;
        int is = l >> 4;
        int sc_idx = (n << 3) + is + (quarter << 1);
        signed char sc = scales[sc_idx];
        int ql_off = (n << 6) + l + ((quarter & 1) ? 32 : 0);
        unsigned char ql_byte = ql[ql_off];
        int nibble = (quarter & 2) ? (ql_byte >> 4) : (ql_byte & 0x0F);
        unsigned char qh_byte = qh[(n << 5) + l];
        int qh_bits = (qh_byte >> (quarter << 1)) & 0x03;
        int q_code = nibble | (qh_bits << 4);
        out[e] = d * (float)sc * ((float)q_code - 32.0f);
    }
}
"#;

/// Generate one WMMA fused-dequant kernel for the given format.
/// Plain-text substitution — no HIP `##` token pasting.
///
/// `a_suffix` disambiguates the FP16-input variant (`"_fp16"`) from the
/// default FP32-input kernel.  When `a_suffix` is non-empty, A is
/// `_Float16*` and the cooperative load reads FP16 directly (no cast),
/// halving activation memory bandwidth on the A reads.
fn wmma_quant_kernel_source(
    fmt: &str,
    blk: u32,
    bytes: u32,
    deq_fn: &str,
    a_suffix: &str,
) -> String {
    let a_ty = if a_suffix.is_empty() {
        "float"
    } else {
        "_Float16"
    };
    let a_cast = if a_suffix.is_empty() {
        "(_Float16)"
    } else {
        ""
    };
    format!(
        r#"
// Tiled WMMA: 128-thread block (4 wave32s) computes a 16x64 output tile.
// Four B tiles per block amortize A load + LDS barriers over 4 mma_sync;
// multi-wave blocks give cross-wave LDS-latency hiding. Requires
// workgroup-scope sync (__syncthreads) - wave_barrier races across waves.
// One 16-element K slice of the A and B tiles, filled into LDS.
//
// Extracted from the kernel body so the double-buffered pipeline below can call
// it for either stage. `lds_b` is a pointer-to-array so one function serves the
// per-stage sub-array (`lds_b[stage]`).
__device__ __forceinline__ void grim_wmma_fill_stage_{fmt}{a_suffix}(
    _Float16* lds_a, _Float16 (*lds_b)[16 * 16],
    const {a_ty}* __restrict__ A, const unsigned char* __restrict__ B_q,
    int tid, int row_base, int col_base, int k0,
    int M, int N, int K, int n_blocks_per_row)
{{
    for (int idx = tid; idx < 256; idx += 128) {{
        int i = idx / 16;
        int j = idx % 16;
        int row = row_base + i;
        int kk = k0 + j;
        lds_a[idx] = (row < M && kk < K) ? {a_cast}A[row * K + kk] : (_Float16)0;
    }}
    for (int idx = tid; idx < 128; idx += 128) {{
        int t = idx / 32;
        int rem = idx % 32;
        int r = rem / 2;
        int half = rem % 2;
        int col = col_base + t * 16 + r;
        int kk = k0 + half * 8;
        _Float16* dst = lds_b[t] + r * 16 + half * 8;
        if (col < N && kk < K) {{
            const unsigned char* blkptr = B_q
                + (long long)col * n_blocks_per_row * {bytes}
                + (long long)(kk / {blk}) * {bytes};
            float acc[8];
            {deq_fn}(blkptr, kk % {blk}, acc);
            #pragma unroll
            for (int e = 0; e < 8; ++e) dst[e] = (_Float16)acc[e];
        }} else {{
            #pragma unroll
            for (int e = 0; e < 8; ++e) dst[e] = (_Float16)0;
        }}
    }}
}}

extern "C" __global__ void grim_wmma_fused_dequant_{fmt}{a_suffix}(
    const {a_ty}* __restrict__ A,
    const unsigned char* __restrict__ B_q,
    float* __restrict__ C,
    int M, int N, int K) {{

    const int tile_row = blockIdx.y;
    const int col_base = blockIdx.x * 64;
    if (tile_row * 16 >= M || col_base >= N) return;

    const int row_base = tile_row * 16;
    const int n_blocks_per_row = K / {blk};
    const int tid = threadIdx.x;              // 0..127

    // DOUBLE-BUFFERED LDS (two stages).
    //
    // The single-buffered version paid THREE __syncthreads() per 16-element K
    // step and serialised fill -> frag_a -> fill -> MMA -> fill, so every global
    // dequant load latency was exposed: measured 2.06 us per K step, and 38x off
    // the kernel's ALU bound overall (see tests/wmma_q4k_shape_bench.rs, which
    // also shows the per-tile cost FALLING as the grid grows — the signature of
    // latency exposure rather than throughput). With two stages the next slice's
    // global loads are issued BEFORE the current slice's MMAs, so that latency
    // overlaps compute, and one barrier per step orders the stages against each
    // other. This is the multi-stage LDS pipeline of AMD's low-latency-GEMM work.
    __shared__ _Float16 lds_a[2][16 * 16];      // 2 x 512 B
    __shared__ _Float16 lds_b[2][4][16 * 16];   // 2 x 2 KB

    fragment<matrix_a, 16, 16, 16, _Float16, row_major> frag_a;
    fragment<matrix_b, 16, 16, 16, _Float16, col_major> frag_b[4];
    fragment<accumulator, 16, 16, 16, float> frag_c[4];
    #pragma unroll
    for (int t = 0; t < 4; ++t) fill_fragment(frag_c[t], 0.0f);

    const int n_k_steps = (K + 15) / 16;

    // Prologue: stage 0 holds K slice 0.
    grim_wmma_fill_stage_{fmt}{a_suffix}(
        lds_a[0], lds_b[0], A, B_q, tid, row_base, col_base, 0,
        M, N, K, n_blocks_per_row);
    __syncthreads();

    for (int it = 0; it < n_k_steps; ++it) {{
        const int cur = it & 1;
        // Prefetch the NEXT slice into the other stage. These are the global
        // dequant loads; issuing them before the MMAs is the whole point.
        if (it + 1 < n_k_steps) {{
            grim_wmma_fill_stage_{fmt}{a_suffix}(
                lds_a[cur ^ 1], lds_b[cur ^ 1], A, B_q, tid, row_base, col_base,
                (it + 1) * 16, M, N, K, n_blocks_per_row);
        }}
        load_matrix_sync(frag_a, lds_a[cur], 16);
        #pragma unroll
        for (int t = 0; t < 4; ++t) {{
            load_matrix_sync(frag_b[t], lds_b[cur][t], 16);
            mma_sync(frag_c[t], frag_a, frag_b[t], frag_c[t]);
        }}
        // One barrier per step. It orders this step's prefetch writes against the
        // next iteration's reads of that stage (last read two iterations ago, so
        // no read/write hazard on the stage being written now).
        __syncthreads();
    }}

    __shared__ float c_out[4][16 * 16];
    #pragma unroll
    for (int t = 0; t < 4; ++t) {{
        store_matrix_sync(c_out[t], frag_c[t], 16, layout_t::mem_row_major);
    }}
    __syncthreads();
    for (int idx = tid; idx < 1024; idx += 128) {{
        int t = idx / 256;
        int rem = idx % 256;
        int i = rem / 16;
        int j = rem % 16;
        int row = tile_row * 16 + i;
        int col = col_base + t * 16 + j;
        if (row < M && col < N) {{
            C[(long long)row * N + col] = c_out[t][i * 16 + j];
        }}
    }}
}}
"#
    )
}
/// Assemble the full kernel source: prologue + all six expansions + epilogue.
///
/// Each format emits TWO kernels: the default FP32-input
/// `grim_wmma_fused_dequant_<fmt>` and the FP16-input
/// `grim_wmma_fused_dequant_<fmt>_fp16`.  The FP16 variant reads activations
/// as `_Float16` directly (no per-element cast), halving A-read bandwidth.
/// Q8_0 gets the FP16 variant (LFM2.5-350M is Q8_0); the other formats
/// currently only need the FP32 path, but both are emitted uniformly so the
/// dispatch can switch on activation dtype without JIT gaps.
pub fn quant_kernel_source() -> String {
    let mut s = String::with_capacity(64 * 1024);
    s.push_str(PROLOGUE_SOURCE);
    for fmt in ["q8_0", "q4k", "q5k", "q2k", "q3k", "q6k"] {
        let (blk, bytes, deq) = match fmt {
            "q8_0" => (32, 34, "grim_deq_q80"),
            "q4k" => (256, 144, "grim_deq_q4k"),
            "q5k" => (256, 176, "grim_deq_q5k"),
            "q2k" => (256, 84, "grim_deq_q2k"),
            "q3k" => (256, 110, "grim_deq_q3k"),
            "q6k" => (256, 210, "grim_deq_q6k"),
            _ => unreachable!(),
        };
        s.push_str(&wmma_quant_kernel_source(fmt, blk, bytes, deq, ""));
        s.push_str(&wmma_quant_kernel_source(fmt, blk, bytes, deq, "_fp16"));
    }
    s.push_str("\n#endif // RDNA3/RDNA4\n");
    s
}

/// Full JIT source (kept for the source_asm push site).
pub const KERNEL_SOURCE: &str = ""; // unused — source_asm calls quant_kernel_source()

#[cfg(test)]
mod self_tests {
    use super::*;

    #[test]
    fn source_contains_all_block_quant_kernels() {
        let src = quant_kernel_source();
        // q8_0 keeps the underscore to match the Rust launcher name.
        // Each format ships an FP32-input kernel and an FP16-input variant.
        for name in ["q8_0", "q4k", "q5k", "q2k", "q3k", "q6k"] {
            assert!(
                src.contains(&format!("grim_wmma_fused_dequant_{name}")),
                "missing WMMA kernel for {name}"
            );
            assert!(
                src.contains(&format!("grim_wmma_fused_dequant_{name}_fp16")),
                "missing FP16-input WMMA kernel for {name}"
            );
        }
    }

    #[test]
    fn source_uses_dual_issue() {
        let src = quant_kernel_source();
        let mma_count = src.matches("mma_sync").count();
        assert!(
            mma_count >= 24,
            "expected >= 24 mma_sync calls (12 kernels x 2), got {mma_count}"
        );
    }

    #[test]
    fn source_uses_lds_not_register_arrays() {
        let src = quant_kernel_source();
        assert!(
            src.contains("__shared__ _Float16 lds_a"),
            "kernels must stage tiles through LDS"
        );
        // The old spill bug: per-thread tile arrays must not return.
        assert!(
            !src.contains("_Float16 a_tile["),
            "per-thread a_tile arrays cause VGPR spilling — use LDS"
        );
        assert!(
            !src.contains("_Float16 b0_tile["),
            "per-thread b0_tile arrays cause VGPR spilling — use LDS"
        );
    }

    #[test]
    fn source_uses_cooperative_fill() {
        let src = quant_kernel_source();
        let coop = src
            .matches("for (int idx = tid; idx < 128; idx += 128)")
            .count();
        assert!(
            coop >= 12,
            "expected >= 12 cooperative dequant loops (12 kernels), got {coop}"
        );
    }

    #[test]
    fn fp16_variant_reads_float16_directly() {
        let src = quant_kernel_source();
        // The FP16-input Q8_0 kernel must read A as _Float16 (no cast), and
        // must not appear in the FP32-input kernel (which casts).
        assert!(
            src.contains("grim_wmma_fused_dequant_q8_0_fp16("),
            "FP16 Q8_0 kernel missing"
        );
        // The FP16 variant reads _Float16 directly: the load line has no
        // `(_Float16)` cast on A.  Find the fp16 kernel body and check.
        let fp16_start = src
            .find("grim_wmma_fused_dequant_q8_0_fp16(")
            .expect("fp16 kernel present");
        let fp16_body = &src[fp16_start..fp16_start + 1200];
        assert!(
            !fp16_body.contains("(_Float16)A["),
            "FP16 variant must NOT cast A — reads _Float16 directly"
        );
    }

    #[test]
    fn source_is_rdna3_rdna4_guarded() {
        let src = quant_kernel_source();
        assert!(
            src.contains("defined(__gfx1100__)") && src.contains("defined(__gfx1200__)"),
            "kernels must be guarded for RDNA3 and RDNA4 targets"
        );
    }

    #[test]
    fn source_uses_workgroup_barriers_around_lds() {
        let src = quant_kernel_source();
        // Multi-wave blocks need workgroup-scope sync; wave_barrier is
        // wave-scope only and races across waves (verified: pad-token garbage).
        let barriers = src.matches("__syncthreads()").count();
        assert!(
            barriers >= 48,
            "expected >= 48 __syncthreads, got {barriers}"
        );
        assert!(
            !src.contains("__builtin_amdgcn_wave_barrier()"),
            "wave_barrier is wave-scope — races in multi-wave blocks"
        );
    }
}
