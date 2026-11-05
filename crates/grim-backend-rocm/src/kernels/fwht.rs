//! FWHT-1024 (Prism/Bonsai Hadamard) HIP kernels.
//!
//! Port of PrismML `fwht.cu` for the Bonsai block width 1024:
//! - `fwht_cuda_smem`  -> `grim_fwht_smem_*`   (prefill, many rows: one row
//!   per block, all log2(N) butterfly stages in LDS, one syncthreads per stage)
//! - `fwht_cuda_block` -> `grim_fwht_block_*`  (decode, few rows: one row per
//!   256-thread block, shfl within a wave + LDS across waves + register pairs)
//!
//! Butterfly sign convention, verbatim from fwht.cu:112-121 ("the low element
//! of a pair takes x + y, the high one x - y"). Scale 1/sqrt(N) folded at load
//! (fwht.cu:103-104). Signs: row r uses `signs[(r % n_blk) * N ...]`
//! (fwht.cu:100). wave32-safe: the block butterfly sizes its stages off the
//! runtime `warpSize`, no 64-wave assumption anywhere.

pub const KERNEL_SOURCE: &str = r#"
#define FWHT_N 1024
#define FWHT_NT 256

// In-register block FWHT over FWHT_N floats held as NE = FWHT_N/FWHT_NT per
// thread, element i*NT + tid in reg[i]. Port of
// ggml_cuda_fwht_block_butterfly<N=1024, NT=256> (fwht.cuh) with the segment
// width B fixed at N. Warp-stage bounds use the runtime warpSize so the same
// binary is correct under wave64 as well as wave32.
__device__ inline void grim_fwht_dev_block_butterfly(float* reg, float* s, const int tid) {
    constexpr int NT = FWHT_NT;
    constexpr int NE = FWHT_N / NT;
    const int ws   = warpSize;
    const int lane = tid % ws;

    // stages within a wave: partner differs in the lane bits
    for (int h = 1; h < ws; h *= 2) {
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFFull, val, h, ws);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // stages across waves: partner differs in the thread-index bits above the lane
    for (int h = ws; h < NT; h *= 2) {
        for (int j = 0; j < NE; j++) {
            s[j * NT + tid] = reg[j];
        }
        __syncthreads();
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = s[j * NT + (tid ^ h)];
            reg[j] = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        __syncthreads();
    }

    // stages above the block width: partner is another register of the same thread
    for (int h = NT; h < FWHT_N; h *= 2) {
        const int step = h / NT;
        for (int j = 0; j < NE; j += 2 * step) {
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
}

__device__ inline float grim_fwht_load_f32(const float v) { return v; }

#define FWHT_SMEM_KERNEL(KNAME, STYPE, LOADF)                                        \
extern "C" __global__ void KNAME(                                                    \
    const STYPE* __restrict__ src, float* __restrict__ dst,                          \
    const long long n_rows, const float scale,                                       \
    const float* __restrict__ signs, const int n_blk)                                \
{                                                                                    \
    __shared__ float s[FWHT_N];                                                      \
    const long long r = blockIdx.x;                                                  \
    if (r >= n_rows) {                                                               \
        return;                                                                      \
    }                                                                                \
    src += r * FWHT_N;                                                               \
    dst += r * FWHT_N;                                                               \
    const float* signs_row = signs ? signs + (r % n_blk) * FWHT_N : 0;               \
    for (int i = threadIdx.x; i < FWHT_N; i += FWHT_NT) {                            \
        float v = LOADF(src[i]) * scale;                                             \
        if (signs_row) {                                                             \
            v *= signs_row[i];                                                       \
        }                                                                            \
        s[i] = v;                                                                    \
    }                                                                                \
    __syncthreads();                                                                 \
    for (int h = 1; h < FWHT_N; h *= 2) {                                            \
        for (int idx = threadIdx.x; idx < FWHT_N / 2; idx += FWHT_NT) {              \
            const int j = ((idx / h) * 2 * h) + (idx % h);                           \
            const float x = s[j];                                                    \
            const float y = s[j + h];                                                \
            s[j]     = x + y;                                                        \
            s[j + h] = x - y;                                                        \
        }                                                                            \
        __syncthreads();                                                             \
    }                                                                                \
    for (int i = threadIdx.x; i < FWHT_N; i += FWHT_NT) {                            \
        dst[i] = s[i];                                                               \
    }                                                                                \
}

#define FWHT_BLOCK_KERNEL(KNAME, STYPE, LOADF)                                       \
extern "C" __global__ void KNAME(                                                    \
    const STYPE* __restrict__ src, float* __restrict__ dst,                          \
    const long long n_rows, const float scale,                                       \
    const float* __restrict__ signs, const int n_blk)                                \
{                                                                                    \
    constexpr int NT = FWHT_NT;                                                      \
    constexpr int NE = FWHT_N / NT;                                                  \
    __shared__ float s[FWHT_N];                                                      \
    const long long r = blockIdx.x;                                                  \
    if (r >= n_rows) {                                                               \
        return;                                                                      \
    }                                                                                \
    src += r * FWHT_N;                                                               \
    dst += r * FWHT_N;                                                               \
    const int tid = threadIdx.x;                                                     \
    const float* signs_row = signs ? signs + (r % n_blk) * FWHT_N : 0;               \
    float reg[NE];                                                                   \
    for (int i = 0; i < NE; ++i) {                                                   \
        float v = LOADF(src[i * NT + tid]) * scale;                                  \
        if (signs_row) {                                                             \
            v *= signs_row[i * NT + tid];                                            \
        }                                                                            \
        reg[i] = v;                                                                  \
    }                                                                                \
    grim_fwht_dev_block_butterfly(reg, s, tid);                                      \
    for (int i = 0; i < NE; ++i) {                                                   \
        dst[i * NT + tid] = reg[i];                                                  \
    }                                                                                \
}

FWHT_SMEM_KERNEL(grim_fwht_smem_f32,  float,          grim_fwht_load_f32)
FWHT_SMEM_KERNEL(grim_fwht_smem_f16,  unsigned short, fp16_to_float_device)
FWHT_BLOCK_KERNEL(grim_fwht_block_f32, float,          grim_fwht_load_f32)
FWHT_BLOCK_KERNEL(grim_fwht_block_f16, unsigned short, fp16_to_float_device)

// ---------------------------------------------------------------------------
// TILED variant (Task 4 graph wiring, Bonsai widths 5120 / 6144 / 17408).
//
// A folded linear whose input dim is `n_blk * FWHT_N` applies a BLOCK-DIAGONAL
// D·H to the full activation row: block b of every row uses
// `signs[b*FWHT_N .. (b+1)*FWHT_N)` and its own independent FWHT. The
// single-launch kernels above stride `src` by the compile-time FWHT_N, so they
// cannot address a row whose stride is a multiple of FWHT_N — calling them
// against a 5120-wide buffer would read the WRONG columns.
//
// This variant takes the row stride as a runtime parameter and spreads the
// work over a 2D grid: `blockIdx.x` = block index b, `blockIdx.y` = activation
// row r. One launch covers the whole `n_blk * rows` grid, so a folded linear
// costs exactly ONE graph node instead of `n_blk` gather/kernel/scatter
// round-trips (which is also what keeps the capture node count bounded).
//
// Signs semantics DIFFER deliberately from the single-launch path: there the
// `(r % n_blk)` modulo encoded fwht.cu's per-ROW block cycling for 1024-wide
// activations. Here the activation row is FULL WIDTH and block b always pairs
// with `signs[b*FWHT_N]`, so the row index does not enter the signs offset —
// `signs + b * FWHT_N`. This is the fork's `build_hadamard_input` contract
// (llama-graph.cpp:1684-1712): one sign tensor per full width, applied to the
// same block column of every row.
#define FWHT_TILED_KERNEL(KNAME, STYPE, LOADF)                                      \
extern "C" __global__ void KNAME(                                                    \
    const STYPE* __restrict__ src, float* __restrict__ dst,                          \
    const long long n_rows, const long long row_stride,                             \
    const float scale, const float* __restrict__ signs)                             \
{                                                                                    \
    __shared__ float s_sh[FWHT_N];                                                   \
    const int b = blockIdx.x;                                                        \
    const long long r = blockIdx.y;                                                 \
    if (r >= n_rows) {                                                               \
        return;                                                                      \
    }                                                                                \
    const long long base = r * row_stride + (long long) b * FWHT_N;                 \
    const float* signs_blk = signs ? signs + (long long) b * FWHT_N : 0;            \
    float reg[FWHT_N / FWHT_NT];                                                     \
    for (int i = 0; i < FWHT_N / FWHT_NT; ++i) {                                     \
        const int idx = i * FWHT_NT + threadIdx.x;                                   \
        float v = LOADF(src[base + idx]) * scale;                                    \
        if (signs_blk) {                                                             \
            v *= signs_blk[idx];                                                     \
        }                                                                            \
        reg[i] = v;                                                                  \
    }                                                                                \
    grim_fwht_dev_block_butterfly(reg, s_sh, threadIdx.x);                          \
    for (int i = 0; i < FWHT_N / FWHT_NT; ++i) {                                     \
        const int idx = i * FWHT_NT + threadIdx.x;                                   \
        dst[base + idx] = reg[i];                                                    \
    }                                                                                \
}

FWHT_TILED_KERNEL(grim_fwht_tiled_f32, float,          grim_fwht_load_f32)
FWHT_TILED_KERNEL(grim_fwht_tiled_f16, unsigned short, fp16_to_float_device)

// GDN ssm_out activation permutation (llama-graph.cpp:1696-1704): reshape
// [hd, nk, rep] -> [hd, rep, nk] before signs + FWHT. One thread per element.
extern "C" __global__ void grim_gdn_perm_f32(
    const float* __restrict__ src, float* __restrict__ dst,
    long long total, int perm_hd, int perm_nk, int perm_rep)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int n_in = perm_hd * perm_nk * perm_rep;
    long long r = idx / n_in;
    int off = (int)(idx % n_in);
    int i0 = off % perm_hd;
    int i1 = (off / perm_hd) % perm_nk;
    int i2 = off / (perm_hd * perm_nk);
    int new_off = i0 + i2 * perm_hd + i1 * perm_hd * perm_rep;
    dst[r * n_in + new_off] = src[r * n_in + off];
}
"#;

#[cfg(test)]
mod fwht_kernel_self_tests {
    use super::*;

    #[test]
    fn fwht_source_has_all_four_entry_points() {
        for name in [
            "grim_fwht_smem_f32",
            "grim_fwht_smem_f16",
            "grim_fwht_block_f32",
            "grim_fwht_block_f16",
        ] {
            assert!(
                // The kernels are macro-stamped, so the entry point shows up
                // as a macro argument, not as `void name(` literal text.
                KERNEL_SOURCE.contains(name),
                "FWHT kernel source missing entry point {name}"
            );
        }
    }

    /// Butterfly convention guard: the low element of a pair must take
    /// x + y and the high one x - y, in BOTH the LDS and block paths
    /// (fwht.cu:112-121). A flipped sign convention halves-parity-fails in
    /// Task 4's fused pipeline and is miserable to bisect there.
    #[test]
    fn fwht_butterfly_sign_convention_low_plus_high_minus() {
        // LDS path: "s[j] = x + y; s[j + h] = x - y"
        let low = KERNEL_SOURCE.matches("s[j]     = x + y;").count();
        let high = KERNEL_SOURCE.matches("s[j + h] = x - y;").count();
        assert_eq!(low, 1, "LDS butterfly low element must be x + y");
        assert_eq!(high, 1, "LDS butterfly high element must be x - y");
        // Block path: reg[j] = x + y, reg[j+k+step] = x - y (register stages)
        assert!(KERNEL_SOURCE.contains("reg[j + k]        = x + y;"));
        assert!(KERNEL_SOURCE.contains("reg[j + k + step] = x - y;"));
        // shfl stages encode the same convention as fwht.cu:52
        assert!(KERNEL_SOURCE.contains("(lane & h) == 0 ? val + val2 : val2 - val"));
    }

    /// Signs must be indexed with the n_blk modulo (fwht.cu:100).
    #[test]
    fn fwht_signs_use_nblk_modulo() {
        assert!(KERNEL_SOURCE.contains("(r % n_blk) * FWHT_N"));
    }
}
