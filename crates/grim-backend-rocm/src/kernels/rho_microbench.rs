//! ρ microbench: the Tensor-Core : CUDA-core throughput ratio on the running
//! GPU, measured with memory-free loops (APEX4-style; see old/res4).
//!
//! - `grim_rho_fp32_fma`: 4 independent FMA accumulators per lane — the
//!   CUDA-core (VALU fp32) rate.
//! - `grim_rho_wmma_f16`: 4 rotating 16x16x16 f16 WMMA accumulators per
//!   wave, fragments loaded once — the Tensor-Core rate.
//!
//! ρ = TC/CC decides whether dequant-on-the-fly (grim's fused fills) is
//! structurally right for this silicon: low ρ means the CUDA cores are too
//! slow to feed the tensor cores, which is exactly what the fused design
//! assumes. Consumes no memory; iter counts are compile-time.

pub const KERNEL_SOURCE: &str = r#"
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1200__) || defined(__gfx1201__)
#include <rocwmma/rocwmma.hpp>
using namespace rocwmma;
#endif

// 4 independent FMA chains x ITERS: ~pure VALU fp32 throughput.
extern "C" __global__ void grim_rho_fp32_fma(float* out, int iters) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float a0 = (float)gid * 1e-9f + 1.0f;
    float a1 = a0 + 0.5f, a2 = a0 + 1.5f, a3 = a0 + 2.5f;
    float b = 0.9999999f;
    for (int i = 0; i < iters; ++i) {
        a0 = fmaf(a0, b, 1.0f);
        a1 = fmaf(a1, b, 1.0f);
        a2 = fmaf(a2, b, 1.0f);
        a3 = fmaf(a3, b, 1.0f);
    }
    out[gid] = a0 + a1 + a2 + a3;
}

// 4 rotating WMMA accumulators x ITERS, fragments loaded once: ~pure TC rate.
extern "C" __global__ void grim_rho_wmma_f16(float* out, int iters) {
    using namespace rocwmma;
    const int wave = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    fragment<matrix_a, 16, 16, 16, _Float16, row_major> ma;
    fragment<matrix_b, 16, 16, 16, _Float16, col_major> mb;
    fragment<accumulator, 16, 16, 16, float> c0, c1, c2, c3;
    fill_fragment(c0, 0.0f); fill_fragment(c1, 0.0f);
    fill_fragment(c2, 0.0f); fill_fragment(c3, 0.0f);
    __shared__ _Float16 buf[256];
    const int lane = threadIdx.x & 31;
    if (threadIdx.x < 256) buf[threadIdx.x] = (_Float16)(1.0f + threadIdx.x * 1e-6f);
    __syncthreads();
    load_matrix_sync(ma, buf, 16);
    load_matrix_sync(mb, buf, 16);
    for (int i = 0; i < iters; i += 4) {
        mma_sync(c0, ma, mb, c0);
        mma_sync(c1, ma, mb, c1);
        mma_sync(c2, ma, mb, c2);
        mma_sync(c3, ma, mb, c3);
    }
    float s = 0.0f;
    for (int e = 0; e < c0.num_elements; ++e) s += c0.x[e] + c1.x[e] + c2.x[e] + c3.x[e];
    if (lane == 0 && wave == (blockIdx.x & 1023)) out[wave & 4095] = s;
}

// res0-T3-9: the INT8 twin — RDNA4's I8 WMMA rate is claimed 2x the FP16
// rate (2048 vs 1024 FLOP/clk/CU, GPUOpen matrix-cores table); this measures
// it. Same skeleton, int8 fragments, i32 accumulators (8192 I8 ops per
// mma_sync, same count as the f16 kernel's FLOPs).
extern "C" __global__ void grim_rho_wmma_i8(float* out, int iters) {
    using namespace rocwmma;
    const int wave = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    fragment<matrix_a, 16, 16, 16, int8_t, row_major> ma;
    fragment<matrix_b, 16, 16, 16, int8_t, col_major> mb;
    fragment<accumulator, 16, 16, 16, int32_t> c0, c1, c2, c3;
    fill_fragment(c0, 0); fill_fragment(c1, 0);
    fill_fragment(c2, 0); fill_fragment(c3, 0);
    __shared__ int8_t buf[256];
    const int lane = threadIdx.x & 31;
    if (threadIdx.x < 256) buf[threadIdx.x] = (int8_t)(threadIdx.x % 7);
    __syncthreads();
    load_matrix_sync(ma, buf, 16);
    load_matrix_sync(mb, buf, 16);
    for (int i = 0; i < iters; i += 4) {
        mma_sync(c0, ma, mb, c0);
        mma_sync(c1, ma, mb, c1);
        mma_sync(c2, ma, mb, c2);
        mma_sync(c3, ma, mb, c3);
    }
    int s = 0;
    for (int e = 0; e < c0.num_elements; ++e) s += c0.x[e] + c1.x[e] + c2.x[e] + c3.x[e];
    if (lane == 0 && wave == (blockIdx.x & 1023)) out[wave & 4095] = (float)s;
}
"#;
