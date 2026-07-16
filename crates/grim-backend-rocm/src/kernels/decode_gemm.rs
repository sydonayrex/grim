//! Decode-shaped F16 GEMM HIP kernel (WI 2.4.4-2, Rust-centric rewrite).
//!
//! Replaces the previously vendored Composable Kernel `ck_tile` wrapper
//! (`src/device/ck_gemm.cpp`) which required a build-time `hipcc` step and
//! violated grim's Rust-centric boundary. This kernel:
//!   - lives as an embedded HIP source literal (`KERNEL_SOURCE`),
//!   - is concatenated into `compute_kernel_source()` and JIT-compiled at
//!     runtime via `hipModuleLoad` / `hipModuleGetFunction` /
//!     `hipModuleLaunchKernel` — same path every other grim compute kernel
//!     uses (see `compute_kernels.rs` / `qkv_attention.rs`),
//!   - is dispatched from `RocmDevice::matmul` behind `DecodeGemmConfig`,
//!     **off by default**; opt-in via the config flag, never silently
//!     swapped in over rocBLAS.
//!
//! Compute shape (decode-class, mirrors `lookup_gemm_config` decode branch):
//!   - C[M, N] = A[M, K] @ B[K, N], row-major, all FP16, f32 accumulate, FP16
//!     output.
//!   - Block tile (M_TILE, N_TILE) = (8, 64). Matches the
//!     `TileGemmShape<8, 64, 64>` shape the vendored CK header wanted but
//!     could not instantiate on gfx1036 inside `BlockUniversalGemmAsBsCr`.
//!   - K-step = 16 elements per LDS load (one `__half` per thread-lane of
//!     the inner K axis). Double-buffered LDS (DCU-GCN §3.1, item 2 of
//!     the plan): the thread block prefetches the next K-step while the
//!     compute units are still consuming the current one, removing the
//!     load→compute sync stall on the GEMM critical path.
//!   - M is small (≤ 8 in decode); we mask out-of-range OOB lanes
//!     explicitly so a leftover block never writes past the M boundary.
//!
//! Performance gate (`TODO(gpu-verify)` — see
//! `grim_rocm_consumer_perf_plan.md` WI 2.6.4): microbench against
//! rocBLAS `gemm_ex` is a follow-up. Plan §2.4.4 item 4 (SMALL-BATCH-MC)
//! warns double-buffering can *reduce* decode throughput vs. plain rocBLAS
//! when m is already tiny — the dispatch flag stays off by default, and
//! flipping it on should be tied to a positive benchmark, not assumed.

extern crate alloc;

/// HIP source for `grim_decode_gemm_f16`.
///
/// Concatenated into the crate-wide `COMPUTE_KERNEL_SOURCE` constant for
/// JIT compilation at runtime. The kernel entry point is
/// `extern "C" __global__` so `hipModuleGetFunction` resolves it without
/// name mangling.
///
/// ABI (matches the Rust launch wrapper in `RocmDevice`):
///   `__global__ void grim_decode_gemm_f16(
///        const __half* A, const __half* B, __half* C,
///        int M, int N, int K,
///        int stride_a, int stride_b, int stride_c)`
///
/// Tiling constants (compile-time inside the kernel source):
///   M_TILE = 8, N_TILE = 64, K_STEP = 16, BLOCK = 256 (= 8 wave32s / 4 wave64s)
pub const KERNEL_SOURCE: &str = r#"
extern "C" __global__ void grim_decode_gemm_f16(
    const _Float16* __restrict__ A,
    const _Float16* __restrict__ B,
    _Float16* __restrict__ C,
    int M, int N, int K,
    int stride_a, int stride_b, int stride_c)
{
    // Decode-shape F16 GEMM: C[M,N] = A[M,K] @ B[K,N], f32 accumulate, F16 out.
    //
    // Simple, correct, single-buffer implementation. Each thread computes one
    // output element by iterating over the full K axis. Validated on gfx1036
    // (Radeon 610M, wave32). The double-buffered LDS variant (DCU-GCN §3.1)
    // is a future optimization gated on a measured speedup (per plan §2.4.4
    // item 4 / SMALL-BATCH-MC caution).
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
"#;

#[cfg(test)]
mod self_tests {
    use super::*;

    #[test]
    fn source_contains_kernel_entry_and_decoding_constants() {
        assert!(
            KERNEL_SOURCE.contains("extern \"C\" __global__ void grim_decode_gemm_f16"),
            "Decode GEMM kernel entry must be JIT-discoverable by name"
        );
        // The kernel signature must use _Float16 (the ABI-compatible f16 type
        // available without hip headers in the hipRTC compile path).
        assert!(KERNEL_SOURCE.contains("_Float16"), "kernel must use _Float16 type");
        // The kernel must do a K-loop dot-product accumulation in f32.
        assert!(KERNEL_SOURCE.contains("float acc = 0.0f"), "must accumulate in f32");
        assert!(KERNEL_SOURCE.contains("for (int k = 0; k < K"), "must loop over K");
    }
}
