//! Decode-shaped F16 GEMM HIP kernel (WI 2.4.4-2, Rust-centric rewrite). [see: `ck_tile`, `src/device/ck_gemm.cpp`, `hipcc`, `KERNEL_SOURCE`]

extern crate alloc;

/// HIP source for `grim_decode_gemm_f16`. [see: `COMPUTE_KERNEL_SOURCE`, `extern "C" __global__`, `hipModuleGetFunction`, `RocmDevice`]
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
        assert!(
            KERNEL_SOURCE.contains("_Float16"),
            "kernel must use _Float16 type"
        );
        // The kernel must do a K-loop dot-product accumulation in f32.
        assert!(
            KERNEL_SOURCE.contains("float acc = 0.0f"),
            "must accumulate in f32"
        );
        assert!(
            KERNEL_SOURCE.contains("for (int k = 0; k < K"),
            "must loop over K"
        );
    }
}
