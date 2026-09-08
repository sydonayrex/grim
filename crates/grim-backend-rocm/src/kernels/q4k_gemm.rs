//! Q4_K Fused Dequantization GEMM HIP kernel (Crow Tier). [see: `block_q4_K`]

/// HIP source for `grim_fused_dequant_gemm_q4k` and `grim_fused_dequant_backward_gemm_q4k`.
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

    __global__ void grim_fused_dequant_gemm_q4k(
        const float* __restrict__ A,
        const unsigned char* __restrict__ B_q4k,
        float* __restrict__ C,
        int M, int N, int K)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * N;
        if (idx >= total) return;

        const int row = (int)(idx / N);
        const int col = (int)(idx % N);

        const int blocks_per_row = K / 256;
        const int row_bytes = blocks_per_row * 144;
        const unsigned char* row_b_ptr = B_q4k + col * row_bytes;

        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a_val = A[row * K + k];
            int sb_idx = k / 256;
            int in_sb = k % 256;
            const unsigned char* block_ptr = row_b_ptr + sb_idx * 144;
            float w_val = dequant_q4k_element(block_ptr, in_sb);
            acc += a_val * w_val;
        }

        C[row * N + col] = acc;
    }

    __global__ void grim_fused_dequant_backward_gemm_q4k(
        const float* __restrict__ dY,
        const unsigned char* __restrict__ B_q4k,
        float* __restrict__ dX,
        int M, int N, int K)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * K;
        if (idx >= total) return;

        const int row = (int)(idx / K);
        const int k_idx = (int)(idx % K);

        const int blocks_per_row = K / 256;
        const int row_bytes = blocks_per_row * 144;

        int sb_idx = k_idx / 256;
        int in_sb = k_idx % 256;

        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            float dy_val = dY[row * N + n];
            const unsigned char* block_ptr = B_q4k + n * row_bytes + sb_idx * 144;
            float w_val = dequant_q4k_element(block_ptr, in_sb);
            acc += dy_val * w_val;
        }

        dX[row * K + k_idx] = acc;
    }

    // SPEED-ROC-6: LDS-tiled Q4_K forward GEMM (prefill path).
    // TILE_N=64, TILE_K=64, TILE_M=4; 256 threads arranged (64 cols, 4 rows).
    // Each weight element is dequantized ONCE per M-tile (vs once per output
    // row in the scalar kernel) and staged through LDS; the A fragment is
    // broadcast from LDS rather than re-read from global per thread.
    // Requires K % 256 == 0 (standard Q4_K row layout); callers guard.
    __global__ void grim_fused_dequant_gemm_q4k_tiled(
        const float* __restrict__ A,
        const unsigned char* __restrict__ B_q4k,
        float* __restrict__ C,
        int M, int N, int K)
    {
        __shared__ float sW[64][64];  // [k_local][col] — 16 KB
        __shared__ float sA[4][64];   // [row][k_local] — 1 KB

        const int col0 = blockIdx.x * 64;
        const int row0 = blockIdx.y * 4;
        const int tx = threadIdx.x;   // 0..63 -> column within tile
        const int ty = threadIdx.y;   // 0..3  -> row within tile

        const int row = row0 + ty;
        const int col = col0 + tx;
        const bool tile_ok = (row < M) && (col < N);

        const int blocks_per_row = K / 256;
        const int row_bytes = blocks_per_row * 144;

        float acc = 0.0f;

        for (int k0 = 0; k0 < K; k0 += 64) {
            // Cooperative weight dequant: 256 threads cover the 64x64 tile
            // (16 elements each), reading raw bytes straight from global and
            // writing dequantized f32 into LDS.
            for (int t = ty * 64 + tx; t < 64 * 64; t += 256) {
                int kk = t >> 6;          // 0..63 local k
                int cc = t & 63;          // column
                int gk = k0 + kk;
                int gcc = col0 + cc;
                float w = 0.0f;
                if (gcc < N && gk < K) {
                    int sb_idx = gk / 256;
                    int in_sb = gk % 256;
                    w = dequant_q4k_element(
                        B_q4k + (long long)gcc * row_bytes + (long long)sb_idx * 144,
                        in_sb);
                }
                sW[kk][cc] = w;
            }
            // A fragment: rows 0..3 x k 0..63 = exactly 256 elements.
            {
                int gk = k0 + tx;
                int rr = row0 + ty;
                sA[ty][tx] = (rr < M && gk < K) ? A[(long long)rr * K + gk] : 0.0f;
            }
            __syncthreads();

            if (tile_ok) {
                #pragma unroll 8
                for (int kk = 0; kk < 64; ++kk) {
                    acc += sA[ty][kk] * sW[kk][tx];
                }
            }
            __syncthreads();
        }

        if (tile_ok) {
            C[(long long)row * N + col] = acc;
        }
    }

    // SPEED-ROC-6b: LDS-tiled Q4_K backward GEMM — dX[M,K] = dY[M,N] * W[N,K].
    // Same tiling discipline as the forward variant: block computes a
    // 4-row x 64-k tile of dX, staging the W super-block elements through LDS
    // once per N-chunk instead of re-dequantizing per output element.
    __global__ void grim_fused_dequant_gemm_q4k_backward_tiled(
        const float* __restrict__ dY,      // [M, N]
        const unsigned char* __restrict__ B_q4k,
        float* __restrict__ dX,            // [M, K]
        int M, int N, int K)
    {
        __shared__ float sW[64][64];  // [n_local][k_local] — 16 KB
        __shared__ float sY[4][64];   // [row][n_local] — 1 KB

        const int k0 = blockIdx.x * 64;
        const int row0 = blockIdx.y * 4;
        const int tx = threadIdx.x;   // 0..63 -> k within tile
        const int ty = threadIdx.y;   // 0..3  -> row within tile

        const int row = row0 + ty;
        const int kcol = k0 + tx;
        const bool tile_ok = (row < M) && (kcol < K);

        const int blocks_per_row = K / 256;
        const int row_bytes = blocks_per_row * 144;

        float acc = 0.0f;

        for (int n0 = 0; n0 < N; n0 += 64) {
            // Cooperative weight dequant: 256 threads cover the 64x64 tile.
            // sW[cc][kk] = W[n0+cc][k0+kk].
            for (int t = ty * 64 + tx; t < 64 * 64; t += 256) {
                int kk = t >> 6;
                int cc = t & 63;
                int gk = k0 + kk;
                int gn = n0 + cc;
                float w = 0.0f;
                if (gn < N && gk < K) {
                    w = dequant_q4k_element(
                        B_q4k + (long long)gn * row_bytes + (long long)(gk / 256) * 144,
                        gk % 256);
                }
                sW[cc][kk] = w;
            }
            // dY fragment: 4 rows x 64 n-cols = exactly 256 elements.
            {
                int gn = n0 + tx;
                int rr = row0 + ty;
                sY[ty][tx] = (rr < M && gn < N) ? dY[(long long)rr * N + gn] : 0.0f;
            }
            __syncthreads();

            if (tile_ok) {
                #pragma unroll 8
                for (int cc = 0; cc < 64; ++cc) {
                    acc += sY[ty][cc] * sW[cc][tx];
                }
            }
            __syncthreads();
        }

        if (tile_ok) {
            dX[(long long)row * K + kcol] = acc;
        }
    }

}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// TEMP-DIAG (GGUF fault hunt): hipRTC-compile ONLY this file's KERNEL_SOURCE and report whether the fused-q4k symbols exist in the resulting module.
    /// Opt-in via GRIM_Q4K_REPRO=1.
    #[test]
    fn diag_q4k_solo_compile_symbols() {
        if std::env::var("GRIM_Q4K_REPRO").is_err() {
            return;
        }
        let dev = crate::RocmDevice::try_new(0).expect("device");
        // Variant (b): exactly what the aggregate gives this file — the
        // shared-device-fn preamble first, then the q4k kernels.
        let combined = format!(
            "{}\n{}",
            crate::kernels::shared_device_fns::KERNEL_SOURCE,
            KERNEL_SOURCE
        );
        let src_ref: &str = &combined;
        match dev.jit_compile_or_cache(src_ref, "grim_fused_dequant_gemm_q4k", None) {
            Ok((path, lowered)) => {
                let bytes = std::fs::read(&path).unwrap();
                let text = String::from_utf8_lossy(&bytes).to_string();
                eprintln!(
                    "DIAG solo: bytes={} lowered={lowered} fwd_sym={} bwd_sym={}",
                    bytes.len(),
                    text.contains("grim_fused_dequant_gemm_q4k"),
                    text.contains("grim_fused_dequant_backward_gemm_q4k"),
                );
            }
            Err(e) => eprintln!("DIAG solo compile ERR: {e}"),
        }
        // Variant (c): bisect the full aggregate. Walk the exact compute_kernel_source() push
        // order cumulatively and find where grim_fused_dequant_gemm_q4k stops appearing in the compiled object.
        use crate::kernels::shared_device_fns;
        let parts: Vec<&str> = vec![
            shared_device_fns::KERNEL_SOURCE,
            crate::kernels::charon::KERNEL_SOURCE,
            crate::kernels::charon_wmma::KERNEL_SOURCE,
            crate::kernels::charon_backward::KERNEL_SOURCE,
            crate::kernels::compute_kernels::OTHER_KERNEL_SOURCE,
            crate::kernels::fused_linear_ce::FUSED_LINEAR_CE_KERNEL_SOURCE,
            crate::kernels::qkv_attention::KERNEL_SOURCE,
            crate::kernels::decode_gemm::KERNEL_SOURCE,
            crate::kernels::fused_dequant_gemm::KERNEL_SOURCE,
            crate::kernels::q4k_gemm::KERNEL_SOURCE,
            crate::kernels::q5k_gemm::KERNEL_SOURCE,
            crate::kernels::q6k_gemm::KERNEL_SOURCE,
            crate::kernels::iq_gemm::KERNEL_SOURCE,
            crate::kernels::kv_dequant_attention::KERNEL_SOURCE,
            crate::kernels::wmma_gemm::KERNEL_SOURCE,
        ];
        let mut cumulative = String::new();
        for (i, part) in parts.iter().enumerate() {
            cumulative.push_str(part);
            let probe_entry = format!("grim_bisect_probe_{i}");
            match dev.jit_compile_or_cache(&cumulative.clone(), &probe_entry, None) {
                Ok((path, _)) => {
                    let bytes = std::fs::read(&path).unwrap_or_default();
                    let has =
                        String::from_utf8_lossy(&bytes).contains("grim_fused_dequant_gemm_q4k");
                    eprintln!("BISECT idx={i} bytes={} q4k_sym={}", bytes.len(), has);
                    if !has {
                        eprintln!("BISECT culprit introduced at idx={i}");
                        break;
                    }
                }
                Err(e) => {
                    eprintln!("BISECT idx={i} compile ERR: {e}");
                    break;
                }
            }
        }
    }

    /// TEMP-DIAG (GGUF fault hunt): minimal M=1,N=16,K=256 launch through the
    /// real launcher. Opt-in via GRIM_Q4K_REPRO=1.
    #[test]
    fn diag_q4k_fused_gemm_minimal_launch() {
        if std::env::var("GRIM_Q4K_REPRO").is_err() {
            return;
        }
        let dev = crate::RocmDevice::try_new(0).expect("device");
        use grim_tensor::{BackendStorage, DType, Shape};

        const M: usize = 1;
        const N: usize = 16;
        const K: usize = 256;

        let a_data: Vec<f32> = (0..M * K).map(|i| ((i % 7) as f32) * 0.25 - 0.75).collect();
        let mut b_bytes = vec![0u8; N * 144];
        for (i, b) in b_bytes.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        let q4k_dtype = grim_tensor::DType {
            arith: grim_tensor::ArithType::F32,
            storage: grim_tensor::dtype::Storage::KQuant(grim_tensor::dtype::KQuantScheme::Q4K),
        };
        let alloc = std::sync::Arc::new(crate::memory::allocator::RocmCachingAllocator::new(
            0,
            1 << 30,
        ));
        let a_s = crate::memory::storage::RocmStorage::copy_from_host(
            &a_data,
            &Shape::new(vec![M, K]),
            DType::F32,
            &alloc,
            0,
        )
        .unwrap();
        let b_s = crate::memory::storage::RocmStorage::copy_from_host_raw_bytes(
            &b_bytes,
            &Shape::new(vec![N * 144]),
            q4k_dtype,
            &alloc,
            0,
        )
        .unwrap();
        let c_s = crate::memory::storage::RocmStorage::alloc_gpu(
            &Shape::new(vec![M, N]),
            DType::F32,
            &alloc,
            0,
        )
        .unwrap();

        let stream = dev
            .launch_fused_dequant_gemm_q4k(&a_s, &b_s, &c_s, M, N, K)
            .unwrap();
        unsafe { crate::hipStreamSynchronize(stream) };

        let host = c_s.to_cpu_vec_f32().unwrap();
        eprintln!("DIAG q4k out = {host:?}");
    }

    #[test]
    fn test_q4k_kernel_source_non_empty() {
        assert!(KERNEL_SOURCE.contains("grim_fused_dequant_gemm_q4k"));
        assert!(KERNEL_SOURCE.contains("grim_fused_dequant_backward_gemm_q4k"));
        // SPEED-ROC-6: the LDS-tiled prefill variant rides the same aggregate.
        assert!(KERNEL_SOURCE.contains("grim_fused_dequant_gemm_q4k_tiled"));
        assert!(KERNEL_SOURCE.contains("grim_fused_dequant_gemm_q4k_backward_tiled"));
    }
}
