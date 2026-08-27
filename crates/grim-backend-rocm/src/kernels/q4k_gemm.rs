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

}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// TEMP-DIAG (GGUF fault hunt): hipRTC-compile ONLY this file's
    /// KERNEL_SOURCE and report whether the fused-q4k symbols exist in the
    /// resulting module. Opt-in via GRIM_Q4K_REPRO=1.
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
        // Variant (c): bisect the full aggregate. Walk the exact
        // compute_kernel_source() push order cumulatively and find where
        // grim_fused_dequant_gemm_q4k stops appearing in the compiled object.
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
    }
}
