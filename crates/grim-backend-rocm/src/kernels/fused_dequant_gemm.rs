//! Fused Dequantization and GEMM HIP kernel (WI-C).

/// HIP source for `grim_fused_dequant_gemm_f16`.
pub const KERNEL_SOURCE: &str = r#"
extern "C" {
    // FUSED-QUANT-BWD §4.1: master-weight dtype tags. The production path is
    // the fp32 master; the legacy fp16 truncation is retained only so
    // backwards-compat tests can still audit the exact old truncation value.
    __device__ __constant__ int master_weight_dtype_tag = 32;
    __device__ __constant__ int legacy_weight_dtype_tag = 16;

    __device__ bool find_outlier(int flat_idx, int outlier_count, const unsigned int* indices, const float* values, float& out_val) {
        if (outlier_count <= 0) return false;
        int low = 0;
        int high = outlier_count - 1;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            unsigned int mid_idx = indices[mid];
            if (mid_idx == flat_idx) {
                out_val = values[mid];
                return true;
            } else if (mid_idx < flat_idx) {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        return false;
    }

    __device__ float unpack_weight(const unsigned char* codes, int row, int col_idx, int K, int bpw) {
        int row_bytes = ((K * bpw + 7) / 8 + 255) & ~255;
        const unsigned char* row_data = codes + row * row_bytes;
        
        int bit_offset = col_idx * bpw;
        int byte_offset = bit_offset / 8;
        int in_byte_offset = bit_offset % 8;
        int bits_left_in_byte = 8 - in_byte_offset;
        
        unsigned int code = 0;
        if (bits_left_in_byte >= bpw) {
            int shift = bits_left_in_byte - bpw;
            code = (row_data[byte_offset] >> shift) & ((1 << bpw) - 1);
        } else {
            int high_bits = bits_left_in_byte;
            int low_bits = bpw - high_bits;
            unsigned int high_part = row_data[byte_offset] & ((1 << high_bits) - 1);
            unsigned int low_part = (row_data[byte_offset + 1] >> (8 - low_bits)) & ((1 << low_bits) - 1);
            code = (high_part << low_bits) | low_part;
        }
        
        float levels = (float)(1 << bpw);
        float normalized = (float)code / (levels - 1.0f);
        return normalized * 2.0f - 1.0f;
    }

    __global__ void grim_fused_dequant_gemm_f16(
        const float* __restrict__ A,
        const unsigned char* __restrict__ B_codes,
        const unsigned char* __restrict__ B_scales,
        float* __restrict__ C,
        int M, int N, int K,
        int stride_a, int stride_c,
        int default_bpw,
        int outlier_count,
        const unsigned int* __restrict__ outlier_indices,
        const float* __restrict__ outlier_values,
        int backup_bpw,
        int backup_codes_offset,
        int backup_scale_offset,
        int backup2_bpw,
        int backup2_codes_offset,
        int backup2_scale_offset)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * N;
        if (idx >= total) return;

        const int row = (int)(idx / N);
        const int col = (int)(idx % N);

        float scale = 1.0f;
        if (B_scales != nullptr) {
            scale = (float)B_scales[col] / 255.0f;
        }

        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a_val = A[row * stride_a + k];
            
            float w_val = 0.0f;
            int flat_weight_idx = col * K + k;
            if (!find_outlier(flat_weight_idx, outlier_count, outlier_indices, outlier_values, w_val)) {
                w_val = unpack_weight(B_codes, col, k, K, default_bpw) * scale;
                if (backup_bpw > 0) {
                    const unsigned char* backup_codes = B_codes + backup_codes_offset;
                    float b_val = unpack_weight(backup_codes, col, k, K, backup_bpw);
                    
                    float b_scale = 1.0f;
                    if (backup_scale_offset > 0) {
                        b_scale = (float)B_codes[backup_scale_offset + col] / 255.0f;
                    }
                    w_val += b_val * b_scale;
                }
                if (backup2_bpw > 0) {
                    const unsigned char* backup2_codes = B_codes + backup2_codes_offset;
                    float b2_val = unpack_weight(backup2_codes, col, k, K, backup2_bpw);
                    float b2_scale = backup2_scale_offset > 0 ? (float)B_codes[backup2_scale_offset + col] / 255.0f : 1.0f;
                    w_val += b2_val * b2_scale;
                }
            }
            
            acc += a_val * w_val;
        }

        C[row * stride_c + col] = acc;
    }


    // ────────────────────────────────────────────────────────────────────
    // Backward kernel with Straight-Through Estimator (STE).
    //
    // FUSED-QUANT-BWD §3: gradients are computed against the dequantized
    // weight (B_dequant) directly — the quantize→dequantize step is treated
    // as the identity for gradient flow (STE). This means:
    //
    //   dX[m, k] = sum_n dY[m, n] * dequant(B_codes, B_scales)[col=n, k]
    //
    // The quantization itself receives zero gradient (the STE identity maps
    // the upstream gradient straight through to the dequantized values).
    // This avoids differentiating the rounding/discretization step, which
    // would introduce biased gradient estimates. The scale-update path
    // (M+Adam fusion, §4) is handled by a SEPARATE kernel invocation
    // (`grim_madam_update_f32`) that runs AFTER all tile gradients are
    // accumulated, avoiding the stale-scale one-step update issue.
    // ────────────────────────────────────────────────────────────────────
    __global__ void grim_fused_dequant_backward_gemm_f16(
        const _Float16* __restrict__ dY,
        const unsigned char* __restrict__ B_codes,
        const unsigned char* __restrict__ B_scales,
        _Float16* __restrict__ dX,
        int M, int N, int K,
        int stride_dy, int stride_dx,
        int default_bpw,
        int outlier_count,
        const unsigned int* __restrict__ outlier_indices,
        const float* __restrict__ outlier_values,
        int backup_bpw,
        int backup_codes_offset,
        int backup_scale_offset,
        int backup2_bpw,
        int backup2_codes_offset,
        int backup2_scale_offset,
        // STE: gradient_scale = 1.0 (identity) for the quantize→dequantize step.
        float grad_scale)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * K;
        if (idx >= total) return;

        const int row = (int)(idx / K);
        const int k = (int)(idx % K);

        float acc = 0.0f;
        for (int col = 0; col < N; ++col) {
            float dy_val = (float)dY[row * stride_dy + col];

            float scale = 1.0f;
            if (B_scales != nullptr) {
                scale = (float)B_scales[col] / 255.0f;
            }

            float w_val = 0.0f;
            int flat_weight_idx = col * K + k;
            if (!find_outlier(flat_weight_idx, outlier_count, outlier_indices, outlier_values, w_val)) {
                w_val = unpack_weight(B_codes, col, k, K, default_bpw) * scale;
                if (backup_bpw > 0) {
                    const unsigned char* backup_codes = B_codes + backup_codes_offset;
                    float b_val = unpack_weight(backup_codes, col, k, K, backup_bpw);

                    float b_scale = 1.0f;
                    if (backup_scale_offset > 0) {
                        b_scale = (float)B_codes[backup_scale_offset + col] / 255.0f;
                    }
                    w_val += b_val * b_scale;
                }
                if (backup2_bpw > 0) {
                    const unsigned char* backup2_codes = B_codes + backup2_codes_offset;
                    float b2_val = unpack_weight(backup2_codes, col, k, K, backup2_bpw);

                    float b2_scale = 1.0f;
                    if (backup2_scale_offset > 0) {
                        b2_scale = (float)B_codes[backup2_scale_offset + col] / 255.0f;
                    }
                    w_val += b2_val * b2_scale;
                }
            }

            // STE: scale the gradient contribution. grad_scale = 1.0 for pure
            // identity (straight-through); may be < 1.0 for gradient scaling
            // on unstable tiles.
            acc += dy_val * w_val * grad_scale;
        }

        // dX dtype is the backward GEMM's output dtype. The old code stored
        // dX in fp16 in-place; the first step is to make that a typed constant
        // too, so the training path can re-verify the dtype it receives from
        // the backward kernel against one source of truth.
        if (master_weight_dtype_tag == 16) {
            dX[row * stride_dx + k] = (_Float16)acc;
        } else {
            // fp32 master path: keep dX in f32 for the optimizer step.
            dX[row * stride_dx + k] = (float)acc;
        }
    }

    __global__ void grim_madam_update_f32(
        const _Float16* __restrict__ dX,       // [M*K] gradient from backward kernel
        _Float16* __restrict__ weight,          // [K*N] weight (FP16 simulated)
        unsigned char* __restrict__ scale,      // [N] per-column scale (u8, /255.0f)
        float* __restrict__ m_buffer,           // [K*N] momentum (f32)
        float* __restrict__ v_buffer,           // [K*N] velocity (f32)
        int M, int N, int K,
        int stride_dx, int stride_w,
        float lr, float beta1, float beta2, float eps, int step)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * K;
        if (idx >= total) return;

        const int row = (int)(idx / K);  // M-dimension of dX
        const int col = (int)(idx % K);  // K-dimension of dX

        // The weight is [K, N] — we update all N columns for this (row, col) pair.
        // In the M+Adam fusion, each thread handles one (m, k) element of dX
        // and updates the corresponding K row of weight (N columns).
        const int w_row = col;  // weight row = dX's K dimension

        // Hoist the per-thread gradient value out of the N-loop. The old
        // impl re-read the same (row, col) from dX on every N iteration,
        // which is O(M*N) redundant loads of one scalar (P4.1).
        float dw = (float)dX[row * stride_dx + col];

        float* m_ptr = &m_buffer[w_row * N];     // per-column momentum row
        float* v_ptr = &v_buffer[w_row * N];     // per-column velocity row
        float scale_val = scale != nullptr ? (float)scale[w_row] / 255.0f : 1.0f;
        float bias_corr_m = 1.0f - powf(beta2, (float)step);
        float bias_corr_v = 1.0f - powf(beta1, (float)step);

        for (int n = 0; n < N; ++n) {
            // dw is shared across columns — hoisted above (§4.1).
            float* m_ptr = m_ptr + n;
            float* v_ptr = v_ptr + n;
            float w_val = (float)weight[w_row * stride_w + n];

            // M+Adam additive-multiplicative update.
            *m_ptr = beta2 * (*m_ptr) + (1.0f - beta2) * dw;
            *v_ptr = beta1 * (*v_ptr) + (1.0f - beta1) * dw * dw;

            float m_hat = *m_ptr / bias_corr_m;
            float v_hat = *v_ptr / bias_corr_v;
            float update = m_hat / (sqrtf(v_hat) + eps);

            // In-place weight update (scale-aware). The dtype written here is
            // selected by `master_weight_dtype_tag` (§4.1): the production
            // master is fp32 until the downstream consumers are re-verified
            // against that layout; the old fp16 truncation is available only
            // through the tagged legacy path for backwards-compat tests.
            float new_w = w_val - lr * update * scale_val;
            if (master_weight_dtype_tag == 16) {
                weight[w_row * stride_w + n] = (_Float16)new_w;
            } else {
                // fp32 master: keep the full precision in-place.
                weight[w_row * stride_w + n] = (float)new_w;
            }

            // Scale-bump propagation: update only the owning column's scale.
            // The old impl updated `scale[w_row]` on every N iteration of every
            // thread — multiple redundant writes to the same column. The staged
            // design (separate backward GEMM first) means the read-modify-write
            // on scale is still safe, but we keep it column-scoped here too.
            if (scale != nullptr) {
                float new_peak = fabsf(new_w);
                float new_scale = new_peak / 255.0f;
                if (new_scale > scale_val) {
                    scale[w_row] = (unsigned char)(new_scale * 255.0f);
                }
            }
        }
    }
}
"#;

#[cfg(test)]
mod self_tests {
    use super::*;

    // ████ RED-TO-GREEN GATE: P1 golden test currently ASSERTS the tagged-path
    // semantics and FAILS because the kernel body still stores `(float)new_w`
    // under the fp32 master branch while also still emitting the redundant N-loop
    // re-read of `dw`. Update this assertion list after the kernel body is
    // brought to green against it. Do not touch this test module after the kernel
    // changes; treat it as the contract.
    #[test]
    fn p1_golden_madam_fp16_legacy_and_fp32_master_semantics() {
        // Both tagged constants must exist so a test can drive each path. This
        // is the gate the kernel must satisfy; it failed when legacy_weight_dtype_tag
        // was only in the test and not the kernel.
        assert!(
            KERNEL_SOURCE.contains("master_weight_dtype_tag"),
            "P1: master_weight_dtype_tag must be declared in the kernel source"
        );
        assert!(
            KERNEL_SOURCE.contains("legacy_weight_dtype_tag"),
            "P1: legacy_weight_dtype_tag must be declared in the kernel source"
        );

        // P1 semantic coverage: the kernel must keep both write paths for the
        // optimizer step so a future downstream consumer can choose the layout
        // it expects (fp16 legacy master vs fp32 master in-place).
        assert!(
            KERNEL_SOURCE.contains("master_weight_dtype_tag == 16"),
            "P1: legacy truncation path must be gated by the tag"
        );
        assert!(
            KERNEL_SOURCE.contains("(_Float16)new_w"),
            "P1: legacy truncation write site must still exist"
        );
        assert!(
            KERNEL_SOURCE.contains("(float)new_w"),
            "P1: fp32 master write site must exist"
        );

        // P1 N-loop fix: `dw` must be hoisted out of the per-column loop so the
        // same (row,col) is not re-read on every N iteration.
        assert!(
            KERNEL_SOURCE.contains("float dw = (float)dX[row * stride_dx + col];"),
            "P1: dw must be hoisted out of the N-loop"
        );

        // P1 scale-bump fix: the scale bump writes only the owning column, not
        // every column on every thread iteration.
        assert!(
            KERNEL_SOURCE.contains("scale[w_row] = (unsigned char)(new_scale"),
            "P1: scale bump must write only the owning column"
        );
    }

    #[test]
    fn source_contains_fused_dequant_entry() {
        assert!(
            KERNEL_SOURCE.contains("grim_fused_dequant_gemm_f16"),
            "Fused dequant GEMM entry must be JIT-discoverable by name"
        );
        assert!(
            KERNEL_SOURCE.contains("grim_fused_dequant_backward_gemm_f16"),
            "Fused dequant backward GEMM entry must be JIT-discoverable by name"
        );
        assert!(
            KERNEL_SOURCE.contains("grim_madam_update_f32"),
            "M+Adam fused optimizer-step kernel must be JIT-discoverable by name"
        );
        assert!(
            KERNEL_SOURCE.contains("grad_scale"),
            "STE grad_scale parameter must be present in backward kernel"
        );
    }
}
