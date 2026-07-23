//! Fused Dequantization and GEMM HIP kernel (WI-C).
//!
//! Fuses the decompression of variable bit-width weights (2-4 bit codes + row scale),
//! outlier override logic, and optional residual backup layers into the GEMM loop.

/// HIP source for `grim_fused_dequant_gemm_f16`.
pub const KERNEL_SOURCE: &str = r#"
extern "C" {
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
        const _Float16* __restrict__ A,
        const unsigned char* __restrict__ B_codes,
        const unsigned char* __restrict__ B_scales,
        _Float16* __restrict__ C,
        int M, int N, int K,
        int stride_a, int stride_c,
        int default_bpw,
        int outlier_count,
        const unsigned int* __restrict__ outlier_indices,
        const float* __restrict__ outlier_values,
        int backup_bpw,
        int backup_codes_offset,
        int backup_scale_offset)
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
            float a_val = (float)A[row * stride_a + k];
            
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
            }
            
            acc += a_val * w_val;
        }

        C[row * stride_c + col] = (_Float16)acc;
    }

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
        int backup_scale_offset)
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
            }

            acc += dy_val * w_val;
        }

        dX[row * stride_dx + k] = (_Float16)acc;
    }
}
"#;

#[cfg(test)]
mod self_tests {
    use super::*;

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
    }
}
