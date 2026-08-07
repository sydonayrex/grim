//! Standalone quantization HIP kernels for ROCm.

/// HIP source for `grim_quant_q8_0` and `grim_quant_fp8`.
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

    __device__ inline unsigned short float_to_fp16_device(float val) {
        if (val == 0.0f) return 0;
        unsigned int u = __float_as_uint(val);
        unsigned int sign = (u >> 31) & 1;
        int exp = (int)((u >> 23) & 0xFF) - 127;
        unsigned int mant = u & 0x7FFFFF;
        if (exp > 15) return (sign << 15) | (31 << 10);
        if (exp < -14) return (sign << 15);
        unsigned int new_exp = (unsigned int)(exp + 15);
        unsigned int new_mant = mant >> 13;
        return (sign << 15) | (new_exp << 10) | new_mant;
    }

    __device__ inline unsigned char float_to_fp8_e4m3_hip(float val) {
        if (isnan(val)) return 0x7F;
        float abs_val = fabsf(val);
        unsigned char sign = (val < 0.0f) ? 0x80 : 0x00;
        if (abs_val > 448.0f) abs_val = 448.0f;
        if (abs_val < 0.001953125f) { // 2^-9
            unsigned int mant = (unsigned int)roundf(abs_val * 512.0f);
            if (mant > 7) mant = 7;
            return sign | (unsigned char)mant;
        }
        unsigned int u = __float_as_uint(abs_val);
        int e = (int)((u >> 23) & 0xFF) - 127;
        unsigned int m = u & 0x7FFFFF;
        int e_fp8 = e + 7;
        if (e_fp8 >= 15) return sign | 0x7E;
        if (e_fp8 <= 0) {
            unsigned int mant = (unsigned int)roundf(abs_val * 512.0f);
            if (mant > 7) mant = 7;
            return sign | (unsigned char)mant;
        }
        unsigned int mant_bits = (m >> 20) & 0x7;
        return sign | ((unsigned char)e_fp8 << 3) | (unsigned char)mant_bits;
    }

    /// Quantize F32 vector into Q8_0 blocks (34 bytes per 32 elements: f16 scale + 32 i8s).
    __global__ void grim_quant_q8_0(
        const float* __restrict__ x,
        unsigned char* __restrict__ out,
        int total)
    {
        int block_idx = blockIdx.x;
        int tid = threadIdx.x; // 0..31 per block
        int idx = block_idx * 32 + tid;

        __shared__ float s_abs[32];
        float val = (idx < total) ? x[idx] : 0.0f;
        s_abs[tid] = fabsf(val);

        __syncthreads();

        for (int stride = 16; stride > 0; stride /= 2) {
            if (tid < stride) {
                if (s_abs[tid + stride] > s_abs[tid]) {
                    s_abs[tid] = s_abs[tid + stride];
                }
            }
            __syncthreads();
        }

        float amax = s_abs[0];
        float d = amax / 127.0f;
        float id = (d > 0.0f) ? (1.0f / d) : 0.0f;

        unsigned char* block_out = out + block_idx * 34;

        if (tid == 0) {
            unsigned short scale_f16 = float_to_fp16_device(d);
            block_out[0] = scale_f16 & 0xFF;
            block_out[1] = (scale_f16 >> 8) & 0xFF;
        }

        __syncthreads();

        if (idx < total) {
            int q = (int)roundf(val * id);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            block_out[2 + tid] = (unsigned char)(q & 0xFF);
        }
    }

    /// Quantize F32 vector to FP8 E4M3 (4-byte float scale = 1.0f prefix + FP8 bytes).
    __global__ void grim_quant_fp8(
        const float* __restrict__ x,
        unsigned char* __restrict__ out,
        int total)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx == 0) {
            float scale = 1.0f;
            unsigned int scale_bits = __float_as_uint(scale);
            out[0] = scale_bits & 0xFF;
            out[1] = (scale_bits >> 8) & 0xFF;
            out[2] = (scale_bits >> 16) & 0xFF;
            out[3] = (scale_bits >> 24) & 0xFF;
        }

        if (idx < total) {
            out[4 + idx] = float_to_fp8_e4m3_hip(x[idx]);
        }
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_standalone_source_contains_entries() {
        assert!(KERNEL_SOURCE.contains("grim_quant_q8_0"));
        assert!(KERNEL_SOURCE.contains("grim_quant_fp8"));
    }
}
