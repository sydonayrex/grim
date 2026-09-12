//! Fused 3-in-1 SwiGLU activation + dynamic scale quantization HIP kernel.

/// HIP C++ source code for `grim_silu_mul_quantize`.
pub const SILU_MUL_QUANT_KERNEL_SOURCE: &str = r#"
extern "C" __global__ void grim_silu_mul_quantize(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    unsigned char* __restrict__ qout,
    float* __restrict__ scale_out,
    int n
) {
    int tid = threadIdx.x;
    float thread_max = 0.0f;

    for (int i = tid; i < n; i += blockDim.x) {
        float g = gate[i];
        float u = up[i];
        float silu = g / (1.0f + __expf(-g));
        float act = silu * u;
        thread_max = fmaxf(thread_max, fabsf(act));
    }

    __shared__ float s_max[256];
    s_max[tid] = thread_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }

    float max_val = s_max[0];
    float scale = (max_val > 0.0f) ? (max_val / 127.0f) : 1.0f;
    if (tid == 0) {
        scale_out[0] = scale;
    }
    float inv_scale = 1.0f / scale;

    for (int i = tid; i < n; i += blockDim.x) {
        float g = gate[i];
        float u = up[i];
        float silu = g / (1.0f + __expf(-g));
        float act = silu * u;
        float q = act * inv_scale;
        if (q > 127.0f) q = 127.0f;
        if (q < -127.0f) q = -127.0f;
        qout[i] = (unsigned char)((signed char)roundf(q));
    }
}

// SPEED-DOT-OPFUSE (Phase 4d): fused SwiGLU (silu(gate)*up) + Q8_1 quantization for M=1 decode.
// Directly emits 36-byte Q8_1 blocks [d fp16 LE][sum fp16 LE][32 int8 codes] consumed by grim_dot4_q80_q81_gemv
// or grim_dot4_q4k_q81_gemv in the ffn_down projection.
extern "C" __global__ void grim_silu_mul_quant_q8_1(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    unsigned char* __restrict__ out_q81,
    int K)
{
    const int tid = threadIdx.x; // 0..31
    const int n_blocks = K / 32;

    for (int blk = tid; blk < n_blocks; blk += 32) {
        const int base = blk * 32;
        float act[32];
        float amax = 0.0f;

        #pragma unroll
        for (int j = 0; j < 32; ++j) {
            float g = gate[base + j];
            float u = up[base + j];
            float silu = g / (1.0f + __expf(-g));
            float val = silu * u;
            act[j] = val;
            amax = fmaxf(amax, fabsf(val));
        }

        const float d = amax / 127.0f;
        const float inv_d = (amax > 1e-9f) ? (127.0f / amax) : 0.0f;
        unsigned char* dst = out_q81 + (long long)blk * 36;
        float fsum = 0.0f;

        #pragma unroll
        for (int j = 0; j < 32; ++j) {
            int q = (int)roundf(act[j] * inv_d);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            dst[4 + j] = (unsigned char)(signed char)q;
            fsum += (float)q;
        }

        _Float16 hd = (_Float16)d;
        _Float16 hs = (_Float16)(fsum * d);
        unsigned short wd, ws;
        __builtin_memcpy(&wd, &hd, 2);
        __builtin_memcpy(&ws, &hs, 2);
        dst[0] = (unsigned char)(wd & 0xFF);
        dst[1] = (unsigned char)(wd >> 8);
        dst[2] = (unsigned char)(ws & 0xFF);
        dst[3] = (unsigned char)(ws >> 8);
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu_mul_quant_q8_1_presence() {
        assert!(
            SILU_MUL_QUANT_KERNEL_SOURCE.contains("grim_silu_mul_quant_q8_1"),
            "grim_silu_mul_quant_q8_1 missing from SILU_MUL_QUANT_KERNEL_SOURCE"
        );
    }
}
