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
"#;
