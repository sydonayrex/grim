//! On-device fused linear cross-entropy kernels.

/// Design B: Threadblock-parallel vocabulary scan with warp-level shuffle reduction
/// and online logsumexp combination across threads.
/// Eliminates single-thread bottleneck on large vocabulary models (128k+).
pub const FUSED_LINEAR_CE_KERNEL_SOURCE: &str = r#"
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 32; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffffffffffffULL, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 32; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffffffffffffULL, val, offset);
    }
    return val;
}

extern "C" __global__ void grim_fused_linear_ce_forward(
    const float* hidden, const float* lm_head, const unsigned int* targets,
    float* loss_out, float* lse_out, int hidden_dim, int vocab_size, int v_tile_size, int batch) {
    int row = blockIdx.x;
    if (row >= batch || row < 0) return;

    int tid = threadIdx.x;
    int bdim = blockDim.x;
    int target = targets[row];

    float local_max = -3.402823466e+38f;
    float local_sum = 0.0f;
    float target_logit = 0.0f;
    bool has_target = false;

    // Strided vocabulary walk across threads in the block
    for (int v = tid; v < vocab_size; v += bdim) {
        float logit = 0.0f;
        for (int d = 0; d < hidden_dim; ++d) {
            logit += hidden[row * hidden_dim + d] * lm_head[v * hidden_dim + d];
        }
        if (v == target) {
            target_logit = logit;
            has_target = true;
        }
        if (logit > local_max) {
            local_sum = local_sum * expf(local_max - logit) + 1.0f;
            local_max = logit;
        } else {
            local_sum += expf(logit - local_max);
        }
    }

    // Block-wide shared memory reduction
    __shared__ float s_max[64];
    __shared__ float s_sum[64];
    __shared__ float s_target;

    if (has_target) {
        s_target = target_logit;
    }
    __syncthreads();

    int lane = tid % 32;
    int wid = tid / 32;

    float w_max = warp_reduce_max(local_max);
    // Broadcast warp max
    w_max = __shfl_sync(0xffffffffffffffffULL, w_max, 0);

    // Rescale sum to warp max
    float w_sum = local_sum * expf(local_max - w_max);
    w_sum = warp_reduce_sum(w_sum);

    if (lane == 0) {
        s_max[wid] = w_max;
        s_sum[wid] = w_sum;
    }
    __syncthreads();

    if (tid == 0) {
        int num_warps = (bdim + 31) / 32;
        float block_max = -3.402823466e+38f;
        for (int w = 0; w < num_warps; ++w) {
            block_max = fmaxf(block_max, s_max[w]);
        }
        float block_sum = 0.0f;
        for (int w = 0; w < num_warps; ++w) {
            block_sum += s_sum[w] * expf(s_max[w] - block_max);
        }
        float lse = block_max + logf(block_sum);
        lse_out[row] = lse;
        loss_out[row] = lse - s_target;
    }
}

extern "C" __global__ void grim_fused_linear_ce_backward(
    const float* hidden, const float* lm_head, const unsigned int* targets, const float* lse,
    float* grad_h, int hidden_dim, int vocab_size, int v_tile_size, float inv_batch, int batch) {
    int row = blockIdx.x;
    if (row >= batch || row < 0) return;

    int tid = threadIdx.x;
    int bdim = blockDim.x;
    int target = targets[row];
    float lse_row = lse[row];

    for (int d = tid; d < hidden_dim; d += bdim) {
        grad_h[row * hidden_dim + d] = 0.0f;
    }
    __syncthreads();

    // Accumulate gradients across vocabulary chunks
    for (int v = 0; v < vocab_size; ++v) {
        float logit = 0.0f;
        for (int d = 0; d < hidden_dim; ++d) {
            logit += hidden[row * hidden_dim + d] * lm_head[v * hidden_dim + d];
        }
        float dl = (expf(logit - lse_row) - (v == target ? 1.0f : 0.0f)) * inv_batch;
        for (int d = tid; d < hidden_dim; d += bdim) {
            atomicAdd(&grad_h[row * hidden_dim + d], dl * lm_head[v * hidden_dim + d]);
        }
    }
}
"#;

