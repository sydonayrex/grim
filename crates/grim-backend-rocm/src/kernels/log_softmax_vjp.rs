//! GPU-native fused log-softmax vector-Jacobian product (VJP) kernel.
//!
//! Evaluates the analytical gradient for cross-entropy and preference alignment loss
//! (DPO, KTO, SimPO, ORPO, GRPO) directly in GPU VRAM without host transfers:
//! \[
//! \frac{\partial \mathcal{L}}{\partial z_{t, v}} = \frac{\partial \mathcal{L}}{\partial \log \pi} \cdot (\mathbb{I}(v = y_t) - P(v))
//! \]

/// HIP C++ kernel source for fused log-softmax VJP evaluation with warp-level reductions.
pub const LOG_SOFTMAX_VJP_KERNEL_SOURCE: &str = r#"
__device__ __forceinline__ float warp_reduce_max_vjp(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down(val, offset, warpSize));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum_vjp(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset, warpSize);
    }
    return val;
}

extern "C" __global__ void grim_log_softmax_vjp_kernel(
    const float* logits,
    const unsigned int* targets,
    float* grad_out,
    float d_loss_d_logp,
    unsigned int ignore_index,
    int vocab_size,
    int seq_len
) {
    int row = blockIdx.x;
    if (row >= seq_len) return;

    unsigned int target = targets[row];
    if (target == ignore_index || target >= (unsigned int)vocab_size) {
        for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
            grad_out[row * vocab_size + v] = 0.0f;
        }
        return;
    }

    int tid = threadIdx.x;
    int bdim = blockDim.x;
    const float* row_logits = &logits[row * vocab_size];

    // 1. Threadblock-level maximum reduction
    float local_max = -3.402823466e+38f;
    for (int v = tid; v < vocab_size; v += bdim) {
        local_max = fmaxf(local_max, row_logits[v]);
    }
    __shared__ float s_max[64];
    float w_max = warp_reduce_max_vjp(local_max);
    int lane = tid % warpSize;
    int wid = tid / warpSize;
    if (lane == 0) s_max[wid] = w_max;
    __syncthreads();

    float row_max = (tid < (bdim / warpSize)) ? s_max[tid] : -3.402823466e+38f;
    row_max = warp_reduce_max_vjp(row_max);
    __shared__ float final_max;
    if (tid == 0) final_max = row_max;
    __syncthreads();
    row_max = final_max;

    // 2. Threadblock-level denominator sum reduction
    float local_sum = 0.0f;
    for (int v = tid; v < vocab_size; v += bdim) {
        local_sum += expf(row_logits[v] - row_max);
    }
    float w_sum = warp_reduce_sum_vjp(local_sum);
    __shared__ float s_sum[64];
    if (lane == 0) s_sum[wid] = w_sum;
    __syncthreads();

    float row_sum = (tid < (bdim / warpSize)) ? s_sum[tid] : 0.0f;
    row_sum = warp_reduce_sum_vjp(row_sum);
    __shared__ float final_sum;
    if (tid == 0) final_sum = row_sum;
    __syncthreads();
    row_sum = final_sum;
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;

    // 3. Write analytical VJP gradient: dL/dz = dL/dlogp * (delta - P(v))
    float* row_grad = &grad_out[row * vocab_size];
    for (int v = tid; v < vocab_size; v += bdim) {
        float prob = expf(row_logits[v] - row_max) * inv_sum;
        float delta = (v == target) ? 1.0f : 0.0f;
        row_grad[v] = d_loss_d_logp * (delta - prob);
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_softmax_vjp_kernel_source_syntax() {
        assert!(LOG_SOFTMAX_VJP_KERNEL_SOURCE.contains("grim_log_softmax_vjp_kernel"));
        assert!(LOG_SOFTMAX_VJP_KERNEL_SOURCE.contains("warp_reduce_max_vjp"));
        assert!(LOG_SOFTMAX_VJP_KERNEL_SOURCE.contains("warp_reduce_sum_vjp"));
    }
}
