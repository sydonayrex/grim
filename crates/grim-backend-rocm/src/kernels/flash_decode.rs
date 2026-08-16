//! FlashDecoding (Split-KV Parallel Attention) for long-context single-token decode.
//!
//! Divides the sequence dimension into `K_splits` independent thread blocks,
//! computes partial online softmax statistics in parallel, and merges partials
//! in a second reduction stage, saturating GPU Compute Units on long contexts.

pub const KERNEL_SOURCE: &str = r#"
extern "C" {

// ---------------------------------------------------------------------------
// FlashDecoding Stage 1: Partial Online Softmax per Sequence Chunk
// ---------------------------------------------------------------------------
//
// Grid: (num_heads, num_splits)
// Block: (128, 1) or (256, 1) — thread index d corresponds to head dimension.
// ---------------------------------------------------------------------------
__global__ void grim_flash_decode_stage1(
    const float* __restrict__ q,          // [num_heads, head_dim]
    const float* __restrict__ k_tensor,   // [kv_seq_len, num_kv_heads, head_dim]
    const float* __restrict__ v_tensor,   // [kv_seq_len, num_kv_heads, head_dim]
    float* __restrict__ mid_out,          // [num_splits, num_heads, head_dim]
    float* __restrict__ mid_max,          // [num_splits, num_heads]
    float* __restrict__ mid_sum,          // [num_splits, num_heads]
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int kv_seq_len,
    int num_splits,
    float inv_sqrt_d
) {
    const int h = blockIdx.x;        // query head index
    const int split_id = blockIdx.y; // sequence chunk split index
    const int d = threadIdx.x;       // head dimension lane

    if (h >= num_heads || split_id >= num_splits) return;

    const int q_per_kv = num_heads / num_kv_heads;
    const int kv_h = h / q_per_kv;

    // Determine sequence slice for this split
    const int chunk_size = (kv_seq_len + num_splits - 1) / num_splits;
    const int start_j = split_id * chunk_size;
    const int end_j = min(kv_seq_len, start_j + chunk_size);

    if (start_j >= kv_seq_len) {
        if (d < head_dim) {
            int out_idx = (split_id * num_heads + h) * head_dim + d;
            mid_out[out_idx] = 0.0f;
        }
        if (d == 0) {
            mid_max[split_id * num_heads + h] = -1e20f;
            mid_sum[split_id * num_heads + h] = 0.0f;
        }
        return;
    }

    extern __shared__ float s_stage1_mem[];
    float* s_q = s_stage1_mem; // [head_dim]
    float* s_dot = s_stage1_mem + head_dim; // [blockDim.x]

    // Load Q into shared memory
    if (d < head_dim) {
        s_q[d] = q[h * head_dim + d];
    }
    __syncthreads();

    float running_max = -1e20f;
    float running_sum = 0.0f;
    float acc = 0.0f;

    for (int j = start_j; j < end_j; ++j) {
        const int kv_base = (j * num_kv_heads + kv_h) * head_dim;

        // Compute dot product Q . K[j] across threads in block
        float local_prod = (d < head_dim) ? (s_q[d] * k_tensor[kv_base + d]) : 0.0f;
        s_dot[threadIdx.x] = local_prod;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                s_dot[threadIdx.x] += s_dot[threadIdx.x + stride];
            }
            __syncthreads();
        }
        float score = s_dot[0] * inv_sqrt_d;
        __syncthreads();

        // Online softmax update
        float new_max = max(running_max, score);
        float alpha = expf(running_max - new_max);
        float beta = expf(score - new_max);

        running_sum = running_sum * alpha + beta;
        running_max = new_max;

        if (d < head_dim) {
            float v_val = v_tensor[kv_base + d];
            acc = acc * alpha + beta * v_val;
        }
    }

    if (d < head_dim) {
        int out_idx = (split_id * num_heads + h) * head_dim + d;
        mid_out[out_idx] = acc;
    }
    if (d == 0) {
        mid_max[split_id * num_heads + h] = running_max;
        mid_sum[split_id * num_heads + h] = running_sum;
    }
}

// ---------------------------------------------------------------------------
// FlashDecoding Stage 2: Merge Partials Across Sequence Splits
// ---------------------------------------------------------------------------
//
// Grid: (num_heads, 1)
// Block: (128, 1) or (256, 1)
// ---------------------------------------------------------------------------
__global__ void grim_flash_decode_stage2(
    const float* __restrict__ mid_out,  // [num_splits, num_heads, head_dim]
    const float* __restrict__ mid_max,  // [num_splits, num_heads]
    const float* __restrict__ mid_sum,  // [num_splits, num_heads]
    float* __restrict__ out,            // [num_heads, head_dim]
    int num_heads,
    int head_dim,
    int num_splits
) {
    const int h = blockIdx.x;  // query head
    const int d = threadIdx.x; // head dimension lane

    if (h >= num_heads) return;

    // Step 1: Find global maximum across splits
    float global_max = -1e20f;
    for (int s = 0; s < num_splits; ++s) {
        float m_s = mid_max[s * num_heads + h];
        if (m_s > global_max) {
            global_max = m_s;
        }
    }

    // Step 2: Compute global sum of exponentials
    float global_sum = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
        float m_s = mid_max[s * num_heads + h];
        float l_s = mid_sum[s * num_heads + h];
        if (l_s > 0.0f) {
            global_sum += l_s * expf(m_s - global_max);
        }
    }

    // Step 3: Rescale and combine partial accumulators
    float final_acc = 0.0f;
    if (d < head_dim) {
        for (int s = 0; s < num_splits; ++s) {
            float m_s = mid_max[s * num_heads + h];
            float l_s = mid_sum[s * num_heads + h];
            if (l_s > 0.0f) {
                float factor = expf(m_s - global_max);
                int mid_idx = (s * num_heads + h) * head_dim + d;
                final_acc += mid_out[mid_idx] * factor;
            }
        }
        if (global_sum > 0.0f) {
            final_acc /= global_sum;
        }
        out[h * head_dim + d] = final_acc;
    }
}

} // extern "C"
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_source_contains_flash_decode_stages() {
        assert!(KERNEL_SOURCE.contains("grim_flash_decode_stage1"));
        assert!(KERNEL_SOURCE.contains("grim_flash_decode_stage2"));
    }
}
