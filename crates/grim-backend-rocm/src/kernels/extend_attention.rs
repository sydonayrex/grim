//! Extend-path Attention & Log-Sum-Exp (LSE) Chunk Merging for ROCm.
//!
//! Enables chunked context processing over long prefix-cached contexts (100k+ tokens)
//! and multi-turn conversations with exact mathematical state merging.

pub const KERNEL_SOURCE: &str = r#"
extern "C" {

// ---------------------------------------------------------------------------
// Extend Attention Chunk Kernel
// ---------------------------------------------------------------------------
//
// Computes attention for a batch of query tokens against a specific slice
// [chunk_start, chunk_end) of the context KV cache.
//
// Grid: (num_tokens, num_heads)
// Block: (head_dim, 1) or (128, 1)
// ---------------------------------------------------------------------------
__global__ void grim_extend_attention_chunk(
    const float* __restrict__ q,          // [num_tokens, num_heads, head_dim]
    const float* __restrict__ k_cache,    // [total_context_len, num_kv_heads, head_dim]
    const float* __restrict__ v_cache,    // [total_context_len, num_kv_heads, head_dim]
    float* __restrict__ chunk_out,        // [num_tokens, num_heads, head_dim]
    float* __restrict__ chunk_lse,        // [num_tokens, num_heads]
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int chunk_start,
    int chunk_end,
    float inv_sqrt_d
) {
    const int t = blockIdx.x; // query token index
    const int h = blockIdx.y; // query head index
    const int d = threadIdx.x; // head dim lane

    if (t >= num_tokens || h >= num_heads) return;

    const int q_per_kv = num_heads / num_kv_heads;
    const int kv_h = h / q_per_kv;
    const int q_base = (t * num_heads + h) * head_dim;

    extern __shared__ float s_ext_mem[];
    float* s_q = s_ext_mem;                 // [head_dim]
    float* s_dot = s_ext_mem + head_dim;    // [blockDim.x]

    if (d < head_dim) {
        s_q[d] = q[q_base + d];
    }
    __syncthreads();

    float running_max = -1e20f;
    float running_sum = 0.0f;
    float acc = 0.0f;

    for (int j = chunk_start; j < chunk_end; ++j) {
        const int kv_base = (j * num_kv_heads + kv_h) * head_dim;

        // Dot product Q . K[j]
        float prod = (d < head_dim) ? (s_q[d] * k_cache[kv_base + d]) : 0.0f;
        s_dot[threadIdx.x] = prod;
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
            float v_val = v_cache[kv_base + d];
            acc = acc * alpha + beta * v_val;
        }
    }

    if (d < head_dim) {
        const int out_idx = (t * num_heads + h) * head_dim + d;
        // Output unnormalized accumulator and LSE
        chunk_out[out_idx] = (running_sum > 0.0f) ? (acc / running_sum) : 0.0f;
    }
    if (d == 0) {
        const int lse_idx = t * num_heads + h;
        chunk_lse[lse_idx] = (running_sum > 0.0f) ? (running_max + logf(running_sum)) : -1e20f;
    }
}

// ---------------------------------------------------------------------------
// Merge Attention States Kernel (Log-Sum-Exp Combination)
// ---------------------------------------------------------------------------
//
// Merges state A (O_a, LSE_a) with state B (O_b, LSE_b) into state Out (O_out, LSE_out)
// using numerically exact log-sum-exp combination.
//
// Grid: (num_tokens, num_heads)
// Block: (head_dim, 1) or (128, 1)
// ---------------------------------------------------------------------------
__global__ void grim_merge_attn_states(
    const float* __restrict__ out_a,      // [num_tokens, num_heads, head_dim]
    const float* __restrict__ lse_a,      // [num_tokens, num_heads]
    const float* __restrict__ out_b,      // [num_tokens, num_heads, head_dim]
    const float* __restrict__ lse_b,      // [num_tokens, num_heads]
    float* __restrict__ out_merged,       // [num_tokens, num_heads, head_dim]
    float* __restrict__ lse_merged,       // [num_tokens, num_heads]
    int num_tokens,
    int num_heads,
    int head_dim
) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;

    if (t >= num_tokens || h >= num_heads) return;

    const int th_idx = t * num_heads + h;
    const float lse_1 = lse_a[th_idx];
    const float lse_2 = lse_b[th_idx];

    // Compute max LSE for numerical stability
    const float max_lse = max(lse_1, lse_2);
    if (max_lse <= -1e19f) {
        if (d < head_dim) {
            out_merged[th_idx * head_dim + d] = 0.0f;
        }
        if (d == 0) {
            lse_merged[th_idx] = -1e20f;
        }
        return;
    }

    const float w1 = expf(lse_1 - max_lse);
    const float w2 = expf(lse_2 - max_lse);
    const float sum_w = w1 + w2;

    if (d < head_dim) {
        const int elem_idx = th_idx * head_dim + d;
        const float val_a = out_a[elem_idx];
        const float val_b = out_b[elem_idx];
        out_merged[elem_idx] = (val_a * w1 + val_b * w2) / sum_w;
    }

    if (d == 0) {
        lse_merged[th_idx] = max_lse + logf(sum_w);
    }
}

} // extern "C"
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_source_contains_extend_and_merge() {
        assert!(KERNEL_SOURCE.contains("grim_extend_attention_chunk"));
        assert!(KERNEL_SOURCE.contains("grim_merge_attn_states"));
    }
}
