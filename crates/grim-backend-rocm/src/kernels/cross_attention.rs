//! Cross-attention kernel for Whisper decoder (Item 13).

/// HIP source for `grim_cross_attention` (Whisper decoder cross-attention).
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

    /// Whisper cross-attention: softmax(Q @ K^T / sqrt(head_dim)) @ V.
    ///
    /// Q:  [seq_len_q, num_heads, head_dim]       (row-major, stride = num_heads * head_dim)
    /// K:  [seq_len_k, num_heads_k, head_dim]     (row-major, stride = num_heads_k * head_dim)
    /// V:  [seq_len_k, num_heads_k, head_dim]     (row-major, stride = num_heads_k * head_dim)
    /// out: [seq_len_q, num_heads, head_dim]      (row-major, stride = num_heads * head_dim)
    ///
    /// GQA: num_heads_k divides num_heads evenly. Each group of
    /// (num_heads/num_heads_k) query heads shares the same K/V projection.
    ///
    /// Full (non-causal) cross-attention: every query attends to every
    /// encoder position. The output projection W_o is applied on the host.
    ///
    /// Launch geometry: one block per (q_pos, head) row. blockIdx.x indexes
    /// the flat (seq_len_q * num_heads) grid; blockDim.x may be any power of
    /// two >= head_dim. Shared memory holds the raw scores [seq_len_k] plus
    /// two per-block partial-reduction arrays of size blockDim.x.
    __global__ void grim_cross_attention(
        const float* __restrict__ Q,      // [seq_len_q, num_heads, head_dim]
        const float* __restrict__ K,      // [seq_len_k, num_heads_k, head_dim]
        const float* __restrict__ V,      // [seq_len_k, num_heads_k, head_dim]
        float* __restrict__ out,          // [seq_len_q, num_heads, head_dim]
        int seq_len_q,
        int seq_len_k,
        int num_heads,
        int num_heads_k,                  // GQA: num_heads_k <= num_heads
        int head_dim,
        float scale)                      // 1.0f / sqrt(head_dim)
    {
        // One block per (query position, head) row.
        const int row = (int)blockIdx.x;
        const int q_pos = row / num_heads;
        const int head = row % num_heads;
        const int kv_head = head % num_heads_k;
        const int tid = (int)threadIdx.x;
        const int bdim = (int)blockDim.x;

        // Shared memory layout: scores[seq_len_k] | red_max[bdim] | red_sum[bdim]
        extern __shared__ float smem[];
        float* scores = smem;                      // [seq_len_k]
        float* red_max = smem + seq_len_k;         // [bdim]
        float* red_sum = red_max + bdim;           // [bdim]

        const int q_base = q_pos * (num_heads * head_dim) + head * head_dim;
        const int kv_base = kv_head * head_dim;

        // Pass 1: scores[q_pos, j] = dot(Q[row, :], K[j, kv_head, :]) * scale
        for (int j = tid; j < seq_len_k; j += bdim) {
            const int kb = j * (num_heads_k * head_dim) + kv_base;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += Q[q_base + d] * K[kb + d];
            }
            scores[j] = dot * scale;
        }
        __syncthreads();

        // Pass 2a: block max of scores (strided partials, then tree reduction).
        float local_max = -1.0f / 0.0f;
        for (int j = tid; j < seq_len_k; j += bdim) {
            local_max = fmaxf(local_max, scores[j]);
        }
        red_max[tid] = local_max;
        __syncthreads();
        for (int s = bdim >> 1; s > 0; s >>= 1) {
            if (tid < s) {
                red_max[tid] = fmaxf(red_max[tid], red_max[tid + s]);
            }
            __syncthreads();
        }
        const float max_v = red_max[0];
        __syncthreads();

        // Pass 2b: exp(scores - max) and block sum.
        float local_sum = 0.0f;
        for (int j = tid; j < seq_len_k; j += bdim) {
            scores[j] = expf(scores[j] - max_v);
            local_sum += scores[j];
        }
        red_sum[tid] = local_sum;
        __syncthreads();
        for (int s = bdim >> 1; s > 0; s >>= 1) {
            if (tid < s) {
                red_sum[tid] += red_sum[tid + s];
            }
            __syncthreads();
        }
        const float inv_sum = (red_sum[0] > 0.0f) ? (1.0f / red_sum[0]) : 0.0f;
        __syncthreads();

        // Pass 3: each thread produces one output dim of the row.
        if (tid < head_dim) {
            const int o_idx = q_base + tid;
            float acc = 0.0f;
            for (int j = 0; j < seq_len_k; ++j) {
                acc += scores[j] * V[j * (num_heads_k * head_dim) + kv_base + tid];
            }
            out[o_idx] = acc * inv_sum;
        }
    }

}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_attention_source_contains_entries() {
        assert!(KERNEL_SOURCE.contains("grim_cross_attention"));
        assert!(KERNEL_SOURCE.contains("encoder"));
        assert!(KERNEL_SOURCE.contains("dot"));
    }
}
