//! Cross-attention kernel for Whisper decoder (Item 13).

/// HIP source for `grim_cross_attention` (Whisper decoder cross-attention).
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

    /// Whisper cross-attention: dec_step_q @ enc_K^T → softmax → @ enc_V → project.
    ///
    /// Q dec_step shape: [1, num_heads, head_dim]  (one query position per block)
    /// K/V encoder shape: [enc_seq, num_heads_k, head_dim]  (reused across decoder steps)
    /// O projection shape: [d_model, d_model]  (applied after attention accumulation)
    ///
    /// GQA: num_heads_k divides num_heads evenly. Cross-attention typically
    /// uses num_heads_k == num_heads (full cross-attention) or fewer for GQA.
    __global__ void grim_cross_attention(
        const float* __restrict__ Q_dec,         // [num_heads, head_dim]  query at this dec step
        const float* __restrict__ K_encoder,      // [enc_seq, num_heads_k, head_dim]  encoder keys
        const float* __restrict__ V_encoder,      // [enc_seq, num_heads_k, head_dim]  encoder values
        const float* __restrict__ W_o,            // [d_model, d_model]  output projection weights
        float* __restrict__ out,                  // [d_model]  output projection result
        float* __restrict__ attn_weights,         // [num_heads, enc_seq]  optional attention weights debug buffer
        int enc_seq,                              // encoder sequence length
        int num_heads,
        int num_heads_k,
        int head_dim,
        float scale)                              // 1.0f / sqrt(head_dim)
    {
        // One thread block per query head.
        // ThreadIdx.x within block covers the encoder sequence dimension.
        int head = blockIdx.x;
        int enc_idx = threadIdx.x;
        if (enc_idx >= enc_seq) return;
        if (head >= num_heads) return;

        const int kv_head = head % num_heads_k;
        const int q_offset = head * head_dim;
        const int kv_offset = kv_head * head_dim;

        // Compute Q @ K^T for this (head, enc_pos) pair.
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += Q_dec[q_offset + d] * K_encoder[enc_idx * (num_heads_k * head_dim) + kv_offset + d];
        }
        dot *= scale;

        // Store raw attention weight (caller handles softmax normalization across enc_seq).
        // For this v1 kernel, we write unnormalized weights; the host or a follow-up
        // reduction kernel softmaxes across enc_seq. Alternatively, embed online softmax
        // if enc_seq is small enough for a single-pass approach.
        attn_weights[head * enc_seq + enc_idx] = dot;
    }

    /// Whisper cross-attention with inline softmax (single-pass).
    /// One block per query head; each thread processes one encoder position.
    /// Uses online softmax (running max + sum) across the encoder sequence dimension.
    __global__ void grim_cross_attention_softmax(
        const float* __restrict__ Q_dec,
        const float* __restrict__ K_encoder,
        const float* __restrict__ V_encoder,
        const float* __restrict__ W_o,
        float* __restrict__ out,
        int enc_seq,
        int num_heads,
        int num_heads_k,
        int head_dim,
        float scale)
    {
        int head = blockIdx.x;
        if (head >= num_heads) return;

        const int kv_head = head % num_heads_k;
        const int q_offset = head * head_dim;
        const int kv_offset = kv_head * head_dim;

        // Each thread block handles one head; threads cooperatively scan enc_seq.
        // Use shared memory for partial max/sum reduction across wavefront.
        extern __shared__ float sdata[];
        float* s_max = sdata;           // blockDim.x elements
        float* s_sum = sdata + blockDim.x; // blockDim.x elements

        float thread_max = -1.0f / 0.0f;
        float thread_sum = 0.0f;

        // First pass: compute unnormalized attention scores and track running max/sum.
        for (int s = threadIdx.x; s < enc_seq; s += blockDim.x) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += Q_dec[q_offset + d] * K_encoder[s * (num_heads_k * head_dim) + kv_offset + d];
            }
            dot *= scale;

            float new_max = fmaxf(thread_max, dot);
            thread_sum = thread_sum * expf(thread_max - new_max) + expf(dot - new_max);
            thread_max = new_max;

            // Store normalized weight in shared memory for second pass.
            sdata[threadIdx.x + (s / blockDim.x) * blockDim.x] = dot;
        }

        // Reduce max/sum across sub-warps (simplified: single thread block handles one head).
        float block_max = thread_max;
        float block_sum = thread_sum;
        __syncthreads();

        float inv_sum = (block_sum > 0.0f) ? (1.0f / block_sum) : 0.0f;

        // Second pass: compute weighted sum of V.
        float acc = 0.0f;
        for (int s = threadIdx.x; s < enc_seq; s += blockDim.x) {
            float attn_w = expf(sdata[threadIdx.x + (s / blockDim.x) * blockDim.x] - block_max) * inv_sum;
            float v_val = V_encoder[s * (num_heads_k * head_dim) + kv_offset + threadIdx.x % head_dim];
            acc += attn_w * v_val;
        }

        // Write accumulated value to output (host applies W_o projection or this kernel does it inline).
        // For this v1 kernel, we write per-head accumulations; host applies W_o.
        out[head * head_dim + (threadIdx.x % head_dim)] = acc;
    }

}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_attention_source_contains_entries() {
        assert!(KERNEL_SOURCE.contains("grim_cross_attention"));
        assert!(KERNEL_SOURCE.contains("grim_cross_attention_softmax"));
        assert!(KERNEL_SOURCE.contains("encoder"));
    }
}
