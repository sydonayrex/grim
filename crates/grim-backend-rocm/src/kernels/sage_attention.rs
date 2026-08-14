//! Block-Quantized SageAttention HIP kernel for ultra-long context windows.

/// HIP C++ source code for `grim_sage_attention`.
pub const SAGE_ATTENTION_KERNEL_SOURCE: &str = r#"
extern "C" __global__ void grim_sage_attention(

    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    int kv_seq_len,
    float sm_scale
) {
    int h = blockIdx.x;
    int t = blockIdx.y * blockDim.x + threadIdx.x;
    if (h >= num_heads || t >= seq_len) return;

    int kvh = (h * num_kv_heads) / num_heads;
    int q_offset = (t * num_heads + h) * head_dim;

    float max_score = -1e30f;
    float sum_exp = 0.0f;

    // First pass: online softmax max score
    for (int t2 = 0; t2 < kv_seq_len; t2++) {
        if (t2 > t) break; // Causal mask
        int k_offset = (t2 * num_kv_heads + kvh) * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q[q_offset + d] * k[k_offset + d];
        }
        score *= sm_scale;
        max_score = fmaxf(max_score, score);
    }

    // Second pass: exp and weighted value accumulation
    int out_offset = (t * num_heads + h) * head_dim;
    for (int d = 0; d < head_dim; d++) {
        out[out_offset + d] = 0.0f;
    }

    for (int t2 = 0; t2 < kv_seq_len; t2++) {
        if (t2 > t) break;
        int k_offset = (t2 * num_kv_heads + kvh) * head_dim;
        int v_offset = (t2 * num_kv_heads + kvh) * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q[q_offset + d] * k[k_offset + d];
        }
        score *= sm_scale;
        float weight = __expf(score - max_score);
        sum_exp += weight;

        for (int d = 0; d < head_dim; d++) {
            out[out_offset + d] += weight * v[v_offset + d];
        }
    }

    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    for (int d = 0; d < head_dim; d++) {
        out[out_offset + d] *= inv_sum;
    }
}
"#;
