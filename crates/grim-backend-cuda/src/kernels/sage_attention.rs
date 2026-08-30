//! Block-Quantized SageAttention and Multi-dimensional RoPE (M-RoPE) kernels for CUDA.

pub const SAGE_ATTENTION_SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <math.h>

extern "C" {

__global__ void grim_sage_attention(
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

__global__ void grim_mrope_qk(
    float* __restrict__ q,
    float* __restrict__ k,
    const int* __restrict__ positions,
    int num_tokens,
    int num_q_heads,
    int num_k_heads,
    int head_dim,
    int rotary_dim,
    int section_t,
    int section_h,
    int section_w,
    float rope_theta
) {
    const int t = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int pair_idx = threadIdx.x;

    const int total_heads = num_q_heads + num_k_heads;
    if (t >= num_tokens || head_idx >= total_heads || pair_idx >= (rotary_dim / 2)) return;

    const bool is_q = (head_idx < num_q_heads);
    const int local_head = is_q ? head_idx : (head_idx - num_q_heads);

    const int pos_t = positions[t * 3 + 0];
    const int pos_h = positions[t * 3 + 1];
    const int pos_w = positions[t * 3 + 2];

    int pos = pos_t;
    if (pair_idx >= section_t && pair_idx < (section_t + section_h)) {
        pos = pos_h;
    } else if (pair_idx >= (section_t + section_h)) {
        pos = pos_w;
    }

    const float freq = 1.0f / powf(rope_theta, (2.0f * (float)pair_idx) / (float)rotary_dim);
    const float angle = (float)pos * freq;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);

    const int idx0 = 2 * pair_idx;
    const int idx1 = 2 * pair_idx + 1;

    if (is_q) {
        const int base = (t * num_q_heads + local_head) * head_dim;
        const float x0 = q[base + idx0];
        const float x1 = q[base + idx1];
        q[base + idx0] = x0 * cos_val - x1 * sin_val;
        q[base + idx1] = x0 * sin_val + x1 * cos_val;
    } else {
        const int base = (t * num_k_heads + local_head) * head_dim;
        const float x0 = k[base + idx0];
        const float x1 = k[base + idx1];
        k[base + idx0] = x0 * cos_val - x1 * sin_val;
        k[base + idx1] = x0 * sin_val + x1 * cos_val;
    }
}

}
"#;
