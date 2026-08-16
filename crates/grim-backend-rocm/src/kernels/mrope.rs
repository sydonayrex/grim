//! Multimodal 3D Rotary Position Embedding (M-RoPE) for Vision-Language Models (Qwen-VL).
//!
//! Decomposes rotary dimensions into 3 coordinate channels: Temporal (T), Height (H), Width (W),
//! applying multi-dimensional rotary frequencies directly to Query and Key tensors.

pub const KERNEL_SOURCE: &str = r#"
extern "C" {

// ---------------------------------------------------------------------------
// 3D Multimodal Rotary Embedding (M-RoPE) for Q and K
// ---------------------------------------------------------------------------
//
// Grid: (num_tokens, num_q_heads + num_k_heads)
// Block: (rotary_dim / 2, 1)
// ---------------------------------------------------------------------------
__global__ void grim_mrope_qk(
    float* __restrict__ q,                // [num_tokens, num_q_heads, head_dim]
    float* __restrict__ k,                // [num_tokens, num_k_heads, head_dim]
    const int* __restrict__ positions,    // [num_tokens, 3] -> (T, H, W)
    int num_tokens,
    int num_q_heads,
    int num_k_heads,
    int head_dim,
    int rotary_dim,
    int section_t,                        // e.g. 16
    int section_h,                        // e.g. 24
    int section_w,                        // e.g. 24
    float rope_theta                      // e.g. 10000.0f
) {
    const int t = blockIdx.x; // token index
    const int head_idx = blockIdx.y; // head index across (Q + K)
    const int pair_idx = threadIdx.x; // pair index d in [0, rotary_dim / 2)

    const int total_heads = num_q_heads + num_k_heads;
    if (t >= num_tokens || head_idx >= total_heads || pair_idx >= (rotary_dim / 2)) return;

    const bool is_q = (head_idx < num_q_heads);
    const int local_head = is_q ? head_idx : (head_idx - num_q_heads);

    // Fetch 3D position coordinates for token t
    const int pos_t = positions[t * 3 + 0];
    const int pos_h = positions[t * 3 + 1];
    const int pos_w = positions[t * 3 + 2];

    // Select position coordinate based on rotary section
    int pos = pos_t;
    if (pair_idx >= section_t && pair_idx < (section_t + section_h)) {
        pos = pos_h;
    } else if (pair_idx >= (section_t + section_h)) {
        pos = pos_w;
    }

    // Compute inverse frequency: freq = 1.0 / (rope_theta ^ (2 * pair_idx / rotary_dim))
    const float freq = 1.0f / powf(rope_theta, (2.0f * (float)pair_idx) / (float)rotary_dim);
    const float angle = (float)pos * freq;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);

    // Apply rotation to pair (x0, x1)
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

} // extern "C"
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_source_contains_mrope() {
        assert!(KERNEL_SOURCE.contains("grim_mrope_qk"));
    }
}
