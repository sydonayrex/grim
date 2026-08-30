//! Preshuffled vector-tiled paged KV-cache attention for CUDA.
//!
//! Ported from grim-backend-rocm `kernels/preshuffled_attention.rs`.
//! Organizes KV cache into 128-bit vector-tiled memory segments:
//!   K_cache: [num_blocks, num_heads, head_dim/4, block_size, 4]
//!   V_cache: [num_blocks, num_heads, block_size/4, head_dim, 4]
//! Enables aligned 128-bit (float4) loads on any SM >= 7.0 (Volta+).

pub const PRESHUFFLED_ATTENTION_SOURCE: &str = r#"
extern "C" {

// ---------------------------------------------------------------------------
// grim_reshape_and_cache_preshuffled — ingest K/V into vector-tiled layout.
//
// Grid: (num_tokens, num_heads)  Block: (head_dim, 1)
// One thread per (token, head, dim_element). Writes K into the D/4 stride,
// V into the block_offset/4 stride.
// ---------------------------------------------------------------------------
__global__ void grim_reshape_and_cache_preshuffled(
    const float* __restrict__ key,        // [num_tokens, num_heads, head_dim]
    const float* __restrict__ value,      // [num_tokens, num_heads, head_dim]
    float* __restrict__ k_cache,          // [num_blocks, num_heads, head_dim/4, block_size, 4]
    float* __restrict__ v_cache,          // [num_blocks, num_heads, block_size/4, head_dim, 4]
    const int* __restrict__ slot_mapping, // [num_tokens] -> global slot index
    int num_tokens,
    int num_heads,
    int head_dim,
    int block_size
) {
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    if (t >= num_tokens || h >= num_heads || d >= head_dim) return;

    const int slot = slot_mapping[t];
    if (slot < 0) return;

    const int block_idx    = slot / block_size;
    const int block_offset = slot % block_size;
    const int token_base   = (t * num_heads + h) * head_dim;

    // K preshuffled: [num_blocks, num_heads, head_dim/4, block_size, 4]
    int d4 = d / 4, dm4 = d % 4;
    k_cache[(((block_idx * num_heads + h) * (head_dim / 4) + d4) * block_size + block_offset) * 4 + dm4] = key[token_base + d];

    // V preshuffled: [num_blocks, num_heads, block_size/4, head_dim, 4]
    int bo4 = block_offset / 4, bom4 = block_offset % 4;
    v_cache[(((block_idx * num_heads + h) * (block_size / 4) + bo4) * head_dim + d) * 4 + bom4] = value[token_base + d];
}

// ---------------------------------------------------------------------------
// grim_preshuffled_paged_attention — decode-time paged attention.
//
// Grid: (num_seqs, num_heads)  Block: (head_dim, 1)
// Shared: (head_dim + blockDim.x) * 4 bytes.
// Reads K/V from preshuffled cache via stride-aligned accesses.
// Online softmax with running max/sum; no second pass needed.
// ---------------------------------------------------------------------------
__global__ void grim_preshuffled_paged_attention(
    const float* __restrict__ q,            // [num_seqs, num_heads, head_dim]
    const float* __restrict__ k_cache,      // [num_blocks, num_heads, head_dim/4, block_size, 4]
    const float* __restrict__ v_cache,      // [num_blocks, num_heads, block_size/4, head_dim, 4]
    const int*   __restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq]
    const int*   __restrict__ context_lens, // [num_seqs]
    float* __restrict__ out,                // [num_seqs, num_heads, head_dim]
    int num_seqs,
    int num_heads,
    int head_dim,
    int block_size,
    int max_num_blocks_per_seq,
    float inv_sqrt_d
) {
    const int seq_idx = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    if (seq_idx >= num_seqs || h >= num_heads) return;

    const int ctx  = context_lens[seq_idx];
    if (ctx <= 0) return;

    const int q_base = (seq_idx * num_heads + h) * head_dim;
    extern __shared__ float s_mem[];
    float* s_q   = s_mem;
    float* s_dot = s_mem + head_dim;

    if (d < head_dim) s_q[d] = q[q_base + d];
    __syncthreads();

    const int* seq_bt = block_tables + seq_idx * max_num_blocks_per_seq;
    float rmax = -1e20f, rsum = 0.0f, acc = 0.0f;

    for (int tok = 0; tok < ctx; ++tok) {
        const int phys = seq_bt[tok / block_size];
        const int boff = tok % block_size;

        // Load K from preshuffled layout
        float k_val = 0.0f;
        if (d < head_dim) {
            int d4 = d / 4, dm4 = d % 4;
            k_val = k_cache[(((phys * num_heads + h) * (head_dim / 4) + d4) * block_size + boff) * 4 + dm4];
        }
        s_dot[threadIdx.x] = (d < head_dim) ? (s_q[d] * k_val) : 0.0f;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) s_dot[threadIdx.x] += s_dot[threadIdx.x + stride];
            __syncthreads();
        }
        float score = s_dot[0] * inv_sqrt_d;
        __syncthreads();

        float new_max = (score > rmax) ? score : rmax;
        float alpha   = expf(rmax - new_max), beta = expf(score - new_max);
        rsum = rsum * alpha + beta;
        rmax = new_max;

        // Load V from preshuffled layout and accumulate
        if (d < head_dim) {
            int bo4 = boff / 4, bom4 = boff % 4;
            float v_val = v_cache[(((phys * num_heads + h) * (block_size / 4) + bo4) * head_dim + d) * 4 + bom4];
            acc = acc * alpha + beta * v_val;
        }
    }
    if (d < head_dim)
        out[q_base + d] = (rsum > 0.0f) ? acc / rsum : 0.0f;
}

} // extern "C"
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_contains_both_preshuffled_kernels() {
        assert!(PRESHUFFLED_ATTENTION_SOURCE.contains("grim_reshape_and_cache_preshuffled"));
        assert!(PRESHUFFLED_ATTENTION_SOURCE.contains("grim_preshuffled_paged_attention"));
    }
}
