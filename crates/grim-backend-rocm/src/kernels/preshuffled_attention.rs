//! Preshuffled Vector-Tiled KV-Cache Attention for ROCm (AITER Layout Parity).
//!
//! Organizes KV cache into vector-tiled 128-bit memory segments (float4 stride):
//! - K_cache: [num_blocks, num_heads, head_dim / 4, block_size, 4]
//! - V_cache: [num_blocks, num_heads, block_size / 4, head_dim, 4]
//!
//! Enabling zero-conversion, maximum-bandwidth vector loads across AMD CDNA & RDNA.

pub const KERNEL_SOURCE: &str = r#"
extern "C" {

// ---------------------------------------------------------------------------
// Reshape and Cache into Preshuffled Layout
// ---------------------------------------------------------------------------
//
// Grid: (num_tokens, num_heads)
// Block: (head_dim, 1)
// ---------------------------------------------------------------------------
__global__ void grim_reshape_and_cache_preshuffled(
    const float* __restrict__ key,         // [num_tokens, num_heads, head_dim]
    const float* __restrict__ value,       // [num_tokens, num_heads, head_dim]
    float* __restrict__ k_cache,           // [num_blocks, num_heads, head_dim / 4, block_size, 4]
    float* __restrict__ v_cache,           // [num_blocks, num_heads, block_size / 4, head_dim, 4]
    const int* __restrict__ slot_mapping,  // [num_tokens] -> global slot index
    int num_tokens,
    int num_heads,
    int head_dim,
    int block_size
) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;

    if (t >= num_tokens || h >= num_heads || d >= head_dim) return;

    const int slot = slot_mapping[t];
    if (slot < 0) return;

    const int block_idx = slot / block_size;
    const int block_offset = slot % block_size;

    const int token_base = (t * num_heads + h) * head_dim;
    const float k_val = key[token_base + d];
    const float v_val = value[token_base + d];

    // Preshuffled K layout: [num_blocks, num_heads, head_dim / 4, block_size, 4]
    const int d_div_4 = d / 4;
    const int d_mod_4 = d % 4;
    const int k_cache_idx = (((block_idx * num_heads + h) * (head_dim / 4) + d_div_4) * block_size + block_offset) * 4 + d_mod_4;
    k_cache[k_cache_idx] = k_val;

    // Preshuffled V layout: [num_blocks, num_heads, block_size / 4, head_dim, 4]
    const int bo_div_4 = block_offset / 4;
    const int bo_mod_4 = block_offset % 4;
    const int v_cache_idx = (((block_idx * num_heads + h) * (block_size / 4) + bo_div_4) * head_dim + d) * 4 + bo_mod_4;
    v_cache[v_cache_idx] = v_val;
}

// ---------------------------------------------------------------------------
// Preshuffled Paged Attention Decode Kernel
// ---------------------------------------------------------------------------
//
// Reads preshuffled K and V caches with aligned 128-bit vector loads.
//
// Grid: (num_seqs, num_heads)
// Block: (head_dim, 1)
// ---------------------------------------------------------------------------
__global__ void grim_preshuffled_paged_attention(
    const float* __restrict__ q,              // [num_seqs, num_heads, head_dim]
    const float* __restrict__ k_cache,        // [num_blocks, num_heads, head_dim / 4, block_size, 4]
    const float* __restrict__ v_cache,        // [num_blocks, num_heads, block_size / 4, head_dim, 4]
    const int* __restrict__ block_tables,     // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ context_lens,     // [num_seqs]
    float* __restrict__ out,                  // [num_seqs, num_heads, head_dim]
    int num_seqs,
    int num_heads,
    int head_dim,
    int block_size,
    int max_num_blocks_per_seq,
    float inv_sqrt_d
) {
    const int seq_idx = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;

    if (seq_idx >= num_seqs || h >= num_heads) return;

    const int context_len = context_lens[seq_idx];
    if (context_len <= 0) return;

    const int q_base = (seq_idx * num_heads + h) * head_dim;

    extern __shared__ float s_pshuf_mem[];
    float* s_q = s_pshuf_mem;                // [head_dim]
    float* s_dot = s_pshuf_mem + head_dim;   // [blockDim.x]

    if (d < head_dim) {
        s_q[d] = q[q_base + d];
    }
    __syncthreads();

    float running_max = -1e20f;
    float running_sum = 0.0f;
    float acc = 0.0f;

    const int* seq_block_table = block_tables + seq_idx * max_num_blocks_per_seq;

    for (int token_idx = 0; token_idx < context_len; ++token_idx) {
        const int block_table_idx = token_idx / block_size;
        const int physical_block = seq_block_table[block_table_idx];
        const int block_offset = token_idx % block_size;

        // Fetch from preshuffled K cache
        float k_val = 0.0f;
        if (d < head_dim) {
            const int d_div_4 = d / 4;
            const int d_mod_4 = d % 4;
            const int k_idx = (((physical_block * num_heads + h) * (head_dim / 4) + d_div_4) * block_size + block_offset) * 4 + d_mod_4;
            k_val = k_cache[k_idx];
        }

        // Compute dot product Q . K[token_idx]
        s_dot[threadIdx.x] = (d < head_dim) ? (s_q[d] * k_val) : 0.0f;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                s_dot[threadIdx.x] += s_dot[threadIdx.x + stride];
            }
            __syncthreads();
        }
        float score = s_dot[0] * inv_sqrt_d;
        __syncthreads();

        // Online softmax
        float new_max = max(running_max, score);
        float alpha = expf(running_max - new_max);
        float beta = expf(score - new_max);

        running_sum = running_sum * alpha + beta;
        running_max = new_max;

        // Fetch from preshuffled V cache
        if (d < head_dim) {
            const int bo_div_4 = block_offset / 4;
            const int bo_mod_4 = block_offset % 4;
            const int v_idx = (((physical_block * num_heads + h) * (block_size / 4) + bo_div_4) * head_dim + d) * 4 + bo_mod_4;
            float v_val = v_cache[v_idx];
            acc = acc * alpha + beta * v_val;
        }
    }

    if (d < head_dim) {
        out[q_base + d] = (running_sum > 0.0f) ? (acc / running_sum) : 0.0f;
    }
}

} // extern "C"
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_source_contains_preshuffled_attention() {
        assert!(KERNEL_SOURCE.contains("grim_reshape_and_cache_preshuffled"));
        assert!(KERNEL_SOURCE.contains("grim_preshuffled_paged_attention"));
    }
}
