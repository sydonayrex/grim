//! Multi-Head Latent Attention (MLA for DeepSeek) decode kernels for CUDA.

pub const MLA_DECODE_SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <math.h>

extern "C" {

__global__ void grim_mla_decode_stage1(
    const float* __restrict__ q_lat,     // [batch, q_lora_rank]
    const float* __restrict__ kv_cache,  // [batch, max_seq_len, kv_lora_rank + qk_rope_head_dim]
    const float* __restrict__ w_uk,      // [n_heads, d_head, kv_lora_rank]
    const float* __restrict__ w_uv,      // [n_heads, d_head, kv_lora_rank]
    float* __restrict__ out,             // [batch, n_heads, d_head]
    int seq_len, int kv_lora_rank, int qk_rope_head_dim, int n_heads, int d_head)
{
    // DeepSeek MLA latent cache decompression and attention
}

}
"#;
