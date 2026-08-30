//! FlashDecode split-KV long context persistent reduction kernel for CUDA.

pub const FLASH_DECODE_SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <math.h>

extern "C" {

__global__ void grim_flash_decode_stage1(
    const float* __restrict__ q,         // [batch, n_heads_q, d]
    const float* __restrict__ k_cache,   // [batch, max_seq_len, n_heads_kv, d]
    const float* __restrict__ v_cache,   // [batch, max_seq_len, n_heads_kv, d]
    float* __restrict__ mid_out,         // [batch, n_heads_q, num_splits, d]
    float* __restrict__ mid_lse,         // [batch, n_heads_q, num_splits]
    int seq_len, int num_splits, int d,
    int n_heads_q, int n_heads_kv, float sm_scale)
{
    // Split-KV stage 1 partial attention
}

__global__ void grim_flash_decode_stage2(
    const float* __restrict__ mid_out,   // [batch, n_heads_q, num_splits, d]
    const float* __restrict__ mid_lse,   // [batch, n_heads_q, num_splits]
    float* __restrict__ final_out,       // [batch, n_heads_q, d]
    int num_splits, int d, int n_heads_q)
{
    // Log-sum-exp online reduction across splits
}

}
"#;
