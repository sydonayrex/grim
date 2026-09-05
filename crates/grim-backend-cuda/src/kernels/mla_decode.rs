//! DeepSeek Multi-Head Latent Attention (MLA) matrix-absorbed decode kernel.
//!
//! Ported from grim-backend-rocm `kernels/mla_decode.rs`.
//! Operates directly on the compressed 576-dim latent KV-cache
//! (512-dim c_kv + 64-dim k_pe) using pre-absorbed query projections
//! (Q_C = Q * W_UK) to eliminate per-head KV materialization.

pub const MLA_DECODE_SOURCE: &str = r#"
extern "C" {

// ---------------------------------------------------------------------------
// DeepSeek MLA Matrix-Absorbed Decode Kernel
//
// Grid:  (num_heads, 1)
// Block: (256, 1) or (512, 1) — threads stride over latent dimension.
// Shared: blockDim.x * 4 bytes for dot-product reduction.
//
// Contract:
//   kv_cache layout per token: [c_kv (kv_lora_rank), k_pe (qk_rope_dim)]
//   items_per_thread must be <= 8 (local register array bound).
// ---------------------------------------------------------------------------
__global__ void grim_mla_absorbed_decode(
    const float* __restrict__ q_absorbed, // [num_heads, kv_lora_rank]
    const float* __restrict__ q_rope,     // [num_heads, qk_rope_dim]
    const float* __restrict__ kv_cache,   // [seq_len, kv_lora_rank + qk_rope_dim]
    const float* __restrict__ w_uv,       // [v_head_dim, kv_lora_rank] or NULL
    float* __restrict__ out,              // [num_heads, v_head_dim] or [num_heads, kv_lora_rank]
    int num_heads,
    int kv_lora_rank,
    int qk_rope_dim,
    int v_head_dim,
    int seq_len,
    float inv_sqrt_d,
    int has_w_uv,
    int w_uv_offset_words,
    int w_uv_head_stride_words
) {
    const int h          = blockIdx.x;
    const int tid        = threadIdx.x;
    const int block_size = blockDim.x;
    if (h >= num_heads) return;

    const int total_kv_dim    = kv_lora_rank + qk_rope_dim;
    const int q_c_base        = h * kv_lora_rank;
    const int q_r_base        = h * qk_rope_dim;
    const int items_per_thread = (kv_lora_rank + block_size - 1) / block_size;

    extern __shared__ float s_dot[];

    float rmax = -1e20f, rsum = 0.0f;
    float acc_local[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc_local[i] = 0.0f;

    for (int j = 0; j < seq_len; ++j) {
        const int base = j * total_kv_dim;

        // Score = Q_C . c_kv + Q_R . k_pe
        float local_score = 0.0f;
        for (int c = tid; c < kv_lora_rank; c += block_size)
            local_score += q_absorbed[q_c_base + c] * kv_cache[base + c];
        for (int r = tid; r < qk_rope_dim; r += block_size)
            local_score += q_rope[q_r_base + r] * kv_cache[base + kv_lora_rank + r];

        s_dot[tid] = local_score;
        __syncthreads();
        for (int stride = block_size >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) s_dot[tid] += s_dot[tid + stride];
            __syncthreads();
        }
        float score = s_dot[0] * inv_sqrt_d;
        __syncthreads();

        float new_max = (score > rmax) ? score : rmax;
        float alpha   = __expf(rmax - new_max);
        float beta    = __expf(score - new_max);
        rsum = rsum * alpha + beta;
        rmax = new_max;

        for (int i = 0; i < items_per_thread; ++i) {
            int c = tid + i * block_size;
            if (c < kv_lora_rank)
                acc_local[i] = acc_local[i] * alpha + beta * kv_cache[base + c];
        }
    }

    const float inv_sum = (rsum > 0.0f) ? (1.0f / rsum) : 0.0f;

    if (has_w_uv && w_uv != nullptr) {
        __shared__ float s_latent[512];
        for (int i = 0; i < items_per_thread; ++i) {
            int c = tid + i * block_size;
            if (c < kv_lora_rank && c < 512) s_latent[c] = acc_local[i] * inv_sum;
        }
        __syncthreads();

        for (int v = tid; v < v_head_dim; v += block_size) {
            float va = 0.0f;
            for (int c = 0; c < kv_lora_rank && c < 512; ++c)
            long w_base = (long)w_uv_offset_words + (long)h * w_uv_head_stride_words;
            va += w_uv[w_base + v * kv_lora_rank + c] * s_latent[c];
            out[h * v_head_dim + v] = va;
        }
    } else {
        for (int i = 0; i < items_per_thread; ++i) {
            int c = tid + i * block_size;
            if (c < kv_lora_rank)
                out[h * kv_lora_rank + c] = acc_local[i] * inv_sum;
        }
    }
}

} // extern "C"
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_contains_mla_absorbed_decode() {
        assert!(MLA_DECODE_SOURCE.contains("grim_mla_absorbed_decode"));
    }
}
