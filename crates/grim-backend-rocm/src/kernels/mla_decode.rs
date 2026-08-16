//! DeepSeek Multi-Head Latent Attention (MLA) Matrix-Absorbed Decode kernel.
//!
//! Performs decode attention directly against the compressed 576-dim latent
//! KV-cache (512-dim `c_kv` + 64-dim `k_pe`), using pre-absorbed query projections
//! ($Q_C = Q \cdot W^{UK}$) to eliminate multi-head Key/Value materialization.

pub const KERNEL_SOURCE: &str = r#"
extern "C" {

// ---------------------------------------------------------------------------
// DeepSeek MLA Matrix-Absorbed Decode Kernel
// ---------------------------------------------------------------------------
//
// Grid: (num_heads, 1)
// Block: (256, 1) or (512, 1) — thread index mapped across latent dimension.
// ---------------------------------------------------------------------------
__global__ void grim_mla_absorbed_decode(
    const float* __restrict__ q_absorbed, // [num_heads, kv_lora_rank] (Q * W_UK)
    const float* __restrict__ q_rope,     // [num_heads, qk_rope_dim]
    const float* __restrict__ kv_cache,   // [seq_len, kv_lora_rank + qk_rope_dim]
    const float* __restrict__ w_uv,       // [v_head_dim, kv_lora_rank] (optional in-kernel projection)
    float* __restrict__ out,              // [num_heads, v_head_dim] or [num_heads, kv_lora_rank]
    int num_heads,
    int kv_lora_rank,                     // e.g. 512
    int qk_rope_dim,                      // e.g. 64
    int v_head_dim,                       // e.g. 128
    int seq_len,
    float inv_sqrt_d,
    int has_w_uv
) {
    const int h = blockIdx.x; // query head index
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    if (h >= num_heads) return;

    const int total_kv_dim = kv_lora_rank + qk_rope_dim;
    const int q_c_base = h * kv_lora_rank;
    const int q_r_base = h * qk_rope_dim;

    extern __shared__ float s_mla_mem[];
    float* s_dot = s_mla_mem; // [block_size]

    float running_max = -1e20f;
    float running_sum = 0.0f;

    // Per-thread accumulator for the latent KV vector
    // Sized for kv_lora_rank / block_size elements per thread
    float acc_local[8];
    const int items_per_thread = (kv_lora_rank + block_size - 1) / block_size;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        acc_local[i] = 0.0f;
    }

    for (int j = 0; j < seq_len; ++j) {
        const int token_base = j * total_kv_dim;

        // 1. Partial dot product Q_C . c_kv
        float local_score = 0.0f;
        for (int c = tid; c < kv_lora_rank; c += block_size) {
            local_score += q_absorbed[q_c_base + c] * kv_cache[token_base + c];
        }
        // 2. Partial dot product Q_R . k_pe
        for (int r = tid; r < qk_rope_dim; r += block_size) {
            local_score += q_rope[q_r_base + r] * kv_cache[token_base + kv_lora_rank + r];
        }

        s_dot[tid] = local_score;
        __syncthreads();

        for (int stride = block_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_dot[tid] += s_dot[tid + stride];
            }
            __syncthreads();
        }
        float score = s_dot[0] * inv_sqrt_d;
        __syncthreads();

        // 3. Online Softmax update
        float new_max = max(running_max, score);
        float alpha = expf(running_max - new_max);
        float beta = expf(score - new_max);

        running_sum = running_sum * alpha + beta;
        running_max = new_max;

        // 4. Update latent value accumulator (c_kv)
        for (int i = 0; i < items_per_thread; ++i) {
            int c = tid + i * block_size;
            if (c < kv_lora_rank) {
                float kv_val = kv_cache[token_base + c];
                acc_local[i] = acc_local[i] * alpha + beta * kv_val;
            }
        }
    }

    // Normalize accumulated latent vector
    const float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;

    if (has_w_uv && w_uv != nullptr) {
        // Shared memory to hold normalized latent vector
        __shared__ float s_latent[512];
        for (int i = 0; i < items_per_thread; ++i) {
            int c = tid + i * block_size;
            if (c < kv_lora_rank && c < 512) {
                s_latent[c] = acc_local[i] * inv_sum;
            }
        }
        __syncthreads();

        // Matrix multiply: out[h, v] = W_UV[v, c] * s_latent[c]
        for (int v = tid; v < v_head_dim; v += block_size) {
            float v_acc = 0.0f;
            for (int c = 0; c < kv_lora_rank && c < 512; ++c) {
                v_acc += w_uv[v * kv_lora_rank + c] * s_latent[c];
            }
            out[h * v_head_dim + v] = v_acc;
        }
    } else {
        // Output the normalized latent vector directly [num_heads, kv_lora_rank]
        for (int i = 0; i < items_per_thread; ++i) {
            int c = tid + i * block_size;
            if (c < kv_lora_rank) {
                out[h * kv_lora_rank + c] = acc_local[i] * inv_sum;
            }
        }
    }
}

} // extern "C"
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_source_contains_mla_absorbed_decode() {
        assert!(KERNEL_SOURCE.contains("grim_mla_absorbed_decode"));
    }
}
