//! FlashDecoding — split-KV parallel attention for long-context single-token decode.
//!
//! Ported verbatim from grim-backend-rocm `kernels/flash_decode.rs`.
//! Divides KV sequence into `num_splits` chunks, computes partial online
//! softmax in stage 1, merges in stage 2. Saturates all SMs on long contexts.
//!
//! CUDA delta from ROCm: `__syncthreads()` in place of `__syncwarp()`; no
//! wavefront-width assumptions. extern "C" wrapping for nvcc JIT symbol export.

pub const FLASH_DECODE_SOURCE: &str = r#"
extern "C" {

// ---------------------------------------------------------------------------
// FlashDecoding Stage 1 — Partial Online Softmax per Sequence Chunk
//
// Grid:  (num_heads, num_splits)
// Block: (head_dim threads, 1)  — one thread per head dimension element.
// Shared: (head_dim + blockDim.x) * 4 bytes.
// ---------------------------------------------------------------------------
__global__ void grim_flash_decode_stage1(
    const float* __restrict__ q,         // [num_heads, head_dim]
    const float* __restrict__ k_tensor,  // [kv_seq_len, num_kv_heads, head_dim]
    const float* __restrict__ v_tensor,  // [kv_seq_len, num_kv_heads, head_dim]
    float* __restrict__ mid_out,         // [num_splits, num_heads, head_dim]
    float* __restrict__ mid_max,         // [num_splits, num_heads]
    float* __restrict__ mid_sum,         // [num_splits, num_heads]
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int kv_seq_len,
    int num_splits,
    float inv_sqrt_d
) {
    const int h        = blockIdx.x;
    const int split_id = blockIdx.y;
    const int d        = threadIdx.x;

    if (h >= num_heads || split_id >= num_splits) return;

    const int q_per_kv  = num_heads / num_kv_heads;
    const int kv_h      = h / q_per_kv;
    const int chunk_sz  = (kv_seq_len + num_splits - 1) / num_splits;
    const int start_j   = split_id * chunk_sz;
    const int end_j     = (start_j + chunk_sz < kv_seq_len) ? start_j + chunk_sz : kv_seq_len;

    if (start_j >= kv_seq_len) {
        if (d < head_dim) mid_out[(split_id * num_heads + h) * head_dim + d] = 0.0f;
        if (d == 0) {
            mid_max[split_id * num_heads + h] = -1e20f;
            mid_sum[split_id * num_heads + h] =  0.0f;
        }
        return;
    }

    extern __shared__ float s_mem[];
    float* s_q   = s_mem;                // [head_dim]
    float* s_dot = s_mem + head_dim;     // [blockDim.x]

    if (d < head_dim) s_q[d] = q[h * head_dim + d];
    __syncthreads();

    float rmax = -1e20f, rsum = 0.0f, acc = 0.0f;

    for (int j = start_j; j < end_j; ++j) {
        const int kv_base = (j * num_kv_heads + kv_h) * head_dim;
        float local = (d < head_dim) ? s_q[d] * k_tensor[kv_base + d] : 0.0f;
        s_dot[threadIdx.x] = local;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) s_dot[threadIdx.x] += s_dot[threadIdx.x + stride];
            __syncthreads();
        }
        float score = s_dot[0] * inv_sqrt_d;
        __syncthreads();

        float new_max = (score > rmax) ? score : rmax;
        float alpha   = __expf(rmax - new_max);
        float beta    = __expf(score - new_max);
        rsum = rsum * alpha + beta;
        rmax = new_max;
        if (d < head_dim) acc = acc * alpha + beta * v_tensor[kv_base + d];
    }

    if (d < head_dim) mid_out[(split_id * num_heads + h) * head_dim + d] = acc;
    if (d == 0) {
        mid_max[split_id * num_heads + h] = rmax;
        mid_sum[split_id * num_heads + h] = rsum;
    }
}

// ---------------------------------------------------------------------------
// FlashDecoding Stage 2 — Merge Partials Across Splits
//
// Grid:  (num_heads, 1)
// Block: (head_dim threads, 1)
// ---------------------------------------------------------------------------
__global__ void grim_flash_decode_stage2(
    const float* __restrict__ mid_out,  // [num_splits, num_heads, head_dim]
    const float* __restrict__ mid_max,  // [num_splits, num_heads]
    const float* __restrict__ mid_sum,  // [num_splits, num_heads]
    float* __restrict__ out,            // [num_heads, head_dim]
    int num_heads,
    int head_dim,
    int num_splits
) {
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    if (h >= num_heads) return;

    float gmax = -1e20f;
    for (int s = 0; s < num_splits; ++s) {
        float m = mid_max[s * num_heads + h];
        if (m > gmax) gmax = m;
    }

    float gsum = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
        float l = mid_sum[s * num_heads + h];
        if (l > 0.0f) gsum += l * __expf(mid_max[s * num_heads + h] - gmax);
    }

    if (d < head_dim) {
        float final_acc = 0.0f;
        for (int s = 0; s < num_splits; ++s) {
            float l = mid_sum[s * num_heads + h];
            if (l > 0.0f) {
                float factor = __expf(mid_max[s * num_heads + h] - gmax);
                final_acc += mid_out[(s * num_heads + h) * head_dim + d] * factor;
            }
        }
        out[h * head_dim + d] = (gsum > 0.0f) ? final_acc / gsum : 0.0f;
    }
}

} // extern "C"
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_contains_both_stages() {
        assert!(FLASH_DECODE_SOURCE.contains("grim_flash_decode_stage1"));
        assert!(FLASH_DECODE_SOURCE.contains("grim_flash_decode_stage2"));
    }
}
