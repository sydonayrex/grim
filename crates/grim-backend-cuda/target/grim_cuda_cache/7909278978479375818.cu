
#include <cuda_fp16.h>
#include <math.h>

extern "C" __global__ void grim_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    c[i] = a[i] + b[i];
}

extern "C" __global__ void grim_mul(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    c[i] = a[i] * b[i];
}

extern "C" __global__ void grim_silu_mul(float* gate, float* up, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float s = g / (1.0f + expf(-g));
    out[i] = s * up[i];
}

extern "C" __global__ void grim_rms_norm(float* x, float* w, float* out,
                                         int row_len, float eps, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int row = idx / row_len;
    int col = idx % row_len;

    // Calculate mean of squares
    float ss = 0.0f;
    for (int j = 0; j < row_len; ++j) {
        float val = x[row * row_len + j];
        ss += val * val;
    }
    float rms = sqrtf(ss / (float)row_len + eps);
    out[idx] = x[idx] * w[col] / rms;
}

extern "C" __global__ void grim_softmax(float* x, float* out, int last_dim, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int row = idx / last_dim;

    float max_val = -1e30f;
    for (int j = 0; j < last_dim; ++j) {
        max_val = fmaxf(max_val, x[row * last_dim + j]);
    }

    float sum = 0.0f;
    for (int j = 0; j < last_dim; ++j) {
        sum += expf(x[row * last_dim + j] - max_val);
    }

    out[idx] = expf(x[idx] - max_val) / sum;
}

extern "C" __global__ void grim_embedding(const float* weight, const int* indices, float* out,
                                          int embedding_dim, int num_indices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_indices * embedding_dim;
    if (idx >= total) return;

    int token_idx = idx / embedding_dim;
    int embed_offset = idx % embedding_dim;

    int word_idx = indices[token_idx];
    out[idx] = weight[word_idx * embedding_dim + embed_offset];
}

extern "C" __global__ void grim_qkv_attention(
    const float* __restrict__ q,
    const float* __restrict__ k_tensor,
    const float* __restrict__ v_tensor,
    float* __restrict__ out,
    float* __restrict__ out_max,
    float* __restrict__ out_sum,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    int kv_seq_len,
    int cache_offset,
    float inv_sqrt_d
) {
    const int i = blockIdx.x;             // query position (0..seq_len)
    const int h = blockIdx.y;             // head index
    if (i >= seq_len || h >= num_heads) return;

    const int q_per_kv = num_heads / num_kv_heads;
    const int kv_head = h / q_per_kv;
    const int q_offset = (i * num_heads + h) * head_dim;
    const int abs_i = cache_offset + i;

    const int tid = threadIdx.x;
    const int wave_size = 32; // CUDA warp size is always 32
    const int wave_id = tid / wave_size;
    const int lane_id = tid % wave_size;
    const int num_waves = 256 / wave_size;

    if (head_dim > 256) {
        for (int chunk = 0; chunk < 8; ++chunk) {
            int d = lane_id + chunk * wave_size;
            if (d < head_dim) {
                out[q_offset + d] = nanf("");
            }
        }
        return;
    }

    __shared__ float s_max[8];
    __shared__ float s_sum[8];
    __shared__ float s_acc[8][256];

    const int hi = (abs_i < kv_seq_len) ? (abs_i + 1) : kv_seq_len;
    const int range_len = hi;

    const int base = range_len / num_waves;
    const int rem  = range_len % num_waves;
    int j_start = wave_id * base + (wave_id < rem ? wave_id : rem);
    int j_end   = j_start + base + (wave_id < rem ? 1 : 0);

    float out_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float running_max = -1e30f;
    float running_sum = 0.0f;

    const float* __restrict__ k_head = &k_tensor[kv_head * head_dim];
    const float* __restrict__ v_head = &v_tensor[kv_head * head_dim];

    for (int j = j_start; j < j_end; ++j) {
        float score = 0.0f;
        #pragma unroll
        for (int dim = 0; dim < 256; ++dim) {
            if (dim < head_dim) {
                score += q[q_offset + dim] * k_head[j * (num_kv_heads * head_dim) + dim];
            }
        }
        score *= inv_sqrt_d;

        const float old_max = running_max;
        running_max = fmaxf(running_max, score);
        const float scale_old = expf(old_max - running_max);
        const float scale_new = expf(score - running_max);

        running_sum = running_sum * scale_old + scale_new;
        for (int chunk = 0; chunk < 8; ++chunk) {
            int d = lane_id + chunk * wave_size;
            if (d < head_dim) {
                out_acc[chunk] = out_acc[chunk] * scale_old + scale_new * v_head[j * (num_kv_heads * head_dim) + d];
            }
        }
    }

    if (lane_id == 0) {
        s_max[wave_id] = running_max;
        s_sum[wave_id] = running_sum;
    }
    for (int chunk = 0; chunk < 8; ++chunk) {
        int d = lane_id + chunk * wave_size;
        if (d < head_dim) {
            s_acc[wave_id][d] = out_acc[chunk];
        } else if (d < 256) {
            s_acc[wave_id][d] = 0.0f;
        }
    }
    __syncthreads();

    if (wave_id != 0) return;

    for (int chunk = 0; chunk < 8; ++chunk) {
        int d = lane_id + chunk * wave_size;
        if (d < head_dim) {
            float m_final = s_max[0];
            float sum_final = s_sum[0];
            float acc_final = s_acc[0][d];
            #pragma unroll
            for (int w = 1; w < 8; ++w) {
                if (w >= num_waves) break;
                const float mw = s_max[w];
                const float uw = s_sum[w];
                const float aw = s_acc[w][d];
                const float m_new = fmaxf(m_final, mw);
                const float scale_a = expf(m_final - m_new);
                const float scale_b = expf(mw - m_new);
                sum_final = sum_final * scale_a + uw * scale_b;
                acc_final = acc_final * scale_a + aw * scale_b;
                m_final = m_new;
            }
            const float inv_sum = (sum_final > 0.0f) ? (1.0f / sum_final) : 0.0f;
            out[q_offset + d] = acc_final * inv_sum;
        }
    }
}

extern "C" __global__ void grim_mul_scalar(const float* x, float scalar, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = x[i] * scalar;
}

extern "C" __global__ void grim_sqrt(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = sqrtf(x[i]);
}

extern "C" __global__ void grim_recip(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = 1.0f / x[i];
}

extern "C" __global__ void grim_rope(const float* x, const int* pos, float* out, int num_tokens, int num_heads, int head_dim, float base) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    int total = num_tokens * num_heads * half_dim;
    if (idx >= total) return;

    int pair_idx = idx % half_dim;
    int head_idx = (idx / half_dim) % num_heads;
    int token_idx = idx / (half_dim * num_heads);

    int p = pos[token_idx];
    float freq = 1.0f / powf(base, (float)(2 * pair_idx) / (float)head_dim);
    float val = (float)p * freq;
    float cos_v = cosf(val);
    float sin_v = sinf(val);

    int base_offset = (token_idx * num_heads + head_idx) * head_dim;
    int i0 = base_offset + pair_idx;
    int i1 = base_offset + pair_idx + half_dim;

    float v0 = x[i0];
    float v1 = x[i1];

    out[i0] = v0 * cos_v - v1 * sin_v;
    out[i1] = v0 * sin_v + v1 * cos_v;
}

extern "C" __global__ void grim_quantized_matmul_q8_0(const float* a, const unsigned char* b, const float* b_scales, float* out, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;

    float sum = 0.0f;
    int blocks_per_row = k / 32;
    for (int b_idx = 0; b_idx < blocks_per_row; ++b_idx) {
        float scale = b_scales[col * blocks_per_row + b_idx];
        int b_offset = (col * blocks_per_row + b_idx) * 32;
        int a_offset = row * k + b_idx * 32;
        for (int i = 0; i < 32; ++i) {
            signed char q = (signed char)b[b_offset + i];
            sum += a[a_offset + i] * ((float)q * scale);
        }
    }
    out[row * n + col] = sum;
}
