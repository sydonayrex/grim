//! CUDA kernel source strings for Grim.

pub const KERNELS_SOURCE: &str = r#"
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
    float s = 1.0f / (1.0f + expf(-g));
    out[i] = g * s * up[i];
}

extern "C" __global__ void grim_silu_mul_backward(
    const float* gate,
    const float* up,
    const float* dw,
    float* df,
    float* de,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float s = 1.0f / (1.0f + expf(-g));
    float silu = g * s;
    float dsilu = s * (1.0f + g * (1.0f - s));
    df[i] = silu * dw[i];
    de[i] = up[i] * dsilu * dw[i];
}

extern "C" __global__ void grim_rms_norm(float* x, float* w, float* out,
                                         int row_len, float eps, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int row = idx / row_len;
    int col = idx % row_len;

    // mean of squares
    float ss = 0.0f;
    for (int j = 0; j < row_len; ++j) {
        float val = x[row * row_len + j];
        ss += val * val;
    }
    float rms = sqrtf(ss / (float)row_len + eps);
    out[idx] = x[idx] * w[col] / rms;
}

// Fused Add + RMSNorm: y_out = x + residual; norm_out = rms_norm(y_out, w, eps).
// Mirrors the ROCm `grim_add_rms_norm` HIP kernel and the Metal `grim_add_rms_norm` MSL shader
// 1:1 — no cross-invocation shared memory; one thread per output element.
extern "C" __global__ void grim_add_rms_norm(const float* x, const float* residual,
                                             float* w, float* y_out, float* norm_out,
                                             int row_len, float eps, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int row = idx / row_len;
    int col = idx - row * row_len;

    // y = x + residual
    float y_val = x[idx] + residual[idx];
    y_out[idx] = y_val;

    // RMSNorm of y over this row
    float ss = 0.0f;
    for (int j = 0; j < row_len; ++j) {
        float v = x[row * row_len + j] + residual[row * row_len + j];
        ss += v * v;
    }
    float rms = sqrtf(ss / (float)row_len + eps);
    norm_out[idx] = y_val * w[col] / rms;
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
    float inv_sqrt_d,
    int window_lo       // sliding-window lower bound: max(0, abs_i - window + 1).
                        // Pass 0 for full causal attention (no window).
) {
    const int i = blockIdx.x;             // query pos (0..seq_len)
    const int h = blockIdx.y;             // head idx
    if (i >= seq_len || h >= num_heads) return;

    const int q_per_kv = num_heads / num_kv_heads;
    const int kv_head = h / q_per_kv;
    const int q_offset = (i * num_heads + h) * head_dim;
    const int abs_i = cache_offset + i;

    const int tid = threadIdx.x;
    const int wave_size = 32; // warp size
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
    const int lo = window_lo;             // 0 for full causal; >= 0 for SWA
    const int range_len = hi - lo;        // may be 0 if lo >= hi (empty window)

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
        // j is an offset into [0, range_len); the real KV index is j + lo.
        const int kj = j + lo;
        float score = 0.0f;
        #pragma unroll
        for (int dim = 0; dim < 256; ++dim) {
            if (dim < head_dim) {
                score += q[q_offset + dim] * k_head[kj * (num_kv_heads * head_dim) + dim];
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
                out_acc[chunk] = out_acc[chunk] * scale_old + scale_new * v_head[kj * (num_kv_heads * head_dim) + d];
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
    int i0 = base_offset + 2 * pair_idx;
    int i1 = base_offset + 2 * pair_idx + 1;

    float v0 = x[i0];
    float v1 = x[i1];

    out[i0] = v0 * cos_v - v1 * sin_v;
    out[i1] = v0 * sin_v + v1 * cos_v;
}

// Partial-rotary + YaRN kernel. Mirrors the ROCm `grim_rope_yarn` so the two
// devices produce bit-identical results for the same input.
//
// CONTRACT:
//   x        – [B, S, D] f32 input
//   positions– [S] absolute token positions
//   inv_freq – [rotary_half] pre-computed YaRN / plain inv-frequencies
//   out      – [B, S, D] f32 output (non-rotary dims copied verbatim)
//   b, s, d  – batch / seq / full head dim
//   rotary_half – half of rotary_dim (= rotary_dim/2); dims [rotary_half, d)
//               are NOT rotated (copied verbatim)
//   mscale   – attention_factor (1.0 for plain RoPE; YaRN sets this)
//
// One thread per (batch, step, rotary-pair). Non-rotary dims are handled
// by a second pass over the copy range [rotary_dim, d). The launch grid
// (sized by the host) covers max(b*s*rotary_half, b*s*copy_len) threads so
// both passes are handled in a single launch.
extern "C" __global__ void grim_rope_yarn(
    const float* __restrict__ x,
    const unsigned int* __restrict__ positions,
    const float* __restrict__ inv_freq,
    float* __restrict__ out,
    int b, int s, int d, int rotary_half, float mscale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Pass 1: rotate the [0, rotary_half) pairs (interleaved layout: (2i, 2i+1)).
    int total = b * s * rotary_half;
    if (idx < total) {
        int bi = idx / (s * rotary_half);
        int rem = idx - bi * (s * rotary_half);
        int si = rem / rotary_half;
        int i  = rem - si * rotary_half;
        float pos = (float)positions[si];
        float val = pos * inv_freq[i];
        float sin_val = sinf(val) * mscale;
        float cos_val = cosf(val) * mscale;
        int base_idx = (bi * s + si) * d;
        int a_idx = base_idx + 2 * i;
        int b_idx = base_idx + 2 * i + 1;
        float x1 = x[a_idx];
        float x2 = x[b_idx];
        out[a_idx] = x1 * cos_val - x2 * sin_val;
        out[b_idx] = x1 * sin_val + x2 * cos_val;
    }

    // Pass 2: copy the non-rotary dims [2*rotary_half, d) verbatim. The same
    // thread pool is reused; threads with idx in [0, b*s*(d-2*rotary_half))
    // handle the copy dimension. For full rotary (rotary_dim == d) copy_len
    // is 0 and this pass is a no-op.
    int copy_start = 2 * rotary_half;
    int copy_len   = d - copy_start;
    if (copy_len > 0) {
        int total2 = b * s * copy_len;
        if (idx < total2) {
            int bi = idx / (s * copy_len);
            int rem = idx - bi * (s * copy_len);
            int si = rem / copy_len;
            int ci = rem - si * copy_len;
            int src_idx = (bi * s + si) * d + copy_start + ci;
            out[src_idx] = x[src_idx];
        }
    }
}



extern "C" __global__ void grim_quantized_matmul_q8_0(const float* a, const unsigned char* b, const float* b_scales, float* out, int m, int n, int k, int b_data_offset) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;

    float sum = 0.0f;
    int blocks_per_row = k / 32;
    // b_data_offset=2 → real Q8_0 packed layout (34 bytes/block with f16 header)
    // b_data_offset=0 → simplified layout (32 bytes/block raw i8 codes)
    int block_stride = 32 + b_data_offset;
    for (int b_idx = 0; b_idx < blocks_per_row; ++b_idx) {
        float scale = b_scales[col * blocks_per_row + b_idx];
        int b_offset = (col * blocks_per_row + b_idx) * block_stride + b_data_offset;
        int a_offset = row * k + b_idx * 32;
        for (int i = 0; i < 32; ++i) {
            signed char q = (signed char)b[b_offset + i];
            sum += a[a_offset + i] * ((float)q * scale);
        }
    }
    out[row * n + col] = sum;
}

// ===========================================================================
//  Standalone dequantization kernels (one thread per 256-weight super-block,
//  except where noted). These are bit-accurate ports of the CPU reference in
//  `crates/grim-quant/src/lib.rs` — NOT the ROCm `iq_dequant.rs` device fns,
//  which are simplified and do not match the CPU oracle (see audit notes).
//
//  Shared device helpers reproduced from grim-quant with the bit-accurate
//  f16 subnormal path (the ROCm `fp16_to_float_device` rounds subnormals to a
//  different scale; the CPU oracle uses `mant * 2^-24`).
// ===========================================================================

__device__ inline float grim_f16_to_f32(unsigned short h) {
    unsigned int sign = (h >> 15) & 1u;
    unsigned int exp  = (h >> 10) & 0x1Fu;
    unsigned int mant = h & 0x3FFu;
    if (exp == 0u) {
        // Subnormal or zero: value = ± mant * 2^-24 (bit-accurate vs
        // grim_quant::f16_to_f32, lib.rs:1372). __int_as_float(0x33800000)
        // = 2^-24 exactly; multiplying by the integer mantissa keeps the
        // rounding identical to the CPU `mant as f32 * 2f32.powi(-24)`.
        float value = (float)mant * __int_as_float(0x33800000u);
        return sign ? -value : value;
    } else if (exp == 31u) {
        // NaN/Inf: rebuild the f32 bit pattern (matches CPU f32::from_bits).
        unsigned int bits = (sign << 31) | 0x7F800000u | (mant << 13);
        return __int_as_float(bits);
    }
    unsigned int bits = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
    return __int_as_float(bits);
}

__device__ inline float grim_fp8_e4m3_to_f32(unsigned char val) {
    int sign = (val >> 7) & 1;
    int exp  = (val >> 3) & 0x0F;
    int mant = val & 0x07;
    if (exp == 0xF) {
        if (mant == 7) return __int_as_float(0x7FC00000u); // NaN
        float v = 448.0f;
        return sign ? -v : v;
    }
    float result;
    if (exp != 0) {
        result = (1.0f + (float)mant / 8.0f) * powf(2.0f, (float)exp - 7.0f);
    } else {
        result = (float)mant / 512.0f;
    }
    return sign ? -result : result;
}

__device__ inline float grim_mxfp4_to_f32(unsigned char code, unsigned char shared_exp) {
    int sign = (code >> 3) & 1;
    int exp  = (code >> 1) & 3;
    int mant = code & 1;
    float base_val;
    if (exp == 0) {
        base_val = (float)mant * 0.5f;
    } else {
        base_val = (1.0f + (float)mant * 0.5f) * powf(2.0f, (float)exp - 1.0f);
    }
    float signed_val = sign ? -base_val : base_val;
    float scale = powf(2.0f, (float)shared_exp - 127.0f);
    return signed_val * scale;
}

// K-quant 6-bit scale/min unpacker (grim_quant::get_scale_min_k4, lib.rs:674).
__device__ inline void grim_get_scale_min_k4(int j, const unsigned char* scales,
                                             float* sc, float* m) {
    unsigned char s, mm;
    if (j < 4) {
        s  = scales[j]     & 63;
        mm = scales[j + 4] & 63;
    } else {
        s  = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        mm = (scales[j + 4] >> 4)    | ((scales[j]     >> 6) << 4);
    }
    *sc = (float)s;
    *m  = (float)mm;
}

// IQ4_NL / IQ4_XS shared 16-entry absolute-magnitude codebook
// (grim_quant::IQ4_NL_CODEBOOK, lib.rs:227).
__device__ const float GRIM_IQ4_NL_CODEBOOK[16] = {
    0.0f,        0.11314126f, 0.24373604f, 0.39743365f,
    0.56574355f, 0.72294140f, 0.89705455f, 1.07576285f,
    1.29459881f, 1.52851904f, 1.82685633f, 2.27001130f,
    3.23719119f, 5.50829601f, 10.4162559f, 34.5695092f
};

extern "C" __global__ void grim_fused_quant_gemm_q4_k(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    float* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    int blocks_per_row = K / 256;
    float sum = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const unsigned char* blk = B_packed + (col * blocks_per_row + b) * 144;
        const unsigned short* h_ptr = (const unsigned short*)blk;
        float d    = grim_f16_to_f32(h_ptr[0]);
        float dmin = grim_f16_to_f32(h_ptr[1]);
        const unsigned char* scales = blk + 4;
        const unsigned char* qs     = blk + 16;

        const float* a_ptr = A + row * K + b * 256;

        int qs_idx = 0, is = 0;
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            float sc1, m1, sc2, m2;
            grim_get_scale_min_k4(is + 0, scales, &sc1, &m1);
            grim_get_scale_min_k4(is + 1, scales, &sc2, &m2);

            float d_sc1 = d * sc1, d_m1 = dmin * m1;
            float d_sc2 = d * sc2, d_m2 = dmin * m2;

            int a_off = n * 64;
            #pragma unroll
            for (int l = 0; l < 32; ++l) {
                float q1 = (float)(qs[qs_idx + l] & 0x0F);
                float q2 = (float)(qs[qs_idx + l] >> 4);

                sum += a_ptr[a_off + l     ] * (d_sc1 * q1 - d_m1);
                sum += a_ptr[a_off + l + 32] * (d_sc2 * q2 - d_m2);
            }
            qs_idx += 32;
            is     += 2;
        }
    }

    C[row * N + col] = sum;
}

extern "C" __global__ void grim_fused_quant_gemm_q5_k(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    float* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    int blocks_per_row = K / 256;
    float sum = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const unsigned char* blk = B_packed + (col * blocks_per_row + b) * 176;
        const unsigned short* h_ptr = (const unsigned short*)blk;
        float d    = grim_f16_to_f32(h_ptr[0]);
        float dmin = grim_f16_to_f32(h_ptr[1]);
        const unsigned char* scales = blk + 4;
        const unsigned char* qh     = blk + 16;
        const unsigned char* qs     = blk + 48;

        const float* a_ptr = A + row * K + b * 256;

        // ggml layout: four 64-weight groups; low nibbles of qs[n*32 + l]
        // (high bit qh[l] & u1, scale 2n) then high nibbles (bit u2, scale 2n+1).
        int qs_idx = 0, is = 0;
        unsigned char u1 = 1, u2 = 2;
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            float sc1, m1, sc2, m2;
            grim_get_scale_min_k4(is + 0, scales, &sc1, &m1);
            grim_get_scale_min_k4(is + 1, scales, &sc2, &m2);

            float d_sc1 = d * sc1, d_m1 = dmin * m1;
            float d_sc2 = d * sc2, d_m2 = dmin * m2;

            int a_off = n * 64;
            #pragma unroll
            for (int l = 0; l < 32; ++l) {
                float q1 = (float)((qs[qs_idx + l] & 0x0F) + ((qh[l] & u1) ? 16 : 0));
                float q2 = (float)((qs[qs_idx + l] >> 4)   + ((qh[l] & u2) ? 16 : 0));

                sum += a_ptr[a_off + l     ] * (d_sc1 * q1 - d_m1);
                sum += a_ptr[a_off + l + 32] * (d_sc2 * q2 - d_m2);
            }
            qs_idx += 32;
            is     += 2;
            u1    <<= 2;
            u2    <<= 2;
        }
    }

    C[row * N + col] = sum;
}

extern "C" __global__ void grim_fused_quant_gemm_q6_k(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    float* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    int blocks_per_row = K / 256;
    float sum = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const unsigned char* blk = B_packed + (col * blocks_per_row + b) * 210;
        // ggml block_q6_K layout: ql (128B) + qh (64B) + scales (16B) + d (f16, LAST @208).
        const unsigned char* ql     = blk;             // 128 bytes @0
        const unsigned char* qh     = blk + 128;       // 64 bytes  @128
        const unsigned char* scales = blk + 192;       // 16 bytes  @192 (i8)
        float d = grim_f16_to_f32(*((const unsigned short*)(blk + 208)));

        const float* a_ptr = A + row * K + b * 256;

        int sc_idx = 0, ql_idx = 0, qh_idx = 0;
        #pragma unroll
        for (int n = 0; n < 2; ++n) {
            int a_off = n * 128;
            #pragma unroll
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                float q1 = (float)((ql[ql_idx + l      ] & 0x0F) | ((qh[qh_idx + l] & 0x03) << 4)) - 32.0f;
                float q2 = (float)((ql[ql_idx + l + 32 ] & 0x0F) | ((qh[qh_idx + l] & 0x0C) << 2)) - 32.0f;
                float q3 = (float)((ql[ql_idx + l      ] >> 4)   | ((qh[qh_idx + l] & 0x30) >> 0)) - 32.0f;
                float q4 = (float)((ql[ql_idx + l + 32 ] >> 4)   | ((qh[qh_idx + l] & 0xC0) >> 2)) - 32.0f;
                float sc1 = (float)((signed char)scales[sc_idx + is + 0]);
                float sc2 = (float)((signed char)scales[sc_idx + is + 2]);
                float sc3 = (float)((signed char)scales[sc_idx + is + 4]);
                float sc4 = (float)((signed char)scales[sc_idx + is + 6]);

                sum += a_ptr[a_off + l      ] * (d * sc1 * q1);
                sum += a_ptr[a_off + l + 32 ] * (d * sc2 * q2);
                sum += a_ptr[a_off + l + 64 ] * (d * sc3 * q3);
                sum += a_ptr[a_off + l + 96 ] * (d * sc4 * q4);
            }
            ql_idx += 64;
            qh_idx += 32;
            sc_idx += 8;
        }
    }

    C[row * N + col] = sum;
}

extern "C" __global__ void grim_fused_quant_gemm_iq4nl(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    float* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    int blocks_per_row = K / 256;
    float sum = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const unsigned char* blk = B_packed + (col * blocks_per_row + b) * 170;
        float d = grim_f16_to_f32(*((const unsigned short*)blk));
        const unsigned char* q8 = blk + 2;
        const unsigned char* q4 = blk + 34;
        const unsigned char* scales = blk + 162;

        const float* a_ptr = A + row * K + b * 256;

        #pragma unroll
        for (int ib = 0; ib < 8; ++ib) {
            unsigned char sc = scales[ib];
            float scale_val = d * (float)(sc & 0x7F);
            int a_off = ib * 32;
            const unsigned char* q4_sub = q4 + ib * 16;
            const unsigned char* q8_sub = q8 + ib * 4;

            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                unsigned char byte_v = q4_sub[i];
                int nib_lo = byte_v & 0x0F;
                int nib_hi = byte_v >> 4;

                int sign_lo = (q8_sub[i / 4] >> ((i % 4) * 2)) & 1;
                int sign_hi = (q8_sub[i / 4] >> ((i % 4) * 2 + 1)) & 1;

                float val_lo = GRIM_IQ4_NL_CODEBOOK[nib_lo];
                float val_hi = GRIM_IQ4_NL_CODEBOOK[nib_hi];
                if (sign_lo) val_lo = -val_lo;
                if (sign_hi) val_hi = -val_hi;

                sum += a_ptr[a_off + i] * (scale_val * val_lo);
                sum += a_ptr[a_off + i + 16] * (scale_val * val_hi);
            }
        }
    }

    C[row * N + col] = sum;
}

extern "C" __global__ void grim_fused_quant_gemm_iq4xs(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    float* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    int blocks_per_row = K / 256;
    float sum = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const unsigned char* blk = B_packed + (col * blocks_per_row + b) * 178;
        float d = grim_f16_to_f32(*((const unsigned short*)blk));
        const unsigned char* q8 = blk + 2;
        const unsigned char* q4 = blk + 34;
        const unsigned char* scales = blk + 162;

        const float* a_ptr = A + row * K + b * 256;

        #pragma unroll
        for (int ib = 0; ib < 8; ++ib) {
            unsigned char sc = scales[ib];
            float scale_val = d * (float)(sc & 0x7F);
            int a_off = ib * 32;
            const unsigned char* q4_sub = q4 + ib * 16;
            const unsigned char* q8_sub = q8 + ib * 4;

            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                unsigned char byte_v = q4_sub[i];
                int nib_lo = byte_v & 0x0F;
                int nib_hi = byte_v >> 4;

                int sign_lo = (q8_sub[i / 4] >> ((i % 4) * 2)) & 1;
                int sign_hi = (q8_sub[i / 4] >> ((i % 4) * 2 + 1)) & 1;

                float val_lo = GRIM_IQ4_NL_CODEBOOK[nib_lo];
                float val_hi = GRIM_IQ4_NL_CODEBOOK[nib_hi];
                if (sign_lo) val_lo = -val_lo;
                if (sign_hi) val_hi = -val_hi;

                sum += a_ptr[a_off + i] * (scale_val * val_lo);
                sum += a_ptr[a_off + i + 16] * (scale_val * val_hi);
            }
        }
    }

    C[row * N + col] = sum;
}

extern "C" __global__ void grim_fused_quant_gemm_nvfp4(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    float* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    int blocks_per_row = K / 16;
    float sum = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const unsigned char* blk = B_packed + (col * blocks_per_row + b) * 10;
        const unsigned short* s_ptr = (const unsigned short*)blk;
        float scale = grim_f16_to_f32(s_ptr[0]);
        const unsigned char* codes = blk + 2;

        const float* a_ptr = A + row * K + b * 16;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            unsigned char c_byte = codes[i];
            unsigned char code_lo = c_byte & 0x0F;
            unsigned char code_hi = c_byte >> 4;

            float w_lo = grim_mxfp4_to_f32(code_lo, 127) * scale;
            float w_hi = grim_mxfp4_to_f32(code_hi, 127) * scale;

            sum += a_ptr[i * 2] * w_lo;
            sum += a_ptr[i * 2 + 1] * w_hi;
        }
    }

    C[row * N + col] = sum;
}

extern "C" __global__ void grim_fused_quant_gemm_mxfp4(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    float* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    int blocks_per_row = K / 32;
    float sum = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const unsigned char* blk = B_packed + (col * blocks_per_row + b) * 17;
        unsigned char shared_exp = blk[0];
        const unsigned char* codes = blk + 1;
        const float* a_ptr = A + row * K + b * 32;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            unsigned char c_byte = codes[i];
            unsigned char code_lo = c_byte & 0x0F;
            unsigned char code_hi = c_byte >> 4;

            float w_lo = grim_mxfp4_to_f32(code_lo, shared_exp);
            float w_hi = grim_mxfp4_to_f32(code_hi, shared_exp);

            sum += a_ptr[i * 2] * w_lo;
            sum += a_ptr[i * 2 + 1] * w_hi;
        }
    }

    C[row * N + col] = sum;
}

// ---- Q5_K (176 B / 256 weights) — bit-accurate vs grim_quant::dequant_q5k ----
extern "C" __global__ void grim_dequant_q5k(const unsigned char* __restrict__ packed,
                                             float* __restrict__ out, int n_blocks) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const unsigned char* blk = packed + b * 176;
    const unsigned short* h_ptr = (const unsigned short*)blk;
    float d    = grim_f16_to_f32(h_ptr[0]);
    float dmin = grim_f16_to_f32(h_ptr[1]);
    const unsigned char* scales = blk + 4;
    const unsigned char* qh = blk + 16;
    const unsigned char* qs = blk + 48;
    float* o = out + b * 256;

    // Mirrors ggml dequantize_row_q5_K: four 64-weight groups, each reading
    // 32 qs bytes (low nibbles then high nibbles) with u1/u2 shifting <<2.
    int qs_idx = 0, is = 0;
    unsigned char u1 = 1, u2 = 2;
    #pragma unroll
    for (int n = 0; n < 4; ++n) {
        float sc1, m1, sc2, m2;
        grim_get_scale_min_k4(is + 0, scales, &sc1, &m1);
        grim_get_scale_min_k4(is + 1, scales, &sc2, &m2);
        float d1 = d * sc1, min1 = dmin * m1;
        float d2 = d * sc2, min2 = dmin * m2;
        #pragma unroll
        for (int l = 0; l < 32; ++l) {
            float q1 = (float)((qs[qs_idx + l] & 0x0F) + ((qh[l] & u1) ? 16 : 0));
            float q2 = (float)((qs[qs_idx + l] >> 4) + ((qh[l] & u2) ? 16 : 0));
            o[l]      = d1 * q1 - min1;
            o[l + 32] = d2 * q2 - min2;
        }
        o      += 64;
        qs_idx += 32;
        is     += 2;
        u1    <<= 2;
        u2    <<= 2;
    }
}

// ---- Q6_K (210 B / 256 weights) — bit-accurate vs grim_quant::dequant_q6k ----
extern "C" __global__ void grim_dequant_q6k(const unsigned char* __restrict__ packed,
                                             float* __restrict__ out, int n_blocks) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const unsigned char* blk = packed + b * 210;
    // ggml block_q6_K layout: ql (128B) + qh (64B) + scales (16B) + d (f16, LAST @208).
    const unsigned char* ql     = blk;             // 128 bytes @0
    const unsigned char* qh     = blk + 128;       // 64 bytes  @128
    const unsigned char* scales = blk + 192;       // 16 bytes  @192 (i8)
    float d = grim_f16_to_f32(*((const unsigned short*)(blk + 208)));
    float* o = out + b * 256;

    int sc_idx = 0, ql_idx = 0, qh_idx = 0;
    #pragma unroll
    for (int n = 0; n < 2; ++n) {
        #pragma unroll
        for (int l = 0; l < 32; ++l) {
            int is = l / 16;
            float q1 = (float)((ql[ql_idx + l      ] & 0x0F) | ((qh[qh_idx + l] & 0x03) << 4)) - 32.0f;
            float q2 = (float)((ql[ql_idx + l + 32 ] & 0x0F) | ((qh[qh_idx + l] & 0x0C) << 2)) - 32.0f;
            float q3 = (float)((ql[ql_idx + l      ] >> 4)   | ((qh[qh_idx + l] & 0x30) >> 0)) - 32.0f;
            float q4 = (float)((ql[ql_idx + l + 32 ] >> 4)   | ((qh[qh_idx + l] & 0xC0) >> 2)) - 32.0f;
            float sc1 = (float)((signed char)scales[sc_idx + is + 0]);
            float sc2 = (float)((signed char)scales[sc_idx + is + 2]);
            float sc3 = (float)((signed char)scales[sc_idx + is + 4]);
            float sc4 = (float)((signed char)scales[sc_idx + is + 6]);
            o[l      ] = d * sc1 * q1;
            o[l + 32 ] = d * sc2 * q2;
            o[l + 64 ] = d * sc3 * q3;
            o[l + 96 ] = d * sc4 * q4;
        }
        o      += 128;
        ql_idx += 64;
        qh_idx += 32;
        sc_idx += 8;
    }
}

// ---- IQ4_NL (170 B / 256 weights) — bit-accurate vs grim_quant::dequant_iq4nl ----
// Layout: d:f16@0, q8:32B@2, q4:128B@34, scales:8B@162.
extern "C" __global__ void grim_dequant_iq4nl(const unsigned char* __restrict__ packed,
                                               float* __restrict__ out, int n_blocks) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const unsigned char* blk = packed + b * 170;
    float d = grim_f16_to_f32(*((const unsigned short*)blk));
    const unsigned char* q8     = blk + 2;
    const unsigned char* q4     = blk + 34;
    const unsigned char* scales = blk + 162;
    float* o = out + b * 256;

    #pragma unroll
    for (int g = 0; g < 16; ++g) {
        unsigned int group_scale = (scales[g / 2] >> ((g % 2) * 4)) & 0x0F;
        float scale = d * (1.0f + 0.125f * (float)group_scale);
        int group_start = g * 16;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int gi = group_start + i;
            unsigned char nibble = (gi % 2 == 0) ? (q4[gi / 2] & 0x0F)
                                                : ((q4[gi / 2] >> 4) & 0x0F);
            unsigned char sign_bit = (q8[gi / 8] >> (gi % 8)) & 0x01;
            float sign = sign_bit ? -1.0f : 1.0f;
            o[gi] = GRIM_IQ4_NL_CODEBOOK[nibble] * scale * sign;
        }
    }
}

// ---- IQ4_XS (136 B / 256 weights) — bit-accurate vs grim_quant::dequant_iq4xs ----
// Layout: d:f16@0, scales:6B@2, qs:128B@8. 8 subblocks × 32 weights.
extern "C" __global__ void grim_dequant_iq4xs(const unsigned char* __restrict__ packed,
                                               float* __restrict__ out, int n_blocks) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const unsigned char* blk = packed + b * 136;
    float d = grim_f16_to_f32(*((const unsigned short*)blk));
    const unsigned char* scales_buf = blk + 2;
    const unsigned char* qs = blk + 8;
    float* o = out + b * 256;

    #pragma unroll
    for (int sb = 0; sb < 8; ++sb) {
        unsigned int sc_val = (scales_buf[(sb * 6) / 8] >> ((sb * 6) % 8)) & 0x3F;
        float scale = d * ((float)sc_val - 32.0f) * (1.0f / 32.0f);
        int sb_start = sb * 32;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            int gi = sb_start + i;
            unsigned char nibble = (gi % 2 == 0) ? (qs[gi / 2] & 0x0F)
                                                : ((qs[gi / 2] >> 4) & 0x0F);
            float code_mag = GRIM_IQ4_NL_CODEBOOK[nibble & 0x07];
            float sign = (nibble & 0x08) ? -1.0f : 1.0f;
            o[gi] = code_mag * scale * sign;
        }
    }
}

// ---- IQ3_XXS (96 B / 256 weights) — bit-accurate vs grim_quant::dequant_iq3xxs ----
// Layout: d:f16@0, qs:64B@2, signs:30B@66. (No signs[30..32]; only 240 sign bits.)
extern "C" __global__ void grim_dequant_iq3xxs(const unsigned char* __restrict__ packed,
                                                float* __restrict__ out, int n_blocks) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const unsigned char* blk = packed + b * 96;
    float d = grim_f16_to_f32(*((const unsigned short*)blk));
    const unsigned char* qs    = blk + 2;
    const unsigned char* signs = blk + 66;
    float* o = out + b * 256;

    #pragma unroll
    for (int i = 0; i < 256; ++i) {
        int q_idx = (i / 8) < 64 ? (i / 8) : 63;
        int grid_idx = (int)qs[q_idx];
        int sub_idx = i % 8;
        float base_val = (float)((grid_idx + sub_idx * 17) % 7) - 3.0f;
        int sb_idx = (i / 8) < 30 ? (i / 8) : 29;
        unsigned char sign_bit = (signs[sb_idx] >> (i % 8)) & 0x01;
        float sign = sign_bit ? -1.0f : 1.0f;
        o[i] = d * base_val * 0.25f * sign;
    }
}

// ---- IQ3_S (110 B / 256 weights) — bit-accurate vs grim_quant::dequant_iq3s ----
// Layout: d:f16@0, qs:64B@2, scales:12B@66, signs:32B@78. 8 subblocks × 32.
extern "C" __global__ void grim_dequant_iq3s(const unsigned char* __restrict__ packed,
                                             float* __restrict__ out, int n_blocks) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const unsigned char* blk = packed + b * 110;
    float d = grim_f16_to_f32(*((const unsigned short*)blk));
    const unsigned char* qs     = blk + 2;
    const unsigned char* scales = blk + 66;
    const unsigned char* signs  = blk + 78;
    float* o = out + b * 256;

    #pragma unroll
    for (int sb = 0; sb < 8; ++sb) {
        float sc = ((float)scales[(sb * 12) / 8] + 1.0f) * 0.125f;
        float scale = d * sc;
        int sb_start = sb * 32;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            int gi = sb_start + i;
            int q_idx = (gi / 8) < 64 ? (gi / 8) : 63;
            float grid_val = (float)(((int)qs[q_idx] + gi) % 7) - 3.0f;
            unsigned char sign_bit = (signs[gi / 8] >> (gi % 8)) & 0x01;
            float sign = sign_bit ? -1.0f : 1.0f;
            o[gi] = scale * grid_val * sign;
        }
    }
}

// ---- IQ2_XXS (66 B / 256 weights) — bit-accurate vs grim_quant::dequant_iq2xxs ----
// Layout: d:f16@0, qs:32B@2, signs:32B@34.
extern "C" __global__ void grim_dequant_iq2xxs(const unsigned char* __restrict__ packed,
                                                float* __restrict__ out, int n_blocks) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const unsigned char* blk = packed + b * 66;
    float d = grim_f16_to_f32(*((const unsigned short*)blk));
    const unsigned char* qs    = blk + 2;
    const unsigned char* signs = blk + 34;
    float* o = out + b * 256;

    #pragma unroll
    for (int i = 0; i < 256; ++i) {
        int q_idx = (i / 8) < 32 ? (i / 8) : 31;
        int grid_idx = (int)qs[q_idx];
        float val = (float)((grid_idx + (i % 8)) % 4) - 1.5f;
        unsigned char sign_bit = (signs[i / 8] >> (i % 8)) & 0x01;
        float sign = sign_bit ? -1.0f : 1.0f;
        o[i] = d * val * sign;
    }
}

// ---- IQ2_XS (74 B / 256 weights) — bit-accurate vs grim_quant::dequant_iq2xs ----
// Layout: d:f16@0, qs:32B@2, scales:8B@34, signs:32B@42. 16 subblocks × 16.
extern "C" __global__ void grim_dequant_iq2xs(const unsigned char* __restrict__ packed,
                                               float* __restrict__ out, int n_blocks) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const unsigned char* blk = packed + b * 74;
    float d = grim_f16_to_f32(*((const unsigned short*)blk));
    const unsigned char* qs     = blk + 2;
    const unsigned char* scales  = blk + 34;
    const unsigned char* signs   = blk + 42;
    float* o = out + b * 256;

    #pragma unroll
    for (int sb = 0; sb < 16; ++sb) {
        float sc = (float)((scales[sb / 2] >> ((sb % 2) * 4)) & 0x0F) * 0.125f + 0.5f;
        float scale = d * sc;
        int sb_start = sb * 16;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int gi = sb_start + i;
            int q_idx = (gi / 8) < 32 ? (gi / 8) : 31;
            int grid_idx = (int)qs[q_idx];
            float val = (float)((grid_idx + (gi % 8)) % 4) - 1.5f;
            unsigned char sign_bit = (signs[gi / 8] >> (gi % 8)) & 0x01;
            float sign = sign_bit ? -1.0f : 1.0f;
            o[gi] = scale * val * sign;
        }
    }
}

// ---- IQ2_S (82 B / 256 weights) — bit-accurate vs grim_quant::dequant_iq2s ----
// Layout: d:f16@0, qs:48B@2, scales:8B@50, signs:24B@58. 16 subblocks × 16.
extern "C" __global__ void grim_dequant_iq2s(const unsigned char* __restrict__ packed,
                                              float* __restrict__ out, int n_blocks) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const unsigned char* blk = packed + b * 82;
    float d = grim_f16_to_f32(*((const unsigned short*)blk));
    const unsigned char* qs     = blk + 2;
    const unsigned char* scales  = blk + 50;
    const unsigned char* signs   = blk + 58;
    float* o = out + b * 256;

    #pragma unroll
    for (int sb = 0; sb < 16; ++sb) {
        float sc = (float)((scales[sb / 2] >> ((sb % 2) * 4)) & 0x0F) * 0.125f + 0.5f;
        float scale = d * sc;
        int sb_start = sb * 16;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int gi = sb_start + i;
            int q_idx = (gi / 8) < 48 ? (gi / 8) : 47;
            int grid_idx = (int)qs[q_idx];
            float code = (float)((grid_idx + (gi % 8)) % 4) - 1.5f;
            int sgn_idx = (gi / 8) < 24 ? (gi / 8) : 23;
            unsigned char sign_bit = (signs[sgn_idx] >> (gi % 8)) & 0x01;
            float sign = sign_bit ? -1.0f : 1.0f;
            o[gi] = scale * code * sign;
        }
    }
}

// ---- FP8 E4M3 (1 byte/weight + 4-byte f32 global scale header) ---------------
// Signature: grim_dequant_fp8(packed, out, n_weights). One thread per weight.
// Matches grim_quant::dequant_fp8 (`lib.rs:1106`): first 4 bytes are the LE
// f32 global scale, then 1 E4M3 byte per weight, value = fp8(byte)*scale.
extern "C" __global__ void grim_dequant_fp8(const unsigned char* __restrict__ packed,
                                             float* __restrict__ out, int n_weights) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_weights) return;
    // 4-byte LE f32 global scale header (matches grim_quant::dequant_fp8).
    float scale = __int_as_float(
        (unsigned int)packed[0] | ((unsigned int)packed[1] << 8) |
        ((unsigned int)packed[2] << 16) | ((unsigned int)packed[3] << 24));
    out[i] = grim_fp8_e4m3_to_f32(packed[4 + i]) * scale;
}

// ---- MXFP4 (length-prefixed codes + E8M0 shared exponents) -----------------
// Signature: grim_dequant_mxfp4(codes, exps, out, n_values).
//   codes:  E2M1 packed nibbles, 2 per byte (low nibble = even i).
//   exps:   1 byte per 32-element group (E8M0 shared exponent).
// Matches grim_quant::dequant_mxfp4 (`lib.rs:1177`). The length-prefix framing
// is split host-side before launch (see lib.rs dequant dispatch).
extern "C" __global__ void grim_dequant_mxfp4(const unsigned char* __restrict__ codes,
                                               const unsigned char* __restrict__ exps,
                                               float* __restrict__ out, int n_values) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_values) return;
    int group_idx = i / 32;
    unsigned char shared_exp = exps[group_idx];
    unsigned char code_byte = codes[i / 2];
    unsigned char code = (i % 2 == 0) ? (code_byte & 0x0F) : ((code_byte >> 4) & 0x0F);
    out[i] = grim_mxfp4_to_f32(code, shared_exp);
}

// ---- MXFP8 (length-prefixed codes + E8M0 shared exponents) -----------------
extern "C" __global__ void grim_dequant_mxfp8(const unsigned char* __restrict__ codes,
                                               const unsigned char* __restrict__ exps,
                                               float* __restrict__ out, int n_values) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_values) return;
    int group_idx = i / 32;
    unsigned char shared_exp = exps[group_idx];
    unsigned char code = codes[i];
    float base_val = grim_fp8_e4m3_to_f32(code);
    float scale = powf(2.0f, (float)shared_exp - 127.0f);
    out[i] = base_val * scale;
}

// ---- Q4_K (144 B / 256 weights) ----------------------------------------------
extern "C" __global__ void grim_dequant_q4k(const unsigned char* __restrict__ packed,
                                             float* __restrict__ out, int n_blocks) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;

    const unsigned char* blk = packed + b * 144;
    const unsigned short* h_ptr = (const unsigned short*)blk;
    float d = grim_f16_to_f32(h_ptr[0]);
    float dmin = grim_f16_to_f32(h_ptr[1]);

    const unsigned char* scales = blk + 4;
    const unsigned char* qs = blk + 16;
    float* o = out + b * 256;

    int is = 0;
    int qs_idx = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float sc1, m1, sc2, m2;
        grim_get_scale_min_k4(is + 0, scales, &sc1, &m1);
        grim_get_scale_min_k4(is + 1, scales, &sc2, &m2);
        float d_sc1 = d * sc1;
        float d_m1 = dmin * m1;
        float d_sc2 = d * sc2;
        float d_m2 = dmin * m2;
        #pragma unroll
        for (int l = 0; l < 32; ++l) {
            float q1 = (float)(qs[qs_idx + l] & 0x0F);
            float q2 = (float)(qs[qs_idx + l] >> 4);
            o[l]      = d_sc1 * q1 - d_m1;
            o[l + 32] = d_sc2 * q2 - d_m2;
        }
        o += 64;
        is += 2;
        qs_idx += 32;
    }
}

// ---- Q8_0 (34 B / 32 weights) ------------------------------------------------
extern "C" __global__ void grim_dequant_q8_0(const unsigned char* __restrict__ packed,
                                              float* __restrict__ out, int n_blocks) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n_blocks * 32) return;

    int block_idx = id / 32;
    int in_block = id % 32;
    const unsigned char* block_ptr = packed + block_idx * 34;
    const unsigned short* h_ptr = (const unsigned short*)block_ptr;
    float d = grim_f16_to_f32(h_ptr[0]);
    signed char q = (signed char)block_ptr[2 + in_block];
    out[id] = d * (float)q;
}

// ===========================================================================
//  Device-side quantization kernels — bit-accurate ports of the CPU reference
//  `grim_quant::quant_*` (lib.rs). These enable per-step activation/gradient
//  quantization without a D2H/H2D round-trip.
// ===========================================================================

// f32 → f16 conversion (mirrors grim_quant::f32_to_f16, lib.rs:2531).
// Truncating rounding (no round-to-nearest), matching the CPU reference exactly.
__device__ inline unsigned short grim_f32_to_f16(float v) {
    unsigned int bits = __float_as_int(v);
    unsigned int sign = (bits >> 31) & 1u;
    int exp = (int)((bits >> 23) & 0xFFu);
    unsigned int mant = bits & 0x7FFFFFu;
    if (exp == 0) return (unsigned short)(sign << 15);
    if (exp >= 0x8D) return (unsigned short)((sign << 15) | 0x7C00u); // overflow → inf
    if (exp <= 0x70) return (unsigned short)(sign << 15);              // underflow → 0
    int new_exp = exp - 127 + 15;
    if (new_exp <= 0) return (unsigned short)(sign << 15);
    return (unsigned short)((sign << 15) | ((unsigned int)new_exp << 10) | (mant >> 13));
}

// f32 → FP8 E4M3 conversion (mirrors grim_quant::f32_to_fp8_e4m3, lib.rs:1662).
__device__ inline unsigned char grim_f32_to_fp8_e4m3(float v) {
    if (isnan(v)) return 0x7F; // NaN in E4M3
    unsigned char sign = signbit(v) ? 0x80u : 0u;
    float abs_v = fabsf(v);
    if (abs_v == 0.0f) return sign;

    unsigned int bits = __float_as_int(abs_v);
    int raw_exp = (int)((bits >> 23) & 0xFFu) - 127;
    unsigned int raw_mant = bits & 0x007FFFFFu;

    int e4m3_exp = raw_exp + 7;

    if (e4m3_exp <= 0) {
        int shift = 1 - e4m3_exp;
        if (shift > 4) return sign;
        unsigned int full_mant = 0x00800000u | raw_mant;
        unsigned int mant = (full_mant >> (20 + shift)) & 0x07u;
        return sign | (unsigned char)mant;
    }
    if (e4m3_exp >= 15) return sign | 0x7Eu;

    unsigned char mant = (unsigned char)(raw_mant >> 20);
    return sign | ((unsigned char)e4m3_exp << 3) | (mant & 0x07u);
}

// ---- Standalone Q8_0 quantization (34 B / 32 weights) ------------------------
// Mirrors grim_quant::quant_q80 (lib.rs:1397). One CUDA block (32 threads) per
// Q8_0 block. Thread 0 finds amax via warp shuffle, computes the f16 scale, and
// writes it; all 32 threads then encode their i8 code in parallel.
extern "C" __global__ void grim_quant_q8_0(const float* __restrict__ x,
                                            unsigned char* __restrict__ out, int n_blocks) {
    int blk = blockIdx.x;
    int lane = threadIdx.x; // 0..31
    if (blk >= n_blocks) return;

    const float* bx = x + blk * 32;
    unsigned char* bout = out + blk * 34;

    // Each thread loads its value (zero-pad if the tensor tail is short).
    float val = (blk * 32 + lane) < n_blocks * 32 ? bx[lane] : 0.0f;
    float abs_val = fabsf(val);

    // Warp-level reduction for amax (32 threads = one warp).
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFFu, abs_val, offset);
        if (other > abs_val) abs_val = other;
    }
    float amax = abs_val; // broadcast to all lanes

    float scale = (amax == 0.0f) ? 1.0f : (amax / 127.0f);
    unsigned short scale_f16 = grim_f32_to_f16(scale);

    if (lane == 0) {
        bout[0] = (unsigned char)(scale_f16 & 0xFFu);
        bout[1] = (unsigned char)((scale_f16 >> 8) & 0xFFu);
    }

    // Encode: q = round(val / scale), clamped to [-128, 127].
    float q_f = (scale == 0.0f) ? 0.0f : (val / scale);
    q_f = rintf(q_f);
    if (q_f > 127.0f) q_f = 127.0f;
    if (q_f < -128.0f) q_f = -128.0f;
    bout[2 + lane] = (unsigned char)(signed char)q_f;
}

// ---- Standalone FP8 E4M3 quantization (4-byte f32 scale + 1 byte/weight) -----
// Mirrors grim_quant::quant_fp8 (lib.rs:1647). One thread per weight. The
// scale header is always 1.0f (matching the CPU reference).
extern "C" __global__ void grim_quant_fp8(const float* __restrict__ x,
                                          unsigned char* __restrict__ out, int n_weights) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_weights) return;

    // Thread 0 of block 0 writes the 4-byte f32 scale header (1.0f).
    if (i == 0) {
        unsigned int scale_bits = __float_as_int(1.0f);
        out[0] = (unsigned char)(scale_bits & 0xFFu);
        out[1] = (unsigned char)((scale_bits >> 8) & 0xFFu);
        out[2] = (unsigned char)((scale_bits >> 16) & 0xFFu);
        out[3] = (unsigned char)((scale_bits >> 24) & 0xFFu);
    }

    out[4 + i] = grim_f32_to_fp8_e4m3(x[i]);
}

// ---- Fused quantize + GEMM: Q8_0 activations ---------------------------------
// Computes C = A_quant @ B where A is quantized to Q8_0 on-the-fly per
// 32-element K-block. Each thread computes one output element C[row, col].
// Grid: (ceil(N/32), ceil(M/8)), Block: (32, 8).
extern "C" __global__ void grim_fused_quant_gemm_q8_0(const float* __restrict__ A,
                                                       const float* __restrict__ B,
                                                       float* __restrict__ C,
                                                       int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    int blocks_per_row = K / 32;

    for (int b_idx = 0; b_idx < blocks_per_row; ++b_idx) {
        // Inline Q8_0 quantization of A[row, b_idx*32 .. b_idx*32+31].
        const float* a_block = A + row * K + b_idx * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) {
            float a = fabsf(a_block[i]);
            if (a > amax) amax = a;
        }
        float scale = (amax == 0.0f) ? 1.0f : (amax / 127.0f);

        // Dot product: quantize-dequantize A, multiply by B.
        const float* b_block = B + (b_idx * 32) * N + col;
        for (int i = 0; i < 32; ++i) {
            float q_f = (scale == 0.0f) ? 0.0f : rintf(a_block[i] / scale);
            if (q_f > 127.0f) q_f = 127.0f;
            if (q_f < -128.0f) q_f = -128.0f;
            float a_deq = q_f * scale;
            sum += a_deq * b_block[i * N];
        }
    }

    C[row * N + col] = sum;
}

// ---- Fused quantize + GEMM: Q8_0 activations (packed, on-device dequant) ------
// Computes C = A_quant @ B where B is stored as Q8_0 packed blocks.
// Each Q8_0 block is 34 bytes: 2-byte f16 scale + 32 i8 codes.
// A is f32; the kernel quantizes A on-the-fly per 32-element K-block.
// Grid: (ceil(N/32), ceil(M/8)), Block: (32, 8).
extern "C" __global__ void grim_fused_quant_gemm_q8_0_packed(const float* __restrict__ A,
                                                              const unsigned char* __restrict__ B_packed,
                                                              float* __restrict__ C,
                                                              int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    int blocks_per_row = K / 32;

    for (int b_idx = 0; b_idx < blocks_per_row; ++b_idx) {
        // Inline Q8_0 quantization of A[row, b_idx*32 .. b_idx*32+31].
        const float* a_block = A + row * K + b_idx * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) {
            float a = fabsf(a_block[i]);
            if (a > amax) amax = a;
        }
        float a_scale = (amax == 0.0f) ? 1.0f : (amax / 127.0f);

        // Q8_0 weight block: 2-byte f16 scale + 32 i8 codes = 34 bytes.
        int packed_block_idx = col * blocks_per_row + b_idx;
        int b_off = packed_block_idx * 34;
        unsigned short scale_bits = ((unsigned int)B_packed[b_off + 1] << 8) | B_packed[b_off];
        float w_scale = grim_f16_to_f32(scale_bits);
        const unsigned char* w_codes = B_packed + b_off + 2;

        for (int i = 0; i < 32; ++i) {
            float q_f = (a_scale == 0.0f) ? 0.0f : rintf(a_block[i] / a_scale);
            if (q_f > 127.0f) q_f = 127.0f;
            if (q_f < -128.0f) q_f = -128.0f;
            float a_deq = q_f * a_scale;
            sum += a_deq * ((float)(signed char)w_codes[i] * w_scale);
        }
    }

    C[row * N + col] = sum;
}

// ---- Fused quantize + GEMM: FP8 E4M3 activations -----------------------------
// Computes C = A_quant @ B where A is quantized to FP8 E4M3 on-the-fly
// per-element (round-trip through fp8_e4m3). Each thread computes one output.
extern "C" __global__ void grim_fused_quant_gemm_fp8(const float* __restrict__ A,
                                                      const float* __restrict__ B,
                                                      float* __restrict__ C,
                                                      int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        unsigned char fp8_code = grim_f32_to_fp8_e4m3(A[row * K + k]);
        float a_deq = grim_fp8_e4m3_to_f32(fp8_code);
        sum += a_deq * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// ---- Fused grouped MoE dispatch (WI-M5) ----------------------------------
// One CUDA thread block (blockIdx.x) carries one (token, expert) routed pair,
// exactly like the ROCm `grim_moe_fused_dispatch` / Vulkan `moe_fused_dispatch`
// P-DAFD path. The host pre-expands top-k routing into flat token/expert/weight
// arrays (`router_tokens`/`router_experts`/`router_weights`), so there is no
// device-side sort or per-expert launch.
//
// Per pair, each thread computes the full SwiGLU expert contribution for one
// token (loop over `hidden` output rows, contracting the `inter` dim for
// gate+up then down) and atomicAdds the `routed_scaling_factor * weight`-scaled
// result into `out[token]`. atomicAdd(float*) requires sm_70+ (RTX 40 is 8.9).
//
// Contract: gate_w/up_w are [e, inter, hidden] (row-major), down_w is
// [e, hidden, inter] (row-major), x is [batch, hidden].
extern "C" __global__ void grim_moe_fused_dispatch(
    const float* x,
    const float* gate_w,
    const float* up_w,
    const float* down_w,
    const unsigned int* router_tokens,
    const unsigned int* router_experts,
    const float* router_weights,
    float* out,
    int hidden, int inter, int num_experts, int batch, float rsf)
{
    int pair = blockIdx.x;
    int tok = (int)router_tokens[pair];
    int exp_id = (int)router_experts[pair];
    float w = router_weights[pair] * rsf;

    int gw_base = exp_id * inter * hidden;
    int uw_base = exp_id * inter * hidden;
    int dw_base = exp_id * hidden * inter;
    int x_base = tok * hidden;

    // gate/up produce per-expert `inter`-dim intermediates; the contraction is
    // over `hidden` (the activation input dim). down then maps `inter` -> `hidden`
    // and atomicAdds the scaled contribution into the shared token output.
    for (int i = 0; i < inter; ++i) {
        float g = 0.0f, u = 0.0f;
        for (int j = 0; j < hidden; ++j) {
            float xv = x[x_base + j];
            g += gate_w[gw_base + i * hidden + j] * xv;
            u += up_w[uw_base + i * hidden + j] * xv;
        }
        float silu_g = g / (1.0f + expf(-g));
        float act = silu_g * u;
        for (int h = 0; h < hidden; ++h) {
            float y = down_w[dw_base + h * inter + i] * act;
            atomicAdd(&out[x_base + h], w * y);
        }
    }
}
"#;
