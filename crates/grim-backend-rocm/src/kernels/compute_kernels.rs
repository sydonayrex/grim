//! HIP/C++ source for the six compute ops (add / mul / mul_scalar / sqrt / [see: `extern "C"`, `hipModuleGetFunction`]

/// HIP source for the six non-QKV compute kernels. [see: `crate::compute_kernel_source`]
pub const OTHER_KERNEL_SOURCE: &str = r#"
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

extern "C" __global__ void grim_mul_scalar(const float* x, float s, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = x[i] * s;
}

extern "C" __global__ void grim_add_scalar(const float* x, float s, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = x[i] + s;
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

// ── On-device Fused Optimizer Step Kernels ───────────────────────────────────

extern "C" __global__ void grim_fused_adamw_step(
    float* __restrict__ p,
    const float* __restrict__ g,
    float* __restrict__ m,
    float* __restrict__ v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bc1,
    float bc2,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float grad = g[i];
    float m_val = beta1 * m[i] + (1.0f - beta1) * grad;
    float v_val = beta2 * v[i] + (1.0f - beta2) * grad * grad;

    m[i] = m_val;
    v[i] = v_val;

    float m_hat = m_val / bc1;
    float v_hat = v_val / bc2;
    float param_val = p[i];

    p[i] = param_val - lr * ((m_hat / (sqrtf(v_hat) + eps)) + weight_decay * param_val);
}

extern "C" __global__ void grim_fused_lion_step(
    float* __restrict__ p,
    const float* __restrict__ g,
    float* __restrict__ exp_avg,
    float lr,
    float beta1,
    float beta2,
    float weight_decay,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float grad = g[i];
    float exp_val = exp_avg[i];

    float update = beta1 * exp_val + (1.0f - beta1) * grad;
    float sign_update = (update > 0.0f) ? 1.0f : ((update < 0.0f) ? -1.0f : 0.0f);

    exp_avg[i] = beta2 * exp_val + (1.0f - beta2) * grad;
    float param_val = p[i];

    p[i] = param_val - lr * (sign_update + weight_decay * param_val);
}

extern "C" __global__ void grim_fused_madam_step(
    float* __restrict__ p,
    const float* __restrict__ g,
    float* __restrict__ m,
    float* __restrict__ v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float gamma,
    float weight_decay,
    float bc1,
    float bc2,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float grad = g[i];
    float m_val = beta1 * m[i] + (1.0f - beta1) * grad;
    float v_val = beta2 * v[i] + (1.0f - beta2) * grad * grad;

    m[i] = m_val;
    v[i] = v_val;

    float m_hat = m_val / bc1;
    float v_hat = v_val / bc2;

    float denom = sqrtf(v_hat) + eps;
    float mult_scale = 1.0f / (1.0f + gamma * (fabsf(grad) / denom));
    float step_val = (m_hat / denom) * mult_scale;
    float param_val = p[i];

    p[i] = param_val - lr * (step_val + weight_decay * param_val);
}

// In-memory transpose of a contiguous [a, b] f32 matrix to [b, a].
// Patch-indexed: each thread writes OUT[j*a + i] = IN[i*b + j], so the
// transposed output is produced directly in device memory — no DtoH + H2D
// round trip for weights that must be available in both [out,in] and
// [in,out] layouts. One thread per output element.
extern "C" __global__ void grim_transpose_2d_f32(const float* __restrict__ in,
                                                 float* __restrict__ out,
                                                 int a, int b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = a * b;
    if (idx >= total) return;
    int i = idx / b;  // row in [a, b]
    int j = idx - i * b;  // col in [a, b]
    // out is [b, a] row-major: out[j * a + i] holds in[i * b + j].
    out[j * a + i] = in[i * b + j];
}

extern "C" __global__ void grim_rope(const float* x, const unsigned int* positions,
                                     float* out,
                                     int b, int s, int d, int half, float base,
                                     int interleaved) {
    // One thread per (batch, step, dim-half-pair) element. Pairing follows
    // RopeConfig.interleaved: GPT-J style (x[2i], x[2i+1]) when set — the CPU
    // reference convention, used by LFM2 — else NeoX half-split (x[i],
    // x[i+half]).
    // CONTRACT: plain full-rotary only (rotary_dim == d). Use grim_rope_yarn
    // for partial rotary or YaRN-modified frequencies.
    int total = b * s * half;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int bi = idx / (s * half);
    int rem = idx - bi * (s * half);
    int si = rem / half;
    int i = rem - si * half;
    float pos = (float)positions[si];
    float freq = 1.0f / powf(base, (2.0f * (float)i) / (float)d);
    float val = pos * freq;
    float sin_val = sinf(val);
    float cos_val = cosf(val);
    int base_idx = (bi * s + si) * d;
    int a_idx = interleaved ? (base_idx + 2 * i) : (base_idx + i);
    int b_idx = interleaved ? (base_idx + 2 * i + 1) : (base_idx + half + i);
    float x1 = x[a_idx];
    float x2 = x[b_idx];
    out[a_idx] = x1 * cos_val - x2 * sin_val;
    out[b_idx] = x2 * cos_val + x1 * sin_val;
}

// Partial-rotary + YaRN kernel.
// Handles rotary_dim <= d (partial) and pre-computed YaRN-ramp frequencies.
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
// by a second pass over the copy range [rotary_dim, d).
extern "C" __global__ void grim_rope_yarn(
    const float* __restrict__ x,
    const unsigned int* __restrict__ positions,
    const float* __restrict__ inv_freq,
    float* __restrict__ out,
    int b, int s, int d, int rotary_half, float mscale, int interleaved
) {
    // Pass 1: rotate the [0, rotary_half) pairs.
    int total = b * s * rotary_half;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
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
        // Pairing follows RopeConfig.interleaved (see grim_rope). CPU
        // `Rope::forward` — the oracle — is interleaved (x[2i], x[2i+1]).
        int a_idx = interleaved ? (base_idx + 2 * i) : (base_idx + i);
        int b_idx = interleaved ? (base_idx + 2 * i + 1) : (base_idx + rotary_half + i);
        float x1 = x[a_idx];
        float x2 = x[b_idx];
        out[a_idx] = x1 * cos_val - x2 * sin_val;
        out[b_idx] = x2 * cos_val + x1 * sin_val;
    }
    // Pass 2: copy the non-rotary dims [2*rotary_half, d) verbatim.
    // We reuse the same thread pool; threads with idx in [0, b*s*(d-2*rotary_half))
    // handle the copy dimension.
    int copy_start = 2 * rotary_half;
    int copy_len   = d - copy_start;  // may be 0 for full rotary
    if (copy_len > 0) {
        int total2 = b * s * copy_len;
        if (idx < total2) {
            int bi = idx / (s * copy_len);
            int rem = idx - bi * (s * copy_len);
            int si = rem / copy_len;
            int ci = rem - si * copy_len;  // offset within the non-rotary tail
            int src_idx = (bi * s + si) * d + copy_start + ci;
            out[src_idx] = x[src_idx];
        }
    }
}

extern "C" __global__ void grim_broadcast_bias(const float* bias, float* out,
                                               int batch, int out_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_dim;
    if (idx >= total) return;
    int col = idx % out_dim;
    out[idx] = bias[col];
}

extern "C" __global__ void grim_scale_bias_epilogue(
    float* out, const float* a_scale, const float* b_scale,
    const float* bias, int batch, int out_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_dim;
    if (idx >= total) return;
    int i = idx / out_dim;  // token
    int j = idx - i * out_dim;  // output channel
    float s = 1.0f;
    if (a_scale) s *= a_scale[i];
    if (b_scale) s *= b_scale[j];
    float v = out[idx] * s;
    if (bias) v += bias[j];
    out[idx] = v;
}


extern "C" __global__ void grim_silu_mul(float* gate, float* up, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float s = g / (1.0f + expf(-g));
    out[i] = s * up[i];
}

extern "C" __global__ void grim_silu_mul_backward(
    float* e, float* g, float* dw, float* df, float* de, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float ei = e[i];
    float sig = 1.0f / (1.0f + expf(-ei));      // sigmoid(e)
    float silu_e = ei * sig;                      // silu(e) = e * sigmoid(e)
    float d_silu = sig * (1.0f + ei * (1.0f - sig)); // silu'(e) = sigmoid(e) * (1 + e*(1-sigmoid(e)))

    df[i] = silu_e * dw[i];                      // dL/dg = silu(e) * dL/dy
    de[i] = d_silu * g[i] * dw[i];               // dL/de = silu'(e) * g * dL/dy
}

// On-device all_reduce accumulator: out[i] = sum_k inputs[k][i].
// `inputs` is a device array of `n_inputs` device pointers (each points to
// `n_elements` floats on the device). One thread per output element.
extern "C" __global__ void grim_all_reduce_accum(
    float* out,
    const float* const* inputs,
    int n_inputs,
    int n_elements
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;
    float acc = 0.0f;
    for (int k = 0; k < n_inputs; ++k) {
        acc += inputs[k][i];
    }
    out[i] = acc;
}

// Warp-per-row RMS norm: one warp owns a row; the sum of squares reduces
// with 5 __shfl_xor butterflies (no barriers). The previous one-thread-per-
// element form made EVERY thread walk the whole row — O(row_len^2) loads per
// row (16.7M redundant reads at hidden=4096).
extern "C" __global__ void __launch_bounds__(256)
grim_rms_norm(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out,
              int row_len, float eps, int total) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int rows = total / row_len;
    if (warp_id >= rows) return;
    const float* x_row = x + (size_t)warp_id * row_len;
    float* o_row = out + (size_t)warp_id * row_len;
    const unsigned long long shfl_mask = 0xffffffffffffffffULL;

    float ss = 0.0f;
    for (int col = lane; col < row_len; col += 32) {
        float v = x_row[col];
        ss += v * v;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        ss += __shfl_xor_sync(shfl_mask, ss, off);
    float rms = sqrtf(ss / (float)row_len + eps);
    for (int col = lane; col < row_len; col += 32) {
        o_row[col] = x_row[col] * w[col] / rms;
    }
}

// Warp-per-row fused residual-add + RMS norm (same reduction structure).
extern "C" __global__ void __launch_bounds__(256)
grim_add_rms_norm(const float* __restrict__ x, const float* __restrict__ residual,
                  const float* __restrict__ w, float* __restrict__ y_out, float* __restrict__ norm_out,
                  int row_len, float eps, int total) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int rows = total / row_len;
    if (warp_id >= rows) return;
    const float* x_row = x + (size_t)warp_id * row_len;
    const float* r_row = residual + (size_t)warp_id * row_len;
    float* y_row = y_out + (size_t)warp_id * row_len;
    float* n_row = norm_out + (size_t)warp_id * row_len;
    const unsigned long long shfl_mask = 0xffffffffffffffffULL;

    // Pass 1: y = x + residual (write-through) + strided sum of squares.
    float ss = 0.0f;
    for (int col = lane; col < row_len; col += 32) {
        float y = x_row[col] + r_row[col];
        y_row[col] = y;
        ss += y * y;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        ss += __shfl_xor_sync(shfl_mask, ss, off);
    float rms = sqrtf(ss / (float)row_len + eps);
    // Pass 2: normalize (y_row is L1/L2-hot from pass 1).
    for (int col = lane; col < row_len; col += 32) {
        n_row[col] = y_row[col] * w[col] / rms;
    }
}

// Warp-per-row online softmax (shuffle max + shuffle sum).
extern "C" __global__ void __launch_bounds__(256)
grim_softmax(const float* __restrict__ x, float* __restrict__ out, int row_len, int total) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int rows = total / row_len;
    if (warp_id >= rows) return;
    const float* x_row = x + (size_t)warp_id * row_len;
    float* o_row = out + (size_t)warp_id * row_len;
    const unsigned long long shfl_mask = 0xffffffffffffffffULL;

    float maxv = -1e30f;
    for (int col = lane; col < row_len; col += 32) {
        float v = x_row[col];
        if (v > maxv) maxv = v;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float o = __shfl_xor_sync(shfl_mask, maxv, off);
        if (o > maxv) maxv = o;
    }
    float sum = 0.0f;
    for (int col = lane; col < row_len; col += 32) {
        sum += expf(x_row[col] - maxv);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_xor_sync(shfl_mask, sum, off);
    float inv = 1.0f / sum;
    for (int col = lane; col < row_len; col += 32) {
        o_row[col] = expf(x_row[col] - maxv) * inv;
    }
}

extern "C" __global__ void grim_embedding(float* weight, float* out,
                                           int* indices, int dim, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int i = idx / dim;
    int j = idx % dim;
    out[idx] = weight[indices[i] * dim + j];
}

extern "C" __global__ void grim_rmsnorm_matmul(
    float* x, float* w_norm, float* weight_mat, float* out,
    int m, int n, int k, float eps
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) return;

    float ss = 0.0f;
    for (int j = 0; j < k; ++j) {
        float val = x[row * k + j];
        ss += val * val;
    }
    float rms = sqrtf(ss / (float)k + eps);

    float sum = 0.0f;
    for (int j = 0; j < k; ++j) {
        float x_norm = x[row * k + j] * w_norm[j] / rms;
        float w_val = weight_mat[j * n + col];
        sum += x_norm * w_val;
    }
    out[row * n + col] = sum;
}

extern "C" __global__ void grim_split_k_reduction(
    const _Float16* __restrict__ partials,
    _Float16* __restrict__ out,
    int m, int n, int split_k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * n;
    if (idx >= total) return;

    float sum = 0.0f;
    for (int k = 0; k < split_k; ++k) {
        sum += (float)partials[k * total + idx];
    }
    out[idx] = (_Float16)sum;
}

// Split-K partial reduction, dtype-specialized. The f16 kernel above is
// the historical entry point; F32 and BF16 GEMMs must reduce their OWN
// element types — routing them through the _Float16 entry read each f32
// partial as half-precision pairs and wrote f16 bits into an f32 output
// buffer (silent garbage for every F32 split-K GEMM, i.e. m > 1 or
// k > 8192; found by the WI-SB6 ring-vs-direct benchmark 2026-08-25).
extern "C" __global__ void grim_split_k_reduction_f32(
    const float* __restrict__ partials,
    float* __restrict__ out,
    int m, int n, int split_k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * n;
    if (idx >= total) return;

    float sum = 0.0f;
    for (int k = 0; k < split_k; ++k) {
        sum += partials[k * total + idx];
    }
    out[idx] = sum;
}

extern "C" __global__ void grim_split_k_reduction_bf16(
    const unsigned short* __restrict__ partials,
    unsigned short* __restrict__ out,
    int m, int n, int split_k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * n;
    if (idx >= total) return;

    float sum = 0.0f;
    for (int k = 0; k < split_k; ++k) {
        // bf16 -> f32 is a 16-bit left shift of the bit pattern.
        unsigned int bits = ((unsigned int)partials[k * total + idx]) << 16;
        sum += __uint_as_float(bits);
    }
    // f32 -> bf16 with round-to-nearest-even.
    unsigned int s = __float_as_uint(sum);
    unsigned int rounded = (s + 0x7fffu + ((s >> 16) & 1u)) >> 16;
    out[idx] = (unsigned short)rounded;
}

extern "C" __global__ void grim_short_conv1d_causal_step(
    const float* x, const float* weight, const float* bias,
    float* conv_state, float* out, int batch, int channels, int kernel_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels;
    if (idx >= total) return;
    int b = idx / channels;
    int c = idx % channels;

    float val = x[idx];
    int state_offset = (b * channels + c) * (kernel_size - 1);
    float sum = val * weight[c * kernel_size + (kernel_size - 1)];
    for (int k = 0; k < kernel_size - 1; ++k) {
        sum += conv_state[state_offset + k] * weight[c * kernel_size + k];
    }
    if (bias) {
        sum += bias[c];
    }
    out[idx] = sum;

    // Shift state buffer left and insert new input
    for (int k = 0; k < kernel_size - 2; ++k) {
        conv_state[state_offset + k] = conv_state[state_offset + k + 1];
    }
    if (kernel_size > 1) {
        conv_state[state_offset + kernel_size - 2] = val;
    }
}

extern "C" __global__ void grim_kda_gated_delta_rule_step(
    const float* q, const float* k, const float* v, const float* beta,
    const float* a_gate, float* S_state, float* out,
    int d_k, int d_v
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= d_v) return;

    // A_t = sigmoid(a_gate)
    float a_t = 1.0f / (1.0f + expf(-a_gate[row]));
    float beta_val = beta[row];

    // Compute a_t_decay = a_t * S_{row, col}
    // and update state S_t = a_t * S_{t-1} + beta_t * (v_t - a_prev) * k_t^T
    float k_dot_s = 0.0f;
    for (int col = 0; col < d_k; ++col) {
        k_dot_s += k[col] * S_state[row * d_k + col];
    }
    float delta_v = v[row] - beta_val * k_dot_s;

    float y_val = 0.0f;
    for (int col = 0; col < d_k; ++col) {
        float old_s = S_state[row * d_k + col];
        float new_s = a_t * old_s + beta_val * delta_v * k[col];
        S_state[row * d_k + col] = new_s;
        y_val += q[col] * new_s;
    }
    out[row] = y_val;
}

extern "C" __global__ void grim_mla_q_kv_norm_split(
    const float* q_raw, const float* kv_raw, const float* q_norm_w, const float* kv_norm_w,
    float* q_nope, float* q_rope, float* kv_nope, float* kv_rope,
    int qk_nope_dim, int qk_rope_dim, int v_dim, float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < qk_nope_dim) {
        // RMSNorm on q_nope
        float ss = 0.0f;
        for (int j = 0; j < qk_nope_dim; ++j) {
            float val = q_raw[j];
            ss += val * val;
        }
        float rms = sqrtf(ss / (float)qk_nope_dim + eps);
        q_nope[idx] = q_raw[idx] * q_norm_w[idx] / rms;
    } else if (idx < qk_nope_dim + qk_rope_dim) {
        int rope_i = idx - qk_nope_dim;
        q_rope[rope_i] = q_raw[idx];
    }

    if (idx < qk_nope_dim) {
        float ss = 0.0f;
        for (int j = 0; j < qk_nope_dim; ++j) {
            float val = kv_raw[j];
            ss += val * val;
        }
        float rms = sqrtf(ss / (float)qk_nope_dim + eps);
        kv_nope[idx] = kv_raw[idx] * kv_norm_w[idx] / rms;
    } else if (idx < qk_nope_dim + qk_rope_dim) {
        int rope_i = idx - qk_nope_dim;
        kv_rope[rope_i] = kv_raw[idx];
    }
}

// Reverse-mode autodiff for RMSNorm (salamander.md Phase 3 & G3):
// dx[i] = (w[i] / rms) * g[i] - x[i] * (sum_j g[j] * w[j] * x[j]) / (hidden_dim * rms^3)
// dw[i] = sum_rows g[row, i] * (x[row, i] / rms)
extern "C" __global__ void __launch_bounds__(256)
grim_rmsnorm_backward(const float* __restrict__ x, const float* __restrict__ w,
                      const float* __restrict__ out_grad, float* __restrict__ dx,
                      int row_len, float eps, int total) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int rows = total / row_len;
    if (warp_id >= rows) return;
    const float* x_row = x + (size_t)warp_id * row_len;
    const float* g_row = out_grad + (size_t)warp_id * row_len;
    float* dx_row = dx + (size_t)warp_id * row_len;
    const unsigned long long shfl_mask = 0xffffffffffffffffULL;

    float ss = 0.0f;
    float sum_g_w_x = 0.0f;
    for (int col = lane; col < row_len; col += 32) {
        float xv = x_row[col];
        float gv = g_row[col];
        float wv = w[col];
        ss += xv * xv;
        sum_g_w_x += gv * wv * xv;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        ss += __shfl_xor_sync(shfl_mask, ss, off);
        sum_g_w_x += __shfl_xor_sync(shfl_mask, sum_g_w_x, off);
    }
    float rms = sqrtf(ss / (float)row_len + eps);
    float rms_inv = 1.0f / rms;
    float scale_sub = (sum_g_w_x / (float)row_len) * (rms_inv * rms_inv * rms_inv);

    for (int col = lane; col < row_len; col += 32) {
        dx_row[col] = (w[col] * rms_inv) * g_row[col] - x_row[col] * scale_sub;
    }
}

// Reverse-mode autodiff for Rotary Position Embedding (RoPE) (salamander.md Phase 3 & G3):
// Orthogonal rotation matrix backward is R(-theta).
// dx0 = g0 * cos + g1 * sin
// dx1 = -g0 * sin + g1 * cos
extern "C" __global__ void __launch_bounds__(256)
grim_rope_backward(const float* __restrict__ out_grad, const float* __restrict__ cos_tab,
                   const float* __restrict__ sin_tab, float* __restrict__ dx,
                   int half_dim, int total_tokens) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int head_dim = half_dim * 2;
    int total_pairs = (total_tokens * head_dim) / 2;
    if (idx >= total_pairs) return;

    int t = idx / half_dim;
    int i = idx % half_dim;
    int offset = t * head_dim;

    float g0 = out_grad[offset + i];
    float g1 = out_grad[offset + half_dim + i];
    float c = cos_tab[i];
    float s = sin_tab[i];

    dx[offset + i] = g0 * c + g1 * s;
    dx[offset + half_dim + i] = -g0 * s + g1 * c;
}

// Reverse-mode autodiff for Softmax (salamander.md Phase 3 & G3):
// dx_i = s_i * (g_i - sum_j g_j * s_j)
extern "C" __global__ void __launch_bounds__(256)
grim_softmax_backward(const float* __restrict__ out_grad, const float* __restrict__ s_out,
                      float* __restrict__ dx, int row_len, int total) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int rows = total / row_len;
    if (warp_id >= rows) return;
    const float* g_row = out_grad + (size_t)warp_id * row_len;
    const float* s_row = s_out + (size_t)warp_id * row_len;
    float* dx_row = dx + (size_t)warp_id * row_len;
    const unsigned long long shfl_mask = 0xffffffffffffffffULL;

    float sum_g_s = 0.0f;
    for (int col = lane; col < row_len; col += 32) {
        sum_g_s += g_row[col] * s_row[col];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum_g_s += __shfl_xor_sync(shfl_mask, sum_g_s, off);
    }
    for (int col = lane; col < row_len; col += 32) {
        dx_row[col] = s_row[col] * (g_row[col] - sum_g_s);
    }
}

// Reverse-mode autodiff for token-embedding lookup (salamander.md P3, the
// 4th fused backward kernel): dweight[token_ids[t], :] += out_grad[t, :].
// Two kernels: a plain zero-fill of the [vocab, hidden] gradient buffer,
// then a grid-strided atomic scatter-add over token x hidden elements.
extern "C" __global__ void __launch_bounds__(256)
grim_zero_f32(float* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = 0.0f;
}

extern "C" __global__ void __launch_bounds__(256)
grim_embedding_backward(const float* __restrict__ out_grad,
                        const unsigned int* __restrict__ token_ids,
                        float* __restrict__ dweight,
                        int num_tokens, int hidden_dim, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tokens * hidden_dim;
    if (idx >= total) return;
    int t = idx / hidden_dim;
    int d = idx - t * hidden_dim;
    unsigned int tok = token_ids[t];
    if (tok >= (unsigned int)vocab_size) return; // mirror CPU bounds check
    atomicAdd(&dweight[(size_t)tok * hidden_dim + d], out_grad[idx]);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kda_mla_short_conv_kernel_presence() {
        assert!(OTHER_KERNEL_SOURCE.contains("grim_short_conv1d_causal_step"));
        assert!(OTHER_KERNEL_SOURCE.contains("grim_kda_gated_delta_rule_step"));
        assert!(OTHER_KERNEL_SOURCE.contains("grim_mla_q_kv_norm_split"));
    }

    /// P3 (4th fused backward kernel): the embedding scatter-add must be in
    /// the JIT module source together with its zero-fill prologue kernel.
    #[test]
    fn test_embedding_backward_kernel_presence() {
        assert!(
            OTHER_KERNEL_SOURCE.contains("grim_embedding_backward"),
            "grim_embedding_backward kernel missing from OTHER_KERNEL_SOURCE"
        );
        assert!(
            OTHER_KERNEL_SOURCE.contains("grim_zero_f32"),
            "grim_zero_f32 prologue kernel missing from OTHER_KERNEL_SOURCE"
        );
        assert!(
            OTHER_KERNEL_SOURCE.contains("atomicAdd"),
            "embedding scatter-add must accumulate with atomicAdd"
        );
    }

    /// `grim_rope_yarn` must carry the full YaRN / partial-rotary contract in
    /// the kernel source so the JIT compiler can resolve all referenced symbols.
    #[test]
    fn test_rope_yarn_kernel_presence() {
        assert!(
            OTHER_KERNEL_SOURCE.contains("grim_rope_yarn"),
            "grim_rope_yarn kernel missing from OTHER_KERNEL_SOURCE"
        );
        assert!(
            OTHER_KERNEL_SOURCE.contains("inv_freq"),
            "inv_freq param missing from grim_rope_yarn"
        );
        assert!(
            OTHER_KERNEL_SOURCE.contains("mscale"),
            "mscale param missing from grim_rope_yarn"
        );
        assert!(
            OTHER_KERNEL_SOURCE.contains("rotary_half"),
            "rotary_half param missing from grim_rope_yarn"
        );
    }

    /// `grim_scale_bias_epilogue` must be present in the HIP source so the JIT
    /// module can resolve it by entry name (same convention as broadcast_bias).
    #[test]
    fn test_scale_bias_epilogue_kernel_presence() {
        assert!(
            OTHER_KERNEL_SOURCE.contains("grim_scale_bias_epilogue"),
            "grim_scale_bias_epilogue kernel missing from OTHER_KERNEL_SOURCE"
        );
        assert!(
            OTHER_KERNEL_SOURCE.contains("a_scale"),
            "a_scale param missing from grim_scale_bias_epilogue"
        );
        assert!(
            OTHER_KERNEL_SOURCE.contains("b_scale"),
            "b_scale param missing from grim_scale_bias_epilogue"
        );
    }
}
