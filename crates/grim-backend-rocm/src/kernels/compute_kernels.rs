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

extern "C" __global__ void grim_rope(const float* x, const unsigned int* positions,
                                     float* out,
                                     int b, int s, int d, int half, float base) {
    // One thread per (batch, step, dim-half-pair) element. Matches CPU
    // `Rope::forward` semantics: 3-D input [B, S, D] with positions[si]
    // per step, applying rotation to pairs (x[i], x[half+i]).
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
    int a_idx = base_idx + i;
    int b_idx = base_idx + half + i;
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
    int b, int s, int d, int rotary_half, float mscale
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
        // Interleaved layout: pair (2i, 2i+1)
        int a_idx = base_idx + 2 * i;
        int b_idx = base_idx + 2 * i + 1;
        float x1 = x[a_idx];
        float x2 = x[b_idx];
        out[a_idx] = x1 * cos_val - x2 * sin_val;
        out[b_idx] = x1 * sin_val + x2 * cos_val;
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

extern "C" __global__ void grim_rms_norm(float* x, float* w, float* out,
                                         int row_len, float eps, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int row = idx / row_len;
    float ss = 0.0f;
    for (int j = 0; j < row_len; ++j) {
        float v = x[row * row_len + j];
        ss += v * v;
    }
    float rms = sqrtf(ss / (float)row_len + eps);
    // w has row_len elements; index by position within the row, not the
    // global linear index.  Using w[idx] (the prior code) reads garbage
    // for every row past the first and makes the hidden state explode.
    int col = idx - row * row_len;
    out[idx] = x[idx] * w[col] / rms;
}

extern "C" __global__ void grim_add_rms_norm(const float* x, const float* residual,
                                             float* w, float* y_out, float* norm_out,
                                             int row_len, float eps, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int row = idx / row_len;
    int col = idx - row * row_len;

    // First pass compute/write updated residual sum y = x + residual
    float y_val = x[idx] + residual[idx];
    y_out[idx] = y_val;

    // Compute mean of squares for this row of y
    float ss = 0.0f;
    for (int j = 0; j < row_len; ++j) {
        float v = x[row * row_len + j] + residual[row * row_len + j];
        ss += v * v;
    }
    float rms = sqrtf(ss / (float)row_len + eps);
    norm_out[idx] = y_val * w[col] / rms;
}

extern "C" __global__ void grim_softmax(float* x, float* out, int row_len, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int row = idx / row_len;
    float maxv = -1e30f;
    for (int j = 0; j < row_len; ++j) {
        float v = x[row * row_len + j];
        if (v > maxv) maxv = v;
    }
    float sum = 0.0f;
    for (int j = 0; j < row_len; ++j) {
        float e = expf(x[row * row_len + j] - maxv);
        sum += e;
    }
    out[idx] = expf(x[idx] - maxv) / sum;
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

    /// `grim_rope_yarn` must carry the full YaRN / partial-rotary contract in
    /// the kernel source so the JIT compiler can resolve all referenced symbols.
    #[test]
    fn test_rope_yarn_kernel_presence() {
        assert!(OTHER_KERNEL_SOURCE.contains("grim_rope_yarn"),
            "grim_rope_yarn kernel missing from OTHER_KERNEL_SOURCE");
        assert!(OTHER_KERNEL_SOURCE.contains("inv_freq"),
            "inv_freq param missing from grim_rope_yarn");
        assert!(OTHER_KERNEL_SOURCE.contains("mscale"),
            "mscale param missing from grim_rope_yarn");
        assert!(OTHER_KERNEL_SOURCE.contains("rotary_half"),
            "rotary_half param missing from grim_rope_yarn");
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


