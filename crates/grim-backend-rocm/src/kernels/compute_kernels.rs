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
    // One thread per (batch, step, dim-half-pair) element. Matches the CPU
    // `BackendDevice::rope` semantics: 3-D input [B, S, D] with positions[si]
    // per step, applying the rotation x1=x[2i], x2=x[2i+1] per pair (interleaved).
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
"#;
