//! HIP/C++ source for the six compute ops (add / mul / mul_scalar / sqrt /
//! silu_mul / rms_norm / softmax / embedding / rmsnorm_matmul / rope).
//!
//! Each entry point is `extern "C"` so `hipModuleGetFunction` resolves it
//! without name mangling.  The Phase-1 QKV attention kernel lives in
//! `kernels::qkv_attention::KERNEL_SOURCE`; [`compute_kernel_source`] in
//! `lib.rs` concatenates this string with that one at runtime for JIT
//! compilation.

/// HIP source for the six non-QKV compute kernels.
///
/// Concatenated into the crate-wide kernel program via
/// [`crate::compute_kernel_source`].
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
    // per step, applying the rotation x1=x[i], x2=x[i+half] per pair.
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
    out[b_idx] = x1 * sin_val + x2 * cos_val;
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