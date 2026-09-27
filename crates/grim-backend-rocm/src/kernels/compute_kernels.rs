//! HIP/C++ source for the six compute ops (add / mul / mul_scalar / sqrt / [see: `extern "C"`, `hipModuleGetFunction`]

/// HIP source for the six non-QKV compute kernels. [see: `crate::compute_kernel_source`]
pub const OTHER_KERNEL_SOURCE: &str = r#"
// `uintptr_t` is used by the vectorised copy path below and hiprtc does not
// pre-include it; without this the whole module fails to compile with
// "use of undeclared identifier 'uintptr_t'".
#include <stdint.h>

extern "C" __global__ void grim_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    c[i] = a[i] + b[i];
}

extern "C" __global__ void grim_sub(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] - b[i];
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

// Per-row vector scale: out[r, c] = scale[r] * x[r, c] (row-major).
// The device-resident primitive for per-token gating (Xing4.0 hyper-connections).
extern "C" __global__ void grim_row_scale(const float* x, const float* scale, float* out,
                                          int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    int r = i / cols;
    out[i] = x[i] * scale[r];
}

extern "C" __global__ void grim_add_scalar(const float* x, float s, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = x[i] + s;
}

// Contiguous row-range copy: dst[(start + r) * cols + c] = src[r * cols + c].
// Backs the narrow_rows / write_rows sub-block addressing primitives.
// Row-range copy, one block per row, threads walking the row: no integer
// division or modulo. (The flat-index form compiles to a float-reciprocal
// divide, which is both slower and a correctness hazard.)
//
// `grim_row_copy`      : dst[r]           = src[start + r]   (read a row range out)
// `grim_row_copy_into` : dst[start + r]   = src[r]           (write a row range in)
// The two directions are separate kernels because the *source* and the
// *destination* carry the offset in each case, and the output buffer of the
// read form is exactly rows*cols floats.
extern "C" __global__ void grim_row_copy(const float* src, float* dst,
                                         int start, int rows, int cols) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const long long src_base = (long long)(start + r) * cols;
    const long long dst_base = (long long)r * cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        dst[dst_base + c] = src[src_base + c];
    }
}

// Column-range copy, one block per row: out[r, c] = src[r * src_cols + start + c].
extern "C" __global__ void grim_col_copy(const float* src, float* dst,
                                         int start, int rows, int src_cols, int cols) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const long long src_base = (long long)r * src_cols + start;
    const long long dst_base = (long long)r * cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        dst[dst_base + c] = src[src_base + c];
    }
}

// in: dst[(start + r) * dst_cols + c] = src[r * cols + c]
extern "C" __global__ void grim_col_copy_into(const float* src, float* dst,
                                              int start, int rows, int dst_cols, int cols) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const long long src_base = (long long)r * cols;
    // `start` is a COLUMN offset here, not a row offset (contrast grim_row_copy_into).
    const long long dst_base = (long long)r * dst_cols + start;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        dst[dst_base + c] = src[src_base + c];
    }
}

extern "C" __global__ void grim_row_copy_into(const float* src, float* dst,
                                              int start, int rows, int cols) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const long long src_base = (long long)r * cols;
    const long long dst_base = (long long)(start + r) * cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        dst[dst_base + c] = src[src_base + c];
    }
}

// Xing4.0 manifold hyper-connection gates — one block per token.
//
// Input `proj` is the [seq, mix] projection with mix = (2 + hc) * hc laid out
// as [pre(hc) | post(hc) | comb(hc*hc)]; `base` is [mix] and `scale` is [3]
// (pre / post / comb multipliers). Writes the collapse gate `pre` [seq, hc], the
// stream-write gate `post` [seq, hc] (already 2*sigmoid), and the Sinkhorn
// combiner `comb` [seq, hc*hc].
//
// The whole gate math is fused here because it is a fixed 4x4 (hc=4) per token:
// the Sinkhorn iterations need per-token row/column reductions over hc*hc
// values, which no dim-wise reduction primitive exposes. Keeping it in one
// block per token leaves the [seq, hc*hc] projection resident and avoids a
// D2H/H2D round-trip of the gates on every layer.
//
// `hc` is a template-style runtime arg but the launch uses 4; the block covers
// MAX_HC*MAX_HC entries so a smaller hc still indexes in-bounds.
#define MHC_MAX_HC 8
extern "C" __global__ void grim_mhc_gates(const float* __restrict__ proj,
                                          const float* __restrict__ base,
                                          const float* __restrict__ scale,
                                          float* __restrict__ pre_out,
                                          float* __restrict__ post_out,
                                          float* __restrict__ comb_out,
                                          int seq, int hc, int iters,
                                          float hc_eps, float clamp_min, float clamp_max) {
    const int tok = blockIdx.x;
    if (tok >= seq) return;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int mix = (2 + hc) * hc;
    const float* prow = proj + (long long)tok * mix;

    // pre = sigmoid(w * scale[0] + base); post = 2 * sigmoid(w * scale[1] + base)
    for (int h = tid; h < hc; h += nthreads) {
        pre_out[(long long)h * seq + tok] =
            1.0f / (1.0f + expf(-(prow[h] * scale[0] + base[h])));
        post_out[(long long)h * seq + tok] =
            2.0f / (1.0f + expf(-(prow[hc + h] * scale[1] + base[hc + h])));
    }

    // Combiner logits -> clamp -> exp (row-max stabilized), then Sinkhorn.
    __shared__ float c[MHC_MAX_HC * MHC_MAX_HC];
    __shared__ float s_red[MHC_MAX_HC * MHC_MAX_HC];
    const int comb_off = 2 * hc;
    float m = -1e30f;
    for (int k = tid; k < hc * hc; k += nthreads) {
        float v = prow[comb_off + k] * scale[2] + base[comb_off + k];
        if (v < clamp_min) v = clamp_min;
        if (v > clamp_max) v = clamp_max;
        c[k] = v;
        if (v > m) m = v;
    }
    // Block-reduce the per-token row max through a separate buffer so `c`
    // keeps its clamped logits until the exp pass.
    s_red[tid] = m;
    __syncthreads();
    for (int stride = (nthreads >> 1); stride > 0; stride >>= 1) {
        if (tid < stride) s_red[tid] = fmaxf(s_red[tid], s_red[tid + stride]);
        __syncthreads();
    }
    const float mx = s_red[0];
    for (int k = tid; k < hc * hc; k += nthreads) {
        c[k] = expf(c[k] - mx);
    }
    __syncthreads();

    for (int it = 0; it < iters; ++it) {
        // Row normalize.
        for (int r = tid; r < hc; r += nthreads) {
            float s = 0.0f;
            for (int i = 0; i < hc; ++i) s += c[r * hc + i];
            const float d = s + hc_eps;
            for (int i = 0; i < hc; ++i) c[r * hc + i] /= d;
        }
        __syncthreads();
        // Column normalize.
        for (int i = tid; i < hc; i += nthreads) {
            float s = 0.0f;
            for (int r = 0; r < hc; ++r) s += c[r * hc + i];
            const float d = s + hc_eps;
            for (int r = 0; r < hc; ++r) c[r * hc + i] /= d;
        }
        __syncthreads();
    }

    // Emit stream-major / token-last: comb_out[(h_out * hc + h_in) * seq + tok].
    // That makes every downstream per-(stream, source-stream) weight vector a
    // contiguous row slice, so the write-back needs no gather.
    for (int k = tid; k < hc * hc; k += nthreads) {
        comb_out[(long long)k * seq + tok] = c[k];
    }
}

// Fused AXPY: out = a + s * b (e.g. residual + residual_multiplier * branch)
extern "C" __global__ void grim_axpy(const float* a, float s, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] + s * b[i];
}


extern "C" __global__ void grim_sqrt(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = sqrtf(x[i]);
}

// Broadcast a per-channel vector [dk] across heads -> [nh, dk]. Used by the
// on-device GDL-2 gate path so the erase/write/decay gates (one dk-vector per
// token, shared across all heads per the GDL-2 spec) reach the fused kernel's
// [slot, dk] layout without a host round-trip.
extern "C" __global__ void grim_broadcast_heads(const float* in, float* out, int dk, int nh) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = dk * nh;
    if (idx >= total) return;
    out[idx] = in[idx % dk];
}

// GQA head-repeat: expand K/V from [nkv, hd] to [nh, hd] by repeating each
// KV head kv_group times (kv_h = h / kv_group). One launch handles both K and V.
extern "C" __global__ void grim_head_repeat(
    const float* k_in, const float* v_in,
    float* k_out, float* v_out,
    int nkv, int nh, int kv_group, int hd
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nh * hd;
    if (idx >= total) return;
    int h = idx / hd;
    int d = idx % hd;
    int kv_h = h / kv_group;
    k_out[idx] = k_in[kv_h * hd + d];
    v_out[idx] = v_in[kv_h * hd + d];
}

extern "C" __global__ void grim_recip(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = 1.0f / x[i];
}

extern "C" __global__ void grim_silu(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    out[i] = v / (1.0f + expf(-v));
}

extern "C" __global__ void grim_sigmoid(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    out[i] = 1.0f / (1.0f + expf(-v));
}

// GeLU (tanh approximation): out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
extern "C" __global__ void grim_gelu(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x_val = x[i];
    const float sqrt_2_over_pi = 0.7978845608028654f;
    float tanh_in = sqrt_2_over_pi * (x_val + 0.044715f * x_val * x_val * x_val);
    out[i] = 0.5f * x_val * (1.0f + tanhf(tanh_in));
}

// GeLU-tanh gated activation: out = (0.5 * gate * (1 + tanh(sqrt(2/pi) * (gate + 0.044715 * gate^3)))) * up
extern "C" __global__ void grim_gelu_tanh_mul(const float* gate, const float* up, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x_val = gate[i];
    const float sqrt_2_over_pi = 0.7978845608028654f;
    float tanh_in = sqrt_2_over_pi * (x_val + 0.044715f * x_val * x_val * x_val);
    float gelu = 0.5f * x_val * (1.0f + tanhf(tanh_in));
    out[i] = gelu * up[i];
}

// In-place or out-of-place tanh softcapping: out = cap * tanh(x / cap)
extern "C" __global__ void grim_tanh_softcap(const float* x, float cap, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = cap * tanhf(x[i] / cap);
}

// ── GPU Reductions: sum, max, argmax (tree reduction with shared memory) ──

#define GRIM_REDUCE_BLOCK 256

extern "C" __global__ void grim_reduce_sum_stage1(const float* __restrict__ x, float* __restrict__ partials, int n) {
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    float acc = 0.0f;
    for (int i = gid; i < n; i += stride) {
        acc += x[i];
    }

    __shared__ float s_data[GRIM_REDUCE_BLOCK];
    s_data[tid] = acc;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partials[blockIdx.x] = s_data[0];
    }
}

extern "C" __global__ void grim_reduce_sum_stage2(const float* __restrict__ partials, float* __restrict__ out, int num_partials) {
    const int tid = threadIdx.x;
    float acc = 0.0f;
    for (int i = tid; i < num_partials; i += blockDim.x) {
        acc += partials[i];
    }

    __shared__ float s_data[GRIM_REDUCE_BLOCK];
    s_data[tid] = acc;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[0] = s_data[0];
    }
}

extern "C" __global__ void grim_reduce_max_stage1(const float* __restrict__ x, float* __restrict__ partials, int n) {
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    float acc = -1e38f;
    for (int i = gid; i < n; i += stride) {
        float val = x[i];
        if (val > acc || acc == -1e38f) {
            acc = val;
        }
    }

    __shared__ float s_data[GRIM_REDUCE_BLOCK];
    s_data[tid] = acc;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_data[tid + s] > s_data[tid]) {
                s_data[tid] = s_data[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        partials[blockIdx.x] = s_data[0];
    }
}

extern "C" __global__ void grim_reduce_max_stage2(const float* __restrict__ partials, float* __restrict__ out, int num_partials) {
    const int tid = threadIdx.x;
    float acc = -1e38f;
    for (int i = tid; i < num_partials; i += blockDim.x) {
        float val = partials[i];
        if (val > acc || acc == -1e38f) {
            acc = val;
        }
    }

    __shared__ float s_data[GRIM_REDUCE_BLOCK];
    s_data[tid] = acc;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_data[tid + s] > s_data[tid]) {
                s_data[tid] = s_data[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[0] = s_data[0];
    }
}

extern "C" __global__ void grim_argmax_stage1(
    const float* __restrict__ x,
    float* __restrict__ partial_vals,
    unsigned int* __restrict__ partial_idxs,
    int n
) {
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    float max_val = -1e38f;
    int max_idx = -1;

    for (int i = gid; i < n; i += stride) {
        float val = x[i];
        // For tie-breaking matching Iterator::max_by / argmax on CPU: last index wins (val >= max_val)
        if (val >= max_val || max_idx == -1) {
            max_val = val;
            max_idx = i;
        }
    }

    __shared__ float s_val[GRIM_REDUCE_BLOCK];
    __shared__ int   s_idx[GRIM_REDUCE_BLOCK];
    s_val[tid] = max_val;
    s_idx[tid] = max_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            // Later thread index has greater original array indices; if >=, later wins
            if (s_val[tid + s] > s_val[tid] || (s_val[tid + s] == s_val[tid] && s_idx[tid + s] > s_idx[tid])) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_vals[blockIdx.x] = s_val[0];
        partial_idxs[blockIdx.x] = (s_idx[0] >= 0) ? (unsigned int)s_idx[0] : 0u;
    }
}

extern "C" __global__ void grim_argmax_stage2(
    const float* __restrict__ partial_vals,
    const unsigned int* __restrict__ partial_idxs,
    unsigned int* __restrict__ out_idx,
    int num_partials
) {
    const int tid = threadIdx.x;
    float max_val = -1e38f;
    int max_idx = -1;

    for (int i = tid; i < num_partials; i += blockDim.x) {
        float val = partial_vals[i];
        unsigned int idx = partial_idxs[i];
        if (val > max_val || (val == max_val && (int)idx > max_idx) || max_idx == -1) {
            max_val = val;
            max_idx = (int)idx;
        }
    }

    __shared__ float s_val[GRIM_REDUCE_BLOCK];
    __shared__ int   s_idx[GRIM_REDUCE_BLOCK];
    s_val[tid] = max_val;
    s_idx[tid] = max_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_val[tid + s] > s_val[tid] || (s_val[tid + s] == s_val[tid] && s_idx[tid + s] > s_idx[tid])) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_idx[0] = (s_idx[0] >= 0) ? (unsigned int)s_idx[0] : 0u;
    }
}

// ── SPEED-ROC: FP32 → FP16 Activation Quantization ──────────────────────────
// Converts FP32 activations to FP16 in-place for WMMA GEMM input.
// Reduces activation memory bandwidth by 2× (4 bytes → 2 bytes per element).

extern "C" __global__ void grim_quantize_fp16(
    const float* __restrict__ src,
    _Float16* __restrict__ dst,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = (_Float16)src[i];
}

// FP16 → FP32 dequantization (for output conversion if needed).
extern "C" __global__ void grim_dequantize_fp16(
    const _Float16* __restrict__ src,
    float* __restrict__ dst,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = (float)src[i];
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
// Patch-indexed: each thread writes OUT[j*a + i] = IN[i*b + j], so the transposed output.
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

// Convert a column-major [rows, cols] matrix to row-major [rows, cols].
// The candidate uses this after BLASLt's canonical column-major output.
extern "C" __global__ void grim_col_major_to_row_major_f32(
    const float* __restrict__ in,
    float* __restrict__ out,
    int rows,
    int cols,
    int ld) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;
    int row = idx / cols;
    int col = idx - row * cols;
    out[idx] = in[(long long)col * ld + row];
}

extern "C" __global__ void grim_rope(const float* x, const unsigned int* positions,
                                     float* out,
                                     int b, int s, int d, int half, float base,
                                     int interleaved) {
    // One thread per (batch, step, dim-half-pair) element.
    // Pairing follows RopeConfig.interleaved: GPT-J style (x[2i], x[2i+1]) when set - the CPU reference convention, used.
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

// Item 2: device-base RoPE for the decode path. Reads the base position per
// batch slot from device memory (`pos_base[bi]`; legacy callers pass a [1]
// buffer — bi is 0 there) and derives each step's position as `base+si`
// internally — so the host never builds and uploads a per-layer per-token
// `positions[]` Vec. Same fp32 rotation math as `grim_rope`; only the position
// source changes. `steps` (s) is the number of query positions per slot (1 for
// decode); each is rotated by base+si.
extern "C" __global__ void grim_rope_dev_base(const float* x, const unsigned int* pos_base,
                                               float* out,
                                               int b, int s, int d, int half, float base,
                                               int interleaved, int num_heads) {
    // One thread per (batch, step, dim-half-pair) element. `s` is the flattened
    // query length = num_heads * steps. All heads within the same step share one
    // position, so the step index is `si / num_heads` and the position is
    // `base + step_idx` — exactly mirroring the host-position `grim_rope` with
    // positions[] = [base+t repeated num_heads times per step].
    int total = b * s * half;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int bi = idx / (s * half);
    int rem = idx - bi * (s * half);
    int si = rem / half;
    int i = rem - si * half;
    int step_idx = (num_heads > 0) ? (si / num_heads) : si;
    float pos = (float)(pos_base[bi] + (unsigned int)step_idx);
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

// PLAN-decode-throughput-restore: fused QK-norm + device-base RoPE. Same math
// as grim_rope_dev_base, but applies the per-head RMS QK-norm (gamma + eps)
// BEFORE the rotation, in the same pass: each warp covers exactly one head row
// (half == 32 == warpSize), so the row's sum of squares is a warp shuffle
// reduce. Reads its own pair before any write — safe in place.
extern "C" __global__ void grim_qk_rope_dev_base(const float* x, const unsigned int* pos_base,
                                                 const float* gamma,
                                                 float* out,
                                                 int b, int s, int d, int half, float base,
                                                 int interleaved, int num_heads, float eps) {
    int total = b * s * half;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int bi = idx / (s * half);
    int rem = idx - bi * (s * half);
    int si = rem / half;
    int i = rem - si * half;
    int step_idx = (num_heads > 0) ? (si / num_heads) : si;
    float pos = (float)(pos_base[bi] + (unsigned int)step_idx);
    float freq = 1.0f / powf(base, (2.0f * (float)i) / (float)d);
    float val = pos * freq;
    float sin_val = sinf(val);
    float cos_val = cosf(val);
    int base_idx = (bi * s + si) * d;
    int a_idx = interleaved ? (base_idx + 2 * i) : (base_idx + i);
    int b_idx = interleaved ? (base_idx + 2 * i + 1) : (base_idx + half + i);
    float x1 = x[a_idx];
    float x2 = x[b_idx];

    // Per-head RMS scale: warp reduce over the row's pairs. Requires
    // half == warpSize (32) so each warp is exactly one head row and every
    // lane is active; the launcher rejects other configs.
    float ss = x1 * x1 + x2 * x2;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        ss += __shfl_xor_sync(0xffffffffffffffffULL, ss, off);
    float inv_rms = rsqrtf(ss / (float)d + eps);
    x1 *= gamma[a_idx - base_idx];
    x2 *= gamma[b_idx - base_idx];
    x1 *= inv_rms;
    x2 *= inv_rms;

    out[a_idx] = x1 * cos_val - x2 * sin_val;
    out[b_idx] = x2 * cos_val + x1 * sin_val;
}

// SPEED-DOT-OPFUSE (Phase 4a): fused RMSNorm + RoPE for Q/K paths.
// Normalizes x with per-channel weights + eps, then applies RoPE rotation in the same kernel.
// Avoids materializing the intermediate normalized tensor in HBM.
extern "C" __global__ void grim_rmsnorm_rope(const float* __restrict__ x,
                                             const float* __restrict__ norm_weight,
                                             const unsigned int* __restrict__ positions,
                                             float* __restrict__ out,
                                             float eps,
                                             int b, int s, int d, int half, float base,
                                             int interleaved) {
    int total = b * s * half;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int bi = idx / (s * half);
    int rem = idx - bi * (s * half);
    int si = rem / half;
    int i = rem - si * half;

    int base_idx = (bi * s + si) * d;

    // First, compute RMSNorm scale across head_dim d for this vector
    float ss = 0.0f;
    for (int k = 0; k < d; ++k) {
        float val = x[base_idx + k];
        ss += val * val;
    }
    float inv_rms = 1.0f / sqrtf(ss / (float)d + eps);

    float pos = (float)positions[si];
    float freq = 1.0f / powf(base, (2.0f * (float)i) / (float)d);
    float val = pos * freq;
    float sin_val = sinf(val);
    float cos_val = cosf(val);

    int a_idx = interleaved ? (base_idx + 2 * i) : (base_idx + i);
    int b_idx = interleaved ? (base_idx + 2 * i + 1) : (base_idx + half + i);

    int a_local = interleaved ? (2 * i) : i;
    int b_local = interleaved ? (2 * i + 1) : (half + i);

    float w1 = norm_weight ? norm_weight[a_local] : 1.0f;
    float w2 = norm_weight ? norm_weight[b_local] : 1.0f;

    float x1 = x[a_idx] * inv_rms * w1;
    float x2 = x[b_idx] * inv_rms * w2;

    out[a_idx] = x1 * cos_val - x2 * sin_val;
    out[b_idx] = x2 * cos_val + x1 * sin_val;
}

// Fused Re-RoPE (Position Retargeting) kernel.
// Un-rotates Key vectors from old_positions and re-rotates them to new_positions in a single pass via.
extern "C" __global__ void grim_rerope(const float* k,
                                       const unsigned int* old_positions,
                                       const unsigned int* new_positions,
                                       float* out,
                                       int b, int s, int d, int half, float base,
                                       int interleaved) {
    int total = b * s * half;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int bi = idx / (s * half);
    int rem = idx - bi * (s * half);
    int si = rem / half;
    int i = rem - si * half;
    float p_old = (float)old_positions[si];
    float p_new = (float)new_positions[si];
    float freq = 1.0f / powf(base, (2.0f * (float)i) / (float)d);

    float val_old = p_old * freq;
    float old_cos = cosf(val_old);
    float old_sin = sinf(val_old);

    float val_new = p_new * freq;
    float new_cos = cosf(val_new);
    float new_sin = sinf(val_new);

    int base_idx = (bi * s + si) * d;
    int a_idx = interleaved ? (base_idx + 2 * i) : (base_idx + i);
    int b_idx = interleaved ? (base_idx + 2 * i + 1) : (base_idx + half + i);

    float k1_rot = k[a_idx];
    float k2_rot = k[b_idx];

    // 1. Un-rotate: inverse 2D rotation [cos, sin; -sin, cos]
    float k1_orig = k1_rot * old_cos + k2_rot * old_sin;
    float k2_orig = -k1_rot * old_sin + k2_rot * old_cos;

    // 2. Re-rotate: forward 2D rotation [cos, -sin; sin, cos]
    out[a_idx] = k1_orig * new_cos - k2_orig * new_sin;
    out[b_idx] = k2_orig * new_cos + k1_orig * new_sin;
}

// Partial-rotary + YaRN kernel. Handles rotary_dim <= d (partial) and pre-computed YaRN-ramp frequencies.
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
    // We reuse the same thread pool; threads with idx in [0, b*s*(d-2*rotary_half)) handle the copy.
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

// PLAN 4 Task 4: QK-rope + KV-append fusion for the K/V pair. Rotates K rows
// (QK-norm + NeoX RoPE, bit-identical to grim_qk_rope_dev_base) and appends
// the rotated K rows AND the raw V rows to their arenas at pos_dev
// (bit-identical to two grim_kv_append launches: same offsets, same f16
// conversion). Thread mapping is pair-identical to qk_rope so the norm
// shuffle reduction is order-exact; all arena writes are order-independent
// (one writer per address). Q rope stays separate (no append); bump stays
// after attention (which reads pre-bump total_dev).
extern "C" __global__ void grim_qk_rope_append_kv(
    const float* k, const unsigned int* pos_base, const float* gamma_k,
    const float* v,
    float* k_out, float* k_arena, float* v_arena,
    int b, int s, int d, int half, float base, int interleaved,
    int nkv_heads, float eps,
    int kv_stride, int steps, int arena_slot_stride, int f16) {
    int total = b * s * half;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int bi = idx / (s * half);
    int rem = idx - bi * (s * half);
    int si = rem / half;
    int i = rem - si * half;
    int step_idx = si / nkv_heads;
    float pos = (float)(pos_base[bi] + (unsigned int)step_idx);
    float freq = 1.0f / powf(base, (2.0f * (float)i) / (float)d);
    float val = pos * freq;
    float sin_val = sinf(val);
    float cos_val = cosf(val);
    int base_idx = (bi * s + si) * d;
    int a_idx = interleaved ? (base_idx + 2 * i) : (base_idx + i);
    int b_idx = interleaved ? (base_idx + 2 * i + 1) : (base_idx + half + i);
    float x1 = k[a_idx];
    float x2 = k[b_idx];

    // Per-head RMS scale: identical mapping and order to qk_rope.
    float ss = x1 * x1 + x2 * x2;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        ss += __shfl_xor_sync(0xffffffffffffffffULL, ss, off);
    float inv_rms = rsqrtf(ss / (float)d + eps);
    x1 *= gamma_k[a_idx - base_idx];
    x2 *= gamma_k[b_idx - base_idx];
    x1 *= inv_rms;
    x2 *= inv_rms;

    float r1 = x1 * cos_val - x2 * sin_val;
    float r2 = x2 * cos_val + x1 * sin_val;
    k_out[a_idx] = r1;
    k_out[b_idx] = r2;

    // Append rotated K pair + raw V pair at the on-device offset, replicating
    // grim_kv_append addressing (off = slot*slot_stride + past*stride + local).
    // si rows are (step,kv_head) pairs: local = si*hd + e = step*kv_stride +
    // khi*hd + e, so the head term must be added explicitly.
    int past = (int)pos_base[bi];
    int khi = si - step_idx * nkv_heads;
    long long row_off = (long long)bi * arena_slot_stride + (long long)(past + step_idx) * kv_stride + (long long)khi * d;
    int e1 = a_idx - base_idx;
    int e2 = b_idx - base_idx;
    if (f16) {
        unsigned short* a16 = (unsigned short*)k_arena;
        unsigned short* v16 = (unsigned short*)v_arena;
        a16[row_off + e1] = f32_to_fp16_bits_device(r1);
        a16[row_off + e2] = f32_to_fp16_bits_device(r2);
        v16[row_off + e1] = f32_to_fp16_bits_device(v[a_idx]);
        v16[row_off + e2] = f32_to_fp16_bits_device(v[b_idx]);
    } else {
        k_arena[row_off + e1] = r1;
        k_arena[row_off + e2] = r2;
        v_arena[row_off + e1] = v[a_idx];
        v_arena[row_off + e2] = v[b_idx];
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
// `inputs` is a device array of `n_inputs` device pointers (each points to `n_elements` floats on the device).
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

// On-device all_reduce accumulator for F16: out[i] = sum_k inputs[k][i], accumulated in float.
extern "C" __global__ void grim_all_reduce_accum_f16(
    __half* out,
    const __half* const* inputs,
    int n_inputs,
    int n_elements
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;
    float acc = 0.0f;
    for (int k = 0; k < n_inputs; ++k) {
        acc += __half2float(inputs[k][i]);
    }
    out[i] = __float2half(acc);
}

// On-device all_reduce accumulator for BF16: out[i] = sum_k inputs[k][i], accumulated in float.
extern "C" __global__ void grim_all_reduce_accum_bf16(
    hip_bfloat16* out,
    const hip_bfloat16* const* inputs,
    int n_inputs,
    int n_elements
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;
    float acc = 0.0f;
    for (int k = 0; k < n_inputs; ++k) {
        acc += float(inputs[k][i]);
    }
    out[i] = hip_bfloat16(acc);
}

// SPEED-ROC-10: multi-tensor (foreach) fused AdamW step. One launch updates
// every parameter tensor in the step: p/g/m/v are device arrays of per-tensor
// pointers, `offsets` is a device [n_tensors+1] prefix-sum of element counts
// over the concatenated virtual buffer, and each thread binary-searches its
// flat index to a tensor. Same update math as grim_fused_adamw_step.
extern "C" __global__ void grim_fused_adamw_step_foreach(
    float* const* __restrict__ p_list,
    float* const* __restrict__ g_list,
    float* const* __restrict__ m_list,
    float* const* __restrict__ v_list,
    const int* __restrict__ offsets,
    int n_tensors,
    long long total,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bc1,
    float bc2
) {
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride) {
        int lo = 0, hi = n_tensors - 1, t = -1;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            if (i >= (long long)offsets[mid + 1]) lo = mid + 1;
            else if (i < (long long)offsets[mid]) hi = mid - 1;
            else { t = mid; break; }
        }
        if (t < 0) continue;

        const long long local = i - (long long)offsets[t];
        float* p = p_list[t];
        const float* g = g_list[t];
        float* m = m_list[t];
        float* v = v_list[t];

        const float grad = g[local];
        const float m_val = beta1 * m[local] + (1.0f - beta1) * grad;
        const float v_val = beta2 * v[local] + (1.0f - beta2) * grad * grad;
        m[local] = m_val;
        v[local] = v_val;

        const float m_hat = m_val / bc1;
        const float v_hat = v_val / bc2;
        const float param_val = p[local];
        p[local] = param_val - lr * ((m_hat / (sqrtf(v_hat) + eps)) + weight_decay * param_val);
    }
}

// Warp-per-row RMS norm: one warp owns a row; the sum of squares reduces with 5 __shfl_xor butterflies (no barriers).
// The previous one-thread-per- element form made EVERY thread walk the whole row - O(row_len^2) loads.
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

// Warp-per-row LayerNorm (mean/variance, unlike RMS norm); bias pointer may be NULL.
// Used by per-head Q/K norms (Chameleon swin_norm) inside decode-graph capture.
extern "C" __global__ void __launch_bounds__(256)
grim_layer_norm(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
                float* __restrict__ out, int row_len, float eps, int total) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int rows = total / row_len;
    if (warp_id >= rows) return;
    const float* x_row = x + (size_t)warp_id * row_len;
    float* o_row = out + (size_t)warp_id * row_len;
    const unsigned long long shfl_mask = 0xffffffffffffffffULL;

    float sum = 0.0f;
    for (int col = lane; col < row_len; col += 32) {
        sum += x_row[col];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_xor_sync(shfl_mask, sum, off);
    const float mean = sum / (float)row_len;

    float var = 0.0f;
    for (int col = lane; col < row_len; col += 32) {
        float d = x_row[col] - mean;
        var += d * d;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        var += __shfl_xor_sync(shfl_mask, var, off);
    const float inv_std = rsqrtf(var / (float)row_len + eps);

    for (int col = lane; col < row_len; col += 32) {
        float val = (x_row[col] - mean) * inv_std * w[col];
        if (b != (const float*)0) val += b[col];
        o_row[col] = val;
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

// Q4_K row gather: out[i, :] = dequantized row indices[i].
//
// Keeps a large-vocabulary embedding table packed on device. `dim` must be a
// whole number of 256-element super-blocks (5120 = 20 blocks), so every row
// starts at a block boundary and the address math stays exact.
//
// The leaf decode is the same `dequant_q4k_element` the weight path uses, so
// there is exactly one Q4_K decoder to trust. Blocks are 144 bytes.
// Graph-capturable Q4_K embedding gather.
//
// Same shape as `grim_embedding_q4k`, but it takes the row count so it can
// BOUND-CHECK the device-resident index. This one runs inside a captured
// graph, where the host cannot validate the token ids (reading them back would
// sync and break capture), so the guard has to live in the kernel. The eager
// `embedding_q4k` checks on the host instead; this keeps that safety on the
// graph path rather than trading it for capture.
//
// An out-of-range id writes 0.0 rather than reading outside the table, so a
// bad token degrades the output instead of faulting the device.
extern "C" __global__ void grim_embedding_q4k_gather(
    const unsigned char* packed, float* out, int* indices, int dim, int total, int rows) {
    const int QK_BLOCK = 256;
    const int QK_BLOCK_BYTES = 144;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int i = idx / dim;
    int j = idx % dim;
    int row = indices[i];
    if (row < 0 || row >= rows) {
        out[idx] = 0.0f;
        return;
    }
    long long e = (long long)row * (long long)dim + (long long)j;
    const unsigned char* blk = packed + (e / QK_BLOCK) * QK_BLOCK_BYTES;
    out[idx] = dequant_q4k_element(blk, (int)(e % QK_BLOCK));
}

extern "C" __global__ void grim_embedding_q4k(const unsigned char* packed, float* out,
                                              int* indices, int dim, int total) {
    const int QK_BLOCK = 256;
    const int QK_BLOCK_BYTES = 144;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int i = idx / dim;
    int j = idx % dim;
    long long row = (long long)indices[i] * (long long)dim + (long long)j;
    const unsigned char* blk = packed + (row / QK_BLOCK) * QK_BLOCK_BYTES;
    out[idx] = dequant_q4k_element(blk, (int)(row % QK_BLOCK));
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

// Split-K partial reduction, dtype-specialized.
// The f16 kernel above is the historical entry point; F32 and BF16 GEMMs must reduce.
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

// PLAN-kernel-launch-reduction Phase C: fused ShortConv step. The in_proj
// GEMV output is [batch, 3*channels] laid out b|x|c per row; this kernel reads
// b/c/x directly from that buffer (no slice copies), computes bx = b*x in
// registers, runs the causal conv with in-place state update, and applies the
// c gate on the output: y = (conv(bx) . c). One launch replaces
// (3 slice copies + mul + conv + mul).
// GRAVE Phase 1: elementwise f32 -> f16 conversion for prefill writes into
// the f16 KV arena (replaces the plain D2D byte copy, which would corrupt).
extern "C" __global__ void grim_f32_to_f16(
    const float* __restrict__ src, unsigned short* __restrict__ dst,
    int src_off, int dst_off, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[dst_off + i] = f32_to_fp16_bits_device(src[src_off + i]);
}

extern "C" __global__ void grim_short_conv1d_fused_step(
    const float* proj, const float* weight, float* conv_state,
    float* y_out, int batch, int channels, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels;
    if (idx >= total) return;
    int b = idx / channels;
    int c = idx % channels;
    const float* row = proj + (long long)b * 3 * channels;

    float b_v = row[c];
    float c_v = row[channels + c];
    float x_v = row[2 * channels + c];
    float bx = b_v * x_v;

    int state_offset = (b * channels + c) * (kernel_size - 1);
    float sum = bx * weight[c * kernel_size + (kernel_size - 1)];
    for (int k = 0; k < kernel_size - 1; ++k) {
        sum += conv_state[state_offset + k] * weight[c * kernel_size + k];
    }
    y_out[idx] = sum * c_v;

    // Shift state buffer left and insert the new bx sample.
    for (int k = 0; k < kernel_size - 2; ++k) {
        conv_state[state_offset + k] = conv_state[state_offset + k + 1];
    }
    if (kernel_size > 1) {
        conv_state[state_offset + kernel_size - 2] = bx;
    }
}

// Universal GPU-side D2D copy kernel: bypasses driver hipMemcpy restrictions on virtual/unified memory
extern "C" __global__ void grim_copy_bytes(void* dst, const void* src, size_t num_bytes) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    // 8-byte transfers if aligned, else byte by byte
    if ((((uintptr_t)dst | (uintptr_t)src | num_bytes) & 7) == 0) {
        uint64_t* d64 = (uint64_t*)dst;
        const uint64_t* s64 = (const uint64_t*)src;
        size_t n64 = num_bytes / 8;
        for (size_t i = idx; i < n64; i += stride) {
            d64[i] = s64[i];
        }
    } else {
        uint8_t* d8 = (uint8_t*)dst;
        const uint8_t* s8 = (const uint8_t*)src;
        for (size_t i = idx; i < num_bytes; i += stride) {
            d8[i] = s8[i];
        }
    }
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

// Causal short conv scan across time steps t in 0..seq_len for prefill / multi-token steps.
// 1 thread per channel c. Correctly shifts and maintains conv_state across the sequence on GPU.
extern "C" __global__ void grim_short_conv1d_scan(
    const float* x_seq, const float* weight, const float* bias,
    float* conv_state, float* out_seq, int seq_len, int channels, int kernel_size
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;

    int state_offset = c * (kernel_size - 1);
    float b_val = bias ? bias[c] : 0.0f;
    float last_w = weight[c * kernel_size + (kernel_size - 1)];

    for (int t = 0; t < seq_len; ++t) {
        float val = x_seq[t * channels + c];
        float sum = val * last_w;
        for (int k = 0; k < kernel_size - 1; ++k) {
            sum += conv_state[state_offset + k] * weight[c * kernel_size + k];
        }
        sum += b_val;
        out_seq[t * channels + c] = sum;

        // Shift conv_state and insert current token sample
        for (int k = 0; k < kernel_size - 2; ++k) {
            conv_state[state_offset + k] = conv_state[state_offset + k + 1];
        }
        if (kernel_size > 1) {
            conv_state[state_offset + kernel_size - 2] = val;
        }
    }
}

// WhiteRaven / Raven format: FP8 E4M3 weights (stored as uint8_t byte array).
// Raven: V_DOT4_F32_FP8_FP8 / WhiteRaven: V_WMMA_F32_16X16X16_FP8_FP8.
// Dequantize in-register: E4M3 unpack (1 sign, 4 exponent with bias 7, 3 mantissa).
__device__ inline float dequant_fp8_e4m3(uint8_t byte) {
    if (byte == 0x7F || byte == 0xFF) return 0.0f; // NaN/Inf
    int sign = (byte >> 7) & 1;
    int exp = (byte >> 3) & 0x0F;
    int mant = byte & 0x07;
    float f;
    if (exp == 0) {
        // subnormal: 2^(-6) * (mant / 8)
        f = (mant / 8.0f) * 0.015625f;
    } else {
        // normal: 2^(exp - 7) * (1 + mant / 8)
        f = (1.0f + mant / 8.0f) * powf(2.0f, (float)(exp - 7));
    }
    return sign ? -f : f;
}

extern "C" __global__ void grim_short_conv1d_fp8_step(
    const float* x, const uint8_t* weight_fp8, const float* scale, const float* bias,
    float* conv_state, float* out, int batch, int channels, int kernel_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels;
    if (idx >= total) return;
    int b = idx / channels;
    int c = idx % channels;

    float s = scale ? scale[c] : 1.0f;
    float val = x[idx];
    int state_offset = (b * channels + c) * (kernel_size - 1);
    float last_w = dequant_fp8_e4m3(weight_fp8[c * kernel_size + (kernel_size - 1)]) * s;
    float sum = val * last_w;
    for (int k = 0; k < kernel_size - 1; ++k) {
        float wk = dequant_fp8_e4m3(weight_fp8[c * kernel_size + k]) * s;
        sum += conv_state[state_offset + k] * wk;
    }
    if (bias) {
        sum += bias[c];
    }
    out[idx] = sum;

    for (int k = 0; k < kernel_size - 2; ++k) {
        conv_state[state_offset + k] = conv_state[state_offset + k + 1];
    }
    if (kernel_size > 1) {
        conv_state[state_offset + kernel_size - 2] = val;
    }
}

// ForestRaven format: Q8_0 / INT8 signed 8-bit quantized weights with per-channel or per-block scale.
// Computes dot with FP32 conv state: w = scale * int8_code.
extern "C" __global__ void grim_short_conv1d_forestraven_step(
    const float* x, const int8_t* weight_i8, const float* scale, const float* bias,
    float* conv_state, float* out, int batch, int channels, int kernel_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels;
    if (idx >= total) return;
    int b = idx / channels;
    int c = idx % channels;

    float s = scale ? scale[c] : 1.0f;
    float val = x[idx];
    int state_offset = (b * channels + c) * (kernel_size - 1);
    float last_w = (float)weight_i8[c * kernel_size + (kernel_size - 1)] * s;
    float sum = val * last_w;
    for (int k = 0; k < kernel_size - 1; ++k) {
        float wk = (float)weight_i8[c * kernel_size + k] * s;
        sum += conv_state[state_offset + k] * wk;
    }
    if (bias) {
        sum += bias[c];
    }
    out[idx] = sum;

    for (int k = 0; k < kernel_size - 2; ++k) {
        conv_state[state_offset + k] = conv_state[state_offset + k + 1];
    }
    if (kernel_size > 1) {
        conv_state[state_offset + kernel_size - 2] = val;
    }
}

// Crow format: Q4_K GGML super-block (or signed 4-bit block quantized).
// 4-bit nibbles with scale and min/zero offset per channel or block.
extern "C" __global__ void grim_short_conv1d_crow_q4k_step(
    const float* x, const uint8_t* weight_q4, const float* scales, const float* mins,
    const float* bias, float* conv_state, float* out, int batch, int channels, int kernel_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels;
    if (idx >= total) return;
    int b = idx / channels;
    int c = idx % channels;

    float d = scales ? scales[c] : 1.0f;
    float m = mins ? mins[c] : 0.0f;
    float val = x[idx];
    int state_offset = (b * channels + c) * (kernel_size - 1);

    auto get_tap = [&](int tap) -> float {
        int flat_tap = c * kernel_size + tap;
        uint8_t byte = weight_q4[flat_tap / 2];
        uint8_t nibble = (flat_tap % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
        return d * (float)nibble - m;
    };

    float last_w = get_tap(kernel_size - 1);
    float sum = val * last_w;
    for (int k = 0; k < kernel_size - 1; ++k) {
        sum += conv_state[state_offset + k] * get_tap(k);
    }
    if (bias) {
        sum += bias[c];
    }
    out[idx] = sum;

    for (int k = 0; k < kernel_size - 2; ++k) {
        conv_state[state_offset + k] = conv_state[state_offset + k + 1];
    }
    if (kernel_size > 1) {
        conv_state[state_offset + kernel_size - 2] = val;
    }
}

// WhiteCrow format: Unsigned INT4 weights (packed 2 nibbles per byte or 8 nibbles per u32)
// with scale and zero-point.
// w_unpacked = scale * (nibble - zero)
extern "C" __global__ void grim_short_conv1d_iu4_step(
    const float* x, const uint8_t* weight_packed, const float* scale, const uint8_t* zero,
    const float* bias, float* conv_state, float* out, int batch, int channels, int kernel_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels;
    if (idx >= total) return;
    int b = idx / channels;
    int c = idx % channels;

    float sc = scale ? scale[c] : 1.0f;
    float zp = zero ? (float)zero[c] : 0.0f;
    float val = x[idx];
    int state_offset = (b * channels + c) * (kernel_size - 1);

    auto get_tap = [&](int tap) -> float {
        int flat_tap = c * kernel_size + tap;
        uint8_t byte = weight_packed[flat_tap / 2];
        uint8_t nibble = (flat_tap % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
        return sc * ((float)nibble - zp);
    };

    float last_w = get_tap(kernel_size - 1);
    float sum = val * last_w;
    for (int k = 0; k < kernel_size - 1; ++k) {
        sum += conv_state[state_offset + k] * get_tap(k);
    }
    if (bias) {
        sum += bias[c];
    }
    out[idx] = sum;

    for (int k = 0; k < kernel_size - 2; ++k) {
        conv_state[state_offset + k] = conv_state[state_offset + k + 1];
    }
    if (kernel_size > 1) {
        conv_state[state_offset + kernel_size - 2] = val;
    }
}

// Gated DeltaNet, one step per row of the value dimension.
//
// Published update (ICLR 2025, Eq. 10), matching the CPU reference in
// `grim-backend-cpu/src/device.rs`:
//
//     decay = exp(a_gate)
//     pred  = sum_k k * (decay * S)     <- decay applied BEFORE the dot
//     delta = beta * (v - pred)         <- beta scales the FULL delta term
//     S_new = decay * S + k * delta
//     out   = sum_k q * S_new
//
// The previous version diverged on all three points: it used sigmoid(a_gate)
// instead of exp, omitted decay from the key dot (so the error term was computed
// against a stale state), and folded beta inside as `v - beta*(k.S)` so beta
// never scaled the v term. Each is a different recurrence, not a rounding
// difference, and with a non-zero initial state the divergence is large — see
// tests/kda_delta_rule_parity.rs, which had no numeric coverage anywhere before.
//
// NOTE the decayed read: `decay * S_state[...]` must be used for BOTH the
// prediction and the carried state, so the decayed row is computed once and
// reused rather than recomputed inside the second loop.
extern "C" __global__ void grim_kda_gated_delta_rule_step(
    const float* q, const float* k, const float* v, const float* beta,
    const float* a_gate, float* S_state, float* out,
    int d_k, int d_v
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= d_v) return;

    // beta and a_gate are PER-CALL SCALARS, not per-row: the CPU reference
    // reads data()[0] for both. The previous kernel indexed them as
    // beta[row] / a_gate[row], so every row past 0 read past the end of a
    // one-element buffer and ran the recurrence on garbage. Row 0 happened to
    // be correct, which is why nothing caught it: there were no numeric tests.
    const float decay = expf(a_gate[0]);
    const float beta_val = beta[0];
    float* s_row = S_state + (long long)row * d_k;

    // pred = sum_k k * (decay * S)
    float pred = 0.0f;
    for (int col = 0; col < d_k; ++col) {
        pred += k[col] * (decay * s_row[col]);
    }

    // delta = beta * (v - pred): beta scales the whole delta term.
    const float delta = beta_val * (v[row] - pred);

    float y_val = 0.0f;
    for (int col = 0; col < d_k; ++col) {
        const float new_s = decay * s_row[col] + k[col] * delta;
        s_row[col] = new_s;
        y_val += q[col] * new_s;
    }
    out[row] = y_val;
}


// Batched Gated DeltaNet (KDA) decode step over EVERY value head of one token.
//
// `grim_kda_gated_delta_rule_step` above is the same recurrence for a single
// head, which forces one launch and one pair of slice views per head. This
// version takes the whole head loop so a decode token costs two launches per
// layer instead of 2 * num_value_heads, and so the [n_v][d_v][d_k] state can
// stay resident on the device rather than being read back each token.
//
// One thread per (value head, state row) pair. A thread owns row `j` of one
// head's row-major [d_v][d_k] state, so the delta rule needs no cross-thread
// communication - the same `d_k`/`d_v` split the CPU reference uses.
//
// The head output's RMS norm DOES reduce over d_v, and that value is not known
// until every row of the head has been updated, so this kernel writes the raw
// per-row accumulator and `grim_kda_head_norm_gate` finishes the head.
//
// The reference runs `ggml_silu` over the conv stream and only then takes the
// q/k/v views, so the SiLU is applied here on read: the same function, without
// a whole-stream elementwise launch per layer per token.
//
// Stream layout is [K key_dim][K key_dim][V value_dim] (llama.cpp qwen35.cpp);
// the key head for value head h is h % num_k (`ggml_repeat_4d` tiles the key
// heads across the value heads), and q and v are the SAME value-stream slice.
__device__ inline float kda_silu_dev(float x) {
    return x / (1.0f + expf(-x));
}

__device__ inline float kda_softplus_dev(float x) {
    // Mirrors the host `kda_softplus` branch for bit-comparable behaviour:
    // exp(x) overflows well before ln_1p does, and the small-x tail is
    // exactly exp(x) to within f32 precision.
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

extern "C" __global__ void grim_kda_gated_delta_rule_batched(
    const float* conv_out, const float* alpha, const float* beta,
    const float* dt_bias, const float* ssm_a,
    float* S, float* acc,
    int num_v, int num_k, int head_dim, float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_v * head_dim;
    if (idx >= total) return;
    int h = idx / head_dim;
    int j = idx - h * head_dim;

    // The conv stream is [q | k | v], NOT [k | k | v]. q and k are each
    // num_k heads wide and tile out to num_v heads (llama.cpp's
    // ggml_repeat_4d); v is num_v heads wide. llama.cpp qwen35.cpp:404-424 puts
    // q_conv at byte offset 0, k_conv at key_dim and v_conv at 2*key_dim; vLLM
    // qwen_gdn_linear_attn.py:704 states the same "[q, k, v, z] order".
    int key_dim = num_k * head_dim;
    const float* q_raw = conv_out + (h % num_k) * head_dim;
    const float* k_raw = conv_out + key_dim + (h % num_k) * head_dim;
    const float* v_raw = conv_out + 2 * key_dim + h * head_dim;

    // gate = softplus(alpha + dt_bias) * ssm_a, beta = sigmoid(beta).
    float gate = kda_softplus_dev(alpha[h] + dt_bias[h]) * ssm_a[h];
    float beta_val = 1.0f / (1.0f + expf(-beta[h]));
    float decay = expf(gate);

    // build_gdn_l2_norm on both k and q, per head.
    float kss = 0.0f, qss = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
        float kk = kda_silu_dev(k_raw[i]);
        float qq = kda_silu_dev(q_raw[i]);
        kss += kk * kk;
        qss += qq * qq;
    }
    float kden = sqrtf(kss + eps);
    float qden = sqrtf(qss + eps);

    float* s_row = S + ((long long)h * head_dim + j) * head_dim;

    // pred = sum_k k * (decay * S)   -- decay BEFORE the dot.
    float pred = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
        float kk = (kden > 0.0f) ? kda_silu_dev(k_raw[i]) / kden : kda_silu_dev(k_raw[i]);
        pred += kk * (decay * s_row[i]);
    }
    // delta = beta * (v - pred): beta scales the whole delta term.
    float delta = beta_val * (kda_silu_dev(v_raw[j]) - pred);

    float a = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
        float kk = (kden > 0.0f) ? kda_silu_dev(k_raw[i]) / kden : kda_silu_dev(k_raw[i]);
        float qq = (qden > 0.0f) ? kda_silu_dev(q_raw[i]) / qden : kda_silu_dev(q_raw[i]);
        float s = decay * s_row[i] + kk * delta;
        s_row[i] = s;
        a += qq * s;
    }
    // The reference scales the head output by 1/sqrt(S_v): `const float scale =
    // 1.0f / sqrtf((float) S_v);` and `attn_data[col] = attn_col * scale` in
    // gated_delta_net.cu:281. The chunked path folds the same factor into the
    // query instead (delta-net-base.cpp:47). The gated RMS norm downstream
    // cancels most of it, but not exactly, because of its eps.
    acc[idx] = a * rsqrtf((float) head_dim);
}

// Second half of the batched KDA step: one thread per value head reduces its
// d_v accumulators, then applies the reference's `build_norm_gated` - RMS over
// the whole head, the ssm_norm weight, then the silu(z) gate.
extern "C" __global__ void grim_kda_head_norm_gate(
    const float* acc, const float* norm_weight, const float* z,
    float* out, int num_v, int head_dim, float eps, int has_z
) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= num_v) return;

    float ss = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
        float a = acc[h * head_dim + i];
        ss += a * a;
    }
    float inv_rms = 1.0f / sqrtf(ss / head_dim + eps);

    for (int i = 0; i < head_dim; ++i) {
        float g = 1.0f;
        if (has_z) {
            float zv = z[h * head_dim + i];
            g = zv / (1.0f + expf(-zv));
        }
        out[h * head_dim + i] = acc[h * head_dim + i] * inv_rms * norm_weight[i] * g;
    }
}

// GPU KDA Recurrent Prefill scan kernel for seq_len > 1.
// Runs across all value heads and recurrent state rows over time t in 0..seq_len.
// Conv stream layout per token: [q (num_k * head_dim) | k (num_k * head_dim) | v (num_v * head_dim)].
extern "C" __global__ void grim_kda_gated_delta_rule_scan(
    const float* conv_out_seq,  // [seq_len, conv_dim]
    const float* alpha_seq,     // [seq_len, num_v]
    const float* beta_seq,      // [seq_len, num_v]
    const float* dt_bias,       // [num_v]
    const float* ssm_a,         // [num_v]
    const float* norm_weight,   // [head_dim]
    const float* z_seq,         // [seq_len, num_v * head_dim] optional
    float* S,                   // [num_v, head_dim, head_dim] persistent recurrent state
    float* out_seq,             // [seq_len, num_v * head_dim] output branch
    int seq_len, int num_v, int num_k, int head_dim, float eps, int has_z
) {
    // One thread per (value_head h, state_row j)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_v * head_dim;
    if (idx >= total) return;
    int h = idx / head_dim;
    int j = idx - h * head_dim;

    int key_dim = num_k * head_dim;
    int value_dim = num_v * head_dim;
    int conv_dim = 2 * key_dim + value_dim;

    float* s_row = S + ((long long)h * head_dim + j) * head_dim;

    for (int t = 0; t < seq_len; ++t) {
        const float* conv_tok = conv_out_seq + (long long)t * conv_dim;
        const float* q_raw = conv_tok + (h % num_k) * head_dim;
        const float* k_raw = conv_tok + key_dim + (h % num_k) * head_dim;
        const float* v_raw = conv_tok + 2 * key_dim + h * head_dim;

        float alpha_t = alpha_seq[t * num_v + h];
        float beta_t = beta_seq[t * num_v + h];

        float gate = kda_softplus_dev(alpha_t + dt_bias[h]) * ssm_a[h];
        float beta_val = 1.0f / (1.0f + expf(-beta_t));
        float decay = expf(gate);

        // GDN L2 norm on k and q
        float kss = 0.0f, qss = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            float kk = kda_silu_dev(k_raw[i]);
            float qq = kda_silu_dev(q_raw[i]);
            kss += kk * kk;
            qss += qq * qq;
        }
        float kden = sqrtf(kss + eps);
        float qden = sqrtf(qss + eps);

        // pred = sum_k k * (decay * S)
        float pred = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            float kk = (kden > 0.0f) ? kda_silu_dev(k_raw[i]) / kden : kda_silu_dev(k_raw[i]);
            pred += kk * (decay * s_row[i]);
        }
        float delta = beta_val * (kda_silu_dev(v_raw[j]) - pred);

        // S_new = decay * S + k * delta
        float a = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            float kk = (kden > 0.0f) ? kda_silu_dev(k_raw[i]) / kden : kda_silu_dev(k_raw[i]);
            float qq = (qden > 0.0f) ? kda_silu_dev(q_raw[i]) / qden : kda_silu_dev(q_raw[i]);
            float s = decay * s_row[i] + kk * delta;
            s_row[i] = s;
            a += qq * s;
        }
        float a_scaled = a * rsqrtf((float)head_dim);

        // Temporary accumulator stored directly into output row before head norm
        out_seq[t * value_dim + h * head_dim + j] = a_scaled;
    }

    // Wait for all rows of all heads to complete per token before head-norm reduction
    __syncthreads();

    // Normalization & gating across each head
    if (j == 0) {
        for (int t = 0; t < seq_len; ++t) {
            float* head_out = out_seq + t * value_dim + h * head_dim;
            float ss = 0.0f;
            for (int i = 0; i < head_dim; ++i) {
                float a = head_out[i];
                ss += a * a;
            }
            float inv_rms = 1.0f / sqrtf(ss / head_dim + eps);

            const float* z_tok = z_seq ? (z_seq + t * value_dim + h * head_dim) : nullptr;
            for (int i = 0; i < head_dim; ++i) {
                float g = 1.0f;
                if (has_z && z_tok) {
                    float zv = z_tok[i];
                    g = zv / (1.0f + expf(-zv));
                }
                head_out[i] = head_out[i] * inv_rms * norm_weight[i] * g;
            }
        }
    }
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

// Reverse-mode autodiff for RMSNorm (salamander.md Phase 3 & G3): dx[i] = (w[i] / rms) * g[i] - x[i] *
// (sum_j g[j] * w[j] * x[j]) / (hidden_dim * rms^3) dw[i] = sum_rows g[row, i] * (x[row, i] / rms)
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

// Reverse-mode autodiff for Rotary Position Embedding (RoPE) (salamander.md Phase 3 & G3): Orthogonal rotation matrix backward is R(-theta).
// dx0 = g0 * cos + g1 * sin dx1 = -g0 * sin +.
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

// Reverse-mode autodiff for token-embedding lookup (salamander.md P3, the 4th fused backward kernel): dweight[token_ids[t], :] += out_grad[t, :].
// Two kernels: a plain zero-fill of the [vocab, hidden] gradient buffer, then a grid-strided atomic.
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

    #[test]
    fn test_all_reduce_accum_f16_bf16_presence() {
        assert!(
            OTHER_KERNEL_SOURCE.contains("grim_all_reduce_accum_f16"),
            "grim_all_reduce_accum_f16 kernel missing from OTHER_KERNEL_SOURCE"
        );
        assert!(
            OTHER_KERNEL_SOURCE.contains("grim_all_reduce_accum_bf16"),
            "grim_all_reduce_accum_bf16 kernel missing from OTHER_KERNEL_SOURCE"
        );
    }

    // ── SPEED-ROC: FP16 activation quantization tests ──────────────────────────

    #[test]
    fn source_contains_fp16_quantize_kernel() {
        assert!(
            OTHER_KERNEL_SOURCE.contains("grim_quantize_fp16"),
            "FP16 quantization kernel must be JIT-discoverable"
        );
        assert!(
            OTHER_KERNEL_SOURCE.contains("grim_dequantize_fp16"),
            "FP16 dequantization kernel must be JIT-discoverable"
        );
    }

    #[test]
    fn fp16_kernel_converts_types() {
        assert!(
            OTHER_KERNEL_SOURCE.contains("(_Float16)src[i]"),
            "quantize kernel must cast float to _Float16"
        );
        assert!(
            OTHER_KERNEL_SOURCE.contains("(float)src[i]"),
            "dequantize kernel must cast _Float16 to float"
        );
    }

    /// SPEED-DOT-OPFUSE (Phase 4a): `grim_rmsnorm_rope` must be present in OTHER_KERNEL_SOURCE.
    #[test]
    fn test_rmsnorm_rope_kernel_presence() {
        assert!(
            OTHER_KERNEL_SOURCE.contains("grim_rmsnorm_rope"),
            "grim_rmsnorm_rope kernel missing from OTHER_KERNEL_SOURCE"
        );
    }

    #[test]
    fn test_reduction_kernels_presence() {
        assert!(OTHER_KERNEL_SOURCE.contains("grim_reduce_sum_stage1"));
        assert!(OTHER_KERNEL_SOURCE.contains("grim_reduce_sum_stage2"));
        assert!(OTHER_KERNEL_SOURCE.contains("grim_reduce_max_stage1"));
        assert!(OTHER_KERNEL_SOURCE.contains("grim_reduce_max_stage2"));
        assert!(OTHER_KERNEL_SOURCE.contains("grim_argmax_stage1"));
        assert!(OTHER_KERNEL_SOURCE.contains("grim_argmax_stage2"));
    }
}
