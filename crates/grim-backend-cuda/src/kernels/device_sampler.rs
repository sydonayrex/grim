//! GPU-native stochastic sampling kernel for CUDA (WI-X3 parity).
//!
//! Ports `grim_sample_logits_stochastic` from grim-backend-rocm `device_sampler.rs`
//! verbatim — same splitmix32 RNG, top-k bisection (24 iterations), top-p
//! mass-bisection (24 iterations), and Gumbel-max sampling.
//!
//! Rust dispatch mirrors ROCm: `sample_logits_on_device` / `sample_logits_on_device_at`
//! return `Ok(None)` on unsupported inputs for CPU fallback.

pub const DEVICE_SAMPLER_SOURCE: &str = r#"
#define GRIM_SAMPLER_BLOCK 256

__device__ unsigned int grim_sampler_hash(unsigned int x) {
    // splitmix32 finalizer: full avalanche, independent per (seed, pos, thread, step).
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__device__ float grim_sampler_uniform(unsigned int seed, int position, int tid, int step) {
    unsigned int h = grim_sampler_hash(
        seed ^ ((unsigned int)position * 0x9e3779b9u)
             ^ ((unsigned int)tid      * 0x85ebca6bu)
             ^ ((unsigned int)step     * 0xc2b2ae35u));
    // Open interval (0,1): +0.5 keeps logf(u) and logf(-logf(u)) safe.
    return ((float)(h >> 8) + 0.5f) * (1.0f / 16777216.0f);
}

extern "C" __global__ void grim_sample_logits_stochastic(
    const float* __restrict__ logits,    // [vocab_size]
    unsigned int* __restrict__ out_token,// [1]
    int vocab_size,
    float temperature,                   // <= 0 -> greedy argmax
    int top_k,                           // 0 = disabled
    float top_p,                         // >= 1.0 = disabled
    unsigned int seed,
    int position
) {
    const int tid = threadIdx.x, block = blockDim.x;
    __shared__ float s_val[GRIM_SAMPLER_BLOCK];
    __shared__ int   s_idx[GRIM_SAMPLER_BLOCK];

    if (temperature <= 0.0f) {
        // Greedy: T->0 collapses softmax to point mass; avoid divide-by-zero.
        float lmax = -1e30f; int lidx = -1;
        for (int v = tid; v < vocab_size; v += block) {
            if (logits[v] > lmax) { lmax = logits[v]; lidx = v; }
        }
        s_val[tid] = lmax; s_idx[tid] = lidx;
        __syncthreads();
        for (int stride = block >> 1; stride > 0; stride >>= 1) {
            if (tid < stride && s_val[tid + stride] > s_val[tid]) {
                s_val[tid] = s_val[tid + stride]; s_idx[tid] = s_idx[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) out_token[0] = (s_idx[0] >= 0) ? (unsigned int)s_idx[0] : 0u;
        return;
    }

    const float inv_t = 1.0f / temperature;

    // Pass 1: global max and min of scaled logits
    float lmax = -1e30f, lmin = 1e30f;
    for (int v = tid; v < vocab_size; v += block) {
        float s = logits[v] * inv_t;
        if (s > lmax) lmax = s; if (s < lmin) lmin = s;
    }
    s_val[tid] = lmax; __syncthreads();
    for (int stride = block >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) s_val[tid] = fmaxf(s_val[tid], s_val[tid + stride]);
        __syncthreads();
    }
    const float s_max = s_val[0]; __syncthreads();
    s_val[tid] = lmin; __syncthreads();
    for (int stride = block >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) s_val[tid] = fminf(s_val[tid], s_val[tid + stride]);
        __syncthreads();
    }
    const float s_min = s_val[0]; __syncthreads();

    // Pass 2: top-k threshold via count-bisection (24 iterations)
    float t_k = -1e30f;
    if (top_k > 0 && top_k < vocab_size) {
        float lo = s_min, hi = s_max;
        for (int it = 0; it < 24; ++it) {
            float mid = 0.5f * (lo + hi);
            if (!(mid > lo) || !(mid < hi)) break;
            float cnt = 0.0f;
            for (int v = tid; v < vocab_size; v += block)
                cnt += (logits[v] * inv_t >= mid) ? 1.0f : 0.0f;
            s_val[tid] = cnt; __syncthreads();
            for (int stride = block >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) s_val[tid] += s_val[tid + stride];
                __syncthreads();
            }
            if (s_val[0] >= (float)top_k) lo = mid; else hi = mid;
            __syncthreads();
        }
        t_k = lo;
    }

    // Pass 3: top-p threshold via mass-bisection (24 iterations)
    float t_p = -1e30f;
    if (top_p < 1.0f) {
        float local_z = 0.0f;
        for (int v = tid; v < vocab_size; v += block) {
            float s = logits[v] * inv_t;
            if (s >= t_k) local_z += __expf(s - s_max);
        }
        s_val[tid] = local_z; __syncthreads();
        for (int stride = block >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) s_val[tid] += s_val[tid + stride];
            __syncthreads();
        }
        const float inv_z = 1.0f / fmaxf(s_val[0], 1e-30f); __syncthreads();
        if (inv_z >= top_p) {
            t_p = s_max;
        } else {
            float lo = s_min, hi = s_max;
            for (int it = 0; it < 24; ++it) {
                float mid = 0.5f * (lo + hi);
                if (!(mid > lo) || !(mid < hi)) break;
                float m = 0.0f;
                for (int v = tid; v < vocab_size; v += block) {
                    float s = logits[v] * inv_t;
                    if (s >= mid && s >= t_k) m += __expf(s - s_max);
                }
                s_val[tid] = m; __syncthreads();
                for (int stride = block >> 1; stride > 0; stride >>= 1) {
                    if (tid < stride) s_val[tid] += s_val[tid + stride];
                    __syncthreads();
                }
                if (s_val[0] * inv_z >= top_p) lo = mid; else hi = mid;
                __syncthreads();
            }
            t_p = lo;
        }
    }

    // Pass 4: Gumbel-max draw over filtered support
    float best_key = -1e30f; int best_v = -1; int n = 0;
    for (int v = tid; v < vocab_size; v += block, ++n) {
        float s = logits[v] * inv_t;
        if (s < t_k || s < t_p) continue;
        float u   = grim_sampler_uniform(seed, position, tid, n);
        float key = s - logf(-logf(u));
        if (key > best_key) { best_key = key; best_v = v; }
    }
    s_val[tid] = best_key; s_idx[tid] = best_v; __syncthreads();
    for (int stride = block >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && s_val[tid + stride] > s_val[tid]) {
            s_val[tid] = s_val[tid + stride]; s_idx[tid] = s_idx[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) out_token[0] = (s_idx[0] >= 0) ? (unsigned int)s_idx[0] : 0u;
}
"#;

/// Largest vocabulary the single-block design handles. Above this, caller should
/// CPU-sample instead (matches ROCm MAX_DEVICE_SAMPLER_VOCAB).
pub const MAX_DEVICE_SAMPLER_VOCAB: usize = 1 << 18; // 262 144

/// Block dimension compiled into the kernel (must match `GRIM_SAMPLER_BLOCK`).
pub const SAMPLER_BLOCK: u32 = 256;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_contains_stochastic_sampler() {
        assert!(DEVICE_SAMPLER_SOURCE.contains("grim_sample_logits_stochastic"));
        assert!(DEVICE_SAMPLER_SOURCE.contains("grim_sampler_hash"));
        assert!(DEVICE_SAMPLER_SOURCE.contains("grim_sampler_uniform"));
        assert!(DEVICE_SAMPLER_SOURCE.contains("top_k"));
        assert!(DEVICE_SAMPLER_SOURCE.contains("top_p"));
    }

    #[test]
    fn sampler_constants_match_kernel_defines() {
        // GRIM_SAMPLER_BLOCK == SAMPLER_BLOCK: verify the define and const agree.
        assert!(DEVICE_SAMPLER_SOURCE.contains("#define GRIM_SAMPLER_BLOCK 256"));
        assert_eq!(SAMPLER_BLOCK, 256);
    }
}
