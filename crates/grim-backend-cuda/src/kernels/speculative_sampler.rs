//! GPU speculative decoding rejection sampler + argmax/stochastic samplers.
//!
//! Ported from grim-backend-rocm `kernels/speculative_sampler.rs`.
//! Four kernels:
//!   1. `grim_speculative_rejection_sample` — modified rejection sampling + residual
//!   2. `grim_sample_logits_argmax`         — zero-CPU greedy argmax
//!   3. `grim_sample_stochastic`            — temperature + top-k + Gumbel-max
//!   4. `grim_speculative_tree_verify`      — Medusa/Eagle tree path verifier

pub const SPECULATIVE_SAMPLER_SOURCE: &str = r#"
extern "C" {

// ---------------------------------------------------------------------------
// Speculative Rejection Sampling
//
// Grid: (batch_size, 1)  Block: (256, 1)
// Verifies draft tokens against target probabilities using the standard
// modified rejection sampling algorithm. On rejection, samples from the
// residual distribution max(0, P - Q) device-side. No host round-trip.
// ---------------------------------------------------------------------------
__global__ void grim_speculative_rejection_sample(
    const float* __restrict__ target_probs,  // [batch, num_draft+1, vocab]
    const float* __restrict__ draft_probs,   // [batch, num_draft, vocab]
    const int*   __restrict__ draft_tokens,  // [batch, num_draft]
    const float* __restrict__ uniform_rands, // [batch, num_draft+1]
    int*   __restrict__ accepted_tokens,     // [batch, num_draft+1]
    int*   __restrict__ accepted_lens,       // [batch]
    int batch_size,
    int num_draft_tokens,
    int vocab_size
) {
    const int b   = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= batch_size) return;

    if (tid == 0) {
        int accepted = 0;
        bool rejected = false;

        for (int k = 0; k < num_draft_tokens; ++k) {
            const int tok = draft_tokens[b * num_draft_tokens + k];
            if (tok < 0 || tok >= vocab_size) { rejected = true; break; }

            const float p = target_probs[(b * (num_draft_tokens + 1) + k) * vocab_size + tok];
            const float q = draft_probs[(b * num_draft_tokens + k) * vocab_size + tok];
            const float r = uniform_rands[b * (num_draft_tokens + 1) + k];
            const float accept_prob = (q > 0.0f) ? fminf(1.0f, p / q) : 1.0f;

            accepted_tokens[b * (num_draft_tokens + 1) + accepted] = tok;

            if (r <= accept_prob) {
                accepted++;
            } else {
                // Sample from residual distribution
                rejected = true;
                float sum_res = 0.0f;
                for (int v = 0; v < vocab_size; ++v) {
                    float pv = target_probs[(b * (num_draft_tokens + 1) + k) * vocab_size + v];
                    float qv = draft_probs[(b * num_draft_tokens + k) * vocab_size + v];
                    sum_res += fmaxf(0.0f, pv - qv);
                }
                float sample_r = uniform_rands[b * (num_draft_tokens + 1) + num_draft_tokens]
                                 * (sum_res > 0.0f ? sum_res : 1.0f);
                float cum = 0.0f;
                int sampled = 0;
                for (int v = 0; v < vocab_size; ++v) {
                    float pv = target_probs[(b * (num_draft_tokens + 1) + k) * vocab_size + v];
                    float qv = draft_probs[(b * num_draft_tokens + k) * vocab_size + v];
                    cum += fmaxf(0.0f, pv - qv);
                    if (cum >= sample_r) { sampled = v; break; }
                }
                accepted_tokens[b * (num_draft_tokens + 1) + accepted] = sampled;
                accepted++;
                break;
            }
        }

        if (!rejected) {
            // All draft tokens accepted — sample bonus from target[num_draft_tokens]
            float r = uniform_rands[b * (num_draft_tokens + 1) + num_draft_tokens];
            float cum = 0.0f;
            int bonus = 0;
            for (int v = 0; v < vocab_size; ++v) {
                cum += target_probs[(b * (num_draft_tokens + 1) + num_draft_tokens) * vocab_size + v];
                if (cum >= r) { bonus = v; break; }
            }
            accepted_tokens[b * (num_draft_tokens + 1) + accepted] = bonus;
            accepted++;
        }

        accepted_lens[b] = accepted;
    }
}

// ---------------------------------------------------------------------------
// Greedy Argmax Sampler
//
// Grid: (batch_size, 1)  Block: (256, 1)
// Finds argmax per row via parallel reduction. D2H only the token id.
// ---------------------------------------------------------------------------
__global__ void grim_sample_logits_argmax(
    const float* __restrict__ logits, // [batch, vocab]
    int*   __restrict__ out_tokens,   // [batch]
    int batch_size,
    int vocab_size
) {
    const int b   = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= batch_size) return;

    __shared__ float s_max[256];
    __shared__ int   s_idx[256];

    const float* row = logits + b * vocab_size;
    float lmax = -1e30f;
    int   lidx = 0;
    for (int v = tid; v < vocab_size; v += blockDim.x) {
        if (row[v] > lmax) { lmax = row[v]; lidx = v; }
    }
    s_max[tid] = lmax; s_idx[tid] = lidx;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && s_max[tid + stride] > s_max[tid]) {
            s_max[tid] = s_max[tid + stride];
            s_idx[tid] = s_idx[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) out_tokens[b] = s_idx[0];
}

// ---------------------------------------------------------------------------
// Stochastic Sampler — temperature + top-k + Gumbel-max (WI-X3)
//
// Grid: (batch_size, 1)  Block: (256, 1)
// Same RNG and bisection algorithm as ROCm version. Reproducible:
// (logits, seed) -> same token on any run.
// ---------------------------------------------------------------------------
__global__ void grim_sample_stochastic(
    const float* __restrict__ logits, // [batch, vocab]
    int*   __restrict__ out_tokens,   // [batch]
    float temperature,
    int top_k,
    unsigned long long seed,
    int batch_size,
    int vocab_size
) {
    const int b   = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= batch_size) return;

    const float* row = logits + b * vocab_size;
    __shared__ float s_val[256];

    // Per-thread LCG stream
    unsigned long long rng = seed * 6364136223846793005ULL
        + (unsigned long long)(b * blockDim.x + tid) * 1442695040888963407ULL;

    // Top-k threshold via count-bisection
    float threshold = -1e30f;
    float t_lo = -1e30f, t_hi = 1e30f;
    if (top_k > 0 && top_k < vocab_size) {
        for (int it = 0; it < 64; ++it) {
            float mid = 0.5f * (t_lo + t_hi);
            float cnt = 0.0f;
            for (int v = tid; v < vocab_size; v += blockDim.x)
                if (row[v] >= mid) cnt += 1.0f;
            s_val[tid] = cnt;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) s_val[tid] += s_val[tid + stride];
                __syncthreads();
            }
            if ((int)s_val[0] >= top_k) t_lo = mid; else t_hi = mid;
            threshold = t_lo;
        }
    }

    // Gumbel-max over surviving candidates
    float best = -1e30f;
    int   best_tok = 0;
    for (int v = tid; v < vocab_size; v += blockDim.x) {
        if (row[v] < threshold) continue;
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        float u = (float)((rng >> 11) & 0x1FFFFFFF) * (1.0f / (float)0x20000000);
        u = fmaxf(u, 1e-7f);
        float score = row[v] / fmaxf(temperature, 1e-6f) - logf(-logf(u));
        if (score > best) { best = score; best_tok = v; }
    }

    s_val[tid] = best;
    __shared__ int s_idx[256];
    s_idx[tid] = best_tok;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && s_val[tid + stride] > s_val[tid]) {
            s_val[tid] = s_val[tid + stride];
            s_idx[tid] = s_idx[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) out_tokens[b] = s_idx[0];
}

// ---------------------------------------------------------------------------
// Tree Speculative Verifier (Medusa / Eagle)
//
// Grid: (num_paths/32, 1)  Block: (num_paths % 32 or 32, 1)
// Evaluates all candidate tree paths in parallel, picks the longest accepted
// prefix via atomicMax. No sequential host round-trips between tree steps.
// ---------------------------------------------------------------------------
__global__ void grim_speculative_tree_verify(
    const int*   __restrict__ candidate_paths,  // [num_paths, max_path_len]
    const float* __restrict__ target_logits,    // [max_path_len, vocab]
    const int*   __restrict__ path_lens,        // [num_paths]
    int*   __restrict__ best_path_idx,          // [1]
    int*   __restrict__ best_accepted_len,      // [1]
    int num_paths,
    int max_path_len,
    int vocab_size
) {
    __shared__ int s_best_len;
    __shared__ int s_best_idx;
    if (threadIdx.x == 0 && blockIdx.x == 0) { s_best_len = 0; s_best_idx = 0; }
    __syncthreads();

    const int path_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_id >= num_paths) return;

    const int plen = path_lens[path_id];
    int accepted = 0;

    for (int step = 0; step < plen && step < max_path_len; ++step) {
        int draft_tok = candidate_paths[path_id * max_path_len + step];
        const float* step_logits = target_logits + step * vocab_size;
        float mval = -1e30f; int marg = 0;
        for (int v = 0; v < vocab_size; ++v) {
            if (step_logits[v] > mval) { mval = step_logits[v]; marg = v; }
        }
        if (draft_tok == marg) accepted++;
        else break;
    }

    atomicMax(&s_best_len, accepted);
    __syncthreads();
    if (accepted == s_best_len && threadIdx.x == 0) {
        atomicExch(best_accepted_len, s_best_len);
        atomicExch(best_path_idx, path_id);
    }
}

} // extern "C"
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_contains_all_sampling_kernels() {
        assert!(SPECULATIVE_SAMPLER_SOURCE.contains("grim_speculative_rejection_sample"));
        assert!(SPECULATIVE_SAMPLER_SOURCE.contains("grim_sample_logits_argmax"));
        assert!(SPECULATIVE_SAMPLER_SOURCE.contains("grim_sample_stochastic"));
        assert!(SPECULATIVE_SAMPLER_SOURCE.contains("grim_speculative_tree_verify"));
    }
}
