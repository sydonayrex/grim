//! GPU-Accelerated Speculative Decoding Rejection Sampler for ROCm.
//!
//! Performs device-side modified rejection sampling and residual distribution
//! recovery sampling without host-device synchronization round-trips.

pub const KERNEL_SOURCE: &str = r#"
extern "C" {

// ---------------------------------------------------------------------------
// GPU Speculative Rejection Sampling Kernel
// ---------------------------------------------------------------------------
//
// Grid: (batch_size, 1)
// Block: (256, 1) — thread index mapped across vocab reduction
// ---------------------------------------------------------------------------
__global__ void grim_speculative_rejection_sample(
    const float* __restrict__ target_probs,   // [batch_size, num_draft_tokens + 1, vocab_size]
    const float* __restrict__ draft_probs,    // [batch_size, num_draft_tokens, vocab_size]
    const int* __restrict__ draft_tokens,     // [batch_size, num_draft_tokens]
    const float* __restrict__ uniform_rands,  // [batch_size, num_draft_tokens + 1]
    int* __restrict__ accepted_tokens,        // [batch_size, num_draft_tokens + 1]
    int* __restrict__ accepted_lens,          // [batch_size]
    int batch_size,
    int num_draft_tokens,
    int vocab_size
) {
    const int b = blockIdx.x; // batch index
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    if (b >= batch_size) return;

    // Single-thread coordinator for sequential draft verification
    if (tid == 0) {
        int accepted_count = 0;
        bool rejected = false;

        for (int k = 0; k < num_draft_tokens; ++k) {
            const int draft_tok = draft_tokens[b * num_draft_tokens + k];
            if (draft_tok < 0 || draft_tok >= vocab_size) {
                rejected = true;
                break;
            }

            const float p = target_probs[(b * (num_draft_tokens + 1) + k) * vocab_size + draft_tok];
            const float q = draft_probs[(b * num_draft_tokens + k) * vocab_size + draft_tok];
            const float r = uniform_rands[b * (num_draft_tokens + 1) + k];

            const float accept_prob = (q > 0.0f) ? fminf(1.0f, p / q) : 1.0f;

            if (r <= accept_prob) {
                // Accepted token
                accepted_tokens[b * (num_draft_tokens + 1) + accepted_count] = draft_tok;
                accepted_count++;
            } else {
                // Rejected at position k: sample from residual distribution max(0, P - Q)
                rejected = true;
                float sum_res = 0.0f;
                for (int v = 0; v < vocab_size; ++v) {
                    float pv = target_probs[(b * (num_draft_tokens + 1) + k) * vocab_size + v];
                    float qv = draft_probs[(b * num_draft_tokens + k) * vocab_size + v];
                    float diff = fmaxf(0.0f, pv - qv);
                    sum_res += diff;
                }

                float sample_r = uniform_rands[b * (num_draft_tokens + 1) + num_draft_tokens] * (sum_res > 0.0f ? sum_res : 1.0f);
                float cum_sum = 0.0f;
                int sampled_tok = 0;
                for (int v = 0; v < vocab_size; ++v) {
                    float pv = target_probs[(b * (num_draft_tokens + 1) + k) * vocab_size + v];
                    float qv = draft_probs[(b * num_draft_tokens + k) * vocab_size + v];
                    cum_sum += fmaxf(0.0f, pv - qv);
                    if (cum_sum >= sample_r) {
                        sampled_tok = v;
                        break;
                    }
                }
                accepted_tokens[b * (num_draft_tokens + 1) + accepted_count] = sampled_tok;
                accepted_count++;
                break;
            }
        }

        // If all draft tokens accepted, sample bonus token from target_probs[num_draft_tokens]
        if (!rejected) {
            float sample_r = uniform_rands[b * (num_draft_tokens + 1) + num_draft_tokens];
            float cum_sum = 0.0f;
            int bonus_tok = 0;
            for (int v = 0; v < vocab_size; ++v) {
                cum_sum += target_probs[(b * (num_draft_tokens + 1) + num_draft_tokens) * vocab_size + v];
                if (cum_sum >= sample_r) {
                    bonus_tok = v;
                    break;
                }
            }
            accepted_tokens[b * (num_draft_tokens + 1) + accepted_count] = bonus_tok;
            accepted_count++;
        }

        accepted_lens[b] = accepted_count;
    }
}

// ---------------------------------------------------------------------------
// GPU Direct Logits Argmax Sampler (Zero-CPU Greedy Token Selection)
// ---------------------------------------------------------------------------
// Grid: (batch_size, 1)
// Block: (256, 1)
// ---------------------------------------------------------------------------
__global__ void grim_sample_logits_argmax(
    const float* __restrict__ logits,  // [batch_size, vocab_size]
    int* __restrict__ out_tokens,       // [batch_size]
    int batch_size,
    int vocab_size
) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= batch_size) return;

    const float* b_logits = &logits[b * vocab_size];

    __shared__ float s_max_val[256];
    __shared__ int s_max_idx[256];

    float local_max = -1e30f;
    int local_idx = 0;

    for (int v = tid; v < vocab_size; v += blockDim.x) {
        float val = b_logits[v];
        if (val > local_max) {
            local_max = val;
            local_idx = v;
        }
    }

    s_max_val[tid] = local_max;
    s_max_idx[tid] = local_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_max_val[tid + stride] > s_max_val[tid]) {
                s_max_val[tid] = s_max_val[tid + stride];
                s_max_idx[tid] = s_max_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_tokens[b] = s_max_idx[0];
    }
}

// ---------------------------------------------------------------------------
// GPU Stochastic Sampler (WI-X3): temperature + top-k filter + Gumbel-max
// sampled argmax, all on device. D2H transfers only the chosen token id.
// Grid: (batch_size, 1)  Block: (256, 1)
// Determinism: `seed` drives a per-thread Philox-style LCG so the same
// (logits, seed) pair yields the same token — testable against the CPU
// reference without statistical tolerance.
// ---------------------------------------------------------------------------
__global__ void grim_sample_stochastic(
    const float* __restrict__ logits,   // [batch_size, vocab_size]
    int* __restrict__ out_tokens,       // [batch_size]
    float temperature,
    int top_k,
    unsigned long long seed,
    int batch_size,
    int vocab_size
) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= batch_size) return;

    const float* b_logits = &logits[b * vocab_size];

    __shared__ float s_val[256];
    __shared__ int s_idx[256];

    // Per-thread LCG stream (cheap, reproducible from `seed`).
    unsigned long long rng = seed * 6364136223846793005ULL
        + (unsigned long long)(b * blockDim.x + tid) * 1442695040888963407ULL;

    // Pass 1: top-k via threshold refinement. For top_k==0 (disabled) the
    // threshold is -inf and every logit passes. We compute the k-th largest
    // value with a two-pass histogram-free approach: first find the max, then
    // binary-search a threshold whose pass-count is >= top_k. For simplicity
    // and single-wave friendliness we do: collect max; if top_k>0, find the
    // k-th largest via iterative threshold narrowing on the block.
    float t_lo = -1e30f, t_hi = 1e30f;
    float threshold = -1e30f;
    if (top_k > 0 && top_k < vocab_size) {
        // Narrow the threshold until at least top_k logits pass. 64 iterations
        // of bisection over the float range is ample for logit magnitudes.
        for (int it = 0; it < 64; ++it) {
            float mid = 0.5f * (t_lo + t_hi);
            int count = 0;
            for (int v = tid; v < vocab_size; v += blockDim.x) {
                if (b_logits[v] >= mid) count++;
            }
            // Block-wide sum of counts.
            s_val[tid] = (float)count;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (tid < stride) s_val[tid] += s_val[tid + stride];
                __syncthreads();
            }
            if ((int)s_val[0] >= top_k) t_lo = mid; else t_hi = mid;
            threshold = t_lo;
        }
    }

    // Pass 2: Gumbel-max over surviving candidates.
    //   score(v) = logit(v)/temperature + gumbel_noise(v)
    // The max of Gumbel-perturbed scores is a sample from softmax(logits/T).
    float local_best = -1e30f;
    int local_tok = 0;
    for (int v = tid; v < vocab_size; v += blockDim.x) {
        float val = b_logits[v];
        if (val < threshold) continue;
        // Uniform in (0, 1] from LCG, then inverse-CDF to Gumbel.
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        float u = (float)((rng >> 11) & 0x1FFFFFFF) / (float)0x20000000;
        u = fmaxf(u, 1e-7f);
        float g = -logf(-logf(u));
        float score = val / fmaxf(temperature, 1e-6f) + g;
        if (score > local_best) {
            local_best = score;
            local_tok = v;
        }
    }

    s_val[tid] = local_best;
    s_idx[tid] = local_tok;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_val[tid + stride] > s_val[tid]) {
                s_val[tid] = s_val[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_tokens[b] = s_idx[0];
    }
}

} // extern "C"
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_source_contains_speculative_sampler() {
        assert!(KERNEL_SOURCE.contains("grim_speculative_rejection_sample"));
        assert!(KERNEL_SOURCE.contains("grim_sample_logits_argmax"));
        assert!(KERNEL_SOURCE.contains("grim_sample_stochastic"));
    }
}
