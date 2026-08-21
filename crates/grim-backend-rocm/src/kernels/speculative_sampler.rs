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

} // extern "C"
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_source_contains_speculative_sampler() {
        assert!(KERNEL_SOURCE.contains("grim_speculative_rejection_sample"));
        assert!(KERNEL_SOURCE.contains("grim_sample_logits_argmax"));
    }
}
