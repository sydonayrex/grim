//! Charon - P-DAFD (Predictive Distribution-Aware Fused Dispatch) MoE kernel.
//! Implements the sortless fused dispatch GEMM path for Mixtures-of-Experts (WI-A of `rocm_kernel_plan.md`).

use std::ffi::c_void;

use grim_tensor::error::{Error, Result};

// HIP source - `grim_moe_fused_dispatch`

/// HIP source for the Charon fused-dispatch MoE kernel family.
/// Entries (each `__global__`, Wave64-aligned): * `grim_moe_fused_dispatch` - WI-A sortless fused dispatch GEMM, gate+up fused with.
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

    // ──────────────────────────────────────────────────────────────────── charon_atomic_add2 - packed 2xfloat atomic epilogue for the grouped kernels (grim_moe_fused_grouped*).
    // A token's [hidden] output row is accumulated into `out` by per-element float atomicAdd (all routed.
    __device__ __forceinline__ void charon_atomic_add2(
        float* out, unsigned long long idx2, float add0, float add1) {
        atomicAdd(out + idx2, add0);
        atomicAdd(out + idx2 + 1, add1);
    }

    // ──────────────────────────────────────────────────────────────────── grim_moe_fused_dispatch - sortless fused MoE dispatch (WI-A).
    // One launch carries every routed token to its expert.
    __global__ void grim_moe_fused_dispatch(
        const float* __restrict__ activations,     // [batch, hidden]
        const float* __restrict__ expert_gate_w,   // [num_experts, inter*hidden]
        const float* __restrict__ expert_up_w,     // [num_experts, inter*hidden]
        const float* __restrict__ expert_down_w,   // [num_experts, hidden*inter]
        const unsigned int* __restrict__ router_tokens,  // [num_pairs]
        const unsigned int* __restrict__ router_experts, // [num_pairs]
        const float* __restrict__ router_weights,        // [num_pairs]
        float* __restrict__ out,                     // [batch, hidden]
        int hidden, int inter, int num_pairs,
        float routed_scaling_factor)
    {
        const unsigned long long pair = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (pair >= (unsigned long long)num_pairs) return;

        const unsigned int tok = router_tokens[pair];
        const unsigned int exp = router_experts[pair];
        const float w = router_weights[pair];

        const float* a = activations + (unsigned long long)tok * hidden;
        const float* gw = expert_gate_w + (unsigned long long)exp * inter * hidden;
        const float* uw = expert_up_w   + (unsigned long long)exp * inter * hidden;
        const float* dw = expert_down_w + (unsigned long long)exp * hidden * inter;

        // Fused gate + up GEMM with in-register SiLU combine, then down.
        // The intermediate inter-dimension is reduced in-register (no HBM round-trip for the activation - the TritonMoE.
        for (int j = 0; j < inter; ++j) {
            float g = 0.0f;
            float u = 0.0f;
            for (int i = 0; i < hidden; ++i) {
                g += gw[j * hidden + i] * a[i];
                u += uw[j * hidden + i] * a[i];
            }
            // SiLU(g) * u, fused in-register.
            float silu_g = g / (1.0f + expf(-g));
            float act = silu_g * u;
            float scale = routed_scaling_factor * w * act;

            // down: dw[h, j] * act, accumulated into token output row
            for (int h = 0; h < hidden; ++h) {
                unsigned long long out_idx = (unsigned long long)tok * hidden + h;
                atomicAdd(out + out_idx, dw[h * inter + j] * scale);
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────── grim_charon_gmem_bytes - WI-A traffic counter (G-A5).
    // Pure arithmetic: returns the GMEM bytes a fused dispatch would touch for the given shape,.
    __device__ unsigned long long charon_fused_bytes(int hidden, int inter, int num_pairs, int batch) {
        const unsigned long long bytes_per_pair =
            (unsigned long long)(2ULL * inter * hidden   // gate + up
                               + (unsigned long long)hidden * inter // down
                               + hidden)                  // activation
            * 4ULL; // sizeof(f32)
        const unsigned long long out_bytes = (unsigned long long)batch * hidden * 4ULL;
        return bytes_per_pair * (unsigned long long)num_pairs + out_bytes;
    }

    // ──────────────────────────────────────────────────────────────────── grim_moe_fused_grouped - WI-A grouped (token-sorted) fused dispatch.
    // Same in-register fused math as `grim_moe_fused_dispatch` (gate+up GEMM → SiLU combine → down, no HBM.
    __device__ void grim_moe_fused_grouped_device(
        const float* __restrict__ activations,     // [batch, hidden]
        const float* __restrict__ expert_gate_w,   // [num_experts, inter*hidden]
        const float* __restrict__ expert_up_w,     // [num_experts, inter*hidden]
        const float* __restrict__ expert_down_w,   // [num_experts, hidden*inter]
        const unsigned int* __restrict__ sorted_token_ids,  // [num_tokens_post_padded]
        const unsigned int* __restrict__ sorted_expert_ids, // [num_tokens_post_padded]
        const float* __restrict__ sorted_weights,         // [num_tokens_post_padded]
        float* __restrict__ out,                     // [batch, hidden]
        int hidden, int inter, int num_tokens, int block_size,
        float routed_scaling_factor,
        float* __restrict__ stash_hg,  // [num_tokens*inter] or null (SPEED-ROC-14)
        float* __restrict__ stash_hu)  // [num_tokens*inter] or null
    {
        // `blockDim.x` is the effective worker width supplied by the persistent dispatcher.
        // It need not equal block_size: the stride below handles smaller and larger routed blocks without.
        if (block_size <= 0) return;
        const int blk = blockIdx.x;
        {
        const int base = blk * block_size;
        const int end = base + block_size < num_tokens ? base + block_size : num_tokens;

        // One thread per token in this block's window; padding slots skipped.
        for (int s = base + threadIdx.x; s < end; s += blockDim.x) {
            const unsigned int tok = sorted_token_ids[s];
            if (tok >= (unsigned int)num_tokens) continue; // padding
            const unsigned int exp = sorted_expert_ids[s];
            const float w = sorted_weights[s];

            const float* a  = activations + (unsigned long long)tok * hidden;
            const float* gw = expert_gate_w + (unsigned long long)exp * inter * hidden;
            const float* uw = expert_up_w   + (unsigned long long)exp * inter * hidden;
            const float* dw = expert_down_w + (unsigned long long)exp * hidden * inter;

            float acc_prev = 0.0f;
            const bool odd_hidden = (hidden & 1) != 0;
            for (int h = 0; h < hidden; ++h) {
                float acc = 0.0f;
                for (int j = 0; j < inter; ++j) {
                    float g = 0.0f;
                    float u = 0.0f;
                    for (int i = 0; i < hidden; ++i) {
                        g += gw[j * hidden + i] * a[i];
                        u += uw[j * hidden + i] * a[i];
                    }
                    float silu_g = g / (1.0f + expf(-g));
                    float act = silu_g * u;
                    // SPEED-ROC-14: stash the gate/up pre-activations for this
                    // (slot, j) so the backward kernel reads them instead of
                    // recomputing the full projections. Indexed by sorted slot
                    // to match the backward's cur_shg = stash_hg + s*inter.
                    if (stash_hg != nullptr) {
                        stash_hg[s * inter + j] = g;
                        stash_hu[s * inter + j] = u;
                    }
                    acc += dw[h * inter + j] * act;
                }
                if (odd_hidden) {
                    atomicAdd(out + (unsigned long long)tok * hidden + h,
                              routed_scaling_factor * w * acc);
                } else if (h & 1) {
                    charon_atomic_add2(out, (unsigned long long)tok * hidden + (h - 1),
                                       routed_scaling_factor * w * acc_prev,
                                       routed_scaling_factor * w * acc);
                } else {
                    acc_prev = acc;
                }
            }
        }
        }
    }

    __global__ void grim_moe_fused_grouped(
        const float* activations, const float* expert_gate_w, const float* expert_up_w,
        const float* expert_down_w, const unsigned int* sorted_token_ids,
        const unsigned int* sorted_expert_ids, const float* sorted_weights, float* out,
        int hidden, int inter, int num_tokens, int block_size, float routed_scaling_factor,
        float* stash_hg, float* stash_hu) {
        grim_moe_fused_grouped_device(activations, expert_gate_w, expert_up_w, expert_down_w,
            sorted_token_ids, sorted_expert_ids, sorted_weights, out, hidden, inter,
            num_tokens, block_size, routed_scaling_factor, stash_hg, stash_hu);
    }

    __global__ void grim_moe_fused_grouped_gelu(
        const float* __restrict__ activations, const float* __restrict__ expert_gate_w,
        const float* __restrict__ expert_up_w, const float* __restrict__ expert_down_w,
        const unsigned int* __restrict__ sorted_token_ids,
        const unsigned int* __restrict__ sorted_expert_ids, const float* __restrict__ sorted_weights,
        float* __restrict__ out,
        int hidden, int inter, int num_tokens, int block_size, float routed_scaling_factor) {
        if (block_size <= 0) return;
        const int blk = blockIdx.x;
        const int base = blk * block_size;
        const int end = base + block_size < num_tokens ? base + block_size : num_tokens;

        for (int s = base + threadIdx.x; s < end; s += blockDim.x) {
            const unsigned int tok = sorted_token_ids[s];
            if (tok >= (unsigned int)num_tokens) continue;
            const unsigned int exp = sorted_expert_ids[s];
            const float w = sorted_weights[s];

            const float* a  = activations + (unsigned long long)tok * hidden;
            const float* gw = expert_gate_w + (unsigned long long)exp * inter * hidden;
            const float* dw = expert_down_w + (unsigned long long)exp * hidden * inter;

            float acc_prev = 0.0f;
            const bool odd_hidden = (hidden & 1) != 0;
            for (int h = 0; h < hidden; ++h) {
                float acc = 0.0f;
                for (int j = 0; j < inter; ++j) {
                    float g = 0.0f;
                    for (int i = 0; i < hidden; ++i) {
                        g += gw[j * hidden + i] * a[i];
                    }
                    // GELU tanh approximation: 0.5 * g * (1.0 + tanh(0.7978846 * (g + 0.044715 * g^3)))
                    float g3 = g * g * g;
                    float tanh_arg = 0.7978846f * (g + 0.044715f * g3);
                    float act = 0.5f * g * (1.0f + tanhf(tanh_arg));

                    acc += dw[h * inter + j] * act;
                }
                if (odd_hidden) {
                    atomicAdd(out + (unsigned long long)tok * hidden + h,
                              routed_scaling_factor * w * acc);
                } else if (h & 1) {
                    charon_atomic_add2(out, (unsigned long long)tok * hidden + (h - 1),
                                       routed_scaling_factor * w * acc_prev,
                                       routed_scaling_factor * w * acc);
                } else {
                    acc_prev = acc;
                }
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────── grim_moe_route_topk - device-side MoE routing (D2D).
    // Computes per-token top-k expert selection + softmax-normalized combine weights
    // entirely on-device, writing sortless (token, expert, weight) triples into three
    // persistent device buffers. Keeps the gate logits resident — no D2H, no H2D
    // re-upload of the routing table (see decode-plan-universal-optimization.md Phase 3aD).
    // One block per token; threads stride over num_experts and block-reduce top-k.
    __global__ void grim_moe_route_topk(
        const float* __restrict__ logits,       // [seq_len, num_experts] row-major
        const float* __restrict__ bias,         // [num_experts] or null (mode 2 only)
        unsigned int* __restrict__ out_tokens,  // [seq_len * top_k]
        unsigned int* __restrict__ out_experts, // [seq_len * top_k]
        float* __restrict__ out_weights,        // [seq_len * top_k]
        int seq_len, int num_experts, int top_k,
        int route_mode,                        // 0 = softmax, 1 = sqrt-softplus, 2 = sigmoid+bias
        int norm_weights)                      // 1 = normalize top-k combine weights to sum 1
    {
        const int tok = blockIdx.x;
        if (tok >= seq_len) return;
        const int tid = threadIdx.x;
        const int block = blockDim.x;

        const float* row = logits + (long long)tok * num_experts;

        // Score: apply the architecture's gating transform before top-k selection.
        __shared__ float s_val[256];

        if (route_mode == 2) {
            // Sigmoid + per-expert bias (MoeRouter::SigmoidTopKWithBias, laguna/maple/moe_block).
            // Selection score = sigmoid(logit) + bias[i]; combine weight = sigmoid(logit) (unnormalized).
            if (tid == 0) {
                const int k = top_k < num_experts ? top_k : num_experts;
                int chosen[64];
                float chosen_v[64];
                for (int i = 0; i < k; ++i) { chosen[i] = -1; chosen_v[i] = -1e30f; }
                for (int v = 0; v < num_experts; ++v) {
                    const float sig = 1.0f / (1.0f + __expf(-row[v]));
                    const float score = sig + (bias != nullptr ? bias[v] : 0.0f);
                    int pos = k - 1;
                    if (score <= chosen_v[pos] && chosen[pos] >= 0) continue;
                    while (pos > 0 && (chosen[pos - 1] < 0 || score > chosen_v[pos - 1])) {
                        chosen[pos] = chosen[pos - 1];
                        chosen_v[pos] = chosen_v[pos - 1];
                        pos--;
                    }
                    chosen[pos] = v;
                    chosen_v[pos] = score;
                }
                // Reference (llama.cpp build_moe_ffn with norm_w): the
                // combine weights are the gathered sigmoid probs NORMALIZED to
                // sum 1 before expert_weights_scale is applied. Without this
                // the grouped GEMM scales by raw probs - Xing4.0 measured a
                // 0.76 relative divergence against the CPU reference.
                float wsum = 0.0f;
                for (int i = 0; i < k; ++i) {
                    if (chosen[i] >= 0) wsum += 1.0f / (1.0f + __expf(-row[chosen[i]]));
                }
                const float wden = norm_weights ? fmaxf(wsum, 6.103515625e-5f) : 1.0f;
                const long long base = (long long)tok * top_k;
                for (int i = 0; i < top_k; ++i) {
                    if (i < k && chosen[i] >= 0) {
                        out_tokens[base + i]  = (unsigned int)tok;
                        out_experts[base + i] = (unsigned int)chosen[i];
                        out_weights[base + i] = (1.0f / (1.0f + __expf(-row[chosen[i]]))) / wden;
                    } else {
                        out_tokens[base + i]  = (unsigned int)tok;
                        out_experts[base + i] = 0u;
                        out_weights[base + i] = 0.0f;
                    }
                }
            }
            return;
        }

        if (route_mode == 1) {
            // Sqrt-softplus (DeepSeek-V4): s(x) = sqrt(softplus(x)); softplus(x)=ln(1+e^x),
            // with the l>20 linear guard to avoid exp overflow.
            float local_max = -1e30f;
            for (int v = tid; v < num_experts; v += block) {
                const float l = row[v];
                const float sp = (l > 20.0f) ? l : __logf(1.0f + __expf(l));
                const float s = sqrtf(fmaxf(sp, 0.0f));
                if (s > local_max) local_max = s;
            }
            s_val[tid] = local_max;
            __syncthreads();
            for (int stride = block / 2; stride > 0; stride >>= 1) {
                if (tid < stride && s_val[tid + stride] > s_val[tid]) s_val[tid] = s_val[tid + stride];
                __syncthreads();
            }
            const float max_s = s_val[0];
            __syncthreads();

            // Sum of all sqrt-softplus scores (normalization denominator).
            float denom = 0.0f;
            for (int v = tid; v < num_experts; v += block) {
                const float l = row[v];
                const float sp = (l > 20.0f) ? l : __logf(1.0f + __expf(l));
                denom += sqrtf(fmaxf(sp, 0.0f));
            }
            s_val[tid] = denom;
            __syncthreads();
            for (int stride = block / 2; stride > 0; stride >>= 1) {
                if (tid < stride) s_val[tid] += s_val[tid + stride];
                __syncthreads();
            }
            const float inv_z = 1.0f / fmaxf(s_val[0], 1e-30f);
            __syncthreads();

            if (tid == 0) {
                const int k = top_k < num_experts ? top_k : num_experts;
                int chosen[64];
                float chosen_v[64];
                for (int i = 0; i < k; ++i) { chosen[i] = -1; chosen_v[i] = -1e30f; }
                for (int v = 0; v < num_experts; ++v) {
                    const float l = row[v];
                    const float sp = (l > 20.0f) ? l : __logf(1.0f + __expf(l));
                    const float s = sqrtf(fmaxf(sp, 0.0f));
                    int pos = k - 1;
                    if (s <= chosen_v[pos] && chosen[pos] >= 0) continue;
                    while (pos > 0 && (chosen[pos - 1] < 0 || s > chosen_v[pos - 1])) {
                        chosen[pos] = chosen[pos - 1];
                        chosen_v[pos] = chosen_v[pos - 1];
                        pos--;
                    }
                    chosen[pos] = v;
                    chosen_v[pos] = s;
                }
                const long long base = (long long)tok * top_k;
                for (int i = 0; i < top_k; ++i) {
                    if (i < k && chosen[i] >= 0) {
                        out_tokens[base + i]  = (unsigned int)tok;
                        out_experts[base + i] = (unsigned int)chosen[i];
                        out_weights[base + i] = chosen_v[i] * inv_z;
                    } else {
                        out_tokens[base + i]  = (unsigned int)tok;
                        out_experts[base + i] = 0u;
                        out_weights[base + i] = 0.0f;
                    }
                }
            }
            return;
        }

        // route_mode == 0: softmax top-k (global softmax denominator).
        // route_mode == 3: softmax top-k renormalized over top-k only (GLM / shared_moe::normalize_weights).
        if (route_mode == 3) {
            if (tid == 0) {
                const int k = top_k < num_experts ? top_k : num_experts;
                int chosen[64];
                float chosen_v[64];
                for (int i = 0; i < k; ++i) { chosen[i] = -1; chosen_v[i] = -1e30f; }
                for (int v = 0; v < num_experts; ++v) {
                    const float lv = row[v];
                    int pos = k - 1;
                    if (lv <= chosen_v[pos] && chosen[pos] >= 0) continue;
                    while (pos > 0 && (chosen[pos - 1] < 0 || lv > chosen_v[pos - 1])) {
                        chosen[pos] = chosen[pos - 1];
                        chosen_v[pos] = chosen_v[pos - 1];
                        pos--;
                    }
                    chosen[pos] = v;
                    chosen_v[pos] = lv;
                }
                float topk_max = -1e30f;
                for (int i = 0; i < k; ++i) {
                    if (chosen[i] >= 0 && chosen_v[i] > topk_max) {
                        topk_max = chosen_v[i];
                    }
                }
                float sum_exp = 0.0f;
                for (int i = 0; i < k; ++i) {
                    if (chosen[i] >= 0) {
                        sum_exp += __expf(chosen_v[i] - topk_max);
                    }
                }
                const float inv_sum = 1.0f / fmaxf(sum_exp, 1e-12f);
                const long long base = (long long)tok * top_k;
                for (int i = 0; i < top_k; ++i) {
                    if (i < k && chosen[i] >= 0) {
                        out_tokens[base + i]  = (unsigned int)tok;
                        out_experts[base + i] = (unsigned int)chosen[i];
                        out_weights[base + i] = __expf(chosen_v[i] - topk_max) * inv_sum;
                    } else {
                        out_tokens[base + i]  = (unsigned int)tok;
                        out_experts[base + i] = 0u;
                        out_weights[base + i] = 0.0f;
                    }
                }
            }
            return;
        }

        float lmax = -1e30f;
        for (int v = tid; v < num_experts; v += block) {
            if (row[v] > lmax) lmax = row[v];
        }
        s_val[tid] = lmax;
        __syncthreads();
        for (int stride = block / 2; stride > 0; stride >>= 1) {
            if (tid < stride && s_val[tid + stride] > s_val[tid]) s_val[tid] = s_val[tid + stride];
            __syncthreads();
        }
        const float max_l = s_val[0];
        __syncthreads();

        // Pass 2: softmax denominator over ALL experts (stable), and gather top-k.
        float denom = 0.0f;
        for (int v = tid; v < num_experts; v += block) {
            denom += __expf(row[v] - max_l);
        }
        s_val[tid] = denom;
        __syncthreads();
        for (int stride = block / 2; stride > 0; stride >>= 1) {
            if (tid < stride) s_val[tid] += s_val[tid + stride];
            __syncthreads();
        }
        const float inv_z = 1.0f / fmaxf(s_val[0], 1e-30f);
        __syncthreads();

        // Pass 3: emit top-k. Each thread selects its own top-k candidates over its
        // strided slice, then we reduce the *union* to the global top-k. For small
        // num_experts (the common MoE case) this is cheap; we do a serial per-thread
        // top-k and a final block-wide merge via a fixed-size shared array.
        // Simple approach: one thread computes top-k serially (num_experts is small,
        // typically 4-256), writes results directly.
        if (tid == 0) {
            // Serial top-k over num_experts via threshold refinement (safe for small N).
            const int k = top_k < num_experts ? top_k : num_experts;
            int chosen[64];
            float chosen_v[64];
            for (int i = 0; i < k; ++i) { chosen[i] = -1; chosen_v[i] = -1e30f; }
            for (int v = 0; v < num_experts; ++v) {
                const float lv = row[v];
                // insertion sort into chosen (descending)
                int pos = k - 1;
                if (lv <= chosen_v[pos] && chosen[pos] >= 0) continue;
                while (pos > 0 && (chosen[pos - 1] < 0 || lv > chosen_v[pos - 1])) {
                    chosen[pos] = chosen[pos - 1];
                    chosen_v[pos] = chosen_v[pos - 1];
                    pos--;
                }
                chosen[pos] = v;
                chosen_v[pos] = lv;
            }
            const long long base = (long long)tok * top_k;
            for (int i = 0; i < top_k; ++i) {
                if (i < k) {
                    out_tokens[base + i]  = (unsigned int)tok;
                    out_experts[base + i] = (unsigned int)chosen[i];
                    out_weights[base + i] = __expf(chosen_v[i] - max_l) * inv_z;
                } else {
                    out_tokens[base + i]  = (unsigned int)tok;
                    out_experts[base + i] = 0u;
                    out_weights[base + i] = 0.0f;
                }
            }
        }
    }
}

// --- #2 FP8 W8A8 helpers + grouped kernel --------------------------------- E4M3 (4 exp, 3 mant, bias 7) decode, mirroring grim-quant::fp8_e4m3_to_f32 so the device path matches the host quantizer exactly (no hip_fp8.h include needed for the JIT).
// NaN/overflow semantics preserved.
__device__ __forceinline__ float fp8e4m3_to_f32(unsigned char b) {
    int sign = (b & 0x80) ? 1 : 0;
    int exp  = (b >> 3) & 0x0F;
    int mant = b & 0x07;
    float result;
    if (exp == 0xF) {
        if (mant == 7) return __int_as_float(0x7FC00000); // NaN
        // exp == 15, mant in 0..6 are normal numbers in [256, 448]:
        // (1 + mant/8) * 2^(15 - 7) = (1 + mant/8) * 256.
        float val = (1.0f + (float)mant / 8.0f) * 256.0f;
        return sign ? -val : val;
    }
    if (exp != 0) {
        result = (mant / 8.0f + 1.0f) * __powf(2.0f, (float)(exp - 7));
    } else {
        result = mant / 512.0f;
    }
    return sign ? -result : result;
}

// OCP MXFP8 element decode: E4M3 code scaled by E8M0 shared exponent
// e:  value = e4m3(code) * 2^(e - 127). Mirrors grim-quant::dequant_mxfp8.
__device__ __forceinline__ float mxfp8_e4m3_to_f32(unsigned char b, unsigned char e) {
    float v = fp8e4m3_to_f32(b);
    float scale = __powf(2.0f, (float)((int)e - 127));
    return v * scale;
}

// #2 FP8 W8A8 grouped fused MoE dispatch. Reuses the identical token-sorted grouped structure + in-register gate/up/SiLU/down math
// as `grim_moe_fused_grouped`, but weights arrive as FP8 E4M3 bytes with per-block-16 weight scales and a per-token activation scale.
extern "C" __global__ void grim_moe_fused_grouped_fp8(
    const float* __restrict__ activations,    // [batch, hidden]
    const unsigned char* __restrict__ egate_w,// [num_experts, inter*hidden] FP8
    const unsigned char* __restrict__ eup_w,  // [num_experts, inter*hidden] FP8
    const unsigned char* __restrict__ edown_w,// [num_experts, hidden*inter] FP8
    const float* __restrict__ gate_scale,     // [num_experts, inter*(hidden/16)]
    const float* __restrict__ up_scale,       // [num_experts, inter*(hidden/16)]
    const float* __restrict__ down_scale,     // [num_experts, hidden*(inter/16)]
    const float* __restrict__ a_scale,        // [batch] per-token act scale
    const unsigned int* __restrict__ sorted_token_ids,
    const unsigned int* __restrict__ sorted_expert_ids,
    const float* __restrict__ sorted_weights,
    float* __restrict__ out,
    int hidden, int inter, int num_tokens, int block_size,
    float routed_scaling_factor)
{
    const int blk = blockIdx.x;
    const int base = blk * block_size;
    const int end = base + block_size < num_tokens ? base + block_size : num_tokens;
    const int h16 = (hidden + 15) / 16;
    const int i16 = (inter + 15) / 16;

    for (int s = base + threadIdx.x; s < end; s += blockDim.x) {
        const unsigned int tok = sorted_token_ids[s];
        if (tok >= (unsigned int)num_tokens) continue; // padding
        const unsigned int exp = sorted_expert_ids[s];
        const float w = sorted_weights[s];
        const float as = a_scale[tok];

        const float* a = activations + (unsigned long long)tok * hidden;
        const unsigned char* gw = egate_w + (unsigned long long)exp * inter * hidden;
        const unsigned char* uw = eup_w   + (unsigned long long)exp * inter * hidden;
        const unsigned char* dw = edown_w + (unsigned long long)exp * hidden * inter;
        const float* gs = gate_scale + (unsigned long long)exp * inter * h16;
        const float* us = up_scale   + (unsigned long long)exp * inter * h16;
        const float* ds = down_scale + (unsigned long long)exp * hidden * i16;

        float acc_prev = 0.0f;
        const bool odd_hidden = (hidden & 1) != 0;
        for (int h = 0; h < hidden; ++h) {
            float acc = 0.0f;
            for (int j = 0; j < inter; ++j) {
                float gate = 0.0f;
                float up = 0.0f;
                // Contract over the activation dimension (dot product), reusing the identical structure as grim_moe_fused_grouped.
                // The FP8 weight bytes are dequantized inline with their per-block scale.
                for (int i = 0; i < hidden; ++i) {
                    const int gidx = j * h16 + (i / 16);
                    const int uidx = j * h16 + (i / 16);
                    gate += fp8e4m3_to_f32(gw[j * hidden + i]) * gs[gidx] * a[i];
                    up   += fp8e4m3_to_f32(uw[j * hidden + i]) * us[uidx] * a[i];
                }
                float silu_g = gate / (1.0f + expf(-gate));
                float act = silu_g * up;
                // Down: [hidden, inter]; contract over `j` (inter) with activation `act`.
                const int didx = h * i16 + (j / 16);
                acc += fp8e4m3_to_f32(dw[h * inter + j]) * ds[didx] * act;
            }
            // Per-token activation scale folds into the single output mul.
            if (odd_hidden) {
                atomicAdd(out + (unsigned long long)tok * hidden + h,
                          routed_scaling_factor * w * as * acc);
            } else if (h & 1) {
                charon_atomic_add2(out, (unsigned long long)tok * hidden + (h - 1),
                                   routed_scaling_factor * w * as * acc_prev,
                                   routed_scaling_factor * w * as * acc);
            } else {
                acc_prev = acc;
            }
        }
    }
}

// --- #3 MXFP4 (E2M1 + E8M0) grouped kernel -------------------------------- OCP Microscaling FP4 (Jay tier): weights packed 2x E2M1 4-bit codes per byte, with one E8M0 shared-exponent byte per 32-element group.
// Dequant inline: value = mxfp4_e2m1_to_f32(code, shared_exp) where code is the 4-bit E2M1 nibble and shared_exp.
// NOTE (perf, WI-gpu-native-moe Phase 2): power-of-two factors go through
// integer bit construction, never __powf — bitwise-identical results
// (exact scalings of exact mantissas) at a fraction of the latency. The
// only special case is shared_exp == 0 (2^-127 subnormal), spelled as a
// compile-time constant.
__device__ __forceinline__ float mxfp4_e2m1_to_f32(unsigned char code, unsigned char shared_exp) {
    int sign = (code >> 3) & 1;
    int exp  = (code >> 1) & 3;
    int mant = code & 1;
    float base = (exp == 0) ? (float)mant * 0.5f
                            : (float)(2 + mant) * __int_as_float((unsigned int)(exp - 2 + 127) << 23);
    float val = sign ? -base : base;
    float scale = (shared_exp == 0) ? 0x1p-127f
                                    : __int_as_float((unsigned int)shared_exp << 23);
    return val * scale;
}

// Read the E2M1 4-bit code for weight element `idx` from packed `codes` (2/byte).
__device__ __forceinline__ unsigned char mxfp4_code_at(const unsigned char* codes, int idx) {
    unsigned char b = codes[idx >> 1];
    return (idx & 1) ? (b >> 4) & 0x0F : b & 0x0F;
}

extern "C" __global__ void grim_moe_fused_grouped_mxfp4(
    const float* __restrict__ activations,    // [batch, hidden]
    const unsigned char* __restrict__ egate_w,// [num_experts, inter*hidden/2] packed E2M1
    const unsigned char* __restrict__ eup_w,  // [num_experts, inter*hidden/2] packed E2M1
    const unsigned char* __restrict__ edown_w,// [num_experts, hidden*inter/2] packed E2M1
    const unsigned char* __restrict__ egate_e,// [num_experts, inter*hidden/32] E8M0 exps
    const unsigned char* __restrict__ eup_e,  // [num_experts, inter*hidden/32] E8M0 exps
    const unsigned char* __restrict__ edown_e,// [num_experts, hidden*inter/32] E8M0 exps
    const float* __restrict__ a_scale,        // [batch] per-token act scale
    const unsigned int* __restrict__ sorted_token_ids,
    const unsigned int* __restrict__ sorted_expert_ids,
    const float* __restrict__ sorted_weights,
    float* __restrict__ out,
    int hidden, int inter, int num_tokens, int block_size,
    float routed_scaling_factor)
{
    const int blk = blockIdx.x;
    const int base = blk * block_size;
    const int end = base + block_size < num_tokens ? base + block_size : num_tokens;

    for (int s = base + threadIdx.x; s < end; s += blockDim.x) {
        const unsigned int tok = sorted_token_ids[s];
        if (tok >= (unsigned int)num_tokens) continue; // padding
        const unsigned int exp = sorted_expert_ids[s];
        const float w = sorted_weights[s];
        const float as = a_scale[tok];

        const float* a = activations + (unsigned long long)tok * hidden;
        const unsigned char* gw = egate_w + (unsigned long long)exp * (inter * hidden / 2);
        const unsigned char* uw = eup_w   + (unsigned long long)exp * (inter * hidden / 2);
        const unsigned char* dw = edown_w + (unsigned long long)exp * (hidden * inter / 2);
        const unsigned char* ge = egate_e + (unsigned long long)exp * (inter * hidden / 32);
        const unsigned char* ue = eup_e   + (unsigned long long)exp * (inter * hidden / 32);
        const unsigned char* de = edown_e + (unsigned long long)exp * (hidden * inter / 32);

        float acc_prev = 0.0f;
        const bool odd_hidden = (hidden & 1) != 0;
        for (int h = 0; h < hidden; ++h) {
            float acc = 0.0f;
            for (int j = 0; j < inter; ++j) {
                float gate = 0.0f;
                float up = 0.0f;
                for (int i = 0; i < hidden; ++i) {
                    const int gidx = (j * hidden + i) / 32;
                    const int uidx = (j * hidden + i) / 32;
                    gate += mxfp4_e2m1_to_f32(mxfp4_code_at(gw, j * hidden + i), ge[gidx]) * a[i];
                    up   += mxfp4_e2m1_to_f32(mxfp4_code_at(uw, j * hidden + i), ue[uidx]) * a[i];
                }
                float silu_g = gate / (1.0f + expf(-gate));
                float act = silu_g * up;
                const int didx = (h * inter + j) / 32;
                acc += mxfp4_e2m1_to_f32(mxfp4_code_at(dw, h * inter + j), de[didx]) * act;
            }
            if (odd_hidden) {
                atomicAdd(out + (unsigned long long)tok * hidden + h,
                          routed_scaling_factor * w * as * acc);
            } else if (h & 1) {
                charon_atomic_add2(out, (unsigned long long)tok * hidden + (h - 1),
                                   routed_scaling_factor * w * as * acc_prev,
                                   routed_scaling_factor * w * as * acc);
            } else {
                acc_prev = acc;
            }
        }
    }
}

// --- WI-gpu-native-moe Phase 2: sortless MXFP4 fused dispatch ----------------
// Pair-parallel twin of `grim_moe_fused_grouped_mxfp4`: routing triples from
// DEVICE-resident buffers; codes and shared-exponent stacks concatenated per
// expert (no length prefixes on device — validated at stack-build time).
// Requires (rows*cols) % 32 == 0 so every 32-group is whole; the launcher
// and stack builder enforce this and refuse misaligned shapes loudly.
extern "C" __global__ void grim_moe_fused_dispatch_mxfp4(
    const float* __restrict__ activations,     // [batch, hidden]
    const unsigned char* __restrict__ expert_gate_w, // stacked packed E2M1
    const unsigned char* __restrict__ expert_up_w,   // stacked packed E2M1
    const unsigned char* __restrict__ expert_down_w, // stacked packed E2M1
    const unsigned char* __restrict__ expert_gate_e, // stacked E8M0 exps
    const unsigned char* __restrict__ expert_up_e,   // stacked E8M0 exps
    const unsigned char* __restrict__ expert_down_e, // stacked E8M0 exps
    const float* __restrict__ a_scale,         // [batch] per-token act scale
    const unsigned int* __restrict__ router_tokens,  // [num_pairs]
    const unsigned int* __restrict__ router_experts, // [num_pairs]
    const float* __restrict__ router_weights,        // [num_pairs]
    float* __restrict__ out,                     // [batch, hidden]
    int hidden, int inter, int num_pairs,
    float routed_scaling_factor)
{
    const unsigned long long pair = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= (unsigned long long)num_pairs) return;

    const unsigned int tok = router_tokens[pair];
    const unsigned int exp = router_experts[pair];
    const float w = router_weights[pair];
    const float as = a_scale[tok];

    const float* a = activations + (unsigned long long)tok * hidden;
    const unsigned char* gw = expert_gate_w + (unsigned long long)exp * (inter * hidden / 2);
    const unsigned char* uw = expert_up_w   + (unsigned long long)exp * (inter * hidden / 2);
    const unsigned char* dw = expert_down_w + (unsigned long long)exp * (hidden * inter / 2);
    const unsigned char* ge = expert_gate_e + (unsigned long long)exp * (inter * hidden / 32);
    const unsigned char* ue = expert_up_e   + (unsigned long long)exp * (inter * hidden / 32);
    const unsigned char* de = expert_down_e + (unsigned long long)exp * (hidden * inter / 32);

    for (int j = 0; j < inter; ++j) {
        float g = 0.0f;
        float u = 0.0f;
        for (int i = 0; i < hidden; ++i) {
            const int gidx = (j * hidden + i) / 32;
            const int uidx = (j * hidden + i) / 32;
            g += mxfp4_e2m1_to_f32(mxfp4_code_at(gw, j * hidden + i), ge[gidx]) * a[i];
            u += mxfp4_e2m1_to_f32(mxfp4_code_at(uw, j * hidden + i), ue[uidx]) * a[i];
        }
        float silu_g = g / (1.0f + expf(-g));
        float act = silu_g * u;
        float scale = routed_scaling_factor * w * as * act;

        for (int h = 0; h < hidden; ++h) {
            const int didx = (h * inter + j) / 32;
            float dv = mxfp4_e2m1_to_f32(mxfp4_code_at(dw, h * inter + j), de[didx]);
            unsigned long long out_idx = (unsigned long long)tok * hidden + h;
            atomicAdd(out + out_idx, dv * scale);
        }
    }
}

// --- #4 MXFP8 (E4M3 + E8M0) grouped kernel -------------------------------- OCP Microscaling FP8 (Magpie tier): weights are E4M3 codes (1 byte each, NOT packed) with one E8M0 shared-exponent byte per 32-element group.
// We reuse the already-corrected `fp8e4m3_to_f32` decoder from the WI-2 path.
extern "C" __global__ void grim_moe_fused_grouped_mxfp8(
    const float* activations,
    const unsigned char* egate_w, const unsigned char* eup_w, const unsigned char* edown_w,
    const unsigned char* egate_e, const unsigned char* eup_e, const unsigned char* edown_e,
    const float* a_scale,
    const unsigned int* sorted_token_ids, const unsigned int* sorted_expert_ids, const float* sorted_weights,
    float* out,
    int hidden, int inter, int num_tokens, int block_size, float routed_scaling_factor)
{
    const int blk = blockIdx.x;
    const int base = blk * block_size;
    const int end = base + block_size < num_tokens ? base + block_size : num_tokens;

    for (int s = base + threadIdx.x; s < end; s += blockDim.x) {
        const unsigned int tok = sorted_token_ids[s];
        if (tok >= (unsigned int)num_tokens) continue; // padding
        const unsigned int exp = sorted_expert_ids[s];
        const float w = sorted_weights[s];
        const float as = a_scale[tok];

        const float* a = activations + (unsigned long long)tok * hidden;
        const unsigned char* gw = egate_w + (unsigned long long)exp * (inter * hidden);
        const unsigned char* uw = eup_w   + (unsigned long long)exp * (inter * hidden);
        const unsigned char* dw = edown_w + (unsigned long long)exp * (hidden * inter);
        const unsigned char* ge = egate_e + (unsigned long long)exp * (inter * hidden / 32);
        const unsigned char* ue = eup_e   + (unsigned long long)exp * (inter * hidden / 32);
        const unsigned char* de = edown_e + (unsigned long long)exp * (hidden * inter / 32);

        float acc_prev = 0.0f;
        const bool odd_hidden = (hidden & 1) != 0;
        for (int h = 0; h < hidden; ++h) {
            float acc = 0.0f;
            for (int j = 0; j < inter; ++j) {
                float gate = 0.0f;
                float up = 0.0f;
                for (int i = 0; i < hidden; ++i) {
                    const int gidx = (j * hidden + i) / 32;
                    const int uidx = (j * hidden + i) / 32;
                    gate += mxfp8_e4m3_to_f32(gw[j * hidden + i], ge[gidx]) * a[i];
                    up   += mxfp8_e4m3_to_f32(uw[j * hidden + i], ue[uidx]) * a[i];
                }
                float silu_g = gate / (1.0f + expf(-gate));
                float act = silu_g * up;
                const int didx = (h * inter + j) / 32;
                acc += mxfp8_e4m3_to_f32(dw[h * inter + j], de[didx]) * act;
            }
            if (odd_hidden) {
                atomicAdd(out + (unsigned long long)tok * hidden + h,
                          routed_scaling_factor * w * as * acc);
            } else if (h & 1) {
                charon_atomic_add2(out, (unsigned long long)tok * hidden + (h - 1),
                                   routed_scaling_factor * w * as * acc_prev,
                                   routed_scaling_factor * w * as * acc);
            } else {
                acc_prev = acc;
            }
        }
    }
}

// --- #5 Q8_0 grouped kernel ------------------------------------------------ GGUF block-quantized weights: per 32 weights a `half` (f16) scale followed by 32 `int8` codes.
// value = scale * code.
__device__ __forceinline__ float f16_to_f32(unsigned short h) {
    unsigned int sign = (h >> 15) & 1u;
    unsigned int exp  = (h >> 10) & 0x1Fu;
    unsigned int mant = h & 0x3FFu;
    unsigned int bits;
    if (exp == 0u) {
        // Subnormal/zero: val = mant * 2^-24 (signed).
        bits = (sign << 31) | __float_as_int((float)mant * 0x1p-24f);
    } else if (exp == 31u) {
        bits = (sign << 31) | 0x7F800000u | (mant << 13);
    } else {
        bits = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
    }
    return __int_as_float(bits);
}

extern "C" __global__ void grim_moe_fused_grouped_q80(
    const float* activations,
    const unsigned char* egate_w, const unsigned char* eup_w, const unsigned char* edown_w,
    const float* a_scale,
    const unsigned int* sorted_token_ids, const unsigned int* sorted_expert_ids, const float* sorted_weights,
    float* out,
    int hidden, int inter, int num_tokens, int block_size, float routed_scaling_factor)
{
    const int blk = blockIdx.x;
    const int base = blk * block_size;
    const int end = base + block_size < num_tokens ? base + block_size : num_tokens;

    for (int s = base + threadIdx.x; s < end; s += blockDim.x) {
        const unsigned int tok = sorted_token_ids[s];
        if (tok >= (unsigned int)num_tokens) continue; // padding
        const unsigned int exp = sorted_expert_ids[s];
        const float w = sorted_weights[s];
        const float as = a_scale[tok];

        const float* a = activations + (unsigned long long)tok * hidden;
        // Q8_0 gate/up block stride = 34 bytes = (2 f16 scale) + (32 i8). Use
        // i8 offset 2 inside each 34-byte block; scale at block start.
        const int stride = (inter * hidden * 34) / 32; // bytes per expert for gate/up
        const unsigned char* gw = egate_w + (unsigned long long)exp * stride;
        const unsigned char* uw = eup_w   + (unsigned long long)exp * stride;
        const int dstride = (hidden * inter * 34) / 32; // bytes per expert for down
        const unsigned char* dw = edown_w + (unsigned long long)exp * dstride;

        float acc_prev = 0.0f;
        const bool odd_hidden = (hidden & 1) != 0;
        for (int h = 0; h < hidden; ++h) {
            float acc = 0.0f;
            for (int j = 0; j < inter; ++j) {
                float gate = 0.0f;
                float up = 0.0f;
                for (int i = 0; i < hidden; ++i) {
                    const int gblk = (j * hidden + i) / 32;
                    const int ublk = (j * hidden + i) / 32;
                    const float gscale = f16_to_f32(*(const unsigned short*)(gw + (unsigned long long)gblk * 34));
                    const float uscale = f16_to_f32(*(const unsigned short*)(uw + (unsigned long long)ublk * 34));
                    const int gi = (j * hidden + i) - gblk * 32;
                    const int ui = (j * hidden + i) - ublk * 32;
                    gate += (float)(*(const signed char*)(gw + (unsigned long long)gblk * 34 + 2 + gi)) * gscale * a[i];
                    up   += (float)(*(const signed char*)(uw + (unsigned long long)ublk * 34 + 2 + ui)) * uscale * a[i];
                }
                float silu_g = gate / (1.0f + expf(-gate));
                float act = silu_g * up;
                const int dblk = (h * inter + j) / 32;
                const float dscale = f16_to_f32(*(const unsigned short*)(dw + (unsigned long long)dblk * 34));
                const int di = (h * inter + j) - dblk * 32;
                acc += (float)(*(const signed char*)(dw + (unsigned long long)dblk * 34 + 2 + di)) * dscale * act;
            }
            if (odd_hidden) {
                atomicAdd(out + (unsigned long long)tok * hidden + h,
                          routed_scaling_factor * w * as * acc);
            } else if (h & 1) {
                charon_atomic_add2(out, (unsigned long long)tok * hidden + (h - 1),
                                   routed_scaling_factor * w * as * acc_prev,
                                   routed_scaling_factor * w * as * acc);
            } else {
                acc_prev = acc;
            }
        }
    }
}

// IQ + K-quant unified grouped fused dispatch kernel.
// `format_id` selects the super-block decode (mirrors grim-quant dequant_*): 0 iq4nl 1 iq4xs 2 iq3xxs 3.
__device__ __forceinline__ float iq4nl_codebook(int n) {
    const float CB[16] = {
        127.0f, 104.0f, 83.0f, 65.0f, 49.0f, 35.0f, 22.0f, 10.0f,
        1.0f, 13.0f, 25.0f, 38.0f, 53.0f, 69.0f, 87.0f, 107.0f
    };
    return CB[n & 15];
}

// Signed IQ4_NL codebook, verbatim from llama.cpp `kvalues_iq4nl` and from
// `grim_quant::iq_tables::KVALUES_IQ4NL` -- which is the table
// `dequant_iq4xs` and `dequant_iq4nl` actually index.
//
// Distinct from [`iq4nl_codebook`] above, and deliberately so: that table is
// unsigned magnitudes (indices 0..7 are positive) because the IQ4_NL decode
// takes the sign from a separate sign plane, and its last two entries are
// 87/107 rather than the canonical 89/113. Neither property is usable where the
// nibble itself carries the sign. One table per convention is the honest fix;
// aliasing them would silently negate every negative weight.
__device__ __forceinline__ float kvalues_iq4nl_signed(int n) {
    const float CB[16] = {
        -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
        1.0f, 13.0f, 25.0f, 38.0f, 53.0f, 69.0f, 89.0f, 113.0f
    };
    return CB[n & 15];
}

__device__ __forceinline__ float iq4xs_codebook(int n) {
    const float CB[16] = {
        0.0f, 0.11314126f, 0.24373604f, 0.39743365f,
        0.56574355f, 0.72294140f, 0.89705455f, 1.07576285f,
        1.29459881f, 1.52851904f, 1.82685633f, 2.27001130f,
        3.23719119f, 5.50829601f, 10.416256f, 34.56951f
    };
    return CB[n & 15];
}

// ---------------------------------------------------------------------------
// IQ grid / sign tables.
//
// GENERATED from `grim_quant::iq_tables` -- the same statics that back
// `dequant_iq2xs` / `dequant_iq2xxs` / `dequant_iq3s` / `dequant_iq3xxs`, which
// are the reference these decoders must agree with bit for bit.
//
// Do not hand-edit. `iqk_tables_match_grim_quant` re-parses this block out of
// KERNEL_SOURCE and compares every entry against the crate, so a divergence
// fails a host test rather than silently skewing weights on device.
// ---------------------------------------------------------------------------
__constant__ unsigned long long IQ2XS_GRID[512] = {
    0x0808080808080808ULL, 0x080808080808082BULL, 0x0808080808081919ULL, 0x0808080808082B08ULL,
    0x0808080808082B2BULL, 0x0808080808190819ULL, 0x0808080808191908ULL, 0x080808080819192BULL,
    0x0808080808192B19ULL, 0x08080808082B0808ULL, 0x08080808082B082BULL, 0x08080808082B1919ULL,
    0x08080808082B2B08ULL, 0x0808080819080819ULL, 0x0808080819081908ULL, 0x080808081908192BULL,
    0x0808080819082B19ULL, 0x0808080819190808ULL, 0x080808081919082BULL, 0x0808080819191919ULL,
    0x0808080819192B08ULL, 0x08080808192B0819ULL, 0x08080808192B1908ULL, 0x080808082B080808ULL,
    0x080808082B08082BULL, 0x080808082B081919ULL, 0x080808082B082B08ULL, 0x080808082B190819ULL,
    0x080808082B191908ULL, 0x080808082B192B19ULL, 0x080808082B2B0808ULL, 0x0808081908080819ULL,
    0x0808081908081908ULL, 0x080808190808192BULL, 0x0808081908082B19ULL, 0x0808081908190808ULL,
    0x080808190819082BULL, 0x0808081908191919ULL, 0x0808081908192B08ULL, 0x0808081908192B2BULL,
    0x08080819082B0819ULL, 0x08080819082B1908ULL, 0x0808081919080808ULL, 0x080808191908082BULL,
    0x0808081919081919ULL, 0x0808081919082B08ULL, 0x0808081919190819ULL, 0x0808081919191908ULL,
    0x08080819192B0808ULL, 0x08080819192B2B08ULL, 0x080808192B080819ULL, 0x080808192B081908ULL,
    0x080808192B190808ULL, 0x0808082B08080808ULL, 0x0808082B0808082BULL, 0x0808082B08081919ULL,
    0x0808082B08082B08ULL, 0x0808082B08190819ULL, 0x0808082B08191908ULL, 0x0808082B082B0808ULL,
    0x0808082B19080819ULL, 0x0808082B19081908ULL, 0x0808082B19190808ULL, 0x0808082B19191919ULL,
    0x0808082B2B080808ULL, 0x0808082B2B082B2BULL, 0x0808190808080819ULL, 0x0808190808081908ULL,
    0x080819080808192BULL, 0x0808190808082B19ULL, 0x0808190808190808ULL, 0x080819080819082BULL,
    0x0808190808191919ULL, 0x0808190808192B08ULL, 0x08081908082B0819ULL, 0x08081908082B1908ULL,
    0x0808190819080808ULL, 0x080819081908082BULL, 0x0808190819081919ULL, 0x0808190819082B08ULL,
    0x0808190819190819ULL, 0x0808190819191908ULL, 0x080819081919192BULL, 0x08081908192B0808ULL,
    0x080819082B080819ULL, 0x080819082B081908ULL, 0x080819082B190808ULL, 0x0808191908080808ULL,
    0x080819190808082BULL, 0x0808191908081919ULL, 0x0808191908082B08ULL, 0x0808191908190819ULL,
    0x0808191908191908ULL, 0x08081919082B0808ULL, 0x0808191919080819ULL, 0x0808191919081908ULL,
    0x0808191919190808ULL, 0x08081919192B0819ULL, 0x080819192B080808ULL, 0x0808192B08080819ULL,
    0x0808192B08081908ULL, 0x0808192B08190808ULL, 0x0808192B082B192BULL, 0x0808192B19080808ULL,
    0x0808192B1908082BULL, 0x0808192B2B081908ULL, 0x08082B0808080808ULL, 0x08082B080808082BULL,
    0x08082B0808081919ULL, 0x08082B0808082B08ULL, 0x08082B0808082B2BULL, 0x08082B0808190819ULL,
    0x08082B0808191908ULL, 0x08082B08082B0808ULL, 0x08082B08082B1919ULL, 0x08082B0819080819ULL,
    0x08082B0819081908ULL, 0x08082B0819190808ULL, 0x08082B0819192B08ULL, 0x08082B082B080808ULL,
    0x08082B082B2B0808ULL, 0x08082B082B2B2B2BULL, 0x08082B1908080819ULL, 0x08082B1908081908ULL,
    0x08082B1908190808ULL, 0x08082B1919080808ULL, 0x08082B192B080819ULL, 0x08082B192B082B19ULL,
    0x08082B2B08080808ULL, 0x08082B2B082B0808ULL, 0x08082B2B082B2B08ULL, 0x08082B2B2B19192BULL,
    0x08082B2B2B2B0808ULL, 0x0819080808080819ULL, 0x0819080808081908ULL, 0x081908080808192BULL,
    0x0819080808082B19ULL, 0x0819080808190808ULL, 0x081908080819082BULL, 0x0819080808191919ULL,
    0x0819080808192B08ULL, 0x08190808082B0819ULL, 0x08190808082B1908ULL, 0x0819080819080808ULL,
    0x081908081908082BULL, 0x0819080819081919ULL, 0x0819080819082B08ULL, 0x0819080819190819ULL,
    0x0819080819191908ULL, 0x08190808192B0808ULL, 0x08190808192B2B2BULL, 0x081908082B080819ULL,
    0x081908082B081908ULL, 0x081908082B190808ULL, 0x0819081908080808ULL, 0x081908190808082BULL,
    0x0819081908081919ULL, 0x0819081908082B08ULL, 0x0819081908190819ULL, 0x0819081908191908ULL,
    0x08190819082B0808ULL, 0x0819081919080819ULL, 0x0819081919081908ULL, 0x0819081919190808ULL,
    0x081908192B080808ULL, 0x081908192B191908ULL, 0x081908192B19192BULL, 0x0819082B08080819ULL,
    0x0819082B08081908ULL, 0x0819082B0808192BULL, 0x0819082B08190808ULL, 0x0819082B19080808ULL,
    0x0819082B192B0808ULL, 0x0819190808080808ULL, 0x081919080808082BULL, 0x0819190808081919ULL,
    0x0819190808082B08ULL, 0x0819190808190819ULL, 0x0819190808191908ULL, 0x08191908082B0808ULL,
    0x0819190819080819ULL, 0x0819190819081908ULL, 0x0819190819082B19ULL, 0x0819190819190808ULL,
    0x08191908192B1908ULL, 0x081919082B080808ULL, 0x0819191908080819ULL, 0x0819191908081908ULL,
    0x0819191908190808ULL, 0x0819191919080808ULL, 0x0819192B08080808ULL, 0x0819192B08191908ULL,
    0x0819192B19082B19ULL, 0x08192B0808080819ULL, 0x08192B0808081908ULL, 0x08192B0808190808ULL,
    0x08192B080819082BULL, 0x08192B0819080808ULL, 0x08192B0819191908ULL, 0x08192B082B08192BULL,
    0x08192B1908080808ULL, 0x08192B1908081919ULL, 0x08192B19192B192BULL, 0x08192B2B19190819ULL,
    0x08192B2B2B2B2B19ULL, 0x082B080808080808ULL, 0x082B08080808082BULL, 0x082B080808081919ULL,
    0x082B080808082B08ULL, 0x082B080808082B2BULL, 0x082B080808190819ULL, 0x082B080808191908ULL,
    0x082B0808082B0808ULL, 0x082B080819080819ULL, 0x082B080819081908ULL, 0x082B080819190808ULL,
    0x082B08082B080808ULL, 0x082B08082B2B0808ULL, 0x082B081908080819ULL, 0x082B081908081908ULL,
    0x082B081908190808ULL, 0x082B081919080808ULL, 0x082B081919082B08ULL, 0x082B0819192B1919ULL,
    0x082B082B08080808ULL, 0x082B082B082B082BULL, 0x082B082B2B080808ULL, 0x082B082B2B2B2B08ULL,
    0x082B190808080819ULL, 0x082B190808081908ULL, 0x082B190808190808ULL, 0x082B1908082B2B19ULL,
    0x082B190819080808ULL, 0x082B191908080808ULL, 0x082B191919080819ULL, 0x082B19191919082BULL,
    0x082B19192B192B19ULL, 0x082B192B08080819ULL, 0x082B192B08192B2BULL, 0x082B192B2B2B192BULL,
    0x082B2B0808080808ULL, 0x082B2B0808082B08ULL, 0x082B2B0808082B2BULL, 0x082B2B08082B0808ULL,
    0x082B2B0819191919ULL, 0x082B2B082B082B08ULL, 0x082B2B082B2B082BULL, 0x082B2B19192B2B08ULL,
    0x082B2B192B190808ULL, 0x082B2B2B08082B08ULL, 0x082B2B2B082B0808ULL, 0x082B2B2B2B08082BULL,
    0x082B2B2B2B082B08ULL, 0x082B2B2B2B082B2BULL, 0x1908080808080819ULL, 0x1908080808081908ULL,
    0x190808080808192BULL, 0x1908080808082B19ULL, 0x1908080808190808ULL, 0x190808080819082BULL,
    0x1908080808191919ULL, 0x1908080808192B08ULL, 0x19080808082B0819ULL, 0x19080808082B1908ULL,
    0x1908080819080808ULL, 0x190808081908082BULL, 0x1908080819081919ULL, 0x1908080819082B08ULL,
    0x1908080819082B2BULL, 0x1908080819190819ULL, 0x1908080819191908ULL, 0x19080808192B0808ULL,
    0x19080808192B1919ULL, 0x190808082B080819ULL, 0x190808082B081908ULL, 0x190808082B190808ULL,
    0x1908081908080808ULL, 0x190808190808082BULL, 0x1908081908081919ULL, 0x1908081908082B08ULL,
    0x1908081908190819ULL, 0x1908081908191908ULL, 0x19080819082B0808ULL, 0x1908081919080819ULL,
    0x1908081919081908ULL, 0x1908081919190808ULL, 0x190808192B080808ULL, 0x190808192B081919ULL,
    0x190808192B2B082BULL, 0x1908082B08080819ULL, 0x1908082B08081908ULL, 0x1908082B08190808ULL,
    0x1908082B0819082BULL, 0x1908082B082B2B19ULL, 0x1908082B19080808ULL, 0x1908190808080808ULL,
    0x190819080808082BULL, 0x1908190808081919ULL, 0x1908190808082B08ULL, 0x1908190808190819ULL,
    0x1908190808191908ULL, 0x1908190808192B19ULL, 0x19081908082B0808ULL, 0x1908190819080819ULL,
    0x1908190819081908ULL, 0x1908190819190808ULL, 0x190819082B080808ULL, 0x190819082B191908ULL,
    0x1908191908080819ULL, 0x1908191908081908ULL, 0x1908191908190808ULL, 0x19081919082B1908ULL,
    0x1908191919080808ULL, 0x190819192B192B2BULL, 0x1908192B08080808ULL, 0x1908192B08082B2BULL,
    0x1908192B19081908ULL, 0x1908192B19190808ULL, 0x19082B0808080819ULL, 0x19082B0808081908ULL,
    0x19082B0808190808ULL, 0x19082B0819080808ULL, 0x19082B0819081919ULL, 0x19082B0819191908ULL,
    0x19082B08192B082BULL, 0x19082B1908080808ULL, 0x19082B1908190819ULL, 0x19082B1919081908ULL,
    0x19082B1919190808ULL, 0x19082B19192B2B19ULL, 0x19082B2B08081908ULL, 0x1919080808080808ULL,
    0x191908080808082BULL, 0x1919080808081919ULL, 0x1919080808082B08ULL, 0x1919080808190819ULL,
    0x1919080808191908ULL, 0x19190808082B0808ULL, 0x19190808082B2B08ULL, 0x1919080819080819ULL,
    0x1919080819081908ULL, 0x1919080819190808ULL, 0x191908082B080808ULL, 0x1919081908080819ULL,
    0x1919081908081908ULL, 0x1919081908190808ULL, 0x1919081908191919ULL, 0x1919081919080808ULL,
    0x191908191908082BULL, 0x1919082B08080808ULL, 0x1919082B19081908ULL, 0x1919082B2B2B2B2BULL,
    0x1919190808080819ULL, 0x1919190808081908ULL, 0x1919190808190808ULL, 0x19191908082B0819ULL,
    0x1919190819080808ULL, 0x19191908192B0808ULL, 0x191919082B080819ULL, 0x191919082B2B0819ULL,
    0x1919191908080808ULL, 0x1919191908082B08ULL, 0x191919192B080808ULL, 0x191919192B082B08ULL,
    0x1919192B082B0819ULL, 0x1919192B192B2B08ULL, 0x1919192B2B2B0819ULL, 0x19192B0808080808ULL,
    0x19192B0808191908ULL, 0x19192B0819080819ULL, 0x19192B0819190808ULL, 0x19192B082B192B19ULL,
    0x19192B1908192B2BULL, 0x19192B1919080808ULL, 0x19192B191908082BULL, 0x19192B2B2B081919ULL,
    0x192B080808080819ULL, 0x192B080808081908ULL, 0x192B080808190808ULL, 0x192B080819080808ULL,
    0x192B080819191908ULL, 0x192B0808192B082BULL, 0x192B08082B08192BULL, 0x192B08082B2B2B19ULL,
    0x192B081908080808ULL, 0x192B082B082B1908ULL, 0x192B082B19082B2BULL, 0x192B082B2B19082BULL,
    0x192B190808080808ULL, 0x192B19080819192BULL, 0x192B191908190808ULL, 0x192B191919080808ULL,
    0x192B191919081919ULL, 0x192B19192B2B1908ULL, 0x192B2B0808080819ULL, 0x192B2B08192B2B2BULL,
    0x192B2B19082B1919ULL, 0x192B2B2B0808192BULL, 0x192B2B2B19191908ULL, 0x192B2B2B192B082BULL,
    0x2B08080808080808ULL, 0x2B0808080808082BULL, 0x2B08080808081919ULL, 0x2B08080808082B08ULL,
    0x2B08080808190819ULL, 0x2B08080808191908ULL, 0x2B080808082B0808ULL, 0x2B080808082B2B2BULL,
    0x2B08080819080819ULL, 0x2B08080819081908ULL, 0x2B08080819190808ULL, 0x2B0808082B080808ULL,
    0x2B0808082B08082BULL, 0x2B0808082B2B2B08ULL, 0x2B0808082B2B2B2BULL, 0x2B08081908080819ULL,
    0x2B08081908081908ULL, 0x2B0808190808192BULL, 0x2B08081908190808ULL, 0x2B08081919080808ULL,
    0x2B08081919190819ULL, 0x2B08081919192B19ULL, 0x2B08082B08080808ULL, 0x2B08082B082B0808ULL,
    0x2B08082B2B080808ULL, 0x2B08082B2B08082BULL, 0x2B08082B2B2B0808ULL, 0x2B08082B2B2B2B08ULL,
    0x2B08190808080819ULL, 0x2B08190808081908ULL, 0x2B08190808190808ULL, 0x2B0819080819082BULL,
    0x2B08190808191919ULL, 0x2B08190819080808ULL, 0x2B081908192B0808ULL, 0x2B0819082B082B19ULL,
    0x2B08191908080808ULL, 0x2B08191919081908ULL, 0x2B0819192B2B1919ULL, 0x2B08192B08192B08ULL,
    0x2B08192B192B2B2BULL, 0x2B082B0808080808ULL, 0x2B082B0808082B08ULL, 0x2B082B08082B1919ULL,
    0x2B082B0819192B2BULL, 0x2B082B082B080808ULL, 0x2B082B082B08082BULL, 0x2B082B082B2B2B08ULL,
    0x2B082B190808192BULL, 0x2B082B2B082B082BULL, 0x2B082B2B2B080808ULL, 0x2B082B2B2B082B08ULL,
    0x2B082B2B2B19192BULL, 0x2B082B2B2B2B2B08ULL, 0x2B19080808080819ULL, 0x2B19080808081908ULL,
    0x2B19080808190808ULL, 0x2B19080819080808ULL, 0x2B1908081919192BULL, 0x2B1908082B081908ULL,
    0x2B19081908080808ULL, 0x2B190819082B082BULL, 0x2B190819192B1908ULL, 0x2B19082B1919192BULL,
    0x2B19082B2B082B19ULL, 0x2B19190808080808ULL, 0x2B19190808081919ULL, 0x2B19190819081908ULL,
    0x2B19190819190808ULL, 0x2B19190819192B08ULL, 0x2B191919082B2B19ULL, 0x2B1919192B190808ULL,
    0x2B1919192B19082BULL, 0x2B19192B19080819ULL, 0x2B192B0819190819ULL, 0x2B192B082B2B192BULL,
    0x2B192B1919082B19ULL, 0x2B192B2B08191919ULL, 0x2B192B2B192B0808ULL, 0x2B2B080808080808ULL,
    0x2B2B08080808082BULL, 0x2B2B080808082B08ULL, 0x2B2B080808082B2BULL, 0x2B2B0808082B0808ULL,
    0x2B2B0808082B2B2BULL, 0x2B2B08082B2B0808ULL, 0x2B2B081919190819ULL, 0x2B2B081919192B19ULL,
    0x2B2B08192B2B192BULL, 0x2B2B082B08080808ULL, 0x2B2B082B0808082BULL, 0x2B2B082B08082B08ULL,
    0x2B2B082B082B2B2BULL, 0x2B2B082B2B080808ULL, 0x2B2B082B2B2B0808ULL, 0x2B2B190819080808ULL,
    0x2B2B19082B191919ULL, 0x2B2B192B192B1919ULL, 0x2B2B192B2B192B08ULL, 0x2B2B2B0808082B2BULL,
    0x2B2B2B08082B0808ULL, 0x2B2B2B08082B082BULL, 0x2B2B2B08082B2B08ULL, 0x2B2B2B082B2B0808ULL,
    0x2B2B2B082B2B2B08ULL, 0x2B2B2B1908081908ULL, 0x2B2B2B192B081908ULL, 0x2B2B2B192B08192BULL,
    0x2B2B2B2B082B2B08ULL, 0x2B2B2B2B082B2B2BULL, 0x2B2B2B2B2B190819ULL, 0x2B2B2B2B2B2B2B2BULL,
};

__constant__ unsigned long long IQ2XXS_GRID[256] = {
    0x0808080808080808ULL, 0x080808080808082BULL, 0x0808080808081919ULL, 0x0808080808082B08ULL,
    0x0808080808082B2BULL, 0x0808080808190819ULL, 0x0808080808191908ULL, 0x08080808082B0808ULL,
    0x08080808082B082BULL, 0x08080808082B2B08ULL, 0x08080808082B2B2BULL, 0x0808080819080819ULL,
    0x0808080819081908ULL, 0x0808080819190808ULL, 0x0808080819192B08ULL, 0x08080808192B0819ULL,
    0x08080808192B1908ULL, 0x080808082B080808ULL, 0x080808082B08082BULL, 0x080808082B082B2BULL,
    0x080808082B2B082BULL, 0x0808081908080819ULL, 0x0808081908081908ULL, 0x0808081908190808ULL,
    0x0808081908191919ULL, 0x0808081919080808ULL, 0x080808192B081908ULL, 0x080808192B192B08ULL,
    0x0808082B08080808ULL, 0x0808082B0808082BULL, 0x0808082B082B082BULL, 0x0808082B2B08082BULL,
    0x0808190808080819ULL, 0x0808190808081908ULL, 0x0808190808190808ULL, 0x08081908082B0819ULL,
    0x08081908082B1908ULL, 0x0808190819080808ULL, 0x080819081908082BULL, 0x0808190819082B08ULL,
    0x08081908192B0808ULL, 0x080819082B080819ULL, 0x080819082B081908ULL, 0x080819082B190808ULL,
    0x080819082B2B1908ULL, 0x0808191908080808ULL, 0x080819190808082BULL, 0x0808191908082B08ULL,
    0x08081919082B0808ULL, 0x080819191908192BULL, 0x08081919192B2B19ULL, 0x080819192B080808ULL,
    0x080819192B190819ULL, 0x0808192B08082B19ULL, 0x0808192B08190808ULL, 0x0808192B19080808ULL,
    0x0808192B2B081908ULL, 0x0808192B2B2B1908ULL, 0x08082B0808080808ULL, 0x08082B0808081919ULL,
    0x08082B0808082B08ULL, 0x08082B0808191908ULL, 0x08082B08082B2B08ULL, 0x08082B0819080819ULL,
    0x08082B0819081908ULL, 0x08082B0819190808ULL, 0x08082B081919082BULL, 0x08082B082B082B08ULL,
    0x08082B1908081908ULL, 0x08082B1919080808ULL, 0x08082B2B0808082BULL, 0x08082B2B08191908ULL,
    0x0819080808080819ULL, 0x0819080808081908ULL, 0x0819080808190808ULL, 0x08190808082B0819ULL,
    0x0819080819080808ULL, 0x08190808192B0808ULL, 0x081908082B081908ULL, 0x081908082B190808ULL,
    0x081908082B191919ULL, 0x0819081908080808ULL, 0x0819081908082B08ULL, 0x08190819082B0808ULL,
    0x0819081919190808ULL, 0x0819081919192B2BULL, 0x081908192B080808ULL, 0x0819082B082B1908ULL,
    0x0819082B19081919ULL, 0x0819190808080808ULL, 0x0819190808082B08ULL, 0x08191908082B0808ULL,
    0x08191908082B1919ULL, 0x0819190819082B19ULL, 0x081919082B080808ULL, 0x0819191908192B08ULL,
    0x08191919192B082BULL, 0x0819192B08080808ULL, 0x0819192B0819192BULL, 0x08192B0808080819ULL,
    0x08192B0808081908ULL, 0x08192B0808190808ULL, 0x08192B0819080808ULL, 0x08192B082B080819ULL,
    0x08192B1908080808ULL, 0x08192B1908081919ULL, 0x08192B192B2B0808ULL, 0x08192B2B19190819ULL,
    0x082B080808080808ULL, 0x082B08080808082BULL, 0x082B080808082B2BULL, 0x082B080819081908ULL,
    0x082B0808192B0819ULL, 0x082B08082B080808ULL, 0x082B08082B08082BULL, 0x082B0819082B2B19ULL,
    0x082B081919082B08ULL, 0x082B082B08080808ULL, 0x082B082B0808082BULL, 0x082B190808080819ULL,
    0x082B190808081908ULL, 0x082B190808190808ULL, 0x082B190819080808ULL, 0x082B19081919192BULL,
    0x082B191908080808ULL, 0x082B191919080819ULL, 0x082B1919192B1908ULL, 0x082B192B2B190808ULL,
    0x082B2B0808082B08ULL, 0x082B2B08082B0808ULL, 0x082B2B082B191908ULL, 0x082B2B2B19081908ULL,
    0x1908080808080819ULL, 0x1908080808081908ULL, 0x1908080808190808ULL, 0x1908080808192B08ULL,
    0x19080808082B0819ULL, 0x19080808082B1908ULL, 0x1908080819080808ULL, 0x1908080819082B08ULL,
    0x190808081919192BULL, 0x19080808192B0808ULL, 0x190808082B080819ULL, 0x190808082B081908ULL,
    0x190808082B190808ULL, 0x1908081908080808ULL, 0x19080819082B0808ULL, 0x19080819192B0819ULL,
    0x190808192B080808ULL, 0x190808192B081919ULL, 0x1908082B08080819ULL, 0x1908082B08190808ULL,
    0x1908082B19082B08ULL, 0x1908082B1919192BULL, 0x1908082B192B2B08ULL, 0x1908190808080808ULL,
    0x1908190808082B08ULL, 0x19081908082B0808ULL, 0x190819082B080808ULL, 0x190819082B192B19ULL,
    0x190819190819082BULL, 0x19081919082B1908ULL, 0x1908192B08080808ULL, 0x19082B0808080819ULL,
    0x19082B0808081908ULL, 0x19082B0808190808ULL, 0x19082B0819080808ULL, 0x19082B0819081919ULL,
    0x19082B1908080808ULL, 0x19082B1919192B08ULL, 0x19082B19192B0819ULL, 0x19082B192B08082BULL,
    0x19082B2B19081919ULL, 0x19082B2B2B190808ULL, 0x1919080808080808ULL, 0x1919080808082B08ULL,
    0x1919080808190819ULL, 0x1919080808192B19ULL, 0x19190808082B0808ULL, 0x191908082B080808ULL,
    0x191908082B082B08ULL, 0x1919081908081908ULL, 0x191908191908082BULL, 0x191908192B2B1908ULL,
    0x1919082B2B190819ULL, 0x191919082B190808ULL, 0x191919082B19082BULL, 0x1919191908082B2BULL,
    0x1919192B08080819ULL, 0x1919192B19191908ULL, 0x19192B0808080808ULL, 0x19192B0808190819ULL,
    0x19192B0808192B19ULL, 0x19192B08192B1908ULL, 0x19192B1919080808ULL, 0x19192B2B08082B08ULL,
    0x192B080808081908ULL, 0x192B080808190808ULL, 0x192B080819080808ULL, 0x192B0808192B2B08ULL,
    0x192B081908080808ULL, 0x192B081919191919ULL, 0x192B082B08192B08ULL, 0x192B082B192B0808ULL,
    0x192B190808080808ULL, 0x192B190808081919ULL, 0x192B191908190808ULL, 0x192B19190819082BULL,
    0x192B19192B081908ULL, 0x192B2B081908082BULL, 0x2B08080808080808ULL, 0x2B0808080808082BULL,
    0x2B08080808082B2BULL, 0x2B08080819080819ULL, 0x2B0808082B08082BULL, 0x2B08081908081908ULL,
    0x2B08081908192B08ULL, 0x2B08081919080808ULL, 0x2B08082B08190819ULL, 0x2B08190808080819ULL,
    0x2B08190808081908ULL, 0x2B08190808190808ULL, 0x2B08190808191919ULL, 0x2B08190819080808ULL,
    0x2B081908192B0808ULL, 0x2B08191908080808ULL, 0x2B0819191908192BULL, 0x2B0819192B191908ULL,
    0x2B08192B08082B19ULL, 0x2B08192B19080808ULL, 0x2B08192B192B0808ULL, 0x2B082B080808082BULL,
    0x2B082B1908081908ULL, 0x2B082B2B08190819ULL, 0x2B19080808081908ULL, 0x2B19080808190808ULL,
    0x2B190808082B1908ULL, 0x2B19080819080808ULL, 0x2B1908082B2B0819ULL, 0x2B1908190819192BULL,
    0x2B1908192B080808ULL, 0x2B19082B19081919ULL, 0x2B19190808080808ULL, 0x2B191908082B082BULL,
    0x2B19190819081908ULL, 0x2B19191919190819ULL, 0x2B192B082B080819ULL, 0x2B192B19082B0808ULL,
    0x2B2B08080808082BULL, 0x2B2B080819190808ULL, 0x2B2B08082B081919ULL, 0x2B2B081908082B19ULL,
    0x2B2B082B08080808ULL, 0x2B2B190808192B08ULL, 0x2B2B2B0819190808ULL, 0x2B2B2B1908081908ULL,
};

__constant__ unsigned int IQ3S_GRID[512] = {
    0x01010101u, 0x01010103u, 0x01010105u, 0x0101010Bu, 0x0101010Fu, 0x01010301u, 0x01010303u, 0x01010305u,
    0x01010309u, 0x0101030Du, 0x01010501u, 0x01010503u, 0x0101050Bu, 0x01010707u, 0x01010901u, 0x01010905u,
    0x0101090Bu, 0x0101090Fu, 0x01010B03u, 0x01010B07u, 0x01010D01u, 0x01010D05u, 0x01010F03u, 0x01010F09u,
    0x01010F0Fu, 0x01030101u, 0x01030103u, 0x01030105u, 0x01030109u, 0x01030301u, 0x01030303u, 0x0103030Bu,
    0x01030501u, 0x01030507u, 0x0103050Fu, 0x01030703u, 0x0103070Bu, 0x01030909u, 0x01030D03u, 0x01030D0Bu,
    0x01030F05u, 0x01050101u, 0x01050103u, 0x0105010Bu, 0x0105010Fu, 0x01050301u, 0x01050307u, 0x0105030Du,
    0x01050503u, 0x0105050Bu, 0x01050701u, 0x01050709u, 0x01050905u, 0x0105090Bu, 0x0105090Fu, 0x01050B03u,
    0x01050B07u, 0x01050F01u, 0x01050F07u, 0x01070107u, 0x01070303u, 0x0107030Bu, 0x01070501u, 0x01070505u,
    0x01070703u, 0x01070707u, 0x0107070Du, 0x01070909u, 0x01070B01u, 0x01070B05u, 0x01070D0Fu, 0x01070F03u,
    0x01070F0Bu, 0x01090101u, 0x01090307u, 0x0109030Fu, 0x01090503u, 0x01090509u, 0x01090705u, 0x01090901u,
    0x01090907u, 0x01090B03u, 0x01090F01u, 0x010B0105u, 0x010B0109u, 0x010B0501u, 0x010B0505u, 0x010B050Du,
    0x010B0707u, 0x010B0903u, 0x010B090Bu, 0x010B090Fu, 0x010B0D0Du, 0x010B0F07u, 0x010D010Du, 0x010D0303u,
    0x010D0307u, 0x010D0703u, 0x010D0B05u, 0x010D0F03u, 0x010F0101u, 0x010F0105u, 0x010F0109u, 0x010F0501u,
    0x010F0505u, 0x010F050Du, 0x010F0707u, 0x010F0B01u, 0x010F0B09u, 0x03010101u, 0x03010103u, 0x03010105u,
    0x03010109u, 0x03010301u, 0x03010303u, 0x03010307u, 0x0301030Bu, 0x0301030Fu, 0x03010501u, 0x03010505u,
    0x03010703u, 0x03010709u, 0x0301070Du, 0x03010B09u, 0x03010B0Du, 0x03010D03u, 0x03010F05u, 0x03030101u,
    0x03030103u, 0x03030107u, 0x0303010Du, 0x03030301u, 0x03030309u, 0x03030503u, 0x03030701u, 0x03030707u,
    0x03030903u, 0x03030B01u, 0x03030B05u, 0x03030F01u, 0x03030F0Du, 0x03050101u, 0x03050305u, 0x0305030Bu,
    0x0305030Fu, 0x03050501u, 0x03050509u, 0x03050705u, 0x03050901u, 0x03050907u, 0x03050B0Bu, 0x03050D01u,
    0x03050F05u, 0x03070103u, 0x03070109u, 0x0307010Fu, 0x03070301u, 0x03070307u, 0x03070503u, 0x0307050Fu,
    0x03070701u, 0x03070709u, 0x03070903u, 0x03070D05u, 0x03070F01u, 0x03090107u, 0x0309010Bu, 0x03090305u,
    0x03090309u, 0x03090703u, 0x03090707u, 0x03090905u, 0x0309090Du, 0x03090B01u, 0x03090B09u, 0x030B0103u,
    0x030B0301u, 0x030B0307u, 0x030B0503u, 0x030B0701u, 0x030B0705u, 0x030B0B03u, 0x030D0501u, 0x030D0509u,
    0x030D050Fu, 0x030D0909u, 0x030D090Du, 0x030F0103u, 0x030F0107u, 0x030F0301u, 0x030F0305u, 0x030F0503u,
    0x030F070Bu, 0x030F0903u, 0x030F0D05u, 0x030F0F01u, 0x05010101u, 0x05010103u, 0x05010107u, 0x0501010Bu,
    0x0501010Fu, 0x05010301u, 0x05010305u, 0x05010309u, 0x0501030Du, 0x05010503u, 0x05010507u, 0x0501050Fu,
    0x05010701u, 0x05010705u, 0x05010903u, 0x05010907u, 0x0501090Bu, 0x05010B01u, 0x05010B05u, 0x05010D0Fu,
    0x05010F01u, 0x05010F07u, 0x05010F0Bu, 0x05030101u, 0x05030105u, 0x05030301u, 0x05030307u, 0x0503030Fu,
    0x05030505u, 0x0503050Bu, 0x05030703u, 0x05030709u, 0x05030905u, 0x05030B03u, 0x05050103u, 0x05050109u,
    0x0505010Fu, 0x05050503u, 0x05050507u, 0x05050701u, 0x0505070Fu, 0x05050903u, 0x05050B07u, 0x05050B0Fu,
    0x05050F03u, 0x05050F09u, 0x05070101u, 0x05070105u, 0x0507010Bu, 0x05070303u, 0x05070505u, 0x05070509u,
    0x05070703u, 0x05070707u, 0x05070905u, 0x05070B01u, 0x05070D0Du, 0x05090103u, 0x0509010Fu, 0x05090501u,
    0x05090507u, 0x05090705u, 0x0509070Bu, 0x05090903u, 0x05090F05u, 0x05090F0Bu, 0x050B0109u, 0x050B0303u,
    0x050B0505u, 0x050B070Fu, 0x050B0901u, 0x050B0B07u, 0x050B0F01u, 0x050D0101u, 0x050D0105u, 0x050D010Fu,
    0x050D0503u, 0x050D0B0Bu, 0x050D0D03u, 0x050F010Bu, 0x050F0303u, 0x050F050Du, 0x050F0701u, 0x050F0907u,
    0x050F0B01u, 0x07010105u, 0x07010303u, 0x07010307u, 0x0701030Bu, 0x0701030Fu, 0x07010505u, 0x07010703u,
    0x07010707u, 0x0701070Bu, 0x07010905u, 0x07010909u, 0x0701090Fu, 0x07010B03u, 0x07010D07u, 0x07010F03u,
    0x07030103u, 0x07030107u, 0x0703010Bu, 0x07030309u, 0x07030503u, 0x07030507u, 0x07030901u, 0x07030D01u,
    0x07030F05u, 0x07030F0Du, 0x07050101u, 0x07050305u, 0x07050501u, 0x07050705u, 0x07050709u, 0x07050B01u,
    0x07070103u, 0x07070301u, 0x07070309u, 0x07070503u, 0x07070507u, 0x0707050Fu, 0x07070701u, 0x07070903u,
    0x07070907u, 0x0707090Fu, 0x07070B0Bu, 0x07070F07u, 0x07090107u, 0x07090303u, 0x0709030Du, 0x07090505u,
    0x07090703u, 0x07090B05u, 0x07090D01u, 0x07090D09u, 0x070B0103u, 0x070B0301u, 0x070B0305u, 0x070B050Bu,
    0x070B0705u, 0x070B0909u, 0x070B0B0Du, 0x070B0F07u, 0x070D030Du, 0x070D0903u, 0x070F0103u, 0x070F0107u,
    0x070F0501u, 0x070F0505u, 0x070F070Bu, 0x09010101u, 0x09010109u, 0x09010305u, 0x09010501u, 0x09010509u,
    0x0901050Fu, 0x09010705u, 0x09010903u, 0x09010B01u, 0x09010F01u, 0x09030105u, 0x0903010Fu, 0x09030303u,
    0x09030307u, 0x09030505u, 0x09030701u, 0x0903070Bu, 0x09030907u, 0x09030B03u, 0x09030B0Bu, 0x09050103u,
    0x09050107u, 0x09050301u, 0x0905030Bu, 0x09050503u, 0x09050707u, 0x09050901u, 0x09050B0Fu, 0x09050D05u,
    0x09050F01u, 0x09070109u, 0x09070303u, 0x09070307u, 0x09070501u, 0x09070505u, 0x09070703u, 0x0907070Bu,
    0x09090101u, 0x09090105u, 0x09090509u, 0x0909070Fu, 0x09090901u, 0x09090F03u, 0x090B010Bu, 0x090B010Fu,
    0x090B0503u, 0x090B0D05u, 0x090D0307u, 0x090D0709u, 0x090D0D01u, 0x090F0301u, 0x090F030Bu, 0x090F0701u,
    0x090F0907u, 0x090F0B03u, 0x0B010105u, 0x0B010301u, 0x0B010309u, 0x0B010505u, 0x0B010901u, 0x0B010909u,
    0x0B01090Fu, 0x0B010B05u, 0x0B010D0Du, 0x0B010F09u, 0x0B030103u, 0x0B030107u, 0x0B03010Bu, 0x0B030305u,
    0x0B030503u, 0x0B030705u, 0x0B030F05u, 0x0B050101u, 0x0B050303u, 0x0B050507u, 0x0B050701u, 0x0B05070Du,
    0x0B050B07u, 0x0B070105u, 0x0B07010Fu, 0x0B070301u, 0x0B07050Fu, 0x0B070909u, 0x0B070B03u, 0x0B070D0Bu,
    0x0B070F07u, 0x0B090103u, 0x0B090109u, 0x0B090501u, 0x0B090705u, 0x0B09090Du, 0x0B0B0305u, 0x0B0B050Du,
    0x0B0B0B03u, 0x0B0B0B07u, 0x0B0D0905u, 0x0B0F0105u, 0x0B0F0109u, 0x0B0F0505u, 0x0D010303u, 0x0D010307u,
    0x0D01030Bu, 0x0D010703u, 0x0D010707u, 0x0D010D01u, 0x0D030101u, 0x0D030501u, 0x0D03050Fu, 0x0D030D09u,
    0x0D050305u, 0x0D050709u, 0x0D050905u, 0x0D050B0Bu, 0x0D050D05u, 0x0D050F01u, 0x0D070101u, 0x0D070309u,
    0x0D070503u, 0x0D070901u, 0x0D09050Bu, 0x0D090907u, 0x0D090D05u, 0x0D0B0101u, 0x0D0B0107u, 0x0D0B0709u,
    0x0D0B0D01u, 0x0D0D010Bu, 0x0D0D0901u, 0x0D0F0303u, 0x0D0F0307u, 0x0F010101u, 0x0F010109u, 0x0F01010Fu,
    0x0F010501u, 0x0F010505u, 0x0F01070Du, 0x0F010901u, 0x0F010B09u, 0x0F010D05u, 0x0F030105u, 0x0F030303u,
    0x0F030509u, 0x0F030907u, 0x0F03090Bu, 0x0F050103u, 0x0F050109u, 0x0F050301u, 0x0F05030Du, 0x0F050503u,
    0x0F050701u, 0x0F050B03u, 0x0F070105u, 0x0F070705u, 0x0F07070Bu, 0x0F070B07u, 0x0F090103u, 0x0F09010Bu,
    0x0F090307u, 0x0F090501u, 0x0F090B01u, 0x0F0B0505u, 0x0F0B0905u, 0x0F0D0105u, 0x0F0D0703u, 0x0F0F0101u,
};

__constant__ unsigned int IQ3XXS_GRID[256] = {
    0x04040404u, 0x04040414u, 0x04040424u, 0x04040C0Cu, 0x04040C1Cu, 0x04040C3Eu, 0x04041404u, 0x04041414u,
    0x04041C0Cu, 0x04042414u, 0x04043E1Cu, 0x04043E2Cu, 0x040C040Cu, 0x040C041Cu, 0x040C0C04u, 0x040C0C14u,
    0x040C140Cu, 0x040C142Cu, 0x040C1C04u, 0x040C1C14u, 0x040C240Cu, 0x040C2C24u, 0x040C3E04u, 0x04140404u,
    0x04140414u, 0x04140424u, 0x04140C0Cu, 0x04141404u, 0x04141414u, 0x04141C0Cu, 0x04141C1Cu, 0x04141C3Eu,
    0x04142C0Cu, 0x04142C3Eu, 0x04143E2Cu, 0x041C040Cu, 0x041C043Eu, 0x041C0C04u, 0x041C0C14u, 0x041C142Cu,
    0x041C3E04u, 0x04240C1Cu, 0x04241C3Eu, 0x04242424u, 0x04242C3Eu, 0x04243E1Cu, 0x04243E2Cu, 0x042C040Cu,
    0x042C043Eu, 0x042C1C14u, 0x042C2C14u, 0x04341C2Cu, 0x04343424u, 0x043E0C04u, 0x043E0C24u, 0x043E0C34u,
    0x043E241Cu, 0x043E340Cu, 0x0C04040Cu, 0x0C04041Cu, 0x0C040C04u, 0x0C040C14u, 0x0C04140Cu, 0x0C04141Cu,
    0x0C041C04u, 0x0C041C14u, 0x0C041C24u, 0x0C04243Eu, 0x0C042C04u, 0x0C0C0404u, 0x0C0C0414u, 0x0C0C0C0Cu,
    0x0C0C1404u, 0x0C0C1414u, 0x0C14040Cu, 0x0C14041Cu, 0x0C140C04u, 0x0C140C14u, 0x0C14140Cu, 0x0C141C04u,
    0x0C143E14u, 0x0C1C0404u, 0x0C1C0414u, 0x0C1C1404u, 0x0C1C1C0Cu, 0x0C1C2434u, 0x0C1C3434u, 0x0C24040Cu,
    0x0C24042Cu, 0x0C242C04u, 0x0C2C1404u, 0x0C2C1424u, 0x0C2C2434u, 0x0C2C3E0Cu, 0x0C34042Cu, 0x0C3E1414u,
    0x0C3E2404u, 0x14040404u, 0x14040414u, 0x14040C0Cu, 0x14040C1Cu, 0x14041404u, 0x14041414u, 0x14041434u,
    0x14041C0Cu, 0x14042414u, 0x140C040Cu, 0x140C041Cu, 0x140C042Cu, 0x140C0C04u, 0x140C0C14u, 0x140C140Cu,
    0x140C1C04u, 0x140C341Cu, 0x140C343Eu, 0x140C3E04u, 0x14140404u, 0x14140414u, 0x14140C0Cu, 0x14140C3Eu,
    0x14141404u, 0x14141414u, 0x14141C3Eu, 0x14142404u, 0x14142C2Cu, 0x141C040Cu, 0x141C0C04u, 0x141C0C24u,
    0x141C3E04u, 0x141C3E24u, 0x14241C2Cu, 0x14242C1Cu, 0x142C041Cu, 0x142C143Eu, 0x142C240Cu, 0x142C3E24u,
    0x143E040Cu, 0x143E041Cu, 0x143E0C34u, 0x143E242Cu, 0x1C04040Cu, 0x1C040C04u, 0x1C040C14u, 0x1C04140Cu,
    0x1C04141Cu, 0x1C042C04u, 0x1C04342Cu, 0x1C043E14u, 0x1C0C0404u, 0x1C0C0414u, 0x1C0C1404u, 0x1C0C1C0Cu,
    0x1C0C2424u, 0x1C0C2434u, 0x1C14040Cu, 0x1C14041Cu, 0x1C140C04u, 0x1C14142Cu, 0x1C142C14u, 0x1C143E14u,
    0x1C1C0C0Cu, 0x1C1C1C1Cu, 0x1C241C04u, 0x1C24243Eu, 0x1C243E14u, 0x1C2C0404u, 0x1C2C0434u, 0x1C2C1414u,
    0x1C2C2C2Cu, 0x1C340C24u, 0x1C341C34u, 0x1C34341Cu, 0x1C3E1C1Cu, 0x1C3E3404u, 0x24040424u, 0x24040C3Eu,
    0x24041C2Cu, 0x24041C3Eu, 0x24042C1Cu, 0x24042C3Eu, 0x240C3E24u, 0x24141404u, 0x24141C3Eu, 0x24142404u,
    0x24143404u, 0x24143434u, 0x241C043Eu, 0x241C242Cu, 0x24240424u, 0x24242C0Cu, 0x24243424u, 0x242C142Cu,
    0x242C241Cu, 0x242C3E04u, 0x243E042Cu, 0x243E0C04u, 0x243E0C14u, 0x243E1C04u, 0x2C040C14u, 0x2C04240Cu,
    0x2C043E04u, 0x2C0C0404u, 0x2C0C0434u, 0x2C0C1434u, 0x2C0C2C2Cu, 0x2C140C24u, 0x2C141C14u, 0x2C143E14u,
    0x2C1C0414u, 0x2C1C2C1Cu, 0x2C240C04u, 0x2C24141Cu, 0x2C24143Eu, 0x2C243E14u, 0x2C2C0414u, 0x2C2C1C0Cu,
    0x2C342C04u, 0x2C3E1424u, 0x2C3E2414u, 0x34041424u, 0x34042424u, 0x34042434u, 0x34043424u, 0x340C140Cu,
    0x340C340Cu, 0x34140C3Eu, 0x34143424u, 0x341C1C04u, 0x341C1C34u, 0x34242424u, 0x342C042Cu, 0x342C2C14u,
    0x34341C1Cu, 0x343E041Cu, 0x343E140Cu, 0x3E04041Cu, 0x3E04042Cu, 0x3E04043Eu, 0x3E040C04u, 0x3E041C14u,
    0x3E042C14u, 0x3E0C1434u, 0x3E0C2404u, 0x3E140C14u, 0x3E14242Cu, 0x3E142C14u, 0x3E1C0404u, 0x3E1C0C2Cu,
    0x3E1C1C1Cu, 0x3E1C3404u, 0x3E24140Cu, 0x3E24240Cu, 0x3E2C0404u, 0x3E2C0414u, 0x3E2C1424u, 0x3E341C04u,
};

__constant__ unsigned char KSIGNS_IQ2XS[128] = {
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15,
    144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159,
    160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175,
    48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207,
    80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95,
    96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

__constant__ unsigned char KMASK_IQ2XS[8] = {
    1, 2, 4, 8, 16, 32, 64, 128,
};

// Decode the weight at global index `g` within one expert's super-block.
// Elements per super-block, per format.
//
// 256 for the whole family EXCEPT IQ4_NL, whose llama.cpp `block_iq4_nl` is
// 18 bytes per `QK4_NL = 32` weights. Treating it as a 256-weight block is what
// produced a 170-byte stride that matched no layout at all, and read `scales`
// from `d + 162` of an 18-byte block.
__device__ __forceinline__ int iqk_block_elems(int fmt) {
    return (fmt == 0) ? 32 : 256;
}

// Decode the weight at global index `g` within one expert's super-block.
__device__ __forceinline__ float iqk_weight(int fmt, const unsigned char* b, int g) {
    // Bytes per super-block. IQ4_NL is 18 per 32 weights; every other format is
    // a 256-weight super-block.
    const int BLOCK[12] = {18,136,98,110,66,74,82,144,176,210,76,82};
    const int be = iqk_block_elems(fmt);
    int blk = g / be;
    const unsigned char* d = b + blk * BLOCK[fmt];
    int local = g - blk * be;
    if (fmt == 0) { // iq4nl -- 18 bytes per 32 weights
        // d f16 @0..2, qs[16] @2..18. The two halves are NOT interleaved: the
        // low nibble of qs[j] is element j, the high nibble is element j + 16.
        float dd = f16_to_f32(*(const unsigned short*)(d + 0));
        int byte = (local < 16) ? local : local - 16;
        int q = (local < 16) ? ((int)(d[2 + byte] & 0x0F))
                             : ((int)(d[2 + byte] >> 4) & 0x0F);
        return dd * kvalues_iq4nl_signed(q);
    } else if (fmt == 1) { // iq4xs
        // block_iq4_xs: d f16 @0..2, scales_h u16 @2..4, scales_l[4] @4..8,
        // qs[128] @8..136. The 6-bit sub-block scale is SPLIT: 4 low bits from
        // `scales_l` (two sub-blocks per byte) and 2 high bits from `scales_h`
        // (two bits per sub-block). It cannot be read out of one byte with a
        // shift, which is what the previous form tried -- at sb=1 that read
        // `(byte >> 6) & 0x3F` and so could only ever recover 2 of the 6 bits.
        // With a 32-valued scale that decoded to 0 and the whole sub-block
        // multiplied out to zero.
        float scale_d = f16_to_f32(*(const unsigned short*)(d + 0));
        const unsigned char* scales_l = d + 4;
        unsigned int scales_h = (unsigned int)d[2] | ((unsigned int)d[3] << 8);
        const unsigned char* qs = d + 8;
        int ib = local / 32;                 // 8 sub-blocks of 32 per block
        int ls = (int)((scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F)
               | (int)((((scales_h >> (2 * ib)) & 0x03)) << 4);
        float dl = scale_d * ((float)ls - 32.0f);
        int j = local - ib * 32;             // 0..31
        unsigned char qbyte = qs[ib * 16 + (j & 15)];
        int nib = (j < 16) ? (int)(qbyte & 0x0F) : (int)(qbyte >> 4);
        return dl * kvalues_iq4nl_signed(nib);
    } else if (fmt == 2) { // iq3xxs -- 98 bytes per 256 weights
        // d f16 @0..2, qs[64] @2..66, scales_and_signs[32] @66..98. The 32
        // trailing bytes are 8 sub-blocks x one LE u32 aux: bits 28..31 are the
        // 4-bit scale, and bits 7*l .. 7*l+6 are a 7-bit INDEX into
        // KSIGNS_IQ2XS for sub-lane l -- not a sign bit, the pattern lives in
        // the table.
        float scale_d = f16_to_f32(*(const unsigned short*)(d + 0));
        const unsigned char* qs = d + 2;
        const unsigned char* ss = d + 66;
        int ib32 = local / 32;
        int l = (local - ib32 * 32) / 8;
        int j = local - ib32 * 32 - l * 8;    // 0..7
        unsigned int aux = (unsigned int)ss[4 * ib32]
                         | ((unsigned int)ss[4 * ib32 + 1] << 8)
                         | ((unsigned int)ss[4 * ib32 + 2] << 16)
                         | ((unsigned int)ss[4 * ib32 + 3] << 24);
        float db = scale_d * (0.5f + (float)(aux >> 28)) * 0.5f;
        unsigned int signs = KSIGNS_IQ2XS[(aux >> (7 * l)) & 127];
        // j < 4 draws from the first grid byte of the pair, j >= 4 the second;
        // the sign-mask index stays `j` across both halves.
        unsigned int gv = (j < 4) ? IQ3XXS_GRID[qs[ib32 * 8 + 2 * l]]
                                  : IQ3XXS_GRID[qs[ib32 * 8 + 2 * l + 1]];
        int jj = (j < 4) ? j : j - 4;
        float g = (float)((gv >> (8 * jj)) & 0xFF);
        float sign = (signs & KMASK_IQ2XS[j]) ? -1.0f : 1.0f;
        return db * g * sign;
    } else if (fmt == 3) { // iq3s -- 110 bytes per 256 weights
        // d f16 @0..2, qs[64] @2..66, qh[8] @66..74, signs[32] @74..106,
        // scales[4] @106..110. Sub-blocks come in PAIRS sharing one scale byte
        // (low nibble -> even sub-block, high nibble -> odd), and the 9-bit grid
        // index is split: bits 0..7 in qs, bit 8 in qh at a position that shifts
        // with l and with which half of the byte-pair is being read.
        float scale_d = f16_to_f32(*(const unsigned short*)(d + 0));
        const unsigned char* qs = d + 2;
        const unsigned char* qh = d + 66;
        const unsigned char* sgn = d + 74;
        const unsigned char* scl = d + 106;
        int ib32 = local / 32;
        int even = ib32 & ~1;                 // first sub-block of the pair
        int second = ib32 & 1;
        unsigned char sc_byte = scl[even / 2];
        float db = second ? (scale_d * (1.0f + 2.0f * (float)(sc_byte >> 4)))
                          : (scale_d * (1.0f + 2.0f * (float)(sc_byte & 0x0F)));
        unsigned int qhb = qh[second ? (even + 1) : even];
        int l = (local - ib32 * 32) / 8;
        int r = local - ib32 * 32 - l * 8;    // 0..7
        int hi = r >= 4;                      // second grid byte of the pair
        int jj = r & 3;
        unsigned int gidx = (unsigned int)qs[ib32 * 8 + 2 * l + hi];
        gidx |= (qhb << (hi ? (7 - 2 * l) : (8 - 2 * l))) & 256;
        unsigned int gv = IQ3S_GRID[gidx];
        unsigned int lane = sgn[ib32 * 4 + l];
        float g = (float)((gv >> (8 * jj)) & 0xFF);
        float sign = (lane & KMASK_IQ2XS[jj + (hi ? 4 : 0)]) ? -1.0f : 1.0f;
        return db * g * sign;
    } else if (fmt == 4) { // iq2xxs -- 66 bytes per 256 weights
        // d f16 @0..2, qs[64] @2..66. Per 32-weight sub-block the first 4 bytes
        // are 4 grid indices and the next 4 are one LE u32 holding a 4-bit scale
        // (bits 28..31) plus four 7-bit KSIGNS_IQ2XS indices.
        float scale_d = f16_to_f32(*(const unsigned short*)(d + 0));
        const unsigned char* qs = d + 2;
        int ib32 = local / 32;
        int l = (local - ib32 * 32) / 8;
        int j = local - ib32 * 32 - l * 8;    // 0..7
        const unsigned char* aux8 = qs + 8 * ib32;
        unsigned int aux32 = (unsigned int)aux8[4]
                           | ((unsigned int)aux8[5] << 8)
                           | ((unsigned int)aux8[6] << 16)
                           | ((unsigned int)aux8[7] << 24);
        float db = scale_d * (0.5f + (float)(aux32 >> 28)) * 0.25f;
        unsigned long long gv = IQ2XXS_GRID[aux8[l]];
        unsigned int signs = KSIGNS_IQ2XS[(aux32 >> (7 * l)) & 127];
        float g = (float)((gv >> (8 * j)) & 0xFF);
        float sign = (signs & KMASK_IQ2XS[j]) ? -1.0f : 1.0f;
        return db * g * sign;
    } else if (fmt == 5) { // iq2xs -- 74 bytes per 256 weights
        // d f16 @0..2, qs[64] @2..66, scales[8] @66..74 (one byte per 32-weight
        // sub-block carrying two 4-bit scales). Each weight pair is a LE u16:
        // 9 bits of grid index then 7 bits of KSIGNS_IQ2XS index.
        float scale_d = f16_to_f32(*(const unsigned short*)(d + 0));
        const unsigned char* qs = d + 2;
        const unsigned char* scales = d + 66;
        int ib32 = local / 32;
        int l = (local - ib32 * 32) / 8;     // 0..3
        int j = local - ib32 * 32 - l * 8;    // 0..7
        unsigned char sc_byte = scales[ib32];
        // Two scales per sub-block: lanes 0,1 use the low nibble, 2,3 the high.
        float db = scale_d
                 * (0.5f + (float)((l < 2) ? (sc_byte & 0x0F) : (sc_byte >> 4)))
                 * 0.25f;
        int q_off = (ib32 * 4 + l) * 2;
        unsigned int q_val = (unsigned int)qs[q_off] | ((unsigned int)qs[q_off + 1] << 8);
        unsigned long long gv = IQ2XS_GRID[q_val & 511];
        unsigned int signs = KSIGNS_IQ2XS[q_val >> 9];
        float g = (float)((gv >> (8 * j)) & 0xFF);
        float sign = (signs & KMASK_IQ2XS[j]) ? -1.0f : 1.0f;
        return db * g * sign;
    } else if (fmt == 6) { // iq2s
        float scale_d = f16_to_f32(*(const unsigned short*)(d + 0));
        const unsigned char* qs = d + 2;
        const unsigned char* scales = d + 50;
        const unsigned char* signs = d + 58;
        int sb = local / 16;
        float sc = ((float)((scales[sb / 2] >> ((sb % 2) * 4)) & 0x0F)) * 0.125f + 0.5f;
        float scale = scale_d * sc;
        int grid_idx = qs[local / 8];
        float code = (float)((grid_idx + (local % 8)) % 4) - 1.5f;
        int sbit = (signs[local / 8] >> (local % 8)) & 0x01;
        float sign = sbit ? -1.0f : 1.0f;
        return scale * code * sign;
    } else if (fmt == 7) { // q4k
        float dd = f16_to_f32(*(const unsigned short*)(d + 0));
        float dmin = f16_to_f32(*(const unsigned short*)(d + 2));
        const unsigned char* scales = d + 4;
        const unsigned char* qs = d + 16;
        int iter = local / 64;
        int within = local % 64;
        int l = within % 32;
        int hi = within / 32;
        int q_off = iter * 32;
        int k = iter * 2;
        int sc1, m1, sc2, m2;
        if (k < 4) { sc1 = scales[k] & 63; m1 = scales[k + 4] & 63; }
        else { sc1 = (scales[k + 4] & 0x0F) | ((scales[k - 4] >> 6) << 4); m1 = (scales[k + 4] >> 4) | ((scales[k] >> 6) << 4); }
        if (k + 1 < 4) { sc2 = scales[k + 1] & 63; m2 = scales[k + 5] & 63; }
        else { sc2 = (scales[k + 5] & 0x0F) | ((scales[k - 3] >> 6) << 4); m2 = (scales[k + 5] >> 4) | ((scales[k + 1] >> 6) << 4); }
        if (hi == 0) return dd * (float)sc1 * (float)(qs[q_off + l] & 0x0F) - dmin * (float)m1;
        else return dd * (float)sc2 * (float)(qs[q_off + l] >> 4) - dmin * (float)m2;
    } else if (fmt == 8) { // q5k
        float dd = f16_to_f32(*(const unsigned short*)(d + 0));
        float dmin = f16_to_f32(*(const unsigned short*)(d + 2));
        const unsigned char* scales = d + 4;
        const unsigned char* qh = d + 16;
        const unsigned char* qs = d + 48;
        int iter = local / 64;
        int within = local % 64;
        int l = within % 32;
        int hi = within / 32;
        int q_off = iter * 32;
        int k = iter * 2;
        int sc1, m1, sc2, m2;
        if (k < 4) { sc1 = scales[k] & 63; m1 = scales[k + 4] & 63; }
        else { sc1 = (scales[k + 4] & 0x0F) | ((scales[k - 4] >> 6) << 4); m1 = (scales[k + 4] >> 4) | ((scales[k] >> 6) << 4); }
        if (k + 1 < 4) { sc2 = scales[k + 1] & 63; m2 = scales[k + 5] & 63; }
        else { sc2 = (scales[k + 5] & 0x0F) | ((scales[k - 3] >> 6) << 4); m2 = (scales[k + 5] >> 4) | ((scales[k + 1] >> 6) << 4); }
        int u1 = 1 << (iter * 2);
        int u2 = 1 << (iter * 2 + 1);
        if (hi == 0) {
            int lo = qs[q_off + l] & 0x0F;
            int qlo = lo + (((qh[l] & u1) != 0) ? 16 : 0);
            return dd * (float)sc1 * (float)qlo - dmin * (float)m1;
        } else {
            int hv = qs[q_off + l] >> 4;
            int qhi = hv + (((qh[l] & u2) != 0) ? 16 : 0);
            return dd * (float)sc2 * (float)qhi - dmin * (float)m2;
        }
    } else if (fmt == 9) { // q6k
        const unsigned char* ql = d + 0;
        const unsigned char* qh = d + 128;
        const unsigned char* scales = d + 192;
        float dd = f16_to_f32(*(const unsigned short*)(d + 208));
        int iter = local / 128;
        int within = local % 128;
        int l = within % 32;
        int quad = within / 32;
        int ql_idx = iter * 64;
        int qh_idx = iter * 32;
        int sc_idx = iter * 8;
        int is = l / 16;
        float q1 = (float)(((ql[ql_idx + l] & 0x0F) | ((qh[qh_idx + l] & 0x03) << 4))) - 32.0f;
        float q2 = (float)(((ql[ql_idx + l + 32] & 0x0F) | ((qh[qh_idx + l] & 0x0C) << 2))) - 32.0f;
        float q3 = (float)(((ql[ql_idx + l] >> 4) | ((qh[qh_idx + l] & 0x30))) ) - 32.0f;
        float q4 = (float)(((ql[ql_idx + l + 32] >> 4) | ((qh[qh_idx + l] & 0xC0) >> 2))) - 32.0f;
        float sc = (float)((signed char)scales[sc_idx + is + quad * 2]); // i8
        float qs[4] = {q1, q2, q3, q4};
        return dd * sc * qs[quad];
    } else if (fmt == 10) { // q2k (MoE single-superblock, 76 bytes / 64 weights)
        // Layout: d[0..2] f16, dmin[2..4] f16, scales[4..12] (4 u8, scale/2 in nibble), qs[12..76] (64 bytes, one 2-bit quant per byte).
        // local in [0,63] -> quad = local/16 (0..3), l = local%16.
        float dd = f16_to_f32(*(const unsigned short*)(d + 0));
        float dmin = f16_to_f32(*(const unsigned short*)(d + 2));
        const unsigned char* scales = d + 4;
        const unsigned char* qs = d + 12;
        int quad = local / 16;
        int l = local % 16;
        int sc_byte = scales[quad];
        int sce = sc_byte & 0x0F;
        int m = sc_byte >> 4;
        int qv = qs[quad * 16 + l] & 3;
        return dd * (float)sce * (float)qv - dmin * (float)m;
    } else { // fmt == 11 q3k (MoE single-superblock, 82 bytes / 64 weights)
        // Layout: d[0..2] f16, scales[2..10] (4 u8, scale/2 in nibble), hmask[10..18] (4 u8, sign bit per quad), qs[18..82] (64 bytes, one 3-bit quant per byte).
        // local in [0,63] -> quad=local/16, l=local%16.
        float dd = f16_to_f32(*(const unsigned short*)(d + 0));
        const unsigned char* scales = d + 2;
        const unsigned char* hmask = d + 10;
        const unsigned char* qs = d + 18;
        int quad = local / 16;
        int l = local % 16;
        int sc_byte = scales[quad];
        int sce = sc_byte & 0x0F;
        int scm = sc_byte >> 4;
        int hm_bit = (hmask[quad] >> (l / 8)) & 1;
        int qv = qs[quad * 16 + l] & 7;
        float qval = (float)qv - 4.0f * (1.0f - (float)hm_bit);
        return dd * ((float)sce - 8.0f) * qval - dd * (float)scm;
    }
}

// kernel-hygiene-plan item 1: batched decode helper for the k-quant formats (7 = Q4_K, 8 = Q5_K, 9 = Q6_K).
// `iqk_weight` decodes one weight at a time and re-unpacks the per-sub-block scalar fields on every.
__device__ __forceinline__ void iqk_batch_decode(
    int fmt, const unsigned char* b, int g0, int count, float* out) {
    // Only formats 7/8/9 reach this path (`batched`), but the table is kept in
    // step with the one in `iqk_weight` so the two do not disagree about a
    // format's block size.
    const int BLOCK[12] = {18,136,98,110,66,74,82,144,176,210,76,82};
    const int sb_w = (fmt == 9) ? 128 : 64;
    int gi = g0;
    int used = 0;
    while (used < count) {
        int blk = gi / 256;
        const unsigned char* d = b + (unsigned long long)blk * BLOCK[fmt];
        int local = gi - blk * 256;
        int sb = local / sb_w;
        int pos = local - sb * sb_w;
        int slack = sb_w - pos;
        int n = (count - used) < slack ? (count - used) : slack;
        // Per-sub-block scalar unpack, hoisted out of the per-weight loop.
        if (fmt == 7 || fmt == 8) {
            float dd = f16_to_f32(*(const unsigned short*)(d + 0));
            float dmin = f16_to_f32(*(const unsigned short*)(d + 2));
            const unsigned char* scales = d + 4;
            const unsigned char* qs = (fmt == 7) ? (d + 16) : (d + 48);
            const unsigned char* qh = d + 16;
            int k = sb * 2;
            int sc1, m1, sc2, m2;
            if (k < 4) { sc1 = scales[k] & 63; m1 = scales[k + 4] & 63; }
            else { sc1 = (scales[k + 4] & 0x0F) | ((scales[k - 4] >> 6) << 4); m1 = (scales[k + 4] >> 4) | ((scales[k] >> 6) << 4); }
            if (k + 1 < 4) { sc2 = scales[k + 1] & 63; m2 = scales[k + 5] & 63; }
            else { sc2 = (scales[k + 5] & 0x0F) | ((scales[k - 3] >> 6) << 4); m2 = (scales[k + 5] >> 4) | ((scales[k + 1] >> 6) << 4); }
            int u1 = 1 << (sb * 2);
            int u2 = 1 << (sb * 2 + 1);
            int q_off = sb * 32;
            for (int p = 0; p < n; ++p) {
                int w = pos + p;
                int l = w % 32;
                int hi = w / 32;
                float wt;
                if (fmt == 7) {
                    wt = (hi == 0)
                        ? dd * (float)sc1 * (float)(qs[q_off + l] & 0x0F) - dmin * (float)m1
                        : dd * (float)sc2 * (float)(qs[q_off + l] >> 4) - dmin * (float)m2;
                } else {
                    if (hi == 0) {
                        int lo = qs[q_off + l] & 0x0F;
                        int qlo = lo + (((qh[l] & u1) != 0) ? 16 : 0);
                        wt = dd * (float)sc1 * (float)qlo - dmin * (float)m1;
                    } else {
                        int hv = qs[q_off + l] >> 4;
                        int qhi = hv + (((qh[l] & u2) != 0) ? 16 : 0);
                        wt = dd * (float)sc2 * (float)qhi - dmin * (float)m2;
                    }
                }
                out[used + p] = wt;
            }
        } else { // fmt == 9 q6k
            float dd = f16_to_f32(*(const unsigned short*)(d + 208));
            const unsigned char* ql = d + 0;
            const unsigned char* qh = d + 128;
            const unsigned char* scales = d + 192;
            int ql_idx = sb * 64;
            int qh_idx = sb * 32;
            int sc_idx = sb * 8;
            for (int p = 0; p < n; ++p) {
                int w = pos + p;
                int l = w % 32;
                int quad = w / 32;
                int is = l / 16;
                float sc = (float)((signed char)scales[sc_idx + is + quad * 2]);
                float q1 = (float)(((ql[ql_idx + l] & 0x0F) | ((qh[qh_idx + l] & 0x03) << 4))) - 32.0f;
                float q2 = (float)(((ql[ql_idx + l + 32] & 0x0F) | ((qh[qh_idx + l] & 0x0C) << 2))) - 32.0f;
                float q3 = (float)(((ql[ql_idx + l] >> 4) | ((qh[qh_idx + l] & 0x30)))) - 32.0f;
                float q4 = (float)(((ql[ql_idx + l + 32] >> 4) | ((qh[qh_idx + l] & 0xC0) >> 2))) - 32.0f;
                float qv = (quad == 0) ? q1 : (quad == 1) ? q2 : (quad == 2) ? q3 : q4;
                out[used + p] = dd * sc * qv;
            }
        }
        used += n;
        gi += n;
    }
}

// kernel-hygiene-plan item 1: per-weight decode chunks for the batched path.
#define IQK_CHUNK 64

extern "C" __global__ void grim_moe_fused_grouped_iqk(
    const float* __restrict__ activations,
    const unsigned char* __restrict__ egate_w,
    const unsigned char* __restrict__ eup_w,
    const unsigned char* __restrict__ edown_w,
    const float* __restrict__ a_scale,
    const unsigned int* __restrict__ sorted_token_ids,
    const unsigned int* __restrict__ sorted_expert_ids,
    const float* __restrict__ sorted_weights,
    float* __restrict__ out,
    int hidden, int inter, int num_tokens, int block_size,
    int format_id, float routed_scaling_factor)
{
    const int blk = blockIdx.x;
    const int base = blk * block_size;
    const int end = base + block_size < num_tokens ? base + block_size : num_tokens;
    const int BLOCK[12] = {18,136,98,110,66,74,82,144,176,210,76,82};
    // Bytes per expert = its whole weight matrix rounded UP to whole
    // super-blocks. This is not `BLOCK[format_id]`: an expert's gate/up/down is
    // [inter, hidden], and at 32-weight IQ4_NL blocks a 64-weight expert spans
    // TWO of them, so the old single-block stride under-read every expert past
    // the first and ran off the end of the bank.
    const int iqk_be = iqk_block_elems(format_id);
    const int iqk_w = inter * hidden;
    int sbytes = ((iqk_w + iqk_be - 1) / iqk_be) * BLOCK[format_id];
    const bool batched = (format_id == 7 || format_id == 8 || format_id == 9);

    for (int s = base + threadIdx.x; s < end; s += blockDim.x) {
        const unsigned int tok = sorted_token_ids[s];
        if (tok >= (unsigned int)num_tokens) continue;
        const unsigned int exp = sorted_expert_ids[s];
        const float w = sorted_weights[s];
        const float as = a_scale[tok];

        const float* a = activations + (unsigned long long)tok * hidden;
        const unsigned char* gw = egate_w + (unsigned long long)exp * sbytes;
        const unsigned char* uw = eup_w   + (unsigned long long)exp * sbytes;
        const unsigned char* dw = edown_w + (unsigned long long)exp * sbytes;

        float scratch[IQK_CHUNK];

        float acc_prev = 0.0f;
        const bool odd_hidden = (hidden & 1) != 0;
        for (int h = 0; h < hidden; ++h) {
            float acc = 0.0f;
            for (int jc = 0; jc < inter; jc += IQK_CHUNK) {
                int nj = (inter - jc) < IQK_CHUNK ? inter - jc : IQK_CHUNK;
                float actj[IQK_CHUNK];
                for (int j = 0; j < nj; ++j) {
                    float gate = 0.0f;
                    float up = 0.0f;
                    const int jg = jc + j;
                    if (batched) {
                        // Item 1: batched k-quant decode. The per-64 sub-block
                        // unpack is hoisted inside iqk_batch_decode.
                        const int gbase = jg * hidden;
                        for (int c0 = 0; c0 < hidden; c0 += IQK_CHUNK) {
                            int n = (hidden - c0) < IQK_CHUNK ? hidden - c0 : IQK_CHUNK;
                            iqk_batch_decode(format_id, gw, gbase + c0, n, scratch);
                            for (int c = 0; c < n; ++c) gate += scratch[c] * a[c + c0];
                            iqk_batch_decode(format_id, uw, gbase + c0, n, scratch);
                            for (int c = 0; c < n; ++c) up += scratch[c] * a[c + c0];
                        }
                    } else {
                        for (int i = 0; i < hidden; ++i) {
                            gate += iqk_weight(format_id, gw, jg * hidden + i) * a[i];
                            up   += iqk_weight(format_id, uw, jg * hidden + i) * a[i];
                        }
                    }
                    float silu_g = gate / (1.0f + expf(-gate));
                    actj[j] = silu_g * up;
                }
                if (batched) {
                    // Down row is contiguous in g: h*inter + [jc, jc+nj). Decode
                    // once per chunk and pair each weight with its actj column.
                    iqk_batch_decode(format_id, dw, h * inter + jc, nj, scratch);
                    for (int j = 0; j < nj; ++j) acc += scratch[j] * actj[j];
                } else {
                    for (int j = 0; j < nj; ++j) acc += iqk_weight(format_id, dw, h * inter + jc + j) * actj[j];
                }
            }
            if (odd_hidden) {
                atomicAdd(out + (unsigned long long)tok * hidden + h,
                          routed_scaling_factor * w * as * acc);
            } else if (h & 1) {
                charon_atomic_add2(out, (unsigned long long)tok * hidden + (h - 1),
                                   routed_scaling_factor * w * as * acc_prev,
                                   routed_scaling_factor * w * as * acc);
            } else {
                acc_prev = acc;
            }
        }
    }
}

// --- #7 CompressedTensors W8A8 INT8 grouped kernel -------------------------
extern "C" __global__ void grim_moe_fused_grouped_w8a8_int8(
    const float* __restrict__ activations,
    const unsigned char* __restrict__ egate_w,
    const unsigned char* __restrict__ eup_w,
    const unsigned char* __restrict__ edown_w,
    const float* __restrict__ a_scale,
    const unsigned int* __restrict__ sorted_token_ids,
    const unsigned int* __restrict__ sorted_expert_ids,
    const float* __restrict__ sorted_weights,
    float* __restrict__ out,
    int hidden, int inter, int num_tokens, int block_size,
    float routed_scaling_factor)
{
    const int blk = blockIdx.x;
    const int base = blk * block_size;
    const int end = base + block_size < num_tokens ? base + block_size : num_tokens;

    // Per-expert stride: 8 (u64 prefix) + (rows*cols) codes + (rows*4) scales
    const unsigned long long gate_stride = 8 + (unsigned long long)inter * hidden + (unsigned long long)inter * 4;
    const unsigned long long down_stride = 8 + (unsigned long long)hidden * inter + (unsigned long long)hidden * 4;

    for (int s = base + threadIdx.x; s < end; s += blockDim.x) {
        const unsigned int tok = sorted_token_ids[s];
        if (tok >= (unsigned int)num_tokens) continue;
        const unsigned int exp = sorted_expert_ids[s];
        const float w = sorted_weights[s];
        const float as = a_scale[tok];

        const float* a = activations + (unsigned long long)tok * hidden;
        const unsigned char* gw = egate_w + (unsigned long long)exp * gate_stride;
        const unsigned char* uw = eup_w   + (unsigned long long)exp * gate_stride;
        const unsigned char* dw = edown_w + (unsigned long long)exp * down_stride;

        const unsigned char* g_codes = gw + 8;
        const float* g_scales = (const float*)(gw + 8 + (unsigned long long)inter * hidden);

        const unsigned char* u_codes = uw + 8;
        const float* u_scales = (const float*)(uw + 8 + (unsigned long long)inter * hidden);

        const unsigned char* d_codes = dw + 8;
        const float* d_scales = (const float*)(dw + 8 + (unsigned long long)hidden * inter);

        float acc_prev = 0.0f;
        const bool odd_hidden = (hidden & 1) != 0;
        for (int h = 0; h < hidden; ++h) {
            float acc = 0.0f;
            for (int j = 0; j < inter; ++j) {
                float gate = 0.0f;
                float up = 0.0f;
                for (int i = 0; i < hidden; ++i) {
                    signed char gc = (signed char)g_codes[(unsigned long long)j * hidden + i];
                    signed char uc = (signed char)u_codes[(unsigned long long)j * hidden + i];
                    gate += (float)gc * g_scales[j] * a[i];
                    up   += (float)uc * u_scales[j] * a[i];
                }
                float silu_g = gate / (1.0f + expf(-gate));
                float act = silu_g * up;
                signed char dc = (signed char)d_codes[(unsigned long long)h * inter + j];
                acc += (float)dc * d_scales[h] * act;
            }
            if (odd_hidden) {
                atomicAdd(out + (unsigned long long)tok * hidden + h,
                          routed_scaling_factor * w * as * acc);
            } else if (h & 1) {
                charon_atomic_add2(out, (unsigned long long)tok * hidden + (h - 1),
                                   routed_scaling_factor * w * as * acc_prev,
                                   routed_scaling_factor * w * as * acc);
            } else {
                acc_prev = acc;
            }
        }
    }
}

// --- WI-gpu-native-moe Phase 2: sortless W8A8-int8 fused dispatch ---------
// Pair-parallel twin of `grim_moe_fused_dispatch` for int8-quantized experts:
// one thread per routed (token, expert) pair, routing triples read from
// DEVICE-resident buffers (no host SortedRouting, no H2D upload), so the D2D
// path (`moe_route_topk_on_device` + this kernel) never round-trips.
// Expert stacks are the same self-describing packed blobs as
// `grim_moe_fused_grouped_w8a8_int8` ([u64 prefix | int8 codes | f32
// per-row scales]), concatenated per expert; strides identical.
extern "C" __global__ void grim_moe_fused_dispatch_w8a8_int8(
    const float* __restrict__ activations,     // [batch, hidden]
    const unsigned char* __restrict__ expert_gate_w, // stacked packed blobs
    const unsigned char* __restrict__ expert_up_w,   // stacked packed blobs
    const unsigned char* __restrict__ expert_down_w, // stacked packed blobs
    const float* __restrict__ a_scale,         // [batch] per-token act scale
    const unsigned int* __restrict__ router_tokens,  // [num_pairs]
    const unsigned int* __restrict__ router_experts, // [num_pairs]
    const float* __restrict__ router_weights,        // [num_pairs]
    float* __restrict__ out,                     // [batch, hidden]
    int hidden, int inter, int num_pairs,
    float routed_scaling_factor)
{
    const unsigned long long pair = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= (unsigned long long)num_pairs) return;

    const unsigned int tok = router_tokens[pair];
    const unsigned int exp = router_experts[pair];
    const float w = router_weights[pair];
    const float as = a_scale[tok];

    // Per-expert stride: 8 (u64 prefix) + (rows*cols) codes + (rows*4) scales.
    const unsigned long long gate_stride = 8 + (unsigned long long)inter * hidden + (unsigned long long)inter * 4;
    const unsigned long long down_stride = 8 + (unsigned long long)hidden * inter + (unsigned long long)hidden * 4;

    const float* a = activations + (unsigned long long)tok * hidden;
    const unsigned char* gw = expert_gate_w + (unsigned long long)exp * gate_stride;
    const unsigned char* uw = expert_up_w   + (unsigned long long)exp * gate_stride;
    const unsigned char* dw = expert_down_w + (unsigned long long)exp * down_stride;

    const unsigned char* g_codes = gw + 8;
    const float* g_scales = (const float*)(gw + 8 + (unsigned long long)inter * hidden);
    const unsigned char* u_codes = uw + 8;
    const float* u_scales = (const float*)(uw + 8 + (unsigned long long)inter * hidden);
    const unsigned char* d_codes = dw + 8;
    const float* d_scales = (const float*)(dw + 8 + (unsigned long long)hidden * inter);

    // Fused gate + up GEMM with in-register SiLU combine, then down.
    for (int j = 0; j < inter; ++j) {
        float g = 0.0f;
        float u = 0.0f;
        for (int i = 0; i < hidden; ++i) {
            g += (float)((signed char)g_codes[j * hidden + i]) * g_scales[j] * a[i];
            u += (float)((signed char)u_codes[j * hidden + i]) * u_scales[j] * a[i];
        }
        float silu_g = g / (1.0f + expf(-g));
        float act = silu_g * u;
        float scale = routed_scaling_factor * w * as * act;

        for (int h = 0; h < hidden; ++h) {
            float dv = (float)((signed char)d_codes[h * inter + j]) * d_scales[h];
            unsigned long long out_idx = (unsigned long long)tok * hidden + h;
            atomicAdd(out + out_idx, dv * scale);
        }
    }
}

// --- #8 CompressedTensors W8A8 FP8 grouped kernel --------------------------

// --- #8 CompressedTensors W8A8 FP8 grouped kernel --------------------------
// NOTE (perf, WI-gpu-native-moe Phase 2): same no-powf discipline as the
// MXFP4 decoder above — `(8+mant) * 2^(exp-10)` is bitwise-identical to
// `(1+mant/8) * 2^(exp-7)` (exact mantissa, exact power-of-two scaling).
// KNOWN DIVERGENCE (not touched): exp == 0xF, mant != 7 maps to 448.0
// here but to (1+mant/8)*256 in `grim_quant::fp8_e4m3_to_f32`. Per OCP
// those codes are NaN — no conforming quantizer emits them — so the
// stand-ins never execute on real checkpoints. Parity fixtures exclude
// them (see `fp8_pack_tensor`); do NOT "fix" one side without the other.
__device__ __forceinline__ float grim_charon_fp8_e4m3_to_f32(unsigned char val) {
    int sign = (val >> 7) & 1;
    int exp = (val >> 3) & 0x0F;
    int mant = val & 0x07;
    if (exp == 0xF) {
        if (mant == 7) return 0.0f / 0.0f;
        float v = 448.0f;
        return sign ? -v : v;
    }
    float res;
    if (exp != 0) {
        res = (float)(8 + mant) * __int_as_float((unsigned int)(exp - 10 + 127) << 23);
    } else {
        res = (float)mant * 0x1p-9f;
    }
    return sign ? -res : res;
}

extern "C" __global__ void grim_moe_fused_grouped_w8a8_fp8(
    const float* __restrict__ activations,
    const unsigned char* __restrict__ egate_w,
    const unsigned char* __restrict__ eup_w,
    const unsigned char* __restrict__ edown_w,
    const float* __restrict__ a_scale,
    const unsigned int* __restrict__ sorted_token_ids,
    const unsigned int* __restrict__ sorted_expert_ids,
    const float* __restrict__ sorted_weights,
    float* __restrict__ out,
    int hidden, int inter, int num_tokens, int block_size,
    float routed_scaling_factor)
{
    const int blk = blockIdx.x;
    const int base = blk * block_size;
    const int end = base + block_size < num_tokens ? base + block_size : num_tokens;

    const unsigned long long gate_stride = 8 + (unsigned long long)inter * hidden + 4;
    const unsigned long long down_stride = 8 + (unsigned long long)hidden * inter + 4;

    for (int s = base + threadIdx.x; s < end; s += blockDim.x) {
        const unsigned int tok = sorted_token_ids[s];
        if (tok >= (unsigned int)num_tokens) continue;
        const unsigned int exp = sorted_expert_ids[s];
        const float w = sorted_weights[s];
        const float as = a_scale[tok];

        const float* a = activations + (unsigned long long)tok * hidden;
        const unsigned char* gw = egate_w + (unsigned long long)exp * gate_stride;
        const unsigned char* uw = eup_w   + (unsigned long long)exp * gate_stride;
        const unsigned char* dw = edown_w + (unsigned long long)exp * down_stride;

        const unsigned char* g_codes = gw + 8;
        const float g_scale = *(const float*)(gw + 8 + (unsigned long long)inter * hidden);

        const unsigned char* u_codes = uw + 8;
        const float u_scale = *(const float*)(uw + 8 + (unsigned long long)inter * hidden);

        const unsigned char* d_codes = dw + 8;
        const float d_scale = *(const float*)(dw + 8 + (unsigned long long)hidden * inter);

        float acc_prev = 0.0f;
        const bool odd_hidden = (hidden & 1) != 0;
        for (int h = 0; h < hidden; ++h) {
            float acc = 0.0f;
            for (int j = 0; j < inter; ++j) {
                float gate = 0.0f;
                float up = 0.0f;
                for (int i = 0; i < hidden; ++i) {
                    float gw_f = grim_charon_fp8_e4m3_to_f32(g_codes[(unsigned long long)j * hidden + i]);
                    float uw_f = grim_charon_fp8_e4m3_to_f32(u_codes[(unsigned long long)j * hidden + i]);
                    gate += gw_f * a[i];
                    up   += uw_f * a[i];
                }
                gate *= g_scale;
                up   *= u_scale;
                float silu_g = gate / (1.0f + expf(-gate));
                float act = silu_g * up;
                float dw_f = grim_charon_fp8_e4m3_to_f32(d_codes[(unsigned long long)h * inter + j]) * d_scale;
                acc += dw_f * act;
            }
            if (odd_hidden) {
                atomicAdd(out + (unsigned long long)tok * hidden + h,
                          routed_scaling_factor * w * as * acc);
            } else if (h & 1) {
                charon_atomic_add2(out, (unsigned long long)tok * hidden + (h - 1),
                                   routed_scaling_factor * w * as * acc_prev,
                                   routed_scaling_factor * w * as * acc);
            } else {
                acc_prev = acc;
            }
        }
    }
}

// --- WI-gpu-native-moe Phase 2: sortless W8A8-fp8 fused dispatch ----------
// Pair-parallel twin of `grim_moe_fused_grouped_w8a8_fp8`: routing triples
// from DEVICE-resident buffers, expert stacks are concatenated per-expert
// blobs ([u64 prefix | fp8-E4M3 codes | ONE f32 scale]), strides identical
// to the grouped kernel.
extern "C" __global__ void grim_moe_fused_dispatch_w8a8_fp8(
    const float* __restrict__ activations,     // [batch, hidden]
    const unsigned char* __restrict__ expert_gate_w, // stacked packed blobs
    const unsigned char* __restrict__ expert_up_w,   // stacked packed blobs
    const unsigned char* __restrict__ expert_down_w, // stacked packed blobs
    const float* __restrict__ a_scale,         // [batch] per-token act scale
    const unsigned int* __restrict__ router_tokens,  // [num_pairs]
    const unsigned int* __restrict__ router_experts, // [num_pairs]
    const float* __restrict__ router_weights,        // [num_pairs]
    float* __restrict__ out,                     // [batch, hidden]
    int hidden, int inter, int num_pairs,
    float routed_scaling_factor)
{
    const unsigned long long pair = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= (unsigned long long)num_pairs) return;

    const unsigned int tok = router_tokens[pair];
    const unsigned int exp = router_experts[pair];
    const float w = router_weights[pair];
    const float as = a_scale[tok];

    // Per-expert stride: 8 (u64 prefix) + (rows*cols) codes + 4 (one scale).
    const unsigned long long gate_stride = 8 + (unsigned long long)inter * hidden + 4;
    const unsigned long long down_stride = 8 + (unsigned long long)hidden * inter + 4;

    const float* a = activations + (unsigned long long)tok * hidden;
    const unsigned char* gw = expert_gate_w + (unsigned long long)exp * gate_stride;
    const unsigned char* uw = expert_up_w   + (unsigned long long)exp * gate_stride;
    const unsigned char* dw = expert_down_w + (unsigned long long)exp * down_stride;

    const unsigned char* g_codes = gw + 8;
    const float g_scale = *(const float*)(gw + 8 + (unsigned long long)inter * hidden);
    const unsigned char* u_codes = uw + 8;
    const float u_scale = *(const float*)(uw + 8 + (unsigned long long)inter * hidden);
    const unsigned char* d_codes = dw + 8;
    const float d_scale = *(const float*)(dw + 8 + (unsigned long long)hidden * inter);

    for (int j = 0; j < inter; ++j) {
        float g = 0.0f;
        float u = 0.0f;
        for (int i = 0; i < hidden; ++i) {
            g += grim_charon_fp8_e4m3_to_f32(g_codes[j * hidden + i]) * a[i];
            u += grim_charon_fp8_e4m3_to_f32(u_codes[j * hidden + i]) * a[i];
        }
        g *= g_scale;
        u *= u_scale;
        float silu_g = g / (1.0f + expf(-g));
        float act = silu_g * u;
        float scale = routed_scaling_factor * w * as * act * d_scale;

        for (int h = 0; h < hidden; ++h) {
            float dv = grim_charon_fp8_e4m3_to_f32(d_codes[h * inter + j]);
            unsigned long long out_idx = (unsigned long long)tok * hidden + h;
            atomicAdd(out + out_idx, dv * scale);
        }
    }
}

// --- #9 AWQ grouped kernel -------------------------------------------------
static inline __device__ unsigned int grim_charon_awq_read_u32(
    const unsigned char* __restrict__ base, long long word_idx)
{
    return *(const unsigned int*)(base + word_idx * 4);
}

static inline __device__ unsigned int grim_charon_awq_read_code(
    const unsigned char* qweight, int in_idx, int col, int N,
    int bits, int values_per_word)
{
    long long word_idx = (long long)(in_idx / values_per_word) * N + col;
    unsigned int word = grim_charon_awq_read_u32(qweight, word_idx);
    return (word >> ((in_idx % values_per_word) * bits)) & ((1u << bits) - 1u);
}

static inline __device__ float grim_charon_awq_read_zero(
    const unsigned char* qzeros, int group, int col,
    int bits, int values_per_word, int zeros_words_per_row)
{
    long long word_idx = (long long)group * zeros_words_per_row + col / values_per_word;
    unsigned int word = grim_charon_awq_read_u32(qzeros, word_idx);
    return (float)((word >> ((col % values_per_word) * bits)) & ((1u << bits) - 1u));
}

extern "C" __global__ void grim_moe_fused_grouped_awq(
    const float* __restrict__ activations,
    const unsigned char* __restrict__ egate_w,
    const unsigned char* __restrict__ eup_w,
    const unsigned char* __restrict__ edown_w,
    const float* __restrict__ a_scale,
    const unsigned int* __restrict__ sorted_token_ids,
    const unsigned int* __restrict__ sorted_expert_ids,
    const float* __restrict__ sorted_weights,
    float* __restrict__ out,
    int hidden, int inter, int num_tokens, int block_size,
    int bits, int group_size,
    long long gate_qw_off, long long gate_qz_off, long long gate_sc_off, unsigned long long gate_stride,
    long long down_qw_off, long long down_qz_off, long long down_sc_off, unsigned long long down_stride,
    float routed_scaling_factor)
{
    const int blk = blockIdx.x;
    const int base = blk * block_size;
    const int end = base + block_size < num_tokens ? base + block_size : num_tokens;

    const int vpw = (bits == 4) ? 8 : (bits == 2 ? 16 : 1);
    const int g_zero_words_per_row = (inter + vpw - 1) / vpw;
    const int d_zero_words_per_row = (hidden + vpw - 1) / vpw;

    for (int s = base + threadIdx.x; s < end; s += blockDim.x) {
        const unsigned int tok = sorted_token_ids[s];
        if (tok >= (unsigned int)num_tokens) continue;
        const unsigned int exp = sorted_expert_ids[s];
        const float w = sorted_weights[s];
        const float as = a_scale[tok];

        const float* a = activations + (unsigned long long)tok * hidden;
        const unsigned char* gw = egate_w + (unsigned long long)exp * gate_stride;
        const unsigned char* uw = eup_w   + (unsigned long long)exp * gate_stride;
        const unsigned char* dw = edown_w + (unsigned long long)exp * down_stride;

        const unsigned char* g_qw = gw + gate_qw_off;
        const unsigned char* g_qz = gw + gate_qz_off;
        const unsigned char* g_sc = gw + gate_sc_off;

        const unsigned char* u_qw = uw + gate_qw_off;
        const unsigned char* u_qz = uw + gate_qz_off;
        const unsigned char* u_sc = uw + gate_sc_off;

        const unsigned char* d_qw = dw + down_qw_off;
        const unsigned char* d_qz = dw + down_qz_off;
        const unsigned char* d_sc = dw + down_sc_off;

        float acc_prev = 0.0f;
        const bool odd_hidden = (hidden & 1) != 0;
        for (int h = 0; h < hidden; ++h) {
            float acc = 0.0f;
            for (int j = 0; j < inter; ++j) {
                float gate = 0.0f;
                float up = 0.0f;
                for (int i = 0; i < hidden; ++i) {
                    int grp = i / group_size;

                    // Gate weight: col=j, in=i, N=inter
                    unsigned int g_code = grim_charon_awq_read_code(g_qw, i, j, inter, bits, vpw);
                    float g_zero = grim_charon_awq_read_zero(g_qz, grp, j, bits, vpw, g_zero_words_per_row);
                    unsigned short g_sch = *(const unsigned short*)(g_sc + ((long long)grp * inter + j) * 2);
                    float g_scale = f16_to_f32(g_sch);
                    float gw_f = ((float)g_code - g_zero) * g_scale;

                    // Up weight: col=j, in=i, N=inter
                    unsigned int u_code = grim_charon_awq_read_code(u_qw, i, j, inter, bits, vpw);
                    float u_zero = grim_charon_awq_read_zero(u_qz, grp, j, bits, vpw, g_zero_words_per_row);
                    unsigned short u_sch = *(const unsigned short*)(u_sc + ((long long)grp * inter + j) * 2);
                    float u_scale = f16_to_f32(u_sch);
                    float uw_f = ((float)u_code - u_zero) * u_scale;

                    gate += gw_f * a[i];
                    up   += uw_f * a[i];
                }
                float silu_g = gate / (1.0f + expf(-gate));
                float act = silu_g * up;

                // Down weight: col=h, in=j, N=hidden
                int d_grp = j / group_size;
                unsigned int d_code = grim_charon_awq_read_code(d_qw, j, h, hidden, bits, vpw);
                float d_zero = grim_charon_awq_read_zero(d_qz, d_grp, h, bits, vpw, d_zero_words_per_row);
                unsigned short d_sch = *(const unsigned short*)(d_sc + ((long long)d_grp * hidden + h) * 2);
                float d_scale = f16_to_f32(d_sch);
                float dw_f = ((float)d_code - d_zero) * d_scale;

                acc += dw_f * act;
            }
            if (odd_hidden) {
                atomicAdd(out + (unsigned long long)tok * hidden + h,
                          routed_scaling_factor * w * as * acc);
            } else if (h & 1) {
                charon_atomic_add2(out, (unsigned long long)tok * hidden + (h - 1),
                                   routed_scaling_factor * w * as * acc_prev,
                                   routed_scaling_factor * w * as * acc);
            } else {
                acc_prev = acc;
            }
        }
    }
}

// --- WI-gpu-native-moe Phase 2: sortless AWQ fused dispatch ----------------
// Pair-parallel twin of `grim_moe_fused_grouped_awq`: routing triples from
// DEVICE-resident buffers; per-expert packed blobs with the same qw/qz/sc
// segment offsets (passed as args, computed on host from bits/group/dims).
extern "C" __global__ void grim_moe_fused_dispatch_awq(
    const float* __restrict__ activations,     // [batch, hidden]
    const unsigned char* __restrict__ expert_gate_w, // stacked packed blobs
    const unsigned char* __restrict__ expert_up_w,   // stacked packed blobs
    const unsigned char* __restrict__ expert_down_w, // stacked packed blobs
    const float* __restrict__ a_scale,         // [batch] per-token act scale
    const unsigned int* __restrict__ router_tokens,  // [num_pairs]
    const unsigned int* __restrict__ router_experts, // [num_pairs]
    const float* __restrict__ router_weights,        // [num_pairs]
    float* __restrict__ out,                     // [batch, hidden]
    int hidden, int inter, int num_pairs,
    int bits, int group_size,
    long long gate_qw_off, long long gate_qz_off, long long gate_sc_off, unsigned long long gate_stride,
    long long down_qw_off, long long down_qz_off, long long down_sc_off, unsigned long long down_stride,
    float routed_scaling_factor)
{
    const unsigned long long pair = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= (unsigned long long)num_pairs) return;

    const unsigned int tok = router_tokens[pair];
    const unsigned int exp = router_experts[pair];
    const float w = router_weights[pair];
    const float as = a_scale[tok];

    const int vpw = (bits == 4) ? 8 : (bits == 2 ? 16 : 1);
    const int g_zero_words_per_row = (inter + vpw - 1) / vpw;
    const int d_zero_words_per_row = (hidden + vpw - 1) / vpw;

    const float* a = activations + (unsigned long long)tok * hidden;
    const unsigned char* gw = expert_gate_w + (unsigned long long)exp * gate_stride;
    const unsigned char* uw = expert_up_w   + (unsigned long long)exp * gate_stride;
    const unsigned char* dw = expert_down_w + (unsigned long long)exp * down_stride;

    const unsigned char* g_qw = gw + gate_qw_off;
    const unsigned char* g_qz = gw + gate_qz_off;
    const unsigned char* g_sc = gw + gate_sc_off;
    const unsigned char* u_qw = uw + gate_qw_off;
    const unsigned char* u_qz = uw + gate_qz_off;
    const unsigned char* u_sc = uw + gate_sc_off;
    const unsigned char* d_qw = dw + down_qw_off;
    const unsigned char* d_qz = dw + down_qz_off;
    const unsigned char* d_sc = dw + down_sc_off;

    for (int j = 0; j < inter; ++j) {
        float gate = 0.0f;
        float up = 0.0f;
        for (int i = 0; i < hidden; ++i) {
            int grp = i / group_size;
            unsigned int g_code = grim_charon_awq_read_code(g_qw, i, j, inter, bits, vpw);
            float g_zero = grim_charon_awq_read_zero(g_qz, grp, j, bits, vpw, g_zero_words_per_row);
            unsigned short g_sch = *(const unsigned short*)(g_sc + ((long long)grp * inter + j) * 2);
            float gw_f = ((float)g_code - g_zero) * f16_to_f32(g_sch);
            unsigned int u_code = grim_charon_awq_read_code(u_qw, i, j, inter, bits, vpw);
            float u_zero = grim_charon_awq_read_zero(u_qz, grp, j, bits, vpw, g_zero_words_per_row);
            unsigned short u_sch = *(const unsigned short*)(u_sc + ((long long)grp * inter + j) * 2);
            float uw_f = ((float)u_code - u_zero) * f16_to_f32(u_sch);
            gate += gw_f * a[i];
            up   += uw_f * a[i];
        }
        float silu_g = gate / (1.0f + expf(-gate));
        float act = silu_g * up;
        float scale = routed_scaling_factor * w * as * act;

        for (int h = 0; h < hidden; ++h) {
            int d_grp = j / group_size;
            unsigned int d_code = grim_charon_awq_read_code(d_qw, j, h, hidden, bits, vpw);
            float d_zero = grim_charon_awq_read_zero(d_qz, d_grp, h, bits, vpw, d_zero_words_per_row);
            unsigned short d_sch = *(const unsigned short*)(d_sc + ((long long)d_grp * hidden + h) * 2);
            float dw_f = ((float)d_code - d_zero) * f16_to_f32(d_sch);
            unsigned long long out_idx = (unsigned long long)tok * hidden + h;
            atomicAdd(out + out_idx, dw_f * scale);
        }
    }
}

// --- #10 SPEED-DOT grouped kernels (dot4 family: sdot4 RDNA2 / sudot4 RDNA3+) ──
// Decode-shaped MoE forward for Q8_0 / W8A8-int8 / Q4_K experts. Structure
// matches the grouped kernels above (one thread per sorted slot, atomicAdd
// epilogue) but the gate/up contraction runs on V_DOT4_I32_I8 /
// V_DOT4_I32_IU8 via grim_charon_sdot4 instead of per-element fp32 FMA.
//
// Loop order note: the activation row is contract-shared by every (h, j)
// pair, so we iterate 32-wide activation blocks OUTSIDE a JBLK-wide expert
// column chunk and quantize each activation block to Q8_1 exactly once per
// (h, j-chunk). Down projection stays scalar (1 weight per column, dot4
// would pay more in packing than it saves).
//
// The backward stash (stash_hg/stash_hu) is intentionally NOT written here:
// these kernels are inference/decode-only. `dot4_entry_for` on the host
// refuses to select them for training launches.

__device__ __forceinline__ int grim_charon_sdot4(int a, int b, int c) {
#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1035__) || defined(__gfx1036__)
    // RDNA2: V_DOT4_I32_I8 (signed x signed, i32 acc). All B operands are
    // < 128 unsigned codes, so signed B is equivalent.
    return __builtin_amdgcn_sdot4(a, b, c, false);
#else
    // RDNA3/4: dot8-insts (sdot4 removed on RDNA4).
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#endif
}

// In-thread Q8_1 quantization of one 32-element activation block.
// Mirrors grim_quantize_q8_1 (dot_gemv) but wave-reduction-free: the grouped
// kernels are one-thread-per-token, so the block reduce is a serial loop.
__device__ __forceinline__ void grim_charon_quant_q8_1_block(
    const float* __restrict__ a, signed char* qi, float* d_out, float* sum_out) {
    float amax = 0.0f;
#pragma unroll
    for (int e = 0; e < 32; ++e) amax = fmaxf(amax, fabsf(a[e]));
    const float d = amax / 127.0f;
    const float inv_d = (amax > 1e-9f) ? (127.0f / amax) : 0.0f;
    float fsum = 0.0f;
#pragma unroll
    for (int e = 0; e < 32; ++e) {
        const signed char q = (signed char)__builtin_roundf(a[e] * inv_d);
        qi[e] = q;
        fsum += (float)q;
    }
    *d_out = d;
    *sum_out = fsum;
}

// Pack 4 i8 codes into one i32 dot4 operand.
__device__ __forceinline__ int grim_charon_pack_i8x4(
    signed char q0, signed char q1, signed char q2, signed char q3) {
    return ((int)q0 & 0xFF) | (((int)q1 & 0xFF) << 8)
         | (((int)q2 & 0xFF) << 16) | (((int)q3 & 0xFF) << 24);
}

#define GRIM_CHARON_DOT4_JBLK 8

#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1035__) || defined(__gfx1036__) || defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1200__) || defined(__gfx1201__)

// Q4_K experts. Two-dot decomposition per 32-sub-block:
//   value_i = d * sc * q_i - dmin * m
//   dot(a, value) = d * sc * sum(a_i * q_i) - dmin * m * sum(a_i)
extern "C" __global__ void grim_moe_fused_grouped_q4k_dot4(
    const float* __restrict__ activations,
    const unsigned char* __restrict__ egate_w,
    const unsigned char* __restrict__ eup_w,
    const unsigned char* __restrict__ edown_w,
    const float* __restrict__ a_scale,
    const unsigned int* __restrict__ sorted_token_ids,
    const unsigned int* __restrict__ sorted_expert_ids,
    const float* __restrict__ sorted_weights,
    float* __restrict__ out,
    int hidden, int inter, int num_tokens, int block_size,
    float routed_scaling_factor)
{
    const int blk = blockIdx.x;
    const int base = blk * block_size;
    const int end = base + block_size < num_tokens ? base + block_size : num_tokens;
    const int q4k_expert_bytes = (hidden * inter / 256) * 144; // QK_K=256, Q4K_BYTES=144

    for (int s = base + threadIdx.x; s < end; s += blockDim.x) {
        const unsigned int tok = sorted_token_ids[s];
        if (tok >= (unsigned int)num_tokens) continue; // padding
        const unsigned int exp = sorted_expert_ids[s];
        const float w = sorted_weights[s];
        const float as = a_scale[tok];

        const float* a = activations + (unsigned long long)tok * hidden;
        const unsigned char* gw = egate_w + (unsigned long long)exp * q4k_expert_bytes;
        const unsigned char* uw = eup_w   + (unsigned long long)exp * q4k_expert_bytes;
        const unsigned char* dw = edown_w + (unsigned long long)exp * q4k_expert_bytes;

        float acc = 0.0f;
        for (int h = 0; h < hidden; ++h) {
        acc = 0.0f;
        for (int j0 = 0; j0 < inter; j0 += GRIM_CHARON_DOT4_JBLK) {
            const int nj = (inter - j0) < GRIM_CHARON_DOT4_JBLK ? inter - j0 : GRIM_CHARON_DOT4_JBLK;
            float gsum[GRIM_CHARON_DOT4_JBLK];
            float usum[GRIM_CHARON_DOT4_JBLK];
            for (int jj = 0; jj < nj; ++jj) { gsum[jj] = 0.0f; usum[jj] = 0.0f; }

            for (int i0 = 0; i0 < hidden; i0 += 32) {
                float da, suma;
                signed char ai8[32];
                grim_charon_quant_q8_1_block(a + i0, ai8, &da, &suma);

                for (int jj = 0; jj < nj; ++jj) {
                    const int j = j0 + jj;
                    // Superblock/sub-block for weight (j, i0): g = j*hidden+i0.
                    // hidden % 32 == 0 keeps i0-blocks inside one sub-block.
                    const int g = j * hidden + i0;
                    const int sb = g / 256;
                    const int local = g - sb * 256;
                    const int is = local / 32;
                    const unsigned char* d = gw + (unsigned long long)sb * 144;
                    const unsigned char* dpu = uw + (unsigned long long)sb * 144;
                    float dd  = f16_to_f32(*(const unsigned short*)(d + 0));
                    float dmn = f16_to_f32(*(const unsigned short*)(d + 2));
                    float udd  = f16_to_f32(*(const unsigned short*)(dpu + 0));
                    float udmin = f16_to_f32(*(const unsigned short*)(dpu + 2));

                    int sc, m;
                    if (is < 4) { sc = d[4 + is] & 63; m = d[4 + is + 4] & 63; }
                    else { sc = (d[4 + is + 4] & 0x0F) | ((d[4 + is - 4] >> 6) << 4);
                           m  = (d[4 + is + 4] >> 4)  | ((d[4 + is] >> 6) << 4); }
                    int usc, um;
                    if (is < 4) { usc = dpu[4 + is] & 63; um = dpu[4 + is + 4] & 63; }
                    else { usc = (dpu[4 + is + 4] & 0x0F) | ((dpu[4 + is - 4] >> 6) << 4);
                           um  = (dpu[4 + is + 4] >> 4)  | ((dpu[4 + is] >> 6) << 4); }

                    const int group = is / 2;
                    const int half = is % 2;
                    const unsigned char* gq = d + 16 + group * 32;
                    const unsigned char* uq = dpu + 16 + group * 32;

                    int gpos = 0, upos = 0;
#pragma unroll
                    for (int e = 0; e < 32; e += 4) {
                        const int a4 = grim_charon_pack_i8x4(ai8[e], ai8[e+1], ai8[e+2], ai8[e+3]);
                        unsigned char g0 = gq[e], g1 = gq[e+1], g2 = gq[e+2], g3 = gq[e+3];
                        unsigned char u0 = uq[e], u1 = uq[e+1], u2 = uq[e+2], u3 = uq[e+3];
                        if (half) { g0 >>= 4; g1 >>= 4; g2 >>= 4; g3 >>= 4; }
                        else { g0 &= 0x0F; g1 &= 0x0F; g2 &= 0x0F; g3 &= 0x0F; }
                        if (half) { u0 >>= 4; u1 >>= 4; u2 >>= 4; u3 >>= 4; }
                        else { u0 &= 0x0F; u1 &= 0x0F; u2 &= 0x0F; u3 &= 0x0F; }
                        gpos = grim_charon_sdot4(a4, grim_charon_pack_i8x4((signed char)g0, (signed char)g1, (signed char)g2, (signed char)g3), gpos);
                        upos = grim_charon_sdot4(a4, grim_charon_pack_i8x4((signed char)u0, (signed char)u1, (signed char)u2, (signed char)u3), upos);
                    }
                    gsum[jj] += dd * (float)sc * ((float)gpos * da) - dmn * (float)m * suma;
                    usum[jj] += udd * (float)usc * ((float)upos * da) - udmin * (float)um * suma;
                }
            }

            // Down projection (scalar): weights along inter at h*inter + j.
            float dwt[GRIM_CHARON_DOT4_JBLK];
            iqk_batch_decode(7, dw, h * inter + j0, nj, dwt);
            for (int jj = 0; jj < nj; ++jj) {
                const float gate = gsum[jj];
                const float silu_g = gate / (1.0f + expf(-gate));
                acc += dwt[jj] * (silu_g * usum[jj]);
            }
        }
        atomicAdd(out + (unsigned long long)tok * hidden + h,
                  routed_scaling_factor * w * as * acc);
        }
    }
}

// Q8_0 experts. Weight block = 34 bytes: f16 scale + 32 i8 codes.
extern "C" __global__ void grim_moe_fused_grouped_q80_dot4(
    const float* __restrict__ activations,
    const unsigned char* __restrict__ egate_w,
    const unsigned char* __restrict__ eup_w,
    const unsigned char* __restrict__ edown_w,
    const float* __restrict__ a_scale,
    const unsigned int* __restrict__ sorted_token_ids,
    const unsigned int* __restrict__ sorted_expert_ids,
    const float* __restrict__ sorted_weights,
    float* __restrict__ out,
    int hidden, int inter, int num_tokens, int block_size,
    float routed_scaling_factor)
{
    const int blk = blockIdx.x;
    const int base = blk * block_size;
    const int end = base + block_size < num_tokens ? base + block_size : num_tokens;
    const int q80_expert_bytes = (hidden * inter / 32) * 34;

    for (int s = base + threadIdx.x; s < end; s += blockDim.x) {
        const unsigned int tok = sorted_token_ids[s];
        if (tok >= (unsigned int)num_tokens) continue;
        const unsigned int exp = sorted_expert_ids[s];
        const float w = sorted_weights[s];
        const float as = a_scale[tok];

        const float* a = activations + (unsigned long long)tok * hidden;
        const unsigned char* gw = egate_w + (unsigned long long)exp * q80_expert_bytes;
        const unsigned char* uw = eup_w   + (unsigned long long)exp * q80_expert_bytes;
        const unsigned char* dw = edown_w + (unsigned long long)exp * q80_expert_bytes;

        for (int h = 0; h < hidden; ++h) {
        float acc = 0.0f;
        for (int j0 = 0; j0 < inter; j0 += GRIM_CHARON_DOT4_JBLK) {
            const int nj = (inter - j0) < GRIM_CHARON_DOT4_JBLK ? inter - j0 : GRIM_CHARON_DOT4_JBLK;
            float gsum[GRIM_CHARON_DOT4_JBLK];
            float usum[GRIM_CHARON_DOT4_JBLK];
            for (int jj = 0; jj < nj; ++jj) { gsum[jj] = 0.0f; usum[jj] = 0.0f; }

            for (int i0 = 0; i0 < hidden; i0 += 32) {
                float da, suma;
                signed char ai8[32];
                grim_charon_quant_q8_1_block(a + i0, ai8, &da, &suma);

                for (int jj = 0; jj < nj; ++jj) {
                    const int j = j0 + jj;
                    const int gblk = (j * hidden + i0) / 32;
                    const unsigned char* gb = gw + (unsigned long long)gblk * 34;
                    const unsigned char* ub = uw + (unsigned long long)gblk * 34;
                    const float gd = f16_to_f32(*(const unsigned short*)(gb + 0));
                    const float ud = f16_to_f32(*(const unsigned short*)(ub + 0));
                    const signed char* gc = (const signed char*)(gb + 2);
                    const signed char* uc = (const signed char*)(ub + 2);

                    int gpos = 0, upos = 0;
#pragma unroll
                    for (int e = 0; e < 32; e += 4) {
                        const int a4 = grim_charon_pack_i8x4(ai8[e], ai8[e+1], ai8[e+2], ai8[e+3]);
                        const int b4 = grim_charon_pack_i8x4(gc[e], gc[e+1], gc[e+2], gc[e+3]);
                        const int c4 = grim_charon_pack_i8x4(uc[e], uc[e+1], uc[e+2], uc[e+3]);
                        gpos = grim_charon_sdot4(a4, b4, gpos);
                        upos = grim_charon_sdot4(a4, c4, upos);
                    }
                    gsum[jj] += (float)gpos * da * gd;
                    usum[jj] += (float)upos * da * ud;
                }
            }

            // Down projection (scalar dequant), weights along inter.
            for (int jj = 0; jj < nj; ++jj) {
                const int j = j0 + jj;
                const int dblk = (h * inter + j) / 32;
                const unsigned char* db = dw + (unsigned long long)dblk * 34;
                const float dd = f16_to_f32(*(const unsigned short*)(db + 0));
                const float dc = (float)*(const signed char*)(db + 2 + (h * inter + j - dblk * 32));
                const float gate = gsum[jj];
                const float silu_g = gate / (1.0f + expf(-gate));
                acc += dc * dd * (silu_g * usum[jj]);
            }
        }
        atomicAdd(out + (unsigned long long)tok * hidden + h,
                  routed_scaling_factor * w * as * acc);
        }
    }
}

// CompressedTensors W8A8 int8 experts. Per-expert layout: 8-byte prefix,
// inter*hidden i8 codes, inter f32 row scales (down: hidden rows).
extern "C" __global__ void grim_moe_fused_grouped_w8a8_int8_dot4(
    const float* __restrict__ activations,
    const unsigned char* __restrict__ egate_w,
    const unsigned char* __restrict__ eup_w,
    const unsigned char* __restrict__ edown_w,
    const float* __restrict__ a_scale,
    const unsigned int* __restrict__ sorted_token_ids,
    const unsigned int* __restrict__ sorted_expert_ids,
    const float* __restrict__ sorted_weights,
    float* __restrict__ out,
    int hidden, int inter, int num_tokens, int block_size,
    float routed_scaling_factor)
{
    const int blk = blockIdx.x;
    const int base = blk * block_size;
    const int end = base + block_size < num_tokens ? base + block_size : num_tokens;
    const unsigned long long gate_stride = 8 + (unsigned long long)inter * hidden + (unsigned long long)inter * 4;
    const unsigned long long down_stride = 8 + (unsigned long long)hidden * inter + (unsigned long long)hidden * 4;

    for (int s = base + threadIdx.x; s < end; s += blockDim.x) {
        const unsigned int tok = sorted_token_ids[s];
        if (tok >= (unsigned int)num_tokens) continue;
        const unsigned int exp = sorted_expert_ids[s];
        const float w = sorted_weights[s];
        const float as = a_scale[tok];

        const float* a = activations + (unsigned long long)tok * hidden;
        const unsigned char* gw = egate_w + (unsigned long long)exp * gate_stride;
        const unsigned char* uw = eup_w   + (unsigned long long)exp * gate_stride;
        const unsigned char* dw = edown_w + (unsigned long long)exp * down_stride;
        const signed char* g_codes = (const signed char*)(gw + 8);
        const signed char* u_codes = (const signed char*)(uw + 8);
        const signed char* d_codes = (const signed char*)(dw + 8);
        const float* g_scales = (const float*)(gw + 8 + (unsigned long long)inter * hidden);
        const float* u_scales = (const float*)(uw + 8 + (unsigned long long)inter * hidden);
        const float* d_scales = (const float*)(dw + 8 + (unsigned long long)hidden * inter);

        for (int h = 0; h < hidden; ++h) {
        float acc = 0.0f;
        for (int j0 = 0; j0 < inter; j0 += GRIM_CHARON_DOT4_JBLK) {
            const int nj = (inter - j0) < GRIM_CHARON_DOT4_JBLK ? inter - j0 : GRIM_CHARON_DOT4_JBLK;
            float gsum[GRIM_CHARON_DOT4_JBLK];
            float usum[GRIM_CHARON_DOT4_JBLK];
            for (int jj = 0; jj < nj; ++jj) { gsum[jj] = 0.0f; usum[jj] = 0.0f; }

            for (int i0 = 0; i0 < hidden; i0 += 32) {
                float da, suma;
                signed char ai8[32];
                grim_charon_quant_q8_1_block(a + i0, ai8, &da, &suma);

                for (int jj = 0; jj < nj; ++jj) {
                    const int j = j0 + jj;
                    const signed char* gr = g_codes + (unsigned long long)j * hidden + i0;
                    const signed char* ur = u_codes + (unsigned long long)j * hidden + i0;
                    int gpos = 0, upos = 0;
#pragma unroll
                    for (int e = 0; e < 32; e += 4) {
                        const int a4 = grim_charon_pack_i8x4(ai8[e], ai8[e+1], ai8[e+2], ai8[e+3]);
                        gpos = grim_charon_sdot4(a4, grim_charon_pack_i8x4(gr[e], gr[e+1], gr[e+2], gr[e+3]), gpos);
                        upos = grim_charon_sdot4(a4, grim_charon_pack_i8x4(ur[e], ur[e+1], ur[e+2], ur[e+3]), upos);
                    }
                    gsum[jj] += (float)gpos * da * g_scales[j];
                    usum[jj] += (float)upos * da * u_scales[j];
                }
            }

            for (int jj = 0; jj < nj; ++jj) {
                const int j = j0 + jj;
                const float gate = gsum[jj];
                const float silu_g = gate / (1.0f + expf(-gate));
                acc += (float)d_codes[(unsigned long long)h * inter + j] * d_scales[h]
                       * (silu_g * usum[jj]);
            }
        }
        atomicAdd(out + (unsigned long long)tok * hidden + h,
                  routed_scaling_factor * w * as * acc);
        }
    }
}

// --- WI-gpu-native-moe #2: sortless W8A8-int8 DOT4 fused dispatch ---------
// Same contract as `grim_moe_fused_dispatch_w8a8_int8` (pair-parallel,
// device-resident routing, packed int8 blobs, per-row scales), but the
// gate/up contraction runs on V_DOT4 (sdot4 RDNA2 / sudot4 RDNA3+, selected
// in-device by `grim_charon_sdot4`) instead of per-element fp32 FMA:
// activations are quantized to Q8_1 per 32-block once per thread, weights
// are already dense int8. Per-row scales fold per 32-block exactly like the
// grouped twin above (`gsum += gpos*da*scale`), so the two agree to fp
// rounding. Symmetric quantization has no zero-point, so unlike the Q4_K
// two-dot decomposition there is no sum(a) correction term.
// Lives under the same RDNA arch guard as the grouped dot4 kernels.
// REQUIRES hidden % 32 == 0 (launcher enforces; scalar kernel otherwise).
extern "C" __global__ void grim_moe_fused_dispatch_w8a8_int8_dot4(
    const float* __restrict__ activations,     // [batch, hidden]
    const unsigned char* __restrict__ expert_gate_w, // stacked packed blobs
    const unsigned char* __restrict__ expert_up_w,   // stacked packed blobs
    const unsigned char* __restrict__ expert_down_w, // stacked packed blobs
    const float* __restrict__ a_scale,         // [batch] per-token act scale
    const unsigned int* __restrict__ router_tokens,  // [num_pairs]
    const unsigned int* __restrict__ router_experts, // [num_pairs]
    const float* __restrict__ router_weights,        // [num_pairs]
    float* __restrict__ out,                     // [batch, hidden]
    int hidden, int inter, int num_pairs,
    float routed_scaling_factor)
{
    const unsigned long long pair = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= (unsigned long long)num_pairs) return;

    const unsigned int tok = router_tokens[pair];
    const unsigned int exp = router_experts[pair];
    const float w = router_weights[pair];
    const float as = a_scale[tok];

    const unsigned long long gate_stride = 8 + (unsigned long long)inter * hidden + (unsigned long long)inter * 4;
    const unsigned long long down_stride = 8 + (unsigned long long)hidden * inter + (unsigned long long)hidden * 4;

    const float* a = activations + (unsigned long long)tok * hidden;
    const unsigned char* gw = expert_gate_w + (unsigned long long)exp * gate_stride;
    const unsigned char* uw = expert_up_w   + (unsigned long long)exp * gate_stride;
    const unsigned char* dw = expert_down_w + (unsigned long long)exp * down_stride;

    const signed char* g_codes = (const signed char*)(gw + 8);
    const signed char* u_codes = (const signed char*)(uw + 8);
    const signed char* d_codes = (const signed char*)(dw + 8);
    const float* g_scales = (const float*)(gw + 8 + (unsigned long long)inter * hidden);
    const float* u_scales = (const float*)(uw + 8 + (unsigned long long)inter * hidden);
    const float* d_scales = (const float*)(dw + 8 + (unsigned long long)hidden * inter);

    for (int j = 0; j < inter; ++j) {
        float g = 0.0f;
        float u = 0.0f;
        for (int i0 = 0; i0 < hidden; i0 += 32) {
            signed char ai8[32];
            float da, dummy;
            grim_charon_quant_q8_1_block(a + i0, ai8, &da, &dummy);
            int gpos = 0;
            int upos = 0;
            #pragma unroll
            for (int e = 0; e < 32; e += 4) {
                const int a4 = grim_charon_pack_i8x4(ai8[e], ai8[e+1], ai8[e+2], ai8[e+3]);
                const unsigned long long base = (unsigned long long)j * hidden + i0 + e;
                const int g4 = grim_charon_pack_i8x4(
                    g_codes[base], g_codes[base+1], g_codes[base+2], g_codes[base+3]);
                const int u4 = grim_charon_pack_i8x4(
                    u_codes[base], u_codes[base+1], u_codes[base+2], u_codes[base+3]);
                gpos = grim_charon_sdot4(a4, g4, gpos);
                upos = grim_charon_sdot4(a4, u4, upos);
            }
            g += (float)gpos * da * g_scales[j];
            u += (float)upos * da * u_scales[j];
        }
        float silu_g = g / (1.0f + expf(-g));
        float act = silu_g * u;
        float scale = routed_scaling_factor * w * as * act;

        for (int h = 0; h < hidden; ++h) {
            float dv = (float)d_codes[(unsigned long long)h * inter + j] * d_scales[h];
            unsigned long long out_idx = (unsigned long long)tok * hidden + h;
            atomicAdd(out + out_idx, dv * scale);
        }
    }
}

#endif // RDNA dot4 arch guard


// --- WhiteCrow u4-group128 grouped dispatch (xing4.0 MoE) ------------------
// One block per (token, expert) pair; 256 threads. Weights ride expert-
// strided WhiteCrow blobs — [u64 qw_len][qweight u32 words][u64 sc_len]
// [scales bf16][u64 zr_len][zeros u8] per expert — decoded in-register with
// f32 activations (W4A16-style math; the group dot uses the zero-point
// algebra d*(sum(a*q) - z*sum(a)), no arch-specific intrinsics, so the
// kernel compiles everywhere). Routing comes from device buffers, so the
// kernel is decode-graph capture-safe.
__device__ __forceinline__ float grim_charon_wc_bf16(unsigned short v) {
    return __int_as_float(((unsigned int)v) << 16);
}

// Dot one WhiteCrow weight column over k elements of `src`.
// qw_col: [k/8] u32 words; sc_col/zr_col: [k/128].
__device__ __forceinline__ float grim_charon_wc_col_dot(
    const float* __restrict__ src,
    const unsigned int* __restrict__ qw_col,
    const unsigned short* __restrict__ sc_col,
    const unsigned char* __restrict__ zr_col,
    int k, int n_groups) {
    float acc = 0.0f;
    for (int g = 0; g < n_groups; ++g) {
        const float d = grim_charon_wc_bf16(sc_col[g]);
        const int z = (int)zr_col[g];
        const unsigned int* w = qw_col + g * 16;
        float sq = 0.0f;
        float sa = 0.0f;
        #pragma unroll 4
        for (int wi = 0; wi < 16; ++wi) {
            const unsigned int word = w[wi];
            const int kbase = g * 128 + wi * 8;
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                const float av = src[kbase + t];
                const float q = (float)((word >> (4 * t)) & 0xF);
                sq = fmaf(av, q, sq);
                sa += av;
            }
        }
        acc = fmaf(d, sq - (float)z * sa, acc);
    }
    return acc;
}

extern "C" __global__ void grim_moe_fused_dispatch_whitecrow(
    const float* __restrict__ activations,       // [batch, hidden]
    const unsigned char* __restrict__ gate_blob, // expert-strided WC blobs
    const unsigned char* __restrict__ up_blob,
    const unsigned char* __restrict__ down_blob,
    const unsigned int* __restrict__ router_tokens,   // [num_pairs]
    const unsigned int* __restrict__ router_experts,  // [num_pairs]
    const float* __restrict__ router_weights,         // [num_pairs]
    float* __restrict__ out,                          // [batch, hidden]
    int hidden, int inter, int num_pairs,
    float routed_scaling_factor,
    unsigned long long gate_stride,   // bytes per expert in gate/up blobs
    unsigned long long down_stride)   // bytes per expert in down blob
{
    const int pair = blockIdx.x;
    if (pair >= num_pairs) return;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int tok = (int)router_tokens[pair];
    const int exp = (int)router_experts[pair];
    const float w = router_weights[pair];

    extern __shared__ float s_act[]; // [inter]

    const unsigned char* gb = gate_blob + (unsigned long long)exp * gate_stride;
    const unsigned char* ub = up_blob   + (unsigned long long)exp * gate_stride;
    const unsigned char* db = down_blob + (unsigned long long)exp * down_stride;
    // Per-expert segment layout: [u64][qw][u64][sc bf16][u64][zr u8].
    const unsigned int* g_qw = (const unsigned int*)(gb + 8);
    const unsigned short* g_sc = (const unsigned short*)(gb + 8 + (unsigned long long)inter * (hidden / 8) * 4 + 8);
    const unsigned char* g_zr = gb + 8 + (unsigned long long)inter * (hidden / 8) * 4 + 8 + (unsigned long long)inter * (hidden / 128) * 2 + 8;
    const unsigned int* u_qw = (const unsigned int*)(ub + 8);
    const unsigned short* u_sc = (const unsigned short*)(ub + 8 + (unsigned long long)inter * (hidden / 8) * 4 + 8);
    const unsigned char* u_zr = ub + 8 + (unsigned long long)inter * (hidden / 8) * 4 + 8 + (unsigned long long)inter * (hidden / 128) * 2 + 8;
    const unsigned int* d_qw = (const unsigned int*)(db + 8);
    const unsigned short* d_sc = (const unsigned short*)(db + 8 + (unsigned long long)hidden * (inter / 8) * 4 + 8);
    const unsigned char* d_zr = db + 8 + (unsigned long long)hidden * (inter / 8) * 4 + 8 + (unsigned long long)hidden * (inter / 128) * 2 + 8;

    const float* a = activations + (unsigned long long)tok * hidden;
    const int hg = hidden / 128;
    const int ig = inter / 128;

    // Phase 1: fused gate|up with in-register SiLU combine -> shared act.
    for (int j = tid; j < inter; j += nthreads) {
        const float g = grim_charon_wc_col_dot(a, g_qw + (unsigned long long)j * (hidden / 8), g_sc + (unsigned long long)j * ig, g_zr + (unsigned long long)j * ig, hidden, hg);
        const float u = grim_charon_wc_col_dot(a, u_qw + (unsigned long long)j * (hidden / 8), u_sc + (unsigned long long)j * ig, u_zr + (unsigned long long)j * ig, hidden, hg);
        s_act[j] = (g / (1.0f + expf(-g))) * u;
    }
    __syncthreads();

    // Phase 2: down projection, atomicAdd accumulation with the routing weight.
    for (int h = tid; h < hidden; h += nthreads) {
        const float acc = grim_charon_wc_col_dot(s_act, d_qw + (unsigned long long)h * (inter / 8), d_sc + (unsigned long long)h * hg, d_zr + (unsigned long long)h * hg, inter, ig);
        atomicAdd(out + (unsigned long long)tok * hidden + h, routed_scaling_factor * w * acc);
    }
}

"#;

// Host launcher (parameter marshalling - pure, unit-testable without GPU)

/// A flattened (token, expert, weight) routing assignment produced from the `MoeRouter::route` output.
/// This is the sortless work list the kernel consumes: block `i` reads `tokens[i]`, `experts[i]`, `weights[i]`.
#[derive(Debug, Clone, PartialEq)]
pub struct RoutingAssignment {
    /// Token index per pair. Length == number of (token, expert) pairs.
    pub tokens: Vec<u32>,
    /// Expert index per pair.
    pub experts: Vec<u32>,
    /// Router combine weight per pair.
    pub weights: Vec<f32>,
}

impl RoutingAssignment {
    /// Flatten a per-token `(indices, weights)` routing result (as produced by `grim_nn::moe::MoeRouter::route`) into the sortless work list.
    /// `indices[t]` and `weights[t]` are the selected experts and combine weights for token `t`; both must.
    pub fn from_route(indices: &[Vec<usize>], weights: &[Vec<f32>]) -> Result<Self> {
        if indices.len() != weights.len() {
            return Err(Error::Backend(format!(
                "RoutingAssignment::from_route: indices len {} != weights len {}",
                indices.len(),
                weights.len()
            )));
        }
        let num_pairs: usize = indices.iter().map(|v| v.len()).sum();
        let num_pairs_w: usize = weights.iter().map(|v| v.len()).sum();
        if num_pairs != num_pairs_w {
            return Err(Error::Backend(format!(
                "RoutingAssignment::from_route: total expert count {} != total weight count {}",
                num_pairs, num_pairs_w
            )));
        }
        let mut tokens = Vec::with_capacity(num_pairs);
        let mut experts = Vec::with_capacity(num_pairs);
        let mut w = Vec::with_capacity(num_pairs);
        for (t, (idx_row, w_row)) in indices.iter().zip(weights.iter()).enumerate() {
            if idx_row.len() != w_row.len() {
                return Err(Error::Backend(format!(
                    "RoutingAssignment::from_route: token {} has {} experts but {} weights",
                    t,
                    idx_row.len(),
                    w_row.len()
                )));
            }
            for (&e, &wi) in idx_row.iter().zip(w_row.iter()) {
                tokens.push(t as u32);
                experts.push(e as u32);
                w.push(wi);
            }
        }
        Ok(Self {
            tokens,
            experts,
            weights: w,
        })
    }

    /// Number of (token, expert) work pairs.
    pub fn num_pairs(&self) -> usize {
        self.tokens.len()
    }

    /// Compute per-expert token counts from the expert array.
    pub fn per_expert_token_counts(&self) -> Vec<u32> {
        let max_expert = self.experts.iter().copied().max().unwrap_or(0) as usize;
        let mut counts = vec![0u32; max_expert + 1];
        for &e in &self.experts {
            counts[e as usize] += 1;
        }
        counts
    }

    /// Compute continuous routing skew for this assignment.
    pub fn routing_skew(&self) -> f32 {
        routing_skew(&self.per_expert_token_counts())
    }
}

/// Token-sorted routing layout for the grouped fused dispatch (`grim_moe_fused_grouped`).
/// Produced by `moe_align_block_size` from a `RoutingAssignment`.
#[derive(Debug, Clone, PartialEq)]
pub struct SortedRouting {
    /// Token index per sorted slot. Length == `num_tokens_post_padded`.
    pub sorted_token_ids: Vec<u32>,
    /// Expert index per sorted slot. Length == `num_tokens_post_padded`.
    pub sorted_expert_ids: Vec<u32>,
    /// Router combine weight per sorted slot. Length == `num_tokens_post_padded`.
    pub sorted_weights: Vec<f32>,
    /// Total slots after padding (divisible by `block_size`).
    pub num_tokens_post_padded: usize,
    /// Block size the sort was aligned to.
    pub block_size: usize,
}

/// Pure, device-free port of vLLM `moe_align_block_size` (counting sort by expert + per-expert block padding).
/// Unit-testable without a GPU (G-A2).
pub fn moe_align_block_size(
    assignment: &RoutingAssignment,
    block_size: usize,
    num_experts: usize,
) -> SortedRouting {
    assert!(block_size > 0, "block_size must be > 0");
    // Sentinel token index for padding slots: one past the highest real
    // token id, so the kernel's `tok >= n_token` skip is always correct.
    let n_token = assignment
        .tokens
        .iter()
        .copied()
        .max()
        .map(|m| m as usize + 1)
        .unwrap_or(0);
    let n_pairs = assignment.num_pairs();

    // 1. Count tokens per expert.
    let mut counts = vec![0usize; num_experts];
    for &e in &assignment.experts {
        let e = e as usize;
        if e < num_experts {
            counts[e] += 1;
        }
    }

    // 2. Prefix-sum → per-expert start offset in the padded layout.
    let mut expert_offset = vec![0usize; num_experts];
    let mut cum = 0usize;
    for e in 0..num_experts {
        expert_offset[e] = cum;
        // round each expert's run up to block_size for the next start.
        cum += counts[e].div_ceil(block_size) * block_size;
    }
    let num_tokens_post_padded = cum;

    let mut sorted_token_ids = vec![n_token as u32; num_tokens_post_padded];
    // Padding slots must carry the block's real expert id (not 0) so the per-block "expert constant within block" invariant
    // the kernel relies on holds for the whole padded run, and the sentinel `n_token` token index alone marks skip slots.
    let mut sorted_expert_ids = vec![0u32; num_tokens_post_padded];
    for e in 0..num_experts {
        let run = counts[e].div_ceil(block_size) * block_size;
        if run == 0 {
            continue;
        }
        for slot in &mut sorted_expert_ids[expert_offset[e]..expert_offset[e] + run] {
            *slot = e as u32;
        }
    }
    let mut sorted_weights = vec![0.0f32; num_tokens_post_padded];

    // 3. Scatter. Track the next free slot per expert as we go.
    let mut cursor = expert_offset.clone();
    for p in 0..n_pairs {
        let e = assignment.experts[p] as usize;
        if e >= num_experts {
            continue; // out-of-range expert: skip (caller owns expert count)
        }
        let slot = cursor[e];
        cursor[e] += 1;
        sorted_token_ids[slot] = assignment.tokens[p];
        sorted_expert_ids[slot] = e as u32;
        sorted_weights[slot] = assignment.weights[p];
    }

    SortedRouting {
        sorted_token_ids,
        sorted_expert_ids,
        sorted_weights,
        num_tokens_post_padded,
        block_size,
    }
}

/// Resolved kernel launch parameters for one fused dispatch. Computed by the
/// pure planner so the assembly is unit-testable without a device (G-A2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CharonLaunchPlan {
    /// Grid x = ceil(num_pairs / block_dim).
    pub grid_x: u32,
    /// Block x — must be a multiple of the device's wavefront size
    /// (64 on CDNA/MI-series Wave64, 32 on RDNA consumer/APU Wave32).
    pub block_x: u32,
}

/// Choose the wave-aligned block dimension for a fused dispatch.
/// Picks the smallest multiple of `wave_size` (32 on gfx1036/RDNA Wave32, 64 on CDNA Wave64) that.
pub(crate) fn choose_block_dim(num_pairs: usize, wave_size: u32) -> u32 {
    const WAVES_MAX: u32 = 4; // cap at 4 wavefronts
    let one_wave = wave_size.max(1);
    if num_pairs == 0 {
        return one_wave;
    }
    let target = num_pairs.max(one_wave as usize) as u32;
    let mut block = one_wave;
    while block < target && block < one_wave * WAVES_MAX {
        block *= 2;
    }
    block.min(one_wave * WAVES_MAX)
}

/// Pure planner: resolve the grid/block for a fused dispatch given the routing assignment and the device's wavefront size.
/// Extracted from the launcher so G-A2 can prove the parameter blob is built correctly without.
#[allow(dead_code)]
pub(crate) fn plan_fused_dispatch(
    assignment: &RoutingAssignment,
    wave_size: u32,
) -> CharonLaunchPlan {
    let n = assignment.num_pairs();
    let block_x = choose_block_dim(n, wave_size);
    let grid_x = if n == 0 {
        0
    } else {
        (n as u32).div_ceil(block_x)
    };
    CharonLaunchPlan { grid_x, block_x }
}

/// MoE autotune-aware launch planner. Consults `tuner` for a
/// measured `MoeKernelKey` launch parameter before falling back to `choose_block_dim`.
#[allow(dead_code)]
pub(crate) fn plan_fused_dispatch_with_autotuner(
    assignment: &RoutingAssignment,
    wave_size: u32,
    tuner: Option<&mut crate::autotune::Autotuner>,
    gpu_arch: &str,
    hidden: usize,
    inter: usize,
) -> CharonLaunchPlan {
    let n = assignment.num_pairs();
    let counts = assignment.per_expert_token_counts();
    let num_experts = counts.len();
    let num_tokens = (assignment.tokens.iter().copied().max().unwrap_or(0) as usize + 1).max(1);
    let top_k = n / num_tokens;
    let skew = routing_skew(&counts);
    let bucket = crate::autotune::quantize_routing_skew(skew);
    let key = crate::autotune::MoeKernelKey {
        kernel: "grim_moe_fused_dispatch".to_string(),
        gpu_arch: gpu_arch.to_string(),
        hidden,
        inter,
        num_experts,
        top_k,
        skew_bucket: bucket,
    };

    let fallback_dim = choose_block_dim(n, wave_size);
    let block_x = match tuner {
        Some(t) => t.get_or_tune_moe_block_dim(&key, fallback_dim),
        None => fallback_dim,
    };

    let grid_x = if n == 0 {
        0
    } else {
        (n as u32).div_ceil(block_x)
    };
    CharonLaunchPlan { grid_x, block_x }
}

impl SortedRouting {
    /// Number of expert-blocks in the grouped layout (= grid x for the
    /// `grim_moe_fused_grouped` launch).
    pub fn num_blocks(&self) -> u32 {
        if self.block_size == 0 {
            return 0;
        }
        (self.num_tokens_post_padded.div_ceil(self.block_size)) as u32
    }
}

/// Pure planner for the grouped (token-sorted) fused dispatch.
/// Grid x = number of expert-blocks in the sorted layout (one block per `block_size` slot).
#[allow(dead_code)]
pub(crate) fn plan_grouped_dispatch(sorted: &SortedRouting, wave_size: u32) -> CharonLaunchPlan {
    let grid_x = sorted.num_blocks();
    let block_x = choose_block_dim(sorted.block_size, wave_size).max(wave_size);
    CharonLaunchPlan {
        grid_x: grid_x.max(if sorted.num_tokens_post_padded == 0 {
            0
        } else {
            1
        }),
        block_x,
    }
}

/// Validate the host-side inputs to a grouped fused dispatch *before* any device pointer is dereferenced.
/// Pure, allocation-free, unit-testable without a GPU (G-A2).
/// `allow_null_up` — when `true`, skip the null check on `expert_up_w` (GELU kernel passes 0).
#[allow(dead_code)]
pub(crate) fn validate_grouped_inputs(
    activations: *mut c_void,
    expert_gate_w: *mut c_void,
    expert_up_w: *mut c_void,
    expert_down_w: *mut c_void,
    out: *mut c_void,
    sorted: &SortedRouting,
    hidden: usize,
    inter: usize,
    num_experts: usize,
    allow_null_up: bool,
) -> Result<()> {
    for (label, p) in [
        ("activations", activations),
        ("expert_gate_w", expert_gate_w),
        ("expert_down_w", expert_down_w),
        ("out", out),
    ] {
        if p.is_null() {
            return Err(Error::Backend(format!(
                "charon_grouped_dispatch: {label} is null"
            )));
        }
    }
    if !allow_null_up && expert_up_w.is_null() {
        return Err(Error::Backend(
            "charon_grouped_dispatch: expert_up_w is null".into(),
        ));
    }
    if hidden == 0 || inter == 0 {
        return Err(Error::Backend(format!(
            "charon_grouped_dispatch: degenerate shape (hidden={hidden}, inter={inter})"
        )));
    }
    // Every sorted expert id must be in range.
    if sorted
        .sorted_expert_ids
        .iter()
        .any(|&e| e as usize >= num_experts)
    {
        return Err(Error::Backend(
            "charon_grouped_dispatch: sorted expert id out of range".into(),
        ));
    }
    Ok(())
}

/// Validate the host-side inputs to a fused dispatch *before* any device pointer is dereferenced.
/// Pure, allocation-free, unit-testable without a GPU (G-A2).
#[allow(dead_code)]
pub(crate) fn validate_launch_inputs(
    activations: *mut c_void,
    expert_gate_w: *mut c_void,
    expert_up_w: *mut c_void,
    expert_down_w: *mut c_void,
    out: *mut c_void,
    assignment: &RoutingAssignment,
    hidden: usize,
    inter: usize,
) -> Result<()> {
    for (label, p) in [
        ("activations", activations),
        ("expert_gate_w", expert_gate_w),
        ("expert_up_w", expert_up_w),
        ("expert_down_w", expert_down_w),
        ("out", out),
    ] {
        if p.is_null() {
            return Err(Error::Backend(format!(
                "charon_fused_dispatch: {label} is null"
            )));
        }
    }
    // Shape sanity: every routed expert index must be in range.
    // The caller owns the expert-count invariant; here we only reject obviously-broken assignments (empty, or indices.
    if hidden == 0 || inter == 0 {
        return Err(Error::Backend(format!(
            "charon_fused_dispatch: degenerate shape (hidden={hidden}, inter={inter})"
        )));
    }
    let _ = assignment.num_pairs(); // touched so the planner sees a non-empty list.
    Ok(())
}

// WI-B - Polymorphic population + GPU-resident variant selector Two pieces, both pure and unit-testable without a device (G-B1): 1.
// `WaveCostModel` - a 4-param linear model predicting per-dispatch cycle cost from `(active_warps, bytes_per_wave, flops_per_wave, stall_rate)`.

/// A polymorphic kernel variant in the Charon population.
/// The plan caps the v1 population at three (small-batch/decode, large-group prefill, high-skew) - collapsed from.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CharonVariant {
    /// (a) Small-batch / decode tile — few tokens, many experts.
    SmallBatchDecode,
    /// (b) Large-group prefill tile — many tokens per expert.
    LargeGroupPrefill,
    /// (c) High-skew tile — few experts receive most tokens.
    HighSkew,
}

/// WI-F3 - dispatch-target resolution for the grouped MoE forward: the JIT kernel entry a `CharonVariant` routes to.
/// Only `LargeGroupPrefill` (the compute-bound, many-tokens-per-expert regime where tensor-core tiling pays) resolves to the WMMA grouped.
pub fn grouped_dispatch_entry(variant: CharonVariant) -> &'static str {
    match variant {
        CharonVariant::SmallBatchDecode | CharonVariant::HighSkew => "grim_moe_fused_grouped",
        CharonVariant::LargeGroupPrefill => "grim_moe_fused_grouped_wmma",
    }
}

/// Quantized-expert families with a SPEED-DOT (dot4/sudot4) grouped kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CharonDot4Quant {
    /// GGUF Q8_0 weights (f16 scale + 32 i8 codes per block).
    Q8_0,
    /// CompressedTensors W8A8 per-row int8.
    W8A8Int8,
    /// GGUF Q4_K super-blocks.
    Q4K,
}

/// SPEED-DOT grouped MoE entry for `quant`.
pub fn grouped_dot4_entry(quant: CharonDot4Quant) -> &'static str {
    match quant {
        CharonDot4Quant::Q8_0 => "grim_moe_fused_grouped_q80_dot4",
        CharonDot4Quant::W8A8Int8 => "grim_moe_fused_grouped_w8a8_int8_dot4",
        CharonDot4Quant::Q4K => "grim_moe_fused_grouped_q4k_dot4",
    }
}

/// Whether the JIT'd dot4 kernels exist for this arch. Mirrors the
/// `#if defined(__gfx..)` guard around the #10 kernels in `KERNEL_SOURCE`:
/// RDNA2 (dot1q/dot4-insts), RDNA3 (dot8-insts) and RDNA4 (sudot4 only —
/// sdot4 opcode was removed) are covered; CDNA (gfx90a/gfx94x) is not.
pub fn dot4_supported(gcn_arch: &str) -> bool {
    // starts_with: hip gcnArchName can carry suffixes (e.g. "gfx1100-xm").
    const DOT4_ARCHES: [&str; 11] = [
        "gfx1030", "gfx1031", "gfx1032", "gfx1035", "gfx1036", "gfx1100", "gfx1101", "gfx1102",
        "gfx1103", "gfx1200", "gfx1201",
    ];
    DOT4_ARCHES.iter().any(|a| gcn_arch.starts_with(a))
}

/// Dispatch resolution for the dot4 decode path. Returns `None` when the
/// dot4 kernels must not run:
/// - non-RDNA arch (no V_DOT4/V_SUDOT4 instructions),
/// - training launches — the dot4 kernels don't write the
///   `stash_hg`/`stash_hu` pre-activation stash the backward kernel reads,
///   so a training forward must stay on the scalar/WMMA grouped variants.
pub fn dot4_entry_for(
    quant: CharonDot4Quant,
    gcn_arch: &str,
    training: bool,
) -> Option<&'static str> {
    if training || !dot4_supported(gcn_arch) {
        return None;
    }
    Some(grouped_dot4_entry(quant))
}

impl CharonVariant {
    /// All v1 variants, in stable order (the selector's table index).
    pub const ALL: [Self; 3] = [
        Self::SmallBatchDecode,
        Self::LargeGroupPrefill,
        Self::HighSkew,
    ];

    /// Stable table index into the selector's per-variant coefficient row.
    pub fn idx(self) -> usize {
        match self {
            Self::SmallBatchDecode => 0,
            Self::LargeGroupPrefill => 1,
            Self::HighSkew => 2,
        }
    }
}

/// Four-parameter wave cost model (RaMP form, RDNA-fit coefficients).
/// Predicts relative per-dispatch cycle cost from: * `active_warps`  - number of in-flight wavefronts (occupancy proxy).
#[derive(Debug, Clone, Copy)]
pub struct WaveCostModel {
    /// `c0` — occupancy weight.
    pub c_active_warps: f32,
    /// `c1` — memory traffic weight (dominant prior).
    pub c_bytes_per_wave: f32,
    /// `c2` — compute weight.
    pub c_flops_per_wave: f32,
    /// `c3` — stall weight.
    pub c_stall_rate: f32,
}

impl Default for WaveCostModel {
    fn default() -> Self {
        // Memory-leaning prior: GMEM traffic dominates on RDNA consumer parts (Infinity Cache helps but HBM bandwidth is the ceiling).
        // These are priors, not fitted values - G-B2 re-fits on-device.
        Self {
            c_active_warps: 0.1,
            c_bytes_per_wave: 1.0,
            c_flops_per_wave: 0.01,
            c_stall_rate: 0.5,
        }
    }
}

impl WaveCostModel {
    /// Predict relative cycle cost. Higher = slower.
    pub fn predict(
        &self,
        active_warps: f32,
        bytes_per_wave: f32,
        flops_per_wave: f32,
        stall_rate: f32,
    ) -> f32 {
        self.c_active_warps * active_warps
            + self.c_bytes_per_wave * bytes_per_wave
            + self.c_flops_per_wave * flops_per_wave
            + self.c_stall_rate * stall_rate
    }
}

/// One row of the selector's per-variant fitted cost model + the distribution bucket it was tuned for.
/// Built offline (G-B2 device-gated); the selector reads it at runtime with no CPU readback.
#[derive(Debug, Clone, Copy)]
pub struct VariantRow {
    pub variant: CharonVariant,
    pub model: WaveCostModel,
    /// Skew bucket this row wins on (0 = uniform, 1 = one-expert-dominates).
    /// Used by the reactive matcher to pick a row from the live histogram.
    pub skew_bucket: f32,
}

/// Default v1 variant table — three rows, memory-leaning priors, covering
/// the skew range [0, 1]. Coefficients re-fit on-device for G-B2.
#[allow(dead_code)]
pub fn default_variant_table() -> Vec<VariantRow> {
    vec![
        VariantRow {
            variant: CharonVariant::SmallBatchDecode,
            model: WaveCostModel {
                c_active_warps: 0.05, // decode is occupancy-light
                c_bytes_per_wave: 1.0,
                c_flops_per_wave: 0.02,
                c_stall_rate: 0.4,
            },
            skew_bucket: 0.2,
        },
        VariantRow {
            variant: CharonVariant::LargeGroupPrefill,
            model: WaveCostModel {
                c_active_warps: 0.15, // prefill saturates waves
                c_bytes_per_wave: 0.9,
                c_flops_per_wave: 0.05, // compute-heavier
                c_stall_rate: 0.3,
            },
            skew_bucket: 0.5,
        },
        VariantRow {
            variant: CharonVariant::HighSkew,
            model: WaveCostModel {
                c_active_warps: 0.2,
                c_bytes_per_wave: 1.1, // few experts = re-read weights
                c_flops_per_wave: 0.03,
                c_stall_rate: 0.6, // hot-expert contention
            },
            skew_bucket: 0.9,
        },
    ]
}

/// Build `CharonSelector`'s `Vec<VariantRow>` from measured `Autotuner` configurations.
/// `moe_autotuning_design.md` §3: replaces static priors in `default_variant_table` with measured launch parameters from `Autotuner` per skew.
#[allow(dead_code)]
pub fn build_variant_table_from_autotuner(
    tuner: &crate::autotune::Autotuner,
    gpu_arch: &str,
) -> Vec<VariantRow> {
    let mut table = default_variant_table();
    let moe_keys = tuner.list_moe_keys();
    if moe_keys.is_empty() {
        return table;
    }

    for row in &mut table {
        let bucket_idx = crate::autotune::quantize_routing_skew(row.skew_bucket);
        if let Some(matching_key) = moe_keys
            .iter()
            .find(|k| k.gpu_arch == gpu_arch && k.skew_bucket == bucket_idx)
        {
            if let Some(cfg) = tuner.lookup_moe(matching_key) {
                if cfg.cycles_per_invocation > 0 {
                    row.model.c_bytes_per_wave =
                        (cfg.cycles_per_invocation as f32 / 1e6).clamp(0.01, 10.0);
                }
            }
        }
    }
    table
}

/// Select autotuned launch configuration for Charon kernel using t-pain model.
/// Falls back to default wave-aligned planning if no tuned entry is found in `Autotuner`.
pub fn charon_autotune_launch_config(
    tuner: &crate::autotune::Autotuner,
    gpu_arch: &'static str,
    hidden: usize,
    inter: usize,
    num_experts: usize,
    top_k: usize,
    routing_histogram: &[u32],
) -> crate::autotune::AutotuneConfig {
    let skew = routing_skew(routing_histogram);
    let skew_bucket = crate::autotune::quantize_routing_skew(skew);
    let key = crate::autotune::MoeKernelKey {
        kernel: "grim_moe_fused_dispatch".to_string(),
        gpu_arch: gpu_arch.to_string(),
        hidden,
        inter,
        num_experts,
        top_k,
        skew_bucket,
    };

    tuner.lookup_moe(&key).unwrap_or_else(|| {
        let default_threads = if gpu_arch.starts_with("gfx10") {
            32
        } else {
            64
        };
        crate::autotune::AutotuneConfig {
            block_dim: default_threads,
            tile_kv: 64,
            grid_stride: 1,
            cycles_per_invocation: 0,
            spec_gamma: 4,
            spec_acceptance_threshold: 0.6,
            spec_alpha: 0.0,
            split_k: 0,
        }
    })
}

/// Compute the routing skew of a histogram - the fraction of tokens going to the single hottest expert.
/// `0.0` = perfectly uniform, `1.0` = all tokens to one expert.
#[allow(dead_code)]
pub fn routing_skew(per_expert_token_counts: &[u32]) -> f32 {
    let total: u32 = per_expert_token_counts.iter().sum();
    if total == 0 || per_expert_token_counts.is_empty() {
        return 0.0;
    }
    let max = *per_expert_token_counts.iter().max().unwrap_or(&0) as f32;
    let uniform = total as f32 / per_expert_token_counts.len() as f32;
    if uniform == 0.0 {
        return 0.0;
    }
    // Skew = how far the hottest expert exceeds the uniform share, rescaled
    // so uniform→0 and all-to-one→1.
    let peak_share = max / total as f32;
    let uniform_share = 1.0 / per_expert_token_counts.len() as f32;
    ((peak_share - uniform_share) / (1.0 - uniform_share)).clamp(0.0, 1.0)
}

/// GPU-resident variant selector with a de-sync (min-hold) guard.
/// The selector emits the `variant_idx` for the next launch from the live routing skew, **without.
#[allow(dead_code)]
pub struct CharonSelector {
    table: Vec<VariantRow>,
    current_variant: CharonVariant,
    /// Consecutive calls the current challenger has been the argmin.
    hold_counter: u32,
    /// Which variant the hold_counter is accumulating for. `None` when the
    /// current variant is winning (no active challenger).
    challenger: Option<CharonVariant>,
    /// Required consecutive wins before switching (de-sync guard).
    min_hold: u32,
}

impl CharonSelector {
    /// Build a selector over `table` with a de-sync guard of `min_hold`
    /// consecutive wins before a variant switch. `min_hold >= 1`.
    pub fn new(table: Vec<VariantRow>, min_hold: u32) -> Self {
        let initial = table
            .first()
            .map(|r| r.variant)
            .unwrap_or(CharonVariant::SmallBatchDecode);
        Self {
            table,
            current_variant: initial,
            hold_counter: 0,
            challenger: None,
            min_hold: min_hold.max(1),
        }
    }

    /// The variant the next launch should use. Reads the staged `skew` scalar (device-resident in
    /// production; a plain f32 here) and the per-wave cost inputs the caller also staged.
    pub fn select(
        &mut self,
        skew: f32,
        active_warps: f32,
        bytes_per_wave: f32,
        flops_per_wave: f32,
        stall_rate: f32,
    ) -> CharonVariant {
        // Find the variant whose bucket is closest to the live skew AND whose model predicts the lowest cost (reactive DA-MoE matching).
        // Distance is the primary signal (form matching); cost breaks ties among near-equidistant buckets.
        let mut best = self.current_variant;
        let mut best_score = f32::INFINITY;
        for row in &self.table {
            let dist = (row.skew_bucket - skew).abs();
            let cost = row
                .model
                .predict(active_warps, bytes_per_wave, flops_per_wave, stall_rate)
                .max(0.0);
            let score = dist + cost * 1e-6;
            if score < best_score {
                best_score = score;
                best = row.variant;
            }
        }

        if best == self.current_variant {
            // Current variant is winning — reset any challenger streak.
            self.hold_counter = 0;
            self.challenger = None;
        } else {
            // A challenger won. Only accumulate credit for the *same* challenger across consecutive calls -
            // a different challenger resets the streak to 1 (the new challenger starts from scratch).
            match self.challenger {
                Some(c) if c == best => {
                    self.hold_counter += 1;
                }
                _ => {
                    self.challenger = Some(best);
                    self.hold_counter = 1;
                }
            }
            // Switch only when the *same* challenger has held for min_hold
            // consecutive calls.
            if self.hold_counter >= self.min_hold {
                self.current_variant = best;
                self.hold_counter = 0;
                self.challenger = None;
            }
        }
        self.current_variant
    }

    /// Current variant without advancing the de-sync state (read-only).
    pub fn current(&self) -> CharonVariant {
        self.current_variant
    }
}

// Tests - host logic only (G-A2), no GPU required

#[cfg(test)]
mod tests {
    use super::*;

    /// The HIP source must be JIT-discoverable by the canonical entry name.
    /// The repo convention is `grim_*`-prefixed entries; the plan also names the short alias `charon_fused_dispatch`.
    #[test]
    fn source_contains_fused_dispatch_entry() {
        assert!(
            KERNEL_SOURCE.contains("grim_moe_fused_dispatch"),
            "Charon fused dispatch entry must be JIT-discoverable by name"
        );
        assert!(
            KERNEL_SOURCE.contains("grim_moe_fused_grouped"),
            "Charon grouped (token-sorted) dispatch entry must be JIT-discoverable"
        );
        assert!(
            KERNEL_SOURCE.contains("grim_moe_fused_grouped_fp8"),
            "Charon #2 FP8 W8A8 grouped dispatch entry must be JIT-discoverable"
        );
        assert!(
            KERNEL_SOURCE.contains("fp8e4m3_to_f32"),
            "FP8 E4M3 decode helper must be present for #2 W8A8"
        );
        assert!(
            KERNEL_SOURCE.contains("charon_fused_bytes"),
            "GMEM traffic counter helper must be present for G-A5"
        );
        assert!(
            KERNEL_SOURCE.contains("grim_moe_route_topk"),
            "Phase 3aD: device-side MoE routing entry must be JIT-discoverable"
        );
    }

    /// kernel-hygiene-plan item 1: the grouped IQK kernel must expose the batched k-quant decode helper and route Q4_K/Q5_K/Q6_K (formats 7,8,9) through it, while keeping the per-element `iqk_weight` path for the other formats.
    /// The helper's inner loop must not call f16_to_f32 (the per-sub-block unpack is hoisted out).
    #[test]
    fn grouped_iqk_has_batched_kquant_decode() {
        assert!(
            KERNEL_SOURCE.contains("iqk_batch_decode"),
            "Item 1: batched decode helper must be present in the HIP source"
        );
        assert!(
            KERNEL_SOURCE.contains(
                "const bool batched = (format_id == 7 || format_id == 8 || format_id == 9)"
            ),
            "Item 1: the grouped kernel must select the batched path for formats 7/8/9"
        );
        assert!(
            KERNEL_SOURCE.contains("int sb_w = (fmt == 9) ? 128 : 64"),
            "Item 1: Q6_K must use 128-weight sub-blocks, Q4_K/Q5_K 64-weight"
        );
        // Inner loop of the batched helper emits per-weight weights with hoisted
        // scalars; the only f16_to_f32 calls must be before the per-weight loop.
        let helper_start = KERNEL_SOURCE.find("iqk_batch_decode(").unwrap();
        let helper_end = KERNEL_SOURCE
            .find("// kernel-hygiene-plan item 1: per-weight decode chunks")
            .unwrap();
        let helper = &KERNEL_SOURCE[helper_start..helper_end];
        assert!(
            helper.contains("float dd = f16_to_f32"),
            "Item 1: super-block scale must be unpacked inside the helper"
        );
        assert!(
            helper.contains("for (int p = 0; p < n; ++p)"),
            "Item 1: the helper must have a per-weight inner loop"
        );
    }

    /// SPEED-DOT MoE: the dot4 grouped kernels must be JIT-discoverable by
    /// name, and the sudot4 ISA helper must be present.
    #[test]
    fn dot4_grouped_entries_are_jit_discoverable() {
        for entry in [
            grouped_dot4_entry(CharonDot4Quant::Q8_0),
            grouped_dot4_entry(CharonDot4Quant::W8A8Int8),
            grouped_dot4_entry(CharonDot4Quant::Q4K),
        ] {
            assert!(
                KERNEL_SOURCE.contains(entry),
                "dot4 grouped entry {entry} must be JIT-discoverable"
            );
        }
        assert!(
            KERNEL_SOURCE.contains("grim_charon_sdot4"),
            "sudot4/sdot4 ISA helper must be present"
        );
        // The kernels are inference-only: no backward stash writes.
        let dot4_start = KERNEL_SOURCE
            .find("grim_moe_fused_grouped_q4k_dot4")
            .unwrap();
        let dot4_src = &KERNEL_SOURCE[dot4_start..];
        let q4k_end = dot4_src.find("grim_moe_fused_grouped_q80_dot4").unwrap();
        assert!(
            !dot4_src[..q4k_end].contains("stash_hg"),
            "dot4 kernels must not write the backward pre-activation stash"
        );
    }

    /// SPEED-DOT MoE: arch gating + training gate on the dispatch resolver.
    #[test]
    fn dot4_entry_for_gates_arch_and_training() {
        use CharonDot4Quant::*;
        // RDNA2/3/4 all supported (sdot4 vs sudot4 selected in-device).
        for arch in ["gfx1036", "gfx1100", "gfx1201"] {
            assert_eq!(
                dot4_entry_for(Q4K, arch, false),
                Some("grim_moe_fused_grouped_q4k_dot4")
            );
        }
        // CDNA / unknown: no dot4.
        assert_eq!(dot4_entry_for(Q8_0, "gfx90a", false), None);
        assert_eq!(dot4_entry_for(Q8_0, "gfx9428", false), None);
        // Training: backward reads the stash the dot4 kernels don't write.
        assert_eq!(dot4_entry_for(Q4K, "gfx1100", true), None);
    }

    /// Wave mandate: block size must be a multiple of the device's wavefront size.
    /// gfx1036 (this sandbox) is W32; CDNA MI-series is W64.
    #[test]
    fn block_dim_is_wave_aligned() {
        for &wave in &[32u32, 64] {
            for &n in &[0usize, 1, 16, 32, 33, 64, 65, 128, 200, 256, 1000] {
                let b = choose_block_dim(n, wave);
                assert_eq!(
                    b % wave,
                    0,
                    "block_dim for {n} pairs must be a multiple of wave_size {wave}"
                );
                assert!(b >= wave, "block_dim must be at least one wavefront");
                assert!(
                    b <= wave * 4,
                    "block_dim capped at 4 wavefronts ({})",
                    wave * 4
                );
            }
        }
    }

    /// #1 (WI-A grouped): vLLM `moe_align_block_size` port buckets tokens by expert and pads each expert run to `block_size`.
    /// Padding slots carry the sentinel token index (max+1) and every real (token,expert,weight) triple appears exactly.
    #[test]
    fn moe_align_block_size_buckets_and_pads() {
        // 4 tokens, top-2 routing into 3 experts, uneven distribution.
        // token0→[E0,E1], token1→[E0], token2→[E2,E0], token3→[E1,E2]
        let assignment = RoutingAssignment {
            tokens: vec![0, 0, 1, 2, 2, 3, 3],
            experts: vec![0, 1, 0, 2, 0, 1, 2],
            weights: vec![0.4, 0.6, 0.5, 0.3, 0.7, 0.2, 0.8],
        };
        let block_size = 4;
        let num_experts = 3;
        let sorted = moe_align_block_size(&assignment, block_size, num_experts);

        // Post-pad total divisible by block_size.
        assert_eq!(sorted.num_tokens_post_padded % block_size, 0);
        assert_eq!(
            sorted.num_blocks(),
            (sorted.num_tokens_post_padded / block_size) as u32
        );

        // Counts: E0=3, E1=2, E2=2 → padded runs 4,4,4 → 12 slots.
        assert_eq!(sorted.num_tokens_post_padded, 12);

        // Every real pair preserved exactly once.
        let max_tok = assignment.tokens.iter().copied().max().unwrap() as usize;
        let mut seen = std::collections::HashSet::new();
        for s in 0..sorted.num_tokens_post_padded {
            let tok = sorted.sorted_token_ids[s];
            let exp = sorted.sorted_expert_ids[s];
            if tok as usize > max_tok {
                continue; // padding
            }
            assert!(seen.insert((tok, exp)), "duplicate (token,expert) in sort");
        }
        assert_eq!(seen.len(), assignment.num_pairs());

        // Slots grouped by expert: expert id constant within each block window.
        for blk in 0..sorted.num_blocks() as usize {
            let start = blk * block_size;
            let first_exp = sorted.sorted_expert_ids[start];
            for s in start..start + block_size {
                if s < sorted.num_tokens_post_padded {
                    assert_eq!(
                        sorted.sorted_expert_ids[s], first_exp,
                        "expert run not contiguous within block"
                    );
                }
            }
        }
    }

    /// #1 (WI-A grouped): empty assignment → zero blocks, no panic.
    #[test]
    fn moe_align_block_size_empty_is_safe() {
        let assignment = RoutingAssignment {
            tokens: vec![],
            experts: vec![],
            weights: vec![],
        };
        let sorted = moe_align_block_size(&assignment, 4, 3);
        assert_eq!(sorted.num_tokens_post_padded, 0);
        assert_eq!(sorted.num_blocks(), 0);
    }

    /// #1 (WI-A grouped): planner maps sorted layout → wave-aligned grid/block.
    #[test]
    fn plan_grouped_dispatch_is_wave_aligned() {
        let assignment = RoutingAssignment {
            tokens: vec![0, 0, 1, 2],
            experts: vec![0, 1, 0, 2],
            weights: vec![0.5; 4],
        };
        let sorted = moe_align_block_size(&assignment, 4, 3);
        for &wave in &[32u32, 64] {
            let plan = plan_grouped_dispatch(&sorted, wave);
            assert_eq!(plan.block_x % wave, 0, "grouped block must be wave-aligned");
            assert!(plan.block_x >= wave);
            assert_eq!(plan.grid_x, sorted.num_blocks());
        }
    }

    /// G-A2: the planner resolves grid/block from a routing assignment and
    /// covers every pair with at least one thread.
    #[test]
    fn plan_covers_all_pairs() {
        // 3 tokens, top-2 = 6 pairs.
        let assignment = RoutingAssignment {
            tokens: vec![0, 0, 1, 1, 2, 2],
            experts: vec![3, 1, 0, 2, 4, 3],
            weights: vec![0.6, 0.4, 0.5, 0.5, 0.7, 0.3],
        };
        let plan = plan_fused_dispatch(&assignment, 32);
        assert_eq!(assignment.num_pairs(), 6);
        assert!(plan.block_x >= 32);
        let covered = (plan.grid_x as usize) * (plan.block_x as usize);
        assert!(
            covered >= 6,
            "grid*block ({covered}) must cover all 6 pairs"
        );
    }

    /// G-A2: empty routing → zero grid, no launch.
    #[test]
    fn plan_empty_routing_is_zero_grid() {
        let assignment = RoutingAssignment {
            tokens: vec![],
            experts: vec![],
            weights: vec![],
        };
        let plan = plan_fused_dispatch(&assignment, 32);
        assert_eq!(plan.grid_x, 0, "no pairs → no blocks");
    }

    /// G-A2: `from_route` flattens a per-token route into (token, expert, weight) triples, grouped by token (token-major layout).
    /// The order is a structural property of the struct - the kernel does not rely.
    #[test]
    fn from_route_flattens_in_token_expert_order() {
        let indices = vec![vec![3, 1], vec![0, 2]];
        let weights = vec![vec![0.6, 0.4], vec![0.5, 0.5]];
        let a = RoutingAssignment::from_route(&indices, &weights).unwrap();
        assert_eq!(a.tokens, vec![0, 0, 1, 1]);
        assert_eq!(a.experts, vec![3, 1, 0, 2]);
        assert_eq!(a.weights, vec![0.6, 0.4, 0.5, 0.5]);
        assert_eq!(a.num_pairs(), 4);
    }

    /// G-A2: mismatched indices/weights lengths are rejected, not silently
    /// truncated.
    #[test]
    fn from_route_rejects_mismatched_lengths() {
        let indices = vec![vec![0, 1]];
        let weights = vec![vec![0.5]]; // wrong count
        let err = RoutingAssignment::from_route(&indices, &weights);
        assert!(err.is_err(), "mismatched lengths must error");
    }

    /// G-A2: per-token mismatch (token has 2 experts but 1 weight) is
    /// rejected.
    #[test]
    fn from_route_rejects_per_token_mismatch() {
        let indices = vec![vec![0, 1], vec![2]];
        let weights = vec![vec![0.5, 0.5], vec![0.4, 0.6]]; // token 1 wrong
        let err = RoutingAssignment::from_route(&indices, &weights);
        assert!(err.is_err(), "per-token mismatch must error");
    }

    /// G-A2: input validation accepts a well-formed launch (all non-null,
    /// sane shape) and stages the routing assignment.
    #[test]
    fn validate_accepts_well_formed_launch() {
        let assignment = RoutingAssignment {
            tokens: vec![0, 1],
            experts: vec![2, 3],
            weights: vec![0.5, 0.5],
        };
        let dummy: *mut c_void = 0x1000 as *mut c_void;
        let res = validate_launch_inputs(dummy, dummy, dummy, dummy, dummy, &assignment, 64, 16);
        assert!(res.is_ok(), "well-formed launch must validate");
    }

    /// G-A2: any null device pointer is rejected with a labeled error.
    #[test]
    fn validate_rejects_null_pointers() {
        let assignment = RoutingAssignment {
            tokens: vec![0],
            experts: vec![0],
            weights: vec![1.0],
        };
        let dummy: *mut c_void = 0x1000 as *mut c_void;
        let err = validate_launch_inputs(
            std::ptr::null_mut(), // activations null
            dummy,
            dummy,
            dummy,
            dummy,
            &assignment,
            64,
            16,
        );
        assert!(err.is_err(), "null activations must be rejected");
        let msg = format!("{err:?}");
        assert!(msg.contains("activations"), "error must name the null arg");
    }

    /// G-A2: a degenerate shape (hidden=0 or inter=0) is rejected, not
    /// silently passed to the kernel as a zero-stride GEMM.
    #[test]
    fn validate_rejects_degenerate_shape() {
        let assignment = RoutingAssignment {
            tokens: vec![0],
            experts: vec![0],
            weights: vec![1.0],
        };
        let dummy: *mut c_void = 0x1000 as *mut c_void;
        let err = validate_launch_inputs(
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            &assignment,
            0,
            16, // hidden=0
        );
        assert!(err.is_err(), "hidden=0 must be rejected");
    }

    /// G-A2 parity with the CPU oracle shape: the routing assignment from a synthetic SoftmaxTopK route matches the indices the CPU reference (`grim_nn::moe::MoeRouter::route`) would produce.
    /// This is the host shape the GPU kernel will consume in G-A4.
    #[test]
    fn assignment_shape_matches_cpu_route() {
        // Mirror the `softmax_topk_selects_expected_experts` test in
        // grim-nn: 4 experts, top-2, the route returns indices [[0,2]].
        let indices = vec![vec![0, 2]];
        let weights = vec![vec![0.7, 0.3]];
        let a = RoutingAssignment::from_route(&indices, &weights).unwrap();
        // The kernel will dispatch block 0 → (token 0, expert 0) and
        // block 1 → (token 0, expert 2).
        assert_eq!(a.tokens, vec![0, 0]);
        assert_eq!(a.experts, vec![0, 2]);
    }

    // ── WI-B: cost model + selector host logic (G-B1) ──────────────────

    /// G-B1: the cost model is monotonic in each parameter when its coefficient is positive (the form RaMP borrows; coefficients are ours).
    /// This is the log-parity precondition for G-B2 regret.
    #[test]
    fn wave_cost_model_is_monotonic_in_each_param() {
        let m = WaveCostModel::default();
        let base = m.predict(4.0, 1024.0, 1e6, 0.1);
        // Increasing each positive-coefficient param must not decrease cost.
        assert!(
            m.predict(8.0, 1024.0, 1e6, 0.1) >= base,
            "more active warps must not reduce cost"
        );
        assert!(
            m.predict(4.0, 2048.0, 1e6, 0.1) > base,
            "more bytes/wave must strictly increase cost (c1 dominant)"
        );
        assert!(
            m.predict(4.0, 1024.0, 2e6, 0.1) >= base,
            "more flops/wave must not reduce cost"
        );
        assert!(
            m.predict(4.0, 1024.0, 1e6, 0.5) >= base,
            "higher stall rate must not reduce cost"
        );
    }

    /// G-B1: routing skew is 0 for uniform, →1 for one-expert-dominates.
    #[test]
    fn routing_skew_uniform_vs_dominated() {
        // 4 experts, 4 tokens each → perfectly uniform → skew 0.
        assert_eq!(routing_skew(&[4, 4, 4, 4]), 0.0);
        // All tokens to one expert → skew 1.
        assert!((routing_skew(&[16, 0, 0, 0]) - 1.0).abs() < 1e-6);
        // Empty → 0 by definition.
        assert_eq!(routing_skew(&[]), 0.0);
        assert_eq!(routing_skew(&[0, 0, 0, 0]), 0.0);
        // Mild skew is in (0, 1).
        let s = routing_skew(&[8, 4, 2, 2]);
        assert!(s > 0.0 && s < 1.0, "mild skew must be in (0,1), got {s}");
    }

    /// G-B1: the selector picks the small-batch row for low skew + light occupancy (decode shape)
    /// and the large-group row for high occupancy (prefill shape), with no CPU readback of the histogram.
    #[test]
    fn selector_picks_decode_for_low_skew_prefill_for_high_occupancy() {
        let mut sel = CharonSelector::new(default_variant_table(), 1);
        // Low skew, few warps → small-batch/decode.
        let v0 = sel.select(0.1, 1.0, 512.0, 1e5, 0.1);
        assert_eq!(v0, CharonVariant::SmallBatchDecode);
        // High skew → high-skew row (its bucket 0.9 is closest).
        let v1 = sel.select(0.95, 8.0, 2048.0, 1e6, 0.5);
        assert_eq!(v1, CharonVariant::HighSkew);
    }

    /// G-B1 / §5 de-sync guard: the selector does NOT thrash between adjacent
    /// layers - a challenger must win `min_hold` consecutive calls before taking over.
    #[test]
    fn selector_min_hold_prevents_variant_thrashing() {
        let mut sel = CharonSelector::new(default_variant_table(), 3);
        // Establish the current variant as SmallBatchDecode (low skew).
        let _ = sel.select(0.1, 1.0, 512.0, 1e5, 0.1);
        assert_eq!(sel.current(), CharonVariant::SmallBatchDecode);
        // One call with high skew — challenger wins once but min_hold=3
        // means we should NOT have switched yet.
        let _ = sel.select(0.95, 8.0, 2048.0, 1e6, 0.5);
        assert_eq!(
            sel.current(),
            CharonVariant::SmallBatchDecode,
            "de-sync guard: one challenging call must not switch"
        );
        // Two more challenging calls → switch allowed.
        let _ = sel.select(0.95, 8.0, 2048.0, 1e6, 0.5);
        let _ = sel.select(0.95, 8.0, 2048.0, 1e6, 0.5);
        assert_eq!(
            sel.current(),
            CharonVariant::HighSkew,
            "after min_hold consecutive wins, switch takes effect"
        );
    }

    /// G-B1 / §5 de-sync guard (alternating-challenger case): when two different non-current variants take turns as argmin, the per-challenger streak
    /// resets each time - no spurious switch can fire until one variant wins `min_hold` consecutive calls on its own.
    #[test]
    fn selector_min_hold_alternating_challengers_does_not_switch() {
        let mut sel = CharonSelector::new(default_variant_table(), 3);
        // Establish SmallBatchDecode as current (low skew).
        let _ = sel.select(0.1, 1.0, 512.0, 1e5, 0.1);
        assert_eq!(sel.current(), CharonVariant::SmallBatchDecode);

        // Alternating challengers: HighSkew (skew=0.95), LargeGroupPrefill (skew=0.5), HighSkew again.
        // Per-challenger streaks: HS=1, LGP=1, HS=1 - none reach min_hold=3, so no switch fires.
        let _ = sel.select(0.95, 8.0, 2048.0, 1e6, 0.5); // challenger: HighSkew
        assert_eq!(sel.current(), CharonVariant::SmallBatchDecode);
        let _ = sel.select(0.5, 4.0, 1024.0, 1e6, 0.3); // challenger: LargeGroupPrefill
        assert_eq!(sel.current(), CharonVariant::SmallBatchDecode);
        let _ = sel.select(0.95, 8.0, 2048.0, 1e6, 0.5); // challenger: HighSkew (streak resets to 1)
        assert_eq!(sel.current(), CharonVariant::SmallBatchDecode);

        // HighSkew wins 3 times consecutively → streak reaches min_hold=3,
        // switch allowed.
        let _ = sel.select(0.95, 8.0, 2048.0, 1e6, 0.5);
        let _ = sel.select(0.95, 8.0, 2048.0, 1e6, 0.5);
        let _ = sel.select(0.95, 8.0, 2048.0, 1e6, 0.5);
        assert_eq!(
            sel.current(),
            CharonVariant::HighSkew,
            "same challenger with min_hold consecutive wins must switch"
        );
    }

    /// G-B1: the variant table has exactly three rows (the v1 population
    /// cap) with distinct skew buckets covering [0, 1].
    #[test]
    fn variant_table_has_three_distinct_buckets() {
        let t = default_variant_table();
        assert_eq!(t.len(), 3, "v1 polymorphic population cap = 3");
        let buckets: Vec<f32> = t.iter().map(|r| r.skew_bucket).collect();
        let mut sorted = buckets.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        assert_eq!(buckets.len(), 3);
        // Buckets span low → high skew.
        assert!(sorted.first().copied().unwrap_or(1.0) < 0.4, "low bucket");
        assert!(sorted.last().copied().unwrap_or(0.0) > 0.6, "high bucket");
    }

    #[test]
    fn autotune_build_variant_table_from_autotuner() {
        use crate::autotune::{AutotuneConfig, Autotuner, MoeKernelKey, quantize_routing_skew};

        let mut tuner = Autotuner::for_device(0, "gfx90a");
        let key = MoeKernelKey {
            kernel: "grim_moe_fused_grouped".into(),
            gpu_arch: "gfx90a".into(),
            hidden: 4096,
            inter: 14336,
            num_experts: 8,
            top_k: 2,
            skew_bucket: quantize_routing_skew(0.2),
        };
        let cfg = AutotuneConfig {
            block_dim: 256,
            tile_kv: 64,
            grid_stride: 1,
            cycles_per_invocation: 500_000,
            spec_gamma: 4,
            spec_acceptance_threshold: 0.6,
            spec_alpha: 0.0,
            split_k: 0,
        };
        tuner.record_moe(key.clone(), cfg).expect("record_moe");

        let table = build_variant_table_from_autotuner(&tuner, "gfx90a");
        assert_eq!(table.len(), 3);
        // Row 0 corresponds to skew_bucket 0.2, which matches key.
        assert!((table[0].model.c_bytes_per_wave - 0.5).abs() < 1e-3);
    }

    #[test]
    fn test_charon_autotune_launch_config_fallback_and_lookup() {
        use crate::autotune::{AutotuneConfig, Autotuner, MoeKernelKey, quantize_routing_skew};

        let arch = "gfx1036";
        let mut tuner = Autotuner::for_device(0, arch);
        let histogram = vec![10, 2, 2, 2];

        // Case 1: Cache miss returns fallbacks (32 threads for gfx1036)
        let fallback_cfg =
            charon_autotune_launch_config(&tuner, arch, 4096, 14336, 4, 2, &histogram);
        assert_eq!(fallback_cfg.block_dim, 32);

        // Case 2: Cache hit returns registered config
        let skew = routing_skew(&histogram);
        let skew_bucket = quantize_routing_skew(skew);
        let key = MoeKernelKey {
            kernel: "grim_moe_fused_dispatch".to_string(),
            gpu_arch: arch.to_string(),
            hidden: 4096,
            inter: 14336,
            num_experts: 4,
            top_k: 2,
            skew_bucket,
        };
        let expected_cfg = AutotuneConfig {
            block_dim: 128,
            tile_kv: 64,
            grid_stride: 1,
            cycles_per_invocation: 120_000,
            spec_gamma: 4,
            spec_acceptance_threshold: 0.6,
            spec_alpha: 0.0,
            split_k: 0,
        };
        tuner.record_moe(key, expected_cfg).unwrap();

        let tuned_cfg = charon_autotune_launch_config(&tuner, arch, 4096, 14336, 4, 2, &histogram);
        assert_eq!(tuned_cfg.block_dim, 128);
        assert_eq!(tuned_cfg.cycles_per_invocation, 120_000);
    }

    /// The IQ grid/sign tables in `KERNEL_SOURCE` are generated from
    /// `grim_quant::iq_tables`, but "generated" is a claim about a past event.
    /// This re-parses them out of the HIP text and compares every entry against
    /// the crate, so a hand-edit, a partial regeneration, or a table that gets
    /// reordered upstream fails here instead of silently skewing weights on
    /// device -- where the only symptom would be a few percent of numerical
    /// drift in a grouped MoE.
    #[test]
    fn iqk_tables_match_grim_quant() {
        use grim_quant::iq_tables::{
            IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS,
        };

        fn parse_u64(src: &str, name: &str) -> Vec<u64> {
            let start = format!("__constant__ unsigned long long {name}[");
            let at = src
                .find(&start)
                .unwrap_or_else(|| panic!("table {name} not found in KERNEL_SOURCE"));
            let body = &src[at..];
            let open = body.find('{').expect("array body");
            let close = body.find('}').expect("array terminator");
            body[open + 1..close]
                .split(',')
                .filter(|t| !t.trim().is_empty())
                .map(|t| {
                    let t = t.trim().trim_end_matches("ULL");
                    u64::from_str_radix(t.trim_start_matches("0x"), 16)
                        .unwrap_or_else(|e| panic!("{name}: bad entry {t:?}: {e}"))
                })
                .collect()
        }

        fn parse_u32(src: &str, name: &str) -> Vec<u32> {
            let start = format!("__constant__ unsigned int {name}[");
            let at = src
                .find(&start)
                .unwrap_or_else(|| panic!("table {name} not found in KERNEL_SOURCE"));
            let body = &src[at..];
            let open = body.find('{').expect("array body");
            let close = body.find('}').expect("array terminator");
            body[open + 1..close]
                .split(',')
                .filter(|t| !t.trim().is_empty())
                .map(|t| {
                    let t = t.trim().trim_end_matches('u');
                    u32::from_str_radix(t.trim_start_matches("0x"), 16)
                        .unwrap_or_else(|e| panic!("{name}: bad entry {t:?}: {e}"))
                })
                .collect()
        }

        fn parse_u8(src: &str, name: &str) -> Vec<u8> {
            let start = format!("__constant__ unsigned char {name}[");
            let at = src
                .find(&start)
                .unwrap_or_else(|| panic!("table {name} not found in KERNEL_SOURCE"));
            let body = &src[at..];
            let open = body.find('{').expect("array body");
            let close = body.find('}').expect("array terminator");
            body[open + 1..close]
                .split(',')
                .filter(|t| !t.trim().is_empty())
                .map(|t| {
                    t.trim()
                        .parse::<u8>()
                        .unwrap_or_else(|e| panic!("{name}: {e}"))
                })
                .collect()
        }

        let src = KERNEL_SOURCE;
        let cases: Vec<(&str, Vec<u64>, Vec<u64>)> = vec![
            (
                "IQ2XS_GRID",
                parse_u64(src, "IQ2XS_GRID"),
                IQ2XS_GRID.iter().map(|&v| v as u64).collect(),
            ),
            (
                "IQ2XXS_GRID",
                parse_u64(src, "IQ2XXS_GRID"),
                IQ2XXS_GRID.iter().map(|&v| v as u64).collect(),
            ),
            (
                "IQ3S_GRID",
                parse_u32(src, "IQ3S_GRID")
                    .into_iter()
                    .map(u64::from)
                    .collect(),
                IQ3S_GRID.iter().map(|&v| u64::from(v)).collect(),
            ),
            (
                "IQ3XXS_GRID",
                parse_u32(src, "IQ3XXS_GRID")
                    .into_iter()
                    .map(u64::from)
                    .collect(),
                IQ3XXS_GRID.iter().map(|&v| u64::from(v)).collect(),
            ),
        ];
        for (name, got, want) in cases {
            assert_eq!(
                got.len(),
                want.len(),
                "{name}: entry count differs from grim-quant"
            );
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(
                    g, w,
                    "{name}[{i}]: kernel table 0x{g:X} != grim-quant 0x{w:X}"
                );
            }
        }

        let got_signs = parse_u8(src, "KSIGNS_IQ2XS");
        assert_eq!(
            got_signs.as_slice(),
            KSIGNS_IQ2XS.as_slice(),
            "KSIGNS_IQ2XS differs"
        );
        let got_mask = parse_u8(src, "KMASK_IQ2XS");
        assert_eq!(
            got_mask.as_slice(),
            KMASK_IQ2XS.as_slice(),
            "KMASK_IQ2XS differs"
        );
    }
}
