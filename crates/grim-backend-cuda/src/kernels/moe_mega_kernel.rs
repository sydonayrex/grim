//! MoE Mega Kernel — Persistent CTA cooperative scheduler for top-k expert dispatch.
//!
//! Implements a persistent thread block scheduler for multi-expert top-k MoE
//! execution, matching `moe_mega_kernel.rs` in grim-backend-rocm. A single
//! long-lived kernel launch replaces per-expert sequential cuBLAS calls.

pub const MOE_MEGA_KERNEL_SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <math.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

extern "C" {

// ---------------------------------------------------------------------------
// grim_moe_mega_kernel — Persistent CTA scheduler for top-k MoE dispatch.
//
// Each CTA persists for the entire duration of the kernel and pulls work
// items from a shared atomic counter. One work item = one (token, expert)
// pair. When a CTA finishes its pair it atomically fetches the next.
//
// This avoids the per-expert kernel launch overhead that sequential cuBLAS
// paths pay for K > 8 experts, and lets the GPU scheduler saturate all SMs
// before any expert's batch completes.
//
// Grid: (num_ctas, 1, 1)  Block: (block_threads, 1, 1)
// ---------------------------------------------------------------------------
__global__ void grim_moe_mega_kernel(
    const float* __restrict__ activations,     // [batch, hidden]
    const float* __restrict__ expert_gate_w,   // [num_experts, inter, hidden]
    const float* __restrict__ expert_up_w,     // [num_experts, inter, hidden]
    const float* __restrict__ expert_down_w,   // [num_experts, hidden, inter]
    const unsigned int* __restrict__ router_tokens,   // [num_pairs]
    const unsigned int* __restrict__ router_experts,  // [num_pairs]
    const float* __restrict__ router_weights,         // [num_pairs]
    float* __restrict__ out,                          // [batch, hidden]
    int* __restrict__ work_counter,                   // [1] — atomic work queue
    int hidden, int inter, int num_pairs,
    float routed_scaling_factor)
{
    // Each CTA pulls work items from the global counter
    while (true) {
        int work_item;
        if (threadIdx.x == 0) {
            work_item = atomicAdd(work_counter, 1);
        }
        work_item = __shfl_sync(0xFFFFFFFF, work_item, 0);

        if (work_item >= num_pairs) break;

        unsigned int tok = router_tokens[work_item];
        unsigned int exp = router_experts[work_item];
        float w = router_weights[work_item];

        const float* a  = activations    + (unsigned long long)tok * hidden;
        const float* gw = expert_gate_w  + (unsigned long long)exp * inter * hidden;
        const float* uw = expert_up_w    + (unsigned long long)exp * inter * hidden;
        const float* dw = expert_down_w  + (unsigned long long)exp * hidden * inter;

        // Cooperate across threads in the warp for the inter loop
        for (int j = threadIdx.x; j < inter; j += blockDim.x) {
            float gate_j = 0.0f, up_j = 0.0f;
            for (int k = 0; k < hidden; ++k) {
                gate_j += gw[j * hidden + k] * a[k];
                up_j   += uw[j * hidden + k] * a[k];
            }
            float silu_g = gate_j / (1.0f + expf(-gate_j));
            float act    = silu_g * up_j * routed_scaling_factor * w;

            for (int h = 0; h < hidden; ++h) {
                atomicAdd(out + tok * hidden + h, dw[h * inter + j] * act);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// grim_moe_align_block_size — Compute per-expert token counts and padded layout
// for the token-sorted grouped dispatch path.
//
// Mirrors `moe_align_block_size` on ROCm: fills sorted_token_ids,
// sorted_expert_ids, sorted_weights, and num_tokens_post_padded.
// ---------------------------------------------------------------------------
__global__ void grim_moe_align_block_size(
    const unsigned int* __restrict__ topk_ids,       // [batch * topk]
    const float* __restrict__ topk_weights,          // [batch * topk]
    unsigned int* __restrict__ sorted_token_ids,     // [padded]
    unsigned int* __restrict__ sorted_expert_ids,    // [padded]
    float* __restrict__ sorted_weights,              // [padded]
    int* __restrict__ expert_token_count,            // [num_experts]
    int* __restrict__ num_tokens_post_padded,        // [1]
    int num_tokens, int topk, int num_experts, int block_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_tokens * topk) return;

    int tok = tid / topk;
    unsigned int exp = topk_ids[tid];
    float wt = topk_weights[tid];

    if (exp >= (unsigned int)num_experts) return;

    int slot = atomicAdd(&expert_token_count[exp], 1);
    // Write into sorted layout; caller pads to block_size multiples
    sorted_token_ids[exp * num_tokens + slot]  = (unsigned int)tok;
    sorted_expert_ids[exp * num_tokens + slot] = exp;
    sorted_weights[exp * num_tokens + slot]    = wt;

    if (tid == 0) {
        // Compute padded total (sum of ceil(count/block_size)*block_size for each expert)
        int total = 0;
        for (int e = 0; e < num_experts; ++e) {
            int cnt = expert_token_count[e];
            total += ((cnt + block_size - 1) / block_size) * block_size;
        }
        *num_tokens_post_padded = total;
    }
}

}
"#;
