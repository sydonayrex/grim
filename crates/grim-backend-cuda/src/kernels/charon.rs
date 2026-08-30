//! Charon — P-DAFD fused MoE dispatch kernel family for CUDA.

/// CUDA C++ source for the Charon fused MoE kernels.
pub const CHARON_SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <math.h>

extern "C" {

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
    unsigned long long pair = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= (unsigned long long)num_pairs) return;

    unsigned int tok = router_tokens[pair];
    unsigned int exp = router_experts[pair];
    float w = router_weights[pair];

    const float* a = activations + (unsigned long long)tok * hidden;
    const float* gw = expert_gate_w + (unsigned long long)exp * inter * hidden;
    const float* uw = expert_up_w   + (unsigned long long)exp * inter * hidden;
    const float* dw = expert_down_w + (unsigned long long)exp * hidden * inter;

    for (int j = 0; j < inter; ++j) {
        float g = 0.0f;
        float u = 0.0f;
        for (int i = 0; i < hidden; ++i) {
            g += gw[j * hidden + i] * a[i];
            u += uw[j * hidden + i] * a[i];
        }
        float silu_g = g / (1.0f + expf(-g));
        float act = silu_g * u;
        float scale = routed_scaling_factor * w * act;

        for (int h = 0; h < hidden; ++h) {
            unsigned long long out_idx = (unsigned long long)tok * hidden + h;
            atomicAdd(out + out_idx, dw[h * inter + j] * scale);
        }
    }
}

__global__ void grim_moe_fused_grouped(
    const float* __restrict__ activations,     // [batch, hidden]
    const float* __restrict__ expert_gate_w,   // [num_experts, inter*hidden]
    const float* __restrict__ expert_up_w,     // [num_experts, inter*hidden]
    const float* __restrict__ expert_down_w,   // [num_experts, hidden*inter]
    const unsigned int* __restrict__ sorted_token_ids,  // [num_tokens_post_padded]
    const unsigned int* __restrict__ sorted_expert_ids, // [num_tokens_post_padded]
    const float* __restrict__ sorted_weights,           // [num_tokens_post_padded]
    float* __restrict__ out,                            // [batch, hidden]
    int hidden, int inter, int num_tokens_post_padded, int batch,
    float routed_scaling_factor)
{
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (unsigned long long)num_tokens_post_padded) return;

    unsigned int tok = sorted_token_ids[idx];
    if (tok >= (unsigned int)batch) return; // padding slot

    unsigned int exp = sorted_expert_ids[idx];
    float w = sorted_weights[idx];

    const float* a = activations + (unsigned long long)tok * hidden;
    const float* gw = expert_gate_w + (unsigned long long)exp * inter * hidden;
    const float* uw = expert_up_w   + (unsigned long long)exp * inter * hidden;
    const float* dw = expert_down_w + (unsigned long long)exp * hidden * inter;

    for (int j = 0; j < inter; ++j) {
        float g = 0.0f;
        float u = 0.0f;
        for (int i = 0; i < hidden; ++i) {
            g += gw[j * hidden + i] * a[i];
            u += uw[j * hidden + i] * a[i];
        }
        float silu_g = g / (1.0f + expf(-g));
        float act = silu_g * u;
        float scale = routed_scaling_factor * w * act;

        for (int h = 0; h < hidden; ++h) {
            unsigned long long out_idx = (unsigned long long)tok * hidden + h;
            atomicAdd(out + out_idx, dw[h * inter + j] * scale);
        }
    }
}

}
"#;
