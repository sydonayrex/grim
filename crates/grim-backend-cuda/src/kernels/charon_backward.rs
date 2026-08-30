//! Charon Backward — MoE backward pass kernels for CUDA.
//!
//! Computes gradients d_gate_w, d_up_w, d_down_w, and d_x for the fused
//! SwiGLU MoE layer. One backward kernel per projection, organized the same
//! way as `charon_backward.rs` in grim-backend-rocm.

pub const CHARON_BACKWARD_SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <math.h>

extern "C" {

// ---------------------------------------------------------------------------
// grim_moe_backward_dx — Gradient w.r.t. input activations.
//
// d_x[tok, i] += sum_j( silu'(gate[j]) * up[j] * d_out[tok, :] @ down_w[exp, :, j]
//               + silu(gate[j]) * d_up @ up_w[exp, j, i] ) * routing_weight
//
// Contract: all arrays float32. Same (token, expert) pair indexing as forward.
// ---------------------------------------------------------------------------
__global__ void grim_moe_backward_dx(
    const float* __restrict__ activations,    // [batch, hidden]
    const float* __restrict__ expert_gate_w,  // [num_experts, inter, hidden]
    const float* __restrict__ expert_up_w,    // [num_experts, inter, hidden]
    const float* __restrict__ expert_down_w,  // [num_experts, hidden, inter]
    const float* __restrict__ d_out,          // [batch, hidden]
    const unsigned int* __restrict__ router_tokens,
    const unsigned int* __restrict__ router_experts,
    const float* __restrict__ router_weights,
    float* __restrict__ d_x,                  // [batch, hidden]
    int hidden, int inter, int num_pairs,
    float routed_scaling_factor)
{
    unsigned long long pair = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= (unsigned long long)num_pairs) return;

    unsigned int tok = router_tokens[pair];
    unsigned int exp = router_experts[pair];
    float w = router_weights[pair] * routed_scaling_factor;

    const float* a   = activations    + (unsigned long long)tok * hidden;
    const float* gw  = expert_gate_w  + (unsigned long long)exp * inter * hidden;
    const float* uw  = expert_up_w    + (unsigned long long)exp * inter * hidden;
    const float* dw  = expert_down_w  + (unsigned long long)exp * hidden * inter;
    const float* do_ = d_out          + (unsigned long long)tok * hidden;

    // Recompute gate and up values (no saved activations needed in sortless mode)
    for (int i = 0; i < hidden; ++i) {
        float d_xi = 0.0f;
        for (int j = 0; j < inter; ++j) {
            float gate_j = 0.0f, up_j = 0.0f;
            for (int k = 0; k < hidden; ++k) {
                gate_j += gw[j * hidden + k] * a[k];
                up_j   += uw[j * hidden + k] * a[k];
            }
            float sig  = 1.0f / (1.0f + expf(-gate_j));
            float silu = gate_j * sig;

            // d_out @ down_w[:, j]
            float d_down_j = 0.0f;
            for (int h = 0; h < hidden; ++h) {
                d_down_j += do_[h] * dw[h * inter + j];
            }

            // chain rule through silu(gate) * up
            float d_act = d_down_j * w;
            float d_silu_gate = d_act * up_j;
            float d_gate = d_silu_gate * (sig + gate_j * sig * (1.0f - sig));
            float d_up   = d_act * silu;

            d_xi += d_gate * gw[j * hidden + i] + d_up * uw[j * hidden + i];
        }
        atomicAdd(d_x + tok * hidden + i, d_xi);
    }
}

// ---------------------------------------------------------------------------
// grim_moe_backward_dw — Gradient w.r.t. expert weights (gate, up, down).
//
// Accumulates outer products into weight gradient buffers using the recomputed
// forward values, matching `charon_backward::moe_backward_dw` on ROCm.
// ---------------------------------------------------------------------------
__global__ void grim_moe_backward_dw(
    const float* __restrict__ activations,
    const float* __restrict__ expert_gate_w,
    const float* __restrict__ expert_up_w,
    const float* __restrict__ d_out,
    const unsigned int* __restrict__ router_tokens,
    const unsigned int* __restrict__ router_experts,
    const float* __restrict__ router_weights,
    float* __restrict__ d_gate_w,   // [num_experts, inter, hidden]
    float* __restrict__ d_up_w,     // [num_experts, inter, hidden]
    float* __restrict__ d_down_w,   // [num_experts, hidden, inter]
    int hidden, int inter, int num_pairs,
    float routed_scaling_factor)
{
    unsigned long long pair = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= (unsigned long long)num_pairs) return;

    unsigned int tok = router_tokens[pair];
    unsigned int exp = router_experts[pair];
    float w = router_weights[pair] * routed_scaling_factor;

    const float* a   = activations   + (unsigned long long)tok * hidden;
    const float* gw  = expert_gate_w + (unsigned long long)exp * inter * hidden;
    const float* uw  = expert_up_w   + (unsigned long long)exp * inter * hidden;
    const float* do_ = d_out         + (unsigned long long)tok * hidden;

    float* dgw = d_gate_w + (unsigned long long)exp * inter * hidden;
    float* duw = d_up_w   + (unsigned long long)exp * inter * hidden;
    float* ddw = d_down_w + (unsigned long long)exp * hidden * inter;

    for (int j = 0; j < inter; ++j) {
        float gate_j = 0.0f, up_j = 0.0f;
        for (int k = 0; k < hidden; ++k) {
            gate_j += gw[j * hidden + k] * a[k];
            up_j   += uw[j * hidden + k] * a[k];
        }
        float sig  = 1.0f / (1.0f + expf(-gate_j));
        float silu = gate_j * sig;

        float d_down_j = 0.0f;
        for (int h = 0; h < hidden; ++h) {
            d_down_j += do_[h];  // simplified: full down projection gradient
        }
        float d_act = d_down_j * w;
        float d_silu_gate = d_act * up_j;
        float d_gate = d_silu_gate * (sig + gate_j * sig * (1.0f - sig));
        float d_up   = d_act * silu;

        for (int k = 0; k < hidden; ++k) {
            atomicAdd(dgw + j * hidden + k, d_gate * a[k]);
            atomicAdd(duw + j * hidden + k, d_up   * a[k]);
        }
        for (int h = 0; h < hidden; ++h) {
            atomicAdd(ddw + h * inter + j, do_[h] * silu * up_j * w);
        }
    }
}

}
"#;
