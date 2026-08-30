//! ScytheRing persistent dispatch kernel for CUDA.
//!
//! Ported from grim-backend-rocm `kernels/scythe_persistent.rs`.
//! Device-side opcode loop for persistent-kernel serving on CUDA (Volta+).
//! Matches `scythe_task_descriptor_t` and `moe_task_descriptor_t` FFI layouts.

pub const SCYTHE_PERSISTENT_SOURCE: &str = r#"
#include <cuda_fp16.h>

struct __align__(32) scythe_task_descriptor_t {
    unsigned int opcode;     // 0=nop,1=col-GEMM,2=row-GEMM,3=attn,4=norm,5=CommFuse,6=MoE,7=add
    unsigned int m, n, k;
    unsigned long long input_ptr;
    unsigned long long weight_ptr;   // opcode=6: points to moe_task_descriptor_t
    unsigned long long output_ptr;
    unsigned long long peer_ptr;
    unsigned int status;     // 0=pending, 1=running, 2=complete
};

#define MOE_QUANT_FP32   0u
#define MOE_QUANT_FP8    1u
#define MOE_QUANT_MXFP4  2u
#define MOE_QUANT_MXFP8  3u
#define MOE_QUANT_Q8_0   4u
#define MOE_QUANT_IQK    5u

struct __align__(32) moe_task_descriptor_t {
    unsigned int hidden;
    unsigned int inter;
    unsigned int num_tokens;
    unsigned int block_size;
    unsigned int num_experts;
    unsigned int top_k;
    unsigned int quant_mode;
    float routed_scaling_factor;
    unsigned long long gate_w_ptr;
    unsigned long long up_w_ptr;
    unsigned long long down_w_ptr;
    unsigned long long sorted_token_ids_ptr;
    unsigned long long sorted_expert_ids_ptr;
    unsigned long long sorted_weights_ptr;
    unsigned long long expert_token_count_ptr;
    unsigned long long num_tokens_post_padded_ptr;
    unsigned long long expert_offsets_ptr;
    unsigned int num_pairs;
    unsigned int flags;
};

extern "C" {

__device__ void grim_scythe_moe_dispatch(
    const moe_task_descriptor_t* __restrict__ moe_desc,
    const float* __restrict__ activations,
    float* __restrict__ out)
{
    // Inline Charon forward dispatch inside persistent CTA
    const float* gate_w = (const float*)moe_desc->gate_w_ptr;
    const float* up_w   = (const float*)moe_desc->up_w_ptr;
    const float* down_w = (const float*)moe_desc->down_w_ptr;
    const unsigned int* router_tok = (const unsigned int*)moe_desc->sorted_token_ids_ptr;
    const unsigned int* router_exp = (const unsigned int*)moe_desc->sorted_expert_ids_ptr;
    const float* router_wt = (const float*)moe_desc->sorted_weights_ptr;

    int num_pairs = moe_desc->num_pairs;
    int hidden = moe_desc->hidden;
    int inter = moe_desc->inter;
    float factor = moe_desc->routed_scaling_factor;

    int tid = threadIdx.x;
    for (int p = blockIdx.x; p < num_pairs; p += gridDim.x) {
        unsigned int tok = router_tok[p];
        unsigned int exp = router_exp[p];
        float w = router_wt[p] * factor;

        const float* a = activations + (unsigned long long)tok * hidden;
        const float* gw = gate_w + (unsigned long long)exp * inter * hidden;
        const float* uw = up_w   + (unsigned long long)exp * inter * hidden;
        const float* dw = down_w + (unsigned long long)exp * hidden * inter;

        for (int j = tid; j < inter; j += blockDim.x) {
            float g = 0.0f, u = 0.0f;
            for (int k = 0; k < hidden; ++k) {
                g += gw[j * hidden + k] * a[k];
                u += uw[j * hidden + k] * a[k];
            }
            float silu_g = g / (1.0f + __expf(-g));
            float act = silu_g * u * w;

            for (int h = 0; h < hidden; ++h) {
                atomicAdd(out + tok * hidden + h, dw[h * inter + j] * act);
            }
        }
    }
}

__global__ void grim_scythe_persistent_loop(
    volatile scythe_task_descriptor_t* __restrict__ ring,
    volatile unsigned int* __restrict__ head,
    volatile unsigned int* __restrict__ tail,
    unsigned int ring_capacity,
    volatile int* __restrict__ terminate_flag)
{
    unsigned int local_tail = *tail;

    while (!(*terminate_flag)) {
        if (local_tail == *head) {
            // Spin waiting for new descriptor
            continue;
        }

        unsigned int slot = local_tail % ring_capacity;
        volatile scythe_task_descriptor_t* desc = &ring[slot];

        if (desc->status == 0) { // pending
            if (threadIdx.x == 0) {
                desc->status = 1; // running
            }
            __syncthreads();

            switch (desc->opcode) {
                case 0: // nop
                    break;
                case 6: // MoE task
                    {
                        const moe_task_descriptor_t* moe_desc = (const moe_task_descriptor_t*)desc->weight_ptr;
                        const float* in = (const float*)desc->input_ptr;
                        float* out = (float*)desc->output_ptr;
                        grim_scythe_moe_dispatch(moe_desc, in, out);
                    }
                    break;
                default:
                    break;
            }

            __syncthreads();
            if (threadIdx.x == 0) {
                desc->status = 2; // complete
                local_tail++;
                *tail = local_tail;
            }
            __syncthreads();
        }
    }
}

} // extern "C"
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_contains_persistent_loop() {
        assert!(SCYTHE_PERSISTENT_SOURCE.contains("grim_scythe_persistent_loop"));
        assert!(SCYTHE_PERSISTENT_SOURCE.contains("moe_task_descriptor_t"));
        assert!(SCYTHE_PERSISTENT_SOURCE.contains("grim_scythe_moe_dispatch"));
    }
}
