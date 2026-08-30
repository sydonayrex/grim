//! Charon WMMA — Tensor Core warp matrix multiply-accumulate kernels for CUDA MoE.
//!
//! Implements `nvcuda::wmma::` 16×16×16 WMMA tiles for fused gate+up+down
//! MoE expert GEMMs. Maps 1:1 to `charon_wmma.rs` in grim-backend-rocm which
//! uses `rocwmma::` 16×16 tiles. CUDA targets Turing+ (`sm_75`) which first
//! exposed `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`.

pub const CHARON_WMMA_SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <mma.h>
#include <math.h>

using namespace nvcuda::wmma;

extern "C" {

// ---------------------------------------------------------------------------
// grim_moe_wmma_gate_up — WMMA-accelerated gate+up fused SiLU projection.
//
// Computes per-expert gate_out[b, j] = silu(A[b,:] @ gate_w[e, j, :]) * (A[b,:] @ up_w[e, j, :])
// using 16×16×16 WMMA tiles. Each warp owns one (expert_tile, inter_tile) pair.
//
// Contract:
//   - A is float16, gate_w and up_w are float16.
//   - hidden and inter must be multiples of 16.
//   - Grid: (num_experts * inter / 16, batch / 16), Block: (32, 1).
// ---------------------------------------------------------------------------
__global__ void grim_moe_wmma_gate_up(
    const half* __restrict__ A,          // [batch, hidden]
    const half* __restrict__ gate_w,     // [num_experts, inter, hidden]
    const half* __restrict__ up_w,       // [num_experts, inter, hidden]
    float* __restrict__ gate_up_out,     // [batch, num_experts, inter]
    int batch, int hidden, int inter, int num_experts)
{
    const int warp_row = blockIdx.y;   // tile along batch (M)
    const int warp_col = blockIdx.x;   // tile along (expert * inter) (N)

    const int expert_idx = warp_col / (inter / 16);
    const int inter_tile = warp_col % (inter / 16);

    if (expert_idx >= num_experts) return;

    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> gate_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> up_frag;
    fragment<accumulator, 16, 16, 16, float> gate_acc, up_acc;

    fill_fragment(gate_acc, 0.0f);
    fill_fragment(up_acc, 0.0f);

    // K-loop in tiles of 16
    for (int k_tile = 0; k_tile < hidden / 16; ++k_tile) {
        load_matrix_sync(a_frag,
            A + warp_row * 16 * hidden + k_tile * 16, hidden);
        load_matrix_sync(gate_frag,
            gate_w + expert_idx * inter * hidden + inter_tile * 16 * hidden + k_tile * 16, hidden);
        load_matrix_sync(up_frag,
            up_w   + expert_idx * inter * hidden + inter_tile * 16 * hidden + k_tile * 16, hidden);

        mma_sync(gate_acc, a_frag, gate_frag, gate_acc);
        mma_sync(up_acc,   a_frag, up_frag,   up_acc);
    }

    // Apply SiLU(gate) * up and write fused output
    float* out_tile = gate_up_out
        + warp_row * 16 * num_experts * inter
        + expert_idx * inter
        + inter_tile * 16;

    // Store temporarily into shared mem, apply SiLU, then write
    __shared__ float gate_tmp[16 * 16];
    __shared__ float up_tmp[16 * 16];
    store_matrix_sync(gate_tmp, gate_acc, 16, mem_row_major);
    store_matrix_sync(up_tmp,   up_acc,   16, mem_row_major);
    __syncwarp();

    int lane = threadIdx.x;
    if (lane < 256) {
        float g = gate_tmp[lane];
        float u = up_tmp[lane];
        float silu_g = g / (1.0f + expf(-g));
        int b_off  = (lane / 16) * num_experts * inter;
        int col_off = lane % 16;
        out_tile[b_off + col_off] = silu_g * u;
    }
}

// ---------------------------------------------------------------------------
// grim_moe_wmma_down — WMMA-accelerated down projection.
//
// Computes out[b, h] += gate_up_out[b, e, :] @ down_w[e, h, :] * routing_weight
// using 16×16×16 WMMA tiles.
//
// Contract:
//   - gate_up_out and down_w are float32 accumulated from gate_up kernel.
//   - hidden and inter must be multiples of 16.
// ---------------------------------------------------------------------------
__global__ void grim_moe_wmma_down(
    const float* __restrict__ gate_up_out,   // [batch, num_experts, inter]
    const half*  __restrict__ down_w,         // [num_experts, hidden, inter]
    const unsigned int* __restrict__ router_tokens,  // [num_pairs]
    const unsigned int* __restrict__ router_experts, // [num_pairs]
    const float* __restrict__ router_weights,        // [num_pairs]
    float* __restrict__ out,                          // [batch, hidden]
    int batch, int hidden, int inter, int num_pairs)
{
    // One thread per (token, expert) pair for the down projection accumulation
    unsigned long long pair = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= (unsigned long long)num_pairs) return;

    unsigned int tok = router_tokens[pair];
    unsigned int exp = router_experts[pair];
    float w = router_weights[pair];

    const float* gu = gate_up_out + tok * (gridDim.y) * inter + exp * inter;
    const half*  dw = down_w + exp * hidden * inter;

    for (int h = 0; h < hidden; ++h) {
        float acc = 0.0f;
        for (int j = 0; j < inter; ++j) {
            acc += gu[j] * __half2float(dw[h * inter + j]);
        }
        atomicAdd(out + tok * hidden + h, acc * w);
    }
}

}
"#;
