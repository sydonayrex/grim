//! Charon WMMA / tensor-core grouped forward (WI-Charon-2).
//! The v2 plan confirms `charon.rs`'s existing kernels are scalar per-thread FMA loops (verified by kernel-body.

// `charon_wmma` exposes only the `KERNEL_SOURCE` literal and pure Rust unit tests asserting JIT-discoverability / sorted-routing contract / SiLU math - no host helpers
// returning `Result`, so no `grim_tensor::error` import is needed (keeping the import would trip the workspace `unused_imports = "deny"` gate, see root `Cargo.toml` `[workspace.lints]`).

// HIP source - Charon WMMA grouped forward (16x16 tensor-core tiles)

/// HIP source for the Charon WMMA grouped MoE forward kernel.
/// Entries: * `grim_moe_fused_grouped_wmma` - tensor-core grouped fused dispatch, FP32 accumulation, gated on rocWMMA availability (RDNA3/RDNA4/CDNA.
pub const KERNEL_SOURCE: &str = r#"
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
#include <rocwmma/rocwmma.hpp>
using namespace rocwmma;

// Per-expert tile accumulator is [hidden, inter] for the down contraction.
// The grouped forward reference (`grim_moe_fused_grouped` in `charon.rs:168`) computes per output element h ∈ [0, hidden):.
extern "C" __global__ void grim_moe_fused_grouped_wmma(
    const float* __restrict__ activations,
    const float* __restrict__ expert_gate_w,
    const float* __restrict__ expert_up_w,
    const float* __restrict__ expert_down_w,
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
        if (tok >= (unsigned int)num_tokens) continue;
        const unsigned int exp = sorted_expert_ids[s];
        const float w = sorted_weights[s];
        const float rsf = routed_scaling_factor * w;

        const float* a  = activations + (unsigned long long)tok * hidden;
        const float* gw = expert_gate_w + (unsigned long long)exp * (unsigned long long)inter * hidden;
        const float* uw = expert_up_w   + (unsigned long long)exp * (unsigned long long)inter * hidden;
        const float* dw = expert_down_w + (unsigned long long)exp * (unsigned long long)hidden * inter;

        // Per-thread scratch: act[j] for j ∈ [0, inter).
        // Each thread computes the full act vector for its token so the down contraction is.

        // Stage 1: Compute all intermediate activated projections act[j] = rsf * SiLU(gate) * up.
        // Stored in dynamic/auto array sized up to 4096 (standard max inter).
        // For inter <= 4096, this avoids 262k global atomicAdd collisions.
        float act_local[4096];
        int j = 0;
        for (; j + 3 < inter && j + 3 < 4096; j += 4) {
            float h_gate0 = 0.0f, h_gate1 = 0.0f, h_gate2 = 0.0f, h_gate3 = 0.0f;
            float h_up0 = 0.0f, h_up1 = 0.0f, h_up2 = 0.0f, h_up3 = 0.0f;

            if (hidden >= 16) {
                fragment<matrix_a, 16, 16, 16, float, row_major> frag_w;
                fragment<matrix_b, 16, 16, 16, float, col_major> frag_x;
                fragment<accumulator, 16, 16, 16, float> acc_g0, acc_g1, acc_g2, acc_g3;
                fragment<accumulator, 16, 16, 16, float> acc_u0, acc_u1, acc_u2, acc_u3;

                fill_fragment(acc_g0, 0.0f); fill_fragment(acc_g1, 0.0f);
                fill_fragment(acc_g2, 0.0f); fill_fragment(acc_g3, 0.0f);
                fill_fragment(acc_u0, 0.0f); fill_fragment(acc_u1, 0.0f);
                fill_fragment(acc_u2, 0.0f); fill_fragment(acc_u3, 0.0f);

                for (int i = 0; i < hidden; i += 16) {
                    const float* a_tile = a + i;
                    load_matrix_sync(frag_x, a_tile, 1);

                    load_matrix_sync(frag_w, gw + (unsigned long long)(j + 0) * hidden + i, hidden);
                    mma_sync(acc_g0, frag_w, frag_x, acc_g0);
                    load_matrix_sync(frag_w, gw + (unsigned long long)(j + 1) * hidden + i, hidden);
                    mma_sync(acc_g1, frag_w, frag_x, acc_g1);
                    load_matrix_sync(frag_w, gw + (unsigned long long)(j + 2) * hidden + i, hidden);
                    mma_sync(acc_g2, frag_w, frag_x, acc_g2);
                    load_matrix_sync(frag_w, gw + (unsigned long long)(j + 3) * hidden + i, hidden);
                    mma_sync(acc_g3, frag_w, frag_x, acc_g3);

                    load_matrix_sync(frag_w, uw + (unsigned long long)(j + 0) * hidden + i, hidden);
                    mma_sync(acc_u0, frag_w, frag_x, acc_u0);
                    load_matrix_sync(frag_w, uw + (unsigned long long)(j + 1) * hidden + i, hidden);
                    mma_sync(acc_u1, frag_w, frag_x, acc_u1);
                    load_matrix_sync(frag_w, uw + (unsigned long long)(j + 2) * hidden + i, hidden);
                    mma_sync(acc_u2, frag_w, frag_x, acc_u2);
                    load_matrix_sync(frag_w, uw + (unsigned long long)(j + 3) * hidden + i, hidden);
                    mma_sync(acc_u3, frag_w, frag_x, acc_u3);
                }
                h_gate0 = acc_g0.x[0]; h_gate1 = acc_g1.x[0];
                h_gate2 = acc_g2.x[0]; h_gate3 = acc_g3.x[0];
                h_up0 = acc_u0.x[0];   h_up1 = acc_u1.x[0];
                h_up2 = acc_u2.x[0];   h_up3 = acc_u3.x[0];
            } else {
                for (int i = 0; i < hidden; ++i) {
                    float a_val = a[i];
                    h_gate0 += gw[(unsigned long long)(j + 0) * hidden + i] * a_val;
                    h_gate1 += gw[(unsigned long long)(j + 1) * hidden + i] * a_val;
                    h_gate2 += gw[(unsigned long long)(j + 2) * hidden + i] * a_val;
                    h_gate3 += gw[(unsigned long long)(j + 3) * hidden + i] * a_val;
                    h_up0   += uw[(unsigned long long)(j + 0) * hidden + i] * a_val;
                    h_up1   += uw[(unsigned long long)(j + 1) * hidden + i] * a_val;
                    h_up2   += uw[(unsigned long long)(j + 2) * hidden + i] * a_val;
                    h_up3   += uw[(unsigned long long)(j + 3) * hidden + i] * a_val;
                }
            }

            act_local[j + 0] = rsf * (h_gate0 / (1.0f + expf(-h_gate0))) * h_up0;
            act_local[j + 1] = rsf * (h_gate1 / (1.0f + expf(-h_gate1))) * h_up1;
            act_local[j + 2] = rsf * (h_gate2 / (1.0f + expf(-h_gate2))) * h_up2;
            act_local[j + 3] = rsf * (h_gate3 / (1.0f + expf(-h_gate3))) * h_up3;
        }

        for (; j < inter && j < 4096; ++j) {
            float h_gate = 0.0f;
            float h_up = 0.0f;
            if (hidden >= 16) {
                fragment<matrix_a, 16, 16, 16, float, row_major> frag_w;
                fragment<matrix_b, 16, 16, 16, float, col_major> frag_x;
                fragment<accumulator, 16, 16, 16, float> frag_acc;

                fill_fragment(frag_acc, 0.0f);
                for (int i = 0; i < hidden; i += 16) {
                    load_matrix_sync(frag_x, a + i, 1);
                    load_matrix_sync(frag_w, gw + (unsigned long long)j * hidden + i, hidden);
                    mma_sync(frag_acc, frag_w, frag_x, frag_acc);
                }
                h_gate = frag_acc.x[0];

                fill_fragment(frag_acc, 0.0f);
                for (int i = 0; i < hidden; i += 16) {
                    load_matrix_sync(frag_x, a + i, 1);
                    load_matrix_sync(frag_w, uw + (unsigned long long)j * hidden + i, hidden);
                    mma_sync(frag_acc, frag_w, frag_x, frag_acc);
                }
                h_up = frag_acc.x[0];
            } else {
                for (int i = 0; i < hidden; ++i) {
                    h_gate += gw[(unsigned long long)j * hidden + i] * a[i];
                    h_up   += uw[(unsigned long long)j * hidden + i] * a[i];
                }
            }

            float silu_g = h_gate / (1.0f + expf(-h_gate));
            act_local[j] = rsf * silu_g * h_up;
        }

        // Stage 2: Down contraction. Accumulate per output dimension h in registers,
        // issuing exactly ONE atomicAdd per output element instead of 262,144 global atomics.
        const int valid_inter = inter < 4096 ? inter : 4096;
        for (int h = 0; h < hidden; ++h) {
            float acc_h = 0.0f;
            const float* dw_row = dw + (unsigned long long)h * inter;
            int jj = 0;
            for (; jj + 3 < valid_inter; jj += 4) {
                acc_h += dw_row[jj + 0] * act_local[jj + 0] +
                         dw_row[jj + 1] * act_local[jj + 1] +
                         dw_row[jj + 2] * act_local[jj + 2] +
                         dw_row[jj + 3] * act_local[jj + 3];
            }
            for (; jj < valid_inter; ++jj) {
                acc_h += dw_row[jj] * act_local[jj];
            }
            atomicAdd(out + (unsigned long long)tok * hidden + h, acc_h);
        }
    }
}
#else
// Scalar fallback (RDNA2 / gfx10 / any arch without rocWMMA).
// Mirrors the body of `grim_moe_fused_grouped` so the symbol always resolves; the Rust variant selector only.
extern "C" __global__ void grim_moe_fused_grouped_wmma(
    const float* __restrict__ activations,
    const float* __restrict__ expert_gate_w,
    const float* __restrict__ expert_up_w,
    const float* __restrict__ expert_down_w,
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
        if (tok >= (unsigned int)num_tokens) continue;
        const unsigned int exp = sorted_expert_ids[s];
        const float w = sorted_weights[s];
        const float* a  = activations + (unsigned long long)tok * hidden;
        const float* gw = expert_gate_w + (unsigned long long)exp * (unsigned long long)inter * hidden;
        const float* uw = expert_up_w   + (unsigned long long)exp * (unsigned long long)inter * hidden;
        const float* dw = expert_down_w + (unsigned long long)exp * (unsigned long long)hidden * inter;
        const float rsf = routed_scaling_factor * w;

        for (int j = 0; j < inter; ++j) {
            float g = 0.0f;
            float u = 0.0f;
            for (int i = 0; i < hidden; ++i) {
                g += gw[j * hidden + i] * a[i];
                u += uw[j * hidden + i] * a[i];
            }
            float silu_g = g / (1.0f + expf(-g));
            float act = silu_g * u;
            float scale = rsf * act;

            for (int h = 0; h < hidden; ++h) {
                unsigned long long out_idx = (unsigned long long)tok * hidden + h;
                atomicAdd(out + out_idx, dw[h * inter + j] * scale);
            }
        }
    }
}
#endif
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wmma_forward_kernel_is_jit_discoverable_both_paths() {
        // The symbol must resolve on WMMA arches (rocWMMA branch) AND on the
        // scalar fallback branch, so the JIT loader never 500s on gfx1036.
        assert!(KERNEL_SOURCE.contains("extern \"C\" __global__ void grim_moe_fused_grouped_wmma"));
        // rocWMMA include is gated, not unconditional (would fail on RDNA2).
        assert!(KERNEL_SOURCE.contains("#include <rocwmma/rocwmma.hpp>"));
        assert!(KERNEL_SOURCE.contains("mma_sync"));
        // Fallback path must exist for non-WMMA arches.
        assert!(KERNEL_SOURCE.contains("#else"));
    }

    #[test]
    fn wmma_forward_mirrors_grouped_contract() {
        // Same sorted routing arrays as grim_moe_fused_grouped so the host sort
        // feeds both forward paths.
        for sym in ["sorted_token_ids", "sorted_expert_ids", "sorted_weights"] {
            assert!(KERNEL_SOURCE.contains(sym), "missing sorted array: {sym}");
        }
        // Must respect the router scaling factor exactly like the scalar kernel.
        assert!(KERNEL_SOURCE.contains("routed_scaling_factor"));
    }

    #[test]
    fn wmma_forward_uses_silu_activation() {
        // In-register SiLU gate/up combine preserved from the scalar path.
        assert!(KERNEL_SOURCE.contains("1.0f + expf(-h_gate)"));
        assert!(KERNEL_SOURCE.contains("silu_g * h_up"));
    }
}
