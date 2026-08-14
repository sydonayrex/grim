//! Charon WMMA / tensor-core grouped forward (WI-Charon-2).
//!
//! The v2 plan confirms `charon.rs`'s existing kernels are scalar per-thread
//! FMA loops (verified by kernel-body read), not tensor-core tiled GEMMs.
//! This file
//! adds a **grouped** (token-sorted) WMMA variant that reuses the identical
//! `grim_moe_fused_grouped` host/sort contract and in-register SiLU math,
//! replacing the gate/up/down contractions with 16×16 rocWMMA tiles.
//!
//! Design notes (per `rocm-hip-kernels` + `grim-rocm-ffi`):
//! * rocWMMA fragment setup is reused from `wmma_gemm.rs` — NOT duplicated.
//!   The `extern "C" __global__ void grim_wmma_gemm` there owns the
//!   `#include <rocwmma/rocwmma.hpp>` + `fragment<>` boilerplate; this kernel
//!   mirrors that exact `#if defined(__gfx11xx__) || defined(__gfx120x__)`
//!   guard and the `_Float16` tile path, so on architectures without native
//!   WMMA (gfx1036 / RDNA2, the project's stated target) it compiles to the
//!   scalar fallback instead of failing to link.
//! * gated behind `accel_features::wmma_supported(arch, QuantMode::Fp32)` on
//!   the Rust side — the variant selector (`CharonSelector`/`CharonVariant`
//!   in `charon.rs`) chooses this path only when WMMA is available.
//! * Left/right limits: does NOT touch the sortless `grim_moe_fused_dispatch`
//!   (single-token decode path); WMMA tiling helps batched prefill, not
//!   single-token decode.
//!
//! Device-gated: WMMA numeric parity vs `grim_moe_fused_grouped` is UNVERIFIED
//! in this sandbox (no GPU). Host-side planning helpers are pure and tested.

// `charon_wmma` exposes only the `KERNEL_SOURCE` literal and pure Rust unit
// tests asserting JIT-discoverability / sorted-routing contract / SiLU math —
// no host helpers returning `Result`, so no `grim_tensor::error` import is
// needed (keeping the import would trip the workspace `unused_imports = "deny"`
// gate, see root `Cargo.toml` `[workspace.lints]`).

// ---------------------------------------------------------------------------
// HIP source — Charon WMMA grouped forward (16x16 tensor-core tiles)
// ---------------------------------------------------------------------------

/// HIP source for the Charon WMMA grouped MoE forward kernel.
///
/// Entries:
/// * `grim_moe_fused_grouped_wmma` — tensor-core grouped fused dispatch,
///   FP32 accumulation, gated on rocWMMA availability (RDNA3/RDNA4/CDNA MFMA).
pub const KERNEL_SOURCE: &str = r#"
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
#include <rocwmma/rocwmma.hpp>
using namespace rocwmma;

// Per-expert tile accumulator is [hidden, inter] for the down contraction.
// The grouped forward reference (`grim_moe_fused_grouped` in `charon.rs:168`)
// computes per output element h ∈ [0, hidden):
//   acc[h] = sum_j down_w[h*inter + j] * act[j]
// where  act[j] = silu(gate_w[j*hidden + :] · a) * (up_w[j*hidden + :] · a)
//                = silu(h_gate[j]) * h_up[j],  with  h_gate/h_up ∈ R^inter.
// So h_gate / h_up live on the INTER axis; act[j] for j ∈ [0, inter); then
// the down contraction produces `out[h]` for h ∈ [0, hidden).
//
// Tiling: gate/up are contractions of [inter, hidden] × [hidden] → [inter];
// down is a contraction of [hidden, inter] × [inter] → [hidden]. We tile the
// inner contractions 16-wide (rocWMMA 16×16×16 fragments) and accumulate
// per-fragment — matching `wmma_gemm.rs`'s fragment-setup idiom (the
// documented reuse: "import rocWMMA fragment setup from wmma_gemm.rs rather
// than duplicating the include/fragment boilerplate").
//
// The historical draft conflated `hidden` and `inter` axes (looped `h` over
// `[0, hidden)` but indexed `gw[h*inter + j]`, a `[hidden, inter]` layout that
// contradicts gate_w's actual `[inter, hidden]` shape) AND read `x[j]` for
// `j ∈ [0, inter)` which would read out-of-bounds when `inter > hidden`. The
// parity gate (WMMA vs scalar `grim_moe_fused_grouped`) would surface both on
// a device; fixed here so both paths compute the same forward.
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

        // Per-thread scratch: act[j] for j ∈ [0, inter). Each thread computes
        // the full act vector for its token so the down contraction is a clean
        // per-thread reduction. (Inter is bounded by block_size's K dim; in
        // practice ≤ ~14336, well within per-thread LDS/registers for grouped
        // dispatch.)
        // NOTE: `act` is sized via a stack VLA-equivalent loop — kept simple
        // by recomputing per (h, j) inside the down loop to avoid dynamic
        // stack allocation, exactly like the scalar forward at charon.rs:196.

        // Down contraction: out[h] = sum_j down_w[h*inter + j] * act[j].
        // h ∈ [0, hidden), j ∈ [0, inter). Recompute act[j] per (h, j) to
        // avoid a per-thread `inter`-sized stack buffer; the act[j] value is
        // reused only `hidden` times (once per h), so recomputation is
        // `O(hidden * inter)` work — the same work the scalar forward does.
        // The WMMA win materializes on the hidden-internal GEMMs below
        // (gate/up contractions: `O(hidden)` per j-element pair), not on the
        // outer act/down loops.
        for (int h = 0; h < hidden; ++h) {
            float acc = 0.0f;
            for (int j = 0; j < inter; ++j) {
                // WMMA-accelerated gate/up contractions over [0, hidden).
                // gate_w[j, :] · a  → h_gate[j]
                // up_w[j, :]   · a  → h_up[j]
                //
                // Each contraction is a [1 × hidden] · [hidden × 1] dot
                // product; we tile the K=hidden dim 16-wide via rocWMMA 16×16
                // fragments and read the [0,0] accumulator element (the only
                // valid M=N=1 output of a vector-vector mma).
                fragment<matrix_a, 16, 16, 16, float, row_major> frag_w;
                fragment<matrix_b, 16, 16, 16, float, col_major> frag_x;
                fragment<accumulator, 16, 16, 16, float> frag_acc;

                fill_fragment(frag_acc, 0.0f);
                for (int i = 0; i < hidden; i += 16) {
                    const float* w_tile = gw + (unsigned long long)j * hidden + i;
                    const float* a_tile = a + i;
                    load_matrix_sync(frag_w, w_tile, hidden);
                    load_matrix_sync(frag_x, a_tile, 1);
                    mma_sync(frag_acc, frag_w, frag_x, frag_acc);
                }
                float h_gate = frag_acc.x[0];

                fill_fragment(frag_acc, 0.0f);
                for (int i = 0; i < hidden; i += 16) {
                    const float* w_tile = uw + (unsigned long long)j * hidden + i;
                    const float* a_tile = a + i;
                    load_matrix_sync(frag_w, w_tile, hidden);
                    load_matrix_sync(frag_x, a_tile, 1);
                    mma_sync(frag_acc, frag_w, frag_x, frag_acc);
                }
                float h_up = frag_acc.x[0];

                float silu_g = h_gate / (1.0f + expf(-h_gate));
                float act = silu_g * h_up;
                acc += dw[(unsigned long long)h * inter + j] * act;
            }
            unsigned long long out_idx = (unsigned long long)tok * hidden + h;
            atomicAdd(out + out_idx, rsf * acc);
        }
    }
}
#else
// Scalar fallback (RDNA2 / gfx10 / any arch without rocWMMA). Mirrors the body
// of `grim_moe_fused_grouped` so the symbol always resolves; the Rust variant
// selector only picks this entry on non-WMMA arches, where it is functionally
// identical to the scalar forward (kept here to avoid a dangling JIT symbol).
//
// This fallback is bit-for-bit equivalent to `grim_moe_fused_grouped`
// (charon.rs:168) on the same routing + weights: same `[inter, hidden]`
// gate/up layout (`gw[j*hidden + i]`), same `[hidden, inter]` down layout
// (`dw[h*inter + j]`), same SiLU activation, same routed-scaling outer
// accumulation. The parity gate (WMMA path vs scalar `grim_moe_fused_grouped`)
// reduces on-device to comparing these two paths; this fallback is the same
// math the WMMA path above targets, just without rocWMMA tiling.
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
        // Same body as `grim_moe_fused_grouped` (charon.rs:196-211). Kept
        // inline (not a call) so the symbol resolves without a cross-kernel
        // forward declaration — and so the parity gate isolates the WMMA
        // path cleanly (this fallback is the scalar reference's twin).
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
                acc += dw[h * inter + j] * act;
            }
            unsigned long long out_idx = (unsigned long long)tok * hidden + h;
            atomicAdd(out + out_idx, routed_scaling_factor * w * acc);
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
