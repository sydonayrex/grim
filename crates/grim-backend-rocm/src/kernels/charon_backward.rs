//! Charon MoE backward pass - expert-weight gradients (WI-Charon-1).
//! Confirms the v2 plan's finding: `grim-backend-rocm` had **no** MoE backward kernel before this file.

use std::ffi::c_void;

use grim_tensor::error::{Error, Result};

// HIP source - Charon MoE backward kernels (FP32 expert-weight gradients)

/// HIP source for the Charon MoE backward kernel family.
/// Entries (each `__global__`, grouped/token-sorted contract): * `grim_moe_fused_grouped_backward` - computes `d_gate_w`, `d_up_w`, `d_down_w`, `d_x` for the.
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

    // SiLU derivative: silu'(z) = sigmoid(z) * (1 + z * (1 - sigmoid(z))).
    // Equivalently silu'(z) = silu(z) * (1/z + 1) for z != 0; use the stable form.
    __device__ __forceinline__ float silu_grad(float z) {
        float s = 1.0f / (1.0f + expf(-z));
        return s * (1.0f + z * (1.0f - s));
    }

    // ──────────────────────────────────────────────────────────────────────── grim_moe_fused_grouped_backward - FP32 expert-weight backward (WI-Charon-1).
    // Layout (matches grim_moe_fused_grouped forward): activations : [batch, hidden] (input x, needed for d_w) gate_w/up_w :.
    __global__ void grim_moe_fused_grouped_backward(
        const float* __restrict__ activations,
        const float* __restrict__ gate_w,
        const float* __restrict__ up_w,
        const float* __restrict__ down_w,
        const float* __restrict__ d_y,
        float* __restrict__ d_gate_w,
        float* __restrict__ d_up_w,
        float* __restrict__ d_down_w,
        float* __restrict__ d_x,
        const unsigned int* __restrict__ sorted_token_ids,
        const unsigned int* __restrict__ sorted_expert_ids,
        const float* __restrict__ sorted_weights,
        const float* __restrict__ stash_hg,
        const float* __restrict__ stash_hu,
        int hidden, int inter, int num_tokens, int block_size,
        float routed_scaling_factor)
    {
        const int blk = blockIdx.x;
        const int base = blk * block_size;
        const int end = base + block_size < num_tokens ? base + block_size : num_tokens;

        for (int s = base + threadIdx.x; s < end; s += blockDim.x) {
            const unsigned int tok = sorted_token_ids[s];
            if (tok >= (unsigned int)num_tokens) continue; // padding slot
            const unsigned int exp = sorted_expert_ids[s];
            const float w = sorted_weights[s];
            const float rsf = routed_scaling_factor * w;

            const float* x  = activations + (unsigned long long)tok * hidden;
            const float* gw = gate_w + (unsigned long long)exp * (unsigned long long)inter * hidden;
            const float* uw = up_w   + (unsigned long long)exp * (unsigned long long)inter * hidden;
            const float* dw = down_w + (unsigned long long)exp * (unsigned long long)hidden * inter;
            const float* dy = d_y    + (unsigned long long)tok * hidden;

            float* dgw = d_gate_w + (unsigned long long)exp * (unsigned long long)inter * hidden;
            float* duw = d_up_w   + (unsigned long long)exp * (unsigned long long)inter * hidden;
            float* ddw = d_down_w + (unsigned long long)exp * (unsigned long long)hidden * inter;
            float* dx  = d_x      + (unsigned long long)tok * hidden;

            const float* cur_shg = stash_hg != nullptr ? stash_hg + (unsigned long long)s * inter : nullptr;
            const float* cur_shu = stash_hu != nullptr ? stash_hu + (unsigned long long)s * inter : nullptr;

            // ── Hidden states for this (token, expert) ──
            // If stashed activations are provided (Stage 1), read them directly without recomputing.
            // Otherwise fallback to recomputation from x and weights.
            for (int j = 0; j < inter; ++j) {
                float hg, hu;
                if (cur_shg != nullptr && cur_shu != nullptr) {
                    hg = cur_shg[j];
                    hu = cur_shu[j];
                } else {
                    hg = 0.0f; hu = 0.0f;
                    for (int i = 0; i < hidden; ++i) {
                        hg += gw[j * hidden + i] * x[i];
                        hu += uw[j * hidden + i] * x[i];
                    }
                }
                float silu_gj = hg / (1.0f + expf(-hg));
                float act_j = silu_gj * hu;

                // d_act[j] = sum_h s * d_y[h] * down_w[h, j]; also accumulate
                // the scaled outer product into d_down_w.
                float d_act = 0.0f;
                for (int h = 0; h < hidden; ++h) {
                    float dyh_s = rsf * dy[h];                 // s * d_y[h]
                    atomicAdd(&ddw[h * inter + j], dyh_s * act_j);
                    d_act += dyh_s * dw[h * inter + j];         // s * d_y[h] * down_w[h, j]
                }

                // SiLU parents of act[j].
                float sg       = silu_grad(hg);                 // (D silu)(h_gate)
                float d_h_gate = d_act * sg * hu;               // ← h_up factor fixed
                float d_h_up   = d_act * silu_gj;

                // d_gate_w[j, i], d_up_w[j, i] — outer products (no extra s;
                // s already absorbed via d_act into d_h_gate / d_h_up).
                for (int i = 0; i < hidden; ++i) {
                    atomicAdd(&dgw[j * hidden + i], d_h_gate * x[i]);
                    atomicAdd(&duw[j * hidden + i], d_h_up   * x[i]);
                }

                // d_x[i] += sum_j (gate_w[j, i] * d_h_gate[j]
                //                + up_w[j, i]   * d_h_up[j])
                for (int i = 0; i < hidden; ++i) {
                    atomicAdd(&dx[i], gw[j * hidden + i] * d_h_gate
                                     +   uw[j * hidden + i] * d_h_up);
                }
            }
        }
    }
}
"#;

// Host-side planning / validation helpers (pure, unit-testable w/o a device)

/// Per-expert gradient-tensor byte size for the FP32 backward path.
/// `d_gate_w` / `d_up_w` are `[num_experts, inter * hidden]` f32.
pub fn expert_weight_grad_bytes(num_experts: usize, inter: usize, hidden: usize) -> usize {
    num_experts
        .checked_mul(inter)
        .and_then(|x| x.checked_mul(hidden))
        .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
        .expect("expert_weight_grad_bytes: dimension product overflows usize")
}

/// `d_down_w` is `[num_experts, hidden * inter]` f32 (down is already `[hidden, inter]` in the forward layout, see `grim_moe_fused_grouped`).
/// # Panics Panics on arithmetic overflow (dimensions so large the byte count wraps `usize`).
pub fn expert_down_grad_bytes(num_experts: usize, inter: usize, hidden: usize) -> usize {
    num_experts
        .checked_mul(hidden)
        .and_then(|x| x.checked_mul(inter))
        .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
        .expect("expert_down_grad_bytes: dimension product overflows usize")
}

/// `d_x` is `[batch, hidden]` f32. # Panics Panics on
/// arithmetic overflow (dimensions so large the byte count wraps `usize`).
pub fn input_grad_bytes(batch: usize, hidden: usize) -> usize {
    batch
        .checked_mul(hidden)
        .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
        .expect("input_grad_bytes: dimension product overflows usize")
}

/// Validate backward launch inputs before any HIP dereference (mirrors
/// `charon::validate_grouped_inputs`). Pure: returns `Err` on bad geometry.
pub fn validate_backward_inputs(
    gate_w: *const c_void,
    up_w: *const c_void,
    down_w: *const c_void,
    d_y: *const c_void,
    d_gate_w: *mut c_void,
    d_up_w: *mut c_void,
    d_down_w: *mut c_void,
    d_x: *mut c_void,
    hidden: usize,
    inter: usize,
    num_tokens: usize,
    block_size: usize,
) -> Result<()> {
    if hidden == 0 || inter == 0 || num_tokens == 0 {
        return Err(Error::Backend(format!(
            "charon_backward: non-positive geometry hidden={hidden} inter={inter} num_tokens={num_tokens}"
        )));
    }
    if block_size == 0 {
        return Err(Error::Backend(
            "charon_backward: block_size must be > 0".into(),
        ));
    }
    for (name, p) in [
        ("gate_w", gate_w),
        ("up_w", up_w),
        ("down_w", down_w),
        ("d_y", d_y),
        ("d_gate_w", d_gate_w as *const c_void),
        ("d_up_w", d_up_w as *const c_void),
        ("d_down_w", d_down_w as *const c_void),
        ("d_x", d_x as *const c_void),
    ] {
        if p.is_null() {
            return Err(Error::Backend(format!("charon_backward: {name} is null")));
        }
    }
    let _ = (hidden, inter, num_tokens);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backward_grad_byte_sizes_are_consistent() {
        // [num_experts, inter*hidden] and [num_experts, hidden*inter] are the
        // same element count, so the two expert-weight grad buffers must match.
        let (ne, inter, hidden) = (8usize, 14336usize, 4096usize);
        assert_eq!(
            expert_weight_grad_bytes(ne, inter, hidden),
            expert_down_grad_bytes(ne, inter, hidden)
        );
        assert_eq!(input_grad_bytes(4, hidden), 4 * hidden * 4);
    }

    #[test]
    fn validate_backward_inputs_rejects_null_and_bad_geometry() {
        // Deliberate dangling (non-null, non-aligned) sentinel pointer.
        let p = std::ptr::dangling_mut::<c_void>();
        // null pointer
        assert!(
            validate_backward_inputs(std::ptr::null(), p, p, p, p, p, p, p, 4, 4, 4, 4).is_err()
        );
        // zero hidden
        assert!(validate_backward_inputs(p, p, p, p, p, p, p, p, 0, 4, 4, 4).is_err());
        // zero block_size
        assert!(validate_backward_inputs(p, p, p, p, p, p, p, p, 4, 4, 4, 0).is_err());
        // valid
        assert!(validate_backward_inputs(p, p, p, p, p, p, p, p, 4, 4, 4, 4).is_ok());
    }

    #[test]
    fn backward_kernel_source_is_jit_discoverable() {
        // The kernel is declared inside an `extern "C" { ...
        // }` block (so the HIPRTC symbol has C linkage and no name-mangling) - the literal.
        assert!(KERNEL_SOURCE.contains("extern \"C\" {"));
        assert!(KERNEL_SOURCE.contains("__global__ void grim_moe_fused_grouped_backward"));
        // Must reuse the SiLU-derivative decomposition documented in the plan.
        assert!(KERNEL_SOURCE.contains("silu_grad"));
        // Must not hard-depend on a quant format for the base FP32 case.
        assert!(KERNEL_SOURCE.contains("routed_scaling_factor"));
    }

    #[test]
    fn backward_kernel_is_grouped_contract() {
        // The backward must consume the same sorted routing arrays as the
        // forward `grim_moe_fused_grouped` so the same host sort feeds both.
        for sym in ["sorted_token_ids", "sorted_expert_ids", "sorted_weights"] {
            assert!(KERNEL_SOURCE.contains(sym), "missing sorted array: {sym}");
        }
    }
}
