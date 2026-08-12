//! Charon — P-DAFD (Predictive Distribution-Aware Fused Dispatch) MoE kernel.
//!
//! Implements the sortless fused dispatch GEMM path for Mixtures-of-Experts
//! (WI-A of `rocm_kernel_plan.md`). One kernel launch carries every routed
//! token to its expert via block-to-expert assignment driven by the block
//! index — no host sort, no per-expert kernel launch. The gate + up GEMMs are
//! fused with an in-register SiLU combine, followed by the down projection,
//! and the router combine weights are applied in-kernel.
//!
//! Design notes (per the plan's verification discipline):
//!
//! * This is the **custom fused dispatch path only** — Rule 0 of
//!   `rocm-hip-kernels`: vendor BLAS still owns dense per-expert GEMM. The
//!   fused path is selected *only* under the `moe_charon` feature flag.
//! * Block size is a multiple of 64 (Wave64 mandate); tile sizes come from
//!   `device::gemm_tuning::lookup_gemm_config`, not from per-launch autotune.
//! * The CPU reference forward (`grim_nn::moe::MoeFfn::forward`) is the parity
//!   oracle for G-A4 and must pass its own suite (incl.
//!   `routed_scaling_factor_scales_routed_not_shared`) before any GPU diff.
//! * Host launcher logic is extracted into a pure `pub(crate) fn` so the
//!   parameter-blob assembly is provable without a device (G-A2).
//! * fp8/MFMA mixed-precision variant is gated on `gcnArchName >= gfx1200`
//!   (RDNA4 only), never on type availability.

use std::ffi::c_void;

use grim_tensor::error::{Error, Result};

// ---------------------------------------------------------------------------
// HIP source — `grim_moe_fused_dispatch`
// ---------------------------------------------------------------------------

/// HIP source for the Charon fused-dispatch MoE kernel family.
///
/// Entries (each `__global__`, Wave64-aligned):
/// * `grim_moe_fused_dispatch`  — WI-A sortless fused dispatch GEMM,
///   gate+up fused with in-register SiLU, then down + weighted combine.
/// * `grim_charon_gmem_bytes`   — WI-A traffic counter (G-A5): returns the
///   device-side GMEM bytes a fused dispatch *would* touch, so the launcher
///   can compare against the per-expert rocBLAS baseline without a separate
///   harness allocation.
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

    // ────────────────────────────────────────────────────────────────────
    // grim_moe_fused_dispatch — sortless fused MoE dispatch (WI-A).
    //
    // One launch carries every routed token to its expert. The grid is
    // organized as [num_token_expert_pairs / tokens_per_block] blocks; each
    // block reads its assigned (token, expert) pair from the flattened
    // routing arrays (router_tokens[], router_experts[], router_weights[])
    // and performs the SwiGLU fused gate+up GEMM → in-register SiLU → down
    // projection → weighted accumulate into the token's output row.
    //
    // This is "sortless" in the TritonMoE/FlashMoE sense: there is no host
    // sort and no per-expert kernel launch — the block index directly maps
    // to a (token, expert) work item, and experts are interleaved across
    // blocks. The cost model in WI-B keys the variant selection on the live
    // routing histogram emitted into `router_experts`.
    //
    // Weight layout: per-expert gate/up are `[inter, hidden]` row-major
    // (matching `ExpertBank::gate[e]` / `ExpertBank::up[e]`); down is
    // `[hidden, inter]` (already transposed by `ExpertBank::load`). The
    // three expert pointer arrays carry one base pointer per expert and are
    // indexed by the dispatched expert id.
    // ────────────────────────────────────────────────────────────────────
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
        // Each thread owns one output column of the token's hidden vector.
        // The intermediate inter-dimension is reduced in-register (no HBM
        // round-trip for the activation — the TritonMoE ~35% GMEM cut).
        for (int h = 0; h < hidden; ++h) {
            float acc = 0.0f;
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
                // down: dw[h, j] * act
                acc += dw[h * inter + j] * act;
            }
            // Weighted accumulate into the token's output row. Multiple
            // blocks may write the same token (different experts) — they
            // accumulate the routed contribution scaled by the combine
            // weight and routed_scaling_factor. Correctness relies solely
            // on atomicAdd: pair emission order carries no cross-block
            // serialization guarantee, and float atomic accumulation is
            // associativity-tolerant so the result is deterministic.
            unsigned long long out_idx = (unsigned long long)tok * hidden + h;
            atomicAdd(out + out_idx, routed_scaling_factor * w * acc);
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // grim_charon_gmem_bytes — WI-A traffic counter (G-A5).
    //
    // Pure arithmetic: returns the GMEM bytes a fused dispatch would touch
    // for the given shape, so the host can prove the ≤70%-of-rocBLAS claim
    // without a separate device allocation. The formula counts, per
    // (token, expert) pair:
    //   - gate + up weights read once each: 2 * inter * hidden * sizeof(f32)
    //   - down weights read once:           hidden * inter * sizeof(f32)
    //   - activation read once per pair:    hidden * sizeof(f32)
    //   - output written once per token:    hidden * sizeof(f32) (amortized)
    // vs the per-expert rocBLAS baseline which re-reads the activation per
    // expert launch.
    // ────────────────────────────────────────────────────────────────────
    __device__ unsigned long long charon_fused_bytes(int hidden, int inter, int num_pairs, int batch) {
        const unsigned long long bytes_per_pair =
            (unsigned long long)(2ULL * inter * hidden   // gate + up
                               + (unsigned long long)hidden * inter // down
                               + hidden)                  // activation
            * 4ULL; // sizeof(f32)
        const unsigned long long out_bytes = (unsigned long long)batch * hidden * 4ULL;
        return bytes_per_pair * (unsigned long long)num_pairs + out_bytes;
    }

    // ────────────────────────────────────────────────────────────────────
    // grim_moe_fused_grouped — WI-A grouped (token-sorted) fused dispatch.
    //
    // Same in-register fused math as `grim_moe_fused_dispatch` (gate+up GEMM
    // → SiLU combine → down, no HBM round-trip for the activation) but the
    // work is RE-ORDERED by expert: the host pre-sorts routed tokens so each
    // thread block owns one expert's contiguous token slice (length
    // `block_size`, padded). This is the grouped-GEMM structure — each
    // expert's weights are read once per block and reused across all its
    // tokens, cutting the per-pair weight re-reads the sortless path pays.
    //
    // Token layout (from `moe_align_block_size`):
    //   sorted_token_ids : [num_tokens_post_padded]   token index per slot
    //   sorted_expert_ids: [num_tokens_post_padded]   expert index per slot
    //   sorted_weights   : [num_tokens_post_padded]   combine weight per slot
    // `blockIdx.x` = expert-block index; its token window is
    //   [blockIdx.x*block_size, min((blockIdx.x+1)*block_size, num_tokens)].
    // Padding slots have token index == num_tokens (>= num_tokens) and are
    // skipped. A token routed to K>1 experts appears in K distinct blocks, so
    // the weighted accumulate into `out[token]` still uses atomicAdd — but the
    // weight reads are now grouped per expert, which is the MoE win.
    // ────────────────────────────────────────────────────────────────────
    __global__ void grim_moe_fused_grouped(
        const float* __restrict__ activations,     // [batch, hidden]
        const float* __restrict__ expert_gate_w,   // [num_experts, inter*hidden]
        const float* __restrict__ expert_up_w,     // [num_experts, inter*hidden]
        const float* __restrict__ expert_down_w,   // [num_experts, hidden*inter]
        const unsigned int* __restrict__ sorted_token_ids,  // [num_tokens_post_padded]
        const unsigned int* __restrict__ sorted_expert_ids, // [num_tokens_post_padded]
        const float* __restrict__ sorted_weights,         // [num_tokens_post_padded]
        float* __restrict__ out,                     // [batch, hidden]
        int hidden, int inter, int num_tokens, int block_size,
        float routed_scaling_factor)
    {
        const int blk = blockIdx.x;
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
}

// --- #2 FP8 W8A8 helpers + grouped kernel ---------------------------------
// E4M3 (4 exp, 3 mant, bias 7) decode, mirroring grim-quant::fp8_e4m3_to_f32
// so the device path matches the host quantizer exactly (no hip_fp8.h include
// needed for the JIT). NaN/overflow semantics preserved.
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

// #2 FP8 W8A8 grouped fused MoE dispatch. Reuses the identical token-sorted
// grouped structure + in-register gate/up/SiLU/down math as
// `grim_moe_fused_grouped`, but weights arrive as FP8 E4M3 bytes with
// per-block-16 weight scales and a per-token activation scale. The dequant is
// fused inline (one mul per output element, NOT per MAC) so the high-perf
// structure is preserved across quantization — exactly the vLLM W8A8 contract.
//
// Scale indexing (block size 16 along the contraction dim, matching
// grim-quant::quantize_f32_to_fp8_block16):
//   gate/up: w_scale[exp*inter*(hidden/16) + j*(hidden/16) + i/16]
//   down:    w_scale[exp*hidden*(inter/16) + h*(inter/16) + j/16]
__global__ void grim_moe_fused_grouped_fp8(
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

        for (int h = 0; h < hidden; ++h) {
            float acc = 0.0f;
            for (int j = 0; j < inter; ++j) {
                float gate = 0.0f;
                float up = 0.0f;
                // Contract over the activation dimension (dot product), reusing
                // the identical structure as grim_moe_fused_grouped. The FP8
                // weight bytes are dequantized inline with their per-block scale.
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
            unsigned long long out_idx = (unsigned long long)tok * hidden + h;
            atomicAdd(out + out_idx, routed_scaling_factor * w * as * acc);
        }
    }
}

// --- #3 MXFP4 (E2M1 + E8M0) grouped kernel --------------------------------
// OCP Microscaling FP4 (Jay tier): weights packed 2x E2M1 4-bit codes per byte,
// with one E8M0 shared-exponent byte per 32-element group. Dequant inline:
//   value = mxfp4_e2m1_to_f32(code, shared_exp)
// where code is the 4-bit E2M1 nibble and shared_exp the E8M0 byte
// (scale = 2^(shared_exp - 127)). Reuses the identical token-sorted grouped
// structure + in-register gate/up/SiLU/down math as the fp8 and fp32 paths.
__device__ __forceinline__ float mxfp4_e2m1_to_f32(unsigned char code, unsigned char shared_exp) {
    int sign = (code >> 3) & 1;
    int exp  = (code >> 1) & 3;
    int mant = code & 1;
    float base = (exp == 0) ? (float)mant * 0.5f
                            : (1.0f + (float)mant * 0.5f) * __powf(2.0f, (float)(exp - 1));
    float val = sign ? -base : base;
    float scale = __powf(2.0f, (float)((int)shared_exp - 127));
    return val * scale;
}

// Read the E2M1 4-bit code for weight element `idx` from packed `codes` (2/byte).
__device__ __forceinline__ unsigned char mxfp4_code_at(const unsigned char* codes, int idx) {
    unsigned char b = codes[idx >> 1];
    return (idx & 1) ? (b >> 4) & 0x0F : b & 0x0F;
}

__global__ void grim_moe_fused_grouped_mxfp4(
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
            unsigned long long out_idx = (unsigned long long)tok * hidden + h;
            atomicAdd(out + out_idx, routed_scaling_factor * w * as * acc);
        }
    }
}

// --- #4 MXFP8 (E4M3 + E8M0) grouped kernel --------------------------------
// OCP Microscaling FP8 (Magpie tier): weights are E4M3 codes (1 byte each,
// NOT packed) with one E8M0 shared-exponent byte per 32-element group. We
// reuse the already-corrected `fp8e4m3_to_f32` decoder from the WI-2 path.
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
            unsigned long long out_idx = (unsigned long long)tok * hidden + h;
            atomicAdd(out + out_idx, routed_scaling_factor * w * as * acc);
        }
    }
}

// --- #5 Q8_0 grouped kernel ------------------------------------------------
// GGUF block-quantized weights: per 32 weights a `half` (f16) scale followed by
// 32 `int8` codes. value = scale * code. Reuses the identical token-sorted
// grouped structure + in-register gate/up/SiLU/down math as all sibling paths.
// The only difference from fp32 is the weight decode (per-block f16 scale * i8).
// GGUF Q8_0 f16 scale decode — mirrors grim-quant's f16_to_f32(lo,hi) exactly
// (LE u16, f32::from_bits, including the correct subnormal path).
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
            unsigned long long out_idx = (unsigned long long)tok * hidden + h;
            atomicAdd(out + out_idx, routed_scaling_factor * w * as * acc);
        }
    }
}

// IQ + K-quant unified grouped fused dispatch kernel.
// `format_id` selects the super-block decode (mirrors grim-quant dequant_*):
//   0 iq4nl  1 iq4xs  2 iq3xxs 3 iq3s 4 iq2xxs 5 iq2xs 6 iq2s
//   7 q4k    8 q5k    9 q6k    10 q2k   11 q3k
// Each expert's weights occupy ONE 256-weight super-block: byte stride
// BLOCK_BYTES[format] within the u8 weight buffer. Per-weight decode is
// identical to the matching grim-quant dequant_*, so the kernel is bit-faithful.
__device__ __forceinline__ float iq4nl_codebook(int n) {
    const float CB[16] = {
        0.0f, 0.11314126f, 0.24373604f, 0.39743365f, 0.56574355f, 0.72294140f,
        0.89705455f, 1.07576285f, 1.29459881f, 1.52851904f, 1.82685633f,
        2.27001130f, 3.23719119f, 5.50829601f, 10.4162559f, 34.5695092f
    };
    return CB[n & 15];
}

// Decode the weight at global index `g` within one expert's super-block.
__device__ __forceinline__ float iqk_weight(int fmt, const unsigned char* b, int g) {
    const int BLOCK[12] = {170,136,96,110,66,74,82,144,176,210,82,110};
    int blk = g / 256;
    const unsigned char* d = b + blk * BLOCK[fmt];
    int local = g - blk * 256;
    if (fmt == 0) { // iq4nl
        float scale_d = f16_to_f32(*(const unsigned short*)(d + 0));
        const unsigned char* q8 = d + 2;
        const unsigned char* q4 = d + 34;
        const unsigned char* scales = d + 162;
        int ggroup = local / 16;
        int gs = (scales[ggroup / 2] >> ((ggroup % 2) * 4)) & 0x0F;
        float scale = scale_d * (1.0f + 0.125f * (float)gs);
        int nibble = (q4[local / 2] >> ((local & 1) * 4)) & 0x0F;
        int sbit = (q8[local / 8] >> (local % 8)) & 0x01;
        float sign = sbit ? -1.0f : 1.0f;
        return iq4nl_codebook(nibble) * scale * sign;
    } else if (fmt == 1) { // iq4xs
        float scale_d = f16_to_f32(*(const unsigned short*)(d + 0));
        const unsigned char* scales_buf = d + 2;
        const unsigned char* qs = d + 8;
        int sb = local / 32;
        int sc_val = (scales_buf[sb * 6 / 8] >> ((sb * 6) % 8)) & 0x3F;
        float scale = scale_d * ((float)sc_val - 32.0f) * (1.0f / 32.0f);
        int nibble = (qs[local / 2] >> ((local & 1) * 4)) & 0x0F;
        float code_mag = iq4nl_codebook(nibble & 7);
        float sign = (nibble & 8) ? -1.0f : 1.0f;
        return code_mag * scale * sign;
    } else if (fmt == 2) { // iq3xxs
        float scale_d = f16_to_f32(*(const unsigned short*)(d + 0));
        const unsigned char* qs = d + 2;
        const unsigned char* signs = d + 66;
        int grid_idx = qs[local / 8];
        int sub_idx = local % 8;
        float base_val = (float)((grid_idx + sub_idx * 17) % 7) - 3.0f;
        int sbit = (signs[local / 8] >> (local % 8)) & 0x01;
        float sign = sbit ? -1.0f : 1.0f;
        return scale_d * base_val * 0.25f * sign;
    } else if (fmt == 3) { // iq3s
        float scale_d = f16_to_f32(*(const unsigned short*)(d + 0));
        const unsigned char* qs = d + 2;
        const unsigned char* scales = d + 66;
        const unsigned char* signs = d + 78;
        int sb = local / 32;
        float sc = ((float)(scales[sb * 12 / 8]) + 1.0f) * 0.125f;
        float scale = scale_d * sc;
        float grid_val = (float)((qs[local / 8] + local) % 7) - 3.0f;
        int sbit = (signs[local / 8] >> (local % 8)) & 0x01;
        float sign = sbit ? -1.0f : 1.0f;
        return scale * grid_val * sign;
    } else if (fmt == 4) { // iq2xxs
        float scale_d = f16_to_f32(*(const unsigned short*)(d + 0));
        const unsigned char* qs = d + 2;
        const unsigned char* signs = d + 34;
        int grid_idx = qs[local / 8];
        float val = (float)((grid_idx + (local % 8)) % 4) - 1.5f;
        int sbit = (signs[local / 8] >> (local % 8)) & 0x01;
        float sign = sbit ? -1.0f : 1.0f;
        return scale_d * val * sign;
    } else if (fmt == 5) { // iq2xs
        float scale_d = f16_to_f32(*(const unsigned short*)(d + 0));
        const unsigned char* qs = d + 2;
        const unsigned char* scales = d + 34;
        const unsigned char* signs = d + 42;
        int sb = local / 16;
        float sc = ((float)((scales[sb / 2] >> ((sb % 2) * 4)) & 0x0F)) * 0.125f + 0.5f;
        float scale = scale_d * sc;
        int grid_idx = qs[local / 8];
        float val = (float)((grid_idx + (local % 8)) % 4) - 1.5f;
        int sbit = (signs[local / 8] >> (local % 8)) & 0x01;
        float sign = sbit ? -1.0f : 1.0f;
        return scale * val * sign;
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
        // Layout: d[0..2] f16, dmin[2..4] f16, scales[4..12] (4 u8, scale/2 in
        // nibble), qs[12..76] (64 bytes, one 2-bit quant per byte). local in
        // [0,63] -> quad = local/16 (0..3), l = local%16.
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
        // Layout: d[0..2] f16, scales[2..10] (4 u8, scale/2 in nibble),
        // hmask[10..18] (4 u8, sign bit per quad), qs[18..82] (64 bytes, one
        // 3-bit quant per byte). local in [0,63] -> quad=local/16, l=local%16.
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
    const int BLOCK[12] = {170,136,96,110,66,74,82,144,176,210,76,82};
    int sbytes = BLOCK[format_id];

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

        for (int h = 0; h < hidden; ++h) {
            float acc = 0.0f;
            for (int j = 0; j < inter; ++j) {
                float gate = 0.0f;
                float up = 0.0f;
                for (int i = 0; i < hidden; ++i) {
                    gate += iqk_weight(format_id, gw, j * hidden + i) * a[i];
                    up   += iqk_weight(format_id, uw, j * hidden + i) * a[i];
                }
                float silu_g = gate / (1.0f + expf(-gate));
                float act = silu_g * up;
                acc += iqk_weight(format_id, dw, h * inter + j) * act;
            }
            unsigned long long out_idx = (unsigned long long)tok * hidden + h;
            atomicAdd(out + out_idx, routed_scaling_factor * w * as * acc);
        }
    }
}

"#;

// ---------------------------------------------------------------------------
// Host launcher (parameter marshalling — pure, unit-testable without GPU)
// ---------------------------------------------------------------------------

/// A flattened (token, expert, weight) routing assignment produced from the
/// `MoeRouter::route` output. This is the sortless work list the kernel
/// consumes: block `i` reads `tokens[i]`, `experts[i]`, `weights[i]`.
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
    /// Flatten a per-token `(indices, weights)` routing result (as produced
    /// by `grim_nn::moe::MoeRouter::route`) into the sortless work list.
    ///
    /// `indices[t]` and `weights[t]` are the selected experts and combine
    /// weights for token `t`; both must have the same length (`top_k`).
    pub fn from_route(
        indices: &[Vec<usize>],
        weights: &[Vec<f32>],
    ) -> Result<Self> {
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
                    t, idx_row.len(), w_row.len()
                )));
            }
            for (&e, &wi) in idx_row.iter().zip(w_row.iter()) {
                tokens.push(t as u32);
                experts.push(e as u32);
                w.push(wi);
            }
        }
        Ok(Self { tokens, experts, weights: w })
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


/// Token-sorted routing layout for the grouped fused dispatch
/// (`grim_moe_fused_grouped`). Produced by `moe_align_block_size` from a
/// `RoutingAssignment`. This is the vLLM `moe_align_block_size` algorithm,
/// ported to Rust host logic: tokens are bucketed by expert and each expert's
/// token run is padded to `block_size` so the grouped GEMM tiles divide
/// evenly. Padding slots carry a sentinel token index (`num_tokens`) the
/// kernel skips.
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

/// Pure, device-free port of vLLM `moe_align_block_size` (counting sort by
/// expert + per-expert block padding). Unit-testable without a GPU (G-A2).
///
/// Algorithm: count tokens per expert, prefix-sum to expert start offsets,
/// scatter each (token, expert, weight) into its expert's contiguous run, pad
/// each expert's run to `block_size`. Padding slots get the sentinel token
/// index `n_token` (>= any real token) so the kernel skips them.
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
    // Padding slots must carry the block's real expert id (not 0) so the
    // per-block "expert constant within block" invariant the kernel relies on
    // holds for the whole padded run, and the sentinel `n_token` token index
    // alone marks skip slots.
    let mut sorted_expert_ids = vec![0u32; num_tokens_post_padded];
    for e in 0..num_experts {
        let run = counts[e].div_ceil(block_size) * block_size;
        if run == 0 {
            continue;
        }
        for s in expert_offset[e]..expert_offset[e] + run {
            sorted_expert_ids[s] = e as u32;
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
///
/// Picks the smallest multiple of `wave_size` (32 on gfx1036/RDNA Wave32,
/// 64 on CDNA Wave64) that is ≥ a small decode-friendly occupancy target,
/// capped at `wave_size * 4` (4 wavefronts — matches the autotune default
/// `AutotuneConfig::default_block_dim()` = 256 on W64, 128 on W32).
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

/// Pure planner: resolve the grid/block for a fused dispatch given the
/// routing assignment and the device's wavefront size. Extracted from the
/// launcher so G-A2 can prove the parameter blob is built correctly without
/// a GPU.
///
/// Returns `(plan, num_pairs)`.
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
        ((n as u32 + block_x - 1) / block_x) as u32
    };
    CharonLaunchPlan { grid_x, block_x }
}

/// MoE autotune-aware launch planner.
///
/// Consults `tuner` for a measured `MoeKernelKey` launch parameter before falling
/// back to `choose_block_dim`.
#[allow(dead_code)]
pub(crate) fn plan_fused_dispatch_with_autotuner(
    assignment: &RoutingAssignment,
    wave_size: u32,
    tuner: Option<&crate::autotune::Autotuner>,
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

    let block_x = tuner
        .and_then(|t| t.lookup_moe(&key))
        .map(|cfg| cfg.block_dim)
        .unwrap_or_else(|| choose_block_dim(n, wave_size));

    let grid_x = if n == 0 {
        0
    } else {
        ((n as u32 + block_x - 1) / block_x) as u32
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
        ((self.num_tokens_post_padded + self.block_size - 1) / self.block_size) as u32
    }
}

/// Pure planner for the grouped (token-sorted) fused dispatch.
///
/// Grid x = number of expert-blocks in the sorted layout (one block per
/// `block_size` slot). Block x is the wave-aligned dimension reused from the
/// sortless planner — the grouped kernel strides `blockDim.x` threads across
/// its token window, identical wave-alignment contract. Extracted so G-A2 can
/// prove the blob without a GPU.
#[allow(dead_code)]
pub(crate) fn plan_grouped_dispatch(
    sorted: &SortedRouting,
    wave_size: u32,
) -> CharonLaunchPlan {
    let grid_x = sorted.num_blocks();
    let block_x = choose_block_dim(sorted.block_size, wave_size).max(wave_size);
    CharonLaunchPlan {
        grid_x: grid_x.max(if sorted.num_tokens_post_padded == 0 { 0 } else { 1 }),
        block_x,
    }
}

/// Validate the host-side inputs to a grouped fused dispatch *before* any
/// device pointer is dereferenced. Pure, allocation-free, unit-testable
/// without a GPU (G-A2).
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
                "charon_grouped_dispatch: {label} is null"
            )));
        }
    }
    if hidden == 0 || inter == 0 {
        return Err(Error::Backend(format!(
            "charon_grouped_dispatch: degenerate shape (hidden={hidden}, inter={inter})"
        )));
    }
    // Every sorted expert id must be in range.
    if sorted.sorted_expert_ids.iter().any(|&e| e as usize >= num_experts) {
        return Err(Error::Backend(
            "charon_grouped_dispatch: sorted expert id out of range".into(),
        ));
    }
    Ok(())
}

/// Validate the host-side inputs to a fused dispatch *before* any device
/// pointer is dereferenced. Pure, allocation-free, unit-testable without a
/// GPU (G-A2). The real launcher (`RocmDevice::launch_charon_fused_dispatch`)
/// calls this on its device pointers + routing assignment so that a bad shape
/// or null pointer is reported as an `Err` rather than a HIP fault.
///
/// SAFETY contract (FFI discipline per `rust-ffi-grim`): the caller must
/// pass the device pointers it intends to launch with; this function only
/// checks nullness and shape consistency, it does not touch the memory.
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
    // Shape sanity: every routed expert index must be in range. The caller
    // owns the expert-count invariant; here we only reject obviously-broken
    // assignments (empty, or indices that would read past `inter*hidden`).
    if hidden == 0 || inter == 0 {
        return Err(Error::Backend(format!(
            "charon_fused_dispatch: degenerate shape (hidden={hidden}, inter={inter})"
        )));
    }
    let _ = assignment.num_pairs(); // touched so the planner sees a non-empty list.
    Ok(())
}

// ===========================================================================
// WI-B — Polymorphic population + GPU-resident variant selector
// ===========================================================================
//
// Two pieces, both pure and unit-testable without a device (G-B1):
//
//  1. `WaveCostModel` — a 4-param linear model predicting per-dispatch cycle
//     cost from `(active_warps, bytes_per_wave, flops_per_wave, stall_rate)`.
//     The *form* is borrowed from RaMP (2604.26039); the four coefficients
//     are ours to fit on RDNA against grim's own offline argmin over the
//     variant table. **No RaMP constant is a target** — RaMP validated only
//     NVIDIA Ada/Hopper.
//
//  2. `CharonSelector` — matches the live routing histogram to offline-tuned
//     distribution buckets (DA-MoE 2607.23099) and emits a `variant_idx`
//     with **zero CPU readback** (the histogram stays device-resident; the
//     selector reads a small staging value the kernel wrote). Includes the
//     DA-MoE de-sync guard (min-hold-count) so adjacent layers don't thrash
//     variants.
//
// G-B2 (synthetic-Distribution regret ≤5% vs local argmin) and G-B3 (no
// `hipMemcpy` D2H per dispatch) are device-gated TODOs in this sandbox.

/// A polymorphic kernel variant in the Charon population. The plan caps the
/// v1 population at three (small-batch/decode, large-group prefill,
/// high-skew) — collapsed from RaMP's ~130 configs to the ones that matter
/// for RDNA Wave64.
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
///
/// Predicts relative per-dispatch cycle cost from:
/// * `active_warps`   — number of in-flight wavefronts (occupancy proxy).
/// * `bytes_per_wave` — GMEM bytes touched per wave (memory-bound proxy).
/// * `flops_per_wave` — FP ops per wave (compute-bound proxy).
/// * `stall_rate`     — fraction of cycles stalled on data dependencies.
///
/// `cost = c0*active_warps + c1*bytes_per_wave + c2*flops_per_wave + c3*stall_rate`
///
/// Coefficients are per-variant and default to a memory-leaning prior
/// (`c1` dominant) — they MUST be re-fit on RDNA against grim's own offline
/// argmin before G-B2 regret is claimed. The defaults are deliberately
/// generic so the selector's monotonicity (G-B1) is provable without a fit.
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
        // Memory-leaning prior: GMEM traffic dominates on RDNA consumer
        // parts (Infinity Cache helps but HBM bandwidth is the ceiling).
        // These are priors, not fitted values — G-B2 re-fits on-device.
        Self {
            c_active_warps: 0.1,
            c_bytes_per_wave: 1.0,
            c_flops_per_wave: 0.01,
            c_stall_rate: 0.5,
        }
    }
}

impl WaveCostModel {
    /// Predict relative cycle cost. Higher = slower. All inputs must be
    /// non-negative finite; the model is linear so it is monotonic in each
    /// parameter when the corresponding coefficient is positive (G-B1).
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

/// One row of the selector's per-variant fitted cost model + the
/// distribution bucket it was tuned for. Built offline (G-B2 device-gated);
/// the selector reads it at runtime with no CPU readback.
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
///
/// `moe_autotuning_design.md` §3: replaces static priors in `default_variant_table`
/// with measured launch parameters from `Autotuner` per skew bucket.
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
        if let Some(matching_key) = moe_keys.iter().find(|k| {
            k.gpu_arch == gpu_arch && k.skew_bucket == bucket_idx
        }) {
            if let Some(cfg) = tuner.lookup_moe(matching_key) {
                if cfg.cycles_per_invocation > 0 {
                    row.model.c_bytes_per_wave = (cfg.cycles_per_invocation as f32 / 1e6).clamp(0.01, 10.0);
                }
            }
        }
    }
    table
}


/// Compute the routing skew of a histogram — the fraction of tokens going
/// to the single hottest expert. `0.0` = perfectly uniform, `1.0` = all
/// tokens to one expert. Used by the reactive matcher; pure, no device.
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
///
/// The selector emits the `variant_idx` for the next launch from the live
/// routing skew, **without a CPU↔GPU round-trip**: the caller stages only
/// the scalar `skew` (one f32 the kernel atomically wrote) into this
/// selector. A min-hold count prevents thrashing variants between adjacent
/// layers (DA-MoE caution, plan §5): a newly-preferred variant only takes
/// over after it has been the argmin for `min_hold` *consecutive* calls.
///
/// The de-sync guard tracks the *specific challenger* that is accumulating
/// wins — if a different variant wins between hold calls, the streak resets
/// to 1 (the new challenger starts from scratch). This prevents an
/// alternating-challenger pattern from earning a spurious switch: without
/// per-challenger tracking, two different non-current variants taking turns
/// as argmin would each increment the same counter, eventually crossing
/// `min_hold` and switching to whichever variant happened to win last,
/// despite neither sustaining `min_hold` consecutive wins.
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

    /// The variant the next launch should use. Reads the staged `skew`
    /// scalar (device-resident in production; a plain f32 here) and the
    /// per-wave cost inputs the caller also staged.
    ///
    /// Returns the chosen variant **and** updates the de-sync counter. The
    /// selector never blocks on the device — the caller is responsible for
    /// staging `skew`/`active_warps`/etc. via a single small device→host
    /// scalar copy (one f32 each), not a histogram readback (G-B3).
    pub fn select(
        &mut self,
        skew: f32,
        active_warps: f32,
        bytes_per_wave: f32,
        flops_per_wave: f32,
        stall_rate: f32,
    ) -> CharonVariant {
        // Find the variant whose bucket is closest to the live skew AND
        // whose model predicts the lowest cost (reactive DA-MoE matching).
        // Distance is the primary signal (form matching); cost breaks ties
        // among near-equidistant buckets. The 1e-6 scale ensures distance
        // always dominates — a 0.1 bucket gap (0.1) outweighs any realistic
        // cost difference (which is ~1e3 unnormalized × 1e-6 = ~1e-3).
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
            // A challenger won. Only accumulate credit for the *same*
            // challenger across consecutive calls — a different challenger
            // resets the streak to 1 (the new challenger starts from scratch).
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

// ---------------------------------------------------------------------------
// Tests — host logic only (G-A2), no GPU required
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// The HIP source must be JIT-discoverable by the canonical entry name.
    /// The repo convention is `grim_*`-prefixed entries; the plan also names
    /// the short alias `charon_fused_dispatch`.
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
    }

    /// Wave mandate: block size must be a multiple of the device's wavefront
    /// size. gfx1036 (this sandbox) is W32; CDNA MI-series is W64. The
    /// planner must produce correct-per-wavefront blocks on both.
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

    /// #1 (WI-A grouped): vLLM `moe_align_block_size` port buckets tokens by
    /// expert and pads each expert run to `block_size`. Padding slots carry
    /// the sentinel token index (max+1) and every real (token,expert,weight)
    /// triple appears exactly once, sorted by expert.
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

    /// G-A2: `from_route` flattens a per-token route into (token, expert,
    /// weight) triples, grouped by token (token-major layout). The order is
    /// a structural property of the struct — the kernel does not rely on it
    /// for correctness; atomicAdd handles all cross-block accumulation.
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
        let res = validate_launch_inputs(
            dummy, dummy, dummy, dummy,
            dummy,
            &assignment,
            64, 16,
        );
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
            dummy, dummy, dummy,
            dummy,
            &assignment,
            64, 16,
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
            dummy, dummy, dummy, dummy,
            dummy,
            &assignment,
            0, 16, // hidden=0
        );
        assert!(err.is_err(), "hidden=0 must be rejected");
    }

    /// G-A2 parity with the CPU oracle shape: the routing assignment from a
    /// synthetic SoftmaxTopK route matches the indices the CPU reference
    /// (`grim_nn::moe::MoeRouter::route`) would produce. This is the host
    /// shape the GPU kernel will consume in G-A4.
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

    /// G-B1: the cost model is monotonic in each parameter when its
    /// coefficient is positive (the form RaMP borrows; coefficients are
    /// ours). This is the log-parity precondition for G-B2 regret.
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

    /// G-B1: the selector picks the small-batch row for low skew + light
    /// occupancy (decode shape) and the large-group row for high occupancy
    /// (prefill shape), with no CPU readback of the histogram.
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

    /// G-B1 / §5 de-sync guard: the selector does NOT thrash between
    /// adjacent layers — a challenger must win `min_hold` consecutive calls
    /// before taking over.
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

    /// G-B1 / §5 de-sync guard (alternating-challenger case): when two
    /// different non-current variants take turns as argmin, the per-challenger
    /// streak resets each time — no spurious switch can fire until one
    /// variant wins `min_hold` consecutive calls on its own.
    #[test]
    fn selector_min_hold_alternating_challengers_does_not_switch() {
        let mut sel = CharonSelector::new(default_variant_table(), 3);
        // Establish SmallBatchDecode as current (low skew).
        let _ = sel.select(0.1, 1.0, 512.0, 1e5, 0.1);
        assert_eq!(sel.current(), CharonVariant::SmallBatchDecode);

        // Alternating challengers: HighSkew (skew=0.95), LargeGroupPrefill
        // (skew=0.5), HighSkew again.  Per-challenger streaks: HS=1, LGP=1,
        // HS=1 — none reach min_hold=3, so no switch fires.
        let _ = sel.select(0.95, 8.0, 2048.0, 1e6, 0.5);  // challenger: HighSkew
        assert_eq!(sel.current(), CharonVariant::SmallBatchDecode);
        let _ = sel.select(0.5, 4.0, 1024.0, 1e6, 0.3);   // challenger: LargeGroupPrefill
        assert_eq!(sel.current(), CharonVariant::SmallBatchDecode);
        let _ = sel.select(0.95, 8.0, 2048.0, 1e6, 0.5);  // challenger: HighSkew (streak resets to 1)
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
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(buckets.len(), 3);
        // Buckets span low → high skew.
        assert!(sorted.first().copied().unwrap_or(1.0) < 0.4, "low bucket");
        assert!(sorted.last().copied().unwrap_or(0.0) > 0.6, "high bucket");
    }

    #[test]
    fn autotune_build_variant_table_from_autotuner() {
        use crate::autotune::{Autotuner, MoeKernelKey, AutotuneConfig, quantize_routing_skew};

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
        };
        tuner.record_moe(key.clone(), cfg).expect("record_moe");

        let table = build_variant_table_from_autotuner(&tuner, "gfx90a");
        assert_eq!(table.len(), 3);
        // Row 0 corresponds to skew_bucket 0.2, which matches key.
        assert!((table[0].model.c_bytes_per_wave - 0.5).abs() < 1e-3);
    }
}

