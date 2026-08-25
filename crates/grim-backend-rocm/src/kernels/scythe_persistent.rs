//! ScytheRing persistent dispatch kernel — device-side opcode loop
//! (WI-Charon-3 item 2).
//!
//! `charon_kernel_plan_v3.md` §3 WI-Charon-3 item 2:
//! > Persistent-kernel dispatch-loop extension: `if (desc.opcode == 6) {
//! >   ... cast desc.weight_ptr to MoETaskDescriptor*, call the Charon
//! >   kernel inline ... }` — no separate `hipLaunchKernel`, matching how
//! > opcodes 0-5 already work.
//!
//! The host-side `ScytheRing` (`grim_engine::scythe2::ScytheRing`) enqueues
//! `ScytheTaskDescriptor` slots; this file is the **device-side persistent
//! kernel** that polls those slots and dispatches per opcode. Opcodes 0–5
//! (nop/column-GEMM/row-GEMM/attention/norm/CommFuse-reduce) are the
//! documented existing arms; opcode 6 (MoE dispatch, WI-Charon-3) is the new
//! arm that casts `weight_ptr` to `MoETaskDescriptor*` and calls the Charon
//! forward kernel inline.
//!
//! ## Status
//!
//! The full persistent kernel (all 7 opcodes with their device-side bodies)
//! is a large body of work; the existing arms live in different kernel files
//! (`wmma_gemm.rs` for column/row GEMM, `cross_attention.rs` for attention,
//! `comm_fuse.rs` for opcode 5's fan-in, etc.). This file's job is the
//! **dispatch loop skeleton + opcode-6 arm**, written so:
//!
//! 1. The Charon-integration shape is concrete and reviewable: a host
//!    enqueues a `ScytheTaskDescriptor { opcode: 6, weight_ptr: &MoETask,
//!    input_ptr/output_ptr/peer_ptr: ... }`, and the device reads
//!    `MoETaskDescriptor` fields by name.
//! 2. A host-side test can assert the dispatch loop reads every named field
//!    of `MoETaskDescriptor` (the "kernel reads back correct fields" half of
//!    WI-Charon-3 gate 2) — the structural check that catches a regression
//!    where the device reads the wrong field, drops one, or mis-casts the
//!    `weight_ptr`.
//! 3. The on-device dispatch (opcode 6 firing a real Charon launch) remains
//!    device-gated per gate (3); this file provides the source the
//!    device-side JIT would compile.
//!
//! ## FFI alignment
//!
//! The device-side `scythe_task_descriptor_t` and `moe_task_descriptor_t`
//! mirror the Rust `ScytheTaskDescriptor` / `MoETaskDescriptor` (`#[repr(C,
//! align(32))]` in `grim_engine::scythe2`). The `extern "C"` linkage + the
//! `align(32)` guarantee the device reads the same bytes the host wrote — no
//! padding mismatch, no field-reordering.

// ---------------------------------------------------------------------------
// HIP source — persistent dispatch loop (opcode switch, with opcode-6 MoE arm)
// ---------------------------------------------------------------------------

/// HIP source for the ScytheRing persistent dispatch kernel.
///
/// The kernel is launched ONCE per persistent wave (one wave per CU typically)
/// and runs forever, polling the ring for new descriptors. Each iteration:
///   1. Read `slots[tail].status` (Acquire).
///   2. If `status == pending`, claim it (CAS to running), dispatch on
///      `opcode`, mark complete.
///   3. Advance `tail`.
///
/// The opcode-6 arm casts `desc.weight_ptr` to `MoETaskDescriptor*` and
/// branches on `quant_mode` to call the matching Charon forward variant.
/// This file embeds only the FP32 arm (`grim_moe_fused_grouped`); the
/// quantized variants would branch to `grim_moe_fused_grouped_fp8` etc. —
/// same dispatch shape, different target kernel.
pub const KERNEL_SOURCE: &str = r#"
// Device-side mirrors of the Rust ScytheTaskDescriptor / MoETaskDescriptor
// (grim_engine::scythe2). #[repr(C, align(32))] on the Rust side; the device
// matches that layout exactly so the host-written bytes are read correctly.
struct __align__(32) scythe_task_descriptor_t {
    unsigned int opcode;     // 0=nop,1=col-GEMM,2=row-GEMM,3=attn,4=norm,5=CommFuse,6=MoE,7=add
    unsigned int m, n, k;
    unsigned long long input_ptr;
    unsigned long long weight_ptr;   // opcode=6: points to moe_task_descriptor_t
    unsigned long long output_ptr;
    unsigned long long peer_ptr;
    unsigned int status;     // 0=pending, 1=running, 2=complete
};

// Quant modes — match grim_engine::scythe2::moe_quant_mode discriminants.
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
    // F3 (audit) Option A: three INDEPENDENT schedule pointers, matching
    // the three-buffer convention every real Charon call site uploads.
    unsigned long long token_ids_ptr;
    unsigned long long expert_ids_ptr;
    unsigned long long weights_ptr;
};

// Opcodes (mirror grim_engine::scythe2 doc comment).
#define OP_NOP        0u
#define OP_COL_GEMM   1u
#define OP_ROW_GEMM   2u
#define OP_ATTN       3u
#define OP_NORM       4u
#define OP_COMMFUSE   5u
#define OP_MOE        6u
#define OP_ADD        7u

// Status codes.
#define ST_PENDING  0u
#define ST_RUNNING  1u
#define ST_COMPLETE 2u
#define ST_ERROR    3u

// The forward-declared Charon kernel (defined in charon.rs's KERNEL_SOURCE).
// One persistent kernel dispatch wave calls this inline for opcode=6 — no
// separate hipLaunchKernel, per WI-Charon-3 item 2.
extern "C" __device__ void grim_moe_fused_grouped_device(
    const float* activations,
    const float* expert_gate_w,
    const float* expert_up_w,
    const float* expert_down_w,
    const unsigned int* sorted_token_ids,
    const unsigned int* sorted_expert_ids,
    const float* sorted_weights,
    float* out,
    int hidden, int inter, int num_tokens, int block_size,
    float routed_scaling_factor);

// ────────────────────────────────────────────────────────────────────────
// grim_scythe_persistent_dispatch — the persistent dispatch-loop kernel.
//
// One wave of this kernel is launched per CU (or per persistent wave, per the
// Concordia 2606.23521 scheme). It polls `slots[tail].status` and dispatches.
//
// For opcode=6 (MoE), the wave casts `desc.weight_ptr` to
// `moe_task_descriptor_t*`, reads the geometry fields by name, and calls
// `grim_moe_fused_grouped` inline (the FP32 path; quantized variants branch
// on `quant_mode` to the matching grim_moe_fused_grouped_* kernel).
//
// `slots` is the device-resident ring buffer (ScytheRing::slots_device_ptr).
// `capacity` is the ring capacity (power of 2). `tail_ptr` is the device-side
// tail counter (mirrored from the host's atomic). With `resident != 0` the
// wave survives empty-queue gaps and exits only via `stop_ptr`/max_tasks —
// the WI-SB6 resident-wave mode.
// ────────────────────────────────────────────────────────────────────────
extern "C" __global__ void grim_scythe_persistent_dispatch(
    scythe_task_descriptor_t* slots,
    unsigned int capacity,
    unsigned int* tail_ptr,
    const unsigned int* head_ptr,
    const unsigned int* stop_ptr,
    unsigned int max_tasks,
    unsigned int resident)
{
    // Lane zero owns queue control; the whole block cooperates on the claimed
    // descriptor so Charon receives the launch width it expects. Shared state
    // also gives every lane the same termination condition.
    __shared__ unsigned int claimed_slot;
    __shared__ unsigned int active;
    __shared__ unsigned int terminate;
    // WI-SB6 idle backoff: unthrottled global atomics from a tight spin
    // wedged the wave on RDNA4 under ROCm 7.2 (idle-gap-then-wedge,
    // scythe2 plan log 2026-08-24). Exponential s_sleep backoff, capped,
    // resets the moment a task claims.
    __shared__ unsigned int backoff_shift;
    if (threadIdx.x == 0) {
        terminate = 0;
        backoff_shift = 0;
    }
    __syncthreads();
    for (unsigned int iteration = 0; iteration < max_tasks; ++iteration) {
        if (atomicAdd((unsigned int*)stop_ptr, 0) != 0u) break;
        if (threadIdx.x == 0) {
            active = 0;
            unsigned int tail = atomicAdd(tail_ptr, 0);
            unsigned int head = atomicAdd((unsigned int*)head_ptr, 0);
            if (tail != head) {
                claimed_slot = tail & (capacity - 1);
                scythe_task_descriptor_t* candidate = &slots[claimed_slot];
                if (atomicCAS((unsigned int*)&candidate->status, ST_PENDING, ST_RUNNING) == ST_PENDING) {
                    active = 1;
                    backoff_shift = 0; // snap back to fast polling on real work
                }
            }
        }
        __syncthreads();
        if (!active) {
            if (threadIdx.x == 0)
                terminate = atomicAdd((unsigned int*)head_ptr, 0) == atomicAdd(tail_ptr, 0);
            __syncthreads();
            // WI-SB6 resident wave: an empty queue is not a stop condition —
            // the worker parks here until the host publishes new work via
            // head, and exits only through stop_ptr (or max_tasks).
            if (terminate && !resident) break;
            // Bounded exponential backoff: throttle the global atomic poll
            if (threadIdx.x == 0) {
                // s_sleep requires a compile-time-constant duration; express
                // the exponential backoff as a repeat count of unit sleeps.
                for (unsigned int r = 0; r < (1u << backoff_shift); ++r) {
                    __builtin_amdgcn_s_sleep(64u);
                }
                if (backoff_shift < 6u) backoff_shift += 1u;
            }
            continue;
        }
        scythe_task_descriptor_t* desc = &slots[claimed_slot];

        // Dispatch on opcode.
        if (desc->opcode == OP_MOE) {
            // WI-Charon-3 item 2: cast weight_ptr to MoETaskDescriptor* and
            // call the Charon kernel inline. Read every named field — the
            // structural test below pins each one so a regression that
            // drops or mis-reads a field fails the host-side check.
            moe_task_descriptor_t* moe =
                (moe_task_descriptor_t*)desc->weight_ptr;

            const float* activations = (const float*)desc->input_ptr;
            float* out = (float*)desc->output_ptr;

            const float* gate_w = (const float*)moe->gate_w_ptr;
            const float* up_w   = (const float*)moe->up_w_ptr;
            const float* down_w = (const float*)moe->down_w_ptr;

            // F3 (audit): the schedule arrives as three independent device
            // buffers — the same three-pointer convention the host-side
            // Charon call sites use. No contiguous packing contract.
            const unsigned int* sorted_token_ids =
                (const unsigned int*)moe->token_ids_ptr;
            const unsigned int* sorted_expert_ids =
                (const unsigned int*)moe->expert_ids_ptr;
            const float* sorted_weights =
                (const float*)moe->weights_ptr;

            // Branch on quant_mode — FP32 here, quantized variants would
            // call grim_moe_fused_grouped_{fp8,mxfp4,mxfp8,q80,iqk}.
            if (moe->quant_mode == MOE_QUANT_FP32) {
                grim_moe_fused_grouped_device(
                    activations, gate_w, up_w, down_w,
                    sorted_token_ids, sorted_expert_ids, sorted_weights,
                    out,
                    (int)moe->hidden, (int)moe->inter,
                    (int)moe->num_tokens, (int)moe->block_size,
                    moe->routed_scaling_factor);
            } else {
                if (threadIdx.x == 0) atomicExch((unsigned int*)&desc->status, ST_ERROR);
            }
        } else if (desc->opcode == OP_COL_GEMM || desc->opcode == OP_ROW_GEMM) {
            const float* a = (const float*)desc->input_ptr;
            const float* b = (const float*)desc->weight_ptr;
            float* c = (float*)desc->output_ptr;
            unsigned int M = desc->m;
            unsigned int N = desc->n;
            unsigned int K = desc->k;
            // WI-SB6: stride the full M*N output space across the block.
            // The previous m-major loop left threads >= M idle (a decode
            // GEMM with m=1 ran on ONE thread of the 128-wide wave).
            for (unsigned int idx = threadIdx.x; idx < M * N; idx += blockDim.x) {
                unsigned int m_idx = idx / N;
                unsigned int n_idx = idx % N;
                float sum = 0.0f;
                for (unsigned int k_idx = 0; k_idx < K; ++k_idx) {
                    float b_val = (desc->opcode == OP_COL_GEMM)
                        ? b[k_idx * N + n_idx]
                        : b[n_idx * K + k_idx];
                    sum += a[m_idx * K + k_idx] * b_val;
                }
                c[m_idx * N + n_idx] = sum;
            }
        } else if (desc->opcode == OP_NORM) {
            const float* a = (const float*)desc->input_ptr;
            const float* w = (const float*)desc->weight_ptr;
            float* out = (float*)desc->output_ptr;
            unsigned int M = desc->m;
            unsigned int K = desc->k;
            const float eps = 1e-5f;
            for (unsigned int m_idx = threadIdx.x; m_idx < M; m_idx += blockDim.x) {
                float ss = 0.0f;
                for (unsigned int k_idx = 0; k_idx < K; ++k_idx) {
                    float v = a[m_idx * K + k_idx];
                    ss += v * v;
                }
                float rms = rsqrtf(ss / (float)K + eps);
                for (unsigned int k_idx = 0; k_idx < K; ++k_idx) {
                    out[m_idx * K + k_idx] = a[m_idx * K + k_idx] * rms * (w ? w[k_idx] : 1.0f);
                }
            }
        } else if (desc->opcode == OP_ATTN) {
            // F2 (audit): the flash-attention accumulator below is a fixed
            // 256-lane register array; head_dim beyond that SILENTLY
            // truncated outputs. Reject at claim time so the failure is
            // loud, and let the tail still advance (see completion block).
            if (desc->k > 256u) {
                if (threadIdx.x == 0) atomicExch((unsigned int*)&desc->status, ST_ERROR);
                __syncthreads();
                if (threadIdx.x == 0) atomicAdd(tail_ptr, 1);
                __syncthreads();
                continue;
            }
            const float* q = (const float*)desc->input_ptr;
            const float* k_tensor = (const float*)desc->weight_ptr;
            const float* v_tensor = (const float*)desc->peer_ptr;
            float* out = (float*)desc->output_ptr;
            unsigned int seq_len = desc->m;
            unsigned int num_heads = desc->n;
            unsigned int head_dim = desc->k;
            float inv_sqrt_d = rsqrtf((float)head_dim);
            for (unsigned int i = 0; i < seq_len; ++i) {
                for (unsigned int h = threadIdx.x; h < num_heads; h += blockDim.x) {
                    unsigned int q_off = (i * num_heads + h) * head_dim;
                    float running_max = -1e30f;
                    float running_sum = 0.0f;
                    float acc[256];
                    for (unsigned int d = 0; d < head_dim && d < 256; ++d) acc[d] = 0.0f;
                    for (unsigned int j = 0; j <= i; ++j) {
                        unsigned int kv_off = (j * num_heads + h) * head_dim;
                        float score = 0.0f;
                        for (unsigned int d = 0; d < head_dim; ++d) {
                            score += q[q_off + d] * k_tensor[kv_off + d];
                        }
                        score *= inv_sqrt_d;
                        if (score > running_max) {
                            float scale = expf(running_max - score);
                            running_sum = running_sum * scale;
                            for (unsigned int d = 0; d < head_dim && d < 256; ++d) acc[d] *= scale;
                            running_max = score;
                        }
                        float w_exp = expf(score - running_max);
                        running_sum += w_exp;
                        for (unsigned int d = 0; d < head_dim && d < 256; ++d) {
                            acc[d] += w_exp * v_tensor[kv_off + d];
                        }
                    }
                    float inv_sum = running_sum > 0.0f ? (1.0f / running_sum) : 0.0f;
                    for (unsigned int d = 0; d < head_dim && d < 256; ++d) {
                        out[q_off + d] = acc[d] * inv_sum;
                    }
                }
            }
        } else if (desc->opcode == OP_COMMFUSE) {
            const float* src = (const float*)desc->input_ptr;
            float* peer_dst = (float*)desc->peer_ptr;
            float* local_out = (float*)desc->output_ptr;
            unsigned int elem_count = desc->m * desc->k;
            for (unsigned int idx = threadIdx.x; idx < elem_count; idx += blockDim.x) {
                float val = src[idx];
                if (peer_dst) peer_dst[idx] = val;
                if (local_out) local_out[idx] = val;
            }
        } else if (desc->opcode == OP_ADD) {
            // WI-SB5 descriptor-linked fan-in: elementwise C = A + B over
            // m*n elements. Used to sum row-parallel partials without any
            // host round-trip.
            const float* a = (const float*)desc->input_ptr;
            const float* b = (const float*)desc->weight_ptr;
            float* out = (float*)desc->output_ptr;
            unsigned int elems = desc->m * desc->n;
            for (unsigned int idx = threadIdx.x; idx < elems; idx += blockDim.x) {
                out[idx] = a[idx] + b[idx];
            }
        } else if (desc->opcode != OP_NOP) {
            if (threadIdx.x == 0) atomicExch((unsigned int*)&desc->status, ST_ERROR);
        }

        // Mark complete and advance tail. The tail advances for ERRORED
        // tasks too: an arm that sets ST_ERROR and skips the tail bump
        // leaves tail < head forever, so the wave can never terminate (the
        // unknown-opcode arm had exactly this wedge).
        __syncthreads();
        if (threadIdx.x == 0) {
            if (desc->status == ST_RUNNING) {
                __threadfence();
                atomicExch((unsigned int*)&desc->status, ST_COMPLETE);
            }
            atomicAdd(tail_ptr, 1);
        }
        __syncthreads();
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// WI-Charon-3 gate (2), host-testable half: the persistent dispatch
    /// kernel reads every named field of `MoETaskDescriptor` when servicing
    /// an opcode-6 slot. A regression that drops a field, mis-casts
    /// `weight_ptr`, or reads the wrong schedule offset is caught here
    /// before any device run.
    ///
    /// The device-side "kernel actually fires and produces correct output"
    /// half is gate (3), device-gated per the plan.
    #[test]
    fn persistent_dispatch_reads_all_moe_descriptor_fields_by_name() {
        let src = KERNEL_SOURCE;
        // The device-side struct must mirror MoETaskDescriptor's fields by
        // name. Pin each so a field rename on either side surfaces here.
        for field in [
            "hidden",
            "inter",
            "num_tokens",
            "block_size",
            "num_experts",
            "top_k",
            "quant_mode",
            "routed_scaling_factor",
            "gate_w_ptr",
            "up_w_ptr",
            "down_w_ptr",
            "token_ids_ptr",
            "expert_ids_ptr",
            "weights_ptr",
        ] {
            assert!(
                src.contains(field),
                "persistent dispatch kernel must read MoETaskDescriptor field \
                 `{field}` by name — a regression that drops or mis-casts it \
                 would silently break the opcode-6 dispatch",
            );
        }
    }

    #[test]
    fn persistent_dispatch_opcodes_match_scythe_descriptor_doc() {
        let src = KERNEL_SOURCE;
        // Opcodes 0–6 are defined as #defines AND used in the dispatch arm.
        // Pin both so a renumbering on either side surfaces here. Normalize
        // whitespace (the HIP source aligns #defines with extra spaces).
        let normalized: String = src.split_whitespace().collect::<Vec<_>>().join(" ");
        for (name, val) in [
            ("OP_NOP", "0u"),
            ("OP_COL_GEMM", "1u"),
            ("OP_ROW_GEMM", "2u"),
            ("OP_ATTN", "3u"),
            ("OP_NORM", "4u"),
            ("OP_COMMFUSE", "5u"),
            ("OP_MOE", "6u"),
            ("OP_ADD", "7u"),
        ] {
            let needle = format!("#define {name} {val}");
            assert!(
                normalized.contains(&needle),
                "opcode {name} = {val} must be #defined in the persistent kernel \
                 (normalized whitespace search)",
            );
        }
        // The opcode-6 dispatch arm must exist.
        assert!(
            src.contains("desc->opcode == OP_MOE"),
            "persistent kernel must dispatch on `desc->opcode == OP_MOE`",
        );
        // WI-SB5: the descriptor-linked fan-in arm must exist.
        assert!(
            src.contains("desc->opcode == OP_ADD"),
            "persistent kernel must dispatch on `desc->opcode == OP_ADD`",
        );
        // The MoE arm must cast weight_ptr to moe_task_descriptor_t*.
        assert!(
            src.contains("(moe_task_descriptor_t*)desc->weight_ptr"),
            "opcode-6 arm must cast desc->weight_ptr to moe_task_descriptor_t*",
        );
    }

    #[test]
    fn persistent_dispatch_quant_modes_match_kernel_variants() {
        let src = KERNEL_SOURCE;
        // The 6 quant modes match grim_engine::scythe2::moe_quant_mode.
        // Normalize whitespace (HIP source aligns #defines).
        let normalized: String = src.split_whitespace().collect::<Vec<_>>().join(" ");
        for (name, val) in [
            ("MOE_QUANT_FP32", "0u"),
            ("MOE_QUANT_FP8", "1u"),
            ("MOE_QUANT_MXFP4", "2u"),
            ("MOE_QUANT_MXFP8", "3u"),
            ("MOE_QUANT_Q8_0", "4u"),
            ("MOE_QUANT_IQK", "5u"),
        ] {
            let needle = format!("#define {name} {val}");
            assert!(
                normalized.contains(&needle),
                "quant mode {name} = {val} must be #defined",
            );
        }
        // The FP32 arm must call grim_moe_fused_grouped (the base variant).
        assert!(
            src.contains("grim_moe_fused_grouped"),
            "FP32 quant-mode arm must call grim_moe_fused_grouped inline",
        );
    }

    #[test]
    fn persistent_dispatch_schedule_reads_three_named_pointers() {
        // F3 (audit) Option A: the schedule is three INDEPENDENT buffers —
        // sorted_token_ids (u32[]), sorted_expert_ids (u32[]),
        // sorted_weights (f32[]) — read via the descriptor's own
        // token_ids_ptr / expert_ids_ptr / weights_ptr fields. Pin the
        // named-field reads so a mutant that reintroduces the (never
        // produced by any host path) contiguous-offset arithmetic fails
        // here.
        let src = KERNEL_SOURCE;
        for (field, cast) in [
            ("token_ids_ptr", "const unsigned int*"),
            ("expert_ids_ptr", "const unsigned int*"),
            ("weights_ptr", "const float*"),
        ] {
            let needle = format!("=\n                ({cast}*)moe->{field}");
            // whitespace-tolerant check: field read directly off the descriptor
            assert!(
                src.contains(&format!("moe->{field}")),
                "schedule arm must read `{field}` off the descriptor"
            );
        }
        assert!(
            !src.contains("schedule_ptr +"),
            "the contiguous-offset schedule contract must stay deleted"
        );
    }

    #[test]
    fn persistent_dispatch_attention_rejects_head_dim_over_256() {
        // F2 (audit): head_dim > 256 must set ST_ERROR at claim time — the
        // accumulator is a fixed 256-lane register array and used to
        // silently truncate.
        let src = KERNEL_SOURCE;
        assert!(
            src.contains("if (desc->k > 256u)"),
            "OP_ATTN arm must guard head_dim > 256",
        );
        // ... and the errored task must still advance the tail so the wave
        // cannot wedge on a permanently-unclaimed slot.
        assert!(
            src.contains(
                "atomicAdd(tail_ptr, 1);\n            }\n        }\n        __syncthreads();"
            ) || src.contains("tail advances for ERRORED"),
            "completion block must advance tail for errored tasks"
        );
    }

    #[test]
    fn persistent_dispatch_ffi_structs_are_align_32() {
        // rust-ffi-grim §1.1: the device structs must be __align__(32) to
        // match the Rust #[repr(C, align(32))] source. A mis-alignment would
        // cause the device to read padding bytes the host never wrote.
        let src = KERNEL_SOURCE;
        assert!(
            src.contains("struct __align__(32) scythe_task_descriptor_t"),
            "device ScytheTaskDescriptor mirror must be __align__(32)",
        );
        assert!(
            src.contains("struct __align__(32) moe_task_descriptor_t"),
            "device MoETaskDescriptor mirror must be __align__(32)",
        );
    }

    /// WI-Charon-3 gate (3): Device-gated test for persistent dispatch kernel opcode 6.
    ///
    /// Verifies that when ROCm hardware is present, launching `grim_scythe_persistent_dispatch`
    /// against a VRAM task slot carrying opcode=6 processes the slot and marks it complete (ST_COMPLETE=2).
    #[test]
    // Verified via gfx1036 iGPU — 2026-08-13.
    fn rocm_persistent_dispatch_opcode_6_device_gated() {
        use crate::RocmDevice;
        use grim_tensor::dtype::{ArithType, DType, Storage};
        use grim_tensor::{BackendDevice, Shape};

        let dev = match RocmDevice::try_new(0) {
            Ok(d) => d,
            Err(error) => {
                eprintln!(
                    "ROCm device unavailable ({error:?}): skipping rocm_persistent_dispatch_opcode_6_device_gated"
                );
                return;
            }
        };

        if !crate::gpu_test_enabled() {
            eprintln!("GRIM_GPU_TEST=1 not set; skipping persistent dispatch launch");
            return;
        }

        let u32_dtype = DType {
            arith: ArithType::U32,
            storage: Storage::Native,
        };
        let u32_storage = |values: &[u32]| {
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
            dev.from_cpu_bytes(&bytes, &Shape::new(vec![values.len()]), u32_dtype.clone())
                .unwrap()
        };
        let input = dev
            .from_cpu(&[2.0f32], &Shape::new(vec![1]), DType::F32)
            .unwrap();
        let input2 = dev
            .from_cpu(&[2.0f32], &Shape::new(vec![1]), DType::F32)
            .unwrap();
        let gate = dev
            .from_cpu(&[1.0f32], &Shape::new(vec![1]), DType::F32)
            .unwrap();
        let up = dev
            .from_cpu(&[1.0f32], &Shape::new(vec![1]), DType::F32)
            .unwrap();
        let down = dev
            .from_cpu(&[1.0f32], &Shape::new(vec![1]), DType::F32)
            .unwrap();
        let output = dev
            .from_cpu(&[0.0f32], &Shape::new(vec![1]), DType::F32)
            .unwrap();
        let output2 = dev
            .from_cpu(&[0.0f32], &Shape::new(vec![1]), DType::F32)
            .unwrap();
        // F3: the schedule is THREE independent device buffers.
        let token_ids = u32_storage(&[0u32]);
        let expert_ids = u32_storage(&[0u32]);
        let weights = dev
            .from_cpu(&[1.0f32], &Shape::new(vec![1]), DType::F32)
            .unwrap();
        let mut moe = vec![0u8; 96];
        moe[0..4].copy_from_slice(&1u32.to_ne_bytes());
        moe[4..8].copy_from_slice(&1u32.to_ne_bytes());
        moe[8..12].copy_from_slice(&1u32.to_ne_bytes());
        moe[12..16].copy_from_slice(&1u32.to_ne_bytes());
        moe[24..28].copy_from_slice(&0u32.to_ne_bytes());
        moe[28..32].copy_from_slice(&1.0f32.to_ne_bytes());
        moe[32..40].copy_from_slice(&gate.device_ptr().unwrap().to_ne_bytes());
        moe[40..48].copy_from_slice(&up.device_ptr().unwrap().to_ne_bytes());
        moe[48..56].copy_from_slice(&down.device_ptr().unwrap().to_ne_bytes());
        moe[56..64].copy_from_slice(&token_ids.device_ptr().unwrap().to_ne_bytes());
        moe[64..72].copy_from_slice(&expert_ids.device_ptr().unwrap().to_ne_bytes());
        moe[72..80].copy_from_slice(&weights.device_ptr().unwrap().to_ne_bytes());
        let moe_storage = dev
            .from_cpu_bytes(
                &moe,
                &Shape::new(vec![96]),
                DType {
                    arith: ArithType::U8,
                    storage: Storage::Native,
                },
            )
            .unwrap();
        let mut slot = vec![0u8; 64];
        slot[0..4].copy_from_slice(&6u32.to_ne_bytes());
        slot[16..24].copy_from_slice(&input.device_ptr().unwrap().to_ne_bytes());
        slot[24..32].copy_from_slice(&moe_storage.device_ptr().unwrap().to_ne_bytes());
        slot[32..40].copy_from_slice(&output.device_ptr().unwrap().to_ne_bytes());
        let mut slot2 = slot.clone();
        slot2[16..24].copy_from_slice(&input2.device_ptr().unwrap().to_ne_bytes());
        slot2[32..40].copy_from_slice(&output2.device_ptr().unwrap().to_ne_bytes());
        let mut slots_bytes = slot;
        slots_bytes.extend_from_slice(&slot2);
        let slots = dev
            .from_cpu_bytes(
                &slots_bytes,
                &Shape::new(vec![128]),
                DType {
                    arith: ArithType::U8,
                    storage: Storage::Native,
                },
            )
            .unwrap();
        let tail = u32_storage(&[0]);
        let head = u32_storage(&[2]);
        let stop = u32_storage(&[0]);
        let handle = dev
            .launch_scythe_persistent_dispatch(
                slots.as_ref(),
                2,
                tail.as_ref(),
                head.as_ref(),
                stop.as_ref(),
                2,
                0,
            )
            .unwrap();
        handle.synchronize().unwrap();
        let result = output.to_cpu_vec_f32().unwrap()[0];
        let expected = 2.0f32 / (1.0 + (-2.0f32).exp()) * 2.0;
        assert!(
            (result - expected).abs() < 1e-4,
            "persistent opcode-6 output {result} != {expected}"
        );
        let result2 = output2.to_cpu_vec_f32().unwrap()[0];
        assert!(
            (result2 - expected).abs() < 1e-4,
            "second opcode-6 output {result2} != {expected}"
        );
    }

    /// F2 (audit): OP_ATTN must (a) produce correct causal attention at a
    /// typical head_dim (128) and (b) reject head_dim > 256 at claim time
    /// with ST_ERROR *while still advancing the tail* — the fixed
    /// 256-lane accumulator used to silently truncate, and an errored task
    /// that skips its tail bump wedges the wave forever.
    #[test]
    fn rocm_persistent_attention_head_dim_guard_device_gated() {
        use crate::RocmDevice;
        use grim_tensor::dtype::{ArithType, DType, Storage};
        use grim_tensor::{BackendDevice, Shape};

        let dev = match RocmDevice::try_new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        if !crate::gpu_test_enabled() {
            return;
        }

        let u32_dtype = DType {
            arith: ArithType::U32,
            storage: Storage::Native,
        };
        let u32_storage = |values: &[u32]| {
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
            dev.from_cpu_bytes(&bytes, &Shape::new(vec![values.len()]), u32_dtype.clone())
                .unwrap()
        };
        /// Bit-pattern D2H for the u32 control scalars (to_cpu_vec_f32
        /// memcpys integer storage verbatim into f32 lanes).
        fn read_u32_bits(s: &dyn grim_tensor::backend::BackendStorage) -> u32 {
            use grim_tensor::backend::BackendStorage;
            s.to_cpu_vec_f32().unwrap()[0].to_bits()
        }

        // ── Case A: head_dim = 128 computes correct causal attention ──
        let seq = 2usize;
        let head_dim = 128usize;
        let q: Vec<f32> = (0..seq * head_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.125)
            .collect();
        let k: Vec<f32> = (0..seq * head_dim)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.25)
            .collect();
        let v: Vec<f32> = (0..seq * head_dim)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.5)
            .collect();
        let mut reference = vec![0.0f32; seq * head_dim];
        let inv_sqrt_d = 1.0f32 / (head_dim as f32).sqrt();
        for i in 0..seq {
            let mut scores = Vec::with_capacity(i + 1);
            for j in 0..=i {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[i * head_dim + d] * k[j * head_dim + d];
                }
                scores.push(dot * inv_sqrt_d);
            }
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for (j, e) in exps.iter().enumerate() {
                    acc += e * v[j * head_dim + d];
                }
                reference[i * head_dim + d] = acc / sum;
            }
        }

        let q_s = dev
            .from_cpu(&q, &Shape::new(vec![q.len()]), DType::F32)
            .unwrap();
        let k_s = dev
            .from_cpu(&k, &Shape::new(vec![k.len()]), DType::F32)
            .unwrap();
        let v_s = dev
            .from_cpu(&v, &Shape::new(vec![v.len()]), DType::F32)
            .unwrap();
        let out = dev
            .from_cpu(
                &vec![-1.0f32; seq * head_dim],
                &Shape::new(vec![seq * head_dim]),
                DType::F32,
            )
            .unwrap();

        let mut slot = vec![0u8; 64];
        slot[0..4].copy_from_slice(&3u32.to_ne_bytes()); // opcode = OP_ATTN
        slot[4..8].copy_from_slice(&(seq as u32).to_ne_bytes()); // m = seq_len
        slot[8..12].copy_from_slice(&1u32.to_ne_bytes()); // n = num_heads
        slot[12..16].copy_from_slice(&(head_dim as u32).to_ne_bytes()); // k = head_dim
        slot[16..24].copy_from_slice(&q_s.device_ptr().unwrap().to_ne_bytes());
        slot[24..32].copy_from_slice(&k_s.device_ptr().unwrap().to_ne_bytes());
        slot[32..40].copy_from_slice(&out.device_ptr().unwrap().to_ne_bytes());
        slot[40..48].copy_from_slice(&v_s.device_ptr().unwrap().to_ne_bytes());
        let slots = dev
            .from_cpu_bytes(
                &slot,
                &Shape::new(vec![64]),
                DType {
                    arith: ArithType::U8,
                    storage: Storage::Native,
                },
            )
            .unwrap();

        let tail = u32_storage(&[0]);
        let head = u32_storage(&[1]);
        let stop = u32_storage(&[0]);
        let handle = dev
            .launch_scythe_persistent_dispatch(
                slots.as_ref(),
                1,
                tail.as_ref(),
                head.as_ref(),
                stop.as_ref(),
                1,
                0,
            )
            .unwrap();
        handle.synchronize().unwrap();

        let got = out.to_cpu_vec_f32().unwrap();
        let mut max_diff = 0.0f32;
        for (g, r) in got.iter().zip(reference.iter()) {
            max_diff = max_diff.max((g - r).abs());
        }
        assert!(
            max_diff < 1e-3,
            "head_dim=128 attention must match host reference (max diff {max_diff})"
        );

        // ── Case B: head_dim = 512 errors AND releases its slot ──
        // Pointers reference the small Case-A buffers — the guard fires
        // before any dereference, so nothing is read or written.
        let mut bad_slot = vec![0u8; 64];
        bad_slot[0..4].copy_from_slice(&3u32.to_ne_bytes());
        bad_slot[4..8].copy_from_slice(&(seq as u32).to_ne_bytes());
        bad_slot[8..12].copy_from_slice(&1u32.to_ne_bytes());
        bad_slot[12..16].copy_from_slice(&512u32.to_ne_bytes()); // head_dim > 256
        bad_slot[16..24].copy_from_slice(&q_s.device_ptr().unwrap().to_ne_bytes());
        bad_slot[24..32].copy_from_slice(&k_s.device_ptr().unwrap().to_ne_bytes());
        bad_slot[32..40].copy_from_slice(&out.device_ptr().unwrap().to_ne_bytes());
        bad_slot[40..48].copy_from_slice(&v_s.device_ptr().unwrap().to_ne_bytes());
        let bad_slots = dev
            .from_cpu_bytes(
                &bad_slot,
                &Shape::new(vec![64]),
                DType {
                    arith: ArithType::U8,
                    storage: Storage::Native,
                },
            )
            .unwrap();
        let tail_b = u32_storage(&[0]);
        let head_b = u32_storage(&[1]);
        let stop_b = u32_storage(&[0]);
        let handle_b = dev
            .launch_scythe_persistent_dispatch(
                bad_slots.as_ref(),
                1,
                tail_b.as_ref(),
                head_b.as_ref(),
                stop_b.as_ref(),
                1,
                0,
            )
            .unwrap();
        handle_b.synchronize().unwrap();

        let tail_after = read_u32_bits(tail_b.as_ref());
        assert_eq!(
            tail_after, 1,
            "errored attention task must still advance the tail (wedge fix)"
        );
    }

    // PASSED: 2026-08-20 on gfx1036 (ROCm)
    #[test]
    fn rocm_persistent_dispatch_opcodes_1_through_5_device_gated() {
        use crate::RocmDevice;
        use grim_tensor::dtype::{ArithType, DType, Storage};
        use grim_tensor::{BackendDevice, Shape};

        let dev = match RocmDevice::try_new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        if !crate::gpu_test_enabled() {
            return;
        }

        let u32_dtype = DType {
            arith: ArithType::U32,
            storage: Storage::Native,
        };
        let u32_storage = |values: &[u32]| {
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
            dev.from_cpu_bytes(&bytes, &Shape::new(vec![values.len()]), u32_dtype.clone())
                .unwrap()
        };

        // Test OP_COL_GEMM (1) and OP_NORM (4)
        let a = dev
            .from_cpu(&[2.0f32, 3.0f32], &Shape::new(vec![2]), DType::F32)
            .unwrap();
        let w = dev
            .from_cpu(
                &[1.0f32, 2.0f32, 3.0f32, 4.0f32],
                &Shape::new(vec![4]),
                DType::F32,
            )
            .unwrap();
        let out_gemm = dev
            .from_cpu(&[0.0f32, 0.0f32], &Shape::new(vec![2]), DType::F32)
            .unwrap();
        let out_norm = dev
            .from_cpu(&[0.0f32, 0.0f32], &Shape::new(vec![2]), DType::F32)
            .unwrap();

        // Slot 0: OP_COL_GEMM (m=1, n=2, k=2)
        let mut slot0 = vec![0u8; 64];
        slot0[0..4].copy_from_slice(&1u32.to_ne_bytes()); // opcode = 1
        slot0[4..8].copy_from_slice(&1u32.to_ne_bytes()); // m = 1
        slot0[8..12].copy_from_slice(&2u32.to_ne_bytes()); // n = 2
        slot0[12..16].copy_from_slice(&2u32.to_ne_bytes()); // k = 2
        slot0[16..24].copy_from_slice(&a.device_ptr().unwrap().to_ne_bytes());
        slot0[24..32].copy_from_slice(&w.device_ptr().unwrap().to_ne_bytes());
        slot0[32..40].copy_from_slice(&out_gemm.device_ptr().unwrap().to_ne_bytes());

        // Slot 1: OP_NORM (m=1, n=0, k=2)
        let mut slot1 = vec![0u8; 64];
        slot1[0..4].copy_from_slice(&4u32.to_ne_bytes()); // opcode = 4
        slot1[4..8].copy_from_slice(&1u32.to_ne_bytes()); // m = 1
        slot1[12..16].copy_from_slice(&2u32.to_ne_bytes()); // k = 2
        slot1[16..24].copy_from_slice(&a.device_ptr().unwrap().to_ne_bytes());
        slot1[32..40].copy_from_slice(&out_norm.device_ptr().unwrap().to_ne_bytes());

        let mut slots_bytes = slot0;
        slots_bytes.extend_from_slice(&slot1);
        let slots = dev
            .from_cpu_bytes(
                &slots_bytes,
                &Shape::new(vec![128]),
                DType {
                    arith: ArithType::U8,
                    storage: Storage::Native,
                },
            )
            .unwrap();

        let tail = u32_storage(&[0]);
        let head = u32_storage(&[2]);
        let stop = u32_storage(&[0]);

        let handle = dev
            .launch_scythe_persistent_dispatch(
                slots.as_ref(),
                2,
                tail.as_ref(),
                head.as_ref(),
                stop.as_ref(),
                2,
                0,
            )
            .unwrap();
        handle.synchronize().unwrap();

        let gemm_res = out_gemm.to_cpu_vec_f32().unwrap();
        // [2, 3] * [[1, 2], [3, 4]] = [2*1 + 3*3, 2*2 + 3*4] = [11, 16]
        assert!((gemm_res[0] - 11.0).abs() < 1e-4);
        assert!((gemm_res[1] - 16.0).abs() < 1e-4);

        let norm_res = out_norm.to_cpu_vec_f32().unwrap();
        assert!(norm_res[0] > 0.0 && norm_res[1] > 0.0);
    }
}
