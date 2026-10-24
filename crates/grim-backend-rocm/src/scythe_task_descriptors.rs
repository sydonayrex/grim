//! Scythe task descriptors relocated from `grim-engine::scythe2` so the
//! backend (and model loaders) can reach them without depending on the engine.
//! `grim-engine` re-exports these for source compatibility.

use grim_tensor::MemoryOps;

// ── ScytheRing + ScytheTaskDescriptor (WI-7) ─────────────────────────────────

/// 32-byte task descriptor for the lock-free VRAM ring (scythe2.md §5.3).
/// The device-resident persistent kernel (Concordia `2606.23521`) polls these descriptors at HBM bandwidth.
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, Default)]
pub struct ScytheTaskDescriptor {
    /// Operation code (see above).
    pub opcode: u32,
    /// GEMM M dimension (or seq_len for attention).
    pub m: u32,
    /// GEMM N dimension (or num_heads for attention).
    pub n: u32,
    /// GEMM K dimension (or head_dim for attention).
    pub k: u32,
    /// Device pointer to input activations (u64 for 64-bit pointers).
    pub input_ptr: u64,
    /// Device pointer to weight shard.
    pub weight_ptr: u64,
    /// Device pointer to output buffer (local).
    pub output_ptr: u64,
    /// Device pointer to peer VRAM target (T0 fused-P2P, 0 = local only).
    pub peer_ptr: u64,
    /// Slot status: 0 = pending, 1 = running, 2 = complete.
    pub status: u32,
    // Pad to 32 bytes (already 44 bytes with u64×4 + u32×5 = 52; using 52 → next align(32) = 64).
    // Since ScytheTaskDescriptor is repr(C, align(32)) and 52 bytes, the struct occupies 64 bytes (rounded up.
}

// ── WI-Charon-3: MoE task descriptor + opcode 6 ────────────────────────────── The plan (charon_kernel_plan_v3.md §3 WI-Charon-3) calls for a companion descriptor carrying the MoE-specific geometry the generic `ScytheTaskDescriptor` (m/n/k + 4 pointers) can't express: hidden dim, inter dim, batch count, routed-scaling factor, expert-bank pointers, the sorted-routing schedule, and quant mode.
// The integration point is ONE new opcode (6 = MoE dispatch) on the existing `ScytheRing`.

/// Quantization mode for a Charon MoE dispatch (WI-Charon-3).
/// Mirrors the 7 forward kernel variants in `charon.rs`: `grim_moe_fused_dispatch` (FP32) + 6 quantized variants.
pub type MoeQuantMode = u32;

/// MoE quant modes (WI-Charon-3). Values pinned by tests so the kernel and
/// host agree on the discriminants.
pub mod moe_quant_mode {
    /// FP32 / BF16-as-f32 path (`grim_moe_fused_dispatch`).
    pub const FP32: super::MoeQuantMode = 0;
    /// FP8 E4M3 (`grim_moe_fused_grouped_fp8`).
    pub const FP8: super::MoeQuantMode = 1;
    /// MXFP4 (`grim_moe_fused_grouped_mxfp4`).
    pub const MXFP4: super::MoeQuantMode = 2;
    /// MXFP8 (`grim_moe_fused_grouped_mxfp8`).
    pub const MXFP8: super::MoeQuantMode = 3;
    /// Q8_0 (`grim_moe_fused_grouped_q80`).
    pub const Q8_0: super::MoeQuantMode = 4;
    /// IQK family (`grim_moe_fused_grouped_iqk`).
    pub const IQK: super::MoeQuantMode = 5;
}

/// MoE-specific geometry companion to `ScytheTaskDescriptor` (WI-Charon-3).
/// The host enqueues a `ScytheTaskDescriptor { opcode: 6, weight_ptr: ptr to MoETaskDescriptor, input_ptr/output_ptr/peer_ptr: as usual,.
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug)]
pub struct MoETaskDescriptor {
    /// Hidden dim (`hidden`).
    pub hidden: u32,
    /// Inter dim (`inter`).
    pub inter: u32,
    /// Number of tokens post-padding (`num_tokens_post_padded`).
    pub num_tokens: u32,
    /// Block size used by the grouped sort (`block_size` from
    /// `moe_align_block_size`).
    pub block_size: u32,
    /// Number of experts in the bank.
    pub num_experts: u32,
    /// Top-k routing count.
    pub top_k: u32,
    /// Quantization mode (see [`moe_quant_mode`]).
    pub quant_mode: MoeQuantMode,
    /// Routed scaling factor (DeepSeek/Laguna convention — scales routed,
    /// not shared). f32 to match the kernel's `float` argument.
    pub routed_scaling_factor: f32,
    /// Device pointer to the expert gate weights, flattened
    /// as `[num_experts, inter*hidden]` (row-major, the layout `grim_moe_fused_grouped` expects).
    pub gate_w_ptr: u64,
    /// Device pointer to the expert up weights, `[num_experts, inter*hidden]`.
    pub up_w_ptr: u64,
    /// Device pointer to the expert down weights, `[num_experts, hidden*inter]`.
    pub down_w_ptr: u64,
    /// Device pointer to `sorted_token_ids` (u32[num_tokens_post_padded]) - F3 (audit) Option A: the routing schedule is carried as THREE INDEPENDENT pointers, matching the convention every
    /// real Charon call site in `roc_device.rs` already uploads (the previous single `schedule_ptr` + contiguous-offset contract matched no producer and had no host-side packing step).
    pub token_ids_ptr: u64,
    /// Device pointer to `sorted_expert_ids` (u32[num_tokens_post_padded]).
    pub expert_ids_ptr: u64,
    /// Device pointer to `sorted_weights` (f32[num_tokens_post_padded]).
    pub weights_ptr: u64,
}

impl Default for MoETaskDescriptor {
    fn default() -> Self {
        Self {
            hidden: 0,
            inter: 0,
            num_tokens: 0,
            block_size: 0,
            num_experts: 0,
            top_k: 0,
            quant_mode: moe_quant_mode::FP32,
            routed_scaling_factor: 1.0,
            gate_w_ptr: 0,
            up_w_ptr: 0,
            down_w_ptr: 0,
            token_ids_ptr: 0,
            expert_ids_ptr: 0,
            weights_ptr: 0,
        }
    }
}

impl MoETaskDescriptor {
    /// Upload this descriptor to device-visible memory on `device` and return the device address the caller MUST pass to [`Self::enqueue_via`].
    /// F4 (audit): `enqueue_via` used to stuff `self as *const Self as u64` - a HOST.
    pub fn upload(&self, device: &crate::RocmDevice) -> crate::Result<u64> {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                std::mem::size_of::<Self>(),
            )
        };
        let storage = device.from_cpu_bytes(
            bytes,
            &grim_tensor::Shape::new(vec![std::mem::size_of::<Self>()]),
            grim_tensor::dtype::DType {
                arith: grim_tensor::ArithType::U8,
                storage: grim_tensor::dtype::Storage::Native,
            },
        )?;
        let ptr = storage
            .as_any()
            .downcast_ref::<crate::RocmStorage>()
            .and_then(|rs| rs.device_ptr_u64())
            .ok_or_else(|| crate::Error::Backend("MoE upload: no device ptr".into()))?;
        // Keep the storage alive for the process lifetime - the ring's opcode-6 arm dereferences this pointer whenever the wave runs, so it must outlive the enqueue.
        // Leaking a 96-byte device buffer per MoE layer is negligible against the expert weights it.
        std::mem::forget(storage);
        Ok(ptr)
    }

    /// Build the parent `ScytheTaskDescriptor` that enqueues this MoE task onto the `ScytheRing`.
    /// The parent carries `opcode = 6` and `moe_dev_ptr` - the DEVICE address returned by [`Self::upload`].
    pub fn enqueue_via(
        moe_dev_ptr: u64,
        input_ptr: u64,
        output_ptr: u64,
        peer_ptr: u64,
    ) -> ScytheTaskDescriptor {
        assert!(
            moe_dev_ptr != 0,
            "MoETaskDescriptor::enqueue_via requires the DEVICE pointer from upload(); \
             a host address here is the F4 audit bug"
        );
        ScytheTaskDescriptor {
            opcode: 6, // MoE dispatch (WI-Charon-3)
            // m/n/k are unused for opcode 6 (geometry lives in the
            // MoETaskDescriptor); zero them for determinism so device-side dumps read cleanly.
            m: 0,
            n: 0,
            k: 0,
            input_ptr,
            weight_ptr: moe_dev_ptr,
            output_ptr,
            peer_ptr,
            status: 0, // pending
        }
    }

    /// Validate the descriptor's geometry before enqueue.
    /// Pure: returns `Err` on a struct that would cause the device kernel to read out-of-bounds.
    pub fn validate(&self) -> Result<(), String> {
        if self.hidden == 0 || self.inter == 0 || self.num_tokens == 0 {
            return Err(format!(
                "MoETaskDescriptor: non-positive geometry hidden={} inter={} num_tokens={}",
                self.hidden, self.inter, self.num_tokens
            ));
        }
        if self.block_size == 0 {
            return Err("MoETaskDescriptor: block_size must be > 0".into());
        }
        if self.num_experts == 0 {
            return Err("MoETaskDescriptor: num_experts must be > 0".into());
        }
        if self.top_k == 0 || self.top_k > self.num_experts {
            return Err(format!(
                "MoETaskDescriptor: top_k={} out of range [1, num_experts={}]",
                self.top_k, self.num_experts
            ));
        }
        // Weight pointers may legitimately be zero in a host-side test that only validates geometry (no device buffer yet).
        // Only flag the schedule pointers as required - a MoE dispatch with no routing schedule.
        if self.token_ids_ptr == 0 || self.expert_ids_ptr == 0 || self.weights_ptr == 0 {
            return Err(
                "MoETaskDescriptor: token_ids_ptr/expert_ids_ptr/weights_ptr must be non-null"
                    .into(),
            );
        }
        Ok(())
    }
}


// ── KDA / recurrent task descriptor + opcode OP_RECURRENT (M5) ───────────────
// The generic `ScytheTaskDescriptor` (m/n/k + 4 pointers) cannot express the
// qwen35 recurrent step: it is THREE kernels (short-conv causal step, batched
// gated delta rule, gated head norm) over ~14 device buffers. The wave's
// `OP_RECURRENT` arm casts `desc->weight_ptr` to this struct and calls the three
// device-callable functions inline — the same pattern `MoETaskDescriptor` gives
// `OP_MOE`.

/// Device-resident descriptor for one qwen35 KDA/recurrent step.
///
/// The pointers are all DEVICE addresses and the state buffers are written in
/// place, so the descriptor itself is immutable for the life of a graph and can
/// be staged once at capture.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct KdaTaskDescriptor {
    /// Conv input: the fused QKV projection output (`q_buf`).
    pub conv_in: u64,
    /// Short-conv weight.
    pub conv_w: u64,
    /// Short-conv bias (0 = none).
    pub conv_bias: u64,
    /// Conv state ring, updated in place.
    pub conv_state: u64,
    /// Conv output (`conv_out`), [q | k | v] layout.
    pub conv_out: u64,
    /// Per-head `alpha` (decay gate input).
    pub alpha: u64,
    /// Per-head `beta` (write gate input).
    pub beta: u64,
    /// Per-head `dt.bias`.
    pub dt_bias: u64,
    /// Per-head `ssm_a`.
    pub ssm_a: u64,
    /// Per-head `ssm_norm` weight.
    pub ssm_norm: u64,
    /// Per-head gate `z` (the fused attn_gate projection).
    pub gate: u64,
    /// Recurrent state S, updated in place.
    pub state: u64,
    /// Per-head accumulators, [num_v, head_dim].
    pub acc: u64,
    /// Branch output (the block's recurrent output).
    pub branch: u64,
    /// Number of value heads.
    pub num_v: u32,
    /// Number of key heads.
    pub num_k: u32,
    /// Head dim.
    pub head_dim: u32,
    /// Batch (tokens in this step).
    pub batch: u32,
    /// Conv channels (= num_k + num_k + num_v heads times head_dim).
    pub channels: u32,
    /// Conv kernel size.
    pub kernel_size: u32,
    /// RMS eps.
    pub eps: f32,
    /// 1 when a gate `z` is present (the qwen35 path always passes one).
    pub has_z: u32,
}

impl KdaTaskDescriptor {
    /// Upload this descriptor to device-visible memory and return the DEVICE
    /// address the caller passes to the ring as `weight_ptr`. Leaks the staging
    /// buffer on purpose: the wave dereferences it whenever it runs, so it must
    /// outlive the enqueue.
    pub fn upload(&self, device: &crate::RocmDevice) -> crate::Result<u64> {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                std::mem::size_of::<Self>(),
            )
        };
        let storage = device.from_cpu_bytes(
            bytes,
            &grim_tensor::Shape::new(vec![std::mem::size_of::<Self>()]),
            grim_tensor::dtype::DType {
                arith: grim_tensor::ArithType::U8,
                storage: grim_tensor::dtype::Storage::Native,
            },
        )?;
        let ptr = storage
            .as_any()
            .downcast_ref::<crate::RocmStorage>()
            .and_then(|rs| rs.device_ptr_u64())
            .ok_or_else(|| crate::Error::Backend("KDA upload: no device ptr".into()))?;
        std::mem::forget(storage);
        Ok(ptr)
    }
}
