//! MoE routing and Charon kernel dispatchers for `RocmDevice`.

//! Charon fused MoE dispatch and mega-dispatch launchers.

use std::ffi::c_void;

use grim_tensor::error::{Error, Result};

use crate::device::roc_device::{ RocmDevice };
#[cfg(feature = "training")]
use crate::device::roc_device::CharonBackwardResult;
use crate::memory::storage::RocmStorage;
use crate::{
    HipDim3, arg, check_hip, hipFreeAsync, hipMemsetAsync, upload_device_buffer,
};

impl RocmDevice {
    /// `tests/golden_charon_moe_gpu.rs` ($\le 10^{-3}$ max-abs-diff).
    /// Caller wiring lives in `grim_nn::moe::MoeFfn::forward_rocm` (gated on the `rocm-mem` feature), reached when the activation is.
    pub fn launch_charon_fused_dispatch(
        &self,
        activations: &RocmStorage,
        expert_gate_w_ptr: u64,
        expert_up_w_ptr: u64,
        expert_down_w_ptr: u64,
        assignment: &crate::kernels::charon::RoutingAssignment,
        out_storage: &RocmStorage,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
    ) -> Result<*mut c_void> {
        // P1-3: raw HIP ops below bind to the calling thread's current
        // device — pin to the owning ordinal (see matmul_op fix, 2026-08-23e).
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let a_ptr = activations.device_ptr.ok_or_else(|| {
            Error::Backend("charon_fused_dispatch: activations has no device ptr".into())
        })?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("charon_fused_dispatch: out has no device ptr".into()))?;

        // Validate shapes + null pointers before any HIP dereference (G-A2
        // host-logic path, unit-tested in kernels::charon::tests).
        crate::kernels::charon::validate_launch_inputs(
            a_ptr as *mut c_void,
            expert_gate_w_ptr as *mut c_void,
            expert_up_w_ptr as *mut c_void,
            expert_down_w_ptr as *mut c_void,
            out_ptr as *mut c_void,
            assignment,
            hidden,
            inter,
        )?;

        // Zero the output buffer before launch: the kernel accumulates per- expert contributions via `atomicAdd`, so any stale bytes in the output storage would be added into the result.
        // This mirrors the `BackendDevice::zeros` path (hipMemset, roc_device.rs:1363).
        check_hip("charon hipMemset(output, 0)", unsafe {
            hipMemsetAsync(
                out_ptr as *mut c_void,
                0,
                out_storage.bytes(),
                self.active_stream(),
            )
        })?;

        // Plan the launch (wave-aligned block, grid over pairs). Pass the
        // device's real wavefront size (W32 on gfx1036, W64 on CDNA).
        let wave = self.wavefront_size() as u32;
        let mut tuner_guard = self.autotuner.lock().ok();
        let plan = crate::kernels::charon::plan_fused_dispatch_with_autotuner(
            assignment,
            wave,
            tuner_guard.as_deref_mut(),
            &self.gpu_target,
            hidden,
            inter,
        );

        if plan.grid_x == 0 {
            // No pairs → nothing to launch; return the active stream.
            return Ok(self.active_stream());
        }
        let grid_dim = HipDim3::new(plan.grid_x, 1, 1);
        let block_dim = HipDim3::new(plan.block_x, 1, 1);

        // Upload the routing arrays (tokens, experts, weights) to the device.
        // These are freed after the launch synchronizes (mirroring the embedding path's transient-buffer discipline).
        let mut tok_ptr = upload_device_buffer(self.ordinal, &assignment.tokens)?;
        let mut exp_ptr = upload_device_buffer(self.ordinal, &assignment.experts)?;
        let mut w_ptr = upload_device_buffer(self.ordinal, &assignment.weights)?;

        let mut a = a_ptr as *mut c_void;
        let mut gw = expert_gate_w_ptr as *mut c_void;
        let mut uw = expert_up_w_ptr as *mut c_void;
        let mut dw = expert_down_w_ptr as *mut c_void;
        let mut optr = out_ptr as *mut c_void;
        let mut hidden_i = hidden as i32;
        let mut inter_i = inter as i32;
        let mut num_pairs_i = assignment.num_pairs() as i32;
        let mut rsf = routed_scaling_factor;

        let stream = self.launch_compute_kernel(
            "grim_moe_fused_dispatch",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut a),
                arg(&mut gw),
                arg(&mut uw),
                arg(&mut dw),
                arg(&mut tok_ptr),
                arg(&mut exp_ptr),
                arg(&mut w_ptr),
                arg(&mut optr),
                arg(&mut hidden_i),
                arg(&mut inter_i),
                arg(&mut num_pairs_i),
                arg(&mut rsf),
            ],
        )?;

        // Free the transient routing buffers after the kernel completes.
        // SPEED-ROC-2: stream-ordered free of the transient routing buffers.
        // The old per-launch hipStreamSynchronize blocked the host once per MoE
        // layer per step and serialized the next layer's enqueue; launch errors
        // are surfaced by launch_compute_kernel's own return code.
        if self.active_capture_stream().is_none() {
            unsafe {
                let free_stream = self.active_stream();
                let _ = hipFreeAsync(tok_ptr, free_stream);
                let _ = hipFreeAsync(exp_ptr, free_stream);
                let _ = hipFreeAsync(w_ptr, free_stream);
            }
        }
        Ok(stream)
    }

    /// Launches the MoE fused comm-compute mega-kernel (UniEP persistent SM model).
    pub fn launch_moe_mega_dispatch(
        &self,
        activations: &RocmStorage,
        expert_gate_w_ptr: u64,
        expert_up_w_ptr: u64,
        expert_down_w_ptr: u64,
        destination_slots: &[u32],
        global_offsets: &[u32],
        expert_counts: &[u32],
        router_tokens: &[u32],
        router_experts: &[u32],
        router_weights: &[f32],
        scoreboard_arrivals: &[u32],
        scoreboard_ready: &[u32],
        out_storage: &RocmStorage,
        config: &crate::kernels::moe_mega_kernel::MoeMegaLaunchConfig,
    ) -> Result<*mut c_void> {
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let a_ptr = activations.device_ptr.ok_or_else(|| {
            Error::Backend("moe_mega_dispatch: activations has no device ptr".into())
        })?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("moe_mega_dispatch: out has no device ptr".into()))?;

        crate::kernels::moe_mega_kernel::validate_mega_kernel_inputs(
            a_ptr as *mut c_void,
            expert_gate_w_ptr as *mut c_void,
            expert_up_w_ptr as *mut c_void,
            expert_down_w_ptr as *mut c_void,
            out_ptr as *mut c_void,
            config,
        )?;

        check_hip("moe_mega hipMemset(output, 0)", unsafe {
            hipMemsetAsync(
                out_ptr as *mut c_void,
                0,
                out_storage.bytes(),
                self.active_stream(),
            )
        })?;

        let mut dest_slots_ptr = upload_device_buffer(self.ordinal, destination_slots)?;
        let mut offsets_ptr = upload_device_buffer(self.ordinal, global_offsets)?;
        let mut counts_ptr = upload_device_buffer(self.ordinal, expert_counts)?;
        let mut tokens_ptr = upload_device_buffer(self.ordinal, router_tokens)?;
        let mut experts_ptr = upload_device_buffer(self.ordinal, router_experts)?;
        let mut weights_ptr = upload_device_buffer(self.ordinal, router_weights)?;
        let mut arrivals_ptr = upload_device_buffer(self.ordinal, scoreboard_arrivals)?;
        let mut ready_ptr = upload_device_buffer(self.ordinal, scoreboard_ready)?;
        let mut cursor_ptr = upload_device_buffer(self.ordinal, &[0u32])?;

        let packed_elem_count = config.total_routed_instances * config.hidden;
        let zero_act = vec![0.0f32; packed_elem_count.max(1)];
        let zero_out = vec![0.0f32; packed_elem_count.max(1)];
        let mut packed_act_ptr = upload_device_buffer(self.ordinal, &zero_act)?;
        let mut packed_out_ptr = upload_device_buffer(self.ordinal, &zero_out)?;

        let grid_dim = HipDim3::new(config.num_sm_blocks as u32, 1, 1);
        let block_dim = HipDim3::new(config.block_threads as u32, 1, 1);

        let mut a = a_ptr as *mut c_void;
        let mut gw = expert_gate_w_ptr as *mut c_void;
        let mut uw = expert_up_w_ptr as *mut c_void;
        let mut dw = expert_down_w_ptr as *mut c_void;
        let mut optr = out_ptr as *mut c_void;
        let mut batch_i = config.batch as i32;
        let mut hidden_i = config.hidden as i32;
        let mut inter_i = config.inter as i32;
        let mut num_exp_i = config.num_experts as i32;
        let mut top_k_i = config.top_k as i32;
        let mut total_routed_i = config.total_routed_instances as i32;
        let mut tile_size_i = config.tile_size as i32;
        let mut num_tiles_i = config.num_tiles as i32;
        let mut n_comm_i = config.n_comm_tasks as i32;
        let mut n_comp_i = config.n_comp_tasks as i32;
        let mut n_relay_i = config.n_relay_tasks as i32;
        let mut rsf = config.routed_scaling_factor;

        let stream = self.launch_compute_kernel(
            "grim_moe_mega_kernel",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut a),
                arg(&mut gw),
                arg(&mut uw),
                arg(&mut dw),
                arg(&mut dest_slots_ptr),
                arg(&mut offsets_ptr),
                arg(&mut counts_ptr),
                arg(&mut tokens_ptr),
                arg(&mut experts_ptr),
                arg(&mut weights_ptr),
                arg(&mut arrivals_ptr),
                arg(&mut ready_ptr),
                arg(&mut cursor_ptr),
                arg(&mut packed_act_ptr),
                arg(&mut packed_out_ptr),
                arg(&mut optr),
                arg(&mut batch_i),
                arg(&mut hidden_i),
                arg(&mut inter_i),
                arg(&mut num_exp_i),
                arg(&mut top_k_i),
                arg(&mut total_routed_i),
                arg(&mut tile_size_i),
                arg(&mut num_tiles_i),
                arg(&mut n_comm_i),
                arg(&mut n_comp_i),
                arg(&mut n_relay_i),
                arg(&mut rsf),
            ],
        )?;

        if self.active_capture_stream().is_none() {
            unsafe {
                let free_stream = self.active_stream();
                let _ = hipFreeAsync(dest_slots_ptr, free_stream);
                let _ = hipFreeAsync(offsets_ptr, free_stream);
                let _ = hipFreeAsync(counts_ptr, free_stream);
                let _ = hipFreeAsync(tokens_ptr, free_stream);
                let _ = hipFreeAsync(experts_ptr, free_stream);
                let _ = hipFreeAsync(weights_ptr, free_stream);
                let _ = hipFreeAsync(arrivals_ptr, free_stream);
                let _ = hipFreeAsync(ready_ptr, free_stream);
                let _ = hipFreeAsync(cursor_ptr, free_stream);
                let _ = hipFreeAsync(packed_act_ptr, free_stream);
                let _ = hipFreeAsync(packed_out_ptr, free_stream);
            }
        }
        Ok(stream)
    }
}
