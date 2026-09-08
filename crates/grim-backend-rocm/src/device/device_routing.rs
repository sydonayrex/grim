//! MoE routing and Charon kernel dispatchers for `RocmDevice`.

use std::ffi::c_void;

use grim_tensor::backend::{BackendStorage, ComputeHandle};
use grim_tensor::dtype::{ArithType, DType, Storage as DTypeStorage};
use grim_tensor::error::{Error, Result};
use grim_tensor::{CoreTensorOps, MemoryOps, Shape};

use crate::device::roc_device::{CharonBackwardResult, RocmDevice};
use crate::memory::storage::RocmStorage;
use crate::{
    HipDim3, RocmHandle, arg, as_rocm, check_hip, dev_ptr, dtype_f32, hipFreeAsync,
    hipMemsetAsync, upload_device_buffer,
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
            hipMemsetAsync(out_ptr as *mut c_void, 0, out_storage.bytes(), self.active_stream())
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

    /// Fused MoE dispatch helper: allocates output buffer, uploads flat expert weights,
    /// and launches `grim_moe_fused_dispatch`.
    pub fn moe_fused_dispatch(
        &self,
        activations: &RocmStorage,
        gate_flat: &[f32],
        up_flat: &[f32],
        down_flat: &[f32],
        assignment: &crate::kernels::charon::RoutingAssignment,
        out_shape: &Shape,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
    ) -> Result<(RocmStorage, RocmHandle)> {
        let gate_buf = self.from_cpu(gate_flat, &Shape::new(vec![gate_flat.len()]), DType::F32)?;
        let up_buf = self.from_cpu(up_flat, &Shape::new(vec![up_flat.len()]), DType::F32)?;
        let down_buf = self.from_cpu(down_flat, &Shape::new(vec![down_flat.len()]), DType::F32)?;

        self.moe_fused_dispatch_resident(
            activations,
            &*gate_buf,
            &*up_buf,
            &*down_buf,
            assignment,
            out_shape,
            hidden,
            inter,
            routed_scaling_factor,
        )
    }

    /// Fused MoE dispatch against weight buffers that are already resident on the device.
    /// Unlike [`Self::moe_fused_dispatch`], no host `&[f32]` weight arrays are uploaded per call - callers keep the.
    pub fn moe_fused_dispatch_resident(
        &self,
        activations: &RocmStorage,
        gate_buf: &dyn BackendStorage,
        up_buf: &dyn BackendStorage,
        down_buf: &dyn BackendStorage,
        assignment: &crate::kernels::charon::RoutingAssignment,
        out_shape: &Shape,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
    ) -> Result<(RocmStorage, RocmHandle)> {
        let gate_r = gate_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("gate_buf downcast failed".into()))?;
        let up_r = up_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("up_buf downcast failed".into()))?;
        let down_r = down_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("down_buf downcast failed".into()))?;

        let out_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;

        let stream = self.launch_charon_fused_dispatch(
            activations,
            gate_r.device_ptr_checked()?,
            up_r.device_ptr_checked()?,
            down_r.device_ptr_checked()?,
            assignment,
            &out_storage,
            hidden,
            inter,
            routed_scaling_factor,
        )?;
        Ok((out_storage, RocmHandle::new(Some(stream))))
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
            hipMemsetAsync(out_ptr as *mut c_void, 0, out_storage.bytes(), self.active_stream())
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

    /// Fused grouped MoE dispatch for CompressedTensors W8A8 INT8.
    pub fn moe_fused_grouped_dispatch_w8a8_int8(
        &self,
        activations: &RocmStorage,
        gate_buf: &dyn BackendStorage,
        up_buf: &dyn BackendStorage,
        down_buf: &dyn BackendStorage,
        a_scale_buf: &dyn BackendStorage,
        sorted: &crate::kernels::charon::SortedRouting,
        out_shape: &Shape,
        hidden: usize,
        inter: usize,
        num_experts: usize,
        routed_scaling_factor: f32,
    ) -> Result<(RocmStorage, RocmHandle)> {
        let gate_r = gate_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("gate_buf downcast failed".into()))?;
        let up_r = up_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("up_buf downcast failed".into()))?;
        let down_r = down_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("down_buf downcast failed".into()))?;
        let ascale_r = a_scale_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("a_scale_buf downcast failed".into()))?;

        let out_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let stream = self.launch_charon_grouped_dispatch_w8a8_int8(
            activations,
            gate_r.device_ptr_checked()?,
            up_r.device_ptr_checked()?,
            down_r.device_ptr_checked()?,
            ascale_r.device_ptr_checked()?,
            sorted,
            &out_storage,
            hidden,
            inter,
            num_experts,
            routed_scaling_factor,
        )?;
        Ok((out_storage, RocmHandle::new(Some(stream))))
    }

    /// Fused grouped MoE dispatch for CompressedTensors W8A8 FP8.
    pub fn moe_fused_grouped_dispatch_w8a8_fp8(
        &self,
        activations: &RocmStorage,
        gate_buf: &dyn BackendStorage,
        up_buf: &dyn BackendStorage,
        down_buf: &dyn BackendStorage,
        a_scale_buf: &dyn BackendStorage,
        sorted: &crate::kernels::charon::SortedRouting,
        out_shape: &Shape,
        hidden: usize,
        inter: usize,
        num_experts: usize,
        routed_scaling_factor: f32,
    ) -> Result<(RocmStorage, RocmHandle)> {
        let gate_r = gate_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("gate_buf downcast failed".into()))?;
        let up_r = up_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("up_buf downcast failed".into()))?;
        let down_r = down_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("down_buf downcast failed".into()))?;
        let ascale_r = a_scale_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("a_scale_buf downcast failed".into()))?;

        let out_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let stream = self.launch_charon_grouped_dispatch_w8a8_fp8(
            activations,
            gate_r.device_ptr_checked()?,
            up_r.device_ptr_checked()?,
            down_r.device_ptr_checked()?,
            ascale_r.device_ptr_checked()?,
            sorted,
            &out_storage,
            hidden,
            inter,
            num_experts,
            routed_scaling_factor,
        )?;
        Ok((out_storage, RocmHandle::new(Some(stream))))
    }

    /// Fused grouped MoE dispatch for AWQ.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_fused_grouped_dispatch_awq(
        &self,
        activations: &RocmStorage,
        gate_buf: &dyn BackendStorage,
        up_buf: &dyn BackendStorage,
        down_buf: &dyn BackendStorage,
        a_scale_buf: &dyn BackendStorage,
        sorted: &crate::kernels::charon::SortedRouting,
        out_shape: &Shape,
        hidden: usize,
        inter: usize,
        num_experts: usize,
        bits: u8,
        group_size: usize,
        gate_qw_off: i64,
        gate_qz_off: i64,
        gate_sc_off: i64,
        gate_stride: u64,
        down_qw_off: i64,
        down_qz_off: i64,
        down_sc_off: i64,
        down_stride: u64,
        routed_scaling_factor: f32,
    ) -> Result<(RocmStorage, RocmHandle)> {
        let gate_r = gate_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("gate_buf downcast failed".into()))?;
        let up_r = up_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("up_buf downcast failed".into()))?;
        let down_r = down_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("down_buf downcast failed".into()))?;
        let ascale_r = a_scale_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("a_scale_buf downcast failed".into()))?;

        let out_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let stream = self.launch_charon_grouped_dispatch_awq(
            activations,
            gate_r.device_ptr_checked()?,
            up_r.device_ptr_checked()?,
            down_r.device_ptr_checked()?,
            ascale_r.device_ptr_checked()?,
            sorted,
            &out_storage,
            hidden,
            inter,
            num_experts,
            bits,
            group_size,
            gate_qw_off,
            gate_qz_off,
            gate_sc_off,
            gate_stride,
            down_qw_off,
            down_qz_off,
            down_sc_off,
            down_stride,
            routed_scaling_factor,
        )?;
        Ok((out_storage, RocmHandle::new(Some(stream))))
    }

    /// Device launcher for the #1 token-sorted (grouped) fused MoE dispatch.
    /// Mirrors `launch_charon_fused_dispatch` but feeds the sorted routing layout (`SortedRouting`) produced by `moe_align_block_size` and calls `grim_moe_fused_grouped`.
    #[allow(dead_code)]
    pub(crate) fn launch_charon_grouped_dispatch(
        &self,
        activations: &RocmStorage,
        expert_gate_w_ptr: u64,
        expert_up_w_ptr: u64,
        expert_down_w_ptr: u64,
        sorted: &crate::kernels::charon::SortedRouting,
        out_storage: &RocmStorage,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
        num_experts: usize,
    ) -> Result<*mut c_void> {
        self.launch_charon_grouped_dispatch_entry(
            activations,
            expert_gate_w_ptr,
            expert_up_w_ptr,
            expert_down_w_ptr,
            sorted,
            out_storage,
            hidden,
            inter,
            routed_scaling_factor,
            num_experts,
            "grim_moe_fused_grouped",
        )
    }

    /// WI-F3 - grouped dispatch against a caller-selected kernel entry, so `CharonSelector` variants can route to the WMMA grouped kernel (`grim_moe_fused_grouped_wmma`) or the scalar grouped kernel via `kernels::charon::grouped_dispatch_entry`.
    /// Same host/sort contract.
    pub(crate) fn launch_charon_grouped_dispatch_entry(
        &self,
        activations: &RocmStorage,
        expert_gate_w_ptr: u64,
        expert_up_w_ptr: u64,
        expert_down_w_ptr: u64,
        sorted: &crate::kernels::charon::SortedRouting,
        out_storage: &RocmStorage,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
        num_experts: usize,
        entry: &str,
    ) -> Result<*mut c_void> {
        // P1-3: raw HIP ops below bind to the calling thread's current
        // device — pin to the owning ordinal (see matmul_op fix, 2026-08-23e).
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let a_ptr = activations.device_ptr.ok_or_else(|| {
            Error::Backend("charon_grouped_dispatch: activations has no device ptr".into())
        })?;
        let out_ptr = out_storage.device_ptr.ok_or_else(|| {
            Error::Backend("charon_grouped_dispatch: out has no device ptr".into())
        })?;

        crate::kernels::charon::validate_grouped_inputs(
            a_ptr as *mut c_void,
            expert_gate_w_ptr as *mut c_void,
            expert_up_w_ptr as *mut c_void,
            expert_down_w_ptr as *mut c_void,
            out_ptr as *mut c_void,
            sorted,
            hidden,
            inter,
            num_experts,
        )?;

        // Output is accumulated via atomicAdd in-kernel; zero first.
        check_hip("charon_grouped hipMemset(output, 0)", unsafe {
            hipMemsetAsync(out_ptr as *mut c_void, 0, out_storage.bytes(), self.active_stream())
        })?;

        let wave = self.wavefront_size() as u32;
        let plan = crate::kernels::charon::plan_grouped_dispatch(sorted, wave);
        if plan.grid_x == 0 {
            return Ok(self.active_stream());
        }
        let grid_dim = HipDim3::new(plan.grid_x, 1, 1);
        let block_dim = HipDim3::new(plan.block_x, 1, 1);

        let mut tok_ptr = upload_device_buffer(self.ordinal, &sorted.sorted_token_ids)?;
        let mut exp_ptr = upload_device_buffer(self.ordinal, &sorted.sorted_expert_ids)?;
        let mut w_ptr = upload_device_buffer(self.ordinal, &sorted.sorted_weights)?;

        let mut a = a_ptr as *mut c_void;
        let mut gw = expert_gate_w_ptr as *mut c_void;
        let mut uw = expert_up_w_ptr as *mut c_void;
        let mut dw = expert_down_w_ptr as *mut c_void;
        let mut optr = out_ptr as *mut c_void;
        let mut hidden_i = hidden as i32;
        let mut inter_i = inter as i32;
        let mut num_tokens_i = sorted.num_tokens_post_padded as i32;
        let mut block_size_i = sorted.block_size as i32;
        let mut rsf = routed_scaling_factor;

        let stream = self.launch_compute_kernel(
            entry,
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
                arg(&mut num_tokens_i),
                arg(&mut block_size_i),
                arg(&mut rsf),
            ],
        )?;

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

    /// Device launcher for the #2 FP8 W8A8 token-sorted grouped dispatch.
    /// Mirrors `launch_charon_grouped_dispatch` (same sorted layout + grid/block) but weights are FP8 E4M3 bytes with per-block-16.
    #[allow(dead_code)]
    pub(crate) fn launch_charon_grouped_dispatch_fp8(
        &self,
        activations: &RocmStorage,
        expert_gate_w_fp8_ptr: u64,
        expert_up_w_fp8_ptr: u64,
        expert_down_w_fp8_ptr: u64,
        expert_gate_scale_ptr: u64,
        expert_up_scale_ptr: u64,
        expert_down_scale_ptr: u64,
        a_scale_ptr: u64,
        sorted: &crate::kernels::charon::SortedRouting,
        out_storage: &RocmStorage,
        hidden: usize,
        inter: usize,
        num_experts: usize,
        routed_scaling_factor: f32,
    ) -> Result<*mut c_void> {
        // P1-3: raw HIP ops below bind to the calling thread's current
        // device — pin to the owning ordinal (see matmul_op fix, 2026-08-23e).
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let a_ptr = activations.device_ptr.ok_or_else(|| {
            Error::Backend("charon_grouped_fp8: activations has no device ptr".into())
        })?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("charon_grouped_fp8: out has no device ptr".into()))?;

        crate::kernels::charon::validate_grouped_inputs(
            a_ptr as *mut c_void,
            expert_gate_w_fp8_ptr as *mut c_void,
            expert_up_w_fp8_ptr as *mut c_void,
            expert_down_w_fp8_ptr as *mut c_void,
            out_ptr as *mut c_void,
            sorted,
            hidden,
            inter,
            num_experts,
        )?;

        check_hip("charon_grouped_fp8 hipMemset(output, 0)", unsafe {
            hipMemsetAsync(out_ptr as *mut c_void, 0, out_storage.bytes(), self.active_stream())
        })?;

        let wave = self.wavefront_size() as u32;
        let plan = crate::kernels::charon::plan_grouped_dispatch(sorted, wave);
        if plan.grid_x == 0 {
            return Ok(self.active_stream());
        }
        let grid_dim = HipDim3::new(plan.grid_x, 1, 1);
        let block_dim = HipDim3::new(plan.block_x, 1, 1);

        let mut tok_ptr = upload_device_buffer(self.ordinal, &sorted.sorted_token_ids)?;
        let mut exp_ptr = upload_device_buffer(self.ordinal, &sorted.sorted_expert_ids)?;
        let mut w_ptr = upload_device_buffer(self.ordinal, &sorted.sorted_weights)?;

        let mut a = a_ptr as *mut c_void;
        let mut gw = expert_gate_w_fp8_ptr as *mut c_void;
        let mut uw = expert_up_w_fp8_ptr as *mut c_void;
        let mut dw = expert_down_w_fp8_ptr as *mut c_void;
        let mut gs = expert_gate_scale_ptr as *mut c_void;
        let mut us = expert_up_scale_ptr as *mut c_void;
        let mut ds = expert_down_scale_ptr as *mut c_void;
        let mut ascale = a_scale_ptr as *mut c_void;
        let mut optr = out_ptr as *mut c_void;
        let mut hidden_i = hidden as i32;
        let mut inter_i = inter as i32;
        let mut num_tokens_i = sorted.num_tokens_post_padded as i32;
        let mut block_size_i = sorted.block_size as i32;
        let mut rsf = routed_scaling_factor;

        let stream = self.launch_compute_kernel(
            "grim_moe_fused_grouped_fp8",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut a),
                arg(&mut gw),
                arg(&mut uw),
                arg(&mut dw),
                arg(&mut gs),
                arg(&mut us),
                arg(&mut ds),
                arg(&mut ascale),
                arg(&mut tok_ptr),
                arg(&mut exp_ptr),
                arg(&mut w_ptr),
                arg(&mut optr),
                arg(&mut hidden_i),
                arg(&mut inter_i),
                arg(&mut num_tokens_i),
                arg(&mut block_size_i),
                arg(&mut rsf),
            ],
        )?;

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

    pub(crate) fn launch_charon_grouped_dispatch_mxfp4(
        &self,
        activations: &RocmStorage,
        egate_w_ptr: u64,
        eup_w_ptr: u64,
        edown_w_ptr: u64,
        egate_e_ptr: u64,
        eup_e_ptr: u64,
        edown_e_ptr: u64,
        a_scale_ptr: u64,
        sorted: &crate::kernels::charon::SortedRouting,
        out_storage: &RocmStorage,
        hidden: usize,
        inter: usize,
        num_experts: usize,
        routed_scaling_factor: f32,
    ) -> Result<*mut c_void> {
        // P1-3: raw HIP ops below bind to the calling thread's current
        // device — pin to the owning ordinal (see matmul_op fix, 2026-08-23e).
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let a_ptr = activations.device_ptr.ok_or_else(|| {
            Error::Backend("charon_grouped_mxfp4: activations has no device ptr".into())
        })?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("charon_grouped_mxfp4: out has no device ptr".into()))?;

        crate::kernels::charon::validate_grouped_inputs(
            a_ptr as *mut c_void,
            egate_w_ptr as *mut c_void,
            eup_w_ptr as *mut c_void,
            edown_w_ptr as *mut c_void,
            out_ptr as *mut c_void,
            sorted,
            hidden,
            inter,
            num_experts,
        )?;

        check_hip("charon_grouped_mxfp4 hipMemset(output, 0)", unsafe {
            hipMemsetAsync(out_ptr as *mut c_void, 0, out_storage.bytes(), self.active_stream())
        })?;

        let wave = self.wavefront_size() as u32;
        let plan = crate::kernels::charon::plan_grouped_dispatch(sorted, wave);
        if plan.grid_x == 0 {
            return Ok(self.active_stream());
        }
        let grid_dim = HipDim3::new(plan.grid_x, 1, 1);
        let block_dim = HipDim3::new(plan.block_x, 1, 1);

        let mut tok_ptr = upload_device_buffer(self.ordinal, &sorted.sorted_token_ids)?;
        let mut exp_ptr = upload_device_buffer(self.ordinal, &sorted.sorted_expert_ids)?;
        let mut w_ptr = upload_device_buffer(self.ordinal, &sorted.sorted_weights)?;

        let mut a = a_ptr as *mut c_void;
        let mut gw = egate_w_ptr as *mut c_void;
        let mut uw = eup_w_ptr as *mut c_void;
        let mut dw = edown_w_ptr as *mut c_void;
        let mut ge = egate_e_ptr as *mut c_void;
        let mut ue = eup_e_ptr as *mut c_void;
        let mut de = edown_e_ptr as *mut c_void;
        let mut ascale = a_scale_ptr as *mut c_void;
        let mut optr = out_ptr as *mut c_void;
        let mut hidden_i = hidden as i32;
        let mut inter_i = inter as i32;
        let mut num_tokens_i = sorted.num_tokens_post_padded as i32;
        let mut block_size_i = sorted.block_size as i32;
        let mut rsf = routed_scaling_factor;

        let stream = self.launch_compute_kernel(
            "grim_moe_fused_grouped_mxfp4",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut a),
                arg(&mut gw),
                arg(&mut uw),
                arg(&mut dw),
                arg(&mut ge),
                arg(&mut ue),
                arg(&mut de),
                arg(&mut ascale),
                arg(&mut tok_ptr),
                arg(&mut exp_ptr),
                arg(&mut w_ptr),
                arg(&mut optr),
                arg(&mut hidden_i),
                arg(&mut inter_i),
                arg(&mut num_tokens_i),
                arg(&mut block_size_i),
                arg(&mut rsf),
            ],
        )?;

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

    pub(crate) fn launch_charon_grouped_dispatch_mxfp8(
        &self,
        activations: &RocmStorage,
        egate_w_ptr: u64,
        eup_w_ptr: u64,
        edown_w_ptr: u64,
        egate_e_ptr: u64,
        eup_e_ptr: u64,
        edown_e_ptr: u64,
        a_scale_ptr: u64,
        sorted: &crate::kernels::charon::SortedRouting,
        out_storage: &RocmStorage,
        hidden: usize,
        inter: usize,
        num_experts: usize,
        routed_scaling_factor: f32,
    ) -> Result<*mut c_void> {
        // P1-3: raw HIP ops below bind to the calling thread's current
        // device — pin to the owning ordinal (see matmul_op fix, 2026-08-23e).
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let a_ptr = activations.device_ptr.ok_or_else(|| {
            Error::Backend("charon_grouped_mxfp8: activations has no device ptr".into())
        })?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("charon_grouped_mxfp8: out has no device ptr".into()))?;

        crate::kernels::charon::validate_grouped_inputs(
            a_ptr as *mut c_void,
            egate_w_ptr as *mut c_void,
            eup_w_ptr as *mut c_void,
            edown_w_ptr as *mut c_void,
            out_ptr as *mut c_void,
            sorted,
            hidden,
            inter,
            num_experts,
        )?;

        check_hip("charon_grouped_mxfp8 hipMemset(output, 0)", unsafe {
            hipMemsetAsync(out_ptr as *mut c_void, 0, out_storage.bytes(), self.active_stream())
        })?;

        let wave = self.wavefront_size() as u32;
        let plan = crate::kernels::charon::plan_grouped_dispatch(sorted, wave);
        if plan.grid_x == 0 {
            return Ok(self.active_stream());
        }
        let grid_dim = HipDim3::new(plan.grid_x, 1, 1);
        let block_dim = HipDim3::new(plan.block_x, 1, 1);

        let mut tok_ptr = upload_device_buffer(self.ordinal, &sorted.sorted_token_ids)?;
        let mut exp_ptr = upload_device_buffer(self.ordinal, &sorted.sorted_expert_ids)?;
        let mut w_ptr = upload_device_buffer(self.ordinal, &sorted.sorted_weights)?;

        let mut a = a_ptr as *mut c_void;
        let mut gw = egate_w_ptr as *mut c_void;
        let mut uw = eup_w_ptr as *mut c_void;
        let mut dw = edown_w_ptr as *mut c_void;
        let mut ge = egate_e_ptr as *mut c_void;
        let mut ue = eup_e_ptr as *mut c_void;
        let mut de = edown_e_ptr as *mut c_void;
        let mut ascale = a_scale_ptr as *mut c_void;
        let mut optr = out_ptr as *mut c_void;
        let mut hidden_i = hidden as i32;
        let mut inter_i = inter as i32;
        let mut num_tokens_i = sorted.num_tokens_post_padded as i32;
        let mut block_size_i = sorted.block_size as i32;
        let mut rsf = routed_scaling_factor;

        let stream = self.launch_compute_kernel(
            "grim_moe_fused_grouped_mxfp8",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut a),
                arg(&mut gw),
                arg(&mut uw),
                arg(&mut dw),
                arg(&mut ge),
                arg(&mut ue),
                arg(&mut de),
                arg(&mut ascale),
                arg(&mut tok_ptr),
                arg(&mut exp_ptr),
                arg(&mut w_ptr),
                arg(&mut optr),
                arg(&mut hidden_i),
                arg(&mut inter_i),
                arg(&mut num_tokens_i),
                arg(&mut block_size_i),
                arg(&mut rsf),
            ],
        )?;

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

    pub(crate) fn launch_charon_grouped_dispatch_q80(
        &self,
        activations: &RocmStorage,
        egate_w_ptr: u64,
        eup_w_ptr: u64,
        edown_w_ptr: u64,
        a_scale_ptr: u64,
        sorted: &crate::kernels::charon::SortedRouting,
        out_storage: &RocmStorage,
        hidden: usize,
        inter: usize,
        num_experts: usize,
        routed_scaling_factor: f32,
    ) -> Result<*mut c_void> {
        // P1-3: raw HIP ops below bind to the calling thread's current
        // device — pin to the owning ordinal (see matmul_op fix, 2026-08-23e).
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let a_ptr = activations.device_ptr.ok_or_else(|| {
            Error::Backend("charon_grouped_q80: activations has no device ptr".into())
        })?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("charon_grouped_q80: out has no device ptr".into()))?;

        crate::kernels::charon::validate_grouped_inputs(
            a_ptr as *mut c_void,
            egate_w_ptr as *mut c_void,
            eup_w_ptr as *mut c_void,
            edown_w_ptr as *mut c_void,
            out_ptr as *mut c_void,
            sorted,
            hidden,
            inter,
            num_experts,
        )?;

        check_hip("charon_grouped_q80 hipMemset(output, 0)", unsafe {
            hipMemsetAsync(out_ptr as *mut c_void, 0, out_storage.bytes(), self.active_stream())
        })?;

        let wave = self.wavefront_size() as u32;
        let plan = crate::kernels::charon::plan_grouped_dispatch(sorted, wave);
        if plan.grid_x == 0 {
            return Ok(self.active_stream());
        }
        let grid_dim = HipDim3::new(plan.grid_x, 1, 1);
        let block_dim = HipDim3::new(plan.block_x, 1, 1);

        let mut tok_ptr = upload_device_buffer(self.ordinal, &sorted.sorted_token_ids)?;
        let mut exp_ptr = upload_device_buffer(self.ordinal, &sorted.sorted_expert_ids)?;
        let mut w_ptr = upload_device_buffer(self.ordinal, &sorted.sorted_weights)?;

        let mut a = a_ptr as *mut c_void;
        let mut gw = egate_w_ptr as *mut c_void;
        let mut uw = eup_w_ptr as *mut c_void;
        let mut dw = edown_w_ptr as *mut c_void;
        let mut ascale = a_scale_ptr as *mut c_void;
        let mut optr = out_ptr as *mut c_void;
        let mut hidden_i = hidden as i32;
        let mut inter_i = inter as i32;
        let mut num_tokens_i = sorted.num_tokens_post_padded as i32;
        let mut block_size_i = sorted.block_size as i32;
        let mut rsf = routed_scaling_factor;

        let stream = self.launch_compute_kernel(
            "grim_moe_fused_grouped_q80",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut a),
                arg(&mut gw),
                arg(&mut uw),
                arg(&mut dw),
                arg(&mut ascale),
                arg(&mut tok_ptr),
                arg(&mut exp_ptr),
                arg(&mut w_ptr),
                arg(&mut optr),
                arg(&mut hidden_i),
                arg(&mut inter_i),
                arg(&mut num_tokens_i),
                arg(&mut block_size_i),
                arg(&mut rsf),
            ],
        )?;

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

    /// Launcher for the unified IQ/K-quant grouped MoE kernel (`grim_moe_fused_grouped_iqk`).
    /// `format_id` selects the super-block decode (0 iq4nl ..
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_charon_grouped_dispatch_iqk(
        &self,
        act_storage: &RocmStorage,
        egate_w_ptr: u64,
        eup_w_ptr: u64,
        edown_w_ptr: u64,
        a_scale_ptr: u64,
        sorted: &crate::kernels::charon::SortedRouting,
        out_storage: &RocmStorage,
        hidden: usize,
        inter: usize,
        num_experts: usize,
        format_id: usize,
        routed_scaling_factor: f32,
    ) -> Result<*mut c_void> {
        // P1-3: raw HIP ops below bind to the calling thread's current
        // device — pin to the owning ordinal (see matmul_op fix, 2026-08-23e).
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        use crate::kernels::charon::plan_grouped_dispatch;
        let _ = num_experts; // validated by caller; kernel reads per-expert super-blocks

        check_hip("charon_grouped_iqk hipMemset(output, 0)", unsafe {
            hipMemsetAsync(
                out_storage.device_ptr.ok_or_else(|| {
                    Error::Backend("charon_grouped_iqk: out has no device ptr".into())
                })? as *mut c_void,
                 0,
                out_storage.bytes(),
                 self.active_stream(),
    )
        })?;

        let wave = self.wavefront_size() as u32;
        let plan = plan_grouped_dispatch(sorted, wave);
        if plan.grid_x == 0 {
            return Ok(self.active_stream());
        }
        let grid_dim = HipDim3::new(plan.grid_x, 1, 1);
        let block_dim = HipDim3::new(plan.block_x, 1, 1);

        let mut tok_ptr = upload_device_buffer(self.ordinal, &sorted.sorted_token_ids)?;
        let mut exp_ptr = upload_device_buffer(self.ordinal, &sorted.sorted_expert_ids)?;
        let mut w_ptr = upload_device_buffer(self.ordinal, &sorted.sorted_weights)?;

        let mut a = act_storage.device_ptr.ok_or_else(|| {
            Error::Backend("charon_grouped_iqk: activations has no device ptr".into())
        })? as *mut c_void;
        let mut gw = egate_w_ptr;
        let mut uw = eup_w_ptr;
        let mut dw = edown_w_ptr;
        let mut ascale = a_scale_ptr;
        let mut optr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("charon_grouped_iqk: out has no device ptr".into()))?
            as *mut c_void;
        let mut hidden_i = hidden as i32;
        let mut inter_i = inter as i32;
        let mut num_tokens_i = sorted.num_tokens_post_padded as i32;
        let mut block_size_i = sorted.block_size as i32;
        let mut format_i = format_id as i32;
        let mut rsf = routed_scaling_factor;

        let stream = self.launch_compute_kernel(
            "grim_moe_fused_grouped_iqk",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut a),
                arg(&mut gw),
                arg(&mut uw),
                arg(&mut dw),
                arg(&mut ascale),
                arg(&mut tok_ptr),
                arg(&mut exp_ptr),
                arg(&mut w_ptr),
                arg(&mut optr),
                arg(&mut hidden_i),
                arg(&mut inter_i),
                arg(&mut num_tokens_i),
                arg(&mut block_size_i),
                arg(&mut format_i),
                arg(&mut rsf),
            ],
        )?;

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

    /// Launcher for CompressedTensors W8A8 INT8 grouped MoE kernel.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_charon_grouped_dispatch_w8a8_int8(
        &self,
        act_storage: &RocmStorage,
        egate_w_ptr: u64,
        eup_w_ptr: u64,
        edown_w_ptr: u64,
        a_scale_ptr: u64,
        sorted: &crate::kernels::charon::SortedRouting,
        out_storage: &RocmStorage,
        hidden: usize,
        inter: usize,
        num_experts: usize,
        routed_scaling_factor: f32,
    ) -> Result<*mut c_void> {
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        use crate::kernels::charon::plan_grouped_dispatch;
        let _ = num_experts;

        check_hip("charon_grouped_w8a8_int8 hipMemset(output, 0)", unsafe {
            hipMemsetAsync(
                out_storage.device_ptr.ok_or_else(|| {
                    Error::Backend("charon_grouped_w8a8_int8: out has no device ptr".into())
                })? as *mut c_void,
                 0,
                out_storage.bytes(),
                 self.active_stream(),
    )
        })?;

        let wave = self.wavefront_size() as u32;
        let plan = plan_grouped_dispatch(sorted, wave);
        if plan.grid_x == 0 {
            return Ok(self.active_stream());
        }
        let grid_dim = HipDim3::new(plan.grid_x, 1, 1);
        let block_dim = HipDim3::new(plan.block_x, 1, 1);

        let mut tok_ptr = upload_device_buffer(self.ordinal, &sorted.sorted_token_ids)?;
        let mut exp_ptr = upload_device_buffer(self.ordinal, &sorted.sorted_expert_ids)?;
        let mut w_ptr = upload_device_buffer(self.ordinal, &sorted.sorted_weights)?;

        let mut a = act_storage.device_ptr.ok_or_else(|| {
            Error::Backend("charon_grouped_w8a8_int8: activations has no device ptr".into())
        })? as *mut c_void;
        let mut gw = egate_w_ptr;
        let mut uw = eup_w_ptr;
        let mut dw = edown_w_ptr;
        let mut ascale = a_scale_ptr;
        let mut optr = out_storage.device_ptr.ok_or_else(|| {
            Error::Backend("charon_grouped_w8a8_int8: out has no device ptr".into())
        })? as *mut c_void;
        let mut hidden_i = hidden as i32;
        let mut inter_i = inter as i32;
        let mut num_tokens_i = sorted.num_tokens_post_padded as i32;
        let mut block_size_i = sorted.block_size as i32;
        let mut rsf = routed_scaling_factor;

        let stream = self.launch_compute_kernel(
            "grim_moe_fused_grouped_w8a8_int8",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut a),
                arg(&mut gw),
                arg(&mut uw),
                arg(&mut dw),
                arg(&mut ascale),
                arg(&mut tok_ptr),
                arg(&mut exp_ptr),
                arg(&mut w_ptr),
                arg(&mut optr),
                arg(&mut hidden_i),
                arg(&mut inter_i),
                arg(&mut num_tokens_i),
                arg(&mut block_size_i),
                arg(&mut rsf),
            ],
        )?;

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

    /// Launcher for CompressedTensors W8A8 FP8 grouped MoE kernel.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_charon_grouped_dispatch_w8a8_fp8(
        &self,
        act_storage: &RocmStorage,
        egate_w_ptr: u64,
        eup_w_ptr: u64,
        edown_w_ptr: u64,
        a_scale_ptr: u64,
        sorted: &crate::kernels::charon::SortedRouting,
        out_storage: &RocmStorage,
        hidden: usize,
        inter: usize,
        num_experts: usize,
        routed_scaling_factor: f32,
    ) -> Result<*mut c_void> {
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        use crate::kernels::charon::plan_grouped_dispatch;
        let _ = num_experts;

        check_hip("charon_grouped_w8a8_fp8 hipMemset(output, 0)", unsafe {
            hipMemsetAsync(
                out_storage.device_ptr.ok_or_else(|| {
                    Error::Backend("charon_grouped_w8a8_fp8: out has no device ptr".into())
                })? as *mut c_void,
                 0,
                out_storage.bytes(),
                 self.active_stream(),
    )
        })?;

        let wave = self.wavefront_size() as u32;
        let plan = plan_grouped_dispatch(sorted, wave);
        if plan.grid_x == 0 {
            return Ok(self.active_stream());
        }
        let grid_dim = HipDim3::new(plan.grid_x, 1, 1);
        let block_dim = HipDim3::new(plan.block_x, 1, 1);

        let mut tok_ptr = upload_device_buffer(self.ordinal, &sorted.sorted_token_ids)?;
        let mut exp_ptr = upload_device_buffer(self.ordinal, &sorted.sorted_expert_ids)?;
        let mut w_ptr = upload_device_buffer(self.ordinal, &sorted.sorted_weights)?;

        let mut a = act_storage.device_ptr.ok_or_else(|| {
            Error::Backend("charon_grouped_w8a8_fp8: activations has no device ptr".into())
        })? as *mut c_void;
        let mut gw = egate_w_ptr;
        let mut uw = eup_w_ptr;
        let mut dw = edown_w_ptr;
        let mut ascale = a_scale_ptr;
        let mut optr = out_storage.device_ptr.ok_or_else(|| {
            Error::Backend("charon_grouped_w8a8_fp8: out has no device ptr".into())
        })? as *mut c_void;
        let mut hidden_i = hidden as i32;
        let mut inter_i = inter as i32;
        let mut num_tokens_i = sorted.num_tokens_post_padded as i32;
        let mut block_size_i = sorted.block_size as i32;
        let mut rsf = routed_scaling_factor;

        let stream = self.launch_compute_kernel(
            "grim_moe_fused_grouped_w8a8_fp8",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut a),
                arg(&mut gw),
                arg(&mut uw),
                arg(&mut dw),
                arg(&mut ascale),
                arg(&mut tok_ptr),
                arg(&mut exp_ptr),
                arg(&mut w_ptr),
                arg(&mut optr),
                arg(&mut hidden_i),
                arg(&mut inter_i),
                arg(&mut num_tokens_i),
                arg(&mut block_size_i),
                arg(&mut rsf),
            ],
        )?;

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

    /// Launcher for AWQ grouped MoE kernel.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_charon_grouped_dispatch_awq(
        &self,
        act_storage: &RocmStorage,
        egate_w_ptr: u64,
        eup_w_ptr: u64,
        edown_w_ptr: u64,
        a_scale_ptr: u64,
        sorted: &crate::kernels::charon::SortedRouting,
        out_storage: &RocmStorage,
        hidden: usize,
        inter: usize,
        num_experts: usize,
        bits: u8,
        group_size: usize,
        gate_qw_off: i64,
        gate_qz_off: i64,
        gate_sc_off: i64,
        gate_stride: u64,
        down_qw_off: i64,
        down_qz_off: i64,
        down_sc_off: i64,
        down_stride: u64,
        routed_scaling_factor: f32,
    ) -> Result<*mut c_void> {
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        use crate::kernels::charon::plan_grouped_dispatch;
        let _ = num_experts;

        check_hip("charon_grouped_awq hipMemset(output, 0)", unsafe {
            hipMemsetAsync(
                out_storage.device_ptr.ok_or_else(|| {
                    Error::Backend("charon_grouped_awq: out has no device ptr".into())
                })? as *mut c_void,
                 0,
                out_storage.bytes(),
                 self.active_stream(),
    )
        })?;

        let wave = self.wavefront_size() as u32;
        let plan = plan_grouped_dispatch(sorted, wave);
        if plan.grid_x == 0 {
            return Ok(self.active_stream());
        }
        let grid_dim = HipDim3::new(plan.grid_x, 1, 1);
        let block_dim = HipDim3::new(plan.block_x, 1, 1);

        let mut tok_ptr = upload_device_buffer(self.ordinal, &sorted.sorted_token_ids)?;
        let mut exp_ptr = upload_device_buffer(self.ordinal, &sorted.sorted_expert_ids)?;
        let mut w_ptr = upload_device_buffer(self.ordinal, &sorted.sorted_weights)?;

        let mut a = act_storage.device_ptr.ok_or_else(|| {
            Error::Backend("charon_grouped_awq: activations has no device ptr".into())
        })? as *mut c_void;
        let mut gw = egate_w_ptr;
        let mut uw = eup_w_ptr;
        let mut dw = edown_w_ptr;
        let mut ascale = a_scale_ptr;
        let mut optr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("charon_grouped_awq: out has no device ptr".into()))?
            as *mut c_void;
        let mut hidden_i = hidden as i32;
        let mut inter_i = inter as i32;
        let mut num_tokens_i = sorted.num_tokens_post_padded as i32;
        let mut block_size_i = sorted.block_size as i32;
        let mut bits_i = bits as i32;
        let mut group_size_i = group_size as i32;

        let mut g_qw = gate_qw_off;
        let mut g_qz = gate_qz_off;
        let mut g_sc = gate_sc_off;
        let mut g_str = gate_stride;

        let mut d_qw = down_qw_off;
        let mut d_qz = down_qz_off;
        let mut d_sc = down_sc_off;
        let mut d_str = down_stride;

        let mut rsf = routed_scaling_factor;

        let stream = self.launch_compute_kernel(
            "grim_moe_fused_grouped_awq",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut a),
                arg(&mut gw),
                arg(&mut uw),
                arg(&mut dw),
                arg(&mut ascale),
                arg(&mut tok_ptr),
                arg(&mut exp_ptr),
                arg(&mut w_ptr),
                arg(&mut optr),
                arg(&mut hidden_i),
                arg(&mut inter_i),
                arg(&mut num_tokens_i),
                arg(&mut block_size_i),
                arg(&mut bits_i),
                arg(&mut group_size_i),
                arg(&mut g_qw),
                arg(&mut g_qz),
                arg(&mut g_sc),
                arg(&mut g_str),
                arg(&mut d_qw),
                arg(&mut d_qz),
                arg(&mut d_sc),
                arg(&mut d_str),
                arg(&mut rsf),
            ],
        )?;

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

    /// (`tests/golden_charon_moe_gpu.rs`, G-A4).
    /// Takes plain host `f32` buffers + a routing assignment, uploads them, zeros the output, launches.
    pub fn charon_fused_dispatch_roundtrip(
        &self,
        activations: &[f32],
        expert_gate_w: &[f32],
        expert_up_w: &[f32],
        expert_down_w: &[f32],
        assignment: &crate::kernels::charon::RoutingAssignment,
        batch: usize,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
    ) -> Result<Vec<f32>> {
        let act_shape = Shape::new(vec![batch, hidden]);
        let exp_gate_shape = Shape::new(vec![expert_gate_w.len()]);
        let exp_up_shape = Shape::new(vec![expert_up_w.len()]);
        let exp_down_shape = Shape::new(vec![expert_down_w.len()]);
        let out_shape = Shape::new(vec![batch, hidden]);

        // Upload host buffers to the device.
        let act_storage: Box<dyn BackendStorage> =
            CoreTensorOps::from_cpu(self, activations, &act_shape, DType::F32)?;
        let gw_storage: Box<dyn BackendStorage> =
            CoreTensorOps::from_cpu(self, expert_gate_w, &exp_gate_shape, DType::F32)?;
        let uw_storage: Box<dyn BackendStorage> =
            CoreTensorOps::from_cpu(self, expert_up_w, &exp_up_shape, DType::F32)?;
        let dw_storage: Box<dyn BackendStorage> =
            CoreTensorOps::from_cpu(self, expert_down_w, &exp_down_shape, DType::F32)?;
        let out_storage: Box<dyn BackendStorage> =
            MemoryOps::alloc_storage(self, &out_shape, DType::F32)?;

        // Downcast to RocmStorage to reach device pointers + the launcher.
        let act_s = as_rocm(act_storage.as_ref())?;
        let gw_s = as_rocm(gw_storage.as_ref())?;
        let uw_s = as_rocm(uw_storage.as_ref())?;
        let dw_s = as_rocm(dw_storage.as_ref())?;
        let out_s = as_rocm(out_storage.as_ref())?;

        let gw_ptr = dev_ptr(gw_s)?;
        let uw_ptr = dev_ptr(uw_s)?;
        let dw_ptr = dev_ptr(dw_s)?;

        self.launch_charon_fused_dispatch(
            act_s,
            gw_ptr,
            uw_ptr,
            dw_ptr,
            assignment,
            out_s,
            hidden,
            inter,
            routed_scaling_factor,
        )?;
        self.synchronize();
        out_storage.to_cpu_vec_f32()
    }

    /// Host-to-host roundtrip for the #1 token-sorted (grouped) fused MoE dispatch.
    /// Mirrors `charon_fused_dispatch_roundtrip` but token-sorts the routing (vLLM `moe_align_block_size`) and launches `grim_moe_fused_grouped`.
    pub fn charon_grouped_dispatch_roundtrip(
        &self,
        activations: &[f32],
        expert_gate_w: &[f32],
        expert_up_w: &[f32],
        expert_down_w: &[f32],
        assignment: &crate::kernels::charon::RoutingAssignment,
        batch: usize,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
    ) -> Result<Vec<f32>> {
        let num_experts = expert_gate_w.len() / (inter * hidden);
        let block_size = 128usize; // token-block the grouped kernel strides across

        let sorted =
            crate::kernels::charon::moe_align_block_size(assignment, block_size, num_experts);

        let act_shape = Shape::new(vec![batch, hidden]);
        let exp_gate_shape = Shape::new(vec![expert_gate_w.len()]);
        let exp_up_shape = Shape::new(vec![expert_up_w.len()]);
        let exp_down_shape = Shape::new(vec![expert_down_w.len()]);
        let out_shape = Shape::new(vec![batch, hidden]);

        let act_storage: Box<dyn BackendStorage> =
            CoreTensorOps::from_cpu(self, activations, &act_shape, DType::F32)?;
        let gw_storage: Box<dyn BackendStorage> =
            CoreTensorOps::from_cpu(self, expert_gate_w, &exp_gate_shape, DType::F32)?;
        let uw_storage: Box<dyn BackendStorage> =
            CoreTensorOps::from_cpu(self, expert_up_w, &exp_up_shape, DType::F32)?;
        let dw_storage: Box<dyn BackendStorage> =
            CoreTensorOps::from_cpu(self, expert_down_w, &exp_down_shape, DType::F32)?;
        let out_storage: Box<dyn BackendStorage> =
            MemoryOps::alloc_storage(self, &out_shape, DType::F32)?;

        let act_s = as_rocm(act_storage.as_ref())?;
        let gw_s = as_rocm(gw_storage.as_ref())?;
        let uw_s = as_rocm(uw_storage.as_ref())?;
        let dw_s = as_rocm(dw_storage.as_ref())?;
        let out_s = as_rocm(out_storage.as_ref())?;

        let gw_ptr = dev_ptr(gw_s)?;
        let uw_ptr = dev_ptr(uw_s)?;
        let dw_ptr = dev_ptr(dw_s)?;

        self.launch_charon_grouped_dispatch(
            act_s,
            gw_ptr,
            uw_ptr,
            dw_ptr,
            &sorted,
            out_s,
            hidden,
            inter,
            routed_scaling_factor,
            num_experts,
        )?;
        self.synchronize();
        out_storage.to_cpu_vec_f32()
    }

    /// WI-F3 - WMMA grouped MoE dispatch roundtrip: sorts via `moe_align_block_size`, then launches the WMMA/tensor-core grouped kernel (`grim_moe_fused_grouped_wmma`, the `CharonVariant::LargeGroupPrefill` dispatch target via `kernels::charon::grouped_dispatch_entry`) and reads the result back.
    /// On non-WMMA arches (gfx1036/RDNA2) the kernel compiles to the scalar fallback, so this roundtrip is.
    pub fn charon_grouped_dispatch_wmma_roundtrip(
        &self,
        activations: &[f32],
        expert_gate_w: &[f32],
        expert_up_w: &[f32],
        expert_down_w: &[f32],
        assignment: &crate::kernels::charon::RoutingAssignment,
        batch: usize,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
    ) -> Result<Vec<f32>> {
        let num_experts = expert_gate_w.len() / (inter * hidden);
        let block_size = 128usize;

        let sorted =
            crate::kernels::charon::moe_align_block_size(assignment, block_size, num_experts);

        let act_shape = Shape::new(vec![batch, hidden]);
        let exp_gate_shape = Shape::new(vec![expert_gate_w.len()]);
        let exp_up_shape = Shape::new(vec![expert_up_w.len()]);
        let exp_down_shape = Shape::new(vec![expert_down_w.len()]);
        let out_shape = Shape::new(vec![batch, hidden]);

        let act_storage: Box<dyn BackendStorage> =
            CoreTensorOps::from_cpu(self, activations, &act_shape, DType::F32)?;
        let gw_storage: Box<dyn BackendStorage> =
            CoreTensorOps::from_cpu(self, expert_gate_w, &exp_gate_shape, DType::F32)?;
        let uw_storage: Box<dyn BackendStorage> =
            CoreTensorOps::from_cpu(self, expert_up_w, &exp_up_shape, DType::F32)?;
        let dw_storage: Box<dyn BackendStorage> =
            CoreTensorOps::from_cpu(self, expert_down_w, &exp_down_shape, DType::F32)?;
        let out_storage: Box<dyn BackendStorage> =
            MemoryOps::alloc_storage(self, &out_shape, DType::F32)?;

        let act_s = as_rocm(act_storage.as_ref())?;
        let gw_s = as_rocm(gw_storage.as_ref())?;
        let uw_s = as_rocm(uw_storage.as_ref())?;
        let dw_s = as_rocm(dw_storage.as_ref())?;
        let out_s = as_rocm(out_storage.as_ref())?;

        let entry = crate::kernels::charon::grouped_dispatch_entry(
            crate::kernels::charon::CharonVariant::LargeGroupPrefill,
        );
        self.launch_charon_grouped_dispatch_entry(
            act_s,
            dev_ptr(gw_s)?,
            dev_ptr(uw_s)?,
            dev_ptr(dw_s)?,
            &sorted,
            out_s,
            hidden,
            inter,
            routed_scaling_factor,
            num_experts,
            entry,
        )?;
        self.synchronize();
        out_storage.to_cpu_vec_f32()
    }

    // Charon MoE backward launcher (P2 - WI-Charon-1 device dispatch)

    /// Device launcher for the FP32 Charon MoE backward kernel (`grim_moe_fused_grouped_backward`).
    /// Mirrors `launch_charon_grouped_dispatch`: validates inputs, zero-initialises the four atomicAdd output buffers, plans the grouped grid/block from.
    pub(crate) fn launch_charon_grouped_backward(
        &self,
        activations: &RocmStorage,
        expert_gate_w_ptr: u64,
        expert_up_w_ptr: u64,
        expert_down_w_ptr: u64,
        d_y: &RocmStorage,
        d_gate_w: &RocmStorage,
        d_up_w: &RocmStorage,
        d_down_w: &RocmStorage,
        d_x: &RocmStorage,
        sorted: &crate::kernels::charon::SortedRouting,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
        _num_experts: usize,
    ) -> Result<*mut c_void> {
        // P1-3: raw HIP ops below bind to the calling thread's current
        // device — pin to the owning ordinal (see matmul_op fix, 2026-08-23e).
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let a_ptr = activations.device_ptr.ok_or_else(|| {
            Error::Backend("charon_grouped_backward: activations has no device ptr".into())
        })?;
        let dy_ptr = d_y.device_ptr.ok_or_else(|| {
            Error::Backend("charon_grouped_backward: d_y has no device ptr".into())
        })?;
        let dgw_ptr = d_gate_w.device_ptr.ok_or_else(|| {
            Error::Backend("charon_grouped_backward: d_gate_w has no device ptr".into())
        })?;
        let duw_ptr = d_up_w.device_ptr.ok_or_else(|| {
            Error::Backend("charon_grouped_backward: d_up_w has no device ptr".into())
        })?;
        let ddw_ptr = d_down_w.device_ptr.ok_or_else(|| {
            Error::Backend("charon_grouped_backward: d_down_w has no device ptr".into())
        })?;
        let dx_ptr = d_x.device_ptr.ok_or_else(|| {
            Error::Backend("charon_grouped_backward: d_x has no device ptr".into())
        })?;

        crate::kernels::charon_backward::validate_backward_inputs(
            expert_gate_w_ptr as *const c_void,
            expert_up_w_ptr as *const c_void,
            expert_down_w_ptr as *const c_void,
            dy_ptr as *const c_void,
            dgw_ptr as *mut c_void,
            duw_ptr as *mut c_void,
            ddw_ptr as *mut c_void,
            dx_ptr as *mut c_void,
            hidden,
            inter,
            sorted.num_tokens_post_padded,
            sorted.block_size,
        )?;

        // All four output buffers are accumulated via atomicAdd; zero first.
        check_hip("charon_backward hipMemset(d_gate_w, 0)", unsafe {
            hipMemsetAsync(dgw_ptr as *mut c_void, 0, d_gate_w.bytes(), self.active_stream())
        })?;
        check_hip("charon_backward hipMemset(d_up_w, 0)", unsafe {
            hipMemsetAsync(duw_ptr as *mut c_void, 0, d_up_w.bytes(), self.active_stream())
        })?;
        check_hip("charon_backward hipMemset(d_down_w, 0)", unsafe {
            hipMemsetAsync(ddw_ptr as *mut c_void, 0, d_down_w.bytes(), self.active_stream())
        })?;
        check_hip("charon_backward hipMemset(d_x, 0)", unsafe {
            hipMemsetAsync(dx_ptr as *mut c_void, 0, d_x.bytes(), self.active_stream())
        })?;

        let wave = self.wavefront_size() as u32;
        let plan = crate::kernels::charon::plan_grouped_dispatch(sorted, wave);
        if plan.grid_x == 0 {
            return Ok(self.active_stream());
        }
        let grid_dim = HipDim3::new(plan.grid_x, 1, 1);
        let block_dim = HipDim3::new(plan.block_x, 1, 1);

        let mut tok_ptr = upload_device_buffer(self.ordinal, &sorted.sorted_token_ids)?;
        let mut exp_ptr = upload_device_buffer(self.ordinal, &sorted.sorted_expert_ids)?;
        let mut w_ptr = upload_device_buffer(self.ordinal, &sorted.sorted_weights)?;

        // Kernel arg order matches grim_moe_fused_grouped_backward signature: activations, gate_w, up_w, down_w, d_y,
        // d_gate_w, d_up_w, d_down_w, d_x, sorted_token_ids, sorted_expert_ids, sorted_weights, hidden, inter, num_tokens, block_size, routed_scaling_factor
        let mut a = a_ptr as *mut c_void;
        let mut gw = expert_gate_w_ptr as *mut c_void;
        let mut uw = expert_up_w_ptr as *mut c_void;
        let mut dw = expert_down_w_ptr as *mut c_void;
        let mut dy = dy_ptr as *mut c_void;
        let mut dgw = dgw_ptr as *mut c_void;
        let mut duw = duw_ptr as *mut c_void;
        let mut ddw = ddw_ptr as *mut c_void;
        let mut dx = dx_ptr as *mut c_void;
        let mut hidden_i = hidden as i32;
        let mut inter_i = inter as i32;
        let mut num_tokens_i = sorted.num_tokens_post_padded as i32;
        let mut block_size_i = sorted.block_size as i32;
        let mut rsf = routed_scaling_factor;

        let stream = self.launch_compute_kernel(
            "grim_moe_fused_grouped_backward",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut a),
                arg(&mut gw),
                arg(&mut uw),
                arg(&mut dw),
                arg(&mut dy),
                arg(&mut dgw),
                arg(&mut duw),
                arg(&mut ddw),
                arg(&mut dx),
                arg(&mut tok_ptr),
                arg(&mut exp_ptr),
                arg(&mut w_ptr),
                arg(&mut hidden_i),
                arg(&mut inter_i),
                arg(&mut num_tokens_i),
                arg(&mut block_size_i),
                arg(&mut rsf),
            ],
        )?;

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

    /// Host-to-host roundtrip for the Charon MoE backward kernel.
    /// Uploads all inputs (activations, expert weights, d_y) and the sorted routing arrays to the device,.
    pub fn charon_grouped_backward_roundtrip(
        &self,
        activations: &[f32],
        expert_gate_w: &[f32],
        expert_up_w: &[f32],
        expert_down_w: &[f32],
        d_y: &[f32],
        assignment: &crate::kernels::charon::RoutingAssignment,
        batch: usize,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
    ) -> Result<CharonBackwardResult> {
        let num_experts = expert_gate_w.len() / (inter * hidden);
        let block_size = 128usize;

        let sorted =
            crate::kernels::charon::moe_align_block_size(assignment, block_size, num_experts);

        let act_shape = Shape::new(vec![batch, hidden]);
        let gw_shape = Shape::new(vec![expert_gate_w.len()]);
        let uw_shape = Shape::new(vec![expert_up_w.len()]);
        let dw_shape = Shape::new(vec![expert_down_w.len()]);
        let dy_shape = Shape::new(vec![d_y.len()]);

        let dgw_shape = Shape::new(vec![num_experts * inter * hidden]);
        let duw_shape = Shape::new(vec![num_experts * inter * hidden]);
        let ddw_shape = Shape::new(vec![num_experts * hidden * inter]);
        let dx_shape = Shape::new(vec![batch * hidden]);

        let act_storage: Box<dyn BackendStorage> =
            CoreTensorOps::from_cpu(self, activations, &act_shape, DType::F32)?;
        let gw_storage: Box<dyn BackendStorage> =
            CoreTensorOps::from_cpu(self, expert_gate_w, &gw_shape, DType::F32)?;
        let uw_storage: Box<dyn BackendStorage> =
            CoreTensorOps::from_cpu(self, expert_up_w, &uw_shape, DType::F32)?;
        let dw_storage: Box<dyn BackendStorage> =
            CoreTensorOps::from_cpu(self, expert_down_w, &dw_shape, DType::F32)?;
        let dy_storage: Box<dyn BackendStorage> =
            CoreTensorOps::from_cpu(self, d_y, &dy_shape, DType::F32)?;

        // Output grad buffers — allocated (not uploaded), kernel fills them.
        let dgw_storage: Box<dyn BackendStorage> =
            MemoryOps::alloc_storage(self, &dgw_shape, DType::F32)?;
        let duw_storage: Box<dyn BackendStorage> =
            MemoryOps::alloc_storage(self, &duw_shape, DType::F32)?;
        let ddw_storage: Box<dyn BackendStorage> =
            MemoryOps::alloc_storage(self, &ddw_shape, DType::F32)?;
        let dx_storage: Box<dyn BackendStorage> =
            MemoryOps::alloc_storage(self, &dx_shape, DType::F32)?;

        let act_s = as_rocm(act_storage.as_ref())?;
        let gw_s = as_rocm(gw_storage.as_ref())?;
        let uw_s = as_rocm(uw_storage.as_ref())?;
        let dw_s = as_rocm(dw_storage.as_ref())?;
        let dy_s = as_rocm(dy_storage.as_ref())?;
        let dgw_s = as_rocm(dgw_storage.as_ref())?;
        let duw_s = as_rocm(duw_storage.as_ref())?;
        let ddw_s = as_rocm(ddw_storage.as_ref())?;
        let dx_s = as_rocm(dx_storage.as_ref())?;

        let gw_ptr = dev_ptr(gw_s)?;
        let uw_ptr = dev_ptr(uw_s)?;
        let dw_ptr = dev_ptr(dw_s)?;

        self.launch_charon_grouped_backward(
            act_s,
            gw_ptr,
            uw_ptr,
            dw_ptr,
            dy_s,
            dgw_s,
            duw_s,
            ddw_s,
            dx_s,
            &sorted,
            hidden,
            inter,
            routed_scaling_factor,
            num_experts,
        )?;
        // SPEED-ROC-9: device-wide sync preserves the "gradients are settled
        // when this returns" contract without the D2H — the buffers stay in
        // VRAM for the optimizer / gradient-reduction stage. Host consumers
        // call CharonBackwardResult::to_cpu().
        self.synchronize();

        Ok(CharonBackwardResult {
            d_gate_w: dgw_storage,
            d_up_w: duw_storage,
            d_down_w: ddw_storage,
            d_x: dx_storage,
        })
    }

    /// Host-to-host roundtrip for the #3 MXFP4 (E2M1 + E8M0) token-sorted grouped dispatch.
    /// Takes packed E2M1 weight codes + E8M0 shared-exponent bytes (one exp per 32-element group along.
    pub fn charon_grouped_dispatch_roundtrip_mxfp4(
        &self,
        activations: &[f32],
        expert_gate_w_codes: &[u8], // packed E2M1, [num_experts, inter*hidden/2]
        expert_up_w_codes: &[u8],
        expert_down_w_codes: &[u8],
        expert_gate_e8m0: &[u8], // [num_experts, inter*hidden/32]
        expert_up_e8m0: &[u8],
        expert_down_e8m0: &[u8],
        a_scale: &[f32],
        assignment: &crate::kernels::charon::RoutingAssignment,
        batch: usize,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
    ) -> Result<Vec<f32>> {
        // num_experts from the packed gate-code layout: [num_experts, inter*hidden/2].
        let num_experts = expert_gate_w_codes.len() / ((inter * hidden / 2).max(1));
        let block_size = 128usize;

        let sorted =
            crate::kernels::charon::moe_align_block_size(assignment, block_size, num_experts);

        let act_shape = Shape::new(vec![batch, hidden]);
        let gw_shape = Shape::new(vec![expert_gate_w_codes.len()]);
        let uw_shape = Shape::new(vec![expert_up_w_codes.len()]);
        let dw_shape = Shape::new(vec![expert_down_w_codes.len()]);
        let ge_shape = Shape::new(vec![expert_gate_e8m0.len()]);
        let ue_shape = Shape::new(vec![expert_up_e8m0.len()]);
        let de_shape = Shape::new(vec![expert_down_e8m0.len()]);
        let as_shape = Shape::new(vec![a_scale.len()]);
        let out_shape = Shape::new(vec![batch, hidden]);

        let act_storage: Box<dyn BackendStorage> =
            CoreTensorOps::from_cpu(self, activations, &act_shape, DType::F32)?;
        let gw_storage: Box<dyn BackendStorage> = MemoryOps::from_cpu_bytes(
            self,
            expert_gate_w_codes,
            &gw_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let uw_storage: Box<dyn BackendStorage> = MemoryOps::from_cpu_bytes(
            self,
            expert_up_w_codes,
            &uw_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let dw_storage: Box<dyn BackendStorage> = MemoryOps::from_cpu_bytes(
            self,
            expert_down_w_codes,
            &dw_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let ge_storage: Box<dyn BackendStorage> = MemoryOps::from_cpu_bytes(
            self,
            expert_gate_e8m0,
            &ge_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let ue_storage: Box<dyn BackendStorage> = MemoryOps::from_cpu_bytes(
            self,
            expert_up_e8m0,
            &ue_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let de_storage: Box<dyn BackendStorage> = MemoryOps::from_cpu_bytes(
            self,
            expert_down_e8m0,
            &de_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let as_storage: Box<dyn BackendStorage> =
            CoreTensorOps::from_cpu(self, a_scale, &as_shape, DType::F32)?;
        let out_storage: Box<dyn BackendStorage> =
            MemoryOps::alloc_storage(self, &out_shape, DType::F32)?;

        let act_s = as_rocm(act_storage.as_ref())?;
        let gw_s = as_rocm(gw_storage.as_ref())?;
        let uw_s = as_rocm(uw_storage.as_ref())?;
        let dw_s = as_rocm(dw_storage.as_ref())?;
        let ge_s = as_rocm(ge_storage.as_ref())?;
        let ue_s = as_rocm(ue_storage.as_ref())?;
        let de_s = as_rocm(de_storage.as_ref())?;
        let as_s = as_rocm(as_storage.as_ref())?;
        let out_s = as_rocm(out_storage.as_ref())?;

        let gw_ptr = dev_ptr(gw_s)?;
        let uw_ptr = dev_ptr(uw_s)?;
        let dw_ptr = dev_ptr(dw_s)?;
        let ge_ptr = dev_ptr(ge_s)?;
        let ue_ptr = dev_ptr(ue_s)?;
        let de_ptr = dev_ptr(de_s)?;
        let as_ptr = dev_ptr(as_s)?;

        self.launch_charon_grouped_dispatch_mxfp4(
            act_s,
            gw_ptr,
            uw_ptr,
            dw_ptr,
            ge_ptr,
            ue_ptr,
            de_ptr,
            as_ptr,
            &sorted,
            out_s,
            hidden,
            inter,
            num_experts,
            routed_scaling_factor,
        )?;
        self.synchronize();
        out_storage.to_cpu_vec_f32()
    }

    /// Host-to-host roundtrip for the #4 MXFP8 (E4M3 + E8M0) token-sorted grouped dispatch.
    /// Takes E4M3 weight codes (1 byte each, NOT packed) + one E8M0 shared-exponent byte per.
    pub fn charon_grouped_dispatch_roundtrip_mxfp8(
        &self,
        activations: &[f32],
        expert_gate_w_codes: &[u8], // E4M3, [num_experts, inter*hidden]
        expert_up_w_codes: &[u8],
        expert_down_w_codes: &[u8],
        expert_gate_e8m0: &[u8], // [num_experts, inter*hidden/32]
        expert_up_e8m0: &[u8],
        expert_down_e8m0: &[u8],
        a_scale: &[f32],
        assignment: &crate::kernels::charon::RoutingAssignment,
        batch: usize,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
    ) -> Result<Vec<f32>> {
        // num_experts from the E4M3 gate-code layout: [num_experts, inter*hidden].
        let num_experts = expert_gate_w_codes.len() / ((inter * hidden).max(1));
        let block_size = 128usize;

        let sorted =
            crate::kernels::charon::moe_align_block_size(assignment, block_size, num_experts);

        let act_shape = Shape::new(vec![batch, hidden]);
        let gw_shape = Shape::new(vec![expert_gate_w_codes.len()]);
        let uw_shape = Shape::new(vec![expert_up_w_codes.len()]);
        let dw_shape = Shape::new(vec![expert_down_w_codes.len()]);
        let ge_shape = Shape::new(vec![expert_gate_e8m0.len()]);
        let ue_shape = Shape::new(vec![expert_up_e8m0.len()]);
        let de_shape = Shape::new(vec![expert_down_e8m0.len()]);
        let as_shape = Shape::new(vec![a_scale.len()]);
        let out_shape = Shape::new(vec![batch, hidden]);

        let act_storage: Box<dyn BackendStorage> =
            CoreTensorOps::from_cpu(self, activations, &act_shape, DType::F32)?;
        let gw_storage: Box<dyn BackendStorage> = MemoryOps::from_cpu_bytes(
            self,
            expert_gate_w_codes,
            &gw_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let uw_storage: Box<dyn BackendStorage> = MemoryOps::from_cpu_bytes(
            self,
            expert_up_w_codes,
            &uw_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let dw_storage: Box<dyn BackendStorage> = MemoryOps::from_cpu_bytes(
            self,
            expert_down_w_codes,
            &dw_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let ge_storage: Box<dyn BackendStorage> = MemoryOps::from_cpu_bytes(
            self,
            expert_gate_e8m0,
            &ge_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let ue_storage: Box<dyn BackendStorage> = MemoryOps::from_cpu_bytes(
            self,
            expert_up_e8m0,
            &ue_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let de_storage: Box<dyn BackendStorage> = MemoryOps::from_cpu_bytes(
            self,
            expert_down_e8m0,
            &de_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let as_storage: Box<dyn BackendStorage> =
            CoreTensorOps::from_cpu(self, a_scale, &as_shape, DType::F32)?;
        let out_storage: Box<dyn BackendStorage> =
            MemoryOps::alloc_storage(self, &out_shape, DType::F32)?;

        let act_s = as_rocm(act_storage.as_ref())?;
        let gw_s = as_rocm(gw_storage.as_ref())?;
        let uw_s = as_rocm(uw_storage.as_ref())?;
        let dw_s = as_rocm(dw_storage.as_ref())?;
        let ge_s = as_rocm(ge_storage.as_ref())?;
        let ue_s = as_rocm(ue_storage.as_ref())?;
        let de_s = as_rocm(de_storage.as_ref())?;
        let as_s = as_rocm(as_storage.as_ref())?;
        let out_s = as_rocm(out_storage.as_ref())?;

        let gw_ptr = dev_ptr(gw_s)?;
        let uw_ptr = dev_ptr(uw_s)?;
        let dw_ptr = dev_ptr(dw_s)?;
        let ge_ptr = dev_ptr(ge_s)?;
        let ue_ptr = dev_ptr(ue_s)?;
        let de_ptr = dev_ptr(de_s)?;
        let as_ptr = dev_ptr(as_s)?;

        self.launch_charon_grouped_dispatch_mxfp8(
            act_s,
            gw_ptr,
            uw_ptr,
            dw_ptr,
            ge_ptr,
            ue_ptr,
            de_ptr,
            as_ptr,
            &sorted,
            out_s,
            hidden,
            inter,
            num_experts,
            routed_scaling_factor,
        )?;
        self.synchronize();
        out_storage.to_cpu_vec_f32()
    }

    /// Takes Q8_0 weight bytes (f16 scale + i8 per 32 weights) + per-token act scale, token-sorts via `moe_align_block_size`, and launches `grim_moe_fused_grouped_q80` reusing the identical in-register math.
    /// Used by the Q8_0-vs-FP32 KAT golden test (WI-5 / G-A4 extension).
    pub fn charon_grouped_dispatch_roundtrip_q80(
        &self,
        activations: &[f32],
        expert_gate_w_q80: &[u8],
        expert_up_w_q80: &[u8],
        expert_down_w_q80: &[u8],
        a_scale: &[f32],
        assignment: &crate::kernels::charon::RoutingAssignment,
        batch: usize,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
    ) -> Result<Vec<f32>> {
        // Q8_0 layout: per 32 weights a 2-byte f16 scale + 32 i8 => 34 bytes.
        let bytes_per_block = 34usize;
        let weights_per_expert = inter * hidden;
        let num_experts =
            expert_gate_w_q80.len() / (bytes_per_block * weights_per_expert.div_ceil(32));
        let block_size = 128usize;

        let sorted =
            crate::kernels::charon::moe_align_block_size(assignment, block_size, num_experts);

        let act_shape = Shape::new(vec![batch, hidden]);
        let gw_shape = Shape::new(vec![expert_gate_w_q80.len()]);
        let uw_shape = Shape::new(vec![expert_up_w_q80.len()]);
        let dw_shape = Shape::new(vec![expert_down_w_q80.len()]);
        let as_shape = Shape::new(vec![a_scale.len()]);
        let out_shape = Shape::new(vec![batch, hidden]);

        let act_storage: Box<dyn BackendStorage> =
            CoreTensorOps::from_cpu(self, activations, &act_shape, DType::F32)?;
        let gw_storage: Box<dyn BackendStorage> = MemoryOps::from_cpu_bytes(
            self,
            expert_gate_w_q80,
            &gw_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let uw_storage: Box<dyn BackendStorage> = MemoryOps::from_cpu_bytes(
            self,
            expert_up_w_q80,
            &uw_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let dw_storage: Box<dyn BackendStorage> = MemoryOps::from_cpu_bytes(
            self,
            expert_down_w_q80,
            &dw_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let as_storage: Box<dyn BackendStorage> =
            CoreTensorOps::from_cpu(self, a_scale, &as_shape, DType::F32)?;
        let out_storage: Box<dyn BackendStorage> =
            MemoryOps::alloc_storage(self, &out_shape, DType::F32)?;

        let act_s = as_rocm(act_storage.as_ref())?;
        let gw_s = as_rocm(gw_storage.as_ref())?;
        let uw_s = as_rocm(uw_storage.as_ref())?;
        let dw_s = as_rocm(dw_storage.as_ref())?;
        let as_s = as_rocm(as_storage.as_ref())?;
        let out_s = as_rocm(out_storage.as_ref())?;

        let gw_ptr = dev_ptr(gw_s)?;
        let uw_ptr = dev_ptr(uw_s)?;
        let dw_ptr = dev_ptr(dw_s)?;
        let as_ptr = dev_ptr(as_s)?;

        self.launch_charon_grouped_dispatch_q80(
            act_s,
            gw_ptr,
            uw_ptr,
            dw_ptr,
            as_ptr,
            &sorted,
            out_s,
            hidden,
            inter,
            num_experts,
            routed_scaling_factor,
        )?;
        self.synchronize();
        out_storage.to_cpu_vec_f32()
    }

    /// Generic host-to-host roundtrip for the unified IQ/K-quant token-sorted grouped dispatch.
    /// `format_id` selects the super-block decode (0 iq4nl ..
    pub fn charon_grouped_dispatch_roundtrip_iqk(
        &self,
        activations: &[f32],
        expert_gate_w_q: &[u8],
        expert_up_w_q: &[u8],
        expert_down_w_q: &[u8],
        a_scale: &[f32],
        assignment: &crate::kernels::charon::RoutingAssignment,
        batch: usize,
        hidden: usize,
        inter: usize,
        format_id: usize,
        block_bytes: usize,
        routed_scaling_factor: f32,
    ) -> Result<Vec<f32>> {
        // Each expert occupies one 256-weight super-block of `block_bytes`.
        let weights_per_expert = (inter * hidden).div_ceil(256) * 256;
        let num_experts = expert_gate_w_q.len() / (block_bytes * (weights_per_expert / 256).max(1));
        let block_size = 128usize;

        let sorted =
            crate::kernels::charon::moe_align_block_size(assignment, block_size, num_experts);

        let act_shape = Shape::new(vec![batch, hidden]);
        let gw_shape = Shape::new(vec![expert_gate_w_q.len()]);
        let uw_shape = Shape::new(vec![expert_up_w_q.len()]);
        let dw_shape = Shape::new(vec![expert_down_w_q.len()]);
        let as_shape = Shape::new(vec![a_scale.len()]);
        let out_shape = Shape::new(vec![batch, hidden]);

        let act_storage: Box<dyn BackendStorage> =
            CoreTensorOps::from_cpu(self, activations, &act_shape, DType::F32)?;
        let gw_storage: Box<dyn BackendStorage> = MemoryOps::from_cpu_bytes(
            self,
            expert_gate_w_q,
            &gw_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let uw_storage: Box<dyn BackendStorage> = MemoryOps::from_cpu_bytes(
            self,
            expert_up_w_q,
            &uw_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let dw_storage: Box<dyn BackendStorage> = MemoryOps::from_cpu_bytes(
            self,
            expert_down_w_q,
            &dw_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let as_storage: Box<dyn BackendStorage> =
            CoreTensorOps::from_cpu(self, a_scale, &as_shape, DType::F32)?;
        let out_storage: Box<dyn BackendStorage> =
            MemoryOps::alloc_storage(self, &out_shape, DType::F32)?;

        let act_s = as_rocm(act_storage.as_ref())?;
        let gw_s = as_rocm(gw_storage.as_ref())?;
        let uw_s = as_rocm(uw_storage.as_ref())?;
        let dw_s = as_rocm(dw_storage.as_ref())?;
        let as_s = as_rocm(as_storage.as_ref())?;
        let out_s = as_rocm(out_storage.as_ref())?;

        let gw_ptr = dev_ptr(gw_s)?;
        let uw_ptr = dev_ptr(uw_s)?;
        let dw_ptr = dev_ptr(dw_s)?;
        let as_ptr = dev_ptr(as_s)?;

        self.launch_charon_grouped_dispatch_iqk(
            act_s,
            gw_ptr,
            uw_ptr,
            dw_ptr,
            as_ptr,
            &sorted,
            out_s,
            hidden,
            inter,
            num_experts,
            format_id,
            routed_scaling_factor,
        )?;
        self.synchronize();
        out_storage.to_cpu_vec_f32()
    }

    /// token-sorts via `moe_align_block_size`, and launches `grim_moe_fused_grouped_fp8` reusing the identical in-register math.
    /// Used by the FP8-vs-FP32 KAT golden test (WI-A / G-A4 extension for WI-2).
    pub fn charon_grouped_dispatch_roundtrip_fp8(
        &self,
        activations: &[f32],
        expert_gate_w_fp8: &[u8],
        expert_up_w_fp8: &[u8],
        expert_down_w_fp8: &[u8],
        expert_gate_scale: &[f32],
        expert_up_scale: &[f32],
        expert_down_scale: &[f32],
        a_scale: &[f32],
        assignment: &crate::kernels::charon::RoutingAssignment,
        batch: usize,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
    ) -> Result<Vec<f32>> {
        // num_experts is derived from the gate-scale layout produced by the test:
        //   gate_scale is [num_experts, inter, hidden/16]  (block_size=16 along hidden)
        let h16 = hidden.div_ceil(16);
        let num_experts = expert_gate_scale.len() / (inter * h16.max(1));
        let block_size = 128usize;

        let sorted =
            crate::kernels::charon::moe_align_block_size(assignment, block_size, num_experts);

        let act_shape = Shape::new(vec![batch, hidden]);
        let gw_shape = Shape::new(vec![expert_gate_w_fp8.len()]);
        let uw_shape = Shape::new(vec![expert_up_w_fp8.len()]);
        let dw_shape = Shape::new(vec![expert_down_w_fp8.len()]);
        let gs_shape = Shape::new(vec![expert_gate_scale.len()]);
        let us_shape = Shape::new(vec![expert_up_scale.len()]);
        let ds_shape = Shape::new(vec![expert_down_scale.len()]);
        let as_shape = Shape::new(vec![a_scale.len()]);
        let out_shape = Shape::new(vec![batch, hidden]);

        let act_storage: Box<dyn BackendStorage> =
            CoreTensorOps::from_cpu(self, activations, &act_shape, DType::F32)?;
        // FP8 weights are uploaded as raw U8 blobs (no DType::F8 on this path).
        let gw_storage: Box<dyn BackendStorage> = MemoryOps::from_cpu_bytes(
            self,
            expert_gate_w_fp8,
            &gw_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let uw_storage: Box<dyn BackendStorage> = MemoryOps::from_cpu_bytes(
            self,
            expert_up_w_fp8,
            &uw_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let dw_storage: Box<dyn BackendStorage> = MemoryOps::from_cpu_bytes(
            self,
            expert_down_w_fp8,
            &dw_shape,
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let gs_storage: Box<dyn BackendStorage> =
            CoreTensorOps::from_cpu(self, expert_gate_scale, &gs_shape, DType::F32)?;
        let us_storage: Box<dyn BackendStorage> =
            CoreTensorOps::from_cpu(self, expert_up_scale, &us_shape, DType::F32)?;
        let ds_storage: Box<dyn BackendStorage> =
            CoreTensorOps::from_cpu(self, expert_down_scale, &ds_shape, DType::F32)?;
        let as_storage: Box<dyn BackendStorage> =
            CoreTensorOps::from_cpu(self, a_scale, &as_shape, DType::F32)?;
        let out_storage: Box<dyn BackendStorage> =
            MemoryOps::alloc_storage(self, &out_shape, DType::F32)?;

        let act_s = as_rocm(act_storage.as_ref())?;
        let gw_s = as_rocm(gw_storage.as_ref())?;
        let uw_s = as_rocm(uw_storage.as_ref())?;
        let dw_s = as_rocm(dw_storage.as_ref())?;
        let gs_s = as_rocm(gs_storage.as_ref())?;
        let us_s = as_rocm(us_storage.as_ref())?;
        let ds_s = as_rocm(ds_storage.as_ref())?;
        let as_s = as_rocm(as_storage.as_ref())?;
        let out_s = as_rocm(out_storage.as_ref())?;

        let gw_ptr = dev_ptr(gw_s)?;
        let uw_ptr = dev_ptr(uw_s)?;
        let dw_ptr = dev_ptr(dw_s)?;
        let gs_ptr = dev_ptr(gs_s)?;
        let us_ptr = dev_ptr(us_s)?;
        let ds_ptr = dev_ptr(ds_s)?;
        let as_ptr = dev_ptr(as_s)?;

        self.launch_charon_grouped_dispatch_fp8(
            act_s,
            gw_ptr,
            uw_ptr,
            dw_ptr,
            gs_ptr,
            us_ptr,
            ds_ptr,
            as_ptr,
            &sorted,
            out_s,
            hidden,
            inter,
            num_experts,
            routed_scaling_factor,
        )?;
        self.synchronize();
        out_storage.to_cpu_vec_f32()
    }
}

impl RocmDevice {
    /// Compute dynamic Expert Parallel Load Balancing (EPLB) greedy LPT placement.
    pub fn eplb_balance_experts(
        &self,
        expert_frequencies: &[f32],
        num_ranks: usize,
        replication_slots: usize,
    ) -> crate::device::eplb::EplbPackingPlan {
        crate::device::eplb::EplbRouter::balance_experts(
            expert_frequencies,
            num_ranks,
            replication_slots,
        )
    }

    /// Plan continuous batch reordering into [Decode : Extend : Prefill] partitions.
    pub fn reorder_batch(
        &self,
        sequences: &[crate::device::batch_orchestrator::SequenceMeta],
    ) -> crate::device::batch_orchestrator::ReorderedBatch {
        crate::device::batch_orchestrator::BatchReorderer::plan(sequences)
    }

    /// Launch one bounded Scythe persistent worker.
    /// The worker is intentionally launched as a single 128-thread block: the callable Charon device function.
    pub fn launch_scythe_persistent_dispatch(
        &self,
        slots: &dyn BackendStorage,
        capacity: u32,
        tail: &dyn BackendStorage,
        head: &dyn BackendStorage,
        stop: &dyn BackendStorage,
        max_tasks: u32,
        resident: u32,
    ) -> Result<Box<dyn ComputeHandle>> {
        let mut slots_ptr = dev_ptr(as_rocm(slots)?)?;
        let mut tail_ptr = dev_ptr(as_rocm(tail)?)?;
        let mut head_ptr = dev_ptr(as_rocm(head)?)?;
        let mut stop_ptr = dev_ptr(as_rocm(stop)?)?;
        let mut cap = capacity;
        let mut limit = max_tasks;
        let mut res = resident;
        if std::env::var_os("GRIM_RING_DIAG").is_some() {
            eprintln!(
                "[launch-diag] persistent wave: cap={cap} max_tasks={max_tasks} resident={res} stream_nonnull=false"
            );
        }
        self.launch_compute_kernel(
            "grim_scythe_persistent_dispatch",
            crate::HipDim3::new(1, 1, 1),
            crate::HipDim3::new(128, 1, 1),
            &mut [
                arg(&mut slots_ptr),
                arg(&mut cap),
                arg(&mut tail_ptr),
                arg(&mut head_ptr),
                arg(&mut stop_ptr),
                arg(&mut limit),
                arg(&mut res),
            ],
        )?;
        Ok(Box::new(RocmHandle::new(Some(self.active_stream()))))
    }

    /// WI-SB6: launch the persistent worker on an EXPLICIT non-blocking stream.
    /// The batch-mode wrapper above uses the device active stream; resident mode must own its stream.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_scythe_persistent_dispatch_on(
        &self,
        slots: &dyn BackendStorage,
        capacity: u32,
        tail: &dyn BackendStorage,
        head: &dyn BackendStorage,
        stop: &dyn BackendStorage,
        max_tasks: u32,
        resident: u32,
        stream: *mut c_void,
    ) -> Result<Box<dyn ComputeHandle>> {
        let mut slots_ptr = dev_ptr(as_rocm(slots)?)?;
        let mut tail_ptr = dev_ptr(as_rocm(tail)?)?;
        let mut head_ptr = dev_ptr(as_rocm(head)?)?;
        let mut stop_ptr = dev_ptr(as_rocm(stop)?)?;
        let mut cap = capacity;
        let mut limit = max_tasks;
        let mut res = resident;
        if std::env::var_os("GRIM_RING_DIAG").is_some() {
            eprintln!(
                "[launch-diag] persistent wave: cap={cap} max_tasks={max_tasks} resident={res} stream_nonnull={}",
                !stream.is_null()
            );
        }
        self.launch_compute_kernel(
            "grim_scythe_persistent_dispatch",
            crate::HipDim3::new(1, 1, 1),
            crate::HipDim3::new(128, 1, 1),
            &mut [
                arg(&mut slots_ptr),
                arg(&mut cap),
                arg(&mut tail_ptr),
                arg(&mut head_ptr),
                arg(&mut stop_ptr),
                arg(&mut limit),
                arg(&mut res),
            ],
        )?;
        Ok(Box::new(RocmHandle::new(Some(stream))))
    }
}
