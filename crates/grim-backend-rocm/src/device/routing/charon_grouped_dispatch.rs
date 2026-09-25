//! MoE routing and Charon kernel dispatchers for `RocmDevice`.

//! Token-sorted grouped Charon dispatch roundtrips and per-format launchers.

use std::ffi::c_void;

use grim_tensor::backend::BackendStorage;
use grim_tensor::dtype::{ArithType, DType, Storage as DTypeStorage};
use grim_tensor::error::{Error, Result};
use grim_tensor::{CoreTensorOps, MemoryOps, Shape};

#[cfg(feature = "training")]
use crate::device::roc_device::CharonBackwardResult;
use crate::device::roc_device::{CharonForwardStash, RocmDevice};
use crate::memory::storage::RocmStorage;
use crate::{
    HipDim3, arg, as_rocm, check_hip, dev_ptr, hipFreeAsync, hipMemsetAsync, upload_device_buffer,
};
impl RocmDevice {
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
    ) -> Result<CharonForwardStash> {
        // SPEED-ROC-14: allocate the stash buffers the FP32 forward kernel
        // writes (h_gate/h_up per sorted slot) and the backward kernel reads.
        let stash_shape = Shape::new(vec![sorted.num_tokens_post_padded, inter]);
        let hg_storage: Box<dyn BackendStorage> =
            MemoryOps::alloc_storage(self, &stash_shape, DType::F32)?;
        let hu_storage: Box<dyn BackendStorage> =
            MemoryOps::alloc_storage(self, &stash_shape, DType::F32)?;
        let hg_s = as_rocm(hg_storage.as_ref())?;
        let hu_s = as_rocm(hu_storage.as_ref())?;

        let stream = self.launch_charon_grouped_dispatch_entry(
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
            Some(hg_s),
            Some(hu_s),
        )?;
        // The stash must outlive the stream until the backward launch; stash the
        // stream handle inside the returned handle for a single synchronize().
        unsafe { crate::hipStreamSynchronize(stream) };
        Ok(CharonForwardStash {
            hg: hg_storage,
            hu: hu_storage,
        })
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
        stash_hg: Option<&RocmStorage>,
        stash_hu: Option<&RocmStorage>,
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
            expert_up_w_ptr == 0,
        )?;

        // Output is accumulated via atomicAdd in-kernel; zero first.
        check_hip("charon_grouped hipMemset(output, 0)", unsafe {
            hipMemsetAsync(
                out_ptr as *mut c_void,
                0,
                out_storage.bytes(),
                self.active_stream(),
            )
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

        // SPEED-ROC-14: the FP32 scalar grouped kernel takes two trailing
        // stash pointers (its global signature was extended). The WMMA/quant
        // entries have no such params, so append only for the FP32 entry.
        let mut shg = stash_hg.and_then(|s| s.device_ptr).unwrap_or(0) as *mut c_void;
        let mut shu = stash_hu.and_then(|s| s.device_ptr).unwrap_or(0) as *mut c_void;
        let stream = if entry == "grim_moe_fused_grouped" {
            self.launch_compute_kernel(
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
                    arg(&mut shg),
                    arg(&mut shu),
                ],
            )?
        } else {
            self.launch_compute_kernel(
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
            )?
        };

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
            false,
        )?;

        check_hip("charon_grouped_fp8 hipMemset(output, 0)", unsafe {
            hipMemsetAsync(
                out_ptr as *mut c_void,
                0,
                out_storage.bytes(),
                self.active_stream(),
            )
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
            false,
        )?;

        check_hip("charon_grouped_mxfp4 hipMemset(output, 0)", unsafe {
            hipMemsetAsync(
                out_ptr as *mut c_void,
                0,
                out_storage.bytes(),
                self.active_stream(),
            )
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
            false,
        )?;

        check_hip("charon_grouped_mxfp8 hipMemset(output, 0)", unsafe {
            hipMemsetAsync(
                out_ptr as *mut c_void,
                0,
                out_storage.bytes(),
                self.active_stream(),
            )
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
            false,
        )?;

        check_hip("charon_grouped_q80 hipMemset(output, 0)", unsafe {
            hipMemsetAsync(
                out_ptr as *mut c_void,
                0,
                out_storage.bytes(),
                self.active_stream(),
            )
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

    /// Generic launcher for the SPEED-DOT (dot4/sudot4) grouped MoE kernels.
    /// All three variants (Q8_0, W8A8-int8, Q4_K) share the scalar grouped
    /// kernel's argument shape — only the JIT entry differs. Callers resolve
    /// the entry via `kernels::charon::dot4_entry_for` (arch + training gate).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_charon_grouped_dispatch_dot4(
        &self,
        entry: &str,
        act_storage: &RocmStorage,
        egate_w_ptr: u64,
        eup_w_ptr: u64,
        edown_w_ptr: u64,
        a_scale_ptr: u64,
        sorted: &crate::kernels::charon::SortedRouting,
        out_storage: &RocmStorage,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
    ) -> Result<*mut c_void> {
        // The dot4 kernels iterate 32-element activation blocks and index
        // Q4_K sub-blocks positionally; misalignment breaks both.
        if hidden % 32 != 0 || inter % 32 != 0 {
            return Err(Error::Backend(format!(
                "charon_grouped_dot4: hidden/inter must be multiples of 32 (hidden={hidden}, inter={inter})"
            )));
        }
        // P1-3: raw HIP ops below bind to the calling thread's current
        // device — pin to the owning ordinal (see matmul_op fix, 2026-08-23e).
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        use crate::kernels::charon::plan_grouped_dispatch;

        check_hip("charon_grouped_dot4 hipMemset(output, 0)", unsafe {
            hipMemsetAsync(
                out_storage.device_ptr.ok_or_else(|| {
                    Error::Backend("charon_grouped_dot4: out has no device ptr".into())
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
            Error::Backend("charon_grouped_dot4: activations has no device ptr".into())
        })? as *mut c_void;
        let mut gw = egate_w_ptr;
        let mut uw = eup_w_ptr;
        let mut dw = edown_w_ptr;
        let mut ascale = a_scale_ptr;
        let mut optr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("charon_grouped_dot4: out has no device ptr".into()))?
            as *mut c_void;
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

        let _stash = self.launch_charon_grouped_dispatch(
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
            None,
            None,
        )?;
        self.synchronize();
        out_storage.to_cpu_vec_f32()
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

    /// SPEED-DOT roundtrip: token-sorts via `moe_align_block_size` and
    /// launches one of the dot4 grouped kernels (`entry`, resolved by
    /// `kernels::charon::dot4_entry_for`). Weight bytes are the native packed
    /// layout of the chosen quant (Q8_0: 34B/32, Q4_K: 144B/256,
    /// W8A8-int8: prefix + codes + row scales). Used by the dot4-vs-scalar
    /// parity test.
    #[allow(clippy::too_many_arguments)]
    pub fn charon_grouped_dispatch_roundtrip_dot4(
        &self,
        entry: &str,
        activations: &[f32],
        expert_gate_w_q: &[u8],
        expert_up_w_q: &[u8],
        expert_down_w_q: &[u8],
        a_scale: &[f32],
        assignment: &crate::kernels::charon::RoutingAssignment,
        batch: usize,
        hidden: usize,
        inter: usize,
        num_experts: usize,
        routed_scaling_factor: f32,
    ) -> Result<Vec<f32>> {
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
        let gw_ptr = dev_ptr(as_rocm(gw_storage.as_ref())?)?;
        let uw_ptr = dev_ptr(as_rocm(uw_storage.as_ref())?)?;
        let dw_ptr = dev_ptr(as_rocm(dw_storage.as_ref())?)?;
        let as_ptr = dev_ptr(as_rocm(as_storage.as_ref())?)?;
        let out_s = as_rocm(out_storage.as_ref())?;

        self.launch_charon_grouped_dispatch_dot4(
            entry,
            act_s,
            gw_ptr,
            uw_ptr,
            dw_ptr,
            as_ptr,
            &sorted,
            out_s,
            hidden,
            inter,
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
