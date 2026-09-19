//! MoE routing and Charon kernel dispatchers for `RocmDevice`.

//! Charon fused/grouped backward roundtrips and step-time benchmarking helpers.

// (import removed: unused after split)

use grim_tensor::backend::{ BackendStorage };
use grim_tensor::dtype::{ DType };
use grim_tensor::error::{ Result };
use grim_tensor::{CoreTensorOps, MemoryOps, Shape};

use crate::device::roc_device::{ RocmDevice };
#[cfg(feature = "training")]
use crate::device::roc_device::CharonBackwardResult;
// (import removed: unused after split)
use crate::{ as_rocm, dev_ptr };
impl RocmDevice {
    /// Device launcher for the FP32 Charon MoE backward kernel (`grim_moe_fused_grouped_backward`).
    /// Mirrors `launch_charon_grouped_dispatch`: validates inputs, zero-initialises the four atomicAdd output buffers, plans the grouped grid/block from.
    #[cfg(feature = "training")]
    pub fn launch_charon_grouped_backward(
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
        stash_hg: Option<&RocmStorage>,
        stash_hu: Option<&RocmStorage>,
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
            hipMemsetAsync(
                dgw_ptr as *mut c_void,
                0,
                d_gate_w.bytes(),
                self.active_stream(),
            )
        })?;
        check_hip("charon_backward hipMemset(d_up_w, 0)", unsafe {
            hipMemsetAsync(
                duw_ptr as *mut c_void,
                0,
                d_up_w.bytes(),
                self.active_stream(),
            )
        })?;
        check_hip("charon_backward hipMemset(d_down_w, 0)", unsafe {
            hipMemsetAsync(
                ddw_ptr as *mut c_void,
                0,
                d_down_w.bytes(),
                self.active_stream(),
            )
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

        // Kernel arg order matches grim_moe_fused_grouped_backward signature:
        // activations, gate_w, up_w, down_w, d_y,
        // d_gate_w, d_up_w, d_down_w, d_x,
        // sorted_token_ids, sorted_expert_ids, sorted_weights,
        // stash_hg, stash_hu,
        // hidden, inter, num_tokens, block_size, routed_scaling_factor
        let mut a = a_ptr as *mut c_void;
        let mut gw = expert_gate_w_ptr as *mut c_void;
        let mut uw = expert_up_w_ptr as *mut c_void;
        let mut dw = expert_down_w_ptr as *mut c_void;
        let mut dy = dy_ptr as *mut c_void;
        let mut dgw = dgw_ptr as *mut c_void;
        let mut duw = duw_ptr as *mut c_void;
        let mut ddw = ddw_ptr as *mut c_void;
        let mut dx = dx_ptr as *mut c_void;
        let mut shg = stash_hg.and_then(|s| s.device_ptr).unwrap_or(0) as *mut c_void;
        let mut shu = stash_hu.and_then(|s| s.device_ptr).unwrap_or(0) as *mut c_void;
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
                arg(&mut shg),
                arg(&mut shu),
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

    /// Host-to-host roundtrip for the Charon MoE backward kernel with optional stashing.
    #[cfg(feature = "training")]
    pub fn charon_grouped_backward_roundtrip_stashed(
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
        use_stash: bool,
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

        let (shg_storage, shu_storage) = if use_stash {
            // SPEED-ROC-14: the stash is written by the FORWARD GPU kernel, not
            // computed on the host. Allocate empty device buffers, run the FP32
            // grouped forward (which fills them via the stash_hg/stash_hu params),
            // then let the backward kernel read them.
            let stash_shape = Shape::new(vec![sorted.num_tokens_post_padded, inter]);
            let hg_storage: Box<dyn BackendStorage> =
                MemoryOps::alloc_storage(self, &stash_shape, DType::F32)?;
            let hu_storage: Box<dyn BackendStorage> =
                MemoryOps::alloc_storage(self, &stash_shape, DType::F32)?;
            let hg_s = as_rocm(hg_storage.as_ref())?;
            let hu_s = as_rocm(hu_storage.as_ref())?;

            // Throwaway forward output buffer (roundtrip only cares about grads).
            let fwd_out_shape = Shape::new(vec![batch, hidden]);
            let fwd_out_storage: Box<dyn BackendStorage> =
                MemoryOps::alloc_storage(self, &fwd_out_shape, DType::F32)?;
            let fwd_out_s = as_rocm(fwd_out_storage.as_ref())?;

            self.launch_charon_grouped_dispatch_entry(
                act_s,
                gw_ptr,
                uw_ptr,
                dw_ptr,
                &sorted,
                fwd_out_s,
                hidden,
                inter,
                routed_scaling_factor,
                num_experts,
                "grim_moe_fused_grouped",
                Some(hg_s),
                Some(hu_s),
            )?;
            self.synchronize();
            (Some(hg_storage), Some(hu_storage))
        } else {
            (None, None)
        };

        let shg_s = shg_storage.as_ref().map(|s| as_rocm(s.as_ref())).transpose()?;
        let shu_s = shu_storage.as_ref().map(|s| as_rocm(s.as_ref())).transpose()?;

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
            shg_s,
            shu_s,
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

    /// Host-to-host roundtrip for the Charon MoE backward kernel.
    /// Uploads all inputs (activations, expert weights, d_y) and the sorted routing arrays to the device,.
    #[cfg(feature = "training")]
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
        self.charon_grouped_backward_roundtrip_stashed(
            activations,
            expert_gate_w,
            expert_up_w,
            expert_down_w,
            d_y,
            assignment,
            batch,
            hidden,
            inter,
            routed_scaling_factor,
            true,
        )
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

    /// Benchmarks MoE backward step time (pure device kernel execution, excluding CPU-GPU uploads)
    /// comparing recompute (use_stash = false) vs stashed (use_stash = true).
    #[cfg(feature = "training")]
    pub fn benchmark_charon_backward_step_time(
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
        iters: usize,
    ) -> Result<(f64, f64)> {
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

        let dgw_storage: Box<dyn BackendStorage> =
            MemoryOps::alloc_storage(self, &dgw_shape, DType::F32)?;
        let duw_storage: Box<dyn BackendStorage> =
            MemoryOps::alloc_storage(self, &duw_shape, DType::F32)?;
        let ddw_storage: Box<dyn BackendStorage> =
            MemoryOps::alloc_storage(self, &ddw_shape, DType::F32)?;
        let dx_storage: Box<dyn BackendStorage> =
            MemoryOps::alloc_storage(self, &dx_shape, DType::F32)?;

        let stash_shape = Shape::new(vec![sorted.num_tokens_post_padded, inter]);
        let shg_storage: Box<dyn BackendStorage> =
            MemoryOps::alloc_storage(self, &stash_shape, DType::F32)?;
        let shu_storage: Box<dyn BackendStorage> =
            MemoryOps::alloc_storage(self, &stash_shape, DType::F32)?;

        let act_s = as_rocm(act_storage.as_ref())?;
        let gw_s = as_rocm(gw_storage.as_ref())?;
        let uw_s = as_rocm(uw_storage.as_ref())?;
        let dw_s = as_rocm(dw_storage.as_ref())?;
        let dy_s = as_rocm(dy_storage.as_ref())?;
        let dgw_s = as_rocm(dgw_storage.as_ref())?;
        let duw_s = as_rocm(duw_storage.as_ref())?;
        let ddw_s = as_rocm(ddw_storage.as_ref())?;
        let dx_s = as_rocm(dx_storage.as_ref())?;
        let shg_s = as_rocm(shg_storage.as_ref())?;
        let shu_s = as_rocm(shu_storage.as_ref())?;

        let gw_ptr = dev_ptr(gw_s)?;
        let uw_ptr = dev_ptr(uw_s)?;
        let dw_ptr = dev_ptr(dw_s)?;

        // Warmup
        let _ = self.launch_charon_grouped_backward(
            act_s, gw_ptr, uw_ptr, dw_ptr, dy_s, dgw_s, duw_s, ddw_s, dx_s,
            &sorted, None, None, hidden, inter, routed_scaling_factor, num_experts,
        )?;
        self.synchronize();

        // 1. Recompute path
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = self.launch_charon_grouped_backward(
                act_s, gw_ptr, uw_ptr, dw_ptr, dy_s, dgw_s, duw_s, ddw_s, dx_s,
                &sorted, None, None, hidden, inter, routed_scaling_factor, num_experts,
            )?;
        }
        self.synchronize();
        let recompute_dur = t0.elapsed();
        let recompute_us = (recompute_dur.as_micros() as f64) / (iters as f64);

        // 2. Stashed path — fill the stash with the FORWARD GPU kernel first
        // (SPEED-ROC-14), exactly as the training path does, then time only the
        // backward kernel reading those stashed activations.
        let fwd_out_shape = Shape::new(vec![batch, hidden]);
        let fwd_out_storage: Box<dyn BackendStorage> =
            MemoryOps::alloc_storage(self, &fwd_out_shape, DType::F32)?;
        let fwd_out_s = as_rocm(fwd_out_storage.as_ref())?;
        self.launch_charon_grouped_dispatch_entry(
            act_s, gw_ptr, uw_ptr, dw_ptr, &sorted, fwd_out_s,
            hidden, inter, routed_scaling_factor, num_experts,
            "grim_moe_fused_grouped", Some(shg_s), Some(shu_s),
        )?;
        self.synchronize();
        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = self.launch_charon_grouped_backward(
                act_s, gw_ptr, uw_ptr, dw_ptr, dy_s, dgw_s, duw_s, ddw_s, dx_s,
                &sorted, Some(shg_s), Some(shu_s), hidden, inter, routed_scaling_factor, num_experts,
            )?;
        }
        self.synchronize();
        let stashed_dur = t1.elapsed();
        let stashed_us = (stashed_dur.as_micros() as f64) / (iters as f64);

        Ok((recompute_us, stashed_us))
    }
}
