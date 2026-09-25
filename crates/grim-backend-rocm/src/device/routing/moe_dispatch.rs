//! MoE routing and Charon kernel dispatchers for `RocmDevice`.

//! Fused MoE resident-dispatch launchers (routing, w8a8, awq, mxfp4 variants).

use std::ffi::c_void;

use grim_tensor::backend::BackendStorage;
use grim_tensor::dtype::DType;
use grim_tensor::error::{Error, Result};
use grim_tensor::{CoreTensorOps, Shape};

#[cfg(feature = "training")]
use crate::device::roc_device::CharonBackwardResult;
use crate::device::roc_device::RocmDevice;
use crate::memory::storage::RocmStorage;
use crate::{HipDim3, RocmHandle, arg, check_hip, dtype_f32, hipMemsetAsync};
impl RocmDevice {
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

    /// Fused grouped MoE dispatch against weight buffers that are already resident on the device.
    ///
    /// Stacks / takes contiguous `[num_experts, inter * hidden]` gate/up and `[num_experts, hidden * inter]` down weights,
    /// token-sorts the routing assignment via `moe_align_block_size`, and launches `grim_moe_fused_grouped`.
    pub fn moe_fused_grouped_dispatch_resident(
        &self,
        activations: &RocmStorage,
        gate_buf: &dyn BackendStorage,
        up_buf: &dyn BackendStorage,
        down_buf: &dyn BackendStorage,
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

        let out_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;

        let stream = self.launch_charon_grouped_dispatch_entry(
            activations,
            gate_r.device_ptr_checked()?,
            up_r.device_ptr_checked()?,
            down_r.device_ptr_checked()?,
            sorted,
            &out_storage,
            hidden,
            inter,
            routed_scaling_factor,
            num_experts,
            "grim_moe_fused_grouped",
            None,
            None,
        )?;
        Ok((out_storage, RocmHandle::new(Some(stream))))
    }

    /// Fused grouped MoE dispatch for GELU-activation experts (e.g. GLM-5.2).
    /// Calls `grim_moe_fused_grouped_gelu` with the resident expert weights and sorted routing.
    pub fn moe_fused_grouped_dispatch_gelu_resident(
        &self,
        activations: &RocmStorage,
        gate_buf: &dyn BackendStorage,
        down_buf: &dyn BackendStorage,
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
        let down_r = down_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("down_buf downcast failed".into()))?;

        let out_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;

        let stream = self.launch_charon_grouped_dispatch_entry(
            activations,
            gate_r.device_ptr_checked()?,
            0,
            down_r.device_ptr_checked()?,
            sorted,
            &out_storage,
            hidden,
            inter,
            routed_scaling_factor,
            num_experts,
            "grim_moe_fused_grouped_gelu",
            None,
            None,
        )?;
        Ok((out_storage, RocmHandle::new(Some(stream))))
    }

    /// Device-side MoE routing (D2D): computes per-token top-k expert selection +
    /// softmax-normalized combine weights entirely on-device via `grim_moe_route_topk`.
    ///
    /// `logits` is the gate projection output `[seq_len, num_experts]` (device-resident).
    /// Writes sortless (token, expert, weight) triples into the three caller-owned
    /// device buffers, sized `seq_len * top_k`. No host round-trip: the gate logits
    /// never leave the device and the routing table is never re-uploaded H2D.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_route_topk_on_device(
        &self,
        logits: &RocmStorage,
        bias: Option<&RocmStorage>,
        out_tokens: &RocmStorage,
        out_experts: &RocmStorage,
        out_weights: &RocmStorage,
        seq_len: usize,
        num_experts: usize,
        top_k: usize,
        route_mode: i32,
    ) -> Result<*mut c_void> {
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let l_ptr = logits
            .device_ptr
            .ok_or_else(|| Error::Backend("moe_route_topk: logits has no device ptr".into()))?;
        let bias_ptr = bias.and_then(|b| b.device_ptr).unwrap_or(0);
        let ot_ptr = out_tokens
            .device_ptr
            .ok_or_else(|| Error::Backend("moe_route_topk: out_tokens has no device ptr".into()))?;
        let oe_ptr = out_experts.device_ptr.ok_or_else(|| {
            Error::Backend("moe_route_topk: out_experts has no device ptr".into())
        })?;
        let ow_ptr = out_weights.device_ptr.ok_or_else(|| {
            Error::Backend("moe_route_topk: out_weights has no device ptr".into())
        })?;

        let mut l = l_ptr as *mut c_void;
        let mut b = bias_ptr as *mut c_void;
        let mut ot = ot_ptr as *mut c_void;
        let mut oe = oe_ptr as *mut c_void;
        let mut ow = ow_ptr as *mut c_void;
        let mut seq_i = seq_len as i32;
        let mut nexp_i = num_experts as i32;
        let mut topk_i = top_k as i32;
        let mut mode_i = route_mode;

        self.launch_compute_kernel(
            "grim_moe_route_topk",
            HipDim3::new(seq_len as u32, 1, 1),
            HipDim3::new(256, 1, 1),
            &mut [
                arg(&mut l),
                arg(&mut b),
                arg(&mut ot),
                arg(&mut oe),
                arg(&mut ow),
                arg(&mut seq_i),
                arg(&mut nexp_i),
                arg(&mut topk_i),
                arg(&mut mode_i),
            ],
        )
    }

    /// Sortless Charon fused dispatch fed by device-resident routing buffers.
    /// Unlike [`Self::moe_fused_dispatch`]/[`Self::moe_fused_dispatch_resident`], no
    /// host `upload_device_buffer` of the routing table is performed — `routing_tokens`,
    /// `routing_experts`, `routing_weights` are already on-device (produced by
    /// [`Self::moe_route_topk_on_device`]), so the dispatch is fully D2D.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_fused_dispatch_resident_routing(
        &self,
        activations: &RocmStorage,
        gate_buf: &dyn BackendStorage,
        up_buf: &dyn BackendStorage,
        down_buf: &dyn BackendStorage,
        routing_tokens: &RocmStorage,
        routing_experts: &RocmStorage,
        routing_weights: &RocmStorage,
        num_pairs: usize,
        out_shape: &Shape,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
    ) -> Result<(RocmStorage, RocmHandle)> {
        let out_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let stream = self.moe_fused_dispatch_resident_routing_into(
            activations,
            gate_buf,
            up_buf,
            down_buf,
            routing_tokens,
            routing_experts,
            routing_weights,
            num_pairs,
            &out_storage,
            hidden,
            inter,
            routed_scaling_factor,
        )?;
        Ok((out_storage, RocmHandle::new(Some(stream))))
    }

    /// M2 (PLAN-kernel-fusion): same launch as
    /// [`Self::moe_fused_dispatch_resident_routing`] but writing into
    /// CALLER-PROVIDED `out` — no allocation inside. Graph-capture-safe:
    /// `out` is a stable pool address. Returns the stream used.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_fused_dispatch_resident_routing_into(
        &self,
        activations: &RocmStorage,
        gate_buf: &dyn BackendStorage,
        up_buf: &dyn BackendStorage,
        down_buf: &dyn BackendStorage,
        routing_tokens: &RocmStorage,
        routing_experts: &RocmStorage,
        routing_weights: &RocmStorage,
        num_pairs: usize,
        out_storage: &RocmStorage,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
    ) -> Result<*mut c_void> {
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

        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let a_ptr = activations.device_ptr.ok_or_else(|| {
            Error::Backend("resident_routing: activations has no device ptr".into())
        })?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("resident_routing: out has no device ptr".into()))?;

        // Zero output (atomicAdd accumulation).
        check_hip("resident_routing hipMemset(output, 0)", unsafe {
            hipMemsetAsync(
                out_ptr as *mut c_void,
                0,
                out_storage.bytes(),
                self.active_stream(),
            )
        })?;

        let wave = self.wavefront_size() as u32;
        let block_x = crate::kernels::charon::choose_block_dim(num_pairs, wave);
        let grid_x = if num_pairs == 0 {
            0
        } else {
            (num_pairs as u32).div_ceil(block_x)
        };
        if grid_x == 0 {
            return Ok(self.active_stream());
        }

        let mut a = a_ptr as *mut c_void;
        let mut gw = gate_r.device_ptr_checked()? as *mut c_void;
        let mut uw = up_r.device_ptr_checked()? as *mut c_void;
        let mut dw = down_r.device_ptr_checked()? as *mut c_void;
        let mut tok_ptr = routing_tokens.device_ptr_checked()? as *mut c_void;
        let mut exp_ptr = routing_experts.device_ptr_checked()? as *mut c_void;
        let mut w_ptr = routing_weights.device_ptr_checked()? as *mut c_void;
        let mut optr = out_ptr as *mut c_void;
        let mut hidden_i = hidden as i32;
        let mut inter_i = inter as i32;
        let mut num_pairs_i = num_pairs as i32;
        let mut rsf = routed_scaling_factor;

        let stream = self.launch_compute_kernel(
            "grim_moe_fused_dispatch",
            HipDim3::new(grid_x, 1, 1),
            HipDim3::new(block_x, 1, 1),
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

        Ok(stream)
    }

    /// WI-gpu-native-moe Phase 2: sortless W8A8-int8 fused dispatch fed by
    /// device-resident routing buffers (D2D). Same contract as
    /// [`Self::moe_fused_dispatch_resident_routing`], but the expert stacks
    /// are packed int8 blobs ([u64 prefix | codes | per-row f32 scales],
    /// concatenated per expert) consumed by
    /// `grim_moe_fused_dispatch_w8a8_int8`, plus a device-resident
    /// per-token activation scale (`a_scale`, `[batch]`).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_fused_dispatch_resident_routing_w8a8_int8(
        &self,
        activations: &RocmStorage,
        gate_buf: &dyn BackendStorage,
        up_buf: &dyn BackendStorage,
        down_buf: &dyn BackendStorage,
        a_scale_buf: &dyn BackendStorage,
        routing_tokens: &RocmStorage,
        routing_experts: &RocmStorage,
        routing_weights: &RocmStorage,
        num_pairs: usize,
        out_shape: &Shape,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
    ) -> Result<(RocmStorage, RocmHandle)> {
        let out_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let stream = self.moe_fused_dispatch_resident_routing_w8a8_int8_into(
            activations,
            gate_buf,
            up_buf,
            down_buf,
            a_scale_buf,
            routing_tokens,
            routing_experts,
            routing_weights,
            num_pairs,
            &out_storage,
            hidden,
            inter,
            routed_scaling_factor,
        )?;
        Ok((out_storage, RocmHandle::new(Some(stream))))
    }

    /// Capture-safe variant of
    /// [`Self::moe_fused_dispatch_resident_routing_w8a8_int8`]: writes into
    /// CALLER-PROVIDED `out` (stable pool address, no allocation inside).
    /// Returns the stream used.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_fused_dispatch_resident_routing_w8a8_int8_into(
        &self,
        activations: &RocmStorage,
        gate_buf: &dyn BackendStorage,
        up_buf: &dyn BackendStorage,
        down_buf: &dyn BackendStorage,
        a_scale_buf: &dyn BackendStorage,
        routing_tokens: &RocmStorage,
        routing_experts: &RocmStorage,
        routing_weights: &RocmStorage,
        num_pairs: usize,
        out_storage: &RocmStorage,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
    ) -> Result<*mut c_void> {
        let gate_r = gate_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| {
                Error::Backend("w8a8 resident_routing: gate_buf downcast failed".into())
            })?;
        let up_r = up_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| {
                Error::Backend("w8a8 resident_routing: up_buf downcast failed".into())
            })?;
        let down_r = down_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| {
                Error::Backend("w8a8 resident_routing: down_buf downcast failed".into())
            })?;
        let ascale_r = a_scale_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| {
                Error::Backend("w8a8 resident_routing: a_scale_buf downcast failed".into())
            })?;

        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let a_ptr = activations.device_ptr.ok_or_else(|| {
            Error::Backend("w8a8 resident_routing: activations has no device ptr".into())
        })?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("w8a8 resident_routing: out has no device ptr".into()))?;

        // Zero output (atomicAdd accumulation).
        check_hip("w8a8 resident_routing hipMemset(output, 0)", unsafe {
            hipMemsetAsync(
                out_ptr as *mut c_void,
                0,
                out_storage.bytes(),
                self.active_stream(),
            )
        })?;

        let wave = self.wavefront_size() as u32;
        let block_x = crate::kernels::charon::choose_block_dim(num_pairs, wave);
        let grid_x = if num_pairs == 0 {
            0
        } else {
            (num_pairs as u32).div_ceil(block_x)
        };
        if grid_x == 0 {
            return Ok(self.active_stream());
        }

        let mut a = a_ptr as *mut c_void;
        let mut gw = gate_r.device_ptr_checked()? as *mut c_void;
        let mut uw = up_r.device_ptr_checked()? as *mut c_void;
        let mut dw = down_r.device_ptr_checked()? as *mut c_void;
        let mut asc = ascale_r.device_ptr_checked()? as *mut c_void;
        let mut tok_ptr = routing_tokens.device_ptr_checked()? as *mut c_void;
        let mut exp_ptr = routing_experts.device_ptr_checked()? as *mut c_void;
        let mut w_ptr = routing_weights.device_ptr_checked()? as *mut c_void;
        let mut optr = out_ptr as *mut c_void;
        let mut hidden_i = hidden as i32;
        let mut inter_i = inter as i32;
        let mut num_pairs_i = num_pairs as i32;
        let mut rsf = routed_scaling_factor;

        let stream = self.launch_compute_kernel(
            "grim_moe_fused_dispatch_w8a8_int8",
            HipDim3::new(grid_x, 1, 1),
            HipDim3::new(block_x, 1, 1),
            &mut [
                arg(&mut a),
                arg(&mut gw),
                arg(&mut uw),
                arg(&mut dw),
                arg(&mut asc),
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

        Ok(stream)
    }

    /// WI-gpu-native-moe #2: sortless W8A8-int8 DOT4 dispatch fed by
    /// device-resident routing buffers (D2D). Same contract as
    /// [`Self::moe_fused_dispatch_resident_routing_w8a8_int8`], but the
    /// gate/up contraction runs on V_DOT4. REQUIRES `hidden % 32 == 0`
    /// (per-32 activation blocks); misaligned shapes are refused loudly so
    /// the caller falls back to the scalar sortless kernel — never silent
    /// wrong-codegen (the kernel simply does not exist off the arch guard
    /// on non-RDNA, same discipline).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_fused_dispatch_resident_routing_w8a8_int8_dot4(
        &self,
        activations: &RocmStorage,
        gate_buf: &dyn BackendStorage,
        up_buf: &dyn BackendStorage,
        down_buf: &dyn BackendStorage,
        a_scale_buf: &dyn BackendStorage,
        routing_tokens: &RocmStorage,
        routing_experts: &RocmStorage,
        routing_weights: &RocmStorage,
        num_pairs: usize,
        out_shape: &Shape,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
    ) -> Result<(RocmStorage, RocmHandle)> {
        let out_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let stream = self.moe_fused_dispatch_resident_routing_w8a8_int8_dot4_into(
            activations,
            gate_buf,
            up_buf,
            down_buf,
            a_scale_buf,
            routing_tokens,
            routing_experts,
            routing_weights,
            num_pairs,
            &out_storage,
            hidden,
            inter,
            routed_scaling_factor,
        )?;
        Ok((out_storage, RocmHandle::new(Some(stream))))
    }

    /// Capture-safe variant of
    /// [`Self::moe_fused_dispatch_resident_routing_w8a8_int8_dot4`].
    #[allow(clippy::too_many_arguments)]
    pub fn moe_fused_dispatch_resident_routing_w8a8_int8_dot4_into(
        &self,
        activations: &RocmStorage,
        gate_buf: &dyn BackendStorage,
        up_buf: &dyn BackendStorage,
        down_buf: &dyn BackendStorage,
        a_scale_buf: &dyn BackendStorage,
        routing_tokens: &RocmStorage,
        routing_experts: &RocmStorage,
        routing_weights: &RocmStorage,
        num_pairs: usize,
        out_storage: &RocmStorage,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
    ) -> Result<*mut c_void> {
        if hidden % 32 != 0 {
            return Err(Error::Backend(format!(
                "w8a8_dot4 resident_routing: hidden={hidden} not a multiple of 32"
            )));
        }
        if !crate::kernels::charon::dot4_supported(self.gcn_arch()) {
            return Err(Error::Backend(format!(
                "w8a8_dot4 resident_routing: no dot4 on {}",
                self.gcn_arch()
            )));
        }
        let gate_r = gate_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| {
                Error::Backend("w8a8_dot4 resident_routing: gate_buf downcast failed".into())
            })?;
        let up_r = up_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| {
                Error::Backend("w8a8_dot4 resident_routing: up_buf downcast failed".into())
            })?;
        let down_r = down_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| {
                Error::Backend("w8a8_dot4 resident_routing: down_buf downcast failed".into())
            })?;
        let ascale_r = a_scale_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| {
                Error::Backend("w8a8_dot4 resident_routing: a_scale_buf downcast failed".into())
            })?;

        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let a_ptr = activations.device_ptr.ok_or_else(|| {
            Error::Backend("w8a8_dot4 resident_routing: activations has no device ptr".into())
        })?;
        let out_ptr = out_storage.device_ptr.ok_or_else(|| {
            Error::Backend("w8a8_dot4 resident_routing: out has no device ptr".into())
        })?;

        // Zero output (atomicAdd accumulation).
        check_hip("w8a8_dot4 resident_routing hipMemset(output, 0)", unsafe {
            hipMemsetAsync(
                out_ptr as *mut c_void,
                0,
                out_storage.bytes(),
                self.active_stream(),
            )
        })?;

        let wave = self.wavefront_size() as u32;
        let block_x = crate::kernels::charon::choose_block_dim(num_pairs, wave);
        let grid_x = if num_pairs == 0 {
            0
        } else {
            (num_pairs as u32).div_ceil(block_x)
        };
        if grid_x == 0 {
            return Ok(self.active_stream());
        }

        let mut a = a_ptr as *mut c_void;
        let mut gw = gate_r.device_ptr_checked()? as *mut c_void;
        let mut uw = up_r.device_ptr_checked()? as *mut c_void;
        let mut dw = down_r.device_ptr_checked()? as *mut c_void;
        let mut asc = ascale_r.device_ptr_checked()? as *mut c_void;
        let mut tok_ptr = routing_tokens.device_ptr_checked()? as *mut c_void;
        let mut exp_ptr = routing_experts.device_ptr_checked()? as *mut c_void;
        let mut w_ptr = routing_weights.device_ptr_checked()? as *mut c_void;
        let mut optr = out_ptr as *mut c_void;
        let mut hidden_i = hidden as i32;
        let mut inter_i = inter as i32;
        let mut num_pairs_i = num_pairs as i32;
        let mut rsf = routed_scaling_factor;

        let stream = self.launch_compute_kernel(
            "grim_moe_fused_dispatch_w8a8_int8_dot4",
            HipDim3::new(grid_x, 1, 1),
            HipDim3::new(block_x, 1, 1),
            &mut [
                arg(&mut a),
                arg(&mut gw),
                arg(&mut uw),
                arg(&mut dw),
                arg(&mut asc),
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

        Ok(stream)
    }

    /// WI-gpu-native-moe Phase 2: sortless W8A8-FP8 fused dispatch fed by
    /// device-resident routing buffers (D2D). Contract mirrors
    /// [`Self::moe_fused_dispatch_resident_routing_w8a8_int8`]; per-expert
    /// blobs are `[u64 prefix | fp8-E4M3 codes | ONE f32 scale]`.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_fused_dispatch_resident_routing_w8a8_fp8(
        &self,
        activations: &RocmStorage,
        gate_buf: &dyn BackendStorage,
        up_buf: &dyn BackendStorage,
        down_buf: &dyn BackendStorage,
        a_scale_buf: &dyn BackendStorage,
        routing_tokens: &RocmStorage,
        routing_experts: &RocmStorage,
        routing_weights: &RocmStorage,
        num_pairs: usize,
        out_shape: &Shape,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
    ) -> Result<(RocmStorage, RocmHandle)> {
        let out_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let stream = self.moe_fused_dispatch_resident_routing_w8a8_fp8_into(
            activations,
            gate_buf,
            up_buf,
            down_buf,
            a_scale_buf,
            routing_tokens,
            routing_experts,
            routing_weights,
            num_pairs,
            &out_storage,
            hidden,
            inter,
            routed_scaling_factor,
        )?;
        Ok((out_storage, RocmHandle::new(Some(stream))))
    }

    /// Capture-safe variant of
    /// [`Self::moe_fused_dispatch_resident_routing_w8a8_fp8`].
    #[allow(clippy::too_many_arguments)]
    pub fn moe_fused_dispatch_resident_routing_w8a8_fp8_into(
        &self,
        activations: &RocmStorage,
        gate_buf: &dyn BackendStorage,
        up_buf: &dyn BackendStorage,
        down_buf: &dyn BackendStorage,
        a_scale_buf: &dyn BackendStorage,
        routing_tokens: &RocmStorage,
        routing_experts: &RocmStorage,
        routing_weights: &RocmStorage,
        num_pairs: usize,
        out_storage: &RocmStorage,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
    ) -> Result<*mut c_void> {
        let gate_r = gate_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| {
                Error::Backend("w8a8_fp8 resident_routing: gate_buf downcast failed".into())
            })?;
        let up_r = up_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| {
                Error::Backend("w8a8_fp8 resident_routing: up_buf downcast failed".into())
            })?;
        let down_r = down_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| {
                Error::Backend("w8a8_fp8 resident_routing: down_buf downcast failed".into())
            })?;
        let ascale_r = a_scale_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| {
                Error::Backend("w8a8_fp8 resident_routing: a_scale_buf downcast failed".into())
            })?;

        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let a_ptr = activations.device_ptr.ok_or_else(|| {
            Error::Backend("w8a8_fp8 resident_routing: activations has no device ptr".into())
        })?;
        let out_ptr = out_storage.device_ptr.ok_or_else(|| {
            Error::Backend("w8a8_fp8 resident_routing: out has no device ptr".into())
        })?;

        // Zero output (atomicAdd accumulation).
        check_hip("w8a8_fp8 resident_routing hipMemset(output, 0)", unsafe {
            hipMemsetAsync(
                out_ptr as *mut c_void,
                0,
                out_storage.bytes(),
                self.active_stream(),
            )
        })?;

        let wave = self.wavefront_size() as u32;
        let block_x = crate::kernels::charon::choose_block_dim(num_pairs, wave);
        let grid_x = if num_pairs == 0 {
            0
        } else {
            (num_pairs as u32).div_ceil(block_x)
        };
        if grid_x == 0 {
            return Ok(self.active_stream());
        }

        let mut a = a_ptr as *mut c_void;
        let mut gw = gate_r.device_ptr_checked()? as *mut c_void;
        let mut uw = up_r.device_ptr_checked()? as *mut c_void;
        let mut dw = down_r.device_ptr_checked()? as *mut c_void;
        let mut asc = ascale_r.device_ptr_checked()? as *mut c_void;
        let mut tok_ptr = routing_tokens.device_ptr_checked()? as *mut c_void;
        let mut exp_ptr = routing_experts.device_ptr_checked()? as *mut c_void;
        let mut w_ptr = routing_weights.device_ptr_checked()? as *mut c_void;
        let mut optr = out_ptr as *mut c_void;
        let mut hidden_i = hidden as i32;
        let mut inter_i = inter as i32;
        let mut num_pairs_i = num_pairs as i32;
        let mut rsf = routed_scaling_factor;

        let stream = self.launch_compute_kernel(
            "grim_moe_fused_dispatch_w8a8_fp8",
            HipDim3::new(grid_x, 1, 1),
            HipDim3::new(block_x, 1, 1),
            &mut [
                arg(&mut a),
                arg(&mut gw),
                arg(&mut uw),
                arg(&mut dw),
                arg(&mut asc),
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

        Ok(stream)
    }

    /// WI-gpu-native-moe Phase 2: sortless AWQ fused dispatch fed by
    /// device-resident routing buffers (D2D). Per-expert blobs are
    /// `[u64 qw_len | qweight | u64 qz_len | qzeros | u64 sc_len | f16
    /// scales]`; segment offsets are derived here from
    /// bits/group/hidden/inter (single implementation of the layout math —
    /// must match `awq_split_bank`).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_fused_dispatch_resident_routing_awq(
        &self,
        activations: &RocmStorage,
        gate_buf: &dyn BackendStorage,
        up_buf: &dyn BackendStorage,
        down_buf: &dyn BackendStorage,
        a_scale_buf: &dyn BackendStorage,
        routing_tokens: &RocmStorage,
        routing_experts: &RocmStorage,
        routing_weights: &RocmStorage,
        num_pairs: usize,
        out_shape: &Shape,
        hidden: usize,
        inter: usize,
        bits: u8,
        group_size: usize,
        routed_scaling_factor: f32,
    ) -> Result<(RocmStorage, RocmHandle)> {
        let out_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let stream = self.moe_fused_dispatch_resident_routing_awq_into(
            activations,
            gate_buf,
            up_buf,
            down_buf,
            a_scale_buf,
            routing_tokens,
            routing_experts,
            routing_weights,
            num_pairs,
            &out_storage,
            hidden,
            inter,
            bits,
            group_size,
            routed_scaling_factor,
        )?;
        Ok((out_storage, RocmHandle::new(Some(stream))))
    }

    /// Capture-safe variant of
    /// [`Self::moe_fused_dispatch_resident_routing_awq`].
    #[allow(clippy::too_many_arguments)]
    pub fn moe_fused_dispatch_resident_routing_awq_into(
        &self,
        activations: &RocmStorage,
        gate_buf: &dyn BackendStorage,
        up_buf: &dyn BackendStorage,
        down_buf: &dyn BackendStorage,
        a_scale_buf: &dyn BackendStorage,
        routing_tokens: &RocmStorage,
        routing_experts: &RocmStorage,
        routing_weights: &RocmStorage,
        num_pairs: usize,
        out_storage: &RocmStorage,
        hidden: usize,
        inter: usize,
        bits: u8,
        group_size: usize,
        routed_scaling_factor: f32,
    ) -> Result<*mut c_void> {
        let vpw = match bits {
            4 => 8usize,
            2 => 16usize,
            8 => 1usize,
            b => {
                return Err(Error::Backend(format!(
                    "awq resident_routing: unsupported bit width {b}"
                )));
            }
        };
        if group_size == 0 {
            return Err(Error::Backend(
                "awq resident_routing: group_size is 0".into(),
            ));
        }
        // (qw_off, qz_off, sc_off, stride) for a [rows=out, cols=k] projection.
        let proj_offsets = |out: usize, k: usize| -> (i64, i64, i64, u64) {
            let qw_len = k.div_ceil(vpw) * out * 4;
            let groups = k.div_ceil(group_size);
            let qz_len = groups * out.div_ceil(vpw) * 4;
            let sc_len = groups * out * 2;
            let qw_off = 8i64;
            let qz_off = (8 + qw_len + 8) as i64;
            let sc_off = (8 + qw_len + 8 + qz_len + 8) as i64;
            let stride = (8 + qw_len + 8 + qz_len + 8 + sc_len) as u64;
            (qw_off, qz_off, sc_off, stride)
        };
        let (g_qw, g_qz, g_sc, g_stride) = proj_offsets(inter, hidden);
        let (d_qw, d_qz, d_sc, d_stride) = proj_offsets(hidden, inter);

        let gate_r = gate_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| {
                Error::Backend("awq resident_routing: gate_buf downcast failed".into())
            })?;
        let up_r = up_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("awq resident_routing: up_buf downcast failed".into()))?;
        let down_r = down_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| {
                Error::Backend("awq resident_routing: down_buf downcast failed".into())
            })?;
        let ascale_r = a_scale_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| {
                Error::Backend("awq resident_routing: a_scale_buf downcast failed".into())
            })?;

        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let a_ptr = activations.device_ptr.ok_or_else(|| {
            Error::Backend("awq resident_routing: activations has no device ptr".into())
        })?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("awq resident_routing: out has no device ptr".into()))?;

        // Zero output (atomicAdd accumulation).
        check_hip("awq resident_routing hipMemset(output, 0)", unsafe {
            hipMemsetAsync(
                out_ptr as *mut c_void,
                0,
                out_storage.bytes(),
                self.active_stream(),
            )
        })?;

        let wave = self.wavefront_size() as u32;
        let block_x = crate::kernels::charon::choose_block_dim(num_pairs, wave);
        let grid_x = if num_pairs == 0 {
            0
        } else {
            (num_pairs as u32).div_ceil(block_x)
        };
        if grid_x == 0 {
            return Ok(self.active_stream());
        }

        let mut a = a_ptr as *mut c_void;
        let mut gw = gate_r.device_ptr_checked()? as *mut c_void;
        let mut uw = up_r.device_ptr_checked()? as *mut c_void;
        let mut dw = down_r.device_ptr_checked()? as *mut c_void;
        let mut asc = ascale_r.device_ptr_checked()? as *mut c_void;
        let mut tok_ptr = routing_tokens.device_ptr_checked()? as *mut c_void;
        let mut exp_ptr = routing_experts.device_ptr_checked()? as *mut c_void;
        let mut w_ptr = routing_weights.device_ptr_checked()? as *mut c_void;
        let mut optr = out_ptr as *mut c_void;
        let mut hidden_i = hidden as i32;
        let mut inter_i = inter as i32;
        let mut num_pairs_i = num_pairs as i32;
        let mut bits_i = bits as i32;
        let mut group_i = group_size as i32;
        let mut g_qw_o = g_qw;
        let mut g_qz_o = g_qz;
        let mut g_sc_o = g_sc;
        let mut g_stride = g_stride;
        let mut d_qw_o = d_qw;
        let mut d_qz_o = d_qz;
        let mut d_sc_o = d_sc;
        let mut d_stride = d_stride;
        let mut rsf = routed_scaling_factor;

        let stream = self.launch_compute_kernel(
            "grim_moe_fused_dispatch_awq",
            HipDim3::new(grid_x, 1, 1),
            HipDim3::new(block_x, 1, 1),
            &mut [
                arg(&mut a),
                arg(&mut gw),
                arg(&mut uw),
                arg(&mut dw),
                arg(&mut asc),
                arg(&mut tok_ptr),
                arg(&mut exp_ptr),
                arg(&mut w_ptr),
                arg(&mut optr),
                arg(&mut hidden_i),
                arg(&mut inter_i),
                arg(&mut num_pairs_i),
                arg(&mut bits_i),
                arg(&mut group_i),
                arg(&mut g_qw_o),
                arg(&mut g_qz_o),
                arg(&mut g_sc_o),
                arg(&mut g_stride),
                arg(&mut d_qw_o),
                arg(&mut d_qz_o),
                arg(&mut d_sc_o),
                arg(&mut d_stride),
                arg(&mut rsf),
            ],
        )?;

        Ok(stream)
    }

    /// WI-gpu-native-moe Phase 2: sortless MXFP4 fused dispatch fed by
    /// device-resident routing buffers (D2D). Codes and shared-exponent
    /// stacks are separate buffers, each concatenated per expert with NO
    /// length prefixes (validated at stack-build time). Shapes must satisfy
    /// `(rows*cols) % 32 == 0`; misaligned shapes are refused loudly here
    /// (the kernel cannot address partial 32-groups).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_fused_dispatch_resident_routing_mxfp4(
        &self,
        activations: &RocmStorage,
        gate_codes: &dyn BackendStorage,
        up_codes: &dyn BackendStorage,
        down_codes: &dyn BackendStorage,
        gate_exps: &dyn BackendStorage,
        up_exps: &dyn BackendStorage,
        down_exps: &dyn BackendStorage,
        a_scale_buf: &dyn BackendStorage,
        routing_tokens: &RocmStorage,
        routing_experts: &RocmStorage,
        routing_weights: &RocmStorage,
        num_pairs: usize,
        out_shape: &Shape,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
    ) -> Result<(RocmStorage, RocmHandle)> {
        let out_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let stream = self.moe_fused_dispatch_resident_routing_mxfp4_into(
            activations,
            gate_codes,
            up_codes,
            down_codes,
            gate_exps,
            up_exps,
            down_exps,
            a_scale_buf,
            routing_tokens,
            routing_experts,
            routing_weights,
            num_pairs,
            &out_storage,
            hidden,
            inter,
            routed_scaling_factor,
        )?;
        Ok((out_storage, RocmHandle::new(Some(stream))))
    }

    /// Capture-safe variant of
    /// [`Self::moe_fused_dispatch_resident_routing_mxfp4`].
    #[allow(clippy::too_many_arguments)]
    pub fn moe_fused_dispatch_resident_routing_mxfp4_into(
        &self,
        activations: &RocmStorage,
        gate_codes: &dyn BackendStorage,
        up_codes: &dyn BackendStorage,
        down_codes: &dyn BackendStorage,
        gate_exps: &dyn BackendStorage,
        up_exps: &dyn BackendStorage,
        down_exps: &dyn BackendStorage,
        a_scale_buf: &dyn BackendStorage,
        routing_tokens: &RocmStorage,
        routing_experts: &RocmStorage,
        routing_weights: &RocmStorage,
        num_pairs: usize,
        out_storage: &RocmStorage,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
    ) -> Result<*mut c_void> {
        if (inter * hidden) % 32 != 0 {
            return Err(Error::Backend(format!(
                "mxfp4 resident_routing: inter*hidden={} not a multiple of 32",
                inter * hidden,
            )));
        }
        let gw_c = gate_codes
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| {
                Error::Backend("mxfp4 resident_routing: gate_codes downcast failed".into())
            })?;
        let uw_c = up_codes
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| {
                Error::Backend("mxfp4 resident_routing: up_codes downcast failed".into())
            })?;
        let dw_c = down_codes
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| {
                Error::Backend("mxfp4 resident_routing: down_codes downcast failed".into())
            })?;
        let gw_e = gate_exps
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| {
                Error::Backend("mxfp4 resident_routing: gate_exps downcast failed".into())
            })?;
        let uw_e = up_exps
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| {
                Error::Backend("mxfp4 resident_routing: up_exps downcast failed".into())
            })?;
        let dw_e = down_exps
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| {
                Error::Backend("mxfp4 resident_routing: down_exps downcast failed".into())
            })?;
        let ascale_r = a_scale_buf
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| {
                Error::Backend("mxfp4 resident_routing: a_scale_buf downcast failed".into())
            })?;

        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let a_ptr = activations.device_ptr.ok_or_else(|| {
            Error::Backend("mxfp4 resident_routing: activations has no device ptr".into())
        })?;
        let out_ptr = out_storage.device_ptr.ok_or_else(|| {
            Error::Backend("mxfp4 resident_routing: out has no device ptr".into())
        })?;

        // Zero output (atomicAdd accumulation).
        check_hip("mxfp4 resident_routing hipMemset(output, 0)", unsafe {
            hipMemsetAsync(
                out_ptr as *mut c_void,
                0,
                out_storage.bytes(),
                self.active_stream(),
            )
        })?;

        let wave = self.wavefront_size() as u32;
        let block_x = crate::kernels::charon::choose_block_dim(num_pairs, wave);
        let grid_x = if num_pairs == 0 {
            0
        } else {
            (num_pairs as u32).div_ceil(block_x)
        };
        if grid_x == 0 {
            return Ok(self.active_stream());
        }

        let mut a = a_ptr as *mut c_void;
        let mut gw = gw_c.device_ptr_checked()? as *mut c_void;
        let mut uw = uw_c.device_ptr_checked()? as *mut c_void;
        let mut dw = dw_c.device_ptr_checked()? as *mut c_void;
        let mut ge = gw_e.device_ptr_checked()? as *mut c_void;
        let mut ue = uw_e.device_ptr_checked()? as *mut c_void;
        let mut de = dw_e.device_ptr_checked()? as *mut c_void;
        let mut asc = ascale_r.device_ptr_checked()? as *mut c_void;
        let mut tok_ptr = routing_tokens.device_ptr_checked()? as *mut c_void;
        let mut exp_ptr = routing_experts.device_ptr_checked()? as *mut c_void;
        let mut w_ptr = routing_weights.device_ptr_checked()? as *mut c_void;
        let mut optr = out_ptr as *mut c_void;
        let mut hidden_i = hidden as i32;
        let mut inter_i = inter as i32;
        let mut num_pairs_i = num_pairs as i32;
        let mut rsf = routed_scaling_factor;

        let stream = self.launch_compute_kernel(
            "grim_moe_fused_dispatch_mxfp4",
            HipDim3::new(grid_x, 1, 1),
            HipDim3::new(block_x, 1, 1),
            &mut [
                arg(&mut a),
                arg(&mut gw),
                arg(&mut uw),
                arg(&mut dw),
                arg(&mut ge),
                arg(&mut ue),
                arg(&mut de),
                arg(&mut asc),
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

        // SPEED-DOT policy: decode-shaped W8A8-int8 MoE prefers the dot4/sudot4
        // grouped kernel on RDNA2/3/4. `dot4_entry_for` gates on arch and
        // returns None for training-unsafe paths (the dot4 kernels don't write
        // the backward pre-activation stash); the scalar kernel stays the
        // fallback so behavior off RDNA is unchanged.
        let dot4_entry = crate::kernels::charon::dot4_entry_for(
            crate::kernels::charon::CharonDot4Quant::W8A8Int8,
            self.gcn_arch(),
            false,
        );
        let stream = match dot4_entry {
            Some(entry) if hidden % 32 == 0 && inter % 32 == 0 => self
                .launch_charon_grouped_dispatch_dot4(
                    entry,
                    activations,
                    gate_r.device_ptr_checked()?,
                    up_r.device_ptr_checked()?,
                    down_r.device_ptr_checked()?,
                    ascale_r.device_ptr_checked()?,
                    sorted,
                    &out_storage,
                    hidden,
                    inter,
                    routed_scaling_factor,
                )?,
            _ => self.launch_charon_grouped_dispatch_w8a8_int8(
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
            )?,
        };
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
}
