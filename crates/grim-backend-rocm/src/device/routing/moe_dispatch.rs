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
        norm_weights: bool,
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
        let mut normw_i = i32::from(norm_weights);

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
                arg(&mut normw_i),
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

impl RocmDevice {
    /// WhiteCrow u4-group128 grouped dispatch (`grim_moe_fused_dispatch_whitecrow`):
    /// expert weights ride stacked per-expert WhiteCrow blobs (one blob per
    /// projection, expert `e` at `e * stride`, each blob
    /// `[u64][qweight u32][u64][scales bf16][u64][zeros u8]`); activations are
    /// f32 and the kernel decodes u4 weights in-register. One block per
    /// (token, expert) pair; routing is read from device buffers, so the
    /// launch is decode-graph capture-safe. Writes `routing_scaling * w`
    /// -accumulated results into `out` (zeroed here via stream memset).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_fused_dispatch_whitecrow_grouped_into(
        &self,
        activations: &RocmStorage,
        gate_blob: &RocmStorage,
        up_blob: &RocmStorage,
        down_blob: &RocmStorage,
        routing_tokens: &RocmStorage,
        routing_experts: &RocmStorage,
        routing_weights: &RocmStorage,
        num_pairs: usize,
        out_storage: &RocmStorage,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
        gate_stride: u64,
        down_stride: u64,
    ) -> Result<*mut c_void> {
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let a_ptr = activations
            .device_ptr
            .ok_or_else(|| Error::Backend("whitecrow dispatch: activations has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("whitecrow dispatch: out has no device ptr".into()))?;

        // atomicAdd accumulation requires a zeroed destination; a stream
        // memset is a capture-safe graph node.
        check_hip("whitecrow dispatch hipMemsetAsync(out, 0)", unsafe {
            hipMemsetAsync(
                out_ptr as *mut c_void,
                0,
                out_storage.bytes(),
                self.active_stream(),
            )
        })?;

        if num_pairs == 0 {
            return Ok(self.active_stream());
        }
        let mut a = a_ptr as *mut c_void;
        let mut gw = gate_blob.device_ptr_checked()? as *mut c_void;
        let mut uw = up_blob.device_ptr_checked()? as *mut c_void;
        let mut dw = down_blob.device_ptr_checked()? as *mut c_void;
        let mut tok_ptr = routing_tokens.device_ptr_checked()? as *mut c_void;
        let mut exp_ptr = routing_experts.device_ptr_checked()? as *mut c_void;
        let mut w_ptr = routing_weights.device_ptr_checked()? as *mut c_void;
        let mut optr = out_ptr as *mut c_void;
        let mut hidden_i = hidden as i32;
        let mut inter_i = inter as i32;
        let mut num_pairs_i = num_pairs as i32;
        let mut rsf = routed_scaling_factor;
        let mut gs = gate_stride;
        let mut ds = down_stride;

        // One 256-thread block per pair; shared staging holds `inter` f32.
        let stream = self.launch_compute_kernel_with_solution(
            "grim_moe_fused_dispatch_whitecrow",
            HipDim3::new(num_pairs as u32, 1, 1),
            HipDim3::new(256, 1, 1),
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
                arg(&mut gs),
                arg(&mut ds),
            ],
            None,
            inter * std::mem::size_of::<f32>(),
        )?;
        Ok(stream)
    }
}

impl RocmDevice {
    /// Native K-quant grouped dispatch (`grim_moe_fused_dispatch_kq_native`):
    /// gate/up decode in-register from the resident IQ3_S per-expert banks,
    /// down from Q4_K, addressed through device pointer arrays (one u64 base
    /// pointer per expert — the banks themselves are the model's own weights,
    /// so this arm allocates no weight bytes). One block per (token, expert)
    /// pair; routing read from device buffers; capture-safe. Output is
    /// zeroed here via a stream memset, then accumulated with
    /// `routed_scaling * w`.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_fused_dispatch_kq_native_into(
        &self,
        activations: &RocmStorage,
        gate_ptrs: &RocmStorage,
        up_ptrs: &RocmStorage,
        down_ptrs: &RocmStorage,
        routing_tokens: &RocmStorage,
        routing_experts: &RocmStorage,
        routing_weights: &RocmStorage,
        num_pairs: usize,
        out_storage: &RocmStorage,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
        gate_row_bytes: u64,
        down_row_bytes: u64,
        down_fmt: i32,
    ) -> Result<*mut c_void> {
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let a_ptr = activations
            .device_ptr
            .ok_or_else(|| Error::Backend("kq dispatch: activations has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("kq dispatch: out has no device ptr".into()))?;

        check_hip("kq dispatch hipMemsetAsync(out, 0)", unsafe {
            hipMemsetAsync(
                out_ptr as *mut c_void,
                0,
                out_storage.bytes(),
                self.active_stream(),
            )
        })?;

        if num_pairs == 0 {
            return Ok(self.active_stream());
        }
        let mut a = a_ptr as *mut c_void;
        let mut g = gate_ptrs.device_ptr_checked()? as *mut c_void;
        let mut u = up_ptrs.device_ptr_checked()? as *mut c_void;
        let mut d = down_ptrs.device_ptr_checked()? as *mut c_void;
        let mut tok_ptr = routing_tokens.device_ptr_checked()? as *mut c_void;
        let mut exp_ptr = routing_experts.device_ptr_checked()? as *mut c_void;
        let mut w_ptr = routing_weights.device_ptr_checked()? as *mut c_void;
        let mut optr = out_ptr as *mut c_void;
        let mut hidden_i = hidden as i32;
        let mut inter_i = inter as i32;
        let mut num_pairs_i = num_pairs as i32;
        let mut num_experts_i = gate_ptrs.shape().dims()[0] as i32 / 2;
        let mut batch_i = out_storage.shape().dims()[0] as i32;
        let mut rsf = routed_scaling_factor;
        let mut phase_mask_i: i32 = std::env::var("GRIM_MOE_KQ_PHASE_MASK")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(3);
        let mut gate_row_bytes_i = gate_row_bytes as i32;
        let mut down_row_bytes_i = down_row_bytes as i32;
        let mut jlimit_i: i32 = std::env::var("GRIM_MOE_KQ_JLIMIT")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(inter as i32);
        let mut down_fmt_i = down_fmt;

        // One 256-thread block per pair; dynamic shared staging holds `inter`
        // f32 — the size MUST be passed at launch (unsized dynamic LDS page-
        // faults the GPU; see the WhiteCrow twin of this launcher).
        // DETERMINISM scratch: per-pair down rows, reduced in fixed order.
        use grim_tensor::{ArithType as _AT, Storage as _ST};
        static PAIR_OUT: std::sync::OnceLock<std::sync::Mutex<Option<(usize, usize, RocmStorage)>>> =
            std::sync::OnceLock::new();
        let pcell = PAIR_OUT.get_or_init(|| std::sync::Mutex::new(None));
        let mut pguard = pcell.lock().unwrap_or_else(|e| e.into_inner());
        let np_key = num_pairs;
        if !matches!(pguard.as_ref(), Some((ord, key, _)) if *ord == self.ordinal && *key == np_key)
        {
            let po = RocmStorage::alloc_gpu(
                &Shape::new(vec![num_pairs * hidden]),
                DType { arith: _AT::F32, storage: _ST::Native },
                &self.allocator,
                self.ordinal,
            )?;
            *pguard = Some((self.ordinal, np_key, po));
        }
        let pair_out_p = pguard
            .as_ref()
            .unwrap()
            .2
            .device_ptr_checked()? as *mut c_void;
        drop(pguard);
        let mut pair_out_p = pair_out_p;
        let mut det_i: i32 = if batch_i == 1 { 1 } else { 0 };
        let stream = self.launch_compute_kernel_with_solution(
            "grim_moe_fused_dispatch_kq_native",
            HipDim3::new(num_pairs as u32, 1, 1),
            HipDim3::new(256, 1, 1),
            &mut [
                arg(&mut a),
                arg(&mut g),
                arg(&mut u),
                arg(&mut d),
                arg(&mut tok_ptr),
                arg(&mut exp_ptr),
                arg(&mut w_ptr),
                arg(&mut optr),
                arg(&mut hidden_i),
                arg(&mut inter_i),
                arg(&mut num_pairs_i),
                arg(&mut rsf),
                arg(&mut num_experts_i),
                arg(&mut batch_i),
                arg(&mut phase_mask_i),
                arg(&mut gate_row_bytes_i),
                arg(&mut down_row_bytes_i),
                arg(&mut jlimit_i),
                arg(&mut down_fmt_i),
                arg(&mut pair_out_p),
                arg(&mut det_i),
            ],
            None,
            inter * std::mem::size_of::<f32>(),
        )?;
        // DETERMINISM: the kernel now writes per-pair rows into pair_out;
        // reduce them into `out` in fixed pair order (batch is always 1 on
        // this decode path — one token's 4..top_k pairs).
        if det_i == 1 {
            let reduce_blocks = hidden.div_ceil(256) as u32;
            let mut po = pair_out_p;
            let mut np_i = num_pairs as i32;
            let mut hi2 = hidden as i32;
            self.launch_compute_kernel(
                "grim_moe_pairs_reduce",
                HipDim3::new(reduce_blocks, 1, 1),
                HipDim3::new(256, 1, 1),
                &mut [arg(&mut po), arg(&mut optr), arg(&mut hi2), arg(&mut np_i)],
            )?;
        }
        Ok(stream)
    }
}

// ---------------------------------------------------------------------------
// PREFILL dequant-once MoE arm (eval/prefill shapes, seq_len >= 32).
// For each expert: dequant its three packed banks into a rolling f32 scratch
// (44 MB, fits where a whole-layer F16 copy would not), gather the pairs
// routed to it, run gate/up/down matmuls on the proven matmul path, and
// scaled scatter-add into the layer output. NOT capture-safe (host routing
// readback) — decode stays on the per-pair kernel.
// ---------------------------------------------------------------------------
pub struct MoEDequantGemmScratch {
    pub bg_t: RocmStorage, // [inter, hidden]  (A x B^T convention)
    pub bu_t: RocmStorage, // [inter, hidden]
    pub bd_t: RocmStorage, // [hidden, inter]
    /// Per-count-bucket gather/matmul buffers, sorted by bucket size. The
    /// matmul validates A's rows against the output's M, so every GEMM must
    /// run at a shape where A, B and C agree — the bucket IS that shape.
    /// Rows beyond the expert's real count hold stale data; only `count`
    /// rows are ever scattered, and silu of stale finite data is finite.
    pub buckets: Vec<(usize, MoEBucketBufs)>,
    /// Pair-indexed down outputs, rows in TOKEN-sorted order (determinism).
    pub dtf: RocmStorage, // [num_pairs, hidden]
    pub key: (usize, usize, usize),
}

pub struct MoEBucketBufs {
    pub xe: RocmStorage, // [bucket, hidden]
    pub yg: RocmStorage, // [bucket, inter]
    pub yu: RocmStorage, // [bucket, inter]
    pub dt: RocmStorage, // [bucket, hidden]
}

static MOE_DG_SCRATCH: std::sync::OnceLock<std::sync::Mutex<Option<(usize, MoEDequantGemmScratch)>>> =
    std::sync::OnceLock::new();

impl RocmDevice {
    /// See the module-level comment. Returns Ok(()) after writing `out`.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_dequant_gemm_prefill_into(
        &self,
        x_rocm: &RocmStorage,
        g_ptrs: &RocmStorage,
        u_ptrs: &RocmStorage,
        d_ptrs: &RocmStorage,
        tokens_rocm: &RocmStorage,
        experts_rocm: &RocmStorage,
        weights_rocm: &RocmStorage,
        num_pairs: usize,
        out: &RocmStorage,
        hidden: usize,
        inter: usize,
        routed_scaling_factor: f32,
        gate_row_bytes: u64,
        down_row_bytes: u64,
        down_q4k: bool,
    ) -> Result<()> {
        use grim_tensor::{ArithType, MemoryOps, Storage};
        if std::env::var_os("GRIM_MOE_PREFILL_GEMM_TRACE").is_some() {
            eprintln!("[dg-entry] arm entered: num_pairs={num_pairs} hidden={hidden} inter={inter}");
        }
        if num_pairs == 0 {
            return Ok(());
        }

        // 1. Routing readback (tiny; prefill is not capture-bound).
        let tok_host = tokens_rocm.copy_to_host().map_err(|e| Error::Backend(e.to_string()))?;
        let exp_host = experts_rocm.copy_to_host().map_err(|e| Error::Backend(e.to_string()))?;
        let w_host = weights_rocm.copy_to_host().map_err(|e| Error::Backend(e.to_string()))?;
        let u32le = |b: &[u8], i: usize| {
            u32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]])
        };
        let f32le = |b: &[u8], i: usize| {
            f32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]])
        };
        let mut order: Vec<usize> = (0..num_pairs).collect();
        order.sort_by_key(|&p| u32le(&exp_host, p));

        // Determinism: rank every pair by (token, original index). Each
        // expert's down rows land at these positions in a pair-indexed
        // buffer, and the final per-token reduction sums them in this fixed
        // order — no atomics.
        let mut tsorted: Vec<usize> = (0..num_pairs).collect();
        tsorted.sort_by_key(|&p| (u32le(&tok_host, p), p));
        let rank_of_orig: Vec<usize> = {
            let mut r = vec![0usize; num_pairs];
            for (pos, &p) in tsorted.iter().enumerate() {
                r[p] = pos;
            }
            r
        };
        let num_tokens = if tsorted.is_empty() {
            1
        } else {
            u32le(&tok_host, tsorted[tsorted.len() - 1]) as usize + 1
        };
        let mut offs: Vec<i32> = vec![0i32; num_tokens + 1];
        {
            let mut s = 0usize;
            while s < tsorted.len() {
                let t = u32le(&tok_host, tsorted[s]) as usize;
                let mut e2 = s;
                while e2 < tsorted.len() && u32le(&tok_host, tsorted[e2]) as usize == t {
                    e2 += 1;
                }
                offs[t] = s as i32;
                offs[t + 1] = e2 as i32;
                s = e2;
            }
            // Forward-fill for tokens with no pairs (lo == hi).
            let mut prev = 0i32;
            for t in 0..num_tokens {
                if offs[t] != offs[t + 1] {
                    prev = offs[t + 1];
                } else {
                    offs[t] = prev;
                    offs[t + 1] = prev;
                }
            }
        }
        let offs_st = MemoryOps::from_cpu_bytes(
            self as &dyn grim_tensor::BackendDevice,
            unsafe { std::slice::from_raw_parts(offs.as_ptr() as *const u8, offs.len() * 4) },
            &Shape::new(vec![offs.len()]),
            DType { arith: ArithType::U32, storage: Storage::Native },
        )
        .map_err(|e2| Error::Backend(format!("dg offs h2d: {e2}")))?;
        let offs_p = crate::device::util::as_rocm(offs_st.as_ref())?
            .device_ptr_checked()? as *mut c_void;
        let sorted_toks: Vec<i32> = order.iter().map(|&p| u32le(&tok_host, p) as i32).collect();
        let sorted_wts: Vec<f32> =
            order.iter().map(|&p| f32le(&w_host, p) * routed_scaling_factor).collect();

        // 2. Scratch + stream. The cache guard is held for the whole arm so
        // the storages stay borrowed (RocmStorage is not Clone); the arm is
        // called once per layer, sequentially.
        let cell = MOE_DG_SCRATCH.get_or_init(|| std::sync::Mutex::new(None));
        let mut guard = cell.lock().unwrap_or_else(|e| e.into_inner());
        let need_key = (hidden, inter, num_pairs);
        if !matches!(guard.as_ref(), Some((ord, s)) if *ord == self.ordinal && s.key == need_key)
        {
            let f32ty = DType { arith: ArithType::F32, storage: Storage::Native };
            let s = MoEDequantGemmScratch {
                bg_t: RocmStorage::alloc_gpu(
                    &Shape::new(vec![inter, hidden]),
                    f32ty.clone(),
                    &self.allocator,
                    self.ordinal,
                )?,
                bu_t: RocmStorage::alloc_gpu(
                    &Shape::new(vec![inter, hidden]),
                    f32ty.clone(),
                    &self.allocator,
                    self.ordinal,
                )?,
                bd_t: RocmStorage::alloc_gpu(
                    &Shape::new(vec![hidden, inter]),
                    f32ty.clone(),
                    &self.allocator,
                    self.ordinal,
                )?,
                buckets: {
                    let mut buckets = Vec::new();
                    let mut b = 32usize;
                    while b < num_pairs {
                        buckets.push((
                            b,
                            MoEBucketBufs {
                                xe: RocmStorage::alloc_gpu(
                                    &Shape::new(vec![b, hidden]),
                                    f32ty.clone(),
                                    &self.allocator,
                                    self.ordinal,
                                )?,
                                yg: RocmStorage::alloc_gpu(
                                    &Shape::new(vec![b, inter]),
                                    f32ty.clone(),
                                    &self.allocator,
                                    self.ordinal,
                                )?,
                                yu: RocmStorage::alloc_gpu(
                                    &Shape::new(vec![b, inter]),
                                    f32ty.clone(),
                                    &self.allocator,
                                    self.ordinal,
                                )?,
                                dt: RocmStorage::alloc_gpu(
                                    &Shape::new(vec![b, hidden]),
                                    f32ty.clone(),
                                    &self.allocator,
                                    self.ordinal,
                                )?,
                            },
                        ));
                        b *= 2;
                    }
                    buckets.push((
                        num_pairs,
                        MoEBucketBufs {
                            xe: RocmStorage::alloc_gpu(
                                &Shape::new(vec![num_pairs, hidden]),
                                f32ty.clone(),
                                &self.allocator,
                                self.ordinal,
                            )?,
                            yg: RocmStorage::alloc_gpu(
                                &Shape::new(vec![num_pairs, inter]),
                                f32ty.clone(),
                                &self.allocator,
                                self.ordinal,
                            )?,
                            yu: RocmStorage::alloc_gpu(
                                &Shape::new(vec![num_pairs, inter]),
                                f32ty.clone(),
                                &self.allocator,
                                self.ordinal,
                            )?,
                            dt: RocmStorage::alloc_gpu(
                                &Shape::new(vec![num_pairs, hidden]),
                                f32ty.clone(),
                                &self.allocator,
                                self.ordinal,
                            )?,
                        },
                    ));
                    buckets
                },
                dtf: RocmStorage::alloc_gpu(
                &Shape::new(vec![num_pairs, hidden]),
                f32ty.clone(),
                &self.allocator,
                self.ordinal,
            )?,
            key: need_key,
            };
            *guard = Some((self.ordinal, s));
        }
        let scratch = &guard.as_ref().unwrap().1;
        let _stream = self.active_stream();

        let gate_ptrs_p = g_ptrs.device_ptr_checked()? as *mut c_void;
        let up_ptrs_p = u_ptrs.device_ptr_checked()? as *mut c_void;
        let down_ptrs_p = d_ptrs.device_ptr_checked()? as *mut c_void;
        let a_x_p = x_rocm.device_ptr_checked()? as *mut c_void;
        let mut out_p = out.device_ptr_checked()? as *mut c_void;

        // 3. Per-expert segments in the sorted order.
        let (mut t_up, mut t_dq, mut t_ga, mut t_gm, mut t_si, mut t_sc) = (
            std::time::Instant::now(),
            std::time::Instant::now(),
            std::time::Instant::now(),
            std::time::Instant::now(),
            std::time::Instant::now(),
            std::time::Instant::now(),
        );
        let mut n_experts_seen = 0usize;
        let mut seg_start = 0usize;
        while seg_start < num_pairs {
            let e = u32le(&exp_host, order[seg_start]) as usize;
            let mut seg_end = seg_start;
            while seg_end < num_pairs && u32le(&exp_host, order[seg_end]) == e as u32 {
                seg_end += 1;
            }
            let count = seg_end - seg_start;
            let toks_slice: Vec<i32> = sorted_toks[seg_start..seg_end].to_vec();
            let wts_slice: Vec<f32> = sorted_wts[seg_start..seg_end].to_vec();
            t_up = std::time::Instant::now();
            let toks_st = MemoryOps::from_cpu_bytes(
                self as &dyn grim_tensor::BackendDevice,
                unsafe { std::slice::from_raw_parts(toks_slice.as_ptr() as *const u8, count * 4) },
                &Shape::new(vec![count]),
                DType { arith: ArithType::U32, storage: Storage::Native },
            )
            .map_err(|e2| Error::Backend(format!("dg toks h2d: {e2}")))?;
            let wts_st = MemoryOps::from_cpu_bytes(
                self as &dyn grim_tensor::BackendDevice,
                unsafe { std::slice::from_raw_parts(wts_slice.as_ptr() as *const u8, count * 4) },
                &Shape::new(vec![count]),
                DType { arith: ArithType::F32, storage: Storage::Native },
            )
            .map_err(|e2| Error::Backend(format!("dg wts h2d: {e2}")))?;
            let mut toks_p = crate::device::util::as_rocm(toks_st.as_ref())?
                .device_ptr_checked()? as *mut c_void;
            let mut wts_p = crate::device::util::as_rocm(wts_st.as_ref())?
                .device_ptr_checked()? as *mut c_void;
            let idx_slice: Vec<i32> = order[seg_start..seg_end]
                .iter()
                .map(|&orig| rank_of_orig[orig] as i32)
                .collect();
            let idx_st = MemoryOps::from_cpu_bytes(
                self as &dyn grim_tensor::BackendDevice,
                unsafe {
                    std::slice::from_raw_parts(idx_slice.as_ptr() as *const u8, count * 4)
                },
                &Shape::new(vec![count]),
                DType { arith: ArithType::U32, storage: Storage::Native },
            )
            .map_err(|e2| Error::Backend(format!("dg idx h2d: {e2}")))?;
            let idx_p = crate::device::util::as_rocm(idx_st.as_ref())?
                .device_ptr_checked()? as *mut c_void;

            // (a) dequant expert e's banks, transposed, into the scratch.
            t_dq = std::time::Instant::now();
            n_experts_seen += 1;
            let mut gate_ptrs_p = gate_ptrs_p;
            let mut up_ptrs_p = up_ptrs_p;
            let mut down_ptrs_p = down_ptrs_p;
            let mut og = scratch.bg_t.device_ptr_checked()? as *mut c_void;
            let mut ou = scratch.bu_t.device_ptr_checked()? as *mut c_void;
            let mut od = scratch.bd_t.device_ptr_checked()? as *mut c_void;
            let mut inter_i = inter as i32;
            let mut hidden_i = hidden as i32;
            let mut expert_i = e as i32;
            let mut grb_i = gate_row_bytes as i32;
            let mut drb_i = down_row_bytes as i32;
            let mut dfmt_i = down_q4k as i32;
            self.launch_compute_kernel(
                "grim_dequant_expert_banks_f32",
                HipDim3::new(hidden.max(inter) as u32, 3, 1),
                HipDim3::new(256, 1, 1),
                &mut [
                    arg(&mut gate_ptrs_p),
                    arg(&mut up_ptrs_p),
                    arg(&mut down_ptrs_p),
                    arg(&mut og),
                    arg(&mut ou),
                    arg(&mut od),
                    arg(&mut inter_i),
                    arg(&mut hidden_i),
                    arg(&mut expert_i),
                    arg(&mut grb_i),
                    arg(&mut drb_i),
                    arg(&mut dfmt_i),
                ],
            )?;

            // Bucket this expert's count: all GEMMs run at the bucket shape
            // (A rows == C rows) so the matmul shape checks pass and every
            // buffer is preallocated — no per-call allocation.
            let bucket = scratch
                .buckets
                .iter()
                .find(|(b, _)| *b >= count)
                .map(|(b, bufs)| (*b, bufs))
                .ok_or_else(|| {
                    Error::Backend(format!("no bucket for count={count} (num_pairs={num_pairs})"))
                })?;
            let (bsize, bb) = bucket;

            // (b) gather this expert's token rows into the bucket's Xe.
            t_ga = std::time::Instant::now();
            let mut a_x_p = a_x_p;
            let mut xe_p = bb.xe.device_ptr_checked()? as *mut c_void;
            self.launch_compute_kernel(
                "grim_moe_gather_rows",
                HipDim3::new(count as u32, 1, 1),
                HipDim3::new(256, 1, 1),
                &mut [arg(&mut a_x_p), arg(&mut toks_p), arg(&mut xe_p), arg(&mut hidden_i)],
            )?;

            // (c) gate / up matmuls on the proven path, into preallocated
            //     bucket outputs: Y[bsize,inter] = Xe[bsize,hidden] x B[hidden,inter]
            t_gm = std::time::Instant::now();
            let bg_ref: &dyn grim_tensor::BackendStorage = &scratch.bg_t;
            let bu_ref: &dyn grim_tensor::BackendStorage = &scratch.bu_t;
            let bd_ref: &dyn grim_tensor::BackendStorage = &scratch.bd_t;
            self.matmul_op_into(&bb.xe, bg_ref, &bb.yg, crate::autotune::GemmOp::Ffn)?;
            self.matmul_op_into(&bb.xe, bu_ref, &bb.yu, crate::autotune::GemmOp::Ffn)?;
            let mut yg_p = bb.yg.device_ptr_checked()? as *mut c_void;
            let mut yu_p = bb.yu.device_ptr_checked()? as *mut c_void;

            // Probe: manual gate dot vs the bucket GEMM (first expert, layer 0).
            if std::env::var_os("GRIM_MOE_PREFILL_GEMM_TRACE").is_some() && seg_start == 0 {
                static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
                if ONCE.set(()).is_ok() {
                    let rd = |ptr: *const c_void, n: usize| -> Vec<f32> {
                        let mut b = vec![0u8; n * 4];
                        let _ = unsafe {
                            crate::hipMemcpy(
                                b.as_mut_ptr() as *mut c_void,
                                ptr,
                                n * 4,
                                crate::HipMemcpyKind::DeviceToHost,
                            )
                        };
                        b.chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect()
                    };
                    let xe0 = rd(bb.xe.device_ptr_checked()? as *const c_void, hidden);
                    let bg0 = rd(scratch.bg_t.device_ptr_checked()? as *const c_void, hidden);
                    let manual: f32 = xe0.iter().zip(&bg0).map(|(a, b)| a * b).sum();
                    let yg0 = rd(yg_p as *const c_void, 4);
                    eprintln!(
                        "[dg-parity] count={count} bucket={bsize} manual_gate_y0={manual:.5} gemm_y0={:.5} xe0[0..3]={:?} bg0[0..3]={:?}",
                        yg0[0],
                        &xe0[..3],
                        &bg0[..3]
                    );
                }
            }

            // (d) silu(gate) * up over the whole bucket.
            t_si = std::time::Instant::now();
            let mut n_silu = (bsize * inter) as i32;
            let blocks = (bsize * inter).div_ceil(256) as u32;
            self.launch_compute_kernel(
                "grim_silu_mul",
                HipDim3::new(blocks, 1, 1),
                HipDim3::new(256, 1, 1),
                &mut [arg(&mut yg_p), arg(&mut yu_p), arg(&mut yg_p), arg(&mut n_silu)],
            )?;

            // (e) down matmul: D[bsize,hidden] = C[bsize,inter] x Bd[inter,hidden]
            self.matmul_op_into(&bb.yg, bd_ref, &bb.dt, crate::autotune::GemmOp::Ffn)?;
            let mut dt_p = bb.dt.device_ptr_checked()? as *mut c_void;

            // Probe: silu output and the down GEMM (first expert, layer 0).
            if std::env::var_os("GRIM_MOE_PREFILL_GEMM_TRACE").is_some() && seg_start == 0 {
                static ONCE2: std::sync::OnceLock<()> = std::sync::OnceLock::new();
                if ONCE2.set(()).is_ok() {
                    let rd = |ptr: *const c_void, n: usize| -> Vec<f32> {
                        let mut b = vec![0u8; n * 4];
                        let _ = unsafe {
                            crate::hipMemcpy(
                                b.as_mut_ptr() as *mut c_void,
                                ptr,
                                n * 4,
                                crate::HipMemcpyKind::DeviceToHost,
                            )
                        };
                        b.chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect()
                    };
                    let c0 = rd(yg_p as *const c_void, inter); // post-silu row 0
                    let yu0 = rd(yu_p as *const c_void, 4);
                    eprintln!("[dg-parity2] yu0[0..4]={:?}", yu0);
                    let bd0 = rd(scratch.bd_t.device_ptr_checked()? as *const c_void, inter);
                    let manual_d: f32 = c0.iter().zip(&bd0).map(|(a, b)| a * b).sum();
                    let dt0 = rd(dt_p as *const c_void, 4);
                    eprintln!(
                        "[dg-parity2] c0[0..3]={:?} bd0[0..3]={:?} manual_d0={manual_d:.5} gemm_d0={:.5}",
                        &c0[..3],
                        &bd0[..3],
                        dt0[0]
                    );
                }
            }

            // (f) scaled scatter-add into the layer output.
            t_sc = std::time::Instant::now();
            if std::env::var_os("GRIM_DG_NO_SCATTER").is_some() {
                seg_start = seg_end;
                continue;
            }
            // DETERMINISTIC: write w*D into pair-indexed rows (token-sorted
            // order), then reduce per token in fixed order. No atomics.
            let mut idx_p2 = idx_p;
            let mut dtf_p = scratch.dtf.device_ptr_checked()? as *mut c_void;
            self.launch_compute_kernel(
                "grim_moe_scatter_rows_idx",
                HipDim3::new(count as u32, 1, 1),
                HipDim3::new(256, 1, 1),
                &mut [
                    arg(&mut dtf_p),
                    arg(&mut idx_p2),
                    arg(&mut wts_p),
                    arg(&mut dt_p),
                    arg(&mut hidden_i),
                ],
            )?;
            let mut offs_p2 = offs_p;
            self.launch_compute_kernel(
                "grim_moe_token_reduce",
                HipDim3::new(num_tokens as u32, 1, 1),
                HipDim3::new(256, 1, 1),
                &mut [arg(&mut dtf_p), arg(&mut offs_p2), arg(&mut out_p), arg(&mut hidden_i)],
            )?;

            if std::env::var_os("GRIM_MOE_PREFILL_GEMM_TRACE").is_some() && seg_start == 0 {
                static ONCE3: std::sync::OnceLock<()> = std::sync::OnceLock::new();
                if ONCE3.set(()).is_ok() {
                    let tok0 = toks_slice[0] as usize;
                    let mut head = [0f32; 4];
                    let _ = unsafe {
                        crate::hipMemcpy(
                            head.as_mut_ptr() as *mut c_void,
                            (out_p as usize + tok0 * hidden * 4) as *mut c_void,
                            16,
                            crate::HipMemcpyKind::DeviceToHost,
                        )
                    };
                    let mut dt_head = [0f32; 4];
                    let _ = unsafe {
                        crate::hipMemcpy(
                            dt_head.as_mut_ptr() as *mut c_void,
                            dt_p as *const c_void,
                            16,
                            crate::HipMemcpyKind::DeviceToHost,
                        )
                    };
                    eprintln!(
                        "[dg-scatter] tok0={tok0} w0={:.4} out[tok0][0..4]={head:?} dt[0][0..4]={dt_head:?}",
                        wts_slice[0]
                    );
                }
            }
            seg_start = seg_end;
        }
        if std::env::var_os("GRIM_MOE_PREFILL_GEMM_TRACE").is_some() {
            let covered: usize = {
                // recompute coverage cheaply from the segments already walked
                let mut c = 0usize;
                let mut s = 0usize;
                while s < num_pairs {
                    let e = u32le(&exp_host, order[s]) as u32;
                    let mut t = s;
                    while t < num_pairs && u32le(&exp_host, order[t]) == e {
                        t += 1;
                    }
                    c += t - s;
                    s = t;
                }
                c
            };
            eprintln!(
                "[dg-trace] layer done: covered={covered}/{num_pairs} experts={} upload={}ms dequant={}ms gather={}ms gemm={}ms silu={}ms scatter={}ms",
                n_experts_seen,
                t_up.elapsed().as_millis(),
                t_dq.elapsed().as_millis(),
                t_ga.elapsed().as_millis(),
                t_gm.elapsed().as_millis(),
                t_si.elapsed().as_millis(),
                t_sc.elapsed().as_millis(),
            );
        }
        Ok(())
    }
}
