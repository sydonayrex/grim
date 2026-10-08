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
            ],
            None,
            inter * std::mem::size_of::<f32>(),
        )?;
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
    pub bg_t: RocmStorage, // [hidden, inter]  gate^T
    pub bu_t: RocmStorage, // [hidden, inter]  up^T
    pub bd_t: RocmStorage, // [inter, hidden]  down^T
    pub xe: RocmStorage,   // [num_pairs, hidden]
    pub key: (usize, usize, usize),
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
                    &Shape::new(vec![hidden, inter]),
                    f32ty.clone(),
                    &self.allocator,
                    self.ordinal,
                )?,
                bu_t: RocmStorage::alloc_gpu(
                    &Shape::new(vec![hidden, inter]),
                    f32ty.clone(),
                    &self.allocator,
                    self.ordinal,
                )?,
                bd_t: RocmStorage::alloc_gpu(
                    &Shape::new(vec![inter, hidden]),
                    f32ty.clone(),
                    &self.allocator,
                    self.ordinal,
                )?,
                xe: RocmStorage::alloc_gpu(
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

            // (a) dequant expert e's banks, transposed, into the scratch.
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

            // (b) gather this expert's token rows into Xe rows 0..count.
            let mut a_x_p = a_x_p;
            let mut xe_p = scratch.xe.device_ptr_checked()? as *mut c_void;
            self.launch_compute_kernel(
                "grim_moe_gather_rows",
                HipDim3::new(count as u32, 1, 1),
                HipDim3::new(256, 1, 1),
                &mut [arg(&mut a_x_p), arg(&mut toks_p), arg(&mut xe_p), arg(&mut hidden_i)],
            )?;

            // (c) gate / up matmuls on the proven path:
            //     Y[count,inter] = Xe[count,hidden] x B[hidden,inter]
            // matmul reads A's rows 0..count (k from its shape's last dim).
            let bg_ref: &dyn grim_tensor::BackendStorage = &scratch.bg_t;
            let bu_ref: &dyn grim_tensor::BackendStorage = &scratch.bu_t;
            let bd_ref: &dyn grim_tensor::BackendStorage = &scratch.bd_t;
            let xe_ref: &dyn grim_tensor::BackendStorage = &scratch.xe;
            let (yg_st, _) = self.matmul_op(
                xe_ref,
                bg_ref,
                &Shape::new(vec![count, inter]),
                crate::autotune::GemmOp::Ffn,
            )?;
            let (yu_st, _) = self.matmul_op(
                xe_ref,
                bu_ref,
                &Shape::new(vec![count, inter]),
                crate::autotune::GemmOp::Ffn,
            )?;
            let mut yg_p = crate::device::util::as_rocm(yg_st.as_ref())?
                .device_ptr_checked()? as *mut c_void;
            let mut yu_p = crate::device::util::as_rocm(yu_st.as_ref())?
                .device_ptr_checked()? as *mut c_void;

            // (d) silu(gate) * up over [count, inter].
            let mut n_silu = (count * inter) as i32;
            let blocks = (count * inter).div_ceil(256) as u32;
            self.launch_compute_kernel(
                "grim_silu_mul",
                HipDim3::new(blocks, 1, 1),
                HipDim3::new(256, 1, 1),
                &mut [arg(&mut yg_p), arg(&mut yu_p), arg(&mut yg_p), arg(&mut n_silu)],
            )?;

            // (e) down matmul: D[count,hidden] = C[count,inter] x Bd[inter,hidden]
            let (dt_st, _) = self.matmul_op(
                yg_st.as_ref(),
                bd_ref,
                &Shape::new(vec![count, hidden]),
                crate::autotune::GemmOp::Ffn,
            )?;
            let mut dt_p = crate::device::util::as_rocm(dt_st.as_ref())?
                .device_ptr_checked()? as *mut c_void;

            // (f) scaled scatter-add into the layer output.
            self.launch_compute_kernel(
                "grim_moe_scatter_add_rows",
                HipDim3::new(count as u32, 1, 1),
                HipDim3::new(256, 1, 1),
                &mut [
                    arg(&mut out_p),
                    arg(&mut toks_p),
                    arg(&mut wts_p),
                    arg(&mut dt_p),
                    arg(&mut hidden_i),
                ],
            )?;

            seg_start = seg_end;
        }
        Ok(())
    }
}
