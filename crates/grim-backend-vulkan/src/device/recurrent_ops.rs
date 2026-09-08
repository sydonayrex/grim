//! `RecurrentOps` implementation for VulkanDevice.
//! Extracted from lib.rs (modularization): trait impls live in `device/`, dispatch plumbing in `kernel.rs`, buffers in.

use grim_tensor::backend::ComputeHandle;
use grim_tensor::dtype::DType;
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendStorage, CoreTensorOps, RecurrentOps, Shape};

use crate::context::global_context;
use crate::kernel::{VulkanKernel, push_params, run_compute_shader_kernel};
use crate::{VulkanDevice, VulkanHandle, VulkanStorage};

impl RecurrentOps for VulkanDevice {
    fn selective_scan(
        &self,
        x: &dyn BackendStorage,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        c: &dyn BackendStorage,
        d: &dyn BackendStorage,
        _state: &dyn BackendStorage,
        batch: usize,
        dim_dstate: usize,
        dim_dinner: usize,
        seq_len: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        // Note: GPU fast path skipped until buffer layout matches CPU semantics and end-to-end golden verification passes.
        tracing::warn!("Vulkan selective_scan: falling back to CPU execution");
        let x_v = x.to_cpu_vec_f32()?;
        let a_v = a.to_cpu_vec_f32()?;
        let b_v = b.to_cpu_vec_f32()?;
        let c_v = c.to_cpu_vec_f32()?;
        let d_v = d.to_cpu_vec_f32()?;

        let mut out = vec![0.0f32; batch * seq_len * dim_dinner];
        for b_idx in 0..batch {
            for d_idx in 0..dim_dinner {
                let mut h = vec![0.0f32; dim_dstate];
                let d_val = if d_v.len() > d_idx { d_v[d_idx] } else { 0.0 };

                for t in 0..seq_len {
                    let x_idx = (b_idx * seq_len + t) * dim_dinner + d_idx;
                    let x_t = x_v[x_idx];
                    let mut y_t = d_val * x_t;

                    for (s, h_s) in h.iter_mut().enumerate() {
                        let a_idx = d_idx * dim_dstate + s;
                        let b_idx_off = (b_idx * seq_len + t) * dim_dstate + s;
                        let c_idx_off = (b_idx * seq_len + t) * dim_dstate + s;

                        let a_val = if a_v.len() > a_idx { a_v[a_idx] } else { 1.0 };
                        let b_val = if b_v.len() > b_idx_off {
                            b_v[b_idx_off]
                        } else {
                            1.0
                        };
                        let c_val = if c_v.len() > c_idx_off {
                            c_v[c_idx_off]
                        } else {
                            1.0
                        };

                        *h_s = a_val * *h_s + x_t * b_val;
                        y_t += c_val * *h_s;
                    }
                    out[x_idx] = y_t;
                }
            }
        }

        let out_storage = self.from_cpu(&out, out_shape, x.dtype())?;
        Ok((out_storage, Box::new(VulkanHandle)))
    }

    fn rwkv_time_mix(
        &self,
        x: &dyn BackendStorage,
        w: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        g: &dyn BackendStorage,
        batch: usize,
        dim: usize,
        seq_len: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        // Note: GPU fast path skipped until buffer layout matches CPU semantics and end-to-end golden verification passes.
        tracing::warn!("Vulkan rwkv_time_mix: falling back to CPU execution");
        let x_vec = x.to_cpu_vec_f32()?;
        let k_vec = k.to_cpu_vec_f32()?;
        let v_vec = v.to_cpu_vec_f32()?;
        let g_vec = g.to_cpu_vec_f32()?;
        let w_vec = w.to_cpu_vec_f32()?;

        let mut out = vec![0.0f32; batch * seq_len * dim];
        for b in 0..batch {
            for d in 0..dim {
                let mut state = 0.0f32;
                let w_val = if w_vec.len() > d { w_vec[d] } else { 0.9f32 };

                for t in 0..seq_len {
                    let idx = (b * seq_len + t) * dim + d;
                    let k_t = if k_vec.len() > idx {
                        k_vec[idx]
                    } else {
                        x_vec[idx]
                    };
                    let v_t = if v_vec.len() > idx {
                        v_vec[idx]
                    } else {
                        x_vec[idx]
                    };
                    let g_t = if g_vec.len() > idx {
                        g_vec[idx]
                    } else {
                        1.0f32
                    };

                    state = w_val * state + k_t * v_t;
                    let sig = 1.0f32 / (1.0f32 + (-g_t).exp());
                    out[idx] = state * sig;
                }
            }
        }

        let out_storage = self.from_cpu(&out, out_shape, x.dtype())?;
        Ok((out_storage, Box::new(VulkanHandle)))
    }

    fn rwkv_channel_mix(
        &self,
        x: &dyn BackendStorage,
        k: &dyn BackendStorage,
        r: &dyn BackendStorage,
        v: &dyn BackendStorage,
        batch: usize,
        dim: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        // Note: GPU fast path skipped until buffer layout matches CPU semantics and end-to-end golden verification passes.
        tracing::warn!("Vulkan rwkv_channel_mix: falling back to CPU execution");
        let x_vec = x.to_cpu_vec_f32()?;
        let k_vec = k.to_cpu_vec_f32()?;
        let r_vec = r.to_cpu_vec_f32()?;
        let v_vec = v.to_cpu_vec_f32()?;

        let elem_count = out_shape.elem_count();
        let mut out = vec![0.0f32; elem_count];
        for i in 0..elem_count {
            let x_val = x_vec[i];
            let k_val = if k_vec.len() > i { k_vec[i] } else { x_val };
            let r_val = if r_vec.len() > i { r_vec[i] } else { 1.0f32 };
            let v_val = if v_vec.len() > i { v_vec[i] } else { x_val };

            let sig_r = 1.0f32 / (1.0f32 + (-r_val).exp());
            let relu_k = k_val.max(0.0f32);
            out[i] = sig_r * (relu_k * relu_k) * v_val;
        }

        let _ = batch;
        let _ = dim;

        let out_storage = self.from_cpu(&out, out_shape, x.dtype())?;
        Ok((out_storage, Box::new(VulkanHandle)))
    }
    // Tier B complex - recurrent/conv-step kernels (audit gap: LFM2/KDA/SSM decode hit Err(Unimplemented) on Vulkan).
    // CPU-reference fallbacks that mirror the documented kernel contract.

    /// Depthwise 1D causal convolution decode step with a rolling conv-state buffer.
    /// `state` holds the past `k_size-1` inputs; the new `x` is the current input.
    fn short_conv1d_causal_step(
        &self,
        x: &dyn BackendStorage,
        weight: &dyn BackendStorage,
        bias: Option<&dyn BackendStorage>,
        state: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_v = x.to_cpu_vec_f32()?;
        let w_v = weight.to_cpu_vec_f32()?;
        let b_v = bias.map(|b| b.to_cpu_vec_f32()).transpose()?;
        let st_v = state.to_cpu_vec_f32()?;
        let dims = out_shape.dims();
        let (batch, k_size, channels) = match dims.len() {
            3 => (dims[0], dims[1], dims[2]),
            2 => (1, dims[0], dims[1]),
            _ => {
                return Err(Error::Shape(
                    "short_conv1d: expected 2-D or 3-D out_shape".into(),
                ));
            }
        };
        // weight layout: [channels, 1, k_size] flattened -> stride k_size.
        if w_v.len() != channels * k_size {
            return Err(Error::Shape("short_conv1d: weight size mismatch".into()));
        }
        if x_v.len() != batch * channels {
            return Err(Error::Shape("short_conv1d: x size mismatch".into()));
        }
        let mut out = vec![0.0f32; batch * k_size * channels];
        for b_idx in 0..batch {
            for k in 0..k_size {
                for c in 0..channels {
                    let mut acc = b_v.as_ref().map(|b| b[c]).unwrap_or(0.0);
                    // Causal: tap t uses state when (k - t) falls in the past.
                    for t in 0..k_size {
                        let w = w_v[c * k_size + t];
                        let src = if t <= k {
                            // current + recent past from x/state
                            let back = k - t;
                            if back == 0 {
                                x_v[b_idx * channels + c]
                            } else {
                                st_v[c * (k_size - 1) + (back - 1)]
                            }
                        } else {
                            0.0
                        };
                        acc += w * src;
                    }
                    out[(b_idx * k_size + k) * channels + c] = acc;
                }
            }
        }
        let storage = self.from_cpu(&out, out_shape, DType::F32)?;
        Ok((storage, Box::new(grim_tensor::backend::ReadyHandle)))
    }

    /// KDA gated delta-rule recurrent step: `S' = g·S + β·v·kᵀ`, `o = q·S'` (gated delta rule, DeltaNet-family).
    /// ROCm kernel: `grim_kda_gated_delta_rule_step`.
    fn kda_gated_delta_rule_step(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        beta: &dyn BackendStorage,
        a_gate: &dyn BackendStorage,
        recurrent_state: &dyn BackendStorage,
        d_k: usize,
        d_v: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let q_v = q.to_cpu_vec_f32()?;
        let k_v = k.to_cpu_vec_f32()?;
        let v_v = v.to_cpu_vec_f32()?;
        let beta_v = beta.to_cpu_vec_f32()?;
        let g_v = a_gate.to_cpu_vec_f32()?;
        let s_v = recurrent_state.to_cpu_vec_f32()?;
        let state_len = d_k * d_v;
        if s_v.len() < state_len {
            return Err(Error::Shape("kda: recurrent_state too small".into()));
        }
        if q_v.len() < d_k || k_v.len() < d_k || v_v.len() < d_v {
            return Err(Error::Shape("kda: q/k/v size mismatch".into()));
        }
        // S is [d_k, d_v]; update S'[i,j] = gate*S[i,j] + beta*v[j]*k[i].
        let mut s_new = vec![0.0f32; state_len];
        let gate = g_v.first().copied().unwrap_or(1.0);
        let b = beta_v.first().copied().unwrap_or(1.0);
        for i in 0..d_k {
            for j in 0..d_v {
                s_new[i * d_v + j] = gate * s_v[i * d_v + j] + b * v_v[j] * k_v[i];
            }
        }
        // output o[i] = Σ_j q[i] ... -> o = q-weighted readout: o[j] = Σ_i q[i]*S'[i,j].
        let out_len = out_shape.elem_count();
        let mut out = vec![0.0f32; out_len];
        for j in 0..d_v.min(out_len) {
            let mut acc = 0.0f32;
            for i in 0..d_k {
                acc += q_v[i] * s_new[i * d_v + j];
            }
            out[j] = acc;
        }
        let storage = self.from_cpu(&out, out_shape, DType::F32)?;
        Ok((storage, Box::new(grim_tensor::backend::ReadyHandle)))
    }

    // --- New kernel dispatch functions (modularization-era additions) ---

    /// Standard delta rule recurrence (DeltaNet / SolarOpen2).
    fn delta_rule_decode(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        beta: f32,
        state: &dyn BackendStorage,
        d_k: usize,
        d_v: usize,
        num_heads: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let q_s = q
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan delta_rule: q is not VulkanStorage".into()))?;
        let k_s = k
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan delta_rule: k is not VulkanStorage".into()))?;
        let v_s = v
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan delta_rule: v is not VulkanStorage".into()))?;
        let state_s = state
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan delta_rule: state is not VulkanStorage".into())
            })?;

        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;

        let out_storage = VulkanStorage::alloc_device_local_gpu(
            out_shape,
            DType::F32,
            ctx.device,
            ctx.physical_device,
        )?;

        let buffers = [
            q_s.buffer,
            k_s.buffer,
            v_s.buffer,
            state_s.buffer,
            out_storage.buffer,
        ];
        let push = push_params(d_k as u32, d_v as u32, 0, 0, 0, beta);
        let grid_y = num_heads.max(1) as u32;

        run_compute_shader_kernel(
            ctx,
            VulkanKernel::DeltaRuleDecode,
            &buffers,
            d_v.max(1) as u32,
            grid_y,
            1,
            Some(&push),
        )
        .map_err(|e| Error::Backend(format!("Vulkan delta_rule dispatch failed: {e}")))?;

        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    /// RWKV-4 WKV weighted key-value recurrence.
    fn rwkv_wkv_recurrence(
        &self,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        r: &dyn BackendStorage,
        time_first: &dyn BackendStorage,
        time_decay: &dyn BackendStorage,
        state_aa: &dyn BackendStorage,
        state_bb: &dyn BackendStorage,
        state_pp: &dyn BackendStorage,
        dim: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let k_s = k
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan rwkv_wkv: k is not VulkanStorage".into()))?;
        let v_s = v
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan rwkv_wkv: v is not VulkanStorage".into()))?;
        let r_s = r
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan rwkv_wkv: r is not VulkanStorage".into()))?;
        let tf_s = time_first
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan rwkv_wkv: time_first is not VulkanStorage".into())
            })?;
        let td_s = time_decay
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan rwkv_wkv: time_decay is not VulkanStorage".into())
            })?;
        let aa_s = state_aa
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan rwkv_wkv: state_aa is not VulkanStorage".into())
            })?;
        let bb_s = state_bb
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan rwkv_wkv: state_bb is not VulkanStorage".into())
            })?;
        let pp_s = state_pp
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| {
                Error::Backend("Vulkan rwkv_wkv: state_pp is not VulkanStorage".into())
            })?;

        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;

        let out_storage = VulkanStorage::alloc_device_local_gpu(
            out_shape,
            DType::F32,
            ctx.device,
            ctx.physical_device,
        )?;

        let buffers = [
            k_s.buffer,
            v_s.buffer,
            r_s.buffer,
            tf_s.buffer,
            td_s.buffer,
            aa_s.buffer,
            bb_s.buffer,
            pp_s.buffer,
            out_storage.buffer,
        ];
        let push = [dim as u32, 0, 0, 0, 0, 0];

        run_compute_shader_kernel(
            ctx,
            VulkanKernel::RwkvWkvRecurrence,
            &buffers,
            dim.max(1) as u32,
            1,
            1,
            Some(&push),
        )
        .map_err(|e| Error::Backend(format!("Vulkan rwkv_wkv dispatch failed: {e}")))?;

        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }

    /// RWKV-4 channel-mix token-shift + gating.
    fn rwkv_channel_mix_full(
        &self,
        x: &dyn BackendStorage,
        mix_k: &dyn BackendStorage,
        mix_r: &dyn BackendStorage,
        ffn_xx: &dyn BackendStorage,
        dim: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = x
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan rwkv_cm: x is not VulkanStorage".into()))?;
        let mk_s = mix_k
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan rwkv_cm: mix_k is not VulkanStorage".into()))?;
        let mr_s = mix_r
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan rwkv_cm: mix_r is not VulkanStorage".into()))?;
        let xx_s = ffn_xx
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .ok_or_else(|| Error::Backend("Vulkan rwkv_cm: ffn_xx is not VulkanStorage".into()))?;

        let ctx_guard = global_context();
        let ctx = ctx_guard
            .as_ref()
            .ok_or_else(|| Error::Backend("Vulkan context uninitialized".into()))?;

        let out_storage = VulkanStorage::alloc_device_local_gpu(
            out_shape,
            DType::F32,
            ctx.device,
            ctx.physical_device,
        )?;

        let buffers = [
            x_s.buffer,
            mk_s.buffer,
            mr_s.buffer,
            xx_s.buffer,
            out_storage.buffer,
        ];
        let push = [dim as u32, 0, 0, 0, 0, 0];

        run_compute_shader_kernel(
            ctx,
            VulkanKernel::RwkvChannelMixFull,
            &buffers,
            dim.max(1) as u32,
            1,
            1,
            Some(&push),
        )
        .map_err(|e| Error::Backend(format!("Vulkan rwkv_cm dispatch failed: {e}")))?;

        Ok((
            Box::new(out_storage),
            Box::new(grim_tensor::backend::ReadyHandle),
        ))
    }
}
