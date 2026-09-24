//! recurrent_ops ops for MetalDevice — moved verbatim from lib.rs.

use grim_tensor::backend::ComputeHandle;
#[allow(unused_imports)]
use grim_tensor::dtype::{
    DType, FloatPackScheme, KQuantScheme, QuantFormat, QuantProvenance, Storage as DTypeStorage,
};
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendStorage, CoreTensorOps, RecurrentOps, Shape};

#[cfg(target_vendor = "apple")]
use objc2::rc::Retained;
#[cfg(target_vendor = "apple")]
use objc2::runtime::ProtocolObject;
#[cfg(target_vendor = "apple")]
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLComputePipelineState, MTLDevice, MTLSize,
};

use crate::*;

impl RecurrentOps for MetalDevice {
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
        Ok((out_storage, Box::new(grim_tensor::backend::ReadyHandle)))
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
        let x_vec = x.to_cpu_vec_f32()?;
        let k_vec = k.to_cpu_vec_f32()?;
        let v_vec = v.to_cpu_vec_f32()?;
        let g_vec = g.to_cpu_vec_f32()?;
        let w_vec = w.to_cpu_vec_f32()?;
        tracing::warn!("Metal rwkv_time_mix: falling back to CPU execution");
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
        Ok((out_storage, Box::new(grim_tensor::backend::ReadyHandle)))
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
        let x_vec = x.to_cpu_vec_f32()?;
        let k_vec = k.to_cpu_vec_f32()?;
        let r_vec = r.to_cpu_vec_f32()?;
        let v_vec = v.to_cpu_vec_f32()?;
        tracing::warn!("Metal rwkv_channel_mix: falling back to CPU execution");
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
        Ok((out_storage, Box::new(grim_tensor::backend::ReadyHandle)))
    }

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
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                use grim_tensor::BackendStorage;
                let q_s = q.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("delta_rule_decode: q not MetalStorage".into())
                })?;
                let k_s = k.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("delta_rule_decode: k not MetalStorage".into())
                })?;
                let v_s = v.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("delta_rule_decode: v not MetalStorage".into())
                })?;
                let st_s = state.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("delta_rule_decode: state not MetalStorage".into())
                })?;

                let q_buf = q_s.buffer.as_ref().ok_or_else(|| {
                    Error::Backend("delta_rule_decode: q buffer None".into())
                })?;
                let k_buf = k_s.buffer.as_ref().ok_or_else(|| {
                    Error::Backend("delta_rule_decode: k buffer None".into())
                })?;
                let v_buf = v_s.buffer.as_ref().ok_or_else(|| {
                    Error::Backend("delta_rule_decode: v buffer None".into())
                })?;
                let st_buf = st_s.buffer.as_ref().ok_or_else(|| {
                    Error::Backend("delta_rule_decode: state buffer None".into())
                })?;

                let total_d = d_k.max(d_v);
                let out_storage = self.zeros(&Shape::new(vec![total_d]), q.dtype())?;
                let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("delta_rule_decode: out allocation failed".into())
                })?;
                let out_buf = out_s.buffer.as_ref().ok_or_else(|| {
                    Error::Backend("delta_rule_decode: out buffer None".into())
                })?;

                let ctx = inner.get_or_create_context()?;
                let cmd = ctx.get_or_create_command_buffer()?;
                let enc = cmd.computeCommandEncoder().ok_or_else(|| {
                    Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                })?;

                enc.setComputePipelineState(&ctx.pipelines.delta_rule_decode);
                enc.setBuffer_offset_atIndex(Some(q_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(k_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(v_buf), 0, 2);
                enc.setBuffer_offset_atIndex(Some(st_buf), 0, 3);
                enc.setBuffer_offset_atIndex(Some(out_buf), 0, 4);

                let betas = [d_k as i32, d_v as i32, (num_heads as f32 * beta) as i32];
                unsafe {
                    for (i, &v) in betas.iter().enumerate() {
                        enc.setBytes_length_atIndex(
                            &v as *const i32 as *const std::ffi::c_void,
                            4,
                            5 + i,
                        );
                    }
                }

                let threads = MTLSize::new(total_d as u64, 1, 1);
                let groups = MTLSize::new((total_d + 255) / 256, 1, 1);
                enc.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
                enc.endEncoding();

                return Ok((out_storage, Box::new(MetalHandle { command_buffer: cmd })));
            } else {
                Err(crate::error::Error::Backend("MetalDevice inner is None".into()))
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (q, k, v, beta, state, d_k, d_v, num_heads, out_shape);
            Err(Error::Unimplemented(
                "delta_rule_decode requires a Metal device".into(),
            ))
        }
    }

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
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                use grim_tensor::BackendStorage;
                let k_s = k.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("rwkv_wkv_recurrence: k not MetalStorage".into())
                })?;
                let v_s = v.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("rwkv_wkv_recurrence: v not MetalStorage".into())
                })?;
                let r_s = r.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("rwkv_wkv_recurrence: r not MetalStorage".into())
                })?;
                let tf_s = time_first.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("rwkv_wkv_recurrence: time_first not MetalStorage".into())
                })?;
                let td_s = time_decay.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("rwkv_wkv_recurrence: time_decay not MetalStorage".into())
                })?;
                let aa_s = state_aa.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("rwkv_wkv_recurrence: state_aa not MetalStorage".into())
                })?;
                let bb_s = state_bb.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("rwkv_wkv_recurrence: state_bb not MetalStorage".into())
                })?;
                let pp_s = state_pp.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("rwkv_wkv_recurrence: state_pp not MetalStorage".into())
                })?;

                let k_buf = k_s.buffer.as_ref().ok_or_else(|| {
                    Error::Backend("rwkv_wkv_recurrence: k buffer None".into())
                })?;
                let v_buf = v_s.buffer.as_ref().ok_or_else(|| {
                    Error::Backend("rwkv_wkv_recurrence: v buffer None".into())
                })?;
                let r_buf = r_s.buffer.as_ref().ok_or_else(|| {
                    Error::Backend("rwkv_wkv_recurrence: r buffer None".into())
                })?;
                let tf_buf = tf_s.buffer.as_ref().ok_or_else(|| {
                    Error::Backend("rwkv_wkv_recurrence: time_first buffer None".into())
                })?;
                let td_buf = td_s.buffer.as_ref().ok_or_else(|| {
                    Error::Backend("rwkv_wkv_recurrence: time_decay buffer None".into())
                })?;
                let aa_buf = aa_s.buffer.as_ref().ok_or_else(|| {
                    Error::Backend("rwkv_wkv_recurrence: state_aa buffer None".into())
                })?;
                let bb_buf = bb_s.buffer.as_ref().ok_or_else(|| {
                    Error::Backend("rwkv_wkv_recurrence: state_bb buffer None".into())
                })?;
                let pp_buf = pp_s.buffer.as_ref().ok_or_else(|| {
                    Error::Backend("rwkv_wkv_recurrence: state_pp buffer None".into())
                })?;

                let out_storage = self.zeros(out_shape, k.dtype())?;
                let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("rwkv_wkv_recurrence: out allocation failed".into())
                })?;
                let out_buf = out_s.buffer.as_ref().ok_or_else(|| {
                    Error::Backend("rwkv_wkv_recurrence: out buffer None".into())
                })?;

                let ctx = inner.get_or_create_context()?;
                let cmd = ctx.get_or_create_command_buffer()?;
                let enc = cmd.computeCommandEncoder().ok_or_else(|| {
                    Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                })?;

                enc.setComputePipelineState(&ctx.pipelines.rwkv_wkv_recurrence);
                enc.setBuffer_offset_atIndex(Some(out_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(aa_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(bb_buf), 0, 2);
                enc.setBuffer_offset_atIndex(Some(pp_buf), 0, 3);
                enc.setBuffer_offset_atIndex(Some(tf_buf), 0, 4);
                enc.setBuffer_offset_atIndex(Some(td_buf), 0, 5);
                enc.setBuffer_offset_atIndex(Some(k_buf), 0, 6);
                enc.setBuffer_offset_atIndex(Some(v_buf), 0, 7);
                enc.setBuffer_offset_atIndex(Some(r_buf), 0, 8);

                let dim_i = dim as i32;
                unsafe {
                    enc.setBytes_length_atIndex(
                        &dim_i as *const i32 as *const std::ffi::c_void,
                        4,
                        9,
                    );
                }

                let threads = MTLSize::new(dim as u64, 1, 1);
                let groups = MTLSize::new((dim + 255) / 256, 1, 1);
                enc.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
                enc.endEncoding();

                return Ok((out_storage, Box::new(MetalHandle { command_buffer: cmd })));
            } else {
                Err(crate::error::Error::Backend("MetalDevice inner is None".into()))
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (k, v, r, time_first, time_decay, state_aa, state_bb, state_pp, dim, out_shape);
            Err(Error::Unimplemented(
                "rwkv_wkv_recurrence requires a Metal device".into(),
            ))
        }
    }

    fn rwkv_channel_mix_full(
        &self,
        x: &dyn BackendStorage,
        mix_k: &dyn BackendStorage,
        mix_r: &dyn BackendStorage,
        ffn_xx: &dyn BackendStorage,
        dim: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                use grim_tensor::BackendStorage;
                let x_s = x.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("rwkv_channel_mix_full: x not MetalStorage".into())
                })?;
                let mk_s = mix_k.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("rwkv_channel_mix_full: mix_k not MetalStorage".into())
                })?;
                let mr_s = mix_r.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("rwkv_channel_mix_full: mix_r not MetalStorage".into())
                })?;
                let ffn_s = ffn_xx.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("rwkv_channel_mix_full: ffn_xx not MetalStorage".into())
                })?;

                let x_buf = x_s.buffer.as_ref().ok_or_else(|| {
                    Error::Backend("rwkv_channel_mix_full: x buffer None".into())
                })?;
                let mk_buf = mk_s.buffer.as_ref().ok_or_else(|| {
                    Error::Backend("rwkv_channel_mix_full: mix_k buffer None".into())
                })?;
                let mr_buf = mr_s.buffer.as_ref().ok_or_else(|| {
                    Error::Backend("rwkv_channel_mix_full: mix_r buffer None".into())
                })?;
                let ffn_buf = ffn_s.buffer.as_ref().ok_or_else(|| {
                    Error::Backend("rwkv_channel_mix_full: ffn_xx buffer None".into())
                })?;

                let out_storage = self.zeros(out_shape, x.dtype())?;
                let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                    Error::Backend("rwkv_channel_mix_full: out allocation failed".into())
                })?;
                let out_buf = out_s.buffer.as_ref().ok_or_else(|| {
                    Error::Backend("rwkv_channel_mix_full: out buffer None".into())
                })?;

                let ctx = inner.get_or_create_context()?;
                let cmd = ctx.get_or_create_command_buffer()?;
                let enc = cmd.computeCommandEncoder().ok_or_else(|| {
                    Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                })?;

                enc.setComputePipelineState(&ctx.pipelines.rwkv_channel_mix_full);
                enc.setBuffer_offset_atIndex(Some(out_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(x_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(mk_buf), 0, 2);
                enc.setBuffer_offset_atIndex(Some(mr_buf), 0, 3);
                enc.setBuffer_offset_atIndex(Some(ffn_buf), 0, 4);

                let dim_i = dim as i32;
                unsafe {
                    enc.setBytes_length_atIndex(
                        &dim_i as *const i32 as *const std::ffi::c_void,
                        4,
                        5,
                    );
                }

                let threads = MTLSize::new(dim as u64, 1, 1);
                let groups = MTLSize::new((dim + 255) / 256, 1, 1);
                enc.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
                enc.endEncoding();

                return Ok((out_storage, Box::new(MetalHandle { command_buffer: cmd })));
            } else {
                Err(crate::error::Error::Backend("MetalDevice inner is None".into()))
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (x, mix_k, mix_r, ffn_xx, dim, out_shape);
            Err(Error::Unimplemented(
                "rwkv_channel_mix_full requires a Metal device".into(),
            ))
        }
    }
}
