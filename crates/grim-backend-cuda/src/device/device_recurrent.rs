//! Recurrent (SSM / RWKV / Conv1D / KDA) operations for `CudaDevice`.

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use grim_tensor::backend::ComputeHandle;
use grim_tensor::dtype::DType;
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendStorage, CoreTensorOps, RecurrentOps, Shape};

use crate::device::cuda_device::CudaDevice;
use crate::device::handles::CudaHandle;
use crate::memory::storage::CudaStorage;

impl RecurrentOps for CudaDevice {
    fn selective_scan(
        &self,
        x: &dyn BackendStorage,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        c: &dyn BackendStorage,
        d: &dyn BackendStorage,
        state: &dyn BackendStorage,
        batch: usize,
        dim_dstate: usize,
        dim_dinner: usize,
        seq_len: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        tracing::warn!("CUDA selective_scan: falling back to CPU execution");
        let x_v = x.to_cpu_vec_f32()?;
        let a_v = a.to_cpu_vec_f32()?;
        let b_v = b.to_cpu_vec_f32()?;
        let c_v = c.to_cpu_vec_f32()?;
        let d_v = d.to_cpu_vec_f32()?;
        let state_v = state.to_cpu_vec_f32()?;

        let mut out = vec![0.0f32; batch * seq_len * dim_dinner];
        for b_idx in 0..batch {
            for d_idx in 0..dim_dinner {
                // Initialize state from the provided state buffer.
                let mut h = vec![0.0f32; dim_dstate];
                for s in 0..dim_dstate {
                    let state_idx = (b_idx * dim_dinner + d_idx) * dim_dstate + s;
                    h[s] = if state_v.len() > state_idx {
                        state_v[state_idx]
                    } else {
                        0.0
                    };
                }
                let d_val = if d_v.len() > d_idx { d_v[d_idx] } else { 0.0 };

                for t in 0..seq_len {
                    let x_idx = (b_idx * seq_len + t) * dim_dinner + d_idx;
                    let x_t = x_v[x_idx];
                    let mut y_t = d_val * x_t;

                    for s in 0..dim_dstate {
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

                        h[s] = a_val * h[s] + x_t * b_val;
                        y_t += c_val * h[s];
                    }
                    out[x_idx] = y_t;
                }
            }
        }

        let out_storage = self.from_cpu(&out, out_shape, x.dtype())?;
        Ok((
            out_storage,
            Box::new(CudaHandle {
                completed: Arc::new(Mutex::new(true)),
            }),
        ))
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
        tracing::warn!("CUDA rwkv_time_mix: falling back to CPU execution");
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
        Ok((
            out_storage,
            Box::new(CudaHandle {
                completed: Arc::new(Mutex::new(true)),
            }),
        ))
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
        tracing::warn!("CUDA rwkv_channel_mix: falling back to CPU execution");
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
        Ok((
            out_storage,
            Box::new(CudaHandle {
                completed: Arc::new(Mutex::new(true)),
            }),
        ))
    }

    /// Depthwise 1D causal convolution step on CUDA GPU.
    /// Executes the causal convolution step kernel against resident GPU state buffers.
    fn short_conv1d_causal_step(
        &self,
        x: &dyn BackendStorage,
        weight: &dyn BackendStorage,
        bias: Option<&dyn BackendStorage>,
        conv_state: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = x
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("short_conv1d: x is not CudaStorage".into()))?;
        let w_s = weight
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("short_conv1d: weight is not CudaStorage".into()))?;
        let st_s = conv_state
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("short_conv1d: conv_state is not CudaStorage".into()))?;

        Self::ensure_f32_input("short_conv1d x", x_s)?;
        Self::ensure_f32_input("short_conv1d weight", w_s)?;
        Self::ensure_f32_input("short_conv1d conv_state", st_s)?;

        let mut x_ptr = Self::dev_ptr_or_err("short_conv1d x", x_s)?;
        let mut w_ptr = Self::dev_ptr_or_err("short_conv1d weight", w_s)?;
        let mut b_ptr = match bias {
            Some(b) => {
                let b_s = b.as_any().downcast_ref::<CudaStorage>().ok_or_else(|| {
                    Error::Backend("short_conv1d: bias is not CudaStorage".into())
                })?;
                Self::ensure_f32_input("short_conv1d bias", b_s)?;
                Self::dev_ptr_or_err("short_conv1d bias", b_s)?
            }
            None => std::ptr::null_mut(),
        };
        let mut st_ptr = Self::dev_ptr_or_err("short_conv1d conv_state", st_s)?;

        let out = CudaStorage::alloc_gpu(out_shape, DType::F32, self.ordinal)?;
        let mut out_ptr = Self::dev_ptr_or_err("short_conv1d out", &out)?;

        let dims = out_shape.dims();
        let mut batch = dims[0] as i32;
        let mut channels = *dims.last().unwrap_or(&1) as i32;
        let mut k_size = (w_s.bytes() / (channels as usize * 4)) as i32;
        let total = (batch * channels) as usize;

        let mut args = [
            &mut x_ptr as *mut *mut c_void as *mut c_void,
            &mut w_ptr as *mut *mut c_void as *mut c_void,
            &mut b_ptr as *mut *mut c_void as *mut c_void,
            &mut st_ptr as *mut *mut c_void as *mut c_void,
            &mut out_ptr as *mut *mut c_void as *mut c_void,
            &mut batch as *mut i32 as *mut c_void,
            &mut channels as *mut i32 as *mut c_void,
            &mut k_size as *mut i32 as *mut c_void,
        ];

        let handle = self.launch_rank1_kernel("grim_short_conv1d_causal_step", &mut args, total)?;
        Ok((Box::new(out), handle))
    }

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
        let q_s = q.as_any().downcast_ref::<CudaStorage>().ok_or_else(|| {
            Error::Backend("kda_gated_delta_rule_step: q is not CudaStorage".into())
        })?;
        let k_s = k.as_any().downcast_ref::<CudaStorage>().ok_or_else(|| {
            Error::Backend("kda_gated_delta_rule_step: k is not CudaStorage".into())
        })?;
        let v_s = v.as_any().downcast_ref::<CudaStorage>().ok_or_else(|| {
            Error::Backend("kda_gated_delta_rule_step: v is not CudaStorage".into())
        })?;
        let beta_s = beta.as_any().downcast_ref::<CudaStorage>().ok_or_else(|| {
            Error::Backend("kda_gated_delta_rule_step: beta is not CudaStorage".into())
        })?;
        let a_gate_s = a_gate
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| {
                Error::Backend("kda_gated_delta_rule_step: a_gate is not CudaStorage".into())
            })?;
        let state_s = recurrent_state
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| {
                Error::Backend(
                    "kda_gated_delta_rule_step: recurrent_state is not CudaStorage".into(),
                )
            })?;

        Self::ensure_f32_input("kda q", q_s)?;
        Self::ensure_f32_input("kda k", k_s)?;
        Self::ensure_f32_input("kda v", v_s)?;
        Self::ensure_f32_input("kda beta", beta_s)?;
        Self::ensure_f32_input("kda a_gate", a_gate_s)?;
        Self::ensure_f32_input("kda state", state_s)?;

        let out = CudaStorage::alloc_gpu(out_shape, DType::F32, self.ordinal)?;
        let mut q_ptr = Self::dev_ptr_or_err("kda q", q_s)?;
        let mut k_ptr = Self::dev_ptr_or_err("kda k", k_s)?;
        let mut v_ptr = Self::dev_ptr_or_err("kda v", v_s)?;
        let mut beta_ptr = Self::dev_ptr_or_err("kda beta", beta_s)?;
        let mut gate_ptr = Self::dev_ptr_or_err("kda a_gate", a_gate_s)?;
        let mut s_ptr = Self::dev_ptr_or_err("kda state", state_s)?;
        let mut out_ptr = Self::dev_ptr_or_err("kda out", &out)?;
        let mut dk_i = d_k as i32;
        let mut dv_i = d_v as i32;

        let mut args: [*mut c_void; 9] = [
            &mut q_ptr as *mut *mut c_void as *mut c_void,
            &mut k_ptr as *mut *mut c_void as *mut c_void,
            &mut v_ptr as *mut *mut c_void as *mut c_void,
            &mut beta_ptr as *mut *mut c_void as *mut c_void,
            &mut gate_ptr as *mut *mut c_void as *mut c_void,
            &mut s_ptr as *mut *mut c_void as *mut c_void,
            &mut out_ptr as *mut *mut c_void as *mut c_void,
            &mut dk_i as *mut i32 as *mut c_void,
            &mut dv_i as *mut i32 as *mut c_void,
        ];

        let handle = self.launch_rank1_kernel("grim_kda_gated_delta_rule_step", &mut args, d_v)?;
        Ok((Box::new(out), handle))
    }
}
