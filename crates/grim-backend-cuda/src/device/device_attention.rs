//! Attention operations for `CudaDevice`.

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use grim_tensor::backend::ComputeHandle;
use grim_tensor::dtype::DType;
use grim_tensor::error::{Error, Result};
use grim_tensor::{AttentionOps, BackendStorage, Shape};

use crate::device::cuda_device::CudaDevice;
use crate::device::handles::{
    cuLaunchKernel, cuModuleGetFunction, cudaDeviceSynchronize, cudaFree, cudaMalloc, cudaMemcpy,
    cudaMemcpyHostToDevice, cudaSuccess, CUfunction, CudaHandle,
};
use crate::device::jit_cache::compile_and_load_kernel;
use crate::memory::storage::CudaStorage;

impl CudaDevice {
    /// Fused QKV attention (Phase-1, mirrors `RocmDevice::qkv_attention`).
    ///
    /// Parameters (q: [S, H, D], k/v: [kv_S, kv_H, D], f32).
    /// Uses grim_qkv_attention kernel (online softmax, per-wave partials merged by wave-0).
    #[allow(clippy::too_many_arguments)]
    pub fn qkv_attention(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        window: Option<usize>,
        out: &Shape,
        out_max: Option<&dyn BackendStorage>,
        out_sum: Option<&dyn BackendStorage>,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let window_lo_i: i32 = match window {
            None => 0,
            Some(w) => {
                let abs_first = cache_offset as usize;
                abs_first.saturating_sub(w.saturating_sub(1)) as i32
            }
        };

        let out_dims = out.dims();
        let (seq_len, num_heads, head_dim) = if out_dims.len() == 3 {
            (out_dims[0], out_dims[1], out_dims[2])
        } else if out_dims.len() == 2 {
            let seq_len = out_dims[0];
            let hidden_dim = out_dims[1];
            let q_dims = q.shape().dims();
            let head_dim = if q_dims.len() == 3 {
                q_dims[2]
            } else if q_dims.len() == 2 && num_kv_heads > 0 {
                q_dims[1] / num_kv_heads
            } else {
                hidden_dim / num_kv_heads.max(1)
            };
            if head_dim == 0 {
                return Err(Error::Shape(
                    "qkv_attention head_dim resolved to zero; malformed model dimension".into(),
                ));
            }
            let num_heads = hidden_dim / head_dim;
            (seq_len, num_heads, head_dim)
        } else {
            return Err(Error::Shape(
                "qkv_attention expects 2-D [seq_len, hidden_dim] or 3-D [seq_len, num_heads, head_dim] output shape".into(),
            ));
        };
        if num_heads == 0 || num_kv_heads == 0 || head_dim == 0 {
            return Err(Error::Shape(
                "qkv_attention: zero-sized num_heads / num_kv_heads / head_dim".into(),
            ));
        }
        if num_heads % num_kv_heads != 0 {
            return Err(Error::Shape(format!(
                "qkv_attention: num_heads ({num_heads}) must be a multiple of num_kv_heads ({num_kv_heads})"
            )));
        }
        if head_dim > 256 {
            return Err(Error::Shape(format!(
                "qkv_attention: head_dim <= 256 supported (got {head_dim})"
            )));
        }

        let q_s = q
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("qkv_attention q is not CudaStorage".into()))?;
        let k_s = k
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("qkv_attention k is not CudaStorage".into()))?;
        let v_s = v
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("qkv_attention v is not CudaStorage".into()))?;
        Self::ensure_f32_input("qkv_attention q", q_s)?;
        Self::ensure_f32_input("qkv_attention k", k_s)?;
        Self::ensure_f32_input("qkv_attention v", v_s)?;

        let max_s = match out_max {
            Some(m) => Some(m.as_any().downcast_ref::<CudaStorage>().ok_or_else(|| {
                Error::Backend("qkv_attention out_max is not CudaStorage".into())
            })?),
            None => None,
        };
        let sum_s = match out_sum {
            Some(s) => Some(s.as_any().downcast_ref::<CudaStorage>().ok_or_else(|| {
                Error::Backend("qkv_attention out_sum is not CudaStorage".into())
            })?),
            None => None,
        };

        let out_storage = CudaStorage::alloc_gpu(out, DType::F32, self.ordinal)?;
        let inv_sqrt_d: f32 = 1.0 / (head_dim as f32).sqrt();

        let mut q_ptr = Self::dev_ptr_or_err("qkv_attention q", q_s)?;
        let mut k_ptr = Self::dev_ptr_or_err("qkv_attention k", k_s)?;
        let mut v_ptr = Self::dev_ptr_or_err("qkv_attention v", v_s)?;
        let mut out_ptr = Self::dev_ptr_or_err("qkv_attention out", &out_storage)?;
        let mut max_ptr: u64 = match max_s {
            Some(m) => m.device_ptr.unwrap_or(0),
            None => 0,
        };
        let mut sum_ptr: u64 = match sum_s {
            Some(s) => s.device_ptr.unwrap_or(0),
            None => 0,
        };
        let mut num_heads_i = num_heads as i32;
        let mut num_kv_heads_i = num_kv_heads as i32;
        let mut head_dim_i = head_dim as i32;
        let mut seq_len_i = seq_len as i32;
        let mut kv_seq_len_i = kv_seq_len as i32;
        let mut cache_offset_i = cache_offset as i32;
        let mut inv_sqrt_d_val = inv_sqrt_d;
        let mut window_lo_val = window_lo_i;
        let mut softcap_val: f32 = 0.0f32;
        let mut alibi_slopes_ptr: *mut c_void = std::ptr::null_mut();
        let mut has_alibi_val: i32 = 0;

        let mut args: [*mut c_void; 17] = [
            &mut q_ptr as *mut *mut c_void as *mut c_void,
            &mut k_ptr as *mut *mut c_void as *mut c_void,
            &mut v_ptr as *mut *mut c_void as *mut c_void,
            &mut out_ptr as *mut *mut c_void as *mut c_void,
            &mut max_ptr as *mut u64 as *mut c_void,
            &mut sum_ptr as *mut u64 as *mut c_void,
            &mut num_heads_i as *mut i32 as *mut c_void,
            &mut num_kv_heads_i as *mut i32 as *mut c_void,
            &mut head_dim_i as *mut i32 as *mut c_void,
            &mut seq_len_i as *mut i32 as *mut c_void,
            &mut kv_seq_len_i as *mut i32 as *mut c_void,
            &mut cache_offset_i as *mut i32 as *mut c_void,
            &mut inv_sqrt_d_val as *mut f32 as *mut c_void,
            &mut window_lo_val as *mut i32 as *mut c_void,
            &mut softcap_val as *mut f32 as *mut c_void,
            &mut alibi_slopes_ptr as *mut *mut c_void as *mut c_void,
            &mut has_alibi_val as *mut i32 as *mut c_void,
        ];

        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;

        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            let func_name = std::ffi::CString::new("grim_qkv_attention")
                .map_err(|e| Error::Backend(format!("invalid kernel name: {e}")))?;
            let res = cuModuleGetFunction(&mut func, module, func_name.as_ptr());
            if res != 0 {
                return Err(Error::Backend(format!(
                    "cuModuleGetFunction(grim_qkv_attention) failed: {res}"
                )));
            }
            let launch_res = cuLaunchKernel(
                func,
                seq_len as u32,
                num_heads as u32,
                1,
                256,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr() as *mut *mut c_void,
                std::ptr::null_mut(),
            );
            if launch_res != 0 {
                return Err(Error::Backend(format!(
                    "cuLaunchKernel(grim_qkv_attention) failed: {launch_res}"
                )));
            }
        }
        let compute_handle = Box::new(CudaHandle {
            completed: Arc::new(Mutex::new(false)),
        });
        Ok((Box::new(out_storage), compute_handle))
    }
}

impl AttentionOps for CudaDevice {
    fn qkv_attention(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        window: Option<usize>,
        out_shape: &Shape,
        out_max: Option<&dyn BackendStorage>,
        out_sum: Option<&dyn BackendStorage>,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        CudaDevice::qkv_attention(
            self,
            q,
            k,
            v,
            num_kv_heads,
            kv_seq_len,
            cache_offset,
            window,
            out_shape,
            out_max,
            out_sum,
        )
    }

    fn qkv_attention_paged(
        &self,
        q: &dyn BackendStorage,
        block_tables: &dyn BackendStorage,
        k_pages: &dyn BackendStorage,
        v_pages: &dyn BackendStorage,
        num_kv_heads: usize,
        max_blocks: usize,
        page_size: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        window: Option<usize>,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let q_s = q
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("qkv_attention_paged: q is not CudaStorage".into()))?;
        let bt_s = block_tables
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("qkv_attention_paged: block_tables is not CudaStorage".into()))?;
        let k_s = k_pages
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("qkv_attention_paged: k_pages is not CudaStorage".into()))?;
        let v_s = v_pages
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("qkv_attention_paged: v_pages is not CudaStorage".into()))?;

        if q_s.device_ptr.is_none()
            || bt_s.device_ptr.is_none()
            || k_s.device_ptr.is_none()
            || v_s.device_ptr.is_none()
        {
            return Err(Error::Backend(
                "qkv_attention_paged: inputs lack a valid device pointer".into(),
            ));
        }

        let out_dims = out_shape.dims();
        let (batch, num_heads, head_dim) = if out_dims.len() == 3 {
            (out_dims[0], out_dims[1], out_dims[2])
        } else if out_dims.len() == 2 {
            (1, out_dims[0], out_dims[1])
        } else {
            return Err(Error::Shape(format!(
                "qkv_attention_paged: unsupported out_shape {out_dims:?}"
            )));
        };

        let window_lo_i: i32 = match window {
            None => 0,
            Some(w) => (cache_offset as usize).saturating_sub(w.saturating_sub(1)) as i32,
        };

        let out_storage = CudaStorage::alloc_gpu(out_shape, DType::F32, self.ordinal)?;
        let inv_sqrt_d = 1.0f32 / (head_dim as f32).sqrt();

        let mut q_ptr = Self::dev_ptr_or_err("qkv_attention_paged q", q_s)?;
        let mut bt_ptr = Self::dev_ptr_or_err("qkv_attention_paged bt", bt_s)?;
        let mut k_ptr = Self::dev_ptr_or_err("qkv_attention_paged k", k_s)?;
        let mut v_ptr = Self::dev_ptr_or_err("qkv_attention_paged v", v_s)?;
        let mut out_ptr = Self::dev_ptr_or_err("qkv_attention_paged out", &out_storage)?;

        let mut num_heads_i = num_heads as i32;
        let mut num_kv_heads_i = num_kv_heads as i32;
        let mut head_dim_i = head_dim as i32;
        let mut max_blocks_i = max_blocks as i32;
        let mut page_size_i = page_size as i32;
        let mut kv_seq_len_i = kv_seq_len as i32;
        let mut cache_offset_i = cache_offset as i32;
        let mut inv_sqrt_d_val = inv_sqrt_d;
        let mut window_lo_val = window_lo_i;

        let mut args: [*mut c_void; 14] = [
            &mut q_ptr as *mut *mut c_void as *mut c_void,
            &mut bt_ptr as *mut *mut c_void as *mut c_void,
            &mut k_ptr as *mut *mut c_void as *mut c_void,
            &mut v_ptr as *mut *mut c_void as *mut c_void,
            &mut out_ptr as *mut *mut c_void as *mut c_void,
            &mut num_heads_i as *mut i32 as *mut c_void,
            &mut num_kv_heads_i as *mut i32 as *mut c_void,
            &mut head_dim_i as *mut i32 as *mut c_void,
            &mut max_blocks_i as *mut i32 as *mut c_void,
            &mut page_size_i as *mut i32 as *mut c_void,
            &mut kv_seq_len_i as *mut i32 as *mut c_void,
            &mut cache_offset_i as *mut i32 as *mut c_void,
            &mut inv_sqrt_d_val as *mut f32 as *mut c_void,
            &mut window_lo_val as *mut i32 as *mut c_void,
        ];

        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            let func_name = std::ffi::CString::new("grim_qkv_attention_paged")
                .map_err(|e| Error::Backend(format!("invalid kernel name: {e}")))?;
            let res = cuModuleGetFunction(&mut func, module, func_name.as_ptr());
            if res != 0 {
                return Err(Error::Backend(format!(
                    "cuModuleGetFunction(grim_qkv_attention_paged) failed: {res}"
                )));
            }
            let launch_res = cuLaunchKernel(
                func,
                batch as u32,
                num_heads as u32,
                1,
                256,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr() as *mut *mut c_void,
                std::ptr::null_mut(),
            );
            if launch_res != 0 {
                return Err(Error::Backend(format!(
                    "cuLaunchKernel(grim_qkv_attention_paged) failed: {launch_res}"
                )));
            }
        }

        Ok((Box::new(out_storage), Box::new(CudaHandle::ready(self.ordinal))))
    }

    fn rope(
        &self,
        x: &dyn BackendStorage,
        positions: &[u32],
        cfg: &grim_tensor::RopeConfig,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let dim = cfg.dim;
        let base = cfg.base;
        let x_storage = x
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("rope x is not CudaStorage".into()))?;
        let out_storage = CudaStorage::alloc_gpu(out_shape, DType::F32, self.ordinal)?;

        if !cfg.is_plain() {
            let dims = out_shape.dims();
            if dims.len() != 3 || dims[2] != cfg.dim {
                return Err(Error::Shape(format!(
                    "rope (yarn): expected [B,S,D={}] out_shape, got {:?}",
                    cfg.dim, dims
                )));
            }
            let (b, s, d) = (dims[0], dims[1], dims[2]);
            let rotary_dim = cfg.rotary_dim.min(d);
            let rotary_half = rotary_dim / 2;
            let yarn = cfg.yarn;
            if positions.len() != s {
                return Err(Error::Shape(
                    "rope (yarn): positions length must match seq_len".into(),
                ));
            }

            let inv_freq: Vec<f32> = (0..rotary_half)
                .map(|i| {
                    let freq = 1.0_f32 / base.powf((2 * i) as f32 / d as f32);
                    match yarn {
                        None => freq,
                        Some(y) => {
                            let wavelength = 2.0 * std::f32::consts::PI / freq;
                            let low = y.original_max_pos as f32 / y.beta_slow;
                            let high = y.original_max_pos as f32 / y.beta_fast;
                            if wavelength < high {
                                freq
                            } else if wavelength > low {
                                freq / y.factor
                            } else {
                                let ramp = (y.original_max_pos as f32 / wavelength - y.beta_slow)
                                    / (y.beta_fast - y.beta_slow);
                                (1.0 - ramp) * (freq / y.factor) + ramp * freq
                            }
                        }
                    }
                })
                .collect();
            let mscale = yarn.map(|y| y.attention_factor).unwrap_or(1.0_f32);

            let pos_bytes = positions.len() * 4;
            let freq_bytes = inv_freq.len() * 4;
            let mut pos_dev_ptr: *mut c_void = std::ptr::null_mut();
            let mut freq_dev_ptr: *mut c_void = std::ptr::null_mut();
            unsafe {
                let res = cudaMalloc(&mut pos_dev_ptr, pos_bytes);
                if res != cudaSuccess {
                    return Err(Error::Backend(format!(
                        "cudaMalloc for rope yarn pos failed: {}",
                        res
                    )));
                }
                let res = cudaMemcpy(
                    pos_dev_ptr,
                    positions.as_ptr() as *const c_void,
                    pos_bytes,
                    cudaMemcpyHostToDevice,
                );
                if res != cudaSuccess {
                    cudaFree(pos_dev_ptr);
                    return Err(Error::Backend(format!(
                        "cudaMemcpy for rope yarn pos failed: {}",
                        res
                    )));
                }
                let res = cudaMalloc(&mut freq_dev_ptr, freq_bytes);
                if res != cudaSuccess {
                    cudaFree(pos_dev_ptr);
                    return Err(Error::Backend(format!(
                        "cudaMalloc for rope yarn inv_freq failed: {}",
                        res
                    )));
                }
                let res = cudaMemcpy(
                    freq_dev_ptr,
                    inv_freq.as_ptr() as *const c_void,
                    freq_bytes,
                    cudaMemcpyHostToDevice,
                );
                if res != cudaSuccess {
                    cudaFree(pos_dev_ptr);
                    cudaFree(freq_dev_ptr);
                    return Err(Error::Backend(format!(
                        "cudaMemcpy for rope yarn inv_freq failed: {}",
                        res
                    )));
                }
            }

            let mut x_ptr = Self::dev_ptr_or_err("rope yarn x", x_storage)?;
            let mut out_ptr = Self::dev_ptr_or_err("rope yarn out", &out_storage)?;
            let mut pos_ptr = pos_dev_ptr;
            let mut freq_ptr = freq_dev_ptr;
            let mut b_i = b as i32;
            let mut s_i = s as i32;
            let mut d_i = d as i32;
            let mut rh_i = rotary_half as i32;
            let mut ms_f = mscale;

            let mut args = [
                &mut x_ptr as *mut *mut c_void as *mut c_void,
                &mut pos_ptr as *mut *mut c_void as *mut c_void,
                &mut freq_ptr as *mut *mut c_void as *mut c_void,
                &mut out_ptr as *mut *mut c_void as *mut c_void,
                &mut b_i as *mut i32 as *mut c_void,
                &mut s_i as *mut i32 as *mut c_void,
                &mut d_i as *mut i32 as *mut c_void,
                &mut rh_i as *mut i32 as *mut c_void,
                &mut ms_f as *mut f32 as *mut c_void,
            ];

            let copy_len = d - 2 * rotary_half;
            let total = b
                * s
                * rotary_half
                    .max(if copy_len > 0 { copy_len } else { 0 })
                    .max(1);
            let handle = self.launch_rank1_kernel("grim_rope_yarn", &mut args, total)?;

            unsafe {
                let _ = cudaDeviceSynchronize();
                cudaFree(pos_dev_ptr);
                cudaFree(freq_dev_ptr);
            }

            return Ok((Box::new(out_storage), handle));
        }

        // Plain full-rotary RoPE (cfg.is_plain()).
        let num_tokens = positions.len();
        let num_heads = out_shape.elem_count() / (num_tokens * dim);
        let head_dim = dim;

        let pos_i32: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
        let pos_bytes = pos_i32.len() * 4;
        let mut pos_dev_ptr: *mut c_void = std::ptr::null_mut();
        unsafe {
            let res = cudaMalloc(&mut pos_dev_ptr, pos_bytes);
            if res != cudaSuccess {
                return Err(Error::Backend(format!(
                    "cudaMalloc for rope pos failed: {}",
                    res
                )));
            }
            let res = cudaMemcpy(
                pos_dev_ptr,
                pos_i32.as_ptr() as *const c_void,
                pos_bytes,
                cudaMemcpyHostToDevice,
            );
            if res != cudaSuccess {
                return Err(Error::Backend(format!(
                    "cudaMemcpy for rope pos failed: {}",
                    res
                )));
            }
        }

        let mut x_ptr = Self::dev_ptr_or_err("rope x", x_storage)?;
        let mut out_ptr = Self::dev_ptr_or_err("rope out", &out_storage)?;
        let mut num_t_i = num_tokens as i32;
        let mut num_h_i = num_heads as i32;
        let mut h_dim_i = head_dim as i32;
        let mut base_f = base;

        let mut args = [
            &mut x_ptr as *mut *mut c_void as *mut c_void,
            &mut pos_dev_ptr as *mut *mut c_void as *mut c_void,
            &mut out_ptr as *mut *mut c_void as *mut c_void,
            &mut num_t_i as *mut i32 as *mut c_void,
            &mut num_h_i as *mut i32 as *mut c_void,
            &mut h_dim_i as *mut i32 as *mut c_void,
            &mut base_f as *mut f32 as *mut c_void,
        ];

        let total_pairs = num_tokens * num_heads * (head_dim / 2);
        let handle = self.launch_rank1_kernel("grim_rope", &mut args, total_pairs)?;

        unsafe {
            let _ = cudaDeviceSynchronize();
            cudaFree(pos_dev_ptr);
        }

        Ok((Box::new(out_storage), handle))
    }

    fn flash_attention(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
        _causal: bool,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let (out_storage, _h) = self.qkv_attention(
            q,
            k,
            v,
            num_kv_heads,
            seq_len,
            0,
            None,
            out_shape,
            None,
            None,
        )?;
        let _ = num_heads;
        let _ = head_dim;
        tracing::debug!(
            target: "grim_cuda",
            seq_len = seq_len,
            num_heads = num_heads,
            "CUDA flash_attention → qkv_attention (op=Attention)"
        );
        Ok((
            out_storage,
            Box::new(CudaHandle {
                completed: Arc::new(Mutex::new(true)),
            }),
        ))
    }

    fn cross_attention(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
        kv_seq_len: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let (out_storage, _h) = self.qkv_attention(
            q, k, v, num_heads, kv_seq_len, 0, None, out_shape, None, None,
        )?;
        tracing::debug!(
            target: "grim_cuda",
            seq_len = seq_len,
            kv_seq_len = kv_seq_len,
            num_heads = num_heads,
            "CUDA cross_attention → qkv_attention (op=Attention)"
        );
        let _ = head_dim;
        let _ = seq_len;
        Ok((
            out_storage,
            Box::new(CudaHandle {
                completed: Arc::new(Mutex::new(true)),
            }),
        ))
    }

    fn sage_attention(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let q_s = q
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("sage_attention: q is not CudaStorage".into()))?;
        let k_s = k
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("sage_attention: k is not CudaStorage".into()))?;
        let v_s = v
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("sage_attention: v is not CudaStorage".into()))?;

        let out_dims = out_shape.dims();
        let (seq_len, num_heads, head_dim) = if out_dims.len() == 3 {
            (out_dims[0], out_dims[1], out_dims[2])
        } else if out_dims.len() == 2 {
            let seq_len = out_dims[0];
            let hidden_dim = out_dims[1];
            let head_dim = hidden_dim / num_kv_heads.max(1);
            let num_heads = hidden_dim / head_dim.max(1);
            (seq_len, num_heads, head_dim)
        } else {
            return Err(Error::Shape(format!(
                "sage_attention: unsupported out_shape {out_dims:?}"
            )));
        };

        let out_storage = CudaStorage::alloc_gpu(out_shape, DType::F32, self.ordinal)?;
        let sm_scale = 1.0f32 / (head_dim as f32).sqrt();

        let mut q_ptr = Self::dev_ptr_or_err("sage_attention q", q_s)?;
        let mut k_ptr = Self::dev_ptr_or_err("sage_attention k", k_s)?;
        let mut v_ptr = Self::dev_ptr_or_err("sage_attention v", v_s)?;
        let mut out_ptr = Self::dev_ptr_or_err("sage_attention out", &out_storage)?;

        let mut num_heads_i = num_heads as i32;
        let mut num_kv_heads_i = num_kv_heads as i32;
        let mut head_dim_i = head_dim as i32;
        let mut seq_len_i = seq_len as i32;
        let mut kv_seq_len_i = kv_seq_len as i32;
        let mut sm_scale_val = sm_scale;

        let mut args: [*mut c_void; 10] = [
            &mut q_ptr as *mut *mut c_void as *mut c_void,
            &mut k_ptr as *mut *mut c_void as *mut c_void,
            &mut v_ptr as *mut *mut c_void as *mut c_void,
            &mut out_ptr as *mut *mut c_void as *mut c_void,
            &mut num_heads_i as *mut i32 as *mut c_void,
            &mut num_kv_heads_i as *mut i32 as *mut c_void,
            &mut head_dim_i as *mut i32 as *mut c_void,
            &mut seq_len_i as *mut i32 as *mut c_void,
            &mut kv_seq_len_i as *mut i32 as *mut c_void,
            &mut sm_scale_val as *mut f32 as *mut c_void,
        ];

        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            let func_name = std::ffi::CString::new("grim_sage_attention")
                .map_err(|e| Error::Backend(format!("invalid kernel name: {e}")))?;
            let res = cuModuleGetFunction(&mut func, module, func_name.as_ptr());
            if res != 0 {
                return Err(Error::Backend(format!(
                    "cuModuleGetFunction(grim_sage_attention) failed: {res}"
                )));
            }
            let block_size = 256u32;
            let grid_y = ((seq_len as u32) + block_size - 1) / block_size;
            let launch_res = cuLaunchKernel(
                func,
                num_heads as u32,
                grid_y.max(1),
                1,
                block_size,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr() as *mut *mut c_void,
                std::ptr::null_mut(),
            );
            if launch_res != 0 {
                return Err(Error::Backend(format!(
                    "cuLaunchKernel(grim_sage_attention) failed: {launch_res}"
                )));
            }
        }

        Ok((Box::new(out_storage), Box::new(CudaHandle::ready(self.ordinal))))
    }

    fn qkv_attention_alibi(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        window: Option<usize>,
        alibi_slopes: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let q_s = q
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("qkv_attention_alibi: q is not CudaStorage".into()))?;
        let k_s = k
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("qkv_attention_alibi: k is not CudaStorage".into()))?;
        let v_s = v
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("qkv_attention_alibi: v is not CudaStorage".into()))?;
        let slopes_s = alibi_slopes
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("qkv_attention_alibi: alibi_slopes is not CudaStorage".into()))?;

        let out_dims = out_shape.dims();
        if out_dims.len() != 3 {
            return Err(Error::Shape(
                "qkv_attention_alibi: out_shape must be [seq, heads, head_dim]".into(),
            ));
        }
        let (seq_len, num_heads, head_dim) = (out_dims[0], out_dims[1], out_dims[2]);

        let window_lo_i: i32 = match window {
            None => 0,
            Some(w) => (cache_offset as usize).saturating_sub(w.saturating_sub(1)) as i32,
        };

        let out_storage = CudaStorage::alloc_gpu(out_shape, DType::F32, self.ordinal)?;
        let inv_sqrt_d = 1.0f32 / (head_dim as f32).sqrt();

        let mut q_ptr = Self::dev_ptr_or_err("qkv_attention_alibi q", q_s)?;
        let mut k_ptr = Self::dev_ptr_or_err("qkv_attention_alibi k", k_s)?;
        let mut v_ptr = Self::dev_ptr_or_err("qkv_attention_alibi v", v_s)?;
        let mut slopes_ptr = Self::dev_ptr_or_err("qkv_attention_alibi slopes", slopes_s)?;
        let mut out_ptr = Self::dev_ptr_or_err("qkv_attention_alibi out", &out_storage)?;
        let mut max_ptr: u64 = 0;
        let mut sum_ptr: u64 = 0;

        let mut num_heads_i = num_heads as i32;
        let mut num_kv_heads_i = num_kv_heads as i32;
        let mut head_dim_i = head_dim as i32;
        let mut seq_len_i = seq_len as i32;
        let mut kv_seq_len_i = kv_seq_len as i32;
        let mut cache_offset_i = cache_offset as i32;
        let mut inv_sqrt_d_val = inv_sqrt_d;
        let mut window_lo_val = window_lo_i;
        let mut softcap_val: f32 = 0.0f32;
        let mut has_alibi_val: i32 = 1;

        let mut args: [*mut c_void; 17] = [
            &mut q_ptr as *mut *mut c_void as *mut c_void,
            &mut k_ptr as *mut *mut c_void as *mut c_void,
            &mut v_ptr as *mut *mut c_void as *mut c_void,
            &mut out_ptr as *mut *mut c_void as *mut c_void,
            &mut max_ptr as *mut u64 as *mut c_void,
            &mut sum_ptr as *mut u64 as *mut c_void,
            &mut num_heads_i as *mut i32 as *mut c_void,
            &mut num_kv_heads_i as *mut i32 as *mut c_void,
            &mut head_dim_i as *mut i32 as *mut c_void,
            &mut seq_len_i as *mut i32 as *mut c_void,
            &mut kv_seq_len_i as *mut i32 as *mut c_void,
            &mut cache_offset_i as *mut i32 as *mut c_void,
            &mut inv_sqrt_d_val as *mut f32 as *mut c_void,
            &mut window_lo_val as *mut i32 as *mut c_void,
            &mut softcap_val as *mut f32 as *mut c_void,
            &mut slopes_ptr as *mut *mut c_void as *mut c_void,
            &mut has_alibi_val as *mut i32 as *mut c_void,
        ];

        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            let func_name = std::ffi::CString::new("grim_qkv_attention")
                .map_err(|e| Error::Backend(format!("invalid kernel name: {e}")))?;
            let res = cuModuleGetFunction(&mut func, module, func_name.as_ptr());
            if res != 0 {
                return Err(Error::Backend(format!(
                    "cuModuleGetFunction(grim_qkv_attention) failed: {res}"
                )));
            }
            let launch_res = cuLaunchKernel(
                func,
                seq_len as u32,
                num_heads as u32,
                1,
                256,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr() as *mut *mut c_void,
                std::ptr::null_mut(),
            );
            if launch_res != 0 {
                return Err(Error::Backend(format!(
                    "cuLaunchKernel(grim_qkv_attention) failed: {launch_res}"
                )));
            }
        }

        Ok((Box::new(out_storage), Box::new(CudaHandle::ready(self.ordinal))))
    }

    fn kv_dequant_attention(
        &self,
        q: &dyn BackendStorage,
        k_tensor: &dyn BackendStorage,
        k_scales: &dyn BackendStorage,
        v_tensor: &dyn BackendStorage,
        v_scales: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        quant_bits: u32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let q_s = q
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("kv_dequant_attention: q is not CudaStorage".into()))?;
        let k_s = k_tensor
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("kv_dequant_attention: k_tensor is not CudaStorage".into()))?;
        let ks_s = k_scales
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("kv_dequant_attention: k_scales is not CudaStorage".into()))?;
        let v_s = v_tensor
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("kv_dequant_attention: v_tensor is not CudaStorage".into()))?;
        let vs_s = v_scales
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("kv_dequant_attention: v_scales is not CudaStorage".into()))?;

        let out_dims = out_shape.dims();
        let (seq_len, num_heads, head_dim) = if out_dims.len() == 3 {
            (out_dims[0], out_dims[1], out_dims[2])
        } else {
            return Err(Error::Shape("kv_dequant_attention: out_shape must be 3-D".into()));
        };

        let out_storage = CudaStorage::alloc_gpu(out_shape, DType::F32, self.ordinal)?;
        let inv_sqrt_d = 1.0f32 / (head_dim as f32).sqrt();

        let mut q_ptr = Self::dev_ptr_or_err("kv_dequant_attention q", q_s)?;
        let mut k_ptr = Self::dev_ptr_or_err("kv_dequant_attention k", k_s)?;
        let mut ks_ptr = Self::dev_ptr_or_err("kv_dequant_attention ks", ks_s)?;
        let mut v_ptr = Self::dev_ptr_or_err("kv_dequant_attention v", v_s)?;
        let mut vs_ptr = Self::dev_ptr_or_err("kv_dequant_attention vs", vs_s)?;
        let mut out_ptr = Self::dev_ptr_or_err("kv_dequant_attention out", &out_storage)?;

        let mut num_heads_i = num_heads as i32;
        let mut num_kv_heads_i = num_kv_heads as i32;
        let mut head_dim_i = head_dim as i32;
        let mut seq_len_i = seq_len as i32;
        let mut kv_seq_len_i = kv_seq_len as i32;
        let mut cache_offset_i = cache_offset as i32;
        let mut inv_sqrt_d_val = inv_sqrt_d;
        let mut quant_bits_i = quant_bits as i32;
        let mut quant_format_i = if quant_bits == 8 { 1i32 } else { 2i32 };

        let mut args: [*mut c_void; 15] = [
            &mut q_ptr as *mut *mut c_void as *mut c_void,
            &mut k_ptr as *mut *mut c_void as *mut c_void,
            &mut ks_ptr as *mut *mut c_void as *mut c_void,
            &mut v_ptr as *mut *mut c_void as *mut c_void,
            &mut vs_ptr as *mut *mut c_void as *mut c_void,
            &mut out_ptr as *mut *mut c_void as *mut c_void,
            &mut num_heads_i as *mut i32 as *mut c_void,
            &mut num_kv_heads_i as *mut i32 as *mut c_void,
            &mut head_dim_i as *mut i32 as *mut c_void,
            &mut seq_len_i as *mut i32 as *mut c_void,
            &mut kv_seq_len_i as *mut i32 as *mut c_void,
            &mut cache_offset_i as *mut i32 as *mut c_void,
            &mut inv_sqrt_d_val as *mut f32 as *mut c_void,
            &mut quant_bits_i as *mut i32 as *mut c_void,
            &mut quant_format_i as *mut i32 as *mut c_void,
        ];

        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            let func_name = std::ffi::CString::new("grim_kv_dequant_attention")
                .map_err(|e| Error::Backend(format!("invalid kernel name: {e}")))?;
            let res = cuModuleGetFunction(&mut func, module, func_name.as_ptr());
            if res != 0 {
                return Err(Error::Backend(format!(
                    "cuModuleGetFunction(grim_kv_dequant_attention) failed: {res}"
                )));
            }
            let total_queries = (seq_len * num_heads) as u32;
            let launch_res = cuLaunchKernel(
                func,
                total_queries,
                1,
                1,
                256,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr() as *mut *mut c_void,
                std::ptr::null_mut(),
            );
            if launch_res != 0 {
                return Err(Error::Backend(format!(
                    "cuLaunchKernel(grim_kv_dequant_attention) failed: {launch_res}"
                )));
            }
        }

        Ok((Box::new(out_storage), Box::new(CudaHandle::ready(self.ordinal))))
    }

    fn mla_absorbed_decode(
        &self,
        q_absorbed: &dyn BackendStorage,
        q_rope: &dyn BackendStorage,
        kv_cache: &dyn BackendStorage,
        w_uv: Option<&dyn BackendStorage>,
        out: &dyn BackendStorage,
        num_heads: usize,
        kv_lora_rank: usize,
        qk_rope_dim: usize,
        v_head_dim: usize,
        seq_len: usize,
        w_uv_offset_words: usize,
        w_uv_head_stride_words: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let q_abs = q_absorbed
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("mla_absorbed_decode: q_absorbed is not CudaStorage".into()))?;
        let q_r = q_rope
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("mla_absorbed_decode: q_rope is not CudaStorage".into()))?;
        let kv = kv_cache
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("mla_absorbed_decode: kv_cache is not CudaStorage".into()))?;
        let o = out
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("mla_absorbed_decode: out is not CudaStorage".into()))?;

        let mut qabs_ptr = Self::dev_ptr_or_err("mla_absorbed_decode q_abs", q_abs)?;
        let mut qrope_ptr = Self::dev_ptr_or_err("mla_absorbed_decode q_rope", q_r)?;
        let mut kv_ptr = Self::dev_ptr_or_err("mla_absorbed_decode kv", kv)?;
        let mut out_ptr = Self::dev_ptr_or_err("mla_absorbed_decode out", o)?;

        let (mut wuv_ptr, has_w_uv_val) = match w_uv {
            Some(w) => {
                let ws = w
                    .as_any()
                    .downcast_ref::<CudaStorage>()
                    .ok_or_else(|| Error::Backend("mla_absorbed_decode: w_uv is not CudaStorage".into()))?;
                (Self::dev_ptr_or_err("mla_absorbed_decode w_uv", ws)?, 1i32)
            }
            None => (std::ptr::null_mut(), 0i32),
        };

        let mut nh = num_heads as i32;
        let mut lora_r = kv_lora_rank as i32;
        let mut rope_d = qk_rope_dim as i32;
        let mut v_dim = v_head_dim as i32;
        let mut slen = seq_len as i32;
        let mut inv_sqrt = 1.0f32 / ((kv_lora_rank + qk_rope_dim) as f32).sqrt();
        let mut has_w = has_w_uv_val;
        let mut w_off = w_uv_offset_words as i32;
        let mut w_stride = w_uv_head_stride_words as i32;

        let mut args: [*mut c_void; 14] = [
            &mut qabs_ptr as *mut *mut c_void as *mut c_void,
            &mut qrope_ptr as *mut *mut c_void as *mut c_void,
            &mut kv_ptr as *mut *mut c_void as *mut c_void,
            &mut wuv_ptr as *mut *mut c_void as *mut c_void,
            &mut out_ptr as *mut *mut c_void as *mut c_void,
            &mut nh as *mut i32 as *mut c_void,
            &mut lora_r as *mut i32 as *mut c_void,
            &mut rope_d as *mut i32 as *mut c_void,
            &mut v_dim as *mut i32 as *mut c_void,
            &mut slen as *mut i32 as *mut c_void,
            &mut inv_sqrt as *mut f32 as *mut c_void,
            &mut has_w as *mut i32 as *mut c_void,
            &mut w_off as *mut i32 as *mut c_void,
            &mut w_stride as *mut i32 as *mut c_void,
        ];

        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            let func_name = std::ffi::CString::new("grim_mla_absorbed_decode")
                .map_err(|e| Error::Backend(format!("invalid kernel name: {e}")))?;
            let res = cuModuleGetFunction(&mut func, module, func_name.as_ptr());
            if res != 0 {
                return Err(Error::Backend(format!(
                    "cuModuleGetFunction(grim_mla_absorbed_decode) failed: {res}"
                )));
            }
            let shared_bytes = (256 * std::mem::size_of::<f32>()) as u32;
            let launch_res = cuLaunchKernel(
                func,
                num_heads as u32,
                1,
                1,
                256,
                1,
                1,
                shared_bytes,
                std::ptr::null_mut(),
                args.as_mut_ptr() as *mut *mut c_void,
                std::ptr::null_mut(),
            );
            if launch_res != 0 {
                return Err(Error::Backend(format!(
                    "cuLaunchKernel(grim_mla_absorbed_decode) failed: {launch_res}"
                )));
            }
        }

        Ok(Box::new(CudaHandle::ready(self.ordinal)))
    }

    fn mla_q_kv_norm_split(
        &self,
        q_raw: &dyn BackendStorage,
        kv_raw: &dyn BackendStorage,
        q_norm_w: &dyn BackendStorage,
        kv_norm_w: &dyn BackendStorage,
        qk_nope_dim: usize,
        qk_rope_dim: usize,
        v_dim: usize,
        eps: f32,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        let q_s = q_raw
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("mla_q_kv_norm_split: q_raw is not CudaStorage".into()))?;
        let kv_s = kv_raw
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("mla_q_kv_norm_split: kv_raw is not CudaStorage".into()))?;
        let qw_s = q_norm_w
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("mla_q_kv_norm_split: q_norm_w is not CudaStorage".into()))?;
        let kvw_s = kv_norm_w
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("mla_q_kv_norm_split: kv_norm_w is not CudaStorage".into()))?;

        let q_nope_st = CudaStorage::alloc_gpu(&Shape::new(vec![qk_nope_dim]), DType::F32, self.ordinal)?;
        let q_rope_st = CudaStorage::alloc_gpu(&Shape::new(vec![qk_rope_dim]), DType::F32, self.ordinal)?;
        let kv_nope_st = CudaStorage::alloc_gpu(&Shape::new(vec![qk_nope_dim]), DType::F32, self.ordinal)?;
        let kv_rope_st = CudaStorage::alloc_gpu(&Shape::new(vec![qk_rope_dim]), DType::F32, self.ordinal)?;

        let mut q_ptr = Self::dev_ptr_or_err("mla_q_kv_norm_split q_raw", q_s)?;
        let mut kv_ptr = Self::dev_ptr_or_err("mla_q_kv_norm_split kv_raw", kv_s)?;
        let mut qw_ptr = Self::dev_ptr_or_err("mla_q_kv_norm_split q_norm_w", qw_s)?;
        let mut kvw_ptr = Self::dev_ptr_or_err("mla_q_kv_norm_split kv_norm_w", kvw_s)?;
        let mut q_nope_ptr = Self::dev_ptr_or_err("mla_q_kv_norm_split q_nope", &q_nope_st)?;
        let mut q_rope_ptr = Self::dev_ptr_or_err("mla_q_kv_norm_split q_rope", &q_rope_st)?;
        let mut kv_nope_ptr = Self::dev_ptr_or_err("mla_q_kv_norm_split kv_nope", &kv_nope_st)?;
        let mut kv_rope_ptr = Self::dev_ptr_or_err("mla_q_kv_norm_split kv_rope", &kv_rope_st)?;

        let mut nope_i = qk_nope_dim as i32;
        let mut rope_i = qk_rope_dim as i32;
        let mut v_i = v_dim as i32;
        let mut eps_f = eps;

        let mut args: [*mut c_void; 12] = [
            &mut q_ptr as *mut *mut c_void as *mut c_void,
            &mut kv_ptr as *mut *mut c_void as *mut c_void,
            &mut qw_ptr as *mut *mut c_void as *mut c_void,
            &mut kvw_ptr as *mut *mut c_void as *mut c_void,
            &mut q_nope_ptr as *mut *mut c_void as *mut c_void,
            &mut q_rope_ptr as *mut *mut c_void as *mut c_void,
            &mut kv_nope_ptr as *mut *mut c_void as *mut c_void,
            &mut kv_rope_ptr as *mut *mut c_void as *mut c_void,
            &mut nope_i as *mut i32 as *mut c_void,
            &mut rope_i as *mut i32 as *mut c_void,
            &mut v_i as *mut i32 as *mut c_void,
            &mut eps_f as *mut f32 as *mut c_void,
        ];

        let total = (qk_nope_dim + qk_rope_dim) as u32;
        let block_size = 256u32;
        let grid_size = (total + block_size - 1) / block_size;

        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            let func_name = std::ffi::CString::new("grim_mla_q_kv_norm_split")
                .map_err(|e| Error::Backend(format!("invalid kernel name: {e}")))?;
            let res = cuModuleGetFunction(&mut func, module, func_name.as_ptr());
            if res != 0 {
                return Err(Error::Backend(format!(
                    "cuModuleGetFunction(grim_mla_q_kv_norm_split) failed: {res}"
                )));
            }
            let launch_res = cuLaunchKernel(
                func,
                grid_size.max(1),
                1,
                1,
                block_size,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr() as *mut *mut c_void,
                std::ptr::null_mut(),
            );
            if launch_res != 0 {
                return Err(Error::Backend(format!(
                    "cuLaunchKernel(grim_mla_q_kv_norm_split) failed: {launch_res}"
                )));
            }
        }

        Ok((
            Box::new(q_nope_st),
            Box::new(q_rope_st),
            Box::new(kv_nope_st),
            Box::new(kv_rope_st),
            Box::new(CudaHandle::ready(self.ordinal)),
        ))
    }
}
