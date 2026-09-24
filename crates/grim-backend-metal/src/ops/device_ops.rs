//! device_ops ops for MetalDevice — moved verbatim from lib.rs.

use grim_tensor::backend::ComputeHandle;
#[allow(unused_imports)]
use grim_tensor::dtype::{
    DType, FloatPackScheme, KQuantScheme, QuantFormat, QuantProvenance, Storage as DTypeStorage,
};
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendStorage, CoreTensorOps, Shape};

#[cfg(target_vendor = "apple")]
use objc2::rc::Retained;
#[cfg(target_vendor = "apple")]
use objc2::runtime::ProtocolObject;
#[cfg(target_vendor = "apple")]
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLComputePipelineState, MTLDevice, MTLSize,
};

use crate::*;

impl MetalDevice {
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
        // The Metal `grim_qkv_attention` kernel accepts a `window_lo` + `has_window` argument pair; SWA layers compute the lower bound host-side and the kernel masks below it.
        // No host fallback needed.
        let _ = &window;

        let out_dims = out.dims();
        if out_dims.len() != 3 {
            return Err(Error::Shape(
                "qkv_attention expects 3-D output shape [seq_len, num_heads, head_dim]".into(),
            ));
        }
        let seq_len = out_dims[0];
        let num_heads = out_dims[1];
        let head_dim = out_dims[2];

        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                if q.dtype().arith != ArithType::F32
                    || k.dtype().arith != ArithType::F32
                    || v.dtype().arith != ArithType::F32
                {
                    return Err(Error::from(MetalError::UnsupportedDType(q.dtype())));
                }

                let q_s = q
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| Error::Backend("qkv_attention q is not MetalStorage".into()))?;
                let k_s = k
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| Error::Backend("qkv_attention k is not MetalStorage".into()))?;
                let v_s = v
                    .as_any()
                    .downcast_ref::<MetalStorage>()
                    .ok_or_else(|| Error::Backend("qkv_attention v is not MetalStorage".into()))?;

                let q_buf = q_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("q has no GPU buffer".into()))?;
                let k_buf = k_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("k has no GPU buffer".into()))?;
                let v_buf = v_s
                    .buffer
                    .as_ref()
                    .ok_or_else(|| Error::Backend("v has no GPU buffer".into()))?;

                let max_s = match out_max {
                    Some(m) => {
                        let ms = m.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                            Error::Backend("qkv_attention out_max is not MetalStorage".into())
                        })?;
                        Some(
                            ms.buffer.as_ref().ok_or_else(|| {
                                Error::Backend("out_max has no GPU buffer".into())
                            })?,
                        )
                    }
                    None => None,
                };
                let sum_s = match out_sum {
                    Some(s) => {
                        let ss = s.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
                            Error::Backend("qkv_attention out_sum is not MetalStorage".into())
                        })?;
                        Some(
                            ss.buffer.as_ref().ok_or_else(|| {
                                Error::Backend("out_sum has no GPU buffer".into())
                            })?,
                        )
                    }
                    None => None,
                };

                let out_storage = self.zeros(out, DType::F32)?;
                let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                let out_buf = out_s.buffer.as_ref().unwrap();

                let cmd_buffer = self.get_or_create_command_buffer()?;
                let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
                    Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
                })?;

                encoder.setComputePipelineState(&inner.pipelines.qkv_attn);
                encoder.setBuffer_offset_atIndex(Some(q_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(k_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(v_buf), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 3);
                encoder.setBuffer_offset_atIndex(max_s.copied(), 0, 4);
                encoder.setBuffer_offset_atIndex(sum_s.copied(), 0, 5);

                let num_heads_val = num_heads as i32;
                let num_kv_heads_val = num_kv_heads as i32;
                let head_dim_val = head_dim as i32;
                let seq_len_val = seq_len as i32;
                let kv_seq_len_val = kv_seq_len as i32;
                let cache_offset_val = cache_offset as i32;
                let inv_sqrt_d_val = 1.0 / (head_dim as f32).sqrt();

                unsafe {
                    encoder.setBytes_length_atIndex(
                        &num_heads_val as *const i32 as *const std::ffi::c_void,
                        4,
                        6,
                    );
                    encoder.setBytes_length_atIndex(
                        &num_kv_heads_val as *const i32 as *const std::ffi::c_void,
                        4,
                        7,
                    );
                    encoder.setBytes_length_atIndex(
                        &head_dim_val as *const i32 as *const std::ffi::c_void,
                        4,
                        8,
                    );
                    encoder.setBytes_length_atIndex(
                        &seq_len_val as *const i32 as *const std::ffi::c_void,
                        4,
                        9,
                    );
                    encoder.setBytes_length_atIndex(
                        &kv_seq_len_val as *const i32 as *const std::ffi::c_void,
                        4,
                        10,
                    );
                    encoder.setBytes_length_atIndex(
                        &cache_offset_val as *const i32 as *const std::ffi::c_void,
                        4,
                        11,
                    );
                    encoder.setBytes_length_atIndex(
                        &inv_sqrt_d_val as *const f32 as *const std::ffi::c_void,
                        4,
                        12,
                    );
                    // SWA: window_lo = max(0, cache_offset - window + 1);
                    // has_window = window.is_some().
                    let abs_first = cache_offset as usize;
                    let window_lo_val: i32 = match window {
                        Some(w) => abs_first.saturating_sub(w.saturating_sub(1)) as i32,
                        None => 0,
                    };
                    let has_window_val: i32 = if window.is_some() { 1 } else { 0 };
                    encoder.setBytes_length_atIndex(
                        &window_lo_val as *const i32 as *const std::ffi::c_void,
                        4,
                        13,
                    );
                    encoder.setBytes_length_atIndex(
                        &has_window_val as *const i32 as *const std::ffi::c_void,
                        4,
                        14,
                    );
                }

                let threads_per_group = MTLSize::new(32, 1, 1);
                let groups = MTLSize::new(seq_len as u64, num_heads as u64, 1);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
                encoder.endEncoding();

                Ok((
                    out_storage,
                    Box::new(MetalHandle {
                        command_buffer: cmd_buffer,
                    }),
                ))
            } else {
                let _ = out_max;
                let _ = out_sum;
                // Host-fallback for unit tests without Apple hardware.
                let q_vec = q.to_cpu_vec_f32()?;
                let k_vec = k.to_cpu_vec_f32()?;
                let v_vec = v.to_cpu_vec_f32()?;

                let mut out_vec = vec![0.0f32; out.elem_count()];
                let inv_sqrt_d = 1.0 / (head_dim as f32).sqrt();

                for i in 0..seq_len {
                    for h in 0..num_heads {
                        let q_per_kv = num_heads / num_kv_heads;
                        let kv_head = h / q_per_kv;
                        let q_offset = (i * num_heads + h) * head_dim;
                        let abs_i = cache_offset as usize + i;
                        let range_len = if abs_i < kv_seq_len {
                            abs_i + 1
                        } else {
                            kv_seq_len
                        };

                        let mut running_max = -1e30_f32;
                        let mut running_sum = 0.0_f32;

                        let mut scores = vec![0.0f32; range_len];
                        for j in 0..range_len {
                            let mut score = 0.0_f32;
                            for d in 0..head_dim {
                                score += q_vec[q_offset + d]
                                    * k_vec[(j * num_kv_heads + kv_head) * head_dim + d];
                            }
                            score *= inv_sqrt_d;
                            scores[j] = score;
                            if score > running_max {
                                running_max = score;
                            }
                        }

                        for j in 0..range_len {
                            running_sum += (scores[j] - running_max).exp();
                        }

                        for d in 0..head_dim {
                            let mut acc = 0.0_f32;
                            for j in 0..range_len {
                                let weight = (scores[j] - running_max).exp()
                                    / (if running_sum > 0.0_f32 {
                                        running_sum
                                    } else {
                                        1.0_f32
                                    });
                                acc += weight * v_vec[(j * num_kv_heads + kv_head) * head_dim + d];
                            }
                            out_vec[q_offset + d] = acc;
                        }
                    }
                }

                let out_storage = self.from_cpu(&out_vec, out, DType::F32)?;
                Ok((out_storage, Box::new(MetalHandle)))
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = out_max;
            let _ = out_sum;
            // Host-fallback for unit tests without Apple hardware.
            let q_vec = q.to_cpu_vec_f32()?;
            let k_vec = k.to_cpu_vec_f32()?;
            let v_vec = v.to_cpu_vec_f32()?;

            let mut out_vec = vec![0.0f32; out.elem_count()];
            let inv_sqrt_d = 1.0 / (head_dim as f32).sqrt();

            for i in 0..seq_len {
                for h in 0..num_heads {
                    let q_per_kv = num_heads / num_kv_heads;
                    let kv_head = h / q_per_kv;
                    let q_offset = (i * num_heads + h) * head_dim;
                    let abs_i = cache_offset as usize + i;
                    let range_len = if abs_i < kv_seq_len {
                        abs_i + 1
                    } else {
                        kv_seq_len
                    };

                    let mut running_max = -1e30_f32;
                    let mut running_sum = 0.0_f32;

                    let mut scores = vec![0.0f32; range_len];
                    for j in 0..range_len {
                        let mut score = 0.0_f32;
                        for d in 0..head_dim {
                            score += q_vec[q_offset + d]
                                * k_vec[(j * num_kv_heads + kv_head) * head_dim + d];
                        }
                        score *= inv_sqrt_d;
                        scores[j] = score;
                        if score > running_max {
                            running_max = score;
                        }
                    }

                    for &score in scores.iter() {
                        running_sum += (score - running_max).exp();
                    }

                    for d in 0..head_dim {
                        let mut acc = 0.0_f32;
                        for (j, &score_j) in scores.iter().enumerate() {
                            let weight = (score_j - running_max).exp()
                                / (if running_sum > 0.0_f32 {
                                    running_sum
                                } else {
                                    1.0_f32
                                });
                            acc += weight * v_vec[(j * num_kv_heads + kv_head) * head_dim + d];
                        }
                        out_vec[q_offset + d] = acc;
                    }
                }
            }

            let out_storage = self.from_cpu(&out_vec, out, DType::F32)?;
            Ok((out_storage, Box::new(MetalHandle)))
        }
    }

    #[cfg(target_vendor = "apple")]
    fn run_elementwise(
        &self,
        inner: &MetalDeviceInner,
        pipeline: &Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a_s = a.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
            Error::Backend("Metal elementwise: input a is not MetalStorage".into())
        })?;
        let b_s = b.as_any().downcast_ref::<MetalStorage>().ok_or_else(|| {
            Error::Backend("Metal elementwise: input b is not MetalStorage".into())
        })?;
        let a_buf = a_s
            .buffer
            .as_ref()
            .ok_or_else(|| Error::Backend("a has no GPU buffer".into()))?;
        let b_buf = b_s
            .buffer
            .as_ref()
            .ok_or_else(|| Error::Backend("b has no GPU buffer".into()))?;

        let out_storage = self.zeros(out, a.dtype())?;
        let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
        let out_buf = out_s.buffer.as_ref().unwrap();

        let total = out.elem_count();

        let cmd_buffer = self.get_or_create_command_buffer()?;
        let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
            Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
        })?;

        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer_offset_atIndex(Some(a_buf), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(b_buf), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 2);

        let total_val = total as i32;
        unsafe {
            encoder.setBytes_length_atIndex(
                &total_val as *const i32 as *const std::ffi::c_void,
                4,
                3,
            );
        }

        let threads_per_group = MTLSize::new(256, 1, 1);
        let groups = MTLSize::new(((total + 255) / 256) as u64, 1, 1);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
        encoder.endEncoding();

        Ok((
            out_storage,
            Box::new(MetalHandle {
                command_buffer: cmd_buffer,
            }),
        ))
    }

    #[cfg(target_vendor = "apple")]
    fn run_unary(
        &self,
        inner: &MetalDeviceInner,
        pipeline: &Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        input: &dyn BackendStorage,
        scalar: Option<f32>,
        out: &Shape,
        scalar_binding: Option<usize>,
        n_binding: usize,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let input_s = input
            .as_any()
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| Error::Backend("Metal unary: input is not MetalStorage".into()))?;
        let input_buf = input_s
            .buffer
            .as_ref()
            .ok_or_else(|| Error::Backend("input has no GPU buffer".into()))?;

        let out_storage = self.zeros(out, input.dtype())?;
        let out_s = out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
        let out_buf = out_s.buffer.as_ref().unwrap();

        let total = out.elem_count();

        let cmd_buffer = self.get_or_create_command_buffer()?;
        let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
            Error::from(MetalError::Ffi("Failed to create compute encoder".into()))
        })?;

        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer_offset_atIndex(Some(input_buf), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 1);

        let total_val = total as i32;
        if let Some(s_val) = scalar {
            if let Some(sb) = scalar_binding {
                unsafe {
                    encoder.setBytes_length_atIndex(
                        &s_val as *const f32 as *const std::ffi::c_void,
                        4,
                        sb as u64,
                    );
                }
            }
        }
        unsafe {
            encoder.setBytes_length_atIndex(
                &total_val as *const i32 as *const std::ffi::c_void,
                4,
                n_binding as u64,
            );
        }

        let threads_per_group = MTLSize::new(256, 1, 1);
        let groups = MTLSize::new(((total + 255) / 256) as u64, 1, 1);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads_per_group);
        encoder.endEncoding();

        Ok((
            out_storage,
            Box::new(MetalHandle {
                command_buffer: cmd_buffer,
            }),
        ))
    }

    #[cfg(not(target_vendor = "apple"))]
    #[allow(dead_code, unused_variables)] // stub for non-Apple builds
    fn run_unary(
        &self,
        _input: &dyn BackendStorage,
        _scalar: Option<f32>,
        out: &Shape,
        _scalar_binding: Option<usize>,
        _n_binding: usize,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let _ = _scalar;
        let _ = _scalar_binding;
        let _ = _n_binding;
        // Non-Apple target — callers should fall back to CPU.
        Err(Error::Backend(
            "Metal unary kernel not available on this platform".into(),
        ))
    }
}
