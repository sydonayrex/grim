//! collective_ops ops for MetalDevice — moved verbatim from lib.rs.

use grim_tensor::backend::ComputeHandle;
#[allow(unused_imports)]
use grim_tensor::dtype::{
    DType, FloatPackScheme, KQuantScheme, QuantFormat, QuantProvenance, Storage as DTypeStorage,
};
use grim_tensor::error::{Error, Result};
use grim_tensor::{
    ArithType, BackendStorage, CollectiveOps, CoreTensorOps, ScythePlacement, Shape,
};

#[cfg(target_vendor = "apple")]
use objc2::rc::Retained;
#[cfg(target_vendor = "apple")]
use objc2::runtime::ProtocolObject;
#[cfg(target_vendor = "apple")]
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLComputePipelineState, MTLDevice, MTLSize,
};

use crate::*;

impl CollectiveOps for MetalDevice {
    #[allow(unused_variables)] // locals only used on the cfg-gated Apple path
    fn all_reduce(
        &self,
        inputs: &[&dyn BackendStorage],
        op: &str,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        if inputs.is_empty() {
            return Err(Error::Backend("all_reduce: no inputs".into()));
        }
        if op != "sum" {
            return Err(Error::Backend(format!(
                "all_reduce: only 'sum' supported, got '{op}'"
            )));
        }
        let shape = inputs[0].shape().clone();
        let dtype = inputs[0].dtype();
        let total = shape.elem_count();
        let is_f32 = dtype.arith == ArithType::F32;

        // All inputs must share the same shape.
        for s in inputs {
            if s.shape() != &shape {
                return Err(Error::Backend("all_reduce: input shape mismatch".into()));
            }
        }

        // ── GPU fast path: zero the output, then accumulate each input in turn.
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                if is_f32 && total > 0 {
                    // Validate that every input is GPU-backed before dispatching.
                    let mut input_bufs: Vec<&Retained<ProtocolObject<dyn MTLBuffer>>> =
                        Vec::with_capacity(inputs.len());
                    let mut valid = true;
                    for input in inputs {
                        match input.as_any().downcast_ref::<MetalStorage>() {
                            Some(s) => match &s.buffer {
                                Some(b) => input_bufs.push(b),
                                None => {
                                    valid = false;
                                    break;
                                }
                            },
                            None => {
                                valid = false;
                                break;
                            }
                        }
                    }
                    if valid {
                        if let Ok(out_storage) = self.zeros(&shape, DType::F32) {
                            let out_s =
                                out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                            let out_buf = out_s.buffer.as_ref().unwrap();

                            let cmd = self.get_or_create_command_buffer()?;
                            let encoder = cmd.computeCommandEncoder().ok_or_else(|| {
                                Error::from(MetalError::Ffi(
                                    "Failed to create compute encoder".into(),
                                ))
                            })?;

                            encoder.setComputePipelineState(&inner.pipelines.all_reduce);
                            let n_val = total as i32;
                            let groups = MTLSize::new(((total + 255) / 256) as u64, 1, 1);
                            let threads = MTLSize::new(256, 1, 1);
                            unsafe {
                                encoder.setBytes_length_atIndex(
                                    &n_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    2,
                                );
                            }
                            for in_buf in &input_bufs {
                                encoder.setBuffer_offset_atIndex(Some(*in_buf), 0, 0);
                                encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 1);
                                encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
                            }
                            encoder.endEncoding();

                            return Ok((
                                out_storage,
                                Box::new(MetalHandle {
                                    command_buffer: cmd,
                                }),
                            ));
                        }
                    }
                }
            }
        }

        // ── CPU fallback ─────────────────────────────────────────────────
        let mut acc = inputs[0].to_cpu_vec_f32()?;
        for other in &inputs[1..] {
            let v = other.to_cpu_vec_f32()?;
            if v.len() != acc.len() {
                return Err(Error::Backend(
                    "all_reduce: input length mismatch during fallback".into(),
                ));
            }
            for (a, b) in acc.iter_mut().zip(v.iter()) {
                *a += b;
            }
        }
        let storage = self.from_cpu(&acc, &shape, dtype)?;
        #[cfg(target_vendor = "apple")]
        {
            let command_buffer = self.get_or_create_command_buffer()?;
            Ok((storage, Box::new(MetalHandle { command_buffer })))
        }
        #[cfg(not(target_vendor = "apple"))]
        Ok((storage, Box::new(MetalHandle)))
    }

    #[allow(unused_variables)] // locals only used on the cfg-gated Apple path
    fn comm_fuse_reduce(
        &self,
        partials: &[(&dyn BackendStorage, &ScythePlacement)],
    ) -> Result<Box<dyn BackendStorage>> {
        if partials.is_empty() {
            return Err(Error::Backend("comm_fuse_reduce: no partials".into()));
        }
        let dims0 = partials[0].0.shape().dims();
        let m = dims0[0];
        let n_total: usize = partials
            .iter()
            .map(|(s, _)| s.shape().dims().get(1).copied().unwrap_or(0))
            .sum();
        let dtype = partials[0].0.dtype();
        let is_f32 = dtype.arith == ArithType::F32;
        let out_shape = Shape::new(vec![m, n_total]);

        // ── GPU fast path: zero the output, then scatter-copy each shard.
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                if is_f32 && n_total > 0 {
                    // Validate that every shard is GPU-backed before dispatching.
                    let mut entries: Vec<(&Retained<ProtocolObject<dyn MTLBuffer>>, usize)> =
                        Vec::with_capacity(partials.len());
                    let mut valid = true;
                    for (storage, _placement) in partials {
                        match storage.as_any().downcast_ref::<MetalStorage>() {
                            Some(s) => match &s.buffer {
                                Some(b) => {
                                    let n_src = s.shape().dims().get(1).copied().unwrap_or(0);
                                    entries.push((b, n_src));
                                }
                                None => {
                                    valid = false;
                                    break;
                                }
                            },
                            None => {
                                valid = false;
                                break;
                            }
                        }
                    }
                    if valid {
                        if let Ok(out_storage) = self.zeros(&out_shape, DType::F32) {
                            let out_s =
                                out_storage.as_any().downcast_ref::<MetalStorage>().unwrap();
                            let out_buf = out_s.buffer.as_ref().unwrap();

                            let cmd = self.get_or_create_command_buffer()?;
                            let encoder = cmd.computeCommandEncoder().ok_or_else(|| {
                                Error::from(MetalError::Ffi(
                                    "Failed to create compute encoder".into(),
                                ))
                            })?;

                            encoder.setComputePipelineState(&inner.pipelines.comm_fuse_reduce);
                            let m_val = m as i32;
                            let n_total_val = n_total as i32;
                            unsafe {
                                encoder.setBytes_length_atIndex(
                                    &m_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    2,
                                );
                                encoder.setBytes_length_atIndex(
                                    &n_total_val as *const i32 as *const std::ffi::c_void,
                                    4,
                                    5,
                                );
                            }
                            let threads = MTLSize::new(16, 16, 1);
                            let mut col_offset = 0usize;
                            for (in_buf, n_src) in &entries {
                                encoder.setBuffer_offset_atIndex(Some(*in_buf), 0, 0);
                                encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 1);
                                let n_src_val = *n_src as i32;
                                let col_offset_val = col_offset as i32;
                                unsafe {
                                    encoder.setBytes_length_atIndex(
                                        &n_src_val as *const i32 as *const std::ffi::c_void,
                                        4,
                                        3,
                                    );
                                    encoder.setBytes_length_atIndex(
                                        &col_offset_val as *const i32 as *const std::ffi::c_void,
                                        4,
                                        4,
                                    );
                                }
                                let groups = MTLSize::new(
                                    ((*n_src + 15) / 16) as u64,
                                    ((m + 15) / 16) as u64,
                                    1,
                                );
                                encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
                                col_offset += *n_src;
                            }
                            encoder.endEncoding();

                            return Ok(Box::new(out_storage));
                        }
                    }
                }
            }
        }

        // ── CPU fallback ─────────────────────────────────────────────────
        let mut assembled = vec![0.0f32; m * n_total];
        let mut col_offset = 0usize;
        for (storage, _placement) in partials {
            let data = storage.to_cpu_vec_f32()?;
            let n_cols = storage.shape().dims().get(1).copied().unwrap_or(0);
            for row in 0..m {
                for col in 0..n_cols {
                    assembled[row * n_total + col_offset + col] += data[row * n_cols + col];
                }
            }
            col_offset += n_cols;
        }
        let storage = self.from_cpu(&assembled, &out_shape, dtype)?;
        Ok(storage)
    }

    fn estimate_gemm_latency_ms(
        &self,
        m: usize,
        n: usize,
        k: usize,
        dtype: DType,
        _placement: &grim_tensor::backend::ScythePlacement,
    ) -> f64 {
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let tflops = match dtype.arith {
            ArithType::F16 | ArithType::BF16 => 200.0,
            ArithType::F32 => 100.0,
            _ => 50.0,
        };
        (flops / (tflops * 1e12) * 1000.0).max(0.01)
    }
}
