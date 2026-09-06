//! Serving, collective operations, memory, and graph capture operations for `RocmDevice`.

use std::ffi::c_void;

use grim_tensor::backend::{ReadyHandle, ScythePlacement};
use grim_tensor::dtype::{ArithType, DType};
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendStorage, CollectiveOps, CoreTensorOps, GraphCaptureOps, MemoryOps, Shape};

use crate::device::roc_device::RocmDevice;
use crate::memory::storage::RocmStorage;
use crate::{
    arg, as_rocm, check_hip, detect_gpu_arch, dev_ptr, dtype_f32, hipMemcpyAsync, HipDim3,
    HipMemcpyKind, RocmHandle,
};

impl CollectiveOps for RocmDevice {


    /// SCYTHE-2 WI-5: BackendDevice::all_reduce for RocmDevice. [see: `RowParallelLinear::forward`, `BackendDevice::all_reduce`]
    ///
    /// Performs the sum collective entirely on the ROCm device:
    /// - Cross-GPU: when an RCCL handle is attached and `num_gpus > 1`, uses
    ///   `RcclAllReduce::sum_gradients_device` for a device-side `ncclAllReduce`.
    /// - Intra-process: sums multiple partial shards on-device via the
    ///   `grim_all_reduce_accum` kernel (F32), avoiding the D2H/H2D round-trip.
    /// - Fallback: CPU fan-in for non-F32 dtypes or mismatched shard shapes.
    fn all_reduce(
        &self,
        inputs: &[&dyn grim_tensor::BackendStorage],
        op: &str,
    ) -> grim_tensor::error::Result<(
        Box<dyn grim_tensor::BackendStorage>,
        Box<dyn grim_tensor::backend::ComputeHandle>,
    )> {
        // P1-3: raw HIP ops below bind to the calling thread's current
        // device — pin to the owning ordinal (see matmul_op fix, 2026-08-23e).
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
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
        let stream = self.active_stream();
        let stream_u64 = stream as u64;
        let rccl = self.rccl.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let is_f32 = dtype.arith == ArithType::F32;
        // TP activations arrive as F16/BF16 single tensors per rank; routing
        // them through RCCL (instead of the old host round-trip) needs the
        // matching NCCL dtype.
        let rccl_dtype = match dtype.arith {
            ArithType::F32 => Some(crate::rccl::NCCL_FLOAT32),
            ArithType::F16 => Some(crate::rccl::NCCL_FLOAT16),
            ArithType::BF16 => Some(crate::rccl::NCCL_BFLOAT16),
            _ => None,
        };

        // ── Cross-GPU all-reduce via RCCL (device-side) ───────────────────
        // When an RCCL handle is attached and we have multiple GPUs, perform
        // the collective directly on device memory via ncclAllReduce.
        if let Some(rccl_handle) = &rccl {
            if rccl_handle.num_gpus > 1 && is_f32 {
                let out_storage =
                    RocmStorage::alloc_gpu(&shape, dtype_f32(), &self.allocator, self.ordinal)?;
                let out_ptr = dev_ptr(&out_storage)?;

                if inputs.len() == 1 {
                    // Single tensor: direct cross-GPU all-reduce.
                    let send_ptr = dev_ptr(as_rocm(inputs[0])?)?;
                    rccl_handle.sum_gradients_device(
                        send_ptr,
                        out_ptr,
                        total,
                        stream_u64,
                        self.ordinal,
                    )?;
                } else {
                    // Multiple shards: accumulate on-device first, then all-reduce.
                    let temp_storage =
                        RocmStorage::alloc_gpu(&shape, dtype_f32(), &self.allocator, self.ordinal)?;
                    let temp_ptr = dev_ptr(&temp_storage)?;
                    self.device_accumulate_f32(inputs, temp_ptr)?;
                    rccl_handle.sum_gradients_device(
                        temp_ptr,
                        out_ptr,
                        total,
                        stream_u64,
                        self.ordinal,
                    )?;
                }

                return Ok((
                    Box::new(out_storage),
                    Box::new(RocmHandle::new(Some(stream))),
                ));
            }

            // F16/BF16 single-shard TP activations: all-reduce in the native
            // dtype — previously this fell through to a full D2H→CPU-sum→H2D
            // round trip per RowParallel layer per token.
            if rccl_handle.num_gpus > 1 && !is_f32 && inputs.len() == 1 {
                if let Some(nccl_dt) = rccl_dtype {
                    let out_storage = RocmStorage::alloc_gpu(
                        &shape,
                        dtype.clone(),
                        &self.allocator,
                        self.ordinal,
                    )?;
                    let out_ptr = dev_ptr(&out_storage)?;
                    let send_ptr = dev_ptr(as_rocm(inputs[0])?)?;
                    rccl_handle.all_reduce_device(
                        send_ptr,
                        out_ptr,
                        total,
                        nccl_dt,
                        stream_u64,
                        self.ordinal,
                    )?;
                    return Ok((
                        Box::new(out_storage),
                        Box::new(RocmHandle::new(Some(stream))),
                    ));
                }
            }
        }

        // ── Intra-process device-side fan-in (no RCCL) ────────────────────
        // Avoid the CPU round-trip: sum partials directly on the GPU.
        if is_f32 && total > 0 {
            if inputs.len() == 1 {
                // Identity: device-to-device copy (no D2H + H2D round-trip).
                let bytes = total * crate::dtype_byte_size(&dtype);
                let out_storage =
                    RocmStorage::alloc_gpu(&shape, dtype.clone(), &self.allocator, self.ordinal)?;
                let src_ptr = dev_ptr(as_rocm(inputs[0])?)? as *const c_void;
                let dst_ptr = out_storage.device_ptr_checked()? as *mut c_void;
                check_hip("hipMemcpyAsync(D2D) all_reduce", unsafe {
                    hipMemcpyAsync(
                        dst_ptr,
                        src_ptr,
                        bytes,
                        HipMemcpyKind::DeviceToDevice,
                        stream,
                    )
                })?;
                return Ok((
                    Box::new(out_storage),
                    Box::new(RocmHandle::new(Some(stream))),
                ));
            }

            // Multi-input: device-side element-wise sum via grim_all_reduce_accum.
            let all_same = inputs.iter().all(|s| s.shape() == inputs[0].shape());
            if all_same {
                let out_storage =
                    RocmStorage::alloc_gpu(&shape, dtype_f32(), &self.allocator, self.ordinal)?;
                let out_ptr = dev_ptr(&out_storage)?;
                self.device_accumulate_f32(inputs, out_ptr)?;
                return Ok((
                    Box::new(out_storage),
                    Box::new(RocmHandle::new(Some(stream))),
                ));
            }
        }

        // ── CPU fallback ───────────────────────────────────────────────────
        // Used for non-F32 dtypes or mismatched shard shapes where the device
        // accum kernel cannot apply.
        let mut acc = inputs[0].to_cpu_vec_f32()?;
        for other in &inputs[1..] {
            let v = other.to_cpu_vec_f32()?;
            if v.len() != acc.len() {
                return Err(Error::Backend(format!(
                    "all_reduce: input shape mismatch (first {} != other {})",
                    acc.len(),
                    v.len()
                )));
            }
            for (a, b) in acc.iter_mut().zip(v.iter()) {
                *a += b;
            }
        }
        let storage = self.from_cpu(&acc, &shape, dtype)?;
        Ok((storage, Box::new(ReadyHandle)))
    }


    /// SCYTHE-2 WI-1/WI-6: WaveTune bilinear latency predictor for RocmDevice. [see: `(M, N, K)`, `2604.10187`]
    fn estimate_gemm_latency_ms(
        &self,
        m: usize,
        n: usize,
        k: usize,
        dtype: DType,
        _placement: &grim_tensor::backend::ScythePlacement,
    ) -> f64 {
        // Peak TFLOPS from arch string (same table as capability_profiler.rs).
        let arch = detect_gpu_arch(self.ordinal as i32);
        let tflops_fp16: f64 = if arch.starts_with("gfx1100") {
            61.4
        } else if arch.starts_with("gfx1102") {
            26.0
        } else if arch.starts_with("gfx12") {
            80.0
        } else if arch.starts_with("gfx9") {
            190.0
        } else {
            20.0 // conservative unknown
        };
        // Apply dtype factor: FP8 is 2× FP16 on RDNA4+; FP32 is 0.5×.
        let dtype_factor = match dtype.arith {
            ArithType::F32 => 0.5,
            _ => 1.0,
        };
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let peak = tflops_fp16 * dtype_factor * 1e12; // FLOPS/s
        if peak <= 0.0 {
            return f64::INFINITY;
        }
        flops / peak * 1e3 // ms
    }


    /// SCYTHE-2 WI-6: CommFuse decomposed P2P fan-in override. [see: `crate::comm_fuse::comm_fuse_fan_in`, `to_cpu_vec_f32`]
    ///
    /// Assembles column-shard partials entirely on the ROCm device:
    /// - Device-side: places each partial at its column offset via row-by-row
    ///   `hipMemcpy` D2D, avoiding the D2H/H2D round-trip. When an RCCL handle
    ///   is attached and `num_gpus > 1`, a cross-GPU `ncclAllReduce` is issued
    ///   after assembly.
    /// - Fallback: CPU fan-in for non-F32 dtypes.
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
        let stream = self.active_stream();
        let stream_u64 = stream as u64;
        let rccl = self.rccl.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let elem_bytes = crate::dtype_byte_size(&dtype);

        // ── Device-side assembly + optional RCCL all-reduce ────────────────
        if is_f32 {
            // WI-M1 context discipline: the synchronous D2D memcpys below
            // execute in the calling thread's current device context; pin
            // THIS device or a drifted thread assembles the fan-in buffer
            // against foreign mappings.
            let _ctx = crate::device::util::DeviceGuard::set(self.ordinal as i32);
            let out_shape = Shape::from_slice(&[m, n_total]);
            let out_storage =
                RocmStorage::alloc_gpu(&out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
            let out_ptr_val = dev_ptr(&out_storage)?;
            let out_ptr_usize = out_ptr_val as usize;

            // Place each partial at its column offset, row by row (D2D memcpy).
            let mut col_offset = 0usize;
            for (storage, _placement) in partials {
                let s = as_rocm(*storage)?;
                let partial_ptr = dev_ptr(s)? as usize;
                let n_cols = s.shape().dims().get(1).copied().unwrap_or(0);
                for row in 0..m {
                    let src = (partial_ptr + row * n_cols * elem_bytes) as *const c_void;
                    let dst =
                        (out_ptr_usize + (row * n_total + col_offset) * elem_bytes) as *mut c_void;
                    check_hip("hipMemcpy(D2D) comm_fuse", unsafe {
                        crate::hipMemcpy(
                            dst,
                            src,
                            n_cols * elem_bytes,
                            crate::HipMemcpyKind::DeviceToDevice,
                        )
                    })?;
                }
                col_offset += n_cols;
            }

            // Optional RCCL cross-GPU all-reduce on the assembled buffer.
            let total_elems = m * n_total;
            if let Some(rccl_handle) = &rccl {
                if rccl_handle.num_gpus > 1 {
                    rccl_handle.sum_gradients_device(
                        out_ptr_val,
                        out_ptr_val,
                        total_elems,
                        stream_u64,
                        self.ordinal,
                    )?;
                }
            }

            return Ok(Box::new(out_storage));
        }

        // ── CPU fallback (non-F32 dtypes) ──────────────────────────────────
        let mut host_data: Vec<Vec<f32>> = Vec::with_capacity(partials.len());
        let mut n_cols_list: Vec<usize> = Vec::with_capacity(partials.len());
        for (storage, _placement) in partials {
            let data = storage.to_cpu_vec_f32()?;
            let n_cols = storage.shape().dims().get(1).copied().unwrap_or(0);
            host_data.push(data);
            n_cols_list.push(n_cols);
        }
        let slice_refs: Vec<(&[f32], usize)> = host_data
            .iter()
            .zip(n_cols_list.iter())
            .map(|(d, &nc)| (d.as_slice(), nc))
            .collect();

        let result =
            crate::kernels::comm_fuse::comm_fuse_fan_in(&slice_refs, m, n_total, partials[0].1)?;

        let out_shape = Shape::from_slice(&[result.shape.0, result.shape.1]);
        let out_storage = self.from_cpu(&result.data, &out_shape, DType::F32)?;
        Ok(out_storage)
    }
}



impl MemoryOps for RocmDevice {


    fn from_cpu_bytes(
        &self,
        data: &[u8],
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn BackendStorage>> {
        RocmStorage::copy_from_host_raw_bytes(data, shape, dtype, &self.allocator, self.ordinal)
            .map(|s| Box::new(s) as Box<dyn BackendStorage>)
    }


    fn alloc_storage(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn BackendStorage>> {
        RocmStorage::alloc_gpu(shape, dtype, &self.allocator, self.ordinal)
            .map(|s| Box::new(s) as Box<dyn BackendStorage>)
    }


    fn copy_slice_into(
        &self,
        dst: &dyn BackendStorage,
        src: &dyn BackendStorage,
        dst_elem_offset: usize,
        count: usize,
    ) -> Result<()> {
        self.copy_slice_range(dst, dst_elem_offset, src, 0, count)
    }

    fn copy_slice_range(
        &self,
        dst: &dyn BackendStorage,
        dst_elem_offset: usize,
        src: &dyn BackendStorage,
        src_elem_offset: usize,
        count: usize,
    ) -> Result<()> {
        // P1-3: raw HIP ops below bind to the calling thread's current
        // device — pin to the owning ordinal (see matmul_op fix, 2026-08-23e).
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let dst_s = as_rocm(dst)?;
        let src_s = as_rocm(src)?;
        if !dst_s.device_ptr_is_valid() || !src_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "copy_slice_range: inputs lack a valid device pointer".into(),
            ));
        }
        if dst_s.device_ordinal() != src_s.device_ordinal() {
            return Err(Error::Backend(format!(
                "copy_slice_range: cross-device D2D (dst ordinal {}, src ordinal {}) — \
                 use copy_via_route for routed transfers",
                dst_s.device_ordinal(),
                src_s.device_ordinal()
            )));
        }
        if dst_elem_offset + count > dst_s.shape().elem_count() {
            return Err(Error::Shape(format!(
                "copy_slice_range: dst overflow (offset={dst_elem_offset} + count={count} > {})",
                dst_s.shape().elem_count()
            )));
        }
        if src_elem_offset + count > src_s.shape().elem_count() {
            return Err(Error::Shape(format!(
                "copy_slice_range: src overflow (offset={src_elem_offset} + count={count} > {})",
                src_s.shape().elem_count()
            )));
        }
        let bytes = count * std::mem::size_of::<f32>();
        let dst_ptr = unsafe {
            (dst_s.device_ptr_checked()? as *mut c_void)
                .add(dst_elem_offset * std::mem::size_of::<f32>())
        };
        let src_ptr = unsafe {
            (src_s.device_ptr_checked()? as *const c_void)
                .add(src_elem_offset * std::mem::size_of::<f32>())
        };
        check_hip("copy_slice_range: hipMemcpyAsync D2D", unsafe {
            hipMemcpyAsync(
                dst_ptr,
                src_ptr,
                bytes,
                HipMemcpyKind::DeviceToDevice,
                self.active_stream(),
            )
        })?;
        Ok(())
    }
}



impl GraphCaptureOps for RocmDevice {
}




impl RocmDevice {

    /// Launch GPU Speculative Rejection Sampling kernel.
    pub fn launch_speculative_rejection_sample(
        &self,
        target_probs_storage: &RocmStorage,
        draft_probs_storage: &RocmStorage,
        draft_tokens_storage: &RocmStorage,
        uniform_rands_storage: &RocmStorage,
        accepted_tokens_storage: &RocmStorage,
        accepted_lens_storage: &RocmStorage,
        batch_size: usize,
        num_draft_tokens: usize,
        vocab_size: usize,
    ) -> Result<*mut c_void> {
        let tp_ptr = target_probs_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("spec_sample: target_probs has no device ptr".into()))?;
        let dp_ptr = draft_probs_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("spec_sample: draft_probs has no device ptr".into()))?;
        let dt_ptr = draft_tokens_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("spec_sample: draft_tokens has no device ptr".into()))?;
        let ur_ptr = uniform_rands_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("spec_sample: uniform_rands has no device ptr".into()))?;
        let at_ptr = accepted_tokens_storage.device_ptr.ok_or_else(|| {
            Error::Backend("spec_sample: accepted_tokens has no device ptr".into())
        })?;
        let al_ptr = accepted_lens_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("spec_sample: accepted_lens has no device ptr".into()))?;

        let block_dim = HipDim3::new(256, 1, 1);
        let grid_dim = HipDim3::new(batch_size as u32, 1, 1);

        let mut tpptr = tp_ptr;
        let mut dpptr = dp_ptr;
        let mut dtptr = dt_ptr;
        let mut urptr = ur_ptr;
        let mut atptr = at_ptr;
        let mut alptr = al_ptr;
        let mut bs = batch_size as i32;
        let mut ndt = num_draft_tokens as i32;
        let mut vs = vocab_size as i32;

        self.launch_compute_kernel(
            "grim_speculative_rejection_sample",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut tpptr),
                arg(&mut dpptr),
                arg(&mut dtptr),
                arg(&mut urptr),
                arg(&mut atptr),
                arg(&mut alptr),
                arg(&mut bs),
                arg(&mut ndt),
                arg(&mut vs),
            ],
        )
    }

    /// Launch GPU stochastic sampler (WI-X3): temperature + top-k + Gumbel-max
    /// on device; only the chosen token id crosses back to host.
    pub fn launch_sample_stochastic(
        &self,
        logits_storage: &RocmStorage,
        out_tokens_storage: &RocmStorage,
        temperature: f32,
        top_k: u32,
        seed: u64,
        batch_size: usize,
        vocab_size: usize,
    ) -> Result<*mut c_void> {
        let lg_ptr = logits_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("sample_stochastic: logits has no device ptr".into()))?;
        let ot_ptr = out_tokens_storage.device_ptr.ok_or_else(|| {
            Error::Backend("sample_stochastic: out_tokens has no device ptr".into())
        })?;

        let block_dim = HipDim3::new(256, 1, 1);
        let grid_dim = HipDim3::new(batch_size as u32, 1, 1);

        let mut lgptr = lg_ptr;
        let mut otptr = ot_ptr;
        let mut temp = temperature;
        let mut tk = top_k as i32;
        let mut sd = seed;
        let mut bs = batch_size as i32;
        let mut vs = vocab_size as i32;

        self.launch_compute_kernel(
            "grim_sample_stochastic",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut lgptr),
                arg(&mut otptr),
                arg(&mut temp),
                arg(&mut tk),
                arg(&mut sd),
                arg(&mut bs),
                arg(&mut vs),
            ],
        )
    }
}
