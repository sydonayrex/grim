//! Core tensor computation, GEMM, elementwise, autograd, and optimizer operations for `RocmDevice`.
//! Kernel-launch infrastructure: JIT cache, autotune persistence, stream plumbing, reductions.

use std::ffi::c_void;

use std::sync::atomic::Ordering;

use grim_tensor::dtype::{ArithType, DType, Storage as DTypeStorage};
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendStorage, Shape};

use crate::device::roc_device::RocmDevice;
use crate::memory::storage::RocmStorage;
use crate::{
    arg, as_rocm, check_hip, dev_ptr, dtype_f32, hipMemcpyAsync, hipModuleGetFunction,
    hipModuleLaunchKernel, hipModuleLoad, hipModuleUnload, hipStreamSynchronize, hipSuccess,
    jit_compile_hsaco, HipDim3, HipMemcpyKind,
};

impl RocmDevice {
    /// H2: GPU tree reduction for sum.
    /// Returns the scalar sum of all elements in `x_storage` without whole-buffer D2H copying.
    pub(crate) fn gpu_reduce_sum(&self, x_storage: &RocmStorage) -> Result<f32> {
        let n = x_storage.shape().elem_count();
        if n == 0 {
            return Err(Error::Backend("reduce_sum: empty tensor".into()));
        }
        let x_ptr = x_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("gpu_reduce_sum: x has no device ptr".into()))?;

        // Guard the active device context
        let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);

        // Grid calculation: 256 threads per block, max 1024 blocks in stage 1
        const BLOCK_SIZE: u32 = 256;
        const MAX_BLOCKS: u32 = 1024;
        let grid_blocks = ((n as u32).div_ceil(BLOCK_SIZE)).clamp(1, MAX_BLOCKS);

        // Ensure preallocated buffers are large enough
        let mut part_guard = self
            .reduce_partials_buf
            .write()
            .unwrap_or_else(|e| e.into_inner());
        if part_guard
            .as_ref()
            .is_none_or(|b| b.shape().elem_count() < grid_blocks as usize)
        {
            *part_guard = Some(RocmStorage::alloc_gpu(
                &Shape::new(vec![MAX_BLOCKS as usize]),
                dtype_f32(),
                &self.allocator,
                self.ordinal,
            )?);
        }
        let partials_buf = part_guard.as_ref().unwrap();
        let partials_ptr = dev_ptr(partials_buf)?;

        let mut out_guard = self
            .reduce_out_buf
            .write()
            .unwrap_or_else(|e| e.into_inner());
        if out_guard.is_none() {
            *out_guard = Some(RocmStorage::alloc_gpu(
                &Shape::new(vec![16usize]),
                dtype_f32(),
                &self.allocator,
                self.ordinal,
            )?);
        }
        let out_buf = out_guard.as_ref().unwrap();
        let out_ptr = dev_ptr(out_buf)?;

        // Launch stage 1
        let mut x_p = x_ptr;
        let mut part_p = partials_ptr;
        let mut n_i = n as i32;
        let _stream = self.launch_compute_kernel(
            "grim_reduce_sum_stage1",
            HipDim3::new(grid_blocks, 1, 1),
            HipDim3::new(BLOCK_SIZE, 1, 1),
            &mut [arg(&mut x_p), arg(&mut part_p), arg(&mut n_i)],
        )?;

        // Launch stage 2
        let mut out_p = out_ptr;
        let mut num_part = grid_blocks as i32;
        let stream = self.launch_compute_kernel(
            "grim_reduce_sum_stage2",
            HipDim3::new(1, 1, 1),
            HipDim3::new(BLOCK_SIZE, 1, 1),
            &mut [arg(&mut part_p), arg(&mut out_p), arg(&mut num_part)],
        )?;

        // D2H copy only 4 bytes of result
        let mut res: f32 = 0.0f32;
        check_hip("gpu_reduce_sum D2H", unsafe {
            hipMemcpyAsync(
                &mut res as *mut f32 as *mut c_void,
                out_p as *mut c_void,
                std::mem::size_of::<f32>(),
                HipMemcpyKind::DeviceToHost,
                stream,
            )
        })?;
        check_hip("gpu_reduce_sum sync", unsafe {
            hipStreamSynchronize(stream)
        })?;

        Ok(res)
    }

    /// H2: GPU tree reduction for max.
    /// Returns the maximum scalar in `x_storage` without whole-buffer D2H copying.
    pub(crate) fn gpu_reduce_max(&self, x_storage: &RocmStorage) -> Result<f32> {
        let n = x_storage.shape().elem_count();
        if n == 0 {
            return Err(Error::Backend("reduce_max: empty tensor".into()));
        }
        let x_ptr = x_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("gpu_reduce_max: x has no device ptr".into()))?;

        let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);

        const BLOCK_SIZE: u32 = 256;
        const MAX_BLOCKS: u32 = 1024;
        let grid_blocks = ((n as u32).div_ceil(BLOCK_SIZE)).clamp(1, MAX_BLOCKS);

        let mut part_guard = self
            .reduce_partials_buf
            .write()
            .unwrap_or_else(|e| e.into_inner());
        if part_guard
            .as_ref()
            .is_none_or(|b| b.shape().elem_count() < grid_blocks as usize)
        {
            *part_guard = Some(RocmStorage::alloc_gpu(
                &Shape::new(vec![MAX_BLOCKS as usize]),
                dtype_f32(),
                &self.allocator,
                self.ordinal,
            )?);
        }
        let partials_buf = part_guard.as_ref().unwrap();
        let partials_ptr = dev_ptr(partials_buf)?;

        let mut out_guard = self
            .reduce_out_buf
            .write()
            .unwrap_or_else(|e| e.into_inner());
        if out_guard.is_none() {
            *out_guard = Some(RocmStorage::alloc_gpu(
                &Shape::new(vec![16usize]),
                dtype_f32(),
                &self.allocator,
                self.ordinal,
            )?);
        }
        let out_buf = out_guard.as_ref().unwrap();
        let out_ptr = dev_ptr(out_buf)?;

        let mut x_p = x_ptr;
        let mut part_p = partials_ptr;
        let mut n_i = n as i32;
        let _ = self.launch_compute_kernel(
            "grim_reduce_max_stage1",
            HipDim3::new(grid_blocks, 1, 1),
            HipDim3::new(BLOCK_SIZE, 1, 1),
            &mut [arg(&mut x_p), arg(&mut part_p), arg(&mut n_i)],
        )?;

        let mut out_p = out_ptr;
        let mut num_part = grid_blocks as i32;
        let stream = self.launch_compute_kernel(
            "grim_reduce_max_stage2",
            HipDim3::new(1, 1, 1),
            HipDim3::new(BLOCK_SIZE, 1, 1),
            &mut [arg(&mut part_p), arg(&mut out_p), arg(&mut num_part)],
        )?;

        let mut res: f32 = 0.0f32;
        check_hip("gpu_reduce_max D2H", unsafe {
            hipMemcpyAsync(
                &mut res as *mut f32 as *mut c_void,
                out_p as *mut c_void,
                std::mem::size_of::<f32>(),
                HipMemcpyKind::DeviceToHost,
                stream,
            )
        })?;
        check_hip("gpu_reduce_max sync", unsafe {
            hipStreamSynchronize(stream)
        })?;

        Ok(res)
    }

    /// H2: GPU tree reduction for argmax.
    /// Returns the index of the maximum element in `x_storage` without whole-buffer D2H copying.
    pub(crate) fn gpu_argmax(&self, x_storage: &RocmStorage) -> Result<u32> {
        let n = x_storage.shape().elem_count();
        if n == 0 {
            return Err(Error::Backend("argmax: empty tensor".into()));
        }
        let x_ptr = x_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("gpu_argmax: x has no device ptr".into()))?;

        let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);

        const BLOCK_SIZE: u32 = 256;
        const MAX_BLOCKS: u32 = 1024;
        let grid_blocks = ((n as u32).div_ceil(BLOCK_SIZE)).clamp(1, MAX_BLOCKS);

        let mut part_guard = self
            .reduce_partials_buf
            .write()
            .unwrap_or_else(|e| e.into_inner());
        if part_guard
            .as_ref()
            .is_none_or(|b| b.shape().elem_count() < grid_blocks as usize)
        {
            *part_guard = Some(RocmStorage::alloc_gpu(
                &Shape::new(vec![MAX_BLOCKS as usize]),
                dtype_f32(),
                &self.allocator,
                self.ordinal,
            )?);
        }
        let partials_buf = part_guard.as_ref().unwrap();
        let partial_vals_ptr = dev_ptr(partials_buf)?;

        let mut idxs_guard = self
            .reduce_argmax_idxs_buf
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if idxs_guard
            .as_ref()
            .is_none_or(|b| b.shape().elem_count() < grid_blocks as usize)
        {
            *idxs_guard = Some(RocmStorage::alloc_gpu(
                &Shape::new(vec![MAX_BLOCKS as usize]),
                DType {
                    arith: ArithType::U32,
                    storage: DTypeStorage::Native,
                },
                &self.allocator,
                self.ordinal,
            )?);
        }
        let idxs_buf = idxs_guard.as_ref().unwrap();
        let partial_idxs_ptr = dev_ptr(idxs_buf)?;

        let mut out_guard = self
            .reduce_out_buf
            .write()
            .unwrap_or_else(|e| e.into_inner());
        if out_guard.is_none() {
            *out_guard = Some(RocmStorage::alloc_gpu(
                &Shape::new(vec![16usize]),
                dtype_f32(),
                &self.allocator,
                self.ordinal,
            )?);
        }
        let out_buf = out_guard.as_ref().unwrap();
        let out_ptr = dev_ptr(out_buf)?;

        let mut x_p = x_ptr;
        let mut part_v_p = partial_vals_ptr;
        let mut part_i_p = partial_idxs_ptr;
        let mut n_i = n as i32;
        let _ = self.launch_compute_kernel(
            "grim_argmax_stage1",
            HipDim3::new(grid_blocks, 1, 1),
            HipDim3::new(BLOCK_SIZE, 1, 1),
            &mut [
                arg(&mut x_p),
                arg(&mut part_v_p),
                arg(&mut part_i_p),
                arg(&mut n_i),
            ],
        )?;

        let mut out_p = out_ptr;
        let mut num_part = grid_blocks as i32;
        let stream = self.launch_compute_kernel(
            "grim_argmax_stage2",
            HipDim3::new(1, 1, 1),
            HipDim3::new(BLOCK_SIZE, 1, 1),
            &mut [
                arg(&mut part_v_p),
                arg(&mut part_i_p),
                arg(&mut out_p),
                arg(&mut num_part),
            ],
        )?;

        let mut res: u32 = 0u32;
        check_hip("gpu_argmax D2H", unsafe {
            hipMemcpyAsync(
                &mut res as *mut u32 as *mut c_void,
                out_p as *mut c_void,
                std::mem::size_of::<u32>(),
                HipMemcpyKind::DeviceToHost,
                stream,
            )
        })?;
        check_hip("gpu_argmax sync", unsafe { hipStreamSynchronize(stream) })?;

        Ok(res)
    }

    /// SPEED-ROC: Quantize FP32 activations to FP16 for WMMA GEMM input.
    /// Reduces activation memory bandwidth by 2× (4 bytes → 2 bytes per element).
    /// Called by prefill producers that want the FP16-input WMMA kernel path.
    pub(crate) fn quantize_fp16(
        &self,
        src: &RocmStorage,
        dst: &RocmStorage,
    ) -> Result<*mut c_void> {
        let src_ptr = src
            .device_ptr
            .ok_or_else(|| Error::Backend("quantize_fp16: src has no device ptr".into()))?;
        let dst_ptr = dst
            .device_ptr
            .ok_or_else(|| Error::Backend("quantize_fp16: dst has no device ptr".into()))?;
        let n = src.shape().elem_count();
        const BLOCK_SIZE: usize = 256;
        let grid_x = (n.div_ceil(BLOCK_SIZE)) as u32;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);
        let mut sptr = src_ptr;
        let mut dptr = dst_ptr;
        let mut nn = n as i32;
        self.launch_compute_kernel(
            "grim_quantize_fp16",
            grid_dim,
            block_dim,
            &mut [arg(&mut sptr), arg(&mut dptr), arg(&mut nn)],
        )
    }

    /// SPEED-ROC: Dequantize FP16 activations back to FP32 (if needed).
    pub(crate) fn dequantize_fp16(
        &self,
        src: &RocmStorage,
        dst: &RocmStorage,
    ) -> Result<*mut c_void> {
        let src_ptr = src
            .device_ptr
            .ok_or_else(|| Error::Backend("dequantize_fp16: src has no device ptr".into()))?;
        let dst_ptr = dst
            .device_ptr
            .ok_or_else(|| Error::Backend("dequantize_fp16: dst has no device ptr".into()))?;
        let n = src.shape().elem_count();
        const BLOCK_SIZE: usize = 256;
        let grid_x = (n.div_ceil(BLOCK_SIZE)) as u32;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);
        let mut sptr = src_ptr;
        let mut dptr = dst_ptr;
        let mut nn = n as i32;
        self.launch_compute_kernel(
            "grim_dequantize_fp16",
            grid_dim,
            block_dim,
            &mut [arg(&mut sptr), arg(&mut dptr), arg(&mut nn)],
        )
    }

    /// SPEED-ROC: overwrite a device-resident tensor's storage with host f32
    /// values in place (no alloc) — decode-loop hot path reuse.
    pub fn write_f32_into(&self, storage: &dyn BackendStorage, host: &[f32]) -> Result<()> {
        let rs = as_rocm(storage)?;
        rs.write_host_f32(host)
    }

    /// Async variant on the active (or capture) stream. No host sync;
    /// ordered vs later launches on the same stream. Falls back sync on null stream.
    pub fn write_f32_into_async(&self, storage: &dyn BackendStorage, host: &[f32]) -> Result<()> {
        let rs = as_rocm(storage)?;
        let stream = self.active_stream();
        rs.write_host_f32_async(host, stream)
    }

    /// SPEED-ROC: launch a compute kernel on an *explicit* stream (used by
    /// graph capture, which records only calls issued on the capture stream).
    /// Falls back to the aggregate-source JIT path; resolves the function from
    /// the (entry, grid, solution) cache when warm.
    pub fn launch_compute_kernel_on_stream(
        &self,
        entry: &str,
        grid: HipDim3,
        block: HipDim3,
        args: &mut [*mut c_void],
        stream: *mut c_void,
        shared_mem_bytes: usize,
    ) -> Result<*mut c_void> {
        // Kernel must be pre-resolved by a prior eager launch (the graph-capture
        // pattern: eager launch warms the JIT + function cache, then capture
        // replays via this stream-bound entry point).
        let fast_key = (self.intern_str(entry), grid.x, grid.y, None);
        let cached_func = self
            .resolved_kernel_cache
            .read()
            .ok()
            .and_then(|c| c.get(&fast_key).copied())
            .ok_or_else(|| Error::Backend(format!(
                "launch_compute_kernel_on_stream: {entry} not pre-resolved —                  issue an eager launch before capturing"
            )))?;
        if cached_func.is_null() {
            return Err(Error::Backend(format!("{entry}: null cached function")));
        }
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let args_ptr = args.as_mut_ptr();
        check_hip("hipModuleLaunchKernel (stream)", unsafe {
            hipModuleLaunchKernel(
                cached_func,
                grid.x,
                grid.y,
                grid.z,
                block.x,
                block.y,
                block.z,
                shared_mem_bytes as u32,
                stream,
                args_ptr,
                std::ptr::null_mut(),
            )
        })?;
        Ok(stream)
    }

    #[allow(dead_code)]
    pub(crate) fn launch_madam_update_f32(
        &self,
        dx_storage: &RocmStorage,
        weight_storage: &RocmStorage,
        scale_storage: Option<&RocmStorage>,
        m_buffer: &RocmStorage,
        v_buffer: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        step: i32,
    ) -> Result<*mut c_void> {
        let dx_ptr = dx_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("madam_update: dX has no device ptr".into()))?;
        let w_ptr = weight_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("madam_update: weight has no device ptr".into()))?;
        let m_ptr = m_buffer
            .device_ptr
            .ok_or_else(|| Error::Backend("madam_update: m_buffer has no device ptr".into()))?;
        let v_ptr = v_buffer
            .device_ptr
            .ok_or_else(|| Error::Backend("madam_update: v_buffer has no device ptr".into()))?;
        let scale_ptr: *const std::ffi::c_void = scale_storage
            .and_then(|s| s.device_ptr)
            .map(|p| p as *const std::ffi::c_void)
            .unwrap_or(std::ptr::null());

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(k as u64)
            .ok_or_else(|| Error::Backend("madam_update: m*k overflow".into()))?;
        let grid_x: u32 = (total_elems.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| {
                Error::Backend(format!(
                    "madam_update: grid too large for u32 ({} blocks)",
                    total_elems / BLOCK_SIZE as u64
                ))
            })?;

        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut dxptr = dx_ptr;
        let mut wptr = w_ptr;
        let mut sptr = scale_ptr;
        let mut mptr = m_ptr;
        let mut vptr = v_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let mut lr_f = lr;
        let mut b1 = beta1;
        let mut b2 = beta2;
        let mut ep = eps;
        let mut stp = step;

        self.launch_compute_kernel(
            "grim_madam_update_f32",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut dxptr),
                arg(&mut wptr),
                arg(&mut sptr),
                arg(&mut mptr),
                arg(&mut vptr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut lr_f),
                arg(&mut b1),
                arg(&mut b2),
                arg(&mut ep),
                arg(&mut stp),
            ],
        )?;
        Ok(std::ptr::null_mut())
    }

    /// Launch the JIT compiled SplitK reduction kernel (WI-D).
    pub(crate) fn launch_split_k_reduction(
        &self,
        partials_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        split_k: u32,
    ) -> Result<*mut c_void> {
        let partials_ptr = partials_storage.device_ptr.ok_or_else(|| {
            Error::Backend("split_k_reduction: partials has no device ptr".into())
        })?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("split_k_reduction: out has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 256;
        let total_elems = m * n;
        let grid_x = (total_elems.div_ceil(BLOCK_SIZE)) as u32;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut p_ptr = partials_ptr;
        let mut o_ptr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut sk = split_k as i32;

        // The reduction entry point must match the partials' element type: the historical f16-only kernel silently corrupted every
        // F32/BF16 split-K GEMM (f32 partials read as _Float16, f16 bits written back into the f32 output buffer).
        let entry = match partials_storage.dtype.arith {
            ArithType::F32 => "grim_split_k_reduction_f32",
            ArithType::BF16 => "grim_split_k_reduction_bf16",
            _ => "grim_split_k_reduction",
        };
        self.launch_compute_kernel(
            entry,
            grid_dim,
            block_dim,
            &mut [
                arg(&mut p_ptr),
                arg(&mut o_ptr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut sk),
            ],
        )
    }

    /// JIT-compile or query the cache, then launch the specified kernel on a [see: `entry`, `module_cache`]
    pub(crate) fn launch_compute_kernel(
        &self,
        entry: &str,
        grid: HipDim3,
        block: HipDim3,
        args: &mut [*mut c_void],
    ) -> Result<*mut c_void> {
        self.launch_compute_kernel_with_solution(entry, grid, block, args, None, 0)
    }

    /// JIT compile source or fetch cached binary.
    /// When a `HardwareSpec` is supplied, the cache key incorporates the hardware fingerprint (wavefront/lds/cu/mp/threads) via `JitCacheKey::from_spec`,.
    pub fn jit_compile_or_cache(
        &self,
        source: &str,
        entry: &str,
        spec: Option<&crate::device::hardware_spec::HardwareSpec>,
    ) -> Result<(std::path::PathBuf, String)> {
        if std::env::var("GRIM_ALLOC_TRACE").is_ok() {
            eprintln!("[jit-trace] compiling entry={}", entry);
        }
        let hash = seahash::hash(source.as_bytes());
        let cache_key = if let Some(spec) = spec {
            crate::kernels::jit_cache::JitCacheKey::from_spec(entry, &self.gpu_target, spec, hash)
                .to_key_string()
        } else {
            format!("grim_{}_{}_{:016x}", entry, self.gpu_target, hash)
        };

        if let Some((cached_path, cached_lowered)) = self
            .hsaco_cache
            .get_cached_kernel_hashed(&cache_key, hash)
        {
            if std::env::var_os("GRIM_RING_DIAG").is_some() {
                eprintln!(
                    "[prov] {} entry={} DISK-HIT key={}",
                    self.ordinal, entry, cache_key
                );
            }
            Ok((cached_path, cached_lowered))
        } else {
            if std::env::var_os("GRIM_RING_DIAG").is_some() {
                eprintln!(
                    "[prov] {} entry={} FRESH-COMPILE key={}",
                    self.ordinal, entry, cache_key
                );
            }
            let (code, lowered) = jit_compile_hsaco(source, entry, &self.gpu_target)?;
            let p = self
                .hsaco_cache
                .cache_kernel(&cache_key, source, &code, &lowered)?;
            Ok((p, lowered.to_string()))
        }
    }

    /// Benchmark kernel execution time in milliseconds using HIP events.
    /// Loads the module, resolves the entry, launches once on the device stream bracketed by start/stop.
    pub fn time_kernel_ms(
        &self,
        hsaco: &std::path::Path,
        lowered: &str,
        dims: crate::kernels::tile_picker::ShapeDims,
        cand: &crate::kernels::tile_picker::TileConfig,
    ) -> f64 {
        use crate::device::handles::{
            hipEventCreate, hipEventDestroy, hipEventElapsedTime, hipEventRecord,
            hipEventSynchronize, hipModuleGetFunction, hipModuleLaunchKernel, hipModuleLoad,
            hipModuleUnload,
        };
        // P1-3: module load, events and the launch all bind to the calling thread's current device -
        // pin to the owning ordinal so autotune timing runs on the device it is tuning for.
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let mut start_event: *mut c_void = std::ptr::null_mut();
        let mut stop_event: *mut c_void = std::ptr::null_mut();

        unsafe {
            if hipEventCreate(&mut start_event) != hipSuccess {
                return 0.5;
            }
            if hipEventCreate(&mut stop_event) != hipSuccess {
                let _ = hipEventDestroy(start_event);
                return 0.5;
            }

            let _ = hipEventRecord(start_event, std::ptr::null_mut());

            let grid = HipDim3::new(
                dims.m.div_ceil(cand.grid_stride_m),
                dims.n.div_ceil(cand.grid_stride_n),
                1,
            );
            let block = HipDim3::new(cand.threads, 1, 1);

            let path_c = match std::ffi::CString::new(hsaco.to_str().unwrap_or("")) {
                Ok(c) => c,
                Err(_) => {
                    let _ = hipEventDestroy(start_event);
                    let _ = hipEventDestroy(stop_event);
                    return 0.5;
                }
            };
            let entry_c = match std::ffi::CString::new(lowered) {
                Ok(c) => c,
                Err(_) => {
                    let _ = hipEventDestroy(start_event);
                    let _ = hipEventDestroy(stop_event);
                    return 0.5;
                }
            };

            let mut module: *mut c_void = std::ptr::null_mut();
            if hipModuleLoad(&mut module, path_c.as_ptr()) == hipSuccess {
                let mut func: *mut c_void = std::ptr::null_mut();
                if hipModuleGetFunction(&mut func, module, entry_c.as_ptr()) == hipSuccess {
                    let mut dummy_args: [*mut c_void; 0] = [];
                    let _ = hipModuleLaunchKernel(
                        func,
                        grid.x,
                        grid.y,
                        grid.z,
                        block.x,
                        block.y,
                        block.z,
                        0,
                        std::ptr::null_mut(),
                        dummy_args.as_mut_ptr(),
                        std::ptr::null_mut(),
                    );
                }
                let _ = hipModuleUnload(module);
            }

            let _ = hipEventRecord(stop_event, std::ptr::null_mut());
            let _ = hipEventSynchronize(stop_event);

            let mut elapsed_ms: f32 = 0.0;
            let status = hipEventElapsedTime(&mut elapsed_ms, start_event, stop_event);

            let _ = hipEventDestroy(start_event);
            let _ = hipEventDestroy(stop_event);

            if status == hipSuccess && elapsed_ms > 0.0 {
                elapsed_ms as f64
            } else {
                0.5
            }
        }
    }

    /// Store empirically discovered winning tile configuration into the autotuner cache.
    /// `winner_ms` is the measured GPU time of the winning candidate; persisted as `cycles_per_invocation` (ns-scale u64).
    pub(crate) fn intern_str(&self, s: &str) -> &'static str {
        if let Ok(mut set) = self.str_interner.lock() {
            if let Some(existing) = set.get(s) {
                return existing;
            }
            let leaked: &'static str = Box::leak(s.to_string().into_boxed_str());
            set.insert(leaked);
            leaked
        } else {
            // Poisoned interner: fall back to a one-shot leak (correctness
            // over boundedness).
            Box::leak(s.to_string().into_boxed_str())
        }
    }

    pub fn store_tune_cache(
        &self,
        entry: &str,
        _spec: &crate::device::hardware_spec::HardwareSpec,
        dims: crate::kernels::tile_picker::ShapeDims,
        winner: &crate::kernels::tile_picker::TileConfig,
        winner_ms: f64,
    ) {
        let mut autotuner = self.autotuner.lock().unwrap_or_else(|e| e.into_inner());
        // &'static str keys via the interner — one leak per unique
        // (entry, arch) pair, not per call.
        let arch_leak: &'static str = self.intern_str(&self.gpu_target);
        let entry_leak: &'static str = self.intern_str(entry);
        let key = crate::autotune::KernelKey {
            kernel: entry_leak,
            gpu_arch: arch_leak,
            m: dims.m as usize,
            n: dims.n as usize,
            k: dims.k as usize,
        };
        let config = crate::autotune::AutotuneConfig {
            block_dim: winner.threads,
            tile_kv: winner.block_k,
            grid_stride: winner.grid_stride_m,
            cycles_per_invocation: (winner_ms * 1e6) as u64,
            spec_gamma: 4,
            spec_acceptance_threshold: 0.6,
            spec_alpha: 0.0,
            split_k: winner.split_k,
        };
        let _ = autotuner.record(key, config);
    }

    /// SPEED-ROC-3: read-only lookup of the persisted GEMM autotune table for
    /// the canonical workload entries (`grim_decode_gemm`, `grim_prefill_gemm`,
    /// `grim_lm_head`). Unlike `get_or_tune_tiles`, this NEVER triggers the
    /// FCP search — on a miss the caller falls back to the static heuristic
    /// table. Only fields the rocBLAS dispatch consumes (split_k) are honored;
    /// older tables without a recorded `split_k` (0) are ignored.
    pub(crate) fn lookup_tuned_gemm_split_k(
        &self,
        entry: &'static str,
        m: usize,
        n: usize,
        k: usize,
    ) -> Option<u32> {
        let autotuner = self.autotuner.lock().ok()?;
        let arch_leak: &'static str = self.intern_str(&self.gpu_target);
        let key = crate::autotune::KernelKey {
            kernel: entry,
            gpu_arch: arch_leak,
            m,
            n,
            k,
        };
        let split_k = autotuner.lookup(key)?.split_k;
        (split_k > 1).then_some(split_k)
    }

    /// Persist the in-memory autotune cache to a JSON file at `path`.
    pub fn save_autotune_cache(&self, path: &std::path::Path) -> Result<()> {
        let autotuner = self.autotuner.lock().unwrap_or_else(|e| e.into_inner());
        autotuner.save_to_file(path)
    }

    /// Return a fresh HardwareSpec snapshot describing this device.
    pub fn hardware_spec(&self) -> crate::device::hardware_spec::HardwareSpec {
        crate::device::hardware_spec::HardwareSpec::from(self)
    }

    /// Read-through tile-cache lookup. On a hit, maps the stored `AutotuneConfig` back to a `TileConfig`.
    pub fn get_or_tune_tiles(
        &self,
        entry: &str,
        spec: &crate::device::hardware_spec::HardwareSpec,
        dims: crate::kernels::tile_picker::ShapeDims,
        shape_class: crate::autotune::ShapeClass,
    ) -> crate::kernels::tile_picker::TileConfig {
        // Same interning as store_tune_cache — one leak per unique (entry, arch).
        let arch_leak: &'static str = self.intern_str(&self.gpu_target);
        let entry_leak: &'static str = self.intern_str(entry);
        let key = crate::autotune::KernelKey {
            kernel: entry_leak,
            gpu_arch: arch_leak,
            m: dims.m as usize,
            n: dims.n as usize,
            k: dims.k as usize,
        };

        // 1. Hot path: in-memory table hit.
        {
            let autotuner = self.autotuner.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(cfg) = autotuner.lookup(key) {
                return crate::kernels::tile_picker::TileConfig {
                    block_m: 0,
                    block_n: 0,
                    block_k: cfg.tile_kv,
                    split_k: 1,
                    grid_stride_m: cfg.grid_stride,
                    grid_stride_n: cfg.grid_stride,
                    lds_double_buffer: 64 * 1024
                        >= 2 * (2
                            * (cfg.tile_kv * (spec.wavefront_size.max(16))
                                + cfg.tile_kv * (spec.wavefront_size.max(16))
                                + (spec.wavefront_size.max(16)) * (spec.wavefront_size.max(16)))),
                    use_wmma: spec.gcn_arch.starts_with("gfx11")
                        || spec.gcn_arch.starts_with("gfx12"),
                    use_mfma: spec.gcn_arch.starts_with("gfx12")
                        || spec.gcn_arch.starts_with("gfx9"),
                    threads: cfg.block_dim,
                }
                .with_block_geometry(spec, shape_class);
            }
        }

        // 2. Cold path: empirical FCP search. Self-persists, so the next call hits step 1.
        crate::kernels::tile_picker::fcp_fallback_tile_search(self, spec, entry, dims, shape_class)
    }

    /// WI-M2/M3: stamp (self_dev, ctx_dev) for the drift gates and print the launch trace - called while this
    /// device's P1-3 guard is held, so the recorded `ctx_dev` is the context the kernel actually launches under.
    fn stamp_launch_post_pin(&self, trace_on: bool, entry: &str, grid: HipDim3) {
        if !(trace_on || cfg!(test)) {
            return;
        }
        let mut cur_dev: i32 = -1;
        unsafe {
            crate::device::handles::hipGetDevice(&mut cur_dev);
        }
        #[cfg(test)]
        crate::device::util::stamp_launch_context(self.ordinal as i32, cur_dev);
        if trace_on {
            eprintln!(
                "[launch-trace] self_dev={} ctx_dev={} {} grid=({},{},{})",
                self.ordinal, cur_dev, entry, grid.x, grid.y, grid.z
            );
        }
    }

    pub(crate) fn launch_compute_kernel_with_solution(
        &self,
        entry: &str,
        grid: HipDim3,
        block: HipDim3,
        args: &mut [*mut c_void],
        solution_index: Option<i32>,
        shared_mem_bytes: usize,
    ) -> Result<*mut c_void> {
        // Fast path: a previously resolved hipFunction for this (entry, grid-shape, solution_index) launches directly - no source rebuild, no seahash, no CString, no module-cache walk.
        // Same solution_index is required because different indices map to different on-disk hsaco files (cache_key includes.
        // SPEED-CEREMONY: the env probe used to run TWICE per launch (~1-3 µs each
        // on glibc's environ lock) — hoisted to a one-time process static.
        static ALLOC_TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let trace_on = *ALLOC_TRACE.get_or_init(|| std::env::var("GRIM_ALLOC_TRACE").is_ok());
        if trace_on {
            eprintln!("[launch-done] {}", entry);
        }
        // SPEED-ROC-12: intern once per unique entry (one leaked &str) — the
        // old `entry.to_string()` heap-allocated on every launch lookup.
        let fast_key = (self.intern_str(entry), grid.x, grid.y, solution_index);
        let cached_func: Option<*mut c_void> = self
            .resolved_kernel_cache
            .read()
            .ok()
            .and_then(|c| c.get(&fast_key).copied());
        if let Some(func) = cached_func {
            if !func.is_null() {
                // P1-3 discipline: the launching thread's HIP context may be parked on another device (profiler probes, multi-device loaders).
                // Pin THIS device or the kernel executes against foreign pointers - observed as GPU page.
                let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
                self.stamp_launch_post_pin(trace_on, entry, grid);
                let stream = self.active_stream();
                let args_ptr = args.as_mut_ptr();
                check_hip("hipModuleLaunchKernel (cached)", unsafe {
                    hipModuleLaunchKernel(
                        func,
                        grid.x,
                        grid.y,
                        grid.z,
                        block.x,
                        block.y,
                        block.z,
                        shared_mem_bytes as u32,
                        stream,
                        args_ptr,
                        std::ptr::null_mut(),
                    )
                })?;
                self.launch_counter.fetch_add(1, Ordering::Relaxed);
                drop(_dev_guard);
                return Ok(stream);
            }
        }

        // Build the kernel source. Under `jit-hw-adaptive`, inject hardware-specific #defines (wavefront/LDS/CU +
        // tile geometry) via `compute_kernel_source_with_spec` and route the compile through the fingerprinted `jit_compile_or_cache`.
        #[cfg(feature = "jit-hw-adaptive")]
        let (path, lowered_name, cache_key) = {
            let spec = self.hardware_spec();
            // `launch_compute_kernel` is a generic launcher (no GEMM M/N/K in its signature), so infer a coarse (m, n) from the grid dims; the per-op TLOLog tagging is handled at the `matmul_op` layer, not here.
            // K is unknown to the generic launcher; use a conservative default - split-K is derived.
            let (m_val, n_val) = if grid.y > 1 { (grid.x, grid.y) } else { (1, 1) };
            let shape_class = crate::autotune::ShapeClass::from_m(m_val as usize);
            let dims = crate::kernels::tile_picker::ShapeDims::new(m_val, n_val, 64);
            let kernel_source = crate::kernels::source_asm::compute_kernel_source_with_spec(
                &spec,
                entry,
                shape_class,
                dims,
                0,
                1,
                None,
            );
            let (p, lowered) = self.jit_compile_or_cache(&kernel_source, entry, Some(&spec))?;
            let mut key = format!(
                "grim_{}_{}_{:016x}",
                entry,
                self.gpu_target,
                seahash::hash(kernel_source.as_bytes())
            );
            if let Some(sol) = solution_index {
                key = format!("{}_sol{}", key, sol);
            }
            (p, lowered, key)
        };

        #[cfg(not(feature = "jit-hw-adaptive"))]
        let (path, lowered_name, cache_key) = {
            let kernel_source = crate::kernels::source_asm::compute_kernel_source();
            let hash = seahash::hash(kernel_source.as_bytes());
            let base_key = format!("grim_{}_{}_{:016x}", entry, self.gpu_target, hash);
            let cache_key = if let Some(sol) = solution_index {
                format!("{}_sol{}", base_key, sol)
            } else {
                base_key
            };
            let (path, lowered_name) = if let Some((cached_path, cached_lowered)) =
                self.hsaco_cache
                    .get_cached_kernel_hashed(&cache_key, hash)
            {
                (cached_path, cached_lowered)
            } else {
                let (code, lowered) = jit_compile_hsaco(&kernel_source, entry, &self.gpu_target)?;
                let p =
                    self.hsaco_cache
                        .cache_kernel(&cache_key, &kernel_source, &code, &lowered)?;
                (p, lowered)
            };
            (path, lowered_name, cache_key)
        };

        let path_c = std::ffi::CString::new(path.to_str().unwrap_or(""))
            .map_err(|e| Error::Backend(format!("hsaco path CString: {}", e)))?;
        let entry_c = std::ffi::CString::new(lowered_name.as_str())
            .map_err(|e| Error::Backend(format!("entry CString: {}", e)))?;

        // Load the HIP module once per unique kernel; reuse the cached module + Pin the current device to self.ordinal before loading: the JIT pipeline queries CapabilityProfiler which sweeps every device and can leave the thread on a foreign ordinal.
        // Loading a gfx1201 hsaco on gfx1200 yields HIP error 209 (no binary for device).
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        self.stamp_launch_post_pin(trace_on, entry, grid);
        let mut module_cache = self.module_cache.lock().unwrap_or_else(|e| e.into_inner());
        let (_module, func) = if let Some(cached) = module_cache.get(&cache_key) {
            let (m, f) = *cached;
            if let Ok(mut fast) = self.resolved_kernel_cache.write() {
                fast.insert(fast_key, f);
            }
            (m, f)
        } else {
            let mut module: *mut c_void = std::ptr::null_mut();
            let load_res = unsafe { hipModuleLoad(&mut module, path_c.as_ptr()) };
            if load_res != hipSuccess {
                // Report the real HIP context, not just our cached ordinal: a
                // mismatch here means the thread was not actually on `self.ordinal`,
                // which is how a `gfx1201` object gets loaded on `gfx1200` (status 209).
                let mut ctx_dev: i32 = -1;
                unsafe {
                    let _ = crate::device::handles::hipGetDevice(&mut ctx_dev);
                }
                let ctx_arch = crate::device::util::detect_gpu_arch(ctx_dev);
                let hsa_override = std::env::var("HSA_OVERRIDE_GFX_VERSION").ok();
                return Err(Error::Backend(format!(
                    "hipModuleLoad failed: {load_res} (entry={entry}, path={}, \
                     gpu_target={}, self.ordinal={}, ctx_device={ctx_dev}, \
                     ctx_arch={ctx_arch}, hsa_override={hsa_override:?})",
                    path.display(),
                    self.gpu_target,
                    self.ordinal
                )));
            }
            let mut func: *mut c_void = std::ptr::null_mut();
            let res = unsafe { hipModuleGetFunction(&mut func, module, entry_c.as_ptr()) };
            if res != hipSuccess {
                unsafe {
                    hipModuleUnload(module);
                }
                return Err(Error::Backend(format!(
                    "hipModuleGetFunction failed: {}",
                    res
                )));
            }
            self.module_load_count.fetch_add(1, Ordering::SeqCst);
            module_cache.insert(cache_key, (module, func));
            if let Ok(mut fast) = self.resolved_kernel_cache.write() {
                fast.insert(fast_key, func);
            }
            (module, func)
        };
        drop(module_cache);

        let stream = self.active_stream();

        let args_ptr = args.as_mut_ptr();
        check_hip("hipModuleLaunchKernel", unsafe {
            hipModuleLaunchKernel(
                func,
                grid.x,
                grid.y,
                grid.z,
                block.x,
                block.y,
                block.z,
                shared_mem_bytes as u32,
                stream,
                args_ptr,
                std::ptr::null_mut(),
            )
        })?;
        self.launch_counter.fetch_add(1, Ordering::Relaxed);
        Ok(stream)
    }
}
