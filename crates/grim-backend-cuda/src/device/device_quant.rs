//! Quantization operations and quantized GEMM dispatch for `CudaDevice`.

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use grim_tensor::backend::ComputeHandle;
use grim_tensor::dtype::{
    ArithType, DType, FloatPackScheme, KQuantScheme, Storage as DTypeStorage,
};
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendStorage, CoreTensorOps, QuantOps, Shape};

use crate::device::cuda_device::CudaDevice;
use crate::device::handles::{
    cublasSgemm_v2, cuLaunchKernel, cuModuleGetFunction, cudaMemcpy, cudaMemcpyDeviceToHost,
    CUBLAS_OP_N, CUBLAS_STATUS_SUCCESS, CUfunction, CudaHandle,
};
use crate::device::jit_cache::compile_and_load_kernel;
use crate::memory::storage::{
    cuda_dequant_quantized_storage, read_length_prefixed, stage_packed_bytes, CudaStorage,
};

impl CudaDevice {
    /// Launches the fused Q8_0 quantized GEMM on a 2-D grid.
    /// out[M,N] = a[M,K] · b_q8[K,N]; b is int8 raw bytes with per-32-element
    /// block scales; requires K % 32 == 0. Runs on the default stream.
    fn launch_quantized_matmul_q8_0(
        &self,
        a_ptr: *const c_void,
        b_ptr: *const c_void,
        b_scales_ptr: *const c_void,
        out_ptr: *mut c_void,
        m: usize,
        n: usize,
        k: usize,
        b_data_offset: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            let func_name = std::ffi::CString::new("grim_quantized_matmul_q8_0")
                .map_err(|e| Error::Backend(format!("invalid kernel name: {e}")))?;
            let res = cuModuleGetFunction(&mut func, module, func_name.as_ptr());
            if res != 0 {
                return Err(Error::Backend(format!(
                    "cuModuleGetFunction(grim_quantized_matmul_q8_0) failed: {res}"
                )));
            }

            let mut a_arg = a_ptr;
            let mut b_arg = b_ptr;
            let mut bs_arg = b_scales_ptr;
            let mut out_arg = out_ptr;
            let mut m_arg = m as i32;
            let mut n_arg = n as i32;
            let mut k_arg = k as i32;
            let mut bdo_arg = b_data_offset as i32;
            let mut args: [*mut c_void; 8] = [
                &mut a_arg as *mut *const c_void as *mut c_void,
                &mut b_arg as *mut *const c_void as *mut c_void,
                &mut bs_arg as *mut *const c_void as *mut c_void,
                &mut out_arg as *mut *mut c_void as *mut c_void,
                &mut m_arg as *mut i32 as *mut c_void,
                &mut n_arg as *mut i32 as *mut c_void,
                &mut k_arg as *mut i32 as *mut c_void,
                &mut bdo_arg as *mut i32 as *mut c_void,
            ];

            const BLOCK_X: u32 = 32;
            const BLOCK_Y: u32 = 8;
            let grid_x = (n as u32).div_ceil(BLOCK_X);
            let grid_y = (m as u32).div_ceil(BLOCK_Y);

            let launch_res = cuLaunchKernel(
                func,
                grid_x,
                grid_y,
                1,
                BLOCK_X,
                BLOCK_Y,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr() as *mut *mut c_void,
                std::ptr::null_mut(),
            );
            if launch_res != 0 {
                return Err(Error::Backend(format!(
                    "cuLaunchKernel(grim_quantized_matmul_q8_0) failed: {launch_res}"
                )));
            }
        }
        Ok(Box::new(CudaHandle {
            completed: Arc::new(Mutex::new(false)),
        }))
    }

    /// Launches a standalone dequant kernel of signature
    /// `(const u8* packed, float* out, int n_blocks)` — one thread per
    /// 256-weight super-block. Used by the Q5_K/Q6_K/IQ4/IQ3/IQ2 family.
    /// `n` is the number of super-blocks; grid = ceil(n/256), block = 256.
    fn launch_dequant_generic(
        &self,
        kernel_name: &str,
        packed_ptr: *const c_void,
        out_ptr: *mut c_void,
        n_blocks: usize,
        weights_per_block: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            let func_name = std::ffi::CString::new(kernel_name)
                .map_err(|e| Error::Backend(format!("invalid kernel name {kernel_name:?}: {e}")))?;
            let res = cuModuleGetFunction(&mut func, module, func_name.as_ptr());
            if res != 0 {
                return Err(Error::Backend(format!(
                    "cuModuleGetFunction({kernel_name}) failed: {res}"
                )));
            }

            let mut packed = packed_ptr;
            let mut out = out_ptr;
            let mut n_blk = n_blocks as i32;
            let mut args: [*mut c_void; 3] = [
                &mut packed as *mut *const c_void as *mut c_void,
                &mut out as *mut *mut c_void as *mut c_void,
                &mut n_blk as *mut i32 as *mut c_void,
            ];

            const BLOCK_SIZE: u32 = 256;
            // The dequantization kernels (grim_dequant_q8_0, etc.) expect one
            // thread per output weight, checking `id >= n_blocks * 32` (or
            // weights_per_block). So the grid must cover n_blocks *
            // weights_per_block threads, not just n_blocks.
            let total_weights = n_blocks.checked_mul(weights_per_block).ok_or_else(|| {
                Error::Backend("launch_dequant_generic: total weight count overflow".into())
            })?;
            let grid_size =
                ((total_weights as u64) + (BLOCK_SIZE as u64) - 1) / (BLOCK_SIZE as u64);
            let launch_res = cuLaunchKernel(
                func,
                grid_size as u32,
                1,
                1,
                BLOCK_SIZE,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr() as *mut *mut c_void,
                std::ptr::null_mut(),
            );
            if launch_res != 0 {
                return Err(Error::Backend(format!(
                    "cuLaunchKernel({kernel_name}) failed: {launch_res}"
                )));
            }
        }
        Ok(Box::new(CudaHandle {
            completed: Arc::new(Mutex::new(false)),
        }))
    }

    /// Launches `grim_dequant_fp8(packed, out, n_weights)` — one thread per
    /// weight. The first 4 bytes of `packed` are the LE f32 global scale.
    fn launch_dequant_fp8(
        &self,
        packed_ptr: *const c_void,
        out_ptr: *mut c_void,
        n_weights: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        if !self
            .caps
            .supports_quant_format(grim_tensor::dtype::QuantFormat::Fp8)
        {
            return Err(Error::Backend(format!(
                "FP8 dequantization is not supported on CUDA Compute Capability {}.{} (requires >= 8.9 / Ada)",
                self.caps.compute_major, self.caps.compute_minor
            )));
        }
        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            let func_name = std::ffi::CString::new("grim_dequant_fp8")
                .map_err(|e| Error::Backend(format!("invalid kernel name: {e}")))?;
            let res = cuModuleGetFunction(&mut func, module, func_name.as_ptr());
            if res != 0 {
                return Err(Error::Backend(format!(
                    "cuModuleGetFunction(grim_dequant_fp8) failed: {res}"
                )));
            }
            let mut packed = packed_ptr;
            let mut out = out_ptr;
            let mut n = n_weights as i32;
            let mut args: [*mut c_void; 3] = [
                &mut packed as *mut *const c_void as *mut c_void,
                &mut out as *mut *mut c_void as *mut c_void,
                &mut n as *mut i32 as *mut c_void,
            ];
            const BLOCK_SIZE: u32 = 256;
            let grid_size = ((n_weights as u64) + (BLOCK_SIZE as u64) - 1) / (BLOCK_SIZE as u64);
            let launch_res = cuLaunchKernel(
                func,
                grid_size as u32,
                1,
                1,
                BLOCK_SIZE,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr() as *mut *mut c_void,
                std::ptr::null_mut(),
            );
            if launch_res != 0 {
                return Err(Error::Backend(format!(
                    "cuLaunchKernel(grim_dequant_fp8) failed: {launch_res}"
                )));
            }
        }
        Ok(Box::new(CudaHandle {
            completed: Arc::new(Mutex::new(false)),
        }))
    }

    /// Launches `grim_dequant_mxfp4(codes, exps, out, n_values)` — one thread
    /// per value. `codes` is packed E2M1 nibbles (2/byte); `exps` holds one
    /// E8M0 shared exponent per 32-element group.
    fn launch_dequant_mxfp4(
        &self,
        codes_ptr: *const c_void,
        exps_ptr: *const c_void,
        out_ptr: *mut c_void,
        n_values: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        self.launch_dequant_mxfp_kernel(
            "grim_dequant_mxfp4",
            codes_ptr,
            exps_ptr,
            out_ptr,
            n_values,
        )
    }

    /// Launches `grim_dequant_mxfp8(codes, exps, out, n_values)` — one thread
    /// per value. `codes` is packed E4M3 bytes; `exps` holds one E8M0 shared exponent per 32-element group.
    fn launch_dequant_mxfp8(
        &self,
        codes_ptr: *const c_void,
        exps_ptr: *const c_void,
        out_ptr: *mut c_void,
        n_values: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        self.launch_dequant_mxfp_kernel(
            "grim_dequant_mxfp8",
            codes_ptr,
            exps_ptr,
            out_ptr,
            n_values,
        )
    }

    fn launch_dequant_mxfp_kernel(
        &self,
        kernel_name: &str,
        codes_ptr: *const c_void,
        exps_ptr: *const c_void,
        out_ptr: *mut c_void,
        n_values: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            let func_name = std::ffi::CString::new(kernel_name)
                .map_err(|e| Error::Backend(format!("invalid kernel name: {e}")))?;
            let res = cuModuleGetFunction(&mut func, module, func_name.as_ptr());
            if res != 0 {
                return Err(Error::Backend(format!(
                    "cuModuleGetFunction({kernel_name}) failed: {res}"
                )));
            }
            let mut codes = codes_ptr;
            let mut exps = exps_ptr;
            let mut out = out_ptr;
            let mut n = n_values as i32;
            let mut args: [*mut c_void; 4] = [
                &mut codes as *mut *const c_void as *mut c_void,
                &mut exps as *mut *const c_void as *mut c_void,
                &mut out as *mut *mut c_void as *mut c_void,
                &mut n as *mut i32 as *mut c_void,
            ];
            const BLOCK_SIZE: u32 = 256;
            let grid_size = ((n_values as u64) + (BLOCK_SIZE as u64) - 1) / (BLOCK_SIZE as u64);
            let launch_res = cuLaunchKernel(
                func,
                grid_size as u32,
                1,
                1,
                BLOCK_SIZE,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr() as *mut *mut c_void,
                std::ptr::null_mut(),
            );
            if launch_res != 0 {
                return Err(Error::Backend(format!(
                    "cuLaunchKernel({kernel_name}) failed: {launch_res}"
                )));
            }
        }
        Ok(Box::new(CudaHandle {
            completed: Arc::new(Mutex::new(false)),
        }))
    }

    /// Dequantize a CUDA-resident packed `CudaStorage` to a new F32 `CudaStorage`
    /// entirely on device, returning the F32 storage. Falls back to
    /// `Err(Backend)` for block types without a device kernel so the caller
    /// (`to_cpu_vec_f32`) can use the bit-accurate host path. This is the
    /// primary mechanism by which the quantized GEMM path stays on-GPU.
    ///
    /// Block byte-size table (matches `grim_quant::dequant_*`):
    ///   Q5_K=176, Q6_K=210, IQ4_NL=170, IQ4_XS=136, IQ3_XXS=96,
    ///   IQ3_S=110, IQ2_XXS=66, IQ2_XS=74, IQ2_S=82. All are 256 weights/block.
    pub fn dequantize_on_device(&self, packed: &CudaStorage) -> Result<CudaStorage> {
        let elem_count = packed.shape.elem_count();
        let packed_ptr = Self::dev_ptr_or_err("dequantize_on_device", packed)? as *const c_void;

        let (kernel, block_bytes, weights_per_block): (&str, usize, usize) =
            match &packed.dtype.storage {
                DTypeStorage::KQuant(scheme) => match scheme {
                    KQuantScheme::Q4K => ("grim_dequant_q4k", 144, 256),
                    KQuantScheme::Q80 => ("grim_dequant_q8_0", 34, 32),
                    KQuantScheme::Q5K => ("grim_dequant_q5k", 176, 256),
                    KQuantScheme::Q6K => ("grim_dequant_q6k", 210, 256),
                    KQuantScheme::IQ4NL => ("grim_dequant_iq4nl", 144, 256),
                    KQuantScheme::IQ4XS => ("grim_dequant_iq4xs", 136, 256),
                    KQuantScheme::IQ3XXS => ("grim_dequant_iq3xxs", 96, 256),
                    KQuantScheme::IQ3S => ("grim_dequant_iq3s", 110, 256),
                    KQuantScheme::IQ2XXS => ("grim_dequant_iq2xxs", 66, 256),
                    KQuantScheme::IQ2XS => ("grim_dequant_iq2xs", 74, 256),
                    KQuantScheme::IQ2S => ("grim_dequant_iq2s", 82, 256),
                    _ => {
                        return Err(Error::Backend(format!(
                            "dequantize_on_device: no GPU kernel for KQuant {:?}",
                            scheme
                        )));
                    }
                },
                DTypeStorage::FloatPack(FloatPackScheme::Fp8) => {
                    // FP8: 4-byte f32 scale header + 1 byte/weight. n_weights = elem_count.
                    let out = CudaStorage::alloc_gpu(&packed.shape, DType::F32, self.ordinal)?;
                    let out_ptr = Self::dev_ptr_or_err("dequantize_on_device(fp8)", &out)?;
                    let handle = self.launch_dequant_fp8(packed_ptr, out_ptr, elem_count)?;
                    handle.synchronize()?;
                    return Ok(out);
                }
                DTypeStorage::FloatPack(FloatPackScheme::MxFp4)
                | DTypeStorage::FloatPack(FloatPackScheme::MxFp8) => {
                    let is_mxfp4 = matches!(
                        packed.dtype.storage,
                        DTypeStorage::FloatPack(FloatPackScheme::MxFp4)
                    );
                    let raw = stage_packed_bytes(packed)?;
                    let mut cursor = 0usize;
                    let codes = read_length_prefixed(&raw, &mut cursor)?;
                    let exps = read_length_prefixed(&raw, &mut cursor)?;
                    let num_groups = elem_count.div_ceil(32);
                    if exps.len() < num_groups {
                        return Err(Error::Backend(format!(
                            "dequantize_on_device(mxfp): expected {num_groups} exp bytes, got {}",
                            exps.len()
                        )));
                    }
                    let min_codes_len = if is_mxfp4 {
                        elem_count.div_ceil(2)
                    } else {
                        elem_count
                    };
                    if codes.len() < min_codes_len {
                        return Err(Error::Backend(format!(
                            "dequantize_on_device(mxfp): expected {} code bytes, got {}",
                            min_codes_len,
                            codes.len()
                        )));
                    }
                    let codes_shape_external = Shape::new(vec![codes.len()]);
                    let codes_storage = CudaStorage::copy_from_host_raw_bytes(
                        &codes,
                        &codes_shape_external,
                        DType {
                            arith: ArithType::U8,
                            storage: DTypeStorage::Native,
                        },
                        self.ordinal,
                    )?;
                    let exps_shape = Shape::new(vec![exps.len()]);
                    let exps_storage = CudaStorage::copy_from_host_raw_bytes(
                        &exps,
                        &exps_shape,
                        DType {
                            arith: ArithType::U8,
                            storage: DTypeStorage::Native,
                        },
                        self.ordinal,
                    )?;
                    let out = CudaStorage::alloc_gpu(&packed.shape, DType::F32, self.ordinal)?;
                    let codes_ptr =
                        Self::dev_ptr_or_err("dequantize_on_device(mxfp codes)", &codes_storage)?;
                    let exps_ptr =
                        Self::dev_ptr_or_err("dequantize_on_device(mxfp exps)", &exps_storage)?;
                    let out_ptr = Self::dev_ptr_or_err("dequantize_on_device(mxfp out)", &out)?;
                    let handle = if is_mxfp4 {
                        self.launch_dequant_mxfp4(codes_ptr, exps_ptr, out_ptr, elem_count)?
                    } else {
                        self.launch_dequant_mxfp8(codes_ptr, exps_ptr, out_ptr, elem_count)?
                    };
                    handle.synchronize()?;
                    return Ok(out);
                }
                // FP4/NF4/MXFP8 and Block(Fp4/Nf4/Fp8Block16) keep the host path.
                _ => {
                    return Err(Error::Backend(format!(
                        "dequantize_on_device: no GPU kernel for dtype {:?}",
                        packed.dtype
                    )));
                }
            };

        // Super-block path (Q5_K/Q6_K/IQ*).
        let n_blocks = elem_count.div_ceil(weights_per_block);
        if packed.bytes < n_blocks * block_bytes {
            return Err(Error::Backend(format!(
                "dequantize_on_device({kernel}): packed buffer too short ({} < {}*{})",
                packed.bytes, n_blocks, block_bytes
            )));
        }
        let out = CudaStorage::alloc_gpu(&packed.shape, DType::F32, self.ordinal)?;
        let out_ptr = Self::dev_ptr_or_err("dequantize_on_device(out)", &out)?;
        let handle = self.launch_dequant_generic(kernel, packed_ptr, out_ptr, n_blocks, weights_per_block)?;
        handle.synchronize()?;
        Ok(out)
    }

    /// Quantize a CUDA-resident F32 `CudaStorage` into packed quantized bytes,
    /// entirely on-device — the device-side mirror of `grim_quant::quant_*`.
    ///
    /// Returns a new `CudaStorage` holding the packed bytes with the
    /// appropriate `Storage` dtype. Currently supports Q8_0 and FP8 (E4M3).
    pub fn quantize_on_device(
        &self,
        x: &CudaStorage,
        format: grim_tensor::QuantFormat,
    ) -> Result<CudaStorage> {
        Self::ensure_f32_input("quantize_on_device", x)?;
        let n_weights = x.shape.elem_count();
        let x_ptr = Self::dev_ptr_or_err("quantize_on_device x", x)? as *const c_void;

        match format {
            grim_tensor::QuantFormat::Q8_0 => {
                if n_weights % 32 != 0 {
                    return Err(Error::Backend(format!(
                        "quantize_on_device(Q8_0): n_weights ({n_weights}) must be a multiple of 32"
                    )));
                }
                let n_blocks = n_weights / 32;
                let out_bytes = n_blocks * 34;
                let out_shape = Shape::new(vec![out_bytes]);
                let out = CudaStorage::alloc_gpu_bytes(
                    &out_shape,
                    DType {
                        arith: ArithType::U8,
                        storage: DTypeStorage::KQuant(KQuantScheme::Q80),
                    },
                    out_bytes,
                    self.ordinal,
                )?;
                let out_ptr = Self::dev_ptr_or_err("quantize_on_device(Q8_0 out)", &out)?;
                let handle = self.launch_quant_q8_0(x_ptr, out_ptr, n_blocks)?;
                handle.synchronize()?;
                Ok(out)
            }
            grim_tensor::QuantFormat::Fp8 => {
                // T1 caps gate: a device without native FP8 (compute < 8.9) must not
                // dispatch the fp8 quantize kernel.
                if !self
                    .caps
                    .supports_quant_format(grim_tensor::QuantFormat::Fp8)
                {
                    return Err(Error::Backend(
                        "quantize_on_device: FP8 not supported on this device".into(),
                    ));
                }
                let out_bytes = 4 + n_weights;
                let out_shape = Shape::new(vec![out_bytes]);
                let out = CudaStorage::alloc_gpu_bytes(
                    &out_shape,
                    DType {
                        arith: ArithType::U8,
                        storage: DTypeStorage::FloatPack(FloatPackScheme::Fp8),
                    },
                    out_bytes,
                    self.ordinal,
                )?;
                let out_ptr = Self::dev_ptr_or_err("quantize_on_device(FP8 out)", &out)?;
                let handle = self.launch_quant_fp8(x_ptr, out_ptr, n_weights)?;
                handle.synchronize()?;
                Ok(out)
            }
            other => Err(Error::Backend(format!(
                "quantize_on_device: no GPU kernel for format {other:?}"
            ))),
        }
    }

    /// Launches `grim_quant_q8_0(x, out, n_blocks)` — one 32-thread block per
    /// Q8_0 block. Warp shuffle reduction for amax.
    fn launch_quant_q8_0(
        &self,
        x_ptr: *const c_void,
        out_ptr: *mut c_void,
        n_blocks: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            let func_name = std::ffi::CString::new("grim_quant_q8_0")
                .map_err(|e| Error::Backend(format!("invalid kernel name: {e}")))?;
            let res = cuModuleGetFunction(&mut func, module, func_name.as_ptr());
            if res != 0 {
                return Err(Error::Backend(format!(
                    "cuModuleGetFunction(grim_quant_q8_0) failed: {res}"
                )));
            }

            let mut x = x_ptr;
            let mut out = out_ptr;
            let mut n_blk = n_blocks as i32;
            let mut args: [*mut c_void; 3] = [
                &mut x as *mut *const c_void as *mut c_void,
                &mut out as *mut *mut c_void as *mut c_void,
                &mut n_blk as *mut i32 as *mut c_void,
            ];

            // One block of 32 threads per Q8_0 block.
            const BLOCK_SIZE: u32 = 32;
            let launch_res = cuLaunchKernel(
                func,
                n_blocks as u32,
                1,
                1,
                BLOCK_SIZE,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr() as *mut *mut c_void,
                std::ptr::null_mut(),
            );
            if launch_res != 0 {
                return Err(Error::Backend(format!(
                    "cuLaunchKernel(grim_quant_q8_0) failed: {launch_res}"
                )));
            }
        }
        Ok(Box::new(CudaHandle {
            completed: Arc::new(Mutex::new(false)),
        }))
    }

    /// Launches `grim_quant_fp8(x, out, n_weights)` — one thread per weight.
    /// The first 4 bytes of `out` are the LE f32 scale (1.0f).
    fn launch_quant_fp8(
        &self,
        x_ptr: *const c_void,
        out_ptr: *mut c_void,
        n_weights: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            let func_name = std::ffi::CString::new("grim_quant_fp8")
                .map_err(|e| Error::Backend(format!("invalid kernel name: {e}")))?;
            let res = cuModuleGetFunction(&mut func, module, func_name.as_ptr());
            if res != 0 {
                return Err(Error::Backend(format!(
                    "cuModuleGetFunction(grim_quant_fp8) failed: {res}"
                )));
            }

            let mut x = x_ptr;
            let mut out = out_ptr;
            let mut n = n_weights as i32;
            let mut args: [*mut c_void; 3] = [
                &mut x as *mut *const c_void as *mut c_void,
                &mut out as *mut *mut c_void as *mut c_void,
                &mut n as *mut i32 as *mut c_void,
            ];

            const BLOCK_SIZE: u32 = 256;
            let grid_size = ((n_weights as u64) + (BLOCK_SIZE as u64) - 1) / (BLOCK_SIZE as u64);
            let launch_res = cuLaunchKernel(
                func,
                grid_size as u32,
                1,
                1,
                BLOCK_SIZE,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr() as *mut *mut c_void,
                std::ptr::null_mut(),
            );
            if launch_res != 0 {
                return Err(Error::Backend(format!(
                    "cuLaunchKernel(grim_quant_fp8) failed: {launch_res}"
                )));
            }
        }
        Ok(Box::new(CudaHandle {
            completed: Arc::new(Mutex::new(false)),
        }))
    }

    /// Launches `grim_fused_quant_gemm_{format}(A, B, C, M, N, K)` — one thread
    /// per output element. Grid/block match `launch_quantized_matmul_q8_0`.
    fn launch_fused_quant_gemm(
        &self,
        kernel_name: &str,
        a_ptr: *const c_void,
        b_ptr: *const c_void,
        out_ptr: *mut c_void,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            let func_name = std::ffi::CString::new(kernel_name)
                .map_err(|e| Error::Backend(format!("invalid kernel name: {e}")))?;
            let res = cuModuleGetFunction(&mut func, module, func_name.as_ptr());
            if res != 0 {
                return Err(Error::Backend(format!(
                    "cuModuleGetFunction({kernel_name}) failed: {res}"
                )));
            }

            let mut a = a_ptr;
            let mut b = b_ptr;
            let mut out = out_ptr;
            let mut m_arg = m as i32;
            let mut n_arg = n as i32;
            let mut k_arg = k as i32;
            let mut args: [*mut c_void; 6] = [
                &mut a as *mut *const c_void as *mut c_void,
                &mut b as *mut *const c_void as *mut c_void,
                &mut out as *mut *mut c_void as *mut c_void,
                &mut m_arg as *mut i32 as *mut c_void,
                &mut n_arg as *mut i32 as *mut c_void,
                &mut k_arg as *mut i32 as *mut c_void,
            ];

            const BLOCK_X: u32 = 32;
            const BLOCK_Y: u32 = 8;
            let grid_x = (n as u32).div_ceil(BLOCK_X);
            let grid_y = (m as u32).div_ceil(BLOCK_Y);

            let launch_res = cuLaunchKernel(
                func,
                grid_x,
                grid_y,
                1,
                BLOCK_X,
                BLOCK_Y,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr() as *mut *mut c_void,
                std::ptr::null_mut(),
            );
            if launch_res != 0 {
                return Err(Error::Backend(format!(
                    "cuLaunchKernel({kernel_name}) failed: {launch_res}"
                )));
            }
        }
        Ok(Box::new(CudaHandle {
            completed: Arc::new(Mutex::new(false)),
        }))
    }

    /// Dequantize Q8_0 packed bytes to an f32 host Vec via host / GPU.
    pub fn dequantize_q8_0_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        let packed = CudaStorage::copy_from_host_raw_bytes(
            bytes,
            &Shape::new(vec![elem_count]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::KQuant(KQuantScheme::Q80),
            },
            self.ordinal,
        )?;
        if let Ok(f32_storage) = self.dequantize_on_device(&packed) {
            return f32_storage.to_cpu_vec_f32();
        }
        let mut out = Vec::with_capacity(elem_count);
        for blk in bytes.chunks_exact(34) {
            let d_bits = u16::from_le_bytes([blk[0], blk[1]]);
            let d = half::f16::from_bits(d_bits).to_f32();
            for &q in &blk[2..34] {
                out.push(d * (q as i8 as f32));
            }
        }
        out.truncate(elem_count);
        Ok(out)
    }

    /// Dequantize Q4_K packed bytes to an f32 host Vec via host / GPU.
    pub fn dequantize_q4k_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        let packed = CudaStorage::copy_from_host_raw_bytes(
            bytes,
            &Shape::new(vec![elem_count]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::KQuant(KQuantScheme::Q4K),
            },
            self.ordinal,
        )?;
        if let Ok(f32_storage) = self.dequantize_on_device(&packed) {
            return f32_storage.to_cpu_vec_f32();
        }
        grim_quant::dequant_q4k(bytes, elem_count)
    }

    pub(crate) fn dequantize_iq_host(
        &self,
        bytes: &[u8],
        elem_count: usize,
        scheme: KQuantScheme,
    ) -> Result<Vec<f32>> {
        let packed = CudaStorage::copy_from_host_raw_bytes(
            bytes,
            &Shape::new(vec![elem_count]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::KQuant(scheme),
            },
            self.ordinal,
        )?;
        if let Ok(f32_storage) = self.dequantize_on_device(&packed) {
            return f32_storage.to_cpu_vec_f32();
        }
        match scheme {
            KQuantScheme::IQ2XXS => grim_quant::dequant_iq2xxs(bytes, elem_count),
            KQuantScheme::IQ2XS => grim_quant::dequant_iq2xs(bytes, elem_count),
            KQuantScheme::IQ2S => grim_quant::dequant_iq2s(bytes, elem_count),
            KQuantScheme::IQ3XXS => grim_quant::dequant_iq3xxs(bytes, elem_count),
            KQuantScheme::IQ3S => grim_quant::dequant_iq3s(bytes, elem_count),
            KQuantScheme::IQ4NL => grim_quant::dequant_iq4nl(bytes, elem_count),
            KQuantScheme::IQ4XS => grim_quant::dequant_iq4xs(bytes, elem_count),
            _ => Err(Error::Backend(format!("Unknown iq scheme {:?}", scheme))),
        }
    }

    pub fn dequantize_iq2xxs_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, KQuantScheme::IQ2XXS)
    }
    pub fn dequantize_iq2xs_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, KQuantScheme::IQ2XS)
    }
    pub fn dequantize_iq2s_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, KQuantScheme::IQ2S)
    }
    pub fn dequantize_iq3xxs_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, KQuantScheme::IQ3XXS)
    }
    pub fn dequantize_iq3s_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, KQuantScheme::IQ3S)
    }
    pub fn dequantize_iq4nl_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, KQuantScheme::IQ4NL)
    }
    pub fn dequantize_iq4xs_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, KQuantScheme::IQ4XS)
    }

    /// Dequantize FP8 packed bytes.
    pub fn dequantize_fp8_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        let packed = CudaStorage::copy_from_host_raw_bytes(
            bytes,
            &Shape::new(vec![elem_count]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::FloatPack(FloatPackScheme::Fp8),
            },
            self.ordinal,
        )?;
        if let Ok(f32_storage) = self.dequantize_on_device(&packed) {
            return f32_storage.to_cpu_vec_f32();
        }
        grim_quant::dequant_fp8(bytes, elem_count)
    }

    /// Dequantize MXFP4 packed bytes.
    pub fn dequantize_mxfp4_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        let packed = CudaStorage::copy_from_host_raw_bytes(
            bytes,
            &Shape::new(vec![elem_count]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::FloatPack(FloatPackScheme::MxFp4),
            },
            self.ordinal,
        )?;
        if let Ok(f32_storage) = self.dequantize_on_device(&packed) {
            return f32_storage.to_cpu_vec_f32();
        }
        grim_quant::dequant_mxfp4(bytes, elem_count)
    }

    /// Dequantize MXFP8 packed bytes.
    pub fn dequantize_mxfp8_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        let packed = CudaStorage::copy_from_host_raw_bytes(
            bytes,
            &Shape::new(vec![elem_count]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::FloatPack(FloatPackScheme::MxFp8),
            },
            self.ordinal,
        )?;
        if let Ok(f32_storage) = self.dequantize_on_device(&packed) {
            return f32_storage.to_cpu_vec_f32();
        }
        grim_quant::dequant_mxfp8(bytes, elem_count)
    }
}

impl QuantOps for CudaDevice {


    fn quantized_matmul(
        &self,
        a: &dyn BackendStorage,
        b_packed: &dyn BackendStorage,
        b_scales: &[f32],
        format: grim_tensor::QuantFormat,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a_dims = a.shape().dims();
        let out_dims = out_shape.dims();
        let m = a_dims[0];
        let k = a_dims[1];
        let n = out_dims[1];

        // ── GPU fast path: fused Q8_0 kernel (K-aligned, GPU-resident only) ──
        if format == grim_tensor::QuantFormat::Q8_0 {
            let a_storage = a.as_any().downcast_ref::<CudaStorage>();
            let b_storage = b_packed.as_any().downcast_ref::<CudaStorage>();
            if let (Some(a_storage), Some(b_storage)) = (a_storage, b_storage) {
                if k >= 32 && k % 32 == 0 && b_storage.bytes() >= k * n {
                    Self::ensure_f32_input("quantized_matmul a", a_storage)?;

                    let out_storage = CudaStorage::alloc_gpu(out_shape, DType::F32, self.ordinal)?;
                    let a_ptr = Self::dev_ptr_or_err("quantized_matmul a", a_storage)?;
                    let b_ptr = Self::dev_ptr_or_err("quantized_matmul b", b_storage)?;
                    let out_ptr = out_storage.device_ptr.ok_or_else(|| {
                        Error::Backend("quantized_matmul: failed to allocate output buffer".into())
                    })? as *mut c_void;

                    // Q8_0: check if B data uses the real 34-byte packed layout
                    // (f16_scale + 32 i8 codes per block) or the simplified raw-u8 layout.
                    // For real packed data, extract per-block f16 scales from the headers.
                    // For simplified data (k*n raw u8 bytes), use the externally-provided
                    // b_scales directly.
                    let blocks_per_col = k / 32;
                    let scale_len = n * blocks_per_col;

                    // Real Q8_0 packed: n * blocks_per_col * 34 bytes
                    // Simplified: k * n bytes (used in tests/legacy path)
                    let b_bytes = b_storage.bytes();
                    let real_packed_size = n * blocks_per_col * 34;

                    let (_scales_storage, scales_device_ptr, b_data_offset) = if b_bytes
                        == real_packed_size
                    {
                        // Real packed: extract f16 scales from headers.
                        let mut host_packed = vec![0u8; b_bytes];
                        if let Some(dev_ptr) = b_storage.device_ptr {
                            unsafe {
                                let res = cudaMemcpy(
                                    host_packed.as_mut_ptr() as *mut c_void,
                                    dev_ptr as *const c_void,
                                    b_bytes,
                                    cudaMemcpyDeviceToHost,
                                );
                                if res != 0 {
                                    return Err(Error::Backend(format!(
                                        "quantized_matmul(Q8_0): cudaMemcpy scales D2H failed: {res}"
                                    )));
                                }
                            }
                        }
                        let mut scales_host = vec![1.0f32; scale_len];
                        for col in 0..n {
                            for block in 0..blocks_per_col {
                                let block_offset = (col * blocks_per_col + block) * 34;
                                let scale = half::f16::from_le_bytes([
                                    host_packed[block_offset],
                                    host_packed[block_offset + 1],
                                ])
                                .to_f32();
                                scales_host[col * blocks_per_col + block] = scale;
                            }
                        }
                        let scales_storage = CudaStorage::copy_from_host(
                            &scales_host,
                            &Shape::new(vec![scale_len]),
                            DType::F32,
                            self.ordinal,
                        )?;
                        let sptr = scales_storage.device_ptr.ok_or_else(|| {
                            Error::Backend(
                                "quantized_matmul: failed to upload scales buffer".into(),
                            )
                        })? as *const c_void;
                        (Some(scales_storage), sptr, 2usize) // kernel skips 2-byte f16 header per block
                    } else {
                        // Simplified: use externally-provided b_scales directly.
                        // The kernel still applies the B data at offset 0 (no f16 header skip).
                        let default_scales: Vec<f32> = vec![1.0f32; scale_len];
                        let scales_storage = if b_scales.is_empty() {
                            CudaStorage::copy_from_host(
                                &default_scales,
                                &Shape::new(vec![scale_len]),
                                DType::F32,
                                self.ordinal,
                            )?
                        } else if b_scales.len() == scale_len {
                            CudaStorage::copy_from_host(
                                &b_scales,
                                &Shape::new(vec![scale_len]),
                                DType::F32,
                                self.ordinal,
                            )?
                        } else {
                            return Err(Error::Shape(format!(
                                "quantized_matmul(Q8_0): b_scales length {} != expected {}",
                                b_scales.len(),
                                scale_len
                            )));
                        };
                        let sptr = scales_storage.device_ptr.ok_or_else(|| {
                            Error::Backend(
                                "quantized_matmul: failed to upload scales buffer".into(),
                            )
                        })? as *const c_void;
                        (Some(scales_storage), sptr, 0usize) // no f16 header to skip
                    };

                    let handle = self.launch_quantized_matmul_q8_0(
                        a_ptr,
                        b_ptr,
                        scales_device_ptr,
                        out_ptr,
                        m,
                        n,
                        k,
                        b_data_offset,
                    )?;
                    return Ok((Box::new(out_storage), handle));
                }
            }
        }

        // ── GPU fast path: fused K-quant and IQ-quant GEMM kernels ─────────
        let fused_kernel_name = match format {
            grim_tensor::QuantFormat::Q4K => Some("grim_fused_dequant_gemm_q4k"),
            grim_tensor::QuantFormat::Q5K => Some("grim_fused_dequant_gemm_q5k"),
            grim_tensor::QuantFormat::Q6K => Some("grim_fused_dequant_gemm_q6k"),
            grim_tensor::QuantFormat::Iq4Nl => Some("grim_fused_dequant_gemm_iq4nl"),
            grim_tensor::QuantFormat::Iq4Xs => Some("grim_fused_dequant_gemm_iq4xs"),
            grim_tensor::QuantFormat::Iq3Xxs => Some("grim_fused_dequant_gemm_iq3xxs"),
            grim_tensor::QuantFormat::Iq3S => Some("grim_fused_dequant_gemm_iq3s"),
            grim_tensor::QuantFormat::Iq2Xxs => Some("grim_fused_dequant_gemm_iq2xxs"),
            grim_tensor::QuantFormat::Iq2Xs => Some("grim_fused_dequant_gemm_iq2xs"),
            grim_tensor::QuantFormat::Iq2S => Some("grim_fused_dequant_gemm_iq2s"),
            _ => None,
        };

        if let Some(kernel_name) = fused_kernel_name {
            let a_storage = a.as_any().downcast_ref::<CudaStorage>();
            let b_storage = b_packed.as_any().downcast_ref::<CudaStorage>();
            if let (Some(a_s), Some(b_s)) = (a_storage, b_storage) {
                if k % 256 == 0 && a_s.device_ptr.is_some() && b_s.device_ptr.is_some() {
                    let out_storage = CudaStorage::alloc_gpu(out_shape, DType::F32, self.ordinal)?;
                    let a_ptr = Self::dev_ptr_or_err("quantized_matmul a", a_s)?;
                    let b_ptr = Self::dev_ptr_or_err("quantized_matmul b", b_s)?;
                    let out_ptr = Self::dev_ptr_or_err("quantized_matmul out", &out_storage)?;

                    let handle = self.launch_fused_quant_gemm(
                        kernel_name,
                        a_ptr,
                        b_ptr,
                        out_ptr,
                        m,
                        n,
                        k,
                    )?;
                    return Ok((Box::new(out_storage), handle));
                }
            }
        }

        // ── CPU fallback: format-accurate dequantization via grim_quant ──────
        // Dispatches on `format` so every supported variant uses its canonical
        // bit-unpacking algorithm. Non-supported formats return Err immediately
        // rather than producing silently wrong output.
        tracing::warn!(
            "CUDA quantized_matmul: falling back to CPU for format {format:?} \
             (m={m}, k={k}, n={n})"
        );

        let a_vec = a.to_cpu_vec_f32()?;

        // Download packed B bytes from GPU if resident.
        let b_bytes: Vec<u8> = if let Some(cs) = b_packed.as_any().downcast_ref::<CudaStorage>() {
            let mut host_bytes = vec![0u8; cs.bytes()];
            if let Some(dev_ptr) = cs.device_ptr {
                unsafe {
                    let res = cudaMemcpy(
                        host_bytes.as_mut_ptr() as *mut c_void,
                        dev_ptr as *const c_void,
                        cs.bytes(),
                        cudaMemcpyDeviceToHost,
                    );
                    if res != 0 {
                        return Err(Error::Backend(format!(
                            "quantized_matmul: cudaMemcpy(B) D2H failed: {res}"
                        )));
                    }
                }
            }
            host_bytes
        } else {
            vec![0u8; k * n]
        };

        // Dequantize B using the canonical grim_quant function for `format`.
        // CONTRACT: every arm must produce exactly `k * n` f32 values in
        // row-major order matching the B[K, N] layout expected by the GEMM below.
        let b_dequant: Vec<f32> = match format {
            grim_tensor::QuantFormat::Q8_0 => {
                // Layout detection: if b_scales is non-empty with the correct length,
                // use the simplified layout (raw u8 at 32-byte stride with external scales).
                // Otherwise, check if the byte layout matches real packed Q8_0 (34-byte blocks
                // with embedded f16 scales).
                let blocks_per_col = k / 32;
                let real_packed_size = n * blocks_per_col * 34;
                let use_simplified = !b_scales.is_empty() && b_scales.len() == n * blocks_per_col;
                let mut out = vec![0.0f32; k * n];
                if use_simplified {
                    // Simplified layout: raw u8 bytes with 32-byte block stride,
                    // scales provided externally via b_scales
                    for col in 0..n {
                        for block in 0..blocks_per_col {
                            let block_offset = (col * blocks_per_col + block) * 32;
                            let scale = b_scales
                                .get(col * blocks_per_col + block)
                                .copied()
                                .unwrap_or(1.0f32);
                            for i in 0..32 {
                                let byte_offset = block_offset + i;
                                let q_val = b_bytes
                                    .get(byte_offset)
                                    .map(|&b| (b as i8) as f32)
                                    .unwrap_or(0.0f32);
                                let r = block * 32 + i;
                                if r < k {
                                    out[r * n + col] = q_val * scale;
                                }
                            }
                        }
                    }
                } else if b_bytes.len() == real_packed_size {
                    // Real Q8_0 packed layout: extract scale from f16 header
                    for col in 0..n {
                        for block in 0..blocks_per_col {
                            let block_offset = (col * blocks_per_col + block) * 34;
                            let scale_bytes = &b_bytes[block_offset..block_offset + 2];
                            let scale =
                                half::f16::from_le_bytes([scale_bytes[0], scale_bytes[1]]).to_f32();
                            for i in 0..32 {
                                let byte_offset = block_offset + 2 + i;
                                let q_val = b_bytes
                                    .get(byte_offset)
                                    .map(|&b| (b as i8) as f32)
                                    .unwrap_or(0.0f32);
                                let r = block * 32 + i;
                                if r < k {
                                    out[r * n + col] = q_val * scale;
                                }
                            }
                        }
                    }
                } else {
                    // Unknown layout: treat as simplified with default scale 1.0
                    for col in 0..n {
                        for block in 0..blocks_per_col {
                            let block_offset = (col * blocks_per_col + block) * 32;
                            for i in 0..32 {
                                let byte_offset = block_offset + i;
                                let q_val = b_bytes
                                    .get(byte_offset)
                                    .map(|&b| (b as i8) as f32)
                                    .unwrap_or(0.0f32);
                                let r = block * 32 + i;
                                if r < k {
                                    out[r * n + col] = q_val;
                                }
                            }
                        }
                    }
                }
                out
            }
            grim_tensor::QuantFormat::Q4K => grim_quant::dequant_q4k(&b_bytes, k * n)
                .map_err(|e| Error::Backend(format!("quantized_matmul Q4K dequant: {e}")))?,
            grim_tensor::QuantFormat::Q5K => grim_quant::dequant_q5k(&b_bytes, k * n)
                .map_err(|e| Error::Backend(format!("quantized_matmul Q5K dequant: {e}")))?,
            grim_tensor::QuantFormat::Q6K => grim_quant::dequant_q6k(&b_bytes, k * n)
                .map_err(|e| Error::Backend(format!("quantized_matmul Q6K dequant: {e}")))?,
            grim_tensor::QuantFormat::Iq4Nl => grim_quant::dequant_iq4nl(&b_bytes, k * n)
                .map_err(|e| Error::Backend(format!("quantized_matmul IQ4NL dequant: {e}")))?,
            grim_tensor::QuantFormat::Iq4Xs => grim_quant::dequant_iq4xs(&b_bytes, k * n)
                .map_err(|e| Error::Backend(format!("quantized_matmul IQ4XS dequant: {e}")))?,
            grim_tensor::QuantFormat::Iq3Xxs => grim_quant::dequant_iq3xxs(&b_bytes, k * n)
                .map_err(|e| Error::Backend(format!("quantized_matmul IQ3XXS dequant: {e}")))?,
            grim_tensor::QuantFormat::Iq3S => grim_quant::dequant_iq3s(&b_bytes, k * n)
                .map_err(|e| Error::Backend(format!("quantized_matmul IQ3S dequant: {e}")))?,
            grim_tensor::QuantFormat::Iq2Xxs => grim_quant::dequant_iq2xxs(&b_bytes, k * n)
                .map_err(|e| Error::Backend(format!("quantized_matmul IQ2XXS dequant: {e}")))?,
            grim_tensor::QuantFormat::Iq2Xs => grim_quant::dequant_iq2xs(&b_bytes, k * n)
                .map_err(|e| Error::Backend(format!("quantized_matmul IQ2XS dequant: {e}")))?,
            grim_tensor::QuantFormat::Iq2S => grim_quant::dequant_iq2s(&b_bytes, k * n)
                .map_err(|e| Error::Backend(format!("quantized_matmul IQ2S dequant: {e}")))?,
            grim_tensor::QuantFormat::Fp4 => grim_quant::dequant_fp4(&b_bytes, k * n)
                .map_err(|e| Error::Backend(format!("quantized_matmul FP4 dequant: {e}")))?,
            grim_tensor::QuantFormat::Fp4Block16 => {
                grim_quant::dequant_fp4_block16(&b_bytes, k * n).map_err(|e| {
                    Error::Backend(format!("quantized_matmul FP4Block16 dequant: {e}"))
                })?
            }
            unsupported => {
                return Err(Error::Backend(format!(
                    "CUDA quantized_matmul: no GPU kernel or CPU dequant path \
                     for format {unsupported:?}"
                )));
            }
        };

        // GEMM: C[M, N] = A[M, K] · B_dequant[K, N]
        let mut c_vec = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a_vec[row * k + p] * b_dequant[p * n + col];
                }
                c_vec[row * n + col] = sum;
            }
        }

        let out_storage = self.from_cpu(&c_vec, out_shape, a.dtype())?;
        Ok((
            out_storage,
            Box::new(CudaHandle {
                completed: Arc::new(Mutex::new(true)),
            }),
        ))
    }


    fn quantized_matmul_backward_dx(
        &self,
        dy: &dyn BackendStorage,
        b_packed: &dyn BackendStorage,
        _b_scales: &[f32],
        _default_bpw: u8,
        m: usize,
        n: usize,
        k: usize,
        out_shape: &Shape,
        _residuals: Option<&grim_tensor::QuantizedMatmulBackwardResiduals>,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let dy_storage = dy.as_any().downcast_ref::<CudaStorage>().ok_or_else(|| {
            Error::Backend("quantized_matmul_backward_dx: dy not CudaStorage".into())
        })?;

        let b_storage = b_packed
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| {
                Error::Backend("quantized_matmul_backward_dx: b_packed not CudaStorage".into())
            })?;

        // Validate the output shape contract (dX is [M, K]).
        if out_shape.dims() != [m, k] {
            return Err(Error::Shape(format!(
                "quantized_matmul_backward_dx: out_shape must be [{m},{k}], got {:?}",
                out_shape.dims()
            )));
        }

        // dy must be F32 to feed cuBLAS.
        if dy_storage.dtype.arith != ArithType::F32 {
            return Err(Error::DTypeMismatch(format!(
                "quantized_matmul_backward_dx: dy must be F32, got {:?}",
                dy_storage.dtype
            )));
        }

        // B is stored as [K, N] row-major in the packed payload.
        let b_elem_count = k * n;
        let b_bytes = b_storage.copy_to_host_raw_bytes()?;
        let b_scales = b_storage.quant_scales();

        // Host dequant: packed bytes -> F32 [K, N] row-major, via grim-quant.
        let b_dequant =
            cuda_dequant_quantized_storage(&b_bytes, b_scales, b_elem_count, &b_storage.dtype)?;

        // Re-upload B as F32, then compute dX = dY @ B^T directly via cuBLAS.
        //
        // The generic `matmul` cannot be reused here: its column-major
        // transpose trick assumes the forward orientation (inner dim K is the
        // leading dimension of B), which only holds when m == n == k and breaks
        // for the backward shape dY[M,N] @ B^T[N,K] -> dX[M,K].
        //
        // Correct row-major GEMM for dX[M,K] = dY[M,N] @ B^T[N,K]:
        //   C_col(K,M) = B_col(K,N) * A_col(N,M)
        // where B^T row-major [N,K] read column-major is B [K,N] (lda = K), and
        // dY row-major [M,N] read column-major is dY^T [N,M] (ldb = N).
        let b_shape = b_storage.shape().clone();
        let (b_rows, b_cols) = (b_shape.dims()[0], b_shape.dims()[1]);
        let mut b_t = vec![0.0f32; b_elem_count];
        for r in 0..b_rows {
            for c in 0..b_cols {
                b_t[c * b_rows + r] = b_dequant[r * b_cols + c];
            }
        }
        let b_t_shape = Shape::new(vec![b_cols, b_rows]);
        let b_t_storage = CoreTensorOps::from_cpu(
            self,
            &b_t,
            &b_t_shape,
            DType {
                arith: ArithType::F32,
                storage: DTypeStorage::Native,
            },
        )?;

        let dx_storage = CudaStorage::alloc_gpu(out_shape, DType::F32, self.ordinal)?;

        let handle = self.get_cublas_handle()?;
        let alpha = 1.0f32;
        let beta = 0.0f32;

        let b_t_ptr = b_t_storage
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| {
                Error::Backend("quantized_matmul_backward_dx: b_t not CudaStorage".into())
            })?
            .device_ptr
            .ok_or_else(|| {
                Error::Backend("quantized_matmul_backward_dx: b_t has no device pointer".into())
            })? as *const c_void;
        let dy_ptr = dy_storage.device_ptr.ok_or_else(|| {
            Error::Backend("quantized_matmul_backward_dx: dY has no device pointer".into())
        })? as *const c_void;
        let dx_ptr = dx_storage.device_ptr.ok_or_else(|| {
            Error::Backend("quantized_matmul_backward_dx: dX has no device pointer".into())
        })? as *mut c_void;

        // SAFETY: all pointers are freshly allocated/uploaded on this device;
        // leading dims are the row counts of the column-major views, all >= 1.
        unsafe {
            let status = cublasSgemm_v2(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                k as i32, // m_cublas = rows of B_col = K
                m as i32, // n_cublas = cols of A_col = M
                n as i32, // k_cublas = inner N
                &alpha,
                b_t_ptr as *const f32, // B^T [N,K] row-major, read col-major as B [K,N]
                k as i32,              // lda = K
                dy_ptr as *const f32,  // dY [M,N] row-major, read col-major as dY^T [N,M]
                n as i32,              // ldb = N
                &beta,
                dx_ptr as *mut f32, // dX [M,K] row-major, read col-major as dX^T [K,M]
                k as i32,           // ldc = K
            );
            if status != CUBLAS_STATUS_SUCCESS {
                return Err(Error::Backend(format!(
                    "cublasSgemm_v2 (backward dx) failed with status {}",
                    status
                )));
            }
        }

        let compute_handle = Box::new(CudaHandle {
            completed: Arc::new(Mutex::new(true)),
        });

        Ok((Box::new(dx_storage), compute_handle))
    }


    fn quantize(
        &self,
        x: &dyn BackendStorage,
        format: grim_tensor::QuantFormat,
    ) -> Result<Box<dyn BackendStorage>> {
        let x_storage = x
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("quantize: x is not CudaStorage".into()))?;
        let out = self.quantize_on_device(x_storage, format)?;
        Ok(Box::new(out))
    }


    fn fused_quant_gemm(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        format: grim_tensor::QuantFormat,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let kernel_name = match format {
            grim_tensor::QuantFormat::Q8_0 => "grim_fused_quant_gemm_q8_0_packed",
            grim_tensor::QuantFormat::Q4K => "grim_fused_quant_gemm_q4_k",
            grim_tensor::QuantFormat::Q5K => "grim_fused_quant_gemm_q5_k",
            grim_tensor::QuantFormat::Q6K => "grim_fused_quant_gemm_q6_k",
            grim_tensor::QuantFormat::Iq4Nl => "grim_fused_quant_gemm_iq4nl",
            grim_tensor::QuantFormat::Iq4Xs => "grim_fused_quant_gemm_iq4xs",
            grim_tensor::QuantFormat::Fp4 => "grim_fused_quant_gemm_mxfp4",
            grim_tensor::QuantFormat::Fp4Block16 => "grim_fused_quant_gemm_nvfp4",
            grim_tensor::QuantFormat::Fp8 => {
                // T1 caps gate: without native FP8 (compute < 8.9), don't select the fp8 shader.
                if !self
                    .caps
                    .supports_quant_format(grim_tensor::QuantFormat::Fp8)
                {
                    return Err(Error::Backend(
                        "fused_quant_gemm: FP8 not supported on this device".into(),
                    ));
                }
                "grim_fused_quant_gemm_fp8"
            }
            other => {
                return Err(Error::Unimplemented(format!(
                    "fused_quant_gemm: no GPU kernel for format {other:?}"
                )));
            }
        };

        let a_storage = a
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("fused_quant_gemm: a is not CudaStorage".into()))?;
        let b_storage = b
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("fused_quant_gemm: b is not CudaStorage".into()))?;

        Self::ensure_f32_input("fused_quant_gemm a", a_storage)?;

        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();
        let out_dims = out_shape.dims();
        if a_dims.len() != 2 || b_dims.len() != 2 || out_dims.len() != 2 {
            return Err(Error::Shape(
                "fused_quant_gemm expects 2-D a, b, out".into(),
            ));
        }
        let (m, k) = (a_dims[0], a_dims[1]);
        // GGUF packed weight b is [out_dim, in_dim] = [N, K].
        // In fused GEMM (A @ B^T), K matches b_dims[1] if b is untransposed [N, K],
        // or b_dims[0] if b_dims was transposed to [K, N].
        let (n, k2) = if b_dims[0] == k {
            (b_dims[1], b_dims[0])
        } else {
            (b_dims[0], b_dims[1])
        };
        if k != k2 {
            return Err(Error::Shape(format!(
                "fused_quant_gemm: a is ({m},{k}) but b is ({n},{k2})"
            )));
        }
        if format == grim_tensor::QuantFormat::Q8_0 && k % 32 != 0 {
            return Err(Error::Shape(format!(
                "fused_quant_gemm(Q8_0): K ({k}) must be a multiple of 32"
            )));
        }

        let a_ptr = Self::dev_ptr_or_err("fused_quant_gemm a", a_storage)?;
        let out = CudaStorage::alloc_gpu(out_shape, DType::F32, self.ordinal)?;
        let out_ptr = Self::dev_ptr_or_err("fused_quant_gemm out", &out)?;

        // For Q8_0, pass the packed unsigned char* directly — the new kernel
        // (grim_fused_quant_gemm_q8_0_packed) reads f16 scales and i8 codes
        // from the 34-byte blocks on-device.
        let b_ptr = if format == grim_tensor::QuantFormat::Q8_0 {
            let b_storage = b
                .as_any()
                .downcast_ref::<CudaStorage>()
                .ok_or_else(|| Error::Backend("fused_quant_gemm: b is not CudaStorage".into()))?;
            let b_bytes = b_storage.bytes();
            let blocks_per_col = k / 32;
            let expected_packed = n * blocks_per_col * 34;
            if b_bytes != expected_packed {
                return Err(Error::Shape(format!(
                    "fused_quant_gemm(Q8_0): B packed size {} != expected {}",
                    b_bytes, expected_packed
                )));
            }
            Self::dev_ptr_or_err("fused_quant_gemm b (packed)", b_storage)?
        } else {
            Self::dev_ptr_or_err("fused_quant_gemm b", b_storage)?
        };

        let handle = self.launch_fused_quant_gemm(kernel_name, a_ptr, b_ptr, out_ptr, m, n, k)?;
        Ok((Box::new(out), handle))
    }
}


impl grim_format::convert::GpuDequant for CudaDevice {
    fn dequantize(
        &self,
        storage: &grim_tensor::dtype::Storage,
        bytes: &[u8],
        elem_count: usize,
    ) -> grim_tensor::error::Result<Option<Vec<f32>>> {
        match storage {
            grim_tensor::dtype::Storage::KQuant(grim_tensor::dtype::KQuantScheme::Q80) => {
                Ok(Some(self.dequantize_q8_0_host(bytes, elem_count)?))
            }
            grim_tensor::dtype::Storage::KQuant(grim_tensor::dtype::KQuantScheme::Q4K) => {
                Ok(Some(self.dequantize_q4k_host(bytes, elem_count)?))
            }
            _ => Ok(None),
        }
    }
}

