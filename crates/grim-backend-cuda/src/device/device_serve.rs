//! Serving, execution routing, memory, and graph capture operations for `CudaDevice`.

use std::ffi::c_void;

use grim_tensor::dtype::{ArithType, DType};
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendStorage, CollectiveOps, GraphCaptureOps, MemoryOps, Shape};

use crate::device::cuda_device::CudaDevice;
use crate::device::handles::{
    cuLaunchKernel, cuModuleGetFunction, cudaMemcpy, cudaMemcpyDeviceToDevice,
    cudaMemcpyDeviceToHost, cudaMemcpyHostToDevice, cudaMemcpyPeer, cudaSetDevice, cudaSuccess,
};
use crate::device::jit_cache::compile_and_load_kernel;
use crate::memory::storage::CudaStorage;

impl CudaDevice {
    pub fn launch_speculative_rejection_sample(
        &self,
        target_probs_storage: &CudaStorage,
        draft_probs_storage: &CudaStorage,
        draft_tokens_storage: &CudaStorage,
        uniform_rands_storage: &CudaStorage,
        accepted_tokens_storage: &CudaStorage,
        accepted_lens_storage: &CudaStorage,
        batch_size: usize,
        num_draft_tokens: usize,
        vocab_size: usize,
    ) -> Result<()> {
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
        let at_ptr = accepted_tokens_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("spec_sample: accepted_tokens has no device ptr".into()))?;
        let al_ptr = accepted_lens_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("spec_sample: accepted_lens has no device ptr".into()))?;

        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut kernel: *mut c_void = std::ptr::null_mut();
        let name = std::ffi::CString::new("grim_speculative_rejection_sample").unwrap();
        unsafe {
            let res = cuModuleGetFunction(&mut kernel, module, name.as_ptr());
            if res != 0 {
                return Err(Error::Backend(format!(
                    "cuModuleGetFunction failed for grim_speculative_rejection_sample: {res}"
                )));
            }

            let mut tp = tp_ptr as *mut c_void;
            let mut dp = dp_ptr as *mut c_void;
            let mut dt = dt_ptr as *mut c_void;
            let mut ur = ur_ptr as *mut c_void;
            let mut at = at_ptr as *mut c_void;
            let mut al = al_ptr as *mut c_void;
            let mut bs = batch_size as i32;
            let mut ndt = num_draft_tokens as i32;
            let mut vs = vocab_size as i32;

            let mut args: [*mut c_void; 9] = [
                &mut tp as *mut _ as *mut c_void,
                &mut dp as *mut _ as *mut c_void,
                &mut dt as *mut _ as *mut c_void,
                &mut ur as *mut _ as *mut c_void,
                &mut at as *mut _ as *mut c_void,
                &mut al as *mut _ as *mut c_void,
                &mut bs as *mut _ as *mut c_void,
                &mut ndt as *mut _ as *mut c_void,
                &mut vs as *mut _ as *mut c_void,
            ];

            let res = cuLaunchKernel(
                kernel,
                batch_size as u32, 1, 1,
                256, 1, 1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            if res != 0 {
                return Err(Error::Backend(format!(
                    "cuLaunchKernel failed for grim_speculative_rejection_sample: {res}"
                )));
            }
        }
        Ok(())
    }

    /// Cross-device direct copy between CUDA devices using cudaMemcpyPeer or staging fallback.
    pub fn copy_via_route(
        &self,
        src_ordinal: i32,
        dst_ordinal: i32,
        src_ptr: *const c_void,
        dst_ptr: *mut c_void,
        bytes: usize,
    ) -> Result<()> {
        if src_ordinal == dst_ordinal {
            unsafe {
                let res = cudaMemcpy(dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToDevice);
                if res != cudaSuccess {
                    return Err(Error::Backend(format!(
                        "copy_via_route D2D failed on device {dst_ordinal}: {res}"
                    )));
                }
            }
            return Ok(());
        }

        // Try direct peer memcpy
        unsafe {
            let res = cudaMemcpyPeer(
                dst_ptr,
                dst_ordinal,
                src_ptr,
                src_ordinal,
                bytes,
            );
            if res == cudaSuccess {
                return Ok(());
            }

            // Fallback via host staging buffer
            let mut staging = vec![0u8; bytes];
            let res_d2h = cudaMemcpy(
                staging.as_mut_ptr() as *mut c_void,
                src_ptr,
                bytes,
                cudaMemcpyDeviceToHost,
            );
            if res_d2h != cudaSuccess {
                return Err(Error::Backend(format!(
                    "copy_via_route D2H fallback failed: {res_d2h}"
                )));
            }
            let _ = cudaSetDevice(dst_ordinal);
            let res_h2d = cudaMemcpy(
                dst_ptr,
                staging.as_ptr() as *const c_void,
                bytes,
                cudaMemcpyHostToDevice,
            );
            if res_h2d != cudaSuccess {
                return Err(Error::Backend(format!(
                    "copy_via_route H2D fallback failed: {res_h2d}"
                )));
            }
        }
        Ok(())
    }


}

impl CollectiveOps for CudaDevice {


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
            ArithType::F16 | ArithType::BF16 => 150.0,
            ArithType::F32 => 75.0,
            _ => 40.0,
        };
        (flops / (tflops * 1e12) * 1000.0).max(0.01)
    }
}



impl MemoryOps for CudaDevice {


    fn from_cpu_bytes(
        &self,
        data: &[u8],
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn BackendStorage>> {
        // Packed quantized storage (KQuant, FloatPack, Block, GroupInt) is
        // smaller than `elem_count * arith.byte_size`; allocate the exact byte
        // length so `CudaStorage::bytes()` reflects the real packed payload.
        // For Native storage `data.len()` already equals `elem_count * byte_size`,
        // so this remains correct for both cases.
        let storage = CudaStorage::copy_from_host_raw_bytes(data, shape, dtype, self.ordinal)?;
        let dev_ptr = storage.device_ptr.ok_or_else(|| {
            Error::Backend("from_cpu_bytes: device_ptr is null after raw byte alloc".into())
        })? as *mut c_void;

        let res = unsafe {
            cudaMemcpy(
                dev_ptr,
                data.as_ptr() as *const c_void,
                data.len(),
                cudaMemcpyHostToDevice,
            )
        };
        if res != cudaSuccess {
            return Err(Error::Backend(format!(
                "cudaMemcpy from_cpu_bytes failed: {}",
                res
            )));
        }

        Ok(Box::new(storage))
    }
}



impl GraphCaptureOps for CudaDevice {
}


