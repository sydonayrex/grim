//! MoE routing operations for `CudaDevice`.

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use grim_tensor::backend::ComputeHandle;
use grim_tensor::dtype::DType;
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendStorage, Shape};

use crate::device::cuda_device::CudaDevice;
use crate::device::handles::{CUfunction, CudaHandle, cuLaunchKernel, cuModuleGetFunction};
use crate::device::jit_cache::compile_and_load_kernel;
use crate::memory::storage::CudaStorage;

impl CudaDevice {
    /// Fused grouped MoE dispatch (WI-M5). Mirrors `grim_moe_fused_dispatch` on ROCm and `moe_fused_dispatch` on Vulkan: one CUDA thread block per routed (token,
    /// expert) pair, computing the full SwiGLU expert contribution and atomicAdding the `routed_scaling_factor * weight`-scaled result into the shared token output.
    pub fn moe_fused_dispatch(
        &self,
        x: &dyn BackendStorage,
        gate_w: &dyn BackendStorage,
        up_w: &dyn BackendStorage,
        down_w: &dyn BackendStorage,
        router_tokens: &dyn BackendStorage,
        router_experts: &dyn BackendStorage,
        router_weights: &dyn BackendStorage,
        out_shape: &Shape,
        hidden: u32,
        inter: u32,
        num_experts: u32,
        batch: u32,
        routed_scaling_factor: f32,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = x
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("moe_fused_dispatch: x is not CudaStorage".into()))?;
        let gw_s = gate_w
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| {
                Error::Backend("moe_fused_dispatch: gate_w is not CudaStorage".into())
            })?;
        let uw_s = up_w
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| Error::Backend("moe_fused_dispatch: up_w is not CudaStorage".into()))?;
        let dw_s = down_w
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| {
                Error::Backend("moe_fused_dispatch: down_w is not CudaStorage".into())
            })?;
        let tok_s = router_tokens
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| {
                Error::Backend("moe_fused_dispatch: router_tokens is not CudaStorage".into())
            })?;
        let exp_s = router_experts
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| {
                Error::Backend("moe_fused_dispatch: router_experts is not CudaStorage".into())
            })?;
        let wt_s = router_weights
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| {
                Error::Backend("moe_fused_dispatch: router_weights is not CudaStorage".into())
            })?;
        Self::ensure_f32_input("moe_fused_dispatch x", x_s)?;
        Self::ensure_f32_input("moe_fused_dispatch gate_w", gw_s)?;
        Self::ensure_f32_input("moe_fused_dispatch up_w", uw_s)?;
        Self::ensure_f32_input("moe_fused_dispatch down_w", dw_s)?;

        // Output is zero-initialized; the kernel atomicAdds contributions.
        // cudaMalloc does not zero memory, so clear it explicitly before launch.
        let out_storage = CudaStorage::alloc_gpu(out_shape, DType::F32, self.ordinal)?;
        out_storage.fill_zeroes()?;
        let mut out_ptr = out_storage.device_ptr.unwrap();

        let mut x_ptr = x_s.device_ptr.unwrap();
        let mut gw_ptr = gw_s.device_ptr.unwrap();
        let mut uw_ptr = uw_s.device_ptr.unwrap();
        let mut dw_ptr = dw_s.device_ptr.unwrap();
        let mut tok_ptr = tok_s.device_ptr.unwrap();
        let mut exp_ptr = exp_s.device_ptr.unwrap();
        let mut wt_ptr = wt_s.device_ptr.unwrap();
        let mut hidden_i = hidden as i32;
        let mut inter_i = inter as i32;
        let mut num_experts_i = num_experts as i32;
        let mut batch_i = batch as i32;
        let mut rsf = routed_scaling_factor;

        let mut args: [*mut c_void; 13] = [
            &mut x_ptr as *mut u64 as *mut c_void,
            &mut gw_ptr as *mut u64 as *mut c_void,
            &mut uw_ptr as *mut u64 as *mut c_void,
            &mut dw_ptr as *mut u64 as *mut c_void,
            &mut tok_ptr as *mut u64 as *mut c_void,
            &mut exp_ptr as *mut u64 as *mut c_void,
            &mut wt_ptr as *mut u64 as *mut c_void,
            &mut out_ptr as *mut u64 as *mut c_void,
            &mut hidden_i as *mut i32 as *mut c_void,
            &mut inter_i as *mut i32 as *mut c_void,
            &mut num_experts_i as *mut i32 as *mut c_void,
            &mut batch_i as *mut i32 as *mut c_void,
            &mut rsf as *mut f32 as *mut c_void,
        ];

        let module = compile_and_load_kernel(crate::kernels::KERNELS_SOURCE, self.ordinal)?;
        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            let func_name = std::ffi::CString::new("grim_moe_fused_dispatch")
                .map_err(|e| Error::Backend(format!("invalid kernel name: {e}")))?;
            let res = cuModuleGetFunction(&mut func, module, func_name.as_ptr());
            if res != 0 {
                return Err(Error::Backend(format!(
                    "cuModuleGetFunction(grim_moe_fused_dispatch) failed: {res}"
                )));
            }
            // grid_x = number of routed pairs (one block per pair).
            let num_pairs = tok_s.shape.elem_count() as u32;
            let launch_res = cuLaunchKernel(
                func,
                num_pairs,
                1,
                1,
                1,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr() as *mut *mut c_void,
                std::ptr::null_mut(),
            );
            if launch_res != 0 {
                return Err(Error::Backend(format!(
                    "cuLaunchKernel(grim_moe_fused_dispatch) failed: {launch_res}"
                )));
            }
        }
        let compute_handle = Box::new(CudaHandle {
            completed: Arc::new(Mutex::new(false)),
        });
        Ok((Box::new(out_storage), compute_handle))
    }

    /// Fused MoE dispatch against resident weights.
    /// Provides parity with resident MoE dispatch entry points on other backend devices.
    pub fn moe_fused_dispatch_resident(
        &self,
        x: &dyn BackendStorage,
        gate_w: &dyn BackendStorage,
        up_w: &dyn BackendStorage,
        down_w: &dyn BackendStorage,
        router_tokens: &dyn BackendStorage,
        router_experts: &dyn BackendStorage,
        router_weights: &dyn BackendStorage,
        out_shape: &Shape,
        hidden: u32,
        inter: u32,
        num_experts: u32,
        batch: u32,
        routed_scaling_factor: f32,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        self.moe_fused_dispatch(
            x,
            gate_w,
            up_w,
            down_w,
            router_tokens,
            router_experts,
            router_weights,
            out_shape,
            hidden,
            inter,
            num_experts,
            batch,
            routed_scaling_factor,
        )
    }
}
