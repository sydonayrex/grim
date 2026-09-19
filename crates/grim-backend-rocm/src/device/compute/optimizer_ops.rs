//! Core tensor computation, GEMM, elementwise, autograd, and optimizer operations for `RocmDevice`.
//! The trait-required `impl OptimizerOps for RocmDevice` block, kept whole.



use grim_tensor::backend::{ ComputeHandle };
use grim_tensor::error::{Error, Result};
use grim_tensor::{ BackendStorage, OptimizerOps };

use crate::device::roc_device::RocmDevice;
use crate::{ RocmHandle, arg, as_rocm, dev_ptr, linear_launch };

impl OptimizerOps for RocmDevice {
    fn fused_adamw_step(
        &self,
        p: &dyn BackendStorage,
        g: &dyn BackendStorage,
        m: &dyn BackendStorage,
        v: &dyn BackendStorage,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        bc1: f32,
        bc2: f32,
        total: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let p_s = as_rocm(p)?;
        let g_s = as_rocm(g)?;
        let m_s = as_rocm(m)?;
        let v_s = as_rocm(v)?;
        if !p_s.device_ptr_is_valid()
            || !g_s.device_ptr_is_valid()
            || !m_s.device_ptr_is_valid()
            || !v_s.device_ptr_is_valid()
        {
            return Err(Error::Backend(
                "fused_adamw_step: inputs lack a valid device pointer".into(),
            ));
        }
        let mut p_ptr = dev_ptr(p_s)?;
        let mut g_ptr = dev_ptr(g_s)?;
        let mut m_ptr = dev_ptr(m_s)?;
        let mut v_ptr = dev_ptr(v_s)?;
        let mut lr = lr;
        let mut beta1 = beta1;
        let mut beta2 = beta2;
        let mut eps = eps;
        let mut weight_decay = weight_decay;
        let mut bc1 = bc1;
        let mut bc2 = bc2;
        let mut n = total as i32;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_fused_adamw_step",
            grid,
            block,
            &mut [
                arg(&mut p_ptr),
                arg(&mut g_ptr),
                arg(&mut m_ptr),
                arg(&mut v_ptr),
                arg(&mut lr),
                arg(&mut beta1),
                arg(&mut beta2),
                arg(&mut eps),
                arg(&mut weight_decay),
                arg(&mut bc1),
                arg(&mut bc2),
                arg(&mut n),
            ],
        )?;
        Ok(Box::new(RocmHandle::new(Some(self.active_stream()))))
    }

    fn fused_lion_step(
        &self,
        p: &dyn BackendStorage,
        g: &dyn BackendStorage,
        exp_avg: &dyn BackendStorage,
        lr: f32,
        beta1: f32,
        beta2: f32,
        weight_decay: f32,
        total: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let p_s = as_rocm(p)?;
        let g_s = as_rocm(g)?;
        let exp_s = as_rocm(exp_avg)?;
        if !p_s.device_ptr_is_valid() || !g_s.device_ptr_is_valid() || !exp_s.device_ptr_is_valid()
        {
            return Err(Error::Backend(
                "fused_lion_step: inputs lack a valid device pointer".into(),
            ));
        }
        let mut p_ptr = dev_ptr(p_s)?;
        let mut g_ptr = dev_ptr(g_s)?;
        let mut exp_ptr = dev_ptr(exp_s)?;
        let mut lr = lr;
        let mut beta1 = beta1;
        let mut beta2 = beta2;
        let mut weight_decay = weight_decay;
        let mut n = total as i32;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_fused_lion_step",
            grid,
            block,
            &mut [
                arg(&mut p_ptr),
                arg(&mut g_ptr),
                arg(&mut exp_ptr),
                arg(&mut lr),
                arg(&mut beta1),
                arg(&mut beta2),
                arg(&mut weight_decay),
                arg(&mut n),
            ],
        )?;
        Ok(Box::new(RocmHandle::new(Some(self.active_stream()))))
    }

    fn fused_madam_step(
        &self,
        p: &dyn BackendStorage,
        g: &dyn BackendStorage,
        m: &dyn BackendStorage,
        v: &dyn BackendStorage,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        gamma: f32,
        weight_decay: f32,
        bc1: f32,
        bc2: f32,
        total: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let p_s = as_rocm(p)?;
        let g_s = as_rocm(g)?;
        let m_s = as_rocm(m)?;
        let v_s = as_rocm(v)?;
        if !p_s.device_ptr_is_valid()
            || !g_s.device_ptr_is_valid()
            || !m_s.device_ptr_is_valid()
            || !v_s.device_ptr_is_valid()
        {
            return Err(Error::Backend(
                "fused_madam_step: inputs lack a valid device pointer".into(),
            ));
        }
        let mut p_ptr = dev_ptr(p_s)?;
        let mut g_ptr = dev_ptr(g_s)?;
        let mut m_ptr = dev_ptr(m_s)?;
        let mut v_ptr = dev_ptr(v_s)?;
        let mut lr = lr;
        let mut beta1 = beta1;
        let mut beta2 = beta2;
        let mut eps = eps;
        let mut gamma = gamma;
        let mut weight_decay = weight_decay;
        let mut bc1 = bc1;
        let mut bc2 = bc2;
        let mut n = total as i32;
        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_fused_madam_step",
            grid,
            block,
            &mut [
                arg(&mut p_ptr),
                arg(&mut g_ptr),
                arg(&mut m_ptr),
                arg(&mut v_ptr),
                arg(&mut lr),
                arg(&mut beta1),
                arg(&mut beta2),
                arg(&mut eps),
                arg(&mut gamma),
                arg(&mut weight_decay),
                arg(&mut bc1),
                arg(&mut bc2),
                arg(&mut n),
            ],
        )?;
        Ok(Box::new(RocmHandle::new(Some(self.active_stream()))))
    }
}

impl RocmDevice {
    /// SPEED-ROC-10: multi-tensor (foreach) fused AdamW — one launch for the
    /// whole parameter list, amortizing per-tensor kernel-launch overhead
    /// (dominant when there are many small tensors, e.g. LoRA or shard splits).
    /// All tensors must be F32 device storages; lengths come from each storage.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_adamw_step_foreach(
        &self,
        params: &[&dyn BackendStorage],
        grads: &[&dyn BackendStorage],
        moments_m: &[&dyn BackendStorage],
        moments_v: &[&dyn BackendStorage],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        bc1: f32,
        bc2: f32,
    ) -> Result<Box<dyn ComputeHandle>> {
        if params.len() != grads.len()
            || params.len() != moments_m.len()
            || params.len() != moments_v.len()
        {
            return Err(Error::Backend(
                "fused_adamw_step_foreach: p/g/m/v list length mismatch".into(),
            ));
        }
        if params.is_empty() {
            return Ok(Box::new(RocmHandle::new(Some(self.active_stream()))));
        }

        // Flatten to raw device pointers + element-count prefix sums.
        let mut p_ptrs = Vec::with_capacity(params.len());
        let mut g_ptrs = Vec::with_capacity(grads.len());
        let mut m_ptrs = Vec::with_capacity(moments_m.len());
        let mut v_ptrs = Vec::with_capacity(moments_v.len());
        let mut offsets = Vec::with_capacity(params.len() + 1);
        offsets.push(0i32);
        let mut total: i64 = 0;
        for (i, p) in params.iter().enumerate() {
            let p_s = as_rocm(*p)?;
            let g_s = as_rocm(grads[i])?;
            let m_s = as_rocm(moments_m[i])?;
            let v_s = as_rocm(moments_v[i])?;
            if !p_s.device_ptr_is_valid()
                || !g_s.device_ptr_is_valid()
                || !m_s.device_ptr_is_valid()
                || !v_s.device_ptr_is_valid()
            {
                return Err(Error::Backend(
                    "fused_adamw_step_foreach: tensor {i} lacks a valid device pointer".into(),
                ));
            }
            p_ptrs.push(dev_ptr(p_s)? as *mut f32);
            g_ptrs.push(dev_ptr(g_s)? as *const f32);
            m_ptrs.push(dev_ptr(m_s)? as *mut f32);
            v_ptrs.push(dev_ptr(v_s)? as *mut f32);
            total += p_s.shape.elem_count() as i64;
            offsets.push(total as i32);
        }
        if total > i32::MAX as i64 {
            return Err(Error::Backend(
                "fused_adamw_step_foreach: concatenated parameter count overflows i32".into(),
            ));
        }

        // Upload the pointer/index arrays; freed stream-ordered after launch.
        let p_dev = crate::upload_device_buffer(self.ordinal, &p_ptrs)?;
        let g_dev = crate::upload_device_buffer(self.ordinal, &g_ptrs)?;
        let m_dev = crate::upload_device_buffer(self.ordinal, &m_ptrs)?;
        let v_dev = crate::upload_device_buffer(self.ordinal, &v_ptrs)?;
        let off_dev = crate::upload_device_buffer(self.ordinal, &offsets)?;

        let mut p_list = p_dev;
        let mut g_list = g_dev;
        let mut m_list = m_dev;
        let mut v_list = v_dev;
        let mut offs = off_dev;
        let mut n_tensors = params.len() as i32;
        let mut total_i = total;
        let mut lr = lr;
        let mut beta1 = beta1;
        let mut beta2 = beta2;
        let mut eps = eps;
        let mut weight_decay = weight_decay;
        let mut bc1 = bc1;
        let mut bc2 = bc2;

        // The kernel grid-strides over the concatenated buffer, so a mild
        // oversubscription suffices regardless of the parameter count.
        let (grid, block_dim) = linear_launch(total as usize / 4 + 1);

        let stream = self.launch_compute_kernel(
            "grim_fused_adamw_step_foreach",
            grid,
            block_dim,
            &mut [
                arg(&mut p_list),
                arg(&mut g_list),
                arg(&mut m_list),
                arg(&mut v_list),
                arg(&mut offs),
                arg(&mut n_tensors),
                arg(&mut total_i),
                arg(&mut lr),
                arg(&mut beta1),
                arg(&mut beta2),
                arg(&mut eps),
                arg(&mut weight_decay),
                arg(&mut bc1),
                arg(&mut bc2),
            ],
        )?;

        // Stream-ordered free of the transient index/pointer arrays.
        if self.active_capture_stream().is_none() {
            unsafe {
                let free_stream = self.active_stream();
                let _ = crate::hipFreeAsync(p_list, free_stream);
                let _ = crate::hipFreeAsync(g_list, free_stream);
                let _ = crate::hipFreeAsync(m_list, free_stream);
                let _ = crate::hipFreeAsync(v_list, free_stream);
                let _ = crate::hipFreeAsync(offs, free_stream);
            }
        }
        Ok(Box::new(RocmHandle::new(Some(stream))))
    }
}
