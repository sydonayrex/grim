//! optimizer_ops ops for MetalDevice — moved verbatim from lib.rs.

use grim_tensor::backend::{ ComputeHandle };
#[allow(unused_imports)]
use grim_tensor::dtype::{
    DType, FloatPackScheme, KQuantScheme, QuantFormat, QuantProvenance, Storage as DTypeStorage,
};
use grim_tensor::error::{ Result };
use grim_tensor::{ BackendStorage, CoreTensorOps, OptimizerOps };


#[cfg(target_vendor = "apple")]
use objc2::rc::Retained;
#[cfg(target_vendor = "apple")]
use objc2::runtime::ProtocolObject;
#[cfg(target_vendor = "apple")]
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLComputePipelineState, MTLDevice, MTLSize,
};

use crate::*;

impl OptimizerOps for MetalDevice {
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
        let mut p_vec = p.to_cpu_vec_f32()?;
        let g_vec = g.to_cpu_vec_f32()?;
        let mut m_vec = m.to_cpu_vec_f32()?;
        let mut v_vec = v.to_cpu_vec_f32()?;

        for i in 0..total.min(p_vec.len()) {
            let grad = g_vec[i];
            let mut param = p_vec[i];
            if weight_decay != 0.0 {
                param -= lr * weight_decay * param;
            }
            let m_val = beta1 * m_vec[i] + (1.0 - beta1) * grad;
            let v_val = beta2 * v_vec[i] + (1.0 - beta2) * grad * grad;
            m_vec[i] = m_val;
            v_vec[i] = v_val;

            let m_hat = m_val / bc1.max(1e-7);
            let v_hat = v_val / bc2.max(1e-7);
            param -= lr * m_hat / (v_hat.sqrt() + eps);
            p_vec[i] = param;
        }

        let _ = self.from_cpu(&p_vec, p.shape(), p.dtype())?;
        Ok(Box::new(grim_tensor::backend::ReadyHandle))
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
        let mut p_vec = p.to_cpu_vec_f32()?;
        let g_vec = g.to_cpu_vec_f32()?;
        let mut m_vec = exp_avg.to_cpu_vec_f32()?;

        for i in 0..total.min(p_vec.len()) {
            let grad = g_vec[i];
            let mut param = p_vec[i];
            if weight_decay != 0.0 {
                param -= lr * weight_decay * param;
            }
            let c = beta1 * m_vec[i] + (1.0 - beta1) * grad;
            let update = if c > 0.0 {
                1.0
            } else if c < 0.0 {
                -1.0
            } else {
                0.0
            };
            param -= lr * update;
            p_vec[i] = param;
            m_vec[i] = beta2 * m_vec[i] + (1.0 - beta2) * grad;
        }

        let _ = self.from_cpu(&p_vec, p.shape(), p.dtype())?;
        Ok(Box::new(grim_tensor::backend::ReadyHandle))
    }
}
