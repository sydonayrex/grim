//! elementwise_ops ops for MetalDevice — moved verbatim from lib.rs.

use grim_tensor::backend::ComputeHandle;
#[allow(unused_imports)]
use grim_tensor::dtype::{
    DType, FloatPackScheme, KQuantScheme, QuantFormat, QuantProvenance, Storage as DTypeStorage,
};
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendStorage, CoreTensorOps, ElementwiseOps, Shape};

#[cfg(target_vendor = "apple")]
use objc2::rc::Retained;
#[cfg(target_vendor = "apple")]
use objc2::runtime::ProtocolObject;
#[cfg(target_vendor = "apple")]
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLComputePipelineState, MTLDevice, MTLSize,
};

use crate::*;

impl ElementwiseOps for MetalDevice {
    fn mul_scalar(
        &self,
        x: &dyn BackendStorage,
        scalar: f32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                return self.run_unary(
                    inner,
                    &inner.pipelines.mul_scalar,
                    x,
                    Some(scalar),
                    out_shape,
                    Some(1),
                    3,
                );
            }
        }
        let x_vec = x.to_cpu_vec_f32()?;
        let res: Vec<f32> = x_vec.into_iter().map(|v| v * scalar).collect();
        let out_storage = self.from_cpu(&res, out_shape, x.dtype())?;
        Ok((out_storage, Box::new(grim_tensor::backend::ReadyHandle)))
    }

    fn sqrt(
        &self,
        x: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                return self.run_unary(inner, &inner.pipelines.sqrt, x, None, out_shape, None, 2);
            }
        }
        let x_vec = x.to_cpu_vec_f32()?;
        let res: Vec<f32> = x_vec.into_iter().map(|v| v.sqrt()).collect();
        let out_storage = self.from_cpu(&res, out_shape, x.dtype())?;
        Ok((out_storage, Box::new(grim_tensor::backend::ReadyHandle)))
    }

    fn recip(
        &self,
        x: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                return self.run_unary(inner, &inner.pipelines.recip, x, None, out_shape, None, 2);
            }
        }
        let x_vec = x.to_cpu_vec_f32()?;
        let res: Vec<f32> = x_vec.into_iter().map(|v| 1.0 / v).collect();
        let out_storage = self.from_cpu(&res, out_shape, x.dtype())?;
        Ok((out_storage, Box::new(grim_tensor::backend::ReadyHandle)))
    }

    fn sub(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                if a.dtype().arith != ArithType::F32 || b.dtype().arith != ArithType::F32 {
                    return Err(Error::from(MetalError::UnsupportedDType(a.dtype())));
                }
                return self.run_elementwise(inner, &inner.pipelines.sub, a, b, out);
            }
        }
        let a_vec = a.to_cpu_vec_f32()?;
        let b_vec = b.to_cpu_vec_f32()?;
        if a_vec.len() != b_vec.len() {
            return Err(Error::Backend(format!(
                "Metal sub length mismatch: {} vs {}",
                a_vec.len(),
                b_vec.len()
            )));
        }
        let res: Vec<f32> = a_vec.iter().zip(b_vec.iter()).map(|(x, y)| x - y).collect();
        let out_storage = self.from_cpu(&res, out, a.dtype())?;
        Ok((out_storage, Box::new(grim_tensor::backend::ReadyHandle)))
    }

    fn reduce_sum(&self, x: &dyn BackendStorage) -> Result<f32> {
        let v = x.to_cpu_vec_f32()?;
        if v.is_empty() {
            return Err(Error::Backend("reduce_sum: empty tensor".into()));
        }
        Ok(v.iter().sum())
    }

    fn reduce_max(&self, x: &dyn BackendStorage) -> Result<f32> {
        let v = x.to_cpu_vec_f32()?;
        v.iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| Error::Backend("reduce_max: empty tensor".into()))
    }

    fn argmax(&self, x: &dyn BackendStorage) -> Result<u32> {
        let v = x.to_cpu_vec_f32()?;
        v.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .ok_or_else(|| Error::Backend("argmax: empty tensor".into()))
    }
}
