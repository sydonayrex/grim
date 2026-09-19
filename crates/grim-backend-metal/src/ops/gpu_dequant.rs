//! gpu_dequant ops for MetalDevice — moved verbatim from lib.rs.

#[allow(unused_imports)]
use grim_tensor::dtype::{
    DType, FloatPackScheme, KQuantScheme, QuantFormat, QuantProvenance, Storage as DTypeStorage,
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

impl grim_format::convert::GpuDequant for MetalDevice {
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
