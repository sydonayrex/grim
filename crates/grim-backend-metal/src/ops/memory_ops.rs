//! memory_ops ops for MetalDevice — moved verbatim from lib.rs.

#[allow(unused_imports)]
use grim_tensor::dtype::{
    DType, FloatPackScheme, KQuantScheme, QuantFormat, QuantProvenance, Storage as DTypeStorage,
};
use grim_tensor::error::Result;
use grim_tensor::{BackendStorage, MemoryOps, Shape};

#[cfg(target_vendor = "apple")]
use objc2::rc::Retained;
#[cfg(target_vendor = "apple")]
use objc2::runtime::ProtocolObject;
#[cfg(target_vendor = "apple")]
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLComputePipelineState, MTLDevice, MTLSize,
};

use crate::*;

impl MemoryOps for MetalDevice {
    fn from_cpu_bytes(
        &self,
        data: &[u8],
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn BackendStorage>> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(ref inner) = self.inner {
                use objc2_metal::MTLResourceOptions;
                let buffer = inner
                    .device
                    .newBufferWithLength_options(
                        data.len() as u64,
                        MTLResourceOptions::StorageModeShared,
                    )
                    .ok_or_else(|| {
                        Error::from(MetalError::AllocationFailed(
                            "Failed to allocate Metal buffer".into(),
                        ))
                    })?;

                let contents = buffer.contents();
                if !contents.is_null() {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            data.as_ptr(),
                            contents as *mut u8,
                            data.len(),
                        );
                    }
                }

                return Ok(Box::new(MetalStorage {
                    buffer: Some(buffer),
                    data: None,
                    shape: shape.clone(),
                    dtype,
                    provenance: QuantProvenance::GrimNative,
                }));
            }
        }
        #[cfg(target_vendor = "apple")]
        {
            Ok(Box::new(MetalStorage {
                buffer: None,
                data: Some(std::sync::Mutex::new(data.to_vec())),
                shape: shape.clone(),
                dtype,
                provenance: QuantProvenance::GrimNative,
            }))
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Ok(Box::new(MetalStorage {
                data: std::sync::Mutex::new(data.to_vec()),
                shape: shape.clone(),
                dtype,
                provenance: QuantProvenance::GrimNative,
            }))
        }
    }
}
