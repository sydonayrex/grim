//! The user-facing `Tensor` — a thin handle over a `BackendStorage`.

use std::sync::Arc;

use crate::backend::BackendStorage;
use crate::dtype::{DType, Device, QuantProvenance};
use crate::error::{Error, Result};
use crate::shape::Shape;

/// A tensor. Shares its underlying storage via `Arc<dyn BackendStorage>`.
/// Layout is fully-static via `Shape`; v1 only supports row-major walks.
#[derive(Clone)]
pub struct Tensor {
    storage: Arc<dyn BackendStorage>,
    layout: Shape,
    dtype: DType,
    provenance: QuantProvenance,
    device: Device,
}

impl Tensor {
    pub fn new(
        storage: Arc<dyn BackendStorage>,
        layout: Shape,
        dtype: DType,
        provenance: QuantProvenance,
        device: Device,
    ) -> Self {
        Self { storage, layout, dtype, provenance, device }
    }

    pub fn storage(&self) -> &Arc<dyn BackendStorage> {
        &self.storage
    }
    pub fn shape(&self) -> &Shape {
        &self.layout
    }
    pub fn dtype(&self) -> DType {
        self.dtype.clone()
    }
    pub fn arith(&self) -> crate::dtype::ArithType {
        self.dtype.arith
    }
    pub fn provenance(&self) -> &QuantProvenance {
        &self.provenance
    }
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Transfer to host `Vec<f32>` — slow path, used by tests and sampling.
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        self.storage.to_cpu_vec_f32()
    }

    /// Shape-checked view access; returns shape mismatch rather than
    /// trigonometrically-obvious "panic on next op".
    pub fn expect_shape(&self, expected: &Shape) -> Result<()> {
        if &self.layout == expected {
            Ok(())
        } else {
            Err(Error::ShapeMismatch {
                expected: expected.dims().to_vec(),
                got: self.layout.dims().to_vec(),
            })
        }
    }

    /// Shape-check this tensor is on the expected device.
    pub fn expect_device(&self, expected: &Device) -> Result<()> {
        if self.device == *expected {
            Ok(())
        } else {
            Err(Error::DeviceMismatch(format!(
                "expected device {expected}, got {}",
                self.device
            )))
        }
    }
}
