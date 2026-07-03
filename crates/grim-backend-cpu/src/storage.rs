//! CPU-side tensor storage: a contiguous `Vec<f32>` on the host.

use std::sync::Arc;

use grim_tensor::dtype::QuantProvenance;
use grim_tensor::{DType, Shape};

/// CPU buffer holding contiguous `f32` data. This is the v1 storage
/// type for the CPU backend; quantized and half-precision backends
/// will introduce separate storage variants owned by those crates.
#[derive(Debug, Clone)]
pub struct CpuStorage {
    pub(crate) data: Arc<Vec<f32>>,
    pub(crate) shape: Shape,
    pub(crate) dtype: DType,
    pub(crate) provenance: QuantProvenance,
}

impl CpuStorage {
    pub fn new(data: Vec<f32>, shape: Shape, dtype: DType) -> Self {
        Self { data: Arc::new(data), shape, dtype, provenance: QuantProvenance::GrimNative }
    }

    pub fn from_arc(data: Arc<Vec<f32>>, shape: Shape, dtype: DType) -> Self {
        Self { data, shape, dtype, provenance: QuantProvenance::GrimNative }
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_arc(&self) -> Arc<Vec<f32>> {
        Arc::clone(&self.data)
    }

    pub fn with_provenance(mut self, provenance: QuantProvenance) -> Self {
        self.provenance = provenance;
        self
    }
}
