//! CPU tensor storage: contiguous `Vec<f32>` on the host.

use std::sync::Arc;

use grim_tensor::dtype::QuantProvenance;
use grim_tensor::{DType, Shape};

/// Contiguous `f32` buffer. v1 storage; quantized/half-precision in own crates.
#[derive(Debug, Clone)]
pub struct CpuStorage {
    pub(crate) data: Arc<Vec<f32>>,
    pub(crate) shape: Shape,
    pub(crate) dtype: DType,
    pub(crate) provenance: QuantProvenance,
    pub(crate) quant_scales: Option<Vec<f32>>,
    pub(crate) raw_bytes: Option<Arc<Vec<u8>>>,
}

impl CpuStorage {
    pub fn new(data: Vec<f32>, shape: Shape, dtype: DType) -> Self {
        Self {
            data: Arc::new(data),
            shape,
            dtype,
            provenance: QuantProvenance::GrimNative,
            quant_scales: None,
            raw_bytes: None,
        }
    }

    pub fn from_raw_bytes(bytes: Vec<u8>, shape: Shape, dtype: DType) -> Self {
        Self {
            data: Arc::new(Vec::new()),
            shape,
            dtype,
            provenance: QuantProvenance::GrimNative,
            quant_scales: None,
            raw_bytes: Some(Arc::new(bytes)),
        }
    }

    pub fn from_arc(data: Arc<Vec<f32>>, shape: Shape, dtype: DType) -> Self {
        Self {
            data,
            shape,
            dtype,
            provenance: QuantProvenance::GrimNative,
            quant_scales: None,
            raw_bytes: None,
        }
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

    pub fn with_quant_scales(mut self, scales: Vec<f32>) -> Self {
        self.quant_scales = Some(scales);
        self
    }
}
