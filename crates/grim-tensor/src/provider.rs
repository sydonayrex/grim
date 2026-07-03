//! `TensorProvider` — abstraction over a checkpoint source. Both GGUF and
//! safetensors-backed readers implement this; `WeightSource` (in
//! `grim-nn`) walks it depth-first by prefix.

use crate::dtype::{DType, QuantProvenance};
use crate::error::Result;

/// Resolved-at-load dtype + provenance for a tensor inside a checkpoint.
/// Read from the checkpoint's per-tensor metadata (GGUF kv, safetensors
/// metadata), with call sites providing defaults.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub dtype: DType,
    pub provenance: QuantProvenance,
    pub shape: Vec<usize>,
}

/// Raw byte source for a single tensor. Backends convert to their native
/// layout (F32 vec on CPU, raw bytes + scale/zero on ROCm, ...) when
/// materializing a tensor from `TensorProvider`.
pub trait TensorProvider: Send + Sync {
    /// Look up a tensor by slash-separated path (e.g. `"model.layers.0.wq"`).
    fn get(&self, name: &str) -> Result<RawTensor>;
    /// Optional hint — metadata the loader wants to expose without
    /// materializing the full tensor (shape, dtype, provenance).
    fn meta(&self, name: &str) -> Result<TensorMeta>;
}

/// Raw tensors read off disk but not yet on a device.
#[derive(Debug, Clone)]
pub struct RawTensor {
    pub bytes: Vec<u8>,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub provenance: QuantProvenance,
}
