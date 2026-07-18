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
    /// Kernel fusion dispatch hints (bit0 = RmsNormMatMul,
    /// bit1 = QkvAttention). Zero = no fusion requested. Source: the
    /// `.grim` tensor capability extension's `fusion_mask` field.
    pub fusion_mask: u8,
}

impl TensorMeta {
    /// `true` if RmsNormMatMul fusion (bit0) is requested.
    pub fn has_rmsnorm_matmul_fusion(&self) -> bool {
        self.fusion_mask & 0b01 != 0
    }
    /// `true` if QkvAttention fusion (bit1) is requested.
    pub fn has_qkv_attention_fusion(&self) -> bool {
        self.fusion_mask & 0b10 != 0
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_meta(fusion_mask: u8) -> TensorMeta {
        TensorMeta {
            dtype: DType::F32,
            provenance: QuantProvenance::GrimNative,
            shape: vec![4, 4],
            fusion_mask,
        }
    }

    /// Phase 7.3: bit0 (RmsNormMatMul) toggle.
    #[test]
    fn tensor_meta_rmsnorm_matmul_accessor() {
        let zero = sample_meta(0);
        assert!(!zero.has_rmsnorm_matmul_fusion());

        let bit0 = sample_meta(0b01);
        assert!(bit0.has_rmsnorm_matmul_fusion());
        assert!(!bit0.has_qkv_attention_fusion());

        let both = sample_meta(0b11);
        assert!(both.has_rmsnorm_matmul_fusion());
        assert!(both.has_qkv_attention_fusion());
    }

    /// Phase 7.3: bit1 (QkvAttention) toggle.
    #[test]
    fn tensor_meta_qkv_attention_accessor() {
        let zero = sample_meta(0);
        assert!(!zero.has_qkv_attention_fusion());

        let bit1 = sample_meta(0b10);
        assert!(bit1.has_qkv_attention_fusion());
        assert!(!bit1.has_rmsnorm_matmul_fusion());
    }
}
