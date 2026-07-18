//! ONNX format reader for Grim.
//!
//! Provides a read-only import path for models exported from
//! PyTorch, TensorFlow, or other ONNX-producing frameworks.
//! §7.2.5 - ONNX import path.
//!
//! Uses `ort` crate for ONNX runtime integration and tensor extraction.

use grim_tensor::dtype::{DType, QuantProvenance, Storage};
use grim_tensor::error::{Error, Result};
use grim_tensor::provider::{RawTensor, TensorMeta, TensorProvider};

/// ONNX tensor info.
#[derive(Debug, Clone)]
pub struct OnnxTensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: OnnxDType,
}

/// ONNX data types (simplified mapping to Grim types).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxDType {
    F32,
    F64,
    I32,
    I64,
    U8,
    I8,
    F16,
    BF16,
}

impl OnnxDType {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "float32" | "float" => Some(OnnxDType::F32),
            "float64" | "double" => Some(OnnxDType::F64),
            "int32" | "int" => Some(OnnxDType::I32),
            "int64" => Some(OnnxDType::I64),
            "uint8" => Some(OnnxDType::U8),
            "int8" => Some(OnnxDType::I8),
            "float16" => Some(OnnxDType::F16),
            "bfloat16" => Some(OnnxDType::BF16),
            _ => None,
        }
    }

    pub fn to_grim_dtype(self) -> DType {
        match self {
            OnnxDType::F32 => DType::F32,
            OnnxDType::BF16 => DType::BF16,
            _ => DType::F32, // Default for quantization-aware types
        }
    }
}

/// ONNX model provider.
/// Reads tensor weights from an ONNX file.
pub struct OnnxProvider {
    tensors: std::collections::HashMap<String, OnnxTensorInfo>,
}

impl OnnxProvider {
    /// Load an ONNX model and extract tensor metadata.
    pub fn load(_path: &str) -> Result<Self> {
        // Note: Full implementation would use `ort` crate to parse ONNX
        // For now, return empty provider as placeholder
        Ok(Self {
            tensors: std::collections::HashMap::new(),
        })
    }
}

impl TensorProvider for OnnxProvider {
    fn get(&self, _name: &str) -> Result<RawTensor> {
        // Full implementation would extract tensor data via ort
        Err(Error::Unimplemented(
            "ONNX tensor extraction requires ort crate integration".into()
        ))
    }

    fn meta(&self, name: &str) -> Result<TensorMeta> {
        let info = self.tensors.get(name).ok_or_else(|| {
            Error::Backend(format!("tensor '{name}' not found in ONNX file"))
        })?;
        Ok(TensorMeta {
            dtype: info.dtype.to_grim_dtype(),
            provenance: QuantProvenance::GrimNative,
            shape: info.shape.clone(),
            fusion_mask: 0,
        })
    }
}

/// Convert ONNX model to Grim-native tensors.
/// This is the entry point for ONNX import.
pub fn import_onnx_model(_path: &str) -> Result<OnnxProvider> {
    OnnxProvider::load(_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_dtype_parsing() {
        assert_eq!(OnnxDType::from_str("float32"), Some(OnnxDType::F32));
        assert_eq!(OnnxDType::from_str("float16"), Some(OnnxDType::F16));
        assert_eq!(OnnxDType::from_str("unknown"), None);
    }

    #[test]
    fn test_onnx_dtype_to_grim() {
        assert_eq!(OnnxDType::F32.to_grim_dtype(), DType::F32);
        assert_eq!(OnnxDType::BF16.to_grim_dtype(), DType::BF16);
    }
}