//! `TensorProvider` implementations for GGUF and safetensors files.
//!
//! Each implements `TensorProvider` so `WeightSource` can walk checkpoints
//! without caring whether they came from GGUF or safetensors.

use std::fs::File;
use std::io::BufReader;

use grim_tensor::dtype::{DType, QuantProvenance};
use grim_tensor::error::{Error, Result};
use grim_tensor::provider::{RawTensor, TensorMeta, TensorProvider};

use crate::gguf::{read_gguf, read_tensor_bytes, GgufFile, GgufTensorInfo};
use crate::safetensors::{read_safetensor_bytes, read_safetensors_header, SafetensorInfo};

/// GGUF-backed `TensorProvider`. Holds the parsed file index and wraps a
/// `BufReader<File>` for lazy tensor reads.
pub struct GgufProvider {
    file: GgufFile,
    reader: std::sync::Mutex<BufReader<File>>,
    tensors: std::collections::HashMap<String, GgufTensorInfo>,
}

impl GgufProvider {
    pub fn open(path: &str) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| Error::Backend(format!("cannot open GGUF file '{path}': {e}")))?;
        let reader = BufReader::new(file);
        let gguf = read_gguf(reader)?;
        let mut tensors = std::collections::HashMap::new();
        for t in &gguf.tensors {
            tensors.insert(t.name.clone(), t.clone());
        }
        let file = File::open(path)
            .map_err(|e| Error::Backend(format!("cannot reopen GGUF file '{path}': {e}")))?;
        let reader = std::sync::Mutex::new(BufReader::new(file));
        Ok(Self {
            file: gguf,
            reader,
            tensors,
        })
    }

    pub fn metadata(&self, key: &str) -> Option<&crate::gguf::GgufValue> {
        self.file.metadata.get(key)
    }

    pub fn architecture(&self) -> Option<&str> {
        self.metadata("general.architecture")?.as_str()
    }
}

impl TensorProvider for GgufProvider {
    fn get(&self, name: &str) -> Result<RawTensor> {
        let info = self.tensors.get(name).ok_or_else(|| {
            Error::Backend(format!("tensor '{name}' not found in GGUF file"))
        })?;
        let mut reader = self.reader.lock().unwrap();
        let bytes = read_tensor_bytes(&mut *reader, &self.file, info)?;
        Ok(RawTensor {
            bytes,
            shape: info.shape(),
            dtype: DType::F32,
            provenance: QuantProvenance::GrimNative,
        })
    }

    fn meta(&self, name: &str) -> Result<TensorMeta> {
        let info = self.tensors.get(name).ok_or_else(|| {
            Error::Backend(format!("tensor '{name}' not found in GGUF file"))
        })?;
        Ok(TensorMeta {
            dtype: DType::F32,
            provenance: QuantProvenance::GrimNative,
            shape: info.shape(),
        })
    }
}

/// Safetensors-backed `TensorProvider`.
pub struct SafetensorsProvider {
    info: std::collections::HashMap<String, SafetensorInfo>,
    reader: std::sync::Mutex<BufReader<File>>,
    data_region_start: u64,
}

impl SafetensorsProvider {
    pub fn open(path: &str) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| Error::Backend(format!("cannot open safetensors file '{path}': {e}")))?;
        let reader = BufReader::new(file);
        let (info, data_region_start) = read_safetensors_header(reader)?;
        let file = File::open(path)
            .map_err(|e| Error::Backend(format!("cannot reopen safetensors file '{path}': {e}")))?;
        let reader = std::sync::Mutex::new(BufReader::new(file));
        Ok(Self {
            info,
            reader,
            data_region_start,
        })
    }
}

impl TensorProvider for SafetensorsProvider {
    fn get(&self, name: &str) -> Result<RawTensor> {
        let info = self.info.get(name).ok_or_else(|| {
            Error::Backend(format!("tensor '{name}' not found in safetensors file"))
        })?;
        let mut reader = self.reader.lock().unwrap();
        let bytes = read_safetensor_bytes(&mut *reader, info, self.data_region_start)?;
        Ok(RawTensor {
            bytes,
            shape: info.shape(),
            dtype: info.grim_dtype(),
            provenance: QuantProvenance::GrimNative,
        })
    }

    fn meta(&self, name: &str) -> Result<TensorMeta> {
        let info = self.info.get(name).ok_or_else(|| {
            Error::Backend(format!("tensor '{name}' not found in safetensors file"))
        })?;
        Ok(TensorMeta {
            dtype: info.grim_dtype(),
            provenance: QuantProvenance::GrimNative,
            shape: info.shape(),
        })
    }
}