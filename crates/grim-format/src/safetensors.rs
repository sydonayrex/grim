//! Minimal safetensors reader. Parses the JSON header, then lazy-reads
//! tensor bytes by offset.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

use grim_tensor::dtype::DType;
use grim_tensor::error::{Error, Result};

/// Parsed safetensors header. Each tensor entry contains its shape, dtype
/// tag, and the [start, end) byte offset within the file.
#[derive(Debug, Clone)]
pub struct SafetensorInfo {
    pub name: String,
    pub dims: Vec<usize>,
    /// Dtype encoded as the safetensors dtype string ("F32", "F16", etc.)
    pub dtype_tag: String,
    pub data_start: u64,
    pub data_end: u64,
}

impl SafetensorInfo {
    pub fn shape(&self) -> Vec<usize> {
        self.dims.clone()
    }
    pub fn elem_count(&self) -> usize {
        self.dims.iter().product()
    }
    pub fn byte_size(&self) -> usize {
        let elem = match self.dtype_tag.as_str() {
            "F32" | "I32" | "U32" => 4,
            "F16" | "BF16" => 2,
            "F64" | "I64" | "U64" => 8,
            "I8" | "U8" => 1,
            _ => 4,
        };
        self.elem_count() * elem
    }
}

/// Parse the safetensors header JSON and return tensor index entries.
/// Does NOT read tensor data — call `read_safetensor_bytes` per tensor.
pub fn read_safetensors_header<R: Read + Seek>(mut reader: R) -> Result<(HashMap<String, SafetensorInfo>, u64)> {
    let mut len_bytes = [0u8; 8];
    reader.read_exact(&mut len_bytes)?;
    let header_len = u64::from_le_bytes(len_bytes) as usize;

    let mut header_json = vec![0u8; header_len];
    reader.read_exact(&mut header_json)?;

    let header: serde_json::Value = serde_json::from_slice(&header_json)
        .map_err(|e| Error::Backend(format!("invalid safetensors JSON header: {e}")))?;

    let header_map = header.as_object()
        .ok_or_else(|| Error::Backend("safetensors header is not a JSON object".into()))?;

    let mut tensors = HashMap::new();
    let mut total_data = 0u64;
    for (key, val) in header_map {
        if key == "__metadata__" {
            continue;
        }
        let obj = val.as_object()
            .ok_or_else(|| Error::Backend(format!("safetensors entry '{key}' is not an object")))?;

        let dtype_tag = obj.get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::Backend(format!("missing dtype for '{key}'")))?
            .to_string();

        let shape = obj.get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| Error::Backend(format!("missing shape for '{key}'")))?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect::<Vec<_>>();

        let data_offsets = obj.get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| Error::Backend(format!("missing data_offsets for '{key}'")))?;

        let data_start = data_offsets[0].as_u64().unwrap_or(0);
        let data_end = data_offsets[1].as_u64().unwrap_or(0);

        total_data = total_data.max(data_end);
        let info = SafetensorInfo {
            name: key.clone(),
            dims: shape,
            dtype_tag,
            data_start,
            data_end,
        };
        tensors.insert(key.clone(), info);
    }

    // Data section starts at header_len + 8 (the length prefix)
    let data_region_start = 8 + header_len as u64;
    Ok((tensors, data_region_start))
}

/// Read one tensor's raw bytes from a safetensors file.
pub fn read_safetensor_bytes<R: Read + Seek>(
    reader: &mut R,
    info: &SafetensorInfo,
    data_region_start: u64,
) -> Result<Vec<u8>> {
    let start = data_region_start + info.data_start;
    let size = (info.data_end - info.data_start) as usize;
    reader.seek(SeekFrom::Start(start))?;
    let mut buf = vec![0u8; size];
    reader.read_exact(&mut buf)?;
    Ok(buf)
}

impl SafetensorInfo {
    /// Map safetensors dtype tag to Grim DType.
    pub fn grim_dtype(&self) -> DType {
        match self.dtype_tag.as_str() {
            "F32" => DType::F32,
            "BF16" => DType::BF16,
            _ => DType::F32,
        }
    }
}

// Note: serde_json is needed for safetensors parsing. We add it in Cargo.toml.