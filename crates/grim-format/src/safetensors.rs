//! Minimal safetensors reader. Parses the JSON header, then lazy-reads
//! tensor bytes by offset.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

use grim_tensor::dtype::{ArithType, DType, Storage};
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
        self.dims
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .unwrap_or(usize::MAX)
    }
    pub fn byte_size(&self) -> usize {
        // FMT-10 fix: unknown dtype tags previously fell through to a 4-byte
        // default, silently mis-sizing buffers for tensors we don't actually
        // support. `grim_dtype()` already rejects unknown tags, so an unknown
        // tag here means the tensor is unsupported — report 0 bytes rather than
        // guessing, which makes any downstream allocation fail loudly instead
        // of reading the wrong number of bytes.
        let elem = match self.dtype_tag.as_str() {
            "F32" | "I32" | "U32" => 4,
            "F16" | "BF16" => 2,
            "F64" | "I64" | "U64" => 8,
            "I8" | "U8" => 1,
            _ => 0,
        };
        self.elem_count().saturating_mul(elem)
    }
}

/// Parse the safetensors header JSON and return tensor index entries and header metadata.
/// Does NOT read tensor data — call `read_safetensor_bytes` per tensor.
pub fn read_safetensors_header<R: Read + Seek>(
    mut reader: R,
) -> Result<(
    HashMap<String, SafetensorInfo>,
    Option<HashMap<String, String>>,
    u64,
)> {
    let mut len_bytes = [0u8; 8];
    reader.read_exact(&mut len_bytes)?;
    let header_len = u64::from_le_bytes(len_bytes) as usize;

    if header_len > 100_000_000 {
        return Err(Error::Backend(format!(
            "safetensors header_len {header_len} exceeds safety limit 100MB"
        )));
    }

    let mut header_json = vec![0u8; header_len];
    reader.read_exact(&mut header_json)?;

    let header: serde_json::Value = serde_json::from_slice(&header_json)
        .map_err(|e| Error::Backend(format!("invalid safetensors JSON header: {e}")))?;

    let header_map = header
        .as_object()
        .ok_or_else(|| Error::Backend("safetensors header is not a JSON object".into()))?;

    let file_len = reader.seek(SeekFrom::End(0))?;

    let mut tensors = HashMap::new();
    let mut metadata = None;
    let mut total_data = 0u64;

    for (key, val) in header_map {
        if key == "__metadata__" {
            if let Some(m) = val.as_object() {
                let mut map = HashMap::new();
                for (k, v) in m {
                    if let Some(s) = v.as_str() {
                        map.insert(k.clone(), s.to_string());
                    }
                }
                metadata = Some(map);
            }
            continue;
        }

        let obj = match val.as_object() {
            Some(o) => o,
            None => continue,
        };

        let dtype_tag = obj
            .get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        if dtype_tag.is_empty() {
            return Err(Error::Backend(format!("missing dtype for '{key}'")));
        }

        let shape = obj
            .get("shape")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|v| v.as_u64().unwrap_or(0) as usize)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        if shape.contains(&0) {
            return Err(Error::Backend(format!(
                "safetensors '{key}' shape contains zero dimension"
            )));
        }

        let data_offsets = obj
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| Error::Backend(format!("missing data_offsets for '{key}'")))?;

        let data_start = data_offsets
            .get(0)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| Error::Backend(format!("invalid data_offsets[0] for '{key}'")))?;
        let data_end = data_offsets
            .get(1)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| Error::Backend(format!("invalid data_offsets[1] for '{key}'")))?;

        if data_start > data_end {
            return Err(Error::Backend(format!(
                "safetensors '{key}' data_start {data_start} > data_end {data_end}"
            )));
        }

        let data_region_start = 8 + header_len as u64;
        let abs_end = data_region_start
            .checked_add(data_end)
            .ok_or_else(|| Error::Backend(format!("safetensors '{key}' offset overflow")))?;

        if abs_end > file_len {
            return Err(Error::Backend(format!(
                "safetensors '{key}' offset {abs_end} exceeds file length {file_len}"
            )));
        }

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
    Ok((tensors, metadata, data_region_start))
}

/// Read one tensor's raw bytes from a safetensors file.
pub fn read_safetensor_bytes<R: Read + Seek>(
    reader: &mut R,
    info: &SafetensorInfo,
    data_region_start: u64,
) -> Result<Vec<u8>> {
    // Safetensors data_offsets in the JSON header are relative to the
    // data section, which starts at `data_region_start` (the header length
    // prefix + JSON header).  Add this base offset to reach the actual
    // file position.
    let start = data_region_start + info.data_start;
    let size = (info.data_end - info.data_start) as usize;
    reader.seek(SeekFrom::Start(start))?;
    let mut buf = vec![0u8; size];
    reader.read_exact(&mut buf)?;
    Ok(buf)
}

impl SafetensorInfo {
    /// Map safetensors dtype tag to Grim DType.
    pub fn grim_dtype(&self) -> Result<DType> {
        match self.dtype_tag.as_str() {
            "F32" => Ok(DType::F32),
            "BF16" => Ok(DType::BF16),
            "F16" => Ok(DType::F16),
            "I8" | "U8" => Ok(DType {
                arith: ArithType::U8,
                storage: Storage::Native,
            }),
            other => Err(Error::Backend(format!(
                "Unsupported safetensors dtype: '{other}'"
            ))),
        }
    }
}

// Note: serde_json is needed for safetensors parsing. We add it in Cargo.toml.
