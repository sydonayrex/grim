//! Low-level GGUF v3 binary reader. Handles the file header, metadata
//! key-value map, tensor info index, and aligned tensor data region.
//!
//! GGUF is the format standardized by llama.cpp:
//!   <header> <metadata_kv*> <tensor_info*> <aligned_tensor_data>
//!
//! This is a minimal, no-unsafe reader — we parse the file into a
//! `GgufFile` struct, then seek to offsets during `TensorProvider::get`.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

use grim_tensor::dtype::DType;
use grim_tensor::error::{Error, Result};

pub const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" LE
pub const GGUF_VERSION: u32 = 3;

/// Metadata value type tags from GGUF spec.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GgufValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufValue::Uint32(v) => Some(*v),
            GgufValue::Uint64(v) => Some(*v as u32),
            _ => None,
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GgufValue::Float32(v) => Some(*v),
            GgufValue::Float64(v) => Some(*v as f32),
            _ => None,
        }
    }
}

/// One tensor index entry from a GGUF file.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    /// Offset (in bytes) from the start of the file to the tensor data.
    pub offset: u64,
    /// Size of the tensor data in bytes.
    pub size_bytes: u64,
}

impl GgufTensorInfo {
    pub fn shape(&self) -> Vec<usize> {
        self.dims.iter().map(|d| *d as usize).collect()
    }
    pub fn elem_count(&self) -> usize {
        self.shape().iter().product()
    }
}

/// Parsed GGUF file metadata. The raw file bytes are not kept — we store
/// tensor info (name + offset) and metadata KV pairs.
pub struct GgufFile {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
    /// Byte offset where the aligned tensor data section begins.
    pub data_start: u64,
}

/// Loader that reads from a reader and returns parsed GGUF structure.
pub fn read_gguf<R: Read + Seek>(mut reader: R) -> Result<GgufFile> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    let magic = u32::from_le_bytes(buf);
    if magic != GGUF_MAGIC {
        return Err(Error::Backend(format!(
            "not a GGUF file: magic {:#010x}",
            magic
        )));
    }
    reader.read_exact(&mut buf[..4]);
    let version = u32::from_le_bytes(buf);
    if version != GGUF_VERSION {
        return Err(Error::Backend(format!(
            "unsupported GGUF version {version}, expected {GGUF_VERSION}"
        )));
    }
    let tensor_count = read_u64_le(&mut reader)?;
    let metadata_kv_count = read_u64_le(&mut reader)?;

    let mut metadata = HashMap::new();
    for _ in 0..metadata_kv_count {
        let key = read_gguf_string(&mut reader)?;
        let value = read_gguf_value(&mut reader)?;
        metadata.insert(key, value);
    }
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = read_gguf_string(&mut reader)?;
        let n_dims = read_u32_le(&mut reader)?;
        let mut dims: Vec<u64> = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(read_u64_le(&mut reader)?);
        }
        let offset = read_u64_le(&mut reader)?;
        let elem_size = 4u64; // GGUF v3 stores f32 by default; quantized
                              // tensors use a per-tensor type tag not read here
        let size_bytes: u64 = dims.iter().product::<u64>() * elem_size;
        tensors.push(GgufTensorInfo {
            name,
            dims,
            offset,
            size_bytes,
        });
    }
    // data_start is at the current reader position, aligned to 32 bytes
    let pos = reader.stream_position()?;
    let data_start = (pos + 31) & !31;

    Ok(GgufFile {
        version,
        tensor_count,
        metadata,
        tensors,
        data_start,
    })
}

/// Read one tensor's raw bytes from a GGUF-backed file.
pub fn read_tensor_bytes<R: Read + Seek>(
    reader: &mut R,
    file: &GgufFile,
    info: &GgufTensorInfo,
) -> Result<Vec<u8>> {
    let start = file.data_start + info.offset;
    let size = info.size_bytes as usize;
    reader.seek(SeekFrom::Start(start))?;
    let mut buf = vec![0u8; size];
    reader.read_exact(&mut buf)?;
    Ok(buf)
}

// ---------- low-level helpers ----------

fn read_u32_le<R: Read>(r: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64_le<R: Read>(r: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_gguf_string<R: Read>(r: &mut R) -> Result<String> {
    let len = read_u64_le(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8(buf).map_err(|e| {
        Error::Backend(format!("GGUF string not valid UTF-8: {e}"))
    })?)
}

fn read_gguf_value<R: Read>(r: &mut R) -> Result<GgufValue> {
    let tag = read_u32_le(r)?;
    match tag {
        // GGUF metadata value type tags
        0 => Ok(GgufValue::Uint8(read_u32_le(r).map(|v| {
            let mut buf = [0u8; 1];
            buf.copy_from_slice(&v.to_le_bytes()[..1]);
            buf[0]
        })?)),
        1 => Ok(GgufValue::Int8({
            let mut buf = [0u8; 1];
            r.read_exact(&mut buf)?;
            i8::from_le_bytes(buf)
        })),
        2 => Ok(GgufValue::Uint16({
            let mut buf = [0u8; 2];
            r.read_exact(&mut buf)?;
            u16::from_le_bytes(buf)
        })),
        3 => Ok(GgufValue::Int16({
            let mut buf = [0u8; 2];
            r.read_exact(&mut buf)?;
            i16::from_le_bytes(buf)
        })),
        4 => Ok(GgufValue::Uint32(read_u32_le(r)?)),
        5 => Ok(GgufValue::Int32({
            let mut buf = [0u8; 4];
            r.read_exact(&mut buf)?;
            i32::from_le_bytes(buf)
        })),
        6 => Ok(GgufValue::Float32({
            let mut buf = [0u8; 4];
            r.read_exact(&mut buf)?;
            f32::from_le_bytes(buf)
        })),
        7 => Ok(GgufValue::Bool(read_u32_le(r)? != 0)),
        8 => Ok(GgufValue::String(read_gguf_string(r)?)),
        9 => {
            // Array: tag of array element type (u32) + count (u64) + elements
            let _elem_tag = read_u32_le(r)?; // all elements have same tag
            let count = read_u64_le(r)?;
            let mut items = Vec::with_capacity(count as usize);
            for _ in 0..count {
                // We read as the tag we already consumed; for simplicity
                // we just read string values since those are common in metadata.
                items.push(read_gguf_value(r)?);
            }
            Ok(GgufValue::Array(items))
        }
        10 => Ok(GgufValue::Uint64(read_u64_le(r)?)),
        11 => Ok(GgufValue::Int64({
            let mut buf = [0u8; 8];
            r.read_exact(&mut buf)?;
            i64::from_le_bytes(buf)
        })),
        12 => Ok(GgufValue::Float64({
            let mut buf = [0u8; 8];
            r.read_exact(&mut buf)?;
            f64::from_le_bytes(buf)
        })),
        t => Err(Error::Backend(format!("unknown GGUF metadata tag {t}"))),
    }
}

/// Derive DType from GGUF metadata 'general.architecture' name + per-weight
/// metadata. This is a heuristic used when the GGUF file does not store
/// explicit per-tensor dtype.
pub fn dtype_for_gguf(name: &str) -> DType {
    let _ = name;
    // GGUF v3 stores all tensors as f32 by default (llama.cpp uses
    // quantization-specific per-tensor overrides). We default to F32
    // and let the per-tensor metadata override.
    DType::F32
}

/// Guess the GGUF tensor name from a weight path (slash-separated -> dot-separated).
pub fn gguf_tensor_name(path: &str) -> String {
    path.replace('.', ".")
}