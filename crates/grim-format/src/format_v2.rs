//! Custom `.grim` V2 (Outlier-Aware Streams & Wave64) Format representation.
//! Incorporates variable bitrates, outlier streams, and RDNA-aligned tiling layouts.

use std::io::{Read, Write};
use grim_tensor::error::{Error, Result};

/// FuckingSorcery magic bytes for `.grim` V2.
pub const FUCKING_SORCERY: [u8; 5] = [0x47, 0x52, 0x49, 0x4d, 0x02]; // "GRIM\x02"

/// Header of `.grim` V2 model format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrimV2Header {
    pub magic: [u8; 5],
    pub metadata_len: u64,
    pub num_tensors: u32,
}

impl GrimV2Header {
    pub fn new(num_tensors: u32, metadata_len: u64) -> Self {
        Self {
            magic: FUCKING_SORCERY,
            metadata_len,
            num_tensors,
        }
    }

    pub fn write<W: Write>(&self, w: &mut W) -> Result<()> {
        w.write_all(&self.magic)
            .map_err(|e| Error::Backend(format!("Header write failed: {e}")))?;
        w.write_all(&self.metadata_len.to_le_bytes())
            .map_err(|e| Error::Backend(format!("Header write failed: {e}")))?;
        w.write_all(&self.num_tensors.to_le_bytes())
            .map_err(|e| Error::Backend(format!("Header write failed: {e}")))?;
        Ok(())
    }

    pub fn read<R: Read>(r: &mut R) -> Result<Self> {
        let mut magic = [0u8; 5];
        r.read_exact(&mut magic)
            .map_err(|e| Error::Backend(format!("Header read failed: {e}")))?;
        if magic != FUCKING_SORCERY {
            return Err(Error::Backend(format!(
                "Invalid Header: FuckingSorcery magic mismatched. Expected {:?}, got {:?}",
                FUCKING_SORCERY, magic
            )));
        }

        let mut metadata_len_bytes = [0u8; 8];
        r.read_exact(&mut metadata_len_bytes)
            .map_err(|e| Error::Backend(format!("Header read failed: {e}")))?;
        let metadata_len = u64::from_le_bytes(metadata_len_bytes);

        let mut num_tensors_bytes = [0u8; 4];
        r.read_exact(&mut num_tensors_bytes)
            .map_err(|e| Error::Backend(format!("Header read failed: {e}")))?;
        let num_tensors = u32::from_le_bytes(num_tensors_bytes);

        Ok(Self {
            magic,
            metadata_len,
            num_tensors,
        })
    }
}

/// Registry entry for a single tensor inside V2 file format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrimV2TensorEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub base_bitwidth: u8,
    pub payload_offset: u64,
    pub payload_size: u64,
    pub outlier_count: u32,
    pub outlier_offset: u64,
}

impl GrimV2TensorEntry {
    pub fn write<W: Write>(&self, w: &mut W) -> Result<()> {
        let name_bytes = self.name.as_bytes();
        w.write_all(&(name_bytes.len() as u16).to_le_bytes())
            .map_err(|e| Error::Backend(format!("Tensor entry write failed: {e}")))?;
        w.write_all(name_bytes)
            .map_err(|e| Error::Backend(format!("Tensor entry write failed: {e}")))?;

        w.write_all(&(self.shape.len() as u8).to_le_bytes())
            .map_err(|e| Error::Backend(format!("Tensor entry write failed: {e}")))?;
        for dim in &self.shape {
            w.write_all(&(*dim as u32).to_le_bytes())
                .map_err(|e| Error::Backend(format!("Tensor entry write failed: {e}")))?;
        }

        w.write_all(&[self.base_bitwidth])
            .map_err(|e| Error::Backend(format!("Tensor entry write failed: {e}")))?;
        w.write_all(&self.payload_offset.to_le_bytes())
            .map_err(|e| Error::Backend(format!("Tensor entry write failed: {e}")))?;
        w.write_all(&self.payload_size.to_le_bytes())
            .map_err(|e| Error::Backend(format!("Tensor entry write failed: {e}")))?;
        w.write_all(&self.outlier_count.to_le_bytes())
            .map_err(|e| Error::Backend(format!("Tensor entry write failed: {e}")))?;
        w.write_all(&self.outlier_offset.to_le_bytes())
            .map_err(|e| Error::Backend(format!("Tensor entry write failed: {e}")))?;

        Ok(())
    }

    pub fn read<R: Read>(r: &mut R) -> Result<Self> {
        let mut name_len_bytes = [0u8; 2];
        r.read_exact(&mut name_len_bytes)
            .map_err(|e| Error::Backend(format!("Tensor entry read failed: {e}")))?;
        let name_len = u16::from_le_bytes(name_len_bytes) as usize;
        let mut name_bytes = vec![0u8; name_len];
        r.read_exact(&mut name_bytes)
            .map_err(|e| Error::Backend(format!("Tensor entry read failed: {e}")))?;
        let name = String::from_utf8(name_bytes)
            .map_err(|e| Error::Backend(format!("Invalid UTF-8 in tensor name: {e}")))?;

        let mut num_dims_bytes = [0u8; 1];
        r.read_exact(&mut num_dims_bytes)
            .map_err(|e| Error::Backend(format!("Tensor entry read failed: {e}")))?;
        let num_dims = num_dims_bytes[0] as usize;
        let mut shape = Vec::with_capacity(num_dims);
        for _ in 0..num_dims {
            let mut dim_bytes = [0u8; 4];
            r.read_exact(&mut dim_bytes)
                .map_err(|e| Error::Backend(format!("Tensor entry read failed: {e}")))?;
            shape.push(u32::from_le_bytes(dim_bytes) as usize);
        }

        let mut base_bw_bytes = [0u8; 1];
        r.read_exact(&mut base_bw_bytes)
            .map_err(|e| Error::Backend(format!("Tensor entry read failed: {e}")))?;
        let base_bitwidth = base_bw_bytes[0];

        let mut offset_bytes = [0u8; 8];
        r.read_exact(&mut offset_bytes)
            .map_err(|e| Error::Backend(format!("Tensor entry read failed: {e}")))?;
        let payload_offset = u64::from_le_bytes(offset_bytes);

        let mut size_bytes = [0u8; 8];
        r.read_exact(&mut size_bytes)
            .map_err(|e| Error::Backend(format!("Tensor entry read failed: {e}")))?;
        let payload_size = u64::from_le_bytes(size_bytes);

        let mut o_count_bytes = [0u8; 4];
        r.read_exact(&mut o_count_bytes)
            .map_err(|e| Error::Backend(format!("Tensor entry read failed: {e}")))?;
        let outlier_count = u32::from_le_bytes(o_count_bytes);

        let mut o_offset_bytes = [0u8; 8];
        r.read_exact(&mut o_offset_bytes)
            .map_err(|e| Error::Backend(format!("Tensor entry read failed: {e}")))?;
        let outlier_offset = u64::from_le_bytes(o_offset_bytes);

        Ok(Self {
            name,
            shape,
            base_bitwidth,
            payload_offset,
            payload_size,
            outlier_count,
            outlier_offset,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_serialization() {
        let header = GrimV2Header::new(42, 1024);
        let mut buf = Vec::new();
        header.write(&mut buf).unwrap();

        let mut reader = &buf[..];
        let decoded = GrimV2Header::read(&mut reader).unwrap();
        assert_eq!(header, decoded);
    }

    #[test]
    fn test_tensor_entry_serialization() {
        let entry = GrimV2TensorEntry {
            name: "model.layers.0.self_attn.q_proj.weight".to_string(),
            shape: vec![4096, 4096],
            base_bitwidth: 3,
            payload_offset: 2048,
            payload_size: 1572864,
            outlier_count: 512,
            outlier_offset: 1574912,
        };
        let mut buf = Vec::new();
        entry.write(&mut buf).unwrap();

        let mut reader = &buf[..];
        let decoded = GrimV2TensorEntry::read(&mut reader).unwrap();
        assert_eq!(entry, decoded);
    }
}
