//! Training-state sidecar format: `model.grim.train` (WI-R6).
//!
//! The V3 `.grim` wire format is weight/inference-only; it has no slot for
//! optimizer state, LoRA/DoRA adapters, or SERQ low-rank error matrices.
//! Research shows consumer fine-tune is viable (LoRA Edge 26× peak-mem cut
//! on Llama-3.2-3B; DoRA ~24% train-mem reduction vs LoRA; SERQ saliency
//! low-rank error for 4-bit GEMM). This module defines a **companion
//! sidecar** — `model.grim.train` written next to `model.grim` — so the
//! inference reader is never touched and legacy files ignore it.
//!
//! Layout (little-endian):
//!
//! ```text
//! [ magic: 8 bytes "GRIMTRN\x01" ]
//! [ header_len: u32 LE ][ header JSON ]
//! [ per-blob: name_len:u16 | name | ndim:u8 | dims:u32×ndim | nbytes:u64 | bytes ]
//! ```
//!
//! Each blob (adapter A/B, optimizer m/v, error matrix, …) is a self-describing
//! byte region. The header JSON records the `fp_format` numeric descriptor and
//! which named blobs belong to which logical slot, so a resumed fine-tune can
//! reconstruct step-N state bit-for-bit.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, Write};

use serde_json::Value;

use grim_tensor::error::{Error, Result};

/// Magic bytes for the `.grim.train` sidecar.
pub const TRAIN_MAGIC: [u8; 8] = [0x47, 0x52, 0x49, 0x4d, 0x54, 0x52, 0x4e, 0x01]; // "GRIMTRN\x01"

/// FP format descriptor for training-state tensors (WI-R6).
///
/// The numeric set RDNA3/4 training targets (Dual-Precision MAC paper:
/// FP8/FP4 rising in inference, FP16/FP32 still dominate training).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainFpFormat {
    Fp16 = 0,
    Fp32 = 1,
    Fp8E4M3 = 2,
    Fp8E5M2 = 3,
    Fp4 = 4,
}

impl TrainFpFormat {
    pub fn from_u8(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Self::Fp16),
            1 => Some(Self::Fp32),
            2 => Some(Self::Fp8E4M3),
            3 => Some(Self::Fp8E5M2),
            4 => Some(Self::Fp4),
            _ => None,
        }
    }
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

/// One named training-state blob (adapter weight, optimizer moment, error matrix).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainBlob {
    pub name: String,
    pub shape: Vec<usize>,
    /// Raw little-endian bytes of the blob (caller owns the numeric encoding;
    /// typically f32/f16/quantized per `TrainState::fp_format`).
    pub data: Vec<u8>,
}

impl TrainBlob {
    fn write<W: Write>(&self, w: &mut W) -> Result<()> {
        let name_bytes = self.name.as_bytes();
        w.write_all(&(name_bytes.len() as u16).to_le_bytes())
            .map_err(|e| Error::Backend(format!("train blob name write failed: {e}")))?;
        w.write_all(name_bytes)
            .map_err(|e| Error::Backend(format!("train blob name write failed: {e}")))?;
        w.write_all(&(self.shape.len() as u8).to_le_bytes())
            .map_err(|e| Error::Backend(format!("train blob shape write failed: {e}")))?;
        for dim in &self.shape {
            w.write_all(&(*dim as u32).to_le_bytes())
                .map_err(|e| Error::Backend(format!("train blob dim write failed: {e}")))?;
        }
        w.write_all(&(self.data.len() as u64).to_le_bytes())
            .map_err(|e| Error::Backend(format!("train blob len write failed: {e}")))?;
        w.write_all(&self.data)
            .map_err(|e| Error::Backend(format!("train blob data write failed: {e}")))?;
        Ok(())
    }

    fn read<R: Read>(r: &mut R) -> Result<Self> {
        let mut name_len = [0u8; 2];
        r.read_exact(&mut name_len)
            .map_err(|e| Error::Backend(format!("train blob name read failed: {e}")))?;
        let name_len = u16::from_le_bytes(name_len) as usize;
        let mut name_bytes = vec![0u8; name_len];
        r.read_exact(&mut name_bytes)
            .map_err(|e| Error::Backend(format!("train blob name read failed: {e}")))?;
        let name = String::from_utf8(name_bytes)
            .map_err(|e| Error::Backend(format!("invalid UTF-8 in train blob name: {e}")))?;

        let mut ndim_b = [0u8; 1];
        r.read_exact(&mut ndim_b)
            .map_err(|e| Error::Backend(format!("train blob shape read failed: {e}")))?;
        let ndim = ndim_b[0] as usize;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            let mut dim_b = [0u8; 4];
            r.read_exact(&mut dim_b)
                .map_err(|e| Error::Backend(format!("train blob dim read failed: {e}")))?;
            shape.push(u32::from_le_bytes(dim_b) as usize);
        }

        let mut len_b = [0u8; 8];
        r.read_exact(&mut len_b)
            .map_err(|e| Error::Backend(format!("train blob len read failed: {e}")))?;
        let len = u64::from_le_bytes(len_b) as usize;
        let mut data = vec![0u8; len];
        r.read_exact(&mut data)
            .map_err(|e| Error::Backend(format!("train blob data read failed: {e}")))?;

        Ok(Self { name, shape, data })
    }
}

/// A training-state sidecar: adapters, optimizer moments, error matrices.
///
/// Optional companion to a `.grim` inference file. The inference reader never
/// requires it; a resumed fine-tune reproduces step-N state from it.
#[derive(Debug, Clone)]
pub struct TrainState {
    /// Numeric format the training-state tensors are encoded in.
    pub fp_format: TrainFpFormat,
    /// Named training-state blobs keyed by logical slot name
    /// (e.g. `lora_a`, `lora_b`, `opt_m`, `opt_v`, `error_matrix`).
    pub blobs: HashMap<String, TrainBlob>,
}

impl Default for TrainState {
    fn default() -> Self {
        Self {
            fp_format: TrainFpFormat::Fp32,
            blobs: HashMap::new(),
        }
    }
}

impl TrainState {
    /// Insert a blob under `name`.
    pub fn add_blob(&mut self, name: impl Into<String>, shape: Vec<usize>, data: Vec<u8>) {
        let name = name.into();
        self.blobs.insert(
            name.clone(),
            TrainBlob {
                name,
                shape,
                data,
            },
        );
    }

    /// Write the sidecar to `path` (conventionally `model.grim.train`).
    pub fn write<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let mut buf = Vec::new();
        buf.write_all(&TRAIN_MAGIC)
            .map_err(|e| Error::Backend(format!("train magic write failed: {e}")))?;

        let header = serde_json::json!({
            "fp_format": self.fp_format.as_u8(),
            "blobs": self.blobs.keys().collect::<Vec<_>>(),
        });
        let header_bytes = serde_json::to_vec(&header)
            .map_err(|e| Error::Backend(format!("train header serialize failed: {e}")))?;
        buf.write_all(&(header_bytes.len() as u32).to_le_bytes())
            .map_err(|e| Error::Backend(format!("train header len write failed: {e}")))?;
        buf.write_all(&header_bytes)
            .map_err(|e| Error::Backend(format!("train header write failed: {e}")))?;

        for blob in self.blobs.values() {
            blob.write(&mut buf)?;
        }

        std::fs::write(path, &buf).map_err(|e| Error::Backend(format!("train write failed: {e}")))
    }

    /// Read a sidecar from `path`. Returns `None` (not an error) when the
    /// file is absent, so inference readers can ignore a missing sidecar.
    pub fn read<P: AsRef<std::path::Path>>(path: P) -> Result<Option<Self>> {
        let path = path.as_ref();
        let file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(_) => return Ok(None),
        };
        let mut reader = std::io::BufReader::new(file);

        let mut magic = [0u8; 8];
        reader
            .read_exact(&mut magic)
            .map_err(|e| Error::Backend(format!("train magic read failed: {e}")))?;
        if magic != TRAIN_MAGIC {
            return Err(Error::Backend(format!(
                "Invalid train sidecar magic: expected {:?}, got {:?}",
                TRAIN_MAGIC, magic
            )));
        }

        let mut header_len_b = [0u8; 4];
        reader
            .read_exact(&mut header_len_b)
            .map_err(|e| Error::Backend(format!("train header len read failed: {e}")))?;
        let header_len = u32::from_le_bytes(header_len_b) as usize;
        let mut header_bytes = vec![0u8; header_len];
        reader
            .read_exact(&mut header_bytes)
            .map_err(|e| Error::Backend(format!("train header read failed: {e}")))?;
        let header: Value = serde_json::from_slice(&header_bytes)
            .map_err(|e| Error::Backend(format!("train header JSON invalid: {e}")))?;
        let fp_format = header
            .get("fp_format")
            .and_then(|v| v.as_u64())
            .and_then(|v| TrainFpFormat::from_u8(v as u8))
            .unwrap_or(TrainFpFormat::Fp32);

        let mut blobs = HashMap::new();
        // The blob stream ends at EOF; read until exhausted.
        loop {
            // Peek: a short read means EOF — stop cleanly.
            let mut peek = [0u8; 2];
            let pos = reader
                .stream_position()
                .map_err(|e| Error::Backend(e.to_string()))?;
            let n = reader
                .read(&mut peek)
                .map_err(|e| Error::Backend(format!("train blob peek failed: {e}")))?;
            if n == 0 {
                break; // clean EOF
            }
            // Rewind the 2 peek bytes and read the full blob.
            reader
                .seek(SeekFrom::Start(pos))
                .map_err(|e| Error::Backend(e.to_string()))?;
            let blob = TrainBlob::read(&mut reader)?;
            blobs.insert(blob.name.clone(), blob);
        }

        Ok(Some(Self { fp_format, blobs }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_state_round_trips_byte_identical() {
        let mut state = TrainState {
            fp_format: TrainFpFormat::Fp8E4M3,
            blobs: HashMap::new(),
        };
        state.add_blob("lora_a", vec![64, 128], (0u8..=127).collect());
        state.add_blob("lora_b", vec![128, 64], (128u8..=255).collect());
        state.add_blob("opt_m", vec![4096], vec![7u8; 4096]);
        state.add_blob("error_matrix", vec![32, 32], (0u8..32).cycle().take(1024).collect());

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.grim.train");
        state.write(&path).unwrap();

        let restored = TrainState::read(&path).unwrap().expect("should read");
        assert_eq!(restored.fp_format, TrainFpFormat::Fp8E4M3);
        assert_eq!(restored.blobs.len(), 4);
        assert_eq!(restored.blobs["lora_a"].data, (0u8..=127).collect::<Vec<_>>());
        assert_eq!(restored.blobs["lora_b"].data, (128u8..=255).collect::<Vec<_>>());
        assert_eq!(restored.blobs["opt_m"].data, vec![7u8; 4096]);
        assert_eq!(restored.blobs["error_matrix"].shape, vec![32, 32]);
    }

    #[test]
    fn train_state_read_missing_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("absent.grim.train");
        let res = TrainState::read(&path).unwrap();
        assert!(res.is_none());
    }

    #[test]
    fn train_fp_format_round_trips() {
        for fmt in [
            TrainFpFormat::Fp16,
            TrainFpFormat::Fp32,
            TrainFpFormat::Fp8E4M3,
            TrainFpFormat::Fp8E5M2,
            TrainFpFormat::Fp4,
        ] {
            assert_eq!(TrainFpFormat::from_u8(fmt.as_u8()), Some(fmt));
        }
        assert_eq!(TrainFpFormat::from_u8(99), None);
    }
}
