//! A unified compressed tensor container format compatible with all
//! production-ready compressed tensor types in grim.
//!
//! This module defines a single container (.gcct — Grim Compact Tensor)
//! that can hold `W8A8Mxfp8`, `WNA16`, and `EmbeddingWNA16Int` tensors
//! in one file, with type-tagged per-tensor metadata and data sections.
//!
//! The container is a binary format with:
//! - A 4-byte magic + u32 version + u32 tensor_count header
//! - Per-tensor: name (u32-prefixed), type tag (u32), metadata (u32-prefixed
//!   length + serde bytes), data (u64-prefixed length + raw bytes)
//!
//! The reader deserializes into `CompressedTensor` values that expose the
//! type tag, parsed metadata, and raw data bytes, so callers can dispatch
//! to the correct dequantizer / kernel path.

use std::fmt;

/// Magic bytes for the GCcT (Grim Compact Tensor) container: `GCT\x01`.
pub const GCCT_MAGIC: &[u8; 4] = b"GCT\x01";
/// Current container format version.
pub const GCCT_VERSION: u32 = 1;

/// Tag identifying which compressed tensor type a section holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompressedTensorType {
    /// W8A8MXFP8: 8-bit MXFP8 activations + weights in some quantized storage.
    W8A8Mxfp8 = 1,
    /// WNA16: weight-only N-bit with 16-bit precision.
    WNA16 = 2,
    /// EmbeddingWNA16Int: embedding weights stored as N-bit integers.
    EmbeddingWNA16Int = 3,
    /// CompressedTensors W8A8 with INT8 weights (SmoothQuant-style, per-channel
    /// int8 codes + per-channel scales).
    CompressedTensorsW8A8Int8 = 4,
    /// CompressedTensors W8A8 with FP8 (OCP E4M3) weights (per-tensor/per-block
    /// fp8 codes + fp8/f32 scale).
    CompressedTensorsW8A8Fp8 = 5,
}

impl CompressedTensorType {
    /// Parse the u32 tag back into a `CompressedTensorType`.
    pub fn from_tag(tag: u32) -> Result<Self, fmt::Error> {
        match tag {
            1 => Ok(Self::W8A8Mxfp8),
            2 => Ok(Self::WNA16),
            3 => Ok(Self::EmbeddingWNA16Int),
            4 => Ok(Self::CompressedTensorsW8A8Int8),
            5 => Ok(Self::CompressedTensorsW8A8Fp8),
            _ => Err(fmt::Error),
        }
    }

    /// The u32 tag written to the `.gcct` container header for this type.
    pub fn to_tag(self) -> u32 {
        match self {
            Self::W8A8Mxfp8 => 1,
            Self::WNA16 => 2,
            Self::EmbeddingWNA16Int => 3,
            Self::CompressedTensorsW8A8Int8 => 4,
            Self::CompressedTensorsW8A8Fp8 => 5,
        }
    }
}
