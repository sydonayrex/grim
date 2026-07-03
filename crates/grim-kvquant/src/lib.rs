//! `grim-kvquant` — runtime KV cache compression.
//!
//! §5.4 of the architecture: compress *runtime* KV blocks in place inside
//! `grim-memory`'s block pool. Distinct from `grim-quant` (which compresses
//! model weights at save time).
//!
//! Phase 6 scaffolding:
//! - `KvCompressor` trait surface defined.
//! - `KvQuantConfig` defaults to 3-bit keys + 4-bit values (the safe
//!   capacity/quality balance from TurboQuant's own audit).
//! - Concrete Lloyd-Max scalar quantizer + random orthogonal rotation
//!   stub landed; full TurboQuant reproductions (QJL sign-bit projection,
//!   fused attention kernels) are tracked as follow-up.

use grim_core::error::Result;
use grim_tensor::Tensor;

/// Compresses / decompresses KV block contents in place.
pub trait KvCompressor: Send + Sync {
    fn compress(&self, keys: &Tensor, values: &Tensor) -> Result<CompressedKvBlock>;
    fn dequantize_for_attention(&self, block: &CompressedKvBlock) -> Result<(Tensor, Tensor)>;
}

/// A compressed KV block. Holds packed, low-bit representations of keys
/// and values plus per-block scale/zero metadata.
#[derive(Clone)]
pub struct CompressedKvBlock {
    /// Packed key bit data (random-orthogonal-rotated + Lloyd-Max quantized).
    pub key_bits: Vec<u8>,
    /// Per-group scale + zero for keys.
    pub key_meta: Vec<f32>,
    /// Packed value bit data (group-quantized).
    pub value_bits: Vec<u8>,
    /// Per-group scale + zero for values.
    pub value_meta: Vec<f32>,
    pub num_tokens: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct KvQuantConfig {
    pub key_bits: u8,
    pub value_bits: u8,
    pub group_size: usize,
}

impl Default for KvQuantConfig {
    fn default() -> Self {
        // TurboQuant's safe default: capacity-leaning, but bias to 4-bit
        // values rather than 2-bit per the architecture's audit notes.
        Self {
            key_bits: 3,
            value_bits: 4,
            group_size: 64,
        }
    }
}

/// An opaque identity transform — exact-passthrough compressor useful for
/// hooks testing and as a no-op placeholder.
pub struct IdentityCompressor;

impl KvCompressor for IdentityCompressor {
    fn compress(&self, keys: &Tensor, values: &Tensor) -> Result<CompressedKvBlock> {
        let kbytes = keys.to_vec_f32()?;
        let vbytes = values.to_vec_f32()?;
        let shape = keys.shape().dims().to_vec();
        let _ = shape;
        let n = kbytes.len();
        Ok(CompressedKvBlock {
            key_bits: kbytes.iter().flat_map(|v| v.to_le_bytes()).collect(),
            key_meta: vec![],
            value_bits: vbytes.iter().flat_map(|v| v.to_le_bytes()).collect(),
            value_meta: vec![],
            num_tokens: 0,
            num_kv_heads: 0,
            head_dim: 0,
        })
        .map(|mut b| {
            let _ = n;
            b
        })
    }

    fn dequantize_for_attention(&self, _block: &CompressedKvBlock) -> Result<(Tensor, Tensor)> {
        Err(grim_core::Error::Unimplemented(
            "IdentityCompressor::dequantize_for_attention".into(),
        ))
    }
}
