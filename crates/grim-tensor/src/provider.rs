//! `TensorProvider` — abstraction over a checkpoint source. Both GGUF and
//! safetensors-backed readers implement this; `WeightSource` (in
//! `grim-nn`) walks it depth-first by prefix.

use crate::dtype::{DType, QuantProvenance, Storage};
use crate::error::{Error, Result};

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
    /// Look up a tensor and return it in a packed, low-bit representation if supported
    /// by the provider, bypassing eager CPU dequantization.
    fn get_packed(&self, name: &str) -> Result<RawTensor> {
        self.get(name)
    }
    /// Optional hint — metadata the loader wants to expose without
    /// materializing the full tensor (shape, dtype, provenance).
    fn meta(&self, name: &str) -> Result<TensorMeta>;

    /// Fetch the rank-th shard of a tensor, splitting along `dim`.
    ///
    /// `dim == 0` shards rows (column-parallel): each rank owns a contiguous
    /// block of output rows. `dim == 1` shards columns (row-parallel): each
    /// rank owns every rank-th stride of each row.
    ///
    /// The default implementation calls [`shard_raw_tensor`], which only works
    /// for native (F32/F16) storage — quantized layouts require a provider
    /// override (see `GgufProvider::get_packed_sharded`).
    fn get_packed_sharded(
        &self,
        name: &str,
        dim: usize,
        rank: usize,
        world_size: usize,
    ) -> Result<RawTensor> {
        let raw = self.get_packed(name)?;
        shard_raw_tensor(raw, dim, rank, world_size)
    }
}

/// Validate that `out_dim` is evenly divisible by `world_size` for the given
/// block size (used by GGUF block-quant sharding). Returns `Err` if the shard
/// boundary would split a quantization block.
pub fn shard_boundary_valid(out_dim: usize, world_size: usize, block_size: usize) -> bool {
    if world_size == 0 {
        return false;
    }
    if out_dim % world_size != 0 {
        return false;
    }
    let shard_size = out_dim / world_size;
    shard_size % block_size == 0
}

/// CPU-side sharding fallback for `get_packed_sharded`. Handles dim==0 (contiguous
/// row slice) and dim==1 (per-row strided copy); errors on non-native storage or
/// quantized provenance that requires a provider-specific byte-range override.
pub fn shard_raw_tensor(
    raw: RawTensor,
    dim: usize,
    rank: usize,
    world_size: usize,
) -> Result<RawTensor> {
    if world_size == 0 {
        return Err(Error::Shape(format!(
            "shard_raw_tensor: world_size must be > 0 (got {world_size})"
        )));
    }
    if dim != 0 && dim != 1 {
        return Err(Error::Shape(format!(
            "shard_raw_tensor: dim must be 0 or 1 (got {dim})"
        )));
    }
    let elem_size = if raw.dtype.storage == Storage::Native {
        raw.dtype.arith.byte_size()
    } else if matches!(raw.provenance, QuantProvenance::ExternalQat { .. }) {
        return Err(Error::Unimplemented(
            "quantized shard requires provider override (GgufProvider::get_packed_sharded)"
                .into(),
        ));
    } else {
        return Err(Error::Unimplemented(
            "quantized byte layout cannot be sliced by default — use get_packed_sharded override"
                .into(),
        ));
    };

    let rank = rank as usize;
    if rank >= world_size {
        return Err(Error::IndexOutOfBounds(format!(
            "rank {rank} >= world_size {world_size}"
        )));
    }

    let shape = &raw.shape;
    let ndim = shape.len();
    if ndim < 2 {
        return Err(Error::Shape(format!(
            "shard_raw_tensor: tensor must be 2D (got {}D)",
            ndim
        )));
    }

    let (rows, cols) = (shape[0], shape[1]);

    if dim == 0 {
        // Column-parallel: contiguous row slice.
        let shard_rows = rows / world_size;
        let start_row = rank * shard_rows;
        let end_row = start_row + shard_rows;
        let row_stride = cols * elem_size;
        let start_byte = start_row * row_stride;
        let shard_bytes = &raw.bytes[start_byte..start_byte + shard_rows * row_stride];

        Ok(RawTensor {
            bytes: shard_bytes.to_vec(),
            shape: vec![shard_rows, cols],
            dtype: raw.dtype.clone(),
            provenance: raw.provenance,
        })
    } else {
        // Row-parallel: per-row strided copy.
        let shard_cols = cols / world_size;
        let start_col = rank * shard_cols;
        let mut out = Vec::with_capacity(rows * shard_cols * elem_size);
        for row in 0..rows {
            let row_start = row * cols * elem_size;
            let col_start = row_start + start_col * elem_size;
            let col_end = col_start + shard_cols * elem_size;
            out.extend_from_slice(&raw.bytes[col_start..col_end]);
        }

        Ok(RawTensor {
            bytes: out,
            shape: vec![rows, shard_cols],
            dtype: raw.dtype.clone(),
            provenance: raw.provenance,
        })
    }
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

    /// shard_raw_tensor: dim=0 contiguous row slice round-trips shape.
    #[test]
    fn sharded_dim0_roundtrips() {
        // 4×2 F32 tensor, rank 1 of 2 → expects 2×2 shard.
        let raw = RawTensor {
            bytes: vec![0u8; 4 * 2 * 4], // 4 rows, 2 cols, f32
            shape: vec![4, 2],
            dtype: DType::F32,
            provenance: QuantProvenance::GrimNative,
        };
        let shard = shard_raw_tensor(raw, 0, 1, 2).expect("dim0 shard ok");
        assert_eq!(shard.shape, vec![2, 2]);
        assert_eq!(shard.bytes.len(), 2 * 2 * 4);
    }

    /// shard_raw_tensor: dim=1 strided copy round-trips shape.
    #[test]
    fn sharded_dim1_roundtrips() {
        let raw = RawTensor {
            bytes: vec![0u8; 4 * 2 * 4],
            shape: vec![4, 2],
            dtype: DType::F32,
            provenance: QuantProvenance::GrimNative,
        };
        let shard = shard_raw_tensor(raw, 1, 0, 2).expect("dim1 shard ok");
        assert_eq!(shard.shape, vec![4, 1]);
        assert_eq!(shard.bytes.len(), 4 * 1 * 4);
    }

    /// shard_boundary_valid: divisibility + block alignment checks.
    #[test]
    fn shard_boundary_valid_checks() {
        // 256/8 = 32 per shard, block_size 16 divides 32 → valid.
        assert!(shard_boundary_valid(256, 8, 16));
        // 255/8 = 31.875 → not divisible → invalid.
        assert!(!shard_boundary_valid(255, 8, 16));
        // 256/7 → not divisible → invalid.
        assert!(!shard_boundary_valid(256, 7, 16));
        // 256/4=64 per shard, 32 divides 64 → valid.
        assert!(shard_boundary_valid(256, 4, 32));
        // world_size 0 → invalid.
        assert!(!shard_boundary_valid(256, 0, 16));
    }
}
