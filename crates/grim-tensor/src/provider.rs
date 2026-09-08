//! `TensorProvider` — abstraction over checkpoint sources like GGUF and Safetensors.
//! Traversed depth-first by `grim_nn::WeightSource`.

use crate::dtype::{DType, QuantProvenance, Storage};
use crate::error::{Error, Result};

/// Resolved-at-load dtype and provenance metadata for a checkpoint tensor.
/// Populated from container keys or defaults.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub dtype: DType,
    pub provenance: QuantProvenance,
    pub shape: Vec<usize>,
    /// Kernel fusion dispatch hints (bit0 = RmsNormMatMul, bit1 = QkvAttention).
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

/// Raw byte source for a single tensor, converted to native backend layout upon materialization.
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

    /// Enumerate all tensor names for ahead-of-time parallel prefetching.
    /// Returns an empty vector by default.
    fn tensor_names(&self) -> Vec<String> {
        Vec::new()
    }

    /// Fetch the rank-th shard of a tensor, splitting along `dim` (0=rows, 1=cols).
    /// Default implementation uses [`shard_raw_tensor`] for unquantized weights.
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

/// Validates that `out_dim` divides evenly by `world_size` without splitting quantization blocks.
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

/// CPU-side fallback for `get_packed_sharded` supporting contiguous rows or strided columns.
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
            "quantized shard requires provider override (GgufProvider::get_packed_sharded)".into(),
        ));
    } else {
        return Err(Error::Unimplemented(
            "quantized byte layout cannot be sliced by default — use get_packed_sharded override"
                .into(),
        ));
    };

    if rank >= world_size {
        return Err(Error::IndexOutOfBounds(format!(
            "rank {rank} >= world_size {world_size}"
        )));
    }

    let shape = &raw.shape;
    let ndim = shape.len();
    if ndim != 2 {
        return Err(Error::Shape(format!(
            "shard_raw_tensor: tensor must be 2D (got {}D) — byte offsets below assume [rows, cols]",
            ndim
        )));
    }

    let (rows, cols) = (shape[0], shape[1]);

    // Reject non-divisible dimensions to prevent silent weight truncation.
    if rows % world_size != 0 || cols % world_size != 0 {
        return Err(Error::Shape(format!(
            "shard_raw_tensor: dims {rows}x{cols} not divisible by world_size {world_size} \
             — sharding would silently drop rows/cols"
        )));
    }

    if dim == 0 {
        // Column-parallel: contiguous row slice.
        let shard_rows = rows / world_size;
        let start_row = rank * shard_rows;
        let _end_row = start_row + shard_rows;
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
        assert_eq!(shard.bytes.len(), 4 * 4);
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

    /// L1 regression: non-divisible dims must ERROR — the old code
    /// truncated and silently dropped the tail rows/cols from every rank.
    #[test]
    fn sharded_non_divisible_dims_error() {
        let raw = RawTensor {
            bytes: vec![0u8; 5 * 2 * 4],
            shape: vec![5, 2],
            dtype: DType::F32,
            provenance: QuantProvenance::GrimNative,
        };
        let err = shard_raw_tensor(raw.clone(), 0, 0, 2).expect_err("5 rows / 2 ranks must error");
        assert!(err.to_string().contains("not divisible"), "{err}");

        let raw = RawTensor {
            bytes: vec![0u8; 4 * 3 * 4],
            shape: vec![4, 3],
            dtype: DType::F32,
            provenance: QuantProvenance::GrimNative,
        };
        let err = shard_raw_tensor(raw, 1, 0, 2).expect_err("3 cols / 2 ranks must error");
        assert!(err.to_string().contains("not divisible"), "{err}");
    }

    /// L2 regression: rank > 2 must ERROR — the old code sliced with 2D
    /// offsets over a 3D buffer and returned wrong bytes.
    #[test]
    fn sharded_rank3_tensor_errors() {
        let raw = RawTensor {
            bytes: vec![0u8; 2 * 2 * 2 * 4],
            shape: vec![2, 2, 2],
            dtype: DType::F32,
            provenance: QuantProvenance::GrimNative,
        };
        let err = shard_raw_tensor(raw, 0, 0, 2).expect_err("3D tensor must error");
        assert!(err.to_string().contains("must be 2D"), "{err}");
    }
}
