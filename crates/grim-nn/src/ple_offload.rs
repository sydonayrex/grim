//! SSD-tiered PLE n-gram embedding table (`OffloadedPleTable`).
//!
//! Reads individual rows from a file-backed GGUF tensor (such as the 28 GB
//! `per_layer_token_embd.weight` Q4_0 table) on demand via
//! `SplitGgufProvider::get_range` with a bounded DRAM row cache (~2.6 MB at
//! 4096 rows x 160 f32), preventing host OOM on 62 GB workstations.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use grim_backend_cpu::CpuDevice;
use grim_format::tprov::SplitGgufProvider;
use grim_tensor::MemoryOps;
use grim_tensor::dtype::{DType, QuantProvenance};
use grim_tensor::error::{Error, Result};
use grim_tensor::provider::TensorProvider;
use grim_tensor::{Device, Shape, Tensor};

const DEFAULT_ROW_CACHE_CAP: usize = 4096;

/// On-demand SSD-backed PLE embedding table with a bounded DRAM row cache.
pub struct OffloadedPleTable {
    provider: SplitGgufProvider,
    tensor_name: String,
    num_rows: usize,
    ngram_dim: usize,
    row_bytes: usize,
    dtype: DType,
    provenance: QuantProvenance,
    cache: Mutex<HashMap<usize, Vec<f32>>>,
    cache_cap: usize,
}

impl OffloadedPleTable {
    /// Opens the GGUF metadata for `tensor_name` without loading tensor data into RAM.
    pub fn open(path: &str, tensor_name: &str, ngram_dim: usize) -> Result<Self> {
        let provider = SplitGgufProvider::open(path)?;
        let meta = provider.meta(tensor_name)?;
        if meta.shape.len() != 2 {
            return Err(Error::Shape(format!(
                "OffloadedPleTable '{tensor_name}': expected 2D shape, got {:?}",
                meta.shape
            )));
        }
        let (num_rows, cols) = if meta.shape[1] == ngram_dim {
            (meta.shape[0], meta.shape[1])
        } else if meta.shape[0] == ngram_dim {
            (meta.shape[1], meta.shape[0])
        } else {
            return Err(Error::Shape(format!(
                "OffloadedPleTable '{tensor_name}': shape {:?} does not match ngram_dim={ngram_dim}",
                meta.shape
            )));
        };
        let row_bytes = meta.dtype.expected_bytes(cols);
        Ok(Self {
            provider,
            tensor_name: tensor_name.to_string(),
            num_rows,
            ngram_dim: cols,
            row_bytes,
            dtype: meta.dtype,
            provenance: meta.provenance,
            cache: Mutex::new(HashMap::new()),
            cache_cap: DEFAULT_ROW_CACHE_CAP,
        })
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn ngram_dim(&self) -> usize {
        self.ngram_dim
    }

    /// Reads and dequantizes a single row `[ngram_dim]` as `f32`, serving from the
    /// bounded DRAM cache on hit or `SplitGgufProvider::get_range` on miss.
    pub fn get_row_f32(&self, row: usize) -> Result<Vec<f32>> {
        if row >= self.num_rows {
            return Ok(vec![0.0f32; self.ngram_dim]);
        }
        {
            let cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(hit) = cache.get(&row) {
                return Ok(hit.clone());
            }
        }
        let offset = (row as u64).saturating_mul(self.row_bytes as u64);
        let bytes = self
            .provider
            .get_range(&self.tensor_name, offset, self.row_bytes as u64)?;
        let dev = CpuDevice::new();
        let shape = Shape::new(vec![1, self.ngram_dim]);
        let storage = dev.from_cpu_bytes(&bytes[..], &shape, self.dtype.clone())?;
        let t = Tensor::new(
            Arc::from(storage),
            shape,
            self.dtype.clone(),
            self.provenance.clone(),
            Device::Cpu,
        );
        let vals = t.to_vec_f32()?;
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        if cache.len() >= self.cache_cap {
            cache.clear();
        }
        cache.insert(row, vals.clone());
        Ok(vals)
    }
}
