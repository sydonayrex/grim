//! Expert weights kept as byte ranges in the mmap'd GGUF shards (SSD-backed,
//! OS-evictable) and staged to the compute device only for the experts the
//! router selects. Prevents materializing the full expert bank in RAM or VRAM.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use grim_backend_cpu::CpuDevice;
use grim_format::tprov::SplitGgufProvider;
use grim_tensor::dtype::{DType, Device, QuantProvenance};
use grim_tensor::error::{Error, Result};
use grim_tensor::provider::TensorProvider;
use grim_tensor::shape::Shape;
use grim_tensor::{MemoryOps, Tensor};

use crate::Linear;

/// Lazy per-expert source: stores only metadata + a re-opened provider handle,
/// so expert bytes live in the mmap (page cache) and are fetched per activation.
pub struct ExpertOffloadBanks {
    provider: SplitGgufProvider,
    gate: (String, [usize; 2], usize),
    up: (String, [usize; 2], usize),
    down: (String, [usize; 2], usize),
    dtype: DType,
    provenance: QuantProvenance,
    cache: Mutex<HashMap<usize, (Linear, Linear, Linear)>>,
}

impl ExpertOffloadBanks {
    pub fn open(
        path: &str,
        gate_name: &str,
        up_name: &str,
        down_name: &str,
        num_experts: usize,
    ) -> Result<Self> {
        let provider = SplitGgufProvider::open(path)?;
        let mut props = Vec::with_capacity(3);
        for name in [gate_name, up_name, down_name] {
            let meta = provider.meta(name)?;
            if meta.shape.len() != 3 || meta.shape[0] != num_experts {
                return Err(Error::Shape(format!(
                    "expert bank '{name}': expected [num_experts, out, in], got {:?}",
                    meta.shape
                )));
            }
            let out = meta.shape[1];
            let in_ = meta.shape[2];
            let elems = num_experts * out * in_;
            let total = meta.dtype.expected_bytes(elems);
            if total % num_experts != 0 {
                return Err(Error::Backend(format!(
                    "expert bank '{name}': {total} bytes not divisible by {num_experts}"
                )));
            }
            props.push((name.to_string(), [out, in_], total / num_experts));
        }
        let meta = provider.meta(gate_name)?;
        Ok(Self {
            provider,
            gate: props[0].clone(),
            up: props[1].clone(),
            down: props[2].clone(),
            dtype: meta.dtype,
            provenance: meta.provenance,
            cache: Mutex::new(HashMap::new()),
        })
    }

    fn staged_bank_tensor(
        &self,
        name: &str,
        shape: [usize; 2],
        stride: usize,
        idx: usize,
    ) -> Result<Tensor> {
        let bytes = self
            .provider
            .get_range(name, (idx * stride) as u64, stride as u64)?;
        let dev = CpuDevice::new();
        let storage = dev.from_cpu_bytes(&bytes[..], &Shape::new(vec![shape[0], shape[1]]), self.dtype.clone())?;
        Ok(Tensor::new(
            Arc::from(storage),
            Shape::new(vec![shape[0], shape[1]]),
            self.dtype.clone(),
            self.provenance.clone(),
            Device::Cpu,
        ))
    }

    /// Fetch expert `idx`'s gate/up/down as staged device Linears.
    pub fn get(&self, idx: usize, target: &Device) -> Result<(Linear, Linear, Linear)> {
        {
            let cache = self.cache.lock().unwrap();
            if let Some(hit) = cache.get(&idx) {
                if hit.0.weight.device() == target {
                    return Ok(hit.clone());
                }
            }
        }
        let triple = (
            self.staged_bank_tensor(&self.gate.0, self.gate.1, self.gate.2, idx)?,
            self.staged_bank_tensor(&self.up.0, self.up.1, self.up.2, idx)?,
            self.staged_bank_tensor(&self.down.0, self.down.1, self.down.2, idx)?,
        );
        let lins = (
            Linear::from_tensor(triple.0, None).staged_to(target)?,
            Linear::from_tensor(triple.1, None).staged_to(target)?,
            Linear::from_tensor(triple.2, None).staged_to(target)?,
        );
        let mut cache = self.cache.lock().unwrap();
        if cache.len() >= 8 {
            cache.clear();
        }
        cache.insert(idx, lins.clone());
        Ok(lins)
    }
}
