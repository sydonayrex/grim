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

#[derive(Clone)]
struct BankSpec {
    name: String,
    shape: [usize; 2],
    stride: usize,
    dtype: DType,
    provenance: QuantProvenance,
}

/// Lazy per-expert source: stores only metadata + a re-opened provider handle,
/// so expert bytes live in the mmap (page cache) and are fetched per activation.
pub struct ExpertOffloadBanks {
    provider: SplitGgufProvider,
    gate: BankSpec,
    up: BankSpec,
    down: BankSpec,
    cache: Mutex<HashMap<usize, (Linear, Linear, Linear)>>,
    /// Use-capped residency state (cap 15, MRU tie-break) behind `GRIM_MOE_USE_VICTIM=1`.
    use_stats: Mutex<(Vec<u16>, Vec<u64>, u64)>,
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
            props.push(BankSpec {
                name: name.to_string(),
                shape: [out, in_],
                stride: total / num_experts,
                dtype: meta.dtype,
                provenance: meta.provenance,
            });
        }
        Ok(Self {
            provider,
            gate: props[0].clone(),
            up: props[1].clone(),
            down: props[2].clone(),
            cache: Mutex::new(HashMap::new()),
            use_stats: Mutex::new((vec![0u16; num_experts], vec![0u64; num_experts], 0u64)),
        })
    }

    fn staged_bank_tensor(&self, bank: &BankSpec, idx: usize) -> Result<Tensor> {
        let bytes = self
            .provider
            .get_range(&bank.name, (idx * bank.stride) as u64, bank.stride as u64)?;
        let dev = CpuDevice::new();
        let storage = dev.from_cpu_bytes(
            &bytes[..],
            &Shape::new(vec![bank.shape[0], bank.shape[1]]),
            bank.dtype.clone(),
        )?;
        Ok(Tensor::new(
            Arc::from(storage),
            Shape::new(vec![bank.shape[0], bank.shape[1]]),
            bank.dtype.clone(),
            bank.provenance.clone(),
            Device::Cpu,
        ))
    }

    /// Fetch expert `idx`'s gate/up/down as staged device Linears.
    pub fn get(&self, idx: usize, target: &Device) -> Result<(Linear, Linear, Linear)> {
        let use_victim = std::env::var("GRIM_MOE_USE_VICTIM")
            .ok()
            .map(|v| {
                let v = v.trim();
                v == "1" || v.eq_ignore_ascii_case("true")
            })
            .unwrap_or(false);
        if use_victim {
            if let Ok(mut st) = self.use_stats.lock() {
                st.2 = st.2.wrapping_add(1);
                let seq = st.2;
                if let Some(c) = st.0.get_mut(idx) {
                    *c = c.saturating_add(1).min(15);
                }
                if let Some(t) = st.1.get_mut(idx) {
                    *t = seq;
                }
            }
        }
        {
            let cache = self.cache.lock().unwrap();
            if let Some(hit) = cache.get(&idx) {
                if hit.0.weight.device() == target {
                    return Ok(hit.clone());
                }
            }
        }
        let triple = (
            self.staged_bank_tensor(&self.gate, idx)?,
            self.staged_bank_tensor(&self.up, idx)?,
            self.staged_bank_tensor(&self.down, idx)?,
        );
        let lins = (
            Linear::from_tensor(triple.0, None).staged_to(target)?,
            Linear::from_tensor(triple.1, None).staged_to(target)?,
            Linear::from_tensor(triple.2, None).staged_to(target)?,
        );
        let mut cache = self.cache.lock().unwrap();
        if cache.len() >= 8 {
            if use_victim {
                if let Ok(st) = self.use_stats.lock() {
                    let mut ids: Vec<usize> = cache.keys().copied().collect();
                    ids.sort_unstable_by(|a, b| {
                        st.0.get(*b)
                            .copied()
                            .unwrap_or(0)
                            .cmp(&st.0.get(*a).copied().unwrap_or(0))
                            .then_with(|| {
                                st.1.get(*b)
                                    .copied()
                                    .unwrap_or(0)
                                    .cmp(&st.1.get(*a).copied().unwrap_or(0))
                            })
                    });
                    let keep: std::collections::HashSet<usize> = ids.into_iter().take(7).collect();
                    cache.retain(|k, _| keep.contains(k));
                } else {
                    cache.clear();
                }
            } else {
                cache.clear();
            }
        }
        cache.insert(idx, lins.clone());
        Ok(lins)
    }
}
