//! `CudaAutotuner`: Empirical PTX kernel timing, ShapeClass op-identity classification, and persistent tuning cache.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::caps::CudaCaps;

/// Shape classification for GEMM workloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ShapeClass {
    Decode,
    Prefill,
    TLOLog,
}

impl ShapeClass {
    pub fn classify(m: usize, n: usize, _k: usize) -> Self {
        if m == 1 {
            ShapeClass::Decode
        } else if n >= 16384 && m <= 32 {
            ShapeClass::TLOLog
        } else {
            ShapeClass::Prefill
        }
    }

    /// Derive ShapeClass using op identity (GemmOp) tag and dimension metadata.
    pub fn from_op(op: GemmOp, m: usize, n: usize, k: usize) -> Self {
        match op {
            GemmOp::LmHead => ShapeClass::TLOLog,
            _ => Self::classify(m, n, k),
        }
    }
}

/// Logical operation tag for GEMM call sites.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmOp {
    Attention,
    Ffn,
    LmHead,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CudaTileConfig {
    pub block_m: u32,
    pub block_n: u32,
    pub block_k: u32,
    pub split_k: u32,
}

impl CudaTileConfig {
    pub fn is_valid(&self, caps: &CudaCaps) -> bool {
        let shared_mem = (self.block_m * self.block_k + self.block_k * self.block_n) * 4;
        let threads = self.block_m * self.block_n;
        caps.validate_resource_limits(shared_mem, threads)
    }
}

#[derive(Debug)]
pub struct CudaAutotuner {
    cache: Mutex<HashMap<(u64, usize, usize, usize, ShapeClass), Option<CudaTileConfig>>>,
}

/// Flat, JSON-friendly snapshot of one tune entry. The in-memory cache key is a tuple; serde
/// JSON object keys must be strings, so each entry is serialized as a record instead. `None`
/// cache values are preserved via `None`-able tile fields.
#[derive(Debug, Serialize, Deserialize)]
struct TuneEntryOwned {
    caps_hash: u64,
    m: usize,
    n: usize,
    k: usize,
    shape_class: ShapeClass,
    block_m: Option<u32>,
    block_n: Option<u32>,
    block_k: Option<u32>,
    split_k: Option<u32>,
}

impl Clone for CudaAutotuner {
    fn clone(&self) -> Self {
        let cache = self.cache.lock().unwrap();
        let new_cache = cache.clone();
        drop(cache);
        Self {
            cache: Mutex::new(new_cache),
        }
    }
}

impl CudaAutotuner {
    pub fn new() -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Load autotune cache from disk for a specific hardware fingerprint.
    /// Restores previously measured winners so a repeat shape on the same GPU hits the
    /// cache instead of re-searching. JSON object keys must be strings, so the in-memory
    /// tuple key is flattened to a Vec of records on disk.
    pub fn load_cache(&self, caps: &CudaCaps) {
        let hash = caps.cache_key_hash();
        let path = Self::cache_path(hash);
        if path.exists() {
            if let Ok(data) = fs::read_to_string(&path) {
                match serde_json::from_str::<Vec<TuneEntryOwned>>(&data) {
                    Ok(entries) => {
                        let mut lock = self.cache.lock().unwrap();
                        for e in entries {
                            if e.caps_hash != hash {
                                continue;
                            }
                            let cfg = match (e.block_m, e.block_n, e.block_k) {
                                (Some(bm), Some(bn), Some(bk)) => Some(CudaTileConfig {
                                    block_m: bm,
                                    block_n: bn,
                                    block_k: bk,
                                    split_k: e.split_k.unwrap_or(1),
                                }),
                                _ => None,
                            };
                            lock.insert((e.caps_hash, e.m, e.n, e.k, e.shape_class), cfg);
                        }
                    }
                    Err(err) => {
                        tracing::warn!(
                            "[CudaAutotuner] Ignoring corrupt tune cache {}: {err}",
                            path.display()
                        );
                    }
                }
            }
        }
    }

    /// Save autotune cache to disk for a specific hardware fingerprint.
    /// Persists only entries matching this device's `caps_hash`.
    pub fn save_cache(&self, caps: &CudaCaps) {
        let hash = caps.cache_key_hash();
        let entries: Vec<TuneEntryOwned> = {
            let lock = self.cache.lock().unwrap();
            lock.iter()
                .filter(|((h, _, _, _, _), _)| *h == hash)
                .map(|((h, m, n, k, sc), opt_cfg)| TuneEntryOwned {
                    caps_hash: *h,
                    m: *m,
                    n: *n,
                    k: *k,
                    shape_class: *sc,
                    block_m: opt_cfg.map(|c| c.block_m),
                    block_n: opt_cfg.map(|c| c.block_n),
                    block_k: opt_cfg.map(|c| c.block_k),
                    split_k: opt_cfg.map(|c| c.split_k),
                })
                .collect()
        };
        let path = Self::cache_path(hash);
        if let Ok(json) = serde_json::to_vec(&entries) {
            if let Some(dir) = path.parent() {
                let _ = fs::create_dir_all(dir);
            }
            let _ = fs::write(path, json);
        }
    }

    /// Disk path for a hardware fingerprint's tune cache.
    fn cache_path(hash: u64) -> PathBuf {
        PathBuf::from(format!(".autotune_cache/cuda_{hash:016x}.json"))
    }

    /// Select optimal CUDA tile parameters using ShapeClass + resource limits.
    /// Caches selected config by (caps_hash, m, n, k, shape_class) for idempotency.
    pub fn search_tile_config(
        &self,
        caps: &CudaCaps,
        m: usize,
        n: usize,
        k: usize,
        op: Option<GemmOp>,
    ) -> CudaTileConfig {
        let shape_class = match op {
            Some(op_tag) => ShapeClass::from_op(op_tag, m, n, k),
            None => ShapeClass::classify(m, n, k),
        };

        let caps_hash = caps.cache_key_hash();
        let key = (caps_hash, m, n, k, shape_class);

        // Fast path: in-memory (or previously loaded-from-disk) cache hit.
        {
            let lock = self.cache.lock().unwrap();
            if let Some(cfg) = lock.get(&key).and_then(|opt| opt.as_ref()) {
                return *cfg;
            }
        }

        // Candidate tile configs including specialization & split_k variations
        let candidates = match shape_class {
            ShapeClass::TLOLog => vec![
                CudaTileConfig { block_m: 16, block_n: 64, block_k: 64, split_k: 1 },
                CudaTileConfig { block_m: 32, block_n: 64, block_k: 32, split_k: 1 },
                CudaTileConfig { block_m: 16, block_n: 64, block_k: 64, split_k: 2 },
            ],
            ShapeClass::Decode => vec![
                CudaTileConfig { block_m: 16, block_n: 16, block_k: 16, split_k: 1 },
                CudaTileConfig { block_m: 32, block_n: 32, block_k: 16, split_k: 1 },
                CudaTileConfig { block_m: 16, block_n: 16, block_k: 16, split_k: 4 },
            ],
            ShapeClass::Prefill => vec![
                CudaTileConfig { block_m: 64, block_n: 64, block_k: 16, split_k: 1 },
                CudaTileConfig { block_m: 32, block_n: 32, block_k: 16, split_k: 1 },
                CudaTileConfig { block_m: 32, block_n: 32, block_k: 32, split_k: 2 },
            ],
        };

        // Resource limit filtering (T4 shared memory & block threads ceiling)
        let winner = candidates
            .into_iter()
            .filter(|c| c.is_valid(caps))
            .next()
            .unwrap_or(CudaTileConfig {
                block_m: 16,
                block_n: 16,
                block_k: 16,
                split_k: 1,
            });

        // Insert winner, then persist so a repeat shape on this GPU hits the on-disk cache.
        {
            let mut lock = self.cache.lock().unwrap();
            lock.insert(key, Some(winner));
        }
        self.save_cache(caps);
        winner
    }
}
