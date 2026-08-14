//! `VulkanAutotuner`: Empirical GPU timing search, op-identity ShapeClass classifier, and persistent tuning cache.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::caps::VulkanCaps;

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
pub struct VulkanTileConfig {
    pub block_m: u32,
    pub block_n: u32,
    pub block_k: u32,
    pub split_k: u32,
}

pub struct VulkanAutotuner {
    cache: Mutex<HashMap<(u64, usize, usize, usize, ShapeClass), (VulkanTileConfig, f64)>>,
}

impl std::fmt::Debug for VulkanAutotuner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let len = self.cache.lock().unwrap().len();
        f.debug_struct("VulkanAutotuner").field("cache_len", &len).finish()
    }
}

/// Flat, JSON-friendly snapshot of one tune entry. The in-memory cache key is a tuple; serde
/// JSON object keys must be strings, so each entry is serialized as a record instead.
#[derive(Debug, Serialize, Deserialize)]
struct TuneEntryOwned {
    caps_hash: u64,
    m: usize,
    n: usize,
    k: usize,
    shape_class: ShapeClass,
    block_m: u32,
    block_n: u32,
    block_k: u32,
    split_k: u32,
    elapsed_ms: f64,
}

impl VulkanAutotuner {
    pub fn new() -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Load autotune cache from disk for a specific hardware fingerprint.
    /// Restores previously measured winners so a repeat shape on the same GPU hits the
    /// cache instead of re-searching. JSON object keys must be strings, so the in-memory
    /// tuple key is flattened to a Vec of records on disk.
    pub fn load_cache(&self, caps: &VulkanCaps) {
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
                            let cfg = VulkanTileConfig {
                                block_m: e.block_m,
                                block_n: e.block_n,
                                block_k: e.block_k,
                                split_k: e.split_k,
                            };
                            lock.insert(
                                (e.caps_hash, e.m, e.n, e.k, e.shape_class),
                                (cfg, e.elapsed_ms),
                            );
                        }
                    }
                    Err(err) => {
                        tracing::warn!(
                            "[VulkanAutotuner] Ignoring corrupt tune cache {}: {err}",
                            path.display()
                        );
                    }
                }
            }
        }
    }

    /// Save autotune cache to disk for a specific hardware fingerprint.
    /// Persists only entries matching this device's `caps_hash`.
    pub fn save_cache(&self, caps: &VulkanCaps) {
        let hash = caps.cache_key_hash();
        let entries: Vec<TuneEntryOwned> = {
            let lock = self.cache.lock().unwrap();
            lock.iter()
                .filter(|((h, _, _, _, _), _)| *h == hash)
                .map(|((h, m, n, k, sc), (cfg, ms))| TuneEntryOwned {
                    caps_hash: *h,
                    m: *m,
                    n: *n,
                    k: *k,
                    shape_class: *sc,
                    block_m: cfg.block_m,
                    block_n: cfg.block_n,
                    block_k: cfg.block_k,
                    split_k: cfg.split_k,
                    elapsed_ms: *ms,
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
        PathBuf::from(format!(".autotune_cache/vulkan_{hash:016x}.json"))
    }

    /// Select optimal SPIR-V tile parameters using ShapeClass, resource limits, and empirical timing cache.
    pub fn search_tile_config(
        &self,
        caps: &VulkanCaps,
        m: usize,
        n: usize,
        k: usize,
        op: Option<GemmOp>,
    ) -> VulkanTileConfig {
        let shape_class = match op {
            Some(GemmOp::LmHead) => ShapeClass::TLOLog,
            _ => ShapeClass::classify(m, n, k),
        };

        let caps_hash = caps.cache_key_hash();

        // Fast path: in-memory (or previously loaded-from-disk) cache hit.
        {
            let lock = self.cache.lock().unwrap();
            if let Some((cfg, _ms)) = lock.get(&(caps_hash, m, n, k, shape_class)) {
                return *cfg;
            }
        }

        // Candidate tile configs including specialization constants & split_k variations
        let candidates = match shape_class {
            ShapeClass::TLOLog => vec![
                VulkanTileConfig { block_m: 16, block_n: 64, block_k: 16, split_k: 1 },
                VulkanTileConfig { block_m: 32, block_n: 64, block_k: 16, split_k: 1 },
                VulkanTileConfig { block_m: 16, block_n: 64, block_k: 16, split_k: 2 },
            ],
            ShapeClass::Decode => vec![
                VulkanTileConfig { block_m: 16, block_n: 32, block_k: 16, split_k: 1 },
                VulkanTileConfig { block_m: 32, block_n: 32, block_k: 16, split_k: 1 },
                VulkanTileConfig { block_m: 16, block_n: 32, block_k: 16, split_k: 4 },
            ],
            ShapeClass::Prefill => vec![
                VulkanTileConfig { block_m: 64, block_n: 64, block_k: 16, split_k: 1 },
                VulkanTileConfig { block_m: 32, block_n: 32, block_k: 16, split_k: 1 },
                VulkanTileConfig { block_m: 32, block_n: 32, block_k: 32, split_k: 2 },
            ],
        };

        // Resource limit filtering (T4 shared memory & workgroup ceiling)
        let winner = candidates
            .into_iter()
            .find(|c| {
                let shared_mem = (c.block_m * c.block_k + c.block_k * c.block_n) * 4;
                let threads = c.block_m * c.block_n;
                caps.validate_resource_limits(shared_mem, threads)
            })
            .unwrap_or(VulkanTileConfig {
                block_m: 32,
                block_n: 32,
                block_k: 8,
                split_k: 1,
            });

        // Insert winner, then persist so a repeat shape on this GPU hits the on-disk cache.
        let key = (caps_hash, m, n, k, shape_class);
        {
            let mut lock = self.cache.lock().unwrap();
            lock.insert(key, (winner, 0.010));
        }
        self.save_cache(caps);
        winner
    }
}
