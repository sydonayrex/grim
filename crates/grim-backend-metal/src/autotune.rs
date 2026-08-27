//! `MetalAutotuner`: Empirical MSL pipeline timing, ShapeClass op-identity classification, and persistent tuning cache.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

use crate::caps::MetalCaps;

/// Shape classification for GEMM workloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct MetalTileConfig {
    pub block_m: u32,
    pub block_n: u32,
    pub block_k: u32,
    pub split_k: u32,
}

impl MetalTileConfig {
    /// Validates whether this tile config satisfies the resource limits of `caps`.
    pub fn is_valid(&self, caps: &MetalCaps) -> bool {
        let shared_mem = (self.block_m * self.block_k + self.block_k * self.block_n) * 4;
        let threads = self.block_m * self.block_n;
        caps.validate_resource_limits(shared_mem, threads)
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct CacheEntry {
    caps_hash: u64,
    m: usize,
    n: usize,
    k: usize,
    shape_class: String,
    config: MetalTileConfig,
    latency_ms: f64,
}

type MetalAutotuneCache =
    Mutex<HashMap<(u64, usize, usize, usize, ShapeClass), (MetalTileConfig, f64)>>;

#[derive(Debug, Default)]
pub struct MetalAutotuner {
    cache: MetalAutotuneCache,
}

impl MetalAutotuner {
    pub fn new() -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Load autotune cache from disk for a specific hardware fingerprint.
    pub fn load_cache(&self, caps: &MetalCaps) {
        let hash = caps.cache_key_hash();
        let path = PathBuf::from(format!(".autotune_cache/metal_{:016x}.json", hash));
        if path.exists() {
            if let Ok(data) = fs::read_to_string(&path) {
                if let Ok(entries) = serde_json::from_str::<Vec<CacheEntry>>(&data) {
                    let mut lock = self.cache.lock().unwrap();
                    for entry in entries {
                        let sc = match entry.shape_class.as_str() {
                            "TLOLog" => ShapeClass::TLOLog,
                            "Decode" => ShapeClass::Decode,
                            _ => ShapeClass::Prefill,
                        };
                        lock.insert(
                            (entry.caps_hash, entry.m, entry.n, entry.k, sc),
                            (entry.config, entry.latency_ms),
                        );
                    }
                    tracing::info!("[MetalAutotuner] Loaded tune cache from {}", path.display());
                }
            }
        }
    }

    /// Save autotune cache to disk.
    pub fn save_cache(&self, caps: &MetalCaps) {
        let hash = caps.cache_key_hash();
        let _ = fs::create_dir_all(".autotune_cache");
        let path = PathBuf::from(format!(".autotune_cache/metal_{:016x}.json", hash));

        let lock = self.cache.lock().unwrap();
        let entries: Vec<CacheEntry> = lock
            .iter()
            .map(
                |(&(c_hash, m, n, k, sc), &(config, latency_ms))| CacheEntry {
                    caps_hash: c_hash,
                    m,
                    n,
                    k,
                    shape_class: format!("{:?}", sc),
                    config,
                    latency_ms,
                },
            )
            .collect();

        if let Ok(json) = serde_json::to_string_pretty(&entries) {
            let _ = fs::write(path, json);
        }
    }

    /// Select optimal Metal tile parameters using ShapeClass, resource limits, and empirical timing cache.
    pub fn search_tile_config(
        &self,
        caps: &MetalCaps,
        m: usize,
        n: usize,
        k: usize,
        op: Option<GemmOp>,
    ) -> MetalTileConfig {
        self.search_tile_config_measured::<fn(&MetalTileConfig) -> Option<f64>>(
            caps, m, n, k, op, None,
        )
    }

    /// Select optimal Metal tile parameters using empirical GPU timing benchmarking when available.
    pub fn search_tile_config_measured<F>(
        &self,
        caps: &MetalCaps,
        m: usize,
        n: usize,
        k: usize,
        op: Option<GemmOp>,
        bench_fn: Option<F>,
    ) -> MetalTileConfig
    where
        F: Fn(&MetalTileConfig) -> Option<f64>,
    {
        let shape_class = match op {
            Some(op_tag) => ShapeClass::from_op(op_tag, m, n, k),
            None => ShapeClass::classify(m, n, k),
        };

        let caps_hash = caps.cache_key_hash();
        let mut lock = self.cache.lock().unwrap();
        let key = (caps_hash, m, n, k, shape_class);

        if let Some((cfg, _ms)) = lock.get(&key) {
            return *cfg;
        }

        // Candidate tile configs including split_k variations
        let candidates = match shape_class {
            ShapeClass::TLOLog => vec![
                MetalTileConfig {
                    block_m: 16,
                    block_n: 64,
                    block_k: 16,
                    split_k: 1,
                },
                MetalTileConfig {
                    block_m: 32,
                    block_n: 64,
                    block_k: 16,
                    split_k: 1,
                },
                MetalTileConfig {
                    block_m: 16,
                    block_n: 64,
                    block_k: 16,
                    split_k: 2,
                },
            ],
            ShapeClass::Decode => vec![
                MetalTileConfig {
                    block_m: 16,
                    block_n: 32,
                    block_k: 16,
                    split_k: 1,
                },
                MetalTileConfig {
                    block_m: 32,
                    block_n: 32,
                    block_k: 16,
                    split_k: 1,
                },
                MetalTileConfig {
                    block_m: 16,
                    block_n: 32,
                    block_k: 16,
                    split_k: 4,
                },
            ],
            ShapeClass::Prefill => vec![
                MetalTileConfig {
                    block_m: 64,
                    block_n: 64,
                    block_k: 16,
                    split_k: 1,
                },
                MetalTileConfig {
                    block_m: 32,
                    block_n: 32,
                    block_k: 16,
                    split_k: 1,
                },
                MetalTileConfig {
                    block_m: 32,
                    block_n: 32,
                    block_k: 32,
                    split_k: 2,
                },
            ],
        };

        // Resource limit filtering using MetalTileConfig::is_valid
        let valid_candidates: Vec<MetalTileConfig> = candidates
            .into_iter()
            .filter(|c| c.is_valid(caps))
            .collect();

        let mut winner = valid_candidates
            .first()
            .copied()
            .unwrap_or(MetalTileConfig {
                block_m: 32,
                block_n: 32,
                block_k: 8,
                split_k: 1,
            });
        let mut best_latency = 0.008f64;

        if let Some(bench) = bench_fn {
            let mut min_t = f64::MAX;
            for candidate in &valid_candidates {
                if let Some(latency) = bench(candidate) {
                    if latency < min_t {
                        min_t = latency;
                        winner = *candidate;
                        best_latency = latency;
                    }
                }
            }
        }

        lock.insert(key, (winner, best_latency));
        winner
    }
}
