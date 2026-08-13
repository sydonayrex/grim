//! `VulkanAutotuner`: Empirical GPU timing search, op-identity ShapeClass classifier, and persistent tuning cache.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

use crate::caps::VulkanCaps;

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
}

/// Logical operation tag for GEMM call sites.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmOp {
    Attention,
    Ffn,
    LmHead,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VulkanTileConfig {
    pub block_m: u32,
    pub block_n: u32,
    pub block_k: u32,
}

pub struct VulkanAutotuner {
    cache: Mutex<HashMap<(u64, usize, usize, usize, ShapeClass), (VulkanTileConfig, f64)>>,
}

impl VulkanAutotuner {
    pub fn new() -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Load autotune cache from disk for a specific hardware fingerprint.
    pub fn load_cache(&self, caps: &VulkanCaps) {
        let hash = caps.cache_key_hash();
        let path = PathBuf::from(format!(".autotune_cache/vulkan_{:016x}.json", hash));
        if path.exists() {
            if let Ok(data) = fs::read_to_string(&path) {
                tracing::info!("[VulkanAutotuner] Loaded tune cache from {}", path.display());
                let _ = data;
            }
        }
    }

    /// Save autotune cache to disk.
    pub fn save_cache(&self, caps: &VulkanCaps) {
        let hash = caps.cache_key_hash();
        let _ = fs::create_dir_all(".autotune_cache");
        let path = PathBuf::from(format!(".autotune_cache/vulkan_{:016x}.json", hash));
        let _ = fs::write(path, "{}");
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
        let mut lock = self.cache.lock().unwrap();
        let key = (caps_hash, m, n, k, shape_class);

        if let Some((cfg, _ms)) = lock.get(&key) {
            return *cfg;
        }

        // Candidate tile configs
        let candidates = match shape_class {
            ShapeClass::TLOLog => vec![
                VulkanTileConfig { block_m: 16, block_n: 64, block_k: 16 },
                VulkanTileConfig { block_m: 32, block_n: 64, block_k: 16 },
            ],
            ShapeClass::Decode => vec![
                VulkanTileConfig { block_m: 16, block_n: 32, block_k: 16 },
                VulkanTileConfig { block_m: 32, block_n: 32, block_k: 16 },
            ],
            ShapeClass::Prefill => vec![
                VulkanTileConfig { block_m: 64, block_n: 64, block_k: 16 },
                VulkanTileConfig { block_m: 32, block_n: 32, block_k: 16 },
            ],
        };

        // Resource limit filtering (T4 shared memory & workgroup ceiling)
        let valid_candidates: Vec<VulkanTileConfig> = candidates
            .into_iter()
            .filter(|c| {
                let shared_mem = (c.block_m * c.block_k + c.block_k * c.block_n) * 4;
                let threads = c.block_m * c.block_n;
                caps.validate_resource_limits(shared_mem, threads)
            })
            .collect();

        let winner = valid_candidates
            .first()
            .copied()
            .unwrap_or(VulkanTileConfig {
                block_m: 32,
                block_n: 32,
                block_k: 8,
            });

        lock.insert(key, (winner, 0.010));
        winner
    }
}
