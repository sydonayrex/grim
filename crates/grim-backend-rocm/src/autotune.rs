//! Phase-3 §3.6 — runtime autotuner. [see: `rocm-profiling-perf`, `rocblas_gemm_ex_get_solutions`]

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use grim_tensor::error::{Error, Result};

/// Cache slot identity. [see: `kernel`, `extern "C"`, `"grim_qkv_attention"`, `gpu_arch`]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KernelKey {
    #[serde(default)]
    pub kernel: &'static str,
    #[serde(default)]
    pub gpu_arch: &'static str,
    #[serde(default)]
    pub m: usize,
    #[serde(default)]
    pub n: usize,
    #[serde(default)]
    pub k: usize,
}

/// Tuned launch parameters for a `(kernel, arch, shape)` slot. [see: `block_dim`, `rocm-hip-kernels`, `tile_kv`, `grid_stride`]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AutotuneConfig {
    #[serde(default = "AutotuneConfig::default_block_dim")]
    pub block_dim: u32,
    #[serde(default = "AutotuneConfig::default_tile_kv")]
    pub tile_kv: u32,
    #[serde(default = "AutotuneConfig::default_grid_stride")]
    pub grid_stride: u32,
    #[serde(default)]
    pub cycles_per_invocation: u64,
}

impl AutotuneConfig {
    fn default_block_dim() -> u32 {
        256
    }
    fn default_tile_kv() -> u32 {
        64
    }
    fn default_grid_stride() -> u32 {
        1
    }
}

impl Default for AutotuneConfig {
    fn default() -> Self {
        Self {
            block_dim: Self::default_block_dim(),
            tile_kv: Self::default_tile_kv(),
            grid_stride: Self::default_grid_stride(),
            cycles_per_invocation: 0,
        }
    }
}

/// Cache key for MoE autotuning launch parameters.
///
/// Encapsulates model geometry (`hidden`, `inter`, `num_experts`, `top_k`)
/// and coarse `skew_bucket` (quantized routing skew 0..7) to index MoE tuned configs.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MoeKernelKey {
    pub kernel: String,
    pub gpu_arch: String,
    pub hidden: usize,
    pub inter: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub skew_bucket: u8,
}

/// Quantize a continuous routing skew `[0.0, 1.0]` into 8 coarse skew buckets (`0..7`).
pub fn quantize_routing_skew(skew: f32) -> u8 {
    let clamped = skew.clamp(0.0, 1.0);
    let bucket = (clamped * 7.999_f32) as u8;
    bucket.min(7)
}

/// Type alias for benchmark closures.
pub type BenchFn<'a> = dyn FnOnce(KernelKey) -> Result<AutotuneConfig> + Send + 'a;

/// Tuned-config cache: [see: `HashMap<KernelKey, AutotuneConfig>`, `cache_dir`, `{dir}/{gpu_arch}.json`, `hipMemcpy`]
#[derive(Debug)]
pub struct Autotuner {
    device_ordinal: usize,
    gpu_arch: &'static str,
    /// In-memory cache for dense GEMM keys.
    cache: HashMap<KernelKey, AutotuneConfig>,
    /// In-memory cache for MoE keys (`MoeKernelKey`).
    moe_cache: HashMap<MoeKernelKey, AutotuneConfig>,
    /// Optional on-disk shadow. `None` means "in-memory only".
    cache_dir: Option<PathBuf>,
}

impl Autotuner {
    /// Construct a tuner for a device on a specific arch. Infallible. [see: `get_or_tune`]
    pub fn for_device(device_ordinal: usize, gpu_arch: &'static str) -> Self {
        Self {
            device_ordinal,
            gpu_arch,
            cache: HashMap::new(),
            moe_cache: HashMap::new(),
            cache_dir: None,
        }
    }

    /// Where the on-disk shadow lives, if set. Files: `{cache_dir}/{gpu_arch}.json`.
    pub fn cache_dir(&self) -> Option<&std::path::Path> {
        self.cache_dir.as_deref()
    }

    /// Device ordinal that this tuner was created for. Diagnostics /
    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    /// Configure the on-disk shadow directory. The autotuner does [see: `save()`, `record()`]
    pub fn set_cache_dir(&mut self, dir: PathBuf) {
        self.cache_dir = Some(dir);
    }

    /// Number of cached entries (in-memory).
    pub fn len(&self) -> usize {
        self.cache.len() + self.moe_cache.len()
    }

    /// Look up a recorded config. Returns `None` if absent.
    pub fn lookup(&self, key: KernelKey) -> Option<AutotuneConfig> {
        self.cache.get(&key).copied()
    }

    /// Look up a recorded MoE config. Returns `None` if absent.
    pub fn lookup_moe(&self, key: &MoeKernelKey) -> Option<AutotuneConfig> {
        self.moe_cache.get(key).copied()
    }

    /// List of keys currently cached (in arbitrary HashMap order).
    pub fn list_keys(&self) -> Vec<KernelKey> {
        self.cache.keys().copied().collect()
    }

    /// List of MoE keys currently cached.
    pub fn list_moe_keys(&self) -> Vec<MoeKernelKey> {
        self.moe_cache.keys().cloned().collect()
    }

    /// Insert a config directly. Used by `get_or_tune` on cache miss. [see: `Err`]
    pub fn record(&mut self, key: KernelKey, config: AutotuneConfig) -> Result<()> {
        if key.gpu_arch != self.gpu_arch {
            return Err(Error::Backend(format!(
                "Autotuner::record: architecture mismatch (key.gpu_arch={}, tuner.gpu_arch={}); \
                 this is a programming mistake, not a runtime condition",
                key.gpu_arch, self.gpu_arch
            )));
        }
        self.cache.insert(key, config);
        Ok(())
    }

    /// Insert an MoE config directly.
    pub fn record_moe(&mut self, key: MoeKernelKey, config: AutotuneConfig) -> Result<()> {
        if key.gpu_arch != self.gpu_arch {
            return Err(Error::Backend(format!(
                "Autotuner::record_moe: architecture mismatch (key.gpu_arch={}, tuner.gpu_arch={})",
                key.gpu_arch, self.gpu_arch
            )));
        }
        self.moe_cache.insert(key, config);
        Ok(())
    }

    /// Read-through cache for standard GEMM keys.
    pub fn get_or_tune<F>(&mut self, key: KernelKey, bench: F) -> Result<AutotuneConfig>
    where
        F: FnOnce(KernelKey) -> Result<AutotuneConfig>,
    {
        if let Some(cfg) = self.cache.get(&key).copied() {
            return Ok(cfg);
        }
        let cfg = bench(key)?;
        self.record(key, cfg)?;
        Ok(cfg)
    }

    /// Read-through cache for MoE keys.
    pub fn get_or_tune_moe<F>(&mut self, key: MoeKernelKey, bench: F) -> Result<AutotuneConfig>
    where
        F: FnOnce(&MoeKernelKey) -> Result<AutotuneConfig>,
    {
        if let Some(cfg) = self.moe_cache.get(&key).copied() {
            return Ok(cfg);
        }
        let cfg = bench(&key)?;
        self.record_moe(key.clone(), cfg)?;
        Ok(cfg)
    }

    /// Read-through lookup for MoE block dimension. On a cache miss, returns `default_block_dim`
    /// without caching fake timing data, so real autotuning sweeps populate true measurements.
    pub fn get_or_tune_moe_block_dim(
        &mut self,
        key: &MoeKernelKey,
        default_block_dim: u32,
    ) -> u32 {
        self.moe_cache
            .get(key)
            .map(|cfg| cfg.block_dim)
            .unwrap_or(default_block_dim)
    }
}



/// On-disk JSON shape. Uses owned `String`s for kernel/arch.
#[derive(Debug, Serialize, Deserialize)]
struct AutotuneSnapshotOwned {
    gpu_arch: String,
    entries: Vec<EntrySnapshotOwned>,
    #[serde(default)]
    moe_entries: Vec<MoeEntrySnapshotOwned>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EntrySnapshotOwned {
    key: KernelKeyOwned,
    config: AutotuneConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MoeEntrySnapshotOwned {
    key: MoeKernelKey,
    config: AutotuneConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct KernelKeyOwned {
    kernel: String,
    gpu_arch: String,
    m: usize,
    n: usize,
    k: usize,
}

impl From<KernelKey> for KernelKeyOwned {
    fn from(k: KernelKey) -> Self {
        Self {
            kernel: k.kernel.to_string(),
            gpu_arch: k.gpu_arch.to_string(),
            m: k.m,
            n: k.n,
            k: k.k,
        }
    }
}

impl Autotuner {
    /// Serialize the entire cache for persistence (owned-string wire format).
    pub fn to_json_bytes(&self) -> Result<Vec<u8>> {
        let snap = AutotuneSnapshotOwned {
            gpu_arch: self.gpu_arch.to_string(),
            entries: self
                .cache
                .iter()
                .map(|(k, v)| EntrySnapshotOwned {
                    key: KernelKeyOwned::from(*k),
                    config: *v,
                })
                .collect(),
            moe_entries: self
                .moe_cache
                .iter()
                .map(|(k, v)| MoeEntrySnapshotOwned {
                    key: k.clone(),
                    config: *v,
                })
                .collect(),
        };
        Ok(serde_json::to_vec_pretty(&snap).map_err(|e| {
            Error::Backend(format!("Autotuner::to_json_bytes: serde_json error: {}", e))
        })?)
    }

    /// Save the JSON snapshot to a file path.
    pub fn save_to_file(&self, path: &std::path::Path) -> Result<()> {
        let bytes = self.to_json_bytes()?;
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(path, bytes).map_err(|e| {
            Error::Backend(format!("Autotuner::save_to_file: {}", e))
        })
    }

    /// Restore from a JSON snapshot.
    pub fn from_json_bytes(

        device_ordinal: usize,
        gpu_arch: &'static str,
        bytes: &[u8],
    ) -> Result<Self> {
        let snap: AutotuneSnapshotOwned = serde_json::from_slice(bytes).map_err(|e| {
            Error::Backend(format!(
                "Autotuner::from_json_bytes: serde_json error: {}",
                e
            ))
        })?;
        let mut t = Self::for_device(device_ordinal, gpu_arch);
        for e in snap.entries {
            if e.key.gpu_arch == gpu_arch {
                let kernel_str: &'static str = Box::leak(e.key.kernel.into_boxed_str());
                let arch_str: &'static str = Box::leak(e.key.gpu_arch.into_boxed_str());
                let key = KernelKey {
                    kernel: kernel_str,
                    gpu_arch: arch_str,
                    m: e.key.m,
                    n: e.key.n,
                    k: e.key.k,
                };
                t.cache.insert(key, e.config);
            }
        }
        for e in snap.moe_entries {
            if e.key.gpu_arch == gpu_arch {
                t.moe_cache.insert(e.key, e.config);
            }
        }
        Ok(t)
    }
}

