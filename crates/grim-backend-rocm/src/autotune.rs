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

/// Type alias for benchmark closures.
pub type BenchFn<'a> = dyn FnOnce(KernelKey) -> Result<AutotuneConfig> + Send + 'a;

/// Tuned-config cache: [see: `HashMap<KernelKey, AutotuneConfig>`, `cache_dir`, `{dir}/{gpu_arch}.json`, `hipMemcpy`]
#[derive(Debug)]
pub struct Autotuner {
    device_ordinal: usize,
    gpu_arch: &'static str,
    /// In-memory cache. Pre-allocated empty.
    cache: HashMap<KernelKey, AutotuneConfig>,
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
        self.cache.len()
    }

    /// Look up a recorded config. Returns `None` if absent.
    pub fn lookup(&self, key: KernelKey) -> Option<AutotuneConfig> {
        self.cache.get(&key).copied()
    }

    /// List of keys currently cached (in arbitrary HashMap order).
    pub fn list_keys(&self) -> Vec<KernelKey> {
        self.cache.keys().copied().collect()
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

    /// Read-through cache: returns the recorded config; if absent, runs [see: `bench`, `rust-gpu-discipline`]
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
}

/// On-disk JSON shape. Uses owned `String`s for kernel/arch so a [see: `HashMap`]
#[derive(Debug, Serialize, Deserialize)]
struct AutotuneSnapshotOwned {
    gpu_arch: String,
    entries: Vec<EntrySnapshotOwned>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EntrySnapshotOwned {
    key: KernelKeyOwned,
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
        };
        Ok(serde_json::to_vec_pretty(&snap).map_err(|e| {
            Error::Backend(format!("Autotuner::to_json_bytes: serde_json error: {}", e))
        })?)
    }

    /// Restore from a JSON snapshot. Keys whose `gpu_arch` does not [see: `gpu_arch`]
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
                let kernel_str: &'static str = Box::leak(e.key.kernel.into_boxed_str()); // cache lifetime only.
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
        Ok(t)
    }
}
