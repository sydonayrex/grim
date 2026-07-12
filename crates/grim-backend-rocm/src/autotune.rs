//! Phase-3 §3.6 — runtime autotuner.
//!
//! The autotuner caches `(kernel_name, gpu_arch, problem_shape) ->
//! AutotuneConfig` in memory and (optionally) on disk as JSON, so the
//! fused-decode scheduler can pull a tuned launch config without
//! re-benchmarking every process invocation.
//!
//! Skill attribution:
//! - `rocm-profiling-perf` — autotune loop methodology, runtime
//!   `rocblas_gemm_ex_get_solutions` enumeration (this module is the
//!   generic wrapper; the GEMM-specific enumerator in
//!   `lib.rs::lookup_solution_index` is its companion).
//! - `rust-gpu-discipline` §4 — every benchmark closure must return a
//!   *measured* number; no synthesized "fast" config sneaks into the
//!   cache on a synthetic closure.
//! - `rust-ai-ml-inference-guide` Action 8 — evaluate under controlled
//!   shapes before deploying (this is the cache that holds the result
//!   of those evaluations).

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use grim_tensor::error::{Error, Result};

/// Cache slot identity.
///
/// Fields:
/// - `kernel`     — `extern "C"` entry name (e.g. `"grim_qkv_attention"`).
/// - `gpu_arch`   — the rocBLAS / `--offload-arch` target (e.g.
///                  `"gfx1036"`, `"gfx942"`, `"gfx1200"`). Configs are
///                  never reused across arches.
/// - `m, n, k`    — the launch shape this config was tuned for. At
///                  minimum we distinguish these three; callers may use
///                  `m == 0` and any of n/k == 0 to mean "doesn't apply".
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

/// Tuned launch parameters for a `(kernel, arch, shape)` slot.
///
/// Field semantics mirror the spec's §3.6 starter:
/// - `block_dim`              — launch block size (multiple of 64 on RDNA;
///                              see `rocm-hip-kernels` Wave64 mandate).
/// - `tile_kv`                 — KV-tile size for attention-style kernels.
/// - `grid_stride`             — persistent-kernel grid stride.
/// - `cycles_per_invocation`   — measured median cycles (host cycles;
///                              can be obtained from `rocprof` counters
///                              per `rocm-profiling-perf`).
///
/// `Default` chooses a sane-ish mid-range config that the caller can
/// override — *not* a synthesized "fast" value per `rust-gpu-discipline`.
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
    fn default_block_dim() -> u32 { 256 }
    fn default_tile_kv() -> u32 { 64 }
    fn default_grid_stride() -> u32 { 1 }
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

/// Tuned-config cache:
/// - in-memory `HashMap<KernelKey, AutotuneConfig>` for hot-path lookups,
/// - optional `cache_dir` shadow file at `{dir}/{gpu_arch}.json` for
///   process restart (per spec §3.6 starter).
///
/// The autotuner does NOT touch the GPU on construction — no `hipMemcpy`,
/// no `rocblas_gemm_ex_get_solutions` until `get_or_tune` runs a closure
/// that itself runs those calls. Building a tuner is a free operation
/// (`rust-gpu-discipline` §0: no synthetic GPU state).
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
    /// Construct a tuner for a device on a specific arch. Infallible.
    /// The actual benchmark loop runs in `get_or_tune`.
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
    /// logging only — never used to influence the cache content.
    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    /// Configure the on-disk shadow directory. The autotuner does
    /// *not* create the directory on construction — call `save()` to
    /// flush, or rely on `record()` + `save()` for symmetric writes.
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
    /// Useful for diagnostics / debugging.
    pub fn list_keys(&self) -> Vec<KernelKey> {
        self.cache.keys().copied().collect()
    }

    /// Insert a config directly. Used by `get_or_tune` on cache miss.
    /// Returns `Err` if the cache is poisoned (shouldn't happen).
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

    /// Read-through cache: returns the recorded config; if absent, runs
    /// `bench` exactly once and records its result. A bench failure is
    /// *not* cached — the next call retries.
    ///
    /// Per `rust-gpu-discipline`: a bench closure is the only legitimate
    /// path into the cache. Repeated identical keys (e.g. two callers
    /// in the same process racing on the same key) must not both run
    /// the bench; we serialize through a single-threaded path because
    /// tuning is by design infrequent.
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

/// On-disk JSON shape. Uses owned `String`s for kernel/arch so a
/// deserialize doesn't have to thread a lifetime through every
/// `HashMap` lookup.
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

    /// Restore from a JSON snapshot. Keys whose `gpu_arch` does not
    /// equal the constructor's `gpu_arch` are dropped (static
    /// protection against cross-arch contamination).
    pub fn from_json_bytes(
        device_ordinal: usize,
        gpu_arch: &'static str,
        bytes: &[u8],
    ) -> Result<Self> {
        let snap: AutotuneSnapshotOwned = serde_json::from_slice(bytes).map_err(|e| {
            Error::Backend(format!("Autotuner::from_json_bytes: serde_json error: {}", e))
        })?;
        let mut t = Self::for_device(device_ordinal, gpu_arch);
        for e in snap.entries {
            if e.key.gpu_arch == gpu_arch {
                let kernel_str: &'static str =
                    Box::leak(e.key.kernel.into_boxed_str()); // cache lifetime only.
                let arch_str: &'static str =
                    Box::leak(e.key.gpu_arch.into_boxed_str());
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
