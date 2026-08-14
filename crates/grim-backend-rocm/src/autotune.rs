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

/// Coarse key hierarchy for kernel compilation (excludes fast-changing dimensions like batch/seq_len).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CompileKey {
    pub kernel: String,
    pub gpu_arch: String,
    pub shape_class: ShapeClass,
    pub features: FeatureSet,
}

/// Fine-grained key hierarchy for exact launch tuning.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TuneKey {
    pub compile_key: CompileKey,
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

/// Broad category of tensor dimension shapes (e.g., Decode vs Prefill vs TLOLog).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ShapeClass {
    Decode,  // m == 1: per-token GEMM
    Prefill, // m > 1: large-batch GEMM
    TLOLog,  // lm_head / logit-projection ONLY — tagged by op-identity, NOT by m
}

/// Known GEMM operational types passed into GEMM classification.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GemmOp {
    Attention,
    Ffn,
    LmHead, // -> ShapeClass::TLOLog
    Other,
}

impl ShapeClass {
    pub fn from_m(m: usize) -> Self {
        if m == 1 { Self::Decode } else { Self::Prefill }
    }

    /// Op-aware classifier. LmHead is TLOLog no matter its m; everything else bins by m.
    pub fn from_op(op: GemmOp, m: usize) -> Self {
        match op {
            GemmOp::LmHead => Self::TLOLog,
            _ => Self::from_m(m),
        }
    }
}

/// Hardware features and instruction sets required by a kernel configuration.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FeatureSet {
    pub requires_wmma: bool,
    pub requires_fp8_mfma: bool,
}

impl FeatureSet {
    pub fn scalar() -> Self {
        Self {
            requires_wmma: false,
            requires_fp8_mfma: false,
        }
    }

    pub fn wmma() -> Self {
        Self {
            requires_wmma: true,
            requires_fp8_mfma: false,
        }
    }

    pub fn fp8_mfma() -> Self {
        Self {
            requires_wmma: true,
            requires_fp8_mfma: true,
        }
    }

    /// Check if target GPU architecture supports this feature set.
    pub fn supported_on(&self, arch: &str) -> bool {
        let is_gfx11 = arch.starts_with("gfx11");
        let is_gfx12 = arch.starts_with("gfx12");

        if self.requires_fp8_mfma {
            return is_gfx12;
        }
        if self.requires_wmma {
            return is_gfx11 || is_gfx12;
        }
        true
    }
}

/// Detailed candidate launch configuration for ROCm kernels.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LaunchConfig {
    pub block_m: u32,
    pub block_n: u32,
    pub block_k: u32,
    pub split_k: u32,
    pub threads: u32,
}

impl LaunchConfig {
    /// Calculate static shared memory (LDS) requirements in bytes for this configuration.
    pub fn smem_cost(&self, bytes_per_elem: u32) -> u32 {
        (self.block_m * self.block_k + self.block_k * self.block_n) * bytes_per_elem
    }

    /// Check if config violates LDS capacity or hardware constraints.
    pub fn is_valid(&self, arch: &str, device_smem_limit: u32, bytes_per_elem: u32) -> bool {
        if self.smem_cost(bytes_per_elem) > device_smem_limit {
            return false;
        }
        // Mutual exclusion: small block_m with high split_k causes excessive synchronization overhead
        if self.block_m <= 8 && self.split_k > 4 {
            return false;
        }
        // Thread block size must be multiple of wave size (32/64)
        if self.threads % 32 != 0 || self.threads == 0 {
            return false;
        }
        let _ = arch;
        true
    }
}

/// Candidate generator for Charon (fused MoE grouped GEMM) scalar path.
pub fn charon_scalar_candidates(arch: &str, device_smem_limit: u32) -> Vec<LaunchConfig> {
    let mut candidates = Vec::new();
    let block_m_opts = [8, 16, 32, 64];
    let block_n_opts = [32, 64, 128];
    let block_k_opts = [32, 64];
    let split_k_opts = [1, 2, 4];
    let threads_opts = [64, 128, 256];

    for &bm in &block_m_opts {
        for &bn in &block_n_opts {
            for &bk in &block_k_opts {
                for &sk in &split_k_opts {
                    for &t in &threads_opts {
                        let cfg = LaunchConfig {
                            block_m: bm,
                            block_n: bn,
                            block_k: bk,
                            split_k: sk,
                            threads: t,
                        };
                        if cfg.is_valid(arch, device_smem_limit, 2) {
                            candidates.push(cfg);
                        }
                    }
                }
            }
        }
    }
    candidates
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
    pub fn get_or_tune_moe_block_dim(&mut self, key: &MoeKernelKey, default_block_dim: u32) -> u32 {
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
        std::fs::write(path, bytes)
            .map_err(|e| Error::Backend(format!("Autotuner::save_to_file: {}", e)))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_class_from_m() {
        assert_eq!(ShapeClass::from_m(1), ShapeClass::Decode);
        assert_eq!(ShapeClass::from_m(2), ShapeClass::Prefill);
        assert_eq!(ShapeClass::from_m(128), ShapeClass::Prefill);
    }

    #[test]
    fn test_feature_set_supported_on() {
        let scalar = FeatureSet::scalar();
        let wmma = FeatureSet::wmma();
        let fp8_mfma = FeatureSet::fp8_mfma();

        // GFX1036 (RDNA2) supports scalar, but NOT WMMA or FP8 MFMA
        assert!(scalar.supported_on("gfx1036"));
        assert!(!wmma.supported_on("gfx1036"));
        assert!(!fp8_mfma.supported_on("gfx1036"));

        // GFX1100 (RDNA3) supports scalar and WMMA, but NOT FP8 MFMA
        assert!(scalar.supported_on("gfx1100"));
        assert!(wmma.supported_on("gfx1100"));
        assert!(!fp8_mfma.supported_on("gfx1100"));

        // GFX1200 (CDNA3/RDNA4) supports scalar, WMMA, and FP8 MFMA
        assert!(scalar.supported_on("gfx1200"));
        assert!(wmma.supported_on("gfx1200"));
        assert!(fp8_mfma.supported_on("gfx1200"));
    }

    #[test]
    fn test_launch_config_pruning() {
        let arch = "gfx1036";
        let smem_limit = 65536; // 64 KiB

        // Valid config: small LDS, acceptable split_k
        let valid = LaunchConfig {
            block_m: 16,
            block_n: 64,
            block_k: 32,
            split_k: 1,
            threads: 128,
        };
        assert!(valid.is_valid(arch, smem_limit, 2));

        // Invalid config: LDS memory exceeds capacity limit
        let smem_overflow = LaunchConfig {
            block_m: 128,
            block_n: 128,
            block_k: 256, // Requires (128*256 + 256*128)*2 = 131,072 bytes > 64 KiB
            split_k: 1,
            threads: 256,
        };
        assert!(!smem_overflow.is_valid(arch, smem_limit, 2));

        // Invalid config: Mutual exclusion (block_m <= 8 with split_k > 4)
        let mutual_excl = LaunchConfig {
            block_m: 8,
            block_n: 64,
            block_k: 32,
            split_k: 8,
            threads: 64,
        };
        assert!(!mutual_excl.is_valid(arch, smem_limit, 2));
    }

    #[test]
    fn test_charon_scalar_candidates_generation() {
        let candidates = charon_scalar_candidates("gfx1036", 65536);
        assert!(
            !candidates.is_empty(),
            "charon candidates generated should be non-empty"
        );

        for cfg in &candidates {
            assert!(cfg.is_valid("gfx1036", 65536, 2));
            assert!(cfg.smem_cost(2) <= 65536);
        }
    }
}
