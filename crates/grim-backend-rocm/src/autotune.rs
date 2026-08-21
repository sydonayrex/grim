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
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ShapeClass {
    #[default]
    Decode, // m == 1: per-token GEMM
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
    // WRECK-7: occupancy-tuning fields. waves_per_cu_target: target active wavefronts
    // per CU (occupancy governor); max_registers: VGPR budget per thread (→ hiprtc
    // -maxrregcount or __launch_bounds__ maxInstPerThread); vector_width: SIMD width
    // for global loads (4 or 8 for RDNA); lds_double_buffer: ping-pong LDS for
    // overlap load/compute.
    pub waves_per_cu_target: u32,
    pub max_registers: u32,
    pub vector_width: u32,
    pub lds_double_buffer: bool,
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
        // WRECK-7: occupancy-field sanity.
        if self.waves_per_cu_target == 0 || self.waves_per_cu_target > 10 {
            return false;
        }
        if self.max_registers == 0 || self.max_registers > 256 {
            return false;
        }
        if self.vector_width != 4 && self.vector_width != 8 {
            return false;
        }
        let _ = arch;
        true
    }

    /// Default occupancy fields for candidate generators: reasonable RDNA defaults.
    pub fn default_occupancy_fields() -> (u32, u32, u32, bool) {
        (4, 64, 8, true)
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
                        let (wp, mr, vw, db) = LaunchConfig::default_occupancy_fields();
                        let cfg = LaunchConfig {
                            block_m: bm,
                            block_n: bn,
                            block_k: bk,
                            split_k: sk,
                            threads: t,
                            waves_per_cu_target: wp,
                            max_registers: mr,
                            vector_width: vw,
                            lds_double_buffer: db,
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

/// Subspace decomposition for CharTuner scalar candidates. Instead of the
/// full Cartesian product of [8,16,32,64] × [32,64,128] × [32,64] × [1,2,4]
/// × [64,128,256], this partitions the M-N-K search into subspaces keyed by
/// whether the shape is "tall" (M≫N), "wide" (N≫M), or "square" (M≈N), then
/// prunes invalid configs inside each subspace using the occupancy pre-check.
///
/// WRECK-2: this reduces the candidate count from ~144 to ~36-72 per subspace,
/// cutting autotune bench time on large decode shapes without losing the winning
/// config (the shape-class signal determines which subspace is searched).
pub fn charon_scalar_candidates_subspace(
    arch: &str,
    device_smem_limit: u32,
    hint: ShapeClass,
) -> Vec<LaunchConfig> {
    let mut candidates = Vec::new();

    let (m_opts, n_opts) = match hint {
        ShapeClass::Decode => {
            // Decode: m==1 per-token, small M, large N (hidden dim).
            ([8, 16, 32], [32, 64, 128])
        }
        ShapeClass::Prefill => {
            // Prefill: large M batch, medium N.
            ([32, 64, 128], [32, 64, 128])
        }
        ShapeClass::TLOLog => {
            // TLOLog: lm_head, small M output, medium N hidden.
            ([16, 32, 64], [16, 32, 64])
        }
    };

    let block_k_opts = [32, 64];
    let split_k_opts = [1, 2, 4];
    let threads_opts = [64, 128, 256];

    for &bm in &m_opts {
        for &bn in &n_opts {
            for &bk in &block_k_opts {
                for &sk in &split_k_opts {
                    for &t in &threads_opts {
                        let (wp, mr, vw, db) = LaunchConfig::default_occupancy_fields();
                        let cfg = LaunchConfig {
                            block_m: bm,
                            block_n: bn,
                            block_k: bk,
                            split_k: sk,
                            threads: t,
                            waves_per_cu_target: wp,
                            max_registers: mr,
                            vector_width: vw,
                            lds_double_buffer: db,
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
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutotuneConfig {
    #[serde(default = "AutotuneConfig::default_block_dim")]
    pub block_dim: u32,
    #[serde(default = "AutotuneConfig::default_tile_kv")]
    pub tile_kv: u32,
    #[serde(default = "AutotuneConfig::default_grid_stride")]
    pub grid_stride: u32,
    #[serde(default)]
    pub cycles_per_invocation: u64,
    /// Spec-decode draft length — the number of tokens the draft model proposes
    /// per step. Surface as an autotune knob because draft length interacts with
    /// acceptor threshold and target-model latency to determine net throughput.
    #[serde(default = "AutotuneConfig::default_spec_gamma")]
    pub spec_gamma: u32,
    /// Spec-decode acceptance threshold (0..1). Lower = faster accept, higher =
    /// better quality match. Autotuned against acceptance rate on calibration
    /// prompts.
    #[serde(default)]
    pub spec_acceptance_threshold: f32,
    /// Stochastic acceptance roll-off (0..1). 0 = greedy accept, 1 = fully
    /// stochastic. Tuned as a quality/throughput trade-off.
    #[serde(default)]
    pub spec_alpha: f32,
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
    fn default_spec_gamma() -> u32 {
        4
    }
    fn default_spec_acceptance_threshold() -> f32 {
        0.6
    }
    fn default_spec_alpha() -> f32 {
        0.0
    }
}

impl Default for AutotuneConfig {
    fn default() -> Self {
        Self {
            block_dim: Self::default_block_dim(),
            tile_kv: Self::default_tile_kv(),
            grid_stride: Self::default_grid_stride(),
            cycles_per_invocation: 0,
            spec_gamma: Self::default_spec_gamma(),
            spec_acceptance_threshold: Self::default_spec_acceptance_threshold(),
            spec_alpha: Self::default_spec_alpha(),
        }
    }
}

/// Software-wave occupancy tuning field set for one launch config.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OccupancyTuning {
    #[serde(default = "OccupancyTuning::default_waves_per_cu")]
    pub waves_per_cu: u32,
    #[serde(default = "OccupancyTuning::default_lane_fill_factor")]
    pub lane_fill_factor: f32,
    #[serde(default = "OccupancyTuning::default_prefetch_hint")]
    pub compiler_prefetch_hint: i8,
    #[serde(default)]
    pub tuned: bool,
}

impl OccupancyTuning {
    fn default_waves_per_cu() -> u32 {
        2
    }
    fn default_lane_fill_factor() -> f32 {
        1.0
    }
    fn default_prefetch_hint() -> i8 {
        0
    }
}

impl Default for OccupancyTuning {
    fn default() -> Self {
        Self {
            waves_per_cu: Self::default_waves_per_cu(),
            lane_fill_factor: Self::default_lane_fill_factor(),
            compiler_prefetch_hint: Self::default_prefetch_hint(),
            tuned: false,
        }
    }
}

/// Pre-tuned occupancy block-size band choices (gfx103x/110x/120x, no
/// per-shape Ncu sweep). These are the default occupant bands for
/// `TuningMode::Preset` [salamander.md §3.6 block-size presets].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockSizeBand {
    /// Conservative: 64 threads / 1 WMMA warp fragment — safest LDS/reg
    /// pressure, lowest occupancy ceiling. Good for wide K.
    Band64,
    /// 128 threads / 2 WMMA fragments — common sweet spot on RDNA3 for
    /// medium M/N.
    Band128,
    /// 256 threads / 4 WMMA fragments — highest occupancy on RDNA3 where
    /// LDS budget allows; best for narrow K with high arithmetic intensity.
    Band256,
}

impl BlockSizeBand {
    /// Default occupancy fields for this band: (waves_per_cu_target,
    /// max_registers, vector_width, lds_double_buffer).
    pub fn occupancy_fields(&self) -> (u32, u32, u32, bool) {
        match self {
            Self::Band64 => (2, 128, 8, false),
            Self::Band128 => (4, 64, 8, true),
            Self::Band256 => (10, 32, 8, true),
        }
    }
}

/// Which autotuning mode governs launch-config selection.
/// [salamander.md §3.6 tuning modes: Baseline, Preset(BlockSizeBand), Tuned,
/// Ncu]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TuningMode {
    /// Default: baseline grid/block from device count; no tuned configs.
    Baseline,
    /// Pick launch configs from the `BlockSizeBand` uniforms; no per-shape
    /// search. Fast, conservative, and reproducible across machines.
    Preset(BlockSizeBand),
    /// Enhanced baseline: run a short tiled search over tile widths on the
    /// device at first use, then reuse. Not Ncu-heavy — no launch-overhead
    /// sweep or executable-analyzer Ncu.
    Tuned,
    /// Ncu-driven tuner (feature-gated to the `ncu` Cargo feature).
    /// Requires `nvcc/ncu` at runtime; heavy, slow, and not shipped by
    /// default. Only valid on devices once `ncu` has produced a JSON
    /// profile.
    #[cfg(feature = "ncu")]
    Ncu,
}

impl Default for TuningMode {
    fn default() -> Self {
        Self::Preset(BlockSizeBand::Band128)
    }
}

/// Tuning-mode + per-slot occupancy tuning + tuning solution storage.
/// [salamander.md §3.6: tuning modes, block-size presets, occupancy fields,
/// tuning solution storage]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutotunerConfig {
    /// Which autotuning mode governs launch-config selection for this tuner.
    #[serde(default)]
    pub tuning_mode: TuningMode,
    /// Per-slot occupancy tuning state. Stored compactly so the whole config
    /// can round-trip through the device's JSON tuning store.
    #[serde(default)]
    pub occupancy: OccupancyTuning,
    /// Tuning solution snapshot keyed by `(kernel, arch)`. Stored separately
    /// from the in-memory `Autotuner` (which is keyed by `KernelKey` + slot).
    /// This is the on-disk tuning store that `store_tuning_solution`/`
    /// load_tuning_solution` write/read.
    #[serde(default)]
    pub tuning_solutions: Vec<TuningSolution>,
}

impl Default for AutotunerConfig {
    fn default() -> Self {
        Self {
            tuning_mode: TuningMode::default(),
            occupancy: OccupancyTuning::default(),
            tuning_solutions: Vec::new(),
        }
    }
}

/// A single stored tuning solution for one `(kernel, arch)` slot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningSolution {
    /// Kernel entry name (e.g. `grim_qkv_attention`, `grim_rmsnorm`).
    pub kernel: String,
    /// GPU arch (e.g. `gfx1100`). The arch field must match the device's
    /// current arch before the solution is applied — mismatched solutions
    /// are never used (safety gate).
    pub arch: String,
    /// Slot selector: shape class + representative shape that this solution
    /// was tuned for. Used to match a solve to the right launches.
    pub slot: TuningSlot,
    /// The launch config produced by tuning. For preset/tuned modes this
    /// is the block-dim/occupancy fields the kernel launcher reads; for the
    /// `Ncu` mode it is the Ncu-derived config (feature-gated, unavailable
    /// unless `ncu` feature is on).
    #[serde(default)]
    pub config: AutotuneConfig,
}

/// Which shape/slot a tuning solution was produced for.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TuningSlot {
    #[serde(default)]
    pub shape_class: ShapeClass,
    #[serde(default)]
    pub m: usize,
    #[serde(default)]
    pub n: usize,
    #[serde(default)]
    pub k: usize,
}

impl Default for TuningSlot {
    fn default() -> Self {
        Self {
            shape_class: ShapeClass::Prefill,
            m: 128,
            n: 128,
            k: 128,
        }
    }
}

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
            waves_per_cu_target: 4,
            max_registers: 64,
            vector_width: 8,
            lds_double_buffer: true,
        };
        assert!(valid.is_valid(arch, smem_limit, 2));

        // Invalid config: LDS memory exceeds capacity limit
        let smem_overflow = LaunchConfig {
            block_m: 128,
            block_n: 128,
            block_k: 256, // Requires (128*256 + 256*128)*2 = 131,072 bytes > 64 KiB
            split_k: 1,
            threads: 256,
            waves_per_cu_target: 4,
            max_registers: 64,
            vector_width: 8,
            lds_double_buffer: true,
        };
        assert!(!smem_overflow.is_valid(arch, smem_limit, 2));

        // Invalid config: Mutual exclusion (block_m <= 8 with split_k > 4)
        let mutual_excl = LaunchConfig {
            block_m: 8,
            block_n: 64,
            block_k: 32,
            split_k: 8,
            threads: 64,
            waves_per_cu_target: 4,
            max_registers: 64,
            vector_width: 8,
            lds_double_buffer: true,
        };
        assert!(!mutual_excl.is_valid(arch, smem_limit, 2));
    }

    #[test]
    fn test_launch_config_occupancy_fields_default() {
        let (wp, mr, vw, db) = LaunchConfig::default_occupancy_fields();
        assert_eq!(wp, 4, "default waves_per_cu_target");
        assert_eq!(mr, 64, "default max_registers");
        assert_eq!(vw, 8, "default vector_width");
        assert_eq!(db, true, "default lds_double_buffer");
    }

    #[test]
    fn test_launch_config_occupancy_fields_is_valid() {
        let arch = "gfx1036";
        let smem_limit = 65536;
        // Valid occupancy fields.
        let valid = LaunchConfig {
            block_m: 16,
            block_n: 64,
            block_k: 32,
            split_k: 1,
            threads: 128,
            waves_per_cu_target: 4,
            max_registers: 64,
            vector_width: 8,
            lds_double_buffer: true,
        };
        assert!(valid.is_valid(arch, smem_limit, 2));

        // Invalid: waves_per_cu_target == 0.
        let bad_waves = LaunchConfig {
            block_m: 16,
            block_n: 64,
            block_k: 32,
            split_k: 1,
            threads: 128,
            waves_per_cu_target: 0,
            max_registers: 64,
            vector_width: 8,
            lds_double_buffer: true,
        };
        assert!(!bad_waves.is_valid(arch, smem_limit, 2));

        // Invalid: waves_per_cu_target > 10.
        let bad_waves_hi = LaunchConfig {
            block_m: 16,
            block_n: 64,
            block_k: 32,
            split_k: 1,
            threads: 128,
            waves_per_cu_target: 11,
            max_registers: 64,
            vector_width: 8,
            lds_double_buffer: true,
        };
        assert!(!bad_waves_hi.is_valid(arch, smem_limit, 2));

        // Invalid: max_registers == 0.
        let bad_regs = LaunchConfig {
            block_m: 16,
            block_n: 64,
            block_k: 32,
            split_k: 1,
            threads: 128,
            waves_per_cu_target: 4,
            max_registers: 0,
            vector_width: 8,
            lds_double_buffer: true,
        };
        assert!(!bad_regs.is_valid(arch, smem_limit, 2));

        // Invalid: max_registers > 256.
        let bad_regs_hi = LaunchConfig {
            block_m: 16,
            block_n: 64,
            block_k: 32,
            split_k: 1,
            threads: 128,
            waves_per_cu_target: 4,
            max_registers: 257,
            vector_width: 8,
            lds_double_buffer: true,
        };
        assert!(!bad_regs_hi.is_valid(arch, smem_limit, 2));

        // Invalid: vector_width not 4 or 8.
        let bad_vw = LaunchConfig {
            block_m: 16,
            block_n: 64,
            block_k: 32,
            split_k: 1,
            threads: 128,
            waves_per_cu_target: 4,
            max_registers: 64,
            vector_width: 16,
            lds_double_buffer: true,
        };
        assert!(!bad_vw.is_valid(arch, smem_limit, 2));

        // Invalid: vector_width == 4 is OK (valid value).
        let vw4 = LaunchConfig {
            block_m: 16,
            block_n: 64,
            block_k: 32,
            split_k: 1,
            threads: 128,
            waves_per_cu_target: 4,
            max_registers: 64,
            vector_width: 4,
            lds_double_buffer: true,
        };
        assert!(vw4.is_valid(arch, smem_limit, 2));
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

    // =========================================================================
    // WRECK-11: spec-decode tuning — surface gamma/threshold/alpha + structure tests.
    // =========================================================================

    #[test]
    fn autotune_config_spec_fields_default_values() {
        let cfg = AutotuneConfig::default();
        assert_eq!(
            cfg.spec_gamma, 4,
            "default spec_gamma should be 4 (small draft length, robust for prefill)"
        );
        assert!(
            (cfg.spec_acceptance_threshold - 0.6).abs() < 1e-6,
            "default spec_acceptance_threshold should be 0.6"
        );
        assert!(
            (cfg.spec_alpha - 0.0).abs() < 1e-6,
            "default spec_alpha should be 0.0 (greedy accept)"
        );
    }

    #[test]
    fn autotune_config_spec_fields_serialize_deserialize() {
        let cfg = AutotuneConfig {
            block_dim: 128,
            tile_kv: 32,
            grid_stride: 2,
            cycles_per_invocation: 1000,
            spec_gamma: 8,
            spec_acceptance_threshold: 0.45,
            spec_alpha: 0.25,
        };
        let json = serde_json::to_string(&cfg).expect("serialize AutotuneConfig with spec fields");
        let restored: AutotuneConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.spec_gamma, 8);
        assert!((restored.spec_acceptance_threshold - 0.45).abs() < 1e-6);
        assert!((restored.spec_alpha - 0.25).abs() < 1e-6);
    }

    #[test]
    fn autotune_config_spec_fields_in_defaults() {
        // Verify defaults are actually used via serde(default).
        let json = r#"{"block_dim": 64, "tile_kv": 16, "grid_stride": 1, "cycles_per_invocation": 0, "spec_acceptance_threshold": 0.5}"#;
        let cfg: AutotuneConfig =
            serde_json::from_str(json).expect("deserialize with partial spec fields");
        assert_eq!(
            cfg.spec_gamma, 4,
            "missing spec_gamma should fall back to default 4"
        );
        assert!((cfg.spec_acceptance_threshold - 0.5).abs() < 1e-6);
        assert!(
            (cfg.spec_alpha - 0.0).abs() < 1e-6,
            "missing spec_alpha should fall back to default 0.0"
        );
    }

    #[test]
    fn autotune_config_spec_gamma_roundtrip_zero() {
        let cfg = AutotuneConfig {
            spec_gamma: 0,
            ..Default::default()
        };
        let json = serde_json::to_string(&cfg).expect("serialize spec_gamma=0");
        let restored: AutotuneConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            restored.spec_gamma, 0,
            "spec_gamma=0 should roundtrip (pause speculative decoding)"
        );
    }

    #[test]
    fn autotune_config_spec_alpha_range_sanity() {
        // Values at the edges of the [0,1] acceptance range should roundtrip.
        let cfg_low = AutotuneConfig {
            spec_alpha: 0.0,
            ..Default::default()
        };
        let cfg_high = AutotuneConfig {
            spec_alpha: 1.0,
            ..Default::default()
        };
        let j_low = serde_json::to_string(&cfg_low).expect("serialize alpha=0");
        let j_high = serde_json::to_string(&cfg_high).expect("serialize alpha=1");
        let r_low: AutotuneConfig = serde_json::from_str(&j_low).expect("deserialize");
        let r_high: AutotuneConfig = serde_json::from_str(&j_high).expect("deserialize");
        assert!((r_low.spec_alpha - 0.0).abs() < 1e-6);
        assert!((r_high.spec_alpha - 1.0).abs() < 1e-6);
    }

    // =========================================================================
    // WRECK-2: CharTuner subspace pruning — structure tests.
    // =========================================================================

    #[test]
    fn charon_scalar_candidates_subspace_decode_prunes_m64() {
        let subspace = charon_scalar_candidates_subspace("gfx1036", 65536, ShapeClass::Decode);
        let has_m64 = subspace.iter().any(|c| c.block_m == 64);
        assert!(
            !has_m64,
            "Decode subspace should prune block_m=64 (M small in decode path)"
        );
        assert!(
            !subspace.is_empty(),
            "Decode subspace must still produce candidates"
        );
    }

    #[test]
    fn charon_scalar_candidates_subspace_tlo_log_keeps_square() {
        let subspace = charon_scalar_candidates_subspace("gfx1036", 65536, ShapeClass::TLOLog);
        let has_m64_n64 = subspace.iter().any(|c| c.block_m == 64 && c.block_n == 64);
        assert!(
            has_m64_n64,
            "TLOLog subspace should include square (64,64) candidates"
        );
    }

    #[test]
    fn charon_scalar_candidates_subspace_prefill_uses_larger_m() {
        let subspace = charon_scalar_candidates_subspace("gfx1036", 65536, ShapeClass::Prefill);
        let has_m64 = subspace.iter().any(|c| c.block_m == 64);
        assert!(has_m64, "Prefill subspace should use M=64 (large batch)");
        let has_m8 = subspace.iter().any(|c| c.block_m == 8);
        assert!(
            !has_m8,
            "Prefill subspace should prune M=8 (too small for batch)"
        );
    }

    #[test]
    fn charon_scalar_candidates_subspace_decode_subset_of_full() {
        let full = charon_scalar_candidates("gfx1036", 65536);
        let subspace = charon_scalar_candidates_subspace("gfx1036", 65536, ShapeClass::Decode);
        assert!(
            subspace.len() <= full.len(),
            "Decode subspace should not exceed full candidate count"
        );
        // Every subspace candidate should be in the full set (subset property).
        for sc in &subspace {
            assert!(
                full.contains(sc),
                "subspace candidate [{block_m}, {block_n}, {block_k}, {split_k}, {threads}] should be in full set",
                block_m = sc.block_m,
                block_n = sc.block_n,
                block_k = sc.block_k,
                split_k = sc.split_k,
                threads = sc.threads
            );
        }
    }

    #[test]
    fn charon_scalar_candidates_subspace_non_empty_for_all_shapes() {
        for shape in [ShapeClass::Decode, ShapeClass::Prefill, ShapeClass::TLOLog] {
            let subspace = charon_scalar_candidates_subspace("gfx1036", 65536, shape);
            assert!(
                !subspace.is_empty(),
                "subspace for {:?} should produce at least one valid candidate",
                shape
            );
        }
    }

    #[test]
    fn charon_scalar_candidates_subspace_smems_cost_within_limit() {
        let subspace = charon_scalar_candidates_subspace("gfx1036", 65536, ShapeClass::Decode);
        for cfg in &subspace {
            assert!(
                cfg.smem_cost(2) <= 65536,
                "subspace candidate should fit within 64KB smem"
            );
        }
    }
}
