//! Backend selection chain for training jobs.
//!
//! Per the architecture, the worker must run real steps on the device the
//! *user* selected, falling back through a priority order when that device
//! is unavailable:
//!
//! ```text
//! ROCm → CUDA → Vulkan → Metal → CPU
//! ```
//!
//! ROCm and CPU are always in the build (ROCm is grim's primary GPU target,
//! CPU is the ultimate reference fallback). CUDA / Vulkan / Metal are gated
//! behind the `gpu-selection` cargo feature so the SDK toolchains aren't
//! forced into builds that don't want them.
//!
//! Each backend is selected only after a genuine liveness probe
//! (`probe()` / `probe_one()`) succeeds *and* a device construct works.
//! We never silently degrade a GPU request to CPU — if ROCm is requested and
//! the ordinal is dead, we surface that and move to the next tier. CPU is the
//! single documented terminal fallback (it is always present and always works).
//!
//! Tensors are created with [`SelectedBackend::make_tensor`], which builds
//! them on the chosen `Device`. The autograd tape already dispatches through
//! `pick_device_for_tensor` (grim-autograd / grim-nn), so a device-tagged
//! tensor runs its matmul / LoRA-accumulate on that device — no CPU detour.

use std::sync::Arc;

use grim_backend_cpu::CpuDevice;
use grim_tensor::backend::BackendDevice;
use grim_tensor::dtype::{DType, QuantProvenance};
use grim_tensor::{Device, Shape, Tensor};
use serde::{Deserialize, Serialize};

/// Which backend the user asked the scheduler to prefer.
///
/// `Auto` means "use the top of the priority chain that is actually present
/// on this machine" (ROCm first, ... , CPU last). This is what the UI sends
/// when the user picks "use my GPU" without naming a vendor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreferredBackend {
    Auto,
    Rocm,
    Cuda,
    Vulkan,
    Metal,
    Cpu,
}

impl PreferredBackend {
    /// Parse the wire string the UI sends (`"rocm"`, `"cuda"`, `"vulkan"`,
    /// `"metal"`, `"cpu"`, `"auto"`). Unknown values default to `Auto`.
    pub fn from_str_opt(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "auto" | "" => PreferredBackend::Auto,
            "rocm" => PreferredBackend::Rocm,
            "cuda" => PreferredBackend::Cuda,
            "vulkan" => PreferredBackend::Vulkan,
            "metal" => PreferredBackend::Metal,
            "cpu" => PreferredBackend::Cpu,
            _ => PreferredBackend::Auto,
        }
    }
}

/// The backend actually chosen for a job, plus the device ordinal where
/// relevant. This is what the worker holds for the lifetime of the run.
#[derive(Clone)]
pub struct SelectedBackend {
    pub device: Device,
    pub label: String,
    // Keep the concrete BackendDevice alive so its stream pools / handles
    // aren't dropped mid-run. `Box<dyn BackendDevice>` is Send+Sync.
    device_impl: Arc<dyn BackendDevice>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainingRank {
    pub rank: usize,
    pub ordinal: usize,
    pub gcn_arch: String,
    pub vram_bytes: u64,
    pub weight_share: f32,
}

/// Rank-local execution context. Model providers, tapes, registries, and
/// optimizers must be owned by the rank that uses them; this type deliberately
/// carries no shared mutable training state.
#[derive(Debug, Clone)]
pub struct RankContext {
    pub rank: TrainingRank,
    pub backend: SelectedBackend,
    pub dataset_rank: usize,
    pub dataset_world_size: usize,
}

impl RankContext {
    pub fn new(
        rank: TrainingRank,
        backend: SelectedBackend,
        world_size: usize,
    ) -> Result<Self, SelectionError> {
        if rank.rank >= world_size || world_size == 0 {
            return Err(SelectionError::Tensor(format!(
                "rank {} is invalid for world size {}",
                rank.rank, world_size
            )));
        }
        Ok(Self {
            dataset_rank: rank.rank,
            dataset_world_size: world_size,
            rank,
            backend,
        })
    }

    /// Create the rank's deterministic JSONL shard with its capability-sized
    /// micro-batch. `local_batch` must come from `allocate_batch_sizes`; it is
    /// passed explicitly so independently constructed rank contexts cannot
    /// accidentally round to a batch larger than the requested global batch.
    pub fn make_dataloader(
        &self,
        path: &str,
        tokenizer: grim_format::tokenizer::GgufTokenizer,
        seq_len: usize,
        local_batch: usize,
    ) -> grim_tensor::error::Result<crate::dataloader::JsonlBatchIterator> {
        crate::dataloader::JsonlBatchIterator::new_sharded(
            path,
            tokenizer,
            seq_len,
            local_batch,
            self.dataset_rank,
            self.dataset_world_size,
        )
    }
}

/// Allocate an exact global batch across ranks using largest-remainder
/// rounding. This avoids dropping or duplicating samples when asymmetric
/// shares produce fractional micro-batches.
pub fn allocate_batch_sizes(ranks: &[TrainingRank], global_batch: usize) -> Vec<usize> {
    if ranks.is_empty() || global_batch == 0 {
        return vec![0; ranks.len()];
    }
    let mut sizes: Vec<usize> = ranks
        .iter()
        .map(|rank| (global_batch as f32 * rank.weight_share).floor() as usize)
        .collect();
    if global_batch >= ranks.len() {
        for size in &mut sizes {
            *size = (*size).max(1);
        }
    }
    let mut assigned: usize = sizes.iter().sum();
    let initial_assigned = assigned;
    let mut order: Vec<(usize, f32)> = ranks
        .iter()
        .enumerate()
        .map(|(i, rank)| (i, global_batch as f32 * rank.weight_share - sizes[i] as f32))
        .collect();
    order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    while assigned < global_batch {
        let index = order[(assigned - initial_assigned) % order.len()].0;
        sizes[index] += 1;
        assigned += 1;
    }
    while assigned > global_batch {
        if let Some(index) = order
            .iter()
            .rev()
            .map(|entry| entry.0)
            .find(|&i| sizes[i] > 0)
        {
            sizes[index] -= 1;
            assigned -= 1;
        } else {
            break;
        }
    }
    sizes
}

/// Return the exact integer batch assigned to each rank context. Keeping this
/// helper next to dataloader construction makes the invariant explicit:
/// `sum(result) == global_batch` whenever the global batch can be represented.
pub fn allocate_context_batch_sizes(contexts: &[RankContext], global_batch: usize) -> Vec<usize> {
    let ranks: Vec<TrainingRank> = contexts
        .iter()
        .map(|context| context.rank.clone())
        .collect();
    allocate_batch_sizes(&ranks, global_batch)
}

/// Execute one rank closure per OS thread and retain rank order in the
/// results. HIP/RCCL collectives require every rank to enter the collective;
/// sequentially iterating these closures would deadlock or silently reduce
/// only one participant.
pub fn run_concurrent_ranks<T, F>(jobs: Vec<F>) -> Vec<Result<T, String>>
where
    T: Send,
    F: FnOnce() -> Result<T, String> + Send,
{
    std::thread::scope(|scope| {
        let handles: Vec<_> = jobs.into_iter().map(|job| scope.spawn(job)).collect();
        handles
            .into_iter()
            .enumerate()
            .map(|(rank, handle)| match handle.join() {
                Ok(result) => result,
                Err(_) => Err(format!("rank {rank} panicked during execution")),
            })
            .collect()
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainingGpu {
    pub ordinal: usize,
    pub gcn_arch: String,
    pub vram_bytes: u64,
}

pub fn build_training_ranks(gpus: &[TrainingGpu]) -> Vec<TrainingRank> {
    let total: u128 = gpus.iter().map(|gpu| gpu.vram_bytes as u128).sum();
    let equal = 1.0 / gpus.len().max(1) as f32;
    gpus.iter()
        .enumerate()
        .map(|(rank, gpu)| TrainingRank {
            rank,
            ordinal: gpu.ordinal,
            gcn_arch: gpu.gcn_arch.clone(),
            vram_bytes: gpu.vram_bytes,
            weight_share: if total == 0 {
                equal
            } else {
                gpu.vram_bytes as f32 / total as f32
            },
        })
        .collect()
}

/// Enumerate the live ROCm devices and capture the capabilities used for
/// data-parallel scheduling.  VRAM is read after selecting each ordinal so
/// mixed cards get proportional work shares instead of assuming symmetry.
pub fn enumerate_training_gpus() -> Result<Vec<TrainingGpu>, SelectionError> {
    let devices = grim_backend_rocm::RocmDevice::probe()
        .map_err(|e| SelectionError::Tensor(format!("ROCm probe failed: {e}")))?;
    let mut gpus = Vec::with_capacity(devices.len());
    for device in devices {
        let ordinal = device.ordinal();
        let gcn_arch = grim_backend_rocm::probe_host_gpu(ordinal)
            .map(|caps| caps.gcn)
            .unwrap_or_else(|_| "unknown".into());
        let vram_bytes = grim_backend_rocm::vram_info(ordinal).1;
        gpus.push(TrainingGpu {
            ordinal,
            gcn_arch,
            vram_bytes,
        });
    }
    Ok(gpus)
}

/// Validate and construct the rank plan requested by a training job.
/// Multi-GPU training is ROCm-only because the gradient collective is RCCL.
pub fn plan_training_ranks(requested: usize) -> Result<Vec<RankContext>, SelectionError> {
    let gpus = enumerate_training_gpus()?;
    if requested == 0 {
        return Err(SelectionError::Tensor(
            "requested GPU count must be greater than zero".into(),
        ));
    }
    if requested > gpus.len() {
        return Err(SelectionError::Tensor(format!(
            "requested {requested} GPUs, but only {} ROCm device(s) are available",
            gpus.len()
        )));
    }
    let ranks = build_training_ranks(&gpus[..requested]);
    build_rank_contexts(&ranks)
}

/// Construct one rank-local backend for every planned GPU. This is intentionally
/// separate from `select_backend`, whose contract is single-device fallback.
pub fn build_rank_contexts(ranks: &[TrainingRank]) -> Result<Vec<RankContext>, SelectionError> {
    let world_size = ranks.len();
    if world_size == 0 {
        return Err(SelectionError::Tensor(
            "cannot build an empty rank plan".into(),
        ));
    }
    ranks
        .iter()
        .cloned()
        .map(|rank| {
            let backend = select_rocm_rank(rank.ordinal)?;
            RankContext::new(rank, backend, world_size)
        })
        .collect()
}

impl std::fmt::Debug for SelectedBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelectedBackend")
            .field("device", &self.device)
            .field("label", &self.label)
            .finish()
    }
}

impl SelectedBackend {
    /// Construct a tensor from host `f32` data on the *selected* device.
    ///
    /// This is the single replacement for `cpu_tensor(...)` in the worker.
    /// The returned `Tensor` carries `self.device`, so every downstream
    /// autograd op dispatches to the real backend.
    pub fn make_tensor(&self, data: Vec<f32>, shape: Shape) -> Result<Tensor, SelectionError> {
        let storage = self
            .device_impl
            .from_cpu(&data, &shape, DType::F32)
            .map_err(|e| SelectionError::Tensor(e.to_string()))?;
        Ok(Tensor::new(
            Arc::from(storage),
            shape,
            DType::F32,
            QuantProvenance::GrimNative,
            self.device.clone(),
        ))
    }

    /// Borrow the concrete device impl (for direct dispatch if ever needed).
    pub fn device_impl(&self) -> &dyn BackendDevice {
        self.device_impl.as_ref()
    }
}

/// Error type for backend selection / tensor creation.
#[derive(Debug, Clone)]
pub enum SelectionError {
    /// No backend in the chain (including CPU) could be constructed.
    NoBackend,
    /// The preferred backend was explicitly requested but unavailable.
    PreferredUnavailable(PreferredBackend),
    /// Tensor materialization onto the chosen device failed.
    Tensor(String),
}

impl std::fmt::Display for SelectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SelectionError::NoBackend => write!(f, "no compute backend available (CPU failed)"),
            SelectionError::PreferredUnavailable(p) => {
                write!(f, "preferred backend {p:?} unavailable on this host")
            }
            SelectionError::Tensor(s) => write!(f, "tensor creation failed: {s}"),
        }
    }
}

impl std::error::Error for SelectionError {}

/// A single candidate's live-or-dead status, for surfacing to the UI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendProbe {
    pub name: String,
    /// Human-readable device kind (e.g. "rocm:0", "cpu") — `Device` itself
    /// is not serde-derivable, so we serialize the label instead.
    pub device_kind: String,
    pub available: bool,
    pub detail: String,
}

/// Tier order for the selection chain. ROCm first, CPU last.
fn tier_order() -> Vec<(PreferredBackend, Device)> {
    vec![
        (PreferredBackend::Rocm, Device::Rocm(0)),
        (PreferredBackend::Cuda, Device::Cuda(0)),
        (PreferredBackend::Vulkan, Device::Vulkan),
        (PreferredBackend::Metal, Device::Metal(0)),
        (PreferredBackend::Cpu, Device::Cpu),
    ]
}

/// Probe every backend in the chain and report what is actually live on this
/// host. Drives the "select GPU" panel in the UI and lets the start handler
/// validate `preferred_backend` before dispatching a worker.
pub fn probe_all() -> Vec<BackendProbe> {
    let mut out = Vec::new();
    // ROCm (always compiled in).
    out.push(probe_rocm());
    // CUDA / Vulkan / Metal only when the feature is on; otherwise they are
    // reported absent rather than faking presence.
    #[cfg(feature = "gpu-selection")]
    {
        out.push(probe_cuda());
        out.push(probe_vulkan());
        out.push(probe_metal());
    }
    #[cfg(not(feature = "gpu-selection"))]
    {
        out.push(BackendProbe {
            name: "cuda".into(),
            device_kind: "cuda:0".into(),
            available: false,
            detail: "not compiled (enable `gpu-selection`)".into(),
        });
        out.push(BackendProbe {
            name: "vulkan".into(),
            device_kind: "vulkan".into(),
            available: false,
            detail: "not compiled (enable `gpu-selection`)".into(),
        });
        out.push(BackendProbe {
            name: "metal".into(),
            device_kind: "metal:0".into(),
            available: false,
            detail: "not compiled (enable `gpu-selection`)".into(),
        });
    }
    out.push(BackendProbe {
        name: "cpu".into(),
        device_kind: "cpu".into(),
        available: true,
        detail: "always available (reference fallback)".into(),
    });
    out
}

fn probe_rocm() -> BackendProbe {
    match grim_backend_rocm::RocmDevice::probe() {
        Ok(devs) if !devs.is_empty() => BackendProbe {
            name: "rocm".into(),
            device_kind: "rocm:0".into(),
            available: true,
            detail: format!("{} device(s) enumerated", devs.len()),
        },
        Ok(_) => BackendProbe {
            name: "rocm".into(),
            device_kind: "rocm:0".into(),
            available: false,
            detail: "no HIP devices enumerated".into(),
        },
        Err(e) => BackendProbe {
            name: "rocm".into(),
            device_kind: "rocm:0".into(),
            available: false,
            detail: format!("probe error: {e}"),
        },
    }
}

#[cfg(feature = "gpu-selection")]
fn probe_cuda() -> BackendProbe {
    match grim_backend_cuda::CudaDevice::probe() {
        Ok(devs) if !devs.is_empty() => BackendProbe {
            name: "cuda".into(),
            device_kind: "cuda:0".into(),
            available: true,
            detail: format!("{} device(s) enumerated", devs.len()),
        },
        Ok(_) => BackendProbe {
            name: "cuda".into(),
            device_kind: "cuda:0".into(),
            available: false,
            detail: "no CUDA devices enumerated".into(),
        },
        Err(e) => BackendProbe {
            name: "cuda".into(),
            device_kind: "cuda:0".into(),
            available: false,
            detail: format!("probe error: {e}"),
        },
    }
}

#[cfg(feature = "gpu-selection")]
fn probe_vulkan() -> BackendProbe {
    match grim_backend_vulkan::VulkanDevice::probe() {
        Ok(devs) if !devs.is_empty() => BackendProbe {
            name: "vulkan".into(),
            device_kind: "vulkan".into(),
            available: true,
            detail: format!("{} device(s) enumerated", devs.len()),
        },
        Ok(_) => BackendProbe {
            name: "vulkan".into(),
            device_kind: "vulkan".into(),
            available: false,
            detail: "no Vulkan devices enumerated".into(),
        },
        Err(e) => BackendProbe {
            name: "vulkan".into(),
            device_kind: "vulkan".into(),
            available: false,
            detail: format!("probe error: {e}"),
        },
    }
}

#[cfg(feature = "gpu-selection")]
fn probe_metal() -> BackendProbe {
    // Metal is Apple-only. On non-Apple targets `probe()` returns an empty
    // vec by design; we never report it as a live GPU here.
    match grim_backend_metal::MetalDevice::probe() {
        Ok(devs) if !devs.is_empty() && cfg!(target_vendor = "apple") => BackendProbe {
            name: "metal".into(),
            device_kind: "metal:0".into(),
            available: true,
            detail: format!("{} device(s) enumerated", devs.len()),
        },
        _ => BackendProbe {
            name: "metal".into(),
            device_kind: "metal:0".into(),
            available: false,
            detail: if cfg!(target_vendor = "apple") {
                "no Metal devices enumerated".into()
            } else {
                "Apple-only backend (unavailable on this platform)".into()
            },
        },
    }
}

/// Try to construct the concrete device impl for one tier.
fn try_build(pref: &PreferredBackend) -> Option<SelectedBackend> {
    match pref {
        PreferredBackend::Rocm => {
            // `RocmDevice::new` is infallible but may return a no-stream
            // fallback device on a GPU-less box; gate on a real probe so we
            // only select ROCm when a device is actually present.
            let probe = probe_rocm();
            if !probe.available {
                return None;
            }
            let dev = grim_backend_rocm::RocmDevice::new(0);
            Some(SelectedBackend {
                device: Device::Rocm(0),
                label: "rocm".into(),
                device_impl: Arc::new(dev),
            })
        }
        PreferredBackend::Cuda => {
            #[cfg(feature = "gpu-selection")]
            {
                let probe = probe_cuda();
                if !probe.available {
                    return None;
                }
                let dev = match grim_backend_cuda::CudaDevice::new(0) {
                    Ok(d) => d,
                    Err(_) => return None,
                };
                Some(SelectedBackend {
                    device: Device::Cuda(0),
                    label: "cuda".into(),
                    device_impl: Arc::new(dev),
                })
            }
            #[cfg(not(feature = "gpu-selection"))]
            {
                None
            }
        }
        PreferredBackend::Vulkan => {
            #[cfg(feature = "gpu-selection")]
            {
                let probe = probe_vulkan();
                if !probe.available {
                    return None;
                }
                let dev = grim_backend_vulkan::VulkanDevice::new();
                Some(SelectedBackend {
                    device: Device::Vulkan,
                    label: "vulkan".into(),
                    device_impl: Arc::new(dev),
                })
            }
            #[cfg(not(feature = "gpu-selection"))]
            {
                None
            }
        }
        PreferredBackend::Metal => {
            // Metal only selectable on Apple platforms; elsewhere it is never
            // a live GPU, so it falls through to the next tier.
            #[cfg(all(feature = "gpu-selection", target_vendor = "apple"))]
            {
                let probe = probe_metal();
                if !probe.available {
                    return None;
                }
                let dev = match grim_backend_metal::MetalDevice::new(0) {
                    Ok(d) => d,
                    Err(_) => return None,
                };
                Some(SelectedBackend {
                    device: Device::Metal(0),
                    label: "metal".into(),
                    device_impl: Arc::new(dev),
                })
            }
            #[cfg(not(all(feature = "gpu-selection", target_vendor = "apple")))]
            {
                None
            }
        }
        PreferredBackend::Cpu => {
            let dev = CpuDevice::new();
            Some(SelectedBackend {
                device: Device::Cpu,
                label: "cpu".into(),
                device_impl: Arc::new(dev),
            })
        }
        PreferredBackend::Auto => unreachable!("Auto is resolved before try_build"),
    }
}

/// Select the backend to run a job on.
///
/// If `preferred` is `Some(PreferredBackend::X)` and tier X is available, it
/// is chosen. Otherwise we walk the priority chain `ROCm → CUDA → Vulkan →
/// Metal → CPU` and pick the first live tier. `Auto` resolves to the top of
/// that chain.
///
/// CPU is always returned as the terminal fallback (it is always present).
pub fn select_backend(preferred: Option<PreferredBackend>) -> SelectedBackend {
    let pref = preferred.unwrap_or(PreferredBackend::Auto);

    // Explicit preference first — if the user named a backend and it is
    // live, honor it exactly.
    if pref != PreferredBackend::Auto {
        if let Some(b) = try_build(&pref) {
            return b;
        }
        // Preferred but unavailable: do NOT silently pretend it worked.
        // Drop through to the priority chain so the job still runs on the
        // next-best live device.
    }

    for (tier_pref, _dev) in tier_order() {
        if let Some(b) = try_build(&tier_pref) {
            return b;
        }
    }

    // Terminal fallback: CPU is always constructible.
    try_build(&PreferredBackend::Cpu).expect("CPU backend must always be available")
}

/// Construct a concrete ROCm backend for a validated rank ordinal.
/// `select_backend` intentionally remains the single-device UI fallback; the
/// multi-rank worker uses this explicit constructor after admission checks.
pub fn select_rocm_rank(ordinal: usize) -> Result<SelectedBackend, SelectionError> {
    let devices = grim_backend_rocm::RocmDevice::probe()
        .map_err(|e| SelectionError::Tensor(e.to_string()))?;
    if !devices.iter().any(|device| device.ordinal() == ordinal) {
        return Err(SelectionError::PreferredUnavailable(PreferredBackend::Rocm));
    }
    let device = grim_backend_rocm::RocmDevice::try_new(ordinal)
        .map_err(|e| SelectionError::Tensor(e.to_string()))?;
    Ok(SelectedBackend {
        device: Device::Rocm(ordinal),
        label: format!("rocm:{ordinal}"),
        device_impl: Arc::new(device),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_always_returns_a_backend() {
        // Even with no GPU, the chain bottoms out at CPU.
        let b = select_backend(None);
        assert!(!b.label.is_empty());
    }

    #[test]
    fn probe_reports_cpu_available() {
        let probes = probe_all();
        assert!(probes.iter().any(|p| p.name == "cpu" && p.available));
    }

    #[test]
    fn preferred_from_str_roundtrip() {
        assert_eq!(
            PreferredBackend::from_str_opt("rocm"),
            PreferredBackend::Rocm
        );
        assert_eq!(
            PreferredBackend::from_str_opt("cuda"),
            PreferredBackend::Cuda
        );
        assert_eq!(
            PreferredBackend::from_str_opt("vulkan"),
            PreferredBackend::Vulkan
        );
        assert_eq!(
            PreferredBackend::from_str_opt("metal"),
            PreferredBackend::Metal
        );
        assert_eq!(PreferredBackend::from_str_opt("cpu"), PreferredBackend::Cpu);
        assert_eq!(
            PreferredBackend::from_str_opt("bogus"),
            PreferredBackend::Auto
        );
    }

    #[test]
    fn training_rank_shares_follow_vram_and_sum_to_one() {
        let gpus = vec![
            TrainingGpu {
                ordinal: 0,
                gcn_arch: "gfx1200".into(),
                vram_bytes: 16,
            },
            TrainingGpu {
                ordinal: 1,
                gcn_arch: "gfx1200".into(),
                vram_bytes: 8,
            },
        ];
        let ranks = build_training_ranks(&gpus);
        assert_eq!(ranks[0].weight_share, 2.0 / 3.0);
        assert_eq!(ranks[1].weight_share, 1.0 / 3.0);
        assert!((ranks.iter().map(|r| r.weight_share).sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(allocate_batch_sizes(&ranks, 12), vec![8, 4]);
        assert_eq!(allocate_batch_sizes(&ranks, 1), vec![1, 0]);
    }

    #[test]
    fn context_batch_allocation_is_exact_for_asymmetric_ranks() {
        let ranks = vec![
            TrainingRank {
                rank: 0,
                ordinal: 0,
                gcn_arch: "gfx1200".into(),
                vram_bytes: 16,
                weight_share: 2.0 / 3.0,
            },
            TrainingRank {
                rank: 1,
                ordinal: 1,
                gcn_arch: "gfx1200".into(),
                vram_bytes: 8,
                weight_share: 1.0 / 3.0,
            },
        ];
        let contexts: Vec<RankContext> = ranks
            .into_iter()
            .map(|rank| {
                RankContext::new(
                    rank,
                    SelectedBackend {
                        device: Device::Cpu,
                        label: "cpu".into(),
                        device_impl: Arc::new(CpuDevice::new()),
                    },
                    2,
                )
                .unwrap()
            })
            .collect();
        assert_eq!(allocate_context_batch_sizes(&contexts, 12), vec![8, 4]);
        assert_eq!(allocate_context_batch_sizes(&contexts, 1), vec![1, 0]);
    }

    #[test]
    fn concurrent_rank_runner_preserves_rank_order() {
        let results =
            run_concurrent_ranks(vec![|| Ok::<_, String>(3usize), || Ok::<_, String>(5usize)]);
        assert_eq!(results, vec![Ok(3), Ok(5)]);
    }
}
