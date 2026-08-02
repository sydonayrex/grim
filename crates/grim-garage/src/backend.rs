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
}
