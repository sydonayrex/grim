//! Backend-agnostic trait surface. Each backend crate
//! (`grim-backend-cpu`, `grim-backend-rocm`, ...) implements these.

use crate::dtype::{DType, QuantProvenance};
use crate::error::Result;
use crate::shape::Shape;

/// A handle to an asynchronous compute operation.
///
/// CPU backends resolve immediately (`synchronize` returns `Ok(())`).
/// GPU backends (ROCm, Vulkan, CUDA, Metal) back the handle with
/// stream/queue state; `synchronize` blocks until the operation
/// it tracks completes. Operations on the same device that consume
/// a buffer as input implicitly wait on any outstanding handle on
/// that buffer — callers only need to synchronize before reading
/// results back to the CPU.
pub trait ComputeHandle: Send {
    fn synchronize(&self) -> Result<()>;
    fn is_ready(&self) -> bool;
}

/// A trivially-ready handle for synchronous backends.
#[derive(Debug)]
pub struct ReadyHandle;

impl ComputeHandle for ReadyHandle {
    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
    fn is_ready(&self) -> bool {
        true
    }
}

/// Per-device compute primitive surface. `grim-tensor` dispatches through
/// this trait and contains no device-specific code itself. Operations
/// return both the result storage and a `ComputeHandle` that tracks the
/// operation's completion.
pub trait BackendDevice: Send + Sync {
    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn BackendStorage>>;

    /// 2-D `a @ b` matmul: `a` is `(M, K)`, `b` is `(K, N)`, returns `(M, N)`.
    fn matmul(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)>;

    /// Elementwise add of two equally-shaped tensors (with broadcast).
    fn add(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)>;

    /// Elementwise multiply.
    fn mul(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)>;

    /// `y = silu(x) * gate` — for LLaMA-style swiglu, fold here for now.
    fn silu_mul(
        &self,
        gate: &dyn BackendStorage,
        up: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)>;

    /// RMSNorm: `y = x * rsqrt(mean(x^2) + eps) * weight`.
    fn rms_norm(
        &self,
        x: &dyn BackendStorage,
        weight: &dyn BackendStorage,
        eps: f32,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)>;

    /// Softmax along the last dim.
    fn softmax(
        &self,
        x: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)>;

    /// Embedding gather: `out[i] = weight[indices[i], :]`.
    /// `indices` is a host-side u32 vector of the same length as the leading
    /// dim of `out`; the backend uses it to write the output storage.
    fn embedding(
        &self,
        weight: &dyn BackendStorage,
        indices: &[u32],
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)>;
}

/// Owned tensor storage on a specific backend. Backends manage their own
/// buffer lifetimes; tensors on the CPU store directly, GPU tensors wrap a
/// device pointer (ROCm/Vulkan/CUDA/Metal).
///
/// `as_any` exists so backends can downcast to their concrete storage type
/// internally without the trait leaking its existence into `grim-tensor`'s
/// public surface.
pub trait BackendStorage: Send + Sync {
    fn dtype(&self) -> DType;
    fn provenance(&self) -> QuantProvenance;
    fn shape(&self) -> &Shape;

    /// Copy the buffer contents into a host `Vec<f32>`. Used for tests,
    /// token sampling, and inter-backend handoff. Production code paths
    /// should keep data on-device and avoid this when possible.
    fn to_cpu_vec_f32(&self) -> Result<Vec<f32>>;

    /// Backend-private downcast hook. Only backends that own the storage
    /// type call this — see `CpuDevice::a_storage`.
    fn as_any(&self) -> &dyn std::any::Any;
}
