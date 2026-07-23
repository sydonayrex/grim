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

/// Unified memory-advice options matching `madvise` and `hipMemAdvise`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemAdvice {
    // OS-level hints (madvise equivalents)
    Sequential,
    Random,
    WillNeed,
    DontNeed,

    // ROCm/HIP unified-memory hints (hipMemAdvise equivalents)
    ReadMostly,
    PreferredLocation { device_id: u32 },
    AccessedBy { device_id: u32 },
    CoarseGrain,
    FineGrain,
}


/// Per-device compute primitive surface. `grim-tensor` dispatches through
/// this trait and contains no device-specific code itself. Operations
/// return both the result storage and a `ComputeHandle` that tracks the
/// operation's completion.
///
/// # Safety Taxonomy
/// Operations implemented by backends conform to the following three-tier model:
/// - **Tier 1 — Safe-by-construction**: Safe Rust code utilizing type-safety rules.
/// - **Tier 2 — Explicit `unsafe` with contract**: Backend operations that execute
///   cross-FFI boundaries (e.g. CUDA/ROCm/Vulkan API calls) requiring caller-side contracts.
/// - **Tier 3 — Raw hardware intrinsics**: Low-level instructions (e.g. LDS swizzling, inline GCN asm).
pub trait BackendDevice: Send + Sync {
    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn BackendStorage>>;

    /// 2-D `a @ b` matmul: `a` is `(M, K)`, `b` is `(K, N)`, returns `(M, N)`.
    fn matmul(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)>;

    /// 2-D matmul with explicit `solution_index` (passed through to rocBLAS).
    /// Default implementation falls back to `matmul` (solution_index = 0).
    fn matmul_with_solution(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
        solution_index: i32,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let _ = solution_index;
        self.matmul(a, b, out)
    }

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

    /// Copy a slice of F32 values from host memory to the device storage.
    fn from_cpu(
        &self,
        data: &[f32],
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn BackendStorage>>;

    /// Provide hints about memory usage/advice patterns to the device/system.
    /// Maps to OS-level `madvise` or backend-specific APIs like `hipMemAdvise`.
    fn advise(&self, storage: &dyn BackendStorage, advice: MemAdvice) -> Result<()>;

    /// Fused dequantized KV-attention (P1-WI-2).
    ///
    /// Runs online-softmax attention while dequantizing packed K/V caches
    /// on the fly. Layouts (per the `grim_kv_dequant_attention` HIP kernel):
    /// - `q`:         `[seq_len, num_heads, head_dim]` (f32)
    /// - `k_tensor`/`v_tensor`: packed K/V `[kv_seq_len, num_kv_heads, head_dim]`
    ///   (8-bit: 1 elem/byte; 4-bit: 2 elems/byte) as `unsigned char`
    /// - `k_scales`/`v_scales`: f32 per `(kv_seq_len, num_kv_heads)` row
    /// - `quant_bits`: 4 or 8
    /// - `kv_seq_len`: length of the K/V cache being attended to
    /// - `cache_offset`: absolute position of `q[head, 0, *]` (for causal mask)
    /// - `out_shape`: `[seq_len, num_heads, head_dim]`
    ///
    /// Default implementation returns `Err(Unsupported)` so backends without a
    /// wired kernel (CPU, CUDA, Vulkan, Metal) are unaffected; only the ROCm
    /// backend overrides this with the real HIP launch.
    fn kv_dequant_attention(
        &self,
        _q: &dyn BackendStorage,
        _k_tensor: &dyn BackendStorage,
        _k_scales: &dyn BackendStorage,
        _v_tensor: &dyn BackendStorage,
        _v_scales: &dyn BackendStorage,
        _num_kv_heads: usize,
        _kv_seq_len: usize,
        _cache_offset: u32,
        _quant_bits: u32,
        _out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        Err(crate::error::Error::Unimplemented(
            "kv_dequant_attention requires a GPU backend with a wired dequant-attention kernel (ROCm)".into(),
        ))
    }

    /// Fused dequantized matmul backward (WI-T3 / F5).
    ///
    /// Computes `dX[M, K] = dY[M, N] @ B^T` where `B` is dequantized on-the-fly
    /// from packed codes + per-column scale, mirroring the forward kernel.
    /// Used by `grim-autograd::matmul_backward` when the frozen-weight operand
    /// `B` is quantized and ROCm-resident. Default implementation returns
    /// `Unimplemented` so CPU/CUDA/Vulkan/Metal fall through unchanged; only
    /// the ROCm backend overrides this with the real HIP launch.
    fn quantized_matmul_backward_dx(
        &self,
        _dy: &dyn BackendStorage,
        _b_packed: &dyn BackendStorage,
        _b_scales: &[f32],
        _default_bpw: u8,
        _m: usize,
        _n: usize,
        _k: usize,
        _out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        Err(crate::error::Error::Unimplemented(
            "quantized_matmul_backward_dx requires ROCm (fused_dequant_backward_gemm_f16)".into(),
        ))
    }
}

/// Owned tensor storage on a specific backend. Backends manage their own
/// buffer lifetimes; tensors on the CPU store directly, GPU tensors wrap a
/// device pointer (ROCm/Vulkan/CUDA/Metal).
///
/// # Safety Taxonomy
/// Access to storage handles conforms to:
/// - **Tier 1 — Safe-by-construction**: Safe CPU vector conversions or metadata queries.
/// - **Tier 2 — Explicit `unsafe` with contract**: Fetching device pointers directly
///   or mapping/unmapping buffers across threads. Invariants must be documented.
/// - **Tier 3 — Raw pointer manipulations**: Raw hardware allocations and pointers.
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
