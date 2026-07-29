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

    /// Elementwise multiply by a scalar broadcast: `out = x * scalar`.
    ///
    /// Used by autograd (`scale_backward`, LoRA grad scaling) and by the
    /// device-resident AdamW optimizer step to scale moment buffers without a
    /// host round-trip or a full broadcast buffer. Default returns
    /// `Err(Unimplemented)` so only backends that wire a kernel override this.
    fn mul_scalar(
        &self,
        x: &dyn BackendStorage,
        scalar: f32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let _ = (x, scalar, out_shape);
        Err(crate::error::Error::Unimplemented(
            "mul_scalar not implemented for this backend".into(),
        ))
    }

    /// Elementwise square root: `out = sqrt(x)`.
    ///
    /// Used by the device-resident AdamW optimizer step to compute
    /// `sqrt(v_hat) + eps` without a host round-trip. Default returns
    /// `Err(Unimplemented)` so only backends that wire a kernel override this.
    fn sqrt(
        &self,
        x: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let _ = (x, out_shape);
        Err(crate::error::Error::Unimplemented(
            "sqrt not implemented for this backend".into(),
        ))
    }

    /// Elementwise reciprocal: `out = 1.0 / x`.
    ///
    /// Used by the device-resident AdamW optimizer step to compute
    /// `1.0 / (sqrt(v_hat) + eps)` without a host round-trip. Default returns
    /// `Err(Unimplemented)` so only backends that wire a kernel override this.
    fn recip(
        &self,
        x: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let _ = (x, out_shape);
        Err(crate::error::Error::Unimplemented(
            "recip not implemented for this backend".into(),
        ))
    }

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

    /// Copy raw byte slice (for packed quantized representations) from host memory to device storage.
    fn from_cpu_bytes(
        &self,
        data: &[u8],
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn BackendStorage>> {
        let _ = (data, shape, dtype);
        Err(crate::error::Error::Unimplemented("from_cpu_bytes not implemented for this backend".into()))
    }

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

    /// Rotary position embedding (RoPE) application on Q or K tensor.
    fn rope(
        &self,
        x: &dyn BackendStorage,
        positions: &[u32],
        dim: usize,
        base: f32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let _ = (x, positions, dim, base, out_shape);
        Err(crate::error::Error::Unimplemented(
            "rope not implemented for this backend".into(),
        ))
    }

    /// Fused GQA attention with causal masking.
    ///
    /// Phase-1 contract:
    /// - `q`:         `[seq_len, num_heads, head_dim]` (f32)
    /// - `k`, `v`:    `[kv_seq_len, num_kv_heads, head_dim]` (f32)
    /// - `num_kv_heads`: real call-site parameter (GQA ratio = num_heads / num_kv_heads)
    /// - `kv_seq_len`:  length of the K/V cache being attended to
    /// - `cache_offset`: absolute position of `q[0, *, *]` (for causal masking)
    /// - `out_shape`:  `[seq_len, num_heads, head_dim]`
    /// - `out_max`/`out_sum`: optional flash-attention-style statistics buffers
    ///
    /// Causal masking: query at absolute position `(cache_offset + i)` attends
    /// only to key positions `j` with `j <= cache_offset + i`.
    fn qkv_attention(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        out_shape: &Shape,
        out_max: Option<&dyn BackendStorage>,
        out_sum: Option<&dyn BackendStorage>,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let _ = (q, k, v, num_kv_heads, kv_seq_len, cache_offset, out_shape, out_max, out_sum);
        Err(crate::error::Error::Unimplemented(
            "qkv_attention not implemented for this backend".into(),
        ))
    }

    /// All-Reduce collective operation across tensor-parallel devices (§4.1).
    fn all_reduce(
        &self,
        inputs: &[&dyn BackendStorage],
        op: &str,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let _ = (inputs, op);
        Err(crate::error::Error::Unimplemented(
            "all_reduce not implemented for this backend".into(),
        ))
    }

    /// Mamba selective scan (Phase 2 — mambo5.md Item 11).
    ///
    /// Computes the recurrent hidden-state update `h_t = a * h_{t-1} + x_t * b_t`
    /// in parallel over the `n` (d_inner) dimension. Default returns
    /// `Err(Unimplemented)` so backends without a wired kernel are unaffected;
    /// only the ROCm backend overrides this with the real HIP launch.
    fn selective_scan(
        &self,
        _x: &dyn BackendStorage,
        _a: &dyn BackendStorage,
        _b: &dyn BackendStorage,
        _c: &dyn BackendStorage,
        _d: &dyn BackendStorage,
        _batch: usize,
        _dim_dstate: usize,
        _dim_dinner: usize,
        _seq_len: usize,
        _out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        Err(crate::error::Error::Unimplemented(
            "selective_scan requires a GPU backend with a wired HIP kernel (ROCm)".into(),
        ))
    }

    /// FlashAttention (Phase 2 — mambo5.md Item 12).
    ///
    /// Online-softmax attention with causal mask and GQA head-sharing.
    /// Default returns `Err(Unimplemented)`.
    fn flash_attention(
        &self,
        _q: &dyn BackendStorage,
        _k: &dyn BackendStorage,
        _v: &dyn BackendStorage,
        _num_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _seq_len: usize,
        _causal: bool,
        _out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        Err(crate::error::Error::Unimplemented(
            "flash_attention requires a GPU backend with a wired HIP kernel (ROCm)".into(),
        ))
    }

    /// Cross-attention for Whisper decoder (Phase 2 — mambo5.md Item 13).
    ///
    /// Encoder K/V projected once, reused across decoder steps.
    /// Default returns `Err(Unimplemented)`.
    fn cross_attention(
        &self,
        _q: &dyn BackendStorage,
        _k: &dyn BackendStorage,
        _v: &dyn BackendStorage,
        _num_heads: usize,
        _head_dim: usize,
        _seq_len: usize,
        _kv_seq_len: usize,
        _out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        Err(crate::error::Error::Unimplemented(
            "cross_attention requires a GPU backend with a wired HIP kernel (ROCm)".into(),
        ))
    }

    /// RWKV time-mix kernel (Phase 2 — mambo5.md Item 14).
    ///
    /// Recurrent linear attention with decay vector w, sigmoid gating.
    /// Default returns `Err(Unimplemented)`.
    fn rwkv_time_mix(
        &self,
        _x: &dyn BackendStorage,
        _w: &dyn BackendStorage,
        _k: &dyn BackendStorage,
        _v: &dyn BackendStorage,
        _g: &dyn BackendStorage,
        _batch: usize,
        _dim: usize,
        _seq_len: usize,
        _out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        Err(crate::error::Error::Unimplemented(
            "rwkv_time_mix requires a GPU backend with a wired HIP kernel (ROCm)".into(),
        ))
    }

    /// RWKV channel-mix kernel (Phase 2 — mambo5.md Item 14).
    ///
    /// RWKV-5/6 FFN-like gating with sigmoid.
    /// Default returns `Err(Unimplemented)`.
    fn rwkv_channel_mix(
        &self,
        _x: &dyn BackendStorage,
        _k: &dyn BackendStorage,
        _r: &dyn BackendStorage,
        _v: &dyn BackendStorage,
        _batch: usize,
        _dim: usize,
        _out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        Err(crate::error::Error::Unimplemented(
            "rwkv_channel_mix requires a GPU backend with a wired HIP kernel (ROCm)".into(),
        ))
    }

    /// Fused LoRA accumulator: `out = base + scale * (x @ A^T) @ B^T`.
    fn lora_accumulate(
        &self,
        base: &dyn BackendStorage,
        x: &dyn BackendStorage,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        scale: f32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_dims = x.shape().dims();
        let (batch, in_features) = match x_dims.len() {
            1 => (1, x_dims[0]),
            2 => (x_dims[0], x_dims[1]),
            _ => (x_dims[..x_dims.len() - 1].iter().product(), x_dims[x_dims.len() - 1]),
        };
        let (rank, in_features_a) = (a.shape().dims()[0], a.shape().dims()[1]);
        let (out_features, rank_b) = (b.shape().dims()[0], b.shape().dims()[1]);

        let owned_x_2d;
        let x_storage_2d: &dyn BackendStorage = if x_dims.len() == 2 {
            x
        } else {
            let vec_x = x.to_cpu_vec_f32()?;
            owned_x_2d = self.from_cpu(&vec_x, &Shape::new(vec![batch, in_features]), DType::F32)?;
            owned_x_2d.as_ref()
        };

        // A is [rank, in_features], A^T is [in_features, rank].
        // x @ A^T requires A_T of shape [in_features, rank].
        let vec_a = a.to_cpu_vec_f32()?;
        let mut vec_a_t = vec![0.0f32; in_features_a * rank];
        for r in 0..rank {
            for i in 0..in_features_a {
                vec_a_t[i * rank + r] = vec_a[r * in_features_a + i];
            }
        }
        let a_t_storage = self.from_cpu(&vec_a_t, &Shape::new(vec![in_features_a, rank]), DType::F32)?;

        let h_2d_shape = Shape::new(vec![batch, rank]);
        let (h_storage, h_handle) = self.matmul(x_storage_2d, a_t_storage.as_ref(), &h_2d_shape)?;
        h_handle.synchronize()?;

        // B is [out_features, rank], B^T is [rank, out_features].
        let vec_b = b.to_cpu_vec_f32()?;
        let mut vec_b_t = vec![0.0f32; rank_b * out_features];
        for o in 0..out_features {
            for r in 0..rank_b {
                vec_b_t[r * out_features + o] = vec_b[o * rank_b + r];
            }
        }
        let b_t_storage = self.from_cpu(&vec_b_t, &Shape::new(vec![rank_b, out_features]), DType::F32)?;

        let delta_2d_shape = Shape::new(vec![batch, out_features]);
        let (delta_storage, delta_handle) = self.matmul(h_storage.as_ref(), b_t_storage.as_ref(), &delta_2d_shape)?;
        delta_handle.synchronize()?;

        let scale_buf = self.from_cpu(
            &vec![scale; out_shape.elem_count()],
            &delta_2d_shape,
            DType::F32,
        )?;
        let (scaled_delta_storage, scaled_delta_handle) =
            self.mul(delta_storage.as_ref(), scale_buf.as_ref(), &delta_2d_shape)?;
        scaled_delta_handle.synchronize()?;

        let owned_base_2d;
        let base_storage_2d: &dyn BackendStorage = if base.shape().dims().len() == 2 {
            base
        } else {
            let vec_base = base.to_cpu_vec_f32()?;
            owned_base_2d = self.from_cpu(&vec_base, &delta_2d_shape, DType::F32)?;
            owned_base_2d.as_ref()
        };

        let (out_storage_2d, add_handle) = self.add(base_storage_2d, scaled_delta_storage.as_ref(), &delta_2d_shape)?;
        add_handle.synchronize()?;

        if base.shape().dims().len() == 2 {
            Ok((out_storage_2d, Box::new(ReadyHandle)))
        } else {
            let vec_out = out_storage_2d.to_cpu_vec_f32()?;
            Ok((self.from_cpu(&vec_out, out_shape, DType::F32)?, Box::new(ReadyHandle)))
        }
    }

    /// Fused dequantized matmul forward (`C = A @ B_dequant^T`).
    ///
    /// Computes matrix multiplication where `B` is a quantized tensor (Q4_K, FP8, MXFP4, MXFP8, etc.).
    /// Default implementation falls back to `matmul`.
    fn quantized_matmul(
        &self,
        _a: &dyn BackendStorage,
        _b_packed: &dyn BackendStorage,
        _b_scales: &[f32],
        _out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        Err(crate::error::Error::Unimplemented(
            "quantized_matmul requires a backend with fused dequantized matmul kernels (ROCm)".into(),
        ))
    }

    /// Fused dequantized matmul backward (WI-T3 / F5).
    ///
    /// Computes `dX[M, K] = dY[M, N] @ B^T` where `B` is dequantized on-the-fly
    /// from packed codes, per-column scale, optional outlier overrides, and
    /// optional backup1/backup2 residual layers, mirroring the forward kernel.
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
        _residuals: Option<&QuantizedMatmulBackwardResiduals>,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        Err(crate::error::Error::Unimplemented(
            "quantized_matmul_backward_dx requires ROCm (fused_dequant_backward_gemm_f16)".into(),
        ))
    }
}

/// Residual and outlier metadata for fused quantized matmul backward dispatch.
#[derive(Debug, Clone, Default)]
pub struct QuantizedMatmulBackwardResiduals {
    /// Count of index-value outlier overrides in device memory.
    pub outlier_count: usize,
    /// Raw device pointer to u32 outlier indices.
    pub outlier_indices_ptr: *const std::ffi::c_void,
    /// Raw device pointer to f32 outlier values.
    pub outlier_values_ptr: *const std::ffi::c_void,
    /// Bitwidth for backup1 residual layer (0 = absent).
    pub backup1_bpw: u8,
    /// Byte offset of packed backup1 codes.
    pub backup1_codes_offset: usize,
    /// Byte offset of per-row backup1 scale values.
    pub backup1_scale_offset: usize,
    /// Bitwidth for backup2 residual layer (0 = absent).
    pub backup2_bpw: u8,
    /// Byte offset of packed backup2 codes.
    pub backup2_codes_offset: usize,
    /// Byte offset of per-row backup2 scale values.
    pub backup2_scale_offset: usize,
}

impl QuantizedMatmulBackwardResiduals {
    /// Extract residual and outlier metadata from a tensor's `QuantProvenance`.
    ///
    /// Checks if the tensor carries `QuantProvenance::WithResiduals` and populates
    /// bitwidths and byte offsets for backup1 and backup2 residual layers, as well as
    /// outlier override counts.
    pub fn from_tensor(tensor: &crate::tensor::Tensor) -> Self {
        if let QuantProvenance::WithResiduals {
            outlier_count,
            backup1_bpw,
            backup1_codes_offset,
            backup1_scale_offset,
            backup2_bpw,
            backup2_codes_offset,
            backup2_scale_offset,
            ..
        } = tensor.provenance()
        {
            Self {
                outlier_count: *outlier_count,
                outlier_indices_ptr: std::ptr::null(),
                outlier_values_ptr: std::ptr::null(),
                backup1_bpw: *backup1_bpw,
                backup1_codes_offset: *backup1_codes_offset,
                backup1_scale_offset: *backup1_scale_offset,
                backup2_bpw: *backup2_bpw,
                backup2_codes_offset: *backup2_codes_offset,
                backup2_scale_offset: *backup2_scale_offset,
            }
        } else {
            Self::default()
        }
    }
}

/// # Safety Taxonomy — Tier 2 (Explicit raw pointers for GPU FFI dispatch)
/// `QuantizedMatmulBackwardResiduals` wraps raw GPU device memory pointers that are thread-safe to pass across worker threads for HIP kernel launch.
unsafe impl Send for QuantizedMatmulBackwardResiduals {}
/// # Safety Taxonomy — Tier 2 (Explicit raw pointers for GPU FFI dispatch)
/// `QuantizedMatmulBackwardResiduals` wraps raw GPU device memory pointers that are thread-safe to pass across worker threads for HIP kernel launch.
unsafe impl Sync for QuantizedMatmulBackwardResiduals {}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::QuantProvenance;

    #[test]
    fn test_quantized_matmul_backward_residuals_from_tensor_default() {
        let def = QuantizedMatmulBackwardResiduals::default();
        assert_eq!(def.outlier_count, 0);
        assert_eq!(def.backup1_bpw, 0);
        assert_eq!(def.backup2_bpw, 0);
        assert!(def.outlier_indices_ptr.is_null());
        assert!(def.outlier_values_ptr.is_null());
    }

    #[test]
    fn test_quantized_matmul_backward_residuals_with_residuals_provenance() {
        let prov = QuantProvenance::WithResiduals {
            outlier_count: 42,
            outlier_indices_offset: 100,
            outlier_values_offset: 200,
            backup1_bpw: 8,
            backup1_codes_offset: 1000,
            backup1_scale_offset: 2000,
            backup2_bpw: 2,
            backup2_codes_offset: 3000,
            backup2_scale_offset: 4000,
        };
        assert!(matches!(prov, QuantProvenance::WithResiduals { outlier_count: 42, .. }));
    }
}
