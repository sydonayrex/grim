//! Backend-agnostic trait surface and device capabilities.

use crate::dtype::{DType, QuantFormat, QuantProvenance};
use crate::error::Result;
use crate::shape::Shape;

/// Per-GPU live capability snapshot and performance metrics.
#[derive(Debug, Clone, Default)]
pub struct GpuCapability {
    /// Effective FP16 TFLOPS at this instant (may drop under throttle).
    pub tflops_fp16: f32,
    /// Effective FP8 TFLOPS — 0.0 if arch < RDNA 4 or un-measured.
    pub tflops_fp8: f32,
    /// HBM read bandwidth in GB/s.
    pub hbm_bandwidth_gbps: f32,
    /// Free VRAM in bytes at the time of the last profiler sweep.
    pub vram_free_bytes: u64,
    /// Current thermal throttle fraction (0.0 = none, 1.0 = fully throttled).
    /// Sampled from `hipDeviceGetAttribute`/SMI at the profiler cadence.
    pub throttle_pct: f32,
    /// HIP ordinal of this GPU.
    pub ordinal: usize,
}

/// GPU inter-connect link type (PeerDirect, PCIe, or Host).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScytheLink {
    /// Direct peer DMA (xGMI / Instinct class).  Maps to `RouteLink::PeerDirect`.
    PeerDirect,
    /// Peer-enabled PCIe (consumer RDNA). Maps to `RouteLink::HostBounce` for
    /// large transfers but the controller may still choose it for latency.
    Pcie,
    /// No peer access; must bounce through host pinned memory.
    Host,
}

impl Default for ScytheLink {
    /// Conservative default: assume host-bounce until a probe confirms otherwise.
    fn default() -> Self {
        ScytheLink::Host
    }
}

/// Output of the C²PLR controller for one (layer, shape) pair.
///
/// A `ScythePlacement` is produced by `C2plrController::decide()` on a cache
/// miss and then stored in `PlacementCache` (array-indexed by `layer_id`).
/// Fields follow scythe2.md §5.1 naming exactly.
#[derive(Debug, Clone)]
pub struct ScythePlacement {
    /// Which GPU ordinals participate in this layer's forward pass (vector r).
    /// May be a strict subset of all GPUs when some are off the critical path.
    pub ranks: Vec<usize>,
    /// Partition ratios p — parallel to `ranks`. Does NOT have to sum to 1.0:
    /// replicated layers (RMSNorm, RoPE) sum to K; offloaded layers sum to 1.
    pub partition: Vec<f32>,
    /// Route matrix q (flattened K×K, row-major).
    /// `routes[i * K + j]` is the link from rank i to rank j.
    pub routes: Vec<ScytheLink>,
}

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

#[derive(Debug, Clone, Copy)]
pub struct YaRNParams {
    pub factor: f32,
    pub original_max_pos: usize,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub attention_factor: f32,
}

#[derive(Debug, Clone)]
pub struct RopeConfig {
    pub dim: usize,
    pub base: f32,
    pub rotary_dim: usize,
    pub yarn: Option<YaRNParams>,
}

impl RopeConfig {
    pub fn new(dim: usize, base: f32) -> Self {
        Self {
            dim,
            base,
            rotary_dim: dim,
            yarn: None,
        }
    }

    /// Whether this config deviates from plain full-rotary RoPE. Backends that
    /// only implement the legacy path use this to return `Err(Unimplemented)`
    /// rather than silently producing wrong output.
    pub fn is_plain(&self) -> bool {
        self.rotary_dim == self.dim && self.yarn.is_none()
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

    /// Elementwise scalar addition: `out = x + scalar`.
    fn add_scalar(
        &self,
        x: &dyn BackendStorage,
        scalar: f32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let _ = (x, scalar, out_shape);
        Err(crate::error::Error::Unimplemented(
            "add_scalar not implemented for this backend".into(),
        ))
    }

    /// Elementwise scalar subtraction: `out = x - scalar`.
    fn sub_scalar(
        &self,
        x: &dyn BackendStorage,
        scalar: f32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let _ = (x, scalar, out_shape);
        Err(crate::error::Error::Unimplemented(
            "sub_scalar not implemented for this backend".into(),
        ))
    }

    /// Elementwise scalar division: `out = x / scalar`.
    fn div_scalar(
        &self,
        x: &dyn BackendStorage,
        scalar: f32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        self.mul_scalar(x, 1.0 / scalar, out_shape)
    }

    /// Fused 3-in-1 SwiGLU activation + dynamic quantization:
    /// `y = silu(gate) * up`, dynamically computes block scale, and quantizes `y` to u8 bytes.
    /// Returns `(quantized_bytes_storage, scales_storage, compute_handle)`.
    fn silu_mul_quantize(
        &self,
        gate: &dyn BackendStorage,
        up: &dyn BackendStorage,
        format: crate::dtype::QuantFormat,
        out_shape: &Shape,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        let (y_unquant, handle) = self.silu_mul(gate, up, out_shape)?;
        let q_bytes = self.quantize(y_unquant.as_ref(), format)?;
        let scale_storage = self.zeros(&Shape::from_slice(&[1]), crate::dtype::DType::F32)?;
        Ok((q_bytes, scale_storage, handle))
    }

    /// Block-Quantized SageAttention:
    /// INT8/FP8 block-scaled attention for ultra-long context windows (>128k tokens).
    fn sage_attention(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        self.qkv_attention(
            q,
            k,
            v,
            num_kv_heads,
            kv_seq_len,
            0,
            None,
            out_shape,
            None,
            None,
        )
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

    /// SwiGLU backward: `(df, de) = silu_mul_backward(gate, up, dw)`.
    fn silu_mul_backward(
        &self,
        e: &dyn BackendStorage,
        g: &dyn BackendStorage,
        dw: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        let _ = (e, g, dw, out_shape);
        Err(crate::error::Error::Unimplemented(
            "silu_mul_backward not implemented for this backend".into(),
        ))
    }

    /// RMSNorm: `y = x * rsqrt(mean(x^2) + eps) * weight`.
    fn rms_norm(
        &self,
        x: &dyn BackendStorage,
        weight: &dyn BackendStorage,
        eps: f32,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)>;

    /// In-place RMSNorm: operates directly on `x` storage when supported, avoiding extra allocation.
    /// Default implementation falls back to `rms_norm`.
    fn rms_norm_inplace(
        &self,
        x: &dyn BackendStorage,
        weight: &dyn BackendStorage,
        eps: f32,
        out: &Shape,
    ) -> Result<Box<dyn ComputeHandle>> {
        let (_storage, handle) = self.rms_norm(x, weight, eps, out)?;
        Ok(handle)
    }

    /// Fused Add + RMSNorm: `res_out = x + residual`, `y_out = rms_norm(res_out, weight, eps)`.
    /// Returns `(y_out, res_out, compute_handle)`.
    fn fused_add_rms_norm(
        &self,
        x: &dyn BackendStorage,
        residual: &dyn BackendStorage,
        weight: &dyn BackendStorage,
        eps: f32,
        out_shape: &Shape,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        let (res_out, _) = self.add(x, residual, out_shape)?;
        let (y_out, handle) = self.rms_norm(res_out.as_ref(), weight, eps, out_shape)?;
        Ok((y_out, res_out, handle))
    }

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
        Err(crate::error::Error::Unimplemented(
            "from_cpu_bytes not implemented for this backend".into(),
        ))
    }

    /// Allocate an uninitialized storage of the given shape on this backend.
    ///
    /// Used to pre-allocate the device-resident KV cache arena so decode
    /// steps can append new rows with `copy_slice_into` instead of
    /// re-uploading the whole cache through host memory. Contents are
    /// undefined until written.
    fn alloc_storage(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn BackendStorage>> {
        let _ = (shape, dtype);
        Err(crate::error::Error::Unimplemented(
            "alloc_storage not implemented for this backend".into(),
        ))
    }

    /// Device-to-device copy of `count` contiguous F32 elements from `src`
    /// into `dst` starting at flat element `dst_elem_offset`.
    ///
    /// Powers the zero-roundtrip KV cache: only the newly produced K/V rows
    /// are copied into the device-resident cache arena; the host never sees
    /// the cache contents.
    fn copy_slice_into(
        &self,
        _dst: &dyn BackendStorage,
        _src: &dyn BackendStorage,
        _dst_elem_offset: usize,
        _count: usize,
    ) -> Result<()> {
        Err(crate::error::Error::Unimplemented(
            "copy_slice_into not implemented for this backend".into(),
        ))
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
        cfg: &RopeConfig,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let _ = (x, positions, cfg, out_shape);
        Err(crate::error::Error::Unimplemented(
            "rope not implemented for this backend".into(),
        ))
    }

    /// LFM2-style fused QKV projection: MXFP4 GEMM (`x @ W_qkv`) followed by
    /// per-head QK-Norm (separate `gamma_q` / `gamma_k`) + RoPE (YaRN-aware via
    /// `inv_freq` + `mscale`). Optional on backends; returns `Unimplemented` by
    /// default so only backends that wire a kernel override this.
    fn fused_mxfp4_gemm_qk_norm_rope_kv(
        &self,
        x: &dyn BackendStorage,
        gamma_q: &dyn BackendStorage,
        gamma_k: &dyn BackendStorage,
        w_codes: &dyn BackendStorage,
        w_exps: &dyn BackendStorage,
        q_out: Option<&dyn BackendStorage>,
        k_cache: Option<&dyn BackendStorage>,
        v_cache: Option<&dyn BackendStorage>,
        out_all: Option<&dyn BackendStorage>,
        positions: Option<&dyn BackendStorage>,
        m: usize,
        k: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        rope_theta: f32,
        inv_freq: Option<&dyn BackendStorage>,
        mscale: f32,
        eps: f32,
        max_seq_len: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let _ = (
            x, gamma_q, gamma_k, w_codes, w_exps, q_out, k_cache, v_cache, out_all, positions, m,
            k, num_q_heads, num_kv_heads, head_dim, rotary_dim, rope_theta, inv_freq, mscale, eps,
            max_seq_len,
        );
        Err(crate::error::Error::Unimplemented(
            "fused_mxfp4_gemm_qk_norm_rope_kv not implemented for this backend".into(),
        ))
    }

    /// Broadcast a 1-D bias tensor `[out_dim]` into 2-D shape `[batch, out_dim]`.
    ///
    /// Contract: replicates the 1-D bias row `batch` times into `out_shape`.
    /// Used by `broadcast_bias` in `grim-nn::modules` to prevent CPU round-trips.
    fn broadcast_bias(
        &self,
        bias: &dyn BackendStorage,
        batch: usize,
        out_dim: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let _ = (bias, batch, out_dim, out_shape);
        Err(crate::error::Error::Unimplemented(
            "broadcast_bias not implemented for this backend".into(),
        ))
    }

    /// In-place scale+bias epilogue on a `[batch, out_dim]` GEMM output.
    ///
    /// Contract: `out[i,j] = out[i,j] * a_scale[i] * b_scale[j] + bias[j]`,
    /// where `a_scale` (`[batch]`, per-token) and `b_scale` (`[out_dim]`,
    /// per-channel) may be `None` (treated as 1.0) and `bias` (`[out_dim]`)
    /// may be `None` (treated as 0.0). The GEMM output is scaled in place.
    ///
    /// Plain rocBLAS exposes no epilogue-fusion API, so a standalone kernel
    /// after the GEMM is the structurally-required path. Mirror of
    /// `broadcast_bias`, but operating on existing output storage.
    fn scale_bias_epilogue(
        &self,
        out: &dyn BackendStorage,
        a_scale: Option<&dyn BackendStorage>,
        b_scale: Option<&dyn BackendStorage>,
        bias: Option<&dyn BackendStorage>,
        batch: usize,
        out_dim: usize,
    ) -> Result<Box<dyn ComputeHandle>> {
        let _ = (out, a_scale, b_scale, bias, batch, out_dim);
        Err(crate::error::Error::Unimplemented(
            "scale_bias_epilogue not implemented for this backend".into(),
        ))
    }

    /// Depthwise 1D causal convolution decode step on device.
    fn short_conv1d_causal_step(
        &self,
        x: &dyn BackendStorage,
        weight: &dyn BackendStorage,
        bias: Option<&dyn BackendStorage>,
        state: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let _ = (x, weight, bias, state, out_shape);
        Err(crate::error::Error::Unimplemented(
            "short_conv1d_causal_step not implemented for this backend".into(),
        ))
    }

    /// Fuse recurrent gating step for KDA (gated delta rule).
    fn kda_gated_delta_rule_step(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        beta: &dyn BackendStorage,
        a_gate: &dyn BackendStorage,
        recurrent_state: &dyn BackendStorage,
        d_k: usize,
        d_v: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let _ = (q, k, v, beta, a_gate, recurrent_state, d_k, d_v, out_shape);
        Err(crate::error::Error::Unimplemented(
            "kda_gated_delta_rule_step not implemented for this backend".into(),
        ))
    }

    /// MLA normalization and projection split on device.
    fn mla_q_kv_norm_split(
        &self,
        q_raw: &dyn BackendStorage,
        kv_raw: &dyn BackendStorage,
        q_norm_w: &dyn BackendStorage,
        kv_norm_w: &dyn BackendStorage,
        qk_nope_dim: usize,
        qk_rope_dim: usize,
        v_dim: usize,
        eps: f32,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        let _ = (
            q_raw,
            kv_raw,
            q_norm_w,
            kv_norm_w,
            qk_nope_dim,
            qk_rope_dim,
            v_dim,
            eps,
        );
        Err(crate::error::Error::Unimplemented(
            "mla_q_kv_norm_split not implemented for this backend".into(),
        ))
    }

    /// QKV attention calculation on device.
    ///
    /// Contracts:
    /// - `q`: `[seq_len, num_heads, head_dim]` (prefill) or `[1, num_heads, head_dim]` (decode)
    /// - `k`: `[kv_seq_len, num_kv_heads, head_dim]` contiguous KV-cache buffer
    /// - `v`: `[kv_seq_len, num_kv_heads, head_dim]` contiguous KV-cache buffer
    /// - `num_kv_heads`: real call-site parameter (GQA ratio = num_heads / num_kv_heads)
    /// - `kv_seq_len`:  length of the K/V cache being attended to
    /// - `cache_offset`: absolute position of `q[0, *, *]` (for causal masking)
    /// - `window`: optional sliding window size (e.g. `Some(512)` for Laguna-S-2.1)
    /// - `out_shape`:  `[seq_len, num_heads, head_dim]`
    /// - `out_max`/`out_sum`: optional flash-attention-style statistics buffers
    ///
    /// Causal masking: query at absolute position `(cache_offset + i)` attends
    /// only to key positions `j` with `j <= cache_offset + i` and `j >= cache_offset + i - window + 1`.
    fn qkv_attention(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        window: Option<usize>,
        out_shape: &Shape,
        out_max: Option<&dyn BackendStorage>,
        out_sum: Option<&dyn BackendStorage>,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let _ = (
            q,
            k,
            v,
            num_kv_heads,
            kv_seq_len,
            cache_offset,
            window,
            out_shape,
            out_max,
            out_sum,
        );
        Err(crate::error::Error::Unimplemented(
            "qkv_attention not implemented for this backend".into(),
        ))
    }

    /// Paged (block-table) attention for KV-cache-serving with paged memory.
    fn qkv_attention_paged(
        &self,
        q: &dyn BackendStorage,
        block_tables: &dyn BackendStorage,
        k_pages: &dyn BackendStorage,
        v_pages: &dyn BackendStorage,
        num_kv_heads: usize,
        max_blocks: usize,
        page_size: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        window: Option<usize>,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let _ = (
            q,
            block_tables,
            k_pages,
            v_pages,
            num_kv_heads,
            max_blocks,
            page_size,
            kv_seq_len,
            cache_offset,
            window,
            out_shape,
        );
        Err(crate::error::Error::Unimplemented(
            "qkv_attention_paged not implemented for this backend".into(),
        ))
    }

    /// Tree attention for Speculative Decoding (DSpark / Medusa).
    ///
    /// Dispatches a tree-attention kernel that verifies multiple draft
    /// positions against the target model's KV cache in a single kernel
    /// launch. `tree_parents` encodes the draft tree structure.
    ///
    /// Default: returns `Err(Unimplemented)`. Only backends with a tree
    /// attention kernel override this.
    fn tree_attention(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        tree_parents: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let _ = (
            q,
            k,
            v,
            tree_parents,
            num_kv_heads,
            kv_seq_len,
            cache_offset,
            out_shape,
        );
        Err(crate::error::Error::Unimplemented(
            "tree_attention not implemented for this backend".into(),
        ))
    }

    /// Begin capturing execution calls into a hardware compute graph (e.g. HIP graph).
    fn begin_graph_capture(&self, key: &str) -> Result<()> {
        let _ = key;
        Err(crate::error::Error::Unimplemented(
            "graph capture not supported on this device backend".into(),
        ))
    }

    /// End graph capture and instantiate the graph executable under `key`.
    fn end_graph_capture(&self, key: &str) -> Result<()> {
        let _ = key;
        Err(crate::error::Error::Unimplemented(
            "graph capture not supported on this device backend".into(),
        ))
    }

    /// Replay the graph captured under `key`. Returns Ok(true) if replayed, Ok(false) if missing.
    fn replay_graph(&self, key: &str) -> Result<bool> {
        let _ = key;
        Ok(false)
    }

    /// Check whether a graph executable is stored under `key`.
    fn has_captured_graph(&self, key: &str) -> bool {
        let _ = key;
        false
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

    /// SCYTHE-2 CommFuse decomposed P2P fan-in (WI-1 / WI-6).
    ///
    /// Replaces `all_reduce` for `RowParallelLinear`: instead of a
    /// `reduce_scatter` + `all_gather` pair (two sync points), each rank
    /// P2P-pushes its partial directly to the rank that owns that output shard.
    /// This eliminates the tail latency identified in CommFuse (`2604.24013`).
    ///
    /// `partials` is a slice of `(storage, placement)` pairs — one per GPU rank
    /// — where `storage` is that rank's partial GEMM output and `placement` is
    /// the controller-assigned routing metadata.
    ///
    /// Default: returns `Err(Unimplemented)` so non-ROCm backends compile
    /// unchanged. The ROCm backend overrides this in WI-6.
    fn comm_fuse_reduce(
        &self,
        partials: &[(&dyn BackendStorage, &ScythePlacement)],
    ) -> Result<Box<dyn BackendStorage>> {
        let _ = partials;
        Err(crate::error::Error::Unimplemented(
            "comm_fuse_reduce not implemented on this backend".into(),
        ))
    }

    /// WaveTune bilinear latency predictor (WI-1).
    ///
    /// Returns estimated milliseconds for a `(M, N, K)` GEMM under the given
    /// `placement` on this device. Used by `C2plrController::decide_miss()` to
    /// rank candidate placements — this is the *offline table-lookup* path
    /// described in WaveTune (`2604.10187` §4.4–4.5), not a candidate-loop.
    ///
    /// Default: returns `f64::INFINITY` so the controller treats this backend
    /// as infinitely expensive and routes away from it (safe fallback).
    fn estimate_gemm_latency_ms(
        &self,
        m: usize,
        n: usize,
        k: usize,
        dtype: DType,
        placement: &ScythePlacement,
    ) -> f64 {
        let _ = (m, n, k, dtype, placement);
        f64::INFINITY
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
            _ => (
                x_dims[..x_dims.len() - 1].iter().product(),
                x_dims[x_dims.len() - 1],
            ),
        };
        let (rank, in_features_a) = (a.shape().dims()[0], a.shape().dims()[1]);
        let (out_features, rank_b) = (b.shape().dims()[0], b.shape().dims()[1]);

        let owned_x_2d;
        let x_storage_2d: &dyn BackendStorage = if x_dims.len() == 2 {
            x
        } else {
            let vec_x = x.to_cpu_vec_f32()?;
            owned_x_2d =
                self.from_cpu(&vec_x, &Shape::new(vec![batch, in_features]), DType::F32)?;
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
        let a_t_storage =
            self.from_cpu(&vec_a_t, &Shape::new(vec![in_features_a, rank]), DType::F32)?;

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
        let b_t_storage = self.from_cpu(
            &vec_b_t,
            &Shape::new(vec![rank_b, out_features]),
            DType::F32,
        )?;

        let delta_2d_shape = Shape::new(vec![batch, out_features]);
        let (delta_storage, delta_handle) =
            self.matmul(h_storage.as_ref(), b_t_storage.as_ref(), &delta_2d_shape)?;
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

        let (out_storage_2d, add_handle) = self.add(
            base_storage_2d,
            scaled_delta_storage.as_ref(),
            &delta_2d_shape,
        )?;
        add_handle.synchronize()?;

        if base.shape().dims().len() == 2 {
            Ok((out_storage_2d, Box::new(ReadyHandle)))
        } else {
            let vec_out = out_storage_2d.to_cpu_vec_f32()?;
            Ok((
                self.from_cpu(&vec_out, out_shape, DType::F32)?,
                Box::new(ReadyHandle),
            ))
        }
    }

    /// Fused dequantized matmul forward (`C = A @ B_dequant^T`).
    ///
    /// Computes matrix multiplication where `B` is a quantized tensor with the specified `format`.
    /// Contracts and hardware dispatches are validated against `format` specifications.
    fn quantized_matmul(
        &self,
        _a: &dyn BackendStorage,
        _b_packed: &dyn BackendStorage,
        _b_scales: &[f32],
        _format: QuantFormat,
        _out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        Err(crate::error::Error::Unimplemented(
            "quantized_matmul requires a backend with fused dequantized matmul kernels".into(),
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

    /// Quantize a device-resident F32 tensor into a packed quantized representation,
    /// entirely on-device — no D2H/H2D round-trip.
    ///
    /// This is the device-side mirror of the CPU `grim_quant::quant_*` reference
    /// functions. It enables per-step activation/gradient quantization inside
    /// the training loop (e.g. for QAT or gradient compression) without
    /// stalling on a host synchronization.
    ///
    /// The returned storage holds the raw packed bytes with the appropriate
    /// `Storage` dtype (e.g. `KQuant(Q80)`, `FloatPack(Fp8)`). Backends that
    /// do not wire a device kernel return `Err(Unimplemented)` so the caller
    /// can fall back to the CPU reference.
    ///
    /// Currently supported device-side formats: `QuantFormat::Q8_0`, `QuantFormat::Fp8`.
    fn quantize(
        &self,
        _x: &dyn BackendStorage,
        _format: QuantFormat,
    ) -> Result<Box<dyn BackendStorage>> {
        Err(crate::error::Error::Unimplemented(
            "quantize not implemented for this backend".into(),
        ))
    }

    /// Fused quantize + matmul: quantize the left operand `a` on-the-fly to
    /// `format`, then compute `out = a_quant @ b`.
    ///
    /// Both `a` `(M, K)` and `b` `(K, N)` are F32 device-resident tensors. The
    /// kernel quantizes each 32-element block of `a` (per-row for Q8_0, or
    /// per-element for FP8) inline inside the GEMM, injecting quantization
    /// noise into the forward activations without a separate quantize pass or
    /// host round-trip. This mirrors the existing `quantized_matmul` /
    /// `fused_dequant_gemm` family but targets the *activations* operand rather
    /// than the frozen *weights* operand.
    ///
    /// Default returns `Err(Unimplemented)`; only backends with a wired fused
    /// kernel (CUDA, Vulkan) override this.
    fn fused_quant_gemm(
        &self,
        _a: &dyn BackendStorage,
        _b: &dyn BackendStorage,
        _format: QuantFormat,
        _out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        Err(crate::error::Error::Unimplemented(
            "fused_quant_gemm not implemented for this backend".into(),
        ))
    }
}

/// Blanket `BackendDevice` impl for `Arc<T>`.
///
/// Backends construct an `Arc`-shared singleton per ordinal (e.g.
/// `RocmDevice::shared`) and hand out cheap `Arc` clones through the
/// `Box<dyn BackendDevice>` API. Without this impl, dropping the temporary
/// `Box` would run the backend's full destructor per call — on ROCm that
/// means a device-wide `hipDeviceSynchronize`, a cache flush, and HIP module
/// unloads on *every* primitive dispatch (the per-op CPU spin/hang). With
/// `Arc`, dropping the box only decrements the refcount and the singleton
/// survives. Every method forwards to the inner device so backend-specific
/// overrides are reached through dynamic dispatch.
impl<T: BackendDevice + ?Sized> BackendDevice for std::sync::Arc<T> {
    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn BackendStorage>> {
        (**self).zeros(shape, dtype)
    }

    fn matmul(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        (**self).matmul(a, b, out)
    }

    fn matmul_with_solution(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
        solution_index: i32,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        (**self).matmul_with_solution(a, b, out, solution_index)
    }

    fn add(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        (**self).add(a, b, out)
    }

    fn mul(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        (**self).mul(a, b, out)
    }

    fn mul_scalar(
        &self,
        x: &dyn BackendStorage,
        scalar: f32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        (**self).mul_scalar(x, scalar, out_shape)
    }

    fn add_scalar(
        &self,
        x: &dyn BackendStorage,
        scalar: f32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        (**self).add_scalar(x, scalar, out_shape)
    }

    fn sub_scalar(
        &self,
        x: &dyn BackendStorage,
        scalar: f32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        (**self).sub_scalar(x, scalar, out_shape)
    }

    fn div_scalar(
        &self,
        x: &dyn BackendStorage,
        scalar: f32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        (**self).div_scalar(x, scalar, out_shape)
    }

    fn silu_mul_quantize(
        &self,
        gate: &dyn BackendStorage,
        up: &dyn BackendStorage,
        format: crate::dtype::QuantFormat,
        out_shape: &Shape,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        (**self).silu_mul_quantize(gate, up, format, out_shape)
    }

    fn sage_attention(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        (**self).sage_attention(q, k, v, num_kv_heads, kv_seq_len, out_shape)
    }

    fn sqrt(
        &self,
        x: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        (**self).sqrt(x, out_shape)
    }

    fn recip(
        &self,
        x: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        (**self).recip(x, out_shape)
    }

    fn silu_mul(
        &self,
        gate: &dyn BackendStorage,
        up: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        (**self).silu_mul(gate, up, out)
    }

    fn silu_mul_backward(
        &self,
        e: &dyn BackendStorage,
        g: &dyn BackendStorage,
        dw: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        (**self).silu_mul_backward(e, g, dw, out_shape)
    }

    fn rms_norm(
        &self,
        x: &dyn BackendStorage,
        weight: &dyn BackendStorage,
        eps: f32,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        (**self).rms_norm(x, weight, eps, out)
    }

    fn rms_norm_inplace(
        &self,
        x: &dyn BackendStorage,
        weight: &dyn BackendStorage,
        eps: f32,
        out: &Shape,
    ) -> Result<Box<dyn ComputeHandle>> {
        (**self).rms_norm_inplace(x, weight, eps, out)
    }

    fn softmax(
        &self,
        x: &dyn BackendStorage,
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        (**self).softmax(x, out)
    }

    fn embedding(
        &self,
        weight: &dyn BackendStorage,
        indices: &[u32],
        out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        (**self).embedding(weight, indices, out)
    }

    fn from_cpu(
        &self,
        data: &[f32],
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn BackendStorage>> {
        (**self).from_cpu(data, shape, dtype)
    }

    fn from_cpu_bytes(
        &self,
        data: &[u8],
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn BackendStorage>> {
        (**self).from_cpu_bytes(data, shape, dtype)
    }

    fn alloc_storage(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn BackendStorage>> {
        (**self).alloc_storage(shape, dtype)
    }

    fn copy_slice_into(
        &self,
        dst: &dyn BackendStorage,
        src: &dyn BackendStorage,
        dst_elem_offset: usize,
        count: usize,
    ) -> Result<()> {
        (**self).copy_slice_into(dst, src, dst_elem_offset, count)
    }

    fn advise(&self, storage: &dyn BackendStorage, advice: MemAdvice) -> Result<()> {
        (**self).advise(storage, advice)
    }

    fn kv_dequant_attention(
        &self,
        q: &dyn BackendStorage,
        k_tensor: &dyn BackendStorage,
        k_scales: &dyn BackendStorage,
        v_tensor: &dyn BackendStorage,
        v_scales: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        quant_bits: u32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        (**self).kv_dequant_attention(
            q,
            k_tensor,
            k_scales,
            v_tensor,
            v_scales,
            num_kv_heads,
            kv_seq_len,
            cache_offset,
            quant_bits,
            out_shape,
        )
    }

    fn rope(
        &self,
        x: &dyn BackendStorage,
        positions: &[u32],
        cfg: &RopeConfig,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        (**self).rope(x, positions, cfg, out_shape)
    }

    fn qkv_attention(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        window: Option<usize>,
        out_shape: &Shape,
        out_max: Option<&dyn BackendStorage>,
        out_sum: Option<&dyn BackendStorage>,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        (**self).qkv_attention(
            q,
            k,
            v,
            num_kv_heads,
            kv_seq_len,
            cache_offset,
            window,
            out_shape,
            out_max,
            out_sum,
        )
    }

    fn qkv_attention_paged(
        &self,
        q: &dyn BackendStorage,
        block_tables: &dyn BackendStorage,
        k_pages: &dyn BackendStorage,
        v_pages: &dyn BackendStorage,
        num_kv_heads: usize,
        max_blocks: usize,
        page_size: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        window: Option<usize>,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        (**self).qkv_attention_paged(
            q,
            block_tables,
            k_pages,
            v_pages,
            num_kv_heads,
            max_blocks,
            page_size,
            kv_seq_len,
            cache_offset,
            window,
            out_shape,
        )
    }

    fn tree_attention(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        tree_parents: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        (**self).tree_attention(
            q,
            k,
            v,
            tree_parents,
            num_kv_heads,
            kv_seq_len,
            cache_offset,
            out_shape,
        )
    }

    fn begin_graph_capture(&self, key: &str) -> Result<()> {
        (**self).begin_graph_capture(key)
    }

    fn end_graph_capture(&self, key: &str) -> Result<()> {
        (**self).end_graph_capture(key)
    }

    fn replay_graph(&self, key: &str) -> Result<bool> {
        (**self).replay_graph(key)
    }

    fn has_captured_graph(&self, key: &str) -> bool {
        (**self).has_captured_graph(key)
    }

    fn all_reduce(
        &self,
        inputs: &[&dyn BackendStorage],
        op: &str,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        (**self).all_reduce(inputs, op)
    }

    fn comm_fuse_reduce(
        &self,
        partials: &[(&dyn BackendStorage, &ScythePlacement)],
    ) -> Result<Box<dyn BackendStorage>> {
        (**self).comm_fuse_reduce(partials)
    }

    fn estimate_gemm_latency_ms(
        &self,
        m: usize,
        n: usize,
        k: usize,
        dtype: DType,
        placement: &ScythePlacement,
    ) -> f64 {
        (**self).estimate_gemm_latency_ms(m, n, k, dtype, placement)
    }

    fn selective_scan(
        &self,
        x: &dyn BackendStorage,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        c: &dyn BackendStorage,
        d: &dyn BackendStorage,
        batch: usize,
        dim_dstate: usize,
        dim_dinner: usize,
        seq_len: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        (**self).selective_scan(
            x, a, b, c, d, batch, dim_dstate, dim_dinner, seq_len, out_shape,
        )
    }

    fn flash_attention(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
        causal: bool,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        (**self).flash_attention(
            q,
            k,
            v,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
            causal,
            out_shape,
        )
    }

    fn cross_attention(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
        kv_seq_len: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        (**self).cross_attention(q, k, v, num_heads, head_dim, seq_len, kv_seq_len, out_shape)
    }

    fn rwkv_time_mix(
        &self,
        x: &dyn BackendStorage,
        w: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        g: &dyn BackendStorage,
        batch: usize,
        dim: usize,
        seq_len: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        (**self).rwkv_time_mix(x, w, k, v, g, batch, dim, seq_len, out_shape)
    }

    fn rwkv_channel_mix(
        &self,
        x: &dyn BackendStorage,
        k: &dyn BackendStorage,
        r: &dyn BackendStorage,
        v: &dyn BackendStorage,
        batch: usize,
        dim: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        (**self).rwkv_channel_mix(x, k, r, v, batch, dim, out_shape)
    }

    fn lora_accumulate(
        &self,
        base: &dyn BackendStorage,
        x: &dyn BackendStorage,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        scale: f32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        (**self).lora_accumulate(base, x, a, b, scale, out_shape)
    }

    fn quantized_matmul(
        &self,
        a: &dyn BackendStorage,
        b_packed: &dyn BackendStorage,
        b_scales: &[f32],
        format: QuantFormat,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        (**self).quantized_matmul(a, b_packed, b_scales, format, out_shape)
    }

    fn quantized_matmul_backward_dx(
        &self,
        dy: &dyn BackendStorage,
        b_packed: &dyn BackendStorage,
        b_scales: &[f32],
        default_bpw: u8,
        m: usize,
        n: usize,
        k: usize,
        out_shape: &Shape,
        residuals: Option<&QuantizedMatmulBackwardResiduals>,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        (**self).quantized_matmul_backward_dx(
            dy,
            b_packed,
            b_scales,
            default_bpw,
            m,
            n,
            k,
            out_shape,
            residuals,
        )
    }

    fn quantize(
        &self,
        x: &dyn BackendStorage,
        format: QuantFormat,
    ) -> Result<Box<dyn BackendStorage>> {
        (**self).quantize(x, format)
    }

    fn fused_quant_gemm(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        format: QuantFormat,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        (**self).fused_quant_gemm(a, b, format, out_shape)
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
    /// Extract residual and outlier metadata from a `QuantProvenance`.
    ///
    /// Checks if the tensor carries `QuantProvenance::WithResiduals` and populates
    /// bitwidths and byte offsets for backup1 and backup2 residual layers, as well as
    /// outlier override counts. Outlier index/value pointers are left null — they must
    /// be supplied by the caller from device memory when outliers are actually present.
    pub fn from_provenance(prov: &QuantProvenance) -> Self {
        if let QuantProvenance::WithResiduals {
            outlier_count,
            backup1_bpw,
            backup1_codes_offset,
            backup1_scale_offset,
            backup2_bpw,
            backup2_codes_offset,
            backup2_scale_offset,
            ..
        } = prov
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

    /// Extract residual and outlier metadata from a tensor's `QuantProvenance`.
    pub fn from_tensor(tensor: &crate::tensor::Tensor) -> Self {
        Self::from_provenance(&tensor.provenance())
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

    /// Update load-time quantization metadata before the storage is attached
    /// to a Tensor. Backends that do not need metadata may keep the default.
    fn set_provenance(&mut self, _provenance: QuantProvenance) {}
    fn shape(&self) -> &Shape;

    /// Return optional per-column/group scales slice for explicit scale formats (`ResidualPacked`/`GroupInt`).
    fn quant_scales(&self) -> Option<&[f32]> {
        None
    }

    /// Copy the buffer contents into a host `Vec<f32>`. Used for tests,
    /// token sampling, and inter-backend handoff. Production code paths
    /// should keep data on-device and avoid this when possible.
    fn to_cpu_vec_f32(&self) -> Result<Vec<f32>>;

    /// Backend-private downcast hook. Only backends that own the storage
    /// type call this — see `CpuDevice::a_storage`.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Device ordinal for this backend instance.
    ///
    /// Used by multi-GPU collective operations (RCCL all-reduce, CommFuse)
    /// to identify which GPU this backend owns. Returns `0` for single-GPU
    /// or non-RCCL backends.
    fn device_ordinal(&self) -> u32 {
        0
    }

    /// Raw GPU device pointer for this storage, if available.
    ///
    /// Returns `Some(ptr)` for GPU-resident storages (ROCm, CUDA, Vulkan, Metal)
    /// and `None` for CPU storages. Used by multi-GPU collectives (RCCL
    /// all-reduce, CommFuse) to pass device pointers directly to NCCL/HIP
    /// without an intermediate host round-trip.
    fn device_ptr(&self) -> Option<u64> {
        None
    }

    /// Request residency on the owning accelerator for managed storage.
    /// Ordinary device and CPU allocations are already resident and return
    /// success immediately.
    fn prefetch_to_device(&self) -> Result<()> {
        Ok(())
    }
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
            outlier_indices: Vec::new(),
            outlier_values_bits: Vec::new(),
            primary_scale_offset: 0,
            primary_scale_size: 0,
            primary_row_scale_dtype: 0,
            primary_scale_bytes: Vec::new(),
            backup1_bpw: 8,
            backup1_codes_offset: 1000,
            backup1_scale_offset: 2000,
            backup2_bpw: 2,
            backup2_codes_offset: 3000,
            backup2_scale_offset: 4000,
        };
        assert!(matches!(
            prov,
            QuantProvenance::WithResiduals {
                outlier_count: 42,
                ..
            }
        ));
    }

    #[test]
    fn test_quantized_matmul_backward_residuals_from_provenance_extracts_fields() {
        let prov = QuantProvenance::WithResiduals {
            outlier_count: 7,
            outlier_indices_offset: 100,
            outlier_values_offset: 200,
            outlier_indices: Vec::new(),
            outlier_values_bits: Vec::new(),
            primary_scale_offset: 0,
            primary_scale_size: 0,
            primary_row_scale_dtype: 0,
            primary_scale_bytes: Vec::new(),
            backup1_bpw: 8,
            backup1_codes_offset: 1000,
            backup1_scale_offset: 2000,
            backup2_bpw: 2,
            backup2_codes_offset: 3000,
            backup2_scale_offset: 4000,
        };
        let res = QuantizedMatmulBackwardResiduals::from_provenance(&prov);
        assert_eq!(res.outlier_count, 7);
        assert_eq!(res.backup1_bpw, 8);
        assert_eq!(res.backup1_codes_offset, 1000);
        assert_eq!(res.backup1_scale_offset, 2000);
        assert_eq!(res.backup2_bpw, 2);
        assert_eq!(res.backup2_codes_offset, 3000);
        assert_eq!(res.backup2_scale_offset, 4000);
        assert!(res.outlier_indices_ptr.is_null());
        assert!(res.outlier_values_ptr.is_null());
    }

    #[test]
    fn test_quantized_matmul_backward_residuals_from_provenance_no_residuals_default() {
        let prov = QuantProvenance::GrimNative;
        let res = QuantizedMatmulBackwardResiduals::from_provenance(&prov);
        assert_eq!(res.outlier_count, 0);
        assert_eq!(res.backup1_bpw, 0);
        assert_eq!(res.backup2_bpw, 0);
        assert!(res.outlier_indices_ptr.is_null());
        assert!(res.outlier_values_ptr.is_null());
    }
}
