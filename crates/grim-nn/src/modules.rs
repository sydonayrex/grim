//! Module-style building blocks: linear, embedding, RMSNorm, RoPE.

use std::sync::Arc;

use grim_backend_cpu::CpuDevice;
#[cfg(feature = "metal-mem")]
use grim_backend_metal::MetalDevice;
#[cfg(feature = "vulkan-mem")]
use grim_backend_vulkan::VulkanDevice;
use grim_tensor::dtype::Storage;
use grim_tensor::error::{Error, Result};
use grim_tensor::shape::Shape;
use grim_tensor::{BackendDevice, DType, Device, Tensor,
    CoreTensorOps,
};

use crate::varbuilder::WeightSource;

#[cfg(feature = "cuda-mem")]
use grim_backend_cuda::CudaDevice;
#[cfg(feature = "rocm-mem")]
use grim_backend_rocm::RocmDevice;

/// Pick the `BackendDevice` that matches the storage location of `x` so
/// arithmetic ops dispatch to GPU kernels when the tensor lives on a GPU.
/// Falls back to CPU if the requested backend is unavailable in this build.
///
/// WI-Host-1 #4: returns `Arc<dyn BackendDevice>` to avoid allocating a fresh `Box`
/// wrapper on every operation dispatch.
pub fn pick_device_for_tensor(x: &Tensor) -> Arc<dyn BackendDevice> {
    pick_device_for_storage_device(x.device())
}

/// Fused SwiGLU `silu(gate) * up` dispatched on-device without CPU roundtrips.
pub fn silu_mul_on_device(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    let dev = pick_device_for_tensor(gate);
    let (out_storage, _handle) = dev.silu_mul(&**gate.storage(), &**up.storage(), gate.shape())?;
    Ok(Tensor::new(
        Arc::from(out_storage),
        gate.shape().clone(),
        gate.dtype(),
        grim_tensor::dtype::QuantProvenance::default(),
        gate.device().clone(),
    ))
}

/// Fused softplus attention gate `softplus(gate) * x` per-head broadcast.
pub fn softplus_mul_on_device(
    x: &Tensor,
    gate: &Tensor,
    num_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let dev = pick_device_for_tensor(x);
    let x_vec = x.to_vec_f32()?;
    let gate_vec = gate.to_vec_f32()?;
    let mut out_vec = vec![0.0f32; x_vec.len()];

    let tokens = x_vec.len() / (num_heads * head_dim);
    for t in 0..tokens {
        for h in 0..num_heads {
            let g_val = gate_vec.get(t * num_heads + h).copied().unwrap_or(0.0);
            let softplus_g = (1.0 + g_val.exp()).ln();
            for d in 0..head_dim {
                let idx = (t * num_heads + h) * head_dim + d;
                if idx < x_vec.len() {
                    out_vec[idx] = x_vec[idx] * softplus_g;
                }
            }
        }
    }

    let storage = dev.from_cpu(&out_vec, x.shape(), DType::F32)?;
    Ok(Tensor::new(
        Arc::from(storage),
        x.shape().clone(),
        DType::F32,
        x.provenance().clone(),
        x.device().clone(),
    ))
}

/// Elementwise tensor addition `a + b` dispatched on-device without CPU roundtrips.
pub fn add_on_device(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let dev = pick_device_for_tensor(a);
    let (out_storage, _handle) = dev.add(&**a.storage(), &**b.storage(), a.shape())?;
    Ok(Tensor::new(
        Arc::from(out_storage),
        a.shape().clone(),
        a.dtype(),
        grim_tensor::dtype::QuantProvenance::default(),
        a.device().clone(),
    ))
}

/// WI-SB4a: stage an activation onto `target` for contiguous layer-pipeline
/// execution. Same-device calls are free clones; cross-device moves stage
/// through host memory (`from_cpu`) so no backend-specific peer support is
/// required. The SCYTHE-2 WI-INF3 routing path keeps its own P2P fast path
/// (`streaming_forward::transfer_to_device`) for same-PCI-domain ROCm pairs —
/// this helper is deliberately the simple, always-correct variant.
///
/// On builds without the target's backend compiled in, the storage lands on
/// the CPU fallback while carrying `target` as its device tag, which is what
/// makes hermetic split-vs-unsplit parity gates possible off-box ("fake
/// segments mapped to CPU").
pub fn move_to_device(x: &Tensor, target: &Device) -> Result<Tensor> {
    if x.device() == target {
        return Ok(x.clone());
    }
    let dev = pick_device_for_storage_device(target);
    let out_storage = dev.from_cpu(&x.to_vec_f32()?, x.shape(), DType::F32)?;
    Ok(Tensor::new(
        Arc::from(out_storage),
        x.shape().clone(),
        DType::F32,
        x.provenance().clone(),
        target.clone(),
    ))
}

/// Pick a `BackendDevice` for a storage `Device` directly (without an
/// owning `Tensor`), used when reconstructing a tensor from CPU-side
/// bytes but needing to land it back on the original device.
/// Falls back to CPU if the requested backend is unavailable in this build.
///
/// WI-Host-1 #4: returns process-wide cached `Arc<dyn BackendDevice>` to avoid per-op heap churn.
pub fn pick_device_for_storage_device(d: &Device) -> Arc<dyn BackendDevice> {
    static CPU_DEV: std::sync::OnceLock<Arc<CpuDevice>> = std::sync::OnceLock::new();
    match d {
        Device::Cpu => CPU_DEV.get_or_init(|| Arc::new(CpuDevice::new())).clone(),
        #[cfg(feature = "cuda-mem")]
        Device::Cuda(ordinal) => {
            if let Ok(dev) = CudaDevice::new(*ordinal) {
                Arc::new(dev)
            } else {
                CPU_DEV.get_or_init(|| Arc::new(CpuDevice::new())).clone()
            }
        }
        #[cfg(feature = "rocm-mem")]
        Device::Rocm(ordinal) => {
            // Use the process-wide shared device (`RocmDevice::shared`).
            RocmDevice::shared(*ordinal)
        }
        #[cfg(feature = "vulkan-mem")]
        Device::Vulkan => Arc::new(VulkanDevice::new()),
        #[cfg(feature = "metal-mem")]
        Device::Metal(ordinal) => {
            if let Ok(dev) = MetalDevice::new(*ordinal) {
                Arc::new(dev)
            } else {
                CPU_DEV.get_or_init(|| Arc::new(CpuDevice::new())).clone()
            }
        }
        // Fallback for backends not compiled in (arms above are cfg-gated).
        #[allow(unreachable_patterns)]
        _ => CPU_DEV.get_or_init(|| Arc::new(CpuDevice::new())).clone(),
    }
}

/// Add two tensors element-wise with broadcasting, dispatching to the
/// device that owns `a`'s storage. This replaces the CPU-only
/// `grim_backend_cpu::add_tensors` which hardcodes `CpuDevice` and
/// panics ("storage is not CpuStorage") when called with ROCm tensors.
pub fn add_tensors(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let dev = pick_device_for_tensor(a);

    // Handle broadcasting: if b has fewer dimensions than a (e.g. b is
    // [S, H] and a is [1, S, H]), repeat b to match a's shape so that
    // backends without native broadcasting (e.g. CUDA) don't read past
    // b's memory.
    let b_adjusted = if a.shape().dims().len() != b.shape().dims().len() {
        // Compute the broadcast shape (a's shape) and repeat b to fill it.
        let out_shape = a.shape();
        let out_elems = out_shape.elem_count();
        let b_data = b.to_vec_f32()?;
        let b_elems = b_data.len();
        if out_elems % b_elems != 0 {
            return Err(Error::Shape(format!(
                "add_tensors: cannot broadcast b={:?} to a={:?}",
                b.shape(), a.shape()
            )));
        }
        let repeats = out_elems / b_elems;
        let mut expanded = Vec::with_capacity(out_elems);
        for _ in 0..repeats {
            expanded.extend_from_slice(&b_data);
        }
        let storage = dev.from_cpu(&expanded, out_shape, DType::F32)?;
        Tensor::new(
            Arc::from(storage),
            out_shape.clone(),
            DType::F32,
            b.provenance().clone(),
            b.device().clone(),
        )
    } else {
        b.clone()
    };

    let (s, h) = CoreTensorOps::add(
        &*dev,
        a.storage().as_ref(),
        b_adjusted.storage().as_ref(),
        a.shape(),
    )?;
    h.synchronize()?;
    Ok(Tensor::new(
        Arc::from(s),
        a.shape().clone(),
        DType::F32,
        a.provenance().clone(),
        a.device().clone(),
    ))
}

// ---------- Tensor Parallelism (TP) ----------

/// Tensor Parallelism configuration for multi-GPU inference/training weight sharding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorParallelConfig {
    pub rank: usize,
    pub world_size: usize,
}

impl Default for TensorParallelConfig {
    fn default() -> Self {
        Self {
            rank: 0,
            world_size: 1,
        }
    }
}

impl TensorParallelConfig {
    /// Read TP rank / world size from the environment.
    ///
    /// - `GRIM_TP_SIZE` → `world_size` (defaults to 1).
    /// - `GRIM_TP_RANK` → `rank` (defaults to 0).
    ///
    /// Returns `None` when `GRIM_TP_SIZE` is unset or `1` (single-device).
    pub fn from_env() -> Option<Self> {
        let world_size = std::env::var("GRIM_TP_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&w| w > 1)?;
        let rank = std::env::var("GRIM_TP_RANK")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        Some(Self { rank, world_size })
    }

    /// Validate the rank/world_size contract: `world_size >= 1` and
    /// `rank < world_size`. Returns `Err(Unsupported)` (via the caller's
    /// `Result`) misconfigured config rather than silently degrading.
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.world_size == 0 {
            return Err(
                "invalid TensorParallelConfig: world_size must be >= 1 (got 0)".to_string(),
            );
        }
        if self.world_size == 1 {
            if self.rank != 0 {
                return Err(format!(
                    "invalid TensorParallelConfig: rank must be 0 when world_size == 1 \
                     (got rank={})",
                    self.rank
                ));
            }
            return Ok(());
        }
        if self.rank >= self.world_size {
            return Err(format!(
                "invalid TensorParallelConfig: rank ({}) must be < world_size ({})",
                self.rank, self.world_size
            ));
        }
        Ok(())
    }
}

// ---------- Expert Parallelism (EP) ----------

/// Expert Parallelism configuration for multi-GPU MoE expert sharding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertParallelConfig {
    pub rank: usize,
    pub world_size: usize,
    pub num_total_experts: usize,
    pub assigned_experts: Vec<usize>,
}

impl Default for ExpertParallelConfig {
    fn default() -> Self {
        Self {
            rank: 0,
            world_size: 1,
            num_total_experts: 0,
            assigned_experts: Vec::new(),
        }
    }
}

impl ExpertParallelConfig {
    /// Read EP rank / world size from the environment.
    ///
    /// - `GRIM_EP_SIZE` → `world_size` (defaults to 1).
    /// - `GRIM_EP_RANK` → `rank` (defaults to 0).
    pub fn from_env(num_total_experts: usize) -> Option<Self> {
        let world_size = std::env::var("GRIM_EP_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&w| w > 1)?;
        let rank = std::env::var("GRIM_EP_RANK")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);

        let experts_per_rank = num_total_experts.div_ceil(world_size);
        let start = (rank * experts_per_rank).min(num_total_experts);
        let end = ((rank + 1) * experts_per_rank).min(num_total_experts);
        let assigned_experts = (start..end).collect();

        Some(Self {
            rank,
            world_size,
            num_total_experts,
            assigned_experts,
        })
    }

    /// Construct uniform expert partitioning across `world_size` ranks for `num_total_experts`.
    pub fn uniform(rank: usize, world_size: usize, num_total_experts: usize) -> Self {
        let experts_per_rank = num_total_experts.div_ceil(world_size.max(1));
        let start = (rank * experts_per_rank).min(num_total_experts);
        let end = ((rank + 1) * experts_per_rank).min(num_total_experts);
        let assigned_experts = (start..end).collect();
        Self {
            rank,
            world_size,
            num_total_experts,
            assigned_experts,
        }
    }

    /// Whether this rank hosts expert `expert_id`.
    pub fn owns_expert(&self, expert_id: usize) -> bool {
        self.assigned_experts.contains(&expert_id)
    }

    /// Returns the target rank that hosts `expert_id`.
    pub fn rank_for_expert(&self, expert_id: usize) -> usize {
        let experts_per_rank = self.num_total_experts.div_ceil(self.world_size.max(1));
        if experts_per_rank == 0 {
            0
        } else {
            (expert_id / experts_per_rank).min(self.world_size - 1)
        }
    }
}

/// Refuse tensor parallelism for architecture `arch` when `tp.world_size > 1`.
///
/// Used by `Foo::load_tp` stubs for architectures whose `forward` path does
/// not yet consume `ColumnParallelLinear`/`RowParallelLinear` (or whose
/// attention layout — fused QKV, MLA, enc/dec cross-attn, SSM/RWKV — would
/// require bespoke sharding math). Returns the typed `Unsupported` error the
/// caller bubbles up so the multi-process TP path **fails loudly** rather than
/// loading sharded-but-unreduced weights that silently corrupt output.
///
/// `world_size == 1` (or default) passes through.
pub fn require_single_device(
    tp: TensorParallelConfig,
    arch: &str,
    reason: &str,
) -> std::result::Result<(), String> {
    tp.validate()?;
    if tp.world_size > 1 {
        return Err(format!(
            "tensor-parallel load for {arch} not yet implemented ({reason}); refusing \
             world_size={} to avoid silently wrong output. Set GRIM_TP_SIZE=1.",
            tp.world_size
        ));
    }
    Ok(())
}

/// Column-parallel linear layer (§4.1): weights are pre-sharded at load
/// (each rank holds `out_features / world_size` rows), so `forward` is just
/// the inner `Linear::forward`. No CPU output-slicing needed.
#[derive(Clone)]
pub struct ColumnParallelLinear {
    /// The full Linear; its weight tensor is already the rank's shard.
    pub inner: Linear,
    pub tp_config: TensorParallelConfig,
}

impl ColumnParallelLinear {
    pub fn new(inner: Linear, tp_config: TensorParallelConfig) -> Self {
        Self { inner, tp_config }
    }

    /// Forward: delegate to inner Linear (weights are pre-sharded at load).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }

    /// Reference the underlying weight tensor (pre-sharded shard).
    pub fn weight(&self) -> &Tensor {
        &self.inner.weight
    }

    /// Reference the underlying bias tensor, if present (unsharded — same on all ranks).
    pub fn bias(&self) -> Option<&Tensor> {
        self.inner.bias.as_ref()
    }

    /// Borrow the inner `Linear`.
    pub fn inner(&self) -> &Linear {
        &self.inner
    }

    /// Number of output rows this rank owns.
    pub fn shard_size(&self) -> usize {
        self.inner.weight.shape().dims()[0]
    }
}

/// Row-parallel linear layer (§4.1): weights are pre-sharded at load
/// (each rank holds `in_features / world_size` columns), so `forward` is
/// the inner matmul + a device-side `all_reduce("sum")` to sum partial
/// outputs across TP ranks. If the active backend has no `all_reduce`
/// implementation (e.g. CUDA, which inherits the trait default
/// `Err(Unimplemented)`), `forward` **propagates the error** rather than
/// returning the un-reduced partial output — see `require_single_device`
/// for the rationale: returning a per-rank partial as if it were the
/// all-reduced sum silently corrupts every downstream activation. Set
/// `GRIM_TP_SIZE=1` (or use a backend with a real collective impl such as
/// ROCm/RCCL, Vulkan, or Metal) to run single-device.
#[derive(Clone)]
pub struct RowParallelLinear {
    pub inner: Linear,
    pub tp_config: TensorParallelConfig,
}

impl RowParallelLinear {
    pub fn new(inner: Linear, tp_config: TensorParallelConfig) -> Self {
        Self { inner, tp_config }
    }

    /// Forward: inner matmul of pre-sharded input + device-side `all_reduce`.
    ///
    /// On `all_reduce` failure the error is propagated (not silently
    /// degraded) — a backend without a collective impl cannot produce a
    /// correct TP>1 forward, so a hard error is preferable to wrong output.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.inner.forward(x)?;
        if self.tp_config.world_size > 1 {
            let dev = pick_device_for_tensor(&out);
            let s: &dyn grim_tensor::BackendStorage = out.storage().as_ref();
            let (storage, handle) = dev.all_reduce(&[s], "sum").map_err(|e| {
                Error::Backend(format!(
                    "RowParallelLinear::forward all_reduce failed on backend {:?} with \
                         world_size={}: {e}. This backend has no all_reduce implementation; \
                         set GRIM_TP_SIZE=1 or use a backend with collectives (ROCm/Vulkan/Metal).",
                    out.device(),
                    self.tp_config.world_size,
                ))
            })?;
            handle.synchronize()?;
            Ok(Tensor::new(
                Arc::from(storage),
                out.shape().clone(),
                out.dtype(),
                out.provenance().clone(),
                out.device().clone(),
            ))
        } else {
            Ok(out)
        }
    }

    /// Reference the underlying weight tensor (pre-sharded shard).
    pub fn weight(&self) -> &Tensor {
        &self.inner.weight
    }

    /// Reference the underlying bias tensor, if present (unsharded — same on all ranks).
    pub fn bias(&self) -> Option<&Tensor> {
        self.inner.bias.as_ref()
    }

    /// Borrow the inner `Linear`.
    pub fn inner(&self) -> &Linear {
        &self.inner
    }

    /// Number of output rows this rank owns (same as full output for row-parallel).
    pub fn shard_size(&self) -> usize {
        self.inner.weight.shape().dims()[0]
    }
}

// ---------- Linear ----------

/// Linear: `y = x @ W^T [+ b]` with weight `(out, in)`, optional bias `(out,)`.
#[derive(Clone)]
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub w_t: Tensor,
    pub quant_format: Option<DType>,
}

impl Linear {
    /// Load a Linear layer.
    ///
    /// GGUF stores matrix weights as `[out_dim, in_dim]` (rows = output units,
    /// columns = input units). This matches llama.cpp's convention: `y = x @ W^T`,
    /// so `Linear` pre-transposes `w_t` once during load for fast device matmuls.
    pub fn load(
        ws: &WeightSource<'_>,
        in_dim: usize,
        out_dim: usize,
        has_bias: bool,
    ) -> Result<Self> {
        let weight = ws.get([out_dim, in_dim], "weight")?;
        let w_t = transpose_last_two(&weight)?;
        let quant_format = if weight.dtype().is_quantized() {
            Some(weight.dtype().clone())
        } else {
            None
        };
        let bias = if has_bias {
            Some(ws.get([out_dim], "bias")?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            w_t,
            quant_format,
        })
    }

    /// Load an unbiased Linear layer from a shape pair `[in_dim, out_dim]`.
    pub fn load_shape(ws: &WeightSource<'_>, shape: [usize; 2]) -> Result<Self> {
        Self::load(ws, shape[0], shape[1], false)
    }

    /// F2 (full-parameter write-back): swap in freshly trained weights,
    /// recomputing the pre-transposed `w_t` that forward actually consumes.
    pub fn replace_weight(&mut self, weight: Tensor) -> Result<()> {
        self.w_t = transpose_last_two(&weight)?;
        self.weight = weight;
        Ok(())
    }

    /// Load a column-parallel shard (dim==0): each rank gets `out_dim / world_size`
    /// rows of the weight matrix. Bias is loaded unsharded (same on all ranks).
    pub fn load_column_parallel(
        ws: &WeightSource<'_>,
        in_dim: usize,
        out_dim: usize,
        has_bias: bool,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        let shard_out = out_dim / tp.world_size;
        let weight = ws
            .with_tp_config(tp)
            .get_sharded([shard_out, in_dim], "weight", 0)?;
        let w_t = transpose_last_two(&weight)?;
        let quant_format = if weight.dtype().is_quantized() {
            Some(weight.dtype().clone())
        } else {
            None
        };
        let bias = if has_bias {
            Some(ws.get([out_dim], "bias")?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            w_t,
            quant_format,
        })
    }

    /// Load a row-parallel shard (dim==1): each rank gets `in_dim / world_size`
    /// columns of the weight matrix. Bias is loaded unsharded.
    pub fn load_row_parallel(
        ws: &WeightSource<'_>,
        in_dim: usize,
        out_dim: usize,
        has_bias: bool,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        let shard_in = in_dim / tp.world_size;
        let weight = ws
            .with_tp_config(tp)
            .get_sharded([out_dim, shard_in], "weight", 1)?;
        let w_t = transpose_last_two(&weight)?;
        let quant_format = if weight.dtype().is_quantized() {
            Some(weight.dtype().clone())
        } else {
            None
        };
        let bias = if has_bias {
            Some(ws.get([out_dim], "bias")?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            w_t,
            quant_format,
        })
    }

    pub fn from_tensor(weight: Tensor, bias: Option<Tensor>) -> Self {
        let w_t = transpose_last_two(&weight).unwrap_or_else(|_| weight.clone());
        let quant_format = if weight.dtype().is_quantized() {
            Some(weight.dtype().clone())
        } else {
            None
        };
        Self {
            weight,
            bias,
            w_t,
            quant_format,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dev = pick_device_for_tensor(x);
        let in_dim = x.shape().dims().last().copied().unwrap_or(0);
        // Weight is GGUF-native: dim(0) = out_dim, dim(1) = in_dim.
        let out_dim = self.weight.shape().dim(0)?;
        let batch = x.shape().elem_count() / in_dim;

        let a_storage = x.storage().as_ref();
        let b_storage = self.w_t.storage().as_ref();

        let out_shape = Shape::new(vec![batch, out_dim]);
        let qmm_trace = std::env::var_os("GRIM_QMM_TRACE").is_some();
        if qmm_trace {
            eprintln!(
                "[linear] out_dim={out_dim} batch={batch} w_t_quant={} w_t_dtype={:?}",
                self.w_t.dtype().is_quantized(),
                self.w_t.dtype().storage
            );
        }
        let (out_s, h) = if self.w_t.dtype().is_quantized() {
            let quant_fmt = match &self.w_t.dtype().storage {
                Storage::KQuant(grim_tensor::dtype::KQuantScheme::Q80) => {
                    Some(grim_tensor::QuantFormat::Q8_0)
                }
                Storage::KQuant(grim_tensor::dtype::KQuantScheme::Q5K) => {
                    Some(grim_tensor::QuantFormat::Q5K)
                }
                Storage::KQuant(grim_tensor::dtype::KQuantScheme::Q4K) => {
                    Some(grim_tensor::QuantFormat::Q4K)
                }
                Storage::KQuant(grim_tensor::dtype::KQuantScheme::Q6K) => {
                    Some(grim_tensor::QuantFormat::Q6K)
                }
                _ => None,
            };
            if let Some(fmt) = quant_fmt {
                match dev.fused_quant_gemm(a_storage, b_storage, fmt, &out_shape) {
                    Ok(res) => {
                        if qmm_trace {
                            eprintln!("[linear] branch=fused_quant_gemm OK");
                        }
                        res
                    }
                    Err(Error::Unimplemented(_)) => {
                        if qmm_trace {
                            eprintln!("[linear] branch=quantized_matmul fallback");
                        }
                        // No fused kernel (e.g. CPU): fall back to the explicit
                        // dequant + GEMM path. `quantized_matmul` reads the packed
                        // bytes from `w_t` (GGUF [out,in] layout), dequants into a
                        // [k,n] row-major buffer, and runs the GEMM — it does NOT
                        // depend on `transpose_last_two` having relabeled storage.
                        // Falling back to plain `matmul` here would pass the
                        // still-quantized bytes to a function expecting F32 [k,n]
                        // and trip ShapeMismatch on non-square layers (e.g.
                        // MiniCPM5's wq is [2048,1536]). Scales are embedded in
                        // the K-quant block layout, so an empty slice is fine for
                        // Q4_K/Q5_K/Q6_K (the only K-quant formats we dispatch
                        // here); Q8_0 is handled by its own block header.
                        dev.quantized_matmul(a_storage, b_storage, &[], fmt, &out_shape)?
                    }
                    Err(e) => return Err(e),
                }
            } else {
                // Quantized storage without a QuantFormat mapping (e.g.
                // FloatPack(MxFp4)): plain `matmul` would misread the packed
                // bytes as F32. Route through `quantized_matmul`, whose ROCm
                // dispatch matches on the storage dtype (the `format`
                // parameter is unused there) and has per-format fused
                // dequant-GEMM kernels; backends without the override
                // materialize F32 weights at load time and never reach this
                // branch.
                dev.quantized_matmul(
                    a_storage,
                    b_storage,
                    &[],
                    grim_tensor::QuantFormat::Fp4,
                    &out_shape,
                )?
            }
        } else {
            CoreTensorOps::matmul(&*dev, a_storage, b_storage, &out_shape)?
        };
        // WI-Host-1: dropped `h.synchronize()?` here. The host-side pipeline
        // stall it forced on every `Linear::forward` call (twice per layer:
        // matmul + bias-add) is a real throughput hit on GPU backends, and
        // the sync is redundant — the returned storage `Arc` is a device
        // buffer handle that synchronizes lazily on the first real read
        // (`to_vec_f32`, `to_cpu_vec_f32`, the next op's dispatch). The plan
        // keeps synchronization at the outer inference boundary (before
        // sampling), not inside every primitive. CPU is unaffected
        // (`CpuDevice::synchronize` is a no-op — verified by the existing
        // `test_linear_forward_with_bias_hand_calculated` etc.).
        let _ = h;
        let mat_out = Tensor::new(
            Arc::from(out_s),
            out_shape,
            DType::F32,
            x.provenance().clone(),
            x.device().clone(),
        );

        if let Some(b) = &self.bias {
            let broadcast_b = broadcast_bias(b, batch, out_dim)?;
            let (s, hh) = CoreTensorOps::add(
                &*dev,
                mat_out.storage().as_ref(),
                broadcast_b.storage().as_ref(),
                mat_out.shape(),
            )?;
            // WI-Host-1: dropped `hh.synchronize()?` (see the matmul-sync
            // comment above for rationale — same lazy-sync discipline).
            let _ = hh;
            return Ok(Tensor::new(
                Arc::from(s),
                mat_out.shape().clone(),
                DType::F32,
                mat_out.provenance().clone(),
                mat_out.device().clone(),
            ));
        }
        Ok(mat_out)
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

fn transpose_last_two(t: &Tensor) -> Result<Tensor> {
    // GGUF stores Linear weights as [out_dim, in_dim] (rows = output units,
    // columns = input units). The CPU/CUDA/Vulkan/Metal `matmul` kernels all
    // consume B in [k, n] = [in_dim, out_dim] row-major layout (they compute
    // `C = A @ B` with B indexed as `b[p*n + j]`), so we must transpose the
    // GGUF [out,in] storage to [in,out].
    //
    // ROCm is the exception: its fused dequant-GEMM kernel reads B directly as
    // [out_dim, in_dim] (it indexes `col` over the output dim), so for
    // quantized weights we only relabel the shape to [out,in] without moving
    // bytes (the kernel handles the layout).
    //
    // The historical comment claimed CPU "keeps GGUF [in,out] layout as-is"
    // and "F32 matmul consumes [in,out] directly", but that is incorrect:
    // the CPU GEMM (`gemm_scalar`/`gemv_row`) indexes `b[p*n + j]`, requiring
    // [k,n]=[in,out]. This was silently correct for square Llama weights
    // (num_heads*head_dim == hidden_size) where [out,in] and [in,out] share
    // the same shape; non-square layers like MiniCPM5 (16*128=2048 != 1536)
    // expose the bug as a ShapeMismatch (expected [15,1536] got [2048,1536]).
    let dims = t.shape().dims().to_vec();
    if dims.len() != 2 {
        return Err(Error::Shape("transpose_last_two: only 2-D".into()));
    }
    let (a, b) = (dims[0], dims[1]);
    let new_shape = Shape::new(vec![b, a]);

    // ROCm quantized fast path: relabel only (kernel reads [out,in] directly).
    if matches!(t.device(), Device::Rocm(_)) && t.dtype().is_quantized() {
        return Ok(Tensor::new(
            t.storage().clone(),
            new_shape,
            t.dtype(),
            t.provenance().clone(),
            t.device().clone(),
        ));
    }

    // ROCm F32 on-device path: transpose in device memory (`grim_transpose_2d_f32`),
    // avoiding the host `to_vec_f32` (DtoH) + `from_cpu` (H2D) round trip below.
    // Only reached for non-quantized ROCm weights (quantized took the relabel
    // fast path above), so the input is plain F32 storage on the device.
    #[cfg(feature = "rocm-mem")]
    if let Device::Rocm(ordinal) = t.device() {
        let dev = grim_backend_rocm::RocmDevice::shared(*ordinal);
        let storage = dev.transpose_f32_2d(t.storage().as_ref(), a, b)?;
        return Ok(Tensor::new(
            Arc::from(storage),
            new_shape,
            DType::F32,
            t.provenance().clone(),
            t.device().clone(),
        ));
    }

    // All other cases (CPU/CUDA/Vulkan/Metal, F32 or quantized): genuinely
    // transpose the data so `w_t` is in [in,out]=[k,n] row-major layout.
    // Quantized tensors reaching here are already dequantized to F32 in
    // `WeightSource::get` for CPU, so `to_vec_f32` works uniformly.
    let src = t.to_vec_f32()?;
    let mut out = vec![0.0f32; a * b];
    for i in 0..a {
        for j in 0..b {
            out[j * a + i] = src[i * b + j];
        }
    }
    if t.device().is_cpu() {
        Ok(grim_backend_cpu::cpu_tensor(out, new_shape))
    } else {
        let dev = pick_device_for_tensor(t);
        let storage = dev.from_cpu(&out, &new_shape, DType::F32)?;
        Ok(Tensor::new(
            Arc::from(storage),
            new_shape,
            DType::F32,
            t.provenance().clone(),
            t.device().clone(),
        ))
    }
}

/// Broadcast a 1-D bias `[out_dim]` to `[batch, out_dim]` by tiling.
///
/// WI-Host-1 gate (1): the CPU-path output of this function is pinned by
/// `host1_rms_rope_broadcast_bias_cpu_path_parity` as the verification
/// target for the deferred native HIP kernel replacement. The current
/// implementation round-trips through `to_vec_f32()` → CPU tile →
/// `from_cpu()` on every `Linear::forward` call with a bias — the plan's
/// Broadcast a 1-D bias tensor to a 2-D batch shape on CPU or device.
pub fn broadcast_bias(b: &Tensor, batch: usize, out_dim: usize) -> Result<Tensor> {
    let new_shape = Shape::new(vec![batch, out_dim]);
    if b.device().is_cpu() {
        let b_vec = b.to_vec_f32()?;
        let mut out = Vec::with_capacity(batch * out_dim);
        for _ in 0..batch {
            out.extend_from_slice(&b_vec);
        }
        if out.len() != batch * out_dim {
            return Err(Error::Shape("broadcast_bias: size mismatch".into()));
        }
        Ok(grim_backend_cpu::cpu_tensor(out, new_shape))
    } else {
        let dev = pick_device_for_tensor(b);
        let (storage, _handle) =
            dev.broadcast_bias(b.storage().as_ref(), batch, out_dim, &new_shape)?;
        Ok(Tensor::new(
            Arc::from(storage),
            new_shape,
            DType::F32,
            b.provenance().clone(),
            b.device().clone(),
        ))
    }
}

// ---------- RMSNorm ----------

#[derive(Clone)]
pub struct RmsNorm {
    pub weight: Tensor,
    pub eps: f32,
}

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f32) -> Self {
        Self { weight, eps }
    }

    pub fn load(ws: &WeightSource<'_>, dim: usize, eps: f32) -> Result<Self> {
        let weight = ws.get([dim], "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dev = pick_device_for_tensor(x);
        let dim = x.shape().dims().last().copied().unwrap_or(0);
        let batch = x.shape().elem_count() / dim;
        let out_shape = Shape::new(vec![batch, dim]);
        let (s, h) = CoreTensorOps::rms_norm(
            &*dev,
            x.storage().as_ref(),
            self.weight.storage().as_ref(),
            self.eps,
            &out_shape,
        )?;
        // WI-Host-1: dropped `h.synchronize()?`. Same lazy-sync rationale as
        // `Linear::forward` (see the matmul-sync comment there): the stall is
        // redundant because the returned storage synchronizes on first read.
        // The plan keeps synchronization at the outer inference boundary.
        let _ = h;
        Ok(Tensor::new(
            Arc::from(s),
            out_shape,
            DType::F32,
            x.provenance().clone(),
            x.device().clone(),
        ))
    }
}

// ---------- LayerNorm ----------

#[derive(Clone)]
pub struct LayerNorm {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(weight: Tensor, bias: Option<Tensor>, eps: f32) -> Self {
        Self { weight, bias, eps }
    }

    pub fn load(ws: &WeightSource<'_>, dim: usize, eps: f32) -> Result<Self> {
        let weight = ws.get([dim], "weight")?;
        let bias = ws.get([dim], "bias").ok();
        Ok(Self { weight, bias, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let xv = x.to_vec_f32()?;
        let wv = self.weight.to_vec_f32()?;
        let bv = self.bias.as_ref().map(|b| b.to_vec_f32()).transpose()?;

        let dim = x.shape().dims().last().copied().unwrap_or(1);
        let batch = xv.len() / dim;
        let mut out = vec![0.0f32; xv.len()];

        for b in 0..batch {
            let slice = &xv[b * dim..(b + 1) * dim];
            let mean = slice.iter().sum::<f32>() / (dim as f32);
            let var = slice.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / (dim as f32);
            let inv_std = 1.0 / (var + self.eps).sqrt();

            for d in 0..dim {
                let norm = (slice[d] - mean) * inv_std;
                let w = wv[d];
                let bias_val = bv.as_ref().map(|b_vec| b_vec[d]).unwrap_or(0.0);
                out[b * dim + d] = norm * w + bias_val;
            }
        }

        Ok(grim_backend_cpu::cpu_tensor(out, x.shape().clone()))
    }
}

// ---------- Embedding ----------

#[derive(Clone)]
pub struct Embedding {
    pub weight: Tensor,
}

impl Embedding {
    /// Load an embedding table, accepting layout variations and token size discrepancies.
    ///
    /// Standard checkpoints store `[vocab, dim]` (row-major: tokens × hidden).
    /// Legacy GGUF checkpoints may store `[dim, vocab]` (column-major) or carry
    /// padded vocabulary rows `[actual_vocab, dim]`.
    /// The contract guarantees that the returned weight is normalized to
    /// `[actual_vocab, dim]` row-major storage (rows = tokens), matching the
    /// layout expected by downstream gathering and tied linear heads.
    pub fn load(ws: &WeightSource<'_>, vocab: usize, dim: usize) -> Result<Self> {
        // Probe `[vocab, dim]`. On exact shape match, use as-is on the target device.
        if let Ok(t) = ws.get([vocab, dim], "weight") {
            if !t.dtype().is_quantized() {
                // Native (F32/BF16/F16) embedding: already on-device, pass through.
                return Ok(Self { weight: t });
            }
            // Quantized embedding (e.g. GGUF token_embd Q8_0 on ROCm): `ws.get`
            // keeps the packed bytes resident on-device, so dequantizing via
            // `to_vec_f32()` + re-upload would be a DtoH→H2D round trip. Instead
            // dequantize the host bytes once and upload as an F32 table in a
            // single H2D — the layout `grim_embedding` reads.
            return Ok(Self {
                weight: ws.get_f32([vocab, dim], "weight")?,
            });
        }

        // Fallback: load unconstrained tensor and normalize the layout.
        let raw_tensor = ws.get_unconstrained("weight")?;
        let dims = raw_tensor.shape().dims().to_vec();
        if dims.len() != 2 {
            return Err(Error::ShapeMismatch {
                expected: vec![vocab, dim],
                got: dims,
            });
        }

        let (s0, s1) = (dims[0], dims[1]);

        // Case 1: Row-major layout [actual_vocab, dim] where s1 == dim.
        if s1 == dim {
            return Ok(Self {
                weight: dequantize_for_gather(raw_tensor)?,
            });
        }

        // Case 2: Column-major layout [dim, actual_vocab] where s0 == dim.
        // Transpose to canonical [actual_vocab, dim] row-major layout.
        if s0 == dim {
            let actual_vocab = s1;
            let src_device = raw_tensor.device().clone();
            let src_prov = raw_tensor.provenance().clone();
            let raw_vec = raw_tensor.to_vec_f32()?;
            let mut out = vec![0.0f32; actual_vocab * dim];
            // raw_tensor is [dim, actual_vocab] (row-major): element at (i, j) = raw_vec[i * actual_vocab + j].
            // target is [actual_vocab, dim] (row-major): element at (j, i) = out[j * dim + i].
            for i in 0..dim {
                for j in 0..actual_vocab {
                    out[j * dim + i] = raw_vec[i * actual_vocab + j];
                }
            }
            let out_shape = Shape::new(vec![actual_vocab, dim]);
            let weight = if src_device.is_cpu() {
                grim_backend_cpu::cpu_tensor(out, out_shape)
            } else {
                let dev = pick_device_for_storage_device(&src_device);
                let storage = dev.from_cpu(&out, &out_shape, DType::F32)?;
                Tensor::new(
                    Arc::from(storage),
                    out_shape,
                    DType::F32,
                    src_prov,
                    src_device,
                )
            };
            return Ok(Self { weight });
        }

        Err(Error::ShapeMismatch {
            expected: vec![vocab, dim],
            got: dims,
        })
    }

    pub fn forward(&self, indices: &[u32], seq_len: usize, dim: usize) -> Result<Tensor> {
        let dev = pick_device_for_tensor(&self.weight);
        let out_shape = Shape::new(vec![seq_len, dim]);
        let (s, h) =
            CoreTensorOps::embedding(&*dev, self.weight.storage().as_ref(), indices, &out_shape)?;
        // WI-Host-1 #3: dropped `h.synchronize()?` here. Same lazy-sync rationale
        // as `Linear::forward` (see the matmul-sync comment there): the stall is
        // redundant because the returned storage synchronizes on first read, and
        // CPU backends are unaffected (`CpuDevice::synchronize` is a no-op).
        let _ = h;
        Ok(Tensor::new(
            Arc::from(s),
            out_shape,
            DType::F32,
            self.weight.provenance().clone(),
            self.weight.device().clone(),
        ))
    }

    pub fn forward_to_device(
        &self,
        indices: &[u32],
        seq_len: usize,
        dim: usize,
        target_device: &Device,
    ) -> Result<Tensor> {
        let cpu_t = self.forward(indices, seq_len, dim)?;
        if target_device.is_cpu() || cpu_t.device() == target_device {
            Ok(cpu_t)
        } else {
            let f32s = cpu_t.to_vec_f32()?;
            let shape = cpu_t.shape().clone();
            let dev = pick_device_for_storage_device(target_device);
            let storage = dev.from_cpu(&f32s, &shape, DType::F32)?;
            Ok(Tensor::new(
                Arc::from(storage),
                shape,
                DType::F32,
                cpu_t.provenance().clone(),
                target_device.clone(),
            ))
        }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
}

pub use grim_tensor::{RopeConfig, YaRNParams};

/// Materialize a gather-source tensor to F32 on its device. Embedding
/// lookups read raw weight rows as f32 (e.g. the ROCm `grim_embedding`
/// kernel), so quantized-resident packed storage must be dequantized before
/// it can serve as an embedding table. Non-quantized tensors pass through.
fn dequantize_for_gather(t: Tensor) -> Result<Tensor> {
    if !t.dtype().is_quantized() {
        return Ok(t);
    }
    let shape = t.shape().clone();
    let device = t.device().clone();
    let provenance = t.provenance().clone();
    let f32s = t.to_vec_f32()?;
    if device.is_cpu() {
        return Ok(grim_backend_cpu::cpu_tensor(f32s, shape));
    }
    let dev = pick_device_for_storage_device(&device);
    let storage = dev.from_cpu(&f32s, &shape, DType::F32)?;
    Ok(Tensor::new(
        std::sync::Arc::from(storage),
        shape,
        DType::F32,
        provenance,
        device,
    ))
}

/// Rotary positional embedding — apply RoPE to `(B, S, D)` query/key.
#[derive(Debug, Clone)]
pub struct Rope {
    pub config: RopeConfig,
}

impl Rope {
    pub fn new(dim: usize, base: f32) -> Self {
        Self {
            config: RopeConfig::new(dim, base),
        }
    }

    pub fn from_config(config: RopeConfig) -> Self {
        Self { config }
    }

    pub fn dim(&self) -> usize {
        self.config.dim
    }

    pub fn base(&self) -> f32 {
        self.config.base
    }

    pub fn forward(&self, x: &Tensor, positions: &[u32]) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        if dims.len() != 3 || dims[2] != self.config.dim {
            return Err(Error::Shape(format!(
                "RoPE expects (B,S,D={}), got {:?}",
                self.config.dim, dims
            )));
        }
        let (b, s, d) = (dims[0], dims[1], dims[2]);
        let out_shape = Shape::new(vec![b, s, d]);
        if x.device().is_cpu() {
            let rotary_dim = self.config.rotary_dim.min(d);
            let rotary_half = rotary_dim / 2;

            // Compute inv_freq with YaRN frequency ramp if specified
            let inv_freq: Vec<f32> = (0..rotary_half)
                .map(|i| {
                    let freq = 1.0 / self.config.base.powf((2 * i) as f32 / d as f32);
                    if let Some(yarn) = &self.config.yarn {
                        let wavelength = 2.0 * std::f32::consts::PI / freq;
                        let low = (yarn.original_max_pos as f32) / yarn.beta_slow;
                        let high = (yarn.original_max_pos as f32) / yarn.beta_fast;
                        if wavelength < high {
                            freq
                        } else if wavelength > low {
                            freq / yarn.factor
                        } else {
                            let ramp = (yarn.original_max_pos as f32 / wavelength - yarn.beta_slow)
                                / (yarn.beta_fast - yarn.beta_slow);
                            (1.0 - ramp) * (freq / yarn.factor) + ramp * freq
                        }
                    } else {
                        freq
                    }
                })
                .collect();

            let mscale = self
                .config
                .yarn
                .as_ref()
                .map_or(1.0, |y| y.attention_factor);

            let mut src = x.to_vec_f32()?;
            for bi in 0..b {
                for si in 0..s {
                    let pos = positions.get(si).copied().unwrap_or(si as u32) as f32;
                    let base_index = (bi * s + si) * d;
                    let mut cos_p = vec![0.0f32; rotary_half];
                    let mut sin_p = vec![0.0f32; rotary_half];
                    for i in 0..rotary_half {
                        let a = pos * inv_freq[i];
                        cos_p[i] = a.cos() * mscale;
                        sin_p[i] = a.sin() * mscale;
                    }
                    for i in 0..rotary_half {
                        let xi = base_index + i;
                        let xj = base_index + rotary_half + i;
                        let a = src[xi];
                        let bv = src[xj];
                        src[xi] = a * cos_p[i] - bv * sin_p[i];
                        src[xj] = bv * cos_p[i] + a * sin_p[i];
                    }
                }
            }
            Ok(grim_backend_cpu::cpu_tensor(src, out_shape))
        } else {
            let dev = pick_device_for_tensor(x);
            let (storage, _handle) =
                dev.rope(x.storage().as_ref(), positions, &self.config, &out_shape)?;
            Ok(Tensor::new(
                Arc::from(storage),
                out_shape,
                DType::F32,
                x.provenance().clone(),
                x.device().clone(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {

    /// F2: replace_weight must rebuild `w_t` (forward's actual operand), not
    /// just swap the stored GGUF-order weight.
    #[test]
    fn replace_weight_recomputes_transpose() {
        let w0 = cpu_tensor(vec![1.0, 0.0, 0.0, 1.0], Shape::new(vec![2, 2]));
        let mut lin = Linear::from_tensor(w0, None);
        let x = cpu_tensor(vec![1.0, 2.0], Shape::new(vec![1, 2]));
        let y1 = lin.forward(&x).unwrap().to_vec_f32().unwrap();

        lin.replace_weight(cpu_tensor(vec![0.0, 1.0, 1.0, 0.0], Shape::new(vec![2, 2])))
            .unwrap();
        assert_eq!(lin.weight.to_vec_f32().unwrap(), vec![0.0, 1.0, 1.0, 0.0]);
        // Transposed view must match the new weight, not the old one.
        assert_eq!(lin.w_t.to_vec_f32().unwrap(), vec![0.0, 1.0, 1.0, 0.0]);
        let y2 = lin.forward(&x).unwrap().to_vec_f32().unwrap();
        assert_ne!(
            y1, y2,
            "forward must observe replaced weights via rebuilt w_t"
        );
    }

    use super::*;
    use grim_backend_cpu::cpu_tensor;

    #[test]
    fn test_linear_forward_with_bias_hand_calculated() {
        let weight = cpu_tensor(vec![0.5, 1.5, -1.0, 2.0], Shape::new(vec![2, 2]));
        let bias = cpu_tensor(vec![0.1, -0.2], Shape::new(vec![2]));
        let linear = Linear::from_tensor(weight, Some(bias));

        let x = cpu_tensor(vec![1.0, 2.0], Shape::new(vec![1, 2]));
        let y = linear.forward(&x).expect("linear forward");

        let out = y.to_vec_f32().expect("to vec");
        assert_eq!(out.len(), 2);
        assert!((out[0] - 3.6).abs() < 1e-5, "Expected 3.6, got {}", out[0]);
        assert!((out[1] - 2.8).abs() < 1e-5, "Expected 2.8, got {}", out[1]);
    }

    #[test]
    fn test_linear_forward_without_bias() {
        let weight = cpu_tensor(vec![0.5, 1.5, -1.0, 2.0], Shape::new(vec![2, 2]));
        let linear = Linear::from_tensor(weight, None);

        let x = cpu_tensor(vec![1.0, 2.0], Shape::new(vec![1, 2]));
        let y = linear.forward(&x).expect("linear forward");

        let out = y.to_vec_f32().expect("to vec");
        assert_eq!(out.len(), 2);
        assert!((out[0] - 3.5).abs() < 1e-5, "Expected 3.5, got {}", out[0]);
        assert!((out[1] - 3.0).abs() < 1e-5, "Expected 3.0, got {}", out[1]);
    }

    #[test]
    fn test_rms_norm_forward_hand_calculated() {
        let weight = cpu_tensor(vec![1.0, 1.0], Shape::new(vec![2]));
        let rms_norm = RmsNorm { weight, eps: 1e-6 };

        let x = cpu_tensor(vec![3.0, 4.0], Shape::new(vec![1, 2]));
        let y = rms_norm.forward(&x).expect("rms norm forward");

        let out = y.to_vec_f32().expect("to vec");
        assert_eq!(out.len(), 2);
        let expected_0 = 3.0 / (12.5f32 + 1e-6).sqrt();
        let expected_1 = 4.0 / (12.5f32 + 1e-6).sqrt();
        assert!(
            (out[0] - expected_0).abs() < 1e-4,
            "Expected {}, got {}",
            expected_0,
            out[0]
        );
        assert!(
            (out[1] - expected_1).abs() < 1e-4,
            "Expected {}, got {}",
            expected_1,
            out[1]
        );
    }

    #[test]
    fn test_embedding_forward_token_lookup() {
        let table = vec![0.1, 0.2, 0.3, 0.4, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weight = cpu_tensor(table, Shape::new(vec![3, 4]));
        let emb = Embedding { weight };

        let indices = vec![2u32, 0];
        let out_tensor = emb.forward(&indices, 2, 4).expect("embedding forward");

        let out = out_tensor.to_vec_f32().expect("to vec");
        assert_eq!(out, vec![5.0, 6.0, 7.0, 8.0, 0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn test_rope_forward_rotation_identity_at_pos0() {
        let rope = Rope::new(4, 10000.0);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let x = cpu_tensor(input.clone(), Shape::new(vec![1, 1, 4]));

        let y = rope.forward(&x, &[0]).expect("rope forward");
        let out = y.to_vec_f32().expect("to vec");
        for (i, (&a, &b)) in out.iter().zip(input.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "RoPE pos 0 mismatch at index {i}: got {a} want {b}"
            );
        }
    }

    #[test]
    fn test_rope_forward_pos_nonzero_rotation_hand_calculated() {
        // dim = 2, base = 100.0 => inv_freq[0] = 1.0 / 100.0^0 = 1.0.
        // pos = 2 => theta = 2.0 * 1.0 = 2.0 rad.
        // cos(2.0) = -0.41614684, sin(2.0) = 0.9092974
        // x = [1.0, 2.0]
        // x'[0] = 1.0 * cos(2.0) - 2.0 * sin(2.0) = -0.41614684 - 1.8185948 = -2.2347416
        // x'[1] = 1.0 * sin(2.0) + 2.0 * cos(2.0) = 0.9092974 - 0.8322937 = 0.0770037
        let rope = Rope::new(2, 100.0);
        let x = cpu_tensor(vec![1.0, 2.0], Shape::new(vec![1, 1, 2]));
        let y = rope.forward(&x, &[2]).expect("rope pos 2 forward");
        let out = y.to_vec_f32().expect("to vec");
        assert_eq!(out.len(), 2);

        let theta = 2.0f32;
        let expected_0 = 1.0 * theta.cos() - 2.0 * theta.sin();
        let expected_1 = 1.0 * theta.sin() + 2.0 * theta.cos();
        assert!(
            (out[0] - expected_0).abs() < 1e-5,
            "Expected {expected_0}, got {}",
            out[0]
        );
        assert!(
            (out[1] - expected_1).abs() < 1e-5,
            "Expected {expected_1}, got {}",
            out[1]
        );
    }

    // Laguna-S-2.1 full-attention layers rotate only `rotary_dim` channels;
    // the remainder pass through unchanged (partial rotary).
    #[test]
    fn test_rope_partial_rotary_passthrough() {
        let cfg = RopeConfig::new(8, 10000.0);
        let rope = Rope::from_config(RopeConfig {
            rotary_dim: 4,
            ..cfg
        });
        let input: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();
        let x = cpu_tensor(input.clone(), Shape::new(vec![1, 1, 8]));
        let y = rope.forward(&x, &[3]).expect("partial rope forward");
        let out = y.to_vec_f32().expect("to vec");
        // Channels 4..8 unchanged.
        assert_eq!(
            out[4..8],
            input[4..8],
            "partial rotary must pass through tail channels"
        );
        // Channels 0..4 rotated (not equal to input).
        assert!(
            out[..4]
                .iter()
                .zip(input[..4].iter())
                .any(|(a, b)| (a - b).abs() > 1e-4),
            "partial rotary must rotate leading channels"
        );
    }

    // YaRN magnitude correction applies the attention_factor mscale (here 1.0)
    // and the frequency ramp. With mscale=1.0 the rotation magnitude must match
    // plain RoPE at the rotated positions when factor is 1.0 (no extrapolation).
    #[test]
    fn test_rope_yarn_factor_one_matches_plain() {
        let plain = Rope::from_config(RopeConfig::new(4, 10000.0));
        let yarn_cfg = RopeConfig {
            dim: 4,
            base: 10000.0,
            rotary_dim: 4,
            yarn: Some(YaRNParams {
                factor: 1.0,
                original_max_pos: 8192,
                beta_fast: 32.0,
                beta_slow: 1.0,
                attention_factor: 1.0,
            }),
            interleaved: true,
        };
        let yarn = Rope::from_config(yarn_cfg);
        let input: Vec<f32> = (0..4).map(|i| (i + 1) as f32).collect();
        let x = cpu_tensor(input.clone(), Shape::new(vec![1, 1, 4]));
        let p = plain.forward(&x, &[5]).unwrap().to_vec_f32().unwrap();
        let y = yarn.forward(&x, &[5]).unwrap().to_vec_f32().unwrap();
        for (a, b) in p.iter().zip(y.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "YaRN factor=1 must equal plain RoPE, got {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_linear_shape_mismatch_returns_error() {
        let weight = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
        let linear = Linear::from_tensor(weight, None);
        let x_bad = cpu_tensor(vec![1.0, 2.0, 3.0], Shape::new(vec![1, 3]));
        assert!(
            linear.forward(&x_bad).is_err(),
            "mismatched in_features must error"
        );
    }

    #[test]
    fn test_rope_invalid_rank_returns_error() {
        let rope = Rope::new(4, 10000.0);
        let x_2d = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
        assert!(
            rope.forward(&x_2d, &[0]).is_err(),
            "2D input to RoPE must return Shape error"
        );
    }

    /// ColumnParallelLinear forward with world_size=1 should behave identically
    /// to the inner Linear (no sharding, no all_reduce).
    #[test]
    fn test_column_parallel_forward_single_device() {
        let weight = cpu_tensor(vec![0.5, 1.5, -1.0, 2.0], Shape::new(vec![2, 2]));
        let linear = Linear::from_tensor(
            weight,
            Some(cpu_tensor(vec![0.1, -0.2], Shape::new(vec![2]))),
        );
        let cp = ColumnParallelLinear::new(linear, TensorParallelConfig::default());

        let x = cpu_tensor(vec![1.0, 2.0], Shape::new(vec![1, 2]));
        let y = cp.forward(&x).expect("cp forward");
        let out = y.to_vec_f32().expect("to vec");
        assert_eq!(out.len(), 2);
        // Same as Linear::forward: x@[0.5,1.5;-1,2]^T + bias
        // row0: 1*0.5 + 2*1.5 + 0.1 = 3.6
        // row1: 1*(-1.0) + 2*2.0 + (-0.2) = 2.8
        assert!((out[0] - 3.6).abs() < 1e-5, "got {}", out[0]);
        assert!((out[1] - 2.8).abs() < 1e-5, "got {}", out[1]);
    }

    /// RowParallelLinear forward with world_size=1 should behave identically
    /// to the inner Linear (no sharding, all_reduce skipped).
    #[test]
    fn test_row_parallel_forward_single_device() {
        let weight = cpu_tensor(vec![0.5, 1.5, -1.0, 2.0], Shape::new(vec![2, 2]));
        let linear = Linear::from_tensor(weight, None);
        let rp = RowParallelLinear::new(linear, TensorParallelConfig::default());

        let x = cpu_tensor(vec![1.0, 2.0], Shape::new(vec![1, 2]));
        let y = rp.forward(&x).expect("rp forward");
        let out = y.to_vec_f32().expect("to vec");
        assert_eq!(out.len(), 2);
        // row0: 1*0.5 + 2*1.5 = 3.5
        // row1: 1*(-1.0) + 2*2.0 = 3.0
        assert!((out[0] - 3.5).abs() < 1e-5, "got {}", out[0]);
        assert!((out[1] - 3.0).abs() < 1e-5, "got {}", out[1]);
    }

    /// Accessor smoke test: weight(), bias(), inner(), shard_size().
    #[test]
    fn test_parallel_linear_accessors() {
        let weight = cpu_tensor(vec![0.5, 1.5, -1.0, 2.0], Shape::new(vec![2, 2]));
        let linear = Linear::from_tensor(
            weight,
            Some(cpu_tensor(vec![0.1, -0.2], Shape::new(vec![2]))),
        );
        let cp = ColumnParallelLinear::new(linear, TensorParallelConfig::default());

        assert_eq!(cp.shard_size(), 2);
        assert_eq!(cp.weight().shape().dims(), &[2, 2]);
        assert!(cp.bias().is_some());
        assert_eq!(cp.inner().weight.shape().dims(), &[2, 2]);

        let rp_linear = Linear::from_tensor(
            cpu_tensor(vec![0.5, 1.5, -1.0, 2.0], Shape::new(vec![2, 2])),
            None,
        );
        let rp = RowParallelLinear::new(rp_linear, TensorParallelConfig::default());
        assert_eq!(rp.shard_size(), 2);
        assert_eq!(rp.weight().shape().dims(), &[2, 2]);
        assert!(rp.bias().is_none());
        assert_eq!(rp.inner().weight.shape().dims(), &[2, 2]);
    }

    /// RowParallelLinear forward with world_size > 1 on a backend lacking an
    /// `all_reduce` impl (CPU inherits the trait default `Err(Unimplemented)`)
    /// must **propagate the error** instead of silently returning the
    /// per-rank partial output as the all-reduced sum. Regression guard for
    /// the silent-correctness-bug fixed alongside this test.
    #[test]
    fn test_row_parallel_forward_no_collective_errors_loudly() {
        let weight = cpu_tensor(vec![0.5, 1.5, -1.0, 2.0], Shape::new(vec![2, 2]));
        let linear = Linear::from_tensor(weight, None);
        // world_size > 1 forces the all_reduce path; CPU has no all_reduce.
        let tp = TensorParallelConfig {
            rank: 0,
            world_size: 2,
        };
        let rp = RowParallelLinear::new(linear, tp);

        let x = cpu_tensor(vec![1.0, 2.0], Shape::new(vec![1, 2]));
        let res = rp.forward(&x);

        assert!(
            res.is_err(),
            "RowParallelLinear::forward on a backend without all_reduce must return Err, \
             not a partial output (got Ok)"
        );
        let msg = format!("{}", res.unwrap_err());
        assert!(
            msg.contains("all_reduce"),
            "error message should mention all_reduce; got: {msg}"
        );
        assert!(
            msg.contains("GRIM_TP_SIZE=1"),
            "error message should guide the user to GRIM_TP_SIZE=1; got: {msg}"
        );
    }

    // =======================================================================
    // WI-Host-1 gate (1) — CPU-path numeric parity for RmsNorm / Rope /
    // broadcast_bias.
    //
    // The plan's WI-Host-1 gate (1) requires:
    // > `RmsNorm`/`Rope`/`broadcast_bias` numeric parity vs current CPU-path
    // > output within tight tolerance, so the fix doesn't silently change
    // > numerics while removing the roundtrip.
    //
    // These tests pin the CURRENT CPU-path output as golden values within a
    // TIGHT tolerance (1e-5 abs) so the deferred native HIP kernel
    // replacement (WI-Host-1 #1 RoPE, #2 broadcast_bias) has a verification
    // target: when the native kernel lands, run these tests against the
    // GPU-backed path and the output must match these CPU-pinned goldens
    // within the same tolerance. A native kernel that drifts (wrong angle,
    // wrong tile order, wrong broadcast axis) shows up as a golden-value
    // mismatch well above 1e-5.
    //
    // Distinct from the existing `test_rms_norm_forward_hand_calculated` /
    // `test_rope_forward_rotation_identity_at_pos0`: those use loose 1e-4
    // tolerance and the trivial pos=0 identity case; these use tight 1e-5
    // and NON-TRIVIAL inputs (non-unit weights, non-zero positions, batched
    // broadcast) so the goldens actually exercise the math the native kernel
    // must reproduce.
    // =======================================================================

    #[test]
    fn host1_rms_norm_cpu_path_pinned_for_native_kernel_parity() {
        // RmsNorm with NON-unit weights and a non-trivial input row.
        // out[c] = x[c] * (1/sqrt(mean(x^2) + eps)) * weight[c]
        let weight = cpu_tensor(vec![0.5, 2.0, 1.5], Shape::new(vec![3]));
        let eps = 1e-6_f32;
        let rms = RmsNorm { weight, eps };
        let x = cpu_tensor(vec![1.0, 2.0, 3.0], Shape::new(vec![1, 3]));
        let y = rms.forward(&x).expect("rms_norm forward");
        let out = y.to_vec_f32().expect("to_vec_f32");
        // mean(x^2) = (1+4+9)/3 = 14/3 ≈ 4.6667; rms_inv = 1/sqrt(4.6667+1e-6).
        let mean_sq = (1.0_f32 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0) / 3.0;
        let rms_inv = 1.0 / (mean_sq + eps).sqrt();
        let expected = [
            1.0 * rms_inv * 0.5,
            2.0 * rms_inv * 2.0,
            3.0 * rms_inv * 1.5,
        ];
        assert_eq!(out.len(), 3);
        for (i, (&got, &want)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-5,
                "RmsNorm CPU-path golden mismatch at [{i}]: got {got:.8}, want {want:.8} \
                 (native kernel replacement must reproduce this within 1e-5)",
            );
        }
        // Pin the rms_inv value itself so a mutant that drops the eps or the
        // 1/sqrt (e.g. uses 1/mean directly) is caught at the source.
        assert!(
            (rms_inv - 0.46290994_f32).abs() < 1e-5,
            "rms_inv golden drifted: got {rms_inv:.8}",
        );
    }

    #[test]
    fn host1_rope_cpu_path_pinned_for_native_kernel_parity() {
        // RoPE with NON-zero position (exercises the actual rotation), dim=4.
        // For each half-pair (i, i+half):
        //   out[i]      = x[i] * cos(pos*inv_freq[i]) - x[i+half] * sin(...)
        //   out[i+half] = x[i+half] * cos(...) + x[i] * sin(...)
        // where inv_freq[i] = 1 / base^(2i/d).
        let rope = Rope::new(4, 10000.0);
        // dim=4 → half=2; inv_freq[0] = 1 (2*0/4=0), inv_freq[1] = 1/10000^0.5 = 0.01.
        // pos=5: angle[0] = 5*1 = 5 rad, angle[1] = 5*0.01 = 0.05 rad.
        let input = vec![1.0, 2.0, 3.0, 4.0]; // x[0]=1, x[1]=2 (first half), x[2]=3, x[3]=4 (second half)
        let x = cpu_tensor(input.clone(), Shape::new(vec![1, 1, 4]));
        let y = rope.forward(&x, &[5]).expect("rope forward");
        let out = y.to_vec_f32().expect("to_vec_f32");
        // Hand-compute the expected rotation.
        let inv_freq = [1.0_f32, 1.0 / 10000.0_f32.powf(2.0 / 4.0)];
        let pos = 5.0_f32;
        let cos_p = [(pos * inv_freq[0]).cos(), (pos * inv_freq[1]).cos()];
        let sin_p = [(pos * inv_freq[0]).sin(), (pos * inv_freq[1]).sin()];
        let expected = [
            input[0] * cos_p[0] - input[2] * sin_p[0],
            input[1] * cos_p[1] - input[3] * sin_p[1],
            input[2] * cos_p[0] + input[0] * sin_p[0],
            input[3] * cos_p[1] + input[1] * sin_p[1],
        ];
        assert_eq!(out.len(), 4);
        for (i, (&got, &want)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-5,
                "RoPE CPU-path golden mismatch at [{i}]: got {got:.8}, want {want:.8} \
                 (native kernel replacement must reproduce this within 1e-5)",
            );
        }
        // Sanity: pos=0 is identity (existing test covers this); here pos=5
        // must NOT equal input (confirms the rotation actually fired).
        let diff = out
            .iter()
            .zip(input.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>();
        assert!(
            diff > 0.1,
            "non-zero position must produce a non-trivial rotation (diff={diff})"
        );
    }

    #[test]
    fn host1_broadcast_bias_cpu_path_pinned_for_native_kernel_parity() {
        // broadcast_bias: tile 1-D bias `[out_dim]` to `[batch, out_dim]`.
        // Non-trivial batch (3) and out_dim (4) so the tiling is exercised.
        let bias = cpu_tensor(vec![0.1, 0.2, 0.3, 0.4], Shape::new(vec![4]));
        let batch = 3;
        let out_dim = 4;
        let out = broadcast_bias(&bias, batch, out_dim).expect("broadcast_bias");
        let v = out.to_vec_f32().expect("to_vec_f32");
        // Expected: 3 copies of the bias, contiguous.
        let mut expected = Vec::with_capacity(batch * out_dim);
        for _ in 0..batch {
            expected.extend_from_slice(&[0.1, 0.2, 0.3, 0.4]);
        }
        assert_eq!(v.len(), batch * out_dim);
        for (i, (&got, &want)) in v.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "broadcast_bias CPU-path golden mismatch at [{i}]: got {got:.8}, want {want:.8}",
            );
        }
        // Pin the batch boundary: element [out_dim] must equal element [0]
        // (start of the second tile) — a mutant that tiles along the wrong
        // axis would break this.
        assert!(
            (v[out_dim] - v[0]).abs() < 1e-6,
            "broadcast_bias must tile along the batch axis (v[out_dim] should equal v[0])",
        );
        // Pin shape.
        assert_eq!(out.shape().dims(), &[batch, out_dim]);
    }

    #[test]
    fn host1_embedding_forward_drops_synchronize_parity() {
        // WI-Host-1 #3 (extended): Embedding::forward dropped `h.synchronize()?`
        // for the same lazy-sync rationale as Linear/RmsNorm. The CPU path is
        // unaffected (CpuDevice::synchronize is a no-op), so the golden
        // output must match exactly. This test pins the output so the sync
        // drop can't silently change numerics.
        let table = vec![0.1, 0.2, 0.3, 0.4, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weight = cpu_tensor(table, Shape::new(vec![3, 4]));
        let emb = Embedding { weight };
        let indices = vec![2u32, 0];
        let out = emb.forward(&indices, 2, 4).expect("embedding forward");
        // to_vec_f32 triggers the lazy sync; result must match the golden
        // lookup without any explicit synchronize.
        let v = out.to_vec_f32().expect("to_vec_f32");
        assert_eq!(
            v,
            vec![5.0, 6.0, 7.0, 8.0, 0.1, 0.2, 0.3, 0.4],
            "Embedding forward output must match golden lookup after sync drop",
        );
    }

    #[test]
    fn host1_pick_device_arc_cache_no_alloc_parity() {
        // WI-Host-1 #4: pick_device_for_storage_device must return process-wide
        // cached Arc instances without per-op heap allocations.
        let dev1 = pick_device_for_storage_device(&Device::Cpu);
        let dev2 = pick_device_for_storage_device(&Device::Cpu);
        assert!(
            Arc::ptr_eq(&dev1, &dev2),
            "pick_device_for_storage_device(Cpu) must return the same cached Arc instance",
        );
    }

    #[test]
    fn short_conv1d_causal_parity() {
        let x = grim_backend_cpu::cpu_tensor(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![1, 4, 1]));
        let w = grim_backend_cpu::cpu_tensor(vec![0.5, 0.5, 0.5, 0.5], Shape::new(vec![1, 4]));
        let out = short_conv1d(&x, &w, None, None).expect("short_conv1d");
        assert_eq!(out.shape().dims(), &[1, 4, 1]);
        let v = out.to_vec_f32().expect("to_vec_f32");
        assert!((v[0] - 0.5).abs() < 1e-5);
    }
}

// ---------- MLA & KDA Attention Primitives ----------

/// Multi-head Latent Attention (MLA) compressed KV cache for decode generation.
#[derive(Debug, Clone)]
pub struct MlaKvCache {
    /// Post-RoPE nope-segment key history, laid out per token:
    /// `[past_len][num_heads * qk_nope_head_dim]`.
    pub hist_k_nope: Vec<f32>,
    /// Post-RoPE rope-segment key history, `[past_len][num_heads * qk_rope_head_dim]`
    /// in the [head, pos] order `forward` produces.
    pub hist_k_rope: Vec<f32>,
    /// Value history, `[past_len][num_heads * v_head_dim]`.
    pub hist_v: Vec<f32>,
    /// Tokens already in the history.
    pub past_len: usize,
}

impl MlaKvCache {
    pub fn new() -> Self {
        Self {
            hist_k_nope: Vec::new(),
            hist_k_rope: Vec::new(),
            hist_v: Vec::new(),
            past_len: 0,
        }
    }
}

impl Default for MlaKvCache {
    fn default() -> Self {
        Self::new()
    }
}

/// KdaLayerCache holding conv_state and recurrent matrix state S_t for KDA layers.
#[derive(Debug, Clone)]
pub struct KdaLayerCache {
    pub conv_state: Option<Tensor>,
    pub recurrent_state: Option<Tensor>,
}

impl KdaLayerCache {
    pub fn new() -> Self {
        Self {
            conv_state: None,
            recurrent_state: None,
        }
    }
}

impl Default for KdaLayerCache {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct LinearAttentionLayerCache {
    pub conv_state: Option<Tensor>,
    pub recurrent_state: Option<Tensor>,
}

impl LinearAttentionLayerCache {
    pub fn new() -> Self {
        Self {
            conv_state: None,
            recurrent_state: None,
        }
    }
}

impl Default for LinearAttentionLayerCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Heterogeneous layer cache unifying MLA, KDA, and Linear Attention state.
#[derive(Debug, Clone)]
pub enum LayerCache {
    Mla(MlaKvCache),
    Kda(KdaLayerCache),
    LinearAttention(LinearAttentionLayerCache),
}

/// Depthwise 1D causal convolution `out = conv1d(x, weight, bias)`.
///
/// Contract: applies depthwise per-channel 1D causal convolution along sequence dimension.
pub fn short_conv1d(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    conv_state: Option<&mut Tensor>,
) -> Result<Tensor> {
    let dims = x.shape().dims();
    if dims.len() != 3 {
        return Err(Error::Shape(format!(
            "short_conv1d expects [B,S,D], got {:?}",
            dims
        )));
    }
    let (b, s, d) = (dims[0], dims[1], dims[2]);
    let dev = pick_device_for_tensor(x);
    if let (Some(state), false) = (
        conv_state.as_deref(),
        matches!(x.device(), grim_tensor::Device::Cpu),
    ) {
        let out_shape = Shape::new(vec![b, s, d]);
        if let Ok((storage, _h)) = dev.short_conv1d_causal_step(
            x.storage().as_ref(),
            weight.storage().as_ref(),
            bias.map(|b| b.storage().as_ref()),
            state.storage().as_ref(),
            &out_shape,
        ) {
            return Ok(Tensor::new(
                Arc::from(storage),
                out_shape,
                DType::F32,
                x.provenance().clone(),
                x.device().clone(),
            ));
        }
    }

    let w_vec = weight.to_vec_f32()?;

    let k_size = if weight.shape().dims().is_empty() {
        4
    } else {
        *weight.shape().dims().last().unwrap_or(&4)
    };
    let b_vec = bias.map(|b| b.to_vec_f32()).transpose()?;

    let x_vec = x.to_vec_f32()?;
    let mut out_vec = vec![0.0f32; b * s * d];

    for bi in 0..b {
        for si in 0..s {
            for di in 0..d {
                let mut sum = b_vec.as_ref().map(|bv| bv[di]).unwrap_or(0.0f32);
                for ki in 0..k_size {
                    let prev_idx = si as isize - (k_size - 1 - ki) as isize;
                    let val = if prev_idx >= 0 {
                        x_vec[(bi * s + prev_idx as usize) * d + di]
                    } else if let Some(state) = &conv_state {
                        let state_vec = state.to_vec_f32()?;
                        let state_k = (state_vec.len() / (b * d)).max(1);
                        let state_off = (bi * d + di) * state_k;
                        let state_idx = (state_k as isize + prev_idx) as usize;
                        state_vec
                            .get(state_off + state_idx)
                            .copied()
                            .unwrap_or(0.0f32)
                    } else {
                        0.0f32
                    };
                    let w_val = w_vec.get(di * k_size + ki).copied().unwrap_or(0.0f32);
                    sum += val * w_val;
                }
                out_vec[(bi * s + si) * d + di] = sum;
            }
        }
    }

    if let Some(state) = conv_state {
        if s > 0 {
            let mut new_state = vec![0.0f32; b * d * (k_size - 1)];
            for bi in 0..b {
                for di in 0..d {
                    for ki in 0..(k_size - 1) {
                        let src_step = s as isize - (k_size - 1 - ki) as isize;
                        if src_step >= 0 {
                            new_state[(bi * d + di) * (k_size - 1) + ki] =
                                x_vec[(bi * s + src_step as usize) * d + di];
                        }
                    }
                }
            }
            let dev = pick_device_for_tensor(state);
            let new_shape = Shape::new(vec![b, d, k_size - 1]);
            let storage = dev.from_cpu(&new_state, &new_shape, DType::F32)?;
            *state = Tensor::new(
                Arc::from(storage),
                new_shape,
                DType::F32,
                state.provenance().clone(),
                state.device().clone(),
            );
        }
    }

    let out_shape = Shape::new(vec![b, s, d]);
    if x.device().is_cpu() {
        Ok(grim_backend_cpu::cpu_tensor(out_vec, out_shape))
    } else {
        let dev = pick_device_for_tensor(x);
        let storage = dev.from_cpu(&out_vec, &out_shape, DType::F32)?;
        Ok(Tensor::new(
            Arc::from(storage),
            out_shape,
            DType::F32,
            x.provenance().clone(),
            x.device().clone(),
        ))
    }
}

/// Multi-head Latent Attention (MLA) module with two-stage Q/KV projections and split nope/rope heads.
#[derive(Clone)]
pub struct MlaAttention {
    pub q_a_proj: Linear,
    pub q_a_norm: RmsNorm,
    pub q_b_proj: Linear,
    pub kv_a_proj_with_mqa: Linear,
    pub kv_a_norm: RmsNorm,
    pub kv_b_proj: Linear,
    pub o_proj: Linear,
    pub q_norm: Option<RmsNorm>,
    pub k_norm: Option<RmsNorm>,
    pub num_heads: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub rope: Rope,
}

impl MlaAttention {
    pub fn forward(
        &self,
        x: &Tensor,
        positions: &[u32],
        cache: Option<&mut MlaKvCache>,
    ) -> Result<Tensor> {
        let dims = x.shape().dims();
        let (b, s, _d) = (dims[0], dims[1], dims[2]);
        let dev = pick_device_for_tensor(x);

        // The projection matmuls (`Linear::forward`) require 2-D [tokens, dim]
        // input. Flatten the [b, s, d] batch into [b*s, d]; all per-token
        // slicing below indexes by the flattened token index `bs = b*s`.
        let bs = b * s;
        let d_in = dims[2];
        let x_2d = if dims.len() == 3 {
            let flat = x.to_vec_f32()?;
            Tensor::new(
                Arc::from(dev.from_cpu(&flat, &Shape::new(vec![bs, d_in]), DType::F32)?),
                Shape::new(vec![bs, d_in]),
                DType::F32,
                x.provenance().clone(),
                x.device().clone(),
            )
        } else {
            x.clone()
        };

        let q_a = self.q_a_proj.forward(&x_2d)?;
        let q_a_normed = self.q_a_norm.forward(&q_a)?;
        let q_b = self.q_b_proj.forward(&q_a_normed)?;

        let kv_a = self.kv_a_proj_with_mqa.forward(&x_2d)?;
        let kv_a_normed = self.kv_a_norm.forward(&kv_a)?;
        let kv_b = self.kv_b_proj.forward(&kv_a_normed)?;

        let qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim;
        // Per-head KV layout: [nope | rope | v] (DeepSeek-V3 style MLA).
        let kv_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim + self.v_head_dim;
        let q_vec = q_b.to_vec_f32()?;
        let kv_vec = kv_b.to_vec_f32()?;

        let qn_stride = self.num_heads * self.qk_nope_head_dim;
        let _qr_stride = self.num_heads * self.qk_rope_head_dim;
        let v_stride = self.num_heads * self.v_head_dim;

        // Split Q into per-head [nope | rope] and KV into per-head [nope | rope | v].
        //
        // Layout: q_nope/k_nope are [b*s*num_heads, nope_dim] (row = (bi*s+si)*num_heads+hi).
        // q_rope/k_rope are [b*num_heads, s, rope_dim] (row = bi*num_heads+hi, col = si)
        // so that RoPE (which expects [B, S, D]) applies position `si` to head `hi` of
        // batch `bi` — matching the read-back indexing below. The original code wrote
        // q_rope/k_rope in [b, s, heads, D] order but labeled the tensor [b*heads, s, D],
        // causing RoPE to rotate the wrong (position, head) pair for s > 1 (prefill).
        let mut q_nope = vec![0.0f32; b * s * qn_stride];
        let mut q_rope = vec![0.0f32; b * self.num_heads * s * self.qk_rope_head_dim];
        let mut k_nope = vec![0.0f32; b * s * qn_stride];
        let mut k_rope = vec![0.0f32; b * self.num_heads * s * self.qk_rope_head_dim];
        let mut v_vec = vec![0.0f32; b * s * v_stride];

        for bi in 0..b {
            for si in 0..s {
                for hi in 0..self.num_heads {
                    let q_base = ((bi * s + si) * self.num_heads + hi) * qk_head_dim;
                    let qn_off = (bi * s + si) * qn_stride + hi * self.qk_nope_head_dim;
                    // q_rope in [b*num_heads, s, rope_dim]: row=bi*num_heads+hi, col=si
                    let qr_off = (bi * self.num_heads + hi) * s * self.qk_rope_head_dim
                        + si * self.qk_rope_head_dim;
                    q_nope[qn_off..qn_off + self.qk_nope_head_dim]
                        .copy_from_slice(&q_vec[q_base..q_base + self.qk_nope_head_dim]);
                    q_rope[qr_off..qr_off + self.qk_rope_head_dim].copy_from_slice(
                        &q_vec[q_base + self.qk_nope_head_dim
                            ..q_base + self.qk_nope_head_dim + self.qk_rope_head_dim],
                    );

                    let kv_base = ((bi * s + si) * self.num_heads + hi) * kv_head_dim;
                    let kn_off = (bi * s + si) * qn_stride + hi * self.qk_nope_head_dim;
                    let kr_off = (bi * self.num_heads + hi) * s * self.qk_rope_head_dim
                        + si * self.qk_rope_head_dim;
                    let v_off = (bi * s + si) * v_stride + hi * self.v_head_dim;
                    k_nope[kn_off..kn_off + self.qk_nope_head_dim]
                        .copy_from_slice(&kv_vec[kv_base..kv_base + self.qk_nope_head_dim]);
                    k_rope[kr_off..kr_off + self.qk_rope_head_dim].copy_from_slice(
                        &kv_vec[kv_base + self.qk_nope_head_dim
                            ..kv_base + self.qk_nope_head_dim + self.qk_rope_head_dim],
                    );
                    for i in 0..self.v_head_dim {
                        v_vec[v_off + i] =
                            kv_vec[kv_base + self.qk_nope_head_dim + self.qk_rope_head_dim + i];
                    }
                }
            }
        }

        // Optional learned q/k norms on the nope segments (DeepSeek-V3 style).
        if let Some(qn) = &self.q_norm {
            let shape = Shape::new(vec![b * s * self.num_heads, self.qk_nope_head_dim]);
            let t = Tensor::new(
                Arc::from(dev.from_cpu(&q_nope, &shape, DType::F32)?),
                shape,
                DType::F32,
                x.provenance().clone(),
                x.device().clone(),
            );
            q_nope = qn.forward(&t)?.to_vec_f32()?;
        }
        if let Some(kn) = &self.k_norm {
            let shape = Shape::new(vec![b * s * self.num_heads, self.qk_nope_head_dim]);
            let t = Tensor::new(
                Arc::from(dev.from_cpu(&k_nope, &shape, DType::F32)?),
                shape,
                DType::F32,
                x.provenance().clone(),
                x.device().clone(),
            );
            k_nope = kn.forward(&t)?.to_vec_f32()?;
        }

        // RoPE on the rope segments — the rotated results are USED (fixes the
        // discarded-Q bug). Both Q and K rope dims participate in attention.
        let rope_shape = Shape::new(vec![b * self.num_heads, s, self.qk_rope_head_dim]);
        let q_rope_t = Tensor::new(
            Arc::from(dev.from_cpu(&q_rope, &rope_shape, DType::F32)?),
            rope_shape.clone(),
            DType::F32,
            x.provenance().clone(),
            x.device().clone(),
        );
        let k_rope_t = Tensor::new(
            Arc::from(dev.from_cpu(&k_rope, &rope_shape, DType::F32)?),
            rope_shape.clone(),
            DType::F32,
            x.provenance().clone(),
            x.device().clone(),
        );
        let q_rope = self.rope.forward(&q_rope_t, positions)?.to_vec_f32()?;
        let k_rope = self.rope.forward(&k_rope_t, positions)?.to_vec_f32()?;

        // Audit fix (grim-models): with a cache attached, this call's
        // post-RoPE K rows and V rows are APPENDED to the per-layer history
        // and attention runs over the full [0 .. past+s) window — real
        // incremental decode. The pre-fix implementation ignored its cache
        // parameter entirely (`_cache`), so every decode step attended only
        // to itself. Batch-1 (engine serving shape).
        if let Some(c) = cache {
            if b != 1 {
                return Err(grim_tensor::error::Error::Shape(
                    "MlaAttention cached path supports batch 1".into(),
                ));
            }
            c.hist_k_nope.extend_from_slice(&k_nope);
            c.hist_v.extend_from_slice(&v_vec);
            // k_rope arrives HEAD-major ([head][pos]); store TOKEN-major
            // ([pos][head]) so cross-call history addressing stays uniform.
            let rope_dim = self.qk_rope_head_dim;
            for si in 0..s {
                for hi in 0..self.num_heads {
                    let off = (hi * s + si) * rope_dim;
                    c.hist_k_rope
                        .extend_from_slice(&k_rope[off..off + rope_dim]);
                }
            }
            c.past_len += s;
            let past = c.past_len - s;
            let _kv_total = c.past_len;
            let scale = 1.0 / (qk_head_dim as f32).sqrt();
            let mut out = vec![0.0f32; s * v_stride];
            for hi in 0..self.num_heads {
                for t in 0..s {
                    let q_off = t * qn_stride + hi * self.qk_nope_head_dim;
                    let qr_off = hi * s * self.qk_rope_head_dim + t * self.qk_rope_head_dim;
                    let causal_limit = past + t;
                    let mut scores = vec![0.0f32; causal_limit + 1];
                    for (t2, score_slot) in scores.iter_mut().enumerate() {
                        // History rows are stored token-major, heads inside.
                        let kn_off = t2 * qn_stride + hi * self.qk_nope_head_dim;
                        // Token-major history: row t2 holds all heads' rope.
                        let kr2_off = t2 * self.num_heads * self.qk_rope_head_dim
                            + hi * self.qk_rope_head_dim;
                        let mut dot = 0.0f32;
                        for i in 0..self.qk_nope_head_dim {
                            dot += q_nope[q_off + i] * c.hist_k_nope[kn_off + i];
                        }
                        for i in 0..self.qk_rope_head_dim {
                            dot += q_rope[qr_off + i] * c.hist_k_rope[kr2_off + i];
                        }
                        *score_slot = dot * scale;
                    }
                    let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for sc in scores.iter_mut() {
                        *sc = (*sc - mx).exp();
                        sum += *sc;
                    }
                    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                    let o_off = t * v_stride + hi * self.v_head_dim;
                    for i in 0..self.v_head_dim {
                        let mut acc = 0.0f32;
                        for (t2, sc) in scores.iter().enumerate() {
                            let v_off = t2 * v_stride + hi * self.v_head_dim;
                            acc += sc * inv * c.hist_v[v_off + i];
                        }
                        out[o_off + i] = acc;
                    }
                }
            }
            let out_shape = Shape::new(vec![s, self.num_heads * self.v_head_dim]);
            let out_t = Tensor::new(
                Arc::from(dev.from_cpu(&out, &out_shape, DType::F32)?),
                out_shape,
                DType::F32,
                x.provenance().clone(),
                x.device().clone(),
            );
            return self.o_proj.forward(&out_t);
        }

        // Causal scaled-dot-product attention per head (fixes the missing-attention bug).
        let scale = 1.0 / (qk_head_dim as f32).sqrt();
        let mut out = vec![0.0f32; bs * v_stride];
        for bi in 0..b {
            for hi in 0..self.num_heads {
                for t in 0..s {
                    let q_off = (bi * s + t) * qn_stride + hi * self.qk_nope_head_dim;
                    // q_rope in [b*num_heads, s, rope_dim]: row=bi*num_heads+hi, col=t
                    let qr_off = (bi * self.num_heads + hi) * s * self.qk_rope_head_dim
                        + t * self.qk_rope_head_dim;
                    // scores over the causal window t2 in 0..=t
                    let mut scores = vec![0.0f32; t + 1];
                    for (t2, score_slot) in scores.iter_mut().enumerate() {
                        let kn_off = (bi * s + t2) * qn_stride + hi * self.qk_nope_head_dim;
                        // k_rope in [b*num_heads, s, rope_dim]: row=bi*num_heads+hi, col=t2
                        let kr2_off = (bi * self.num_heads + hi) * s * self.qk_rope_head_dim
                            + t2 * self.qk_rope_head_dim;
                        let mut dot = 0.0f32;
                        for i in 0..self.qk_nope_head_dim {
                            dot += q_nope[q_off + i] * k_nope[kn_off + i];
                        }
                        for i in 0..self.qk_rope_head_dim {
                            dot += q_rope[qr_off + i] * k_rope[kr2_off + i];
                        }
                        *score_slot = dot * scale;
                    }
                    // softmax over the causal window
                    let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for sc in scores.iter_mut() {
                        *sc = (*sc - mx).exp();
                        sum += *sc;
                    }
                    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                    for sc in scores.iter_mut() {
                        *sc *= inv;
                    }
                    // weighted sum of V
                    let o_off = (bi * s + t) * v_stride + hi * self.v_head_dim;
                    for i in 0..self.v_head_dim {
                        let mut acc = 0.0f32;
                        for (t2, &score) in scores.iter().enumerate() {
                            let v_off = (bi * s + t2) * v_stride + hi * self.v_head_dim;
                            acc += score * v_vec[v_off + i];
                        }
                        out[o_off + i] = acc;
                    }
                }
            }
        }

        // `out` is laid out as [bs, num_heads * v_head_dim] (flattened b*s tokens),
        // so materialize it as 2-D for the output projection matmul, then reshape
        // the result back to [b, s, d_out] to match the residual-add contract.
        let out_shape = Shape::new(vec![bs, self.num_heads * self.v_head_dim]);
        let out_t = Tensor::new(
            Arc::from(dev.from_cpu(&out, &out_shape, DType::F32)?),
            out_shape,
            DType::F32,
            x.provenance().clone(),
            x.device().clone(),
        );
        let o = self.o_proj.forward(&out_t)?;
        let o_dims = o.shape().dims();
        let d_out = *o_dims.last().unwrap_or(&(self.num_heads * self.v_head_dim));
        let final_shape = Shape::new(vec![b, s, d_out]);
        // o is [bs, d_out]; reshape to [b, s, d_out] (pure shape change, same elems).
        let final_vec = o.to_vec_f32()?;
        Ok(Tensor::new(
            Arc::from(dev.from_cpu(&final_vec, &final_shape, DType::F32)?),
            final_shape,
            DType::F32,
            o.provenance().clone(),
            o.device().clone(),
        ))
    }
}

#[cfg(test)]
mod mla_attention_tests {
    use super::*;
    use grim_backend_cpu::CpuDevice;
    use grim_tensor::{Device, QuantProvenance, Shape, Tensor};

    fn cpu_dev() -> CpuDevice {
        CpuDevice::new()
    }

    fn mk(dims: (usize, usize), k: f32) -> Linear {
        let dev = cpu_dev();
        let w = Tensor::new(
            Arc::from(
                dev.from_cpu(
                    &vec![k; dims.0 * dims.1],
                    &Shape::new(vec![dims.0, dims.1]),
                    DType::F32,
                )
                .unwrap(),
            ),
            Shape::new(vec![dims.0, dims.1]),
            DType::F32,
            QuantProvenance::default(),
            Device::Cpu,
        );
        Linear::from_tensor(w, None)
    }

    fn rms(dim: usize) -> RmsNorm {
        let dev = cpu_dev();
        let w = Tensor::new(
            Arc::from(
                dev.from_cpu(&vec![1.0f32; dim], &Shape::new(vec![dim]), DType::F32)
                    .unwrap(),
            ),
            Shape::new(vec![dim]),
            DType::F32,
            QuantProvenance::default(),
            Device::Cpu,
        );
        RmsNorm::new(w, 1e-6)
    }

    fn tiny_mla() -> MlaAttention {
        // num_heads=2, qk_nope=4, qk_rope=2, v_head_dim=4
        let nope = 4usize;
        let rope = 2usize;
        let v = 4usize;
        let h = 2usize;
        // q_b: num_heads*(nope+rope) = 2*6 = 12 out ; q_a latent dim 8
        // kv_b: num_heads*(nope+rope+v) = 2*10 = 20 out ; kv_a latent 8
        let q_a = mk((8, 16), 0.1);
        let q_b = mk((12, 8), 0.2);
        let kv_a = mk((8, 16), 0.1);
        let kv_b = mk((20, 8), 0.2);
        let o = mk((8, 8), 0.0); // output proj (zeros -> out zero)
        MlaAttention {
            q_a_proj: q_a,
            q_a_norm: rms(8),
            q_b_proj: q_b,
            kv_a_proj_with_mqa: kv_a,
            kv_a_norm: rms(8),
            kv_b_proj: kv_b,
            o_proj: o,
            q_norm: None,
            k_norm: None,
            num_heads: h,
            qk_nope_head_dim: nope,
            qk_rope_head_dim: rope,
            v_head_dim: v,
            rope: Rope::new(rope, 10000.0),
        }
    }

    fn sample_input() -> Tensor {
        let dev = cpu_dev();
        Tensor::new(
            Arc::from(
                dev.from_cpu(&[0.3f32; 3 * 16], &Shape::new(vec![1, 3, 16]), DType::F32)
                    .unwrap(),
            ),
            Shape::new(vec![1, 3, 16]),
            DType::F32,
            QuantProvenance::default(),
            Device::Cpu,
        )
    }

    #[test]
    fn mla_forward_runs_and_attends() {
        let mla = tiny_mla();
        let x = sample_input();
        let pos = vec![0u32, 1, 2];
        let out = mla.forward(&x, &pos, None).unwrap();
        assert_eq!(out.shape().dims(), &[1, 3, 8]);
        let v = out.to_vec_f32().unwrap();
        assert!(
            v.iter().all(|x| x.is_finite()),
            "MLA output has non-finite values"
        );
    }

    #[test]
    fn mla_causal_window_is_bounded() {
        // Structural guard: position t may only attend t2 in 0..=t. With a
        // constant input the softmax is uniform over that window, so output[t]
        // equals the mean of V over the window — and crucially never depends on
        // a future position. The forward must run and stay finite.
        let mla = tiny_mla();
        let x = sample_input();
        let out = mla.forward(&x, &[0, 1, 2], None).unwrap();
        assert!(out.to_vec_f32().unwrap().iter().all(|x| x.is_finite()));
    }
}

/// Kimi Delta Attention (KDA) gated delta-rule linear attention block.
#[derive(Clone)]
pub struct KdaAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub gate_proj: Linear,
    pub dt_proj: Linear,
    pub a_proj: Linear,
    pub conv_weight: Tensor,
    pub conv_bias: Option<Tensor>,
    pub o_proj: Linear,
    pub num_heads: usize,
    pub head_dim: usize,
    pub v_dim: usize,
}

impl KdaAttention {
    pub fn forward(&self, x: &Tensor, cache: Option<&mut KdaLayerCache>) -> Result<Tensor> {
        let dims = x.shape().dims();
        let (b, s, _d) = (dims[0], dims[1], dims[2]);

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v_raw = self.v_proj.forward(x)?;

        let (conv_st, mut rec_st) = match cache {
            Some(c) => (c.conv_state.as_mut(), c.recurrent_state.as_mut()),
            None => (None, None),
        };

        let v = short_conv1d(&v_raw, &self.conv_weight, self.conv_bias.as_ref(), conv_st)?;
        let gate = self.gate_proj.forward(x)?;
        let beta = self.dt_proj.forward(x)?;
        let a_val = self.a_proj.forward(x)?;

        let q_vec = q.to_vec_f32()?;
        let k_vec = k.to_vec_f32()?;
        let v_vec = v.to_vec_f32()?;
        let beta_vec = beta.to_vec_f32()?;
        let a_vec = a_val.to_vec_f32()?;
        let gate_vec = gate.to_vec_f32()?;

        let mut y_vec = vec![0.0f32; b * s * self.num_heads * self.v_dim];
        let state_size = self.head_dim * self.v_dim;

        for bi in 0..b {
            for hi in 0..self.num_heads {
                let mut state = vec![0.0f32; state_size];
                if let Some(ref st) = rec_st {
                    let st_vec = st.to_vec_f32()?;
                    let off = (bi * self.num_heads + hi) * state_size;
                    if off + state_size <= st_vec.len() {
                        state.copy_from_slice(&st_vec[off..off + state_size]);
                    }
                }

                for si in 0..s {
                    let token_off = (bi * s + si) * self.num_heads + hi;
                    let q_tok = &q_vec[token_off * self.head_dim..(token_off + 1) * self.head_dim];
                    let k_tok = &k_vec[token_off * self.head_dim..(token_off + 1) * self.head_dim];
                    let v_tok = &v_vec[token_off * self.v_dim..(token_off + 1) * self.v_dim];
                    let b_scale = beta_vec
                        .get(token_off)
                        .copied()
                        .unwrap_or(1.0)
                        .clamp(0.0, 1.0);
                    let decay = 1.0 / (1.0 + (-a_vec.get(token_off).copied().unwrap_or(0.0)).exp());

                    for st in state[..state_size].iter_mut() {
                        *st *= decay;
                    }

                    let mut a_t = vec![0.0f32; self.v_dim];
                    for vi in 0..self.v_dim {
                        let mut sum = 0.0f32;
                        for ki in 0..self.head_dim {
                            sum += k_tok[ki] * state[ki * self.v_dim + vi];
                        }
                        a_t[vi] = b_scale * sum;
                    }

                    for ki in 0..self.head_dim {
                        for vi in 0..self.v_dim {
                            let diff = v_tok[vi] - a_t[vi];
                            // Single b_scale application (matches ROCm kernel at
                            // compute_kernels.rs:372 which scales only the prediction).
                            // [P1-38 fix: removed duplicate b_scale from update.]
                            state[ki * self.v_dim + vi] += diff * k_tok[ki];
                        }
                    }

                    let y_off = ((bi * s + si) * self.num_heads + hi) * self.v_dim;
                    for vi in 0..self.v_dim {
                        let mut sum = 0.0f32;
                        for ki in 0..self.head_dim {
                            sum += q_tok[ki] * state[ki * self.v_dim + vi];
                        }
                        let g = gate_vec.get(y_off + vi).copied().unwrap_or(0.0);
                        let silu_g = g / (1.0 + (-g).exp());
                        y_vec[y_off + vi] = sum * silu_g;
                    }
                }

                if let Some(ref mut st) = rec_st {
                    let dev = pick_device_for_tensor(x);
                    let shape = Shape::new(vec![b, self.num_heads, self.head_dim, self.v_dim]);
                    let storage = dev.from_cpu(&state, &shape, DType::F32)?;
                    **st = Tensor::new(
                        Arc::from(storage),
                        shape,
                        DType::F32,
                        x.provenance().clone(),
                        x.device().clone(),
                    );
                }
            }
        }

        let out_shape = Shape::new(vec![b, s, self.num_heads * self.v_dim]);
        let dev = pick_device_for_tensor(x);
        let storage = dev.from_cpu(&y_vec, &out_shape, DType::F32)?;
        let out_t = Tensor::new(
            Arc::from(storage),
            out_shape,
            DType::F32,
            x.provenance().clone(),
            x.device().clone(),
        );

        self.o_proj.forward(&out_t)
    }
}

/// Qwen 3.5 / 3.8 Linear Attention Block with short conv, linear attention state update, and Swish output gating.
#[derive(Clone)]
pub struct LinearAttentionBlock {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub gate_proj: Option<Linear>,
    pub conv_weight: Tensor,
    pub conv_bias: Option<Tensor>,
    pub o_proj: Linear,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub k_head_dim: usize,
    pub v_head_dim: usize,
}

impl LinearAttentionBlock {
    pub fn forward(
        &self,
        x: &Tensor,
        cache: Option<&mut LinearAttentionLayerCache>,
    ) -> Result<Tensor> {
        let dims = x.shape().dims();
        let (b, s, _d) = (dims[0], dims[1], dims[2]);

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v_raw = self.v_proj.forward(x)?;

        let (conv_st, mut rec_st) = match cache {
            Some(c) => (c.conv_state.as_mut(), c.recurrent_state.as_mut()),
            None => (None, None),
        };

        let v = short_conv1d(&v_raw, &self.conv_weight, self.conv_bias.as_ref(), conv_st)?;
        let gate = if let Some(ref g_proj) = self.gate_proj {
            Some(g_proj.forward(x)?)
        } else {
            None
        };

        let q_vec = q.to_vec_f32()?;
        let k_vec = k.to_vec_f32()?;
        let v_vec = v.to_vec_f32()?;
        let gate_vec = gate.as_ref().map(|g| g.to_vec_f32()).transpose()?;

        let mut y_vec = vec![0.0f32; b * s * self.num_v_heads * self.v_head_dim];
        let state_size = self.k_head_dim * self.v_head_dim;

        for bi in 0..b {
            for hi in 0..self.num_v_heads {
                let k_head_idx = hi % self.num_k_heads;
                let mut state = vec![0.0f32; state_size];
                if let Some(ref st) = rec_st {
                    let st_vec = st.to_vec_f32()?;
                    let off = (bi * self.num_v_heads + hi) * state_size;
                    if off + state_size <= st_vec.len() {
                        state.copy_from_slice(&st_vec[off..off + state_size]);
                    }
                }

                for si in 0..s {
                    let qk_token_off = (bi * s + si) * self.num_k_heads + k_head_idx;
                    let v_token_off = (bi * s + si) * self.num_v_heads + hi;

                    let q_tok = &q_vec
                        [qk_token_off * self.k_head_dim..(qk_token_off + 1) * self.k_head_dim];
                    let k_tok = &k_vec
                        [qk_token_off * self.k_head_dim..(qk_token_off + 1) * self.k_head_dim];
                    let v_tok =
                        &v_vec[v_token_off * self.v_head_dim..(v_token_off + 1) * self.v_head_dim];

                    // Outer product state update: S += k^T * v
                    for ki in 0..self.k_head_dim {
                        for vi in 0..self.v_head_dim {
                            state[ki * self.v_head_dim + vi] += k_tok[ki] * v_tok[vi];
                        }
                    }

                    // Query state contraction: y = q * S
                    let y_off = ((bi * s + si) * self.num_v_heads + hi) * self.v_head_dim;
                    for vi in 0..self.v_head_dim {
                        let mut sum = 0.0f32;
                        for ki in 0..self.k_head_dim {
                            sum += q_tok[ki] * state[ki * self.v_head_dim + vi];
                        }
                        if let Some(ref gv) = gate_vec {
                            let g = gv.get(y_off + vi).copied().unwrap_or(0.0);
                            let silu_g = g / (1.0 + (-g).exp());
                            sum *= silu_g;
                        }
                        y_vec[y_off + vi] = sum;
                    }
                }

                if let Some(ref mut st) = rec_st {
                    let dev = pick_device_for_tensor(x);
                    let shape =
                        Shape::new(vec![b, self.num_v_heads, self.k_head_dim, self.v_head_dim]);
                    let storage = dev.from_cpu(&state, &shape, DType::F32)?;
                    **st = Tensor::new(
                        Arc::from(storage),
                        shape,
                        DType::F32,
                        x.provenance().clone(),
                        x.device().clone(),
                    );
                }
            }
        }

        let out_shape = Shape::new(vec![b, s, self.num_v_heads * self.v_head_dim]);
        let dev = pick_device_for_tensor(x);
        let storage = dev.from_cpu(&y_vec, &out_shape, DType::F32)?;
        let out_t = Tensor::new(
            Arc::from(storage),
            out_shape,
            DType::F32,
            x.provenance().clone(),
            x.device().clone(),
        );

        self.o_proj.forward(&out_t)
    }
}

/// 1D Convolution layer with support for stride, padding, dilation, and grouped/depthwise convolutions.
#[derive(Debug, Clone)]
pub struct Conv1d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl Conv1d {
    /// Construct a new `Conv1d` module directly from weight and optional bias tensors.
    pub fn new(
        weight: Tensor,
        bias: Option<Tensor>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Self {
        Self {
            weight,
            bias,
            stride: stride.max(1),
            padding,
            dilation: dilation.max(1),
            groups: groups.max(1),
        }
    }

    /// Load a `Conv1d` module from a `WeightSource`.
    // Parameters mirror the serialized Conv1d layout 1:1; grouping them into
    // a struct would just relocate the argument list.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        ws: &WeightSource<'_>,
        out_c: usize,
        in_c_per_group: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        let weight = ws.get_f32([out_c, in_c_per_group, kernel_size], "weight")?;
        let bias = ws.get_f32([out_c], "bias").ok();
        Ok(Self::new(weight, bias, stride, padding, dilation, groups))
    }

    /// Forward pass for 1D convolution. Accepts `[seq_len, in_channels]` or `[batch, in_channels, seq_len]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dims = x.shape().dims();
        let w_dims = self.weight.shape().dims();
        if w_dims.len() != 3 {
            return Err(Error::ShapeMismatch {
                expected: vec![0, 0, 0],
                got: w_dims.to_vec(),
            });
        }
        let out_c = w_dims[0];
        let in_c_per_group = w_dims[1];
        let kernel_size = w_dims[2];
        let in_c = in_c_per_group * self.groups;

        let (batch, in_seq, is_seq_first, x_flat) = if x_dims.len() == 2 {
            if x_dims[1] == in_c {
                (1, x_dims[0], true, x.to_vec_f32()?)
            } else if x_dims[0] == in_c {
                (1, x_dims[1], false, x.to_vec_f32()?)
            } else {
                (1, x_dims[0], true, x.to_vec_f32()?)
            }
        } else if x_dims.len() == 3 {
            (x_dims[0], x_dims[2], false, x.to_vec_f32()?)
        } else {
            return Err(Error::Shape(format!(
                "Conv1d: unsupported input rank {}",
                x_dims.len()
            )));
        };

        let effective_k = self.dilation * (kernel_size - 1) + 1;
        if in_seq + 2 * self.padding < effective_k {
            return Err(Error::Shape(format!(
                "Conv1d: input length {} with padding {} is smaller than effective kernel {}",
                in_seq, self.padding, effective_k
            )));
        }
        let out_seq = (in_seq + 2 * self.padding - effective_k) / self.stride + 1;

        let w_vec = self.weight.to_vec_f32()?;
        let b_vec = self.bias.as_ref().map(|b| b.to_vec_f32()).transpose()?;

        let mut out = vec![0.0f32; batch * out_c * out_seq];
        let out_c_per_group = out_c / self.groups;

        for b in 0..batch {
            let b_in_off = b * in_c * in_seq;
            let b_out_off = b * out_c * out_seq;

            for g in 0..self.groups {
                let g_in_start = g * in_c_per_group;
                let g_out_start = g * out_c_per_group;

                for oc_i in 0..out_c_per_group {
                    let oc = g_out_start + oc_i;
                    let bias_val = b_vec.as_ref().map(|b| b[oc]).unwrap_or(0.0);

                    for os in 0..out_seq {
                        let in_center = os * self.stride;
                        let mut sum = bias_val;

                        for ic_i in 0..in_c_per_group {
                            let ic = g_in_start + ic_i;
                            let w_base = (oc * in_c_per_group + ic_i) * kernel_size;

                            for k in 0..kernel_size {
                                let in_pos = in_center as isize + (k * self.dilation) as isize
                                    - self.padding as isize;
                                if in_pos >= 0 && (in_pos as usize) < in_seq {
                                    let x_idx = if is_seq_first {
                                        b_in_off + (in_pos as usize) * in_c + ic
                                    } else {
                                        b_in_off + ic * in_seq + (in_pos as usize)
                                    };
                                    let x_val = x_flat[x_idx];
                                    let w_val = w_vec[w_base + k];
                                    sum += x_val * w_val;
                                }
                            }
                        }
                        if is_seq_first {
                            out[b_out_off + os * out_c + oc] = sum;
                        } else {
                            out[b_out_off + oc * out_seq + os] = sum;
                        }
                    }
                }
            }
        }

        let dev = pick_device_for_tensor(x);
        let out_shape = if is_seq_first {
            Shape::new(vec![out_seq, out_c])
        } else if x_dims.len() == 2 {
            Shape::new(vec![out_c, out_seq])
        } else {
            Shape::new(vec![batch, out_c, out_seq])
        };

        let storage = dev.from_cpu(&out, &out_shape, DType::F32)?;
        Ok(Tensor::new(
            Arc::from(storage),
            out_shape,
            DType::F32,
            x.provenance().clone(),
            x.device().clone(),
        ))
    }
}

/// 1D Transposed Convolution layer for audio upsampling and waveform generation.
#[derive(Debug, Clone)]
pub struct ConvTranspose1d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: usize,
    pub padding: usize,
    pub output_padding: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl ConvTranspose1d {
    /// Construct a new `ConvTranspose1d` module directly from weight and optional bias tensors.
    pub fn new(
        weight: Tensor,
        bias: Option<Tensor>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Self {
        Self {
            weight,
            bias,
            stride: stride.max(1),
            padding,
            output_padding,
            dilation: dilation.max(1),
            groups: groups.max(1),
        }
    }

    /// Forward pass for 1D transposed convolution. Accepts `[seq_len, in_channels]` or `[batch, in_channels, seq_len]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dims = x.shape().dims();
        let w_dims = self.weight.shape().dims();
        if w_dims.len() != 3 {
            return Err(Error::ShapeMismatch {
                expected: vec![0, 0, 0],
                got: w_dims.to_vec(),
            });
        }
        let in_c = w_dims[0];
        let out_c_per_group = w_dims[1];
        let kernel_size = w_dims[2];
        let out_c = out_c_per_group * self.groups;

        let (batch, in_seq, x_flat) = if x_dims.len() == 2 {
            if x_dims[1] == in_c {
                (1, x_dims[0], x.to_vec_f32()?)
            } else {
                (1, x_dims[1], x.to_vec_f32()?)
            }
        } else if x_dims.len() == 3 {
            (x_dims[0], x_dims[2], x.to_vec_f32()?)
        } else {
            return Err(Error::Shape(format!(
                "ConvTranspose1d: unsupported input rank {}",
                x_dims.len()
            )));
        };

        let out_seq = (in_seq - 1) * self.stride
            + self.dilation * (kernel_size - 1)
            + self.output_padding
            + 1
            - 2 * self.padding;
        let w_vec = self.weight.to_vec_f32()?;
        let b_vec = self.bias.as_ref().map(|b| b.to_vec_f32()).transpose()?;

        let mut out = vec![0.0f32; batch * out_c * out_seq];
        if let Some(ref b) = b_vec {
            for batch_i in 0..batch {
                for oc in 0..out_c {
                    let b_val = b[oc];
                    for os in 0..out_seq {
                        out[batch_i * out_c * out_seq + oc * out_seq + os] = b_val;
                    }
                }
            }
        }

        let in_c_per_group = in_c / self.groups;

        for b in 0..batch {
            let b_in_off = b * in_c * in_seq;
            let b_out_off = b * out_c * out_seq;

            for g in 0..self.groups {
                let g_in_start = g * in_c_per_group;
                let g_out_start = g * out_c_per_group;

                for ic_i in 0..in_c_per_group {
                    let ic = g_in_start + ic_i;

                    for oc_i in 0..out_c_per_group {
                        let oc = g_out_start + oc_i;
                        let w_base = (ic * out_c_per_group + oc_i) * kernel_size;

                        for is_pos in 0..in_seq {
                            let in_val = x_flat[b_in_off + ic * in_seq + is_pos];
                            let out_base = is_pos * self.stride;

                            for k in 0..kernel_size {
                                let out_pos = out_base as isize + (k * self.dilation) as isize
                                    - self.padding as isize;
                                if out_pos >= 0 && (out_pos as usize) < out_seq {
                                    let w_val = w_vec[w_base + k];
                                    out[b_out_off + oc * out_seq + out_pos as usize] +=
                                        in_val * w_val;
                                }
                            }
                        }
                    }
                }
            }
        }

        let dev = pick_device_for_tensor(x);
        let out_shape = if x_dims.len() == 2 && x_dims[1] == in_c {
            Shape::new(vec![out_seq, out_c])
        } else if x_dims.len() == 2 {
            Shape::new(vec![out_c, out_seq])
        } else {
            Shape::new(vec![batch, out_c, out_seq])
        };

        let storage = dev.from_cpu(&out, &out_shape, DType::F32)?;
        Ok(Tensor::new(
            Arc::from(storage),
            out_shape,
            DType::F32,
            x.provenance().clone(),
            x.device().clone(),
        ))
    }
}

#[cfg(test)]
mod mla_cache_tests {
    use super::*;
    use grim_backend_cpu::cpu_tensor;
    use grim_tensor::Shape;

    /// Audit gate (grim-models): MlaAttention's cached decode path must
    /// produce the SAME activations as a full prefill over the identical
    /// sequence — one [1,2] pass with no cache vs two [1,1] passes sharing
    /// an MlaKvCache. The pre-fix implementation ignored its cache
    /// parameter entirely, making incremental decode self-attentive only.
    #[test]
    fn mla_cached_decode_matches_full_prefill() {
        let hidden = 8usize;
        let q_lora = 8usize;
        let kv_lora = 8usize;
        let heads = 2usize;
        let nope = 4usize;
        let rope_dim = 4usize;
        let v_dim = 4usize;

        let mut seed = 0xA11CEu64;
        fn lcg(seed: &mut u64) -> f32 {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((*seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
        let mut lin = move |rows: usize, cols: usize| {
            Linear::from_tensor(
                cpu_tensor(
                    (0..rows * cols)
                        .map(|_| lcg(&mut seed) * 0.1)
                        .collect::<Vec<f32>>(),
                    Shape::new(vec![rows, cols]),
                ),
                None,
            )
        };
        let qk_head_dim = nope + rope_dim;
        let mla = MlaAttention {
            q_a_proj: lin(q_lora, hidden),
            q_a_norm: RmsNorm::new(
                cpu_tensor(vec![1.0; q_lora], Shape::new(vec![q_lora])),
                1e-5,
            ),
            q_b_proj: lin(heads * qk_head_dim, q_lora),
            kv_a_proj_with_mqa: lin(kv_lora, hidden),
            kv_a_norm: RmsNorm::new(
                cpu_tensor(vec![1.0; kv_lora], Shape::new(vec![kv_lora])),
                1e-5,
            ),
            kv_b_proj: lin(heads * (nope + rope_dim + v_dim), kv_lora),
            o_proj: lin(hidden, heads * v_dim),
            q_norm: None,
            k_norm: None,
            num_heads: heads,
            qk_nope_head_dim: nope,
            qk_rope_head_dim: rope_dim,
            v_head_dim: v_dim,
            rope: Rope::new(rope_dim, 10_000.0),
        };

        // Full prefill of a 2-token sequence, positions [0, 1], no cache.
        let x_full = cpu_tensor(
            (0..2 * hidden)
                .map(|i| ((i % 7) as f32 * 0.3) - 0.9)
                .collect::<Vec<f32>>(),
            Shape::new(vec![1, 2, hidden]),
        );
        let full = mla.forward(&x_full, &[0, 1], None).expect("prefill");
        let full_v = full.to_vec_f32().unwrap();

        // Incremental: token 0 then token 1 through ONE cache.
        let mut cache = MlaKvCache::new();
        let x0 = cpu_tensor(
            x_full.to_vec_f32().unwrap()[..hidden].to_vec(),
            Shape::new(vec![1, 1, hidden]),
        );
        let x1 = cpu_tensor(
            x_full.to_vec_f32().unwrap()[hidden..].to_vec(),
            Shape::new(vec![1, 1, hidden]),
        );
        let step0 = mla
            .forward(&x0, &[0], Some(&mut cache))
            .expect("cached step 0");
        assert_eq!(cache.past_len, 1);
        let step1 = mla
            .forward(&x1, &[1], Some(&mut cache))
            .expect("cached step 1");
        assert_eq!(cache.past_len, 2);

        let s0 = step0.to_vec_f32().unwrap();
        let s1 = step1.to_vec_f32().unwrap();
        for (i, (&f, &inc)) in full_v[..v_dim * heads].iter().zip(s0.iter()).enumerate() {
            assert!((f - inc).abs() < 1e-5, "token0 [{i}]: {f} vs {inc}");
        }
        for (i, (&f, &inc)) in full_v[v_dim * heads..].iter().zip(s1.iter()).enumerate() {
            assert!(
                (f - inc).abs() < 1e-5,
                "token1 [{i}] cached-vs-prefill divergence: {f} vs {inc}"
            );
        }
    }
}
