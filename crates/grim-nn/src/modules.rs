//! Module-style building blocks: linear, embedding, RMSNorm, RoPE.

use std::sync::Arc;

use grim_backend_cpu::CpuDevice;
#[cfg(feature = "metal-mem")]
use grim_backend_metal::MetalDevice;
#[cfg(feature = "vulkan-mem")]
use grim_backend_vulkan::VulkanDevice;
use grim_tensor::error::{Error, Result};
use grim_tensor::shape::Shape;
use grim_tensor::{BackendDevice, DType, Device, Tensor};

use crate::varbuilder::WeightSource;

#[cfg(feature = "cuda-mem")]
use grim_backend_cuda::CudaDevice;
#[cfg(feature = "rocm-mem")]
use grim_backend_rocm::RocmDevice;

/// Pick the `BackendDevice` that matches the storage location of `x` so
/// arithmetic ops dispatch to GPU kernels when the tensor lives on a GPU.
/// Falls back to CPU if the requested backend is unavailable in this build.
pub fn pick_device_for_tensor(x: &Tensor) -> Box<dyn BackendDevice> {
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

/// Pick a `BackendDevice` for a storage `Device` directly (without an
/// owning `Tensor`), used when reconstructing a tensor from CPU-side
/// bytes but needing to land it back on the original device.
/// Falls back to CPU if the requested backend is unavailable in this build.
pub fn pick_device_for_storage_device(d: &Device) -> Box<dyn BackendDevice> {
    match d {
        Device::Cpu => Box::new(CpuDevice::new()),
        #[cfg(feature = "cuda-mem")]
        Device::Cuda(ordinal) => {
            if let Ok(dev) = CudaDevice::new(*ordinal) {
                Box::new(dev)
            } else {
                Box::new(CpuDevice::new())
            }
        }
        #[cfg(feature = "rocm-mem")]
        Device::Rocm(ordinal) => {
            if let Ok(dev) = RocmDevice::try_new(*ordinal) {
                Box::new(dev)
            } else {
                Box::new(CpuDevice::new())
            }
        }
        #[cfg(feature = "vulkan-mem")]
        Device::Vulkan => Box::new(VulkanDevice::new()),
        #[cfg(feature = "metal-mem")]
        Device::Metal(ordinal) => {
            if let Ok(dev) = MetalDevice::new(*ordinal) {
                Box::new(dev)
            } else {
                Box::new(CpuDevice::new())
            }
        }
        // Fallback for backends not compiled in (arms above are cfg-gated).
        #[allow(unreachable_patterns)]
        _ => Box::new(CpuDevice::new()),
    }
}

/// Add two tensors element-wise with broadcasting, dispatching to the
/// device that owns `a`'s storage. This replaces the CPU-only
/// `grim_backend_cpu::add_tensors` which hardcodes `CpuDevice` and
/// panics ("storage is not CpuStorage") when called with ROCm tensors.
pub fn add_tensors(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let dev = pick_device_for_tensor(a);
    let (s, h) = BackendDevice::add(&*dev, a.storage().as_ref(), b.storage().as_ref(), a.shape())?;
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
            return Err(format!(
                "invalid TensorParallelConfig: world_size must be >= 1 (got 0)"
            ));
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
        let (out_s, h) = BackendDevice::matmul(&*dev, a_storage, b_storage, &out_shape)?;
        h.synchronize()?;
        let mat_out = Tensor::new(
            Arc::from(out_s),
            out_shape,
            DType::F32,
            x.provenance().clone(),
            x.device().clone(),
        );

        if let Some(b) = &self.bias {
            let broadcast_b = broadcast_bias(b, batch, out_dim)?;
            let (s, hh) = BackendDevice::add(
                &*dev,
                mat_out.storage().as_ref(),
                broadcast_b.storage().as_ref(),
                mat_out.shape(),
            )?;
            hh.synchronize()?;
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
    // Dequantize first so quantized GGUF weights (Q8_0, etc.) are genuinely
    // transposed to [in_dim, out_dim]. The previous `is_quantized()` early
    // return left w_t as the untransposed [out_dim, in_dim] storage; on ROCm
    // (where weights stay quantized after materialize) that made in_proj's
    // w_t=[3072,1024] instead of [1024,3072], so the matmul's k-dim check
    // failed with ShapeMismatch{expected:[12,1024], got:[3072,1024]}. On CPU
    // the GGUF loader dequantizes during materialize, so the early return was
    // simply never taken there — masking the bug. `to_vec_f32` dequantizes
    // quantized dtypes, so the path below is correct for both F32 and quant.
    let dims = t.shape().dims().to_vec();
    if dims.len() != 2 {
        return Err(Error::Shape("transpose_last_two: only 2-D".into()));
    }
    let (a, b) = (dims[0], dims[1]);
    let src = t.to_vec_f32()?;
    let mut out = vec![0.0f32; a * b];
    for i in 0..a {
        for j in 0..b {
            out[j * a + i] = src[i * b + j];
        }
    }
    let new_shape = Shape::new(vec![b, a]);
    if t.device().is_cpu() {
        Ok(grim_backend_cpu::cpu_tensor(out, new_shape))
    } else {
        // Re-upload transposed (dequantized) weights back to the source
        // device so the downstream matmul sees matching device storages and
        // the correct [in_dim, out_dim] layout.
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

fn broadcast_bias(b: &Tensor, batch: usize, out_dim: usize) -> Result<Tensor> {
    let b_vec = b.to_vec_f32()?;
    let mut out = Vec::with_capacity(batch * out_dim);
    for _ in 0..batch {
        out.extend_from_slice(&b_vec);
    }
    if out.len() != batch * out_dim {
        return Err(Error::Shape("broadcast_bias: size mismatch".into()));
    }
    let new_shape = Shape::new(vec![batch, out_dim]);
    if b.device().is_cpu() {
        Ok(grim_backend_cpu::cpu_tensor(out, new_shape))
    } else {
        let dev = pick_device_for_tensor(b);
        let storage = dev.from_cpu(&out, &new_shape, DType::F32)?;
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
    pub fn load(ws: &WeightSource<'_>, dim: usize, eps: f32) -> Result<Self> {
        let weight = ws.get([dim], "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dev = pick_device_for_tensor(x);
        let dim = x.shape().dims().last().copied().unwrap_or(0);
        let batch = x.shape().elem_count() / dim;
        let out_shape = Shape::new(vec![batch, dim]);
        let (s, h) = BackendDevice::rms_norm(
            &*dev,
            x.storage().as_ref(),
            self.weight.storage().as_ref(),
            self.eps,
            &out_shape,
        )?;
        h.synchronize()?;
        Ok(Tensor::new(
            Arc::from(s),
            out_shape,
            DType::F32,
            x.provenance().clone(),
            x.device().clone(),
        ))
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
        // Probe `[vocab, dim]`. On exact shape match, use as-is.
        if let Ok(t) = ws.get([vocab, dim], "weight") {
            let weight = if t.dtype().is_quantized() {
                // Embedding kernel expects f32 weights; dequantize if quantized.
                eprintln!(
                    "[Embedding::load] weight is quantized: dtype={:?}, device={:?}",
                    t.dtype(),
                    t.device()
                );
                let f32s = t.to_vec_f32()?;
                eprintln!("[Embedding::load] dequantized to f32, len={}", f32s.len());
                let shape = t.shape().clone();
                let dev = pick_device_for_tensor(&t);
                let storage = dev.from_cpu(&f32s, &shape, DType::F32)?;
                Tensor::new(
                    Arc::from(storage),
                    shape,
                    DType::F32,
                    t.provenance().clone(),
                    t.device().clone(),
                )
            } else {
                t
            };
            return Ok(Self { weight });
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
            return Ok(Self { weight: raw_tensor });
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
            BackendDevice::embedding(&*dev, self.weight.storage().as_ref(), indices, &out_shape)?;
        h.synchronize()?;
        Ok(Tensor::new(
            Arc::from(s),
            out_shape,
            DType::F32,
            self.weight.provenance().clone(),
            self.weight.device().clone(),
        ))
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
}

// ---------- RoPE ----------

/// Rotary positional embedding — apply RoPE to `(B, S, D)` query/key.
#[derive(Debug, Clone, Copy)]
pub struct Rope {
    pub dim: usize,
    pub base: f32,
}

impl Rope {
    pub fn new(dim: usize, base: f32) -> Self {
        Self { dim, base }
    }

    pub fn forward(&self, x: &Tensor, positions: &[u32]) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        if dims.len() != 3 || dims[2] != self.dim {
            return Err(Error::Shape(format!(
                "RoPE expects (B,S,D={}), got {:?}",
                self.dim, dims
            )));
        }
        let (b, s, d) = (dims[0], dims[1], dims[2]);
        let half = d / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / self.base.powf((2 * i) as f32 / d as f32))
            .collect();
        let mut src = x.to_vec_f32()?;
        for bi in 0..b {
            for si in 0..s {
                let pos = positions.get(si).copied().unwrap_or(si as u32) as f32;
                let base_index = (bi * s + si) * d;
                let mut cos_p = vec![0.0f32; half];
                let mut sin_p = vec![0.0f32; half];
                for i in 0..half {
                    let a = pos * inv_freq[i];
                    cos_p[i] = a.cos();
                    sin_p[i] = a.sin();
                }
                for i in 0..half {
                    let xi = base_index + i;
                    let xj = base_index + half + i;
                    let a = src[xi];
                    let bv = src[xj];
                    src[xi] = a * cos_p[i] - bv * sin_p[i];
                    src[xj] = bv * cos_p[i] + a * sin_p[i];
                }
            }
        }
        let out_shape = Shape::new(vec![b, s, d]);
        if x.device().is_cpu() {
            Ok(grim_backend_cpu::cpu_tensor(src, out_shape))
        } else {
            let dev = pick_device_for_tensor(x);
            let storage = dev.from_cpu(&src, &out_shape, DType::F32)?;
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
}
