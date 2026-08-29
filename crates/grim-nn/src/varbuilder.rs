//! `WeightSource` — depth-first cursor over a `TensorProvider`.
//!
//! Mirrors Candle's `VarBuilder` exactly: every model constructor walks a
//! config-defined layer hierarchy and pulls tensors by prefix. Per-tensor
//! dtype/provenance resolution (§4.2, §7.2) happens in `get()`.

use std::sync::Arc;

use grim_tensor::dtype::{
    BlockDtype, DType, Device, FloatPackScheme, KQuantScheme, QuantProvenance, Storage,
};
use grim_tensor::error::{Error, Result};
use grim_tensor::shape::Shape;
use grim_tensor::tensor::Tensor;
use grim_tensor::{RawTensor,
    CoreTensorOps, MemoryOps,
};

use grim_backend_cpu::{CpuDevice, cpu_tensor};
use grim_quant::{
    dequant_fp4, dequant_fp4_block16, dequant_fp8, dequant_fp8_block16, dequant_iq2s,
    dequant_iq2xs, dequant_iq2xxs, dequant_iq3s, dequant_iq3xxs, dequant_iq4nl, dequant_iq4xs,
    dequant_mxfp4, dequant_mxfp8, dequant_nf4, dequant_q2k, dequant_q3k, dequant_q4k, dequant_q5k,
    dequant_q6k, dequant_q80,
};

#[cfg(feature = "cuda-mem")]
use grim_backend_cuda::CudaDevice;
#[cfg(feature = "metal-mem")]
use grim_backend_metal::MetalDevice;
#[cfg(feature = "rocm-mem")]
use grim_backend_rocm::RocmDevice;
#[cfg(feature = "rocm-mem")]
#[allow(unused_imports)]
use grim_backend_rocm::RocmStorage;
#[cfg(feature = "vulkan-mem")]
use grim_backend_vulkan::VulkanDevice;

use crate::TensorParallelConfig;

/// A handle that walks a `TensorProvider` by hierarchical prefix. Models
/// call `ws.pp("model").pp("layers").pp("0").get(...)` to materialize
/// tensors; the call-site shape determines what storage type comes back.
pub struct WeightSource<'a> {
    tensors: &'a dyn grim_tensor::TensorProvider,
    prefix: Vec<String>,
    default_dtype: DType,
    default_provenance: QuantProvenance,
    device: Device,
    /// Tensor-parallel config for sharded weight loading. Defaults to
    /// single-device (rank 0, world_size 1).
    tp_config: TensorParallelConfig,
    /// Parallel-prefetch cache: tensors fetched + per-tensor CPU passes
    /// (e.g. MXFP4 reframing) done ahead of the layer loop by `prefetch_all`.
    /// Keyed by the *full* (prefixed) tensor name so it matches what `get`
    /// requests. Shared across `pp`/`with_tp_config` clones via `Arc`.
    prefetch_cache: std::sync::Arc<
        parking_lot::Mutex<
            std::collections::HashMap<String, std::sync::Arc<grim_tensor::provider::RawTensor>>,
        >,
    >,
    /// Host-f32 dequant cache, populated by [`prefetch_all`] in the parallel
    /// worker for formats that the `materialize` path dequantizes to host f32
    /// (native F32/BF16/F16 and GroupInt; quantized-resident GPU formats like
    /// KQuant/FloatPack on ROCm/CUDA stay packed and are excluded). Keyed by
    /// full (prefixed) name, matching `prefetch_cache`. When present, `get_f32`
    /// and the host-dequant `materialize` branch reuse it instead of re-running
    /// `dequant_to_f32` on the layer-construction thread — overlapping the
    /// CPU dequant cost with disk I/O and device uploads.
    dequant_cache: std::sync::Arc<
        parking_lot::Mutex<std::collections::HashMap<String, std::sync::Arc<Vec<f32>>>>,
    >,
}

impl<'a> WeightSource<'a> {
    pub fn new(
        tensors: &'a dyn grim_tensor::TensorProvider,
        default_dtype: DType,
        default_provenance: QuantProvenance,
        device: Device,
    ) -> Self {
        Self {
            tensors,
            prefix: Vec::new(),
            default_dtype,
            default_provenance,
            device,
            tp_config: TensorParallelConfig::default(),
            prefetch_cache: std::sync::Arc::new(parking_lot::Mutex::new(
                std::collections::HashMap::new(),
            )),
            dequant_cache: std::sync::Arc::new(parking_lot::Mutex::new(
                std::collections::HashMap::new(),
            )),
        }
    }

    /// Root-level builder from a `TensorProvider`.
    pub fn root(tensors: &'a dyn grim_tensor::TensorProvider, device: Device) -> Self {
        Self::new(tensors, DType::F32, QuantProvenance::GrimNative, device)
    }

    /// Attach a `TensorParallelConfig` for sharded weight loading.
    pub fn with_tp_config(&self, tp: TensorParallelConfig) -> WeightSource<'a> {
        WeightSource {
            tensors: self.tensors,
            prefix: self.prefix.clone(),
            default_dtype: self.default_dtype.clone(),
            default_provenance: self.default_provenance.clone(),
            device: self.device.clone(),
            tp_config: tp,
            prefetch_cache: self.prefetch_cache.clone(),
            dequant_cache: self.dequant_cache.clone(),
        }
    }

    /// Return a new `WeightSource` targeting a different `device`.
    pub fn with_device(&self, device: Device) -> WeightSource<'a> {
        WeightSource {
            tensors: self.tensors,
            prefix: self.prefix.clone(),
            default_dtype: self.default_dtype.clone(),
            default_provenance: self.default_provenance.clone(),
            device,
            tp_config: self.tp_config,
            prefetch_cache: self.prefetch_cache.clone(),
            dequant_cache: self.dequant_cache.clone(),
        }
    }

    /// Parallel-prefetch every tensor the underlying provider can name.
    ///
    /// Reads + per-tensor CPU passes (MXFP4 reframing, etc.) run concurrently
    /// across a rayon worker pool instead of serially behind the provider's
    /// read path, and the results are cached keyed by full (prefixed) name.
    /// This is the primary load-time speedup: by the time the layer-
    /// construction loop calls `get`, each `RawTensor` is already resident in
    /// the cache and only the device upload remains.
    ///
    /// Tensors that fail to fetch are skipped (left for on-demand fallback),
    /// so a partial provider (or a name the model doesn't reference) never
    /// aborts the load.
    pub fn prefetch_all(&self) {
        let names = self.tensors.tensor_names();
        if names.is_empty() {
            return;
        }
        let cache = self.prefetch_cache.clone();
        // De-dupe against already-cached entries without holding the lock
        // across the (potentially slow) fetch.
        let pending: Vec<String> = {
            let guard = cache.lock();
            names
                .into_iter()
                .filter(|n| !guard.contains_key(n))
                .collect()
        };
        if pending.is_empty() {
            return;
        }
        use rayon::prelude::*;
        let dq_cache = self.dequant_cache.clone();
        pending.par_iter().for_each(|name| {
            match self.tensors.get_packed(name) {
                Ok(raw) => {
                    cache
                        .lock()
                        .insert(name.clone(), std::sync::Arc::new(raw.clone()));
                    // Move the CPU dequant into the worker so it overlaps the
                    // remaining disk I/O and device uploads (task: dequant in
                    // prefetch worker). Only formats that `materialize` always
                    // dequantizes to host f32 (native F32/BF16/F16, GroupInt,
                    // ResidualPacked) are pre-dequantized — quantized KQuant /
                    // FloatPack / Block formats stay packed-resident on
                    // ROCm/CUDA/CPU, so their dequant is left to the device-
                    // aware `materialize` path. A dequant failure here is a
                    // cache miss (on-demand path recomputes it) — never an
                    // abort.
                    let storage = &raw.dtype.storage;
                    let host_f32_route = matches!(
                        storage,
                        Storage::Native | Storage::GroupInt(_) | Storage::ResidualPacked(_)
                    );
                    if host_f32_route {
                        let dtype = raw.dtype.clone();
                        if let Ok(f32s) = dequant_to_f32(&raw, &dtype) {
                            dq_cache
                                .lock()
                                .insert(name.clone(), std::sync::Arc::new(f32s));
                        }
                    }
                }
                Err(_) => { /* leave for on-demand fetch */ }
            }
        });
    }

    /// Look up a host-dequantized f32 buffer produced by the prefetch worker.
    /// Returns `None` when the tensor was not pre-dequantized (either not
    /// prefetched, or a packed-resident format whose dequant is device-aware).
    fn prefetched_f32(&self, name: &str) -> Option<std::sync::Arc<Vec<f32>>> {
        self.dequant_cache.lock().get(name).cloned()
    }

    /// Fetch a raw tensor, consulting the prefetch cache first.
    fn cached_raw(&self, name: &str) -> Result<grim_tensor::provider::RawTensor> {
        if let Some(arc) = self.prefetch_cache.lock().get(name) {
            return Ok((**arc).clone());
        }
        let raw = self.tensors.get_packed(name)?;
        self.prefetch_cache
            .lock()
            .insert(name.to_string(), std::sync::Arc::new(raw.clone()));
        Ok(raw)
    }

    /// Read-only access to the current TP config.
    pub fn tp_config(&self) -> TensorParallelConfig {
        self.tp_config
    }

    /// Fetch the rank-th shard of a tensor (delegates to the underlying
    /// provider's `get_packed_sharded`, which may do zero-copy byte-range reads
    /// for GGUF block-quant formats).
    pub fn get_sharded(&self, shape: impl Into<Shape>, leaf: &str, dim: usize) -> Result<Tensor> {
        let shape = shape.into();
        let name = self.full_name(leaf);
        let raw = self.cached_raw_sharded(&name, dim)?;
        let (dtype, provenance) = match self.tensors.meta(&name) {
            Ok(m) => (m.dtype, m.provenance),
            Err(_) => (self.default_dtype.clone(), self.default_provenance.clone()),
        };
        materialize(raw, shape, dtype, provenance, &self.device)
    }

    /// Fetch the rank-th shard without enforcing an expected shape (used by
    /// callers that need to inspect the loaded shape first).
    pub fn get_unconstrained_sharded(&self, leaf: &str, dim: usize) -> Result<Tensor> {
        let name = self.full_name(leaf);
        let raw = self.cached_raw_sharded(&name, dim)?;
        let shape = Shape::new(raw.shape.clone());
        let (dtype, provenance) = match self.tensors.meta(&name) {
            Ok(m) => (m.dtype, m.provenance),
            Err(_) => (self.default_dtype.clone(), self.default_provenance.clone()),
        };
        materialize(raw, shape, dtype, provenance, &self.device)
    }

    /// Resolve a rank shard for a tensor, consulting the parallel-prefetch
    /// cache first.
    ///
    /// `prefetch_all` fetches the *full* tensor (zero-copy out of the mmap) and
    /// caches the packed bytes by full name. When a shard is requested for a
    /// name the prefetch has already made resident, we shard that cached full
    /// tensor client-side instead of issuing a second provider read — so TP
    /// (`get_sharded`) rides the same parallel prefetch the single-device
    /// (`get`) path does.
    ///
    /// Block-quant GGUF formats keep their provider-specific byte-range read
    /// (block boundaries do not align with clean f32 shard slices), so for
    /// those we fall through to `get_packed_sharded` exactly as before.
    fn cached_raw_sharded(
        &self,
        name: &str,
        dim: usize,
    ) -> Result<grim_tensor::provider::RawTensor> {
        let full = self
            .prefetch_cache
            .lock()
            .get(name)
            .map(|arc| (**arc).clone());
        if let Some(full_raw) = full {
            if full_raw.dtype.storage == grim_tensor::dtype::Storage::Native {
                return grim_tensor::provider::shard_raw_tensor(
                    full_raw,
                    dim,
                    self.tp_config.rank,
                    self.tp_config.world_size,
                );
            }
            // Quantized (block-packed) full tensor: the client-side byte layout
            // cannot be sliced safely; fall through to the provider override.
        }
        self.tensors
            .get_packed_sharded(name, dim, self.tp_config.rank, self.tp_config.world_size)
    }

    /// Push a path segment and return a new `WeightSource` whose prefix is
    /// `self.prefix + [name]`. Mirrors `candle::VarBuilder::pp`.
    pub fn pp(&self, name: &str) -> WeightSource<'a> {
        let mut next = self.clone_prefix();
        next.prefix.push(name.to_owned());
        next
    }

    /// Alias for `pp` to scope sub-modules.
    pub fn scoped(&self, name: &str) -> WeightSource<'a> {
        self.pp(name)
    }

    /// Returns whether a tensor with the given leaf name exists in this weight source.
    pub fn has_tensor(&self, leaf: &str) -> bool {
        let name = self.full_name(leaf);
        self.tensors.meta(&name).is_ok()
    }

    fn clone_prefix(&self) -> WeightSource<'a> {
        WeightSource {
            tensors: self.tensors,
            prefix: self.prefix.clone(),
            default_dtype: self.default_dtype.clone(),
            default_provenance: self.default_provenance.clone(),
            device: self.device.clone(),
            tp_config: self.tp_config,
            prefetch_cache: self.prefetch_cache.clone(),
            dequant_cache: self.dequant_cache.clone(),
        }
    }

    /// Returns the target device for this WeightSource (CPU or GPU).
    pub fn device(&self) -> Device {
        self.device.clone()
    }

    /// Fetch the raw packed bytes for a tensor BEFORE materialization. Used by
    /// loaders that slice a 3D expert bank into per-expert tensors so each
    /// expert can be materialized (quantized-resident on GPU) individually
    /// instead of dequantizing the whole bank to host f32.
    pub fn get_raw_packed(&self, leaf: &str) -> Result<grim_tensor::provider::RawTensor> {
        let name = self.full_name(leaf);
        self.cached_raw(&name)
    }

    /// Materialize an already-fetched `RawTensor` on this WeightSource's
    /// device (quantized formats stay packed/resident on GPU; native formats
    /// become f32 tensors). Crate-internal helper shared by module loaders.
    pub(crate) fn materialize_raw(
        &self,
        raw: grim_tensor::provider::RawTensor,
        shape: Shape,
    ) -> Result<Tensor> {
        let dtype = raw.dtype.clone();
        let provenance = raw.provenance.clone();
        materialize(raw, shape, dtype, provenance, &self.device)
    }

    fn full_name(&self, leaf: &str) -> String {
        let mut s = self.prefix.join(".");
        if !s.is_empty() {
            s.push('.');
        }
        s.push_str(leaf);
        s
    }

    /// Materialize a tensor of the given `shape` and `leaf` name under the
    /// current prefix. Resolves dtype + provenance per-tensor: first from
    /// the checkpoint's per-tensor metadata, then falls back to defaults.
    pub fn get(&self, shape: impl Into<Shape>, leaf: &str) -> Result<Tensor> {
        let shape = shape.into();
        let name = self.full_name(leaf);
        let raw = self.cached_raw(&name)?;

        if raw.shape != shape.dims() {
            eprintln!(
                "[get-trace] ShapeMismatch at name={} expected={:?} got={:?}",
                name,
                shape.dims(),
                raw.shape
            );
            return Err(Error::ShapeMismatch {
                expected: shape.dims().to_vec(),
                got: raw.shape.clone(),
            });
        }
        let (dtype, provenance) = match self.tensors.meta(&name) {
            Ok(m) => (m.dtype, m.provenance),
            Err(_) => (self.default_dtype.clone(), self.default_provenance.clone()),
        };

        materialize(raw, shape, dtype, provenance, &self.device)
    }

    /// Materialize a tensor without enforcing an expected shape up-front.
    ///
    /// This contract enables callers to perform dynamic shape inspection and
    /// layout normalization (for instance, [`Embedding::load`] handling
    /// transposed weights or token dimension padding in model checkpoints).
    pub fn get_unconstrained(&self, leaf: &str) -> Result<Tensor> {
        let name = self.full_name(leaf);
        let raw = self.cached_raw(&name)?;
        let shape = Shape::new(raw.shape.clone());
        let (dtype, provenance) = match self.tensors.meta(&name) {
            Ok(m) => (m.dtype, m.provenance),
            Err(_) => (self.default_dtype.clone(), self.default_provenance.clone()),
        };

        materialize(raw, shape, dtype, provenance, &self.device)
    }

    /// Materialize a tensor as an F32 tensor on the target device, dequantizing
    /// on the host exactly once.
    ///
    /// This is the single-transfer alternative to loading a quantized weight
    /// with [`get`] (which keeps packed bytes resident on-device for GPU) and
    /// then pulling it back with `to_vec_f32()` + re-upload (`dequantize_for_gather`'s
    /// DtoH→H2D round trip). `get_f32` dequantizes the packed bytes to a host
    /// `Vec<f32>` and uploads that host f32 buffer once (one H2D, zero DtoH).
    /// Used by embedding tables, whose lookup kernels read F32 rows, and by any
    /// gather source that cannot consume quantized-resident storage.
    pub fn get_f32(&self, shape: impl Into<Shape>, leaf: &str) -> Result<Tensor> {
        let shape = shape.into();
        let name = self.full_name(leaf);
        let raw = self.cached_raw(&name)?;
        if raw.shape != shape.dims() {
            return Err(Error::ShapeMismatch {
                expected: shape.dims().to_vec(),
                got: raw.shape.clone(),
            });
        }
        let (dtype, provenance) = match self.tensors.meta(&name) {
            Ok(m) => (m.dtype, m.provenance),
            Err(_) => (self.default_dtype.clone(), self.default_provenance.clone()),
        };
        // Dequantize to a host f32 buffer. Reuse the prefetch worker's result
        // when available (dequant already overlapped with disk I/O + other
        // uploads); otherwise dequant here.
        let f32s: Vec<f32> = if let Some(cached) = self.prefetched_f32(&name) {
            (*cached).clone()
        } else {
            dequant_to_f32(&raw, &dtype)?
        };
        if self.device.is_cpu() {
            return Ok(cpu_tensor(f32s, shape));
        }
        // ROCm: use the stream-ordered (async, non-blocking) upload path so the
        // per-tensor H2D copy queues on the stream pool and overlaps with the
        // dequant + upload of the next tensor, instead of blocking the layer
        // loop. The pinned host buffer is retained and freed at the next device
        // synchronize (model load syncs before the first inference step).
        #[cfg(feature = "rocm-mem")]
        if let Device::Rocm(ordinal) = self.device {
            let dev = grim_backend_rocm::RocmDevice::shared(ordinal);
            let storage = dev.upload_from_host_stream_ordered(&f32s, &shape, DType::F32)?;
            return Ok(Tensor::new(
                std::sync::Arc::from(storage),
                shape,
                DType::F32,
                provenance,
                self.device.clone(),
            ));
        }
        let dev = crate::modules::pick_device_for_storage_device(&self.device);
        let storage = dev.from_cpu(&f32s, &shape, DType::F32)?;
        Ok(Tensor::new(
            std::sync::Arc::from(storage),
            shape,
            DType::F32,
            provenance,
            self.device.clone(),
        ))
    }

    /// Materialize a tensor for training. Quantized storage types (Q4_K, Q5_K,
    /// Q6_K, Q8_0, ...) are dequantized to native F32 in CPU memory so the
    /// optimization pass has full-precision weights to take gradients against.
    /// Native dtypes flow through unchanged.
    pub fn get_for_training(&self, shape: impl Into<Shape>, leaf: &str) -> Result<Tensor> {
        let shape = shape.into();
        let name = self.full_name(leaf);
        let raw = self.tensors.get_packed(&name)?;

        if raw.shape != shape.dims() {
            return Err(Error::ShapeMismatch {
                expected: shape.dims().to_vec(),
                got: raw.shape.clone(),
            });
        }
        let (dtype, provenance) = match self.tensors.meta(&name) {
            Ok(m) => (m.dtype, m.provenance),
            Err(_) => (self.default_dtype.clone(), self.default_provenance.clone()),
        };

        materialize(raw, shape, dtype, provenance, &self.device)
    }
}

// Materialization helpers — each arm is a self-contained branch so cfg(...)
// attributes on `use` statements don't create non-exhaustive match arms.

#[cfg(feature = "cuda-mem")]
fn materialize_cuda(
    f32s: Vec<f32>,
    shape: Shape,
    _dtype: DType,
    provenance: QuantProvenance,
    device: &Device,
    ordinal: usize,
) -> Result<Tensor> {
    let dev = CudaDevice::new(ordinal)?;
    // Storage is F32 bytes regardless of the GGUF-stored quantization tag:
    // `f32s` was already dequantized in `materialize` above. Pass F32 to
    // `from_cpu` so the CUDA storage carries DType::F32, which downstream
    // embedding/matmul kernels require.
    let storage = CoreTensorOps::from_cpu(&dev, &f32s, &shape, DType::F32)?;
    Ok(Tensor::new(
        Arc::from(storage),
        shape,
        DType::F32,
        provenance,
        device.clone(),
    ))
}

#[cfg(not(feature = "cuda-mem"))]
fn materialize_cuda(
    _f32s: Vec<f32>,
    _shape: Shape,
    _dtype: DType,
    _provenance: QuantProvenance,
    _device: &Device,
    ordinal: usize,
) -> Result<Tensor> {
    Err(Error::Unimplemented(format!(
        "CUDA materialization: enable 'cuda-mem' feature on grim-nn (ordinal={})",
        ordinal
    )))
}

#[cfg(feature = "rocm-mem")]
fn rocm_managed_weight_mode(ordinal: usize, bytes: usize) -> bool {
    match std::env::var("GRIM_ROCM_MANAGED_WEIGHTS").ok().as_deref() {
        Some("1") | Some("true") | Some("always") => true,
        Some("auto") => {
            let (free, total) = grim_backend_rocm::vram_info(ordinal);
            let budget = std::env::var("GRIM_ROCM_VRAM_BUDGET_BYTES")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or_else(|| total.saturating_mul(9) / 10);
            free < bytes as u64 || total.saturating_sub(free) > budget
        }
        _ => false,
    }
}

#[cfg(feature = "rocm-mem")]
fn materialize_rocm(
    f32s: Vec<f32>,
    shape: Shape,
    _dtype: DType,
    provenance: QuantProvenance,
    device: &Device,
    ordinal: usize,
) -> Result<Tensor> {
    let dev = RocmDevice::shared(ordinal);
    // Storage is F32 bytes (already dequantized in `materialize`). Mirror
    // CUDA: stamp the storage as DType::F32 so ROCm kernels that check
    // input dtype (embedding, matmul) accept the result.
    // Opt-in unified-memory residency for large model weights. HIP kernels
    // can dereference this storage normally; the runtime migrates pages
    // between VRAM and system RAM. Keep the default on ordinary VRAM until
    // a global budget policy is selected by the caller.
    let managed = rocm_managed_weight_mode(ordinal, f32s.len() * std::mem::size_of::<f32>());
    let storage = if managed {
        dev.from_cpu_managed(&f32s, &shape, DType::F32)?
    } else {
        CoreTensorOps::from_cpu(dev.as_ref(), &f32s, &shape, DType::F32)?
    };
    Ok(Tensor::new(
        Arc::from(storage),
        shape,
        DType::F32,
        provenance,
        device.clone(),
    ))
}

#[cfg(not(feature = "rocm-mem"))]
fn materialize_rocm(
    _f32s: Vec<f32>,
    _shape: Shape,
    _dtype: DType,
    _provenance: QuantProvenance,
    _device: &Device,
    ordinal: usize,
) -> Result<Tensor> {
    Err(Error::Unimplemented(format!(
        "ROCm materialization: enable 'rocm-mem' feature on grim-nn (ordinal={})",
        ordinal
    )))
}

#[cfg(feature = "metal-mem")]
fn materialize_metal(
    f32s: Vec<f32>,
    shape: Shape,
    _dtype: DType,
    provenance: QuantProvenance,
    device: &Device,
    ordinal: usize,
) -> Result<Tensor> {
    let dev = MetalDevice::try_new(ordinal)?;
    let storage = CoreTensorOps::from_cpu(&dev, &f32s, &shape, DType::F32)?;
    Ok(Tensor::new(
        Arc::from(storage),
        shape,
        DType::F32,
        provenance,
        device.clone(),
    ))
}

#[cfg(not(feature = "metal-mem"))]
fn materialize_metal(
    _f32s: Vec<f32>,
    _shape: Shape,
    _dtype: DType,
    _provenance: QuantProvenance,
    _device: &Device,
    ordinal: usize,
) -> Result<Tensor> {
    Err(Error::Unimplemented(format!(
        "Metal materialization: enable 'metal-mem' feature on grim-nn (ordinal={})",
        ordinal
    )))
}

#[cfg(feature = "vulkan-mem")]
fn materialize_vulkan(
    f32s: Vec<f32>,
    shape: Shape,
    _dtype: DType,
    provenance: QuantProvenance,
    device: &Device,
) -> Result<Tensor> {
    let dev = VulkanDevice::new();
    let storage = CoreTensorOps::from_cpu(&dev, &f32s, &shape, DType::F32)?;
    Ok(Tensor::new(
        Arc::from(storage),
        shape,
        DType::F32,
        provenance,
        device.clone(),
    ))
}

#[cfg(not(feature = "vulkan-mem"))]
fn materialize_vulkan(
    _f32s: Vec<f32>,
    _shape: Shape,
    _dtype: DType,
    _provenance: QuantProvenance,
    _device: &Device,
) -> Result<Tensor> {
    Err(Error::Unimplemented(
        "Vulkan materialization: enable 'vulkan-mem' feature on grim-nn".into(),
    ))
}

fn materialize(
    raw: RawTensor,
    shape: Shape,
    dtype: DType,
    provenance: QuantProvenance,
    device: &Device,
) -> Result<Tensor> {
    // ROCm-only fast path for GPTQ/EfficientQAT `GroupInt` weights. Keeps the
    // packed four-segment blob resident on-device so the engine's
    // `Linear::forward` -> `quantized_matmul` reads it directly through the
    // fused `grim_gptq_dequant_gemm` kernel (roc_device.rs GroupInt arm),
    // instead of inflating every tensor to host f32 at load.
    //
    // Scoped to ROCm: the ROCm backend is the only one whose
    // `quantized_matmul` has a `GroupInt` dispatch arm backed by a live GPU
    // kernel. The CPU/Vulkan/Metal/CUDA backends lack that arm, so for them
    // `GroupInt` intentionally falls through to the host-f32 dequant below.
    // 3-bit GroupInt has no fused device kernel (segment-offset + kernel only
    // support 2/4/8-bit), so it also falls through to host f32 here and is
    // served by `dequant_gptq_group_int` at load.
    #[cfg(feature = "rocm-mem")]
    if let Device::Rocm(ordinal) = device {
        if let Storage::GroupInt(cfg) = &dtype.storage {
            if matches!(cfg.bits, 2 | 4 | 8) {
                let dev = grim_backend_rocm::RocmDevice::shared(*ordinal);
                let mut storage = dev.from_cpu_bytes(&raw.bytes, &shape, dtype.clone())?;
                storage.set_provenance(provenance.clone());
                return Ok(Tensor::new(
                    std::sync::Arc::from(storage),
                    shape,
                    dtype,
                    provenance,
                    device.clone(),
                ));
            }
        }
    }

    if dtype.is_quantized()
        && !matches!(dtype.storage, Storage::ResidualPacked(_))
        && !matches!(dtype.storage, Storage::GroupInt(_))
    {
        #[cfg(feature = "rocm-mem")]
        if let Device::Rocm(ordinal) = device {
            // Priority 2 diagnostic (opt-in): set GRIM_ROCM_FASTPATH_TRACE to
            // confirm which path each quantized tensor takes on ROCm. Left in
            // as a debug aid — silent otherwise so it never spams large MoE
            // loads (thousands of expert tensors).
            if std::env::var_os("GRIM_ROCM_FASTPATH_TRACE").is_some() {
                eprintln!(
                    "[grim][fastpath] ROCm on-device packed residency for {:?} ({} elems) on ordinal {ordinal}",
                    dtype.storage,
                    raw.shape.iter().product::<usize>()
                );
            }
            let dev = RocmDevice::shared(*ordinal);
            let mut storage = dev.from_cpu_bytes(&raw.bytes, &shape, dtype.clone())?;
            storage.set_provenance(provenance.clone());
            return Ok(Tensor::new(
                Arc::from(storage),
                shape,
                dtype,
                provenance,
                device.clone(),
            ));
        }

        // CUDA: keep KQuant / FloatPack / Block storage resident on-device
        // (raw packed bytes) instead of dequantizing to F32 at load time. This
        // mirrors ROCm's Q8_0 residency and enables the CUDA fused
        // `quantized_matmul_backward_dx` path in `grim-autograd::matmul_backward`.
        // ResidualPacked is excluded (no host dequant exists); GroupInt is left
        // dequantized-to-F32 to preserve its multi-segment loader semantics.
        #[cfg(feature = "cuda-mem")]
        if let Device::Cuda(ordinal) = device {
            let dev = CudaDevice::new(*ordinal)?;
            let storage = dev.from_cpu_bytes(&raw.bytes, &shape, dtype.clone())?;
            return Ok(Tensor::new(
                Arc::from(storage),
                shape,
                dtype,
                provenance,
                device.clone(),
            ));
        }

        #[cfg(not(any(feature = "rocm-mem", feature = "cuda-mem")))]
        {
            let _ = (&raw, &shape, &dtype);
        }

        // CPU: keep quantized (KQuant / FloatPack / Block) bytes resident in
        // system memory rather than decompressing to F32 at load time. The CPU
        // `quantized_matmul` path dequantizes on-the-fly, so a 12B MXFP4 model
        // stays ~6GB in RAM instead of inflating to ~48GB F32 and blowing past
        // system RAM / thrashing swap. Mirrors the ROCm/CUDA residency semantics.
        if let Device::Cpu = device {
            let dev = CpuDevice::new();
            let storage = dev.from_cpu_bytes(&raw.bytes, &shape, dtype.clone())?;
            return Ok(Tensor::new(
                Arc::from(storage),
                shape,
                dtype,
                provenance,
                device.clone(),
            ));
        }
    }

    if device.is_cpu() {
        let f32s = dequant_to_f32(&raw, &dtype)?;
        return Ok(cpu_tensor(f32s, shape));
    }

    let f32s = {
        // Priority 2 diagnostic (opt-in): this is the host-side scalar dequant
        // fallthrough. If GRIM_ROCM_FASTPATH_TRACE is set and this fires for an
        // MXFP4/FloatPack tensor, the ROCm fast path was NOT taken for it.
        if std::env::var_os("GRIM_ROCM_FASTPATH_TRACE").is_some() {
            eprintln!(
                "[grim][fastpath] host dequant fallthrough for {:?} ({} elems)",
                dtype.storage,
                raw.shape.iter().product::<usize>()
            );
        }
        dequant_to_f32(&raw, &dtype)?
    };
    match device {
        Device::Cpu => Err(Error::Backend(
            "Device::Cpu reached after is_cpu early-return — unreachable".into(),
        )),
        Device::Cuda(ordinal) => materialize_cuda(f32s, shape, dtype, provenance, device, *ordinal),
        Device::Rocm(ordinal) => materialize_rocm(f32s, shape, dtype, provenance, device, *ordinal),
        Device::Vulkan => materialize_vulkan(f32s, shape, dtype, provenance, device),
        Device::Metal(ordinal) => {
            materialize_metal(f32s, shape, dtype, provenance, device, *ordinal)
        }
    }
}

/// Materialize any supported storage format to a flat `Vec<f32>` of
/// `raw.shape` length. This is the single dequant dispatch shared by the
/// inference `get()` path (and mirrors the training `get_for_training`
/// layout). Supports native F32/BF16/F16, the K-quant family (Q2K–Q8K,
/// IQ4_NL), and low-bit float packs (FP4/NF4/FP8).
fn dequant_to_f32(raw: &RawTensor, dtype: &DType) -> Result<Vec<f32>> {
    let n = raw.shape.iter().product::<usize>();
    match &dtype.storage {
        Storage::Native => match dtype.arith {
            grim_tensor::ArithType::F32 => bytes_to_f32(&raw.bytes, n),
            grim_tensor::ArithType::BF16 => {
                Ok(raw.bytes.chunks_exact(2).map(bf16_to_f32).collect())
            }
            grim_tensor::ArithType::F16 => {
                Ok(raw.bytes.chunks_exact(2).map(f16_to_f32_le).collect())
            }
            other => Err(Error::Unimplemented(format!(
                "WeightSource native materialization for arith {other:?} not supported"
            ))),
        },
        Storage::KQuant(scheme) => match scheme {
            KQuantScheme::Q2K => dequant_q2k(&raw.bytes, n),
            KQuantScheme::Q3K => dequant_q3k(&raw.bytes, n),
            KQuantScheme::Q4K => dequant_q4k(&raw.bytes, n),
            KQuantScheme::Q5K => dequant_q5k(&raw.bytes, n),
            KQuantScheme::Q6K => dequant_q6k(&raw.bytes, n),
            KQuantScheme::Q80 => dequant_q80(&raw.bytes, n),
            KQuantScheme::IQ4NL => dequant_iq4nl(&raw.bytes, n),
            KQuantScheme::IQ4XS => dequant_iq4xs(&raw.bytes, n),
            KQuantScheme::IQ3XXS => dequant_iq3xxs(&raw.bytes, n),
            KQuantScheme::IQ3S => dequant_iq3s(&raw.bytes, n),
            KQuantScheme::IQ2XXS => dequant_iq2xxs(&raw.bytes, n),
            KQuantScheme::IQ2XS => dequant_iq2xs(&raw.bytes, n),
            KQuantScheme::IQ2S => dequant_iq2s(&raw.bytes, n),
        },
        Storage::FloatPack(fp) => match fp {
            FloatPackScheme::Fp4 => dequant_fp4(&raw.bytes, n),
            FloatPackScheme::Nf4 => dequant_nf4(&raw.bytes, n),
            FloatPackScheme::Fp8 => dequant_fp8(&raw.bytes, n),
            FloatPackScheme::MxFp4 => dequant_mxfp4(&raw.bytes, n),
            FloatPackScheme::MxFp8 => dequant_mxfp8(&raw.bytes, n),
        },
        Storage::Block(block_type) => match block_type {
            BlockDtype::Fp4 => dequant_fp4(&raw.bytes, n),
            BlockDtype::Nf4 => dequant_nf4(&raw.bytes, n),
            BlockDtype::Fp8 => dequant_fp8(&raw.bytes, n),
            BlockDtype::Fp4Block16 => dequant_fp4_block16(&raw.bytes, n),
            BlockDtype::Fp8Block16 => dequant_fp8_block16(&raw.bytes, n),
        },
        Storage::ResidualPacked(_) => Err(Error::Unimplemented(
            "dequant_to_f32: ResidualPacked not yet supported".into(),
        )),
        Storage::GroupInt(cfg) => {
            // Unpack the four parallel arrays from raw.bytes
            let mut cursor = 0;
            let read_segment = |bytes: &[u8], cursor: &mut usize| -> Result<Vec<u8>> {
                if *cursor + 8 > bytes.len() {
                    return Err(Error::Backend("Truncated GPTQ packed header".into()));
                }
                let len =
                    u64::from_le_bytes(bytes[*cursor..*cursor + 8].try_into().unwrap()) as usize;
                *cursor += 8;
                if *cursor + len > bytes.len() {
                    return Err(Error::Backend(format!(
                        "Truncated GPTQ packed segment (expected {len} bytes)"
                    )));
                }
                let segment = bytes[*cursor..*cursor + len].to_vec();
                *cursor += len;
                Ok(segment)
            };

            let qweight = read_segment(&raw.bytes, &mut cursor)?;
            let qzeros = read_segment(&raw.bytes, &mut cursor)?;
            let scales = read_segment(&raw.bytes, &mut cursor)?;
            let g_idx = read_segment(&raw.bytes, &mut cursor)?;

            let g_idx_opt = if g_idx.is_empty() {
                None
            } else {
                Some(&g_idx[..])
            };

            grim_quant::dequant_gptq_group_int(
                &qweight,
                &qzeros,
                &scales,
                g_idx_opt,
                &raw.shape,
                cfg.bits as u32,
                cfg.group_size,
            )
        }
        Storage::CompressedTensorsW8A8Int8 | Storage::CompressedTensorsW8A8Fp8 => {
            // These W8A8 formats are resident-capable on ROCm/CUDA and reach the
            // fused dequant GEMM; host dequant isn't wired for them on the CPU fallback.
            Err(Error::Unimplemented(format!(
                "dequant_to_f32: storage {:?} requires resident GPU dequant path",
                dtype.storage
            )))
        }
        _ => Err(Error::Unimplemented(format!(
            "dequant_to_f32: storage {:?} has no host dequant path",
            dtype.storage
        ))),
    }
}

fn bytes_to_f32(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if bytes.len() != n * std::mem::size_of::<f32>() {
        return Err(Error::Backend(format!(
            "byte buffer length {} does not match f32 count {n}",
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        out.push(v);
    }
    Ok(out)
}

/// Convert a little-endian BF16 (brain float 16) byte pair to F32.
pub(crate) fn bf16_to_f32(bytes: &[u8]) -> f32 {
    let bits = u32::from(bytes[0]) | (u32::from(bytes[1]) << 8);
    f32::from_bits(bits << 16)
}

/// Convert a little-endian F16 (IEEE half) byte pair to F32.
pub(crate) fn f16_to_f32_le(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    let sign = (bits >> 15) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        // Subnormal or zero. An f16 subnormal encodes `mant * 2^-24`, but
        // `f32::from_bits((sign<<31)|(mant<<13))` instead yields
        // `mant * 2^-136` (≈2^112 too small). Build the correct value; this
        // mirrors the fix in grim-quant::f16_to_f32 so the two decoders agree.
        let value = (mant as f32) * 2f32.powi(-24);
        if sign != 0 { -value } else { value }
    } else if exp == 31 {
        f32::from_bits((sign << 31) | 0x7F80_0000 | (mant << 13))
    } else {
        f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyProvider {
        raw: RawTensor,
        dtype: DType,
    }

    impl grim_tensor::TensorProvider for DummyProvider {
        fn get(&self, _name: &str) -> Result<RawTensor> {
            Ok(self.raw.clone())
        }
        fn get_packed(&self, _name: &str) -> Result<RawTensor> {
            Ok(self.raw.clone())
        }
        fn meta(&self, _name: &str) -> Result<grim_tensor::TensorMeta> {
            Ok(grim_tensor::TensorMeta {
                dtype: self.dtype.clone(),
                provenance: QuantProvenance::GrimNative,
                shape: self.raw.shape.clone(),
                fusion_mask: 0,
            })
        }
        fn tensor_names(&self) -> Vec<String> {
            vec!["weight".to_string()]
        }
    }

    #[test]
    fn test_quantized_weight_cpu_dequantizes() {
        let q8_dtype = DType {
            arith: grim_tensor::ArithType::F32,
            storage: Storage::KQuant(KQuantScheme::Q80),
        };
        let dummy = DummyProvider {
            raw: RawTensor {
                bytes: vec![0u8; 64],
                shape: vec![2, 16],
                dtype: q8_dtype.clone(),
                provenance: QuantProvenance::GrimNative,
            },
            dtype: q8_dtype,
        };
        let ws = WeightSource::root(&dummy, Device::Cpu);
        let tensor = ws.get(Shape::new(vec![2, 16]), "weight").unwrap();
        assert_eq!(tensor.device(), &Device::Cpu);
    }

    #[test]
    fn test_weight_source_pp_path_concatenation() {
        let dummy = DummyProvider {
            raw: RawTensor {
                bytes: vec![0u8; 16],
                shape: vec![4],
                dtype: DType::F32,
                provenance: QuantProvenance::GrimNative,
            },
            dtype: DType::F32,
        };
        let ws = WeightSource::root(&dummy, Device::Cpu);
        let scoped = ws.pp("model").pp("layers.0");
        assert_eq!(scoped.full_name("weight"), "model.layers.0.weight");
    }

    #[test]
    fn prefetch_all_populates_cache_and_get_hits_it() {
        let q8 = DType {
            arith: grim_tensor::ArithType::F32,
            storage: Storage::KQuant(KQuantScheme::Q80),
        };
        let sentinel = vec![0xABu8; 64];
        let dummy = DummyProvider {
            raw: RawTensor {
                bytes: sentinel.clone(),
                shape: vec![2, 16],
                dtype: q8.clone(),
                provenance: QuantProvenance::GrimNative,
            },
            dtype: q8,
        };
        let ws = WeightSource::root(&dummy, Device::Cpu);
        // Before prefetch the cache is empty; get_raw_packed should still work
        // (on-demand path) and populate the cache.
        let before = ws.get_raw_packed("weight").unwrap();
        assert_eq!(before.bytes, sentinel);

        // Explicit parallel prefetch must also populate the cache.
        ws.prefetch_all();

        // A scoped (prefixed) source shares the cache Arc; its get must hit
        // the same cached entry keyed by full name.
        let scoped = ws.pp("model").pp("layers.0");
        let after = scoped.get_raw_packed("weight").unwrap();
        assert_eq!(after.bytes, sentinel);
    }

    // ===== audit probe: does prefetch actually populate + hit through a
    // RemappingTensorProvider with an hf->gguf name map (the Mellum2 path)? =====
    use std::collections::HashMap as HMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingNamedProvider {
        map: HMap<String, RawTensor>,
        fetches: AtomicUsize,
    }

    impl grim_tensor::TensorProvider for CountingNamedProvider {
        fn get(&self, name: &str) -> Result<RawTensor> {
            self.fetches.fetch_add(1, Ordering::SeqCst);
            Ok(self.map.get(name).expect("tensor present").clone())
        }
        fn get_packed(&self, name: &str) -> Result<RawTensor> {
            self.fetches.fetch_add(1, Ordering::SeqCst);
            Ok(self.map.get(name).expect("tensor present").clone())
        }
        fn meta(&self, _name: &str) -> Result<grim_tensor::TensorMeta> {
            Ok(grim_tensor::TensorMeta {
                dtype: DType::F32,
                provenance: QuantProvenance::GrimNative,
                shape: vec![2, 16],
                fusion_mask: 0,
            })
        }
        fn tensor_names(&self) -> Vec<String> {
            self.map.keys().cloned().collect()
        }
    }

    #[test]
    fn audit_prefetch_hits_through_hf_to_gguf_remap() {
        // PRODUCTION scenario for Lfm2/Mellum2: the checkpoint is GGUF-named
        // (blk.*) and the loader requests GGUF names; `remap_hf_to_gguf` only
        // has HF keys, so it is the identity on GGUF names.
        let mut map = HMap::new();
        let gguf_name = "blk.0.attn_norm.weight".to_string();
        map.insert(
            gguf_name.clone(),
            RawTensor {
                bytes: vec![0x7u8; 64],
                shape: vec![2, 16],
                dtype: DType::F32,
                provenance: QuantProvenance::GrimNative,
            },
        );
        let inner = CountingNamedProvider {
            map,
            fetches: AtomicUsize::new(0),
        };

        // Real engine remap: hf->gguf. Identity on GGUF names (no HF key matches).
        let remapped =
            grim_format::tprov::RemappingTensorProvider::new(&inner, |n: &str| -> String {
                if n == "model.layers.0.attn_norm.weight" {
                    "blk.0.attn_norm.weight".to_string()
                } else {
                    n.to_string()
                }
            });

        let ws = WeightSource::root(&remapped, Device::Cpu);
        ws.prefetch_all();

        // Loader requests the GGUF (blk.*) name.
        let scoped = ws.pp("blk").pp("0");
        let got = scoped.get_raw_packed("attn_norm.weight").unwrap();
        assert_eq!(got.bytes, vec![0x7u8; 64], "prefetch fetched wrong tensor");

        // The cache must have served the get WITHOUT a second provider fetch:
        // 1 tensor fetched exactly once during prefetch, 0 extra on get.
        let total = inner.fetches.load(Ordering::SeqCst);
        assert_eq!(
            total, 1,
            "cache did NOT absorb the get() fetch (fetches={total}); prefetch is ineffective",
        );
    }

    // ===== audit probe 2: does prefetch_all actually parallelize per-tensor
    // CPU work (the MXFP4 reframe dominates the read phase)? =====
    struct BusyNamedProvider {
        names: Vec<String>,
        work_us: u64,
    }
    impl grim_tensor::TensorProvider for BusyNamedProvider {
        fn get(&self, _n: &str) -> Result<RawTensor> {
            spin(self.work_us);
            Ok(raw(vec![0u8; 64], vec![2, 16], DType::F32))
        }
        fn get_packed(&self, _n: &str) -> Result<RawTensor> {
            spin(self.work_us);
            Ok(raw(vec![0u8; 64], vec![2, 16], DType::F32))
        }
        fn meta(&self, _n: &str) -> Result<grim_tensor::TensorMeta> {
            Ok(grim_tensor::TensorMeta {
                dtype: DType::F32,
                provenance: QuantProvenance::GrimNative,
                shape: vec![2, 16],
                fusion_mask: 0,
            })
        }
        fn tensor_names(&self) -> Vec<String> {
            self.names.clone()
        }
    }
    fn spin(us: u64) {
        let end = std::time::Instant::now() + std::time::Duration::from_micros(us);
        let mut x: u64 = 0;
        while std::time::Instant::now() < end {
            x = x.wrapping_add(x ^ 0x9E3779B97F4A7C15);
        }
        std::hint::black_box(x);
    }

    #[test]
    fn audit_prefetch_parallelism_speedup() {
        let n_tensors = 256u64;
        let per_tensor_us = 400u64; // simulate an MXFP4 reframe cost
        let names: Vec<String> = (0..n_tensors).map(|i| format!("blk.{i}.weight")).collect();
        let busy = BusyNamedProvider {
            names,
            work_us: per_tensor_us,
        };
        let ws = WeightSource::root(&busy, Device::Cpu);

        // Serial baseline: fetch every tensor one-by-one on the main thread.
        let t0 = std::time::Instant::now();
        for i in 0..n_tensors {
            let _ = ws.get_raw_packed(&format!("blk.{i}.weight")).unwrap();
        }
        let serial = t0.elapsed();

        // Prefetch: parallel fan-out.
        let t1 = std::time::Instant::now();
        ws.prefetch_all();
        // Then read from cache (mirrors the layer loop hitting the cache).
        for i in 0..n_tensors {
            let _ = ws.get_raw_packed(&format!("blk.{i}.weight")).unwrap();
        }
        let parallel = t1.elapsed();

        eprintln!(
            "[audit] serial={:.1?} prefetch+cache={:.1?} (n={n_tensors}, {per_tensor_us}us/tensor)",
            serial, parallel
        );
        // Prefetch + cache-read must be meaningfully faster than fully serial.
        assert!(
            parallel < serial,
            "prefetch did not beat serial read ({parallel:?} >= {serial:?})"
        );
    }

    // ===== golden dequant_to_f32 tests (see header below) =====

    fn close(got: f32, want: f32, ctx: &str) {
        let abs = (got - want).abs();
        let denom = want.abs().max(1e-7);
        assert!(got.is_finite(), "{ctx}: non-finite {got:?} (want {want:?})");
        assert!(
            abs == 0.0 || (abs / denom) < 1e-5,
            "{ctx}: got {got:?} want {want:?} (abs={abs})"
        );
    }

    fn raw(bytes: Vec<u8>, shape: Vec<usize>, dtype: DType) -> RawTensor {
        RawTensor {
            bytes,
            shape,
            dtype,
            provenance: QuantProvenance::GrimNative,
        }
    }

    #[test]
    fn dequant_to_f32_q80_routes_and_applies_f16_scale() {
        let dtype = DType {
            arith: grim_tensor::ArithType::F32,
            storage: Storage::KQuant(KQuantScheme::Q80),
        };
        let mut bytes = vec![0u8; 34];
        bytes[0..2].copy_from_slice(&0x4000u16.to_le_bytes());
        let codes: [i8; 32] = [
            1, -1, 127, -128, 64, -64, 0, 5, 10, -10, 50, -50, 25, -25, 100, -100, 3, -3, 7, 7, 9,
            9, 12, -12, 33, -33, 11, -11, 2, 4, 8, 16,
        ];
        for (i, &c) in codes.iter().enumerate() {
            bytes[2 + i] = c as u8;
        }
        let r = raw(bytes, vec![32], dtype.clone());
        let out = dequant_to_f32(&r, &dtype).unwrap();
        assert_eq!(out.len(), 32);
        for (i, &c) in codes.iter().enumerate() {
            close(out[i], (c as f32) * 2.0, &format!("q80[{i}]"));
        }
    }

    #[test]
    fn dequant_to_f32_bf16_native_shifts_left_16() {
        let dtype = DType {
            arith: grim_tensor::ArithType::BF16,
            storage: Storage::Native,
        };
        let bytes: Vec<u8> = vec![0x80, 0x3F, 0x00, 0xC0, 0x49, 0x40];
        let r = raw(bytes, vec![3], dtype.clone());
        let out = dequant_to_f32(&r, &dtype).unwrap();
        assert_eq!(out.len(), 3);
        close(out[0], 1.0, "bf16 1.0");
        close(out[1], -2.0, "bf16 -2.0");
        close(out[2], 3.140625, "bf16 ~pi");
    }

    #[test]
    fn dequant_to_f32_f16_native_normalized_and_subnormal() {
        let dtype = DType {
            arith: grim_tensor::ArithType::F16,
            storage: Storage::Native,
        };
        let bytes: Vec<u8> = vec![0x00, 0x3C, 0x00, 0x40, 0x48, 0x42, 0x00, 0x02, 0x00, 0xC0];
        let r = raw(bytes, vec![5], dtype.clone());
        let out = dequant_to_f32(&r, &dtype).unwrap();
        assert_eq!(out.len(), 5);
        close(out[0], 1.0, "f16 1.0");
        close(out[1], 2.0, "f16 2.0");
        close(out[2], 3.140625, "f16 ~pi");
        close(out[3], (512.0f32) * 2f32.powi(-24), "f16 subnormal 0x0200");
        close(out[4], -2.0, "f16 -2.0");
    }

    #[test]
    fn bytes_to_f32_rejects_wrong_length() {
        let dtype = DType::F32;
        let r = raw(vec![0u8, 1, 2], vec![1], dtype.clone());
        assert!(
            dequant_to_f32(&r, &dtype).is_err(),
            "non-f32-multiple must error"
        );
        let r2 = raw(vec![0u8, 0, 0x40, 0x40], vec![1], dtype.clone());
        let out = dequant_to_f32(&r2, &dtype).unwrap();
        assert_eq!(out.len(), 1);
        close(out[0], 3.0, "f32 native 3.0");
    }

    #[test]
    fn dequant_to_f32_group_int_unpacks_length_prefixed_segments() {
        use grim_tensor::dtype::{GpuIntConfig, GroupQuantScheme};
        let dtype = DType {
            arith: grim_tensor::ArithType::F32,
            storage: Storage::GroupInt(GpuIntConfig {
                bits: 4,
                group_size: 8,
                scheme: GroupQuantScheme::Asymmetric,
                desc_act: false,
            }),
        };
        let mut qw: u32 = 0;
        for i in 0..8u32 {
            qw |= (i & 0xF) << (i * 4);
        }
        let qweight = qw.to_le_bytes().to_vec();
        let mut qz: u32 = 0;
        for col in 0..8u32 {
            qz |= 7u32 << (col * 4);
        }
        let qzeros = qz.to_le_bytes().to_vec();
        let scales = 0.5f32.to_le_bytes().to_vec();
        let g_idx: Vec<u8> = Vec::new();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(qweight.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&qweight);
        bytes.extend_from_slice(&(qzeros.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&qzeros);
        bytes.extend_from_slice(&(scales.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&scales);
        bytes.extend_from_slice(&(g_idx.len() as u64).to_le_bytes());
        let r = raw(bytes, vec![8, 1], dtype.clone());
        let out = dequant_to_f32(&r, &dtype).expect("group-int dequant");
        assert_eq!(out.len(), 8);
        for (i, &v) in out.iter().enumerate() {
            close(v, (i as f32 - 8.0) * 0.5, &format!("groupint[{i}]"));
        }
    }
}
