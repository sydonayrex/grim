//! Streaming block-wise forward execution with gradient checkpointing (WI-T2).
//!
//! Provides `StreamingBlockForward` that reads quantized transformer weights
//! lazily block-by-block from a `TensorProvider`, runs fused forward operations,
//! and manages activation recomputation buffers (`GradientCheckpointBuffer`).

use crate::rope_scaling::{RopeScalingMethod, scaling_base};
use grim_autograd::AutogradScope;
use grim_backend_rocm::RocmDevice;
use grim_core::error::{Error, Result};
use grim_models_transformer::{LlamaBlock, LlamaConfig};
use grim_nn::WeightSource;
use grim_nn::modules::{pick_device_for_storage_device, pick_device_for_tensor};
use grim_tensor::{DType, Device, Shape, Tensor, TensorProvider};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;
use grim_tensor::{MemoryOps};

/// Saved activation checkpoint for a transformer block.
#[derive(Debug, Clone)]
pub struct LayerActivationCheckpoint {
    pub layer_idx: usize,
    pub input_x: Tensor,
}

/// Gradient checkpointing buffer enforcing bounded peak memory by retaining only block inputs.
#[derive(Debug, Default)]
pub struct GradientCheckpointBuffer {
    checkpoints: HashMap<usize, LayerActivationCheckpoint>,
}

fn prefetch_block_weights(block: &LlamaBlock) -> Result<()> {
    let mut tensors: Vec<&Tensor> = vec![
        &block.attn_norm.weight,
        block.wq.weight(),
        block.wk.weight(),
        block.wv.weight(),
        block.wo.weight(),
        &block.ffn_norm.weight,
    ];
    if let Some(ref wg) = block.w_gate {
        tensors.push(wg.weight());
    }
    if let Some(ref wu) = block.w_up {
        tensors.push(wu.weight());
    }
    if let Some(ref wd) = block.w_down {
        tensors.push(wd.weight());
    }
    for tensor in tensors {
        tensor
            .storage()
            .prefetch_to_device()
            .map_err(Error::Tensor)?;
    }
    Ok(())
}

impl GradientCheckpointBuffer {
    /// Create a new empty checkpoint buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Save input activation checkpoint for layer `layer_idx`.
    pub fn save(&mut self, layer_idx: usize, input_x: Tensor) {
        self.checkpoints
            .insert(layer_idx, LayerActivationCheckpoint { layer_idx, input_x });
    }

    /// Retrieve input activation checkpoint for layer `layer_idx`.
    pub fn get(&self, layer_idx: usize) -> Option<&Tensor> {
        self.checkpoints.get(&layer_idx).map(|c| &c.input_x)
    }

    /// Clear stored checkpoints after backward pass completion.
    pub fn clear(&mut self) {
        self.checkpoints.clear();
    }
}

/// Transfer `x` to `target_device`, returning a tensor owned by that device.
///
/// ROCm→ROCm moves use a true device-to-device copy (peer-routed through
/// `RocmDevice::copy_via_route`), so activations never detour through host
/// memory. CUDA↔CUDA moves within the same PCI domain would also support
/// peer memcpy via `cuMemcpyPeer` but is not yet wired. Any other cross-backend
/// move (different backends, or different PCI domains) stages through host memory
/// since cross-device P2P transfer requires backend-specific peer support that
/// is not universally available (e.g. ROCm↔CUDA GPUDirect requires both drivers
/// and hardware support).
fn transfer_to_device(x: &Tensor, target_device: &Device) -> Result<Tensor> {
    if x.device() == target_device {
        return Ok(x.clone());
    }
    // Same-backend ROCm move: peer memcpy, no host round-trip.
    if let (Device::Rocm(src_ord), Device::Rocm(dst_ord)) = (x.device(), target_device) {
        if src_ord != dst_ord {
            let dev = RocmDevice::shared(*dst_ord);
            let dst_storage = dev.alloc_storage(x.shape(), DType::F32)?;
            let src_ptr = x.storage().device_ptr().ok_or_else(|| {
                Error::Tensor(grim_tensor::Error::Backend(
                    "transfer_to_device: source lacks a device pointer".into(),
                ))
            })? as *const c_void;
            let dst_ptr = dst_storage.device_ptr().ok_or_else(|| {
                Error::Tensor(grim_tensor::Error::Backend(
                    "transfer_to_device: destination lacks a device pointer".into(),
                ))
            })? as *mut c_void;
            let bytes = x.shape().elem_count() * std::mem::size_of::<f32>();
            dev.copy_via_route(*src_ord as i32, *dst_ord as i32, src_ptr, dst_ptr, bytes)?;
            return Ok(Tensor::new(
                Arc::from(dst_storage),
                x.shape().clone(),
                DType::F32,
                x.provenance().clone(),
                target_device.clone(),
            ));
        }
        // Same ordinal, distinct device identity: no data movement needed.
        return Ok(x.clone());
    }
    // Same-backend CUDA move: peer memcpy / staging.
    #[cfg(feature = "cuda")]
    if let (Device::Cuda(src_ord), Device::Cuda(dst_ord)) = (x.device(), target_device) {
        if src_ord != dst_ord {
            if let Ok(dev) = grim_backend_cuda::CudaDevice::new(*dst_ord) {
                if let Ok(dst_storage) = dev.alloc_storage(x.shape(), DType::F32) {
                    if let (Some(src_ptr), Some(dst_ptr)) = (x.storage().device_ptr(), dst_storage.device_ptr()) {
                        let bytes = x.shape().elem_count() * std::mem::size_of::<f32>();
                        if dev.copy_via_route(*src_ord as i32, *dst_ord as i32, src_ptr as *const c_void, dst_ptr as *mut c_void, bytes).is_ok() {
                            return Ok(Tensor::new(
                                Arc::from(dst_storage),
                                x.shape().clone(),
                                DType::F32,
                                x.provenance().clone(),
                                target_device.clone(),
                            ));
                        }
                    }
                }
            }
        }
        return Ok(x.clone());
    }
    // Cross-backend: host-staged upload onto the target device.
    let dev = pick_device_for_storage_device(target_device);
    let out_storage = dev.from_cpu(&x.to_vec_f32()?, x.shape(), DType::F32)?;
    Ok(Tensor::new(
        Arc::from(out_storage),
        x.shape().clone(),
        DType::F32,
        x.provenance().clone(),
        target_device.clone(),
    ))
}

/// SCYTHE-2 placement routing attached to a [`StreamingBlockForward`] (WI-INF3).
///
/// When present, every `forward_block` consults the controller with the real
/// per-call `layer_idx` and activation shape — the direct analog of the fix
/// landed in `grim-garage::run_rank_sft_forward` (no synthetic keys) — and
/// executes the block on the controller-chosen device. Placement changes which
/// device runs the block, never the math: every rank runs identical weights and
/// kernels, so numerics are invariant under re-placement (pinned by the parity
/// gate test).
pub struct ScytheRoute {
    /// The online placement router. Shared ownership so the engine can keep
    /// warming/observing the same controller it handed off.
    pub ctrl: std::sync::Arc<std::sync::Mutex<crate::scythe2::C2plrController>>,
    /// Live capability source; refreshed only when the capability epoch moves,
    /// so the ~50 ns decode hit path never pays for a topology probe.
    /// `None` in CPU-only harnesses, which seed `caps`/`links` directly.
    pub profiler: Option<Arc<grim_backend_rocm::CapabilityProfiler>>,
    /// Capability snapshot used when `profiler` is `None` (and the seed value
    /// otherwise overwritten on first epoch refresh).
    pub caps: Vec<grim_tensor::backend::GpuCapability>,
    /// Link-matrix snapshot paired with `caps` (same refresh discipline).
    pub links: Vec<grim_tensor::backend::ScytheLink>,
    /// Epoch the `caps`/`links` snapshot was taken at (`CAPABILITY_EPOCH`).
    pub caps_epoch: u32,
    /// Rank → execution device. The engine maps rank → `Device::Rocm(rank)`;
    /// CPU-only harnesses map every rank to `Device::Cpu`, which is what lets
    /// the parity gate prove numerics-invariance without GPUs.
    pub device_for_rank: Arc<dyn Fn(usize) -> Device + Send + Sync>,
}

impl ScytheRoute {
    /// Refresh `caps`/`links` iff the capability epoch moved since the snapshot
    /// was taken. Topology probes (`link_matrix`) cost host µs-to-ms per pair,
    /// so they must never run per layer call — only on epoch bumps (~100 ms
    /// cadence or GPU-leave), where they amortize against a full cache miss.
    fn refresh_if_stale(&mut self) {
        let epoch = grim_backend_rocm::current_epoch();
        if !self.caps.is_empty() && self.caps_epoch == epoch {
            return;
        }
        if let Some(ref profiler) = self.profiler {
            let caps = profiler.capabilities();
            let n = caps.len().max(1);
            self.links = grim_backend_rocm::CapabilityProfiler::link_matrix(n);
            self.caps = caps;
            self.caps_epoch = epoch;
        } else if self.caps.is_empty() {
            // No live source and nothing seeded: single anonymous device.
            self.caps = vec![grim_tensor::backend::GpuCapability::default()];
            self.links = vec![grim_tensor::backend::ScytheLink::PeerDirect];
            self.caps_epoch = epoch;
        }
    }
}

/// Block-wise streaming forward executor for memory-bounded QLoRA fine-tuning.
pub struct StreamingBlockForward {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub checkpoint_buffer: GradientCheckpointBuffer,
    pub rope_scaling: Option<RopeScalingMethod>,
    /// Session-lifetime cache of materialized `LlamaBlock`s, keyed by
    /// `(layer_idx, device)` so block weights are loaded/uploaded once and
    /// reused across forward and recompute passes. Cleared explicitly via
    /// [`Self::clear_block_cache`] when the model or configuration changes.
    block_cache: std::sync::Mutex<HashMap<(usize, Device), LlamaBlock>>,
    /// Optional SCYTHE-2 routing (WI-INF3). `None` keeps blocks on the
    /// incoming tensor's device — unchanged behavior and the default.
    pub scythe_route: Option<ScytheRoute>,
}

impl StreamingBlockForward {
    /// Create a new `StreamingBlockForward` instance.
    pub fn new(num_layers: usize, hidden_size: usize) -> Self {
        Self {
            num_layers,
            hidden_size,
            checkpoint_buffer: GradientCheckpointBuffer::new(),
            rope_scaling: None,
            block_cache: std::sync::Mutex::new(HashMap::new()),
            scythe_route: None,
        }
    }

    /// Attach SCYTHE-2 placement routing (WI-INF3). While attached, every
    /// `forward_block` runs its block on the controller-chosen device.
    pub fn attach_scythe_route(&mut self, route: ScytheRoute) {
        self.scythe_route = Some(route);
    }

    /// Detach SCYTHE-2 routing; subsequent blocks run on the incoming device.
    pub fn clear_scythe_route(&mut self) {
        self.scythe_route = None;
    }

    /// WI-INF3 decision step: consult the controller with the real layer index
    /// and activation shape, then move `x` to the chosen rank's device.
    ///
    /// Cache-hit path ~50 ns/layer; miss path ~10 µs/layer (TODO(gpu-verify):
    /// budget claim measured host-side only so far — see
    /// `benches/scythe2_decide_miss.rs` and WI-INF4).
    fn route_for_execution(&mut self, layer_idx: usize, x: &Tensor) -> Result<Tensor> {
        let Some(route) = self.scythe_route.as_mut() else {
            return Ok(x.clone());
        };
        route.refresh_if_stale();
        let epoch = grim_backend_rocm::current_epoch();
        let shape = x.shape().dims();
        let placement = {
            let mut ctrl = route.ctrl.lock().unwrap_or_else(|e| e.into_inner());
            ctrl.decide(layer_idx as u32, shape, &route.caps, &route.links, epoch)
        };
        // Primary owner = the rank holding the largest partition share; the
        // whole block executes there. Any rank yields identical numerics —
        // only load balance differs.
        let target_rank = placement
            .partition
            .iter()
            .copied()
            .zip(placement.ranks.iter().copied())
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, r)| r)
            .unwrap_or(0);
        let target_device = (route.device_for_rank)(target_rank);
        transfer_to_device(x, &target_device)
    }

    /// Drop the cached materialized blocks. Call after the underlying model
    /// or configuration changes so the next forward pass reloads weights.
    pub fn clear_block_cache(&self) {
        self.block_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    /// F2 (full-parameter write-back): copy stepped base weights from the
    /// autograd registry into the cached blocks, so every subsequent forward
    /// pass (which clones from this cache) trains on the updated weights.
    /// CPU-resident blocks only — device-resident weight mirrors land with
    /// the dirty-block upload work.
    pub fn overwrite_base_weights(&self, reg: &grim_autograd::AutogradRegistry) -> Result<()> {
        use grim_autograd::{LoRAInjectionPoint, ParamId};
        let mut cache = self.block_cache.lock().unwrap_or_else(|e| e.into_inner());
        for ((layer_idx, dev), block) in cache.iter_mut() {
            if *dev != Device::Cpu {
                continue;
            }
            let layer = *layer_idx;
            let apply =
                |point: LoRAInjectionPoint, linear: &mut grim_nn::modules::Linear| -> Result<()> {
                    if let Some(p) = reg.params.get(ParamId::base(layer, point)) {
                        linear.replace_weight(p.data.clone())?;
                    }
                    Ok(())
                };
            apply(LoRAInjectionPoint::QProj, &mut block.wq.inner)?;
            apply(LoRAInjectionPoint::KProj, &mut block.wk.inner)?;
            apply(LoRAInjectionPoint::VProj, &mut block.wv.inner)?;
            apply(LoRAInjectionPoint::OProj, &mut block.wo.inner)?;
            if let Some(g) = block.w_gate.as_mut() {
                apply(LoRAInjectionPoint::GateProj, &mut g.inner)?;
            }
            if let Some(u) = block.w_up.as_mut() {
                apply(LoRAInjectionPoint::UpProj, &mut u.inner)?;
            }
            if let Some(d) = block.w_down.as_mut() {
                apply(LoRAInjectionPoint::DownProj, &mut d.inner)?;
            }
        }
        Ok(())
    }

    /// F2 test hook: clone of a cached block's QProj weight (`None` if the
    /// layer is not materialized on `device`).
    pub fn cached_qproj_weight(&self, layer_idx: usize, device: &Device) -> Option<Tensor> {
        let cache = self.block_cache.lock().unwrap_or_else(|e| e.into_inner());
        cache
            .get(&(layer_idx, device.clone()))
            .map(|b| b.wq.inner.weight.clone())
    }

    /// Load (or return the cached) `LlamaBlock` for `layer_idx` on `device`.
    fn block(
        &self,
        provider: &dyn TensorProvider,
        cfg: &LlamaConfig,
        layer_idx: usize,
        device: &Device,
    ) -> Result<LlamaBlock> {
        let mut cache = self.block_cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(block) = cache.get(&(layer_idx, device.clone())) {
            return Ok(block.clone());
        }
        let ws = WeightSource::root(provider, device.clone());
        let block_ws = ws.pp("layers").pp(&layer_idx.to_string());
        let block = LlamaBlock::load(&block_ws, cfg)?;
        prefetch_block_weights(&block)?;
        cache.insert((layer_idx, device.clone()), block.clone());
        Ok(block)
    }

    /// Configure a RoPE scaling method for long-context training.
    pub fn with_rope_scaling(mut self, method: RopeScalingMethod) -> Self {
        self.rope_scaling = Some(method);
        self
    }

    /// Run streaming block-wise forward pass for `layer_idx`.
    ///
    /// Reads layer input `x`, records activation checkpoint in `checkpoint_buffer`,
    /// then loads block weights lazily from `provider` and runs real
    /// transformer block math (RMSNorm → GQA attention → residual →
    /// RMSNorm → SwiGLU FFN → residual) via `LlamaBlock`.
    ///
    /// `positions` carries RoPE position ids for this block (required for correct
    /// positional encoding in loss-eval / reference-model paths). When `None`,
    /// an empty slice is passed (RoPE not applied), matching prior behavior.
    /// Run streaming forward block for `layer_idx` targeting a SCYTHE-2 assigned GPU device.
    ///
    /// If `target_device` differs from `x.device()`, transfers activation `x` to
    /// `target_device` before running block forward. ROCm→ROCm moves use a true
    /// device-to-device copy (no host round-trip); other pairs stage through host.
    pub fn forward_block_on_device(
        &mut self,
        provider: &dyn TensorProvider,
        cfg: &LlamaConfig,
        layer_idx: usize,
        x: &Tensor,
        positions: Option<&[u32]>,
        target_device: &grim_tensor::Device,
    ) -> Result<Tensor> {
        let x_target = transfer_to_device(x, target_device)?;
        self.forward_block(provider, cfg, layer_idx, &x_target, positions)
    }

    pub fn forward_block(
        &mut self,
        provider: &dyn TensorProvider,
        cfg: &LlamaConfig,
        layer_idx: usize,
        x: &Tensor,
        positions: Option<&[u32]>,
    ) -> Result<Tensor> {
        if layer_idx >= self.num_layers {
            return Err(Error::Config(format!(
                "layer_idx {} out of bounds for num_layers {}",
                layer_idx, self.num_layers
            )));
        }

        // WI-INF3: controller-supplied placement when a SCYTHE-2 route is
        // attached; otherwise the incoming device (fixed-rank behavior).
        let x = self.route_for_execution(layer_idx, x)?;

        // Save input checkpoint for recomputation during backward pass
        self.checkpoint_buffer.save(layer_idx, x.clone());

        // Load block weights lazily from provider on target tensor device, run real forward
        let block = self.block(provider, cfg, layer_idx, x.device())?;
        block.forward(&x, positions.unwrap_or(&[]))
    }

    /// Recompute block forward pass from saved input checkpoint during backward traversal.
    /// Reuses the session-cached `LlamaBlock` weights instead of reloading them.
    pub fn recompute_block(
        &self,
        provider: &dyn TensorProvider,
        cfg: &LlamaConfig,
        layer_idx: usize,
        positions: Option<&[u32]>,
    ) -> Result<Tensor> {
        let input_x = self.checkpoint_buffer.get(layer_idx).ok_or_else(|| {
            Error::Config(format!(
                "missing activation checkpoint for layer {}",
                layer_idx
            ))
        })?;

        let block = self.block(provider, cfg, layer_idx, input_x.device())?;
        block.forward(input_x, positions.unwrap_or(&[]))
    }

    /// Run streaming block-wise forward pass for `layer_idx` with autograd tape recording.
    ///
    /// Loads block weights lazily from `provider`, executes pre-norm attention and SwiGLU FFN,
    /// applies enabled LoRA adapters via `apply_and_record_lora`, and records `LoRAApply` and `Add`
    /// entries on `tape`. Returns `(output_tensor_id, output_tensor)`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_block_with_autograd(
        &mut self,
        provider: &dyn TensorProvider,
        cfg: &LlamaConfig,
        autograd_reg: &grim_autograd::AutogradRegistry,
        tape: &mut grim_autograd::Tape,
        layer_idx: usize,
        x: &Tensor,
        x_id: grim_autograd::TensorId,
    ) -> Result<(grim_autograd::TensorId, Tensor)> {
        if layer_idx >= self.num_layers {
            return Err(Error::Config(format!(
                "layer_idx {} out of bounds for num_layers {}",
                layer_idx, self.num_layers
            )));
        }

        self.checkpoint_buffer.save(layer_idx, x.clone());

        let block = self.block(provider, cfg, layer_idx, x.device())?;

        // Pre-attention norm & Q/K/V projections
        let x_norm = block.attn_norm.forward(x)?;
        let x_norm_base_id = tape.register(x_norm.clone());

        let q_base = block.wq.forward(&x_norm)?;
        // WI-T8 (FullParameter): record base-weight MatMul so gradients flow
        // back through every base weight, not just LoRA adapters.
        let q_base_id = if autograd_reg.scope == AutogradScope::FullParameter {
            let q_m = q_base.shape().dims()[0];
            let q_k = x_norm.shape().dims()[1];
            let q_n = q_base.shape().dims()[1];
            let wq_id = tape.register(block.wq.weight().clone());
            let q_base_param =
                grim_autograd::ParamId::base(layer_idx, grim_autograd::LoRAInjectionPoint::QProj);
            tape.record_matmul(
                x_norm_base_id,
                wq_id,
                q_base.clone(),
                false,
                true,
                q_m,
                q_k,
                q_n,
                Some(q_base_param),
            )
        } else {
            tape.register(q_base.clone())
        };
        let (_q_id, q) = grim_autograd::apply_and_record_lora(
            autograd_reg,
            tape,
            layer_idx,
            grim_autograd::LoRAInjectionPoint::QProj,
            q_base,
            q_base_id,
            x_norm.clone(),
            x_norm_base_id,
        )?;

        let k_base = block.wk.forward(&x_norm)?;
        let k_base_id = if autograd_reg.scope == AutogradScope::FullParameter {
            let k_m = k_base.shape().dims()[0];
            let k_k = x_norm.shape().dims()[1];
            let k_n = k_base.shape().dims()[1];
            let wk_id = tape.register(block.wk.weight().clone());
            tape.record_matmul(
                x_norm_base_id,
                wk_id,
                k_base.clone(),
                false,
                true,
                k_m,
                k_k,
                k_n,
                Some(grim_autograd::ParamId::base(
                    layer_idx,
                    grim_autograd::LoRAInjectionPoint::KProj,
                )),
            )
        } else {
            tape.register(k_base.clone())
        };
        let (_k_id, k) = grim_autograd::apply_and_record_lora(
            autograd_reg,
            tape,
            layer_idx,
            grim_autograd::LoRAInjectionPoint::KProj,
            k_base,
            k_base_id,
            x_norm.clone(),
            x_norm_base_id,
        )?;

        let v_base = block.wv.forward(&x_norm)?;
        let v_base_id = if autograd_reg.scope == AutogradScope::FullParameter {
            let v_m = v_base.shape().dims()[0];
            let v_k = x_norm.shape().dims()[1];
            let v_n = v_base.shape().dims()[1];
            let wv_id = tape.register(block.wv.weight().clone());
            tape.record_matmul(
                x_norm_base_id,
                wv_id,
                v_base.clone(),
                false,
                true,
                v_m,
                v_k,
                v_n,
                Some(grim_autograd::ParamId::base(
                    layer_idx,
                    grim_autograd::LoRAInjectionPoint::VProj,
                )),
            )
        } else {
            tape.register(v_base.clone())
        };
        let (_v_id, v) = grim_autograd::apply_and_record_lora(
            autograd_reg,
            tape,
            layer_idx,
            grim_autograd::LoRAInjectionPoint::VProj,
            v_base,
            v_base_id,
            x_norm.clone(),
            x_norm_base_id,
        )?;

        // Apply RoPE + qkv_attention via placement-aware BackendDevice.
        // Q/K arrive as [total_tokens, num_head_dims] (flat layout
        // (i*num_heads + h)*head_dim, which is exactly what the attention
        // kernel's q_offset/k_offset expect). RoPE operates on (B, S, D=head_dim),
        // so we view the same storage as [1, total_tokens*num_heads, head_dim]
        // with the token position repeated per head — a pure shape change, no
        // data movement. V is never rotated.
        let num_head_dims = cfg.num_heads * cfg.head_dim;
        let total_tokens = q.shape().elem_count() / num_head_dims;

        // RoPE base respects config rope_theta, optionally scaled for long-context training.
        let rope_base = self.rope_scaling.as_ref().map_or(cfg.rope_theta, |m| {
            scaling_base(m, cfg.rope_theta, cfg.head_dim)
        });

        let dev = pick_device_for_tensor(&q);

        let mut q_positions = Vec::with_capacity(total_tokens * cfg.num_heads);
        for t in 0..total_tokens as u32 {
            for _ in 0..cfg.num_heads {
                q_positions.push(t);
            }
        }
        let mut k_positions = Vec::with_capacity(total_tokens * cfg.num_kv_heads);
        for t in 0..total_tokens as u32 {
            for _ in 0..cfg.num_kv_heads {
                k_positions.push(t);
            }
        }
        let q_shape = Shape::new(vec![1, total_tokens * cfg.num_heads, cfg.head_dim]);
        let k_shape = Shape::new(vec![1, total_tokens * cfg.num_kv_heads, cfg.head_dim]);
        let out_shape_2d = Shape::new(vec![total_tokens, num_head_dims]);

        // Reshape Q/K to 3D [1, total_tokens*num_heads, head_dim] for the
        // `rope` call below. The data is already laid out as
        // (i*num_heads + h)*head_dim, and rope() only reads the storage's
        // device pointer plus the explicitly-passed shape — so for
        // device-resident Q/K we pass the original storage straight through
        // with the 3-D shape (zero copies). The previous form did
        // to_vec_f32 + from_cpu: a full D2H+H2D round trip per tensor per
        // layer per token. CPU tensors still take the upload path.
        let q_up: Option<Box<dyn grim_tensor::BackendStorage>> = if q.device().is_cpu() {
            Some(dev.from_cpu(&q.to_vec_f32()?, &q_shape, DType::F32)?)
        } else {
            None
        };
        let k_up: Option<Box<dyn grim_tensor::BackendStorage>> = if k.device().is_cpu() {
            Some(dev.from_cpu(&k.to_vec_f32()?, &k_shape, DType::F32)?)
        } else {
            None
        };
        let q_in: &dyn grim_tensor::BackendStorage = match &q_up {
            Some(b) => b.as_ref(),
            None => &**q.storage(),
        };
        let k_in: &dyn grim_tensor::BackendStorage = match &k_up {
            Some(b) => b.as_ref(),
            None => &**k.storage(),
        };

        let rope_cfg = grim_tensor::RopeConfig::new(cfg.head_dim, rope_base);
        let (q_rot_s, _) = dev.rope(q_in, &q_positions, &rope_cfg, &q_shape)?;
        let (k_rot_s, _) = dev.rope(k_in, &k_positions, &rope_cfg, &k_shape)?;

        let q_rot = Tensor::new(
            std::sync::Arc::from(q_rot_s),
            q_shape,
            DType::F32,
            q.provenance().clone(),
            q.device().clone(),
        );
        let k_rot = Tensor::new(
            std::sync::Arc::from(k_rot_s),
            k_shape,
            DType::F32,
            k.provenance().clone(),
            k.device().clone(),
        );

        let (attn_s, _) = dev.qkv_attention(
            q_rot.storage().as_ref(),
            k_rot.storage().as_ref(),
            v.storage().as_ref(),
            cfg.num_kv_heads,
            total_tokens,
            0,
            None,
            &out_shape_2d,
            None,
            None,
        )?;

        let attn_raw = Tensor::new(
            std::sync::Arc::from(attn_s),
            out_shape_2d,
            DType::F32,
            q.provenance().clone(),
            q.device().clone(),
        );
        let attn_raw_id = tape.register(attn_raw.clone());

        let wo_base = block.wo.forward(&attn_raw)?;
        let wo_base_id = if autograd_reg.scope == AutogradScope::FullParameter {
            let wo_m = wo_base.shape().dims()[0];
            let wo_k = attn_raw.shape().dims()[1];
            let wo_n = wo_base.shape().dims()[1];
            let wo_w_id = tape.register(block.wo.weight().clone());
            tape.record_matmul(
                attn_raw_id,
                wo_w_id,
                wo_base.clone(),
                false,
                true,
                wo_m,
                wo_k,
                wo_n,
                Some(grim_autograd::ParamId::base(
                    layer_idx,
                    grim_autograd::LoRAInjectionPoint::OProj,
                )),
            )
        } else {
            tape.register(wo_base.clone())
        };
        let (wo_id, wo_out) = grim_autograd::apply_and_record_lora(
            autograd_reg,
            tape,
            layer_idx,
            grim_autograd::LoRAInjectionPoint::OProj,
            wo_base,
            wo_base_id,
            attn_raw,
            attn_raw_id,
        )?;

        // Residual addition 1
        let dev = grim_autograd::pick_device_for_tensor(x);
        let (res1_storage, _) =
            dev.add(x.storage().as_ref(), wo_out.storage().as_ref(), x.shape())?;
        let res1 = Tensor::new(
            std::sync::Arc::from(res1_storage),
            x.shape().clone(),
            grim_tensor::DType::F32,
            x.provenance().clone(),
            x.device().clone(),
        );
        let res1_id = tape.record_add(x_id, wo_id, res1.clone(), None);

        // FFN pre-norm & Gate/Up/Down projections
        let ffn_norm_out = block.ffn_norm.forward(&res1)?;
        let ffn_norm_id = tape.register(ffn_norm_out.clone());

        let gate_base = block
            .w_gate
            .as_ref()
            .expect("dense FFN")
            .forward(&ffn_norm_out)?;
        let gate_base_id = if autograd_reg.scope == AutogradScope::FullParameter {
            let g_m = gate_base.shape().dims()[0];
            let g_k = ffn_norm_out.shape().dims()[1];
            let g_n = gate_base.shape().dims()[1];
            let wg_id = tape.register(block.w_gate.as_ref().expect("dense FFN").weight().clone());
            tape.record_matmul(
                ffn_norm_id,
                wg_id,
                gate_base.clone(),
                false,
                true,
                g_m,
                g_k,
                g_n,
                Some(grim_autograd::ParamId::base(
                    layer_idx,
                    grim_autograd::LoRAInjectionPoint::GateProj,
                )),
            )
        } else {
            tape.register(gate_base.clone())
        };
        let (_gate_id, gate) = grim_autograd::apply_and_record_lora(
            autograd_reg,
            tape,
            layer_idx,
            grim_autograd::LoRAInjectionPoint::GateProj,
            gate_base,
            gate_base_id,
            ffn_norm_out.clone(),
            ffn_norm_id,
        )?;

        let up_base = block
            .w_up
            .as_ref()
            .expect("dense FFN")
            .forward(&ffn_norm_out)?;
        let up_base_id = if autograd_reg.scope == AutogradScope::FullParameter {
            let u_m = up_base.shape().dims()[0];
            let u_k = ffn_norm_out.shape().dims()[1];
            let u_n = up_base.shape().dims()[1];
            let wu_id = tape.register(block.w_up.as_ref().expect("dense FFN").weight().clone());
            tape.record_matmul(
                ffn_norm_id,
                wu_id,
                up_base.clone(),
                false,
                true,
                u_m,
                u_k,
                u_n,
                Some(grim_autograd::ParamId::base(
                    layer_idx,
                    grim_autograd::LoRAInjectionPoint::UpProj,
                )),
            )
        } else {
            tape.register(up_base.clone())
        };
        let (_up_id, up) = grim_autograd::apply_and_record_lora(
            autograd_reg,
            tape,
            layer_idx,
            grim_autograd::LoRAInjectionPoint::UpProj,
            up_base,
            up_base_id,
            ffn_norm_out.clone(),
            ffn_norm_id,
        )?;

        let (silu_storage, _) =
            dev.silu_mul(gate.storage().as_ref(), up.storage().as_ref(), gate.shape())?;
        let silu_tensor = Tensor::new(
            std::sync::Arc::from(silu_storage),
            gate.shape().clone(),
            grim_tensor::DType::F32,
            gate.provenance().clone(),
            gate.device().clone(),
        );
        let silu_id = tape.register(silu_tensor.clone());

        let down_base = block
            .w_down
            .as_ref()
            .expect("dense FFN")
            .forward(&silu_tensor)?;
        let down_base_id = if autograd_reg.scope == AutogradScope::FullParameter {
            let d_m = down_base.shape().dims()[0];
            let d_k = silu_tensor.shape().dims()[1];
            let d_n = down_base.shape().dims()[1];
            let wd_id = tape.register(block.w_down.as_ref().expect("dense FFN").weight().clone());
            tape.record_matmul(
                silu_id,
                wd_id,
                down_base.clone(),
                false,
                true,
                d_m,
                d_k,
                d_n,
                Some(grim_autograd::ParamId::base(
                    layer_idx,
                    grim_autograd::LoRAInjectionPoint::DownProj,
                )),
            )
        } else {
            tape.register(down_base.clone())
        };
        let (down_id, down_out) = grim_autograd::apply_and_record_lora(
            autograd_reg,
            tape,
            layer_idx,
            grim_autograd::LoRAInjectionPoint::DownProj,
            down_base,
            down_base_id,
            silu_tensor,
            silu_id,
        )?;

        // Residual addition 2
        let (res2_storage, _) = dev.add(
            res1.storage().as_ref(),
            down_out.storage().as_ref(),
            x.shape(),
        )?;
        let res2 = Tensor::new(
            std::sync::Arc::from(res2_storage),
            x.shape().clone(),
            grim_tensor::DType::F32,
            x.provenance().clone(),
            x.device().clone(),
        );
        let res2_id = tape.record_add(res1_id, down_id, res2.clone(), None);

        Ok((res2_id, res2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_backend_cpu::cpu_tensor;
    use grim_tensor::dtype::{DType, QuantProvenance};
    use grim_tensor::{RawTensor, Shape, TensorProvider};

    struct StubProvider {
        cfg: LlamaConfig,
    }

    impl StubProvider {
        fn new() -> Self {
            Self {
                cfg: LlamaConfig {
                    vocab_size: 256,
                    hidden_size: 32,
                    num_heads: 2,
                    num_kv_heads: 1,
                    head_dim: 16,
                    num_layers: 4,
                    intermediate_size: 64,
                    rms_norm_eps: 1e-5,
                    rope_theta: 10000.0,
                    max_seq_len: 64,

                    partial_rotary_factor: 1.0,
                    yarn: None,
                },
            }
        }
    }

    impl TensorProvider for StubProvider {
        fn get(&self, name: &str) -> grim_tensor::error::Result<RawTensor> {
            let c = &self.cfg;
            let (n, shape) = if name.contains("attn_norm") || name.contains("ffn_norm") {
                (c.hidden_size, vec![c.hidden_size])
            } else if name.contains("wq") || name.contains("wo") {
                let rows = if name.contains("wq") {
                    c.num_heads * c.head_dim
                } else {
                    c.hidden_size
                };
                let cols = if name.contains("wq") {
                    c.hidden_size
                } else {
                    c.num_heads * c.head_dim
                };
                (rows * cols, vec![rows, cols])
            } else if name.contains("wk") || name.contains("wv") {
                (
                    c.num_kv_heads * c.head_dim * c.hidden_size,
                    vec![c.num_kv_heads * c.head_dim, c.hidden_size],
                )
            } else if name.contains("w_gate") || name.contains("w_up") {
                (
                    c.intermediate_size * c.hidden_size,
                    vec![c.intermediate_size, c.hidden_size],
                )
            } else if name.contains("w_down") {
                (
                    c.hidden_size * c.intermediate_size,
                    vec![c.hidden_size, c.intermediate_size],
                )
            } else {
                let default_elems = c.hidden_size.max(128);
                (default_elems, vec![default_elems])
            };
            Ok(RawTensor {
                bytes: vec![0u8; n * 4],
                shape,
                dtype: DType::F32,
                provenance: QuantProvenance::GrimNative,
            })
        }

        fn meta(&self, _name: &str) -> grim_tensor::error::Result<grim_tensor::TensorMeta> {
            Ok(grim_tensor::TensorMeta {
                dtype: DType::F32,
                provenance: QuantProvenance::GrimNative,
                shape: vec![],
                fusion_mask: 0,
            })
        }
    }

    #[test]
    fn gradient_checkpoint_buffer_saves_and_retrieves() {
        let mut buf = GradientCheckpointBuffer::new();
        let t = cpu_tensor(vec![1.0, 2.0], Shape::new(vec![1, 2]));
        buf.save(0, t.clone());

        let retrieved = buf.get(0).unwrap();
        assert_eq!(retrieved.to_vec_f32().unwrap(), vec![1.0, 2.0]);
    }

    #[test]
    fn streaming_block_forward_runs_real_llama_block() {
        let provider = StubProvider::new();
        let cfg = provider.cfg.clone();
        let mut forward = StreamingBlockForward::new(4, cfg.hidden_size);
        let x = cpu_tensor(
            vec![0.5; cfg.hidden_size],
            Shape::new(vec![1, cfg.hidden_size]),
        );
        let positions = vec![0u32];

        let out = forward
            .forward_block(&provider, &cfg, 0, &x, Some(&positions))
            .unwrap();

        // Output must have same shape as input (real block forward ran without error)
        assert_eq!(out.shape().dims(), x.shape().dims());
    }

    #[test]
    fn streaming_block_recompute_matches_forward() {
        let provider = StubProvider::new();
        let cfg = provider.cfg.clone();
        let mut forward = StreamingBlockForward::new(4, cfg.hidden_size);
        let x = cpu_tensor(
            vec![0.5; cfg.hidden_size],
            Shape::new(vec![1, cfg.hidden_size]),
        );
        let positions = vec![0u32];

        let out = forward
            .forward_block(&provider, &cfg, 0, &x, Some(&positions))
            .unwrap();
        let recomputed = forward
            .recompute_block(&provider, &cfg, 0, Some(&positions))
            .unwrap();

        // Recomputed output must match the original forward output
        let out_vals = out.to_vec_f32().unwrap();
        let rec_vals = recomputed.to_vec_f32().unwrap();
        assert_eq!(out_vals.len(), rec_vals.len());
        for (a, b) in out_vals.iter().zip(rec_vals.iter()) {
            assert!((a - b).abs() < 1e-6, "recompute mismatch: {a} vs {b}");
        }
    }

    /// WI-INF3 Gate: placement changes device routing, never numerics.
    ///
    /// The route is attached over an asymmetric 2-GPU topology with every
    /// rank mapped back to `Device::Cpu`, so the routed block must produce
    /// byte-identical output to the plain forward no matter which rank the
    /// controller picks — the property "placement moves work, never math".
    /// Also pins the WI-INF5 first-cut fallback: the untrained (all-zero) MLP
    /// must steer the block to the faster card (rank 1), not sticky rank 0.
    #[test]
    fn scythe_route_parity_and_untrained_fallback_gate() {
        use grim_tensor::backend::{GpuCapability, ScytheLink};
        use std::sync::Arc;

        let provider = StubProvider::new();
        let cfg = provider.cfg.clone();
        let x = cpu_tensor(
            vec![0.5; cfg.hidden_size],
            Shape::new(vec![1, cfg.hidden_size]),
        );
        let positions = vec![0u32];

        let mut plain = StreamingBlockForward::new(4, cfg.hidden_size);
        let plain_vec = plain
            .forward_block(&provider, &cfg, 0, &x, Some(&positions))
            .unwrap()
            .to_vec_f32()
            .unwrap();

        let caps: Vec<GpuCapability> = [(8.0f32, 0usize), (80.0, 1usize)]
            .into_iter()
            .map(|(tflops, ordinal)| GpuCapability {
                tflops_fp16: tflops,
                tflops_fp8: 0.0,
                hbm_bandwidth_gbps: 100.0,
                vram_free_bytes: 16 << 30,
                throttle_pct: 0.0,
                ordinal,
            })
            .collect();
        let links = vec![
            ScytheLink::PeerDirect,
            ScytheLink::Host,
            ScytheLink::Host,
            ScytheLink::PeerDirect,
        ];

        let mut routed = StreamingBlockForward::new(4, cfg.hidden_size);
        routed.attach_scythe_route(ScytheRoute {
            ctrl: Arc::new(std::sync::Mutex::new(crate::scythe2::C2plrController::new(
                4, 2, 10.0,
            ))),
            profiler: None,
            caps,
            links,
            caps_epoch: 0,
            device_for_rank: Arc::new(|_rank| Device::Cpu),
        });
        let routed_vec = routed
            .forward_block(&provider, &cfg, 0, &x, Some(&positions))
            .unwrap()
            .to_vec_f32()
            .unwrap();

        assert_eq!(
            plain_vec, routed_vec,
            "SCYTHE-2 placement routing must not change block numerics"
        );

        let ctrl = routed.scythe_route.as_ref().unwrap().ctrl.clone();
        let ctrl = ctrl.lock().unwrap_or_else(|e| e.into_inner());
        let placed_rank = ctrl
            .cache
            .get(0, crate::scythe2::bucketize(&[1, cfg.hidden_size]))
            .map(|p| p.ranks[0]);
        assert_eq!(
            placed_rank,
            Some(1),
            "untrained MLP must fall back to the WaveTune latency estimate \
             (argmin) and land on the 80-TFLOPS card, not sticky rank 0"
        );
    }
}
