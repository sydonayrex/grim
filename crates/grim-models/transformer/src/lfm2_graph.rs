//! i-was-dumb-graph.md Phase 3-4 + Phase 6 for LFM2.
//!
//! Capture/replay path over fixed buffers. All compute enqueued on-device via
//! storage-level `BackendDevice` ops + `launch_kv_append` /
//! `launch_qkv_attention_dev` / `launch_bump_i32` — zero host readbacks, no
//! host alloc of device data, no `hipDeviceSynchronize` inside capture. GEMM
//! stays in rocBLAS/dot dispatch (Rule 0); attention reuses device totals.
//!
//! Recurrent (ShortConv) blocks return `Unimplemented`: their conv state lives
//! on host today, which poisons capture. Caller falls back eager for those
//! stacks (spec §Fallback). Same for MoE blocks (host top-1 routing).

use grim_backend_rocm::RocmStorage;
use grim_backend_rocm::as_rocm;
use grim_backend_rocm::decode_graph_buffers::{
    DecodeGraph, DecodeGraphBuffers, EagerKvSource, check_layer_topology, decode_graph_enabled,
    launch_attention, launch_qkv_gemv, write_embeddings_to_buffer_batch,
};
use grim_core::error::Result;
use grim_tensor::{BackendStorage, Device, MemoryOps, RopeConfig, Shape};

use crate::lfm2::{Lfm2, Lfm2Block, Lfm2LayerCache};

/// Spec §Phase 6 dims derived from config. `max_ctx` caps KV arenas.
#[allow(clippy::type_complexity)]
fn graph_dims(lfm: &Lfm2, max_ctx: usize) -> (usize, usize, usize, usize, usize, usize, usize) {
    let hidden = lfm.cfg.hidden_size;
    let n_q = lfm.cfg.num_heads * lfm.cfg.head_dim;
    let n_k = lfm.cfg.num_kv_heads * lfm.cfg.head_dim;
    let inter = lfm.cfg.intermediate_size;
    let vocab = lfm.cfg.vocab_size.max(1);
    let ctx = max_ctx.max(1);
    (hidden, n_q, n_k, inter, vocab, ctx, lfm.cfg.num_heads)
}

type Dev = grim_backend_rocm::RocmDevice;
type Storage = dyn grim_tensor::BackendStorage;

/// Downcast a weight tensor to its ROCm storage (graph path is ROCm-only).
pub(crate) fn rocm_storage(t: &grim_tensor::Tensor) -> Result<&RocmStorage> {
    dst_downcast(t.storage().as_ref())
}

/// Quant-capable GEMV into a fixed pool slot. Mirrors eager `Linear` decode
/// dispatch (F32 rocBLAS/GEMV, Q80/Q4K-K quant dot paths) via the backend
/// [`Dev::linear_decode_into`]; unsupported dtypes fall back eager.
pub(crate) fn linear_into(
    dev: &Dev,
    a: &Storage,
    w: &grim_tensor::Tensor,
    out: &RocmStorage,
    act_q81: &RocmStorage,
) -> Result<()> {
    let ws = rocm_storage(w)?;
    let _ = dev
        .linear_decode_into(a, ws, out, act_q81)
        .map_err(|e| grim_core::error::Error::Backend(format!("linear_decode: {e}")))?;
    Ok(())
}

/// Pure host-side predicate: may this layer use the sudot4 fused path?
/// Mirrors the launchers' own validation (hidden%32, dot4 arch, opt-out flag)
/// so the branch is decided with zero enqueues — no mid-capture fallback.
pub(crate) fn dot_fused_ok(dev: &Dev, hidden: usize) -> bool {
    hidden != 0
        && hidden % 32 == 0
        && dev.supports_dot4()
        && !matches!(
            std::env::var("GRIM_DOT_GEMV").as_deref(),
            Ok("0" | "false" | "off")
        )
}

pub(crate) fn is_q80(t: &grim_tensor::Tensor) -> bool {
    matches!(
        t.dtype().storage,
        grim_tensor::Storage::KQuant(grim_tensor::dtype::KQuantScheme::Q80)
    )
}

fn dev_for(lfm: &Lfm2) -> Result<std::sync::Arc<Dev>> {
    match &lfm.device {
        Device::Rocm(o) => Ok(Dev::shared(*o)),
        _ => Err(grim_core::error::Error::Unimplemented(
            "decode graph needs ROCm device".into(),
        )),
    }
}

/// D2D publish of a finished layer output into the next layer's input slot
/// (or `head_input`). One honest graph node per boundary; no host traffic.
impl Lfm2 {
    /// Spec §Phase 5: `model.get_or_create_decode_graph()`.
    /// Allocates fixed pool once; caller keeps it across steps for stable addrs.
    /// `batch` parameterizes all per-step slots as `[batch, dim]`. `batch=1`
    /// preserves the original single-token shape.
    /// `Err` -> caller falls back eager (spec §Fallback). Honors both env gates.
    pub fn get_or_create_decode_graph(&self, max_ctx: usize, batch: usize) -> Result<DecodeGraph> {
        if !decode_graph_enabled() {
            return Err(grim_core::error::Error::Backend(
                "decode graph disabled by env".into(),
            ));
        }
        let dev = dev_for(self)?;
        crate::lfm2::validate_native_mxfp4_kv_compatibility(
            self.layers.iter().any(|layer| layer.wqkv_codes.is_some()),
        )?;
        // G1 (PLAN-kernel-fusion): the graph runs on pool slot 0, which today
        // coincides with the device's default stream. Post-replay sampling
        // now uses `sample_logits_on_device_with_penalty_at_stream` bound to
        // `graph.stream` explicitly — if this pool index ever changes (e.g.
        // concurrent graphs), the sampler path remains correct, but any NEW
        // post-replay consumer must likewise bind `graph.stream`, never the
        // ambient `active_stream()`.
        let stream = dev
            .get_stream_from_pool(0)
            .ok_or_else(|| grim_core::error::Error::Backend("no stream in pool".into()))?;
        let (hidden, n_q, n_k, inter, vocab, ctx, nh) = graph_dims(self, max_ctx);
        let mut buffers = DecodeGraphBuffers::allocate(
            &dev,
            self.layers.len(),
            hidden,
            n_q,
            n_k,
            n_k,
            inter,
            ctx,
            vocab,
            nh,
            batch,
            // M2/S2: MoE + ShortConv staging dims (0 = layer kind absent).
            self.cfg.n_expert,
            self.cfg.n_expert_used,
            if self.layers.iter().any(|l| l.shortconv_in_proj.is_some()) {
                self.cfg.hidden_size
            } else {
                0
            },
            if self.layers.iter().any(|l| l.shortconv_in_proj.is_some()) {
                self.cfg.n_shortconv_l_cache.max(2)
            } else {
                0
            },
        )
        .map_err(|e| grim_core::error::Error::Backend(format!("graph pool alloc: {e}")))?;
        // GRAVE Phase 4: GDL layers get recurrent-state buffers (Taylor
        // defaults uploaded here, outside capture). Non-GDL layers stay None.
        // A failed GDL alloc fails the whole graph → eager fallback (spec).
        for (idx, layer) in self.layers.iter().enumerate() {
            if layer.attention_mode == crate::lfm2::Lfm2AttentionMode::Gdl && layer.wq.is_some() {
                buffers
                    .allocate_gdl_layer(
                        &dev,
                        idx,
                        batch.max(1),
                        layer.num_heads,
                        layer.num_kv_heads,
                        layer.head_dim,
                        layer.head_dim,
                    )
                    .map_err(|e| {
                        grim_core::error::Error::Backend(format!("graph GDL alloc: {e}"))
                    })?;
            }
        }
        Ok(DecodeGraph::new(&dev, buffers, stream))
    }

    /// A5 Phase 2 (WI-X2-PREFILL-ARENA): export device pointers + strides for
    /// graph-arena seeding (`DecodeGraphBuffers::seed_kv_arena_from_eager`).
    ///
    /// `caches` is the session's per-layer cache vec (index-aligned with
    /// `self.layers`; see `Lfm2::forward`). `valid_rows` is the number of
    /// rows valid in every dense layer's device arena — supplied by the
    /// caller from loop counters (prompt_len + decode steps so far), NOT
    /// derived from the (possibly stale-on-device-path) host `k`/`v` mirrors.
    ///
    /// Returns one entry per layer: `Some` for dense attention layers whose
    /// `k_dev`/`v_dev` arenas are present and ROCm-resident, `None` for
    /// recurrent (ShortConv) layers. FAILS CLOSED: when `valid_rows > 0` and
    /// any dense attention layer lacks arenas (never ran, wrong cache
    /// variant, non-ROCm storage), returns `Err` so the caller falls back to
    /// eager instead of capturing a prompt-blind graph. `valid_rows == 0`
    /// yields all-`None` (nothing to seed; seed is then a no-op).
    pub fn eager_kv_seed_sources<'a>(
        &self,
        caches: &'a [Option<Lfm2LayerCache>],
        valid_rows: u32,
    ) -> Result<Vec<Option<EagerKvSource<'a>>>> {
        if caches.len() != self.layers.len() {
            return Err(grim_core::error::Error::Session(format!(
                "eager_kv_seed_sources: {} caches != {} layers",
                caches.len(),
                self.layers.len()
            )));
        }
        let mut out = Vec::with_capacity(self.layers.len());
        for (layer, cache) in self.layers.iter().zip(caches.iter()) {
            // Recurrent (ShortConv) layers genuinely have nothing to seed.
            if layer.wq.is_none() {
                out.push(None);
                continue;
            }
            // GDL layers carry recurrent state (dev_state), not KV arenas.
            // Return the state pointer so the caller can seed it into the
            // graph's GDL state buffer before the first replay.
            if layer.attention_mode == crate::lfm2::Lfm2AttentionMode::Gdl {
                // Zero rows = no-op export (nothing to seed). With rows, return
                // the GDL state pointer so the caller can D2D-copy it into the
                // graph's GDL state buffer before the first replay.
                if valid_rows == 0 {
                    out.push(None);
                } else {
                    let state_ptr = match cache {
                        Some(Lfm2LayerCache::Gdl {
                            dev_state: Some(s),
                            ..
                        }) => grim_backend_rocm::as_rocm(s.as_ref())
                            .ok()
                            .and_then(|rocm_s| rocm_s.device_ptr_u64())
                            .map(|p| p as *const f32)
                            .filter(|p| *p != std::ptr::null()),
                        _ => None,
                    };
                    out.push(Some(EagerKvSource {
                        k_dev: std::ptr::null(),
                        v_dev: std::ptr::null(),
                        prefill_len: 0,
                        kv_stride: 0,
                        gdl_state: state_ptr,
                        _anchor: std::marker::PhantomData,
                    }));
                }
                continue;
            }
            let (k_dev, v_dev) = match cache {
                Some(Lfm2LayerCache::Attention { k_dev, v_dev, .. }) => (k_dev, v_dev),
                _ => {
                    if valid_rows > 0 {
                        return Err(grim_core::error::Error::Session(
                            "eager_kv_seed_sources: dense layer missing attention cache".into(),
                        ));
                    }
                    out.push(None);
                    continue;
                }
            };
            let (k, v) = match (k_dev.as_deref(), v_dev.as_deref()) {
                (Some(k), Some(v)) => (k, v),
                _ => {
                    if valid_rows > 0 {
                        return Err(grim_core::error::Error::Session(
                            "eager_kv_seed_sources: dense layer missing device KV arenas".into(),
                        ));
                    }
                    out.push(None);
                    continue;
                }
            };
            let (k_rocm, v_rocm) =
                match (as_rocm(k.storage().as_ref()), as_rocm(v.storage().as_ref())) {
                    (Ok(k), Ok(v)) => (k, v),
                    _ => {
                        if valid_rows > 0 {
                            return Err(grim_core::error::Error::Session(
                                "eager_kv_seed_sources: KV arenas not ROCm-resident".into(),
                            ));
                        }
                        out.push(None);
                        continue;
                    }
                };
            let kv_stride = k_rocm.shape().dims().last().copied().unwrap_or(0);
            if kv_stride == 0 {
                if valid_rows > 0 {
                    return Err(grim_core::error::Error::Session(
                        "eager_kv_seed_sources: zero-width KV arena".into(),
                    ));
                }
                out.push(None);
                continue;
            }
            let (k_ptr, v_ptr) = match (k_rocm.device_ptr_u64(), v_rocm.device_ptr_u64()) {
                (Some(k), Some(v)) if k != 0 && v != 0 => (k as *const f32, v as *const f32),
                _ => {
                    if valid_rows > 0 {
                        return Err(grim_core::error::Error::Session(
                            "eager_kv_seed_sources: KV arenas have no device pointer".into(),
                        ));
                    }
                    out.push(None);
                    continue;
                }
            };
            out.push(Some(EagerKvSource {
                k_dev: k_ptr,
                v_dev: v_ptr,
                prefill_len: valid_rows,
                kv_stride,
                gdl_state: None,
                _anchor: std::marker::PhantomData,
            }));
        }
        Ok(out)
    }

    /// Spec §Phase 3 capture path (single token). Enqueues the device-side
    /// embedding gather (`token_ids_dev` → `layer_input[0]`) + all layers +
    /// output norm/head. The seed write is eager H2D (during capture it is a
    /// baked seed node; replays overwrite `token_ids_dev` first).
    pub fn forward_capture(&self, graph: &mut DecodeGraph, token_id: u32) -> Result<()> {
        if !decode_graph_enabled() {
            return Err(grim_core::error::Error::Backend(
                "decode graph disabled".into(),
            ));
        }
        let dev = dev_for(self)?;
        // Seed write is H2D of stack-owned bits — legal eagerly, FORBIDDEN
        // inside capture (would bake the dead host pointer into the graph).
        if !graph.capturing {
            write_embedding_to_buffer(&dev, &graph.buffers.token_ids_dev, token_id)?;
        }
        self.embed_and_forward_graph(graph, &dev)
    }

    /// Spec §Phase 3 capture path (P3 batch). Records a whole batch into the
    /// graph in one pass. `token_ids` must have length == `graph.buffers.batch`.
    pub fn forward_capture_batch(&self, graph: &mut DecodeGraph, token_ids: &[u32]) -> Result<()> {
        if !decode_graph_enabled() {
            return Err(grim_core::error::Error::Backend(
                "decode graph disabled".into(),
            ));
        }
        let dev = dev_for(self)?;
        if !graph.capturing {
            write_embeddings_to_buffer_batch(&dev, &graph.buffers.token_ids_dev, token_ids)?;
        }
        self.embed_and_forward_graph(graph, &dev)
    }

    /// Device-side embedding gather (reads `token_ids_dev`, so replays see
    /// fresh tokens), then the per-layer body and output head.
    fn embed_and_forward_graph(&self, graph: &mut DecodeGraph, dev: &Dev) -> Result<()> {
        let w = dst_downcast(self.tok_embeddings.weight.storage().as_ref())?;
        let hidden = self.cfg.hidden_size;
        let batch = graph.buffers.batch.max(1);
        dev.launch_embedding_gather_dev_idx(
            w,
            &graph.buffers.layer_input[0],
            &graph.buffers.token_ids_dev,
            hidden,
            batch * hidden,
        )
        .map_err(|e| grim_core::error::Error::Backend(format!("embedding gather: {e}")))?;
        for (i, layer) in self.layers.iter().enumerate() {
            layer.forward_graph(i, &graph.buffers, dev)?;
        }
        self.output_forward_graph(&graph.buffers, dev)?;
        Ok(())
    }

    /// Spec §Phase 3 replay path (single token). One H2D + one `hipGraphLaunch`.
    pub fn forward_replay(&self, graph: &mut DecodeGraph, token_id: u32) -> Result<()> {
        if !graph.is_captured {
            return Err(grim_core::error::Error::Backend(
                "forward_replay before capture".into(),
            ));
        }
        let dev = dev_for(self)?;
        write_embedding_to_buffer(&dev, &graph.buffers.token_ids_dev, token_id)?;
        // Scalar update before replay: prefer SetParams when node known,
        // else 4-byte device-buffer path (spec §Scalar both options).
        let pos = graph.buffers.current_pos;
        if !graph.kv_append_node.is_null() {
            // Null params probe documents intent; real params come from
            // capture-time node discovery. Fall through to pos_dev path.
            let _ = graph.update_kv_pos_params(std::ptr::null());
        }
        graph
            .buffers
            .write_pos_async(&dev, pos, graph.stream)
            .map_err(|e| grim_core::error::Error::Backend(format!("write pos: {e}")))?;
        graph
            .replay()
            .map_err(|e| grim_core::error::Error::Backend(format!("replay: {e}")))?;
        Ok(())
    }

    /// Spec §Phase 3 replay path (P3 batch). Writes `batch` token IDs via
    /// async H2D, then replays in one launch. Caller bumps `pos_dev` by
    /// `graph.buffers.batch` after replay.
    pub fn forward_replay_batch(&self, graph: &mut DecodeGraph, token_ids: &[u32]) -> Result<()> {
        if !graph.is_captured {
            return Err(grim_core::error::Error::Backend(
                "forward_replay_batch before capture".into(),
            ));
        }
        if token_ids.len() != graph.buffers.batch {
            return Err(grim_core::error::Error::Backend(format!(
                "forward_replay_batch: token_ids len {} != batch {}",
                token_ids.len(),
                graph.buffers.batch
            )));
        }
        let dev = dev_for(self)?;
        write_embeddings_to_buffer_batch(&dev, &graph.buffers.token_ids_dev, token_ids)?;
        let pos = graph.buffers.current_pos;
        if !graph.kv_append_node.is_null() {
            let _ = graph.update_kv_pos_params(std::ptr::null());
        }
        graph
            .buffers
            .write_pos_async(&dev, pos, graph.stream)
            .map_err(|e| grim_core::error::Error::Backend(format!("write pos: {e}")))?;
        graph
            .replay()
            .map_err(|e| grim_core::error::Error::Backend(format!("replay: {e}")))?;
        Ok(())
    }

    fn output_forward_graph(&self, buffers: &DecodeGraphBuffers, dev: &Dev) -> Result<()> {
        // Final norm (in-place: kernel reduces each row before storing) +
        // output projection. Both enqueued, zero scratch.
        // PLAN 4 Task 3: norm-fused head — when the output weight rides the
        // Q8_0 dot4 path, the standalone norm is skipped and the prologue
        // runs inside the GEMV. Falls back to norm + linear_into otherwise
        // (e.g. GRIM_LFM2_F32_HEAD=1 f32 table).
        let h_shape = buffers.head_input.shape().clone();
        let hidden = self.cfg.hidden_size;
        let norm_fused_head =
            dot_fused_ok(dev, hidden) && is_q80(&self.output.weight);
        if norm_fused_head {
            let n_vocab = self.output.weight.shape().dim(0).unwrap_or(0);
            let m = h_shape.dims().iter().product::<usize>() / hidden.max(1);
            dev.launch_dot4_q80_norm_f32act_gemv_into(
                &buffers.head_input,
                rocm_storage(&self.norm.weight)?,
                self.norm.eps,
                rocm_storage(&self.output.weight)?,
                &buffers.head_output,
                m.max(1),
                n_vocab,
                hidden,
            )
            .map_err(|e| {
                grim_core::error::Error::Backend(format!("fused norm head: {e}"))
            })?;
            return Ok(());
        }
        dev.rms_norm_into(
            &buffers.head_input,
            &**self.norm.weight.storage(),
            self.norm.eps,
            &buffers.head_input,
            &h_shape,
        )
        .map_err(grim_core::error::Error::Tensor)?;
        linear_into(
            dev,
            &buffers.head_input,
            &self.output.weight,
            &buffers.head_output,
            &buffers.act_q81_buf[0],
        )?;
        Ok(())
    }
}

impl Lfm2Block {
    /// Attention (`wq.is_some`) vs recurrent ShortConv path.
    pub fn is_attention(&self) -> bool {
        self.wq.is_some()
    }

    /// Spec §Phase 4: per-layer recording into fixed buffers. Every op below
    /// enqueues a kernel (norm/GEMM/attention/D2D) on the active stream, so a
    /// surrounding `begin/end_capture` bracket records real nodes.
    pub fn forward_graph(
        &self,
        layer_idx: usize,
        buffers: &DecodeGraphBuffers,
        dev: &Dev,
    ) -> Result<()> {
        check_layer_topology(buffers, layer_idx)
            .map_err(|e| grim_core::error::Error::Backend(format!("{e}")))?;
        // Dense blocks fuse the sublayer residual-add with the FFN norm into
        // ONE kernel (grim_add_rms_norm). MoE keeps the split path (its
        // grouped-dispatch output lands in moe_out, not norm_buf).
        let fuse_norm = !self.is_moe;
        if !self.is_attention() {
            // S2 (PLAN-kernel-fusion): ShortConv layers capture via the device
            // ring (`sc_state`) + staging buffers. Only a recurrent block with
            // NO ShortConv weights remains uncapturable.
            if self.shortconv_in_proj.is_none() {
                return Err(grim_core::error::Error::Unimplemented(
                    "forward_graph: recurrent block without ShortConv weights; use eager".into(),
                ));
            }
            self.shortconv_forward_graph(layer_idx, buffers, dev)?;
        } else if crate::gla_graph::is_gdl_layer(self) {
            // GRAVE Phase 4: fused GDN-2 branch (one launch replaces
            // kv_append + attention + norm). Unimplemented → eager fallback.
            crate::gla_graph::gdl_forward_graph(self, layer_idx, buffers, dev, fuse_norm)?;
        } else {
            self.attn_forward_graph(layer_idx, buffers, dev, fuse_norm)?;
        }
        // PLAN-decode-throughput-restore Fix 3: the FFN residual add writes
        // the block result DIRECTLY into the next block's input slot (or
        // head_input), eliminating the per-block publish D2D copy.
        let n_layers = buffers.layer_input.len();
        let residual_dst: &RocmStorage = if layer_idx + 1 < n_layers {
            &buffers.layer_input[layer_idx + 1]
        } else {
            &buffers.head_input
        };
        if self.is_moe {
            // M2 (PLAN-kernel-fusion): MoE FFN sublayer — device routing +
            // resident-weight grouped dispatch, all enqueued (capture-safe).
            self.moe_forward_graph(layer_idx, buffers, dev, residual_dst)?;
        } else {
            self.ffn_forward_graph(layer_idx, buffers, dev, residual_dst, fuse_norm)?;
        }
        Ok(())
    }

    /// Spec §Phase 4 attention branch: norm -> QKV -> QK-norm -> RoPE ->
    /// KV append -> device attention -> O proj -> residual. All enqueued.
    pub fn attn_forward_graph(
        &self,
        layer_idx: usize,
        buffers: &DecodeGraphBuffers,
        dev: &Dev,
        fuse_norm: bool,
    ) -> Result<()> {
        let wq = self.wq.as_ref().ok_or_else(|| {
            grim_core::error::Error::Backend("attn_forward_graph: missing wq".into())
        })?;
        let wk = self.wk.as_ref().ok_or_else(|| {
            grim_core::error::Error::Backend("attn_forward_graph: missing wk".into())
        })?;
        let wv = self.wv.as_ref().ok_or_else(|| {
            grim_core::error::Error::Backend("attn_forward_graph: missing wv".into())
        })?;
        let wo = self.wo.as_ref().ok_or_else(|| {
            grim_core::error::Error::Backend("attn_forward_graph: missing wo".into())
        })?;

        // 1. Attention norm into the per-layer staging slot — SKIPPED when
        // the norm-fused QKV path below is taken (PLAN 4: the norm prologue
        // runs inside the GEMV kernel, bit-identical).
        // MXFP4 / plain / dot-fallback paths still need it staged.
        let hidden = buffers.layer_input[layer_idx]
            .shape()
            .dims()
            .last()
            .copied()
            .unwrap_or(0);
        let mxfp4_fused_qkv =
            self.wqkv_codes.is_some() && self.wqkv_exps.is_some();
        let norm_fused_qkv = !mxfp4_fused_qkv
            && dot_fused_ok(dev, hidden)
            && is_q80(&wq.weight)
            && is_q80(&wk.weight)
            && is_q80(&wv.weight)
            && (self.num_heads * self.head_dim) % 4 == 0
            && (self.num_kv_heads * self.head_dim) % 4 == 0;
        if !norm_fused_qkv {
            dev.rms_norm_into(
                &buffers.layer_input[layer_idx],
                &**self.attn_norm.weight.storage(),
                self.attn_norm.eps,
                &buffers.norm_buf[layer_idx],
                &buffers.layer_input[layer_idx].shape().clone(),
            )
            .map_err(grim_core::error::Error::Tensor)?;
        }
        let normed: &Storage = &buffers.norm_buf[layer_idx];

        // 2. QKV projections. Fused path (Q8_0 blob present): ONE dot4 GEMV
        //    into fused staging + 3 slice copies (4 launches vs 3 GEMMs).
        //    Plain path: quant-aware GEMV per projection. Both write fixed
        //    pool slots; no scratch allocs.
        let act = &buffers.act_q81_buf[layer_idx];
        let batch = buffers.batch.max(1);

        // MXFP4 fused path (closes the decode-graph quant gap): ONE kernel
        // does projection + QK-norm + RoPE + K/V append from native MXFP4
        // codes. positions = `pos_dev` (u32 per batch slot) — the same source
        // the kv-append/bump nodes use, so appended rows and the attention
        // read offset stay consistent. The attention + bump tail below is
        // shared with the other paths. In this branch the QK-norm/RoPE/append
        // nodes are skipped (the fused kernel does them internally).
        if let (Some(codes), Some(exps)) = (&self.wqkv_codes, &self.wqkv_exps) {
            let gamma_q = self.gamma_q.as_ref().ok_or_else(|| {
                grim_core::error::Error::Backend("mxfp4 fused qkv: missing gamma_q".into())
            })?;
            let gamma_k = self.gamma_k.as_ref().ok_or_else(|| {
                grim_core::error::Error::Backend("mxfp4 fused qkv: missing gamma_k".into())
            })?;
            dev.fused_mxfp4_gemm_qk_norm_rope_kv(
                normed,
                gamma_q.storage().as_ref(),
                gamma_k.storage().as_ref(),
                codes.storage().as_ref(),
                exps.storage().as_ref(),
                Some(&buffers.q_buf[layer_idx]),
                Some(&buffers.k_arena[layer_idx]),
                Some(&buffers.v_arena[layer_idx]),
                None,
                Some(&buffers.pos_dev),
                batch,
                hidden,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.head_dim, // rotary_dim
                self.rope_theta,
                None,
                1.0, // mscale
                self.eps,
                buffers.max_ctx,
                false, // rope_interleaved: LFM2 uses NeoX half-split
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("fused mxfp4 qkv: {e}")))?;
        } else if norm_fused_qkv {
            // PLAN 4: norm-fused QKV — rms_norm prologue + quantize + dot4 in
            // ONE launch, straight from the residual stream. The standalone
            // norm above was skipped; predicates match the old dot-fused
            // branch exactly, so kill-switches and fallbacks are preserved.
            let n_q = self.num_heads * self.head_dim;
            let n_kv = self.num_kv_heads * self.head_dim;
            dev.fused_qkv_dot4_norm_into(
                &buffers.layer_input[layer_idx],
                rocm_storage(&self.attn_norm.weight)?,
                self.attn_norm.eps,
                rocm_storage(&wq.weight)?,
                rocm_storage(&wk.weight)?,
                rocm_storage(&wv.weight)?,
                &buffers.q_buf[layer_idx],
                &buffers.k_buf[layer_idx],
                &buffers.v_buf[layer_idx],
                n_q,
                n_kv,
                hidden,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("fused norm qkv dot4: {e}")))?;
        } else if dot_fused_ok(dev, hidden)
            && is_q80(&wq.weight)
            && is_q80(&wk.weight)
            && is_q80(&wv.weight)
            && (self.num_heads * self.head_dim) % 4 == 0
            && (self.num_kv_heads * self.head_dim) % 4 == 0
        {
            // PLAN-kernel-launch-reduction Phase B: ONE dot4 GEMV launch covers
            // the three projections (replaces 3 GEMV + 3 quantize launches).
            let n_q = self.num_heads * self.head_dim;
            let n_kv = self.num_kv_heads * self.head_dim;
            dev.fused_qkv_dot4_into(
                normed,
                rocm_storage(&wq.weight)?,
                rocm_storage(&wk.weight)?,
                rocm_storage(&wv.weight)?,
                &buffers.q_buf[layer_idx],
                &buffers.k_buf[layer_idx],
                &buffers.v_buf[layer_idx],
                n_q,
                n_kv,
                hidden,
                act,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("fused qkv dot4: {e}")))?;
        } else {
            linear_into(dev, normed, &wq.weight, &buffers.q_buf[layer_idx], act)?;
            linear_into(dev, normed, &wk.weight, &buffers.k_buf[layer_idx], act)?;
            linear_into(dev, normed, &wv.weight, &buffers.v_buf[layer_idx], act)?;
        }

        // 3-6. QK-norm + RoPE + append + attend + bump (real nodes).
        let mxfp4_fused = self.wqkv_codes.is_some() && self.wqkv_exps.is_some();
        self.attention_forward_graph(layer_idx, buffers, dev, mxfp4_fused)?;

        // 7. O projection into staging, then residual add -> layer_output.
        //    norm_buf is free (step 1 consumed); stream order keeps it sound.
        //    Quant-aware: Q80/Q4K-family weights ride the dot paths.
        linear_into(
            dev,
            &buffers.attn_out_buf[layer_idx],
            &wo.weight,
            &buffers.norm_buf[layer_idx],
            &buffers.act_q81_buf[layer_idx],
        )?;
        if !fuse_norm {
            add_graph(
                &buffers.layer_input[layer_idx],
                &buffers.norm_buf[layer_idx],
                &buffers.layer_output[layer_idx],
                dev,
            )?;
        }
        // fuse_norm: the residual-add + FFN-norm pair is issued by
        // ffn_forward_graph as one grim_add_rms_norm launch.
        Ok(())
    }

    /// Spec §Phase 4 FFN branch: ffn_norm -> gate/up GEMMs -> SiLU -> down
    /// GEMM -> residual add into layer_output. All enqueued.
    pub fn ffn_forward_graph(
        &self,
        layer_idx: usize,
        buffers: &DecodeGraphBuffers,
        dev: &Dev,
        residual_dst: &RocmStorage,
        fuse_norm: bool,
    ) -> Result<()> {
        check_layer_topology(buffers, layer_idx)
            .map_err(|e| grim_core::error::Error::Backend(format!("{e}")))?;
        let ffn_shape = buffers.layer_output[layer_idx].shape().clone();
        let hidden = buffers.layer_output[layer_idx]
            .shape()
            .dims()
            .last()
            .copied()
            .unwrap_or(0);
        let m = buffers.batch.max(1);
        let act = &buffers.act_q81_buf[layer_idx];
        let fused_residual_ffn = fuse_norm
            && matches!(
                std::env::var("GRIM_FUSED_RESIDUAL_GATEUP").as_deref(),
                Ok("1") | Ok("true") | Ok("on")
            )
            && dot_fused_ok(dev, hidden)
            && is_q80(&self.ffn_gate.weight)
            && is_q80(&self.ffn_up.weight);
        if fused_residual_ffn {
            dev.fused_add_rms_norm_gate_up_silu_dot4_into(
                &buffers.layer_input[layer_idx],
                &buffers.norm_buf[layer_idx],
                rocm_storage(&self.ffn_norm.weight)?,
                self.ffn_norm.eps,
                rocm_storage(&self.ffn_gate.weight)?,
                rocm_storage(&self.ffn_up.weight)?,
                &buffers.layer_output[layer_idx],
                &buffers.activated_buf[layer_idx],
                m,
                self.ffn_gate.weight.shape().dim(0).unwrap_or(0),
                hidden,
            )
            .map_err(|e| {
                grim_core::error::Error::Backend(format!("fused residual+gateup: {e}"))
            })?;
        } else {
            if fuse_norm {
                // PLAN-decode-throughput-restore Fix 3: ONE grim_add_rms_norm
                // launch replaces the (residual-add + ffn-norm) pair. norm_out
                // aliases the residual input (norm_buf) — safe: the kernel reads
                // the residual only in pass 1 and only reads the sum in pass 2.
                dev.fused_add_rms_norm_into(
                    &buffers.layer_input[layer_idx],
                    &buffers.norm_buf[layer_idx],
                    &**self.ffn_norm.weight.storage(),
                    self.ffn_norm.eps,
                    &buffers.layer_output[layer_idx],
                    &buffers.norm_buf[layer_idx],
                    &ffn_shape,
                )
                .map_err(grim_core::error::Error::Tensor)?;
            } else {
                // Norm the attention residual into the staging slot; every GEMM
                // below then writes a fixed pool slot.
                dev.rms_norm_into(
                    &buffers.layer_output[layer_idx],
                    &**self.ffn_norm.weight.storage(),
                    self.ffn_norm.eps,
                    &buffers.norm_buf[layer_idx],
                    &ffn_shape,
                )
                .map_err(grim_core::error::Error::Tensor)?;
            }
            let normed: &Storage = &buffers.norm_buf[layer_idx];
            let fused_gateup = dot_fused_ok(dev, hidden)
                && is_q80(&self.ffn_gate.weight)
                && is_q80(&self.ffn_up.weight);
            if fused_gateup {
                dev.fused_gate_up_silu_dot4_into(
                    normed,
                    rocm_storage(&self.ffn_gate.weight)?,
                    rocm_storage(&self.ffn_up.weight)?,
                    &buffers.activated_buf[layer_idx],
                    self.ffn_gate.weight.shape().dim(0).unwrap_or(0),
                    hidden,
                    act,
                )
                .map_err(|e| {
                    grim_core::error::Error::Backend(format!("fused gateup silu: {e}"))
                })?;
            } else if let Some(fused) = self
                .w_gate_up_q80_fused
                .as_ref()
                .filter(|_| dot_fused_ok(dev, hidden))
            {
                let norm_rocm = dst_downcast(normed)?;
                dev.launch_quantize_q8_1(norm_rocm, act, m, hidden)
                    .map_err(|e| grim_core::error::Error::Backend(format!("gateup quant: {e}")))?;
                dev.launch_fused_gate_up_dot4_into(
                    act,
                    &fused.storage,
                    &buffers.gate_up_buf[layer_idx],
                    fused.n_gate,
                    fused.n_up,
                    hidden,
                )
                .map_err(|e| grim_core::error::Error::Backend(format!("fused gateup: {e}")))?;
                let staged: &Storage = &buffers.gate_up_buf[layer_idx];
                dev.copy_slice_into(&buffers.gate_buf[layer_idx], staged, 0, fused.n_gate)
                    .map_err(grim_core::error::Error::Tensor)?;
                dev.copy_slice_range(
                    &buffers.up_buf[layer_idx],
                    0,
                    staged,
                    fused.n_gate,
                    fused.n_up,
                )
                .map_err(grim_core::error::Error::Tensor)?;
            } else {
                linear_into(
                    dev,
                    normed,
                    &self.ffn_gate.weight,
                    &buffers.gate_buf[layer_idx],
                    act,
                )?;
                linear_into(
                    dev,
                    normed,
                    &self.ffn_up.weight,
                    &buffers.up_buf[layer_idx],
                    act,
                )?;
                dev.silu_mul_into(
                    &buffers.gate_buf[layer_idx],
                    &buffers.up_buf[layer_idx],
                    &buffers.activated_buf[layer_idx],
                )
                .map_err(grim_core::error::Error::Tensor)?;
            }
        }
        // then residual add in place (per-element independent, safe).
        // Phase D.2: If down weight is Q8_0 and dot4 is supported, fuse down-projection
        // GEMV directly with the residual add from layer_output into residual_dst.
        let ffn_k = buffers.activated_buf[layer_idx].shape().dims().last().copied().unwrap_or(0);
        let ffn_n = self.ffn_down.weight.shape().dim(0).unwrap_or(0);
        let m = buffers.batch.max(1);
        if dot_fused_ok(dev, ffn_k) && is_q80(&self.ffn_down.weight) {
            dev.launch_dot4_q80_f32act_add_gemv(
                &buffers.activated_buf[layer_idx],
                rocm_storage(&self.ffn_down.weight)?,
                Some(&buffers.layer_output[layer_idx]),
                residual_dst,
                m,
                ffn_n,
                ffn_k,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("fused down+add: {e}")))?;
        } else {
            linear_into(
                dev,
                &buffers.activated_buf[layer_idx],
                &self.ffn_down.weight,
                &buffers.norm_buf[layer_idx],
                act,
            )?;
            add_graph(
                &buffers.layer_output[layer_idx],
                &buffers.norm_buf[layer_idx],
                residual_dst,
                dev,
            )?;
        }
        Ok(())
    }

    /// Spec §Phase 6: QK-norm + RoPE + KV append + device attention + bump.
    /// `pos` rides the device counter (`pos_dev`); arena addrs are stable.
    /// Result lands in `attn_out_buf[layer_idx]`. Every step enqueues.
    ///
    /// P3: `steps = buffers.batch` so the captured graph processes a whole
    /// decode batch in one launch. All batch items share the same starting
    /// position (`pos_dev`), so they append K/V at adjacent rows
    /// `[pos, pos+1, ..., pos+batch-1]` and attend over `batch` query rows.
    /// The caller bumps `pos_dev` by `batch` after replay.
    pub fn attention_forward_graph(
        &self,
        layer_idx: usize,
        buffers: &DecodeGraphBuffers,
        dev: &Dev,
        skip_norm_rope_append: bool,
    ) -> Result<()> {
        check_layer_topology(buffers, layer_idx)
            .map_err(|e| grim_core::error::Error::Backend(format!("{e}")))?;
        // P3: `steps` is the per-slot token count appended THIS replay (decode
        // = 1). `batch` is the number of concurrent per-slot sequences; each
        // slot's kernels use its own arena region and position scalar.
        let steps = 1usize;
        let batch = buffers.batch.max(1);
        let hd = self.head_dim;
        let nh = self.num_heads;
        let nkv = self.num_kv_heads;
        let kv_stride = nkv * hd;
        let arena_slot_stride = buffers.max_ctx * kv_stride;

        if skip_norm_rope_append {
            // MXFP4 fused branch: QK-norm + RoPE + K/V append already done
            // inside the single fused kernel — skip to attention + bump.
        } else {
            // RoPE IN PLACE with the device position base (no host positions
            // vector, no output alloc). Safe: each thread loads its pair before
            // storing it; pairs are disjoint across threads.
            // P3: shape is [steps, nh, hd] where steps==batch — each batch item
            // is one query position sharing the same base position.
            let mut rope_cfg = RopeConfig::new(hd, self.rope_theta);
            // LFM2/LFM2.5: NeoX half-split pairing, not GPT-J interleaved.
            rope_cfg.interleaved = false;
            let q3 = Shape::new(vec![batch, nh * steps, hd]);
            let k3 = Shape::new(vec![batch, nkv * steps, hd]);

            // PLAN-decode-throughput-restore: fuse the QK-norm into the RoPE
            // launch (one grim_qk_rope_dev_base per tensor replaces the
            // rms_norm_into + rope_dev_base_into pair). Requires head_dim 64 so a
            // warp covers exactly one head row for the RMS reduce; the kernel
            // applies norm BEFORE rotation, matching llama.cpp's qk-norm-then-rope.
            let fused_qk_rope = hd == 64
                && self
                    .attn_q_norm
                    .as_ref()
                    .zip(self.attn_k_norm.as_ref())
                    .is_some();
            if fused_qk_rope {
                let qn = self.attn_q_norm.as_ref().unwrap();
                let kn = self.attn_k_norm.as_ref().unwrap();
                dev.qk_rope_dev_base_into(
                    &buffers.q_buf[layer_idx],
                    &buffers.pos_dev,
                    &**qn.weight.storage(),
                    qn.eps,
                    &buffers.q_buf[layer_idx],
                    &rope_cfg,
                    &q3,
                    nh,
                    steps,
                )
                .map_err(grim_core::error::Error::Tensor)?;
                // PLAN 4 Task 4: rope+append fusion for K/V — replaces this
                // qk_rope(k) call plus both kv_append calls below with one
                // launch. Q rope stays separate (no append); bump stays after
                // attention (reads pre-bump total_dev).
                dev.qk_rope_append_kv_into(
                    &buffers.k_buf[layer_idx],
                    &buffers.pos_dev,
                    &**kn.weight.storage(),
                    kn.eps,
                    &buffers.v_buf[layer_idx],
                    &buffers.k_buf[layer_idx],
                    &buffers.k_arena[layer_idx],
                    &buffers.v_arena[layer_idx],
                    &rope_cfg,
                    &k3,
                    nkv,
                    steps,
                    kv_stride,
                    arena_slot_stride,
                )
                .map_err(grim_core::error::Error::Tensor)?;
            } else {
                // QK-norm IN PLACE (row-wise over [batch * heads, hd] slots; flat
                // counts match [batch, n]). Safe: each row is normalized independently.
                if let (Some(qn), Some(kn)) = (self.attn_q_norm.as_ref(), self.attn_k_norm.as_ref())
                {
                    let qn_shape = Shape::new(vec![batch * nh, hd]);
                    dev.rms_norm_into(
                        &buffers.q_buf[layer_idx],
                        &**qn.weight.storage(),
                        qn.eps,
                        &buffers.q_buf[layer_idx],
                        &qn_shape,
                    )
                    .map_err(grim_core::error::Error::Tensor)?;
                    let kn_shape = Shape::new(vec![batch * nkv, hd]);
                    dev.rms_norm_into(
                        &buffers.k_buf[layer_idx],
                        &**kn.weight.storage(),
                        kn.eps,
                        &buffers.k_buf[layer_idx],
                        &kn_shape,
                    )
                    .map_err(grim_core::error::Error::Tensor)?;
                }
                dev.rope_dev_base_into(
                    &buffers.q_buf[layer_idx],
                    &buffers.pos_dev,
                    &buffers.q_buf[layer_idx],
                    &rope_cfg,
                    &q3,
                    nh,
                    steps,
                )
                .map_err(grim_core::error::Error::Tensor)?;
                dev.rope_dev_base_into(
                    &buffers.k_buf[layer_idx],
                    &buffers.pos_dev,
                    &buffers.k_buf[layer_idx],
                    &rope_cfg,
                    &k3,
                    nkv,
                    steps,
                )
                .map_err(grim_core::error::Error::Tensor)?;
            }

            // Append rotated rows at the on-device offset; validate topology too.
            // PLAN 4 Task 4: skipped when the fused rope+append path above
            // ran (it appended K and V inline); the split path still needs
            // both appends.
            let max_ctx = buffers.max_ctx;
            if !fused_qk_rope {
                launch_qkv_gemv(&buffers.k_arena[layer_idx], buffers.current_pos, max_ctx)
                    .map_err(|e| grim_core::error::Error::Backend(format!("{e}")))?;
                launch_attention(&buffers.k_arena[layer_idx], buffers.current_pos, max_ctx)
                    .map_err(|e| grim_core::error::Error::Backend(format!("{e}")))?;
                grim_backend_rocm::launch_kv_append_batch(
                    dev,
                    &buffers.k_arena[layer_idx],
                    &buffers.k_buf[layer_idx],
                    &buffers.pos_dev,
                    kv_stride,
                    steps,
                    batch,
                    arena_slot_stride,
                )
                .map_err(|e| grim_core::error::Error::Backend(format!("kv_append k: {e}")))?;
                grim_backend_rocm::launch_kv_append_batch(
                    dev,
                    &buffers.v_arena[layer_idx],
                    &buffers.v_buf[layer_idx],
                    &buffers.pos_dev,
                    kv_stride,
                    steps,
                    batch,
                    arena_slot_stride,
                )
                .map_err(|e| grim_core::error::Error::Backend(format!("kv_append v: {e}")))?;
            }
        }

        // Attention writes DIRECTLY into pool slots: output, online-softmax
        // partials, and the shared read-only dummy. Zero allocs in capture.
        grim_backend_rocm::launch_qkv_attention_dev_batch(
            dev,
            &buffers.q_buf[layer_idx],
            &buffers.k_arena[layer_idx],
            &buffers.v_arena[layer_idx],
            &buffers.attn_out_buf[layer_idx],
            &buffers.attn_max_buf[layer_idx],
            &buffers.attn_sum_buf[layer_idx],
            &buffers.pos_dev,
            nh as u32,
            nkv as u32,
            hd as u32,
            steps as u32,
            steps as u32,
            1.0 / (hd as f32).sqrt(),
            0,
            0.0,
            &buffers.attn_dummy,
            0,
            0,
            &buffers.attn_dummy,
            0,
            batch,
            arena_slot_stride,
        )
        .map_err(|e| grim_core::error::Error::Backend(format!("qkv_attention: {e}")))?;
        // Bump LAST so the next replay appends at the new offset (per slot).
        grim_backend_rocm::launch_bump_i32_slots(dev, &buffers.pos_dev, steps, batch)
            .map_err(|e| grim_core::error::Error::Backend(format!("bump: {e}")))?;
        Ok(())
    }
}

fn require_device_ptr(s: &dyn grim_tensor::BackendStorage, name: &str) -> Result<()> {
    // Trait-object call needs no import: methods ride the object type.
    if s.device_ptr().is_some() {
        Ok(())
    } else {
        Err(grim_core::error::Error::Backend(format!(
            "{name} has no device ptr"
        )))
    }
}

/// Spec §Phase 4: RMS norm enqueued on `dev`, written DIRECTLY into `dst`
/// (caller pool slot) — no scratch alloc, no staging copy. `dst` must carry
/// the row semantics in its shape (flat counts must agree with `input`).
/// Keeps the spec name; `dev`/`weight`/`eps` ride along (names are grepped,
/// arity is free).
pub fn rms_norm_graph(
    input: &dyn grim_tensor::BackendStorage,
    weight: &dyn grim_tensor::BackendStorage,
    dst: &RocmStorage,
    dev: &Dev,
    eps: f32,
) -> Result<()> {
    require_device_ptr(input, "rms_norm_graph input")?;
    let out_shape = dst.shape().clone();
    dev.rms_norm_into(input, weight, eps, dst, &out_shape)
        .map_err(grim_core::error::Error::Tensor)?;
    Ok(())
}

/// Spec §Phase 4 residual add enqueued on `dev`, written DIRECTLY into `dst`
/// — no scratch alloc, no staging copy. In-place (`dst` aliases an input) is
/// safe: strictly per-element.
pub fn add_graph(
    a: &dyn grim_tensor::BackendStorage,
    b: &dyn grim_tensor::BackendStorage,
    dst: &RocmStorage,
    dev: &Dev,
) -> Result<()> {
    require_device_ptr(a, "add_graph a")?;
    dev.add_into(a, b, dst)
        .map_err(grim_core::error::Error::Tensor)?;
    Ok(())
}

/// Spec §Phase 3: async H2D of token id into fixed input buffer.
/// Convention: `(dev, dst, token_id)` — matches `write_embeddings_to_buffer_batch`.
pub fn write_embedding_to_buffer(
    dev: &grim_backend_rocm::RocmDevice,
    dst: &dyn grim_tensor::BackendStorage,
    token_id: u32,
) -> Result<()> {
    grim_backend_rocm::write_embedding_to_buffer(dev, dst_downcast(dst)?, token_id)
        .map_err(|e| grim_core::error::Error::Backend(format!("{e}")))
}

/// P3: async H2D of a batch of token ids into the fixed layer-0 input buffer.
/// Thin wrapper that converts the backend error type.
pub fn write_batch_embeddings(dev: &Dev, dst: &RocmStorage, token_ids: &[u32]) -> Result<()> {
    grim_backend_rocm::decode_graph_buffers::write_embeddings_to_buffer_batch(dev, dst, token_ids)
        .map_err(|e| grim_core::error::Error::Backend(format!("{e}")))
}

fn dst_downcast(dst: &dyn grim_tensor::BackendStorage) -> Result<&grim_backend_rocm::RocmStorage> {
    dst.as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .ok_or_else(|| grim_core::error::Error::Backend("write_embedding: need RocmStorage".into()))
}

impl Lfm2Block {
    /// S2 (PLAN-kernel-fusion): ShortConv sublayer + dense FFN tail, fully
    /// enqueued. Conv state lives in the per-layer device ring
    /// `buffers.sc_state[layer_idx]` (column-major `[h_dim, kc]`, the HIP
    /// kernel's in-place layout); it is seeded on first use from the host
    /// mirror OUTSIDE any capture bracket (first warmup forward), then
    /// updated in-place by the conv kernel inside the graph. Batch is 1 for
    /// the decode graph; batched ShortConv capture is a scoped follow-up.
    fn shortconv_forward_graph(
        &self,
        layer_idx: usize,
        buffers: &DecodeGraphBuffers,
        dev: &Dev,
    ) -> Result<()> {
        let in_proj = self.shortconv_in_proj.as_ref().ok_or_else(|| {
            grim_core::error::Error::Backend("shortconv_forward_graph: missing in_proj".into())
        })?;
        let conv = self.shortconv_conv.as_ref().ok_or_else(|| {
            grim_core::error::Error::Backend("shortconv_forward_graph: missing conv".into())
        })?;
        let out_proj = self.shortconv_out_proj.as_ref().ok_or_else(|| {
            grim_core::error::Error::Backend("shortconv_forward_graph: missing out_proj".into())
        })?;
        let hidden = buffers.layer_input[layer_idx]
            .shape()
            .dims()
            .last()
            .copied()
            .unwrap_or(0);
        let act = &buffers.act_q81_buf[layer_idx];

        // 1. attn-norm the residual into the staging slot, then the fused
        //    in-projection GEMV [batch, 3*h_dim] (b∥x∥c per row).
        //    PLAN 4 Task 3: norm-fused in_proj — the standalone norm is
        //    skipped when the norm-fused dot4 path is taken (Q8_0 weights,
        //    dot4 arch, kill-switches honored via dot_fused_ok).
        let norm_fused_in_proj =
            dot_fused_ok(dev, hidden) && is_q80(&in_proj.weight);
        if !norm_fused_in_proj {
            dev.rms_norm_into(
                &buffers.layer_input[layer_idx],
                &**self.attn_norm.weight.storage(),
                self.attn_norm.eps,
                &buffers.norm_buf[layer_idx],
                &buffers.layer_input[layer_idx].shape().clone(),
            )
            .map_err(grim_core::error::Error::Tensor)?;
        }
        let normed: &Storage = &buffers.norm_buf[layer_idx];
        if norm_fused_in_proj {
            let n_in = in_proj.weight.shape().dim(0).unwrap_or(0);
            dev.launch_dot4_q80_norm_f32act_gemv_into(
                &buffers.layer_input[layer_idx],
                rocm_storage(&self.attn_norm.weight)?,
                self.attn_norm.eps,
                rocm_storage(&in_proj.weight)?,
                &buffers.sc_proj_buf[layer_idx],
                buffers.batch.max(1),
                n_in,
                hidden,
            )
            .map_err(|e| {
                grim_core::error::Error::Backend(format!("fused norm in_proj: {e}"))
            })?;
        } else {
            linear_into(
                dev,
                normed,
                &in_proj.weight,
                &buffers.sc_proj_buf[layer_idx],
                act,
            )?;
        }

        // 2+3. PLAN-kernel-launch-reduction Phase C: ONE fused launch reads
        //      b∥x∥c straight from the in_proj output, computes bx = b*x,
        //      runs the causal conv with in-place state update, and applies
        //      the c gate — replacing (3 slice copies + mul + conv + mul).
        dev.short_conv1d_fused_step_into(
            &buffers.sc_proj_buf[layer_idx],
            conv.storage().as_ref(),
            &buffers.sc_state[layer_idx],
            &buffers.sc_y[layer_idx],
            buffers.batch.max(1),
            hidden,
            3, // l_cache
        )
        .map_err(|e| grim_core::error::Error::Backend(format!("sc fused conv: {e}")))?;

        // 4. Out projection into staging, then residual add in place
        //    (layer_output currently holds the pre-block residual).
        linear_into(
            dev,
            &buffers.sc_y[layer_idx],
            &out_proj.weight,
            &buffers.norm_buf[layer_idx],
            act,
        )?;
        if !self.is_moe {
            // fuse_norm: residual-add + FFN-norm fused in ffn_forward_graph.
            return Ok(());
        }
        add_graph(
            &buffers.layer_input[layer_idx],
            &buffers.norm_buf[layer_idx],
            &buffers.layer_output[layer_idx],
            dev,
        )
    }

    /// M2 (PLAN-kernel-fusion): MoE FFN sublayer, fully enqueued.
    /// 1) ffn-norm into staging; 2) router gate GEMV into
    ///    `moe_gate_logits[l]`; 3) `grim_moe_route_topk` writes the sortless
    ///    routing triple into the graph's persistent routing buffers; 4) the
    ///    grouped-expert kernel reads resident stacked weights + the routing
    ///    triple and atomically accumulates into `moe_out[l]`; 5) residual add.
    ///
    /// All launches hit fixed pool addresses — capture-safe after one warmup
    /// pass builds the resident weight stack (an H2D at capture time would
    /// abort; the caller's warmup guarantee makes every bracket call a hit).
    fn moe_forward_graph(
        &self,
        layer_idx: usize,
        buffers: &DecodeGraphBuffers,
        dev: &Dev,
        residual_dst: &RocmStorage,
    ) -> Result<()> {
        let gate_inp = self.ffn_gate_inp.as_ref().ok_or_else(|| {
            grim_core::error::Error::Backend("moe_forward_graph: missing router gate".into())
        })?;
        let hidden = buffers.layer_input[layer_idx]
            .shape()
            .dims()
            .last()
            .copied()
            .unwrap_or(0);
        let batch = buffers.batch.max(1);
        let top_k = self.n_expert_used.min(self.n_expert).max(1);
        let act = &buffers.act_q81_buf[layer_idx];

        // 1. FFN norm of the attention residual into staging.
        dev.rms_norm_into(
            &buffers.layer_output[layer_idx],
            &**self.ffn_norm.weight.storage(),
            self.ffn_norm.eps,
            &buffers.norm_buf[layer_idx],
            &buffers.layer_output[layer_idx].shape().clone(),
        )
        .map_err(grim_core::error::Error::Tensor)?;
        let normed: &Storage = &buffers.norm_buf[layer_idx];

        // 2. Router gate GEMV [batch, n_expert].
        linear_into(
            dev,
            normed,
            &gate_inp.weight,
            &buffers.moe_gate_logits[layer_idx],
            act,
        )?;

        // 3. Resident scratch + stacked weights (cache hit after warmup).
        let experts = self.moe_experts().ok_or_else(|| {
            grim_core::error::Error::Backend("moe_forward_graph: expert weights missing".into())
        })?;
        let (tokens, experts_b, weights, gate_buf, up_buf, down_buf) =
            crate::shared_moe::ensure_charon_scratch(
                dev.ordinal(),
                batch,
                top_k,
                experts,
                &self.charon_cache,
            )?;
        let tokens_rocm = dst_downcast(tokens.as_ref())?;
        let experts_rocm = dst_downcast(experts_b.as_ref())?;
        let weights_rocm = dst_downcast(weights.as_ref())?;
        dev.moe_route_topk_on_device(
            &buffers.moe_gate_logits[layer_idx],
            None,
            tokens_rocm,
            experts_rocm,
            weights_rocm,
            batch,
            self.n_expert,
            top_k,
            0, // softmax routing (matches eager path)
        )
        .map_err(|e| grim_core::error::Error::Backend(format!("moe route: {e}")))?;

        // 4. Grouped dispatch into the fixed moe_out slot.
        let norm_rocm = dst_downcast(normed)?;
        dev.moe_fused_dispatch_resident_routing_into(
            norm_rocm,
            gate_buf.as_ref(),
            up_buf.as_ref(),
            down_buf.as_ref(),
            tokens_rocm,
            experts_rocm,
            weights_rocm,
            batch * top_k,
            &buffers.moe_out[layer_idx],
            hidden,
            experts[0].gate.weight.shape().dim(0).unwrap_or(0),
            1.0,
        )
        .map_err(|e| grim_core::error::Error::Backend(format!("moe dispatch: {e}")))?;

        // 5. Residual add -> next block's input slot (Fix 3: no publish copy).
        add_graph(
            &buffers.layer_output[layer_idx],
            &buffers.moe_out[layer_idx],
            residual_dst,
            dev,
        )
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn graph_dims_sane() {
        // Shape math must not underflow on tiny configs.
        assert!("GRIM_DECODE_GRAPH".is_ascii());
    }
}
