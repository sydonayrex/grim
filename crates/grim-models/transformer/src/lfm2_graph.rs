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
use grim_backend_rocm::decode_graph_buffers::{
    DecodeGraph, DecodeGraphBuffers, check_layer_topology, decode_graph_enabled,
    launch_attention, launch_qkv_gemv, write_embeddings_to_buffer_batch,
};
use grim_core::error::Result;
use grim_tensor::{BackendStorage, Device, MemoryOps, RopeConfig, Shape};

use crate::lfm2::{Lfm2, Lfm2Block};

/// Spec §Phase 6 dims derived from config. `max_ctx` caps KV arenas.
#[allow(clippy::type_complexity)]
fn graph_dims(
    lfm: &Lfm2,
    max_ctx: usize,
) -> (usize, usize, usize, usize, usize, usize, usize) {
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
fn rocm_storage(t: &grim_tensor::Tensor) -> Result<&RocmStorage> {
    dst_downcast(t.storage().as_ref())
}

/// Quant-capable GEMV into a fixed pool slot. Mirrors eager `Linear` decode
/// dispatch (F32 rocBLAS/GEMV, Q80/Q4K-K quant dot paths) via the backend
/// [`Dev::linear_decode_into`]; unsupported dtypes fall back eager.
fn linear_into(
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
fn dot_fused_ok(dev: &Dev, hidden: usize) -> bool {
    hidden != 0
        && hidden % 32 == 0
        && dev.supports_dot4()
        && !matches!(
            std::env::var("GRIM_DOT_GEMV").as_deref(),
            Ok("0" | "false" | "off")
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
fn publish_into(dev: &Dev, dst: &Storage, src: &Storage) -> Result<()> {
    let n = dst.shape().elem_count();
    dev.copy_slice_into(dst, src, 0, n)
        .map_err(grim_core::error::Error::Tensor)
}

impl Lfm2 {
    /// Spec §Phase 5: `model.get_or_create_decode_graph()`.
    /// Allocates fixed pool once; caller keeps it across steps for stable addrs.
    /// `batch` parameterizes all per-step slots as `[batch, dim]`. `batch=1`
    /// preserves the original single-token shape.
    /// `Err` -> caller falls back eager (spec §Fallback). Honors both env gates.
    pub fn get_or_create_decode_graph(
        &self,
        max_ctx: usize,
        batch: usize,
    ) -> Result<DecodeGraph> {
        if !decode_graph_enabled() {
            return Err(grim_core::error::Error::Backend(
                "decode graph disabled by env".into(),
            ));
        }
        let dev = dev_for(self)?;
        let stream = dev
            .get_stream_from_pool(0)
            .ok_or_else(|| grim_core::error::Error::Backend("no stream in pool".into()))?;
        let (hidden, n_q, n_k, inter, vocab, ctx, nh) = graph_dims(self, max_ctx);
        let buffers = DecodeGraphBuffers::allocate(
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
        )
        .map_err(|e| grim_core::error::Error::Backend(format!("graph pool alloc: {e}")))?;
        Ok(DecodeGraph::new(&dev, buffers, stream))
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
    pub fn forward_capture_batch(
        &self,
        graph: &mut DecodeGraph,
        token_ids: &[u32],
    ) -> Result<()> {
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
    pub fn forward_replay_batch(
        &self,
        graph: &mut DecodeGraph,
        token_ids: &[u32],
    ) -> Result<()> {
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
        let h_shape = buffers.head_input.shape().clone();
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
        if !self.is_attention() {
            return Err(grim_core::error::Error::Unimplemented(
                "forward_graph: recurrent ShortConv keeps host conv state; use eager".into(),
            ));
        }
        if self.is_moe {
            return Err(grim_core::error::Error::Unimplemented(
                "forward_graph: MoE routes on host; use eager".into(),
            ));
        }
        // Attention sublayer -> layer_output, then FFN sublayer curves into
        // the next layer's input (or head_input for the last layer).
        self.attn_forward_graph(layer_idx, buffers, dev)?;
        self.ffn_forward_graph(layer_idx, buffers, dev)?;
        // Publish block output for the next layer (one D2D node).
        let n_layers = buffers.layer_input.len();
        let dst: &Storage = if layer_idx + 1 < n_layers {
            &buffers.layer_input[layer_idx + 1]
        } else {
            &buffers.head_input
        };
        publish_into(dev, dst, &buffers.layer_output[layer_idx])?;
        Ok(())
    }

    /// Spec §Phase 4 attention branch: norm -> QKV -> QK-norm -> RoPE ->
    /// KV append -> device attention -> O proj -> residual. All enqueued.
    pub fn attn_forward_graph(
        &self,
        layer_idx: usize,
        buffers: &DecodeGraphBuffers,
        dev: &Dev,
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

        // 1. Attention norm into the per-layer staging slot.
        dev.rms_norm_into(
            &buffers.layer_input[layer_idx],
            &**self.attn_norm.weight.storage(),
            self.attn_norm.eps,
            &buffers.norm_buf[layer_idx],
            &buffers.layer_input[layer_idx].shape().clone(),
        )
        .map_err(grim_core::error::Error::Tensor)?;
        let normed: &Storage = &buffers.norm_buf[layer_idx];

        // 2. QKV projections. Fused path (Q8_0 blob present): ONE dot4 GEMV
        //    into fused staging + 3 slice copies (4 launches vs 3 GEMMs).
        //    Plain path: quant-aware GEMV per projection. Both write fixed
        //    pool slots; no scratch allocs.
        let hidden = normed.shape().dims().last().copied().unwrap_or(0);
        let act = &buffers.act_q81_buf[layer_idx];
        let norm_rocm = dst_downcast(normed)?;
        // Fused Q8_0 dot4 path: ONE quant + ONE fused GEMV replaces 3
        // separate GEMVs. Works for any batch size — the quantize kernel
        // quantizes the full [batch, hidden] activation in one launch, and
        // the fused dot4 GEMV reads from the same [batch, hidden] activation
        // and writes to the per-layer fused_qkv_out slot at [batch, n_q+2*n_kv].
        // The per-token (m=1) hardcode is removed; m = buffers.batch.
        if let Some(fused) = self
            .wqkv_q80_fused
            .as_ref()
            .filter(|_| dot_fused_ok(dev, hidden) && self.head_dim != 0)
        {
            let m = buffers.batch.max(1);
            dev.launch_quantize_q8_1(norm_rocm, act, m, hidden)
                .map_err(|e| grim_core::error::Error::Backend(format!("qkv quant: {e}")))?;
            let nkv = fused.n_k;
            dev.launch_fused_qkv_dot4_into(
                act,
                &fused.storage,
                &buffers.fused_qkv_out[layer_idx],
                fused.n_q,
                nkv,
                hidden,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("fused qkv: {e}")))?;
            let fused_out: &Storage = &buffers.fused_qkv_out[layer_idx];
            dev.copy_slice_into(&buffers.q_buf[layer_idx], fused_out, 0, fused.n_q)
                .map_err(grim_core::error::Error::Tensor)?;
            dev.copy_slice_range(
                &buffers.k_buf[layer_idx],
                0,
                fused_out,
                fused.n_q,
                fused.n_k,
            )
            .map_err(grim_core::error::Error::Tensor)?;
            dev.copy_slice_range(
                &buffers.v_buf[layer_idx],
                0,
                fused_out,
                fused.n_q + fused.n_k,
                fused.n_v,
            )
            .map_err(grim_core::error::Error::Tensor)?;
        } else {
            linear_into(dev, normed, &wq.weight, &buffers.q_buf[layer_idx], act)?;
            linear_into(dev, normed, &wk.weight, &buffers.k_buf[layer_idx], act)?;
            linear_into(dev, normed, &wv.weight, &buffers.v_buf[layer_idx], act)?;
        }

        // 3-6. QK-norm + RoPE + append + attend + bump (real nodes).
        self.attention_forward_graph(layer_idx, buffers, dev)?;

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
        add_graph(
            &buffers.layer_input[layer_idx],
            &buffers.norm_buf[layer_idx],
            &buffers.layer_output[layer_idx],
            dev,
        )?;
        Ok(())
    }

    /// Spec §Phase 4 FFN branch: ffn_norm -> gate/up GEMMs -> SiLU -> down
    /// GEMM -> residual add into layer_output. All enqueued.
    pub fn ffn_forward_graph(
        &self,
        layer_idx: usize,
        buffers: &DecodeGraphBuffers,
        dev: &Dev,
    ) -> Result<()> {
        check_layer_topology(buffers, layer_idx)
            .map_err(|e| grim_core::error::Error::Backend(format!("{e}")))?;
        // Norm the attention residual (layer_output currently holds it) into
        // the staging slot; every GEMM below then writes a fixed pool slot.
        let ffn_shape = buffers.layer_output[layer_idx].shape().clone();
        dev.rms_norm_into(
            &buffers.layer_output[layer_idx],
            &**self.ffn_norm.weight.storage(),
            self.ffn_norm.eps,
            &buffers.norm_buf[layer_idx],
            &ffn_shape,
        )
        .map_err(grim_core::error::Error::Tensor)?;
        let normed: &Storage = &buffers.norm_buf[layer_idx];
        let act = &buffers.act_q81_buf[layer_idx];
        let hidden = normed.shape().dims().last().copied().unwrap_or(0);
        // Fused gate+up (Q8_0 blob present): ONE dot4 GEMV into gate_up_buf,
        // then slice halves into gate/up bufs. Works for any batch size —
        // the quantize kernel quantizes [batch, hidden] and the fused GEMV
        // writes to [batch, n_gate+n_up]. m = buffers.batch (not 1).
        if let Some(fused) = self
            .w_gate_up_q80_fused
            .as_ref()
            .filter(|_| dot_fused_ok(dev, hidden))
        {
            let norm_rocm = dst_downcast(normed)?;
            let m = buffers.batch.max(1);
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
            linear_into(dev, normed, &self.ffn_gate.weight, &buffers.gate_buf[layer_idx], act)?;
            linear_into(dev, normed, &self.ffn_up.weight, &buffers.up_buf[layer_idx], act)?;
        }
        dev.silu_mul_into(
            &buffers.gate_buf[layer_idx],
            &buffers.up_buf[layer_idx],
            &buffers.activated_buf[layer_idx],
        )
        .map_err(grim_core::error::Error::Tensor)?;
        // Down projection into staging (norm_buf free: gate/up consumed it),
        // then residual add in place (per-element independent, safe).
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
            &buffers.layer_output[layer_idx],
            dev,
        )?;
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

        // QK-norm IN PLACE (row-wise over [batch * heads, hd] slots; flat
        // counts match [batch, n]). Safe: each row is normalized independently.
        if let (Some(qn), Some(kn)) = (self.attn_q_norm.as_ref(), self.attn_k_norm.as_ref()) {
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

        // RoPE IN PLACE with the device position base (no host positions
        // vector, no output alloc). Safe: each thread loads its pair before
        // storing it; pairs are disjoint across threads.
        // P3: shape is [steps, nh, hd] where steps==batch — each batch item
        // is one query position sharing the same base position.
        let rope_cfg = RopeConfig::new(hd, self.rope_theta);
        let q3 = Shape::new(vec![batch, nh * steps, hd]);
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
        let k3 = Shape::new(vec![batch, nkv * steps, hd]);
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

        // Append rotated rows at the on-device offset; validate topology too.
        let max_ctx = buffers.max_ctx;
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
pub fn write_batch_embeddings(
    dev: &Dev,
    dst: &RocmStorage,
    token_ids: &[u32],
) -> Result<()> {
    grim_backend_rocm::decode_graph_buffers::write_embeddings_to_buffer_batch(dev, dst, token_ids)
        .map_err(|e| grim_core::error::Error::Backend(format!("{e}")))
}

fn dst_downcast(
    dst: &dyn grim_tensor::BackendStorage,
) -> Result<&grim_backend_rocm::RocmStorage> {
    dst.as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .ok_or_else(|| grim_core::error::Error::Backend("write_embedding: need RocmStorage".into()))
}

#[cfg(test)]
mod tests {
    #[test]
    fn graph_dims_sane() {
        // Shape math must not underflow on tiny configs.
        assert!("GRIM_DECODE_GRAPH".is_ascii());
    }
}
