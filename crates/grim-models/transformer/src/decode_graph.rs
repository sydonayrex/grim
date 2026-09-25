//! Generic Decode Graph Capture & Replay trait for ROCm CausalLM models.
//!
//! Provides `DecodeGraphModel` trait, standardizing decode-step HIP graph recording,
//! KV seeding, and replay across architectures (LFM2, Llama, and derivatives).

use std::sync::Arc;

use grim_backend_rocm::as_rocm;
use grim_backend_rocm::decode_graph_buffers::{
    ConvRingSeed, DecodeGraph, DecodeGraphBuffers, EagerKvSource, check_layer_topology,
    decode_graph_enabled, launch_attention, launch_qkv_gemv,
};
use grim_core::error::Result;
use grim_tensor::{BackendStorage, Device, MemoryOps, RopeConfig, Shape};

use crate::block::LlamaBlock;
use crate::chameleon::{Chameleon, ChameleonBlock};
use crate::deepseek2::DeepSeek2;
use crate::deepseek4::DeepSeek4;
use crate::deepseek32::DeepSeek32;
use crate::gemma2::{Gemma2, Gemma2Block};
use crate::glm4_moe_lite::Glm4MoeLite;
use crate::granite_moe_hybrid::GraniteMoeHybrid;
use crate::hyv3::HyV3;
use crate::minimax_m3::MiniMaxM3;
use crate::model::Llama;
use crate::qwen35::{Qwen35, Qwen35Block, Qwen35LayerCache};

type Dev = grim_backend_rocm::RocmDevice;
type Storage = dyn grim_tensor::BackendStorage;

/// Common trait for models that support single-step or batched decode HIP graph capture and replay.
pub trait DecodeGraphModel: Send + Sync {
    /// Allocate or retrieve a fixed decode buffer pool for this model.
    fn get_or_create_decode_graph(&self, max_ctx: usize, batch: usize) -> Result<DecodeGraph>;

    /// Record a decode step (single token) into the provided DecodeGraph bracket.
    fn forward_capture(&self, graph: &mut DecodeGraph, token_id: u32) -> Result<()>;

    /// Replay an instantiated decode graph with the given token ID.
    fn forward_replay(&self, graph: &mut DecodeGraph, token_id: u32) -> Result<()>;

    /// Export eager device KV arenas for prefill KV seeding into the graph pool.
    fn eager_kv_seed_sources<'a>(
        &self,
        session: &'a dyn grim_core::session::SessionT,
        valid_rows: u32,
    ) -> Result<Vec<Option<EagerKvSource<'a>>>>;

    /// Host conv-ring snapshots for recurrent-layer seeding (default: no
    /// conv layers). Indexed by layer; `None` for attention layers. Borrowed
    /// from the session state — must outlive the seed H2D copy.
    fn eager_conv_seed_rings<'a>(
        &self,
        _session: &'a dyn grim_core::session::SessionT,
    ) -> Result<Vec<Option<ConvRingSeed<'a>>>> {
        Ok(Vec::new())
    }
}

// ─── Helpers for Graph Recording ──────────────────────────────────────────

fn dev_for_llama(llama: &Llama) -> Result<Arc<Dev>> {
    match &llama.device {
        Device::Rocm(o) => Ok(Dev::shared(*o)),
        _ => Err(grim_core::error::Error::Unimplemented(
            "decode graph needs ROCm device".into(),
        )),
    }
}

fn rocm_storage(t: &grim_tensor::Tensor) -> Result<&grim_backend_rocm::RocmStorage> {
    dst_downcast(t.storage().as_ref())
}

fn dst_downcast(dst: &dyn grim_tensor::BackendStorage) -> Result<&grim_backend_rocm::RocmStorage> {
    dst.as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .ok_or_else(|| grim_core::error::Error::Backend("need RocmStorage".into()))
}

fn linear_into(
    dev: &Dev,
    a: &Storage,
    w: &grim_tensor::Tensor,
    out: &grim_backend_rocm::RocmStorage,
    act_q81: &grim_backend_rocm::RocmStorage,
) -> Result<()> {
    let ws = rocm_storage(w)?;
    let _ = dev
        .linear_decode_into(a, ws, out, act_q81)
        .map_err(|e| grim_core::error::Error::Backend(format!("linear_decode: {e}")))?;
    Ok(())
}

fn dot_fused_ok(dev: &Dev, hidden: usize) -> bool {
    hidden != 0
        && hidden % 32 == 0
        && dev.supports_dot4()
        && !matches!(
            std::env::var("GRIM_DOT_GEMV").as_deref(),
            Ok("0" | "false" | "off")
        )
}

fn publish_into(dev: &Dev, dst: &Storage, src: &Storage) -> Result<()> {
    let n = dst.shape().elem_count();
    dev.copy_slice_into(dst, src, 0, n)
        .map_err(grim_core::error::Error::Tensor)
}

fn add_graph(
    a: &dyn grim_tensor::BackendStorage,
    b: &dyn grim_tensor::BackendStorage,
    dst: &grim_backend_rocm::RocmStorage,
    dev: &Dev,
) -> Result<()> {
    dev.add_into(a, b, dst)
        .map_err(grim_core::error::Error::Tensor)?;
    Ok(())
}

fn axpy_graph(
    a: &dyn grim_tensor::BackendStorage,
    s: f32,
    b: &dyn grim_tensor::BackendStorage,
    dst: &grim_backend_rocm::RocmStorage,
    dev: &Dev,
) -> Result<()> {
    dev.axpy_into(a, s, b, dst)
        .map_err(grim_core::error::Error::Tensor)?;
    Ok(())
}

// ─── Llama Block Graph Forward ────────────────────────────────────────────

impl LlamaBlock {
    /// Enqueue all kernels for this Llama block into the decode graph bracket.
    pub fn forward_graph(
        &self,
        layer_idx: usize,
        buffers: &DecodeGraphBuffers,
        dev: &Dev,
    ) -> Result<()> {
        check_layer_topology(buffers, layer_idx)
            .map_err(|e| grim_core::error::Error::Backend(format!("{e}")))?;

        // 1. Attention RMS norm into norm_buf
        dev.rms_norm_into(
            &buffers.layer_input[layer_idx],
            &**self.attn_norm.weight.storage(),
            self.attn_norm.eps,
            &buffers.norm_buf[layer_idx],
            &buffers.layer_input[layer_idx].shape().clone(),
        )
        .map_err(grim_core::error::Error::Tensor)?;
        let normed: &Storage = &buffers.norm_buf[layer_idx];
        let act = &buffers.act_q81_buf[layer_idx];
        let hidden = normed.shape().dims().last().copied().unwrap_or(0);
        let batch = buffers.batch.max(1);

        // 2. QKV projection
        let fused_qkv = self
            .wqkv_q80_fused
            .as_ref()
            .filter(|_| dot_fused_ok(dev, hidden));

        if let Some(fused) = fused_qkv {
            let norm_rocm = dst_downcast(normed)?;
            let m = buffers.batch.max(1);
            dev.launch_quantize_q8_1(norm_rocm, act, m, hidden)
                .map_err(|e| grim_core::error::Error::Backend(format!("qkv quant: {e}")))?;
            dev.launch_fused_qkv_dot4_into(
                act,
                &fused.storage,
                &buffers.fused_qkv_out[layer_idx],
                fused.n_q,
                fused.n_k,
                hidden,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("fused qkv: {e}")))?;
            let staged: &Storage = &buffers.fused_qkv_out[layer_idx];
            dev.copy_slice_into(&buffers.q_buf[layer_idx], staged, 0, fused.n_q)
                .map_err(grim_core::error::Error::Tensor)?;
            dev.copy_slice_range(&buffers.k_buf[layer_idx], 0, staged, fused.n_q, fused.n_k)
                .map_err(grim_core::error::Error::Tensor)?;
            dev.copy_slice_range(
                &buffers.v_buf[layer_idx],
                0,
                staged,
                fused.n_q + fused.n_k,
                fused.n_v,
            )
            .map_err(grim_core::error::Error::Tensor)?;
        } else {
            linear_into(
                dev,
                normed,
                self.wq.weight(),
                &buffers.q_buf[layer_idx],
                act,
            )?;
            linear_into(
                dev,
                normed,
                self.wk.weight(),
                &buffers.k_buf[layer_idx],
                act,
            )?;
            linear_into(
                dev,
                normed,
                self.wv.weight(),
                &buffers.v_buf[layer_idx],
                act,
            )?;
        }

        // 3. Optional QK-norm
        let hd = self._cfg.head_dim;
        let nh = self._cfg.local_num_heads;
        let nkv = self._cfg.local_num_kv_heads;

        if let Some(qn) = &self.q_norm {
            let qn_shape = Shape::new(vec![batch * nh, hd]);
            dev.rms_norm_into(
                &buffers.q_buf[layer_idx],
                &**qn.weight.storage(),
                qn.eps,
                &buffers.q_buf[layer_idx],
                &qn_shape,
            )
            .map_err(grim_core::error::Error::Tensor)?;
        }
        if let Some(kn) = &self.k_norm {
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

        // 4. RoPE in-place using device position base (`pos_dev`)
        let steps = 1usize;
        let rope_cfg = RopeConfig::new(hd, self.rope.config.base);
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

        // 5. KV append to arenas
        let kv_stride = nkv * hd;
        let arena_slot_stride = buffers.max_ctx * kv_stride;
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

        // 6. Attention kernel
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

        // Bump pos_dev
        grim_backend_rocm::launch_bump_i32_slots(dev, &buffers.pos_dev, steps, batch)
            .map_err(|e| grim_core::error::Error::Backend(format!("bump: {e}")))?;

        // 7. Output projection (wo) + residual add
        linear_into(
            dev,
            &buffers.attn_out_buf[layer_idx],
            self.wo.weight(),
            &buffers.norm_buf[layer_idx],
            act,
        )?;

        add_graph(
            &buffers.layer_input[layer_idx],
            &buffers.norm_buf[layer_idx],
            &buffers.layer_output[layer_idx],
            dev,
        )?;

        // 8. FFN sublayer
        if !self.ffn_disabled {
            let ffn_shape = buffers.layer_output[layer_idx].shape().clone();
            dev.rms_norm_into(
                &buffers.layer_output[layer_idx],
                &**self.ffn_norm.weight.storage(),
                self.ffn_norm.eps,
                &buffers.norm_buf[layer_idx],
                &ffn_shape,
            )
            .map_err(grim_core::error::Error::Tensor)?;

            let normed_ffn: &Storage = &buffers.norm_buf[layer_idx];

            if let Some(fused_gu) = self
                .w_gate_up_q80_fused
                .as_ref()
                .filter(|_| dot_fused_ok(dev, hidden))
            {
                let norm_rocm = dst_downcast(normed_ffn)?;
                dev.launch_quantize_q8_1(norm_rocm, act, batch, hidden)
                    .map_err(|e| grim_core::error::Error::Backend(format!("gateup quant: {e}")))?;
                dev.launch_fused_gate_up_dot4_into(
                    act,
                    &fused_gu.storage,
                    &buffers.gate_up_buf[layer_idx],
                    fused_gu.n_gate,
                    fused_gu.n_up,
                    hidden,
                )
                .map_err(|e| grim_core::error::Error::Backend(format!("fused gateup: {e}")))?;
                let staged: &Storage = &buffers.gate_up_buf[layer_idx];
                dev.copy_slice_into(&buffers.gate_buf[layer_idx], staged, 0, fused_gu.n_gate)
                    .map_err(grim_core::error::Error::Tensor)?;
                dev.copy_slice_range(
                    &buffers.up_buf[layer_idx],
                    0,
                    staged,
                    fused_gu.n_gate,
                    fused_gu.n_up,
                )
                .map_err(grim_core::error::Error::Tensor)?;
            } else {
                let wg = self
                    .w_gate
                    .as_ref()
                    .ok_or_else(|| grim_core::error::Error::Backend("missing w_gate".into()))?;
                let wu = self
                    .w_up
                    .as_ref()
                    .ok_or_else(|| grim_core::error::Error::Backend("missing w_up".into()))?;
                linear_into(
                    dev,
                    normed_ffn,
                    wg.weight(),
                    &buffers.gate_buf[layer_idx],
                    act,
                )?;
                linear_into(
                    dev,
                    normed_ffn,
                    wu.weight(),
                    &buffers.up_buf[layer_idx],
                    act,
                )?;
            }

            dev.silu_mul_into(
                &buffers.gate_buf[layer_idx],
                &buffers.up_buf[layer_idx],
                &buffers.activated_buf[layer_idx],
            )
            .map_err(grim_core::error::Error::Tensor)?;

            let wd = self
                .w_down
                .as_ref()
                .ok_or_else(|| grim_core::error::Error::Backend("missing w_down".into()))?;
            linear_into(
                dev,
                &buffers.activated_buf[layer_idx],
                wd.weight(),
                &buffers.norm_buf[layer_idx],
                act,
            )?;

            add_graph(
                &buffers.layer_output[layer_idx],
                &buffers.norm_buf[layer_idx],
                &buffers.layer_output[layer_idx],
                dev,
            )?;
        }

        // Publish to next layer or head_input
        let n_layers = buffers.layer_input.len();
        let dst: &Storage = if layer_idx + 1 < n_layers {
            &buffers.layer_input[layer_idx + 1]
        } else {
            &buffers.head_input
        };
        publish_into(dev, dst, &buffers.layer_output[layer_idx])?;
        Ok(())
    }
}

// ─── DecodeGraphModel implementation for Llama ─────────────────────────────

impl DecodeGraphModel for Llama {
    fn get_or_create_decode_graph(&self, max_ctx: usize, batch: usize) -> Result<DecodeGraph> {
        if !decode_graph_enabled() {
            return Err(grim_core::error::Error::Backend(
                "decode graph disabled by env".into(),
            ));
        }
        let dev = dev_for_llama(self)?;
        let stream = dev
            .get_stream_from_pool(0)
            .ok_or_else(|| grim_core::error::Error::Backend("no stream in pool".into()))?;

        let hidden = self.cfg.hidden_size;
        let n_q = self.cfg.num_heads * self.cfg.head_dim;
        let n_k = self.cfg.num_kv_heads * self.cfg.head_dim;
        let n_v = n_k;
        let inter = self.cfg.intermediate_size;
        let vocab = self.cfg.vocab_size.max(1);
        let ctx = max_ctx.max(1);
        let nh = self.cfg.num_heads;

        let (n_expert, top_k) = self
            .moe_blocks
            .iter()
            .find_map(|mb| {
                mb.as_ref()
                    .map(|m| (m.moe.router.num_experts, m.moe.router.top_k))
            })
            .unwrap_or((0, 0));

        let buffers = DecodeGraphBuffers::allocate(
            &dev,
            self.layers.len(),
            hidden,
            n_q,
            n_k,
            n_v,
            inter,
            ctx,
            vocab,
            nh,
            batch,
            n_expert,
            top_k,
            0,
            0,
        )
        .map_err(|e| grim_core::error::Error::Backend(format!("graph pool alloc: {e}")))?;

        Ok(DecodeGraph::new(&dev, buffers, stream))
    }

    fn forward_capture(&self, graph: &mut DecodeGraph, token_id: u32) -> Result<()> {
        if !decode_graph_enabled() {
            return Err(grim_core::error::Error::Backend(
                "decode graph disabled".into(),
            ));
        }
        let dev = dev_for_llama(self)?;
        if !graph.capturing {
            crate::lfm2_graph::write_embedding_to_buffer(
                &dev,
                &graph.buffers.token_ids_dev,
                token_id,
            )?;
        }

        // Embedding gather
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

        // Forward all layers
        for (i, layer) in self.layers.iter().enumerate() {
            layer.forward_graph(i, &graph.buffers, &dev)?;

            if let Some(Some(moe_block)) = self.moe_blocks.get(i) {
                let act = &graph.buffers.act_q81_buf[i];
                let ffn_shape = graph.buffers.layer_output[i].shape().clone();

                // 1. FFN norm into staging
                dev.rms_norm_into(
                    &graph.buffers.layer_output[i],
                    &**moe_block.ffn_norm.weight.storage(),
                    moe_block.ffn_norm.eps,
                    &graph.buffers.norm_buf[i],
                    &ffn_shape,
                )
                .map_err(grim_core::error::Error::Tensor)?;
                let normed: &Storage = &graph.buffers.norm_buf[i];

                // 2. Router gate GEMV [batch, n_expert]
                linear_into(
                    &dev,
                    normed,
                    moe_block.moe.router.gate.weight(),
                    &graph.buffers.moe_gate_logits[i],
                    act,
                )?;

                // 3. Top-K routing on-device
                let num_experts = moe_block.moe.router.num_experts;
                let top_k = moe_block.moe.router.top_k.min(num_experts).max(1);
                let (route_mode, bias_rocm) = match moe_block.moe.router.kind {
                    grim_nn::moe::RouterKind::SoftmaxTopK => (0, None),
                    grim_nn::moe::RouterKind::SoftmaxTopKRenorm => (3, None),
                    grim_nn::moe::RouterKind::SigmoidTopKWithBias => {
                        let b_rocm = moe_block.moe.router.correction_bias.as_ref().and_then(|b| {
                            b.storage()
                                .as_ref()
                                .as_any()
                                .downcast_ref::<grim_backend_rocm::RocmStorage>()
                        });
                        (2, b_rocm)
                    }
                };

                dev.moe_route_topk_on_device(
                    &graph.buffers.moe_gate_logits[i],
                    bias_rocm,
                    &graph.buffers.moe_route_tokens,
                    &graph.buffers.moe_route_experts,
                    &graph.buffers.moe_route_weights,
                    batch,
                    num_experts,
                    top_k,
                    route_mode,
                )
                .map_err(|e| grim_core::error::Error::Backend(format!("moe route: {e}")))?;

                // 4. Resident scratch + stacked weights (cache hit after warmup)
                let experts = (0..num_experts)
                    .map(|e| crate::shared_moe::MoeExpert {
                        gate: moe_block.moe.experts.gate[e].clone(),
                        up: moe_block.moe.experts.up[e].clone(),
                        down: moe_block.moe.experts.down[e].clone(),
                    })
                    .collect::<Vec<_>>();

                // Global / layer charon cache: ensure resident weights
                static CHARON_CACHES: std::sync::OnceLock<
                    std::sync::Mutex<
                        std::collections::HashMap<
                            usize,
                            std::sync::Arc<crate::shared_moe::CharonCache>,
                        >,
                    >,
                > = std::sync::OnceLock::new();
                let caches = CHARON_CACHES
                    .get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
                let cache = {
                    let mut guard = caches.lock().unwrap_or_else(|e| e.into_inner());
                    guard
                        .entry(i)
                        .or_insert_with(|| {
                            std::sync::Arc::new(crate::shared_moe::CharonCache::new())
                        })
                        .clone()
                };

                let (_, _, _, gate_buf, up_buf, down_buf) =
                    crate::shared_moe::ensure_charon_scratch(
                        dev.ordinal(),
                        batch,
                        top_k,
                        &experts,
                        &cache,
                    )?;

                let norm_rocm = dst_downcast(normed)?;
                dev.moe_fused_dispatch_resident_routing_into(
                    norm_rocm,
                    gate_buf.as_ref(),
                    up_buf.as_ref(),
                    down_buf.as_ref(),
                    &graph.buffers.moe_route_tokens,
                    &graph.buffers.moe_route_experts,
                    &graph.buffers.moe_route_weights,
                    batch * top_k,
                    &graph.buffers.moe_out[i],
                    hidden,
                    experts[0].gate.weight.shape().dim(0).unwrap_or(0),
                    moe_block.moe.routed_scaling_factor,
                )
                .map_err(|e| grim_core::error::Error::Backend(format!("moe dispatch: {e}")))?;

                // 5. Residual add in place
                add_graph(
                    &graph.buffers.layer_output[i],
                    &graph.buffers.moe_out[i],
                    &graph.buffers.layer_output[i],
                    &dev,
                )?;

                // Publish to next layer or head_input
                let n_layers = graph.buffers.layer_input.len();
                let dst: &Storage = if i + 1 < n_layers {
                    &graph.buffers.layer_input[i + 1]
                } else {
                    &graph.buffers.head_input
                };
                publish_into(&dev, dst, &graph.buffers.layer_output[i])?;
            }
        }

        // Final norm + output head
        let h_shape = graph.buffers.head_input.shape().clone();
        dev.rms_norm_into(
            &graph.buffers.head_input,
            &**self.norm.weight.storage(),
            self.norm.eps,
            &graph.buffers.head_input,
            &h_shape,
        )
        .map_err(grim_core::error::Error::Tensor)?;

        linear_into(
            &dev,
            &graph.buffers.head_input,
            self.output.weight(),
            &graph.buffers.head_output,
            &graph.buffers.act_q81_buf[0],
        )?;

        Ok(())
    }

    fn forward_replay(&self, graph: &mut DecodeGraph, token_id: u32) -> Result<()> {
        if !graph.is_captured {
            return Err(grim_core::error::Error::Backend(
                "forward_replay before capture".into(),
            ));
        }
        let dev = dev_for_llama(self)?;
        crate::lfm2_graph::write_embedding_to_buffer(&dev, &graph.buffers.token_ids_dev, token_id)?;

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

    fn eager_kv_seed_sources<'a>(
        &self,
        session: &'a dyn grim_core::session::SessionT,
        valid_rows: u32,
    ) -> Result<Vec<Option<EagerKvSource<'a>>>> {
        let caches = session
            .model_state()
            .and_then(|s| s.downcast_ref::<Vec<Option<crate::block::LlamaLayerCache>>>())
            .ok_or_else(|| {
                grim_core::error::Error::Session(
                    "missing or invalid LlamaLayerCache in session".into(),
                )
            })?;

        if caches.len() != self.layers.len() {
            return Err(grim_core::error::Error::Session(format!(
                "eager_kv_seed_sources: {} caches != {} layers",
                caches.len(),
                self.layers.len()
            )));
        }

        let mut out = Vec::with_capacity(self.layers.len());
        for cache in caches.iter() {
            let (k_dev, v_dev) = match cache {
                Some(c) => (&c.k_device, &c.v_device),
                None => {
                    if valid_rows > 0 {
                        return Err(grim_core::error::Error::Session(
                            "eager_kv_seed_sources: missing layer cache".into(),
                        ));
                    }
                    out.push(None);
                    continue;
                }
            };

            let (k_st, v_st) = match (k_dev.as_deref(), v_dev.as_deref()) {
                (Some(k), Some(v)) => (k, v),
                _ => {
                    if valid_rows > 0 {
                        return Err(grim_core::error::Error::Session(
                            "eager_kv_seed_sources: missing device KV arenas".into(),
                        ));
                    }
                    out.push(None);
                    continue;
                }
            };

            let (k_rocm, v_rocm) = match (as_rocm(k_st), as_rocm(v_st)) {
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
}

// ─── DecodeGraphModel implementation for LFM2 ──────────────────────────────

impl DecodeGraphModel for crate::lfm2::Lfm2 {
    fn get_or_create_decode_graph(&self, max_ctx: usize, batch: usize) -> Result<DecodeGraph> {
        self.get_or_create_decode_graph(max_ctx, batch)
    }

    fn forward_capture(&self, graph: &mut DecodeGraph, token_id: u32) -> Result<()> {
        self.forward_capture(graph, token_id)
    }

    fn forward_replay(&self, graph: &mut DecodeGraph, token_id: u32) -> Result<()> {
        self.forward_replay(graph, token_id)
    }

    fn eager_kv_seed_sources<'a>(
        &self,
        session: &'a dyn grim_core::session::SessionT,
        valid_rows: u32,
    ) -> Result<Vec<Option<EagerKvSource<'a>>>> {
        let caches = session
            .model_state()
            .and_then(|s| s.downcast_ref::<Vec<Option<crate::lfm2::Lfm2LayerCache>>>())
            .ok_or_else(|| {
                grim_core::error::Error::Session(
                    "missing or invalid Lfm2LayerCache in session".into(),
                )
            })?;
        self.eager_kv_seed_sources(caches, valid_rows)
    }

    fn eager_conv_seed_rings<'a>(
        &self,
        session: &'a dyn grim_core::session::SessionT,
    ) -> Result<Vec<Option<ConvRingSeed<'a>>>> {
        let caches = session
            .model_state()
            .and_then(|s| s.downcast_ref::<Vec<Option<crate::lfm2::Lfm2LayerCache>>>())
            .ok_or_else(|| {
                grim_core::error::Error::Session(
                    "missing or invalid Lfm2LayerCache in session".into(),
                )
            })?;
        let mut out = Vec::with_capacity(self.layers.len());
        for (layer, cache) in self.layers.iter().zip(caches.iter()) {
            let Some(crate::lfm2::Lfm2LayerCache::ShortConv { host, .. }) = cache else {
                out.push(None);
                continue;
            };
            // in_proj outputs 3*h_dim (b, c, x); conv kernel taps l_cache with
            // ring depth kc = l_cache - 1.
            let out_dim = layer
                .shortconv_in_proj
                .as_ref()
                .and_then(|l| l.weight.shape().dim(0).ok())
                .ok_or_else(|| {
                    grim_core::error::Error::Session(
                        "conv cache on layer without shortconv_in_proj".into(),
                    )
                })?;
            if out_dim % 3 != 0 {
                return Err(grim_core::error::Error::Session(
                    "shortconv_in_proj out_dim not divisible by 3".into(),
                ));
            }
            let h_dim = out_dim / 3;
            let l_cache = layer
                .shortconv_conv
                .as_ref()
                .and_then(|c| c.shape().dims().last().copied())
                .ok_or_else(|| {
                    grim_core::error::Error::Session(
                        "conv cache on layer without shortconv_conv".into(),
                    )
                })?;
            let kc = l_cache.saturating_sub(1);
            if host.len() != h_dim * kc {
                return Err(grim_core::error::Error::Session(format!(
                    "conv ring {} != h_dim*kc {}",
                    host.len(),
                    h_dim * kc
                )));
            }
            out.push(Some(ConvRingSeed {
                host,
                h_dim,
                kc,
                _anchor: std::marker::PhantomData,
            }));
        }
        Ok(out)
    }
}

// ─── Qwen3.5 Block Graph Forward ──────────────────────────────────────────

fn dev_for_qwen35(qwen: &Qwen35) -> Result<Arc<Dev>> {
    match &qwen.device {
        Device::Rocm(o) => Ok(Dev::shared(*o)),
        _ => Err(grim_core::error::Error::Unimplemented(
            "decode graph needs ROCm device".into(),
        )),
    }
}

impl Qwen35Block {
    /// Enqueue all kernels for this Qwen35 block into the decode graph bracket.
    pub fn forward_graph(
        &self,
        layer_idx: usize,
        buffers: &DecodeGraphBuffers,
        dev: &Dev,
    ) -> Result<()> {
        let batch = buffers.batch.max(1);
        let act = &buffers.act_q81_buf[layer_idx];

        // 1. RMSNorm into norm_buf
        let h_shape = buffers.layer_input[layer_idx].shape().clone();
        dev.rms_norm_into(
            &buffers.layer_input[layer_idx],
            &**self.attn_norm.weight.storage(),
            self.attn_norm.eps,
            &buffers.norm_buf[layer_idx],
            &h_shape,
        )
        .map_err(grim_core::error::Error::Tensor)?;
        let normed: &Storage = &buffers.norm_buf[layer_idx];

        let hd = self.head_dim;
        let nh = self.num_heads;
        let nkv = self.num_kv_heads;

        if self.is_full_attention {
            // 2. Full Attention Path: Q, K, V projections
            let fused_qkv = self
                .wqkv_q80_fused
                .as_ref()
                .filter(|_| dot_fused_ok(dev, self.hidden_size));

            if let Some(fused) = fused_qkv {
                let norm_rocm = dst_downcast(normed)?;
                let m = buffers.batch.max(1);
                dev.launch_quantize_q8_1(norm_rocm, act, m, self.hidden_size)
                    .map_err(|e| {
                        grim_core::error::Error::Backend(format!("qwen35 qkv quant: {e}"))
                    })?;
                dev.launch_fused_qkv_dot4_into(
                    act,
                    &fused.storage,
                    &buffers.fused_qkv_out[layer_idx],
                    fused.n_q,
                    fused.n_k,
                    self.hidden_size,
                )
                .map_err(|e| grim_core::error::Error::Backend(format!("qwen35 fused qkv: {e}")))?;

                let staged: &Storage = &buffers.fused_qkv_out[layer_idx];
                dev.copy_slice_into(&buffers.q_buf[layer_idx], staged, 0, fused.n_q)
                    .map_err(grim_core::error::Error::Tensor)?;
                dev.copy_slice_range(&buffers.k_buf[layer_idx], 0, staged, fused.n_q, fused.n_k)
                    .map_err(grim_core::error::Error::Tensor)?;
                dev.copy_slice_range(
                    &buffers.v_buf[layer_idx],
                    0,
                    staged,
                    fused.n_q + fused.n_k,
                    fused.n_v,
                )
                .map_err(grim_core::error::Error::Tensor)?;
            } else {
                let wq = self.wq.as_ref().ok_or_else(|| {
                    grim_core::error::Error::Backend("missing wq in full attention block".into())
                })?;
                let wk = self.wk.as_ref().ok_or_else(|| {
                    grim_core::error::Error::Backend("missing wk in full attention block".into())
                })?;
                let wv = self.wv.as_ref().ok_or_else(|| {
                    grim_core::error::Error::Backend("missing wv in full attention block".into())
                })?;
                linear_into(dev, normed, wq.weight(), &buffers.q_buf[layer_idx], act)?;
                linear_into(dev, normed, wk.weight(), &buffers.k_buf[layer_idx], act)?;
                linear_into(dev, normed, wv.weight(), &buffers.v_buf[layer_idx], act)?;
            }

            // 3. Optional per-head Q/K norm
            if let Some(qn) = &self.attn_q_norm {
                let qn_shape = Shape::new(vec![batch * nh, hd]);
                dev.rms_norm_into(
                    &buffers.q_buf[layer_idx],
                    &**qn.weight.storage(),
                    qn.eps,
                    &buffers.q_buf[layer_idx],
                    &qn_shape,
                )
                .map_err(grim_core::error::Error::Tensor)?;
            }
            if let Some(kn) = &self.attn_k_norm {
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

            // 4. RoPE
            let steps = 1usize;
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

            // 5. KV append to arenas
            let kv_stride = nkv * hd;
            let arena_slot_stride = buffers.max_ctx * kv_stride;
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

            // 6. Attention kernel
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

            // Bump pos_dev
            grim_backend_rocm::launch_bump_i32_slots(dev, &buffers.pos_dev, steps, batch)
                .map_err(|e| grim_core::error::Error::Backend(format!("bump: {e}")))?;

            // 7. Output projection (wo)
            let wo = self.wo.as_ref().ok_or_else(|| {
                grim_core::error::Error::Backend("missing wo in full attention block".into())
            })?;
            linear_into(
                dev,
                &buffers.attn_out_buf[layer_idx],
                wo.weight(),
                &buffers.norm_buf[layer_idx],
                act,
            )?;
        } else {
            // Recurrent / ShortConv / SSM path
            if let Some(ref qkv_lin) = self.attn_qkv {
                linear_into(
                    dev,
                    normed,
                    qkv_lin.weight(),
                    &buffers.q_buf[layer_idx],
                    act,
                )?;

                // If conv weight is present and we have allocated sc_state, run causal conv step
                if let Some(ref conv_t) = self.ssm_conv1d {
                    if layer_idx < buffers.sc_state.len() {
                        dev.short_conv1d_causal_step_into(
                            &buffers.q_buf[layer_idx],
                            conv_t.storage().as_ref(),
                            None,
                            &buffers.sc_state[layer_idx],
                            &buffers.attn_out_buf[layer_idx],
                        )
                        .map_err(|e| {
                            grim_core::error::Error::Backend(format!("qwen35 sc conv: {e}"))
                        })?;
                        // SiLU in-place on attn_out_buf
                        dev.silu_into(
                            &buffers.attn_out_buf[layer_idx],
                            &buffers.attn_out_buf[layer_idx],
                        )
                        .map_err(grim_core::error::Error::Tensor)?;
                    } else {
                        dev.silu_into(&buffers.q_buf[layer_idx], &buffers.attn_out_buf[layer_idx])
                            .map_err(grim_core::error::Error::Tensor)?;
                    }
                } else {
                    dev.silu_into(&buffers.q_buf[layer_idx], &buffers.attn_out_buf[layer_idx])
                        .map_err(grim_core::error::Error::Tensor)?;
                }
            } else {
                return Err(grim_core::error::Error::Backend(
                    "recurrent block missing attn_qkv".into(),
                ));
            }

            // Optional attn_gate (sigmoid gate)
            if let Some(ref gate_lin) = self.attn_gate {
                linear_into(
                    dev,
                    normed,
                    gate_lin.weight(),
                    &buffers.gate_buf[layer_idx],
                    act,
                )?;
                dev.sigmoid_into(&buffers.gate_buf[layer_idx], &buffers.gate_buf[layer_idx])
                    .map_err(grim_core::error::Error::Tensor)?;
                dev.mul_into(
                    &buffers.attn_out_buf[layer_idx],
                    &buffers.gate_buf[layer_idx],
                    &buffers.attn_out_buf[layer_idx],
                )
                .map_err(grim_core::error::Error::Tensor)?;
            }

            // Output projection (wo or ssm_out)
            let out_proj = self.wo.as_ref().or(self.ssm_out.as_ref()).ok_or_else(|| {
                grim_core::error::Error::Backend("missing wo/ssm_out projection".into())
            })?;
            linear_into(
                dev,
                &buffers.attn_out_buf[layer_idx],
                out_proj.weight(),
                &buffers.norm_buf[layer_idx],
                act,
            )?;
        }

        // Residual 1 add: layer_input + branch_proj -> layer_output
        add_graph(
            &buffers.layer_input[layer_idx],
            &buffers.norm_buf[layer_idx],
            &buffers.layer_output[layer_idx],
            dev,
        )?;

        // 8. FFN sublayer
        let ffn_shape = buffers.layer_output[layer_idx].shape().clone();
        dev.rms_norm_into(
            &buffers.layer_output[layer_idx],
            &**self.post_attention_norm.weight.storage(),
            self.post_attention_norm.eps,
            &buffers.norm_buf[layer_idx],
            &ffn_shape,
        )
        .map_err(grim_core::error::Error::Tensor)?;
        let normed_ffn: &Storage = &buffers.norm_buf[layer_idx];

        linear_into(
            dev,
            normed_ffn,
            self.ffn_gate.weight(),
            &buffers.gate_buf[layer_idx],
            act,
        )?;
        linear_into(
            dev,
            normed_ffn,
            self.ffn_up.weight(),
            &buffers.up_buf[layer_idx],
            act,
        )?;

        dev.silu_mul_into(
            &buffers.gate_buf[layer_idx],
            &buffers.up_buf[layer_idx],
            &buffers.activated_buf[layer_idx],
        )
        .map_err(grim_core::error::Error::Tensor)?;

        linear_into(
            dev,
            &buffers.activated_buf[layer_idx],
            self.ffn_down.weight(),
            &buffers.norm_buf[layer_idx],
            act,
        )?;

        // Residual 2 add: layer_output + ffn_out -> layer_output
        add_graph(
            &buffers.layer_output[layer_idx],
            &buffers.norm_buf[layer_idx],
            &buffers.layer_output[layer_idx],
            dev,
        )?;

        // Publish to next layer or head_input
        let n_layers = buffers.layer_input.len();
        let dst: &Storage = if layer_idx + 1 < n_layers {
            &buffers.layer_input[layer_idx + 1]
        } else {
            &buffers.head_input
        };
        publish_into(dev, dst, &buffers.layer_output[layer_idx])?;
        Ok(())
    }
}

// ─── DecodeGraphModel implementation for Qwen3.5 ───────────────────────────

impl DecodeGraphModel for Qwen35 {
    fn get_or_create_decode_graph(&self, max_ctx: usize, batch: usize) -> Result<DecodeGraph> {
        if !decode_graph_enabled() {
            return Err(grim_core::error::Error::Backend(
                "decode graph disabled by env".into(),
            ));
        }
        let dev = dev_for_qwen35(self)?;
        let stream = dev
            .get_stream_from_pool(0)
            .ok_or_else(|| grim_core::error::Error::Backend("no stream in pool".into()))?;

        let hidden = self.cfg.hidden_size;
        let n_q = self.cfg.num_heads * self.cfg.head_dim;
        let n_k = self.cfg.num_kv_heads * self.cfg.head_dim;
        let n_v = n_k;
        let inter = self.cfg.intermediate_size;
        let vocab = self.cfg.vocab_size.max(1);
        let ctx = max_ctx.max(1);
        let nh = self.cfg.num_heads;
        let sc_h_dim = n_q;
        let sc_l_cache = self.cfg.ssm_d_conv.max(4);

        let buffers = DecodeGraphBuffers::allocate(
            &dev,
            self.blocks.len(),
            hidden,
            n_q,
            n_k,
            n_v,
            inter,
            ctx,
            vocab,
            nh,
            batch,
            0,
            0,
            sc_h_dim,
            sc_l_cache,
        )
        .map_err(|e| grim_core::error::Error::Backend(format!("graph pool alloc: {e}")))?;

        Ok(DecodeGraph::new(&dev, buffers, stream))
    }

    fn forward_capture(&self, graph: &mut DecodeGraph, token_id: u32) -> Result<()> {
        if !decode_graph_enabled() {
            return Err(grim_core::error::Error::Backend(
                "decode graph disabled".into(),
            ));
        }
        let dev = dev_for_qwen35(self)?;
        if !graph.capturing {
            crate::lfm2_graph::write_embedding_to_buffer(
                &dev,
                &graph.buffers.token_ids_dev,
                token_id,
            )?;
        }

        // Embedding gather
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

        // Forward all layers
        for (i, block) in self.blocks.iter().enumerate() {
            block.forward_graph(i, &graph.buffers, &dev)?;
        }

        // Final norm + output head
        let h_shape = graph.buffers.head_input.shape().clone();
        dev.rms_norm_into(
            &graph.buffers.head_input,
            &**self.output_norm.weight.storage(),
            self.output_norm.eps,
            &graph.buffers.head_input,
            &h_shape,
        )
        .map_err(grim_core::error::Error::Tensor)?;

        linear_into(
            &dev,
            &graph.buffers.head_input,
            self.output.weight(),
            &graph.buffers.head_output,
            &graph.buffers.act_q81_buf[0],
        )?;

        Ok(())
    }

    fn forward_replay(&self, graph: &mut DecodeGraph, token_id: u32) -> Result<()> {
        if !graph.is_captured {
            return Err(grim_core::error::Error::Backend(
                "forward_replay before capture".into(),
            ));
        }
        let dev = dev_for_qwen35(self)?;
        crate::lfm2_graph::write_embedding_to_buffer(&dev, &graph.buffers.token_ids_dev, token_id)?;

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

    fn eager_kv_seed_sources<'a>(
        &self,
        session: &'a dyn grim_core::session::SessionT,
        valid_rows: u32,
    ) -> Result<Vec<Option<EagerKvSource<'a>>>> {
        let caches = session
            .model_state()
            .and_then(|s| s.downcast_ref::<Vec<Qwen35LayerCache>>())
            .ok_or_else(|| {
                grim_core::error::Error::Session(
                    "missing or invalid Qwen35LayerCache in session".into(),
                )
            })?;

        if caches.len() != self.blocks.len() {
            return Err(grim_core::error::Error::Session(format!(
                "eager_kv_seed_sources: {} caches != {} blocks",
                caches.len(),
                self.blocks.len()
            )));
        }

        let mut out = Vec::with_capacity(self.blocks.len());
        for (i, cache) in caches.iter().enumerate() {
            if !self.blocks[i].is_full_attention {
                // Non-attention (SSM/recurrent) layer has no dense KV cache arena
                out.push(None);
                continue;
            }

            let (k_st, v_st) = match (cache.k_device.as_deref(), cache.v_device.as_deref()) {
                (Some(k), Some(v)) => (k, v),
                _ => {
                    if valid_rows > 0 {
                        return Err(grim_core::error::Error::Session(
                            "eager_kv_seed_sources: missing device KV arenas in full attention layer".into(),
                        ));
                    }
                    out.push(None);
                    continue;
                }
            };

            let (k_rocm, v_rocm) = match (as_rocm(k_st), as_rocm(v_st)) {
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

            // The eager Qwen35 KV arena is 3-D `[rows, num_kv_heads, head_dim]`,
            // while the decode-graph arena it seeds is 2-D `[rows, nkv*hd]`. Taking
            // `dims().last()` reports `head_dim` (256) rather than the row stride
            // (1024), which aborted graph capture with an arena-width mismatch. The
            // row stride is the product of every dim after the row dim.
            let kv_stride: usize = k_rocm.shape().dims().iter().skip(1).product();
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
}

// ─── Gemma2 Block Graph Forward ───────────────────────────────────────────

fn dev_for_gemma2(gemma: &Gemma2) -> Result<Arc<Dev>> {
    match &gemma.device {
        Device::Rocm(o) => Ok(Dev::shared(*o)),
        _ => Err(grim_core::error::Error::Unimplemented(
            "decode graph needs ROCm device".into(),
        )),
    }
}

impl Gemma2Block {
    /// Enqueue all kernels for this Gemma2 block into the decode graph bracket.
    pub fn forward_graph(
        &self,
        layer_idx: usize,
        buffers: &DecodeGraphBuffers,
        dev: &Dev,
    ) -> Result<()> {
        let batch = buffers.batch.max(1);
        let act = &buffers.act_q81_buf[layer_idx];

        // 1. input_layernorm into norm_buf
        let h_shape = buffers.layer_input[layer_idx].shape().clone();
        dev.rms_norm_into(
            &buffers.layer_input[layer_idx],
            &**self.input_layernorm.weight.storage(),
            self.input_layernorm.eps,
            &buffers.norm_buf[layer_idx],
            &h_shape,
        )
        .map_err(grim_core::error::Error::Tensor)?;
        let normed: &Storage = &buffers.norm_buf[layer_idx];

        // 2. QKV projections
        linear_into(
            dev,
            normed,
            self.wq.weight(),
            &buffers.q_buf[layer_idx],
            act,
        )?;
        linear_into(
            dev,
            normed,
            self.wk.weight(),
            &buffers.k_buf[layer_idx],
            act,
        )?;
        linear_into(
            dev,
            normed,
            self.wv.weight(),
            &buffers.v_buf[layer_idx],
            act,
        )?;

        // 3. RoPE in-place using device position base (`pos_dev`)
        let hd = self.head_dim;
        let nh = self.num_heads;
        let nkv = self.num_kv_heads;
        let steps = 1usize;
        let rope_cfg = RopeConfig::new(hd, self.rope.config.base);

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

        // 4. KV append to arenas
        let kv_stride = nkv * hd;
        let arena_slot_stride = buffers.max_ctx * kv_stride;
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

        // 5. Attention kernel with optional softcapping
        let softcap = self.attn_logit_softcapping.unwrap_or(0.0);
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
            softcap,
            &buffers.attn_dummy,
            0,
            0,
            &buffers.attn_dummy,
            0,
            batch,
            arena_slot_stride,
        )
        .map_err(|e| grim_core::error::Error::Backend(format!("qkv_attention: {e}")))?;

        // Bump pos_dev
        grim_backend_rocm::launch_bump_i32_slots(dev, &buffers.pos_dev, steps, batch)
            .map_err(|e| grim_core::error::Error::Backend(format!("bump: {e}")))?;

        // 6. Output projection (wo)
        linear_into(
            dev,
            &buffers.attn_out_buf[layer_idx],
            self.wo.weight(),
            &buffers.norm_buf[layer_idx],
            act,
        )?;

        // 7. post_attention_layernorm on wo output
        dev.rms_norm_into(
            &buffers.norm_buf[layer_idx],
            &**self.post_attention_layernorm.weight.storage(),
            self.post_attention_layernorm.eps,
            &buffers.norm_buf[layer_idx],
            &h_shape,
        )
        .map_err(grim_core::error::Error::Tensor)?;

        // 8. Residual 1 add: layer_input + post_attn -> layer_output
        add_graph(
            &buffers.layer_input[layer_idx],
            &buffers.norm_buf[layer_idx],
            &buffers.layer_output[layer_idx],
            dev,
        )?;

        // 9. pre_feedforward_layernorm into norm_buf
        dev.rms_norm_into(
            &buffers.layer_output[layer_idx],
            &**self.pre_feedforward_layernorm.weight.storage(),
            self.pre_feedforward_layernorm.eps,
            &buffers.norm_buf[layer_idx],
            &h_shape,
        )
        .map_err(grim_core::error::Error::Tensor)?;
        let normed_ffn: &Storage = &buffers.norm_buf[layer_idx];

        // 10. Gemma2 MLP: gate_proj, up_proj, gelu_tanh_mul_into, down_proj
        linear_into(
            dev,
            normed_ffn,
            self.mlp.gate_proj.weight(),
            &buffers.gate_buf[layer_idx],
            act,
        )?;
        linear_into(
            dev,
            normed_ffn,
            self.mlp.up_proj.weight(),
            &buffers.up_buf[layer_idx],
            act,
        )?;

        dev.gelu_tanh_mul_into(
            &buffers.gate_buf[layer_idx],
            &buffers.up_buf[layer_idx],
            &buffers.activated_buf[layer_idx],
        )
        .map_err(grim_core::error::Error::Tensor)?;

        linear_into(
            dev,
            &buffers.activated_buf[layer_idx],
            self.mlp.down_proj.weight(),
            &buffers.norm_buf[layer_idx],
            act,
        )?;

        // 11. post_feedforward_layernorm on mlp_out
        dev.rms_norm_into(
            &buffers.norm_buf[layer_idx],
            &**self.post_feedforward_layernorm.weight.storage(),
            self.post_feedforward_layernorm.eps,
            &buffers.norm_buf[layer_idx],
            &h_shape,
        )
        .map_err(grim_core::error::Error::Tensor)?;

        // 12. Residual 2 add: layer_output + post_ffn -> layer_output
        add_graph(
            &buffers.layer_output[layer_idx],
            &buffers.norm_buf[layer_idx],
            &buffers.layer_output[layer_idx],
            dev,
        )?;

        // Publish to next layer or head_input
        let n_layers = buffers.layer_input.len();
        let dst: &Storage = if layer_idx + 1 < n_layers {
            &buffers.layer_input[layer_idx + 1]
        } else {
            &buffers.head_input
        };
        publish_into(dev, dst, &buffers.layer_output[layer_idx])?;
        Ok(())
    }
}

// ─── DecodeGraphModel implementation for Gemma2 ───────────────────────────

impl DecodeGraphModel for Gemma2 {
    fn get_or_create_decode_graph(&self, max_ctx: usize, batch: usize) -> Result<DecodeGraph> {
        if !decode_graph_enabled() {
            return Err(grim_core::error::Error::Backend(
                "decode graph disabled by env".into(),
            ));
        }
        let dev = dev_for_gemma2(self)?;
        let stream = dev
            .get_stream_from_pool(0)
            .ok_or_else(|| grim_core::error::Error::Backend("no stream in pool".into()))?;

        let hidden = self.cfg.hidden_size;
        let n_q = self.cfg.num_attention_heads * self.cfg.head_dim;
        let n_k = self.cfg.num_key_value_heads * self.cfg.head_dim;
        let n_v = n_k;
        let inter = self.cfg.intermediate_size;
        let vocab = self.cfg.vocab_size.max(1);
        let ctx = max_ctx.max(1);
        let nh = self.cfg.num_attention_heads;

        let buffers = DecodeGraphBuffers::allocate(
            &dev,
            self.layers.len(),
            hidden,
            n_q,
            n_k,
            n_v,
            inter,
            ctx,
            vocab,
            nh,
            batch,
            0,
            0,
            0,
            0,
        )
        .map_err(|e| grim_core::error::Error::Backend(format!("graph pool alloc: {e}")))?;

        Ok(DecodeGraph::new(&dev, buffers, stream))
    }

    fn forward_capture(&self, graph: &mut DecodeGraph, token_id: u32) -> Result<()> {
        if !decode_graph_enabled() {
            return Err(grim_core::error::Error::Backend(
                "decode graph disabled".into(),
            ));
        }
        let dev = dev_for_gemma2(self)?;
        if !graph.capturing {
            crate::lfm2_graph::write_embedding_to_buffer(
                &dev,
                &graph.buffers.token_ids_dev,
                token_id,
            )?;
        }

        // Embedding gather
        let w = dst_downcast(self.tok_embeddings.weight().storage().as_ref())?;
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

        // Forward all layers
        for (i, layer) in self.layers.iter().enumerate() {
            layer.forward_graph(i, &graph.buffers, &dev)?;
        }

        // Final norm + output head
        let h_shape = graph.buffers.head_input.shape().clone();
        dev.rms_norm_into(
            &graph.buffers.head_input,
            &**self.norm.weight.storage(),
            self.norm.eps,
            &graph.buffers.head_input,
            &h_shape,
        )
        .map_err(grim_core::error::Error::Tensor)?;

        linear_into(
            &dev,
            &graph.buffers.head_input,
            self.output.weight(),
            &graph.buffers.head_output,
            &graph.buffers.act_q81_buf[0],
        )?;

        // Optional final logit softcapping inside device graph
        if let Some(cap) = self.cfg.final_logit_softcapping {
            dev.tanh_softcap_into(&*graph.buffers.head_output, cap, &graph.buffers.head_output)
                .map_err(grim_core::error::Error::Tensor)?;
        }

        Ok(())
    }

    fn forward_replay(&self, graph: &mut DecodeGraph, token_id: u32) -> Result<()> {
        if !graph.is_captured {
            return Err(grim_core::error::Error::Backend(
                "forward_replay before capture".into(),
            ));
        }
        let dev = dev_for_gemma2(self)?;
        crate::lfm2_graph::write_embedding_to_buffer(&dev, &graph.buffers.token_ids_dev, token_id)?;

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

    fn eager_kv_seed_sources<'a>(
        &self,
        _session: &'a dyn grim_core::session::SessionT,
        _valid_rows: u32,
    ) -> Result<Vec<Option<EagerKvSource<'a>>>> {
        // Gemma2 currently uses a stateless session or eager KV cache;
        // returns None for each layer so graph seeding succeeds cleanly.
        Ok((0..self.layers.len()).map(|_| None).collect())
    }
}

// ─── Chameleon Block Graph Forward ────────────────────────────────────────

impl ChameleonBlock {
    /// Enqueue all kernels for this Chameleon block into the decode graph bracket.
    /// Mirrors the eager forward: pre-RMSNorm → fused/split QKV → per-head
    /// Q/K LayerNorm (swin_norm) → RoPE → KV append → GQA → wo → SwiGLU FFN.
    pub fn forward_graph(
        &self,
        layer_idx: usize,
        buffers: &DecodeGraphBuffers,
        dev: &Dev,
    ) -> Result<()> {
        let batch = buffers.batch.max(1);
        let act = &buffers.act_q81_buf[layer_idx];

        // 1. Pre-attention RMSNorm into norm_buf
        let h_shape = buffers.layer_input[layer_idx].shape().clone();
        dev.rms_norm_into(
            &buffers.layer_input[layer_idx],
            &**self.attn_norm.weight.storage(),
            self.attn_norm.eps,
            &buffers.norm_buf[layer_idx],
            &h_shape,
        )
        .map_err(grim_core::error::Error::Tensor)?;
        let normed: &Storage = &buffers.norm_buf[layer_idx];

        let hd = self.head_dim;
        let nh = self.num_heads;
        let nkv = self.num_kv_heads;

        // 2. Q/K/V projections (fused Q8_0 dot4 when available, else 3 GEMVs)
        if let Some(fused) = self
            .wqkv_q80_fused
            .as_ref()
            .filter(|f| dot_fused_ok(dev, f.hidden))
        {
            let norm_rocm = dst_downcast(normed)?;
            let m = batch;
            dev.launch_quantize_q8_1(norm_rocm, act, m, fused.hidden)
                .map_err(|e| {
                    grim_core::error::Error::Backend(format!("chameleon qkv quant: {e}"))
                })?;
            dev.launch_fused_qkv_dot4_into(
                act,
                &fused.storage,
                &buffers.fused_qkv_out[layer_idx],
                fused.n_q,
                fused.n_k,
                fused.hidden,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("chameleon fused qkv: {e}")))?;

            let staged: &Storage = &buffers.fused_qkv_out[layer_idx];
            dev.copy_slice_into(&buffers.q_buf[layer_idx], staged, 0, fused.n_q)
                .map_err(grim_core::error::Error::Tensor)?;
            dev.copy_slice_range(&buffers.k_buf[layer_idx], 0, staged, fused.n_q, fused.n_k)
                .map_err(grim_core::error::Error::Tensor)?;
            dev.copy_slice_range(
                &buffers.v_buf[layer_idx],
                0,
                staged,
                fused.n_q + fused.n_k,
                fused.n_v,
            )
            .map_err(grim_core::error::Error::Tensor)?;
        } else {
            linear_into(
                dev,
                normed,
                self.wq.weight(),
                &buffers.q_buf[layer_idx],
                act,
            )?;
            linear_into(
                dev,
                normed,
                self.wk.weight(),
                &buffers.k_buf[layer_idx],
                act,
            )?;
            linear_into(
                dev,
                normed,
                self.wv.weight(),
                &buffers.v_buf[layer_idx],
                act,
            )?;
        }

        // 3. Per-head Q/K LayerNorm (swin_norm) — mean/variance, NOT RMS.
        // Eager applies these over [seq*heads, head_dim]; seq == 1 in decode.
        if let Some(qn) = &self.q_norm {
            let qn_shape = Shape::new(vec![batch * nh, hd]);
            let qn_bias = qn.bias.as_ref().map(|b| b.storage().as_ref() as &Storage);
            dev.layer_norm_into(
                &buffers.q_buf[layer_idx],
                &**qn.weight.storage(),
                qn_bias,
                qn.eps,
                &buffers.q_buf[layer_idx],
                &qn_shape,
            )
            .map_err(grim_core::error::Error::Tensor)?;
        }
        if let Some(kn) = &self.k_norm {
            let kn_shape = Shape::new(vec![batch * nkv, hd]);
            let kn_bias = kn.bias.as_ref().map(|b| b.storage().as_ref() as &Storage);
            dev.layer_norm_into(
                &buffers.k_buf[layer_idx],
                &**kn.weight.storage(),
                kn_bias,
                kn.eps,
                &buffers.k_buf[layer_idx],
                &kn_shape,
            )
            .map_err(grim_core::error::Error::Tensor)?;
        }

        // 4. RoPE
        let steps = 1usize;
        let rope_cfg = self.rope.config.clone();
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

        // 5. KV append to arenas
        let kv_stride = nkv * hd;
        let arena_slot_stride = buffers.max_ctx * kv_stride;
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

        // 6. Attention kernel
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

        // Bump pos_dev
        grim_backend_rocm::launch_bump_i32_slots(dev, &buffers.pos_dev, steps, batch)
            .map_err(|e| grim_core::error::Error::Backend(format!("bump: {e}")))?;

        // 7. Output projection (wo)
        linear_into(
            dev,
            &buffers.attn_out_buf[layer_idx],
            self.wo.weight(),
            &buffers.norm_buf[layer_idx],
            act,
        )?;

        // Residual 1 add: layer_input + branch_proj -> layer_output
        add_graph(
            &buffers.layer_input[layer_idx],
            &buffers.norm_buf[layer_idx],
            &buffers.layer_output[layer_idx],
            dev,
        )?;

        // 8. FFN sublayer (pre-RMSNorm → SwiGLU)
        let ffn_shape = buffers.layer_output[layer_idx].shape().clone();
        dev.rms_norm_into(
            &buffers.layer_output[layer_idx],
            &**self.ffn_norm.weight.storage(),
            self.ffn_norm.eps,
            &buffers.norm_buf[layer_idx],
            &ffn_shape,
        )
        .map_err(grim_core::error::Error::Tensor)?;
        let normed_ffn: &Storage = &buffers.norm_buf[layer_idx];

        linear_into(
            dev,
            normed_ffn,
            self.w_gate.weight(),
            &buffers.gate_buf[layer_idx],
            act,
        )?;
        linear_into(
            dev,
            normed_ffn,
            self.w_up.weight(),
            &buffers.up_buf[layer_idx],
            act,
        )?;

        dev.silu_mul_into(
            &buffers.gate_buf[layer_idx],
            &buffers.up_buf[layer_idx],
            &buffers.activated_buf[layer_idx],
        )
        .map_err(grim_core::error::Error::Tensor)?;

        linear_into(
            dev,
            &buffers.activated_buf[layer_idx],
            self.w_down.weight(),
            &buffers.norm_buf[layer_idx],
            act,
        )?;

        // Residual 2 add: layer_output + ffn_out -> layer_output
        add_graph(
            &buffers.layer_output[layer_idx],
            &buffers.norm_buf[layer_idx],
            &buffers.layer_output[layer_idx],
            dev,
        )?;

        // Publish to next layer or head_input
        let n_layers = buffers.layer_input.len();
        let dst: &Storage = if layer_idx + 1 < n_layers {
            &buffers.layer_input[layer_idx + 1]
        } else {
            &buffers.head_input
        };
        publish_into(dev, dst, &buffers.layer_output[layer_idx])?;
        Ok(())
    }
}

// ─── Thin Llama-family wrappers: decode-graph delegation (who-dat: graph coverage) ───

/// Every thin wrapper that renames a `Llama` (`pub inner: Llama`) inherits
/// decode-graph capture/replay/seeding verbatim. Capture, replay, and KV
/// seeding all read the inner `Llama`'s weights and the shared
/// `LlamaLayerCache` session state, so delegation is behavior-identical.
macro_rules! impl_llama_wrapper_graph {
    ($($module:ident :: $name:ident),+ $(,)?) => {
        $(
            impl DecodeGraphModel for crate::$module::$name {
                fn get_or_create_decode_graph(&self, max_ctx: usize, batch: usize) -> Result<DecodeGraph> {
                    self.inner.get_or_create_decode_graph(max_ctx, batch)
                }
                fn forward_capture(&self, graph: &mut DecodeGraph, token_id: u32) -> Result<()> {
                    self.inner.forward_capture(graph, token_id)
                }
                fn forward_replay(&self, graph: &mut DecodeGraph, token_id: u32) -> Result<()> {
                    self.inner.forward_replay(graph, token_id)
                }
                fn eager_conv_seed_rings<'a>(
                    &self,
                    session: &'a dyn grim_core::session::SessionT,
                ) -> Result<Vec<Option<ConvRingSeed<'a>>>> {
                    self.inner.eager_conv_seed_rings(session)
                }
                fn eager_kv_seed_sources<'a>(
                    &self,
                    session: &'a dyn grim_core::session::SessionT,
                    valid_rows: u32,
                ) -> Result<Vec<Option<EagerKvSource<'a>>>> {
                    self.inner.eager_kv_seed_sources(session, valid_rows)
                }
            }
        )+
    };
}

impl_llama_wrapper_graph!(
    afmoe::AfMoe,
    arcee::Arcee,
    apertus::Apertus,
    chatglm::ChatGlm,
    arctic::Arctic,
    codeshell::Codeshell,
    gemma4_assistant::Gemma4Assistant,
    cohere2moe::Cohere2Moe,
    internlm2::InternLm2,
    lladamoe::LladaMoe,
    minimax_m2::MiniMaxM2,
    nemotron::Nemotron,
    bailingmoe2::BailingMoe2,
    ernie45::Ernie45,
    plamo2::Plamo2,
    baichuan::Baichuan,
    eurobert::Eurobert,
    granite::Granite,
    gemma_embedding::GemmaEmbedding,
    exaone4::Exaone4,
    qwen3next::Qwen3Next,
    jais::Jais,
    granite_moe::GraniteMoe,
    cohere2::Cohere2,
    grok::Grok,
    smollm3::SmolLm3,
    bitnet::BitNet,
    llama_embed::LlamaEmbed,
    deci::Deci,
    dflash::DFlash,
    jais2::Jais2,
    mistral4::Mistral4,
    llada::Llada,
    bailingmoe::BailingMoe,
    exaone_moe::ExaoneMoe,
    llama4::Llama4,
    dream::Dream,
    dots1::Dots1,
    olmo::Olmo,
    mistral3::Mistral3,
    exaone::Exaone,
    plm::Plm,
    glm4::Glm4,
    olmoe::Olmoe,
    deepseek2ocr::DeepSeek2Ocr,
    olmo2::Olmo2,
    hunyuan_dense::HunyuanDense,
    openai_moe::OpenAiMoe,
    plamo::Plamo,
    ernie4_5_moe::Ernie45Moe,
    plamo3::Plamo3,
    qwen2moe::Qwen2Moe,
    starcoder2::Starcoder2,
    qwen3::Qwen3,
    stablelm::StableLm,
    qwen::Qwen,
    gptneox::GptNeoX,
    starcoder::Starcoder,
    glmdsa::GlmDsa,
    maincoder::MainCoder,
    hunyuan_moe::HunyuanMoe,
    kimi_linear::KimiLinear,
    laguna::Laguna,
    maple::Maple,
    mimo2::Mimo2,
    mpt::Mpt,
    grovemoe::GroveMoe,
    paddle_ocr::PaddleOcr,
    mellum::Mellum,
    orion::Orion,
    openelm::OpenElm,
    seed_oss::SeedOss,
    pangu_embed::PanguEmbed,
    phi2::Phi2,
    talkie::Talkie,
    rnd1::Rnd1,
    refact::Refact,
    qwen3moe::Qwen3Moe,
    xverse::Xverse,
    smallthinker::SmallThinker,
    smollm2::SmolLm2,
    glm4moe::Glm4Moe,
    step35::Step35,
);

/// Downcast an opaque model handle to whichever thin Llama wrapper it is,
/// viewed as its decode-graph capability. One arm in the decode loop covers
/// every wrapper above.
pub fn llama_wrapper_graph_model(model: &dyn std::any::Any) -> Option<&dyn DecodeGraphModel> {
    macro_rules! try_wrapper {
        ($($module:ident :: $name:ident),+ $(,)?) => {
            $(
                if let Some(m) = model.downcast_ref::<crate::$module::$name>() {
                    return Some(m);
                }
            )+
        };
    }
    try_wrapper!(
        afmoe::AfMoe,
        arcee::Arcee,
        apertus::Apertus,
        chatglm::ChatGlm,
        arctic::Arctic,
        codeshell::Codeshell,
        gemma4_assistant::Gemma4Assistant,
        cohere2moe::Cohere2Moe,
        internlm2::InternLm2,
        lladamoe::LladaMoe,
        minimax_m2::MiniMaxM2,
        nemotron::Nemotron,
        bailingmoe2::BailingMoe2,
        ernie45::Ernie45,
        plamo2::Plamo2,
        baichuan::Baichuan,
        eurobert::Eurobert,
        granite::Granite,
        gemma_embedding::GemmaEmbedding,
        exaone4::Exaone4,
        qwen3next::Qwen3Next,
        jais::Jais,
        granite_moe::GraniteMoe,
        cohere2::Cohere2,
        grok::Grok,
        smollm3::SmolLm3,
        bitnet::BitNet,
        llama_embed::LlamaEmbed,
        deci::Deci,
        dflash::DFlash,
        jais2::Jais2,
        mistral4::Mistral4,
        llada::Llada,
        bailingmoe::BailingMoe,
        exaone_moe::ExaoneMoe,
        llama4::Llama4,
        dream::Dream,
        dots1::Dots1,
        olmo::Olmo,
        mistral3::Mistral3,
        exaone::Exaone,
        plm::Plm,
        glm4::Glm4,
        olmoe::Olmoe,
        deepseek2ocr::DeepSeek2Ocr,
        olmo2::Olmo2,
        hunyuan_dense::HunyuanDense,
        openai_moe::OpenAiMoe,
        plamo::Plamo,
        ernie4_5_moe::Ernie45Moe,
        plamo3::Plamo3,
        qwen2moe::Qwen2Moe,
        starcoder2::Starcoder2,
        qwen3::Qwen3,
        stablelm::StableLm,
        qwen::Qwen,
        gptneox::GptNeoX,
        starcoder::Starcoder,
        glmdsa::GlmDsa,
        maincoder::MainCoder,
        hunyuan_moe::HunyuanMoe,
        kimi_linear::KimiLinear,
        laguna::Laguna,
        maple::Maple,
        mimo2::Mimo2,
        mpt::Mpt,
        grovemoe::GroveMoe,
        paddle_ocr::PaddleOcr,
        mellum::Mellum,
        orion::Orion,
        openelm::OpenElm,
        seed_oss::SeedOss,
        pangu_embed::PanguEmbed,
        phi2::Phi2,
        talkie::Talkie,
        rnd1::Rnd1,
        refact::Refact,
        qwen3moe::Qwen3Moe,
        xverse::Xverse,
        smallthinker::SmallThinker,
        smollm2::SmolLm2,
        glm4moe::Glm4Moe,
        step35::Step35,
    );
    None
}

// ─── DecodeGraphModel implementation for MiniMaxM3 ────────────────────────

fn dev_for_minimax_m3(m: &MiniMaxM3) -> Result<Arc<Dev>> {
    match &m.device {
        Device::Rocm(o) => Ok(Dev::shared(*o)),
        _ => Err(grim_core::error::Error::Unimplemented(
            "decode graph needs ROCm device".into(),
        )),
    }
}

impl DecodeGraphModel for MiniMaxM3 {
    fn get_or_create_decode_graph(&self, max_ctx: usize, batch: usize) -> Result<DecodeGraph> {
        if !decode_graph_enabled() {
            return Err(grim_core::error::Error::Backend(
                "decode graph disabled by env".into(),
            ));
        }
        let dev = dev_for_minimax_m3(self)?;
        let stream = dev
            .get_stream_from_pool(0)
            .ok_or_else(|| grim_core::error::Error::Backend("no stream in pool".into()))?;

        let hidden = self.cfg.hidden_size;
        let n_q = self.cfg.num_attention_heads * self.cfg.head_dim;
        let n_k = self.cfg.num_key_value_heads * self.cfg.head_dim;
        let n_v = n_k;
        let inter = self.cfg.intermediate_size;
        let vocab = self.cfg.vocab_size.max(1);
        let ctx = max_ctx.max(1);
        let nh = self.cfg.num_attention_heads;

        let buffers = DecodeGraphBuffers::allocate(
            &dev,
            self.layers.len(),
            hidden,
            n_q,
            n_k,
            n_v,
            inter,
            ctx,
            vocab,
            nh,
            batch,
            self.cfg.num_experts,
            self.cfg.num_experts_per_tok,
            0,
            0,
        )
        .map_err(|e| grim_core::error::Error::Backend(format!("graph pool alloc: {e}")))?;

        Ok(DecodeGraph::new(&dev, buffers, stream))
    }

    fn forward_capture(&self, graph: &mut DecodeGraph, token_id: u32) -> Result<()> {
        if !decode_graph_enabled() {
            return Err(grim_core::error::Error::Backend(
                "decode graph disabled".into(),
            ));
        }
        let dev = dev_for_minimax_m3(self)?;
        if !graph.capturing {
            crate::lfm2_graph::write_embedding_to_buffer(
                &dev,
                &graph.buffers.token_ids_dev,
                token_id,
            )?;
        }

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

        let hd = self.cfg.head_dim;
        let nh = self.cfg.num_attention_heads;
        let nkv = self.cfg.num_key_value_heads;
        let h_shape = Shape::new(vec![batch, hidden]);

        for (i, layer) in self.layers.iter().enumerate() {
            let act = &graph.buffers.act_q81_buf[i];

            // 1. Attention pre-norm
            dev.rms_norm_into(
                &graph.buffers.layer_input[i],
                &**layer.input_layernorm.weight.storage(),
                layer.input_layernorm.eps,
                &graph.buffers.norm_buf[i],
                &h_shape,
            )
            .map_err(grim_core::error::Error::Tensor)?;
            let normed: &Storage = &graph.buffers.norm_buf[i];

            // 2. QKV projections
            linear_into(
                &dev,
                normed,
                layer.wq.weight(),
                &graph.buffers.q_buf[i],
                act,
            )?;
            linear_into(
                &dev,
                normed,
                layer.wk.weight(),
                &graph.buffers.k_buf[i],
                act,
            )?;
            linear_into(
                &dev,
                normed,
                layer.wv.weight(),
                &graph.buffers.v_buf[i],
                act,
            )?;

            // 3. RoPE
            let steps = 1usize;
            let rope_cfg = layer.rope.config.clone();
            let q3 = Shape::new(vec![batch, nh * steps, hd]);
            dev.rope_dev_base_into(
                &graph.buffers.q_buf[i],
                &graph.buffers.pos_dev,
                &graph.buffers.q_buf[i],
                &rope_cfg,
                &q3,
                nh,
                steps,
            )
            .map_err(grim_core::error::Error::Tensor)?;

            let k3 = Shape::new(vec![batch, nkv * steps, hd]);
            dev.rope_dev_base_into(
                &graph.buffers.k_buf[i],
                &graph.buffers.pos_dev,
                &graph.buffers.k_buf[i],
                &rope_cfg,
                &k3,
                nkv,
                steps,
            )
            .map_err(grim_core::error::Error::Tensor)?;

            // 4. KV append & attention
            let kv_stride = nkv * hd;
            let arena_slot_stride = graph.buffers.max_ctx * kv_stride;
            let max_ctx = graph.buffers.max_ctx;
            launch_qkv_gemv(
                &graph.buffers.k_arena[i],
                graph.buffers.current_pos,
                max_ctx,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("{e}")))?;
            launch_attention(
                &graph.buffers.k_arena[i],
                graph.buffers.current_pos,
                max_ctx,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("{e}")))?;

            grim_backend_rocm::launch_kv_append_batch(
                &dev,
                &graph.buffers.k_arena[i],
                &graph.buffers.k_buf[i],
                &graph.buffers.pos_dev,
                kv_stride,
                steps,
                batch,
                arena_slot_stride,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("kv_append k: {e}")))?;

            grim_backend_rocm::launch_kv_append_batch(
                &dev,
                &graph.buffers.v_arena[i],
                &graph.buffers.v_buf[i],
                &graph.buffers.pos_dev,
                kv_stride,
                steps,
                batch,
                arena_slot_stride,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("kv_append v: {e}")))?;

            grim_backend_rocm::launch_qkv_attention_dev_batch(
                &dev,
                &graph.buffers.q_buf[i],
                &graph.buffers.k_arena[i],
                &graph.buffers.v_arena[i],
                &graph.buffers.attn_out_buf[i],
                &graph.buffers.attn_max_buf[i],
                &graph.buffers.attn_sum_buf[i],
                &graph.buffers.pos_dev,
                nh as u32,
                nkv as u32,
                hd as u32,
                steps as u32,
                steps as u32,
                1.0 / (hd as f32).sqrt(),
                0,
                0.0,
                &graph.buffers.attn_dummy,
                0,
                0,
                &graph.buffers.attn_dummy,
                0,
                batch,
                arena_slot_stride,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("qkv_attention: {e}")))?;

            grim_backend_rocm::launch_bump_i32_slots(&dev, &graph.buffers.pos_dev, steps, batch)
                .map_err(|e| grim_core::error::Error::Backend(format!("bump: {e}")))?;

            linear_into(
                &dev,
                &graph.buffers.attn_out_buf[i],
                layer.wo.weight(),
                &graph.buffers.norm_buf[i],
                act,
            )?;
            add_graph(
                &graph.buffers.layer_input[i],
                &graph.buffers.norm_buf[i],
                &graph.buffers.layer_output[i],
                &dev,
            )?;

            // 5. Post-attention RMSNorm -> MoE FFN
            dev.rms_norm_into(
                &graph.buffers.layer_output[i],
                &**layer.post_attention_layernorm.weight.storage(),
                layer.post_attention_layernorm.eps,
                &graph.buffers.norm_buf[i],
                &h_shape,
            )
            .map_err(grim_core::error::Error::Tensor)?;
            let normed_moe: &Storage = &graph.buffers.norm_buf[i];

            // 6. MoE Gate + top-k route (mode 3 renorm)
            linear_into(
                &dev,
                normed_moe,
                layer.block_sparse_moe.gate.weight(),
                &graph.buffers.moe_gate_logits[i],
                act,
            )?;
            let num_exp = self.cfg.num_experts;
            let top_k = self.cfg.num_experts_per_tok.min(num_exp);
            dev.moe_route_topk_on_device(
                &graph.buffers.moe_gate_logits[i],
                None,
                &graph.buffers.moe_route_tokens,
                &graph.buffers.moe_route_experts,
                &graph.buffers.moe_route_weights,
                batch,
                num_exp,
                top_k,
                3, // mode 3: softmax renormalized over top-k
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("moe route: {e}")))?;

            // 7. Resident grouped dispatch
            let experts = layer
                .block_sparse_moe
                .experts
                .iter()
                .map(|e| crate::shared_moe::MoeExpert {
                    gate: e.w1.clone(),
                    up: e.w3.clone(),
                    down: e.w2.clone(),
                })
                .collect::<Vec<_>>();

            let (_, _, _, gate_buf, up_buf, down_buf) = crate::shared_moe::ensure_charon_scratch(
                dev.ordinal(),
                batch,
                top_k,
                &experts,
                &layer.block_sparse_moe.charon_cache,
            )?;

            let norm_rocm = dst_downcast(normed_moe)?;
            dev.moe_fused_dispatch_resident_routing_into(
                norm_rocm,
                gate_buf.as_ref(),
                up_buf.as_ref(),
                down_buf.as_ref(),
                &graph.buffers.moe_route_tokens,
                &graph.buffers.moe_route_experts,
                &graph.buffers.moe_route_weights,
                batch * top_k,
                &graph.buffers.moe_out[i],
                hidden,
                self.cfg.intermediate_size,
                1.0,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("moe dispatch: {e}")))?;

            add_graph(
                &graph.buffers.layer_output[i],
                &graph.buffers.moe_out[i],
                &graph.buffers.layer_output[i],
                &dev,
            )?;

            let n_layers = graph.buffers.layer_input.len();
            let dst: &Storage = if i + 1 < n_layers {
                &graph.buffers.layer_input[i + 1]
            } else {
                &graph.buffers.head_input
            };
            dev.copy_slice_into(dst, &graph.buffers.layer_output[i], 0, batch * hidden)
                .map_err(grim_core::error::Error::Tensor)?;
        }

        // Final RMSNorm + LM head
        dev.rms_norm_into(
            &graph.buffers.head_input,
            &**self.norm.weight.storage(),
            self.norm.eps,
            &graph.buffers.head_input,
            &h_shape,
        )
        .map_err(grim_core::error::Error::Tensor)?;

        linear_into(
            &dev,
            &graph.buffers.head_input,
            self.output.weight(),
            &graph.buffers.head_output,
            &graph.buffers.act_q81_buf[0],
        )?;

        Ok(())
    }

    fn forward_replay(&self, graph: &mut DecodeGraph, token_id: u32) -> Result<()> {
        if !graph.is_captured {
            return Err(grim_core::error::Error::Backend(
                "forward_replay before capture".into(),
            ));
        }
        let dev = dev_for_minimax_m3(self)?;
        crate::lfm2_graph::write_embedding_to_buffer(&dev, &graph.buffers.token_ids_dev, token_id)?;

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

    fn eager_kv_seed_sources<'a>(
        &self,
        session: &'a dyn grim_core::session::SessionT,
        valid_rows: u32,
    ) -> Result<Vec<Option<EagerKvSource<'a>>>> {
        let caches = session
            .model_state()
            .and_then(|s| {
                s.downcast_ref::<Vec<Option<(grim_tensor::Tensor, grim_tensor::Tensor)>>>()
            })
            .ok_or_else(|| {
                grim_core::error::Error::Session(
                    "missing or invalid MiniMaxM3 KV cache in session".into(),
                )
            })?;

        if caches.len() != self.layers.len() {
            return Err(grim_core::error::Error::Session(format!(
                "eager_kv_seed_sources: {} caches != {} layers",
                caches.len(),
                self.layers.len()
            )));
        }

        let mut out = Vec::with_capacity(self.layers.len());
        for cache in caches.iter() {
            let (k_st, v_st) = match cache {
                Some((k, v)) => (k, v),
                None => {
                    if valid_rows > 0 {
                        return Err(grim_core::error::Error::Session(
                            "eager_kv_seed_sources: missing layer cache".into(),
                        ));
                    }
                    out.push(None);
                    continue;
                }
            };

            let (k_rocm, v_rocm) = match (
                as_rocm(k_st.storage().as_ref()),
                as_rocm(v_st.storage().as_ref()),
            ) {
                (Ok(k), Ok(v)) => (k, v),
                _ => {
                    if valid_rows > 0 {
                        return Err(grim_core::error::Error::Session(
                            "eager_kv_seed_sources: KV caches not ROCm-resident".into(),
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
                        "eager_kv_seed_sources: zero-width KV cache".into(),
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
                            "eager_kv_seed_sources: KV caches have no device pointer".into(),
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
}

// ─── DecodeGraphModel implementation for Glm4MoeLite ──────────────────────

fn dev_for_glm4_moe_lite(m: &Glm4MoeLite) -> Result<Arc<Dev>> {
    match &m.device {
        Device::Rocm(o) => Ok(Dev::shared(*o)),
        _ => Err(grim_core::error::Error::Unimplemented(
            "decode graph needs ROCm device".into(),
        )),
    }
}

impl DecodeGraphModel for Glm4MoeLite {
    fn get_or_create_decode_graph(&self, max_ctx: usize, batch: usize) -> Result<DecodeGraph> {
        if !decode_graph_enabled() {
            return Err(grim_core::error::Error::Backend(
                "decode graph disabled by env".into(),
            ));
        }
        let dev = dev_for_glm4_moe_lite(self)?;
        let stream = dev
            .get_stream_from_pool(0)
            .ok_or_else(|| grim_core::error::Error::Backend("no stream in pool".into()))?;

        let hidden = self.cfg.hidden_size;
        let n_q = self.cfg.num_attention_heads * self.cfg.head_dim;
        let n_k = self.cfg.num_key_value_heads * self.cfg.head_dim;
        let n_v = n_k;
        let inter = self.cfg.intermediate_size;
        let vocab = self.cfg.vocab_size.max(1);
        let ctx = max_ctx.max(1);
        let nh = self.cfg.num_attention_heads;

        let buffers = DecodeGraphBuffers::allocate(
            &dev,
            self.layers.len(),
            hidden,
            n_q,
            n_k,
            n_v,
            inter,
            ctx,
            vocab,
            nh,
            batch,
            self.cfg.num_experts,
            self.cfg.num_experts_per_tok,
            0,
            0,
        )
        .map_err(|e| grim_core::error::Error::Backend(format!("graph pool alloc: {e}")))?;

        Ok(DecodeGraph::new(&dev, buffers, stream))
    }

    fn forward_capture(&self, graph: &mut DecodeGraph, token_id: u32) -> Result<()> {
        if !decode_graph_enabled() {
            return Err(grim_core::error::Error::Backend(
                "decode graph disabled".into(),
            ));
        }
        let dev = dev_for_glm4_moe_lite(self)?;
        if !graph.capturing {
            crate::lfm2_graph::write_embedding_to_buffer(
                &dev,
                &graph.buffers.token_ids_dev,
                token_id,
            )?;
        }

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

        let hd = self.cfg.head_dim;
        let nh = self.cfg.num_attention_heads;
        let nkv = self.cfg.num_key_value_heads;
        let h_shape = Shape::new(vec![batch, hidden]);

        for (i, layer) in self.layers.iter().enumerate() {
            let act = &graph.buffers.act_q81_buf[i];

            // 1. Attention pre-norm
            dev.rms_norm_into(
                &graph.buffers.layer_input[i],
                &**layer.input_layernorm.weight.storage(),
                layer.input_layernorm.eps,
                &graph.buffers.norm_buf[i],
                &h_shape,
            )
            .map_err(grim_core::error::Error::Tensor)?;
            let normed: &Storage = &graph.buffers.norm_buf[i];

            // 2. QKV projections
            linear_into(
                &dev,
                normed,
                layer.wq.weight(),
                &graph.buffers.q_buf[i],
                act,
            )?;
            linear_into(
                &dev,
                normed,
                layer.wk.weight(),
                &graph.buffers.k_buf[i],
                act,
            )?;
            linear_into(
                &dev,
                normed,
                layer.wv.weight(),
                &graph.buffers.v_buf[i],
                act,
            )?;

            // 3. RoPE
            let steps = 1usize;
            let rope_cfg = layer.rope.config.clone();
            let q3 = Shape::new(vec![batch, nh * steps, hd]);
            dev.rope_dev_base_into(
                &graph.buffers.q_buf[i],
                &graph.buffers.pos_dev,
                &graph.buffers.q_buf[i],
                &rope_cfg,
                &q3,
                nh,
                steps,
            )
            .map_err(grim_core::error::Error::Tensor)?;

            let k3 = Shape::new(vec![batch, nkv * steps, hd]);
            dev.rope_dev_base_into(
                &graph.buffers.k_buf[i],
                &graph.buffers.pos_dev,
                &graph.buffers.k_buf[i],
                &rope_cfg,
                &k3,
                nkv,
                steps,
            )
            .map_err(grim_core::error::Error::Tensor)?;

            // 4. KV append & attention
            let kv_stride = nkv * hd;
            let arena_slot_stride = graph.buffers.max_ctx * kv_stride;
            let max_ctx = graph.buffers.max_ctx;
            launch_qkv_gemv(
                &graph.buffers.k_arena[i],
                graph.buffers.current_pos,
                max_ctx,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("{e}")))?;
            launch_attention(
                &graph.buffers.k_arena[i],
                graph.buffers.current_pos,
                max_ctx,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("{e}")))?;

            grim_backend_rocm::launch_kv_append_batch(
                &dev,
                &graph.buffers.k_arena[i],
                &graph.buffers.k_buf[i],
                &graph.buffers.pos_dev,
                kv_stride,
                steps,
                batch,
                arena_slot_stride,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("kv_append k: {e}")))?;

            grim_backend_rocm::launch_kv_append_batch(
                &dev,
                &graph.buffers.v_arena[i],
                &graph.buffers.v_buf[i],
                &graph.buffers.pos_dev,
                kv_stride,
                steps,
                batch,
                arena_slot_stride,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("kv_append v: {e}")))?;

            grim_backend_rocm::launch_qkv_attention_dev_batch(
                &dev,
                &graph.buffers.q_buf[i],
                &graph.buffers.k_arena[i],
                &graph.buffers.v_arena[i],
                &graph.buffers.attn_out_buf[i],
                &graph.buffers.attn_max_buf[i],
                &graph.buffers.attn_sum_buf[i],
                &graph.buffers.pos_dev,
                nh as u32,
                nkv as u32,
                hd as u32,
                steps as u32,
                steps as u32,
                1.0 / (hd as f32).sqrt(),
                0,
                0.0,
                &graph.buffers.attn_dummy,
                0,
                0,
                &graph.buffers.attn_dummy,
                0,
                batch,
                arena_slot_stride,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("qkv_attention: {e}")))?;

            grim_backend_rocm::launch_bump_i32_slots(&dev, &graph.buffers.pos_dev, steps, batch)
                .map_err(|e| grim_core::error::Error::Backend(format!("bump: {e}")))?;

            linear_into(
                &dev,
                &graph.buffers.attn_out_buf[i],
                layer.wo.weight(),
                &graph.buffers.norm_buf[i],
                act,
            )?;
            add_graph(
                &graph.buffers.layer_input[i],
                &graph.buffers.norm_buf[i],
                &graph.buffers.layer_output[i],
                &dev,
            )?;

            // 5. Post-attention RMSNorm -> MoE FFN
            dev.rms_norm_into(
                &graph.buffers.layer_output[i],
                &**layer.post_attention_layernorm.weight.storage(),
                layer.post_attention_layernorm.eps,
                &graph.buffers.norm_buf[i],
                &h_shape,
            )
            .map_err(grim_core::error::Error::Tensor)?;
            let normed_moe: &Storage = &graph.buffers.norm_buf[i];

            // 6. MoE Gate + top-k route (mode 3 renorm)
            linear_into(
                &dev,
                normed_moe,
                layer.moe.gate.weight(),
                &graph.buffers.moe_gate_logits[i],
                act,
            )?;
            let num_exp = self.cfg.num_experts;
            let top_k = self.cfg.num_experts_per_tok.min(num_exp);
            dev.moe_route_topk_on_device(
                &graph.buffers.moe_gate_logits[i],
                None,
                &graph.buffers.moe_route_tokens,
                &graph.buffers.moe_route_experts,
                &graph.buffers.moe_route_weights,
                batch,
                num_exp,
                top_k,
                3, // mode 3: softmax renormalized over top-k
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("moe route: {e}")))?;

            // 7. Resident grouped dispatch
            let experts = layer
                .moe
                .experts
                .iter()
                .map(|e| crate::shared_moe::MoeExpert {
                    gate: e.gate_proj.clone(),
                    up: e.up_proj.clone(),
                    down: e.down_proj.clone(),
                })
                .collect::<Vec<_>>();

            let (_, _, _, gate_buf, up_buf, down_buf) = crate::shared_moe::ensure_charon_scratch(
                dev.ordinal(),
                batch,
                top_k,
                &experts,
                &layer.moe.charon_cache,
            )?;

            let norm_rocm = dst_downcast(normed_moe)?;
            dev.moe_fused_dispatch_resident_routing_into(
                norm_rocm,
                gate_buf.as_ref(),
                up_buf.as_ref(),
                down_buf.as_ref(),
                &graph.buffers.moe_route_tokens,
                &graph.buffers.moe_route_experts,
                &graph.buffers.moe_route_weights,
                batch * top_k,
                &graph.buffers.moe_out[i],
                hidden,
                self.cfg.intermediate_size,
                1.0,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("moe dispatch: {e}")))?;

            // If shared expert present, project through gate/up/down and accumulate
            if let Some(ref shared) = layer.moe.shared_expert {
                linear_into(
                    &dev,
                    normed_moe,
                    shared.gate_proj.weight(),
                    &graph.buffers.gate_buf[i],
                    act,
                )?;
                linear_into(
                    &dev,
                    normed_moe,
                    shared.up_proj.weight(),
                    &graph.buffers.up_buf[i],
                    act,
                )?;
                dev.silu_mul_into(
                    &graph.buffers.gate_buf[i],
                    &graph.buffers.up_buf[i],
                    &graph.buffers.activated_buf[i],
                )
                .map_err(grim_core::error::Error::Tensor)?;
                linear_into(
                    &dev,
                    &graph.buffers.activated_buf[i],
                    shared.down_proj.weight(),
                    &graph.buffers.norm_buf[i],
                    act,
                )?;
                add_graph(
                    &graph.buffers.moe_out[i],
                    &graph.buffers.norm_buf[i],
                    &graph.buffers.moe_out[i],
                    &dev,
                )?;
            }

            add_graph(
                &graph.buffers.layer_output[i],
                &graph.buffers.moe_out[i],
                &graph.buffers.layer_output[i],
                &dev,
            )?;

            let n_layers = graph.buffers.layer_input.len();
            let dst: &Storage = if i + 1 < n_layers {
                &graph.buffers.layer_input[i + 1]
            } else {
                &graph.buffers.head_input
            };
            dev.copy_slice_into(dst, &graph.buffers.layer_output[i], 0, batch * hidden)
                .map_err(grim_core::error::Error::Tensor)?;
        }

        // Final RMSNorm + LM head
        dev.rms_norm_into(
            &graph.buffers.head_input,
            &**self.norm.weight.storage(),
            self.norm.eps,
            &graph.buffers.head_input,
            &h_shape,
        )
        .map_err(grim_core::error::Error::Tensor)?;

        linear_into(
            &dev,
            &graph.buffers.head_input,
            self.output.weight(),
            &graph.buffers.head_output,
            &graph.buffers.act_q81_buf[0],
        )?;

        Ok(())
    }

    fn forward_replay(&self, graph: &mut DecodeGraph, token_id: u32) -> Result<()> {
        if !graph.is_captured {
            return Err(grim_core::error::Error::Backend(
                "forward_replay before capture".into(),
            ));
        }
        let dev = dev_for_glm4_moe_lite(self)?;
        crate::lfm2_graph::write_embedding_to_buffer(&dev, &graph.buffers.token_ids_dev, token_id)?;

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

    fn eager_kv_seed_sources<'a>(
        &self,
        _session: &'a dyn grim_core::session::SessionT,
        _valid_rows: u32,
    ) -> Result<Vec<Option<EagerKvSource<'a>>>> {
        Ok((0..self.layers.len()).map(|_| None).collect())
    }
}

// ─── DecodeGraphModel implementation for GraniteMoeHybrid ─────────────────

fn dev_for_granite_moe_hybrid(m: &GraniteMoeHybrid) -> Result<Arc<Dev>> {
    match &m.device {
        Device::Rocm(o) => Ok(Dev::shared(*o)),
        _ => Err(grim_core::error::Error::Unimplemented(
            "decode graph needs ROCm device".into(),
        )),
    }
}

impl DecodeGraphModel for GraniteMoeHybrid {
    fn get_or_create_decode_graph(&self, max_ctx: usize, batch: usize) -> Result<DecodeGraph> {
        if !decode_graph_enabled() {
            return Err(grim_core::error::Error::Backend(
                "decode graph disabled by env".into(),
            ));
        }
        let dev = dev_for_granite_moe_hybrid(self)?;
        let stream = dev
            .get_stream_from_pool(0)
            .ok_or_else(|| grim_core::error::Error::Backend("no stream in pool".into()))?;

        let hidden = self.cfg.hidden_size;
        let n_q = self.cfg.num_attention_heads * self.cfg.head_dim;
        let n_k = self.cfg.num_key_value_heads * self.cfg.head_dim;
        let n_v = n_k;
        let inter = self.cfg.intermediate_size;
        let vocab = self.cfg.vocab_size.max(1);
        let ctx = max_ctx.max(1);
        let nh = self.cfg.num_attention_heads;

        let buffers = DecodeGraphBuffers::allocate(
            &dev,
            self.layers.len(),
            hidden,
            n_q,
            n_k,
            n_v,
            inter,
            ctx,
            vocab,
            nh,
            batch,
            self.cfg.num_local_experts,
            self.cfg.num_experts_per_tok,
            0,
            0,
        )
        .map_err(|e| grim_core::error::Error::Backend(format!("graph pool alloc: {e}")))?;

        Ok(DecodeGraph::new(&dev, buffers, stream))
    }

    fn forward_capture(&self, graph: &mut DecodeGraph, token_id: u32) -> Result<()> {
        if !decode_graph_enabled() {
            return Err(grim_core::error::Error::Backend(
                "decode graph disabled".into(),
            ));
        }
        let dev = dev_for_granite_moe_hybrid(self)?;
        if !graph.capturing {
            crate::lfm2_graph::write_embedding_to_buffer(
                &dev,
                &graph.buffers.token_ids_dev,
                token_id,
            )?;
        }

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

        let hd = self.cfg.head_dim;
        let nh = self.cfg.num_attention_heads;
        let nkv = self.cfg.num_key_value_heads;
        let h_shape = Shape::new(vec![batch, hidden]);

        for (i, layer) in self.layers.iter().enumerate() {
            let act = &graph.buffers.act_q81_buf[i];

            // 1. Attention pre-norm
            dev.rms_norm_into(
                &graph.buffers.layer_input[i],
                &**layer.input_layernorm.weight.storage(),
                layer.input_layernorm.eps,
                &graph.buffers.norm_buf[i],
                &h_shape,
            )
            .map_err(grim_core::error::Error::Tensor)?;
            let normed: &Storage = &graph.buffers.norm_buf[i];

            // 2. QKV projections
            linear_into(
                &dev,
                normed,
                layer.wq.weight(),
                &graph.buffers.q_buf[i],
                act,
            )?;
            linear_into(
                &dev,
                normed,
                layer.wk.weight(),
                &graph.buffers.k_buf[i],
                act,
            )?;
            linear_into(
                &dev,
                normed,
                layer.wv.weight(),
                &graph.buffers.v_buf[i],
                act,
            )?;

            // 3. RoPE
            let steps = 1usize;
            let rope_cfg = layer.rope.config.clone();
            let q3 = Shape::new(vec![batch, nh * steps, hd]);
            dev.rope_dev_base_into(
                &graph.buffers.q_buf[i],
                &graph.buffers.pos_dev,
                &graph.buffers.q_buf[i],
                &rope_cfg,
                &q3,
                nh,
                steps,
            )
            .map_err(grim_core::error::Error::Tensor)?;

            let k3 = Shape::new(vec![batch, nkv * steps, hd]);
            dev.rope_dev_base_into(
                &graph.buffers.k_buf[i],
                &graph.buffers.pos_dev,
                &graph.buffers.k_buf[i],
                &rope_cfg,
                &k3,
                nkv,
                steps,
            )
            .map_err(grim_core::error::Error::Tensor)?;

            // 4. KV append & attention
            let kv_stride = nkv * hd;
            let arena_slot_stride = graph.buffers.max_ctx * kv_stride;
            let max_ctx = graph.buffers.max_ctx;
            launch_qkv_gemv(
                &graph.buffers.k_arena[i],
                graph.buffers.current_pos,
                max_ctx,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("{e}")))?;
            launch_attention(
                &graph.buffers.k_arena[i],
                graph.buffers.current_pos,
                max_ctx,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("{e}")))?;

            grim_backend_rocm::launch_kv_append_batch(
                &dev,
                &graph.buffers.k_arena[i],
                &graph.buffers.k_buf[i],
                &graph.buffers.pos_dev,
                kv_stride,
                steps,
                batch,
                arena_slot_stride,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("kv_append k: {e}")))?;

            grim_backend_rocm::launch_kv_append_batch(
                &dev,
                &graph.buffers.v_arena[i],
                &graph.buffers.v_buf[i],
                &graph.buffers.pos_dev,
                kv_stride,
                steps,
                batch,
                arena_slot_stride,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("kv_append v: {e}")))?;

            grim_backend_rocm::launch_qkv_attention_dev_batch(
                &dev,
                &graph.buffers.q_buf[i],
                &graph.buffers.k_arena[i],
                &graph.buffers.v_arena[i],
                &graph.buffers.attn_out_buf[i],
                &graph.buffers.attn_max_buf[i],
                &graph.buffers.attn_sum_buf[i],
                &graph.buffers.pos_dev,
                nh as u32,
                nkv as u32,
                hd as u32,
                steps as u32,
                steps as u32,
                1.0 / (hd as f32).sqrt(),
                0,
                0.0,
                &graph.buffers.attn_dummy,
                0,
                0,
                &graph.buffers.attn_dummy,
                0,
                batch,
                arena_slot_stride,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("qkv_attention: {e}")))?;

            grim_backend_rocm::launch_bump_i32_slots(&dev, &graph.buffers.pos_dev, steps, batch)
                .map_err(|e| grim_core::error::Error::Backend(format!("bump: {e}")))?;

            linear_into(
                &dev,
                &graph.buffers.attn_out_buf[i],
                layer.wo.weight(),
                &graph.buffers.norm_buf[i],
                act,
            )?;
            // Residual 1: out = in + residual_multiplier * attn
            axpy_graph(
                &graph.buffers.layer_input[i],
                layer.residual_multiplier,
                &graph.buffers.norm_buf[i],
                &graph.buffers.layer_output[i],
                &dev,
            )?;

            // 5. Post-attention RMSNorm -> MoE FFN
            dev.rms_norm_into(
                &graph.buffers.layer_output[i],
                &**layer.post_attention_layernorm.weight.storage(),
                layer.post_attention_layernorm.eps,
                &graph.buffers.norm_buf[i],
                &h_shape,
            )
            .map_err(grim_core::error::Error::Tensor)?;
            let normed_moe: &Storage = &graph.buffers.norm_buf[i];

            // 6. MoE Gate + top-k route (mode 3 renorm)
            linear_into(
                &dev,
                normed_moe,
                layer.moe.gate.weight(),
                &graph.buffers.moe_gate_logits[i],
                act,
            )?;
            let num_exp = self.cfg.num_local_experts;
            let top_k = self.cfg.num_experts_per_tok.min(num_exp);
            dev.moe_route_topk_on_device(
                &graph.buffers.moe_gate_logits[i],
                None,
                &graph.buffers.moe_route_tokens,
                &graph.buffers.moe_route_experts,
                &graph.buffers.moe_route_weights,
                batch,
                num_exp,
                top_k,
                3, // mode 3: softmax renormalized over top-k
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("moe route: {e}")))?;

            // 7. Resident grouped dispatch
            let experts = layer
                .moe
                .experts
                .iter()
                .map(|e| crate::shared_moe::MoeExpert {
                    gate: e.gate_proj.clone(),
                    up: e.up_proj.clone(),
                    down: e.down_proj.clone(),
                })
                .collect::<Vec<_>>();

            let (_, _, _, gate_buf, up_buf, down_buf) = crate::shared_moe::ensure_charon_scratch(
                dev.ordinal(),
                batch,
                top_k,
                &experts,
                &layer.moe.charon_cache,
            )?;

            let norm_rocm = dst_downcast(normed_moe)?;
            dev.moe_fused_dispatch_resident_routing_into(
                norm_rocm,
                gate_buf.as_ref(),
                up_buf.as_ref(),
                down_buf.as_ref(),
                &graph.buffers.moe_route_tokens,
                &graph.buffers.moe_route_experts,
                &graph.buffers.moe_route_weights,
                batch * top_k,
                &graph.buffers.moe_out[i],
                hidden,
                self.cfg.intermediate_size,
                1.0,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("moe dispatch: {e}")))?;

            // If shared expert present, project through gate/up/down and accumulate
            if let Some(ref shared) = layer.moe.shared_expert {
                linear_into(
                    &dev,
                    normed_moe,
                    shared.gate_proj.weight(),
                    &graph.buffers.gate_buf[i],
                    act,
                )?;
                linear_into(
                    &dev,
                    normed_moe,
                    shared.up_proj.weight(),
                    &graph.buffers.up_buf[i],
                    act,
                )?;
                dev.silu_mul_into(
                    &graph.buffers.gate_buf[i],
                    &graph.buffers.up_buf[i],
                    &graph.buffers.activated_buf[i],
                )
                .map_err(grim_core::error::Error::Tensor)?;
                linear_into(
                    &dev,
                    &graph.buffers.activated_buf[i],
                    shared.down_proj.weight(),
                    &graph.buffers.norm_buf[i],
                    act,
                )?;
                add_graph(
                    &graph.buffers.moe_out[i],
                    &graph.buffers.norm_buf[i],
                    &graph.buffers.moe_out[i],
                    &dev,
                )?;
            }

            // Residual 2: out = out + residual_multiplier * moe_out
            axpy_graph(
                &graph.buffers.layer_output[i],
                layer.residual_multiplier,
                &graph.buffers.moe_out[i],
                &graph.buffers.layer_output[i],
                &dev,
            )?;

            let n_layers = graph.buffers.layer_input.len();
            let dst: &Storage = if i + 1 < n_layers {
                &graph.buffers.layer_input[i + 1]
            } else {
                &graph.buffers.head_input
            };
            dev.copy_slice_into(dst, &graph.buffers.layer_output[i], 0, batch * hidden)
                .map_err(grim_core::error::Error::Tensor)?;
        }

        // Final RMSNorm + LM head
        dev.rms_norm_into(
            &graph.buffers.head_input,
            &**self.norm.weight.storage(),
            self.norm.eps,
            &graph.buffers.head_input,
            &h_shape,
        )
        .map_err(grim_core::error::Error::Tensor)?;

        linear_into(
            &dev,
            &graph.buffers.head_input,
            self.output.weight(),
            &graph.buffers.head_output,
            &graph.buffers.act_q81_buf[0],
        )?;

        Ok(())
    }

    fn forward_replay(&self, graph: &mut DecodeGraph, token_id: u32) -> Result<()> {
        if !graph.is_captured {
            return Err(grim_core::error::Error::Backend(
                "forward_replay before capture".into(),
            ));
        }
        let dev = dev_for_granite_moe_hybrid(self)?;
        crate::lfm2_graph::write_embedding_to_buffer(&dev, &graph.buffers.token_ids_dev, token_id)?;

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

    fn eager_kv_seed_sources<'a>(
        &self,
        _session: &'a dyn grim_core::session::SessionT,
        _valid_rows: u32,
    ) -> Result<Vec<Option<EagerKvSource<'a>>>> {
        Ok((0..self.layers.len()).map(|_| None).collect())
    }
}

// ─── DecodeGraphModel implementation for HyV3 ─────────────────────────────

fn dev_for_hyv3(m: &HyV3) -> Result<Arc<Dev>> {
    match &m.device {
        Device::Rocm(o) => Ok(Dev::shared(*o)),
        _ => Err(grim_core::error::Error::Unimplemented(
            "decode graph needs ROCm device".into(),
        )),
    }
}

impl DecodeGraphModel for HyV3 {
    fn get_or_create_decode_graph(&self, max_ctx: usize, batch: usize) -> Result<DecodeGraph> {
        if !decode_graph_enabled() {
            return Err(grim_core::error::Error::Backend(
                "decode graph disabled by env".into(),
            ));
        }
        let dev = dev_for_hyv3(self)?;
        let stream = dev
            .get_stream_from_pool(0)
            .ok_or_else(|| grim_core::error::Error::Backend("no stream in pool".into()))?;

        let hidden = self.cfg.hidden_size;
        let n_q = self.cfg.num_attention_heads * self.cfg.head_dim;
        let n_k = self.cfg.num_key_value_heads * self.cfg.head_dim;
        let n_v = n_k;
        let inter = self.cfg.intermediate_size;
        let vocab = self.cfg.vocab_size.max(1);
        let ctx = max_ctx.max(1);
        let nh = self.cfg.num_attention_heads;

        let buffers = DecodeGraphBuffers::allocate(
            &dev,
            self.layers.len(),
            hidden,
            n_q,
            n_k,
            n_v,
            inter,
            ctx,
            vocab,
            nh,
            batch,
            self.cfg.num_experts,
            self.cfg.num_experts_per_tok,
            0,
            0,
        )
        .map_err(|e| grim_core::error::Error::Backend(format!("graph pool alloc: {e}")))?;

        Ok(DecodeGraph::new(&dev, buffers, stream))
    }

    fn forward_capture(&self, graph: &mut DecodeGraph, token_id: u32) -> Result<()> {
        if !decode_graph_enabled() {
            return Err(grim_core::error::Error::Backend(
                "decode graph disabled".into(),
            ));
        }
        let dev = dev_for_hyv3(self)?;
        if !graph.capturing {
            crate::lfm2_graph::write_embedding_to_buffer(
                &dev,
                &graph.buffers.token_ids_dev,
                token_id,
            )?;
        }

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

        let hd = self.cfg.head_dim;
        let nh = self.cfg.num_attention_heads;
        let nkv = self.cfg.num_key_value_heads;
        let h_shape = Shape::new(vec![batch, hidden]);

        for (i, layer) in self.layers.iter().enumerate() {
            let act = &graph.buffers.act_q81_buf[i];

            // 1. Attention pre-norm
            dev.rms_norm_into(
                &graph.buffers.layer_input[i],
                &**layer.input_layernorm.weight.storage(),
                layer.input_layernorm.eps,
                &graph.buffers.norm_buf[i],
                &h_shape,
            )
            .map_err(grim_core::error::Error::Tensor)?;
            let normed: &Storage = &graph.buffers.norm_buf[i];

            // 2. QKV projections
            linear_into(
                &dev,
                normed,
                layer.wq.weight(),
                &graph.buffers.q_buf[i],
                act,
            )?;
            linear_into(
                &dev,
                normed,
                layer.wk.weight(),
                &graph.buffers.k_buf[i],
                act,
            )?;
            linear_into(
                &dev,
                normed,
                layer.wv.weight(),
                &graph.buffers.v_buf[i],
                act,
            )?;

            // 3. Per-head QK-norm
            let qn_shape = Shape::new(vec![batch * nh, hd]);
            dev.rms_norm_into(
                &graph.buffers.q_buf[i],
                &**layer.q_norm.weight.storage(),
                layer.q_norm.eps,
                &graph.buffers.q_buf[i],
                &qn_shape,
            )
            .map_err(grim_core::error::Error::Tensor)?;

            let kn_shape = Shape::new(vec![batch * nkv, hd]);
            dev.rms_norm_into(
                &graph.buffers.k_buf[i],
                &**layer.k_norm.weight.storage(),
                layer.k_norm.eps,
                &graph.buffers.k_buf[i],
                &kn_shape,
            )
            .map_err(grim_core::error::Error::Tensor)?;

            // 4. RoPE
            let steps = 1usize;
            let rope_cfg = layer.rope.config.clone();
            let q3 = Shape::new(vec![batch, nh * steps, hd]);
            dev.rope_dev_base_into(
                &graph.buffers.q_buf[i],
                &graph.buffers.pos_dev,
                &graph.buffers.q_buf[i],
                &rope_cfg,
                &q3,
                nh,
                steps,
            )
            .map_err(grim_core::error::Error::Tensor)?;

            let k3 = Shape::new(vec![batch, nkv * steps, hd]);
            dev.rope_dev_base_into(
                &graph.buffers.k_buf[i],
                &graph.buffers.pos_dev,
                &graph.buffers.k_buf[i],
                &rope_cfg,
                &k3,
                nkv,
                steps,
            )
            .map_err(grim_core::error::Error::Tensor)?;

            // 5. KV append & attention
            let kv_stride = nkv * hd;
            let arena_slot_stride = graph.buffers.max_ctx * kv_stride;
            let max_ctx = graph.buffers.max_ctx;
            launch_qkv_gemv(
                &graph.buffers.k_arena[i],
                graph.buffers.current_pos,
                max_ctx,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("{e}")))?;
            launch_attention(
                &graph.buffers.k_arena[i],
                graph.buffers.current_pos,
                max_ctx,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("{e}")))?;

            grim_backend_rocm::launch_kv_append_batch(
                &dev,
                &graph.buffers.k_arena[i],
                &graph.buffers.k_buf[i],
                &graph.buffers.pos_dev,
                kv_stride,
                steps,
                batch,
                arena_slot_stride,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("kv_append k: {e}")))?;

            grim_backend_rocm::launch_kv_append_batch(
                &dev,
                &graph.buffers.v_arena[i],
                &graph.buffers.v_buf[i],
                &graph.buffers.pos_dev,
                kv_stride,
                steps,
                batch,
                arena_slot_stride,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("kv_append v: {e}")))?;

            grim_backend_rocm::launch_qkv_attention_dev_batch(
                &dev,
                &graph.buffers.q_buf[i],
                &graph.buffers.k_arena[i],
                &graph.buffers.v_arena[i],
                &graph.buffers.attn_out_buf[i],
                &graph.buffers.attn_max_buf[i],
                &graph.buffers.attn_sum_buf[i],
                &graph.buffers.pos_dev,
                nh as u32,
                nkv as u32,
                hd as u32,
                steps as u32,
                steps as u32,
                1.0 / (hd as f32).sqrt(),
                0,
                0.0,
                &graph.buffers.attn_dummy,
                0,
                0,
                &graph.buffers.attn_dummy,
                0,
                batch,
                arena_slot_stride,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("qkv_attention: {e}")))?;

            grim_backend_rocm::launch_bump_i32_slots(&dev, &graph.buffers.pos_dev, steps, batch)
                .map_err(|e| grim_core::error::Error::Backend(format!("bump: {e}")))?;

            linear_into(
                &dev,
                &graph.buffers.attn_out_buf[i],
                layer.wo.weight(),
                &graph.buffers.norm_buf[i],
                act,
            )?;
            add_graph(
                &graph.buffers.layer_input[i],
                &graph.buffers.norm_buf[i],
                &graph.buffers.layer_output[i],
                &dev,
            )?;

            // 6. Post-attention RMSNorm -> MoE FFN
            dev.rms_norm_into(
                &graph.buffers.layer_output[i],
                &**layer.post_attention_layernorm.weight.storage(),
                layer.post_attention_layernorm.eps,
                &graph.buffers.norm_buf[i],
                &h_shape,
            )
            .map_err(grim_core::error::Error::Tensor)?;
            let normed_moe: &Storage = &graph.buffers.norm_buf[i];

            // 7. MoE Gate + top-k route (mode 3 renorm)
            linear_into(
                &dev,
                normed_moe,
                layer.moe.gate.weight(),
                &graph.buffers.moe_gate_logits[i],
                act,
            )?;
            let num_exp = self.cfg.num_experts;
            let top_k = self.cfg.num_experts_per_tok.min(num_exp);
            dev.moe_route_topk_on_device(
                &graph.buffers.moe_gate_logits[i],
                None,
                &graph.buffers.moe_route_tokens,
                &graph.buffers.moe_route_experts,
                &graph.buffers.moe_route_weights,
                batch,
                num_exp,
                top_k,
                3, // mode 3: softmax renormalized over top-k
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("moe route: {e}")))?;

            // 8. Resident grouped dispatch
            let experts = layer
                .moe
                .experts
                .iter()
                .map(|e| crate::shared_moe::MoeExpert {
                    gate: e.gate_proj.clone(),
                    up: e.up_proj.clone(),
                    down: e.down_proj.clone(),
                })
                .collect::<Vec<_>>();

            let (_, _, _, gate_buf, up_buf, down_buf) = crate::shared_moe::ensure_charon_scratch(
                dev.ordinal(),
                batch,
                top_k,
                &experts,
                &layer.moe.charon_cache,
            )?;

            let norm_rocm = dst_downcast(normed_moe)?;
            dev.moe_fused_dispatch_resident_routing_into(
                norm_rocm,
                gate_buf.as_ref(),
                up_buf.as_ref(),
                down_buf.as_ref(),
                &graph.buffers.moe_route_tokens,
                &graph.buffers.moe_route_experts,
                &graph.buffers.moe_route_weights,
                batch * top_k,
                &graph.buffers.moe_out[i],
                hidden,
                self.cfg.intermediate_size,
                1.0,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("moe dispatch: {e}")))?;

            // If shared expert present, project through gate/up/down and accumulate
            if let Some(ref shared) = layer.moe.shared_expert {
                linear_into(
                    &dev,
                    normed_moe,
                    shared.gate_proj.weight(),
                    &graph.buffers.gate_buf[i],
                    act,
                )?;
                linear_into(
                    &dev,
                    normed_moe,
                    shared.up_proj.weight(),
                    &graph.buffers.up_buf[i],
                    act,
                )?;
                dev.silu_mul_into(
                    &graph.buffers.gate_buf[i],
                    &graph.buffers.up_buf[i],
                    &graph.buffers.activated_buf[i],
                )
                .map_err(grim_core::error::Error::Tensor)?;
                linear_into(
                    &dev,
                    &graph.buffers.activated_buf[i],
                    shared.down_proj.weight(),
                    &graph.buffers.norm_buf[i],
                    act,
                )?;
                add_graph(
                    &graph.buffers.moe_out[i],
                    &graph.buffers.norm_buf[i],
                    &graph.buffers.moe_out[i],
                    &dev,
                )?;
            }

            add_graph(
                &graph.buffers.layer_output[i],
                &graph.buffers.moe_out[i],
                &graph.buffers.layer_output[i],
                &dev,
            )?;

            let n_layers = graph.buffers.layer_input.len();
            let dst: &Storage = if i + 1 < n_layers {
                &graph.buffers.layer_input[i + 1]
            } else {
                &graph.buffers.head_input
            };
            dev.copy_slice_into(dst, &graph.buffers.layer_output[i], 0, batch * hidden)
                .map_err(grim_core::error::Error::Tensor)?;
        }

        // Final RMSNorm + LM head
        dev.rms_norm_into(
            &graph.buffers.head_input,
            &**self.norm.weight.storage(),
            self.norm.eps,
            &graph.buffers.head_input,
            &h_shape,
        )
        .map_err(grim_core::error::Error::Tensor)?;

        linear_into(
            &dev,
            &graph.buffers.head_input,
            self.output.weight(),
            &graph.buffers.head_output,
            &graph.buffers.act_q81_buf[0],
        )?;

        Ok(())
    }

    fn forward_replay(&self, graph: &mut DecodeGraph, token_id: u32) -> Result<()> {
        if !graph.is_captured {
            return Err(grim_core::error::Error::Backend(
                "forward_replay before capture".into(),
            ));
        }
        let dev = dev_for_hyv3(self)?;
        crate::lfm2_graph::write_embedding_to_buffer(&dev, &graph.buffers.token_ids_dev, token_id)?;

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

    fn eager_kv_seed_sources<'a>(
        &self,
        _session: &'a dyn grim_core::session::SessionT,
        _valid_rows: u32,
    ) -> Result<Vec<Option<EagerKvSource<'a>>>> {
        Ok((0..self.layers.len()).map(|_| None).collect())
    }
}

// ─── DecodeGraphModel implementation for Chameleon ────────────────────────

fn dev_for_chameleon(m: &Chameleon) -> Result<Arc<Dev>> {
    match &m.device {
        Device::Rocm(o) => Ok(Dev::shared(*o)),
        _ => Err(grim_core::error::Error::Unimplemented(
            "decode graph needs ROCm device".into(),
        )),
    }
}

impl DecodeGraphModel for Chameleon {
    fn get_or_create_decode_graph(&self, max_ctx: usize, batch: usize) -> Result<DecodeGraph> {
        if !decode_graph_enabled() {
            return Err(grim_core::error::Error::Backend(
                "decode graph disabled by env".into(),
            ));
        }
        let dev = dev_for_chameleon(self)?;
        let stream = dev
            .get_stream_from_pool(0)
            .ok_or_else(|| grim_core::error::Error::Backend("no stream in pool".into()))?;

        let hidden = self.cfg.hidden_size;
        let n_q = self.cfg.num_heads * self.cfg.head_dim;
        let n_k = self.cfg.num_kv_heads * self.cfg.head_dim;
        let n_v = n_k;
        let inter = self.cfg.intermediate_size;
        let vocab = self.cfg.vocab_size.max(1);
        let ctx = max_ctx.max(1);
        let nh = self.cfg.num_heads;

        let buffers = DecodeGraphBuffers::allocate(
            &dev,
            self.layers.len(),
            hidden,
            n_q,
            n_k,
            n_v,
            inter,
            ctx,
            vocab,
            nh,
            batch,
            0,
            0,
            0,
            0,
        )
        .map_err(|e| grim_core::error::Error::Backend(format!("graph pool alloc: {e}")))?;

        Ok(DecodeGraph::new(&dev, buffers, stream))
    }

    fn forward_capture(&self, graph: &mut DecodeGraph, token_id: u32) -> Result<()> {
        if !decode_graph_enabled() {
            return Err(grim_core::error::Error::Backend(
                "decode graph disabled".into(),
            ));
        }
        let dev = dev_for_chameleon(self)?;
        if !graph.capturing {
            crate::lfm2_graph::write_embedding_to_buffer(
                &dev,
                &graph.buffers.token_ids_dev,
                token_id,
            )?;
        }

        // Embedding gather
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

        // Forward all layers
        for (i, layer) in self.layers.iter().enumerate() {
            layer.forward_graph(i, &graph.buffers, &dev)?;
        }

        // Final norm + output head
        let h_shape = graph.buffers.head_input.shape().clone();
        dev.rms_norm_into(
            &graph.buffers.head_input,
            &**self.norm.weight.storage(),
            self.norm.eps,
            &graph.buffers.head_input,
            &h_shape,
        )
        .map_err(grim_core::error::Error::Tensor)?;

        linear_into(
            &dev,
            &graph.buffers.head_input,
            self.output.weight(),
            &graph.buffers.head_output,
            &graph.buffers.act_q81_buf[0],
        )?;

        Ok(())
    }

    fn forward_replay(&self, graph: &mut DecodeGraph, token_id: u32) -> Result<()> {
        if !graph.is_captured {
            return Err(grim_core::error::Error::Backend(
                "forward_replay before capture".into(),
            ));
        }
        let dev = dev_for_chameleon(self)?;
        crate::lfm2_graph::write_embedding_to_buffer(&dev, &graph.buffers.token_ids_dev, token_id)?;

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

    fn eager_kv_seed_sources<'a>(
        &self,
        session: &'a dyn grim_core::session::SessionT,
        valid_rows: u32,
    ) -> Result<Vec<Option<EagerKvSource<'a>>>> {
        // Chameleon eager forward stores full-history (k, v) tensors per layer.
        let caches = session
            .model_state()
            .and_then(|s| {
                s.downcast_ref::<Vec<Option<(grim_tensor::Tensor, grim_tensor::Tensor)>>>()
            })
            .ok_or_else(|| {
                grim_core::error::Error::Session(
                    "missing or invalid Chameleon KV cache in session".into(),
                )
            })?;

        if caches.len() != self.layers.len() {
            return Err(grim_core::error::Error::Session(format!(
                "eager_kv_seed_sources: {} caches != {} layers",
                caches.len(),
                self.layers.len()
            )));
        }

        let mut out = Vec::with_capacity(self.layers.len());
        for cache in caches.iter() {
            let (k_st, v_st) = match cache {
                Some((k, v)) => (k, v),
                None => {
                    if valid_rows > 0 {
                        return Err(grim_core::error::Error::Session(
                            "eager_kv_seed_sources: missing layer cache".into(),
                        ));
                    }
                    out.push(None);
                    continue;
                }
            };

            let (k_rocm, v_rocm) = match (
                as_rocm(k_st.storage().as_ref()),
                as_rocm(v_st.storage().as_ref()),
            ) {
                (Ok(k), Ok(v)) => (k, v),
                _ => {
                    if valid_rows > 0 {
                        return Err(grim_core::error::Error::Session(
                            "eager_kv_seed_sources: KV caches not ROCm-resident".into(),
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
                        "eager_kv_seed_sources: zero-width KV cache".into(),
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
                            "eager_kv_seed_sources: KV caches have no device pointer".into(),
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
}

// ─── DecodeGraphModel implementation for DeepSeek4 ────────────────────────

fn dev_for_deepseek4(ds: &DeepSeek4) -> Result<Arc<Dev>> {
    match &ds.device {
        Device::Rocm(o) => Ok(Dev::shared(*o)),
        _ => Err(grim_core::error::Error::Unimplemented(
            "decode graph needs ROCm device".into(),
        )),
    }
}

impl DecodeGraphModel for DeepSeek4 {
    fn get_or_create_decode_graph(&self, max_ctx: usize, batch: usize) -> Result<DecodeGraph> {
        if !decode_graph_enabled() {
            return Err(grim_core::error::Error::Backend(
                "decode graph disabled by env".into(),
            ));
        }
        let dev = dev_for_deepseek4(self)?;
        let stream = dev
            .get_stream_from_pool(0)
            .ok_or_else(|| grim_core::error::Error::Backend("no stream in pool".into()))?;

        let hidden = self.cfg.hidden_size;
        let n_q = self.cfg.num_heads * (self.cfg.qk_nope_head_dim + self.cfg.qk_rope_head_dim);
        let n_k = self.cfg.num_kv_heads * self.cfg.head_dim;
        let n_v = n_k;
        let inter = self.cfg.intermediate_size;
        let vocab = self.cfg.vocab_size.max(1);
        let ctx = max_ctx.max(1);
        let nh = self.cfg.num_heads;
        let latent_dim = self.cfg.kv_lora_rank + self.cfg.qk_rope_head_dim;

        let buffers = DecodeGraphBuffers::allocate_with_mla(
            &dev,
            self.layers.len(),
            hidden,
            n_q,
            n_k,
            n_v,
            inter,
            ctx,
            vocab,
            nh,
            batch,
            self.cfg.n_routed_experts,
            self.cfg.num_experts_per_tok,
            0,
            0,
            latent_dim,
        )
        .map_err(|e| grim_core::error::Error::Backend(format!("graph pool alloc: {e}")))?;

        Ok(DecodeGraph::new(&dev, buffers, stream))
    }

    fn forward_capture(&self, graph: &mut DecodeGraph, token_id: u32) -> Result<()> {
        if !decode_graph_enabled() {
            return Err(grim_core::error::Error::Backend(
                "decode graph disabled".into(),
            ));
        }
        let dev = dev_for_deepseek4(self)?;
        if !graph.capturing {
            crate::lfm2_graph::write_embedding_to_buffer(
                &dev,
                &graph.buffers.token_ids_dev,
                token_id,
            )?;
        }

        // Embedding gather
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

        let h_shape = Shape::new(vec![batch, hidden]);

        // Forward each block
        for (i, layer) in self.layers.iter().enumerate() {
            // 1. Attention RMS norm into norm_buf
            dev.rms_norm_into(
                &graph.buffers.layer_input[i],
                &**layer.attn_norm.weight.storage(),
                layer.attn_norm.eps,
                &graph.buffers.norm_buf[i],
                &h_shape,
            )
            .map_err(grim_core::error::Error::Tensor)?;

            let normed_attn: &Storage = &graph.buffers.norm_buf[i];
            let act = &graph.buffers.act_q81_buf[i];

            // 2. MLA Attention decode via launch_mla_absorbed_decode
            let rank = self.cfg.kv_lora_rank;
            let nope = layer.self_attn.qk_nope_head_dim;
            let rope_d = layer.self_attn.qk_rope_head_dim;
            let vd = layer.self_attn.v_head_dim;
            let nh = layer.self_attn.num_heads;

            // Project Q: norm_buf -> q_buf
            if let (Some(qa), Some(qb)) = (&layer.self_attn.q_a_proj, &layer.self_attn.q_b_proj) {
                linear_into(
                    &dev,
                    normed_attn,
                    qa.weight(),
                    &graph.buffers.gate_buf[i],
                    act,
                )?;
                linear_into(
                    &dev,
                    &graph.buffers.gate_buf[i],
                    qb.weight(),
                    &graph.buffers.q_buf[i],
                    act,
                )?;
            } else if let Some(ref q_direct) = layer.self_attn.q_proj_direct {
                linear_into(
                    &dev,
                    normed_attn,
                    q_direct.weight(),
                    &graph.buffers.q_buf[i],
                    act,
                )?;
            }

            // Project KV: norm_buf -> gate_up_buf (staged latent)
            linear_into(
                &dev,
                normed_attn,
                layer.self_attn.kv_a_proj.weight(),
                &graph.buffers.gate_up_buf[i],
                act,
            )?;

            // Append to latent_kv_arena
            let latent_row = rank + rope_d;
            let arena_slot_stride = graph.buffers.max_ctx * latent_row;
            grim_backend_rocm::launch_kv_append_batch(
                &dev,
                &graph.buffers.latent_kv_arena[i],
                &graph.buffers.gate_up_buf[i],
                &graph.buffers.pos_dev,
                latent_row,
                1,
                batch,
                arena_slot_stride,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("latent kv_append: {e}")))?;

            // Multi-head absorbed decode launch
            let w_src = dst_downcast(layer.self_attn.kv_b_proj.weight.storage().as_ref())?;
            let qa = &graph.buffers.q_buf[i];
            let out_attn = &graph.buffers.attn_out_buf[i];
            let kv_arena = &graph.buffers.latent_kv_arena[i];

            dev.launch_mla_absorbed_decode(
                qa,
                qa,
                kv_arena,
                Some(w_src),
                out_attn,
                nh,
                rank,
                rope_d,
                vd,
                graph.buffers.max_ctx.max(1),
                nope * rank,
                (nope + vd) * rank,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("mla_absorbed_decode: {e}")))?;

            // Project Attention Out: attn_out_buf -> norm_buf
            linear_into(
                &dev,
                &graph.buffers.attn_out_buf[i],
                layer.self_attn.o_proj.weight(),
                &graph.buffers.norm_buf[i],
                act,
            )?;

            // Residual 1: layer_input + attn_out -> layer_output
            add_graph(
                &graph.buffers.layer_input[i],
                &graph.buffers.norm_buf[i],
                &graph.buffers.layer_output[i],
                &dev,
            )?;

            // 3. FFN branch
            dev.rms_norm_into(
                &graph.buffers.layer_output[i],
                &**layer.ffn_norm.weight.storage(),
                layer.ffn_norm.eps,
                &graph.buffers.norm_buf[i],
                &h_shape,
            )
            .map_err(grim_core::error::Error::Tensor)?;

            let normed_ffn: &Storage = &graph.buffers.norm_buf[i];
            if let Some(ref mlp) = layer.mlp {
                linear_into(
                    &dev,
                    normed_ffn,
                    mlp.w1.weight(),
                    &graph.buffers.gate_buf[i],
                    act,
                )?;
                linear_into(
                    &dev,
                    normed_ffn,
                    mlp.w3.weight(),
                    &graph.buffers.up_buf[i],
                    act,
                )?;
                dev.silu_mul_into(
                    &graph.buffers.gate_buf[i],
                    &graph.buffers.up_buf[i],
                    &graph.buffers.activated_buf[i],
                )
                .map_err(grim_core::error::Error::Tensor)?;
                linear_into(
                    &dev,
                    &graph.buffers.activated_buf[i],
                    mlp.w2.weight(),
                    &graph.buffers.norm_buf[i],
                    act,
                )?;
            }

            // Residual 2: layer_output + ffn_out -> layer_output
            add_graph(
                &graph.buffers.layer_output[i],
                &graph.buffers.norm_buf[i],
                &graph.buffers.layer_output[i],
                &dev,
            )?;

            // Publish to next layer or head_input
            let n_layers = graph.buffers.layer_input.len();
            let dst: &Storage = if i + 1 < n_layers {
                &graph.buffers.layer_input[i + 1]
            } else {
                &graph.buffers.head_input
            };
            publish_into(&dev, dst, &graph.buffers.layer_output[i])?;
        }

        // Final norm + output head
        dev.rms_norm_into(
            &graph.buffers.head_input,
            &**self.norm.weight.storage(),
            self.norm.eps,
            &graph.buffers.head_input,
            &h_shape,
        )
        .map_err(grim_core::error::Error::Tensor)?;

        linear_into(
            &dev,
            &graph.buffers.head_input,
            self.output.weight(),
            &graph.buffers.head_output,
            &graph.buffers.act_q81_buf[0],
        )?;

        Ok(())
    }

    fn forward_replay(&self, graph: &mut DecodeGraph, token_id: u32) -> Result<()> {
        if !graph.is_captured {
            return Err(grim_core::error::Error::Backend(
                "forward_replay before capture".into(),
            ));
        }
        let dev = dev_for_deepseek4(self)?;
        crate::lfm2_graph::write_embedding_to_buffer(&dev, &graph.buffers.token_ids_dev, token_id)?;

        let pos = graph.buffers.current_pos;
        graph
            .buffers
            .write_pos_async(&dev, pos, graph.stream)
            .map_err(|e| grim_core::error::Error::Backend(format!("write pos: {e}")))?;
        graph
            .replay()
            .map_err(|e| grim_core::error::Error::Backend(format!("replay: {e}")))?;
        Ok(())
    }

    fn eager_kv_seed_sources<'a>(
        &self,
        _session: &'a dyn grim_core::session::SessionT,
        _valid_rows: u32,
    ) -> Result<Vec<Option<EagerKvSource<'a>>>> {
        Ok((0..self.layers.len()).map(|_| None).collect())
    }
}

// ─── DecodeGraphModel implementation for DeepSeek2 ────────────────────────

fn dev_for_deepseek2(ds: &DeepSeek2) -> Result<Arc<Dev>> {
    match &ds.device {
        Device::Rocm(o) => Ok(Dev::shared(*o)),
        _ => Err(grim_core::error::Error::Unimplemented(
            "decode graph needs ROCm device".into(),
        )),
    }
}

impl DecodeGraphModel for DeepSeek2 {
    fn get_or_create_decode_graph(&self, max_ctx: usize, batch: usize) -> Result<DecodeGraph> {
        if !decode_graph_enabled() {
            return Err(grim_core::error::Error::Backend(
                "decode graph disabled by env".into(),
            ));
        }
        let dev = dev_for_deepseek2(self)?;
        let stream = dev
            .get_stream_from_pool(0)
            .ok_or_else(|| grim_core::error::Error::Backend("no stream in pool".into()))?;

        let hidden = self.cfg.hidden_size;
        let n_q = self.cfg.num_heads * (self.cfg.qk_nope_head_dim + self.cfg.qk_rope_head_dim);
        let n_k = self.cfg.num_kv_heads * self.cfg.head_dim;
        let n_v = n_k;
        let inter = self.cfg.intermediate_size;
        let vocab = self.cfg.vocab_size.max(1);
        let ctx = max_ctx.max(1);
        let nh = self.cfg.num_heads;
        let latent_dim = self.cfg.kv_lora_rank + self.cfg.qk_rope_head_dim;

        let buffers = DecodeGraphBuffers::allocate_with_mla(
            &dev,
            self.layers.len(),
            hidden,
            n_q,
            n_k,
            n_v,
            inter,
            ctx,
            vocab,
            nh,
            batch,
            self.cfg.n_routed_experts,
            self.cfg.num_experts_per_tok,
            0,
            0,
            latent_dim,
        )
        .map_err(|e| grim_core::error::Error::Backend(format!("graph pool alloc: {e}")))?;

        Ok(DecodeGraph::new(&dev, buffers, stream))
    }

    fn forward_capture(&self, graph: &mut DecodeGraph, token_id: u32) -> Result<()> {
        if !decode_graph_enabled() {
            return Err(grim_core::error::Error::Backend(
                "decode graph disabled".into(),
            ));
        }
        let dev = dev_for_deepseek2(self)?;
        if !graph.capturing {
            crate::lfm2_graph::write_embedding_to_buffer(
                &dev,
                &graph.buffers.token_ids_dev,
                token_id,
            )?;
        }

        // Embedding gather
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

        let h_shape = Shape::new(vec![batch, hidden]);

        // Forward each block
        for (i, layer) in self.layers.iter().enumerate() {
            // 1. Attention RMS norm into norm_buf
            dev.rms_norm_into(
                &graph.buffers.layer_input[i],
                &**layer.attn_norm.weight.storage(),
                layer.attn_norm.eps,
                &graph.buffers.norm_buf[i],
                &h_shape,
            )
            .map_err(grim_core::error::Error::Tensor)?;

            let normed_attn: &Storage = &graph.buffers.norm_buf[i];
            let act = &graph.buffers.act_q81_buf[i];

            // 2. MLA Attention decode
            let rank = self.cfg.kv_lora_rank;
            let nope = layer.self_attn.qk_nope_head_dim;
            let rope_d = layer.self_attn.qk_rope_head_dim;
            let vd = layer.self_attn.v_head_dim;
            let nh = layer.self_attn.num_heads;

            // Project Q: norm_buf -> q_buf
            linear_into(
                &dev,
                normed_attn,
                layer.self_attn.q_proj.weight(),
                &graph.buffers.q_buf[i],
                act,
            )?;

            // Project KV: norm_buf -> gate_up_buf (staged latent)
            linear_into(
                &dev,
                normed_attn,
                layer.self_attn.kv_a_proj.weight(),
                &graph.buffers.gate_up_buf[i],
                act,
            )?;

            // Append to latent_kv_arena
            let latent_row = rank + rope_d;
            let arena_slot_stride = graph.buffers.max_ctx * latent_row;
            grim_backend_rocm::launch_kv_append_batch(
                &dev,
                &graph.buffers.latent_kv_arena[i],
                &graph.buffers.gate_up_buf[i],
                &graph.buffers.pos_dev,
                latent_row,
                1,
                batch,
                arena_slot_stride,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("latent kv_append: {e}")))?;

            // Multi-head absorbed decode launch
            let w_src = dst_downcast(layer.self_attn.kv_b_proj.weight.storage().as_ref())?;
            let qa = &graph.buffers.q_buf[i];
            let out_attn = &graph.buffers.attn_out_buf[i];
            let kv_arena = &graph.buffers.latent_kv_arena[i];

            dev.launch_mla_absorbed_decode(
                qa,
                qa,
                kv_arena,
                Some(w_src),
                out_attn,
                nh,
                rank,
                rope_d,
                vd,
                graph.buffers.max_ctx.max(1),
                nope * rank,
                (nope + vd) * rank,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("mla_absorbed_decode: {e}")))?;

            // Project Attention Out: attn_out_buf -> norm_buf
            linear_into(
                &dev,
                &graph.buffers.attn_out_buf[i],
                layer.self_attn.o_proj.weight(),
                &graph.buffers.norm_buf[i],
                act,
            )?;

            // Residual 1: layer_input + attn_out -> layer_output
            add_graph(
                &graph.buffers.layer_input[i],
                &graph.buffers.norm_buf[i],
                &graph.buffers.layer_output[i],
                &dev,
            )?;

            // 3. FFN branch
            dev.rms_norm_into(
                &graph.buffers.layer_output[i],
                &**layer.ffn_norm.weight.storage(),
                layer.ffn_norm.eps,
                &graph.buffers.norm_buf[i],
                &h_shape,
            )
            .map_err(grim_core::error::Error::Tensor)?;

            let normed_ffn: &Storage = &graph.buffers.norm_buf[i];
            if let Some(ref mlp) = layer.mlp {
                linear_into(
                    &dev,
                    normed_ffn,
                    mlp.w1.weight(),
                    &graph.buffers.gate_buf[i],
                    act,
                )?;
                linear_into(
                    &dev,
                    normed_ffn,
                    mlp.w3.weight(),
                    &graph.buffers.up_buf[i],
                    act,
                )?;
                dev.silu_mul_into(
                    &graph.buffers.gate_buf[i],
                    &graph.buffers.up_buf[i],
                    &graph.buffers.activated_buf[i],
                )
                .map_err(grim_core::error::Error::Tensor)?;
                linear_into(
                    &dev,
                    &graph.buffers.activated_buf[i],
                    mlp.w2.weight(),
                    &graph.buffers.norm_buf[i],
                    act,
                )?;
            }

            // Residual 2: layer_output + ffn_out -> layer_output
            add_graph(
                &graph.buffers.layer_output[i],
                &graph.buffers.norm_buf[i],
                &graph.buffers.layer_output[i],
                &dev,
            )?;

            // Publish to next layer or head_input
            let n_layers = graph.buffers.layer_input.len();
            let dst: &Storage = if i + 1 < n_layers {
                &graph.buffers.layer_input[i + 1]
            } else {
                &graph.buffers.head_input
            };
            publish_into(&dev, dst, &graph.buffers.layer_output[i])?;
        }

        // Final norm + output head
        dev.rms_norm_into(
            &graph.buffers.head_input,
            &**self.norm.weight.storage(),
            self.norm.eps,
            &graph.buffers.head_input,
            &h_shape,
        )
        .map_err(grim_core::error::Error::Tensor)?;

        linear_into(
            &dev,
            &graph.buffers.head_input,
            self.output.weight(),
            &graph.buffers.head_output,
            &graph.buffers.act_q81_buf[0],
        )?;

        Ok(())
    }

    fn forward_replay(&self, graph: &mut DecodeGraph, token_id: u32) -> Result<()> {
        if !graph.is_captured {
            return Err(grim_core::error::Error::Backend(
                "forward_replay before capture".into(),
            ));
        }
        let dev = dev_for_deepseek2(self)?;
        crate::lfm2_graph::write_embedding_to_buffer(&dev, &graph.buffers.token_ids_dev, token_id)?;

        let pos = graph.buffers.current_pos;
        graph
            .buffers
            .write_pos_async(&dev, pos, graph.stream)
            .map_err(|e| grim_core::error::Error::Backend(format!("write pos: {e}")))?;
        graph
            .replay()
            .map_err(|e| grim_core::error::Error::Backend(format!("replay: {e}")))?;
        Ok(())
    }

    fn eager_kv_seed_sources<'a>(
        &self,
        _session: &'a dyn grim_core::session::SessionT,
        _valid_rows: u32,
    ) -> Result<Vec<Option<EagerKvSource<'a>>>> {
        Ok((0..self.layers.len()).map(|_| None).collect())
    }
}

// ─── DecodeGraphModel implementation for DeepSeek32 ───────────────────────

fn dev_for_deepseek32(ds: &DeepSeek32) -> Result<Arc<Dev>> {
    match &ds.device {
        Device::Rocm(o) => Ok(Dev::shared(*o)),
        _ => Err(grim_core::error::Error::Unimplemented(
            "decode graph needs ROCm device".into(),
        )),
    }
}

impl DecodeGraphModel for DeepSeek32 {
    fn get_or_create_decode_graph(&self, max_ctx: usize, batch: usize) -> Result<DecodeGraph> {
        if !decode_graph_enabled() {
            return Err(grim_core::error::Error::Backend(
                "decode graph disabled by env".into(),
            ));
        }
        let dev = dev_for_deepseek32(self)?;
        let stream = dev
            .get_stream_from_pool(0)
            .ok_or_else(|| grim_core::error::Error::Backend("no stream in pool".into()))?;

        let hidden = self.cfg.hidden_size;
        let n_q = self.cfg.num_heads * (self.cfg.qk_nope_head_dim + self.cfg.qk_rope_head_dim);
        let n_k = self.cfg.num_kv_heads * self.cfg.head_dim;
        let n_v = n_k;
        let inter = self.cfg.intermediate_size;
        let vocab = self.cfg.vocab_size.max(1);
        let ctx = max_ctx.max(1);
        let nh = self.cfg.num_heads;
        let latent_dim = self.cfg.kv_lora_rank + self.cfg.qk_rope_head_dim;

        let buffers = DecodeGraphBuffers::allocate_with_mla(
            &dev,
            self.layers.len(),
            hidden,
            n_q,
            n_k,
            n_v,
            inter,
            ctx,
            vocab,
            nh,
            batch,
            self.cfg.n_routed_experts,
            self.cfg.num_experts_per_tok,
            0,
            0,
            latent_dim,
        )
        .map_err(|e| grim_core::error::Error::Backend(format!("graph pool alloc: {e}")))?;

        Ok(DecodeGraph::new(&dev, buffers, stream))
    }

    fn forward_capture(&self, graph: &mut DecodeGraph, token_id: u32) -> Result<()> {
        if !decode_graph_enabled() {
            return Err(grim_core::error::Error::Backend(
                "decode graph disabled".into(),
            ));
        }
        let dev = dev_for_deepseek32(self)?;
        if !graph.capturing {
            crate::lfm2_graph::write_embedding_to_buffer(
                &dev,
                &graph.buffers.token_ids_dev,
                token_id,
            )?;
        }

        // Embedding gather
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

        let h_shape = Shape::new(vec![batch, hidden]);

        // Forward each block
        for (i, layer) in self.layers.iter().enumerate() {
            // 1. Attention RMS norm into norm_buf
            dev.rms_norm_into(
                &graph.buffers.layer_input[i],
                &**layer.attn_norm.weight.storage(),
                layer.attn_norm.eps,
                &graph.buffers.norm_buf[i],
                &h_shape,
            )
            .map_err(grim_core::error::Error::Tensor)?;

            let normed_attn: &Storage = &graph.buffers.norm_buf[i];
            let act = &graph.buffers.act_q81_buf[i];

            // 2. MLA Attention decode
            let rank = self.cfg.kv_lora_rank;
            let nope = layer.self_attn.qk_nope_head_dim;
            let rope_d = layer.self_attn.qk_rope_head_dim;
            let vd = layer.self_attn.v_head_dim;
            let nh = layer.self_attn.num_heads;

            // Project Q: norm_buf -> q_buf
            if let (Some(qa), Some(qb)) = (&layer.self_attn.q_a_proj, &layer.self_attn.q_b_proj) {
                linear_into(
                    &dev,
                    normed_attn,
                    qa.weight(),
                    &graph.buffers.gate_buf[i],
                    act,
                )?;
                linear_into(
                    &dev,
                    &graph.buffers.gate_buf[i],
                    qb.weight(),
                    &graph.buffers.q_buf[i],
                    act,
                )?;
            } else if let Some(ref q_direct) = layer.self_attn.q_proj_direct {
                linear_into(
                    &dev,
                    normed_attn,
                    q_direct.weight(),
                    &graph.buffers.q_buf[i],
                    act,
                )?;
            }

            // Project KV: norm_buf -> gate_up_buf (staged latent)
            linear_into(
                &dev,
                normed_attn,
                layer.self_attn.kv_a_proj.weight(),
                &graph.buffers.gate_up_buf[i],
                act,
            )?;

            // Append to latent_kv_arena
            let latent_row = rank + rope_d;
            let arena_slot_stride = graph.buffers.max_ctx * latent_row;
            grim_backend_rocm::launch_kv_append_batch(
                &dev,
                &graph.buffers.latent_kv_arena[i],
                &graph.buffers.gate_up_buf[i],
                &graph.buffers.pos_dev,
                latent_row,
                1,
                batch,
                arena_slot_stride,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("latent kv_append: {e}")))?;

            // Multi-head absorbed decode launch
            let w_src = dst_downcast(layer.self_attn.kv_b_proj.weight.storage().as_ref())?;
            let qa = &graph.buffers.q_buf[i];
            let out_attn = &graph.buffers.attn_out_buf[i];
            let kv_arena = &graph.buffers.latent_kv_arena[i];

            dev.launch_mla_absorbed_decode(
                qa,
                qa,
                kv_arena,
                Some(w_src),
                out_attn,
                nh,
                rank,
                rope_d,
                vd,
                graph.buffers.max_ctx.max(1),
                nope * rank,
                (nope + vd) * rank,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("mla_absorbed_decode: {e}")))?;

            // Project Attention Out: attn_out_buf -> norm_buf
            linear_into(
                &dev,
                &graph.buffers.attn_out_buf[i],
                layer.self_attn.o_proj.weight(),
                &graph.buffers.norm_buf[i],
                act,
            )?;

            // Residual 1: layer_input + attn_out -> layer_output
            add_graph(
                &graph.buffers.layer_input[i],
                &graph.buffers.norm_buf[i],
                &graph.buffers.layer_output[i],
                &dev,
            )?;

            // 3. FFN branch
            dev.rms_norm_into(
                &graph.buffers.layer_output[i],
                &**layer.ffn_norm.weight.storage(),
                layer.ffn_norm.eps,
                &graph.buffers.norm_buf[i],
                &h_shape,
            )
            .map_err(grim_core::error::Error::Tensor)?;

            let normed_ffn: &Storage = &graph.buffers.norm_buf[i];
            if let Some(ref mlp) = layer.mlp {
                linear_into(
                    &dev,
                    normed_ffn,
                    mlp.w1.weight(),
                    &graph.buffers.gate_buf[i],
                    act,
                )?;
                linear_into(
                    &dev,
                    normed_ffn,
                    mlp.w3.weight(),
                    &graph.buffers.up_buf[i],
                    act,
                )?;
                dev.silu_mul_into(
                    &graph.buffers.gate_buf[i],
                    &graph.buffers.up_buf[i],
                    &graph.buffers.activated_buf[i],
                )
                .map_err(grim_core::error::Error::Tensor)?;
                linear_into(
                    &dev,
                    &graph.buffers.activated_buf[i],
                    mlp.w2.weight(),
                    &graph.buffers.norm_buf[i],
                    act,
                )?;
            }

            // Residual 2: layer_output + ffn_out -> layer_output
            add_graph(
                &graph.buffers.layer_output[i],
                &graph.buffers.norm_buf[i],
                &graph.buffers.layer_output[i],
                &dev,
            )?;

            // Publish to next layer or head_input
            let n_layers = graph.buffers.layer_input.len();
            let dst: &Storage = if i + 1 < n_layers {
                &graph.buffers.layer_input[i + 1]
            } else {
                &graph.buffers.head_input
            };
            publish_into(&dev, dst, &graph.buffers.layer_output[i])?;
        }

        // Final norm + output head
        dev.rms_norm_into(
            &graph.buffers.head_input,
            &**self.norm.weight.storage(),
            self.norm.eps,
            &graph.buffers.head_input,
            &h_shape,
        )
        .map_err(grim_core::error::Error::Tensor)?;

        linear_into(
            &dev,
            &graph.buffers.head_input,
            self.output.weight(),
            &graph.buffers.head_output,
            &graph.buffers.act_q81_buf[0],
        )?;

        Ok(())
    }

    fn forward_replay(&self, graph: &mut DecodeGraph, token_id: u32) -> Result<()> {
        if !graph.is_captured {
            return Err(grim_core::error::Error::Backend(
                "forward_replay before capture".into(),
            ));
        }
        let dev = dev_for_deepseek32(self)?;
        crate::lfm2_graph::write_embedding_to_buffer(&dev, &graph.buffers.token_ids_dev, token_id)?;

        let pos = graph.buffers.current_pos;
        graph
            .buffers
            .write_pos_async(&dev, pos, graph.stream)
            .map_err(|e| grim_core::error::Error::Backend(format!("write pos: {e}")))?;
        graph
            .replay()
            .map_err(|e| grim_core::error::Error::Backend(format!("replay: {e}")))?;
        Ok(())
    }

    fn eager_kv_seed_sources<'a>(
        &self,
        _session: &'a dyn grim_core::session::SessionT,
        _valid_rows: u32,
    ) -> Result<Vec<Option<EagerKvSource<'a>>>> {
        Ok((0..self.layers.len()).map(|_| None).collect())
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::LlamaConfigRefs;
    use crate::moe_block::MoeBlock;
    use crate::qwen35::Qwen35Config;
    use grim_backend_rocm::RocmDevice;
    use grim_core::model::CausalLm;
    use grim_core::session::Inner as SessionInner;
    use grim_nn::moe::{ExpertBank, MoeFfn, MoeRouter, RouterKind};
    use grim_nn::{
        ColumnParallelLinear, Embedding, Linear, RmsNorm, Rope, RowParallelLinear,
        TensorParallelConfig,
    };
    use grim_tensor::{ArithType, CoreTensorOps, DType, QuantProvenance, Shape, Storage, Tensor};

    fn rocm_tensor(dev: &RocmDevice, ordinal: usize, data: Vec<f32>, shape: Shape) -> Tensor {
        let storage = dev.from_cpu(&data, &shape, DType::F32).unwrap();
        Tensor::new(
            Arc::from(storage),
            shape,
            DType::F32,
            QuantProvenance::GrimNative,
            Device::Rocm(ordinal),
        )
    }

    fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.2
            })
            .collect()
    }

    fn test_linear(
        dev: &RocmDevice,
        ordinal: usize,
        out_dim: usize,
        in_dim: usize,
        seed: u64,
    ) -> Linear {
        let w = rocm_tensor(
            dev,
            ordinal,
            rand_vec(out_dim * in_dim, seed),
            Shape::new(vec![out_dim, in_dim]),
        );
        Linear {
            weight: w.clone(),
            bias: None,
            w_t: w,
            quant_format: None,
        }
    }

    fn test_linear_q80(
        dev: &RocmDevice,
        ordinal: usize,
        out_dim: usize,
        in_dim: usize,
        seed: u64,
    ) -> Linear {
        let bytes = grim_quant::quant_q80(&rand_vec(out_dim * in_dim, seed)).unwrap();
        let dtype = DType {
            arith: ArithType::F32,
            storage: Storage::KQuant(grim_tensor::dtype::KQuantScheme::Q80),
        };
        let shape = Shape::new(vec![out_dim, in_dim]);
        let storage = dev.from_cpu_bytes(&bytes, &shape, dtype.clone()).unwrap();
        let w = Tensor::new(
            Arc::from(storage),
            shape,
            dtype,
            QuantProvenance::GrimNative,
            Device::Rocm(ordinal),
        );
        Linear {
            weight: w.clone(),
            bias: None,
            w_t: w,
            quant_format: None,
        }
    }

    fn test_norm(dev: &RocmDevice, ordinal: usize, dim: usize) -> RmsNorm {
        let ones = vec![1.0f32; dim];
        let w = rocm_tensor(dev, ordinal, ones, Shape::new(vec![dim]));
        RmsNorm {
            weight: w,
            eps: 1e-5,
        }
    }

    fn assert_graph_matches_eager_single_token<M: DecodeGraphModel>(
        model: &M,
        dev: &RocmDevice,
        token_id: u32,
        eager_logits: &[f32],
        label: &str,
    ) {
        let mut graph = model
            .get_or_create_decode_graph(128, 1)
            .unwrap_or_else(|e| panic!("{label}: graph allocation failed: {e}"));
        model
            .forward_capture(&mut graph, token_id)
            .unwrap_or_else(|e| panic!("{label}: warmup 1 failed: {e}"));
        model
            .forward_capture(&mut graph, token_id)
            .unwrap_or_else(|e| panic!("{label}: warmup 2 failed: {e}"));
        graph
            .begin_capture()
            .unwrap_or_else(|e| panic!("{label}: begin capture failed: {e}"));
        model
            .forward_capture(&mut graph, token_id)
            .unwrap_or_else(|e| panic!("{label}: capture failed: {e}"));
        graph
            .end_capture()
            .unwrap_or_else(|e| panic!("{label}: end capture failed: {e}"));
        model
            .forward_replay(&mut graph, token_id)
            .unwrap_or_else(|e| panic!("{label}: replay failed: {e}"));
        dev.synchronize();

        let storage = graph.logits_device_storage();
        let rocm = storage
            .as_any()
            .downcast_ref::<grim_backend_rocm::RocmStorage>()
            .unwrap_or_else(|| panic!("{label}: logits storage is not ROCm"));
        let host_bytes = rocm.copy_to_host().expect("copy graph logits");
        assert_eq!(
            host_bytes.len(),
            eager_logits.len() * 4,
            "{label}: graph/eager logit length mismatch"
        );
        let graph_logits = unsafe {
            std::slice::from_raw_parts(host_bytes.as_ptr() as *const f32, eager_logits.len())
        };
        let max_rel = graph_logits
            .iter()
            .zip(eager_logits)
            .map(|(got, want)| (got - want).abs() / (want.abs() + 1e-3))
            .fold(0.0f32, f32::max);
        assert!(
            max_rel < 5e-2,
            "{label}: graph/eager max relative difference {max_rel:.6}"
        );
    }

    fn make_test_llama(
        dev: &RocmDevice,
        ordinal: usize,
        q80: bool,
    ) -> (Llama, crate::model::LlamaConfig) {
        let vocab_size = 64;
        let hidden_size = 64;
        let intermediate_size = 128;
        let num_heads = 4;
        let num_kv_heads = 2; // GQA (2:1)
        let head_dim = hidden_size / num_heads;
        let num_layers = 2;

        let cfg = crate::model::LlamaConfig {
            vocab_size,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            num_layers,
            intermediate_size,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 512,
            partial_rotary_factor: 1.0,
            yarn: None,
        };

        let embed = rocm_tensor(
            dev,
            ordinal,
            rand_vec(vocab_size * hidden_size, 42),
            Shape::new(vec![vocab_size, hidden_size]),
        );

        let mk_lin = |out_d, in_d, seed| {
            if q80 {
                test_linear_q80(dev, ordinal, out_d, in_d, seed)
            } else {
                test_linear(dev, ordinal, out_d, in_d, seed)
            }
        };

        let tp = TensorParallelConfig::default();
        let cfg_refs = LlamaConfigRefs {
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            tp_world_size: 1,
            local_num_heads: num_heads,
            local_num_kv_heads: num_kv_heads,
            kv_head_replica_factor: 1,
            sliding_window: None,
        };

        let mut layers = Vec::new();
        for l in 0..num_layers {
            let seed = (l as u64) * 1000 + 10;
            let q_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;

            let wq = mk_lin(q_dim, hidden_size, seed + 1);
            let wk = mk_lin(kv_dim, hidden_size, seed + 2);
            let wv = mk_lin(kv_dim, hidden_size, seed + 3);
            let wo = mk_lin(hidden_size, q_dim, seed + 4);
            let w_gate = mk_lin(intermediate_size, hidden_size, seed + 5);
            let w_up = mk_lin(intermediate_size, hidden_size, seed + 6);
            let w_down = mk_lin(hidden_size, intermediate_size, seed + 7);

            let (wqkv_q80_fused, w_gate_up_q80_fused) = if q80 {
                let fqkv = dev
                    .build_fused_qkv_q80(
                        wq.weight.storage().as_ref(),
                        wk.weight.storage().as_ref(),
                        wv.weight.storage().as_ref(),
                    )
                    .ok()
                    .map(Arc::new);
                let fgu = dev
                    .build_fused_gate_up_q80(
                        w_gate.weight.storage().as_ref(),
                        w_up.weight.storage().as_ref(),
                    )
                    .ok()
                    .map(Arc::new);
                (fqkv, fgu)
            } else {
                (None, None)
            };

            layers.push(LlamaBlock {
                attn_norm: test_norm(dev, ordinal, hidden_size),
                wq: ColumnParallelLinear::new(wq, tp),
                wk: ColumnParallelLinear::new(wk, tp),
                wv: ColumnParallelLinear::new(wv, tp),
                wo: RowParallelLinear::new(wo, tp),
                g_proj: None,
                q_norm: None,
                k_norm: None,
                ffn_norm: test_norm(dev, ordinal, hidden_size),
                w_gate: Some(ColumnParallelLinear::new(w_gate, tp)),
                w_up: Some(ColumnParallelLinear::new(w_up, tp)),
                w_down: Some(RowParallelLinear::new(w_down, tp)),
                rope: Rope::from_config(RopeConfig::new(head_dim, cfg.rope_theta)),
                tp_config: tp,
                _dev: Device::Rocm(ordinal),
                _cfg: cfg_refs,
                alibi_slopes: None,
                wqkv_q80_fused,
                w_gate_up_q80_fused,
                ffn_disabled: false,
                silu_q81_scratch: Arc::new(std::sync::Mutex::new(None)),
            });
        }

        let norm = test_norm(dev, ordinal, hidden_size);
        let output = mk_lin(vocab_size, hidden_size, 1001);

        let mut moe_blocks = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            moe_blocks.push(None);
        }

        let model = Llama {
            cfg: cfg.clone(),
            device: Device::Rocm(ordinal),
            tok_embeddings: Embedding { weight: embed },
            layers,
            moe_blocks,
            norm,
            output,
            layer_devices: vec![Device::Rocm(ordinal); num_layers],
            boundary_moves: std::sync::atomic::AtomicUsize::new(0),
        };

        (model, cfg)
    }

    #[test]
    fn test_llama_decode_graph_capture_replay() {
        if !grim_backend_rocm::device::util::gpu_test_enabled() {
            eprintln!("skip: set GRIM_GPU_TEST=1 for GPU graph test");
            return;
        }
        let _gpu_guard = grim_backend_rocm::device::util::gpu_test_lock();
        let dev = RocmDevice::shared(0);
        unsafe {
            std::env::set_var("GRIM_DECODE_GRAPH", "1");
        }

        let (llama, cfg) = make_test_llama(&dev, 0, false);

        // 1. Allocate decode graph buffers
        let mut graph = llama
            .get_or_create_decode_graph(512, 1)
            .expect("alloc graph");

        // Warmup before capture (JIT + allocator)
        let token_id = 5u32;
        llama
            .forward_capture(&mut graph, token_id)
            .expect("warmup 1");
        llama
            .forward_capture(&mut graph, token_id)
            .expect("warmup 2");

        // 2. Capture a step
        graph.begin_capture().expect("begin capture");
        llama
            .forward_capture(&mut graph, token_id)
            .expect("forward capture");
        graph.end_capture().expect("end capture");

        // 3. Replay with a token
        llama
            .forward_replay(&mut graph, token_id)
            .expect("replay token");
        dev.synchronize();

        // 4. Read back logits from graph storage
        let logits_storage = graph.logits_device_storage();
        let rocm_st = logits_storage
            .as_any()
            .downcast_ref::<grim_backend_rocm::RocmStorage>()
            .expect("RocmStorage");
        let host_bytes = rocm_st.copy_to_host().expect("copy_to_host");
        let host_logits: &[f32] = unsafe {
            std::slice::from_raw_parts(host_bytes.as_ptr() as *const f32, host_bytes.len() / 4)
        };

        assert_eq!(host_logits.len(), cfg.vocab_size);
        for (i, &val) in host_logits.iter().enumerate() {
            assert!(val.is_finite(), "logit {i} is not finite: {val}");
        }

        // 5. Test KV seeding from session
        let mut session = SessionInner::new(Device::Rocm(0));
        let prompt_tokens = vec![1.0f32, 2.0, 3.0];
        let prompt_tensor = rocm_tensor(&dev, 0, prompt_tokens, Shape::new(vec![1, 3]));
        let pos_tensor = rocm_tensor(&dev, 0, vec![0.0f32, 1.0, 2.0], Shape::new(vec![3]));
        let _ = llama.forward(&mut session, &prompt_tensor, &pos_tensor, &[]);

        let seed_sources = llama
            .eager_kv_seed_sources(&session, 3)
            .expect("eager_kv_seed_sources");
        assert_eq!(seed_sources.len(), cfg.num_layers);
        for (i, src) in seed_sources.iter().enumerate() {
            assert!(
                src.is_some(),
                "layer {i} should have valid eager KV seed source"
            );
        }

        // Seed into graph buffers
        graph
            .buffers
            .seed_kv_arena_from_eager(&dev, &seed_sources)
            .expect("seed_kv_arena_from_eager");
        assert_eq!(graph.buffers.current_pos, 3);
    }

    #[test]
    fn test_llama_decode_graph_q80_capture_replay() {
        if !grim_backend_rocm::device::util::gpu_test_enabled() {
            eprintln!("skip: set GRIM_GPU_TEST=1 for GPU graph test");
            return;
        }
        let _gpu_guard = grim_backend_rocm::device::util::gpu_test_lock();
        let dev = RocmDevice::shared(0);
        unsafe {
            std::env::set_var("GRIM_DECODE_GRAPH", "1");
        }

        let (llama, cfg) = make_test_llama(&dev, 0, true);

        let mut graph = llama
            .get_or_create_decode_graph(512, 1)
            .expect("alloc graph");

        let token_id = 7u32;
        // Warmup before capture
        llama
            .forward_capture(&mut graph, token_id)
            .expect("warmup 1");
        llama
            .forward_capture(&mut graph, token_id)
            .expect("warmup 2");

        graph.begin_capture().expect("begin capture");
        llama
            .forward_capture(&mut graph, token_id)
            .expect("forward capture");
        graph.end_capture().expect("end capture");

        llama
            .forward_replay(&mut graph, token_id)
            .expect("replay token");
        dev.synchronize();

        let logits_storage = graph.logits_device_storage();
        let rocm_st = logits_storage
            .as_any()
            .downcast_ref::<grim_backend_rocm::RocmStorage>()
            .expect("RocmStorage");
        let host_bytes = rocm_st.copy_to_host().expect("copy_to_host");
        let host_logits: &[f32] = unsafe {
            std::slice::from_raw_parts(host_bytes.as_ptr() as *const f32, host_bytes.len() / 4)
        };

        assert_eq!(host_logits.len(), cfg.vocab_size);
        for (i, &val) in host_logits.iter().enumerate() {
            assert!(val.is_finite(), "Q8_0 logit {i} is not finite: {val}");
        }
    }

    #[test]
    fn test_qwen35_decode_graph_capture_replay() {
        if !grim_backend_rocm::device::util::gpu_test_enabled() {
            eprintln!("skip: set GRIM_GPU_TEST=1 for GPU graph test");
            return;
        }
        let _gpu_guard = grim_backend_rocm::device::util::gpu_test_lock();
        let dev = RocmDevice::shared(0);
        unsafe {
            std::env::set_var("GRIM_DECODE_GRAPH", "1");
        }

        let vocab_size = 64;
        let hidden_size = 64;
        let intermediate_size = 128;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 16;
        let num_layers = 2;

        let cfg = Qwen35Config {
            vocab_size,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            num_layers,
            intermediate_size,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 512,
            full_attention_interval: 2, // layer 0 recurrent, layer 1 full attention
            ssm_d_state: 16,
            ssm_d_inner: 64,
            ssm_d_conv: 4,
            ssm_dt_rank: 8,
            ssm_n_group: 2,
            devices: Vec::new(),
        };

        let embed = rocm_tensor(
            &dev,
            0,
            rand_vec(vocab_size * hidden_size, 42),
            Shape::new(vec![vocab_size, hidden_size]),
        );

        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let mut blocks = Vec::new();

        for l in 0..num_layers {
            let is_full = (l + 1) % cfg.full_attention_interval == 0;
            let seed = (l as u64) * 1000 + 10;
            let (wq, wk, wv, wo, attn_qkv, ssm_out, ssm_conv1d) = if is_full {
                (
                    Some(test_linear(&dev, 0, q_dim, hidden_size, seed + 1)),
                    Some(test_linear(&dev, 0, kv_dim, hidden_size, seed + 2)),
                    Some(test_linear(&dev, 0, kv_dim, hidden_size, seed + 3)),
                    Some(test_linear(&dev, 0, hidden_size, q_dim, seed + 4)),
                    None,
                    None,
                    None,
                )
            } else {
                let conv_shape = Shape::new(vec![q_dim, cfg.ssm_d_conv]);
                let conv_storage = dev
                    .from_cpu(
                        &vec![0.25f32; q_dim * cfg.ssm_d_conv],
                        &conv_shape,
                        DType::F32,
                    )
                    .unwrap();
                let conv_t = Tensor::new(
                    Arc::from(conv_storage),
                    conv_shape,
                    DType::F32,
                    QuantProvenance::GrimNative,
                    Device::Rocm(0),
                );
                (
                    None,
                    None,
                    None,
                    None,
                    Some(test_linear(&dev, 0, q_dim, hidden_size, seed + 1)),
                    Some(test_linear(&dev, 0, hidden_size, q_dim, seed + 4)),
                    Some(conv_t),
                )
            };

            let block = Qwen35Block {
                device: Device::Rocm(0),
                attn_norm: test_norm(&dev, 0, hidden_size),
                wq,
                wk,
                wv,
                wo,
                attn_q_norm: None,
                attn_k_norm: None,
                attn_qkv,
                attn_gate: None,
                ssm_out,
                ssm_conv1d,
                ssm_conv_vec: None,
                ssm_a: None,
                ssm_alpha: None,
                ssm_beta: None,
                ssm_dt_bias: None,
                ssm_norm: None,
                post_attention_norm: test_norm(&dev, 0, hidden_size),
                ffn_gate: test_linear(&dev, 0, intermediate_size, hidden_size, seed + 5),
                ffn_up: test_linear(&dev, 0, intermediate_size, hidden_size, seed + 6),
                ffn_down: test_linear(&dev, 0, hidden_size, intermediate_size, seed + 7),
                is_full_attention: is_full,
                layer_idx: l,
                num_heads,
                num_kv_heads,
                head_dim,
                rope_theta: cfg.rope_theta,
                hidden_size,
                intermediate_size,
                wqkv_q80_fused: None,
                w_gate_up_q4k_fused: None,
            };
            blocks.push(block);
        }

        let qwen = Qwen35 {
            cfg: cfg.clone(),
            device: Device::Rocm(0),
            tok_embeddings: Embedding { weight: embed },
            blocks,
            output_norm: test_norm(&dev, 0, hidden_size),
            output: test_linear(&dev, 0, vocab_size, hidden_size, 999),
        };

        let mut graph = qwen
            .get_or_create_decode_graph(512, 1)
            .expect("alloc graph");

        let token_id = 11u32;
        // Warmup before capture
        qwen.forward_capture(&mut graph, token_id)
            .expect("warmup 1");
        qwen.forward_capture(&mut graph, token_id)
            .expect("warmup 2");

        graph.begin_capture().expect("begin capture");
        qwen.forward_capture(&mut graph, token_id)
            .expect("forward capture");
        graph.end_capture().expect("end capture");

        qwen.forward_replay(&mut graph, token_id)
            .expect("replay token");
        dev.synchronize();

        let logits_storage = graph.logits_device_storage();
        let rocm_st = logits_storage
            .as_any()
            .downcast_ref::<grim_backend_rocm::RocmStorage>()
            .expect("RocmStorage");
        let host_bytes = rocm_st.copy_to_host().expect("copy_to_host");
        let host_logits: &[f32] = unsafe {
            std::slice::from_raw_parts(host_bytes.as_ptr() as *const f32, host_bytes.len() / 4)
        };

        assert_eq!(host_logits.len(), cfg.vocab_size);
        for (i, &val) in host_logits.iter().enumerate() {
            assert!(val.is_finite(), "Qwen3.5 logit {i} is not finite: {val}");
        }

        // Test eager_kv_seed_sources with a session
        let session = qwen.new_session();
        let seed_sources = qwen
            .eager_kv_seed_sources(session.as_ref(), 0)
            .expect("seed sources");
        assert_eq!(seed_sources.len(), num_layers);
        // Layer 0 is recurrent -> None
        assert!(seed_sources[0].is_none());
        // Layer 1 is full attention, but empty cache with valid_rows=0 -> None
        assert!(seed_sources[1].is_none());
    }

    #[test]
    fn test_gemma2_decode_graph_capture_replay() {
        if !grim_backend_rocm::device::util::gpu_test_enabled() {
            eprintln!("skip: set GRIM_GPU_TEST=1 for GPU graph test");
            return;
        }
        let _gpu_guard = grim_backend_rocm::device::util::gpu_test_lock();
        let dev = RocmDevice::shared(0);
        unsafe {
            std::env::set_var("GRIM_DECODE_GRAPH", "1");
        }

        let vocab_size = 64;
        let hidden_size = 64;
        let intermediate_size = 128;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 16;
        let num_layers = 2;

        let cfg = crate::gemma2::Gemma2Config {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers: num_layers,
            num_attention_heads: num_heads,
            num_key_value_heads: num_kv_heads,
            head_dim,
            attn_logit_softcapping: Some(50.0),
            final_logit_softcapping: Some(30.0),
            sliding_window: None,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_position_embeddings: 512,
            yarn: None,
        };

        let embed = rocm_tensor(
            &dev,
            0,
            rand_vec(vocab_size * hidden_size, 42),
            Shape::new(vec![vocab_size, hidden_size]),
        );

        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let mut layers = Vec::new();

        for l in 0..num_layers {
            let seed = (l as u64) * 1000 + 20;
            let block = Gemma2Block {
                wq: test_linear(&dev, 0, q_dim, hidden_size, seed + 1),
                wk: test_linear(&dev, 0, kv_dim, hidden_size, seed + 2),
                wv: test_linear(&dev, 0, kv_dim, hidden_size, seed + 3),
                wo: test_linear(&dev, 0, hidden_size, q_dim, seed + 4),
                input_layernorm: test_norm(&dev, 0, hidden_size),
                post_attention_layernorm: test_norm(&dev, 0, hidden_size),
                pre_feedforward_layernorm: test_norm(&dev, 0, hidden_size),
                post_feedforward_layernorm: test_norm(&dev, 0, hidden_size),
                mlp: crate::gemma2::Gemma2Mlp {
                    gate_proj: test_linear(&dev, 0, intermediate_size, hidden_size, seed + 5),
                    up_proj: test_linear(&dev, 0, intermediate_size, hidden_size, seed + 6),
                    down_proj: test_linear(&dev, 0, hidden_size, intermediate_size, seed + 7),
                },
                rope: Rope::new(head_dim, cfg.rope_theta),
                num_heads,
                num_kv_heads,
                head_dim,
                attn_logit_softcapping: cfg.attn_logit_softcapping,
            };
            layers.push(block);
        }

        let gemma = Gemma2 {
            cfg: cfg.clone(),
            device: Device::Rocm(0),
            tok_embeddings: Linear {
                weight: embed.clone(),
                bias: None,
                w_t: embed,
                quant_format: None,
            },
            layers,
            norm: test_norm(&dev, 0, hidden_size),
            output: test_linear(&dev, 0, vocab_size, hidden_size, 999),
        };

        let mut graph = gemma
            .get_or_create_decode_graph(512, 1)
            .expect("alloc graph");

        let token_id = 9u32;
        // Warmup before capture
        gemma
            .forward_capture(&mut graph, token_id)
            .expect("warmup 1");
        gemma
            .forward_capture(&mut graph, token_id)
            .expect("warmup 2");

        graph.begin_capture().expect("begin capture");
        gemma
            .forward_capture(&mut graph, token_id)
            .expect("forward capture");
        graph.end_capture().expect("end capture");

        gemma
            .forward_replay(&mut graph, token_id)
            .expect("replay token");
        dev.synchronize();

        let logits_storage = graph.logits_device_storage();
        let rocm_st = logits_storage
            .as_any()
            .downcast_ref::<grim_backend_rocm::RocmStorage>()
            .expect("RocmStorage");
        let host_bytes = rocm_st.copy_to_host().expect("copy_to_host");
        let host_logits: &[f32] = unsafe {
            std::slice::from_raw_parts(host_bytes.as_ptr() as *const f32, host_bytes.len() / 4)
        };

        assert_eq!(host_logits.len(), cfg.vocab_size);
        for (i, &val) in host_logits.iter().enumerate() {
            assert!(val.is_finite(), "Gemma2 logit {i} is not finite: {val}");
            // Softcapping test: magnitude must be <= cap (30.0)
            if let Some(cap) = cfg.final_logit_softcapping {
                assert!(
                    val.abs() <= cap + 1e-4,
                    "Logit {val} exceeds final softcap {cap}"
                );
            }
        }
    }

    fn test_layer_norm(dev: &RocmDevice, ordinal: usize, dim: usize) -> crate::falcon::LayerNorm {
        let ones = vec![1.0f32; dim];
        let w = rocm_tensor(dev, ordinal, ones, Shape::new(vec![dim]));
        crate::falcon::LayerNorm {
            weight: w,
            bias: None,
            eps: 1e-5,
        }
    }

    #[test]
    fn test_chameleon_decode_graph_capture_replay() {
        if !grim_backend_rocm::device::util::gpu_test_enabled() {
            eprintln!("skip: set GRIM_GPU_TEST=1 for GPU graph test");
            return;
        }
        let _gpu_guard = grim_backend_rocm::device::util::gpu_test_lock();
        let dev = RocmDevice::shared(0);
        unsafe {
            std::env::set_var("GRIM_DECODE_GRAPH", "1");
        }

        let vocab_size = 64;
        let hidden_size = 64;
        let intermediate_size = 128;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 16;
        let num_layers = 2;

        let cfg = crate::chameleon::ChameleonConfig {
            vocab_size,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            num_layers,
            intermediate_size,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 512,
            swin_norm: true,
        };

        let embed = rocm_tensor(
            &dev,
            0,
            rand_vec(vocab_size * hidden_size, 7),
            Shape::new(vec![vocab_size, hidden_size]),
        );

        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let mut layers = Vec::new();

        for l in 0..num_layers {
            let seed = (l as u64) * 1000 + 30;
            let block = ChameleonBlock {
                wq: test_linear(&dev, 0, q_dim, hidden_size, seed + 1),
                wk: test_linear(&dev, 0, kv_dim, hidden_size, seed + 2),
                wv: test_linear(&dev, 0, kv_dim, hidden_size, seed + 3),
                wo: test_linear(&dev, 0, hidden_size, q_dim, seed + 4),
                q_norm: Some(test_layer_norm(&dev, 0, head_dim)),
                k_norm: Some(test_layer_norm(&dev, 0, head_dim)),
                attn_norm: test_norm(&dev, 0, hidden_size),
                ffn_norm: test_norm(&dev, 0, hidden_size),
                w_gate: test_linear(&dev, 0, intermediate_size, hidden_size, seed + 5),
                w_up: test_linear(&dev, 0, intermediate_size, hidden_size, seed + 6),
                w_down: test_linear(&dev, 0, hidden_size, intermediate_size, seed + 7),
                rope: Rope::new(head_dim, cfg.rope_theta),
                num_heads,
                num_kv_heads,
                head_dim,
                swin_norm: cfg.swin_norm,
                wqkv_q80_fused: None,
            };
            layers.push(block);
        }

        let chameleon = Chameleon {
            cfg: cfg.clone(),
            device: Device::Rocm(0),
            tok_embeddings: Linear {
                weight: embed.clone(),
                bias: None,
                w_t: embed,
                quant_format: None,
            },
            layers,
            norm: test_norm(&dev, 0, hidden_size),
            output: test_linear(&dev, 0, vocab_size, hidden_size, 888),
        };

        let mut graph = chameleon
            .get_or_create_decode_graph(512, 1)
            .expect("alloc graph");

        let token_id = 9u32;
        // Warmup before capture
        chameleon
            .forward_capture(&mut graph, token_id)
            .expect("warmup 1");
        chameleon
            .forward_capture(&mut graph, token_id)
            .expect("warmup 2");

        graph.begin_capture().expect("begin capture");
        chameleon
            .forward_capture(&mut graph, token_id)
            .expect("forward capture");
        graph.end_capture().expect("end capture");

        chameleon
            .forward_replay(&mut graph, token_id)
            .expect("replay token");
        dev.synchronize();

        let logits_storage = graph.logits_device_storage();
        let rocm_st = logits_storage
            .as_any()
            .downcast_ref::<grim_backend_rocm::RocmStorage>()
            .expect("RocmStorage");
        let host_bytes = rocm_st.copy_to_host().expect("copy_to_host");
        let host_logits: &[f32] = unsafe {
            std::slice::from_raw_parts(host_bytes.as_ptr() as *const f32, host_bytes.len() / 4)
        };

        assert_eq!(host_logits.len(), cfg.vocab_size);
        for (i, &val) in host_logits.iter().enumerate() {
            assert!(val.is_finite(), "Chameleon logit {i} is not finite: {val}");
        }
    }

    // ─── Thin Llama wrapper: dispatcher + capture/replay through the wrapper ───
    #[test]
    fn llama_wrapper_dispatch_and_graph_replay() {
        if !grim_backend_rocm::device::util::gpu_test_enabled() {
            eprintln!("skip: set GRIM_GPU_TEST=1 for GPU graph test");
            return;
        }
        let _gpu_guard = grim_backend_rocm::device::util::gpu_test_lock();
        let dev = RocmDevice::shared(0);
        let (llama, cfg) = make_test_llama(&dev, 0, false);

        // Dispatcher: the wrapper resolves, the inner type does not (it is
        // matched by name in run.rs before the wrapper arm).
        let wrapper = crate::olmo::Olmo {
            cfg: crate::olmo::OlmoConfig {
                vocab_size: cfg.vocab_size,
                hidden_size: cfg.hidden_size,
                num_heads: cfg.num_heads,
                num_kv_heads: cfg.num_kv_heads,
                head_dim: cfg.head_dim,
                num_layers: cfg.num_layers,
                intermediate_size: cfg.intermediate_size,
                max_seq_len: cfg.max_seq_len,
                rope_theta: cfg.rope_theta,
                rms_norm_eps: cfg.rms_norm_eps,
            },
            device: Device::Rocm(0),
            inner: llama,
        };
        let any: &dyn std::any::Any = &wrapper;
        let dg = super::llama_wrapper_graph_model(any)
            .expect("wrapper must resolve to DecodeGraphModel");

        let mut graph = dg.get_or_create_decode_graph(512, 1).expect("alloc graph");
        let token = 9u32;
        dg.forward_capture(&mut graph, token).expect("warmup 1");
        dg.forward_capture(&mut graph, token).expect("warmup 2");
        graph.begin_capture().expect("begin capture");
        dg.forward_capture(&mut graph, token).expect("capture");
        graph.end_capture().expect("end capture");
        dg.forward_replay(&mut graph, token).expect("replay");
        dev.synchronize();

        let rocm_st = graph
            .logits_device_storage()
            .as_any()
            .downcast_ref::<grim_backend_rocm::RocmStorage>()
            .expect("RocmStorage");
        let host = rocm_st.copy_to_host().expect("copy_to_host");
        assert!(!host.is_empty(), "no logits after replay");
    }

    fn make_test_llama_moe(dev: &RocmDevice) -> (Llama, crate::model::LlamaConfig) {
        let (mut llama, cfg) = make_test_llama(dev, 0, false);
        let num_experts = 4;
        let top_k = 2;
        let hidden_size = cfg.hidden_size;
        let intermediate_size = cfg.intermediate_size;
        for (i, layer) in llama.layers.iter_mut().enumerate() {
            layer.ffn_disabled = true;
            let seed = (i as u64) * 1000 + 40;
            let router_gate = test_linear(dev, 0, num_experts, hidden_size, seed + 20);
            let router = MoeRouter::new(
                router_gate,
                RouterKind::SoftmaxTopK,
                top_k,
                num_experts,
                None,
            );
            let mut egate = Vec::new();
            let mut eup = Vec::new();
            let mut edown = Vec::new();
            for e in 0..num_experts {
                let eseed = seed + 30 + (e as u64) * 10;
                egate.push(test_linear(
                    dev,
                    0,
                    intermediate_size,
                    hidden_size,
                    eseed + 1,
                ));
                eup.push(test_linear(
                    dev,
                    0,
                    intermediate_size,
                    hidden_size,
                    eseed + 2,
                ));
                edown.push(test_linear(
                    dev,
                    0,
                    hidden_size,
                    intermediate_size,
                    eseed + 3,
                ));
            }
            let moe_ffn = MoeFfn::new(
                router,
                ExpertBank::from_linears(egate, eup, edown),
                None,
                1.0,
            );
            llama.moe_blocks[i] = Some(MoeBlock {
                ffn_norm: test_norm(dev, 0, hidden_size),
                moe: moe_ffn,
                tp_config: layer.tp_config,
            });
        }
        (llama, cfg)
    }

    fn eager_single_token_logits(model: &Llama, dev: &RocmDevice, token_id: u32) -> Vec<f32> {
        let mut session = SessionInner::new(Device::Rocm(0));
        let input = rocm_tensor(dev, 0, vec![token_id as f32], Shape::new(vec![1, 1]));
        let positions = rocm_tensor(dev, 0, vec![0.0], Shape::new(vec![1, 1]));
        model
            .forward(&mut session, &input, &positions, &[])
            .expect("eager single-token forward")
            .to_vec_f32()
            .expect("eager logits readback")
    }

    #[test]
    fn test_llama_dense_and_moe_graph_match_eager_via_shared_contract() {
        if !grim_backend_rocm::device::util::gpu_test_enabled() {
            eprintln!("skip: set GRIM_GPU_TEST=1 for GPU graph test");
            return;
        }
        let _gpu_guard = grim_backend_rocm::device::util::gpu_test_lock();
        let dev = RocmDevice::shared(0);
        unsafe {
            std::env::set_var("GRIM_DECODE_GRAPH", "1");
        }
        let token_id = 17u32;

        let (dense, _) = make_test_llama(&dev, 0, false);
        let dense_eager = eager_single_token_logits(&dense, &dev, token_id);
        assert_graph_matches_eager_single_token(
            &dense,
            &dev,
            token_id,
            &dense_eager,
            "Llama dense",
        );

        let (moe, _) = make_test_llama_moe(&dev);
        let moe_eager = eager_single_token_logits(&moe, &dev, token_id);
        assert_graph_matches_eager_single_token(&moe, &dev, token_id, &moe_eager, "Llama MoE");
    }

    #[test]
    fn test_llama_moe_decode_graph_capture_replay() {
        if !grim_backend_rocm::device::util::gpu_test_enabled() {
            eprintln!("skip: set GRIM_GPU_TEST=1 for GPU graph test");
            return;
        }
        let _gpu_guard = grim_backend_rocm::device::util::gpu_test_lock();
        let dev = RocmDevice::shared(0);
        unsafe {
            std::env::set_var("GRIM_DECODE_GRAPH", "1");
        }

        let (llama, cfg) = make_test_llama_moe(&dev);

        let mut graph = llama
            .get_or_create_decode_graph(512, 1)
            .expect("alloc decode graph with MoE");
        let token_id = 17u32;

        // Warmup
        llama
            .forward_capture(&mut graph, token_id)
            .expect("warmup 1");
        llama
            .forward_capture(&mut graph, token_id)
            .expect("warmup 2");

        // Capture
        graph.begin_capture().expect("begin capture");
        llama
            .forward_capture(&mut graph, token_id)
            .expect("forward capture");
        graph.end_capture().expect("end capture");

        // Replay
        llama
            .forward_replay(&mut graph, token_id)
            .expect("replay token");
        dev.synchronize();

        let logits_storage = graph.logits_device_storage();
        let rocm_st = logits_storage
            .as_any()
            .downcast_ref::<grim_backend_rocm::RocmStorage>()
            .expect("RocmStorage");
        let host_bytes = rocm_st.copy_to_host().expect("copy_to_host");
        let host_logits: &[f32] = unsafe {
            std::slice::from_raw_parts(host_bytes.as_ptr() as *const f32, host_bytes.len() / 4)
        };

        assert_eq!(host_logits.len(), cfg.vocab_size);
        for (i, &val) in host_logits.iter().enumerate() {
            assert!(
                val.is_finite(),
                "MoE decode graph logit {i} is not finite: {val}"
            );
        }
    }
}
