//! Generic Decode Graph Capture & Replay trait for ROCm CausalLM models.
//!
//! Provides `DecodeGraphModel` trait, standardizing decode-step HIP graph recording,
//! KV seeding, and replay across architectures (LFM2, Llama, and derivatives).

use std::sync::Arc;

use grim_backend_rocm::as_rocm;
use grim_backend_rocm::decode_graph_buffers::{
    check_layer_topology, decode_graph_enabled, launch_attention, launch_qkv_gemv, ConvDeviceSeed,
    ConvRingSeed, DecodeGraph, DecodeGraphBuffers, EagerKdaSource, EagerKvSource,
};
use grim_core::error::Result;
use grim_nn::NormKind;
use grim_tensor::backend::ElementwiseOps;
use grim_tensor::{BackendStorage, CoreTensorOps, Device, MemoryOps, RopeConfig, Shape};

use crate::block::LlamaBlock;
use crate::chameleon::{Chameleon, ChameleonBlock};
use crate::deepseek2::DeepSeek2;
use crate::deepseek32::DeepSeek32;
use crate::deepseek4::DeepSeek4;
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

    /// Export eager device KDA states for recurrent state seeding.
    fn eager_kda_seed_sources<'a>(
        &self,
        _session: &'a dyn grim_core::session::SessionT,
    ) -> Result<Vec<Option<EagerKdaSource<'a>>>> {
        Ok(Vec::new())
    }

    /// Host conv-ring snapshots for recurrent-layer seeding (default: no
    /// conv layers). Indexed by layer; `None` for attention layers. Borrowed
    /// from the session state — must outlive the seed H2D copy.
    fn eager_conv_seed_rings<'a>(
        &self,
        _session: &'a dyn grim_core::session::SessionT,
    ) -> Result<Vec<Option<ConvRingSeed<'a>>>> {
        Ok(Vec::new())
    }

    /// Device conv rings for recurrent-layer seeding: when the eager prefill
    /// ran the D2D conv, the LIVE ring is `conv_state_dev` and the host mirror
    /// is stale — seeding from the mirror uploads zeros. Indexed by layer;
    /// non-empty return means "seed rings D2D from these" and the host-ring
    /// seed is skipped. Default: empty (host-ring seeding).
    fn eager_conv_device_seed_sources<'a>(
        &self,
        _session: &'a dyn grim_core::session::SessionT,
    ) -> Result<Vec<Option<ConvDeviceSeed<'a>>>> {
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

/// One decode GEMV into a preallocated graph buffer.
///
/// `what` and `layer` are not decoration: a shape mismatch here aborts the
/// WHOLE capture, so the run silently drops to eager for every subsequent
/// token. A bare "expected [1, 4096], got [1, 12288]" names neither the
/// projection nor the layer, which is why this class of bug kept costing a
/// full round of guessing — name them at the point of failure instead.
fn linear_into(
    dev: &Dev,
    a: &Storage,
    w: &grim_tensor::Tensor,
    out: &grim_backend_rocm::RocmStorage,
    act_q81: &grim_backend_rocm::RocmStorage,
) -> Result<()> {
    linear_into_named(dev, a, w, out, act_q81, "?", usize::MAX)
}

fn linear_into_named(
    dev: &Dev,
    a: &Storage,
    w: &grim_tensor::Tensor,
    out: &grim_backend_rocm::RocmStorage,
    act_q81: &grim_backend_rocm::RocmStorage,
    what: &str,
    layer: usize,
) -> Result<()> {
    let ws = rocm_storage(w)?;
    // Shapes up front: a failure below is almost always a buffer sized for a
    // different projection, and having both ends makes that visible.
    let w_dims = w.shape().dims().to_vec();
    let a_dims = a.shape().dims().to_vec();
    let o_dims = out.shape().dims().to_vec();

    if std::env::var_os("GRIM_XING_TRACE").is_some() {
        eprintln!(
            "[xing-linear] L{layer} {what}: {:?} {:?}",
            w.dtype().storage,
            w_dims
        );
    }


    // WhiteRaven blocked FP8 (16x16-blocked E4M3) cannot use
    // `linear_decode_into` here: its act path does a D2H readback plus an
    // allocation, and inside a capture the allocation is CAPTURE_POISON while
    // the sync stalls replay. Route it through the capture-safe leg, which
    // quantizes A on device into a preallocated scratch
    // (`linear_decode_blocked_into`). A hard error if the scratch is missing —
    // a silent eager fallback here would capture a subset of the graph and
    // then diverge from eager, which is the failure this whole file guards.
    if matches!(
        w.dtype().storage,
        grim_tensor::Storage::FloatPack(grim_tensor::FloatPackScheme::Fp8Blocked16)
    ) {
        let a_s = dst_downcast(a).map_err(|e| {
            grim_core::error::Error::Backend(format!(
                "linear_decode[{what} @ layer {layer}]: act not RocmStorage: {e}"
            ))
        })?;
        // Scratch is published by `DecodeGraphBuffers::begin_capture` and
        // cleared by end/abort_capture, so it cannot outlive its buffers.
        let pad = grim_backend_rocm::decode_graph_buffers::capture_fp8_pad_scratch().ok_or_else(
            || {
                grim_core::error::Error::Backend(format!(
                "linear_decode[{what} @ layer {layer}]: WhiteRaven blocked weight found but no \
                 padded-fp8 act scratch is installed for capture (act {:?}, weight {:?}, out {:?})",
                a_dims, w_dims, o_dims
            ))
            },
        )?;
        let k = a_s.shape().dims().last().copied().unwrap_or(0);
        let m = a_s.shape().elem_count().checked_div(k.max(1)).unwrap_or(0);
        let need = m.div_ceil(16) * 16 * k;
        if pad.bytes() < need {
            return Err(grim_core::error::Error::Backend(format!(
                "linear_decode[{what} @ layer {layer}]: fp8 scratch {}B < {need}B",
                pad.bytes()
            )));
        }
        dev.linear_decode_blocked_into(a_s, ws, out, pad)
            .map(|_| ())
            .map_err(|e| {
                grim_core::error::Error::Backend(format!(
                    "linear_decode_blocked[{what} @ layer {layer}]: {e} \
                     (act {:?}, weight {:?}, out {:?})",
                    a_dims, w_dims, o_dims
                ))
            })?;
        return Ok(());
    }

    dev.linear_decode_into(a, ws, out, act_q81).map_err(|e| {
        grim_core::error::Error::Backend(format!(
            "linear_decode[{what} @ layer {layer}]: {e} \
             (act {:?}, weight {:?}, out {:?})",
            a_dims, w_dims, o_dims
        ))
    })?;
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

        // 1. Attention norm into norm_buf. Dispatch on kind: the fused decode
        // graph has a `rms_norm` and a `layer_norm` kernel, and picking the
        // wrong one is the phi2 bug (rms where the reference needs layer).
        // `device_weight` supplies a ones vector when the checkpoint has no
        // norm weight, which `olmo.cpp:65-67` does.
        let attn_in = &buffers.layer_input[layer_idx];
        let attn_w = self
            .attn_norm
            .device_weight(self._cfg.hidden_size)
            .map_err(grim_core::error::Error::Tensor)?;
        match self.attn_norm.kind {
            NormKind::Rms => {
                dev.rms_norm_into(
                    attn_in,
                    &**attn_w.storage(),
                    self.attn_norm.eps,
                    &buffers.norm_buf[layer_idx],
                    &attn_in.shape().clone(),
                )
                .map_err(grim_core::error::Error::Tensor)?;
            }
            NormKind::LayerNorm => {
                let b = self.attn_norm.bias.as_ref().map(|t| t.storage());
                dev.layer_norm_into(
                    attn_in,
                    &**attn_w.storage(),
                    b.as_deref().map(|s| s.as_ref()),
                    self.attn_norm.eps,
                    &buffers.norm_buf[layer_idx],
                    &attn_in.shape().clone(),
                )
                .map_err(grim_core::error::Error::Tensor)?;
            }
        }
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

        // 6b. Attention OUTPUT GATE — `out = out * sigmoid(gate)`.
        //
        // This was MISSING: the graph went from the attention straight to `wo`,
        // so the fused gate half was computed and then discarded. The reference
        // applies it (qwen35.cpp:321-328):
        //     gate_sigmoid = ggml_sigmoid(ctx0, gate);
        //     cur          = ggml_mul(ctx0, cur, gate_sigmoid);
        // and Qwen3.5/3.8 always has an output gate on its attention layers, so
        // this silently changes every attention layer's output. It is a
        // CORRECTNESS fix, not a shape fix: with the shapes right this would
        // still be wrong.
        //
        // `q_head_buf` is dead after the RoPE, so the sigmoid lands there and no
        // third buffer is needed.
        dev.sigmoid_into(
            &buffers.q_gate_buf[layer_idx],
            &buffers.q_head_buf[layer_idx],
        )
        .map_err(|e| grim_core::error::Error::Backend(format!("qwen35 attn gate sigmoid: {e}")))?;
        dev.mul_into(
            &buffers.attn_out_buf[layer_idx],
            &buffers.q_head_buf[layer_idx],
            &buffers.attn_out_buf[layer_idx],
        )
        .map_err(|e| grim_core::error::Error::Backend(format!("qwen35 attn gate mul: {e}")))?;

        // 7. Output projection (wo) + residual add
        linear_into_named(
            dev,
            &buffers.attn_out_buf[layer_idx],
            self.wo.weight(),
            &buffers.norm_buf[layer_idx],
            act,
            "attn.wo_out",
            layer_idx,
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
            // Same kind dispatch as the attention norm above.
            let ffn_w = self
                .ffn_norm
                .device_weight(self._cfg.hidden_size)
                .map_err(grim_core::error::Error::Tensor)?;
            match self.ffn_norm.kind {
                NormKind::Rms => {
                    dev.rms_norm_into(
                        &buffers.layer_output[layer_idx],
                        &**ffn_w.storage(),
                        self.ffn_norm.eps,
                        &buffers.norm_buf[layer_idx],
                        &ffn_shape,
                    )
                    .map_err(grim_core::error::Error::Tensor)?;
                }
                NormKind::LayerNorm => {
                    let b = self.ffn_norm.bias.as_ref().map(|t| t.storage());
                    dev.layer_norm_into(
                        &buffers.layer_output[layer_idx],
                        &**ffn_w.storage(),
                        b.as_deref().map(|s| s.as_ref()),
                        self.ffn_norm.eps,
                        &buffers.norm_buf[layer_idx],
                        &ffn_shape,
                    )
                    .map_err(grim_core::error::Error::Tensor)?;
                }
            }

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
                linear_into_named(
                    dev,
                    normed_ffn,
                    wg.weight(),
                    &buffers.gate_buf[layer_idx],
                    act,
                    "ffn.gate",
                    layer_idx,
                )?;
                linear_into_named(
                    dev,
                    normed_ffn,
                    wu.weight(),
                    &buffers.up_buf[layer_idx],
                    act,
                    "ffn.up",
                    layer_idx,
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
            linear_into_named(
                dev,
                &buffers.activated_buf[layer_idx],
                wd.weight(),
                &buffers.norm_buf[layer_idx],
                act,
                "ffn.down",
                layer_idx,
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
            // Non-hybrid call site: the branch width IS n_q here, so
            // behaviour is unchanged for it.
            n_q,
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
                    false, // softmax mode already normalizes
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

        // Final norm + output head. Dispatch on kind exactly as the per-layer
        // norms do: picking rms where the reference needs layer is the phi2
        // bug, and the final norm is a second place it can happen.
        let h_shape = graph.buffers.head_input.shape().clone();
        let head_w = self
            .norm
            .device_weight(self.cfg.hidden_size)
            .map_err(grim_core::error::Error::Tensor)?;
        match self.norm.kind {
            NormKind::Rms => {
                dev.rms_norm_into(
                    &graph.buffers.head_input,
                    &**head_w.storage(),
                    self.norm.eps,
                    &graph.buffers.head_input,
                    &h_shape,
                )
                .map_err(grim_core::error::Error::Tensor)?;
            }
            NormKind::LayerNorm => {
                let b = self.norm.bias.as_ref().map(|x| x.storage());
                dev.layer_norm_into(
                    &graph.buffers.head_input,
                    &**head_w.storage(),
                    b.as_deref().map(|s| s.as_ref()),
                    self.norm.eps,
                    &graph.buffers.head_input,
                    &h_shape,
                )
                .map_err(grim_core::error::Error::Tensor)?;
            }
        }

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
                linear_into_named(
                    dev,
                    normed,
                    wq.weight(),
                    &buffers.q_buf[layer_idx],
                    act,
                    "attn.q",
                    layer_idx,
                )?;
                linear_into_named(
                    dev,
                    normed,
                    wk.weight(),
                    &buffers.k_buf[layer_idx],
                    act,
                    "attn.k",
                    layer_idx,
                )?;
                linear_into_named(
                    dev,
                    normed,
                    wv.weight(),
                    &buffers.v_buf[layer_idx],
                    act,
                    "attn.v",
                    layer_idx,
                )?;
            }

            // 3. Split the fused Q|gate. `q_buf` holds [Q | gate] at 2*q_dim;
            // the norm and the RoPE below apply to Q alone, and the gate is
            // needed later to scale the attention output. llama.cpp splits with
            // a view (qwen35.cpp:289-293) and keeps the halves apart
            // thereafter; nothing downstream here may read the fused buffer as
            // if it were Q.
            dev.copy_slice_range(
                &buffers.q_head_buf[layer_idx],
                0,
                &buffers.q_buf[layer_idx],
                0,
                nh * hd,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("qwen35 split Q: {e}")))?;
            dev.copy_slice_range(
                &buffers.q_gate_buf[layer_idx],
                0,
                &buffers.q_buf[layer_idx],
                nh * hd,
                nh * hd,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("qwen35 split gate: {e}")))?;

            // 3b. Optional per-head Q/K norm
            if let Some(qn) = &self.attn_q_norm {
                let qn_shape = Shape::new(vec![batch * nh, hd]);
                dev.rms_norm_into(
                    &buffers.q_head_buf[layer_idx],
                    &**qn.weight.storage(),
                    qn.eps,
                    &buffers.q_head_buf[layer_idx],
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
                &buffers.q_head_buf[layer_idx],
                &buffers.pos_dev,
                &buffers.q_head_buf[layer_idx],
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
            // KDA Recurrent layer forward pass inside captured HIP graph
            let qkv_lin = self.attn_qkv.as_ref().ok_or_else(|| {
                grim_core::error::Error::Backend("recurrent block missing attn_qkv".into())
            })?;
            let alpha_lin = self.ssm_alpha.as_ref().ok_or_else(|| {
                grim_core::error::Error::Backend("recurrent block missing ssm_alpha".into())
            })?;
            let beta_lin = self.ssm_beta.as_ref().ok_or_else(|| {
                grim_core::error::Error::Backend("recurrent block missing ssm_beta".into())
            })?;
            let gate_lin = self.attn_gate.as_ref().ok_or_else(|| {
                grim_core::error::Error::Backend("recurrent block missing attn_gate (z)".into())
            })?;
            let conv_w = self.ssm_conv1d.as_ref().ok_or_else(|| {
                grim_core::error::Error::Backend("recurrent block missing ssm_conv1d".into())
            })?;
            if conv_w.storage().device_ptr().is_none() {
                return Err(grim_core::error::Error::Backend(
                    "ssm_conv1d weight lacks valid device pointer for graph capture".into(),
                ));
            }
            let dt_bias_d = self.ssm_dt_bias_dev.as_ref().ok_or_else(|| {
                grim_core::error::Error::Backend("recurrent block missing ssm_dt_bias_dev".into())
            })?;
            let ssm_a_d = self.ssm_a_dev.as_ref().ok_or_else(|| {
                grim_core::error::Error::Backend("recurrent block missing ssm_a_dev".into())
            })?;
            let ssm_norm_d = self.ssm_norm_dev.as_ref().ok_or_else(|| {
                grim_core::error::Error::Backend("recurrent block missing ssm_norm_dev".into())
            })?;
            let kda_buf = buffers
                .kda
                .get(layer_idx)
                .and_then(|o| o.as_ref())
                .ok_or_else(|| {
                    grim_core::error::Error::Backend(format!(
                        "missing KDA graph buffers for layer {layer_idx}"
                    ))
                })?;

            // 1. Projections: QKV -> q_buf, alpha -> kda_alpha, beta -> kda_beta, gate (z) -> gate_buf
            linear_into_named(
                dev,
                normed,
                qkv_lin.weight(),
                &buffers.q_buf[layer_idx],
                act,
                "kda.attn_qkv",
                layer_idx,
            )?;
            linear_into_named(
                dev,
                normed,
                alpha_lin.weight(),
                &kda_buf.alpha,
                act,
                "kda.alpha",
                layer_idx,
            )?;
            linear_into_named(
                dev,
                normed,
                beta_lin.weight(),
                &kda_buf.beta,
                act,
                "kda.beta",
                layer_idx,
            )?;
            linear_into_named(
                dev,
                normed,
                gate_lin.weight(),
                &kda_buf.gate,
                act,
                "kda.attn_gate",
                layer_idx,
            )?;

            // 2. Short conv causal step into kda_buf.conv_out
            if layer_idx >= buffers.sc_state.len() {
                return Err(grim_core::error::Error::Backend(format!(
                    "layer {layer_idx} >= sc_state {}",
                    buffers.sc_state.len()
                )));
            }
            dev.short_conv1d_causal_step_into(
                &buffers.q_buf[layer_idx],
                conv_w.storage().as_ref(),
                None,
                &buffers.sc_state[layer_idx],
                &kda_buf.conv_out,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("qwen35 sc conv: {e}")))?;

            // 3. Batched KDA gated delta rule + head norm gate into attn_out_buf
            let n_val_heads = self.cfg_ssm_num_value_heads();
            let n_key_heads = self.cfg_ssm_num_key_heads();
            let head_dim = self.cfg_ssm_head_dim();
            let eps = self.attn_norm.eps;
            dev.kda_gated_delta_rule_batched_into(
                &kda_buf.conv_out,
                &kda_buf.alpha,
                &kda_buf.beta,
                dt_bias_d.as_ref(),
                ssm_a_d.as_ref(),
                ssm_norm_d.as_ref(),
                Some(&kda_buf.gate),
                &kda_buf.state,
                &kda_buf.acc_scratch,
                &kda_buf.branch,
                n_val_heads,
                n_key_heads,
                head_dim,
                eps,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("kda gated delta rule: {e}")))?;

            // 4. Output projection (ssm_out)
            let out_proj = self.ssm_out.as_ref().or(self.wo.as_ref()).ok_or_else(|| {
                grim_core::error::Error::Backend("missing ssm_out projection".into())
            })?;
            linear_into(
                dev,
                &kda_buf.branch,
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

        // The recurrent conv width. `q_buf` doubles as the conv scratch, and
        // the Mamba short-conv staging below is sized from it, so it is
        // derived here rather than inherited from the attention query width —
        // coupling the two is what made the conv scratch 2048 rows too narrow
        // on Qwen3.5-9B and aborted capture at step 1.
        let kda_conv = (self.cfg.ssm_dt_rank + 2 * self.cfg.ssm_n_group) * self.cfg.ssm_d_state;
        let n_q = (self.cfg.num_heads * self.cfg.head_dim).max(kda_conv);
        let n_k = self.cfg.num_kv_heads * self.cfg.head_dim;
        let n_v = n_k;
        let inter = self.cfg.intermediate_size;
        let vocab = self.cfg.vocab_size.max(1);
        let ctx = max_ctx.max(1);
        let nh = self.cfg.num_heads;
        // Short-conv hidden width == the KDA conv width, NOT the attention
        // query width. The allocator stages `3 * sc_h_dim` for it, so taking
        // `n_q` here staged a buffer whose width tracked whatever the
        // attention query happened to be.
        let sc_h_dim = if kda_conv > 0 { kda_conv } else { n_q };
        let sc_l_cache = self.cfg.ssm_d_conv.max(4);
        let hidden = self.cfg.hidden_size;

        // `n_q` is `(q_dim).max(kda_conv)`: correct for `q_buf`, which holds a
        // PROJECTION OUTPUT and must fit conv_dim (recurrent) or 2*q_dim
        // (fused Q|gate, attention). It is NOT the branch width. `attn_out_buf`
        // is consumed only by attention layers now — recurrent layers write
        // their branch to `KdaLayerBuffers::branch` at value_dim — so it gets
        // the true `q_dim`. See `attn_out_buf`'s doc for the two references.
        let n_attn_out = self.cfg.num_heads * self.cfg.head_dim;
        let mut buffers = DecodeGraphBuffers::allocate(
            &dev,
            self.blocks.len(),
            hidden,
            n_q,
            n_attn_out,
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

        if self.cfg.ssm_dt_rank > 0 {
            let n_val = self.cfg.ssm_dt_rank;
            let n_key = self.cfg.ssm_n_group;
            let d_state = self.cfg.ssm_d_state;
            for (layer_idx, block) in self.blocks.iter().enumerate() {
                if !block.is_full_attention {
                    buffers.allocate_kda_layer(
                        &dev, layer_idx, batch, n_val, n_key, d_state, kda_conv,
                    )?;
                }
            }
        }

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

    fn eager_kda_seed_sources<'a>(
        &self,
        session: &'a dyn grim_core::session::SessionT,
    ) -> Result<Vec<Option<EagerKdaSource<'a>>>> {
        let caches = match session
            .model_state()
            .and_then(|s| s.downcast_ref::<Vec<Qwen35LayerCache>>())
        {
            Some(c) => c,
            None => return Ok(Vec::new()),
        };

        let mut out = Vec::with_capacity(caches.len());
        for (i, cache) in caches.iter().enumerate() {
            if self
                .blocks
                .get(i)
                .map(|b| b.is_full_attention)
                .unwrap_or(true)
            {
                out.push(None);
                continue;
            }
            let ssm_ptr = cache
                .ssm_state_dev
                .as_ref()
                .and_then(|s| as_rocm(s.as_ref()).ok())
                .and_then(|r| r.device_ptr_u64())
                .map(|p| p as *const f32);

            match ssm_ptr {
                Some(ptr) => out.push(Some(EagerKdaSource {
                    kda_state: ptr,
                    _anchor: std::marker::PhantomData,
                })),
                None => out.push(None),
            }
        }
        Ok(out)
    }

    fn eager_conv_seed_rings<'a>(
        &self,
        session: &'a dyn grim_core::session::SessionT,
    ) -> Result<Vec<Option<ConvRingSeed<'a>>>> {
        let caches = match session
            .model_state()
            .and_then(|s| s.downcast_ref::<Vec<Qwen35LayerCache>>())
        {
            Some(c) => c,
            None => return Ok(Vec::new()),
        };

        let mut out = Vec::with_capacity(caches.len());
        let kc = self.cfg.ssm_d_conv.saturating_sub(1);
        let kda_conv = (self.cfg.ssm_dt_rank + 2 * self.cfg.ssm_n_group) * self.cfg.ssm_d_state;

        for (i, cache) in caches.iter().enumerate() {
            if self
                .blocks
                .get(i)
                .map(|b| b.is_full_attention)
                .unwrap_or(true)
            {
                out.push(None);
                continue;
            }
            if cache.conv_state.is_empty() || kc == 0 || kda_conv == 0 {
                out.push(None);
                continue;
            }
            out.push(Some(ConvRingSeed {
                host: &cache.conv_state,
                h_dim: kda_conv,
                kc,
                _anchor: std::marker::PhantomData,
            }));
        }
        Ok(out)
    }

    /// KDA-fix: when the eager prefill ran the D2D conv (the default), the
    /// LIVE conv ring is `conv_state_dev`; the host `conv_state` mirror is
    /// never written on that path, so seeding the graph from the mirror
    /// replays every recurrent layer against zeros. Prefer the device ring
    /// whenever it exists; the host-ring seed stays as the fallback for the
    /// host-prefill path.
    fn eager_conv_device_seed_sources<'a>(
        &self,
        session: &'a dyn grim_core::session::SessionT,
    ) -> Result<Vec<Option<ConvDeviceSeed<'a>>>> {
        let caches = match session
            .model_state()
            .and_then(|s| s.downcast_ref::<Vec<Qwen35LayerCache>>())
        {
            Some(c) => c,
            None => return Ok(Vec::new()),
        };

        let mut out = Vec::with_capacity(caches.len());
        for (i, cache) in caches.iter().enumerate() {
            if self
                .blocks
                .get(i)
                .map(|b| b.is_full_attention)
                .unwrap_or(true)
            {
                out.push(None);
                continue;
            }
            let ring_ptr = cache
                .conv_state_dev
                .as_ref()
                .and_then(|s| as_rocm(s.as_ref()).ok())
                .and_then(|r| r.device_ptr_u64())
                .map(|p| p as *const f32);
            match ring_ptr {
                Some(ptr) => out.push(Some(ConvDeviceSeed {
                    dev_ring: ptr,
                    _anchor: std::marker::PhantomData,
                })),
                None => out.push(None),
            }
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
            // Non-hybrid call site: the branch width IS n_q here, so
            // behaviour is unchanged for it.
            n_q,
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
    granite::Granite,
    gemma_embedding::GemmaEmbedding,
    exaone4::Exaone4,
    qwen3next::Qwen3Next,
    jais::Jais,
    granite_moe::GraniteMoe,
    cohere2::Cohere2,
    grok::Grok,
    bitnet::BitNet,
    deci::Deci,
    dflash::DFlash,
    jais2::Jais2,
    llada::Llada,
    bailingmoe::BailingMoe,
    exaone_moe::ExaoneMoe,
    llama4::Llama4,
    dots1::Dots1,
    olmo::Olmo,
    mistral3::Mistral3,
    exaone::Exaone,
    plm::Plm,
    glm4::Glm4,
    olmoe::Olmoe,
    deepseek2ocr::DeepSeek2Ocr,
    olmo2::Olmo2,
    openai_moe::OpenAiMoe,
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
        granite::Granite,
        gemma_embedding::GemmaEmbedding,
        exaone4::Exaone4,
        qwen3next::Qwen3Next,
        jais::Jais,
        granite_moe::GraniteMoe,
        cohere2::Cohere2,
        grok::Grok,
        bitnet::BitNet,
        deci::Deci,
        dflash::DFlash,
        jais2::Jais2,
        llada::Llada,
        bailingmoe::BailingMoe,
        exaone_moe::ExaoneMoe,
        llama4::Llama4,
        dots1::Dots1,
        olmo::Olmo,
        mistral3::Mistral3,
        exaone::Exaone,
        plm::Plm,
        glm4::Glm4,
        olmoe::Olmoe,
        deepseek2ocr::DeepSeek2Ocr,
        olmo2::Olmo2,
        openai_moe::OpenAiMoe,
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
                3,     // mode 3: softmax renormalized over top-k,
                false, // softmax mode already normalizes
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
                3,     // mode 3: softmax renormalized over top-k,
                false, // softmax mode already normalizes
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
                3,     // mode 3: softmax renormalized over top-k,
                false, // softmax mode already normalizes
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
                3,     // mode 3: softmax renormalized over top-k,
                false, // softmax mode already normalizes
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
                None,
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
                None,
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
                None,
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
        ColumnParallelLinear, Embedding, Linear, Norm, RmsNorm, Rope, RowParallelLinear,
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

    /// An `RmsNorm` on the ROCm device, for block types that still hold one
    /// (`Qwen35Block`, and `Llama`'s final `norm`). Only the per-layer
    /// `LlamaBlock` norms became `Norm`; see `test_llama_norm`.
    fn test_norm(dev: &RocmDevice, ordinal: usize, dim: usize) -> RmsNorm {
        let ones = vec![1.0f32; dim];
        let w = rocm_tensor(dev, ordinal, ones, Shape::new(vec![dim]));
        RmsNorm {
            weight: w,
            eps: 1e-5,
        }
    }

    /// The `Norm` form of [`test_norm`], for `LlamaBlock`'s per-layer norms.
    fn test_llama_norm(dev: &RocmDevice, ordinal: usize, dim: usize) -> Norm {
        let ones = vec![1.0f32; dim];
        let w = rocm_tensor(dev, ordinal, ones, Shape::new(vec![dim]));
        let mut n = Norm::new(NormKind::Rms, 1e-5);
        n.weight = Some(w);
        n
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
            norm_kind: grim_nn::NormKind::Rms,
            has_norm_bias: false,
            has_attn_post_norm: false,
            use_parallel_residual: false,
            has_output_bias: false,
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
                attn_post_norm: None,
                attn_norm: test_llama_norm(dev, ordinal, hidden_size),
                wq: ColumnParallelLinear::new(wq, tp),
                wk: ColumnParallelLinear::new(wk, tp),
                wv: ColumnParallelLinear::new(wv, tp),
                wo: RowParallelLinear::new(wo, tp),
                g_proj: None,
                q_norm: None,
                k_norm: None,
                ffn_norm: test_llama_norm(dev, ordinal, hidden_size),
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
                use_parallel_residual: false,
            });
        }

        // The final norm is a `Norm` now, unlike the per-layer RmsNorm.
        let norm = test_llama_norm(dev, ordinal, hidden_size);
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
            rotary_dim: None,
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
                ssm_dt_bias_dev: None,
                ssm_a_dev: None,
                ssm_norm_dev: None,
                ssm_dt_rank_hint: 0,
                ssm_n_group_hint: 0,
                ssm_d_state_hint: 0,
                ssm_d_conv_hint: 4,
                post_attention_norm: test_norm(&dev, 0, hidden_size),
                ffn_gate: test_linear(&dev, 0, intermediate_size, hidden_size, seed + 5),
                ffn_up: test_linear(&dev, 0, intermediate_size, hidden_size, seed + 6),
                ffn_down: test_linear(&dev, 0, hidden_size, intermediate_size, seed + 7),
                is_full_attention: is_full,
                layer_idx: l,
                num_heads,
                num_kv_heads,
                head_dim,
                rotary_dim: head_dim,
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

// ─── Xing40 decode-graph support (appended) ───
use crate::xing40::Xing40;

type GBox = Box<dyn BackendStorage>;

fn dev_for_xing40(m: &Xing40) -> Result<Arc<Dev>> {
    match &m.device {
        Device::Rocm(o) => Ok(Dev::shared(*o)),
        _ => Err(grim_core::error::Error::Unimplemented(
            "decode graph needs ROCm device".into(),
        )),
    }
}

/// Per-layer capture-safe scratch for one Xing40 decode step. All buffers are
/// allocated once at `get_or_create_decode_graph` (never inside capture);
/// the hc gate vectors live here so `grim_mhc_gates` writes them for the
/// collapse / write-back kernels to read in the same captured stream.
pub struct Xing40GraphScratch {
    /// Stream state: layer i reads `sin[i]` (layer 0's is the seed) and
    /// writes `sout[i]`, which feeds layer i+1.
    pub sin: Vec<GBox>,
    pub sout: Vec<GBox>,
    /// hc input_norm output (`[1, flat]`).
    pub hcn: Vec<GBox>,
    /// hc_fn projection `[1, mix]`.
    pub proj: Vec<GBox>,
    /// Gate vectors, token 0 of the `[*, seq]` token-last layout.
    pub pre: Vec<GBox>,
    pub post: Vec<GBox>,
    pub comb: Vec<GBox>,
    /// Collapse scratch `[1, hidden]`.
    pub col: Vec<GBox>,
    /// q_a LoRA output `[1, q_lora_rank]`.
    pub qlora: Vec<GBox>,
    /// q_b output `[1, nh*(nope+rope_d)]`.
    pub qf: Vec<GBox>,
    /// Split planes: q_nope `[1, nh*nope]`, q_pe `[1, nh*rope_d]`.
    pub qn: Vec<GBox>,
    pub qp: Vec<GBox>,
    /// Absorbed query `[1, nh*rank]` (reused for the decode kernel's
    /// normalized-latent output when W_UV runs as the per-head GEMV).
    pub qabs: Vec<GBox>,
    /// Scale-corrected query planes: the fused MLA decode kernel hardcodes
    /// `1/sqrt(rank + rope_d)` while the model's kq_scale is
    /// Attention output latent `[1, nh * rank]`.
    ///
    /// The fused MLA decode kernel writes its result here, NEVER back into
    /// `qabs`: with `out == q_absorbed`, head `h` writes `[h*vd, h*vd+vd)`
    /// while head `h'` reads `[h'*rank, h'*rank+rank)`, and those windows
    /// overlap (vd=128, rank=512), so heads clobber each other's queries and
    /// every replayed step is garbage.
    pub qattn_latent: Vec<GBox>,
    /// Latent staging: kv_a output `[1, rank+rope_d]`, then the split
    /// c_kv / k_pe planes, then the packed row.
    pub kvstage: Vec<GBox>,
    pub ckv: Vec<GBox>,
    pub kpe: Vec<GBox>,
    pub latent: Vec<GBox>,
    /// Attention output `[1, nh*vd]`.
    pub attn: Vec<GBox>,
    /// Shared-expert scratch (MoE layers): `[1, moe_inter]` x3 + `[1, hidden]`.
    pub shg: Vec<GBox>,
    pub shu: Vec<GBox>,
    pub sha: Vec<GBox>,
    pub sho: Vec<GBox>,
    /// Head mean `[1, hidden]`.
    pub meaned: GBox,
    /// Pre-dequantized F32 hc_fn weights (attn, ffn) — hc_fn ships F16 and
    /// the graph GEMV path wants F32; dequantized once, outside capture.
    pub hc_fn_attn: Vec<grim_tensor::Tensor>,
    pub hc_fn_ffn: Vec<grim_tensor::Tensor>,
    /// Gate bias + scales on device.
    pub hc_base: Vec<GBox>,
    pub hc_scale: Vec<GBox>,
    /// The FFN branch carries its OWN gate bias/scale. Feeding it the attn
    /// branch's (what this did before) rescales every ffn gate; layer 0 hid it
    /// because its `post` gate is ~1e-7, so only the layers with a larger
    /// `post` showed the error.
    /// Snapshot of the ATTENTION branch's gates, taken inside the capture by a
    /// device-to-device copy. `post`/`comb` are overwritten by the FFN's gate
    /// call moments later and cannot be read post-replay, while the warmup
    /// computes its gates from a stale stream - so the attention gates were
    /// the one layer-0 quantity with no observable value at a real prefix.
    /// Snapshot of the attention branch's `o_proj` output, taken before the
    /// FFN reuses `norm_buf`. With the absorb and both gate sets verified,
    /// this is the last unverified input to the attention write-back.
    /// Snapshot of `sin[0]` as layer 0 sees it. Post-replay sin[0] holds the
    /// LAST layer's output, so any oracle needing layer 0's input stream must
    /// read this instead - reading sin[0] late silently produces a reference
    /// ~1900x too large.
    /// Per-layer snapshot of the router's output. `moe_route_experts` /
    /// `moe_route_weights` are single shared buffers that EVERY layer
    /// overwrites, so post-replay they hold the LAST layer's routing; reading
    /// them against an earlier layer's `norm_buf` compares two different
    /// layers and "fails" at rel ~1.6 with nothing wrong in the model.
    pub route_experts_snap: Vec<GBox>,
    pub route_weights_snap: Vec<GBox>,
    pub sin0_snap: Vec<GBox>,
    pub attn_out_snap: Vec<GBox>,
    pub attn_post_snap: Vec<GBox>,
    pub attn_comb_snap: Vec<GBox>,
    pub hc_base_ffn: Vec<GBox>,
    pub hc_scale_ffn: Vec<GBox>,
    /// The checkpoint's rope (base + YaRN), interleaved pairing.
    pub rope_cfg: RopeConfig,
    pub mix: usize,
    pub flat: usize,
    pub rank: usize,
    pub rope_d: usize,
    pub nope: usize,
    pub vd: usize,
    pub nh: usize,
    pub hc: usize,
}

/// Build the scratch on `dev` (called once, outside capture).
fn build_xing_scratch(model: &Xing40, dev: &Dev, batch: usize) -> Result<Xing40GraphScratch> {
    let cfg = &model.cfg;
    let hc = cfg.hc_mult;
    let hidden = cfg.hidden_size;
    let flat = hc * hidden;
    let rank = cfg.kv_lora_rank;
    let rope_d = cfg.qk_rope_head_dim;
    let nope = cfg.qk_nope_head_dim;
    let vd = cfg.v_head_dim;
    let nh = cfg.num_heads;
    let mix = (2 + hc) * hc;
    let n_layers = model.layers.len();
    let moe_inter = cfg.moe_intermediate_size;
    let mut s = Xing40GraphScratch {
        sin: Vec::with_capacity(n_layers),
        sout: Vec::with_capacity(n_layers),
        hcn: Vec::with_capacity(n_layers),
        proj: Vec::with_capacity(n_layers),
        pre: Vec::with_capacity(n_layers),
        post: Vec::with_capacity(n_layers),
        comb: Vec::with_capacity(n_layers),
        col: Vec::with_capacity(n_layers),
        qlora: Vec::with_capacity(n_layers),
        qf: Vec::with_capacity(n_layers),
        qn: Vec::with_capacity(n_layers),
        qp: Vec::with_capacity(n_layers),
        qabs: Vec::with_capacity(n_layers),
        qattn_latent: Vec::with_capacity(n_layers),
        kvstage: Vec::with_capacity(n_layers),
        ckv: Vec::with_capacity(n_layers),
        kpe: Vec::with_capacity(n_layers),
        latent: Vec::with_capacity(n_layers),
        attn: Vec::with_capacity(n_layers),
        shg: Vec::with_capacity(n_layers),
        shu: Vec::with_capacity(n_layers),
        sha: Vec::with_capacity(n_layers),
        sho: Vec::with_capacity(n_layers),
        meaned: dev.zeros(&Shape::new(vec![batch, hidden]), grim_tensor::DType::F32)?,
        hc_fn_attn: Vec::with_capacity(n_layers),
        hc_fn_ffn: Vec::with_capacity(n_layers),
        hc_base: Vec::with_capacity(n_layers),
        hc_scale: Vec::with_capacity(n_layers),
        route_experts_snap: Vec::with_capacity(n_layers),
        route_weights_snap: Vec::with_capacity(n_layers),
        sin0_snap: Vec::with_capacity(1),
        attn_out_snap: Vec::with_capacity(n_layers),
        attn_post_snap: Vec::with_capacity(n_layers),
        attn_comb_snap: Vec::with_capacity(n_layers),
        hc_base_ffn: Vec::with_capacity(n_layers),
        hc_scale_ffn: Vec::with_capacity(n_layers),
        rope_cfg: {
            let mut rc = RopeConfig::new(rope_d, model.layers[0].self_attn.rope.config.base);
            rc.yarn = model.layers[0].self_attn.rope.config.yarn;
            // INTERLEAVED — the pairing every verified path uses (the d2d
            // k_pe NeoX defect is exactly what this must not reintroduce).
            rc.interleaved = true;
            rc
        },
        mix,
        flat,
        rank,
        rope_d,
        nope,
        vd,
        nh,
        hc,
    };
    for layer in &model.layers {
        s.sin
            .push(dev.zeros(&Shape::new(vec![batch, flat]), grim_tensor::DType::F32)?);
        s.sout
            .push(dev.zeros(&Shape::new(vec![batch, flat]), grim_tensor::DType::F32)?);
        s.hcn
            .push(dev.zeros(&Shape::new(vec![batch, flat]), grim_tensor::DType::F32)?);
        s.proj
            .push(dev.zeros(&Shape::new(vec![batch, mix]), grim_tensor::DType::F32)?);
        s.pre
            .push(dev.zeros(&Shape::new(vec![hc, batch]), grim_tensor::DType::F32)?);
        s.post
            .push(dev.zeros(&Shape::new(vec![hc, batch]), grim_tensor::DType::F32)?);
        s.sin0_snap
            .push(dev.zeros(&Shape::new(vec![batch, flat]), grim_tensor::DType::F32)?);
        s.route_experts_snap.push(dev.zeros(
            &Shape::new(vec![batch * cfg.n_routed_experts.max(1)]),
            grim_tensor::DType::U32,
        )?);
        s.route_weights_snap.push(dev.zeros(
            &Shape::new(vec![batch * cfg.n_routed_experts.max(1)]),
            grim_tensor::DType::F32,
        )?);
        s.attn_out_snap
            .push(dev.zeros(&Shape::new(vec![batch, hidden]), grim_tensor::DType::F32)?);
        s.attn_post_snap
            .push(dev.zeros(&Shape::new(vec![hc, batch]), grim_tensor::DType::F32)?);
        s.attn_comb_snap
            .push(dev.zeros(&Shape::new(vec![hc * hc, batch]), grim_tensor::DType::F32)?);
        s.comb
            .push(dev.zeros(&Shape::new(vec![hc * hc, batch]), grim_tensor::DType::F32)?);
        s.col
            .push(dev.zeros(&Shape::new(vec![batch, hidden]), grim_tensor::DType::F32)?);
        let qlora_w = layer
            .self_attn
            .q_a_proj
            .as_ref()
            .map(|p| p.weight.shape().dim(0).unwrap_or(0))
            .unwrap_or(0);
        s.qlora.push(dev.zeros(
            &Shape::new(vec![batch, qlora_w.max(1)]),
            grim_tensor::DType::F32,
        )?);
        s.qf.push(dev.zeros(
            &Shape::new(vec![batch, nh * (nope + rope_d)]),
            grim_tensor::DType::F32,
        )?);
        s.qn.push(dev.zeros(&Shape::new(vec![batch, nh * nope]), grim_tensor::DType::F32)?);
        s.qp.push(dev.zeros(
            &Shape::new(vec![batch, nh * rope_d]),
            grim_tensor::DType::F32,
        )?);
        s.qabs
            .push(dev.zeros(&Shape::new(vec![batch, nh * rank]), grim_tensor::DType::F32)?);
        s.qattn_latent
            .push(dev.zeros(&Shape::new(vec![batch, nh * rank]), grim_tensor::DType::F32)?);
        s.kvstage.push(dev.zeros(
            &Shape::new(vec![batch, rank + rope_d]),
            grim_tensor::DType::F32,
        )?);
        s.ckv
            .push(dev.zeros(&Shape::new(vec![batch, rank]), grim_tensor::DType::F32)?);
        s.kpe
            .push(dev.zeros(&Shape::new(vec![batch, rope_d]), grim_tensor::DType::F32)?);
        s.latent.push(dev.zeros(
            &Shape::new(vec![batch, rank + rope_d]),
            grim_tensor::DType::F32,
        )?);
        s.attn
            .push(dev.zeros(&Shape::new(vec![batch, nh * vd]), grim_tensor::DType::F32)?);
        s.shg
            .push(dev.zeros(&Shape::new(vec![batch, moe_inter]), grim_tensor::DType::F32)?);
        s.shu
            .push(dev.zeros(&Shape::new(vec![batch, moe_inter]), grim_tensor::DType::F32)?);
        s.sha
            .push(dev.zeros(&Shape::new(vec![batch, moe_inter]), grim_tensor::DType::F32)?);
        s.sho
            .push(dev.zeros(&Shape::new(vec![batch, hidden]), grim_tensor::DType::F32)?);
        // Pre-dequant the F16 hc_fn weights to F32 device tensors (once).
        for (hc_mod, sink) in [
            (&layer.attn_hc, &mut s.hc_fn_attn),
            (&layer.ffn_hc, &mut s.hc_fn_ffn),
        ] {
            let w = hc_mod.hc_fn_weight();
            let v = w.to_vec_f32()?;
            let st = dev.from_cpu(&v, w.shape(), grim_tensor::DType::F32)?;
            sink.push(grim_tensor::Tensor::new(
                Arc::from(st),
                w.shape().clone(),
                grim_tensor::DType::F32,
                w.provenance().clone(),
                w.device().clone(),
            ));
        }
        let (base, scale) = layer.attn_hc.gate_params();
        s.hc_base.push(dev.from_cpu(
            base,
            &Shape::new(vec![base.len()]),
            grim_tensor::DType::F32,
        )?);
        s.hc_scale
            .push(dev.from_cpu(scale, &Shape::new(vec![3]), grim_tensor::DType::F32)?);
        let (fbase, fscale) = layer.ffn_hc.gate_params();
        s.hc_base_ffn.push(dev.from_cpu(
            fbase,
            &Shape::new(vec![fbase.len()]),
            grim_tensor::DType::F32,
        )?);
        s.hc_scale_ffn
            .push(dev.from_cpu(fscale, &Shape::new(vec![3]), grim_tensor::DType::F32)?);
    }
    Ok(s)
}

impl DecodeGraphModel for Xing40 {
    fn get_or_create_decode_graph(&self, max_ctx: usize, batch: usize) -> Result<DecodeGraph> {
        if !decode_graph_enabled() {
            return Err(grim_core::error::Error::Backend(
                "decode graph disabled by env".into(),
            ));
        }
        let dev = dev_for_xing40(self)?;
        let stream = dev
            .get_stream_from_pool(0)
            .ok_or_else(|| grim_core::error::Error::Backend("no stream in pool".into()))?;
        let cfg = &self.cfg;
        let latent_dim = cfg.kv_lora_rank + cfg.qk_rope_head_dim;
        // The latent cache rides the k_arena slot (width = latent_dim) so the
        // generic eager-kv seeding works unmodified; v_arena is dead weight.
        let buffers = DecodeGraphBuffers::allocate_with_mla(
            &dev,
            self.layers.len(),
            cfg.hidden_size,
            cfg.num_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim),
            cfg.num_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim),
            latent_dim,
            latent_dim,
            cfg.intermediate_size,
            max_ctx.max(1),
            cfg.vocab_size.max(1),
            cfg.num_heads,
            batch,
            cfg.n_routed_experts,
            cfg.num_experts_per_tok,
            0,
            0,
            latent_dim,
        )
        .map_err(|e| grim_core::error::Error::Backend(format!("graph pool alloc: {e}")))?;
        // Capture retries call this again; the scratch is reusable as-is.
        if self.graph_scratch.get().is_none() {
            let built = std::sync::Mutex::new(build_xing_scratch(self, &dev, batch)?);
            let _ = self.graph_scratch.set(built);
        }
        if std::env::var_os("GRIM_XING_TRACE").is_some() {
            let kinds: Vec<String> = self
                .layers
                .iter()
                .enumerate()
                .map(|(i, l)| {
                    let k = if l.mlp.is_some() {
                        "dense"
                    } else if l.moe.is_some() {
                        if l.moe.as_ref().unwrap().shared_experts.is_some() { "moe+shared" } else { "moe" }
                    } else { "none" };
                    format!("{i}:{k}")
                })
                .collect();
            eprintln!("[xing-graph] layer kinds {}", kinds.join(" "));
        }
        // Prewarm the MoE native K-quant pointer arrays (zero weight VRAM;
        // the per-expert banks ARE the model's own weights). Tiny H2D of
        // host-owned data — done here, outside capture.
        for layer in &self.layers {
            if let Some(ref moe) = layer.moe {
                let experts: Vec<crate::shared_moe::MoeExpert> = moe
                    .experts
                    .iter()
                    .map(|e| crate::shared_moe::MoeExpert {
                        gate: e.w1.clone(),
                        up: e.w3.clone(),
                        down: e.w2.clone(),
                    })
                    .collect();
                if let Err(e) = crate::shared_moe::ensure_kq_native(
                    dev.ordinal(),
                    &experts,
                    &moe.charon_cache,
                ) {
                    eprintln!("[xing40-graph] kq-native prewarm failed: {e}");
                }
            }
        }
        // Prewarm the MoE WhiteCrow stacks (GRIM_MOE_NATIVE_WHITECROW) OUTSIDE
        // any capture bracket: the conversion D2Hs, which would poison a
        // capture. The stacks are PER LAYER and must ALL be resident during a
        // replay, so the whole batch has to fit in free VRAM — measure the
        // first build, extrapolate over the MoE layer count, and refuse (fail
        // closed to the f32 path / eager) when it cannot. Building all 38
        // xing40 layers needs ~12.9 GiB beside a 13.3 GiB model: it does not
        // fit on this card, and half-building would fault mid-replay.
        let mut moe_layers: Vec<&crate::xing40::Xing40Block> =
            self.layers.iter().filter(|l| l.moe.is_some()).collect();
        let n_moe = moe_layers.len();
        if n_moe > 0 {
            let first = moe_layers.remove(0);
            if let Some(ref moe) = first.moe {
                let experts: Vec<crate::shared_moe::MoeExpert> = moe
                    .experts
                    .iter()
                    .map(|e| crate::shared_moe::MoeExpert {
                        gate: e.w1.clone(),
                        up: e.w3.clone(),
                        down: e.w2.clone(),
                    })
                    .collect();
                match crate::shared_moe::ensure_whitecrow_scratch(
                    dev.ordinal(),
                    &experts,
                    &moe.charon_cache,
                ) {
                    Ok(first_stacks) => {
                        // Per-layer resident bytes: (gate + up + down) blobs,
                        // each (stride x n_routed_experts); gate and up share
                        // a stride.
                        let per_layer = (first_stacks.gate_stride * 2
                            + first_stacks.down_stride)
                            * self.cfg.n_routed_experts as u64;
                        let projected = per_layer * n_moe as u64;
                        let free = grim_backend_rocm::device::capability_profiler::free_device_memory(dev.ordinal()).unwrap_or(0);
                        if projected > free {
                            eprintln!(
                                "[xing40-graph] whitecrow stacks need {projected:.0} B across {n_moe} MoE layers, {free} B free; skipping prewarm (fail closed)"
                            );
                        } else {
                            for layer in moe_layers {
                                if let Some(ref moe) = layer.moe {
                                    let experts: Vec<crate::shared_moe::MoeExpert> = moe
                                        .experts
                                        .iter()
                                        .map(|e| crate::shared_moe::MoeExpert {
                                            gate: e.w1.clone(),
                                            up: e.w3.clone(),
                                            down: e.w2.clone(),
                                        })
                                        .collect();
                                    let _ = crate::shared_moe::ensure_whitecrow_scratch(
                                        dev.ordinal(),
                                        &experts,
                                        &moe.charon_cache,
                                    );
                                }
                            }
                        }
                    }
                    Err(e) => eprintln!(
                        "[xing40-graph] whitecrow prewarm unavailable ({e}); capture will use the f32 path or fail closed"
                    ),
                }
            }
        }
        Ok(DecodeGraph::new(&dev, buffers, stream))
    }

    fn forward_capture(&self, graph: &mut DecodeGraph, token_id: u32) -> Result<()> {
        if !decode_graph_enabled() {
            return Err(grim_core::error::Error::Backend(
                "decode graph disabled".into(),
            ));
        }
        let dev = dev_for_xing40(self)?;
        if !graph.capturing {
            crate::lfm2_graph::write_embedding_to_buffer(
                &dev,
                &graph.buffers.token_ids_dev,
                token_id,
            )?;
        }
        let mut scratch_guard = self
            .graph_scratch
            .get()
            .ok_or_else(|| grim_core::error::Error::Backend("xing40 graph scratch missing".into()))?
            .lock()
            .map_err(|_| {
                grim_core::error::Error::Backend("xing40 graph scratch poisoned".into())
            })?;
        let scratch = &mut *scratch_guard;
        let buffers = &mut graph.buffers;
        let batch = buffers.batch.max(1);
        let hidden = self.cfg.hidden_size;
        let hc = scratch.hc;
        let flat = scratch.flat;
        let cfg = &self.cfg;

        // Aliasing probe (env-gated): the graph writes NaN over sin[0] on every
        // replay, and sin is read-only in this body. A pool buffer that OVERLAPS
        // sin[0] would explain it — the node writing that buffer lands in sin.
        if std::env::var_os("GRIM_XING_TRACE").is_some() {
            let mut spans: Vec<(u64, u64, String)> = Vec::new();
            let mut add = |st: &dyn BackendStorage, name: String| {
                if let Some(r) = st.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>() {
                    if let Some(p) = r.device_ptr_u64() {
                        let n = r.shape().elem_count().max(1) as u64 * 4;
                        spans.push((p, p + n, name));
                    }
                }
            };
            macro_rules! pool {
                ($v:expr, $n:literal) => {
                    if let Some(b) = $v.first() {
                        add(b, format!("{}[0]", $n));
                    }
                };
            }
            macro_rules! scr {
                ($v:expr, $n:literal) => {
                    if let Some(b) = $v.first() {
                        add(b.as_ref(), format!("{}[0]", $n));
                    }
                };
            }
            pool!(buffers.layer_input, "pool.layer_input");
            pool!(buffers.layer_output, "pool.layer_output");
            pool!(buffers.q_buf, "pool.q_buf");
            pool!(buffers.k_buf, "pool.k_buf");
            pool!(buffers.v_buf, "pool.v_buf");
            pool!(buffers.attn_out_buf, "pool.attn_out_buf");
            pool!(buffers.gate_up_buf, "pool.gate_up_buf");
            pool!(buffers.gate_buf, "pool.gate_buf");
            pool!(buffers.up_buf, "pool.up_buf");
            pool!(buffers.activated_buf, "pool.activated_buf");
            pool!(buffers.norm_buf, "pool.norm_buf");
            pool!(buffers.act_q81_buf, "pool.act_q81_buf");
            pool!(buffers.k_arena, "pool.k_arena");
            scr!(scratch.sin, "scratch.sin");
            scr!(scratch.sout, "scratch.sout");
            scr!(scratch.hcn, "scratch.hcn");
            scr!(scratch.proj, "scratch.proj");
            scr!(scratch.pre, "scratch.pre");
            scr!(scratch.col, "scratch.col");
            scr!(scratch.attn, "scratch.attn");
            scr!(scratch.qabs, "scratch.qabs");
            let mut overlaps = Vec::new();
            for i in 0..spans.len() {
                for j in i + 1..spans.len() {
                    let (a0, a1, an) = &spans[i];
                    let (b0, b1, bn) = &spans[j];
                    if a0 < b1 && b0 < a1 {
                        overlaps.push(format!("{an} [{a0:#x},{a1:#x}) OVERLAPS {bn} [{b0:#x},{b1:#x})"));
                    }
                }
            }
            eprintln!("[xing-graph] span count {}; OVERLAPS: {}", spans.len(), overlaps.len());
            for o in overlaps.iter().take(12) {
                eprintln!("[xing-graph]   {o}");
            }
        }

        // Embedding gather → seed streams of layer 0.
        let w_emb = dst_downcast(self.tok_embeddings.weight.storage().as_ref())?;
        dev.launch_embedding_gather_dev_idx(
            w_emb,
            &buffers.layer_input[0],
            &buffers.token_ids_dev,
            hidden,
            batch * hidden,
        )
        .map_err(|e| grim_core::error::Error::Backend(format!("embedding gather: {e}")))?;
        // Broadcast the gathered row across the hc streams (the same column
        // writes seed_streams_device performs; write_cols is a pure kernel).
        for h in 0..hc {
            dev.write_cols(
                scratch.sin[0].as_mut(),
                flat,
                h * hidden,
                &buffers.layer_input[0],
                batch,
                hidden,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("hc seed: {e}")))?;
        }
        // Snapshot layer 0's input stream before the last layer overwrites it.
        {
            let dst = as_rocm(scratch.sin0_snap[0].as_ref())?;
            dev.copy_slice_into(dst, scratch.sin[0].as_ref(), 0, flat)?;
        }
        let h_shape = Shape::new(vec![batch, hidden]);
        let n_layers = self.layers.len();
        // Fault bisection: capture only the first N layers (default: all).
        // With 0 the graph holds just the embedding gather + hc fan-out.
        let cap_layers: usize = std::env::var("GRIM_XING_CAPTURE_LAYERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(usize::MAX);
        for (i, layer) in self.layers.iter().enumerate().take(cap_layers) {
            let act = &buffers.act_q81_buf[i];
            // The running stream for this layer. It MUST be sin[i]: the
            // write-back at the end of a layer publishes into sin[i+1] (and
            // the last layer into sin[0] for the head), so sin[i] is the slot
            // this layer reads. It used to read sout[i-1] here while every
            // op below used sout[i] — two different buffers for one value, and
            // sout was never written, so every layer after the first consumed
            // uninitialized memory and the last layer published that NaN back
            // into sin[0].
            let sin: &GBox = &scratch.sin[i];
            // Staging for the attention write-back (see the aliasing note
            // there): sin[i] is this layer's input, sout[i] is the stream
            // after the attention contribution has been folded in.
            let sout: &GBox = &scratch.sout[i];

            // ── attn hc: gates over the raw stream state ──
            dev.rms_norm_into(
                sin.as_ref(),
                &**layer.attn_hc.input_norm.weight.storage(),
                layer.attn_hc.input_norm.eps,
                as_rocm(scratch.hcn[i].as_ref())?,
                &Shape::new(vec![batch, flat]),
            )
            .map_err(grim_core::error::Error::Tensor)?;
            linear_into(
                &dev,
                as_rocm(scratch.hcn[i].as_ref())?,
                &scratch.hc_fn_attn[i],
                as_rocm(scratch.proj[i].as_ref())?,
                act,
            )?;
            let (iters, geps, cmin, cmax) = layer.attn_hc.gate_consts();
            dev.mhc_gates_launch_into(
                as_rocm(scratch.proj[i].as_ref())?,
                scratch.hc_base[i].as_ref(),
                scratch.hc_scale[i].as_ref(),
                as_rocm(scratch.pre[i].as_ref())?,
                as_rocm(scratch.post[i].as_ref())?,
                as_rocm(scratch.comb[i].as_ref())?,
                batch,
                hc,
                iters,
                geps,
                cmin,
                cmax,
            )?;
            // Snapshot the attention gates before the FFN overwrites them.
            // Capture-safe: a device-to-device copy on the capturing stream,
            // no allocation and no sync.
            {
                let pdst = as_rocm(scratch.attn_post_snap[i].as_ref())?;
                dev.copy_slice_into(pdst, scratch.post[i].as_ref(), 0, hc * batch)?;
                let cdst = as_rocm(scratch.attn_comb_snap[i].as_ref())?;
                dev.copy_slice_into(cdst, scratch.comb[i].as_ref(), 0, hc * hc * batch)?;
            }
                        // CAVEAT: this is a WARMUP pass. run.rs runs warmups on UNSEEDED
            // buffers and then re-seeds, so only values UPSTREAM of the
            // attention (embedding, collapse, attn gates, q path) are
            // comparable to an eager run. Anything downstream of the KV
            // attention reads a stale arena and legitimately differs — two
            // rounds of chasing that were wasted before this was written down.
            if i == 0 && !graph.capturing && std::env::var_os("GRIM_XING_TRACE").is_some() {
                let _ = grim_backend_rocm::device::helpers::hip_stream_synchronize(graph.stream);
                for (lbl, b) in [("post", scratch.post.first()), ("comb", scratch.comb.first())] {
                    if let Some(b) = b {
                        if let Ok(st) = as_rocm(b.as_ref()) {
                            if let Ok(v) = st.to_cpu_vec_f32() {
                                if lbl == "comb" {
                                    let rows: Vec<f32> = (0..hc)
                                        .map(|h| (0..hc).map(|i| v[h * hc + i]).sum())
                                        .collect();
                                    eprintln!("[xing-graph] WARM comb rowsums {rows:?}");
                                } else {
                                    eprintln!("[xing-graph] WARM post {v:?}");
                                }
                            }
                        }
                    }
                }
            }
dev.launch_hc_collapse_step(
                // The attention branch reads the layer INPUT. `sout[i]` is the
                // staging buffer the attention write-back fills further down,
                // so it is still zeros at this point.
                sin.as_ref(),
                as_rocm(scratch.pre[i].as_ref())?,
                as_rocm(scratch.col[i].as_ref())?,
                hc,
                hidden,
            )?;
            dev.rms_norm_into(
                as_rocm(scratch.col[i].as_ref())?,
                &**layer.attn_norm.weight.storage(),
                layer.attn_norm.eps,
                as_rocm(scratch.col[i].as_ref())?,
                &h_shape,
           )?;
            // The FFN branch overwrites `col` later in this layer, so the
            // attention-branch value is only observable here. Dump during the
            // WARMUP pass (capturing=false => syncing is legal and the values
            // are real); during the recorded pass this buffer is stale.
            if i == 0 && !graph.capturing && std::env::var_os("GRIM_XING_TRACE").is_some() {
                let _ = grim_backend_rocm::device::helpers::hip_stream_synchronize(graph.stream);
                if let Ok(st) = as_rocm(scratch.col[0].as_ref()) {
                    if let Ok(v) = st.to_cpu_vec_f32() {
                        let r = (v.iter().map(|x| x * x).sum::<f32>() / v.len().max(1) as f32)
                            .sqrt();
                        eprintln!(
                            "[xing-graph] WARM attn_in L{i}: n {} rms {r:.4e} head {:?}",
                            v.len(),
                            &v[..v.len().min(4)]
                        );
                    }
                }
            }

            if std::env::var("GRIM_XING_STOP").ok().as_deref() == Some("pre") {
                return Ok(());
            }
            // ── MLA ──
            let sa = &layer.self_attn;
            let rank = cfg.kv_lora_rank;
            let nope = cfg.qk_nope_head_dim;
            let rope_d = cfg.qk_rope_head_dim;
            let vd = cfg.v_head_dim;
            let nh = cfg.num_heads;
            if let (Some(qa), Some(qn_norm), Some(qb)) =
                (&sa.q_a_proj, &sa.q_a_layernorm, &sa.q_b_proj)
            {
                linear_into(
                    &dev,
                    scratch.col[i].as_ref(),
                    qa.weight(),
                    as_rocm(scratch.qlora[i].as_ref())?,
                    act,
                )?;
                dev.rms_norm_into(
                    as_rocm(scratch.qlora[i].as_ref())?,
                    &**qn_norm.weight.storage(),
                    qn_norm.eps,
                    as_rocm(scratch.qlora[i].as_ref())?,
                    &Shape::new(vec![batch, scratch.qlora[i].shape().dims()[1]]),
                )
                .map_err(grim_core::error::Error::Tensor)?;
                linear_into(
                    &dev,
                    scratch.qlora[i].as_ref(),
                    qb.weight(),
                    as_rocm(scratch.qf[i].as_ref())?,
                    act,
                )?;
            } else if let Some(ref q_direct) = sa.q_proj_direct {
                linear_into(
                    &dev,
                    scratch.col[i].as_ref(),
                    q_direct.weight(),
                    as_rocm(scratch.qf[i].as_ref())?,
                    act,
                )?;
            }
            dev.launch_xing_q_split(
                scratch.qf[i].as_ref(),
                as_rocm(scratch.qn[i].as_ref())?,
                as_rocm(scratch.qp[i].as_ref())?,
                nh,
                nope,
                rope_d,
            )?;
            dev.rope_dev_base_into(
                as_rocm(scratch.qp[i].as_ref())?,
                &buffers.pos_dev,
                as_rocm(scratch.qp[i].as_ref())?,
                &scratch.rope_cfg,
                &Shape::new(vec![batch, nh, rope_d]),
                nh,
                1,
            )
            .map_err(grim_core::error::Error::Tensor)?;
            // A/B the q rope: the graph ropes the whole [1, nh, rope_d] slab
            // in place with rope_dev_base_into; eager ropes ONE HEAD AT A TIME
            // with dev.rope and writes each result back per head. Two
            // implementations of the same rotation - if one folds the YaRN
            // mscale and the other does not, every attention output shifts.
            // `qp` now holds the graph's ANSWER; the pre-rope bytes are
            // re-derived from `qf` through the same q_split the capture used.
            // Warmup only: allocation is capture-poison.
            if i == 0 && !graph.capturing && std::env::var_os("GRIM_XING_TRACE").is_some() {
                let _ = grim_backend_rocm::device::helpers::hip_stream_synchronize(graph.stream);
                let Ok(got) = as_rocm(scratch.qp[i].as_ref())?.to_cpu_vec_f32() else { return Ok(()) };
                let pre_shape = Shape::new(vec![1, nh * rope_d]);
                let nope_shape = Shape::new(vec![1, nh * nope]);
                let t_pre = dev.zeros(&pre_shape, grim_tensor::DType::F32);
                let t_nope = dev.zeros(&nope_shape, grim_tensor::DType::F32);
                if let (Ok(t_pre), Ok(t_nope)) = (t_pre, t_nope) {
                    let t_pre_r: &grim_backend_rocm::RocmStorage =
                        t_pre.as_any().downcast_ref().expect("rocm");
                    let t_nope_r: &grim_backend_rocm::RocmStorage =
                        t_nope.as_any().downcast_ref().expect("rocm");
                    if dev
                        .launch_xing_q_split(
                            as_rocm(scratch.qf[i].as_ref())?,
                            t_nope_r,
                            t_pre_r,
                            nh,
                            nope,
                            rope_d,
                        )
                        .is_ok()
                    {
                        if let Ok(pre) = t_pre.to_cpu_vec_f32() {
                            let pos = vec![buffers.current_pos];
                            let head_shape = Shape::new(vec![1, 1, rope_d]);
                            if let Ok(mut refbuf) = dev.zeros(&pre_shape, grim_tensor::DType::F32) {
                                let mut ok = true;
                                for h in 0..nh {
                                    let hs = Shape::new(vec![1, rope_d]);
                                    let head_src = match dev.from_cpu(
                                        &pre[h * rope_d..(h + 1) * rope_d],
                                        &hs,
                                        grim_tensor::DType::F32,
                                    ) {
                                        Ok(v) => v,
                                        Err(_) => { ok = false; break; }
                                    };
                                    let mut ho = match dev.zeros(&head_shape, grim_tensor::DType::F32) {
                                        Ok(v) => v,
                                        Err(_) => { ok = false; break; }
                                    };
                                    match grim_tensor::AttentionOps::rope(dev.as_ref(), head_src.as_ref(), &pos, &scratch.rope_cfg, &head_shape) {
                                        Ok((t, _)) => {
                                            if dev.write_cols(
                                                refbuf.as_mut(), nh * rope_d, h * rope_d,
                                                t.as_ref(), 1, rope_d,
                                            ).is_err() { ok = false; break; }
                                        }
                                        _ => { ok = false; break; }
                                    }
                                    let _ = &mut ho;
                                }
                                if ok {
                                    dev.synchronize();
                                    if let Ok(b) = refbuf.to_cpu_vec_f32() {
                                        let md = got.iter().zip(b.iter())
                                            .map(|(x, y)| (x - y).abs())
                                            .fold(0.0f32, f32::max);
                                        let rg = (got.iter().map(|x| x * x).sum::<f32>()
                                            / got.len().max(1) as f32).sqrt();
                                        eprintln!("[xing-graph] QROPE in_place_vs_perhead: max_abs {md:.4e} rel {:.4e} rms {rg:.4e}", md as f64 / (rg as f64).max(1e-12));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Absorb W_UK into the nope plane (F32 bank cache, [nh*rank, nope]).
            if let Some(w_uk) = sa.w_kc_device(0)? {
                dev.launch_xing_q_absorb(
                    as_rocm(scratch.qn[i].as_ref())?,
                    w_uk,
                    as_rocm(scratch.qabs[i].as_ref())?,
                    nh,
                    rank,
                    nope,
                )?;
            } else {
                return Err(grim_core::error::Error::Backend(
                    "xing40 graph: no device W_UK cache".into(),
                ));
            }
            // KV latent: project, norm c_kv, rope k_pe, pack, append.
            linear_into(
                &dev,
                scratch.col[i].as_ref(),
                sa.kv_a_proj.weight(),
                as_rocm(scratch.kvstage[i].as_ref())?,
                act,
            )?;
            dev.launch_xing_q_split(
                scratch.kvstage[i].as_ref(),
                as_rocm(scratch.ckv[i].as_ref())?,
                as_rocm(scratch.kpe[i].as_ref())?,
                1,
                rank,
                rope_d,
            )?;
            dev.rms_norm_into(
                as_rocm(scratch.ckv[i].as_ref())?,
                &**sa.kv_a_layernorm.weight.storage(),
                sa.kv_a_layernorm.eps,
                as_rocm(scratch.ckv[i].as_ref())?,
                &Shape::new(vec![batch, rank]),
            )
            .map_err(grim_core::error::Error::Tensor)?;
            dev.rope_dev_base_into(
                as_rocm(scratch.kpe[i].as_ref())?,
                &buffers.pos_dev,
                as_rocm(scratch.kpe[i].as_ref())?,
                &scratch.rope_cfg,
                &Shape::new(vec![batch, 1, rope_d]),
                1,
                1,
            )
            .map_err(grim_core::error::Error::Tensor)?;
            dev.launch_xing_pack_latent(
                as_rocm(scratch.ckv[i].as_ref())?,
                as_rocm(scratch.kpe[i].as_ref())?,
                as_rocm(scratch.latent[i].as_ref())?,
                rank,
                rope_d,
            )?;
            grim_backend_rocm::launch_kv_append_batch(
                &dev,
                &buffers.k_arena[i],
                as_rocm(scratch.latent[i].as_ref())?,
                &buffers.pos_dev,
                rank + rope_d,
                1,
                batch,
                buffers.max_ctx * (rank + rope_d),
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("latent kv_append: {e}")))?;
            // Latent-space attention over the LIVE length (pos_dev + 1); the
            // normalized latent lands in qabs and W_UV runs as the per-head
            // GEMV into attn.
            // The softmax denominator is handed to the kernel rather than
            // applied by a separate elementwise pass over q. The kernel's
            // default is 1/sqrt(rank + rope_d); this model folds a YaRN
            // attention_factor^2 against the *nope + rope* width instead.
            // Issuing that extra `grim_mul_scalar` inside the capture bracket
            // hangs the capture dead - bisected with GRIM_KQ_PRESCALE: it hangs
            // even writing in place into qabs/qp (mode 4), so it is the launch
            // in the bracket, not the destination buffers.
            let mscale = cfg.rope_yarn.map(|y| y.attention_factor).unwrap_or(1.0f32);
            let inv_sqrt_d = mscale * mscale / ((nope + rope_d) as f32).sqrt();
            if i == 0 && !graph.capturing && std::env::var_os("GRIM_XING_TRACE").is_some() {
                eprintln!(
                    "[xing-graph] SCALE mscale {mscale:.6} nope {nope} rope_d {rope_d} inv_sqrt_d {inv_sqrt_d:.6e} cfg_yarn {:?}",
                    cfg.rope_yarn.is_some()
                );
            }
            if std::env::var("GRIM_XING_STOP").ok().as_deref() == Some("append") {
                return Ok(());
            }
            dev.launch_mla_absorbed_decode_scaled(
                as_rocm(scratch.qabs[i].as_ref())?,
                as_rocm(scratch.qp[i].as_ref())?,
                &buffers.k_arena[i],
                None,
                as_rocm(scratch.qattn_latent[i].as_ref())?,
                nh,
                rank,
                rope_d,
                vd,
                buffers.max_ctx.max(1),
                0,
                0,
                Some(&buffers.pos_dev),
                inv_sqrt_d,
            )
            .map_err(|e| grim_core::error::Error::Backend(format!("mla_absorbed_decode: {e}")))?;
            if std::env::var("GRIM_XING_STOP").ok().as_deref() == Some("mlaout") {
                return Ok(());
            }
            // A/B the value up-projection: `xing_q_absorb` (what the graph
            // uses) vs a per-head matmul (what eager uses) over the SAME
            // latent and the SAME w_vc. Both sides are computed here, so
            // neither depends on a buffer that has not been written yet.
            // Warmup only: allocating is capture-poison.
            if i == 0 && !graph.capturing && std::env::var_os("GRIM_XING_TRACE").is_some() {
                if let Some(w_vc) = sa.w_vc_device(0)? {
                    let _ = grim_backend_rocm::device::helpers::hip_stream_synchronize(graph.stream);
                    let lat = as_rocm(scratch.qattn_latent[i].as_ref())?;
                    let w_r = as_rocm(w_vc)?;
                    let attn_shape = Shape::new(vec![1, nh * vd]);
                    if let (Ok(mut mm), Ok(xg)) = (
                        dev.zeros(&attn_shape, grim_tensor::DType::F32),
                        dev.zeros(&attn_shape, grim_tensor::DType::F32),
                    ) {
                        let mm_m: &mut dyn grim_tensor::BackendStorage = mm.as_mut();
                        let xg_r: &grim_backend_rocm::RocmStorage =
                            xg.as_any().downcast_ref().expect("rocm");
                        dev.launch_xing_q_absorb(lat, w_r, xg_r, nh, vd, rank).ok();
                        // Each head's [1, vd] must land at column h*vd, not
                        // overwrite the whole buffer - matmul_into writes the
                        // full `out`, so stage per head and write_cols it in.
                        let head_shape = Shape::new(vec![1, vd]);
                        let mut ok = dev.zeros(&head_shape, grim_tensor::DType::F32).is_ok();
                        if ok {
                            for h in 0..nh {
                                let tmp = match dev.zeros(&head_shape, grim_tensor::DType::F32) {
                                    Ok(t) => t,
                                    Err(_) => { ok = false; break; }
                                };
                                let tmp_r: &grim_backend_rocm::RocmStorage =
                                    tmp.as_any().downcast_ref().expect("rocm");
                                let lat_h = dev.narrow_cols(
                                    lat, nh * rank, h * rank, 1, rank,
                                    &Shape::new(vec![1, rank]),
                                );
                                let w_h = dev.narrow_rows(
                                    w_r, h * vd, vd, rank,
                                    &Shape::new(vec![vd, rank]),
                                );
                                if let (Ok((l, _)), Ok((w, _))) = (lat_h, w_h) {
                                    if dev.matmul_into(l.as_ref(), w.as_ref(), tmp_r).is_err() {
                                        ok = false;
                                        break;
                                    }
                                    if dev.write_cols(mm_m, nh * vd, h * vd, tmp_r, 1, vd).is_err() {
                                        ok = false;
                                        break;
                                    }
                                } else {
                                    ok = false;
                                    break;
                                }
                            }
                        }
                        if ok {
                            dev.synchronize();
                            if let (Ok(a), Ok(b)) = (xg.to_cpu_vec_f32(), mm.to_cpu_vec_f32()) {
                                let md = a.iter().zip(b.iter())
                                    .map(|(x, y)| (x - y).abs())
                                    .fold(0.0f32, f32::max);
                                let ra = (a.iter().map(|x| x * x).sum::<f32>()
                                    / a.len().max(1) as f32).sqrt();
                                eprintln!("[xing-graph] ABSORB xing_vs_matmul: max_abs {md:.4e} rel {:.4e} rms {ra:.4e}", md / ra.max(1e-12));
                            }
                        }
                    }
                }
            }
            if let Some(w_vc) = sa.w_vc_device(0)? {
            if std::env::var("GRIM_XING_STOP").ok().as_deref() == Some("mla") {
                return Ok(());
            }
                dev.launch_xing_q_absorb(
                    as_rocm(scratch.qattn_latent[i].as_ref())?,
                    w_vc,
                    as_rocm(scratch.attn[i].as_ref())?,
                    nh,
                    vd,
                    rank,
                )?;
            } else {
                return Err(grim_core::error::Error::Backend(
                    "xing40 graph: no device W_VC cache".into(),
                ));
            }
            if std::env::var("GRIM_XING_STOP").ok().as_deref() == Some("absorb") {
                return Ok(());
            }
            linear_into(
                &dev,
                scratch.attn[i].as_ref(),
                sa.o_proj.weight(),
                &buffers.norm_buf[i],
                act,
            )?;
            // Snapshot o_proj's output before the FFN reuses norm_buf.
            {
                let dst = as_rocm(scratch.attn_out_snap[i].as_ref())?;
                dev.copy_slice_into(dst, &buffers.norm_buf[i], 0, hidden)?;
            }
            // Host oracle for the Q4_K o_proj. `fused_quant_gemm` cannot be
            // A/B'd in-process (it deadlocks the warmup), so dequantize the
            // weight on the host and compute the GEMV directly: that is ground
            // truth for BOTH paths, and tells us whether the graph's result is
            // right or merely different. First 512 outputs only - a full
            // 3584x4096 host GEMV is minutes in a debug build.
            if i == 0 && !graph.capturing && std::env::var_os("GRIM_XING_TRACE").is_some() {
                let _ = grim_backend_rocm::device::helpers::hip_stream_synchronize(graph.stream);
                let n_out = 512.min(hidden);
                if let (Ok(a), Ok(w), Ok(nb)) = (
                    as_rocm(scratch.attn[i].as_ref())?.to_cpu_vec_f32(),
                    sa.o_proj.weight().to_vec_f32(),
                    as_rocm(&buffers.norm_buf[i]).unwrap().to_cpu_vec_f32(),
                ) {
                    let k = a.len();
                    let mut max_abs = 0.0f32;
                    let mut ref_rms = 0.0f64;
                    for j in 0..n_out {
                        let mut acc = 0.0f32;
                        for c in 0..k {
                            acc += a[c] * w[j * k + c];
                        }
                        ref_rms += (acc as f64) * (acc as f64);
                        let d = (acc - nb[j]).abs();
                        if d > max_abs {
                            max_abs = d;
                        }
                    }
                    ref_rms = (ref_rms / n_out as f64).sqrt();
                    let got_rms =
                        (nb[..n_out].iter().map(|x| x * x).sum::<f32>() / n_out as f32).sqrt();
                    eprintln!(
                        "[xing-graph] OPROJ vs HOST oracle: max_abs {max_abs:.4e} rel {:.4e} ref_rms {ref_rms:.4e} got_rms {got_rms:.4e}",
                        max_abs as f64 / ref_rms.max(1e-12)
                    );
                    // Same product through the fused-dequant leg, which takes
                    // F32 A instead of quantizing it to Q8_1. If this lands at
                    // ~1e-6 while the dot4 leg sits at ~6e-3, the Q8_1
                    // ACTIVATION quantization is the entire error.
                    if let (Ok(attn_r), Ok(o_ws2)) = (
                        as_rocm(scratch.attn[i].as_ref()),
                        as_rocm(sa.o_proj.weight().storage().as_ref()),
                    ) {
                        if let Ok(fb) = dev.zeros(&Shape::new(vec![1, hidden]), grim_tensor::DType::F32) {
                            let fb_r: &grim_backend_rocm::RocmStorage =
                                fb.as_any().downcast_ref().expect("rocm");
                            if dev
                                .launch_fused_dequant_gemm_q4k_for_ab(attn_r, o_ws2, fb_r, 1, hidden, nh * vd)
                                .is_ok()
                            {
                                dev.synchronize();
                                if let Ok(b) = fb.to_cpu_vec_f32() {
                                    let mut md2 = 0.0f32;
                                    for j in 0..n_out.min(hidden) {
                                        let mut acc = 0.0f32;
                                        for c in 0..(nh * vd) {
                                            acc += a[c] * w[j * (nh * vd) + c];
                                        }
                                        let d = (acc - b[j]).abs();
                                        if d > md2 {
                                            md2 = d;
                                        }
                                    }
                                    eprintln!(
                                        "[xing-graph] OPROJ fused-dequant vs HOST oracle: max_abs {md2:.4e} rel {:.4e}",
                                        md2 as f64 / ref_rms.max(1e-12)
                                    );
                                }
                            }
                        }
                    }
                }
            }
            if std::env::var("GRIM_XING_STOP").ok().as_deref() == Some("oproj") {
                return Ok(());
            }
            // The out buffer must NOT alias the streams input. The kernel's
            // thread (h,d) writes element (h,d) while EVERY thread (i,d)
            // reads element (h,d), so in-place corrupts the row another head
            // is still reading — measured as all-NaN streams on every replay.
            // sout[i] is that staging buffer; the FFN branch below reads it.
            dev.launch_hc_write_back_step(
                &buffers.norm_buf[i],
                as_rocm(scratch.post[i].as_ref())?,
                as_rocm(scratch.comb[i].as_ref())?,
                sin.as_ref(),
                as_rocm(scratch.sout[i].as_ref())?,
                hc,
                hidden,
            )?;

            if std::env::var("GRIM_XING_STOP").ok().as_deref() == Some("attn") {
                return Ok(());
            }
            // HOST ORACLE for the attention write-back. Placed HERE on purpose:
            // `post`/`comb` are overwritten by the FFN's gate call moments
            // later, so an oracle further down compares the attention
            // write-back against the FFN's gates and reports garbage (it
            // reported rel 4.5 that way before this note existed).
            //   out[h*hidden+d] = post[h]*y[d] + sum_i comb[h*hc+i]*s[i*hidden+d]
            if i == 0 && !graph.capturing && std::env::var_os("GRIM_XING_TRACE").is_some() {
                let _ = grim_backend_rocm::device::helpers::hip_stream_synchronize(graph.stream);
                let sv = as_rocm(scratch.sin[i].as_ref()).and_then(|s| s.to_cpu_vec_f32());
                let nb = as_rocm(&buffers.norm_buf[i]).and_then(|s| s.to_cpu_vec_f32());
                let pv = as_rocm(scratch.post[i].as_ref()).and_then(|s| s.to_cpu_vec_f32());
                let cv = as_rocm(scratch.comb[i].as_ref()).and_then(|s| s.to_cpu_vec_f32());
                let ov = as_rocm(scratch.sout[i].as_ref()).and_then(|s| s.to_cpu_vec_f32());
                if let (Ok(sinv), Ok(nb), Ok(postv), Ok(combv), Ok(soutv)) = (sv, nb, pv, cv, ov) {
                    let (mut md, mut rr) = (0.0f32, 0.0f64);
                    for d in 0..hidden {
                        for h in 0..hc {
                            let mut acc = postv[h] * nb[d];
                            for k in 0..hc {
                                acc += combv[h * hc + k] * sinv[k * hidden + d];
                            }
                            let diff = (acc - soutv[h * hidden + d]).abs();
                            if diff > md {
                                md = diff;
                            }
                            rr += (acc as f64) * (acc as f64);
                        }
                    }
                    rr = (rr / (hc * hidden) as f64).sqrt();
                    let rg = (soutv.iter().map(|x| x * x).sum::<f32>()
                        / soutv.len().max(1) as f32)
                        .sqrt();
                    let rms = |v: &Vec<f32>| {
                        (v.iter().map(|x| x * x).sum::<f32>() / v.len().max(1) as f32).sqrt()
                    };
                    eprintln!(
                        "[xing-graph] WB inputs rms: sin {:.4e} nb {:.4e} post {:?} comb {:.4e} sout {:.4e}",
                        rms(&sinv), rms(&nb), &postv[..hc.min(postv.len())], rms(&combv), rg
                    );
                    eprintln!(
                        "[xing-graph] ATTN WRITEBACK vs HOST oracle: max_abs {md:.4e} rel {:.4e} ref_rms {rr:.4e} got_rms {rg:.4e}",
                        md as f64 / rr.max(1e-12)
                    );
                }
            }
            // ── ffn hc ──
            // Reads the stream the attention write-back just produced.
            dev.rms_norm_into(
                sout.as_ref(),
                &**layer.ffn_hc.input_norm.weight.storage(),
                layer.ffn_hc.input_norm.eps,
                as_rocm(scratch.hcn[i].as_ref())?,
                &Shape::new(vec![batch, flat]),
            )
            .map_err(grim_core::error::Error::Tensor)?;
            linear_into(
                &dev,
                scratch.hcn[i].as_ref(),
                &scratch.hc_fn_ffn[i],
                as_rocm(scratch.proj[i].as_ref())?,
                act,
            )?;
            let (iters, geps, cmin, cmax) = layer.ffn_hc.gate_consts();
            dev.mhc_gates_launch_into(
                as_rocm(scratch.proj[i].as_ref())?,
                // The FFN branch's OWN bias/scale, matching eager's
                // `self.ffn_hc.gates_d2d` — not the attention branch's.
                scratch.hc_base_ffn[i].as_ref(),
                scratch.hc_scale_ffn[i].as_ref(),
                as_rocm(scratch.pre[i].as_ref())?,
                as_rocm(scratch.post[i].as_ref())?,
                as_rocm(scratch.comb[i].as_ref())?,
                batch,
                hc,
                iters,
                geps,
                cmin,
                cmax,
            )?;
                        if i == 0 && !graph.capturing && std::env::var_os("GRIM_XING_TRACE").is_some() {
                let _ = grim_backend_rocm::device::helpers::hip_stream_synchronize(graph.stream);
                // Same warmup caveat as above: the FFN proj reads a stream
                // built on a stale KV arena.
                if let Some(b) = scratch.proj.first() {
                    if let Ok(st) = as_rocm(b.as_ref()) {
                        if let Ok(v) = st.to_cpu_vec_f32() {
                            let r = (v.iter().map(|x| x * x).sum::<f32>() / v.len().max(1) as f32).sqrt();
                            eprintln!("[xing-graph] WARMFFN proj rms {r:.4e} head {:?}", &v[..6.min(v.len())]);
                        }
                    }
                }
                if let Some(b) = scratch.comb.first() {
                    if let Ok(st) = as_rocm(b.as_ref()) {
                        if let Ok(v) = st.to_cpu_vec_f32() {
                            let rows: Vec<f32> = (0..hc).map(|h| (0..hc).map(|i| v[h * hc + i]).sum()).collect();
                            eprintln!("[xing-graph] WARMFFN comb rowsums {rows:?}");
                        }
                    }
                }
                if let Some(b) = scratch.post.first() {
                    if let Ok(st) = as_rocm(b.as_ref()) {
                        if let Ok(v) = st.to_cpu_vec_f32() {
                            eprintln!("[xing-graph] WARMFFN post {v:?}");
                        }
                    }
                }
                if let Some(b) = scratch.col.first() {
                    if let Ok(st) = as_rocm(b.as_ref()) {
                        if let Ok(v) = st.to_cpu_vec_f32() {
                            let r = (v.iter().map(|x| x * x).sum::<f32>() / v.len().max(1) as f32).sqrt();
                            eprintln!("[xing-graph] WARMFFN ffn_in n {} rms {r:.4e} head {:?}", v.len(), &v[..4.min(v.len())]);
                        }
                    }
                }
            }
// The FFN branch reads the stream the ATTENTION write-back just
            // produced (`sout[i]`), not the layer input (`sin[i]`) — a replace-all
            // had flattened both collapses onto `sin`, which the near-zero `post`
            // gate of layer 0 hid until layer 1's larger gate exposed it.
            dev.launch_hc_collapse_step(
                sout.as_ref(),
                as_rocm(scratch.pre[i].as_ref())?,
                as_rocm(scratch.col[i].as_ref())?,
                hc,
                hidden,
            )?;
            dev.rms_norm_into(
                as_rocm(scratch.col[i].as_ref())?,
                &**layer.ffn_norm.weight.storage(),
                layer.ffn_norm.eps,
                as_rocm(scratch.col[i].as_ref())?,
                &h_shape,
            )
            .map_err(grim_core::error::Error::Tensor)?;

            // ── FFN branch: dense SwiGLU or noaux_tc MoE ──
            if let Some(ref mlp) = layer.mlp {
                linear_into(
                    &dev,
                    scratch.col[i].as_ref(),
                    mlp.w1.weight(),
                    &buffers.gate_buf[i],
                    act,
                )?;
                linear_into(
                    &dev,
                    scratch.col[i].as_ref(),
                    mlp.w3.weight(),
                    &buffers.up_buf[i],
                    act,
                )?;
                dev.silu_mul_into(
                    &buffers.gate_buf[i],
                    &buffers.up_buf[i],
                    &buffers.activated_buf[i],
                )
                .map_err(grim_core::error::Error::Tensor)?;
                linear_into(
                    &dev,
                    &buffers.activated_buf[i],
                    mlp.w2.weight(),
                    &buffers.norm_buf[i],
                    act,
                )?;
            } else if let Some(ref moe) = layer.moe {
                linear_into(
                    &dev,
                    as_rocm(scratch.col[i].as_ref())?,
                    moe.gate.weight(),
                    &buffers.moe_gate_logits[i],
                    act,
                )?;
                let bias = moe.correction_bias_dev.as_ref().and_then(|b| {
                    b.storage()
                        .as_any()
                        .downcast_ref::<grim_backend_rocm::RocmStorage>()
                });
                dev.moe_route_topk_on_device(
                    &buffers.moe_gate_logits[i],
                    bias,
                    &buffers.moe_route_tokens,
                    &buffers.moe_route_experts,
                    &buffers.moe_route_weights,
                    batch,
                    cfg.n_routed_experts,
                    moe.num_experts_per_tok,
                    2,    // sigmoid + e_score_correction_bias (noaux_tc)
                    true, // xing4_0.expert_weights_norm
                )
                .map_err(|e| grim_core::error::Error::Backend(format!("moe route: {e}")))?;

                // Snapshot this layer's routing before the next layer
                // overwrites the shared buffers.
                {
                    let n = batch * moe.num_experts_per_tok;
                    let e = as_rocm(scratch.route_experts_snap[i].as_ref())?;
                    dev.copy_slice_into(e, &buffers.moe_route_experts, 0, n)?;
                    let w = as_rocm(scratch.route_weights_snap[i].as_ref())?;
                    dev.copy_slice_into(w, &buffers.moe_route_weights, 0, n)?;
                }
                // HOST ORACLE for the noaux_tc routing, at the first MoE
                // layer. Independent of the device kernel: sigmoid the gate
                // logits, select the top-k on (sigmoid + correction bias),
                // and normalise the selected sigmoid scores. If the graph
                // picks different experts than eager, nothing downstream can
                // agree - and layers 2..39 are all moe+shared, so this is the
                // first thing to check once layer 0 is oracle-verified.
                if i == 2 && !graph.capturing && std::env::var_os("GRIM_XING_TRACE").is_some() {
                    let _ = grim_backend_rocm::device::helpers::hip_stream_synchronize(graph.stream);
                    let gl = as_rocm(&buffers.moe_gate_logits[i]).and_then(|s| s.to_cpu_vec_f32());
                    let bs = match bias {
                        Some(b) => match as_rocm(b) { Ok(r) => r.to_cpu_vec_f32().ok(), Err(_) => None },
                        None => None,
                    };
                    // u32, NOT f32: reading it with to_cpu_vec_f32 reports
                    // zeros and looks exactly like a routing bug.
                    let de = as_rocm(&buffers.moe_route_experts)
                        .and_then(|s| s.to_cpu_vec_u32());
                    let dw = as_rocm(&buffers.moe_route_weights).and_then(|s| s.to_cpu_vec_f32());
                    eprintln!(
                        "[xing-graph] ROUTING probe L2: logits_ok {} bias_ok {} experts_ok {:?} w_ok {}",
                        gl.is_ok(), bs.is_some(), de.as_ref().map(|v| &v[..4.min(v.len())]), dw.is_ok()
                    );
                    if let (Ok(gl), Some(bs), Ok(de), Ok(dw)) = (gl, bs, de, dw) {
                        let ne = gl.len();
                        let sig: Vec<f32> = gl.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
                        let mut order: Vec<usize> = (0..ne).collect();
                        order.sort_by(|a, b| {
                            (sig[*b] + bs[*b])
                                .partial_cmp(&(sig[*a] + bs[*a]))
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        let sel: Vec<usize> = order.iter().take(moe.num_experts_per_tok).copied().collect();
                        let sum: f32 = sel.iter().map(|j| sig[*j]).sum();
                        let dev_sel: Vec<usize> = de
                            .iter()
                            .take(moe.num_experts_per_tok)
                            .map(|v| *v as usize)
                            .collect();
                        let _ = &dev_sel;
                        let devs: Vec<f32> =
                            dw.iter().take(moe.num_experts_per_tok).copied().collect();
                        let same_set = sel.iter().filter(|j| dev_sel.contains(j)).count();
                        eprintln!(
                            "[xing-graph] ROUTING oracle: dev_experts {dev_sel:?} dev_w {:?} | host_experts {sel:?} host_w {:?} | overlap {same_set}/{}",
                            devs.iter().map(|w| format!("{w:.4}")).collect::<Vec<_>>(),
                            sel.iter().map(|j| format!("{:.4}", sig[*j] / sum)).collect::<Vec<_>>(),
                            moe.num_experts_per_tok
                        );
                    }
                }
                // Arm 1 — native K-quant (zero extra VRAM): PEEK the pointer
                // arrays prewarmed at pool build; a miss must not upload
                // inside the capture bracket.
                let kq = crate::shared_moe::peek_kq_native(&moe.charon_cache);
                // Arm 2 — WhiteCrow stacks (budget-gated). PEEK, never ensure:
                // a miss here must not convert (D2H inside the capture
                // bracket poisons it).
                let wc = crate::shared_moe::peek_whitecrow_stacks(&moe.charon_cache);
                if let Some(kq) = kq {
                    let g_ptrs = kq
                        .gate_ptrs
                        .as_any()
                        .downcast_ref::<grim_backend_rocm::RocmStorage>()
                        .ok_or_else(|| {
                            grim_core::error::Error::Backend("kq gate ptrs not RocmStorage".into())
                        })?;
                    let u_ptrs = kq
                        .up_ptrs
                        .as_any()
                        .downcast_ref::<grim_backend_rocm::RocmStorage>()
                        .ok_or_else(|| {
                            grim_core::error::Error::Backend("kq up ptrs not RocmStorage".into())
                        })?;
                    let d_ptrs = kq
                        .down_ptrs
                        .as_any()
                        .downcast_ref::<grim_backend_rocm::RocmStorage>()
                        .ok_or_else(|| {
                            grim_core::error::Error::Backend("kq down ptrs not RocmStorage".into())
                        })?;
                    let col_r = as_rocm(scratch.col[i].as_ref())?;
                    dev.moe_fused_dispatch_kq_native_into(
                        col_r,
                        g_ptrs,
                        u_ptrs,
                        d_ptrs,
                        &buffers.moe_route_tokens,
                        &buffers.moe_route_experts,
                        &buffers.moe_route_weights,
                        batch * moe.num_experts_per_tok,
                        &buffers.moe_out[i],
                        hidden,
                        cfg.moe_intermediate_size,
                        moe.routed_scaling_factor,
                        kq.gate_bytes / cfg.moe_intermediate_size as u64,
                        kq.down_bytes / hidden as u64,
                        i32::from(kq.down_q4k),
                    )?;
                } else if let Some(wc) = wc {
                    let g_wc = wc
                        .gate
                        .as_any()
                        .downcast_ref::<grim_backend_rocm::RocmStorage>()
                        .ok_or_else(|| {
                            grim_core::error::Error::Backend("wc gate not RocmStorage".into())
                        })?;
                    let u_wc = wc
                        .up
                        .as_any()
                        .downcast_ref::<grim_backend_rocm::RocmStorage>()
                        .ok_or_else(|| {
                            grim_core::error::Error::Backend("wc up not RocmStorage".into())
                        })?;
                    let d_wc = wc
                        .down
                        .as_any()
                        .downcast_ref::<grim_backend_rocm::RocmStorage>()
                        .ok_or_else(|| {
                            grim_core::error::Error::Backend("wc down not RocmStorage".into())
                        })?;
                    let col_r = as_rocm(scratch.col[i].as_ref())?;
                    dev.moe_fused_dispatch_whitecrow_grouped_into(
                        col_r,
                        g_wc,
                        u_wc,
                        d_wc,
                        &buffers.moe_route_tokens,
                        &buffers.moe_route_experts,
                        &buffers.moe_route_weights,
                        batch * moe.num_experts_per_tok,
                        &buffers.moe_out[i],
                        hidden,
                        cfg.moe_intermediate_size,
                        moe.routed_scaling_factor,
                        wc.gate_stride,
                        wc.down_stride,
                    )?;
                } else {
                    let experts: Vec<crate::shared_moe::MoeExpert> = moe
                        .experts
                        .iter()
                        .map(|e| crate::shared_moe::MoeExpert {
                            gate: e.w1.clone(),
                            up: e.w3.clone(),
                            down: e.w2.clone(),
                        })
                        .collect();
                    let (_, _, _, g, u, d) = crate::shared_moe::ensure_charon_scratch(
                        dev.ordinal(),
                        batch,
                        moe.num_experts_per_tok,
                        &experts,
                        &moe.charon_cache,
                    )?;
                    let col_r = as_rocm(scratch.col[i].as_ref())?;
                    dev.moe_fused_dispatch_resident_routing_into(
                        col_r,
                        g.as_ref(),
                        u.as_ref(),
                        d.as_ref(),
                        &buffers.moe_route_tokens,
                        &buffers.moe_route_experts,
                        &buffers.moe_route_weights,
                        batch * moe.num_experts_per_tok,
                        &buffers.moe_out[i],
                        hidden,
                        cfg.moe_intermediate_size,
                        moe.routed_scaling_factor,
                    )?;
                }
                // Shared expert (dense SwiGLU at the shared width) added on top.
                if let Some(ref shared) = moe.shared_experts {
                    linear_into(
                        &dev,
                        scratch.col[i].as_ref(),
                        shared.w1.weight(),
                        as_rocm(scratch.shg[i].as_ref())?,
                        act,
                    )?;
                    linear_into(
                        &dev,
                        scratch.col[i].as_ref(),
                        shared.w3.weight(),
                        as_rocm(scratch.shu[i].as_ref())?,
                        act,
                    )?;
                    dev.silu_mul_into(
                        scratch.shg[i].as_ref(),
                        scratch.shu[i].as_ref(),
                        as_rocm(scratch.sha[i].as_ref())?,
                    )
                    .map_err(grim_core::error::Error::Tensor)?;
                    linear_into(
                        &dev,
                        scratch.sha[i].as_ref(),
                        shared.w2.weight(),
                        as_rocm(scratch.sho[i].as_ref())?,
                        act,
                    )?;
                    add_graph(
                        &buffers.moe_out[i],
                        scratch.sho[i].as_ref(),
                        &buffers.norm_buf[i],
                        &dev,
                    )?;
                } else {
                    publish_into(&dev, &buffers.norm_buf[i], &buffers.moe_out[i])?;
                }
            } else {
                return Err(grim_core::error::Error::Backend(format!(
                    "xing40 graph: layer {i} has neither mlp nor moe"
                )));
            }

            if std::env::var("GRIM_XING_STOP").ok().as_deref() == Some("ffn") {
                return Ok(());
            }
            // HOST ORACLE for the dense FFN's w2 GEMV — the last unverified
            // link in layer 0. Every other stage is now either bit-identical
            // to eager or checked against a host oracle, but this one feeds
            // norm_buf straight into the FFN write-back.
            //   out[j] = sum_c activated[c] * w2[j, c]
            if i == 0 && !graph.capturing && std::env::var_os("GRIM_XING_TRACE").is_some() {
                if let Some(mlp) = &layer.mlp {
                    let _ = grim_backend_rocm::device::helpers::hip_stream_synchronize(graph.stream);
                    let act = as_rocm(&buffers.activated_buf[i]).and_then(|s| s.to_cpu_vec_f32());
                    let w = mlp.w2.weight().to_vec_f32();
                    let nb = as_rocm(&buffers.norm_buf[i]).and_then(|s| s.to_cpu_vec_f32());
                    if let (Ok(act), Ok(w), Ok(nb)) = (act, w, nb) {
                        let k = act.len();
                        let n_out = 512.min(hidden);
                        let (mut md, mut rr) = (0.0f32, 0.0f64);
                        for jx in 0..n_out {
                            let mut acc = 0.0f32;
                            for c in 0..k {
                                acc += act[c] * w[jx * k + c];
                            }
                            let d = (acc - nb[jx]).abs();
                            if d > md {
                                md = d;
                            }
                            rr += (acc as f64) * (acc as f64);
                        }
                        rr = (rr / n_out as f64).sqrt();
                        let rg = (nb[..n_out].iter().map(|x| x * x).sum::<f32>()
                            / n_out as f32)
                            .sqrt();
                        eprintln!(
                            "[xing-graph] DENSE W2 vs HOST oracle: max_abs {md:.4e} rel {:.4e} ref_rms {rr:.4e} got_rms {rg:.4e}",
                            md as f64 / rr.max(1e-12)
                        );
                    }
                }
            }
            // ── ffn write-back: streams carry to the next layer ──
            let dst: &GBox = if i + 1 < n_layers {
                &scratch.sin[i + 1]
            } else {
                &scratch.sin[0] // last layer publishes into the seed slot
            };
            dev.launch_hc_write_back_step(
                &buffers.norm_buf[i],
                as_rocm(scratch.post[i].as_ref())?,
                as_rocm(scratch.comb[i].as_ref())?,
                sout.as_ref(),
                as_rocm(dst.as_ref())?,
                hc,
                hidden,
            )?;
            // HOST ORACLE for the FFN write-back, i.e. sin[1] — the value the
            // per-layer ratio table has tracked since the first session (graph
            // 3.9997e-3 vs eager 4.03218e-3). Same formula as the attention
            // write-back, over sout[i] and the FFN's own gates.
            if i == 0 && !graph.capturing && std::env::var_os("GRIM_XING_TRACE").is_some() {
                let _ = grim_backend_rocm::device::helpers::hip_stream_synchronize(graph.stream);
                let sv = as_rocm(scratch.sout[i].as_ref()).and_then(|s| s.to_cpu_vec_f32());
                let nb = as_rocm(&buffers.norm_buf[i]).and_then(|s| s.to_cpu_vec_f32());
                let pv = as_rocm(scratch.post[i].as_ref()).and_then(|s| s.to_cpu_vec_f32());
                let cv = as_rocm(scratch.comb[i].as_ref()).and_then(|s| s.to_cpu_vec_f32());
                let ov = as_rocm(scratch.sin[1].as_ref()).and_then(|s| s.to_cpu_vec_f32());
                if let (Ok(sinv), Ok(nb), Ok(postv), Ok(combv), Ok(outv)) = (sv, nb, pv, cv, ov) {
                    let (mut md, mut rr) = (0.0f32, 0.0f64);
                    for d in 0..hidden {
                        for h in 0..hc {
                            let mut acc = postv[h] * nb[d];
                            for k2 in 0..hc {
                                acc += combv[h * hc + k2] * sinv[k2 * hidden + d];
                            }
                            let diff = (acc - outv[h * hidden + d]).abs();
                            if diff > md {
                                md = diff;
                            }
                            rr += (acc as f64) * (acc as f64);
                        }
                    }
                    rr = (rr / (hc * hidden) as f64).sqrt();
                    let rg =
                        (outv.iter().map(|x| x * x).sum::<f32>() / outv.len().max(1) as f32).sqrt();
                    let rms = |v: &Vec<f32>| {
                        (v.iter().map(|x| x * x).sum::<f32>() / v.len().max(1) as f32).sqrt()
                    };
                    eprintln!(
                        "[xing-graph] FFNWB inputs rms: sout {:.4e} nb {:.4e} post {:?} comb {:.4e} sin1 {:.4e}",
                        rms(&sinv), rms(&nb), &postv[..hc.min(postv.len())], rms(&combv), rg
                    );
                    eprintln!(
                        "[xing-graph] FFN WRITEBACK vs HOST oracle: max_abs {md:.4e} rel {:.4e} ref_rms {rr:.4e} got_rms {rg:.4e}",
                        md as f64 / rr.max(1e-12)
                    );
                }
            }
        }
        if cap_layers < self.layers.len() {
            return Ok(());
        }

        // Head: mean the final streams, output_norm, lm_head.
        dev.launch_hc_mean_step(
            scratch.sin[0].as_ref(),
            as_rocm(scratch.meaned.as_ref())?,
            hc,
            hidden,
        )?;
        dev.rms_norm_into(
            scratch.meaned.as_ref(),
            &**self.norm.weight.storage(),
            self.norm.eps,
            as_rocm(scratch.meaned.as_ref())?,
            &h_shape,
        )
        .map_err(grim_core::error::Error::Tensor)?;
        linear_into(
            &dev,
            scratch.meaned.as_ref(),
            self.output.weight(),
            &buffers.head_output,
            &buffers.act_q81_buf[0],
        )?;
        Ok(())
    }

    fn forward_replay(&self, graph: &mut DecodeGraph, token_id: u32) -> Result<()> {
        if !graph.is_captured {
            return Err(grim_core::error::Error::Backend(
                "forward_replay before capture".into(),
            ));
        }
        let dev = dev_for_xing40(self)?;
        crate::lfm2_graph::write_embedding_to_buffer(&dev, &graph.buffers.token_ids_dev, token_id)?;
        let pos = graph.buffers.current_pos;
        // Decode-step token feed, position by position: compare against
        // eager's EAGERFEED line to find the FIRST step where the two runs
        // diverge - after which every cross-path comparison is a different
        // context.
        if std::env::var_os("GRIM_XING_TRACE").is_some() {
            eprintln!("[xing-graph] GRAPHFEED pos {pos} token {token_id}");
        }
        graph
            .buffers
            .write_pos_async(&dev, pos, graph.stream)
            .map_err(|e| grim_core::error::Error::Backend(format!("write pos: {e}")))?;
        graph
            .replay()
            .map_err(|e| grim_core::error::Error::Backend(format!("replay: {e}")))?;        // Diagnostics: join BOTH streams before any readback.
        if std::env::var_os("GRIM_XING_TRACE").is_some() {
            dev.synchronize();
            if let Some(scratch) = self.graph_scratch.get() {
                if let Ok(g) = scratch.lock() {
                    let dump = |label: &str, buf: &GBox| {
                        if let Ok(st) = as_rocm(buf.as_ref()) {
                            if let Ok(v) = st.to_cpu_vec_f32() {
                                let rms = (v.iter().map(|x| x * x).sum::<f32>()
                                    / v.len().max(1) as f32)
                                    .sqrt();
                                let nan = v.iter().filter(|x| x.is_nan()).count();
                                eprintln!(
                                    "[xing-graph] {label}: len {} rms {rms:.4e} nan {nan} head {:?}",
                                    v.len(),
                                    &v[..v.len().min(3)]
                                );
                            }
                        }
                    };
                    for (label, buf) in [
                        ("sin0", g.sin.first()),
                        ("hcn0", g.hcn.first()),
                        ("proj0", g.proj.first()),
                        ("pre0", g.pre.first()),
                        ("post0", g.post.first()),
                        ("comb0", g.comb.first()),
                        ("col0", g.col.first()),
                        ("qlora0", g.qlora.first()),
                        ("qf0", g.qf.first()),
                        ("qn0", g.qn.first()),
                        ("qp0", g.qp.first()),
                        ("qabs0", g.qabs.first()),
                        ("ckv0", g.ckv.first()),
                        ("kpe0", g.kpe.first()),
                        ("qattn0", g.qattn_latent.first()),
                        ("attn0", g.attn.first()),
                        ("shg0", g.shg.first()),
                        ("shu0", g.shu.first()),
                    ] {
                        if let Some(b) = buf {
                            dump(label, b);
                        }
                    }
                    if let Some(sb) = g.sout.first().map(|x| x.as_ref()) {
                        let Ok(b) = as_rocm(sb) else { return Ok(()) };
                        if let Ok(v) = b.to_cpu_vec_f32() {
                            let nan = v.iter().filter(|x| x.is_nan()).count();
                            eprintln!("[xing-graph] sout0: len {} nan {nan}", v.len());
                        }
                    }
                    if let Some(nb) = graph.buffers.norm_buf.first() {
                        if let Ok(v) = nb.to_cpu_vec_f32() {
                            let nan = v.iter().filter(|x| x.is_nan()).count();
                            let rms = (v.iter().map(|x| x * x).sum::<f32>() / v.len().max(1) as f32).sqrt();
                            eprintln!("[xing-graph] norm_buf0: len {} nan {nan} rms {rms:.4e}", v.len());
                        }
                    }
                    // The gathered embedding row: the graph uses
                    // launch_embedding_gather_dev_idx (IQ3_S kernel) where
                    // eager uses grim_nn::embedding_gather_on_device.
                    if let Some(li) = graph.buffers.layer_input.first() {
                        if let Ok(v) = li.to_cpu_vec_f32() {
                            let r = (v.iter().map(|x| x * x).sum::<f32>() / v.len().max(1) as f32).sqrt();
                            eprintln!("[xing-graph] embedding row: n {} rms {r:.4e} head {:?}", v.len(), &v[..v.len().min(4)]);
                        }
                    }
                    // Per-layer attn-branch input rms (post attn_norm), the
                    // direct counterpart of eager's "attn_in" trace.
                    {
                        let parts: Vec<String> = g
                            .col
                            .iter()
                            .enumerate()
                            .take(6)
                            .filter_map(|(i, b)| {
                                as_rocm(b.as_ref()).ok().and_then(|st| {
                                    st.to_cpu_vec_f32().ok().map(|v| {
                                        let r = (v.iter().map(|x| x * x).sum::<f32>()
                                            / v.len().max(1) as f32)
                                            .sqrt();
                                        format!("L{i}:{r:.4e}")
                                    })
                                })
                            })
                            .collect();
                        eprintln!("[xing-graph] col rms {}", parts.join(" "));
                    }
                    // Layer 0 vs layer 1 internals: which stage diverges.
                    for li in 0..2usize {
                        for (lbl, buf) in [
                            ("sout", g.sout.get(li)),
                            ("col", g.col.get(li)),
                            ("attn", g.attn.get(li)),
                            ("hcn", g.hcn.get(li)),
                        ] {
                            if let Some(b) = buf {
                                dump(&format!("L{li} {lbl}"), b);
                            }
                        }
                        if let Some(nb) = graph.buffers.norm_buf.get(li) {
                            if let Ok(v) = nb.to_cpu_vec_f32() {
                                let r = (v.iter().map(|x| x * x).sum::<f32>() / v.len().max(1) as f32).sqrt();
                                eprintln!("[xing-graph] L{li} norm_buf rms {r:.4e}");
                            }
                        }
                    }
                    // Per-layer stream rms: compare against the eager path's
                    // "L{i} in" trace to find the first diverging layer.
                    {
                        let parts: Vec<String> = g
                            .sin
                            .iter()
                            .enumerate()
                            .filter_map(|(i, b)| {
                                as_rocm(b.as_ref()).ok().and_then(|st| {
                                    st.to_cpu_vec_f32().ok().map(|v| {
                                        let r = (v.iter().map(|x| x * x).sum::<f32>()
                                            / v.len().max(1) as f32)
                                            .sqrt();
                                        format!("L{i}:{r:.4e}")
                                    })
                                })
                            })
                            .collect();
                        eprintln!("[xing-graph] sin rms {}", parts.join(" "));
                    // ATTENTION WRITE-BACK oracle at the real prefix. Every
                    // input is now individually verified (absorb exact, attn
                    // gates exact, o_proj exact) and all three survive the
                    // replay via the snapshots, so this checks the write-back
                    // itself: sout[h*hidden+d] = post[h]*y[d] +
                    // sum_k comb[h*hc+k]*sin[k*hidden+d].
                    {
                        let sv = g.sin0_snap
                            .first()
                            .and_then(|b| as_rocm(b.as_ref()).ok())
                            .and_then(|b| b.to_cpu_vec_f32().ok());
                        let nb = g.attn_out_snap
                            .first()
                            .and_then(|b| as_rocm(b.as_ref()).ok())
                            .and_then(|b| b.to_cpu_vec_f32().ok());
                        let pv = g.attn_post_snap
                            .first()
                            .and_then(|b| as_rocm(b.as_ref()).ok())
                            .and_then(|b| b.to_cpu_vec_f32().ok());
                        let cv = g.attn_comb_snap
                            .first()
                            .and_then(|b| as_rocm(b.as_ref()).ok())
                            .and_then(|b| b.to_cpu_vec_f32().ok());
                        let ov = g.sout
                            .first()
                            .and_then(|b| as_rocm(b.as_ref()).ok())
                            .and_then(|b| b.to_cpu_vec_f32().ok());
                        if let (Some(sinv), Some(nbv), Some(pv), Some(cv), Some(ov)) =
                            (sv, nb, pv, cv, ov)
                        {
                            let hid = self.cfg.hidden_size;
                            let hc4 = self.cfg.hc_mult;
                            let (mut md, mut rr) = (0.0f32, 0.0f64);
                            for d in 0..hid {
                                for h in 0..hc4 {
                                    let mut acc = pv[h] * nbv[d];
                                    for k in 0..hc4 {
                                        acc += cv[h * hc4 + k] * sinv[k * hid + d];
                                    }
                                    let diff = (acc - ov[h * hid + d]).abs();
                                    if diff > md {
                                        md = diff;
                                    }
                                    rr += (acc as f64) * (acc as f64);
                                }
                            }
                            rr = (rr / (hc4 * hid) as f64).sqrt();
                            let rg = (ov.iter().map(|x| x * x).sum::<f32>()
                                / ov.len().max(1) as f32)
                                .sqrt();
                            eprintln!(
                                "[xing-graph] ATTNWB post-replay vs HOST: max_abs {md:.4e} rel {:.4e} ref_rms {rr:.4e} got_rms {rg:.4e}",
                                md as f64 / rr.max(1e-12)
                            );
                        }
                    }

                    // The PRE-scale query, held unscaled here (the ratio is folded into
                    // `inv_sqrt_d`). This is the quantity eager dumps as QABS/QROPE
                    // and the comparison that locates the divergence upstream of
                    // the W_UV, which the fused/absorb alignment ruled out.
                    for (nm, buf) in [("QABS", &g.qabs), ("QROPE", &g.qp)] {
                        if let Some(b) = buf.first() {
                            if let Ok(st) = as_rocm(b.as_ref()) {
                                if let Ok(v) = st.to_cpu_vec_f32() {
                                    let r = (v.iter().map(|x| x * x).sum::<f32>()
                                        / v.len().max(1) as f32)
                                        .sqrt();
                                    eprintln!(
                                        "[xing-graph] {nm} pos {} rms {r:.6e} head {:?}",
                                        graph.buffers.current_pos,
                                        &v[..4.min(v.len())]
                                    );
                                }
                            }
                        }
                    }

                    // The o_proj output the graph fed to the attention
                    // write-back, snapshotted in-graph at the real prefix.
                    if let Some(nb) = g.attn_out_snap
                        .first()
                        .and_then(|b| as_rocm(b.as_ref()).ok())
                        .and_then(|b| b.to_cpu_vec_f32().ok())
                    {
                        let r =
                            (nb.iter().map(|x| x * x).sum::<f32>() / nb.len().max(1) as f32).sqrt();
                        eprintln!(
                            "[xing-graph] ATTNOUT pos {} rms {r:.6e} head {:?}",
                            graph.buffers.current_pos,
                            &nb[..4.min(nb.len())]
                        );
                    }

                    // OPROJ oracle against the SNAPSHOT taken in-graph at the
                    // real prefix: attn[0] x o_proj^T. attn[0] is exact, the
                    // attention gates are exact, so if this fails the whole
                    // attention write-back is explained.
                    {
                        let a = g.attn
                            .first()
                            .and_then(|b| as_rocm(b.as_ref()).ok())
                            .and_then(|b| b.to_cpu_vec_f32().ok());
                        let nb = g.attn_out_snap
                            .first()
                            .and_then(|b| as_rocm(b.as_ref()).ok())
                            .and_then(|b| b.to_cpu_vec_f32().ok());
                        let w = self.layers[0]
                            .self_attn
                            .o_proj
                            .weight()
                            .to_vec_f32()
                            .ok();
                        if let (Some(a), Some(nb), Some(w)) = (a, nb, w) {
                            let k = a.len();
                            let n_out = 512.min(nb.len());
                            let (mut md, mut rr) = (0.0f32, 0.0f64);
                            for j in 0..n_out {
                                let mut acc = 0.0f32;
                                for c in 0..k {
                                    acc += a[c] * w[j * k + c];
                                }
                                let d = (acc - nb[j]).abs();
                                if d > md {
                                    md = d;
                                }
                                rr += (acc as f64) * (acc as f64);
                            }
                            rr = (rr / n_out as f64).sqrt();
                            let rg = (nb[..n_out].iter().map(|x| x * x).sum::<f32>()
                                / n_out as f32)
                                .sqrt();
                            eprintln!(
                                "[xing-graph] OPROJ post-replay vs HOST: max_abs {md:.4e} rel {:.4e} ref_rms {rr:.4e} got_rms {rg:.4e}",
                                md as f64 / rr.max(1e-12)
                            );
                        }
                    }

                    // The ATTENTION branch's gates, snapshotted in-graph at a
                    // real prefix. Directly comparable with eager's
                    // position-tagged ATTNGATES line.
                    {
                        let hc4 = self.cfg.hc_mult;
                        let pv = g.attn_post_snap
                            .first()
                            .and_then(|b| as_rocm(b.as_ref()).ok())
                            .and_then(|b| b.to_cpu_vec_f32().ok());
                        let cv = g.attn_comb_snap
                            .first()
                            .and_then(|b| as_rocm(b.as_ref()).ok())
                            .and_then(|b| b.to_cpu_vec_f32().ok());
                        if let (Some(pv), Some(cv)) = (pv, cv) {
                            let rows: Vec<f32> = (0..hc4)
                                .map(|h| (0..hc4).map(|k| cv[h * hc4 + k]).sum())
                                .collect();
                            eprintln!(
                                "[xing-graph] ATTNGATES pos {} comb_rowsums {:?} post {:?}",
                                graph.buffers.current_pos,
                                rows,
                                &pv[..hc4.min(pv.len())]
                            );
                        }
                    }

                    // MLA latent-attention ORACLE, post-replay, over the real
                    // arena prefix. qattn_latent feeds the absorb, which feeds
                    // o_proj, which feeds the write-back — and attn_out has just
                    // been measured DIFFERING between the paths while every
                    // stage below it is host-exact, so this is the last link.
                    {
                        let rank = self.cfg.kv_lora_rank;
                        let rope_d = self.cfg.qk_rope_head_dim;
                        let nh = self.cfg.num_heads;
                        let mscale =
                            self.cfg.rope_yarn.map(|y| y.attention_factor).unwrap_or(1.0);
                        let inv = mscale * mscale
                            / ((self.cfg.qk_nope_head_dim + rope_d) as f32).sqrt();
                        let qa = g.qabs
                            .first()
                            .and_then(|b| as_rocm(b.as_ref()).ok())
                            .and_then(|b| b.to_cpu_vec_f32().ok());
                        let qr = g.qp
                            .first()
                            .and_then(|b| as_rocm(b.as_ref()).ok())
                            .and_then(|b| b.to_cpu_vec_f32().ok());
                        let kv = graph.buffers.k_arena
                            .first()
                            .and_then(|b| b.to_cpu_vec_f32().ok());
                        let got = g.qattn_latent
                            .first()
                            .and_then(|b| as_rocm(b.as_ref()).ok())
                            .and_then(|b| b.to_cpu_vec_f32().ok());
                        if let (Some(qa), Some(qr), Some(kv), Some(got)) = (qa, qr, kv, got) {
                            let row = rank + rope_d;
                            let slen = ((graph.buffers.current_pos as usize) + 1)
                                .min(kv.len() / row.max(1));
                            let mut scores = vec![0.0f32; slen];
                            let (mut md, mut rr) = (0.0f32, 0.0f64);
                            for h in 0..nh {
                                for t in 0..slen {
                                    let mut sc = 0.0f32;
                                    for c in 0..rank {
                                        sc += qa[h * rank + c] * kv[t * row + c];
                                    }
                                    for r in 0..rope_d {
                                        sc += qr[h * rope_d + r] * kv[t * row + rank + r];
                                    }
                                    scores[t] = sc * inv;
                                }
                                let mx =
                                    scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                                let mut ex = vec![0.0f32; slen];
                                let mut sum = 0.0f32;
                                for t in 0..slen {
                                    ex[t] = (scores[t] - mx).exp();
                                    sum += ex[t];
                                }
                                let iv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                                for c in 0..rank {
                                    let mut acc = 0.0f32;
                                    for t in 0..slen {
                                        acc += ex[t] * iv * kv[t * row + c];
                                    }
                                    let d = (acc - got[h * rank + c]).abs();
                                    if d > md {
                                        md = d;
                                    }
                                    rr += (acc as f64) * (acc as f64);
                                }
                            }
                            rr = (rr / (nh * rank) as f64).sqrt();
                            let n = nh * rank;
                            let rg =
                                (got[..n].iter().map(|x| x * x).sum::<f32>() / n as f32).sqrt();
                            eprintln!(
                                "[xing-graph] QATTN post-replay vs HOST (slen {slen}): max_abs {md:.4e} rel {:.4e} ref_rms {rr:.4e} got_rms {rg:.4e}",
                                md as f64 / rr.max(1e-12)
                            );
                        }
                    }

                    // Value-absorb ORACLE, post-replay: attn[h,v] =
                    // sum_c qattn_latent[h,c] * w_vc[h,v,c]. Both inputs
                    // survive the replay, so this checks the absorb at the
                    // real prefix rather than the warmup's slen=1.
                    {
                        let w = self.layers[0]
                            .self_attn
                            .w_vc_device(0)
                            .ok()
                            .flatten()
                            .and_then(|w| as_rocm(w).ok())
                            .and_then(|w| w.to_cpu_vec_f32().ok());
                        let q = g.qattn_latent
                            .first()
                            .and_then(|b| as_rocm(b.as_ref()).ok())
                            .and_then(|b| b.to_cpu_vec_f32().ok());
                        let a = g.attn
                            .first()
                            .and_then(|b| as_rocm(b.as_ref()).ok())
                            .and_then(|b| b.to_cpu_vec_f32().ok());
                        if let (Some(w), Some(q), Some(a)) = (w, q, a) {
                            let nh = self.cfg.num_heads;
                            let vd = self.cfg.v_head_dim;
                            let rank = self.cfg.kv_lora_rank;
                            let (mut md, mut rr) = (0.0f32, 0.0f64);
                            for h in 0..nh {
                                for v in 0..vd {
                                    let mut acc = 0.0f32;
                                    for c in 0..rank {
                                        acc += q[h * rank + c] * w[(h * vd + v) * rank + c];
                                    }
                                    let d = (acc - a[h * vd + v]).abs();
                                    if d > md {
                                        md = d;
                                    }
                                    rr += (acc as f64) * (acc as f64);
                                }
                            }
                            rr = (rr / (nh * vd) as f64).sqrt();
                            let rg =
                                (a.iter().map(|x| x * x).sum::<f32>() / a.len().max(1) as f32).sqrt();
                            eprintln!(
                                "[xing-graph] ABSORB post-replay vs HOST: max_abs {md:.4e} rel {:.4e} ref_rms {rr:.4e} got_rms {rg:.4e}",
                                md as f64 / rr.max(1e-12)
                            );
                        }
                    }

                    // Post-attention stream (sout[0]) - the other input to the
                    // FFN write-back, and the product of the attention branch.
                    if let Some(sb) = g.sout.first() {
                        if let Ok(st) = as_rocm(sb.as_ref()) {
                            if let Ok(v) = st.to_cpu_vec_f32() {
                                let r = (v.iter().map(|x| x * x).sum::<f32>()
                                    / v.len().max(1) as f32)
                                    .sqrt();
                                eprintln!(
                                    "[xing-graph] STREAMATTN pos {} rms {r:.6e} head {:?}",
                                    graph.buffers.current_pos,
                                    &v[..4]
                                );
                            }
                        }
                    }
                    // IN-SITU MoE ORACLE for layer 2 (the first moe+shared
                    // layer). The dispatch has a bit-identical UNIT test but has
                    // never been checked on the real routing: norm_buf[i] is
                    // only ever an INPUT to the write-back oracle, which passes
                    // regardless of whether the dispatch itself was right.
                    // Routing comes from the PER-LAYER snapshot - the shared
                    // buffers hold the last layer's routing post-replay.
                    if std::env::var_os("GRIM_XING_TRACE").is_some() {
                        const ML: usize = 2;
                        const NOUT: usize = 256;
                        if let Some(moe) = &self.layers.get(ML).and_then(|l| l.moe.as_ref()) {
                            let col = g
                                .col
                                .get(ML)
                                .and_then(|b| as_rocm(b.as_ref()).ok())
                                .and_then(|s| s.to_cpu_vec_f32().ok());
                            let nb = graph
                                .buffers
                                .norm_buf
                                .get(ML)
                                .and_then(|s| s.to_cpu_vec_f32().ok());
                            let sel = g
                                .route_experts_snap
                                .get(ML)
                                .and_then(|b| as_rocm(b.as_ref()).ok())
                                .and_then(|s| s.to_cpu_vec_u32().ok());
                            let wts = g
                                .route_weights_snap
                                .get(ML)
                                .and_then(|b| as_rocm(b.as_ref()).ok())
                                .and_then(|s| s.to_cpu_vec_f32().ok());
                            if let (Some(col), Some(nb), Some(sel), Some(wts)) = (col, nb, sel, wts)
                            {
                                let hidden = col.len();
                                let topk = sel.len().min(wts.len());
                                let mut acc = vec![0.0f32; NOUT.min(hidden)];
                                let mut ok = true;
                                for t in 0..topk {
                                    let e = sel[t] as usize;
                                    let w = wts[t] * moe.routed_scaling_factor;
                                    let Some(x) = moe.experts.get(e) else { ok = false; break };
                                    let (Ok(g1), Ok(u1), Ok(d1)) = (
                                        x.w1.weight().to_vec_f32(),
                                        x.w3.weight().to_vec_f32(),
                                        x.w2.weight().to_vec_f32(),
                                    ) else { ok = false; break };
                                    let inter = g1.len() / hidden.max(1);
                                    let mut act = vec![0.0f32; inter];
                                    for j in 0..inter {
                                        let (mut ga, mut ua) = (0.0f32, 0.0f32);
                                        for c in 0..hidden.min(g1.len() / inter.max(1)) {
                                            ga += col[c] * g1[j * hidden + c];
                                            ua += col[c] * u1[j * hidden + c];
                                        }
                                        act[j] = (ga / (1.0 + (-ga).exp())) * ua;
                                    }
                                    for (j, o) in acc.iter_mut().enumerate().take(NOUT.min(hidden)) {
                                        let mut v = 0.0f32;
                                        for c in 0..inter.min(act.len()) {
                                            v += act[c] * d1[j * inter + c];
                                        }
                                        *o += w * v;
                                    }
                                }
                                if ok {
                                    if let Some(sh) = &moe.shared_experts {
                                        if let (Ok(s1), Ok(s3), Ok(s2)) = (
                                            sh.w1.weight().to_vec_f32(),
                                            sh.w3.weight().to_vec_f32(),
                                            sh.w2.weight().to_vec_f32(),
                                        ) {
                                            let sinter = s1.len() / hidden.max(1);
                                            let mut act = vec![0.0f32; sinter];
                                            for j in 0..sinter {
                                                let (mut ga, mut ua) = (0.0f32, 0.0f32);
                                                for c in 0..hidden.min(s1.len() / sinter.max(1)) {
                                                    ga += col[c] * s1[j * hidden + c];
                                                    ua += col[c] * s3[j * hidden + c];
                                                }
                                                act[j] = (ga / (1.0 + (-ga).exp())) * ua;
                                            }
                                            for (j, o) in
                                                acc.iter_mut().enumerate().take(NOUT.min(hidden))
                                            {
                                                let mut v = 0.0f32;
                                                for c in 0..sinter.min(act.len()) {
                                                    v += act[c] * s2[j * sinter + c];
                                                }
                                                *o += v;
                                            }
                                        }
                                    }
                                    let n = acc.len().min(nb.len());
                                    let md = acc[..n]
                                        .iter()
                                        .zip(nb[..n].iter())
                                        .map(|(x, y)| (x - y).abs())
                                        .fold(0.0f32, f32::max);
                                    let rr =
                                        (acc[..n].iter().map(|x| x * x).sum::<f32>() / n.max(1) as f32)
                                            .sqrt();
                                    let rg =
                                        (nb[..n].iter().map(|x| x * x).sum::<f32>() / n.max(1) as f32)
                                            .sqrt();
                                    eprintln!(
                                        "[xing-graph] MOE-ORACLE L{ML} (first {n} outs, top{topk}): max_abs {md:.4e} rel {:.4e} host_rms {rr:.4e} dev_rms {rg:.4e}",
                                        md as f64 / (rr as f64).max(1e-12)
                                    );
                                }
                            }
                        }
                    }

                    // FFN WRITE-BACK ORACLE ACROSS DEPTH, post-replay. Every
                    // input survives the replay for EVERY layer (sout[i] is
                    // written once by the attention branch, norm_buf[i] by the
                    // FFN, post[i]/comb[i] by the FFN's gate call, and the
                    // result lands in sin[i+1]). Running it per layer turns it
                    // into a depth scan: the first layer whose oracles fail
                    // localises the residual, which layer 0 alone cannot do.
                    {
                        let hid = self.cfg.hidden_size;
                        let hc4 = self.cfg.hc_mult;
                        let n_l = self.layers.len();
                        let mut bad: Vec<String> = Vec::new();
                        for li in 0..n_l {
                            let dst = if li + 1 < n_l { g.sin.get(li + 1) } else { g.sin.first() };
                            let (Some(sv), Some(nv), Some(pv), Some(cv), Some(ov)) = (
                                g.sout.get(li),
                                graph.buffers.norm_buf.get(li),
                                g.post.get(li),
                                g.comb.get(li),
                                dst,
                            ) else {
                                continue;
                            };
                            fn rd<E>(r: std::result::Result<&grim_backend_rocm::RocmStorage, E>) -> Option<Vec<f32>> {
                                r.ok().and_then(|x| x.to_cpu_vec_f32().ok())
                            }
                            let got = (
                                rd(as_rocm(sv.as_ref())),
                                rd(as_rocm(nv)),
                                rd(as_rocm(pv.as_ref())),
                                rd(as_rocm(cv.as_ref())),
                                rd(as_rocm(ov.as_ref())),
                            );
                            if let (Some(sinv), Some(nbv), Some(postv), Some(combv), Some(outv)) = got {
                                let (mut md, mut rr) = (0.0f32, 0.0f64);
                                for d in 0..hid {
                                    for h in 0..hc4 {
                                        let mut acc = postv[h] * nbv[d];
                                        for k in 0..hc4 {
                                            acc += combv[h * hc4 + k] * sinv[k * hid + d];
                                        }
                                        let diff = (acc - outv[h * hid + d]).abs();
                                        if diff > md {
                                            md = diff;
                                        }
                                        rr += (acc as f64) * (acc as f64);
                                    }
                                }
                                rr = (rr / (hc4 * hid) as f64).sqrt();
                                let rg = (outv.iter().map(|x| x * x).sum::<f32>()
                                    / outv.len().max(1) as f32)
                                    .sqrt();
                                let rel = md as f64 / rr.max(1e-12);
                                if rel > 1e-5 {
                                    bad.push(format!(
                                        "L{li} rel {rel:.3e} (max_abs {md:.3e}, rms {rg:.3e})"
                                    ));
                                }
                            }
                        }
                        if bad.is_empty() {
                            eprintln!(
                                "[xing-graph] FFWB-DEPTH: all {n_l} layers pass (<1e-5)"
                            );
                        } else {
                            eprintln!(
                                "[xing-graph] FFWB-DEPTH: {} of {n_l} layers FAIL: {}",
                                bad.len(),
                                bad.iter().take(6).cloned().collect::<Vec<_>>().join(" | ")
                            );
                        }
                    }

                    // Layer 0's FFN hc gates, post-replay: the FFN gate call
                    // is the last writer of comb[0]/post[0], so these survive
                    // and are directly comparable with eager's FFNGATES line.
                    if let Some(c) = g.comb.first() {
                        if let Ok(st) = as_rocm(c.as_ref()) {
                            if let Ok(v) = st.to_cpu_vec_f32() {
                                let hc4 = self.cfg.hc_mult;
                                let rows: Vec<f32> = (0..hc4)
                                    .map(|h| (0..hc4).map(|k| v[h * hc4 + k]).sum())
                                    .collect();
                                eprintln!(
                                    "[xing-graph] FFNGATES pos {} comb_rowsums {:?}",
                                    graph.buffers.current_pos,
                                    rows
                                );
                            }
                        }
                    }
                    // Layer 0's OUTPUT stream at this replay, tagged with the
                    // position so it can be matched against eager's "L1 in".
                    if let Some(b) = g.sin.get(1) {
                        if let Ok(st) = as_rocm(b.as_ref()) {
                            if let Ok(v) = st.to_cpu_vec_f32() {
                                let r = (v.iter().map(|x| x * x).sum::<f32>()
                                    / v.len().max(1) as f32)
                                    .sqrt();
                                eprintln!(
                                    "[xing-graph] SIN1 pos {} rms {r:.6e} head {:?}",
                                    graph.buffers.current_pos,
                                    &v[..v.len().min(4)]
                                );
                            }
                        }
                    }
                    }
                    // Top-5 of the head output: the direct comparison point
                    // against eager's --logprobs line for the same step.
                    if let Ok(v) = graph.buffers.head_output.to_cpu_vec_f32() {
                        {
                            let last = &v[v.len().saturating_sub(self.cfg.vocab_size)..];
                            let mx = last.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                            let lse = mx as f64
                                + last.iter().map(|x| ((x - mx) as f64).exp()).sum::<f64>().ln();
                            let mut idx: Vec<u32> = (0..last.len() as u32).collect();
                            idx.sort_by(|a, b| last[*b as usize].total_cmp(&last[*a as usize]));
                            idx.truncate(5);
                            let parts: Vec<String> = idx.iter().map(|i| {
                                format!("{}:{:.4}", i, last[*i as usize] as f64 - lse)
                            }).collect();
                            let nonzero = last.iter().filter(|x| **x != 0.0).count();
                            let within = last.iter().filter(|x| (**x - mx).abs() < 1e-3).count();
                            eprintln!("[xing-graph] head stats: n {} nonzero {nonzero} within1e-3-of-max {within} max {mx:.4} min {:.4}", last.len(), last.iter().copied().fold(f32::INFINITY, f32::min));
                            // KV arena rows around the replay position: the seed
                            // is the one input the graph does not recompute, so
                            // an off-by-one there is invisible everywhere else.
                            let kvr = match graph.buffers.k_arena.first() { Some(k) => k.to_cpu_vec_f32(), None => return Ok(()) };
                            let row = self.cfg.kv_lora_rank + self.cfg.qk_rope_head_dim;
                            let p = graph.buffers.current_pos as usize;
                            for (li, ka) in graph.buffers.k_arena.iter().enumerate().take(3) {
                                if let Ok(kv) = ka.to_cpu_vec_f32() {
                                    if p * row + row <= kv.len() {
                                        let seg = &kv[p * row..p * row + row];
                                        eprintln!(
                                            "[xing-graph] ARENAROW L{li} pos {p} rms {:.6e} head {:?}",
                                            (seg.iter().map(|x| x * x).sum::<f32>() / row as f32).sqrt(),
                                            &seg[..4]
                                        );
                                    }
                                }
                            }
                            if let Ok(kv) = kvr {
                                let row = self.cfg.kv_lora_rank + self.cfg.qk_rope_head_dim;
                                let p = graph.buffers.current_pos as usize;
                                // Include a SEEDED row (not one this replay
                            // appended): rows 0..p-1 come from the eager
                            // session, and a single stale one out of ~36 is
                            // exactly the magnitude of the residual.
                            for r in [10usize, p.saturating_sub(2), p - 1, p] {
                                    if r * row + row <= kv.len() {
                                        let seg = &kv[r * row..r * row + row];
                                        eprintln!(
                                            "[xing-graph] ARENA row {r}: rms {:.4e} head {:?}",
                                            (seg.iter().map(|x| x * x).sum::<f32>() / row as f32).sqrt(),
                                            &seg[..4]
                                        );
                                    }
                                }
                            }
                            eprintln!(
                                "[xing-graph] REPLAY at kv_pos {} head top5 [{}]",
                                graph.buffers.current_pos,
                                parts.join(" ")
                            );
                        }
                    }
                    if let Ok(b) = as_rocm(g.meaned.as_ref()) {
                        if let Ok(v) = b.to_cpu_vec_f32() {
                            let rms = (v.iter().map(|x| x * x).sum::<f32>() / v.len().max(1) as f32).sqrt();
                            eprintln!("[xing-graph] HEADINPUT pos {} rms {rms:.6e} head {:?}", graph.buffers.current_pos, &v[..v.len().min(4)]);
                        }
                    }
                }
            }
        }

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
                grim_core::error::Error::Backend(
                    "xing40 seed: session state must be the per-layer latent kv vec".into(),
                )
            })?;
        let latent_dim = self.cfg.kv_lora_rank + self.cfg.qk_rope_head_dim;
        let mut out: Vec<Option<EagerKvSource<'a>>> = Vec::with_capacity(self.layers.len());
        for c in caches.iter() {
            match c {
                Some((latent, _)) => {
                    let storage = latent.storage().as_ref();
                    let k_dev = storage
                        .as_any()
                        .downcast_ref::<grim_backend_rocm::RocmStorage>()
                        .and_then(|r| r.device_ptr())
                        .ok_or_else(|| {
                            grim_core::error::Error::Backend(
                                "xing40 seed: latent cache has no device ptr".into(),
                            )
                        })? as *const f32;
                    // Clamp to the rows the cache actually holds. run.rs
                    // seeds BEFORE the step's eager forward runs, so the
                    // caller's `valid_rows` is routinely one ahead of the
                    // session's real row count; honouring it copied one
                    // uninitialized row into the arena, and the replay then
                    // attended over a latent whose c_kv half was zeros while
                    // eager attended over the real one. Measured: arena row
                    // 35 head [0,0,0,0] while every eager row's c_kv rms
                    // was ~1.33.
                    let have = latent.shape().dim(0).unwrap_or(0) as u32;
                    let rows = valid_rows.min(have);
                    if rows != valid_rows {
                        eprintln!(
                            "[xing40-graph] kv seed: clamped valid_rows {valid_rows} -> {rows} \
                             (cache holds {have} rows)"
                        );
                    }
                    out.push(Some(EagerKvSource {
                        k_dev,
                        v_dev: k_dev,
                        prefill_len: rows,
                        kv_stride: latent_dim,
                        gdl_state: None,
                        _anchor: std::marker::PhantomData,
                    }));
                }
                None => out.push(None),
            }
        }
        eprintln!(
            "[xing40-graph] kv seed: {} of {} layers have an eager latent cache (valid_rows {valid_rows}, latent_dim {latent_dim})",
            out.iter().filter(|x| x.is_some()).count(),
            out.len()
        );
        Ok(out)
    }
}
