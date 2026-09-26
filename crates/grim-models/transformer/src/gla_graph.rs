//! GRAVE Phase 4 — capture-safe graph branch for GDL layers (`gla_graph.rs`).
//!
//! Replaces k/v arenas + `kv_append` + softmax attention + bump for GDL
//! layers with ONE fused launch (`grim_gla_state_update_output`: state
//! update + output matvec + gated RMSNorm). The launch reads the K/V GEMV
//! outputs directly — same buffer-wiring discipline as Phase B (fused QKV).
//!
//! Every op enqueues a kernel on the active stream; no allocations, no host
//! readbacks, no H2D inside — safe under `begin/end_capture`. Anything the
//! fused path cannot serve (GQA mismatch, non-64 dims, missing GDL buffers)
//! returns `Unimplemented` so the caller falls back eager (spec §Fallback).

use grim_backend_rocm::RocmStorage;
use grim_backend_rocm::decode_graph_buffers::DecodeGraphBuffers;
use grim_backend_rocm::device::compute::gla_launchers::GlaLaunchArgs;
use grim_core::error::{Error, Result};
use grim_tensor::BackendStorage;

use crate::lfm2::{Lfm2AttentionMode, Lfm2Block};
use crate::lfm2_graph::{add_graph, dot_fused_ok, is_q80, linear_into, rocm_storage};

type Dev = grim_backend_rocm::RocmDevice;
type Storage = dyn grim_tensor::BackendStorage;

/// GDL attention branch: norm → QKV GEMVs → fused GDN-2 → O proj → residual.
/// Mirrors `attn_forward_graph` steps 1–2 and 7, replacing steps 3–6
/// (QK-norm/RoPE/append/attend/bump) with the single fused launch.
pub fn gdl_forward_graph(
    block: &Lfm2Block,
    layer_idx: usize,
    buffers: &DecodeGraphBuffers,
    dev: &Dev,
    fuse_norm: bool,
) -> Result<()> {
    // Perimeter defense: only the LDS fast path is capturable today.
    if block.attention_mode != Lfm2AttentionMode::Gdl {
        return Err(Error::Unimplemented(
            "gdl_forward_graph: non-GDL layer; use attn_forward_graph".into(),
        ));
    }
    if block.num_heads % block.num_kv_heads != 0 {
        return Err(Error::Unimplemented(format!(
            "gdl_forward_graph: GQA ({}q/{}kv) needs nh % nkv == 0; use eager",
            block.num_heads, block.num_kv_heads
        )));
    }
    if block.head_dim != 64 {
        return Err(Error::Unimplemented(format!(
            "gdl_forward_graph: head_dim {} != 64 LDS path; use eager",
            block.head_dim
        )));
    }
    let wq = block
        .wq
        .as_ref()
        .ok_or_else(|| Error::Backend("gdl_forward_graph: missing wq".into()))?;
    let wk = block
        .wk
        .as_ref()
        .ok_or_else(|| Error::Backend("gdl_forward_graph: missing wk".into()))?;
    let wv = block
        .wv
        .as_ref()
        .ok_or_else(|| Error::Backend("gdl_forward_graph: missing wv".into()))?;
    let wo = block
        .wo
        .as_ref()
        .ok_or_else(|| Error::Backend("gdl_forward_graph: missing wo".into()))?;
    let gdl = buffers
        .gdl
        .get(layer_idx)
        .and_then(|o| o.as_ref())
        .ok_or_else(|| {
            Error::Unimplemented(
                "gdl_forward_graph: no GDL buffers (allocate_gdl_layer first); use eager".into(),
            )
        })?;
    if gdl.heads != block.num_heads || gdl.dk != 64 || gdl.dv != 64 {
        return Err(Error::Backend(
            "gdl_forward_graph: GDL buffer dims mismatch layer; re-allocate".into(),
        ));
    }

    // 1. Attention norm into the per-layer staging slot (same as softmax path).
    dev.rms_norm_into(
        &buffers.layer_input[layer_idx],
        &**block.attn_norm.weight.storage(),
        block.attn_norm.eps,
        &buffers.norm_buf[layer_idx],
        &buffers.layer_input[layer_idx].shape().clone(),
    )
    .map_err(grim_core::error::Error::Tensor)?;
    let normed: &Storage = &buffers.norm_buf[layer_idx];

    // 2. QKV projections (same Phase-B wiring as the softmax path).
    let hidden = normed.shape().dims().last().copied().unwrap_or(0);
    let act = &buffers.act_q81_buf[layer_idx];
    let n_q = block.num_heads * block.head_dim;
    let n_kv = block.num_kv_heads * block.head_dim;
    if dot_fused_ok(dev, hidden)
        && is_q80(&wq.weight)
        && is_q80(&wk.weight)
        && is_q80(&wv.weight)
        && n_q % 4 == 0
        && n_kv % 4 == 0
    {
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
        .map_err(|e| Error::Backend(format!("gdl fused qkv dot4: {e}")))?;
    } else {
        linear_into(dev, normed, &wq.weight, &buffers.q_buf[layer_idx], act)?;
        linear_into(dev, normed, &wk.weight, &buffers.k_buf[layer_idx], act)?;
        linear_into(dev, normed, &wv.weight, &buffers.v_buf[layer_idx], act)?;
    }

    // 3. GQA head-repeat: expand K/V from [nkv, hd] to [nh, hd] when
    //    nh != nkv. The GEMV writes compact [nkv, hd] into k_buf/v_buf;
    //    the expanded buffers are what the GDL kernel consumes.
    let (k_launch, v_launch): (&RocmStorage, &RocmStorage) =
        if block.num_heads != block.num_kv_heads {
            let k_exp = gdl
                .k_exp_buf
                .as_ref()
                .ok_or_else(|| Error::Backend("gdl_forward_graph: missing k_exp_buf".into()))?;
            let v_exp = gdl
                .v_exp_buf
                .as_ref()
                .ok_or_else(|| Error::Backend("gdl_forward_graph: missing v_exp_buf".into()))?;
            dev.head_repeat(
                &buffers.k_buf[layer_idx],
                &buffers.v_buf[layer_idx],
                k_exp,
                v_exp,
                block.num_kv_heads,
                block.num_heads,
                block.num_heads / block.num_kv_heads,
                64,
            )
            .map_err(|e| Error::Backend(format!("gdl head_repeat: {e}")))?;
            (k_exp, v_exp)
        } else {
            (&buffers.k_buf[layer_idx], &buffers.v_buf[layer_idx])
        };

    // 4. ONE fused launch replaces kv_append + attention + norm.
    //    Gate G4 profile marker: zero kv_append/attention launches here.
    dev.launch_gla_state_update_output_into(&GlaLaunchArgs {
        q: &buffers.q_buf[layer_idx],
        k: k_launch,
        v: v_launch,
        alpha: &gdl.alpha,
        b_gate: &gdl.b_gate,
        w_gate: &gdl.w_gate,
        norm_w: &gdl.norm_w,
        out_gate: &gdl.out_gate,
        state: &gdl.state,
        out: &buffers.attn_out_buf[layer_idx],
        heads: block.num_heads,
        batch: buffers.batch.max(1),
        dk: 64,
        dv: 64,
        eps: block.attn_norm.eps,
    })
    .map_err(|e| Error::Backend(format!("gdl fused launch: {e}")))?;

    // 4. O projection into staging, then residual add -> layer_output
    //    (same tail as the softmax path).
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
    Ok(())
}

/// Helper: is this layer served by the fused GDL graph branch?
pub fn is_gdl_layer(block: &Lfm2Block) -> bool {
    block.attention_mode == Lfm2AttentionMode::Gdl && block.wq.is_some()
}
