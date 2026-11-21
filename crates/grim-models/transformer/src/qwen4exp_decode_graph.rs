//! ROCm HIP decode graph capture & replay for Qwen3.8-Flash-Next (Q8).
//!
//! Uses string graph keys and `RocmDevice::{begin,end,replay}_graph_capture_force`
//! (mirroring `decode_graph.rs:6237-6267`) without touching `DecodeGraphKey`.
//! PLE lookup, QSA host indexer, and host-staged MoE stay outside captured regions.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

use grim_backend_cpu::cpu_tensor;
use grim_core::error::{Error, Result};
use grim_tensor::{Device, Shape, Tensor};

use crate::qwen4exp_gdn_device::{gdn_decode_step_on_device, Qwen38GdnDeviceCache};

static GRAPH_TAG_GEN: AtomicU64 = AtomicU64::new(1);

/// Returns `true` when `GRIM_FLASH_DECODE_GRAPH` is truthy (`1` or `true`).
#[inline]
pub fn flash_decode_graph_enabled() -> bool {
    std::env::var("GRIM_FLASH_DECODE_GRAPH")
        .ok()
        .map(|v| {
            let v = v.trim();
            v == "1" || v.eq_ignore_ascii_case("true")
        })
        .unwrap_or(false)
}

/// Fixed-address device I/O buffers and capture registry for Flash-Next GDN decode graphs.
#[derive(Debug, Clone)]
pub struct FlashDecodeGraphState {
    tag: u64,
    qkv_in: Tensor,
    alpha_in: Tensor,
    beta_in: Tensor,
    gate_in: Tensor,
    out_buf: Tensor,
    captured_layers: HashSet<usize>,
}

impl FlashDecodeGraphState {
    pub fn new(
        device: &Device,
        conv_dim: usize,
        n_v_heads: usize,
        val_dim: usize,
    ) -> Result<Self> {
        let qkv_in = grim_nn::modules::move_to_device(
            &cpu_tensor(vec![0.0f32; conv_dim], Shape::new(vec![1, conv_dim])),
            device,
        )?;
        let alpha_in = grim_nn::modules::move_to_device(
            &cpu_tensor(vec![0.0f32; n_v_heads], Shape::new(vec![1, n_v_heads])),
            device,
        )?;
        let beta_in = grim_nn::modules::move_to_device(
            &cpu_tensor(vec![0.0f32; n_v_heads], Shape::new(vec![1, n_v_heads])),
            device,
        )?;
        let gate_in = grim_nn::modules::move_to_device(
            &cpu_tensor(vec![0.0f32; val_dim], Shape::new(vec![1, val_dim])),
            device,
        )?;
        let out_buf = grim_nn::modules::move_to_device(
            &cpu_tensor(vec![0.0f32; val_dim], Shape::new(vec![1, val_dim])),
            device,
        )?;
        Ok(Self {
            tag: GRAPH_TAG_GEN.fetch_add(1, Ordering::Relaxed),
            qkv_in,
            alpha_in,
            beta_in,
            gate_in,
            out_buf,
            captured_layers: HashSet::new(),
        })
    }

    pub fn is_layer_captured(&self, layer_idx: usize) -> bool {
        self.captured_layers.contains(&layer_idx)
    }
}

fn as_rocm<'a>(t: &'a Tensor, name: &str) -> Result<&'a grim_backend_rocm::RocmStorage> {
    t.storage()
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .ok_or_else(|| Error::Backend(format!("flash_decode_graph: {name} is not RocmStorage")))
}

/// Captures (on first call) or replays (on subsequent calls) a HIP graph for one
/// Flash-Next GDN decode step (`seq_len == 1`) using fixed device I/O buffers.
/// Falls back loudly via `[flash-decode-graph] fallback: <reason>` to eager
/// `gdn_decode_step_on_device` on any error.
#[allow(clippy::too_many_arguments)]
pub fn capture_or_replay_gdn_step(
    layer_idx: usize,
    qkv: &Tensor,
    conv_w: &Tensor,
    alpha: &Tensor,
    beta: &Tensor,
    dt_bias: &Tensor,
    ssm_a: &Tensor,
    ssm_norm_w: &Tensor,
    gate: &Tensor,
    eps: f32,
    n_v_heads: usize,
    n_k_heads: usize,
    head_dim: usize,
    cache: &mut Qwen38GdnDeviceCache,
    state: &mut FlashDecodeGraphState,
) -> Result<Tensor> {
    let ordinal = match qkv.device().ordinal() {
        Some(o) => o,
        None => {
            return gdn_decode_step_on_device(
                qkv, conv_w, alpha, beta, dt_bias, ssm_a, ssm_norm_w, gate, eps, n_v_heads,
                n_k_heads, head_dim, cache,
            );
        }
    };
    let rocm = match grim_backend_rocm::RocmDevice::shared_if_alive(ordinal) {
        Some(d) => d,
        None => {
            eprintln!("[flash-decode-graph] fallback: no live ROCm device {ordinal}");
            return gdn_decode_step_on_device(
                qkv, conv_w, alpha, beta, dt_bias, ssm_a, ssm_norm_w, gate, eps, n_v_heads,
                n_k_heads, head_dim, cache,
            );
        }
    };

    // Pre-materialize tiled norm weight outside capture so zero D2H happens inside graph.
    let val_dim = n_v_heads * head_dim;
    let out_shape = Shape::new(vec![1, val_dim]);
    if cache.norm_tiled.is_none() {
        let nw = ssm_norm_w.to_vec_f32()?;
        let mut tiled = Vec::with_capacity(val_dim);
        for _ in 0..n_v_heads {
            for d in 0..head_dim {
                tiled.push(nw.get(d).copied().unwrap_or(1.0));
            }
        }
        cache.norm_tiled = Some(grim_nn::modules::move_to_device(
            &cpu_tensor(tiled, out_shape.clone()),
            qkv.device(),
        )?);
    }

    // D2D stage inputs into fixed-address graph buffers.
    let dev = grim_nn::modules::pick_device_for_tensor(qkv);
    dev.copy_slice_range(
        state.qkv_in.storage().as_ref(),
        0,
        qkv.storage().as_ref(),
        0,
        qkv.shape().elem_count(),
    )?;
    dev.copy_slice_range(
        state.alpha_in.storage().as_ref(),
        0,
        alpha.storage().as_ref(),
        0,
        alpha.shape().elem_count(),
    )?;
    dev.copy_slice_range(
        state.beta_in.storage().as_ref(),
        0,
        beta.storage().as_ref(),
        0,
        beta.shape().elem_count(),
    )?;
    dev.copy_slice_range(
        state.gate_in.storage().as_ref(),
        0,
        gate.storage().as_ref(),
        0,
        gate.shape().elem_count(),
    )?;

    let key = format!("flash_gdn_{:x}_d{ordinal}_l{layer_idx}", state.tag);
    if state.captured_layers.contains(&layer_idx) && rocm.has_captured_graph(&key) {
        match rocm.replay_graph_force(&key) {
            Ok(true) => return Ok(state.out_buf.clone()),
            Ok(false) => {}
            Err(err) => {
                eprintln!("[flash-decode-graph] fallback: replay failed ({err})");
                return gdn_decode_step_on_device(
                    qkv, conv_w, alpha, beta, dt_bias, ssm_a, ssm_norm_w, gate, eps, n_v_heads,
                    n_k_heads, head_dim, cache,
                );
            }
        }
    }

    // Capture the GDN decode step into `key`.
    let cap_res: Result<()> = (|| {
        rocm.begin_graph_capture_force(&key)?;
        let qkv_r = as_rocm(&state.qkv_in, "qkv_in")?;
        let conv_w_r = as_rocm(conv_w, "conv_w")?;
        let conv_state_r = as_rocm(&cache.conv_state, "conv_state")?;
        let conv_out_r = as_rocm(&cache.conv_out_buf, "conv_out_buf")?;
        rocm.short_conv1d_causal_step_into(qkv_r, conv_w_r, None, conv_state_r, conv_out_r)?;

        let alpha_r = as_rocm(&state.alpha_in, "alpha_in")?;
        let beta_r = as_rocm(&state.beta_in, "beta_in")?;
        let dt_r = as_rocm(dt_bias, "dt_bias")?;
        let a_r = as_rocm(ssm_a, "ssm_a")?;
        let ssm_state_r = as_rocm(&cache.ssm_state, "ssm_state")?;
        let acc_r = as_rocm(&cache.acc_buf, "acc_buf")?;
        rocm.launch_kda_delta_rule_raw_into(
            conv_out_r,
            alpha_r,
            beta_r,
            dt_r,
            a_r,
            ssm_state_r,
            acc_r,
            n_v_heads,
            n_k_heads,
            head_dim,
            eps * head_dim as f32,
        )?;

        let norm_tiled = cache
            .norm_tiled
            .as_ref()
            .ok_or_else(|| Error::Backend("norm_tiled missing".into()))?;
        let out_r = as_rocm(&state.out_buf, "out_buf")?;
        rocm.mul_into(
            cache.acc_buf.storage().as_ref(),
            norm_tiled.storage().as_ref(),
            out_r,
        )?;
        rocm.sigmoid_mul_into(state.gate_in.storage().as_ref(), out_r)?;
        rocm.end_graph_capture_force(&key)?;
        Ok(())
    })();

    match cap_res {
        Ok(()) => {
            state.captured_layers.insert(layer_idx);
            rocm.replay_graph_force(&key)?;
            Ok(state.out_buf.clone())
        }
        Err(err) => {
            let _ = rocm.end_graph_capture_force(&key);
            eprintln!("[flash-decode-graph] fallback: capture failed ({err})");
            gdn_decode_step_on_device(
                qkv, conv_w, alpha, beta, dt_bias, ssm_a, ssm_norm_w, gate, eps, n_v_heads,
                n_k_heads, head_dim, cache,
            )
        }
    }
}
