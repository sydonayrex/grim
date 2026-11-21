//! Device Gated DeltaNet (GDN) decode step for Qwen3.8-Flash-Next (Q3).
//!
//! Reuses existing ROCm kernels (`short_conv1d_causal_step_into`,
//! `grim_kda_gated_delta_rule_batched`, `grim_mul`, `sigmoid_on_device`) with
//! zero new HIP kernel source.

use std::sync::Arc;

use grim_backend_cpu::cpu_tensor;
use grim_core::error::{Error, Result};
use grim_tensor::backend::CoreTensorOps;
use grim_tensor::{DType, Device, QuantProvenance, Shape, Tensor};

use crate::qwen4exp_gdn::Qwen38GdnCache;

/// Returns `true` when `GRIM_GDN_DEVICE` is truthy (`1` or `true`).
#[inline]
pub fn gdn_device_enabled() -> bool {
    std::env::var("GRIM_GDN_DEVICE")
        .ok()
        .map(|v| {
            let v = v.trim();
            v == "1" || v.eq_ignore_ascii_case("true")
        })
        .unwrap_or(false)
}

/// Per-layer device-resident GDN recurrent state and scratch buffers.
#[derive(Debug, Clone)]
pub struct Qwen38GdnDeviceCache {
    pub conv_state: Tensor,
    pub ssm_state: Tensor,
    conv_out_buf: Tensor,
    acc_buf: Tensor,
    norm_tiled: Option<Tensor>,
}

impl Qwen38GdnDeviceCache {
    /// Allocates zero-initialized device state for one GDN layer.
    pub fn new(
        device: &Device,
        conv_dim: usize,
        taps: usize,
        n_v_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let state_taps = taps.saturating_sub(1).max(1);
        let val_dim = n_v_heads * head_dim;
        let conv_state = grim_nn::modules::move_to_device(
            &cpu_tensor(
                vec![0.0f32; conv_dim * state_taps],
                Shape::new(vec![conv_dim, state_taps]),
            ),
            device,
        )?;
        let ssm_state = grim_nn::modules::move_to_device(
            &cpu_tensor(
                vec![0.0f32; n_v_heads * head_dim * head_dim],
                Shape::new(vec![n_v_heads, head_dim, head_dim]),
            ),
            device,
        )?;
        let conv_out_buf = grim_nn::modules::move_to_device(
            &cpu_tensor(vec![0.0f32; conv_dim], Shape::new(vec![1, conv_dim])),
            device,
        )?;
        let acc_buf = grim_nn::modules::move_to_device(
            &cpu_tensor(vec![0.0f32; val_dim], Shape::new(vec![1, val_dim])),
            device,
        )?;
        Ok(Self {
            conv_state,
            ssm_state,
            conv_out_buf,
            acc_buf,
            norm_tiled: None,
        })
    }

    /// Uploads an existing host `Qwen38GdnCache` to device once (e.g., after prefill).
    pub fn from_host_cache(
        device: &Device,
        host: &Qwen38GdnCache,
        conv_dim: usize,
        taps: usize,
        n_v_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let state_taps = taps.saturating_sub(1).max(1);
        let need_conv = conv_dim * state_taps;
        let mut conv_vec = host.conv_state.clone();
        conv_vec.resize(need_conv, 0.0);
        let need_ssm = n_v_heads * head_dim * head_dim;
        let mut ssm_vec = host.ssm_state.clone();
        ssm_vec.resize(need_ssm, 0.0);
        let val_dim = n_v_heads * head_dim;

        let conv_state = grim_nn::modules::move_to_device(
            &cpu_tensor(conv_vec, Shape::new(vec![conv_dim, state_taps])),
            device,
        )?;
        let ssm_state = grim_nn::modules::move_to_device(
            &cpu_tensor(ssm_vec, Shape::new(vec![n_v_heads, head_dim, head_dim])),
            device,
        )?;
        let conv_out_buf = grim_nn::modules::move_to_device(
            &cpu_tensor(vec![0.0f32; conv_dim], Shape::new(vec![1, conv_dim])),
            device,
        )?;
        let acc_buf = grim_nn::modules::move_to_device(
            &cpu_tensor(vec![0.0f32; val_dim], Shape::new(vec![1, val_dim])),
            device,
        )?;
        Ok(Self {
            conv_state,
            ssm_state,
            conv_out_buf,
            acc_buf,
            norm_tiled: None,
        })
    }
}

fn as_rocm_storage<'a>(t: &'a Tensor, name: &str) -> Result<&'a grim_backend_rocm::RocmStorage> {
    t.storage()
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .ok_or_else(|| Error::Backend(format!("gdn_decode_step_on_device: {name} is not RocmStorage")))
}

/// Runs one Flash-Next GDN decode step (`seq_len == 1`, `KdaHeadPairing::Interleaved`)
/// entirely on ROCm:
/// 1. `short_conv1d_causal_step_into` (updates `cache.conv_state` in place)
/// 2. `launch_kda_delta_rule_raw_into` (internal SiLU, L2 norm with `eps * head_dim`,
///    delta update of `cache.ssm_state`, `rsqrt(head_dim)` read-out into `cache.acc_buf`)
/// 3. `acc * ssm_norm_weight * sigmoid(gate)` via device `mul` + `sigmoid_on_device`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_step_on_device(
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
) -> Result<Tensor> {
    let ordinal = qkv
        .device()
        .ordinal()
        .ok_or_else(|| Error::Backend("gdn_decode_step_on_device: non-GPU tensor".into()))?;
    let dev = grim_backend_rocm::RocmDevice::shared_if_alive(ordinal).ok_or_else(|| {
        Error::Backend(format!(
            "gdn_decode_step_on_device: no live ROCm device {ordinal}"
        ))
    })?;

    let qkv_r = as_rocm_storage(qkv, "qkv")?;
    let conv_w_r = as_rocm_storage(conv_w, "conv_w")?;
    let conv_state_r = as_rocm_storage(&cache.conv_state, "conv_state")?;
    let conv_out_r = as_rocm_storage(&cache.conv_out_buf, "conv_out_buf")?;
    dev.short_conv1d_causal_step_into(qkv_r, conv_w_r, None, conv_state_r, conv_out_r)?;

    let alpha_r = as_rocm_storage(alpha, "alpha")?;
    let beta_r = as_rocm_storage(beta, "beta")?;
    let dt_r = as_rocm_storage(dt_bias, "dt_bias")?;
    let a_r = as_rocm_storage(ssm_a, "ssm_a")?;
    let ssm_state_r = as_rocm_storage(&cache.ssm_state, "ssm_state")?;
    let acc_r = as_rocm_storage(&cache.acc_buf, "acc_buf")?;

    dev.launch_kda_delta_rule_raw_into(
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
    let norm_tiled = cache
        .norm_tiled
        .as_ref()
        .ok_or_else(|| Error::Backend("norm_tiled missing".into()))?;

    let gate_sig = grim_nn::modules::sigmoid_on_device(gate)?;
    let (acc_normed, _) = dev.mul(
        cache.acc_buf.storage().as_ref(),
        norm_tiled.storage().as_ref(),
        &out_shape,
    )?;
    let (gated_st, _) = dev.mul(
        acc_normed.as_ref(),
        gate_sig.storage().as_ref(),
        &out_shape,
    )?;

    Ok(Tensor::new(
        Arc::from(gated_st),
        out_shape,
        DType::F32,
        QuantProvenance::default(),
        qkv.device().clone(),
    ))
}
