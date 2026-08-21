//! Shared attention entry point for transformer loaders.
//!
//! One canonical function replaces the ~25 per-model scalar CPU attention
//! loops (lfm2.rs-style). It first tries the fused device kernel
//! (`BackendDevice::qkv_attention`, which handles GQA, causal masking and
//! sliding windows on ROCm/CUDA/Metal/Vulkan/CPU), and falls back to the
//! reference scalar loop (ported from `block.rs::cpu_attention_fallback`)
//! when the backend returns `Unimplemented` or the tensors live on CPU.
//!
//! See docs/adr/0001-attention-own-vs-delegate.md.

use grim_core::error::Result;
use grim_nn::modules::pick_device_for_storage_device;
use grim_tensor::{DType, Device, Shape, Tensor};
use std::sync::Arc;

/// Inputs are flat host buffers with the layouts the scalar loops already use:
/// - `q`: `[steps, num_heads, head_dim]` (post-RoPE)
/// - `k_history` / `v_history`: `[kv_len, num_kv_heads, head_dim]`, already
///   extended with the current step's keys/values (so `kv_len >= steps` and
///   `cache_offset = kv_len - steps`).
///
/// Returns a `[steps, num_heads * head_dim]` tensor on `device`.
///
/// Causal/window contract (matches `BackendDevice::qkv_attention`): query at
/// absolute position `cache_offset + i` attends to keys `j` with
/// `j <= cache_offset + i` and, when `window` is set,
/// `j >= cache_offset + i - window + 1`.
#[allow(clippy::too_many_arguments)]
pub fn fused_or_scalar_attention(
    q: &[f32],
    k_history: &[f32],
    v_history: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    steps: usize,
    window: Option<usize>,
    device: &Device,
) -> Result<Tensor> {
    let kv_stride = num_kv_heads * head_dim;
    let kv_len = k_history.len() / kv_stride;
    debug_assert_eq!(k_history.len(), kv_len * kv_stride);
    debug_assert_eq!(v_history.len(), kv_len * kv_stride);
    let cache_offset = kv_len.saturating_sub(steps);

    let out_shape = Shape::new(vec![steps, num_heads, head_dim]);
    let dev = pick_device_for_storage_device(device);

    let q_st = dev.from_cpu(q, &out_shape, DType::F32)?;
    let kv_shape = Shape::new(vec![kv_len, num_kv_heads, head_dim]);
    let k_st = dev.from_cpu(k_history, &kv_shape, DType::F32)?;
    let v_st = dev.from_cpu(v_history, &kv_shape, DType::F32)?;

    match dev.qkv_attention(
        q_st.as_ref(),
        k_st.as_ref(),
        v_st.as_ref(),
        num_kv_heads,
        kv_len,
        cache_offset as u32,
        window,
        &out_shape,
        None,
        None,
    ) {
        Ok((storage, _handle)) => Ok(Tensor::new(
            Arc::from(storage),
            out_shape,
            DType::F32,
            grim_tensor::QuantProvenance::default(),
            device.clone(),
        )),
        Err(_) => scalar_attention(
            q,
            k_history,
            v_history,
            num_heads,
            num_kv_heads,
            head_dim,
            steps,
            kv_len,
            cache_offset,
            window,
            1.0 / (head_dim as f32).sqrt(),
            &dev,
            device,
        ),
    }
}

/// Like [`fused_or_scalar_attention`] but with an explicit softmax scale
/// override (e.g. `qk_scale_factor / sqrt(head_dim)`). Used by models whose
/// config carries a non-unit `qk_scale_factor` (muse_glimmer-class); the
/// device kernel contract has no scale parameter, so those models always
/// take the scalar path when `scale` differs from `1/sqrt(head_dim)`.
#[allow(clippy::too_many_arguments)]
pub fn fused_or_scalar_attention_scaled(
    q: &[f32],
    k_history: &[f32],
    v_history: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    steps: usize,
    window: Option<usize>,
    scale: f32,
    device: &Device,
) -> Result<Tensor> {
    let kv_stride = num_kv_heads * head_dim;
    let kv_len = k_history.len() / kv_stride;
    let cache_offset = kv_len.saturating_sub(steps);
    let dev = pick_device_for_storage_device(device);
    scalar_attention(
        q,
        k_history,
        v_history,
        num_heads,
        num_kv_heads,
        head_dim,
        steps,
        kv_len,
        cache_offset,
        window,
        scale,
        &dev,
        device,
    )
}

/// Reference scalar attention with causal + sliding-window masking.
/// Direct port of `block.rs::cpu_attention_fallback`, taking explicit dims
/// so loaders without a `BlockConfig` can use it.
#[allow(clippy::too_many_arguments)]
fn scalar_attention(
    q: &[f32],
    k_history: &[f32],
    v_history: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    steps: usize,
    kv_len: usize,
    cache_offset: usize,
    window: Option<usize>,
    scale: f32,
    dev: &std::sync::Arc<dyn grim_tensor::backend::BackendDevice>,
    device: &Device,
) -> Result<Tensor> {
    let num_head_dims = num_heads * head_dim;
    let kv_stride = num_kv_heads * head_dim;
    let mut out = vec![0.0f32; steps * num_head_dims];

    for h in 0..num_heads {
        let kvh = (h * num_kv_heads) / num_heads;
        for t in 0..steps {
            let causal_limit = cache_offset + t;
            let window_start = match window {
                Some(w) => (causal_limit + 1).saturating_sub(w),
                None => 0,
            };
            let mut scores = vec![0.0f32; kv_len];
            for t2 in 0..kv_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[t * num_head_dims + h * head_dim + d]
                        * k_history[t2 * kv_stride + kvh * head_dim + d];
                }
                scores[t2] = dot * scale;
            }
            for (t2, s) in scores.iter_mut().enumerate() {
                if t2 > causal_limit || t2 < window_start {
                    *s = f32::NEG_INFINITY;
                }
            }

            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - mx).exp();
                sum += *s;
            }
            for s in &mut scores {
                *s /= sum;
            }
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for t2 in window_start..=causal_limit {
                    acc += scores[t2] * v_history[t2 * kv_stride + kvh * head_dim + d];
                }
                out[t * num_head_dims + h * head_dim + d] = acc;
            }
        }
    }

    let flat = Shape::new(vec![steps, num_head_dims]);
    let storage = dev.from_cpu(&out, &flat, DType::F32)?;
    Ok(Tensor::new(
        Arc::from(storage),
        flat,
        DType::F32,
        grim_tensor::QuantProvenance::default(),
        device.clone(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scalar fallback must match a straightforward textbook reference
    /// with causal + sliding-window masking, including at nonzero offsets.
    #[test]
    fn scalar_attention_matches_reference_with_window() {
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        let kv_len = 24usize;
        let steps = 3usize;
        let window = Some(10usize);

        let mut q = vec![0.0f32; steps * num_heads * head_dim];
        let mut k = vec![0.0f32; kv_len * num_kv_heads * head_dim];
        let mut v = vec![0.0f32; kv_len * num_kv_heads * head_dim];
        let mut seed = 0x1234_5678u64;
        let mut rand = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        for x in &mut q {
            *x = rand();
        }
        for x in &mut k {
            *x = rand();
        }
        for x in &mut v {
            *x = rand();
        }

        let dev: std::sync::Arc<dyn grim_tensor::backend::BackendDevice> =
            pick_device_for_storage_device(&Device::Cpu);
        let got = scalar_attention(
            &q, &k, &v, num_heads, num_kv_heads, head_dim, steps, kv_len,
            kv_len - steps, window, 1.0 / (head_dim as f32).sqrt(), &dev, &Device::Cpu,
        )
        .unwrap();
        let got = got.to_vec_f32().unwrap();

        // Independent reference.
        let scale = 1.0 / (head_dim as f32).sqrt();
        let kv_stride = num_kv_heads * head_dim;
        let cache_offset = kv_len - steps;
        for t in 0..steps {
            for h in 0..num_heads {
                let kvh = (h * num_kv_heads) / num_heads;
                let causal_limit = cache_offset + t;
                let window_start = (causal_limit + 1).saturating_sub(window.unwrap());
                let mut scores = Vec::with_capacity(causal_limit - window_start + 1);
                for t2 in window_start..=causal_limit {
                    let mut dot = 0.0;
                    for d in 0..head_dim {
                        dot += q[t * num_heads * head_dim + h * head_dim + d]
                            * k[t2 * kv_stride + kvh * head_dim + d];
                    }
                    scores.push(dot * scale);
                }
                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = scores.iter().map(|s| (s - mx).exp()).sum();
                for d in 0..head_dim {
                    let mut acc = 0.0;
                    for (i, s) in scores.iter().enumerate() {
                        let t2 = window_start + i;
                        acc += ((s - mx).exp() / sum)
                            * v[t2 * kv_stride + kvh * head_dim + d];
                    }
                    let expect = acc;
                    let idx = t * num_heads * head_dim + h * head_dim + d;
                    assert!(
                        (got[idx] - expect).abs() < 1e-5,
                        "t={t} h={h} d={d}: got {} expect {}",
                        got[idx],
                        expect
                    );
                }
            }
        }
    }
}
