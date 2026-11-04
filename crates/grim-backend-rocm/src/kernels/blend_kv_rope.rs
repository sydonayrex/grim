//! Fused RoPE + Scatter KV Cache Blending Kernel.
//! Reuses partially-matched prefix block contents (tokens `0..divergence_token`), computes rotary positional embeddings (RoPE) only for the.

use grim_tensor::error::{Error, Result};

/// Configuration parameters for fused KV cache blending.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlendConfig {
    pub block_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub divergence_token: usize,
}

/// Fused KV cache blending CPU reference implementation.
/// Preserves `k_dst[..divergence_token]` and `v_dst[..divergence_token]` from existing cache,
/// and applies RoPE to `k_src[divergence_token..]` before scattering into `k_dst`.
pub fn blend_kv_rope_cpu(
    cfg: &BlendConfig,
    k_src: &[f32],
    v_src: &[f32],
    k_dst: &mut [f32],
    v_dst: &mut [f32],
    base_pos: usize,
    rope_theta: f32,
) -> Result<()> {
    blend_kv_rope_transform_cpu(
        cfg,
        k_src,
        v_src,
        k_dst,
        v_dst,
        None, // no unrotate; k_src is unrotated
        base_pos,
        rope_theta,
    )
}

/// Fused KV cache blending with two-position RoPE unrotate(old_pos) -> rotate(new_pos).
/// When `old_base_pos` is `Some(old)`, `k_src` is assumed to be POST-RoPE at `old + t` and is first unrotated
/// by `-angle_old` before being rotated by `+angle_new` at `new + t`.
/// If `old_base_pos` is `None`, standard forward-only RoPE rotation from `new + t` is performed.
pub fn blend_kv_rope_transform_cpu(
    cfg: &BlendConfig,
    k_src: &[f32],
    v_src: &[f32],
    k_dst: &mut [f32],
    v_dst: &mut [f32],
    old_base_pos: Option<usize>,
    new_base_pos: usize,
    rope_theta: f32,
) -> Result<()> {
    let elem_per_token = cfg.num_heads * cfg.head_dim;
    let expected_len = cfg.block_size * elem_per_token;

    if k_src.len() != expected_len || v_src.len() != expected_len {
        return Err(Error::Backend(format!(
            "blend_kv_rope: input buffer size mismatch (expected {expected_len}, got k={}, v={})",
            k_src.len(),
            v_src.len()
        )));
    }
    if k_dst.len() != expected_len || v_dst.len() != expected_len {
        return Err(Error::Backend(format!(
            "blend_kv_rope: destination buffer size mismatch (expected {expected_len}, got k={}, v={})",
            k_dst.len(),
            v_dst.len()
        )));
    }

    if cfg.divergence_token > cfg.block_size {
        return Err(Error::Backend(format!(
            "blend_kv_rope: divergence_token {} exceeds block_size {}",
            cfg.divergence_token, cfg.block_size
        )));
    }

    // Copy divergent tail for V (no RoPE on V)
    let tail_start = cfg.divergence_token * elem_per_token;
    v_dst[tail_start..].copy_from_slice(&v_src[tail_start..]);

    // Apply RoPE and copy divergent tail for K
    let half_dim = cfg.head_dim / 2;
    for t in cfg.divergence_token..cfg.block_size {
        let token_offset = t * elem_per_token;

        for h in 0..cfg.num_heads {
            let head_offset = token_offset + h * cfg.head_dim;

            for d in 0..half_dim {
                let freq = 1.0 / (rope_theta.powf((2 * d) as f32 / cfg.head_dim as f32));
                let delta_pos = match old_base_pos {
                    Some(old) => (new_base_pos as f32) - (old as f32),
                    None => (new_base_pos + t) as f32,
                };
                let angle = delta_pos * freq;
                let (sin_val, cos_val) = angle.sin_cos();

                let k0 = k_src[head_offset + d];
                let k1 = k_src[head_offset + d + half_dim];

                // Rotary rotation by angle (delta_pos)
                k_dst[head_offset + d] = k0 * cos_val - k1 * sin_val;
                k_dst[head_offset + d + half_dim] = k0 * sin_val + k1 * cos_val;
            }
        }
    }

    Ok(())
}

/// GPU HipRTC kernel source template for fused RoPE + scatter cache blending.
pub const BLEND_KV_ROPE_HIP_SRC: &str = r#"
extern "C" __global__ void blend_kv_rope_transform_kernel(
    const float* __restrict__ k_src,
    const float* __restrict__ v_src,
    float* __restrict__ k_dst,
    float* __restrict__ v_dst,
    int block_size,
    int num_heads,
    int head_dim,
    int divergence_token,
    int old_base_pos,
    int new_base_pos,
    int is_post_rope,
    float rope_theta
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int total_elems = block_size * num_heads * head_dim;
    if (tid >= total_elems) return;

    int elem_per_token = num_heads * head_dim;
    int token_idx = tid / elem_per_token;

    // Tokens before divergence remain untouched in k_dst / v_dst
    if (token_idx < divergence_token) return;

    // Copy value directly
    v_dst[tid] = v_src[tid];

    // Compute RoPE for key
    int rem = tid % elem_per_token;
    int h = rem / head_dim;
    int d = rem % head_dim;
    int half_dim = head_dim / 2;

    float delta_pos = (is_post_rope != 0)
        ? (float)(new_base_pos - old_base_pos)
        : (float)(new_base_pos + token_idx);

    int d_base = (d < half_dim) ? d : (d - half_dim);
    float freq = 1.0f / __powf(rope_theta, (float)(2 * d_base) / (float)head_dim);
    float angle = delta_pos * freq;
    float sin_val, cos_val;
    __sincosf(angle, &sin_val, &cos_val);

    int head_offset = token_idx * elem_per_token + h * head_dim;
    float k0 = k_src[head_offset + d_base];
    float k1 = k_src[head_offset + d_base + half_dim];

    if (d < half_dim) {
        k_dst[tid] = k0 * cos_val - k1 * sin_val;
    } else {
        k_dst[tid] = k0 * sin_val + k1 * cos_val;
    }
}

extern "C" __global__ void blend_kv_rope_kernel(
    const float* __restrict__ k_src,
    const float* __restrict__ v_src,
    float* __restrict__ k_dst,
    float* __restrict__ v_dst,
    int block_size,
    int num_heads,
    int head_dim,
    int divergence_token,
    int base_pos,
    float rope_theta
) {
    blend_kv_rope_transform_kernel(
        k_src, v_src, k_dst, v_dst,
        block_size, num_heads, head_dim, divergence_token,
        0, base_pos, 0, rope_theta
    );
}
"#;

use crate::device::roc_device::RocmDevice;
use crate::memory::storage::RocmStorage;
use std::ffi::c_void;

/// Launch GPU-accelerated fused RoPE + scatter KV cache blending kernel with unrotate(old_pos) -> rotate(new_pos).
pub fn launch_blend_kv_rope_transform(
    dev: &RocmDevice,
    cfg: &BlendConfig,
    k_src: &RocmStorage,
    v_src: &RocmStorage,
    k_dst: &RocmStorage,
    v_dst: &RocmStorage,
    old_base_pos: Option<i32>,
    new_base_pos: i32,
    rope_theta: f32,
) -> Result<*mut c_void> {
    let k_src_ptr = k_src
        .device_ptr
        .ok_or_else(|| Error::Backend("blend_kv_rope: k_src has no device ptr".into()))?;
    let v_src_ptr = v_src
        .device_ptr
        .ok_or_else(|| Error::Backend("blend_kv_rope: v_src has no device ptr".into()))?;
    let k_dst_ptr = k_dst
        .device_ptr
        .ok_or_else(|| Error::Backend("blend_kv_rope: k_dst has no device ptr".into()))?;
    let v_dst_ptr = v_dst
        .device_ptr
        .ok_or_else(|| Error::Backend("blend_kv_rope: v_dst has no device ptr".into()))?;

    let total_elems = cfg.block_size * cfg.num_heads * cfg.head_dim;
    let block_dim = crate::HipDim3::new(256, 1, 1);
    let grid_dim = crate::HipDim3::new((total_elems as u32).div_ceil(256), 1, 1);

    let mut ks = k_src_ptr;
    let mut vs = v_src_ptr;
    let mut kd = k_dst_ptr;
    let mut vd = v_dst_ptr;
    let mut bs = cfg.block_size as i32;
    let mut nh = cfg.num_heads as i32;
    let mut hd = cfg.head_dim as i32;
    let mut dt = cfg.divergence_token as i32;
    let mut old_bp = old_base_pos.unwrap_or(0);
    let mut new_bp = new_base_pos;
    let mut is_post = if old_base_pos.is_some() { 1i32 } else { 0i32 };
    let mut theta = rope_theta;

    dev.launch_compute_kernel(
        "blend_kv_rope_transform_kernel",
        grid_dim,
        block_dim,
        &mut [
            crate::arg(&mut ks),
            crate::arg(&mut vs),
            crate::arg(&mut kd),
            crate::arg(&mut vd),
            crate::arg(&mut bs),
            crate::arg(&mut nh),
            crate::arg(&mut hd),
            crate::arg(&mut dt),
            crate::arg(&mut old_bp),
            crate::arg(&mut new_bp),
            crate::arg(&mut is_post),
            crate::arg(&mut theta),
        ],
    )
}

/// Launch GPU-accelerated fused RoPE + scatter KV cache blending kernel (Task 10.7).
pub fn launch_blend_kv_rope(
    dev: &RocmDevice,
    cfg: &BlendConfig,
    k_src: &RocmStorage,
    v_src: &RocmStorage,
    k_dst: &RocmStorage,
    v_dst: &RocmStorage,
    base_pos: i32,
    rope_theta: f32,
) -> Result<*mut c_void> {
    launch_blend_kv_rope_transform(
        dev, cfg, k_src, v_src, k_dst, v_dst, None, base_pos, rope_theta,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blend_kv_rope_cpu_preserves_prefix() {
        let cfg = BlendConfig {
            block_size: 4,
            num_heads: 2,
            head_dim: 4,
            divergence_token: 2,
        };

        let total_len = cfg.block_size * cfg.num_heads * cfg.head_dim;
        let k_src = vec![1.0f32; total_len];
        let v_src = vec![2.0f32; total_len];

        // Destination initialized with previous cached prefix in tokens 0..2
        let mut k_dst = vec![0.0f32; total_len];
        let mut v_dst = vec![0.0f32; total_len];
        k_dst[..16].fill(7.0); // 2 tokens * 2 heads * 4 dim = 16 elements
        v_dst[..16].fill(8.0);

        blend_kv_rope_cpu(&cfg, &k_src, &v_src, &mut k_dst, &mut v_dst, 0, 10000.0).unwrap();

        // Tokens 0..2 should stay untouched
        assert_eq!(&k_dst[..16], &[7.0; 16]);
        assert_eq!(&v_dst[..16], &[8.0; 16]);

        // Tokens 2..4 in V should be updated with v_src (2.0)
        assert_eq!(&v_dst[16..], &[2.0; 16]);

        // Tokens 2..4 in K should be rotated and finite
        assert!(k_dst[16..].iter().all(|x| x.is_finite()));
    }

    #[test]
    #[ignore]
    fn test_blend_kv_rope_gpu_parity() {
        let dev = match RocmDevice::try_new(0) {
            Ok(d) => d,
            Err(_) => return,
        };

        let cfg = BlendConfig {
            block_size: 4,
            num_heads: 2,
            head_dim: 4,
            divergence_token: 2,
        };

        let total_len = cfg.block_size * cfg.num_heads * cfg.head_dim;
        let k_src = vec![1.0f32; total_len];
        let v_src = vec![2.0f32; total_len];

        let mut k_dst_cpu = vec![0.0f32; total_len];
        let mut v_dst_cpu = vec![0.0f32; total_len];
        k_dst_cpu[..16].fill(7.0);
        v_dst_cpu[..16].fill(8.0);

        let k_dst_gpu_init = k_dst_cpu.clone();
        let v_dst_gpu_init = v_dst_cpu.clone();

        blend_kv_rope_cpu(&cfg, &k_src, &v_src, &mut k_dst_cpu, &mut v_dst_cpu, 0, 10000.0).unwrap();

        use crate::as_rocm;
        use grim_tensor::backend::CoreTensorOps;
        let shape = grim_tensor::Shape::new(vec![total_len]);
        let k_src_dev = CoreTensorOps::from_cpu(&dev, &k_src, &shape, grim_tensor::DType::F32).unwrap();
        let v_src_dev = CoreTensorOps::from_cpu(&dev, &v_src, &shape, grim_tensor::DType::F32).unwrap();
        let k_dst_dev = CoreTensorOps::from_cpu(&dev, &k_dst_gpu_init, &shape, grim_tensor::DType::F32).unwrap();
        let v_dst_dev = CoreTensorOps::from_cpu(&dev, &v_dst_gpu_init, &shape, grim_tensor::DType::F32).unwrap();

        let k_src_s = as_rocm(k_src_dev.as_ref()).unwrap();
        let v_src_s = as_rocm(v_src_dev.as_ref()).unwrap();
        let k_dst_s = as_rocm(k_dst_dev.as_ref()).unwrap();
        let v_dst_s = as_rocm(v_dst_dev.as_ref()).unwrap();

        launch_blend_kv_rope(&dev, &cfg, k_src_s, v_src_s, k_dst_s, v_dst_s, 0, 10000.0).unwrap();
        dev.synchronize();

        let k_dst_gpu = k_dst_dev.to_cpu_vec_f32().unwrap();
        let v_dst_gpu = v_dst_dev.to_cpu_vec_f32().unwrap();

        assert_eq!(k_dst_cpu.len(), k_dst_gpu.len());
        assert_eq!(v_dst_cpu.len(), v_dst_gpu.len());

        for (i, (&c, &g)) in k_dst_cpu.iter().zip(k_dst_gpu.iter()).enumerate() {
            let diff = (c - g).abs();
            assert!(diff < 1e-5, "k_dst mismatch at {i}: cpu {c} vs gpu {g}");
        }
        for (i, (&c, &g)) in v_dst_cpu.iter().zip(v_dst_gpu.iter()).enumerate() {
            let diff = (c - g).abs();
            assert!(diff < 1e-5, "v_dst mismatch at {i}: cpu {c} vs gpu {g}");
        }
    }
}
