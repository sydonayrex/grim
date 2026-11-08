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
    /// Rotary dimension (ggml `n_rot`). Only dims `[0, rotary_dim)` rotate;
    /// `[rotary_dim, head_dim)` pass through untouched. Must be nonzero,
    /// even, and `<= head_dim`. For full-rotary models set it to `head_dim`.
    /// (rerope-golden rule: the freq denominator is `rotary_dim`, never
    /// `head_dim` — the partial-rotary trap behind the Gate D root cause.)
    pub rotary_dim: usize,
    /// GPT-J interleaved pairing (adjacent elements) instead of NeoX
    /// half-split pairing within the rotary region.
    pub interleaved: bool,
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
    let n_rot = cfg.rotary_dim;
    if n_rot == 0 || n_rot > cfg.head_dim || n_rot % 2 != 0 || cfg.head_dim % 2 != 0 {
        return Err(Error::Backend(format!(
            "blend_kv_rope: invalid rotary_dim {n_rot} for head_dim {} (need 0 < n_rot <= head_dim, both even)",
            cfg.head_dim
        )));
    }

    // Copy divergent tail for V (no RoPE on V)
    let tail_start = cfg.divergence_token * elem_per_token;
    v_dst[tail_start..].copy_from_slice(&v_src[tail_start..]);

    // Apply RoPE and copy divergent tail for K. Golden conventions
    // (crates/grim-backend-cpu/tests/rerope_golden.rs): freq denominator is
    // n_rot; pairing lives entirely within [0, n_rot); dims >= n_rot pass
    // through untouched; retargeting applies no mscale (the cached K already
    // carries M·R(old)·x, so unrotate→rotate is a pure delta rotation).
    let half_rot = n_rot / 2;
    for t in cfg.divergence_token..cfg.block_size {
        let token_offset = t * elem_per_token;

        for h in 0..cfg.num_heads {
            let head_offset = token_offset + h * cfg.head_dim;

            // Pass-through dims beyond the rotary region
            for d in n_rot..cfg.head_dim {
                k_dst[head_offset + d] = k_src[head_offset + d];
            }

            for d in 0..half_rot {
                let freq = 1.0 / (rope_theta.powf((2 * d) as f32 / n_rot as f32));
                let delta_pos = match old_base_pos {
                    Some(old) => (new_base_pos as f32) - (old as f32),
                    None => (new_base_pos + t) as f32,
                };
                let angle = delta_pos * freq;
                let (sin_val, cos_val) = angle.sin_cos();

                let (i0, i1) = if cfg.interleaved {
                    (2 * d, 2 * d + 1)
                } else {
                    (d, d + half_rot)
                };

                let k0 = k_src[head_offset + i0];
                let k1 = k_src[head_offset + i1];

                // Rotary rotation by angle (delta_pos)
                k_dst[head_offset + i0] = k0 * cos_val - k1 * sin_val;
                k_dst[head_offset + i1] = k0 * sin_val + k1 * cos_val;
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
    int rotary_dim,
    int interleaved,
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
    int half_rot = rotary_dim / 2;

    // Golden conventions (grim-backend-cpu/tests/rerope_golden.rs):
    // freq denominator is n_rot (rotary_dim), never head_dim; dims beyond
    // the rotary region pass through untouched; no mscale on retarget.
    if (d >= rotary_dim) {
        int head_offset = token_idx * elem_per_token + h * head_dim;
        k_dst[tid] = k_src[head_offset + d];
        return;
    }

    float delta_pos = (is_post_rope != 0)
        ? (float)(new_base_pos - old_base_pos)
        : (float)(new_base_pos + token_idx);

    int d_base = (d < half_rot) ? d : (d - half_rot);
    int f_idx = (interleaved != 0) ? (d / 2) : d_base;
    float freq = 1.0f / powf(rope_theta, (float)(2 * f_idx) / (float)rotary_dim);
    float angle = delta_pos * freq;
    float sin_val, cos_val;
    sincosf(angle, &sin_val, &cos_val);

    int head_offset = token_idx * elem_per_token + h * head_dim;
    int i0, i1;
    if (interleaved != 0) {
        // GPT-J pairing: adjacent elements within the rotary region
        int base = (d / 2) * 2;
        i0 = base;
        i1 = base + 1;
    } else {
        // NeoX half-split within [0, rotary_dim)
        i0 = d_base;
        i1 = d_base + half_rot;
    }

    float k0 = k_src[head_offset + i0];
    float k1 = k_src[head_offset + i1];

    if (d == i0) {
        k_dst[tid] = k0 * cos_val - k1 * sin_val;
    } else {
        k_dst[tid] = k0 * sin_val + k1 * cos_val;
    }
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
    let mut nrot = cfg.rotary_dim as i32;
    let mut inter = if cfg.interleaved { 1i32 } else { 0i32 };
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
            crate::arg(&mut nrot),
            crate::arg(&mut inter),
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
            rotary_dim: 4,
            interleaved: false,
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
            rotary_dim: 4,
            interleaved: false,
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

    /// Golden-semantics gate (crates/grim-backend-cpu/tests/rerope_golden.rs
    /// conventions applied to the blend path):
    /// 1. partial rotary — dims >= n_rot pass through bit-exact;
    /// 2. pure retargeting — rotate(old) then unrotate(old)->rotate(new)
    ///    equals a direct forward rotation at new;
    /// 3. freq denominator is n_rot, not head_dim (Gate D root cause).
    #[test]
    fn test_blend_partial_rotary_golden_semantics() {
        // Bonsai-27B-like geometry in miniature: head_dim 8, rotary_dim 2.
        let cfg = BlendConfig {
            block_size: 4,
            num_heads: 2,
            head_dim: 8,
            divergence_token: 0,
            rotary_dim: 2,
            interleaved: false,
        };
        let total_len = cfg.block_size * cfg.num_heads * cfg.head_dim;
        let raw_k: Vec<f32> = (0..total_len).map(|i| ((i as f32) * 0.37) - 3.1).collect();
        let v_src = vec![9.0f32; total_len];
        let theta = 10000.0f32;
        let (old_pos, new_pos) = (64usize, 128usize);
        let mut dummy_v = vec![0.0f32; total_len];

        // 1. Pass-through: rotate raw at old; dims >= n_rot must be bit-exact.
        let mut at_old = vec![0.0f32; total_len];
        blend_kv_rope_cpu(&cfg, &raw_k, &v_src, &mut at_old, &mut dummy_v, old_pos, theta).unwrap();
        let elem = cfg.num_heads * cfg.head_dim;
        for t in 0..cfg.block_size {
            for h in 0..cfg.num_heads {
                let off = t * elem + h * cfg.head_dim;
                for d in cfg.rotary_dim..cfg.head_dim {
                    assert_eq!(
                        at_old[off + d], raw_k[off + d],
                        "pass-through dim {d} mutated at token {t} head {h}"
                    );
                }
            }
        }

        // 2. Pure retargeting: at_old -> retarget(new) == direct rotate at new.
        let mut retargeted = vec![0.0f32; total_len];
        blend_kv_rope_transform_cpu(
            &cfg, &at_old, &v_src, &mut retargeted, &mut dummy_v, Some(old_pos), new_pos, theta,
        )
        .unwrap();
        let mut direct = vec![0.0f32; total_len];
        blend_kv_rope_cpu(&cfg, &raw_k, &v_src, &mut direct, &mut dummy_v, new_pos, theta).unwrap();
        for i in 0..total_len {
            let diff = (retargeted[i] - direct[i]).abs();
            assert!(diff < 1e-5, "retarget != direct at {i}: {diff}");
        }

        // 3. Freq denominator: rotation of a pure pair [1,0] at pos 1 must
        //    use theta^(-2i/n_rot). With n_rot=2 the only freq is
        //    1/theta^(0/2)=1, so pair 0 at pos 1 rotates by exactly 1 rad.
        let one_pair = BlendConfig {
            block_size: 1,
            num_heads: 1,
            head_dim: 4,
            divergence_token: 0,
            rotary_dim: 2,
            interleaved: false,
        };
        let k_in = vec![1.0f32, 0.0, 5.0, 5.0];
        let mut k_out = vec![0.0f32; 4];
        blend_kv_rope_cpu(&one_pair, &k_in, &v_src[..4], &mut k_out, &mut dummy_v[..4], 1, theta)
            .unwrap();
        let (s, c) = 1.0f32.sin_cos();
        assert!((k_out[0] - c).abs() < 1e-6, "k_out[0]={}", k_out[0]);
        assert!((k_out[1] - s).abs() < 1e-6, "k_out[1]={}", k_out[1]);
        // head_dim 4, n_rot 2: dims 2..4 pass through
        assert_eq!(k_out[2], 5.0);
        assert_eq!(k_out[3], 5.0);

        // 4. GPT-J interleaved: pair (x0,x1) rotates with freq idx 0,
        //    pair (x2,x3) with freq idx 1 = 1/theta^(2/4).
        let gptj = BlendConfig {
            block_size: 1,
            num_heads: 1,
            head_dim: 4,
            divergence_token: 0,
            rotary_dim: 4,
            interleaved: true,
        };
        let k_in = vec![1.0f32, 0.0, 1.0, 0.0];
        let mut k_out = vec![0.0f32; 4];
        blend_kv_rope_cpu(&gptj, &k_in, &v_src[..4], &mut k_out, &mut dummy_v[..4], 1, theta).unwrap();
        let (s0, c0) = 1.0f32.sin_cos();
        let f1 = 1.0f32 / theta.powf(2.0 / 4.0);
        let (s1, c1) = f1.sin_cos();
        assert!((k_out[0] - c0).abs() < 1e-6);
        assert!((k_out[1] - s0).abs() < 1e-6);
        assert!((k_out[2] - c1).abs() < 1e-6);
        assert!((k_out[3] - s1).abs() < 1e-6);
    }

    /// GPU parity on a partial-rotary post-rope retarget (Bonsai-like shape:
    /// head_dim 8, rotary_dim 2), CPU twin as oracle.
    #[test]
    #[ignore]
    fn test_blend_kv_rope_gpu_partial_rotary_retarget_parity() {
        let dev = match RocmDevice::try_new(0) {
            Ok(d) => d,
            Err(_) => return,
        };

        let cfg = BlendConfig {
            block_size: 8,
            num_heads: 3,
            head_dim: 8,
            divergence_token: 3,
            rotary_dim: 2,
            interleaved: false,
        };
        let total_len = cfg.block_size * cfg.num_heads * cfg.head_dim;
        let k_src: Vec<f32> = (0..total_len).map(|i| ((i as f32) * 0.23) - 1.7).collect();
        let v_src: Vec<f32> = (0..total_len).map(|i| ((i as f32) * 0.11) + 0.5).collect();
        let (old_pos, new_pos) = (64i32, 192i32);
        let theta = 10000.0f32;

        let mut k_dst_cpu = vec![7.0f32; total_len];
        let mut v_dst_cpu = vec![3.0f32; total_len];
        blend_kv_rope_transform_cpu(
            &cfg, &k_src, &v_src, &mut k_dst_cpu, &mut v_dst_cpu, Some(old_pos as usize), new_pos as usize, theta,
        )
        .unwrap();

        use crate::as_rocm;
        use grim_tensor::backend::CoreTensorOps;
        let shape = grim_tensor::Shape::new(vec![total_len]);
        let k_src_dev = CoreTensorOps::from_cpu(&dev, &k_src, &shape, grim_tensor::DType::F32).unwrap();
        let v_src_dev = CoreTensorOps::from_cpu(&dev, &v_src, &shape, grim_tensor::DType::F32).unwrap();
        let k_dst_dev = CoreTensorOps::from_cpu(&dev, &vec![7.0f32; total_len], &shape, grim_tensor::DType::F32).unwrap();
        let v_dst_dev = CoreTensorOps::from_cpu(&dev, &vec![3.0f32; total_len], &shape, grim_tensor::DType::F32).unwrap();

        let k_src_s = as_rocm(k_src_dev.as_ref()).unwrap();
        let v_src_s = as_rocm(v_src_dev.as_ref()).unwrap();
        let k_dst_s = as_rocm(k_dst_dev.as_ref()).unwrap();
        let v_dst_s = as_rocm(v_dst_dev.as_ref()).unwrap();

        launch_blend_kv_rope_transform(
            &dev, &cfg, k_src_s, v_src_s, k_dst_s, v_dst_s, Some(old_pos), new_pos, theta,
        )
        .unwrap();
        dev.synchronize();

        let k_gpu = k_dst_dev.to_cpu_vec_f32().unwrap();
        let v_gpu = v_dst_dev.to_cpu_vec_f32().unwrap();
        for (i, (&c, &g)) in k_dst_cpu.iter().zip(k_gpu.iter()).enumerate() {
            let diff = (c - g).abs();
            assert!(diff < 1e-5, "k mismatch at {i}: cpu {c} vs gpu {g}");
        }
        for (i, (&c, &g)) in v_dst_cpu.iter().zip(v_gpu.iter()).enumerate() {
            let diff = (c - g).abs();
            assert!(diff < 1e-5, "v mismatch at {i}: cpu {c} vs gpu {g}");
        }
    }
}
