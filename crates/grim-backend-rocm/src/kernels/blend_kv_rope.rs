//! Fused RoPE + Scatter KV Cache Blending Kernel.
//!
//! Reuses partially-matched prefix block contents (tokens `0..divergence_token`),
//! computes rotary positional embeddings (RoPE) only for the divergent token tail
//! (`divergence_token..block_size`), and scatters the merged result into the target block.

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
///
/// Preserves `k_dst[..divergence_token]` and `v_dst[..divergence_token]` from existing cache,
/// and applies RoPE to `k_src[divergence_token..]` before scattering into `k_dst` and `v_dst`.
pub fn blend_kv_rope_cpu(
    cfg: &BlendConfig,
    k_src: &[f32],
    v_src: &[f32],
    k_dst: &mut [f32],
    v_dst: &mut [f32],
    base_pos: usize,
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
            cfg.divergence_token,
            cfg.block_size
        )));
    }

    // Copy divergent tail for V (no RoPE on V)
    let tail_start = cfg.divergence_token * elem_per_token;
    v_dst[tail_start..].copy_from_slice(&v_src[tail_start..]);

    // Apply RoPE and copy divergent tail for K
    let half_dim = cfg.head_dim / 2;
    for t in cfg.divergence_token..cfg.block_size {
        let pos = (base_pos + t) as f32;
        let token_offset = t * elem_per_token;

        for h in 0..cfg.num_heads {
            let head_offset = token_offset + h * cfg.head_dim;

            for d in 0..half_dim {
                let freq = 1.0 / (rope_theta.powf((2 * d) as f32 / cfg.head_dim as f32));
                let angle = pos * freq;
                let (sin_val, cos_val) = angle.sin_cos();

                let k0 = k_src[head_offset + d];
                let k1 = k_src[head_offset + d + half_dim];

                // Rotary rotation
                k_dst[head_offset + d] = k0 * cos_val - k1 * sin_val;
                k_dst[head_offset + d + half_dim] = k0 * sin_val + k1 * cos_val;
            }
        }
    }

    Ok(())
}

/// GPU HipRTC kernel source template for fused RoPE + scatter cache blending.
pub const BLEND_KV_ROPE_HIP_SRC: &str = r#"
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

    int pos = base_pos + token_idx;
    int d_base = (d < half_dim) ? d : (d - half_dim);
    float freq = 1.0f / __powf(rope_theta, (float)(2 * d_base) / (float)head_dim);
    float angle = (float)pos * freq;
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
"#;

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
}
