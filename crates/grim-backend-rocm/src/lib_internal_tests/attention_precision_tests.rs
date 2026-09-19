//! attention precision tests — split from the original monolithic lib_internal_tests.rs.

#[cfg(test)]
mod tests {
    use crate::*;

    const GPU_TEST_ENV: &str = "GRIM_GPU_TEST";

    #[test]
    fn test_is_attention_projection() {
        let cases = &[
            ("blk.48.attn_q.weight", true),
            ("blk.48.attn_k.weight", true),
            ("blk.48.attn_v.weight", true),
            ("blk.48.attn_o.weight", true),
            ("model.embed_tokens.weight", false),
            ("model.layers.48.mlp.gate_proj.weight", false),
            ("model.layers.48.mlp.up_proj.weight", false),
            ("model.layers.48.mlp.down_proj.weight", false),
            ("blk.48.ffn_gate", false),
            ("self_attn.q_proj.weight", true),
            ("self_attn.k_proj.weight", true),
            ("self_attn.v_proj.weight", true),
            ("self_attn.o_proj.weight", true),
        ];
        for (name, expected) in cases {
            assert_eq!(
                is_attention_projection(name),
                *expected,
                "failed for {name}"
            );
        }
    }

    #[test]
    fn test_enforce_attention_precision() {
        assert_eq!(enforce_attention_precision(3), 5);
        assert_eq!(enforce_attention_precision(4), 5);
        assert_eq!(enforce_attention_precision(5), 5);
        assert_eq!(enforce_attention_precision(6), 6);
        assert_eq!(enforce_attention_precision(8), 8);
    }

    #[test]
    fn test_attention_min_bpw() {
        assert_eq!(attention_min_bpw(), 5);
    }

    #[test]
    fn test_qkv_attention_large_head_dim_compiles() {
        if !crate::gpu_test_enabled() {
            return;
        }
        let kernel_source = crate::kernels::source_asm::compute_kernel_source();
        let target = detect_gpu_arch(0);
        let res = jit_compile_hsaco(&kernel_source, "grim_qkv_attention", &target);
        assert!(
            res.is_ok(),
            "Failed to JIT compile grim_qkv_attention with large head_dim support: {:?}",
            res.err()
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn woody_attention_online_f32(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        kv_seq_len: usize,
        cache_offset: u32,
    ) -> Vec<f32> {
        assert!(
            num_heads % num_kv_heads == 0,
            "GQA: num_heads must be multiple of num_kv_heads"
        );
        let q_per_kv = num_heads / num_kv_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let q_stride = num_heads * head_dim;
        let kv_stride = num_kv_heads * head_dim;
        let mut out = vec![0.0f32; seq_len * num_heads * head_dim];

        for h in 0..num_heads {
            let kv_head = h / q_per_kv;
            for qt in 0..seq_len {
                let abs_i = (cache_offset as usize) + qt;
                let hi = (abs_i + 1).min(kv_seq_len);

                // Per-d online softmax running state.
                let mut acc = vec![0.0f32; head_dim];
                let mut running_max = vec![f32::NEG_INFINITY; head_dim];
                let mut running_sum = vec![0.0f32; head_dim];

                for j in 0..hi {
                    // Score = (q · k[j]) * scale
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[qt * q_stride + h * head_dim + d]
                            * k[j * kv_stride + kv_head * head_dim + d];
                    }
                    let s = dot * scale;
                    for d in 0..head_dim {
                        let prev_m = running_max[d];
                        // Stable online softmax update.
                        let new_m = if s > prev_m { s } else { prev_m };
                        // scale = exp(prev_m - new_m): 1.0 when prev_m == -inf
                        let scale_prev = if new_m == f32::NEG_INFINITY {
                            0.0
                        } else {
                            (prev_m - new_m).exp()
                        };
                        running_sum[d] *= scale_prev;
                        acc[d] *= scale_prev;
                        running_max[d] = new_m;
                        // Weight for this j.
                        let w = if s == new_m {
                            1.0f32
                        } else {
                            (s - new_m).exp()
                        };
                        running_sum[d] += w;
                        acc[d] += w * v[j * kv_stride + kv_head * head_dim + d];
                    }
                }

                // Final write: out = acc / sum (with F5 zero-guard for empty ranges).
                for d in 0..head_dim {
                    let denom = running_sum[d];
                    out[qt * q_stride + h * head_dim + d] =
                        if denom > 0.0 { acc[d] / denom } else { 0.0 };
                }
            }
        }
        out
    }

    fn lcg_f32(seed: u32) -> Vec<f32> {
        // Wyrand-style: Cheap and reproducible in f32.
        let mut state = seed.wrapping_add(0x9E3779B9);
        let mut out = Vec::new();
        for _ in 0..4096 {
            state = state.wrapping_mul(0x85EBCA6B).wrapping_add(0xC2B2AE35);
            let x = (state as f32) / (u32::MAX as f32) * 4.0 - 2.0; // ~[-2, 2]
            out.push(x);
        }
        out
    }

    fn run_qkv_attention(
        env_present: bool,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        num_kv_heads: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Option<Vec<f32>> {
        if !env_present {
            return None;
        }
        let dev = RocmDevice::new(0);
        let q_s = dev
            .from_cpu(
                q,
                &Shape::from_slice(&[seq_len, num_heads, head_dim]),
                DType::F32,
            )
            .ok()?;
        let k_s = dev
            .from_cpu(
                k,
                &Shape::from_slice(&[kv_seq_len, num_kv_heads, head_dim]),
                DType::F32,
            )
            .ok()?;
        let v_s = dev
            .from_cpu(
                v,
                &Shape::from_slice(&[kv_seq_len, num_kv_heads, head_dim]),
                DType::F32,
            )
            .ok()?;
        let (out, _h) = dev
            .qkv_attention(
                q_s.as_ref(),
                k_s.as_ref(),
                v_s.as_ref(),
                num_kv_heads,
                kv_seq_len,
                cache_offset,
                None, // window: full causal
                &Shape::from_slice(&[seq_len, num_heads, head_dim]),
                None,
                None,
            )
            .ok()?;
        out.to_cpu_vec_f32().ok()
    }

    fn approx_close(a: f32, b: f32, abs_tol: f32, rel_tol: f32) -> bool {
        let diff = (a - b).abs();
        let scale = a.abs().max(b.abs());
        diff <= abs_tol.max(rel_tol * scale)
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "len mismatch: {} vs {}", a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn shape_fixture() -> (usize, usize, usize, usize) {
        // (num_heads, num_kv_heads, head_dim, seq_len)
        (8, 4, 32, 4)
    }

    fn build_inputs(
        seed: u32,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
        kv_seq_len: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let q_len = seq_len * num_heads * head_dim;
        let kv_len = kv_seq_len * num_kv_heads * head_dim;
        let mut stream = lcg_f32(seed);
        let mut take = |n: usize| -> Vec<f32> {
            if stream.len() < n {
                // deterministic refill for big shapes
                let mut s = (seed.wrapping_add(n as u32)).wrapping_add(0x9E3779B9);
                for _ in 0..n {
                    s = s.wrapping_mul(0x85EBCA6B).wrapping_add(0xC2B2AE35);
                    stream.push((s as f32) / (u32::MAX as f32) * 4.0 - 2.0);
                }
            }
            stream.drain(..n).collect()
        };
        let q = take(q_len);
        let k = take(kv_len);
        let v = take(kv_len);
        (q, k, v)
    }

    #[test]
    fn wi1_qkv_attention_kvseq_mod4_eq0() {
        let _ = approx_close; // silence dead-code warning for helper-only tests
        let (nh, nkv, hd, sl) = shape_fixture();
        let kv_seq = 64usize; // divisible by 4
        let cache_off = 4u32; // ensures causal path active
        let (q, k, v) = build_inputs(0xA1, nh, nkv, hd, sl, kv_seq);
        let cpu = woody_attention_online_f32(&q, &k, &v, sl, nh, nkv, hd, kv_seq, cache_off);
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        let _gpu_guard = env.then(crate::device::util::gpu_test_lock);
        let got = run_qkv_attention(env, &q, &k, &v, nkv, kv_seq, cache_off, sl, nh, hd);
        if let Some(out) = got {
            let max = max_abs_diff(&out, &cpu);
            assert!(max <= 1e-3, "wi1 mod4=0 max_abs_diff {} too large", max);
        }
    }

    #[test]
    fn wi1_qkv_attention_kvseq_mod4_ne0() {
        let _ = approx_close;
        let (nh, nkv, hd, sl) = shape_fixture();
        let kv_seq = 65usize; // 65 mod 4 == 1 — splits unevenly across waves
        let cache_off = 16u32; // forces 17 valid js, all in the past window
        let (q, k, v) = build_inputs(0xB2, nh, nkv, hd, sl, kv_seq);
        let cpu = woody_attention_online_f32(&q, &k, &v, sl, nh, nkv, hd, kv_seq, cache_off);
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        let _gpu_guard = env.then(crate::device::util::gpu_test_lock);
        let got = run_qkv_attention(env, &q, &k, &v, nkv, kv_seq, cache_off, sl, nh, hd);
        if let Some(out) = got {
            let max = max_abs_diff(&out, &cpu);
            assert!(max <= 1e-3, "wi1 mod4!=0 max_abs_diff {} too large", max);
        }
    }

    #[test]
    fn wi1_qkv_attention_kvseq_lt_4() {
        let _ = approx_close;
        let (nh, nkv, hd, sl) = shape_fixture();
        let kv_seq = 3usize; // smaller than wavefront count — most waves idle
        let cache_off = 0u32;
        let (q, k, v) = build_inputs(0xC3, nh, nkv, hd, sl, kv_seq);
        let cpu = woody_attention_online_f32(&q, &k, &v, sl, nh, nkv, hd, kv_seq, cache_off);
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        let _gpu_guard = env.then(crate::device::util::gpu_test_lock);
        let got = run_qkv_attention(env, &q, &k, &v, nkv, kv_seq, cache_off, sl, nh, hd);
        if let Some(out) = got {
            let max = max_abs_diff(&out, &cpu);
            assert!(max <= 1e-3, "wi1 kv<4 max_abs_diff {} too large", max);
        }
    }

    #[test]
    fn wi1_qkv_attention_kvseq_eq_1_bit_exact() {
        // kv_seq_len=1 has zero softmax-precision noise (only one valid j and
        let _ = approx_close;
        let (nh, nkv, hd, sl) = shape_fixture();
        let kv_seq = 1usize;
        let cache_off = 0u32;
        let (q, k, v) = build_inputs(0xD4, nh, nkv, hd, sl, kv_seq);
        let cpu = woody_attention_online_f32(&q, &k, &v, sl, nh, nkv, hd, kv_seq, cache_off);
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        let _gpu_guard = env.then(crate::device::util::gpu_test_lock);
        let got = run_qkv_attention(env, &q, &k, &v, nkv, kv_seq, cache_off, sl, nh, hd);
        if let Some(out) = got {
            let max = max_abs_diff(&out, &cpu);
            assert!(
                max <= 1e-5,
                "wi1 kv=1 (bit-exact) max_abs_diff {} too large",
                max
            );
        }
    }

    #[test]
    fn wi1_qkv_attention_skewed_short_seq() {
        // Different head_dim (32, num_heads=4, num_kv_heads=2) and a sharply
        let _ = approx_close;
        let nh = 4usize;
        let nkv = 2usize;
        let hd = 32usize;
        let sl = 2usize;
        let kv_seq = 1021usize;
        let cache_off = 0u32;
        let (q, k, v) = build_inputs(0xE5, nh, nkv, hd, sl, kv_seq);
        let cpu = woody_attention_online_f32(&q, &k, &v, sl, nh, nkv, hd, kv_seq, cache_off);
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        let _gpu_guard = env.then(crate::device::util::gpu_test_lock);
        let got = run_qkv_attention(env, &q, &k, &v, nkv, kv_seq, cache_off, sl, nh, hd);
        if let Some(out) = got {
            let max = max_abs_diff(&out, &cpu);
            assert!(max <= 5e-3, "wi1 skewed max_abs_diff {} too large", max);
        }
    }

}
