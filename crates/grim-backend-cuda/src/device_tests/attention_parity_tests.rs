use super::common::*;

#[cfg(test)]
mod tests {
    use super::*;
    use grim_tensor::dtype::DType;
    use grim_tensor::{CoreTensorOps, Shape};

    #[test]
    fn test_cuda_qkv_attention_parity() {
        let Some(dev) = dequant_test_device() else {
            return;
        };
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 64;
        let steps = 3;
        let kv_len = 5;
        let cache_offset = 2;

        let q: Vec<f32> = (0..steps * num_heads * head_dim)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let k: Vec<f32> = (0..kv_len * num_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.02).cos())
            .collect();
        let v: Vec<f32> = (0..kv_len * num_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.03).sin())
            .collect();

        let q_shape = Shape::new(vec![steps, num_heads, head_dim]);
        let kv_shape = Shape::new(vec![kv_len, num_kv_heads, head_dim]);
        let out_shape = Shape::new(vec![steps, num_heads * head_dim]);

        let q_s = dev.from_cpu(&q, &q_shape, DType::F32).unwrap();
        let k_s = dev.from_cpu(&k, &kv_shape, DType::F32).unwrap();
        let v_s = dev.from_cpu(&v, &kv_shape, DType::F32).unwrap();

        let (out_s, _) = dev
            .qkv_attention(
                q_s.as_ref(),
                k_s.as_ref(),
                v_s.as_ref(),
                num_kv_heads,
                kv_len,
                cache_offset,
                None,
                &out_shape,
                None,
                None,
            )
            .unwrap();

        let actual = out_s.to_cpu_vec_f32().unwrap();

        // CPU reference calculation:
        let scale = 1.0 / (head_dim as f32).sqrt();
        let kv_stride = num_kv_heads * head_dim;
        for t in 0..steps {
            for h in 0..num_heads {
                let kvh = (h * num_kv_heads) / num_heads;
                let causal_limit = cache_offset as usize + t;
                let mut scores = Vec::new();
                for t2 in 0..=causal_limit.min(kv_len - 1) {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[t * num_heads * head_dim + h * head_dim + d]
                            * k[t2 * kv_stride + kvh * head_dim + d];
                    }
                    scores.push(dot * scale);
                }
                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = scores.iter().map(|s| (s - mx).exp()).sum();
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for (i, s) in scores.iter().enumerate() {
                        acc += ((s - mx).exp() / sum) * v[i * kv_stride + kvh * head_dim + d];
                    }
                    let idx = t * num_heads * head_dim + h * head_dim + d;
                    assert!(
                        (actual[idx] - acc).abs() < 1e-4,
                        "Mismatch at t={t}, h={h}, d={d}: got {}, want {}",
                        actual[idx],
                        acc
                    );
                }
            }
        }
    }
}
