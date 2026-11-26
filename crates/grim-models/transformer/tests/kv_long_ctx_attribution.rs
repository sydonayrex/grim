//! Phase 2B: K-vs-V Quantization Attribution & Needle Retrieval Gate
//! (`F32` oracle vs `F16`, `Q8_1` K-only/V-only/both, `INT4` K-only/V-only/both,
//! and `K8V4` at `2k / 8k / 32k` on `Qwen3.8-27B` geometry `nh=24, nkv=4, hd=256`).

/// Round-trip f32 through IEEE-754 binary16 (10-bit mantissa) without external crate.
fn f16_roundtrip(x: f32) -> f32 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    let bits = x.to_bits();
    // Round mantissa to 10 bits (mask low 13 bits with round-to-nearest)
    let rounded = (bits.wrapping_add(1 << 12)) & !((1u32 << 13) - 1);
    f32::from_bits(rounded)
}

fn quant_dequant_q81_block(xs: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; xs.len()];
    for (src, dst) in xs.chunks(32).zip(out.chunks_mut(32)) {
        let amax = src.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let d = if amax > 0.0 {
            f16_roundtrip(amax / 127.0)
        } else {
            0.0
        };
        for (s, o) in src.iter().zip(dst.iter_mut()) {
            let q = if d > 0.0 {
                (s / d).round().clamp(-127.0, 127.0) as i8
            } else {
                0i8
            };
            *o = (q as f32) * d;
        }
    }
    out
}

fn quant_dequant_int4_block(xs: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; xs.len()];
    for (src, dst) in xs.chunks(32).zip(out.chunks_mut(32)) {
        let amax = src.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let d = if amax > 0.0 {
            f16_roundtrip(amax / 7.0)
        } else {
            0.0
        };
        for (s, o) in src.iter().zip(dst.iter_mut()) {
            let q = if d > 0.0 {
                (s / d).round().clamp(-7.0, 7.0) as i8
            } else {
                0i8
            };
            *o = (q as f32) * d;
        }
    }
    out
}

fn quant_dequant_f16(xs: &[f32]) -> Vec<f32> {
    xs.iter().map(|&v| f16_roundtrip(v)).collect()
}

fn exact_attention_single_query(q: &[f32], k: &[f32], v: &[f32], kv_len: usize, hd: usize) -> Vec<f32> {
    let inv_sqrt_d = 1.0f64 / (hd as f64).sqrt();
    let mut scores = vec![0.0f64; kv_len];
    let mut max_s = f64::NEG_INFINITY;
    for t in 0..kv_len {
        let krow = &k[t * hd..(t + 1) * hd];
        let mut dot = 0.0f64;
        for d in 0..hd {
            dot += (q[d] as f64) * (krow[d] as f64);
        }
        let s = dot * inv_sqrt_d;
        scores[t] = s;
        if s > max_s {
            max_s = s;
        }
    }
    let mut sum = 0.0f64;
    for s in &mut scores {
        *s = (*s - max_s).exp();
        sum += *s;
    }
    let mut out = vec![0.0f64; hd];
    for t in 0..kv_len {
        let p = scores[t] / sum;
        let vrow = &v[t * hd..(t + 1) * hd];
        for d in 0..hd {
            out[d] += p * (vrow[d] as f64);
        }
    }
    out.into_iter().map(|x| x as f32).collect()
}

fn metrics(ref_out: &[f32], cand: &[f32]) -> (f32, f32, f32) {
    let mut diffs: Vec<f32> = ref_out
        .iter()
        .zip(cand.iter())
        .map(|(a, b)| (a - b).abs())
        .collect();
    let mse: f64 = diffs.iter().map(|&d| (d as f64) * (d as f64)).sum::<f64>() / (diffs.len() as f64);
    let rmse = mse.sqrt() as f32;
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p99 = diffs[(diffs.len() * 99) / 100];
    let max_abs = *diffs.last().unwrap_or(&0.0);
    (rmse, p99, max_abs)
}

#[test]
fn kv_k_vs_v_attribution_and_needle_retrieval() {
    let hd = 256usize;
    let mut rng = 0x2026_1126_0000_2b01u64;
    let mut next_f = || -> f32 {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((rng >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };

    for &kv_len in &[2048usize, 8192, 32_768, 65_536, 131_072, 200_000, 262_144] {
        let mut q: Vec<f32> = (0..hd).map(|_| next_f() * 0.5).collect();
        let mut k: Vec<f32> = (0..kv_len * hd).map(|_| next_f() * 0.5).collect();
        let mut v: Vec<f32> = (0..kv_len * hd).map(|_| next_f() * 0.5).collect();

        // Plant a distinct needle at 50% depth with high Q.K alignment (> ln(32768)) and known V signature
        let needle_pos = kv_len / 2;
        for d in 0..hd {
            k[needle_pos * hd + d] = q[d] * 8.0;
            v[needle_pos * hd + d] = if d % 2 == 0 { 2.0 } else { -2.0 };
        }
        // Add a small outlier channel on K to mirror real RoPE outlier behavior
        q[0] = 4.0;
        k[needle_pos * hd] = 4.0;

        let ref_out = exact_attention_single_query(&q, &k, &v, kv_len, hd);
        let k_f16 = quant_dequant_f16(&k);
        let v_f16 = quant_dequant_f16(&v);
        let k_q8 = quant_dequant_q81_block(&k);
        let v_q8 = quant_dequant_q81_block(&v);
        let k_i4 = quant_dequant_int4_block(&k);
        let v_i4 = quant_dequant_int4_block(&v);

        let arms: [(&str, &[f32], &[f32]); 7] = [
            ("F16 (K+V)     ", &k_f16, &v_f16),
            ("Q8_1 (K-only) ", &k_q8, &v),
            ("Q8_1 (V-only) ", &k, &v_q8),
            ("Q8_1 (K+V)    ", &k_q8, &v_q8),
            ("INT4 (K-only) ", &k_i4, &v),
            ("INT4 (V-only) ", &k, &v_i4),
            ("K8V4 (Q8K+I4V)", &k_q8, &v_i4),
        ];

        eprintln!("\n=== Phase 2B KV Attribution @ kv_len={kv_len} (hd={hd}, needle@{needle_pos}) ===");
        eprintln!("{:<15} {:>10} {:>10} {:>10} {:>8}", "arm", "rmse", "p99", "max_abs", "needle");
        for (name, ka, va) in arms {
            let out = exact_attention_single_query(&q, ka, va, kv_len, hd);
            let (rmse, p99, max_abs) = metrics(&ref_out, &out);
            // Needle retrieval check: sign agreement with planted V signature (+2/-2)
            let needle_ok = out
                .iter()
                .enumerate()
                .all(|(d, &val)| (d % 2 == 0 && val > 0.5) || (d % 2 == 1 && val < -0.5));
            eprintln!("{name:<15} {rmse:>10.3e} {p99:>10.3e} {max_abs:>10.3e} {:>8}", if needle_ok { "PASS" } else { "FAIL" });
            if name.starts_with("Q8_1 (K+V)") {
                assert!(needle_ok, "Q8_1 (K+V) must pass needle retrieval at {kv_len}");
                assert!(rmse < 2e-2, "Q8_1 (K+V) RMSE too high at {kv_len}: {rmse}");
            }
        }
    }
}
