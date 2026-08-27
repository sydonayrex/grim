//! WI-X2/BISECT probe: `dev.qkv_attention` correctness at the LFM2.5 shape
//! (num_heads == num_kv_heads == 16, head_dim 64) across prefill/decode and
//! cache-offset variants, against an inline scalar reference.
//! Env-gated: GRIM_RUN_GPU_TESTS=1.

use grim_backend_rocm::{BackendDevice, RocmDevice, Shape};
use grim_tensor::DType;

// Argument list mirrors the GPU kernel's launch parameters one-to-one.
#[allow(clippy::too_many_arguments)]
fn reference_attention(
    q: &[f32], // [steps, heads*dim]
    k: &[f32], // [kv_len, kv_heads*dim]
    v: &[f32],
    steps: usize,
    _kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    cache_offset: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; steps * num_heads * head_dim];
    let scale = 1.0 / (head_dim as f32).sqrt();
    for t in 0..steps {
        for h in 0..num_heads {
            let kvh = if num_kv_heads == num_heads {
                h
            } else {
                (h * num_kv_heads) / num_heads
            };
            let limit = cache_offset + t + 1; // causal
            let mut scores = vec![0.0f32; limit];
            for t2 in 0..limit {
                let mut dot = 0.0;
                for d in 0..head_dim {
                    dot += q[t * num_heads * head_dim + h * head_dim + d]
                        * k[t2 * num_kv_heads * head_dim + kvh * head_dim + d];
                }
                scores[t2] = dot * scale;
            }
            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = scores.iter().map(|s| (s - mx).exp()).sum();
            for d in 0..head_dim {
                let mut acc = 0.0;
                for (t2, &s) in scores.iter().enumerate() {
                    acc +=
                        (s - mx).exp() / sum * v[t2 * num_kv_heads * head_dim + kvh * head_dim + d];
                }
                out[t * num_heads * head_dim + h * head_dim + d] = acc;
            }
        }
    }
    out
}

fn run_case(dev: &RocmDevice, label: &str, steps: usize, history: usize, offset: usize) -> bool {
    let heads = 16usize;
    let kv_heads = 16usize;
    let dim = 64usize;
    let kv_len = offset + steps;
    assert_eq!(kv_len, history);

    let mut seed = 0xC0FFEEu64;
    let mut rand = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let q: Vec<f32> = (0..steps * heads * dim).map(|_| rand()).collect();
    // k/v cover the FULL kv_len (arena style); the kernel sees the whole buffer
    // and uses cache_offset/kv_seq_len to know what's live.
    let k_full: Vec<f32> = (0..history * kv_heads * dim).map(|_| rand()).collect();
    let v_full: Vec<f32> = (0..history * kv_heads * dim).map(|_| rand()).collect();

    // Match the loader reality: rope outputs carry a leading batch dim of 1.
    let q_shape = Shape::new(vec![1, steps * heads, dim]);
    let kv_shape = Shape::new(vec![1, history * kv_heads, dim]);
    let q_st = dev.from_cpu(&q, &q_shape, DType::F32).expect("q upload");
    let k_st = dev
        .from_cpu(&k_full, &kv_shape, DType::F32)
        .expect("k upload");
    let v_st = dev
        .from_cpu(&v_full, &kv_shape, DType::F32)
        .expect("v upload");

    let out_shape = Shape::new(vec![steps, heads, dim]);
    let res = dev.qkv_attention(
        q_st.as_ref(),
        k_st.as_ref(),
        v_st.as_ref(),
        kv_heads,
        kv_len,
        offset as u32,
        None,
        &out_shape,
        None,
        None,
    );
    let got = match res {
        Ok((s, _h)) => s.to_cpu_vec_f32().expect("download"),
        Err(e) => {
            println!("[{label}] kernel error: {e}");
            return false;
        }
    };
    let want = reference_attention(
        &q, &k_full, &v_full, steps, history, heads, kv_heads, dim, offset,
    );
    let max_diff = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let ok = max_diff < 5e-4;
    println!(
        "[{label}] steps={steps} kv_len={history} offset={offset}: max_diff={max_diff:.6} {}",
        if ok { "OK" } else { "MISMATCH" }
    );
    ok
}

#[test]
fn qkv_attention_matches_reference_at_lfm2_shapes() {
    if std::env::var("GRIM_RUN_GPU_TESTS").unwrap_or_default() != "1" {
        eprintln!("Skipping GPU test (set GRIM_RUN_GPU_TESTS=1)");
        return;
    }
    let dev = match RocmDevice::try_new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("ROCm device 0 not available: {e}");
            return;
        }
    };

    let mut all_ok = true;
    all_ok &= run_case(&dev, "prefill-from-empty", 24, 24, 0);
    all_ok &= run_case(&dev, "decode-step", 1, 25, 24);
    all_ok &= run_case(&dev, "chunked-prefill", 8, 32, 24);
    all_ok &= run_case(&dev, "lfm2-real-prefill", 2048, 2048, 0);
    assert!(all_ok, "grim_qkv_attention diverges from scalar reference");
}

// Sequential stress: 16 layers x mixed prefill/decode calls in one process,
// mirroring generation's call cadence. Detects cross-call state pollution.
#[test]
fn qkv_attention_survives_generation_cadence() {
    if std::env::var("GRIM_RUN_GPU_TESTS").unwrap_or_default() != "1" {
        eprintln!("Skipping GPU test (set GRIM_RUN_GPU_TESTS=1)");
        return;
    }
    let dev = match RocmDevice::try_new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("no device: {e}");
            return;
        }
    };
    let mut all_ok = true;
    for layer in 0..16 {
        all_ok &= run_case(&dev, &format!("L{layer}-prefill"), 24, 24, 0);
        for step in 1..4 {
            all_ok &= run_case(
                &dev,
                &format!("L{layer}-decode{step}"),
                1,
                24 + step,
                24 + step - 1,
            );
        }
    }
    assert!(all_ok, "sequential cadence corrupted results");
}
