//! Q4 integration test & A/B benchmark: `qsa_device_gather_attention` vs host oracle
//! (`masked_gqa_attention`) at 1k and 19k context on ROCm Card 1.

use std::time::Instant;

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result as GrimResult;
use grim_tensor::{Device, Shape};

use grim_models_transformer::qwen4exp_flash_next::qsa_device_gather_attention;
use grim_models_transformer::qwen4exp_qsa::{build_top_k_mask, masked_gqa_attention};

fn lcg_signed(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((s >> 33) as f32) / (u32::MAX as f32 * 0.5) - 1.0;
        out.push(u * scale);
    }
    out
}

#[ignore = "ROCm Card 1 QSA device gather parity (1k & 19k) + 12-layer decode A/B"]
#[test]
fn qsa_device_gather_matches_host_oracle_at_1k_and_19k() -> GrimResult<()> {
    let dev = Device::Rocm(0);
    // Exact Flash-Next full-attention geometry
    let num_heads = 24usize;
    let num_kv_heads = 2usize;
    let head_dim = 256usize;
    let q_stride = num_heads * head_dim; // 6144
    let kv_stride = num_kv_heads * head_dim; // 512

    let median3 = |mut v: Vec<f64>| -> f64 {
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };

    for (label, kv_len, n_ids) in [("1k", 1024usize, 256usize), ("19k", 19456usize, 2048usize)] {
        let q_vec = lcg_signed(101, q_stride, 0.25);
        let k_vec = lcg_signed(202, kv_len * kv_stride, 0.25);
        let v_vec = lcg_signed(303, kv_len * kv_stride, 0.25);

        // Deterministic top-k selection spread across [0, kv_len)
        let step = (kv_len / n_ids).max(1);
        let sel: Vec<usize> = (0..n_ids).map(|i| (i * step) % kv_len).collect();
        assert_eq!(sel.len(), n_ids);

        let mask = build_top_k_mask(kv_len, &sel, None)?;
        let host_out = masked_gqa_attention(
            &q_vec,
            &k_vec,
            &v_vec,
            num_heads,
            num_kv_heads,
            head_dim,
            1,
            kv_len,
            kv_len - 1,
            &mask,
        )?;

        let q_dev = grim_nn::modules::move_to_device(
            &cpu_tensor(q_vec, Shape::new(vec![1, q_stride])),
            &dev,
        )?;
        let k_dev = grim_nn::modules::move_to_device(
            &cpu_tensor(k_vec, Shape::new(vec![kv_len, kv_stride])),
            &dev,
        )?;
        let v_dev = grim_nn::modules::move_to_device(
            &cpu_tensor(v_vec, Shape::new(vec![kv_len, kv_stride])),
            &dev,
        )?;

        let dev_out_t = qsa_device_gather_attention(
            &q_dev,
            &k_dev,
            &v_dev,
            &sel,
            num_heads,
            num_kv_heads,
            head_dim,
        )?;
        grim_backend_rocm::RocmDevice::shared(0).synchronize();
        let dev_out = dev_out_t.to_vec_f32()?;

        let max_diff = dev_out
            .iter()
            .zip(host_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[Q4-PARITY] ctx={label} (kv_len={kv_len}, n_ids={n_ids}): max_abs_diff={max_diff:.6e}");
        assert!(max_diff <= 1e-4, "ctx={label}: max_abs_diff={max_diff} > 1e-4");

        // 3-run A/B for 12 QSA layers (Flash-Next has 12 full-attention layers)
        let mut dev_runs = Vec::with_capacity(3);
        for _ in 0..3 {
            let t0 = Instant::now();
            for _ in 0..12 {
                let _ = qsa_device_gather_attention(
                    &q_dev,
                    &k_dev,
                    &v_dev,
                    &sel,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                )?;
            }
            grim_backend_rocm::RocmDevice::shared(0).synchronize();
            dev_runs.push(t0.elapsed().as_secs_f64() * 1e3);
        }

        let mut host_runs = Vec::with_capacity(3);
        for _ in 0..3 {
            let t0 = Instant::now();
            for _ in 0..12 {
                let qh = q_dev.to_vec_f32()?;
                let kh = k_dev.to_vec_f32()?;
                let vh = v_dev.to_vec_f32()?;
                let ho = masked_gqa_attention(
                    &qh,
                    &kh,
                    &vh,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    1,
                    kv_len,
                    kv_len - 1,
                    &mask,
                )?;
                let _ = grim_nn::modules::move_to_device(
                    &cpu_tensor(ho, Shape::new(vec![1, q_stride])),
                    &dev,
                )?;
            }
            grim_backend_rocm::RocmDevice::shared(0).synchronize();
            host_runs.push(t0.elapsed().as_secs_f64() * 1e3);
        }

        eprintln!(
            "[Q4-BENCH] ctx={label} 12-layer QSA host_runs={host_runs:?} ms (median={:.3} ms), dev_runs={dev_runs:?} ms (median={:.3} ms)",
            median3(host_runs.clone()),
            median3(dev_runs.clone())
        );
    }
    Ok(())
}
