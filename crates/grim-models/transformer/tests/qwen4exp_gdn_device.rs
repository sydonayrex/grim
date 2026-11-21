//! TDD tests for Q3 (`GRIM_GDN_DEVICE=1`): device GDN decode step vs host oracle.

use std::sync::Mutex;
use std::time::Instant;

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result as GrimResult;
use grim_tensor::{Device, Shape};

use grim_models_transformer::qwen4exp_gdn::{
    gated_delta_net_forward, GdnParams, KdaHeadPairing, Qwen38GdnCache,
};
use grim_models_transformer::qwen4exp_gdn_device::{
    gdn_decode_step_on_device, gdn_device_enabled, Qwen38GdnDeviceCache,
};

static ENV_LOCK: Mutex<()> = Mutex::new(());

struct EnvGuard {
    key: &'static str,
    prev: Option<String>,
}

impl EnvGuard {
    fn set(key: &'static str, val: &str) -> Self {
        let prev = std::env::var(key).ok();
        unsafe { std::env::set_var(key, val) };
        Self { key, prev }
    }
    fn unset(key: &'static str) -> Self {
        let prev = std::env::var(key).ok();
        unsafe { std::env::remove_var(key) };
        Self { key, prev }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        unsafe {
            match &self.prev {
                Some(v) => std::env::set_var(self.key, v),
                None => std::env::remove_var(self.key),
            }
        }
    }
}

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

#[test]
fn env_flag_defaults_off_and_parses_truthy() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    {
        let _e = EnvGuard::unset("GRIM_GDN_DEVICE");
        assert!(!gdn_device_enabled());
    }
    for v in ["0", "false", ""] {
        let _e = EnvGuard::set("GRIM_GDN_DEVICE", v);
        assert!(!gdn_device_enabled(), "value={v}");
    }
    for v in ["1", "true", "TRUE"] {
        let _e = EnvGuard::set("GRIM_GDN_DEVICE", v);
        assert!(gdn_device_enabled(), "value={v}");
    }
}

#[ignore = "ROCm Card 1 GDN device step oracle parity (1-step & 10-step drift) + 36-layer A/B"]
#[test]
fn gdn_device_step_matches_host_oracle_1step_and_10step_drift() -> GrimResult<()> {
    let dev = Device::Rocm(0);
    let n_v_heads = 48usize;
    let n_k_heads = 16usize;
    let head_dim = 128usize;
    let key_dim = n_k_heads * head_dim;
    let val_dim = n_v_heads * head_dim;
    let conv_dim = 2 * key_dim + val_dim; // 10240
    let taps = 4usize;
    let eps = 1e-6f32;

    let conv_w_vec = lcg_signed(11, conv_dim * taps, 0.2);
    let dt_bias_vec = lcg_signed(22, n_v_heads, 0.3);
    let ssm_a_vec: Vec<f32> = lcg_signed(33, n_v_heads, 0.25)
        .into_iter()
        .map(|v| -v.abs() - 0.1)
        .collect();
    let norm_w_vec = lcg_signed(44, head_dim, 0.5)
        .into_iter()
        .map(|v| 1.0 + v)
        .collect::<Vec<_>>();

    let conv_w_cpu = cpu_tensor(conv_w_vec.clone(), Shape::new(vec![conv_dim, taps]));
    let conv_w_dev = grim_nn::modules::move_to_device(&conv_w_cpu, &dev)?;
    let dt_bias_dev = grim_nn::modules::move_to_device(
        &cpu_tensor(dt_bias_vec.clone(), Shape::new(vec![n_v_heads])),
        &dev,
    )?;
    let ssm_a_dev = grim_nn::modules::move_to_device(
        &cpu_tensor(ssm_a_vec.clone(), Shape::new(vec![n_v_heads])),
        &dev,
    )?;
    let norm_w_dev = grim_nn::modules::move_to_device(
        &cpu_tensor(norm_w_vec.clone(), Shape::new(vec![head_dim])),
        &dev,
    )?;

    let mut host_cache = Qwen38GdnCache::new(n_v_heads, head_dim, head_dim, taps, conv_dim);
    let mut dev_cache = Qwen38GdnDeviceCache::new(&dev, conv_dim, taps, n_v_heads, head_dim)?;

    let mut max_step_out_diff = 0.0f32;
    let mut max_step_state_diff = 0.0f32;

    for step in 0..10usize {
        let qkv_vec = lcg_signed(100 + step as u64, conv_dim, 0.3);
        let alpha_vec = lcg_signed(200 + step as u64, n_v_heads, 0.4);
        let beta_vec = lcg_signed(300 + step as u64, n_v_heads, 0.4);
        let gate_vec = lcg_signed(400 + step as u64, val_dim, 0.4);

        // Host oracle (exact qwen4exp_flash_next.rs GDN path)
        let need = (taps - 1) * conv_dim;
        let mut state_t = cpu_tensor(
            host_cache.conv_state[..need].to_vec(),
            Shape::new(vec![1, taps - 1, conv_dim]),
        );
        let qkv_b1 = cpu_tensor(qkv_vec.clone(), Shape::new(vec![1, 1, conv_dim]));
        let conv_out =
            grim_nn::modules::short_conv1d(&qkv_b1, &conv_w_cpu, None, Some(&mut state_t))?
                .to_vec_f32()?;
        let hist = state_t.to_vec_f32()?;
        host_cache.conv_state[..need].copy_from_slice(&hist[..need]);
        let conv_mix: Vec<f32> = conv_out.iter().map(|v| v / (1.0 + (-v).exp())).collect();

        let mut gdn_out = vec![0.0f32; val_dim];
        gated_delta_net_forward(
            &GdnParams {
                conv_mix: &conv_mix,
                alpha: &alpha_vec,
                beta: &beta_vec,
                a: &ssm_a_vec,
                dt_bias: &dt_bias_vec,
                norm: &norm_w_vec,
                eps,
                n_v_heads,
                n_k_heads,
                head_dim,
                conv_dim,
                seq_len: 1,
                pairing: KdaHeadPairing::Interleaved,
            },
            &mut host_cache,
            &mut gdn_out,
        )?;
        let host_gated: Vec<f32> = gdn_out
            .iter()
            .zip(gate_vec.iter())
            .map(|(g, s)| g / (1.0 + (-s).exp()))
            .collect();

        // Device step
        let qkv_dev = grim_nn::modules::move_to_device(
            &cpu_tensor(qkv_vec, Shape::new(vec![1, conv_dim])),
            &dev,
        )?;
        let alpha_dev = grim_nn::modules::move_to_device(
            &cpu_tensor(alpha_vec, Shape::new(vec![1, n_v_heads])),
            &dev,
        )?;
        let beta_dev = grim_nn::modules::move_to_device(
            &cpu_tensor(beta_vec, Shape::new(vec![1, n_v_heads])),
            &dev,
        )?;
        let gate_dev = grim_nn::modules::move_to_device(
            &cpu_tensor(gate_vec, Shape::new(vec![1, val_dim])),
            &dev,
        )?;

        let dev_out_t = gdn_decode_step_on_device(
            &qkv_dev,
            &conv_w_dev,
            &alpha_dev,
            &beta_dev,
            &dt_bias_dev,
            &ssm_a_dev,
            &norm_w_dev,
            &gate_dev,
            eps,
            n_v_heads,
            n_k_heads,
            head_dim,
            &mut dev_cache,
        )?;
        grim_backend_rocm::RocmDevice::shared(0).synchronize();
        let dev_gated = dev_out_t.to_vec_f32()?;
        let dev_ssm = dev_cache.ssm_state.to_vec_f32()?;
        let dev_conv = dev_cache.conv_state.to_vec_f32()?;

        let out_diff = dev_gated
            .iter()
            .zip(host_gated.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let state_diff = dev_ssm
            .iter()
            .zip(host_cache.ssm_state.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let conv_diff = dev_conv
            .iter()
            .zip(host_cache.conv_state.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        max_step_out_diff = max_step_out_diff.max(out_diff);
        max_step_state_diff = max_step_state_diff.max(state_diff);
        assert!(
            out_diff < 1e-5,
            "step {step} output diff={out_diff:.6e} >= 1e-5"
        );
        assert!(
            state_diff < 1e-5,
            "step {step} ssm_state diff={state_diff:.6e} >= 1e-5"
        );
        assert!(
            conv_diff < 1e-5,
            "step {step} conv_state diff={conv_diff:.6e} >= 1e-5"
        );
    }
    eprintln!(
        "[Q3-PARITY] 10-step Flash-Next GDN (48x128): max_out_diff={max_step_out_diff:.6e}, max_state_diff={max_step_state_diff:.6e}"
    );

    // 3-run A/B timing of 36 GDN layers (Flash-Next decode has 36 GDN layers)
    let qkv_dev = grim_nn::modules::move_to_device(
        &cpu_tensor(lcg_signed(901, conv_dim, 0.2), Shape::new(vec![1, conv_dim])),
        &dev,
    )?;
    let alpha_dev = grim_nn::modules::move_to_device(
        &cpu_tensor(lcg_signed(902, n_v_heads, 0.2), Shape::new(vec![1, n_v_heads])),
        &dev,
    )?;
    let beta_dev = grim_nn::modules::move_to_device(
        &cpu_tensor(lcg_signed(903, n_v_heads, 0.2), Shape::new(vec![1, n_v_heads])),
        &dev,
    )?;
    let gate_dev = grim_nn::modules::move_to_device(
        &cpu_tensor(lcg_signed(904, val_dim, 0.2), Shape::new(vec![1, val_dim])),
        &dev,
    )?;

    let median3 = |mut v: Vec<f64>| -> f64 {
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };

    let mut dev_runs = Vec::with_capacity(3);
    for _ in 0..3 {
        let t0 = Instant::now();
        for _ in 0..36 {
            let _ = gdn_decode_step_on_device(
                &qkv_dev,
                &conv_w_dev,
                &alpha_dev,
                &beta_dev,
                &dt_bias_dev,
                &ssm_a_dev,
                &norm_w_dev,
                &gate_dev,
                eps,
                n_v_heads,
                n_k_heads,
                head_dim,
                &mut dev_cache,
            )?;
        }
        grim_backend_rocm::RocmDevice::shared(0).synchronize();
        dev_runs.push(t0.elapsed().as_secs_f64() * 1e3);
    }

    let mut host_runs = Vec::with_capacity(3);
    for _ in 0..3 {
        let t0 = Instant::now();
        for _ in 0..36 {
            let qkv_h = qkv_dev.to_vec_f32()?;
            let need = (taps - 1) * conv_dim;
            let mut state_t = cpu_tensor(
                host_cache.conv_state[..need].to_vec(),
                Shape::new(vec![1, taps - 1, conv_dim]),
            );
            let qkv_b1 = cpu_tensor(qkv_h, Shape::new(vec![1, 1, conv_dim]));
            let conv_out =
                grim_nn::modules::short_conv1d(&qkv_b1, &conv_w_cpu, None, Some(&mut state_t))?
                    .to_vec_f32()?;
            let hist = state_t.to_vec_f32()?;
            host_cache.conv_state[..need].copy_from_slice(&hist[..need]);
            let conv_mix: Vec<f32> = conv_out.iter().map(|v| v / (1.0 + (-v).exp())).collect();
            let alpha_h = alpha_dev.to_vec_f32()?;
            let beta_h = beta_dev.to_vec_f32()?;
            let mut gdn_out = vec![0.0f32; val_dim];
            gated_delta_net_forward(
                &GdnParams {
                    conv_mix: &conv_mix,
                    alpha: &alpha_h,
                    beta: &beta_h,
                    a: &ssm_a_vec,
                    dt_bias: &dt_bias_vec,
                    norm: &norm_w_vec,
                    eps,
                    n_v_heads,
                    n_k_heads,
                    head_dim,
                    conv_dim,
                    seq_len: 1,
                    pairing: KdaHeadPairing::Interleaved,
                },
                &mut host_cache,
                &mut gdn_out,
            )?;
            let gate_h = gate_dev.to_vec_f32()?;
            let gated: Vec<f32> = gdn_out
                .iter()
                .zip(gate_h.iter())
                .map(|(g, s)| g / (1.0 + (-s).exp()))
                .collect();
            let _ = grim_nn::modules::move_to_device(
                &cpu_tensor(gated, Shape::new(vec![1, val_dim])),
                &dev,
            )?;
        }
        grim_backend_rocm::RocmDevice::shared(0).synchronize();
        host_runs.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    eprintln!(
        "[Q3-BENCH] 36-layer GDN host_runs={host_runs:?} ms (median={:.3} ms), dev_runs={dev_runs:?} ms (median={:.3} ms)",
        median3(host_runs.clone()),
        median3(dev_runs.clone())
    );
    Ok(())
}
