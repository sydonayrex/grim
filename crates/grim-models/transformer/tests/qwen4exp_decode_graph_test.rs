//! TDD tests for Q8 (`GRIM_FLASH_DECODE_GRAPH=1`): ROCm HIP graph capture & replay
//! for Flash-Next GDN decode steps (string graph keys, zero DecodeGraphKey changes).

use std::sync::Mutex;
use std::time::Instant;

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result as GrimResult;
use grim_tensor::{Device, Shape};

use grim_models_transformer::qwen4exp_decode_graph::{
    capture_or_replay_gdn_step, flash_decode_graph_enabled, FlashDecodeGraphState,
};
use grim_models_transformer::qwen4exp_gdn_device::{
    gdn_decode_step_on_device, Qwen38GdnDeviceCache,
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
        let _e = EnvGuard::unset("GRIM_FLASH_DECODE_GRAPH");
        assert!(!flash_decode_graph_enabled());
    }
    for v in ["0", "false", ""] {
        let _e = EnvGuard::set("GRIM_FLASH_DECODE_GRAPH", v);
        assert!(!flash_decode_graph_enabled(), "value={v}");
    }
    for v in ["1", "true", "TRUE"] {
        let _e = EnvGuard::set("GRIM_FLASH_DECODE_GRAPH", v);
        assert!(flash_decode_graph_enabled(), "value={v}");
    }
}

#[ignore = "ROCm Card 1 Flash-Next decode graph 10-step parity & 36-layer A/B"]
#[test]
fn flash_decode_graph_gdn_10step_parity_and_36layer_ab() -> GrimResult<()> {
    let dev = Device::Rocm(0);
    let n_v_heads = 48usize;
    let n_k_heads = 16usize;
    let head_dim = 128usize;
    let key_dim = n_k_heads * head_dim;
    let val_dim = n_v_heads * head_dim;
    let conv_dim = 2 * key_dim + val_dim; // 10240
    let taps = 4usize;
    let eps = 1e-6f32;

    let conv_w_dev = grim_nn::modules::move_to_device(
        &cpu_tensor(lcg_signed(11, conv_dim * taps, 0.2), Shape::new(vec![conv_dim, taps])),
        &dev,
    )?;
    let dt_bias_dev = grim_nn::modules::move_to_device(
        &cpu_tensor(lcg_signed(22, n_v_heads, 0.3), Shape::new(vec![n_v_heads])),
        &dev,
    )?;
    let ssm_a_dev = grim_nn::modules::move_to_device(
        &cpu_tensor(
            lcg_signed(33, n_v_heads, 0.25)
                .into_iter()
                .map(|v| -v.abs() - 0.1)
                .collect(),
            Shape::new(vec![n_v_heads]),
        ),
        &dev,
    )?;
    let norm_w_dev = grim_nn::modules::move_to_device(
        &cpu_tensor(
            lcg_signed(44, head_dim, 0.5)
                .into_iter()
                .map(|v| 1.0 + v)
                .collect(),
            Shape::new(vec![head_dim]),
        ),
        &dev,
    )?;

    let mut eager_cache = Qwen38GdnDeviceCache::new(&dev, conv_dim, taps, n_v_heads, head_dim)?;
    let mut graph_cache = Qwen38GdnDeviceCache::new(&dev, conv_dim, taps, n_v_heads, head_dim)?;
    let mut graph_state = FlashDecodeGraphState::new(&dev, conv_dim, n_v_heads, val_dim)?;

    let mut max_out_diff = 0.0f32;
    let mut max_state_diff = 0.0f32;

    for step in 0..10usize {
        let qkv_dev = grim_nn::modules::move_to_device(
            &cpu_tensor(lcg_signed(100 + step as u64, conv_dim, 0.3), Shape::new(vec![1, conv_dim])),
            &dev,
        )?;
        let alpha_dev = grim_nn::modules::move_to_device(
            &cpu_tensor(lcg_signed(200 + step as u64, n_v_heads, 0.4), Shape::new(vec![1, n_v_heads])),
            &dev,
        )?;
        let beta_dev = grim_nn::modules::move_to_device(
            &cpu_tensor(lcg_signed(300 + step as u64, n_v_heads, 0.4), Shape::new(vec![1, n_v_heads])),
            &dev,
        )?;
        let gate_dev = grim_nn::modules::move_to_device(
            &cpu_tensor(lcg_signed(400 + step as u64, val_dim, 0.4), Shape::new(vec![1, val_dim])),
            &dev,
        )?;

        let eager_out = gdn_decode_step_on_device(
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
            &mut eager_cache,
        )?;
        let graph_out = capture_or_replay_gdn_step(
            0,
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
            &mut graph_cache,
            &mut graph_state,
        )?;
        grim_backend_rocm::RocmDevice::shared(0).synchronize();

        let ev = eager_out.to_vec_f32()?;
        let gv = graph_out.to_vec_f32()?;
        let es = eager_cache.ssm_state.to_vec_f32()?;
        let gs = graph_cache.ssm_state.to_vec_f32()?;

        let od = ev.iter().zip(gv.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let sd = es.iter().zip(gs.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        max_out_diff = max_out_diff.max(od);
        max_state_diff = max_state_diff.max(sd);
        assert!(od < 1e-5, "step {step} graph out diff={od:.6e}");
        assert!(sd < 1e-5, "step {step} graph state diff={sd:.6e}");
    }
    assert!(
        graph_state.is_layer_captured(0),
        "HIP graph must have captured cleanly (no fallback)"
    );
    eprintln!(
        "[Q8-PARITY] 10-step captured graph vs eager GDN: max_out_diff={max_out_diff:.6e}, max_state_diff={max_state_diff:.6e}"
    );

    // 3-run A/B of 36 GDN layers: eager vs captured HIP graph replay
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

    let mut eager_runs = Vec::with_capacity(3);
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
                &mut eager_cache,
            )?;
        }
        grim_backend_rocm::RocmDevice::shared(0).synchronize();
        eager_runs.push(t0.elapsed().as_secs_f64() * 1e3);
    }

    let mut graph_runs = Vec::with_capacity(3);
    for _ in 0..3 {
        let t0 = Instant::now();
        for _ in 0..36 {
            let _ = capture_or_replay_gdn_step(
                0,
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
                &mut graph_cache,
                &mut graph_state,
            )?;
        }
        grim_backend_rocm::RocmDevice::shared(0).synchronize();
        graph_runs.push(t0.elapsed().as_secs_f64() * 1e3);
    }

    eprintln!(
        "[Q8-BENCH] 36-layer GDN eager_runs={eager_runs:?} ms (median={:.3} ms), graph_runs={graph_runs:?} ms (median={:.3} ms)",
        median3(eager_runs.clone()),
        median3(graph_runs.clone())
    );
    Ok(())
}
