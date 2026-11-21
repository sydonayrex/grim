//! TDD tests for Q6 (`GRIM_QWEN4_DEVICE_ARGMAX=1`): device argmax over full 248320 vocab.

use std::sync::Mutex;
use std::time::Instant;

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result as GrimResult;
use grim_tensor::{Device, Shape};

use grim_models_transformer::qwen4exp_flash_next::{
    argmax_last_row_on_device, qwen4_device_argmax_enabled,
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

fn cpu_argmax_highest_tie(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

#[test]
fn env_flag_defaults_off_and_parses_truthy() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    {
        let _e = EnvGuard::unset("GRIM_QWEN4_DEVICE_ARGMAX");
        assert!(!qwen4_device_argmax_enabled());
    }
    for v in ["0", "false", ""] {
        let _e = EnvGuard::set("GRIM_QWEN4_DEVICE_ARGMAX", v);
        assert!(!qwen4_device_argmax_enabled(), "value={v}");
    }
    for v in ["1", "true", "TRUE"] {
        let _e = EnvGuard::set("GRIM_QWEN4_DEVICE_ARGMAX", v);
        assert!(qwen4_device_argmax_enabled(), "value={v}");
    }
}

#[ignore = "ROCm Card 1 full 248320-vocab device argmax parity (ties, uniform, last-idx) & A/B"]
#[test]
fn device_argmax_full_248320_vocab_parity_and_ab() -> GrimResult<()> {
    const VOCAB: usize = 248320;
    let dev = Device::Rocm(0);

    // 1) Spread logits
    let mut spread = vec![0.0f32; VOCAB];
    let mut s = 777u64;
    for v in &mut spread {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *v = ((s >> 33) as f32) / (u32::MAX as f32 * 0.5) - 1.0;
    }
    spread[123456] = 9.5;

    // 2) Ties (grim_argmax_stage1 uses >= / highest-index tie-break, matching Iterator::max_by)
    let mut ties = spread.clone();
    ties[500] = 12.0;
    ties[200000] = 12.0;

    // 3) All-equal uniform row
    let uniform = vec![0.5f32; VOCAB];

    // 4) Last-index max
    let mut last_max = vec![-1.0f32; VOCAB];
    last_max[VOCAB - 1] = 4.25;

    for (label, row) in [
        ("spread", &spread),
        ("ties_highest_wins", &ties),
        ("uniform", &uniform),
        ("last_index_max", &last_max),
    ] {
        let expected = cpu_argmax_highest_tie(row);
        let t_dev = grim_nn::modules::move_to_device(
            &cpu_tensor(row.clone(), Shape::new(vec![1, VOCAB])),
            &dev,
        )?;
        let got = argmax_last_row_on_device(&t_dev)?;
        eprintln!("[Q6-PARITY] {label}: expected={expected} got={got}");
        assert_eq!(got, expected, "{label}: device argmax mismatch");
    }

    // 3-run A/B over 100 steps: host 993 KB D2H + CPU argmax vs device 4-byte D2H argmax
    let t_dev = grim_nn::modules::move_to_device(
        &cpu_tensor(spread.clone(), Shape::new(vec![1, VOCAB])),
        &dev,
    )?;
    let median3 = |mut v: Vec<f64>| -> f64 {
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };

    let mut host_runs = Vec::with_capacity(3);
    for _ in 0..3 {
        let t0 = Instant::now();
        for _ in 0..100 {
            let v = t_dev.to_vec_f32()?;
            let _ = cpu_argmax_highest_tie(&v);
        }
        host_runs.push(t0.elapsed().as_secs_f64() * 1e3);
    }

    let mut dev_runs = Vec::with_capacity(3);
    for _ in 0..3 {
        let t0 = Instant::now();
        for _ in 0..100 {
            let _ = argmax_last_row_on_device(&t_dev)?;
        }
        dev_runs.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    eprintln!(
        "[Q6-BENCH] 100x 248320-vocab argmax host_runs={host_runs:?} ms (median={:.3} ms), dev_runs={dev_runs:?} ms (median={:.3} ms)",
        median3(host_runs.clone()),
        median3(dev_runs.clone())
    );
    Ok(())
}
