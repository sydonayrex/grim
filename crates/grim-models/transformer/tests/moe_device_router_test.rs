//! TDD tests for Q2 (`GRIM_MOE_DEVICE_ROUTER=1`): device MoE router for Flash-Next.

use std::sync::Mutex;
use std::time::Instant;

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result as GrimResult;
use grim_tensor::{Device, Shape};

use grim_models_transformer::qwen4exp_moe_device::{
    moe_device_router_enabled, route_row_host, route_topk_flash_next_on_device,
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

#[test]
fn env_flag_defaults_off_and_parses_truthy() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    {
        let _e = EnvGuard::unset("GRIM_MOE_DEVICE_ROUTER");
        assert!(!moe_device_router_enabled());
    }
    for v in ["0", "false", ""] {
        let _e = EnvGuard::set("GRIM_MOE_DEVICE_ROUTER", v);
        assert!(!moe_device_router_enabled(), "value={v}");
    }
    for v in ["1", "true", "TRUE"] {
        let _e = EnvGuard::set("GRIM_MOE_DEVICE_ROUTER", v);
        assert!(moe_device_router_enabled(), "value={v}");
    }
}

#[ignore = "ROCm Card 1 device MoE router 512-expert tie/uniform parity & A/B"]
#[test]
fn bench_moe_device_router_gpu_card1() -> GrimResult<()> {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dev = Device::Rocm(0);
    let num_experts = 512usize;
    let top_k = 10usize;
    let scale = 1.5f32;

    // 1) Tie + uniform parity against route_row_host
    let mut row_ties = vec![0.0f32; num_experts];
    for e in 0..num_experts {
        row_ties[e] = ((e as f32) * 0.29).cos() * 2.0;
    }
    row_ties[31] = 4.0;
    row_ties[32] = 4.0;
    let row_uniform = vec![0.75f32; num_experts];

    for (label, row) in [("ties", &row_ties), ("uniform", &row_uniform)] {
        let (host_ids, host_w) = route_row_host(row, top_k, scale);
        let logits_cpu = cpu_tensor(row.clone(), Shape::new(vec![1, num_experts]));
        let logits_dev = grim_nn::modules::move_to_device(&logits_cpu, &dev)?;
        let (dev_ids, dev_w) =
            route_topk_flash_next_on_device(&logits_dev, num_experts, top_k, scale)?;
        assert_eq!(dev_ids, host_ids, "{label}: expert IDs mismatch");
        let max_diff = dev_w
            .iter()
            .zip(host_w.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[Q2-PARITY] {label}: IDs exact, max_w_diff={max_diff:.6e}");
        assert!(max_diff < 1e-4, "{label}: weight diff={max_diff}");
    }

    // 2) 3-run A/B microbench: 48 layers x 1-token router step (host D2H+sort vs device router)
    let logits_cpu = cpu_tensor(row_ties.clone(), Shape::new(vec![1, num_experts]));
    let logits_dev = grim_nn::modules::move_to_device(&logits_cpu, &dev)?;
    let median3 = |mut v: Vec<f64>| -> f64 {
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };

    let mut host_runs = Vec::with_capacity(3);
    for _ in 0..3 {
        let t0 = Instant::now();
        for _ in 0..48 {
            let v = logits_dev.to_vec_f32()?;
            let _ = route_row_host(&v, top_k, scale);
        }
        host_runs.push(t0.elapsed().as_secs_f64() * 1e3);
    }

    let mut dev_runs = Vec::with_capacity(3);
    for _ in 0..3 {
        let t0 = Instant::now();
        for _ in 0..48 {
            let _ = route_topk_flash_next_on_device(&logits_dev, num_experts, top_k, scale)?;
        }
        dev_runs.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    eprintln!(
        "[Q2-BENCH] 48-layer router host_runs={host_runs:?} ms (median={:.3} ms), dev_runs={dev_runs:?} ms (median={:.3} ms)",
        median3(host_runs.clone()),
        median3(dev_runs.clone())
    );
    Ok(())
}
