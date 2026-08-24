//! WI-SB3: end-to-end SCYTHE-2 TTFT/ITL A/B driver for syd-beasty.
//!
//! One invocation runs one arm (`--arm on|off`) over the fixed prompt mix and
//! appends §setup-4 JSON lines to the results file. Run both arms (and both
//! `GRIM_GPUS` ordinal orders, interleaved to cancel thermal drift); once the
//! file holds samples from both arms, every run prints the A/B table and the
//! WI-INF4 verdict (mean TTFT ≤ 5 %, p95 ITL ≤ 2 % ⇒ eligible to flip the
//! `GRIM_SCYTHE_INFERENCE` default).
use grim_backend_rocm::CapabilityProfiler;
use grim_engine::scythe_ab::{
    ScytheAbSample, default_results_path, format_ab_report, parse_samples,
};
use grim_engine::{Engine, EngineConfig, Request};
use std::{
    env, fs,
    path::PathBuf,
    process,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

fn arg(name: &str) -> Option<String> {
    let mut it = env::args().skip(1);
    while let Some(a) = it.next() {
        if a == name {
            return it.next();
        }
    }
    None
}

/// Which GPU is fast, per the live profiler — no hardcoded arch table here.
fn detect_order() -> String {
    let gpus: Vec<usize> = env::var("GRIM_GPUS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|t| t.trim().parse::<usize>().ok())
                .collect()
        })
        .unwrap_or_default();
    if gpus.len() < 2 {
        return "unknown".into();
    }
    let caps = CapabilityProfiler::new().capabilities();
    let Some(fast) = caps
        .iter()
        .max_by(|a, b| a.tflops_fp16.total_cmp(&b.tflops_fp16))
        .map(|c| c.ordinal)
    else {
        return "unknown".into();
    };
    // F-first = the faster card is listed first in GRIM_GPUS.
    if gpus.first() == Some(&fast) {
        "F-first".into()
    } else {
        "S-first".into()
    }
}

fn git_commit() -> String {
    process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = arg("--model").ok_or("--model is required")?;
    let arm = arg("--arm").unwrap_or_else(|| "off".into());
    // Default resolves against this example's crate dir so the harness runs
    // from any working directory.
    let prompts = arg("--prompts").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/prompts_scythe_ab.txt")
    });
    let iters: usize = arg("--iters").unwrap_or_else(|| "1".into()).parse()?;
    let out = arg("--out")
        .map(PathBuf::from)
        .unwrap_or_else(default_results_path);
    let armed = arm == "on";
    if armed {
        unsafe { env::set_var("GRIM_SCYTHE_INFERENCE", "1") };
    } else {
        unsafe { env::remove_var("GRIM_SCYTHE_INFERENCE") };
    }
    let order = detect_order();
    // One throwaway profiler for thermal telemetry along both legs.
    let telemetry = CapabilityProfiler::new();

    let mut engine = Engine::new(EngineConfig::default());
    engine.load_and_register_scythe_farm_speculative("scythe-ab", &model, None, false)?;
    let text = fs::read_to_string(prompts)?;
    let prompt_mix: Vec<&str> = text
        .split("\n\n")
        .filter(|p| !p.trim().is_empty())
        .collect();
    let commit = git_commit();

    let mut samples: Vec<ScytheAbSample> = Vec::new();
    let mut id = 1u64;
    for _ in 0..iters {
        for prompt in &prompt_mix {
            let n = prompt.split_whitespace().count();
            // Fresh per-sample trace: last_ttft_ms stays Some forever once a
            // prefill has run, so without this clear every later sample
            // breaks out of the loop instantly and records the stale value.
            engine.clear_latency_trace();
            engine.enqueue_request_with_kv(Request {
                id,
                prompt_tokens: n,
                max_new_tokens: 32,
                model_id: Some("scythe-ab".into()),
                ..Default::default()
            })?;
            let start = Instant::now();
            // Prefill leg: tick until this request's TTFT is observable.
            let mut ticks = 0usize;
            while engine.last_ttft_ms().is_none() && ticks < 512 {
                let _ = engine.tick()?;
                ticks += 1;
            }
            // Decode leg: keep draining until the request stops producing
            // decode steps (max_new_tokens reached), bounded defensively.
            let mut idle = 0usize;
            let mut decoded = 0usize;
            while idle < 3 && decoded <= 64 && ticks < 1024 {
                let out = engine.tick()?;
                let steps = out.decode_ids.len();
                decoded += steps;
                idle = if steps == 0 { idle + 1 } else { 0 };
                ticks += 1;
            }
            samples.push(ScytheAbSample {
                arm_on: armed,
                order: order.clone(),
                prompt_tokens: n,
                elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
                ttft_ms: engine.last_ttft_ms(),
                itl_ms: engine.last_itl_ms(),
                tokens_per_sec_ema: engine.tokens_per_sec(),
                throttle_pct: {
                    telemetry.tick();
                    telemetry
                        .capabilities()
                        .first()
                        .map(|c| c.throttle_pct)
                        .unwrap_or(0.0)
                },
            });
            engine.finish_request(id);
            id += 1;
        }
    }

    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let lines = grim_engine::scythe_ab::append_samples(&out, &samples, &commit, ts)?;
    println!(
        "wrote {lines} sample lines ({arm}, {order}) → {}",
        out.display()
    );
    for s in &samples {
        println!(
            "{{\"arm\":\"{}\",\"order\":\"{}\",\"prompt_tokens\":{},\"ttft_ms\":{},\"itl_ms\":{}}}",
            if s.arm_on { "on" } else { "off" },
            s.order,
            s.prompt_tokens,
            s.ttft_ms.map_or("?".into(), |v| format!("{v:.2}")),
            s.itl_ms.map_or("?".into(), |v| format!("{v:.2}")),
        );
    }

    // Verdict pass: needs both arms in the results file; a single leg prints
    // an INCOMPLETE report instead of inventing a comparison.
    let stored = parse_samples(&fs::read_to_string(&out)?);
    let on: Vec<_> = stored.iter().filter(|m| m.arm_on).cloned().collect();
    let off: Vec<_> = stored.iter().filter(|m| !m.arm_on).cloned().collect();
    print!("{}", format_ab_report(&on, &off));
    Ok(())
}
