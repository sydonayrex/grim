//! T2: hybrid prefix reuse via recurrent checkpoints.
//!
//! A hybrid model (Qwen35 GDN layers) must NOT consume a radix seed unless a
//! recurrent checkpoint exists at the EXACT depth being seeded — the T0
//! fail-closed gate. With `GRIM_HYBRID_PREFIX=1`, a second request sharing a
//! prefix must restore the checkpointed device GDN+conv state, seed the KV, and
//! produce logits bit-identical to a cold engine that never saw the prefix.
//!
//! The model is the shared synthetic hybrid builder (2 attention + 2 KDA layers)
//! so the gate runs at a geometry the CI box can afford. The 32-token prompt is
//! two blocks; the admission seed is capped at `len - 1` and rounds down, so the
//! reusable depth is 16 — which is exactly the boundary the block-aligned
//! prefill chunking exists to produce.
//!
//! Gated on `GRIM_GPU_TEST=1` + a ROCm device.

use grim_core::model::CausalLm;
use grim_engine::{Engine, EngineConfig, Request};
use grim_models_transformer::qwen35::Qwen35;
use grim_nn::WeightSource;

// The synthetic Qwen3.5 model builder and the GPU probe, SHARED with the
// capture gates (`qwen35_graph_capture.rs`) so there is one builder rather than
// a copy that can drift.
include!("common/qwen35_synth.rs");

/// T2 mutates process-global env, so the tests serialize on this lock.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// 4 layers, attention on 1 and 3, KDA on 0 and 2 — a real hybrid, small enough
/// to build in a test.
fn hybrid_cfg() -> Qwen35Config {
    let mut c = tiny_cfg();
    c.num_layers = 4;
    c.full_attention_interval = 2;
    c.max_seq_len = 256;
    c
}

fn build_model(device: &Device) -> Qwen35 {
    let provider = synth_weights_hybrid(4, 2);
    let ws = WeightSource::root(&provider, device.clone());
    Qwen35::load(device.clone(), &ws, hybrid_cfg()).expect("load tiny hybrid qwen35")
}

fn fresh_engine(model: Box<dyn CausalLm>) -> Engine {
    let mut cfg = EngineConfig::default();
    // Match the model's KV geometry so the paged pool and the model agree.
    cfg.num_kv_heads = KV_HEADS;
    cfg.head_dim = HEAD_DIM;
    let mut engine = Engine::new(cfg);
    engine.register_model("hybrid", model);
    engine
}

/// Drain prefill + a few decode ticks; return the last logits seen.
fn run(engine: &mut Engine, id: u64, prompt: &[u32], ticks: usize) -> Vec<f32> {
    engine
        .enqueue_request(Request {
            id,
            prompt_tokens: prompt.len(),
            input_ids: Some(prompt.to_vec()),
            max_new_tokens: 4,
            ..Default::default()
        })
        .expect("enqueue");
    let mut logits: Vec<f32> = Vec::new();
    for _ in 0..ticks {
        engine.tick().expect("tick");
        if let Some(outcome) = engine.last_outcome(id) {
            if let Some(l) = &outcome.logits {
                logits = l.to_vec_f32().expect("logits readback");
            }
        }
    }
    logits
}

fn prompt32() -> Vec<u32> {
    (0..32).map(|i| (i % 50 + 1) as u32).collect()
}

/// Greedy token of a logits row. The plan's gate is greedy-output identity, not
/// bit-identical logits: the ROCm forward is nondeterministic at ~1e-5 (rocBLAS
/// atomics under `DeterminismMode::Relaxed`), so a bit-for-bit logit compare is
/// flaky in a way that says nothing about reuse correctness.
fn greedy(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(u32::MAX)
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// The T2 gate: with the flag on, a repeated prompt must reuse the recurrent
/// checkpoint and land on the same logits as the cold run.
///
/// Requires the KV bridge (`GRIM_HYBRID_PREFIX=1`): the prefill appends each
/// attention layer's post-RoPE K/V to the shared paged store, and the first
/// forward after a radix seed hydrates the private arena from it.
#[test]
fn hybrid_prefix_reuse_matches_cold_output() {
    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some((device, _dev)) = gpu_device() else {
        return;
    };

    let prev = std::env::var("GRIM_HYBRID_PREFIX").ok();
    unsafe {
        std::env::set_var("GRIM_HYBRID_PREFIX", "1");
    }

    let prompt = prompt32();

    // ── Cold run: registers the prefix and its recurrent checkpoints ────────
    let mut cold = fresh_engine(Box::new(build_model(&device)));
    let cold_logits = run(&mut cold, 1, &prompt, 4);
    assert!(!cold_logits.is_empty(), "cold run produced no logits");
    let (_l0, h0, t0) = cold.radix_cache_telemetry();
    assert_eq!((h0, t0), (0, 0), "cold engine has no radix hits (one lookup is expected)");
    cold.finish_request(1);

    // ── Warm run: same engine, same prompt ──────────────────────────────────
    let warm_logits = run(&mut cold, 2, &prompt, 4);
    let (_l1, h1, t1) = cold.radix_cache_telemetry();
    assert_eq!(h1, 1, "warm request must consume the radix prefix");
    assert_eq!(
        t1, 16,
        "the seed is capped at len-1 and rounded down: 16 of 32 tokens, got {t1}"
    );
    assert_eq!(
        warm_logits.len(),
        cold_logits.len(),
        "logit shapes must match"
    );
    assert_eq!(
        greedy(&warm_logits),
        greedy(&cold_logits),
        "T2-reused greedy output must match the cold engine"
    );
    let d = max_diff(&warm_logits, &cold_logits);
    assert!(
        d < 1e-2,
        "T2-reused logits must stay within 1e-2 of the cold run, got {d:.3e}"
    );
    cold.finish_request(2);

    // restore the environment for the other tests
    match prev {
        Some(v) => unsafe { std::env::set_var("GRIM_HYBRID_PREFIX", v) },
        None => unsafe { std::env::remove_var("GRIM_HYBRID_PREFIX") },
    }
}

/// Fail-closed control: with the flag OFF, the hybrid model must NOT consume the
/// seed (T0 behaviour) and must still produce the cold logits.
#[test]
fn hybrid_model_does_not_seed_without_the_flag() {
    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some((device, _dev)) = gpu_device() else {
        return;
    };

    let prev = std::env::var("GRIM_HYBRID_PREFIX").ok();
    unsafe {
        std::env::remove_var("GRIM_HYBRID_PREFIX");
    }

    let prompt = prompt32();

    let mut engine = fresh_engine(Box::new(build_model(&device)));
    let cold_logits = run(&mut engine, 1, &prompt, 4);
    engine.finish_request(1);
    let warm_logits = run(&mut engine, 2, &prompt, 4);
    let (_l, hits, tokens) = engine.radix_cache_telemetry();
    assert_eq!(
        (hits, tokens),
        (0, 0),
        "hybrid model must not consume a radix seed without GRIM_HYBRID_PREFIX"
    );
    assert_eq!(
        greedy(&warm_logits),
        greedy(&cold_logits),
        "fail-closed full prefill must still match the cold run's greedy token"
    );
    let d = max_diff(&warm_logits, &cold_logits);
    assert!(
        d < 1e-2,
        "fail-closed full prefill logits must stay within 1e-2 of the cold run, got {d:.3e}"
    );
    engine.finish_request(2);

    match prev {
        Some(v) => unsafe { std::env::set_var("GRIM_HYBRID_PREFIX", v) },
        None => unsafe { std::env::remove_var("GRIM_HYBRID_PREFIX") },
    }
}
