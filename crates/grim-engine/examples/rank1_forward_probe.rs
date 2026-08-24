//! Root-cause probe (validation log 2026-08-23e): does a model loaded onto
//! ordinal 1 — exactly as farm replicas are (`load_from_path_on_device`) —
//! produce sane logits when BOTH GPUs are visible?
//!
//! Modes (REPRO_MODE env): "replica1" (farm-style load on Rocm(1)),
//! "control0" (plain load on Rocm(0)), "farm" (full farm registration,
//! request pinned via GRIM_SCYTHE_SPREAD).
use grim_engine::{Engine, EngineConfig, Request};

fn logits_once(engine: &mut Engine, id: u64, model_id: &str) -> (usize, f32, Vec<f32>) {
    engine.clear_latency_trace();
    engine
        .enqueue_request_with_kv(Request {
            id,
            prompt_tokens: 8,
            max_new_tokens: 1,
            model_id: Some(model_id.to_string()),
            ..Default::default()
        })
        .expect("enqueue");
    let _ = engine.tick();
    let outcome = engine.last_outcome(id).expect("outcome");
    let logits = outcome.logits.as_ref().expect("logits present");
    let v = logits.to_vec_f32().expect("logits readback");
    // Last-vocab tail is what sampling reads; report whole-vector stats plus
    // a head sample.
    let nonzero = v.iter().filter(|&&x| x != 0.0).count();
    let sum: f32 = v.iter().take(1024).sum();
    (nonzero, sum, v[..8.min(v.len())].to_vec())
}

fn main() {
    let mode = std::env::var("REPRO_MODE").unwrap_or_else(|_| "replica1".into());
    let model_path = "models/LFM2.5-230M-Q4_K_M.gguf";
    println!("== rank1_forward_probe mode={mode}");

    match mode.as_str() {
        "farm" => {
            unsafe {
                std::env::set_var("GRIM_SCYTHE_INFERENCE", "1");
                std::env::set_var("GRIM_SCYTHE_SPREAD", "1");
            }
            let mut engine = Engine::new(EngineConfig::default());
            engine
                .load_and_register_scythe_farm_speculative("probe", model_path, None, false)
                .expect("farm load");
            let (nonzero, sum, head) = logits_once(&mut engine, 1, "probe");
            println!("farm/pinned: nonzero={nonzero} sum1k={sum} head={head:?}");
        }
        other => {
            let dev = if other == "control0" {
                grim_tensor::Device::Rocm(0)
            } else {
                grim_tensor::Device::Rocm(1)
            };
            let mut engine = Engine::new(EngineConfig::default());
            let model = grim_engine::model_loader::load_from_path_on_device(model_path, dev)
                .expect("replica load");
            engine.register_model("probe", model);
            let (nonzero, sum, head) = logits_once(&mut engine, 1, "probe");
            println!("{other}: nonzero={nonzero} sum1k={sum} head={head:?}");
        }
    }
}
