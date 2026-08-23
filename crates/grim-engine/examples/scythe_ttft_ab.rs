//! Small end-to-end SCYTHE-2 A/B driver. Hardware runs append JSONL samples.
use grim_engine::{Engine, EngineConfig, Request};
use std::{env, fs, path::PathBuf, time::Instant};

fn arg(name: &str) -> Option<String> {
    let mut it = env::args().skip(1);
    while let Some(a) = it.next() {
        if a == name {
            return it.next();
        }
    }
    None
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = arg("--model").ok_or("--model is required")?;
    let arm = arg("--arm").unwrap_or_else(|| "off".into());
    let prompts =
        PathBuf::from(arg("--prompts").unwrap_or_else(|| "examples/prompts_scythe_ab.txt".into()));
    let iters: usize = arg("--iters").unwrap_or_else(|| "1".into()).parse()?;
    let armed = arm == "on";
    if armed {
        unsafe { env::set_var("GRIM_SCYTHE_INFERENCE", "1") };
    } else {
        unsafe { env::remove_var("GRIM_SCYTHE_INFERENCE") };
    }
    let mut engine = Engine::new(EngineConfig::default());
    engine.load_and_register_scythe_farm_speculative("scythe-ab", &model, None, false)?;
    let text = fs::read_to_string(prompts)?;
    let mut id = 1u64;
    for _ in 0..iters {
        for prompt in text.split("\n\n").filter(|p| !p.trim().is_empty()) {
            let n = prompt.split_whitespace().count();
            engine.enqueue_request_with_kv(Request {
                id,
                prompt_tokens: n,
                max_new_tokens: 32,
                model_id: Some("scythe-ab".into()),
                ..Default::default()
            })?;
            let start = Instant::now();
            for _ in 0..64 {
                let _ = engine.tick()?;
                if engine.last_ttft_ms().is_some() {
                    break;
                }
            }
            let sample = serde_json::json!({"arm": armed, "request_id": id, "prompt_tokens": n, "elapsed_ms": start.elapsed().as_secs_f64()*1000.0, "ttft_ms": engine.last_ttft_ms(), "itl_ms": engine.last_itl_ms(), "tokens_per_sec_ema": engine.tokens_per_sec()});
            println!("{}", sample);
            engine.finish_request(id);
            id += 1;
        }
    }
    Ok(())
}
