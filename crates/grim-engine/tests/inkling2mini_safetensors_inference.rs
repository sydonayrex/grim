//! Real-checkpoint smoke test for the `inkling2_loop` text tower
//! (`models/Inkling2mini/` — TML Inkling-2-Mini, NVFP4 MoE).
//!
//! Contract: load `model-00001-of-00001.safetensors` via the engine safetensors
//! loader (host f32 reference path), tokenize from the HF `tokenizer.json`, run a
//! short-prompt greedy generation, and require finite, EOS-or-length-terminated
//! output. This is a smoke gate, NOT an HF-parity gate.
//!
//! Run: `GRIM_INKLING2_TEST=1 cargo test -p grim-engine --release --test inkling2mini_safetensors_inference`

use grim_engine::model_loader::load_model_from_safetensors;
use grim_format::tokenizer::GgufTokenizer;
use grim_tensor::{Device, Shape, Tensor};
use std::path::Path;

#[test]
fn inkling2mini_safetensors_load_and_generate() {
    if std::env::var("GRIM_INKLING2_TEST").is_err() {
        eprintln!("[test-skip] set GRIM_INKLING2_TEST=1 to run (loads ~20 GB checkpoint)");
        return;
    }
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
    let workspace_root = Path::new(&manifest_dir).parent().unwrap().parent().unwrap();
    let weights = workspace_root.join("models/Inkling2mini/model-00001-of-00001.safetensors");
    let tokenizer_json = workspace_root.join("models/Inkling2mini/tokenizer.json");
    assert!(weights.exists(), "missing {weights:?}");
    assert!(tokenizer_json.exists(), "missing {tokenizer_json:?}");

    eprintln!("[test-step] loading inkling2_loop weights (host f32 reference path)...");
    let model = load_model_from_safetensors(weights.to_str().unwrap(), Device::Cpu)
        .expect("load inkling2_loop from safetensors");

    eprintln!("[test-step] tokenizer from HF tokenizer.json...");
    let tokenizer = GgufTokenizer::from_hf_json(tokenizer_json.to_str().unwrap())
        .expect("HF tokenizer.json");
    let eos = tokenizer.eos_token_id;

    let prompt = "The capital of France is";
    let ids = tokenizer.encode(prompt);
    eprintln!("[test-step] prompt tokens: {ids:?}");
    assert!(!ids.is_empty(), "empty tokenization");

    let mut tokens = ids.clone();
    let mut out_tokens = Vec::new();
    let max_steps: usize = std::env::var("GRIM_INKLING2_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    for step in 0..max_steps {
        let n = tokens.len();
        let (input, pos) = seq_tensors(&tokens);
        let mut session = model.new_session();
        let logits = model
            .forward(session.as_mut(), &input, &pos, &[])
            .expect("forward");
        let row = logits.to_vec_f32().expect("logits f32");
        let vocab = row.len() / n;
        assert!(row.iter().all(|v| v.is_finite()), "non-finite logits at step {step}");
        let last = &row[(n - 1) * vocab..];
        let next = argmax(last);
        out_tokens.push(next);
        tokens.push(next);
        eprintln!("[decode {step}] next={next}");
        if Some(next) == eos {
            break;
        }
    }
    let text = tokenizer.decode(&out_tokens);
    eprintln!("[test-step] generated: {text:?}");
    assert!(!out_tokens.is_empty());
}

fn seq_tensors(tokens: &[u32]) -> (Tensor, Tensor) {
    let n = tokens.len();
    let ids_f32: Vec<f32> = tokens.iter().map(|&t| t as f32).collect();
    let pos_f32: Vec<f32> = (0..n as u32).map(|p| p as f32).collect();
    let input = grim_backend_cpu::cpu_tensor(ids_f32, Shape::new(vec![n]));
    let pos = grim_backend_cpu::cpu_tensor(pos_f32, Shape::new(vec![n]));
    (input, pos)
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0 as u32
}
