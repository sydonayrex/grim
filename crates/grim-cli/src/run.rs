//! `grim run` — load a model, run a prompt, or start HTTP server.


use grim_core::error::Result;
use grim_core::CausalLm;
use grim_core::session::Inner as SessionInner;
use grim_engine::EngineConfig;
use grim_models_transformer::{Llama, LlamaConfig};

pub async fn cmd_run(model_path: String, prompt: Option<String>, serve: bool, address: String) -> Result<()> {
    let _model_path = &model_path;

    if serve {
        // Build a minimal engine + server.
        let engine = grim_engine::Engine::new(EngineConfig::default());
        println!("Starting HTTP server on {address}...");
        grim_server::serve(&address, engine).await?;
        return Ok(());
    }

    // One-shot: load a tiny random model and run the prompt.
    let cfg = LlamaConfig {
        vocab_size: 512,
        hidden_size: 64,
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 32,
        num_layers: 1,
        intermediate_size: 128,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_seq_len: 64,
    };
    let model = Llama::random(cfg);
    let prompt = prompt.unwrap_or_else(|| "Hello".to_string());
    println!("Prompt: {prompt}");

    // Convert prompt bytes to token ids (v1: use byte values as token ids).
    let tokens: Vec<u32> = prompt.bytes().map(|b| b as u32 % 512).collect();
    let input_tensor = grim_backend_cpu::cpu_tensor(
        tokens.iter().map(|t| *t as f32).collect(),
        grim_tensor::Shape::new(vec![tokens.len()]),
    );

    let mut session = SessionInner::new(model.device.clone());
    let logits = CausalLm::forward(&model, &mut session, &input_tensor, &input_tensor)?;
    let logits_vec = logits.to_vec_f32()?;

    // Get the argmax of the last token
    let vocab = 512usize;
    let last_start = logits_vec.len() - vocab;
    let last_token = logits_vec[last_start..]
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    println!("Next token id: {last_token}");
    Ok(())
}