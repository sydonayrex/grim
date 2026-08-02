//! `grim bench` — benchmark / smoke test.

use grim_core::error::Result;
use grim_core::model::CausalLm;
use grim_tensor::Device;

pub async fn cmd_bench(tokens: usize, concurrency: usize, model_path: Option<&str>) -> Result<()> {
    let device = Device::Cpu;
    // Use provided model path or fall back to a random Llama for smoke testing.
    let model: Box<dyn CausalLm> = if let Some(path) = model_path {
        let lower = path.to_lowercase();
        if lower.ends_with(".gguf") {
            grim_engine::model_loader::load_model_from_gguf(path, device.clone())?
        } else if lower.ends_with(".grim") {
            grim_engine::model_loader::load_model_from_grim(path, device.clone())?
        } else if lower.ends_with(".safetensors") || lower.ends_with(".bin") {
            grim_engine::model_loader::load_model_from_safetensors(path, device.clone())?
        } else {
            return Err(grim_core::error::Error::Config(format!(
                "unsupported model format for '{}'",
                path
            )));
        }
    } else {
        let cfg = grim_models_transformer::LlamaConfig {
            vocab_size: 512,
            hidden_size: 64,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 32,
            num_layers: 1,
            intermediate_size: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 256,
        };
        Box::new(grim_models_transformer::Llama::random(device.clone(), cfg))
    };
    let start = std::time::Instant::now();

    use grim_core::session::Inner;
    for _ in 0..concurrency {
        let inp = grim_backend_cpu::cpu_tensor(
            (0..tokens).map(|t| (t % 512) as f32).collect(),
            grim_tensor::Shape::new(vec![tokens]),
        );
        let mut sess = Inner::new(model.device().clone());
        let _ = model.forward(&mut sess, &inp, &inp, &[])?;
    }

    let elapsed = start.elapsed();
    println!(
        "[grim] bench: {} tokens x {} concurrency in {:?}",
        tokens, concurrency, elapsed
    );
    Ok(())
}
