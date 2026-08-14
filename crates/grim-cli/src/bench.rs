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

            partial_rotary_factor: 1.0,
            yarn: None,
        };
        Box::new(grim_models_transformer::Llama::random(device.clone(), cfg))
    };
    let start = std::time::Instant::now();

    for _ in 0..concurrency {
        // P1-3.6: Llama `forward` expects a 1-D `[seq_len]` input_ids tensor
        // (token IDs as f32, cast to u32 internally) and a matching positions
        // tensor. The original bench passed a flat `[tokens]` tensor for both,
        // which worked for `run` but caused a ShapeMismatch when the model's
        // RmsNorm / Linear layers flattened the 3-D hidden state to 2-D
        // `[batch, hidden]` before matmul — the residual add then saw
        // `[tokens, hidden]` where `[head_dim, hidden]` was expected.
        //
        // Reshape to `[1, tokens]` (explicit batch=1) so the model's
        // shape arithmetic (`elem_count / in_dim`) lands on the correct batch
        // dimension instead of collapsing 3-D to a flat 2-D.
        let input_data: Vec<f32> = (0..tokens).map(|t| (t % 512) as f32).collect();
        let inp =
            grim_backend_cpu::cpu_tensor(input_data, grim_tensor::Shape::new(vec![1, tokens]));
        // Separate positions tensor — values 0..seq_len, shape [1, tokens].
        let pos_data: Vec<f32> = (0..tokens).map(|t| t as f32).collect();
        let pos = grim_backend_cpu::cpu_tensor(pos_data, grim_tensor::Shape::new(vec![1, tokens]));
        let mut sess = model.new_session();
        let _ = model.forward(&mut *sess, &inp, &pos, &[])?;
    }

    let elapsed = start.elapsed();
    println!(
        "[grim] bench: {} tokens x {} concurrency in {:?}",
        tokens, concurrency, elapsed
    );
    Ok(())
}
