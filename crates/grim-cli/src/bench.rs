//! `grim bench` — benchmark / smoke test.

use grim_core::error::Result;
use grim_models_transformer::{Llama, LlamaConfig};

pub async fn cmd_bench(tokens: usize, concurrency: usize) -> Result<()> {
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
        max_seq_len: 256,
    };
    let model = Llama::random(cfg);
    let start = std::time::Instant::now();

    use grim_core::CausalLm;
    use grim_core::session::Inner;
    for _ in 0..concurrency {
        let inp = grim_backend_cpu::cpu_tensor(
            (0..tokens).map(|t| (t % 512) as f32).collect(),
            grim_tensor::Shape::new(vec![tokens]),
        );
        let mut sess = Inner::new(model.device.clone());
        let _ = model.forward(&mut sess, &inp, &inp)?;
    }
    let elapsed = start.elapsed();
    println!(
        "Bench: {concurrency} run(s), {tokens} tokens each = {} total tokens in {:.2}s ({:.1} tok/s)",
        concurrency * tokens,
        elapsed.as_secs_f64(),
        (concurrency * tokens) as f64 / elapsed.as_secs_f64()
    );
    Ok(())
}