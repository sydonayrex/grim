//! `grim bench` — benchmark / smoke test.

use grim_core::error::{Error, Result};
use grim_core::model::CausalLm;
use grim_tensor::Device;

/// Fixed prompt list for serving-mode load generation (§WI-E2). Sized for
/// 200–600 token prompts against a chat template.
const SERVE_PROMPTS: &[&str] = &[
    "Explain the difference between TCP and UDP, and when each is appropriate.",
    "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
    "Summarize the causes and consequences of the Industrial Revolution in three paragraphs.",
    "Given a list of daily temperatures, describe an efficient algorithm to find how many days until warmer weather for each day.",
    "Compare and contrast supervised, unsupervised, and reinforcement learning with concrete examples.",
    "Draft a polite email declining a meeting invitation while proposing an alternative time.",
    "Explain how a transformer's self-attention mechanism works, step by step.",
    "What are the tradeoffs between a hash table and a balanced binary search tree?",
    "Describe the water cycle from evaporation to precipitation in detail.",
    "Write a Python function that merges two sorted lists without using built-in sort.",
    "Explain why the sky is blue during the day and red at sunset.",
    "Outline a training plan for a beginner preparing for a 10K run in eight weeks.",
    "Describe how modern compilers optimize loops, including vectorization.",
    "What were the main artistic movements of the twentieth century and their defining traits?",
    "Explain the CAP theorem and its practical implications for distributed databases.",
    "Write a product description for a stainless steel water bottle, 100 words.",
    "How does public-key cryptography work? Explain RSA at a high level.",
    "Describe the process of photosynthesis including the light and dark reactions.",
    "What is technical debt, how does it accumulate, and how should teams manage it?",
    "Explain the difference between latency and bandwidth with real-world analogies.",
];

pub async fn cmd_bench(
    tokens: usize,
    concurrency: usize,
    model_path: Option<&str>,
    mode: &str,
    port: u16,
    duration_secs: u64,
) -> Result<()> {
    if mode == "serve" {
        return cmd_bench_serve(port, concurrency, duration_secs).await;
    }
    let device = Device::Cpu;
    // F-5: resolve catalog/cache names ("LFM2.5-350M-Q8_0", "name:gguf") to
    // real paths before the extension dispatch, mirroring `grim run`. A bare
    // filename without an extension previously failed with "No such file".
    let model: Box<dyn CausalLm> = if let Some(path) = model_path {
        let resolved = crate::catalog::resolve_model_path(path).ok_or_else(|| {
            grim_core::error::Error::Config(format!(
                "model '{path}' not found in local cache or on disk"
            ))
        })?;
        let path_str = resolved.to_string_lossy().to_lowercase();
        if path_str.ends_with(".gguf") {
            grim_engine::model_loader::load_model_from_gguf(
                resolved.to_string_lossy().as_ref(),
                device.clone(),
            )?
        } else if path_str.ends_with(".grim") {
            grim_engine::model_loader::load_model_from_grim(
                resolved.to_string_lossy().as_ref(),
                device.clone(),
            )?
        } else if path_str.ends_with(".safetensors") || path_str.ends_with(".bin") {
            grim_engine::model_loader::load_model_from_safetensors(
                resolved.to_string_lossy().as_ref(),
                device.clone(),
            )?
        } else {
            return Err(grim_core::error::Error::Config(format!(
                "unsupported model format for '{}'",
                resolved.display()
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

/// WI-E2 serving mode: drive concurrent `/v1/chat/completions` load against a
/// running server, measure per-request wall time and inter-token latency
/// percentiles. Non-streaming requests; ITL is approximated as
/// request_wall_time / completion_tokens (per-request mean ITL).
async fn cmd_bench_serve(port: u16, concurrency: usize, duration_secs: u64) -> Result<()> {
    let addr = format!("127.0.0.1:{port}");
    // Health check first — fail loudly if the server isn't up.
    let health = reqwest::Client::new()
        .get(format!("http://{addr}/health"))
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await;
    if health.is_err() || !health.unwrap().status().is_success() {
        return Err(Error::Config(format!(
            "serving bench: no server at {addr} (start one with 'grim-cli serve')"
        )));
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .map_err(|e| Error::Config(format!("client: {e}")))?;
    let url = format!("http://{addr}/v1/chat/completions");

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(duration_secs.max(1));
    let mut request_latencies_ms: Vec<f64> = Vec::new();
    let mut completion_tokens_total: usize = 0;
    let mut request_count: usize = 0;

    let prompts_from_file = std::fs::read_to_string("docs/eval/prompts.txt").ok();
    let file_prompts: Vec<String> = prompts_from_file
        .as_ref()
        .map(|s| {
            s.lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .collect()
        })
        .unwrap_or_default();

    let worker_count = concurrency.max(1);
    let mut handles = Vec::with_capacity(worker_count);
    for w in 0..worker_count {
        let client = client.clone();
        let url = url.clone();
        let prompt = if !file_prompts.is_empty() {
            file_prompts[w % file_prompts.len()].clone()
        } else {
            SERVE_PROMPTS[w % SERVE_PROMPTS.len()].to_string()
        };
        handles.push(tokio::spawn(async move {
            let mut local_latencies: Vec<f64> = Vec::new();
            let mut local_itls: Vec<f64> = Vec::new();
            let mut local_tokens: usize = 0;
            let mut local_count: usize = 0;
            while std::time::Instant::now() < deadline {
                let body = serde_json::json!({
                    "model": "default",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 128
                });
                let t0 = std::time::Instant::now();
                let resp = match client.post(&url).json(&body).send().await {
                    Ok(r) => r,
                    Err(_) => break,
                };
                let v: serde_json::Value = match resp.json().await {
                    Ok(v) => v,
                    Err(_) => break,
                };
                let wall = t0.elapsed().as_secs_f64() * 1000.0;
                let n_tok = v["usage"]["completion_tokens"].as_u64().unwrap_or(0) as usize;
                local_latencies.push(wall);
                let itl = if n_tok > 0 {
                    wall / n_tok as f64
                } else {
                    wall / 128.0
                };
                local_itls.push(itl);
                local_tokens += n_tok;
                local_count += 1;
            }
            (local_latencies, local_itls, local_tokens, local_count)
        }));
    }

    let mut all_itls: Vec<f64> = Vec::new();
    for h in handles {
        if let Ok((lat, itls, toks, count)) = h.await {
            request_latencies_ms.extend(lat);
            all_itls.extend(itls);
            completion_tokens_total += toks;
            request_count += count;
        }
    }

    if request_latencies_ms.is_empty() {
        return Err(Error::Config(
            "serving bench: zero successful requests".into(),
        ));
    }
    request_latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    all_itls.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let pct = |p: f64| -> f64 {
        let idx = (((request_latencies_ms.len() as f64) * p) as usize)
            .clamp(1, request_latencies_ms.len())
            - 1;
        request_latencies_ms[idx]
    };
    let total_wall = duration_secs.max(1) as f64;
    let tps = completion_tokens_total as f64 / total_wall;
    let itl_pct = |p: f64| -> f64 {
        let idx = (((all_itls.len() as f64) * p) as usize).clamp(1, all_itls.len()) - 1;
        all_itls[idx]
    };

    println!(
        "serving_bench: requests={} completion_tokens={} tokens_per_sec={tps:.1}",
        request_count, completion_tokens_total
    );
    println!(
        "request_latency_ms: p50={:.1} p95={:.1} p99={:.1}",
        pct(0.50),
        pct(0.95),
        pct(0.99)
    );
    println!(
        "mean_itl_ms: p50={:.2} p95={:.2} p99={:.2}",
        itl_pct(0.50),
        itl_pct(0.95),
        itl_pct(0.99)
    );
    Ok(())
}

/// ITL percentile math on canned latencies (unit-testable core of serve mode).
pub fn percentiles_of(mut v: Vec<f64>) -> (f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| -> f64 {
        let idx = (((v.len() as f64) * p) as usize).clamp(1, v.len()) - 1;
        v[idx]
    };
    (pct(0.50), pct(0.95), pct(0.99))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn itl_percentile_math_known_answers() {
        let v: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let (p50, p95, p99) = percentiles_of(v);
        assert_eq!(p50, 50.0);
        assert_eq!(p95, 95.0);
        assert_eq!(p99, 99.0);
        // Single-element edge case.
        let (p50, _, _) = percentiles_of(vec![42.0]);
        assert_eq!(p50, 42.0);
    }

    #[test]
    fn serve_prompt_list_is_fixed_size() {
        // The plan pins a fixed 20-prompt list.
        assert_eq!(SERVE_PROMPTS.len(), 20);
        assert!(SERVE_PROMPTS.iter().all(|p| p.len() > 40));
    }
}
