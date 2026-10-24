//! Perplexity of a token range WITH a preceding context, scored the way
//! llama.cpp's `perplexity` tool does it.
//!
//! The tool evaluates a context window of `n_ctx` tokens and computes the
//! perplexity over the LAST HALF of it (`const int first = n_ctx/2;` in
//! `tools/perplexity/perplexity.cpp`), so every scored token already has
//! `n_ctx/2` tokens of context. Grim's own `GRIM_SCORE_FILE` harness scores
//! every token but the first, including the context-free one - the two
//! harnesses are NOT comparable as they stand.
//!
//! This probe reproduces llama.cpp's convention exactly: it feeds the first
//! `n_ctx + 1` tokens and scores the logits at positions `n_ctx/2 .. n_ctx`,
//! which predict the tokens `n_ctx/2 + 1 ..= n_ctx`.
//!
//! Usage: `cargo run --release --example gemma4_ppl_probe -- <model.gguf> <text> [n_ctx]`

use grim_core::model::CausalLm;
use grim_core::session::Inner as SessionInner;
use grim_tensor::Device;

fn main() {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().expect("usage: gemma4_ppl_probe <model> <text> [n_ctx]");
    let text = args.next().expect("text file");
    let n_ctx: usize = args
        .next()
        .map(|v| v.parse().unwrap())
        .unwrap_or(512);

    let text = std::fs::read_to_string(&text).expect("read text");
    let provider = grim_format::GgufProvider::open(&model_path).expect("open gguf");
    let tok = provider.tokenizer().expect("tokenizer");
    let mut ids = tok.encode(&text);
    // llama.cpp's perplexity tool adds the BOS to the first chunk.
    if tok.add_bos_token {
        if let Some(bos) = tok.bos_token_id {
            ids.insert(0, bos);
        }
    }
    println!(
        "[probe] {} tokens in the file, scoring with n_ctx={n_ctx}",
        ids.len()
    );
    if ids.len() < n_ctx + 1 {
        eprintln!("[probe] need at least {} tokens", n_ctx + 1);
        std::process::exit(2);
    }

    let model = grim_engine::model_loader::load_from_path_on_device(&model_path, Device::Cpu)
        .expect("load model");

    // Feed the first n_ctx + 1 tokens: the extra one is the target of the
    // last scored position.
    let feed = &ids[..n_ctx + 1];
    let input = grim_backend_cpu::cpu_tensor(
        feed.iter().map(|&i| i as f32).collect::<Vec<f32>>(),
        grim_tensor::Shape::new(vec![feed.len()]),
    );
    let positions = grim_backend_cpu::cpu_tensor(
        (0..feed.len()).map(|i| i as f32).collect::<Vec<f32>>(),
        grim_tensor::Shape::new(vec![feed.len()]),
    );

    let mut session = SessionInner::new(Device::Cpu);
    let logits = CausalLm::forward(&*model, &mut session, &input, &positions, &[]).expect("forward");
    let lv = logits.to_vec_f32().expect("logits");
    let vocab = lv.len() / feed.len();

    // llama.cpp scores the positions >= n_ctx/2; the logits at position p
    // predict the token at p + 1.
    let first = n_ctx / 2;
    let mut nll = 0.0f64;
    let mut n = 0usize;
    for p in first..n_ctx {
        let row = &lv[p * vocab..(p + 1) * vocab];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let lse: f32 = row.iter().map(|&x| (x - max).exp()).sum::<f32>().ln() + max;
        let tgt = ids[p + 1] as usize;
        nll += -((row[tgt] - lse) as f64);
        n += 1;
    }
    println!(
        "[probe] scored {n} tokens (positions {first}..{}), ppl={:.4}",
        n_ctx - 1,
        (nll / n as f64).exp()
    );
}
