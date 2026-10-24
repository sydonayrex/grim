//! T4.2 — long-context RETRIEVAL gate for the int4 KV arena.
//!
//! Standard perplexity windows stay clean while multi-needle retrieval breaks
//! (ai-cache-expert / ai-arch-expert), so a 256-token PPL gate is not evidence
//! that int4 is safe at depth. This test plants a unique fact at ~10%, ~50% and
//! ~90% depth of a haystack, asks for it verbatim, and compares the LOGPROB of
//! the correct answer token under `f16` and `int4`.
//!
//! The logprob is the measurement that matters: at 4 bits a single-bit answer
//! difference is a coin flip, so "did it get the token right" is noise; the
//! margin by which the right token leads is the signal.
//!
//! The gate asserts int4 does not LOSE a needle f16 retrieves (by more than
//! `MARGIN` nats). A loss is reported as a real limitation of int4, not
//! loosened away.
//!
//! Run (single GPU so the shared multi-GPU P2P path is not exercised):
//! `HIP_VISIBLE_DEVICES=0 GRIM_GPU_TEST=1 GRIM_CONTEXT=4096 \
//!  GRIM_PPL_GATE_MODEL=models/qwen35-9b/Qwen3.5-9B-Q4_K_M.gguf \
//!  cargo test -p grim-cli --test kv_int4_retrieval -- --nocapture --test-threads=1`

use grim_tensor::{CoreTensorOps, Device, Shape, Tensor};

/// The fact to retrieve, and the question that asks for it.
const NEEDLE: &str = " IMPORTANT: the secret vault passphrase is ZQX-7391. Remember it.";
const QUESTION: &str =
    "\n\nQuestion: What is the secret vault passphrase?\nAnswer: The secret vault passphrase is";
/// The first token of the expected answer, as literal text (leading space, as
/// the model is asked to continue a sentence).
const ANSWER_TEXT: &str = " Z";
/// Depths (fraction of the haystack) at which the needle is planted.
const DEPTHS: [f64; 3] = [0.10, 0.50, 0.90];
/// int4 may trail f16 by at most this many nats on the correct answer token.
const MARGIN: f32 = 2.0;
/// Target haystack size in tokens (bounded so prefill stays cheap).
const HAYSTACK_TOKENS: usize = 1200;

fn set_var(k: &str, v: &str) {
    // SAFETY: single-threaded by construction (`--test-threads=1`).
    unsafe { std::env::set_var(k, v) }
}

fn workspace_path(rel: &str) -> std::path::PathBuf {
    let p = std::path::Path::new(rel);
    if p.is_absolute() {
        return p.to_path_buf();
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    root.join(rel)
}

fn tensor_on(dev: &Device, data: Vec<f32>, shape: Shape) -> Tensor {
    let storage = match dev {
        Device::Rocm(o) => grim_backend_rocm::RocmDevice::shared(*o)
            .from_cpu(&data, &shape, grim_tensor::DType::F32)
            .expect("from_cpu"),
        other => panic!("unsupported device {other:?}"),
    };
    Tensor::new(
        std::sync::Arc::from(storage),
        shape,
        grim_tensor::DType::F32,
        grim_tensor::QuantProvenance::GrimNative,
        dev.clone(),
    )
}

fn log_softmax(row: &[f32]) -> Vec<f32> {
    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0f64;
    for &x in row {
        s += ((x - m) as f64).exp();
    }
    let ln = (s).ln() as f32;
    row.iter().map(|&x| (x - m) - ln).collect()
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

fn set_arm(arm: &str) {
    match arm {
        "f16" => {
            set_var("GRIM_KV_ARENA", "f16");
            set_var("GRIM_F16_KV", "1");
        }
        other => {
            set_var("GRIM_KV_ARENA", other);
            set_var("GRIM_F16_KV", "1");
        }
    }
}

/// Load `model`, forward one prompt, and return (logprob of `answer_id` at the
/// last row, argmax id at the last row, rank of `answer_id`).
fn answer_logprob(model_path: &str, prompt: &str, answer_id: u32) -> (f32, u32, usize) {
    let dev = Device::Rocm(0);
    let provider = grim_format::GgufProvider::open(model_path).expect("gguf open");
    let tokenizer = provider.tokenizer().expect("tokenizer");
    let mut tokens = tokenizer.encode(prompt);
    if tokenizer.add_bos_token {
        if let Some(bos) = tokenizer.bos_token_id {
            tokens.insert(0, bos);
        }
    }
    let n = tokens.len();
    let model = grim_engine::model_loader::load_model_from_gguf(model_path, dev.clone())
        .expect("load model");
    let ids = tensor_on(&dev, tokens.iter().map(|&t| t as f32).collect(), Shape::new(vec![1, n]));
    let positions = tensor_on(&dev, (0..n).map(|p| p as f32).collect(), Shape::new(vec![1, n]));
    let mut sess = model.new_session();
    let logits = model
        .forward(&mut *sess, &ids, &positions, &[])
        .expect("forward");
    let all = logits.to_vec_f32().expect("logits d2h");
    let vocab = all.len() / n.max(1);
    let row = log_softmax(&all[(n - 1) * vocab..n * vocab]);
    let best = argmax(&row) as u32;
    let lp = row[answer_id as usize];
    let rank = row.iter().filter(|&&x| x > lp).count();
    (lp, best, rank)
}

#[test]
fn kv_int4_retrieval_gate() {
    if std::env::var("GRIM_GPU_TEST").map(|v| v != "1").unwrap_or(true) {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run the int4 retrieval gate");
        return;
    }
    let model_rel = std::env::var("GRIM_PPL_GATE_MODEL")
        .unwrap_or_else(|_| "models/qwen35-9b/Qwen3.5-9B-Q4_K_M.gguf".to_string());
    let model_path = workspace_path(&model_rel);
    if !model_path.exists() {
        eprintln!("[SKIP] model {} not present", model_path.display());
        return;
    }
    let model_path = model_path.to_string_lossy().into_owned();
    set_var("GRIM_BACKEND", "rocm:0");
    set_var("GRIM_CONTEXT", "4096");

    let corpus = workspace_path("docs/eval/wikitext2.sample.txt");
    let filler = std::fs::read_to_string(&corpus).unwrap_or_else(|e| panic!("corpus: {e}"));

    // Build the haystack ONCE, then reuse the exact same prompt for both arms so
    // the only variable is the arena representation.
    let provider = grim_format::GgufProvider::open(&model_path).expect("gguf open");
    let tokenizer = provider.tokenizer().expect("tokenizer");
    let answer_id = tokenizer
        .encode(ANSWER_TEXT)
        .first()
        .copied()
        .expect("answer token");
    let words: Vec<&str> = filler.split_whitespace().collect();

    let mut prompts: Vec<(f64, String)> = Vec::new();
    for &depth in DEPTHS.iter() {
        let mut text = String::new();
        let target_words = HAYSTACK_TOKENS * 3 / 4; // ~0.75 words per token
        let at = (target_words as f64 * depth) as usize;
        for (i, w) in words.iter().take(target_words).enumerate() {
            if i == at {
                text.push_str(NEEDLE);
            }
            text.push(' ');
            text.push_str(w);
        }
        text.push_str(QUESTION);
        prompts.push((depth, text));
    }

    let mut results: Vec<(f64, f32, f32, u32, u32, usize, usize)> = Vec::new();
    for arm in ["f16", "int4"] {
        set_arm(arm);
        for (depth, prompt) in &prompts {
            let (lp, best, rank) = answer_logprob(&model_path, prompt, answer_id);
            eprintln!(
                "[retrieval] arm={arm} depth={depth:.2} ntok={} logp(correct)={lp:.3} \
                 argmax={best} correct_rank={rank}",
                tokenizer.encode(prompt).len()
            );
            // Stash per-arm; assembled below in the same order.
            if arm == "f16" {
                results.push((*depth, lp, 0.0, best, 0, rank, 0));
            } else {
                let slot = results
                    .iter_mut()
                    .find(|r| (r.0 - depth).abs() < 1e-9)
                    .expect("f16 row first");
                slot.2 = lp;
                slot.4 = best;
                slot.6 = rank;
            }
        }
    }

    let mut losses = 0usize;
    for (depth, lp_f16, lp_i4, best_f16, best_i4, rank_f16, rank_i4) in &results {
        let delta = lp_i4 - lp_f16;
        eprintln!(
            "[retrieval-gate] depth={depth:.2} f16: logp={lp_f16:.3} argmax={best_f16} rank={rank_f16} \
             | int4: logp={lp_i4:.3} argmax={best_i4} rank={rank_i4} | delta={delta:+.3} nats"
        );
        if delta < -MARGIN {
            losses += 1;
        }
    }
    assert_eq!(
        losses, 0,
        "int4 lost a needle that f16 retrieved (>{MARGIN} nats behind) at {losses} depth(s); \
         record this as a real int4 limitation rather than loosening the margin"
    );
}
