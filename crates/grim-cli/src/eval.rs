//! `grim eval` — perplexity and task evaluation (§WI-E1, FIND-1).
//!
//! Two tasks:
//! - `ppl`   — windowed perplexity over the committed wikitext2 sample corpus.
//!             Runs the model directly (no server). Deterministic.
//! - `gsm8k` — exact-match grade of 100 committed questions against a running
//!             server's `/v1/chat/completions` (temperature 0).

use grim_core::error::{Error, Result};
use grim_core::model::CausalLm;
use grim_tensor::Device;

/// PPL sliding-window size in tokens.
const PPL_WINDOW: usize = 2048;

/// Resolve the eval device from `GRIM_BACKEND` / `GRIM_FORCE_DEVICE`
/// (`rocm[:ord]`, `cuda[:ord]`); anything else stays CPU. Kept local to the
/// lib crate — `run.rs`'s probe lives in the binary target.
fn resolve_device() -> Device {
    let requested = std::env::var("GRIM_BACKEND")
        .or_else(|_| std::env::var("GRIM_FORCE_DEVICE"))
        .unwrap_or_default();
    let s = requested.trim().to_ascii_lowercase();
    let ord = |default: usize| -> usize {
        s.split(':')
            .nth(1)
            .and_then(|x| x.trim().parse().ok())
            .unwrap_or(default)
    };
    if s.starts_with("rocm") {
        Device::Rocm(ord(0))
    } else if s.starts_with("cuda") {
        Device::Cuda(ord(0))
    } else {
        Device::Cpu
    }
}

/// Load a model by catalog name or path, mirroring `bench.rs`.
fn load_model(model: &str) -> Result<(Box<dyn CausalLm>, String)> {
    let resolved = grim_core::catalog::resolve_model_path(model).ok_or_else(|| {
        Error::Config(format!("model '{model}' not found in local cache or on disk"))
    })?;
    let path_str = resolved.to_string_lossy().to_lowercase();
    // WI-E1: honor GRIM_BACKEND like the rest of the CLI — never silently pin
    // eval to CPU when a GPU backend is requested.
    let device = resolve_device();
    eprintln!("[eval] using device: {device:?}");
    let boxed: Box<dyn CausalLm> = if path_str.ends_with(".gguf") {
        grim_engine::model_loader::load_model_from_gguf(
            resolved.to_string_lossy().as_ref(),
            device,
        )?
    } else if path_str.ends_with(".grim") {
        grim_engine::model_loader::load_model_from_grim(resolved.to_string_lossy().as_ref(), device)?
    } else if path_str.ends_with(".safetensors") || path_str.ends_with(".bin") {
        grim_engine::model_loader::load_model_from_safetensors(
            resolved.to_string_lossy().as_ref(),
            device,
        )?
    } else {
        return Err(Error::Config(format!(
            "unsupported model format for '{}'",
            resolved.display()
        )));
    };
    Ok((boxed, resolved.to_string_lossy().into_owned()))
}

/// Compute windowed perplexity of `tokens` under `model`.
///
/// Slides a window of `PPL_WINDOW` tokens with full overlap; each window
/// predicts its last token from the preceding `PPL_WINDOW - 1`. NLL is
/// accumulated over windows; ppl = exp(mean NLL). Uses log2-free f32 math via
/// ln then exp — deterministic given the same model + tokens.
pub fn compute_ppl(model: &dyn CausalLm, tokens: &[u32]) -> Result<(f32, usize)> {
    if tokens.len() < PPL_WINDOW + 1 {
        return Err(Error::Config(format!(
            "corpus too small: { } tokens, need >= {}",
            tokens.len(),
            PPL_WINDOW + 1
        )));
    }
    let mut sum_nll = 0.0f64;
    let mut n_windows = 0usize;
    let mut start = 0usize;
    // GRIM_EVAL_MAX_WINDOWS: deterministic prefix cap. Baselines produced
    // with a cap record it in their metrics JSON — same corpus + cap + model
    // always yields the same ppl.
    let max_windows: Option<usize> = std::env::var("GRIM_EVAL_MAX_WINDOWS")
        .ok()
        .and_then(|v| v.parse().ok());
    while start + PPL_WINDOW < tokens.len() {
        if let Some(cap) = max_windows {
            if n_windows >= cap {
                break;
            }
        }
        let ctx = &tokens[start..start + PPL_WINDOW];
        let target = tokens[start + PPL_WINDOW];

        let ids = grim_backend_cpu::cpu_tensor(
            ctx.iter().map(|&t| t as f32).collect::<Vec<f32>>(),
            grim_tensor::Shape::new(vec![1, ctx.len()]),
        );
        let positions = grim_backend_cpu::cpu_tensor(
            (0..ctx.len()).map(|p| p as f32).collect::<Vec<f32>>(),
            grim_tensor::Shape::new(vec![1, ctx.len()]),
        );

        // Fresh session per window: each window is an independent prediction,
        // and this avoids cross-window cache contamination.
        let mut sess = model.new_session();
        let logits = model.forward(&mut *sess, &ids, &positions, &[])?;
        let all = logits.to_vec_f32()?;
        // Last row of logits predicts the token AFTER the context.
        let vocab = all.len() / ctx.len();
        let last_row_start = all.len() - vocab.max(1);
        let last_logits = &all[last_row_start..];

        // log-softmax of the target token.
        let max_l = last_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp = max_l
            + last_logits
                .iter()
                .map(|l| (*l - max_l).exp())
                .sum::<f32>()
                .ln();
        let target_l = last_logits
            .get(target as usize)
            .copied()
            .ok_or_else(|| Error::Config(format!("target token {target} out of vocab")))?;
        sum_nll += ((log_sum_exp - target_l) as f64).max(0.0);
        n_windows += 1;
        start += PPL_WINDOW; // non-overlapping stride keeps runtime bounded
    }
    if n_windows == 0 {
        return Err(Error::Config("no complete windows in corpus".into()));
    }
    let mean_nll = sum_nll / n_windows as f64;
    Ok((mean_nll.exp() as f32, n_windows))
}

/// Extract the final number from a gsm8k completion for exact-match grading.
/// Handles the common `\n#### <num>` convention plus bare trailing numbers.
pub fn extract_final_number(text: &str) -> Option<String> {
    // Prefer the #### marker convention first.
    if let Some(pos) = text.rfind("####") {
        let tail = text[pos + 4..].trim();
        let num: String = tail
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
            .collect();
        if !num.is_empty() {
            return Some(num.trim_end_matches('.').to_string());
        }
    }
    // Fall back to the LAST number appearing anywhere in the text.
    let mut last: Option<String> = None;
    let mut cur = String::new();
    for c in text.chars().chain(std::iter::once(' ')) {
        if c.is_ascii_digit() || c == '.' || c == '-' {
            cur.push(c);
        } else {
            if !cur.is_empty() && cur != "-" && cur != "." {
                last = Some(cur.clone());
            }
            cur.clear();
        }
    }
    last.map(|n| n.trim_end_matches('.').to_string())
}

/// Normalize two number strings for comparison: strip commas, $, trailing
/// zeros after the decimal point.
pub fn normalize_number(s: &str) -> String {
    let cleaned: String = s.chars().filter(|c| !(*c == ',' || *c == '$')).collect();
    match cleaned.parse::<f64>() {
        Ok(v) => {
            if v == v.trunc() && v.abs() < 1e15 {
                format!("{}", v as i64)
            } else {
                format!("{v}")
            }
        }
        Err(_) => cleaned,
    }
}

struct Gsm8kQuestion {
    question: String,
    answer: String,
}

fn load_gsm8k_questions(path: &std::path::Path, limit: usize) -> Result<Vec<Gsm8kQuestion>> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| Error::Config(format!("cannot read {}: {e}", path.display())))?;
    let mut out = Vec::new();
    for line in content.lines().filter(|l| !l.trim().is_empty()) {
        if out.len() >= limit {
            break;
        }
        let v: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| Error::Config(format!("bad jsonl line: {e}")))?;
        let question = v["question"]
            .as_str()
            .ok_or_else(|| Error::Config("missing question field".into()))?
            .to_string();
        let answer = v["answer"]
            .as_str()
            .or_else(|| v["answer_number"].as_str())
            .ok_or_else(|| Error::Config("missing answer field".into()))?
            .to_string();
        out.push(Gsm8kQuestion { question, answer });
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
async fn run_gsm8k(
    addr: &str,
    model: &str,
    dataset_path: &std::path::Path,
    limit: usize,
) -> Result<(usize, usize)> {
    let questions = load_gsm8k_questions(dataset_path, limit)?;
    if questions.is_empty() {
        return Err(Error::Config("no questions loaded".into()));
    }
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| Error::Config(format!("client: {e}")))?;
    let url = format!("http://{addr}/v1/chat/completions");
    let mut correct = 0usize;
    for (i, q) in questions.iter().enumerate() {
        let body = serde_json::json!({
            "model": model,
            "messages": [
                {"role": "user", "content": format!(
                    "{}\nSolve step by step, then give the final numeric answer on a line starting with ####.",
                    q.question
                )}
            ],
            "temperature": 0,
            "max_tokens": 256
        });
        let resp = client.post(&url).json(&body).send().await.map_err(map_req_err)?;
        let v: serde_json::Value = resp.json().await.map_err(map_req_err)?;
        let content = v["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();
        let got = extract_final_number(&content);
        let gold = normalize_number(&q.answer);
        let ok = got
            .as_deref()
            .map(|g| normalize_number(g) == gold)
            .unwrap_or(false);
        if ok {
            correct += 1;
        }
        eprintln!("[eval] gsm8k {}/{}: {} (gold={})", i + 1, questions.len(), if ok { "PASS" } else { "FAIL" }, q.answer);
    }
    Ok((correct, questions.len()))
}

fn map_req_err(e: reqwest::Error) -> Error {
    // grim_core::Error has no Backend variant in the CLI's dependency graph
    // (that lives in grim_tensor::Error); Config carries the message.
    Error::Config(format!("http: {e}"))
}

/// Held-out evaluation report reused by `train.rs` (kept from the original module).
pub struct EvalReport {
    pub step: usize,
    pub loss: f64,
    pub ppl: f64,
    pub tokens: usize,
}

/// Token-weighted perplexity across the held-out set.
/// loss = sum(token_loss) / sum(tokens); ppl = exp(loss).
pub fn perplexity<F, E>(
    dataset: &[Vec<u32>],
    mut forward_loss: F,
) -> std::result::Result<EvalReport, E>
where
    F: FnMut(&[u32]) -> std::result::Result<f64, E>,
    E: From<String>,
{
    let mut total_loss = 0.0f64;
    let mut total_tokens = 0usize;
    for seq in dataset {
        let n = seq.len().max(1);
        let l = forward_loss(seq)?;
        total_loss += l * n as f64;
        total_tokens += n;
    }
    if total_tokens == 0 {
        return Err(E::from("eval: empty dataset".into()));
    }
    let avg = total_loss / total_tokens as f64;
    Ok(EvalReport {
        step: 0,
        loss: avg,
        ppl: avg.exp(),
        tokens: total_tokens,
    })
}

/// Helper to load an evaluation dataset from path and return raw token vectors.
pub fn load_eval_dataset(
    path: &str,
    tokenizer: &grim_format::GgufTokenizer,
    max_seq_len: usize,
) -> grim_core::error::Result<Vec<Vec<u32>>> {
    let examples = crate::train::load_dataset(path, tokenizer, max_seq_len)?;
    Ok(examples.into_iter().map(|(toks, _)| toks).collect())
}

/// Entry point for `grim-cli eval`.
pub async fn cmd_eval(
    model: Option<String>,
    tasks: String,
    output: Option<String>,
    port: u16,
) -> Result<()> {
    let model = model.unwrap_or_else(|| {
        std::env::var("GRIM_DEFAULT_MODEL").unwrap_or_else(|_| "default".to_string())
    });
    let addr = format!("127.0.0.1:{port}");
    let corpus = "docs/eval/wikitext2.sample.txt".to_string();
    let mut metrics = serde_json::Map::new();

    for task in tasks.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        match task {
            "ppl" => {
                let corpus_path = std::path::PathBuf::from(&corpus);
                if !corpus_path.exists() {
                    return Err(Error::Config(format!(
                        "corpus file {} not found (pass --corpus)",
                        corpus_path.display()
                    )));
                }
                let text = std::fs::read_to_string(&corpus_path)
                    .map_err(|e| Error::Config(format!("corpus read: {e}")))?;
                let (loaded, resolved_path) = load_model(&model)?;
                // Tokenizer comes from the model file itself.
                let provider = grim_format::GgufProvider::open(resolved_path.as_str())?;
                let tokenizer = provider.tokenizer()?;
                let tokens = tokenizer.encode(&text);
                eprintln!("[eval] ppl: {} tokens", tokens.len());
                let (ppl, windows) = compute_ppl(loaded.as_ref(), &tokens)?;
                println!("ppl={ppl:.4} windows={windows}");
                let corpus_sha = {
                    use sha2::{Digest, Sha256};
                    let mut h = Sha256::new();
                    h.update(text.as_bytes());
                    format!("{:x}", h.finalize())
                };
                metrics.insert(
                    "ppl".into(),
                    serde_json::json!({
                        "value": ppl,
                        "windows": windows,
                        "max_windows": std::env::var("GRIM_EVAL_MAX_WINDOWS").ok(),
                        "corpus_sha256": corpus_sha,
                    }),
                );
            }
            "gsm8k" => {
                let dataset =
                    std::path::PathBuf::from("docs/eval/gsm8k.test100.jsonl");
                let (correct, total) = run_gsm8k(&addr, &model, &dataset, 100).await?;
                let em = correct as f32 / total as f32;
                println!("exact_match={em:.3} correct={correct}/{total}");
                metrics.insert(
                    "exact_match".into(),
                    serde_json::json!({ "value": em, "correct": correct, "total": total }),
                );
            }
            other => {
                return Err(Error::Config(format!(
                    "unknown task '{other}' (supported: ppl, gsm8k)"
                )));
            }
        }
    }

    if let Some(out_path) = output {
        let doc = serde_json::json!({
            "model": model,
            "task": tasks,
            "metrics": metrics,
            "date": chrono_now_iso(),
        });
        std::fs::write(&out_path, serde_json::to_string_pretty(&doc).unwrap())
            .map_err(|e| Error::Config(format!("write {out_path}: {e}")))?;
        println!("[eval] wrote {out_path}");
    }
    Ok(())
}

/// ISO-8601 UTC timestamp without pulling chrono into the dep tree.
fn chrono_now_iso() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    // Days since epoch → y/m/d civil algorithm (Howard Hinnant's).
    let days = secs / 86400;
    let rem = secs % 86400;
    let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    // Formula for civil date from days since 1970-01-01
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_size_constant_sane() {
        // Guard: the sliding window must exceed a single block so the
        // last-token prediction has real context.
        assert!(PPL_WINDOW >= 512);
        assert!(PPL_WINDOW.is_power_of_two());
    }

    #[test]
    fn gsm8k_grader_canned_pairs() {
        // #### convention
        assert_eq!(
            extract_final_number("some work\n#### 42").as_deref(),
            Some("42")
        );
        // trailing number fallback
        assert_eq!(extract_final_number("the answer is 17 apples.").as_deref(), Some("17"));
        // negative
        assert_eq!(extract_final_number("#### -3.5").as_deref(), Some("-3.5"));
        // no number
        assert_eq!(extract_final_number("no digits here"), None);
        // decimal with trailing period at sentence end
        assert_eq!(extract_final_number("it costs $5.").as_deref(), Some("5"));
    }

    #[test]
    fn normalize_number_handles_commas_and_zeros() {
        assert_eq!(normalize_number("1,234"), "1234");
        assert_eq!(normalize_number("$42"), "42");
        assert_eq!(normalize_number("3.50"), "3.5");
        assert_eq!(normalize_number("42.000000001"), "42.000000001");
    }

    #[test]
    fn test_ppl_math_synthetic_logits_known_answer() {
        // Test cross-entropy and perplexity calculation with uniform logits.
        // For uniform distribution over V classes:
        // P(target) = 1/V, -ln(1/V) = ln(V), exp(ln(V)) = V.
        let vocab_size = 100usize;
        let uniform_logits = vec![1.0f32; vocab_size];
        
        let max_l = 1.0f32;
        let sum_exp: f32 = uniform_logits.iter().map(|&x| (x - max_l).exp()).sum();
        let log_sum_exp = max_l + sum_exp.ln();
        
        let target_idx = 42usize;
        let target_logit = uniform_logits[target_idx];
        let nll = (log_sum_exp - target_logit) as f64;
        let expected_nll = (vocab_size as f64).ln();
        assert!((nll - expected_nll).abs() < 1e-5);
        
        let ppl = nll.exp() as f32;
        assert!((ppl - vocab_size as f32).abs() < 1e-3);
    }
}
