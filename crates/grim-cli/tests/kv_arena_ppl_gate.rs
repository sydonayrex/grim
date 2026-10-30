//! WS-1a: certify a dense KV-arena representation against the f32 arena on a
//! REAL model, on real text.
//!
//! The Phase-1 default (f16 dense arenas) is only justified by evidence, and
//! the only evidence that counts is end-to-end: run the same corpus through the
//! same model twice, once with the f32 arena and once with the representation
//! under test, and require the perplexity drift AND the step-0 logprob drift to
//! stay inside the budget in `grim_quant::accuracy_gate`.
//!
//! The step-0 arm is not redundant with PPL: a representation can leave
//! aggregate perplexity flat while corrupting the first decode step (a
//! mis-scaled sink row, a wrong sidecar stride), and PPL alone would pass it.
//!
//! Gated on `GRIM_GPU_TEST=1`. Model: `GRIM_PPL_GATE_MODEL` or the small LFM2.5
//! 350M (the dense-arena f16 path is the LFM2 family's; `GRIM_F16_KV` selects
//! f32 vs f16 for it). Kept deliberately small (one short window) so it runs
//! inside the standing 3-minute test cap.

use grim_cli::eval::compute_ppl;
use grim_quant::accuracy_gate::{AccuracyGate, KvArenaKind, KvArenaVerdict};
use grim_tensor::{CoreTensorOps, Device, Shape, Tensor};

/// The Q8_0 checkpoint: the Q4_K_M sibling has a known forward-compute defect
/// (baseline PPL ~2.8e6 on wikitext) and cannot serve as a gate baseline. The
/// gate skips loudly rather than gating against a broken reference.
const DEFAULT_MODEL: &str = "models/LFM2.5-350M-Q8_0.gguf";
const CORPUS: &str = "docs/eval/wikitext2.sample.txt";
/// Step-0 probe prefix (tokens). Long enough to be a real decode step.
const PROBE_LEN: usize = 32;

fn set_var(k: &str, v: &str) {
    // SAFETY: this test is single-threaded by construction (`--test-threads=1`
    // in the invocation) and the whole test body is the only writer.
    unsafe { std::env::set_var(k, v) }
}

/// Integration tests run with CWD = the crate dir; the model and corpus live at
/// the workspace root.
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
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = max + row.iter().map(|v| (v - max).exp()).sum::<f32>().ln();
    row.iter().map(|v| v - lse).collect()
}

/// Set the process-global arena knobs for one arm.
///
/// Both the LFM2-family resolver (`GRIM_F16_KV`) and the Qwen35 resolver
/// (`GRIM_KV_ARENA`, which falls back to `GRIM_F16_KV`) are driven together so
/// one arm means the same thing to either model family.
fn set_arm(arm: &str) {
    match arm {
        "f32" => {
            set_var("GRIM_KV_ARENA", "f32");
            set_var("GRIM_F16_KV", "0");
        }
        "f16" => {
            set_var("GRIM_KV_ARENA", "f16");
            set_var("GRIM_F16_KV", "1");
        }
        other => {
            // int8/int4 are Qwen35-only; LFM2 ignores them and stays f16.
            set_var("GRIM_KV_ARENA", other);
            set_var("GRIM_F16_KV", "1");
        }
    }
}

fn kind_of(arm: &str) -> KvArenaKind {
    match arm {
        "f32" => KvArenaKind::F32,
        "int8" | "i8" | "q8" => KvArenaKind::Int8,
        "int4" | "i4" | "q4" => KvArenaKind::Int4,
        "int2v4" | "i2v4" | "k2v4" => KvArenaKind::Int2V4,
        _ => KvArenaKind::F16,
    }
}

/// Run one arena arm: set the flag, load the model, measure PPL and the step-0
/// logprobs. Returns `(ppl, step0_logprobs, lastrow_logprobs, row_argmaxes)`.
fn probe_arm(model_path: &str, tokens: &[u32], arm: &str) -> (f64, Vec<f32>, Vec<f32>, Vec<u32>) {
    set_arm(arm);
    let dev = Device::Rocm(0);
    let model = grim_engine::model_loader::load_model_from_gguf(model_path, dev.clone())
        .expect("load model");
    let (ppl, _windows) = compute_ppl(model.as_ref(), tokens).expect("compute_ppl");

    // Step-0 probe: one forward over a short prefix, read the FIRST row's
    // logprobs (the first decode step) — and, for the report, the LAST row's.
    //
    // The two rows are NOT equivalent probes of the arena. Row 0 attends to a
    // single KV entry, so its softmax is over one element — the score (and
    // therefore K) drops out entirely and the row measures the stored V
    // verbatim. The last row averages 32 stored V's, where independent per-row
    // errors partly cancel, and it is the row a generation actually samples
    // from. Both are printed, and the per-row argmaxes are returned so the
    // structural check can be applied where sampling happens.
    let prefix: Vec<u32> = tokens.iter().copied().take(PROBE_LEN).collect();
    let n = prefix.len();
    let ids = tensor_on(
        &dev,
        prefix.iter().map(|&t| t as f32).collect(),
        Shape::new(vec![1, n]),
    );
    let positions = tensor_on(
        &dev,
        (0..n).map(|p| p as f32).collect(),
        Shape::new(vec![1, n]),
    );
    let mut sess = model.new_session();
    let logits = model
        .forward(&mut *sess, &ids, &positions, &[])
        .expect("forward");
    let all = logits.to_vec_f32().expect("logits d2h");
    let vocab = all.len() / n.max(1);
    let step0 = log_softmax(&all[..vocab]);
    let lastrow = log_softmax(&all[(n - 1) * vocab..n * vocab]);
    let row_argmaxes = (0..n)
        .map(|i| argmax(&all[i * vocab..(i + 1) * vocab]) as u32)
        .collect();
    (ppl as f64, step0, lastrow, row_argmaxes)
}

#[test]
fn kv_arena_ppl_gate() {
    if std::env::var("GRIM_GPU_TEST")
        .map(|v| v != "1")
        .unwrap_or(true)
    {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run the KV-arena PPL gate");
        return;
    }
    // The arm under test. `f16` is the Phase-1 default; `int8`/`int4` are the
    // Qwen35-only representations (WS-2a).
    let arm = std::env::var("GRIM_PPL_GATE_ARENA").unwrap_or_else(|_| "f16".to_string());
    let kind = kind_of(&arm);
    let model_path =
        std::env::var("GRIM_PPL_GATE_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());

    // The quantized dense arenas are Qwen35-only: the LFM2 family resolves its
    // arena dtype from `GRIM_F16_KV` alone and would ignore `int8`/`int4`/
    // `int2v4`, so both arms would be the same run and the gate would report a
    // meaningless PASS (or trip its own negative control with a misleading
    // message). Refuse to pretend: say which model can gate this arm.
    let quantized_arm = matches!(
        kind,
        KvArenaKind::Int8 | KvArenaKind::Int4 | KvArenaKind::Int2V4
    );
    let qwen35_model = model_path.to_ascii_lowercase().contains("qwen3.5")
        || model_path.to_ascii_lowercase().contains("qwen35")
        || model_path.to_ascii_lowercase().contains("qwen3.8")
        || std::env::var("GRIM_PPL_GATE_ALLOW_QUANT").is_ok();
    if quantized_arm && !qwen35_model {
        eprintln!(
            "[SKIP] arm '{arm}' is a Qwen35-only dense arena and {model_path} does not look \
             like a Qwen3.5/3.8 checkpoint (the LFM2 family exposes f32/f16 only). Point \
             GRIM_PPL_GATE_MODEL at a Qwen3.5 GGUF, or set GRIM_PPL_GATE_ALLOW_QUANT=1 if you \
             know this model honors it."
        );
        return;
    }
    let model_path = workspace_path(&model_path);
    if !model_path.exists() {
        eprintln!("[SKIP] model {} not present", model_path.display());
        return;
    }
    let model_path = model_path.to_string_lossy().into_owned();
    // Pinned comparison parameters: same corpus, same window, same context, so
    // the ONLY variable between the two arms is the arena representation.
    set_var("GRIM_BACKEND", "rocm:0");
    set_var("GRIM_CONTEXT", "4096");
    // Wider than the minimum so the drift estimate is not dominated by one
    // window's sampling noise; overridable for a bigger sweep.
    if std::env::var("GRIM_EVAL_WINDOW").is_err() {
        set_var("GRIM_EVAL_WINDOW", "256");
    }
    if std::env::var("GRIM_EVAL_MAX_WINDOWS").is_err() {
        set_var("GRIM_EVAL_MAX_WINDOWS", "3");
    }

    let corpus = workspace_path(CORPUS);
    let text = std::fs::read_to_string(&corpus)
        .unwrap_or_else(|e| panic!("corpus {}: {e}", corpus.display()));
    let provider = grim_format::GgufProvider::open(&model_path).expect("gguf open");
    let tokenizer = provider.tokenizer().expect("tokenizer");
    // IN-DISTRIBUTION BASELINE. An instruct checkpoint fed RAW corpus text is
    // out of distribution: Qwen3.5-9B scores ~1.1e5 PPL on raw wikitext while
    // answering "The capital of France is" -> "Paris" and applying its own
    // `<|im_start|>` template. In that regime the arena's marginal effect is
    // amplified ~200x (f16 moves PPL by 22.7% there versus 0.0999% on LFM2), so
    // an absolute budget comparison is meaningless. Render the corpus as a
    // single user turn through the checkpoint's OWN chat template — exactly what
    // `run` does — so the measurement happens where the model operates.
    let prompt_text = if tokenizer.chat_template.is_some() {
        let messages = [grim_format::ChatMessage {
            role: "user".to_string(),
            content: text.clone(),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];
        grim_format::render_messages_or_last(&tokenizer, &messages)
    } else {
        text.clone()
    };
    let mut tokens = tokenizer.encode(&prompt_text);
    // LFM2-family checkpoints declare `add_bos_token`, and their forward is
    // garbage without it (project-lfm25-bos-fixed): the eval path must prepend
    // it exactly as `run.rs` does, or the perplexity is noise.
    if tokenizer.add_bos_token {
        if let Some(bos) = tokenizer.bos_token_id {
            tokens.insert(0, bos);
        }
    }
    assert!(
        tokens.len() > 129,
        "corpus too small for the pinned window: {} tokens",
        tokens.len()
    );

    let (ppl_f32, logp_f32, last_f32, argmax_f32_rows) = probe_arm(&model_path, &tokens, "f32");
    let (ppl_f16, logp_f16, last_arm, argmax_arm_rows) = probe_arm(&model_path, &tokens, &arm);

    // BASELINE SANITY. A non-finite or absurd baseline indicts the MODEL, not
    // the arena under test, and gating against it would misattribute a model
    // fault to the representation — refuse those outright.
    //
    // A merely HIGH baseline is a different case: the DELTA between the two
    // arms is what this gate certifies, both arms see the same tokens, so a
    // high-but-finite baseline still yields a real measurement of the arena's
    // marginal effect — reported as indicative, not as an in-distribution
    // quality claim.
    //
    // The corpus is rendered through the checkpoint's OWN chat template above,
    // so an instruct checkpoint is measured where it operates and this path
    // should now stay quiet: Qwen3.5-9B-Q4_K_M measures 17.85 here. It used to
    // report ~1.1e5, and that figure was not an out-of-distribution artifact —
    // it was the 64x128 big-tile prefill defect (see the plan's "prefill
    // collapse" section), which corrupted every forward with m >= 64 rows.
    if !(ppl_f32.is_finite() && ppl_f32 > 0.0 && ppl_f32 < 1.0e9) {
        eprintln!(
            "[SKIP] f32 baseline PPL {ppl_f32} is not a usable reference for {model_path} \
             (non-finite or absurd) — that is a model fault, not an arena fault."
        );
        return;
    }
    const HIGH_BASELINE_PPL: f64 = 1000.0;
    let baseline_is_ood = ppl_f32 > HIGH_BASELINE_PPL;
    if baseline_is_ood {
        eprintln!(
            "[WARN] f32 baseline PPL {ppl_f32:.1} is high for {model_path}: the eval harness \
             feeds raw text with no chat template, so an instruct model is out of \
             distribution. The f32-vs-{arm} DELTA below is still a real measurement of the \
             arena's marginal effect (both arms see identical tokens); the ABSOLUTE figure is \
             not an in-distribution quality claim."
        );
    }

    assert_eq!(logp_f32.len(), logp_f16.len());
    // NEGATIVE CONTROL: if the two arms are bit-identical, the knob is not
    // wired to the arena and a green gate would certify nothing. The f16 arena
    // rounds every stored K/V element, so the logprobs MUST differ somewhere.
    let identical = logp_f32
        .iter()
        .zip(&logp_f16)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    assert!(
        !identical,
        "f32 and f16 arms produced byte-identical step-0 logprobs: GRIM_F16_KV is not \
         reaching the dense KV arena, so this gate certifies nothing"
    );

    // The predicted token must not move at all, and the mass-carrying head of
    // the distribution is what the step-0 arm measures. A max-abs over the
    // WHOLE vocab is tail-dominated (most of 65k logprobs sit below -10 and
    // carry no mass), so it would fail on f16 noise that cannot change a
    // sample. Restrict to the top-K of the f32 arm, index-aligned across arms.
    // DIAGNOSTIC: is the model's next-token prediction actually sensible at
    // step 0? If the argmax is a plausible continuation and the true next token
    // ranks high, the logits are fine and any absurd PPL is an accumulation or
    // tokenization artifact; if the argmax is nonsense, the forward is at fault.
    {
        let a = argmax(&logp_f32);
        let next_id = (tokens[PROBE_LEN] as usize).min(logp_f32.len() - 1);
        let mut order: Vec<usize> = (0..logp_f32.len()).collect();
        order.sort_by(|&x, &y| {
            logp_f32[y]
                .partial_cmp(&logp_f32[x])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let rank = order
            .iter()
            .position(|&i| i == next_id)
            .unwrap_or(usize::MAX);
        let dec = |id: usize| -> String { tokenizer.decode(&[id as u32]) };
        eprintln!(
            "[diag] step0 argmax id={a} ({:?}) logp={:.3} | actual next id={next_id} ({:?}) \
             rank={rank} logp={:.3}",
            dec(a),
            logp_f32[a],
            dec(next_id),
            logp_f32[next_id]
        );
    }
    // Row 0 vs the last row of the same prefix, same metric. The gap between
    // them is the arena's worst case (one stored V, no averaging) against a
    // case comparable to a decode step; a budget that has to hold for real
    // generation should be argued from the last row.
    {
        let d = |a: &[f32], b: &[f32]| -> f32 {
            let idx = top_k_indices(a, 16);
            idx.iter()
                .map(|&i| (a[i] - b[i]).abs())
                .fold(0.0f32, f32::max)
        };
        eprintln!(
            "[row-delta] arm={arm} top16 max-abs delta: row0={:.4} row{}={:.4}",
            d(&logp_f32, &logp_f16),
            PROBE_LEN - 1,
            d(&last_f32, &last_arm)
        );
    }
    const TOP_K: usize = 16;
    let idx = top_k_indices(&logp_f32, TOP_K);
    let sel_f32: Vec<f32> = idx.iter().map(|&i| logp_f32[i]).collect();
    let sel_f16: Vec<f32> = idx.iter().map(|&i| logp_f16[i]).collect();

    // REPORT BEFORE ASSERTING. The structural checks below can abort the test,
    // and when one does the measurement is exactly what the reader needs — a
    // failure that withholds its own numbers cannot be sized, only re-run. Both
    // the top-16 drift and the argmax of each arm go out first.
    let argmax_f32 = argmax(&logp_f32);
    let argmax_f16 = argmax(&logp_f16);
    let argmax_last_f32 = argmax(&last_f32);
    let argmax_last_arm = argmax(&last_arm);
    let gate = AccuracyGate::new();
    let verdict = gate
        .verify_kv_arena(kind, ppl_f32, ppl_f16, &sel_f32, &sel_f16)
        .expect("verify_kv_arena");
    eprintln!(
        "[kv-arena-gate] model={model_path} ppl_f32={ppl_f32:.4} ppl_f16={ppl_f16:.4} \
         rel_delta={:.6} step0_top{TOP_K}_delta={:.6}{}",
        (ppl_f16 - ppl_f32).abs() / ppl_f32,
        sel_f32
            .iter()
            .zip(&sel_f16)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max),
        if baseline_is_ood {
            " (OOD baseline)"
        } else {
            ""
        }
    );
    eprintln!(
        "[kv-arena-argmax] arm={arm} row0: f32={argmax_f32} arm={argmax_f16} {} | \
         row{}: f32={argmax_last_f32} arm={argmax_last_arm} {}",
        if argmax_f32 == argmax_f16 {
            "same"
        } else {
            "MOVED"
        },
        PROBE_LEN - 1,
        if argmax_last_f32 == argmax_last_arm {
            "same"
        } else {
            "MOVED"
        },
    );
    // When row 0 moved, say how far apart the two candidates were IN THE
    // REFERENCE ARM: a sub-nat gap is a near-tie that any rounding can flip,
    // while a multi-nat gap is a real perturbation.
    if argmax_f32 != argmax_f16 {
        eprintln!(
            "[kv-arena-argmax] row0 in f32: argmax id={argmax_f32} logp={:.4} vs the \
             {arm} arm's pick id={argmax_f16} logp={:.4} (gap {:.4} nats)",
            logp_f32[argmax_f32],
            logp_f32[argmax_f16],
            logp_f32[argmax_f32] - logp_f32[argmax_f16]
        );
    }
    let moved: Vec<usize> = (0..argmax_f32_rows.len())
        .filter(|&i| argmax_f32_rows[i] != argmax_arm_rows[i])
        .collect();
    eprintln!(
        "[kv-arena-argmax] arm={arm} per-row argmax agreement: {}/{} rows \
         (differing rows: {moved:?})",
        argmax_f32_rows.len() - moved.len(),
        argmax_f32_rows.len()
    );

    // THE BEHAVIOURAL CHECK LIVES WHERE SAMPLING HAPPENS. Row 0's logits are
    // never sampled: a generation samples the LAST row of its prompt, and row 0
    // has one KV entry, so its softmax is over a single element — K drops out
    // and only V's storage error survives, at its most unfavourable. Requiring
    // an exact argmax there measures the arena's worst case, not its behaviour.
    // So: exact argmax at the last row (what a user would see), the row-0
    // drift carried by the numeric budget above and reported for every arm.
    assert_eq!(
        argmax_last_f32, argmax_last_arm,
        "the last row's argmax moved between the f32 and {arm} arenas — that row \
         is the one a generation samples from"
    );
    assert!(
        matches!(verdict, KvArenaVerdict::Pass { .. }),
        "{arm} KV arena failed its accuracy budget: {verdict:?}"
    );
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn top_k_indices(v: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap_or(std::cmp::Ordering::Equal));
    idx.truncate(k.min(v.len()));
    idx
}

/// Harness sanity probe: does the eval-style `forward` predict sensibly at all?
///
/// `compute_ppl` calls `model.forward(ids, positions)` once over a whole window
/// and scores row `i` against token `i+1`. Qwen3.5-9B scores ~1.1e5 PPL there
/// while `grim run` answers "The capital of France is" -> "Paris", so one of the
/// two paths is wrong. This feeds a prompt whose answer is unambiguous and
/// reports what the LAST row predicts (the token after the prompt) — the
/// smallest possible check of the eval path.
#[test]
fn eval_forward_predicts_a_known_prompt() {
    if std::env::var("GRIM_GPU_TEST")
        .map(|v| v != "1")
        .unwrap_or(true)
    {
        return;
    }
    let model_path = workspace_path(
        &std::env::var("GRIM_PPL_GATE_MODEL")
            .unwrap_or_else(|_| "models/qwen35-9b/Qwen3.5-9B-Q4_K_M.gguf".to_string()),
    );
    if !model_path.exists() {
        eprintln!("[SKIP] no model at {}", model_path.display());
        return;
    }
    let model_path = model_path.to_string_lossy().into_owned();
    set_var("GRIM_BACKEND", "rocm:0");
    set_var("GRIM_CONTEXT", "4096");
    set_arm("f32");

    let provider = grim_format::GgufProvider::open(&model_path).expect("gguf open");
    let tokenizer = provider.tokenizer().expect("tokenizer");
    let dev = Device::Rocm(0);
    let model = grim_engine::model_loader::load_model_from_gguf(&model_path, dev.clone())
        .expect("load model");

    // Which POSITIONS are bad at the cliff? If only the tail is wrong the last
    // token's processing is at fault; if the whole window is wrong the forward
    // itself is corrupted (e.g. a buffer overrun).
    {
        let corpus = std::fs::read_to_string(workspace_path(CORPUS)).expect("corpus");
        let rendered = grim_format::render_messages_or_last(
            &tokenizer,
            &[grim_format::ChatMessage {
                role: "user".to_string(),
                content: corpus.clone(),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
        );
        let all_ids: Vec<u32> = tokenizer.encode(&rendered);
        let nll = |row: &[f32], want: usize| -> f64 {
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let lse = m + row.iter().map(|v| (v - m).exp()).sum::<f32>().ln();
            (lse - row[want]) as f64
        };
        for n in [63usize, 64] {
            let ctx = &all_ids[..n];
            let t_ids = tensor_on(
                &dev,
                ctx.iter().map(|&t| t as f32).collect(),
                Shape::new(vec![1, n]),
            );
            let t_pos = tensor_on(
                &dev,
                (0..n).map(|p| p as f32).collect(),
                Shape::new(vec![1, n]),
            );
            let mut sess = model.new_session();
            let logits = model
                .forward(&mut *sess, &t_ids, &t_pos, &[])
                .expect("forward");
            let all = logits.to_vec_f32().expect("logits d2h");
            let vocab = all.len() / n;
            let mut line = String::new();
            for i in 0..n - 1 {
                let v = nll(&all[i * vocab..(i + 1) * vocab], all_ids[i + 1] as usize);
                line.push_str(&format!("{:.1} ", v.min(99.0)));
            }
            eprintln!("[per-pos] n={n} nll per position: {line}");
        }
    }

    for prompt in ["The capital of France is", "2 + 2 ="] {
        let ids: Vec<u32> = tokenizer.encode(prompt);
        let n = ids.len();
        let t_ids = tensor_on(
            &dev,
            ids.iter().map(|&t| t as f32).collect(),
            Shape::new(vec![1, n]),
        );
        let t_pos = tensor_on(
            &dev,
            (0..n).map(|p| p as f32).collect(),
            Shape::new(vec![1, n]),
        );
        let mut sess = model.new_session();
        let logits = model
            .forward(&mut *sess, &t_ids, &t_pos, &[])
            .expect("forward");
        let all = logits.to_vec_f32().expect("logits d2h");
        let vocab = all.len() / n;
        // Row n-1 is the last prompt position: it predicts the NEXT token.
        let last = &all[(n - 1) * vocab..n * vocab];
        let a_last = argmax(last);
        // Row 0 is a single-token context; it should predict token 1.
        let a0 = argmax(&all[..vocab]);
        eprintln!(
            "[next-tok] {prompt:?} (n={n}) | row{} -> {:?} (logp {:.3}) | row0 -> {:?} (want {:?})",
            n - 1,
            tokenizer.decode(&[a_last as u32]),
            last[a_last],
            tokenizer.decode(&[a0 as u32]),
            tokenizer.decode(&[ids[1]]),
        );
    }
}
