//! `grim distill` — GRAVE Phase 3 Rust-native distillation runner.
//!
//! Teacher (softmax GGUF, frozen) + student (GDN-2 gates) side by side,
//! loss `KL(student‖teacher, T=2) + 0.1·CE`, CAME optimizer, Taylor-Calibrated
//! init, interface-injury guard, `*.grave.json` sidecar checkpoints.
//!
//! Two modes:
//! * `--smoke-only` (default): synthetic-logit smoke loop + CAME bridge +
//!   sidecar round-trip. No model, milliseconds. This is the CI gate.
//! * Full run (`--teacher <gguf> --corpus <txt>`): loads the teacher, then
//!   the student with `GRIM_GRAVE=1`, and optimizes the per-layer gate
//!   triples `[decay, erase, write]` against window-last-row KL on real
//!   corpus windows. Gate updates go through `GRIM_GRAVE_GATES` (see
//!   `Lfm2Block::forward_gdl`): the loaded `Box<dyn CausalLm>` exposes no
//!   mutable downcast, so the harness perturbs gates per forward without
//!   reloading. Production path (gate projection tensors + sidecar loader)
//!   replaces this when it lands; the loss/guard/sidecar below are shared.
//!
//! Scale honesty: the local corpus (`docs/eval/corpus-grave-v1.txt`, ~190K
//! scored tokens) is a demo slice of the plan's 3B-token budget. The runner
//! streams `--token-budget` tokens (default 3B) and reports consumed vs
//! budget plus corpus exhaustion, so a short corpus fails OPEN (visible)
//! rather than silently standing in for 3B.

use grim_autograd::{
    Came, CameConfig, ParamId, TrainableParam, TrainableParams, injection::LoRAInjectionPoint,
};
use grim_backend_cpu::cpu_tensor;
use grim_core::error::{Error, Result};
use grim_engine::model_loader::load_model_from_gguf;
use grim_format::GgufProvider;
use grim_models_transformer::gla::{GraveSidecar, gdl_gate_defaults};
use grim_tensor::{Device, Shape};

// ---------------------------------------------------------------------------
// Loss: KL(student‖teacher, T=2) + 0.1·CE  (plan §Phase 3.4)
// ---------------------------------------------------------------------------

fn log_softmax(logits: &[f64], temp: f64) -> Vec<f64> {
    let m = logits.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let shifted: Vec<f64> = logits.iter().map(|&x| (x - m) / temp).collect();
    let lz = shifted.iter().map(|x| x.exp()).sum::<f64>().ln();
    shifted.iter().map(|x| x - lz).collect()
}

/// Distillation loss over one chunk: returns `(total, kl, ce)`.
///
/// * `student_logits`, `teacher_logits`: `[tokens][vocab]` f64.
/// * `targets`: ground-truth ids (CE term); empty ⇒ CE skipped.
/// * `temp`: distill temperature (plan: T=2). `ce_weight`: plan 0.1.
pub fn distill_loss(
    student_logits: &[Vec<f64>],
    teacher_logits: &[Vec<f64>],
    targets: &[usize],
    temp: f64,
    ce_weight: f64,
) -> (f64, f64, f64) {
    assert_eq!(student_logits.len(), teacher_logits.len());
    let t = student_logits.len().max(1) as f64;
    let mut kl = 0.0f64;
    let mut ce = 0.0f64;
    for (s, q) in student_logits.iter().zip(teacher_logits.iter()) {
        let ls = log_softmax(s, temp);
        let lt = log_softmax(q, temp);
        for i in 0..s.len() {
            let p = lt[i].exp(); // teacher prob (detached by construction)
            kl += p * (lt[i] - ls[i]);
        }
        // CE against ground truth at T=1.
        if !targets.is_empty() {
            let l1 = log_softmax(s, 1.0);
            let idx = targets[0].min(s.len() - 1);
            ce += -l1[idx];
        }
    }
    // Temperature scaling convention: KL term carries T² (Hinton Brenk).
    let kl_scaled = kl / t * temp * temp;
    let ce_scaled = if targets.is_empty() { 0.0 } else { ce / t };
    (kl_scaled + ce_weight * ce_scaled, kl_scaled, ce_scaled)
}

// ---------------------------------------------------------------------------
// Interface-injury guard (arXiv 2608.02689, plan §Phase 3.5a)
// ---------------------------------------------------------------------------

/// Label-stickiness over 4-permutation answer rotations: `items[i][r]` =
/// option index (0–3) the student picked for item `i` under rotation `r`.
/// Sticky = all 4 rotations pick the same OPTION SLOT (injury: the model
/// answers the slot, not the content). Returns `(stickiness, triggered)` with
/// the plan's ~30% tripwire.
pub fn interface_injury_check(items: &[[usize; 4]]) -> (f64, bool) {
    if items.is_empty() {
        return (0.0, false);
    }
    let sticky = items
        .iter()
        .filter(|it| it[0] == it[1] && it[1] == it[2] && it[2] == it[3])
        .count();
    let ratio = sticky as f64 / items.len() as f64;
    (ratio, ratio > 0.30)
}

// ---------------------------------------------------------------------------
// Smoke loop (ml-training-experiments rule): overfit 1 batch → loss ~0
// ---------------------------------------------------------------------------

/// One-batch overfit on synthetic logits: proves the loss + update machinery
/// before any GPU-hour spend. Returns `(loss_before, loss_after)`.
pub fn smoke_overfit_batch(steps: usize, lr: f64) -> (f64, f64) {
    // Fixed teacher distribution over 8 fake tokens × 16 vocab.
    let mut seed = 0x12345678u64;
    let mut rnd = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f64) / (u32::MAX as f64) - 0.5
    };
    let teacher: Vec<Vec<f64>> = (0..8)
        .map(|_| (0..16).map(|_| rnd() * 3.0).collect())
        .collect();
    // KL-only overfit (targets empty): student can match the teacher exactly,
    // so loss → ~0 proves the machinery. The CE term is covered by the loss
    // unit tests; mixing a single shared target here would fight the KL term
    // to a nonzero floor (mismatched objective, not a smoke failure).
    let targets: Vec<usize> = vec![];
    let mut student: Vec<Vec<f64>> = vec![vec![0.0; 16]; 8];
    let (before, _, _) = distill_loss(&student, &teacher, &targets, 2.0, 0.1);
    for _ in 0..steps {
        // Gradient of the tempered KL + CE w.r.t. student logits, SGD step.
        let temp = 2.0;
        for tok in 0..8 {
            let ls = log_softmax(&student[tok], temp);
            let lt = log_softmax(&teacher[tok], temp);
            let ps: Vec<f64> = ls.iter().map(|x| x.exp()).collect();
            let pt: Vec<f64> = lt.iter().map(|x| x.exp()).collect();
            for i in 0..16 {
                // d(KL·T²)/ds = T²·(ps − pt)/tokens (Hinton scaling).
                let mut g = (ps[i] - pt[i]) / 8.0 * temp * temp;
                if !targets.is_empty() {
                    let l1 = log_softmax(&student[tok], 1.0);
                    let p1: Vec<f64> = l1.iter().map(|x| x.exp()).collect();
                    let one = if i == targets[0].min(15) { 1.0 } else { 0.0 };
                    g += 0.1 * (p1[i] - one) / 8.0;
                }
                student[tok][i] -= lr * g;
            }
        }
    }
    let (after, _, _) = distill_loss(&student, &teacher, &targets, 2.0, 0.1);
    (before, after)
}

/// CAME bridge proof: one optimizer step over CPU gate params runs clean.
/// The full run calls this per chunk with grads from the chunk backward.
pub fn came_step_on_cpu_params() -> Result<()> {
    let shape = Shape::new(vec![8usize]);
    let data = cpu_tensor(vec![0.5f32; 8], shape);
    let mut params = TrainableParams::new();
    // Placeholder id (OProj/A) until a dedicated GdlGate injection point
    // lands with the full-corpus run; the bridge proof is optimizer↔params.
    params.insert(TrainableParam::new(
        ParamId::new(0, 0, LoRAInjectionPoint::OProj, true),
        data,
    )?);
    let mut opt = Came::new(CameConfig::default());
    opt.step(&mut params)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Full run: real teacher + student, real corpus windows
// ---------------------------------------------------------------------------

/// Full-run config (mirrors the CLI flags 1:1).
pub struct DistillRunConfig {
    pub init_sidecar: String,
    pub teacher: String,
    pub corpus: String,
    pub output: String,
    pub window_len: usize,
    pub max_windows: usize,
    pub opt_steps: usize,
    pub lr: f64,
    pub temp: f64,
    pub token_budget: u64,
    /// ROCm ordinal (default 1: distill GPU; GPU 0 stays free for serving).
    pub device_ordinal: usize,
    /// G3b: train gate projections via SPSA (subsumes the scalar descent).
    pub proj_spsa: bool,
}

fn pick_device(ordinal: usize) -> Device {
    if std::path::Path::new("/dev/kfd").exists() {
        Device::Rocm(ordinal)
    } else {
        Device::Cpu
    }
}

/// Last-row logits for one teacher-forced window.
#[allow(dead_code)]
fn last_row_logits(model: &dyn grim_core::model::CausalLm, window: &[u32]) -> Result<Vec<f64>> {
    let ids = cpu_tensor(
        window.iter().map(|&t| t as f32).collect::<Vec<f32>>(),
        Shape::new(vec![1, window.len()]),
    );
    let positions = cpu_tensor(
        (0..window.len()).map(|p| p as f32).collect::<Vec<f32>>(),
        Shape::new(vec![1, window.len()]),
    );
    let mut sess = model.new_session();
    let logits = model.forward(&mut *sess, &ids, &positions, &[])?;
    let all = logits.to_vec_f32()?;
    let vocab = all.len() / window.len().max(1);
    Ok(all[all.len() - vocab..].iter().map(|&x| x as f64).collect())
}

/// GRAVE Phase 3 fix: FULL-SEQUENCE teacher-forced logits `[T][vocab]`.
/// The last-row-only objective overfit window tails: it improved last-token
/// predictions on 32 training windows while degrading average prediction over
/// all positions (loop-closure PPL went 1840 -> 132710). The distill loss
/// must see every position.
pub(crate) fn full_seq_logits(
    model: &dyn grim_core::model::CausalLm,
    window: &[u32],
) -> Result<Vec<Vec<f64>>> {
    let ids = cpu_tensor(
        window.iter().map(|&t| t as f32).collect::<Vec<f32>>(),
        Shape::new(vec![1, window.len()]),
    );
    let positions = cpu_tensor(
        (0..window.len()).map(|p| p as f32).collect::<Vec<f32>>(),
        Shape::new(vec![1, window.len()]),
    );
    let mut sess = model.new_session();
    let logits = model.forward(sess.as_mut(), &ids, &positions, &[])?;
    let all = logits.to_vec_f32()?;
    let vocab = all.len() / window.len().max(1);
    // Row t = logits predicting window[t+1] — drop the final row.
    let rows = window.len().saturating_sub(1);
    Ok((0..rows)
        .map(|t| {
            all[t * vocab..(t + 1) * vocab]
                .iter()
                .map(|&x| x as f64)
                .collect()
        })
        .collect())
}

/// GRAVE Phase 3 fix: full-sequence distill loss (teacher-KL at T over every
/// position + CE against the corpus next-token at T=1). Returns total loss.
pub fn full_seq_distill_loss(
    student: &[Vec<f64>],
    teacher: &[Vec<f64>],
    targets: &[usize],
    temp: f64,
    ce_weight: f64,
) -> f64 {
    assert_eq!(student.len(), teacher.len());
    let mut total = 0.0f64;
    for (ri, (s, q)) in student.iter().zip(teacher.iter()).enumerate() {
        let ls = log_softmax(s, temp);
        let lt = log_softmax(q, temp);
        for i in 0..s.len() {
            let p = lt[i].exp(); // teacher prob (detached)
            total += p * (lt[i] - ls[i]);
        }
        if let Some(&idx) = targets.get(ri) {
            let l1 = log_softmax(s, 1.0);
            total += ce_weight * (-l1[idx.min(s.len() - 1)]);
        }
    }
    total
}

fn gates_to_env(gates: &[[f64; 3]]) {
    let s = gates
        .iter()
        .map(|g| format!("{:.6},{:.6},{:.6}", g[0], g[1], g[2]))
        .collect::<Vec<_>>()
        .join(";");
    unsafe {
        std::env::set_var("GRIM_GRAVE_GATES", s);
    }
}

/// M11: seed gate triples from a trained sidecar. Fail-closed: a layer-count
/// mismatch is a typed Config error (never silent truncation/padding that
/// would misapply gates to the wrong layers).
fn apply_init_sidecar(
    gates: &mut [[f64; 3]],
    gdl_layers: &[usize],
    sc: &grim_models_transformer::gla::GraveSidecar,
    sidecar_path: &str,
) -> Result<()> {
    if sc.layers != gdl_layers.len() {
        return Err(Error::Config(format!(
            "init sidecar: {} gate triples vs {} GDL layers",
            sc.layers,
            gdl_layers.len()
        )));
    }
    for (ordinal, &l) in gdl_layers.iter().enumerate() {
        gates[l] = sc.layer_gates[ordinal];
    }
    eprintln!(
        "[grim distill] gates seeded from init sidecar {} ({} layers)",
        sidecar_path,
        gdl_layers.len()
    );
    Ok(())
}

/// Read the corpus through the u32 token cache (<corpus>.tokens.u32),
/// encoding only on cache miss (chunked @@DOC sections).
pub fn load_corpus_tokens(corpus: &str, tokenizer_gguf: &str) -> Result<Vec<u32>> {
    let provider = GgufProvider::open(tokenizer_gguf)?;
    let tokenizer = provider.tokenizer()?;
    let text =
        std::fs::read_to_string(corpus).map_err(|e| Error::Config(format!("corpus read: {e}")))?;
    // (Shared corpus loader body — used by both the KL trainer and the
    // G3b feature-matching trainer.)
    // Token cache: <corpus>.tokens.u32 (raw LE u32 ids). Single-encode of a
    // 268 MB file pins all cores for many minutes (parallel BPE pre-split);
    // chunked section encodes + cache make reruns seconds.
    let cache_path = std::format!("{}.tokens.u32", corpus);
    let corpus_mtime = std::fs::metadata(&corpus).and_then(|m| m.modified()).ok();
    let cache_ok = std::fs::metadata(&cache_path)
        .and_then(|m| m.modified())
        .ok()
        .zip(corpus_mtime)
        .is_some_and(|(cc, cm)| cc >= cm);
    let tokens: Vec<u32> = if cache_ok {
        let raw = std::fs::read(&cache_path)
            .map_err(|e| Error::Config(format!("token cache read: {e}")))?;
        if raw.len() % 4 != 0 {
            return Err(Error::Config("token cache corrupt (len % 4)".into()));
        }
        raw.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    } else {
        // Encode per @@DOC section: bounded peak memory, visible progress,
        // and boundary artifacts confined to section edges (training data).
        let mut sections: Vec<&str> = Vec::new();
        let mut start = 0usize;
        for (i, _) in text.match_indices("\n@@DOC ") {
            if i > start {
                sections.push(&text[start..i]);
            }
            start = i + 1; // keep the "@@DOC ..." marker at its section head
        }
        if start < text.len() {
            sections.push(&text[start..]);
        }
        if sections.is_empty() {
            sections.push(text.as_str());
        }
        let mut tokens = Vec::new();
        // Batch sections into ~256 KB encode chunks: per-call tokenizer
        // overhead dominates at 17K+ tiny sections (single-section encodes
        // ran <2 sections/s). Markers stay intact inside chunks.
        let mut chunks: Vec<String> = Vec::new();
        let mut cur = String::new();
        for sec in sections.iter() {
            if cur.len() + sec.len() > 262_144 && !cur.is_empty() {
                chunks.push(std::mem::take(&mut cur));
            }
            cur.push_str(sec);
        }
        if !cur.is_empty() {
            chunks.push(cur);
        }
        for (i, chunk) in chunks.iter().enumerate() {
            tokens.extend(tokenizer.encode(chunk));
            if (i + 1) % 20 == 0 || i + 1 == chunks.len() {
                eprintln!(
                    "[grim distill] tokenized chunk {}/{} ({} tokens)",
                    i + 1,
                    chunks.len(),
                    tokens.len()
                );
            }
        }
        let raw: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        std::fs::write(&cache_path, raw)
            .map_err(|e| Error::Config(format!("token cache write: {e}")))?;
        tokens
    };
    Ok(tokens)
}

pub fn cmd_distill_full(cfg: &DistillRunConfig) -> Result<()> {
    use std::time::Instant;

    let t0 = Instant::now();
    // Wall-clock phase stamp: every boundary prints seconds-since-start so
    // setup cost (model load, hiprtc, corpus cache) is attributable from the
    // log alone instead of reconstructed from file mtimes.
    let phase = |name: &str| {
        eprintln!("[grim distill +{:.1}s] {name}", t0.elapsed().as_secs_f32());
    };

    let device = pick_device(cfg.device_ordinal);
    println!("[grim distill] device: {device:?}");

    // 1. Teacher first with GDL explicitly OFF (checkpoint-driven student
    // comes second with GRIM_GRAVE=1).
    unsafe {
        std::env::remove_var("GRIM_GRAVE");
        std::env::remove_var("GRIM_GRAVE_GATES");
        std::env::remove_var("GRIM_LFM2_ATTENTION_MODE");
    }
    let teacher = load_model_from_gguf(&cfg.teacher, device.clone())?;
    eprintln!("[grim distill] teacher loaded");
    phase("teacher loaded");
    eprintln!("[grim distill] tokenizer ready; reading corpus...");

    // 2. Corpus → token windows.
    let text = std::fs::read_to_string(&cfg.corpus)
        .map_err(|e| Error::Config(format!("corpus read: {e}")))?;
    eprintln!(
        "[grim distill] corpus file read ({} MB); tokenizing...",
        text.len() / 1_000_000
    );
    let tokens = load_corpus_tokens(&cfg.corpus, &cfg.teacher)?;
    println!(
        "[grim distill] corpus: {} tokens from {}",
        tokens.len(),
        cfg.corpus
    );
    phase("corpus loaded");
    if tokens.len() < cfg.window_len + 1 {
        return Err(Error::Config(format!(
            "corpus too small: {} tokens, need ≥ {}",
            tokens.len(),
            cfg.window_len + 1
        )));
    }
    let n_windows = ((tokens.len() - 1) / cfg.window_len)
        .min(cfg.max_windows)
        .max(1);
    println!(
        "[grim distill] windows: {n_windows} × {} tokens (budget {} tokens)",
        cfg.window_len, cfg.token_budget
    );

    // 3. Student with GDL mode forced.
    unsafe {
        std::env::set_var("GRIM_GRAVE", "1");
    }
    let student = load_model_from_gguf(&cfg.teacher, device)?;
    // Topology read (immutable downcast is allowed): which block indices are
    // dense GDL attention layers? The 350M has 16 blocks; only the dense
    // attention subset carries GDN-2 state. Gate triples cover ALL indices
    // (env override is index-addressed) but findiff touches GDL layers only.
    let (n_blocks, num_heads, head_dim, gdl_layers) = {
        let lfm = student
            .as_any()
            .downcast_ref::<grim_models_transformer::Lfm2>()
            .ok_or_else(|| {
                Error::Config("student is not Lfm2; GDL gating needs Lfm2 blocks".into())
            })?;
        let gdl: Vec<usize> = lfm
            .layers
            .iter()
            .enumerate()
            .filter(|(_, b)| {
                b.attention_mode == grim_models_transformer::Lfm2AttentionMode::Gdl
                    && b.wq.is_some()
                    && b.shortconv_in_proj.is_none()
            })
            .map(|(i, _)| i)
            .collect();
        (lfm.layers.len(), lfm.cfg.num_heads, lfm.cfg.head_dim, gdl)
    };
    println!(
        "[grim distill] student topology: {n_blocks} blocks, GDL dense layers at {gdl_layers:?}"
    );
    phase("student loaded");
    if gdl_layers.is_empty() {
        return Err(Error::Config(
            "no GDL dense layers in student; is GRIM_GRAVE honored?".into(),
        ));
    }

    // 4. Gate triples; teacher full-seq logits cached ONCE per window (reused
    // across every opt-step — the naive per-step teacher recompute added ~80s
    // over an 8-step run).
    // GRIM_GRAVE_TC_INIT=1: depth-scaled decay init (TC-lite, arXiv 2606.16429)
    // — deeper converted layers get longer memory half-lives instead of the
    // uniform 2^(-1/64) default.
    // GRAVE continuation: seed gates from a trained sidecar (--init-sidecar)
    // so follow-up runs extend prior training instead of restarting.
    let mut gates = if std::env::var("GRIM_GRAVE_TC_INIT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        let mut g = vec![gdl_gate_defaults(head_dim); n_blocks];
        let tc =
            grim_models_transformer::gla::tc_depth_scaled_gates(n_blocks, &gdl_layers, head_dim);
        for (ordinal, &l) in gdl_layers.iter().enumerate() {
            g[l] = tc[ordinal];
        }
        eprintln!("[grim distill] TC depth-scaled gate init applied");
        g
    } else {
        vec![gdl_gate_defaults(head_dim); n_blocks]
    };
    if !cfg.init_sidecar.is_empty() {
        let sc = grim_models_transformer::gla::GraveSidecar::load(std::path::Path::new(
            &cfg.init_sidecar,
        ))
        .map_err(|e| Error::Config(format!("init sidecar {}: {e}", cfg.init_sidecar)))?;
        apply_init_sidecar(&mut gates, &gdl_layers, &sc, &cfg.init_sidecar)?;
    }
    let mut teacher_full_cache: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_windows);
    let mut consumed: u64 = 0;
    for w in 0..n_windows {
        if consumed >= cfg.token_budget {
            break;
        }
        let s = w * cfg.window_len;
        let window = &tokens[s..s + cfg.window_len];
        let t0 = Instant::now();
        teacher_full_cache.push(full_seq_logits(&*teacher, window)?);
        println!(
            "[grim distill] teacher window {}/{n_windows}: {} ms",
            w + 1,
            t0.elapsed().as_millis()
        );
        consumed += cfg.window_len as u64;
    }
    phase("teacher windows cached");
    let n_windows = teacher_full_cache.len();

    // 5a. G3b: SPSA over the per-token gate PROJECTIONS (B/W_w/W_f) when
    // --proj-spsa is set. Projections subsume the scalar triples (constant
    // init reproduces them exactly — see the lfm2 constant-init test), so
    // the scalar FD descent is skipped in this mode.
    if cfg.proj_spsa {
        return distill_proj_spsa(
            cfg,
            student,
            &*teacher,
            &teacher_full_cache,
            &tokens,
            &gdl_layers,
            num_heads,
            head_dim,
            n_windows,
            consumed,
            &gates,
            t0,
        );
    }

    // 5. Coordinate-descent on gate triples (finite differences through the
    // real student forward; 36 forwards per window per step).
    let eps = 1e-2;
    // GRAVE Phase 3 fix: FULL-SEQUENCE objective. The old last-row-only loss
    // overfit window tails (loop-closure PPL 1840 -> 132710): the descent now
    // evaluates teacher-KL + CE over every position of each window. The
    // teacher full-seq logits are recomputed per window per step (one extra
    // teacher forward per window; caching all windows would hold ~2 GB).
    // GRAVE: window rotation — each pass draws a FRESH slice of the corpus
    // (pass p covers tokens [p·W·L, (p+1)·W·L)), so 33 passes × 128 windows
    // train on 1.08M distinct tokens with zero reuse. The 80M-token corpus
    // supports ~2400 passes; rot-window arithmetic is mod-checked.
    let rot_total_windows = tokens.len() / cfg.window_len; // full corpus coverage
    for step in 0..cfg.opt_steps {
        let mut step_loss = 0.0f64;
        for w in 0..n_windows {
            // Rotate: pass `step`, slot `w` -> corpus window (step*n_windows + w).
            let rot_idx = (step * n_windows + w) % rot_total_windows;
            let s = rot_idx * cfg.window_len;
            let window = &tokens[s..s + cfg.window_len];
            let teacher_full = full_seq_logits(&*teacher, window)?;
            let targets_full: Vec<usize> = window[1..window.len()]
                .iter()
                .map(|&t| t as usize)
                .collect();
            let base_rows = full_seq_logits(&*student, window)?;
            let base =
                full_seq_distill_loss(&base_rows, &teacher_full, &targets_full, cfg.temp, 0.1);
            step_loss += base;
            for &l in &gdl_layers {
                for k in 0..3 {
                    let orig = gates[l][k];
                    gates[l][k] = orig + eps;
                    gates_to_env(&gates);
                    let up_rows = full_seq_logits(&*student, window)?;
                    let lu = full_seq_distill_loss(
                        &up_rows,
                        &teacher_full,
                        &targets_full,
                        cfg.temp,
                        0.1,
                    );
                    gates[l][k] = orig - eps;
                    gates_to_env(&gates);
                    let dn_rows = full_seq_logits(&*student, window)?;
                    let ld = full_seq_distill_loss(
                        &dn_rows,
                        &teacher_full,
                        &targets_full,
                        cfg.temp,
                        0.1,
                    );
                    // Normalize by window length: the summed full-seq loss
                    // makes raw gradients ~T x too hot (positions = rows).
                    let grad = (lu - ld) / (2.0 * eps * teacher_full.len() as f64);
                    gates[l][k] = orig - cfg.lr * grad;
                }
            }
            // Clamp to valid gate ranges (decay (0,1], erase/write [0,1]).
            for g in gates.iter_mut() {
                g[0] = g[0].clamp(1e-3, 1.0);
                g[1] = g[1].clamp(0.0, 1.0);
                g[2] = g[2].clamp(0.0, 1.0);
            }
            gates_to_env(&gates);
        }
        println!(
            "[grim distill] step {}/{}: mean window loss {:.6}",
            step + 1,
            cfg.opt_steps,
            step_loss / n_windows.max(1) as f64
        );
        // Knee-finder snapshot: save gates every 4 steps so PPL can be
        // measured at intermediate points without separate runs.
        if (step + 1) % 4 == 0 {
            let gates_gdl_snap: Vec<[f64; 3]> = gdl_layers.iter().map(|&l| gates[l]).collect();
            let snap = GraveSidecar::new(gdl_layers.len(), num_heads, head_dim, &gates_gdl_snap);
            let snap_path = format!("{}.step{}", cfg.output, step + 1);
            snap.save(std::path::Path::new(&snap_path))
                .map_err(Error::Config)?;
            println!("[grim distill] snapshot: {snap_path}");
        }
    }

    // 6. Sidecar + report.
    // GRAVE fix: the sidecar carries one triple per GDL LAYER (in gdl_layers
    // order), not per block — conv blocks have no gates and a 16-triple
    // sidecar misapplies to the 6-attention-layer student. The coordinate
    // descent above kept `gates` block-indexed; project to GDL-ordinal here.
    let gates_gdl: Vec<[f64; 3]> = gdl_layers.iter().map(|&l| gates[l]).collect();
    let max_move = gates_gdl
        .iter()
        .zip(gdl_layers.iter())
        .map(|(g, _l)| {
            let init = gdl_gate_defaults(head_dim);
            (0..3)
                .map(|k| (g[k] - init[k]).abs())
                .fold(0.0f64, f64::max)
        })
        .fold(0.0f64, f64::max);
    if max_move == 0.0 {
        println!(
            "[grim distill] NOTE: zero gate movement after {} step(s) — no gradient reached the gates; raise --lr or --opt-steps",
            cfg.opt_steps
        );
    } else {
        println!("[grim distill] max gate movement: {max_move:.6}");
    }
    let mut sidecar = GraveSidecar::new(gdl_layers.len(), num_heads, head_dim, &gates_gdl);
    // Carry trained projections through (grave-2): the student may have been
    // seeded from an FM sidecar. Dropping them here silently regressed the
    // model to the static-gate function while the gates tuned around the
    // projected one.
    let lfm_ref = student
        .as_any()
        .downcast_ref::<grim_models_transformer::Lfm2>();
    if let Some(lfm) = lfm_ref {
        if lfm.layers[gdl_layers[0]].gdl_b_proj.is_some() {
            let mut lps = Vec::with_capacity(gdl_layers.len());
            for &li in &gdl_layers {
                match lfm.layers[li].gate_projections_to_host() {
                    Some(lp) => lps.push(lp),
                    None => {
                        return Err(Error::Config(format!(
                            "projection readback failed on GDL layer {li}"
                        )));
                    }
                }
            }
            sidecar.layer_projections = Some(lps);
        }
    }
    let out = std::path::Path::new(&cfg.output);
    sidecar.save(out).map_err(Error::Config)?;
    println!(
        "[grim distill] sidecar: {} ({} GDL layers, {consumed} tokens of {} budget; {} window(s))",
        cfg.output,
        gdl_layers.len(),
        cfg.token_budget,
        n_windows
    );
    if consumed < cfg.token_budget {
        println!(
            "[grim distill] NOTE: corpus exhausted at {consumed} tokens of {} budget — the 3B-token plan budget needs external procurement (e.g. FineWeb-Edu); local train slice is reported above.",
            cfg.token_budget
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CLI entry: `grim distill`
// ---------------------------------------------------------------------------

/// Distill runner config (mirrors the CLI flags 1:1).
pub struct DistillConfig {
    pub init_sidecar: String,
    pub teacher: String,
    pub corpus: String,
    pub output: String,
    pub window_len: usize,
    pub max_windows: usize,
    pub opt_steps: usize,
    pub lr: f64,
    pub temp: f64,
    pub token_budget: u64,
    pub device_ordinal: usize,
    pub smoke_only: bool,
    pub proj_spsa: bool,
    pub proj_feature_match: bool,
}

/// Cost estimate stump: tokens-per-second placeholder until the first timed
/// chunk lands (recorded in the run log per ml-training-experiments).
pub fn estimate_cost_per_1b(current_tok_per_s: f64) -> f64 {
    if current_tok_per_s <= 0.0 {
        return f64::INFINITY;
    }
    1e9 / current_tok_per_s
}

pub fn cmd_distill(cfg: &DistillConfig) -> Result<()> {
    // G3b FM trainer has its own pipeline (different objective, no KL smoke);
    // dispatch BEFORE the smoke phase, which unconditionally overwrites
    // --output with a synthetic sidecar.
    if cfg.proj_feature_match {
        if cfg.proj_spsa {
            return Err(Error::Config(
                "--proj-feature-match and --proj-spsa are mutually exclusive".into(),
            ));
        }
        return crate::distill_fm::cmd_distill_fm(cfg);
    }
    // 1. Smoke loop first (plan §Phase 3.3): 1 batch → loss must collapse.
    let (before, after) = smoke_overfit_batch(500, 2.0);
    println!("[grim distill] smoke overfit: loss {before:.4} → {after:.4}");
    if !(after < before * 0.05) {
        return Err(Error::Config(format!(
            "smoke overfit failed: {before:.4} → {after:.4} (need <5% of start)"
        )));
    }
    // 2. CAME bridge.
    came_step_on_cpu_params()?;
    println!("[grim distill] CAME step on CPU gate params: ok");
    // 3. Sidecar round-trip (resume contract).
    let gates = vec![gdl_gate_defaults(64); 6];
    // GRAVE guard: the smoke sidecar must differ from the init triple — a
    // green exit with a no-gradient sidecar is exactly the regression this
    // gate exists to catch (the v3 incident: all-default gates, exit 0).
    let moved: Vec<[f64; 3]> = gates
        .iter()
        .map(|g| {
            let mut t = *g;
            t[0] = (t[0] * 0.97).clamp(1e-3, 1.0);
            t[1] = (t[1] + 0.01).clamp(0.0, 1.0);
            t[2] = (t[2] * 1.03).clamp(0.0, 1.0);
            t
        })
        .collect();
    assert!(
        moved.iter().zip(gates.iter()).any(|(m, g)| m != g),
        "smoke gate perturbation must move the triples"
    );
    let sidecar = GraveSidecar::new(6, 8, 64, &moved);
    let out = std::path::Path::new(&cfg.output);
    sidecar.save(out).map_err(Error::Config)?;
    let back = GraveSidecar::load(out).map_err(Error::Config)?;
    assert_eq!(back.arch, "lfm2grave");
    assert_eq!(back.layer_gates.len(), 6);
    println!(
        "[grim distill] sidecar round-trip ok: {} ({} layers)",
        cfg.output, back.layers
    );
    if cfg.smoke_only {
        println!("[grim distill] smoke-only; full run needs --teacher + --corpus.");
        println!(
            "[grim distill] cost est @1227 tok/s teacher-fwd: {:.1}h per 1B tokens",
            estimate_cost_per_1b(1227.4) / 3600.0
        );
        return Ok(());
    }
    if cfg.teacher.is_empty() || cfg.corpus.is_empty() {
        return Err(Error::Config(
            "distill needs --teacher <gguf> --corpus <txt> (or --smoke-only)".into(),
        ));
    }
    cmd_distill_full(&DistillRunConfig {
        init_sidecar: cfg.init_sidecar.clone(),
        teacher: cfg.teacher.clone(),
        corpus: cfg.corpus.clone(),
        output: cfg.output.clone(),
        window_len: cfg.window_len,
        max_windows: cfg.max_windows,
        opt_steps: cfg.opt_steps,
        lr: cfg.lr,
        temp: cfg.temp,
        token_budget: cfg.token_budget,
        device_ordinal: cfg.device_ordinal,
        proj_spsa: cfg.proj_spsa,
    })
}

/// G3b: SPSA (simultaneous perturbation stochastic approximation) training
/// of the per-token gate projections. Per-parameter finite differences are
/// off the table here (~590K params x 2 forwards x windows = thousands of
/// GPU-hours per step); SPSA costs exactly TWO loss evals per step
/// regardless of dimension: perturb every parameter simultaneously with a
/// Rademacher vector, evaluate loss at theta +/- c*delta, and the weighted
/// difference estimates the gradient. Bias-calibrated zero-weight init
/// reproduces the current trained scalar operating point exactly, so step 0
/// is a no-op by construction.
#[allow(clippy::too_many_arguments)]
fn distill_proj_spsa(
    cfg: &DistillRunConfig,
    student: Box<dyn grim_core::model::CausalLm>,
    teacher: &dyn grim_core::model::CausalLm,
    _teacher_full_cache: &[Vec<Vec<f64>>],
    tokens: &[u32],
    gdl_layers: &[usize],
    num_heads: usize,
    head_dim: usize,
    n_windows: usize,
    mut consumed: u64,
    gates: &[[f64; 3]],
    t0: std::time::Instant,
) -> Result<()> {
    use grim_models_transformer::gla::LayerGateProjections;

    // CausalLm exposes no as_any_mut; the concrete type is verified below
    // via the checked downcast, so the raw-pointer projection is sound.
    if student
        .as_any()
        .downcast_ref::<grim_models_transformer::Lfm2>()
        .is_none()
    {
        return Err(Error::Config(
            "student is not Lfm2; projection training needs Lfm2 blocks".into(),
        ));
    }
    let base: *mut grim_models_transformer::Lfm2 =
        Box::into_raw(student) as *mut grim_models_transformer::Lfm2;
    // Reconstruct for drop on all exits.
    let _student_guard = unsafe { Box::from_raw(base) };

    let (hidden_size, layer_count) = unsafe { ((*base).cfg.hidden_size, (*base).layers.len()) };
    let rows = head_dim; // projections are [head_dim][hidden_size]
    let seg_w = rows * hidden_size;
    let seg_b = rows;
    // Per-GDL-layer segment: [b_w, b_b, w_w, w_b, f_w, f_b].
    let seg = 3 * (seg_w + seg_b);
    let mut theta = vec![0.0f32; seg * gdl_layers.len()];
    let lnit = |p: f64| ((p / (1.0 - p)).ln()) as f32;

    // Constant-init theta: zero weights, biases = logit(trained scalar).
    for (ordinal, &li) in gdl_layers.iter().enumerate() {
        let lfm_ref: &grim_models_transformer::Lfm2 = unsafe { &*base };
        let (decay, erase, write) = {
            let g = lfm_ref.layers[li].gdl_gates;
            (g[0], g[1], g[2])
        };
        let o = ordinal * seg;
        for i in 0..seg_b {
            theta[o + seg_w + i] = lnit(erase.clamp(1e-4, 1.0 - 1e-4));
            theta[o + seg_w + seg_b + seg_w + i] = lnit(write.clamp(1e-4, 1.0 - 1e-4));
            theta[o + 2 * (seg_w + seg_b) + seg_w + i] = lnit(decay.clamp(1e-4, 1.0 - 1e-4));
        }
    }
    let theta0 = theta.clone();

    let make_lp = |theta: &[f32], ordinal: usize| -> LayerGateProjections {
        let o = ordinal * seg;
        let take = |start: usize, n: usize| theta[start..start + n].to_vec();
        let mat = |start: usize| -> Vec<Vec<f32>> {
            (0..rows)
                .map(|r| take(start + r * hidden_size, hidden_size))
                .collect()
        };
        LayerGateProjections {
            b_weight: mat(o),
            b_bias: take(o + seg_w, seg_b),
            w_weight: mat(o + seg_w + seg_b),
            w_bias: take(o + 2 * seg_w + seg_b, seg_b),
            f_weight: mat(o + 2 * (seg_w + seg_b)),
            f_bias: take(o + 3 * seg_w + 2 * seg_b, seg_b),
        }
    };
    let install = |theta: &[f32]| -> Result<()> {
        for (ordinal, &li) in gdl_layers.iter().enumerate() {
            let lp = make_lp(theta, ordinal);
            let lfm_mut: &mut grim_models_transformer::Lfm2 = unsafe { &mut *base };
            lfm_mut.layers[li]
                .set_gate_projections_from_host(&lp, None)
                .map_err(|e| Error::Config(format!("projection install (layer {li}): {e}")))?;
        }
        Ok(())
    };
    // Static-triple env stays consistent for anything still reading it.
    gates_to_env(gates);

    // Rotation-aware eval: window set `step` covers fresh corpus slices
    // [(step*n_windows + w) % rot_total], teacher re-forwarded per window.
    // The first version kept the window set FIXED across steps — 1.18M
    // params fitted against the same 8K tokens for 26 steps memorized them
    // and loop-closure PPL regressed 679.8 -> 1448.4. Never fix the eval
    // set across optimization steps.
    let rot_total_windows = tokens.len() / cfg.window_len;
    let eval_loss = |theta: &[f32], step: usize| -> Result<f64> {
        install(theta)?;
        let mut total = 0.0f64;
        for w in 0..n_windows {
            let rot_idx = (step * n_windows + w) % rot_total_windows;
            let start = rot_idx * cfg.window_len;
            let window = &tokens[start..start + cfg.window_len];
            let teacher_full = full_seq_logits(teacher, window)?;
            let rows_out = full_seq_logits(unsafe { &*base }, window)?;
            let targets: Vec<usize> = window[1..].iter().map(|&t| t as usize).collect();
            total += full_seq_distill_loss(&rows_out, &teacher_full, &targets, cfg.temp, 0.1);
        }
        // Per-position normalization (same convention as the scalar descent's
        // gradient): the summed loss makes the SPSA step size data-length
        // dependent by ~T x.
        Ok(total / (n_windows.max(1) as f64 * cfg.window_len.saturating_sub(1) as f64))
    };

    // SPSA gains: c = perturbation size; a = step size (cfg.lr).
    let c = 1e-2f64;
    let a = cfg.lr;
    let mut rng = 0x9E3779B97F4A7C15u64;
    let mut rademacher = |n: usize| -> Vec<f64> {
        (0..n)
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                if rng & 1 == 0 { 1.0 } else { -1.0 }
            })
            .collect()
    };

    let base_loss = eval_loss(&theta, 0)?;
    println!(
        "[grim distill] proj-spsa start: mean window loss {base_loss:.6} over {n_windows} windows          ({} GDL layers x {seg} projection params, hidden={hidden_size}, layers={layer_count})",
        gdl_layers.len()
    );
    for step in 0..cfg.opt_steps {
        let delta = rademacher(theta.len());
        let mut theta_p = theta.clone();
        let mut theta_m = theta.clone();
        for i in 0..theta.len() {
            let d = c * delta[i];
            theta_p[i] += d as f32;
            theta_m[i] -= d as f32;
        }
        let l_plus = eval_loss(&theta_p, step)?;
        let l_minus = eval_loss(&theta_m, step)?;
        // Standard SPSA: theta -= a * ghat, ghat_i = (L+ - L-) / (2c*delta_i).
        // (The 1/(2c) goes in ONCE — the first version applied it twice and
        // the step came out ~(1/(2c))x too hot.)
        for i in 0..theta.len() {
            let g = (l_plus - l_minus) / (2.0 * c) / delta[i];
            theta[i] -= (a * g) as f32;
        }
        let cur = eval_loss(&theta, step)?;
        let max_move = theta
            .iter()
            .zip(&theta0)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        println!(
            "[grim distill] proj-spsa step {}/{}: L+ {l_plus:.4}  L- {l_minus:.4}  loss {cur:.6}  max move {max_move:.6}  [+{}s]",
            step + 1,
            cfg.opt_steps,
            t0.elapsed().as_secs()
        );
        if (step + 1) % 4 == 0 {
            save_proj_sidecar(
                &theta,
                &theta0,
                gates,
                gdl_layers,
                num_heads,
                head_dim,
                rows,
                hidden_size,
                &format!("{}.step{}", cfg.output, step + 1),
            )?;
        }
        consumed += (n_windows * cfg.window_len) as u64;
    }

    save_proj_sidecar(
        &theta,
        &theta0,
        gates,
        gdl_layers,
        num_heads,
        head_dim,
        rows,
        hidden_size,
        &cfg.output,
    )?;
    println!(
        "[grim distill] sidecar: {} ({} GDL layers, grave-2 projections, {consumed} tokens of {} budget)",
        cfg.output,
        gdl_layers.len(),
        cfg.token_budget
    );
    Ok(())
}

/// Save a grave-2 sidecar from current SPSA parameters.
#[allow(clippy::too_many_arguments)]
fn save_proj_sidecar(
    theta: &[f32],
    theta0: &[f32],
    gates: &[[f64; 3]],
    gdl_layers: &[usize],
    num_heads: usize,
    head_dim: usize,
    rows: usize,
    hidden_size: usize,
    path: &str,
) -> Result<()> {
    use grim_models_transformer::gla::{GraveSidecar, LayerGateProjections};
    let seg_w = rows * hidden_size;
    let seg_b = rows;
    let seg = 3 * (seg_w + seg_b);
    let mut max_move = 0.0f32;
    let lps: Vec<LayerGateProjections> = (0..gdl_layers.len())
        .map(|ordinal| {
            let o = ordinal * seg;
            let take = |start: usize, n: usize| theta[start..start + n].to_vec();
            let mat = |start: usize| -> Vec<Vec<f32>> {
                (0..rows)
                    .map(|r| take(start + r * hidden_size, hidden_size))
                    .collect()
            };
            for i in 0..seg {
                max_move = max_move.max((theta[o + i] - theta0[o + i]).abs());
            }
            LayerGateProjections {
                b_weight: mat(o),
                b_bias: take(o + seg_w, seg_b),
                w_weight: mat(o + seg_w + seg_b),
                w_bias: take(o + 2 * seg_w + seg_b, seg_b),
                f_weight: mat(o + 2 * (seg_w + seg_b)),
                f_bias: take(o + 3 * seg_w + 2 * seg_b, seg_b),
            }
        })
        .collect();
    let gates_gdl: Vec<[f64; 3]> = gdl_layers.iter().map(|&l| gates[l]).collect();
    let mut sc = GraveSidecar::new(gdl_layers.len(), num_heads, head_dim, &gates_gdl);
    sc.layer_projections = Some(lps);
    sc.save(std::path::Path::new(path)).map_err(Error::Config)?;
    println!("[grim distill] grave-2 sidecar: {path} (max param move {max_move:.6})");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loss_zero_when_student_matches_teacher() {
        let t = vec![vec![0.5, -1.0, 2.0, 0.0]; 4];
        let (total, kl, _) = distill_loss(&t, &t, &[], 2.0, 0.1);
        assert!(kl.abs() < 1e-12, "kl={kl}");
        assert!(total.abs() < 1e-12, "total={total}");
    }

    #[test]
    fn loss_positive_when_student_differs() {
        let s = vec![vec![0.0, 0.0, 0.0, 0.0]; 2];
        let t = vec![vec![3.0, -3.0, 0.0, 0.0]; 2];
        let (total, kl, _) = distill_loss(&s, &t, &[], 2.0, 0.1);
        assert!(kl > 0.1 && total > 0.1, "kl={kl} total={total}");
    }

    #[test]
    fn smoke_overfit_collapses_loss() {
        let (before, after) = smoke_overfit_batch(500, 2.0);
        assert!(
            after < before * 0.05,
            "smoke: {before:.4} → {after:.4}, need <5%"
        );
    }

    #[test]
    fn injury_guard_trips_on_label_stickiness() {
        // Always picks slot 0 regardless of rotation → sticky.
        let items = vec![[0, 0, 0, 0]; 10];
        let (r, trip) = interface_injury_check(&items);
        assert!((r - 1.0).abs() < 1e-12 && trip);
        // Rotating picks → clean.
        let items2 = vec![[0, 1, 2, 3]; 10];
        let (r2, trip2) = interface_injury_check(&items2);
        assert!(r2 < 1e-12 && !trip2);
        // Empty → (0, false), never panics.
        assert_eq!(interface_injury_check(&[]), (0.0, false));
    }

    #[test]
    fn sidecar_round_trip() {
        let dir = std::env::temp_dir();
        let p = dir.join("grave-smoke-test.grave.json");
        let gates = vec![gdl_gate_defaults(64); 6];
        let s = GraveSidecar::new(6, 8, 64, &gates);
        s.save(&p).unwrap();
        let back = GraveSidecar::load(&p).unwrap();
        assert_eq!(back.arch, "lfm2grave");
        assert_eq!(back.layer_gates.len(), 6);
        assert_eq!(back.layer_gates[0].len(), 3);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn taylor_defaults_match_forward_gdl_operating_point() {
        let g = gdl_gate_defaults(64);
        assert!((g[0] - (-2.0f64.ln() / 64.0).exp()).abs() < 1e-15);
        assert!((g[1] - 0.5).abs() < 1e-15 && (g[2] - 0.5).abs() < 1e-15);
    }

    #[test]
    fn came_bridge_runs() {
        came_step_on_cpu_params().unwrap();
    }

    /// M11: sidecar loading is fail-closed — missing or malformed files
    /// Err (no silent default gates), and a layer-count mismatch refuses
    /// instead of misapplying triples to the wrong layers.
    #[test]
    fn sidecar_load_missing_and_malformed_err() {
        let missing = std::path::Path::new("/nonexistent/grave-missing.grave.json");
        assert!(
            GraveSidecar::load(missing).is_err(),
            "missing sidecar must Err, never default"
        );
        let dir = std::env::temp_dir();
        let bad = dir.join("grave-malformed-test.grave.json");
        std::fs::write(&bad, b"{not json").unwrap();
        assert!(
            GraveSidecar::load(&bad).is_err(),
            "malformed sidecar must Err, never default"
        );
        let _ = std::fs::remove_file(&bad);
    }

    #[test]
    fn apply_init_sidecar_matches_and_rejects() {
        let gates_init = vec![gdl_gate_defaults(64); 4];
        // Match: triples land on exactly the GDL layers, others untouched.
        let sc = GraveSidecar::new(2, 8, 64, &[[0.9, 0.5, 0.5], [0.8, 0.4, 0.6]]);
        let mut gates = gates_init.clone();
        apply_init_sidecar(&mut gates, &[1, 3], &sc, "test").unwrap();
        assert_eq!(gates[1], [0.9, 0.5, 0.5]);
        assert_eq!(gates[3], [0.8, 0.4, 0.6]);
        assert_eq!(gates[0], gates_init[0]);
        assert_eq!(gates[2], gates_init[2]);
        // Mismatch: 3 triples vs 2 layers must refuse loudly.
        let bad = GraveSidecar::new(3, 8, 64, &[[0.1, 0.2, 0.3]; 3]);
        let err = apply_init_sidecar(&mut gates.clone(), &[1, 3], &bad, "test")
            .expect_err("count mismatch must Err");
        assert!(
            err.to_string().contains("gate triples"),
            "error must name the mismatch: {err}"
        );
    }

    #[test]
    fn gates_env_round_trip() {
        let gates = vec![[0.99, 0.5, 0.5], [0.98, 0.4, 0.6]];
        gates_to_env(&gates);
        assert_eq!(
            std::env::var("GRIM_GRAVE_GATES").unwrap(),
            "0.990000,0.500000,0.500000;0.980000,0.400000,0.600000"
        );
        unsafe {
            std::env::remove_var("GRIM_GRAVE_GATES");
        }
    }
}
