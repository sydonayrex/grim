//! Decisive probe for the 27B CLI generation garbage: at the MODEL level,
//! does the cached incremental-decode path produce the same logits as a
//! full-sequence forward over the identical ids?
//!
//! Path A: forward([prompt..20, gen]) as ONE [1,21] input → last-row logits.
//! Path B: forward([prompt..20]) → cache; forward([gen], seq_len=1) → logits.
//! Path A == the eval/teacher-forced route (proven good, PPL 6.12).
//! Path B == the CLI decode route (produces garbage free-running).
//! If A != B the cache/incremental path is broken; per-layer GRIM_I8_PROBE_DIR
//! dumps localize the layer.

use grim_tensor::{Device, Shape};

const MODEL: &str = "/drive/bigfast/grim/models/qwen38-27b/Qwen3.8-27B-Q4_K_M.gguf";
const VOCAB: usize = 248320;

#[test]
#[ignore = "needs the 27B model + ROCm device; run with GRIM_GPU_TEST=1"]
fn cli_decode_step_matches_full_sequence() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if !std::path::Path::new(MODEL).exists() {
        eprintln!("skipping: no model");
        return Ok(());
    }
    let device = Device::Rocm(0);
    let model = grim_engine::model_loader::load_model_from_gguf(MODEL, device.clone())
        .map_err(|e| format!("load: {e}"))?;
    // The exact CLI prompt ids (grim template, "Explain how a transistor...")
    let prompt: Vec<u32> = vec![
        248045, 1156, 198, 814, 20139, 1204, 264, 94562, 4138, 303, 1330, 41228,
        25, 248046, 198, 248045, 17313, 198, 248068, 198,
    ];
    let mk = |vals: &[f32], len: usize| -> grim_tensor::Tensor {
        grim_backend_cpu::cpu_tensor(vals.to_vec(), Shape::new(vec![1, len]))
    };

    // Prefill the prompt once to get the first generated token id (greedy).
    let mut sess = model.new_session();
    let pos: Vec<f32> = (0..prompt.len()).map(|i| i as f32).collect();
    let input = mk(&prompt.iter().map(|&v| v as f32).collect::<Vec<_>>(), prompt.len());
    let pos_t = mk(&pos, pos.len());
    let logits = model
        .forward(sess.as_mut(), &input, &pos_t, &[])
        .map_err(|e| format!("prefill: {e}"))?;
    let flat = logits.to_vec_f32().map_err(|e| format!("prefill read: {e}"))?;
    let row = &flat[flat.len() - VOCAB..];
    let mut idx: Vec<u32> = (0..VOCAB as u32).collect();
    idx.sort_by(|&a, &b| row[b as usize].total_cmp(&row[a as usize]));
    let gen0 = idx[0];
    eprintln!("[parity] prefill top1 = {gen0} (logit {:.4})", row[gen0 as usize]);
    eprintln!("[parity] prefill top5 = {:?}", &idx[..5]);

    // ── Path A: full-sequence forward over prompt + gen0 ──────────────────
    let mut full = prompt.clone();
    full.push(gen0);
    let mut sess_a = model.new_session();
    let pos_a: Vec<f32> = (0..full.len()).map(|i| i as f32).collect();
    let in_a = mk(&full.iter().map(|&v| v as f32).collect::<Vec<_>>(), full.len());
    let pos_at = mk(&pos_a, pos_a.len());
    let la = model
        .forward(sess_a.as_mut(), &in_a, &pos_at, &[])
        .map_err(|e| format!("full-seq: {e}"))?;
    let fa = la.to_vec_f32().map_err(|e| format!("full-seq read: {e}"))?;
    let ra = &fa[fa.len() - VOCAB..];
    let mut idx_a: Vec<u32> = (0..VOCAB as u32).collect();
    idx_a.sort_by(|&a, &b| ra[b as usize].total_cmp(&ra[a as usize]));
    eprintln!("[parity] path A (full-seq) top1 = {} (logit {:.4})", idx_a[0], ra[idx_a[0] as usize]);
    eprintln!("[parity] path A top5 = {:?}", &idx_a[..5]);

    // ── Path B: cached incremental decode ─────────────────────────────────
    let mut sess_b = model.new_session();
    let pos_b0: Vec<f32> = (0..prompt.len()).map(|i| i as f32).collect();
    let in_b0 = mk(&prompt.iter().map(|&v| v as f32).collect::<Vec<_>>(), prompt.len());
    let pos_b0t = mk(&pos_b0, pos_b0.len());
    let _ = model
        .forward(sess_b.as_mut(), &in_b0, &pos_b0t, &[])
        .map_err(|e| format!("path B prefill: {e}"))?;
    let gen_t = mk(&[gen0 as f32], 1);
    let pos_b1 = mk(&[prompt.len() as f32], 1);
    let lb = model
        .forward(sess_b.as_mut(), &gen_t, &pos_b1, &[])
        .map_err(|e| format!("path B decode: {e}"))?;
    let fb = lb.to_vec_f32().map_err(|e| format!("path B read: {e}"))?;
    let rb = &fb[fb.len() - VOCAB..];
    let mut idx_b: Vec<u32> = (0..VOCAB as u32).collect();
    idx_b.sort_by(|&a, &b| rb[b as usize].total_cmp(&rb[a as usize]));
    eprintln!("[parity] path B (cached) top1 = {} (logit {:.4})", idx_b[0], rb[idx_b[0] as usize]);
    eprintln!("[parity] path B top5 = {:?}", &idx_b[..5]);

    // Logit diff
    let mut max_diff = 0.0f32;
    let mut at = 0usize;
    for (i, (a, b)) in ra.iter().zip(rb.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_diff {
            max_diff = d;
            at = i;
        }
    }
    eprintln!(
        "[parity] path A vs B: max logit diff {max_diff:.4e} at token {at} (A {:.4} B {:.4}); top1 A={} B={}",
        ra[at], rb[at], idx_a[0], idx_b[0]
    );
    if idx_a[0] != idx_b[0] {
        eprintln!(
            "[parity] *** DEFECT CONFIRMED: full-seq and cached decode disagree on the next token ***"
        );
    } else if max_diff > 1e-2 {
        eprintln!("[parity] *** WARNING: same top1 but logits diverge {max_diff:.4e} ***");
    }
    Ok(())
}
