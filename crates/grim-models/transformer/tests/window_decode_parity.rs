//! Verify-window parity probe (MTP gate (b) support).
//!
//! The MTP accept path commits state from a VERIFY forward over a 2-token
//! window `[A, B]`; plain decode processes the same tokens as two sequential
//! `seq_len == 1` forwards. Greedy `auto` vs `off` byte-identity therefore
//! requires the window forward to be numerically identical to the two-step
//! path — NOT the (much larger) full-seq-vs-decode divergence of
//! `cli_decode_parity`.
//!
//! This test isolates exactly that window: prefill a prompt, take the greedy
//! gen0, greedy gen1 from a decode step, then compare
//!   path A: one `seq_len == 2` forward over `[gen0, gen1]` (cached prefix)
//!   path B: `seq_len == 1` forward(gen0) then forward(gen1)
//! Per-layer residual dumps go to `$GRIM_LAYERS_DUMP_DIR` as
//! `grim_lres_v2_<layer>.bin` (path A) / `grim_lres_dec_<layer>.bin` (path B)
//! so the first divergent layer can be bisected like in cli_decode_parity.
//!
//! Run: GRIM_GPU_TEST=1 cargo test --release -p grim-models-transformer \
//!      --test window_decode_parity -- --ignored --nocapture

use grim_tensor::{Device, Tensor};

#[ignore = "needs the 27B model + ROCm device; run with GRIM_GPU_TEST=1"]
#[test]
fn verify_window_matches_two_decode_steps() -> Result<(), Box<dyn std::error::Error + Send + Sync>>
{
    let _ = std::fs::remove_dir_all("/tmp/window_parity_a");
    let _ = std::fs::remove_dir_all("/tmp/window_parity_b");
    std::fs::create_dir_all("/tmp/window_parity_a")?;
    std::fs::create_dir_all("/tmp/window_parity_b")?;

        // Probe model: GRIM_PROBE_MODEL overrides (default = the 27B). A small
    // extracted checkpoint (scripts/extract_probe_model.py) lets the
    // correctness probes run entirely in VRAM on ONE GPU instead of
    // forcing a 16 GB model onto a 16 GB card through host-backed memory.
    let model_path = std::env::var("GRIM_PROBE_MODEL")
        .unwrap_or_else(|_| "/drive/bigfast/grim/models/qwen38-27b/Qwen3.8-27B-Q4_K_M.gguf".to_string());
    #[allow(non_snake_case)]
    let MODEL: &str = &model_path;
    const VOCAB: usize = 248320;
    if !std::path::Path::new(MODEL).exists() {
        eprintln!("skipping: no model");
        return Ok(());
    }
    let device = Device::Rocm(0);
    let model = grim_engine::model_loader::load_model_from_gguf(MODEL, device.clone())
        .map_err(|e| format!("load: {e}"))?;
    // Same CLI prompt ids as cli_decode_parity ("Explain how a transistor...").
    let prompt: Vec<u32> = vec![
        248045, 1156, 198, 814, 20139, 1204, 264, 94562, 4138, 303, 1330, 41228, 25, 248046, 198,
        248045, 17313, 198, 248068, 198,
    ];
    let mk = |vals: &[f32], len: usize| -> Tensor {
        grim_backend_cpu::cpu_tensor(vals.to_vec(), grim_tensor::Shape::new(vec![1, len]))
    };
    let argmax = |row: &[f32]| -> (u32, f32) {
        let mut idx: Vec<u32> = (0..VOCAB as u32).collect();
        idx.sort_by(|&a, &b| row[b as usize].total_cmp(&row[a as usize]));
        (idx[0], row[idx[0] as usize])
    };

    // Prefill; greedy gen0. (Single session reused for the probe tokens —
    // both paths must see the SAME cached prefix, so the prefix is built once
    // and cloned per path via a fresh prefill with identical inputs.)
    let mut sess = model.new_session();
    let pos: Vec<f32> = (0..prompt.len()).map(|i| i as f32).collect();
    let input = mk(&prompt.iter().map(|&v| v as f32).collect::<Vec<_>>(), prompt.len());
    let pos_t = mk(&pos, pos.len());
    let logits = model
        .forward(sess.as_mut(), &input, &pos_t, &[])
        .map_err(|e| format!("prefill: {e}"))?;
    let flat = logits.to_vec_f32().map_err(|e| format!("prefill read: {e}"))?;
    let (gen0, _) = argmax(&flat[flat.len() - VOCAB..]);
    eprintln!("[wparity] prefill top1 = {gen0}");

    // greedy gen1 from a plain decode step.
    let g0_t = mk(&[gen0 as f32], 1);
    let p1 = mk(&[prompt.len() as f32], 1);
    let l1 = model
        .forward(sess.as_mut(), &g0_t, &p1, &[])
        .map_err(|e| format!("gen1 step: {e}"))?;
    let f1 = l1.to_vec_f32().map_err(|e| format!("gen1 read: {e}"))?;
    let (gen1, _) = argmax(&f1[f1.len() - VOCAB..]);
    eprintln!("[wparity] decode top1 (gen1) = {gen1}");

    // ── Path B: two sequential seq_len == 1 steps (the reference decode) ──
    let mut sess_b = model.new_session();
    let _ = model
        .forward(sess_b.as_mut(), &input, &pos_t, &[])
        .map_err(|e| format!("B prefill: {e}"))?;
    unsafe { std::env::set_var("GRIM_LAYERS_DUMP_DIR", "/tmp/window_parity_b"); }
    std::fs::create_dir_all("/tmp/window_parity_b")?;
    let lb0 = model
        .forward(sess_b.as_mut(), &g0_t, &p1, &[])
        .map_err(|e| format!("B step0: {e}"))?;
    let fb0 = lb0.to_vec_f32().map_err(|e| format!("B step0 read: {e}"))?;
    let rb0 = fb0[fb0.len() - VOCAB..].to_vec();
    let p2 = mk(&[(prompt.len() + 1) as f32], 1);
    let g1_t = mk(&[gen1 as f32], 1);
    let lb1 = model
        .forward(sess_b.as_mut(), &g1_t, &p2, &[])
        .map_err(|e| format!("B step1: {e}"))?;
    let fb1 = lb1.to_vec_f32().map_err(|e| format!("B step1 read: {e}"))?;
    let rb1 = fb1[fb1.len() - VOCAB..].to_vec();

    // ── Path A: one seq_len == 2 forward over [gen0, gen1] (verify shape) ──
    // Fresh session, same cached prefix (prompt prefill), then the window.
    unsafe { std::env::set_var("GRIM_LAYERS_DUMP_DIR", "/tmp/window_parity_a"); }
    let mut sess_a = model.new_session();
    sess_a.set_full_output_logits(true);
    let _ = model
        .forward(sess_a.as_mut(), &input, &pos_t, &[])
        .map_err(|e| format!("A prefill: {e}"))?;
    let win = mk(&[gen0 as f32, gen1 as f32], 2);
    let wpos = mk(&[prompt.len() as f32, (prompt.len() + 1) as f32], 2);
    let la = model
        .forward(sess_a.as_mut(), &win, &wpos, &[])
        .map_err(|e| format!("A window: {e}"))?;
    let fa = la.to_vec_f32().map_err(|e| format!("A read: {e}"))?;
    // Both rows' logits: row 0 predicts the token after gen0, row 1 after gen1.
    let ra0 = fa[fa.len() - 2 * VOCAB..fa.len() - VOCAB].to_vec();
    let ra1 = fa[fa.len() - VOCAB..].to_vec();

    for (name, ra, rb) in [
        ("row0 (after gen0)", &ra0, &rb0),
        ("row1 (after gen1)", &ra1, &rb1),
    ] {
        let mut max_diff = 0.0f32;
        let mut at = 0usize;
        for (i, (a, b)) in ra.iter().zip(rb.iter()).enumerate() {
            let d = (a - b).abs();
            if d > max_diff {
                max_diff = d;
                at = i;
            }
        }
        let (ta, _) = argmax(ra);
        let (tb, _) = argmax(rb);
        eprintln!(
            "[wparity] {name}: max logit diff {max_diff:.4e} at tok {at} (A {:.4} B {:.4}); top1 A={ta} B={tb}",
            ra[at], rb[at]
        );
        if ta != tb {
            eprintln!("[wparity] *** DEFECT: window and step path disagree on next token ({name}) ***");
        }
    }
    Ok(())
}
