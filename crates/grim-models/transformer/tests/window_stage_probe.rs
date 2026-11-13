//! Stage-level window probe: dump layer-0 KDA stage tensors for a seq_len==2
//! verify window (`win_*`) and for two sequential seq_len==1 decode steps
//! (`dec_*`, `dec2_*`), then diff stage by stage to localize the first
//! divergent tensor inside the layer. Companion to window_decode_parity.
//!
//! Requires GRIM_KDA_STEP_PROBE=1 (set after gen derivation so the counters
//! line up). Run:
//! GRIM_GPU_TEST=1 cargo test --release -p grim-models-transformer \
//!   --test window_stage_probe -- --ignored --nocapture

use grim_tensor::{Device, Tensor};

#[ignore = "needs the 27B model + ROCm device; run with GRIM_GPU_TEST=1"]
#[test]
fn window_vs_two_steps_stage_probe() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    unsafe {
        std::env::remove_var("GRIM_KDA_STEP_PROBE");
        std::env::remove_var("GRIM_ATTN_DUMP_DIR");
    }
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
    let prompt: Vec<u32> = vec![
        248045, 1156, 198, 814, 20139, 1204, 264, 94562, 4138, 303, 1330, 41228, 25, 248046, 198,
        248045, 17313, 198, 248068, 198,
    ];
    let mk = |vals: &[f32], len: usize| -> Tensor {
        grim_backend_cpu::cpu_tensor(vals.to_vec(), grim_tensor::Shape::new(vec![1, len]))
    };
    let argmax = |row: &[f32]| -> u32 {
        let mut best = 0usize;
        for (i, v) in row.iter().enumerate() {
            if v.total_cmp(&row[best]).is_gt() {
                best = i;
            }
        }
        best as u32
    };

    let mut sess = model.new_session();
    let pos: Vec<f32> = (0..prompt.len()).map(|i| i as f32).collect();
    let input = mk(&prompt.iter().map(|&v| v as f32).collect::<Vec<_>>(), prompt.len());
    let pos_t = mk(&pos, pos.len());
    let logits = model
        .forward(sess.as_mut(), &input, &pos_t, &[])
        .map_err(|e| format!("prefill: {e}"))?;
    let flat = logits.to_vec_f32().map_err(|e| format!("prefill read: {e}"))?;
    let gen0 = argmax(&flat[flat.len() - VOCAB..]);
    let g0_t = mk(&[gen0 as f32], 1);
    let p1 = mk(&[prompt.len() as f32], 1);
    let l1 = model
        .forward(sess.as_mut(), &g0_t, &p1, &[])
        .map_err(|e| format!("gen1 step: {e}"))?;
    let f1 = l1.to_vec_f32().map_err(|e| format!("gen1 read: {e}"))?;
    let gen1 = argmax(&f1[f1.len() - VOCAB..]);
    eprintln!("[stage] gen0={gen0} gen1={gen1}");

    // Enable the probe for the compared forwards only.
    unsafe {
        std::env::set_var("GRIM_KDA_STEP_PROBE", "1");
        std::fs::create_dir_all("/tmp/attn_dump").ok();
        std::env::set_var("GRIM_ATTN_DUMP_DIR", "/tmp/attn_dump");
    }

    // Path B: two sequential decode steps (fresh session, same prefix).
    let mut sess_b = model.new_session();
    let _ = model
        .forward(sess_b.as_mut(), &input, &pos_t, &[])
        .map_err(|e| format!("B prefill: {e}"))?;
    let _ = model
        .forward(sess_b.as_mut(), &g0_t, &p1, &[])
        .map_err(|e| format!("B step1: {e}"))?;
    let p2 = mk(&[(prompt.len() + 1) as f32], 1);
    let g1_t = mk(&[gen1 as f32], 1);
    let _ = model
        .forward(sess_b.as_mut(), &g1_t, &p2, &[])
        .map_err(|e| format!("B step2: {e}"))?;

    // Path A: one seq_len==2 window (fresh session, same prefix).
    let mut sess_a = model.new_session();
    sess_a.set_full_output_logits(true);
    let _ = model
        .forward(sess_a.as_mut(), &input, &pos_t, &[])
        .map_err(|e| format!("A prefill: {e}"))?;
    let win = mk(&[gen0 as f32, gen1 as f32], 2);
    let wpos = mk(&[prompt.len() as f32, (prompt.len() + 1) as f32], 2);
    let _ = model
        .forward(sess_a.as_mut(), &win, &wpos, &[])
        .map_err(|e| format!("A window: {e}"))?;

    eprintln!("[stage] dumps written to /tmp/kda_probe_d2d_*; diff win_* vs dec_*/dec2_*");
    Ok(())
}
