//! Lint: no new `to_vec_f32()` regressions in model forward paths.
//!
//! The crate converged on zero host round-trips per decode step. The
//! current baseline below matches live counts at the time of the sweep
//! (2026-09). The test fails only when a file's count GROWS. Reducing a
//! file's baseline after a cleanup pass is a one-line edit and encouraged.

use std::path::PathBuf;

fn src_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src")
}

fn count_to_vec_f32(path: &std::path::Path) -> usize {
    std::fs::read_to_string(path)
        .map(|s| s.matches("to_vec_f32").count())
        .unwrap_or(0)
}

#[test]
fn roundtrip_budget_not_exceeded() {
    // Per-file baseline: every entry equals the count at the time of the
    // sweep. Missing entries default to 0 — those files are fully clean and
    // must stay that way.
    let baseline: &[(&str, usize)] = &[
        ("block.rs", 32),
        ("bloom.rs", 8),
        ("chameleon.rs", 22),
        ("commandr.rs", 11),
        ("delta_net_base.rs", 20),
        ("deepseek.rs", 11),
        ("deepseek2.rs", 12),
        ("deepseek4.rs", 16),
        ("deepseek32.rs", 12),
        ("falcon.rs", 18),
        ("falcon_h1.rs", 21),
        ("gemma.rs", 15),
        ("gemma2.rs", 13),
        ("glm5_2.rs", 19),
        ("gpt2.rs", 11),
        ("kv_attention.rs", 1),
        ("lfm2.rs", 24),
        ("lib.rs", 4),
        ("mellum.rs", 18),
        ("minicpm.rs", 16),
        ("minimax_m3.rs", 21),
        ("mistral3.rs", 0),
        ("muse_glimmer.rs", 16),
        ("native_mtp.rs", 8),
        ("qwen35.rs", 12),
        ("qwen35_perf.rs", 4),
        ("qwen35moe.rs", 22),
        ("qwen38_flash_next.rs", 22),
        ("shared_attention.rs", 3),
        ("solar_open2.rs", 2),
        ("t5.rs", 3),
        ("wav_tokenizer_dec.rs", 14),
        ("eagle3.rs", 12),
        ("inkling_small.rs", 18),
        ("interns2_mobius.rs", 16),
        ("cogvlm.rs", 18),
        ("hunyuan_vl.rs", 18),
        ("qwen2vl.rs", 18),
        ("qwen3vl.rs", 18),
        ("diffusion_gemma.rs", 18),
        ("granite_moe_hybrid.rs", 14),
        ("gemma3n.rs", 18),
        ("longcat_flash.rs", 12),
        ("dots3_note.rs", 12),
        ("bailingmoe3.rs", 2),
        ("dbrx.rs", 13),
        ("kimi_k3.rs", 35),
        ("lora.rs", 8),
        ("moe_block.rs", 1),
        ("model.rs", 8),
        ("exaone4_5.rs", 12),
        ("glm4_moe_lite.rs", 14),
        ("gpt_oss.rs", 12),
        ("gptj.rs", 10),
        ("hyv3.rs", 14),
        ("hy_v4.rs", 12),
    ];

    let mut violations = Vec::new();
    for entry in std::fs::read_dir(src_dir()).expect("src dir readable") {
        let path = entry.expect("entry").path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.ends_with(".rs") {
            continue;
        }
        let count = count_to_vec_f32(&path);
        let allowed = baseline
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, c)| *c)
            .unwrap_or(0);
        if count > allowed {
            violations.push(format!(
                "{name}: {} to_vec_f32 calls ({allowed} baseline). \
                 New host round-trips in forward paths are not allowed.",
                count
            ));
        }
    }

    assert!(
        violations.is_empty(),
        "round-trip budget exceeded:\n{}",
        violations.join("\n")
    );
}
