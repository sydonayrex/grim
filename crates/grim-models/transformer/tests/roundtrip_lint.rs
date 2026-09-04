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
        ("bloom.rs", 6),
        ("chameleon.rs", 5),
        ("commandr.rs", 5),
        ("delta_net_base.rs", 13),
        ("deepseek.rs", 9),
        ("deepseek2.rs", 10),
        ("deepseek4.rs", 10),
        ("deepseek32.rs", 10),
        ("falcon.rs", 10),
        ("falcon_h1.rs", 25), // WI-A: +4 test-module asserts; ratchet down in WI-G
        ("gemma.rs", 15),
        ("gemma2.rs", 8),
        ("glm5_2.rs", 7),
        ("gpt2.rs", 11),
        ("kv_attention.rs", 1),
        ("lfm2.rs", 27), // WI-F: +2 (device-path bx fetch + decode test); decode no longer pulls proj
        ("lib.rs", 4),
        ("mellum.rs", 0),
        ("minicpm.rs", 17),
        ("minimax_m3.rs", 5),
        ("mistral3.rs", 0),
        ("muse_glimmer.rs", 16),
        ("native_mtp.rs", 7),
        ("qwen35.rs", 10),
        ("qwen35_perf.rs", 4),
        ("qwen35moe.rs", 6),
        ("qwen38_flash_next.rs", 20),
        ("shared_attention.rs", 8),
        ("solar_open2.rs", 2),
        ("t5.rs", 3),
        ("wav_tokenizer_dec.rs", 14),
        ("eagle3.rs", 5),
        ("inkling_small.rs", 12),
        ("interns2_mobius.rs", 3),
        ("cogvlm.rs", 2),
        ("hunyuan_vl.rs", 2),
        ("qwen2vl.rs", 2),
        ("qwen3vl.rs", 2),
        ("diffusion_gemma.rs", 14),
        ("granite_moe_hybrid.rs", 8),
        ("gemma3n.rs", 5),
        ("longcat_flash.rs", 5),
        ("dots3_note.rs", 5),
        ("bailingmoe3.rs", 2),
        ("dbrx.rs", 3),
        ("kimi_k3.rs", 25),
        ("lora.rs", 7),
        ("moe_block.rs", 1),
        ("model.rs", 7),
        ("exaone4_5.rs", 5),
        ("glm4_moe_lite.rs", 4),
        ("gpt_oss.rs", 2),
        ("gptj.rs", 6),
        ("hyv3.rs", 4),
        ("hy_v4.rs", 5),
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
