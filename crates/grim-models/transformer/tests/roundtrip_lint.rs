//! Lint: no new `to_vec_f32()` regressions in model forward paths.
//!
//! Counts non-test code only: top-level `#[cfg(test)] mod ...` regions are
//! skipped so parity-test asserts stop polluting the forward-path PCIe
//! budget (anarchy-uk C5). Baselines below are ceilings from the 2026-09
//! sweep (which counted whole files); ratchet them down to the new counter
//! as files are touched. The test fails only when a file's count GROWS.
//! Reducing a file's baseline after a cleanup pass is a one-line edit and
//! encouraged.

use std::path::PathBuf;

fn src_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// Count `to_vec_f32()` calls in non-test code only. Test-module asserts
/// (`#[cfg(test)] mod ...`) verify numerics host-side by design; counting
/// them against the forward-path PCIe budget conflates two behaviors (C5)
/// and punishes adding parity cover. Skips top-level test modules via brace
/// tracking; inline `#[cfg(test)]` helpers without a `mod` still count.
fn count_to_vec_f32(path: &std::path::Path) -> usize {
    let Ok(src) = std::fs::read_to_string(path) else {
        return 0;
    };
    let mut count = 0;
    let mut skip_depth: Option<usize> = None;
    let mut pending_cfg_test = false;
    let mut depth = 0usize;
    for line in src.lines() {
        let trimmed = line.trim();
        if trimmed == "#[cfg(test)]" {
            pending_cfg_test = true;
            continue;
        }
        let opens = line.matches('{').count();
        let closes = line.matches('}').count();
        if let Some(_d) = skip_depth {
            depth += opens;
            depth = depth.saturating_sub(closes);
            if depth == 0 {
                skip_depth = None;
            }
            pending_cfg_test = false;
            continue;
        }
        if pending_cfg_test {
            pending_cfg_test = false;
            // Top-level `mod name {` opens a test module: skip to its close.
            if trimmed.starts_with("mod ") && line.contains('{') {
                skip_depth = Some(0);
                depth = opens.saturating_sub(closes);
                if depth == 0 {
                    skip_depth = None;
                }
                continue;
            }
        }
        count += line.matches("to_vec_f32").count();
        depth += opens;
        depth = depth.saturating_sub(closes);
    }
    count
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
        ("delta_net_base.rs", 15),
        ("deepseek.rs", 9),
        ("deepseek2.rs", 11), // +1 Phase 3c fused-norm-gate logits pull (forward_pre_norm)
        ("deepseek4.rs", 11), // +1 Phase 3c (forward_pre_norm)
        ("deepseek32.rs", 11), // +1 Phase 3c (forward_pre_norm)
        ("falcon.rs", 10),
        ("falcon_h1.rs", 25), // WI-A: +4 test-module asserts; ratchet down in WI-G
        ("gemma.rs", 15),
        ("gemma2.rs", 13),
        ("glm5_2.rs", 9),
        ("gpt2.rs", 11),
        ("kv_attention.rs", 1),
        ("lfm2.rs", 40), // WI-F: +2 bx fetch; M1: +5 one-time expert-slice reads; Phase 2: +4 GDL host oracle probe (q,k,v,out); audit: +1 test assert (gdl_forward_matches_gla_oracle); 1-2-many P2: +3 GDL gate-projection probes (nb,nw,nf)
        ("lib.rs", 4),
        ("mellum.rs", 0),
        ("minicpm.rs", 17),
        ("minimax_m3.rs", 8),
        ("mistral3.rs", 0),
        ("muse_glimmer.rs", 16),
        ("native_mtp.rs", 7),
        ("qwen35.rs", 12),
        ("qwen35_perf.rs", 4),
        ("qwen35moe.rs", 6),
        ("qwen38_flash_next.rs", 23),
        ("shared_attention.rs", 8),
        ("shared_moe.rs", 3), // Phase 3a: once-built Charon resident weight stack (not per-decode)
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
        ("kimi_k3.rs", 26), // +1 Phase 3c (forward_pre_norm)
        ("lora.rs", 7),
        ("moe_block.rs", 1),
        ("model.rs", 7),
        ("exaone4_5.rs", 5),
        ("glm4_moe_lite.rs", 6),
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
