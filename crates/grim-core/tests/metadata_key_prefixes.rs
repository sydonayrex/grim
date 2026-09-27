//! Arch-prefixed metadata keys must be built by interpolation, never written out
//! literally.
//!
//! `ArchHyperparameters::extract` reads its keys as
//! `get_u32(&format!("{arch_name}.ssm.state_size"))` so one function serves
//! every architecture. A single literal crept in —
//! `"qwen35.ssm.time_step_rank_DOES_NOT_EXIST"` — so that read could never
//! resolve, `hparams.ssm_dt_rank` was always `None`, and every load fell back
//! to `unwrap_or(48)`. The 27B really is 48, so nothing looked wrong; Qwen3.5-9B
//! is 32, so its `n_val_heads`, `value_dim` and `conv_dim` came out 1.5x too
//! wide while the file's own `attn_qkv` stayed at the true 8192.
//!
//! The failure is silent by construction: a key that never matches just
//! returns `None` and takes the default. So this asserts the shape of the code
//! rather than one value — a literal arch prefix inside the extractor is the
//! smell, and there should be none.

use std::fs;
use std::path::Path;

#[test]
fn no_hardcoded_arch_prefixes_in_the_hyperparameter_extractor() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/hyperparams.rs");
    let src = fs::read_to_string(&path).expect("read hyperparams.rs");

    // Only the extractor matters; other lookups elsewhere in the file are
    // allowed to name an arch explicitly.
    let start = src
        .find("pub fn extract")
        .expect("ArchHyperparameters::extract must exist");
    let body = &src[start..];

    let mut offenders = Vec::new();
    for (i, line) in body.lines().enumerate() {
        let t = line.trim();
        if !t.contains("get_u32") && !t.contains("get_i32") && !t.contains("get_f32") {
            continue;
        }
        // A literal arch prefix inside a quoted key: `"qwen35.` / `"qwen4exp.`
        // etc. Interpolated reads look like `format!("{arch_name}...` instead.
        let quoted = t.split('"').nth(1).unwrap_or("");
        if quoted.starts_with("qwen") && quoted.contains('.') {
            offenders.push(format!("  line {}: {}", start + i + 1, t));
        }
    }
    assert!(
        offenders.is_empty(),
        "hardcoded arch prefix(es) in the extractor — these cannot resolve for \
         any other architecture and will silently take a default:\n{}",
        offenders.join("\n")
    );
}

/// The extractor must read the KDA geometry from the file, not from constants.
#[test]
fn the_extractor_reads_the_kda_geometry_keys() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/hyperparams.rs");
    let src = fs::read_to_string(&path).expect("read hyperparams.rs");
    let start = src
        .find("pub fn extract")
        .expect("ArchHyperparameters::extract must exist");
    let body = &src[start..];

    for key in [
        "ssm.state_size",
        "ssm.inner_size",
        "ssm.conv_kernel",
        "ssm.time_step_rank",
        "ssm.group_count",
        "full_attention_interval",
    ] {
        assert!(
            body.contains(&format!("{{arch_name}}.{key}")),
            "the extractor must read {key} via {{arch_name}} interpolation"
        );
    }
}
