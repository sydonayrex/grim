//! Structural gate for the model ports (plan item 6.1).
//!
//! Catches the two failure shapes that a per-model test suite cannot see:
//!
//! 1. **Orphan modules.** A `.rs` file exists in `src/` but no `pub mod` in
//!    `lib.rs` declares it, so it is never compiled. An orphan cannot be
//!    "mostly right" or "mostly wrong" — it is inert, and it is invisible
//!    to review. Nine existed at the time of writing (`gemma3`, `gemma4`,
//!    `phi3`, `qwen2`, `phimoe`, `ernie45_moe`, `lfm2moe`, `minicpm3`,
//!    `qwen3vl_moe`).
//!
//! 2. **Undisclosed passthroughs.** A module that forwards `forward()` to
//!    an inner `Llama` while the llama.cpp reference implements a
//!    different architecture. The module compiles, loads weights and
//!    emits logits — wrong ones, with no error. 68 of 72 such files
//!    carried no warning in their header.
//!
//! The allowlist below is the escape hatch, and every entry must name the
//! reference line that justifies it. A name alone is not a reason.
//!
//! Design note, learned the hard way in the audit that produced this test:
//! an allowlist entry that merely restates the code's own behaviour is
//! worthless — the `tiled_quant_block_sizes_match_llama_cpp` gate passed
//! on a wrong block size because it recomputed the reference with the
//! same off-by-two as the table it was checking. So each entry here
//! carries a `file:line` into `ggml-common.h` or the reference `.cpp`, and
//! a test asserts the citation is non-empty. That does not prove the
//! claim is true, but it makes "I checked" a falsifiable claim rather
//! than a shrug.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

fn src_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

fn read(p: &Path) -> String {
    std::fs::read_to_string(p)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()))
}

/// Model files present in `src/` but absent from `lib.rs`, so never compiled.
///
/// Each is a *superseded* port, not a duplicate of a live one: the
/// architecture the loader actually dispatches resolves to a DIFFERENT
/// module, verified per row below. That is what makes them dead code that
/// looks like an implementation — the type, the config and the
/// `ModelConfig::name` are the only record of what the architecture
/// should look like, and none of it is compiled or reviewed.
///
/// Disposition (plan item 6.1): the correct end state is to implement these
/// and give each its own loader arm. Until then they are listed here so the
/// gate can distinguish "known and tracked" from "nobody looked", and
/// `known_orphans_are_still_referenced_by_nothing` makes each entry expire
/// the moment the module is declared.
///
/// `(module, why it is still uncompiled, the loader path that supersedes it)`
const KNOWN_ORPHANS: &[(&str, &str, &str)] = &[
    (
        "gemma3",
        "unimplemented: q/k norm, attn_post_norm, ffn_norm, SWA, logit softcap",
        "ModelArchitecture::Gemma3 shares an arm with Gemma and constructs Gemma",
    ),
    (
        "gemma4",
        "unimplemented: MoE, per-layer embeddings, shared-KV, SWA, softcap, attn_post_norm",
        "ModelArchitecture::Gemma4 shares an arm with Gemma and constructs Gemma",
    ),
    (
        "phi3",
        "unimplemented: ffn_norm, SWA, rope-factoring",
        "ModelArchitecture::Phi3 shares an arm with Phi2 and constructs Phi2",
    ),
    (
        "phimoe",
        "unimplemented: the entire expert stack (runs as a dense Phi-2)",
        "ModelArchitecture::PhiMoe shares an arm with Phi2 and constructs Phi2",
    ),
    (
        "qwen2",
        "unimplemented: ffn_norm",
        "ModelArchitecture::Qwen2 shares an arm with Qwen and constructs Qwen",
    ),
    (
        "lfm2moe",
        "unimplemented: the MoE half of LFM2 (the dense half is lfm2.rs)",
        "ModelArchitecture::Lfm2Moe has an arm, but it does not use this module",
    ),
    (
        "ernie45_moe",
        "unimplemented: MoE, shared expert, ffn_exp_probs_b, dense-lead",
        "ModelArchitecture::Ernie45Moe constructs Qwen3Moe",
    ),
    (
        "qwen3vl_moe",
        "unimplemented: MoE, q/k norm, ffn_norm",
        "ModelArchitecture::Qwen3VlMoe constructs Qwen3Moe",
    ),
    (
        "minicpm3",
        "unimplemented: ffn_norm, rope-factoring",
        "ModelArchitecture::MiniCpm3 constructs SmolLm2",
    ),
];

/// Every module that forwards `forward()` to an inner `Llama` while the
/// llama.cpp reference is *not* a plain Llama.
///
/// Each entry is `(module, "reference.cpp:line", what the passthrough drops)`.
/// The `what` column is not decoration: it is the known-wrong behaviour that
/// entry is trading for, so a reader of the list learns which models are
/// quietly approximate rather than assuming the list is a to-do of
/// housekeeping.
///
/// Generated from the per-file audit in `plans/plan-yodellers-inc.md`; the
/// citation is checked against the real file by
/// `allowlist_citations_name_a_real_reference_file`, so an entry cannot rot
/// into pointing at a file that no longer exists.
const ALLOWED_PASSTHROUGHS: &[(&str, &str, &str)] = &[
    ("afmoe", "afmoe.cpp:38", "attn_post_norm not loaded, sliding-window attention, MoE expert stack, shared expert, ffn_exp_probs_b"),
    ("apertus", "apertus.cpp:17", "per-layer ROPE_FREQS table; RopeConfig is scalar-base only"),
    ("arcee", "arcee.cpp:13", "per-layer ROPE_FREQS table; RopeConfig is scalar-base only"),
    ("arctic", "arctic.cpp:16", "MoE expert stack"),
    ("baichuan", "baichuan.cpp:17", "alibi (f_max_alibi_bias) never set on block's alibi_slopes"),
    ("bailingmoe", "bailingmoe.cpp:18", "MoE expert stack, shared expert"),
    ("bailingmoe2", "bailingmoe2.cpp:20", "MoE expert stack, shared expert, ffn_exp_probs_b, MTP head"),
    ("bitnet", "bitnet.cpp:12", "1.58-bit ternary + per-projection scales; not a Llama at all"),
    ("chatglm", "chatglm.cpp:25", "none found in the reference beyond Llama geometry"),
    ("codeshell", "codeshell.cpp:12", "LayerNorm (LLM_NORM) not RMSNorm, attn_norm_b/ffn_norm_b dropped (has_bias=false)"),
    ("cohere2", "cohere2.cpp:21", "sliding-window attention, logit/attn/resid/emb scale"),
    ("cohere2moe", "cohere2moe.cpp:40", "MoE expert stack, shared expert, MTP head, logit/attn/resid/emb scale"),
    ("deci", "deci.cpp:14", "per-layer ROPE_FREQS table; RopeConfig is scalar-base only"),
    ("deepseek2ocr", "deepseek2ocr.cpp:23", "MoE expert stack, shared expert, ffn_exp_probs_b"),
    ("dflash", "dflash.cpp:107", "attn_post_norm not loaded, MoE expert stack, shared expert, ffn_exp_probs_b, logit softcapping, logit/attn/resid/emb scale"),
    ("dots1", "dots1.cpp:18", "MoE expert stack, shared expert, ffn_exp_probs_b"),
    ("dream", "dream.cpp:18", "none found in the reference beyond Llama geometry"),
    ("ernie45", "ernie4-5.cpp:23", "MoE expert stack, shared expert, ffn_exp_probs_b"),
    ("ernie4_5_moe", "ernie4-5-moe.cpp:1", "MoE expert stack, shared expert, ffn_exp_probs_b"),
    ("eurobert", "eurobert.cpp:11", "encoder: bidirectional mask, no causal LM head"),
    ("exaone", "exaone.cpp:12", "per-layer ROPE_FREQS table; RopeConfig is scalar-base only"),
    ("exaone4", "exaone4.cpp:24", "attn_post_norm not loaded, sliding-window attention, MTP head"),
    ("exaone_moe", "exaone-moe.cpp:28", "sliding-window attention, MoE expert stack, shared expert, ffn_exp_probs_b, MTP head"),
    ("gemma4_assistant", "gemma4-assistant.cpp:21", "attn_post_norm not loaded, MTP head"),
    ("gemma_embedding", "gemma-embedding.cpp:30", "attn_post_norm not loaded, sliding-window attention"),
    ("glm4", "glm4.cpp:15", "attn_post_norm not loaded, MTP head"),
    ("glm4moe", "glm4-moe.cpp:28", "attn_post_norm not loaded"),
    ("glmdsa", "glm-dsa.cpp:74", "MoE expert stack, shared expert, ffn_exp_probs_b, MTP head"),
    ("gptneox", "gptneox.cpp:50", "LayerNorm (LLM_NORM) not RMSNorm, attn_norm_b/ffn_norm_b dropped (has_bias=false)"),
    ("granite", "granite.cpp:59", "MoE expert stack, shared expert, logit/attn/resid/emb scale"),
    ("granite_moe", "granite-moe.cpp:21", "alias; real implementation is GraniteMoeHybrid"),
    ("grok", "grok.cpp:35", "MoE expert stack, logit softcapping, logit/attn/resid/emb scale"),
    ("grovemoe", "grovemoe.cpp:16", "MoE expert stack"),
    ("hunyuan_dense", "hunyuan-dense.cpp:1", "195-byte ref, inherits load_arch_tensors from llama_model_hunyuan_vl"),
    ("hunyuan_moe", "hunyuan-moe.cpp:14", "MoE expert stack, shared expert"),
    ("internlm2", "internlm2.cpp:13", "none found in the reference beyond Llama geometry"),
    ("jais", "jais.cpp:15", "LayerNorm (LLM_NORM) not RMSNorm, attn_norm_b/ffn_norm_b dropped (has_bias=false)"),
    ("jais2", "jais2.cpp:13", "LayerNorm (LLM_NORM) not RMSNorm, attn_norm_b/ffn_norm_b dropped (has_bias=false)"),
    ("kimi_linear", "kimi-linear.cpp:34", "MoE expert stack, shared expert, ffn_exp_probs_b"),
    ("llada", "llada.cpp:19", "per-layer ROPE_FREQS table; RopeConfig is scalar-base only"),
    ("lladamoe", "llada-moe.cpp:16", "MoE expert stack"),
    ("llama4", "llama4.cpp:44", "sliding-window attention, MoE expert stack, shared expert"),
    ("llama_embed", "llama-embed.cpp:1", "MoE expert stack, shared expert"),
    ("maincoder", "maincoder.cpp:12", "none found in the reference beyond Llama geometry"),
    ("mimo2", "mimo2.cpp:25", "MoE expert stack, ffn_exp_probs_b, MTP head"),
    ("minimax_m2", "minimax-m2.cpp:14", "MoE expert stack, ffn_exp_probs_b"),
    ("mistral3", "mistral3.cpp:29", "MoE expert stack, shared expert"),
    ("mistral4", "mistral4.cpp:1", "MoE expert stack, shared expert, ffn_exp_probs_b, MTP head"),
    ("mpt", "mpt.cpp:15", "LayerNorm (LLM_NORM) not RMSNorm, attn_norm_b/ffn_norm_b dropped (has_bias=false)"),
    ("nemotron", "nemotron.cpp:12", "LayerNorm (LLM_NORM) not RMSNorm, attn_norm_b/ffn_norm_b dropped (has_bias=false)"),
    ("olmo", "olmo.cpp:15", "LayerNorm (LLM_NORM) not RMSNorm"),
    ("olmo2", "olmo2.cpp:27", "attn_post_norm not loaded, sliding-window attention"),
    ("openai_moe", "openai-moe.cpp:22", "attn_post_norm not loaded, sliding-window attention, MoE expert stack"),
    ("openelm", "openelm.cpp:15", "none found in the reference beyond Llama geometry"),
    ("orion", "orion.cpp:12", "LayerNorm (LLM_NORM) not RMSNorm, attn_norm_b/ffn_norm_b dropped (has_bias=false)"),
    ("paddle_ocr", "paddleocr.cpp:1", "MoE expert stack, shared expert, ffn_exp_probs_b"),
    ("pangu_embed", "pangu-embed.cpp:13", "per-layer ROPE_FREQS table; RopeConfig is scalar-base only"),
    ("phi2", "phi2.cpp:13", "LayerNorm (LLM_NORM) not RMSNorm; attn_norm_b/output_norm_b/wo_b/ffn_up_b/ffn_down_b dropped"),
    ("plamo", "plamo.cpp:12", "none found in the reference beyond Llama geometry"),
    ("plamo2", "plamo2.cpp:35", "attn_post_norm not loaded"),
    ("plamo3", "plamo3.cpp:20", "attn_post_norm not loaded, sliding-window attention"),
    ("plm", "plm.cpp:13", "none found in the reference beyond Llama geometry"),
    ("qwen", "qwen.cpp:13", "ffn_norm never loaded; fused attn_qkv never assembled"),
    ("qwen3", "qwen3.cpp:15", "CLS_OUT head and build_lora_mm output scale both missing"),
    ("qwen3next", "qwen3next.cpp:31", "attn_post_norm not loaded, MoE expert stack, shared expert, MTP head"),
    ("refact", "refact.cpp:15", "MoE expert stack, shared expert"),
    ("rnd1", "rnd1.cpp:16", "MoE expert stack"),
    ("seed_oss", "seed-oss.cpp:12", "attn_post_norm not loaded"),
    ("smallthinker", "smallthinker.cpp:30", "sliding-window attention, MoE expert stack"),
    ("smollm2", "(no upstream reference)", "no upstream reference file exists for smollm2"),
    ("smollm3", "smollm3.cpp:13", "none found in the reference beyond Llama geometry"),
    ("stablelm", "stablelm.cpp:14", "LayerNorm (LLM_NORM) not RMSNorm, attn_norm_b/ffn_norm_b dropped (has_bias=false)"),
    ("starcoder", "starcoder.cpp:15", "LayerNorm (LLM_NORM) not RMSNorm, attn_norm_b/ffn_norm_b dropped (has_bias=false)"),
    ("starcoder2", "starcoder2.cpp:16", "LayerNorm (LLM_NORM) not RMSNorm, attn_norm_b/ffn_norm_b dropped (has_bias=false)"),
    ("step35", "step35.cpp:37", "MoE expert stack, shared expert, ffn_exp_probs_b, MTP head"),
    ("talkie", "talkie.cpp:13", "logit/attn/resid/emb scale"),
    ("xverse", "xverse.cpp:14", "none found in the reference beyond Llama geometry"),
];

/// Does this file forward `forward()` to an inner model without a block of
/// its own? Matches the shape the audit found: a struct holding `inner`,
/// no `*Block` type, no `MoeBlock`/`MoESpec`/`RouterKind` wiring.
fn is_passthrough(src: &str) -> bool {
    let has_inner = src.contains("inner: Llama") || src.contains("inner: Llama,");
    let has_own_block = {
        let mut found = false;
        for line in src.lines() {
            let t = line.trim_start_matches("pub struct ").trim_start();
            if let Some(rest) = t.strip_prefix("Llama") {
                if rest.starts_with("Block") {
                    found = true;
                }
            }
            if let Some(rest) = t.strip_prefix("MoE") {
                if rest.starts_with("Block") {
                    found = true;
                }
            }
        }
        found
    };
    let has_moe_wiring = src.contains("MoeBlock")
        || src.contains("MoESpec")
        || src.contains("RouterKind")
        || src.contains("moe_block");
    let has_forward = src.contains("fn forward");
    has_inner && has_forward && !has_own_block && !has_moe_wiring
}

/// Every module name declared in `lib.rs`.
fn declared_modules(lib: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for line in lib.lines() {
        let t = line.trim();
        for kw in ["pub mod ", "mod "] {
            if let Some(rest) = t.strip_prefix(kw) {
                if let Some(name) = rest.strip_suffix(';') {
                    if !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                        out.insert(name.to_string());
                    }
                }
            }
        }
    }
    out
}

#[test]
fn no_undeclared_orphan_model_files() {
    let src = src_dir();
    let lib = read(&src.join("lib.rs"));
    let declared = declared_modules(&lib);
    assert!(
        declared.len() > 100,
        "lib.rs yielded only {} module declarations; the parser is probably broken",
        declared.len()
    );

    let mut unlisted: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(&src).expect("read src/") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if matches!(
            stem,
            "lib" | "mod" | "block" | "configs" | "model" | "attention_dispatcher"
        ) {
            continue; // infrastructure, not model files
        }
        if declared.contains(stem) {
            continue;
        }
        // An orphan is a file nobody compiles: it cannot be reviewed, tested
        // or caught by the compiler. Unlisted, that is a defect. Listed in
        // KNOWN_ORPHANS, it is a tracked decision, and
        // known_orphans_are_still_referenced_by_nothing below makes the list
        // self-expiring.
        if !KNOWN_ORPHANS.iter().any(|(m, _, _)| *m == stem) {
            unlisted.push(format!(
                "{} ({} B) -- declare it and implement it, delete it, or list it in \
                 KNOWN_ORPHANS with a reason",
                path.display(),
                read(&path).len()
            ));
        }
    }
    assert!(
        unlisted.is_empty(),
        "undeclared model files in src/ (present, never compiled, unreviewable):\n  {}",
        unlisted.join("\n  ")
    );
}

#[test]
fn known_orphans_are_still_referenced_by_nothing() {
    // If a known orphan becomes referenced (a loader arm added, or the module
    // declared), it stops being an orphan and the entry must go, otherwise the
    // allowlist rots into a place where it stops meaning anything.
    let src = src_dir();
    let lib = read(&src.join("lib.rs"));
    let declared = declared_modules(&lib);
    let mut stale = Vec::new();
    for (module, _why, _since) in KNOWN_ORPHANS {
        let path = src.join(format!("{module}.rs"));
        if !path.exists() {
            continue; // deleted: entry can stay until the next edit
        }
        if declared.contains(*module) {
            stale.push(format!(
                "{module} is declared in lib.rs, so it is no longer an orphan -- \
                 remove it from KNOWN_ORPHANS"
            ));
        }
    }
    assert!(stale.is_empty(), "{}", stale.join("\n"));
}

#[test]
fn passthrough_modules_are_allowlisted_with_a_citation() {
    let src = src_dir();
    let lib = read(&src.join("lib.rs"));
    let declared = declared_modules(&lib);
    let allow: BTreeMap<_, _> = ALLOWED_PASSTHROUGHS
        .iter()
        .map(|(m, cite, why)| (*m, (*cite, *why)))
        .collect();

    let mut unlisted: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(&src).expect("read src/") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if !declared.contains(stem) {
            continue; // orphans are the previous test's job
        }
        let body = read(&path);
        if !is_passthrough(&body) {
            continue;
        }
        match allow.get(stem) {
            Some((cite, why)) => {
                // Every entry must justify itself. A citation is preferred but
                // not always possible: `smollm2` has no upstream reference at
                // all, so demanding a `file:line` there would be demanding a
                // fiction. What is not negotiable is a non-empty reason.
                assert!(
                    !why.trim().is_empty(),
                    "{stem} is allowlisted with an empty reason"
                );
                if !cite.starts_with("(no upstream reference") {
                    assert!(
                        cite.contains(".cpp:") || cite.contains(".h:"),
                        "{stem} is allowlisted but its citation {cite:?} names no reference line"
                    );
                }
            }
            None => unlisted.push(stem.to_string()),
        }
    }
    assert!(
        unlisted.is_empty(),
        "these modules forward forward() to an inner Llama with no block of their own and \
         no allowlist entry:\n  {}\n\
         Either the reference really is Llama-shaped -- in which case add it to \
         ALLOWED_PASSTHROUGHS with a citation into the reference -- or it is not, in which \
         case the file is a silent wrong-numerics bug.",
        unlisted.join("\n  ")
    );
}

#[test]
fn allowlist_citations_name_a_real_reference_file() {
    // A citation that points at a file which does not exist is worse than no
    // citation, because it looks like the entry was checked.
    //
    // CARGO_MANIFEST_DIR is `crates/grim-models/transformer`, so the repo
    // root is THREE levels up, not two.
    let ref_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .expect("crates/grim-models/transformer -> repo root")
        .join("old/repo/llama.cpp-master/src/models");
    assert!(
        ref_dir.exists(),
        "reference dir {} does not exist; the path arithmetic is wrong",
        ref_dir.display()
    );
    for (module, cite, _why) in ALLOWED_PASSTHROUGHS {
        if cite.starts_with("(no upstream reference") {
            continue; // grim invented this one; nothing to check against
        }
        let file = cite.split(':').next().unwrap_or_default();
        assert!(
            ref_dir.join(file).exists(),
            "{module} cites {cite:?} but {file} does not exist in {}",
            ref_dir.display()
        );
    }
}
