//! Structural gate for the Llama-shaped consolidation
//! (`plans/PLAN-macho-merge.md`).
//!
//! ## Why this exists
//!
//! `crates/grim-models/transformer/src/` carries 150 model files. Most are
//! real implementations. Twelve were a `struct` whose `forward` forwarded to
//! an inner `Llama`, duplicating one another 96% word for word and never
//! being constructed by anything. They are now one table
//! (`src/llama_shaped.rs`).
//!
//! Two properties have to hold afterwards, or the deletion is just a way to
//! lose information:
//!
//! 1. Nothing Llama-shaped is left as a file.
//! 2. Nothing *not* Llama-shaped is gone. Those 80 files are the only record
//!    of what each model needs -- `afmoe` needs `ATTN_GATE`, `bitnet` needs
//!    `ATTN_SUB_NORM`, `bailingmoe2` needs a fused `ATTN_QKV` plus `NEXTN_*`.
//!    Deleting one is a silent wrong-numerics bug, so each is listed here and
//!    its absence fails the build.
//!
//! ## The classification, and why it is measured
//!
//! "Llama-shaped" is decided by the *reference*, not by the grim file's size
//! or its similarity to its neighbours. An architecture qualifies only when
//! all four hold against `old/repo/llama.cpp-master/src/models/`:
//!
//! * it creates exactly the tensor set `llama.cpp` creates;
//! * it has no router (`FFN_GATE_INP` / `n_expert`) -- tensor-name equality
//!   is not enough, because `bailingmoe` reuses llama.cpp's expert tensor
//!   names and is genuinely MoE;
//! * it has no sliding-window pattern;
//! * it never calls `build_norm(..., LLM_NORM, ...)`, i.e. it is RMS.
//!
//! Two earlier versions of this gate inferred architecture from surface form
//! and over-counted: one generalised `gptneox`'s tensor count to all 84
//! candidates, another counted `grep -c` lines as operations. Both would
//! have deleted files recording real deviations.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn src_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

fn read(p: &Path) -> String {
    std::fs::read_to_string(p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()))
}

fn ref_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .expect("crates/grim-models/transformer -> repo root")
        .join("old/repo/llama.cpp-master/src/models")
}

/// Every `.rs` in `src/`, as module stems, paired with its body.
///
/// One walk: three separate walks of the same directory was both slower and
/// the reason the same filter had to be repeated three times.
fn model_files() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(src_dir()).expect("read src/") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
            out.push((stem.to_string(), read(&path)));
        }
    }
    out
}

/// True when the file declares a `struct` holding `inner: Llama` and no
/// block of its own -- i.e. a passthrough.
///
/// Comment lines are skipped: `decode_graph.rs` names `inner: Llama` in a
/// doc comment, and the first version of this check flagged it.
fn is_llama_passthrough(body: &str) -> bool {
    let has_field = body
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .any(|l| l.contains("inner: Llama"));
    has_field && !body.contains("pub struct LlamaBlock")
}

/// Module stems declared in `lib.rs`.
fn declared_modules(lib: &str) -> BTreeSet<String> {
    lib.lines()
        .filter_map(|l| l.trim().strip_prefix("pub mod "))
        .filter_map(|l| l.strip_suffix(';'))
        .filter(|n| !n.is_empty() && n.chars().all(|c| c.is_alphanumeric() || c == '_'))
        .map(str::to_string)
        .collect()
}

/// Architectures whose reference is genuinely Llama-shaped, with the
/// reference file each verdict came from. Twelve entries.
///
/// Kept as `data` rather than re-derived: the test asserts each citation
/// names a file that exists, so an entry cannot rot into looking checked.
const LLAMA_SHAPED: &[(&str, &str)] = &[
    ("baichuan", "baichuan.cpp"),
    ("chatglm", "chatglm.cpp"),
    ("dream", "dream.cpp"),
    ("eurobert", "eurobert.cpp"),
    // 195-byte reference: build_arch_graph only, inherits load_arch_tensors
    // from llama_model_hunyuan_vl.
    ("hunyuan_dense", "hunyuan-dense.cpp"),
    // 199-byte reference, inherits load_arch_tensors from llama_model_llama.
    ("llama_embed", "llama-embed.cpp"),
    // 190-byte reference, inherits from llama_model_deepseek2 (models.h:1393).
    ("mistral4", "mistral4.cpp"),
    // Vision encoder (build_inp_embd/build_cvec), not a causal LM.
    ("paddle_ocr", "paddleocr.cpp"),
    ("plamo", "plamo.cpp"),
    ("qwen2", "qwen2.cpp"),
    ("smollm3", "smollm3.cpp"),
    ("xverse", "xverse.cpp"),
];

/// Modules that forward `forward()` to an inner `Llama` while their
/// reference is NOT Llama-shaped. They must keep their own file: that file
/// is the only record of what the model needs.
const DEVIANT_PASSTHROUGHS: &[&str] = &[
    "afmoe",
    "apertus",
    "arcee",
    "arctic",
    "bailingmoe",
    "bailingmoe2",
    "bitnet",
    "codeshell",
    "cohere2",
    "cohere2moe",
    "deci",
    "deepseek2ocr",
    "dflash",
    "dots1",
    "ernie45",
    "ernie45_moe",
    "ernie4_5_moe",
    "exaone",
    "exaone4",
    "exaone_moe",
    "gemma3",
    "gemma4",
    "gemma4_assistant",
    "gemma_embedding",
    "glm4",
    "glm4moe",
    "glmdsa",
    "gptneox",
    "granite",
    "granite_moe",
    "grok",
    "grovemoe",
    "hunyuan_moe",
    "internlm2",
    "jais",
    "jais2",
    "kimi_linear",
    "laguna",
    "lfm2moe",
    "llada",
    "lladamoe",
    "llama4",
    "maincoder",
    "maple",
    "mellum",
    "mimo2",
    "minicpm3",
    "minimax_m2",
    "mistral3",
    "mpt",
    "nemotron",
    "olmo",
    "olmo2",
    "olmoe",
    "openai_moe",
    "openelm",
    "orion",
    "pangu_embed",
    "phi2",
    "phi3",
    "phimoe",
    "plamo2",
    "plamo3",
    "plm",
    "qwen",
    "qwen2moe",
    "qwen3",
    "qwen3moe",
    "qwen3next",
    "qwen3vl_moe",
    "refact",
    "rnd1",
    "seed_oss",
    "smallthinker",
    "smollm2",
    "stablelm",
    "starcoder",
    "starcoder2",
    "step35",
    "talkie",
];

/// Files present in `src/` but absent from `lib.rs`, so never compiled, with
/// what the loader does with the architecture instead.
///
/// NOT disjoint from `DEVIANT_PASSTHROUGHS`: all eight are also
/// passthroughs. Two lists, not one taxonomy, for that reason.
const KNOWN_ORPHANS: &[&str] = &[
    "ernie45_moe",
    "gemma3",
    "gemma4",
    "lfm2moe",
    "minicpm3",
    "phi3",
    "phimoe",
    "qwen3vl_moe",
];

/// Not model files: shared infrastructure that legitimately holds `Llama`.
const INFRA: &[&str] = &[
    "lib",
    "mod",
    "block",
    "configs",
    "model",
    "decode_graph",
    "llama_shaped",
    "attention_dispatcher",
];

#[test]
fn the_table_cites_reference_files_that_exist() {
    let dir = ref_dir();
    assert!(dir.exists(), "reference dir {} is missing", dir.display());
    for (module, reference) in LLAMA_SHAPED {
        assert!(
            dir.join(reference).exists(),
            "{module} cites {reference}, which is not in {}",
            dir.display()
        );
    }
}

#[test]
fn the_table_is_well_formed() {
    let mut seen = BTreeSet::new();
    for (module, _) in LLAMA_SHAPED {
        assert!(seen.insert(*module), "{module} appears twice");
        assert!(
            !DEVIANT_PASSTHROUGHS.contains(module),
            "{module} is in both lists"
        );
        assert!(!KNOWN_ORPHANS.contains(module), "{module} is in both lists");
    }
}

#[test]
fn no_llama_shaped_module_file_remains() {
    // Makes the deletion permanent: without it the twelve files can return,
    // and a reviewer will not notice twelve new ~2,900-byte files that all
    // say the same thing.
    let table: BTreeSet<&str> = LLAMA_SHAPED.iter().map(|(m, _)| *m).collect();
    let left: Vec<String> = model_files()
        .into_iter()
        .filter(|(stem, _)| table.contains(stem.as_str()))
        .map(|(stem, _)| stem)
        .collect();
    assert!(
        left.is_empty(),
        "these are in LLAMA_SHAPED but their .rs files still exist: {left:?}"
    );
}

#[test]
fn every_deviant_passthrough_file_still_exists() {
    let present: BTreeSet<String> = model_files().into_iter().map(|(s, _)| s).collect();
    let missing: Vec<&&str> = DEVIANT_PASSTHROUGHS
        .iter()
        .filter(|m| !present.contains(**m))
        .collect();
    assert!(
        missing.is_empty(),
        "recorded as deviant passthroughs but their .rs files are gone: {missing:?}\n\
         Either one was deleted without re-checking the reference, or it was \
         implemented and DEVIANT_PASSTHROUGHS needs updating."
    );
}

#[test]
fn the_deviant_list_is_exhaustive() {
    let listed: BTreeSet<&str> = DEVIANT_PASSTHROUGHS.iter().copied().collect();
    let table: BTreeSet<&str> = LLAMA_SHAPED.iter().map(|(m, _)| *m).collect();
    let orphans: BTreeSet<&str> = KNOWN_ORPHANS.iter().copied().collect();
    let unlisted: Vec<String> = model_files()
        .into_iter()
        .filter(|(stem, _)| !INFRA.contains(&stem.as_str()))
        .filter(|(stem, _)| !table.contains(stem.as_str()) && !orphans.contains(stem.as_str()))
        .filter(|(_, body)| is_llama_passthrough(body))
        .map(|(stem, _)| stem)
        .filter(|s| !listed.contains(s.as_str()))
        .collect();
    assert!(
        unlisted.is_empty(),
        "these forward forward() to an inner Llama but are in none of the three \
         lists:\n  {}\n\
         Either the reference genuinely is Llama-shaped -- add it to LLAMA_SHAPED \
         with a citation and delete its file -- or it is not, and this is a \
         deviant model that must be listed so its gap stays visible.",
        unlisted.join("\n  ")
    );
}

#[test]
fn every_orphan_is_still_an_orphan() {
    let declared = declared_modules(&read(&src_dir().join("lib.rs")));
    assert!(
        declared.len() > 100,
        "lib.rs parser broke: {} modules",
        declared.len()
    );
    for stem in KNOWN_ORPHANS {
        assert!(
            !declared.contains(*stem),
            "{stem} is declared in lib.rs, so it is no longer an orphan -- \
             move it out of KNOWN_ORPHANS"
        );
    }
}

#[test]
fn no_undeclared_file_outside_the_three_lists() {
    let declared = declared_modules(&read(&src_dir().join("lib.rs")));
    let table: BTreeSet<&str> = LLAMA_SHAPED.iter().map(|(m, _)| *m).collect();
    let orphans: BTreeSet<&str> = KNOWN_ORPHANS.iter().copied().collect();
    let unlisted: Vec<String> = model_files()
        .into_iter()
        .filter(|(stem, _)| !INFRA.contains(&stem.as_str()))
        .filter(|(stem, _)| {
            !declared.contains(stem.as_str())
                && !table.contains(stem.as_str())
                && !orphans.contains(stem.as_str())
        })
        .map(|(stem, _)| stem)
        .collect();
    assert!(
        unlisted.is_empty(),
        "undeclared and unlisted: present in src/, never compiled, unreviewable:\n  {}",
        unlisted.join("\n  ")
    );
}
