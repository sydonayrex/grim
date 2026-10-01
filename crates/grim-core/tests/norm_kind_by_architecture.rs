//! Which architectures use LayerNorm -- plan item 5, norm-kind row.
//!
//! ## The reference fact
//!
//! `build_norm` takes one of exactly two kinds, `LLM_NORM` (LayerNorm) or
//! `LLM_NORM_RMS`, and each reference passes the kind as a literal. There is no
//! norm-type hyperparameter: `llama-arch.cpp` has
//! `attention.layer_norm_epsilon` and `attention.layer_norm_rms_epsilon`, but
//! nothing that says which. So the kind is a property of the architecture.
//!
//! Twenty-five references pass `LLM_NORM` for a *layer* norm. Five of the
//! thirty that use `LLM_NORM` anywhere use it only for the per-head q/k norms,
//! which have their own field and are not covered here.
//!
//! ## What this test does
//!
//! Reads the references at test time and asserts grim's table agrees. A
//! hand-copied list would rot the moment llama.cpp adds a model, and a test that
//! cannot notice its own staleness is worse than none.
//!
//! Twenty of the twenty-five exist as grim `ModelArchitecture` variants; the
//! other five (`command_r`, `modern_bert`, `nemotron_h_moe`, `pockettts`,
//! `wavtokenizer_dec`) are not implemented yet, so they are excluded rather
//! than asserted as gaps.
//!
//! ## The parser
//!
//! Newlines are squeezed before matching, because `build_norm(` calls wrap
//! over several lines and a single-line matcher misses them. That undercount is
//! not hypothetical: it made an earlier count of this same list read 12 instead
//! of 30, and the error was in the pattern, not the reference.

use grim_core::ModelArchitecture;
use std::collections::BTreeSet;
use std::path::PathBuf;

fn reference_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("old/repo/llama.cpp-master/src/models")
}

/// Reference models whose *layer* norm is LayerNorm.
///
/// A `build_norm` call on `attn_q_norm` / `attn_k_norm` is not a layer norm and
/// is excluded: those tensors have their own spec field, and a model may
/// LayerNorm its q/k norms while keeping RMS elsewhere (chameleon does).
fn reference_layernorm_models() -> BTreeSet<String> {
    let dir = reference_dir();
    let mut out = BTreeSet::new();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return out;
    };
    for e in entries.flatten() {
        let path = e.path();
        if path.extension().and_then(|s| s.to_str()) != Some("cpp") {
            continue;
        }
        let Ok(raw) = std::fs::read_to_string(&path) else {
            continue;
        };
        let flat = raw.replace('\n', " ");
        for call in flat.split("build_norm(").skip(1) {
            let call = &call[..call.find(");").map_or(call.len(), |i| i + 2)];
            // Word-bounded so `LLM_NORM_RMS,` does not match `LLM_NORM,`.
            if !call.contains("LLM_NORM,") || call.contains("LLM_NORM_RMS,") {
                continue;
            }
            let qk = ["attn_q_norm", "attn_k_norm", "q_norm", "k_norm"]
                .iter()
                .any(|n| call.contains(n));
            if !qk {
                out.insert(
                    path.file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or_default()
                        .to_string(),
                );
                break;
            }
        }
    }
    out
}

/// grim variant name -> reference file stem, for the LayerNorm models.
const LAYERNORM_ARCHS: &[(&str, &str)] = &[
    ("Bert", "bert"),
    ("Bloom", "bloom"),
    ("Codeshell", "codeshell"),
    ("Cohere2", "cohere2"),
    ("Dbrx", "dbrx"),
    ("Falcon", "falcon"),
    ("Gpt2", "gpt2"),
    ("GptNeoX", "gptneox"),
    ("Jais", "jais"),
    ("Jais2", "jais2"),
    ("Mpt", "mpt"),
    ("Nemotron", "nemotron"),
    ("Olmo", "olmo"),
    ("Orion", "orion"),
    ("Phi2", "phi2"),
    ("Rwkv6", "rwkv6"),
    ("Rwkv7", "rwkv7"),
    ("StableLm", "stablelm"),
    ("Starcoder", "starcoder"),
    ("Starcoder2", "starcoder2"),
];

#[test]
fn the_reference_is_readable_and_the_parser_finds_the_known_models() {
    let dir = reference_dir();
    assert!(
        dir.exists(),
        "{} missing -- this test cannot check anything without the reference, \\
         and passing on an empty read would be worse than failing",
        dir.display()
    );
    let found = reference_layernorm_models();
    assert!(
        found.len() >= 25,
        "only {} references read as layer-norm LayerNorm; expected 25. The \\
         parser is broken, not the reference.",
        found.len()
    );
    for (_, stem) in LAYERNORM_ARCHS {
        assert!(
            found.contains(*stem),
            "{} should use LayerNorm per the reference but the parser did not \\
             find it; the list in LAYERNORM_ARCHS may be stale",
            stem
        );
    }
}

#[test]
fn every_layernorm_architecture_is_marked_in_grim() {
    // RED until `ModelArchitecture::uses_layernorm()` exists and is correct.
    // Each miss is a model computing the wrong normalisation: LayerNorm
    // subtracts the mean, RMS does not, and on a non-zero mean they disagree.
    let missed: Vec<&str> = LAYERNORM_ARCHS
        .iter()
        .filter(|(name, _)| !ModelArchitecture::from_str(name).uses_layernorm())
        .map(|(name, _)| *name)
        .collect();
    assert!(
        missed.is_empty(),
        "these use LayerNorm per the reference but grim serves them as RMSNorm:\n  {:?}",
        missed
    );
}

#[test]
fn the_marked_set_and_the_reference_agree_exactly() {
    // The other direction: a false positive is as wrong as a miss, so compare
    // sets rather than only checking membership.
    let reference = reference_layernorm_models();
    let marked: BTreeSet<&str> = LAYERNORM_ARCHS
        .iter()
        .filter(|(name, _)| ModelArchitecture::from_str(name).uses_layernorm())
        .map(|(_, stem)| *stem)
        .collect();
    let expected: BTreeSet<&str> = LAYERNORM_ARCHS.iter().map(|(_, s)| *s).collect();
    assert_eq!(
        marked, expected,
        "grim's marked set and the table disagree"
    );
    let mut extra: Vec<&str> = marked
        .iter()
        .filter(|s| !reference.contains(**s))
        .copied()
        .collect();
    extra.sort_unstable();
    assert!(
        extra.is_empty(),
        "marked LayerNorm but the reference does not: {extra:?}"
    );
}