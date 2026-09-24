//! No-dead-docs gate for `docs/debug-vars.md`.
//!
//! Every `GRIM_*` variable documented there must still be read somewhere
//! under `crates/` (any `std::env::var("GRIM_X")` / `var_os` / `getenv`
//! spelling — the scan is a plain substring search over `.rs` sources).
//! If a variable is removed or renamed in code without updating the doc,
//! this test fails with the stale names.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Extract `GRIM_[A-Z0-9_]+` tokens from prose. Requires at least one
/// suffix character so a bare `GRIM_*` wildcard never becomes an entry.
fn documented_vars(doc: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let bytes = doc.as_bytes();
    let mut i = 0;
    while i + 5 < bytes.len() {
        if &bytes[i..i + 5] == b"GRIM_" {
            let mut j = i + 5;
            while j < bytes.len() && (bytes[j].is_ascii_uppercase() || bytes[j].is_ascii_digit() || bytes[j] == b'_') {
                j += 1;
            }
            if j > i + 5 {
                if let Ok(name) = std::str::from_utf8(&bytes[i..j]) {
                    out.insert(name.to_string());
                }
            }
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

fn collect_rs(dir: &Path, acc: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir).expect("read crates dir");
    for entry in entries {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.is_dir() {
            collect_rs(&path, acc);
        } else if path.extension().is_some_and(|e| e == "rs") {
            acc.push(path);
        }
    }
}

#[test]
fn documented_vars_are_all_read_in_sources() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root above crates/grim-cli");
    let doc_path = root.join("docs/debug-vars.md");
    let doc = std::fs::read_to_string(&doc_path).expect("read docs/debug-vars.md");
    let vars = documented_vars(&doc);
    assert!(
        vars.len() >= 124,
        "debug-vars.md documents {} vars, expected at least 124",
        vars.len()
    );

    let crates_dir = root.join("crates");
    let mut rs_files = Vec::new();
    collect_rs(&crates_dir, &mut rs_files);
    assert!(!rs_files.is_empty(), "no .rs files found under crates/");

    let mut missing = Vec::new();
    for var in &vars {
        let mut found = false;
        for path in &rs_files {
            // Read as lossy: sources are UTF-8; a single unreadable file
            // must not fail the gate, it just cannot confirm this var.
            if let Ok(text) = std::fs::read(path) {
                let text = String::from_utf8_lossy(&text);
                if text.contains(var.as_str()) {
                    found = true;
                    break;
                }
            }
        }
        if !found {
            missing.push(var.clone());
        }
    }
    assert!(
        missing.is_empty(),
        "documented but never read under crates/: {missing:?}"
    );
}
