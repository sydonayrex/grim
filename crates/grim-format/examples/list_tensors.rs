//! List GGUF tensor names with counts per pattern.
/// Usage: `cargo run --example list_tensors -- <path-to.gguf>`
use std::collections::BTreeMap;
use std::fs::File;
use std::process::ExitCode;

use grim_format::gguf::read_gguf;

fn main() -> ExitCode {
    let path = match std::env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: list_tensors <path-to.gguf>");
            return ExitCode::from(2);
        }
    };
    let mut f = match File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("open {path}: {e}");
            return ExitCode::from(1);
        }
    };
    let gguf = match read_gguf(&mut f) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("read_gguf: {e:?}");
            return ExitCode::from(1);
        }
    };

    if let Some(v) = gguf.metadata.get("general.architecture") {
        println!("general.architecture = {v:?}");
    }
    if let Some(v) = gguf.metadata.get("general.name") {
        println!("general.name = {v:?}");
    }

    let mut prefixes: BTreeMap<String, usize> = BTreeMap::new();
    for t in &gguf.tensors {
        let parts: Vec<&str> = t.name.split('.').collect();
        // normalize digits to N in each part
        let normed: Vec<String> = parts
            .iter()
            .map(|p| {
                let s: String = p
                    .chars()
                    .map(|c| if c.is_ascii_digit() { 'N' } else { c })
                    .collect();
                s
            })
            .collect();
        let pre = if normed.len() >= 2 {
            format!("{}.{}", normed[0], normed[1])
        } else {
            normed.join(".")
        };
        *prefixes.entry(pre).or_insert(0) += 1;
    }
    println!("total tensors: {}", gguf.tensors.len());
    for (k, v) in &prefixes {
        println!("{k} x{v}");
    }
    println!("=== tensors in blk.0 ===");
    for t in gguf.tensors.iter().filter(|t| t.name.starts_with("blk.0.")) {
        println!("  {}", t.name);
    }
    for t in gguf
        .tensors
        .iter()
        .filter(|t| t.name.starts_with("output") || t.name.starts_with("token_embd"))
    {
        println!("  {}", t.name);
    }
    ExitCode::SUCCESS
}
