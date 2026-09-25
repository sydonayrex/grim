//! S4 corpus prep (GRAVE plan §3 Task S4): turn a directory of downloaded
//! plain-text documents (Project Gutenberg cache dumps) into the training
//! corpus format the distill trainers expect: one UTF-8 file, documents
//! separated by `\n@@DOC <id>\n` markers.
//!
//! Filtering:
//!   - Project Gutenberg license headers/footers stripped (`*** START OF`/
//!     `*** END OF` markers, fallback: strip nothing and skip files without them
//!     only if they also lack any text — raw dumps without markers are kept).
//!   - Files <10 KB after stripping (front-matter, indexes) skipped.
//!   - Non-English heuristic: >25% non-ASCII bytes after stripping → skipped.
//!   - The PPL eval sample is never an input here, but as a guard we reject
//!     any input whose path contains "wikitext2.sample".
//!
//! Usage: corpus_prep <raw_dir> <out_corpus.txt>
//!
//! Rust-only per the standing project rule (no Python in grim).

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: corpus_prep <raw_dir> <out_corpus.txt>");
        std::process::exit(2);
    }
    let raw_dir = Path::new(&args[1]);
    let out_path = Path::new(&args[2]);
    if out_path.exists() {
        eprintln!(
            "refusing to overwrite existing corpus {} — delete it first if intentional",
            out_path.display()
        );
        std::process::exit(1);
    }

    let mut files: Vec<_> = std::fs::read_dir(raw_dir)
        .expect("raw_dir readable")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .collect();
    files.sort();
    let n_files = files.len();
    let mut out = std::io::BufWriter::new(std::fs::File::create(out_path).expect("create out"));
    let (mut kept, mut skipped_small, mut skipped_lang, mut skipped_mark, mut total_bytes) =
        (0usize, 0usize, 0usize, 0usize, 0u64);
    let mut title_seen: HashMap<String, ()> = HashMap::new();

    for path in files {
        if path.to_string_lossy().contains("wikitext2.sample") {
            skipped_mark += 1;
            continue;
        }
        let raw = match std::fs::read(&path) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let text = String::from_utf8_lossy(&raw);
        let body = strip_gutenberg(&text);
        if body.len() < 10_000 {
            skipped_small += 1;
            continue;
        }
        let non_ascii = body.bytes().filter(|b| *b > 127).count();
        if non_ascii * 4 > body.len() {
            skipped_lang += 1;
            continue;
        }
        // Near-dupe guard: the PG title from the START marker when present
        // (generic first lines like "[Illustration]" collide otherwise),
        // else a prefix of the body.
        let key: String = match text.find("*** START OF THE PROJECT GUTENBERG EBOOK ") {
            Some(i) => text[i + "*** START OF THE PROJECT GUTENBERG EBOOK ".len()..]
                .lines()
                .next()
                .unwrap_or("")
                .trim_end_matches(" ***")
                .trim()
                .to_lowercase(),
            None => body.chars().take(2000).collect(),
        };
        if !key.is_empty() && title_seen.insert(key, ()).is_some() {
            skipped_mark += 1;
            continue;
        }
        let id = path.file_stem().and_then(|s| s.to_str()).unwrap_or("doc");
        writeln!(out, "@@DOC {id}").expect("write marker");
        out.write_all(body.as_bytes()).expect("write doc");
        out.write_all(b"\n").expect("write sep");
        kept += 1;
        total_bytes += body.len() as u64;
        if kept % 1000 == 0 {
            eprintln!(
                "[corpus_prep] kept {kept} docs, {:.2} GB so far",
                total_bytes as f64 / 1e9
            );
        }
    }
    out.flush().unwrap();
    println!(
        "[corpus_prep] done: {kept} docs, {:.2} GB text (scanned {n_files} files: \
         {skipped_small} too-small, {skipped_lang} non-English, {skipped_mark} dup/mark)",
        total_bytes as f64 / 1e9
    );
}

/// Strip the Project Gutenberg license header/footer. The cache dumps carry
/// `*** START OF THE PROJECT GUTENBERG EBOOK <TITLE> ***` and a matching
/// `*** END OF ...`. Return the body between them; if either marker is
/// missing, return the whole text (still filtered by size/language).
fn strip_gutenberg(text: &str) -> String {
    let start_marker = "*** START OF";
    let end_marker = "*** END OF";
    let start = match text.find(start_marker) {
        Some(i) => match text[i..].find("\n") {
            Some(nl) => i + nl + 1,
            None => return text.to_string(),
        },
        None => return text.to_string(),
    };
    let body = &text[start..];
    match body.find(end_marker) {
        Some(j) => body[..j].to_string(),
        None => body.to_string(),
    }
}
