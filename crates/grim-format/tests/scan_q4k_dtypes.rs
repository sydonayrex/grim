use grim_format::gguf::{GgufDType, read_gguf};
use std::fmt::Write;
use std::fs::File;
use std::io::BufReader;

#[test]
fn scan_q4k_model_dtypes() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .find(|p| p.join("models").is_dir())
        .expect("repo root with models/")
        .to_path_buf();
    let path = repo_root.join("models/MiniCPM5-1B-Q4_K_M.gguf");
    let f = match File::open(&path) {
        Ok(f) => f,
        Err(_) => {
            eprintln!("skip");
            return;
        }
    };
    let mut reader = BufReader::new(f);
    let file = read_gguf(&mut reader).expect("read_gguf");
    let mut counts: std::collections::HashMap<String, usize> = Default::default();
    for t in &file.tensors {
        let s = format!("{:?}", t.dtype);
        *counts.entry(s).or_insert(0) += 1;
    }
    let mut out = String::new();
    for (k, v) in counts.iter() {
        writeln!(out, "{k}: {v}").unwrap();
    }
    eprintln!("[scan]\n{out}");
    // also print any tensor whose dtype is NOT Q4K
    for t in &file.tensors {
        if t.dtype != GgufDType::Q4K {
            eprintln!("[non-q4k] {} {:?} dims={:?}", t.name, t.dtype, t.dims);
        } else {
            if t.name.contains("blk.0.") {
                eprintln!("[q4k] {} dims={:?}", t.name, t.dims);
            }
        }
    }
}
