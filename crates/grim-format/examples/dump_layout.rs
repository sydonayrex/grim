use grim_format::gguf::read_gguf;
use std::fs::File;
fn main() {
    let p = std::env::args().nth(1).unwrap();
    let mut f = File::open(p).unwrap();
    let g = read_gguf(&mut f).unwrap();
    println!("--- metadata (head/attn/ssm/split) ---");
    for (k, v) in &g.metadata {
        let kk = k.to_lowercase();
        if kk.contains("head")
            || kk.contains("ssm")
            || kk.contains("attn")
            || kk.contains("split")
            || kk.contains("general")
        {
            println!("  {k} = {v:?}");
        }
    }
    println!("--- root tensors ---");
    for t in &g.tensors {
        if !t.name.starts_with("blk.") {
            println!("  {:<30} {:?}", t.name, t.shape());
        }
    }
    for t in &g.tensors {
        if t.name.starts_with("blk.0.") {
            println!(
                "  {:<28} {:?}",
                t.name.trim_start_matches("blk.0."),
                t.shape()
            );
        }
    }
    println!("--- layer 3 (full attention, per (i+1)%4==0) ---");
    for t in &g.tensors {
        if t.name.starts_with("blk.3.") {
            println!(
                "  {:<28} {:?}",
                t.name.trim_start_matches("blk.3."),
                t.shape()
            );
        }
    }
}
