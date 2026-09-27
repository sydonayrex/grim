// Print every tensor name belonging to one layer, so a "no .wq/.wk/.wv"
// result can be read against what the checkpoint DOES carry instead of
// guessed at.
//
// Usage: cargo run -p grim-format --example dump_layer_tensors -- <model.gguf> [layer]

use grim_format::gguf::read_gguf;
use std::fs::File;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: dump_layer_tensors <model.gguf> [layer]");
    let want: Option<usize> = args.next().and_then(|s| s.parse().ok());

    let mut f = File::open(&path).expect("open gguf");
    let g = read_gguf(&mut f).expect("read gguf");

    for t in &g.tensors {
        let parts: Vec<&str> = t.name.split('.').collect();
        let layer: Option<usize> = parts.get(1).and_then(|s| s.parse().ok());
        if let Some(w) = want {
            if layer != Some(w) {
                continue;
            }
        }
        println!(
            "{:>6}  {:?}  {}",
            layer.map_or_else(|| "-".into(), |l| l.to_string()),
            t.dtype,
            t.name
        );
    }
}
