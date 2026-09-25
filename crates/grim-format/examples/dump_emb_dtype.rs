use grim_format::gguf::read_gguf;
use std::fs::File;
fn main() {
    let p = std::env::args().nth(1).unwrap();
    let mut f = File::open(p).unwrap();
    let g = read_gguf(&mut f).unwrap();
    for t in &g.tensors {
        if t.name == "token_embd.weight" || t.name == "output.weight" {
            println!("{}: shape={:?} dtype={:?} bytes={}", t.name, t.shape(), t.dtype, t.size_bytes);
        }
    }
}
