use grim_format::gguf::read_gguf;
use std::fs::File;
fn main() {
    let mut f = File::open(std::env::args().nth(1).unwrap()).unwrap();
    let g = read_gguf(&mut f).unwrap();
    for (k, v) in &g.metadata {
        if k.starts_with("tokenizer") {
            match v {
                grim_format::gguf::GgufValue::Array(a) => {
                    println!("{k} = Array(len={})", a.len());
                }
                other => println!("{k} = {other:?}"),
            }
        }
    }
}
