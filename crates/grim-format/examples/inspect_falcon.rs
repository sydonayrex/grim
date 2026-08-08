use std::collections::HashMap;
use std::fs::File;
use grim_format::gguf::read_gguf;
use grim_format::GgufValue;

fn main() {
    let mut f = File::open(std::env::args().nth(1).unwrap()).unwrap();
    let gguf = read_gguf(&mut f).unwrap();
    let meta: &HashMap<String, GgufValue> = &gguf.metadata;
    let mut keys: Vec<&String> = meta.keys().collect();
    keys.sort();
    for k in keys {
        if k.starts_with("tokenizer.") {
            continue;
        }
        let v = meta.get(k).unwrap();
        let shown = match v {
            GgufValue::String(s) => {
                if s.len() > 200 {
                    format!("{s:?}...({} chars)", s.chars().count())
                } else {
                    format!("{s:?}")
                }
            }
            GgufValue::Array(_) => "(array skipped)".to_string(),
            other => format!("{other:?}"),
        };
        println!("{k} = {shown}");
    }
    println!("=== SSM + attn + ffn tensors blk.0 shapes ===");
    for t in &gguf.tensors {
        if t.name.starts_with("blk.0.") {
            let dims: Vec<String> = t.shape().iter().map(|d| d.to_string()).collect();
            println!("{} {:?} [{}]", t.name, t.dtype, dims.join(", "));
        }
    }
    for t in &gguf.tensors {
        if t.name.starts_with("token_embd") || t.name.starts_with("output") {
            let dims: Vec<String> = t.shape().iter().map(|d| d.to_string()).collect();
            println!("{} {:?} [{}]", t.name, t.dtype, dims.join(", "));
        }
    }
}
