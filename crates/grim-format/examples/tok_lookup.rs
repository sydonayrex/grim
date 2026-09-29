//! Print the token text for specific ids, so a logprob id can be named.
use std::fs::File;
use grim_format::gguf::read_gguf;
use grim_format::tokenizer::GgufTokenizer;

fn main() {
    let path = std::env::args().nth(1).expect("usage: tok_lookup <gguf> <id>...");
    let ids: Vec<u32> = std::env::args().skip(2).filter_map(|a| a.parse().ok()).collect();
    let mut f = File::open(&path).expect("open gguf");
    let g = read_gguf(&mut f).expect("read_gguf");
    let meta: std::collections::HashMap<String, grim_format::GgufValue> =
        g.metadata.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    let tok = GgufTokenizer::from_metadata(&meta).expect("tokenizer");
    // `--` prefix means "encode this text and report the ids".
    for a in std::env::args().skip(2) {
        if let Some(text) = a.strip_prefix("--") {
            println!("encode {:?} -> ids {:?}", text, tok.encode(text));
            continue;
        }
        if let Ok(id) = a.parse::<u32>() {
            println!("id={:<8} {:?}", id, tok.decode(&[id]));
        }
    }
    let _ = ids;
}
