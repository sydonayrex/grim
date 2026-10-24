//! Tokenize text with a GGUF's tokenizer and print the ids and pieces.
//! Usage: `cargo run --release --example gguf_tokenize -- <file.gguf> <text>`

use std::fs::File;

use grim_format::gguf::read_gguf;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: gguf_tokenize <file.gguf> <text>");
    let text = args.collect::<Vec<_>>().join(" ");
    let mut f = File::open(&path).expect("open");
    let g = read_gguf(&mut f).expect("read_gguf");
    let tok = grim_format::tokenizer::GgufTokenizer::from_metadata(&g.metadata).expect("tokenizer");
    let ids = tok.encode(&text);
    println!("{} tokens: {:?}", ids.len(), ids);
    let pieces: Vec<String> = ids
        .iter()
        .map(|&i| {
            if (i as usize) < tok.tokens.len() {
                tok.tokens[i as usize].clone()
            } else {
                String::from("<oob>")
            }
        })
        .collect();
    println!("pieces: {:?}", pieces);
}
