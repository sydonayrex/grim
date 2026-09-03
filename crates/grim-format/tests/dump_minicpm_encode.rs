use grim_format::{GgufTokenizer, read_gguf};
use std::fs::File;
#[test]
fn dump() {
    let f = read_gguf(File::open("/drive/bigfast/grim/models/MiniCPM5-1B-Q4_K_M.gguf").unwrap()).unwrap();
    let tok = GgufTokenizer::from_metadata(&f.metadata).unwrap();
    let text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n";
    let ids = tok.encode(&text);
    println!("ID COUNT: {}", ids.len());
    println!("FIRST 16 IDS: {:?}", &ids[..ids.len().min(16)]);
    for probe in ["<|im_start|>", "<|im_end|>", "hello", "You"] {
        println!("probe {:?} -> {:?}", probe, tok.encode(probe));
    }
    println!("<|im_start|> id = {:?}", tok.token_to_id.get("<|im_start|>"));
    println!("add_bos={:?} bos_id={:?}", tok.add_bos_token, tok.bos_token_id);
}
#[test]
fn probe_nl() {
    let f = read_gguf(File::open("/drive/bigfast/grim/models/MiniCPM5-1B-Q4_K_M.gguf").unwrap()).unwrap();
    let tok = GgufTokenizer::from_metadata(&f.metadata).unwrap();
    println!("user\\nhello -> {:?}", tok.encode("user\nhello"));
    println!("a\\nb -> {:?}", tok.encode("a\nb"));
    println!("newline token -> {:?}", tok.encode("\n"));
    println!("id 220 -> {:?}", tok.decode(&[220]));
    println!("id 59800 -> {:?}", tok.decode(&[59800]));
    println!("id 17261 -> {:?}", tok.decode(&[17261]));
}
#[test]
fn probe2() {
    let f = read_gguf(File::open("/drive/bigfast/grim/models/MiniCPM5-1B-Q4_K_M.gguf").unwrap()).unwrap();
    let tok = GgufTokenizer::from_metadata(&f.metadata).unwrap();
    for id in [40837u32, 59800, 2311, 457, 280, 12088, 18495, 35, 1945] {
        println!("id {id} -> {:?}", tok.decode(&[id]));
    }
}
#[test]
fn probe3() {
    let f = read_gguf(File::open("/drive/bigfast/grim/models/MiniCPM5-1B-Q4_K_M.gguf").unwrap()).unwrap();
    let tok = GgufTokenizer::from_metadata(&f.metadata).unwrap();
    for s in ["<|startoftext|>", "<s>", "</s>", "<|im_start|>", "[INST]"] {
        println!("vocab {:?} = {:?}", s, tok.token_to_id.get(s));
    }
    println!("bos_token_id field = {:?}", tok.bos_token_id);
    println!("eos_token_id field = {:?}", tok.eos_token_id);
    println!("full encode with bos? -> first id of template = 130072 (im_start)");
}
