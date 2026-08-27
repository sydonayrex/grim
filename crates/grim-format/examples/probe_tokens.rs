//! Throwaway: print special-token ids resolved by grim's tokenizer for a
//! MiniCPM5 GGUF, plus a range around 130065..=130080. Usage:
//!   cargo run --example probe_tokens -- <path-to.gguf>

use std::collections::HashMap;
use std::fs::File;
use std::process::ExitCode;

use grim_format::GgufValue;
use grim_format::tokenizer::GgufTokenizer;

fn main() -> ExitCode {
    let path = match std::env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: probe_tokens <path-to.gguf>");
            return ExitCode::from(2);
        }
    };
    let mut f = match File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("open {path}: {e}");
            return ExitCode::from(1);
        }
    };
    let gguf = grim_format::gguf::read_gguf(&mut f).map_err(|e| eprintln!("read_gguf: {e:?}"));
    let gguf = match gguf {
        Ok(g) => g,
        Err(_) => return ExitCode::from(1),
    };
    let meta: &HashMap<String, GgufValue> = &gguf.metadata;
    let tok = match GgufTokenizer::from_metadata(meta) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("from_metadata: {e:?}");
            return ExitCode::from(1);
        }
    };

    println!(
        "bos_token_id = {:?}  add_bos_token = {}  eos_token_id = {:?}",
        tok.bos_token_id, tok.add_bos_token, tok.eos_token_id
    );

    println!("\n-- tokenizer.ggml.* metadata keys --");
    let mut tkeys: Vec<&String> = meta
        .keys()
        .filter(|k| k.starts_with("tokenizer.ggml") && !k.contains("tokens"))
        .collect();
    tkeys.sort();
    for k in tkeys {
        println!("{k} = {:?}", meta.get(k).unwrap());
    }
    if let Some(GgufValue::String(s)) = meta.get("tokenizer.chat_template") {
        std::fs::write("/tmp/minicpm5_chat_template.jinja", s).expect("write template");
        println!(
            "\nwrote chat template to /tmp/minicpm5_chat_template.jinja ({} chars)",
            s.len()
        );
    }
    println!("\n-- general.* metadata keys --");
    let mut gkeys: Vec<&String> = meta.keys().filter(|k| k.starts_with("general.")).collect();
    gkeys.sort();
    for k in gkeys {
        println!("{k} = {:?}", meta.get(k).unwrap());
    }

    println!("\n-- tokens around 130060..=130080 --");
    for id in 130060u32..=130080u32 {
        if let Some(s) = tok.tokens.get(id as usize) {
            println!("tokens[{id}] = {s:?}");
        }
    }

    println!("\n-- token_to_id lookup --");
    let syms: [&str; 14] = [
        "<|im_start|>",
        "<|im_end|>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "assistant",
        "user",
        "system",
        "<s>",
        "</s>",
        "<pad>",
        "<|unk|>",
    ];
    for sym in syms {
        println!("token_to_id[{sym:?}] = {:?}", tok.token_to_id.get(sym));
    }
    ExitCode::SUCCESS
}
