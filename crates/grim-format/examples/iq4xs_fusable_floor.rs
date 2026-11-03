//! Task 2.6 Step 1: compute the REAL fusable-norm floor for the decode graph.
//!
//! For each layer: a fused norm+q8_1-quantize epilogue eliminates a
//! `grim_quantize_q8_1` launch iff the normed activation has AT LEAST ONE
//! consumer whose `linear_decode_quant_kind` requires q8_1. Attention site:
//! consumers attn_q/k/v. FFN site: ffn_gate/up (fused gate_up counts once).
//!
//! Format -> q81 mapping mirrors `linear_decode_quant_kind` (dot4 arch,
//! production defaults, m=1): Q4K -> u4 (w4a4); Q5K/Q3K/Q2K/IQ4XS -> q81;
//! Q6K -> q81 iff n < vocab-head cols; Q8_0 -> q8_0 act (not q8_1);
//! IQ3S/IQ1/IQ2 family -> no q81; F32/F16 -> native.
//!
//! Usage: cargo run -p grim-format --example iq4xs_fusable_floor -- <model.gguf>

use grim_format::gguf::{read_gguf, GgufDType};
use std::collections::BTreeMap;
use std::fs::File;

/// Q6K dot4 leg is refused at vocab-head scale (kquant_dot4_ok_for).
const VOCAB_HEAD_MIN_COLS: usize = 65536;

fn uses_q81(dtype: GgufDType, n_out: u64) -> bool {
    match dtype {
        GgufDType::Q5K | GgufDType::Q3K | GgufDType::Q2K | GgufDType::IQ4_XS => true,
        GgufDType::Q6K => (n_out as usize) < VOCAB_HEAD_MIN_COLS,
        // Q4_K rides the w4a4 u4 lanes in production decode (m=1).
        GgufDType::Q4K => false,
        // Q8_0 weights consume a q8_0 activation (or f32), never q8_1.
        GgufDType::Q8_0 => false,
        // IQ3_S/IQ1/IQ2 family: exact decode or scalar arms, no q8_1.
        GgufDType::IQ3_S
        | GgufDType::IQ1_S
        | GgufDType::IQ1_M
        | GgufDType::IQ2_S
        | GgufDType::IQ2_XS
        | GgufDType::IQ2_XXS
        | GgufDType::IQ3_XXS => false,
        _ => false,
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: <model.gguf>");
    let mut f = File::open(&path).expect("open gguf");
    let g = read_gguf(&mut f).expect("read gguf");

    // per-layer: name suffix -> (dtype, n_out = dims[0])
    let mut layers: BTreeMap<usize, Vec<(String, GgufDType, u64)>> = BTreeMap::new();
    for t in &g.tensors {
        let parts: Vec<&str> = t.name.split('.').collect();
        if parts.first() != Some(&"blk") {
            continue;
        }
        let layer: usize = match parts.get(1).and_then(|s| s.parse::<usize>().ok()) {
            Some(l) => l,
            None => continue,
        };
        let suffix = parts[2..].join(".");
        // n_out = first dim in GGUF (ne[0])
        layers.entry(layer).or_default().push((suffix, t.dtype, t.dims[0]));
    }

    let n_layers = layers.len();
    let mut attn_fusable = 0usize;
    let mut ffn_fusable = 0usize;
    let mut attn_detail: Vec<String> = Vec::new();
    let mut ffn_detail: Vec<String> = Vec::new();

    for (l, ts) in &layers {
        let get = |suffix: &str| -> Option<(GgufDType, u64)> {
            ts.iter()
                .find(|(n, _, _)| n == suffix || n.starts_with(&format!("{suffix}")))
                .map(|(_, d, n)| (*d, *n))
        };
        // Attention site: any of q/k/v requires q8_1.
        let attn: Vec<bool> = ["attn_q", "attn_k", "attn_v"]
            .iter()
            .filter_map(|s| get(s))
            .map(|(d, n)| uses_q81(d, n))
            .collect();
        let a = attn.iter().any(|&b| b);
        if a {
            attn_fusable += 1;
        }
        attn_detail.push(format!(
            "layer {l}: attn q81={} [{:?}]",
            a,
            ["attn_q", "attn_k", "attn_v"]
                .iter()
                .filter_map(|s| get(s).map(|(d, _)| format!("{s}={d:?}")))
                .collect::<Vec<_>>()
        ));
        // FFN site: gate or up requires q8_1.
        let ffn: Vec<bool> = ["ffn_gate", "ffn_up"]
            .iter()
            .filter_map(|s| get(s))
            .map(|(d, n)| uses_q81(d, n))
            .collect();
        let f = ffn.iter().any(|&b| b);
        if f {
            ffn_fusable += 1;
        }
        ffn_detail.push(format!(
            "layer {l}: ffn q81={} [{:?}]",
            f,
            ["ffn_gate", "ffn_up"]
                .iter()
                .filter_map(|s| get(s).map(|(d, _)| format!("{s}={d:?}")))
                .collect::<Vec<_>>()
        ));
    }

    println!("layers: {n_layers}");
    println!("attn sites fusable (first-consumer q8_1): {attn_fusable}/{n_layers}");
    println!("ffn  sites fusable:                       {ffn_fusable}/{n_layers}");
    println!("FLOOR: fused norm+quant eliminates at most {} of 217 quantize_q8_1 launches",
             attn_fusable + ffn_fusable);
    println!("target quantize count after fusion: {}", 217 - (attn_fusable + ffn_fusable));
    if std::env::var_os("FLOOR_DETAIL").is_some() {
        for d in &attn_detail { println!("{d}"); }
        for d in &ffn_detail { println!("{d}"); }
    }
}
