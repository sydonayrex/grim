// Dump the tensor dtypes that gate the qwen35 D2D decode paths.
//
// `gated_delta_net_forward_d2d` declines unless `ssm_conv1d.weight` is already
// plain f32 on the device (`Storage::Native`), and `attention_layer_d2d`
// declines unless the full-attention layers carry separated wq/wk/wv. This
// example prints exactly those facts straight from the GGUF, so the "is the
// fallback firing because the checkpoint is quantized?" question is answered by
// the file rather than by inference.
//
// Usage: cargo run -p grim-format --example dump_d2d_gate_tensors -- <model.gguf>

use grim_format::gguf::{GgufDType, read_gguf};
use std::fs::File;

fn main() {
    let path = match std::env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: dump_d2d_gate_tensors <model.gguf>");
            std::process::exit(2);
        }
    };
    let mut f = File::open(&path).expect("open gguf");
    let g = read_gguf(&mut f).expect("read gguf");

    let mut conv_dtypes = Vec::new();
    let mut conv_shapes = Vec::new();
    let mut per_head_dtypes: Vec<(String, GgufDType)> = Vec::new();
    let mut attn_qkv: Vec<(usize, GgufDType, Vec<usize>)> = Vec::new();
    let mut full_attn_proj: Vec<(usize, String, GgufDType)> = Vec::new();

    for t in &g.tensors {
        let n = &t.name;
        let parts: Vec<&str> = n.split('.').collect();
        let layer: usize = parts
            .get(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(usize::MAX);
        let kind = t.dtype;

        if n.ends_with("ssm_conv1d.weight") {
            conv_dtypes.push(kind);
            conv_shapes.push(t.shape().to_vec());
        }
        // Tiny per-head vectors: the loader to_vec_f32()s these, so their GGUF
        // type only matters for confirming they are not unexpectedly huge.
        for leaf in [
            "ssm_a",
            "ssm_dt.bias",
            "ssm_norm.weight",
            "ssm_alpha.weight",
            "ssm_beta.weight",
            "attn_gate.weight",
            "ssm_out.weight",
        ] {
            if n.ends_with(leaf) && layer != usize::MAX {
                per_head_dtypes.push((n.clone(), kind));
            }
        }
        if n.ends_with("attn_qkv.weight") {
            attn_qkv.push((layer, kind, t.shape().to_vec()));
        }
        // Separated projections on a FULL-attention layer are what
        // `attention_layer_d2d` requires.
        if layer != usize::MAX
            && (n.ends_with(".wq.weight") || n.ends_with(".wk.weight") || n.ends_with(".wv.weight"))
        {
            full_attn_proj.push((layer, n.clone(), kind));
        }
    }

    println!("model: {path}");
    println!();

    println!("== ssm_conv1d.weight (KDA D2D requires Native/f32) ==");
    let mut sorted: Vec<String> = conv_dtypes.iter().map(|k| format!("{k:?}")).collect();
    sorted.sort();
    sorted.dedup();
    println!("  count: {}", conv_dtypes.len());
    println!("  dtypes present: {sorted:?}");
    let all_f32 = !conv_dtypes.is_empty() && conv_dtypes.iter().all(|k| *k == GgufDType::F32);
    println!("  ALL F32 (D2D can accept as loaded): {all_f32}");
    if let Some(s) = conv_shapes.first() {
        println!("  shape[0]: {s:?}");
    }
    println!();

    println!("== attn_qkv.weight (recurrent-layer fused projection) ==");
    for (l, k, s) in attn_qkv.iter().take(4) {
        println!("  layer {l:>2}: {k:?} {s:?}");
    }
    println!("  total: {}", attn_qkv.len());
    println!();

    println!("== separated wq/wk/wv on full-attention layers ==");
    if full_attn_proj.is_empty() {
        println!("  NONE. The checkpoint has no separated .wq/.wk/.wv tensors,");
        println!("  so attention_layer_d2d declines on every layer, every token.");
    } else {
        let mut by_layer: std::collections::BTreeMap<usize, Vec<String>> = Default::default();
        for (l, n, _) in &full_attn_proj {
            by_layer.entry(*l).or_default().push(n.clone());
        }
        for (l, names) in &by_layer {
            println!("  layer {l:>2}: {} tensors", names.len());
        }
    }
    println!();

    println!("== per-head / small tensors (loader dequantizes these) ==");
    let mut seen = std::collections::BTreeSet::new();
    for (n, k) in &per_head_dtypes {
        let leaf = n.split('.').nth(2).unwrap_or(n).to_string();
        if seen.insert(leaf.clone()) {
            println!("  {leaf:<20} {k:?}");
        }
    }
}
