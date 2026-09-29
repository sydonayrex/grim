//! IS EVERY K-QUANT WEIGHT BEING READ TRANSPOSED?
//!
//! The head test found `output.weight` reads A = row j of [out, in] and
//! B = column j of [in, out] (its ggml dims are [4096, 248320], so ne0 is the
//! HIDDEN dim and the blob is [in][out]). Reading A is what the model uses and
//! it lands 13.98 from Ollama's logprob gap where Ollama says -2.37. Reading B
//! lands 3.51 and correctly ranks " known"/" London" below " Paris". Neither
//! matches, which is what you expect if EVERY weight has the same problem: the
//! hidden state feeding the head is itself built from transposed weights.
//!
//! It could not have been caught earlier. Every weight-level gate tonight
//! compared grim against an oracle derived from grim's OWN dequantised weight,
//! so a CONSISTENT transpose cancels in every comparison.
//!
//! THE TEST: rebuild all 32 layers with reading-B weights — `W'[j*in + k] =
//! w[k*out + j]`, out = ne1, in = ne0 — and run the whole forward, then the head,
//! and look for " Paris" (id 11751), which Ollama puts at rank 0 with logprob
//! -0.46090.
//!
//! `ssm_conv1d` is deliberately NOT transposed: it is read as `cw[ch*4 + tap]`,
//! which IS `tap + d_conv*ch` in ggml terms (`ggml_ssm_conv` takes
//! `d_conv = c->ne[0]`), so it is already correct (commit b48501ef).
//!
//! CPU path, one layer resident at a time. Gated: needs the 9B checkpoint.

use grim_format::gguf::{GgufDType, read_gguf, read_tensor_bytes};
use grim_models_transformer::qwen35::{Qwen35Block, Qwen35Config, Qwen35LayerCache};
use grim_nn::modules::Linear;
use grim_tensor::{Device, Shape};

const HIDDEN: usize = 4096;
const NV: usize = 32;
const NK: usize = 16;
const HD: usize = 128;
const TAPS: usize = 4;
const S: usize = 5;
const OLLAMA_PARIS_LOGPROB: f64 = -0.46090;

fn model_path() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("GRIM_CHECKPOINT") {
        let p = std::path::PathBuf::from(p);
        if p.exists() { return Some(p); }
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    for up in ["../../..", "../..", ".."] {
        let p = root.join(up).join("models/qwen35-9b/Qwen3.5-9B-Q4_K_M.gguf");
        if p.exists() { return Some(p); }
    }
    eprintln!("[SKIP] 9B checkpoint not found");
    None
}

fn cfg() -> Qwen35Config {
    let mut c = Qwen35Config::default();
    c.vocab_size = 248320;
    c.hidden_size = HIDDEN;
    c.num_heads = 16;
    c.num_kv_heads = 4;
    c.head_dim = 256;
    c.num_layers = 32;
    c.intermediate_size = 12288;
    c.full_attention_interval = 4;
    c.ssm_d_conv = TAPS;
    c.ssm_d_state = HD;
    c.ssm_dt_rank = NV;
    c.ssm_n_group = NK;
    c.ssm_d_inner = NV * HD;
    c.devices = Vec::new();
    c
}

/// Dequantise one GGUF tensor with the VERIFIED decoders. Returns
/// `(flat, ne0, ne1)` where the blob is row-major `[ne0][ne1]`.
fn load_tensor(
    reader: &mut std::io::BufReader<std::fs::File>,
    file: &grim_format::gguf::GgufFile,
    name: &str,
) -> Option<(Vec<f32>, usize, usize)> {
    let t = file.tensors.iter().find(|t| t.name == name)?;
    let b = read_tensor_bytes(reader, file, t).ok()?;
    let n = t.dims[0] as usize * t.dims[1] as usize;
    let v = match t.dtype {
        GgufDType::Q4K => grim_quant::dequant_q4k(&b, n).ok()?,
        GgufDType::Q5K => grim_quant::dequant_q5k(&b, n).ok()?,
        GgufDType::Q6K => grim_quant::dequant_q6k(&b, n).ok()?,
        GgufDType::Q8_0 => grim_quant::dequant_q80(&b, n).ok()?,
        GgufDType::F32 => b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        other => { eprintln!("[read] {name}: unhandled dtype {other:?}"); return None; }
    };
    Some((v, t.dims[0] as usize, t.dims[1] as usize))
}

/// Replace one Linear's weight with `W'[j*in + k] = w[k*out + j]`.
fn transpose_into(lin: &mut Linear, flat: &[f32], ne0: usize, ne1: usize, tag: &str) {
    let (in_n, out_n) = (ne0, ne1);
    let mut w = vec![0.0f32; in_n * out_n];
    for k in 0..in_n {
        for j in 0..out_n {
            w[j * in_n + k] = flat[k * out_n + j];
        }
    }
    let t = grim_backend_cpu::cpu_tensor(w, Shape::new(vec![out_n, in_n]));
    *lin = Linear::from_tensor(t, None);
    eprintln!("[read] {tag}: re-read as [out={out_n}, in={in_n}] from a [{ne0},{ne1}] blob");
}

#[test]
fn whole_stack_with_transposed_weights_predicts_paris() {
    let Some(path) = model_path() else { return };
    let prov = grim_format::tprov::GgufProvider::open(path.to_str().expect("utf8 path"))
        .expect("open 9B");
    let mut c = cfg();
    c.rotary_dim = Some(64);
    let positions: Vec<u32> = (0..S as u32).collect();
    let v = c.vocab_size;

    // Real embedding, reading A (the model's).
    let mut h: Vec<f32> = {
        use std::io::BufReader;
        let f = std::fs::File::open(&path).expect("open");
        let mut r = BufReader::new(f);
        let g = read_gguf(&mut r).expect("gguf");
        let (tbl, ne0, ne1) = load_tensor(&mut r, &g, "token_embd.weight").expect("embd");
        eprintln!("[read] token_embd blob [{ne0},{ne1}] -> reading A rows of {ne0}");
        let mut out = vec![0.0f32; S * HIDDEN];
        for (i, &tok) in [561u32, 6511, 314, 9338, 369].iter().enumerate() {
            let st = tok as usize * ne0;
            out[i * HIDDEN..(i + 1) * HIDDEN].copy_from_slice(&tbl[st..st + HIDDEN]);
        }
        out
    };

    for k in 0..32usize {
        let pref = format!("blk.{k}");
        let ws = grim_nn::WeightSource::root(&prov, Device::Cpu).pp(&pref);
        let mut blk = Qwen35Block::load_tp(&ws, &c, k, grim_nn::TensorParallelConfig::default())
            .unwrap_or_else(|e| panic!("{pref}: {e}"));

        // Re-read every 2-D weight in the block under reading B. A helper fn
        // rather than closures: both closures would need `&mut r`.
        {
            use std::io::BufReader;
            let f = std::fs::File::open(&path).expect("open");
            let mut r = BufReader::new(f);
            let g = read_gguf(&mut r).expect("gguf");
            fn fix_opt(
                r: &mut std::io::BufReader<std::fs::File>,
                g: &grim_format::gguf::GgufFile,
                pref: &str,
                lin: &mut Option<Linear>,
                nm: &str,
            ) {
                let full = format!("{pref}.{nm}.weight");
                if let Some((flat, ne0, ne1)) = load_tensor(r, g, &full) {
                    if let Some(l) = lin.as_mut() { transpose_into(l, &flat, ne0, ne1, &full); }
                }
            }
            fn fix_val(
                r: &mut std::io::BufReader<std::fs::File>,
                g: &grim_format::gguf::GgufFile,
                pref: &str,
                lin: &mut Linear,
                nm: &str,
            ) {
                let full = format!("{pref}.{nm}.weight");
                if let Some((flat, ne0, ne1)) = load_tensor(r, g, &full) {
                    transpose_into(lin, &flat, ne0, ne1, &full);
                }
            }
            if blk.is_full_attention {
                fix_opt(&mut r, &g, &pref, &mut blk.wq, "attn_q");
                fix_opt(&mut r, &g, &pref, &mut blk.wk, "attn_k");
                fix_opt(&mut r, &g, &pref, &mut blk.wv, "attn_v");
                fix_opt(&mut r, &g, &pref, &mut blk.wo, "attn_output");
            } else {
                fix_opt(&mut r, &g, &pref, &mut blk.attn_qkv, "attn_qkv");
                fix_opt(&mut r, &g, &pref, &mut blk.attn_gate, "attn_gate");
                fix_opt(&mut r, &g, &pref, &mut blk.ssm_out, "ssm_out");
                fix_opt(&mut r, &g, &pref, &mut blk.ssm_alpha, "ssm_alpha");
                fix_opt(&mut r, &g, &pref, &mut blk.ssm_beta, "ssm_beta");
            }
            fix_val(&mut r, &g, &pref, &mut blk.ffn_gate, "ffn_gate");
            fix_val(&mut r, &g, &pref, &mut blk.ffn_up, "ffn_up");
            fix_val(&mut r, &g, &pref, &mut blk.ffn_down, "ffn_down");
        }

        let mut cache = Qwen35LayerCache::new(&c);
        let t = grim_backend_cpu::cpu_tensor(h.clone(), Shape::new(vec![S, HIDDEN]));
        h = blk.forward(&t, &positions, &mut cache).expect("fwd").to_vec_f32().expect("read");
    }

    // Head, also under reading B.
    let ws_out = grim_nn::WeightSource::root(&prov, Device::Cpu);
    let norm = grim_nn::modules::RmsNorm::load(&ws_out.pp("output_norm"), HIDDEN, c.rms_norm_eps)
        .expect("output_norm");
    let last = &h[(S - 1) * HIDDEN..S * HIDDEN];
    let lt = grim_backend_cpu::cpu_tensor(last.to_vec(), Shape::new(vec![1, HIDDEN]));
    let normed = norm.forward(&lt).expect("norm").to_vec_f32().expect("read");

    let head = {
        use std::io::BufReader;
        let f = std::fs::File::open(&path).expect("open");
        let mut r = BufReader::new(f);
        let g = read_gguf(&mut r).expect("gguf");
        let (flat, ne0, ne1) = load_tensor(&mut r, &g, "output.weight").expect("output.weight");
        eprintln!("[read] output.weight blob [{ne0},{ne1}]");
        let mut w = vec![0.0f32; ne0 * ne1];
        for k in 0..ne0 { for j in 0..ne1 { w[j * ne0 + k] = flat[k * ne1 + j]; } }
        let t = grim_backend_cpu::cpu_tensor(w, Shape::new(vec![ne1, ne0]));
        Linear::from_tensor(t, None)
    };
    let bt = grim_backend_cpu::cpu_tensor(normed, Shape::new(vec![1, HIDDEN]));
    let logits = head.forward(&bt).expect("head").to_vec_f32().expect("read");
    assert_eq!(logits.len(), v);

    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let lse = logits.iter().map(|x| (*x as f64).exp()).sum::<f64>().ln() + mx as f64;
    let mut idx: Vec<u32> = (0..v as u32).collect();
    idx.sort_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
    eprintln!("[b] top5 with ALL weights re-read as [out,in] from [in,out] blobs:");
    for i in idx.iter().take(5) {
        eprintln!(
            "[b]   id={i} logprob={:.5}",
            logits[*i as usize] as f64 - lse
        );
    }
    let rank = idx.iter().position(|&i| i == 11751).unwrap_or(usize::MAX);
    let lp = logits[11751] as f64 - lse;
    eprintln!(
        "[b] \" Paris\" (11751): rank {}, logprob {:.5}  (llama.cpp via Ollama: rank 0, logprob {:.5})",
        rank, lp, OLLAMA_PARIS_LOGPROB
    );
    assert_eq!(
        rank, 0,
        "with EVERY weight re-read transposed, \" Paris\" is still not the top token \
         (rank {}, logprob {:.5} vs llama.cpp's {:.5}). So a consistent weight transpose is \
         NOT the gibberish.",
        rank, lp, OLLAMA_PARIS_LOGPROB
    );
}
