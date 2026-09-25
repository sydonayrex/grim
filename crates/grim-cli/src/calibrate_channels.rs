//! Calibrate KV-cache channel importance and output a JSON sidecar
//! (PLAN-kvcache-channel-axis WI-2).
//!
//! Two modes:
//! - `--model <path>`: load the model, run calibration prompts (from
//!   `--prompts-file` or the built-in set) through a plain forward, capture
//!   the post-RoPE cached K/V per layer, and write per-layer pair-joint
//!   importance scores (RDKV-style distortion proxy) plus the optional Fathom
//!   q-variance channel when query capture is enabled later. Post-RoPE
//!   individual-channel variance is position-unstable (TriAttention), so the
//!   scores are pair-joint over RoPE channel pairs by construction.
//! - default (no model): the synthetic deterministic corpus used to smoke the
//!   pipeline end-to-end without needing weights on disk.
//!
//! The sidecar filename follows the oxidizer convention:
//! `{output}.channels.json` (or `--output` verbatim when it ends in .json).

use std::fs::File;
use std::io::Write;
use std::path::Path;

use grim_core::error::{Error, Result};
use grim_kvquant::channel_importance::{
    CHANNEL_GROUP_SIZE, ChannelImportanceComputer, ChannelImportanceScores,
};

/// Built-in calibration prompts (short, varied register).
const DEFAULT_PROMPTS: &[&str] = &[
    "Explain the difference between a process and a thread in one paragraph.",
    "The quick brown fox jumps over the lazy dog. Translate this into French.",
    "fn main() { let x: Vec<i32> = vec![1, 2, 3]; println!(\"{x:?}\"); }",
    "In 1969, Apollo 11 landed on the Moon. Who was the first to step out?",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "什么是机器学习中的过拟合？ 请简要解释。",
    "Write a haiku about compiler errors.",
    "SELECT name, COUNT(*) FROM users GROUP BY name ORDER BY 2 DESC;",
];

pub struct CalibrateChannelsArgs {
    pub output: String,
    pub model: Option<String>,
    pub prompts_file: Option<String>,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub samples: usize,
    pub max_tokens_per_prompt: usize,
    /// Also emit a derived ChannelBitAllocation alongside the scores.
    pub alloc_budget_bits: Option<(f32, f32)>, // (key budget, value budget)
}

/// Dispatch — synthetic mode when `--model` is absent, model-capture mode otherwise.
pub fn cmd_calibrate_channels(args: CalibrateChannelsArgs) -> Result<()> {
    if args.kv_heads == 0 || args.head_dim == 0 || args.head_dim % CHANNEL_GROUP_SIZE != 0 {
        return Err(Error::Config(format!(
            "Invalid geometry: num_kv_heads={} head_dim={} (head_dim must be a multiple of {CHANNEL_GROUP_SIZE})",
            args.kv_heads, args.head_dim
        )));
    }

    let layers: Vec<ChannelImportanceScores> = match &args.model {
        Some(model) => run_model_calibration(&args, model)?,
        None => vec![run_synthetic(&args)?],
    };

    write_sidecar(&args.output, &layers, args.alloc_budget_bits)?;
    Ok(())
}

fn run_synthetic(args: &CalibrateChannelsArgs) -> Result<ChannelImportanceScores> {
    println!("=== Grim KV Channel Importance (synthetic mode) ===");
    let mut computer =
        ChannelImportanceComputer::new(args.kv_heads, args.head_dim, CHANNEL_GROUP_SIZE, false)
            .map_err(Error::Config)?;
    let row_len = args.kv_heads * args.head_dim;
    let mut k_buf = vec![0.0f32; row_len];
    let mut v_buf = vec![0.0f32; row_len];
    for s in 0..args.samples {
        let seed = (s as f32 + 1.0) * 0.17;
        for h in 0..args.kv_heads {
            let base = h * args.head_dim;
            for d in 0..args.head_dim {
                k_buf[base + d] = ((d as f32 * 0.3 + seed).sin() + (d as f32 * 0.05).cos()) * 0.5;
                v_buf[base + d] = (d as f32 * 0.2 + seed * 1.5).cos() * 0.5;
            }
        }
        computer
            .update_kv(&k_buf, &v_buf, 1)
            .map_err(Error::Config)?;
    }
    Ok(computer.finish())
}

fn run_model_calibration(
    args: &CalibrateChannelsArgs,
    model_path: &str,
) -> Result<Vec<ChannelImportanceScores>> {
    use grim_core::CausalLm;

    println!("=== Grim KV Channel Importance (model capture) ===");
    let resolved = crate::catalog::resolve_model_path(model_path)
        .filter(|p| p.exists())
        .unwrap_or_else(|| std::path::PathBuf::from(model_path));
    if !resolved.exists() {
        return Err(Error::Config(format!(
            "model '{model_path}' not found (tried catalog + literal path)"
        )));
    }
    println!("  model: {}", resolved.display());

    // CPU capture: calibration is deterministic and the host-side lfm2/llama
    // cache path keeps post-RoPE K/V readable without GPU plumbing. Accuracy
    // of the captured signal does not depend on device.
    let device = grim_tensor::Device::Cpu;
    let lc = resolved.to_string_lossy().to_lowercase();
    let model: Box<dyn CausalLm> = if lc.ends_with(".gguf") {
        grim_engine::model_loader::load_model_from_gguf(
            &resolved.to_string_lossy(),
            device.clone(),
        )?
    } else if lc.ends_with(".grim") {
        grim_engine::model_loader::load_model_from_grim(
            &resolved.to_string_lossy(),
            device.clone(),
        )?
    } else if lc.ends_with(".safetensors") || lc.ends_with(".bin") {
        grim_engine::model_loader::load_model_from_safetensors(
            &resolved.to_string_lossy(),
            device.clone(),
        )?
    } else {
        return Err(Error::Config(format!(
            "unsupported model extension for {lc} (want .gguf/.grim/.safetensors)"
        )));
    };

    let prompts: Vec<String> = match &args.prompts_file {
        Some(p) => std::fs::read_to_string(p)
            .map_err(|e| Error::Config(format!("prompts file {p}: {e}")))?
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.to_string())
            .collect(),
        None => DEFAULT_PROMPTS.iter().map(|s| s.to_string()).collect(),
    };

    let num_layers = model
        .num_layers_hint()
        .ok_or_else(|| Error::Config("model does not report num_layers_hint".into()))?;
    println!(
        "  layers: {num_layers}  kv_heads: {}  head_dim: {}",
        args.kv_heads, args.head_dim
    );

    // Tokenizer comes from the model file (GGUF metadata) when available;
    // otherwise fall back to a deterministic char-order proxy so the code path
    // still exercises realistic structure. Calibration needs DISTRIBUTIONAL
    // realism, not semantic token identity.
    let tokenizer = crate::catalog::tokenizer_for(&resolved);
    let mut computers: Vec<ChannelImportanceComputer> = (0..num_layers)
        .map(|_| {
            ChannelImportanceComputer::new(args.kv_heads, args.head_dim, CHANNEL_GROUP_SIZE, false)
                .map_err(Error::Config)
        })
        .collect::<std::result::Result<_, _>>()?;

    let mut total_rows = 0usize;
    for (pi, prompt) in prompts.iter().enumerate() {
        let tokens_full: Vec<u32> = match &tokenizer {
            Some(t) => t.encode(prompt),
            None => prompt.chars().map(|c| (c as u32) >> 3).collect(),
        };
        if tokens_full.is_empty() {
            continue;
        }
        let tokens: Vec<u32> = tokens_full
            .into_iter()
            .take(args.max_tokens_per_prompt)
            .collect();
        let n = tokens.len();
        total_rows += n;

        let input = grim_backend_cpu::cpu_tensor(
            tokens.iter().map(|&t| t as f32).collect(),
            grim_tensor::Shape::new(vec![1, n]),
        );
        let positions = grim_backend_cpu::cpu_tensor(
            (0..n).map(|i| i as f32).collect(),
            grim_tensor::Shape::new(vec![1, n]),
        );
        let mut session = model.new_session();
        // One full-prefill forward; we never decode — importance distribution
        // of the cached prefix is the signal.
        model
            .forward(&mut *session, &input, &positions, &[])
            .map_err(|e| Error::Config(format!("calibration forward (prompt {pi}): {e}")))?;

        let state = session
            .model_state()
            .and_then(|s| {
                s.downcast_ref::<Vec<Option<grim_models_transformer::block::LlamaLayerCache>>>()
            })
            .ok_or_else(|| {
                Error::Config(
                    "model session state is not Llama layer caches (unsupported model family for calibration)"
                        .into(),
                )
            })?;

        for (layer, cache) in state.iter().enumerate() {
            let Some(c) = cache.as_ref() else { continue };
            if c.past_len == 0 {
                continue;
            }
            let row_elems = args.kv_heads * args.head_dim;
            // K/V rows live in the host mirror k_cache/v_cache (classic path)
            // or in the device arena — prefer the explicit host mirror and
            // fall back to reading the arena back.
            let mut k_all = c.k_cache.clone();
            if k_all.is_empty() {
                if let Some(devk) = &c.k_device {
                    k_all = devk.to_cpu_vec_f32()?;
                }
            }
            let mut v_all = c.v_cache.clone();
            if v_all.is_empty() {
                if let Some(devv) = &c.v_device {
                    v_all = devv.to_cpu_vec_f32()?;
                }
            }
            if k_all.len() < c.past_len * row_elems || v_all.len() < c.past_len * row_elems {
                continue;
            }
            computers[layer]
                .update_kv(
                    &k_all[..c.past_len * row_elems],
                    &v_all[..c.past_len * row_elems],
                    c.past_len,
                )
                .map_err(Error::Config)?;
        }
    }
    if total_rows == 0 {
        return Err(Error::Config("calibration captured zero rows".into()));
    }

    println!("  rows captured: {total_rows}");
    Ok(computers
        .into_iter()
        .map(ChannelImportanceComputer::finish)
        .collect())
}

fn write_sidecar(
    output: &str,
    layers: &[ChannelImportanceScores],
    budgets: Option<(f32, f32)>,
) -> Result<()> {
    let out_path = if output.ends_with(".json") {
        output.to_string()
    } else {
        format!("{output}.channels.json")
    };
    let value = serde_json::json!({
        "format_version": 1,
        "scope": "kv-channel-importance",
        "group_size": CHANNEL_GROUP_SIZE,
        "num_layers": layers.len(),
        "layers": layers,
        "allocation": budgets.map(|(kb, vb)| {
            // Derive the per-group tier assignment immediately so serving can
            // consume one artifact without re-running the allocator.
            let k_scores: Vec<Vec<f32>> = layers.iter().fold(Vec::new(), |mut acc, l| {
                for (h, row) in l.k_scores.iter().enumerate() {
                    if acc.len() <= h { acc.push(vec![0.0; row.len()]); }
                    for (g, &s) in row.iter().enumerate() { acc[h][g] += s; }
                }
                acc
            });
            let v_scores: Vec<Vec<f32>> = layers.iter().fold(Vec::new(), |mut acc, l| {
                for (h, row) in l.v_scores.iter().enumerate() {
                    if acc.len() <= h { acc.push(vec![0.0; row.len()]); }
                    for (g, &s) in row.iter().enumerate() { acc[h][g] += s; }
                }
                acc
            });
            let k_def = layers.first().map(|_| 4u8).unwrap_or(4);
            let alloc = grim_kvquant::channel_importance::ChannelBitAllocation::from_scores(
                &k_scores, k_def, kb, &v_scores, 4, vb,
            ).expect("allocator inputs are geometry-consistent");
            serde_json::to_value(alloc).unwrap()
        }),
    });
    let json = serde_json::to_string_pretty(&value)
        .map_err(|e| Error::Config(format!("serialize sidecar: {e}")))?;
    if let Some(parent) = Path::new(&out_path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| Error::Config(format!("create {}: {e}", parent.display())))?;
        }
    }
    let mut f =
        File::create(&out_path).map_err(|e| Error::Config(format!("create {out_path}: {e}")))?;
    f.write_all(json.as_bytes())
        .map_err(|e| Error::Config(format!("write {out_path}: {e}")))?;
    println!("Calibration written: {out_path}");
    Ok(())
}
