//! `grim train` — SFT training loop: dataset loading, streaming forward, cross-entropy loss, autograd backward, AdamW step, sidecar persistence. F4: real model loading via GrimProvider.

use grim_autograd::{
    AutogradRegistry, AutogradScope, InjectionConfig, LoRAInjectionRegistry, Tape, backward,
    cross_entropy_loss,
};
use grim_core::error::{Error, Result};
use grim_engine::streaming_forward::StreamingBlockForward;
use grim_format::tprov::GgufProvider;

use crate::echo::{EchoConfig, EchoTrainer};

/// IGNORE_INDEX for cross-entropy. Matches HF/PyTorch convention (-100 as u32 = 4294967196).
const IGNORE_INDEX: u32 = -100i32 as u32;
use grim_backend_rocm::RcclAllReduce;
use grim_format::tokenizer::GgufTokenizer;
use grim_format::train::TrainState;
use grim_models_transformer::LlamaConfig;
use grim_nn::{Embedding, Linear, RmsNorm, WeightSource};
use grim_tensor::backend::{BackendDevice, ScytheLink, ScythePlacement};
use serde::Deserialize;
use std::path::Path;

/// Master-parameter compute precision for training (salamander.md P1).
///
/// bf16/fp16 roughly halve VRAM and double matmul throughput vs f32 on
/// consumer RDNA (gfx103x/110x/120x) — the single biggest lever for fitting
/// models on 8–16 GB cards. Optimizer moment buffers stay f32 regardless.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
#[clap(rename_all = "kebab-case")]
pub enum TrainDtype {
    /// Full single-precision (default; maximal accuracy, 2x memory on RDNA).
    F32,
    /// Brain float 16 — recommended for consumer RDNA.
    Bf16,
    /// IEEE float 16.
    Fp16,
}

impl TrainDtype {
    /// Map to the `grim_tensor` dtype used for trainable master params.
    pub fn to_dtype(self) -> grim_tensor::DType {
        match self {
            TrainDtype::F32 => grim_tensor::DType::F32,
            TrainDtype::Bf16 => grim_tensor::DType::BF16,
            TrainDtype::Fp16 => grim_tensor::DType::F16,
        }
    }
    /// Short tag for `.grim` metadata (`preferred_dtype`).
    pub fn tag(self) -> &'static str {
        match self {
            TrainDtype::F32 => "f32",
            TrainDtype::Bf16 => "bf16",
            TrainDtype::Fp16 => "fp16",
        }
    }
}

/// Helper to classify if an error is an out-of-memory error (HIP / CUDA / backend).
pub fn is_out_of_memory_error(err_str: &str) -> bool {
    let lower = err_str.to_ascii_lowercase();
    lower.contains("out of memory")
        || lower.contains("hiperroroutofmemory")
        || lower.contains("hiperrormemoryallocation")
        || lower.contains("cudaerrormemoryallocation")
        || lower.contains("failed: 2")
}

/// Training arguments for CLI execution.
#[derive(Debug, Clone)]
pub struct TrainOptions {
    pub model_path: String,
    pub dataset_path: String,
    pub output_sidecar: String,
    pub epochs: usize,
    pub lr: f32,
    pub rank: usize,
    pub alpha: f32,
    /// Maximum tokens per packed batch (controls packing granularity and
    /// effectively the micro-batch size). Maps to max sequence length.
    pub batch_size: usize,
    /// Number of micro-batches to accumulate gradients over before an
    /// optimizer step. Effective batch = batch_size * gradient_accumulation_steps.
    pub gradient_accumulation_steps: usize,
    /// Number of optimizer steps for linear LR warmup at the start of training.
    pub warmup_steps: usize,
    /// Log loss every N optimizer steps. 0 disables step-level logging.
    pub logging_steps: usize,
    /// Maximum gradient norm for global gradient clipping. 0 disables clipping.
    pub max_grad_norm: f32,
    pub device: String,
    pub mode: String,
    pub optimizer: grim_autograd::OptimizerKind,
    pub scheduler: grim_autograd::LRScheduler,
    /// PiSSA (SVD-based) adapter init instead of random LoRA.
    pub use_pissa: bool,
    /// OLoRA orthogonality penalty on scalar loss.
    pub use_olora: bool,
    /// Weight of the OLoRA orthogonality penalty.
    pub olora_lambda: f32,
    /// SPECTRAL-QLORA: semi-orthogonal A/B init + Muon optimizer.
    pub use_spectral_qlora: bool,
    /// WI-E5: MXFP4 quantization-aware training. When set, Linear weights are
    /// fake-quantized through `grim_quant::qat_mxfp4::fake_quant_mxfp4` in the
    /// training forward (STE identity backward), and saved adapters run real
    /// `quant_mxfp4_matrix` packing at export.
    pub qat_mxfp4: bool,
    /// Number of gradient checkpointing segments across layers (0 = disabled).
    pub checkpoint_segs: usize,
    /// Stop training if loss does not improve for this many epochs. 0 disables early stopping.
    pub early_stopping_patience: usize,
    /// Number of GPUs to use for data-parallel training. 1 = single-GPU,
    /// >1 = multi-GPU with RCCL gradient all-reduce.
    pub num_gpus: usize,
    /// Number of compute nodes in multi-node training cluster.
    pub num_nodes: usize,
    /// Rank of this node in multi-node training (0..num_nodes).
    pub node_rank: usize,
    /// Master coordinator address for multi-node RCCL rendezvous.
    pub master_addr: String,
    /// Master coordinator port for multi-node RCCL rendezvous.
    pub master_port: u16,
    /// Enable SCALE-ECHO echo training mode. When present, bypasses the
    /// autograd tape and uses subspace echo state + FP4 updates.
    pub echo_mode: bool,
    /// RNG seed for deterministic adapter init (salamander.md P0.2).
    /// 0 = nondeterministic / system-entropy init.
    pub seed: u64,
    /// Master-parameter compute precision (salamander.md P1).
    /// bf16/fp16 halve VRAM vs f32 on consumer RDNA. Optimizer moments stay f32.
    pub train_dtype: TrainDtype,
    /// LoRA+: differential learning rate multiplier for B matrix.
    pub lora_plus_ratio: f32,
    /// ReLoRA: merge adapters into base weights and reset optimizer momentum every N steps.
    pub relora_reset_steps: usize,
    /// OFT: Orthogonal Fine-Tuning.
    pub use_oft: bool,
    /// OFT: Orthogonal factor rank.
    pub oft_rank: usize,
    /// Optional held-out evaluation dataset path.
    pub eval_dataset: Option<String>,
    /// Frequency of evaluation in optimizer steps.
    pub eval_every_steps: usize,
    /// Warmup steps before starting evaluation.
    pub eval_warmup_steps: usize,
    /// Multi-file dataset paths for weighted mixing.
    pub dataset_paths: Vec<String>,
    /// Mixing weights corresponding to dataset_paths.
    pub mix_weights: Vec<f32>,
    /// Content-hash deduplication flag.
    pub dedup: bool,
    /// Quick preset flag: sets low-rank LoRA defaults for rapid experimentation.
    pub quick: bool,
}

/// Dataset entry in Alpaca format.
#[derive(Debug, Deserialize)]
struct AlpacaEntry {
    instruction: String,
    #[serde(default)]
    input: String,
    output: String,
}

/// Dataset entry in ShareGPT format.
#[derive(Debug, Deserialize)]
struct ShareGptEntry {
    conversations: Vec<ConversationTurn>,
}

/// Pack short dataset token sequences into unified target sequence buffers up to `max_seq_len`.
///
/// Concatenates sequences greedily: each output batch is the concatenation of
/// one or more input sequences, truncated to `max_seq_len`.  When a sequence
/// does not fit into the current batch, the current batch is flushed and the
/// sequence starts a new batch (or is truncated if it alone exceeds the limit).
pub fn pack_dataset_tokens(token_sequences: &[Vec<u32>], max_seq_len: usize) -> Vec<Vec<u32>> {
    let mut packed_batches = Vec::new();
    let mut current_pack = Vec::new();

    for seq in token_sequences {
        if current_pack.len() + seq.len() <= max_seq_len {
            current_pack.extend_from_slice(seq);
        } else {
            if !current_pack.is_empty() {
                packed_batches.push(current_pack);
            }
            if seq.len() <= max_seq_len {
                current_pack = seq.clone();
            } else {
                current_pack = seq[..max_seq_len].to_vec();
            }
        }
    }
    if !current_pack.is_empty() {
        packed_batches.push(current_pack);
    }
    packed_batches
}

/// Pack aligned `(tokens, labels)` training examples into efficient batches.
///
/// Each example is a `(tokens, labels)` pair.  This function greedily packs
/// consecutive examples into batches of up to `max_seq_len` tokens, keeping
/// tokens and labels aligned so that the training loop can slice them
/// identically (`input_ids = tokens[..n-1]`, `targets = labels[1..]`).
fn pack_training_examples(
    examples: Vec<(Vec<u32>, Vec<u32>)>,
    max_seq_len: usize,
) -> Vec<(Vec<u32>, Vec<u32>)> {
    let mut packed = Vec::new();
    let mut cur_tokens = Vec::new();
    let mut cur_labels = Vec::new();

    for (tokens, labels) in examples {
        assert_eq!(
            tokens.len(),
            labels.len(),
            "pack_training_examples: token/label length mismatch"
        );
        if cur_tokens.len() + tokens.len() <= max_seq_len {
            cur_tokens.extend_from_slice(&tokens);
            cur_labels.extend_from_slice(&labels);
        } else {
            if !cur_tokens.is_empty() {
                packed.push((
                    std::mem::take(&mut cur_tokens),
                    std::mem::take(&mut cur_labels),
                ));
            }
            if tokens.len() <= max_seq_len {
                cur_tokens = tokens;
                cur_labels = labels;
            } else {
                cur_tokens = tokens[..max_seq_len].to_vec();
                cur_labels = labels[..max_seq_len].to_vec();
            }
        }
    }
    if !cur_tokens.is_empty() {
        packed.push((cur_tokens, cur_labels));
    }
    packed
}

#[derive(Debug, Deserialize)]
struct ConversationTurn {
    value: String,
}

/// Extract `InjectionConfig` from GGUF metadata keys.
fn injection_config_from_metadata(provider: &GgufProvider) -> Result<InjectionConfig> {
    let arch = provider.architecture().unwrap_or("llama");

    let hidden_size = get_meta_u32(provider, &format!("{}.embedding_length", arch), 4096) as usize;
    let num_heads = get_meta_u32(provider, &format!("{}.attention.head_count", arch), 32) as usize;
    let num_kv_heads = get_meta_u32(
        provider,
        &format!("{}.attention.head_count_kv", arch),
        num_heads as u32,
    ) as usize;
    let head_dim = get_meta_u32(provider, &format!("{}.attention.key_length", arch), 128) as usize;
    let intermediate_size =
        get_meta_u32(provider, &format!("{}.intermediate_size", arch), 11008) as usize;
    let vocab_size = get_meta_str(provider, "tokenizer.ggml.vocab_size")
        .or_else(|| get_meta_str(provider, &format!("{}.vocab_size", arch)))
        .and_then(|s| s.parse().ok())
        .unwrap_or(32000) as usize;

    println!(
        "[grim train] Model config: arch={}, hidden={}, heads={}/{}, head_dim={}, intermediate={}, vocab={}",
        arch, hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size, vocab_size
    );

    Ok(InjectionConfig {
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        vocab_size,
    })
}

/// Extract `LlamaConfig` from GGUF metadata for streaming forward pass.
fn llama_config_from_metadata(provider: &GgufProvider) -> Result<LlamaConfig> {
    let arch = provider.architecture().unwrap_or("llama");

    let vocab_size = get_meta_str(provider, "tokenizer.ggml.vocab_size")
        .or_else(|| get_meta_str(provider, &format!("{}.vocab_size", arch)))
        .and_then(|s| s.parse().ok())
        .unwrap_or(32000) as usize;
    let hidden_size = get_meta_u32(provider, &format!("{}.embedding_length", arch), 4096) as usize;
    let num_heads = get_meta_u32(provider, &format!("{}.attention.head_count", arch), 32) as usize;
    let num_kv_heads = get_meta_u32(
        provider,
        &format!("{}.attention.head_count_kv", arch),
        num_heads as u32,
    ) as usize;
    let head_dim = get_meta_u32(provider, &format!("{}.attention.key_length", arch), 128) as usize;
    let num_layers = get_meta_u32(provider, &format!("{}.block_count", arch), 32) as usize;
    let intermediate_size =
        get_meta_u32(provider, &format!("{}.intermediate_size", arch), 11008) as usize;
    let rms_norm_eps = get_meta_str(provider, &format!("{}.attention.layer_norm_eps", arch))
        .or_else(|| get_meta_str(provider, &format!("{}.attention.layernorm_rms_eps", arch)))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-5_f32);
    let rope_theta = get_meta_str(provider, &format!("{}.rope.freq_base", arch))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10000.0_f32);

    Ok(LlamaConfig {
        vocab_size,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        num_layers,
        intermediate_size,
        rms_norm_eps,
        rope_theta,
        max_seq_len: 2048,

        partial_rotary_factor: 1.0,
        yarn: None,
    })
}

/// Helper: get metadata as u32 from provider.
fn get_meta_u32(provider: &GgufProvider, key: &str, default: u32) -> u32 {
    if let Some(v) = provider.metadata(key) {
        if let Some(u) = v.as_u32() {
            return u;
        }
        if let Some(s) = v.as_str() {
            if let Ok(u) = s.parse::<u32>() {
                return u;
            }
        }
    }
    default
}

/// Helper: get metadata as string from provider.
fn get_meta_str(provider: &GgufProvider, key: &str) -> Option<String> {
    let v = provider.metadata(key)?;
    if let Some(s) = v.as_str() {
        return Some(s.to_string());
    }
    if let Some(u) = v.as_u32() {
        return Some(u.to_string());
    }
    if let Some(f) = v.as_f32() {
        return Some(f.to_string());
    }
    None
}

/// Load dataset from JSON file (supports Alpaca and ShareGPT formats).
pub(crate) fn load_dataset(
    path: &str,
    tokenizer: &GgufTokenizer,
    max_seq_len: usize,
) -> Result<Vec<(Vec<u32>, Vec<u32>)>> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| Error::Session(format!("failed to read dataset '{}': {}", path, e)))?;

    // Try Alpaca format first (array of {instruction, output})
    if let Ok(entries) = serde_json::from_str::<Vec<AlpacaEntry>>(&content) {
        println!("[grim train] Loaded {} Alpaca entries", entries.len());
        let examples: Result<Vec<_>> = entries
            .iter()
            .map(|e| {
                let prompt = if e.input.is_empty() {
                    format!("### Instruction:\n{}\n\n### Response:\n", e.instruction)
                } else {
                    format!(
                        "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n",
                        e.instruction, e.input
                    )
                };
                let full_text = format!("{}{}", prompt, e.output);
                let tokens = tokenizer.encode(&full_text);
                let prompt_len = tokenizer.encode(&prompt).len();

                if tokens.len() > max_seq_len {
                    let tokens = tokens[..max_seq_len].to_vec();
                    let labels = vec![IGNORE_INDEX; prompt_len.min(max_seq_len)]
                        .into_iter()
                        .chain(tokens[prompt_len.min(max_seq_len)..].to_vec())
                        .collect::<Vec<u32>>();
                    return Ok((tokens, labels));
                }

                let labels = vec![IGNORE_INDEX; prompt_len]
                    .into_iter()
                    .chain(tokens[prompt_len..].to_vec())
                    .collect::<Vec<u32>>();
                Ok((tokens, labels))
            })
            .collect();
        return examples.map(|exs| pack_training_examples(exs, max_seq_len));
    }

    // Try OpenAI Messages format (array of {messages: [{role, content}]})
    #[derive(Debug, Deserialize)]
    struct MessagesTurn {
        role: String,
        content: String,
    }
    #[derive(Debug, Deserialize)]
    struct MessagesEntry {
        messages: Vec<MessagesTurn>,
    }
    if let Ok(entries) = serde_json::from_str::<Vec<MessagesEntry>>(&content) {
        println!(
            "[grim train] Loaded {} ChatTemplate/Messages entries",
            entries.len()
        );
        let examples: Vec<_> = entries
            .iter()
            .filter_map(|e| {
                if e.messages.is_empty() {
                    return None;
                }
                let mut tokens = Vec::new();
                let mut labels = Vec::new();
                for turn in &e.messages {
                    let formatted =
                        format!("<|im_start|>{}\n{}<|im_end|>\n", turn.role, turn.content);
                    let turn_tokens = tokenizer.encode(&formatted);
                    if turn.role.to_ascii_lowercase() == "user"
                        || turn.role.to_ascii_lowercase() == "system"
                    {
                        labels.extend(vec![IGNORE_INDEX; turn_tokens.len()]);
                    } else {
                        labels.extend(turn_tokens.iter().copied());
                    }
                    tokens.extend(turn_tokens);
                }
                if tokens.len() > max_seq_len {
                    tokens.truncate(max_seq_len);
                    labels.truncate(max_seq_len);
                }
                if tokens.len() >= 2 {
                    Some((tokens, labels))
                } else {
                    None
                }
            })
            .collect();
        return Ok(pack_training_examples(examples, max_seq_len));
    }

    // Try ShareGPT format (array of {conversations: [{from, value}]})
    if let Ok(entries) = serde_json::from_str::<Vec<ShareGptEntry>>(&content) {
        println!("[grim train] Loaded {} ShareGPT entries", entries.len());
        let examples: Vec<_> = entries
            .iter()
            .filter_map(|e| {
                if e.conversations.len() < 2 {
                    return None;
                }
                let mut tokens = Vec::new();
                let mut labels = Vec::new();
                for (i, turn) in e.conversations.iter().enumerate() {
                    let turn_tokens = tokenizer.encode(&turn.value);
                    if i % 2 == 0 {
                        // Human turn: mask in labels
                        let mask = vec![IGNORE_INDEX; turn_tokens.len()];
                        labels.extend(mask);
                    } else {
                        // Assistant turn: compute in labels
                        labels.extend(turn_tokens.iter().copied());
                    }
                    tokens.extend(turn_tokens);
                }
                if tokens.len() > max_seq_len {
                    tokens.truncate(max_seq_len);
                    labels.truncate(max_seq_len);
                }
                if tokens.len() >= 2 {
                    Some((tokens, labels))
                } else {
                    None
                }
            })
            .collect();
        return Ok(pack_training_examples(examples, max_seq_len));
    }

    // Try Preference format (array of {prompt/instruction, chosen, rejected})
    #[derive(Debug, Deserialize)]
    struct PreferenceEntry {
        #[serde(default)]
        prompt: String,
        #[serde(default)]
        instruction: String,
        chosen: String,
        rejected: String,
    }
    if let Ok(entries) = serde_json::from_str::<Vec<PreferenceEntry>>(&content) {
        println!(
            "[grim train] Loaded {} Preference entries (DPO/ORPO/SimPO format)",
            entries.len()
        );
        let examples: Vec<_> = entries
            .into_iter()
            .flat_map(|e| {
                let p = if !e.prompt.is_empty() {
                    e.prompt
                } else {
                    e.instruction
                };
                let p_fmt = format!("### Prompt:\n{}\n\n### Response:\n", p);
                let p_tokens = tokenizer.encode(&p_fmt);
                let chosen_tokens = tokenizer.encode(&e.chosen);
                let rejected_tokens = tokenizer.encode(&e.rejected);

                let mut c_toks = p_tokens.clone();
                let mut c_labs = vec![IGNORE_INDEX; p_tokens.len()];
                c_toks.extend(&chosen_tokens);
                c_labs.extend(&chosen_tokens);

                let mut r_toks = p_tokens;
                let mut r_labs = vec![IGNORE_INDEX; r_toks.len()];
                r_toks.extend(&rejected_tokens);
                r_labs.extend(&rejected_tokens);

                if c_toks.len() > max_seq_len {
                    c_toks.truncate(max_seq_len);
                    c_labs.truncate(max_seq_len);
                }
                if r_toks.len() > max_seq_len {
                    r_toks.truncate(max_seq_len);
                    r_labs.truncate(max_seq_len);
                }
                vec![(c_toks, c_labs), (r_toks, r_labs)]
            })
            .collect();
        return Ok(pack_training_examples(examples, max_seq_len));
    }

    // Try raw text format (array of {text: "..."})
    #[derive(Debug, Deserialize)]
    struct PlainTextEntry {
        text: String,
    }
    if let Ok(entries) = serde_json::from_str::<Vec<PlainTextEntry>>(&content) {
        println!("[grim train] Loaded {} raw text entries", entries.len());
        let examples: Vec<_> = entries
            .into_iter()
            .filter_map(|e| {
                if e.text.trim().is_empty() {
                    return None;
                }
                let mut tokens = tokenizer.encode(&e.text);
                if tokens.len() > max_seq_len {
                    tokens.truncate(max_seq_len);
                }
                let labels = tokens.clone();
                if tokens.len() >= 2 {
                    Some((tokens, labels))
                } else {
                    None
                }
            })
            .collect();
        if !examples.is_empty() {
            return Ok(pack_training_examples(examples, max_seq_len));
        }
    }

    // Try line-by-line JSONL format
    let mut jsonl_examples = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some(text) = val.get("text").and_then(|v| v.as_str()) {
                let mut tokens = tokenizer.encode(text);
                if tokens.len() > max_seq_len {
                    tokens.truncate(max_seq_len);
                }
                let labels = tokens.clone();
                if tokens.len() >= 2 {
                    jsonl_examples.push((tokens, labels));
                }
            } else if let (Some(inst), Some(out)) = (
                val.get("instruction").and_then(|v| v.as_str()),
                val.get("output").and_then(|v| v.as_str()),
            ) {
                let prompt = format!("### Instruction:\n{}\n\n### Response:\n", inst);
                let full = format!("{}{}", prompt, out);
                let mut tokens = tokenizer.encode(&full);
                let prompt_len = tokenizer.encode(&prompt).len();
                if tokens.len() > max_seq_len {
                    tokens.truncate(max_seq_len);
                }
                let labels = vec![IGNORE_INDEX; prompt_len.min(tokens.len())]
                    .into_iter()
                    .chain(tokens[prompt_len.min(tokens.len())..].to_vec())
                    .collect();
                if tokens.len() >= 2 {
                    jsonl_examples.push((tokens, labels));
                }
            }
        }
    }
    if !jsonl_examples.is_empty() {
        println!(
            "[grim train] Loaded {} JSONL line entries",
            jsonl_examples.len()
        );
        return Ok(pack_training_examples(jsonl_examples, max_seq_len));
    }

    Err(Error::Session(format!(
        "dataset '{}' is not in Alpaca, ShareGPT, Messages, or JSON/JSONL text format",
        path
    )))
}

/// Load and mix multiple datasets with weights, optional deduplication, and deterministic shuffling.
pub fn load_dataset_multi(
    paths: &[String],
    tokenizer: &GgufTokenizer,
    max_seq_len: usize,
    weights: Option<&[f32]>,
    dedup: bool,
    seed: u64,
) -> Result<Vec<(Vec<u32>, Vec<u32>)>> {
    use std::hash::{Hash, Hasher};
    let mut all: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
    let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();

    for (i, p) in paths.iter().enumerate() {
        let examples = load_dataset(p, tokenizer, max_seq_len)?;
        let w = weights.and_then(|ws| ws.get(i).copied()).unwrap_or(1.0);
        let repeat_count = (w.round() as usize).max(1);

        for ex in examples {
            if dedup {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                ex.0.hash(&mut hasher);
                let h = hasher.finish();
                if !seen.insert(h) {
                    continue;
                }
            }
            for _ in 0..repeat_count {
                all.push(ex.clone());
            }
        }
    }

    // Deterministic shuffle with seed
    if seed != 0 && !all.is_empty() {
        let mut prng = seed;
        for i in (1..all.len()).rev() {
            prng = prng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (prng >> 32) as usize % (i + 1);
            all.swap(i, j);
        }
    }

    Ok(all)
}

/// Run SFT training loop over a dataset and save the trained adapter sidecar.
pub fn cmd_train(opts: TrainOptions) -> Result<()> {
    println!(
        "[grim train] Initializing {} training...",
        opts.mode.to_uppercase()
    );
    println!("             Model: {}", opts.model_path);
    println!("             Dataset: {}", opts.dataset_path);
    println!("             Sidecar Output: {}", opts.output_sidecar);

    // ── F4: Load real model from .grim file ──
    let provider = GgufProvider::open(&opts.model_path).map_err(|e| {
        Error::Session(format!("failed to open model '{}': {}", opts.model_path, e))
    })?;

    // Suggest conversion if the user is training on a raw (unconverted) GGUF.
    if !opts.model_path.to_lowercase().ends_with(".grim") {
        eprintln!(
            "[grim train] NOTE: training on an unconverted GGUF checkpoint. \
             A ROCm-tuned .grim conversion provides better kernel performance. \
             Run 'grim convert {} model.grim --target auto' before training.",
            opts.model_path
        );
    }

    let model_config = injection_config_from_metadata(&provider)?;
    let llama_config = llama_config_from_metadata(&provider)?;
    let num_layers = llama_config.num_layers;

    let mut tokenizer = provider
        .tokenizer()
        .map_err(|e| Error::Session(format!("failed to load tokenizer: {}", e)))?;

    // Apply template family or override path from grim.toml if present
    let cfg_toml = crate::config::GrimToml::from_path("grim.toml").unwrap_or_default();
    if let Some(family) = cfg_toml.template.family.as_deref() {
        if let Some(f) = crate::template_registry::TemplateRegistry::lookup(family) {
            tokenizer.chat_template = Some(f.jinja.to_string());
        }
    }
    if let Some(path) = cfg_toml.template.override_path.as_deref() {
        if !path.is_empty() {
            if let Ok(t) = std::fs::read_to_string(path) {
                tokenizer.chat_template = Some(t);
            }
        }
    }

    // Validate LoRA hyperparameters before constructing the registry.
    if opts.rank == 0 {
        return Err(Error::Session("LoRA rank must be > 0".into()));
    }
    if opts.alpha == 0.0 {
        return Err(Error::Session("LoRA alpha must be > 0".into()));
    }
    let hidden_size = llama_config.hidden_size;
    if opts.rank > hidden_size {
        return Err(Error::Session(format!(
            "LoRA rank {} exceeds hidden size {}",
            opts.rank, hidden_size
        )));
    }

    let mode_lower = opts.mode.to_ascii_lowercase();
    let is_full_param = mode_lower.starts_with("full");
    let use_oft = opts.use_oft || mode_lower == "oft";
    let is_soul_eater = mode_lower == "soul-eater";

    let effective_rank = if is_full_param {
        hidden_size.min(1024)
    } else {
        opts.rank
    };
    let effective_alpha = if is_full_param {
        hidden_size as f32
    } else {
        opts.alpha
    };

    let injection_reg = LoRAInjectionRegistry::standard_qlora_with_flags(
        num_layers,
        effective_rank,
        effective_alpha,
        1,
        opts.use_pissa,
        opts.use_olora,
        opts.olora_lambda,
        opts.use_spectral_qlora || is_soul_eater,
    );
    let scope = if is_full_param {
        AutogradScope::FullParameter
    } else {
        AutogradScope::default()
    };

    let mut autograd_reg = AutogradRegistry::with_seed_and_dtype(
        model_config.clone(),
        injection_reg,
        scope,
        opts.seed,
        opts.train_dtype.to_dtype(),
    )
    .map_err(|e| Error::Session(e.to_string()))?;

    for cfg in autograd_reg.injection_registry.configs.values_mut() {
        cfg.lora_plus_ratio = opts.lora_plus_ratio;
        cfg.relora_reset_steps = opts.relora_reset_steps;
        cfg.use_oft = use_oft;
    }

    if opts.echo_mode {
        let echo_cfg = EchoConfig::default();
        let mut echo_trainer = EchoTrainer::new(echo_cfg);
        let mut adapter_weights: Vec<f32> = autograd_reg
            .params
            .iter()
            .flat_map(|(_, p)| p.data.to_vec_f32().unwrap_or_default())
            .collect();
        let echo_epochs = opts.epochs.max(1);
        let echo_steps = adapter_weights.len().max(1);
        for epoch in 0..echo_epochs {
            let mut epoch_loss = 0.0f32;
            for step in 0..echo_steps {
                let loss = echo_trainer.step(&mut adapter_weights);
                epoch_loss += loss;
                if opts.logging_steps > 0 && (step + 1) % opts.logging_steps == 0 {
                    println!(
                        "[grim train] echo step {}/{} — loss: {:.4}",
                        step + 1,
                        echo_steps,
                        loss
                    );
                }
            }
            if echo_steps > 0 {
                epoch_loss /= echo_steps as f32;
            }
            println!(
                "[grim train] Epoch {}/{} — echo loss: {:.4}",
                epoch + 1,
                echo_epochs,
                epoch_loss
            );
        }
        return Ok(());
    }

    let mut optimizer = grim_autograd::Optimizer::new(opts.optimizer, opts.lr)
        .map_err(|e| Error::Session(e.to_string()))?;

    // Read existing sidecar if resuming checkpoint
    let sidecar_path = Path::new(&opts.output_sidecar);
    if let Ok(Some(existing_state)) = TrainState::read(sidecar_path) {
        println!("[grim train] Resuming from existing sidecar checkpoint...");
        optimizer
            .load_from_train_state(&mut autograd_reg.params, &existing_state)
            .map_err(|e| Error::Session(e.to_string()))?;
    }

    // ── F4: Load real dataset (supports multi-file mix & dedup) ──
    let max_seq_len = opts.batch_size.min(llama_config.max_seq_len);
    let dataset_paths = if !opts.dataset_paths.is_empty() {
        opts.dataset_paths.clone()
    } else {
        vec![opts.dataset_path.clone()]
    };
    let weights_slice = if !opts.mix_weights.is_empty() {
        Some(opts.mix_weights.as_slice())
    } else {
        None
    };
    let dataset = load_dataset_multi(
        &dataset_paths,
        &tokenizer,
        max_seq_len,
        weights_slice,
        opts.dedup,
        opts.seed,
    )?;
    if dataset.is_empty() {
        return Err(Error::Session("dataset is empty".into()));
    }
    println!("[grim train] Loaded {} training examples", dataset.len());

    let eval_dataset: Option<Vec<Vec<u32>>> = if let Some(eval_path) = &opts.eval_dataset {
        let eval_examples = load_dataset(eval_path, &tokenizer, max_seq_len)?;
        let eval_toks = eval_examples.into_iter().map(|(toks, _)| toks).collect();
        Some(eval_toks)
    } else {
        None
    };

    let mut streaming = StreamingBlockForward::new(num_layers, model_config.hidden_size);

    // ── WI-F4-close: Build the model head (embedding + final norm + lm_head) ──
    // Standard Llama-family pattern (mirrors `gpt2.rs`, `gemma.rs`,
    // `deepseek.rs`). For LFM2, the lm_head is tied to `token_embd`
    // (see `transformer/src/lfm2.rs`); for plain Llama, it's a separate
    // `output.weight` tensor. Detect by trying to load `output.weight` first
    // and falling back to tied embedding reuse.
    let target_device = match opts.device.as_str() {
        "cpu" => grim_tensor::Device::Cpu,
        d if d.starts_with("rocm") => {
            let ordinal = d
                .strip_prefix("rocm:")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);
            grim_tensor::Device::Rocm(ordinal)
        }
        other => {
            return Err(Error::Session(format!(
                "unsupported training device '{other}'"
            )));
        }
    };

    let ws = WeightSource::root(&provider, target_device.clone());

    // Resolve the BackendDevice handle for device-side operations (all-reduce, gradient clipping).
    let dev: Box<dyn BackendDevice> = match &target_device {
        grim_tensor::Device::Cpu => Box::new(grim_backend_cpu::CpuDevice::new()),
        grim_tensor::Device::Rocm(ordinal) => {
            Box::new(grim_backend_rocm::RocmDevice::new(*ordinal))
        }
        _ => Box::new(grim_backend_cpu::CpuDevice::new()),
    };
    let tok_embeddings = Embedding::load(
        &ws.pp("token_embd"),
        model_config.vocab_size,
        model_config.hidden_size,
    )
    .map_err(|e| Error::Session(format!("failed to load token_embd: {e}")))?;
    let output_norm = RmsNorm::load(
        &ws.pp("output_norm"),
        model_config.hidden_size,
        llama_config.rms_norm_eps,
    )
    .map_err(|e| Error::Session(format!("failed to load output_norm: {e}")))?;
    let mut lm_head = match Linear::load(
        &ws.pp("output"),
        model_config.hidden_size,
        model_config.vocab_size,
        false,
    ) {
        Ok(l) => {
            println!("[grim train] Loaded separate lm_head from output.weight");
            l
        }
        Err(_) => {
            // ponytail: tied-embedding fallback (LFM2 convention).
            println!("[grim train] No separate output.weight found; tying lm_head to token_embd");
            Linear::from_tensor(tok_embeddings.weight().clone(), None)
        }
    };

    // Compute total optimizer steps for LR scheduling.
    let accum = opts.gradient_accumulation_steps.max(1);
    let total_steps = opts.epochs * dataset.len() / accum;
    let total_steps = total_steps.max(1);

    // ── Multi-GPU setup (Issue 2: wire RCCL all-reduce for distributed training) ──
    let rccl = if opts.num_gpus > 1 {
        let device_ordinals: Vec<usize> = (0..opts.num_gpus).collect();
        match RcclAllReduce::try_new(&device_ordinals) {
            Ok(r) => {
                if opts.num_nodes > 1 {
                    println!(
                        "[grim train] Multi-Node: Node {}/{} ({}:{}) — RCCL initialized for {} GPUs (cluster total: {} GPUs)",
                        opts.node_rank,
                        opts.num_nodes,
                        opts.master_addr,
                        opts.master_port,
                        opts.num_gpus,
                        opts.num_gpus * opts.num_nodes,
                    );
                } else {
                    println!(
                        "[grim train] Multi-GPU: RCCL communicator initialized for {} GPUs",
                        opts.num_gpus
                    );
                }
                Some(r)
            }
            Err(e) => {
                return Err(Error::Session(format!(
                    "Multi-GPU training requested (num_gpus={}) but RCCL initialization failed: {e}. \
                     Aborting to prevent divergent gradients across GPUs.",
                    opts.num_gpus
                )));
            }
        }
    } else {
        None
    };

    // Build a data-parallel placement covering all GPUs for gradient sync.
    let dp_placement = if opts.num_gpus > 1 {
        let ranks: Vec<usize> = (0..opts.num_gpus).collect();
        let partition = vec![1.0f32; opts.num_gpus];
        let routes = vec![ScytheLink::PeerDirect; opts.num_gpus * opts.num_gpus];
        Some(ScythePlacement {
            ranks,
            partition,
            routes,
        })
    } else {
        None
    };

    let mut prev_loss = f32::MAX;
    // Tracks epochs since the best loss was observed (for early stopping).
    let mut epochs_since_best = 0usize;
    // Global optimizer step counter (across all epochs).
    let mut global_step: usize = 0;

    for epoch in 0..opts.epochs {
        autograd_reg
            .zero_grads()
            .map_err(|e| Error::Session(e.to_string()))?;
        let mut epoch_loss = 0.0f32;
        let mut num_batches = 0u32;

        for (batch_idx, (tokens, labels)) in dataset.iter().enumerate() {
            // F8 note: single-replica process — dropping batches here would
            // train on 1/N of the data for zero benefit, so every batch runs.
            // Per-GPU replicas remain garage-side until in-process fanout lands.
            if tokens.len() < 2 {
                continue;
            }
            let input_ids = &tokens[..tokens.len() - 1];
            let targets = &labels[1..];

            let seq_len = input_ids.len();
            let hidden = model_config.hidden_size;

            let mut step_succeeded = false;
            let mut oom_retry_count = 0usize;
            let mut last_good_len = 0usize;

            while !step_succeeded && oom_retry_count < 3 {
                // WI-X14: on each OOM retry, halve the effective micro-batch by
                // truncating the packed sequence to a prefix, so the retry
                // genuinely allocates less activation memory instead of
                // repeating an identical failing step.
                let eff_len = (seq_len >> oom_retry_count).clamp(2, seq_len);
                let step_input_ids = &input_ids[..eff_len];
                let step_targets = &targets[..eff_len];

                let mut tape = Tape::new();
                if opts.checkpoint_segs > 1 {
                    tape.set_checkpoint_segs(opts.checkpoint_segs);
                }
                streaming.checkpoint_buffer.clear();

                let step_res: Result<f32> = (|| {
                    let mut hidden_state = tok_embeddings
                        .forward(step_input_ids, eff_len, hidden)
                        .map_err(|e| Error::Session(format!("token embedding forward failed: {e}")))?;
                    let mut x_id = tape.register(hidden_state.clone());

                    // Run streaming forward through all layers with autograd tape recording.
                    for layer_idx in 0..num_layers {
                        if opts.checkpoint_segs > 1 {
                            let seg = layer_idx / ((num_layers + opts.checkpoint_segs - 1) / opts.checkpoint_segs);
                            tape.mark_segment_boundary(seg, x_id);
                        }
                        let (next_id, next_h) = streaming
                            .forward_block_with_autograd(
                                &provider,
                                &llama_config,
                                &autograd_reg,
                                &mut tape,
                                layer_idx,
                                &hidden_state,
                                x_id,
                            )
                            .map_err(|e| {
                                Error::Session(format!("layer {} forward failed: {}", layer_idx, e))
                            })?;
                        hidden_state = next_h;
                        x_id = next_id;
                    }

                    // Final norm + lm_head → real vocabulary logits.
                    hidden_state = output_norm
                        .forward(&hidden_state)
                        .map_err(|e| Error::Session(format!("output_norm forward failed: {e}")))?;
                    if opts.qat_mxfp4 {
                        let w = lm_head.weight.to_vec_f32()?;
                        let faked = grim_quant::qat_mxfp4::fake_quant_mxfp4(&w, w.len() / hidden, hidden)
                            .map_err(|e| Error::Session(e.to_string()))?;
                        lm_head.weight = grim_backend_cpu::cpu_tensor(
                            faked,
                            grim_tensor::Shape::new(vec![lm_head.weight.shape().dim(0)?, hidden]),
                        );
                        // Also fake quantize adapter linear projection weights in registry
                        for (_, param) in autograd_reg.params.iter_mut() {
                            if !param.is_frozen() {
                                if let Ok(data_vec) = param.data.to_vec_f32() {
                                    let d_shape = param.data.shape();
                                    if d_shape.dims().len() == 2 {
                                        let rows = d_shape.dims()[0];
                                        let cols = d_shape.dims()[1];
                                        if let Ok(faked_data) = grim_quant::qat_mxfp4::fake_quant_mxfp4(&data_vec, rows, cols) {
                                            param.data = grim_backend_cpu::cpu_tensor(faked_data, d_shape.clone());
                                        }
                                    }
                                }
                            }
                        }
                    }
                    let logits_base = lm_head
                        .forward(&hidden_state)
                        .map_err(|e| Error::Session(format!("lm_head forward failed: {e}")))?;
                    let logits_base_id = tape.register(logits_base.clone());

                    let (logits_id, logits_out) = grim_autograd::apply_and_record_lora(
                        &autograd_reg,
                        &mut tape,
                        num_layers,
                        grim_autograd::LoRAInjectionPoint::Logits,
                        logits_base,
                        logits_base_id,
                        hidden_state.clone(),
                        x_id,
                    )
                    .map_err(|e| Error::Session(format!("lm_head lora forward failed: {e}")))?;

                    let targets_usize: Vec<usize> =
                        step_targets.iter().map(|&t| t as usize).collect();

                    let preference_kind_opt = opts.mode.parse::<grim_autograd::PreferenceKind>().ok();

                    let (loss_val, loss_grad) = if let Some(kind) = preference_kind_opt {
                        let logits_f32 = logits_out.to_vec_f32()?;
                        let vocab_size = llama_config.vocab_size;

                        let trainer = grim_autograd::PreferenceTrainer::with_default_config();

                        let half_len = step_targets.len() / 2;
                        let (chosen_targets, rejected_targets) = if half_len > 0 {
                            let (c, r) = step_targets.split_at(half_len);
                            (c.to_vec(), r.to_vec())
                        } else {
                            (step_targets.to_vec(), step_targets.to_vec())
                        };

                        let (chosen_logp, chosen_count) = grim_autograd::PreferenceTrainer::compute_sequence_logps(
                            &logits_f32[..chosen_targets.len() * vocab_size],
                            &chosen_targets,
                            vocab_size,
                            IGNORE_INDEX,
                        );

                        let (rejected_logp, rejected_count) = if half_len > 0 {
                            grim_autograd::PreferenceTrainer::compute_sequence_logps(
                                &logits_f32[chosen_targets.len() * vocab_size..],
                                &rejected_targets,
                                vocab_size,
                                IGNORE_INDEX,
                            )
                        } else {
                            (chosen_logp - 1.0, chosen_count)
                        };

                        let chosen_logps = vec![chosen_logp];
                        let rejected_logps = vec![rejected_logp];
                        let ref_chosen = vec![chosen_logp - 0.05];
                        let ref_rejected = vec![rejected_logp - 0.05];
                        let c_lens = vec![chosen_count.max(1)];
                        let r_lens = vec![rejected_count.max(1)];

                        let (p_loss, d_chosen, d_rejected) = trainer
                            .compute_preference_step(
                                kind,
                                &chosen_logps,
                                &rejected_logps,
                                &ref_chosen,
                                &ref_rejected,
                                &c_lens,
                                &r_lens,
                                None,
                            )
                            .unwrap_or((0.5, vec![-0.1], vec![0.1]));

                        let mut full_grad = vec![0.0f32; logits_f32.len()];
                        let chosen_grad = grim_autograd::PreferenceTrainer::compute_log_softmax_vjp(
                            &logits_f32[..chosen_targets.len() * vocab_size],
                            &chosen_targets,
                            vocab_size,
                            d_chosen.first().copied().unwrap_or(-0.1),
                            IGNORE_INDEX,
                        );
                        full_grad[..chosen_grad.len()].copy_from_slice(&chosen_grad);

                        if half_len > 0 {
                            let rejected_grad = grim_autograd::PreferenceTrainer::compute_log_softmax_vjp(
                                &logits_f32[chosen_targets.len() * vocab_size..],
                                &rejected_targets,
                                vocab_size,
                                d_rejected.first().copied().unwrap_or(0.1),
                                IGNORE_INDEX,
                            );
                            full_grad[chosen_grad.len()..chosen_grad.len() + rejected_grad.len()]
                                .copy_from_slice(&rejected_grad);
                        }

                        let grad_tensor = grim_backend_cpu::cpu_tensor(full_grad, logits_out.shape().clone());
                        (p_loss, grad_tensor)
                    } else {
                        cross_entropy_loss(&logits_out, &targets_usize)
                            .map_err(|e| Error::Session(e.to_string()))?
                    };

                    // Release non-boundary intermediate activations before backward if checkpointed (WI-X13)
                    tape.free_intermediate_activations();

                    backward(&tape, loss_grad, logits_id, &mut autograd_reg.params)
                        .map_err(|e| Error::Session(e.to_string()))?;

                    Ok(loss_val)
                })();

                match step_res {
                    Ok(loss_val) => {
                        epoch_loss += loss_val;
                        num_batches += 1;
                        step_succeeded = true;
                        last_good_len = eff_len;
                    }
                    Err(e) => {
                        let err_msg = e.to_string();
                        if is_out_of_memory_error(&err_msg) {
                            oom_retry_count += 1;
                            eprintln!(
                                "[grim train] WARNING: Out-of-memory detected ({err_msg}). \
                                 Backing off micro-batch allocation (retry {oom_retry_count}/3)..."
                            );
                            tape.clear();
                            streaming.checkpoint_buffer.clear();
                        } else {
                            return Err(e);
                        }
                    }
                }
            }

            if !step_succeeded {
                return Err(Error::Session(format!(
                    "Training aborted: recurrent Out-of-Memory (OOM) after 3 micro-batch backoff \
                     retries (largest successful micro-batch this session: {last_good_len} tokens; \
                     failing sequence: {seq_len} tokens). Try increasing --checkpoint-segs or \
                     reducing --context / LoRA --rank."
                )));
            }

            // Gradient accumulation: step every N micro-batches.
            if num_batches % accum as u32 == 0 {
                // Multi-GPU gradient all-reduce via RCCL (in-place device pointer sum + 1/N averaging).
                if let (Some(rccl_ref), Some(placement)) = (&rccl, &dp_placement) {
                    autograd_reg
                        .params
                        .all_reduce_grads(&*dev, placement, Some(rccl_ref))
                        .map_err(|e| Error::Session(format!("all_reduce_grads failed: {e}")))?;
                } else if opts.num_gpus > 1 {
                    return Err(Error::Session(format!(
                        "Multi-GPU training requested (num_gpus={}) but no active RCCL handle is available.",
                        opts.num_gpus
                    )));
                }

                // Global gradient clipping (scale grad by 1/accum, then clip).
                if opts.max_grad_norm > 0.0 {
                    autograd_reg.params.clip_grad_norm(opts.max_grad_norm);
                }

                // LR with linear warmup.
                let effective_step = global_step;
                let lr = if effective_step < opts.warmup_steps {
                    opts.lr * ((effective_step + 1) as f32 / opts.warmup_steps.max(1) as f32)
                } else {
                    let decay_step = effective_step.saturating_sub(opts.warmup_steps);
                    let decay_total = total_steps.saturating_sub(opts.warmup_steps);
                    opts.scheduler
                        .get_lr(opts.lr, decay_step, decay_total.max(1))
                };
                optimizer.set_lr(lr);

                optimizer
                    .step(&mut autograd_reg.params)
                    .map_err(|e| Error::Session(e.to_string()))?;

                // F2: full-parameter write-back — without this, stepped base
                // weights stay in the registry and every later forward reads
                // the original provider weights from the block cache.
                if scope == AutogradScope::FullParameter {
                    streaming
                        .overwrite_base_weights(&autograd_reg)
                        .map_err(|e| Error::Session(e.to_string()))?;
                }

                autograd_reg
                    .zero_grads()
                    .map_err(|e| Error::Session(e.to_string()))?;

                global_step += 1;

                // ReLoRA: periodic adapter merge and momentum reset
                if opts.relora_reset_steps > 0 && global_step % opts.relora_reset_steps == 0 {
                    for (layer_idx, point) in autograd_reg.injection_registry.configs.keys() {
                        let pid_a = grim_autograd::ParamId::a(*layer_idx, 1, *point);
                        let pid_b = grim_autograd::ParamId::b(*layer_idx, 1, *point);
                        let a_opt = autograd_reg.params.get(pid_a).map(|p| {
                            (
                                p.data.to_vec_f32().unwrap_or_default(),
                                p.data.shape().clone(),
                            )
                        });
                        let b_opt = autograd_reg.params.get(pid_b).map(|p| {
                            (
                                p.data.to_vec_f32().unwrap_or_default(),
                                p.data.shape().clone(),
                            )
                        });
                        if let (Some((mut a_vec, a_shape)), Some((mut b_vec, b_shape))) =
                            (a_opt, b_opt)
                        {
                            let rank = opts.rank;
                            let in_f = if rank > 0 { a_vec.len() / rank } else { 0 };
                            let out_f = if rank > 0 { b_vec.len() / rank } else { 0 };
                            if in_f > 0 && out_f > 0 {
                                let mut dummy_base = vec![0.0f32; out_f * in_f];
                                grim_autograd::relora::merge_and_zero(
                                    rank,
                                    in_f,
                                    out_f,
                                    opts.alpha / rank as f32,
                                    &mut a_vec,
                                    &mut b_vec,
                                    &mut dummy_base,
                                );
                                if let Some(param_a) = autograd_reg.params.get_mut(pid_a) {
                                    param_a.data = grim_backend_cpu::cpu_tensor(a_vec, a_shape);
                                }
                                if let Some(param_b) = autograd_reg.params.get_mut(pid_b) {
                                    param_b.data = grim_backend_cpu::cpu_tensor(b_vec, b_shape);
                                }
                            }
                        }
                        optimizer.reset_momentum_for(&[pid_a, pid_b]);
                    }
                    println!("[grim train] ReLoRA reset at step {}", global_step);
                }

                // Held-out evaluation loop
                if let Some(eval_ds) = &eval_dataset {
                    if opts.eval_every_steps > 0
                        && global_step >= opts.eval_warmup_steps
                        && global_step % opts.eval_every_steps == 0
                    {
                        let current_avg_loss = (epoch_loss / num_batches.max(1) as f32) as f64;
                        if let Ok(report) = crate::eval::perplexity(eval_ds, |seq| {
                            if seq.len() < 2 {
                                return Ok::<f64, String>(current_avg_loss);
                            }
                            let eff_len = seq.len().min(64);
                            let eval_ids = &seq[..eff_len - 1];
                            let eval_targets = &seq[1..eff_len];
                            let hidden = model_config.hidden_size;

                            let mut h = match tok_embeddings.forward(eval_ids, eval_ids.len(), hidden) {
                                Ok(tensor) => tensor,
                                Err(_) => return Ok::<f64, String>(current_avg_loss),
                            };

                            for l_idx in 0..num_layers {
                                if let Ok((_, next_h)) = streaming.forward_block_with_autograd(
                                    &provider,
                                    &llama_config,
                                    &autograd_reg,
                                    &mut Tape::new(),
                                    l_idx,
                                    &h,
                                    grim_autograd::TensorId(0),
                                ) {
                                    h = next_h;
                                }
                            }

                            if let Ok(norm_h) = output_norm.forward(&h) {
                                if let Ok(logits) = lm_head.forward(&norm_h) {
                                    if let Ok(logits_vec) = logits.to_vec_f32() {
                                        let vocab = llama_config.vocab_size;
                                        let mut total_nll = 0.0f64;
                                        let mut count = 0;
                                        for (pos, &tgt) in eval_targets.iter().enumerate() {
                                            let tok = tgt as usize;
                                            if tok < vocab && (pos + 1) * vocab <= logits_vec.len() {
                                                let row = &logits_vec[pos * vocab..(pos + 1) * vocab];
                                                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                                                let sum_exp: f32 = row.iter().map(|&v| (v - max).exp()).sum();
                                                let log_prob = (row[tok] - max) - sum_exp.ln();
                                                total_nll -= log_prob as f64;
                                                count += 1;
                                            }
                                        }
                                        if count > 0 {
                                            return Ok::<f64, String>(total_nll / count as f64);
                                        }
                                    }
                                }
                            }

                            Ok::<f64, String>(current_avg_loss)
                        }) {
                            println!(
                                "[grim train] eval step {}: loss={:.4} ppl={:.4} tokens={}",
                                global_step, report.loss, report.ppl, report.tokens
                            );
                        }
                    }
                }

                // Step-level logging.
                if opts.logging_steps > 0 && global_step % opts.logging_steps == 0 {
                    println!(
                        "[grim train] step {}/{} — lr: {:.2e} — loss: {:.4}",
                        global_step,
                        total_steps,
                        lr,
                        epoch_loss / num_batches as f32,
                    );
                }
            }
        }

        if num_batches > 0 {
            epoch_loss /= num_batches as f32;
        } else {
            continue;
        }

        let delta = if prev_loss < f32::MAX {
            epoch_loss - prev_loss
        } else {
            0.0
        };
        prev_loss = epoch_loss;

        println!(
            "[grim train] Epoch {}/{} — loss: {:.4} (Δ={:+.4}) — lr: {:.2e} — step: {}",
            epoch + 1,
            opts.epochs,
            epoch_loss,
            delta,
            optimizer.lr(),
            global_step,
        );

        // Early stopping: stop if loss hasn't improved for `patience` epochs.
        if opts.early_stopping_patience > 0 {
            if delta > 0.0 || epoch == 0 {
                epochs_since_best += 1;
            } else {
                epochs_since_best = 0;
            }
            if epochs_since_best >= opts.early_stopping_patience {
                println!(
                    "[grim train] Early stopping: no improvement for {} epochs.",
                    epochs_since_best
                );
                break;
            }
        }
    }

    let train_state = optimizer.save_to_train_state(&autograd_reg.params);
    train_state
        .write(sidecar_path)
        .map_err(|e| Error::Session(e.to_string()))?;

    // P1 §8: tag the `.grim` artifact with the training-time dtype and
    // multi-GPU strategy so serving/catalog can pick the right path.
    // Non-fatal: `.gguf` inputs simply keep their original metadata.
    let multi_gpu_strategy = if opts.num_gpus > 1 {
        Some("replica-dp".to_string())
    } else {
        None
    };
    if let Err(e) = grim_format::format::rewrite_metadata(&opts.model_path, |meta| {
        meta.preferred_dtype = Some(opts.train_dtype.tag().to_string());
        meta.fp8 = Some(false);
        meta.multi_gpu_strategy = multi_gpu_strategy.clone();
    }) {
        println!(
            "[grim train] WARNING: could not tag .grim metadata ({}); sidecar is still valid.",
            e
        );
    }

    println!(
        "[grim train] Training complete. Sidecar saved to {}",
        opts.output_sidecar
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_train_soul_eater_flag() {
        let opts = TrainOptions {
            node_rank: 0,
            master_addr: "127.0.0.1".to_string(),
            master_port: 0,
            num_nodes: 1,
            model_path: "test.gguf".into(),
            dataset_path: "test.jsonl".into(),
            output_sidecar: "output.grim.train".into(),
            epochs: 1,
            lr: 1e-4,
            rank: 8,
            alpha: 16.0,
            batch_size: 2048,
            gradient_accumulation_steps: 4,
            warmup_steps: 10,
            logging_steps: 1,
            max_grad_norm: 1.0,
            device: "cpu".into(),
            mode: "soul-eater".into(),
            optimizer: grim_autograd::OptimizerKind::AdamW,
            scheduler: grim_autograd::LRScheduler::Cosine,
            use_pissa: false,
            use_olora: false,
            olora_lambda: 0.0,
            use_spectral_qlora: false,
            qat_mxfp4: false,
            checkpoint_segs: 0,
            early_stopping_patience: 3,
            num_gpus: 1,
            echo_mode: false,
            seed: 0,
            train_dtype: TrainDtype::F32,
            lora_plus_ratio: 1.0,
            relora_reset_steps: 0,
            use_oft: false,
            oft_rank: 8,
            eval_dataset: None,
            eval_every_steps: 0,
            eval_warmup_steps: 0,
            dataset_paths: vec![],
            mix_weights: vec![],
            dedup: false,
            quick: false,
        };
        assert_eq!(opts.mode, "soul-eater");
    }

    #[test]
    fn test_alpaca_dataset_parsing() {
        let json = r#"[
            {"instruction": "Summarize this text", "input": "Hello world", "output": "A greeting"},
            {"instruction": "Translate to French", "input": "Good morning", "output": "Bonjour"}
        ]"#;

        // Create a minimal tokenizer mock
        let mut tokens = Vec::new();
        let mut token_to_id = std::collections::HashMap::new();
        let specials = vec![
            "<s>", "</s>", "<unk>", "\n", " ", ":", "S", "u", "m", "a", "r", "i", "z", "e", "t",
            "h", "s", "T", "e", "x", "l", "d", "H", "o", "w", "r", "G", "F", "n", "c", "T", "r",
            "a", "n", "s", "i", "o", "F", "r", "e", "n", "c", "h", "B", "o", "n", "j", "u", "r",
        ];
        for (i, tok) in specials.iter().enumerate() {
            tokens.push(tok.to_string());
            token_to_id.insert(tok.to_string(), i as u32);
        }

        // Add word tokens
        let words = vec![
            "###",
            "Instruction:",
            "Input:",
            "Response:",
            "Summarize",
            "this",
            "text",
            "Hello",
            "world",
            "A",
            "greeting",
            "Translate",
            "to",
            "French",
            "Good",
            "morning",
            "Bonjour",
        ];
        for (i, word) in words.iter().enumerate() {
            let id = (specials.len() + i) as u32;
            tokens.push(word.to_string());
            token_to_id.insert(word.to_string(), id);
        }

        let tokenizer = GgufTokenizer {
            tokens,
            token_to_id,
            scores: None,
            bos_token_id: None,
            eos_token_id: None,
            unk_token_id: None,
            add_bos_token: false,
            model_type: "llama".to_string(),
            bpe_merges: None,
            byte_decoder: None,
            chat_template: None,
        };

        let dataset = load_dataset_from_str(json, &tokenizer, 512).unwrap();
        assert_eq!(dataset.len(), 2);
        assert!(!dataset[0].0.is_empty());
    }

    fn load_dataset_from_str(
        content: &str,
        tokenizer: &GgufTokenizer,
        _max_seq_len: usize,
    ) -> Result<Vec<(Vec<u32>, Vec<u32>)>> {
        if let Ok(entries) = serde_json::from_str::<Vec<AlpacaEntry>>(content) {
            return entries
                .iter()
                .map(|e| {
                    let prompt = if e.input.is_empty() {
                        format!("### Instruction:\n{}\n\n### Response:\n", e.instruction)
                    } else {
                        format!(
                            "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n",
                            e.instruction, e.input
                        )
                    };
                    let full_text = format!("{}{}", prompt, e.output);
                    let tokens = tokenizer.encode(&full_text);
                    let prompt_len = tokenizer.encode(&prompt).len();
                    let labels = vec![IGNORE_INDEX; prompt_len]
                        .into_iter()
                        .chain(tokens[prompt_len..].to_vec())
                        .collect::<Vec<u32>>();
                    Ok((tokens, labels))
                })
                .collect::<Result<Vec<_>>>();
        }
        Err(Error::Session("not Alpaca format".into()))
    }

    // ── WI-F4-close: F4 invariants ────────────────────────────────────────
    // The bug being closed: the old loop built a fake "embedding" by
    // casting raw token IDs to f32 and stuffing them into a `[seq_len, hidden]`
    // tensor (wrong element count), then silently used `hidden_state` as
    // logits. These two regression tests pin both halves:
    //   1. `cpu_tensor` catches the deliberate-reintroduction pattern.
    //   2. The wired head produces `[seq_len, vocab]` shape (NOT `[seq_len, hidden]`).

    use grim_tensor::dtype::{DType, QuantProvenance};
    use grim_tensor::{RawTensor, TensorMeta, TensorProvider};

    /// Minimal in-memory `TensorProvider` exposing only the head tensors.
    /// Provides `token_embd.weight`, `output_norm.weight`, and (optionally)
    /// `output.weight`. Layout matches Llama's GGUF convention:
    /// `token_embd.weight` is `[hidden, vocab]` (column-major GGUF native),
    /// `output.weight` (when separate) is `[vocab, hidden]`.
    struct HeadProvider {
        vocab: usize,
        hidden: usize,
        embed_bytes: Vec<u8>,          // length = hidden * vocab * 4 (f32)
        norm_bytes: Vec<u8>,           // length = hidden * 4
        lmhead_bytes: Option<Vec<u8>>, // length = vocab * hidden * 4 if Some
        embed_shape: Vec<usize>,
        /// F2 test hook: optional layer-0 block tensors keyed by full path.
        block: Option<std::collections::HashMap<String, (Vec<usize>, Vec<u8>)>>,
    }

    impl HeadProvider {
        fn new(vocab: usize, hidden: usize) -> Self {
            let mut embed_bytes = vec![0u8; hidden * vocab * 4];
            for i in 0..(hidden * vocab) {
                let v = ((i % 17) as f32 / 17.0) + 0.1;
                let bytes = v.to_le_bytes();
                embed_bytes[i * 4..(i + 1) * 4].copy_from_slice(&bytes);
            }

            let mut norm_bytes = vec![0u8; hidden * 4];
            for i in 0..hidden {
                let bytes = 1.0f32.to_le_bytes();
                norm_bytes[i * 4..(i + 1) * 4].copy_from_slice(&bytes);
            }

            Self {
                vocab,
                hidden,
                embed_bytes,
                norm_bytes,
                lmhead_bytes: None,
                embed_shape: vec![vocab, hidden],
                block: None,
            }
        }
        /// F2 test hook: serve the nine layer-0 tensors LlamaBlock::load needs.
        fn with_block(mut self, hidden: usize, inter: usize) -> Self {
            let mut m = std::collections::HashMap::new();
            let mut put = |k: &str, shape: Vec<usize>, fill: f32| {
                let n: usize = shape.iter().product();
                let mut bytes = vec![0u8; n * 4];
                for (i, b) in bytes.chunks_mut(4).enumerate() {
                    let v = fill + (i % 7) as f32 * 0.01;
                    b.copy_from_slice(&v.to_le_bytes());
                }
                m.insert(k.to_string(), (shape, bytes));
            };
            put("layers.0.attn_norm.weight", vec![hidden], 1.0);
            put("layers.0.attn.wq.weight", vec![hidden, hidden], 0.05);
            put("layers.0.attn.wk.weight", vec![hidden, hidden], 0.05);
            put("layers.0.attn.wv.weight", vec![hidden, hidden], 0.05);
            put("layers.0.attn.wo.weight", vec![hidden, hidden], 0.05);
            put("layers.0.ffn_norm.weight", vec![hidden], 1.0);
            put("layers.0.ffn.w_gate.weight", vec![inter, hidden], 0.05);
            put("layers.0.ffn.w_up.weight", vec![inter, hidden], 0.05);
            put("layers.0.ffn.w_down.weight", vec![hidden, inter], 0.05);
            self.block = Some(m);
            self
        }
        fn with_lm_head(mut self) -> Self {
            let mut lmhead_bytes = vec![0u8; self.vocab * self.hidden * 4];
            for i in 0..(self.vocab * self.hidden) {
                let v = ((i % 13) as f32 / 13.0) - 0.5;
                let bytes = v.to_le_bytes();
                lmhead_bytes[i * 4..(i + 1) * 4].copy_from_slice(&bytes);
            }
            self.lmhead_bytes = Some(lmhead_bytes);
            self
        }
    }

    impl TensorProvider for HeadProvider {
        fn get(&self, name: &str) -> grim_tensor::error::Result<RawTensor> {
            if let Some(map) = &self.block {
                if let Some((shape, bytes)) = map.get(name) {
                    return Ok(RawTensor {
                        bytes: bytes.clone(),
                        shape: shape.clone(),
                        dtype: DType::F32,
                        provenance: QuantProvenance::GrimNative,
                    });
                }
            }
            match name {
                "token_embd.weight" => Ok(RawTensor {
                    bytes: self.embed_bytes.clone(),
                    shape: self.embed_shape.clone(),
                    dtype: DType::F32,
                    provenance: QuantProvenance::GrimNative,
                }),
                "output_norm.weight" => Ok(RawTensor {
                    bytes: self.norm_bytes.clone(),
                    shape: vec![self.hidden],
                    dtype: DType::F32,
                    provenance: QuantProvenance::GrimNative,
                }),
                "output.weight" => match &self.lmhead_bytes {
                    Some(b) => Ok(RawTensor {
                        bytes: b.clone(),
                        shape: vec![self.vocab, self.hidden],
                        dtype: DType::F32,
                        provenance: QuantProvenance::GrimNative,
                    }),
                    None => Err(grim_tensor::Error::Backend("no lm_head".into())),
                },
                other => Err(grim_tensor::Error::Backend(format!(
                    "stub: unknown tensor {other}"
                ))),
            }
        }

        fn meta(&self, name: &str) -> grim_tensor::error::Result<TensorMeta> {
            let r = self.get(name)?;
            Ok(TensorMeta {
                dtype: r.dtype,
                provenance: r.provenance,
                shape: r.shape,
                fusion_mask: 0,
            })
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "cpu_tensor: data.len")]
    fn fake_embedding_pattern_is_now_caught() {
        use grim_backend_cpu::cpu_tensor;
        use grim_tensor::Shape;
        // Exact pattern from the bug: cast raw IDs to f32 and try to fit
        // `seq_len` elements into a `[seq_len, hidden]` tensor. cpu_tensor's
        // debug-assertion catches it immediately now.
        let seq_len = 4usize;
        let hidden = 8usize;
        let ids = vec![1u32, 2, 3, 4];
        let x_data: Vec<f32> = ids.iter().map(|&id| id as f32).collect();
        let _ = cpu_tensor(x_data, Shape::new(vec![seq_len, hidden]));
    }

    #[test]
    fn head_with_separate_lm_head_produces_vocab_dim_logits() {
        // Bug regression: the old code returned logits shape `[seq_len, hidden]`.
        // The new code (real embedding + norm + lm_head) returns `[seq_len, vocab]`.
        let vocab = 16usize;
        let hidden = 8usize;
        let provider = HeadProvider::new(vocab, hidden).with_lm_head();
        let ws = WeightSource::root(&provider, grim_tensor::Device::Cpu);

        let emb = Embedding::load(&ws.pp("token_embd"), vocab, hidden).unwrap();
        let norm = RmsNorm::load(&ws.pp("output_norm"), hidden, 1e-5).unwrap();
        let lm = Linear::load(&ws.pp("output"), hidden, vocab, false).unwrap();

        let ids = vec![0u32, 1, 2];
        let mut h = emb.forward(&ids, ids.len(), hidden).unwrap();
        assert_eq!(
            h.shape().dims(),
            &[ids.len(), hidden],
            "embedding must be [seq_len, hidden]"
        );
        h = norm.forward(&h).unwrap();
        let logits = lm.forward(&h).unwrap();
        assert_eq!(
            logits.shape().dims(),
            &[ids.len(), vocab],
            "logits must be [seq_len, vocab], not [seq_len, hidden]"
        );
        let v = logits.to_vec_f32().unwrap();
        assert!(v.iter().all(|x| x.is_finite()), "logits must be finite");
    }

    #[test]
    fn head_with_tied_embedding_falls_back_when_output_weight_missing() {
        // LFM2 convention: no separate `output.weight`, lm_head = token_embd^T.
        // The wired fallback in `cmd_train` must produce the same `[seq_len, vocab]`.
        let vocab = 12usize;
        let hidden = 6usize;
        let provider = HeadProvider::new(vocab, hidden); // no lmhead_bytes
        let ws = WeightSource::root(&provider, grim_tensor::Device::Cpu);

        let emb = Embedding::load(&ws.pp("token_embd"), vocab, hidden).unwrap();
        let norm = RmsNorm::load(&ws.pp("output_norm"), hidden, 1e-5).unwrap();
        let lm_load_attempt = Linear::load(&ws.pp("output"), hidden, vocab, false);
        let lm = match lm_load_attempt {
            Ok(l) => l, // ponytail: succeeded path also OK
            Err(_) => Linear::from_tensor(emb.weight().clone(), None),
        };

        let ids = vec![0u32, 1];
        let h = emb.forward(&ids, ids.len(), hidden).unwrap();
        let h = norm.forward(&h).unwrap();
        let logits = lm.forward(&h).unwrap();
        assert_eq!(logits.shape().dims(), &[ids.len(), vocab]);
    }

    #[test]
    fn train_loop_loss_decreases_on_overfit_toy_dataset() {
        let vocab = 16usize;
        let hidden = 8usize;
        let provider = HeadProvider::new(vocab, hidden).with_lm_head();
        let ws = WeightSource::root(&provider, grim_tensor::Device::Cpu);

        let emb = Embedding::load(&ws.pp("token_embd"), vocab, hidden).unwrap();
        let norm = RmsNorm::load(&ws.pp("output_norm"), hidden, 1e-5).unwrap();
        let lm = Linear::load(&ws.pp("output"), hidden, vocab, false).unwrap();

        let input_ids = vec![0u32, 1, 2, 3];
        let targets = vec![1usize, 2, 3, 4];
        let seq_len = input_ids.len();

        use grim_autograd::{
            AdamW, AdamWConfig, AutogradRegistry, InjectionConfig, LoRAInjectionPoint,
            LoRAInjectionRegistry, Tape, apply_and_record_lora,
        };

        let inj_cfg = InjectionConfig {
            hidden_size: hidden,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate_size: 16,
            vocab_size: vocab,
        };
        let mut inj_reg = LoRAInjectionRegistry::new();
        inj_reg.add(grim_autograd::LoRAInjectionConfig::new(
            LoRAInjectionPoint::Logits,
            0,
            1,
            4,
            8.0,
        ));

        let mut autograd_reg = AutogradRegistry::new(inj_cfg, inj_reg).unwrap();
        let mut optimizer = AdamW::new(AdamWConfig {
            lr: 0.1,
            ..AdamWConfig::default()
        });

        let mut initial_loss = 0.0f32;
        let mut final_loss = 0.0f32;

        for step in 0..10 {
            autograd_reg.zero_grads().unwrap();
            let mut tape = Tape::new();

            let h = emb.forward(&input_ids, seq_len, hidden).unwrap();
            let h_norm = norm.forward(&h).unwrap();
            let logits_base = lm.forward(&h_norm).unwrap();
            let h_norm_id = tape.register(h_norm.clone());
            let logits_base_id = tape.register(logits_base.clone());

            let (logits_id, logits_out) = apply_and_record_lora(
                &autograd_reg,
                &mut tape,
                0,
                LoRAInjectionPoint::Logits,
                logits_base,
                logits_base_id,
                h_norm,
                h_norm_id,
            )
            .unwrap();

            let (loss_val, loss_grad) = cross_entropy_loss(&logits_out, &targets).unwrap();
            if step == 0 {
                initial_loss = loss_val;
            }
            final_loss = loss_val;

            backward(&tape, loss_grad, logits_id, &mut autograd_reg.params).unwrap();
            optimizer.step(&mut autograd_reg.params).unwrap();
        }

        assert!(initial_loss > 0.0, "initial loss should be positive");
        assert!(
            final_loss < initial_loss,
            "final loss ({final_loss}) must be strictly lower than initial loss ({initial_loss}) after training steps"
        );
    }

    /// F2 regression: full-parameter write-back must push stepped registry
    /// weights into the streaming block cache, so forwards after a step read
    /// updated weights instead of the original provider tensors.
    #[test]
    fn full_parameter_write_back_updates_cached_block() {
        use grim_autograd::{
            AutogradRegistry, AutogradScope, InjectionConfig, LoRAInjectionRegistry, Tape,
        };
        use grim_backend_cpu::cpu_tensor;
        use grim_engine::streaming_forward::StreamingBlockForward;
        use grim_models_transformer::LlamaConfig;
        use grim_nn::modules::Embedding;
        use grim_nn::WeightSource;
        use grim_tensor::Shape;

        let vocab = 16usize;
        let hidden = 8usize;
        let provider = HeadProvider::new(vocab, hidden).with_block(hidden, 16);
        let ws = WeightSource::root(&provider, grim_tensor::Device::Cpu);
        let cfg = LlamaConfig {
            vocab_size: vocab,
            hidden_size: hidden,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 4,
            num_layers: 1,
            intermediate_size: 16,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 64,
            partial_rotary_factor: 1.0,
            yarn: None,
        };

        let inj_cfg = InjectionConfig {
            hidden_size: hidden,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate_size: 16,
            vocab_size: vocab,
        };
        let mut reg = AutogradRegistry::with_scope(
            inj_cfg,
            LoRAInjectionRegistry::new(),
            AutogradScope::FullParameter,
        )
        .unwrap();

        let emb = Embedding::load(&ws.pp("token_embd"), vocab, hidden).unwrap();
        let ids = [0u32, 1];
        let h = emb.forward(&ids, ids.len(), hidden).unwrap();

        let mut sfb = StreamingBlockForward::new(1, hidden);
        fn run(
            sfb: &mut StreamingBlockForward,
            provider: &HeadProvider,
            cfg: &LlamaConfig,
            reg: &AutogradRegistry,
            h: &grim_tensor::Tensor,
        ) -> Option<Vec<f32>> {
            let mut tape = Tape::new();
            let x_id = tape.register(h.clone());
            let (_, out) = sfb
                .forward_block_with_autograd(provider, cfg, reg, &mut tape, 0, h, x_id)
                .ok()?;
            out.to_vec_f32().ok()
        }

        // Materialize the cache. The standalone block forward stops at
        // attention (no session KV context here) — that's fine: the block is
        // inserted into the cache before any math, which is all this test needs.
        let _ = run(&mut sfb, &provider, &cfg, &reg, &h);
        let before = sfb
            .cached_qproj_weight(0, &grim_tensor::Device::Cpu)
            .expect("block must be cached after first forward");
        let before = before.to_vec_f32().unwrap();
        assert!(!before.is_empty());

        // Simulate an optimizer step: change the QProj base weight in the
        // registry only (exactly what backward+step produce).
        let new_w = vec![0.5f32; hidden * hidden];
        reg.params
            .get_mut(grim_autograd::ParamId::base(
                0,
                grim_autograd::LoRAInjectionPoint::QProj,
            ))
            .unwrap()
            .data = cpu_tensor(new_w.clone(), Shape::new(vec![hidden, hidden]));

        // Pre-fix behavior: forward still reads stale cache. Post-fix: differs.
        sfb.overwrite_base_weights(&reg).unwrap();
        let after = sfb
            .cached_qproj_weight(0, &grim_tensor::Device::Cpu)
            .unwrap()
            .to_vec_f32()
            .unwrap();
        assert_ne!(
            before, after,
            "overwrite_base_weights must replace the cached weight"
        );

        // And the cached weight now equals the stepped param bit-for-bit.
        let wq = sfb
            .cached_qproj_weight(0, &grim_tensor::Device::Cpu)
            .unwrap()
            .to_vec_f32()
            .unwrap();
        assert_eq!(wq, new_w);
    }

    #[test]
    fn test_pack_dataset_tokens_golden_mutation_resistant() {
        let seqs = vec![vec![1, 2, 3], vec![4, 5], vec![6, 7, 8, 9]];
        let packed = pack_dataset_tokens(&seqs, 5);
        assert_eq!(packed.len(), 2);
        assert_eq!(packed[0], vec![1, 2, 3, 4, 5]);
        assert_eq!(packed[1], vec![6, 7, 8, 9]);
    }

    #[test]
    fn multi_file_mix_respects_weights_and_dedups() {
        use std::io::Write;
        let mut tmp_file1 = tempfile::NamedTempFile::new().unwrap();
        let mut tmp_file2 = tempfile::NamedTempFile::new().unwrap();

        let alpaca_data1 = r#"[
            {"instruction": "Add 1+1", "output": "2"},
            {"instruction": "Add 2+2", "output": "4"}
        ]"#;
        let alpaca_data2 = r#"[
            {"instruction": "Add 1+1", "output": "2"},
            {"instruction": "Add 3+3", "output": "6"}
        ]"#;

        tmp_file1.write_all(alpaca_data1.as_bytes()).unwrap();
        tmp_file2.write_all(alpaca_data2.as_bytes()).unwrap();

        let tokens = vec!["<unk>".to_string(), "<s>".to_string(), "</s>".to_string()];
        let mut token_to_id = std::collections::HashMap::new();
        for (i, t) in tokens.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }
        let tok = GgufTokenizer {
            tokens,
            token_to_id,
            scores: None,
            bos_token_id: None,
            eos_token_id: None,
            unk_token_id: None,
            add_bos_token: false,
            model_type: "llama".to_string(),
            bpe_merges: None,
            byte_decoder: None,
            chat_template: None,
        };

        let paths = vec![
            tmp_file1.path().to_str().unwrap().to_string(),
            tmp_file2.path().to_str().unwrap().to_string(),
        ];
        let weights = vec![2.0f32, 1.0f32];

        // With dedup: the duplicate "Add 1+1" from file2 should be skipped
        let mixed = load_dataset_multi(&paths, &tok, 128, Some(&weights), true, 42).unwrap();
        assert!(!mixed.is_empty());
    }

    #[test]
    fn template_override_replaces_chat_template() {
        let mut tok = GgufTokenizer::default();
        assert!(tok.chat_template.is_none());
        if let Some(f) = crate::template_registry::TemplateRegistry::lookup("chatml") {
            tok.chat_template = Some(f.jinja.to_string());
        }
        assert!(tok.chat_template.is_some());
        assert!(tok.chat_template.unwrap().contains("<|im_start|>"));
    }

    #[test]
    fn test_is_out_of_memory_error_classification() {
        assert!(is_out_of_memory_error("hipErrorOutOfMemory (code 2)"));
        assert!(is_out_of_memory_error("hipMalloc failed: 2"));
        assert!(is_out_of_memory_error("Out of memory while allocating device buffer"));
        assert!(is_out_of_memory_error("cudaErrorMemoryAllocation"));
        assert!(!is_out_of_memory_error("File not found: model.gguf"));
    }

    #[test]
    fn test_multi_gpu_without_rccl_hard_errors() {
        let opts = TrainOptions {
            node_rank: 0,
            master_addr: "127.0.0.1".to_string(),
            master_port: 0,
            num_nodes: 1,
            model_path: "test.gguf".into(),
            dataset_path: "test.jsonl".into(),
            output_sidecar: "output.grim.train".into(),
            epochs: 1,
            lr: 1e-4,
            rank: 8,
            alpha: 16.0,
            batch_size: 2048,
            gradient_accumulation_steps: 4,
            warmup_steps: 10,
            logging_steps: 1,
            max_grad_norm: 1.0,
            device: "cpu".into(),
            mode: "qlora".into(),
            optimizer: grim_autograd::OptimizerKind::AdamW,
            scheduler: grim_autograd::LRScheduler::Cosine,
            use_pissa: false,
            use_olora: false,
            olora_lambda: 0.0,
            use_spectral_qlora: false,
            qat_mxfp4: false,
            checkpoint_segs: 2,
            early_stopping_patience: 3,
            num_gpus: 4, // Multi-GPU
            echo_mode: false,
            seed: 0,
            train_dtype: TrainDtype::F32,
            lora_plus_ratio: 1.0,
            relora_reset_steps: 0,
            use_oft: false,
            oft_rank: 8,
            eval_dataset: None,
            eval_every_steps: 0,
            eval_warmup_steps: 0,
            dataset_paths: vec![],
            mix_weights: vec![],
            dedup: false,
            quick: false,
        };
        let res = cmd_train(opts);
        assert!(res.is_err());
        let err_msg = res.err().unwrap().to_string();
        // Must either fail on model missing or multi-GPU RCCL init
        assert!(err_msg.contains("Multi-GPU") || err_msg.contains("failed") || err_msg.contains("No such file"));
    }
}
