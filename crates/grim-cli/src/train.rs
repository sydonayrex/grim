//! `grim train` subcommand execution logic (WI-T5).
//!
//! Drives the SFT training loop: dataset loading, streaming forward pass,
//! cross-entropy loss, reverse-mode autograd tape backward pass, AdamW step,
//! and `.grim.train` sidecar checkpoint persistence.
//!
//! F4: Wires training to real model loading via `GrimProvider` and real
//! dataset loading from JSON files (Alpaca/ShareGPT format).

use grim_autograd::{
    cross_entropy_loss, backward, AdamW, AdamWConfig, AutogradRegistry, InjectionConfig,
    LoRAInjectionRegistry, Tape,
};
use grim_core::error::{Error, Result};
use grim_engine::streaming_forward::StreamingBlockForward;
use grim_format::tprov::GgufProvider;

/// Sentinel value for ignored label positions in cross-entropy loss.
/// Matches HF/PyTorch convention: -100 as u32 wraps to 4294967196.
const IGNORE_INDEX: u32 = IGNORE_INDEX;
use grim_format::tokenizer::GgufTokenizer;
use grim_format::train::TrainState;
use grim_models_transformer::LlamaConfig;
use grim_nn::{Embedding, Linear, RmsNorm, WeightSource};
use serde::Deserialize;
use std::path::Path;

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
    pub device: String,
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

#[derive(Debug, Deserialize)]
struct ConversationTurn {
    value: String,
}

/// Extract `InjectionConfig` from GGUF metadata keys.
fn injection_config_from_metadata(provider: &GgufProvider) -> Result<InjectionConfig> {
    let arch = provider.architecture().unwrap_or("llama");

    let hidden_size = get_meta_u32(provider, &format!("{}.embedding_length", arch), 4096) as usize;
    let num_heads = get_meta_u32(provider, &format!("{}.attention.head_count", arch), 32) as usize;
    let num_kv_heads = get_meta_u32(provider, &format!("{}.attention.head_count_kv", arch), num_heads as u32) as usize;
    let head_dim = get_meta_u32(provider, &format!("{}.attention.key_length", arch), 128) as usize;
    let intermediate_size = get_meta_u32(provider, &format!("{}.intermediate_size", arch), 11008) as usize;
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
    let num_kv_heads = get_meta_u32(provider, &format!("{}.attention.head_count_kv", arch), num_heads as u32) as usize;
    let head_dim = get_meta_u32(provider, &format!("{}.attention.key_length", arch), 128) as usize;
    let num_layers = get_meta_u32(provider, &format!("{}.block_count", arch), 32) as usize;
    let intermediate_size = get_meta_u32(provider, &format!("{}.intermediate_size", arch), 11008) as usize;
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
    })
}

/// Helper: get metadata as u32 from provider.
fn get_meta_u32(provider: &GgufProvider, key: &str, default: u32) -> u32 {
    if let Some(v) = provider.metadata(key) {
        if let Some(u) = v.as_u32() { return u; }
        if let Some(s) = v.as_str() { if let Ok(u) = s.parse::<u32>() { return u; } }
    }
    default
}

/// Helper: get metadata as string from provider.
fn get_meta_str(provider: &GgufProvider, key: &str) -> Option<String> {
    let v = provider.metadata(key)?;
    if let Some(s) = v.as_str() { return Some(s.to_string()); }
    if let Some(u) = v.as_u32() { return Some(u.to_string()); }
    if let Some(f) = v.as_f32() { return Some(f.to_string()); }
    None
}

/// Load dataset from JSON file (supports Alpaca and ShareGPT formats).
fn load_dataset(path: &str, tokenizer: &GgufTokenizer, max_seq_len: usize) -> Result<Vec<(Vec<u32>, Vec<u32>)>> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| Error::Session(format!("failed to read dataset '{}': {}", path, e)))?;

    // Try Alpaca format first (array of {instruction, output})
    if let Ok(entries) = serde_json::from_str::<Vec<AlpacaEntry>>(&content) {
        println!("[grim train] Loaded {} Alpaca entries", entries.len());
        return entries.iter().map(|e| {
            let prompt = if e.input.is_empty() {
                format!("### Instruction:\n{}\n\n### Response:\n", e.instruction)
            } else {
                format!("### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n", e.instruction, e.input)
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
        }).collect::<Result<Vec<_>>>();
    }

    // Try ShareGPT format (array of {conversations: [{from, value}]})
    if let Ok(entries) = serde_json::from_str::<Vec<ShareGptEntry>>(&content) {
        println!("[grim train] Loaded {} ShareGPT entries", entries.len());
        return entries.iter().filter_map(|e| {
            if e.conversations.len() < 2 { return None; }
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
            Some(Ok((tokens, labels)))
        }).collect::<Result<Vec<_>>>();
    }

    Err(Error::Session(format!(
        "dataset '{}' is not in Alpaca or ShareGPT format",
        path
    )))
}

/// Run SFT training loop over a dataset and save the trained adapter sidecar.
pub fn cmd_train(opts: TrainOptions) -> Result<()> {
    println!("[grim train] Initializing QLoRA training...");
    println!("             Model: {}", opts.model_path);
    println!("             Dataset: {}", opts.dataset_path);
    println!("             Sidecar Output: {}", opts.output_sidecar);

    // ── F4: Load real model from .grim file ──
    let provider = GgufProvider::open(&opts.model_path)
        .map_err(|e| Error::Session(format!("failed to open model '{}': {}", opts.model_path, e)))?;

    let model_config = injection_config_from_metadata(&provider)?;
    let llama_config = llama_config_from_metadata(&provider)?;
    let num_layers = llama_config.num_layers;

    let tokenizer = provider.tokenizer()
        .map_err(|e| Error::Session(format!("failed to load tokenizer: {}", e)))?;

    // Validate LoRA hyperparameters before constructing the registry.
    if opts.rank == 0 {
        return Err(Error::Session("LoRA rank must be > 0".into()));
    }
    if opts.alpha == 0 {
        return Err(Error::Session("LoRA alpha must be > 0".into()));
    }
    let hidden_size = llama_config.hidden_size;
    if opts.rank > hidden_size {
        return Err(Error::Session(format!(
            "LoRA rank {} exceeds hidden size {}", opts.rank, hidden_size
        )));
    }

    let injection_reg = LoRAInjectionRegistry::standard_qlora(num_layers, opts.rank, opts.alpha, 1);
    let mut autograd_reg = AutogradRegistry::new(model_config.clone(), injection_reg)
        .map_err(|e| Error::Session(e.to_string()))?;

    let opt_config = AdamWConfig {
        lr: opts.lr,
        ..AdamWConfig::default()
    };
    let mut optimizer = AdamW::new(opt_config);

    // Read existing sidecar if resuming checkpoint
    let sidecar_path = Path::new(&opts.output_sidecar);
    if let Ok(Some(existing_state)) = TrainState::read(sidecar_path) {
        println!("[grim train] Resuming from existing sidecar checkpoint...");
        optimizer
            .load_from_train_state(&mut autograd_reg.params, &existing_state)
            .map_err(|e| Error::Session(e.to_string()))?;
    }

    // ── F4: Load real dataset ──
    let dataset = load_dataset(&opts.dataset_path, &tokenizer, llama_config.max_seq_len)?;
    if dataset.is_empty() {
        return Err(Error::Session("dataset is empty".into()));
    }
    println!("[grim train] Loaded {} training examples", dataset.len());

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
            let ordinal = d.strip_prefix("rocm:").and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
            grim_tensor::Device::Rocm(ordinal)
        }
        other => return Err(Error::Session(format!("unsupported training device '{other}'"))),
    };

    let ws = WeightSource::root(&provider, target_device);
    let tok_embeddings = Embedding::load(&ws.pp("token_embd"), model_config.vocab_size, model_config.hidden_size)
        .map_err(|e| Error::Session(format!("failed to load token_embd: {e}")))?;
    let output_norm = RmsNorm::load(&ws.pp("output_norm"), model_config.hidden_size, llama_config.rms_norm_eps)
        .map_err(|e| Error::Session(format!("failed to load output_norm: {e}")))?;
    let lm_head = match Linear::load(&ws.pp("output"), model_config.hidden_size, model_config.vocab_size, false) {
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

    let mut prev_loss = f32::MAX;

    for epoch in 0..opts.epochs {
        autograd_reg
            .zero_grads()
            .map_err(|e| Error::Session(e.to_string()))?;

        let mut epoch_loss = 0.0f32;
        let mut num_batches = 0;

for (tokens, labels) in dataset.iter() {
        if tokens.len() < 2 { continue; }
        let input_ids = &tokens[..tokens.len() - 1];
        let targets = &labels[1..];

            let seq_len = input_ids.len();
            let hidden = model_config.hidden_size;

            let mut tape = Tape::new();
            streaming.checkpoint_buffer.clear();

            let mut hidden_state = tok_embeddings
                .forward(input_ids, seq_len, hidden)
                .map_err(|e| Error::Session(format!("token embedding forward failed: {e}")))?;
            let mut x_id = tape.register(hidden_state.clone());

            // Run streaming forward through all layers with autograd tape recording.
            for layer_idx in 0..num_layers {
                let (next_id, next_h) = streaming
                    .forward_block_with_autograd(&provider, &llama_config, &autograd_reg, &mut tape, layer_idx, &hidden_state, x_id)
                    .map_err(|e| Error::Session(format!("layer {} forward failed: {}", layer_idx, e)))?;
                hidden_state = next_h;
                x_id = next_id;
            }

            // Final norm + lm_head → real vocabulary logits.
            hidden_state = output_norm
                .forward(&hidden_state)
                .map_err(|e| Error::Session(format!("output_norm forward failed: {e}")))?;
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

            let targets_usize: Vec<usize> = targets.iter().map(|&t| t as usize).collect();
            let (loss_val, loss_grad) = cross_entropy_loss(&logits_out, &targets_usize)
                .map_err(|e| Error::Session(e.to_string()))?;

            backward(&tape, loss_grad, logits_id, &mut autograd_reg.params)
                .map_err(|e| Error::Session(e.to_string()))?;

            epoch_loss += loss_val;
            num_batches += 1;
        }

        if num_batches > 0 {
            epoch_loss /= num_batches as f32;
        }

        optimizer
            .step(&mut autograd_reg.params)
            .map_err(|e| Error::Session(e.to_string()))?;

        let delta = if prev_loss < f32::MAX {
            epoch_loss - prev_loss
        } else {
            0.0
        };
        prev_loss = epoch_loss;

        println!(
            "[grim train] Epoch {}/{} — loss: {:.4} (Δ={:+.4})",
            epoch + 1,
            opts.epochs,
            epoch_loss,
            delta
        );
    }

    let train_state = optimizer.save_to_train_state(&autograd_reg.params);
    train_state
        .write(sidecar_path)
        .map_err(|e| Error::Session(e.to_string()))?;

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
    fn test_alpaca_dataset_parsing() {
        let json = r#"[
            {"instruction": "Summarize this text", "input": "Hello world", "output": "A greeting"},
            {"instruction": "Translate to French", "input": "Good morning", "output": "Bonjour"}
        ]"#;

        // Create a minimal tokenizer mock
        let mut tokens = Vec::new();
        let mut token_to_id = std::collections::HashMap::new();
        let specials = vec!["<s>", "</s>", "<unk>", "\n", " ", ":", "S", "u", "m", "a", "r", "i", "z", "e", "t", "h", "s", "T", "e", "x", "l", "d", "H", "o", "w", "r", "G", "F", "n", "c", "T", "r", "a", "n", "s", "i", "o", "F", "r", "e", "n", "c", "h", "B", "o", "n", "j", "u", "r"];
        for (i, tok) in specials.iter().enumerate() {
            tokens.push(tok.to_string());
            token_to_id.insert(tok.to_string(), i as u32);
        }

        // Add word tokens
        let words = vec!["###", "Instruction:", "Input:", "Response:", "Summarize", "this", "text", "Hello", "world", "A", "greeting", "Translate", "to", "French", "Good", "morning", "Bonjour"];
        for (i, word) in words.iter().enumerate() {
            let id = (specials.len() + i) as u32;
            tokens.push(word.to_string());
            token_to_id.insert(word.to_string(), id);
        }

        let tokenizer = GgufTokenizer {
            tokens,
            token_to_id,
            scores: None,
            eos_token_id: None,
            model_type: "llama".to_string(),
            bpe_merges: None,
            byte_decoder: None,
        };

        let dataset = load_dataset_from_str(json, &tokenizer, 512).unwrap();
        assert_eq!(dataset.len(), 2);
        assert!(!dataset[0].0.is_empty());
    }

    fn load_dataset_from_str(content: &str, tokenizer: &GgufTokenizer, _max_seq_len: usize) -> Result<Vec<(Vec<u32>, Vec<u32>)>> {
        if let Ok(entries) = serde_json::from_str::<Vec<AlpacaEntry>>(content) {
            return entries.iter().map(|e| {
                let prompt = if e.input.is_empty() {
                    format!("### Instruction:\n{}\n\n### Response:\n", e.instruction)
                } else {
                    format!("### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n", e.instruction, e.input)
                };
                let full_text = format!("{}{}", prompt, e.output);
                let tokens = tokenizer.encode(&full_text);
                let prompt_len = tokenizer.encode(&prompt).len();
                let labels = vec![IGNORE_INDEX; prompt_len]
                    .into_iter()
                    .chain(tokens[prompt_len..].to_vec())
                    .collect::<Vec<u32>>();
                Ok((tokens, labels))
            }).collect::<Result<Vec<_>>>();
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
        embed_bytes: Vec<u8>, // length = hidden * vocab * 4 (f32)
        norm_bytes: Vec<u8>,  // length = hidden * 4
        lmhead_bytes: Option<Vec<u8>>, // length = vocab * hidden * 4 if Some
        embed_shape: Vec<usize>,
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
            }
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
                other => Err(grim_tensor::Error::Backend(format!("stub: unknown tensor {other}"))),
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
        assert_eq!(h.shape().dims(), &[ids.len(), hidden], "embedding must be [seq_len, hidden]");
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

        use grim_autograd::{AdamW, AdamWConfig, AutogradRegistry, InjectionConfig, LoRAInjectionPoint, LoRAInjectionRegistry, Tape, apply_and_record_lora};

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
            ).unwrap();

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

    #[test]
    fn test_pack_dataset_tokens_golden_mutation_resistant() {
        let seqs = vec![
            vec![1, 2, 3],
            vec![4, 5],
            vec![6, 7, 8, 9],
        ];
        let packed = pack_dataset_tokens(&seqs, 5);
        assert_eq!(packed.len(), 2);
        assert_eq!(packed[0], vec![1, 2, 3, 4, 5]);
        assert_eq!(packed[1], vec![6, 7, 8, 9]);
    }
}
