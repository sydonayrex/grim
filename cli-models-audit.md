# GRIM-CLI & GRIM-MODELS AUDIT

Audit date: 2026-07-26
Scope: `crates/grim-cli/`, `crates/grim-models/`, `crates/grim-models-transformer/`, `crates/grim-models-mamba/`, `crates/grim-models-vision/`, `crates/grim-models-audio/`

---

## 🔴 CRITICAL BUGS

### CRIT-1: `grim run` passes same tensor for input_ids AND positions — BROKEN POSITIONAL ENCODING

**File:** `crates/grim-cli/src/run.rs:356`

```rust
let logits = CausalLm::forward(&*model, &mut session, &input_tensor, &input_tensor, &[])?;
```

The 4th argument should be a `positions` tensor (different from `input_ids`). The code passes `input_tensor` (token IDs as f32) for both. This means:

1. **No positional information is provided** — the model receives identical tensors for token IDs and positions
2. **All models that use the positions tensor will be broken** — this includes Llama (RoPE), GPT2 (absolute positional embeddings), Gemma, DeepSeek, etc.

**Impact:** Any model using positional encoding produces garbage output. The `run.rs` code completely bypasses proper position handling.

**Fix:** Build a proper positions tensor:
```rust
let pos_ids: Vec<f32> = if first_pass {
    (0..input_ids.len()).map(|i| i as f32).collect()
} else {
    vec![tokens.len() as f32 - 1.0]  // for incremental decode
};
// Build positions tensor same way as input_tensor...
```

---

### CRIT-2: `Lfm2Block::load` — incomplete implementation (cuts off mid-function)

**File:** `crates/grim-models/transformer/src/lfm2.rs:74-465`

The `Lfm2Block::load` function is **incomplete** — the source file ends at line 465 with a comment "verified by direct byte-for-byte comparison..." but the actual block construction and return statement are missing. The function loads some weights but never creates/returns the `Lfm2Block` struct.

**Impact:** Any attempt to load an LFM2 model will fail to compile or panic at runtime.

---

### CRIT-3: `Gpt2Block::forward` — NO ATTENTION COMPUTATION

**File:** `crates/grim-models/transformer/src/gpt2.rs:94-106`

```rust
pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
    let norm_x = self.ln_1.forward(x)?;
    let qkv = self.wqkv.forward(&norm_x)?;
    let attn_out = self.c_proj.forward(&qkv)?;  // BUG: no actual attention!
    let x_res1 = add_tensors(x, &attn_out)?;
    // ... FFN follows
}
```

The GPT2 attention is completely broken:
- `wqkv` produces combined Q/K/V projections
- **No split into Q, K, V**
- **No attention scores computed**
- **No softmax**
- Just passes `qkv` directly through `c_proj`

This is a structural placeholder, not a working implementation.

---

### CRIT-4: `GemmaBlock::forward` — FAKE ATTENTION

**File:** `crates/grim-models/transformer/src/gemma.rs:73-90`

```rust
pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
    let norm_x = self.attn_norm.forward(x)?;
    let q = self.wq.forward(&norm_x)?;
    let _k = self.wk.forward(&norm_x)?;
    let _v = self.wv.forward(&norm_x)?;
    // Simple attention approximation
    let attn_out = self.wo.forward(&q)?;  // BUG: uses only Q!
    // ...
}
```

The attention computation is completely missing:
- Computes Q, K, V but **ignores K and V** (bound to `_`)
- Passes only `q` to output projection `wo`
- No attention scores, no softmax, no weighted sum of values

---

### CRIT-5: `DeepSeekBlock::forward` — NO MLA IMPLEMENTATION

**File:** `crates/grim-models/transformer/src/deepseek.rs:77-80`

```rust
pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
    let norm_x = self.attn_norm.forward(x)?;
    let q_latent = self.q_a_proj.forward(&norm_x)?;
    let q = self.q_b_proj.forward(&q_latent)?;
    let kv_latent = self.kv_a_proj.forward(&norm_x)?;
    let kv = self.kv_b_proj.forward(&kv_latent)?;
    let wo = self.wo.forward(&q)?  // BUG: no actual attention!
    // ...
}
```

Multi-head Latent Attention (MLA) is not implemented:
- Projects Q and KV through low-rank bottlenecks (q_a/q_b, kv_a/kv_b)
- **Then just passes `q` through `wo`** — completely ignores KV!
- No attention computation at all

---

### CRIT-6: `Gpt2` uses `LayerNorm` with bias but `LayerNorm::load` requires bias from GGUF

**File:** `crates/grim-models/transformer/src/gpt2.rs:34-45, 75-83`

```rust
pub struct LayerNorm {
    pub weight: Tensor,
    pub bias: Tensor,  // REQUIRED
    pub eps: f32,
}

impl LayerNorm {
    pub fn load(ws: &WeightSource<'_>, dim: usize, eps: f32) -> Result<Self> {
        let weight = ws.get([dim], "weight")?;
        let bias = ws.get([dim], "bias")?;  // FAILS if bias not in GGUF
        Ok(Self { weight, bias, eps })
    }
}
```

GPT2 uses LayerNorm with bias. If the GGUF file doesn't have `bias` for a LayerNorm, this panics. Most GGUF conversions from safetensors omit LayerNorm bias (folded into weights). This will crash on load.

---

### CRIT-7: `run.rs` generation loop doesn't handle EOS — infinite generation until max_tokens

**File:** `crates/grim-cli/src/run.rs:417-421`

```rust
// Note: EOS handling deferred — `GgufTokenizer` does not yet expose
// `tokenizer.ggml.eos_token_id`, so hardcoding `vocab_size - 1`
// incorrectly stops LFM2-family models whose PAD id is the vocab
// ceiling. For now we exhaust `max_tokens` and let the caller decide.
```

This is a **documented known bug** — the generation loop never stops at EOS. For LFM2, PAD == vocab_size-1, so checking EOS against that value would incorrectly stop generation.

---

### CRIT-8: `Gpt2::load` and `Gemma::load` hardcode `Device::Cpu` ignoring the passed device

**Files:** `gpt2.rs:132`, `gemma.rs:112`

```rust
// gpt2.rs:132
pub fn load(ws: &WeightSource<'_>, cfg: Gpt2Config) -> Result<Self> {
    // ...
    device: Device::Cpu,  // IGNORES ws.device()
```

```rust
// gemma.rs:112
Ok(Self {
    cfg,
    device: Device::Cpu,  // IGNORES ws.device()
```

The `WeightSource` is created with the target device (from CLI `--rocm` or auto-detection), but these models force CPU regardless.

---

### CRIT-9: `Gemma` uses tied embeddings for output but NO output projection exists

**File:** `crates/grim-models/transformer/src/gemma.rs:160-177`

```rust
// Gemma weight tying: output projection uses token embedding weights transposed
// logits = h @ weight.T  where weight is [vocab_size, hidden_size]
let weight = &self.tok_embeddings.weight;
let dev = grim_backend_cpu::CpuDevice::new();
let (s, _h) = grim_tensor::BackendDevice::matmul(
    &dev,
    h.storage().as_ref(),
    weight.storage().as_ref(),
    &grim_tensor::Shape::new(vec![seq_len, self.cfg.vocab_size]),
)?;
```

**Mathematical issue:** `h` is `[seq_len, hidden_size]`, `weight` is `[vocab_size, hidden_size]` (row-major). The matmul computes `h @ weight` = `[seq_len, vocab_size]`.

But the GGUF convention stores embeddings as `[vocab, hidden]` and Linear weights as `[out, in]`. For tied weights, the output should be `h @ weight.T` = `h @ [hidden, vocab]` = `[seq_len, vocab]`. Here it computes `h @ [vocab, hidden]` which requires `hidden == vocab` to even work, producing wrong output when they differ.

**The fix:** Transpose the weight before matmul, or use `Linear::from_tensor(tok_embeddings.weight.clone(), None)` which pre-transposes.

---

## 🟠 MAJOR ISSUES

### MAJ-1: `run.rs` builds token tensors fresh on every device type with massive code duplication

**File:** `crates/grim-cli/src/run.rs:321-390`

The same tensor construction logic is repeated for CPU, CUDA, ROCm, Vulkan, Metal — 5 nearly identical code blocks. This is unmaintainable and error-prone.

**Fix:** Extract a `build_tensor(device, data, shape)` helper.

---

### MAJ-2: `run.rs` tokenization uses hardcoded BOS candidates without model awareness

**File:** `crates/grim-cli/src/run.rs:264-271`

```rust
let bos_candidates = ["<|startoftext|>", "<s>", "<s>"];
for bos in &bos_candidates {
    if let Some(&id) = tok.token_to_id.get(*bos) {
        ids.push(id);
        break;
    }
}
```

This assumes specific tokenizer vocabularies. Different models use different BOS tokens. Should derive from model config or tokenizer metadata.

---

### MAJ-3: `Lfm2Config` includes `is_recr: Vec<bool>` but no validation against `num_layers`

**File:** `crates/grim-models/transformer/src/lfm2.rs:24-25`

```rust
pub is_recr: Vec<bool>, // Whether each layer is recurrent
```

If `is_recr.len() != num_layers`, `cfg.is_recr.get(layer_idx)` will return `None` and default to `false` silently (line 77: `.copied().unwrap_or(false)`). This misconfigures the model architecture silently.

---

### MAJ-4: `apply_adapters_to_logits` CPU path is a structural placeholder that doesn't match GPU semantics

**File:** `crates/grim-models/transformer/src/lora.rs:33-47`

```rust
if adapters.is_empty() {
    return Ok(logits.clone());
}
let shape_dims = logits.shape().dims().to_vec();
if shape_dims.len() != 2 {
    // CPU structural placeholder — GPU path fuses this into the
    // output projection. Anything other than `[seq, vocab]` is
    // a misuse here; return the input untouched.
    return Ok(logits.clone());
}
```

If logits are 3D (e.g., `[batch, seq, vocab]` from some models), the CPU path silently skips LoRA application. The comment says "GPU path fuses this" but there's no guarantee the GPU path is used.

---

### MAJ-5: Oxidizer CLI `cmd_oxidizer_convert` writes importance scores but doesn't persist all metadata

**File:** `crates/grim-cli/src/oxidizer.rs:204-294`

The `grim_meta` is built but only `quant_overrides` are passed to `convert_to_grim`. The `grim_meta` includes `rocm_fusion_ops`, `kv_layout_optimized`, `train_fusion_ops`, etc. but these are not forwarded.

---

### MAJ-6: Service manager `ExecStart` uses `run --serve` but `grim run --serve` expects a model argument

**File:** `crates/grim-cli/src/service.rs:168, 381`

```rust
// systemd unit:
ExecStart={} run --serve --config {}
// launchd plist:
<string>run</string>
<string>--serve</string>
<string>--config</string>
<string>{config}</string>
```

But `Commands::Run` in `main.rs:67-108` requires a `model: Option<String>` argument. The service will fail to start because no model is specified.

The `Run` command does support `serve` mode (line 587 in main.rs), but it requires the model to be specified. The service config doesn't pass a model.

---

### MAJ-7: `Gpt2` position embeddings loaded from GGUF but no validation of `max_seq_len`

**File:** `crates/grim-models/transformer/src/gpt2.rs:122, 178`

```rust
let wpe = Embedding::load(&ws.pp("wpe"), cfg.max_seq_len, cfg.hidden_size)?;
```

If the GGUF has fewer position embeddings than `max_seq_len`, this will panic or load garbage. Should validate the loaded tensor shape matches expected dimensions.

---

### MAJ-8: `run.rs` uses `tokenizer.decode(&[next_token])` which may not handle byte-level BPE correctly

**File:** `crates/grim-cli/src/run.rs:404`

```rust
let token_text = tok.decode(&[next_token]);
```

GgufTokenizer's `decode` is called with a single token. If the tokenizer uses byte-pair encoding with continuation markers (like `Ġ` for space), single-token decode may produce incomplete UTF-8 or missing preceding space.

---

## 🟡 MODERATE ISSUES

### MOD-1: `Bench` command uses random toy model instead of real model

**File:** `crates/grim-cli/src/bench.rs:8-20`

```rust
let model = Llama::random(Device::Cpu, cfg);
```

Benchmarks a randomly initialized model, not a loaded checkpoint. Should use `load_model_from_gguf` or similar.

---

### MOD-2: `Doctor` check for RDNA 2 incorrectly reports as ERROR

**File:** `crates/grim-cli/src/doctor.rs:253-261`

```rust
if c.gcn.starts_with("gfx10") {
    report.errors.push(format!(
        "Host GPU architecture {} is RDNA 2. RDNA 2 does not support wave64 and is incompatible with .grim optimizations",
        c.gcn
    ));
```

This treats RDNA 2 as a hard error, but the CPU backend would still work. Should be a warning.

---

### MOD-3: `Oxidizer` `bitwidth_to_dtype` maps bitwidth 0-2 to Q2K, 3 to Q3K, etc. but no validation

**File:** `crates/grim-cli/src/oxidizer.rs:624-649`

```rust
fn bitwidth_to_dtype(bw: u32) -> GgufDType {
    match bw {
        0..=2 => GgufDType::Q2K,
        3 => GgufDType::Q3K,
        4 => GgufDType::Q4K,
        // ...
    }
}
```

Bitwidth 0 or 1 would map to Q2K, which doesn't make sense. Should validate or clamp to [2, 8].

---

### MOD-4: `train.rs` LoRA rank/alpha validation missing

**File:** `crates/grim-cli/src/train.rs:251`

```rust
let injection_reg = LoRAInjectionRegistry::standard_qlora(num_layers, opts.rank, opts.alpha, 1);
```

No validation that `rank > 0`, `alpha > 0`, `rank <= hidden_size`, etc. Invalid values will cause cryptic failures later.

---

### MOD-5: `run.rs` vocab size detection only handles Llama/Mamba/LFM2 configs

**File:** `crates/grim-cli/src/run.rs:285-293`

```rust
let vocab = if let Some(cfg) = model.config().as_any().downcast_ref::<LlamaConfig>() {
    cfg.vocab_size
} else if let Some(cfg) = model.config().as_any().downcast_ref::<MambaConfig>() {
    cfg.vocab_size
} else if let Some(cfg) = model.config().as_any().downcast_ref::<Lfm2Config>() {
    cfg.vocab_size
} else {
    512  // WRONG DEFAULT
};
```

Falls back to 512 for unknown model types (GPT2, Gemma, DeepSeek, T5, BERT, RWKV). This breaks sampling for those models.

---

### MOD-6: `run.rs` prompt tokenization doesn't add BOS for models that need it consistently

**File:** `crates/grim-cli/src/run.rs:264-282`

The BOS detection tries 3 hardcoded tokens. If none found, it proceeds without BOS. Some models (like LFM2) require `<|startoftext|>` but it may not be in the tokenizer vocab under that exact string.

---

## 🔵 MINOR / NITPICKS

### MIN-1: `main.rs` has duplicate `Commands::Server` and `Commands::Serve` — both do the same thing

Lines 53-65 (`Serve`) and 147-158 (`Server`) both start HTTP server. One should be an alias.

---

### MIN-2: `Oxidizer` `open_provider` returns `GrimMetadata` but safetensors path returns default

**File:** `crates/grim-cli/src/oxidizer.rs:28-42`

```rust
if lower.ends_with(".safetensors") || lower.ends_with(".bin") {
    // ...
    Ok((Box::new(provider), names, sizes, GrimMetadata::default()))  // no metadata
}
```

Safetensors input loses all GrimMetadata (fusion ops, ROCm profile, etc.)

---

### MIN-3: `Compat` command generates plugin but no validation of the generated manifest

**File:** `crates/grim-cli/src/compat.rs` (re-exported from grim-core)

The generated `.grimplugin` manifest should be validated against the schema before writing.

---

### MIN-4: `train.rs` uses `Vec<u32>` for labels but -100 masking casts to u32

**File:** `crates/grim-cli/src/train.rs:185-188`

```rust
let labels = vec![-100i32 as u32; prompt_len.min(max_seq_len)]
```

`-100i32 as u32` = 4294967196 (wraps). This is technically the correct IGNORE_INDEX value for u32, but it's a magic number that should be a named constant.

---

### MIN-5: `service.rs` Windows SCM `binPath` quoting is fragile

**File:** `crates/grim-cli/src/service.rs:547-551`

```rust
let bin_path = format!(
    "\"{}\" service run --config \"{}\"",
    cfg.exec_path.display(),
    cfg.config_path.display()
);
```

If paths contain spaces or special characters, the quoting may break. Should use proper Windows command-line escaping.

---

### MIN-6: Multiple model implementations duplicate `Linear::from_tensor(emb.weight.clone(), None)` for tied output

- `Llama::load` (model.rs:63-66) ✓ correct
- `Gemma::forward` (gemma.rs:160-177) ✗ manual matmul with wrong transpose
- `Lfm2::load` (lfm2.rs - incomplete) likely has similar issue

Should use a shared helper.

---

## 📋 TESTING GAPS

| Gap | Affected Code |
|-----|---------------|
| No integration test for `grim run` full generation | `run.rs` |
| No test for `grim run --serve` HTTP server | `main.rs:584-624` |
| No test for LFM2 loading (incomplete) | `lfm2.rs` |
| No test for GPT2 attention (broken) | `gpt2.rs` |
| No test for Gemma attention (fake) | `gemma.rs` |
| No test for DeepSeek MLA (missing) | `deepseek.rs` |
| No test for service install/start/stop cycle | `service.rs` |
| No test for Oxidizer full pipeline | `oxidizer.rs` |
| No test for LoRA adapter application on CPU | `lora.rs` |

---

## VERDICT

**11 critical bugs** (CRIT-1 through CRIT-9 + CRIT-3/4/5), **8 major issues**, **6 moderate issues**, **6 minor issues**.

The CLI has **one fundamental correctness bug** (CRIT-1: positions tensor = input_ids) that breaks all positional encoding for every model.

The model implementations have **4 structurally incomplete/broken attention implementations** (GPT2, Gemma, DeepSeek, LFM2). Only Llama (with its own broken CPU attention from the grim-nn audit) and T5/BERT are structurally complete.

The Oxidizer and training pipelines are architecturally sound but have integration gaps with the model loading paths.

**Priority order for fixes:**
1. CRIT-1: Fix positions tensor in `run.rs`
2. CRIT-3/4/5: Implement real attention for GPT2, Gemma, DeepSeek
3. CRIT-2: Complete LFM2 implementation
4. CRIT-6/7/8/9: Device handling, LayerNorm bias, tied weights
5. MAJ-1/2/6/7/8: Code quality and integration fixes