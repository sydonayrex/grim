# Data Model

This document describes the data structures and serialization formats used by Grim.

## Model Types

### Trait Hierarchy

Grim uses a trait-based approach for different model architectures:

```
Model (base trait)
├── CausalLm (autoregressive language models)
├── Encoder (vision/audio encoders)
├── EncoderDecoderLm (Whisper-style)
├── StatefulSequence (Mamba/SSM)
└── DiffusionModel (UNet/DiT)
```

### Model Traits

```rust
pub trait Model: Send + Sync {
    fn config(&self) -> &dyn ModelConfig;
    fn device(&self) -> &Device;
    fn param_arith(&self) -> ArithType;
    fn as_any(&self) -> &dyn std::any::Any;
}
```

### Modality Hints

```rust
pub enum ModalityHint {
    TextInTextOut,      // LLaMA, Mistral, etc.
    VisionEncoder,      // ViT, CLIP
    AudioEncoderDecoder,// Whisper
    Diffusion,            // UNet, DiT
}
```

## Adapter Handles (LoRA/QLoRA)

```rust
pub struct AdapterHandle {
    pub id: u32,
    pub a: Tensor,      // Down projection (rank x input_dim)
    pub b: Tensor,      // Up projection (output_dim x rank)
    pub alpha: f32,     // Scaling factor (alpha / rank)
}
```

Adapters are fused into the forward pass via low-rank updates.

## GGUF Format Support

Grim reads GGUF v1+ format. Key tensor types:

| Format | Extension | Description |
|---|---|---|
| GGUF | `.gguf` | Llama.cpp compatible (primary format) |
| GRIM | `.grim` | ROCm-optimized with metadata |
| Safetensors | `.safetensors` | PyTorch-compatible |
| PyTorch | `.bin` | Legacy PyTorch checkpoint |

### GGUF Metadata

Key metadata fields extracted from GGUF:

| Field | Type | Description |
|---|---|---|
| `general.name` | string | Model name |
| `general.architecture` | string | Architecture identifier |
| `general.version` | int | Model version |
| `tokenizer.model` | string | Tokenizer type |
| `tokenizer.trainable` | bool | Whether tokenizer is trainable |

## .grim Format

Grim's native format includes additional metadata:

### Header Structure

```rust
pub struct GrimHeader {
    magic: [u8; 4],        // "GRIM"
    version: u32,          // Format version
    tensor_count: u64,       // Number of tensors
}
```

### Tensor Entries

```rust
pub struct GrimTensorEntry {
    name_offset: u32,      // Offset in name table
    dims: Vec<u64>,        // Tensor dimensions
    data_offset: u64,      // Byte offset to data
    data_len: u64,         // Length in bytes
    quant_hint: u32,       // Quantization scheme hint
}
```

### Grim-Specific Metadata

- ROCm kernel fusion hints
- GCN architecture target
- Quantization calibration data
- Training metadata (for QLoRA)

## KV Cache Structure

### KvBlock (Paged)

```rust
pub struct KvBlock {
    // Allocated on device or spilled
    k: Vec<f32>,  // or compressed representation
    v: Vec<f32>,
}
```

### Block Allocator

```rust
pub struct KvBlockPool {
    capacity: usize,        // Total blocks
    num_kv_heads: usize,
    head_dim: usize,
    blocks: Vec<BlockState>,
}
```

## Serialization Formats

### Tensor Dtypes

```rust
pub enum DType {
    // Native formats
    F32, BF16, F16, F64,
    // Quantization
    Q80, Q4K, Q5K, Q6K,
    // Low-bit formats
    FP4, NF4, FP8,
    // Block formats
    Q2K, Q3K,
    // Group-quant
    GroupInt(GpuIntConfig),
}

pub enum Storage {
    Native,                    // No dequant needed
    KQuant(KQuantScheme),      // k-quant block format
    GroupInt(GpuIntConfig),    // GPTQ/efficientQAT
    FloatPack(FloatPackScheme), // FP4/NF4/FP8 packs
    Block(BlockDtype),         // Custom block formats
}
```

## Session State

### CausalLm Session

```rust
pub struct Session {
    input_ids: Vec<u32>,      // Prompt tokens
    positions: Vec<u32>,      // Position indices
    kv_cache: Option<KvCache>, // Page-locked KV
    rng: Pcg64,               // Deterministic RNG
}
```

### SSM State (Mamba)

```rust
pub struct SsmState {
    // O(model_dim) per sequence
    hidden: Vec<f32>,
    // Convolutional state
    x: Vec<f32>,
}
```

## Sampler State

### SamplingParams

```rust
pub struct SamplingParams {
    temperature: f32,    // 0 = greedy
    top_p: f32,          // Nucleus sampling
    top_k: u32,          // Top-k limit
    repeat_penalty: f32, // Frequency penalty
}
```

## Data Flow Summary

1. **Model Loading**: GGUF/Safetensors → `TgufProvider` → Tensors in `grim-tensor`
2. **Forward Pass**: Input tokens → `Session` → `Model::forward` → `SpeculativeCausalLm` → `Logits`
3. **Sampling**: `Logits` + `SamplingParams` → `Sampler` → Token ID
4. **KV Management**: KV blocks allocated via `KvBlockPool`, managed by `Scheduler`