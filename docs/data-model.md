# Data Model

This document describes the data structures and serialization formats used by Grim.

## Trait Hierarchy

Grim uses a trait-based approach for different model architectures:

```
Model (base trait)
├── CausalLm (autoregressive language models — LLaMA, Mistral, etc.)
├── Encoder (vision/audio encoders — CLIP, ViT, Whisper)
├── EncoderDecoderLm (Whisper-style)
├── StatefulSequence (Mamba/SSM)
└── DiffusionModel (UNet/DiT)
```

## Core Traits

```rust
pub trait Model: Send + Sync {
    fn config(&self) -> &dyn ModelConfig;
    fn device(&self) -> &Device;
    fn param_arith(&self) -> ArithType;  // compute-time type, not storage
    fn as_any(&self) -> &dyn std::any::Any;
}

pub trait CausalLm: Model {
    fn new_session(&self) -> Box<dyn SessionT>;
    fn forward(
        &self,
        session: &mut dyn SessionT,
        input_ids: &Tensor,
        positions: &Tensor,
        adapters: &[AdapterHandle],
    ) -> Result<Tensor>;
}

pub trait Encoder: Model {
    fn encode(&self, input: &Tensor) -> Result<Tensor>;
}

pub trait EncoderDecoderLm: Model {
    // ... encode/decode methods
}

pub trait StatefulSequence: Model {
    fn init_state(&self, batch: usize) -> Box<dyn SsmState>;
    fn step(&self, state: &mut dyn SsmState, input: &Tensor) -> Result<Tensor>;
}
```

## Adapter Handles (LoRA/QLoRA)

```rust
pub struct AdapterHandle {
    pub id: u32,
    pub a: Tensor,    // Down projection (rank x input_dim)
    pub b: Tensor,    // Up projection (output_dim x rank)
    pub alpha: f32,   // Scaling factor (alpha / rank)
}
```

Adapters are fused into the forward pass via low-rank updates.

Related code: `grim-core/src/model.rs`.

## Modality Hints

```rust
pub enum ModalityHint {
    TextInTextOut,      // LLaMA, Mistral, etc.
    VisionEncoder,      // ViT, CLIP
    AudioEncoderDecoder,// Whisper
    Diffusion,          // UNet, DiT
}
```

## Tensor Dtype System

### Arithmetic Type

The compute-time type — what the hardware computes in. Most backends compute in F32 or F16 regardless of weight storage.

```rust
pub enum ArithType {
    F32,
    F16,
    BF16,
    I64,
    U32,
    U8,
}
```

### Storage Encoding

Physical storage encoding. When storage differs from the arithmetic type, dequantization is needed before compute.

```rust
pub enum Storage {
    Native,                    // Stored in native encoding — no dequant needed
    KQuant(KQuantScheme),      // Block-quantized K-quant format (llama.cpp-compatible)
    GroupInt(GpuIntConfig),    // Grouped INT weights (GPTQ, EfficientQAT)
    FloatPack(FloatPackScheme), // Low-bit float packs (FP4, NF4, FP8, MXFP4, MXFP8)
    Block(BlockDtype),         // Custom block formats (FP4/NF4/FP8 blocks)
    ResidualPacked(ResidualPackedConfig), // With outlier overrides (backup1/backup2)
}
```

### Quantization Schemes

```rust
pub enum KQuantScheme {
    Q2K, Q3K, Q4K, Q5K, Q6K, Q80,
    IQ4NL, IQ4XS, IQ3XXS, IQ3S, IQ2XXS, IQ2XS, IQ2S,
}

pub enum BlockDtype {
    Fp4, Nf4, Fp8, Fp4Block16, Fp8Block16,
}

pub enum FloatPackScheme {
    Fp4,     // E2M1 4-bit float
    Nf4,     // Normalized float-4 (Quanto/Unsluth-style)
    Fp8,     // E4M3 by default; E5M2 recognized
    MxFp4,   // OCP Microscaling 4-bit float with shared E8M0 scale
    MxFp8,   // OCP Microscaling 8-bit float with shared scale
}
```

## Device Enum

```rust
pub enum Device {
    Cpu,           // Always available reference
    Rocm(usize),   // ROCm/HIP primary target (hip/rocBLAS-backed)
    Vulkan,        // Platform-agnostic compute
    Cuda(usize),   // Optional CUDA target
    Metal(usize),  // Optional Metal target
}
```

## GGUF Format Support

Grim reads GGUF v1+ format. Key tensor types:

| Format | Extension | Description |
|---|---|---|
| GGUF | `.gguf` | Llama.cpp compatible (primary format) |
| GRIM | `.grim` | Native format with metadata, KV layout hints, ROCm fusion hints |
| Safetensors | `.safetensors` | PyTorch-compatible |
| PyTorch | `.bin` | Legacy PyTorch checkpoint |

### `.grim` Header

```rust
pub struct GrimHeader {
    pub magic: [u8; 5],           // FUCKING_SORCERY constant
    pub metadata_len: u64,        // Byte length of metadata JSON after header
    pub num_tensors: u32,         // Number of tensor entries
}
```

### `.grim` Tensor Entries

```rust
pub struct GrimTensorEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub base_bitwidth: u8,        // Target average bits-per-weight
    pub payload_offset: u64,      // Byte offset to compressed data
    pub payload_size: u64,        // Length of compressed data in bytes
    pub outlier_count: u32,       // Number of outlier values (residuals)
    pub outlier_offset: u64,      // Byte offset to outlier data
    // Persistent KV layout fields (WI-R4)
    pub kv_present: u8,
    pub kv_rotated: u8,
    pub kv_bits_k: u8,
    pub kv_bits_v: u8,
    pub kv_head_bits_table_offset: u64,
    pub kv_eviction_map_offset: u64,
    pub kv_eviction_map_size: u64,
    pub kv_sink_fp16: u8,
    pub kv_compressed_offset: u64,
    pub kv_compressed_size: u64,
}
```

## KV Cache Structure

### Paged KvBlock (in `grim-memory`)

```rust
pub const BLOCK_SIZE: usize = 16;

// Private — allocated within KvBlockPool
struct KvBlock {
    _id: usize,
    key_data: Vec<f32>,           // Flat [BLOCK_SIZE, num_kv_heads, head_dim]
    value_data: Vec<f32>,         // Flat [BLOCK_SIZE, num_kv_heads, head_dim]
    num_tokens: usize,            // Tokens currently filled in this block
}
```

### Block Allocator

```rust
pub struct KvBlockPool {
    blocks: Vec<KvBlock>,
    free_list: VecDeque<BlockId>,
    ref_counts: HashMap<BlockId, u32>,     // 0 = eligble for tiering
    prefix_cache: HashMap<u64, BlockId>,   // prefix hash → block ID
    ssm_states: HashMap<u32, Vec<f32>>,    // Mamba/SSM state cache
    block_major_layout: bool,              // rocm-aiter feature flag
    recently_zero: VecDeque<BlockId>,      // refcount-zero blocks retained for one cycle
    num_heads: usize,
    head_dim: usize,
    block_bytes: usize,                    // BLOCK_SIZE * num_heads * head_dim * 4
    compressor: Option<Arc<dyn KvCompressor>>,
    spill: Option<Arc<SharedSpillManager>>,
}
```

### KV Cache Trait

```rust
pub trait KvCache: Send {
    fn append_slot(&mut self) -> Result<()>;
    fn tentative_append(&mut self, n: usize) -> Result<()>;   // draft tokens
    fn commit(&mut self, accepted_len: usize) -> Result<()>;   // speculative acceptance
    fn rollback_to(&mut self, len: usize) -> Result<()>;       // rejection rollback
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn kv_mut(&mut self) -> Option<&mut (dyn KvCache + 'static)>;
}
```

### Compressed KV Block (in `grim-kvquant`)

```rust
pub struct CompressedKvBlock {
    pub key_bits: Vec<u8>,             // Packed key data (random-orthogonal-rotated + Q)
    pub key_meta: Vec<f32>,            // Per-group scale for keys
    pub value_bits: Vec<u8>,           // Packed value data (group-quantized)
    pub value_meta: Vec<f32>,           // Per-group scale + zero for values
    pub num_tokens: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub modality: KvModality,          // Text, Audio, Visual
}
```

## Sampling

### Sampler Trait

```rust
pub trait Sampler: Send + Sync {
    fn sample(&self, logits: &Tensor, history: &[u32]) -> Result<u32>;
    fn name(&self) -> &str;
}
```

### SamplingParams

```rust
pub struct SamplingParams {
    pub temperature: f32,    // 0 = greedy / deterministic
    pub top_p: f32,          // Nucleus sampling (cumulative probability)
    pub top_k: u32,          // Top-k candidate bound before top-p
    pub repeat_penalty: f32, // 1.0 = disabled (Ollama default: 1.10)
}
```

### Session State

The per-request session is defined by the `SessionT` trait (not a concrete struct):

```rust
pub trait SessionT: Send {
    fn device(&self) -> &Device;
    fn current_pos(&self) -> usize;
    fn advance_pos(&mut self, by: usize);
    fn has_kv(&self) -> bool;
    fn append_kv(&mut self, _k: &Tensor, _v: &Tensor) -> Result<()>;
    fn kv_mut(&mut self) -> Option<&mut (dyn KvCache + 'static)>;
    fn rollback_kv_to(&mut self, len: usize);
    fn get_hip_graph_handle(&self) -> Option<u64>;  // ROCm graph capture
    fn set_hip_graph_handle(&mut self, _handle: u64);
    fn eval_eager(&mut self, op: &str, inputs: &[&Tensor]) -> Result<Tensor>;
    fn get_last_hidden_state(&self) -> Option<Tensor>;
    fn set_last_hidden_state(&mut self, _hidden: Tensor);
    fn model_state(&self) -> Option<&(dyn std::any::Any + Send)>;
    fn model_state_mut(&mut self) -> Option<&mut (dyn std::any::Any + Send)>;
    fn set_model_state(&mut self, _state: Box<dyn std::any::Any + Send>);
    fn request_rng(&self) -> Option<&SimpleRng>;
    fn request_rng_mut(&mut self) -> Option<&mut SimpleRng>;
    fn live_gpu_utilization(&self) -> f32;
    fn batch_pressure(&self) -> usize;
}
```

A concrete `Inner` implementation is returned by `Session::new_storage`.

## Data Flow Summary

1. **Model Loading**: GGUF/Safetensors → `GgufProvider` (in `grim-format`) → `WeightSource` → tensors in `grim-tensor`.
2. **Weight Application**: `VarBuilder`-like interface in `grim-nn` materializes tensors.
3. **Inference**: Input tokens → `SessionT` → `CausalLm::forward` (optionally wrapped by `SpeculativeCausalLm` in `grim-speculative`) → logits.
4. **Sampling**: Logits + `SamplingParams` → `Sampler::sample` → token ID.
5. **KV Management**: KV blocks allocated via `KvBlockPool` in `grim-memory`, managed by `Scheduler` in `grim-scheduler`. Speculative draft tokens are appended tentatively and either committed or rolled back.
