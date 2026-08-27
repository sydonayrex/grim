# Data Model Reference

This document describes the persistent binary file schemas, in-memory tensor representations, and wire-format metadata structures across Grim.

---

## 1. Native Binary Format (`.grim`)

The `.grim` container format is a single-file binary container designed for direct GPU DMA mapping and rapid metadata inspection.

### File Layout

```
+-----------------------------------------------+
| Magic bytes: "GRIM" (4 bytes)                |
| Version: u32 (little-endian)                  |
| Header size: u32                              |
| JSON Metadata string (UTF-8, variable length) |
+-----------------------------------------------+
| Tensor Directory (Array of GrimTensorEntry)   |
|   - Name: String                              |
|   - DType: u32 discriminant                   |
|   - Shape: Dimensions array                   |
|   - Offset: u64 (64-byte aligned)             |
|   - Size: u64 bytes                           |
+-----------------------------------------------+
| Raw Tensor Payloads (64-byte aligned)         |
+-----------------------------------------------+
```

### JSON Metadata Schema (`GrimMetadata`)

```rust
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GrimMetadata {
    pub architecture: String,
    pub hidden_dim: u32,
    pub intermediate_dim: u32,
    pub num_layers: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub vocab_size: u32,
    pub context_length: u32,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub preferred_dtype: Option<String>,
    pub gemm_backend: Option<String>,
    pub fp8: Option<bool>,
    pub multi_gpu_strategy: Option<String>,
}
```

---

## 2. Model Footprint Schema (`ModelFootprint`)

Header-only representation of model geometry and resource requirements extracted without materializing tensor weights:

```rust
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelFootprint {
    pub architecture: Option<String>,
    pub parameter_count: u64,
    pub weight_bytes: u64,
    pub kv_cache_bytes_per_seq_token: u64,
    pub context_length: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub vocab_size: usize,
}
```

---

## 3. Training State Sidecar (`.grim.train`)

JSON sidecar metadata file saved alongside checkpoints during fine-tuning:

```rust
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainState {
    pub step: usize,
    pub epoch: usize,
    pub loss: f32,
    pub lr: f32,
    pub fp_format: TrainFpFormat,
    pub param_dtypes: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TrainFpFormat {
    Fp32,
    Bf16,
    Fp16Param,
}
```

---

## 4. In-Memory Tensor Data Model (`grim-tensor`)

```rust
pub struct Tensor {
    shape: Shape,
    dtype: DType,
    device: Device,
    storage: Box<dyn BackendStorage>,
}

pub struct DType {
    pub arith: ArithType,
    pub storage: Storage,
}

pub enum Storage {
    Native,
    KQuant(KQuantScheme),
    Block(BlockDtype),
    FloatPack(FloatPackScheme),
    W4A16(W4A16StorageConfig),
    GroupInt(GroupIntStorageConfig),
    WNA16,
    CompressedTensorsW8A8Int8,
    CompressedTensorsW8A8Fp8,
    Awq(AwqStorageConfig),
    ResidualPacked(ResidualPackedConfig),
}
```

---

## 5. Paged Memory Blocks (`grim-memory`)

KV Cache allocations are divided into fixed-size token blocks:

```rust
pub struct BlockTable {
    pub block_size: usize, // typically 16 tokens
    pub physical_block_indices: Vec<usize>,
    pub prefix_hash: u64,
}
```
