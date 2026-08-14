# Data Model Reference

## Checkpoint Formats

### GGUF Format
GGUF is supported via the `grim_format` crate. It stores hyperparameter metadata and tensor structures in an aggregated format.

```rust
pub struct GgufMeta {
    pub key_values: HashMap<String, GgufValue>,
    pub tensor_infos: Vec<TensorInfo>,
}
```

### Safetensors Format
Safetensors is supported via `grim_format::tprov::SafetensorsProvider`. It requires a sibling `config.json` for hyperparameters.

```rust
pub fn load_model_from_safetensors(path: &str, device: Device) -> Result<Box<dyn CausalLm>> {
// ...
```

### GRIM Format
`.grim` is the native serialized checkpoint format optimized for rapid loading.

## Memory Structures

The `grim_core` crate manages memory allocations through the `Device` and `Tensor` primitives. Memory layouts follow PyTorch conventions, supporting FP32, FP16, and specific quantized types (e.g., Q4_K).

```rust
pub struct Tensor {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub device: Device,
    // Internal buffer reference
}
```

## Constraints
* Tensors require explicit shapes and memory layouts before execution.
* `.safetensors` files mandate a collocated `config.json` and `tokenizer.json` for successful initialization.
