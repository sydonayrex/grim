# grim-models-transformer

## Purpose

`grim-models-transformer` provides autoregressive causal language model implementations for dense and mixture-of-experts transformer architectures, including LLaMA, Mistral, Gemma, DeepSeek, and Qwen (including Qwen3.8-Flash-Next with Gated DeltaNet linear attention, 4-branch gated residual streams, PLE N-gram embedding projections, and 512 routed experts).

## Boundaries

`grim-models-transformer` does **not**:
- Allocate physical KV cache pages or maintain prefix hash trees (delegated to `grim-memory`).
- Parse raw file byte containers directly (delegated to `grim-format`).
- Perform token sampling or continuous batch scheduling (delegated to `grim-core` and `grim-scheduler`).

## Dependency Graph

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#2b2d42', 'edgeLabelBackground':'#ffffff', 'tertiaryColor': '#edf2f4'}}}%%
flowchart TD
    subgraph Sibling Dependents
        grim_engine["grim-engine"]
        grim_server["grim-server"]
        grim_cli["grim-cli"]
    end

    subgraph Focal Node
        grim_models_transformer["grim-models/transformer"]
    end

    subgraph Workspace Dependencies
        grim_tensor["grim-tensor"]
        grim_nn["grim-nn"]
        grim_core["grim-core"]
        grim_backend_cpu["grim-backend-cpu"]
        grim_models_vision["grim-models/vision"]
        grim_quant["grim-quant"]
    end

    subgraph External Dependencies
        thiserror["thiserror"]
        serde["serde"]
        serde_json["serde_json"]
    end

    grim_engine --> grim_models_transformer
    grim_server --> grim_models_transformer
    grim_cli --> grim_models_transformer

    grim_models_transformer --> grim_tensor
    grim_models_transformer --> grim_nn
    grim_models_transformer --> grim_core
    grim_models_transformer --> grim_backend_cpu
    grim_models_transformer --> grim_models_vision
    grim_models_transformer --> grim_quant
    grim_models_transformer --> thiserror
    grim_models_transformer --> serde
    grim_models_transformer --> serde_json

    classDef focal fill:#d90429,stroke:#ef233c,stroke-width:2px,color:#ffffff;
    classDef workspace fill:#2b2d42,stroke:#8d99ae,stroke-width:1px,color:#edf2f4;
    classDef sibling fill:#4a4e69,stroke:#9a8c98,stroke-width:1px,color:#f2e9e4;
    classDef external fill:#1f2421,stroke:#495867,stroke-width:1px,color:#f0f3f4;

    class grim_models_transformer focal;
    class grim_tensor,grim_nn,grim_core,grim_backend_cpu,grim_models_vision,grim_quant workspace;
    class grim_engine,grim_server,grim_cli sibling;
    class thiserror,serde,serde_json external;
```

## Public API Overview

Exposed from `src/lib.rs`:

```rust
/// Qwen3.8-Flash-Next architecture with GDN, Gated Residual streams, and PLE embeddings.
pub struct Qwen38FlashNext {
    // ...
}

/// Configuration matching HuggingFace/vLLM qwen4_exp_text checkpoints.
pub struct Qwen38FlashNextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub ngram_vocab_size: Option<usize>,
    pub ngram_dim: Option<usize>,
    pub split_ngram_parts: usize,
    pub ngram_size: usize,
    // ...
}

impl Qwen38FlashNext {
    pub fn load(device: Device, ws: &WeightSource<'_>, cfg: Qwen38FlashNextConfig) -> Result<Self, Error>;
    pub fn load_tp(device: Device, ws: &WeightSource<'_>, cfg: Qwen38FlashNextConfig, tp: &TensorParallelConfig) -> Result<Self, Error>;
}

/// Canonical CausalLm interface implementation for autoregressive token prediction.
impl CausalLm for Qwen38FlashNext {
    fn forward(&self, session: &mut dyn SessionT, input_ids: &Tensor, positions: &Tensor, mask: &[u8]) -> Result<Tensor, Error>;
}
```

## Usage Example

```rust
use grim_models_transformer::qwen38_flash_next::{Qwen38FlashNext, Qwen38FlashNextConfig};
use grim_tensor::Device;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cfg = Qwen38FlashNextConfig::default();
    cfg.vocab_size = 152064;
    cfg.hidden_size = 2560;

    let model = Qwen38FlashNext::random(Device::Cpu, cfg);
    println!("Instantiated Qwen3.8-Flash-Next causal LM model");
    Ok(())
}
```

## Use Cases

- Running high-throughput autoregressive inference across standard dense architectures (LLaMA 3, Mistral, Gemma).
- Serving frontier architectures like Qwen3.8-Flash-Next with hybrid linear attention and auxiliary PLE position-aware N-gram embeddings.
- Loading sharded SafeTensors parameters with tensor parallelism.

## Edge Cases, Limitations, and Quirks

1. **PLE Weight Availability**: `Qwen38FlashNext::load` requires both `ngram_embedding` table shards and `key_proj` projections when `ngram_vocab_size` is specified; missing keys fail loudly with `Error::Config`.
2. **Residual Stream Dimensionality**: The 4-branch gated residual mixer operates on `4 * hidden_size` activation vectors scaled by $1 / \sqrt{4} = 0.5$.

## Build Flags, Feature Flags, and Environment Variables

- `default`: No special features.
