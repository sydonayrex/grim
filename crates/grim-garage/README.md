# grim-garage

Grim's Garage — local-first training dashboard & web app (Rust).

## Purpose

Provides a training dashboard with:
- Web-based UI for monitoring training jobs
- Native Rust CVKG UI for local job management
- Training execution and monitoring

Used for fine-tuning LoRA adapters and tracking training progress.

## Boundaries

- Is both a binary (`grim-garage`) and a library
- Does not perform inference — only training/fine-tuning
- Does not manage model serving — use `grim-server` for that

## Dependency Graph

```mermaid
graph LR
    A[grim-garage] -->|DType, Device| B[grim-tensor]
    A -->|Tensor, modules| C[grim-nn]
    A -->|Engine| D[grim-engine]
    A -->|Server| E[grim-server]
    A -->|Format| F[grim-format]
    A -->|Tensor Graph| G[grim-tensor-graph]
    A -->|Autograd| H[grim-autograd]
    A -->|CPU backend| I[grim-backend-cpu]
    A -->|ROCm| J[grim-backend-rocm]
    A -->|Plugins| K[grim-plugin]
    
    style A fill:#fce4ec
```

## Public API

### GarageApp

```rust
pub struct GarageApp {
    engine: Engine,
    jobs: JobRegistry,
    ui: CvatDashboard,
}

pub struct JobConfig {
    pub base_model: String,
    pub dataset_path: PathBuf,
    pub epochs: usize,
    pub learning_rate: f32,
    pub rank: usize,
    pub alpha: f32,
    pub mode: String,
}
```

## Usage Example

```bash
# Start the training dashboard
grim-garage

# Or run training directly
grim train -m llama3 -d dataset.jsonl -o adapter.grim.train
```

## Feature Flags

| Flag | Default | Description |
|---|---|---|
| gpu-selection | - | Enable all GPU backends for training |

## Edge Cases

1. **GPU selection**: `gpu-selection` feature enables CUDA/Vulkan/Metal backends
2. **Training format**: Outputs QLoRA adapters as `.grim.train` sidecar files
3. **UI binding**: CVKG UI bindings may need platform-specific configuration