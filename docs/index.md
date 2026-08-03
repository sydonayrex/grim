# Grim Documentation

Grim is a pure-Rust inference engine optimized for AMD GPUs (ROCm) with CUDA, Vulkan, Metal, and CPU backends.

## Getting Started

- **[Onboarding](onboarding.md)** — 5-minute quick start: clone, build, test, serve
- **[Installation Guide](howto/install-grim.md)** — Install from source

## User Guides

- **[How-to Guides](howto/)** — Practical tasks
  - [Download Models](howto/download-model.md)
  - [Run Inference](howto/run-inference.md)
  - [Convert Models](howto/convert-model.md)
  - [Train Adapters](howto/train-adapter.md)

## Reference

- **[CLI Reference](cli.md)** — All command-line options
- **[Configuration](configuration.md)** — Environment variables and config files
- **[Architecture](architecture.md)** — Workspace design, dependency graph, performance strategy
- **[Data Model](data-model.md)** — Core data structures and formats
- **[Integrations](integrations.md)** — HuggingFace, Ollama, OpenAI API compatibility
- **[Observability](observability.md)** — Metrics, logging, and telemetry
- **[Troubleshooting](troubleshooting.md)** — Common errors and fixes
- **[Glossary](glossary.md)** — Domain terms
- **[Release & Deployment](release.md)** — Build, versioning, and CI

## Crate Documentation

Each crate has its own README with detailed documentation:

- `grim-tensor` — Device, DType, Shape, tensors
- `grim-tensor-graph` — Fusion patterns
- `grim-quant` — Weight quantization (Q4_K, Q8_0, NF4, FP8, IQ)
- `grim-format` — GGUF/.grim/Safetensors I/O
- `grim-backend-cpu` — CPU backend (SIMD, OxiBLAS)
- `grim-backend-rocm` — ROCm/HIP backend (primary GPU target)
- `grim-backend-cuda` — CUDA backend (cuBLAS)
- `grim-backend-vulkan` — Vulkan backend
- `grim-backend-metal` — Metal backend (Apple)
- `grim-nn` — Neural network modules
- `grim-core` — Model traits, Session, KV cache, Sampler
- `grim-models-transformer` — Transformer model architectures
- `grim-models-mamba` — Mamba/SSM model architectures
- `grim-models-vision` — Vision encoder models
- `grim-models-audio` — Audio encoder models
- `grim-models-diffusion` — Diffusion model architectures
- `grim-memory` — Paged KV cache pool, prefix sharing, spilling
- `grim-kvquant` — Runtime KV cache compression
- `grim-kvtransport` — Tiered KV transport (GPU → RAM → NVMe)
- `grim-scheduler` — Continuous-batching scheduler
- `grim-speculative` — Speculative decoding (DSpark, Native MTP)
- `grim-autograd` — LoRA/QLoRA backward pass
- `grim-engine` — Runtime orchestrator
- `grim-server` — HTTP serving layer
- `grim-cli` — Command-line interface
- `grim-disagg` — Disaggregation layer
- `grim-plugin` — Plugin system
- `grim-garage` — Training dashboard web app

See root README for the full crate list and workspace map.
