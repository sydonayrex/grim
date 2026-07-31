# Grim Documentation

Welcome to the Grim documentation. This is a high-performance LLM inference system optimized for AMD GPUs (ROCm) with CUDA and CPU fallback.

## Getting Started

- **[Onboarding](onboarding.md)** — 5-minute quick start: clone, build, test, serve
- **[Installation](howto/install-grim.md)** — Install from source or binaries

## User Guides

- **[How-to Guides](howto/)** — Practical tasks
  - [Download Models](howto/download-model.md)
  - [Run Inference](howto/run-inference.md)
  - [Convert Models](howto/convert-model.md)
  - [Train Adapters](howto/train-adapter.md)

## Reference

- **[CLI Reference](cli.md)** — All command-line options
- **[Configuration](configuration.md)** — Environment variables and config files
- **[Integrations](integrations.md)** — Hugging Face, Ollama, OpenAI API compatibility
- **[Data Model](data-model.md)** — Core data structures and formats
- **[Troubleshoot](troubleshooting.md)** — Common errors and fixes
- **[Glossary](glossary.md)** — Domain terms

## Architecture

- **[Architecture](architecture.md)** — Workspace design, dependency graph, performance strategy

## Crate Documentation

Each crate has its own README with detailed documentation:

- `grim-core` — Model traits, Session, KvCache
- `grim-engine` — Engine, tick(), self-tuning
- `grim-server` — HTTP/OpenAI API server
- `grim-tensor` — Device, DType, Shape, tensors
- `grim-backend-*` — GPU backends (ROCm, CUDA, Vulkan, Metal, CPU)
- `grim-kvquant` — TurboQuant KV compression
- `grim-tensor-graph` — Fusion patterns
- See root README for full crate list