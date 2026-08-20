# Grim

Grim is a pure-Rust neural network inference and fine-tuning engine supporting language models, state-space models, vision encoders, audio encoders, and diffusion architectures across CPU, ROCm, CUDA, Vulkan, and Metal backends.

## Problem Solved and Target Audience

Grim provides execution and fine-tuning of neural network models without reliance on C/C++ runtime dependencies or vendor-locked frameworks. It targets machine learning researchers, systems software developers, and platform engineers who require configurable inference serving, continuous batching, and local adapter fine-tuning.

## Prerequisites

- **Rust toolchain**: Edition 2024, version 1.85 or higher (`rustup update 1.85`)
- **System C compiler / LLVM**: `clang` and `llvm-dev` (required for ROCm JIT kernel compilation)
- **ROCm runtime**: `libhipblas.so` and `librocblas.so` (required for AMD GPU execution)
- **CUDA toolkit**: Version 11.8 or higher (optional, for NVIDIA GPU support)
- **Vulkan SDK**: Optional, for Vulkan compute execution
- **macOS SDK**: Metal framework (optional, for Apple Silicon GPU execution)

## Quick Start

```bash
git clone https://github.com/Nelsk/Grim.git
cd Grim
cargo build --release
cargo test --workspace
./target/release/grim serve
```

## Workspace Map

| Crate | Scope |
|---|---|
| [`grim-tensor`](crates/grim-tensor/README.md) | Core tensor structures, shape handling, data types, and device storage traits. |
| [`grim-tensor-graph`](crates/grim-tensor-graph/README.md) | Computation graph representation and subgraph optimization passes. |
| [`grim-backend-cpu`](crates/grim-backend-cpu/README.md) | CPU compute backend using SIMD vector primitives and scalar execution fallback. |
| [`grim-backend-rocm`](crates/grim-backend-rocm/README.md) | AMD ROCm/HIP primary GPU backend with JIT kernel compilation and rocBLAS integration. |
| [`grim-backend-cuda`](crates/grim-backend-cuda/README.md) | NVIDIA CUDA compatibility backend using cuBLAS and CUDA runtime APIs. |
| [`grim-backend-vulkan`](crates/grim-backend-vulkan/README.md) | Vulkan compute backend executing SPIR-V compute pipelines. |
| [`grim-backend-metal`](crates/grim-backend-metal/README.md) | Apple Metal compute backend using Metal Performance Shaders and MSL shaders. |
| [`grim-nn`](crates/grim-nn/README.md) | Neural network layer building blocks and weight loader traits. |
| [`grim-core`](crates/grim-core/README.md) | Core model traits, session execution, error definitions, and sampler interfaces. |
| [`grim-models/transformer`](crates/grim-models/transformer/README.md) | Dense transformer model family implementations (LLaMA, Mistral, Qwen). |
| [`grim-models/mamba`](crates/grim-models/mamba/README.md) | Mamba and Mamba2 state-space model architecture implementations. |
| [`grim-models/vision`](crates/grim-models/vision/README.md) | Vision transformer and CLIP patch encoder architecture implementations. |
| [`grim-models/audio`](crates/grim-models/audio/README.md) | Whisper audio encoder and decoder model architecture implementations. |
| [`grim-models/diffusion`](crates/grim-models/diffusion/README.md) | UNet and DDIM/Euler diffusion sampling pipeline implementations. |
| [`grim-format`](crates/grim-format/README.md) | GGUF reader and writer, safetensors parser, and native `.grim` metadata I/O. |
| [`grim-quant`](crates/grim-quant/README.md) | Weight quantization schemes (Q8_0, Q4_K, Q5_K, NF4, FP8, MXFP) and calibration routines. |
| [`grim-memory`](crates/grim-memory/README.md) | Paged KV cache allocator, prefix caching index, and SSM state pools. |
| [`grim-scheduler`](crates/grim-scheduler/README.md) | Continuous batching scheduler with admission control and priority queues. |
| [`grim-engine`](crates/grim-engine/README.md) | Top-level execution orchestrator linking model, memory, scheduler, and backends. |
| [`grim-server`](crates/grim-server/README.md) | Axum HTTP server delivering OpenAI-compatible and Ollama-compatible REST APIs. |
| [`grim-cli`](crates/grim-cli/README.md) | Command-line interface entry points for serving, inference, testing, and administration. |
| [`grim-speculative`](crates/grim-speculative/README.md) | Speculative decoding execution algorithms including draft models and n-gram heads. |
| [`grim-kvquant`](crates/grim-kvquant/README.md) | KV cache compression and quantization algorithms. |
| [`grim-plugin`](crates/grim-plugin/README.md) | WASM plugin runtime sandbox and dynamic library loader. |
| [`grim-kvtransport`](crates/grim-kvtransport/README.md) | Tiered KV cache transfer between GPU VRAM, system RAM, and NVMe storage. |
| [`grim-disagg`](crates/grim-disagg/README.md) | Disaggregated serving split for separate prefill and decode execution clusters. |
| [`grim-autograd`](crates/grim-autograd/README.md) | Reverse-mode automatic differentiation engine for parameter-efficient fine-tuning. |
| [`grim-garage`](crates/grim-garage/README.md) | Embedded web dashboard backend for monitoring training and engine status. |
| [`grim-constrain`](crates/grim-constrain/README.md) | Grammar-based and schema-constrained token masking and sampling for JSON and structured generation. |

## Top-Level Directories

| Directory / File | Description |
|---|---|
| `.agents/` | Agent configuration and local customization skills for project tooling. |
| `.github/` | Continuous integration workflow definitions for build, test, and lint checking. |
| `.opencode/`, `.poolside/`, `.rocm/`, `.zcode/`, `.zl/` | Hardware probe files and environment settings for ROCm and tool configurations. |
| `crates/` | Workspace crate source implementations. |
| `dist/` | Distribution installation scripts and systemd service configuration files. |
| `docs/` | Comprehensive technical documentation and user guides. |
| `models/` | Cache directory for downloaded model weights and GGUF checkpoints. |
| `mutants.toml` | Mutation testing configuration file for `cargo-mutants`. |
| `old/` | Historical specification files and legacy reference prompts. |
| `plugins/` | Native and WebAssembly plugin artifacts. |
| `third-party/` | Vendor-patched third-party dependency source crates. |

## Documentation Links

- [Onboarding Guide](docs/onboarding.md)
- [Architecture Overview](docs/architecture.md)
- [CLI Reference](docs/cli.md)
- [Configuration Reference](docs/configuration.md)
- [Data Model Reference](docs/data-model.md)
- [External Integrations](docs/integrations.md)
- [Observability Reference](docs/observability.md)
- [Troubleshooting Guide](docs/troubleshooting.md)
- [Glossary](docs/glossary.md)
- [Release and Deployment](docs/release.md)
- [How-To: Install Grim](docs/howto/install-grim.md)
- [How-To: Run Inference](docs/howto/run-inference.md)
- [How-To: Download Model](docs/howto/download-model.md)
- [How-To: Convert Model](docs/howto/convert-model.md)
- [How-To: Train Adapter](docs/howto/train-adapter.md)