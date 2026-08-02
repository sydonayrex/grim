# Grim — Rust Inference & Training Engine

## What this project is

A pure-Rust inference and fine-tuning engine that runs autoregressive language models, SSM-based architectures, vision encoders, audio encoders, and diffusion models on CPU or GPU backends (ROCm primary, with CUDA, Vulkan, and Metal fallbacks). It uses GGUF-compatible checkpoint loading, continuous batching, speculative decoding by default, parameter-efficient fine-tuning (LoRA, QLoRA, Vera, SoulEater, QGaLore, PISSA, OLORA), an OpenAI-compatible HTTP API server, and a local-first web training dashboard (`grim-garage`).

## Problem it solves

Grim provides a single, pure-Rust codebase for executing and training large language models and multi-modal neural architectures without C/C++ toolchain dependencies or vendor-specific CUDA lock-in. It addresses the need for:

- **Cross-platform GPU support** (ROCm/HIP primary, CUDA, Vulkan SPIR-V, Metal) from a single unified codebase
- **Efficient continuous batching** for high multi-request throughput
- **Low-latency inference** through speculative decoding (DSpark, Markov heads, zero-config MTP)
- **Local-first adapter fine-tuning** and real-time web dashboard management (`grim-garage`)
- **Seamless deployment** with GGUF checkpoint compatibility, safetensors bridging, and OpenAI/Ollama-compatible REST APIs

Grim targets researchers, machine learning engineers, and system developers who want high-performance LLM serving and fine-tuning without heavy external framework dependencies.

## Prerequisites

- **Rust toolchain**: edition 2024, minimum version 1.85 (`rustup update 1.85`)
- **LLVM development libraries**: `llvm-dev` (for ROCm JIT kernel compilation)
- **ROCm runtime libraries**: for AMD GPU backend support (`libhipblas.so`, `librocblas.so`, etc.)
- **CUDA toolkit**: optional, for NVIDIA GPU compilation (version 11.8+)
- **macOS Frameworks**: Metal framework active for Apple Silicon GPU acceleration
- **Vulkan SDK**: optional, for cross-platform Vulkan SPIR-V compute acceleration

## Quick start (five commands)

```bash
git clone https://github.com/Nelsk/Grim.git
cd Grim
cargo build --release                # builds all 28 workspace member crates in one invocation
cargo test                           # runs the full workspace test suite
# Optional: ROCm GPU tests (set GRIM_RUN_GPU_TESTS=1)
GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm --features rocm-aiter
```

## Workspace map — all 28 crates

| Crate | Purpose (one sentence) |
|---|---|
| `grim-tensor` | Core tensor, DType, Shape, Device abstractions, and `BackendStorage`/`BackendDevice` trait surfaces |
| `grim-tensor-graph` | Checkpoint-derived IR for operator fusion-pattern detection and subgraph optimization |
| `grim-backend-cpu` | CPU reference compute backend using SIMD OxiBLAS GEMM, parallel loops, and scalar fallbacks |
| `grim-backend-rocm` | AMD ROCm/HIP primary GPU target (hipRTC JIT, rocBLAS GEMM, HIP graph capture, matrix-core WMMA/MFMA) |
| `grim-backend-cuda` | CUDA compatibility backend (cuBLAS GEMM and CUDA device memory allocation) |
| `grim-backend-vulkan` | Platform-agnostic Vulkan compute backend with compiled SPIR-V GLSL shaders |
| `grim-backend-metal` | Metal backend for Apple Silicon GPUs with MPS and Metal Shading Language compute pipelines |
| `grim-nn` | Neural-network modules (`Linear`, `Embedding`, `RmsNorm`, `RoPE`, `SwiGLU`) and `WeightSource` loading |
| `grim-core` | Base `Model` trait family, `Session`, KV cache interfaces, sampler pipelines, and core error types |
| `grim-models/transformer` | Llama/Mistral/Qwen dense transformer architecture implementation (`LlamaModel`, `LlamaBlock`, `MtpLayer`) |
| `grim-models/mamba` | Mamba / Mamba2 state-space model (SSM) stateful sequence architecture |
| `grim-models/vision` | ViT / CLIP-style vision patch encoder architecture |
| `grim-models/audio` | Whisper-style audio encoder-decoder architecture |
| `grim-models/diffusion` | UNet + DDIM/Euler sampler image generation pipeline |
| `grim-format` | GGUF reader/writer, safetensors bridge, GPTQ import/export, and Grim `.grim` metadata format |
| `grim-quant` | Quantization routines (Q8_0, Q4_K, Q5_K, Q6_K, Q2_K, Q3_K, IQ*, FP4, NF4, FP8, MXFP) and GGN calibration |
| `grim-memory` | Paged KV cache manager, block allocator, prefix-caching hash map, and SSM state memory pool |
| `grim-scheduler` | Continuous-batching scheduler featuring latency-aware admission control and a three-queue architecture |
| `grim-engine` | Core runtime orchestrator integrating scheduler, memory manager, and model registry into a unified `Engine` |
| `grim-server` | HTTP/HTTPS serving layer (Axum) with OpenAI-compatible REST endpoints, Ollama support, and SSE streaming |
| `grim-cli` | Subcommand CLI: `serve`, `run`, `bench`, `quantize`, `plugin` |
| `grim-speculative` | Speculative decoding engines (DSpark drafter, Markov n-gram head, confidence head, zero-config MTP path) |
| `grim-kvquant` | Runtime KV-cache compression routines (TurboQuant rotation + Lloyd-Max scalar quantization) |
| `grim-plugin` | WebAssembly (WASM) plugin sandbox and dynamic library (`.so`/`.dylib`/`.dll`) loader |
| `grim-kvtransport` | Multi-tiered KV cache transfer engine (GPU VRAM $\leftrightarrow$ Host System RAM $\leftrightarrow$ NVMe disk spill) |
| `grim-disagg` | Distributed serving decoupling layer for separated Prefill and Decode GPU clusters |
| `grim-autograd` | Scoped reverse-mode autograd engine for adapter training (LoRA, QLoRA, Vera, SoulEater, QGaLore, PISSA, OLORA) |
| `grim-garage` | Local-first training dashboard web application (Axum backend, REST/SSE APIs, embedded web UI, and ROCm telemetry) |

## Non-crate top-level directories

| Directory/File | Contents |
|---|---|
| `old/doc.md` | Legacy documentation specification prompt; contains historical checklist and rules |
| `Cargo.toml` (workspace root) | Workspace definition with 28 members, edition 2024, Rust 1.85 minimum, MIT OR Apache-2.0 license |
| `.github/workflows/ci.yml` | CI configuration for build, test, clippy, and cargo-mutants |
| `docs/` | Documentation files (onboarding, architecture, how-to guides, etc.) |
| `crates/` | Crate source code, each with its own Cargo.toml |

## Links — other documentation

- [Onboarding guide](docs/onboarding.md) — step-by-step development setup
- [Architecture overview](docs/architecture.md) — workspace dependency graph and design
- [Troubleshooting](docs/troubleshooting.md) — common issues and solutions
- [Integration reference](docs/integrations.md) — external systems and protocols
- [Glossary](docs/glossary.md) — domain-specific terms
- [Configuration reference](docs/configuration.md) — env vars and config files
- [CLI reference](docs/cli.md) — command-line interface
- [Data model](docs/data-model.md) — schemas and serialization formats
- [Per-crate READMEs](crates/) — crate-specific documentation