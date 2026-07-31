# Grim — Rust Inference Engine

## What this project is

A pure-Rust inference engine that runs autoregressive language models, SSM-based architectures, diffusion models, audio encoders, and vision encoders on CPU or GPU backends (ROCm primary, with CUDA, Vulkan, and Metal fallbacks). It uses GGUF-compatible checkpoint loading, a continuous-batching scheduler, speculative decoding by default, and provides an OpenAI-compatible HTTP API with Ollama-mode serving.

## Problem it solves

Grim provides a single, pure-Rust codebase for running large language models and other modalities on modern hardware without GPU vendor lock-in. It addresses the need for:

- Cross-platform GPU support (ROCm, CUDA, Vulkan, Metal) from a single codebase
- Efficient continuous-batching for multi-request throughput
- Low-latency inference through speculative decoding
- Simple deployment with GGUF checkpoint compatibility

Grim targets researchers and engineers who want to deploy LLM serving without C/C++ toolchain dependencies or vendor-specific CUDA code.

## Prerequisites

- **Rust toolchain**: edition 2024, minimum version 1.85 (`rustup update 1.85`)
- **LLVM development libraries**: `llvm-dev` (for ROCm JIT kernel compilation)
- **ROCm runtime libraries**: for GPU backend support (libcudnn-ops.infer.so, libhipblas.so, etc.)
- **CUDA toolkit**: optional, for NVIDIA GPU compilation (version 11.8+)
- **macOS**: Metal framework active for Metal backend support
- **Windows SDK**: for Windows service support

## Quick start (five commands)

```bash
git clone https://github.com/Nelsk/Grim.git
cd Grim
cargo build --release                # builds all 31 crates in one invocation
cargo test                           # runs the full test suite across workspace members
# Optional: ROCm GPU tests (set GRIM_RUN_GPU_TESTS=1)
GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm --features rocm-aiter
```

## Workspace map — all 31 crates

| Crate | Purpose (one sentence) |
|---|---|
| `grim-tensor` | Core tensor, DType, Shape, Device abstractions and backend-agnostic trait surface |
| `grim-tensor-graph` | Checkpoint-derived IR for tensor fusion-pattern detection |
| `grim-backend-cpu` | CPU reference backend using row-major Vec<f32>, OxiBLAS SIMD GEMM or scalar fallback |
| `grim-backend-rocm` | ROCm/HIP primary GPU target (rocBLAS, hip graph capture, fused kernels) |
| `grim-backend-cuda` | CUDA compat backend (cuBLAS GEMM only; other ops return Unimplemented) |
| `grim-backend-vulkan` | Vulkan platform-agnostic compute fallback with simulated JIT/autotuning |
| `grim-backend-metal` | Metal on Apple Silicon; CPU-fallback for all binary ops |
| `grim-nn` | Neural-network modules and WeightSource (VarBuilder-equivalent) — embedding, linear, rmsnorm, rope |
| `grim-core` | Model trait family + Session + KV cache + sampler + error types for orchestration |
| `grim-engine` | Runtime orchestrator — wires scheduler + memory + model registry into one `Engine` struct |
| `grim-scheduler` | Continuous-batching scheduler with latency-aware admission control three-queue design |
| `grim-memory` | Paged KV cache, block allocator, prefix cache, and SSM state pool |
| `grim-format` | GGUF reader/writer and safetensors bridge, GPTQ dequantization, Grim metadata layer |
| `grim-quant` | Block quantizers (Q8_0, Q4_K, Q5_K, Q6_K, FP4, NF4, FP8), Fisher/GGN diagonal calibration |
| `grim-speculative` | Speculative decoding — DSpark drafter + Markov head + confidence head + Zero-config MTP path |
| `grim-kvquant` | Runtime KV cache compression (TurboQuant rotation + Lloyd-Max scalar quant) |
| `grim-kvtransport` | Tiered KV Cache local transport (GPU -> Host RAM -> NVMe spill) |
| `grim-plugin` | WASM sandbox plugin loading and dylib dynamic library loading |
| `grim-server` | HTTP/HTTPS serving layer (axum) with OpenAI-compatible endpoints plus SSE native streaming |
| `grim-cli` | Subcommand CLI: serve, run, bench, quantize, plugin management |
| `grim-autograd` | Scoped autograd for adapter-only backward pass (LoRA / QLoRA) |
| `grim-disagg` | Distributed serving — Prefill/Decode decoupling layer |
| `grim-garage` | Training dashboard with native CVKG UI for local training/job management |
| `grim-models-transformer` | Llama/Mistral dense CausalLm implementation |
| `grim-models-mamba` | Mamba/SSM stateful-sequence architecture |
| `grim-models-vision` | ViT/CLIP-style vision encoder |
| `grim-models-audio` | Whisper-style audio encoder-decoder |
| `grim-models-diffusion` | UNet + DDIM/Euler diffusion model for image generation |

## Non-crate top-level directories

| Directory/File | Contents |
|---|---|
| `old/doc.md` | Legacy documentation specification prompt; contains the master checklist and rules for documentation |
| `Cargo.toml` (workspace root) | Workspace definition with 31 members, edition 2024, Rust 1.85 minimum, MIT OR Apache-2.0 license |
| `.github/workflows/ci.yml` | CI configuration for build, test, clippy, and cargo-mutants |
| `docs/` | Documentation files (onboarding, architecture, how-to guides, etc.) |
| `crates/` | Crate source code, each with its own Cargo.toml |

## Links — other documentation

- [Onboarding guide](docs/onboarding.md) — step-by-step development setup
- [Architecture overview](docs/architecture.md) — workspace dependency graph and design
- [How-to guides](docs/howto/) directory
- [Troubleshooting](docs/troubleshooting.md) — common issues and solutions
- [Integration reference](docs/integrations.md) — external systems and protocols
- [Glossary](docs/glossary.md) — domain-specific terms
- [Configuration reference](docs/configuration.md) — env vars and config files
- [CLI reference](docs/cli.md) — command-line interface
- [Data model](docs/data-model.md) — schemas and serialization formats
- [Per-crate READMEs](crates/) — crate-specific documentation