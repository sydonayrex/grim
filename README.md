# Grim — Rust Inference Engine

## What this project is

A pure-Rust inference engine that runs autoregressive language models, SSM-based architectures, diffusion models, audio encoders, and vision encoders on CPU or GPU backends (ROCm primary, with CUDA, Vulkan, and Metal fallbacks). It uses GGUF-compatible checkpoint loading, a continuous-batching scheduler, speculative decoding by default, and provides an OpenAI-compatible HTTP API with Ollama-mode serving.

## Prerequisites

- Rust toolchain: edition 2024, minimum version 1.85 (`rustup update`)
- Linux development libraries: `clang`, `llvm-dev` (for LLVM bitcode compilation in ROCm JIT), plus ROCm runtime library files for GPU backends
- Optionally installed CUDA toolkit on NVIDIA hardware to build the ROCm/CUDA/Neural Core backend
- macOS system with the Metal framework active — the Metal and Vulkan backends will be built for `target_vendor = "apple"` only
- Windows Service SCM support via Windows SDK

## Quick start (five commands)

```
git clone https://github.com/your-repo/grim.git
cd grim
cargo build --release                # builds all 30 crates in one invocation
cargo test                           # runs the full test suite across workspace members
GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm --features rocm-aiter  # optional: ROCm GPU tests after setting GRIM_RUN_GPU_TESTS=true
```

## Workspace map — all 30 crates

| Crate | Purpose (one sentence) |
|---|---|
| grim-tensor | Core tensor, DType, Shape, Device abstractions and backend-agnostic trait surface |
| grim-tensor-graph | Checkpoint-derived IR for tensor fusion-pattern detection |
| grim-backend-cpu | CPU reference backend using row-major Vec\<f32\>, OxiBLAS SIMD GEMM or scalar fallback |
| grim-backend-rocm | ROCm/HIP primary GPU target (rocBLAS, hip graph capture, fused kernels) |
| grim-backend-cuda | CUDA compat backend (cuBLAS GEMM only; other ops return Unimplemented) |
| grim-backend-vulkan | Vulkan platform-agnostic compute fallback with simulated JIT/autotuning |
| grim-backend-metal | Metal on Apple Silicon; CPU-fallback for all binary ops |
| grim-nn | Neural-network modules and WeightSource (VarBuilder-equivalent) — embedding, linear, rmsnorm, rope |
| grim-core | Model trait family + Session + KV cache + sampler + error types for orchestration |
| grim-models-transformer | Llama/Mistral dense CausalLm implementation |
| grim-models-mamba | Mamba/SSM stateful-sequence architecture |
| grim-models-vision | ViT/CLIP-style vision encoder (implements Encoder trait) |
| grim-models-audio | Whisper-style audio encoder-decoder |
| grim-models-diffusion | UNet + DDIM/Euler diffusion model for image generation |
| grim-format | GGUF reader/writer and safetensors bridge, GPTQ dequantization, Grim metadata layer |
| grim-quant | Block quantizers (Q8_0, Q4_K, Q5_K, Q6_K, FP4, NF4, FP8), Fisher/GGN diagonal calibration |
| grim-memory | Paged KV cache with prefix caching, demote-before-drop eviction, speculative slots |
| grim-scheduler | Continuous-batching scheduler with latency-aware admission control three-queue design |
| grim-engine | Runtime orchestrator — wires scheduler + memory + model registry into one `Engine` struct |
| grim-server | HTTP/HTTPS serving layer (axum) with OpenAI-compatible endpoints plus SSE native streaming |
| grim-cli | Subcommand CLI: serve, run, bench, quantize, plugin management |
| grim-speculative | Speculative decoding — DSpark drafter + Markov head + confidence head + Zero-config MTP path |
| grim-kvquant | Runtime KV cache compression (TurboQuant rotation + Lloyd-Max scalar quant) |
| grim-kvtransport | Tiered KV transport: GPU RAM → Host RAP → NVMe spill |
| grim-plugin | WASM sandbox plugin loading and dylib dynamic library loading |
| grim-disagg | Distributed serving — Prefill/Decode decoupling layer |
| grim-garage | Training dashboard with native CVKG UI for local training/job management |

## Non-crate top-level directories

| Directory/File | Contents |
|---|---|
| `old/doc.md` | Legacy documentation specification prompt that describes how to write all docs below; contains the master checklist and rules |
| Cargo.toml (workspace root) | Workspace definition with 30 members, edition 2024, Rust 1.85 minimum, MIT OR Apache-2.0 license |

## Links — other documents produced in this repo

- Onboarding guide: `/docs/onboarding.md`  
- Architecture overview: `/docs/architecture.md`  
- How-to guides: `/docs/howto/` directory  
- Troubleshooting: `/docs/troubleshooting.md`  
- Integrations reference: `/docs/integrations.md`  
- Glossary: `/docs/glossary.md`  
- Configuration reference: `/docs/configuration.md`  
- CLI reference: `/docs/cli.md`  
- Data model: `/docs/data-model.md`  

## Build and test references

- The workspace uses Thin LTO, single codegen unit in release profile (`Cargo.toml` lines 72–75).
- Every dependency is pinned either via the `workspace.dependencies` table (internal crates) or directly (external crates like `thiserror = "1"`, `parking_lot = "0.12"`).

## TODO: confirm with maintainer

- No CHANGELOG, release-please config, or versioning convention exists in this repo; confirm semver or any other convention (per `old/doc.md` item 13 condition).
