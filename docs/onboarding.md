# Onboarding Guide

This guide walks new contributors through setting up a development environment for Grim.

## Prerequisites

- **Rust toolchain**: edition 2024, minimum version 1.85
  ```bash
  rustup update 1.85
  rustup default stable
  ```
- **Git** (for cloning the repository)
- **C++ compiler** (for ROCm JIT kernel compilation, requires `clang`)
- **LLVM development libraries** (for `llvm-dev` / `llvm-devel` package)

### Platform-specific requirements

**Linux (ROCm)**:
- ROCm runtime libraries (optional, for GPU backend)
- `llvm-dev` package for JIT compilation

**macOS**:
- Xcode Command Line Tools
- Metal framework enabled
- `brew install llvm` for JIT compilation

**Windows**:
- Windows SDK (for service support)

## Step 1: Clone the repository

```bash
git clone https://github.com/Nelsk/Grim.git
cd Grim
```

## Step 2: Build the project

Build all 28 crates in release mode:

```bash
cargo build --release
```

For faster iteration during development, use debug builds:

```bash
cargo build
```

## Step 3: Run the test suite

Run the full test suite across all workspace members:

```bash
cargo test --workspace
```

### Running specific crate tests

To run tests for a single crate:

```bash
cargo test -p grim-tensor
cargo test -p grim-core
cargo test -p grim-engine
```

### Running GPU tests (ROCm)

GPU tests require the `GRIM_RUN_GPU_TESTS` environment variable:

```bash
GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm --features rocm-aiter
```

### Running a single test by name

```bash
cargo test -p grim-core test_models_dir_env_override
```

## Step 4: Make a change and verify

Edit a source file, then rebuild and test:

```bash
# Edit src/lib.rs in the desired crate
cargo build -p <crate-name>
cargo test -p <crate-name>
```

## Source Layout

```
grim/
├── Cargo.toml              # Workspace definition (28 crates)
├── crates/                 # Individual crates
│   ├── grim-tensor/        # Core tensor abstractions (DType, Device, Shape)
│   ├── grim-tensor-graph/  # Fusion patterns for tensor operations
│   ├── grim-quant/         # Weight quantization (Q4_K, Q8_0, NF4, FP8, IQ)
│   ├── grim-format/        # GGUF/Safetensors/.grim I/O
│   ├── grim-backend-cpu/   # CPU backend (SIMD, OxiBLAS)
│   ├── grim-backend-rocm/  # ROCm/HIP backend (primary GPU target)
│   ├── grim-backend-cuda/  # CUDA backend (cuBLAS)
│   ├── grim-backend-vulkan/# Vulkan backend
│   ├── grim-backend-metal/ # Metal backend (Apple)
│   ├── grim-nn/            # Neural network modules (Linear, Embedding, etc.)
│   ├── grim-core/          # Model traits, Session, KV cache, Sampler
│   ├── grim-models/        # Model architecture crates
│   │   ├── transformer/    # Transformer models (LLaMA, Mistral, etc.)
│   │   ├── mamba/          # Mamba/SSM models
│   │   ├── vision/         # Vision encoder models
│   │   ├── audio/          # Audio encoder models
│   │   └── diffusion/      # Diffusion model architectures
│   ├── grim-memory/        # Paged KV cache pool, prefix sharing, spilling
│   ├── grim-kvquant/       # Runtime KV cache compression (§5.4)
│   ├── grim-kvtransport/   # Tiered KV transport (GPU→RAM→NVMe)
│   ├── grim-scheduler/     # Continuous-batching scheduler (3-queue)
│   ├── grim-speculative/   # Default-on speculative decoding (§5.3)
│   ├── grim-autograd/      # LoRA/QLoRA backward pass tracing
│   ├── grim-engine/        # Runtime orchestrator (Engine, tick)
│   ├── grim-server/        # HTTP serving layer (OpenAI-compatible)
│   ├── grim-cli/           # Command-line interface
│   ├── grim-disagg/        # Disaggregation layer (prefill/decode split)
│   ├── grim-plugin/        # Plugin system (dylib + WASM)
│   ├── grim-constrain/     # Structured & JSON Schema grammar decoding
│   └── grim-garage/        # Training dashboard web app
├── docs/                   # Documentation
│   ├── onboarding.md       # This file
│   ├── architecture.md     # Architecture overview
│   ├── cli.md              # CLI reference
│   ├── configuration.md    # Configuration reference
│   ├── data-model.md       # Data structures and formats
│   ├── glossary.md         # Domain terms
│   ├── integrations.md     # External integrations
│   ├── troubleshooting.md  # Common errors and fixes
│   └── howto/              # How-to guides
└── models/                 # Local model cache (created on first run)
```

## Test Location Map

| Crate | Unit Tests | Integration Tests | Doc Tests |
|---|---|---|---|
| `grim-tensor` | `src/tensor.rs` | `tests/golden_*.rs` | `src/lib.rs` |
| `grim-core` | `src/` | N/A | `src/lib.rs` |
| `grim-engine` | `src/` | `tests/` | `src/lib.rs` |
| `grim-backend-cpu` | `src/` | `tests/` | `src/lib.rs` |
| `grim-backend-rocm` | `src/` | `tests/` | `src/lib.rs` |
| `grim-models/*` | `src/` | `tests/` | `src/lib.rs` |

## Build Configuration

The workspace uses Thin LTO with a single codegen unit in release mode:

```toml
[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1
```

### Feature Flags

Key feature flags:

- `rocm` - Enables the ROCm backend
- `rocm` - Enables the ROCm backend (workspace-level)
- `rocm-aiter` - ROCm AI tensor operations (grim-backend-rocm)
- `rocm-profile` - ROCm profiling support (grim-backend-rocm)
- `rccl` - ROCm collective communications (grim-backend-rocm)
- `cubecl` - Cubecl HIP runtime integration (grim-backend-rocm)
- `rocm-mem` - ROCm memory allocation (grim-nn)
- `cuda-mem` - CUDA memory allocation (grim-nn)
- `vulkan-mem` - Vulkan memory allocation (grim-nn)
- `metal-mem` - Metal memory allocation (grim-nn)
- `wasm-sandbox` - WASM plugin sandboxing (grim-plugin)
- `gpu-selection` - All GPU backends enabled

Enable features during build:

```bash
cargo build --features "rocm,wasm-sandbox"
```

## Workspace-wide Commands

Build the entire workspace:

```bash
cargo build --workspace
```

Run clippy:

```bash
cargo clippy --workspace --all-targets
```

Format code:

```bash
cargo fmt
cargo fmt --check  # to verify formatting
```

## Maintainer Information

Maintainer Status: Actively maintained. Please see the GitHub repository for the current list of maintainers.

## Quick Reference

| Task | Command |
|---|---|
| Build all crates | `cargo build --release` |
| Run tests | `cargo test --workspace` |
| Run clippy | `cargo clippy --workspace --all-targets -D warnings` |
| Format check | `cargo fmt --check` |
| Run specific crate tests | `cargo test -p <crate-name>` |
| Enable ROCm features | `cargo build --features rocm` |
| Run GPU tests | `GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm --features rocm-aiter,rccl` |