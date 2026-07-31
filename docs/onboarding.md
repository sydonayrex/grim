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

Build all 31 crates in release mode:

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
├── Cargo.toml              # Workspace definition
├── crates/                 # Individual crates
│   ├── grim-tensor/        # Core tensor abstractions
│   ├── grim-core/          # Model traits, Session, KV cache, Sampler
│   ├── grim-engine/        # Runtime orchestrator
│   ├── grim-server/        # HTTP serving layer
│   ├── grim-cli/           # Command-line interface
│   └── ... (26 more crates)
├── docs/                   # Documentation
│   ├── onboarding.md       # This file
│   ├── architecture.md     # Architecture overview
│   └── howto/              # How-to guides
└── models/                 # Local model cache (created on first run)
```

## Test Location Map

| Crate | Unit Tests | Integration Tests | Doc Tests |
|---|---|---|---|
| `grim-tensor` | `src/tensor.rs` | `tests/golden_*.rs` | `src/lib.rs` |
| `grim-core` | `src/` | N/A | `src/lib.rs` |
| `grim-engine` | `src/` | `tests/` | `src/lib.rs` |
| `grim-backend-cpu` | `src/` | N/A | `src/lib.rs` |
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
- `rocm-aiter` - ROCm AI tensor operations
- `rocm-kernel-macros` - ROCm kernel macros
- `wasm-sandbox` - WASM plugin sandboxing
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

TODO: add maintainer contact

## Quick Reference

| Task | Command |
|---|---|
| Build all crates | `cargo build --release` |
| Run tests | `cargo test --workspace` |
| Run clippy | `cargo clippy --workspace --all-targets -D warnings` |
| Format check | `cargo fmt --check` |
| Run specific crate tests | `cargo test -p <crate-name>` |
| Enable ROCm features | `cargo build --features rocm` |
| Run GPU tests | `GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm --features rocm-aiter` |