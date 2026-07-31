# How to Install Grim

## Prerequisites

### Linux (ROCm)

```bash
# Install ROCm 7.0+ (follow AMD's official installation guide)
# Verify installation:
/opt/rocm/bin/rocminfo

# Install Rust 1.85+:
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Linux (CUDA)

```bash
# Install CUDA 11.8+ (follow NVIDIA's official installation guide)
# Verify installation:
nvcc --version
nvidia-smi

# Install Rust 1.85+:
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### macOS (Metal)

```bash
# macOS 13.0+ with Apple Silicon
rustup update stable
```

### Windows

```powershell
# Install Rust via rustup-init.exe
# Install Visual Studio Build Tools
# Install CUDA if using NVIDIA GPU
```

## Build from Source

```bash
# Clone the repository
git clone https://github.com/poolside-ai/grim.git
cd grim

# Build in release mode
cargo build --release -p grim-cli -p grim-server

# Or build the full workspace
cargo build --release --all
```

## Optional: Run Tests

```bash
# Run unit tests
cargo test --workspace

# Run GPU tests (ROCm only)
GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm
```

## Installation

```bash
# Copy binary to PATH
cp target/release/grim /usr/local/bin/

# Verify installation
grim --version
grim doctor
```

## Service Installation

```bash
# Linux - systemd
grim service install

# macOS - launchd
grim service install --exec-path /usr/local/bin/grim

# Windows - SCM (admin)
grim service run --config /etc/grim/grim.toml
```