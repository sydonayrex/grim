# How to Install Grim

## Goal

Install Grim, build it from source, and optionally register it as a background service.

## Prerequisites

### Linux (ROCm or CUDA)

```bash
# For AMD GPUs (ROCm 7.0+):
/opt/rocm/bin/rocminfo

# For NVIDIA GPUs (CUDA 11.8+):
nvcc --version
nvidia-smi

# Rust 1.85+ is required for both:
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

## Steps

1. **Build from source**
   Clone the repository and build the CLI.

   ```bash
   git clone https://github.com/poolside-ai/grim.git
   cd grim
   cargo build --release -p grim-cli
   ```

2. **Install the binary**
   Copy the binary to a location in your PATH.

   ```bash
   cp target/release/grim-cli /usr/local/bin/grim
   ```

3. **Verify the installation**
   Run the doctor command to re-verify every claim Grim makes about itself.

   ```bash
   grim doctor
   ```

   You can also run a pre-flight model check to predict memory fit before attempting to serve:
   ```bash
   grim doctor --model /path/to/model.gguf
   ```

4. **(Optional) Install as a background service**
   Register Grim to run continuously as a system service.

   ```bash
   grim service install
   
   # Or with a custom service name and config file:
   grim service install --name grim-daemon --config /etc/grim/grim.toml
   ```

## Expected Output

When running `grim doctor`, you should see validation checks passing for the unit on disk, OS service visibility, HTTP health, and GPU backend:
```
[grim] checking unit on disk... ok
[grim] checking OS service visibility... ok
...
```

## What Can Go Wrong

- **Missing GPU toolkits**: If ROCm or CUDA are not found during the build, the compiler will fail to build the GPU backends. **Recovery**: Ensure `/opt/rocm/bin` or the CUDA toolkit paths are in your environment variables.
- **Service installation fails**: The OS service manager may deny permission. **Recovery**: Run the service installation step with elevated privileges (e.g., `sudo` or Admin prompt).