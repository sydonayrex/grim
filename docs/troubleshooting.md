# Troubleshooting

This document covers common issues and their causes, derived from code evidence in the repository.

## Build Failures

### Missing Rust toolchain (version 1.85+)

**Symptom**: `error: failed to parse manifest... edition 2024 is not supported`

**Cause**: Your Rust version is too old.

**Fix**: Update your toolchain:
```bash
rustup update 1.85
rustup default stable
```

### Missing LLVM development libraries

**Symptom**: `linker error: cannot find -l LLVM` or `llvm-sys` build failure

**Cause**: The `llvm-dev` package is not installed.

**Fix**: Install LLVM development libraries:
```bash
# Ubuntu/Debian
sudo apt-get install llvm-dev clang

# Fedora/RHEL
sudo dnf install llvm-devel clang-devel

# macOS
brew install llvm
export LLVM_CONFIG="$HOME/opt/llvm/bin/llvm-config"
```

### ROCm library not found

**Symptom**: `error: cannot find -lhip_hcc` or `librocBLAS not found`

**Cause**: ROCm runtime libraries are not installed or not in the library path.

**Fix**: Install ROCm or set the library path:
```bash
# Set ROCm path
export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
```

### CUDA toolkit not found

**Symptom**: `error: cannot find -lcudart` or CUDA-related linker errors

**Cause**: CUDA toolkit is not installed.

**Fix**: Install CUDA toolkit matching your driver version:
```bash
# Ubuntu - use NVIDIA official repo
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb -o cuda-keyring.deb
sudo dpkg -i cuda-keyring.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-1
```

## Runtime Errors

### Model file not found

**Symptom**: `Model 'llama3' not found on disk; no mock fallback is provided`

**Cause**: The model is not in the model cache directory.

**Fix**: Download the model first:
```bash
grim pull llama3
# or
grim dl hf.co/user/model
```

Check model cache location:
```bash
env | grep GRIM_MODELS_DIR
```

### Invalid GGUF file

**Symptom**: `invalid magic bytes` or `failed to parse GGUF header`

**Cause**: Corrupted or non-GGUF model file.

**Fix**: Download the model again or verify the file:
```bash
grim oxidizer info broken-model.gguf
```

### CUDA out of memory

**Symptom**: `CUDA error: out of memory`

**Cause**: Model is too large for GPU VRAM.

**Fix**: 
1. Use a smaller model
2. Convert to a lower-quantization format (Q4_K instead of Q8_0)
3. Set environment variable:
```bash
export GRIM_AVAILABLE_VRAM=16000000000  # 16GB in bytes
```

### ROCm device not found

**Symptom**: `No ROCm GPU found` or `HIP error: HIP_ERROR_NOT_INITIALIZED`

**Cause**: ROCm is not properly installed or accessible.

**Fix**:
```bash
# Check ROCm status
/opt/rocm/bin/rocminfo

# Verify GPU is visible
clinfo  # if available
```

### KV cache eviction during long context

**Symptom**: Slow response times or "context full" errors

**Cause**: Prompt exceeds KV cache capacity.

**Fix**: 
1. Use a model with smaller context length
2. Enable KV compression:
```rust
// In code
use grim_kvquant::{OmniKvCompressor, KvModality};
engine.config.kv_compressor = Some(Arc::new(
    OmniKvCompressor::new(KvModality::Text, 0.5)
));
```

### Permission denied on model directory

**Symptom**: `Permission denied` when accessing `/var/lib/grim/models`

**Cause**: Insufficient permissions for system-wide install.

**Fix**: Use user directory or fix permissions:
```bash
# Option 1: Use user cache
export GRIM_MODELS_DIR=$HOME/.grim/models

# Option 2: Fix permissions (requires sudo)
sudo chown -R $USER:$USER /var/lib/grim
```

## Error Types Reference

### grim-tensor::Error

| Variant | Description |
|---|---|
| `ShapeMismatch { expected, got }` | Tensor shape mismatch (expected vs. actual dims) |
| `DTypeMismatch(String)` | Data type mismatch or unsupported |
| `DeviceMismatch(String)` | Device allocation or transfer failure |
| `Backend(String)` | Backend operation error |
| `Shape(String)` | Shape validation error |
| `Unimplemented(String)` | Operation not implemented on this backend |

### grim-core::Error

| Variant | Description |
|---|---|
| `Tensor(TensorError)` | Tensor operation failed (wraps `grim_tensor::Error`) |
| `Config(String)` | Configuration problem |
| `Session(String)` | Session-related error |
| `KvCache(String)` | KV cache operation failed |
| `Sampler(String)` | Sampling error |

### grim-engine (via grim-core::Error)

| Source | Description |
|---|---|
| `Engine::load_model` | Model not found in catalog — returns `Config` variant |
| `Engine::register_adapter` | Adapter ID not found — returns `Config` variant |
| `Engine::tick` | Unknown request ID — returns `Session` variant |

## Server Errors

### HTTP 400 Bad Request

**Symptom**: `{"error": "unknown request field 'invalid_field'"}`

**Cause**: Request contains unrecognized field.

**Fix**: Review the request body against the known fields in `docs/cli.md` or `grim-server/src/lib.rs`.

### HTTP 404 Not Found

**Symptom**: `Model 'model-name' is not loaded and could not be found in the catalog`

**Cause**: Model not available in cache.

**Fix**: Download the model with `grim pull model-name`.

### HTTP 500 Internal Server Error

**Symptom**: Generic server error with details in logs

**Cause**: Backend operation failed.

**Fix**: Check server logs and verify model file integrity.

## GPU Tests Require Special Setup

**Symptom**: `cargo test -p grim-backend-rocm` fails with "GPU tests skipped"

**Cause**: `GRIM_RUN_GPU_TESTS` not set.

**Fix**:
```bash
export GRIM_RUN_GPU_TESTS=1
cargo test -p grim-backend-rocm --features rocm-aiter
```

## Known Limitations

### CUDA Backend

- Only implements cuBLAS GEMM; other operations return `Unimplemented`
- No fused kernels (attention, MLP fusion)
- Not all ops are supported

### Vulkan Backend

- Platform-agnostic fallback
- May fall back to CPU for unsupported ops
- No kernel fusion

### Metal Backend

- Only functional on Apple Silicon (M1/M2/M3)
- Intel Macs use CPU fallback
- Requires `target_vendor = "apple"` in target triple

### Speculative Decoding

- Requires draft model and heads for full DSpark mode
- Falls back to plain autoregressive if heads missing
- Acceptance threshold based on confidence head predictions

### KV Quantization

- Runtime compression trade-off between memory and accuracy
- 4-bit compression typical for good balance
- 8-bit recommended for high accuracy requirements

## Debug Mode

Enable verbose logging:

```bash
RUST_LOG=debug grim serve
```

For GPU debugging:

```bash
# ROCm
ROCM_LOG_LEVEL=debug grim serve

# Vulkan
VK_LOADER_DEBUG=debug grim serve
```