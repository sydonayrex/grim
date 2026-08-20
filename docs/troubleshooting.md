# Troubleshooting Guide

This guide catalogs common failure modes, error messages, and their resolutions based on codebase implementations across Grim.

---

## 1. Vulkan Backend Issues

### Symptom: `cargo test` hangs indefinitely or exceeds 60 seconds on Vulkan tests
- **Cause**: Implicit system Vulkan layers (e.g. Steam overlay, MangoHud, gamescope WSI) intercept `vkCreateInstance` and block attempting to establish a connection to an active X11/Wayland display compositor.
- **Fix**: Disable implicit Vulkan layers during test runs:
  ```bash
  VK_LOADER_LAYERS_DISABLE="~all~" cargo test -p grim-backend-vulkan
  ```
  *(Note: Grim automatically sets this in `.cargo/config.toml` and `VulkanContext::init()` unless explicitly overridden).*

### Symptom: `vkMapMemory failed with status -5` (`VK_ERROR_MEMORY_MAP_FAILED`)
- **Cause**: Attempting to call `vkMapMemory` directly on a buffer allocated in `DEVICE_LOCAL` VRAM on discrete graphics cards (which is not host-mappable).
- **Fix**: Allocate buffers that require CPU zero-initialization or host upload using `VulkanStorage::alloc_gpu` (which requests `HOST_VISIBLE | HOST_COHERENT` memory), or execute zeroing via a GPU compute shader.

---

## 2. AMD ROCm / HIP Issues

### Symptom: `RocmDevice::probe()` returns empty list or fails with missing shared library
- **Cause**: Dynamic loader cannot find `libamdhip64.so.6` or `librocblas.so.4` in system search paths.
- **Fix**: Verify ROCm installation and export `LD_LIBRARY_PATH`:
  ```bash
  export ROCM_PATH=/opt/rocm
  export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
  grim doctor
  ```

### Symptom: `hiprtcCompileProgram failed` during JIT kernel compilation
- **Cause**: Target GCN architecture mismatch or missing LLVM device compiler backend.
- **Fix**: Explicitly set the GPU architecture via `GRIM_GPU_TARGET`:
  ```bash
  export GRIM_GPU_TARGET=gfx1100  # for RX 7900 / RDNA3
  ```

---

## 3. NVIDIA CUDA Issues

### Symptom: `nvcc failed with status 127`
- **Cause**: `nvcc` binary is not present in `$PATH`.
- **Fix**: Set the `NVCC` or `CUDA_PATH` environment variable:
  ```bash
  export CUDA_PATH=/usr/local/cuda
  export PATH=$CUDA_PATH/bin:$PATH
  ```

---

## 4. Model Loading & Tokenization Issues

### Symptom: `minijinja::Error: unknown syntax / method not found in template`
- **Cause**: Model's Jinja chat template contains Python-specific dictionary methods (e.g. `.get('key')`, `.items()`, `.startswith(...)`) not supported by standard MiniJinja.
- **Fix**: The template must pass through `grim_format::tokenizer::sanitize_jinja_template()` which rewrites Python method invocations into MiniJinja filter expressions.

### Symptom: `TensorError: weight tensor appears to be zeroed or corrupt`
- **Cause**: GGUF/SafeTensors weight offset points to corrupt or incomplete download payload.
- **Fix**: Run model verification and re-download:
  ```bash
  grim verify path/to/model.gguf
  grim pull <model-name>
  ```

---

## 5. Training & Fine-Tuning Issues

### Symptom: `Gradients contain NaN / Inf` during LoRA fine-tuning
- **Cause**: Learning rate too high or numerical underflow in mixed-precision backward passes.
- **Fix**:
  - Lower the learning rate (`--lr 1e-4` or `5e-5`).
  - Enable warmup steps (`--warmup_steps 50`).
  - Use AdamW with gradient clipping (`--max_grad_norm 1.0`).
