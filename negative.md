# Grim Project: What This Project Does NOT Do

This document lists explicit boundaries, missing features, and non-goals of Grim.

## Features Genuinely Not Implemented (No Partial Code)

### General-Purpose ML Framework

Grim is **NOT** a general-purpose machine learning framework like PyTorch or TensorFlow. It is specifically designed for LLM inference with:
- No autograd beyond adapter training
- No general-purpose neural network layers
- No dynamic graph construction
- No training of base model weights

### Authentication & Security

Grim **does NOT include**:
- User authentication/authorization
- API key management
- TLS/SSL termination (handled by reverse proxy)
- Audit logging

### Cloud Integrations

Grim **does NOT have**:
- AWS SageMaker integration
- GCP Vertex AI integration
- Azure ML integration

### Container Orchestration

Grim **does NOT provide**:
- Kubernetes deployment manifests
- Docker orchestration
- Service mesh integration (use a service mesh)

### Framework Interoperability

Grim **does NOT integrate with**:
- LangChain
- LlamaIndex
- HuggingFace transformers library
- DeepSpeed
- FlashAttention libraries

## Features with Stub Implementations

### ONNX Support

The `grim-format/src/onnx.rs` module exists as a **stub** that requires the `ort` (ONNX Runtime) Rust crate to enable. This is **not a simple wiring fix** - it requires:
1. Adding `ort` as a workspace dependency
2. Implementing proper tensor extraction from ONNX
3. Mapping ONNX operators to Grim operations

**Status**: Requires external dependency, not yet enabled.

### PyTorch Bindings

No Python FFI bindings exist. This requires:
- `pyo3` or similar for Rust-Python interop
- Binding every model function
- Handling GIL and memory management

**Status**: Not a wiring issue, requires new infrastructure.

## Backend Limitations (By Design)

### CUDA Backend

The CUDA backend **does NOT implement**:
- FlashAttention (returns `Unimplemented` error)
- Tensor cores via WMMA
- Memory-efficient attention variants (xFormers)

It only uses cuBLAS for GEMM operations - this is intentional design, not missing wiring.

### Metal Backend

The Metal backend **does NOT support**:
- Intel Macs (uses CPU fallback)
- macOS 12 and earlier

This is by design - Metal API requires Apple Silicon.

## Serving Architecture

### No Model Version Semantics

Grim **does NOT implement**:
- Model version semantics in requests
- Traffic routing by version
- Canary deployments

These are architectural choices, not unwired components.

## What Exists and IS Wired

These features are fully integrated and working:
- ✅ GGUF, GRIM, and safetensors model loading
- ✅ GGUF tokenizer extraction and encoding
- ✅ LoRA/QLoRA adapter loading and inference
- ✅ Speculative decoding (DSpark/MTP)
- ✅ KV cache with spilling to RAM/NVMe
- ✅ KV quantization (TurboQuant)
- ✅ Auto-scaling batch scheduler
- ✅ Op-level plugin system (WASM and dylib paths)
- ✅ Service management (systemd/launchd/SCM)

## Future Considerations

The following are planned but not yet implemented:
1. ONNX model support (requires `ort` crate integration)
2. Full multi-GPU tensor parallelism
3. Quantization-aware training capabilities
4. Native PyTorch binding support