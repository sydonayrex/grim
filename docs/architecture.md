# Architecture Overview

Grim is a pure-Rust neural network inference and fine-tuning engine supporting autoregressive language models, state-space models, vision encoders, audio encoders, and diffusion architectures across CPU, ROCm, CUDA, Vulkan, and Metal backends.

## Workspace Structure

The workspace is organized into five functional layers:

- **Foundation Layer** (`grim-tensor`, `grim-tensor-graph`, `grim-quant`, `grim-format`): Tensors, quantization codecs (Q8_0, Q4_K, Q5_K, Q6_K, IQ4_NL, FP8, MXFP4), and container file I/O (GGUF, SafeTensors, `.grim`).
- **Backend Layer** (`grim-backend-cpu`, `grim-backend-rocm`, `grim-backend-cuda`, `grim-backend-vulkan`, `grim-backend-metal`): Hardware execution engines and vendor runtime bindings.
- **Model Layer** (`grim-nn`, `grim-models-transformer`, `grim-models-mamba`, `grim-models-vision`, `grim-models-audio`, `grim-models-diffusion`): Neural network building blocks, weight loaders, and architecture implementations.
- **Runtime Layer** (`grim-core`, `grim-engine`, `grim-scheduler`, `grim-memory`, `grim-kvquant`, `grim-kvtransport`, `grim-autograd`, `grim-speculative`, `grim-constrain`): Inference orchestration, continuous batching, paged memory management, speculative execution, grammar-constrained decoding, and adapter autograd.
- **Service Layer** (`grim-server`, `grim-cli`, `grim-plugin`, `grim-disagg`, `grim-garage`): HTTP/REST API endpoints, command-line interface, WASM plugin sandbox, disaggregated serving, and training telemetry dashboard.

## Workspace Dependency Graph

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#2b2d42', 'edgeLabelBackground':'#ffffff', 'tertiaryColor': '#edf2f4'}}}%%
flowchart TD
    subgraph Foundation Layer
        tensor["grim-tensor"]
        graph["grim-tensor-graph"]
        quant["grim-quant"]
        format["grim-format"]
    end

    subgraph Backend Layer
        cpu["grim-backend-cpu"]
        rocm["grim-backend-rocm"]
        cuda["grim-backend-cuda"]
        vulkan["grim-backend-vulkan"]
        metal["grim-backend-metal"]
    end

    subgraph Model Layer
        nn["grim-nn"]
        transformer["grim-models-transformer"]
        mamba["grim-models-mamba"]
        vision["grim-models-vision"]
        audio["grim-models-audio"]
        diffusion["grim-models-diffusion"]
    end

    subgraph Runtime Layer
        core["grim-core"]
        engine["grim-engine"]
        scheduler["grim-scheduler"]
        memory["grim-memory"]
        kvquant["grim-kvquant"]
        kvtransport["grim-kvtransport"]
        autograd["grim-autograd"]
        speculative["grim-speculative"]
        constrain["grim-constrain"]
    end

    subgraph Service Layer
        server["grim-server"]
        cli["grim-cli"]
        plugin["grim-plugin"]
        disagg["grim-disagg"]
        garage["grim-garage"]
    end

    tensor --> quant
    tensor --> format
    tensor --> cpu
    tensor --> rocm
    tensor --> cuda
    tensor --> vulkan
    tensor --> metal

    format --> quant
    nn --> tensor
    nn --> transformer
    nn --> mamba
    nn --> vision
    nn --> audio
    nn --> diffusion

    core --> tensor
    core --> format
    memory --> core
    kvquant --> core
    kvtransport --> core
    constrain --> core
    constrain --> format

    engine --> core
    engine --> memory
    engine --> autograd
    engine --> speculative
    engine --> rocm
    engine --> cpu

    scheduler --> core
    scheduler --> memory

    server --> engine
    server --> scheduler
    server --> constrain
    server --> plugin

    cli --> engine
    cli --> server
    cli --> autograd
    cli --> format
    cli --> quant

    garage --> autograd
    disagg --> core

    classDef foundation fill:#2b2d42,stroke:#8d99ae,stroke-width:1px,color:#edf2f4;
    classDef backend fill:#1d3557,stroke:#457b9d,stroke-width:1px,color:#f1faee;
    classDef model fill:#4a4e69,stroke:#9a8c98,stroke-width:1px,color:#f2e9e4;
    classDef runtime fill:#3d5a80,stroke:#98c1d9,stroke-width:1px,color:#e0fbfc;
    classDef service fill:#d90429,stroke:#ef233c,stroke-width:1px,color:#ffffff;

    class tensor,graph,quant,format foundation;
    class cpu,rocm,cuda,vulkan,metal backend;
    class nn,transformer,mamba,vision,audio,diffusion model;
    class core,engine,scheduler,memory,kvquant,kvtransport,autograd,speculative,constrain runtime;
    class server,cli,plugin,disagg,garage service;
```

## Key Types and Traits

### Data Flow Pipeline

1. **Model Ingestion** (`grim-format`): Reads GGUF, SafeTensors, or `.grim` files; exposes tensors via `TensorProvider`.
2. **Weight Instantiation** (`grim-nn`): Loads tensors into model layers using `WeightSource`.
3. **Execution Scheduling** (`grim-scheduler`): Coordinates request queues (waiting, running, swapped) and continuous batching.
4. **Token Generation** (`grim-engine` / `grim-speculative`): Computes forward passes, evaluates draft tokens, and updates KV cache states.
5. **Sampling & Constraint** (`grim-core` / `grim-constrain`): Applies temperature, top-k/top-p, and FSM token masking for structured JSON outputs.
6. **API Delivery** (`grim-server` / `grim-cli`): Streams Server-Sent Events (SSE) or returns complete JSON responses.

### Core Abstractions

- **`BackendDevice`** (`grim-tensor`): Hardware-agnostic compute contract (dense matmul, quantized matmul, RMSNorm, RoPE, SiLU-Mul, softmax).
- **`BackendStorage`** (`grim-tensor`): Device-allocated memory buffer.
- **`ComputeHandle`** (`grim-tensor`): Asynchronous stream and event tracker.
- **`KvCache`** (`grim-core`): Key-value state cache abstraction.
- **`SessionT`** (`grim-core`): Request context tracking token history and PRNG state.
- **`Sampler`** (`grim-core`): Logit transformation and token selection interface.
- **`Tape`** (`grim-autograd`): Differentiable execution tape for parameter-efficient adapter fine-tuning.

## Backend Architecture

### Device Abstraction

Hardware devices are identified via the `Device` enum:

```rust
pub enum Device {
    Cpu,
    Rocm(usize),
    Cuda(usize),
    Vulkan,
    Metal(usize),
}
```

### Backend Implementations

- **`grim-backend-cpu`**: Vectorized execution using AVX2/AVX-512/NEON SIMD intrinsics with scalar fallback.
- **`grim-backend-rocm`**: Primary GPU backend with rocBLAS bindings, HIP runtime FFI, Wave32/Wave64 kernel specialization, and JIT kernel compilation.
- **`grim-backend-cuda`**: NVIDIA GPU backend with cuBLAS GEMM dispatch and runtime PTX compilation.
- **`grim-backend-vulkan`**: Platform-portable compute shader backend dispatching SPIR-V pipelines.
- **`grim-backend-metal`**: Apple Silicon GPU backend using Metal Performance Shaders and MSL shaders.

## Error Handling Conventions

The workspace uses `thiserror` for library error definitions:

- `grim-tensor::Error`: Tensor allocation, dimension mismatch, and backend execution errors.
- `grim-core::Error`: Engine session, KV cache, and configuration errors.
- `grim-format::Error`: File header corruption, missing tensors, and tokenizer errors.
- `grim-autograd::Error`: Gradient tape shape mismatches and optimizer numerical errors.

## Non-Obvious Design Decisions

1. **Self-Contained FSM Masking**: `grim-constrain` implements finite-state JSON parsing in pure Rust without external C grammar engines.
2. **Reverse-Mode Adapter Scoping**: `grim-autograd` only records operations that mutate trainable adapter weights, keeping base model parameters immutable in VRAM.
3. **Implicit Vulkan Layer Suppression**: `grim-backend-vulkan` suppresses display-compositor-dependent layers to avoid headless hangs in CI environments.
4. **Paged Memory Allocation**: `grim-memory` uses block tables and prefix hashing to eliminate memory fragmentation during long-context generation.
