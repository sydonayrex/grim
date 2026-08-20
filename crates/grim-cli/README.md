# `grim-cli`

`grim-cli` is the primary executable binary and command-line interface for the Grim framework. It exposes commands for serving OpenAI-compatible APIs, interactive terminal chat, model downloads, hardware pre-flight diagnostics, quantization, benchmarking, and adapter fine-tuning.

## Boundaries

`grim-cli` does **not**:
- Implement neural network forward or backward passes directly (delegated to `grim-models-*`, `grim-autograd`, and `grim-engine`).
- Execute raw GPU shaders or manage low-level driver contexts directly (delegated to `grim-backend-*`).
- Contain custom tensor storage allocators (delegated to `grim-tensor` and `grim-memory`).

## Dependency Graph

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#2b2d42', 'edgeLabelBackground':'#ffffff', 'tertiaryColor': '#edf2f4'}}}%%
flowchart TD
    subgraph Sibling Dependents
        User["User / Shell / Systemd"]
    end

    subgraph Focal Node
        grim_cli["grim-cli"]
    end

    subgraph Workspace Dependencies
        grim_engine["grim-engine"]
        grim_server["grim-server"]
        grim_autograd["grim-autograd"]
        grim_format["grim-format"]
        grim_quant["grim-quant"]
        grim_memory["grim-memory"]
        grim_disagg["grim-disagg"]
        grim_backend_rocm["grim-backend-rocm"]
        grim_backend_cuda["grim-backend-cuda"]
        grim_backend_vulkan["grim-backend-vulkan"]
        grim_backend_metal["grim-backend-metal"]
        grim_backend_cpu["grim-backend-cpu"]
        grim_core["grim-core"]
        grim_tensor["grim-tensor"]
    end

    subgraph External Dependencies
        clap["clap"]
        tokio["tokio"]
        serde_json["serde_json"]
    end

    User -->|executes binary| grim_cli

    grim_cli --> grim_engine
    grim_cli --> grim_server
    grim_cli --> grim_autograd
    grim_cli --> grim_format
    grim_cli --> grim_quant
    grim_cli --> grim_memory
    grim_cli --> grim_disagg
    grim_cli --> grim_backend_rocm
    grim_cli --> grim_backend_cuda
    grim_cli --> grim_backend_vulkan
    grim_cli --> grim_backend_metal
    grim_cli --> grim_backend_cpu
    grim_cli --> grim_core
    grim_cli --> grim_tensor
    grim_cli --> clap
    grim_cli --> tokio
    grim_cli --> serde_json

    classDef focal fill:#d90429,stroke:#ef233c,stroke-width:2px,color:#ffffff;
    classDef workspace fill:#2b2d42,stroke:#8d99ae,stroke-width:1px,color:#edf2f4;
    classDef sibling fill:#4a4e69,stroke:#9a8c98,stroke-width:1px,color:#f2e9e4;
    classDef external fill:#1f2421,stroke:#495867,stroke-width:1px,color:#f0f3f4;

    class grim_cli focal;
    class grim_engine,grim_server,grim_autograd,grim_format,grim_quant,grim_memory,grim_disagg,grim_backend_rocm,grim_backend_cuda,grim_backend_vulkan,grim_backend_metal,grim_backend_cpu,grim_core,grim_tensor workspace;
    class User sibling;
    class clap,tokio,serde_json external;
```

## Public API Overview

`grim-cli` produces the `grim` binary. Subcommands include:

- **`serve`**: Bootstraps the HTTP server with OpenAI and Ollama compatible endpoints.
- **`run`**: Runs a single prompt or interactive REPL session from the command line.
- **`chat`**: Multi-turn chat session with automatic template sanitization.
- **`pull` / `dl`**: Downloads model checkpoints from Hugging Face or Ollama registries.
- **`convert` / `oxidize`**: Converts GGUF or SafeTensors weights into optimized `.grim` files.
- **`doctor`**: Diagnostic tool checking hardware drivers, VRAM fit (`--model`), and ROCm/CUDA/Vulkan availability.
- **`train`**: Fine-tunes LoRA, QLoRA, LoRA+, PiSSA, and VeRA adapters with deterministic `--seed`.
- **`eval`**: Evaluates model loss, perplexity, and benchmark accuracy.
- **`bench`**: Measures token generation speed, TTFT, and throughput.
- **`quantize`**: Calibrates and quantizes model weights into Q8_0, Q4_K, Q5_K, and FP8 formats.
- **`inspect`**: Inspects tensor headers and metadata of `.grim` and `.gguf` files.
- **`export`**: Merges trained adapters back into base model checkpoints.

## Usage Example

```bash
# 1. Inspect hardware environment and model fit
grim doctor --model models/llama-3.2-1b.gguf

# 2. Run inference server on port 8080
grim serve --port 8080 --model models/llama-3.2-1b.gguf

# 3. Train a LoRA adapter with deterministic seed
grim train \
  --model models/llama-3.2-1b.gguf \
  --data data/train.jsonl \
  --output adapters/my-lora.grim \
  --seed 42 \
  --lr 1e-4 \
  --epochs 3
```

## Use Cases

- Standalone deployment binary for local and production inference serving.
- Hardware qualification and VRAM pre-flight verification on edge and datacenter systems.
- Scriptable pipeline integration for model conversion, calibration, and fine-tuning.

## Edge Cases, Limitations, and Quirks

1. **Pre-flight Device Selection**: Setting `--device <backend>` configures the global target backend before engine initialization.
2. **Deterministic PRNG**: Supplying `--seed <u64>` to `grim train` initializes reproducible adapter weight distributions across CPU and ROCm/CUDA backends.
3. **Local Catalog Resolution**: Model names matching local files in the current working directory or `models/` take precedence over remote registry downloads.

## Build Flags, Feature Flags, and Environment Variables

- `default = ["rocm"]`: ROCm GPU backend enabled by default.
- `cuda`: Enables NVIDIA CUDA compilation.
- `vulkan`: Enables Vulkan compute compilation.
- `metal`: Enables Apple Metal compilation on macOS.
- **Environment variables**: `GRIM_HOST`, `GRIM_PORT`, `GRIM_BACKEND`, `GRIM_LOG`.
