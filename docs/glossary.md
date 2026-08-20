# Glossary

Project-specific domain terms, structures, and acronyms used across the Grim codebase.

- **BlockTable** (`grim-memory`): Paged memory index mapping virtual token positions to physical non-contiguous VRAM memory blocks.
- **BlockSizeBand** (`grim-backend-rocm`): Workgroup occupancy classification (`Small`, `Medium`, `Large`) used by the ROCm autotuner to choose optimal tile dimensions.
- **ConstrainedSampler** (`grim-constrain`): Decorator wrapping any `Sampler` to mask logit outputs according to finite-state grammar transitions (JSON-mode or JSON-Schema).
- **Disaggregation** (`grim-disagg`): Distributed execution strategy separating compute-heavy prefill operations from memory-bandwidth-bound decode token generation across distinct instances.
- **EvoPress** (`grim-quant`): Evolutionary search algorithm evaluating layer-by-layer quantization bit-width allocations against importance calibration matrices.
- **GGUF** (`grim-format`): Standard binary format storing tensor weights, vocabulary token maps, and architecture metadata headers.
- **Grim Container (`.grim`)** (`grim-format`): Native single-file binary container format with 64-byte aligned tensor offsets for zero-copy DMA uploads.
- **ModelFootprint** (`grim-format`): Header-only descriptor computing expected VRAM footprint and context buffer allocations without loading model weights.
- **Oxidizer** (`grim-quant` / `grim-cli`): Pipeline automating model calibration, evolutionary quantization search, and ROCm kernel optimization into `.grim` files.
- **PiSSA** (`grim-autograd`): Principal Singular values and Singular vectors Adaptation — LoRA variant initializing adapters using the principal SVD components of the base weights.
- **SessionT** (`grim-core`): Object-safe trait encapsulating per-request context history, KV cache slots, and request-scoped PRNG states.
- **SpeculativeCausalLm** (`grim-speculative`): Engine wrapper orchestrating draft token generation and verification against target causal language models.
- **WavefrontTiled** (`grim-format`): Memory layout rearranging 2D tensor blocks into Wave32/Wave64 native hardware tile formats for accelerated memory access.
