# Grim Turbocharge Plan: Burn Integration, Unsloth-Style Training, and the `.grim` ROCm Format

## 1. Ecosystem Review & Architectural Insights

### Insights from Awesome-Rust-MachineLearning
The Rust ML ecosystem is rapidly maturing with a strong emphasis on safety, performance, and backend flexibility. Key takeaways for grim include:
*   **Tensor Math Libraries**: Crates like `ndarray`, `tch-rs` (PyTorch bindings), and specialized quantization libraries provide a foundation but often lack the deep, hardware-specific kernel fusion required for high-performance LLM inference and training on AMD ROCm.
*   **GGUF/GGML Ecosystem**: The existing `ggml-org/gguf` ecosystem is highly optimized for CPU and generic GPU inference via custom quant formats (Q4_K, Q5_K, etc.). However, it is primarily an *inference-only* format family. A native grim training/optimization pipeline must bridge the gap between GGUF's storage efficiency and a format capable of holding gradients or mixed-precision states for ROCm-specific fine-tuning (LoRA/QLoRA).

### Insights from Burn (tracel-ai/burn)
Burn is a flexible, comprehensive machine learning framework written in pure Rust. Key architectural patterns applicable to grim's native tensor processing:
*   **Backend-Agnostic Tensor Abstraction**: Burn separates the high-level tensor operations (`burn-tensor`) from the low-level backend implementations (CPU, CUDA, Metal, WebGPU). Grim can adopt this pattern to ensure its ROCm-specific optimizations (`grim-backend-rocm`) are cleanly separated from the high-level model graph execution logic.
*   **Operator Fusion via Computation Graphs**: Burn represents models as computation graphs where operations (like MatMul, RMSNorm, and activation functions) can be fused into a single kernel launch. This drastically reduces memory bandwidth overhead—an issue highly relevant to AMD RDNA4 architectures where compute is abundant but VRAM access is the primary bottleneck.
*   **Custom Operator Registration**: Burn allows developers to register custom low-level operations (e.g., specific HIP or cuBLAS kernels) and expose them through a high-level, composable API. Grim's native method should leverage this pattern to expose `rocblas_gemm_ex` and custom fused attention/QKV projection kernels directly to the model execution layer without requiring external C/C++ bridge layers where possible.

### Insights from Unsloth (unslothai/unsloth)
Unsloth focuses on extreme training efficiency for LLMs, particularly through kernel fusion and advanced quantization techniques:
*   **Kernel Fusion Strategies**: Unsloth fuses operations like RMSNorm + MatMul and QKV projection + Attention calculation into single HIP/CUDA kernels. This avoids writing intermediate tensor states to VRAM, dramatically increasing tokens-per-second (TPS) during both inference and training.
*   **Quantization-Aware Training & Format Support**: Unsloth excels at efficient LoRA/QLoRA training using advanced quantized formats like NF4, FP4, FP8, and BF16. It materializes weights to higher precision only when necessary for the backward pass (gradients), keeping the forward pass in a lower-precision state to save VRAM.
*   **RoPE Scaling & Attention Optimization**: Unsloth implements highly optimized Rotary Position Embedding (RoPE) scaling and FlashAttention variants that minimize texture memory usage and maximize compute unit utilization on modern GPUs (including AMD ROCm targets).

---

## 2. The "Grim Oxidizer" v2 - Turbocharging the `.grim` Format via Burn Patterns

To leverage these insights, the **Grim Oxidizer** (`crates/grim-oxidizer/`) must evolve from a simple quantization/layout engine into a comprehensive model transformation and graph-fusion pipeline.

### A. Burn-Inspired Tensor Graph Fusion for ROCm
Instead of treating MatMul, LayerNorm, and activation functions as discrete tensor operations in the `grim-backend-rocm` layer, the Oxidizer will analyze the GGUF or source checkpoint and generate a **fused operation graph** specific to the target hardware's wavefront size (e.g., W32 for RDNA4).

*   **Fusion Strategy**: The Oxidizer identifies common LLM patterns (e.g., `RMSNorm -> Linear -> Silu/Gelu`) and pre-generates or hints at fused ROCm HIP kernel configurations.
*   **Backend Integration**: These fused operations are exposed via a native Rust API within `grim-backend-rocm`, wrapping `rocblas_gemm_ex` and custom HIP kernels, eliminating the need for intermediate memory allocations between logical steps in the model graph.

### B. The `.grim` Format as an Inference + Training Artifact
While standard GGUF is designed purely for inference (read-only weight materialization), the **`.grim` format** (`model-name-rocm-optimized.grim`) will be designed to support:
1.  **Inference Optimization**: Pre-aligned tensor layouts, fused kernel metadata, and RDNA4 wavefront-specific KV cache strategies.
2.  **Training Readiness**: Metadata and weight structures that can be materialized into mixed-precision states (BF16/FP8) for training workflows like LoRA or QLoRA, without requiring a full conversion back to raw Safetensors or FP32 checkpoints.

---

## 3. Unsloth-Inspired Native ROCm Training Capabilities

For grim to eventually support native model updates and fine-tuning on AMD hardware (RX 9070 XT and future RDNA architectures), the training pipeline must support advanced quantization and mixed-precision workflows.

### A. Supported Training Formats
The Oxidizer and subsequent training engine (`grim-train` or `grim-finetune`) will support the following format ingestion and materialization:
*   **GGUF**: Read-only ingestion for base model weights (e.g., Q4_K, Q5_K). During training preparation, GGUF weights are de-quantized and mapped to a higher precision buffer (BF16 or FP32) for gradient computation.
*   **`.grim`**: The native format that can store quantized inference weights *and* hold metadata required for fast materialization into BF16/FP8 training states. 
*   **FP4 / NF4 / FP8**: Support for Quanto-style or Unsloth-style 4-bit and 8-bit quantizations during the training phase (e.g., QLoRA). The Oxidizer will prepare `.grim` files that store weights in these formats but include the necessary dequantization kernels (matrices and scales) to materialize them on-the-fly into BF16 during the backward pass.
*   **BF16**: The primary target format for gradient accumulation and weight updates during ROCm-native training, as BF16 is natively supported by AMD RDNA4 compute units without the precision loss of FP16.

### B. Kernel Fusion for Training (Unsloth Patterns)
During training or fine-tuning (specifically LoRA/QLoRA), memory bandwidth is the primary constraint. The `grim-backend-rocm` will implement Unsloth-inspired fused kernels:
*   **RMSNorm + MatMul Fusion**: Combine the normalization step and the linear projection into a single HIP kernel launch, preventing the temporary storage of the normalized hidden states in VRAM.
*   **QKV Projection + Attention Fusion**: Fuse the query/key/value generation with the FlashAttention-like score calculation to maximize compute unit utilization on RDNA4 wavefronts.

### C. Native ROCm LoRA / QLoRA Support
Grim's training capabilities will natively support Low-Rank Adaptation (LoRA) and Quantized LoRA (QLoRA), leveraging the existing `grim-engine` adapter registry (`register_adapter`, `resolve_adapters`). The Oxidizer will generate `.grim` artifacts where LoRA rank decompositions are pre-aligned with ROCm memory layouts, ensuring that adapters can be injected into the inference or training pipeline without runtime reallocation penalties.

---

## 4. Technical Implementation Blueprint for `.grim` Format & Oxidizer Upgrades

### Step 1: Extend GGUF/.grim Parsing for Training & Fusion Metadata
*   **File**: `crates/grim-format/src/gguf.rs`, `crates/grim-format/src/tprov.rs`
*   **Action**: Enhance the `GgufProvider` and `.grim` parser to recognize a new namespace of metadata tags:
    *   `grim.train.quant_mode: fp4|fp8|bf16` (Indicates the target materialization precision for training loops).
    *   `grim.train.fusion_ops: rmsnorm_matmul, qkv_attention` (Hints for the Rust-native tensor graph fuser derived from Burn patterns).
    *   `grim.rocm.wavefront_size`, `grim.rocm.xnack_enabled`, `grim.rocm.kv_layout_optimized` (Existing inference tags, maintained for compatibility).

### Step 2: Implement Burn-Inspired Tensor Graph Fuser in Rust
*   **File**: New crate `crates/grim-tensor-graph/` or extension to `grim-backend-rocm/src/lib.rs`
*   **Action**: Develop a native Rust tensor graph representation that can parse the model architecture (e.g., Llama, Mistral) and identify fusable operation sequences. 
    *   This graph will not execute directly but will serve as an intermediate representation (IR) to configure `rocblas_gemm_ex` dispatch tables or generate/customize HIP kernel launch parameters.
    *   Ensure the IR is backend-agnostic at the high level, while allowing ROCm-specific optimizations (wavefront size, memory alignment) to be applied during the lowering phase to executable tensor operations.

### Step 3: Unsloth-Inspired Quantization-Aware Materialization for Training
*   **File**: `crates/grim-engine/src/lib.rs`, `crates/grim-models/transformer/src/lora.rs`
*   **Action**: Update the engine's forward and backward pass to support mixed-precision materialization:
    *   When a `.grim` or GGUF model is loaded with training enabled, weight materialization (`WeightSource::root()`) should map quantized tensors (FP4/FP8/Q4_K) into a hybrid buffer structure: storing the base quantized weights in VRAM/RAM but projecting them to BF16 only at the moment of matrix multiplication or gradient computation.
    *   Integrate dequantization kernels that are optimized for RDNA4, ensuring that the overhead of decoding FP4/FP8 weights does not bottleneck the training loop's throughput.

### Step 4: Oxidizer CLI Upgrades for Training Artifacts
*   **File**: `crates/grim-oxidizer/src/cli.rs`, `crates/grim-oxide/` (new or extended)
*   **Action**: Add new subcommands to the Grim Oxidizer:
    *   `grim oxide prepare --train --format bf16|fp8|fp4 <input.gguf|safetensors> --output model.grim`: Prepares a `.grim` file optimized for downstream training workflows, embedding fusion metadata and dequantization kernels.
    *   `grim oxide fuse --rocm <input.grim> --output fused.grim`: Analyzes the model graph and bakes in ROCm-specific kernel fusion configurations (RMSNorm+MatMul fusing hints).

---

## 5. Summary of Benefits & Turbocharged `.grim` Format Capabilities

By integrating **Burn-inspired tensor graph concepts** and **Unsloth-style kernel fusion and quantization-aware training patterns**, the Grim Oxidizer and the `.grim` format will deliver:

1.  **Next-Gen Inference Performance**: Pre-fused operation graphs and ROCm-aligned memory layouts for `rocblas_gemm_ex` will push RDNA4 throughput well beyond standard GGUF baselines, reliably exceeding the >80 tokens/sec decode target on RX 9070 XT hardware.
2.  **Native ROCm Training Readiness**: The `.grim` format and Oxidizer will enable grim to support advanced fine-tuning workflows (LoRA/QLoRA) with FP4, FP8, and BF16 materialization strategies, making grim a true end-to-end inference *and* training platform for AMD hardware.
3.  **Memory Bandwidth Optimization**: By adopting fusion strategies that avoid writing intermediate tensor states to VRAM, training and large-batch inference will see dramatic reductions in memory bandwidth pressure, maximizing the compute density of RDNA4 architectures.
4.  **Rust-Native Ecosystem Alignment**: Leveraging principles from Burn ensures grim's tensor orchestration remains purely native Rust, avoiding reliance on PyTorch/C++ bridges for model loading or graph optimization, resulting in a highly secure, performant, and maintainable ML stack.

---

## Implementation Verification Checklist

- [ ] **Burn Pattern Adoption**: Define a native Rust computation graph IR within `grim-tensor-graph` (or similar) to identify fusable operation sequences (MatMul + Norm + Activation).
- [ ] **Oxidizer Metadata Expansion**: Update `gguf.rs` and `.grim` parsers to accept and store `grim.train.*` and `grim.rocm.fusion_ops` metadata tags.
- [ ] **ROCm Kernel Fusion Integration**: Implement or expose fused RMSNorm+MatMul and QKV+Attention HIP kernel configurations within `grim-backend-rocm`.
- [ ] **Training Format Materialization Pipeline**: Ensure the engine's weight materialization (`WeightSource::root()`) can correctly map FP4/FP8/Q4_K quantized states to BF16 for gradient computations during training passes.
- [ ] **Oxidizer CLI Training Subcommands**: Implement `grim oxide prepare --train` and `grim oxide fuse --rocm` workflows to generate `.grim` artifacts explicitly optimized for ROCm-native training and inference.
