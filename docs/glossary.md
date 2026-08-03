# Glossary

This document defines domain-specific terms used in Grim.

## A

### Adapter

In the context of LoRA/QLoRA, an adapter is a low-rank update module that modifies base model weights. Adapters are stored as pairs of matrices (A, B) with a scaling factor alpha.

See also: `AdapterHandle` in `grim-core/src/model.rs`.

### Attention (KV Attention)

Key-Value attention where queries attend to previously generated keys and values. In autoregressive language models, the KV cache stores past keys and values to avoid recomputation.

Related code: `grim-tensor/src/backend.rs` `qkv_attention` method.

## C

### Causal Language Model (CausalLm)

A language model that predicts the next token given all previous tokens. Causal masking ensures autoregressive behavior.

Trait: `grim-core::Model` → `CausalLm`

### Continuous Batching

A scheduling strategy where new requests are added to batches as existing requests complete, maintaining high GPU utilization.

Related code: `grim-scheduler`

## D

### Dequantization

The process of converting quantized weights back to floating-point for computation. Dequant kernels read packed quantized data and emit F32 values.

Related code: `grim-backend-cpu/src/dequant_gemm.rs`, `grim-quant`

### Diffusion Model

A model that iteratively denoises a latent representation to generate outputs (e.g., images). Uses UNet architecture with noise schedulers.

Trait: `grim-core::DiffusionModel`

## F

### FlashAttention

A fused attention kernel that computes attention in a numerically stable way while minimizing memory bandwidth. Returns `Err(Unimplemented)` for non-GPU backends.

Related code: `grim-tensor/src/backend.rs` `flash_attention` method.

### GGUF

GPT-Generated Unified Format — a binary weight format developed by llama.cpp. Contains tensor data, metadata, and tokenizer info.

Related code: `grim-format/src/gguf.rs`

## H

### HIP

Heterogeneous-compute Interface for Portability (ROCm's CUDA equivalent). Grim uses HIP for AMD GPU operations via rocBLAS.

Related code: `grim-backend-rocm`

## I

### IQ Quant

Importance-matrix-optimized quantization formats (IQ4_NL, IQ2XS, etc.) from EfficientQAT/GPTQ pipelines. Distinct from KQuant because they require different dequant kernels.

Related code: `grim-tensor/src/dtype.rs` `KQuantScheme`, `QuantProvenance::WithResiduals`

### K-Quant

Block quantization format where each 16-element block has a scale and 4-6 bits per weight. Compatible with llama.cpp.

Related code: `grim-tensor/src/dtype.rs` `KQuantScheme::Q4K`, etc.

## L

### LoRA (Low-Rank Adaptation)

A technique for fine-tuning large language models by injecting trainable low-rank matrices into attention layers. Grim supports QLoRA (quantized LoRA).

Related code: `grim-autograd`, `grim-core::AdapterHandle`

### LoRA Rank

The dimensionality of the low-rank adaptation matrices. Higher rank = more capacity but more trainable parameters.

Typical values: 4, 8, 16, 32, 64

## M

### MIGRATION, MTP, MIN-

MIN-1, MIN-3, MIN-4: Internal design document references for specific implementation milestones.

MTP (Multi-Token Prediction): Speculative decoding technique where a draft model predicts multiple tokens ahead.

WI-1, WI-3, WI-4, etc.: Work Item references in the architecture specification.

### Mamba

A state-space model architecture using selective scanning for linear-time sequence modeling. Uses SSM (Structured State Space Model).

Related code: `grim-models/mamba`

### Memory Spilling

Moving KV cache blocks from GPU VRAM to host RAM or NVMe storage when VRAM is full.

Related code: `grim-kvtransport`

## N

### NVLink / xGMI

High-bandwidth interconnect between GPUs. Grim uses this for efficient GPU-GPU communication when available.

### NVMe Spill

Persisting KV cache blocks to NVMe SSD when both GPU and RAM are exhausted.

## O

### Oxidizer

Grim's GGUF-to-.grim conversion tool. Performs importance-matrix calibration, EvoPress evolution, and ROCm kernel optimization.

CLI: `grim oxidizer`

## P

### P2P Access (Peer-to-Peer)

Direct GPU-to-GPU memory access without going through host. Required for multi-GPU setups.

Related code: `grim-backend-rocm/src/device/peer_access.rs`

### Prefix Cache

Cache optimization where requests with common prompts share KV state.

Related code: `grim-memory`

### QLoRA (Quantized LoRA)

LoRA training with quantized base model weights. Allows training adapter weights while keeping base model quantized.

## R

### RocBLAS

ROCm's BLAS library for matrix operations. Grim uses rocBLAS for GEMM operations on AMD GPUs.

### RoPE (Rotary Position Embedding)

Rotary position embedding that encodes position information via rotation matrices. Standard in transformer models.

### Speculative Decoding

Technique where a draft model generates multiple tokens ahead, then a verifier checks their correctness. Accepted tokens are immediately committed.

Related code: `grim-speculative`

## S

### Scythe Placement

C²PLR (Contention-Aware Placement for Linear layers) controller output specifying GPU ranks for a layer and partition ratios.

Related code: `grim-tensor/src/backend.rs` `ScythePlacement`, `grim-engine/src/scythe2.rs`

### SSM (State Space Model)

Structured State Space Model for sequence modeling. Used in Mamba architecture for linear-time inference.

### SwigLU

SiLU GLU — activation function used in MLP layers.

## T

### Tensor Parallelism

Splitting a model's weights across multiple GPUs. Grim implements this via the `all_reduce` operation.

### KV Quantization

Runtime compression of KV cache blocks using random-orthogonal rotation + Lloyd-Max scalar
quantization. Reduces KV memory footprint during serving, trading compute for memory.

Related code: `grim-kvquant`, `grim-memory/src/lib.rs` (`KvBlockPool` stores `compressor: Option<Arc<dyn KvCompressor>>`).

## V

### VRAM

Video RAM (GPU memory). Grim manages VRAM allocation with spillover to host memory when needed.

### ViT (Vision Transformer)

Transformer architecture for vision tasks. Grim provides CLIP-style vision encoders.

## W

### WGKS (Wavefront Group Kernel Split)

AMD GPU warp-level primitive for parallel operations.

### Worker

In the scheduler context, a worker is a unit of compute that processes requests. Grim's scheduler manages workers in a three-queue system.