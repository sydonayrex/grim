# Implementation Plan: Grim Oxidizer v2 & Unsloth-Inspired Native ROCm Training Capabilities

## Overview

This implementation plan provides specific, actionable steps to fix the remaining items for Grim Oxidizer v2 (.grim format development with Burn-inspired tensor graph fusion) and Unsloth-inspired native ROCm training capabilities. The plan includes exact file modifications, code patterns, and integration points required to complete these features.

---

## Phase 1: Grim Oxidizer v2 & .grim Format Development (Burn-Inspired Tensor Graph Fusion)

### 1.1 Define Native Rust Computation Graph IR (`crates/grim-tensor-graph/src/lib.rs`)

**Objective**: Create the `grim-tensor-graph` crate to represent models as computation graphs that can identify fusable operation sequences (e.g., MatMul + Norm + Activation).

**Implementation Steps**:

1. **Create `crates/grim-tensor-graph/Cargo.toml`**:
```toml
[package]
name = "grim-tensor-graph"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
grim-core = { path = "../grim-core" }
grim-tensor = { path = "../grim-tensor" }
thiserror = "1"
```

2. **Define the Computation Graph IR in `crates/grim-tensor-graph/src/ir.rs`**:
```rust
//! Computation Graph Intermediate Representation for Grim model fusion.

use std::collections::{HashMap, VecDeque};
use grim_tensor::{ArithType, Shape};

/// Operation types that can be fused in the ROCm backend.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OpType {
    MatMul,
    RmsNorm,
    Silu,
    Gelu,
    QkvProjection,
    AttentionScore,
    Linear,
}

/// A single node in the computation graph.
#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: usize,
    pub op_type: OpType,
    pub input_tensors: Vec<String>,
    pub output_tensor: String,
    pub shape: Option<Shape>,
    pub dtype: ArithType,
}

/// The computation graph representing a model or layer.
#[derive(Debug, Clone)]
pub struct ComputationGraph {
    pub nodes: Vec<GraphNode>,
    pub entry_points: Vec<String>,
    pub fusion_candidates: Vec<FusionSequence>,
}

#[derive(Debug, Clone)]
pub struct FusionSequence {
    pub ops: Vec<OpType>,
    pub target_backend_op: String, // e.g., "fused_rmsnorm_matmul_rocm"
}

impl ComputationGraph {
    /// Identify fusable operation sequences from the graph.
    pub fn identify_fusion_sequences(&mut self) {
        self.fusion_candidates.clear();
        
        let mut current_sequence: Vec<OpType> = vec![];
        
        for node in &self.nodes {
            match node.op_type {
                OpType::RmsNorm => {
                    current_sequence.push(OpType::RmsNorm);
                }
                OpType::MatMul | OpType::Linear if matches!(current_sequence.last(), Some(&OpType::RmsNorm)) => {
                    current_sequence.push(node.op_type.clone());
                    self.fusion_candidates.push(FusionSequence {
                        ops: current_sequence.clone(),
                        target_backend_op: "fused_rmsnorm_matmul_rocm".to_string(),
                    });
                    current_sequence.clear();
                }
                OpType::QkvProjection => {
                    current_sequence.push(OpType::QkvProjection);
                }
                OpType::AttentionScore if matches!(current_sequence.last(), Some(&OpType::QkvProjection)) => {
                    current_sequence.push(OpType::AttentionScore);
                    self.fusion_candidates.push(FusionSequence {
                        ops: current_sequence.clone(),
                        target_backend_op: "fused_qkv_attention_rocm".to_string(),
                    });
                    current_sequence.clear();
                }
                _ => {
                    current_sequence.clear();
                }
            }
        }
    }
}
```

3. **Expose the IR in `crates/grim-tensor-graph/src/lib.rs`**:
```rust
pub mod ir;
pub use ir::{ComputationGraph, GraphNode, OpType, FusionSequence};

pub fn build_graph_from_model_config(model_type: &str, layer_defs: &[LayerDef]) -> ComputationGraph {
    // Parse model configuration and build computation graph IR
    let mut graph = ComputationGraph {
        nodes: vec![],
        entry_points: vec![],
        fusion_candidates: vec![],
    };
    
    // Identify fusable sequences
    graph.identify_fusion_sequences();
    graph
}
```

### 1.2 Oxidizer Metadata Expansion (`crates/grim-format/src/gguf.rs` and `crates/grim-format/src/tprov.rs`)

**Objective**: Update GGUF parsing to accept and store a new namespace of metadata tags:
- `grim.train.quant_mode: fp4|fp8|bf16`
- `grim.train.fusion_ops: rmsnorm_matmul, qkv_attention`

**Implementation Steps**:

1. **Verify existing implementation in `crates/grim-format/src/gguf.rs`** (Lines 261-311):
The types `GrimTrainQuantMode` and `GrimFusionOp` are already implemented:
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrimTrainQuantMode {
    Fp4,
    Nf4,
    Fp8,
    Bf16,
}

impl GrimTrainQuantMode {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "fp4" => Some(Self::Fp4),
            "nf4" => Some(Self::Nf4),
            "fp8" => Some(Self::Fp8),
            "bf16" => Some(Self::Bf16),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fp4 => "fp4",
            Self::Nf4 => "nf4",
            Self::Fp8 => "fp8",
            Self::Bf16 => "bf16",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GrimFusionOp {
    RmsNormMatMul,
    QkvAttention,
}

impl GrimFusionOp {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "rmsnorm_matmul" => Some(Self::RmsNormMatMul),
            "qkv_attention" => Some(Self::QkvAttention),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::RmsNormMatMul => "rmsnorm_matmul",
            Self::QkvAttention => "qkv_attention",
        }
    }
}
```

2. **Verify metadata parsing in `crates/grim-format/src/gguf.rs`** (Lines 406-417):
The metadata is already being parsed:
```rust
let train_quant_mode = metadata
    .get("grim.train.quant_mode")
    .and_then(|v| v.as_str())
    .and_then(GrimTrainQuantMode::from_str);
let train_fusion_ops = metadata
    .get("grim.train.fusion_ops")
    .and_then(read_grim_fusion_ops)
    .unwrap_or_default();
let rocm_fusion_ops = metadata
    .get("grim.rocm.fusion_ops")
    .and_then(read_grim_fusion_ops)
    .unwrap_or_default();
```

3. **Update `crates/grim-format/src/tprov.rs`** to expose the training metadata:
Add a method to `GgufProvider` to check if training mode is enabled:
```rust
impl GgufProvider {
    // ... existing methods ...

    /// Check if this file has training quantization mode specified.
    pub fn train_quant_mode(&self) -> Option<GrimTrainQuantMode> {
        self.grim.train_quant_mode
    }

    /// Get ROCm fusion operations for this model/graph.
    pub fn rocm_fusion_ops(&self) -> &[GrimFusionOp] {
        &self.grim.rocm_fusion_ops
    }

    /// Check if RMSNorm+MatMul fusion is requested.
    pub fn has_rmsnorm_matmul_fusion(&self) -> bool {
        self.grim.train_fusion_ops.contains(&GrimFusionOp::RmsNormMatMul) ||
        self.grim.rocm_fusion_ops.contains(&GrimFusionOp::RmsNormMatMul)
    }

    /// Check if QKV+Attention fusion is requested.
    pub fn has_qkv_attention_fusion(&self) -> bool {
        self.grim.train_fusion_ops.contains(&GrimFusionOp::QkvAttention) ||
        self.grim.rocm_fusion_ops.contains(&GrimFusionOp::QkvAttention)
    }
}
```

### 1.3 Oxidizer CLI Training Subcommands (`crates/grim-oxidizer/src/cli.rs`)

**Objective**: Implement the workflows for `grim oxide prepare --train --format bf16|fp8|fp4 <input> --output model.grim` and `grim oxide fuse --rocm <input.grim> --output fused.grim`.

**Implementation Steps**:

1. **Create `crates/grim-oxidizer/Cargo.toml`**:
```toml
[package]
name = "grim-oxidizer"
version.workspace = true
edition.workspace = true
license.workspace = true

[[bin]]
name = "grim-oxide"
path = "src/main.rs"

[dependencies]
grim-format = { path = "../grim-format" }
grim-tensor-graph = { path = "../grim-tensor-graph" }
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

2. **Implement CLI in `crates/grim-oxidizer/src/cli.rs`**:
```rust
use clap::{Parser, Subcommand};
use grim_format::gguf::{GrimTrainQuantMode, GrimFusionOp, GrimmMetadataBuilder};

#[derive(Parser)]
#[command(name = "grim-oxide", about = "Grim Oxidizer: .grim format optimization and fusion")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Prepare a .grim file for training workflows
    Prepare {
        /// Input GGUF or safetensors file
        input: String,
        /// Output .grim file
        #[arg(short, long)]
        output: String,
        /// Training quantization mode
        #[arg(long, value_enum)]
        format: TrainFormat,
    },
    /// Fuse ROCm operations for a .grim file
    FuseRocm {
        /// Input .grim file
        input: String,
        /// Output fused .grim file
        #[arg(short, long)]
        output: String,
    },
}

#[derive(clap::ValueEnum, Clone, Debug)]
enum TrainFormat {
    Bf16,
    Fp8,
    Fp4,
}

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Prepare { input, output, format } => {
            exec_prepare(&input, &output, format)?;
        }
        Commands::FuseRocm { input, output } => {
            exec_fuse_rocm(&input, &output)?;
        }
    }
    
    Ok(())
}

fn exec_prepare(input: &str, output: &str, format: TrainFormat) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("[grim-oxide] Preparing .grim file for training...");
    eprintln!("  Input: {}", input);
    eprintln!("  Output: {}", output);
    eprintln!("  Format: {:?}", format);
    
    // Load GGUF/safetensors and create grim metadata
    let mut builder = GrimmMetadataBuilder::new();
    
    let quant_mode = match format {
        TrainFormat::Bf16 => GrimTrainQuantMode::Bf16,
        TrainFormat::Fp8 => GrimTrainQuantMode::Fp8,
        TrainFormat::Fp4 => GrimTrainQuantMode::Fp4,
    };
    
    builder.with_train_quant_mode(quant_mode);
    builder.with_fusion_ops(vec![GrimFusionOp::RmsNormMatMul, GrimFusionOp::QkvAttention]);
    
    // Generate .grim file with metadata
    let grim_metadata = builder.build();
    // Save to output path with .grim extension
    
    eprintln!("[grim-oxide] Preparation complete: {}", output);
    Ok(())
}

fn exec_fuse_rocm(input: &str, output: &str) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("[grim-oxide] Fusing ROCm operations...");
    eprintln!("  Input: {}", input);
    eprintln!("  Output: {}", output);
    
    // Load .grim file, analyze graph, bake in fusion configs
    // Generate fused.grim with rocm_fusion_ops metadata
    
    eprintln!("[grim-oxide] Fusion complete: {}", output);
    Ok(())
}
```

---

## Phase 2: Unsloth-Inspired Native ROCm Training Capabilities

### 2.1 ROCm Kernel Fusion Integration (`crates/grim-backend-rocm/src/fusion.rs`)

**Objective**: Implement or expose fused RMSNorm+MatMul and QKV+Attention HIP kernel configurations within `grim-backend-rocm`.

**Implementation Steps**:

1. **Create fusion module in `crates/grim-backend-rocm/src/fusion.rs`**:
```rust
//! ROCm kernel fusion for Unsloth-inspired performance optimizations.

use grim_tensor::{Shape, ArithType};

/// Fusion configuration for RMSNorm + MatMul operation.
#[derive(Debug, Clone)]
pub struct RmsNormMatMulFusionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub wavefront_size: u32,
    pub lds_size: u32,
}

/// Fusion configuration for QKV Projection + Attention operation.
#[derive(Debug, Clone)]
pub struct QkvAttentionFusionConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub wavefront_size: u32,
}

impl RmsNormMatMulFusionConfig {
    /// Generate HIP kernel launch parameters for fused RMSNorm+MatMul.
    pub fn hip_launch_params(&self) -> HipKernelLaunch {
        let block_dim = if self.wavefront_size == 32 { 128 } else { 256 };
        let grid_dims_x = (self.intermediate_size + block_dim - 1) / block_dim;
        
        HipKernelLaunch {
            grid_dim: hipDim3 { x: grid_dims_x as u32, y: 1, z: 1 },
            block_dim: hipDim3 { x: block_dim as u32, y: 1, z: 1 },
            shared_mem_bytes: self.lds_size.min(65536) as usize,
        }
    }
}

impl QkvAttentionFusionConfig {
    /// Generate HIP kernel launch parameters for fused QKV+Attention.
    pub fn hip_launch_params(&self) -> HipKernelLaunch {
        let block_dim = if self.wavefront_size == 32 { 128 } else { 256 };
        let grid_dims_x = (self.num_heads + block_dim - 1) / block_dim;
        
        HipKernelLaunch {
            grid_dim: hipDim3 { x: grid_dims_x as u32, y: 1, z: 1 },
            block_dim: hipDim3 { x: block_dim as u34, y: 1, z: 1 },
            shared_mem_bytes: (self.head_dim * 4).min(32768) as usize,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct hipDim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[derive(Debug, Clone)]
pub struct HipKernelLaunch {
    pub grid_dim: hipDim3,
    pub block_dim: hipDim3,
    pub shared_mem_bytes: usize,
}
```

2. **Expose fusion APIs in `crates/grim-backend-rocm/src/lib.rs`**:
Add imports and re-exports for the fusion module:
```rust
pub mod fusion;
pub use fusion::{RmsNormMatMulFusionConfig, QkvAttentionFusionConfig, HipKernelLaunch};
```

### 2.2 Training Format Materialization Pipeline (`crates/grim-nn/src/varbuilder.rs`)

**Objective**: Ensure the engine's weight materialization (`WeightSource::root()`) can correctly map quantized states (FP4/FP8/Q4_K) to BF16 for gradient computations during training passes without bottlenecking throughput.

**Implementation Steps**:

1. **Update `crates/grim-nn/src/varbuilder.rs`** to support mixed-precision materialization:
```rust
use grim_tensor::dtype::{DType, Device, QuantProvenance, KQuantScheme};
// ... existing imports ...

impl<'a> WeightSource<'a> {
    // ... existing methods ...

    /// Materialize a tensor with training-aware dequantization to BF16.
    pub fn get_for_training(&self, shape: impl Into<Shape>, leaf: &str) -> Result<Tensor> {
        let shape = shape.into();
        let name = self.full_name(leaf);
        let raw = self.tensors.get(&name)?;

        if raw.shape != shape.dims() {
            return Err(Error::ShapeMismatch {
                expected: shape.dims().to_vec(),
                got: raw.shape.clone(),
            });
        }
        
        let (dtype, provenance) = match self.tensors.meta(&name) {
            Ok(m) => (m.dtype, m.provenance),
            Err(_) => (self.default_dtype.clone(), self.default_provenance.clone()),
        };

        // For training, map quantized states to BF16/FP32 for gradient computation
        let target_dtype = if dtype.is_quantized() || matches!(dtype.storage, grim_tensor::Storage::KQuant(_)) {
            // Materialize to BF16 for training/gradient computations
            DType {
                arith: grim_tensor::ArithType::BF16,
                storage: grim_tensor::Storage::Native,
            }
        } else {
            dtype.clone()
        };

        tensor_from_raw_for_training(raw, shape, target_dtype, provenance, self.device.clone())
    }
}

fn tensor_from_raw_for_training(
    raw: RawTensor,
    shape: Shape,
    dtype: DType,
    provenance: QuantProvenance,
    device: Device,
) -> Result<Tensor> {
    // Handle quantized to BF16/FP32 materialization for training
    if let grim_tensor::Storage::KQuant(scheme) = &dtype.storage {
        // Dequantize K-quant format to F32/BF16
        // This would invoke ROCm-optimized dequant kernels in phase 3
        let f32s = dequant_k_quant_to_f32(&raw.bytes, *scheme, raw.shape.iter().product::<usize>())?;
        let t = match device {
            Device::Cpu => grim_backend_cpu::cpu_tensor(f32s, shape),
            _ => return Err(Error::Unimplemented(
                "Training materialization to non-CPU devices not yet implemented".into()
            )),
        };
        
        // Apply quant provenance for gradient tracking
        Ok(t.with_provenance(provenance))
    } else if dtype.arith == grim_tensor::ArithType::F32 || 
              (dtype.storage == grim_tensor::Storage::Native && matches!(dtype.arith, grim_tensor::ArithType::BF16 | grim_tensor::ArithType::F32)) {
        let f32s = bytes_to_f32(&raw.bytes, raw.shape.iter().product::<usize>())?;
        let _ = CpuDevice::new();
        Ok(grim_backend_cpu::cpu_tensor(f32s, shape).with_provenance(provenance))
    } else {
        Err(Error::Unimplemented(
            "Training materialization only supports F32/BF16/Native or KQuant formats".into()
        ))
    }
}

fn dequant_k_quant_to_f32(bytes: &[u8], scheme: KQuantScheme, n: usize) -> Result<Vec<f32>> {
    // Implement ROCm-optimized dequantization for Q4_K, Q5_K, Q8_0, etc.
    // This is a placeholder for the actual dequant kernel implementation
    let mut out = vec![0.0f32; n];
    // Dequant logic would go here based on KQuantScheme
    Ok(out)
}
```

### 2.3 Native ROCm LoRA / QLoRA Support (`crates/grim-engine/src/lib.rs` and `crates/grim-models/transformer/src/lora.rs`)

**Objective**: Leverage the existing `grim-engine` adapter registry to generate `.grim` artifacts where LoRA rank decompositions are pre-aligned with ROCm memory layouts.

**Implementation Steps**:

1. **Update `crates/grim-models/transformer/src/lora.rs`** to include ROCm layout alignment:
```rust
use grim_tensor::Device;

#[derive(Debug, Clone)]
pub struct LoRAWeights {
    pub down_proj: Tensor,   // Rank decomposition matrix A
    pub up_proj: Tensor,     // Rank decomposition matrix B
    pub alpha_scale: f32,
}

impl LoRAWeights {
    /// Load LoRA weights with ROCm memory layout pre-alignment.
    pub fn load_for_rocm(ws: &WeightSource<'_>, prefix: &str, rank: usize) -> Result<Self> {
        // For ROCm, ensure matrices are row-major or block-tiled based on wavefront
        let down_proj = ws.pp(prefix).pp("lora_A").get((rank, ws.get_hidden_size()), "weight")?;
        let up_proj = ws.pp(prefix).pp("lora_B").get((ws.get_hidden_size(), rank), "weight")?;
        
        // Align to ROCm memory layout if on Rocm device
        let (down_proj, up_proj) = if down_proj.device().is_rocm() {
            let down_aligned = align_tensor_for_rocm_gemm(&down_proj)?;
            let up_aligned = align_tensor_for_rocm_gemm(&up_proj)?;
            (down_aligned, up_aligned)
        } else {
            (down_proj, up_proj)
        };

        let alpha_scale = ws.pp(prefix).get(1, "lora_alpha")?.to_vec_f32()?[0];

        Ok(LoRAWeights {
            down_proj,
            up_proj,
            alpha_scale,
        })
    }
}

/// Align tensor for ROCm GEMM (rocblas_gemm_ex) memory layout requirements.
fn align_tensor_for_rocm_gemm(tensor: &Tensor) -> Result<Tensor> {
    // Ensure 32-byte or 64-byte alignment for optimal HIP memory transfers
    // and vectorized matrix core instructions
    let aligned_shape = tensor.shape().align_to_block_size(32);
    // ... implementation of layout re-alignment ...
    Ok(tensor.clone())
}
```

2. **Integrate LoRA weight loading with engine's Rocm device context in `crates/grim-engine/src/lib.rs`**:
Ensure that when `resolve_adapters` is called, the adapter weights are loaded using the ROCm-optimized path if the device is `Device::Rocm`.

---

## Implementation Verification Checklist

- [ ] **Computation Graph IR Crate**: `crates/grim-tensor-graph/` exists with `ir.rs` containing `ComputationGraph`, `GraphNode`, `OpType`, and `FusionSequence`.
- [ ] **Metadata Expansion in GGUF**: `crates/grim-format/src/gguf.rs` correctly parses `grim.train.quant_mode` and `grim.train.fusion_ops` / `grim.rocm.fusion_ops`.
- [ ] **TProv Training Metadata Accessors**: `GgufProvider` exposes `train_quant_mode()`, `rocm_fusion_ops()`, `has_rmsnorm_matmul_fusion()`, and `has_qkv_attention_fusion()`.
- [ ] **Oxidizer CLI Subcommands**: `grim oxide prepare --train --format bf16|fp8|fp4 <input> --output model.grim` and `grim oxide fuse --rocm <input.grim> --output fused.grim` are implemented in `crates/grim-oxidizer/src/cli.rs`.
- [ ] **ROCm Kernel Fusion Module**: `crates/grim-backend-rocm/src/fusion.rs` exposes `RmsNormMatMulFusionConfig` and `QkvAttentionFusionConfig` with HIP launch parameters.
- [ ] **Training Format Materialization**: `WeightSource::get_for_training()` in `crates/grim-nn/src/varbuilder.rs` correctly maps quantized states (FP4/FP8/Q4_K) to BF16 for gradient computations.
- [ ] **ROCm LoRA Pre-alignment**: `LoRAWeights::load_for_rocm()` in `crates/grim-models/transformer/src/lora.rs` includes ROCm memory layout alignment via `align_tensor_for_rocm_gemm()`.

---

## Execution Order Recommendations

1. **Step 1: Oxidizer Metadata Expansion** (`gguf.rs`, `tprov.rs`) - This enables the `.grim` format to carry training and fusion metadata, forming the foundation for all subsequent work.
2. **Step 2: Computation Graph IR Crate** (`grim-tensor-graph/`) - Define the IR before implementing the Oxidizer CLI, as the CLI will need to generate and validate fusion sequences.
3. **Step 3: Oxidizer CLI Subcommands** (`grim-oxidizer/src/cli.rs`) - Implement `prepare` and `fuse` workflows using the metadata expansion and graph IR.
4. **Step 4: ROCm Kernel Fusion Integration** (`grim-backend-rocm/src/fusion.rs`) - Add fusion configurations that the Oxidizer can reference when baking `.grim` artifacts.
5. **Step 5: Training Format Materialization Pipeline** (`varbuilder.rs`) - Enable `WeightSource::get_for_training()` to correctly map quantized states to BF16 for training passes.
6. **Step 6: Native ROCm LoRA / QLoRA Support** (`lora.rs`, `grim-engine/src/lib.rs`) - Leverage the adapter registry and ROCm layout alignment for fine-tuning workflows.

---

## Phase 3: Grim's Garage Web UI Implementation (Local-First Training Dashboard)

### Overview
**Grim's Garage** is a local-first, ROCm-optimized Web UI for fine-tuning and training LLMs using the Grim engine. It abstracts away complex CLI commands into an intuitive interface with dropdowns and toggles—similar to Unsloth Studio—but natively leverages `.grim` metadata, ROCm kernel fusion flags, and AMD-specific hardware telemetry. 

The web UI follows **CVKG design principles**:
- Token-driven design using OKLCH color space (Light/Dark theme variants)
- Modular component architecture (button_primary_background, input_border_focus, card_surface_disabled)
- Accessible form inputs: dropdowns, numeric fields, toggle switches
- High-end visual design for telemetry dashboards (VRAM usage line charts, loss scatter plots, tokens/sec KPI headers)

### 3.1 Technology Stack & Architecture

#### Frontend
- **Framework**: React + Vite + TypeScript
- **UI Component Library**: CVKG-designed `shadcn/ui` or Radix UI components using OKLCH token-driven theming (surface, on_surface, primary, on_primary, error).
- **State & API**: TanStack Query for fetching model/dataset lists and managing training job states.
- **Real-time Updates**: WebSocket or Server-Sent Events (SSE) for live VRAM metrics, loss curves, and token throughput.

#### Backend / App Layer
- **Desktop Wrapper / Web Serve**: Axum acting as a static file server for the React build, with REST/WS APIs for training orchestration. Alternatively, Tauri v2 (Rust backend + Web frontend) to provide native file system access and local ROCm telemetry without browser sandbox restrictions.

#### Training Orchestration
- The UI does not run training directly; it fires off `grim-train` or `grim-finetune` CLI/library invocations via a local JSON-RPC job queue, updating the UI through job status SSE streams.

### 3.2 Core UI Components (CVKG-Inspired Modular Architecture)

#### 3.2.1 Model Selector Panel
**Purpose**: Allow users to select the base model for training/fine-tuning.

- **Dropdown/Selector Component** (`input_dropdown_model_select`): "Select Base Model"
  - Populated via `/api/models` endpoint which scans local directories (e.g., `~/.grim/models/`, `./models/`) for `.gguf` and `.grim` files.
- **Model Metadata Card Component** (`card_metadata_model_display`):
  - Format detected: `GGUF` or `.grim`
  - Quantization type: e.g., `Q4_K`, `BF16`, `FP8`
  - If `.grim`: displays training metadata (`grim.train.quant_mode`, supported fusion ops like `rmsnorm_matmul`, `qkv_attention`).
  - VRAM Estimation: Based on model size and selected training type (Full vs LoRA/QLoRA).

#### 3.2.2 Training Type Selector Dropdown
**Purpose**: Define the fine-tuning approach.

- **Dropdown Component** (`input_dropdown_training_mode`): "Training Mode"
  - Options: `LoRA Fine-Tune`, `QLoRA Fine-Tune (Quantized)`, `BF16 Full Fine-Tune`.
- **Conditional Display Panel**: 
  - If `QLoRA Selected`: show quantization format dropdown.

#### 3.2.3 Quantization Format Dropdown (For QLoRA)
**Purpose**: Select the target quantization for weight materialization during training.

- **Dropdown Component** (`input_dropdown_quant_format`): "Quantization Format"
  - Options derived from `.grim` metadata or GGUF parser: `FP4 / NF4`, `FP8`, `BF16`.
  - Auto-suggest based on model's native format and ROCm RDNA4 compatibility (prioritizing BF16 for gradient stability, FP8/FP4 for VRAM saving).

#### 3.2.4 Dataset Management Panel
**Purpose**: Select or upload the training dataset.

- **Dataset Selector Dropdown** (`input_dropdown_dataset_select`): Lists JSONL/Parquet files in the `./datasets/` directory.
- **Upload Button Component** (`button_upload_dataset`): Drag-and-drop zone for `.jsonl`, `.json`, or `.parquet` files.
- **Preview Toggle Switch** (`toggle_switch_dataset_preview`): Show a sample of 3 rows from the selected dataset to verify formatting (instruction/response or chat format).

#### 3.2.5 Hyperparameter Controls (Simplified)
**Purpose**: Provide essential training knobs without overwhelming the user.

- **LoRA Rank `r` Dropdown** (`input_dropdown_lora_rank`): Options: `8`, `16`, `32`, `64`.
- **Learning Rate Slider/Dropdown** (`slider_numeric_learning_rate` or `input_number_learning_rate`): Pre-set common values or a custom input (`1e-5`, `2e-5`, `5e-5`).
- **Epochs / Max Steps Input** (`input_number_epochs_steps`): Numeric fields for training duration.

#### 3.2.6 ROCm Optimization Toggles (Grim-Specific)
**Purpose**: Expose the Burn-inspired and Unsloth-style kernel fusion features directly in the UI.

- **Toggle Switch Components** (`toggle_switch_rocm_fusion_rmsnorm_matmul`, `toggle_switch_rocm_fusion_qkv_attention`):
  - [ ] Enable `RMSNorm + MatMul` Fusion (ROCm HIP Kernel)
  - [ ] Enable `QKV Projection + Attention` Fusion
- **Checkbox/Toggle Components**:
  - [ ] Auto-detect Wavefront Size (W32 for RDNA3/4, W64 for CDNA)
  - [ ] Enable XNACK-aware Memory Allocation (if supported by GPU profile in `.grim` metadata)

#### 3.2.7 Training Monitor & Telemetry Dashboard (CVKG High-End Visual Design)
**Purpose**: Provide real-time feedback during the training job using CVKG token-driven colors and accessible chart components.

- **VRAM Usage Graph Component** (`chart_line_vram_usage`): Line chart showing VRAM consumption over time, queried via ROCm FFI (`hipDeviceGetAttribute`) or parsing `rocm-smi --json`. Uses primary/on_primary OKLCH colors for the line, surface/on_surface background.
- **Throughput Metrics KPI Component** (`kpi_throughput_tokens_sec`): Tokens/sec or Steps/sec displayed in the top header using heading typography tokens and success/error semantic color aliases.
- **Loss Curve Chart Component** (`chart_scatter_loss_per_step`): Scatter/line plot of training loss per step, updated via SSE from the `grim-train` process.
- **Log Console Component** (`terminal_log_console_training`): A scrollable terminal-like view showing `grim-finetune` stdout/stderr, using mono typography and surface/on_surface contrast.

#### 3.2.8 Start Training & Job History
- **Primary CTA Button Component** (`button_primary_start_training`): "Start Training Session" (disabled if model or dataset is not selected; uses disabled component tokens: `button_background_disabled`, `on_button_disabled`).
- **Job History Panel Component** (`list_job_history_training_sessions`): Lists past training sessions with status (`Completed`, `Failed`, `Running`) and paths to saved `.grimo` (fused/quantized output) checkpoints.

### 3.3 Backend API Endpoints (Axum / Tauri Rust Layer)

To support the UI, the Rust backend must expose the following APIs:

#### 3.3.1 Model & Dataset Discovery
- `GET /api/models`: Returns a JSON list of available models with metadata.
  ```json
  {
    "models": [
      {
        "id": "llama3-8b-q4k.grim",
        "path": "./models/llama3-8b-q4k.grim",
        "format": "grim",
        "train_quant_mode": "bf16",
        "fusion_ops": ["rmsnorm_matmul", "qkv_attention"],
        "estimated_vram_gb": 12.5
      }
    ]
  }
  ```
- `GET /api/datasets`: Returns list of available training datasets in the local `datasets/` directory.

#### 3.3.2 Training Job Orchestration
- `POST /api/train/start`: Accepts a JSON payload with the selected configuration (model, dataset, lora_rank, quant_format, fusion_toggles). Starts the `grim-train --config ...` process and returns a `job_id`.
- `GET /api/train/status/{job_id}`: Returns current job state (`pending`, `running`, `completed`, `failed`) and logs.
- `WS /ws/metrics/{job_id}` or `SSE /sse/metrics/{job_id}`: WebSocket/SSE endpoint streaming JSON metrics (VRAM usage, loss, tokens/sec) for the live dashboard.

#### 3.3.3 ROCm Hardware Telemetry
- `GET /api/rocm/devices`: Returns attached AMD GPUs via HIP FFI probe (`hipGetDeviceCount`, `hipDeviceGetAttribute`).
  - Includes: GPU name, VRAM capacity, active XNACK status, wavefront size (W32/W64).

### 3.4 Implementation Steps for the Web UI

**Step 1**: Set up Tauri + React/Vite project structure or Axum-static-serve project in `crates/grim-studio/`. Scaffold frontend with React + Vite + Tailwind using CVKG token-driven OKLCH color themes (light/dark variants).

**Step 2**: Implement Backend API Layer (Rust) exposing Axum routes or Tauri commands for `/api/models`, `/api/datasets`, `/api/train/start`. Integrate with `grim-format` to parse `.grim` and GGUF files for metadata (quant mode, fusion ops). Integrate with `grim-backend-rocm` to probe ROCm hardware stats.

**Step 3**: Build Frontend UI Components using CVKG modular component architecture: Model Selector Dropdown, Training Type Dropdown, Quantization Format Dropdown, ROCm Optimization Toggle Switches. Implement the Training Monitor dashboard with chart components (using `Recharts` or `Chart.js`) for VRAM and Loss metrics, applying OKLCH semantic color aliases (`primary`, `on_primary`, `surface`, `error`).

**Step 4**: Integrate Real-time Metrics via WebSockets/SSE. Modify the `grim-train` execution pipeline to emit progress events (loss, step number, VRAM snapshots). Serve these via SSE or a local WebSocket endpoint so the React frontend can update the dashboard live without page refreshes.

**Step 5**: Test with Local `.grim` and GGUF Artifacts. Validate that selecting a `.grim` file with `grim.train.fusion_ops: rmsnorm_matmul` correctly pre-enables the ROCm fusion toggles in the UI. Ensure QLoRA mode correctly maps to FP4/FP8/BF16 materialization paths in the backend training runner.

### 3.5 Summary of Benefits for Grim's Garage Web UI

1. **Zero-CLI Training Workflow**: Users can fine-tune GGUF or `.grim` models entirely through a visual, dropdown-driven interface, removing the barrier to entry for researchers and developers who prefer GUI tools like Unsloth Studio.
2. **ROCm-Native Optimization Visibility**: The UI explicitly surfaces ROCm-specific toggles (RMSNorm+MatMul fusion, QKV Attention fusion, Wavefront auto-detection), ensuring users can leverage the Grim Oxidizer v2 capabilities without understanding the underlying HIP kernel parameters.
3. **Real-Time Hardware Telemetry**: Direct integration with AMD ROCm APIs to show live VRAM usage and device health, preventing out-of-memory errors during QLoRA or BF16 training sessions.
4. **CVKG-Standard High-End Visual Design & Accessibility**: Token-driven OKLCH theming ensures perceptually uniform colors across light/dark modes, accessible form inputs (dropdowns, toggles) using semantic component tokens, and a high-end visual telemetry dashboard for loss curves and VRAM metrics.
5. **Seamless .grim Format Integration**: The UI natively reads `.grim` metadata (`train_quant_mode`, `fusion_ops`) to pre-configure optimal training presets, making the advanced Unsloth-inspired ROCm tuning accessible with a few simple dropdown selections.
