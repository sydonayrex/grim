# Grim upgrade and remediation plan

> **For agentic workers:** use the executing-plans skill to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn grim-garage from a well-architected dashboard wired to a simulation engine into the fastest, most capable training system for ROCm GPUs, with a codec lineup that no Python framework matches.

**Architecture:** Fix the GPU dispatch blocker first (A.0 kernel collision), then connect the real model forward/backward pass to the job worker, then expose grim's existing codec infrastructure through the garage API. ROCm-specific wins (HIP graph capture, FP8 native on RDNA4, RCCL) land after the training loop is real and testable.

**Tech stack:** Rust 1.79+, HIP/HIPRTC, axum 0.7, tokio, grim-autograd, grim-quant, grim-format, grim-backend-rocm

## Format names

| Name | Type | bpw | Arch gate |
|------|------|-----|-----------|
| Crow | Q4K (GGML K-quant 4-bit) | 4.5 | RDNA2+ |
| Raven | FP8 E4M3 native | 8.0 | RDNA4, CDNA3 |
| Rook | MXFP4 emulated (dequant-in-tile to BF16) | 4.1 | RDNA2+ |
| Jay | MXFP4 block-16 (existing Fp4Block16) | 4.1 | RDNA2+ |
| Jackdaw | MXFP8 emulated (dequant-in-tile to BF16) | 8.0 | RDNA2+ |
| Magpie | MXFP8 block-16 (existing Fp8Block16) | 8.0 | RDNA2+ |

## Global constraints

- No Python runtime dependency anywhere in the training path
- A.0 must be fixed before any GPU kernel work is merged
- `QuantMode::Fp8` renamed to `Fp8Native`; RDNA2/3 gate stays intact
- `MxFp4Emulated` (Rook) and `MxFp8Emulated` (Jackdaw) are new variants allowed on RDNA2+
- Every new public fn needs a doc comment stating what it does and what it takes/returns
- No placeholder steps: every step contains real code or real commands

---

## Phase 0: GPU dispatch unblocked

**Why first:** nothing else in this plan can be tested on real hardware until HIPRTC can compile. A.0 is a confirmed hardware failure, not a theory.

---

### Task 0.1: Fix kernel collision in compute_kernel_source

**Files:**
- Modify: `crates/grim-backend-rocm/src/kernels/shared_device_fns.rs`
- Modify: `crates/grim-backend-rocm/src/kernels/source_asm.rs`
- Modify: `crates/grim-backend-rocm/src/kernels/mxfp_standalone.rs`
- Modify: `crates/grim-backend-rocm/src/kernels/fp8_standalone.rs`
- Test: `crates/grim-backend-rocm/tests/lib_internal_tests.rs`

**Interfaces:**
- Produces: `compute_kernel_source() -> String` that compiles without symbol redefinitions on gfx1036+

The four symbols that collide across the 21 concatenated modules are `fp16_to_float_device`, `fp8_e4m3_to_float_hip`, `mxfp4_to_float_hip`, and `dequant_q4k_element`. They are already defined once in `shared_device_fns.rs`. Every other module that also defines them needs those definitions removed.

- [ ] **Step 1: Write a test that detects duplicate symbol definitions**

```rust
// crates/grim-backend-rocm/tests/lib_internal_tests.rs
#[test]
fn kernel_source_has_no_duplicate_device_fns() {
    use grim_backend_rocm::kernels::source_asm::compute_kernel_source;
    let src = compute_kernel_source();
    let symbols = [
        "fp16_to_float_device",
        "fp8_e4m3_to_float_hip",
        "mxfp4_to_float_hip",
        "dequant_q4k_element",
    ];
    for sym in &symbols {
        let count = src.matches(sym).count();
        assert_eq!(
            count, 1,
            "symbol '{}' appears {} times in concatenated kernel source; expected 1",
            sym, count
        );
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test -p grim-backend-rocm kernel_source_has_no_duplicate_device_fns -- --nocapture
```

Expected: FAIL showing each symbol appears more than once.

- [ ] **Step 3: Remove the duplicate definitions from each module**

In `mxfp_standalone.rs`, remove the inline `__device__` definitions of `fp8_e4m3_to_float_hip` and `mxfp4_to_float_hip` from the `KERNEL_SOURCE` string. Both are already in `shared_device_fns.rs`. Keep only `grim_dequant_mxfp4` and `grim_dequant_mxfp8` in that file.

In `fp8_standalone.rs`, remove the inline definition of `fp8_e4m3_to_float_hip` from the `KERNEL_SOURCE` string. Keep only `grim_dequant_fp8`.

In `source_asm.rs`, ensure `shared_device_fns::KERNEL_SOURCE` is prepended first in the concatenation before any other module, and that no other module in the chain defines those four symbols. The build order must be:

```rust
// crates/grim-backend-rocm/src/kernels/source_asm.rs
pub fn compute_kernel_source() -> String {
    // shared_device_fns MUST come first so all __device__ helpers
    // are defined before any kernel that calls them.
    [
        crate::kernels::shared_device_fns::KERNEL_SOURCE,
        crate::kernels::compute_kernels::OTHER_KERNEL_SOURCE,
        crate::kernels::qkv_attention::KERNEL_SOURCE,
        // ... remaining modules in existing order, minus any
        // that now duplicate shared_device_fns symbols
    ]
    .concat()
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cargo test -p grim-backend-rocm kernel_source_has_no_duplicate_device_fns -- --nocapture
```

Expected: PASS.

- [ ] **Step 5: Build to confirm no compile errors**

```bash
cargo build -p grim-backend-rocm 2>&1 | grep -E "^error"
```

Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add crates/grim-backend-rocm/src/kernels/
git commit -m "fix(rocm): remove 4 duplicate __device__ symbols from concatenated HIPRTC source (A.0)"
```

---

### Task 0.2: Move 5 ROCm ops inside BackendDevice trait impl (C.1)

**Files:**
- Modify: `crates/grim-backend-rocm/src/device/roc_device.rs` (or wherever `impl RocmDevice` and `impl BackendDevice for RocmDevice` live)
- Test: `crates/grim-backend-rocm/tests/lib_internal_tests.rs`

**Interfaces:**
- Produces: `selective_scan`, `flash_attention`, `cross_attention`, `rwkv_time_mix`, `rwkv_channel_mix` reachable via `dyn BackendDevice`

- [ ] **Step 1: Write a test that calls each op through the trait**

```rust
#[test]
#[ignore = "requires GRIM_RUN_GPU_TESTS=1"]
fn rocm_trait_ops_are_reachable_via_dyn() {
    let dev: Box<dyn grim_tensor::backend::BackendDevice> =
        Box::new(grim_backend_rocm::RocmDevice::new(0).unwrap());
    // Each call must NOT return Err(Unimplemented); any other error
    // (e.g. shape mismatch) is acceptable since we pass dummy tensors.
    let dummy = dev.zeros(&[1, 1], grim_tensor::DType::BF16).unwrap();
    let r = dev.selective_scan(&dummy, &dummy, &dummy, &dummy);
    assert!(
        !matches!(r, Err(grim_tensor::error::Error::Unimplemented(_))),
        "selective_scan must not return Unimplemented when called via dyn BackendDevice"
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm rocm_trait_ops_are_reachable_via_dyn -- --ignored --nocapture
```

Expected: FAIL with `Unimplemented`.

- [ ] **Step 3: Move each method into the trait impl**

Find each method in `impl RocmDevice` (inherent) and cut/paste it into the body of `impl BackendDevice for RocmDevice`. Do not change any logic. The only change is which `impl` block the method lives in. Do this for: `selective_scan`, `flash_attention`, `cross_attention`, `rwkv_time_mix`, `rwkv_channel_mix`.

- [ ] **Step 4: Verify test passes**

```bash
GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm rocm_trait_ops_are_reachable_via_dyn -- --ignored --nocapture
```

Expected: PASS or a non-Unimplemented error.

- [ ] **Step 5: Commit**

```bash
git add crates/grim-backend-rocm/src/device/
git commit -m "fix(rocm): move selective_scan/flash_attention/cross_attention/rwkv ops into BackendDevice impl (C.1)"
```

---

## Phase 1: Real training loop

**Why now:** the parity gap with Unsloth and Axolotl is not about missing features. It's that `run_training_worker` never calls a real model. Phases 2-5 build on a real training loop.

---

### Task 1.1: Real forward pass via CausalLm

**Files:**
- Modify: `crates/grim-garage/src/jobs.rs`
- Modify: `crates/grim-garage/Cargo.toml` (add `grim-models-transformer` dependency)
- Test: `crates/grim-garage/tests/integration.rs`

**Interfaces:**
- Consumes: `TrainingJob { model_path, dataset_path, training_mode, lora_rank, learning_rate, ... }`
- Produces: a worker that calls `CausalLm::forward(input_ids, labels)` and returns real loss

- [ ] **Step 1: Write a test that catches a fake-loss training run**

```rust
// crates/grim-garage/tests/integration.rs
#[tokio::test]
async fn training_worker_produces_decreasing_loss_on_toy_model() {
    // Start a training job on a tiny test model and real toy dataset.
    // After 5 steps, the loss in step 5 must be lower than in step 1.
    // This test catches the simulation (which emits constant-trend fake loss).
    let (registry, _state) = build_test_state();
    let jid = start_toy_training_job(&registry).await;
    let metrics = collect_metrics(&registry, &jid, 5).await;
    assert!(
        metrics[4].loss < metrics[0].loss,
        "loss did not decrease: {:?}",
        metrics
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test -p grim-garage training_worker_produces_decreasing_loss_on_toy_model -- --nocapture
```

Expected: FAIL (constant trend or random loss from simulation).

- [ ] **Step 3: Replace the simulation block in run_training_worker**

In `crates/grim-garage/src/jobs.rs`, find `run_training_worker`. Remove the section that creates constant tensors and fake loss. Replace it with:

```rust
// Load the model from the .grim file at job.model_path.
let model = grim_models_transformer::CausalLm::from_grim(
    &job.model_path,
    backend.clone(),
)?;

// Load the LoRA injection if training_mode is LoRA or QLoRA.
if matches!(job.training_mode, TrainingMode::Lora | TrainingMode::QLoRA) {
    let registry = grim_autograd::injection::LoRAInjectionRegistry::standard_qlora(
        job.lora_rank,
    );
    model.inject_lora(&registry)?;
}

// Create the optimizer.
let mut optimizer = grim_autograd::adamw::AdamW::new(
    model.parameters(),
    job.learning_rate as f32,
    0.01,  // weight_decay
);

for step in 0..total_steps {
    let (input_ids, labels) = dataloader.next_batch()?;
    let loss = model.forward(&input_ids, &labels)?;
    loss.backward()?;
    optimizer.step()?;
    optimizer.zero_grad();

    update_status_and_broadcast(
        &registry,
        &jid,
        JobStatus::Running,
        Metric { step, loss: loss.item()? as f64, tokens: (step + 1) * batch_tokens },
    ).await;
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cargo test -p grim-garage training_worker_produces_decreasing_loss_on_toy_model -- --nocapture
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/grim-garage/src/jobs.rs crates/grim-garage/Cargo.toml
git commit -m "feat(garage): replace simulation with real CausalLm forward/backward in training worker"
```

---

### Task 1.2: JSONL dataloader

**Files:**
- Create: `crates/grim-garage/src/dataloader.rs`
- Modify: `crates/grim-garage/src/lib.rs` (add `mod dataloader`)
- Modify: `crates/grim-garage/src/jobs.rs` (use `JsonlBatchIterator`)
- Test: `crates/grim-garage/tests/integration.rs`

**Interfaces:**
- Produces: `JsonlBatchIterator::new(path, tokenizer, seq_len, batch_size) -> Result<Self>`
- Produces: `fn next_batch(&mut self) -> Result<(Tensor, Tensor)>` returning `(input_ids, labels)`

Labels are `input_ids` shifted left by one position with the last token set to the pad token ID. This is standard causal language model training.

- [ ] **Step 1: Write a failing test for the dataloader**

```rust
// crates/grim-garage/tests/integration.rs
#[test]
fn jsonl_dataloader_returns_correct_shapes() {
    let path = "tests/fixtures/toy.jsonl"; // "text": "hello world" x 10 lines
    let tokenizer = load_test_tokenizer();
    let mut loader = JsonlBatchIterator::new(path, tokenizer, 64, 2).unwrap();
    let (inputs, labels) = loader.next_batch().unwrap();
    assert_eq!(inputs.shape(), &[2, 64]);
    assert_eq!(labels.shape(), &[2, 64]);
    // labels[i][j] == inputs[i][j+1] for j < 63
    assert_eq!(labels.item_at(&[0, 0]), inputs.item_at(&[0, 1]));
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test -p grim-garage jsonl_dataloader_returns_correct_shapes -- --nocapture
```

Expected: FAIL with module not found.

- [ ] **Step 3: Implement JsonlBatchIterator**

```rust
// crates/grim-garage/src/dataloader.rs

use grim_format::tokenizer::GgufTokenizer;
use grim_tensor::{Tensor, DType};
use grim_tensor::error::Result;
use std::io::{BufRead, BufReader};
use std::fs::File;

/// Reads a `.jsonl` file where each line is `{"text": "..."}`,
/// tokenizes each line, packs tokens into fixed-length sequences,
/// and yields `(input_ids, labels)` tensor pairs.
///
/// Labels are input_ids shifted left by 1 (next-token prediction).
/// Sequences that exceed `seq_len` are split; shorter ones are padded
/// with the tokenizer's pad token ID.
pub struct JsonlBatchIterator {
    token_buffer: Vec<u32>,
    seq_len: usize,
    batch_size: usize,
    tokenizer: GgufTokenizer,
    reader: BufReader<File>,
    exhausted: bool,
}

impl JsonlBatchIterator {
    pub fn new(
        path: &str,
        tokenizer: GgufTokenizer,
        seq_len: usize,
        batch_size: usize,
    ) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| grim_tensor::error::Error::Backend(e.to_string()))?;
        Ok(Self {
            token_buffer: Vec::new(),
            seq_len,
            batch_size,
            tokenizer,
            reader: BufReader::new(file),
            exhausted: false,
        })
    }

    /// Returns the next batch as `(input_ids, labels)` tensors of shape
    /// `[batch_size, seq_len]`. Returns Err when the file is exhausted.
    pub fn next_batch(&mut self) -> Result<(Tensor, Tensor)> {
        let needed = self.batch_size * self.seq_len;
        while self.token_buffer.len() < needed && !self.exhausted {
            self.fill_buffer()?;
        }
        if self.token_buffer.len() < needed {
            return Err(grim_tensor::error::Error::Backend(
                "dataloader exhausted".into(),
            ));
        }
        let flat: Vec<u32> = self.token_buffer.drain(..needed).collect();
        let input_ids = Tensor::from_slice_u32(&flat, &[self.batch_size, self.seq_len])?;
        // Labels: shift left by 1, fill last column with pad token.
        let mut labels_flat = flat.clone();
        for row in 0..self.batch_size {
            let start = row * self.seq_len;
            for col in 0..(self.seq_len - 1) {
                labels_flat[start + col] = flat[start + col + 1];
            }
            labels_flat[start + self.seq_len - 1] = self.tokenizer.pad_token_id();
        }
        let labels = Tensor::from_slice_u32(&labels_flat, &[self.batch_size, self.seq_len])?;
        Ok((input_ids, labels))
    }

    fn fill_buffer(&mut self) -> Result<()> {
        let mut line = String::new();
        match self.reader.read_line(&mut line) {
            Ok(0) => { self.exhausted = true; }
            Ok(_) => {
                let v: serde_json::Value = serde_json::from_str(line.trim())
                    .map_err(|e| grim_tensor::error::Error::Backend(e.to_string()))?;
                let text = v["text"].as_str().unwrap_or("");
                let tokens = self.tokenizer.encode(text)?;
                self.token_buffer.extend(tokens);
            }
            Err(e) => return Err(grim_tensor::error::Error::Backend(e.to_string())),
        }
        Ok(())
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cargo test -p grim-garage jsonl_dataloader_returns_correct_shapes -- --nocapture
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/grim-garage/src/dataloader.rs crates/grim-garage/src/lib.rs
git commit -m "feat(garage): add JsonlBatchIterator for real training data loading"
```

---

### Task 1.3: Gradient accumulation

**Files:**
- Modify: `crates/grim-garage/src/jobs.rs`
- Modify: `crates/grim-garage/src/view_model/hyperparam.rs`

**Interfaces:**
- Consumes: `TrainingJob.accumulation_steps: u32` (add this field)
- Produces: optimizer step fires every `accumulation_steps` micro-steps; loss reported as average over the accumulation window

- [ ] **Step 1: Add accumulation_steps to TrainingJob and StartTrainingRequest**

```rust
// In StartTrainingRequest:
#[serde(default = "default_accumulation_steps")]
pub accumulation_steps: u32,

fn default_accumulation_steps() -> u32 { 1 }
```

Add the same field to `TrainingJob`.

- [ ] **Step 2: Update the worker loop**

```rust
let mut accum_loss = 0.0f32;
for micro_step in 0..total_steps {
    let (input_ids, labels) = dataloader.next_batch()?;
    let loss = model.forward(&input_ids, &labels)?;
    // Scale loss by 1/accumulation_steps so accumulated gradients
    // have the same magnitude as a single-step gradient.
    let scaled = loss.scale(1.0 / job.accumulation_steps as f32)?;
    scaled.backward()?;
    accum_loss += scaled.item()? as f32;

    if (micro_step + 1) % job.accumulation_steps as usize == 0 {
        optimizer.step()?;
        optimizer.zero_grad();
        let step = micro_step / job.accumulation_steps as usize;
        update_status_and_broadcast(
            &registry, &jid, JobStatus::Running,
            Metric { step: step as u64, loss: accum_loss as f64, tokens: /* ... */ },
        ).await;
        accum_loss = 0.0;
    }
}
```

- [ ] **Step 3: Write and run a test**

```rust
#[tokio::test]
async fn gradient_accumulation_produces_same_loss_as_single_step() {
    // Two runs on identical data: one with accumulation_steps=1,
    // one with accumulation_steps=4. After one full optimizer step,
    // the parameters must be equal (within f32 tolerance).
    let loss_1 = run_toy_training(accumulation_steps: 1, steps: 1).await;
    let loss_4 = run_toy_training(accumulation_steps: 4, steps: 4).await;
    assert!((loss_1 - loss_4).abs() < 1e-4, "{} vs {}", loss_1, loss_4);
}
```

```bash
cargo test -p grim-garage gradient_accumulation -- --nocapture
```

- [ ] **Step 4: Commit**

```bash
git add crates/grim-garage/src/jobs.rs crates/grim-garage/src/view_model/hyperparam.rs
git commit -m "feat(garage): gradient accumulation (accumulation_steps field + scaled backward)"
```

---

## Phase 2: Format wiring (Crow / Raven / Rook / Jay / Jackdaw / Magpie)

**Why now:** the codecs exist. Three small changes connect them to the garage API. Nothing new to build, only to wire.

---

### Task 2.1: WeightFormat enum and QuantMode extensions

**Files:**
- Modify: `crates/grim-backend-rocm/src/quantization.rs`
- Create: `crates/grim-garage/src/weight_format.rs`
- Modify: `crates/grim-garage/src/lib.rs`

**Interfaces:**
- Produces: `WeightFormat` enum with Crow, Raven, Rook, Jay, Jackdaw, Magpie variants
- Produces: `QuantMode::MxFp4Emulated` and `QuantMode::MxFp8Emulated` allowed on RDNA2+
- Produces: `QuantMode::Fp8` renamed to `Fp8Native` (existing gate unchanged)

- [ ] **Step 1: Write tests for the new arch gate rules**

```rust
// crates/grim-backend-rocm/src/quantization.rs
#[test]
fn rook_and_jackdaw_allowed_on_rdna2() {
    for arch in [GcnArch::RDNA2, GcnArch::RDNA3, GcnArch::RDNA4, GcnArch::CDNA3] {
        assert_eq!(
            resolve_quant_mode(arch, QuantMode::MxFp4Emulated),
            QuantMode::MxFp4Emulated,
            "{:?}: MxFp4Emulated must be allowed", arch
        );
        assert_eq!(
            resolve_quant_mode(arch, QuantMode::MxFp8Emulated),
            QuantMode::MxFp8Emulated,
            "{:?}: MxFp8Emulated must be allowed", arch
        );
    }
}

#[test]
fn fp8_native_still_blocked_on_rdna2_rdna3() {
    for arch in [GcnArch::RDNA1, GcnArch::RDNA2, GcnArch::RDNA3] {
        let resolved = resolve_quant_mode(arch, QuantMode::Fp8Native);
        assert_ne!(
            resolved, QuantMode::Fp8Native,
            "{:?}: Fp8Native must not be allowed", arch
        );
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cargo test -p grim-backend-rocm rook_and_jackdaw_allowed_on_rdna2 fp8_native_still_blocked -- --nocapture
```

- [ ] **Step 3: Update QuantMode and resolve_quant_mode**

```rust
// crates/grim-backend-rocm/src/quantization.rs
pub enum QuantMode {
    Fp32,
    F16,
    Bf16,
    /// Native FP8 MFMA: RDNA4 (gfx1200+) and CDNA3 (gfx94x) only.
    /// On RDNA2/3 this mode is downshifted to Bf16 by resolve_quant_mode.
    Fp8Native,
    /// MXFP4 emulated: dequant E2M1 weights to BF16 in LDS, then WMMA BF16 GEMM.
    /// Safe on RDNA2+. Uses existing grim_dequant_mxfp4 kernel.
    MxFp4Emulated,
    /// MXFP8 emulated: dequant E4M3 weights to BF16 in LDS, then WMMA BF16 GEMM.
    /// Safe on RDNA2+. Strictly better than Fp8Native emulation at same bpw
    /// because the shared E8M0 exponent per 32 weights captures outlier blocks.
    MxFp8Emulated,
}
```

Update `resolve_quant_mode` to pass `MxFp4Emulated` and `MxFp8Emulated` through on all arches, and to downshift `Fp8Native` on RDNA1/2/3 to Bf16 (same behavior as the old `Fp8` rule).

Update `arch_capability` to rename `fp8` to `fp8_native`.

- [ ] **Step 4: Create WeightFormat in grim-garage**

```rust
// crates/grim-garage/src/weight_format.rs

use serde::{Deserialize, Serialize};

/// Names the codec used to store base model weights during training.
/// The names are grim's internal bird-themed aliases.
///
/// On arches that don't support a format natively, grim falls back
/// via resolve_quant_mode: Raven -> Bf16 on RDNA2/3; all others pass through.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum WeightFormat {
    /// BF16 full precision. No quantization. Default for all arches.
    #[default]
    Bf16,
    /// Crow: Q4_K GGML super-block 4-bit. 4.5 bpw. RDNA2+.
    Crow,
    /// Raven: FP8 E4M3 native GEMM. 8 bpw. RDNA4 and CDNA3 only.
    /// Downshifts to Bf16 on RDNA2/3 via resolve_quant_mode.
    Raven,
    /// Rook: MXFP4 E2M1 emulated. Dequant in LDS to BF16, WMMA GEMM. ~4.1 bpw. RDNA2+.
    Rook,
    /// Jay: MXFP4 block-16. Alias for Fp4Block16. ~4.1 bpw. RDNA2+.
    Jay,
    /// Jackdaw: MXFP8 E4M3 emulated. Dequant in LDS to BF16, WMMA GEMM. ~8 bpw. RDNA2+.
    /// Better than Raven at same bpw: shared E8M0 exponent captures outlier blocks.
    Jackdaw,
    /// Magpie: MXFP8 block-16. Alias for Fp8Block16. ~8 bpw. RDNA2+.
    Magpie,
}
```

- [ ] **Step 5: Run tests and build**

```bash
cargo test -p grim-backend-rocm rook_and_jackdaw_allowed_on_rdna2 fp8_native_still_blocked -- --nocapture
cargo build -p grim-garage 2>&1 | grep "^error"
```

Expected: both tests pass, no build errors.

- [ ] **Step 6: Commit**

```bash
git add crates/grim-backend-rocm/src/quantization.rs crates/grim-garage/src/weight_format.rs crates/grim-garage/src/lib.rs
git commit -m "feat: add WeightFormat enum (Crow/Raven/Rook/Jay/Jackdaw/Magpie) and QuantMode::MxFp4Emulated/MxFp8Emulated"
```

---

### Task 2.2: Wire WeightFormat into StartTrainingRequest and convert_model_route

**Files:**
- Modify: `crates/grim-garage/src/jobs.rs`
- Modify: `crates/grim-garage/src/routes.rs`

- [ ] **Step 1: Add weight_format to StartTrainingRequest and TrainingJob**

```rust
// In StartTrainingRequest:
#[serde(default)]
pub weight_format: WeightFormat,

// In TrainingJob (add field alongside existing fields):
pub weight_format: WeightFormat,
```

- [ ] **Step 2: Add target_format to ConvertModelRequest**

```rust
// In ConvertModelRequest:
#[serde(default)]
pub target_format: Option<String>, // "crow", "raven", "rook", "jay", "jackdaw", "magpie"
```

Pass it through to `grim_format::convert_to_grim()` when present. Map the string to the appropriate `QuantFormat` variant before calling the converter.

- [ ] **Step 3: Write a route-level test**

```rust
// crates/grim-garage/tests/web_routes_integration.rs
#[tokio::test]
async fn start_training_with_crow_format_is_accepted() {
    let app = build_test_app();
    let body = json!({
        "model_path": "test_model",
        "dataset_path": "test_data",
        "weight_format": "crow"
    });
    let resp = post(&app, "/api/train/start", &body).await;
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["job_id"].is_string());
}
```

- [ ] **Step 4: Run tests, commit**

```bash
cargo test -p grim-garage start_training_with_crow_format_is_accepted -- --nocapture
git add crates/grim-garage/src/jobs.rs crates/grim-garage/src/routes.rs
git commit -m "feat(garage): wire WeightFormat into StartTrainingRequest and ConvertModelRequest"
```

---

## Phase 3: ROCm differentiation

These capabilities have no equivalent in any Python training framework on ROCm today.

---

### Task 3.1: Extend HIP graph capture to cover the training step

**Files:**
- Modify: `crates/grim-backend-rocm/src/graph_capture.rs`
- Modify: `crates/grim-garage/src/jobs.rs`

**Interfaces:**
- Produces: `TrainStepGraphKey { batch: u32, seq_len: u32, lora_rank: u32, quant_mode: QuantMode }`
- Produces: `GraphCaptureManager::capture_train_step(key, fn) -> Result<()>` and `replay_train_step(key) -> Result<()>`

The existing `GraphCaptureManager` and `DecodeGraph` already implement the full HIP graph lifecycle (`hipStreamBeginCapture`, `hipGraphInstantiate`, `hipGraphLaunch`). Adding train-step capture reuses this infrastructure with a new key type.

- [ ] **Step 1: Add TrainStepGraphKey**

```rust
// crates/grim-backend-rocm/src/graph_capture.rs

/// Cache key for a captured training step graph.
/// One graph is captured per (batch, seq_len, lora_rank, quant_mode) tuple.
/// Invalidate by dropping the manager and recreating.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TrainStepGraphKey {
    pub batch: u32,
    pub seq_len: u32,
    pub lora_rank: u32,
    pub quant_mode: QuantMode,
}
```

- [ ] **Step 2: Add capture_train_step and replay_train_step to GraphCaptureManager**

These methods follow the same pattern as the existing `capture` and `replay` methods for decode graphs. The only difference is the key type. No new HIP API calls needed.

- [ ] **Step 3: Enable graph capture in the worker after 3 warmup steps**

```rust
// crates/grim-garage/src/jobs.rs (inside run_training_worker)
const WARMUP_STEPS: usize = 3;
let mut graph_captured = false;

for step in 0..total_steps {
    if step == WARMUP_STEPS && job.rocm_fusion_qkv_attention {
        graph_manager.capture_train_step(graph_key, || {
            // The forward+backward closure to capture.
            // JIT kernels are warm after WARMUP_STEPS so the graph
            // records real PTX, not stub dispatches.
            model.forward_backward(&batch_input, &batch_labels)
        })?;
        graph_captured = true;
    }

    if graph_captured {
        // Reload batch into the pre-captured input buffers, then replay.
        model.update_graph_inputs(&batch_input, &batch_labels)?;
        graph_manager.replay_train_step(graph_key)?;
    } else {
        let loss = model.forward(&batch_input, &batch_labels)?;
        loss.backward()?;
    }
    // optimizer step and metric broadcast as before
}
```

- [ ] **Step 4: Commit**

```bash
git add crates/grim-backend-rocm/src/graph_capture.rs crates/grim-garage/src/jobs.rs
git commit -m "feat(rocm): HIP graph capture of training step (TrainStepGraphKey + 3-step warmup gate)"
```

---

### Task 3.2: FP8 native training on RDNA4 / MI300X

**Files:**
- Create: `crates/grim-backend-rocm/src/kernels/fp8_gemm_rdna4.rs`
- Modify: `crates/grim-backend-rocm/src/kernels/mod.rs`
- Modify: `crates/grim-backend-rocm/src/quantization.rs`

This task adds the forward path only. Stochastic rounding for backward is Task 3.3.

- [ ] **Step 1: Write a test gating on RDNA4 arch detection**

```rust
#[test]
#[ignore = "requires gfx1200+ hardware"]
fn fp8_native_forward_produces_finite_output_on_rdna4() {
    let dev = grim_backend_rocm::RocmDevice::new(0).unwrap();
    assert!(
        matches!(
            grim_backend_rocm::quantization::gcn_arch(&dev.arch_name()),
            GcnArch::RDNA4 | GcnArch::CDNA3
        ),
        "this test requires RDNA4 or CDNA3"
    );
    // 4x4 FP8 matmul should produce finite f32 output.
    let a = dev.ones_fp8(&[4, 4]).unwrap();
    let b = dev.ones_fp8(&[4, 4]).unwrap();
    let c = dev.fp8_matmul(&a, &b).unwrap();
    let c_host = dev.to_cpu_f32(&c).unwrap();
    assert!(c_host.iter().all(|x| x.is_finite()), "{:?}", c_host);
}
```

- [ ] **Step 2: Create fp8_gemm_rdna4.rs with the HIP kernel source**

The kernel calls `rocblas_gemm_ex` with `ROCBLAS_DATATYPE_F8_R` types and per-tensor scale pointers. The scale pointers follow the hipBLASLt API introduced in ROCm 6.2.

```rust
// crates/grim-backend-rocm/src/kernels/fp8_gemm_rdna4.rs

/// HIP kernel source for FP8 E4M3 GEMM using hipBLASLt scaled matmul.
/// Only compiled and dispatched on gfx1200+ (RDNA4) and gfx94x (CDNA3).
/// On other arches, the dispatch path returns Err(Unimplemented) and
/// the caller falls back via resolve_quant_mode.
pub const KERNEL_SOURCE: &str = r#"
// No HIPRTC kernel needed: dispatches via hipBLASLt rocblas_gemm_ex_fp8.
// This file holds the Rust dispatch wrapper only.
"#;

/// Dispatch a FP8 E4M3 matmul via hipBLASLt on gfx1200+/gfx94x.
/// a: [M, K] FP8 E4M3, b: [K, N] FP8 E4M3, returns c: [M, N] BF16.
/// scale_a and scale_b are per-tensor float scalars.
pub fn fp8_gemm(
    handle: *mut std::ffi::c_void,
    a: *const u8,
    b: *const u8,
    c: *mut u8,
    m: i32, n: i32, k: i32,
    scale_a: f32, scale_b: f32,
) -> grim_tensor::error::Result<()> {
    // hipBLASLt call with ROCBLAS_DATATYPE_F8_R inputs and
    // ROCBLAS_DATATYPE_BF16_R output. Scale pointers passed as
    // the d_scale and e_scale arguments introduced in ROCm 6.2.
    unsafe {
        // extern "C" fn rocblas_gemm_ex_fp8(...) defined via build.rs
        // linking against librocblas.so
        todo!("link hipBLASLt fp8 path via build.rs rocblas extern block")
    }
}
```

- [ ] **Step 3: Commit (even if the extern block is still a stub)**

```bash
git add crates/grim-backend-rocm/src/kernels/fp8_gemm_rdna4.rs
git commit -m "feat(rocm): fp8_gemm_rdna4 stub for hipBLASLt FP8 path on RDNA4/MI300X"
```

---

### Task 3.3: RCCL data-parallel gradient all-reduce

**Files:**
- Modify: `crates/grim-backend-rocm/src/rccl.rs`
- Modify: `crates/grim-garage/src/backend.rs`
- Modify: `crates/grim-garage/src/jobs.rs`

**Interfaces:**
- Consumes: `probe_rocm()` result (already lists visible devices by ordinal)
- Produces: `RcclAllReduce::sum_gradients(grad_bufs: &[&mut Tensor]) -> Result<()>`

- [ ] **Step 1: Remove the hardcoded device-0 ordinal**

In `crates/grim-garage/src/backend.rs`, `probe_rocm` calls `RocmDevice::new(0)`. Replace with `RocmDevice::new_best()` which picks the device with the most available VRAM, or the first enumerated device if all are equal. Add `new_best()` to `RocmDevice`.

- [ ] **Step 2: Add multi-device detection to TrainingJob**

```rust
// Add to StartTrainingRequest:
#[serde(default)]
pub num_gpus: u32,  // 0 or 1 = single GPU; >1 = data-parallel across N GPUs
```

- [ ] **Step 3: Wire RCCL all-reduce after optimizer.zero_grad**

```rust
// In run_training_worker, after loss.backward() and before optimizer.step():
if job.num_gpus > 1 {
    rccl_comm.all_reduce_sum(model.grad_buffers())?;
    // Scale gradients by 1/num_gpus to match single-GPU effective LR.
    model.scale_grads(1.0 / job.num_gpus as f32)?;
}
```

- [ ] **Step 4: Commit**

```bash
git add crates/grim-backend-rocm/src/rccl.rs crates/grim-garage/src/backend.rs crates/grim-garage/src/jobs.rs
git commit -m "feat(rocm): RCCL data-parallel all-reduce + multi-device probe (remove ordinal-0 hardcode)"
```

---

## Phase 4: Training quality

These close the feature gap with Axolotl.

---

### Task 4.1: Cosine-with-warmup LR scheduler

**Files:**
- Create: `crates/grim-autograd/src/lr_schedule.rs`
- Modify: `crates/grim-autograd/src/lib.rs`
- Modify: `crates/grim-garage/src/jobs.rs`

**Interfaces:**
- Produces: `CosineWarmupSchedule::new(total_steps, warmup_steps, base_lr, min_lr) -> Self`
- Produces: `fn lr_at_step(&self, step: usize) -> f32`

- [ ] **Step 1: Write tests for schedule values**

```rust
// crates/grim-autograd/src/lr_schedule.rs (inline test)
#[test]
fn warmup_reaches_base_lr_at_warmup_step() {
    let sched = CosineWarmupSchedule::new(100, 10, 1e-4, 1e-6);
    assert!((sched.lr_at_step(10) - 1e-4).abs() < 1e-8);
}

#[test]
fn lr_at_final_step_is_min_lr() {
    let sched = CosineWarmupSchedule::new(100, 10, 1e-4, 1e-6);
    assert!((sched.lr_at_step(100) - 1e-6).abs() < 1e-8);
}

#[test]
fn lr_is_monotone_decreasing_after_warmup() {
    let sched = CosineWarmupSchedule::new(100, 10, 1e-4, 1e-6);
    for step in 10..100 {
        assert!(sched.lr_at_step(step) >= sched.lr_at_step(step + 1));
    }
}
```

- [ ] **Step 2: Implement CosineWarmupSchedule**

```rust
pub struct CosineWarmupSchedule {
    total_steps: usize,
    warmup_steps: usize,
    base_lr: f32,
    min_lr: f32,
}

impl CosineWarmupSchedule {
    pub fn new(total_steps: usize, warmup_steps: usize, base_lr: f32, min_lr: f32) -> Self {
        Self { total_steps, warmup_steps, base_lr, min_lr }
    }

    /// Returns the learning rate for `step` (0-indexed).
    /// Linear ramp from 0 to base_lr during warmup, then cosine decay to min_lr.
    pub fn lr_at_step(&self, step: usize) -> f32 {
        if step <= self.warmup_steps {
            self.base_lr * (step as f32 / self.warmup_steps.max(1) as f32)
        } else {
            let progress = (step - self.warmup_steps) as f32
                / (self.total_steps - self.warmup_steps).max(1) as f32;
            let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
            self.min_lr + (self.base_lr - self.min_lr) * cosine
        }
    }
}
```

- [ ] **Step 3: Wire into training worker**

Before `optimizer.step()` each step, call `optimizer.set_lr(schedule.lr_at_step(step))`.

- [ ] **Step 4: Run tests and commit**

```bash
cargo test -p grim-autograd lr_schedule -- --nocapture
git add crates/grim-autograd/src/lr_schedule.rs crates/grim-autograd/src/lib.rs crates/grim-garage/src/jobs.rs
git commit -m "feat(autograd): cosine-with-warmup LR schedule + wire into training worker"
```

---

### Task 4.2: Checkpoint save and resume

**Files:**
- Modify: `crates/grim-format/src/train.rs`
- Modify: `crates/grim-garage/src/jobs.rs`
- Modify: `crates/grim-garage/src/routes.rs`

**Interfaces:**
- Produces: `TrainState::save_checkpoint(path, step, optimizer_state, lora_weights) -> Result<()>`
- Produces: `StartTrainingRequest.resume_from_checkpoint: Option<String>`

- [ ] **Step 1: Write a checkpoint round-trip test**

```rust
// crates/grim-format/src/train.rs
#[test]
fn checkpoint_round_trips_optimizer_state_and_step() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ckpt.train");
    let state = TrainState::with_step(42, vec![1.0f32, 2.0, 3.0]);
    state.write(path.to_str().unwrap()).unwrap();
    let loaded = TrainState::read(path.to_str().unwrap()).unwrap().unwrap();
    assert_eq!(loaded.step(), 42);
    assert_eq!(loaded.optimizer_moments(), &[1.0, 2.0, 3.0]);
}
```

- [ ] **Step 2: Add step field and optimizer_moments to TrainState**

Extend the existing `TrainState` struct (already in `train.rs`) with a `step: u64` field and `optimizer_moments: Vec<f32>` for first and second moment vectors. Serialize/deserialize via the existing binary format.

- [ ] **Step 3: Save checkpoint after each epoch in the worker**

```rust
// After completing one epoch's worth of steps:
let ckpt_path = format!("{}.epoch{}.train", job.model_path, epoch);
TrainState::save_checkpoint(
    &ckpt_path,
    global_step,
    optimizer.moments(),
    model.lora_weights(),
)?;
tracing::info!("checkpoint saved: {}", ckpt_path);
```

- [ ] **Step 4: Load checkpoint when resume_from_checkpoint is set**

```rust
// At the top of run_training_worker, before the step loop:
let start_step = if let Some(ref ckpt) = job.resume_from_checkpoint {
    let state = TrainState::read(ckpt)?.ok_or(
        Error::Backend(format!("checkpoint not found: {ckpt}"))
    )?;
    optimizer.load_moments(state.optimizer_moments())?;
    model.load_lora_weights(state.lora_weights())?;
    state.step()
} else {
    0
};
```

- [ ] **Step 5: Add resume_from_checkpoint to StartTrainingRequest and validate path**

```rust
#[serde(default)]
pub resume_from_checkpoint: Option<String>,
```

Apply `validate_job_path` to this field in the route handler.

- [ ] **Step 6: Run tests and commit**

```bash
cargo test -p grim-format checkpoint_round_trips -- --nocapture
git add crates/grim-format/src/train.rs crates/grim-garage/src/jobs.rs crates/grim-garage/src/routes.rs
git commit -m "feat: checkpoint save/resume (step + optimizer moments + LoRA weights, per-epoch)"
```

---

### Task 4.3: Richer metrics (grad_norm, lr, vram_used_mb, samples_per_sec)

**Files:**
- Modify: `crates/grim-garage/src/jobs.rs`

**Interfaces:**
- Produces: `Metric { step, loss, tokens, grad_norm: f32, lr: f32, vram_used_mb: u32, samples_per_sec: f32 }`

- [ ] **Step 1: Extend Metric struct**

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metric {
    pub step: u64,
    pub loss: f64,
    pub tokens: u64,
    pub grad_norm: f32,
    pub lr: f32,
    pub vram_used_mb: u32,
    pub samples_per_sec: f32,
}
```

- [ ] **Step 2: Compute each field in the worker loop**

```rust
// After loss.backward(), before optimizer.step():
let grad_norm = model.grad_global_norm()?;  // sqrt(sum of squared grad norms)

// After optimizer.step():
let lr = schedule.lr_at_step(step);
let vram_used_mb = backend.vram_used_bytes()? / (1024 * 1024);

// Timing:
let elapsed = step_start.elapsed().as_secs_f32();
let samples_per_sec = job.batch_size as f32 / elapsed.max(1e-6);
```

- [ ] **Step 3: Commit**

```bash
git add crates/grim-garage/src/jobs.rs
git commit -m "feat(garage): richer training metrics (grad_norm, lr, vram_used_mb, samples_per_sec)"
```

---

## Phase 5: Beyond-parity features

These give grim capabilities neither Unsloth nor Axolotl offers on ROCm.

---

### Task 5.1: SpQR sparse residuals on top of Crow Q4K

**Files:**
- Modify: `crates/grim-quant/src/lib.rs`
- Modify: `crates/grim-format/src/format.rs` (tensor extension metadata)
- Create: `crates/grim-quant/src/spqr.rs`

**Interfaces:**
- Produces: `spqr_identify_salient(weights: &[f32], curvature: &[f32], threshold: f32) -> (Vec<u32>, Vec<f32>)` returning (indices, values) of salient weights
- Produces: `GrimTensorExt.spqr_indices: Vec<u32>`, `GrimTensorExt.spqr_values: Vec<f16>` (new sidecar fields)

SpQR stores roughly 1% of weights in FP16 by identifying the ones with the largest Hessian sensitivity. The remaining 99% use INT4/Crow. At inference, the sparse residual is added back after INT4 dequant. This beats EvoPress-style global bitwidth search on instruction-following tasks at equivalent bit budget.

- [ ] **Step 1: Write a test that SpQR identifies high-curvature weights**

```rust
// crates/grim-quant/src/spqr.rs
#[test]
fn spqr_selects_high_curvature_weights_as_salient() {
    let weights = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
    // Curvature is very high at index 4.
    let curvature = vec![0.01f32, 0.01, 0.01, 0.01, 100.0];
    let (indices, _values) = spqr_identify_salient(&weights, &curvature, 0.01);
    assert!(indices.contains(&4), "index 4 must be selected: {:?}", indices);
}
```

- [ ] **Step 2: Implement spqr_identify_salient**

```rust
// crates/grim-quant/src/spqr.rs

/// Identifies salient weights: those whose Hessian curvature exceeds
/// `threshold * mean_curvature`. These weights are stored in FP16
/// alongside the INT4 base. At inference, they are added back after dequant.
///
/// Returns (indices, original_f32_values) for the salient positions.
pub fn spqr_identify_salient(
    weights: &[f32],
    curvature: &[f32],
    threshold_multiplier: f32,
) -> (Vec<u32>, Vec<f32>) {
    let mean_curv = curvature.iter().sum::<f32>() / curvature.len().max(1) as f32;
    let cutoff = mean_curv * threshold_multiplier;
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for (i, (&w, &c)) in weights.iter().zip(curvature.iter()).enumerate() {
        if c > cutoff {
            indices.push(i as u32);
            values.push(w);
        }
    }
    (indices, values)
}
```

- [ ] **Step 3: Add SpQR sidecar to GrimTensorExt in format.rs**

```rust
// Extend GrimTensorExt (or its equivalent in format.rs):
pub spqr_indices: Vec<u32>,  // indices of salient weights stored in FP16
pub spqr_values: Vec<u16>,   // FP16 values for those indices
```

- [ ] **Step 4: Write and run tests**

```bash
cargo test -p grim-quant spqr_selects_high_curvature -- --nocapture
```

- [ ] **Step 5: Commit**

```bash
git add crates/grim-quant/src/spqr.rs crates/grim-quant/src/lib.rs crates/grim-format/src/format.rs
git commit -m "feat(quant): SpQR salient-weight identification and sidecar format for Crow+FP16 residual"
```

---

### Task 5.2: Real EvoPress GA loop for per-layer bitwidth search

**Files:**
- Modify: `crates/grim-quant/src/lib.rs`
- Modify: `crates/grim-format/src/convert.rs`

**Interfaces:**
- Produces: `evopress_search(tensor_plans: &mut [TensorRewritePlan], eval_fn: impl Fn(&[TensorRewritePlan]) -> f32, budget_bpw: f32, generations: u32) -> Vec<TensorRewritePlan>`

The `evopress_generations` parameter already reaches `convert_to_grim`. It currently does nothing. This task makes it run a real GA loop.

- [ ] **Step 1: Write a test that shows generations > 0 produces lower eval cost**

```rust
#[test]
fn evopress_with_generations_finds_lower_cost_plan() {
    let plans_0 = run_evopress_search(generations: 0);
    let plans_10 = run_evopress_search(generations: 10);
    let cost_0 = eval_plan_cost(&plans_0);
    let cost_10 = eval_plan_cost(&plans_10);
    assert!(cost_10 <= cost_0, "GA must not make things worse: {} vs {}", cost_10, cost_0);
}
```

- [ ] **Step 2: Implement evopress_search**

```rust
/// Evolutionary search over per-layer format assignments.
/// Finds the assignment of formats (Crow/Rook/Jay/Jackdaw/etc.) to layers
/// that minimizes eval_fn(plans) while keeping mean bpw <= budget_bpw.
///
/// eval_fn is called with each candidate plan set and returns a scalar cost
/// (lower is better: typically perplexity or weighted MSE).
///
/// The GA loop:
///   - Population of 8 candidates, each initialized to Crow (4.5 bpw).
///   - Each generation: mutate one layer's format in each candidate,
///     evaluate, keep the best 4 by cost under budget, fill rest by crossover.
///   - After `generations` iterations, return the best candidate.
pub fn evopress_search(
    tensor_plans: &[TensorRewritePlan],
    eval_fn: impl Fn(&[TensorRewritePlan]) -> f32,
    budget_bpw: f32,
    generations: u32,
) -> Vec<TensorRewritePlan> {
    // ... GA implementation
}
```

- [ ] **Step 3: Run tests and commit**

```bash
cargo test -p grim-quant evopress_with_generations -- --nocapture
git add crates/grim-quant/src/lib.rs crates/grim-format/src/convert.rs
git commit -m "feat(quant): real EvoPress GA loop for per-layer bitwidth search (evopress_generations now functional)"
```

---

### Task 5.3: Int4 quantization-aware training (QAT) via straight-through estimator

**Files:**
- Modify: `crates/grim-autograd/src/ops.rs`
- Modify: `crates/grim-autograd/src/backward.rs`

**Interfaces:**
- Produces: `Op::FakeQuantInt4 { group_size: usize, bits: u32 }` added to the `Op` enum
- Produces: backward rule for FakeQuantInt4: STE (pass gradient through unchanged)

Fake-quantize wraps the forward: `x -> quant(x) -> dequant(x)`. The gradient of this through the quantizer is 1 (straight-through estimator), so backward is a no-op on the quantizer itself.

- [ ] **Step 1: Write a test that the STE gradient is pass-through**

```rust
#[test]
fn fake_quant_int4_backward_is_identity() {
    let x = Tensor::from_slice_f32(&[0.1, 0.5, 0.9], &[3]).with_grad();
    let y = fake_quant_int4(&x, 3, 4);
    let loss = y.sum();
    loss.backward().unwrap();
    // STE: grad of x must be all-ones (same as if no quantizer were present).
    let grad = x.grad().unwrap();
    assert!(grad.to_f32_vec().unwrap().iter().all(|&g| (g - 1.0).abs() < 1e-5));
}
```

- [ ] **Step 2: Add FakeQuantInt4 to Op enum**

```rust
// crates/grim-autograd/src/ops.rs
FakeQuantInt4 { group_size: usize, bits: u32 },
```

Forward: call `grim_quant::quant_packed_symmetric` then `dequant_*` to get the round-tripped f32.
Backward: push input gradient unchanged (STE).

- [ ] **Step 3: Run tests and commit**

```bash
cargo test -p grim-autograd fake_quant_int4_backward_is_identity -- --nocapture
git add crates/grim-autograd/src/ops.rs crates/grim-autograd/src/backward.rs
git commit -m "feat(autograd): FakeQuantInt4 op with straight-through estimator for QAT"
```

---

## Phase 6: Remediation (existing bugs, security, and UX friction)

---

### Task 6.1: Fix validate_job_path to allow absolute paths

**Files:**
- Modify: `crates/grim-garage/src/routes.rs`

The current check `value.contains('/')` rejects all absolute paths on Linux. This makes the API unusable for standard model directories like `/opt/models/`.

- [ ] **Step 1: Write tests for the intended behavior**

```rust
#[test]
fn validate_job_path_rejects_traversal() {
    assert!(validate_job_path("model_path", "../etc/passwd").is_err());
    assert!(validate_job_path("model_path", "foo/../bar").is_err());
}

#[test]
fn validate_job_path_accepts_absolute_paths() {
    assert!(validate_job_path("model_path", "/opt/models/llama.grim").is_ok());
    assert!(validate_job_path("model_path", "/home/user/data").is_ok());
}

#[test]
fn validate_job_path_accepts_relative_paths() {
    assert!(validate_job_path("model_path", "models/llama.grim").is_ok());
}
```

- [ ] **Step 2: Update validate_job_path**

```rust
pub(crate) fn validate_job_path(field: &str, value: &str) -> std::result::Result<(), String> {
    // Reject only path traversal components, not absolute paths.
    // A path with ".." anywhere is always forbidden regardless of context.
    // Backslashes are forbidden because grim runs on Linux only; a backslash
    // in a path is almost certainly a traversal attempt via encoding confusion.
    let has_traversal = value
        .split('/')
        .any(|component| component == ".." || component == ".");
    if has_traversal || value.contains('\\') {
        Err(format!(
            "{field}: invalid path {value:?} (forbidden: path traversal or backslash)"
        ))
    } else {
        Ok(())
    }
}
```

- [ ] **Step 3: Run tests**

```bash
cargo test -p grim-garage validate_job_path -- --nocapture
```

- [ ] **Step 4: Commit**

```bash
git add crates/grim-garage/src/routes.rs
git commit -m "fix(garage): validate_job_path allows absolute paths, rejects only .. traversal components"
```

---

### Task 6.2: Max concurrent jobs guard

**Files:**
- Modify: `crates/grim-garage/src/jobs.rs`
- Modify: `crates/grim-garage/src/routes.rs`

- [ ] **Step 1: Add max_concurrent to JobRegistry**

```rust
pub struct JobRegistry {
    jobs: Arc<RwLock<HashMap<JobId, TrainingJob>>>,
    metrics_tx: tokio::sync::broadcast::Sender<MetricStreamEvent>,
    max_concurrent: usize,
}
```

Default to `max_concurrent: 4`. Read from `GRIM_MAX_CONCURRENT_JOBS` env var at startup.

- [ ] **Step 2: Enforce limit in start_training_route**

```rust
// Before inserting the new job:
let running_count = registry.running_count().await;
if running_count >= registry.max_concurrent {
    return Err((
        StatusCode::TOO_MANY_REQUESTS,
        Json(json!({ "error": format!("max concurrent jobs ({}) reached", registry.max_concurrent) })),
    ));
}
```

- [ ] **Step 3: Test**

```rust
#[tokio::test]
async fn cannot_start_more_than_max_concurrent_jobs() {
    let app = build_test_app_with_max_concurrent(2);
    start_job(&app).await;
    start_job(&app).await;
    let resp = try_start_job(&app).await;
    assert_eq!(resp.status(), 429);
}
```

- [ ] **Step 4: Commit**

```bash
git add crates/grim-garage/src/jobs.rs crates/grim-garage/src/routes.rs
git commit -m "feat(garage): max_concurrent_jobs guard (default 4, GRIM_MAX_CONCURRENT_JOBS env override)"
```

---

## Phase summary

| Phase | What it delivers | Who it beats |
|-------|-----------------|--------------|
| P0: GPU unblock | HIPRTC compiles on real hardware; 5 ROCm ops reachable via trait | Prerequisite for everything |
| P1: Real loop | Actual forward/backward on real data; gradient accumulation; JSONL dataloader | Closes the simulation gap |
| P2: Format wiring | Crow/Raven/Rook/Jay/Jackdaw/Magpie selectable in API | No Python framework exposes this codec breadth |
| P3: ROCm differentiation | HIP graph capture of train step; FP8 native on RDNA4; RCCL multi-GPU | Neither Unsloth nor Axolotl on ROCm |
| P4: Training quality | Cosine LR, checkpoints, richer metrics | Parity with Axolotl |
| P5: Beyond parity | SpQR sparse residuals; real EvoPress GA; Int4 QAT | Nothing in Python ML land on ROCm |
| P6: Remediation | Path validation fix; max concurrent guard | Security and UX |

After P0 and P1 are done, grim is a real trainer. After P3, it is the fastest one on ROCm. After P5, it is the only one that does SpQR + EvoPress + QAT natively in Rust on AMD hardware.
