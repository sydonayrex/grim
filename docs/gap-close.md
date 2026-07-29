# Gap-Close Implementation Plan

**Goal**: Bring grim to parity with — or beyond — Ollama (inference serving) and Unsloth (fine-tuning) as a drop-in replacement for AMD RDNA 2, 3, and 4 users.

**Audit baseline (re-verified, measured numbers)**:
- `cargo build --workspace` ................. ok, 8.28s, 8 dead-code warnings, 0 errors
- `cargo test --workspace` .................. 15 (vulkan) + 1032 (rest) = **1047 passed, 0 failed, 5 ignored** — full workspace is GREEN, no excludes needed
- `GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm -- --ignored` — 1 pass (decode_gemm), **1 FAIL: `gpu_fused_attention_matches_cpu_reference`** on gfx1036 (kv_dequant_attention HIP compile error: 4 duplicate `__device__` helpers in one TU)
- Hardware on this box: AMD Ryzen 7 7745HX + Radeon 610M iGPU = gfx1036 (RDNA2 only). RDNA3/RDNA4 capability-declared, no live device.

**Notes on audit corrections from previous pass**:
- Vulkan init WAS already fixed; tests pass (15/15, 0.27s). The session-memory note about `--exclude grim-backend-vulkan` is stale and should be removed from memory after this plan lands.
- `kernels::shared_device_fns::KERNEL_SOURCE` already exists with all 4 helpers (`fp16_to_float_device`, `fp8_e4m3_to_float_hip`, `mxfp4_to_float_hip`, `dequant_q4k_element`) and is already pushed first by `source_asm::compute_kernel_source()`. The per-quant kernel files STILL duplicate-define them — that duplication is the actual root cause of the kv_dequant_attention compile failure.
- The Crow Tier Q4K / Q2K / Q3K / IQ kernels ALREADY have `grim_fused_dequant_backward_gemm_*` kernels defined. Forward/backward pairs exist for K-quants and IQ; backward exists for FP8 (`grim_fused_dequant_backward_gemm_fp8`) and for the FP8 MFMA variant (`grim_fused_dequant_backward_gemm_fp8_mfma`).
- `matmul_backward` in `crates/grim-autograd/src/ops.rs` already has a GPU/ROCm dispatch arm (`quantized_matmul_backward_dx`) wired for `KQuant | Block | FloatPack | GroupInt` storage on `Device::Rocm`. The autograd IS routing backward through quantized ROCm kernels — this is real.
- Jay (MXFP4) and Magpie (MXFP8) **forward** GEMMs exist: `grim_fused_dequant_gemm_mxfp4`, `grim_fused_dequant_gemm_mxfp8`, `grim_fused_dequant_gemm_fp8_mfma`. **Backward exists for FP8 MFMA only; backward for MXFP4 (Jay) and MXFP8 (Magpie) is MISSING** — that is the genuine Unsloth-tier gap on the Jay/Magpie axis.

---

## Axis A — Ollama Drop-In Parity (serving + REST)

`crates/grim-server/src/lib.rs` route table is 85% Ollama-compat today. Missing endpoints, in rough effort order:

| Route | Status | Effort | Notes |
|---|---|---|---|
| `/api/show` | MISSING | S | `GET` — return model metadata (modelfile, parameters, template, details). Reuse `catalog::resolve_model` + GGUF metadata already parsed by `grim-format::gguf::read_gguf`. |
| `/api/ps` | MISSING | S | `GET` — list running models. Reuse `Engine::running_sessions()` (or expose a new `Engine::ps()`). Map to Ollama's `{name, model, size, digest, expires_at, size_vram}` shape. |
| `/api/copy` | MISSING | S | `POST {source, destination}` — call `Engine::register_model(dest, engine.get(source).clone())`. |
| `/api/rm` | MISSING | S | `DELETE` from catalog + unload from engine. |
| `/api/version` | MISSING | XS | Return grim version string. |
| `/api/create` | MISSING | M | `POST` — modify-from-existing: pull base, apply Modelfile-like overrides (system prompt, template, params), re-export as new GGUF with grim metadata layer. Noun: uses `grim-format::convert` + `grim-format::spec` layers. |
| `/api/push` | MISSING | M | Registry push. Needs a remote registry protocol (HuggingFace Hub or Ollama registry). Lower priority — Ollama users rarely push. Defer or stub with 501. |
| `/api/blob/*` | MISSING | M | Content-addressable blob store. Real change to `catalog` module. Defer unless `/api/push` is requested. |

**Work items A.1–A.8**: implement each endpoint above; re-use existing `Engine`, `catalog`, `grim-format::gguf` infrastructure. Each endpoint gets a test in `crates/grim-server/src/lib.rs` mirroring the existing `test_grim_compatibility_shims`. No new crates.

**Pass criterion A**: A client pointed at `localhost:PORT` configured as if it were Ollama (base URL only) can `pull`, `show`, `ps`, `chat`, `generate`, `tags`, `copy`, `rm`, `create`, `version` — full lifecycle. `push` and `blob` may return 501 with a clear message.

**Verification command A**:
```
cargo test -p grim-server
# + a new integration test `test_ollama_full_lifecycle` that drives every endpoint above.
```

---

## Axis B — Unsloth Drop-In Parity (training/fine-tuning)

The skeleton is real: `grim-autograd` has AdamW + backward + LoRA injection + `lora_backward`. `grim-cli train` runs a real `standard_qlora` path that passes `train_loop_loss_decreases_on_overfit_toy_dataset`. The kernel side is more complete than my first audit said:

**Already there**:
- Crow Tier K-quants: forward + backward GEMM kernels for Q4_K, Q2_K, Q3_K, Q5_K, Q6_K ✓
- Crow Tier IQ family: forward + backward for iq2xxs, iq2xs, iq2s, iq3xxs, iq3s, iq4nl, iq4xs, q8_0 ✓
- Raven FP8: `grim_fused_dequant_gemm_fp8` (forward) + `grim_fused_dequant_backward_gemm_fp8` (backward) ✓
- Raven FP8 MFMA: `grim_fused_dequant_gemm_fp8_mfma` + `grim_fused_dequant_backward_gemm_fp8_mfma` (RDNA4 path) ✓
- Jay MXFP4: `grim_fused_dequant_gemm_mxfp4` (forward) — **no backward**
- Magpie MXFP8: `grim_fused_dequant_gemm_mxfp8` (forward) — **no backward**
- `matmul_backward` in `grim-autograd/src/ops.rs` already dispatches to ROCm quantized backward when `b` is on `Device::Rocm` and storage is `KQuant | Block | FloatPack | GroupInt` ✓

**Genuine gaps for Unsloth parity**:

### B.1 — Fix the kv_dequant_attention GPU blocker (BLOCKER)
**Problem**: `compute_kernel_source()` in `kernels/source_asm.rs` concatenates `shared_device_fns::KERNEL_SOURCE` first (correct), but then each per-quant kernel (`q4k_gemm.rs`, `q4k_dequant.rs`, `q2k_gemm.rs`, `q3k_gemm.rs`, `q5k_gemm.rs`, `q6k_gemm.rs`, `q8_0_dequant.rs`, `iq_gemm.rs`, `iq_dequant.rs`, `wmma_gemm.rs`, `fp8_standalone.rs`, `mxfp_standalone.rs`) STILL contains its own `__device__ inline` copy of `fp16_to_float_device` (and friends, depending on the file). HIP rejects redefinition in the same translation unit → `gpu_fused_attention_matches_cpu_reference` fails to compile for gfx1036.

**Fix**: For each of those 12 kernel files, delete the local `__device__ inline float fp16_to_float_device(...)` (and `fp8_e4m3_to_float_hip`, `mxfp4_to_float_hip`, `dequant_q4k_element` where applicable). They are now provided by `shared_device_fns::KERNEL_SOURCE` which is prepended. Keep the surrounding `extern "C" { ... }` block intact if it contains other kernel declarations; remove only the duplicate helper definitions.

**Risk**: Low. Bodies are byte-identical (verified). Removing duplicates is a no-op semantically.

**Verification B.1**:
```
GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm -- --ignored
# Must show: gpu_fused_attention_matches_cpu_reference ... ok
# Must show: gpu_fused_attn_decode_throughput_vs_dense ... ok
# Then full workspace:
cargo test --workspace
# Must remain fully green; expect 1047+2 = 1049 passed, 0 failed (the previously-ignored become green)
```

### B.2 — Add Jay MXFP4 backward GEMM kernel
**Problem**: `grim_fused_dequant_backward_gemm_mxfp4` does not exist. A user fine-tuning a model whose weights were quantized to MXFP4 cannot route backward through the fused kernel — falls back to either dequant-then-matmul (memory blowup) or CPU.

**Implementation** (in `crates/grim-backend-rocm/src/kernels/wmma_gemm.rs`, under the "Jay (MXFP4) & Magpie (MXFP8) Kernels" comment block, mirror the pattern of `grim_fused_dequant_backward_gemm_fp8`):
1. Read the forward kernel `grim_fused_dequant_gemm_mxfp4` (lines ~148–178 of `wmma_gemm.rs`) for the dequant expression: `mxfp4_to_float_hip(code, exp_val)`.
2. Backward needs: grad_A = grad_out @ B^T (with B dequantized on-the-fly per element), grad_B = A^T @ grad_out (accumulated into MXFP4 block layout — but grad_B is fp32 in LoRA; only grad_A needs on-the-fly dequant of B).
3. Emit a `grim_fused_dequant_backward_gemm_mxfp4` kernel: signature `(A, B_mxfp4, B_exp, grad_out, grad_a, M, N, K, ...)` mirroring the FP8 backward.
4. Add `check_kernel!("grim_fused_dequant_backward_gemm_mxfp4")` unit test next to the existing `check_kernel!` tests in `wmma_gemm.rs`.
5. Gate dispatch on `GcnArch::RDNA4` (MXFP4 native is RDNA4; on RDNA2/3 the dequant is emulated via the helper, so the kernel still works but is slower — that's the correct behavior per `rust-gpu-discipline` §2 #12, butInverse: backward MXFP4 is fine on RDNA2/3 because the helper does software dequant; it just doesn't use MFMA. Document).

**Verification B.2**:
```
cargo test -p grim-backend-rocm --lib kernels::wmma_gemm::tests
# New test test_check_backward_mxfp4_kernel_present passes.
# Plus a host-reference test: small M=N=K=64, random MXFP4 B, run forward + backward on CPU reference + on GPU, compare grad_a within 1e-3.
```

### B.3 — Add Magpie MXFP8 backward GEMM kernel
Identical to B.2 but for `grim_fused_dequant_backward_gemm_mxfp8`. Same file, same pattern, same gating logic. RDNA4 has native MFMA paths (mirror `grim_fused_dequant_backward_gemm_fp8_mfma`); RDNA2/3 fall back to the `fp8_e4m3_to_float_hip` software helper.

**Verification B.3**: same shape as B.2, different kernel name.

### B.4 — Route Jay/Magpie backward through `matmul_backward` autograd dispatch
**Problem**: `crates/grim-autograd/src/ops.rs:91-130` has a `quantized_matmul_backward_dx` dispatch but only checks `Storage::KQuant | Block | FloatPack | GroupInt`. MXFP4 and MXFP8 storages need to be added to that match (whatever their `Storage` enum variant is — verify in `grim-tensor`).

**Implementation**:
1. Find the `Storage::MXFP4` / `Storage::MXFP8` variants (or equivalent) in `crates/grim-tensor/src/lib.rs`.
2. Extend the `matches!` in `matmul_backward` to include them.
3. Extend `RocmDevice::quantized_matmul_backward_dx` to dispatch to the new B.2/B.3 kernels by storage kind.

**Verification B.4**:
```
cargo test -p grim-autograd
# Plus a new test test_matmul_backward_dispatches_to_mxfp4 and test_matmul_backward_dispatches_to_mxfp8
# that constructs a small MXFP4/MXFP8 tensor on Device::Rocm and verifies the backward path
# routes to the new fused kernel (assert via a flag or by checking the kernel name in trace output).
```

### B.5 — BF16/FP16 mixed-precision training path
**Problem**: `crates/grim-cli/src/train.rs` and `crates/grim-autograd/src/{ops.rs, backward.rs, adamw.rs}` are f32-only. `grep -E "bf16|bfloat16|f16|BF16|F16" crates/grim-cli/src/train.rs` returns zero hits. Unsloth's headline is 4-bit QLoRA + BF16 mixed precision; grim has neither.

**Implementation** (multi-step):
1. **B.5.a**: Audit `grim-tensor::DType::BF16` and `DType::F16` — confirm they exist and that the ROCm backend already has matmul kernels for them (the ROCm GEMM `rocblas_gemm_ex` supports f16/bf16 natively on RDNA2+ per `QuantCapability`).
2. **B.5.b**: Extend `AdamWConfig` in `grim-autograd/src/adamw.rs` with a `compute_dtype: DType` field (default F32; allow BF16/F16). Optimizer state (`m`, `v`) stays F32; the gradient is upcast from compute dtype to F32 before the Adam update (standard mixed-precision recipe,Matches Unsloth's `fp16_mixed_precision`).
3. **B.5.c**: Extend `grim-cli/src/train.rs` `TrainOpts` with `--compute-dtype` flag (default f32; choices: f32, bf16, f16). Cast the model's forward activations to the compute dtype pre-forward; cast gradients back to f32 pre-optimizer-step.
4. **B.5.d**: Add test `test_adamw_bf16_mixed_precision_matches_f32_baseline` in `grim-autograd/src/adamw.rs` — run 100 steps on a toy toy dataset in both f32 and bf16, assert final loss differs by < 5% (mixed precision must not diverge).
5. **B.5.e**: Wire `grim-cli train` to plumb the flag through to the engine's forward pass.

**Risk**: M. Need loss-scaling for FP16 to avoid underflow; BF16 is safer (no loss scaling needed) and is the Unsloth default on AMD. Recommend shipping BF16 first, FP16 in a follow-up.

**Verification B.5**:
```
cargo test -p grim-autograd --lib adamw
cargo test -p grim-cli train_loop_loss_decreases_on_overfit_toy_dataset -- --features bf16
# New test passes, existing test still passes.
```

### B.6 — Flash-attention backward
**Problem**: `crates/grim-backend-rocm/src/kernels/flash_attn.rs` has only the forward kernel. `crates/grim-autograd/src/ops.rs` does not have a flash-attn backward dispatch. Backward goes through the reference `qkv_attention` CPU path → silent CPU-degrade (rust-gpu-discipline #6).

**Implementation**:
1. Add `grim_flash_attn_backward` to `flash_attn.rs` (the backward of flash attention is well-documented — Rabe & Wolf 2022; the kernel recomputes the forward softmax stats and accumulates grads to Q, K, V).
2. Add `test_check_flash_attn_backward_kernel_present` to flash_attn.rs's tests.
3. Wire `grim-autograd/src/ops.rs` `attention_backward` (if it doesn't exist — verify) to dispatch to the new kernel when on `Device::Rocm`.

**Verification B.6**:
```
cargo test -p grim-backend-rocm --lib kernels::flash_attn::tests
GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm --test flash_attn_gpu  # new test file
# Host-reference grad check vs CPU flash-attn backward; tolerance 1e-3 for f16, 1e-5 for f32.
```

### B.7 — Gradient checkpointing
**Problem**: `grep "gradient.checkpoint" crates/grim-autograd/ crates/grim-cli/src/train.rs` returns zero. For 7B+ models on 16GB cards, no checkpointing = OOM during training even with LoRA.

**Implementation**:
1. Add a `CheckpointScope` to `grim-autograd/src/lib.rs` — a region recorder that drops intermediate activations and recomputes them on backward.
2. In `crates/grim-models/transformer/src/block.rs` `LlamaBlock::forward`, add a `checkpoint: bool` field (or take it via the tape context); if true, mark the block's activations as checkpointed.
3. On `backward()`, the tape walks the checkpointed blocks in reverse and recomputes forward activations on-the-fly before computing the block's local backward.

**Risk**: M. Recomputeforward1x adds ~30% training time but cuts activation memory by ~sqrt(L) for naive chunking, more for stratified.

**Verification B.7**:
```
cargo test -p grim-autograd test_gradient_checkpointing_recomputes_correctly
cargo test -p grim-autograd test_gradient_checkpointing_memory_in_test_decreases
# A test that constructs a 4-layer LlamaBlock, runs forward+backward with and without checkpoint,
# asserts gradients are equal within 1e-5 AND peak tensor count is lower with checkpoint=true.
```

### B.8 — RSLoRA and DoRA injection variants
**Problem**: `crates/grim-autograd/src/injection.rs` has `InjectionConfig` and `LoRAInjectionRegistry::standard_qlora` but no rank-stabilized LoRA (RSLoRA) or weight-decomposed LoRA (DoRA). Unsloth ships both.

**Implementation**:
1. Add `InjectionVariant::RSLoRA { alpha_scaling: f32 }` — multiplies the LoRA output by `alpha / sqrt(rank)` instead of `alpha / rank`.
2. Add `InjectionVariant::DoRA { magnitude_init: f32 }` — splits the base weight into magnitude + direction, applies LoRA to direction only, re-scales by magnitude.
3. Extend `apply_and_record_lora` and `lora_backward` to handle the variants.

**Verification B.8**: toy-test each variant in `grim-autograd/src/injection.rs` — forward output matches the unsloth reference within 1e-5 for a 64×64 toy.

---

## Axis C — ROCm RDNA 2 / 3 / 4 verification & coverage

### C.1 — Verify RDNA2 (gfx1036) full GPU test suite lives on this box
After B.1 lands, `GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm -- --ignored` should pass on this machine. Run it and document actual pass/fail per test. If any test still fails on gfx1036, file as a separate work item.

### C.2 — CI runner for RDNA3 (gfx110x) and RDNA4 (gfx1200)
**Problem**: This box only has RDNA2. RDNA3/RDNA4 capability-gating is correct in `quantization.rs` but the kernels themselves are unverified on actual hardware. Without CI on those GPUs, the "drop-in for RDNA 2/3/4" claim is honest only for RDNA2.

**Implementation**:
1. Stand up a CI matrix in `.github/workflows/` (or equivalent) with three GPU tiers: gfx1036 (RDNA2), gfx110x (RDNA3), gfx1200 (RDNA4). Use whatever CI provider the project uses (verify — none found in repo today; if none exists this becomes "set up GPU CI" which is a separate meta-task).
2. Each tier runs `GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm -- --ignored`.
3. Surface a per-arch dashboard badge.

**If a CI runner is not available**: document in README that RDNA3/RDNA4 are capability-declared and unit-tested in software emulation on RDNA2, but live-hardware verification is pending. DO NOT claim "verified on RDNA3/4" without it.

### C.3 — Wave-64 audit of GEMM kernels
**Problem**: The `kv_dequant_attention` kernel uses `warpSize` (correct, 64 on RDNA). Verify each GEMM kernel (`q4k_gemm`, `q5k_gemm`, `q6k_gemm`, `q2k_gemm`, `q3k_gemm`, `iq_gemm`, `wmma_gemm`) uses `__launch_bounds__` that's a multiple of 64 and that `blockDim.x` is a multiple of 64 — otherwise half the SIMD wave is wasted on RDNA2/3 (rust-ai-ml-inference-guide §9).

**Implementation**: read each GEMM kernel's `__launch_bounds__` and `blockDim` declarations; raise a work item for any kernel using 32-thread blocks.

### C.4 — `hipinfo` probe convenience script
Add `scripts/hipinfo-grim.sh` that prints `gcnArchName` + `GcnArch::from_arch()` mapping + `QuantCapability` for the detected device. Useful for users to diagnose "why is fp8 falling back to bf16".

---

## Axis D — Vulkan init / workspace test sanity

**Already done**: Vulkan init was fixed (no ENABLE_PRIMUS, rejects VK_PHYSICAL_DEVICE_TYPE_CPU). `cargo test -p grim-backend-vulkan` passes 15/15 in 0.27s. `cargo test --workspace` is fully green.

### D.1 — Remove stale memory note
After landing this plan, update the Hermes memory entry that says "vulkan hangs, requires --exclude" — it is no longer true. This is a memory edit, not a code edit.

### D.2 — README quick-start fix
The README quick-start currently shows `cargo test` (no excludes). That's now correct. No edit needed. Do update the "optional: ROCm GPU tests" line to mention `--ignored` is also gated by `GRIM_RUN_GPU_TESTS=1` and the rest of the suite is green.

### D.3 — Audit `workspace.features` warning
`cargo build` shows `unused manifest key: workspace.features`. The `[workspace.features]` block in root `Cargo.toml` is non-standard (workspace-level features need to be specified per-crate). Either remove the block or migrate to `[workspace.dependencies]` features. Cosmetic but noisy.

---

## Recommended execution order

| # | Item | Effort | Unblocks | Risk |
|---|---|---|---|---|
| 1 | **B.1** kv_dequant_attention fix (delete duplicate helpers from 12 kernel files) | S | GPU test suite on RDNA2; C.1 | Low |
| 2 | **A.1–A.5** Small Ollama endpoints (show, ps, copy, rm, version) | S/M | Ollama parity claim | Low |
| 3 | **A.6** `/api/create` (modify-from-existing) | M | Ollama parity claim | M |
| 4 | **B.2** Jay MXFP4 backward kernel | M | Unsloth MXFP4 training | M |
| 5 | **B.3** Magpie MXFP8 backward kernel | M | Unsloth MXFP8 training | M |
| 6 | **B.4** Autograd dispatch wiring for Jay/Magpie | S | B.2, B.3 usable | Low |
| 7 | **B.5** BF16 mixed-precision training | M/L | Unsloth headline feature | M |
| 8 | **B.6** Flash-attention backward | M | Performance + memory | M |
| 9 | **B.7** Gradient checkpointing | M | 7B+ training on 16GB cards | M |
| 10 | **B.8** RSLoRA / DoRA variants | S | Unsloth feature parity | Low |
| 11 | **A.7** `/api/push` (or 501 stub) | M | Ollama parity completeness | Low (stub) |
| 12 | **A.8** `/api/blob/*` | M | `/api/push` real impl | M |
| 13 | **C.2** CI runner for RDNA3+RDNA4 | L | "Verified on RDNA2/3/4" claim | M (infra) |
| 14 | **C.3** Wave-64 audit | S | Perf correctness | Low |
| 15 | **C.4** hipinfo-grim script | XS | UX | Low |
| 16 | **D.1** Memory note update | XS | N/A | Low |
| 17 | **D.3** `workspace.features` cosmetic fix | XS | N/A | Low |

Total: 17 work items. Item 1 alone closes the audit's only hard blocker. Items 1–6 together close most of the named Unsloth gap (Crow Q4K already done, Raven FP8 already done, Jay + Magpie backward added). Items 1–10 close the core Unsloth parity claim. Items 1–13 close the full drop-in claim for RDNA 2/3/4.

---

## Final pass-criterion (project-level)

```
cargo build --workspace                                    # ok, 0 errors
cargo test --workspace                                     # fully green, no --exclude
GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm -- --ignored  # passes on gfx1036 (this box)
# On a CI box with gfx110x:    GRIM_RUN_GPU_TESTS=1 passes
# On a CI box with gfx1200:     GRIM_RUN_GPU_TESTS=1 passes
cargo test -p grim-server --test ollama_full_lifecycle     # new test, passes
grim-cli train --bf16 --qlora --rank 32 toy-config.json   # passes; loss decreases; no OOM on 7B @ 16GB
grim-cli train --mxfp4 --qlora --rank 32 toy-config.json   # passes; uses Jay backward kernel
grim-cli train --mxfp8 --qlora --rank 32 toy-config.json   # passes; uses Magpie backward kernel
```

When every line above is green, the project is on par with Ollama AND Unsloth as a drop-in replacement for AMD RDNA 2, 3, and 4 users. Items 1, 2–6, 7–10 deliver the bulk; 11–17 deliver completeness.

---

## Honest non-goals (out of scope for this plan)

- Multi-node / cluster training (grim-disagg is single-machine; cluster stays out per scope).
- Pre-training (only fine-tuning; pre-training a base model is out of Unsloth's scope too).
- TF32 / FP32 emulation ('high' matmul precision) — Ollama/Unsloth do not expose this; not user-facing.
- reformer / RWKV state-sharding training (RWKV6/7 forward paths exist; backward is open work but Unsloth doesn't ship that either).
- ONNX loader (`grim-format/src/onnx.rs` is a placeholder; Ollama doesn't load ONNX either; defer).

---

## Source references (file:line)

- Route table: `crates/grim-server/src/lib.rs:1345-1367`
- Missing helpers cleanup target: `crates/grim-backend-rocm/src/kernels/source_asm.rs:27-48` + the 12 kernel files listed in B.1
- Crow backward kernels (already done): `crates/grim-backend-rocm/src/kernels/q4k_gemm.rs:40`, `q2k_gemm.rs:77`, `q3k_gemm.rs:127`, `iq_gemm.rs:363-713`
- Raven kernel set: `crates/grim-backend-rocm/src/kernels/wmma_gemm.rs:98, 122, 228, 265`
- Jay/Magpie forward kernels (need backward pairs): `crates/grim-backend-rocm/src/kernels/wmma_gemm.rs:148, 179`
- Autograd quantized backward dispatch: `crates/grim-autograd/src/ops.rs:91-130`
- f32-only training: `crates/grim-cli/src/train.rs` (zero BF16/F16 hits)
- Quant capability gating: `crates/grim-backend-rocm/src/quantization.rs:183-188`
- Vulkan init (already fixed): `crates/grim-backend-vulkan/src/lib.rs:526-577`
