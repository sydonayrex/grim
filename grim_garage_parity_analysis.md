# Grim-Garage: Parity Analysis vs Unsloth & Axolotl

> **Skills applied:** ponytail · caveman · rust-ffi-grim · llm-training · rust-ml-llm-architecture · rust  
> **Scope:** `crates/grim-garage/src/` — `jobs.rs`, `backend.rs`, `routes.rs`, `rocm.rs`, `discovery.rs`, `view_model/`, plus top-level `grim_corrections.md`.  
> **Comparators:** Unsloth v2024.x (Python, ROCm/CUDA), Axolotl v0.4.x (Python, HF Transformers).

---

## Legend

| Symbol | Meaning |
|--------|---------|
| 🔴 **DEFICIENCY** | Grim is behind; Unsloth or Axolotl does this better |
| 🟢 **WIN** | Grim leads or uniquely solves something they don't |
| 🟡 **TIE** | Roughly equivalent capability |

---

## 1. Training Loop Fidelity

### 🔴 DEFICIENCY — Simulated worker, not real forward/backward pass

`run_training_worker` (`jobs.rs:471–604`) runs a **toy simulation**:

- Fixed 10 steps/epoch (`steps_per_epoch: u64 = 10` — line 396).
- `hidden_size = 4096`, `vocab_size = 32000` hardcoded (line 436–437) regardless of actual model.
- No real dataloader — zero-constant tensors (`vec![0.1f32; hidden_size]`) stand in for inputs.
- Token count is fake: `(step + 1) * 512` (line 580).

**Unsloth/Axolotl:** full `model.forward()` → real loss → `optimizer.step()` per batch from real data.  
**Impact:** grim-garage cannot actually fine-tune a model today. It's a dashboard wired to a simulation engine. The autograd machinery (`grim-autograd`) is real, but the training loop that feeds it is not.

---

### 🔴 DEFICIENCY — No real dataloader / tokenization integration

- `discovery.rs` finds `.jsonl`/`.parquet`/`.json` files by extension but **never reads them**.
- No tokenizer call inside `run_training_worker`; the tokenizer the `AppState` holds is behind a `Mutex<Option<GgufTokenizer>>` and never touched in the training path.
- Batch assembly, padding, sequence packing, attention masking: absent.

**Unsloth:** `UnslothTrainer` calls HF `DataCollatorForSeq2Seq`; packs sequences to fill context window.  
**Axolotl:** `DatasetProcessor` + configurable `sequence_len` + `sample_packing`.

---

### 🔴 DEFICIENCY — No gradient accumulation

`AdamW::step` fires once per step with no gradient-accumulation counter.  
Both competitors support `gradient_accumulation_steps`; essential for consumer-VRAM ROCm cards.

---

### 🔴 DEFICIENCY — No mixed-precision (BF16/FP16) AMP path

`TrainingJob.training_mode` has `Bf16Full` but `make_tensor` in `backend.rs:92` always emits `DType::F32`. No autocast, no loss scaler.  
**Unsloth:** native BF16 throughout with `torch.cuda.amp`; achieves 2× throughput on RDNA3+.

---

### 🔴 DEFICIENCY — No real LoRA/QLoRA weight loading

`LoRAInjectionRegistry::standard_qlora` initialises random LoRA A/B matrices; it does not load adapter weights from disk or from a model file.  
No merge path (adapter → base model weight folding) exists.

---

### 🔴 DEFICIENCY — No evaluation loop / validation split

No `eval_loss`, `perplexity`, or held-out dataset evaluation. Metric surface is `{step, loss, tokens}` only.  
**Axolotl:** built-in eval loop with configurable `eval_steps`; writes eval metrics to W&B / MLflow.

---

## 2. ROCm / GPU Integration

### 🟢 WIN — Native ROCm-first backend selection chain

`backend.rs:select_backend()` → `ROCm → CUDA → Vulkan → Metal → CPU` — ROCm is tried first, not as an afterthought. Neither Unsloth nor Axolotl implements this kind of explicit multi-backend probe.

### 🟢 WIN — Real HIP device probe at job-start time

`probe_rocm()` calls `grim_backend_rocm::RocmDevice::probe()` before committing to a backend; fallthrough to next tier is clean and logged. Not silent degradation.

### 🟢 WIN — rocm_fusion_rmsnorm_matmul / rocm_fusion_qkv_attention toggles

Per-job fusion flags are surfaced in the UI and stored in `TrainingJob` — an explicit ROCm-tuning control surface no Python framework exposes at this granularity.

### 🔴 DEFICIENCY — A.0 kernel collision breaks every HIPRTC dispatch

From `grim_corrections.md` A.0 (confirmed, hardware-verified):

```
error: redefinition of 'fp16_to_float_device'
error: redefinition of 'dequant_q4k_element'
```

`compute_kernel_source()` concatenates 21 HIP modules into one HIPRTC TU; 4 symbols collide. **Every GPU kernel dispatch fails on every gfx target until fixed.** Training worker calls `make_tensor` → `from_cpu` on the ROCm backend — if any downstream kernel dispatch fires, HIPRTC bombs.

### 🔴 DEFICIENCY — C.1: 5 ROCm ops unreachable via `BackendDevice` trait

`selective_scan`, `flash_attention`, `cross_attention`, `rwkv_time_mix`, `rwkv_channel_mix` exist as `impl RocmDevice` inherent methods but are **outside** `impl BackendDevice for RocmDevice`. All callers using `dyn BackendDevice` silently fall through to `Err(Unimplemented)`.

### 🔴 DEFICIENCY — No multi-GPU (RCCL all-reduce) support in training

`backend.rs` hardcodes device ordinal `0`; no RCCL ring, no tensor parallelism, no data-parallel training.  
**Unsloth:** multi-GPU via `torchrun`; Axolotl: `deepspeed` ZeRO.

### 🔴 DEFICIENCY — No fp8/bf16 quantized training kernels for CDNA/RDNA3+

`grim-quant` has FP8 block quant but no GEMM kernel that uses it during the forward pass. MI300X fp8 GEMM (hipBLASLt) is unused.  
**Unsloth** has `fp8` training path specifically for H100/MI300.

---

## 3. Fine-Tuning Method Coverage

### 🟡 TIE — Training mode surface (LoRA, QLoRA, BF16, DPO, ORPO, GRPO)

`TrainingMode` enum covers the same 6 modes Axolotl's YAML config exposes. Mode switching works in `run_training_worker` (autograd dispatches correctly per mode). **However**, the worker is simulated, so coverage is nominal not functional.

### 🔴 DEFICIENCY — No PPO / RLHF reward model path

Unsloth and Axolotl both expose PPO. Grim has no reward model abstraction.

### 🔴 DEFICIENCY — No KTO (Kahneman-Tversky Optimization)

KTO is Axolotl's newest RLHF method; not in grim.

### 🔴 DEFICIENCY — No DoRA, LoRA+, or rank-stabilized LoRA variants

Unsloth implements DoRA (Weight-Decomposed LoRA) and rsLoRA. Grim has only vanilla LoRA/QLoRA.

---

## 4. Model Coverage

### 🔴 DEFICIENCY — Only Llama/Mistral dense transformer

`grim-models-transformer` is the only model in the training path. Mamba, audio, vision, and diffusion crates exist for inference but are not wired into the garage training loop.

**Axolotl** supports Llama, Mistral, Mixtral, Phi, Falcon, Qwen, CodeLlama, Gemma, and more.  
**Unsloth** has hand-unrolled kernels for Llama3, Qwen2, Gemma2, Phi3, Mistral.

### 🔴 DEFICIENCY — No MoE (Mixture-of-Experts) training support

Mixtral, DeepSeek-V3-style MoE — neither model crate nor training loop handles expert routing.

---

## 5. Data Pipeline

### 🟡 TIE — File format discovery

`discovery.rs` correctly classifies `.jsonl`, `.parquet`, `.json`, `.safetensors`, `.gguf`, FP8/FP4/MXFP4 extensions. This is table-stakes, but the coverage is solid and recursive.

### 🔴 DEFICIENCY — No HuggingFace Hub dataset pull

Unsloth/Axolotl both integrate `datasets.load_dataset()`. Grim discovery is local-only.

### 🔴 DEFICIENCY — No chat-template application in data pipeline

Axolotl's `chat_template` config key applies tokenizer chat templates at dataset-processing time. Grim has no equivalent; raw JSONL goes to the worker unformatted.

### 🔴 DEFICIENCY — No sample packing / sequence packing

Context-window filling via sample packing eliminates padding waste. Key for throughput on ROCm.  
Neither simulated nor planned in grim-garage.

---

## 6. Optimizer & Learning Rate Schedule

### 🟡 TIE — AdamW with configurable LR

`AdamW` in `grim-autograd` supports `lr`, `weight_decay`, `beta1`, `beta2`, `eps` (defaults). Comparable to Axolotl's default optimizer config.

### 🔴 DEFICIENCY — No LR scheduler (cosine, linear, warmup)

`run_training_worker` uses a flat LR from step 0 to the end. No warmup, no cosine decay.  
Both competitors default to cosine-with-warmup.

### 🔴 DEFICIENCY — No 8-bit Adam / paged Adam

Unsloth uses `bitsandbytes` 8-bit Adam, halving optimizer memory. Grim has no equivalent.

---

## 7. Checkpointing & Experiment Tracking

### 🟢 WIN — `.train` sidecar format (non-destructive adapter storage)

`train_state.write(&sidecar_path)` writes optimizer state + LoRA weights adjacent to the base model without modifying it. Clean separation; Unsloth merges adapters in-place by default.

### 🔴 DEFICIENCY — No checkpoint-resume support

If a job is cancelled or fails after epoch 1, there is no `resume_from_checkpoint` path. The worker starts fresh on every `POST /api/train/start`.

### 🔴 DEFICIENCY — No W&B / MLflow / TensorBoard integration

`Metric { step, loss, tokens }` is SSE-streamed to the UI only. No experiment tracking integrations.  
**Axolotl** natively emits to W&B, MLflow, and Comet.

### 🔴 DEFICIENCY — Metrics surface is too thin

`step`, `loss`, `tokens` — no `grad_norm`, `lr`, `epoch`, `samples_per_second`, `vram_used`, `eval_loss`.

---

## 8. API & UX Surface

### 🟢 WIN — OpenAI-compatible API + local-first architecture

Grim-garage is a **Rust binary** — no Python, no pip, no virtual env. Single-binary deploy.  
Axolotl requires Python 3.10+, torch 2.x, flash-attn build, bitsandbytes. Unsloth same.

### 🟢 WIN — Bolt-on adapter attach/detach API

`GET/POST/DELETE /api/models/{id}/bolt-ons` — non-destructive LoRA attachment to running model without restart. Neither competitor has this concept.

### 🟢 WIN — SSE live metrics stream with terminal event guarantee

`update_status_and_broadcast` ensures SSE subscribers get a terminal event on `Completed`/`Failed`/`Cancelled` — no polling required. Clean design.

### 🟢 WIN — Atomic cancel with preserved terminal status

`request_cancel` under a single write lock: signals `CancellationToken`, updates status only if still `Pending|Running`. Completed jobs preserve their real status. Correct.

### 🟢 WIN — Path-traversal protection on all job paths

`validate_job_path` and `sanitize_model_id` reject `..`, `/`, `\\` on all user-controlled path fields. Fixed in M1; Axolotl has had CVEs in this space.

### 🟡 TIE — Backend probe UI (`/api/backends`)

`probe_all()` gives a per-tier live/dead report. Axolotl has no equivalent; Unsloth has `FastLanguageModel.is_loaded_in_4bit()` type checks — coarser.

### 🔴 DEFICIENCY — validate_job_path rejects absolute paths: over-restrictive

`value.contains('/')` rejects all absolute paths. On Linux, model files live at `/home/...` or `/opt/...`. The validation is correct against traversal but forces relative-only paths, making the UI unusable for standard model directories.

---

## 9. Security & Correctness

### 🟢 WIN — `lora_rank == 0` and QLoRA ceiling enforced at API layer

`LoraRank::new(0)` → `400 Bad Request` before worker spawns. Axolotl silently starts and OOMs.

### 🟢 WIN — Duplicate job ID rejection

`insert_with_id` returns `Err(Duplicate)` rather than silently overwriting.

### 🔴 DEFICIENCY — No rate limiting on `POST /api/train/start`

Any caller can spawn arbitrarily many workers concurrently. No max-concurrent-jobs guard.

### 🔴 DEFICIENCY — `steps_per_epoch` hardcoded to 10

Simulated training produces meaningless step counts. A real epoch should be `ceil(dataset_tokens / seq_len / batch_size)`.

---

## 10. Testing

### 🟢 WIN — Mutation-resistant golden tests for fallback numerics

`fallback_tests` module (`jobs.rs:682–779`) pins exact f64 values via hand-derived formulas. Mutation-resistant by design.

### 🟢 WIN — `snapshot()` race-fix for list/get N+1 pattern

Single read-lock snapshot eliminates the TOCTOU race between `list()` and per-job `get()` that produced ghost job cards. Documented in code.

### 🟡 TIE — Integration test coverage

`tests/integration.rs` exists (23 KB); `tests/poller.rs`, `tests/view_model.rs`, `tests/backend_selection.rs` all present. Comparable to Axolotl's unit test footprint.

### 🔴 DEFICIENCY — No end-to-end training correctness test

No test verifies that a full training run on a toy dataset produces a measurably lower loss than the start. All tests verify plumbing, not outcome.

---

## Priority Improvement Roadmap (ROCm Supremacy)

### P0 — Unblock real GPU dispatch (prerequisite for everything)

1. **Fix A.0:** Split `compute_kernel_source()` — compile each kernel group into its own HIPRTC TU; deduplicate shared device-function headers via a single included `common.hip.h`. Eliminates the 4 symbol collisions. Without this, no GPU kernel fires.
2. **Fix C.1:** Move `selective_scan`, `flash_attention`, `cross_attention`, `rwkv_time_mix`, `rwkv_channel_mix` inside `impl BackendDevice for RocmDevice`. One-line move each.

### P1 — Real training loop (closes 80% of the Unsloth/Axolotl gap)

3. **Real forward pass:** wire `grim-models-transformer`'s `CausalLm::forward()` into `run_training_worker`. Replace the constant-tensor mock with actual model forward.
4. **Real dataloader:** implement a `JsonlBatchIterator` that reads `.jsonl` → tokenizes via `GgufTokenizer` → pads/packs to `seq_len` → yields `(input_ids, labels)`.
5. **Gradient accumulation:** add `accumulation_steps: u32` to `TrainingJob`; only call `optimizer.step()` after N micro-steps.
6. **BF16 autocast:** `DType::BF16` throughout the forward pass; keep optimizer state in F32. Doubles throughput on RDNA3 (gfx1100).

### P2 — ROCm-specific wins (differentiate from Python frameworks)

7. **hipBLASLt FP8 GEMM for gfx1200+ (MI300X):** plumb FP8 dtype through `from_cpu` → `grim-backend-rocm`'s GEMM dispatch. Unsloth doesn't have this on ROCm yet.
8. **HIP graph capture for the training loop:** capture the forward+backward graph after warmup steps; subsequent steps replay without kernel-launch overhead. ~20% throughput gain.
9. **RCCL data-parallel training:** `N×RocmDevice` → ring all-reduce of gradients between optimizer steps. Prerequisite: fix ordinal-0 hardcode in `backend.rs:150–156`.
10. **ROCm-specific flash attention kernel:** `flash_attention` exists on ROCm but is unreachable (C.1). After P0 fix: it becomes the default for `seq_len > 512`.

### P3 — Parity with Axolotl feature set

11. **LR scheduler:** cosine-with-warmup. Needs `step_count` + `total_steps` + `warmup_steps` in `TrainingJob`; LR = `schedule(step)` called before `optimizer.step()`.
12. **Checkpoint resume:** serialize optimizer state + LoRA adapter per epoch; `resume_from_checkpoint` field in `StartTrainingRequest`.
13. **Richer metrics:** add `grad_norm`, `lr`, `epoch`, `vram_used_mb`, `samples_per_sec` to `Metric`. Already streamed via SSE; consumers get them for free.
14. **Fix `validate_job_path` to allow absolute paths:** permit leading `/`; reject only `..` traversal components. Keeps security, removes UX friction.
15. **Max concurrent jobs guard:** add `max_concurrent: usize` to `JobRegistry`; return `429 Too Many Requests` when limit reached.

### P4 — Ecosystem pull (closes HuggingFace Hub gap)

16. **HuggingFace Hub dataset download:** `GET /api/datasets/hub?repo=...` → stream JSONL into `GRIM_DATASETS_DIR`. Ponytail: call `hf_hub_download` via a thin subprocess wrapper; no Python dependency.
17. **W&B / MLflow telemetry:** optional `GRIM_WANDB_KEY` → HTTP POST metrics to W&B API per step. ~50 lines; no SDK needed.

---

## Summary Scoreboard

| Dimension | Grim | Unsloth | Axolotl |
|-----------|------|---------|---------|
| Real training loop | ❌ (simulated) | ✅ | ✅ |
| ROCm-first backend | ✅ (best in class) | ⚠️ (CUDA-primary) | ⚠️ (CUDA-primary) |
| GPU kernel dispatch (today) | ❌ (A.0 blocker) | ✅ | ✅ |
| LoRA/QLoRA | ⚠️ (plumbing exists, not real) | ✅ | ✅ |
| DPO/ORPO/GRPO | ⚠️ (simulated) | ✅ | ✅ |
| BF16 AMP | ❌ | ✅ | ✅ |
| Gradient accumulation | ❌ | ✅ | ✅ |
| Real dataloader | ❌ | ✅ | ✅ |
| LR scheduler | ❌ | ✅ | ✅ |
| Multi-GPU | ❌ | ✅ | ✅ (DeepSpeed) |
| FP8 training (MI300X) | ❌ | ⚠️ (CUDA only) | ❌ |
| HIP graph capture | ❌ | ❌ | ❌ |
| Bolt-on adapter API | ✅ (unique) | ❌ | ❌ |
| Single-binary / no Python | ✅ (unique) | ❌ | ❌ |
| SSE live metrics stream | ✅ | ❌ | ❌ |
| Atomic cancel + status | ✅ | ❌ | ❌ |
| Path traversal protection | ✅ | ❌ | ⚠️ |
| Mutation-resistant tests | ✅ | ❌ | ❌ |
| Experiment tracking (W&B) | ❌ | ✅ | ✅ |
| DoRA / rsLoRA | ❌ | ✅ | ✅ |
| Model coverage breadth | ❌ (Llama only) | ✅ | ✅ |

**Wins:** 9  **Ties:** 1  **Deficiencies:** 20+

**Critical path to ROCm supremacy:** P0 (A.0 fix) → P1 (real loop) → P2 (hipBLASLt FP8 + graph capture) → announce. Those 3 phases make grim the only native-Rust, ROCm-first training system with FP8 and HIP graph capture — a gap neither Unsloth nor Axolotl can close quickly.
