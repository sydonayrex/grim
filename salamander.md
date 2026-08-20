# Salamander — Grim as a First-Class Training System

> Review + audit + plan for making **Grim** a first-class LLM fine-tuning
> system, ROCm-first, optimized for consumer AMD GPUs and APUs in the
> **RDNA/XDNA family** (RDNA2 `gfx1036`, RDNA3 `gfx1100/1101/1102`, RDNA4
> `gfx1200/1201`; Ryzen AI APUs with unified memory; XDNA NPUs scoped out —
> see §9).

**Date:** 2026-08-20
**Method:** direct audit of `grim/` source (`train.rs`, `grim-autograd`,
`grim-backend-rocm`, `grim-nn`, `grim-engine`, `grim-quant`) + direct review
of `old/repos/unsloth-main`, `old/repos/LlamaFactory-main`,
`old/repos/axolotl-main`. Auto-inventory agents on the large Python repos
degraded and were abandoned in favor of targeted reads (evidence over process).
**Skills applied:** caveman (terse verdicts), rust-ffi-grim (ABI/discovery),
rocm-hip / rocm-kernels / amd-kernel-optimization / hip (ROCm constraints),
rust-ml-llm-architecture (device-trait isolation + `.grim` metadata),
llm-training (distillation/RL framing), clean-code-guard + rust-expert
(quality gates), ml-ai-project-planning (phases/metrics/non-goals).

---

## 1. Verdict (caveman)

Grim ALREADY training-capable. Not toy. `grim train` = real QLoRA SFT on
ROCm. Backward wired. Optimizers rich (Galore/Muon/Paged — ahead of some
competitors). ROCm backend deep (fused dequant GEMM, fp8 RDNA4, managed
memory).

BUT gaps block "first-class":

- **No mixed precision.** Autograd path f32-only. On 8–16 GB RDNA this = 2x
  VRAM + 2x slow vs bf16. Biggest single gap.
- **No true batched / varlen training.** Loop = one packed sequence per step.
  No multi-sequence batch, no block-diagonal attention. ROCm occupancy left
  on table.
- **No data ecosystem.** Alpaca + ShareGPT only. No chat templates, no HF
  datasets streaming, no multi-source mix/dedup.
- **No eval / logging during training.** Loss-only. No eval split, no metrics,
  no best-model tracking.
- **Multi-GPU incomplete.** RCCL comm built but no per-rank model replica.
- **Kernel coverage partial on RDNA.** rmsnorm/rope/softmax/attention backward
  still CPU-fallback on ROCm.

Plan in §7 closes these phased, small-batch, ROCm-first.

---

## 2. Grim Training Stack — Audit (with evidence)

### 2.1 What exists and is real

| Capability | Evidence | Notes |
|---|---|---|
| QLoRA SFT CLI | `crates/grim-cli/src/train.rs::cmd_train` | Alpaca + ShareGPT load, packing, grad-accum, warmup, clip, early-stop, resume sidecar |
| Adapter-only autograd tape | `grim-autograd/src/lib.rs::AutogradScope::LoRAOnly`, `tape.rs` | Records only MatMul/Add/Scale/LoRAApply — unsloth-style "base stays frozen" |
| Streaming per-layer forward (never materializes full model) | `grim-engine/src/streaming_forward.rs::forward_block_with_autograd` | Memory win analogous to unsloth's core thesis |
| Real embedding + `output_norm` + `lm_head` | `train.rs` (post WI-F4-close) | Was faked; now real |
| Fused dequant backward GEMM on ROCm | `ops.rs::matmul_backward` (line ~440) dispatches `dev.quantized_matmul_backward_dx`; `roc_device.rs::launch_fused_dequant_backward_gemm_f16` (line ~3538) | F5-close shipped. CPU fallback preserved |
| Fused linear cross-entropy (no `[B,V]` alloc) | `loss.rs::fused_linear_cross_entropy_loss` → `fused_linear_cross_entropy_forward` on ROCm | CE backward on GPU |
| Fused SiLU·Mul backward on ROCm | `ops.rs::silu_mul_backward` (line ~868) | HIP kernel `grim_silu_mul_backward` |
| Rich optimizers | `adamw.rs` `OptimizerKind`: AdamW, AdamW8Bit, PagedAdamW, AdamWBnb, PagedAdamW8Bit, **QGaLoreAdamW8Bit, GaloreAdamW, GaloreAdamW8Bit**, Lion, Lion8Bit, LionVote, MAdam, **Adafactor**, **Muon**, Scythe1, SoulEater | Galore/Apollo/Muon already present — ahead of LlamaFactory's galore/apollo/badam flags |
| LoRA variants | `injection.rs` (PiSSA, OLoRA), `ops.rs` (DoRA, VeRA), `TrainOptions::use_spectral_qlora` | SPECTRAL-QLORA = semi-orthogonal + Muon |
| Preference / alignment losses | `preference_loss.rs` (dpo, kto, orpo, simpo, grpo), `mm_grpo.rs` (multi-GPU reward norm), `contrast_omni.rs` | GRPO already in house (unsloth also has grpo) |
| Distillation / pruning | `soul_eater.rs`, `scythe1.rs`, `turbo_finetune.rs`, `tops_prune.rs`, `omnilo_prune.rs`, `omnigrad.rs` | llm-training skill framing already embedded |
| ROCm quant base (frozen) | `roc_device.rs` fused dequant GEMM for Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/IQ2*/IQ3*/IQ4* | Base stays quantized — exactly the QLoRA philosophy |
| FP8 GEMM (RDNA4) | `kernels/fp8_gemm_rdna4.rs`, `roc_device.rs::launch_..._fp8` (gfx1200+) | Native MFMA fp8 path exists |
| Memory budget + HIP managed fallback | `memory/budget.rs` (`GRIM_ROCM_VRAM_BUDGET_BYTES`, `note_managed_fallback`) | Critical for APU / VRAM-limited cards |
| Kernel autotune | `tune.rs` (`grim tune`), `autotune.rs`, `kernels/tile_picker.rs` | RDNA4 fp8 tile picker present |
| Merge / sidecar persistence | `train.rs` `--output` sidecar; `grim merge` | Bake adapter into base |

### 2.2 Gaps (ranked by impact on consumer-AMD training)

| # | Gap | Why it hurts RDNA/APU | Evidence |
|---|---|---|---|
| G1 | **No mixed precision (f32 master)** | 2x VRAM, ~2x slower matmul vs bf16/fp16 on 8–16 GB cards | `param.rs:111` `DType::F32`; every backward op `to_vec_f32()` (`ops.rs` rmsnorm/rope/softmax/embedding backward) |
| G2 | **No batched / varlen training** | One sequence/step = low ROCm occupancy; padding waste | `train.rs::cmd_train` loop iterates `dataset` one `(tokens,labels)` at a time; `batch_size` = pack length |
| G3 | **RMSNorm/RoPE/Softmax/Attn backward CPU-fallback on ROCm** | Host round-trips kill throughput | `ops.rs` `rmsnorm_backward`/`rope_backward`/`softmax_backward`/`embedding_backward` compute in f32 then `dev.from_cpu` |
| G4 | **No data ecosystem** | Manual Alpaca/ShareGPT only; no chat templates, no HF streaming, no mix/dedup | `train.rs::load_dataset` (two formats, fixed template) |
| G5 | **No eval / logging / best-model** | Blind training; no quality gate | `train.rs` streams loss only; no eval split |
| G6 | **Multi-GPU incomplete** | RCCL all-reduce over single in-process replica = incorrect/partial | `train.rs` builds `RcclAllReduce` but loads model once on one ordinal; no per-rank copy |
| G7 | **Thin LR schedulers** | Only `CosineWarmupSchedule` | `lr_schedule.rs`, `LRScheduler` enum |
| G8 | **No seeds / determinism** | Non-reproducible runs | `train.rs` has no RNG seed |
| G9 | **RDNA2/3 kernel coverage unverified** | `gfx1036`/`gfx1100` consumer majority under-tested vs `gfx1200` | tiling tuned for RDNA4; verify Wave32 occupancy |
| G10 | **XDNA (NPU) not addressable** | Different runtime/ISA; needs separate plan | no `xnpu`/`xnack` handling in `grim-backend-rocm` |

---

## 3. Competitor Review — unsloth (`old/repos/unsloth-main`)

**Claim:** 2x faster, up to 80% less VRAM vs HF; official AMD/ROCm support.

| Feature | File | Relevance to grim |
|---|---|---|
| `FastLanguageModel.from_pretrained` (4bit nf4/fp4, `device_map`, bf16) | `models/loader.py::from_pretrained` | grim uses GGUF Q4_K frozen base — same philosophy, native not bnb |
| "Unsloth" gradient checkpointing (non-offloaded) | `models/loader.py::apply_unsloth_gradient_checkpointing` | grim streaming recompute already covers this; keep |
| Fast LoRA linear patching (bf16 compute, no backward-through-dequant) | `models/_patch_linear.py` | mirrored by grim `fused dequant backward GEMM` |
| Custom Triton/CUDA kernels: `rms_layernorm`, `rope_embedding`, `swiglu`, `fast_lora`, `cross_entropy_loss`, `geglu`, `fp8`, `flex_attention` | `kernels/` | **Port RMSNorm/RoPE/SwiGLU/CE to ROCm-HIP** (G3). rocm-kernels skill has ready Triton→HIP patterns (manual tanh, `next_power_of_2` BLOCK, `tl.minimum`) |
| Embedding/head VRAM trick | (bf16 LoRA in f32 math) | grim: add bf16 LoRA param storage (G1) |
| Sequence packing / padding-free | `utils/packing.py` (`configure_sample_packing`, `enable_padding_free_metadata`, block-mask builders) | **Adopt varlen/sample packing** (G2) |
| GRPO / DPO | `models/rl.py`, `dpo.py`, `trainer.py` | grim already has GRPO+DPO; parity |
| **ROCm init patches** | `_gpu_init.py` (`fix_bitsandbytes_rocm_arch_detection`, `maybe_set_windows_rocm_bnb_version`, `is_hip` branches, `DEVICE_TYPE=="hip"`) | **Reference for grim's ROCm detection** (rust-ffi-grim dynamic discovery + gfx probe) |
| AMD wheels / guide | `pyproject.toml`, README "AMD: Training, RL, chat and deployment work on Windows, WSL and Linux" | Confirms ROCm-first is viable & expected |

**Takeaway:** unsloth's *philosophy* (frozen quantized base + fused kernels +
bf16 + packing) already matches grim's design. The portable, high-value
additions are **bf16 mixed precision (G1)** and **ROCm fused elementwise
kernels (G3)** + **packing (G2)**.

---

## 4. Competitor Review — LlamaFactory (`old/repos/LlamaFactory-main`)

**Breadth leader.** Stages: `pt, sft, rm, ppo, dpo, kto` + `mca` (Megatron
adapter) + `hyper_parallel` (sequence/ZeRO3). Finetuning types: `lora, oft,
freeze, full` + `galore, apollo, badam, boft, qa_lora, pissa, dora`.

| Feature | File | Relevance |
|---|---|---|
| 20+ data formatters + jinja chat templates + multimodal plugin | `data/processor/*`, `data/template.py`, `data/mm_plugin.py`, `data/formatter.py` | **G4**: dataset abstraction + chat-template engine |
| Eval: `eval_steps`, predict, ROUGE/BLEU metrics | `train/sft/metric.py`, `hparams/*.py` | **G5**: eval harness |
| Quant loading: BNB 4/8bit, GPTQ, AWQ, AQLM, EETQ, FP8 | `model/model_utils/quantization.py`, `hparams/parser.py` | grim native GGUF; add FP8 train path + GPTQ/AWQ load |
| `neat_packing` / `block_diag_attn` (padding-free SFT) | `hparams/data_args.py`, `parser.py` | **G2**: block-diagonal attention |
| Schedulers: cosine/linear/constant_warmup/polynomial/inverse_sqrt + `warmup_ratio` | transformers `TrainingArguments` | **G7**: expand scheduler set |
| WebUI (LlamaBoard) | `src/llamafactory/webui/` | grim-garage already the dashboard — extend for training control |
| `use_ref_model` for DPO (ORPO/SimPO skip ref) | `finetuning_args.py` | grim already ORPO/SimPO/DPO |

**Takeaway:** grim already covers most *methods* (lora/full/dora/galore/
apollo-ish/pissa/dpo/kto/orpo/simpo/grpo). Gaps vs LlamaFactory are
**data/templates (G4)**, **eval (G5)**, **broader method flags (OFT/BOFT/
QaLoRA/BAdam)**, and **real ZeRO/FSDP multi-GPU**.

---

## 5. Competitor Review — axolotl (`old/repos/axolotl-main`)

**Config-driven + dataset-pipeline leader** (YAML single-source).

| Feature | File | Relevance |
|---|---|---|
| YAML config drives whole run | `utils/schemas/config.py` | **Add `training.toml`/`training.yaml` mode** to grim (serde; pure-Rust ergonomics) |
| 20+ prompt strategies + loss masking | `prompt_strategies/*` (alpaca, sharegpt, chat_template, completion, dpo, kto, orpo, pretrain, messages…) | **G4**: strategy registry |
| `sample_packing` (block-diag attn), `sample_packing_group_size`, sequentially | `schemas/config.py` | **G2** |
| `relora` (reset LoRA mid-train), `use_dora`, `loraplus`, `qlora` | `schemas/config.py`, `monkeypatch/` | Add ReLoRA, LoRA+, MiLoRA |
| DeepSpeed/FSDP/liger-kernel/flash-attn | `loaders/`, `kernels/` | ROCm equivalents: grim's own kernels + aiter-style dispatch |
| `merge_lora` | `utils/` | grim `merge` exists |
| Streaming HF datasets, dedup, mix | `datasets.py`, `utils/data/shared.py` | **G4**: multi-source mixing |
| wandb/tensorboard, eval during training, `saves_per_epoch` | `schemas/config.py`, `integrations/` | **G5** |

**Takeaway:** axolotl's value is **config schema + dataset pipeline + packing
+ ReLoRA/DoRA/LoRA+**. Port the *logic* (not the Python): TOML config, prompt
strategy registry, sample packing, ReLoRA/LoRA+.

---

## 6. Competitive Feature → Grim Fit Matrix

| Competitor feature | grim today | Port? | ROCm/consumer fit |
|---|---|---|---|
| Frozen quantized base + fused dequant backward | ✅ done | keep | ✅ core |
| bf16/fp16 mixed precision | ❌ | **YES (G1)** | ✅ 2x VRAM/speed on RDNA |
| ROCm fused RMSNorm/RoPE/SwiGLU/CE kernels | partial (CE, SiLU·Mul only) | **YES (G3)** | ✅ Triton→HIP patterns exist |
| Sample / varlen packing (block-diag attn) | ❌ (concat pack only) | **YES (G2)** | ✅ occupancy win on small RDNA |
| Chat templates + 20+ formatters | ❌ | YES (G4) | ✅ enables real instruct SFT |
| HF datasets streaming / mix / dedup | ❌ | YES (G4) | ✅ leverages system RAM on APU |
| Eval split + metrics + best-model | ❌ | YES (G5) | ✅ quality gate |
| TOML/YAML config | ❌ (flags only) | YES | ✅ ergonomics (axolotl-style) |
| OFT/BOFT, QaLoRA, BAdam, ReLoRA, MiLoRA, LoRA+ | partial (DoRA/VeRA/PiSSA/OLoRA/Spectral) | partial YES | ✅ memory-saving methods |
| ZeRO/FSDP multi-GPU | ❌ (RCCL stub) | YES (G6) | ✅ APU+iGPU or multi-dGPU |
| FP8 training (gfx1200+) | gemm only | YES | ✅ RDNA4 9070 |
| GRPO / DPO / KTO / ORPO / SimPO | ✅ | keep | ✅ in-house |
| Distillation (SoulEater/Scythe1) | ✅ | keep | ✅ unique to grim |

---

## 7. Plan — Phased, ROCm-First, Small-Batch

Each phase ships behind a green test gate. Phases are cumulative; do not skip
G1 (it unblocks VRAM for everything else on consumer cards).

### Phase 0 — Baseline & instrumentation (do first)
- **P0.1** Add `training_correctness` test: overfit a tiny dataset on CPU,
  assert loss decreases. Gate on the WI-F4 smoke test (real embeddings +
  real logits now exist). Reuse `grim-autograd` tests + `quant_backward_audit`.
- **P0.2** Add `--seed` to `TrainOptions` + deterministic init path (G8).
- **P0.3** Emit training telemetry to **grim-garage** (loss, LR, grad-norm,
  step/s, VRAM) — dashboard already exists; wire the training events.

### Phase 1 — Mixed precision (bf16/fp16 master, fp32 moments)  ← biggest win
- **P1.1** Extend `DType`/`Storage` so a trainable master param carries
  `bf16`/`f16` while optimizer moments stay `f32` (PagedAdamW already
  page-offs host RAM — combine for max fit).
- **P1.2** Make autograd backward honor param dtype: GEMM in bf16 on ROCm,
  moment update in f32. Keep CPU path f32 (CI).
- **P1.3** Numerics: per-format tolerance vs f32 reference; **fail-then-pass**
  discipline (corrupt a scale byte, confirm test catches it) — extend
  `quant_backward_audit.rs`.
- **Gate:** toy-overfit passes (CPU + GPU-gated); bf16 trainable params fit
  ~2x more layers than f32 on the same card.

### Phase 2 — Batched varlen training + sample packing (G2)
- **P2.1** Add a real batch dim: `[B, T]` forward with `cu_seqlens`
  (block-diagonal / varlen attention) — mirroring LlamaFactory
  `neat_packing`/`block_diag_attn` and unsloth `configure_sample_packing`.
- **P2.2** Extend grim's existing `pack_dataset_tokens` to emit attention
  masks + loss masks (no cross-example leakage).
- **P2.3** Training attention backward on ROCm with varlen masking (not
  recompute-full). Reuse `kernels/qkv_attention.rs`; add backward.
- **Gate:** packed multi-seq batch == sum of single-seq runs (loss, within
  tolerance); ROCm occupancy (waves/ CU) improves vs single-seq.

### Phase 3 — ROCm fused elementwise kernels (G3)
- **P3.1** Port `rmsnorm_backward`, `rope_backward`, `softmax_backward`,
  `embedding_backward` to ROCm-HIP (today CPU-fallback). Use rocm-kernels
  skill patterns: `next_power_of_2(BLOCK_D)` (never autotune BLOCK_D),
  manual tanh, `tl.minimum/maximum`, no `tl.libdevice`.
- **P3.2** Tune for **Wave32** (RDNA) — verify `num_warps` heuristic; add to
  `tune.rs` autotune map keyed by `gfx10xx/gfx11xx/gfx12xx`.
- **Gate:** GPU-vs-CPU parity for each (fail-then-pass); `rocprof` bandwidth
  util reported.

### Phase 4 — Data + eval ecosystem (G4, G5)
- **P4.1** Dataset abstraction: local JSONL/parquet + optional HF datasets
  streaming; multi-source mix + dedup. Serde `TrainingConfig` TOML/YAML
  (axolotl-style).
- **P4.2** Chat-template engine (jinja subset) + strategy registry
  (alpaca/sharegpt/chat_template/completion/dpo/kto/orpo/pretrain…).
- **P4.3** Eval split + metrics (loss, ppl, ROUGE/BLEU on gen) + best-model
  checkpoint tracking + tensorboard/grim-garage logging.

### Phase 5 — Method breadth + multi-GPU + FP8
- **P5.1** Add OFT/BOFT, QaLoRA, BAdam (layer-wise), ReLoRA, MiLoRA,
  LoRA+ (diff lr A/B). RM + PPO stages (grim already has GRPO/DPO/KTO).
- **P5.2** FP8 training path for `gfx1200+` (master fp8, moments f32) — lever
  existing `fp8_gemm_rdna4`.
- **P5.3** Real multi-GPU: per-rank model replica + RCCL all-reduce (fix G6);
  single-process multi-device path for **APU iGPU + dGPU**.

### Phase 6 — RDNA consumer coverage + XDNA scoping (G9, G10)
- **P6.1** Verify kernel coverage & Wave32 occupancy on `gfx1036` (RDNA2),
  `gfx1100/1101/1102` (RDNA3), `gfx1200/1201` (RDNA4). Add to `tune.rs`.
- **P6.2** Document XDNA (Ryzen AI NPU) as **out of scope for training**
  (separate runtime/ISA); keep ROCm iGPU (RDNA) path as the APU story.

---

## 8. RDNA / XDNA Optimization Notes (AMD-specific)

- **Wavefront:** RDNA is **Wave32** (CDNA is Wave64). ROCm kernel `num_warps`
  and LDS budgeting differ — confirm against `rocm-kernels` R9700 findings
  (RMSNorm 2.9x, AdaLN 3.0x on RDNA4 via Triton; grim's HIP JIT should
  target similar).
- **APU / unified memory:** Ryzen AI shares system RAM. grim's
  `GRIM_ROCM_VRAM_BUDGET_BYTES` + HIP managed fallback (`budget.rs`) already
  handle oversubscription. Make APU first-class: bf16 + Q4 frozen base to fit
  16–32 GB; PagedAdamW offloads moments to host.
- **Small-batch matrices:** consumer decode/train shapes are small-`m`. Use
  two-tier MatrixCore (16×16 `rocWMMA` vs raw `mfma` intrinsics) per
  `grim_rocm_consumer_perf_plan_v3` (14–18.6x over rocBLAS at tiny `m`).
- **FP8:** `gfx1200+` only (`fp8` valid when `target_gfx >= gfx1200`, per
  `rust-ml-llm-architecture`). Tag `.grim` with `fp8` + `gemm_backend`.
- **Tag `.grim` metadata** (rust-ml-llm-architecture): `target_gfx`,
  `preferred_dtype`, `fp8`, `gemm_backend`, `kv_cache_layout`,
  `multi_gpu.strategy` — server picks fastest valid path without probing.

---

## 9. Non-Goals (explicit)

- **XDNA NPU training** — different runtime/ISA; not in ROCm path.
- **Reimplementing transformers/peft/trl** — grim is pure Rust; borrow
  *logic* (formulas, packing, schedulers), not the ecosystem.
- **Full MoE / speculative training** this cycle — inference-side only.
- **Windows WSL ROCm** parity testing beyond what CI can run.

---

## 10. Code-Quality Gates (clean-code-guard + rust-expert + rust-ffi-grim)

Applied to every Phase deliverable:

1. **No `unwrap()`/`expect()`** in training/ROCm path outside `#[cfg(test)]`;
   map ROCm errors to `Result` (clean-code 15; rust-expert verification list).
2. **FFI safety** (rust-ffi-grim): `#[repr(C)]` for any struct crossing the
   ROCm `.so` boundary; verify symbols non-null; **panic must never cross
   FFI**; dynamic ROCm discovery with graceful fallback (reference unsloth
   `_gpu_init.py` ROCm branch, ported to Rust).
3. **Names reveal intent**; functions ≤20 lines; no copy-from-similar —
   **re-derive** backward kernels from spec (clean-code 1, 19). Off-by-one in
   gradient kernels enters via copy-paste.
4. **Never disable/skip a test to pass** (clean-code 18). Gate GPU tests with
   `GPU_TEST_ENV` (existing convention in `quant_backward_audit`).
5. **"hipify compiles" ≠ correct** (rocm-hip default posture). Run `rocprof`
   + bench with warmup/iterations (amd-kernel-optimization): 3 warmup, 10
   iters, report mean±std.
6. **Backend isolation** (rust-ml-llm-architecture): keep HIP/rocBLAS behind
   `BackendDevice`; kernel bodies in `grim-backend-rocm`; core math
   backend-agnostic.
7. **cargo fmt + clippy -p grim-autograd -p grim-cli -p grim-backend-rocm**
   clean per phase; reuse the standing `rust-clippy.yml` CI.

---

## 11. Verification Plan (concrete)

| Level | Test | Gate |
|---|---|---|
| Unit | Backward kernel GPU-vs-CPU parity (rmsnorm/rope/softmax/attn + existing dequant/CE) | fail-then-pass; tolerance per dtype |
| Integration | Toy overfit, loss decreases (CPU; ROCm gated) | WI-F4 smoke gate |
| Numerics | bf16/fp16 trainable vs f32 reference (P1.3) | per-format tolerance |
| Perf | `rocprof` bandwidth util for training GEMMs on gfx1100/gfx1200 | mean±std, warmup; ≥ target util |
| Fit | Max model size at fixed VRAM: bf16 vs f32 (P1) | ~2x layer count |
| Correctness | Packed multi-seq batch == summed single-seq (P2.2) | within tolerance, no cross-example leak |
| CI | Add `training-correctness` (CPU) + GPU-gated job; keep `rust-clippy` | green |

---

## 12. Suggested Order (summary)

P0 (baseline) → **P1 (bf16 — unblocks VRAM)** → P2 (batched varlen + packing)
→ P3 (ROCm fused elementwise kernels) → P4 (data + eval) → P5 (methods +
multi-GPU + fp8) → P6 (RDNA coverage + XDNA scope).

Grim is closer to "first-class" than the competitor repos assume a Rust
engine could be: the autograd/optimizer/ROCm substrate is real. The work is
**precision, batching, data, eval, and kernel coverage on consumer RDNA** —
not a rewrite.
