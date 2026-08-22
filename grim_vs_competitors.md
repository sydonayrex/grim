# grim vs. Unsloth, Ollama, vLLM, SGLang, LlamaFactory, Axolotl

**Method:** grim's 24 uploaded crates were read directly (no docs-only inference). All six competitors were shallow-cloned from their public GitHub repos and their claims grep/read-verified against source, not README marketing copy. Where a claim could not be verified from what's here, it's marked `UNVERIFIED`.

---

## 1. What each tool actually is

| Tool | Core identity | Inference engine origin |
|---|---|---|
| **grim** | Pure-Rust, from-scratch inference + fine-tuning engine, AMD ROCm-primary | Native, written from scratch |
| **Ollama** | Model management/serving wrapper | Vendors a `ggml`/llama.cpp fork (`ml/backend/ggml/ggml/src/ggml-cuda/...`) with Go orchestration on top — confirmed in source, not a from-scratch engine |
| **vLLM** | Production serving engine (PagedAttention) | Native, written from scratch (Python/C++/CUDA/HIP) |
| **SGLang** | Production serving engine (RadixAttention) | Native, written from scratch (Python/C++/CUDA/HIP) |
| **Unsloth** | Fine-tuning acceleration library | Wraps/patches HF `transformers`/`peft`/`trl`, not an inference server |
| **LlamaFactory** | Fine-tuning framework, broad method coverage | Wraps HF `transformers`/`trl`, unified CLI/WebUI over many training paradigms |
| **Axolotl** | Fine-tuning framework, config-driven | Wraps HF stack, YAML-driven, plugin/integration system |

grim is the only tool in this set that is simultaneously a from-scratch inference engine *and* a from-scratch (no PyTorch/no PyO3) training stack. That's a genuinely distinct position — nothing else here does both natively.

---

## 2. AMD ROCm support: the actual differentiation claim, checked

grim's standing positioning is "AMD/ROCm-first, not CUDA-first-with-ROCm-bolted-on." I checked this by counting files referencing RDNA-generation GPU targets (`gfx1030/1036/1100/1101/1102/1103/1150-1153/1200/1201` — consumer Radeon) vs. CDNA-generation targets (`gfx906/90a/942/950` — MI-series datacenter cards) in each ROCm-capable engine's source:

| Engine | RDNA-referencing files | CDNA-referencing files | Ratio |
|---|---|---|---|
| **grim** (`grim-backend-rocm`) | 54 | 18 | ~3:1 RDNA-majority |
| **vLLM** | 17 | 55 | ~1:3 CDNA-majority |
| **SGLang** | 1 | 49 | ~1:49 CDNA-only |

This is a real, structural difference, not marketing. vLLM's own docs (`docs/getting_started/installation/gpu.rocm.inc.md`) do list Radeon RX 7900 (gfx1100/1101) and RX 9000 (gfx1200/1201) as supported, and it ships real RDNA3-specific kernels (`rdna3_w4a16.py`, `compressed_tensors_moe_wna16_rdna3.py`) — so the claim "vLLM has zero RDNA support" would be false and I'm not making it. But the codebase's center of gravity is unambiguously CDNA/MI-series. SGLang's AMD support is almost entirely CDNA (gfx942/950); it has essentially no RDNA-specific code path.

grim's inversion of this ratio is the one claim in the standing positioning that source review actually substantiates with numbers.

**Depth check, not just file count**: I re-verified this isn't an artifact of counting decorative comments. Per-file hit density for grim's `grim-backend-rocm` shows real, load-bearing usage — e.g. `kernels/fp8_gemm_rdna4.rs` contains actual `#if defined(__gfx1200__)` / `#if defined(__gfx1100__)` HIPRTC preprocessor branches selecting distinct GEMM tile code per architecture generation, not just an arch string in a comment, and has zero CDNA references in that file (an RDNA-exclusive kernel). Similarly, vLLM's `rdna3_w4a16.py` (18 hits) is a dedicated, real RDNA3 weight-quantization kernel, not a stub. Both codebases' RDNA-tagged files are substantively real.

**What the ratio does and doesn't prove**: the 3:1 file-count inversion (grim RDNA-majority vs. vLLM CDNA-majority) is a genuine structural signal about where each project's *design center* sits — grim was built RDNA-out, vLLM was extended CDNA-first with RDNA support added later. It is **not** a proxy for which project's RDNA support is more mature, complete, or better-tested in absolute terms — vLLM is a far larger, more battle-tested codebase overall, and a smaller RDNA-specific file count sitting inside a much bigger, more mature serving engine could still outperform or out-cover grim's RDNA kernels on functionality grim doesn't have yet (e.g. vLLM's RDNA files benefit from PagedAttention, extensive quant-backend integration, and production hardening that grim's equivalent code doesn't have behind it). The ratio is legitimate evidence for the *positioning* claim ("AMD-first design center"), not for a *quality or completeness* claim, and the original document should not be read as implying the latter.

One caveat on the raw grim number: 52 of grim's 54 RDNA-file hits are concentrated partly in `trace.rs`, where a large share of the count is the same `"gfx1036"` literal repeated across test fixtures (Syd's dual RX 9070 XT/9060 XT rig is gfx1200-class; gfx1036 tests likely target other hardware in the support matrix) — real, but somewhat inflated relative to files with distinct kernel logic per hit.

**Ollama** doesn't have its own GPU kernels for ROCm at all — it inherits whatever llama.cpp's `ggml-hip` backend supports, which is a separate upstream project's decision, not Ollama's.

---

## 3. Serving engine feature comparison

### 3.1 Continuous batching / KV cache management

**Correction:** an earlier version of this document claimed grim has no block-table paged allocator and no prefix cache. That was wrong — it was based on grepping `grim-scheduler`, `grim-engine`, and `grim-kvquant` only and stopping. The actual implementation lives in `grim-memory`, a crate that grep pass never touched.

- **vLLM**: `PagedAttention` (`v1/attention/ops/paged_attn.py`, `chunked_prefill_paged_decode.py`) — block-table-based paged KV cache, the technique that made vLLM's throughput numbers famous.
- **SGLang**: `RadixAttention` (`srt/mem_cache/radix_cache.py`, 863 lines, plus SWA/mamba/storage-tiered variants) — trie-based automatic prefix cache sharing across requests.
- **grim**: `grim-memory` (1,792 lines across `lib.rs`, `radix.rs`, `moe_budget.rs`) implements:
  - `KvBlockPool` — fixed-size (`BLOCK_SIZE = 16`) physical block pool with a free-list allocator and refcounting (`alloc`, `free`, `free_with_tier`, `add_ref`).
  - `RadixTree` (`radix.rs`, 339 lines, 4 tests) — block-granular prefix tree, explicitly modeled on RadixAttention per its own doc comment, exposed via `match_prefix`, `insert_prefix`, `find_or_share_prefix_tokens`.
  - `PagedKvCache` — the block-table wrapper (`BlockTable` + `KvBlockPool`) with GPU↔Host↔NVMe tiering (`promote_to_gpu`, `demote_cold_prefix`, `CacheTier`, `SharedSpillManager`).
  - 15 `#[test]` functions across the crate.
  - **Confirmed wired into the live request path**, not orphaned: `grim-engine::lib.rs` constructs a `PagedKvCache` and calls `match_prefix_promoting` during actual generation (verified at the call sites, not just import lines), and `grim-server` routes all requests through `grim_engine::Engine` (verified via `use grim_engine::{Engine, model_loader}` and live `Engine::new(...)` construction in the server's request-handling code).

grim's scheduler (`grim-scheduler`, 768 + 459 lines) is a separate concern layered on top: a TTFT/ITL-aware admission controller with pause/resume and backlog-based throughput estimation (`AdmissionController::admit`, `predict_ttft`). It doesn't own block allocation itself — that's `grim-memory`'s job — but the two compose.

**Net assessment**: grim has a real, tested, wired paged-KV-cache-plus-prefix-cache system in the same functional category as vLLM's PagedAttention and SGLang's RadixAttention. I have not done a maturity/scale comparison (eviction policy sophistication, multi-request concurrent-tree correctness under load, cross-node prefix sharing, etc.) rigorous enough to claim parity or a gap in either direction — that would need a dedicated pass reading vLLM's `paged_attn.py`/block manager and SGLang's `radix_cache.py` at the same depth as `grim-memory` above. What I can say confidently is that "grim has no paging or prefix caching" is false.

### 3.2 Speculative decoding

- **vLLM**: EAGLE, Medusa, n-gram (CPU+GPU), draft-model, suffix decoding — ~49 files, CUDA-graph integrated.
- **SGLang**: EAGLE (multi-layer, disaggregated variants), n-gram, "dflash"/"dspark" custom methods — ~78 files, deeply integrated with CUDA graphs and disaggregated serving.
- **grim**: `grim-speculative` (13 files, ~2,438 lines) — confidence-head gating, entropy-based confidence, Markov-head drafting, tiny draft backbones, an MTP (multi-token-prediction) adapter for Llama, and a Mamba-specific draft path. It's real (only ~30% of functions lack `#[test]` coverage, and it's wired into `grim-engine::speculative_loop.rs`, not a dead crate) but narrower in scope — no EAGLE implementation, no CUDA/HIP-graph capture path for the spec loop specifically (though `graph_capture.rs` exists more generally in the ROCm backend).
- **Ollama**: No dedicated speculative decoding subsystem found in `server/` or `llm/` beyond what upstream llama.cpp provides.

### 3.3 Structured/constrained output — this one surprised me relative to internal notes

Prior internal notes characterized this as "confirmed gap, JSON-mode-only shippable milestone still pending." Source review shows this is **stale** — `grim-constrain` (json_fsm.rs 739 lines, schema.rs 229 lines, sampler.rs 256 lines) implements a real JSON-Schema-aware FSM constraint, and it's **wired end-to-end** into `grim-server`: `response_format.type == "json_schema"` → `Constraint::json_schema(schema)` → `ConstrainedSampler` (confirmed at `grim-server/src/lib.rs:961-974`). This is a genuine, callable feature today, not a stub.

That said, the two constraint modes are *not* equally engineered, and the gap is bigger than the code's own `TODO(perf)` comment implies:

- **`Constraint::JsonObject`** (plain JSON-mode) uses `TokenMaskCache`, a real cache keyed on `JsonState` (`grim-constrain/src/json_fsm.rs`): the FSM walk over the vocabulary happens once per *distinct FSM state* and is memoized (`HashMap<JsonState, Arc<[bool]>>`), so repeated visits to the same state are O(1). This is a reasonably well-designed cache, not a stub.
- **`Constraint::JsonSchema`** has **no caching at all**. Its `compute_mask` (`grim-constrain/src/sampler.rs`, lines 185–199) does, for every sampling step: for every token in the vocabulary, `format!("{output}{t}")` then a full `serde_json::from_str` parse plus a full recursive schema `validate()` call. This is uncached, unbounded work repeated at every decode step — not merely "not yet optimized to precompute per-token validity" as the `TODO(perf)` comment in `schema.rs` line 31 characterizes it, but the single most expensive possible implementation of that check, redone from scratch every step with no memoization structure in place at all.

So: JSON-mode is production-reasonable as-is; JSON-Schema mode is functionally correct but has no performance engineering behind it yet, which is a materially bigger gap than "hasn't precomputed per-token validity" suggests.

- **vLLM**: dedicated `config/structured_outputs.py`, `v1/worker/gpu/structured_outputs.py`, backed by xgrammar.
- **SGLang**: xgrammar backend, outlines backend, plus jump-forward decoding optimization (`outlines_jump_forward.py`) — this is exactly the kind of perf optimization grim's TODO flags as missing.
- **Ollama**: supported via its OpenAI-compatible layer (`openai/responses.go`), backed by whatever llama.cpp's grammar engine provides.

So: grim has real, working structured output — a correction upward from prior notes — but vLLM/SGLang's implementations are more mature specifically on the performance axis grim's own code admits it hasn't solved yet.

### 3.4 Disaggregated (prefill/decode split) serving — a real bug found on deep trace, not just a scope gap

- **vLLM**: ~73 files under `disagg`/`kv_transfer`, pluggable connectors (NIXL, Mooncake, etc.)
- **SGLang**: ~62 files, similarly connector-based, plus EAGLE-disaggregation variants.
- **grim**: `grim-disagg` (627 lines) + `grim-kvtransport` (1,166 lines) — real network-transfer machinery, RDMA-flaggable (`enable_rdma`), prefill/decode router (`DisaggRouter`), receiver server, proper mutex handling and error propagation (no fabricated data on connection failure). Smaller surface area than vLLM/SGLang's pluggable multi-backend systems, which is an honest scope difference. But tracing the actual data path surfaced something more serious than a scope gap:

  `grim-disagg::transfer_kv_cache_real` reads the data it sends over the network via `pool.read_keys(block_id)` / `pool.read_values(block_id)` on a `grim_memory::KvBlockPool`. Each `KvBlock` inside that pool is single-layer storage — `[BLOCK_SIZE, num_kv_heads, head_dim]` with no layer dimension (confirmed in the `KvBlock` struct definition). The only method that writes into that pool's block storage is `PagedKvCache::store_kv` — and grepping every call site of `store_kv` across the entire workspace turns up **zero callers** outside its own trait definition and grim-memory's unit tests.

  The actual live model forward pass — verified in `grim-models/transformer/src/block.rs` and `grim-models/transformer/src/minicpm.rs`, the real per-layer attention code — calls a *different* method, `append_kv_layer(layer, k, v)`, which writes into `PagedKvCache`'s own separate `k_pages: Vec<Vec<f32>>` / `v_pages: Vec<Vec<f32>>` fields (genuinely per-layer, confirmed by reading the full method body). It never touches `self.pool`'s `KvBlock` storage at all beyond calling `append_slot()` for physical-block-ID bookkeeping.

  **Net effect**: for any model with more than one transformer layer — i.e. every real model grim would actually serve — the live inference path's real per-layer KV data lives in `PagedKvCache::k_pages`/`v_pages`, while `grim-disagg`'s transfer code reads from a parallel, disconnected `KvBlockPool` block store that the live forward pass never populates. As wired today, a prefill/decode disaggregation handoff would transfer zeroed or stale single-layer blocks rather than the model's real attention state. This is a genuine, traceable bug — not a documentation gap, not a scope difference, and not something visible from file counts or grep hits. It only surfaces by following one variable's write and read sites across three separate crates (`grim-memory`, `grim-models`, `grim-disagg`).

  I want to flag the limits of this finding honestly: I have not run the disaggregation path, so I can't rule out some reconciliation step elsewhere in the workspace I haven't found, and 29-crate workspaces sometimes have call sites that don't show up in a straightforward `grep -rn`. But the trace above is direct and the absence of any `store_kv` caller is a strong, specific signal, not a vague absence-of-evidence claim.

### 3.5 Quantization format breadth — the named tiers are a small subset of what's actually implemented

**Correction to scope**: the previous revision of this document compared vLLM's ~15 quant backends against grim's "6 named weight-format tiers" (Crow/Raven/Rook/Jay/Magpie/Jackdaw). That undersold grim significantly — those 6 names are a curated storage-codec layer sitting on top of a much larger `QuantFormat` enum (`grim-tensor/src/dtype.rs`) that the bird-name tiers don't fully surface. Reading that enum and its backing implementations properly:

**`QuantFormat` has 16 variants**, not 6: `Q8_0`, `Q4K`, `Q5K`, `Q6K`, `Fp4`, `Nf4`, `Fp8`, `Fp4Block16`, `Fp8Block16`, and a full llama.cpp-compatible i-quant (importance-matrix-optimized) family — `Iq4Nl`, `Iq4Xs`, `Iq3Xxs`, `Iq3S`, `Iq2Xxs`, `Iq2Xs`, `Iq2S` — spanning down to ~2 bits/weight. There's also `Storage::GroupInt` (GPTQ/EfficientQAT-style grouped-int quantization, with documented byte layout and both `desc_act`/sequential activation-ordering variants) and `Storage::ResidualPacked` (grim's own SpQR-style variable-bitwidth packed format with outlier and residual-backup layers, consumed by `grim_fused_dequant_gemm_f16`).

**Round-trip correctness, checked per format, not assumed:**
- 6 of the 7 i-quant formats (`Iq4Nl`, `Iq4Xs`, `Iq3Xxs`, `Iq3S`, `Iq2Xxs`, `Iq2Xs`) have real, matching `quant_*`/`dequant_*` functions in `grim-quant/src/lib.rs` — genuine round-trip implementations, not stubs.
- `Iq2S` is the one exception, and it's handled the same honest way `grim-constrain` handles unsupported JSON-Schema keywords: `quant_iq2s`/`dequant_iq2s` take underscore-prefixed (intentionally unused) parameters and return `Err(Unimplemented("...requires grid-vector lookup table; use Q2_K or Q4_K"))` — a real, explicit rejection with a documented reason and a fallback suggestion, not a silent wrong answer.

**Device kernel coverage, checked separately from CPU reference correctness, since these are genuinely different questions:**
- The generic `BackendDevice::quantize` dispatch entrypoint (used for on-the-fly, e.g. runtime KV-cache, quantization) is honestly documented in the enum's own doc comment as covering only `Q8_0`/`Fp8` on device, with everything else falling back to `Err(Unimplemented)` at that specific entrypoint. This is a real, self-acknowledged, narrow gap — but it's about *runtime requantization*, not about *inference on pre-quantized weights*.
- Separately, `grim-backend-rocm/src/kernels/iq_gemm.rs` (730 lines of real HIPRTC kernel source) implements fused dequant+GEMM `__global__` kernels for the i-quant family for both forward inference and backward (training) passes — including, notably, a complete, working `grim_fused_dequant_gemm_iq2s` kernel. This means grim can run inference (and even train) on an IQ2S-quantized model today; what it currently cannot do is *produce* IQ2S weights itself via its own conversion pipeline, since that's exactly the CPU-side `quant_iq2s` function that's stubbed. That's a narrower, more specific gap than "IQ2S unimplemented" — it's "IQ2S consumption works, IQ2S production doesn't yet."
- `grim-backend-rocm/src/gptq_kernel.rs` (286 lines) similarly has a dedicated GPTQ GEMM kernel, a correction kernel, and a scale-fit kernel, with wavefront-size-aware compilation per AMD architecture generation (`wavefront_size_for_gcn`) — real hardware-adaptive detail, not boilerplate.

**Net assessment, revised**: grim's quantization *format* breadth (16 `QuantFormat` variants plus GPTQ/SpQR-residual support) is closer to vLLM's ~15 quant-backend count than the original "6 tiers" framing suggested — it's not a smaller number in a meaningfully different league, it's comparable in count, with the difference being that vLLM's are mostly separate maintained integrations of external projects (AutoAWQ, AutoGPTQ, compressed-tensors, TorchAO) while grim's are largely hand-implemented in `grim-quant` and `grim-backend-rocm` directly. The one confirmed real gap is narrow and specific: `Iq2S` production (not consumption) and the generic runtime-`quantize` dispatch path outside `Q8_0`/`Fp8`.

Plus `grim-quant` has SpQR (`spqr.rs`) and QAT-MXFP4 (`qat_mxfp4.rs`) support, and `grim oxidizer convert` does genuine EvoPress-style importance-weighted requantization (confirmed against `grim oxidizer`'s implementation in prior sessions, not re-verified in this pass).

### 3.6 Local-server security posture (SSRF/auth) — memory claim re-checked

Prior notes state grim's `/api/pull` SSRF exposure and lack of local-API auth "matches Ollama's design posture." Checked against Ollama source: Ollama's `AuthorizationError` type and `Bearer` token handling (`server/images.go:1386`, `server/auth_test.go`) are entirely about registry push/pull auth against `ollama.com`, not gating local API access. Ollama's local `/api/generate`-style endpoints are unauthenticated by default, same as grim's local API. **This claim holds up** — it's not a case of grim being laxer than a comparable tool; both treat local-network auth as out of scope by design, consistent with how llama.cpp-style local servers generally behave.

---

## 4. Fine-tuning comparison

### 4.1 Method coverage

| Tool | LoRA/QLoRA | Full FT | ReLoRA | DPO/ORPO/KTO/SimPO | GRPO/PPO | GaLore/other memory-efficient | Multi-GPU/FSDP |
|---|---|---|---|---|---|---|---|
| **grim** | Yes (`InjectionConfig`, `all_standard_qlora()`, target-scoped: attention-only/MLP-only) | Not found as an orchestrated path | Yes (`relora.rs`) | Loss math implemented (`preference_loss.rs`: `dpo_loss`, `orpo_odds_ratio_loss`, `kto_loss`, `simpo_loss`) | Loss math implemented (`grpo_loss`, `mm_grpo.rs` for multimodal) | Not found | `fsdp.rs` exists in ROCm backend |
| **Unsloth** | Yes, its core strength | Limited | UNVERIFIED | Via TRL integration (`models/dpo.py`, `models/rl.py`) | Via TRL `GRPOTrainer` integration | UNVERIFIED | Primarily single/limited multi-GPU (historically Unsloth's main constraint vs. the others) |
| **LlamaFactory** | Yes | Yes (`finetuning_type: lora/oft/freeze/full`) | UNVERIFIED | Yes, `reward_model_type` and stage-based (SFT/RM/PPO/DPO) | Yes | GaLore, Apollo, BAdam all present as config flags (`finetuning_args.py`) | Yes, broad |
| **Axolotl** | Yes | Yes | Yes (`monkeypatch/relora.py`) | Yes, via `integrations/hatchery/rl_trainer.py` | Yes | Via integrations | Yes, broad, plugin-based |

### 4.2 grim's `Train` command is far richer than a first pass suggested — and has its own silent-fallback bugs

**Correction to scope**: an earlier version of this document undersold `grim train`'s method coverage, describing it roughly as "LoRA/QLoRA plus ReLoRA." Reading the full `Train` CLI variant (`grim-cli/src/main.rs`) shows substantially more: PiSSA (SVD-based adapter init), the OLoRA orthogonality penalty, LoRA+ (differential B-matrix LR), ReLoRA, OFT (Orthogonal Fine-Tuning) as an alternative to LoRA entirely, full-parameter fine-tuning in bf16/fp16, a custom "SCALE-ECHO" mode that bypasses the autograd tape, gradient checkpointing, held-out eval during training, and multi-dataset weighted mixing with dedup. This is a genuinely broad single-command surface — closer to Axolotl's YAML breadth than my original characterization implied.

It also declares 14 named optimizers (`grim_autograd::OptimizerKind`) and 8 named LR schedulers.

**The DPO/GRPO/ORPO wiring gap holds up on re-verification.** Checked against the complete `Commands` enum in `grim-cli/src/main.rs` (all ~50 top-level variants) — there is no `Dpo`, `Grpo`, `Rlhf`, or equivalent subcommand, and `Train`'s `mode` field only accepts `qlora | lora | full-bf16 | full-fp16 | soul-eater | oft`. Cross-checked `grim-cli/src/train.rs` directly: no call to `dpo_loss`, `grpo_loss`, or `orpo_odds_ratio_loss` anywhere in the file. The math in `preference_loss.rs` (DPO, ORPO, KTO, SimPO, GRPO, including autograd-integrated variants `dpo_loss_autograd`/`grpo_loss_autograd`) and `mm_grpo.rs` (multimodal GRPO reward normalization) is real, non-trivial, and tested — but genuinely unreachable from any CLI entry point today. This remains the most concrete gap in grim's training story.

**New finding, not in the original document — three optimizers are silently aliased to AdamW.** Reading `grim-autograd/src/adamw.rs`'s `Optimizer::new` match block line by line (not just grepping for the enum names) shows:

```rust
OptimizerKind::LOMO | OptimizerKind::Adalomo => {
    Ok(Optimizer::AdamW(AdamW::new(AdamWConfig { lr, ..AdamWConfig::default() })))
}
OptimizerKind::CAME | OptimizerKind::Sophia => {
    Ok(Optimizer::AdamW(AdamW::new(AdamWConfig { lr, ..AdamWConfig::default() })))
}
```

`--optimizer lomo`, `adalomo`, `came`, and `sophia` all silently construct plain `AdamW` — no error, no warning, no log line. A user requesting Sophia (a second-order method) or LOMO (a fused-backward, memory-efficient method specifically built to avoid materializing full optimizer state) gets ordinary AdamW with no indication anything different happened.

A second instance in the same match block: `GaloreAdamW` and `GaloreAdamW8Bit` are both routed to the same `QGaLoreAdamW8Bit` struct with default config —

```rust
OptimizerKind::QGaLoreAdamW8Bit
| OptimizerKind::GaloreAdamW
| OptimizerKind::GaloreAdamW8Bit => Ok(Optimizer::QGaLoreAdamW8Bit(
    QGaLoreAdamW8Bit::new(QGaLoreAdamW8BitConfig { lr, ..QGaLoreAdamW8BitConfig::default() }),
)),
```

— meaning the non-quantized "GaLore" and the "GaLore-8bit" variants are indistinguishable from QGaLore internally, despite their names implying three distinct memory/precision tradeoffs.

This is the same "looks implemented, isn't" failure pattern grim is already known for, but at the level of an enum variant rather than a whole subsystem — and it's a more actionable finding than the general pattern, because the fix is small (either implement the distinct math or make the CLI reject unimplemented optimizer names explicitly, the same "explicit rejection over silent under-delivery" principle `grim-constrain`'s schema compiler already applies correctly to unsupported JSON-Schema keywords).

By contrast, the LR scheduler enum (`LRScheduler`, same file) does **not** show this pattern — all 8 variants (`Cosine`, `Linear`, `Polynomial`, `Constant`, `InverseSqrt`, `Yolo`, `OneCycle`, `ReduceOnPlateau`) have their own distinct formula in `get_lr`'s match block, verified line by line. Worth noting precisely because it shows the aliasing isn't a codebase-wide habit — it's localized to specific optimizer variants in one function.

grim's SFT (supervised fine-tuning) path itself, by contrast, is real end-to-end: dataset loading (Alpaca/ShareGPT formats), multi-dataset weighted mixing with dedup, sequence packing, gradient checkpointing, OOM detection, and adapter sidecar persistence are all implemented in `grim-cli/src/train.rs` (1,678 lines), not just declared in the CLI surface.

### 4.3 Framework breadth vs. focus

LlamaFactory and Axolotl are explicitly *frameworks* — YAML/CLI-driven, plugin-based, supporting dozens of architectures via HF `transformers` for free. Unsloth trades some of that breadth for hand-written fused Triton kernels that measurably speed up training on supported architectures. grim gets neither HF's architecture breadth (it maintains its own model implementations in `grim-models`, 168 files) nor Unsloth's kernel-fusion maturity for training — `fused_add_rms_norm` is present in the ROCm and Metal backends but confirmed absent from grim's own CUDA and Vulkan backends, a parity gap even within grim's own multi-backend design.

---

## 5. Deployment model differences

- **Ollama, grim**: single local binary/daemon, model-pull UX, designed for a single machine (grim: also multi-GPU on one host via `grim-disagg`/RCCL).
- **vLLM, SGLang**: designed for cluster-scale serving from the start — richer multi-node, disaggregation, and load-balancing story, at the cost of being much heavier to stand up for a single-user local workflow.
- **Unsloth, LlamaFactory, Axolotl**: not serving engines at all — training-only, output is a checkpoint/adapter you then serve elsewhere (commonly via vLLM, SGLang, or Ollama).

grim is unusual in trying to cover both serving and training in one binary/workspace. That's an ambitious scope no other single tool here attempts, and it's also why grim is behind each specialist on the specific things that specialist optimizes for (vLLM/SGLang on paging+prefix-cache+disagg scale; the three trainers on RLHF-stage orchestration and architecture breadth).

---

## 6. Summary table

| Axis | grim's position |
|---|---|
| RDNA (consumer AMD) depth | **Ahead** of vLLM and SGLang by file-count ratio — genuinely differentiated (pending a deeper rigor pass — see Section 7) |
| CDNA (datacenter AMD) depth | Behind vLLM; SGLang is CDNA-only so not comparable on RDNA |
| Paged KV cache | **Present** — `grim-memory::PagedKvCache`, block pool + refcounting + GPU/Host/NVMe tiering, wired into the live decode path. Same functional category as vLLM's PagedAttention; relative maturity not yet assessed |
| Prefix cache sharing | **Present** — `grim-memory::RadixTree`, block-granular, explicitly modeled on RadixAttention, wired into `grim-engine` via `match_prefix_promoting`. Relative maturity vs. SGLang's `radix_cache.py` not yet assessed |
| Speculative decoding | Present, real, narrower method coverage than vLLM/SGLang (no EAGLE) |
| Structured/JSON output | JSON-object mode: real, cached, production-reasonable. JSON-Schema mode: correct but **entirely uncached** — full parse+validate per vocab token per step. Both wired end-to-end into `grim-server` |
| Disaggregated serving | Present, single-connector, smaller surface than vLLM/SGLang's pluggable systems |
| Quantization format count | 16 `QuantFormat` variants (K-quants, i-quants Iq4Nl→Iq2Xs, FP4/NF4/FP8, block-16 variants) plus GPTQ/SpQR-residual — comparable in count to vLLM's ~15 backends; one confirmed narrow gap (`Iq2S` production, not consumption, and generic runtime-quantize dispatch outside Q8_0/Fp8) |
| Fine-tuning method breadth | Broader than initially assessed: LoRA/QLoRA/full-bf16/full-fp16/ReLoRA/OFT/PiSSA/OLoRA/LoRA+, 14 named optimizers, 8 LR schedulers, gradient checkpointing, weighted multi-dataset mixing |
| Optimizer correctness | **Mixed at the variant level** — AdamW/AdamW8Bit/PagedAdamW/Lion/Lion8Bit/Adafactor/Muon/MAdam/LionVote all have distinct real implementations; `lomo`/`adalomo`/`came`/`sophia` silently alias to plain AdamW; `galore`/`galore-8bit` silently alias to `QGaLoreAdamW8Bit` — no errors or warnings in either case |
| DPO/ORPO/GRPO/KTO/SimPO | Loss math real, tested, autograd-integrated; **not wired to any trainer/CLI path** — unusable end-to-end currently |
| Architecture/model breadth | Narrower than HF-backed frameworks (LlamaFactory/Axolotl/Unsloth), since grim maintains its own model zoo rather than inheriting `transformers` |
| Local-API auth/SSRF posture | Matches Ollama's posture — not a comparative weakness |

## 7. Deep dive: Charon, Scythe (1/2/persistent), Soul Eater — subsystems not covered above

A prior pass of this document mentioned Charon and Scythe only in passing (file/line counts) without reading their contents. Corrected here, plus two direct corrections to stale internal notes this review surfaced.

### 7.1 Charon (fused MoE kernel family)

`grim-backend-rocm/src/kernels/charon.rs` (2,336 lines) is a real, substantial HIPRTC kernel suite: fused MoE dispatch/grouped-GEMM `__global__` kernels for FP8, MXFP4, MXFP8, Q8_0, and the full i-quant (`iqk`) family, plus a runtime kernel-variant selection system (`CharonSelector`, `WaveCostModel`, `CharonVariant::{SmallBatchDecode, LargeGroupPrefill, HighSkew}`) that picks the right kernel shape per routing-histogram regime with hysteresis (`min_hold`) to prevent thrashing between variants on adjacent layers. `charon_wmma.rs` (254 lines) adds a WMMA tensor-core path for the large-group-prefill regime specifically; `charon_backward.rs` (335 lines) provides the training-side backward kernels.

The code is explicit about what's device-validated vs. not: a comment block cites the acceptance gates (G-B2: synthetic-distribution regret ≤5% vs. local argmin; G-B3: zero `hipMemcpy` D2H per dispatch) as "device-gated TODOs in this sandbox" — i.e. the *design* is complete and the cost model / selector logic is unit-tested without a device, but the two performance-acceptance criteria haven't been validated against real hardware in this environment. That's an honest, precisely-scoped gap, not a vague one.

### 7.2 Correction: `tile_picker.rs::roofline_cost` — the bug memory describes as confirmed no longer reproduces

Prior internal notes state `tile_picker.rs::roofline_cost` has "a confirmed bug: `compute_time_s` divides FLOPs by a bandwidth term rather than peak FLOPs, making it structurally unable to distinguish compute-bound from memory-bound kernels." Reading the current function directly:

```rust
pub fn roofline_cost(spec: &HardwareSpec, dims: ShapeDims, _tiles: &TileConfig) -> f64 {
    let muflops = 2.0 * (dims.m as f64) * (dims.n as f64) * (dims.k as f64);
    let compute_time_s = muflops / spec.peak_flops_fp16;   // <- peak FLOPs, not bandwidth
    ...
    let memory_time_s = bytes_total / (spec.mem_bandwidth_gb_s * 1e9);
    compute_time_s.max(memory_time_s)
}
```

`compute_time_s` divides by `spec.peak_flops_fp16`, not a bandwidth term — the described bug is not present in the code as it stands. There is also a regression test directly guarding this: `roofline_cost_compute_time_uses_peak_flops_not_bandwidth` (name chosen specifically to prevent this exact regression), which passes. **This appears to be stale information** — either the bug was fixed after the note was recorded, or the note no longer reflects the current source. Either way, it should not be treated as an open item without re-confirming against the live codebase, and any downstream planning that assumed this was still broken should be revisited.

### 7.3 Correction: Scythe persistent-kernel device-gated test — the "string-containment" gap memory describes is not what the current test does

Prior internal notes describe the residual gap in `scythe_persistent.rs` as: "device-gated test remains a string-containment check rather than a real HIP launch." Reading `rocm_persistent_dispatch_opcode_6_device_gated` (and the analogous `..._opcodes_1_through_5_device_gated`) directly: this is a genuine, non-trivial hardware integration test. It constructs real GPU-resident tensors via `RocmDevice::try_new`/`dev.from_cpu_bytes`, builds a real device-visible slot/schedule/MoE-descriptor buffer layout with actual device pointers packed in, calls `dev.launch_scythe_persistent_dispatch(...)` (a real kernel launch, not a mock), synchronizes, reads the result back from the GPU, and asserts it against a hand-computed expected value (`2.0 / (1 + e^-2) * 2.0`, the correct SiLU-gated-MoE output for the test's fixture inputs) within a `1e-4` tolerance. The test is annotated "Verified via gfx1036 iGPU — 2026-08-13," indicating it has actually been run against real hardware, not just compiled. **This does not match "string-containment check"** — this is a real, hardware-validated launch-and-verify test. Same conclusion as 7.2: this looks like stale information rather than a currently-accurate description of the code.

The `__threadfence()`/`atomicExch` status-transition fixes memory separately describes as applied are confirmed present and correct in the current source (`scythe_persistent.rs` lines 224, 315, 321–322).

The one part of the original memory note this pass did *not* contradict: `grim_moe_fused_grouped_device`'s launch geometry genuinely is decided by the host-side Rust launcher rather than declared in the kernel source itself (the `__global__` wrapper is a thin pass-through to a shared `__device__` function with no grid/block-dim logic of its own) — I did not audit every Rust-side call site closely enough to confirm or refute whether that geometry is set correctly everywhere, so that specific claim is left as an open item rather than corrected.

### 7.4 Soul Eater + SCYTHE1 (low-rank adapter with FIM-preconditioned optimizer)

Two files share the name across two crates, and they are not duplicates — they're a real math/orchestration split:
- `grim-quant/src/soul_eater.rs` (269 lines): the numerical core — exact 16×16 Jacobi symmetric eigendecomposition, a condition-number/rank-deficiency check (`ConditionNumberError::{IllConditioned, RankDeficient}`), and `subspace_newton_schulz_step`, an adaptive cubic Newton-Schulz iteration for orthogonalizing tall/thin subspace matrices.
- `grim-autograd/src/soul_eater.rs` (757 lines): the adapter/optimizer orchestration layer, depending directly on the above (`use grim_quant::soul_eater::subspace_newton_schulz_step`). Implements `SoulEaterAdapter` — an SVD-style low-rank parameterization `ΔW = U·Σ·V^T` with a documented forward pass `Y = X·W0^T + (α/r)·(X·V)·Σ·U^T` — and `SoulEaterOptimizer`, which uses 1-bit Sign-SGD for the singular values Σ and momentum-accelerated Newton-Schulz orthogonalization for the U/V bases.
- `grim-autograd/src/scythe1.rs` (367 lines) extends this directly: `SCYTHE1 = SOUL EATER adapter + Natural GaLore-style inverse-Fisher-Information-Matrix preconditioning` in the low-rank adapter subspace — a running-average FIM estimate over the (small, rank-16-sized) adapter parameters, inverted and used to precondition gradients before the Newton-Schulz step.

This is real, mathematically coherent, cross-crate work — not a stub, and not something the earlier passes of this document examined at all. It's also a second, independent LoRA-family optimization strategy alongside grim's already-documented PiSSA/OLoRA/LoRA+/ReLoRA/OFT/GaLore machinery in `grim-cli`'s `Train` command — worth noting `soul-eater` is in fact one of the `Train` command's documented `mode` values ("soul-eater: orthogonal weight matrix evolution"), so unlike the LOMO/CAME/Sophia optimizer aliasing found earlier, this one *is* wired to a real, distinct implementation reachable from the CLI, not silently aliased to something else.

### 7.5 SCYTHE-2 (multi-GPU capacity-calibrated layer placement) — real, sophisticated, but wired into training, not inference

This is the single largest previously-unexamined subsystem found in this pass. Two files, again not duplicates:
- `grim-nn/src/scythe2.rs` (490 lines): `Scythe2Linear`, the leaf sharded-linear layer. `forward_placed` slices weights per a runtime-chosen `ScythePlacement`, dispatches shards to different GPUs, and reassembles output via either P2P fan-in (row-parallel) or concatenation (column-parallel). Explicitly designed so a stale placement decision degrades to *suboptimal, never incorrect* — a documented staleness-safety contract, not an afterthought.
- `grim-engine/src/scythe2.rs` (2,576 lines — larger than most entire crates in this workspace): `C2plrController`, an online per-layer GPU-placement controller. `decide()` checks a `PlacementCache` (epoch-versioned, ~50ns/layer on a cache hit per the code's own budget analysis) and on a miss runs `decide_miss`, which combines a WaveTune-style bilinear GEMM-latency estimate per GPU with a small learned MLP over layer/shape/GPU-capability features, sampled via Gumbel — falling back to balanced round-robin placement when the MLP weights are still zero (untrained). The design cites external work (WaveTune, DA-MoE) by arXiv ID in its own comments and is explicit that grim's constants are independently fit for RDNA, not imported from the cited papers' NVIDIA-validated numbers.

**Wiring check**: `C2plrController::decide` is *not* called from `grim-engine`'s inference/serving path — despite living in the `grim-engine` crate, its only real (non-comment, non-test) call sites are in `grim-garage/src/jobs.rs`, inside the multi-GPU **training** loop, alongside real GPU-capability probing and real inter-GPU link-topology probing (not mocked). The call site itself contains an honest, specific self-flagged limitation: the per-call `layer_id` is currently derived as `micro_step % num_layers` rather than a true per-forward-pass layer binding, with an explicit comment that "WI-EP1 will replace this with a true per-layer loop binding."

**Net assessment**: SCYTHE-2 is a real, carefully-designed, and substantively large multi-GPU tensor-parallelism placement system — comparable in ambition to what a production multi-GPU serving engine would need — but it currently optimizes GPU placement for **training**, not for the inference/serving path this document's Section 3 otherwise evaluates against vLLM/SGLang. That's an important scope note: none of Section 3's serving-engine comparisons should be read as crediting grim with SCYTHE-2's placement intelligence, since inference requests don't currently go through it.

### 7.6 What this section changes about the rest of the document

Two corrections stand out as more important than the subsystem writeups themselves: the `roofline_cost` bug and the `scythe_persistent` test-quality gap, both cited in prior internal notes as confirmed open issues, do not reproduce against the current source and each has direct evidence (a passing named regression test; a hardware-verified integration test) suggesting they were already fixed. Anyone planning work against those two items should re-verify against current source before prioritizing them, rather than trusting the older notes.

---

# Re-verification @ 77b4bc0e (2026-08-22) — deltas against everything above

Full re-audit of grim at HEAD plus fresh source audits of all six competitor snapshots. Sections below supersede conflicting statements above.

## D1. Disagg KV-transfer bug (§3.4): FIXED at the data-path level; handoff semantics still rough

`PagedKvCache::append_kv_layer` (`grim-memory/src/lib.rs:918-977`) now mirrors every token's post-RoPE K/V into the shared pool via `pool.write_layer_keys/write_layer_values` (lines 971-974), so `transfer_kv_cache_real` reads live data. Prefill sends per-layer block slices then the pool-level transfer (`grim-engine/src/lib.rs:798-841`); decode pulls un-received blocks and writes them into both pool and page tensors (`engine lib.rs:856-910`). Content-sniffing replaced by an explicit received bit; byte-exact loopback tests exist (`grim-disagg lib.rs:616-669`, `grim-engine/tests/disagg_engine_loopback.rs:85-219`, real TCP between two engines). Residual gaps: layers >0 are re-fetched over TCP every decode step (no per-layer arrival tracking, `engine lib.rs:880-884`); prefill nodes start no receiver so the pull path logs connection errors (push is the reliable channel); the loopback test also runs a local 4-token prefill on the decode side, so it proves transport, not pure transferred-KV decode.

## D2. Optimizer aliasing (§4.2): PARTIALLY FIXED

`Optimizer::new` (`grim-autograd/src/adamw.rs:275-349`) now returns `Error::Unimplemented` with suggested alternatives for `lomo`, `adalomo`, `came`, `sophia`, `adamw-bnb` — silent AdamW aliasing is gone. Still aliased: `GaloreAdamW` and `GaloreAdamW8Bit` both construct `QGaLoreAdamW8Bit` with default config (lines 306-313). New real implementations since §4.2 was written: Muon, MAdam, LionVote. Note: the CLI help for `--optimizer` still advertises lomo/adalomo/came/sophia as accepted values that now hard-error at training start.

## D3. Preference-loss wiring (§4.1/§4.2): split verdict — garage real, CLI degenerate

The prior "implemented but unreachable" status is outdated in BOTH directions:

- **Reachable and correct via grim-garage**: `TrainingMode::{Orpo,Dpo,Kto,SimPo,Grpo}` (`grim-garage/src/jobs.rs:186-208`) load chosen/rejected pairs from JSONL (`dataloader.rs:98-160`), run four forwards (chosen/rejected × policy/frozen-reference), apply `preference_loss_and_grads` with distinct gradients per input, and train via web API `POST /api/train/start`. Multi-rank RCCL variant exists. Integration test executes the real worker on a tiny GGUF ("PASSED 2026-08-20 on gfx1036").
- **Reachable but WRONG via grim-cli**: `grim train --mode dpo|orpo|simpo|kto|grpo` passes the SAME logp vector as chosen=rejected=ref_chosen=ref_rejected (`grim-cli/src/train.rs:1000,1011,1016,1022`) making the loss constant −log σ(0) ≈ 0.693, hardcodes GRPO rewards to 1.0 (zero advantage), and backprops a hand-picked scalar (−0.1/−0.2) times a softmax CE gradient (train.rs:1028-1046) instead of the loss derivative. This is scaled-down SFT wearing a preference label.

## D4. Correction to §4.2: `--mode` does NOT select a training path

`opts.mode` is consumed only for printing and preference-mode detection (`train.rs:592,968,998`). `lora | full-bf16 | full-fp16 | soul-eater | oft` all execute the identical QLoRA-style adapter injection (`standard_qlora_with_flags`, train.rs:651-660). There is **no full-parameter fine-tuning** in grim today, contrary to §4.2's list of Train capabilities and the CLI help/docs. Soul Eater/Scythe1 math remains implemented-and-tested but has zero production callers (garage labels for those modes fall through to generic SFT, jobs.rs:1949-1969).

## D5. New serving findings (not in earlier sections)

- **Speculative decoding is default-ON**: `Engine::register_model` wraps every registered model in `SpeculativeCausalLm::auto` (strategies Plain/DSpark/NativeMtp, VRAM-aware fallback, acceptance-rate telemetry). EAGLE3 exists as a model file (`grim-models/transformer/src/eagle3.rs`) but nothing implements it as a drafter.
- **Radix prefix-cache reuse defaults OFF** behind `GRIM_RADIX` env (`grim-engine/src/lib.rs:368-371`) — §3.1 described the mechanism but not the gate.
- **Hot-path perf debt**: `paged_kv_handles` re-uploads K/V pages host→device every call/every layer/every step despite cache fields existing (`grim-memory lib.rs:987-1023`).
- **Endpoint honesty**: `/v1/embeddings` returns hardcoded 501; `/v1/images/generations` runs a Flux2 forward but puts the literal string `format!("flux2_generated_pixels_{n}")` in `b64_json`; `/v1/audio/transcriptions` feeds an all-zero mel and returns canned `"Transcribed audio content"` even when Whisper is loaded (`grim-server/src/lib.rs:2382-2396,2632,2497-2518`). Zero auth anywhere (bind-defaults + optional TLS only).
- **Multimodal Studio (grim-garage)**: Diffusion/Audio Studio tabs wired end-to-end at the transport level (UI→HTTP→pipeline→base64 BMP/WAV), but demo-grade: random-init default configs, synthetic prompt embeddings, `char % 256` tokenization for TTS, sine-synth mel for audio2audio; no checkpoint loading (`grim-garage/src/routes.rs:1674-1958`).
- **Eval**: two tasks only (wikitext2 windowed PPL, GSM8K exact-match vs a running server). In-training held-out eval is a stub returning exp(training loss) (`grim-cli/src/train.rs:1185-1199`).
- **SCYTHE-2 improvement**: the `micro_step % num_layers` placeholder is gone; `C2plrController.decide` runs per layer with real indices/shapes inside the rank-SFT forward and feeds step-level gradient-sync routing with measured-latency updates (`jobs.rs:1125-1136, 2628-2659`). Still training-only.
- **Checkpoint-replay** (commit 647f3e87): segment replay during backward proven gradient-parity-equal to uncheckpointed backward; wired into CLI train when `--checkpoint-segs > 1`.
- **QAT reality**: `--qat-mxfp4` fake-quantizes ONLY `lm_head.weight`; the field doc claiming all Linear weights are fake-quantized is currently false (`train.rs:939-947` vs `train.rs:103-107`).

## D6. Competitor snapshot corrections (materially different from common descriptions)

- **Ollama no longer vendors a ggml fork** (supersedes §1 table): the vendored tree is empty; upstream llama.cpp (`LLAMA_CPP_VERSION=b10091`) is fetched at build time via FetchContent and built as a `llama-server` binary; the Go daemon spawns one subprocess per loaded model and proxies HTTP (`llm/llama_server.go`), with a 105-line compat hook patch. First-party stacks: a Go MLX runner (own prefix-cache trie, MTP drafting) and an experimental Flux-architecture image generator. Anthropic Messages API served locally alongside OpenAI compat. Vulkan default-enabled.
- **vLLM now contains a first-party Rust workspace** (`rust/`: server, tokenizer, parser, chat; 306 `.rs` files bridged via PyO3) — the "Python+CUDA" description is already outdated for this snapshot. Original PagedAttention CUDA kernel no longer exists; the design doc is self-flagged historical. RDNA presence: ~138 mentions/16 files incl. purpose-built rdna3 W4A16 WMMA kernels + tests; CDNA ~524 mentions/101 files. No training loops; RL-ready weight-update/pause-resume plumbing instead.
- **SGLang: zero consumer-RDNA code** (0 files for any RDNA gfx target; 73 files gfx90a/942/950/1250) — stronger version of §2's ratio claim. Eight builtin speculative algorithms + plugin registry; PD-disaggregation with 4 connectors; ~188 registered architecture classes; no training loops.
- **Unsloth (2026.8.15)**: ROCm support is now substantial (radeon.com-pinned wheel matrix incl. Windows ROCm, HIP branches throughout); multi-GPU *training* raises a hard RuntimeError while README advertises multi-GPU (that refers to Studio inference). Hard runtime dependency on unsloth_zoo; DPO/KTO patching moved to the zoo leaving `models/dpo.py` an empty stub.
- **LlamaFactory (0.9.6.dev0)**: stages stop at pt/sft/rm/ppo/dpo/kto — **no GRPO stage** (spun out to sibling EasyR1); `llamafactory-cli eval` raises `NotImplementedError` (benchmark evaluation deprecated, evaluator orphaned); ~634 registered checkpoints/131 model groups, ~124 chat templates, 24 multimodal plugins incl. audio; export writes Ollama Modelfiles but has no GGUF path; Ascend NPU support is real (torch-npu pins, fused NPU MoE/RMSNorm kernels); ROCm footprint is one Dockerfile.
- **Axolotl (0.18.0)**: RL coverage DPO/IPO/SimPO/ORPO/KTO/GRPO/GDPO/EBFT + async GRPO with vLLM weight sync; ReLoRA in-tree; N-D parallelism (FSDP×TP×CP×EP via DeviceMesh); `src/axolotl/integrations/` carries a separate restrictive community license inside the Apache-2.0 repo; no first-party general-purpose inference server (vLLM serve is aimed at GRPO rollouts).
