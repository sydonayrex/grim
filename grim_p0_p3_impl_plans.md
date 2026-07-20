# grim Implementation Plans — P0 → P3

Companion to `rocm-server-gaps.md` (S1–S6) and `grim_format_compat_work_items.md`
(F1–F4). These four plans pick up where those leave off and target the two
goals stated by the maintainer:

1. **Drop-in Ollama replacement** — the API surface is ~80% there (S1–S6, F1/F3
   server shims). The blocker is *real inference quality* + *format
   correctness*, not API shape.
2. **Top-tier ROCm performance** — the architecture is scaffolded (scheduler,
   paged KV cache, rocBLAS GEMM, RCCL/p2p wiring, KV-quant CPU code) but the
   throughput wins (WMMA packed GEMM, fused dequant-attention, multi-GPU
   serving) are not yet realized in the serving path.

Every work item carries gates in the project's established order —
**correctness → compile → architecture → `TODO(gpu-verify)` perf** — and a
`[Scoped]` left/right limit. Status tags: `[Done]`, `[Partial]`, `[Missing]`,
`[Stub]`.

**Code-grounded status at time of writing** (verified in tree):
- `chat_completions` non-stream caps at `for step in 0..5` (lib.rs:296) and
  streams cap at 256 (lib.rs:212); both always argmax, never reading
  `temperature`/`top_p`/`max_tokens`/`stop`.
- `load_model_for_server` (server lib.rs:1403) calls
  `model_loader::load_from_path`, which **only** calls `load_model_from_gguf`
  (model_loader.rs:248–265) — a `.grim` can never be served.
- `SafetensorsProvider::get`/`meta` unconditionally stamp
  `QuantProvenance::GrimNative` (tprov.rs:169,180,271,295,369) — WI-F2.
- Engine exposes `StepOutcome.logits` (engine lib.rs:80–83) and a `Sampler`
  trait (grim-core sampler.rs:14), but the server ignores both.
- ROCm GEMM: real rocBLAS (roc_device.rs:1388) + opt-in JIT decode/WMMA kernels
  gated off (roc_device.rs:1341,1352). No kernel reads `GrimTensorExt` yet.
- `grim-kvquant::fused_attention` is a CPU scalar stub (kvquant lib.rs:406).
- RCCL + `p2p_route` + `peer_access` exist, feature-gated, uninvoked in serving.
- `build.rs` hardcodes `dylib=rccl` (WI-R0) — links fail without RCCL.

---

# P0 — Make the server actually serve correct inference

P0 is the gate for "drop-in Ollama." None of it is about new formats or kernels;
it is about (a) real generation length/sampling, (b) not emitting silently
wrong numbers from quantized safetensors, and (c) being able to *load the very
`.grim` files WI-S5/S6 exist to produce*. Until P0 lands, the compatibility
shims are a demo, not a replacement.

## P0-WI-1 — Real token generation in `chat_completions`

### Why
`chat_completions` (`crates/grim-server/src/lib.rs`) emits a fixed 5 tokens
(non-stream) or caps at 256 (stream), always argmax, and ignores
`temperature`/`top_p`/`max_tokens`/`stop`. An Ollama client sending
`{"options":{"num_predict":512,"temperature":0.7,"stop":["\n"]}}` gets 5
argmax tokens with none of those honored. This is the single biggest blocker
for drop-in status — everything downstream (chat responses, tool use,
streaming UIs) depends on real length + sampling control.

### Where
- `crates/grim-server/src/lib.rs::chat_completions` (non-stream loop ~296,
  stream unfold ~205).
- `crates/grim-engine/src/lib.rs::StepOutcome` (logits available at :80–83),
  `Engine::tick` / `enqueue_request`.
- `crates/grim-core/src/sampler.rs::Sampler::sample` (trait exists at :14;
  need a concrete impl: argmax + temperature + top-p).

### What already exists
- `StepOutcome.logits: Option<Arc<Tensor>>` is recorded per tick — the raw
  material for sampling already flows out of the engine.
- Engine does real autoregressive position advancement
  (`engine_tick_runs_prefill_then_decode_advancing_pos` test, engine lib.rs:628).
- A `Sampler` trait already models `sample(&logits, &history) -> u32`.

### What to build
1. Add a concrete `Sampler` impl (e.g. `TopPSampler`) in `grim-core::sampler`
   that takes `temperature`, `top_p`, and `top_k`, returns argmax when
   `temperature == 0`, else samples. Keep the existing argmax path as the
   `temperature == 0` branch so behavior is unchanged for deterministic callers.
2. In `chat_completions`, read `temperature`/`top_p`/`max_tokens`/`stop` from
   the whitelisted fields (they are already accepted at :111–121) and thread
   them into both the non-stream and stream branches.
3. Replace the hardcoded `0..5` / `256` caps with `max_tokens` (default to a
   sane cap like 2048, never unbounded). Honor `stop` sequences by checking
   the decoded token text against the stop list and ending the loop.
4. Each generated token uses `Sampler::sample` on `StepOutcome.logits` instead
   of inline `max_by` argmax.

### Left-right limits
- **Left limit:** do not change `Engine::tick`/`decode_one` semantics — sample
  at the server boundary using the logits the engine already returns.
- **Right limit:** do not implement beam search / penalties / repetition
  beyond top-p + temperature + stop in this item; those are follow-ups.

### Gates
1. **Correctness:** a request with `max_tokens: 20` yields ≤20 content tokens;
   `temperature: 0` is byte-identical to the old argmax path; a `stop:["END"]`
   request ends when the token decodes to that string.
2. **Compile:** `cargo check -p grim-server -p grim-core`.
3. **Architecture-cleanliness:** sampling lives in `grim-core::sampler`, not
   inline in the server; `chat_completions` stays the only caller that maps
   OpenAI fields → sampler params.
4. **Perf (non-blocking):** none — sampling is O(vocab) per token, negligible.

## P0-WI-2 — Fix `SafetensorsProvider` provenance (WI-F2)

### Why
`tprov.rs` stamps `QuantProvenance::GrimNative` for **every** tensor a plain
safetensors file yields (tprov.rs:169,180,271,295,369), including real GPTQ/
AWQ `.qweight` groups when the provider is opened standalone (not via the
GptqProvider delegation path). Packed int32 weights get read as floats →
silently wrong output. This is higher-severity than a crash.

### Where
- `crates/grim-format/src/tprov.rs::SafetensorsProvider` (`get` :256, `meta` :284).
- `crates/grim-format/src/gptq.rs` (post-WI-F1 `read_quant_params`, already real).

### What already exists
- WI-F1 DONE: `GptqProvider` reads real `bits`/`group_size` and returns
  `ExternalQat` tensors. `SafetensorsProvider` already delegates
  `.qweight`-suffixed names to Gptq-backed entries (tprov.rs:284,351) — only the
  *standalone* safetensors path and the non-delegated branches need fixing.

### What to build
1. At `SafetensorsProvider::open`, scan header tensor names for `.qweight`
   (the same detector `GptqProvider` uses). If found, hold an inner
   `GptqProvider` and route those names to it; report `ExternalQat` for them.
2. For names not part of a `.qweight` group, keep the existing native path and
   `GrimNative` provenance (regression-safe — plain F16/F32 safetensors must
   stay byte-identical to today).
3. No second safetensors header parser — delegate to/reuse `GptqProvider`'s
   logic, matching the WI-F2 architecture gate (no duplicated parsing).

### Left-right limits
- **Left limit:** do not change `read_safetensors_header` binary format parsing.
- **Right limit:** do not add FP8/NVFP4 detection here (that's P2-WI-F4).

### Gates
1. **Correctness:** loading a mixed plain+GPTQ safetensors file returns
   `GrimNative` for plain tensors and `ExternalQat` for grouped ones; a
   regression fixture of a plain safetensors file round-trips byte-identical.
2. **Compile:** `cargo check -p grim-format`.
3. **Architecture-cleanliness:** single header-parse path; `SafetensorsProvider`
   delegates to `GptqProvider`, no reimplementation.
4. **Perf (non-blocking):** none.

## P0-WI-3 — Serve `.grim` from the loading paths

### Why
`model_loader::load_from_path` (model_loader.rs:248–265) **only** calls
`load_model_from_gguf`. So even after WI-S5/S6 produce a ROCm-tuned `.grim`,
neither `/v1/models/load` (`load_model`, server lib.rs:557) nor the on-demand
`load_model_for_server` (server lib.rs:1403) can load it. The conversion
pipeline's output is currently unservable — the gap is self-defeating.

### Where
- `crates/grim-engine/src/model_loader.rs::load_from_path` (add `.grim` branch).
- `crates/grim-format/src/tprov.rs::GrimProvider::open` (already exists :309).
- `crates/grim-server/src/lib.rs::load_model` (:557), `load_model_for_server`
  (:1403) — should prefer `.grim` (WI-S6 preference) and fall back to `.gguf`.

### What already exists
- `GrimProvider::open` + a working `.grim` reader (`tprov.rs:309`, `:358`).
- WI-S6 already prefers `.grim` in the CLI; the server's auto-load scan was
  just extended to prefer `.grim` too. This item extends that to the explicit
  load endpoints.

### What to build
1. In `load_from_path`, branch on extension: `.grim` → build a `CausalLm` from
   `GrimProvider` (reuse the GGUF weight-loading path but feed it the
   `GrimProvider` tensors + `GrimTensorExt` metadata); `.gguf` → existing path.
2. `load_model` and `load_model_for_server` resolve the model name through
   `catalog::resolve_model_preferring_grim` (WI-S6) so a `.grim` sibling wins,
   then pass the resolved path to `load_from_path`.
3. Keep `GgufProvider::open` for tokenizer extraction (tokenizer lives in GGUF
   metadata; if only a `.grim` is present and has no tokenizer, fall back to a
   sibling `.gguf`'s tokenizer or emit a clear warning).

### Left-right limits
- **Left limit:** do not modify `GrimProvider`'s read logic — only add a caller
  in `model_loader`.
- **Right limit:** do not change `load_model_from_gguf`; `.grim` is an added
  branch, `.gguf` behavior unchanged.

### Gates
1. **Correctness:** `grim pull X` → `grim oxidize convert X --rocml-profile
   rdna3` → `POST /v1/models/load {name:"X"}` loads the `.grim` and serves
   tokens; `/api/chat` on it works.
2. **Compile:** `cargo check -p grim-engine -p grim-server`.
3. **Architecture-cleanliness:** single dispatch point in `load_from_path`;
   server endpoints share `resolve_model_preferring_grim`.
4. **Perf (non-blocking):** none — loading path only.

---

# P1 — Realize the ROCm throughput path

P1 turns the scaffolded ROCm pipeline into measurable performance. Two items:
make the **packed-low-bit WMMA GEMM** path actually dispatch (it's gated off
today and nothing reads `GrimTensorExt`), and build the **fused
dequant-attention HIP kernel** for compressed KV (today a CPU stub). These are
the two highest-leverage consumer-RDNA wins per the research review.

## P1-WI-1 — Enable + complete the WMMA / packed-low-bit GEMM path

### Why
rocBLAS is the default GEMM (real, fine for generic F16), but the ROCm-tuned
paths — JIT decode GEMM and WMMA GEMM — are **gated off** (roc_device.rs:1341,
1352) and no kernel consumes `GrimTensorExt` (the packed/block-scale weight
layout WI-S5/WI-S7 were meant to enable). On RDNA3/4 (no INT4 tensor core, but
WMMA matrix cores), the win is *pre-packed low-bit weights fed to WMMA* — which
is exactly what grim's format was designed for but the kernel side never reads.

### Where
- `crates/grim-backend-rocm/src/device/roc_device.rs` (GEMM launch dispatch,
  `launch_wmma_gemm` :1352, `launch_decode_gemm_f16` :1341,
  `lookup_gemm_config` :1232).
- `crates/grim-format/src/spec.rs::GrimTensorExt` (block_size, row_scale_dtype,
  layout_hint).
- `crates/grim-backend-rocm/src/device/layout.rs` (PackedQuant layout, WI-R7).
- `grim_v2_gap_plan.md` WI-G / WI-C (kernel work, not yet built).

### What already exists
- WMMA launch kernel skeleton present, default-off.
- `GrimTensorExt` container format complete (per-row bpw, per-block scale).
- `LayoutHintTag::{WavefrontTiled,BlockSparse}` exists; needs a packed-low-bit
  WMMA variant (WI-R7).

### What to build
1. Add `LayoutHintTag::PackedQuantWmma { bits, frag_mn }` (WI-R7) and wire it
   from `GrimTensorExt.layout_hint`.
2. In the GEMM dispatch, when a weight carries `PackedQuantWmma` + a detected
   RDNA3/4 device, route to `launch_wmma_gemm` (enable by default on RDNA3/4,
   keep rocBLAS fallback for other arch / unsupported shapes).
3. Implement the dequant-then-WMMA (or fused if feasible) path reading
   `GrimTensorExt` block scales — start from `dequant_q4k`'s two-level pattern.
4. Feature-gate behind `rocm-wmma` so non-AMD / CI builds stay on rocBLAS.

### Left-right limits
- **Left limit:** do not change the rocBLAS path; it remains the fallback.
- **Right limit:** no NVFP4 hardware fast-path here (P2); this is storage-aware
  WMMA dispatch only.

### Gates
1. **Correctness:** WMMA GEMM output numerically matches the rocBLAS path
   (bit-for-bit or within tolerance) on a Q4_K weight.
2. **Compile:** `cargo check -p grim-backend-rocm` with/without the feature.
3. **Architecture-cleanliness:** dispatch keyed on `layout_hint` + device, no
   per-call arch branching scattered through the kernel.
4. **Perf (`TODO(gpu-verify)`):** measure tok/s on gfx1100/gfx1200 vs rocBLAS
   baseline on a 7B Q4_K model; annotate with real hardware numbers.

## P1-WI-2 — Fused dequant-attention HIP kernel for compressed KV

### Why
`grim-kvquant::KvCompressor::fused_attention` is a CPU scalar stub
(kvquant lib.rs:406). RotateKV / KVTuner / RocketKV (2–3× KV memory, +21%
throughput) need a real kernel consuming `CompressedKvBlock`. Without it, KV
quantization is computed-correct but never actually runs on GPU — the biggest
single consumer-RDNA throughput gap.

### Where
- `crates/grim-kvquant/src/lib.rs::fused_attention` (the trait stub).
- `crates/grim-backend-rocm/src/kernels/qkv_attention.rs` (existing QKV HIP
  kernel entry point).
- `crates/grim-kvquant` `CompressedKvBlock` (key_bits, key_meta, value_bits).

### What already exists
- CPU `LloydMaxCompressor` / `IdentityCompressor` produce real `CompressedKvBlock`.
- A QKV attention HIP kernel skeleton exists (`qkv_attention.rs`); the
  compressed-dequant variant is missing.

### What to build
1. Add a HIP kernel (or extend the existing one) that reads rotated/quantized
   KV (`CompressedKvBlock`) and dequantizes inside the attention loop.
2. Gate behind `KvDequantAttentionConfig` (default-off); CPU reference path
   stays for verification.
3. Wire `Engine` decode to call the GPU path when a compressed KV cache is
   active; keep the `grim-kvquant` CPU compressor for producing the blocks.

### Left-right limits
- **Left limit:** do not change the CPU compressor correctness — the kernel
  must match it numerically.
- **Right limit:** no new quantization *algorithm* here (RotateKV/etc.
  integration is its own follow-up); implement the dequant-attention for the
  already-produced `CompressedKvBlock`.

### Gates
1. **Correctness (DONE):** GPU fused-attention output matches a pure-float
   reference within 0.05 on gfx1036 — verified by
   `grim-backend-rocm/tests/kv_dequant_attention_gpu.rs` across
   1:1 heads, GQA 8:2, and GQA 4:1 with head_dim>64. The
   kernel dequantizes signed 8-bit K/V (`(byte-128)/127*scale`) with
   a per-buffer scale; the CPU "reference" in `fused_attention`
   itself is left as-is (it injects an INT8-sim that models the
   SageAttention tile path and is intentionally NOT ground truth).
2. **Compile (DONE):** `cargo build --workspace` + `cargo test -p grim-kvquant`
   + the GPU test all pass.
3. **Architecture-cleanliness (DONE):** one `fused_attention` dispatch
   point; `BackendDevice::kv_dequant_attention` trait default returns
   `Err(Backend)` (CPU/CUDA/Vulkan/Metal unaffected); `RocmDevice`
   overrides and delegates to the wired HIP kernel; `grim-kvquant`
   stays backend-agnostic (no `grim-backend-rocm` dependency).
4. **Perf (`TODO(gpu-verify)`):** measure decode tok/s + KV memory at 2–3-bit
   vs dense attention on RDNA3/4.

---

# P2 — Accuracy-per-bit + multi-GPU transport

P2 adds NVFP4-equivalent block scaling (accuracy at low bitwidth) and wires the
multi-GPU transport that already exists but is never invoked in serving.

## P2-WI-1 — NVFP4-equivalent two-level block scale (WI-F4)

### Why
`Storage::FloatPack(Fp4|Nf4|Fp8)` uses one global f32 scale (grim-quant
`dequant_fp4`/`dequant_fp8`), worse accuracy-per-bit than `Q4_K` or AngelSlim
NVFP4 (16-element block + FP8 second-level scale). This caps how good a 4-bit
grim model can get.

### Where
- `crates/grim-format/src/spec.rs::GrimTensorExt` (add `RowScaleDtype::Fp8`,
  validate `block_size = 16`).
- `crates/grim-quant/src/lib.rs::dequant_fp4/dequant_fp8` (add block mode).
- `crates/grim-tensor/src/dtype.rs` (confirm no `Storage` change needed).

### What already exists
- `GrimTensorExt` block-scale container; `dequant_q4k` two-level pattern to
  mirror; legacy single-scale functions retained for `block_size == 0`.

### What to build
1. `RowScaleDtype::Fp8 = 2`; accept `block_size = 16`; block-scaled FP4/FP8
   dequant reading per-block scale (Fp8) + one tensor-level f32 scale.
2. Round-trip test vs single-scale on an outlier-channel tensor.

### Left-right limits
- **Left limit:** no wire-version bump; rides `GrimTensorExt` JSON layer.
- **Right limit:** no Blackwell tensor-core fast-path; generic dequant only.

### Gates
1. **Correctness:** block-mode round-trip error < single-scale on a synthetic
   outlier tensor.
2. **Compile:** `cargo check -p grim-format -p grim-quant -p grim-tensor`.
3. **Architecture-cleanliness:** no new `Storage`/`DType` variant.
4. **Perf (`TODO(gpu-verify)`):** annotate decode cost of per-block scale
   lookup vs single-scale.

## P2-WI-2 — Make RCCL/peer transport invokable in serving (WI-R0→R3)

### Why
RCCL (`/opt/rocm/lib/librccl.so`) is on the link line, `p2p_route.rs` +
`peer_access.rs` + `rccl.rs` exist with tests, but **nothing in the serving
path calls them** (feature-gated, uninvoked). For dual-GPU consumer RDNA3/4
rigs, CommFuse-style P2P decomposition is a 36.9% latency win — currently
orphaned. Also `build.rs` hardcodes `dylib=rccl`, so RCCL-less systems fail to
link (WI-R0).

### Where
- `crates/grim-backend-rocm/build.rs` (WI-R0 feature-gate + env discovery).
- `crates/grim-backend-rocm/src/rccl.rs` (collective API, :51 feature-gated).
- `crates/grim-backend-rocm/src/p2p_route.rs` (`copy_route` :189; actual
  `hipMemcpyPeerAsync` bridge still deferred).
- `crates/grim-disagg` (PP stage scheduling — `PoolRole` exists, no scheduler).

### What already exists
- Full RCCL FFI + `p2p_memcpy_async` (rccl.rs), peer probe (peer_access.rs),
  route decision (p2p_route.rs). System RCCL link present.

### What to build
1. WI-R0: `resolve_rocm_lib_dir()` checking `RCCL_PATH`→`ROCM_RCCL_PATH`→
   `ROCM_PATH/lib`→`/opt/rocm/lib`; emit the link directive only when
   `feature="rccl"` and discovery succeeds.
2. WI-R1: expose `all_reduce`/`reduce_scatter`/`all_gather`/`all_to_all` +
   `p2p_copy` behind a `CollectiveConfig`, `#[cfg(feature="rccl")]`.
3. WI-R2: implement the `hipMemcpyPeerAsync` bridge behind `p2p_route.rs`.
4. WI-R3: replace vanilla collectives in a TP/PP path with CommFuse-style P2P
   decomposition fused against compute (start with a thin TP all-reduce call
   from the engine scheduler).

### Left-right limits
- **Left limit:** do not break the RCCL-less default build (must link/compile
  with feature off).
- **Right limit:** no cross-vendor collective (HetCCL) here; system RCCL only.

### Gates
1. **Correctness:** WI-R2 D2D copy == host-bounce copy; WI-R3 TP all-reduce
   numerically == rocprim baseline.
2. **Compile:** `cargo check -p grim-backend-rocm` feature on/off; no hard link
   fail on RCCL-less system.
3. **Architecture-cleanliness:** collective calls isolated behind `rccl.rs`,
   no reach into `grim-scheduler` internals beyond a TP hook.
4. **Perf (`TODO(gpu-verify)`):** bandwidth/latency on 2× RDNA3 vs rocprim.

---

# P3 — Format completeness for sustained advantage

P3 extends `.grim` beyond weight-only storage — the research review's
conclusion that `.grim` should be the single source of truth for everything
low-precision (weights, KV, activations, adapters). These are what keep grim
ahead once P1/P2 land, not prerequisites for them.

## P3-WI-1 — Persistent KV-cache layout in `.grim` (WI-R4)

### Why
`grim-kvquant` produces `CompressedKvBlock` at runtime, but there is **no
on-disk schema** — a reloaded session can't resume a compressed cache, and
cross-GPU KV sharding has no contract (blocks WI-R3's KV sharding).

### What to build
New optional metadata region: `kv_rotated: bool`, per-head `kv_bits_k/v`,
`kv_eviction_map_offset`, `kv_sink_fp16`. `grim-kvquant` writes the produced
`CompressedKvBlock` shape into it. Legacy files lacking the region stay
readable.

### Gates
1. **Correctness:** reload reproduces the compressed cache bit-for-bit.
2. **Compile:** `cargo check -p grim-format -p grim-kvquant`.
3. **Architecture:** back-compat optional region; `GrimProvider::open` ignores
   absent region.
4. **Perf (`TODO(gpu-verify)`):** resume-without-recompression latency.

## P3-WI-2 — Activation-quant metadata (WI-R5 enabler)

### Why
ELUTQ LUT-GEMM / SERQ low-rank error need *activation* quant params, not just
weights. V3 is weight-only.

### What to build
`act_quant_dtype` + `act_scale_layout` fields on `GrimTensorExt`; consumed by a
future fused dequant-attention (P1-WI-2 extension).

### Gates
1. **Correctness:** act-quant params round-trip; no effect on weight-only
   readers.
2. **Compile:** `cargo check -p grim-format`.
3. **Architecture:** additive metadata only.

## P3-WI-3 — Training-extension region (WI-R6)

### Why
Enables LoRA/DoRA/optimizer persistence for consumer fine-tune (26× mem
evidence in the research review). `.grim` is weight/inference-only today.

### What to build
Optional `train` region: `optimizer` state, `lora_a/b` offsets, `error_matrix`
offset, `fp_format`. Prefer a `.grim.train` sidecar until training matures (per
review open question #4) to keep inference readers clean.

### Gates
1. **Correctness:** round-trip adapter weights through the sidecar.
2. **Compile:** `cargo check -p grim-format`.
3. **Architecture:** sidecar, never breaks `GrimProvider::open`.

## P3-WI-4 — WMMA layout hint completion (WI-R7, pairs with P1-WI-1)

### Why
`LayoutHintTag::PackedQuantWmma` (added in P1-WI-1) needs the matching
`PackedQuantLayout::Wmma` in `device/layout.rs` so the kernel dispatches
without re-deriving strides.

### What to build
`PackedQuantLayout::Wmma` + consume in P1-WI-1's dispatch; verify WMMA output ==
scalar kernel.

### Gates
1. **Correctness:** WMMA output == scalar reference.
2. **Compile:** `cargo check -p grim-backend-rocm`.
3. **Perf (`TODO(gpu-verify)`):** throughput on gfx110x/gfx1200.

---

# Sequencing summary

```
P0 (drop-in gate) ────────────── P1 (ROCm perf) ──────────── P2 ──────────── P3
 WI-1 real gen                   WI-1 WMMA GEMM              WI-1 NVFP4        WI-1 KV layout
 WI-2 safetensors prov (F2)      WI-2 fused KV-attn HIP     WI-2 RCCL serving  WI-2 act quant
 WI-3 serve .grim                                                   (R0→R3)     WI-3 train region
                                                                      WI-4 WMMA hint
```

- **P0 is strictly prerequisite for "drop-in Ollama"** and is independent of
  the kernel work — do it first; it is mostly server/engine glue.
- **P1-WI-1 and P1-WI-2 can run in parallel** (different crates: backend-rocm
  GEMM vs kvquant kernel).
- **P2-WI-1 (NVFP4) is independent**; P2-WI-2 (RCCL) is a transport chain
  R0→R1→R2→R3, blocked on R0 (build feature-gate) first.
- **P3 is entirely additive** to P1/P2 and can start once P1-WI-1's layout hint
  exists; none of it blocks P0/P1/P2.

**Definition of done for the two goals:**
- *Drop-in Ollama:* P0 complete + P0-WI-2 (no silent wrong-output) + P0-WI-3
  (can serve what you convert).
- *Top-tier ROCm:* P1 complete on RDNA3/4 (WMMA GEMM enabled + fused KV-attn
  kernel), with P2-WI-1 (NVFP4) and P2-WI-2 (multi-GPU) as the stretch to
  "top of class" on both single- and dual-GPU consumer rigs.
