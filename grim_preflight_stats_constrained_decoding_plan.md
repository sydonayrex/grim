# Grim: Pre-Flight Prediction, `/api/stats` Gaps, and Constrained Decoding — Implementation Plan

Audience: a lower-context AI coding agent working directly in the `grim` workspace.
Format: Why / Where / What-already-exists / What-to-build / Left-right-limits / Gates, per work item.
Gate order for every item: correctness → compile → architecture-cleanliness → performance (non-blocking).
All new code paths ship behind feature flags or additive fields; nothing here changes existing default behavior for callers who don't opt in.

---

## Correction to prior findings (read this first)

An earlier pass claimed `/api/stats` returns hardcoded zeros. That claim is **wrong** and is retracted here. Direct verification of `grim-server/src/lib.rs::stats_endpoint` and `grim-engine/src/lib.rs` shows:

- `probe_vram_and_gpus` and `probe_sys_ram` do live `/proc/meminfo` and real `grim_backend_rocm::vram_info` / CUDA / Metal / Vulkan probing.
- `engine.tokens_per_sec()` returns a real EMA (`tokens_per_sec_ema`), `None` only when no model is loaded or no tokens generated yet — not a fabricated value.
- `engine.kv_cache_telemetry()` reads real block-pool state (`capacity`, `used_count`, `block_bytes`), returning `(0,0,0,0)` only if the lock is poisoned.

So **WI-1 in this plan is not "wire up fake telemetry."** It's the two real, narrower gaps still open: `compute` is a hardcoded `0u32` in every GPU JSON entry across `probe_vram_and_gpus`/`probe_cuda_vram` (4 call sites), and per-GPU attribution for `tokens_per_sec`/`kv_cache` is aggregate-only (no per-device breakdown) in a multi-GPU disaggregated or tensor-parallel setup. Scope WI-1 accordingly — do not re-plumb telemetry that already works.

---

## WI-1: `/api/stats` — GPU compute utilization + per-device attribution

### Why
`compute` is hardcoded to `0u32` at every call site in `probe_vram_and_gpus` (ROCm, CUDA, Metal, Vulkan branches) and in `probe_cuda_vram`. Anyone building a dashboard or autoscaler off `/api/stats` today sees real VRAM/RAM/KV/tok-per-sec but a permanently-zero compute column, which is worse than omitting the field — it looks live but lies. Separately, once `grim-disagg`/multi-GPU tensor-parallel setups are exercised, `tokens_per_sec` and `kv_cache` are engine-global aggregates; there's no per-device breakdown in the JSON even though `gpus` is already an array.

### Where
- `grim-server/src/lib.rs`: `probe_vram_and_gpus` (ROCm/CUDA/Metal/Vulkan branches), `probe_cuda_vram`, `stats_endpoint`.
- `grim-backend-rocm/src/device/roc_device.rs` (or wherever `vram_info` lives) — need to confirm whether a utilization query exists at the ROCm driver level (`rsmi`/`rocm_smi` bindings) or needs adding.
- `grim-backend-cuda`, `grim-backend-vulkan`, `grim-backend-metal` — equivalent utilization probes, if the driver APIs expose one.
- `grim-engine/src/lib.rs`: `tokens_per_sec_ema` and `block_pool` — check whether these are already tracked per-device internally or engine-global only.

### What already exists (read first)
- `vram_info(ordinal) -> (free, total)` per backend — confirmed real, already called.
- `RocmDevice::probe()` returns a `Vec<RocmDevice>` with `.ordinal()`, `.wavefront_size()`, `.xnack_enabled()` — confirmed in `doctor.rs`. Check whether `RocmDevice` or a sibling type exposes a utilization percentage; if not, this is new driver-binding work, not just plumbing.
- `stats_endpoint` already emits `gpus: [{index, compute, memory, name}]` — the shape is fixed and correct; only `compute`'s value is wrong.

### What to build
1. Add a `compute_utilization(ordinal: usize) -> Option<u32>` probe per backend:
   - ROCm: via `rocm_smi`/`rsmi` FFI or by shelling to `rocm-smi --showuse --json` if no direct binding exists yet — prefer FFI, fall back to shell-out only if no binding is feasible within scope, and mark that fallback clearly with a code comment (shell-out is fragile and shouldn't be silently treated as equivalent to a direct query).
   - CUDA: `nvmlDeviceGetUtilizationRates` via existing CUDA FFI surface if `grim-backend-cuda` already links NVML; otherwise scope as `Option::None` (feature-gated absence) rather than adding a new heavyweight dependency without discussion.
   - Metal/Vulkan: research first — Metal has no standard cross-vendor utilization API; Vulkan has no core-spec query either (vendor extensions only). It is acceptable and expected for these to return `None`/omit the field rather than fabricate a value. Do not synthesize a number from indirect signals (e.g., queue depth) and label it `compute` — that reintroduces the exact lying-zero problem this WI exists to fix.
2. Update `probe_vram_and_gpus` and `probe_cuda_vram` to call the new probe and populate `compute` with the real value when available, `null` (not `0`) when not.
   - This is a **breaking JSON shape change** for consumers currently reading `compute` as always-present `u32`. Document it in the endpoint's doc comment and bump nothing at the wire level (JSON has no version field currently) — just note the field is now `Option<u32>`.
3. Add per-device `tokens_per_sec` and `kv_cache` breakdown only if `grim-engine` already tracks these per-ordinal internally (check `Engine` struct for a `Vec<...>` keyed by device before assuming this needs new counters). If the engine is currently single-aggregate by design for non-disagg runs, do **not** add per-device fields to the JSON yet — that's speculative plumbing ahead of an actual multi-GPU consumer. Flag this as a follow-up gated on `grim-disagg` maturity, not part of this WI's done-criteria.

### Left/right limits
- Do not touch `tokens_per_sec` or `kv_cache` aggregate logic — confirmed real and correct, out of scope.
- Do not add a new dependency (NVML, rocm_smi crate) without checking `Cargo.toml` first for whether one is already a transitive dep of an existing backend crate.
- Do not fabricate `compute` from proxy signals (queue depth, memory bandwidth estimate). `None` is the correct value in the absence of a real utilization query — this is the entire point of the WI.
- Do not add per-device breakdown fields speculatively; gate on actual multi-GPU internal tracking existing.

### Gates
1. **Correctness**: on a real ROCm device, `compute` reflects `rocm-smi`'s reported utilization within reasonable polling-interval drift (compare manually during a load test). On backends without a utilization API, field is `null`, not `0`.
2. **Compile**: `cargo build -p grim-server --all-features` and per-backend feature combos (`--no-default-features --features cuda`, etc.).
3. **Architecture-cleanliness**: utilization probe lives in the same module as `vram_info` for each backend, same signature convention (`Option` return, ordinal-indexed).
4. **Performance (non-blocking)**: utilization query must not block the stats endpoint for more than ~5ms; if a backend's query is slow (e.g. shell-out), cache with a short TTL (~1s) rather than querying synchronously per request.

---

## WI-2: `grim doctor --model <path>` — pre-flight model/hardware compatibility prediction

### Why
`grim doctor` today verifies the *system* is healthy (ROCm present, service correct, sandbox enforced, wavefront size) but has no way to check whether a *specific model file* will actually load and run well on the detected hardware before the user attempts a load. Given `WeightFormat` has real per-architecture support tiers (`resolve_quant_mode` in `grim-backend-rocm/src/quantization.rs` already encodes fallback logic like "Fp8Native forbidden on RDNA2/3, falls back to Bf16"), the information needed to predict compatibility already exists on both sides — it's just never joined together before load time. Today a user finds out about a mismatch via a crash, a silent quality-degrading fallback, or an OOM.

### Where
- `grim-cli/src/doctor.rs` — extend `DoctorReport` and `run_doctor`, add a new code path gated on an optional `--model` arg.
- `grim-cli/src/main.rs` — extend the `Doctor` subcommand's clap args with `model: Option<PathBuf>`.
- `grim-format/src/gguf.rs` — `GgufFile`, `GgufTensorInfo`, `GrimMetadata`/`GrimMetadataV2` structs already carry architecture, tensor count, and (need to confirm) per-tensor dtype/shape — this is the size-estimation data source for GGUF.
- `grim-format/src/spec.rs` — `GrimTensorExt` for `.grim` files; check whether it stores enough to compute total resident bytes without loading all tensor data.
- `grim-garage/src/weight_format.rs` — `WeightFormat` enum and its RDNA/CDNA support tiers.
- `grim-backend-rocm/src/quantization.rs` — `resolve_quant_mode`, `arch_capability(arch: GcnArch)` — the existing compat-resolution logic to reuse, not reimplement.
- `grim-backend-rocm` device probe (already used in `doctor.rs`) for `GcnArch`/`gfx` target of the detected hardware.

### What already exists (read first)
- Hardware side: `doctor.rs::check_gpu_backend` already gets `RocmDevice::probe()`, `probe_host_gpu(ordinal)` → `gcn`, `wavefront_size`, `lds_size_bytes`. This is the "what hardware do we have" half, fully built.
- Compat side: `arch_capability(GcnArch) -> Caps` and `resolve_quant_mode(arch, requested) -> QuantMode` already encode which quant modes are natively supported vs. require fallback per architecture. **Do not reimplement this logic in `doctor.rs`** — call into it.
- Model side: `GgufFile { version, tensor_count, metadata: HashMap<String, GgufValue>, tensors: Vec<GgufTensorInfo>, data_start }` — confirm `GgufTensorInfo` has per-tensor byte size or shape+dtype (need to grep before building; if it only has shape+dtype, byte size is a straightforward derived computation, not new parsing).
- `.grim` side: `GrimMetadata`/`GrimMetadataV2` in `grim-format/src/gguf.rs` (note: despite the filename, confirm whether `.grim`-specific metadata types actually live here or in `grim-format/src/format.rs`/`spec.rs` — verify before writing the reader, the crate layout suggests format.rs or spec.rs is the .grim-native home and gguf.rs may only house GGUF-side + a shared override type).

### What to build
1. **`ModelFootprint` struct** (new, in `grim-format` — this belongs in the format-reading crate, not `grim-cli`, so `grim-garage`'s dashboard can reuse it later without a CLI dependency):
   ```rust
   pub struct ModelFootprint {
       pub architecture: String,        // e.g. "llama", "qwen2moe"
       pub param_count: u64,
       pub quant_format: WeightFormat,  // or existing quant enum if WeightFormat isn't the parse-time type
       pub estimated_weight_bytes: u64, // sum of tensor byte sizes from header, no tensor data load
       pub context_length_default: Option<u32>,
       pub is_moe: bool,
   }
   ```
   Populate from `GgufFile` (read header only — `GgufFile` parsing already avoids loading tensor data based on `data_start` being an offset, confirm this before assuming it's cheap) and from `.grim` header equivalently. **Read-header-only is a hard requirement** — this must be fast enough to run before every load attempt, not require streaming the weights.
2. **VRAM estimate function**: `estimate_vram_bytes(footprint: &ModelFootprint, context_length: u32, batch_size: u32) -> u64`. Formula: weight bytes (from footprint, already quant-aware) + KV cache estimate (reuse whatever sizing logic `grim-engine`'s `block_pool`/`kv_cache_telemetry` already uses for real allocations — check `grim-core/src/kv_cache.rs` for an existing size-per-token-per-layer calculation before deriving a new one) + a fixed activation/overhead margin (start conservative, e.g. 10-15% margin, and mark this constant clearly as a heuristic to be tuned against real measurements, not treated as exact).
3. **Compat check function**: `check_weight_format_support(format: WeightFormat, arch: GcnArch) -> CompatResult` where `CompatResult` is an enum `{ NativeSupport, FallbackSupport { to: QuantMode, reason: String }, Unsupported { reason: String } }`. This is a thin wrapper that calls `resolve_quant_mode`/`arch_capability` and classifies the result — again, reuse, don't reimplement.
4. **`doctor --model <path>` flow** in `grim-cli/src/doctor.rs`:
   - Parse header via the new `ModelFootprint::from_gguf_header`/`from_grim_header`.
   - Get detected hardware (`gcn` target, VRAM total via existing `probe_vram_and_gpus`-equivalent or the ROCm probe already in `doctor.rs`).
   - Run `check_weight_format_support` → print `[OK]`/`[WARN]`/`[ERR]` in the same style as existing doctor checks (match the `println!`/`eprintln!` + `report.errors`/`report.warnings` convention already established — do not introduce a second output style).
   - Run `estimate_vram_bytes` vs. detected free VRAM → `[OK]` if comfortably fits, `[WARN]` if tight (e.g. within 10% margin), `[ERR]` if it won't fit.
   - Extend the existing remediation-suggestion block (the `if err.contains(...)` chain at the end of `run_doctor`) with new suggestion branches, e.g. VRAM-insufficient → suggest a smaller quant tier by name (pull the suggestion from `WeightFormat`'s known smaller-bpw siblings, e.g. suggest Jay/Rook over Raven/Jackdaw if OOM-adjacent) and RDNA2/3-with-Raven → suggest Rook/Jay instead (native support vs. forced fallback).
5. **Wire into `main.rs`**: add `model: Option<PathBuf>` to the `Doctor` clap variant; if present, call the new check in addition to (not instead of) the existing system checks — `doctor --model x.gguf` should still run the full existing suite plus the new model section.

### Left/right limits
- Do not load any tensor data to compute the footprint — header-only. If a format's header doesn't carry enough info for a field (e.g. some GGUF files omit context length), leave that field `None` and don't guess.
- Do not reimplement `resolve_quant_mode`/`arch_capability` logic inside `doctor.rs` or the new format code — call the existing `grim-backend-rocm` functions. If they're not `pub` at the crate boundary currently, that's a small, explicitly-called-out prerequisite change (make them `pub`), not a rewrite.
- Do not extend this to CUDA/Vulkan/Metal compat prediction in this WI unless `arch_capability`-equivalent logic already exists for those backends — check first. If it doesn't, scope this WI to ROCm only (consistent with grim's AMD-first design center) and note CUDA/Vulkan/Metal pre-flight as an explicit follow-up, not silently absent.
- Do not turn this into a general "benchmark predictor" (predicted tokens/sec, etc.) — that requires empirical calibration data this plan doesn't scope. Stick to binary/tiered fit prediction (fits / tight / doesn't fit; native / fallback / unsupported).
- KV cache estimate constant/margin must be clearly labeled as a heuristic in a doc comment, with a `TODO(calibrate)` pointing at comparing predicted vs. actual `kv_cache_telemetry()` output post-load — that comparison is the natural correctness check but is explicitly out of scope to automate in this WI.

### Gates
1. **Correctness**: for at least 3 real model files (a `.grim` file and a GGUF file already used in existing golden tests, plus one deliberately oversized for the test GPU), the tool's fit/no-fit and native/fallback verdicts match manually-verified ground truth. This requires `TODO(gpu-verify)` on any claim about actual load success — the predictor's job is to warn, not guarantee.
2. **Compile**: `cargo build -p grim-format -p grim-cli`.
3. **Architecture-cleanliness**: `ModelFootprint` and the estimate functions live in `grim-format` (reusable by `grim-garage` dashboard later without pulling in `grim-cli`); `doctor.rs` only orchestrates and prints.
4. **Performance (non-blocking)**: header-only parse must complete in well under 1 second for a large (70B-class) model file — if the current GGUF/`.grim` header reader is already fast (likely, since it seeks past `data_start`), this should be trivially met; flag if not.

---

## WI-3: Constrained/structured decoding (`response_format: json_schema`, grammar-constrained generation)

### Why
Confirmed via direct grep: `response_format`/`json_schema` do not appear anywhere in `grim-server/src/*.rs`. This is a real, complete gap — vLLM and SGLang both ship this as core infrastructure (xgrammar/outlines-equivalent), and it's a common integration requirement for any OpenAI-API-compatible caller doing structured extraction or tool-call reliability. Grim's `Sampler` trait (`grim-core/src/sampler.rs`) is a clean, already-abstracted hook point — this is additive, not a rework.

### Where
- `grim-core/src/sampler.rs` — `trait Sampler { fn sample(&self, logits: &Tensor, history: &[u32]) -> Result<u32>; fn name(&self) -> &str; }`.
- `grim-server/src/lib.rs` — OpenAI-compatible request parsing (`/v1/chat/completions`, `/v1/completions`) needs a new optional `response_format` field on the request struct.
- New crate or module: `grim-nn` (if grammar state machines belong near model/generation logic) vs. a new `grim-constrain` crate (if this is substantial enough to warrant isolation — likely yes, given xgrammar/outlines are themselves dedicated libraries upstream). **Recommend a new crate `grim-constrain`** to keep `grim-core` (used everywhere, including backends) free of grammar-compilation dependencies (regex/JSON-schema-to-FSM compilation pulls in real dependency weight that shouldn't leak into every backend crate).
- `grim-cli/src/tui/worker.rs` / chat REPL — lower priority, but if `/v1/chat/completions` grows a `response_format` field, the interactive `grim tui` should eventually expose a `/schema` slash command; scope as a follow-up, not part of this WI's done-criteria.

### What already exists (read first)
- `Sampler` trait — confirmed minimal and clean: takes full logits tensor + history, returns one token. This is the correct interception point: a constrained sampler wraps an inner `Sampler`, masks/reweights `logits` before delegating, based on which tokens are valid continuations under the active grammar/schema state.
- `ThinkingLevel` enum in the same file shows the established pattern for request-level generation-control enums that get threaded from the OpenAI-compatible request struct down to the sampler — follow this pattern for how `response_format` should flow, rather than inventing a new threading mechanism.
- No existing grammar/FSM/schema-to-token-mask code anywhere in the workspace — this is fully new implementation, not wiring.

### What to build

This is the largest of the three items — break it into three sequential sub-work-items with independent gates, since a partial "JSON-mode-only" landing is a legitimate, shippable milestone before full JSON-Schema/grammar support.

**WI-3a: JSON-mode only (`response_format: {"type": "json_object"}`)**
1. New crate `grim-constrain` with a minimal JSON-syntax-only FSM: valid-next-token-set constrained to producing syntactically valid JSON (balanced braces/brackets/quotes, no bare schema validation yet). This is the smallest correctly-scoped first milestone — matches OpenAI's own tiered rollout (`json_object` shipped before `json_schema` historically) and de-risks the tokenizer-vocabulary integration before adding schema complexity.
2. `ConstrainedSampler<S: Sampler>` wrapping an inner sampler: on each `sample()` call, compute the current FSM state from `history`, get the valid-token mask from the FSM, apply it to `logits` (set invalid-token logits to `-inf` or equivalent before delegating to the inner sampler's actual sampling strategy — temperature/top-p/etc. still apply within the masked set).
3. Tokenizer-vocabulary bridge: the FSM operates on characters/bytes; token-level masking requires knowing, for each vocabulary token, whether appending it keeps the output on a valid FSM path. This is the expensive part upstream (xgrammar's whole value proposition is fast vocabulary-mask computation) — for this milestone, a naive per-token-per-step FSM simulation is acceptable if correctness-gated, with performance explicitly deferred to WI-3c.
4. Wire `response_format` field into the `/v1/chat/completions` and `/v1/completions` request structs in `grim-server`, threaded down to sampler construction the same way `ThinkingLevel`/`reasoning_effort` are threaded today (confirmed real and fixed per prior verification — follow that exact pattern).

**WI-3b: JSON-Schema-constrained (`response_format: {"type": "json_schema", "json_schema": {...}}`)**
1. Extend `grim-constrain` with a JSON-Schema → FSM/grammar compiler covering at minimum: `type`, `properties`, `required`, `enum`, `items`, nested `object`/`array`. Do not attempt full JSON-Schema spec coverage (`$ref`, `oneOf`/`anyOf` composition, format validators) in the first landing — scope to the subset that covers realistic tool-call/extraction schemas, and document unsupported schema features as an explicit rejection (`400` with a clear error) rather than silently ignoring them.
2. Schema compilation should happen once per request (cache-able if the same schema recurs across requests — consider an LRU keyed on schema hash, but only after correctness is proven; don't add caching complexity before the compiler itself is gated green).

**WI-3c: Performance — vocabulary mask precomputation**
1. Once WI-3a/b are correctness-gated, profile the naive per-step FSM simulation against a real vocabulary size (check the tokenizer vocab size grim actually uses — likely 32k-150k range depending on model family). If per-token latency is unacceptable (compare against `grim-engine`'s existing decode-step latency budget), build a precomputed transition-table structure (this is the core technique xgrammar uses — a pushdown-automaton-to-token-mask cache) rather than shipping the naive version as final.
2. This sub-item is explicitly gated *behind* WI-3a/b's correctness gates — do not optimize before the FSM/schema-compiler is proven correct on real cases.

### Left/right limits
- Do not build this as a call-out to an external process/service (no shelling to a Python `outlines`/`xgrammar` process) — grim's whole design principle is no Python bridge; this must be native Rust, consistent with the rest of the workspace.
- Do not couple `grim-constrain` to any single backend (ROCm/CUDA/etc.) — it operates purely on the `Tensor` logits and token vocabulary, backend-agnostic, same as the existing `Sampler` trait.
- Do not attempt grammar support (arbitrary CFG/Lark-style grammars, the way some xgrammar/outlines modes support) in this plan's scope — JSON-mode and JSON-Schema only. Full grammar support is a legitimate future WI but roughly doubles the compiler surface and isn't needed to close the concrete gap identified (OpenAI-API-compatible `response_format` parity).
- Do not skip the `400`-on-unsupported-schema-feature behavior in favor of best-effort partial constraint — silently under-constraining is worse than a clear rejection, since callers relying on schema conformance would get malformed output with no signal.

### Gates (apply per sub-item: WI-3a, then WI-3b, then WI-3c)
1. **Correctness**: for WI-3a, generate N samples under `json_object` mode and verify 100% are syntactically valid JSON via a standard parser (not the FSM's own state, to avoid a tautological test — this echoes the project's existing "tests must prove the real thing" discipline). For WI-3b, verify against a JSON-Schema validator library (or hand-written checks) that outputs conform to the schema, across schemas covering each supported keyword (`enum`, `required`, nested `object`, `items`).
2. **Compile**: `cargo build -p grim-constrain -p grim-server`.
3. **Architecture-cleanliness**: `grim-constrain` has no backend-specific dependencies; `Sampler` trait is unmodified (wrapping, not altering, the existing interface) — confirm no existing `Sampler` implementors need changes.
4. **Performance (non-blocking)**: WI-3c only — document measured per-token latency overhead of constrained vs. unconstrained sampling before/after the precomputed-mask optimization; no hard target set here without a real decode-latency budget to compare against, but the before/after delta must be reported, not assumed.

---

## Sequencing recommendation

1. **WI-1** first — smallest, most self-contained, and closes a real (if narrower-than-previously-claimed) gap.
2. **WI-2** second — depends on nothing from WI-1/3, high user-visible value, reuses existing compat logic almost entirely (low new-logic risk).
3. **WI-3a → WI-3b → WI-3c** last, sequentially — largest scope, and 3a alone is a legitimate, shippable milestone if time-boxed.

## Explicit exclusions (documented so this isn't re-derived later)
- No re-litigation of `/api/stats` tokens/sec or KV cache telemetry — confirmed real, out of scope.
- No CUDA/Vulkan/Metal pre-flight compat prediction unless those backends are confirmed to have `resolve_quant_mode`-equivalent logic (check before assuming absence).
- No full CFG/grammar support beyond JSON-mode/JSON-Schema in this plan.
- No automated calibration of the VRAM-estimate heuristic margin against real measurements — flagged as a `TODO(calibrate)`, not built here.
