# Implementation Plan: Kernel Fusion and the Path to Minimum Launch Count

Status: DRAFT — verified against uploaded `crates.zip` source on 2026-09-15.
Companion to `PLAN-reduce-d2h-h2d.md`. That plan fixes *when* transfers and
launches happen (gating/defaults). This plan fixes *how many* launches the
work itself requires once gating is correct — i.e. kernel fusion — with a
dedicated section for MoE and ShortConv, which the dense-path graph-capture
work does not cover today.

Every finding below cites an exact file:line from the archive.

---

## 0. Summary of what's real vs. dormant vs. missing

| Category | Examples | Status |
|---|---|---|
| Fused + wired (dense decode) | `launch_fused_qkv_dot4`, `launch_fused_gate_up_dot4` (Q8_0 only), `grim_qkv_attention` | Real, active, correctly gated |
| Fused + wired (prefill) | `fused_mxfp4_gemm_qk_norm_rope_kv` | Real, but `mxfp4_qkv_attention` defaults off (see D2H/H2D plan A4) |
| Fused, tested, **dormant** (dense) | `fused_qkv_proj`, `fused_attn_o_proj` (`device_compute.rs:5016`, `:5051`), `fused_add_rms_norm` (`device_compute.rs:5698`) | Kernel exists, backend-level parity tests exist, **zero callers in `grim-models`** |
| Fused, tested, **dormant** (MoE) | `fused_moe_dispatch`, `fused_moe_dispatch_from_logits` (`shared_moe.rs:114,154`) — Charon grouped-expert kernel | Real, tested (`tests/golden_charon_moe_gpu.rs`), correctly gated (`GRIM_MOE_CHARON != "0"`, default on) — **but `lfm2.rs` never calls into `shared_moe.rs` at all** |
| Independently reimplemented, **not fused** (MoE) | `forward_moe_ffn_device` (`lfm2.rs:1755`) | Real on-device path, avoids D2H, but ~10 separate kernel launches per token and re-transposes expert weights from device memory every call — **and currently unreachable, see below** |
| **Loader bug — MoE unreachable** | `Lfm2Config` construction (`model_loader.rs:1274`, `:3032`) | Hardcodes `n_expert: 0` for every LFM2 load, including `Lfm2Moe`; real `hparams.expert_count`/`expert_used_count` fields exist and are used by 15+ other architectures in the same file but never wired for LFM2 — `is_moe` is always `false` today |
| Independently reimplemented, **not fused, has self-inflicted H2D** (ShortConv) | `shortconv_step_device` (`lfm2.rs:1685`) | Real device path, but unconditionally reformats+H2Ds the conv state on host every call (`lfm2.rs:1708-1714`), even in `decode_graph` mode where it should already be device-resident per the `!decode_graph` guard two lines later |

The load-bearing finding: **LFM2's MoE and ShortConv layers do not use the
dedicated fusion module (`shared_moe.rs`) that already exists and is tested
for exactly this purpose.** They have separate, less-fused, hand-rolled
device paths bolted directly into `lfm2.rs`. This is the direct cause of
the "MoE and ShortConv layers break the graph path" problem — not because
graph capture is fundamentally incompatible with MoE/ShortConv, but because
the *current* MoE/ShortConv code paths were never built to be capture-safe
or launch-minimal in the first place.

**Second, more severe finding, discovered during verification of this
plan (not in the original request):** the fusion/graph-capture question
for MoE is currently moot. Both `Lfm2Config` construction sites in the
model loader (`grim-engine/src/model_loader.rs:1274`, `:3032`) hardcode
`n_expert: 0` regardless of whether the loaded architecture is
`ModelArchitecture::Lfm2Moe`, so `is_moe` (`lfm2.rs:352`) is `false` for
every layer of every LFM2 model this codebase can currently load —
`forward_moe_ffn`/`forward_moe_ffn_device` are dead code today, not
merely unfused. This is a loader bug, not a fusion bug, and it blocks
everything else in the MoE section below. See M0.

---

## 1. Dense-path fusion (attention + FFN blocks)

### F1. Wire `fused_add_rms_norm` at both residual+norm seams
- **Where:** `lfm2.rs:1142-1145` (residual add → `ffn_norm`) and the
  equivalent seam feeding `attn_norm` at the top of the next layer
  (`lfm2.rs:579`, fed by the previous layer's final `add_tensors` at
  `lfm2.rs:1160`).
- **What exists:** `device_compute.rs:5698` — single HIP kernel
  `grim_add_rms_norm`, warp-per-row reduction, already computes
  `res_out = x + residual` and `y_out = rms_norm(res_out)` in one launch.
  Tested cross-backend (CPU/CUDA/Metal/Vulkan/ROCm all have
  `fused_add_rms_norm_tests.rs`).
- **Change:** replace the two-call pattern
  `add_tensors(...)` + `.forward(...)` with one
  `dev.fused_add_rms_norm(x, residual, weight, eps, out_shape)` call at
  both seams. `res_out` (the addition result) must still be threaded
  through as `x_added` for the final-residual path — the function already
  returns both `(y_out, res_out, handle)`, so no data is lost.
- **Yield:** 2 launches/layer removed (up to 64 across a 32-layer model).
- **Gate:** correctness (unit test comparing fused vs. unfused output,
  bit-identical or within float tolerance — reuse the existing
  `fused_add_rms_norm_tests.rs` pattern, extended to assert LFM2's forward
  pass is unaffected) → compile → architecture-cleanliness (must not
  break the `Lfm2LayerCache` residual bookkeeping used elsewhere) → perf
  (`TODO(gpu-verify)` on `syd-beasty`).
- **Risk:** low. This is the safest item in either plan — existing,
  tested, single-kernel, drop-in.

### F2. Wire `fused_qkv_proj` and `fused_attn_o_proj`
- **Where:** `lfm2.rs:780-782` (separate `wq`/`wk`/`wv` GEMVs) and the
  `wo` GEMM at the end of the attention block.
- **What exists:** `device_compute.rs:5016` (`fused_qkv_proj`) and
  `:5051` (`fused_attn_o_proj`) — defined, kernel-level parity-tested
  (`qkv_proj_fusion.rs`, `attn_o_proj_fusion.rs`), **zero integration
  test proving they're numerically correct inside LFM2's actual forward
  pass.**
- **Change:** before wiring into `lfm2.rs`, add an integration test in
  `grim-backend-tests` (or `grim-models/transformer/tests`) that runs a
  full LFM2 layer forward twice — once through the existing separate-GEMV
  path, once through the fused path — and asserts output parity within
  tolerance. Only after that passes, route `lfm2.rs`'s Q/K/V projection
  and output projection through these functions, gated the same way
  `GRIM_FUSED_FFN` gates the existing gate/up fusion (opt-out, weight-
  dtype-conditional where the fused kernel requires a specific quant
  format).
- **Yield:** up to 3 launches/layer removed for QKV (→1), 0-1 for output
  projection depending on whether `fused_attn_o_proj` folds in the
  residual add too (check its signature before assuming — do not assume
  parity with F1's scope).
- **Gate:** correctness (new integration parity test, described above,
  is a hard prerequisite, not optional) → compile →
  architecture-cleanliness → perf.
- **Risk:** medium. Unlike F1, there is currently no proof these
  kernels are correct in the exact tensor layout LFM2 uses (head
  reshape, GQA head-count mismatch between Q and K/V). Do not skip the
  parity test to save time.

### F3. Sequencing note: do fusion before widening graph capture
Fewer, larger kernels per layer means a smaller node count when
`hipGraphBeginCapture`/`End` wraps the sequence (the smackdown's own
example showed Q8_0 fusion shrinking a graph from ~800 to ~640 nodes).
Smaller graphs capture faster and are less likely to hit an operation
that aborts capture. Land F1 and F2 **before** doing the
`PLAN-reduce-d2h-h2d.md` A1 gate-unification work, or interleave them —
either order works, but do not widen graph capture to more model types
first and then retrofit fusion, since that means re-validating capture
twice.

---

## 2. MoE-specific resolution

This is the highest-value area. **Three** separate problems exist, not
two — a loader bug discovered during verification of this plan sits
upstream of the other two and must be fixed first, or the other two
fixes have nothing to act on.

### M0. Fix the LFM2 loader: MoE checkpoints never reach MoE code today (correctness, blocking)
- **Where:** `grim-engine/src/model_loader.rs:1274-1287` (HF/safetensors-
  config loader for `ModelArchitecture::Lfm2 | ModelArchitecture::Lfm2Moe`)
  and `:3032-3050` (native-GGUF loader, same match arm).
- **What's actually happening:** both `Lfm2Config` construction sites
  hardcode `n_expert: 0` and `n_layer_dense_lead: num_layers` regardless
  of the model being loaded — including when the matched architecture is
  explicitly `ModelArchitecture::Lfm2Moe`. Since
  `is_moe = layer_idx >= cfg.n_layer_dense_lead` (`lfm2.rs:352`), and
  `n_layer_dense_lead` is always set equal to the total layer count,
  **`is_moe` evaluates to `false` for every layer of every LFM2 model
  loaded through either path, including genuine `Lfm2Moe` checkpoints.**
  `forward_moe_ffn`/`forward_moe_ffn_device` are consequently **dead code
  in the current archive** — not merely unfused, unreachable.
- **This is not a hypothetical gap.** The struct field these two loader
  sites are supposed to read already exists, is already populated from
  real GGUF metadata (`grim-core/src/hyperparams.rs:277-282`, generic
  `{arch_name}.expert_count` / `{arch_name}.expert_used_count` keys —
  the same generic lookup every *other* MoE-capable architecture's loader
  in this file already uses, confirmed at 15+ other `Lfm2Config`-adjacent
  call sites such as `model_loader.rs:2398-2399`, `:2589-2590`,
  `:2662-2663`, `:3185-3187`, etc., all doing
  `hparams.expert_count.unwrap_or(N)` /
  `hparams.expert_used_count.unwrap_or(N)`). The two LFM2 sites are
  the only ones in this file that ignore fields already sitting in scope.
- **Change (both sites):**
  1. At `model_loader.rs:1274` (HF-config path): the in-scope `config`
     value is a format-specific struct (`SafetensorsConfig`-style,
     `model_loader.rs:309`) — confirm it exposes an expert-count field
     under this loader's naming convention (likely `num_local_experts`/
     `num_experts_per_tok` or similar HF-style keys, check
     `SafetensorsConfig`'s definition before assuming the exact field
     name) and thread it through to `n_expert` / `n_expert_used`, with
     `n_layer_dense_lead` set from a real "first MoE layer index" value
     if the format provides one, or left as `num_layers` (all-dense)
     only when the architecture truly has no MoE layers.
  2. At `model_loader.rs:3032` (native-GGUF path): `hparams` is already
     in scope (used for `hparams.num_layers`, `hparams.vocab_size`
     immediately above). Change
     `n_expert: 0` → `hparams.expert_count.unwrap_or(0)` and add
     `n_expert_used: hparams.expert_used_count.unwrap_or(1)`
     (matching the fallback convention used everywhere else in this
     file), and set `n_layer_dense_lead` from the real leading-dense-
     layer-count metadata key if LFM2/LFM2-MoE's GGUF spec defines one
     (check for an `lfm2.leading_dense_block_count`-style key, following
     the `expert_feed_forward_length` naming pattern established at
     `hyperparams.rs:283-286`); if no such key exists for this
     architecture, `n_layer_dense_lead` should be derived from which
     layers actually have expert tensors present in the checkpoint
     (`ws.pp("blk").pp(&i.to_string())` tensor presence check, same
     technique already used for `is_recr` detection at
     `model_loader.rs:1259-1269`), not left at a constant.
- **Gate:** correctness — this is a hard prerequisite for M1/M2 below,
  not parallel work. Add a loader-level test that constructs a synthetic
  GGUF/safetensors fixture with `expert_count > 0` for an `Lfm2Moe`
  architecture tag and asserts the resulting `Lfm2Config.n_expert > 0`
  and at least one layer has `is_moe == true`. Until this test exists
  and passes, M1 and M2 below are fixing dead code.
- **Risk:** low to implement (mechanical field threading, same pattern
  used 15+ times elsewhere in the same file), but **blocking** — nothing
  downstream in this section has value until it lands.

### M1. Route LFM2's MoE forward through `shared_moe.rs` instead of its own hand-rolled device path
- **Precondition:** M0 must land first. Everything below currently
  cannot be exercised by any model this archive's loaders can produce.
- **Where:** `lfm2.rs:1852` (`forward_moe_ffn`) and `:1755`
  (`forward_moe_ffn_device`) currently reimplement top-1 MoE dispatch
  from scratch, never calling `shared_moe::fused_moe_dispatch` or
  `fused_moe_dispatch_from_logits`.
- **What exists and is being skipped:** `shared_moe.rs:114-137` — Charon
  grouped-expert kernel (`grim_moe_fused_grouped`), described in its own
  doc comment as "one token-sorted launch, all active experts evaluated
  in-register with SiLU fused, output accumulated by atomicAdd" —
  already cross-checked against the per-expert loop by
  `tests/golden_charon_moe_gpu.rs` (≤1e-3 max-abs-diff). This is
  precisely the primitive LFM2's `forward_moe_ffn_device` should be
  calling instead of its current ~10-launch-per-token hand-rolled loop
  (`copy_slice_range` ×4, `transpose_2d` ×2, `matmul` ×2, `silu_mul` ×1,
  `mul_scalar` ×1, `copy_slice_range` ×1 — see `lfm2.rs:1790-1838`).
- **Also available and skipped:** `fused_moe_dispatch_from_logits`
  (`shared_moe.rs:154-...`) — fully device-resident routing computed via
  `grim_moe_route_topk`, explicitly designed to eliminate "the two host
  round-trips of the legacy path: the gate logits no longer D2H, and the
  routing table is no longer H2D-uploaded per launch." LFM2's current
  path does neither — it D2Hs the gate logits unconditionally at
  `lfm2.rs:1859` (`gate_logits.to_vec_f32()?`) **before** even attempting
  the device path, and does host-side softmax/argmax over the routing
  probabilities on the CPU (`lfm2.rs:1872-1888`) every single call,
  regardless of whether the device path later succeeds.
- **Change:**
  1. Replace the host-side gate-logit D2H + CPU softmax/argmax
     (`lfm2.rs:1859-1888`) with a call to `fused_moe_dispatch_from_logits`,
     which performs routing on-device via `grim_moe_route_topk`. This
     removes the unconditional D2H that currently happens even on the
     success path.
  2. Replace the entire body of `forward_moe_ffn_device`
     (`lfm2.rs:1755-1848`) with a call to `fused_moe_dispatch` /
     `fused_moe_dispatch_from_logits`, using the `CharonCache` weight-
     stacking mechanism already described in `shared_moe.rs:104-105`
     ("the stacked weight buffers are built here on first use and
     reused thereafter — no per-decode round-trips"). This directly
     fixes the per-token weight re-transpose problem (`lfm2.rs:1819-1820`,
     `transpose_2d` called on every token instead of once at load time).
  3. Note LFM2's current implementation is **top-1 only**
     (`lfm2.rs:1782-1788`, picks a single `best_e`), while
     `shared_moe.rs`'s interface takes `top_k` generically
     (`fused_moe_dispatch_from_logits` signature,
     `shared_moe.rs:160`) and `TokenRouting` structures for multi-expert
     combine weights. **Verified:** every current `Lfm2Config`
     construction site hardcodes `n_expert_used: 1` (or omits it via
     the M0 bug, which is worse) — so top-1 happens to be consistent
     with the *current, broken* loader output, but that consistency is
     an artifact of M0's bug, not evidence the top-1 assumption is
     correct for real LFM2-MoE checkpoints. Once M0 lands and
     `n_expert_used` is read from real metadata
     (`hparams.expert_used_count`), re-check whether any LFM2-MoE
     checkpoint actually specifies `expert_used_count > 1`. If so,
     `forward_moe_ffn`'s top-1 selection (`lfm2.rs:1904-1912`) is a
     latent correctness bug that M0 will newly expose (it cannot fire
     today because `is_moe` is always `false`, but will fire the
     moment M0 makes MoE layers reachable). Fix top-1→top-k as part of
     wiring M1's replacement dispatch, not as a follow-on — the
     `shared_moe.rs` interface already expects top-k
     (`TokenRouting`), so implementing M1 correctly *is* the fix,
     provided the top-1 assumption is not silently carried into the
     new call site.
- **Yield:** replaces ~10 launches/token + 1 D2H (gate logits) + repeated
  host softmax with 1-2 launches/token (routing kernel + grouped-expert
  kernel) and zero host round-trips on the steady-state path.
- **Gate:** correctness (M0 must be verified landed and its loader test
  passing — see M0's gate — before this item can be tested at all,
  since it needs a real `is_moe == true` layer to exercise). With M0
  landed: (a) confirm `expert_used_count` for actual LFM2-MoE checkpoints
  now flowing through the fixed loader, (b) implement M1's replacement
  dispatch as top-k from the start (matching `shared_moe.rs`'s
  `TokenRouting`/`top_k` interface) rather than porting the old top-1
  logic forward, (c) validate with CPU-reference oracle-parity testing
  (existing pattern) against a correct top-k reference, not against the
  old top-1 code (the old code cannot be assumed correct — it was
  effectively untested dead code per M0's finding) → compile →
  architecture-cleanliness (does this retire `forward_moe_ffn_device`
  entirely, or keep it as the CPU/non-Charon fallback path? Recommend:
  keep the existing scalar `forward_moe_ffn` host loop,
  `lfm2.rs:1898-1945`, as the true last-resort fallback for non-ROCm
  devices — but note it is *also* top-1-only today and will need the
  same top-k fix, since it's the fallback path M1's Charon dispatch
  degrades to; retire only the semi-fused `forward_moe_ffn_device` in
  favor of `shared_moe.rs`) → perf (`TODO(gpu-verify)`).

### M2. Make MoE layers graph-capturable
- **Where:** `block.rs`'s graph-capture gate (`block.rs:1324-1329`)
  excludes MoE implicitly — nothing in the condition list checks for MoE,
  but the smackdown's own investigation found MoE returns
  `Unimplemented` from a graph-mode path and falls back to eager. Given
  M1 replaces the MoE dispatch with device-resident routing and D2D
  weight caching, capture-safety becomes newly feasible: the blocking
  issue with the *old* code was the unconditional host D2H of gate
  logits (`lfm2.rs:1859`) and the host-side per-token argmax — both of
  which are host syncs that abort HIP graph capture (same class of bug
  documented in `lfm2.rs:1729`'s comment about `to_cpu_vec_f32` aborting
  capture with error 901). Once M1 removes those, re-attempt graph
  capture for MoE layers and see whether it succeeds without further
  changes.
- **Change:** after M1 lands, add MoE to the graph-capture integration
  test suite (`lfm2_graph_capture.rs`, `lfm2_graph_p4.rs`) with an actual
  MoE checkpoint or synthetic MoE config, and confirm
  `begin_capture()`/`forward_capture()`/`end_capture()` succeeds. If it
  still fails, capture the specific error (graph capture failures return
  identifiable HIP error codes) and file it as a scoped follow-up rather
  than leaving "MoE breaks graph mode" as an undifferentiated blanket
  statement.
- **Gate:** correctness (parity between captured-replay and eager MoE
  output) → compile → perf. This item is **blocked on M1**, not
  independent — do not attempt graph capture on the current hand-rolled
  MoE device path, since M1's D2H removal is very likely a prerequisite
  for capture to succeed at all.

---

## 3. ShortConv-specific resolution

### S1. Eliminate the per-token host round-trip of conv state
- **Where:** `lfm2.rs:1685-1735` (`shortconv_step_device`).
- **Problem:** every call reformats `state` (a host `&mut [f32]`) into a
  column-major layout (`st_cm`, `lfm2.rs:1708-1713`) and uploads it via
  `dev.from_cpu(&st_cm, ...)` (`lfm2.rs:1714`) — **unconditionally**,
  regardless of `decode_graph`. This is a genuine self-inflicted H2D that
  the existing `!decode_graph` guard two lines later (`lfm2.rs:1730`,
  gating only the *download* side) does not cover on the upload side.
  Given `decode_graph` in this function is meant to signal "keep
  everything device-resident for capture" (per the comment at
  `lfm2.rs:1729`, "host syncs... abort HIP graph capture"), the upload
  at line 1714 is equally capture-hostile and is presumably *why*
  ShortConv layers currently fall back to eager even when
  `GRIM_DECODE_GRAPH` is on — same root cause pattern as MoE's gate-logit
  D2H in M1.
- **Change:** maintain the conv state as a **device-resident ring buffer**
  from the point the cache is first allocated (analogous to
  `decode_attention_device`'s `k_dev`/`v_dev`/`past_dev` pattern already
  used for attention, `lfm2.rs:1594-1610`), updated in-place via a device
  kernel (`short_conv1d_causal_step` already exists and is called at
  `lfm2.rs:1717` — the fix is to source `state_st` from a persistent
  device buffer instead of re-uploading a freshly-reformatted host copy
  every call). This removes both the per-call CPU-side transpose loop
  (`lfm2.rs:1709-1713`, pure Rust, real CPU cost too, not just PCIe) and
  the H2D itself.
- **Yield:** removes 1 H2D + 1 host-side transpose loop per ShortConv
  layer per token — comparable in shape to the attention arena fix
  already covered by `decode_attention_device`, just not yet applied to
  ShortConv.
- **Gate:** correctness (device ring-buffer must produce byte-identical
  `state` evolution vs. the current host-mirrored version — add an
  oracle test analogous to the Q4_K GEMM scale-index regression tests
  already in the codebase's history) → compile →
  architecture-cleanliness (extend `Lfm2LayerCache::ShortConv` variant
  to hold a device-resident state buffer, mirroring how
  `Lfm2LayerCache::Attention` already holds `k_dev`/`v_dev`) → perf.

### S2. Confirm graph capture succeeds for ShortConv once S1 lands
- Same shape as M2: this is a downstream validation step, not
  independent work. Add ShortConv coverage to the graph-capture test
  suite once S1 removes the capture-aborting H2D, and treat "does
  capture actually succeed" as an empirical question to answer with a
  test, not an assumption to carry forward from the smackdown's
  characterization.

---

## 4. Graph-replay sampling path: implicit stream-identity dependency (correctness, cross-cutting)

Found while auditing the ROCm sampling path for bugs — not part of the
original fusion inventory, but directly relevant to graph-capture
correctness (M2/S2 depend on this class of issue, so it belongs here
rather than in the D2H/H2D plan).

### G1. `sample_storage_on_rocm` relies on stream identity that nothing enforces
- **Where:** `grim-cli/src/run.rs:964-986` (`sample_storage_on_rocm`, the
  post-replay GPU sampling call in `try_graph_decode_step`) →
  `grim-backend-rocm/src/kernels/device_sampler.rs:249-323` (`sample_impl`)
  → `device.launch_compute_kernel(...)` → `launch_compute_kernel_with_solution`
  (`device_compute.rs:5241`, fast-path launch at `:5279`) → `self.active_stream()`
  (`roc_device.rs:1061-1069`).
- **The bug:** `active_stream()` returns `capture_stream` only while
  `capture_active` is true (i.e., during the capture recording itself).
  During **replay**, `capture_active` is false, so it falls through to
  `self.default_stream`. Meanwhile the graph itself is launched via
  `graph.replay()` (`decode_graph_buffers.rs:534-544`,
  `hipGraphLaunch(self.exec, self.stream)`) on `graph.stream` — a stream
  obtained from `dev.get_stream_from_pool(0)` at graph-construction time
  (`lfm2_graph.rs:110-112`). `hipGraphLaunch` is asynchronous: it returns
  to the host immediately while the graph's ~800 nodes are still
  executing. The sampler kernel then launches on `active_stream()`
  (`default_stream`) and reads the logits tensor the graph just wrote,
  **with no `hipEventRecord`/`hipStreamWaitEvent` pairing anywhere in
  this call chain** between `graph.stream` and `default_stream` —
  verified: zero matches for either primitive in `lfm2_graph.rs` or
  `run.rs`.
- **Why it doesn't misbehave today:** `default_stream` is initialized as
  `streams.first()` (`roc_device.rs:729`) — i.e., `stream_pool[0]` — and
  `get_or_create_decode_graph` always requests pool index `0`
  (`lfm2_graph.rs:111`, hardcoded; it is the **only** call site of
  `get_stream_from_pool` in the archive). So `graph.stream` and
  `default_stream` are the same physical HIP stream today, and ordering
  is correct only because of normal same-stream program order — not
  because any code establishes or documents that requirement. This is
  an implicit invariant, not a guaranteed one: nothing in the type
  system, a comment, or a runtime assertion ties `active_stream()`'s
  replay-time fallback to `graph.stream`.
- **Why this matters for this plan specifically:** M2 and S2 both
  propose adding MoE/ShortConv coverage to the graph-capture test
  suite, and any future work toward running multiple concurrent decode
  graphs (e.g., batched/multi-session serving, or splitting streams
  across `syd-beasty`'s dual GPUs) is a natural next step once dense-path
  capture is reliable. Either of those changes — a second decode graph
  requesting a non-zero pool index, or any refactor of `active_stream()`
  that changes its replay-time fallback — would silently break this
  invariant with no test currently able to catch it, since no existing
  test forces the pool indices apart.
- **Change:**
  1. Make the dependency explicit instead of implicit: either (a) thread
     `graph.stream` through to `sample_storage_on_rocm`/`sample_impl`
     as an explicit parameter, and have `sample_impl` call
     `launch_compute_kernel_on_stream(..., stream, ...)`
     (`device_compute.rs:3646`, already exists, already used by the
     capture-time fast path per its own doc comment at `:3655-3657`)
     instead of the ambient `active_stream()`-resolving
     `launch_compute_kernel`; or (b) add an explicit
     `hipEventRecord` on `graph.stream` immediately after
     `hipGraphLaunch` in `graph.replay()`, and a matching
     `hipStreamWaitEvent` on `active_stream()` before the sampler
     kernel launches. Option (a) is preferred — it removes the
     dependency on stream identity entirely rather than papering over
     it with a synchronization primitive that itself depends on the
     event being on the right stream.
  2. Add a regression test that forces `graph.stream` and
     `default_stream` to be **different** pool slots (e.g., a test-only
     constructor path that requests pool index `1` instead of the
     hardcoded `0`) and asserts sampled-token correctness still holds.
     This is the only way to prove the fix actually removes the
     dependency rather than continuing to pass by coincidence — the
     existing test suite cannot currently distinguish "correctly
     synchronized" from "accidentally same stream," since it never
     varies the pool index.
  3. Document the invariant at the `get_stream_from_pool(0)` call site
     (`lfm2_graph.rs:111`) regardless of whether (1) is implemented
     immediately, so a future change to concurrent-graph support does
     not reintroduce this silently.
- **Gate:** correctness — this is a genuine race-condition-in-waiting,
  not a performance item, so it should land ahead of or alongside M2/S2
  rather than after. The regression test in step 2 is a hard
  prerequisite for calling this fixed, not optional polish → compile →
  architecture-cleanliness (prefer option (a) over (b) per the reasoning
  above) → perf (should be a no-op on `syd-beasty`'s current single-
  decode-graph usage; the value is in preventing a future regression,
  not in a measurable speedup today).
- **Risk of NOT fixing:** low today (single decode graph, hardcoded pool
  index 0), but this is exactly the kind of bug that survives correct
  for a long time and then reappears as a nondeterministic, hard-to-
  reproduce wrong-token bug the moment someone touches stream pooling
  for an unrelated reason (e.g., multi-GPU work already tracked
  elsewhere in this project's history, or adding a second concurrent
  decode graph for batched serving).



1. **F1** — cheapest, safest, do immediately regardless of what else is
   in flight; independent of everything else in this plan.
2. **M0** — fix the loader (`n_expert`/`n_expert_used`/
   `n_layer_dense_lead` wiring). Blocking for M1/M2. Do this before any
   other MoE work — until it lands there is no way to even test whether
   `forward_moe_ffn` produces correct or incorrect output, because it
   cannot currently run.
3. **S1** — independent of MoE, similar shape to already-proven attention
   arena work, moderate effort. Can run in parallel with M0.
4. **M1** — now includes the top-1→top-k fix as part of wiring the
   replacement dispatch (see M1's updated correctness gate). Strictly
   after M0.
5. **F2** — after F1, once the parity-test prerequisite is built.
   Independent of the M-track.
6. **M2, S2** — graph-capture validation for MoE/ShortConv, strictly
   after M1/S1 respectively; do not attempt earlier since the current
   host-sync patterns in both paths are very likely why capture already
   fails for these layer types.

## 5. What NOT to do
- Do not wire `fused_qkv_proj`/`fused_attn_o_proj` (F2) without the new
  integration parity test — unlike F1, there is currently no proof of
  correctness inside LFM2's actual tensor layout, only isolated kernel
  tests.
- Do not attempt MoE or ShortConv graph capture (M2, S2) before their
  respective host-sync removals (M1, S1) land — the current code has
  host syncs on every token that will abort capture regardless of what
  the graph-capture gating logic says, so attempting capture first will
  just reproduce the existing "falls back to eager" behavior and waste
  validation effort.
- Do not attempt M1 (wiring `shared_moe.rs`) before M0 lands — there is
  currently no reachable code path that constructs an LFM2 layer with
  `is_moe == true`, so M1 work would be unverifiable until the loader is
  fixed.
- Do not port `forward_moe_ffn`'s existing top-1 selection logic forward
  into the new `shared_moe.rs`-based dispatch under the assumption it's
  already correct — it has never been exercised by a real MoE load in
  this archive (per M0's finding), so its correctness is unverified, not
  merely unoptimized. Implement top-k from the `shared_moe.rs` interface
  directly instead of preserving the old logic's shape.
