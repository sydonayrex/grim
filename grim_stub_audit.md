# grim — Stub & TODO Audit

Source-verified against the uploaded tree (`grim-main/`). Every item below was
read in context, not matched by grep alone. Legitimate patterns — cfg-gated
platform fallbacks that error loudly, and trait-default methods every real
backend overrides — are excluded; they are correct design, not stubs.

Items are ranked by real-world impact: **silent wrongness** (worst — runs,
produces bad output, no error) ranks above **loud failure** (errors clearly)
ranks above **honestly-labeled future work**.

---

## Tier 1 — Silent wrongness (fixes model output without telling anyone)

### 1.1 Mamba forward pass discards real computation
**Where:** `crates/grim-models/mamba/src/lib.rs:235`
```rust
let _ = (dev, h_in);
```
**What exists:** Real weight loading, real selective-scan kernel dispatch
plumbing (`grim_selective_scan` HIP kernel exists in
`kernels/selective_scan.rs`).
**What's broken:** The forward function computes `dev`/`h_in` (device state,
hidden input) and then discards them instead of running them through the
scan. The function returns *something* — not an error — so a caller loading
a real Mamba checkpoint gets plausible-looking but wrong output.
**Fix:** Wire `dev`/`h_in` into the actual `grim_selective_scan` kernel
launch (the launcher already exists in the ROCm backend — confirm it's
reachable from this call site) or, if the scan isn't ready, replace the
silent discard with `Err(Error::Unimplemented(...))` until it is. An error
is strictly better than a wrong answer here.

### 1.2 RWKV time-mix discards k/v/r
**Where:** `crates/grim-models/mamba/src/rwkv.rs:168`
```rust
let _ = (k, v, r);
```
**What's broken:** Same shape as 1.1 — the RWKV time-mixing step receives
key/value/receptance tensors and drops them. `rwkv::KERNEL_SOURCE` exists in
the ROCm kernel set (`grim_rwkv_time_mix` is asserted present by
`compute_kernel_source_contains_phase2_kernels`), so the kernel itself is
compiled into every binary — it's just never called from this Rust-side
forward function.
**Fix:** Same as 1.1: wire the launch or fail loudly. Given the kernel
already compiles cleanly (per the existing unit test), this is likely the
smallest fix in this tier — the hard part (kernel authoring) is done.

### 1.3 Diffusion UNet discards conv weights
**Where:** `crates/grim-models/diffusion/src/unet.rs:104`
```rust
let _ = (&self.conv_w, &self.conv_b, self.hidden);
```
**What's broken:** Real conv weights (`conv_w`, `conv_b`) and hidden state
are loaded onto the struct and then never used in this method — the UNet
block returns without convolving anything.
**Fix:** Either implement the actual convolution (im2col + GEMM, or a direct
conv kernel — check whether one already exists under
`grim-backend-rocm/src/kernels/` before writing a new one) or hard-fail.
Diffusion is a lower-traffic path than the core LLM inference loop, so this
is lower urgency than 1.1/1.2, but it should not silently return zeros/noise
to a caller who thinks they're getting a real diffusion step.

### 1.4 Vision Transformer discards attention weights (two sites)
**Where:** `crates/grim-models/vision/src/vit.rs:115-116`
```rust
let _ = (h, seq, self.num_heads, self.head_dim);
let _ = (&self.wq, &self.wk, &self.wv, &self.wo);
```
**What's broken:** This is the worst instance in the tier — both the
*shape/config* parameters (heads, head_dim, sequence length) and the
*weight matrices* (Q/K/V/O projections) are discarded in the same method.
The attention block does nothing and returns unmodified input, or an
empty/placeholder tensor.
**Fix:** Implement the actual multi-head attention forward (standard
QKV-projection → scaled-dot-product → output-projection), reusing the
existing `grim_qkv_attention` or `grim_flash_attention` kernels already
compiled into the ROCm kernel bundle if the layout is compatible with
ViT's non-causal, no-KV-cache attention pattern. If a new kernel variant
is needed, that's real work — but the current state means anyone running a
vision-language checkpoint through grim today gets confidently-wrong output.

### 1.5 RCCL `sum_gradients` — feature flag doesn't matter, always fails
**Where:** `crates/grim-backend-rocm/src/rccl.rs:398-419`
```rust
pub fn sum_gradients(&self, _grads: &mut [f32]) -> Result<()> {
    if self.num_gpus <= 1 { return Ok(()); }
    // TODO: In a real RCCL build, call `ncclAllReduce` with
    // sum op and divide by num_gpus to average gradients.
    Err(Error::Backend("... requires the `rccl` feature flag".into()))
}
```
**Why this is worse than a normal cfg-gated stub:** Compare this to the
sibling function `tp_all_reduce` two functions above it, which correctly
branches on `#[cfg(feature = "rccl")]` and calls the real
`comm.all_reduce` when the feature is compiled in. `sum_gradients` has *no*
`#[cfg]` branch at all — it errors unconditionally for `num_gpus > 1`
**even when built with `--features rccl`**, and the parameter is
underscore-prefixed (`_grads`), meaning it was never wired even in the
feature-enabled path. It's not gated correctly, it's just not implemented,
mislabeled as "stub" in its own doc comment, and has zero callers anywhere
in the codebase.
**Fix:** Rewrite to match `tp_all_reduce`'s pattern —
`#[cfg(feature = "rccl")]` branch that calls `comm.all_reduce` on `_grads`
(dropping the underscore once it's used), non-feature branch keeps the
current error. Then wire a real caller into the training worker's
multi-GPU gradient-sync step (currently nothing calls this at all, so even
after the fix it does nothing until wired in).

---

## Tier 2 — Dangerous half-wiring (executes, but on the wrong conditions)

### 2.1 WMMA GEMM unconditionally enabled on incompatible hardware
**Where:** `crates/grim-backend-rocm/src/device/roc_device.rs:345`
```rust
wmma_gemm_config: Mutex::new(WmmaGemmConfig { enabled: true, wavefront_size: warp_size as u32 }),
```
**What's broken:** This flag is hardcoded `true` for every ROCm device at
init, despite the surrounding comment claiming "default-on for RDNA3/4
devices." The actual capability classifier
(`AMDArchitecture::is_wmma_capable()` / `is_mfma_capable()` in
`device/accel_features.rs`) is fully implemented but has **zero callers**
anywhere in the codebase — it's never consulted to set this flag or to gate
the dispatch. The dispatch itself (`launch_wmma_gemm`, called from the real
matmul path at `roc_device.rs:1440`) is genuinely wired now — this isn't
dead code, it fires on real inference requests.
**Impact:** WMMA instructions don't exist on RDNA2 (gfx1036) or CDNA
(gfx908/gfx90a/gfx942). On those devices this will either hard-fail the
kernel launch or, worse, silently assemble to something unintended.
**Fix:** Gate the `enabled` field at construction time:
```rust
enabled: self.gpu_arch.is_wmma_capable(),
```
using whatever `AMDArchitecture` value is already resolved during device
init (it's used elsewhere in this same file for the RDNA2/CDNA aliasing
logic). This is a one-line-plus-plumbing fix and should be the single
highest-priority item in this document — it can break a currently-working
install the moment this code path fires on the wrong GPU.

### 2.2 `launch_fused_dequant_gemm_f16` compiles but is provably unreachable
**Where:** `crates/grim-backend-rocm/src/device/roc_device.rs:3001`
```rust
/// TODO(WI-C): Kernel + config exist; wire dispatch in matmul path when enabled.
#[allow(dead_code)]
pub(crate) fn launch_fused_dequant_gemm_f16(...)
```
**What's broken:** Self-labeled by its own TODO and `#[allow(dead_code)]` —
the kernel and its config struct exist, but nothing in the matmul dispatch
chain calls this function. Separately, this kernel has **no corresponding
`Storage` variant in `dtype.rs`**, meaning even a correctly-wired call site
would have nowhere to put its output type today.
**Fix:** Two-part: (1) add the missing `Storage`/`DType` variant this kernel
needs, (2) add a dispatch branch in `launch_compute_kernel_with_solution` /
the matmul entry point analogous to the WI 2.4.4-2 decode-GEMM and WI-G
WMMA branches already present in that function — same `enabled` +
shape-guard pattern, consulting real GPU capability rather than a bare
`true`.

### 2.3 `RcclAllReduce::sum_gradients` — see 1.5
Listed here too because its downstream effect (multi-GPU training silently
never averages gradients across devices even when correctly configured) is
a wiring problem, not just a missing implementation. Cross-referenced, not
double-counted.

### 2.4 Speculative decoding "pickup" step called with dummy input
**Where:** `crates/grim-backend-rocm/src/speculative.rs:311`
```rust
let _ = (self.pickup)("dummy");
```
**What's broken:** Unlike every other `let _ = (...)` site audited, this one
is not silencing an unused-variable warning — it is **actively invoking**
the `pickup` closure with a hardcoded literal `"dummy"` string, then
discarding whatever it returns. The surrounding comment frames this as
intentional: "the pickup closure is reserved for the GPU kernel pickup
step; in the CPU-only primitive it isn't consumed but is asserted to be
non-trivial so the wiring is real." In practice this means the GPU pickup
path is exercised with fabricated input on every CPU-driven speculative
step, and its real output is thrown away — so nothing about this call
actually verifies the GPU pickup kernel behaves correctly on real
draft/target data. It's a hollow self-test disguised as wiring.
**Fix:** Either (a) remove the call entirely from the CPU-only primitive if
it truly has nothing to do here, or (b) if the intent is to smoke-test that
`pickup` is callable, assert on a real invariant (return type shape, no
panic) rather than silently discarding a dummy call — and add a genuine
integration test that runs `pickup` with real draft/target tensors on
hardware. As written, this line provides no verification value and reads
as verification.

---

## Tier 3 — Correctness gaps in the quantized training path

### 3.1 `quantized_matmul_backward_dx` — scales computed, never passed to kernel
**Where:** `crates/grim-backend-rocm/src/device/roc_device.rs` (scales
buffer + pointer built ~line 2273-2360; launcher signatures ~line 3226)
**What's broken:** `b_scales` is copied to a GPU buffer and
`b_scales_ptr` is computed, but none of the
`launch_fused_dequant_backward_gemm_*` launcher functions accept a scales
parameter in their signature — the pointer is dead. For Q4_K this is
harmless by luck (Q4_K blocks carry scale/min internally), but Q5_K, Q6_K,
Q2_K, and Q3_K backward gradients depend on externally-supplied scales that
are silently never reaching the kernel. Anyone fine-tuning with those quant
formats gets wrong gradients with no error.
**Fix:** Add a `b_scales_ptr: *const c_void` parameter to each
`launch_fused_dequant_backward_gemm_*` launcher and thread it through to the
actual kernel argument list (`arg(&mut ...)` calls), matching the pattern
already used for `dy_ptr`/`b_ptr`/`dx_ptr`. Verify against the existing
(but currently `#[ignore]`d, Q8_0-only) `quant_backward_gpu.rs` test,
extended to cover Q5_K/Q6_K/Q2_K/Q3_K specifically.

### 3.2 GPU numerics test exists but has never been proven to run
**Where:** `crates/grim-backend-rocm/tests/quant_backward_gpu.rs`
**What's broken:** Not a stub in the traditional sense, but functionally
equivalent to one — the test is `#[ignore]`d, requires manual hardware
execution, and there's no artifact (CI log, recorded output) in this tree
showing it has actually been run against real gfx1036/gfx110x silicon. The
`TODO(gpu-verify)` convention used throughout the codebase is honest about
this, but it means the "verified" claim for the primary dequant/GEMM path
is currently just aspirational.
**Fix:** Not a code fix — an operational one. Run the ignored test suite
against each target architecture (gfx1036, gfx110x, gfx1200, gfx942) at
least once, record the output, and either commit the results somewhere
durable or wire a CI runner with real hardware access. Extend coverage
beyond Q8_0 to the other quant formats per 3.1 before trusting the fix.

---

## Tier 4 — Honestly-labeled future work (no fix needed beyond doing it)

These are explicitly and correctly marked as incomplete-by-design, with
loud errors or debug-assertions rather than silent failures. Listed for
completeness, not urgency.

### 4.1 Split-K GEMM reduction — clamped to 1, documented, debug-asserted
**Where:** `crates/grim-backend-rocm/src/device/gemm_tuning.rs:25-40`
`split_k` is explicitly suggestion-only; every call site clamps it back to
1 before a real kernel launch, enforced by a `debug_assert_eq!` that will
panic in debug builds rather than silently write incomplete K-sums. This is
the correct way to leave work unfinished. **No fix needed** beyond
eventually writing the cross-block K-reduction kernel (WI 2.4.2/2.4.5).

### 4.2 RCCL point-to-point collectives — only allreduce/reduce_scatter/allgather exist
**Where:** `crates/grim-backend-rocm/src/rccl.rs` (FFI declarations,
top of file)
No `ncclBroadcast` or `ncclSend`/`ncclRecv` bindings exist at all — not
stubbed, simply absent. This limits multi-GPU wiring to the collective ops
already bound; genuine point-to-point transfer (needed for some pipeline-
parallel schemes) would need new FFI declarations plus wrapper functions
following the existing `all_reduce`/`reduce_scatter`/`all_gather` pattern.
**Fix:** Add `ncclBroadcast`/`ncclSend`/`ncclRecv` FFI bindings and Rust
wrappers only when tensor/pipeline parallelism work actually begins — no
value in stubbing them earlier.

### 4.3 Tensor/pipeline parallelism — absent under any naming
Not a stub because there's nothing to point at — confirmed absent from the
inference path in earlier review. Real design work, not a quick fix. Given
`all_reduce` now has a genuine caller in `grim-nn`, the data-parallel
gradient-sync half of multi-GPU has a foothold; tensor/pipeline
parallelism for serving large models across GPUs is still fully unstarted.

---

---

## Tier 0 — Cross-cutting severe findings (added after training-path and inference-path deep dives)

These were found in follow-up passes focused specifically on the training loop, inference serving loop, and model conversion pipeline. Ranked above Tier 1 because each affects the primary advertised use case (QLoRA fine-tuning, chat completion, model conversion) rather than a secondary architecture or dead code path.

### 0.1 QLoRA training only ever trains layer 0's Q-projection adapter
**Where:** `crates/grim-garage/src/jobs.rs`, the real per-step training loop
```rust
let (logits_id, logits_out) = match grim_autograd::apply_and_record_lora(
    &autograd_reg, &mut tape,
    0,                              // layer_idx — hardcoded
    LoRAInjectionPoint::QProj,      // point — hardcoded
    logits_base, logits_base_id, x_tensor, x_id,
) { ... }
```
**What's broken:** This is the only LoRA injection call site in the training worker. `LoRAInjectionRegistry::standard_qlora(num_layers, ...)` correctly builds real adapter configs for every layer x all seven standard QLoRA points (Q/K/V/O/Gate/Up/Down), and `TrainableParams` is correctly populated with the full set — but the forward/backward pass only ever exercises one of them. Every other adapter's gradient buffer stays at zero for the entire run.
**Why it's worse than "13/14ths untrained":** `optimizer.step()` iterates every parameter, not just the ones with nonzero gradient. Under AdamW's decoupled weight decay, a zero-gradient parameter still gets `w = w - lr * weight_decay * w` applied every step — every untouched adapter matrix is actively decayed toward zero, silently, for the whole training run. Loss curves will look plausible (real data, real single-projection signal) while the rest of the adapter set quietly degrades underneath.
**Fix:** Loop `apply_and_record_lora` over `LoRAInjectionPoint::all_standard_qlora() x 0..num_layers`, threading the LoRA-injected activations through the actual transformer forward pass layer by layer rather than projecting straight from one QProj application to the loss. This is effectively the missing multi-layer training forward pass, not a small patch.

### 0.2 `repeat_penalty` is parsed, defaulted, and documented — but never actually applied
**Where:** `crates/grim-server/src/lib.rs`, the sole `sample()` call site inside `sample_next_token`:
```rust
Some(t) => sampler.sample(&t, &[]).unwrap_or(step as u32),
```
**What's broken:** `history` is hardcoded to an empty slice on every call — the only call site in the server, shared by both streaming and non-streaming completion paths. `apply_repeat_penalty` bails immediately when `history.is_empty()`, so the correctly-implemented penalty math (proper positive/negative-logit handling, deduped history scan) never runs regardless of what the client requests.
**Why it's serious:** `repeat_penalty` is a real, documented, Ollama-compatible request field, defaulted to `1.0` (off) but expected to work at Ollama's own documented default of `1.10`. A client that explicitly sets it to suppress repetition gets no error and no effect. The `GreedySampler` doc comment even names the exact failure mode this causes ("without it, greedy decoding gets stuck emitting the same token forever") — the risk was known and the wiring still didn't happen.
**Fix:** Accumulate every sampled token per-request (a `HashMap<u64, Vec<u32>>` alongside the existing `request_last_token`) and pass the real history into `sampler.sample(...)` instead of `&[]`. Small, surgical fix — the penalty math itself is already correct.

### 0.3 GPTQ/EfficientQAT tensors are byte-reinterpreted as FP32 garbage during conversion
**Where:** `crates/grim-format/src/convert.rs`, `pack_tensors()`, `Storage::GroupInt` branch:
```rust
grim_tensor::dtype::Storage::GroupInt(_) => {
    // GroupInt not implemented - fallback
    raw.bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect::<Vec<f32>>()
}
```
**What's broken:** `grim-format/src/gptq.rs` is a real, complete GPTQ reader that correctly tags tensors `Storage::GroupInt(GpuIntConfig)`. But the conversion pipeline never dequantizes these — it reads the packed GPTQ integer bytes as if they were IEEE-754 `f32`, then runs that garbage through SmoothQuant scaling, SpinQuant rotation, and re-packing as if it were real weight data. The output `.grim` file is structurally valid (correct shapes, correct payload sizes) and will load and run — it just produces nonsense for every affected tensor. No error anywhere in this path.
**Reachable how:** `SafetensorsProvider` correctly identifies real-world GPTQ safetensors checkpoints (a common distribution format) and tags them `GroupInt` — this is a live path, not dead code.
**Fix — smaller than it looks:** A real, complete, already-tested GPTQ dequantizer already exists: `grim_quant::dequant_gptq_group_int` (used correctly by `crates/grim-nn/src/varbuilder.rs`'s `dequant_to_f32` for actual model loading/inference — confirmed working, with its own passing unit test `dequant_to_f32_group_int_unpacks_length_prefixed_segments`). `convert.rs`'s `pack_tensors()` just needs to unpack the same length-prefixed `qweight`/`qzeros`/`scales`/`g_idx` segments from `raw.bytes` (see `varbuilder.rs` lines ~440-470 for the exact unpacking logic to mirror) and call `dequant_gptq_group_int` instead of the byte-reinterpret fallback. This is a copy-the-existing-pattern fix, not new math. Until fixed, this branch should `Err(...)` rather than fabricate data.

### 0.4 EvoPress importance scoring silently treats every non-Q8_0 quantized tensor as unimportant
**Where:** `crates/grim-format/src/convert.rs`, EvoPress pre-pass inside `convert_to_grim`:
```rust
let data: Vec<f32> = match &raw.dtype.storage {
    grim_tensor::dtype::Storage::Native => /* real f32 read */,
    _ => grim_quant::dequant_q80(&raw.bytes, rows * cols).unwrap_or_default(),
};
```
**What's broken:** Every non-`Native` tensor — Q4_K, Q5_K, Q6_K, Q2_K, Q3_K, IQ*, GPTQ, NF4, FP8, all of it — is funneled through `dequant_q80`, a decoder specific to Q8_0's block layout. `dequant_q80` does correctly validate length and return `Err` on mismatch, but `.unwrap_or_default()` discards that into an empty vector, which then fails `randomized_svd_importance` and falls back to `scores.push(0.0)`. EvoPress's evolutionary search then treats every one of these tensors as zero-importance and preferentially crushes them to the lowest bitwidth — the opposite of the intended calibration behavior.
**Scope note:** This bug is confined to the *scoring/planning* phase. `pack_tensors()`'s actual bit-packing dispatch is correctly implemented per-scheme (verified: every `KQuantScheme`/`BlockDtype`/`FloatPackScheme` variant routes to its real dequantizer), so final tensor data isn't corrupted by this bug — only the compression-aggressiveness decision is. It also degrades SpQR salient-weight selection for the same tensors, since that reuses the same `f32_values`.
**Fix:** Replace the `_ => dequant_q80(...)` catch-all with the same per-scheme dispatch already correct in `pack_tensors()` (ideally extracted into one shared helper both call). Replace `.unwrap_or_default()` with real error propagation or a visible warning.

### 0.5 Ollama-compat options object silently drops top_k and repeat_penalty
Where: crates/grim-server/src/lib.rs, translate_options()

```rust
fn translate_options(req: &serde_json::Value, payload: &mut serde_json::Value) {
    if let Some(options) = req.get("options").and_then(|v| v.as_object()) {
        if let Some(temp) = options.get("temperature") { payload["temperature"] = temp.clone(); }
        if let Some(num_predict) = options.get("num_predict") { payload["max_tokens"] = num_predict.clone(); }
        if let Some(top_p) = options.get("top_p") { payload["top_p"] = top_p.clone(); }
        if let Some(stop) = options.get("stop") { payload["stop"] = stop.clone(); }
    }
}
```

What's broken: This is the bridge from Ollama's nested `options: {...}` request shape (used by /api/chat and /api/generate, handled by grim_chat/grim_generate) into the flat payload that chat_completions expects. It copies temperature, num_predict, top_p, and stop — but not top_k or repeat_penalty, even though chat_completions correctly parses both of those at the top level (see its KNOWN_FIELDS allowlist and the SamplingParams construction a few lines below it). An Ollama client sending `{"options": {"repeat_penalty": 1.1, "top_k": 40}}` — completely standard Ollama usage — has both values silently discarded before they reach the (separately broken, see 0.2) sampler.

Why the existing safety net doesn't catch this: chat_completions's KNOWN_FIELDS guard rejects unrecognized top-level fields with a 400 — a good defensive pattern — but top_k/repeat_penalty here are nested under options, which is itself a recognized field, so the guard never fires. The values are just quietly gone.

Fix: Add the two missing lines to translate_options:
```rust
if let Some(top_k) = options.get("top_k") { payload["top_k"] = top_k.clone(); }
if let Some(rp) = options.get("repeat_penalty") { payload["repeat_penalty"] = rp.clone(); }
```
Trivial fix, but should land together with 0.2 — fixing the sampler's empty-history bug without also fixing this leaves Ollama-compat clients (likely the majority of grim's actual traffic, given the project's Ollama-replacement positioning) still unable to set repeat penalty at all.

### 0.6 ColumnParallelLinear/RowParallelLinear claim to shard but don't — a second, competing TP implementation exists and is the real one
Where: crates/grim-nn/src/modules.rs

```rust
impl ColumnParallelLinear {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)   // no sharding — full unsharded matmul
    }
}
```

What's broken: The doc comment claims this "shards output features out_features / world_size across GPUs," but the implementation is a pure pass-through — every rank computes the complete, unsharded output. RowParallelLinear::forward does call a real dev.all_reduce(...), but correctness of row-parallel all-reduce depends on each rank holding only its row-shard of the weight and summing partial results; since nothing upstream shards the weight, every rank's contribution is already the full result, so a real all-reduce("sum") across ranks would silently scale the output by world_size if this were ever wired to actual multi-GPU execution.

Important context: crates/grim-nn/src/scythe2.rs contains a second, genuinely correct tensor-parallel implementation (Scythe2Linear) that does real column/row weight slicing (slice_output_dim/slice_input_dim) with correct shard concatenation and a golden-test gate. Neither implementation is currently called from grim-models' actual transformer forward pass — both are dead with respect to live inference — which is why this hasn't produced wrong output yet. But it means the crate has two competing "the TP layer" candidates, one broken and one real, and nothing routes to either.

Fix: When tensor parallelism work begins (existing item 4.3), wire model forward passes to Scythe2Linear, not ColumnParallelLinear/RowParallelLinear. Either delete the broken pair or fix them to actually shard (mirroring scythe2.rs's slicing) so a future contributor doesn't reach for the wrong one by name-matching "ColumnParallelLinear" against literature terminology and wire up the non-functional implementation.

---

## Tier 5 — Orphaned multi-GPU / speculative-decode plumbing (real kernels, zero production callers)

A context pass over the GPU-parallelism and multi-GPU surface. Positive baseline first, so this section isn't misread:

- **The `BackendDevice` trait surface is genuinely wired to real HIP kernels.** `quantized_matmul` / `quantized_matmul_backward_dx` (roc_device.rs:2082/2217) dispatch to real fused-dequant GEMM kernels for every KQuant scheme (Q4K…Q8_0, IQ2/IQ3/IQ4, FP8), and `qkv_attention` (2430), `flash_attention` (2660), `cross_attention` (2688) all launch real JIT kernels with gating and validation. Nothing in the trait surface is a stub.
- What *is* broken is the layer above it: three chunks of real, complete, GPU-side machinery exist with zero production callers. They are not silent wrongness (they never run) — they are the "wired but never called" trap: they look like live support, and the codebase's own docs route new work at them.

### 5.1 `qkv_attention_paged` / `launch_paged_attention` — orphaned paged-attention entry points
**Where:** `crates/grim-backend-rocm/src/device/roc_device.rs:4390` (inherent method `qkv_attention_paged`), `crates/grim-backend-rocm/src/kernels/qkv_attention.rs:582` (`launch_paged_attention`, JIT kernel `grim_qkv_attention_paged` at :259), re-exported at `lib.rs:125`.
**What exists:** A complete, real multi-query attention kernel over paged KV blocks (`block_tables` → `k_pages`/`v_pages`), with structural validation (3-D output, device-pointer checks) and a real launch.
**What's broken:** This is an inherent `pub fn` on `RocmDevice`, not a `BackendDevice` trait method — the trait's attention surface (`qkv_attention`/`flash_attention`/`cross_attention`, backend.rs:301/449/470) is fully implemented and real, but *paged* attention is not part of it. Grep for production callers: none — only `tests/paged_attention.rs:65` and the lib.rs re-export. Its natural consumer, `grim-speculative`, contains no reference at all; the only "attention" there is the toy score at `tiny_draft_backbone.rs:100`.
**Why it matters:** The paged-KV decode kernel every vLLM-style serving path needs exists, is tested, and is unreachable. Whoever wires up paged KV-cache serving will either never find it or assume it's already on the critical path.
**Fix — not a one-line swap:** grim-engine already owns a `KvBlockPool` (lib.rs:105) and does session-level prefix caching, but the live decode forward (`streaming_forward.rs:296`) still calls the *non-paged* `dev.qkv_attention` over contiguous K/V — no block tables, no page scatter. Consuming `qkv_attention_paged` means reworking that forward to allocate pages per request, maintain block tables, scatter K/V into pages, and call the paged kernel with its different output layout (`[batch, num_heads, head_dim]`). Real engine work, not a wiring patch. Either do that or add paged attention to the `BackendDevice` trait so a future consumer (e.g. `grim-speculative`) can reach it generically.

### 5.2 `tree_attention` / `launch_tree_attention` — orphaned tree-attention entry point
**Where:** `roc_device.rs:4442` (`tree_attention`), `kernels/qkv_attention.rs:658` (`launch_tree_attention`, kernel `grim_tree_attention` at :420).
**What exists:** A real tree/gamma-attention kernel — batch of `[1+gamma]` query positions against shared K/V, full GQA validation, head_dim ≤ 256, Wave64-aware. Real launch via `crate::launch_tree_attention`.
**What's broken:** Same shape as 5.1: inherent method, not trait; zero production callers (only `tests/tree_attention.rs:63` and the RED/GREEN contract test `tests/tree_attention_device.rs`). `grim-speculative` never references it.
**Why it matters:** Tree attention is the compute core of speculative decoding (validating a whole draft tree in one pass). grim ships the kernel and the crate that should consume it, and nothing connects them.
**Fix — requires building the draft-tree machinery first:** `grim-speculative`'s current verify path is token-level — `ConfidenceScheduler` (`scheduler.rs`) picks a verify length and consumes a `DraftBlock` (tokens + per-position scores); there is no tree representation, no `tree_parents`, no gamma-batched query assembly. `tree_attention` takes `q` as `[batch, 1+gamma, num_heads, head_dim]` plus a `tree_parents` tensor and returns 4-D output — that data structure and the draft→verify-tree construction step have to be written before the kernel is reachable. Significant new code in `grim-speculative` (or whichever crate owns the verify loop), then a direct call to `RocmDevice::tree_attention` or a trait addition.

### 5.3 The entire `rccl` module is unreachable from inference — real NCCL/RCCL collectives are dead code
**Where:** `crates/grim-backend-rocm/src/rccl.rs` — `RocmComm` FFI + `all_reduce` (:167), `reduce_scatter` (:196), `all_gather` (:225), `fuse_reduce_scatter` (:254), `fuse_all_gather` (:305), `p2p_memcpy_async` (:370), `tp_all_reduce` (:412), `RcclAllReduce` (:441), `sum_gradients` (:495); plus the P2P backing primitives `peer_access.rs::enable_peer_access` (:127) and `p2p_route.rs::copy_route` (:189).
**What exists:** A complete NCCL binding layer — `ncclAllReduce`/`ncclReduceScatter`/`ncclAllGather`/`ncclGroupStart`/`ncclGroupEnd`/`ncclSend`/`ncclRecv` + `hipMemcpyPeerAsync` — with correct `#[cfg(feature = "rccl")]` branches that call the real comms when compiled in.
**What's broken:** Zero production callers. Cross-crate grep shows no reference to `rccl`/`RocmComm`/`tp_all_reduce`/`fuse_*` outside grim-backend-rocm itself — none in grim-engine, grim-garage, grim-nn, grim-models, grim-server, grim-scheduler, or grim-speculative. The trait-level route is closed too: `BackendDevice::all_reduce` (backend.rs:366) is implemented on ROCm as a host-side CPU fan-in (roc_device.rs:2793 — correct for intra-process partials and documented as such), and `BackendDevice::comm_fuse_reduce` (backend.rs:390) is *never overridden* by the ROCm backend, so any caller gets the default `Err(Unimplemented)`. `comm_fuse_fan_in` (kernels/comm_fuse.rs:106) — the fused reduce-scatter/all-gather orchestrator — is likewise only called from its own tests. Net effect: **NCCL/RCCL is never reached from any production path.** Multi-GPU inference runs as a single rank; `num_gpus > 1` training hits the always-error `sum_gradients` (item 1.5).
**Why it matters:** This is the purest "wired but never called" case in the repo. The collective stack is complete, real, and build-gated, and grim's own docs route new work at it — `roc_device.rs:2788` explicitly tells readers to use `rccl::tp_all_reduce` for the cross-process reduce. That function has no callers.
**Fix — real multi-GPU work, not a wiring patch:** The training worker (`jobs.rs`) is a single process; `num_gpus > 1` currently only feeds a `ScythePlacement` (ranks/partition/routes) that is passed into the no-op `all_reduce_grads` (jobs.rs:905-915) — there is no process-per-GPU launch, no `RocmComm` init, and `TrainableParams::all_reduce_grads` (grim-autograd/src/param.rs:192) ignores the partition entirely (it self-adds the local gradient). Consuming RCCL requires: (1) a real multi-device world in the worker (or spawning rank processes), (2) `ncclCommInitRank` per rank, (3) partition-aware gradient sharding/reduction in `all_reduce_grads`, (4) `ncclAllReduce` + average. Alongside that, wire `tp_all_reduce` as the cross-process counterpart to `RowParallelLinear`/`Scythe2Linear` (0.6) and override `comm_fuse_reduce` on `RocmDevice` so `comm_fuse_fan_in` becomes reachable. Until then, gate these behind `#[cfg(feature = "rccl")]` or annotate them as non-live so nobody mistakes them for active multi-GPU support.

---

## Summary table

| # | Item | Tier | Effort | Blocks |
|---|------|------|--------|--------|
| 0.1 | QLoRA trains only layer-0 QProj, decays the rest | Silent, primary use case | Large (missing forward pass) | All QLoRA fine-tuning |
| 0.2 | `repeat_penalty` parsed but never applied | Silent, primary use case | Small | All chat completion |
| 0.3 | GPTQ tensors byte-reinterpreted as FP32 | Silent, primary use case | Medium | GPTQ model conversion |
| 0.4 | EvoPress scores non-Q8_0 tensors as unimportant | Silent, primary use case | Small | Calibrated quantization quality |
| 0.5 | Ollama options drops top_k/repeat_penalty | Silent, primary use case | Trivial | Ollama-compat clients specifically |
| 0.6 | Fake TP linear layers coexist with real ones | Dead but misleading | Small (delete or fix) | Future tensor-parallel work |
| 2.1 | WMMA unconditional-enable | Dangerous | Small (1-line + plumbing) | Any RDNA2/CDNA user today |
| 1.5 / 2.3 | `sum_gradients` mislabeled stub | Silent + wiring | Small | Multi-GPU training |
| 1.4 | ViT attention fully discarded | Silent | Medium | Vision-language inference |
| 1.1 | Mamba forward discarded | Silent | Medium | Mamba inference |
| 1.2 | RWKV time-mix discarded | Silent | Small (kernel exists) | RWKV inference |
| 1.3 | Diffusion UNet conv discarded | Silent | Medium | Diffusion inference |
| 2.4 | Speculative pickup dummy-call | Dangerous (false confidence) | Small | GPU speculative decode verification |
| 3.1 | Quant backward scales dropped | Correctness | Medium | Q5/Q6/Q2/Q3_K fine-tuning |
| 2.2 | Fused dequant GEMM f16 unreachable | Dead code | Medium (needs dtype work) | Perf, not correctness |
| 3.2 | GPU numerics test unproven | Verification gap | Operational | Trust in the whole stack |
| 4.1-4.3 | Honestly-labeled future work | None | Large | Multi-GPU scaling, split-K perf |
| 5.1 | `qkv_attention_paged`/`launch_paged_attention` orphaned | Dead but misleading | Large (engine decode rework) | Paged-KV serving, speculative decode |
| 5.2 | `tree_attention`/`launch_tree_attention` orphaned | Dead but misleading | Large (draft-tree machinery) | Speculative decode |
| 5.3 | Entire `rccl` module unreachable from inference | Dead but misleading | Large (multi-GPU world + grad sync) | Any real multi-GPU run |

Recommended fix order: 0.5 -> 0.2 -> 0.4 -> 2.1 -> 0.3 -> 1.5/2.3 -> 2.4 -> 1.2 -> 1.1 -> 1.4 -> 1.3 -> 3.1 -> 3.2 -> 2.2 -> 0.6 -> 0.1 -> 4.x. (0.5, 0.2, and 0.4 lead because they're the cheapest severe fixes — 0.3 also turned out cheap once the existing dequant_gptq_group_int function was found, so it moved up accordingly. 0.1 stays near the end despite its severity because it requires the missing multi-layer training forward pass, not a patch. 0.6 is low urgency since neither implementation is on a live path today — it just needs to be resolved before tensor-parallel work begins.)
