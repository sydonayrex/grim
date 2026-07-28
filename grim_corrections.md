# Grim — Verification Findings & Required Fixes (running document)

Two review passes are combined here: (A) the `grim-backend-rocm` test suite, and (B) the
GGUF-load → CLI-chat → server-chat pipeline (`grim-cli`, `grim-engine`, `grim-server`). Method
for both: manual source read-through and cross-referencing against real trait/impl
definitions, not execution — **no Rust toolchain was available in this environment**, so
nothing below was compiled or run. Confidence is noted per item; items marked "confirmed" had
their full mechanism traced end-to-end through real source (definition → call site → effect).

---

## Summary table

| # | Area | Issue | Severity | Confidence |
|---|---|---|---|---|
| A.1 | `grim-backend-rocm` tests | 3 golden-mutation tests upload dequantized f32 via the wrong path, mislabeled as packed data | Test gives false confidence; silently never exercised (env-gated off by default) | Confirmed |
| A.2 | `grim-backend-rocm` tests | 2 GPU tests missing the suite's standard CI-safety gate | Hard-fails `cargo test` on non-GPU machines | Confirmed |
| A.3 | `grim-backend-rocm` tests | 8 "source contains symbol name" tests prove nothing about kernel correctness | Low — cheap tripwire only, not a correctness gap | Confirmed |
| A.4 | `grim-backend-rocm` tests | 2 `assert!(true, ...)` dead assertions | Low — disclosed as link-time-only checks | Confirmed |
| A.5 | `grim-backend-rocm` tests | 1 no-op test computes values and discards them | Cosmetic | Confirmed |
| A.6 | `grim-quant`/`grim-backend-rocm` | No live GPU test exists anywhere for `quantized_matmul_backward_dx` | Coverage gap, not a bug | Confirmed |
| B.1 | `grim-cli` | `run.rs` defines a private, less-capable `load_model_from_gguf` that shadows the real `grim_engine` one via a missing `use` import | **GGUF chat is broken/degraded for many architectures** | Confirmed |
| B.2 | `grim-cli` | Shadow GGUF loader hardcodes `Device::Cpu` for Mamba and Bert regardless of resolved device | GPU ignored for those architectures | Confirmed |
| B.3 | `grim-engine` | `enqueue_request`/`enqueue_request_with_kv` hardcode `Device::Cpu` for session construction | Same hardcoded-CPU pattern as B.2, in the engine's own request path | Confirmed |
| B.4 | `grim-cli` | Interactive REPL (`grim run model.gguf` with no prompt) reloads the model and rebuilds the session from scratch on every line typed | No real multi-turn memory; very slow after turn 1 | Confirmed |
| B.5 | `grim-cli` | No chat template is applied anywhere in the one-shot or REPL prompt path | Instruction-tuned models will underperform vs. Ollama for the same GGUF | Confirmed |
| B.6 | `grim-server` | `/v1/chat/completions` (streaming and non-streaming) hardcodes the prompt to the literal string `"Hello"`, ignoring the request's `messages` entirely | **Every chat request — OpenAI-compat and Ollama-compat alike — generates from `"Hello"`, not the user's input.** | Confirmed |

Rows B.1 and B.6 are the two blocking bugs for "can I get GGUF chat working like Ollama."
B.1 breaks/degrades loading; B.6 breaks generation content even once loading is fixed.

---

# Part A — `grim-backend-rocm` test suite

**Scope:** Static review of `grim-backend-rocm/{src,tests}` against real implementations in
`grim-tensor`, `grim-quant`, `grim-format`. All four crates were available for this pass.

## A.1 Three golden mutation tests upload data through the wrong path (confirmed)

**Files:**
- `tests/golden_q4k_gpu_mutation.rs`
- `tests/golden_raven_fp8_gpu_mutation.rs`
- `tests/golden_jay_magpie_gpu_mutation.rs`

**Defect:** All three call `BackendDevice::from_cpu(&dev, &b_dequant, &b_shape, <quantized_dtype>)`,
passing an already-**dequantized `f32` buffer** tagged with a **packed/quantized `DType`**
(`Storage::KQuant(Q4K)`, `Storage::Block(Fp8)`, `Storage::FloatPack(MxFp4)` respectively).

**Root cause, traced through the call chain:**

1. `BackendDevice::from_cpu` (`grim-tensor/src/backend.rs:189`) is typed `fn from_cpu(&self, data: &[f32], ...)` — it is *always* the plain-float upload path, regardless of the `dtype` argument passed alongside it.
2. `RocmDevice::from_cpu` (`grim-backend-rocm/src/device/roc_device.rs:1208`) forwards directly to `RocmStorage::copy_from_host`.
3. `RocmStorage::copy_from_host` (`grim-backend-rocm/src/memory/storage.rs:91`) branches **only on `dtype.arith`**: `F16` and `BF16` get element-wise converted before upload; every other arith value (including `F32`, which is what all three quantized `DType`s in these tests carry) falls into the default arm and does a **raw `hipMemcpy` of the `f32` slice's bytes, unmodified**. The function's own doc comment concedes this: *"pulls the bytes from a `&[f32]` (the only dtype currently wired through)."*
4. Meanwhile `RocmStorage::alloc_gpu` sizes the destination buffer as `shape.elem_count() * dtype_byte_size(&dtype)` — using the **packed** per-element size for a quantized `dtype` (fractional bytes/element for Q4K and MXFP4, 1 byte/element for FP8), not the 4 bytes/element the source `f32` data actually occupies.
5. Net effect: `copy_from_host` copies `storage.bytes` (small, packed-sized) starting from the `f32` host pointer. This is **not** a crash or an overflow — it silently **truncates** the source, copying only the first `packed_size` bytes of the `f32` array's raw bit pattern, then the kernel reinterprets those bits as if they were quantized codes (Q4K super-block headers, FP8 E4M3 codes, or MXFP4 nibbles). The uploaded data is meaningless.

**Consequence:** None of the three tests can currently validate what their names claim
(dequant-GEMM correctness on real GPU hardware). If run with `GRIM_RUN_GPU_TESTS=1` against
real hardware, all three would very likely fail the `max_err < 1e-3` assertion — but that
failure would indict the *test harness*, not the kernel under test. Because all three are
gated behind `GRIM_RUN_GPU_TESTS` (unset by default) and silently return `Ok(())` otherwise,
this appears to have never actually been exercised.

**Fix — identical shape in all three files:** use `from_cpu_bytes(&packed_bytes, &b_shape, dtype)`
with the already-quantized byte buffer, not `from_cpu` with the dequantized `f32` buffer.
The packed bytes are already computed and then discarded in two of the three files:

| File | Already has packed bytes as | Currently discards it, uploads instead |
|---|---|---|
| `golden_q4k_gpu_mutation.rs` | `b_packed` (from `quant_q4k`) | `b_dequant` via `from_cpu` |
| `golden_jay_magpie_gpu_mutation.rs` | `b_codes` (from `f32_to_mxfp4_e2m1`) | `b_dequant` via `from_cpu` |
| `golden_raven_fp8_gpu_mutation.rs` | `fp8_bytes` (already raw codes) | `b_dequant` via `from_cpu` (never needed `dequant_fp8` for the GPU side at all) |

**Suggested patch — `golden_q4k_gpu_mutation.rs`:**
```rust
// was:
let b_dev = BackendDevice::from_cpu(&dev, &b_dequant, &b_shape, q4k_dtype)?;
// should be:
let b_dev = BackendDevice::from_cpu_bytes(&dev, &b_packed, &b_shape, q4k_dtype)?;
```

**Suggested patch — `golden_raven_fp8_gpu_mutation.rs`:**
```rust
// was:
let b_dev = BackendDevice::from_cpu(&dev, &b_dequant, &b_shape, fp8_dtype)?;
// should be:
let b_dev = BackendDevice::from_cpu_bytes(&dev, &fp8_bytes, &b_shape, fp8_dtype)?;
```

**Suggested patch — `golden_jay_magpie_gpu_mutation.rs`:**
```rust
// was:
let b_dev = BackendDevice::from_cpu(&dev, &b_dequant, &b_shape, mxfp4_dtype)?;
// should be:
let b_dev = BackendDevice::from_cpu_bytes(&dev, &b_codes, &b_shape, mxfp4_dtype)?;
```

Note: confirm `from_cpu_bytes`'s exact signature takes `&dev` as receiver the same way
(`BackendDevice::from_cpu_bytes(&dev, ...)` vs `dev.from_cpu_bytes(...)`) — both forms appear
elsewhere in the suite; either compiles, pick whichever the surrounding file already uses for
consistency (these three use the trait-qualified form for `from_cpu`, so mirror that).

---

## A.2 Two GPU tests are missing the suite-standard CI-safety gate (confirmed)

**Files:**
- `tests/q8_0_diag.rs::q8_0_kernel_matches_cpu_dequant`
- `tests/kv_dequant_attention_gpu.rs::gpu_fused_attention_matches_cpu_reference`

**Defect:** Every other GPU-dependent test in this suite (12+ files checked, including the
three golden-mutation tests above, `wmma_gemm.rs`, `caching_allocator_reuse.rs`,
`hybrid_attention.rs`, etc.) follows the same pattern: check the `GRIM_RUN_GPU_TESTS` env
var, return/skip cleanly if unset, so `cargo test` is safe to run on any machine including
CPU-only CI. These two files break that convention — they call `RocmDevice::new(0)`
unconditionally at the top of the test body and `.unwrap()` every subsequent GPU call, with
no env-var check and no `#[ignore]` attribute.

By contrast, `tests/kv_dequant_perf.rs::gpu_fused_attn_decode_throughput_vs_dense` — which
exercises very similar code paths — correctly uses `#[ignore]` to make it opt-in only.

**Consequence:** `cargo test -p grim-backend-rocm` will hard-fail on any machine without a
real ROCm device present (e.g. any CPU-only CI runner), rather than skipping cleanly like
every sibling test in the suite.

**Fix options (pick one, consistent with the rest of the suite):**

1. Match the `GRIM_RUN_GPU_TESTS` pattern used everywhere else:
```rust
const GPU_TEST_ENV: &str = "GRIM_RUN_GPU_TESTS";
fn gpu_device() -> Option<RocmDevice> {
    if std::env::var(GPU_TEST_ENV).is_err() { return None; }
    match std::panic::catch_unwind(|| RocmDevice::new(0)) {
        Ok(d) => Some(d),
        Err(_) => None,
    }
}
// then in the test body:
let Some(dev) = gpu_device() else { return Ok(()); };
```
2. Or, if these were only ever meant to run manually on real hardware (the `q8_0_diag.rs`
   header comment — *"Run with: cargo test ... -- --nocapture"* — suggests this was the
   intent), add `#[ignore]` to match `kv_dequant_perf.rs`'s convention instead.

Either is a one-line-per-test fix; the important thing is picking one and applying it
consistently, since right now the suite has three different conventions for the same problem
(env-gate / `#[ignore]` / nothing) across otherwise-similar tests.

---

## A.3 Confirmed weak/tautological tests (non-blocking, but flagged as prior art)

### A.3.1 "Source contains symbol name" tests — proves nothing about kernel behavior

**Instances found:**
- `src/kernels/wmma_gemm.rs::source_contains_wmma_kernel_entry`
- `src/kernels/q2k_gemm.rs::q2k_kernel_source_contains_entries`
- `src/kernels/q3k_gemm.rs::q3k_kernel_source_contains_entries`
- `src/kernels/q5k_gemm.rs::q5k_kernel_source_contains_entries`
- `src/kernels/q6k_gemm.rs::q6k_kernel_source_contains_entries`
- `src/kernels/q8_0_dequant.rs::q8_0_kernel_source_contains_entry`
- `src/kernels/source_asm.rs::compute_kernel_source_contains_both_sub_sources`
- `src/kernels/source_asm.rs::compute_kernel_source_contains_phase2_kernels`

**Issue:** These assert `KERNEL_SOURCE.contains("extern \"C\" __global__ void grim_some_kernel")`.
This proves the string literal exists somewhere in the Rust-embedded HIP source — it proves
nothing about whether the kernel compiles under HIPRTC, launches without error, or computes
the correct result. A typo inside the kernel body, a wrong loop bound, or a completely
inverted sign would all pass this test unchanged.

**Not recommended as a blocking fix** — these are cheap smoke tests that do catch one real
regression class (accidentally deleting/renaming the kernel entry point so JIT lookup fails
at runtime with a confusing error), so removing them outright loses a small amount of value.
But they should never be mistaken for correctness coverage. Where a real numerics test exists
for the same kernel elsewhere in the suite (as it does for WMMA via
`tests/wmma_gemm.rs::test_wmma_gemm_infrastructure_and_correctness`), that's the test that
matters; the `contains()` check is just a cheap tripwire.

### A.3.2 `assert!(true, ...)` — dead assertions

**Instances found:**
- `tests/rccl.rs::rocm_comm_ffi_linked`
- `tests/rccl.rs::p2p_ffi_linked`

**Issue:** Both are literally `assert!(true, "... (build-time check)")`. If the test function
is reached at all, the crate already linked successfully against `librccl.so` — that's a
compile/link-time fact, not something the runtime assertion is checking. The assertion is
dead weight; it cannot fail. This is disclosed reasonably honestly in the surrounding doc
comments ("we assert only that the FFI is wired... not a runtime call"), so it's not
misleading in the way the golden-mutation bug above is — but it should be replaced with
either a real (minimal, safe) runtime call, or removed and the intent captured as a doc
comment instead of a fake test function.

### A.3.3 No-op "truthy for the lint detector" test

**Instance:** `tests/quantization.rs::gcn_arch_rounds_known_arch_strings_to_canonical_variant`

Computes four `gcn_arch(...)` values and discards them via `let _ = (...)`, with a comment
admitting this is "truthy for the lint detector." Asserts nothing. Harmless — the same logic
is properly asserted two tests later
(`gcn_arch_with_revision_hex_returns_canonical_variant`, `gcn_arch_partition_is_clean_per_buckets`)
— but the function name overclaims relative to what it does. Low priority; consider deleting
or merging into the assertion-bearing tests nearby.

---

## A.4 Verified as sound (spot-checked, no issues found)

For calibration — these were checked in similar depth to the flagged items above and found
to have real, meaningful assertions against real computed values, with no hardware-dependency
gating problems:

- `tests/select_gemm_algo_dispatch.rs` — includes an unusual but well-constructed static
  source-scanner (`every_gemm_call_site_uses_select_gemm_algo`) that walks
  `device/roc_device.rs` at test time and structurally verifies every rocBLAS GEMM FFI call
  routes `solution_index` through the correct dispatch helper. This is the right kind of
  "check the source" test — it verifies a structural invariant across every call site, not
  just that one symbol name exists.
- `tests/quantization.rs` (aside from A.3.3) — pure-logic capability-table tests with real
  assertions on real return values; correctly encodes the "no silent fp8 emulation on
  RDNA2/3" safety property.
- `tests/gemm_algo.rs`, `tests/perf_gate.rs` — pure-logic, no hardware dependency, real
  assertions.
- `tests/kv_dequant_attention_gpu.rs` (aside from the missing gate in A.2) — the test logic
  itself is well-designed: it drives quantization through the real
  `LloydMaxCompressor::compress`/`fused_attention` library API rather than hand-constructing
  device buffers, which sidesteps the `from_cpu`/`from_cpu_bytes` bug entirely.
- `tests/caching_allocator_reuse.rs` — correctly gated, asserts a real bound
  (`grown <= iters / 8`) on real allocator statistics rather than an exact/fragile equality.
- `tests/speculative.rs` — pure-logic tree-attention-mask and accept/reject tests with
  concrete bit-level assertions, no hardware dependency.
- `tests/p2p_route.rs` — covers only routing-decision logic and host-staging bookkeeping, and
  says so explicitly in its header comment ("Heavy GPU work ... is the next PR's surface").
  This is an honest scope disclosure, not a fake test — it matches the known, previously
  documented gap that real cross-GPU data movement isn't wired yet.

---

## A.5 Not verifiable in this pass

- **`quant_backward_audit_rocm_q8_0_gemm_dx_numerics`** (`grim-quant/tests/quant_backward_audit.rs`)
  — permanently disabled via `#[cfg(any())]` due to a dependency-cycle refactor
  (`grim-quant` dropped its `rocm` feature to break `grim-format → grim-quant →
  grim-backend-rocm → grim-format`). No equivalent test was found wired into
  `grim-backend-rocm` itself. **`quantized_matmul_backward_dx` currently has no real
  GPU-path test coverage anywhere in the crates reviewed.** This isn't a "wrong test," it's
  an absent one — worth a tracked follow-up to either move the test to a
  workspace-level integration crate (as the `#[cfg(any())]` comment itself suggests) or add
  an equivalent directly in `grim-backend-rocm/tests/`.
- No Rust toolchain was available to actually build/run the workspace, so all findings above
  are from static tracing of the call chain (trait definition → impl → allocator), not from
  an executed failing test. High confidence on A.1 and A.2 since the full mechanism was
  traced end-to-end through real source; recommend a real `cargo test -p grim-backend-rocm
  --features rocm -- --ignored` run (with `GRIM_RUN_GPU_TESTS=1` on real hardware) to confirm
  before merging fixes.
- `lib_internal_tests.rs` (1934 lines, 98 test functions) and `qkv_attention.rs` (578 lines)
  were scanned for the specific antipatterns above (no hits) but not verified function-by-
  function against implementation semantics — flag if you want a deeper pass on either.

---

# Part B — GGUF load → CLI chat → server chat pipeline

**Scope:** `grim-cli`, `grim-engine`, `grim-server`, triggered by the question "why can't I
get a GGUF file to load and converse properly." **Trigger for this whole investigation:**
user reported GGUF loading/conversation not working via `grim run model/model.gguf`.

## B.1 `grim-cli/src/run.rs` shadows the real GGUF loader with a private, less-capable copy (confirmed)

**File:** `grim-cli/src/run.rs`

**Defect:** `run.rs` defines its own `pub fn load_model_from_gguf(...)` at line 471, with the
same name and signature as `grim_engine::model_loader::load_model_from_gguf`. The `use`
import at the top of the file only pulls in `load_model_from_grim` and
`load_model_from_safetensors` from `grim_engine::model_loader` — **not**
`load_model_from_gguf`:

```rust
use grim_engine::{Engine, EngineConfig, model_loader::{load_model_from_grim, load_model_from_safetensors}};
```

Because of this, every call to `load_model_from_gguf(...)` inside `run.rs` (both the `serve`
branch and the one-shot branch) silently resolves to the local, older copy in the same file —
Rust's ordinary name resolution, no compile error, no warning.

**Why this breaks GGUF chat, concretely:**

1. **Narrower architecture coverage.** The engine's real loader
   (`grim_engine::model_loader::load_model_with_providers`) dispatches through a shared
   `HyperparameterExtractor` and covers a wide architecture matrix confirmed to include
   Falcon, Bloom, Phi2/Phi3/PhiMoe, and more. The shadow copy in `run.rs` only special-cases
   `mamba` and `lfm2` explicitly, then falls through to a handful of hardcoded branches
   (Gpt2, Gemma, DeepSeek, Bert, T5, Rwkv, Llama-as-default). If a GGUF's
   `general.architecture` metadata names a family only handled by the real loader, the
   shadow version mis-dispatches it into the Llama fallback branch, producing garbage output
   rather than a clean error — this matches "loads but doesn't converse properly" rather than
   an outright load failure.
2. Any bug fix or format nuance handled in `grim-engine/src/model_loader.rs` never reaches
   actual `grim run` invocations, since the shadow copy duplicates the hyperparameter
   extraction and tensor-name remapping logic by hand instead of sharing the engine's code.

**Fix:**

```rust
// in grim-cli/src/run.rs, change:
use grim_engine::{Engine, EngineConfig, model_loader::{load_model_from_grim, load_model_from_safetensors}};
// to:
use grim_engine::{Engine, EngineConfig, model_loader::{load_model_from_gguf, load_model_from_grim, load_model_from_safetensors}};
```

Then delete the local `pub fn load_model_from_gguf` (line 471 onward in `run.rs`) entirely,
since it is now fully superseded by the imported version.

---

## B.2 Shadow GGUF loader hardcodes CPU for Mamba and Bert (confirmed)

**File:** `grim-cli/src/run.rs`, inside the shadow `load_model_from_gguf` from B.1

**Defect:** Within the same shadow loader, most architecture branches correctly forward the
resolved `device`:

```rust
let m = Gpt2::load(device.clone(), &ws, cfg)?;
let m = Gemma::load(device.clone(), &ws, cfg)?;
let m = DeepSeek::load(device.clone(), &ws, cfg)?;
let m = Rwkv::load(&ws, cfg, device.clone())?;
let m = Llama::load(device, &ws, cfg)?;
```

But two branches hardcode `Device::Cpu` regardless of what was actually resolved (which may
be `Device::Rocm(n)`):

```rust
let m = grim_models_mamba::Mamba::load(Device::Cpu, &ws, cfg)?;   // line 542
let m = Bert::load(Device::Cpu, &ws, cfg)?;                        // line 647
```

**Consequence:** Loading a Mamba-family or Bert-family GGUF on a machine with a ROCm GPU will
silently run on CPU no matter what device was probed/selected — much slower than expected,
and a possible source of behavioral divergence if any GPU/CPU numerical differences exist
elsewhere in the pipeline.

**Fix:** Same one-line change in both spots — replace the hardcoded `Device::Cpu` with the
already-in-scope `device.clone()` (or `device` for the final use if ownership allows),
matching the pattern used by every other branch in the same function. Superseded once B.1's
fix removes this function entirely, but flagged here in case the equivalent branches in
`grim_engine::model_loader::load_model_with_providers` weren't already checked for the same
mistake — worth a quick grep there too before assuming the engine's version is clean.

---

## B.3 `grim-engine::Engine` hardcodes CPU for request session construction (confirmed)

**File:** `grim-engine/src/lib.rs`

**Defect:** Both `enqueue_request` (line ~450) and `enqueue_request_with_kv` (line ~477)
construct the per-request `SessionInner` with a hardcoded device, ignoring wherever the model
itself was actually loaded:

```rust
let session = Box::new(grim_core::session::Inner::new(grim_tensor::Device::Cpu));
// and
let session = Box::new(grim_core::session::Inner::with_kv(
    grim_tensor::Device::Cpu,
    Box::new(kv),
));
```

**Consequence:** This is the same hardcoded-CPU pattern as B.2, but inside the engine's own
scheduler-backed request path (used by `grim-server`, not just the CLI's one-shot path). If a
model is registered on a ROCm device via `Engine::register_model`, per-request sessions
(including KV cache state) are still built for CPU — worth checking whether this causes an
outright device mismatch error downstream, or a silent, much slower fallback. Given the
severity, this should be checked before assuming `--serve` correctly uses the GPU at all for
any model.

**Fix:** Both functions need the target device threaded in — either from `request.model_id`
looked up against the registered `LoadedModel`'s device, or as an explicit parameter. Exact
fix depends on how `LoadedModel` tracks device, which wasn't traced in this pass — flag for a
follow-up read of `LoadedModel`'s definition (`grim-engine/src/lib.rs:33`) before patching.

---

## B.4 Interactive REPL reloads the model and loses all state every turn (confirmed)

**File:** `grim-cli/src/main.rs`, the `Commands::Run` handler's interactive branch (no
`prompt` supplied)

**Defect:**

```rust
loop {
    print!(">>> ");
    // ... read a line ...
    if let Err(e) = run::cmd_run(resolved.clone(), Some(trimmed.to_string()), false, address.clone(), &plugins, temperature, top_p, top_k, max_tokens, seed, repeat_penalty).await {
        eprintln!("[grim run] Command failed: {e}");
    }
    println!();
}
```

Every loop iteration calls `run::cmd_run(...)` fresh, and `cmd_run` itself:
- Calls `load_model_from_gguf`/`load_model_from_grim`/`load_model_from_safetensors` at the
  top of its body — **the model is reloaded from disk on every message typed.**
- Constructs `let mut session = SessionInner::new(model.device().clone());` fresh each call —
  KV cache / session state is discarded at the end of every `cmd_run` invocation.

**Consequence:** The `>>>` prompt loop looks like a chat UI but has no real memory between
turns — each message is answered in total isolation, with no awareness of prior turns, and
with the full model-load cost paid again every time. For any non-trivial model this will feel
unusably slow and will not behave like a conversation at all past the first message.

**Fix (sketch — not a one-liner):** Move model loading and `Engine`/session construction
*outside* the `loop {}`, before the first prompt. Accumulate prior turns into the token
history fed to each `forward` call (or reuse a persistent KV cache across turns via the
session object), rather than calling `cmd_run` as a fresh one-shot each time. This likely
means factoring `cmd_run`'s generation loop out from its load/setup logic so the REPL can
call just the generation part repeatedly against one long-lived session.

---

## B.5 No chat template applied anywhere in the CLI prompt path (confirmed)

**File:** `grim-cli/src/run.rs`, prompt tokenization inside `cmd_run`

**Defect:** The tokenization logic inserts a best-effort BOS token by checking a short list of
candidates (`<|startoftext|>`, `<s>`, `<|im_start|>`) and then encodes the raw prompt text —
no chat-template wrapping (e.g. `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`) is
applied anywhere in this path.

**Consequence:** For instruction-tuned/chat-tuned GGUF models that expect a specific chat
format, sending raw prompt text without that structure will produce noticeably worse or
stranger completions than the same model run through Ollama (which does apply the embedded
`tokenizer.chat_template` from GGUF metadata, when present).

**Confirmed by follow-up search: no template renderer exists anywhere in the codebase.**
Searched every available crate (`grim-format`, `grim-nn`, `grim-cli`, `grim-server`,
`grim-engine`, `grim-tensor`, `grim-quant`, `grim-backend-rocm`, `grim-garage`,
`grim-autograd`) for `chat_template`, `jinja`, `ChatTemplate`, `apply_chat`,
`render_prompt`/`render_messages`, `PromptTemplate`, and template-engine dependencies
(`minijinja`, `tera`, `handlebars`, `liquid`) in every `Cargo.toml`. Result: nothing, aside
from two false positives (a CSS `grid-template-columns` rule in the server dashboard, and a
code comment). Specifically:

1. `tokenizer.chat_template` isn't extracted from GGUF metadata at all.
   `GgufTokenizer::from_metadata` (`grim-format/src/tokenizer.rs:116`) only reads
   `tokenizer.ggml.model`, `tokenizer.ggml.tokens`, `tokenizer.ggml.scores`, and
   `tokenizer.ggml.eos_token_id`. The Jinja template string most instruction-tuned GGUFs embed
   under `tokenizer.chat_template` is never read into the struct.
2. It's mechanically *reachable*, just unused: `GgufProvider::metadata(key: &str) ->
   Option<&GgufValue>` (`grim-format/src/tprov.rs:73`) is a generic key accessor, so
   `provider.metadata("tokenizer.chat_template")` would pull the raw string out — nothing
   currently calls it with that key.
3. No Jinja/template-rendering crate is a dependency anywhere in the workspace. Even with the
   raw string extracted, there's no library present to evaluate it.
4. The one place a ChatML-shaped prompt appears at all
   (`grim-engine/tests/lfm2_350m_safetensors_inference.rs:53`) is a hand-written raw string
   literal in a single test (`"user\nwhat is the capital of france? \nassistant\n"`) — not
   shared infrastructure, and not even correctly formatted (missing the `<|im_start|>`/
   `<|im_end|>` special tokens ChatML/LFM2 actually expect).

**This means B.5 (and B.6, which needs the same renderer) require new code, not a rewire.**
Scoped implementation plan:

1. **Add a template-engine dependency.** `minijinja` is the right choice: no Python
   dependency, small, and it's the de facto standard other Rust local-inference projects use
   for this exact GGUF field. Add to the relevant `Cargo.toml`(s) — likely `grim-format`
   (where the type lives) or a new small crate (e.g. `grim-chat-template`) if `grim-format`
   should stay free of a template-engine dependency for compile-time/dependency-hygiene
   reasons. Given `grim-format` already owns `GgufTokenizer`, the simplest first cut is adding
   it there; split out later if it proves too heavy a dependency for that crate's role.

2. **Extract the raw template string.** In `grim-format/src/tokenizer.rs`, extend
   `GgufTokenizer` (or wrap it) to also capture `tokenizer.chat_template` via the existing
   `GgufProvider::metadata()` accessor at load time:
   ```rust
   pub struct GgufTokenizer {
       // ...existing fields...
       pub chat_template: Option<String>,
   }

   // in from_metadata / wherever the struct is constructed from a GgufProvider:
   let chat_template = provider
       .metadata("tokenizer.chat_template")
       .and_then(|v| v.as_str())
       .map(|s| s.to_string());
   ```
   Note this requires `from_metadata`'s signature (or an adjacent constructor) to have access
   to the `GgufProvider`, not just the raw `HashMap<String, GgufValue>` — check the current
   call sites in `run.rs`/`model_loader.rs` to confirm the provider is in scope wherever
   `GgufTokenizer` gets built, or add a provider-aware constructor alongside the existing one.

3. **Write the renderer.** A single function, likely in `grim-format` next to
   `GgufTokenizer`, or in a new small module if kept separate from the dependency addition in
   step 1:
   ```rust
   /// Renders an OpenAI-style `messages` array through a model's Jinja chat template,
   /// producing the final prompt string ready for tokenization.
   pub fn render_chat_template(
       template: &str,
       messages: &[ChatMessage],       // { role: String, content: String }
       add_generation_prompt: bool,    // true for "generate the next assistant turn"
   ) -> Result<String> {
       let env = minijinja::Environment::new();
       let tmpl = env.template_from_str(template)?;
       let ctx = minijinja::context! {
           messages => messages,
           add_generation_prompt => add_generation_prompt,
           // Most HF/GGUF templates also reference bos_token/eos_token; thread these
           // through from GgufTokenizer once available, defaulting to "" if absent.
       };
       Ok(tmpl.render(ctx)?)
   }
   ```
   `ChatMessage` needs `serde::Serialize` (minijinja renders from any serializable value) with
   `role`/`content` fields matching what HF-style templates expect. Real HF chat templates
   commonly reference `messages`, `add_generation_prompt`, `bos_token`, `eos_token`, and
   sometimes `system_message` — start with the common subset and expand if a specific model's
   template errors on a missing variable (minijinja will report the undefined-variable name).

4. **Add a raw-prompt fallback.** Base (non-instruction-tuned) GGUFs commonly have no
   `tokenizer.chat_template` field at all. When `chat_template` is `None`, fall back to
   today's behavior (raw prompt text + best-effort BOS insertion) rather than erroring —
   this preserves current behavior for models that were already working correctly.

5. **Wire it into `grim-cli/src/run.rs`'s `cmd_run`.** Replace the raw
   `tok.encode(&prompt)` call with: build a single-turn `messages` array
   (`[{role: "user", content: prompt}]`), call `render_chat_template` if
   `tokenizer.chat_template` is `Some`, then encode the rendered string. This also sets up the
   multi-turn accumulation B.4's REPL fix will need — once turns are tracked, extend the
   `messages` array across turns instead of rebuilding it as one user message each time.

---

## B.6 `grim-server`'s chat endpoints ignore the client's message content entirely (confirmed — most severe finding)

**File:** `grim-server/src/lib.rs`, `chat_completions` handler

**Defect:** Both generation paths inside `chat_completions` hardcode the prompt to tokenize
the literal string `"Hello"`, regardless of the request body's actual `messages`:

Streaming path (~line 331):
```rust
let stream = futures::stream::unfold(
    (0u64, String::new(), tokenizer_clone.as_ref().map(|t| t.encode("Hello")).unwrap_or_default()),
    ...
```

Non-streaming path (~line 391):
```rust
let prompt_tokens = tokenizer.as_ref().map(|t| t.encode("Hello")).unwrap_or_default();
```

The request's `messages` field is read into `body_obj` earlier in the handler purely for the
`KNOWN_FIELDS` allow-list validation pass (confirming it's present and well-formed) — its
actual content is **never extracted or used to build the prompt** in either branch.

`grim_chat` (the Ollama-compatible `/api/chat` route) correctly forwards the client's
`messages` array into the payload it builds, but that payload is then passed straight into
`chat_completions`, which discards it the same way. So the bug is universal across every
chat-shaped entry point in the server: `/v1/chat/completions` (OpenAI-compat, streaming and
non-streaming) and `/api/chat` (Ollama-compat) all generate a completion as if the user had
sent the single word "Hello," no matter what was actually asked.

**Consequence:** This is the most severe finding in this document. Even with B.1–B.5 fixed,
`grim run --serve` cannot produce a meaningful chat response to any request — not a
multi-turn/history problem, but a complete loss of the current turn's content. This is very
likely a debugging placeholder left in from when the SSE streaming mechanics were being built
and validated in isolation (a fixed test string is a reasonable thing to hardcode while
getting the stream plumbing working), and never wired up to the real request body afterward.

**Scoped implementation plan** (depends on B.5's renderer existing first):

1. **Parse `messages` into `ChatMessage` structs.** Right after the existing `KNOWN_FIELDS`
   validation block (~line 190-206), extract and validate the array shape instead of only
   confirming the key's presence:
   ```rust
   #[derive(serde::Deserialize, serde::Serialize, Clone)]
   struct ChatMessage {
       role: String,
       content: String,
   }

   let messages: Vec<ChatMessage> = match body_obj.get("messages").and_then(|v| v.as_array()) {
       Some(arr) => match serde_json::from_value(serde_json::Value::Array(arr.clone())) {
           Ok(m) => m,
           Err(e) => {
               return (
                   StatusCode::BAD_REQUEST,
                   Json(serde_json::json!({"error": format!("invalid 'messages' array: {e}")})),
               ).into_response();
           }
       },
       None => {
           return (
               StatusCode::BAD_REQUEST,
               Json(serde_json::json!({"error": "'messages' field is required and must be a non-empty array"})),
           ).into_response();
       }
   };
   if messages.is_empty() {
       return (
           StatusCode::BAD_REQUEST,
           Json(serde_json::json!({"error": "'messages' must contain at least one message"})),
       ).into_response();
   }
   ```
   This also closes a latent gap: today an empty or missing `messages` array silently passes
   the `KNOWN_FIELDS` check and falls through to the `"Hello"` hardcode, so there's currently
   no request-validation signal distinguishing "well-formed chat request" from "malformed/
   empty" — both produce the same (wrong) behavior. Real validation here is a prerequisite
   for the fix, not just cleanup.

2. **Render the prompt once, before both the streaming and non-streaming branches split.**
   Currently the `"Hello"` hardcode is duplicated in both branches; the fix should compute
   `prompt_tokens` a single time and pass it into whichever branch runs, removing the
   duplication rather than patching both copies separately:
   ```rust
   let prompt_text = match &state.tokenizer.lock().unwrap().as_ref().and_then(|t| t.chat_template.clone()) {
       Some(template) => grim_format::render_chat_template(template, &messages, true)
           .unwrap_or_else(|e| {
               eprintln!("[grim-server] chat template render failed, falling back to last-message content: {e}");
               messages.last().map(|m| m.content.clone()).unwrap_or_default()
           }),
       None => {
           // No embedded template (base model, or tokenizer not yet loaded) — fall back to
           // the last user message's raw content rather than the whole array, to avoid
           // dumping unformatted multi-turn history at the model with no role markers.
           messages.last().map(|m| m.content.clone()).unwrap_or_default()
       }
   };
   let prompt_tokens = tokenizer.as_ref().map(|t| t.encode(&prompt_text)).unwrap_or_default();
   ```
   Then delete both existing hardcoded-`"Hello"` lines (streaming ~line 331, non-streaming
   ~line 391) and reference this single `prompt_tokens` binding from both branches.

3. **Multi-turn state.** This fix makes a *single* request correctly reflect its own
   `messages` array (fixing B.6's core defect), but each HTTP request to
   `/v1/chat/completions` is still stateless from the engine's point of view — the standard
   OpenAI-compatible contract already expects the *client* to resend the full `messages`
   history on every call (that's how Ollama/OpenAI-compatible chat UIs work — the history
   lives client-side, not server-side per session). So no additional server-side session
   persistence is needed for `/v1/chat/completions` itself once steps 1-2 land; B.4's
   CLI-REPL-side history accumulation is the piece that needs to track and resend growing
   `messages` arrays turn over turn, using this now-correctly-behaving endpoint underneath.

4. **Apply the same fix to `grim_generate`** (~line 1101, the Ollama `/api/generate`
   single-prompt-not-messages endpoint) — worth checking whether it has its own separate
   hardcoded-prompt issue or correctly forwards its `prompt` field, since it wasn't traced in
   this pass. It builds a `messages: [{"role": "user", "content": prompt}]` payload and also
   delegates to `chat_completions`, so it should be automatically fixed by steps 1-2 above,
   but confirm after patching rather than assuming.

---

## B.7 Open follow-ups

- **`LoadedModel` device tracking** (`grim-engine/src/lib.rs:33`) — needs to be read in full
  to design the correct fix for B.3; not yet reviewed in this pass.
- **`grim_engine::model_loader::load_model_with_providers`** — confirmed as the correct,
  capable target for B.1's fix, but not yet checked for the same hardcoded-`Device::Cpu`
  mistake found in B.2's shadow copy (worth a quick grep for `Device::Cpu` literals in that
  function before assuming it's clean).
- **Where `GgufTokenizer` gets constructed** — B.5 step 2 needs the `GgufProvider` in scope
  wherever `GgufTokenizer`/`from_metadata` is currently called (`run.rs`'s loader,
  `model_loader.rs`'s loader, possibly others); call sites weren't fully enumerated in this
  pass and should be checked before implementing so the provider-aware constructor change
  doesn't miss a caller.
- **`grim_generate`'s own prompt handling** (B.6 step 4) — assumed fixed transitively via
  `chat_completions`, but not independently confirmed.
- **minijinja variable coverage** — real HF/GGUF chat templates vary in which Jinja variables
  they reference beyond `messages`/`add_generation_prompt` (e.g. `bos_token`, `eos_token`,
  `system_message`, tool-calling-related variables for newer templates). The B.5 sketch covers
  the common subset; expect to need to widen the render context as specific models' templates
  are tested against it — minijinja will surface undefined-variable errors naming exactly
  what's missing, so this should be fast to iterate once real GGUF files are tested end to end.
- No Rust toolchain was available to compile/run any of `grim-cli`, `grim-engine`,
  `grim-server`, or `grim-format` in this environment — all Part B findings, including the
  B.5/B.6 implementation plans above, are static-trace confirmed and design-stage only, not
  execution-confirmed. Recommend implementing B.5's renderer first (it's the shared
  dependency for both), then B.1 and B.6, then an actual end-to-end `grim run <model>.gguf`
  and `grim run --serve` + real chat client test to confirm conversation behavior before
  considering this document's Part B closed.
