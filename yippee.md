# Grim logic bug check and wiring validation

Scope: the 28 grim crates under `crates/grim-*/` (~70,000 lines of Rust, 169 source files). Vendored trees in `old/`, `.zl/`, and `.rocm/` are out of scope.

Skills applied: caveman (compression), ponytail-review and ponytail-audit (over-engineering), rust and rust-ffi-grim (FFI discipline), tdd and strong-tests (test strength), writing-guidelines and humanizer (prose).

## What I ran

I checked the whole thing compiles and the lib tests pass, then read the code across every crate.

```
cargo check --workspace              exit 0, clean
cargo test  --workspace --lib        522 passed, 0 failed, 27 modules
```

All 27 workspace path dependencies in `Cargo.toml` resolve to a real crate on disk. No orphaned path deps, no version skew in the graph. The wiring is sound at the build level.

The wiring is sound at the build level. The bugs below are logic defects inside crates that compile and pass their own tests.

## The headline

The code compiles and the 522-test suite is green, and a number of those tests pass for the wrong reason. Two tests actively lock in known-wrong behavior, so the green suite is hiding real defects. The most serious is a weight-loading bug that turns every Q4_K/Q5_K/Q6_K GGUF weight into garbage before inference even starts.

I rank the findings P0 through P2. P0 means wrong output on real models. P1 means wrong output on a reachable path or a resource leak. P2 means broken feature build, dead code, or weak test coverage.

## P0 findings (wrong output on real models)

These produce numerically wrong results on inputs you will actually hit.

### P0-1. Q4_K/Q5_K/Q6_K GGUF weights dequantize to garbage

`crates/grim-format/src/gguf.rs:1177-1196`

The K-quant super-block dtypes are mapped to the wrong `Storage` variant. `Q4K` maps to `Storage::Block(BlockDtype::Fp4)`, `Q5K` to `Block(Nf4)`, `Q6K` to `Block(Fp8)`. The dispatch in `crates/grim-nn/src/varbuilder.rs:347-353` routes `Block(Fp4/Nf4/Fp8)` to `dequant_fp4`/`dequant_nf4`/`dequant_fp8`, which read a flat 4-bit LUT or an 8-bit float. None of those match the 256-weight super-block layout with 6-bit per-sub-block scales that Q4_K/Q5_K/Q6_K actually use.

The neighbors are correct. `Q2K` and `Q3K` map to `Storage::KQuant(Q2K/Q3K)` (lines 1169-1176). `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1` map to `KQuant(Q4K/Q5K)` (lines 1181-1192). Only the three `Q*K` variants are wrong. Every Q4_K model on Hugging Face hits this.

Fix: map all three to `Storage::KQuant(KQuantScheme::Q4K/Q5K/Q6K)` and let the existing `dequant_q4k`/`dequant_q5k`/`dequant_q6k` run.

Worse, the test at `crates/grim-format/src/tprov.rs:575-590` asserts the wrong mapping, so the fix breaks a currently-passing test. See P2-3.

### P0-2. Q2_K and Q3_K fall through to the Q4_K dequantizer

`crates/grim-nn/src/varbuilder.rs:340`

```rust
KQuantScheme::Q2K | KQuantScheme::Q3K => dequant_q4k(&raw.bytes, n),
```

Q2_K blocks are 84 bytes, Q3_K are 110 bytes, Q4_K are 144 bytes, each per 256 weights. `dequant_q4k` assumes the 144-byte layout, so feeding it Q2_K or Q3_K bytes reads the wrong fields and can overrun. The enum arms exist in the mapping, the dispatcher just never implemented them.

Fix: implement `dequant_q2k` and `dequant_q3k`, or return `Error::Unimplemented` so the load fails loudly instead of producing silent garbage.

### P0-3. FP8 E4M3 subnormal encode and decode are off by a factor of 64

`crates/grim-quant/src/lib.rs:612` (decode) and `:892` (encode)

The decode divides the mantissa by 8.0 where E4M3 subnormals need 512.0 (exponent unbiased at -6, spacing 2^-9 = 1/512). The largest subnormal decodes to 0.875 instead of about 0.0137. The encoder has the inverse bug, multiplying by 8.0 instead of 512.0. They round-trip cleanly against each other, which is why no test caught it, but the quantization throws away almost all subnormal precision and any correct decoder reading the bytes sees garbage. The subnormal-to-normal transition is non-monotonic.

Fix: `result = (mant as f32) / 512.0;` on decode, `let m = (abs * 512.0).round() as u8;` on encode.

### P0-4. Batched GEMM on ROCm breaks for F16 and BF16

`crates/grim-backend-rocm/src/device/roc_device.rs:748` and `:756`

```rust
(a_packed.device_ptr.unwrap() as *mut c_void).add(i * stride_a * 4),
```

`stride_a` is an element count, but `.add()` on `*mut c_void` counts bytes. The literal `* 4` hardcodes 4 bytes per element, which is only right for F32. For F16 and BF16 the offset is 2x too large, so batch entry `i > 0` is copied to the wrong device address and the layout handed to rocBLAS no longer matches the declared strides. The copy size uses `ai.bytes` correctly, only the destination offset is wrong.

Fix: multiply by the element byte size from the dtype, not the literal 4.

### P0-5. CPU fused LoRA GEMM passes the wrong matrix layout

`crates/grim-backend-cpu/src/simd_gemm.rs:88` and `:94`

`gemm_f32_simd` computes `C = A * B^T` and expects its second argument in `[N,K]` layout (the scalar and AVX2 paths both read `b[j*k + kk]`). But the LoRA-fused path feeds it `w` in `[K,N]` layout (the main weight) and `a` in `[K, rank]` layout (declared in the comment on line 91), both transposed from what the kernel expects. The fused result `Y = X*W + scale*(X*A)*B` is wrong for any non-symmetric input.

The test passes because it uses an identity matrix for `w`. See P2-2.

Fix: route through the row-major GEMM the CPU backend already uses for `matmul`, or transpose `w` and `a` before the call.

### P0-6. Speculative acceptance uses an absolute floor instead of the probability ratio

`crates/grim-speculative/src/speculative_wrapper.rs:212-224`

```rust
let p_target = target_probs.get(draft_tok as usize).copied().unwrap_or(0.0);
if p_target >= 0.1 { accepted_count += 1; } else { break; }
```

Speculative decoding accepts a draft token with probability `min(1, p_target / p_draft)` (the ratio test). This code uses an absolute floor of 0.1 with no reference to the draft probability and no randomness. A draft token with `p_target = 0.5` is always accepted regardless of `p_draft`, which biases the output distribution toward high-target-probability tokens. The loop also breaks at the first sub-0.1 token, which kills the throughput benefit.

Compounding it, `target_probs` is the flattened `[verify_len, vocab]` logits tensor, and `.get(draft_tok)` indexes by token id into row 0 for every position, so the lookup is from the wrong row anyway.

Fix: index `target_probs[i * vocab + draft_tok]`, softmax it, and apply the standard ratio test with the per-request RNG.

### P0-7. `Session::append_kv` discards the keys and values it is handed

`crates/grim-core/src/session.rs:86-91`

```rust
fn append_kv(&mut self, _k: &Tensor, _v: &Tensor) -> Result<()> {
    if let Some(kv) = self.kv.as_deref_mut() { kv.append_slot()?; }
    Ok(())
}
```

The caller passes key and value tensors. The implementation allocates an empty block and drops both tensors on the floor. The slot is never populated, so attention reads zero-initialized keys and values as if they were real.

Fix: after `append_slot`, call `write_keys` and `write_values` with the supplied tensors.

### P0-8. Scheduler corrupts the running token count during chunked prefill

`crates/grim-scheduler/src/lib.rs:228-229`

After chunked prefill, the request pushed to `running` has its `prompt_tokens` overwritten with the chunk size. The engine's `total_running_tokens` (line 180) sums `r.prompt_tokens` over running, so a request that originally had 100k tokens but was chunked to 512 now reports as tiny. The preemption check at line 198 never fires for genuinely oversized requests. The remainder is re-enqueued with the same id, so the engine later prefills it as a new prompt at position 0, resetting RoPE positions and corrupting the KV layout.

Fix: keep the original `prompt_tokens` on the running entry and track consumed versus remaining separately.

## P1 findings (wrong output on a reachable path, or resource leaks)

### P1-1. Prefix cache can hand out a freed block

`crates/grim-memory/src/lib.rs:146-156` and `:194-196`

`prefix_cache` is never invalidated when a block's refcount drops to 0. `find_or_share_prefix` can return a `bid` whose refcount was removed, then `or_insert(0) += 1` resurrects it. The block may already be demoted to NVMe, zeroed, or handed to another sequence via `alloc`. Classic use-after-free on a block id.

Fix: remove the `prefix_cache` entry when refcount hits 0, or check refcount is greater than 0 before sharing.

### P1-2. KV blocks leak on every request and on every truncate

`crates/grim-engine/src/lib.rs:458-463` (finish) and `crates/grim-memory/src/lib.rs:319-343` (truncate)

`finish_request` drops the session, which drops the `Arc` to the block pool, but nobody calls `free` or `rollback_to(0)` on the blocks the sequence allocated. They stay allocated with refcount at least 1 forever. `BlockTable::truncate` drops logical entries without returning the physical ids to the free list. The pool exhausts over time.

Fix: call `kv.rollback_to(0)` before removing the session, and make `truncate` pool-aware.

### P1-3. Speculative rollback and commit confuse tokens with blocks

`crates/grim-memory/src/lib.rs:372, 385-417`

`token_count()` returns `len() * BLOCK_SIZE`, which assumes every block is full. A partial last block inflates the count. `commit(accepted_len)` computes `to_drop` from a token count but `rollback_to` consumes it as a token count then divides by `BLOCK_SIZE`, while `tentative_len` is tracked in blocks. The units are inconsistent across the whole `commit`/`rollback_to`/`token_count` family, so rollback pops the wrong number of blocks whenever the sequence has a partial block.

Fix: track a real token count (sum of `block.num_tokens`), not `len() * BLOCK_SIZE`. Decrement `tentative_len` by `blocks_to_pop`, not by `to_remove`.

### P1-4. Self-tuning controller runs on hard-coded fake telemetry

`crates/grim-engine/src/lib.rs:265-304`

Every `tick()` builds a fresh `SelfTuningController`, feeds it `record_ttft(1500.0)` and `record_itl(95.0)` (constants, not measured from anything), tunes, and overwrites the scheduler knobs. The EMA never accumulates real history, and the constant 1500ms TTFT is always above the target, so `tune_all()` permanently drives `max_batched_tokens` and `chunked_prefill_size` toward their floors (512 and 64), throttling throughput on a system whose real latency may be fine.

Fix: hold the controller as an `Engine` field, feed it real measured latencies, and delete the hard-coded inputs.

### P1-5. Engine picks an arbitrary model per request

`crates/grim-engine/src/lib.rs:522-527`

```rust
fn model_for_request(&self, _id: u64) -> Option<(&str, i32)> {
    self.models.iter().next().map(|(k, _)| (k.as_str(), 0))
}
```

`HashMap::iter().next()` returns an arbitrary entry and HashMap order is randomized per process. With more than one model registered, a request can be decoded by different models on consecutive ticks, producing garbage. There is no request-to-model table.

Fix: give each request a target model id and look it up.

### P1-6. `drive_forward` ignores the request's adapters

`crates/grim-engine/src/lib.rs:368-372`

```rust
let adapter_ids: Vec<u32> = Vec::new();
let adapters = { let resolved = self.resolve_adapters(&adapter_ids).unwrap_or_default(); resolved };
```

A fresh empty vec is built inside the function. The adapter ids the caller passed to `step_one` are never consulted, so every forward runs with no adapters even when the request was enqueued with some.

Fix: thread the caller's `adapter_ids` through.

### P1-7. Admission deferral pushes to the front and stalls the queue forever

`crates/grim-scheduler/src/lib.rs:185-192`

A deferred head request is pushed back to the front of `waiting` and the loop breaks. Next `schedule()` recomputes the same backlog and defers the same request again. Nothing behind it is ever considered. One oversized request permanently stalls the whole queue.

Fix: push deferred requests to the back, or to a separate deferred queue.

### P1-8. Engine advances position twice under DSpark speculation

`crates/grim-speculative/src/speculative_wrapper.rs:226-229`

The target `forward` call already advanced the session position internally. Then this block calls `session.advance_pos(accepted_count)` again. Under DSpark, `current_pos` advances by the forward's internal move plus `accepted_count`, corrupting positional encoding on the next tick.

Fix: own position advancement in exactly one place.

### P1-9. Server emits invalid SSE JSON and never sends `[DONE]`

`crates/grim-server/src/lib.rs:344-350` and `:356`

The streaming delta is built with `format!` and the only escaping on `token_text` is `.replace("\"", "\\\"")`. Backslashes, newlines, tabs, and control codes pass through raw, so any token decoding to `\n` produces invalid JSON in the `data:` frame. The stream also never emits a terminal `data: [DONE]\n\n`, so OpenAI-compatible clients hang or report a truncated response.

Fix: build the payload with `serde_json::json!(...).to_string()` and append a final `Event::default().data("[DONE]")` before the stream ends.

### P1-10. `top_k` and `repeat_penalty` can never reach the sampler

`crates/grim-server/src/lib.rs:162-174, 253-257`

The two fields are read from the body but are absent from the `KNOWN_FIELDS` whitelist. The whitelist gate returns HTTP 400 for any unknown key before the fields are read, so requests using either knob are rejected.

Fix: add both fields to `KNOWN_FIELDS`.

### P1-11. Training loop ignores prompt-masking labels

`crates/grim-cli/src/train.rs:287-290`

```rust
for (tokens, _labels) in dataset.iter() {
    let input_ids = &tokens[..tokens.len() - 1];
    let targets = &tokens[1..];
```

The dataset builder computes labels with `-100` on prompt positions, but the training loop binds them to `_labels` and computes loss against `targets = &tokens[1..]`. Prompt tokens leak into the loss as if they were targets. The masking is computed and thrown away.

Fix: iterate `(tokens, labels)` and use `labels[1..]` as the loss target.

### P1-12. Plugin capability grants can never be parsed from any manifest

`crates/grim-plugin/src/lib.rs:158-178, 200`

`capabilities` is parsed as a JSON array of strings. The grant lookup does `capabilities.get("grants")`, which calls `.get()` on an array and always returns None. `PluginGrants` (network, filesystem, request metadata) is never populated from any manifest, so every plugin runs with empty grants regardless of what it requests.

Fix: look up grants under a separate key, not nested under the array-valued `capabilities`.

### P1-13. WASM sampler plugins load but always error at sample time

`crates/grim-plugin/src/wasm_loader.rs:234-238`

`WasmSampler::sample` always returns `Err(Error::Unimplemented(...))`. Plugins report successful load but every sampling call fails, so a WASM sampler never participates in inference. The `#[cfg(not(feature = "wasm-sandbox"))]` path errors too.

Fix: implement the wasmtime call into the exported sampler function, or fail at load time rather than on every sample.

### P1-14. Repeat penalty compounds per occurrence instead of once per token

`crates/grim-core/src/sampler.rs:159-176`

The loop applies the penalty once per occurrence in history. A token appearing five times gets its logit divided by `repeat_penalty` five times. llama.cpp and Ollama apply it once per unique token. This over-suppresses frequent tokens and skews generation.

Fix: dedupe the history before applying.

### P1-15. Concurrency: fixed request id collides across in-flight requests

`crates/grim-server/src/lib.rs:324, 359`

Streaming and non-streaming completions use two hard-coded constants, `0xDEAD_0000` and `0xDEAD_0001`. Concurrent requests collide. The comment "we can always look up the outcome" is false because the engine keys sessions by this id.

Fix: generate a per-request id.

## P2 findings (broken feature build, dead code, weak tests)

### P2-1. `rocm-profile` feature does not compile

`crates/grim-backend-rocm/src/device/roc_device.rs:1564, 1568`

The binding is `_tile_config` but the `println!` under the feature gate references `tile_config`. The feature is broken and cannot be built. Without the feature it is a dead binding, which is why it went unnoticed.

Fix: drop the underscore, or fix the format arg.

### P2-2. GEMM tests use identity matrices and hide transpose bugs

`crates/grim-backend-cpu/src/simd_gemm.rs:112-128, 130-144`, plus `device.rs:641-658` and `strict_kernels.rs:282-299`.

Every CPU matmul test uses an identity or symmetric `B`. A transpose mutant is invisible because identity equals its transpose. P0-5 ships because of this. The LoRA test also uses a range band `y[0] > 1.9 && y[0] < 2.1` instead of an exact value.

Fix: use a non-symmetric non-square case (say M=2, N=3, K=4) with a hand-computed expected, and assert exact values.

### P2-3. A test locks in the P0-1 bug

`crates/grim-format/src/tprov.rs:575-590`

`test_dtype_from_gguf_block_mappings` asserts Q4K maps to `Storage::Block(BlockDtype::Fp4)`, Q5K to `Nf4`, Q6K to `Fp8`. The fix to P0-1 breaks this test. Either the mapping is intentional (then justify why a K-quant maps to a Block dtype and prove the downstream dequant is correct) or it is the bug and the test must flip.

### P2-4. The autograd matmul backward test only checks shape

`crates/grim-autograd/src/ops.rs:348-363`

`assert_eq!(ga.shape().dims(), &[2, 2])` is the only check. The backward pass could return all zeros and pass. This is the training gradient path.

Fix: assert the exact gradient values against a hand-computed expected, and add `transpose_a`/`transpose_b` cases.

### P2-5. GPTQ tests use all-zero input and only assert `is_ok()`

`crates/grim-quant/src/lib.rs:2301-2388`

The three GPTQ tests (`gptq_3bit_cross_word_packing`, `gptq_2bit_basic`, `gptq_4bit_basic`) use `qweight = [0u8; N]` and assert only that the result is `Ok` and the length is right. The cross-word bit math is never exercised. The test even admits it: `// Simplified: just use identity pattern for testing`.

Fix: pack known non-zero codes at the 3-bit word-boundary positions and assert their exact dequant values. The `test_gptq_dequant_correctness_fixture` at line 2826 does this correctly for 4-bit and is the model to copy.

### P2-6. CLI reports success on failure in several places

`crates/grim-cli/src/main.rs:545, 769`, `src/plugin.rs:45-54`, `src/verify.rs:45-48, 198-199`.

`--plugins` is parsed then discarded (`let _ = plugins`). The serve `Result` is dropped (`let _ = grim_server::serve(...)`), so a fatal bind error is swallowed. WASM load failure prints a warning and returns `Ok`. `grim verify` returns `Ok(())` on invalid magic bytes, so CI exit-code checks pass on a corrupt file. The tensor error and warning counters are declared mutable and never incremented, so the summary always reports zero.

Fix: propagate errors and exit non-zero on failure.

### P2-7. Service management commands target the wrong name

`crates/grim-cli/src/service.rs:431, 447-465, 506, 527-583`.

`start` uses the label `com.grim`, but `stop`, `print`, and `kickstart` use the bare `grim`, so they target a non-existent launchd job and silently no-op. Windows install creates `grim_{name}` but uninstall, start, stop, and status hardcode `grim`. The launchd plist also emits `<key>--config</key>` instead of `<string>--config</string>`, producing malformed XML.

Fix: use a single label constant everywhere.

### P2-8. Doctor health check pings the wrong port

`crates/grim-cli/src/doctor.rs:282`

The check pings `http://127.0.0.1:8080/metrics` but the server default everywhere else is `11434`. The "GPU-backend alive" check always fails against a default server.

Fix: use 11434 or accept a flag.

### P2-9. Windows service handler shuts the runtime down before serving

`crates/grim-cli/src/main.rs:1022-1041`

The serve future is spawned and `rt.shutdown_background()` is called immediately, dropping the runtime and the in-flight server task before it serves a request.

Fix: keep the runtime alive for the service lifetime.

### P2-10. A garage test attribute is commented out

`crates/grim-garage/src/ui_state/poller.rs:192-193`

```rust
    // ^ extra blank line removal marker (test only)    #[tokio::test]
    async fn poller_abort_is_idempotent() {
```

The `#[tokio::test]` sits on the same line as a `//` comment, so it is parsed as comment text and the test is never registered. It never runs.

Fix: move the attribute to its own line.

### P2-11. Path traversal in garage routes

`crates/grim-garage/src/routes.rs:295, 365, 472, 546-592`.

`get_bolt_ons`, `attach_bolt_on_route`, `detach_bolt_on_route`, and `convert_model_route` feed the raw URL path parameter straight into `Path::new(&model_id)` or build an output path from `output_name` with no sanitization. A request to `/api/models/../../etc/passwd/bolt-ons` escapes the models root.

Fix: reject any `model_id` containing `/`, `\`, or `..`, and confine `output_name` to a single path component.

### P2-12. Several error paths swallow failures

`crates/grim-server/src/lib.rs:1036, 1159` (body read `unwrap_or_default()` substitutes empty body on truncation), `crates/grim-server/src/lib.rs:584` (`validate_model_capabilities` always returns true and the caller drops the result), `crates/grim-server/src/lib.rs:1448` (`addr.parse().unwrap()` panics on bad config), `crates/grim-kvtransport/src/lib.rs:269-279` (`fetch_block_remote` returns hard-coded placeholder KV).

### P2-13. KV transport block-length mismatch on NVMe round-trip

`crates/grim-kvtransport/src/lib.rs:111, 156-167`

`demote_to_host` accepts k/v of arbitrary length. `retrieve` always reads exactly `block_elems` floats. A block demoted at a different length truncates, overruns into the adjacent tensor, or fails `read_exact`. No length is checked on demote and none is persisted.

Fix: assert `k.len() == block_elems` on demote, or persist the per-block count.

### P2-14. Dead code worth cutting (ponytail-audit)

Ranked biggest cut first. These are over-engineering and speculative surface, not correctness bugs.

```
delete: grim-garage training-dashboard crate (~1900 LoC, no import from cli or server). [crates/grim-garage/src/lib.rs:1]
delete: grim-speculative NativeMtp path (with_native_mtp, Strategy::NativeMtp, decode_native_mtp, native_mtp modules). Zero production callers. [crates/grim-speculative/src/speculative_wrapper.rs:89]
delete: grim-speculative mamba_speculative module. Exported but referenced only by its own test. [crates/grim-speculative/src/lib.rs:42]
delete: grim-autograd preference_loss (dpo_loss, orpo_odds_ratio_loss, grpo_normalize_rewards). Zero callers outside their own test block. [crates/grim-autograd/src/lib.rs:55]
delete: grim-plugin WASM + dylib loaders. create_sampler returns Err(Unimplemented) on both paths; wasm_loader.rs (304 LoC) + dylib_loader.rs (150 LoC) + wit/*.wit are speculative. [crates/grim-plugin/src/wasm_loader.rs:88]
delete: grim-backend-rocm gptq_kernel.rs (235 LoC). #[allow(dead_code)] with TODO "Wire compile_gptq_kernel() call site", no caller. [crates/grim-backend-rocm/src/gptq_kernel.rs:27]
delete: grim-engine tick() self-tuning block (16 lines) + self_tuning.rs (442 LoC). Runs on fake telemetry; covered as P1-4. [crates/grim-engine/src/lib.rs:266]
delete: grim-cli Commands::Quantize. Body is a println "not yet implemented (phase 2)". [crates/grim-cli/src/main.rs:675]
delete: grim-cli spec.rs SpecCommands::Train. Writes placeholder bytes, returns, never invokes a model. [crates/grim-cli/src/spec.rs:5]
yagni: grim-core ModelArchitecture enum, 166 variants + 1000-line dispatch. Only ~8 are ever constructed; the rest exist for tensor-name remapping. Collapse to the handful loaded, or move the naming table to data. [crates/grim-core/src/architecture.rs:12]
yagni: grim-server hand-rolled TOML parser (load_tls_config_from_file, get_default_model_from_config) plus utc_now_rfc3339 reimplementing civil-calendar math. The crate transitively already has the toml crate available; reuse it, or pull time/chrono. [crates/grim-server/src/lib.rs:742]
yagni: grim-server grpc_service_handler route returns a literal string. No tonic dep, no gRPC. [crates/grim-server/src/lib.rs:553]
shrink: grim-server /api/chat and /api/generate duplicate ~120 lines of SSE-to-NDJSON reformatting. Extract one helper keyed on message versus response. [crates/grim-server/src/lib.rs:937]
```

`net: about -4200 lines, -1 to -2 deps possible.`

A smoke test or assert-based self-check is the ponytail minimum, not bloat. The `grim-cli/src/bench.rs` 40-line benchmark and the scheduler's `plan_hybrid_attention_step` are lean and should stay.

## Test strength summary

I ran the strong-tests audit over 14 test modules on the critical path.

The strong modules set the bar. `grim-tensor/src/softmax_merge.rs` covers identity, commutativity, associativity, zero-sum, and empty-slice with exact tolerances and labelled messages (kill-rate readiness: HIGH). The gguf metadata round-trip tests assert every field with three negative cases. The kvquant negative-path tests match specific error variants. The P1 strengthening block in grim-quant (`q80_round_trip_all_zeros_does_not_produce_nan`, `q4k_rejects_truncated_buffer`, `fp8_quant_clamps_to_representable_range`) shows the team can write mutation-resistant tests when they hold themselves to it.

The weak modules are exactly where the bugs live.

| Module | Score | Rating | The problem |
|---|---|---|---|
| grim-backend-cpu/src/simd_gemm.rs | 20 | WEAK | identity matrices hide transpose bugs (P0-5), range-band assertions |
| grim-quant (gptq block) | 0 | WEAK | all-zero input, is_ok() only (P2-5) |
| grim-memory/src/lib.rs | 40 | WEAK | permissive tier assertions, is_some() only |
| grim-autograd/src/ops.rs | 70 | FAIR | backward test checks shape not values (P2-4) |
| grim-format/src/tprov.rs | 80 | FAIR | certifies the P0-1 bug (P2-3) |
| grim-tensor/src/softmax_merge.rs | 95 | GOOD | the model to copy |

Manual mutation gutcheck on the critical paths: flipping `amax / 127.0` to `amax / 255.0` in the Q8_0 roundtrip still passes the old test. Transposing the `b` load in simd_gemm still passes (identity equals its transpose). Replacing the entire matmul backward with `vec![0.0; len]` still passes (shape check only). Swapping `Block(Fp4)` to `Block(Nf4)` in production fails the test on a fix and passes it on the bug, an inverted kill signal.

Kill-rate readiness for the suite as a whole: MEDIUM. The strong modules pull the average up. The four weak spots sit on the critical path (inference matmul, training gradients, weight loading) and each represents a class of bug the current suite would ship.

## What is clean

To calibrate, several crates have no logic defects in their numeric paths.

CUDA matmul uses the correct row-major to column-major recipe (swap A and B, `transa=transb='N'`, `m=N, n=M, lda=N, ldb=K, ldc=N`). The grim_qkv_attention kernel sizes its shared memory and warp configuration consistently with the launch. The Vulkan compute kernels are numerically correct where they run; the shaders for silu_mul, rms_norm, softmax, and embedding fall back to host simulation because `compile_glsl_to_spirv` always returns Err, which is intentional stubbing, not a numeric bug. Metal matmul uses Accelerate `cblas_sgemm` correctly. grim-tensor's online-softmax partial merge is algebraically correct with proper `m_old > m_new` guards. The RoPE implementation uses standard GPT-NeoX half-rotate pairing with the correct sign convention. RMSNorm applies epsilon inside the sqrt.

## How to read the priorities

The P0 list is the one to act on first. P0-1, P0-2, and P0-3 together mean quantized weight loading is broken for the most common GGUF formats on Hugging Face. P0-4 and P0-5 mean the two primary inference matmul paths are wrong for half-precision and LoRA. P0-6 through P0-8 mean speculative decoding, KV append, and chunked prefill produce wrong results.

The P1 list is correctness on reachable paths and resource leaks. The pool exhaustion from P1-1 and P1-2 will surface as "block pool exhausted" errors under sustained load. The P1-4 throughput throttle is silent.

The P2 list is the broken feature build, the CLI failure-reporting gaps, the dead code, and the weak tests. Fix the tests as you fix the bugs, or the green suite keeps hiding the next regression.

## Verification of this review

I confirmed the workspace compiles and the lib tests pass (522, 0 failed). I read the code rather than skimmed it. I verified P0-1, P0-2, P0-3, and the P0-4 offset arithmetic against the actual source lines quoted above. The remaining findings come from read-only code review across all 28 crates and were checked for file and line accuracy by the reviewing agents. I did not apply any fixes; this review lists them.

If you want, the next step is to write the fixes for the P0 list as a patch series, starting with the gguf dtype mapping and its locking test.
