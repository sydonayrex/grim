# Qwen3.5 GPU inference speed — implementation plan

**Context**: `crates/grim-models/transformer/src/qwen35.rs` runs inference on GPU but is slow. Investigation traced the bottleneck to host round-trips in the forward body, not to the linear layers or norms (those already dispatch device kernels).

**Goal**: eliminate avoidable host round-trips in `Qwen35Block::forward` and `Qwen35::forward` so a loaded Qwen3.5 runs at GPU speed, while keeping the existing device-resident primitives (`Linear::forward`, `RmsNorm::forward`, `silu_mul_on_device`, `add_on_device`, `Rope::forward`, `fused_or_scalar_attention_arena`) as the safe replacement surface.

**Success criteria (honest, scoped to what the existing API can deliver)**:
1. `Qwen35Block::forward` body contains zero `.to_vec_f32()` calls that pull device tensors back to host for CPU-side math.
2. `Qwen35::forward` body contains zero `.to_vec_f32()` calls for the same reason.
3. K and V stay resident on device across steps; full K/V history is no longer re-uploaded on every attention call.
4. The remaining Q upload (required by the existing `fused_or_scalar_attention_arena` signature, which takes `q: &[f32]`) is acknowledged as an open item, not claimed as eliminated in this phase.
5. CPU parity test on a TinyQwen harness passes before and after each phase, so correctness is bracketed.

**Scope (what this plan does NOT touch)**:
- `Linear::forward`, `RowParallelLinear::forward`, `RmsNorm::forward`, `Embedding::forward`, `Rope::forward` — those already dispatch device kernels or have documented lazy-sync discipline. Verified by reading `crates/grim-nn/src/modules.rs`.
- The ROCm/CUDA backend kernels themselves (`silu_mul`, `qkv_attention`, `broadcast_bias`, `rope`, `add`) — they exist and are wired; the work is calling them from qwen35.rs instead of re-implementing the math on the host.
- `apply_rope_neox` in `qwen35.rs` — it is a shared helper used by 25+ other model files. Replacing it is a separate concern. This plan's Phase 0 structural test explicitly excludes it from the "zero host round-trips" claim for the full-attention path, because Q must be uploaded for attention anyway while the shared attention API takes `q: &[f32]`.

---

## Findings (what's wrong, one line each)

- `qwen35.rs:361-418`: full-attention path pulls `wq.forward` result to host via `.to_vec_f32()?` (line 363), same for `wk` (372), `wv` (381), then runs `apply_rope_neox` on host slices, then uploads K/V into a host `Vec` cache, then calls `fused_or_scalar_attention` which re-uploads everything. Each of those is a D2H or H2D sync.
- `qwen35.rs:421-431`: SSM path pulls `attn_qkv.forward` to host (423), then manually slices and applies `silu` in a host loop (425-431). Should be `silu_mul_on_device` on the device tensor.
- `qwen35.rs:435-442`: gate sigmoid is a host loop over `out_branch` (438-441). Should dispatch a device elementwise kernel or fuse into the Qslice path.
- `qwen35.rs:461-464`: SwiGLU FFN pulls both gate and up to host via `.to_vec_f32()?` in the local `silu_mul` (697-705), loops scalar, then re-uploads. `silu_mul_on_device` already exists at `modules.rs:35` and is used by `minicpm.rs`, `block.rs`, `muse_glimmer.rs`, `tp_layers.rs`.
- `qwen35.rs:455,467`: two residual adds go through `add_tensors` which materializes on host (or dispatches device add with a host bounce for broadcasting). Residual shapes in this model are always identical (both operands are `[seq_len, hidden_size]`), so `add_on_device` at `modules.rs:79` is safe once we assert the equal-shape invariant.
- `qwen35.rs:404-417`: KV cache is a host `Vec<f32>` that grows via `extend_from_slice`, and every attention call re-uploads the entire history. The arena path `fused_or_scalar_attention_arena` already exists in `shared_attention.rs` to avoid this, but qwen35.rs calls the host-history overload instead.
- `qwen35.rs:646-651,669-675`: cross-device layer transfer bounces the full hidden state through host via `transfer_tensor` (D2H + H2D). `move_to_device` at `modules.rs:110` is the same operation with a clearer name; both are host bounces. On a single-device load this path is a no-op (device matches), but on multi-GPU it costs a full round-trip per boundary. For the single-GPU case this is not the hot path; the fix is a future P2P async copy, out of scope here.
- `qwen35.rs:697-705`: local `silu_mul` duplicates `grim_nn::modules::silu_mul_on_device`. Dead code once Phase 1 lands.

---

## Phase 0: harness + failing tests (red)

**Files to create/modify**:
- `crates/grim-models/transformer/src/qwen35_perf.rs` — already written, compiles. Contains:
  - `TinyQwen` harness: 2-layer Qwen3.5 with synthetic deterministic weights, implements `Model + CausalLm` by delegating to the same block loop `Qwen35::forward` uses.
  - `phase0_cpu_parity_baseline` — runs TinyQwen on CPU, asserts output shape and non-NaN. Must pass now.
  - `phase0_structural_host_roundtrip_in_forward_body` — reads `qwen35.rs` on disk, extracts the `Qwen35Block::forward` body, counts `.to_vec_f32()` calls. Expects 0. Must FAIL now (current code has several).
  - `phase0_device_parity_target` — ignored, requires ROCm. Future gate: once Phase 3 lands, enable and assert CPU/GPU output parity within 1e-3 relative.

**What must fail before any implementation**:
- `phase0_structural_host_roundtrip_in_forward_body` — it counts real `.to_vec_f32()` calls in the current forward body. Run it, confirm it fails with count > 0. Do not edit the test to make it pass.

**Run**:
```
cargo test -p grim-models-transformer phase0_structural_host_roundtrip_in_forward_body -- --nocapture
```
Expected: assertion failure with the current count.

**Freeze rule**: after Phase 0, do not edit these tests except to fix genuinely broken test code. If a later phase breaks parity, fix the implementation, not the test.

---

## Phase 1: FFN silu_mul → silu_mul_on_device

**Scope**: `qwen35.rs` lines 461-464 and the local `silu_mul` helper at 697-705.

**Current code**:
```
let gate = self.ffn_gate.forward(&h_normed)?;
let up = self.ffn_up.forward(&h_normed)?;
let act = silu_mul(&gate, &up)?;
let ffn_out = self.ffn_down.forward(&act)?;
```

**Replacement**:
```
let gate = self.ffn_gate.forward(&h_normed)?;
let up = self.ffn_up.forward(&h_normed)?;
let act = grim_nn::modules::silu_mul_on_device(&gate, &up)?;
let ffn_out = self.ffn_down.forward(&act)?;
```

Drop the local `silu_mul` at 697-705 once nothing else in this file references it.

**Why this is safe**:
- `silu_mul_on_device` dispatches `dev.silu_mul` on the device that owns `gate`. Both `gate` and `up` are outputs of `Linear::forward`, which returns device-resident tensors (verified: `modules.rs:648-777` dispatches GEMM on device, lazy-syncs).
- The ROCm backend has `silu_mul` implemented (cubecl.rs:156, roc_device.rs:2385). CPU backend has its own path.
- Correctness anchor: `phase0_cpu_parity_baseline` runs the same harness on CPU before and after; if the device kernel diverges from the host `silu_mul` reference, the CPU test still passes (it runs on CPU where the old code and new code both use the same CPU path), so this phase's risk is GPU-only divergence, caught later by the ignored `phase0_device_parity_target`.

**Test that must still pass**: `phase0_cpu_parity_baseline`.

**Commit point**: after `cargo check` and `cargo test -p grim-models-transformer phase0_cpu_parity_baseline` pass.

---

## Phase 2: residual adds → add_on_device

**Precondition (must assert before relying on add_on_device)**:
In `Qwen35Block::forward`, both residual adds are:
- `add_tensors(x, &proj_out)` at line 455 — `x` is the input `[seq_len, hidden_size]`, `proj_out` is `wo.forward` or `ssm_out.forward` output, same shape.
- `add_tensors(&h, &ffn_out)` at line 467 — `h` is the post-attention residual `[seq_len, hidden_size]`, `ffn_out` is `ffn_down.forward` output, same shape.

Both operands always have identical shape because every `Linear::forward` in this model preserves the batch dimension and the output dim matches `hidden_size`. This is true for this model by construction (qwen35.rs config: `hidden_size=5120`, every row-parallel/output linear projects back to `hidden_size`). Document this as an assertion in the plan, and add a debug-assert or a test that the two operands have equal shape at the call site if you want it encoded. For now, the claim is: residual operands are always equal-shape in this model, so `add_on_device` (which has no broadcast path) is safe.

**Replacement**:
```
use grim_nn::modules::add_on_device;

// Residual 1
let h = add_on_device(x, &proj_out)?;

// ... FFN ...

// Residual 2
let out = add_on_device(&h, &ffn_out)?;
```

Keep using `add_tensors` for anything that might need broadcasting. In this file, after the replacement, the only add calls left should be these two, both equal-shape.

**Why this is safe**:
- `add_on_device` at `modules.rs:79` dispatches `dev.add` on device. No host bounce for equal-shape tensors.
- CPU backend path: `add_on_device` still works on CPU tensors (it picks the device from the tensor, falls back to CPU device). So the CPU harness still passes.

**Commit point**: after CPU parity passes.

---

## Phase 3: full-attention path — RoPE stays host, K/V → device arena, Q upload acknowledged

This is the phase the review flagged as overclaimed. Here is the corrected, honest version.

### What Phase 3 actually does

1. **K and V stay on device across steps.** Replace the host `Vec<f32>` cache (`cache.k_cache`, `cache.v_cache`) with device-resident K/V storage, and stop calling `cache.k_cache.extend_from_slice(&k_all)` + `cache.v_cache.extend_from_slice(&v_all)` + re-uploading the whole history every call.

2. **Use `fused_or_scalar_attention_arena` instead of `fused_or_scalar_attention`.** The arena path keeps K/V on device and only materializes the new step's rows for the host fallback. This eliminates the per-step full-history upload.

3. **Q upload remains.** `fused_or_scalar_attention_arena` takes `q: &[f32]`, so Q still gets uploaded once per attention call (H2D of the current step's Q, not the whole history). The plan does NOT claim to eliminate this in Phase 3. Eliminating Q upload requires a new attention entry point that accepts device-resident Q, which is future work.

4. **RoPE on Q stays on host for now.** The current flow: pull Q to host via `.to_vec_f32()?` → `apply_rope_neox(&mut q_all, ...)` on host → upload Q to device for attention. To eliminate the Q upload you'd also need a device RoPE kernel path for this model's Q (the shared `Rope::forward` exists and dispatches `dev.rope` on device, but qwen35.rs calls the local `apply_rope_neox` on a host slice). Phase 3 leaves this as the remaining host round-trip in the full-attention path: Q is uploaded once (post-RoPE) per attention call, and that upload is the residual the structural test is scoped to allow.

### What changes in the code

**KV cache representation**: The current `Qwen35LayerCache` has `k_cache: Vec<f32>`, `v_cache: Vec<f32>`. For the arena path, these need to become device-resident storage handles. This is a struct change + a session-state change (the cache is stored in `session.model_state` as `Vec<Qwen35LayerCache>`).

The arena path wants K/V as `&dyn BackendStorage` that lives on device. The simplest correct approach:
- Allocate device K/V buffers of max capacity once (or grow geometrically), track `kv_len`.
- On each step, copy the new K/V rows into the device buffer (device-side copy or H2D of just the new rows), update `kv_len`.
- Call `fused_or_scalar_attention_arena` with the device K/V storage and the current Q host slice.

This is more invasive than Phase 1/2 because it changes the cache type and the session state. That's fine — Phase 3 is the big one. The correctness anchor is still `phase0_cpu_parity_baseline` (CPU path keeps using the host-history `fused_or_scalar_attention`, unchanged) plus the future `phase0_device_parity_target` when enabled.

### The honest success criterion for Phase 3

After Phase 3:
- `Qwen35Block::forward` full-attention path no longer re-uploads full K/V history every step.
- Q is still uploaded once per attention call (required by the existing arena API).
- `apply_rope_neox` is still called on a host Q slice.
- The structural test `phase0_structural_host_roundtrip_in_forward_body` counts `.to_vec_f32()` calls in the forward body. Phase 3 will remove several but leave the Q upload (and possibly the RoPE buffer materialization). If the test is to go green, its scope must be adjusted to count only the avoidable round-trips, not the Q upload that the current shared attention API mandates. That adjustment is a test-scope change, not an implementation gap, and it must be made explicit before the test is expected to pass.

**Recommendation**: before Phase 3 implementation, rewrite `phase0_structural_host_roundtrip_in_forward_body`'s assertion to enumerate the specific patterns that must be gone (e.g. "no `.to_vec_f32()?` on a tensor returned by `Linear::forward` except where the result is consumed by an API that requires `&[f32]` and no device-resident variant exists"), rather than a blunt count == 0. That keeps the test honest and achievable.

---

## Phase 4: SSM Qslice path (only if Phase 1-3 leave a host round-trip there)

After Phase 1, the FFN uses `silu_mul_on_device`. The SSM path at 421-431 still:
- pulls `attn_qkv.forward` to host (423),
- manually slices Q portion and applies `silu` in a host loop (425-431).

The gate path at 435-442 still applies sigmoid on host.

Whether this matters for speed depends on how many layers are SSM vs full-attention. Qwen3.5 config: `full_attention_interval=4`, so 15 full-attention layers and 50 SSM layers out of 65. The SSM path is the majority. If the Qslice + gate can be fused into a device kernel, that's a real win on the SSM-heavy path.

**Option A (punted)**: leave the SSM path as-is for now. Phase 1-3 already remove the hot-host-math from full-attention and FFN. The SSM path's host round-trip is bounded: it pulls `qkv` to host, slices, gates, then uploads `branch_tensor` back to device for `ssm_out.forward`. That's one D2H + one H2D per SSM layer per step. On a GPU-heavy workload the full-attention path's KV re-upload (eliminated in Phase 3) is the bigger cost, so Phase 4 can be deferred.

**Option B (if you want the full elimination)**: write a device kernel or use existing primitives to:
- apply `silu` to the Q portion of `attn_qkv` output on device,
- apply the gate sigmoid on device,
- produce `branch_tensor` on device without host materialization.

There is no existing fused "Qslice + silu + gate-sigmoid" kernel in the backend today (I checked: `silu_mul` exists, `rope` exists, `qkv_attention` exists, `add` exists, but no fused qkvslice-and-activate). Writing one is a kernel authoring task, beyond the scope of "delete host round-trips and call existing device primitives." Flag Phase 4 as: if you want zero host round-trips in the SSM path, someone needs to author that kernel; otherwise Phase 4 = remove the host `silu` and sigmoid loops by calling device elementwise ops on the device tensor before slicing, which still leaves the Qslice itself on host unless a slice kernel exists.

**Honest recommendation**: Phase 4 = replace the host `silu` loop and host sigmoid loop with device elementwise calls where the tensor is still on device, and accept that the Qslice extraction (`out_branch[t * q_dim + d] = silu(qkv_vec[base + d])`) currently requires a host buffer because `attn_qkv.forward` returns a single tensor with Q+K+V concatenated and the code slices the Q portion. To do that slice on device you'd need either a slice kernel or a restructured linear that outputs Q separately. That's a design change, not a refactor. Leave Phase 4 as "remove host elementwise loops; Qslice still host" unless someone authors the slice kernel.

---

## Phase 5: transfer_tensor → move_to_device + dead code cleanup

**Scope**: `qwen35.rs` lines 669-675 (`transfer_tensor`) and the calls at 648, 655.

**Replacement**: use `grim_nn::modules::move_to_device` (same semantics: host bounce if devices differ, clone if same). `transfer_tensor` is a duplicate with less clear naming.

**Reality check**: on a single-device load (the common case), `h.device() == block.device()` is always true, so the transfer path is never entered. The cross-device path is only hit in multi-GPU layer pipelining. Phase 5 is cleanup, not a speed win for single-GPU. Don't oversell it.

**Dead code to remove once Phase 1 lands**: local `silu_mul` at 697-705.

**Commit point**: after cleanup compiles and CPU parity passes.

---

## Verification gates (run in order)

After Phase 0:
- `cargo check -p grim-models-transformer`
- `cargo test -p grim-models-transformer phase0_cpu_parity_baseline` — must pass
- `cargo test -p grim-models-transformer phase0_structural_host_roundtrip_in_forward_body` — must FAIL (red)

After each phase 1-2 and 5:
- `cargo check -p grim-models-transformer`
- `cargo test -p grim-models-transformer phase0_cpu_parity_baseline` — must still pass
- `cargo test -p grim-models-transformer phase0_structural_host_roundtrip_in_forward_body` — re-run after each phase; count should decrease; do not edit the test to force pass unless the scope adjustment in Phase 3 is applied.

Before Phase 3 implementation:
- Rewrite the structural test's assertion to the honest scope (per Phase 3 section above).
- Decide whether to implement the device KV arena now or defer. If deferred, Phase 3 is just the test-scope adjustment + documentation of the Q-upload residual.

After Phase 3 (if implemented):
- Enable `phase0_device_parity_target` on a machine with ROCm and run it. Expect it to fail until the device path is correct; do not disable it to make CI green.

---

## Open items (not fixed by this plan)

1. **Q upload in full-attention path** — remains until a device-resident-Q attention entry point exists. Phase 0 structural test must be scoped to allow it, or Q upload must be eliminated with a new kernel/entry point.
2. **SSM Qslice on device** — requires either a slice kernel or restructuring `attn_qkv` to output Q separately. Phase 4 currently only removes the host elementwise loops, not the slice itself.
3. **VRAM numbers** — the investigation noted the model is large (65 layers, hidden 5120, 24 Q heads, 4 KV heads, head_dim 256, intermediate 17408). Exact VRAM depends on the GGUF quantization and KV cache growth. Treat any "fits in X GB" claim as illustrative until measured against a real loaded checkpoint.
4. **Cross-device transfer speed** — `transfer_tensor`/`move_to_device` both bounce through host. A true multi-GPU speedup needs P2P async copies or NVLink/RCCL-aware transfers. Out of scope.
5. **`GRIM_QKV_FUSED=0` escape hatch** — `shared_attention.rs:51` forces the scalar path. If set, even the fused device kernel is bypassed. Verify this env var is not set in the run environment, or the device kernel is never called regardless of qwen35.rs changes.

---

## Commit strategy

One commit per phase, each with:
- the code change,
- the test that gates it,
- `cargo check` and the relevant `cargo test` output in the commit body if the user wants evidence.

Do not bundle unrelated phases into one commit. Phase 1 and Phase 2 can be one commit if they're both trivial replacements and both pass CPU parity, since they touch neighboring lines in the same function. Phase 3 is its own commit (it changes the cache type and session state). Phase 5 cleanup is its own commit.

---

## Skills applied

- `caveman-review` format for findings (terse, one line, location + problem + fix).
- `safe-refactor` discipline: behavior-preservation boundary is the CPU parity test; each phase keeps the CPU path passing; GPU path is bracketed by the ignored device-parity test until enabled.
- `rust-testing` for test organization: harness in-file with `#[cfg(test)]` module pattern matches existing transformer crate conventions (see `deepseek.rs` tests, `minicpm.rs` tests).
- `project-planner` phase structure: each phase has a clear done criterion, a verification gate, and a commit point.
- `verification-before-completion` gate: no phase is "done" until its test passes; no claim of GPU speedup until `phase0_device_parity_target` passes on real hardware.
