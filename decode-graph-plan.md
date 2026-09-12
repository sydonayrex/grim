# DECODE-GRAPH PLAN: 3 items to collapse grim decode host overhead

**Goal:** LFM2.5-350M-Q8_0 on gfx1201: 118 tok/s → ≥300 tok/s (stretch 400).
**Method:** red-green-refactor per item. Every item lands with GPU parity tests + launch-count assertions + full-model benchmark.

**ISA basis (runtime-verified on gfx1201, ROCm 7.2 — the AMD doc portal is login-walled; do NOT re-derive from the RDNA1–4 PDFs):**
- `__builtin_amdgcn_sudot4(true,a,true,b,c,false)` = signed i8×i8 dot4, probe 8/8. Raw asm `v_dot4_i32_i8` is WRONG on gfx12 (unsigned-ish). Scalar `sdot4` absent (`dot1-insts` removed).
- **VERIFIED vs RDNA4 ISA §7.7/§16.10 (old/amd-isa/rdna4-…pdf):** `V_DOT4_I32_IU8` exists (opcode 22, base feature — "does not depend on inference/DL features"). Sign interpretation is controlled by the **per-operand NEG modifier** (0=unsigned, 1=signed per input). Raw asm without modifiers → NEG=0 → unsigned → our probe's 1020-instead-of-−4. The sudot4 builtin's `true, a, true` args set the signed NEG bits. **Never replace the builtin with raw asm.** Also present: `V_DOT8_I32_IU4` (8-wide int4 — future int4 GEMV) and `V_DOT4_F32_{FP8,BF8}` variants (FP8 dot4 — future FP8 path).
- `v_dot2_f32_f16`: signed fp16 dot, correct.
- WMMA: wave32-only, block-16-only (rocWMMA 2.2 static_assert).
- HIP graphs: capture-safe only if ALL launches inside the bracket have byte-identical args (pointers AND scalars) every replay. Host syncs (D2H memcpy, hipStreamSynchronize, hipMemcpy blocking) inside a bracket abort capture.
- HIP graph launch fixed cost ≈ 40–60 µs; only wins when a graph replaces ≥20 launches. Measured: 32 small (6-node) graphs = net +5% (see `device/segment_replay.rs` doc comment — keep that gate).

---

## EXISTS (do not re-implement)

| Artifact | File | State |
|---|---|---|
| sudot4 MMVQ GEMV (q8_1 act quant, 4-col) | `kernels/dot_gemv.rs`, dispatch `device_quant.rs:228+` | default-on, parity green. `GRIM_DOT_GEMV=0` disables, `GRIM_DOT_GEMV_LEGACY=1` forces dot2 |
| Fused RMSNorm+q8_1 quant kernel | `kernels/rmsnorm_quant.rs` + launcher `device/rmsnorm_quant_launch.rs` (`launch_rmsnorm_quant_i8`) | parity green; wired into `lfm2.rs` decode path (U8-typed tensor marker → dispatch skips re-quant) |
| Graph primitives | `roc_device.rs`: `begin_graph_capture(key)` / `end_graph_capture(key)` / `replay_graph(key)` / `has_captured_graph(key)`; suppression flag `device/segment_replay.rs` (`SEGMENT_SUPPRESS` + `run_segment`) | tested by `tests/wmma_quant_graph_capture.rs`; segment variant measured wash, gated `GRIM_SEGMENT_GRAPH=1` |
| Preallocated decode tensors | `run.rs` decode loops: `decode_input`/`decode_pos` rewritten per step via `write_f32_into` (stable device ptrs) | green |
| Launch-ceremony strip | thread-local DeviceGuard, ALLOC_TRACE OnceLock, upload-fence flag | green |
| GPU sampler | `run.rs::sample_on_rocm` (default; `GRIM_CPU_SAMPLER=1` escape) | green |

## DO NOT TOUCH

- `kernels/dot_gemv.rs` sudot4 kernels + their unit tests (other AI's file; parity-green).
- `device_quant.rs` existing dispatch arms except the exact insertion points named below.
- `run.rs` sampler selection logic; CPU backend (`grim-backend-cpu`); prefill paths (multi-row stays eager); IQ/GSQ kernels; `roc_device.rs` graph primitives internals; `segment_replay.rs` gate (it stays opt-in and untouched).
- `bench`/`eval` device auto-probe (GPU-first now — regression = fail).
- Anything in `crates/grim-models/transformer/src/lfm2.rs` outside the exact functions named below (file is shared — keep diffs surgical).

## NUMERIC REQUIREMENTS (apply to every item)

1. Parity: greedy decode of "Say hello" (12 tok, temp 0) produces token IDs identical to current eager sudot4 path for the first 6 tokens (prefix through `today`); full-sequence output may tie-flip only at logged near-ties — assert `max_diff(logits) ≤ 5e-1` vs eager at every step (capture logits both paths in the test via `GRIM_QMM_TRACE`-style hooks or direct device calls).
2. Launch-count: assert HIP API calls per decode token via `rocprofv3 --hip-trace` drop ≥40% after Item 1+2, ≥80% after Item 3 (baseline 565/token — recorded in this plan).
3. Perf gate per item: T(200)−T(1) subtraction benchmark, 4 runs, must not regress >2% vs 8.37 ms/tok baseline; final combined target ≤4.0 ms/tok (≥250 tok/s), stretch ≤2.8 (llama.cpp parity).
4. All existing suites stay green: 413 lib tests, `dot_gemv_parity`, `rmsnorm_quant_parity`, `wmma_quant_parity`, `wmma_quant_graph_capture`.

---

## ITEM 1: Fused QKV — one GEMV instead of three

**Now:** decode attention block calls `wq/wk/wv` = 3 Linears on the SAME q8_1 input → 3 GEMV launches (quantize already shared after OPFUSE). Each GEMV re-reads the same activation and issues its own launch.
**Change:** at model load, concatenate the three packed Q8_0 weight blobs `[n_q+2·n_kv, hidden]` (row order: all Q rows, then all K rows, then all V rows) into ONE storage; decode issues ONE `grim_dot4_q80_q81_gemv` (its 4-column blocking already handles N = n_q+2·n_kv) writing `attn_out [1, n_q+2·n_kv]`; then slice `q = out[..n_q]`, `k = out[n_q..n_q+n_kv]`, `v = out[n_q+n_kv..]` as zero-copy `Tensor::new(storage.clone(), shape, …)` views (storage is `Arc` — slicing via new Tensor over same storage, offset allowed only if `Tensor::new` supports it; if not, add `Tensor::slice_rows(offset, rows)` returning an aliasing Tensor — 15 lines in `grim-tensor`).

**Files:** load path in `lfm2.rs` block builder (where `wq/wk/wv` Linears load — search `Lfm2Block {` at 1638/1693 variants); decode pre-attn bracket in `lfm2.rs::forward` (the `qkv_in` block inserted for OPFUSE); NEW `RocmStorage::view_bytes` in `grim-backend-rocm`.

**RESOLVED DESIGN NOTE (Tensor has no offset field):** `Tensor::new` carries dims only — no strides, no byte-offset; Tensor assumes storage from byte zero. Do NOT thread an offset through `grim-tensor` (every storage-reading op would need review). Instead add a backend-level aliasing view, ~40 lines in `grim-backend-rocm/src/memory/storage.rs` (or sibling file):

```rust
pub struct RocmStorageView {
    ptr: *mut c_void,          // parent.device_ptr + byte_offset
    bytes: usize,
    dtype: DType,
    shape: Shape,
    _parent: Arc<dyn BackendStorage>,   // keeps the allocation alive
}
// impl BackendStorage: device_ptr -> Some(self.ptr), bytes, dtype,
// provenance (GrimNative), as_any. Everything else default/unimplemented.
pub fn view_bytes(parent: &Arc<dyn BackendStorage>, off: usize, len: usize) -> Box<dyn BackendStorage>
```

Consumers (dot4 GEMV launcher, rmsnorm qk-norms, kv `copy_slice_into`) only touch `device_ptr` + `bytes`, so views work with zero kernel changes. Wrap views in `Tensor::new(Arc::from(view), shape, dtype, GrimNative, device)` as usual — Tensor stays offset-free. Row math: `off_bytes = row_idx * (k/32)*34`, `len_bytes = rows * (k/32)*34`.

**TDD:**
1. RED: `grim-models-transformer` unit test `fused_qkv_load_matches_split` — load a tiny LFM2 config, assert fused blob rows == `wq ∥ wk ∥ wv` rows byte-identical (Q8_0 blocks copied verbatim — both use 34-byte blocks; assert row bytes equal, not dequantized floats).
2. RED: GPU test `fused_qkv_gemv_parity` — run 3-GEMV path vs fused-GEMV path on same q8_1 input, assert `max_diff ≤ 1e-4` (same kernel, same math, only launch grouping differs — must be near-exact).
3. GREEN: wire decode path; prefill (steps>1) and non-Rocm keep the 3-Linear path.
4. Assert launch count: attention layer decode = 1 fused-norm-quant + 1 GEMV (was 1 + 3).
5. REFACTOR: remove dead `wq/wk/wv` per-decode calls only after parity green; keep fields (prefill uses them).

## ITEM 2: RoPE positions — device-side, uploaded once

**Now:** `dev.rope(q_norm, &q_positions, …)` uploads a host `Vec<u32>` (cache_offset+t repeated per head) EVERY layer EVERY token (~2 uploads × attn layers × tokens; ~64+ `hipMemcpy` calls/token).
**Change:** upload positions ONCE per generation: buffer `pos_base_dev [1] u32` + modify the RoPE decode kernel to read the base position from device memory and iterate `t = 0..steps` internally (positions = base+t, repeated per head as today). New kernel entry `grim_rope_dev_base` alongside existing `grim_rope` (do NOT change existing entry — prefill keeps host-position path).
**Files:** rope kernel source (find entry in `kernels/` via `grep -rn "grim_rope" kernels/`); new launcher `launch_rope_dev_base(&q_storage, &pos_base_dev, head_dim, rope_theta, heads, steps)` in `device_compute.rs`; call site in `lfm2.rs` replacing the `q_positions`/`k_positions` Vec build (lines ~613-635).
**Past counter:** add `pos_base_dev` to `Lfm2LayerCache::Attention` (per-layer buffer, 4 bytes) OR one shared per-generation buffer — SHARED is required for Item 3 (one counter). Update per token: `pos_base_dev[0] = cache_offset` via `write_host_f32`-style write (host write, OUTSIDE any graph in Item 2; in Item 3 the graph itself increments it — see below).

**TDD:**
1. RED: GPU test `rope_dev_base_parity` — random q [1, heads, head_dim], base positions B ∈ {0, 17, 500}: new kernel vs existing `grim_rope` with host positions [B+t], assert `max_diff ≤ 1e-6` (fp32 math identical).
2. GREEN: wire decode call site.
3. Assert: `hipMemcpy` H2D count per decode token drops by 2×attn_layers (rocprofv3 --hip-trace, query `rocpd_event` count).
4. REFACTOR: delete the per-step `q_positions`/`k_positions` Vec construction in the decode branch only.

## ITEM 3: Whole-decode-step graph (device-side KV offset + counter-increment node)

**Depends on:** Items 1+2 landed. **Now-blocked-by:** KV append `copy_slice_into(k_dev, k_rot, off=past*stride, …)` and attention `total=past+steps` are host scalars baked into launches; `past` changes per token → byte-identical args impossible.
**Change — make the offset device-driven:**
1. New per-generation device buffers (allocate in `run.rs`, store in session): `past_dev [1] i32` (starts at prefill token count) and keep `pos_base_dev` from Item 2 aliased to it.
2. New KV-append kernel `grim_kv_append(k_arena, k_rot, past_dev, kv_stride, steps)`: each thread reads `past = *past_dev`, computes `off = past*stride + tid`, copies `steps*stride` elems — offsets computed ON DEVICE. Replaces `copy_slice_into` (which takes host `off_elems`).
3. Attention kernel variant `grim_qkv_attention_dev(q, k_arena, v_arena, attn_out, past_dev, kv_stride, heads, kv_heads, head_dim, steps)`: reads `total = *past_dev + steps` from device; loops KV 0..total. Base it on the existing qkv_attention kernel body (find entry via `grep -rn "grim_qkv_attention" kernels/`).
4. Counter-bump kernel `grim_bump_i32(past_dev, steps)`: `*past_dev += steps` — LAST node of the graph (graphs may mutate their own buffers; replay N increments exactly N times ✓).
5. Capture in `run.rs` decode loop: after prefill, step 1 = `begin_graph_capture("grim_decode_full")` → `model.forward(...)` (now fully static: q81 fused norm+quant, fused QKV GEMV, rope reads pos_base_dev, kv-append/attention read past_dev, bump last) → `end_graph_capture` → `replay_graph`. Steps ≥2: `write token+pos into fixed bufs` → `replay_graph` → sample. Session reset / turn change: `has_captured_graph` stays but buffers changed → force re-capture by keying graph on session id or adding `RocmDevice::drop_captured_graph(key)` (5 lines; call on turn start).
6. Constraints enforced by asserts in code: M==1 only; graph bracket contains NO D2H (sample_on_rocm stays outside; CPU-sampler fallback path skips capture entirely); capture failure → `CAPTURE_POISON`-style per-process eager fallback (mirror `segment_replay.rs` pattern).

**TDD:**
1. RED: GPU test `kv_append_dev_parity` — random arena + rot rows, past ∈ {0, 17, 500}: device-offset append == host-offset `copy_slice_into` result, byte-exact.
2. RED: GPU test `qkv_attention_dev_parity` — vs existing `qkv_attention` with identical total, `max_diff ≤ 1e-5` (fp32 same math).
3. RED: GPU test `bump_monotonic` — replay a captured graph 5×, assert `past_dev` increments by steps each replay (proves counter-in-graph works).
4. GREEN: whole-step capture in run.rs; prefix-parity vs eager (req 1); launch-count req 2; perf req 3.
5. REFACTOR: only after parity, delete the now-dead eager decode branches behind `GRIM_NO_GRAPH=1` (keep them — escape hatch, do NOT delete; refactor = extract the graph setup into `run.rs::try_capture_decode_graph()`).

## SEQUENCE & GATES

1. Item 1 (independent) → parity+bench → land.
2. Item 2 (independent) → parity+bench → land.
3. Item 3 (depends on 2; benefits from 1) → parity+bench → land.
4. Final: rocprofv3 --hip-trace re-run; report API calls/token (target <100), tok/s, suite status.

## RISKS

| Risk | Mitigation |
|---|---|
| Q8_0 row concatenation invalidates 34-byte alignment for K rows | Rows are byte-granular — GEMV already byte-loads (verified in dot4 kernel); parity test 1 covers it |
| Rope kernel reading device base changes numerics | It must not: same fp32 ops, only position source moves. Test 1 gates at 1e-6 |
| Graph captures an accidental host sync → abort | `run_segment`-style poison fallback to eager; test 4 verifies eager path still correct |
| `Tensor::slice_rows` aliasing breaks drop/allocator assumptions | Storage is `Arc<dyn BackendStorage>` — aliasing Tensors share the Arc; no new allocator interaction |
| Session/turn reset leaves stale graph | Key includes session generation counter; `drop_captured_graph` on turn start (interactive loop) |
