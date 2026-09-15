# twinkie-zombieland.md — finish the decode graph

The 6 phases of `i-was-dumb-graph.md` are implemented and compile. What remains is
making the graph path actually *faster* than eager — today the graph captures real
kills but the CLI reads back the full vocab logits every step and samples on CPU,
which re-introduces the exact overhead the graph was built to kill.

Order = impact first. Each phase lands independently; fallback eager preserved.

---

## P0 — GPU sampler on the graph path (kill the full-vocab D2H)

**Why:** Biggest remaining win. `try_graph_decode_step` calls `read_logits_f32()`
(sync D2H of entire vocab) then samples on CPU. The GPU sampler path (`gpu_sample_ok`
in run.rs:688) is never reached because the graph returns a host `Vec<f32>`, not a
device tensor. Every replay pays a full-vocab D2H + CPU sampling.

**Now:**
- `crates/grim-cli/src/run.rs:80` — `g.read_logits_f32().ok()?` → host vec
- `crates/grim-cli/src/run.rs:684-691` — `gpu_sample_ok` block uses `sample_on_rocm` but
  only sees logits from the eager `CausalLm::forward` path
- `crates/grim-backend-rocm/src/decode_graph_buffers.rs` — `DecodeGraph::read_logits_f32`
  exists only as a D2H helper

**Do:**
- Add `DecodeGraph::logits_device_storage(&self) -> &RocmStorage` returning a reference to
  `buffers.head_output` (the live device buffer the graph last wrote). Replay rewrites the
  *same* device memory, so the reference stays valid across replays.
- Change `try_graph_decode_step` (run.rs:28-83) to return the device storage + shape instead
  of a host vec when the GPU sampler can use it:
  - Return type becomes `Option<GraphDecodeResult>` where
    `GraphDecodeResult { storage: Arc<RocmStorage>, shape: Shape }` (or keep returning a
    `Tensor` whose storage is `head_output` — `Arc::clone` of the pool storage).
  - Keep `read_logits_f32()` for the CPU-sampler fallback only.
- In run.rs:674-692, when `graph_hit` is a device-backed logits tensor, set
  `gpu_sample_ok = true` and route through `sample_on_rocm`. Zero D2H on the GPU-sampler
  path.
- Repeat-penalty path (history non-empty, penalty > 1.0) still falls back to CPU
  sampler + D2H — acceptable, cold path.

**Accept:** `rocprof --hip-trace` per decode step on GPU-sampler path: 1 H2D (4B token) +
1 H2D (4B pos) + 1 `hipGraphLaunch` + 0 D2H. Logits stay device-resident.
**Test:** `GRIM_CPU_SAMPLER=1` forces the old D2H path (parity guard). Default path: 0
`hipMemcpyDtoH` per step in rocprof. Output parity vs eager within float eps.

---

## P1 — F32 M=1 GEMV for projections

**Why:** F32 projections (QKV, gate/up, O, down, lm_head) still dispatch through rocBLAS
via `linear_decode_into` → `matmul_into`. rocBLAS GEMM at M=1 wastes launch overhead and is
not reliably graph-capturable across replays. The dot4/WMMA decode kernels handle quant;
F32 needs its own M=1 path.

**Now:**
- `crates/grim-backend-rocm/src/device/device_compute.rs:4661` — `linear_decode_into`
  - `DTypeStorage::Native` (F32) → `self.matmul_into(a, w, out)` → rocBLAS `sgemm`
  - Q8_0 → dot4; Q4K/Q5K/Q6K →专用 GEMV. Only F32 is rocBLAS-bound.
- `launch_wmma_gemm_b_transposed` / `grim_wmma_gemm_b_transposed` exist (F16) but no F32
  M=1 kernel is wired for decode.

**Do:**
- Add `RocmDevice::launch_f32_gemv_into(&self, act: &RocmStorage, weight: &RocmStorage,
  out: &RocmStorage, n: usize, k: usize) -> Result<*c_void>` — grid (N,1,1), block
  (32,1,1), one wave per output element, `fdot2`-style dot product across K. Reuses the
  `grim_rms_norm` warp-row launch helper for the reduction.
- Or (simpler, capture-safe): route F32 M=1 through the existing WMMA path by casting to
  the `launch_wmma_gemm_b_transposed` decode GEMM (`grim_decode_gemm`) which already
  handles M≤8 — but that kernel is F16-only today. Prefer a dedicated F32 kernel.
- Wire `linear_decode_into`'s `DTypeStorage::Native` arm to call `launch_f32_gemv_into`
  instead of `matmul_into` when `m == 1`.
- Gate behind `GRIM_F32_GEMV != 0` opt-out (like other decode kernels); default-on once
  parity-green.

**Accept:** rocprof F32 decode step: projections show `hipModuleLaunchKernel` (custom GEMV)
not `rocblas_sgemm`. Parity vs eager F32 eager path bit-identical (same f32 FMA math).
**Test:** `GRIM_F32_GEMV=0` falls back to rocBLAS (parity guard). 100-step output match vs
eager on LFM2-350M-F32.

---

## P2 — wire DecodeGraphBuffers into grim-engine (unify capture paths)

**Why:** Two parallel capture systems today — CLI uses the new fixed-buffer path
(`Lfm2::forward_capture` + `DecodeGraphBuffers`); the server/engine uses the old per-op
stream capture (`drive_forward_graph_capture` → device `GraphCaptureManager` running the
full eager `decode_one`). The old path still pays eager overhead on capture and relies on
rocBLAS-in-graph. One path is simpler to debug and the fixed-buffer path is correct by
construction (stable addresses, no scratch allocs in capture).

**Now:**
- `crates/grim-engine/src/lib.rs:2131` — `drive_forward_graph_capture` uses
  `rocm.begin_graph_capture` / `decode_one` / `replay_graph` (device GraphCaptureManager).
- `crates/grim-engine/src/lib.rs:1958` — `try_graph_decode_item` (P4 batched) is the old path.
- `crates/grim-cli/src/run.rs:28` — `try_graph_decode_step` uses the new fixed-buffer path.

**Do:**
- Add a request-scoped `DecodeGraph` to the engine's session/state
  (`EngineSession` or a per-request `decode_graph: Option<FullDecodeGraph>`), allocated
  lazily via `Lfm2::get_or_create_decode_graph(max_ctx)` on first decode.
- In `drive_forward_graph_capture`, when `loaded.model` is `Lfm2` on ROCm:
  - No graph yet → `get_or_create_decode_graph` + `begin_capture` + `forward_capture` +
    `end_capture` (one eager run records the graph). Seed `current_pos`.
  - Graph exists → async H2D token + pos (reuse `write_embedding_to_buffer` +
    `write_pos_async`), `replay()`. No `decode_one`, no `GraphCaptureManager`.
  - Cache the logits `Arc` keyed by `capture_key` (already done at lib.rs:2172) — replay
    rewrites `head_output` in place, cached Arc stays valid.
- Fall back to the old `decode_one` path for non-LFM2 models, non-ROCm, MoE/ShortConv
  stacks (the new path returns `Unimplemented` for those — lib.rs downcast to `Lfm2`
  fails → fallback).
- Delete the now-dead `decode_one`-on-capture branch in `drive_forward_graph_capture`
  once parity is green; keep `GraphCaptureManager` for non-LFM2 models.

**Accept:** `rocprof` engine decode step: 1 `hipGraphLaunch`, 0 `hipModuleLaunchKernel`
outside capture, 0 `hipStreamSynchronize` on hot path. Same logits as CLI graph path.
**Test:** Server (`grim-server`) 100-step generation parity vs CLI eager on same prompt/seed.

---

## P3 — batch decode (DecodeBucketGraphPool)

**Why:** Batch 1 leaves CUs idle; the attention dot kernels handle M≤8 with the same
launch. Batching amortizes the layer-loop overhead across N requests.

**Now:**
- `crates/grim-backend-rocm/src/graph_capture.rs:266` — `DecodeBatchBucket`,
  `DecodeBucketGraphPool`, `DecodeGraphKey` already exist for GEMM-level batching.
- `crates/grim-cli/src/run.rs` — single-stream loop, no batching.
- Engine `step_batch` exists but routes through the old capture path.

**Do:**
- `DecodeGraphBuffers::allocate` gains a `batch: usize` param; pool slots become
  `[batch, hidden]` / `[batch, n]` (the dim-0 broadcast in kernels handles batch via the
  existing `[1, ...]` → `[batch, ...]` shape; most elementwise kernels are batch-agnostic,
  attention reads `steps = batch`).
- `Lfm2::forward_capture` / `forward_replay` take `batch: usize`; QKV/FFN GEMVs broadcast
  over batch (M=1 GEMV → M=batch); `attention_forward_graph` passes `steps = batch` to
  `launch_kv_append` / `launch_qkv_attention_dev`.
- Engine `try_graph_decode_item` groups pending decode items by shape bucket
  (`DecodeBatchBucket { batch, seq_len, ... }`), captures one graph per bucket, one launch
  per bucket.
- CLI stays batch 1 (single-user); engine gets the batch speedup.

**Accept:** Batch 4 decode tokens/sec/token > 2.5× batch 1. rocprof: 1 launch per bucket.
**Test:** `decode_graph_input_buffers` per-request isolation test (different tokens →
different outputs). 4-request batch parity vs 4× single-request.

---

## P4 — tests (correctness, benchmark, stress)

**Why:** No verification that the graph actually matches eager or survives long runs.
`im-with-stupid.md` verify section requires bit-exact (F32) / <0.01 (quant) parity,
200-step min, and a 1000-step stress test.

**Do:**
- `crates/grim-models-transformer/src/lfm2_graph.rs` test module (or a new
  `tests/decode_graph_parity.rs` integration test):
  - Load LFM2-350M (F32 + Q8_0) on gfx1201/gfx1036.
  - 50-step eager forward (collect logits + sampled tokens).
  - 50-step graph forward (capture step 0, replay steps 1-49, `GRIM_DECODE_GRAPH=1`).
  - Assert token sequences bit-identical (F32) or logit L2 < 0.01 (quant).
- Stress: 1000+ graph replays, assert no alloc growth (`GRIM_ALLOC_TRACE` clean), no
  address drift (logits unchanged for fixed input), no hipGraph error after 1000 replays.
- Benchmark harness: warmup 50, measure 200, report ms/tok + tok/s. Compare eager vs
  graph vs graph+GPUsampler.

**Accept:** All three pass on gfx1201 (RDNA4) and gfx1036 (RDNA3). CI skips gracefully
when `gpu_test_enabled() == false`.
**Env:** `GRIM_DECODE_GRAPH=0` → eager, zero behavior change (already wired).

---

## Verify each phase

- `cargo check --workspace` + `cargo test -p grim-backend-rocm --lib graph` +
  `cargo test -p grim-models-transformer --lib -- --test-threads=1`
- Correctness: eager vs graph bit-exact (F32) / <0.01 (quant), 200 steps min.
- Bench: warmup 50, measure 200, `rocprofv3 --hip-trace` launch count +
  `hipStreamSynchronize` count.
- Fallback: any capture/alloc/replay `Err` → eager, `graph_failed` latches, no panic.

## Order recap

P0 GPU sampler → P1 F32 GEMV → P2 engine unify → P3 batch → P4 tests. Stop when
`~2.6ms/tok (~385 tok/s)` hit on the GPU-sampler path or rocprof shows compute-bound
(VALU busy, no SQ stall on launches).
