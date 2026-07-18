# Implementation Plan: `.grim` V2 — Closing the Format↔Kernel Gap

## 0. Reality check (supersedes `grim_v2.md`)

`grim_v2.md` describes `.grim` V2 as greenfield work. It isn't. `crates/grim-format`
already implements the format side non-breakingly on top of the existing `GRIM\x01`
wire format:

| `grim_v2.md` proposal | Actual state in `grim-format` |
| :--- | :--- |
| "New" header/registry/dual-stream layout (`grim_v2.md` implies a version bump) | Already shipping as `GRIM\x01` (`format.rs`); `grim_v2.md §1` is cited in its own doc comment as the spec source. Wire version stays `\x01` — see below |
| Outlier-Aware Streams (flat u32 index + f16 value) | Shipping, but the *legacy* path. `spec.rs` already has a denser `DeltaVarint` encoding (delta-varint indices + delta-u8 residuals) as the default going forward |
| Variable bitrates (EvoPress-style) | Shipping as `PerRowBpwMode::PerRowTable` + `bpw_table_offset`, with `Uniform` as the zero-cost default |
| GPTQ second-order correction | Shipping at two layers: `spec.rs::BackupLayer` (residual/backup stream format) on the format side, and `grim-backend-rocm/src/gptq_kernel.rs` (wavefront-parallel Fisher-diagonal correction kernel) on the compute side |
| Multi-format converter (safetensors/GGUF/AWQ/GPTQ) | Shipping (`gguf.rs`, `safetensors.rs`, `onnx.rs`, `tprov.rs`, `gptq.rs`, routed by `convert.rs`) |
| Wave64-aligned layout | Shipping pervasively — `WAVE64_SEGMENT_BYTES`, `LayoutHintTag::WavefrontTiled`, and on the kernel side `device/layout.rs::WavefrontTiledLayout` |

**So the format spec, the converter pipeline, and the GPTQ calibration-time
correction kernel are done.** Re-specifying them as new V2 work would mean
breaking the wire format to re-add things it already has, and regressing the
outlier encoding back to the flat/legacy scheme. None of that should happen.

**Wire version stays `GRIM\x01`, full stop.** No `.grim` file has actually
been converted yet — V1 is still being validated end-to-end, not proven out.
Bumping to `\x02` now would break compatibility to add capabilities the
`GrimTensorExt` JSON extension layer in `spec.rs` already provides
non-breakingly. Everything in this plan (WI-A through WI-G) reads the
existing V1 file layout and its extension struct; none of it requires or
implies a version bump. If V1 validation below (§2) turns up an actual
on-disk layout defect that the extension layer can't express, that's a
separate, explicit decision to be made later — not a default outcome of this
plan.

## 1. What is actually missing

I searched `grim-backend-rocm` for the inference-time consumer of the format's
quantized tensor layout (`dequant`, `outlier`, `vgpr`, `v_alignbyte`, packed-bpw
GEMM) and came up empty outside of comments. Concretely:

- **No kernel reads `GrimTensorExt` at all.** Nothing in `grim-backend-rocm`
  parses `row_scale_dtype`, `per_row_bpw_mode`, `bpw_table_offset`,
  `backup1`/`backup2`, or either outlier encoding. The format can describe a
  4-bit-with-outliers tensor perfectly; no kernel can run one.
- **`kernels/decode_gemm.rs` (`grim_decode_gemm_f16`) is plain dense F16 GEMM.**
  It takes `A`/`B` as `_Float16*` — there is no packed-int input path, no
  per-row dequant scale application, no outlier merge step. This is the
  kernel `grim_v2.md §4`'s "fused_dequant_gemm.rs" was supposed to be, and it
  isn't that yet.
- **`gptq_kernel.rs` is calibration-time only.** It corrects `weight_approx`
  against `weight_orig` using the Fisher diagonal — this runs once during
  conversion, on full f32 tensors. It has no relationship to what happens at
  token-generation time when a packed row has to be unpacked into VGPRs.
- **`device/gemm_tuning.rs`'s `split_k` is suggestion-only.** `roc_device.rs`
  hard-clamps `split_k_effective = 1` at every launch site with a
  `debug_assert_eq!`, and the code says so explicitly:
  `TODO(split-k-kernel): wire split_k_effective into a real SplitK
  reduction kernel`. `grim_v2.md` doesn't mention this at all, but it's a
  real, already-scaffolded gap directly relevant to decode-phase GEMM
  performance (WI 2.4.1 / 2.6.2 in the existing perf plan).
- **`device/layout.rs::WeightLayout` has no quantized/packed variant.**
  `RowMajor`, `WavefrontTiled`, `BlockSparse` are all dense-element layouts.
  There's no representation of "N bits per element, packed across u32 words,
  Wave64-segment-aligned" — the thing the format's `LayoutHintTag::WavefrontTiled`
  + `layout_descriptor` fields are meant to drive.
- **`quantization.rs` is an arch/dtype capability gate, not a dequant path.**
  It answers "can gfx1200 run fp8" for f32/f16/bf16/fp8 — it has nothing to do
  with 2–4 bit packed weights or outlier merging.

Everything else `grim_v2.md §4` asked for (VGPR on-the-fly unpack,
`v_alignbyte_b32`-style shifts, outlier merge before MAC, Wave64-coalesced
tile loads) is genuinely unbuilt. This is the one real, still-open piece of
work.

## 2. Prerequisite: prove V1 end-to-end before building on it

Since nothing has been converted through the pipeline yet, WI-C should not be
the first thing to exercise `grim-format`'s write path. Building a new HIP
kernel on top of an unvalidated file format means any bug found later is
ambiguous — is it the kernel, or the bytes it's reading? Do this first:

**What to build:** A real conversion run — one small safetensors model
(F16, a few small layers is enough) through `convert_to_grim`, producing an
actual `.grim` file on disk, then read back through `grim-format`'s reader
and dequantized on CPU (this doubles as the harness WI-B needs anyway).
Confirm the round-trip is bit-exact (or within f16 epsilon for anything that
touches quantization). This exercises the header, JSON metadata layer,
tensor registry, and both normals/outliers streams for real, not just via
the existing unit tests' synthetic buffers.

**Gate:** This blocks WI-C's correctness gate. It does not block WI-A
(pure layout math, no file I/O) or WI-D (unrelated to the format).

**Relevant Skills:**
- `writing-plans` (Writing plans)
- `writing-guidelines` (Writing guidelines)
- `specification-writing` (Specification-driven development)
- `requirements-clarity` (Specification requirements)

## 3. Work items

### WI-A — Packed-weight `WeightLayout` variant + host-side pack/unpack (format↔kernel bridge)

**Why:** Nothing on the kernel side can currently describe a packed 2–4 bit
tensor in device memory. This has to exist before any kernel can consume one.

**Where:** `grim-backend-rocm/src/device/layout.rs`

**What already exists:** `WeightLayout::WavefrontTiled { wavefront_size }` and
`WavefrontTiledLayout::{tile,untile}` operate on dense `f32` — reuse the tiling
math, not the element type assumption. `align_tensor_for_rocm_gemm` already
establishes the Wave64-row-padding convention to mirror.

**What to build:**
- `WeightLayout::PackedQuant { bits: u8, wavefront_size: u32 }` variant.
- A `PackedQuantLayout` struct mirroring `WavefrontTiledLayout`'s shape but
  operating on bit-packed `u32` words (2/3/4-bit lanes packed low-to-high),
  built directly from `grim_format::spec::GrimTensorExt` fields
  (`per_row_bpw_mode`, `default_bpw`/`bpw_table_offset`, `row_scale_dtype`,
  `scale_offset`) — this struct is the bridge type; `grim-format` stays the
  source of truth for the on-disk layout, this crate only reads it.
- Host-side `pack()`/`unpack()` for testing/CPU-fallback parity (needed for
  WI-C's correctness gate) — no GPU dependency, pure Rust, mirrors what
  `format_v2.rs`'s conversion-fidelity test already checks on the write side.

**Left/right limits:** This crate reads `GrimTensorExt`; it does not define
new on-disk fields or reach back into `grim-format` to change the spec. If a
kernel needs a field `spec.rs` doesn't have, that's a `grim-format` change
proposed separately, not smuggled in here.

**Gates:** Correctness (unpack∘pack = identity on random packed tensors,
including odd row counts that don't divide the wavefront size) → compile →
no changes to existing `WeightLayout` variants' behavior.

**Relevant Skills:**
- `rust-tdd` & `tdd` (Rust TDD / Test-Driven Development)
- `rust-architecture` (Architecture)
- `writing-guidelines` (Writing guidelines)
- `humanizer` (Humanizer)

---

### WI-B — Outlier merge + per-row dequant scale application (host-side reference kernel)

**Why:** Before writing HIP, get the numerics right on CPU where they're
debuggable. `grim-backend-cpu` already exists as the reference backend per
your architecture — this is where GPTQ/EvoPress accuracy claims get
verified against ground truth before trusting a GPU kernel's output.

**Where:** `grim-backend-cpu` (new module, e.g. `dequant_gemm.rs`) — *not*
`grim-backend-rocm*, per the existing "no other-backend reach-in" scope fence.

**What already exists:** `spec.rs`'s `decode_outliers_delta_varint` /
`encode_outliers_delta_varint` already do the outlier codec; this work item
consumes their output, it doesn't re-implement outlier decoding.

**What to build:** A reference `dequant_row(packed_bits, scale, backup_layer,
outliers) -> Vec<f32>` that: unpacks normals at `default_bpw`/per-row bpw,
applies the u8 row scale, adds the backup/residual layer if present
(`BackupLayer::is_present`), then applies outlier overrides at their exact
indices. This is the numerical spec the HIP kernel in WI-C must match.

**Gates:** Correctness only — this is a test fixture, not a perf path. Assert
output matches a float64 reference dequant within tight epsilon on synthetic
tensors covering: uniform bpw, per-row bpw table, one backup layer, two
backup layers, delta-varint outliers, flat-u32 legacy outliers.

**Relevant Skills:**
- `rust-tdd` & `tdd` (Rust TDD / Test-Driven Development)
- `rust-architecture` (Architecture)
- `writing-guidelines` (Writing guidelines)
- `humanizer` (Humanizer)

---

### WI-C — `fused_dequant_gemm.rs`: the actual missing kernel

**Why:** This is the one `grim_v2.md §4` item with zero existing
implementation. It's also the one with the largest performance payoff (avoids
materializing dequantized weights in VRAM at all — the whole point of the
outlier-aware format).

**Where:** `grim-backend-rocm/src/kernels/fused_dequant_gemm.rs` (new file,
registered in `kernels/mod.rs` and `kernels::source_asm::compute_kernel_source()`
next to `decode_gemm`, following the exact pattern `decode_gemm.rs` already
established for JIT-source concatenation).

**What already exists to build on, and must not be reimplemented:**
- The `DecodeGemmConfig::enabled`-gated dispatch pattern in `roc_device.rs`
  (search `QkvAttentionFusionConfig::enabled` — same gate style is mandatory
  here; **default off**, exactly like `decode_gemm`'s own doc comment
  insists, until a real benchmark justifies flipping it).
- `device/gemm_tuning.rs::lookup_gemm_config` for tile shape and the
  bank-conflict pad (WI 2.4.4-3) — the new kernel's tiling should call into
  this, not duplicate tile-size logic.
- Wave64 mandate constants (`WAVE64_SEGMENT_BYTES` in `grim-format`,
  `device/util.rs`'s 256-thread/4-wavefront default) — reuse, don't
  re-derive.
- WI-A's `PackedQuantLayout` for the input tensor shape/stride math.
- WI-B's reference dequant as the correctness oracle.

**What to build:**
1. HIP kernel `grim_fused_dequant_gemm_f16`: each thread loads packed bits for
   its output element(s) into registers, unpacks via shift/mask (start with
   portable `<<`/`&`/`>>` HIP C — do **not** hand-write
   `v_alignbyte_b32` inline asm on the first pass; that's an optimization
   layer once the portable version is correctness-gated and benchmarked, per
   your own gate-ordering discipline: correctness → compile → architecture →
   perf, in that order).
2. Row-scale application per WI-B's numerical spec.
3. Outlier merge: a second small buffer of (index, value) pairs read
   concurrently and added post-dequant, pre-MAC — matches `grim_v2.md §4`'s
   description, now actually implemented.
4. Rust launcher in `RocmDevice`, gated behind a new `FusedDequantGemmConfig`
   struct in `fusion.rs` mirroring `DecodeGemmConfig` exactly (same
   `enabled: bool` default-off field, same doc-comment style).
5. `TODO(gpu-verify)` annotation on any throughput claim, per your existing
   convention — no asserted speedup numbers until measured on real gfx1036/
   gfx110x/gfx1200 hardware.

**Left/right limits:**
- Does not touch `qkv_attention.rs` or the QKV fusion path — this is a GEMM
  kernel for FFN/projection weights, not attention.
- Does not attempt SplitK (WI-D below) in the same change — keep the two
  perf axes separable and independently benchmarkable.
- Does not add a new outlier encoding — consumes whichever encoding
  `GrimTensorExt::outlier_index_encoding` says is on disk; both `FlatU32` and
  `DeltaVarint` must be supported since old files use the former.

**Gates:**
1. Correctness: kernel output matches WI-B's CPU reference within f16
   epsilon, for every case WI-B's test matrix covers, plus a real
   sub-3-bit-average mixed-bitwidth tensor (the actual target case per
   `grim_v2.md`'s claimed "sub-3-bit averages" benefit).
2. Compile: `cargo build` clean, `cargo check` on the synthetic workspace
   (same verification style already used for the `rocblas_gemm_ex` fix).
3. Architecture cleanliness: no reach-into `grim-scheduler`/`grim-memory`
   from this kernel file; config-gated off by default.
4. Non-blocking perf: `TODO(gpu-verify)` micro-bench vs. (a) dense
   `decode_gemm_f16` on an equivalent-accuracy dense tensor, and (b) rocBLAS
   dequant-then-GEMM (current de facto path — dequant to a scratch buffer,
   then plain GEMM) to prove the *fusion* itself is worth the added kernel
   complexity, not just that quantization saves memory.

**Relevant Skills:**
- `rocm-hip-kernels` & `rocm-kernel-design` (Rust FFI ROCm / GPU Kernels)
- `rust-ffi-grim` & `rust-ffi` (Rust FFI ROCm)
- `system-design` (Architecture)
- `writing-guidelines` (Writing guidelines)
- `humanizer` (Humanizer)

---

### WI-D — Wire up `split_k` for real (separate from WI-C)

**Why:** Already scaffolded and already flagged in-repo as blocking
(`debug_assert_eq!(split_k_effective, 1)` + explicit `TODO(split-k-kernel)`).
Not in `grim_v2.md` at all, but it's real, cheap, decode-path-relevant work
sitting right next to what this doc is about.

**Where:** `grim-backend-rocm/src/device/roc_device.rs` (the clamp site) +
`device/gemm_tuning.rs` (the suggestion source, already correct).

**What already exists:** The suggestion logic (`split_k = 2` when
`k >= 4096` in decode shape) is implemented and tested
(`f2_split_k_suggested_at_lookup_only_for_kheavy_decode`). Only the
reduction kernel and the launch-site wiring are missing.

**What to build:** A small reduction kernel that sums `split_k` partial-C
buffers, plus removing the hard clamp once that kernel exists and is gated
behind its own default-off config flag.

**Gates:** Same four-gate ordering as WI-C. Keep this independent of WI-C so
a regression in one doesn't block the other's rollout.

**Relevant Skills:**
- `rocm-hip-kernels` & `rocm-kernel-design` (Rust FFI ROCm / GPU Kernels)
- `writing-guidelines` (Writing guidelines)
- `humanizer` (Humanizer)

### WI-C follow-up — capture the fused dequant GEMM into the decode graph

**Why:** `graph_capture.rs`'s `GraphCaptureManager` already exists and is the
source of a measured 15–30µs/token gain by replaying a cached
`hipStreamBeginCapture`/`EndCapture` graph instead of re-issuing individual
kernel launches on the decode hot path. If WI-C's fused dequant GEMM kernel
gets launched *outside* that captured graph — as its own standalone
`hipModuleLaunchKernel` call — the decode step gets a slower kernel to call
into *and* forfeits graph-replay's launch-overhead savings for every step
that touches a quantized weight. This is easy to get wrong by accident:
`FusedDequantGemmConfig::enabled` gates whether the kernel runs at all, but
says nothing about whether it's captured.

**Where:** `grim-backend-rocm/src/graph_capture.rs` (the capture/replay
boundary), `grim-backend-rocm/src/device/roc_device.rs` (the decode-step
call site that decides what's inside vs. outside the captured closure).

**What already exists:** `GraphCaptureManager` caches per shape-key
(`DecodegKey`) and invalidates via `hipGraphExecUpdate` on weight/shape
change — this machinery doesn't need to change. `decode_gemm.rs`'s dense F16
kernel is presumably already inside the captured decode step (verify this as
part of this work item, don't assume it); the fused dequant kernel should
follow the identical inclusion pattern once WI-C lands.

**What to build:** Nothing new architecturally — this is a wiring/ordering
fix, not a new kernel. Confirm (with a test, not just a read-through) that
when `FusedDequantGemmConfig::enabled` is true, the kernel launch happens
*inside* the same `hipStreamBeginCapture`/`EndCapture` region as QKV
attention and the existing decode GEMM, so a single graph replay covers the
full quantized decode step. Add a `DecodegKey` variant (or extend the
existing one) that distinguishes "decode step includes fused dequant GEMM"
from "decode step is all-dense," so the cache doesn't serve a stale graph
across a config flip.

**Gates:** Correctness (graph replay output matches un-captured sequential
launch output, same epsilon as WI-C's own gate) → compile → architecture
(no change to `GraphCaptureManager`'s public surface unless the
`DecodegKey` extension requires it) → `TODO(gpu-verify)` perf: measure the
15–30µs/token figure still holds with the dequant kernel included, not just
assume it carries over from the dense-only baseline.

**Relevant Skills:**
- `rocm-hip` (Rust FFI ROCm)
- `system-design` (Architecture)
- `writing-guidelines` (Writing guidelines)

---

### WI-E — Wire the speculative-decoding CPU orchestrator to the real GPU tree-attention verifier

**Correction to prior guidance:** I previously said the tree-attention GPU
kernel didn't exist yet ("next PR," per `speculative.rs`'s module doc
comment). That comment is stale. The kernel is built: `grim_tree_attention`
in `kernels/qkv_attention.rs`, launched via `launch_tree_attention`, exposed
as `RocmDevice::tree_attention` (`device/roc_device.rs`), with its own
passing test (`tests/tree_attention.rs`, GPU-gated via
`GRIM_RUN_GPU_TESTS`) plus a RED→GREEN device-wiring test confirming
`RocmDevice::tree_attention` delegates to the launcher.

**Why:** Speculative decoding's entire performance case is amortizing
per-token latency across `gamma` draft candidates verified in one target
forward pass. Right now the two halves of that system are each real and
each independently tested, but disconnected:
- The GPU verifier (`RocmDevice::tree_attention`) takes real tensors and a
  `tree_parents` array and returns real attention output.
- The CPU orchestrator (`SpeculativeDecoder<D, T, P>`, `TokenAcceptor`,
  `TreeMaskBuilder`) takes a generic `target: Fn(&[u32], &[u32]) ->
  Vec<(f32, f32)>` closure and has never been constructed with anything but
  a synthetic test closure — grep `tests/speculative.rs` and every call site
  passes a hand-written `target_score`/`target` fn, never
  `RocmDevice::tree_attention`.

No code path in this crate currently runs an actual speculative-decode step
end-to-end on GPU. The format and orchestration are both done; the adapter
between them is the missing piece.

**Where:** New adapter, likely `grim-backend-rocm/src/speculative_gpu.rs` (or
directly in `speculative.rs` if keeping it colocated is preferred — decide
based on whether `speculative.rs` stays backend-agnostic per its own stated
design goal of orchestration primitives living apart from backend kernels;
if that separation is intentional, the adapter belongs in a new file, not
inside `speculative.rs` itself).

**What already exists to build on, and must not be reimplemented:**
- `TreeMaskBuilder` already produces the ancestor bitmask; `tree_attention`'s
  HIP kernel already consumes a `tree_parents` array via `is_ancestor()` —
  confirm these two representations line up (bitmask vs. parent-pointer
  array) and write the (probably small) conversion between them rather than
  changing either existing primitive.
- `TokenAcceptor::decide` already implements the Leviathan-style accept/
  reject rule against `(p_draft, p_target)` pairs — the adapter's job is
  producing those pairs from `tree_attention`'s real output logits, not
  reimplementing acceptance sampling.
- `SpeculativeDecoder::new(gamma, draft, target, pickup)`'s generic-closure
  design is the extension point — the adapter is *a* `target` closure
  implementation backed by `RocmDevice::tree_attention`, not a replacement
  for `SpeculativeDecoder`.

**What to build:**
1. A `target` closure (or small struct implementing the same `Fn(&[u32],
   &[u32]) -> Vec<(f32, f32)>` shape) that: builds the tree mask via
   `TreeMaskBuilder` from the draft token tree, calls
   `RocmDevice::tree_attention` with it, runs the result through the model's
   output projection + softmax to get `(p_draft, p_target)` pairs per
   position.
2. Wire this into one real construction of `SpeculativeDecoder` in a
   non-test code path (currently every construction is inside `tests/`).
3. Confirm `entropy_confidence_head`/`confidence_scheduler` (per project
   memory, these live in `grim-speculative`, outside this crate) can gate
   *when* this adapter's tree-attention verifier runs vs. when a cheaper
   single-token path suffices — this work item only builds the wiring the
   confidence scheduler needs to call into; it does not re-implement the
   confidence-gating logic itself, which is out of scope per crate
   boundaries (`grim-backend-rocm` doesn't reach into `grim-speculative`).

**Left/right limits:** Does not modify `grim_tree_attention`'s kernel body
or `TokenAcceptor`'s acceptance math — both are correct and tested as-is.
Does not implement the draft model's own forward pass acceleration (assumed
to already run through the standard `qkv_attention`/`decode_gemm` path as
any other small model would). Does not reach into `grim-speculative`'s
confidence-scheduler crate; this item produces the callable surface that
crate needs, nothing more.

**Gates:**
1. Correctness: adapter-produced `(p_draft, p_target)` pairs, fed through
   `TokenAcceptor::decide`, match a reference CPU-only speculative-decode
   step (same draft tree, same weights, dense non-quantized case first) —
   the existing `tests/speculative.rs` synthetic-closure tests become the
   oracle to diff against once the real closure is substituted.
2. Compile.
3. Architecture: adapter lives outside `speculative.rs`'s backend-agnostic
   core if that separation is real (verify, don't assume); no new coupling
   from `grim-backend-cpu` or `grim-scheduler` into this adapter.
4. `TODO(gpu-verify)` perf: measure realized tokens/forward-pass speedup on
   real hardware — this is the number the whole feature exists to produce,
   and it's currently unmeasurable because there's nothing to measure yet.

**Relevant Skills:**
- `rust-tdd` & `tdd` (Rust TDD / Test-Driven Development)
- `system-design` (Architecture)
- `writing-guidelines` (Writing guidelines)
- `humanizer` (Humanizer)

---

### WI-F — Remove the `head_dim > 64` guard: port cubecl's existing tiled attention into the primary kernel path

**Correction to prior guidance:** I originally scoped this as writing new
HIP C++ LDS-tiling code from scratch. That was wrong — a working
implementation already exists. `device/cubecl.rs` (behind the `cubecl`
feature flag) has a pure-Rust attention kernel that already handles
`head_dim` up to 256 via exactly the tiling approach this item needs:
`chunks = (head_dim + 63) / 64`, each lane owns one dimension per chunk,
`plane_sum` reduces within the wavefront across chunks. Its own doc comment
says it's *"proven correct against a CPU reference on gfx1036"* and covers
all three attention variants the HIP path has (`qkv_attention_kernel`,
`paged_attention_kernel`, `tree_attention_kernel`). The head_dim=128 problem
is already solved once, in this module. It's unused only because — per its
own doc comment — it's *"not yet wired into `RocmDevice`'s tensor dispatch."*

**Why it isn't simply "flip a switch and ship it":** project memory
already flags why this code stays behind a feature flag: *"intermittent GPU
test flakiness (~1 in 5 runs) is attributed to a known cubecl-hip 0.10
first-dispatch limitation."* `cubecl.rs` mitigates this with
`warmup_every_kernel`/`warmup_verify`, but that's a workaround for an
upstream dependency issue, not a fix. Wiring the *default* head_dim=128
attention path — something models like Llama-3 need for basic
functionality — through a dependency with a known ~20% first-dispatch fault
rate is a worse tradeoff than the current hard `Err`, which at least fails
loudly and predictably instead of intermittently.

**Where:** `grim-backend-rocm/src/kernels/qkv_attention.rs` (the primary,
non-feature-gated HIP source), informed by `device/cubecl.rs`'s existing
`qkv_attention_kernel`/`tree_attention_kernel` as a reference implementation
and correctness oracle — not `device/cubecl.rs` itself as the shipped path.

**What already exists:** Everything needed to validate correctness:
`tests/qkv_head_dim_128.rs`'s CPU reference (the original oracle), *and*
now `cubecl.rs`'s GPU implementation as a second, independent oracle already
proven on real gfx1036 hardware. Three-way agreement (CPU reference ↔
cubecl GPU output ↔ new HIP kernel output) is a stronger correctness bar
than the original plan had.

**What to build:**
1. Port `cubecl.rs::qkv_attention_kernel`'s chunking technique — literally
   the same `chunks = (head_dim + 63) / 64` loop structure, same
   neutral-zero padding for lanes beyond `head_dim`, same wave-level
   reduction shape — into `qkv_attention.rs`'s HIP C++ source. This is a
   translation task (Rust/cubecl DSL → HIP C++), not new kernel design; the
   algorithm is already validated.
2. Apply the same port to `grim_tree_attention` (WI-E depends on this kernel
   too — the speculative-decoding verifier inherits the same head_dim limit
   otherwise).
3. Remove the `head_dim > 64` guard and the `Err` in `roc_device.rs` once
   the ported kernel passes the three-way correctness check.
4. Leave `device/cubecl.rs` as-is — don't delete it. It remains a useful
   independent-implementation cross-check and a candidate full backend if
   cubecl-hip's first-dispatch issue gets fixed upstream. This item doesn't
   touch the `cubecl` feature or its warmup machinery.

**Left/right limits:** Same as before — doesn't add MFMA/WMMA (that's
WI-G below, RDNA3/4-only in scope); doesn't touch the `head_dim <= 64`
path; fixes `grim_qkv_attention` and `grim_tree_attention` together since
they share the guard.

**Gates:** Correctness — three-way match (CPU reference, cubecl GPU output,
new HIP kernel output) at atol=1e-4, per the existing test's own tolerance
→ compile → architecture (no dependency on the `cubecl` feature flag from
the default build path) → `TODO(gpu-verify)` perf: confirm head_dim=64
throughput doesn't regress from the generalized chunk-loop replacing the
single-element path, on RDNA2, RDNA3, and RDNA4 separately (per the
occupancy-tradeoff caution already noted for this item — don't assume one
arch's numbers carry over).

**Relevant Skills:**
- `rocm-hip-kernels` & `rocm-kernel-design` (Rust FFI ROCm / GPU Kernels)
- `rust-architecture` (Architecture)
- `writing-guidelines` (Writing guidelines)
- `humanizer` (Humanizer)

---

### WI-G — WMMA matrix-core dispatch for RDNA3/4 GEMM kernels

**Why:** Every real GEMM-shaped kernel in this crate (`decode_gemm.rs`'s
`grim_decode_gemm_f16`, the QKV projection matmuls, and WI-C's planned
fused dequant GEMM) is a scalar per-thread dot-product loop on VGPRs. RDNA3
and RDNA4 (GFX11+) have WMMA matrix-core hardware that these kernels never
touch. `accel_features.rs` already documents this precisely: *"RDNA (gfx10/
11/12) has no MFMA — it uses WMMA/rocWMMA (GFX11+)"* and *"the `ck_tile`
GEMM path is valid on both RDNA (Wave32 WMMA pipeline) and CDNA (MFMA
pipeline) — the kernel wrapper selects the pipeline via
`-DCK_TILE_USE_WMMA` at compile time."* The capability classification is
real and already correct; nothing dispatches to it. This is scalar-loop
GEMM leaving matrix-core throughput on the table on exactly the consumer
hardware this project targets.

**Scope note — WMMA only, not MFMA:** MFMA is CDNA-exclusive (`gfx908`/
`gfx90a`/`gfx940-942`) and is a different intrinsic family running through
a different capability path (`mfma_supported`, already gated correctly in
`accel_features.rs`). Bundling both into one work item risks conflating two
different hardware targets and two different intrinsic ABIs. This item is
scoped to **WMMA on RDNA3/RDNA4 (GFX11+) only** — `gfx110x` and `gfx1200`.
A CDNA3/MFMA equivalent, if wanted, should be its own separate item so its
gates and hardware validation don't get entangled with this one's.

**Also out of scope: RDNA2.** `ck_supported()` currently returns `true` for
`GcnArch::RDNA2`, but the WMMA hardware itself is GFX11+ only — RDNA2 is
GFX10. That `ck_supported` inclusion appears to be about CK library
buildability in general (CK can target RDNA2 via its non-WMMA wave32 path),
not WMMA hardware presence. This item should not extend WMMA dispatch to
RDNA2; if `ck_supported`'s RDNA2 inclusion needs tightening to avoid future
confusion, that's a small separate fix, not part of this item's build.

**Where:** New kernel variant(s) alongside `kernels/decode_gemm.rs`
(matching its existing JIT-source-concatenation pattern via
`kernels::source_asm::compute_kernel_source()`), dispatch wiring in
`device/roc_device.rs`, capability check via the already-correct
`accel_features.rs` (no changes needed there — it's the precondition this
item finally consumes).

**What already exists to build on:** The `DecodeGemmConfig::enabled`-style
default-off gate pattern (reuse exactly, per every other kernel in this
plan); `gemm_tuning.rs::lookup_gemm_config`'s tile-shape logic (WMMA tiles
still need to respect the existing bank-conflict-pad and split_k
heuristics, not bypass them); `ck_supported()`/`ck_dispatch()` as the arch
gate, already correctly scoped to include RDNA3/RDNA4.

**What to build:**
1. A WMMA-based GEMM kernel variant for decode-shaped and prefill-shaped
   matmuls, using the compiler's `__builtin_amdgcn_wmma_*` intrinsics (or
   HIP's `rocwmma` header if available in the build environment — verify
   presence first per `rust-ffi`'s "hand-write only what resolves" policy
   already applied elsewhere in this crate, e.g. `accel_ffi.rs`'s MIOpen
   dlopen fallback for exactly this kind of "library might not be present"
   situation).
2. Dispatch gate: `RocmDevice::matmul` checks `ck_supported(arch) &&
   matches!(arch, GcnArch::RDNA3 | GcnArch::RDNA4)` (or an equivalent
   direct WMMA capability check if one gets added to `accel_features.rs`
   rather than reusing the broader CK gate) before offering the WMMA path;
   falls back to the existing scalar kernel everywhere else, silently and
   correctly — no behavior change off-target.
3. Config-gated default-off, exactly like `DecodeGemmConfig` and every
   other kernel in this plan.

**Left/right limits:** RDNA3/RDNA4 only, as scoped above. Does not touch
CDNA/MFMA. Does not touch the attention kernels (WI-F's scope) — this is
GEMM only. Does not change `accel_features.rs`'s existing capability
classification, which is already correct; this item is purely the
dispatch consumer of it.

**Gates:** Correctness (WMMA output matches the existing scalar kernel's
output within f16 epsilon, across the same shape matrix WI-C/WI-D use) →
compile (including a build-environment check for whether `rocwmma` headers
or the compiler intrinsics are actually available — don't assume, verify,
per the project's standing rule against fabricated ABIs) → architecture
(config-gated off by default; no silent kernel swap) → `TODO(gpu-verify)`
perf: this is the one gate where a real number matters most in this whole
plan — matrix-core vs. scalar-loop throughput is the kind of claim that
needs an actual measured ratio on real RDNA3 and RDNA4 hardware before
anyone treats it as delivered, not an assumed multiplier from the general
GPU literature.

**Relevant Skills:**
- `rocm-hip-kernels` & `rocm-kernel-design` (Rust FFI ROCm / GPU Kernels)
- `rust-ffi-grim` & `rust-ffi` (Rust FFI ROCm)
- `system-design` (Architecture)
- `writing-guidelines` (Writing guidelines)
- `humanizer` (Humanizer)

---

## 4. Sequencing

§2 (real end-to-end V1 conversion) should happen first, or at least
in parallel with WI-A — it's cheap, has no GPU dependency, and de-risks
everything downstream. WI-A and WI-B have no GPU dependency either and can
be done in parallel; both, plus §2, are prerequisites for WI-C's
correctness gate. The WI-C graph-capture follow-up depends on WI-C landing
first (nothing to capture into a graph until the kernel exists). WI-D, WI-E,
WI-F, and WI-G are all independent of WI-A/B/C and of each other. WI-E in
particular has zero dependency on the dequant/format work, since it's
purely about connecting two already-built pieces (the tree-attention kernel
and the CPU orchestrator). WI-F is likewise fully decoupled — it's a
self-contained port of an already-proven algorithm into the primary kernel
path, with its own pre-existing test oracles (plural, now — CPU reference
and cubecl GPU output), and doesn't touch anything WI-A through WI-D
produce, though WI-E's verifier kernel shares WI-F's head_dim fix and
should land after it, not before. WI-G is independent of everything else in
this plan but is the largest single item here — its `TODO(gpu-verify)` perf
gate is also the one most worth prioritizing a real hardware measurement
for, since the entire point of the item is a throughput claim. Recommended
order: **§2 + WI-A → WI-B → WI-C → WI-C-follow-up**, with WI-D, WI-F, WI-E
(in that order, since WI-E depends on WI-F), and WI-G done whenever
convenient relative to the format-side chain.

## 5. Verification plan

- `cargo check` across the workspace after each WI, mirroring the existing
  synthetic-workspace verification already used for the `rocblas_gemm_ex` fix.
- WI-C's correctness gate blocks its own perf gate — no `TODO(gpu-verify)`
  speedup claims land until the dequant output is bit-for-bit-within-epsilon
  matched against WI-B's CPU reference.
- No default-behavior changes anywhere: `FusedDequantGemmConfig::enabled` and
  the eventual split-k flag both default to `false`, following the existing
  `QkvAttentionFusionConfig`/`DecodeGemmConfig` convention exactly.
