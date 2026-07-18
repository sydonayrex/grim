# `.grim` File Format Specification (V3)

**Date**: 2026-07-16
**Status**: Draft
**Author**: grim maintainers
**Supersedes**: `grim_v2.md` §1 (the half-shipped layout)
**Affects**: `crates/grim-format/src/{format.rs,gguf.rs,tprov.rs,convert.rs}`, `crates/grim-quant`, `crates/grim-backend-rocm/src/kernels/fused_dequant_gemm.rs`

---

## Overview

V3 is the file layout that actually delivers what `grim_v2.md` promised plus the residual/mixed-precision machinery EXL2 uses to beat bitsandbytes/AWQ on the bpw-vs-perplexity curve. Two new streams on disk, one new descriptor per tensor, no breaking magic change (`GRIM\x03`, readers fall back to `\x01`/`\x02`).

The format is a pure on-disk layout spec. Calibration (EvoPress) and reconstruction (GPTQ) are off-file concerns owned by `grim-quant`; this doc only defines the slots they fill in.

## Motivation

### Current State (`GrimTensorEntry`, `format.rs:78`)

```rust
pub struct GrimTensorEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub base_bitwidth: u8,        // ONE bitwidth per whole tensor
    pub payload_offset: u64,     // normals only
    pub payload_size: u64,
    pub outlier_count: u32,
    pub outlier_offset: u64,     // u32 index | f16 value, 6 bytes/outlier, flat
}
```

`normals_packed_size()` is `bits = (N - outlier_count) * base_bitwidth`, aligned to 256B. No scale, no zero, no block concept, no per-row anything.

### Problems

1. **Single bitwidth per tensor.** Closer to bitsandbytes NF4 than EXL2. Sensitive layer and boring FFN layer get the same `bpw` — you either lose accuracy on the sensitive layer or waste size on the boring one. `grim_v2.md` promised "Variable Bitrates (EvoPress)"; not delivered.
2. **No per-row / per-block scale.** Every K-quant competitor (GGUF Q4_K, AWQ, EXL2) renormalizes a 32- or 128-wide block against a scale. Without one, your 4-bit grid has 16 levels and no notion of the block's range. Outliers get manufactured by the quantizer that would disappear with proper scaling — outlier counts are inflated.
3. **Outliers are 6 bytes flat with f16 values.** `u32 index | f16 value`, no grouping, no residual encoding. A 7B at 4bpw / 1% outliers is ~9 MB of outlier stream. EXL2's residual backup is ~2 bytes per correction after delta + 8-bit quant.
4. **No GPTQ-ORDERED flag.** EXL2's two-level residual requires the weights to be GPTQ-inverse-ordered before quantization. The format carries no signal that this happened, so the kernel cannot apply the inverse at dequant time.
5. **`importance_score` and `layout_hint` are dead freight.** Stored on `GrimQuantOverride` (`gguf.rs:300`), referenced nowhere in dequant. The whole "mixed-precision via known salient columns" path is unimplemented.

### Desired State

A tensor carries enough metadata to answer, at load time without running calibration:

- "This row is 4-bit, that row is 6-bit, averaged to 4.3 bpw at this block size."
- "Outliers are residual quantized at 8-bit, GPTQ-ordered, delta-encoded indices."
- "Use the fused dequant-matmul kernel on this one, dimensions match Wave64 tiling."

The converter (`grim-quant`, not in scope here) decides bitwidths and residual structure. **This spec only defines the byte layout those decisions land in.**

## Research Findings

### Per-tensor bitdepth / scale granularity across formats

| Format  | Bitwidth granularity | Scale granularity       | Residual level | Ordering flag | Kernel alignment |
|---------|---------------------|-------------------------|----------------|---------------|-------------------|
| bitsandbytes NF4 | whole model          | per-row 32-elt block    | 0              | —             | CUDA warp (32B)   |
| AWQ            | whole model          | per-group 128-elt       | FP16 gate col  | —             | CUDA warp         |
| GGUF Q4_K      | fixed 4-bit          | per-superblock (256)    | 0              | —             | 32B               |
| GGUF Q6_K      | fixed 6-bit          | per-superblock           | 0              | —             | 32B               |
| EXL2           | **per-layer mix {2,3,4,6,8}** | **per-row 8-bit** | **1 (8-bit backup)** | **yes** | row + warp        |
| `.grim` V2 (shipped) | per-tensor          | none                    | flat f16 outliers | no            | **Wave64 (256B)** |
| `.grim` V3 (this) | **per-row mix {2..8} (EXL2-like)** | **per-row u8** | **2 (residual: u8 delta + optional vector backup)** | **yes** | **Wave64 (256B)** |

**Key finding 1**: Per-row 8-bit scales match FP16 dynamic range with negligible accuracy loss (EXL2 §3.2). Per-32 6-bit scales (GGUF Q4_K) are purely a CUDA-occupancy choice. The 8-bit per-row choice is **strictly better at the same byte cost** once block size is ≥ 128, and draws cleaner kernels on ROCm (one scale per lane in Wave64 row-tile).

**Key finding 2**: Two-level residual (codes → 8-bit backup → optional second backup) is what puts EXL2 below AWQ/NF4 on the perplexity curve at fixed bpw. A single residual pass (`backup_bpw=8`) is the 80/20.

**Key finding 3**: Outliers compress by 3–5× with delta-encoded varint indices (sorted outlier positions cluster in contiguous column runs), plus another 2× when the residual value is quantized to the block's peripheral precision rather than always-f16.

### Wave64 layout

`WAVE64_SEGMENT_BYTES = 256` (`format.rs:18`) is already the right number. One wavefront = 64 lanes * 4 bytes = 256B per coalesced load. **No change** to the constant. What V3 changes is what lives inside one segment: codes + their per-row scale, packed so the kernel issues one seg-load + one broadcast load and emits dequantized values.

## Design Decisions

| # | Decision | Class | Choice | Rationale |
|---|----------|-------|--------|-----------|
| D1 | Magic version byte | 1 evidence | `GRIM\x03`; readers MUST accept `\x01`/`\x02` | Back-compat verified by `format.rs:51` magic check |
| D2 | Scale granularity | 1 evidence | per-row u8 | EXL2 §3.2; 1 byte cost per row, covers FP16 dynamic range |
| D3 | Bitwidth granularity | 2 coherence | per-row `bpw: u8 ∈ {2..8}` mixed within the tensor | EXL2 per-layer is coarse; per-row is finer and what EvoPress Hessian produces at row resolution |
| D4 | Residual layers | 3 taste | Up to **2** residual streams (`backup_bpw` u8 each, 0=none) | EXL2 shows 1 backup is the win; 2 caps complexity. >2 is churn. Revisit when accuracy data says otherwise. |
| D5 | Residual encoding | 2 coherence | Additive delta from prior level, quantized to `backup_bpw`, signed | `w ≈ quant_low(code_i) + Σ backup_add[level]`; lets backup compress |
| D6 | Outlier index encoding | 3 taste | Delta-varint over sorted indices | 3–5× smaller than flat u32; sorted is cheap to enforce at write time |
| D7 | Outlier residual encoding | 2 coherence | Same as residual streams — delta from reconstructed value, not raw value | Lets outlier `value` drop from f16 to u8 in most cases (B2) |
| D8 | GPTQ-ORDERED flag | 1 evidence | u8 bitfield on entry | EXL2 kernel dispatches inverse-Hessian only when this is set; match its ABI |
| D9 | GPTQ inverse storage | 3 taste | Not on disk. Carried in metadata JSON as `gptq_h_order_ref` | The Hessian inverse is per-tensor heavy (~row² doubles); store only a hash/reference, kernel reads from `grim-quant` cache at load. **Keep off-file.** |
| D10 | Keep outliers as a separate stream | 2 coherence | Yes | Reuse V2's `outlier_offset`/`outlier_count`; only the per-record encoding changes |
| D11 | Keep Wave64 segment alignment | 1 evidence | 256B, unchanged | `format.rs:18`, kernel invariant |
| D12 | Block size | 1 evidence | `block_size: u16 ∈ {0, 32, 64, 128, 256}` (0 = "single block, per-row scale mode") | EXL2 row mode = block_size 0; GGUF K-quant compat near 256. Field exists so writers can pick; readers dispatch on it |
| D13 | Compression on payload | 3 taste | Optional `compression: u8 ∈ {0=raw, 1=zstd}` per tensor | AWQ/GGUF ship uncompressed; this is free size with no accuracy cost. zstd dep already in transitively available scope. Default off for V3. |
| D14 | Don't move scales to a separate stream | 2 coherence | Interleave per-row scale at row-group boundary inside normals | One coalesced load = code + scale; matches Q4_K's layout trick |
| D15 | Don't add a per-row fp16 zero-point | 3 taste | Symmetric quantization only | One byte saved per row over FP16 zp; EXL2 is symmetric too. Revisit if accuracy data forces asymmetric. |

---

## Architecture

### File layout (V3)

```
+------------------------------------------------------------------------+
| Header (16 B)                                                          |
|   magic: [u8;5]   = "GRIM\x03"                                         |
|   format_version: u8       = 3                                         |
|   metadata_len:   u64                                                  |
|   num_tensors:     u32                                                 |
+------------------------------------------------------------------------+
| Metadata JSON Layer (metadata_len bytes)                               |
|   architecture, target_gcn, wavefront_size, profile, quant_version,    |
|   block_size (default), compression_default, gptq_h_order_ref,        |
|   rocm_fusion_ops, kv_layout_optimized, ...                            |
+------------------------------------------------------------------------+
| Tensor Registry (num_tensors entries)                                 |
|   See GrimTensorEntry V3 below                                         |
+------------------------------------------------------------------------+
| Payload Region                                                         |
|   For each tensor, in registry order, Wave64-aligned (256 B):          |
|   [ Normals Stream: codes + interleaved per-row u8 scales ]            |
|   [ Backup Level 1 stream — if backup_bpw_1 > 0 ]                      |
|   [ Backup Level 2 stream — if backup_bpw_2 > 0 ]                      |
|   [ Outliers Stream: delta-varint indices + delta-u8/f16 values ]      |
+------------------------------------------------------------------------+
```

Header is still 17 bytes total; the extra `format_version` byte is the only difference from `\x01`/`\x02` readers apart from the magic terminator byte (`\x03` instead of `\x01`). Old readers fail the magic check at the byte level, which is the desired hard-fail behavior.

### GrimTensorEntry V3 (on-disk binary)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ GrimTensorEntry                                                        │
├──────────────────────┬─────────┬────────────────────────────────────────┤
│ field                │ type    │ notes                                  │
├──────────────────────┼─────────┼────────────────────────────────────────┤
│ name                 │ str     │ u16 len + UTF-8 (unchanged)            │
│ shape                │ u8 ndim │ + u32 dims (unchanged)                 │
│ row_count            │ u64     │ Π(shape[:-1]); 1 if 1-D                │
│ row_stride           │ u64     │ shape[-1]                              │
│ block_size           │ u16     │ 0 = per-row scale (EXL2 mode),         │
│                      │         │ else 32/64/128/256                     │
│ per_row_bpw_mode     │ u8      │ 0 = uniform at `default_bpw`           │
│                      │         │ 1 = per-row table at `bpw_table_off`  │
│ own_bpw_table        │ u8      │ 0/1 — table inline or shared in meta   │
│ default_bpw          │ u8      │ bits, 2..8                             │
│ row_scale_dtype      │ u8      │ 0 = u8, 1 = f16 (asymmetric fallback)  │
│ gptq_ordered         │ u8      │ 0/1                                    │
│ outlier_residual_bpw │ u8      │ 8 or 16; 0 = no residual encoding      │
│ fusion_mask          │ u8      │ bit0 = RmsNormMatMul, bit1 = Qkv       │
│ layout_hint          │ u8      │ 0=default, 1=WavefrontTiled, 2=BlockSp │
│ layout_descriptor    │ [u32;4] │ kernel dispatch hint, opaque to reader │
│ compression          │ u8      │ 0=raw, 1=zstd, 2=lz4 (TBD)             │
│ bpw_table_offset     │ u64     │ 0 if per_row_bpw_mode = 0              │
│ bpw_table_count      │ u32     │ 0 = use default_bpw                    │
│ codes_offset         │ u64     │ inside payload region, Wave64-aligned   │
│ codes_size           │ u64     │ packed-bit bytes (codes only)          │
│ scale_offset         │ u64     │ per-row/block scale bytes              │
│ scale_size           │ u64     │ row_count * dtype_size                 │
│ backup1_offset       │ u64     │ 0 = absent                             │
│ backup1_size         │ u64     │ packed at backup_bpw_1 bits            │
│ backup1_bpw          │ u8      │ 0/4/8                                  │
│ backup2_offset       │ u64     │ 0 = absent                             │
│ backup2_size         │ u64     │ packed at backup_bpw_2 bits            │
│ backup2_bpw          │ u8      │ 0/4/8                                  │
│ outlier_count        │ u32     │ unchanged meaning                      │
│ outlier_offset       │ u64     │ delta-varint format (see below)        │
│ outlier_index_width  │ u8      │ 0 = varint, 1 = u32 flat (legacy)     │
└──────────────────────┴─────────┴────────────────────────────────────────┘
```

This is a bigger entry (~120 B vs ~50 B today). For a 7B model (~600 tensors) that adds ~40 KB of registry — negligible against the size savings. Revisit if profiling says otherwise (Open Question Q2).

### Normals Stream — interleaved codes + per-row scales

`block_size = 0` (EXL2 mode, the V3 default):

```
For each row r in [0, row_count):
    [ scale_r          : u8            ]   // broadcast load, 1/lane
    [ codes_r[c..c+W]  : W packed bits ]   // W = row_stride * default_bpw, byte-aligned
    -> each row padded to next multiple of 256 B (Wave64 segment)
```

`block_size > 0` (GGUF K-quant-compatible mode):

```
For each row r:
  For each block b in row (b covers block_size elems):
    [ scale_r_b   : u8 ]
    [ codes_r_b[..] : block_size * default_bpw bits ]
  -> pad row to 256 B
```

Codes are stored in **packed-bit form** at `default_bpw` bits, big-endian-bit, little-endian-byte — same convention as EXL2/GPTQ so a port-from-existing-quant codepath is direct.

### Residual (backup) streams

`backup_n` carries the additive correction at indices that benefit from higher precision. Layout for each backup:

```
[ scale_offset_n : u64 ]   // per-row u8 scales (same encoding as row scales)
[ codes_n        : packed at backup_n_bpw bits, Wave64-aligned ]
```

The kernel reads:

```
reconstructed_w[i] = dequant(codes[i], row_scale[r])
                   + (backup1_bpw > 0 ? dequant(b1[i], b1_scale[r]) * gptq_inv_r : 0)
                   + (backup2_bpw > 0 ? dequant(b2[i], b2_scale[r]) * gptq_inv_r : 0)
                   + (is_outlier[i] ? outlier_residual[i] : 0)
```

`gptq_ordered=1` lets `gptq_inv_r` (row scaling) be applied; if `gptq_ordered=0`, backups are NOT applied even if present — the field is reserved (Open Question Q3).

### Outliers Stream — compressed variant

`outlier_index_width = 0` (V3 preferred):

```
[ outlier_count  : u32 ]                           // already in entry
[ delta_varint[idx_0=0] | delta_varint[idx_1-idx_0] | ... ]
[ value_dtype : u8 ]                               // 0 = u8 (residual to block),
                                                    // 1 = f16 (raw, legacy)
[ delta_u8_res[0] | delta_u8_res[1] | ... ]        // OR f16 values if dtype=1
```

Delta-varint with **bitpacked indices** is an Open Question (Q4) — 1 byte/correction vs flat u32 ≈ 4× cheaper. For sorted indices (~most outliers are contiguous in a column run), the deltas compress sharply.

`outlier_index_width = 1` (legacy V2 path): `u32 index | f16 value`, 6 B/outlier. Kept so V3 files can mix legacy and new tensors if the writer chose so.

### Reader side (host)

```
STEP 1: parse header + JSON metadata
─────────────────────────────────────
STEP 2: parse tensor registry (one entry per tensor)
─────────────────────────────────────
STEP 3: for each tensor, lazily:
  - seek to codes_offset / scale_offset / backup_n / outlier_offset as needed
  - codes/scale/backup: returned as packed Vec<u8> (kernel dequants)
  - outliers: decoded host-side into Vec<GrimOutlierV3>
STEP 4: TensorProvider.get(name) returns RawTensor with bytes across the above
        and a DType carrying enough Storage info for the backend kernel to dispatch
```

### Writer side (converter)

```
STEP 1: read source (GGUF / safetensors / EXL2 / AWQ) via TensorProvider
STEP 2: run grim-quant machinery: Hessian (EvoPress cardinality), per-row bpw solve
STEP 3: run GPTQ second-order reconstruction on codes + backups
STEP 4: choose block_size, residual_bpw layout, per-row bpw table
STEP 5: pack codes interleaved with per-row u8 scales, Wave64-align
STEP 6: delta-varint encode outliers + delta-u8 outlier values
STEP 7: optionally zstd the entire payload region per-tensor
STEP 8: write GrimTensorEntry + metadata JSON, let GrimFile::write recompute offsets
```

Steps 2/3 are out-of-scope for this format spec — owned by `grim-quant`. The format defines the slots; `grim-quant` decides what fills them.

### Backend (ROCm) dispatch

```
On GrimProvider::open():
    parse metadata + tensor registry (CPU, cheap)
    mmap payload region if backend supports external memory (D11)
Else:
    lazy-read normals per tensor on first get()

On kernel dispatch:
    look at fusion_mask   -> select fused vs split kernel
    look at gptq_ordered   -> apply row inverse if 1
    look at block_size     -> row-tiled vs block-tiled dispatch
    look at backup_bpw_n   -> chained dequant VGPR pipeline
    look at per_row_bpw_mode -> per-row mixed kernel if 1
```

## Implementation Plan

Phased so wave 1 is shippable on its own (back-compat parity), waves 2+ add the differentiators in order of leverage.

### Phase 1 — Format scaffold (back-compat parity, no win yet)

- [ ] **1.1** Bump `FUCKING_SORCERY` terminator byte to `\x03`; add `format_version: u8` field after magic in `GrimHeader`. Reader accepts `\x01`/`\x02` (back-compat reads the V2 layout under the old path).
- [ ] **1.2** Replace `GrimTensorEntry` with V3 shape. All V2 fields map 1:1; new fields default to legacy meaning when zeroed (`block_size=0, per_row_bpw_mode=0, default_bpw=base_bitwidth, backup_bpw_n=0, outlier_index_width=1`).
- [ ] **1.3** Update `GrimFile::read`/`write` for V3 entry (registry byte-size recomputation).
- [ ] **1.4** Add `GrimOutlierV3` (delta-varint + residual-u8) alongside `GrimOutlier` (legacy, 6 B flat). `read_outliers` dispatches on `outlier_index_width`.
- [ ] **1.5** Tests: V2 file still opens via V3 reader; V3 file with all-zero V3 fields round-trips and reads identical to V2; entry serialization round-trips.

### Phase 2 — Per-row scales (the GGUF-baseline match)

- [ ] **2.1** Add `row_count`, `row_stride`, `block_size`, `we_scale_offset/_size`, `row_scale_dtype` to `GrimTensorEntry`.
- [ ] **2.2** Update `normals_packed_size` to compute codes_size + scale_size, both Wave64-aligned.
- [ ] **2.3** Add `write_normals` that interleaves codes + per-row u8 scale, padding each row to 256B.
- [ ] **2.4** Update `read_normals` to return a struct `{ codes: Vec<u8>, scales: Vec<u8> }` OR two separate reads — the implementer decides based on whether the kernel wants one mmap region or two.
- [ ] **2.5** Tests: RTN quantizer (no calibration) writes a 4bpw tensor with per-row scales and the dequantized result matches F32 within 1 block error.

### Phase 3 — Per-row mixed bitwidth (EXL2 beat)

- [ ] **3.1** Add `per_row_bpw_mode`, `default_bpw`, `bpw_table_offset`, `bpw_table_count` to entry. Layout: `[u8; row_count]` of bitwidths each in {2..8}.
- [ ] **3.2** `write_normals` learns to pack rows of varying bpw into one Wave64-aligned stream (each row padded to 256B independently).
- [ ] **3.3** `read_normals` returns enough metadata for kernel mixed-bpw dispatch.
- [ ] **3.4** Tests: a 2-row tensor with [2bpw, 6bpw] dequantizes correctly, file size is between uniform-2 and uniform-6.

### Phase 4 — Residual / backup streams

- [ ] **4.1** Add `backup1_offset/_size/_bpw`, `backup2_*`, `gptq_ordered` to entry.
- [ ] **4.2** Define `write_backup_n(level, codes, scales_per_row)` and `read_backup_n`.
- [ ] **4.3** Kernel dispatch: chained dequant (codes → backup1 → backup2 → outliers) — owned by `grim-backend-rocm`, this phase only wires the slots.
- [ ] **4.4** Tests: synthetic 4-bpw + 8-bit backup reconstructs to within 0.1% of a F16 target.

### Phase 5 — Outlier compression

- [ ] **5.1** Add `outlier_index_width`, `outlier_residual_bpw` to entry.
- [ ] **5.2** Writer path: sort outlier indices ascending (cheap), delta-varint encode, encode values as deltas from block-reconstructed values at the outlier resolution.
- [ ] **5.3** Reader path: decode delta-varint, decode residual values, add to dequant base.
- [ ] **5.4** Tests: 1000 outliers in a contiguous run compress to < 1200 bytes (3× under flat-u32 + f16).

### Phase 6 — Compression + mmap (size + decode speed)

- [ ] **6.1** Per-tensor `compression: u8` field. zstd-decompress on read if set.
- [ ] **6.2** `mmap_registry_offset`/`mmap_payload_region` in `GrimFile` so ROCm backend gets a handle for external memory interop.
- [ ] **6.3** Tests: round-trip with compression=1, byte-identical decoded output.

### Phase 7 — Fusion metadata wiring

- [ ] **7.1** Populate `fusion_mask` from `GrimFusionOp`s already on `GrimMetadata`.
- [ ] **7.2** Kernel reads `fusion_mask` and selects fused RMSnorm+matmul / QKV-attention path.
- [ ] **7.3** Tests: a tensor with `fusion_mask=0b11` returns a `TensorMeta` flagging compatibility with the fused path.

### Phase 8 — V2/V3 migration

- [ ] **8.1** Standalone `grim convert --in foo.grim --out bar.grim --v3` subcommand: read V2 file, write V3 file with `block_size=0, default_bpw=v2.base_bitwidth, per_row_bpw_mode=0, backups=0, outlier_index_width=1` (basically identical contents but in V3 entry encoding).
- [ ] **8.2** `convert_to_grim` writes V3 by default after Phase 1 lands.

---

## Skills by Phase

Each phase lists the skills to load in order when picking up the work. Only load skills that apply — running `clean-code-guard` on a 2.5-line routine is ceremony; running `rust-tdd` without writing tests first is failure of the discipline.

Mapping for the categories named in the brief:

| User-named bucket | Skill used | Why this is the right resolution |
|---|---|---|
| writing guidelines | `writing-guidelines` | Doc-comms rules, terse `file:line` findings, doc-only scope |
| project planning | `writing-plans` | Bite-sized tasks, exact files/types, TDD steps |
| creative / brainstorming | `brainstorming` | Interactive design and trade-off exploration before code |
| clean-code guard | `clean-code-guard` | Always-applied imperatives + self-check before delivery |
| Rust TDD | `rust-tdd` (Red/Green with the right assertion tooling: `assert_eq!`/`insta`/`proptest`) | Tools and asset-call guidance for Rust test-first |
| specification driven | `specification-writing` (formal spec authoring) + `writing-plans` (breaking the spec into tasks) | Best-available substitute; aligns with how this doc itself was produced |

Skills listed under each phase are loaded with the `skill` tool, in the order given. Subsequent/optional skills are loaded only when the phase triggers them.

### Phase 1 — Format scaffold (back-compat parity)

In order:
1. `specification-writing` — re-read D1–D15; confirm Phase 1 implements only D1/D2/D14.
2. `rust-tdd` — write entry round-trip tests first (`GrimTensorEntry` V3 read → write → read equality, V2 fixture still opens, V3 fixture with all-zero V3 fields round-trips).
3. `clean-code-guard` — apply imperatives 1–23 after the first impl pass; watch names (no `data`/`temp`/`flag`), small functions, no swallowed errors, no `unimplemented!()` stubs in any reachable reader path.
4. `writing-guidelines` — review doc edits to `grim-file.md` for clarity and `file:line` citation.
5. (Optional) `writing-plans` — only if Phase 1 needs a task split for a separate executor.

Skip for Phase 1: `brainstorming` (spec is fixed), `rust-architect` (architecture is set), any FFI/ROCm skill (kernel work is in Phase 7).

### Phase 2 — Per-row scales

In order:
1. `rust-tdd` — first failing test for `normals_packed_size` with the new codes+scales codepath. `assert_eq!` (default) for round-trip; reach for `insta` only if the row-layout test would be ~20 lines of hand-coded expectations.
2. `clean-code-guard` — imperative 14 (no speculative anything: don't add `block_size` consumer code yet, only the field); imperative 18 (the RTN quantizer must actually do work, no fixture-encoded "ok" return).
3. `specification-writing` — confirm Phase 2 implements exactly D2+D11+D14, no more.
4. (Optional) `rust-architect` — if the kernel-side question "separate scale stream vs interleaved" comes up.

Skip for Phase 2: `writing-guidelines` (no prose), `brainstorming` (resolved).

### Phase 3 — Per-row mixed bitwidth (EXL2 beat)

In order:
1. `brainstorming` — once, only if the kernel-side dispatch for per-row bpw needs trade-off exploration.
2. `rust-tdd` — `[2bpw, 6bpw]` two-row tensor test (Phase 3.4). `assert_eq!` on dequantized values; `proptest` for "any table of valid bpws in {2..8} round-trips losslessly into the kernel's expected codes layout."
3. `clean-code-guard` — imperative 8 (OCP: per-row bpw must extend by new variant, not edit existing dispatch); imperative 20 (enumerate `{2..8}` in a comment before doing anything).
4. `writing-plans` — only if mixing bpw rows forces a kernel-side task split later.

Skip for Phase 3: `specification-writing`, `writing-guidelines`.

### Phase 4 — Residual / backup streams

In order:
1. `brainstorming` — once, only if chaining 3 layers (codes+backup1+backup2+outlier) raises a kernel ordering question not yet pinned in §Architecture.
2. `rust-tdd` — synthetic 4-bpw + 8-bit backup test (Phase 4.4). `assert!` with explicit percent tolerance, not `==` (FP tolerance is the assertion).
3. `clean-code-guard` — imperative 12 (the chain is one thing; if it grows, extract a `reconstruct_residual_chain` helper); imperative 15 (never swallow zstd/zp errors).
4. (Optional) `rust-architect` — if deciding `Option`-slot struct vs separate fields for backup levels.

Skip for Phase 4: doc-side skills (no prose target).

### Phase 5 — Outlier compression

In order:
1. `rust-tdd` — `outlier_index_width=0` reader test (Phase 5.4). `proptest` for "any sorted index sequence round-trips losslessly through delta-varint."
2. `clean-code-guard` — imperatives 19 (re-derive the varint, don't copy from existing impl — off-by-one bugs cluster here) and 16 (don't defensively check unfeasible-no-overflow arithmetic).
3. `specification-writing` — verify Q4 (Open Question on bitmap vs varint) is resolved or stays open; if stayed open, leave the section alone.

Skip for Phase 5: `brainstorming` (resolved), `rust-architect`.

### Phase 6 — Compression + mmap

In order:
1. `brainstorming` — **the only phase where this should hold the work for any non-trivial duration.** mmap external-memory interop is unresolved (Open Question Q7). Don't write the mmap half until the implementer has a clear ABI answer.
2. `rust-tdd` — round-trip with `compression=1`, byte-identical decode (Phase 6.3). `assert_eq!` only — no snapshot, byte-equality is the assertion.
3. `clean-code-guard` — imperative 5 (comments are "why", not what `zstd::decode` does); imperative 21 (dead-code strip after Phase 6 if you conditionally compile the mmap path).

Skip for Phase 6: `specification-writing`, `writing-plans`.

### Phase 7 — Fusion metadata wiring (kernel-side)

In order:
1. `rust-architect` — once. Phase 7 introduces kernel-side dispatch. Run the architect review **before code lands** for `fusion_mask` ABI shape.
2. `rust-tdd` — `fusion_mask=0b11` returns a `TensorMeta` flag (Phase 7.3); test in `grim-format` asserting the flag is set; kernel-side test deferred to ROCm CI.
3. `clean-code-guard` — imperative 17 (verify every kernel-side import. **Do not** generate HIP/HIPBLAS calls from memory — USENIX Sec '25 hallucination risk on AMD-only headers).
4. `writing-guidelines` — review the `fused_dequant_gemm.rs` doc-additions for clarity.

Skip for Phase 7: `specification-writing`, `brainstorming`.

### Phase 8 — V2/V3 migration

In order:
1. `specification-writing` — once. Verify D1 + V2-preserved path semantics match. Update the spec only if a real drift appears (Phase 8 is a wrapper, not a spec change).
2. `rust-tdd` — convert-in→out golden test: V2 fixture + V3 fixture both pass through the converter; round-trip-stable.
3. `clean-code-guard` — imperatives 17 (verify `--v3` flag wired correctly in CLI), 21 (remove legacy code path after one full release cycle, not now).
4. `writing-guidelines` — review CLI help text and CHANGELOG entry.

Skip for Phase 8: `brainstorming`, `rust-architect`, any non-TDD `rust-tdd` use.

## Edge Cases

### V2 reader opens V3 file
1. V2 reader sees `magic=[0x47,'R','I','M',0x03]`.
2. V2 `GrimHeader::read` aborts at `magic != FUCKING_SORCERY` (format.rs:51).
3. Correct behavior — hard fail with a message telling the user to upgrade.

### V3 reader opens V2 file
1. V3 reader sees magic `[0x47,0x52,0x49,0x4d,0x01]`.
2. Dispatches to the V2 path (preserved in `format_v2.rs`).
3. Output: a `GrimFileV3` with `format_version=2` and all V3 extras zeroed. Modules downstream treat as "V2-shaped" (no scale stream, no backups).

### Compressed payload read without zstd available
1. Entry has `compression=1` but the build disabled zstd (`#[cfg]`-gated).
2. Reader fails with a clear error pointing at the cfg. No silent corruption.

### Per-row bpw table references rows not present
1. `bpw_table_count != row_count`.
2. Reader rejects with `Error::Backend`. Writer must assert at pack time.

### Outliers and backup overlap
1. An outlier index coincides with a backed-up position.
2. Spec allows it; kernel must apply `codes + backup + outlier_residual`. Not an error.

## Open Questions

1. **Backups beyond 2 levels.** D4 caps at 2. EXL2 sometimes uses 3 on hard tensors. Should the field be `backup_count: u8` + variable-array, or fixed 2?
   - Options: (a) variable array via offsets table, (b) fixed 2 fields.
   - **Recommendation**: fixed 2 for V3, variable array revisited when a real accuracy regression needs a 3rd. (Class 3.)

2. **Registry entry size.** V3 entry is ~120B vs V2 ~50B. At 100k tensors (large MoE) that's 7 MB just of registry.
   - Options: (a) split into `entry_core` + `entry_v3_ext` (variable), (b) keep flat, (c) compress the registry table with zstd after the metadata JSON.
   - **Recommendation**: keep flat for V3, revisit for very large MoE configs. (Class 3.)

3. **`gptq_ordered=0` with backups present.** Spec says backups are skipped. Should this be an error?
   - **Recommendation**: silently skip and warn; the kernel treats it as "backups disabled." (Class 2.)

4. **Outlier index encoding.** Bitpacked bitmap vs delta-varint. Bitmap is O(N/8) regardless of count; delta-varint is O(outlier_count × log_delta). At outlier_count > N/64 the bitmap wins.
   - **Recommendation**: delta-varint for V3, add `outlier_index_width=2` (bitmap) later if dense-outlier models need it. (Class 3.)

5. **Scale dtype for asymmetric quant.** D15 says symmetric only. Some activations need asymmetric (zero-point). The `row_scale_dtype` field reserves f16 for future asymmetry — should the V3 spec formally ban asymmetric or leave the slot?
   - **Recommendation**: leave the slot, document as reserved. Writers MUST set `row_scale_dtype=0` for V3. (Class 3.)

6. **Per-row bpw table layout.** Inline at `bpw_table_offset` vs shared in metadata JSON.
   - **Recommendation**: inline per-entry for now (simpler lifetime), revisit when many tensors share the same table. (Class 2.)

7. **mmap external memory ABI.** Hip's external-memory interop ABI vs AMD's IPC — neither standardized for this use.
   - **Recommendation**: defer the actual impl to `grim-backend-rocm`, leave the `mmap_*` slots as reserved in this spec. Don't block V3 ship. (Class 3.)

## Decisions Log

- **Keep** `FUCKING_SORCERY` constant name. Constraint: legacy readers depend on byte-level magic comparison at `format.rs:51`. Revisit when: rename is requested and a migration story for in-flight files is provided. (Class 3 keep.)
- **Keep** outliers as a dedicated stream (not folded into residuals). Constraint: V2 reader/writer compat on this stream. Revisit when: outlier_count is consistently <1% and the residual+outliers path simplifies into one. (Class 3 keep.)
- **Keep** `block_size` field even though EXL2 mode (per-row) makes it 0. Constraint: GGUF K-quant import path will want `block_size=256` to land directly into Q4_K-shaped tensors without rewriting. Revisit when: K-quant import path is dropped. (Class 2 keep.)
- **Keep** JSON metadata layer uncompressed for now. A4 was speculative; the metadata is already ~1 KB compressed. Revisit when: total file size profiling shows JSON dominates (it won't). (Class 3 keep, defer.)

## Success Criteria

- [ ] `cargo build` succeeds with V3 entry.
- [ ] All Phase 1 tests pass; V2 files still open; V3 file round-trips.
- [ ] Phase 2 implementation: a 4-bpw RTN-quantized 1B-param dummy tensor in V3 has **smaller or equal** file size to a GGUF Q4_K-encoded copy of the same tensor at the same RMSE.
- [ ] Phase 3 implementation: a tensor with per-row mixed [3bpw, 5bpw, 8bpw] dequantizes correctly and the file size is between uniform-3 and uniform-8.
- [ ] Phase 5 implementation: outliers compress to <25% of V2 flat-format size at the same outlier_count.
- [ ] Phase 6 implementation (compression only): enabling zstd on normals reduces total payload by ≥30% for dense layer weights.
- [ ] No public API break for `GrimProvider::open` / `TensorProvider::get` / `TensorProvider::meta` call sites outside `grim-format`.

## References

- [`crates/grim-format/src/format.rs`](file:///D/rex/projects/grim/crates/grim-format/src/format.rs) — current V2 header + entry + `read_normals`/`read_outliers` to expand.
- [`crates/grim-format/src/gguf.rs`](file:///D/rex/projects/grim/crates/grim-format/src/gguf.rs) — `GrimMetadata`, `from_gguf_metadata`, `GrimQuantOverride` (the `importance_score` slot V3 wires into).
- [`crates/grim-format/src/tprov.rs`](file:///D/rex/projects/grim/crates/grim-format/src/tprov.rs) — `GrimProvider` `TensorProvider` impl to extend.
- [`crates/grim-format/src/convert.rs`](file:///D/rex/projects/grim/crates/grim-format/src/convert.rs) — `convert_to_grim` writer path to update.
- [`crates/grim-quant`](file:///D/rex/projects/grim/crates/grim-quant) — owns EvoPress search + GPTQ; out of scope for this spec, but the slots this spec defines are the bridge.
- [`crates/grim-backend-rocm/src/kernels/fused_dequant_gemm.rs`](file:///D/rex/projects/grim/crates/grim-backend-rocm/src/kernels/fused_dequant_gemm.rs) — consumer of the V3 entries; reads `fusion_mask`, dispatches on `block_size`, applies `gptq_inv` when `gptq_ordered`.
- [`grim_v2.md`](file:///D/rex/projects/grim/grim_v2.md) — superseded §1 layout and the four-pillar motivation this spec finally implements.
