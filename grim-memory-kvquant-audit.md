# grim-memory + grim-kvquant audit verification & remediation record

Scope: crates/grim-memory (8 submodules, block pool / paged cache / tiered
spilling) and crates/grim-kvquant (Lloyd-Max KV compression, KV-OMNI
multimodal paths), as of this verification.

## Claim-by-claim verification

### grim-memory

| Claim | Verdict | Evidence |
|---|---|---|
| `free_with_tier`/`evict_cold` mark blocks HostRam + `recently_zero` even when `demote_to_host` fails, never reaching `free_list` | **CONFIRMED** | old lib.rs:450-464, 558-572 — errors were only `eprintln!`'d, then `location = HostRam` unconditionally |
| `promote_to_gpu` comment says mismatch "is a panic" but code silently truncates via `.min()` | **CONFIRMED** | old lib.rs:491-496 |
| `snapshot_block` always builds full `BLOCK_SIZE` shape regardless of `num_tokens` | **CONFIRMED** | old lib.rs:596-602 |
| `PagedKvCache` uses `pool.lock().unwrap()` on some paths | **CONFIRMED** | 10 sites (append_slot, tentative_append, commit, rollback_to, current_k/v, store_kv, new, seed_prefix) vs `unwrap_or_else(poisoning-tolerant)` elsewhere |
| compressor output recorded as metadata; spill stores raw f32; compression-to-spill loop not closed | **CONFIRMED** | `compress_block` was never called on the free/demote path; `demote_to_host(id, k, v)` always shipped raw f32 |

### grim-kvquant

| Claim | Verdict | Evidence |
|---|---|---|
| `random_orthogonal_matrix` simplified Gram-Schmidt unstable beyond dim ≈16 (tests tolerate 1e-2) | **CONFIRMED** | single-pass classical GS in f32; the test comment admitted the instability |
| `apply_rotation` O(dim²)/token, CPU reference only | CONFIRMED, by design | documented CPU-reference cost; see stale-claim note below for the GPU path |
| `dispatch_gpu_fused_attention` is a non-functional stub returning `Err(Unsupported)` | **STALE / INCORRECT** | the dispatcher packs K/V and calls `BackendDevice::kv_dequant_attention`; BOTH the CPU backend (device.rs:912, with a reference-equivalence test) and the ROCm backend (roc_device.rs:2782, real HIP kernel) implement the trait method. `fused_attention_gpu_path_returns_err_unsupported_when_no_kernel` asserts `is_ok` and passes. No action needed. |
| `compress_visual_tucker` embeds R in `value_meta` with `r_start = 1` assumption, no version/format tag; rank re-derived from the reader's `layer_depth_ratio` | **CONFIRMED** (worse than stated: a reader with a different depth ratio would mis-decode) | kv_omni.rs old dequantize_visual_tucker |
| `merge_across_modalities` has no reverse operation | **CONFIRMED** — and worse: the header recorded only value_meta lens, so a split was impossible even in principle (key_meta/key_bits/value_bits/num_tokens boundaries were lost) | kv_omni.rs old merge |
| `to_bytes`/`from_bytes` format version inferred from 24-vs-25-byte length | **CONFIRMED** with a concrete failure: any legacy 24-byte-header blob whose payload exceeded 25 bytes was parsed as new-format (phantom modality byte shifts the whole header) | old from_bytes `if buf.len() >= NEW_MIN` |

## Remediations applied

### grim-kvtransport (enabler for the memory fix)
- `LocalSpillManager`/`SharedSpillManager` gained a compressed-blob tier:
  `demote_compressed` (host RAM), `demote_compressed_to_nvme` (atomic
  write+rename), `retrieve_compressed` (non-destructive host reads; NVMe
  reads promote into the host tier, mirroring raw `retrieve` semantics).
  Raw and compressed residency are mutually exclusive per id; `evict`
  covers both.

### grim-memory
1. **Failed-demotion fallback**: `free_with_tier` and `evict_cold` now try
   compressed demotion first, then raw; only if BOTH fail do they fall back
   to the in-place release (zero + free list). A failed demotion can no
   longer strand a block in a fake HostRam state with no reclaim path.
2. **Strict promote validation**: `promote_to_gpu` errors with the actual
   geometry on any spill/pool length mismatch — the `.min()` silent
   truncation is gone (comment now matches code).
3. **Compressed-promote path**: `promote_to_gpu` retrieves the compressed
   blob, parses it, and decompresses via the attached compressor on CPU
   before the (strict) capacity validation.
4. **True token-count snapshots**: `snapshot_block` slices to the block's
   actual `num_tokens` rows; `compress_block` returns `None` for empty
   blocks. Compression no longer does work on padding.
5. **Poisoned-mutex safety**: all 10 `pool.lock().unwrap()` sites replaced
   with poisoning-tolerant `unwrap_or_else(|e| e.into_inner())`.
6. **Loop closed**: free/evict with compressor + spill attached now store
   the serialized `CompressedKvBlock` as the spilled bytes (host tier, then
   NVMe); raw f32 demotion is the fallback when compression fails.

### grim-kvquant
1. **Numerically stable orthogonalization**: `random_orthogonal_matrix` is
   now modified Gram-Schmidt with a reorthogonalization pass ("twice is
   enough"), accumulated in f64, cast to f32 at the end. Orthogonality
   gate re-tuned: dim 16 @ 1e-5, dim 64 @ 5e-5, dim 128 @ 2e-4 (was 1e-2
   at dim 16 only), plus isometry (L2-norm preservation) at the same
   tolerances, plus a rotation+transpose round-trip identity test at
   head_dim 128 (max err < 1e-4) and a determinism test.
2. **Self-describing blob format (v2)**: `to_bytes` now emits a
   `GKVB` magic + version byte + all four section lengths (including
   `value_bits_len`, making the blob fully length-validating).
   `from_bytes` dispatches on magic (unknown version ⇒ error) and falls
   back to a validated legacy trial parse: candidate A (modality byte,
   must be a valid tag 0..=2) then candidate B (24-byte header, Text).
   The old length-sniff misparse is structurally impossible now.
3. **Merge/split pair**: the merge header now records ALL sub-block
   boundaries (num_tokens, key_meta, key_bits, value_bits, value_meta
   lens — 16-f32 header), and `KvOmniEvictor::split_merged_block`
   reconstructs the three per-modality sub-blocks exactly, with
   header-consistency validation (boundaries must sum to payload lens).
   Tests: dummy-block merge→split byte-identity, real compressor outputs
   merge→split→dequantize round trip, and a reject-non-merged-block error
   test.
4. **Self-describing Tucker layout**: `compress_visual_tucker` embeds the
   rank in `value_meta` (`[reserved, rank, R_flat…]`), so
   `dequantize_visual_tucker` reads the rank from the block instead of
   re-deriving it from the reader-side `layer_depth_ratio` (which would
   corrupt decoding across differing depth ratios). Legacy single-rank
   blocks still decode via the old layout.

## New numeric regression tests

grim-memory (lib tests 34 → 38):
- `failed_demotion_falls_back_to_in_place_release` — mismatched spill
  geometry must release the slot, not strand it
- `promote_geometry_mismatch_is_an_error` — mismatch must error, block
  contents untouched
- `compress_block_uses_actual_num_tokens` — partial block compresses 5
  rows, empty block → None
- `compressor_spill_loop_is_closed_end_to_end` — free stores the COMPRESSED
  blob (raw tier empty), promote decompresses bit-identically to the
  compressor's own reference dequantization, values exact for constants,
  keys bounded by quantizer granularity, tier/location/received/token-count
  invariants restored

grim-kvquant (lib tests 32 → 36 + kv_omni updates):
- orthogonality/isometry at dims 16/64/128 with 1e-5..2e-4 tolerances
- rotation round-trip identity at head_dim 128
- rotation-matrix determinism
- legacy A/B blob parsing (incl. the exact >25-byte payload case the old
  reader misparsed)
- v2 corruption gate: every truncation prefix errors; unknown version errors
- merge→split round trips (dummy + real compressor outputs, incl. split
  visual block still dequantizing through the Tucker path)
- split rejects non-merged blocks
- visual tucker meta asserts the embedded rank field

## Verification

- `cargo check --workspace --tests` clean.
- grim-kvquant + grim-memory + grim-kvtransport + grim-engine:
  269 tests passed, 0 failed.
- No stubs were introduced; the one audit-claimed stub (GPU fused-attention
  dispatch) was verified to be already functional end-to-end (CPU reference
  + ROCm HIP kernel) and needed no work.
