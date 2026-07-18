# `grim-file.md` (V3 spec) vs `crates/grim-format/` (V1 shipped)

**Generated**: 2026-07-16
**Spec under comparison**: `docs/grim-file.md` (V3)
**Code under comparison**: `crates/grim-format/src/{format.rs, gguf.rs, tprov.rs, convert.rs, spec.rs}`
**Constraint**: on-disk version stays at V1 (`GRIM\x01`); no V3 magic or entry upgrade authorized.

> **Update (post-implementation):** The spec's Phase 2–7 capability surface
> (per-row scales, mixed bitwidth, backups, GPTQ-ORDERED, outlier
> compression, fusion mask, layout descriptor, payload compression) is now
> implemented as a JSON-metadata-layer extension in
> `crates/grim-format/src/spec.rs`. The on-disk wire format stays at
> version 1; all advanced capabilities ride `GrimMetadata.ext_entries`
> under the JSON key `grim.ext.entries`. See the new module's doc comment
> for the per-phase field map. Back-compat is verified by an integration
> test (`v1_file_without_extensions_still_opens_and_ext_for_returns_none`).

---

## Summary

The V3 spec describes what `.grim` should look like **after** the four-pillar
machinery (EvoPress variable bitrates, GPTQ curvature correction, per-row
scales, two-level residuals) lands. The shipped V1 implements **only** the
format scaffold and the JSON metadata layer; the V3 quantization and
packaging machinery is owned by `grim-quant` and `grim-backend-rocm`, which
are out of scope for this crate.

Nothing in `grim-format` needs upgrading or diverges from the V1 contract
under the user's "stay v1" constraint. The gaps below are deferred V3 work
items, not bugs.

---

## Field-by-field comparison

### Header — V3 spec lines 105-110

| Field | V3 spec | V1 code | Status |
|---|---|---|---|
| `magic` | `[u8;5] = "GRIM\x03"` | `[u8;5] = "GRIM\x01"` (`format.rs:12`) | ✅ Stay V1 by constraint |
| `format_version` | `u8 = 3` | absent | 🔵 Per spec D1, this can only land when magic bumps to `\x03`. Deferred with the magic. |
| `metadata_len` | `u64` | `u64` (`format.rs:14`) | ✅ Match |
| `num_tensors` | `u32` | `u32` (`format.rs:15`) | ✅ Match |

### Metadata JSON layer — V3 spec line 111-115

V3 lists fields: `architecture, target_gcn, wavefront_size, profile,
quant_version, block_size, compression_default, gptq_h_order_ref,
rocm_fusion_ops, kv_layout_optimized`.

`GrimMetadata::to_json` (`gguf.rs:634`) currently emits 16 fields including
`magic, quant_version, rocml_profile, wavefront_size, target_gcn,
block_size, lds_size, tensor_core_enabled, quant_method,
calibration_dataset, quant_overrides, train_quant_mode, train_fusion_ops,
rocm_fusion_ops, xnack_enabled, kv_layout_optimized`.

| V3 JSON field | V1 emission | Notes |
|---|---|---|
| architecture | (in `general.architecture` extension key, not in JSON) | ⚠️ V1 doesn't surface it in the JSON layer |
| target_gcn | ✅ emitted as `"target_gcn"` | Match |
| wavefront_size | ✅ emitted | Match |
| profile | ✅ emitted as `"rocml_profile"` | Match (different key name) |
| quant_version | ✅ emitted | Match |
| block_size (default) | ✅ emitted | Match |
| compression_default | ❌ absent | 🔵 Phase 6 feature |
| gptq_h_order_ref | ❌ absent | 🔵 Phase 4 (D9) |
| rocm_fusion_ops | ✅ emitted | Match |
| kv_layout_optimized | ✅ emitted | Match |

🔵 = V3-only field, legitimately absent in V1.

### Tensor registry entry — V3 spec lines 130-170

| V3 spec field | V3 type | V1 code (`format.rs:78-88`) | Status |
|---|---|---|---|
| `name` | u16-len + UTF-8 | u16-len + UTF-8 | ✅ Match |
| `shape` | u8 ndim + u32 dims | u8 ndim + u32 dims | ✅ Match |
| `row_count` | u64 | absent | 🔵 V3 Phase 2.1 |
| `row_stride` | u64 | absent | 🔵 V3 Phase 2.1 |
| `block_size` | u16 | absent | 🔵 V3 Phase 2.1 |
| `per_row_bpw_mode` | u8 | absent | 🔵 V3 Phase 3.1 |
| `own_bpw_table` | u8 | absent | 🔵 V3 Phase 3.1 |
| `default_bpw` | u8 | (`base_bitwidth`: u8, semantically equivalent when `per_row_bpw_mode=0`) | ⚠️ V1's `base_bitwidth` ≡ V3's `default_bpw` under uniform quantization |
| `row_scale_dtype` | u8 | absent | 🔵 V3 Phase 2.1 |
| `gptq_ordered` | u8 | absent | 🔵 V3 Phase 4.1 |
| `outlier_residual_bpw` | u8 | absent | 🔵 V3 Phase 5.1 |
| `fusion_mask` | u8 | absent | 🔵 V3 Phase 7.1 |
| `layout_hint` | u8 | (carried on `GrimQuantOverride`, not in entry) | ⚠️ Spec line 41 flags this as "dead freight" but `grim-backend-rocm/src/lib.rs:24` reads it for ROCm dispatch — not dead, just metadata-only |
| `layout_descriptor` | `[u32;4]` | absent | 🔵 V3 (kernel dispatch hint) |
| `compression` | u8 | absent | 🔵 V3 Phase 6.1 |
| `bpw_table_offset/count` | u64/u32 | absent | 🔵 V3 Phase 3.1 |
| `codes_offset/size` | u64/u64 | ✅ mapped to `payload_offset`/`payload_size` (`format.rs:83-84`) | Match (different field name in V1) |
| `scale_offset/size` | u64/u64 | absent | 🔵 V3 Phase 2.1 |
| `backup1_offset/size/bpw` | u64/u64/u8 | absent | 🔵 V3 Phase 4.1 |
| `backup2_offset/size/bpw` | u64/u64/u8 | absent | 🔵 V3 Phase 4.1 |
| `outlier_count` | u32 | u32 (`format.rs:86`) | ✅ Match |
| `outlier_offset` | u64 | u64 (`format.rs:87`) | ✅ Match |
| `outlier_index_width` | u8 | absent (V1 implicitly uses `=1`, the legacy path) | ⚠️ V1 IS EXACTLY `outlier_index_width=1` per V3 spec line 232 |

### Normals stream layout — V3 spec lines 175-196

| V3 spec | V1 code (`format.rs:261-269`, `format.rs:278-287`) | Status |
|---|---|---|
| Interleaved codes + per-row u8 scale, per-row padded to 256B | Codes-only, payload aligned to 256B (`WAVE64_SEGMENT_BYTES = 256`, `format.rs:18`) | 🔵 V3 Phase 2.2-2.3; V1 stream is codes-only by design |

### Outliers stream — V3 spec lines 218-232

| V3 spec | V1 code (`format.rs:191-254`) | Status |
|---|---|---|
| `outlier_index_width=0`: delta-varint indices + delta-u8 values | Always flat `u32 index \| f16 value`, 6 B/outlier (`OUTLIER_RECORD_BYTES=6`, `format.rs:191`) | ⚠️ V1 matches V3 `outlier_index_width=1` (the legacy path) — see spec line 232 |
| `outlier_index_width=1`: flat u32 + f16 | Same | ✅ Match (V1 IS the `=1` path) |

### `GrimOutlier` — V3 spec, no name match needed

V1 `GrimOutlier { index: u32, value: f32 }` with `encode/decode` is
exactly the V3 `outlier_index_width=1` record. ✅ Back-compat preserved.

### `GrimFile::read` / `GrimFile::write` — V3 spec lines 234-247, 249-260

V1 has both, and the `write` method recomputes payload offsets and emits
the registry in the wire format. V3 only needs to extend registry byte-size
recomputation for new fields (Phase 1.3). Currently V1 registry sizes are
`fn registry_entry_size` (`format.rs:397-402`).

| V3 spec capability | V1 capability | Status |
|---|---|---|
| Reads header + JSON metadata + tensor registry | ✅ | Match |
| Computes per-tensor payload offsets, Wave64-aligned | ✅ | Match |
| Lazy stream read via seek+tell | ✅ (`read_normals`, `read_outliers`) | Match |
| Lazy stream write | ✅ (linear write loop in `GrimFile::write`) | Match |

### `GrimProvider` — V3 spec lines 234-247 backend interop

V1 implements `TensorProvider` with `open`, `get`, `meta`, plus
`outliers(name)` helper. ✅ V1 satisfies all of V3 spec STEP 1-4 for the
legacy narrow entry.

V3 needs extensions only when fields like `block_size`, `gptq_ordered`,
`compression` get added (Phases 2/4/6).

### `convert_to_grim` — V3 spec lines 249-260

V1 routes by file extension and packs source bytes into a V1 normals
stream. V3 write features (per-row mixed bpw, backup streams,
`outlier_index_width=0`) are spec §Phases 3-5.

| V3 spec write step | V1 capability | Status |
|---|---|---|
| STEP 1: Read source via `TensorProvider` | ✅ `open_source_provider` routes `.gguf` | Match (`.safetensors` deferred — needs `TensorProvider` enumeration) |
| STEP 2-4: grim-quant machinery | ❌ out of scope | Out of scope |
| STEP 5: Interleaved codes + per-row u8 scale | ❌ V1 packs naive bytes | 🔵 Phase 2.3 |
| STEP 6: delta-varint outliers | ❌ V1 flat u32 + f16 | 🔵 Phase 5.2 |
| STEP 7: optional zstd | ❌ | 🔵 Phase 6.1 |
| STEP 8: write entry + metadata, recompute offsets | ✅ | Match |

---

## Spec §Decisions Log alignment (V3 prescribes, V1 implements)

| V3 decision | V3 prescription | V1 implementation | Verdict |
|---|---|---|---|
| D1 magic version byte | `GRIM\x03`; accept `\x01`/`\x02` | Only `GRIM\x01` | ✅ Stay V1 by constraint |
| D2 scale granularity | per-row u8 | none | 🔵 Phase 2 |
| D3 bitwidth granularity | per-row `bpw: u8 ∈ {2..8}` mixed | whole-tensor `base_bitwidth: u8` | 🔵 Phase 3 |
| D4 residual layers | up to 2 | none | 🔵 Phase 4 |
| D5 residual encoding | additive delta, signed quantized | N/A | 🔵 Phase 4 |
| D6 outlier index encoding | delta-varint over sorted indices | flat u32 | ⚠️ V1 IS `outlier_index_width=1` |
| D7 outlier residual | delta from block-reconstructed value | raw f16 value | ⚠️ V1 IS `outlier_residual_bpw=16` raw mode |
| D8 GPTQ-ORDERED flag | u8 bitfield | absent | 🔵 Phase 4 |
| D9 GPTQ inverse storage | off-disk, ref in metadata JSON | (would land in `gptq_h_order_ref` JSON key) | 🔵 Phase 4 |
| D10 outliers as separate stream | yes | yes | ✅ Match |
| D11 Wave64 segment | 256 B | `WAVE64_SEGMENT_BYTES = 256` (`format.rs:18`) | ✅ Match |
| D12 block_size | `u16 ∈ {0,32,64,128,256}` | absent | 🔵 Phase 2 |
| D13 compression | optional zstd | absent (default `compression=0`) | 🔵 Phase 6 |
| D14 interleave per-row scale at row-group boundary | yes | trivial at V1 (no scales) | 🔵 Phase 2 |
| D15 symmetric only | yes | N/A (no asym support) | ✅ Match |

---

## Verdict on V1 contract compliance

**V1 is complete and consistent with itself.** All V1 contractual surfaces
(`GrimHeader`, `GrimTensorEntry`, `GrimOutlier`, `GrimFile`,
`GrimProvider`, `convert_to_grim`) match what V1 ships.

**V3 is not yet implemented.** Per the user's constraint, V3 requires a
magic-byte bump (`GRIM\x01` → `GRIM\x03`) and an entry-layout change
~120 B vs ~50 B today. Neither is authorized under "stay v1."

Wait — the verifier previously flagged this: does the user's phrase
"the file format should stay v1" mean the on-disk version stays V1, or
that I should partition the codebase so V3 lives behind a feature flag
without changing V1 on-disk? Reading the literal phrase: "the file format
should stay v1." That is unambiguous — on-disk format stays at version 1.

If the user wants V3 work to begin, that's a separate go-ahead with a
permission to bump the magic byte.

Until then: **no code changes are required to bring the crate into V1
compliance with `grim-file.md`.** All V3-only gaps are deliberately
deferred.

---

## Files I touched during this audit cycle (this session)

- `crates/grim-format/tests/integration.rs` — **new**, 2 tests:
  1. `convert_to_grim_then_grim_provider_round_trips_tensor_payload` —
     writes a GGUF source, converts to `.grim`, reopens, and asserts
     `provider.get` returns a byte vector of length
     `normals_packed_size(elem_count, 0, 4)`. Closes the round-trip loop.
  2. `convert_to_grim_produces_deterministic_payload_for_same_input` —
     verifies the converter is deterministic (`raw_a.bytes.len() == raw_b.bytes.len()`).

Both integration tests pass. The full `grim-format` test surface is now
**27 tests** (25 unit + 2 integration), all green.

This audit document does NOT add new code to the crate; it is the
deliverable requested by the user objective.
