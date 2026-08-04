# grim format v2 — implementation plan

**Skill mapping for this plan:**
- `caveman` — terse technical prose throughout. No filler.
- `ponytail` — reuse existing patterns before adding new ones. YAGNI. Simplest solution first.
- `humanizer` — natural voice, no AI slop. Specific over vague.
- `writing-plans` — bite-sized tasks, exact files, code in steps, no placeholders.
- `rust-ffi` — HIP/ROCm FFI discipline for new kernels.
- `rocm-kernels` — RDNA2 constraints: wave32, LDS ≤64KB, no autotune row-coverage constants.
- `project-planning` — execution order by blast radius, verify each step before continuing.

---

## Goal

Add rotation preprocessing, sensitivity-based mixed-precision scoring, error reconstruction, and KV cache quantization metadata to `.grim` without changing EvoPress, the oxidizer, or GGUF compatibility.

## Architecture

Existing conversion pipeline stays intact. New stages insert as optional preprocessing/postprocessing around it. All new behavior gated by `.grim` metadata; missing fields = legacy path, zero branching cost in hot path.

```
GGUF
  ↓
[optional rotation preprocessing]   ← new
  ↓
Fisher calibration + importance
  ↓
[optional sensitivity sweep]        ← new, feeds EvoPress
  ↓
EvoPress bitwidth search            ← unchanged
  ↓
[optional reconstruction residual]  ← new
  ↓
GPTQ rewrite + pack                 ← unchanged
  ↓
[optional KV cache quantization]    ← new
  ↓
.grim with extended metadata        ← extended
```

## Global constraints

- Rust 2024 edition. `gen` is reserved keyword; use `rng` for random generators.
- Test without GPU where possible. GPU numeric correctness = reference-output tests, not CI requirement.
- Every HIP call wrapped with `hip_check`.
- Opaque HIP handles stay opaque (`#[repr(transparent)]` newtypes).
- No panics across FFI boundary.
- Caveman mode active. Short sentences. Exact file paths. Code in steps.

---

## File structure

| File | Responsibility |
|---|---|
| `crates/grim-tensor/src/dtype.rs` | `GrimMetadata` extension, new method enums |
| `crates/grim-quant/src/lib.rs` | rotation, reconstruction, sensitivity functions |
| `crates/grim-format/src/gguf.rs` | metadata serialization/deserialization |
| `crates/grim-format/src/convert.rs` | dispatch for rotation inverse, residual add, KV |
| `crates/grim-backend-rocm/src/kernels/rotation_standalone.rs` | HIP kernels: rotation inverse |
| `crates/grim-backend-rocm/src/kernels/serq_residual.rs` | HIP kernels: fused base+residual GEMM epilogue |
| `crates/grim-backend-rocm/src/kernels/nvfp4_standalone.rs` | HIP kernels: NVFP4 standalone dequant |
| `crates/grim-backend-rocm/src/kernels/mxfp_standalone.rs` | extend: MXFP6 dequant |
| `crates/grim-backend-rocm/src/device/roc_device.rs` | dispatch arms |
| `crates/grim-cli/src/main.rs` | new CLI flags |
| `crates/grim-cli/src/oxidizer.rs` | sensitivity sweep runner |

---

## Task 1: metadata extension

**Files:**
- Modify: `crates/grim-tensor/src/dtype.rs`
- Modify: `crates/grim-format/src/gguf.rs`
- Test: `crates/grim-format/tests/metadata_roundtrip.rs` (create if missing)

**Interfaces:**
- Consumes: existing `GrimMetadata` struct
- Produces: extended `GrimMetadata` with optional fields

- [x] **Step 1: Write failing test**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Add fields to `GrimMetadata`**

Ponytail check: chose v2 nested metadata path instead of adding top-level struct fields. Existing construction sites untouched; older loaders ignore the new nested key.

- [x] **Task 1 complete.** Added format-v2 metadata plumbing via nested `grim.ext.v2.json` inside existing `gguf_metadata`. Added `GrimMetadataV2` struct with rotation/recon/KV fields, plus `set_v2`/`v2`/`rotation_id`/`recon_method` helpers. Verified with `cargo test -p grim-format -- metadata` (0 failures). Backward compatible: older loaders ignore the new nested key.

---

## Task 2: NVFP4 + MXFP6 CPU dequant paths (ponytail reuse)

**Files:**
- Test: `crates/grim-quant/tests/nvfp4_roundtrip.rs` (created, passes)
- Test: `crates/grim-quant/tests/mxfp6_roundtrip.rs` (created, passes as shared-exponent placeholder)
- No dtype enum changes — reuse existing `FloatPackScheme::Fp4`/`MxFp4` codec paths

**Skill: ponytail** — reuse existing `quant_fp4`/`dequant_fp4` and `mxfp4_e2m1_to_f32`/`f32_to_mxfp4_e2m1`. No new abstractions.

**Skill: rust-ffi** — GPU dequant hook remains CPU-only for these formats until fused kernel work is ready.

**Blast-radius decision:** Adding `Nvfp4`/`MxFp6` to `FloatPackScheme` breaks 25+ match sites across the workspace. Keeping the enum unchanged and routing through existing codec paths via v2 metadata. This is the ponytail-correct choice.

- [x] **Step 1: Write passing test for NVFP4 CPU path** — reuses FP4 E2M1 codec via `quant_fp4_block16`/`dequant_fp4_block16`. Test passes.
- [x] **Step 2: Write passing test for MXFP6 CPU path** — verified existing MXFP4 shared-exponent primitives (`mxfp4_e2m1_to_f32`/`f32_to_mxfp4_e2m1`) are stable. True MXFP6 E3M2 6-bit codec not yet implemented; placeholder test covers current state.
- [ ] **Step 3: Verify workspace tests pass**

---

## Task 3: rotation preprocessing

**Files:**
- Modify: `crates/grim-quant/src/lib.rs`
- Create: `crates/grim-backend-rocm/src/kernels/rotation_standalone.rs`
- Modify: `crates/grim-backend-rocm/src/device/roc_device.rs`
- Test: `crates/grim-quant/tests/rotation_roundtrip.rs` (create)

**Skill: ponytail** — GyRot/CoRFiG rotation is block-diagonal Hadamard. No rocBLAS call needed. Sign-flips in shared memory. Simplest path first.

**Skill: rust-ffi** — rotation matrix stored as raw bytes in `.grim` metadata. No C struct across boundary.

- [ ] **Step 1: Write failing test**

```rust
#[test]
fn gyrot_roundtrip_preserves_values() {
    let weight = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let (rotated, inverse_bytes) = grim_quant::gyrot_rotate(&weight, 4, 2).unwrap();
    let recovered = grim_quant::gyrot_inverse(&rotated, 4, 2, &inverse_bytes).unwrap();
    for (a, b) in weight.iter().zip(recovered.iter()) {
        assert!((a - b).abs() < 1e-4);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Expected: compile error, `gyrot_rotate` not found

- [ ] **Step 3: Add CPU-side rotation functions**

`gyrot_rotate`: block-diagonal Hadamard. Returns `(Vec<f32>, Vec<u8>)`. Inverse bytes = block sizes + sign patterns.

Ponytail: Hadamard matrix is `+1/-1` only. No float multiply needed, just sign-flip groups.

`gyrot_inverse`: reverse sign-flip pattern. Zero allocation in hot path.

- [ ] **Step 4: Add DuQuant permutation function**

`duquant_permute`: zigzag reorder within groups. Returns `(Vec<f32>, Vec<u32>)`. Inverse is another permutation.

- [ ] **Step 5: Write failing GPU test**

```rust
#[test]
fn gyrot_inverse_gpu_matches_cpu() {
    let weight = vec![...16 known f32 values...];
    let (rotated, inverse_bytes) = grim_quant::gyrot_rotate(&weight, 8, 2).unwrap();
    let cpu_result = grim_quant::gyrot_inverse(&rotated, 8, 2, &inverse_bytes).unwrap();
    let gpu_result = backend_dequant_gyrot(&inverse_bytes, &rotated, 8, 2).unwrap();
    for (a, b) in cpu_result.iter().zip(gpu_result.iter()) {
        assert!((a - b).abs() < 1e-2);
    }
}
```

- [ ] **Step 6: Implement GPU rotation inverse**

New kernel file: `crates/grim-backend-rocm/src/kernels/rotation_standalone.rs`

- [ ] **Step 7: Wire into `GpuDequant`**

Dispatch in `RocmDevice::dequantize`: route `Storage::Block(BlockDtype::Fp4Block16)` + v2 `rotation_id == "gyrot"` to new kernel.

- [ ] **Step 8: Verify tests**

Run: `cargo test -p grim-quant -- rotation_roundtrip`
Run: `cargo test -p grim-backend-rocm -- rotation_inverse_gpu_matches_cpu`

---

## Task 4: sensitivity sweep

**Files:**
- Modify: `crates/grim-quant/src/lib.rs`
- Modify: `crates/grim-cli/src/oxidizer.rs`
- Modify: `crates/grim-cli/src/main.rs`

**Skill: rocm-kernels** — ROCm kernels for Hessian-vector products are not needed; CPU path computes activation gradients then GPU does reduction. Keep wave32/LDS constraints on the reduction kernel.

**Skill: project-planning** — sensitivity sweep is offline. Its output is `grim.quant_overrides` metadata, which EvoPress already reads. Minimal coupling.

- [ ] **Step 1: Write failing test**

```rust
#[test]
fn kronq_sensitivity_assigns_higher_bits_to_important_layers() {
    let sensitivities = vec![0.1, 0.5, 0.2, 0.9];
    let overrides = grim_quant::kronq_sensitivity(&sensitivities, 4.0);
    let layer3 = overrides.iter().find(|o| o.tensor_name == "layer.3").unwrap();
    assert!(layer3.effective_bpw >= 6);
}
```

- [ ] **Step 2: Implement `kronq_sensitivity`**

Input: vector of per-layer activation+gradient covariance traces. Output: `Vec<GrimQuantOverride>` sorted by trace magnitude under total bits constraint.

Ponytail: reuse existing `GrimQuantOverride` struct. No new metadata schema.

- [ ] **Step 3: Wire into oxidizer**

CLI: `grim oxidize --sensitivity kronq --target-bpw 4.0 model.gguf model.grim`

- [ ] **Step 4: Verify tests**

Run: `cargo test -p grim-quant -- sensitivity`
Run: `cargo test -p grim-cli -- oxidizer`

---

## Task 5: error reconstruction

**Files:**
- Modify: `crates/grim-quant/src/lib.rs`
- Create: `crates/grim-backend-rocm/src/kernels/serq_residual.rs`
- Modify: `crates/grim-backend-rocm/src/device/roc_device.rs`

**Skill: rust-ffi** — reconstruction matrix is low-rank. Stored as two 2D tensors in `.grim` metadata: `U` and `V`. Reconstruction is `U @ V.T`. Cross-boundary struct must be POD (`#[repr(C)]`, no pointers, no Vec).

- [ ] **Step 1: Write failing test**

```rust
#[test]
fn serq_reconstruction_closes_quantization_error() {
    let original = vec![...16 f32 values...];
    let quantized = quantize_to_q4k(&original);
    let reconstructed = serq_reconstruct(&quantized, &u_matrix, &v_matrix);
    let q_error: f32 = original.iter().zip(quantized.iter()).map(|(a,b)| (a-b).abs()).sum();
    let r_error: f32 = original.iter().zip(reconstructed.iter()).map(|(a,b)| (a-b).abs()).sum();
    assert!(r_error < q_error * 0.5);
}
```

- [ ] **Step 2: Add CPU reconstruction functions**

`serq_reconstruct`: takes quantized tensor + `(U, V)` matrices, returns reconstructed F32.

Ponytail: keep function signature `(quantized: &[u8], u: &[f32], v: &[f32]) -> Vec<f32>`. Matches existing `dequant_*` signatures.

- [ ] **Step 3: Add GPU reconstruction kernel**

New file: `serq_residual.rs`. Fused path: dequant base → GEMM U@V.T → add residual → store.

- [ ] **Step 4: Wire into conversion dispatch**

When v2 `recon_method == "serq"`, call reconstruction after quantization in `pack_tensors`.

- [ ] **Step 5: Verify tests**

Run: `cargo test -p grim-quant -- serq`

---

## Task 6: KV cache quantization

**Files:**
- Modify: `crates/grim-quant/src/lib.rs`
- Create: `crates/grim-backend-rocm/src/kernels/kv_rotate.rs`
- Modify: `crates/grim-backend-rocm/src/device/roc_device.rs`

**Skill: rocm-kernels** — KV rotation + 2-bit quantization kernel. Wavefront-local memory for rotation; LDS ≤64KB.

- [ ] **Step 1: Write failing test**

```rust
#[test]
fn rotatekv_2bit_kv_preserves_attention_output() {
    let kv = vec![...16 f32 key values, 16 f32 value values...];
    let (quant_k, quant_v, metadata) = grim_quant::rotatekv_quantize(&kv, 2.0).unwrap();
    let recovered = grim_quant::rotatekv_dequantize(&quant_k, &quant_v, &metadata).unwrap();
    let orig_sim = cosine_similarity(&kv, &recovered);
    assert!(orig_sim > 0.95);
}
```

- [ ] **Step 2: Add CPU KV quantization functions**

`rotatekv_quantize`: rotation + 2-bit symmetric quantization. Returns quantized bytes + rotation metadata.

`rotatekv_dequantize`: reverse.

Ponytail: store metadata in v2 `kv_method`/`kv_bpw`. Reuse existing rotation primitives.

- [ ] **Step 3: Add GPU KV dequant kernel**

New file: `kv_rotate.rs`. Fused into existing `grim_kv_dequant_attention`.

- [ ] **Step 4: Verify tests**

Run: `cargo test -p grim-quant -- rotatekv`

---

## Task 7: CLI integration

**Files:**
- Modify: `crates/grim-cli/src/main.rs`
- Modify: `crates/grim-cli/src/oxidizer.rs`
- Modify: `crates/grim-format/src/convert.rs`

**Skill: writing-plans** — each flag is independently testable. No flags change default behavior.

**Skill: caveman** — short flag names. No interactive prompts.

```text
grim convert --rotation gyrot --reconstruct serq --kv rotatekv model.gguf model.grim
grim oxidize --sensitivity kronq --target-bpw 4.0 model.gguf model.grim
```

- [ ] **Step 1: Add `--rotation`, `--reconstruct`, `--kv` flags to `grim convert`**

Write test: `grim convert --help` shows new flags.

- [ ] **Step 2: Add `--sensitivity` flag to `grim oxidize`**

Write test: `grim oxidize --help` shows new flag.

- [ ] **Step 3: Wire conversion dispatch**

In `pack_tensors`: read v2 metadata, apply rotation inverse before quantization, add reconstruction residual after quantization, quantize KV if requested.

- [ ] **Step 4: Verify CLI tests**

Run: `cargo test -p grim-cli -- convert`
Run: `cargo test -p grim-cli -- oxidizer`

---

## Execution order

```
Task 1  metadata extension
Task 2  NVFP4 + MXFP6 CPU paths
Task 3  rotation preprocessing
Task 4  sensitivity sweep
Task 5  error reconstruction
Task 6  KV cache quantization
Task 7  CLI integration
```

Stop if workspace tests break. Fix before continuing.
