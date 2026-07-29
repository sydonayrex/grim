# file_fix.md — Implementation Plan for EvoPress / GPTQ / SpQR / AWQ / OBQ Updates

## Context

`grim_formats_evopress_ceiling.md` identified 5 improvement areas. Review against actual code (`grim-quant`, `grim-format`, `grim-tensor`, `grim-backend-cpu`, `grim-backend-rocm`) shows:

- **Part 1 (Wiring WeightFormat)** — ALREADY DONE. `weight_format.rs` has all 7 codecs; wired into `TrainingJob` + `ConvertModelRequest`.
- **Part 3 (MXFP8 on RDNA2)** — ALREADY DONE. `QuantMode::MxFp8Emulated` ("Jackdaw") + `MxFp4Emulated` ("Rook") exist in `quantization.rs:129`; `resolve_quant_mode` gates `Fp8Native` on RDNA4 only.
- **Part 2** — EvoPress GA loop EXISTS in `grim-quant/src/lib.rs:2480` (`EvoPressConfig`, `Individual`, `evopress_search`, `tournament_select`, `crossover`, `eval_individual`) but is NOT wired into `convert.rs`. `eval_individual` uses importance-weighted BPW matching (not perplexity). GPTQ uses diagonal Fisher only (`apply_block_diagonal_update`, no Cholesky/inverse).

This plan covers the 5 genuine gaps from Part 2 that would substantially improve grim.

---

## Update 1: Wire `evopress_search` into `convert.rs`

### Problem
`evopress_search()` exists in `grim-quant` but `convert_to_grim()` in `convert.rs` only accepts `evopress_bitwidths: Option<Vec<u32>>` — a static per-tensor bitwidth list. The GA never runs during conversion.

### Files
- `crates/grim-format/src/convert.rs` (primary)
- `crates/grim-quant/src/lib.rs` (read `evopress_search`, `EvoPressConfig`, `randomized_svd_importance`)

### Implementation
1. In `convert_to_grim()`, when `generations > 0` AND `evopress_bitwidths` is None:
   - Call `randomized_svd_importance()` on each tensor to get importance scores (already exists, `lib.rs:2068`).
   - Build `EvoPressConfig { generations, population_size: 128, target_bpw, ..Default::default() }`.
   - Call `evopress_search(&config, &importance_scores, &tensor_sizes)`.
   - Use returned `Vec<u32>` as the per-tensor bitwidths (same path as the existing `evopress_bitwidths` override).
2. Pass `generations` through `build_entries_from_source` → `pack_tensors` (currently `pack_tensors` takes `evopress_bitwidths: Option<Vec<u32>>` — wire the GA result into this).

### Verification
```
cargo test -p grim-format --lib convert
# New test: test_convert_runs_evopress_when_generations_set
# Asserts: evopress_search called, bitwidths differ per tensor, quant_method = "evopress-gptq"
```

### Risk: Low. Pure wiring. No new algorithms.

---

## Update 2: GPTQ Hessian (Cholesky/inverse) — replace diagonal Fisher

### Problem
`apply_block_diagonal_update` (`grim-quant/src/lib.rs:1813`) uses `block_curvature` as a diagonal. True GPTQ uses the inverse Hessian across columns to propagate quantization error to neighboring weights — the source of GPTQ's quality advantage.

### Files
- `crates/grim-quant/src/lib.rs` (primary)
- `crates/grim-format/src/spec.rs` (add `HessianBlock` to `GrimTensorExt` if needed)
- `crates/grim-backend-cpu/src/dequant_gemm.rs` (read `gptq_ordered` flag — already exists)

### Implementation
1. Add `HessianBlock { cholesky_factor: Vec<f32>, block_cols: usize }` as an optional field in `TensorRewritePlan`.
2. In `prepare_row_with_sequential_update`, when `HessianBlock` is present:
   - Replace `error * curvature[i]` (diagonal) with `H_inv @ error` (Cholesky solve).
   - Use `lapacke` or a hand-rolled Cholesky (grimoire has no LAPACK dep yet — check `Cargo.toml`; if absent, add `cholesky = "0.10"` or implement a 256×256 Cholesky inline since block_cols ≤ 256).
3. Set `gptq_ordered: 1` in the output `GrimTensorExt` when Hessian path is used (kernel already checks this flag in `dequant_gemm.rs:81`).

### Verification
```
cargo test -p grim-quant --lib quant
# New test: test_gptq_hessian_vs_diagonal_quality
# Construct a 256×256 weight block, quantize with diagonal vs Hessian,
# assert Hessian version has lower reconstruction error (MSE < 0.95× diagonal).
```

### Risk: Medium. Cholesky on 256×256 blocks is numerically sensitive. Gate behind `cfg(test)` first, then production.

---

## Update 3: OBQ row ordering

### Problem
`prepare_row_with_sequential_update` goes left-to-right. OBQ (Frantar et al., 2022) quantizes columns in order of increasing Hessian diagonal — hardest weights last — reducing error accumulation.

### Files
- `crates/grim-quant/src/lib.rs`

### Implementation
1. Before the sequential pass in `prepare_row_with_sequential_update`, sort row indices by `curvature` ascending.
2. Process rows in sorted order; write results back to original positions.

### Verification
```
cargo test -p grim-quant --lib quant
# New test: test_obq_row_ordering_reduces_error
# Same block, diagonal curvature, compare MSE of left-to-right vs sorted.
```

### Risk: Low. Pure reordering. ~10 lines.

---

## Update 4: SpQR sparse residuals

### Problem
SpQR (Dettmers 2023) keeps ~1% of weights with largest Hessian sensitivity in FP16, stores the rest in INT4. Consistently beats EvoPress at same bit budget. Neither Unsloth nor Axolotl implements this for ROCm.

### Files
- `crates/grim-format/src/spec.rs` (add `SparseResidual` to `GrimTensorExt`)
- `crates/grim-quant/src/lib.rs` (compute salient weights, emit residual)
- `crates/grim-backend-cpu/src/dequant_gemm.rs` (read + apply residual in dequant)
- `crates/grim-backend-rocm/src/kernels/q4k_dequant.rs` (add ROCm path — optional, can start CPU-only)

### Implementation
1. Add to `GrimTensorExt`:
   ```rust
   pub struct SparseResidual {
       pub indices: Vec<u32>,   // FP16-stored weight indices
       pub values: Vec<f16>,   // FP16 values
   }
   pub sparse_residual: Option<SparseResidual>,
   ```
2. In `rewrite_tensor_data`, when `sparsity_threshold` is set:
   - Compute per-weight Hessian diagonal (reuse `curvature` from `TensorRewritePlan`).
   - Select top-1% indices by curvature.
   - Quantize remaining 99% to INT4 via existing `quant_packed_symmetric`.
   - Store salient weights as `SparseResidual`.
3. In dequant path (`dequant_gemm.rs`), after INT4 dequant, add `sparse_residual.values` back at `sparse_residual.indices`.

### Verification
```
cargo test -p grim-quant --lib quant
cargo test -p grim-backend-cpu --lib dequant
# New test: test_spqr_residual_reconstruction
# Quantize block with sparsity_threshold=0.01, dequant, assert MSE < 0.5× non-sparse.
```

### Risk: High effort (~500 LOC across 4 files) but highest quality win. Defer ROCm kernel — CPU path first.

---

## Update 5: AWQ channel scaling

### Problem
AWQ scales weight channels by inverse activation magnitudes before quantizing. Grim has `importance` weights per tensor — if sourced from per-channel activation stats (not Fisher/SVD), the existing `fit_block_quantization` path picks up the AWQ benefit.

### Files
- `crates/grim-quant/src/lib.rs` (`randomized_svd_importance` — change source)
- `crates/grim-format/src/convert.rs` (accept calibration dataset)

### Implementation
1. Add `calibration_dataset: Option<&str>` param to `convert_to_grim` (already has `dataset: Option<&str>` — verify it's wired to importance computation).
2. In `pack_tensors`, when `dataset` is provided:
   - Run a forward pass on the calibration dataset.
   - Compute per-channel activation magnitude (mean of |activation| per output channel).
   - Use `1.0 / (activation_magnitude + epsilon)` as the `importance` weights.
   - Pass to `TensorRewritePlan.importance`.
3. The existing `quant_packed_symmetric(data, bits, importance, curvature, shape)` already uses `importance` for scale selection — no change needed to the quantizer itself.

### Verification
```
cargo test -p grim-format --lib convert
# New test: test_awq_channel_scaling_from_calibration
# Calibrate on a small dataset, assert importance weights differ from SVD-based.
```

### Risk: Medium. Requires a calibration forward pass (model loading). Start with a stub that reads pre-computed activation stats from a JSON sidecar.

---

## Execution Order

| # | Update | Effort | ROI | Risk |
|---|--------|--------|-----|------|
| 1 | Wire EvoPress GA | S | High | Low |
| 2 | GPTQ Hessian | M | High | Medium |
| 3 | OBQ row ordering | S | Medium | Low |
| 4 | SpQR sparse residuals | L | Very High | High |
| 5 | AWQ channel scaling | M | Medium | Medium |

## Final Verification (all updates)

```
cargo test --workspace
# Full suite remains green.
cargo test -p grim-quant --lib quant
# New tests for Hessian, OBQ, SpQR pass.
cargo test -p grim-format --lib convert
# EvoPress GA + AWQ calibration tests pass.
# End-to-end: convert a 7B model with --generations 50 --calibration dataset.json
# Verify output .grim file has per-tensor bitwidths in GrimTensorExt,
# gptq_ordered=1, and sparse_residual present when --sparsity 0.01.
```

## Files Modified (summary)

- `crates/grim-format/src/convert.rs` — Updates 1, 5
- `crates/grim-quant/src/lib.rs` — Updates 1, 2, 3, 4
- `crates/grim-format/src/spec.rs` — Update 4 (SparseResidual, HessianBlock)
- `crates/grim-backend-cpu/src/dequant_gemm.rs` — Update 4 (apply residual)
- `crates/grim-backend-rocm/src/kernels/q4k_dequant.rs` — Update 4 (ROCm residual path, optional)
