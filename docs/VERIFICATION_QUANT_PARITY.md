# Quant Variant Feature Parity — Verification Protocol

## Context

`grimoire-backend-rocm` now has feature gates for quant families: `q2k`, `q3k`, `q4k`, `q5k`, `q6k`.
Each gate controls one family's kernel modules in `src/kernels/mod.rs`. Unconditional families
(`q8_0`, `fp8`, `mxfp4`, `iq`) stay on because they have cross-module symbol dependencies that
would require auditing every caller.

## Gates present

| Feature | Kernel modules controlled | Notes |
|---|---|---|
| `q2k` | `kernels::q2k_gemm` | Fused forward + backward, `block_q2_K` |
| `q3k` | `kernels::q3k_gemm` | Fused forward + backward, `block_q3_K` |
| `q4k` | `kernels::q4k_gemm` + `kernels::q4k_dequant` | Fused GEMM + standalone dequant |
| `q5k` | `kernels::q5k_gemm` | Fused forward + backward, `block_q5_K` |
| `q6k` | `kernels::q6k_gemm` | Fused forward + backward, `block_q6_K` |

Unconditional (always compiled):
- `q8_0_dequant` — KV-cache dequant baseline, `kv_dequant_attention.rs` hardcodes `row_bytes_q8_0`
- `fp8_standalone` + `fp8_gemm_rdna4` — first-class compute dtype, charon references fp8 symbols
- `mxfp4_gemm` + `mxfp_standalone` — primary 4-bit dtype on RDNA4, charon emits `grim_moe_fused_grouped_mxfp4`
- `iq_dequant` + `iq_gemm` — crowd-tier baseline for IQ2_XXS/Q3_XXS/Q4_XXS

## Verification steps

### Step 1: Clean-room compile per gate

```bash
# Baseline: no quant features at all (only rocm backend, no gated families)
cargo check -p grim-backend-rocm --no-default-features

# Single family
cargo check -p grim-backend-rocm --no-default-features --features q4k
cargo check -p grim-backend-rocm --no-default-features --features q5k
cargo check -p grim-backend-rocm --no-default-features --features q6k

# Multiple families
cargo check -p grim-backend-rocm --no-default-features --features q4k,q5k,q6k
cargo check -p grim-backend-rocm --no-default-features --features q2k,q3k,q4k,q5k,q6k
```

Expected: all succeed, no unused-crate warnings for gated modules when their feature is off.

### Step 2: Module presence check

For each gate `F` enabled, verify:

```bash
grep -rn "pub mod ${F}wick" src/kernels/mod.rs
# e.g. for q4k: should have `#[cfg(feature = "q4k")] pub mod q4k_gemm` and
#            `#[cfg(feature = "q4k")] pub mod q4k_dequant`
```

Current state (checked, lines in mod.rs):
- `q2k_gemm` — present, gated by feature q2k (line 51)
- `q3k_gemm` — present, gated by feature q3k (line 52)
- `q4k_gemm` — present, gated by feature q4k (line 54)
- `q4k_dequant` — present, gated by feature q4k (line 53)
- `q5k_gemm` — present, gated by feature q5k (line 55)
- `q6k_gemm` — present, gated by feature q6k (line 56)

### Step 3: Symbol presence check (charon + wmma siblings)

For each gated family, check whether charon.rs or wmma_gemm.rs emits a variant-specific symbol.
The gated families (q2k–q6k) are isolated GEMM paths — charon/wmma do NOT emit per-q2k..q6k
symbols. The symbol-emitting families (fp8, mxfp4, mxfp8, iq) are unconditional.

Verify with:

```bash
grep -c "grim_moe_fused_grouped_q2k" src/kernels/charon.rs  # expected 0
grep -c "grim_fused_dequant_gemm_q4k_wmma" src/kernels/wmma_gemm.rs  # expected 0
```

The gated families intentionally have no charon/wmma siblings — that's why they're safe to gate.

### Step 4: Cache type wiring (HsaoKernelCache)

HsaoKernelCache registers kernels via `kernels::source_asm::compute_kernel_source()` which
conditionally pushes each family's `KERNEL_SOURCE` based on `#[cfg(feature = "...")]` guards
in `source_asm.rs`.

Verify:

```bash
grep -A2 "q2k_gemm::KERNEL_SOURCE" src/kernels/source_asm.rs
grep -A2 "q4k_gemm::KERNEL_SOURCE" src/kernels/source_asm.rs
```

Each `push_str!` for a gated family must be inside a `#[cfg(feature = "F")]` block.

### Step 5: Test that expected tests are gated

Tests in `src/kernels/{family}/*.rs` should be compiled only when the feature is on.
Verify:

```bash
# Without q4k, q4k_dequant tests should not be compiled
cargo test -p grim-backend-rocm --no-default-features -- q4k --no-run 2>&1 | grep -c "q4k"
# With q4k, they should be
cargo test -p grim-backend-rocm --no-default-features --features q4k -- q4k --no-run 2>&1 | grep -c "q4k"
```

### Step 6: Full build with all gates

```bash
cargo check -p grim-backend-rocm --features q2k,q3k,q4k,q5k,q6k
cargo test -p grim-backend-rocm --features q2k,q3k,q4k,q5k,q6k --lib 2>&1 | tail -3
```

Expected: 323 tests pass (same as baseline, since gated tests are additive).

## What doesn't need gating

- `q8_0` — KV-cache dequant, every attention path uses it, gating would require auditing
  `kv_dequant_attention.rs`, `source_asm.rs`, `qkv_attention.rs`, and the test suite
- `fp8` — charon forward/backward reference `fp8e4m3_to_f32`, `grim_moe_fused_grouped_fp8`,
  `grim_fp8_gemm_rdna4`; gating would require reworking charon.rs dispatch table
- `mxfp4` — charon emits `grim_moe_fused_grouped_mxfp4` unconditionally; `mxfp4_gemm.rs`
  exports 5 kernel entry points used by `fused_linear_ce.rs` and `scythe_persistent.rs`
- `iq` — test suite exercises IQ dequant parity against `grim_quant::dequant_iq*`; gating
  would drop the parity tests

## Adding a new gate

To gate `mxfp8` (example):

1. Add `mxfp8 = []` to `[features]` in Cargo.toml
2. In `src/kernels/mod.rs`: `#[cfg(feature = "mxfp8")] pub mod mxfp8_gemm` (create module if absent)
3. In `src/kernels/source_asm.rs`: wrap `push_str!(mxfp8_gemm::KERNEL_SOURCE)` in `#[cfg(feature = "mxfp8")]`
4. In `src/kernels/charon.rs`: wrap `grim_moe_fused_grouped_mxfp8` symbol in `#[cfg(feature = "mxfp8")]`
   — only if charon currently emits it unconditionally; check first
5. In `src/kernels/wmma_gemm.rs`: wrap any `grim_fused_dequant_gemm_mxfp8_wmma` in `#[cfg(feature = "mxfp8")]`
   — only if wmma currently emits it unconditionally; check first
6. Run verification steps 1–6 above
7. Add row to parity table in this doc

## Current verification status

- [x] Cargo.toml features added: `q2k`, `q3k`, `q4k`, `q5k`, `q6k`
- [x] `src/kernels/mod.rs` modules gated by `#[cfg(feature = "...")]`
- [x] `--no-default-features` compiles clean (no gated symbols referenced)
- [x] `--features q4k,q5k` compiles
- [x] `--features q2k,q3k,q4k,q5k,q6k` compiles
- [ ] source_asm.rs `#[cfg]` guards for each gated family (verify each push_str! is guarded)
- [ ] charon.rs — confirm no per-q2k..q6k symbol emissions (should be none)
- [ ] wmma_gemm.rs — confirm no per-q2k..q6k symbol emissions (should be none)
- [ ] HsaoKernelCache key registration guarded per family in source_asm.rs
- [ ] Test gating: confirm tests in each family module only compile when feature on
- [ ] Full `cargo test --features q2k,q3k,q4k,q5k,q6k` passes (323 baseline + additive)

## Date

Verified: 2026-08-21 (gfx1036 RDNA2, ROCm nightly)
