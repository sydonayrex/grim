# MoE Autotuning Kernel Mechanism — Design

Status: DESIGN, post-v3. Assumes `charon_kernel_plan_v3.md` is implemented
and running. Every "what already exists" claim below was verified directly
against the `crates.zip` upload in this session before being relied on —
this design is primarily composition of existing, proven infrastructure,
not a new subsystem. Exactly one genuinely new piece is called out
explicitly in §3.

---

## 0. Starting point: most of this already exists

Before designing anything, the actual substrate was checked directly
against source rather than assumed:

| Sub-problem | Existing mechanism | Verified |
|---|---|---|
| "Which launch config for this (kernel, shape) on this GPU?" | `Autotuner::get_or_tune(KernelKey, bench_fn)` — read-through cache, `record`/`lookup` | Real, `grim-backend-rocm/src/autotune.rs` |
| "Persist tuned configs across restarts" | `Autotuner::to_json_bytes`/`from_json_bytes`, `{cache_dir}/{gpu_arch}.json` | Real — one caveat, §2 |
| "Compile a kernel variant once, reuse the binary" | `HsacoKernelCache`, `seahash`-of-source keyed, on-disk `.hsaco` | Real, `grim-backend-rocm/src/kernels/jit_cache.rs` |
| "Switch between kernel *variants* at runtime based on live conditions" | `CharonSelector` — hold-counter de-sync guard, confirmed fixed for alternating-challenger thrashing earlier this session | Real, `charon.rs` |
| "Dispatch without a host round-trip" | `ScytheRing`/`ScytheTaskDescriptor`, opcode 6 (Charon v3's WI-Charon-3) | Real mechanism; opcode 6 is v3's addition |
| "Static tile-size lookup for known shapes" | `gemm_tuning.rs::lookup_gemm_config`/`resolve_gemm_solution` | Real, exists for dense GEMM |

Given all of this exists and is proven, the design below is primarily
wiring these four pieces together correctly, not inventing new mechanism.

---

## 1. Three layers, matched to three different timescales

### Layer 1 — Compile-time variant space (fixed at build, JIT'd on first use)

For each of Charon's kernel families (7 quant variants × {scalar, WMMA once
WI-Charon-2 lands} × {sortless, grouped}), the HIP-C source already exists
as a string constant. Nothing new needed beyond what v3 already scopes —
`HsacoKernelCache` already handles compile-once/reuse-forever correctly,
keyed on source hash (`seahash`), so adding WMMA variants as new source
strings slots into this layer with zero new caching logic.

### Layer 2 — Per-shape launch config (tuned once per shape, persisted)

This is where `Autotuner` does the real work, and it needs exactly one
MoE-specific extension: `KernelKey`'s `(m, n, k)` triple doesn't naturally
capture MoE's actual shape-determining variables — `num_experts`, `top_k`,
`hidden`, `inter`, and critically, **routing skew** (already computed as
`routing_skew()` in `charon.rs`'s `WaveCostModel`, confirmed real). A dense
GEMM's `(m,n,k)` is stable for a given batch size; a MoE dispatch's
*effective* per-expert shape varies with routing skew even at fixed batch
size, so autotuning must key on more than `(m,n,k)`.

**Concretely**: add a parallel `MoeKernelKey { kernel, gpu_arch, hidden,
inter, num_experts, top_k, skew_bucket }`, where `skew_bucket` is a coarse
quantization of `routing_skew()`'s continuous output (4-8 buckets — fine
enough to catch the low-skew/high-skew regime distinction `CharonSelector`
already reasons about, coarse enough that the tuning cache doesn't explode
into one entry per infinitesimally different skew value). Reuse
`Autotuner`'s `get_or_tune`/persistence machinery unchanged — this is a new
key type, not a new cache mechanism. Leave the existing dense-GEMM
`KernelKey` untouched.

### Layer 3 — Runtime variant selection (per-dispatch, no persistence, adapts within a session)

This is `CharonSelector`'s job already, and it's the right layer for it —
Layer 2 answers "what's the best config for *this* shape, once measured,"
but MoE routing skew shifts *within* a single inference session (different
prompts route differently), so the variant picked per-call needs the live,
low-overhead, no-host-sync mechanism `CharonSelector` already is, not a
full autotune-and-persist cycle every time skew shifts. The de-sync-guard
fix confirmed earlier this session is what makes this safe to run at real
dispatch frequency without thrashing between variants.

---

## 2. Correctness constraint carried from source, not left implicit

`Autotuner::from_json_bytes` `Box::leak`s every restored key's `kernel`/
`gpu_arch` strings to satisfy the `&'static str` requirement on `KernelKey`
— confirmed directly in source. This is a bounded, one-time cost acceptable
for loading a modest tuned-config file once at engine init, but becomes an
unbounded leak if called repeatedly (e.g., per-request or per-inference-
step). **Design constraint: `from_json_bytes` is called exactly once at
startup, or on first-encounter of a new GPU arch — never on the hot path.**

---

## 3. The one genuinely new piece: bridging offline tuning into the live selector

`CharonSelector`'s `default_variant_table()` currently ships fixed
`VariantRow` bucket boundaries derived from `WaveCostModel`'s coefficients
— explicitly marked as unvalidated priors, confirmed earlier this session.
The actual new work this design requires: **`Autotuner`-measured results
should replace those priors, not sit alongside them as a second,
disconnected mechanism.**

Concretely: a new, small function —

```
fn build_variant_table_from_autotuner(
    tuner: &Autotuner,
    gpu_arch: &str,
) -> Vec<VariantRow>
```

— that, for each `CharonVariant`, looks up its best measured
`AutotuneConfig` per skew-bucket (Layer 2) and derives the bucket boundary
from *measured* crossover points between variants, rather than
`WaveCostModel`'s guessed coefficients. This is the genuine "autotuning
kernel mechanism" this design exists to build. Everything else here is
composition of proven infrastructure; this is the actual missing link
between "we measured what's fastest offline" and "the live selector uses
what we measured."

---

## 4. JIT's actual role, stated precisely

JIT compilation itself is not the autotuning mechanism —
`HsacoKernelCache` already does compile-once-cache-forever correctly, and
that's orthogonal to *which* variant gets selected. JIT's real relevance is
narrower: **specializing a kernel's source string per-shape at compile
time** (baking `num_experts`/`hidden`/`inter` in as compile-time constants
rather than runtime kernel arguments, letting the HIP compiler unroll loops
and eliminate bounds checks) is a legitimate autotuning *lever* — one more
axis `Autotuner` could tune over (specialized-JIT vs. generic-kernel-with-
runtime-args), not the tuning mechanism itself. If pursued, it's an
extension of Layer 1 (one more source-string variant per common shape, same
`HsacoKernelCache` compile-cache handles it identically to the existing 7
quant variants) — not a new subsystem.

---

## 5. Explicitly out of scope for a first version

**Full autotuning-at-model-load** (benchmark every kernel/shape/skew-bucket
combination before first inference) is expensive and the wrong default —
`Autotuner::get_or_tune`'s read-through design already handles this
correctly via lazy population: the first request for a novel shape pays
the bench cost once and gets cached from then on. A design that tries to
pre-warm everything at startup would be working against the grain of the
mechanism that already exists, not with it.

---

## 6. Summary

| Layer | Timescale | Mechanism | New work required |
|---|---|---|---|
| 1 — kernel source variants | Build time, JIT'd once | `HsacoKernelCache` | None (v3's WMMA variants slot in for free) |
| 2 — per-shape launch config | Tuned once per shape, persisted | `Autotuner` + new `MoeKernelKey` | New key type only, reuses all persistence/lookup logic |
| 3 — live variant selection | Per-dispatch, adapts within a session | `CharonSelector` | None (already correct, already fixed for thrashing) |
| Bridge — offline measurement into live selection | Built once, refreshed per tuning run | `build_variant_table_from_autotuner` | **New — the actual missing piece** |
