# t-pain: Unsloth-style autotuning for grim-backend-rocm — Rust-only, TDD-driven

**Goal:** give grim ROCm kernels the Unsloth autotuning discipline — pruned candidate-space sweep + per-launch benchmarked selection + cross-run persistence — implemented entirely in Rust, with Red-Green-Refactor TDD for every kernel round. No Python, no Triton, no Python benchmark harness, no Python glue.

**Audience:** grim ROCm backend team. Target arch: gfx1036 (RDNA2, W64) primary; W32 iGPU secondary.

**Two invariant constraints for this whole plan:**
1. **All code is Rust.** Every new file, every test, every benchmark harness, every config generator, every pruning rule, every persistence layer, every CI gate — Rust. The only non-Rust artifacts are the existing HIP source literals already in the kernel files (those stay as `pub const KERNEL_SOURCE: &str = r#"..."#`). We do not add Python, we do not add a Python benchmark harness, we do not shell out to Python for tuning.
2. **TDD for every round.** Every behavioral change gets a failing or freshly-meaningful test first (Red), then the minimal code to satisfy it (Green), then cleanup (Refactor). We do not write autotune framework code before we have a test that demands it. Where a behavior already exists (e.g. the static lookup path), we write the test against that behavior first so we have a green baseline before we add the autotuned path.

---

## 1. What to steal from Unsloth (and what to leave) — Rust-only audit

### Steal (concepts, re-implemented in Rust)

| Unsloth piece | What it gives grim | Rust form |
|---|---|---|
| **Config-space generation** — cartesian product over tunable axes | A candidate set larger than one heuristic pick | Rust fns returning `Vec<LaunchConfig>`, one per kernel variant |
| **SMEM-capacity pruning** — `estimate_smem_reqs` / `exceeds_smem_capacity` | Reject configs that overflow LDS before touching the device | Rust fn `smem_cost(config, arch) -> Option<u32>` per kernel; pruner rejects `Some(cost)` where `cost > device_smem_limit` |
| **Mutual-exclusion pruning** — `permute_x && permute_y → drop` | Rule out logically broken combos without measurement | Rust: candidate generators enforce mutual exclusion at generation time; tests assert the forbidden combos are absent from the generated set |
| **Key design: exclude the fast-changing dimension** — Unsloth drops `NUM_TOKENS` from the key | Avoid recompilation-per-sequence-length | Rust: `CompileKey` is coarse (shape class, not exact m/n/k); `TuneKey` keeps exact shape for launch tuning |
| **Separate compile key from launch-config key** — Triton compiles once per (expert_count, n, k, permute_mode) | Two-level key hierarchy | Rust: `CompileKey` + `TuneKey`, mirroring the existing `Autotuner` cache shape |
| **Runtime benchmark + persistent cache** — Unsloth uses Triton's in-process cache; grim already has `Autotuner` with on-disk JSON | grim is ahead on persistence | Keep `Autotuner` as-is (Rust). Wire it to the new benchmarked path |

### Leave (do NOT import, do NOT port to Rust)

| Unsloth piece | Why not |
|---|---|
| Triton runtime / `@triton.jit` / `triton.autotune` / Python | Not Rust. grim is Rust+HIP. Do not add it. |
| Python benchmark harness | Not Rust. grim's benchmark harness is Rust + HIP timers. |
| TMA axes (USE_TMA_LOAD_X/W, USE_TMA_STORE) | TMA is Blackwell-only; on RDNA it's inert. grim's equivalent hardware-feature flags are WMMA availability (`__gfx1100__`) and fp8/MFMA availability (`__gfx1200__`). Model those flags in Rust, not TMA. |
| `calculate_settings(n)` heuristic for elementwise kernels (norm, rope, geglu, swiglu) | Those kernels aren't grim's bottleneck and aren't standalone HIP kernels in grim-backend-rocm today. Don't autotune what you don't have. If grim adds standalone norm/rope/activation HIP kernels later, revisit — in Rust. |
| NUM_INT32_ELEMENTS / LONG_INDEXING safety path | Triton index-overflow guard for >2^31 elements. grim's HIP kernels already use `unsigned long long` for flat indices (charon, fused_dequant_gemm). Not needed. |

**All-Rust audit of the existing grim code we're building on:**
- `autotune.rs` — Rust ✓
- `gemm_tuning.rs` — Rust ✓
- `charon.rs`, `wmma_gemm.rs`, `fused_dequant_gemm.rs`, `qkv_attention.rs`, `cross_attention.rs`, `iq_gemm.rs`, `q4k_gemm.rs`, `scythe_persistent.rs` — Rust wrapper modules around HIP source literals ✓ (the HIP source is C/HIP inside a Rust string; that's existing and stays. The new autotune layer is pure Rust around it.)
- `accel_ffi.rs` — Rust FFI ✓
- `jit_cache.rs` — Rust ✓

So the existing codebase is already Rust-only. The plan keeps it that way. The only thing we add is more Rust.

---

## 2. TDD model for this plan (read this before writing any code)

TDD here is Red-Green-Refactor, but "Red" has two flavors in this codebase because we're adding behavior on top of existing behavior:

### Flavor A — Red test for NEW behavior (the common case)

For each new capability (candidate generation, pruning, benchmarking, cache hit/miss, launch integration):

1. **Red:** Write a test that asserts the NEW behavior. It fails because the code doesn't exist yet (compilation error → "function not there" → test panics/asserts-false). This is the real Red.
2. **Green:** Write the minimal code to make the test pass. Not the final design — the smallest thing that satisfies the test.
3. **Refactor:** Clean up naming, extract helpers, split modules, add the next test. Never skip this.

### Flavor B — Red test for BEHAVIOR THAT ALREADY EXISTS (baseline capture)

Before we add the autotuned path for a kernel, we write tests against the existing static path so we have a green, documented baseline:

1. **Red (often already green):** Write a test that captures the current behavior — e.g. "for shape X on gfx1036, `lookup_gemm_config` returns block_m=8, block_n=64, block_k=64". If the static path already does this, the test is green immediately. That's fine — it's a baseline test, not a failure.
2. **Green (baseline):** The static path is the green. We now have a documented, tested baseline.
3. **Refactor (into the autotune layer):** Write the NEW test that asserts the autotuned path produces the same result (or better), then implement the autotuned path, then optionally deprecate/remove the static path for that kernel.

This is important: **we never replace the static lookup with the autotuned path without a test proving the autotuned path matches or beats it.** The "red" for the replacement is the parity test; the "green" is the autotuned implementation; the refactor is wiring the launch to use the autotuned winner.

### The loop per kernel round

Each round (section 4) follows:

1. **Baseline tests (green or red→green):** tests against the existing static path / CPU reference for that kernel, on representative shapes. If they don't exist, write them first. These give us a green baseline.
2. **Framework tests (red→green):** tests for the autotune framework pieces needed by THIS kernel — candidate generation, pruning, key types, cache. Write the failing test, make it pass, refactor.
3. **Parity test (red→green):** test that the autotuned path produces the same output as the baseline (within tolerance). This is the Red that forces the autotuned implementation.
4. **Perf test (green, observational):** benchmark test that records the autotuned winner's time vs the baseline. Not a pass/fail gate in the unit-test sense — it's an observed number we assert on in CI or eyeball. But it still lives in a `#[test]` or a benchmark binary built by `cargo build`/`cargo test` — Rust, not Python.
5. **Integration test (red→green):** test that the kernel launch actually uses the autotuned config on cache hit and re-benchmarks on cache miss.

### What "Red" looks like concretely

- A `#[test]` that calls `charon_candidates(arch_gfx1036)` and asserts `len(candidates) == expected_count` — Red if the fn doesn't exist yet.
- A `#[test]` that asserts `candidates` contains no WMMA configs on gfx1036 — Red if the pruning rule isn't wired.
- A `#[test]` that asserts `Autotuner` returns `Err` or a specific cache-miss behavior when the key is absent — Red if the cache path isn't wired to the benchmark.
- A `#[test]` that calls the autotuned charon launch on a fixed input and asserts output ≈ CPU reference — Red if the autotuned launch isn't wired yet.

### What "Green" looks like concretely

- The minimal candidate generator that returns the expected count.
- The minimal pruning filter that removes the forbidden combos.
- The minimal benchmark harness stub that returns a default config (not the real benchmark yet — just enough to make the cache-hit test pass).
- The real benchmark harness once the parity test demands it.

### What "Refactor" looks like concretely

- Rename `KernelKey` → `CompileKey`/`TuneKey` split after the tests prove the split is needed.
- Extract `smem_cost` from the candidate generator into a per-kernel trait/ fn after two kernels use it.
- Split `autotune.rs` into `autotune_core.rs` (cache/framework) + `autotune_kernels.rs` (candidate generators) once the file gets too big — after the tests still pass.

---

## 3. Audit what grim already has (read this before writing code)

Files to read in full before touching anything:

```
crates/grim-backend-rocm/src/autotune.rs        # existing Autotuner skeleton
crates/grim-backend-rocm/src/device/gemm_tuning.rs  # static lookup tables
crates/grim-backend-rocm/src/kernels/mod.rs     # kernel inventory
crates/grim-backend-rocm/src/kernels/charon.rs
crates/grim-backend-rocm/src/kernels/fused_dequant_gemm.rs
crates/grim-backend-rocm/src/kernels/wmma_gemm.rs
crates/grim-backend-rocm/src/kernels/scythe_persistent.rs
crates/grim-backend-rocm/src/kernels/qkv_attention.rs
crates/grim-backend-rocm/src/kernels/iq_gemm.rs
crates/grim-backend-rocm/src/kernels/q4k_gemm.rs
crates/grim-backend-rocm/src/device/accel_ffi.rs       # launch API surface
crates/grim-backend-rocm/src/device/roc_device.rs      # RocmDevice::matmul, split_k clamp
crates/grim-backend-rocm/src/jit_cache.rs              # JIT compile cache (if it exists)
crates/grim-backend-rocm/src/lib.rs                   # module tree — confirm autotune.rs is wired in
```

Confirm three things from those reads:

1. **Is `autotune.rs` declared in the module tree?** If not, the first Red test is "autotune module is reachable" — add it to `lib.rs`/`mod.rs`, write the test that names it, make it green. Don't add the types before the module is reachable and tested.
2. **Which kernels actually use `lookup_gemm_config` today?** Expect charon, scythe_persistent (opcode 1/2), and `RocmDevice::matmul`. Those are the candidates for the new benchmarked path. Write baseline tests for each before touching them.
3. **What does `accel_ffi.rs` expose for launching a HIP kernel with tunable block_dim / grid?** You need the launch-API surface before you can write a benchmark closure that varies block_dim and grid_stride. Write a test that calls the existing launch path first (green baseline), then a test that calls a tunable launch (red→green).

---

## 4. Kernel-by-kernel rollout — each round is its own Red-Green-Refactor cycle

Do NOT try to autotune all kernels at once. Roll out kernel by kernel.

### Round 0: framework — types, cache, candidate contract (no kernel changes yet)

This round establishes the types and the cache contract. It is framework-only; no kernel launch path changes yet.

**TDD sequence:**

1. **Red:** Write a test in `autotune.rs`'s test module (or a new `tests/` integration test) that uses `CompileKey`, `TuneKey`, `ShapeClass`, `FeatureSet`, `LaunchConfig` — these types don't exist yet, so the test won't compile. That's Red.
2. **Green:** Add the minimal types to make the test compile and pass. Not the full design — just enough.
3. **Refactor:** Split `CompileKey`/`TuneKey` if the test demands it. Extract `ShapeClass` from the existing `lookup_gemm_config` decode-vs-prefill logic (reuse the existing rule, don't reinvent it).
4. **Red:** Write a test for `FeatureSet::supported_on(arch)` — e.g. "scalar is supported on gfx1036; WMMA is NOT supported on gfx1036; WMMA IS supported on gfx1100". Red because the fn doesn't exist.
5. **Green:** Implement `supported_on` with the arch gates. Mirror the `#if defined(__gfx1100__)` / `gcnArchName >= gfx1200` guards from the HIP source. Test both gfx1036 and gfx1100 (and gfx1200 if you have a definition for it).
6. **Red:** Write a test for candidate generation — e.g. `charon_scalar_candidates(arch_gfx1036)` returns a `Vec<LaunchConfig>` with expected count and no WMMA configs.
7. **Green:** Implement the candidate generator. Minimal — just the scalar candidates for gfx1036.
8. **Red:** Write a test for pruning — generate full cartesian, apply pruner, assert forbidden combos are gone (SMEM overflow, unsupported arch, mutual exclusion).
9. **Green:** Implement the pruner. Start with the SMEM rule ( Rule A). Add mutual-exclusion (Rule B) and arch gating (Rule C) as separate tests.
10. **Red:** Write a test for the `Autotuner` cache — lookup absent key → cache miss; insert → lookup returns it; serialize → deserialize round-trips.
11. **Green:** The existing `Autotuner` probably already passes some of these. Write the tests that fail, make them pass. If `Autotuner` isn't wired into the module tree, the test that names it is Red until you wire it in.
12. **Refactor:** Once the framework types, candidate generators, pruner, and cache all have passing tests, refactor naming and module structure. Then move to round 1.

**All-Rust note:** the framework tests are pure Rust, no device needed. They test the candidate set logic, the pruner logic, the cache logic. No HIP launch, no GPU. This is the bulk of round 0 and it's all device-free Rust tests.

### Round 1: charon (sortless fused MoE dispatch) — the first kernel

Charon is the highest-value target — grim's custom MoE kernel, the thing Unsloth also autotunes (grouped_gemm forward/backward).

**TDD sequence:**

1. **Baseline tests (green):** Write tests against the existing charon static path — for representative shapes on gfx1036, assert the launch parameters come from `lookup_gemm_config` as expected. If these tests don't exist, write them first. They capture the baseline.
2. **Baseline tests (green):** Write tests against `grim_nn::moe::MoeFfn::forward` CPU reference for representative inputs. These are the parity oracle named in charon.rs. If these tests don't exist, write them. They give us the "correct output" reference.
3. **Red:** Write a parity test: "autotuned charon launch on shape X produces output ≈ CPU reference within tolerance". Red because the autotuned launch isn't wired yet.
4. **Green:** Wire the charon launch to use the `Autotuner` — on cache hit, use cached `LaunchConfig`; on cache miss, run the benchmark and cache the winner. Start with a stub benchmark that returns the static-lookup config (so the parity test can turn green without a real benchmark). That's the minimal green.
5. **Refactor:** Once parity passes with the stub, replace the stub benchmark with the real benchmark (section 3.4) — warm-up + timed launches + winner selection. Re-run the parity test; it should still pass. Then add the perf test.
6. **Red:** Write a cache-behavior test: "first launch for a TuneKey benchmarks and caches; second launch for the same TuneKey is a cache hit and returns immediately". Red if the cache-hit path isn't wired.
7. **Green:** Wire the cache-hit path. Test it.
8. **Refactor:** Log pruning stats, cache hits/misses (section 5.6). Add observability tests if you want them asserted.

**Correctness gate (asserted by tests, not hand-waved):**
- autotuned charon output ≈ CPU reference (`grim_nn::moe::MoeFfn::forward`) within tolerance, on a representative set of shapes.
- autotuned charon output ≈ current static-lookup charon output (no regress).
- on gfx1036, the autotuned candidate set has no WMMA configs (test asserts this at candidate-generation time).

**Perf gate (observational, but still Rust):**
- a Rust benchmark binary or `#[test]` with `#[ignore]` (or a CI-only bench) that records autotuned time vs static time on representative shapes. Not a unit-test pass/fail — it's a number. But it's Rust, and it's committed/run in CI, not a notebook.

**All-Rust note:** the charon benchmark harness is Rust. It allocates scratch via the existing HIP allocation path (via `accel_ffi`), launches via the existing HIP launch path, times via HIP events. No Python, no Triton, no external process.

### Round 2: fused_dequant_gemm (generic f16 + quant formats)

`fused_dequant_gemm.rs` has a large parameter space: `default_bpw`, `outlier_count`, `backup_bpw`, `backup2_bpw`, per-column scale. Some are data-dependent, not tunable.

**TDD sequence:**

1. **Baseline tests (green):** tests against the existing static fused_dequant_gemm path on representative shapes + representative quant formats. Capture the baseline.
2. **Baseline tests (green):** tests against `grim_quant::dequant_*` CPU dequant + reference GEMM, for parity oracle. If these don't exist, write them.
3. **Red:** parity test: "autotuned fused_dequant_gemm launch produces output ≈ baseline within tolerance". Red until wired.
4. **Green:** wire the launch to the `Autotuner`, start with stub benchmark returning the static config.
5. **Refactor:** replace stub with real benchmark. Re-run parity.
6. **Red:** cache-behavior test (same as round 1).
7. **Green:** wire cache-hit path.
8. **Refactor:** log stats.

**Correctness gate (asserted by tests):**
- autotuned output ≈ CPU dequant + reference GEMM within tolerance, across a few representative outlier-count / backup-codebook combos (the data-dependent params).
- autotuned output ≈ static path (no regress).

**All-Rust note:** the candidate set for fused_dequant_gemm should be valid across different outlier counts — test that in Rust by generating candidates once and running the parity test with a few representative outlier-count inputs. The candidate set itself doesn't depend on outlier count; the parity test does.

### Round 3: wmma_gemm variants (fp8, mxfp4, mxfp8, q8_0 fused)

`wmma_gemm.rs` has multiple entries: `grim_wmma_gemm` (f16, WMMA on gfx1100+, scalar fallback), `grim_wmma_gemm_fp8`, `grim_fused_dequant_gemm_fp8`.

**TDD sequence:**

1. **Baseline tests (green):** tests against the scalar fallback path on gfx1036 for representative shapes. Capture the baseline.
2. **Baseline tests (green):** tests against the WMMA path on gfx1100+ (if you have a gfx1100 definition; otherwise skip the WMMA baseline and test the scalar path only). The existing plan says "FP32 first; FP8/MXFP4/MXFP8/Q8_0/IQK WMMA variants follow the pattern `wmma_gemm.rs` already establishes" and "Device-gated for WMMA numeric parity vs the scalar grouped kernel." So the parity oracle is the scalar path.
3. **Red:** parity test: "autotuned wmma launch on gfx1100+ produces output ≈ scalar path within tolerance". Red until wired.
4. **Green:** wire the launch to the `Autotuner`, stub benchmark returning the scalar config.
5. **Refactor:** replace stub with real benchmark. Re-run parity.
6. **Red:** arch-gating test: "on gfx1036, the wmma candidate set is empty (no WMMA configs)". Red if the arch gating isn't wired.
7. **Green:** implement `FeatureSet::supported_on` for WMMA (gfx1100+) and fp8 MFMA (gfx1200+), and the candidate generator filters by it. Test on gfx1036 (empty WMMA set) and gfx1100+ (WMMA set present).
8. **Refactor:** log stats.

**Correctness gate (asserted by tests):**
- autotuned WMMA output ≈ scalar path within tolerance (on archs where WMMA exists).
- on gfx1036, no WMMA config is ever launched (assert at candidate-generation time and at launch-selection time).

**All-Rust note:** the arch gating is the key Rust test here. The `#if defined(__gfx1100__)` guards in the HIP source are the source of truth; `FeatureSet::supported_on` mirrors them, and the test asserts the mirror is correct. If the HIP source adds a new arch guard, the Rust test must be updated to match — that's the contract.

### Round 4: qkv_attention / cross_attention — measurement-gated, not assumed

These are already wave-aware and hand-tuned. The launch geometry is `grid=(seq_len, num_heads, 1)`, `block=(256,1,1)`. Autotuning block_dim for a 256-thread block that's already wave-aligned may not help.

**TDD sequence:**

1. **Baseline tests (green):** tests against the existing attention launch parameters. Capture the baseline.
2. **Red→Green perf test:** a Rust benchmark that sweeps block_dim for the attention kernel on representative shapes and records the time. This is the decision gate: if block_dim sweep shows < 3% variation, skip autotuning for attention. The sweep itself is Rust (candidate generator + benchmark harness, already built in rounds 0-1).
3. **If the sweep says "worth it":** continue with parity test (red→green) and cache-behavior test (red→green) as in rounds 1-2.
4. **If the sweep says "not worth it":** the test result is "skip". No autotuning for attention. The Rust benchmark is still committed as evidence.

**All-Rust note:** the decision is data-driven by a Rust benchmark, not a guess. The "skip" outcome is still a tested, documented outcome.

### Rounds 5+: iq_gemm, q4k_gemm, rwkv, selective_scan — lower priority

Defer until rounds 0-3 are solid and the framework is proven. IQ-gemm has many format variants; each might want its own candidate set. Don't open that can until the framework handles charon + fused_dequant_gemm cleanly, with passing tests.

When you do start round 5, the TDD sequence is the same as rounds 1-2: baseline tests → parity test (red) → autotuned launch with stub benchmark (green) → real benchmark (refactor) → cache-behavior test (red→green). Reuse the framework from rounds 0-1; add a new candidate generator per kernel variant.

---

## 5. Practical concerns — all Rust

### 5.1 Scratch buffer management in benchmarks

Benchmarking multiple candidates per `TuneKey` means multiple kernel launches with temporary buffers. In Rust:

- **Per-benchmark scratch pool:** allocate once per benchmark session via the existing HIP allocation path (`accel_ffi`), reuse across candidates. Avoids repeated hipMalloc/hipFree in the tuning hot path.
- **NOT persistent scratch in `Autotuner`:** different kernels need different scratch shapes. Keep scratch per-bench (Rust struct owned by the benchmark harness), not in the cache.

### 5.2 Timing: GPU timers, Rust side

HIP launches are async. `std::time::Instant::now()` before and after a hip launch measures CPU-side launch overhead, not kernel time. Use HIP events (`hipEventRecord` / `hipEventSynchronize` / `hipEventElapsedTime`) via `accel_ffi` / the existing ROCm FFI. The existing `cycles_per_invocation` in `AutotuneConfig` suggests grim has some cycle-counting already — confirm what it measures (GPU-side or CPU-side) and use the GPU-side timer for selection.

The benchmark harness is Rust. It calls HIP event APIs through the existing FFI. No Python timing.

### 5.3 Warm-up for JIT — Rust side

The FFI skill documents this: "The first GEMM on a fresh process can trigger a one-time kernel compile that dwarfs the call. Absorb it with a tiny warm-up GEMM at startup, or latency benchmarks will be misleading."

In Rust: before timing any candidate, run one launch with a tiny input (e.g. 2×2) to trigger any one-time hipRTC compilation. Do this per kernel (the compilation is per-kernel-source, not per-config), not per candidate. Implement as a Rust fn `warm_up(kernel_handle)` called once per kernel at benchmark start. Test it: a benchmark without warm-up should produce misleadingly high first-launch times; a benchmark with warm-up should not. You can assert this in a Rust test if you can observe the JIT effect, or at least document it in the benchmark harness code.

### 5.4 Arch gating in the candidate set, not in the benchmark

Don't bench WMMA configs on gfx1036 and rely on the kernel to fall back — that wastes benchmark time and risks confusion. Prune unsupported feature sets at candidate generation (Rule B), using `FeatureSet::supported_on(arch)`. The `#if defined(__gfx1100__)` guards in the HIP source are the source of truth; `FeatureSet::supported_on` mirrors them in Rust.

Test: `FeatureSet::supported_on(gfx1036)` returns false for WMMA; `FeatureSet::supported_on(gfx1100)` returns true for WMMA. If the HIP source changes the guard, update the Rust test.

### 5.5 Persistence: keep grim's JSON, all Rust

grim's `Autotuner` already serializes to JSON (`to_json_bytes` / `save_to_file` / `from_json_bytes`). Keep that for launch-config persistence. All Rust. Add a separate compile-cache persistence if `jit_cache.rs` doesn't already cover it — you don't want to re-JIT the same kernel+arch+features on every process start. Test the round-trip: serialize a cache, deserialize, assert the deserialized cache returns the same `LaunchConfig` for the same `TuneKey`.

### 5.6 Stats / observability — Rust logs

Unsloth logs pruning: `logger.debug(f"Pruning configs: {len(configs)}")` → `logger.debug(f"Pruned configs: {len(pruned_configs)}")`. grim should log (via `log` crate or `tracing`, whatever the crate uses):

- number of candidates generated
- number pruned (and why: SMEM, unsupported arch, mutual exclusion)
- number benchmarked
- winner chosen + its config
- cache hit/miss for subsequent launches

This is cheap and makes the autotuning legible when it misbehaves. In Rust, these are `trace!`/`info!` calls. You can test log output if the crate's logging is set up for it, or just eyeball in CI.

---

## 6. What success looks like (verification — all Rust tests)

For each kernel that gets autotuned, the following are asserted by Rust tests (not hand-waved):

1. **Correctness:** autotuned-path output == static-lookup-path output (or CPU reference) within tolerance, on a representative set of shapes. Tested via parity tests (round 1-3 sequence).
2. **No regress vs static:** autotuned winner is at least as fast as the current static `lookup_gemm_config` pick, on the shapes where the static pick exists. Observed via Rust benchmark (section 4 perf tests).
3. **Cache works:** second run with the same `TuneKey` is a cache hit (no re-benchmark). Tested via cache-behavior test (red→green).
4. **Persistence works:** process restart reloads from JSON. Tested via serialize/deserialize round-trip test.
5. **Pruning works:** on gfx1036, WMMA candidates are absent from the candidate set (test asserts at candidate-generation time). On gfx1100+, they're present (test asserts).
6. **SMEM safety:** no config that exceeds device SMEM makes it to a launch. Tested via SMEM pruner test (red→green) — generate a config that exceeds SMEM, assert it's pruned.
7. **Arch gating works:** `FeatureSet::supported_on` mirrors the HIP `#if` guards. Tested via arch-gating test on gfx1036 / gfx1100 / gfx1200.

---

## 7. Order of operations (concrete, with the TDD gate per step)

1. Read the files in section 3. Confirm module-tree status of `autotune.rs`.
2. **Round 0, step 1 (Red):** write a test that uses `CompileKey`, `TuneKey`, `ShapeClass`, `FeatureSet`, `LaunchConfig`. Fails to compile. Add the types. Green. Refactor the split.
3. **Round 0, step 2 (Red):** write a test for `FeatureSet::supported_on(arch)`. Fails. Implement. Green. Test gfx1036 / gfx1100 / gfx1200.
4. **Round 0, step 3 (Red):** write a test for candidate generation (charon scalar, gfx1036). Fails. Implement minimal generator. Green.
5. **Round 0, step 4 (Red):** write a test for pruning (SMEM, mutual exclusion, arch gating). Fails. Implement pruner. Green. Add rules one at a time, each with its own test.
6. **Round 0, step 5 (Red):** write a test for `Autotuner` cache (miss, insert, lookup, serialize/deserialize). Fails where the existing cache doesn't cover it. Make it pass. If `autotune.rs` isn't in the module tree, wire it in first (that's a Red test: "autotune module is reachable").
7. **Round 0, step 6 (Refactor):** once framework types, candidate generators, pruner, and cache all have passing tests, refactor naming and module structure.
8. **Round 1, step 1 (baseline, green):** write tests against existing charon static path + CPU reference. If they don't exist, write them (they may be green immediately — baseline capture).
9. **Round 1, step 2 (Red):** parity test: autotuned charon ≈ CPU reference. Fails. Wire launch to `Autotuner` with stub benchmark. Green.
10. **Round 1, step 3 (Refactor):** replace stub with real benchmark. Re-run parity. Add perf benchmark (Rust).
11. **Round 1, step 4 (Red):** cache-behavior test. Fails. Wire cache-hit path. Green.
12. **Round 1, step 5 (Refactor):** log stats. Then round 2.
13. **Round 2:** same TDD sequence as round 1, for fused_dequant_gemm.
14. **Round 3:** same TDD sequence as round 1, for wmma_gemm variants, plus the arch-gating test.
15. **Round 4:** measurement-gated — Rust benchmark sweep first; autotune only if the sweep says so.
16. **Rounds 5+:** same TDD sequence, deferred.

---

## 8. What this does NOT do (all-Rust restatement)

- Does not add Triton as a dependency. Triton is Python/C++, not Rust. Grim stays Rust+HIP.
- Does not add Python. No Python benchmark harness, no Python tuning script, no Python glue. Every new artifact is a `.rs` file or a `#[test]` in an existing `.rs` file.
- Does not autotune norm/rope/activation kernels (grim doesn't have standalone HIP versions of those today; if it adds them, revisit — in Rust, with TDD).
- Does not change the static `gemm_tuning.rs` lookup tables for rocBLAS solution indices — those are a separate concern (rocBLAS GEMM selection, not grim's own HIP kernel selection). The autotuning here is for grim's own kernel launches, supplanting `lookup_gemm_config` for those kernels where it's wired in, with tests proving the supplanting is correct.
- Does not add new kernels. It adds a tuned-launch layer (Rust) on top of existing kernels (Rust wrapper + HIP source literal).

---

## 9. All-Rust audit (final checklist)

Before calling the plan done, every one of these is a Rust file or a Rust test:

- [ ] `CompileKey`, `TuneKey`, `ShapeClass`, `FeatureSet`, `LaunchConfig`, `CandidateSet` — Rust types in `autotune.rs` or a new `autotune_*.rs` module.
- [ ] `FeatureSet::supported_on(arch)` — Rust fn, tested on gfx1036 / gfx1100 / gfx1200.
- [ ] Candidate generators — Rust fns, one per kernel variant, tested for count + pruning.
- [ ] Pruner — Rust fn, tested for SMEM / mutual exclusion / arch gating.
- [ ] `Autotuner` cache — existing Rust; tested for miss/insert/lookup/serialize-deserialize.
- [ ] Benchmark harness — Rust fn, uses HIP events via `accel_ffi`, warm-up per kernel, winner selection by GPU time.
- [ ] Launch integration — Rust: `RocmDevice`/`accel_ffi` accepts `LaunchConfig`, tested for cache-hit and cache-miss behavior.
- [ ] Parity tests — Rust `#[test]`s asserting autotuned output ≈ baseline / CPU reference within tolerance.
- [ ] Cache-behavior tests — Rust `#[test]`s asserting cache hit on second launch, cache miss on first.
- [ ] Persistence tests — Rust `#[test]`s asserting serialize/deserialize round-trip.
- [ ] SMEM safety tests — Rust `#[test]`s asserting over-SMEM configs are pruned.
- [ ] Arch-gating tests — Rust `#[test]`s asserting WMMA absent on gfx1036, present on gfx1100+.
- [ ] Perf benchmarks — Rust benchmark binary or `#[ignore]` tests, committed, run in CI.
- [ ] Observability — Rust `trace!`/`info!` calls, optionally tested if the crate's logging supports it.

If any of these is not a Rust file or a Rust test, the plan is not satisfied. Fix it.

---

## 10. References

- Unsloth autotuning source (the conceptual model, not the code to port): `old/repos/unsloth-main/unsloth/kernels/moe/grouped_gemm/kernels/autotuning.py` and `tuning.py`.
- grim existing autotune skeleton (Rust, keep): `crates/grim-backend-rocm/src/autotune.rs`.
- grim static lookup (Rust, augment/replace where wired in): `crates/grim-backend-rocm/src/device/gemm_tuning.rs`.
- grim kernel inventory (Rust): `crates/grim-backend-rocm/src/kernels/mod.rs`.
- grim charon (Rust + HIP source literal): `crates/grim-backend-rocm/src/kernels/charon.rs`.
- grim wmma_gemm (Rust + HIP source literal): `crates/grim-backend-rocm/src/kernels/wmma_gemm.rs`.
- grim fused_dequant_gemm (Rust + HIP source literal): `crates/grim-backend-rocm/src/kernels/fused_dequant_gemm.rs`.
- grim launch FFI (Rust): `crates/grim-backend-rocm/src/device/accel_ffi.rs`.
- grim matmul + split_k clamp (Rust): `crates/grim-backend-rocm/src/device/roc_device.rs`.
- ROCm FFI gotchas (warm-up, stream binding, solution_index misuse): rust-ffi skill, ROCm section.
- WMMA/arch gating in HIP source: `wmma_gemm.rs` (`#if defined(__gfx1100__)`...), `charon.rs` (fp8/MFMA gated on `gcnArchName >= gfx1200`).
- Rust TDD / test organization: rust-testing skill.
