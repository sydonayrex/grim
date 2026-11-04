# Prism Hadamard Runtime (Bonsai-27B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** grim loads and runs `models/Bonsai-27B/Ternary-Bonsai-2-27B-PQ2_0.gguf` correctly by implementing the Prism rotated-domain runtime: `prism.hadamard.*` metadata parsing, sign-flip + FWHT activation transforms in the qwen35 decode graph, and a wave32 HIP FWHT-1024 kernel — producing token output that matches the reference `llama-prism` binary.

**Architecture:** Prism (QuaRot-style) folds `W_f = W·D·H` into every linear weight (D = per-channel ±1 signs, H = normalized Sylvester Walsh-Hadamard, block 1024). Because H·H = D·D = I, the runtime recovers `W·x = W_f·(H·(D·x))`: multiply the linear's input by the sign vector, FWHT it, then run the (already landed) PQ2_0 GEMV unchanged. The reference implementation is fully extracted: fork source at `/tmp/prism-src` (PrismML-Eng/llama.cpp, branch `prism`, HEAD `62c418c`), binary dist at `old/repo/llama-prism-b12128-d7b2896/` (used as the ground-truth output oracle).

**Tech Stack:** Rust workspace, HIP/hipRTC JIT (gfx1200/gfx1201 wave32), existing PQ2_0 device path (commit `c7be16f2`: `grim_dot4_pq2_0_q80_gemv` + `BlockDtype::Pq2_0`), grim-graph builders, qwen35 `decode_graph.rs`.

**Parent documents:** `plans/PLAN-production-ready-consumer-rocm.md` (Task 10.5 correction block), reference review in session memory (`project-production-readiness-review-2026-11.md`, "PQ2_0 DEVICE PATH LANDED" section).

## Reference map (verified on disk — read before each task)

| Contract piece | Exact reference location |
|---|---|
| Metadata parser + every validation rule | `/tmp/prism-src/src/llama-model.cpp:2090-2262` (`load_hparams_hadamard`) |
| Metadata key spellings | `/tmp/prism-src/src/llama-arch.cpp:495-505` |
| Foldable-weight whitelist + arch whitelist | `/tmp/prism-src/src/llama-model.cpp:2173-2205` |
| H matrix construction (parity-bit form) | `/tmp/prism-src/src/llama-model.cpp:2338-2341` |
| Sign tensor per-width lookup (`width = weight->ne[0]`) | `/tmp/prism-src/src/llama-model.cpp:2395-2396` |
| GDN ssm_out grouped perm | `/tmp/prism-src/src/llama-model.h:692-701` + `llama-model.cpp:2408-2413` |
| Graph transform application (signs → FWHT, per-folded-linear) | `/tmp/prism-src/src/llama-graph.cpp:1684-1712` (`build_hadamard_input`) |
| Inverse side order (FWHT → signs after embd lookup) | `/tmp/prism-src/src/models/qwen35.cpp:750-754` |
| FWHT CUDA kernels (register / LDS / decode-block variants) | `/tmp/prism-src/ggml/src/ggml-cuda/fwht.cu` (323 lines) + `fwht.cuh` |
| CPU FWHT (scalar + SIMD golden oracle) | `/tmp/prism-src/ggml/src/ggml-cpu/ops.cpp:12301-12375` |
| Upstream dispatch pattern (hint replaces mul_mat) | `/tmp/prism-src/ggml/src/ggml-cuda/ggml-cuda.cu:1821` |
| Fused add_rms_norm+FWHT (later optimization, NOT this plan) | `/tmp/prism-src/ggml/src/ggml-cuda/norm.cu:746` |
| Fused FWHT+q8_1 quantizer (~390 launches saved; later, NOT this plan) | `/tmp/prism-src/ggml/src/ggml-cuda/ggml-cuda.cu:3205-3260` |

## Bonsai-27B ground truth (parsed from the real GGUF header)

- `prism.hadamard.version = 1` (so `tied_output` absent/false; `output.weight` IS present)
- `block_size = 1024`, `transform = "normalized-sylvester-walsh-hadamard"`, `axis = "input-last-dimension"`, `sign_mode = "explicit"`
- `sign_widths = [5120, 6144, 17408]` (each % 1024 == 0; sum = 28672 = `sign_values.len()`)
- `sign_values` = 28,672 i32, every value ±1
- `weight_names` = 401 tensors; `inverse_weight_names = ["token_embd.weight"]`
- `gdn_v_grouped = 1` with `ssm_dt_rank=48`, `ssm_n_group=16`, `ssm_inner_size=6144` → the `.ssm_out.` tensors get the GDN perm: reshape `[hd=128, nk=16, rep=3]` → permute → `[hd=128, rep=3, nk=16]` (see Task 4 formula)
- arch = `qwen35` (grim's green path; no BitNet wrapper involved)
- PQ2_0 device path already landed: `grim_dot4_pq2_0_q80_gemv`, `BlockDtype::Pq2_0`, provider serving (commits `c7be16f2`, `0a94441a`) — this plan does NOT touch those kernels.

## Global constraints

- All verification is local on RX 9070 XT / 9060 XT (gfx1201/gfx1200), wave32. `rocm-smi` is GONE — use `amd-smi metric`.
- DON'T-TOUCH: `grim_dot4_pq2_0_q80_gemv` kernel body and its parity test (`tests/pq2_0_gemm_parity.rs` must stay green untouched); the PQ2_0 dispatch arm in `crates/grim-backend-rocm/src/device/quant/mod.rs`; decode-graph capture no-sync discipline; DeviceGuard usage.
- FWHT kernel rules (rocm-hip-kernels skill): block dims multiples of 32; `--offload-arch` = running `gcnArchName`; no host readback inside the decode pipeline.
- Numerics: FWHT is orthonormal — parity gates run at f32 tolerance `<= 1e-4` relative unless stated bit-exact.
- On any gate failure: STOP, record output, report. No partial commits with success messages.
- Reference source at `/tmp/prism-src` may be re-cloned: `git clone --depth 1 --branch prism https://github.com/PrismML-Eng/llama.cpp /tmp/prism-src`.
- Test target location rule from the parent plan applies: `qwen35_graph_capture` lives in `grim-models-transformer`.

## File structure (locked in)

- Create `crates/grim-quant/src/hadamard.rs` — host math: H matrix construction, `fwht_f32` scalar oracle, signed variant. One responsibility: the math.
- Create `crates/grim-format/src/prism_hadamard.rs` — metadata struct + GGUF key parsing + validation. One responsibility: the schema contract.
- Create `crates/grim-backend-rocm/src/kernels/fwht.rs` — HIP source string (port of `fwht.cu`) + launcher. One responsibility: the device transform.
- Modify `crates/grim-backend-rocm/src/kernels/source_asm.rs` — push the new kernel source (one line + presence assertion).
- Create `crates/grim-backend-rocm/src/device/hadamard_launch.rs` — `RocmDevice` launcher methods.
- Modify `crates/grim-models/transformer/src/qwen35.rs` + `decode_graph.rs` — `HadamardPlan` construction from parsed metadata + graph emission hooks (signs-mul + FWHT before folded linears; inverse after embedding lookup).
- Tests: `crates/grim-quant/tests/hadamard_oracle.rs`, `crates/grim-format/tests/prism_hadamard_meta.rs`, `crates/grim-backend-rocm/tests/fwht_parity.rs`, `crates/grim-models/transformer/tests/bonsai_hadamard_e2e.rs`.

---

### Task 1: Host FWHT oracle + H construction (grim-quant)

**Files:**
- Create: `crates/grim-quant/src/hadamard.rs`
- Modify: `crates/grim-quant/src/lib.rs` (add `pub mod hadamard;`)
- Test: `crates/grim-quant/tests/hadamard_oracle.rs`

**Interfaces:**
- Produces (used by Tasks 2, 3, 4, 5):
  - `pub fn hadamard_entry(row: u32, col: u32, n: u32) -> f32` — `±1/√n` by parity of `row & col`
  - `pub fn build_hadamard_matrix(n: u32) -> Vec<f32>` — row-major n×n
  - `pub fn fwht_f32(data: &mut [f32])` — in-place normalized FWHT over the whole slice; `data.len()` must be a power of 2
  - `pub fn fwht_signed_f32(data: &mut [f32], signs: &[i32])` — applies `data[i] *= signs[i]` then `fwht_f32` (the Task-4 runtime order: D then H)

- [ ] **Step 1: Write failing tests**

```rust
// crates/grim-quant/tests/hadamard_oracle.rs
use grim_quant::hadamard::{build_hadamard_matrix, fwht_f32, fwht_signed_f32, hadamard_entry};

fn approx(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol * a.abs().max(1.0)
}

#[test]
fn h_entry_is_parity_of_row_and_col_over_sqrt_n() {
    // Reference: llama-model.cpp:2338-2341 — sign = parity of (r & c) XOR-folded.
    let n = 8.0f32;
    let s = 1.0 / n.sqrt();
    assert_eq!(hadamard_entry(0, 0, 8), s);
    assert_eq!(hadamard_entry(1, 1, 8), -s); // 1 & 1 = 1 -> odd parity
    assert_eq!(hadamard_entry(1, 2, 8), s);  // 1 & 2 = 0 -> even
    assert_eq!(hadamard_entry(3, 1, 8), -s); // 3 & 1 = 1 -> odd
    let _ = n;
}

#[test]
fn h_h_equals_identity_for_1024() {
    let n = 1024usize;
    let h = build_hadamard_matrix(n as u32);
    // (H·H)[i][j] = delta_ij since H is orthonormal (H^2 = I).
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f32;
            for k in 0..n {
                acc += h[i * n + k] * h[k * n + j];
            }
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(approx(acc, want, 1e-4), "H·H[{i}][{j}]={acc}");
        }
    }
}

#[test]
fn fwht_matches_matrix_product_on_random_rows() {
    // The butterfly must equal the explicit matrix product H·x.
    let n = 1024usize;
    let h = build_hadamard_matrix(n as u32);
    let mut state = 0x243F6A8885A308D3u64;
    let mut rand = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f32 / (1u64 << 53) as f32 * 4.0 - 2.0
    };
    for _row in 0..4 {
        let x: Vec<f32> = (0..n).map(|_| rand()).collect();
        let mut via_butterfly = x.clone();
        fwht_f32(&mut via_butterfly);
        for i in 0..n {
            let mut acc = 0.0f32;
            for k in 0..n {
                acc += h[i * n + k] * x[k];
            }
            assert!(approx(via_butterfly[i], acc, 1e-4));
        }
    }
}

#[test]
fn fwht_signed_applies_signs_then_h() {
    // D then H: fwht_signed(x, d)[i] == sum_k H[i][k] * d[k] * x[k]
    let n = 64usize;
    let h = build_hadamard_matrix(n as u32);
    let signs: Vec<i32> = (0..n).map(|i| if i % 3 == 0 { -1 } else { 1 }).collect();
    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25 - 4.0).collect();
    let mut got = x.clone();
    fwht_signed_f32(&mut got, &signs);
    for i in 0..n {
        let mut acc = 0.0f32;
        for k in 0..n {
            acc += h[i * n + k] * (signs[k] as f32) * x[k];
        }
        assert!(approx(got[i], acc, 1e-4));
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p grim-quant --test hadamard_oracle`
Expected: FAIL — `hadamard` module not found.

- [ ] **Step 3: Implement `hadamard.rs`**

Port exactly from the references. The parity trick is `llama-model.cpp:2338-2341`; the butterfly stages are `ops.cpp:12330-12333` (scalar) — in-place, `len` doubling, low element `u+v`, high `u−v`, with the `1/√n` scale folded in first (reference does scale-then-butterfly: `ops.cpp:12325-12328`).

```rust
// crates/grim-quant/src/hadamard.rs
//! Prism rotated-domain host math (Task reference: plans/PLAN-prism-hadamard-bonsai.md Task 1).
//! H = normalized Sylvester Walsh-Hadamard; entry sign is the parity of `row & col`
//! (bitwise-AND, then XOR-fold all bits — llama-model.cpp:2338-2341).
//! `fwht_f32` is the llama.cpp CPU butterfly (ops.cpp:12301) ported verbatim:
//! scale by 1/sqrt(n) first, then len-doubling passes, low = u+v, high = u−v.

/// One H entry without materializing the matrix: `(parity(row & col) ? -1 : 1) / sqrt(n)`.
pub fn hadamard_entry(row: u32, col: u32, n: u32) -> f32 {
    let scale = 1.0 / (n as f32).sqrt();
    let mut parity = row & col;
    parity ^= parity >> 16;
    parity ^= parity >> 8;
    parity ^= parity >> 4;
    parity ^= parity >> 2;
    parity ^= parity >> 1;
    if parity & 1 == 1 {
        -scale
    } else {
        scale
    }
}

/// Row-major n×n H. n must be a power of two.
pub fn build_hadamard_matrix(n: u32) -> Vec<f32> {
    assert!(n.is_power_of_two(), "hadamard n must be a power of two");
    (0..n * n)
        .map(|idx| hadamard_entry(idx / n, idx % n, n))
        .collect()
}

/// In-place normalized fast Walsh-Hadamard transform. `data.len()` must be a power of two.
pub fn fwht_f32(data: &mut [f32]) {
    let n = data.len();
    assert!(n.is_power_of_two(), "fwht len must be a power of two");
    let scale = 1.0 / (n as f32).sqrt();
    for v in data.iter_mut() {
        *v *= scale;
    }
    let mut len = 1usize;
    while len < n {
        for i in (0..n).step_by(2 * len) {
            for j in 0..len {
                let u = data[i + j];
                let v = data[i + len + j];
                data[i + j] = u + v;
                data[i + len + j] = u - v;
            }
        }
        len <<= 1;
    }
}

/// The runtime fold order for a Hadamard-aware linear: signs D applied to the
/// input, then H. Equivalent to `W_f · (H·(D·x))` against a folded weight.
pub fn fwht_signed_f32(data: &mut [f32], signs: &[i32]) {
    assert_eq!(data.len(), signs.len(), "sign vector length mismatch");
    for (v, s) in data.iter_mut().zip(signs.iter()) {
        *v *= *s as f32;
    }
    fwht_f32(data);
}
```

Add to `crates/grim-quant/src/lib.rs`:

```rust
pub mod hadamard;
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p grim-quant --test hadamard_oracle`
Expected: 4 passed. If `fwht_matches_matrix_product` fails, the butterfly sign convention is wrong — re-read `ops.cpp:12330-12333` (low = `u+v`, high = `u−v`, scale BEFORE passes) before changing anything.

- [ ] **Step 5: Commit**

```bash
git add crates/grim-quant/src/hadamard.rs crates/grim-quant/src/lib.rs crates/grim-quant/tests/hadamard_oracle.rs
git commit -m "feat(quant): host Walsh-Hadamard oracle — parity H construction + normalized FWHT"
```

---

### Task 2: `prism.hadamard.*` metadata parser (grim-format)

**Files:**
- Create: `crates/grim-format/src/prism_hadamard.rs`
- Modify: `crates/grim-format/src/lib.rs` (add `pub mod prism_hadamard;`)
- Modify: `crates/grim-format/src/gguf.rs` — expose raw i32/string array access if not already public (check `GgufMetadataLookup`; add only what's missing)
- Test: `crates/grim-format/tests/prism_hadamard_meta.rs`

**Interfaces:**
- Consumes: `GgufProvider` metadata access (`gguf.rs`).
- Produces (used by Task 4):
  - `pub struct PrismHadamardMeta { pub version: u32, pub block_size: u32, pub sign_widths: Vec<i32>, pub sign_values: Vec<i32>, pub weight_names: Vec<String>, pub inverse_weight_names: Vec<String>, pub gdn_v_grouped: bool }`
  - `pub fn parse_prism_hadamard(meta: &GgufMetadata) -> Result<Option<PrismHadamardMeta>>` — `Ok(None)` when no `prism.hadamard.version` key (plain models); every validation failure = `Err` naming the key, mirroring `load_hparams_hadamard` (`llama-model.cpp:2090-2160`) rule for rule.
  - `pub fn sign_vector_for<'a>(meta: &'a PrismHadamardMeta, width: usize) -> Option<&'a [i32]>` — the per-tensor lookup (`width % block_size == 0` and present in `sign_widths`).

- [ ] **Step 1: Write failing tests** — validation table mirrors the reference error strings; the last test gates against the REAL Bonsai file.

```rust
// crates/grim-format/tests/prism_hadamard_meta.rs
use grim_format::prism_hadamard::{parse_prism_hadamard, sign_vector_for, PrismHadamardMeta};

// Helper: build a minimal metadata map the same way gguf.rs tests do
// (reuse the existing test GGUF writer helper pattern from
// crates/grim-format/tests/prism_unsupported_format_test.rs::write_gguf,
// extended with a KV injector — copy that helper in here; do not factor it
// out of the existing test file in this task).

fn meta_with(kv: Vec<(&str, KvVal)>) -> GgufMetadata { /* write_gguf-style helper */ }

#[test]
fn no_hadamard_keys_parses_to_none() { /* ... Ok(None) ... */ }

#[test]
fn version_must_be_1_or_2() { /* version=3 -> Err contains "unsupported prism.hadamard.version: 3" */ }

#[test]
fn v2_tied_output_contract() { /* (v==2) != tied_output -> Err "version 2 requires tied_output=true; version 1 forbids it" */ }

#[test]
fn block_size_must_be_pow2() { /* block_size=1000 -> Err "invalid prism.hadamard.block_size: 1000" */ }

#[test]
fn transform_axis_sign_mode_whitelists() { /* each wrong value -> Err naming it; transform must equal "normalized-sylvester-walsh-hadamard"; axis "input-last-dimension"; sign_mode "identity"|"explicit" */ }

#[test]
fn explicit_sign_rules() {
    /* explicit + empty widths -> Err "sign_mode is explicit but sign_widths is empty";
       width=20 with block=1024 -> Err "invalid prism.hadamard sign width: 20" (20 % 1024 != 0);
       duplicate width -> Err "duplicate prism.hadamard sign width";
       value 0 -> Err "sign values must be +/-1";
       widths sum != values.len() -> Err "sign_values length mismatch" */
}

#[test]
fn bonsai_real_file_parses_exact() {
    // The real checkpoint pins every field. Skip when the file is absent.
    const MODEL: &str = "models/Bonsai-27B/Ternary-Bonsai-2-27B-PQ2_0.gguf";
    if !std::path::Path::new(MODEL).exists() { eprintln!("skipping: no Bonsai file"); return; }
    let provider = GgufProvider::open(MODEL).unwrap();
    let m = parse_prism_hadamard(provider.metadata()).unwrap().expect("bonsai must carry hadamard");
    assert_eq!(m.version, 1);
    assert_eq!(m.block_size, 1024);
    assert_eq!(m.sign_widths, vec![5120, 6144, 17408]);
    assert_eq!(m.sign_values.len(), 28672);
    assert!(m.sign_values.iter().all(|&v| v == 1 || v == -1));
    assert_eq!(m.weight_names.len(), 401);
    assert_eq!(m.inverse_weight_names, vec!["token_embd.weight"]);
    assert!(m.gdn_v_grouped);
    // Per-tensor lookup by full input dim:
    assert_eq!(sign_vector_for(&m, 5120).unwrap().len(), 5120);
    assert_eq!(sign_vector_for(&m, 10240).unwrap().len(), 10240); // 10240 = 5120+5120 slice? NO —
    // CAREFUL: sign_vector_for finds the width GROUP whose value equals the width.
    // 10240 is NOT a width group in Bonsai; tensors of dim 10240 must FAIL lookup:
    assert!(sign_vector_for(&m, 10240).is_none());
}
```

**CORRECTION while writing (from the reference, `llama-model.cpp:2395-2396`):** the per-tensor lookup keys on the tensor's full input dim `weight->ne[0]`. Bonsai's three width groups are 5120/6144/17408 — so a folded tensor whose `ne[0]` is, say, 10240 (e.g. a fused qkv) would find NO sign vector and the reference THROWS (`prism.hadamard has no sign vector for width %u`). The real Bonsai tensors are single-projection shapes (the header shows `attn_qkv` at 5120 input), so lookups succeed — but the parser + Task 4 wiring must reproduce the throw, not silently skip. Fold `sign_vector_for` returning `None` into a hard error at graph-build time (Task 4).

Rewrite the last assertions accordingly:

```rust
    assert!(sign_vector_for(&m, 10240).is_none(), "width groups are exact-match, not prefix");
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p grim-format --test prism_hadamard_meta`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement the parser**

Transcribe `load_hparams_hadamard` (`/tmp/prism-src/src/llama-model.cpp:2090-2160`) rule-for-rule. Key spellings from `llama-arch.cpp:495-505`:

```rust
// crates/grim-format/src/prism_hadamard.rs
//! prism.hadamard.* GGUF metadata contract (PrismML fork, llama-model.cpp:2090-2262).
//! Every error message mirrors the fork so a bad checkpoint fails identically.

pub const KEY_VERSION: &str = "prism.hadamard.version";
pub const KEY_BLOCK_SIZE: &str = "prism.hadamard.block_size";
pub const KEY_TRANSFORM: &str = "prism.hadamard.transform";
pub const KEY_AXIS: &str = "prism.hadamard.axis";
pub const KEY_SIGN_MODE: &str = "prism.hadamard.sign_mode";
pub const KEY_SIGN_WIDTHS: &str = "prism.hadamard.sign_widths";
pub const KEY_SIGN_VALUES: &str = "prism.hadamard.sign_values";
pub const KEY_WEIGHT_NAMES: &str = "prism.hadamard.weight_names";
pub const KEY_INVERSE_WEIGHT_NAMES: &str = "prism.hadamard.inverse_weight_names";
pub const KEY_GDN_V_GROUPED: &str = "prism.hadamard.gdn_v_grouped";
pub const KEY_TIED_OUTPUT: &str = "prism.hadamard.tied_output";

pub const TRANSFORM_SYLVESTER: &str = "normalized-sylvester-walsh-hadamard";
pub const AXIS_INPUT_LAST: &str = "input-last-dimension";

#[derive(Debug, Clone)]
pub struct PrismHadamardMeta {
    pub version: u32,
    pub block_size: u32,
    pub sign_widths: Vec<i32>,
    pub sign_values: Vec<i32>,
    pub weight_names: Vec<String>,
    pub inverse_weight_names: Vec<String>,
    pub gdn_v_grouped: bool,
    pub tied_output: bool,
}

impl PrismHadamardMeta {
    /// Sign lookup by FULL tensor input dimension (llama-model.cpp:2395-2396:
    /// `const uint32_t width = (uint32_t) weight->ne[0];` exact match only).
    /// `None` = no group with this width; callers MUST hard-error (the fork throws).
    pub fn sign_vector_for(&self, width: usize) -> Option<&[i32]> {
        if width as u32 % self.block_size != 0 {
            return None;
        }
        let mut off = 0usize;
        for &w in &self.sign_widths {
            let w = w as usize;
            if w == width {
                return Some(&self.sign_values[off..off + w]);
            }
            off += w;
        }
        None
    }
}
```

`parse_prism_hadamard` walks the metadata map: absence of `KEY_VERSION` AND `KEY_TIED_OUTPUT` → `Ok(None)`; otherwise enforce, in order: version ∈ {1,2}; `(version == 2) != tied_output` reject; tied_output forbids `output.weight` (needs the tensor table — take it as a `&dyn Fn(&str) -> bool` parameter `has_weight`); block_size pow2; transform/axis/sign_mode exact strings; explicit → widths non-empty, each >0, `% block_size == 0`, unique, values all ±1, widths sum == values len.

- [ ] **Step 4: Run tests → pass**

Run: `cargo test -p grim-format --test prism_hadamard_meta`
Expected: all pass, including the Bonsai real-file gate (7.2 GB header read only — it mmaps, no full scan).

Also run the untouched neighbors: `cargo test -p grim-format` → the 259 existing tests stay green (259/259 was the count at commit `0a94441a`).

- [ ] **Step 5: Commit**

```bash
git add crates/grim-format/src/prism_hadamard.rs crates/grim-format/src/lib.rs crates/grim-format/tests/prism_hadamard_meta.rs crates/grim-format/src/gguf.rs
git commit -m "feat(format): prism.hadamard metadata parser — full fork validation contract + Bonsai gate"
```

---

### Task 3: GPU FWHT kernel + launcher (grim-backend-rocm)

**Files:**
- Create: `crates/grim-backend-rocm/src/kernels/fwht.rs`
- Create: `crates/grim-backend-rocm/src/device/hadamard_launch.rs`
- Modify: `crates/grim-backend-rocm/src/kernels/mod.rs` + `src/lib.rs` + `src/kernels/source_asm.rs` (push + presence assertion)
- Test: `crates/grim-backend-rocm/tests/fwht_parity.rs`

**Interfaces:**
- Consumes: `grim_quant::hadamard::fwht_signed_f32` (oracle), existing `launch_compute_kernel` plumbing.
- Produces (used by Task 4):
  - `pub fn launch_fwht_signed(dev: &RocmDevice, src: &RocmStorage, dst: &RocmStorage, signs: Option<&RocmStorage>, n: u32, rows: usize) -> Result<*mut c_void>` — `n` must be 1024 in this task (extend the switch when a model needs more); `src` f32 or f16; `dst` f32; one kernel launch.
  - HIP symbol `grim_fwht_f32` with signature `(const T* src, float* dst, long long rows, float scale, const float* signs, int n_blk)` — port of `fwht.cu` `fwht_cuda_smem` + `fwht_cuda_block` for N=1024.

- [ ] **Step 1: Write failing parity test** — GPU vs the Task-1 oracle, both signed and unsigned, decode shape (few rows) and prefill shape (many rows).

```rust
// crates/grim-backend-rocm/tests/fwht_parity.rs
use grim_backend_rocm::device::roc_device::RocmDevice;
use grim_quant::hadamard::fwht_signed_f32;
use grim_tensor::{CoreTensorOps, DType, Shape};
use std::panic;

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() { return None; }
    panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::new")).ok()
}

#[test]
#[ignore]
fn fwht_1024_gpu_matches_host_oracle_signed_and_unsigned() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()); };
    let n = 1024usize;
    let mut state = 0x243F6A8885A308D3u64;
    let mut rand = move || {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        (state >> 11) as f32 / (1u64 << 53) as f32 * 4.0 - 2.0
    };
    // Deterministic ±1 signs of width 5120 — but the kernel transforms ONE
    // row of n at a time over `rows` rows; the reference signs tensor covers
    // n_blk * n elements and row r uses signs[(r % n_blk)*n ..]. Use n_blk=5
    // (mimicking width 5120 = 5 blocks of 1024, the Bonsai attn shape).
    let n_blk = 5usize;
    let signs: Vec<i32> = (0..n_blk * n).map(|_| if rand() < 0.5 { -1 } else { 1 }).collect();
    let signs_f32: Vec<f32> = signs.iter().map(|&s| s as f32).collect();

    for &rows in &[1usize, 7usize, 64usize] {
        let src: Vec<f32> = (0..rows * n).map(|_| rand()).collect();

        for use_signs in [false, true] {
            let mut want = src.clone();
            if use_signs {
                // Oracle applies the same per-row modulo: row r uses block (r % n_blk).
                // The kernel's n_blk semantics live in fwht.cu:100
                // (`signs + (r % n_blk) * N`) — mirror it exactly.
                for r in 0..rows {
                    let off = (r % n_blk) * n;
                    let row = &mut want[r * n..(r + 1) * n];
                    for (i, v) in row.iter_mut().enumerate() {
                        *v *= signs[off + i] as f32;
                    }
                    fwht_signed_f32(row, &vec![1i32; n]); // signs already applied; now pure H
                }
            } else {
                for r in 0..rows {
                    fwht_signed_f32(&mut want[r * n..(r + 1) * n], &vec![1i32; n]);
                }
            }

            let src_dev = CoreTensorOps::from_cpu(&dev, &src, &Shape::new(vec![rows * n]), DType::F32)?;
            let dst_dev = CoreTensorOps::zeros(&dev, &Shape::new(vec![rows * n]), DType::F32)?;
            let signs_dev = grim_backend_rocm::MemoryOps::from_cpu_bytes(
                &dev,
                &signs_f32.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>(),
                &Shape::new(vec![n_blk * n]),
                DType::F32,
            )?;
            let s = grim_backend_rocm::device::util::as_rocm(src_dev.as_ref())?;
            let d = grim_backend_rocm::device::util::as_rocm(dst_dev.as_ref())?;
            let sg = grim_backend_rocm::device::util::as_rocm(signs_dev.as_ref())?;
            grim_backend_rocm::device::hadamard_launch::launch_fwht_signed(
                &dev, s, d, if use_signs { Some(sg) } else { None }, n as u32, rows,
            )?;
            dev.synchronize();
            let got = dst_dev.to_cpu_vec_f32()?;
            for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
                assert!((g - w).abs() <= 1e-4 * w.abs().max(1.0),
                    "rows={rows} signs={use_signs} idx={i}: gpu={g} oracle={w}");
            }
        }
    }
    println!("fwht-1024 GPU parity ok (signed + unsigned, rows 1/7/64)");
    Ok(())
}
```

- [ ] **Step 2: Run to verify failure**

Run: `GRIM_GPU_TEST=1 cargo test --release -p grim-backend-rocm --test fwht_parity -- --ignored`
Expected: FAIL — `hadamard_launch` not found.

- [ ] **Step 3: Write the kernel** — port `fwht.cu` variants for N=1024. Include BOTH the LDS variant (prefill, many rows) and the 256-thread block variant (decode, few rows), selected by `rows` in the launcher. Copy the butterfly sign convention verbatim: low `x+y`, high `x−y`, scale folded at load.

```cpp
// crates/grim-backend-rocm/src/kernels/fwht.rs — HIP source (inside the raw string):
// Port of PrismML fwht.cu (fwht_cuda_smem + fwht_cuda_block) for the Bonsai
// block width 1024. Butterfly: low element x+y, high x-y; scale 1/sqrt(N)
// folded at load (fwht.cu:99-104). Signs: row r uses signs[(r % n_blk)*N ...]
// (fwht.cu:100). wave32-safe: no warp-size assumptions in the LDS path.
#define FWHT_N 1024

extern "C" __global__ void grim_fwht_smem_f32(
    const float* __restrict__ src, float* __restrict__ dst,
    const long long n_rows, const float scale,
    const float* __restrict__ signs, const int n_blk)
{
    __shared__ float s[FWHT_N];
    const long long r = blockIdx.x;
    if (r >= n_rows) return;
    src += r * FWHT_N;
    dst += r * FWHT_N;
    const float* signs_row = signs ? signs + (long long)(r % n_blk) * FWHT_N : nullptr;

    for (int i = threadIdx.x; i < FWHT_N; i += blockDim.x) {
        float v = src[i] * scale;
        if (signs_row) v *= signs_row[i];
        s[i] = v;
    }
    __syncthreads();

    for (int h = 1; h < FWHT_N; h *= 2) {
        for (int idx = threadIdx.x; idx < FWHT_N / 2; idx += blockDim.x) {
            const int j = ((idx / h) * 2 * h) + (idx % h);
            const float x = s[j];
            const float y = s[j + h];
            s[j] = x + y;
            s[j + h] = x - y;
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < FWHT_N; i += blockDim.x) {
        dst[i] = s[i];
    }
}
```

Launcher (`crates/grim-backend-rocm/src/device/hadamard_launch.rs`): `block_dim = (256,1,1)` (multiple of 32, rocm-hip-kernels rule), `grid_dim = (rows,1,1)`; wrap `hipMemcpy`-free; signs `Option<&RocmStorage>` passes null ptr when absent. Follow `launch_dot4_pq2_0_q80_gemv`'s arg-packing pattern exactly (`device/compute/dot_gemv.rs`, the PQ2_0 launcher from commit `c7be16f2`). Push the source in `source_asm.rs` next to the Q2_0 push (`source_asm.rs:37-41`) and extend the presence test there.

- [ ] **Step 4: Run → pass**

Run: `GRIM_GPU_TEST=1 cargo test --release -p grim-backend-rocm --test fwht_parity -- --ignored --nocapture`
Expected: PASS. Then the untouched-neighbor gate: `cargo test -p grim-backend-rocm --lib` → 595/595 (count at `0a94441a`).

- [ ] **Step 5: Commit**

```bash
git add crates/grim-backend-rocm/src/kernels/fwht.rs crates/grim-backend-rocm/src/kernels/mod.rs crates/grim-backend-rocm/src/kernels/source_asm.rs crates/grim-backend-rocm/src/device/hadamard_launch.rs crates/grim-backend-rocm/src/lib.rs crates/grim-backend-rocm/tests/fwht_parity.rs
git commit -m "feat(rocm): FWHT-1024 kernel + launcher — port of Prism fwht.cu, signs-aware, wave32"
```

---

### Task 4: HadamardPlan + qwen35 graph wiring (grim-models)

**Files:**
- Modify: `crates/grim-models/transformer/src/qwen35.rs` (plan construction)
- Modify: `crates/grim-models/transformer/src/decode_graph.rs` (emission hooks)
- Test: `crates/grim-models/transformer/tests/bonsai_hadamard_e2e.rs`

**Interfaces:**
- Consumes: `PrismHadamardMeta` (Task 2), `launch_fwht_signed` (Task 3), `emit_linear_dedup` (grim-graph).
- Produces:
  - `pub struct HadamardPlan { block_size: u32, gdn_v_grouped: bool, ssm_perm: Option<GdnPerm>, folded: HashSet<String> }` + `fn from_meta(meta, tensor_names, ssm_dt_rank, ssm_n_group) -> Result<Self>` implementing the foldable-kind + arch whitelist and the `sign_vector_for` hard-error (`has no sign vector for width %u`)
  - decode-graph hook: before ANY folded weight's linear, emit signs-mul + `grim_fwht_smem_f32` on the input; after the token-embedding lookup for an inverse-listed table, emit FWHT then signs (reversed order, `qwen35.cpp:750-754`)

**The math hook (single place):** foldable kinds (transcribe `llama-model.cpp:2173-2191` exactly): `attn_q`, `attn_k`, `attn_v`, `attn_qkv`, `attn_gate`, `attn_output`, `ffn_gate`, `ffn_up`, `ffn_down`, `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`, `ffn_gate_up_exps`, `ffn_gate_shexp`, `ffn_up_shexp`, `ffn_down_shexp`, `ssm_out`, `nextn.draft_head`, plus `output.weight`. Match on the weight name after `blk.<n>.`.

- [ ] **Step 1: Write the failing e2e test** — synthetic first (no GPU model load), because it pins the ORDER and the GDN perm cheaply.

```rust
// crates/grim-models/transformer/tests/bonsai_hadamard_e2e.rs
// Gate A (CPU, no GPU): HadamardPlan::from_meta on the real Bonsai metadata
// accepts all 401 names, rejects an unknown kind, resolves ssm perm
// hd=128, rep=3, nk=16 (inner 6144 / dt_rank 48 / n_group 16).
// Gate B (GPU): one folded linear, W·x identity —
//   arm A: dequant_pq2_0 weights, primal x, plain PQ2_0 GEMV
//   arm B: folded weights (dequant -> fold W·D·H on host via Task-1 math,
//          requantize with a fold-aware packer NOT needed: fold on the ORACLE
//          side instead) — i.e. oracle computes W_f·(H·(D·x)) in f32 and the
//          graph path (signs-mul -> fwht -> GEMV on folded packed weights)
//          must agree within 1e-4 relative.
```

Gate B construction detail (write it exactly in the test): take a random `W [n=8, k=1024]` f32, signs `d[1024]` (a real-width group), compute `W_f = W · diag(d) · H` in f32 (host, Task-1 matrix math — H is orthonormal so this is just per-row `W_f[i] = W[i]·D·H`), pack `W_f` codes by `(q−1)·δ` quantization onto the PQ2_0 layout with a test-local packer (round `w/δ + 1` to 2 bits, δ = row absmax / 2 — the test tolerates pack error since BOTH arms read the same packed codes; arm A's oracle must then use the SAME folded-packed codes: oracle = `Σ packed_decode(W_f)[j]·(H·(D·x))[j]`). The gate asserts: graph(signs+fwht+GEMV over W_f-packed) == oracle(GEMV over W_f-packed against H·D·x), NOT primal recovery — primal recovery is Task 6's e2e job against the real model.

- [ ] **Step 2: Run to verify failure**

Run: `cargo test --release -p grim-models-transformer --test bonsai_hadamard_e2e`
Expected: FAIL — `HadamardPlan` not found.

- [ ] **Step 3: Implement** — in this order:
  1. `HadamardPlan` in `qwen35.rs`: build from `parse_prism_hadamard` output + the provider's tensor-name list; implement the foldable-kind whitelist + `blk.<n>.` name walk + the sign-width hard error message verbatim (`prism.hadamard has no sign vector for width {w} ({name})`).
  2. GDN perm (`llama-model.h:692-701`): for `.ssm_out.` weights when `gdn_v_grouped`: `perm_hd = n_in / n_v`, `perm_nk = ssm_n_group`, `perm_rep = n_v / ssm_dt_rank` with `n_v = ssm_dt_rank`?? — NO: transcribe the CALL SITE `llama-model.cpp:2410-2412`: `set_gdn_v_perm(weight->ne[0], hparams.ssm_dt_rank, hparams.ssm_n_group)` → `n_in = ne[0]`, `n_v = ssm_dt_rank (48)`, `n_k = ssm_n_group (16)`; require `n_v % n_k == 0` and `n_in % n_v == 0` → Bonsai: `perm_hd = 6144/48 = 128`, `perm_nk = 16`, `perm_rep = 3`. Emission order (`llama-graph.cpp:1696-1704`): permute FIRST, then signs, then FWHT: reshape `[hd, nk, rep]` → swap nk/rep → reshape flat.
  3. Decode-graph emission: in `decode_graph.rs`, at each linear emission site where the weight name ∈ `plan.folded`, insert before the linear (and before its activation quantize — the transform output is the quantize input):
     - upload one signs buffer per width group once (5 groups max; key = width), reuse across layers
     - signs-mul: reuse an existing elementwise-mul launcher (grep `grim_mul` / elementwise mul in `kernels/` — one exists; do NOT add a new kernel)
     - FWHT: `launch_fwht_signed` handles signs inside the kernel, so the graph emits ONE FWHT node with the signs tensor and NO separate mul node — matching the fork's fused semantics, not its unfused graph shape. Document this divergence in a comment: the fork emits MUL+FWHT and fuses later; grim emits the fused kernel directly because capture-node count matters.
  4. Inverse side: `token_embd` lookup result → `launch_fwht_signed` with the 5120-width signs (token_embd input dim 5120 from the header) → then continue the existing pipeline. Output head (`output.weight` ∈ folded names, width 5120): same hook at the LM-head matmul input.
  5. Wire `HadamardPlan` construction into model load: parse metadata from the SAME `GgufProvider` the loader already holds; `None` metadata → `plan = None` → zero behavioral change for every other model (this is the "must not touch other models" guarantee).

- [ ] **Step 4: Run gates**

Run: `cargo test --release -p grim-models-transformer --test bonsai_hadamard_e2e` → PASS (Gates A + B).
Then: `GRIM_GPU_TEST=1 cargo test --release -p grim-models-transformer --test qwen35_graph_capture` → all prior gates green (the 13-test suite from the parent plan; a changed node census for NON-hadamard models = regression, STOP).
Then: `cargo test -p grim-cli` (full suite — bin tests).

- [ ] **Step 5: Commit** (one commit; revert-to-old-path per block kind on failure per the parent plan's rule)

```bash
git add crates/grim-models/transformer/src/qwen35.rs crates/grim-models/transformer/src/decode_graph.rs crates/grim-models/transformer/tests/bonsai_hadamard_e2e.rs
git commit -m "feat(qwen35): prism.hadamard graph wiring — signs+FWHT before folded linears, inverse after embd lookup"
```

---

### Task 5: Numerical parity — folded vs primal on the GPU

**Files:**
- Modify: `crates/grim-models/transformer/tests/bonsai_hadamard_e2e.rs` (add Gate C)

**Interfaces:**
- Consumes: everything above.
- Produces: the correctness certificate for Task 6.

- [ ] **Step 1: Gate C — primal identity.** For a synthetic folded linear (k=1024, n=8): pack the PRIMAL `W` to PQ2_0 (test-local packer); separately pack the FOLDED `W_f`. Assert: `GEMV(W_f_packed, fwht_signed(x, d)) ≈ GEMV(W_packed, x)` within f32 + pack tolerance `2e-2` relative (the 2-bit pack error dominates; both arms quantize DIFFERENT weights so pack error does NOT cancel — this gate is about the transform identity, and its tolerance is the PQ2_0 pack envelope, same class as the BitNet int8 arm's 2e-2 in `bitnet_gemm_parity.rs`). Assert additionally that arm-vs-arm argmax over 64 random x agree 100%.

- [ ] **Step 2: Run → pass**, then commit:

```bash
git add crates/grim-models/transformer/tests/bonsai_hadamard_e2e.rs
git commit -m "test(qwen35): primal-vs-folded PQ2_0 identity gate through signs+FWHT path"
```

---

### Task 6: Bonsai-27B e2e — load + parity vs the reference binary

**Files:**
- Modify: `crates/grim-models/transformer/tests/bonsai_hadamard_e2e.rs` (Gate D, `#[ignore]`, needs both cards sanctioned per parent-plan GPU protocol)
- Modify: `docs/benchmarks.md` (record numbers)

**Interfaces:**
- Consumes: full stack. The reference oracle: `old/repo/llama-prism-b12128-d7b2896/llama-cli` (binary dist, verified present).

- [ ] **Step 1: Reference run.** Generate 64 tokens greedy (temp 0) on a fixed prompt with the reference binary; record token ids:

```bash
cd old/repo/llama-prism-b12128-d7b2896 && ./llama-cli -m ../../../models/Bonsai-27B/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  -p "Explain how a transistor works in two paragraphs:" -n 64 --temp 0 --seed 42 --no-display-prompt 2>&1 | tee /tmp/bonsai_ref.txt
```

Expected: coherent text. If the reference binary itself produces garbage, STOP — the ground truth is broken, report before touching grim.

- [ ] **Step 2: grim run.** Same prompt, greedy, same seed, via grim-cli on the two-card split (default placement). Compare TOKEN IDS, not text: greedy decoding must be token-identical or diverge only where logits are within float noise. Gate: first 32 tokens identical; any divergence → print both logit gaps and STOP for analysis (do not weaken to "similar text").

- [ ] **Step 3: Perf record.** `GRIM_PHASE_TRACE` protocol per `docs/benchmarks.md:1114-1122` (≥32-token window), clocks pinned, `amd-smi metric` logged. Record wall/token vs the reference binary's `llama-bench` on the same file. Expectation (not gate): grim slower than the fork's fused kernels on first landing — the FWHT+q8 fusion (~390 launches/step) is explicitly deferred. Record, don't chase.

- [ ] **Step 4: Commit**

```bash
git add crates/grim-models/transformer/tests/bonsai_hadamard_e2e.rs docs/benchmarks.md
git commit -m "test(qwen35): Bonsai-27B e2e — token parity vs llama-prism reference + perf record"
```

---

### Task 7 (deferred, NOT in scope): fused transforms

FWHT+q8_1 quantizer fusion (~390 launches/step) and `add_rms_norm_fwht_f32` fusion — references at the top of this plan. Separate plan AFTER Task 6 proves correctness. Parked deliberately.

## Dependencies

```
T1 (host oracle) ──> T3 (GPU kernel, parity target) ──> T4 (graph wiring) ──> T5 (identity gate) ──> T6 (e2e)
T2 (metadata) ─────────────────────────────────────────> T4
```

## Risks

| Risk | Mitigation |
|---|---|
| Butterfly sign convention wrong (silent ±) | Task 1 test pins fwht == matrix product; H·H=I gate |
| Sign order (D before H vs after) | Task 1 `fwht_signed_applies_signs_then_h` + reference line cited in code |
| Width-group mismatch (10240-dim tensor) | Hard error reproduced verbatim; Bonsai shapes verified 5120/6144/17408 |
| GDN perm wrong on ssm_out | Perm formula + call-site semantics extracted (`llama-model.cpp:2410`); Task 4 Gate A asserts [128,3,16] |
| Decode graph capture node-count drift for OTHER models | plan=None short-circuit + full `qwen35_graph_capture` suite re-run (Task 4 Step 4) |
| Reference binary itself broken | Task 6 Step 1 runs it FIRST; garbage from it = STOP and report |
| VRAM: signs buffers + FWHT scratch | 3 sign groups ≈ 114 KB device total; FWHT dst = one activation buffer — negligible vs 2.25 GiB headroom, but verify post-prewarm free in Task 6 |
