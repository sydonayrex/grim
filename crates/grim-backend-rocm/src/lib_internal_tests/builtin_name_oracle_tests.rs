//! Builtin-name oracle: pins every AMDGPU builtin name grim emits against the
//! **actual toolchain**, by compiling a probe and reading back what clang knows.
//!
//! # Why this exists
//!
//! `kernels/dot_gemv.rs` guarded the FP8 dot behind
//! `__has_builtin(__builtin_amdgcn_fdot4_f32_fp8_fp8)`. That builtin **does not
//! exist** — clang's name is `__builtin_amdgcn_dot4_f32_fp8_fp8` (no `f`).
//! `__has_builtin` therefore returned 0, the `#else` branch was taken, and
//! `grim_dot4_fp8_gemv` ran a scalar `fmaf` loop that decodes each E4M3 element
//! through `fp8_e4m3_to_float_hip`, which calls `powf()`. On gfx1201 — which has
//! the native `v_dot4_f32_fp8_fp8` instruction — the "native FP8 GEMV" was
//! paying a transcendental per element for four products at a time.
//!
//! A source-level grep cannot catch this: the name is syntactically plausible
//! and the guard is written in exactly the defensive style that hides it. Only
//! asking the compiler catches it.
//!
//! The same drift hits the WMMA and sparse-WMMA families, whose real names carry
//! a wave-size suffix (`_w32`/`_w64`) and, for 16x16x16 FP8, a `_gfx12` suffix.
//! `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8` — the form that reads most
//! naturally off the ISA manual — is **not** a builtin. So this module pins all
//! of them.
//!
//! Compiled-and-verified on ROCm 7.2.4 / AMD clang 22.0.0, target gfx1201.
//! The oracle degrades to a skip (not a false pass) when no `hipcc` is present.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Builtin names grim depends on, with the ISA instruction each lowers to.
///
/// `(label, builtin, feature)` — `feature` is the `TARGET_BUILTIN` feature
/// string from `BuiltinsAMDGPU.def`, quoted here so a toolchain bump that drops
/// support fails loudly instead of silently falling back to a scalar path.
const PINNED: &[(&str, &str, &str)] = &[
    // Raven — V_DOT4_F32_FP8_FP8. NOTE: `dot4`, not `fdot4`.
    (
        "Raven",
        "__builtin_amdgcn_dot4_f32_fp8_fp8",
        "dot11-insts",
    ),
    // WhiteRaven — V_WMMA_F32_16X16X16_FP8_FP8. `_w32` + `_gfx12` are both required.
    (
        "WhiteRaven",
        "__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12",
        "gfx12-insts,wavefrontsize32",
    ),
    // GreyRaven — V_SWMMAC_F32_16X16X32_FP8_FP8 (2:4 sparse). `_w32` required.
    (
        "GreyRaven",
        "__builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32",
        "gfx12-insts,wavefrontsize32",
    ),
    // Raven C5 — RDNA4 pack-converting FP8 CVT.
    (
        "Raven-cvt-pk",
        "__builtin_amdgcn_cvt_pk_fp8_f32",
        "fp8-conversion-insts",
    ),
    (
        "Raven-cvt-sr",
        "__builtin_amdgcn_cvt_sr_fp8_f32",
        "fp8-conversion-insts",
    ),
    // ForestRaven / WhiteCrow — the integer dot family, used by
    // `grim_sdot4` (dot_gemv.rs) and `grim_dot8_w4a4_gemv`.
    ("ForestRaven", "__builtin_amdgcn_sudot4", "dot8-insts"),
    ("WhiteCrow", "__builtin_amdgcn_sudot8", "dot8-insts"),
    // Existing dense FP16 WMMA, for reference against the FP8 forms.
    (
        "FP16-wmma",
        "__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12",
        "gfx12-insts,wavefrontsize32",
    ),
];

/// Names that must **not** exist — the drift that hid the Raven bug. If a future
/// toolchain ever introduces one of these, the source is still wrong and the
/// mismatch must be revisited rather than silently left in place.
const MUST_NOT_EXIST: &[(&str, &str)] = &[
    // The name grim actually shipped. Kept here so a future bump to a toolchain
    // that adds it surfaces as a test failure demanding a decision, not a
    // silent behaviour change.
    ("Raven-bad-name", "__builtin_amdgcn_fdot4_f32_fp8_fp8"),
    // Reads naturally off the ISA manual; not a real builtin (needs _w32/_gfx12).
    (
        "WhiteRaven-unsuffixed",
        "__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8",
    ),
    (
        "GreyRaven-unsuffixed",
        "__builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8",
    ),
];

fn hipcc() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("HIPCC") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    for c in ["/opt/rocm/bin/hipcc", "/usr/bin/hipcc"] {
        let p = Path::new(c);
        if p.exists() {
            return Some(p.to_path_buf());
        }
    }
    None
}

/// Target arch to probe for. gfx1201 is the RDNA4 part the FP8 WMMA and sparse
/// WMMA forms require; on anything older those two are expected to be absent,
/// so the caller must scope the assertion accordingly.
fn probe_arch() -> String {
    std::env::var("GRIM_PROBE_ARCH").unwrap_or_else(|_| "gfx1201".to_string())
}

/// Compile a probe that reports `__has_builtin` for every name, via
/// `#pragma message` so the verdict is emitted for the *device* pass (AMDGPU
/// builtins and arch macros are invisible in host code — probing from `main()`
/// reports 0 for everything, which is how this test's first draft was wrong).
fn probe_src() -> String {
    let mut s = String::from(
        "#if defined(__has_builtin)\n#  define PROBE(x) __has_builtin(x)\n#else\n#  define PROBE(x) 0\n#endif\n\
         #define STR2(x) #x\n#define STR(x) STR2(x)\n",
    );
    for (i, (_, name, _)) in PINNED.iter().enumerate() {
        s.push_str(&format!(
            "#pragma message(\"CORVID {i} {}\")\n#pragma message(\"PROBE {i} \" STR(PROBE({})))\n",
            name, name
        ));
    }
    for (i, (_, name)) in MUST_NOT_EXIST.iter().enumerate() {
        s.push_str(&format!(
            "#pragma message(\"CORVIDX {i} {}\")\n#pragma message(\"PROBEX {i} \" STR(PROBE({})))\n",
            name, name
        ));
    }
    s.push_str("int main(){return 0;}\n");
    s
}

/// Extract a probe verdict from compiler output.
///
/// A `#pragma message` line looks like:
/// ```text
/// probe.cpp:8:9: warning: PROBE 0 1 [-W#pragma-messages]
/// ```
/// so the verdict is the token two positions after the `PROBE` tag. Taking the
/// *last* token gets `[-W#pragma-messages]` — which is how this module's first
/// draft reported every builtin as missing.
///
/// clang also echoes the offending source line, and that echo contains the same
/// `PROBE <i>` text, so only lines carrying `warning:` are considered.
fn verdict(text: &str, tag: &str, index: usize) -> Option<bool> {
    let needle = format!("{tag} {index}");
    for line in text.lines() {
        if !line.contains("warning:") || !line.contains(&needle) {
            continue;
        }
        let toks: Vec<&str> = line.split_whitespace().collect();
        let Some(pos) = toks.iter().position(|t| *t == tag) else {
            continue;
        };
        return toks.get(pos + 2).and_then(|v| v.parse::<u32>().ok()).map(|v| v != 0);
    }
    None
}

/// Run the probe. Returns `(label, name, available)` for every pinned builtin.
fn run_probe() -> Option<Vec<(String, String, bool)>> {
    let cc = hipcc()?;
    let arch = probe_arch();
    let dir = std::env::temp_dir().join(format!("grim-builtin-oracle-{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok()?;
    let src = dir.join("probe.cpp");
    let bin = dir.join("probe");
    std::fs::write(&src, probe_src()).ok()?;

    let out = Command::new(&cc)
        .args([
            &format!("--offload-arch={arch}"),
            "-std=c++17",
            "-x",
            "hip",
            "-o",
        ])
        .arg(&bin)
        .arg(&src)
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stderr).into_owned();

    // A probe that failed to compile yields no `warning:` lines at all. Treat
    // that as a skip rather than as "every builtin is missing" — otherwise a
    // broken toolchain path masquerades as a real regression.
    let mut results = Vec::new();
    for (i, (label, name, _)) in PINNED.iter().enumerate() {
        results.push((
            label.to_string(),
            name.to_string(),
            verdict(&text, "PROBE", i).unwrap_or(false),
        ));
    }
    for (i, (label, name)) in MUST_NOT_EXIST.iter().enumerate() {
        results.push((
            label.to_string(),
            name.to_string(),
            verdict(&text, "PROBEX", i).unwrap_or(false),
        ));
    }
    let _ = std::fs::remove_dir_all(&dir);

    if !text.contains("PROBE 0") {
        eprintln!("skip: probe produced no verdicts (compile failed?); not asserting");
        return None;
    }
    Some(results)
}

/// RED for the Raven bug: every pinned builtin must be one the toolchain knows.
///
/// Before the `fdot4` -> `dot4` fix this failed for `Raven`. It is the direct
/// guard against the whole failure class.
#[test]
fn every_pinned_builtin_is_known_to_the_toolchain() {
    let Some(results) = run_probe() else {
        eprintln!("skip: no hipcc found; builtin-name oracle not exercised");
        return;
    };
    let arch = probe_arch();
    let mut missing = Vec::new();
    for (label, name, ok) in results.iter().take(PINNED.len()) {
        // The FP8 WMMA and sparse-WMMA forms are gfx12-only; on older targets
        // their absence is correct, not drift.
        if !ok && arch.starts_with("gfx12") {
            missing.push(format!("{label}: {name} unknown on {arch}"));
        } else if !ok {
            eprintln!("note: {label} ({name}) absent on {arch} — expected pre-gfx12");
        }
    }
    assert!(
        missing.is_empty(),
        "toolchain is missing builtins grim depends on; the __has_builtin guards \
         will silently take their scalar fallbacks:\n  {}",
        missing.join("\n  ")
    );
}

/// The names that hid the bug must stay non-existent, so the fix is not undone
/// by a toolchain bump that happens to add an alias.
#[test]
fn known_bad_builtin_names_stay_absent() {
    let Some(results) = run_probe() else {
        eprintln!("skip: no hipcc found");
        return;
    };
    for (label, name, present) in results.iter().skip(PINNED.len()) {
        assert!(
            !present,
            "{label}: {name} now EXISTS. grim does not call it and must not — \
             re-examine whether the source should move to this name or keep the \
             current spelling. Do not silently switch."
        );
    }
}

/// Source-level guard: the FP8 dot helper must reference the name clang knows.
/// Complements the compile oracle by catching the edit without a toolchain.
#[test]
fn fp8_dot_helper_uses_the_real_builtin_name() {
    let src = crate::kernels::dot_gemv::KERNEL_SOURCE;
    assert!(
        src.contains("__builtin_amdgcn_dot4_f32_fp8_fp8"),
        "grim_fdot4_fp8 must call __builtin_amdgcn_dot4_f32_fp8_fp8 \
         (dot4-insts). Without the builtin the __has_builtin guard is false and \
         the kernel silently runs a scalar powf-based decode."
    );
    assert!(
        !src.contains("__builtin_amdgcn_fdot4_f32_fp8_fp8"),
        "__builtin_amdgcn_fdot4_f32_fp8_fp8 does not exist in any ROCm clang. \
         It was a typo for dot4 and made the native FP8 dot unreachable."
    );
}

/// The scalar fallback must stay *correct* even though it is now unreachable on
/// gfx12 — RDNA2/RDNA3 have no dot11-insts and still take this branch.
#[test]
fn fp8_dot_fallback_remains_present_for_pre_dot11_arches() {
    let src = crate::kernels::dot_gemv::KERNEL_SOURCE;
    assert!(
        src.contains("fp8_e4m3_to_float_hip"),
        "the #else branch must keep a software decode; gfx10/gfx11 lack dot11-insts"
    );
}
