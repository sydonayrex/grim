//! WI-M1 source-level contract gate (gguf_multigpu_context_plan.md).
//!
//! The ctx_dev=2 page fault exists because some seam executed HIP work while
//! the calling thread's context was parked on another device. M1 pins every
//! such seam; this gate makes the discipline structurally enforceable:
//!
//! **The only permitted `hipSetDevice` call sites in this crate are**
//! - `src/device/handles.rs` — the FFI *declaration* itself;
//! - `src/device/util.rs`   — `DeviceGuard` (save/restore RAII) and
//!   `raw_set_device` (the traced setter for the legitimate unguarded
//!   callers: `RocmDevice::try_new` construction and `peer_access.rs`'s
//!   save/restore pair).
//!
//! Any new bare `hipSetDevice(` anywhere else fails this test. Route it
//! through `DeviceGuard::set` / `raw_set_device` instead so the context is
//! restored and the `[ctx-trace]` drift watch sees the flip.
//!
//! Purely host-side: reads this crate's own sources from disk, no GPU needed.

use std::path::{Path, PathBuf};

/// The two files allowed to spell `hipSetDevice(` in call position.
const ALLOWED_FILES: &[&str] = &["device/handles.rs", "device/util.rs"];

/// Reduce a source line to its code content: drop `//` comment tails and the
/// contents of ordinary string literals, so error-message text like
/// `"hipSetDevice({ordinal}) failed"` does not masquerade as a call site.
/// Raw strings / char literals containing quotes are not used near the
/// audited seams; if one ever carries this token the test over-reports and a
/// human can move the literal — fail-loud is the safe direction for a gate.
fn code_only(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut in_str = false;
    let mut escaped = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if in_str {
            if escaped {
                escaped = false;
                continue;
            }
            match c {
                '\\' => escaped = true,
                '"' => in_str = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '/' if chars.peek() == Some(&'/') => break,
            _ => out.push(c),
        }
    }
    out
}

fn rust_sources(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let entries = match std::fs::read_dir(&d) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().map(|e| e == "rs").unwrap_or(false) {
                found.push(p);
            }
        }
    }
    found.sort();
    found
}

fn relative_to(file: &Path, root: &Path) -> String {
    file.strip_prefix(root)
        .unwrap_or(file)
        .to_string_lossy()
        .replace('\\', "/")
}

#[test]
fn hip_set_device_has_no_bare_call_sites_outside_the_guard_module() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    assert!(
        root.is_dir(),
        "expected crate src dir at {}",
        root.display()
    );

    let mut violations: Vec<String> = Vec::new();
    for file in rust_sources(&root) {
        let rel = relative_to(&file, &root);
        let allowed = ALLOWED_FILES.contains(&rel.as_str());
        let body = std::fs::read_to_string(&file)
            .unwrap_or_else(|e| panic!("read {}: {e}", file.display()));
        for (idx, line) in body.lines().enumerate() {
            let code = code_only(line);
            if code.contains("hipSetDevice(") && !allowed {
                violations.push(format!(
                    "{}:{}: bare hipSetDevice call site: {}",
                    rel,
                    idx + 1,
                    line.trim()
                ));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "WI-M1 contract breached — route context switches through \
         DeviceGuard::set / raw_set_device:\n{}",
        violations.join("\n")
    );
}

#[test]
fn guard_module_still_provides_both_sanctioned_setters() {
    // The allow-list above is meaningless if util.rs stops providing the
    // sanctioned setters, or handles.rs stops declaring the FFI. Pin their
    // existence so deleting one cannot silently widen the contract.
    let util = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/device/util.rs"),
    )
    .expect("util.rs exists");
    assert!(
        util.contains("pub fn raw_set_device"),
        "raw_set_device must stay available for the legitimate unguarded callers"
    );
    assert!(
        util.contains("impl DeviceGuard"),
        "DeviceGuard must remain the RAII pin used at every guarded seam"
    );

    let handles = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/device/handles.rs"),
    )
    .expect("handles.rs exists");
    assert!(
        handles.contains("pub fn hipSetDevice"),
        "the FFI declaration lives in handles.rs"
    );
}

#[test]
fn peer_access_save_restore_pair_stays_balanced() {
    // WI-M1 audit result: peer_access manages its own prev/save pair around
    // hipDeviceEnablePeerAccess (which acts on the CURRENT device). It must
    // switch via raw_set_device exactly twice per grant: once to `src`, once
    // to restore. A third switch here would be an unpinned drift source.
    let body = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/peer_access.rs"),
    )
    .expect("peer_access.rs exists");
    let switches = body
        .lines()
        .filter(|l| l.contains("raw_set_device("))
        .count();
    assert_eq!(
        switches, 2,
        "peer_access.rs must keep exactly the set(src)/restore(prev) pair"
    );
}

/// WI-P13 (2026-08-23e follow-up): every raw device-context-sensitive FFI
/// call — rocBLAS GEMMs, `hipModuleLaunchKernel`, `hipMemcpy(Async)`,
/// `hipMemset(Async)`, `hipMalloc`/`hipFree(Async)`, `hipEventCreate`,
/// `hipStreamCreate` — executes against the CALLING THREAD's current HIP
/// device. The rank-1 zero-logits crash existed because `matmul_op` /
/// `matmul_with_solution` launched rocBLAS on a drifted thread. This gate
/// enforces the audit: in `device/roc_device.rs` and `p2p_route.rs`, any
/// function body containing one of those calls must also contain a
/// `DeviceGuard::set` (or `raw_set_device`) before its use, unless the
/// function is on the by-design allowlist below.
///
/// Purely host-side: parses this crate's own sources; no GPU needed.
#[test]
fn raw_device_bound_ffi_calls_sit_inside_guarded_functions() {
    // By-design exceptions (verified by hand, see scythe2 plan log 23e):
    // - try_new: documented context-neutral constructor that pins via
    //   raw_set_device + restore around construction, and its inline lazy
    //   rocblas_create_handle runs inside that pinned window.
    // - fallback: RocmDevice::fallback constructor (no raw launches).
    const ALLOWED_UNGUARDED: &[&str] = &["try_new", "fallback", "build"];

    let audited: &[(&str, &str)] = &[
        ("device/roc_device.rs", include_str!("../src/device/roc_device.rs")),
        ("p2p_route.rs", include_str!("../src/p2p_route.rs")),
    ];

    let risky = [
        "rocblas_sgemm(",
        "rocblas_gemm_ex(",
        "rocblas_gemm_strided_batched_ex(",
        "hipModuleLaunchKernel(",
        "hipMemcpyAsync(",
        "hipMemsetAsync(",
        "hipMalloc(",
        "hipMallocManaged(",
        "hipFree(",
        "hipFreeAsync(",
        "hipEventCreate(",
        "hipStreamCreate(",
    ];

    let mut violations: Vec<String> = Vec::new();
    for (rel, body) in audited {
        let mut current_fn: Option<(String, usize)> = None;
        let mut guarded = false;
        for (idx, line) in body.lines().enumerate() {
            let trimmed = line.trim_start();
            if let Some(rest) = trimmed.strip_prefix("fn ").or_else(|| {
                trimmed
                    .strip_prefix("pub fn ")
                    .or_else(|| trimmed.strip_prefix("pub(crate) fn "))
            }) {
                let name: String = rest
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                if !name.is_empty() {
                    current_fn = Some((name, idx + 1));
                    guarded = false;
                }
            }
            if line.contains("DeviceGuard::set") || line.contains("raw_set_device") {
                guarded = true;
            }
            let code = code_only(line);
            if risky.iter().any(|r| code.contains(r)) {
                if let Some((fname, _)) = &current_fn {
                    if !guarded && !ALLOWED_UNGUARDED.contains(&fname.as_str()) {
                        violations.push(format!(
                            "{rel}:{}: `{fname}` issues a context-bound FFI call with no DeviceGuard in scope",
                            idx + 1
                        ));
                    }
                }
            }
        }
    }
    assert!(
        violations.is_empty(),
        "P1-3 contract breached — raw device-context FFI outside a guard:\n{}\n\
         Route it through DeviceGuard::set (see matmul_op fix, 2026-08-23e).",
        violations.join("\n")
    );
}
