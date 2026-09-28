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
//! restored and the `[ctxtrace]` drift watch sees the flip.
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

/// The name of a free function or inherent-impl method declared on this line,
/// or `None` if the line is not a declaration.
///
/// `unsafe fn` and `async fn` are matched too. They used to be missed, which
/// meant a `hipMemcpyAsync` inside an `unsafe fn` was attributed to whatever
/// function happened to be scanned before it — or to none at all, in which case
/// the `if let Some((fname, _))` below skipped the check entirely and the gate
/// reported green.
fn fn_name(trimmed: &str) -> Option<String> {
    const QUALIFIERS: &[&str] = &["pub(crate) ", "pub ", "unsafe ", "async ", "extern \"C\" "];
    let mut rest = trimmed;
    loop {
        match QUALIFIERS.iter().find(|q| rest.starts_with(**q)) {
            Some(q) => rest = &rest[q.len()..],
            None => break,
        }
    }
    let rest = rest.strip_prefix("fn ")?;
    let name: String = rest
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    (!name.is_empty()).then_some(name)
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
/// enforces the audit: every source file containing one of those calls must
/// also contain a `DeviceGuard::set` (or `raw_set_device`) before its use,
/// unless the function is on the by-design allowlist below.
///
/// Purely host-side: parses this crate's own sources; no GPU needed.
///
/// SCOPE: this walks the WHOLE `src/` tree, as `hip_set_device_has_no_bare_call
/// _sites_outside_the_guard_module` already does. It previously audited a
/// hand-maintained list of 15 files; 32 files in this crate make raw
/// device-bound FFI calls, so the other 17 were never checked at all — and
/// three of them carried real, unguarded seams (`graph_capture.rs`'s capture
/// stream, `dot_gemv.rs`'s cached q8_1 event, `optimizer_ops.rs`'s frees).
#[test]
fn raw_device_bound_ffi_calls_sit_inside_guarded_functions() {
    // By-design exceptions (verified by hand, see scythe2 plan log 23e):
    // - try_new: documented context-neutral constructor that pins via
    //   raw_set_device + restore around construction, and its inline lazy
    //   rocblas_create_handle runs inside that pinned window.
    // - fallback: RocmDevice::fallback constructor (no raw launches).
    // - build: hsaco build path; context established by caller.
    const ALLOWED_UNGUARDED: &[&str] = &["try_new", "fallback", "build"];

    /// The FFI declaration/shim layer. `handles.rs` wraps the raw symbols in
    /// tracing shims (`hipMemcpyAsync` -> `raw_hipMemcpyAsync`) and does not
    /// itself decide which device a call belongs to — that stays with the
    /// caller. It is already the sanctioned exception in
    /// `ALLOWED_FILES` above for the same reason.
    const FFI_LAYER: &[&str] = &["device/handles.rs", "device/rocblas.rs"];

    /// Pure FFI declaration blocks. `pub fn hipMemcpyAsync(` here DECLARES the
    /// symbol; it is not a call site, and the old list-based gate sidestepped
    /// the question by never including these files.
    fn extern_block_lines(body: &str) -> Vec<bool> {
        let mut out = Vec::with_capacity(body.lines().count());
        let mut in_extern = false;
        for line in body.lines() {
            if line.contains("extern \"C\"") || line.contains("extern \"C-ABI\"") {
                in_extern = true;
            } else if in_extern && line.trim() == "}" {
                in_extern = false;
            }
            out.push(in_extern);
        }
        out
    }

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

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut violations: Vec<String> = Vec::new();
    let mut audited_files = 0usize;

    for file in rust_sources(&root) {
        let rel = relative_to(&file, &root);
        let body = std::fs::read_to_string(&file)
            .unwrap_or_else(|e| panic!("read {}: {e}", file.display()));
        if !risky.iter().any(|r| body.contains(r)) {
            continue;
        }
        // Breadth is counted BEFORE the FFI-layer exemption, so this number
        // tracks what the sweep sees rather than how many it audits.
        audited_files += 1;
        if FFI_LAYER.contains(&rel.as_str()) {
            continue;
        }
        let in_extern = extern_block_lines(&body);

        let mut current_fn: Option<(String, usize)> = None;
        let mut guarded = false;
        for (idx, line) in body.lines().enumerate() {
            if in_extern[idx] {
                continue;
            }
            let trimmed = line.trim_start();
            if let Some(name) = fn_name(trimmed) {
                current_fn = Some((name, idx + 1));
                guarded = false;
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
        audited_files > 31,
        "expected the whole-crate sweep to cover more files than the old \
         hand-maintained 15-file list did; only {audited_files} carried raw FFI. \
         A shrunken tree means the gate stopped seeing something.",
    );
    assert!(
        violations.is_empty(),
        "P1-3 contract breached — raw device-context FFI outside a guard:\n{}\n\
         Route it through DeviceGuard::set (see matmul_op fix, 2026-08-23e).",
        violations.join("\n")
    );
}

/// The cross-crate seam: raw HIP FFI called from OUTSIDE `grim-backend-rocm`.
///
/// Every seam check in this file used to be scoped to this crate, which is
/// exactly where the discipline is easiest to keep and exactly where it is
/// least likely to be the bug. The cross-card faults live in the callers: a
/// `hipMemcpyAsync` issued from `grim-engine` against a stream it does not own
/// acts on the CALLING THREAD's device, so a drifted thread transfers on the
/// wrong ordinal. That is invisible from inside the backend crate.
///
/// Found by measurement: `scythe2.rs` guarded one of two control-stream copies
/// and not the other, and `run.rs` reached for the raw `hipStreamSynchronize`
/// symbol instead of the sanctioned wrapper. Both are now pinned.
#[test]
fn raw_hip_ffi_outside_the_backend_crate_is_guarded() {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/grim-backend-rocm -> crates/")
        .to_path_buf();
    assert!(workspace.is_dir(), "expected crates dir at {}", workspace.display());

    const RISKY: &[&str] = &[
        "hipMemcpyAsync(",
        "hipMemcpy(",
        "hipMemsetAsync(",
        "hipMalloc(",
        "hipFree(",
        "hipFreeAsync(",
        "hipStreamCreate(",
        "hipStreamSynchronize(",
        "hipEventCreate(",
        "hipModuleLaunchKernel(",
        "rocblas_sgemm(",
        "rocblas_gemm_ex(",
    ];

    let mut violations: Vec<String> = Vec::new();
    let mut sites = 0usize;
    for file in rust_sources(&workspace) {
        let rel = relative_to(&file, &workspace);
        // The backend crate has its own, wider gate above.
        if rel.starts_with("grim-backend-rocm/") {
            continue;
        }
        // `fn` declarations in an extern block are the backend's re-exports.
        if rel.contains("/tests/") || rel.contains("/examples/") {
            continue;
        }
        let body = std::fs::read_to_string(&file)
            .unwrap_or_else(|e| panic!("read {}: {e}", file.display()));
        if !RISKY.iter().any(|r| body.contains(r)) {
            continue;
        }
        let mut current_fn: Option<(String, usize)> = None;
        let mut guarded = false;
        for (idx, line) in body.lines().enumerate() {
            if let Some(name) = fn_name(line.trim_start()) {
                current_fn = Some((name, idx + 1));
                guarded = false;
            }
            if line.contains("DeviceGuard::set") || line.contains("raw_set_device") {
                guarded = true;
            }
            let code = code_only(line);
            if let Some(hit) = RISKY.iter().find(|r| code.contains(**r)) {
                sites += 1;
                if let Some((fname, _)) = &current_fn {
                    if !guarded {
                        violations.push(format!(
                            "{rel}:{}: `{fname}` calls `{hit}` with no DeviceGuard in scope \
                             (raw HIP from outside the backend crate)",
                            idx + 1
                        ));
                    }
                }
            }
        }
    }

    assert!(
        violations.is_empty(),
        "cross-crate HIP seam breached — raw device-context FFI outside \
         grim-backend-rocm with no guard:\n{}\n\
         Either route the call through a guarded backend entry point, or pin \
         the ordinal with DeviceGuard::set before it.",
        violations.join("\n")
    );
    assert!(
        sites > 0,
        "expected at least one cross-crate raw HIP call site to police; found {sites}. \
         If these were all removed, delete this gate rather than leaving it vacuous."
    );
}
