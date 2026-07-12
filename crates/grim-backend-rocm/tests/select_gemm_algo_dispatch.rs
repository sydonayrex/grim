//! RED-GREEN tests for the `select_gemm_algo` dispatch helper.
//!
//! Bug context (grim_qkv_attention_kernel_spec.md section 3.6
//! cross-task correction):
//!
//! > grim's existing `lookup_solution_index` table is currently a
//! > silent no-op: it calls `rocblas_gemm_ex` with
//! > `algo = rocblas_gemm_algo::standard` together with a non-zero
//! > `solution_index`. Per rocBLAS semantics, `solution_index` is
//! > **ignored** unless `algo == rocblas_gemm_algo::solution_index`.
//!
//! The `select_gemm_algo` helper exists already; what's missing is
//! pinning its behavior as a test so a regression to "always
//! standard" fails CI. These tests exercise the function directly
//! (no GPU required) and a call-site guard (`include_str!` of
//! device/roc_device.rs) that asserts every rocBLAS GEMM FFI call
//! routes through the helper rather than passing
//! `rocblas_gemm_algo::standard` directly.
//!
//! Skill attribution:
//! - `rust-gpu-discipline` #11 - silent-no-op detection: every
//!   tuned value must reach the FFI to count as 'tuned'.
//! - `rocm-profiling-perf` - autotune cache hits only matter if
//!   the dispatch honors them.

use grim_backend_rocm::select_gemm_algo;
use grim_backend_rocm::rocblas_gemm_algo;

/// RED: solution_index == 0 must dispatch to `standard`, never to
/// `solution_index`. Otherwise every GEMM call (including the
/// pre-warmup pass that we don't want to pay solution_index
/// selection costs for) would request a specific rocBLAS engine.
#[test]
fn zero_solution_index_uses_standard_algo() {
    let algo = select_gemm_algo(0);
    assert_eq!(
        algo,
        rocblas_gemm_algo::standard,
        "solution_index=0 must dispatch to rocblas_gemm_algo::standard"
    );
}

/// RED: any non-zero solution_index must dispatch to
/// `rocblas_gemm_algo::solution_index`. Without this the tuned
/// index is silently dropped.
#[test]
fn nonzero_solution_index_uses_solution_index_algo() {
    for &n in &[1_i32, 4, 65, 100, i32::MAX] {
        let algo = select_gemm_algo(n);
        assert_eq!(
            algo,
            rocblas_gemm_algo::solution_index,
            "solution_index={} must dispatch to rocblas_gemm_algo::solution_index",
            n
        );
    }
}

/// RED: the bound between input and output is exact - there is no
/// "fallback to standard" once a non-zero index is requested.
#[test]
fn boundary_at_zero_is_exact_no_partial_fallback() {
    assert_eq!(select_gemm_algo(-1), rocblas_gemm_algo::solution_index);
    assert_eq!(select_gemm_algo(1), rocblas_gemm_algo::solution_index);
    assert_eq!(select_gemm_algo(0), rocblas_gemm_algo::standard);
}

/// RED: dispatch is a pure function - same input always yields same
/// output (the autotune cache assumes this when it pipes cached
/// solution_index through the same dispatch path repeatedly).
#[test]
fn dispatch_is_deterministic() {
    for n in [0_i32, 1, 2, 4, 8, 11, 65, 100, -1, i32::MAX, i32::MIN] {
        let a = select_gemm_algo(n);
        let b = select_gemm_algo(n);
        assert_eq!(a, b, "select_gemm_algo({n}) non-deterministic");
    }
}

/// RED -- call-site guard for the silent-no-op bug fix.
///
/// Every rocBLAS GEMM FFI call in `RocmDevice` (`matmul`,
/// `matmul_batched`, `matmul_with_solution`) takes an `algo` enum
/// and a `solution_index` integer. Per rocBLAS semantics, the
/// integer is silently ignored unless `algo ==
/// `rocblas_gemm_algo::solution_index`.
///
/// This test scans at compile time the call-site file
/// (`device/roc_device.rs`) and asserts two structural properties
/// for every `rocblas_gemm_ex(...)` /
/// `rocblas_gemm_strided_batched_ex(...)` block:
///
/// 1. The `algo` argument is `select_gemm_algo(<some expr>)`, never
///    `rocblas_gemm_algo::standard` directly.
/// 2. The block also passes a `solution_index` argument, otherwise
///    the lookup table would be a silent no-op.
///
/// Both shapes of the FFI (`gemm_ex` and `strided_batched_ex`) are
/// checked -- the `matmul_batched` path uses the strided variant.
/// This catches any future regression that bypasses the helper.
/// Pure-Rust; no GPU required.
#[test]
fn every_gemm_call_site_uses_select_gemm_algo() {
    static SRC: &str = include_str!("../src/device/roc_device.rs");
    let bytes = SRC.as_bytes();

    // Both FFI shapes that take an `algo` argument.
    let call_openers: &[&[u8]] = &[
        b"rocblas_gemm_ex(",
        b"rocblas_gemm_strided_batched_ex(",
    ];
    // Pre-flight: at least 3 call sites in matmul + matmul_batched +
    // matmul_with_solution. Catches the case where someone rewrites
    // the GEMM surface to a different FFI symbol entirely.
    let openers_total: usize = call_openers.iter().map(|n| count_subslice(bytes, n)).sum();
    assert!(
        openers_total >= 3,
        "expected at least 3 rocBLAS GEMM FFI call sites in device/roc_device.rs \
         (matmul + matmul_batched + matmul_with_solution) -- found {openers_total}. \
         Update this test if the dispatch surface changed."
    );

    let mut cursor = 0usize;
    let mut call_idx = 0usize;
    loop {
        // Find the next opener (across both patterns) past `cursor`.
        let next = call_openers
            .iter()
            .enumerate()
            .filter_map(|(i, n)| find_subslice(&bytes[cursor..], n).map(|off| (i, off)))
            .min_by_key(|(_, off)| *off);
        let Some((opener_idx, pos)) = next else { break };
        let abs = cursor + pos;
        let opener = call_openers[opener_idx];
        let close = match_call_close(&bytes[abs + opener.len()..])
            .unwrap_or_else(|| panic!("gemm call #{call_idx} missing closing `)`"));
        let block_bytes = &bytes[abs + opener.len()..abs + opener.len() + close];
        let block = std::str::from_utf8(block_bytes).expect("utf-8");
        assert!(
            block.contains("select_gemm_algo("),
            "gemm call #{call_idx} (opener={}) at byte {abs} passes `algo` without \
             routing through `select_gemm_algo(solution_index)` -- this is the \
             silent-no-op regression fix #1.\nBlock contents:\n{block}",
            std::str::from_utf8(opener).unwrap_or("?"),
        );
        assert!(
            block.contains("solution_index"),
            "gemm call #{call_idx} (opener={}) does not pass a `solution_index` argument -- \
             the lookup table will be a silent no-op.",
            std::str::from_utf8(opener).unwrap_or("?"),
        );
        cursor = abs + opener.len() + close + 1;
        call_idx += 1;
    }
}

// --- helpers for the call-site scan ---

/// Naive `memmem`-style byte-slice substring search.
fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// Count non-overlapping occurrences of `needle` in `haystack`.
fn count_subslice(haystack: &[u8], needle: &[u8]) -> usize {
    if needle.is_empty() {
        return 0;
    }
    let mut count = 0usize;
    let mut start = 0usize;
    while let Some(pos) = find_subslice(&haystack[start..], needle) {
        count += 1;
        start += pos + needle.len();
    }
    count
}

/// Walk `bytes` (assumed to start right after an opening `(`) to the
/// matching `)`. Counts parens, skipping over `//` line comments,
/// `/* ... */` block comments, and `"string"` literals, so a paren
/// inside a comment or string doesn't break the walk.
fn match_call_close(bytes: &[u8]) -> Option<usize> {
    let mut depth: usize = 1;
    let mut i = 0usize;
    while i < bytes.len() {
        let c = bytes[i];
        match c {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            b'/' if bytes.get(i + 1) == Some(&b'/') => {
                if let Some(end) = bytes[i..].iter().position(|c| *c == b'\n') {
                    i += end + 1;
                    continue;
                }
            }
            b'/' if bytes.get(i + 1) == Some(&b'*') => {
                if let Some(end_off) = bytes[i + 2..].windows(2).position(|w| w == b"*/") {
                    i += 2 + end_off + 2;
                    continue;
                }
            }
            b'"' => {
                i += 1;
                while i < bytes.len() {
                    match bytes[i] {
                        b'\\' => i += 2,
                        b'"' => break,
                        _ => i += 1,
                    }
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}
