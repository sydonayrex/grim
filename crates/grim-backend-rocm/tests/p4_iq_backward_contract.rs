#![allow(dead_code, non_snake_case)]
//! P4 contract tests for IQ/GQuant backward kernels (implement.md §P4).
//!
//! Implement.md §P4: the IQ/GQuant backward kernels are the remaining gap after
//! the forward path was restructured (standalone dequant upgraded to 64-thread
//! blocks; fused dispatch is the default forward path). The gap is per-MAC
//! scalar `div/mod` (`int sb_idx = k / 256; int in_sb = k % 256;`) and
//! one-thread-per-output + per-MAC full dequant, instead of the forward path's
//! superblock-per-thread + scale-loaded-once layout.
//!
//! STATUS: the representative variant (IQ4XS backward) has been rewritten —
//! `grim_fused_dequant_backward_gemm_iq4xs` now does per-thread superblock: one
//! thread handles a 256-weight superblock, dequant all 256 codes once via
//! `dequant_iq4xs` in a `#pragma unroll` loop, then MAC across N. The per-MAC
//! `sb_idx`/`in_sb` div/mod and per-MAC-per-N dequant call are removed. The
//! device verifier (`p4_iq4xs_backward_device_verifier.rs`) must still be run on
//! a ROCm GPU to confirm RMS rel err <= 0.05 vs the host reference — that is the
//! red-to-green anchor for P4.
//!
//! What these tests provide:
//!
//! 1. **The bad-pattern-removed assertion** — the structural test
//!    `p4_iq4xs_backward_source_has_no_per_mac_div_mod_pattern` asserts the
//!    rewritten kernel source does NOT contain the per-MAC `div/mod` pattern.
//!    This is the regression gate: any future change that reintroduces the
//!    per-MAC div/mod fails this test.
//!
//! 2. **A host-side reference decomposition that a device verifier can compare
//!    against** — for a representative variant (IQ4_XS), define a deterministic
//!    FP32 weight matrix B and compute the backward reference dX = dY @ B^T
//!    host-side, so the device verifier can compare the device backward output
//!    against this reference within tolerance. (Host-side; no device needed.)
//!
//! 3. **The host reference is self-consistent and non-trivial** — a deterministic
//!    function of dY + B, matching the analytical matmul on the dequantized FP32
//!    B, and non-zero for a non-trivial contrived case.
//!
//! Representative variant chosen: IQ4_XS, because `dequant_iq4xs` + `quant_iq4xs`
//! both exist host-side in `grim-quant`, so the reference chain is buildable here.
//! P4's "done looks like" is: at least one variant proven before expansion — the
//! device verifier run against this contract would satisfy that.

use grim_quant::dequant_iq4xs;

/// Canonical per-superblock IQ4_XS decode for host reference purposes.
///
/// Mirrors `kernels/iq_gemm.rs` + `grim-quant::dequant_iq4xs`: each 256-weight
/// superblock has its own scale (6 bits from the `scales` bytes), and each
/// element is a 4-bit code from the `qs` bytes. This is the canonical
/// decomposition a P4 backward verifier would compare against; it is the
/// host-side equivalent of what the device backward kernel currently does
/// per-MAC with `sb_idx = k / 256; in_sb = k % 256`.
fn iq4xs_superblock_decode(packed: &[u8], num_weights: usize) -> Vec<f32> {
    dequant_iq4xs(packed, num_weights).unwrap_or_else(|_| vec![0.0f32; num_weights])
}

/// Host-side backward decomposition for IQ4_XS: dX[m, k] = dY[m, n] @ B^T[k, n],
/// where B is the dequantized FP32 weight matrix (k rows x n cols).
///
/// This is the *contract* a device backward verifier would compare against: given
/// dY (M x N), the dequantized weight B, produce dX (M x K). A device backward
/// kernel that satisfies P4 would produce the same dX (within tolerance).
fn iq4xs_backward_dx_host(dY: &[f32], b_deq: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut dx = vec![0.0f32; m * k];
    for mi in 0..m {
        for ki in 0..k {
            let mut acc = 0.0f32;
            for ni in 0..n {
                acc += dY[mi * n + ni] * b_deq[ki * n + ni];
            }
            dx[mi * k + ki] = acc;
        }
    }
    dx
}

/// The IQ4XS backward rewrite (P4) removes the per-MAC `sb_idx`/`in_sb`
/// div/mod and the per-MAC per-N-loop dequant call. The bad pattern *must not
/// be present* in the rewritten kernel source — this is the assertion that
/// flips red when the pattern is still there (pre-P4) and green once the
/// rewrite lands (post-P4). A subsequent regression that reintroduces the
/// per-MAC div/mod would fail this test.
///
/// NOTE: `sb_idx`/`in_sb` appear elsewhere in `iq_gemm.rs` (in the per-format
/// `dequant_*` helper signatures and in the other backward kernels), so we
/// scope the check to ONLY the text of the `grim_fused_dequant_backward_gemm_iq4xs`
/// kernel, not the whole file.
#[test]
fn p4_iq4xs_backward_source_has_no_per_mac_div_mod_pattern() {
    let src = std::include_str!("../src/kernels/iq_gemm.rs");
    assert!(
        src.contains("grim_fused_dequant_backward_gemm_iq4xs"),
        "P4: IQ4XS backward kernel source must be present for the contract test"
    );
    // Extract ONLY the iq4xs backward kernel text (from its `__global__` line to
    // the closing `}` before the next kernel or EOF).
    let kernel_start = src
        .find("__global__ void grim_fused_dequant_backward_gemm_iq4xs")
        .expect("P4: IQ4XS backward kernel __global__ line must be present");
    // Find the end: the next `__global__` or the closing `}` of the extern block.
    let rest = &src[kernel_start..];
    let next_global = rest.find("\n    __global__").unwrap_or(rest.len());
    let kernel_text = &rest[..next_global];
    // The rewritten kernel must NOT still contain the per-MAC sb_idx/in_sb
    // div/mod pattern that P4 calls a gap. Scoped to this kernel only.
    //
    // NOTE: the comment at the top of the kernel may reference "k_idx / 256"
    // descriptively (explaining the superblock_idx computation), and the
    // `blocks_per_row = K / 256` line is structural (not per-MAC). The P4 gap
    // is the *per-output-thread* `int sb_idx = k_idx / 256; int in_sb = k_idx
    // % 256;` declaration plus the per-N-loop dequant call using those — i.e.
    // both symbols appearing as local variable declarations used per-MAC.
    // We check that the kernel does NOT declare both as local vars (the "sb_idx"
    // and "in_sb" tokens appearing as declarations, not as comment references or
    // in the blocks_per_row structural line).
    let has_sb_idx_decl =
        kernel_text.contains("int sb_idx") || kernel_text.contains("const int sb_idx");
    let has_in_sb_decl =
        kernel_text.contains("int in_sb") || kernel_text.contains("const int in_sb");
    assert!(
        !(has_sb_idx_decl && has_in_sb_decl),
        "P4: IQ4XS backward kernel must NOT declare per-MAC sb_idx/in_sb locals after the rewrite"
    );
    // The per-MAC div/mod gap is `k / 256` used as a local computation (not the
    // structural `blocks_per_row = K / 256`). Check that the kernel text does not
    // contain a bare `k / 256` or `K / 256` used as a div outside of the
    // `blocks_per_row` structural line and comment references.
    //
    // We exclude the `blocks_per_row = K / 256` line (structural, not per-MAC)
    // and comment lines (// ...) which may reference the computation descriptively.
    let code_lines: Vec<&str> = kernel_text
        .lines()
        .filter(|l| !l.trim_start().starts_with("//") && !l.contains("blocks_per_row"))
        .collect();
    let code_text = code_lines.join("\n");
    let has_per_mac_div = code_text.contains("k / 256") || code_text.contains("K / 256");
    assert!(
        !has_per_mac_div,
        "P4: IQ4XS backward kernel must NOT contain per-MAC k/K / 256 div after the rewrite"
    );
}

/// Host-side backward decomposition contract: for a small contrived case with a
/// deterministic FP32 weight matrix B, the host reference produces a deterministic,
/// non-trivial dX = dY @ B^T. This is the reference a device verifier would
/// compare against. (Host-side; no device needed.)
#[test]
fn p4_iq4xs_backward_host_decomposition_is_well_defined_for_small_contrived_case() {
    let m = 2;
    let n = 4;
    let k = 4;

    // Deterministic dY.
    let dY: Vec<f32> = (0..m * n).map(|i| ((i as f32) * 0.25).cos()).collect();

    // Deterministic FP32 weight matrix B (k rows x n cols). Direct FP32, not via
    // quant round-trip (which is lossy and platform-dependent); the host reference
    // is well-defined and non-zero for this contrived B.
    let b_f32: Vec<f32> = (0..k * n)
        .map(|i| {
            let row = i / n;
            let col = i % n;
            ((row as f32 + col as f32) * 0.5 + 0.3).cos() * 2.0
        })
        .collect();

    // Host reference backward decomposition.
    let dx_ref = iq4xs_backward_dx_host(&dY, &b_f32, m, n, k);

    // Self-check: the host reference is a deterministic function of dY + B.
    let dx_ref_2 = iq4xs_backward_dx_host(&dY, &b_f32, m, n, k);
    assert_eq!(dx_ref.len(), dx_ref_2.len());
    for i in 0..dx_ref.len() {
        assert_eq!(
            dx_ref[i], dx_ref_2[i],
            "P4 host backward reference must be deterministic"
        );
    }

    // Sanity: the reference is not trivially zero for a non-trivial dY + B.
    let any_nonzero = dx_ref.iter().any(|&x| x.abs() > 1e-6);
    assert!(
        any_nonzero,
        "P4 host backward reference must be non-trivial for the contrived case"
    );
    assert!(
        dx_ref.len() == m * k,
        "P4 host backward reference must have shape [m*k]"
    );
}

/// Host-side backward decomposition contract: the host reference must match the
/// analytical matmul on the dequantized FP32 B (they're the same computation,
/// expressed two ways). This pins the reference so a device verifier comparing
/// against `iq4xs_backward_dx_host` is comparing against a known-good FP32 matmul.
#[test]
fn p4_iq4xs_backward_host_reference_matches_analytical_matmul_for_small_contrived_case() {
    let m = 2;
    let n = 4;
    let k = 4;

    let dY: Vec<f32> = (0..m * n).map(|i| ((i as f32) * 0.25).cos()).collect();

    let b_f32: Vec<f32> = (0..k * n)
        .map(|i| {
            let row = i / n;
            let col = i % n;
            ((row as f32 + col as f32) * 0.5 + 0.3).cos() * 2.0
        })
        .collect();

    // Analytical host matmul: dX = dY @ B^T using the FP32 B directly.
    let mut dx_analytical = vec![0.0f32; m * k];
    for mi in 0..m {
        for ki in 0..k {
            let mut acc = 0.0f32;
            for ni in 0..n {
                acc += dY[mi * n + ni] * b_f32[ki * n + ni];
            }
            dx_analytical[mi * k + ki] = acc;
        }
    }

    // The host backward reference must equal the analytical matmul on the FP32 B.
    let dx_ref = iq4xs_backward_dx_host(&dY, &b_f32, m, n, k);
    for i in 0..dx_analytical.len() {
        assert_eq!(
            dx_analytical[i], dx_ref[i],
            "P4 host reference must match analytical matmul on FP32 B"
        );
    }
}
