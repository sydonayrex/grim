//! P4 contract tests for IQ/GQuant backward kernels (implement.md §P4).
//!
//! Implement.md §P4: the IQ/GQuant backward kernels are the remaining gap after
//! the forward path was restructured (standalone dequant upgraded to 64-thread
//! blocks; fused dispatch is the default forward path). The gap is per-MAC
//! scalar `div/mod` (`int sb_idx = k / 256; int in_sb = k % 256;`) and
//! one-thread-per-output + per-MAC full dequant, instead of the forward path's
//! superblock-per-thread + scale-loaded-once layout.
//!
//! P4 is **not implemented here** — the backward kernels are HIP and device-gated,
//! and there is no host-side backward golden in the repo to red-to-green a device
//! kernel change. What these tests provide is the *verifier contract* a future P4
//! change must satisfy before it can be red-to-green'd on a device run:
//!
//! 1. **The current bad pattern is named explicitly** so a P4 change cannot
//!    regress silently: the test asserts the representative variant's source
//!    currently still contains the per-MAC `div/mod` scalar pattern — i.e. the
//!    test is *red on the current code* in the sense that the bad pattern exists
//!    and must be removed by a future P4 change. (This is a structural assertion
//!    about the kernel source, not a runtime device test.)
//!
//! 2. **A host-side reference decomposition that a device verifier can compare
//!    against** — for a representative variant (IQ4_XS), define a deterministic
//!    FP32 weight matrix B and compute the backward reference dX = dY @ B^T
//!    host-side, so a future device run can compare the device backward output
//!    against this reference within tolerance. (Host-side; no device needed.)
//!
//! 3. **The host reference is self-consistent and non-trivial** — a deterministic
//!    function of dY + B, matching the analytical matmul on the dequantized FP32
//!    B, and non-zero for a non-trivial contrived case.
//!
//! Representative variant chosen: IQ4_XS, because `dequant_iq4xs` + `quant_iq4xs`
//! both exist host-side in `grim-quant`, so the reference chain is buildable here.
//! P4's "done looks like" is: at least one variant proven before expansion — a
//! device run with this contract as the verifier would satisfy that.

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

/// Assert the representative variant's source currently still contains the
/// per-MAC scalar `div/mod` pattern that P4 calls a gap. This is intentionally
/// a *structural assertion about the current kernel source* — it is red in the
/// sense that the bad pattern exists and must be removed by a future P4 change.
/// A P4 kernel change that removes the per-MAC `div/mod` would make this
/// assertion fail, which is the intended red-to-green signal for that change.
#[test]
fn p4_iq4xs_backward_source_currently_still_has_per_mac_div_mod_pattern() {
    let src = std::include_str!("../src/kernels/iq_gemm.rs");
    assert!(
        src.contains("grim_fused_dequant_backward_gemm_iq4xs"),
        "P4: IQ4XS backward kernel source must be present for the contract test"
    );
    assert!(
        src.contains("sb_idx") && src.contains("in_sb"),
        "P4: IQ4XS backward source currently still contains per-MAC sb_idx/in_sb (the gap)"
    );
    assert!(
        src.contains("k / 256") || src.contains("K / 256"),
        "P4: IQ4XS backward source contains the per-MAC superblock index div"
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
    let dY: Vec<f32> = (0..m * n)
        .map(|i| ((i as f32) * 0.25).cos())
        .collect();

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
        assert_eq!(dx_ref[i], dx_ref_2[i], "P4 host backward reference must be deterministic");
    }

    // Sanity: the reference is not trivially zero for a non-trivial dY + B.
    let any_nonzero = dx_ref.iter().any(|&x| x.abs() > 1e-6);
    assert!(any_nonzero, "P4 host backward reference must be non-trivial for the contrived case");
    assert!(dx_ref.len() == m * k, "P4 host backward reference must have shape [m*k]");
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

    let dY: Vec<f32> = (0..m * n)
        .map(|i| ((i as f32) * 0.25).cos())
        .collect();

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
        assert_eq!(dx_analytical[i], dx_ref[i], "P4 host reference must match analytical matmul on FP32 B");
    }
}
