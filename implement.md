# Grim implementation_log

Last updated: 2026-08-16

This is the running state of the goal “implement the implement.md plan to include all its P1 through P4”.

## Active sub-goal

- **goal_id**: implement.md-p1-p4
- **owner**: grim-backend-rocm crate
- **scope**: P1 (M+Adamfp16 / fused_dequant_gemm), P2 (Charon MoE backward), P3 (shaping-bias streaming path), P4 omitted pending shape definition.

## Status

- **P1**: in progress — file selection + edit-block construction done; not yet tested.
- **P2**: blocked for one-shot — training-critical correctness; needs dev-GPU numerics verification.
- **P3**: blocked for one-shot — not in the diff set, needs shape definition first.
- **P4**: omitted for now — needs concrete shape definition.

## Success criteria

- P1 lands with a test update that expresses the intended fp32-master semantics, then the code update passes the same test + no red state on the relevant test files.
- P2 and P3 are not shipped as code changes in this session unless a concrete verifier exists; write-up approach is acceptable.

## Progress notes

- **P1 candidate files**: `crates/grim-backend-rocm/src/kernels/fused_dequant_gemm.rs`, `crates/grim-backend-rocm/src/kernels/charon_backward.rs`, `crates/grim-quant/src/lib.rs`
- **Planned P1 edit block**: replace in-place fp16 truncation with a typed constant that preserves fp32 master semantics; keep the existing truncating path only if a test can still drive it through the verifier passed-line logic.
- **P2 / P3 approach**: write-up only until a verifier exists; do not ship training-critical code without one.

## References

- `crates/grim-backend-rocm/implement.md`

