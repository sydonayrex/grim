# WI-F2-KERNEL-BROKEN — fused_qkv_proj / fused_attn_o_proj dormant kernels are broken on the real GPU path

Status: BLOCKED (2026-09-16). Wiring reverted; document only.

PLAN-kernel-fusion §F2 assumed `fused_qkv_proj`/`fused_attn_o_proj` (ROCm,
`device_compute.rs:5053`/`:5088`) were "tested, dormant" — kernel-level parity
present, only integration logic missing. FALSE for `fused_qkv_proj`.

## Measured evidence (gfx1201, worktree grim-fusion, commit as of 90d2)

`GRIM_GPU_TEST=1 cargo test -p grim-backend-rocm --test qkv_proj_fusion`:

```
qkv_proj_fusion_matches_unfused ... FAILED
    called `Result::unwrap()` on an `Err` value:
    ShapeMismatch { expected: [4, 64], got: [64, 192] }
qkv_proj_fused_uses_single_launch ... FAILED (same ShapeMismatch class)
```

`fused_qkv_proj` validates/shapes the fused weight badly (`matmul_op`
internals receive the `[hidden, n_total]` concat built by
`concat_qkv_weights` and reject the shape). Independent hostile check with a
raw A×B (plain F32 data, no LFM2) reproduced full-magnitude divergence
(`got[0..4]=[-2.05, 3.34, ...]` vs reference `[2.91, -0.12, ...]`) even when
the call was coaxed past the shape error — the kernel never produced correct
output on this path. This is a live kernel bug, not a wiring gap.

## What this means

- F2 (wiring `fused_qkv_proj` / `fused_attn_o_proj` into LFM2) is blocked
  until the underlying fused GEMM path produces correct output on
  `[tokens, hidden] × [hidden, n_total]` — currently neither stock
  kernel-level parity nor host-reference parity hold.
- The wiring was implemented (load-time `w_t` concat + call-site +
  `fused_qkv_f32_decode` + `lfm2_f32_qkv_fusion` integration test) and
  REVERTED the moment the parity requirement surfaced — per F2's gate,
  "do not wire without the new integration parity test passing," and the
  parity evidence says NO.
- F2 wiring required NO LFM2 structural change other than the call site —
  any future fix to the backend's fused GEMM path (or the shape-validation
  branch inside `fused_qkv_proj`) unblocks the wiring directly.

## What still works

- Q8_0 fused dot4 path (`wqkv_q80_fused` + `fused_qkv_dot4_decode`) — real,
  default-on, verified end-to-end (P4 test asserts fewer launches than
  plain; CLI e2e produces sane greedy output on LFM2.5-350M-Q8_0).
- F1 (`fused_add_rms_norm` seam) — wired and verified (32 hits/ref run,
  e2e output matches).
- B1 device sampler + penalty — live and GPU-parity-verified.

## Fix path for a future session

1. Fix `fused_qkv_proj`'s shape handling (`matmul_op(
   x, w, out, GemmOp::Attention)`) so the kernel-level tests in
   `crates/grim-backend-rocm/tests/qkv_proj_fusion.rs` pass.
2. Only then re-apply the wiring (the reverted diff is in git history at
   commit path under `fusion-f1` work branch before the revert commit).
