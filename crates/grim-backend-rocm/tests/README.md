# ROCm backend tests — runner contract (P0 GPU-gate)

Two tiers. Default `cargo test` runs CPU-only tests; GPU tests are
`#[ignore]`d and never run by accident.

```sh
# Tier 1 — CPU only (CI default, no GPU needed):
cargo test -p grim-backend-rocm

# Tier 2 — GPU equivalence (needs ROCm GPU + HSACO):
GRIM_RUN_GPU_TESTS=1 GRIM_GPU_TEST=1 \
  cargo test -p grim-backend-rocm --no-fail-fast -- --ignored
```

Conventions every GPU test follows:

- `#[ignore]` + env-gate via `gpu_test_enabled()` (`GRIM_GPU_TEST=1`,
  legacy aliases `GRIM_RUN_GPU_TESTS=1`, `GRIM_RUN_GPU_TEST=1`). Both layers:
  `--ignored` selects the test, the env-gate skips gracefully without a GPU.
- `gpu_test_lock()` serializes tests driving the whole device (concurrent
  HIP graph capture segfaults).
- Long benches print heartbeat progress lines; watch with `--nocapture`.
  A frozen heartbeat = stalled launch, not a silent hang.
- Known waivers (not gates to weaken): `gpu_sampler_decode_throughput_gate`
  (device 0.17× vs CPU D2H on gfx1201 debug — real perf deficit, kernel work),
  `charon_wmma_vs_scalar_parity` (`hipModuleLoad 209` on some RDNA4 boxes —
  see file header).
