# grim-backend-tests

Cross-backend numerical parity test suite and Known-Answer Tests (KAT) validating execution consistency across CPU, ROCm, CUDA, Vulkan, and Metal backends in Grim.

[Back to project README](../../README.md)

## Dependencies

```mermaid
graph LR
    grim_backend_tests["grim-backend-tests"]
    grim_backend_cpu["grim-backend-cpu"]
    grim_backend_tests --> grim_backend_cpu
    grim_backend_cuda["grim-backend-cuda"]
    grim_backend_tests --> grim_backend_cuda
    grim_backend_metal["grim-backend-metal"]
    grim_backend_tests --> grim_backend_metal
    grim_backend_rocm["grim-backend-rocm"]
    grim_backend_tests --> grim_backend_rocm
    grim_backend_vulkan["grim-backend-vulkan"]
    grim_backend_tests --> grim_backend_vulkan
    grim_quant["grim-quant"]
    grim_backend_tests --> grim_quant
    grim_tensor["grim-tensor"]
    grim_backend_tests --> grim_tensor
    style grim_backend_tests stroke-width:3px
```

External dependencies: `rand`.

## Flamegraph

Not applicable - library crate (no binary).

## Source Layout

| Folder | Purpose |
|---|---|
| `src/` | Parity matrix test constants and shared test harness helpers |

## Key Responsibilities

- Multi-format numerical parity tests verifying identical results across accelerator backends and CPU reference math.
- Known-Answer Tests for quantized operations including Q8_0, Q4_K, Q5_K, Q6_K, and Iq4Nl across varying hidden dimensions.
- Cross-crate pipeline integration tests ensuring stable device dispatch and tensor lifetime management.
