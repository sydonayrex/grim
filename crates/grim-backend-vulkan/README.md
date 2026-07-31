# grim-backend-vulkan

Vulkan compatibility backend for Grim.

## Purpose

Platform-agnostic compute backend:
- Vulkan compute for cross-platform GPU support
- Simulated JIT/autotuning for kernel selection
- Fallback for systems without ROCm/CUDA

## Boundaries

- Does not perform model architecture — only tensor operations
- May fall back to CPU for some operations
- No vendor-specific optimizations like ROCm's fused kernels

## Dependency Graph

```mermaid
graph LR
    A[grim-backend-vulkan] -->|DType, Device, Shape| B[grim-tensor]
    
    style A fill:#f3e5f5
```

## Public API

### VulkanDevice

```rust
pub struct VulkanDevice;

impl BackendDevice for VulkanDevice {
    // Vulkan compute pipeline implementations
}
```

## Feature Flags

This crate has no feature flags.

## Edge Cases

1. **CPU fallback**: Some operations run on CPU when GPU not available
2. **Autotuning**: Kernel parameters tuned at runtime for best performance
3. **Cross-platform**: Works on Windows/Linux/macOS with Vulkan driver