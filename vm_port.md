# vm_port.md — Vulkan + Metal Backend Feature Parity Plan

Port the missing `BackendDevice` features (gated to ROCm/CUDA today) onto the Vulkan and Metal backends so the quantized backward and comm/attention features work on all three GPU backends.

## 1. Gap Summary

Both Vulkan and Metal miss these (ROCm/CUDA implement them):

| Feature | Vulkan | Metal | Notes |
|---|---|---|---|
| `quantized_matmul_backward_dx` | shader exists, **not dispatched** | **no MSL kernel** | Trait default `backend.rs:752` errors. ROCm ref `roc_device.rs:2240`. |
| `all_reduce` | shader exists, **not dispatched** | **no MSL kernel** | Trait default `backend.rs:463`. |
| `comm_fuse_reduce` | shader exists, **not dispatched** | **no MSL kernel** | Trait default `backend.rs:487`. |
| `qkv_attention_paged` | shader exists, **not dispatched** | **no MSL kernel** | Trait default `backend.rs:398`. |
| `tree_attention` | **no shader** | **no MSL kernel** | Trait default `backend.rs:436`. |
| `kv_dequant_attention` | shader exists, **not dispatched** | **done** (`lib.rs:1212`) | Vulkan-only gap. ROCm ref `roc_device.rs:2006`. |
| `matmul_with_solution` | **not a real gap** | **not a real gap** | Trait default `backend.rs:130` delegates to `matmul`; ROCm doesn't override either. |

Not real gaps (generic defaults, ROCm also doesn't override): `silu_mul_backward`, `lora_accumulate`, `rms_norm_inplace`.

### Vulkan vs Metal asymmetry
- **Vulkan** compiles SPIR-V for *every* missing kernel already (`build.rs` + `VulkanKernel` enum `lib.rs:2364` + `spirv_for` `:2393`). The work is almost entirely **dispatch wiring** — the `impl BackendDevice for VulkanDevice` (`lib.rs:1320`) never calls these kernels.
- **Metal** has MSL kernels only for 15 forward ops; all 6 missing features need **new MSL kernels + new `MetalPipelines` fields + dispatch methods**.
- Vulkan additionally needs a **new Q8_0 backward shader** (the existing `quantized_matmul_backward_dx.comp` decodes Q4K 144-byte blocks, not Q8_0 34-byte blocks) and a **new `tree_attention.comp`** (no shader exists at all).

## 2. Data-Layout Contract (critical)

Q8_0 block layout (grim-quant `lib.rs:192-194`, ROCm kernel `iq_gemm.rs:706`):
- 34 bytes/block = **2-byte f16 scale + 32 × int8 codes**.
- `blocks_per_row = K/32`, `row_bytes = blocks_per_row * 34`.
- ROCm backward kernel: flattened grid over `M*K`, each thread `row*K+k_idx`, accumulates over all `N` columns: `dX[row*K+k_idx] = Σ_n dY[row*N+n] * dequant_q80(B_q80 + n*row_bytes + sb_idx*34, in_sb)`.

**Convention mismatch to resolve during implementation:**
- Vulkan forward `quantized_matmul` (`lib.rs:2199`) dispatches `FusedDequantGemmQ80` (16×16 tiles) using the **34-byte packed** format.
- Metal forward `quantized_matmul` (`lib.rs:1864`) dispatches `grim_quantized_matmul_q8_0` using **raw i8 bytes + a separate `f32` scale buffer** (`n*(k/32)` scales), NOT the packed 34-byte format.
- The new Metal/Vulkan backward kernels must read the **same byte layout their forward does**, otherwise grad-A is silently wrong. Recommend unifying both forward+backward on the packed 34-byte format (matches ROCm + grim-quant) and auditing the Metal forward path as part of the port.

## 3. Trait Signatures (from `crates/grim-tensor/src/backend.rs`)

- `quantized_matmul_backward_dx(&self, dy, b_packed, b_scales: &[f32], default_bpw: u8, m, n, k, out_shape, residuals: Option<&QuantizedMatmulBackwardResiduals>) -> (storage, handle)` — `:752`
- `all_reduce(&self, inputs: &[&dyn BackendStorage], op: &str) -> (storage, handle)` — `:463`
- `comm_fuse_reduce(&self, partials: &[(&dyn BackendStorage, &ScythePlacement)]) -> Box<dyn BackendStorage>` — `:487`
- `qkv_attention_paged(&self, q, block_tables, k_pages, v_pages, num_kv_heads, max_blocks, page_size, kv_seq_len, cache_offset, out_shape) -> (storage, handle)` — `:398`
- `tree_attention(&self, q, k, v, tree_parents, num_kv_heads, kv_seq_len, cache_offset, out_shape) -> (storage, handle)` — `:436`
- `kv_dequant_attention(&self, q, k_tensor, k_scales, v_tensor, v_scales, num_kv_heads, kv_seq_len, cache_offset, quant_bits, out_shape) -> (storage, handle)` — `:313`

`QuantizedMatmulBackwardResiduals` struct at `backend.rs:770` (outlier_count, outlier ptrs, backup1/2 bpw + offsets). `from_tensor` in autograd `ops.rs:490`.

## 4. Autograd Caller Gate

`crates/grim-autograd/src/ops.rs:470-471`:
```rust
let b_on_rocm = matches!(args.b.device(), grim_tensor::Device::Rocm(_));
let b_on_cuda = matches!(args.b.device(), grim_tensor::Device::Cuda(_));
```
Gate at `:488`: `if b_quantized && (b_on_rocm || b_on_cuda)`. Must add `b_on_vulkan` (`Device::Vulkan`) and `b_on_metal` (`Device::Metal(_)`) before GPU dispatch works for the new backends. `Device::Vulkan` is a unit variant (no ordinal, `dtype.rs:14`).

The caller already passes `Some(&residuals)`; ROCm's Q8_0 simple path ignores residuals — keep the same behavior (or honor residuals if the residualpacked shader is reused).

## 5. Implementation Phases

### Phase 1 — Vulkan dispatch wiring (shaders already compiled)
Add methods to `impl BackendDevice for VulkanDevice` (`crates/grim-backend-vulkan/src/lib.rs:1320`), mirroring the existing `mul_scalar`/`sqrt` dispatch pattern (`:1785`, `:1816`):
- Downcast inputs to `VulkanStorage` (pattern `:1791-1794`).
- Grab `GLOBAL_CONTEXT`, alloc output via `VulkanStorage::alloc_gpu` (`:719`).
- `spirv_for(VulkanKernel::X)` + `run_compute_shader(ctx, spirv, &buffers, gx, gy, gz, Some(push))` (`:974`).
- On failure → CPU fallback (pattern `quantized_matmul` `:2253`), not hard error.

Kernels/enum/spirv to wire:
1. `kv_dequant_attention` → `KvDequantAttention` (port logic from Metal `:1212` / ROCm `roc_device.rs:2006`).
2. `qkv_attention_paged` → `QkvAttentionPaged` (port from ROCm).
3. `all_reduce` → `AllReduce` (ROCm `all_reduce` impl; op string: sum/mean handling).
4. `comm_fuse_reduce` → `CommFuseReduce` (ROCm WI-6 impl; `ScythePlacement` partial fan-in).
5. `silu_mul_backward` → `SiluMulBackward` (optional — pure parity win, cheap: 3 buffers + push).

### Phase 2 — Vulkan new shaders (2 files)
1. **`kernels/quantized_matmul_backward_dx_q8_0.comp`** — Q8_0 34-byte-block backward mirroring ROCm `grim_fused_dequant_backward_gemm_q8_0` (`iq_gemm.rs:706`). Grid over `M*K` (use `gl_GlobalInvocationID` x=k_idx, y=row), per-thread N-loop. Add enum variant + `spirv_for` + `build.rs` entry.
   - Existing `quantized_matmul_backward_dx.comp` (Q4K 144-byte k4-scale-min layout) can be kept for the `k%256==0` / Q4K path, mirroring the forward's Q4K/Q80 branch (`lib.rs:2222-2226`).
2. **`kernels/tree_attention.comp`** — new. Port from ROCm's tree-attention kernel (parents-bounded attention; no GQA paging). Add enum variant + `spirv_for` + `build.rs`.

### Phase 3 — Metal MSL kernels + pipeline wiring
`crates/grim-backend-metal/src/lib.rs`:
1. Add MSL kernels to `kernels.msl`: `grim_quantized_matmul_backward_dx_q8_0`, `grim_all_reduce`, `grim_comm_fuse_reduce`, `grim_qkv_attention_paged`, `grim_tree_attention` (and optionally `grim_silu_mul_backward`).
2. Add fields to `MetalPipelines` (`:65-80`) + `get_pipeline` calls in context init (`:200`).
3. Implement methods in `impl BackendDevice for MetalDevice` (starts `:507`), mirroring forward `quantized_matmul` encoder pattern (`:1914-1956`): `setBuffer_offset_atIndex` × 4 + 3 scalar `setBytes` for m/n/k; 16×16 threadgroups.
4. `quantized_matmul_backward_dx`: grid `M*K` flattened like ROCm; read b as packed Q8_0 (34-byte) blocks — **and align the forward `grim_quantized_matmul_q8_0` kernel + its scales handling to the same packed format** (see §2). Keep CPU fallback.

### Phase 4 — autograd gate
`ops.rs:470-471`: add
```rust
let b_on_vulkan = matches!(args.b.device(), grim_tensor::Device::Vulkan);
let b_on_metal = matches!(args.b.device(), grim_tensor::Device::Metal(_));
```
and include in the gate at `:488`.

### Phase 5 — parity tests
Mirror the CUDA backward test (`test_cuda_quantized_matmul_backward_dx_q8_0`): `quant_q80` → `from_cpu_bytes` → `quantized_matmul_backward_dx` vs. host `dequant_q80` reference.
- Vulkan: `crates/grim-backend-vulkan/tests/vulkan_tests.rs` (tests use `#[test]`, CPU-simulated where no GPU — e.g. `test_vulkan_autotuner_and_spirv`, `test_vulkan_matmul_simulated`). Add a **SPIR-V-present + simulated** test for each new kernel so CI passes without a GPU.
- Metal: no test dir yet (`crates/grim-backend-metal/tests/` is empty); add `metal_tests.rs` with `#[cfg(target_vendor = "apple")]`-gated GPU tests + CPU-fallback parity tests.
- Add `all_reduce`/`comm_fuse_reduce`/`qkv_attention_paged`/`tree_attention` round-trip tests vs CPU reference.

## 6. Verification
- `cargo build -p grim-backend-vulkan` and `cargo test -p grim-backend-vulkan` (headless-safe: tests simulate when no GPU).
- `cargo build -p grim-backend-metal` (`cargo check` on non-Apple hosts; kernels.msl compiles via `mtlpp`/objc2 only under `target_vendor="apple"`).
- `cargo test -p grim-autograd` (after Phase 4) to prove the gate change doesn't regress CPU/ROCm paths.
- On GPU hardware (ROCm box / Mac), run parity tests vs `dequant_q80` reference.

## 7. Risks / Open Items
- **Layout mismatch** (§2): Metal's forward uses raw-bytes + separate scales; Vulkan/ROCm use packed 34-byte. Must pick one (recommend packed Q8_0 everywhere) and audit both forwards before writing backwards.
- Metal builds are apple-gated; cannot compile MSL on this Linux box — Phase 3 needs a Mac to verify, or careful `#[cfg(target_vendor="apple")]`-guarded code review.
- `tree_attention` has no ROCm-side doc anchor found in this scan — locate the ROCm kernel before porting.
- `comm_fuse_reduce` depends on `ScythePlacement` routing metadata; the Vulkan/Metal implementations can be a plain sum-of-partials reduction (correct for non-TP single-GPU use) and full routing later.
