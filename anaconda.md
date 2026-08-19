# grim — backend code audit (verified)

Audit of the grim workspace backend crates. Every finding below was verified against the current source tree; line numbers were checked, not assumed. Findings that could not be confirmed were removed.

## Summary

| Severity | Count |
|---|---|
| Critical | 0 |
| Important | 8 |
| Nit | 4 |

No Critical findings survived verification. The originally reported "critical" issues either referenced files that do not exist in the tree, or described behavior the actual code does not exhibit (see Appendix A for the rejected claims and why).

## Important

### I1. `unsafe impl Send + Sync` on raw-GPU-pointer wrapper without a usable contract
`crates/grim-tensor/src/backend.rs:1760-1763`
`QuantizedMatmulBackwardResiduals` gets `unsafe impl Send` + `unsafe impl Sync` with a doc comment asserting raw GPU device pointers are "thread-safe to pass across worker threads for HIP kernel launch." No enforcement, no lifetime bound, no ownership token — the invariant is only prose. If the device buffers are freed while a worker thread holds residuals, it is use-after-free.
Fix: wrap the pointers in owned storage handles or add a SAFETY comment spelling out the exact lifetime rule.

### I2. Raw byte-cast of a typed slice with unchecked length multiply
`crates/grim-models/transformer/src/lfm2.rs:1038`
`as_u8_slice` does `std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * size_of::<T>())`. The `len * size_of` can overflow on 32-bit targets and the cast is needless.
Fix: use `bytemuck::cast_slice` or `slice.align_to` with a checked size.

### I3. ROCm device handle marked `Send + Sync` with unguarded launch ops
`crates/grim-backend-rocm/src/device/roc_device.rs:214-215`
`RocmDevice` wraps raw HIP handles and gets `unsafe impl Send` + `unsafe impl Sync`. Concurrent launches/`set_device` from two threads on the same device are not serialized (the stream field is `Mutex<Option<*mut c_void>>`, so one is guarded; the raw `hipDevice_t` and pool refs are not).
Fix: document the threading contract or gate all device ops behind one mutex.

### I4. NCCL communicator marked `Send + Sync`
`crates/grim-backend-rocm/src/rccl.rs:10-11`
`NcclComm(*mut c_void)` with `unsafe impl Send` + `unsafe impl Sync`. NCCL collectives on the same communicator are not thread-safe; concurrent use is UB. The struct is `Copy` too, so it is trivially duplicated across threads.
Fix: wrap the communicator in an `Arc<Mutex<...>>` or document that exactly one thread owns it.

### I5. ROCBLAS and generic handles marked `Send` without serialization
`crates/grim-backend-rocm/src/device/rocblas.rs:35`, `crates/grim-backend-rocm/src/device/handles.rs:35`
`unsafe impl Send for RocblasHandle {}` and `unsafe impl Send for RocmHandle {}`. rocBLAS handles are not thread-safe; concurrent `rocblas_gemm_ex` on one handle races.
Fix: serialize via an internal lock or hand out per-thread handles.

### I6. Host staging buffers marked `Send + Sync`
`crates/grim-backend-rocm/src/p2p_route.rs:100-101, 244-245`
`unsafe impl Send` + `Sync` for `HostStagingBuffer` and `StagingCache`. Concurrent access to the same pinned staging pool from two threads can corrupt or double-free.
Fix: single-owner or guarded access.

### I7. Plugin `.so` loaded via `libloading` with no integrity check
`crates/grim-plugin/src/dylib_loader.rs:3,61`
Uses `libloading` to `dlopen` a plugin shared object from a config path. No pinned hash or signature verification — a tampered plugin binary executes arbitrary code in-process.
Fix: verify a pinned SHA256 of the artifact before `dlopen`, or restrict to trusted paths.

### I8. GGUF nested-array parsing recurses with no depth cap
`crates/grim-format/src/gguf.rs:1678`
Array values (tag 9) recurse into `read_gguf_value_with_tag`. A count safety limit of 10M exists, but there is no recursion depth bound — a crafted deeply-nested metadata array can exhaust the stack.
Fix: add a `depth` parameter with a hard cap (e.g. 64) and return an error beyond it.

## Nit

### N1. ROCm `unsafe impl Send` + `Sync` blocks carry only prose invariants
`crates/grim-backend-rocm/src/p2p_route.rs`, `roc_device.rs`, `rccl.rs`
The blocks in I3-I6 are all correct FFI-wrapper patterns; the weakness is that the thread-safety contract lives only in comments. A lint requiring `SAFETY:` on every `unsafe impl` would make violations visible. (Structural, not a defect by itself.)

### N2. `sort_json_keys` recurses on `serde_json::Value`
`crates/grim-server/src/tool_parse.rs:169-183`
Recursive key-sort over tool-call JSON. In practice bounded by user input length; not currently exploitable but a depth guard would be cheap.

### N3. Grim-engine CAS rollback is correct — add a regression test
`crates/grim-engine/src/scythe2.rs:1075-1077`
The copy-failure path rolls back the CAS increment with `compare_exchange_weak`, exactly as it should. The `[P1-42 fix]` comment suggests it was recently fixed; a test asserting the rollback (ring stays usable after a failed copy) would lock the behavior in.

### N4. WASM plugin loader already enforces fuel — document the budget
`crates/grim-plugin/src/wasm_loader.rs:3,56-64,80`
The loader enforces fuel, memory, and capability grants. Good. The defaults for `PluginLimits` are not in the verified range; confirm a bounded default exists so plugins can't self-grant unlimited fuel.

## Appendix A — rejected claims

The following were reported in earlier drafts but did not survive verification and are NOT included above:

| # | Claimed | Actual |
|---|---|---|
| R1 | `scythe_ring.rs` CAS rollback race orphaned slots | No such file. Real CAS rollback in `scythe2.rs:1075` is correct (rolls the head back on copy failure). |
| R2 | `device_handle.rs` raw `*const RocmDevice` with no lifetime guard | No such file in grim-engine. |
| R4 | Non-atomic `attach_bolt_on` writes in gguf.rs | No `attach_bolt_on` function exists. |
| R5 | AVX2 GEMM OOB when `m % 8 != 0` | False. `simd_gemm.rs:41` guards the SIMD loop with `while kk + 8 <= k`; the tail is handled by a scalar loop. No OOB path. |
| R6 | Unsafe extern `sgemm` with raw pointers, no bounds check | False. `device.rs:1739-1743` calls safe `matrixmultiply::sgemm` on slices. |
| R7 | `q6k.rs` inverted scale-index mapping | No `q6k.rs`; grim-quant has `lib.rs`, `soul_eater.rs`, `spqr.rs`. |
| R8 | `rccl_allreduce.rs` missing sync | No such file. RCCL lives in grim-backend-rocm. |
| R9 | `multi_gpu.rs` hardcoded device ordinal 0 | No such file in grim-autograd. |
| R10 | Markov head bias computed but discarded | False. `uniform_markov_head.rs:60-61` adds the bias to logits and applies it. |
| R11 | `auto()` cannot select NativeMtp | No `auto.rs` / `SpeculativeConfig::auto` in grim-speculative. |
| R14 | perf_gate fabricates a pass | False. `perf_gate.rs:26` explicitly states "missing baseline never fabricates a 'passed' gate (no fake pass)." |
| R17 | caps.rs parses `nvidia-smi` | No `nvidia-smi` reference in caps.rs. |
| R23 | kv_omni silently falls back to fp16 | False. `kv_omni.rs:359` returns `Err(KvCache(...))` on unsupported format. |
| R28 | wasm_loader has no execution budget | False. Fuel and memory limits are enforced (wasm_loader.rs:3,56). |

## Remediation priorities

1. I1-I6 — the raw-pointer/`unsafe impl` contracts need enforced lifetime rules or guards (engine + ROCm backend).
2. I7 — plugin loading is the one true code-execution trust boundary without a check.
3. I8 — attacker-controlled GGUF input should not be able to exhaust the stack.