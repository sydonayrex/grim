Prompt: review the codebase for correctness, logic bugs, numerical pathway correctness and accuracy, inter-crate and intra-crate pub and import correctness, ffi accuracy, rocm kernel correctness, and issues that negatively impact the speed of model loading and model response for inference and training - to include single gpu vs multi-gpu parallelism.


Audit Results
1. ROCm selective_scan.rs — CRITICAL (multiple)
The launch and the kernel describe two different contracts and were never reconciled.
- Arg count + order mismatch (ABI corruption). roc_device.rs:11050-11060 passes 10 args: x, a, b, c, d, out, batch, dim_dstate, dim_dinner, seq_len. The kernel at selective_scan.rs:27-36 declares 9 params in a different order: a_log, b_tensor, d_tensor, h_in_out, x_tensor, y_data, batch, d_inner, d_state.
Consequence of the positional binding:
- launch c_ptr (arg 4) → kernel h_in_out (arg 4) — c_tensor is being used as the SSM state buffer, not a C output.
- launch d_ptr/out_ptr → kernel x_tensor/y_data — input x receives the D-bias pointer; c (a real output you care about) is dropped.
- launch dim_dstate (arg 8) → kernel d_inner; launch dim_dinner (arg 9) → kernel d_state — d_inner and d_state are swapped, plus an extra unused seq_len.
Every pointer and both dimension scalars land on the wrong register. Output is undefined garbage. Fix: make the launch arg array and the kernel signature one and the same (same order, same count).
- Missing C-matmul. Confirmed — the kernel (selective_scan.rs:56-73) does y = sum_s(h_new) + D·x and never multiplies by C. Correct Mamba is y = (h_new·C) + D·x. The launch does pass a c storage, so the author intended C, but the kernel never consumes it. Severity: high (wrong output even if the ABI is fixed).
- Dynamic shared memory never allocated. Kernel uses extern __shared__ float lds_h[] (selective_scan.rs:42,43) but launch_selective_scan routes through launch_compute_kernel → ..._with_solution(..., 0) (roc_device.rs:9113) = 0 shared bytes. my_h becomes a null pointer. On RDNA (RDNA has no CUDA-style per-block shared mem; __syncthreads/extern __shared__ are CUDA idioms) this whole kernel is suspect. Severity: critical — will crash or read null on the launch path.

2. Flash Attention — cross_attention.rs
- Online softmax stability: OK. Max is computed (Pass 2a, :64-77) before exp(scores - max) (Pass 2b, :83). Stable. ✓
- Causal mask: N/A. This is explicitly non-causal cross-attention (Whisper decoder) per the :17-18 doc; there is no lower-triangle mask, which is correct for this use. There is no causal self-attention kernel in scope here, so nothing to flag.
- GQA indexing: convention risk (LOW/MED). kv_head = head % num_heads_k (:40) gives interleaved assignment (kv0→{0,2,4}, kv1→{1,3,5}). The comment (:14-15) and standard GQA use contiguous groups head / (num_heads/num_heads_k). This is only correct if the Q-projection layout was built with the same interleaved convention; otherwise each query head reads the wrong KV head. Fix: use head / (num_heads/num_heads_k) or document the interleaved contract.

3. Decode GEMM — decode_gemm.rs
- Tiling: none exist. There are no BLOCK_M/N/K constants — it's a one-thread-per-output-element kernel (:21-35), no shared memory. So "powers of 2" is N/A; it is simply not a tiled GEMM. Not a bug, but the doc/expectation of a tiled kernel isn't met.
- FMA: implicit only. acc += a_val * b_val (:32) — relies on the compiler contracting to vfmadds. No explicit fma/__fmaf. Acceptable but not guaranteed without #pragma / fast-math.
- Coalescing: OK. Output written row-major C[row*stride_c + col] (:35); row=idx/N, col=idx%N means consecutive threads write consecutive col within a row → coalesced. A/B reads along the fixed row are likewise adjacent. ✓

4. Autograd
- adamw.rs AdamW — CORRECT. step = m_hat/denom + weight_decay*w then w = w - lr*step (:533-541) = decoupled decay on the parameter, not the gradient. That is proper AdamW. ✓ (Despite the variable being named step_grad.)
- Side finding: Lion8Bit::step (:1224-1231) computes updated = -lr*τ + wd*w, dropping the base w and adding wd*w instead of subtracting — a genuine update-rule bug (compare the correct Lion at :805-808).
- loss.rs cross-entropy — CORRECT & stable. Max-subtraction before exp (:251-258), gradient (softmax - one_hot)/batch (:268-270). Matches d_loss/d_logits = softmax - 1{target}. ✓
- charon.rs forward / charon_backward.rs — CORRECT. The backward decomposition (charon_backward.rs:139-178) recomputes h_gate/h_up/act and derives d_down_w, d_act, d_h_gate = d_act·silu′(hg)·h_up, d_h_up = d_act·silu(hg), d_gate_w, d_up_w, d_x. I verified each against the chain rule for y = down( silu(gate(x)) ⊙ up(x) ) scaled by rsf — all consistent. silu_grad (:53-56) = s·(1+z(1−s)) is the correct derivative. ✓
- Minor doc bug: charon_atomic_add2 (charon.rs:61-64) is documented as a packed 64-bit atomic add but is just two scalar atomicAdds. Functionally equivalent; comment is misleading.

5. KV Quantization — kv_omni.rs
- Format: modality-dependent — Text: Lloyd-Max (K8V4 shallow / K4V2 deep, :50-73); Audio: rotate + 2-bit (K=V=2, :74-84); Visual: low-rank "Tucker" projection (rank 16) + INT8 keys.
- Lossy or lossless: lossy (INT2/4/8 quant + low-rank projection). No path is lossless.
- K vs V sensitivity: K is given ≥ V bits everywhere (K8≥V4, K4≥V2, K2=V2) — i.e. K is quantized at equal or higher precision, which is the correct direction (K is more sensitive). ✓
- Dequant-on-read: Yes. dequantize_with_modality fully reconstructs K and V to f32, then standard attention runs in dequant (f32) space (fused_attention, :502-561). It does not compute attention in quantized space — a performance choice, but correct.
- Note: The Visual path claims "values stored as raw f32 (Tucker-16)" but V is actually low-rank-projected then reconstructed as V_proj @ Rᵀ (:427-441), which is lossy whenever rank < head_dim (the roundtrip test only passes because it uses head_dim=16 so rank==head_dim → identity). Not a bug given the ranking design, but the "raw/exact V" framing (e.g. :985-989) is inaccurate.

6. Speculative Decoding
- Accept/reject — CORRECT. p_accept = min(1, p_target/p_draft) (speculative_wrapper.rs:244-248, and the native-MTP path :366-367), break on first rejection (contiguous-prefix accept), then kv.commit(accepted_count) (:258) which truncates the tentative append. ✓ (Simpler than sampling the bonus token from the target residual — acceptable variant.)
- State rollback — logic CORRECT, but doesn't restore. MambaSpeculativeEngine::record_state/rollback_to (mamba_speculative.rs:34-57) correctly pops history newer than target_step and returns the last recorded ssm_state/conv_state. However it only returns the snapshot; nothing in this crate writes it back into the live SSM/conv state — the caller must apply it. There is no RwkvState variant at all (Mamba-only). Flag as an integration gap: rollback must be wired to the actual model state or it has no effect.

Summary by severity
#	Issue	Severity
1	selective_scan launch↔kernel ABI mismatch (10 vs 9 args, reordered, d_inner/d_state swapped, c misbound as state)	Critical
1	selective_scan missing C matmul	High
1	selective_scan uses extern __shared__ but launched with 0 shared bytes (null LDS; CUDA idiom on RDNA)	Critical
4	Lion8Bit update rule drops base w, adds wd*w	High
2	GQA head % num_heads_k (interleaved) vs standard contiguous grouping	Low/Med
3	No actual tiling; FMA only via compiler contraction	Info
6	Mamba rollback returns state but never restores it into the model	Med (integration)
5	Visual "raw f32 V" is actually low-rank (lossy when rank<head_dim)	Info
—	adamw, loss, charon fwd/bwd, speculative accept: verified correct	—


FFI / memory / sync audit

1. memory/pinned.rs
#	file:line	finding
P1	pinned.rs:92-99	OK — not a bug. Drop correctly calls hipHostFree.
P2	pinned.rs:18	OK — not a bug. unsafe impl<T: Send> Send for RocmPinnedBuffer<T> is the correct pattern for a raw-pointer wrapper whose lifetime is owned (one *mut T + PhantomData).
P3	pinned.rs:20-24	OK — not a bug. impl<T> Debug is present, so RocmDevice's #[derive(Debug)] satisfies the bound.
pinned.rs is internally consistent. All three of the user's suspicions clear.

2. memory/pool.rs + memory/
#	file:line	finding	severity
POOL-1	pool.rs:54-60 + 163-167	OK — no double free. PooledBuffer::drop returns the pointer to the pool's bucket (does not free); DeviceScratchPool::drop → drain() calls hipFree exactly once per cached pointer.	—
POOL-2	pool.rs:128-134	Bug — silent memory leak on mutex poisoning. return_buffer swallows Err and drops the pointer; the buffer is neither held in the bucket nor freed → VRAM leak.	High (silent leak)
POOL-3	pool.rs:33-37	Missing Send. PooledBuffer contains *mut std::ffi::c_void + Arc<DeviceScratchPool>. It is !Send/!Sync. If a PooledBuffer is ever stored across thread boundaries (e.g. as part of a kernel argument vector returned to another worker thread), the borrow-checker will reject it — a latent issue the call sites likely work around with raw-pointer copies.	Medium
POOL-4	pool.rs:145-160	Order-of-operations bug in drain(). hipDeviceSynchronize() is not called before hipFree. A pooled buffer is only returned to the bucket on PooledBuffer::drop (host thread), but the device-stream work that used it may still be in flight (the stream is not synchronized between kernel-completion and the host-side drop). The Drop for DeviceScratchPool path in RocmDevice::drop does call hipDeviceSynchronize first (roc_device.rs:1373) but only for the pool's device — see POOL-5 below.	High
POOL-5	allocator.rs:88-96 (free) + roc_device.rs:1369-1406 (Drop for RocmDevice)	Bug — RocmDevice::drop frees allocator memory but not the scratch pool, and the two are different pools. self.allocator.empty_cache() frees the allocator's pool field (allocator.rs), but scratch_pool is an independent Arc<DeviceScratchPool>. If no other Arc copy is alive, the scratch pool is dropped (and drained) as a side-effect of Arc::drop — which is covered by DeviceScratchPool::drop, but only after allocator.empty_cache() runs. Ordering: empty_cache calls hipDeviceSynchronize() (allocator.rs:111), then the next Arc::drop fires DeviceScratchPool::drop → drain without its own sync. In the common case this is OK, but if any PooledBuffer is still held at device-drop time (i.e. leaked), drain never touches it — and if the pool is dropped from a different thread than the one that ran the last kernel, hipDeviceSynchronize may not have waited for all work. Low probability; medium impact.	Medium
POOL-6	allocator.rs:16-18	Bug — RocmCachingAllocator is not Send/Sync. pool: Mutex<HashMap<usize, Vec<u64>>> is Send+Sync (Mutex is), so the struct itself is — but the Arc<RocmCachingAllocator> in RocmStorage is used across threads (via Box<dyn BackendStorage> returned to engine). RocmDevice itself has unsafe impl Send + Sync (roc_device.rs:186-187) which masks this, but if someone constructs a fresh RocmCachingAllocator outside a RocmDevice, it is !Send. Verify no such construction path exists.	Low
POOL-7	allocator.rs:43-49	Bug — size_class is not Eq/unique across allocations of different bytes that round to the same class. alloc(15) → cls=16; alloc(16) → cls=16; both draw from the same bucket, but the caller expects ≥15 bytes and is served 16 (OK). However alloc(17) → cls=32, while a free(17-byte buffer) returns to cls=32. So a buffer allocated at 32 bytes can later be served as 17 bytes — no bug. However, a buffer allocated at 16 bytes and freed with bytes=16 (cls=16) is correct. The asymmetry is: a buffer allocated by alloc_gpu_with_bytes(shape, ..., 20, ...) gets cls=32, but RocmStorage::drop calls allocator.free(ptr, self.bytes=20) → cls=32. OK, symmetric. No actual bug.	—

3. roc_device.rs — upload_from_host_stream_ordered & synchronize
#	file:line	finding	severity
UP-1	roc_device.rs:1169-1205	Race — TOCTOU between hipMemcpyAsync/pins.push and synchronize()'s hipDeviceSynchronize+clear. The contract is "pin alive ≥ copy in flight", but on two threads: Thread A enqueues hipMemcpyAsync with pinned P (copy in flight), acquires the mutex, pushes P, releases mutex. Thread B's hipDeviceSynchronize() had already returned before A pushed P — it only waited for work up to its own call. B acquires the mutex next and pins.clear() → P is hipHostFree'd while its hipMemcpyAsync is still in flight. The (enqueue, push) pair is not atomic w.r.t. (sync, clear).	Critical (heap use-after-free of a page-locked host buffer)
UP-2	roc_device.rs:740-749	Design flaw — synchronize() drains retained_pins on a global basis, but hipDeviceSynchronize is per-device. If another RocmDevice (different ordinal) is being used on another thread, hipDeviceSynchronize only syncs the calling thread's current device. A retained_pins entry whose copy ran on the other device is freed before that device's copy completes, because the other device's synchronize() is what actually waits.	Critical (cross-device UAF)
UP-3	roc_device.rs:1137-1155 (copy_from_host_async)	Bug — the same pinned buffer is freed by RAII immediately after hipMemcpyAsync returns, even though the copy is async. pinned is a local RocmPinnedBuffer; it is dropped at the end of the function (line 1156, just before return). The copy engine reads pinned.as_ptr() after the CPU has returned → host UAF on every decode step. Compare to upload_from_host_stream_ordered (line 1201-1203) which correctly retains the pin — this path does not.	Critical (use-after-free)

4. rocblas.rs / roc_device.rs — handle caching & stream binding
#	file:line	finding	severity
RB-1	rocblas.rs:25-31	Design smell. RoclabsHandle is #[derive(Clone, Copy)] + unsafe impl Send + Sync. This is only sound if rocBLAS handles are truly thread-safe AND if the per-handle mutable state (bound stream) is protected externally. In this codebase, the bound stream is set by rocblas_set_stream at multiple call sites without locking — see RB-2.	High (design)
RB-2	roc_device.rs:1775-1856 (matmul_with_solution) and 9526-9604 (matmul_op)	Bug — rocblas_set_stream is NOT called before rocblas_gemm_ex/rocblas_sgemm on the main matmul path. Compare to matmul_batched (line 1002-1006) which does call it. Consequence: the GEMM runs on the handle's last-set stream (initially the default stream from creation time), while the returned RocmHandle (line 1856, 9615) points at self.active_stream() (a pooled stream). Caller calls synchronize() on the pooled stream → returns immediately even though the GEMM is still running on the default stream. Silent correctness bug under any overlapping workload.	Critical (wrong-stream GEMM + premature sync return)
RB-3	roc_device.rs:251-259 + 1472-1532	Bug — lazy creation of the rocBLAS handle can capture the wrong device. try_new calls hipSetDevice(ordinal) then rocblas_create_handle on the same thread — OK. But get_rocblas_handle (line 1472) can be called on any thread (e.g. an engine worker) that has a different "current device" set on that thread. The handle is cached in RocmDevice.handle_cache — if the first caller was on the wrong device, every subsequent GEMM on this device runs on the wrong GPU.	High (cross-device)
RB-4	roc_device.rs:1775, 9526, 9427	Missing stream binding is also a synchronization issue for the split-K path. launch_split_k_reduction (line 9064) runs on self.active_stream(). The preceding rocblas_gemm_strided_batched_ex (line 9443) ran on the handle's current stream (no set_stream call between them). If they are different streams, the reduction kernel may start before the GEMM partials are written.	Critical (data race)

5. handles.rs
#	file:line	finding
HD-1	handles.rs:23-49	OK. RocmHandle wraps Option<*mut c_void>, has unsafe impl Send (line 35). synchronize() uses hipStreamSynchronize (the correct primitive for a known stream). is_ready is a no-op returning true — documented behavior.
HD-2	rocblas.rs:30-31 vs handles.rs:35	Inconsistency. RoclabsHandle has unsafe impl Send + Sync; RocmHandle has only Send. The two wrappers have different thread-safety contracts for the same kind of FFI object. Sync on RoclabsHandle (see RB-1/RB-2) is unsound.
HD-3	handles.rs:109-113	OK — hipFreeAsync is declared with the correct signature. Used in allocator.rs:88-96.
HD-4	handles.rs:129-132	hipDeviceSynchronize and hipGetLastError are declared but no code in the crate calls hipGetLastError() after an FFI error. Errors are only inspected at the immediate call site. A kernel launched async that faults later will not be detected until the next HIP call on that thread returns hipErrorLaunchFailure. This masks device faults.

6. accel_ffi.rs
#	file:line	finding
AF-1	accel_ffi.rs:36-65	OK. MiopenLib::probe creates then immediately destroys a MIOpen handle. No UAF.
AF-2	accel_ffi.rs:73	OK. RCCL was moved to rccl.rs.
AF-3	accel_ffi.rs:21-33	No Drop on MiopenLib. Library is owned by the struct and has its own Drop (via libloading), so the dlopen'd handle is released when MiopenLib is dropped. OK.

7. multi_gpu_launch.rs & rccl.rs
#	file:line	finding	severity
MG-1	multi_gpu_launch.rs:42-69	Partial OK. Per-device stream isolation is preserved inside each RocmDevice (its stream_pool is per-RocmDevice), but the launch loop is sequential, not parallel. All N sharded kernels are submitted one after another on the calling thread. True parallelism requires the N kernel launches to be on different HIP contexts/streams — which requires hipSetDevice(ordinal) between launches (not shown). As written, the N launches all execute on the single HIP context the calling thread last set, and only the one whose ordinal matches that context actually runs on a GPU.	Critical (silent no-op for N-1 ranks)
MG-2	multi_gpu_launch.rs:71-80	**Bug — rccl.sum_gradients_device(out_ptr, out_ptr, count, 0) uses the pointer of the last-arg of the kernel launch, which is a single device pointer valid on one specific rank. RCCL's ncclAllReduce is expected to be called with each rank's local pointer to its own gradient shard — but here all ranks pass the same value. Additionally, stream=0 (default stream) is used, and no hipStreamSynchronize/hipDeviceSynchronize is called before the function returns. The caller may then read the result while the all-reduce is still in flight, OR the function may return before NCCL has started the transfer.	Critical
MG-3	multi_gpu_launch.rs:73-76	Bug — count is computed as full_dims.m * full_dims.n, but the kernel shards on the M dimension (shard_m at line 45), so each rank's output buffer is shard_m * n, not full_dims.m * n. NCCL will transfer count elements, reading/writing past the end of the actual shard. Heap/GPU-memory corruption.	Critical
RC-1	rccl.rs:557-575 (init_comm)	Bug — ncclCommInitAll returns an array of communicators (one per rank), but the caller uses self.comms.first() (line 606-610) for all collectives. For a multi-rank training loop where this single RcclAllReduce is shared, every rank will use rank-0's communicator. For in-process multi-GPU, this is at best wrong and at best a no-op. For multi-process training, each process should have its own RcclAllReduce constructed per-rank (via ncclCommInitRank), which is the correct pattern — but the code does not appear to do this.	Critical (collective deadlock / wrong data)
RC-2	rccl.rs:577-582	Bug — init_comm under not(feature="rccl") returns Err, but try_new (line 545-555) only calls init_comm when num_gpus > 1 and the feature is enabled. When num_gpus <= 1 it returns Ok(RcclAllReduce { num_gpus, comms: Vec::new() }). The two paths are consistent — sum_gradients_device (line 601) is a no-op for num_gpus <= 1. OK, just awkwardly split.	—
RC-3	rccl.rs:437-447 (Drop for RocmComm) + 667-679 (Drop for RcclAllReduce)	OK — both Drop impls call ncclCommDestroy exactly once per comm. No leak.	—
RC-4	rccl.rs:606-610	Bug — RcclAllReduce is Mutex<Option<Arc<RcclAllReduce>>> (roc_device.rs:183). RcclAllReduce has #[derive(Debug)] (rccl.rs:527), but it contains Vec<NcclComm> where NcclComm is #[repr(transparent)] newtype over *mut c_void. NcclComm has a #[derive(Debug)] — which is fine (just prints the pointer). However, RcclAllReduce is not Send/Sync despite being stored behind Mutex<Option<Arc<...>>> inside RocmDevice which is unsafe impl Send + Sync. The Arc<RcclAllReduce> is Send (since NcclComm: Send + Sync is implemented at line 10-11 of rccl.rs), so this compiles. OK.	—
RC-5	rccl.rs:186-213 and 215-252	Missing ncclCommRegisterBuffer / no buffer pinning. NCCL requires that the send/recv buffers be registered (ncclCommRegisterBuffer) once before use in async mode, or the transfer may not start on the correct stream. The code passes raw pointers without registration. On ROCm this is typically OK for device buffers, but it is a portability risk.	Medium

Cross-cutting: rocBLAS handle creation order vs hipSetDevice
#	file:line	finding
X-1	roc_device.rs:240-296 (try_new)	Bug — hipSetDevice is called on the current thread (line 242), but RocmHandle::synchronize is later called from potentially different threads (handles.rs:40). hipSetDevice is thread-local; a RocmHandle created for ordinal 0 will only affect the thread that called set. This is correct HIP behavior, but it means any multi-threaded code that assumes a RocmDevice is "pinned" to one GPU from a global perspective is wrong. The codebase relies on unsafe impl Send + Sync (line 186-187) to move the device between threads — this is sound only because HIP operations are context-switching based, not device-pinned in the Rust sense. Callers must remember to call their own hipSetDevice before any hip FFI call on a new thread. No such call exists in RocmHandle::synchronize (handles.rs:38-48) — it will synchronize the calling thread's current device, which may be a different device from self.ordinal.*

Summary of severity

Critical (will corrupt memory or produce wrong numerics):
1. UP-1 — retained_pins drain race (roc_device.rs:740-749 + 1169-1205)
2. UP-2 — synchronize() is per-device but retained_pins is shared across devices
3. UP-3 — copy_from_host_async (roc_device.rs:1123-1156) drops pinned buffer immediately after async copy
4. RB-2 / RB-4 — main GEMM path never calls rocblas_set_stream → GEMM on wrong stream + RocmHandle::synchronize no-ops
5. MG-1 — multi-GPU loop is sequential with no hipSetDevice between ranks
6. MG-2/MG-3 — NCCL all-reduce uses wrong pointer, wrong count, no sync
7. RC-1 — only comms.first() is used; per-rank comm is ignored
8. X-1 — RocmHandle::synchronize does not call hipSetDevice(self.ordinal) first

High (silent leaks or data loss):
- POOL-2 (mutex-poison leak)
- RB-3 (rocBLAS handle may capture wrong device)
- HD-4 (no hipGetLastError recovery)

Low / OK: pinned.rs, pool.rs Drop/drain, allocator.rs Send/Sync (masked by RocmDevice), handles.rs Send impl, rccl.rs Drop impls, accel_ffi.rs.

The single highest-priority fix is UP-3: copy_from_host_async releases the pinned buffer before the async H2D copy completes — this is a use-after-free on the hot decode path and likely the source of any observed corruption.

