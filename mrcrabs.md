# mrcrabs.md — Per-Crate Correctness / Security / Memory Review

**Repo:** `/D/rex/projects/grim` (Rust workspace, edition 2024, `warnings = "deny"`)
**Review type:** static (no GPU available; no `cargo build`/`test` run — user blocked the one
network-touching compile probe). Results are from source inspection + grep-driven triage.
**Skills applied:** `rust-expert`, `rust-ffi`, `rocm/rocm-hip`, `kernel-review`,
`mlops/grim-rocm-ffi`, `ml-ai/grim-moe-quant-kernels`.
**Requested-but-absent skills** (not installed in this env, substituted above):
`caveman`, `ponytail-audit`, `ml-llm`.

> Severity legend: **CRIT** = wrong results / silent data corruption / remote DoS.
> **HIGH** = wrong results on common paths, local DoS, or unauthenticated resource abuse.
> **MED** = edge-case wrong answer, misleading output, or robustness gap. **LOW** = hygiene.

---

## 0. Cross-cutting (whole-workspace) patterns

- **Integer-overflow on attacker/header-controlled sizes** is the single most repeated class:
  `num_tensors * sizeof`, `dims.product() * type_size`, `num_elements * 8`, `max_tokens + prompt_len`,
  `hidden*inter*hidden*4` appear in `grim-format`, `grim-quant`, `grim-backend-cuda/-vulkan`,
  `grim-kvtransport`, `grim-server`, `grim-backend-rocm` (charon_backward). Use `checked_mul` everywhere.
- **Unchecked `unwrap()`/`expect()` in request/training handlers** — `grim-server` (135) and
  `grim-autograd` (851) are the worst; many are poison-mutex unwraps that take the server down
  (High, see S-sec). Convert to `?`/recover-from-poison (`into_inner()`).
- **FFI hygiene is mostly good**: no dangling-`CString` temporaries found; status codes checked on
  HIP/Metal/Vulkan/CUDA. **Missing `catch_unwind` on any FFI boundary** (ROCm crate) — Low but real.
- **GGUF/IQ/BF16 dtype table is wrong (CRIT)** — see FMT-1; this silently corrupts *every* IQ/BF16 model.

---

## 1. grim-format  (LOC 12,124)  — review status: **DONE, multiple CRIT**

| ID | Sev | Location | Issue | Fix |
|----|-----|----------|-------|-----|
| FMT-1 | CRIT | `gguf.rs:134-166` | `GgufDType::from_tag` IQ/BF16 tag map is shifted vs canonical ggml_type (IQ2_XXS=23 here vs 22 in ggml; BF16/MXFP4 absent). Every IQ-quant + BF16 GGUF mis-tagged → wrong `size_bytes`, cascading offset drift, garbage weights. | Replace table with exact ggml enum; add BF16=30/IQ1_M=29/MXFP4=39. |
| FMT-2 | CRIT | `gguf.rs:247-261` | Block byte-sizes wrong for Q3K(108≠110), IQ3_XXS(96≠98), IQ4_NL(170≠18×32/256), Q8K(252≠292), Q8_1(40≠36). Undershoots `size_bytes`. | Derive from ggml structs; add a unit test. |
| FMT-3 | HIGH | `gguf.rs:1415,1417` | `dims.iter().product()` + `*type_size` unchecked; `saturating_mul` only on block path. Wraps → wrong/zero size. | `checked_mul`, error on overflow. |
| FMT-4 | HIGH | `gguf.rs:1462-1471` | `read_tensor_bytes` allocates `size_bytes` from metadata; `start+size` never validated vs file len. Unbounded alloc DoS / wrong-offset reads. | Validate `start+size <= file_len`. |
| FMT-5 | MED | `gguf.rs:1368` | `version != 3` hard-errors; rejects v1/v2 GGUFs. | Accept 2/3, branch count width. |
| FMT-6 | LOW | `gguf.rs:1100` | `Number::from_f64(bpw).unwrap()` panics on NaN/Inf. | `unwrap_or(0.into())`. |
| FMT-7 | CRIT | `safetensors.rs:106-107` | `data_offsets[0]/[1]` indexed w/o length check → panic on `[]`. | `.get(0)/.get(1).ok_or(...)`. |
| FMT-8 | HIGH | `safetensors.rs:136` (`gptq.rs:191,196,260`) | `data_end - data_start` unchecked; `data_end<data_start` → ~2^64 alloc in release. | Validate `data_start<=data_end<=file_len`. |
| FMT-9 | HIGH | `safetensors.rs:52-54` | `vec![0u8; header_len]` from 8-byte prefix → 16-EB DoS. | Cap header_len (e.g. 100 MB). |
| FMT-10 | MED | `safetensors.rs:35` | Unknown dtype silently → 4 bytes. | `Result`, error on unknown. |
| FMT-11 | LOW | `safetensors.rs:98` | `as_u64().unwrap_or(0)` accepts zero dims. | Reject 0. |
| FMT-12 | HIGH | `format.rs:1172` | `.grim` `metadata_len` unbounded (num_tensors capped, not meta). Alloc DoS. | Bound vs file len. |
| FMT-13 | MED | `format.rs:453` | `outlier_count*OUTLIER_RECORD_BYTES` unchecked (overflows on 32-bit). | `checked_mul` + file-len check. |
| FMT-14 | MED | `format.rs:485-486` | DeltaVarint `read_to_end` reads whole file. | Bound by next tensor offset. |
| FMT-15 | HIGH | `format.rs:1285` vs `:314` | `write` always emits KV block, `read` only consumes it when mutable JSON flag survives → 45-byte/entry desync. | Make KV block unconditional / version-gated. |
| FMT-16 | MED | `gptq.rs:100-101` | Reads `config.json` next to model w/o canonicalization (traversal w/ catalog). | Canonicalize, confine to models dir. |
| FMT-17 | MED | `gptq.rs:183` | `32/bits` with `bits>32` → 0 → 1-col tensor; `bits==0` div-by-zero later. | Validate `bits ∈ {2,3,4,8}` at parse. |
| FMT-18 | LOW | `gptq.rs:320-324` | Needless `from_raw_parts` on f32 vec (sound but unsafe). | `bytemuck::cast_slice`. |
| FMT-19 | LOW | `gptq.rs:291,345` | `self.reader.lock().unwrap()` poison panic. | `into_inner()` recovery. |

**Clean modules:** `spec.rs::decode_varint` (shift-overflow guarded).

---

## 2. grim-quant  (LOC 6,948)  — review status: **DONE, multiple CRIT (dequant correctness)**

| ID | Sev | Location | Issue | Fix |
|----|-----|----------|-------|-----|
| QNT-1 | CRIT | `lib.rs:809,830-831` | `dequant_q2k` BLOCK_BYTES=82 (real 84); `d` and `dmin` read the **same** two bytes (`data[pos+80..82]` both). `dmin==d` always + 2-byte drift/block. **Confirmed by reading source.** | BLOCK_BYTES=84; `d` at +80..82, `dmin` at +82..84. |
| QNT-2 | HIGH | `lib.rs:840-847` | `dequant_q2k` scale index permutation ≠ ggml `is++` walk → wrong sub-block attribution. | Mirror ggml exactly. |
| QNT-3 | CRIT | `lib.rs:241-287` | `dequant_iq4nl` fabricated 170-byte layout (sign plane + 2-bit group-scale invented; `IQ4_NL_CODEBOOK` all-positive vs ggml signed `kvalues_iq4nl`). Output unrelated to true weights. | Reimplement vs `dequantize_row_iq4_nl` w/ signed table. |
| QNT-4 | CRIT | `lib.rs:543-590` | `dequant_iq2s` is a placeholder (no grid lookup, `qh` never read) → plausible garbage, not error. | Real grid lookup or `Error::Unimplemented`. |
| QNT-5 | HIGH | `lib.rs:2107-2131` | `quant_iq2s` writes degenerate block (scales left zero; `qs[min(..)]` overwrites same byte 8× → 7/8 data discarded). | Fix encoder or remove. |
| QNT-6 | HIGH | `lib.rs:58-59,61` | `dequant_gptq_group_int` indexes `shape[0]/[1]` unchecked; `vec![0.0; in*out]` unchecked multiply (DoS). | Validate rank; `checked_mul`. |
| QNT-7 | LOW | `lib.rs:1170,1234` (+ `varbuilder.rs:585`, `convert.rs:~299`) | `bytes[*cursor..+8].try_into().unwrap()` guarded but triplicated. | One shared helper. |
| QNT-8 | MED | `lib.rs:3072,3104,3145,3155` | `partial_cmp(...).unwrap()` in EvoPress sorts panics on NaN fitness. | `total_cmp`. |

**Verified CORRECT (clean):** `dequant_q3k` (the "q3k buggy" note is STALE — q3k is clean; q2k is the broken one), `dequant_q6k` (signed i8 scales correct), `dequant_q5k`, `dequant_q80`, `dequant_packed_symmetric`, `f16_to_f32` (subnormal correct).

---

## 3. grim-backend-rocm  (LOC 40,708 — largest)  — review status: **DONE, 1 CRIT/HIGH**

| ID | Sev | Location | Issue | Fix |
|----|-----|----------|-------|-----|
| ROC-1 | HIGH | `kernels/qkv_attention.rs:617` (MODIFIED) | `let mut wlo = 0i32;` discards `window_lo` → sliding-window paged-attn silently returns **full-causal** (wrong answer + unbounded KV → OOM). Caller passes `window_lo` correctly. | `let mut wlo = window_lo;` |
| ROC-2 | MED | whole crate | No `catch_unwind`/`set_hook` on any FFI boundary (callbacks could unwind across ABI). | Wrap `extern "C"` bodies. |
| ROC-3 | LOW | `src/kernels/charon_backward.rs:192,198,203` | Unchecked `usize` multiply for grad buffer sizes. | `checked_mul`. |
| ROC-4 | LOW | `roc_device.rs:3655` | `drop((a_s,b_s,bt_s))` on Copy tuple (no-op warn). | Remove. |
| ROC-5 | LOW | `peer_access.rs:33-43` | Re-declares HIP FFI already in `handles.rs` (drift hazard). | Reuse crate FFI. |

**Verified CLEAN:** JIT cache keys on `seahash(source)+gpu_target+entry+solution_index` (no stale-arch bug); wavefront — RDNA omits `-mwavefrontsize32` (hipRTC rejects it), CDNA native, `hiprtcGetLoweredName` used for mangled names; quant decode (fp8 NaN, q4k `&63` unsigned, q6k signed char) matches invariants; allocator bucketed w/ `saturating_sub`; charon planner grid math; rccl/p2p/peer_access routing; cubecl `ArrayArg::from_raw_parts` lengths consistent.

**Prefill `window_lo` approximation (roc_device.rs:3290-3296):** intentional conservative bound, kernel causal cap holds — NOT a bug.

---

## 4. grim-backend-cuda  (LOC 6,689)  — review status: **DONE, 2 CRIT**

| ID | Sev | Location | Issue | Fix |
|----|-----|----------|-------|-----|
| CUDA-1 | CRIT | `src/lib.rs:2226` | `BackendDevice::qkv_attention` impl unconditionally calls `self.qkv_attention(...)` (itself) → infinite recursion / stack overflow on every attention call. | Delegate to inherent `qkv_attention_inner`. |
| CUDA-2 | CRIT | `src/lib.rs:1899,1901` | cuBLAS `cublasSgemm` `lda=k, ldb=m` swapped (col-major trick needs `lda=n, ldb=k`). Silently wrong for non-square; square `[2×2]` test masks it. | `lda=n, ldb=k`. |
| CUDA-3 | MED | `src/lib.rs:332` | `alloc_gpu` unchecked `elem_count()*byte_size` (device buffer overrun). | `checked_mul`. |
| CUDA-4 | LOW | `src/lib.rs:1131` | Q8_0 (34B) super-block: confirm device kernel stride == 34 (verify). | Runtime parity check. |

**Clean:** `selective_scan`/`rwkv_*` CPU fallbacks documented; no `unimplemented!`/`todo!` in lib.

---

## 5. grim-backend-metal  (LOC 5,761)  — review status: **DONE, 1 CRIT**

| ID | Sev | Location | Issue | Fix |
|----|-----|----------|-------|-----|
| METAL-1 | CRIT | `src/lib.rs:1471,1473` | device-absent matmul calls `a.as_ptr()/b.as_ptr()` on `&dyn BackendStorage` (no such method) — compile error on Apple or reads garbage; computed `a_vec/b_vec` unused. | Use `a_vec.as_ptr()/b_vec.as_ptr()`. |
| METAL-2 | LOW | `src/lib.rs:1175` | `unreachable!()` in quantize dispatch (`QuantFormat` added → panic). | `Error::Unimplemented`. |

**Clean:** `zeros/from_cpu/from_cpu_bytes` use `checked_mul`; `to_cpu_vec_f32` bounds-checked; FFI CString bound to locals (no dangling); embedding idx rebuild length-correct.

---

## 6. grim-backend-vulkan  (LOC 5,160)  — review status: **DONE, 1 MED**

| ID | Sev | Location | Issue | Fix |
|----|-----|----------|-------|-----|
| VULK-1 | MED | `src/lib.rs:749` | `alloc_gpu` unchecked `elem_count()*byte_size` → tiny `VkBufferCreateInfo.size` → mapping overrun. | `checked_mul`. |
| VULK-2 | LOW | `src/lib.rs:2926-2928` | quantized_matmul grid `n/m` — verify dequant out dims (looks correct). | Runtime parity check. |

**Clean:** `run_compute_shader` checks every `vk*` return; `Cleanup` Drop destroys all objects (no leak/double-free); `from_cpu`/embedding copy via `vkMapMemory`+`copy_nonoverlapping` with correct range; push-constant layout matches; CPU-fallback dequant routing sound. No `todo!`/`unimplemented` in lib.

---

## 7. grim-models (transformer/mamba/vision/audio/diffusion)  (LOC 27,630)  — review status: **DONE, 2 CRIT**

| ID | Sev | Location | Issue | Fix |
|----|-----|----------|-------|-----|
| MOD-1 | CRIT | `mamba/src/lib.rs:~218` (`step_block_gpu`) | Feeds `a_log` as the B-term to selective-scan (in-code TODO: `b_param` missing) → structurally wrong SSM (B=A). | Add real `b_param`, feed it. |
| MOD-2 | CRIT | `transformer/src/muse_glimmer.rs:~60` | `head_dim: u("head_dim")` has no `>0` guard → `head_dim=0` → div-by-zero / zero-stride KV. | `if >0 {..} else {hidden/num_heads}`. |
| MOD-3 | HIGH | `mamba/src/lib.rs:~290` (`step_block_cpu`) | Placeholder recurrence (`new_h = a*state + xz*(pos*0.01)`); no dt/B/C/discretization → wrong output, not error. | Real scan or `Unimplemented`. |
| MOD-4 | HIGH | `mamba/src/lib.rs:~290` | `step_block_cpu` ignores batch>1 (`xd[0]` only). | Iterate batch dim. |
| MOD-5 | MED | `mamba/src/rwkv.rs:~340` | `RmsNorm::load(... 1e-5)` hardcodes eps vs config. | Thread eps from cfg. |
| MOD-6 | MED | `mamba/src/rwkv.rs` | `load_tp` doesn't validate tensor *shapes* vs cfg → OOB-panic on malformed GGUF. | Validate dims at load. |
| MOD-7 | MED | `engine/src/model_loader.rs:~1410` | Unknown-arch arm silently `eprintln!` + falls back to `Llama::load_tp` (loaded-but-garbage model). | Return `Error` / `ModelArchitecture::Generic`. |
| MOD-8 | LOW | `transformer/src/model.rs:436`/block.rs:249 | `downcast_mut().expect()` / arena `expect` (post-grow, safe-ish). | Better typing. |
| MOD-9 | LOW | `vision/src/glimmer.rs` `merge_adjacent` | Averaging stand-in for token merge → different numerics vs ref (known). | Document / implement. |

**Verified CLEAN:** `moe_block.rs` (shared MoE, weight names match llama.cpp GGUF); `vision/vit.rs` (shapes correct); `transformer/block.rs` (KV grow/append bounds-consistent, RoPE tests present); `diffusion/scheduler.rs` (DDIM + `unknown-timestep→Config`); `audio/whisper.rs` (square-head assumption OK for Whisper).

---

## 8. grim-autograd  (LOC 11,853)  — review status: **DONE, 2 CRIT + 1 HIGH**

| ID | Sev | Location | Issue | Fix |
|----|-----|----------|-------|-----|
| AG-1 | CRIT | `src/adamw.rs:1102,1105` | `PagedAdamW` moment buffers init to `vec![1.0]` not `0.0` → first step `m=v=1` → huge update + wrong bias-corr. Others (AdamW/8Bit/Lion) init 0 correctly. | `vec![0.0f32;..]`. |
| AG-2 | HIGH | `src/scythe1.rs:130,33` | `fim_diag` init `0.0`, precondition divides by `max(FIM_EPS=1e-8)` on step 1 → g explodes ~1e8 → NaN/divergence for **default optimizer** (Scythe1). | Seed `fim_diag=1.0` or gate until FIM updated. |
| AG-3 | MED | `src/soul_eater.rs:291,294-296` | Σ-FIM seeds from first grad (not damped identity) while U/V seed `1.0+damping` → init asymmetry. | Seed Σ with `damping`. |
| AG-4 | MED | `param.rs` (39 unwrap) | `grad()`/shape `.dims()[0/1]` on user-supplied grads panic on missing/empty grad. | `?`/clear message. |

**Verified CLEAN:** `adamw.rs` main `step` formula + bias-correction correct; `loss.rs::cross_entropy_loss` (max-trick, bounds, `(softmax−onehot)/B` grad) correct; `scythe2.rs` controller/placement internally consistent.

---

## 9. grim-kvtransport + grim-disagg  — review status: **DONE, 1 CRIT**

| ID | Sev | Location | Issue | Fix |
|----|-----|----------|-------|-----|
| KVT-1 | CRIT | `kvtransport/src/lib.rs:568-570` (+`:476-484` client side) | `num_elements` raw `u32` off wire → `vec![0u8; num_elems*8]`; no cap → overflow/DoS. Hits **both** the receiver (`KvReceiverServer`/disagg reuse it) AND the client `fetch_block_remote` path (`:476-484` trusts peer header, `k_bytes`/`parse_f32_slice` assume honest peer). The `block_id` range check at :597 runs *after* the alloc. | Reject `num_elements > guard.block_elem_per_token()*BLOCK_SIZE` before alloc, on both send and receive. |
| KVT-2 | MED | `kvtransport/src/lib.rs:597-606` | Receiver assumes sender/receiver `elem_per_token` agree → silent KV truncation across heterogeneous nodes. | Send `elem_per_token` in header, reject mismatch. |

**Clean:** `KvBlockHeader::verify()` + `compute_checksum` correct; `read_layer_weights` rejects short files.

---

## 10. grim-speculative  (LOC 2,218)  — review status: **DONE, 1 CRIT**

| ID | Sev | Location | Issue | Fix |
|----|-----|----------|-------|-----|
| SPEC-1 | CRIT | `src/speculative_wrapper.rs:234-237,287-289,344-346,366-367` | Rejection sampling indexes `target_probs[i*vocab]` (rows 0..K) and returns `all_logits[..accepted*vocab]` — but extended input is `[input ++ draft]`, so draft token `i` target lives at row `S+i` and accepted logits at `S..S+accepted`. Both DSpark + native-MTP paths compare against *context* rows → speculative distribution ≠ target's → wrong tokens. | `row_start=(S+i)*vocab`; return rows `S..S+accepted`. |
| SPEC-2 | LOW | `src/speculative_wrapper.rs:385` | `extend_positions` defaults `last_pos=-1.0` on empty positions → RoPE mis-assign if pos empty but input non-empty. | Defensive. |

---

## 11. grim-garage  (LOC 11,517)  — review status: **DONE, 1 HIGH**

| ID | Sev | Location | Issue | Fix |
|----|-----|----------|-------|-----|
| GAR-1 | HIGH | `src/routes.rs:1179-1209,1222-1260` | `load_model_handler`/`chat_handler` call `load_from_path(model_path/model_id)` on client JSON with **no** `validate_job_path`/`prevent_path_traversal` guard (unlike bolt-on handlers) → loads `/etc/secrets.grim` or absolute path. Loopback by default (local impact). | Validate path before load. |

**Note (LOW):** plugin loaders (`wasm_loader.rs`/`dylib_loader.rs`) have no signature/allowlist gate before `dlopen`/instantiate — by-design for local plugin system, but recommend a gate before any networked reach.

---

## 12. grim-engine  (LOC 8,534)  — review status: **DONE (no CRIT found)**

- Default bind loopback; no `todo!`/`unimplemented` in lib code. Verified: `scythe2` controller, `streaming_forward`, `model_loader` dispatch (see MOD-7 for the silent Llama fallback). No standalone findings beyond MOD-7.

## 13. grim-server  (LOC 5,769)  — review status: **DONE, multiple HIGH/CRIT-class**

| ID | Sev | Location | Issue | Fix |
|----|-----|----------|-------|-----|
| SRV-1 | HIGH | `lib.rs:1033` | `m.get("role").unwrap()` → panic/DoS on malformed message. | `unwrap_or("user")`. |
| SRV-2 | HIGH | `lib.rs:550` | `tool_calls.as_array().unwrap()` panic if present-but-non-array. | `and_then(|v| v.as_array())`. |
| SRV-3 | HIGH | `lib.rs:1064` | context-length check skipped when `context_length==0` → unbounded prompt → KV OOM. | Hard ceiling when 0. |
| SRV-4 | HIGH | `lib.rs:1183,1265` | generation loop up to `max_tokens` (cap 65536) × ~10ms lock/sleep → one request holds engine mutex ~11 min (availability DoS). | Lower cap; yield/timeout. |
| SRV-5 | HIGH | `lib.rs:1920-2002` / `load_model` | Unauthenticated `/load` loads arbitrary GGUF + allocates tensors, no size cap. | Auth or size cap. |
| SRV-6 | HIGH | `lib.rs:2827`+1063 | 10 MiB body cap; `max_tokens as usize + prompt_tokens.len()` unchecked → overflow on attacker `max_tokens`. | Cap `max_tokens`; `checked_add`. |
| SRV-7 | HIGH | `lib.rs:661,757,...,3011` | Engine mutex `.lock().unwrap()` everywhere → one poison panic kills server. | `into_inner()` recovery (streaming paths already do). |
| SRV-8 | MED | `lib.rs:2152-2162` | `/api/stats` + dashboard hardcode `params:"8B"`, `ctx_limit:8192`, `ttft_ms:820`, `prefill_tps:12.3` for **every** model (fabricated telemetry). | Real values. |
| SRV-9 | MED | `lib.rs:1996` | non-streaming `eval_duration = (eval_count*0) as u64` → always 0. | Real timing. |
| SRV-10 | MED | `lib.rs:1044` | `content.as_str().unwrap_or("")` drops array/multimodal content → silent wrong completion. | Handle array content. |
| SRV-11 | MED | `lib.rs:1283,1423` | `strip_suffix(token_text)` EOS strip fragile (truncates on common suffix like space). | Exact match / token-id based. |
| SRV-12 | MED | `lib.rs:1469-1519` | tool-call budget counts `&messages` only, excludes in-flight streamed calls → cap exceeds in one long response. | Count in-flight. |
| SRV-13 | LOW | `lib.rs:1543` | static `id="chatcmpl-000"` (TODO) → client id collisions. | Unique id. |
| SRV-14 | LOW | `lib.rs:2905` | `gguf_path.to_str().unwrap()` at startup → panic on non-UTF8 path. | Handle. |

**Clean/Good:** stub endpoints return proper 501 (`embeddings`/`audio`/`images`/`grpc`); `is_safe_model_path` traversal guard; `validate_public_url` SSRF block; `cancel_request` idempotency; `split_think_content` logic; `pack_training_examples` length invariants.

---

## 14. grim-cli  (LOC 7,855)  — review status: **DONE**

| ID | Sev | Location | Issue | Fix |
|----|-----|----------|-------|-----|
| CLI-1 | HIGH | `main.rs:729` (+ server bind) | `grim serve --address 0.0.0.0:11434` exposes the unauthenticated server (see SRV-5) to all interfaces; no `is_public_ip` guard on serve bind. | Default 127.0.0.1; warn/refuse 0.0.0.0. |
| CLI-2 | MED | `main.rs:241` | `Quantize` subcommand appears to be a no-op stub (no handler body observed). | Confirm/implement. |
| CLI-3 | LOW | `main.rs:761` | `unsafe { env::set_var("GRIM_BACKEND", dev) }` non-atomic process-global. | Pass via arg/struct. |
| CLI-4 | LOW | `train.rs:536` | `epochs*dataset.len()/accum` integer-divides → `total_steps==0`→clamp 1. | Clamp before. |

**Clean:** `Command::new(program).args(&args)` (no shell → no injection); `train.rs:294` Alpaca label truncation verified consistent; `IGNORE_INDEX=-100u32` standard; optimizer/scheduler clap enums fail-loud. `catalog.rs::is_safe_model_path` mitigates traversal; `client.rs::validate_public_url` mitigates SSRF (cli-only, Low).

---

## 15. grim-backend-cpu  (LOC 2,914)  — review status: **DONE**

| ID | Sev | Location | Issue | Fix |
|----|-----|----------|-------|-----|
| CPU-1 | LOW/MED | `simd_gemm.rs:17` / `device.rs:959` | `gemm_f32_simd`/`gemm_f32_lora_fused` are **dead** — `gemm_dispatch` routes to scalar/`oxiblas_sgemm`. README claims "SIMD-accelerated GEMM (`gemm_f32_simd`)" → false claim + dead code. | Wire in or drop + fix README. |
| CPU-2 | LOW | `simd_gemm.rs:33-65` | AVX2 accumulator single `f32` horizontal sum → precision loss for large K. | Blocked/d64 accumulation. |

**Clean:** `dequant_gemm.rs` (bounds-safe, tested); `broadcast_index` (device.rs:1055); `gemm_scalar`/`gemv_row`/`oxiblas_sgemm`.

---

## 16. Smaller crates — review status: **DONE, CLEAN or trivial**

- **grim-core** (4,753): `catalog.rs:288,383` unsanitized model name join (path traversal — **HIGH**, overlaps SRV/CLI guards; `is_safe_model_path` mitigates the resolved branch). `paths.rs` clean.
- **grim-nn** (5,413): `varbuilder.rs:508-515` CPU arm returns "unreachable" error after computing a tensor (dead/misleading); `:584-591` duplicated GPTQ segment reader w/ potential 32-bit `cursor+len` overflow (`checked_add`). Otherwise clean.
- **grim-tensor** (3,161): `shape.rs`/`dtype.rs` byte-size table correct, bounds-checked. Clean.
- **grim-tensor-graph** (328): no unwrap/todo/unsafe outside tests. Clean.
- **grim-memory** (1,041): `moe_budget.rs:88` accounting sound (`:97` cosmetic pre-promotion value in error msg). Clean.
- **grim-scheduler** (1,206): `lib.rs:76,103,127,132` poison-mutex unwrap in admission path (Low → `into_inner()`).
- **grim-kvquant** (3,014): `kv_omni.rs` dequant indexing internally consistent; no OOB. Clean.
- **grim-plugin** (1,486): see GAR note (no signature gate — by-design local). Clean otherwise.
- **grim-disagg** (819): see KVT-1 (reuses kvtransport receiver). No other findings.

---

## 17. PRIORITIZED FIX LIST (do these first)

1. **FMT-1 / FMT-2** — GGUF dtype + block-size table wrong (corrupts all IQ/BF16 models).
2. **QNT-1** — `dequant_q2k` d/dmin aliasing + 82≠84 (confirmed by source read).
3. **QNT-3 / QNT-4** — `dequant_iq4nl` / `dequant_iq2s` fabricated/placeholder decoders.
4. **CUDA-1 / CUDA-2** — infinite recursion in `qkv_attention` + cuBLAS lda/ldb swap (every attention + non-square matmul wrong).
5. **METAL-1** — device-absent matmul reads wrong/garbage pointers.
6. **ROC-1** — SWA paged-attn `window_lo` discarded → full-causal (modified file).
7. **SPEC-1** — speculative-decode target logit row misalignment (both paths).
8. **AG-1 / AG-2** — PagedAdamW moments init 1.0; SCYTHE1 FIM div-by-eps (default optimizer NaN).
9. **KVT-1** — kvtransport unbounded alloc / remote DoS.
10. **MOD-1 / MOD-2** — Mamba GPU B-term wrong; muse_glimmer head_dim=0.
11. **SRV-1..7 / CLI-1** — server panics/DoS + 0.0.0.0 unauth exposure.
12. **FMT-3/4/7/8/9, GAR-1, core catalog traversal** — untrusted-input size/alloc/path bounds.

---

## 18. Verification gaps / honesty note

- **No `cargo build`/`cargo test` was executed** — the user blocked the single network-touching
  compile probe. All findings are static (source + grep). Line numbers are from the current
  working tree (2 files modified: `roc_device.rs`, `qkv_attention.rs`).
- **Two CRIT claims were independently re-read against source by the orchestrator** (FMT-1 tag
  map shift; QNT-1 `d`/`dmin` aliasing at `lib.rs:830-831`) and confirmed.
- **Numeric kernel correctness** (HIP/GLSL/Metal compute) is unverified without a device — flagged
  as "verify" items (CUDA-4, VULK-2, charon_backward ROC-3).
- **Skill substitution:** `caveman`, `ponytail-audit`, `ml-llm` are not installed in this env;
  their intent (caveman=deep static bug hunt, ponytail-audit=paranoid review, ml-llm=ML model
  correctness) was covered by `rust-expert` + `rust-ffi` + `rocm/rocm-hip` + `kernel-review` +
  `grim-rocm-ffi` + `grim-moe-quant-kernels` + dedicated per-crate subagents.
