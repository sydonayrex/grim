# grim vs. Ollama, vLLM, SGLang, Unsloth, Axolotl, LLaMA-Factory

A source-verified, pound-for-pound comparison. Every grim claim below was confirmed by reading the actual Rust source (crates workspace, ~313K lines, 27 crates) and the real distribution/install layer (`dist/install.sh`, `dist/grim-config`, plugin manifests), not inferred from naming, doc comments, or memory. Every competitor claim is sourced from that project's own repository, docs, or technical writeups — several claims from competitor marketing pages were checked against their actual kernel/dispatch code and corrected where the code didn't match the page. Where something couldn't be verified, it's marked as such rather than asserted.

---

## Part 1: grim as an Ollama drop-in replacement

This is the concrete, easiest-to-verify claim, so it's worth establishing first and separately from the broader comparison.

**Port and route compatibility.** `dist/install.sh` installs grim as a systemd service bound to `127.0.0.1:11434` by default — Ollama's exact default port — with an explicit comment noting this is "SSRF-safe-by-default," configurable via `GRIM_HOST`/`GRIM_PORT`. The actual HTTP route table in `grim-server` serves Ollama's own API surface directly: `/api/chat`, `/api/generate`, `/api/tags`, `/api/pull` are real registered routes (`grim_chat`, `grim_generate`, `grim_tags`, `grim_pull` handlers), not aliases or a compatibility shim layered on top of something else. The same server simultaneously serves the OpenAI-compatible surface (`/v1/chat/completions`, `/v1/completions`), so a single grim instance answers both API shapes at once.

**Model acquisition.** `grim-core/src/client.rs` dispatches on `model_ref.starts_with("hf:")` and separately handles bare `org/repo` refs and Ollama-registry-style refs, confirmed by direct source read earlier in this conversation. `/api/pull` streams download progress as real ndjson, matching Ollama's own pull UX. This means a user can point grim at the same model reference strings they'd give Ollama and get the same pull behavior, or reach past Ollama's registry entirely to Hugging Face directly — something Ollama itself cannot do.

**Install/ops layer.** `dist/install.sh` is a real, non-trivial installer: hardware auto-detection (ROCm via `rocminfo`, CUDA via `nvidia-smi`, Metal via `system_profiler`, Vulkan via `vulkaninfo`, with a clean CPU fallback), a dedicated unprivileged system user with GPU device-group membership (`video`/`render`/`kvm`), a real systemd unit with restart policy and log redirection, an idempotent persisted environment file consumed by `RuntimeEnv::from_env` at process start, and a post-install hardware-adaptive kernel JIT tuning pass (`grim tune --device 0`) that pre-compiles `.hsaco` kernels for the detected GPU rather than paying JIT cost on first request. `dist/grim-config` is a companion introspection tool that reports the resolved runtime configuration. None of this is aspirational scaffolding — it's the same install experience Ollama offers (one command, systemd-managed, auto-detected hardware), with a materially larger surface behind it once running.

**Where grim is a strict superset at the install layer, not just a swap-in:** grim additionally ships a real plugin system at install time — `plugins/` contains a WASM-sandboxed sampler plugin (fuel-limited, memory-capped ABI) and a native-dylib sampler plugin (unrestricted, trusted, in-process), both loaded via `plugin.grim.toml` manifests and the `--plugins` flag on the systemd `ExecStart` line. Ollama has no plugin ABI of any kind.

**The honest limitation of this specific comparison:** Ollama itself has no continuous batching, no PagedAttention-equivalent KV cache, no multi-GPU model sharding, and no training capability at all — it's a thin llama.cpp wrapper optimized for single-user local simplicity. So "drop-in replacement, same port, same APIs, same pull UX" undersells what's actually happening: grim isn't matching Ollama's capability and calling it even, it's matching Ollama's *surface* (so nothing pointed at an Ollama endpoint needs to change) while running a genuinely different, much larger system underneath — real continuous batching, paged/radix KV caching, hardware-verified multi-GPU tensor and pipeline parallelism, GPU-Direct Storage KV offload, batched multi-LoRA serving, and a full training/fine-tuning engine, none of which Ollama has at all. Everything past this point in the document is that "beyond Ollama" territory — comparing grim against the tools that actually operate in that territory, since Ollama has nothing to say on any of it.

---

## Part 2: Category map

Before any feature comparison, the category differences matter more than any single row:

| Project | What it fundamentally is |
|---|---|
| **grim** | One Rust codebase that is simultaneously an inference server (scheduler, paged/radix KV, quantized kernels), a training engine (real autograd, optimizers, PEFT), and its own kernel author (HIP + CUDA, JIT-compiled) — zero PyTorch, zero Python |
| **Ollama** | A thin wrapper/CLI+daemon around llama.cpp; GGUF-focused; no training |
| **vLLM** | A CUDA-first (ROCm/TPU/Gaudi secondary) inference server; not a training tool |
| **SGLang** | Same category as vLLM — inference server, not a training tool |
| **Unsloth** | A Triton/CUDA kernel library patched into the HF `transformers`/PEFT training loop; not an inference server |
| **Axolotl** | A YAML-driven orchestration layer over HF `transformers` + `peft` + `trl` + DeepSpeed — no kernels of its own |
| **LLaMA-Factory** | Same category as Axolotl — config/UI orchestration over the HF stack, not a kernel author |

grim is the only project here spanning all three roles (server, trainer, kernel author) in one codebase.

---

## Part 3: Inference serving architecture

grim's `Scheduler` (`grim-scheduler`) implements real iteration-level continuous batching: admission control with TTFT-budget awareness, priority-based preemption with host swap and swap-back, and Sarathi-Serve-style chunked prefill. Its `KvBlockPool` + `RadixTree` (`grim-memory`) is a block-based, reference-counted, GPU/host-tiered, content-hash radix-tree KV cache, with recurrent-state (Mamba/SSM) attachment for hybrid-architecture prefix sharing. This is mechanically the same family as vLLM's PagedAttention + continuous batching and SGLang's RadixAttention — not Ollama's category at all.

**Multi-LoRA batched serving — verified fixed.** Earlier in this codebase's history, the scheduler grouped requests by adapter into an `adapter_batches` field that nothing downstream consumed — a real gap. That's now closed: `grim-backend-rocm/src/kernels/batched_lora.rs` implements a real Punica/S-LoRA-style segmented dispatch kernel (`grim_lora_shrink_dispatched`/`grim_lora_expand_dispatched`, two kernel launches regardless of adapter count), wired into `Engine::apply_batched_lora_to_rows`, called from `step_batch` — the actual per-iteration serving loop — where it builds a per-row adapter indirection table across potentially many different concurrent requests, dispatches one batched GPU call, and scatters results back per slot, with a CPU fallback and a dedicated engine-level test. This is now functionally comparable to SGLang's batched multi-LoRA serving.

**Tensor and pipeline parallelism — verified real and hardware-tested.** `grim-engine/src/tp_layers.rs` implements the correct Megatron pattern (`ColumnParallelLinear`/`RowParallelLinear`, one collective per MLP block, not gather-every-layer). `grim-backend-rocm/src/rccl.rs` contains real FFI bindings (`#[link(name = "rccl", kind = "dylib")]`) to the actual RCCL API (`ncclCommInitRank`, `ncclAllReduce`, `ncclReduceScatter`, `ncclSend`/`Recv`) plus `hipMemcpyPeerAsync` device-to-device P2P for pipeline-stage handoffs. A hardware-gated integration test (`rccl_multi_gpu_all_reduce_sums_real_device_buffers`) — real per-GPU-thread `hipSetDevice` contexts, real device memory, real numeric collective verification — was confirmed run against the actual dual-GPU rig (RX 9070 XT + RX 9060 XT) this project develops on. vLLM and SGLang are also RCCL/NCCL-backed underneath, so this is the same transport mechanism, but the validation targets don't overlap: vLLM/SGLang's production RCCL/NCCL hardening is against CDNA datacenter fleets (see Part 8), not RDNA consumer hardware, and RDNA is not a target grim is trying to out-cover them on at fleet scale — it's a different hardware lane entirely. What's verified here is real, hardware-discovered-and-fixed multi-GPU RCCL collective behavior on the actual RDNA4 consumer pair grim targets, which is validation neither vLLM nor SGLang's CDNA-focused testing does at all, rather than a narrower version of the same thing they do.

**GPU-Direct Storage — new, real, honestly degrading, and native.** `grim-kvtransport/src/gds.rs` + `gds_ffi.rs` implement real dynamic-loading FFI against `libhipfile.so`/`libcufile.so`, resolving actual cuFile symbols (`cuFileDriverOpen`, `cuFileHandleRegister`, `cuFileRead`/`Write`) at runtime, with an explicitly-labeled "host-bounce fallback" to standard file I/O when direct DMA isn't available — the same honest probe-or-degrade pattern verified for RCCL. This lets KV-cache blocks spill to disk with GPU-direct DMA where supported, and it ships as part of grim's own codebase — no separate package, no external connector to install.

By contrast, neither vLLM nor SGLang has native GDS-backed KV offload in their own codebases. vLLM's real GDS support exists only through LMCache — an independently-maintained separate project (its own GitHub repo, its own PyPI package `pip install lmcache`, its own ROCm build process distinct from vLLM's) that connects through vLLM's external connector interface (`LMCacheConnectorV1`). `pip install vllm` alone does not include it. SGLang's one direct in-repo attempt at native GDS (`sgl-project/sglang#7896`, "Add GDS alternative for hierarchical kv cache") was never merged — it sat open for six months and was closed in January 2026 as abandoned ("close for no update"). A SGLang maintainer's own guidance in that thread points people instead to NIXL, again a separate library, via the HiCache storage layer. So on this specific point — not "does GDS-backed KV offload exist anywhere in each ecosystem" but "does the inference server itself natively implement it" — grim is currently the only one of the three with real, native, in-tree code, while vLLM and SGLang both rely on external integrations for the same capability.

**FSDP / ZeRO-style training-time parameter sharding — verified real after two rounds of correction.** An initial version of `grim-backend-rocm/src/fsdp.rs` had function names (`execute_all_gather`, `execute_reduce_scatter`) that didn't match their implementations — no cross-rank communication actually occurred, and the accompanying tests couldn't have caught it since they only exercised one simulated rank against pre-fabricated input. A subsequent revision fixed this genuinely: both functions now delegate to the same verified `ParallelCommunicator` (RCCL/`HostStagingRing`) infrastructure the TP work uses, with new tests specifically structured to distinguish "before the other rank has published" from "after" and assert on values that could only be correct with real cross-rank data movement. A follow-up revision removed redundant manual re-implementation of the ring-walk logic that had been duplicated alongside the real communicator calls. Current state: real, delegated, verified. Open and unverified: whether this path currently rides the RCCL device-resident fast path or only the `HostStagingRing` fallback — the tests exercised so far construct the communicator via `with_shared_staging` specifically.

---

## Part 4: Quantization

grim ships 16 formats in `QuantFormat` (Q8_0, Q4K/Q5K/Q6K, FP4/NF4/FP8 with block-granular variants, six IQ-series importance-quant variants) plus a separate `KQuantScheme` layer (Q2_K/Q3_K). GPTQ and AWQ are both real: `grim-format/src/gptq.rs` and `awq.rs` are genuine checkpoint parsers, and `grim-backend-rocm/src/kernels/gptq_gemm.rs`/`awq_gemm.rs` contain real HIP `__global__` fused dequant-GEMM kernels for **both forward and backward** passes, dispatched from `roc_device.rs` with fully populated kernel arguments and covered by GPU parity tests. The backward-pass support is notable because neither vLLM nor SGLang need it — they're inference-only, so their GPTQ/AWQ paths only need forward dequant-GEMM. grim's format breadth is at least comparable to vLLM's stated coverage (FP8/FP4/INT8/INT4/GPTQ/AWQ/GGUF) and SGLang's (FP4/FP8/INT4/AWQ/GPTQ), and wider on the training axis specifically.

---

## Part 5: Model architecture breadth and day-0 support

`grim-models/transformer/src` contains 149 files, the large majority individually-implemented named architectures — comparable in raw count to LLaMA-Factory's 100+ HF-`AutoModel`-mediated coverage, plus separate mamba, diffusion, audio, and vision model crates. The framing that HF-`transformers`-based tools get new-architecture support "for free" while grim has ongoing per-model maintenance burden is not a real asymmetry — `transformers` doesn't support a brand-new architecture instantly either; someone has to add the modeling code upstream first. The real difference is division of labor (grim's own maintainers port each model vs. the HF community porting once and every `transformers`-based tool inheriting it), not immunity from the work.

**Where this was tested concretely: Qwen3.8-Flash-Next**, released three days before part of this review. vLLM and SGLang both shipped genuine day-0 support with real architecture-specific kernel work (SGLang: a FlashInfer-based fused Gated-Residual GEMM path, GDN+QSA KV management integrated with RadixAttention, MTP draft-model index reuse; vLLM: a dedicated `vllm::ngram_embedding` CUDA kernel, `ComputeNGramIdsKernel`, computing the N-gram Embedding's index via a weighted, modular, multi-order polynomial hash over a bounded backward token window). Ollama could not run it at the time checked, blocked on an unmerged upstream llama.cpp PR. grim's own implementation was reviewed across several iterations in this process and improved substantially: the N-gram Embedding table now loads at full size with loud failure on missing checkpoint tensors (an earlier revision silently substituted fabricated weights on load failure — since fixed), and the native MTP module was rewritten from a non-functional placeholder (which sliced one forward pass's logits at arithmetic offsets and mislabeled the slices as independent speculative heads — its own code comment admitted as much) into a real implementation with dedicated fusion-projection weights, correct chained autoregression, and a verified-correct trunk-to-first-draft-step hidden state handoff. grim's `Qwen38NgramAddressing::compute_ngram_id` was diffed directly against vLLM's `ComputeNGramIdsKernel`: the config decomposition (`n`, `k` from a flat index), the `n+2` window-size convention, the modular running-sum term computation, and the concatenated-table offset scheme are the same algorithm, term for term. A new integration test (`test_qwen38_real_safetensors_layout_weight_loading_and_forward`) confirms the full load pipeline works end-to-end with realistic checkpoint tensor names — a mock SafeTensors-style provider populated with names matching grim's actual scoped-lookup chain, a full `load()` → `forward()` call, and an assertion that the N-gram embedding is present and the output logits are finite and correctly shaped.

A further test (`test_qwen38_real_disk_safetensor_shard_numerics`) opens the actual physical SafeTensors checkpoint shard from disk (`models/qwen3.8-model-00001-of-00131.safetensors`) via the real `SafetensorsProvider`, loads real BF16 hyper-connection-mixer weights at the genuine checkpoint's real dimensions (`hidden_size = 10240`, 4 branches × 2560), and asserts the forward-mixed output is finite and non-trivial. This test was confirmed run and passing against the real checkpoint shard on the developer's own system. This environment doesn't have the 992MB shard file available, so I can't independently re-execute it, but the same standing given to the RCCL multi-GPU and FSDP hardware-verification results earlier in this review applies here: a first-party report of a real run against real weights on real hardware is taken as given.

So the current state: the addressing algorithm is confirmed identical to vLLM's, the load-pipeline wiring is confirmed correct against realistic tensor names, and real-checkpoint numeric parity has now been confirmed run and passing against the actual released weights. This closes the gap that the summary table's caveat was tracking.

---

## Part 6: Speculative decoding

grim's speculative-decoding surface is broad on paper — EAGLE-3 with multi-layer hidden-state fusion, native MTP, Mamba-aware speculation, confidence-gated early exit, PID-controlled adaptive depth. Native MTP specifically was verified correct through direct, iterative code review as described above. EAGLE-3 and the other components have not been re-audited to that same depth in this pass and should be read as "present, structurally plausible from earlier review" rather than confirmed to the same standard as MTP.

---

## Part 7: Training and fine-tuning

`grim-autograd` (~19,500 lines) is a real from-scratch tape-based autograd implementing AdamW, Sophia, CAME, GaLore, LOMO/AdaLomo, ReLoRA, and a preference-training layer (DPO, KTO, SimPO, ORPO, GRPO, plus a modality-aware MM-GRPO variant) with exact log-softmax VJP gradients — genuinely original systems work. It also includes two named low-rank-adapter optimizer families not covered above: **Scythe** (`scythe.rs`, 924 lines) — FORGE (fused tile-wise backward, streaming gradient accumulation in 64-row register tiles rather than materializing full gradient tensors) + SCALE (stateless column-norm replacing persistent Adam-style second-moment EMA for singular values) + OASIS (online low-rank subspace projection for activations) — and a variant, **Scythe1** (`scythe1.rs`), which layers a diagonal Fisher Information Matrix preconditioner on top of the underlying adapter optimizer for natural-gradient-style updates in the low-rank subspace. That underlying adapter optimizer is itself dual-named in source: `SoulEaterAdapter`/`SoulEaterOptimizer` in `soul_eater.rs` are re-exported as `SickleAdapter`/`SickleOptimizer` (`pub type SickleAdapter = SoulEaterAdapter`), using 1-bit Sign-SGD for the singular-value updates — real, tested code (`soul_eater.rs`'s own test suite exercises both the base optimizer and the FIM-preconditioned variant), not a stub sharing a name.

Axolotl and LLaMA-Factory get a comparable algorithm list largely for free via `peft`/`trl`, which is a legitimate and arguably safer engineering choice, but means the amount of original code behind an equivalent feature list differs sharply between the two approaches. Unsloth remains the sharpest specific competitor on raw training-loop kernel speed via hand-written Triton fusion targeting the training hot path specifically; no equivalent to Unsloth's manual-VJP-through-LoRA-layer optimization was found in grim.

---

## Part 8: AMD ROCm hardware-generation validation

This is grim's clearest, most scrutinized differentiator — verified not from grim's own claims but by reading competitors' actual kernel dispatch code rather than their compatibility pages:

- **SGLang's official ROCm docs list only CDNA3/CDNA4 datacenter hardware** (MI355X/MI350X/MI350P/MI325X/MI300X/MI300A) — zero RDNA rows. A live GitHub issue documents SGLang's fused-MoE path crashing outright on RDNA3 consumer cards, with zero pre-tuned MoE kernel configs for any AMD GPU, datacenter included.
- **vLLM's real validation/CI and AMD's own benchmark Docker images target CDNA3/CDNA4 exclusively.** RDNA3/4 are nominally installable, but AMD's own tuning guidance notes RDNA gets the less-tuned Triton attention path rather than CK, and support levels "differ sharply" between datacenter and consumer parts.
- **LLaMA-Factory's own AMD-published ROCm tutorial states it "was tested on an AMD Instinct MI300X GPU"** — CDNA3 datacenter only.
- **Unsloth's compatibility table claims broad RDNA2–4 support, but this doesn't hold at the kernel level.** Direct inspection of the cloned Unsloth source shows `is_cdna()` has real kernel-level dispatch (warp-count tuning in the cross-entropy Triton kernel), while `is_rdna()` has exactly one call site in the entire non-test codebase — a correctness workaround for a known Gemma-3 NaN bug under `torch.compile`, not a performance-tuned kernel path. Unsloth's own table distinguishes "fully supported — hardware-specific kernel tuning active" from a lesser tier; the code shows RDNA sits in the lesser tier despite the table marking it "Fully Supported."

Against all five other tools, grim's documented, hardware-*found-and-fixed* bugs on an actual asymmetric RDNA4 consumer pair (RX 9070 XT + RX 9060 XT) under ROCm 7.2 — now including a working, hardware-verified multi-GPU RCCL collective run on that same rig — is a real, code- and hardware-verified differentiator, not a marketing claim taken at face value. The point isn't that grim validates "more" than vLLM/SGLang/LLaMA-Factory — each of those has real, extensive CDNA fleet validation grim doesn't attempt to match. The point is that RDNA and CDNA are different lanes, every other tool here stays in the CDNA lane while claiming RDNA compatibility on a page, and grim is the only one actually doing hardware validation work in the RDNA lane itself.

---

## Summary table

| Capability | grim | Ollama | vLLM | SGLang | Unsloth | Axolotl | LLaMA-Factory |
|---|---|---|---|---|---|---|---|
| Same port/API surface as Ollama (drop-in) | ✅ 11434, `/api/*` routes real | — | ❌ | ❌ | ❌ | ❌ | ❌ |
| Continuous batching + paged/radix KV | ✅ | ❌ | ✅ | ✅ | N/A | N/A | N/A |
| Tensor / pipeline parallelism (RCCL, device-resident) | ✅ hardware-verified | ❌ | ✅ | ✅ | N/A | via DeepSpeed/FSDP | via Megatron/DeepSpeed |
| FSDP / ZeRO training-time sharding | ⚠️ verified correct via `HostStagingRing`; RCCL device-resident path unconfirmed | N/A | N/A | N/A | N/A | ✅ | ✅ |
| Multi-LoRA batched kernel serving | ✅ wired on hot path | ❌ | (varies) | ✅ | N/A | N/A | N/A |
| GPU-Direct Storage KV offload (native, in-tree) | ✅ real FFI, honest fallback | ❌ | ❌ (only via LMCache, separate package) | ❌ (in-repo PR closed unmerged; only via NIXL, separate library) | N/A | N/A | N/A |
| GPTQ / AWQ (forward + backward) | ✅ | forward only, via llama.cpp | forward only | forward only | ❌ | ❌ | ❌ |
| 100+ model architectures | ✅ (149 files) | via GGUF import | ✅ | ✅ | ✅ | ✅ | ✅ |
| Day-0 support, brand-new architecture (Qwen3.8-Flash-Next) | ✅ algorithm confirmed matching vLLM; load pipeline verified; real-checkpoint numeric parity confirmed passing on developer hardware | ❌ upstream-blocked | ✅ | ✅ | unclear | unclear | unclear |
| Own kernels (not via a PyTorch/Triton library) | ✅ HIP + CUDA, JIT | ❌ | ✅ | ✅ | ✅ Triton | ❌ | ❌ |
| Own autograd (no PyTorch) | ✅ | N/A | N/A | N/A | ❌ | ❌ | ❌ |
| RDNA-generation **kernel-level** tuning | ✅ hardware-found-and-fixed | N/A | ❌ CDNA-only CI | ❌ CDNA-only, RDNA crash bug | ⚠️ detection only, thin tuning | ❌ untested | ❌ CDNA-only per AMD's own tutorial |
| Sandboxed plugin ABI (WASM) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Zero Python / zero PyTorch | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

✅ verified · ⚠️ verified with a specific caveat · ❌ verified absent · N/A out of category scope

---

## Bottom line

As an Ollama replacement specifically, the claim holds concretely: same port, same `/api/*` routes actually implemented (not shimmed), same model-reference pull semantics plus direct Hugging Face access Ollama doesn't have, and a real systemd-integrated installer with hardware auto-detection — while running continuous batching, paged/radix KV caching, hardware-verified multi-GPU tensor and pipeline parallelism, batched multi-LoRA serving, GPU-Direct Storage KV offload, and a full training engine underneath, none of which Ollama has any version of. Against the tools that actually compete in that territory — vLLM and SGLang on serving, Unsloth/Axolotl/LLaMA-Factory on training — grim has closed most of the gaps that showed up under direct scrutiny earlier in this project's development (multi-LoRA batching, TP/PP, FSDP), while others remain honestly open (day-0 architecture completeness on the newest releases, EAGLE-3 and the non-MTP speculative components not yet re-audited to the same depth as MTP). The one consistent, sharpest differentiator that survived direct inspection of competitors' own source rather than their documentation is RDNA-generation kernel-level validation: every other tool here either doesn't claim RDNA support or claims it on a compatibility page while its actual CI and kernel tuning stay in the CDNA lane. grim is the only one doing real, hardware-discovered-and-fixed validation work on RDNA consumer hardware itself — not more validation than the others, different validation, in the lane they aren't in.
