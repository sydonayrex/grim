# 1-to-Many: GDL + Inference/Training Perf Rollout for `grim-models/transformers`

> Audit → plan. Gold standard: `crates/grim-models/transformer/src/lfm2.rs` (+ `gla*.rs`, `lfm2_graph.rs`).
> Scope: all 158 files in `crates/grim-models/transformer/src/*.rs`.
> Goal: every model file uses GDL **effectively where needed** — and every model file uses the
> applicable grim performance optimizations for inference and training. GDL is **not** universal;
> applying it to pure-softmax / MLA / SSM architectures would corrupt model math. This plan makes the
> correct per-architecture assignment explicit.

---

## 1. What currently exists (evidence, not opinion)

### 1.1 The lfm2.rs gold standard — GDL inference-engine requirements

GDL = GRAVE Gated DeltaNet-2 linear attention (arXiv 2605.22791 Eq. 8–12), f64 oracle in
`crates/grim-models/transformer/src/gla.rs`. lfm2.rs is the **only** model that wires the full stack:

| Requirement | Location | Contract |
|---|---|---|
| Mode enum | `lfm2.rs:51-56` `Lfm2AttentionMode::{Softmax,Gdl}` | `#[default] Softmax`; GDL is opt-in per config |
| Config flag | `lfm2.rs:80` `attention_mode` | Selects GDL vs softmax for non-recurrent layers |
| Block fields | `lfm2.rs:268-270` `gdl_b_proj/w_proj/f_proj: Option<Linear>`; `lfm2.rs:297` `attention_mode`; `lfm2.rs:301` `gdl_gates: [f64;3]` | Erase/write/decay-modulation projections (None = static scalar gates); per-layer `[decay, erase, write]` operating point |
| Gate init (single source of truth) | `gla.rs:26` `gdl_gate_defaults(head_dim)` → `[2^(-1/d), 0.5, 0.5]`; `gla.rs:38` `tc_depth_scaled_gates()` | Taylor-Calibrated neutral point; depth-scaled variant for hybrids |
| Load defaults | `lfm2.rs:598-601` | Every block gets `gdl_gate_defaults(cfg.head_dim)`, projections `None` until distillation writes them |
| Load-time validation | `lfm2.rs:605-666` `validate()` | ShortConv variant requires conv triple; **GDL variant (`:626`) requires `wq,wk,wv,wo`**; softmax variant additionally requires `attn_q_norm/k_norm` + coherent `wqkv_codes/exps`; MoE requires `ffn_gate_inp` |
| Forward dispatch | `lfm2.rs:668` `forward()` → `:852-853` `forward_gdl()` when `attention_mode == Gdl` | ShortConv → `shortconv_*`; MoE-FFN → `forward_moe_ffn`; GDL → `forward_gdl`; else fused/softmax attention |
| Eager GDL (host oracle path) | `lfm2.rs:2522` `forward_gdl()` | QKV projections → `grave_gate_override(index).unwrap_or(gdl_gates)` (`:2538`) → try `forward_gdl_device` first (`:2544-2550`), fall back to host loop (L2-normalized q/k per GDN-2 §3.5, GQA head-repeat via `kv_group`) |
| Device GDL (fused, zero readback) | `lfm2.rs:2402` `forward_gdl_device()` | ROCm-only; **hard constraint `:2428` `hd != 64 \|\| nh != nkv` → `Unimplemented`** (GQA + non-64 stay on CPU until kernels land); state `[nh*64*64]` owned/restored via cache; gates uploaded once (~1KB); per-step D2D slices + `launch_gla_state_update_output_into` (`:2486`); `wo` projection on-device (`:2520`) |
| Recurrent cache | `lfm2.rs:145-152` `Lfm2LayerCache::Gdl { state: Vec<f32>, dev_state: Option<Box<dyn BackendStorage>> }` | `state` = `[heads*dk*dv]` host mirror; `dev_state` = `[heads,dk,dv]` device-resident, allocated once, updated in place, zero host readbacks; `Clone` (`:188-191`) intentionally drops `dev_state` (step state, not data) |
| Fused graph branch | `gla_graph.rs:27` `gdl_forward_graph()`; `gla_graph.rs:168` `is_gdl_layer()` | Guards that degrade to eager (never wrong math): non-GDL (`:35`), GQA (`:42`), `head_dim != 64` (`:48`), missing `wq/wk/wv/wo` (`:55-67`), missing GDL buffers (`:74`), buffer-dim mismatch (`:79`) |
| Graph allocation | `lfm2_graph.rs:154-156` → `DecodeGraphBuffers::allocate_gdl_layer()` | `crates/grim-backend-rocm/src/decode_graph_buffers.rs:565`; `GdlLayerBuffers` struct (`:117`); **LDS fast path needs `dk==dv==64`** (`:587`); failed GDL alloc fails whole graph → eager fallback |
| Graph dispatch | `lfm2_graph.rs:455-458`; ShortConv vs GDL vs dense branch `:433-460` | `is_gdl_layer` → `gdl_forward_graph`, else `shortconv_forward_graph` / `attn_forward_graph`; GDL layers skip KV arenas (`:204`: `layer.wq.is_none() \|\| attention_mode == Gdl`) |
| Sidecar (no GGUF rewrite) | `gla.rs:54-92` `GraveSidecar { format: "grave-1", arch: "lfm2grave", layer_gates }` | Base GGUF + `*.grave.json`; `save`/`load`; fail-closed (missing sidecar errors, applied line prints) |
| Training math | `gla_train.rs:162` `gdn2_chunkwise_forward` (f64), `:358` f32 variant; `:222` `gdn2_backward`; `gla.rs:184` `gdn2_chunkwise_step`, `:335` chunkwise forward | Chunkwise-WY (Eq. 18–25) bit-comparable to serial oracle (`chunkwise_matches_token_serial_oracle` ≤1e-11); consumed by `grim-cli/src/distill.rs` |
| Env gates | `lfm2.rs:18-27` `GRIM_KV_CACHE_LEN` (default 4096); `:40-49` `GRIM_F16_KV`; `:369+` `GRIM_LFM2_MXFP4_QKV`; `:523+` `GRIM_FUSED_FFN`; `:1007+` `GRIM_ROPE_DEV_BASE`; `lib.rs:22-30` `decode_graph_active()` ← `GRIM_DECODE_GRAPH` opt-out; `GRIM_GRAVE_GATES` override; `GRIM_DEBUG_SHORTCONV`, `GRIM_FORWARD_TRACE`, `GRIM_LFM2_F32_HEAD`, `GRIM_LFM2_KV_ARENA` | Unified gate: ROCm + not-opt-out = graph path; CPU never graphs |

Perf optimizations lfm2.rs uses **beyond** GDL (the actual "1-to-many" payload for non-GDL models):

1. **KV arena (zero-D2H/H2D)**: device arenas `k_dev/v_dev` + `dev_pos` host mirror (`Lfm2LayerCache::Attention`), geometric growth, `copy_slice_into` D2D append, `fused_or_scalar_attention_arena_device` (`shared_attention.rs:279`), host `k/v` mirrors stay empty in arena mode.
2. **F16 KV**: `kv_arena_dtype()` + `GRIM_F16_KV=1` halves attention bytes; alloc-site dtype must match launcher flag.
3. **Fused MXFP4 QKV**: `wqkv_codes/exps + gamma_q/k` (`lfm2.rs:252-254`), built **only** for native-MXFP4 weights on ROCm (`:360-387`); one kernel instead of three GEMVs.
4. **Fused Q8_0 GateUp FFN**: `w_gate_up_q80_fused` (`:260`), built when both gate+up are Q80 on ROCm and `GRIM_FUSED_FFN != 0` (`:523-558`); decode path `fused_gate_up_dot4_decode` (`:1728`).
5. **Device-base RoPE**: single persistent `pos_base_dev` u32 + `past_dev` counter **aliased to the same storage** (`:983-988`); kernel derives `base+step`; eliminates per-layer per-token host `positions[]` build+upload. Gate: `GRIM_ROPE_DEV_BASE != 0 && decode_graph_active`.
6. **Stable graph buffers**: `q_rot_dev/k_rot_dev/attn_out_dev` + `graph_*` intermediates (`:129-141`) allocated once at capture, reused across replays so graph args stay byte-identical.
7. **ShortConv device ring**: `Lfm2LayerCache::ShortConv { host, dev }`, `shortconv_step_device` (`:2011`), `short_conv1d_causal_step_into` / `fused_step_into` (ROCm `device_recurrent.rs:269,325`); decode `b·x + conv + c-gate` on-device; host mirror re-synced (Clone-correct); multi-token prefill invalidates stale `dev_state`.
8. **Charon grouped MoE dispatch**: `charon_cache: CharonCache`, lazy `moe_experts_cache: OnceLock`, `forward_moe_ffn` (`:2174`); one grouped kernel instead of per-expert launches.
9. **Unified decode-graph gate**: `decode_graph_active()` (`lib.rs:22-30`) — the RoPE-seed, attention, and block gates agree; ShortConv gate deliberately separate (opt-in; default-on regressed gfx1201 greedy decode 2026-09-16).
10. **Fail-soft discipline**: device-path failure → eager fallback with `eprintln!` (e.g. `decode_attention_device` P0-2 guard); `Unimplemented` = "backend lacks kernel, host fallback OK"; any other error propagates.
11. **Promoted gfx1200 residual/GateUp fusion**: `grim_dot4_add_rms_norm_gate_up_silu_q80_gemv` now coalesces residual-row stores across all 32 lanes. The path is automatic only for `gfx1200`; `gfx1201` and other targets retain the split path. `GRIM_FUSED_RESIDUAL_GATEUP=0` is the explicit rollback switch. GPU-1 warm raw measurements are 354–357 tok/s versus 328–329 tok/s for the split default.

### 1.2 Audit findings: all transformer model files vs the gold standard

Method: `rg` over `crates/grim-models/transformer/src/*.rs` (158 files) + manual read of all bespoke files.
Headline counts (reproducible via the `python3` snippet in §8):

- **GDL present in exactly 9 files, zero other models**: `gla.rs`, `gla_graph.rs`, `gla_mega.rs`,
  `gla_train.rs`, `lfm2.rs`, `lfm2_graph.rs`, `lib.rs` (re-export), 2× `*.snippet` parity tests.
  `rg -l "gdl|Gdl|GDL|Lfm2AttentionMode|gla::|gdn2"` → 9 files. **No other model file references GDL.**
- **89 thin wrappers** (`Thin wrapper around Llama`): afmoe, apertus, arcee, arctic, baichuan,
  bailingmoe(1,2), bitnet, chatglm, codeshell, cohere2(+moe), deci, deepseek2ocr, dflash, dots1, dream,
  ernie45(+2 moe), eurobert, exaone(4,+moe), gemma3/4(+assistant,+embedding), glm4(+moe), glmdsa, gptneox,
  granite(+moe), grok, grovemoe, hunyuan_dense(+moe), internlm2, jais(1,2), **kimi_linear**,
  **lfm2moe**, llada(+moe), llama4, llama_embed, maincoder, mimo2, minicpm3, **minimax_m2**,
  mistral3/4, mpt, nemotron(+hmoe), olmo(2,+e), openai_moe, openelm, orion, paddle_ocr, pangu_embed,
  phi2/3(+moe), plamo(2,3,+plm), qwen(2,+2moe,3,**3next**), qwen3vl_moe, refact, rnd1, seed_oss,
  smallthinker, smollm2/3, stablelm, starcoder(2), step35, talkie, xverse.
  All delegate 100% to `crate::model::Llama`. They inherit whatever `Llama`/`LlamaBlock` does.
- **49 bespoke (non-wrapper) files**: bailingmoe3, bloom, chameleon, cogvlm, commandr, dbrx, deepseek,
  deepseek2, deepseek32, deepseek4, delta_net_base, diffusion_gemma, dots3_note, eagle3, exaone4_5, falcon,
  falcon_h1, gemma, gemma2, gemma3n, glm4_moe_lite, glm5_2, gpt2, gpt_oss, gptj, granite_moe_hybrid,
  hunyuan_vl, hy_v4, hyv3, inkling_small, interns2_mobius, kimi_k3, laguna, longcat_flash, maple, mellum,
  minicpm, minimax_m3, muse_glimmer, qwen2vl, qwen35(+_perf), qwen35moe, qwen38_flash_next, qwen3moe,
  qwen3vl, solar_open2, t5, wav_tokenizer_dec.
- **Shared-`LlamaBlock` users (≈13)**: `model.rs` itself + chameleon, diffusion_gemma, eagle3, falcon_h1
  (attention leg only), gemma, laguna, maple, mellum, minicpm, muse_glimmer, qwen2, qwen35,
  qwen38_flash_next, solar_open2 (GQA leg). Everyone else runs a bespoke attention loop.
- **Charon (`shared_moe`) users (≈15)**: dbrx, deepseek2/32/4, glm4_moe_lite, glm5_2, granite_moe_hybrid,
  hyv3, kimi_k3, lfm2, minimax_m3, olmoe, qwen2moe (partial), qwen38_flash_next, plus infra
  (`shared_moe.rs`, `moe_block.rs`, `decode_graph.rs`, `lfm2_graph.rs`, `model.rs`).
  Bespoke-MoE files **without** Charon (host routing / per-expert launches): bailingmoe3, gpt_oss,
  qwen3moe, qwen35moe, commandr, deepseek (v1).
- **`DecodeGraph` wired in only 6 files**: `block.rs`, `decode_graph.rs`, `gla_graph.rs`,
  `lfm2_graph.rs`, `lfm2.rs`, `lib.rs`. No bespoke model outside the Llama/LFM2 paths captures a graph.
- **KV-arena coverage is partial**: `block.rs`/`model.rs`/`lfm2.rs` + ~20 bespoke files have
  `k_device/v_device` or `k_dev/v_dev` arenas; ~15 bespoke files (bloom, gpt2, gptj, t5, mpt, commandr,
  exaone4_5, hy_v4, bailingmoe3, …) still do host-mirror + per-token D2H/H2D.
- **RoPE-dev-base / `pos_base/past_dev` exists only in `block.rs` + `lfm2.rs`**. All other bespoke
  attention builds host `positions[]` per step.
- **Fused Q8_0 GateUp exists only in `block.rs` + `lfm2.rs`**. No other FFN fuses.

Per-class verdict (the core "where is GDL needed" answer):

| Class | Members | GDL-capable? | GDL present? | Correct action |
|---|---|---|---|---|
| A. LFM2 native | `lfm2.rs` | Yes (by design) | Yes — gold | Keep; harden, do not redesign |
| B. Pure softmax dense (Llama-family) | 89 wrappers + `model.rs`/`block.rs` + bespoke dense (gemma2, gpt2/j, bloom, mpt, falcon, chameleon, …) | **No** — weights are softmax QK^T; GDL recurrence is a different function | No (correct) | **Do NOT add GDL.** Roll out non-GDL perf (§4/P1): arena, RoPE-dev-base, fused FFN, graph capture |
| C1. True linear-recurrent, GDL-shaped | `delta_net_base.rs` (delta-rule state recurrence, `delta_rule_decode` device path) + `solar_open2.rs` (1×GQA + 3×KDA layers via `DeltaNetBase`) | **Yes — primary GDL expansion target** | No — bespoke delta path, no `gla::*`, no GDL buffers, no sidecar | **Migrate KDA legs onto `gla` oracle + `gdl_forward_graph` + `allocate_gdl_layer`** (§4/P2) |
| C2. True SSM, NOT GDL-shaped | `falcon_h1.rs` (Mamba-2: `ssm_in/out`, conv state, `ssm_state`), `mamba/` crate (mamba2, rwkv) | No — selective-scan ≠ delta-rule; different state `[d_state×d_inner]` vs GDL `[dk×dv]` | No (correct) | **Do NOT convert.** Give Falcon-H1 the non-GDL perf (arena for its GQA leg, RoPE-dev-base, graph) + keep SSM on its own kernels |
| D. MLA / NSA / sparse / absorbed attention | deepseek2/32/4, kimi_k3 (`mla_common`: `extract_kv_b_up_projs`, `mla_absorbed_decode`), longcat_flash, qwen38_flash_next (hybrid), glmdsa (sparse selector), minimax_m3 | No — latent/absorbed/sparse math is not GDN-2 | No (correct) | **Do NOT convert.** Keep MLA/absorbed path; add arena + Charon + graph where missing |
| E. MoE-only variation | dbrx, glm4_moe_lite, glm5_2, granite_moe_hybrid, hyv3, qwen3moe/qwen35moe, gpt_oss, bailingmoe3 | N/A (MoE is orthogonal to attention) | No (correct) | **Do NOT touch attention.** Standardize on Charon D2D dispatch (§4/P1.4) |
| F. Mislabeled "linear/hybrid" wrappers | kimi_linear, qwen3next, minimax_m2, lfm2moe (+ dflash) — all `Thin wrapper around Llama` serving **dense softmax**, no linear kernel anywhere | No (name lies; arch is dense) | No (correct for what they serve) | **Do NOT add GDL.** Fix names/docs/configs (§4/P4); if real linear checkpoints land, model as new bespoke files, not by mutating wrappers |

The two most dangerous misreadings this plan explicitly rejects:
1. *"All 150 models need GDL"* — false. GDL changes the attention function; loading softmax weights
   into a GDL recurrence silently corrupts quality. Only LFM2-native and DeltaNet-shaped KDA legs qualify.
2. *"Thin wrappers already cover hybrids"* — false for C1/D. `kimi_linear`/`qwen3next`/`minimax_m2`/`lfm2moe`
   discard their arch's linear attention and serve dense Llama. Quality-sensitive users of those
   checkpoints need real bespoke blocks (tracked as follow-ups, **not** in this rollout).

### 1.3 Perf inventory that the plan must incorporate (what exists today)

**Inference — prefill**: chunked prefill (`grim-scheduler: chunked_prefill_size=512`, Sarathi-style drain);
continuous batching (`Scheduler::schedule`); radix prefix match/promote (`grim-memory: KvBlockPool::match_prefix*`);
MoE prefill pipeline (`grim-engine/pipelines/moe_prefill_pipeline.rs`); MXFP4 fused QKV prefill kernel (lfm2);
quant WMMA/tiled GEMM (`grim-backend-rocm/kernels/{qkv_attention,wmma_quantized_gemm,quant_tiled_gemm}.rs`).

**Inference — decode**: device KV arenas + `launch_kv_append[_batch]` + `launch_qkv_attention_dev[_batch]`
(`qkv_attention.rs:1717,1767,1819,1855`); `fused_or_scalar_attention_arena[_device]` (`shared_attention.rs:279,385`);
RoPE-dev-base; `past_dev` bump kernels (`launch_bump_i32[_slots]`); fused Q8_0 GateUp decode (`lfm2.rs:1728`);
GDN-2 fused `launch_gla_state_update_output_into` (`gla_launchers`); ShortConv fused step; MLA absorbed decode
(`mla_absorbed_decode`); paged/tree attention (`launch_paged_attention[_quant]`, `launch_tree_attention`);
per-model stable graph buffers; HIP graph capture+replay (`DecodeGraph::{begin_capture,end_capture,replay}`,
`decode_graph_buffers.rs:1016,1054,1094`; `RocmDevice::{begin/end_graph_capture,replay_graph}`); CUDA
`graph_capture.rs` + `has_decode_step_graph`; `decode_graph_enabled()` / `decode_graph_active()` gates;
autotune (`*-backend-*/autotune.rs`, `accel_features::{mfma,wmma}_dispatch`); Scythe publish/write slots.

**Memory / KV**: `KvBlockPool` (alloc/free/pin/spill/demote/promote_to_gpu, layer ranges, device-ordinal pools);
`match_prefix[_with_recurrent,_blending,_promoting]`, `insert_prefix*`, `get/put_ssm_state` (recurrent-aware
prefix cache — GDL/SSM states participate, not just KV); `kv_mirror.rs`, `hidden_state_store.rs`,
`encoder_cache.rs`; F16 KV halves bytes; `moe_budget.rs`; engine `GraphCaptureInputBuffers`.

**MoE**: Charon grouped dispatch (`kernels/charon.rs: grouped_dot4_entry`, `moe_align_block_size`;
`shared_moe::{fused_moe_dispatch, fused_moe_dispatch_from_logits, route_topk}`); `moe_mega_kernel.rs`
+ `validate_mega_kernel_inputs`; `moe_hybrid_exec.rs` (GPU/CPU split, `plan_step`); per-model `CharonCache`
+ `moe_experts_cache: OnceLock`; D2D tri-modal forward (logits-on-device → `*_device_d2d`, logits-host →
`*_device`, else `*_host`).

**Quantization**: `grim-quant` full family — dequant `q80/iq*/q*k/fp4/fp8/mxfp4/nvfp4/mxfp8/wna16/nf4/ostquant/gptq/awq`
+ quant mirrors + `reframe_{mxfp4,nvfp4}` + `rewrite_tensor_data` + `packed_gemm` + `qat_mxfp4` + `gsq/spqr/rco`;
ROCm fused dequant-GEMM toggles (`set_fused_dequant_gemm_enabled`, `set_mxfp4_*`); QAT/soul-eater paths.

**Multi-GPU / collective**: `rccl.rs` / CUDA `nccl.rs`, `multi_gpu_launch.rs`, `p2p_route.rs`, `peer_access.rs`;
`fsdp.rs` (both ROCm + CUDA); `tp_layers.rs`; `plan_kv_head_sharding` (`block.rs:118`);
`TensorParallelConfig` in every `load_tp`.

**Speculative / MTP**: `grim-speculative` (eagle3_drafter, native_mtp, llama_mtp_adapter, mamba_speculative,
markov/uniform heads, confidence heads+scheduler, distill); `native_mtp.rs` + `eagle3.rs` model-side;
engine `speculative_loop.rs` + `speculative.rs`.

**Training / finetune**: `gla_train.rs` chunkwise fwd/bwd (GDL-native training — the only recurrent
training math in-tree); `grim-autograd` (adamw, sophia, galore, lomo, came, oasis, relora, mm_grpo,
preference_trainer/loss, contrast_omni, turbo_finetune, tops_prune/omnilo_prune, soul_eater, scythe(1),
replay, collate, lr_schedule, tape/ops/param/registry/scale); `lora.rs: apply_adapters_to_logits`;
engine `train_packed.rs`; `grim-cli/src/distill.rs` (GRAVE distillation into sidecar + gate projections).

---

## 2. What code needs to change (and how)

### P0 — Guardrails first (Status: COMPLETE)

1. `transformer/src/lib.rs`: **[DONE]** Added `pub fn gdl_eligibility(arch: &str) -> GdlEligibility`
   (`EligibleNative | EligibleKdaMigration | IneligibleSoftmax | IneligibleSsm | IneligibleMlaSparse`)
   + `gdl_eligibility_for_config(&dyn ModelConfig)` and pinned exhaustive arch unit tests (`lib.rs:541-588`).
2. `transformer/tests/roundtrip_lint.rs`: **[DONE]** Extended per-file fetch-budget baseline table to all
   bespoke files; ratchets down host transfers and guards non-test `to_vec_f32()` from regressing.
3. Perf-gate harness: `grim-backend-rocm/src/perf_gate.rs` + `qkv_attention.rs: qkv_arena_fallback_stats()`
   (`shared_attention.rs:364`) — assert `device_attempts > 0 && arena_fallbacks == 0` on ROCm CI for every
   migrated path; sticky-failed configs must be empty.

### P1 — Universal non-GDL perf rollout (Status: PENDING / IN PROGRESS)

Applies to **all Class-B dense + every bespoke file with `arena=0 / ropedev=0 / fusedffn=0`** in the §1.2
matrix. No attention-math change; purely data-movement + launch-count.

*Architectural Consolidation Note*: Standardize bespoke attention loops using `crates/grim-models/transformer/src/attention_dispatcher.rs`
and `shared_attention.rs` / `block.rs::LlamaBlock` where possible, rather than proliferating duplicate kernels.

**P1.1 Device KV arena + `dev_pos` mirror** (pattern: `block.rs:146-168,251-304` + `lfm2.rs:1136-1149`).
Targets (bespoke, `arena=0` today): `bailingmoe3.rs`, `bloom.rs`, `commandr.rs`, `dots3_note.rs`,
`exaone4_5.rs`, `gemma2.rs`, `gpt2.rs`, `gpt_oss.rs`, `gptj.rs`, `hy_v4.rs`, `longcat_flash.rs`,
`maple.rs`, `mellum.rs`, `qwen3moe.rs`, `t5.rs`, `wav_tokenizer_dec.rs`, `bailingmoe3.rs`, `laguna.rs`
(partial), `eagle3.rs`.
How: replace host `Vec<f32>` history + per-step full-rebuild with `k_device/v_device` arenas,
`grow_kv_arena` doubling, `cache_append_kv` D2D (`copy_slice_into`), `fused_or_scalar_attention_arena[_device]`.
Keep the host materialize-fallback (`to_cpu_vec_f32` slice) **only** as the `Unimplemented` branch —
never the default on ROCm (cf. who-dat P1-5 precedent in `lfm2.rs:1120-1124`).

**P1.2 Device-base RoPE** (pattern: `block.rs:158-164,664` + `lfm2.rs:1007-1059`).
Targets: every bespoke file (none outside block/lfm2 has it).
How: add `pos_base_dev: Option<Box<Tensor>> + past_dev: Option<Box<Tensor>> + past_dev_seeded: Option<usize>`
to each bespoke `*LayerCache`; lazily allocate + H2D-seed **outside** the graph bracket; alias
`past_dev` ↔ `pos_base_dev` to the same storage Arc; route decode through `rope_dev_base`; keep host
`positions[]` path for CPU + `GRIM_ROPE_DEV_BASE=0`. Gate through the unified `decode_graph_active()`.

**P1.3 Fused Q8_0 GateUp FFN** (pattern: `lfm2.rs:523-558,1728` + `block.rs:519`).
Targets: all dense FFNs where gate+up are Q80 (check `is_q80` on both; build `FusedGateUpWeights` via
`RocmDevice::build_fused_gate_up_q80`); skip MoE experts (Charon owns those).
How: add `w_gate_up_q80_fused` field per block; build at load (log + fall back on failure, never error);
decode calls the single-dot4 path; prefill keeps the two-GEMV reference for parity.

**P1.3 promotion update (GPU-1 / gfx1200):** The LFM2 Q8_0 decode path is now promoted for
`gfx1200` after a clean-process review. The promoted kernel is
`grim_dot4_add_rms_norm_gate_up_silu_q80_gemv`; its residual store is distributed
across 32 lanes. The default split path remains available through
`GRIM_FUSED_RESIDUAL_GATEUP=0`. `gfx1201` and other targets must not inherit
this promotion without their own measurement.

Clean-process evidence:

- `cargo clean` removed 5.9 GiB and HSACO/JIT caches were cleared.
- First cold-cache fused process: `347 tok/s`.
- Warm fresh-process fused runs: `356, 355, 352 tok/s`.
- Warm default runs: `328, 329, 328 tok/s`.
- Promoted default warm runs: `357, 354, 357 tok/s`.
- Kill-switch run: `324 tok/s`.
- Fused graph suite: `11 passed`; ROCm backend units: `468 passed`.
- Deterministic greedy and stochastic model output matched after excluding the
  hardware calibration diagnostic line.
- `rocprofv3` launch guidance: `193` graph launches for both paths and HIP
  kernel launches reduced from `1,180` to `1,132`.

**P1.3 remaining scope:** Complete the same measured rollout for the remaining
Class-B dense FFNs; do not infer promotion from the LFM2/gfx1200 result.

**P1.4 Charon D2D MoE standardization** (pattern: `qwen38_flash_next.rs:212-277` / `kimi_k3.rs:463-532`).
Targets (bespoke MoE without Charon): `bailingmoe3.rs`, `gpt_oss.rs`, `qwen3moe.rs`, `qwen35moe.rs`,
`commandr.rs`, `deepseek.rs` (v1).
How: add `charon_cache: CharonCache`; implement tri-modal `forward_moe_{device_d2d,device,host}`;
`experts: Vec<MoeExpert>` built from stacked weights once per layer; route via
`fused_moe_dispatch_from_logits` (logits on device) else `fused_moe_dispatch` (logits host) else host.
Do not change router math (softmax vs sqrt-softplus vs sigmoid stays per-arch).

**P1.5 Graph-capture eligibility** (pattern: `lfm2_graph.rs:101-165` + `decode_graph.rs` Llama wrapper).
Targets: `qwen35.rs`, `gemma.rs`, `minicpm.rs`, `qwen38_flash_next.rs` first (block.rs-based, lowest risk);
then remaining block.rs users; bespoke MLA/sparse last (likely ineligible — must prove fallback, not force capture).
How: per-layer `check_layer_topology` + `seed_kv_arena_from_eager` (+ `seed_conv_rings` /
`seed_latent_kv_arena_from_eager` where applicable); GQA-mismatch / non-64 / missing-buffer layers stay eager
by construction. Never bake host scalars into launches (the `past*kv_stride` bug class).

### P0.5 — Model baseline preparation harness (Status: COMPLETE)

A model-agnostic baseline seam is now available before every checkpoint is present:

- **CLI:** `grim-cli baseline --model-dir models [--manifest path] [--run] [--device rocm]`.
- **Discovery:** recursively inventories `.gguf`, `.grim`, `.safetensors`, and `.bin`
  files. A manifest may list future paths; missing entries are reported as
  `pending` without failing the inventory.
- **Execution:** `--run` loads each available checkpoint through the existing
  `grim_engine::model_loader` and runs a fixed prompt-length prefill baseline with
  configurable warmup and measured iterations. Output is JSON with path, model
  architecture, device, status, per-iteration samples, mean forward time, and
  errors. This is intentionally a preparation/measurement surface, not a new
  model-specific code path.
- **Files:** `crates/grim-cli/src/baseline.rs`, `crates/grim-cli/src/main.rs`,
  and the shared device-tensor constructor in `crates/grim-cli/src/run.rs`.
- **Validation:** baseline unit tests pass (`3 passed`); inventory mode found
  all current model files; execution mode passed on the available
  `LFM2.5-230M-Q4_K_M` (`463.7 ms` mean 8-token forward) and
  `LFM2.5-350M-Q4_K_M` (`608.6 ms`); manifest mode reported a missing future
  checkpoint as `pending`. These are prefill forward baselines, not decode
  tok/s and not comparable to the 350 tok/s promotion gate.

### P2 — GDL where needed (Status: PARTIALLY IMPLEMENTED)

**P2.1 `solar_open2.rs` + `delta_net_base.rs` → GDN-2 (EligibleKdaMigration).**
Why these two only: Solar's KDA legs and DeltaNet's `delta_rule_decode` are the sole non-LFM2 recurrences
with `[dk×dv]` delta-rule state shaped like GDL. Everything else is softmax/MLA/SSM (see §1.2 table).
How, file by file:
1. `delta_net_base.rs`: **[INITIAL HOOKS LANDED]** Added `gdl_opt_in: bool` (default false), `use_gdl()`, env gate
   `GRIM_DELTANET_GDL`, and `gdl_gates()` neutral defaults.
   *Remaining*: Route opt-in steps through `gla::gdn2_step` (host) / `forward_gdl_device`-equivalent (device) and
   validate with the existing `delta_numeric_reference_tests` + new `gdl_forward_matches_gla_oracle` test.
2. `solar_open2.rs`: **[PENDING]** Expose `gdl_opt_in` on `SolarOpen2Config` (passed down in `load_tp` instead of
   hardcoded `false`); allocate per-layer `GdlLayerBuffers` in the graph path (`allocate_gdl_layer`, `dk==dv==64`
   enforced, GQA legs keep KV arenas).
3. Distillation & Sidecar: **[PENDING]** extend `grim-cli/src/distill.rs` + `GraveSidecar` to emit per-KDA-leg triples
   (`tc_depth_scaled_gates` for the 3:1 interleave) alongside the existing LFM2 sidecar. Fail-closed:
   missing sidecar = legacy delta path, never silent GDL with default gates on a converted checkpoint
   (quality gate in §6.3).
4. Constraints honored: `hd==64 && nh==nkv` for the fused path; GQA or non-64 KDA legs run the host GDL loop.

**P2.2 lfm2.rs hardening (Status: MOSTLY COMPLETE).**
- **[DONE]** `gate_projections_to_host` and `gdl_b/w/f_proj` probes wired in `lfm2.rs:576-613`.
- Extend `validate()` + `is_gdl_layer` coverage to the MoE+GDL combination (`lfm2moe` follow-up stays
  out of scope — see §5.6).
- Keep `gla_mega.rs:61` (`is_gdl` branch) and `lfm2_graph.rs` GDL allocation as the single graph path;
  no second GDL graph implementation.

### P3 — Training integration (Status: PENDING)

1. GDL training stays in `gla_train.rs` chunkwise fwd/bwd; expose the f32 path (`:358`) to
   `train_packed.rs` for short-sequence distillation; keep f64 as the CI oracle.
2. Autograd optimizers (`adamw/sophia/galore/lomo/came/oasis`) need no per-model change — they operate
   below the model layer. Add one integration test: 2-step × 512-token loop-closure (PPL must not regress;
   precedent: `docs/eval/baseline-ctx-2026-09-19.json:126` untrained-GDL 1840.7 → 1816.8 after finetune).
3. LoRA (`lora.rs`) + `apply_adapters_to_logits` unchanged; verify zero-adapter parity test pattern
   (`lib.rs:360` `smoke_llama_with_empty_adapters_matches_baseline`) still passes for every P1-touched model.

### P4 — Wrapper hygiene + docs (Status: PARTIALLY IMPLEMENTED)

1. **[DONE]** Doc-comment headers added to the five wrappers (`kimi_linear.rs`, `qwen3next.rs`, `minimax_m2.rs`,
   `lfm2moe.rs`, `dflash.rs`) explicitly documenting that they serve dense Llama and do NOT implement linear attention.
2. **[PENDING]** `README.md` (transformer crate) + `docs/`: add the §1.2 class table so future audits converge instantly.

---

## 3. Recommendations (ordered; stop conditions included)

1. **Land P0 first.** Without the eligibility test + fetch-budget lint, P1/P2 will be re-broken by the next
   model-file copy-paste. P0 is ~200 lines, zero risk, unlocks everything.
2. **Do P1 in arena → RoPE → FFN → MoE → graph order.** Each step is independently shippable and
   parity-testable; graph capture last because it amplifies any earlier data-movement bug into a
   capture-time abort (good — but only after the earlier steps are green).
3. **Do P2 behind an opt-in flag per KDA leg, default off.** Quality risk lives entirely here;
   the flag + sidecar-fail-closed rule keeps every existing checkpoint bit-identical until distillation lands.
4. **Prefer `block.rs`/`shared_*` reuse over per-model kernels.** New fused kernels belong in
   `grim-backend-rocm` + `shared_attention.rs`/`shared_moe.rs`, called from thin per-model shims —
   the lfm2→block precedent, not the pre-block bespoke sprawl.
5. **Stop condition for any P1 sub-step**: if `qkv_arena_fallback_stats` shows sticky failures or the
   parity test needs tolerance looser than the file's existing baseline, keep that file eager and file
   a kernel-gap issue instead of widening tolerance. Throughput never outranks correctness.

---

## 4. What should NOT be changed (guardrails — violations are release-blockers)

1. **No GDL for Class B/D/E/F.** Softmax weights into a GDL recurrence, MLA latent into GDN-2 state,
   Mamba-2 `ssm_state` into `[dk×dv]` GDL state, or MoE-router "GDL" are all silent quality corruption.
   The `gdl_eligibility` test (P0.1) enforces this.
2. **No change to the `gla.rs` oracle math** (`gdn2_step`, `gdn2_chunkwise_step`, gate defaults) except
   via the plan-gate test `chunkwise_matches_token_serial_oracle` (≤1e-11) + `gdn2_degenerates_to_vanilla_gdn`.
   Kernels chase the oracle; the oracle never chases kernels.
3. **No weakening of `Unimplemented`-vs-error discipline.** `is_unimplemented()` (`lib.rs:5`) is the only
   license for host fallback. Real errors (`Backend`, `Config`, `Session`) must propagate — never
   `eprintln!`-and-continue except at the documented P0-2 decode sites.
4. **No host scalars baked into graph launches.** `past*kv_stride`, `total=past+steps`, per-token
   `positions[]` inside capture brackets abort HIP capture or worse — bake dead pointers. Device counters
   (`past_dev`/`pos_base_dev`) + stable buffers only.
5. **No `Clone` that aliases device step-state.** `Lfm2LayerCache::Gdl{dev_state:None-on-clone}` and
   `Attention{q/k_rot/attn_out/graph_*:None-on-clone}` are load-bearing (see `lfm2.rs:155-194`).
   Sessions that needMID-step forks must re-seed from host mirrors.
6. **No GGUF rewrite for gates.** Sidecar (`*.grave.json`) only. Checkpoint bytes stay canonical.
7. **No ShortConv default-on outside LFM2.** The gfx1201 greedy-decode regression (bisected 2026-09-16)
   keeps the ShortConv device gate opt-in until its parity cover lands.
8. **`lfm2moe.rs` stays a dense wrapper in this rollout.** A true MoE+GDL hybrid needs router↔gate
   co-design; forcing GDL into the current wrapper (or vice versa) risks both. Tracked follow-up, not P2.

---

## 5. Testing strategy (regressions, bugs, memory overflows, other issues)

### 5.1 Parity (correctness before speed) — every P1/P2 file gets all three

| Level | Test | Tolerance / gate | Precedent |
|---|---|---|---|
| Unit | `gdl_forward_matches_gla_oracle`-style per-block test (norm→attn→residual→FFN vs `gla::gdn2_*`) | f32 vs f64 ≤1e-4 per element; f64 vs f64 ≤1e-11 | `lfm2_gdl_parity_test.snippet`, `lfm2.rs:2990` |
| E2E prefill→decode | CPU-vs-ROCm logits on 8/64/512-token prompts, greedy + sampled | Top-1 agreement 100% greedy; KL < 1e-5 sampled | `lfm2_seed_parity.rs`, `lfm2_prefill_arena_parity.rs`, `transformer_e2e_integration.rs` |
| Graph | capture→replay×N vs eager, incl. GDL + ShortConv + MoE legs | Bit-identical replay; 2nd-capture-succeeds-after-`CAPTURE_POISON` | `lfm2_graph_capture.rs`, `lfm2_graph_p4.rs`, `native_arm_gates_gpu.rs` |

MoE extras: `moe_all_models_parity_gpu.rs`, `moe_special_cases_gpu.rs` must pass for every P1.4 file
(zero-expert, single-expert, all-expert-hot, top-k boundary). ShortConv extras: `lfm2_shortconv_ring.rs`
(device-vs-host step + prefill→decode continuity) for any file touching the conv ring.

### 5.2 Regression (CI must fail loudly)

- **Roundtrip fetch-budget lint** (`tests/roundtrip_lint.rs:42` pattern): per-file D2H/H2D + GEMV-launch
  budget; P1's whole point is fewer launches — assert the count *drops* (arena: 3→1 QKV GEMV; fused FFN:
  2→1; RoPE-dev-base: −1 upload/layer/token). Any increase fails.
- **Perf-gate telemetry**: `qkv_arena_fallback_stats()` — `arena_fallbacks == 0`, no new sticky configs;
  `perf_gate.rs` throughput floor per arch (score path precedent: 997 tok/s @ 378 GB/s effective, GDL mode).
- **Golden-logits files**: store per-model argmax + top-5 logprobs for a fixed prompt set; diff in CI.
  Zero-adapter LoRA parity (`lib.rs:360`) runs for every model (adapters must not perturb the base).

### 5.3 Memory-overflow / OOM strategy

1. **Arena growth tests**: `grow_kv_arena` doubling from 1 → `GRIM_KV_CACHE_LEN` × 2 with
   `capacity_rows`/`tokens`/`row_elems` invariants asserted each doubling; over-capacity returns a typed
   error, never wraps or scribbles (cf. `allocate_gdl_layer: layer >= len` / `zero dim` guards).
2. **GDL state sizing**: `nh*64*64` f32 per layer asserted at cache creation + on `state.len()` mismatch
   resize-path (`lfm2.rs:2557-2570`); device `zeros([nh*64*64])` allocation failure → error, not host fallback
   (state has no host-computed equivalent mid-step).
3. **Graph-buffer residency**: capture test asserts every replay writes to the *same* device pointers
   (stable `q_rot/k_rot/attn_out/graph_*`); replay-after-free or pointer-churn fails.
4. **Long-context soak**: `GRIM_KV_CACHE_LEN=32768` prefill→decode to EOS on one P1 file per class;
   track peak device bytes (telemetry in `spill_telemetry()`); spill/demote path (`KvBlockPool`) exercised,
   not just the happy-resident path. Mamba/SSM states go through `put/get_ssm_state` so prefix-cache
   accounting includes recurrent bytes.
5. **asan/miri-adjacent**: `cargo test -p grim-models-transformer` + `cargo clippy` clean (workspace
   `warnings = deny`); debug-assert the `[dk][dv]` orientations (`gdn2_step` debug_asserts) in CI debug profile.

### 5.4 Bug-class checklist (reviewer must tick per PR)

- [ ] GQA head-repeat correct on host GDL fallback (`kv_group = nh/nkv`, `kv_h = h/kv_group`)?
- [ ] `dk==dv==64` + `nh==nkv` enforced *before* any device launch, with `Unimplemented` fallback?
- [ ] `past_dev` seeded outside capture; first-capture `CAPTURE_POISON` → eager → second capture clean?
- [ ] `dev_pos` mirror updated on every arena append; prefill invalidates stale device rings?
- [ ] `Clone for *LayerCache` drops device step-state (no aliased writes across sessions)?
- [ ] MoE routing math untouched (only dispatch changed)? Router-kind pinned by test?
- [ ] No new `to_vec_f32` / `to_cpu_vec_f32` on the decode hot path (grep the diff)?
- [ ] Env-var polarity documented + tested both spellings (`0/false/off` vs unset/`1`)?

### 5.5 Other issues

- **Determinism**: seed-pinned tests (`lfm2_seed_parity.rs` pattern) for every migrated file; nondeterministic
  kernel (if any) must be opt-in + documented, never the default decode path.
- **Quantization interplay**: Q80-fused FFN + MXFP4-QKV + F16-KV tested in all 4 combinations per file
  (off/off, on/off, off/on, on/on); `kv_arena_dtype` mismatch = hard error, not silent reinterpret.
- **Multi-GPU**: `load_tp` sharding test (`plan_kv_head_sharding`) for P1 files; NCCL/RCCL单测 untouched.
- **Speculative/MTP**: `eagle3.rs` + `native_mtp.rs` draft-verify agreement re-run for P1-touched target models.

---

## 6. Rollout checklist

- [x] P0: `gdl_eligibility` + classification test; fetch-budget lint extended; perf-gate assertions on.
- [ ] P1.1: arenas for the 16 `arena=0` files; per-file parity + budget-drop proven (standardizing on `AttentionDispatcher`).
- [ ] P1.2: RoPE-dev-base for all bespoke decode paths; `GRIM_ROPE_DEV_BASE=0` fallback proven.
- [x] P1.3a: LFM2 Q8_0 gfx1200 residual/GateUp fusion promoted; coalesced residual stores, 11-test graph parity, 468-test ROCm unit suite, clean-process 350+ tok/s gate, and `GRIM_FUSED_RESIDUAL_GATEUP=0` rollback switch verified.
- [ ] P1.3b: extend fused GateUp to remaining Class-B dense Q80 models; 4-combo quant matrix and per-file budget/parity evidence green.
- [x] P0.5: model baseline harness discovers available/pending checkpoints, executes fixed prefill baselines through the normal loader, and reports JSON; small LFM2 Q4 executions and pending-manifest behavior verified.
- [x] Promotion review: default is limited to `gfx1200`; cold-cache startup is recorded separately from warm tok/s; deterministic and stochastic output parity passed after excluding the calibration diagnostic.
- [ ] P1.4: Charon for the 6 host-MoE files; `moe_*_parity_gpu` green.
- [ ] P1.5: graph capture for block.rs-first cohort; replay bit-identical.
- [ ] P2.1: Solar KDA + DeltaNet GDL opt-in + oracle test + sidecar; default-off; legacy path bit-identical (DeltaNet config/gate hooks done; Solar wiring pending).
- [x] P2.2: lfm2 `gdl_b/w/f_proj` load wiring and host probe helpers (`gate_projections_to_host`).
- [ ] P3: distillation loop-closure (PPL non-regression) + optimizer integration test.
- [ ] P4: wrapper doc headers [x] + crate README class table [ ].
- [ ] Final: full `cargo test -p grim-models-transformer` + ROCm device-test suite + long-context soak green.

---

## 7. Appendix — key file:line index (audit trail)

- Gold: `transformer/src/lfm2.rs:51-56,80,145-152,155-194,268-270,297-301,598-601,605-666,668,852-853,1007-1059,1136-1149,1728,2011,2174,2402,2428,2522,2538,2544-2550,2990`
- Oracle/train: `gla.rs:26,38,54-92,98,149,184,335,384` · `gla_train.rs:162,222,358` · `gla_graph.rs:27,35,42,48,55-79,168` · `gla_mega.rs:61`
- Graph: `lfm2_graph.rs:101-165,154-156,204,433-460,455-458,483-594` · `decode_graph_buffers.rs:32,117,155,565,587,654-842,947-1284` · `decode_graph.rs` · `block.rs:22-47,98-168,251-304,308-355,370-519,600-664,1377,1895`
- Shared: `shared_attention.rs:279,364,385` · `kv_attention.rs:150-239` · `shared_moe.rs` · `moe_block.rs` · `mla_common.rs` · `attention_dispatcher.rs` · `model.rs:22,389`
- Candidates: `delta_net_base.rs:17,59,92,166-168,254,379-390` · `solar_open2.rs:2,123,142-166,213,251-271` · `falcon_h1.rs:1,62-87,104-119,342-435` · `qwen38_flash_next.rs:212-277` · `kimi_k3.rs:463-532` · `longcat_flash.rs:156-213` · `glmdsa.rs:70-120` · `granite_moe_hybrid.rs:101-163`
- Wrappers (§1.2 list, 89 files — all `Thin wrapper around Llama` → `crate::model::Llama`)
- Tests: `tests/{lfm2_seed_parity,lfm2_prefill_arena_parity,lfm2_graph_capture,lfm2_graph_p4,lfm2_shortconv_ring,moe_all_models_parity_gpu,moe_special_cases_gpu,transformer_e2e_integration,roundtrip_lint,native_arm_gates_gpu}.rs`
- Backends: `grim-backend-rocm/{device/roc_device.rs:882-1422,device/device_recurrent.rs:269,325,kernels/qkv_attention.rs:9,1222-2043,kernels/charon.rs:2281,2593,kernels/gla_kernels.rs:37,kernels/moe_mega_kernel.rs,device/moe_hybrid_exec.rs,autotune.rs,rccl.rs,fsdp.rs,multi_gpu_launch.rs}` · `grim-backend-cuda/{graph_capture.rs,fsdp.rs,autotune.rs,nccl.rs,kernels/,device_tests/}` · `grim-engine/{engine/,pipelines/,scheduler.rs,speculative_loop.rs,train_packed.rs}` · `grim-scheduler/src/lib.rs:53-634` · `grim-memory/src/lib.rs:162-729` · `grim-quant/src/lib.rs` · `grim-speculative/src/` · `grim-autograd/src/` · `grim-cli/src/distill.rs`
