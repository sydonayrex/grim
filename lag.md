Laguna-S-2.1 hybrid attention + dual/YaRN RoPE + attention gate — implementation plan
Scope: all three items (4+5+6) full, ROCm + CPU backends first, full YaRN magnitude correction, partial rotary on both arms. CUDA/Vulkan/Metal keep Err(Unimplemented) for new params until a later pass.

Spec sources: vLLM laguna.py (attention forward + per-layer rope), llama.cpp laguna.cpp (tensor names + SWA + per-layer RoPE), grim muse_glimmer.rs (in-repo per-layer-rope + sliding-window precedent).

Skills applied: rocm-hip-kernels (Wave64/MFMA/hipRTC/LDS), rust-ffi-grim (FFI safety across trait boundary), clean-code-guard (per-layer registry not tag-branches; CQS; no dead params), rust-ml-llm-architecture (device-resident tensors, backend isolation), caveman (this doc).

Architecture decisions
1. Per-layer attention config via a Vec<LayerAttentionSpec>, not tag-fields on LlamaConfig. Mirrors the existing Vec<Option<MoESpec>> pattern in laguna.rs:119-127 (clean-code-guard #8 — extension via new code, not type-tag branches; #14 — registry over branch-per-caller). Avoids polluting the dense-Llama LlamaConfig with Laguna-only fields.

2. New params on BackendDevice trait are Option-typed with None = full causal. Backends that don't implement them return Err(Unimplemented) (rust-ffi-grim §3 — explicit failure, no silent fallback; clean-code-guard #18 — no fake success). The model layer translates a per-layer spec into the concrete args at each call site.

3. YaRN + partial-rotary ride on the rope trait method via a single RopeConfig struct. Replaces the flat (dim, base) pair. One struct, one signature change, all fields have a "plain RoPE" default. Avoids a 9-arg signature (clean-code-guard #3 — 4-arg ceiling → introduce a config object).

4. Attention gate is model-layer only. No backend/trait change (confirmed: gate is a g_proj → softplus → mul inserted between attention output and o_proj; block.rs:398-399). New ColumnParallelLinear on LlamaBlock loaded from a new GGUF tensor blk.{i}.attn_gate.weight.

5. All new tensor loads guarded. Missing attn_gate/attn_gate_swa on dense layers → gate disabled for that layer, not an error (matches llama.cpp weightless-fixture fallback, laguna.cpp:121).

Phase A — Model plumbing (no kernel work, unblocks B/C)
A1. New LayerAttentionSpec + AttentionType (crates/grim-models/transformer/src/block.rs, new pub types)

rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionType { Full, Sliding }

#[derive(Debug, Clone)]
pub struct LayerAttentionSpec {
    pub attn_type: AttentionType,
    pub num_heads: usize,        // per-layer override (48 vs 72)
    pub num_kv_heads: usize,     // uniform in S-2.1, but carried for generality
    pub rope: RopeConfig,        // per-layer rope (theta, n_rot, yarn)
    pub sliding_window: Option<usize>,  // Some(512) on sliding layers
    pub has_attn_gate: bool,     // load g_proj?
}
A2. Extend LlamaConfigRefs (block.rs:12-28)
Add: sliding_window: Option<usize>. Per-layer head count is already carried as local_num_heads (derived per-layer in A4). This feeds the forward attention call site (block.rs:611) and the CPU fallback (block.rs:680-683).

A3. New RopeConfig (crates/grim-nn/src/modules.rs, alongside Rope at line 829)

rust
#[derive(Debug, Clone)]
pub struct RopeConfig {
    pub dim: usize,              // head_dim
    pub base: f32,               // theta
    pub rotary_dim: usize,       // n_rot <= dim; partial rotary
    pub yarn: Option<YaRNParams>,// None = plain RoPE
}
#[derive(Debug, Clone, Copy)]
pub struct YaRNParams {
    pub factor: f32,
    pub original_max_pos: usize,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub attention_factor: f32,   // mscale
}
Rope::new(dim, base) stays as a convenience constructor → RopeConfig { dim, base, rotary_dim: dim, yarn: None }. Existing callers unchanged.

A4. LlamaBlock::load_tp takes a &LayerAttentionSpec (block.rs:257)
Signature: load_tp(ws, cfg_global, spec: &LayerAttentionSpec, tp). Inside:

Rope::new(cfg.head_dim, cfg.rope_theta) (block.rs:314) → Rope::from_config(&spec.rope).
Q/K/V column-parallel load (block.rs:263-283) uses spec.num_heads/spec.num_kv_heads instead of model-global.
plan_kv_head_sharding (block.rs:316) called with per-layer head counts.
LlamaConfigRefs (block.rs:332) gets sliding_window: spec.sliding_window.
Optional g_proj: Option<ColumnParallelLinear> loaded from ws.pp("attn").pp("gate") when spec.has_attn_gate; shape [hidden, num_heads] (per-head) detected from tensor width (matches llama.cpp laguna.cpp:109-123).
A5. Llama::load_tp + load_tp_moe thread layer index + spec (model.rs:82-194)
Loops (model.rs:92, :153) pass i and &attn_specs[i] into LlamaBlock::load_tp.
New param attn_specs: &[LayerAttentionSpec], length-checked against num_layers (mirrors moe_spec check at model.rs:142).
Dense Llama (non-Laguna) builds a uniform spec vec from the flat LlamaConfig — behavior identical to today.
A6. Laguna::load_tp builds Vec<LayerAttentionSpec> from existing dead fields (laguna.rs:84-135)
The config fields already exist (layer_types, sliding_window, num_attention_heads_per_layer, full_rope_theta, sliding_rope_theta, full_partial_rotary_factor, sliding_partial_rotary_factor) but are dropped. Wire them:

For each layer i: read layer_types[i] → AttentionType; pick rope_theta/partial_rotary_factor by type; build RopeConfig (with YaRN for full layers from parsed rope_parameters); set sliding_window; has_attn_gate = gating == "per-head".
Pass the vec into Llama::load_tp_moe alongside moe_spec.
A7. Attention gate forward (block.rs, between 398 and 399)
After attention output computed, before self.wo.forward:


text
if let Some(g) = &self.g_proj {
    let gate = g.forward(&x_norm)?;           // from PRE-attn hidden (vLLM laguna.py:451)
    let gated = softplus_mul_on_device(&attn_out, &gate, per_head=true)?;
    attn_out = gated;
}
softplus_mul_on_device mirrors existing silu_mul_on_device (block.rs:415) — per-head broadcast (reshape to [S, num_heads, head_dim], mul gate[:, :, None]). Compute softplus in f32 (vLLM does .float()).

A8. Parse rope_parameters from config.json (model_loader.rs)
Drop the hardcoded full_rope_theta: 500000.0 literals (model_loader.rs:520-523). Add rope_parameters: Option<Value> to SafetensorsConfig and parse the nested {full_attention: {rope_type, rope_theta, factor, original_max_position_embeddings, beta_fast, beta_slow, attention_factor, partial_rotary_factor}, sliding_attention: {...}} into LagunaConfig YaRN fields.

A9. GGUF tensor remap (architecture.rs:768-809)
Add: model.layers.{i}.self_attn.g_proj.weight → blk.{i}.attn_gate.weight. Per-layer head counts come from GGUF hparams, not tensors.

Phase A exit criteria: cargo check clean. New unit test: build a Vec<LayerAttentionSpec> matching S-2.1's 48-layer pattern (1 full + 3 sliding × 12, 48/72 heads), assert gate loaded on all layers, assert rope configs differ by layer type. Dense Llama path unchanged (regression guard).

Phase B — Backend RoPE: YaRN + partial rotary (per-layer)
B1. Widen BackendDevice::rope signature (backend.rs:366-379)

rust
fn rope(&self, x, positions, cfg: &RopeConfig, out_shape: &Shape) -> Result<...>
Replaces (dim, base). Default impl returns Err(Unimplemented).

B2. CPU rope (device.rs:297-342)
half = d/2 (device.rs:314) → rotary_half = cfg.rotary_dim / 2.
Rotate only the leading cfg.rotary_dim channels; pass-through the rest unchanged (partial rotary — matches llama.cpp n_rot).
YaRN: when cfg.yarn is Some, compute inv_freq with YaRN frequency ramp between beta_fast/beta_slow boundaries and apply attention_factor mscale. Reference: llama.cpp ggml_rope_ext YaRN path. Implement the standard YaRN formula (range-normalized), validated against a known fixture.
B3. ROCm rope (roc_device.rs:3415, kernel grim_rope at compute_kernels.rs:35)
Add rotary_dim, YaRN params to the HIPRTC kernel args.
Kernel: same pair-rotate loop but bounded by rotary_dim; YaRN branch computes the ramp in-kernel (or precompute inv_freq on host and pass as a constant buffer — preferred, keeps kernel register pressure low per rocm-hip-kernels "register pressure is the #1 occupancy killer").
--offload-arch already matches running GPU (rocm-hip-kernels). Re-cache compiled kernel by (src_hash, arch) — existing module_cache pattern.
Block size multiple of 64 (rocm-hip-kernels Wave64 mandate). Grid linear_launch(b*s*rotary_half).
B4. CUDA/Vulkan/Metal rope
cfg.rotary_dim < dim → Err(Unimplemented) for now (clean-code-guard #18 — explicit failure). rotary_dim == dim && yarn.is_none() → existing path (regression-safe).

Phase B exit criteria: CPU RoPE test — rotary_dim=64, dim=128 rotates first 64 channels, leaves 64-127 unchanged. CPU YaRN test — compare inv_freq ramp against reference formula. ROCm cargo test on gfx1036.

Phase C — Backend attention: sliding window
C1. Widen BackendDevice::qkv_attention + qkv_attention_paged (backend.rs:468, 507)
Add window: Option<usize> param. None = full causal (default behavior).

C2. ROCm HIPRTC kernel (qkv_attention.rs:7-130)
Add int window arg (0 = no window) to grim_qkv_attention signature (qkv_attention.rs:7-22).
Causal bound (qkv_attention.rs:92-94): insert lower bound:

text
const int lo = (window > 0) ? max(0, abs_i - window + 1) : 0;
const int hi = (abs_i < kv_seq_len) ? (abs_i + 1) : kv_seq_len;
const int range_len = hi - lo;
Quarter-stride partition (qkv_attention.rs:97-100) operates on [lo, hi) — j_start += lo, inner dot loop indexes absolute j. One localized edit.
Same for grim_qkv_attention_paged (qkv_attention.rs:201).
Launch arg count 13 → 14 (roc_device.rs:3379-3398). Block stays __launch_bounds__(256) = 4 wavefronts (rocm-hip-kernels).
C3. CPU qkv_attention (device.rs:392-429)
for t2 in 0..kv_seq_len (device.rs:398) → for t2 in window_start..kv_seq_len where window_start = q_abs.saturating_sub(window - 1).
Branch: if t2 > q_abs { -inf } else if t2 < window_start { -inf } else { dot }.
Weighted-V loop (device.rs:423) can skip masked t2 (perf, optional).
Same edit in qkv_attention_paged (device.rs:583+).
C4. Forward call sites (block.rs:611-621, block.rs:652 fallback)
Pass self._cfg.sliding_window as the window arg. CPU fallback (block.rs:680-683) gets the same window-start bound (mirrors muse_glimmer.rs:451-471).

C5. CUDA/Vulkan/Metal qkv_attention
window.is_some() → Err(Unimplemented). window.is_none() → existing causal path.

Phase C exit criteria: CPU attention test — window=4, seq=8, assert scores below q_abs-3 are -inf. ROCm attention test on gfx1036 — sliding vs full output matches CPU reference within fp tolerance. Profile with rocprof-compute (rocm-hip-kernels "measure before claiming fast") — occupancy ≥50%.

Phase D — Wire end-to-end + validate
D1. Laguna::load_tp final wiring
Pass attn_specs (A6) into Llama::load_tp_moe (A5). Update the laguna.rs:8-11 doc comment — remove "not supported" disclaimer, document the hybrid path.

D2. Remove dead stub RouterKind::SigmoidTopKPerHead (grim-nn/src/moe.rs:60-61,130,161)
It was a no-op alias (verified in prior session). The attention gate (A7) is the real feature the "per-head" config names. clean-code-guard #21 — strip dead code.

D3. GGUF path parity (model_loader.rs:1404-1411, the GGUF Laguna arm)
Build the same Vec<LayerAttentionSpec> from GGUF hparams (sliding_window, layer_types array, rope_freq_base_train/_swa, n_rot/n_rot_swa, YaRN keys). Parse LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN for the 1:3 period.

D4. Integration test
Synthetic mini-Laguna checkpoint (8 layers, 1 full + 3 sliding × 2, 4/6 heads, gate tensors present). Load on CPU → forward pass → assert: (a) sliding layers mask out-of-window KV, (b) full layers attend globally, (c) gate multiplies attention output, (d) rotary_dim differs by layer type. Run on ROCm gfx1036 if available → compare to CPU reference within tolerance.

D5. Full-model smoke test against real Laguna-S-2.1 shards
If a checkpoint is available locally, load + generate. Compare logits slice against llama.cpp reference output for a fixed prompt (rust-ml-llm-architecture — "treat model output as untrusted"; validate against reference, not vibes).

Phase D exit criteria: cargo test -p grim-models-transformer green. cargo check across workspace clean. No new warnings in laguna.rs / block.rs / moe.rs (clean-code-guard self-check).

Skill-driven quality gates (applied throughout, not a phase)
rust-ffi-grim §1-3: kernel arg structs passed to hipRTC are #[repr(C)]; null-check every device pointer; cargo check + cargo build + runtime test on missing-ROCm fallback.
rocm-hip-kernels: block size multiple of 64; coalesced loads; LDS double-buffer if bandwidth-bound; cache compiled kernel by (hash, arch); validate with rocprof-compute before claiming speedup.
clean-code-guard: per-layer dispatch via Vec<LayerAttentionSpec> not type-tags (#8); RopeConfig struct holds the 4-arg ceiling (#3); no Option param without a caller (#14); no fake-success returns (#18); strip the dead SigmoidTopKPerHead stub (#21); re-derive attention mask logic from spec, don't copy muse_glimmer blindly (#19).
rust-ml-llm-architecture: tensors device-resident (gate compute stays on-device via softplus_mul_on_device); ROCm kernel bodies in grim-backend-rocm, not in core dispatch.
Risk register
YaRN formula correctness (B2/B3). Highest risk — YaRN's ramp + mscale is easy to get subtly wrong. Mitigation: implement CPU first with a unit test against a published YaRN reference vector; ROCm must match CPU bit-close (within fp16 tolerance). If reference vector unavailable, derive from llama.cpp ggml_rope_ext YaRN branch directly.
Per-layer head count vs TP sharding (A4). plan_kv_head_sharding (block.rs:38-61) assumes uniform heads; 48 and 72 both divide cleanly by TP=1/2/4/8 but the assertion must be per-layer, not global. Validate each num_heads against tp.world_size.
Gate tensor width detection (A4). Width [num_heads] vs [num_heads*head_dim] disambiguation needs the loaded tensor shape, not the config. Follow llama.cpp: read tensor ne[1], branch on exact match to either.
ROCm kernel recompile cache invalidation (B3). New args → new kernel source hash → automatic recompile (existing cache key covers this). Verify no stale .co loaded.
Block.rs refactor blast radius (A4-A7). LlamaBlock is shared by every dense model (Llama, Falcon, Phi, Qwen…). Dense path must build an identity LayerAttentionSpec (Full, full rotary, no gate) → behavior bit-identical. Regression test: load a small Llama model, compare logits pre/post refactor.
Sequencing summary
Phase	Touches	Unblocks	Risk
A (plumbing)	laguna.rs, block.rs, model.rs, modules.rs, model_loader.rs, architecture.rs	B, C, D	Low (pure Rust, testable)
B (RoPE)	backend.rs, device.rs (CPU), roc_device.rs + compute_kernels.rs (ROCm)	D	High (YaRN formula)
C (window)	backend.rs, device.rs (CPU), qkv_attention.rs + roc_device.rs (ROCm)	D	Medium (kernel arg threading)
D (wire + validate)	laguna.rs, model_loader.rs GGUF arm, moe.rs stub removal	—	Medium (integration)
Phases B and C are independent and can be parallelized after A lands. A is the critical path.

Out of scope (explicit)
QK RMSNorm (q_norm/k_norm, vLLM laguna.py:426-427) — not in S-2.1 config, deferred.
Attention sink (swa_attention_sink_enabled) — config sets it false; deferred.
CUDA/Vulkan/Metal sliding-window + YaRN kernels — stubbed Err this pass, full pass later.
Per-element gate variant (gating: "per-element") — implement per-head (S-2.1's value) now; per-element is a shape branch in A7, trivial to add later.
