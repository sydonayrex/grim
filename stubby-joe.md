# stubby-joe.md — implementation plan for 19 thin-wrapper / stub transformer files

## Goal

For each of the 19 files in `crates/grim-models/transformer/src/` that are either thin `Llama` wrappers with wrong/incomplete topology or stubs returning `Unimplemented`, produce a fully working model file: correct config, loader, block(s), forward, session/decode, tests, and merge gates. Each file must load a real checkpoint for its architecture family and produce correct numerics for prefill + decode.

## Scope

19 files:

Wrappers with wrong/incomplete topology (12):
1. chameleon.rs
2. falcon.rs
3. qwen2vl.rs
4. qwen3vl.rs
5. cogvlm.rs
6. gemma3n.rs
7. hunyuan_vl.rs
8. wav_tokenizer_dec.rs
9. deepseek2.rs
10. deepseek32.rs
11. deepseek4.rs
12. qwen35moe.rs (if it is a pure thin wrapper with no MoE topology; audit needed before treating as wrong)

Stubs returning Unimplemented (7):
13. interns2_mobius.rs
14. inkling_small.rs
15. minimax_m3.rs
16. kimi_k3.rs
17. glm5_2.rs
18. diffusion_gemma.rs
19. delta_net_base.rs

## What's already on disk

Configs and weight keys are available locally for **all 19** families in `old/mods/` as of this verification pass. **8 folders have full modeling code on disk** (the model can be written from local artifacts alone without any HF download): chameleon, cogvlm, deepseek2, hunyuanvl, interns2_mobius, kimik3, wav_tokenizer_dec, and **Deepseek4 Flash**. **No folder needs HF for config shape.** The remaining 11 families have config.json + weight-key index on disk but no modeling.py — they need either HF modeling code or reverse-engineering from config+index+README.

Local artifacts by folder (verified on disk as of this pass):

- `old/mods/chameleon/` — config.json + model.safetensors.index.json + tokenizer files + generation_config.json
- `old/mods/cogvlm/` — config.json + model.safetensors.index.json + **modeling_cogvlm.py + configuration_cogvlm.py + visual.py + util.py** (full modeling code on disk)
- `old/mods/deepseek2/` — config.json + model.safetensors.index.json + **modeling_deepseek.py + configuration_deepseek.py + tokenization_deepseek_fast.py** (full modeling code on disk)
- `old/mods/falcon/` — config.json + model.safetensors.index.json + tokenizer files
- `old/mods/gemma3/` — config.json + model.safetensors.index.json + tokenizer files (gemma3 VLM, NOT gemma3n)
- `old/mods/hunyuanvl/` — config.json + pytorch_model.bin.index.json + **modeling_hunyuan.py + configuration_hunyuan.py** + tokenizer files (PyTorch-format index, not safetensors)
- `old/mods/qwen2-vl/` — config.json + model.safetensors.index.json + tokenizer files + vocab.json
- `old/mods/qwen3-vl/` — config.json + model.safetensors.index.json + tokenizer files + vocab.json + video_preprocessor_config.json
- `old/mods/qwen2.5/` — config.json + model.safetensors.index.json + tokenizer files + vocab.json (plain text Qwen2.5, confirms qwen2.rs wrapper is correct)
- `old/mods/wav_tokenizer_dec/` — config.json + **configuration_wavtokenizer.py + modeling_wavtokenizer.py + convert_wavtokenizer.py** + YAML training config (HF WavTokenizer, NOT a transformer — local modeling code on disk; no weight shard files on disk)
- `old/mods/deepseek32/` — config.json + model.safetensors.index.json + tokenizer files (config+index on disk; modeling code NOT on disk — needs HF or reverse-engineer from config+index)
- `old/mods/Deepseek4 Flash/` — config.json + **model.py (961 lines) + kernel.py (536 lines) + convert.py + generate.py + encoding_dsv4.py + jang_config.json** + model.safetensors.index.json (in JANG format, 48 shards) + tokenizer files (full local reference on disk — NOT HF-gated; weight shard files NOT on disk)
- `old/mods/interns2_mobius/` — config.json + model.safetensors.index.json + **modeling_interns2_mobius.py + configuration_interns2_mobius.py + processing_interns2_mobius.py** + tokenizer files + vocab.json (full modeling code on disk)
- `old/mods/inkling/` — config.json + model.safetensors.index.json + tokenizer files (config+index on disk; modeling code NOT on disk — needs HF or reverse-engineer)
- `old/mods/minimax-m3/` — config.json + model.safetensors.index.json + **configuration_minimax_m3_vl.py** + processor + tokenizer files + vocab.json (config+index on disk; modeling code NOT on disk — needs HF)
- `old/mods/kimik3/` — config.json + model.safetensors.index.json + **modeling_kimi_k3.py + modeling_kimi_linear.py + configuration_kimi_k3.py + encoding_k3.py** + tokenizer files (full modeling code on disk)
- `old/mods/glm52/` — config.json + model.safetensors.index.json + tokenizer files + README.md (config+index on disk; modeling code NOT on disk — needs HF)
- `old/mods/diffusiongemma/` — config.json + model.safetensors.index.json + model_index.json + tokenizer files + README.md + scheduler_config.json (config+index on disk; modeling code NOT on disk — needs HF or use model_index.json reference to transformers DiffusionGemmaForBlockDiffusion)
- `old/mods/deltanetbase/` — config.json + model.safetensors.index.json + tokenizer files + tokenizer.model (config+index on disk; modeling code NOT on disk — needs HF or reverse-engineer)

11 families have config+index on disk but no modeling.py: falcon, deepseek32, inkling, minimax-m3, glm52, diffusiongemma, deltanetbase. See Appendix A for the exact HF model_type / repo / config.json URL for each. No folder needs HF for config shape — the config.json for every family is already on disk.

## Architecture facts confirmed from local configs

### chameleon

- model_type: `chameleon`
- `swin_norm: true` — this is the per-head Q/K norm flag
- Per-layer safetensors keys:
  - `model.layers.{i}.self_attn.q_norm.weight` + `.bias`
  - `model.layers.{i}.self_attn.k_norm.weight` + `.bias`
  - `model.layers.{i}.self_attn.q_proj.weight`
  - `model.layers.{i}.self_attn.k_proj.weight`
  - `model.layers.{i}.self_attn.v_proj.weight`
  - `model.layers.{i}.self_attn.o_proj.weight`
  - `model.layers.{i}.input_layernorm.weight`
  - `model.layers.{i}.post_attention_layernorm.weight`
  - `model.layers.{i}.mlp.gate_proj.weight`
  - `model.layers.{i}.mlp.up_proj.weight`
  - `model.layers.{i}.mlp.down_proj.weight`
- Top-level: `model.embed_tokens.weight`, `lm_head.weight`
- Config: `hidden_size: 8192`, `num_attention_heads: 64`, `num_key_value_heads: 8`, `num_hidden_layers: 48`, `intermediate_size: 22016`, `rope_theta: 10000.0`, `rms_norm_eps: 1e-05`, `hidden_act: silu`, `tie_word_embeddings: false`, `vocab_size: 65536`, `max_position_embeddings: 4096`
- Special tokens: `<image>` (8711), `<eoss>` (8196), `<pad>` (1)
- Q/K norm is applied per-head after the Q/K projections, before RoPE. The weight shapes are per-head: `q_norm` is `LayerNorm(head_dim)` applied to each head's Q, same for K.
- This is a text+image multimodal model from the config (has `<image>` token), but the safetensors index shows only transformer weights — no separate vision encoder weights. Chameleon's original architecture uses a VQGAN tokenizer for images, and the image tokens are discrete tokens fed into the transformer. So the multimodal part is at the token level, not a separate ViT projection. A full implementation may need the VQGAN tokenizer or at least the image token handling.

### deepseek2 (DeepSeek V2)

- model_type: `deepseek_v2`
- MLA keys per layer:
  - `model.layers.{i}.self_attn.q_proj.weight` — Q latent projection
  - `model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight` — KV latent projection (with MQA, i.e. single KV head per group)
  - `model.layers.{i}.self_attn.kv_a_layernorm.weight` — layernorm on KV latent
  - `model.layers.{i}.self_attn.kv_b_proj.weight` — KV expansion projection (produces K and V)
  - `model.layers.{i}.self_attn.o_proj.weight`
- Config:
  - `hidden_size: 2048`, `num_attention_heads: 16`, `num_key_value_heads: 16`, `num_hidden_layers: 27`, `intermediate_size: 10944`
  - `kv_lora_rank: 512`, `q_lora_rank: null`, `qk_nope_head_dim: 128`, `qk_rope_head_dim: 64`, `v_head_dim: 128`
  - `rms_norm_eps: 1e-06`, `rope_theta: 10000`
  - `rope_scaling: {type: yarn, factor: 40, original_max_position_embeddings: 4096, beta_fast: 32, beta_slow: 1, mscale: 0.707, mscale_all_dim: 0.707}`
  - `moe_intermediate_size: 1408`, `n_routed_experts: 64`, `n_shared_experts: 2`, `num_experts_per_tok: 6`, `moe_layer_freq: 1`, `first_k_dense_replace: 1`
  - `routed_scaling_factor: 1.0`, `scoring_func: softmax`, `norm_topk_prob: false`
  - `vocab_size: 102400`, `max_position_embeddings: 163840`, `tie_word_embeddings: false`
- MLP keys per layer:
  - `model.layers.{i}.mlp.gate_proj.weight`
  - `model.layers.{i}.mlp.up_proj.weight`
  - `model.layers.{i}.mlp.down_proj.weight`
  - `model.layers.{i}.mlp.shared_experts.gate_proj/up_proj/down_proj.weight` (2 shared experts)
  - `model.layers.{i}.mlp.experts.{e}.gate_proj/up_proj/down_proj.weight` (64 routed experts)
- Top-level: `model.embed_tokens.weight`, `model.norm.weight`, `lm_head.weight`
- Layer norms per layer: `input_layernorm.weight`, `post_attention_layernorm.weight`
- The existing `deepseek.rs` has `DeepSeekBlock` with `q_a_proj`, `q_b_proj`, `kv_a_proj`, `kv_b_proj` — V2 uses different naming and has `kv_a_layernorm` as a separate norm. The existing block is close but not identical; V2 also adds MoE with shared experts and the `first_k_dense_replace` logic (first layer is dense, rest are MoE).
- The existing `deepseek.rs` also does NOT have YaRN. V2 config has YARN with specific parameters. YARN must be implemented or the RoPE scaling must match.
- The existing `deepseek.rs` uses `head_dim: 128` hardcoded and `Rope::new(128, 10000.0)`. V2 uses `qk_nope_head_dim: 128`, `qk_rope_head_dim: 64`, `v_head_dim: 128` — so the Q head is split into nope (128) + rope (64) = 192 per head, and V is 128 per head. The existing block's `head_dim: 128` is wrong for V2.

### falcon

- model_type: `falcon`
- `parallel_attn: true`, `new_decoder_architecture: true`, `multi_query: true`
- Fused QKV: `transformer.h.{i}.self_attention.query_key_value.weight` (one fused projection)
- Attention output: `transformer.h.{i}.self_attention.dense.weight`
- Two norms per layer: `transformer.h.{i}.ln_attn.weight/bias` (shared for attention + MLP input in parallel arch), `transformer.h.{i}.ln_mlp.weight/bias`
- MLP: `transformer.h.{i}.mlp.dense_h_to_4h.weight`, `transformer.h.{i}.mlp.dense_4h_to_h.weight`
- Top-level (need to confirm from full index): `lm_head.weight`, and embedding key (likely `transformer.word_embeddings.weight` or similar — need to read full index)
- Config: `hidden_size: 14848`, `num_attention_heads: 232`, `num_kv_heads: 8`, `num_hidden_layers: 80`, `layer_norm_epsilon: 1e-05`, `vocab_size: 65024`
- Falcon's parallel attention means the attention and MLP run in parallel from the same input, with the output being `x + attn_out + mlp_out`. The `ln_attn` is used for both the attention input and the MLP input (hence only two norms, not three).
- The existing `falcon.rs` wrapper assumes Llama topology (three norms: attn_norm, ffn_norm, plus the residual structure) — this is wrong.

### qwen2-vl

- model_type: `qwen2_5_vl`
- Text config: `hidden_size: 3584`, `num_attention_heads: 28`, `num_key_value_heads: 4`, `num_hidden_layers: 28`, `intermediate_size: 18944`, `hidden_act: silu`, `rms_norm_eps: 1e-06`, `rope_theta: 1000000.0`, `sliding_window: 32768`, `vocab_size: 152064`
- Vision config: `hidden_size: 1280`, `num_heads: 16`, `depth: 32`, `patch_size: 14`, `spatial_patch_size: 14`, `temporal_patch_size: 2`, `in_chans: 3`, `window_size: 112`, `spatial_merge_size: 2`, `intermediate_size: 3420`, `out_hidden_size: 3584`, `fullatt_block_indexes: [7,15,23,31]`, `tokens_per_second: 2`
- Special tokens: `vision_start_token_id: 151652`, `vision_end_token_id: 151653`, `vision_token_id: 151654`, `image_token_id: 151655`, `video_token_id: 151656`
- RoPE: `rope_scaling: {type: mrope, mrope_section: [16,24,24]}`
- Safetensors keys: `model.embed_tokens.weight`, `model.layers.{i}.input_layernorm.weight`, `self_attn.q_proj/k_proj/v_proj.weight`, `self_attn.o_proj.weight`, `post_attention_layernorm.weight`, `mlp.gate_proj/up_proj/down_proj.weight`, `model.norm.weight`, `lm_head.weight`
- The vision encoder is a ViT with window attention, spatial merging, and a projector that maps to `out_hidden_size: 3584` (same as text hidden size). The projector weights are NOT in the safetensors index excerpt we have — need to check if they're stored separately or in the same files. Qwen2-VL typically stores the vision encoder and projector in the same model files.
- Full implementation needs: ViT encoder, projector, vision token insertion, M-RoPE handling.

### qwen3-vl

- model_type: `qwen3_vl`
- Text config: `hidden_size: 4096`, `num_attention_heads: 32`, `num_key_value_heads: 8`, `num_hidden_layers: 36`, `intermediate_size: 12288`, `head_dim: 128`, `hidden_act: silu`, `rms_norm_eps: 1e-06`, `rope_theta: 5000000.0`, `rope_scaling: {mrope_interleaved: true, mrope_section: [24,20,20], rope_type: default}`, `vocab_size: 151936`, `max_position_embeddings: 262144`
- Vision config: `hidden_size: 1152`, `num_heads: 16`, `depth: 27`, `patch_size: 16`, `in_channels: 3`, `intermediate_size: 4304`, `out_hidden_size: 4096`, `spatial_merge_size: 2`, `temporal_patch_size: 2`, `num_position_embeddings: 2304`, `deepstack_visual_indexes: [8,16,24]`
- Special tokens: `image_token_id: 151655`, `video_token_id: 151656`, `vision_start_token_id: 151652`, `vision_end_token_id: 151653`
- Safetensors keys: same naming convention as qwen2-vl (`model.embed_tokens.weight`, `model.layers.{i}...`, `model.norm.weight`, `lm_head.weight`)
- Qwen3-VL uses interleaved M-RoPE (mrope_interleaved: true) with different section sizes than Qwen2-VL. The M-RoPE implementation must handle the interleaved variant.
- `deepstack_visual_indexes` indicates which vision layers produce outputs that are used — Qwen3-VL has a deeper vision stack with multiple output layers.

### hunyuanvl

- model_type: `hunyuan_vl`
- Text config: `hidden_size: 64`, `num_attention_heads: 4`, `num_key_value_heads: 4`, `num_hidden_layers: 2`, `intermediate_size: 128`, `head_dim: 16`, `hidden_act: silu`, `rms_norm_eps: 1e-05`, `rope_parameters: {rope_type: default, rope_theta: 10000.0, mrope_section: [2,2,2,2]}`, `vocab_size: 120818`
- Vision config: `hidden_size: 64`, `num_attention_heads: 4`, `num_key_value_heads: 4`, `num_hidden_layers: 2`, `patch_size: 16`, `num_channels: 3`, `intermediate_size: 128`, `out_hidden_size: 64`, `text_hidden_size: 64`, `max_image_size: 64`, `min_image_size: 64`, `max_vit_seq_len: 16`, `img_max_token_num: 4096`, `spatial_merge_size: 1`, `temporal_patch_size: 1`, `interpolate_mode: bilinear`, `rms_norm_eps: 1e-05`
- Special tokens: `image_token_id: 5`, `im_start_id: 120118`, `im_end_id: 120119`, `im_newline_id: 120121`
- This is a very small model (hidden_size: 64) but has a real multimodal architecture with vision encoder, text encoder, and special image tokens.
- No safetensors index on disk — weights not available locally.

### gemma3 (NOT gemma3n — but informative for gemma family)

- model_type: `gemma3` (this is gemma3 VLM, not gemma3n)
- Text config: `hidden_size: 3840`, `num_attention_heads: 16`, `num_key_value_heads: 8`, `num_hidden_layers: 48`, `intermediate_size: 15360`, `head_dim: 256`, `hidden_activation: gelu_pytorch_tanh` (GeGLU), `rms_norm_eps: 1e-06`, `rope_theta: 1000000.0`, `rope_scaling: {factor: 8.0, rope_type: linear}`, `sliding_window: 1024`, `layer_types: [sliding_attention x ... , full_attention x ...]`, `use_bidirectional_attention: false`, `query_pre_attn_scalar: 256`, `vocab_size: 262208`
- Vision config: `model_type: siglip_vision_model`, `hidden_size: 1152`, `num_attention_heads: 16`, `num_hidden_layers: 27`, `patch_size: 14`, `image_size: 896`, `num_channels: 3`, `intermediate_size: 4304`, `vision_use_head: false`
- Special tokens: `boi_token_index: 255999`, `eoi_token_index: 262144`, `image_token_index: 262144`, `mm_tokens_per_image: 256`, `eos_token_id: [1, 106]`
- Safetensors keys: `language_model.model.embed_tokens.weight`, `language_model.model.layers.{i}.input_layernorm.weight`, `self_attn.k_norm.weight`, `self_attn.q_norm.weight` (wait — gemma3 has Q/K norms?), `self_attn.q_proj/k_proj/v_proj/o_proj.weight`, `post_attention_layernorm.weight`, `pre_feedforward_layernorm.weight`, `post_feedforward_layernorm.weight`, `mlp.down_proj/gate_proj/up_proj.weight`, `language_model.model.norm.weight`, `lm_head.weight`
- Gemma3 has BOTH `pre_feedforward_layernorm` AND `post_feedforward_layernorm` AND `post_attention_layernorm` — three norms per layer, not two. The existing `gemma.rs` has `attn_norm` + `ffn_norm` (two norms). This is a topology difference.
- Wait — let me re-check. The safetensors index shows `post_attention_layernorm.weight`, `pre_feedforward_layernorm.weight`, `post_feedforward_layernorm.weight`. That's three norms. But the existing gemma.rs has `attn_norm` and `ffn_norm`. This means gemma3 is a different architecture from the gemma family we have in the crate.
- Actually, gemma3 is a VLM with SigLIP vision, GeGLU, sliding attention, and three norms. The existing `gemma.rs` is for the original Gemma (gemma-7b, gemma-2b) which has two norms and full attention. Gemma3 is a different architecture.
- For gemma3n.rs: we don't have a config. Gemma3n is the on-device version. Need HF download.

## What each file needs

### 1. chameleon.rs — WRONG, needs real implementation

**Current state:** Thin `Llama` wrapper. Llama has no per-head Q/K norms.

**What's needed:**
- Config: `ChameleonConfig` with `swin_norm: bool` field (already in the current config struct)
- Block: `ChameleonBlock` (or extend `LlamaBlock`) with optional `q_norm: Option<RmsNorm>` and `k_norm: Option<RmsNorm>` per head
- Forward: after Q/K projections, reshape to (S, num_heads, head_dim), apply `q_norm` to Q and `k_norm` to K per head, reshape back, then RoPE, then attention
- Load: map `self_attn.q_norm.weight/bias`, `k_norm.weight/bias` from weights
- The Q/K norms are per-head LayerNorms with `head_dim` size. They need to be applied per head, which means the forward path must reshape Q and K to (S, num_heads, head_dim), apply norm, reshape back to (S, num_heads * head_dim).

**Loading mapping:**
- `model.layers.{i}.self_attn.q_proj.weight` → `wq`
- `model.layers.{i}.self_attn.k_proj.weight` → `wk`
- `model.layers.{i}.self_attn.v_proj.weight` → `wv`
- `model.layers.{i}.self_attn.o_proj.weight` → `wo`
- `model.layers.{i}.self_attn.q_norm.weight/bias` → `q_norm`
- `model.layers.{i}.self_attn.k_norm.weight/bias` → `k_norm`
- `model.layers.{i}.input_layernorm.weight` → `attn_norm`
- `model.layers.{i}.post_attention_layernorm.weight` → `ffn_norm`
- `model.layers.{i}.mlp.gate_proj/up_proj/down_proj.weight` → `w_gate/w_up/w_down`
- `model.embed_tokens.weight` → `tok_embeddings`
- `lm_head.weight` → `output`

**Open questions:**
- Is Chameleon multimodal in this implementation? The safetensors index shows only transformer weights, no vision encoder. Chameleon uses VQGAN for images, so the image tokens are discrete tokens. A full implementation might need the VQGAN tokenizer, or at minimum the image token handling. For a text-only load, the transformer weights alone work, but the model is not "Chameleon" without the image token handling.
- What is the `swin_norm` exact mechanism? The HF Chameleon implementation applies LayerNorm to each head's Q and K. Need to verify the exact per-head norm application (is it per-head RMSNorm or LayerNorm?).

**Definition of done:**
- Load a Chameleon checkpoint (the one in `old/mods/chameleon/` if weights are available, or download from HF)
- Prefill + decode produce correct numerics (need a reference — e.g. HuggingFace transformers Chameleon implementation, or the modeling_chameleon.py from the HF repo)
- Tests: smoke test with random weights, config parse test

### 2. falcon.rs — WRONG, needs real implementation

**Current state:** Thin `Llama` wrapper. Falcon uses fused QKV + parallel attention + two norms.

**What's needed:**
- Config: `FalconConfig` with `parallel_attn: bool`, `new_decoder_architecture: bool`, `multi_query: bool` fields
- Block: `FalconBlock` with fused QKV projection (`query_key_value`), `dense` output, `ln_attn`, `ln_mlp`
- Forward: parallel attention + MLP from same input:
  ```
  attn_normed = ln_attn(x)
  mlp_normed = ln_mlp(x)
  qkv = query_key_value(attn_normed)  # fused projection
  attn_out = attention(qkv) → dense
  mlp_out = mlp(mlp_normed)  # dense_h_to_4h → act → dense_4h_to_h
  output = x + attn_out + mlp_out
  ```
- The fused QKV projection produces Q, K, V in one matrix multiply. The output is split into Q, K, V after the projection.
- Note: Falcon's `new_decoder_architecture` uses `gelu` activation for the MLP (not SwiGLU). The config doesn't specify the activation explicitly, but the original Falcon uses `gelu` (not SwiGLU). Need to verify from the modeling_falcon.py.

**Loading mapping:**
- `transformer.h.{i}.self_attention.query_key_value.weight` → fused QKV projection
- `transformer.h.{i}.self_attention.dense.weight` → `wo`
- `transformer.h.{i}.ln_attn.weight/bias` → `ln_attn`
- `transformer.h.{i}.ln_mlp.weight/bias` → `ln_mlp`
- `transformer.h.{i}.mlp.dense_h_to_4h.weight` → `w_up` (gate is not separate in Falcon — it's just dense_h_to_4h → gelu → dense_4h_to_h)
- `transformer.h.{i}.mlp.dense_4h_to_h.weight` → `w_down`
- Top-level: need to read full index for embedding + lm_head keys

**Open questions:**
- What is the exact embedding key name? Need full safetensors index.
- Does Falcon use `gelu` or `silu`? The config doesn't specify `hidden_act`. The original Falcon uses `gelu`. Need to verify from modeling_falcon.py.
- Falcon's MLP is `dense_h_to_4h → gelu → dense_4h_to_h` (no separate gate/up). This is different from Llama's SwiGLU (`gate_proj` + `up_proj` → silu → `down_proj`).

**Definition of done:**
- Load a Falcon checkpoint (need to download from HF — the `old/mods/falcon/` folder has config + index but need weights)
- Prefill + decode produce correct numerics vs HuggingFace transformers Falcon implementation
- Tests: smoke test, config parse test

### 3. qwen2vl.rs — WRONG, needs real VL implementation

**Current state:** Thin `Llama` wrapper. Qwen2-VL is a VLM with ViT + projector + vision tokens.

**What's needed:**
- Config: `Qwen2VlConfig` with vision config nested (`Qwen2VlVisionConfig`), special token IDs, M-RoPE config
- Two sub-modules:
  a. **Vision encoder**: ViT with window attention, spatial merging, depth 32, hidden_size 1280, patch_size 14, out_hidden_size 3584
  b. **Projector**: linear layer mapping vision hidden (3584) to text hidden (3584) — actually `out_hidden_size: 3584` already matches text hidden, so the projector may be a simple linear or may not be needed if the vision encoder output already matches
  c. **Text transformer**: Llama-style text backbone (this part can use the existing Llama block)
- Forward: process image through ViT → projector → insert vision tokens into text sequence → run text transformer
- The vision tokens are inserted at the `image_token_id` / `vision_token_id` positions in the text sequence.
- M-RoPE: `rope_scaling: {type: mrope, mrope_section: [16,24,24]}` — RoPE is applied with different frequencies for different dimensions (temporal, height, width). The existing `Rope` implementation needs to support M-RoPE or a new `MRope` type is needed.

**Loading mapping:**
- ViT weights: `model.vision_model.*` (need to check exact naming from safetensors index — the excerpt we have only shows text transformer keys)
- Projector weights: `model.visual projector.*` or similar
- Text transformer weights: `model.embed_tokens.weight`, `model.layers.{i}.*`, `model.norm.weight`, `lm_head.weight`

**Open questions:**
- What is the exact ViT weight naming in the safetensors files? Need to read the full index.
- Is there a separate projector, or does the vision encoder output already match the text hidden size? The config says `out_hidden_size: 3584` which equals `hidden_size: 3584`, so the projector might be a no-op or a simple linear.
- How are vision tokens inserted into the text sequence? Qwen2-VL uses a specific pattern: image tokens are inserted at specific positions, and the ViT output is projected and placed at those positions.

**Definition of done:**
- Load a Qwen2-VL checkpoint (need weights — `old/mods/qwen2-vl/` has config + index but need weights)
- Image + text forward produces correct vision token embeddings
- Text-only forward matches the existing Llama wrapper (backward compatibility)
- Tests: smoke test with random ViT + text weights, config parse test

### 4. qwen3vl.rs — WRONG, needs real VL implementation

**Current state:** Thin `Llama` wrapper. Same issues as qwen2vl.

**What's needed:**
- Config: `Qwen3VlConfig` with nested vision config, special token IDs, interleaved M-RoPE
- Vision encoder: ViT with depth 27, hidden_size 1152, patch_size 16, out_hidden_size 4096, deepstack visual indexes
- Projector: maps vision output (4096) to text hidden (4096) — same size, may be simple linear
- Text transformer: Llama-style backbone
- Interleaved M-RoPE: `rope_scaling: {mrope_interleaved: true, mrope_section: [24,20,20]}` — the interleaved variant is different from Qwen2-VL's non-interleaved M-RoPE
- `deepstack_visual_indexes: [8,16,24]` — Qwen3-VL uses outputs from multiple vision layers, not just the final layer

**Loading mapping:**
- Same naming convention as qwen2-vl: `model.embed_tokens.weight`, `model.layers.{i}.*`, `model.norm.weight`, `lm_head.weight`
- ViT weights: need to check exact naming

**Open questions:**
- What is the exact ViT weight naming?
- How does `deepstack_visual_indexes` work? Does the model use outputs from layers 8, 16, 24 of the ViT and concatenate/sum them?
- Interleaved vs non-interleaved M-RoPE: the implementation must handle both.

**Definition of done:**
- Load a Qwen3-VL checkpoint (need weights)
- Image + text forward produces correct vision token embeddings
- Text-only forward matches Llama wrapper
- Tests: smoke test, config parse test

### 5. cogvlm.rs — WRONG, needs real implementation

**Current state:** Thin `Llama` wrapper. CogVLM has a visual expert (QFormer) that is injected into the attention layers.

**What's needed:**
- Config: `CogVlmConfig` with vision config, QFormer config, visual expert injection points
- CogVLM architecture:
  - A vision transformer (ViT) that extracts visual features
  - A QFormer (query-based transformer) that compresses visual features into a set of query vectors
  - The QFormer output is concatenated with text embeddings at specific layers (the "visual expert" injection)
  - The visual expert is an additional MLP that runs in parallel with the FFN at injection layers
- Full implementation needs: ViT, QFormer, visual expert injection, modified forward path

**Open questions:**
- What is the exact CogVLM version? CogVLM, CogVLM2, CogVLM2-Chat, etc. have different architectures.
- What are the injection layer indices?
- What is the QFormer query count?
- Need HF download for config + weights.

**Definition of done:**
- Load a CogVLM checkpoint
- Image + text forward produces correct visual expert embeddings
- Tests: smoke test, config parse test

### 6. gemma3n.rs — WRONG (likely), needs config + implementation

**Current state:** Thin `Llama` wrapper. Gemma3n is a different model_type from gemma3.

**What's needed:**
- HF download for gemma3n config.json (model_type: `gemma3n` or similar)
- Gemma3n architecture: need to understand from config + modeling_gemma3n.py
- The existing `gemma.rs` has GeGLU + two norms. If gemma3n uses the same architecture as gemma3 (GeGLU, three norms, sliding attention, SigLIP vision), then a thin wrapper is wrong.
- If gemma3n is a text-only model with the same architecture as gemma, then the existing gemma.rs could be reused (but with a different config name).

**Open questions:**
- What is gemma3n's exact architecture? Need HF config.
- Is gemma3n multimodal (like gemma3) or text-only?
- Does gemma3n use GeGLU, three norms, sliding attention?

**Definition of done:**
- After HF download: implement correct topology based on config
- Load a gemma3n checkpoint
- Prefill + decode produce correct numerics
- Tests: smoke test, config parse test

### 7. hunyuan_vl.rs — WRONG, needs real VL implementation

**Current state:** Thin `Llama` wrapper. HunYuan-VL is a VLM with vision encoder + text encoder + special image tokens.

**What's needed:**
- Config: `HunyuanVlConfig` with nested text_config, vision_config, special token IDs
- Two encoders:
  a. **Vision encoder**: ViT with hidden_size 64, num_heads 4, num_layers 2, patch_size 16, out_hidden_size 64, max_image_size 64, min_image_size 64, max_vit_seq_len 16, spatial_merge_size 1, temporal_patch_size 1
  b. **Text encoder**: Llama-style backbone with hidden_size 64, num_heads 4, num_kv_heads 4, num_layers 2, intermediate_size 128, head_dim 16
- Special tokens: `im_start_id: 120118`, `im_end_id: 120119`, `im_newline_id: 120121`, `image_token_id: 5`
- M-RoPE: `rope_parameters: {mrope_section: [2,2,2,2]}`
- Forward: process image through ViT → insert vision tokens at `im_start`/`im_end` positions → run text encoder

**Open questions:**
- No safetensors index on disk. Need HF download for weights + exact weight naming.
- How are vision tokens inserted? HunYuan-VL uses a specific format: `<im_start> image tokens <im_end>` followed by text.

**Definition of done:**
- Load a HunYuan-VL checkpoint (need HF download)
- Image + text forward produces correct vision token embeddings
- Text-only forward matches Llama wrapper
- Tests: smoke test, config parse test

### 8. wav_tokenizer_dec.rs — WRONG, needs real Vocos/ISTFT decoder implementation

**Current state:** Thin `Llama` wrapper. WavTokenizer is an acoustic codec model, not a text LLM.

**Local artifacts on disk:** `old/mods/wav_tokenizer_dec/` has `config.json`, `configuration_wavtokenizer.py`, `modeling_wavtokenizer.py`, `convert_wavtokenizer.py`, and a YAML training config.

**Config from disk (`config.json`):**
```json
{
  "model_type": "wavtokenizer",
  "sample_rate": 24000,
  "n_fft": 1280,
  "hop_length": 320,
  "n_mels": 128,
  "feature_dim": 512,
  "encoder_dim": 32,
  "encoder_rates": [2, 4, 5, 8],
  "latent_dim": 512,
  "codebook_size": 4096,
  "codebook_dim": 512,
  "num_quantizers": 1,
  "backbone_type": "vocos",
  "backbone_dim": 768,
  "backbone_num_blocks": 12,
  "backbone_intermediate_dim": 2304,
  "backbone_kernel_size": 7,
  "head_type": "istft",
  "head_dim": 641,
  "use_attention": false,
  "attention_dim": 768,
  "attention_heads": 8,
  "attention_layers": 0
}
```

**Config from `configuration_wavtokenizer.py` (Python-side defaults, differ from local config.json):**
- `backbone_num_blocks` default: 8 (local config has 12)
- `backbone_dim` default: 512 (local config has 768)
- `backbone_intermediate_dim` default: 1536 (local config has 2304)
- `head_dim` default: 1025 = n_fft/2+1 (local config has 641)
- `attention_layers` default: 1 (local config has 0 — no attention in this variant)
- `use_attention` default: True (local config has False)
- `encoder_rates` default: [8,5,4,2] (local config has [2,4,5,8] — different order!)

**Architecture from `modeling_wavtokenizer.py` (the real decoder topology):**

WavTokenizer = FeatureExtractor + Backbone + Head, NOT a transformer.

1. **FeatureExtractor** (`feature_extractor.encodec.*`):
   - EncoderModel: Conv1d stack with downsampling ratios [2,4,5,8], LSTM, output conv
   - Quantizer: VQ with codebook (embed, cluster_size, embed_avg buffers)
   - This is the ENCODER side (audio -> codes), not the decoder

2. **Backbone** (`backbone.*`) — the DECODER:
   - `embed`: Conv1d(input_dim=512, dim=768, kernel_size=7, padding=3)
   - `norm`: AdaLayerNorm(dim=768, num_bandwidths=4) — bandwidth-conditioned adaptive layernorm
   - `convnext`: ModuleList of 12 ConvNeXtBlock(dim=768, intermediate_dim=2304, kernel_size=7, num_bandwidths=4)
   - `final_layer_norm`: LayerNorm(768)
   - Each ConvNeXtBlock: dwconv (Conv1d 768->768, kernel=7, groups=768) + AdaLayerNorm + pwconv1 (Linear 768->2304) + GELU + pwconv2 (Linear 2304->768) + gamma (scale) + residual

3. **Head** (`head.*`) — iSTFT:
   - `out`: Linear(768, n_fft+2=1282) — outputs magnitude+phase
   - `istft.window`: registered buffer, Hann window of size n_fft=1280
   - Forward: backbone output (B, C, T) -> transpose -> Linear -> split mag/phase -> complex STFT -> istft -> audio (B, 1, samples)

**Forward contract (decode path):**
- Input: discrete codes (B, T) or quantized features (B, D, T')
- `codes_to_features`: VQ codebook lookup (embed table) -> (B, D, T)
- Backbone: embed -> AdaLayerNorm -> 12 ConvNeXt blocks -> final LayerNorm -> (B, 768, T)
- Head: Linear(768->1282) -> mag/phase split -> iSTFT -> audio waveform
- NO transformer, NO attention (use_attention=false in this config), NO RoPE, NO vocab

**Weight keys (from modeling_wavtokenizer.py checkpoint structure):**
```
# Feature extractor (encoder — may or may not be needed for decode-only)
feature_extractor.encodec.encoder.model.*     # encoder conv stack
feature_extractor.encodec.quantizer.vq.layers.0._codebook.embed       # codebook (4096, 512)
feature_extractor.encodec.quantizer.vq.layers.0._codebook.inited      # buffer
feature_extractor.encodec.quantizer.vq.layers.0._codebook.cluster_size # buffer
feature_extractor.encodec.quantizer.vq.layers.0._codebook.embed_avg   # buffer

# Backbone (THE DECODER — this is what "wav_tokenizer_dec" should implement)
backbone.embed.weight                         # (768, 512, 7)
backbone.embed.bias                           # (768)
backbone.norm.scale.weight                    # (4, 768) — AdaLayerNorm scale
backbone.norm.shift.weight                    # (4, 768) — AdaLayerNorm shift
backbone.convnext.0.dwconv.weight            # (768, 1, 7)
backbone.convnext.0.dwconv.bias              # (768)
backbone.convnext.0.norm.scale.weight         # (4, 768)
backbone.convnext.0.norm.shift.weight         # (4, 768)
backbone.convnext.0.pwconv1.weight            # (2304, 768)
backbone.convnext.0.pwconv1.bias              # (2304)
backbone.convnext.0.pwconv2.weight            # (768, 2304)
backbone.convnext.0.pwconv2.bias              # (768)
backbone.convnext.0.gamma                    # (768)
... (convnext.1 through convnext.11, same pattern)
backbone.final_layer_norm.weight              # (768, 768)
backbone.final_layer_norm.bias                # (768)

# Head (iSTFT)
head.out.weight                               # (1282, 768)
head.out.bias                                 # (1282)
head.istft.window                             # (1280,) — Hann window, registered buffer
```

**Critical discrepancies between local config.json and modeling_wavtokenizer.py defaults:**
1. **encoder_rates order**: config.json says [2,4,5,8], Python defaults say [8,5,4,2]. The encoder downsampling order matters for weight shape matching.
2. **backbone_num_blocks**: config.json says 12, Python default says 8. This model has 12 ConvNeXt blocks.
3. **backbone_dim**: config.json says 768, Python default says 512.
4. **backbone_intermediate_dim**: config.json says 2304, Python default says 1536.
5. **head_dim**: config.json says 641 (=1280/2+1), Python default says 1025 (=2048/2+1). Different n_fft!
6. **use_attention**: config.json says false, Python default says true. This variant has NO attention layers.
7. **attention_layers**: config.json says 0, Python default says 1.

The local config.json is the ground truth for this specific checkpoint variant. The Python defaults are for a different variant.

**Discrepancy with llama.cpp `wavtokenizer-dec.cpp`:**
llama.cpp's wavtokenizer-dec.cpp implements a DIFFERENT architecture:
- POSNET (6-layer residual net with attention at layer 2) + ConvNeXt (depthwise separable conv blocks)
- No AdaLayerNorm, no ConvNeXtBlock with pwconv1/pwconv2/GELU, no iSTFT head
- This is a llama.cpp-specific reimplementation, NOT the HuggingFace WavTokenizer

The HF WavTokenizer (from `modeling_wavtokenizer.py`) uses:
- Vocos backbone (ConvNeXt blocks with AdaLayerNorm + GELU FFN)
- iSTFT head (Linear -> mag/phase -> complex STFT -> istft)

These are two different decoder architectures for "WavTokenizer." The HF version is the one matching `config.json` + `modeling_wavtokenizer.py` on disk.

**What the current thin wrapper gets wrong:**
- Uses LlamaConfig with vocab_size=4096, hidden_size, num_heads, num_layers — treating it as a tokenizer LM
- Has RoPE, RMSNorm, GQA attention, SwiGLU — none of which exist in WavTokenizer
- The "vocab_size: 4096" is actually codebook_size (the VQ codebook), not a token vocabulary
- The 4096 "vocab" is the discrete codebook indices, decoded via codebook.embed lookup, not an LM head

**What a correct implementation needs:**
1. **WavTokenizerDecConfig** — replace the Llama-style fields with:
   - `backbone_dim: 768`
   - `backbone_num_blocks: 12`
   - `backbone_intermediate_dim: 2304`
   - `backbone_kernel_size: 7`
   - `latent_dim: 512` (input dim to backbone)
   - `n_fft: 1280`
   - `hop_length: 320`
   - `head_dim: 641` (= n_fft/2 + 1)
   - `codebook_size: 4096`
   - `codebook_dim: 512`
   - `num_bandwidths: 4` (for AdaLayerNorm)
   - Audio params: sample_rate, n_fft, hop_length, n_mels (for metadata/stats only)

2. **AdaLayerNorm** — bandwidth-conditioned adaptive layernorm:
   - scale: Embedding(4, 768), shift: Embedding(4, 768)
   - Forward: standard LayerNorm, then x * scale.unsqueeze(-1) + shift.unsqueeze(-1)
   - Takes bandwidth_id (which bandwidth condition, 0-3) — for decode, typically bandwidth_id=0

3. **ConvNeXtBlock** (x12):
   - dwconv: Conv1d(dim, dim, kernel_size=7, groups=dim, padding=3) — depthwise
   - norm: AdaLayerNorm(dim, num_bandwidths)
   - pwconv1: Linear(dim, intermediate_dim) — 768 -> 2304
   - GELU activation
   - pwconv2: Linear(intermediate_dim, dim) — 2304 -> 768
   - gamma: Parameter(dim) — layer scale, initialized to 1e-6
   - Forward: x -> dwconv -> norm(bandwidth_id) -> transpose -> pwconv1 -> GELU -> pwconv2 -> transpose -> gamma* -> residual + x

4. **Backbone**:
   - embed: Conv1d(512, 768, kernel_size=7, padding=3)
   - norm: AdaLayerNorm(768, 4)
   - convnext: [ConvNeXtBlock x 12]
   - final_layer_norm: LayerNorm(768)
   - Forward: x (B, 512, T) -> embed -> AdaLayerNorm -> 12 blocks -> transpose -> LayerNorm -> transpose -> (B, 768, T)

5. **ISTFTHead**:
   - out: Linear(768, 1282) — 1282 = n_fft + 2 = 1280 + 2 (mag bins + phase)
   - istft_window: registered buffer, Hann window (1280,)
   - Forward: (B, 768, T) -> transpose -> Linear -> (B, T, 1282) -> split: mag=(B,T,641), phase=(B,T,641) -> exp(mag) * cos/sin(phase) -> complex STFT (B, 641, T) -> istft -> audio (B, 1, samples)

6. **Codebook** (for codes_to_features):
   - embed: (4096, 512) — the VQ codebook embedding table
   - Forward: codes (B, T) -> embedding lookup -> (B, 512, T)

7. **Full decode path**:
   - Input: codes (B, T) [discrete VQ indices]
   - codes_to_features(codes) -> (B, 512, T)
   - backbone(features) -> (B, 768, T)
   - head(backbone_out) -> audio (B, 1, samples)
   - Output: audio waveform

**Loading mapping (from modeling_wavtokenizer.py checkpoint keys):**
```rust
// Backbone
backbone.embed.weight        -> Conv1d(512, 768, 7) weight
backbone.embed.bias          -> Conv1d bias
backbone.norm.scale.weight   -> AdaLayerNorm scale embedding (4, 768)
backbone.norm.shift.weight   -> AdaLayerNorm shift embedding (4, 768)
for i in 0..12:
    backbone.convnext.{i}.dwconv.weight      -> Conv1d(768, 768, 7, groups=768)
    backbone.convnext.{i}.dwconv.bias        -> bias
    backbone.convnext.{i}.norm.scale.weight  -> AdaLayerNorm scale
    backbone.convnext.{i}.norm.shift.weight  -> AdaLayerNorm shift
    backbone.convnext.{i}.pwconv1.weight     -> Linear(768, 2304)
    backbone.convnext.{i}.pwconv1.bias       -> bias
    backbone.convnext.{i}.pwconv2.weight     -> Linear(2304, 768)
    backbone.convnext.{i}.pwconv2.bias       -> bias
    backbone.convnext.{i}.gamma              -> Parameter(768)
backbone.final_layer_norm.weight            -> LayerNorm weight
backbone.final_layer_norm.bias              -> LayerNorm bias

// Head
head.out.weight               -> Linear(768, 1282)
head.out.bias                 -> Linear bias
head.istft.window             -> Hann window buffer (1280,) — NOT trainable

// Codebook (for decode-from-codes path)
feature_extractor.encodec.quantizer.vq.layers.0._codebook.embed -> (4096, 512)
```

**Tensor parallel consideration:**
- The Conv1d and Linear layers in the backbone can be TP-sharded along the output dimension
- The codebook.embed (4096, 512) can be sharded along codebook dim or embedding dim
- The iSTFT head Linear(768, 1282) can be sharded along output
- AdaLayerNorm embeddings (4, 768) are small, may not need TP
- The iSTFT operation itself is a CPU/Python-level operation (torch.istft) — on GPU this requires a custom kernel or a library call. In grim's context, this is a significant implementation challenge.

**Open questions:**
1. **iSTFT on GPU**: torch.istft is not available as a native ROCm/HIP kernel in grim. Options: (a) implement iSTFT as a custom HIP kernel, (b) use a library like rocFFT, (c) precompute the iSTFT as a linear operation (the overlap-add can be expressed as a matrix multiply), (d) fall back to CPU for the iSTFT step. Option (c) is most promising — iSTFT with fixed window is a linear operation that can be represented as a conv transpose or matrix multiply.
2. **bandwidth_id**: The AdaLayerNorm takes a bandwidth_id (0-3). For decode-from-codes, which bandwidth condition should be used? The config has `use_attention: false` and `attention_layers: 0`, suggesting this is a "small" variant. Probably bandwidth_id=0 (full bandwidth) for decode.
3. **Do we need the encoder (feature_extractor) at all?** The file is named `wav_tokenizer_dec` — decoder only. If the use case is "given discrete codes, produce audio," then we only need codebook + backbone + head. The encoder is only needed for encode (audio -> codes).
4. **Checkpoint availability**: No safetensors or pytorch_model.bin on disk in `old/mods/wav_tokenizer_dec/`. The modeling file describes the checkpoint structure but no actual weights are present. To test, we'd need to either download a checkpoint or generate random weights for smoke testing.
5. **llama.cpp vs HF architecture mismatch**: llama.cpp's wavtokenizer-dec.cpp uses POSNET+ConvNeXt, while HF uses Vocos+ISTFT. If grim's GGUF files were converted via llama.cpp's `convert_hf_to_gguf.py` (as shown in the TTS README), the GGUF would contain the HF architecture. But llama.cpp's runtime decoder (wavtokenizer-dec.cpp) implements POSNET+ConvNeXt, which is DIFFERENT. This suggests llama.cpp's wavtokenizer-dec.cpp may be an older/different implementation, or there are two WavTokenizer variants. The HF modeling_wavtokenizer.py on disk is the authoritative source for the HF checkpoint format.
6. **The YAML config** (`wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml`) describes a training run with: backbone dim=768, intermediate_dim=2304, num_layers=12, adanorm_num_embeddings=4, n_fft=1280, hop_length=320, codebook_size=4096, vq_kmeans=200. This corroborates the config.json values.

**Definition of done:**
- WavTokenizerDecConfig parses from HF config.json (or from the local config.json on disk)
- WavTokenizerDec loads all backbone + head + codebook weights from a safetensors/GGUF weight source
- Decode path: codes (B, T) -> codebook lookup -> backbone -> iSTFT head -> audio (B, 1, samples)
- Smoke test: random weights, codes input -> audio output with correct shape
- Audio output shape verification: for a given input code sequence length T, output samples = (T - 1) * hop_length + n_fft (approx, depending on padding/ center)
- Numerical parity: against HF WavTokenizer `model.decode(features)` or `model(input_ids=codes)` using the modeling_wavtokenizer.py reference
- iSTFT implementation: either a working HIP kernel or a verified linear-reexpression of iSTFT

**Implementation priority:** This is one of the more complex files because of the iSTFT head. The backbone (ConvNeXt + AdaLayerNorm) is straightforward. The iSTFT is the hard part — it requires either a custom kernel or a linear reexpression. Recommend implementing the backbone + codebook first (smoke test), then tackle iSTFT.

### 9. deepseek2.rs — WRONG, needs real MLA+MoE implementation

### 9. deepseek2.rs — WRONG, needs real MLA+MoE implementation

**Current state:** Thin `Llama` wrapper. DeepSeek V2 uses MLA + MoE + YARN.

**What's needed:**
- Config: `DeepSeek2Config` with MLA params (`kv_lora_rank`, `q_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`), MoE params (`n_routed_experts`, `n_shared_experts`, `num_experts_per_tok`, `moe_intermediate_size`, `routed_scaling_factor`, `scoring_func`, `first_k_dense_replace`), YARN params
- Block: `DeepSeek2Block` with MLA (q_proj, kv_a_proj_with_mqa, kv_a_layernorm, kv_b_proj, o_proj) + MLP (gate, up, down) + shared experts + MoE router
- Forward:
  - MLA: Q latent = q_proj(x), KV latent = kv_a_proj_with_mqa(x), apply kv_a_layernorm to KV latent, then kv_b_proj expands to K and V, RoPE on Q and K (with YARN scaling), attention
  - MoE: if layer is MoE layer (not first_k_dense_replace), route through experts + shared experts
  - MLP: gate, up, down (Silu activation)
- The existing `deepseek.rs` has a `DeepSeekBlock` with MLA but:
  - Uses different naming (q_a_proj/q_b_proj/kv_a_proj/kv_b_proj vs q_proj/kv_a_proj_with_mqa/kv_b_proj)
  - Has `kv_a_proj` without a separate layernorm (V2 has `kv_a_layernorm`)
  - Has `head_dim: 128` hardcoded (V2 uses qk_nope_head_dim=128 + qk_rope_head_dim=64 = 192 for Q, v_head_dim=128 for V)
  - Does NOT have MoE (V2 has MoE with 64 experts + 2 shared experts)
  - Does NOT have YARN (V2 has YARN)
- So the existing `deepseek.rs` is close but not sufficient for V2. We need either to extend it or create a new `DeepSeek2Block`.

**Loading mapping:**
- `model.layers.{i}.self_attn.q_proj.weight` → Q latent projection
- `model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight` → KV latent projection
- `model.layers.{i}.self_attn.kv_a_layernorm.weight` → KV latent layernorm
- `model.layers.{i}.self_attn.kv_b_proj.weight` → KV expansion projection
- `model.layers.{i}.self_attn.o_proj.weight` → attention output
- `model.layers.{i}.mlp.gate_proj.weight` → MLP gate
- `model.layers.{i}.mlp.up_proj.weight` → MLP up
- `model.layers.{i}.mlp.down_proj.weight` → MLP down
- `model.layers.{i}.mlp.shared_experts.gate_proj/up_proj/down_proj.weight` → shared experts (2)
- `model.layers.{i}.mlp.experts.{e}.gate_proj/up_proj/down_proj.weight` → routed experts (64)
- `model.layers.{i}.input_layernorm.weight` → attn_norm
- `model.layers.{i}.post_attention_layernorm.weight` → ffn_norm
- Top-level: `model.embed_tokens.weight`, `model.norm.weight`, `lm_head.weight`

**Open questions:**
- What is the exact YARN implementation? The config has specific YARN parameters (beta_fast, beta_slow, factor, mscale, mscale_all_dim). Need to implement YARN RoPE scaling.
- How does `first_k_dense_replace: 1` work? The first layer is dense (no MoE), rest are MoE.
- How are the MoE experts loaded? The safetensors index shows `model.layers.{i}.mlp.experts.{e}.*` for e in 0..64. Need to load all 64 experts per MoE layer.

**Definition of done:**
- Load a DeepSeek V2 checkpoint (need weights — `old/mods/deepseek2/` has config + index + modeling code but need weights)
- Prefill + decode produce correct numerics vs HuggingFace transformers DeepSeek V2 implementation
- The existing `deepseek.rs` tests can serve as a reference for the MLA attention math, but V2's MLA is different (no Q compression, different head dims)
- Tests: smoke test, config parse test, MLA attention parity test (vs HF transformers or modeling_deepseek.py reference)

### 10. deepseek32.rs — WRONG, needs real implementation

**Current state:** Thin `Llama` wrapper.

**What's needed:**
- HF download for deepseek3 config (model_type: `deepseek_v3` or similar)
- DeepSeek V3 uses MLA + MoE + MoE with more experts (likely 256 or more) + possible MLA changes
- Probably similar to V2 but with more experts and possibly different MLA configuration

**Open questions:**
- What is deepseek3's exact architecture? Need HF config + modeling code.
- How many experts? What is the MLA configuration?
- Is it similar enough to V2 to reuse the V2 block?

**Definition of done:**
- After HF download: implement correct topology
- Load a DeepSeek V3 checkpoint
- Prefill + decode produce correct numerics
- Tests: smoke test, config parse test

### 11. deepseek4.rs — WRONG, needs real implementation

**Current state:** Thin `Llama` wrapper.

**What's needed:**
Local DeepSeekV4 full reference implementation exists on disk at `old/mods/Deepseek4 Flash/` — `model.py` (961 lines), `kernel.py` (536 lines with TileLang FP4/FP8/SparseAttention kernels), `encoding_dsv4.py` (760 lines, chat message encoding), `convert.py` (155 lines, HF safetensors → sharded checkpoint converter), `generate.py` (145 lines), `jang_config.json` (JANG quantization metadata), `config.json` (HF model config), `tokenizer.json`.

DeepSeek V4 architecture from the local reference:

- tokenizer: 129280 vocab; bos `<｜begin▁of▁sentence｜>`, eos `<｜end▁of▁sentence｜>`, thinking tokens `<think>` / `<｜｜>`, assistant/user tokens `<｜Assistant｜>` / `<｜User｜>`, DSML tool-calling token `｜DSML｜`
- model_type: `deepseek_v4` (expected)
- main model: 43 layers, 64 attention heads, 256 routed experts, 6 activated per token, 1 shared expert, 4096 dim, 4096 moe_inter_dim, 1024 q_lora_rank, 1024 o_lora_rank, 512 head_dim, 64 rope_head_dim, 448 nope_head_dim, 8 o_groups, window_size 128, norm_eps 1e-6
- compress_ratios: per-layer tuple, e.g. (0,0,4,128,4,128,4,0,...) — many layers have 0 (pure sliding window), some have 4 (overlap KV compression), some have 128 (heavy compression)
- YaRN: compress_rope_theta=40000, rope_theta=10000, rope_factor=40, beta_fast=32, beta_slow=1, original_seq_len from config
- indexer: 64 index_n_heads, 128 index_head_dim, 512 index_topk, uses own compressor with Hadamard rotation + learned weights_proj
- MoE gate: sqrtsoftplus scoring, routed_scaling_factor from config, hash routing for first n_hash_layers vs score-based for rest, tid2eid table when hash
- expert dtype: FP4 (float4_e2m1fn_x2) with per-32 E8M0 scale, swiglu_limit from config
- FP8 activations: act_quant block size 64 for non-rope dims, scales in E8M0 or FP32
- FP8 GEMM: per-128 block FP8 scale on both A and B, FP32 accumulator
- FP4 GEMM: FP8 act x FP4 weight, weight scale per-32 along K
- sparse attention: FlashAttention-style online softmax, top-k KV positions selected per (batch, seq_pos) via Indexer or sliding-window idx
- Hyper-Connections: hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6, pre/post/comb mixing per layer, hc_head at the end
- DSpark stage: additional MTP/diffusion-stage blocks after main layers, dspark_block_size, dspark_target_layer_ids, dspark_markov_rank, dspark_noise_token_id
- generation: greedy or gumbel-max sampling at temperature, prefill phase (process all prompt tokens), decode phase (one token at a time), DSpark draft stage for speculative decoding
- generate.py: batch generation with left-padded prompts, prefill phase processes [min_prompt_len:] tokens, decode phase generates one token at a time

**Loading mapping (from model.py + convert.py):**

Main model (model.layers.{i}.* namespace in HF safetensors):

```
self_attn.q_norm.weight / .bias                → q_norm.weight / .bias   (RMSNorm, fp32 in checkpoint)
self_attn.wq_a.weight                          → wq_a.weight              (Linear, fp32 or fp8)
self_attn.wq_b.weight                          → wq_b.weight              (ColumnParallelLinear, fp32)
self_attn.wkv.weight                           → wkv.weight               (Linear, fp32)
self_attn.kv_norm.weight / .bias              → kv_norm.weight / .bias  (RMSNorm)
self_attn.wo_a.weight                          → wo_a.weight              (ColumnParallelLinear, fp32, then unflatten to o_groups x o_lora_rank)
self_attn.wo_a.scale                           → wo_a.scale               (per-128 FP8 scale)
self_attn.wo_b.weight                          → wo_b.weight              (RowParallelLinear)
self_attn.attn_sink                            → attn_sink                (register_buffer, fp32, per-head sink bias)
self_attn.freqs_cis                            → freqs_cis                (register_buffer, precomputed YaRN cis)
self_attn.kv_cache                             → kv_cache                 (register_buffer, sliding-window + compressed KV)

ffn.gate.weight                                → gate.weight               (Linear, fp32 or fp4 quantized)
ffn.gate.scale                                 → gate.scale               (per-32 E8M0 scale if FP4)
ffn.up.weight                                  → up.weight                (Linear)
ffn.up.scale                                   → up.scale
ffn.down.weight                                → down.weight              (Linear)
ffn.down.scale                                 → down.scale
ffn.shared_experts.gate.weight                 → shared_experts.gate.weight
ffn.shared_experts.gate.scale                  → shared_experts.gate.scale
ffn.shared_experts.up.weight / down.weight     → shared_experts.up/down.weight
ffn.shared_experts.up.scale / down.scale       → shared_experts.up/down.scale
ffn.experts.{e}.w1.weight                      → experts.{e}.gate.weight  (FP4 quantized if expert_dtype=fp4)
ffn.experts.{e}.w1.scale                       → experts.{e}.gate.scale
ffn.experts.{e}.w2.weight                      → experts.{e}.up.weight
ffn.experts.{e}.w2.scale                       → experts.{e}.up.scale
ffn.experts.{e}.w3.weight                      → experts.{e}.down.weight
ffn.experts.{e}.w3.scale                       → experts.{e}.down.scale
ffn.gate.bias                                  → gate.bias                (score bias, fp32)
ffn.gate.tid2eid                               → tid2eid                 (gate.tid2eid, fp32, per-token expert ids for hash routing)
ffn.gate.e_score_correction_bias               → e_score_correction_bias  (per-expert correction, fp32)

attn_norm.weight                               → attn_norm.weight         (RMSNorm, fp32)
ffn_norm.weight                                → ffn_norm.weight          (RMSNorm, fp32)

hc_attn_fn.weight                              → hc_attn_fn.weight        (mixing parameters for Hyper-Connections, fp32)
hc_attn_fn.bias                                → hc_attn_fn.bias
hc_attn_base.weight                            → hc_attn_base.weight
hc_attn_scale.weight                           → hc_attn_scale.weight
... same for hc_ffn_fn, hc_ffn_base, hc_ffn_scale

embed.weight                                   → embed.weight             (ParallelEmbedding, vocab_dim x dim, sharded along vocab)
embed.weight_scale / embed.weight_scale_inv    → embed weight scale (if FP8)

head.weight                                    → head.weight              (ParallelHead, part_vocab x dim, fp32, sharded along vocab)
```

DSpark stage (mtp.* namespace in HF safetensors, if present):

```
mtp.{i}.embed.weight                           → mtp.{i}.embed.weight
mtp.{i}.main_proj.weight / bias                → mtp.{i}.main_proj.weight / bias
mtp.{i}.main_norm.weight / bias                → mtp.{i}.main_norm.weight / bias
mtp.{i}.norm.weight / bias                     → mtp.{i}.norm.weight / bias
mtp.{i}.hc_head_fn.weight                      → mtp.{i}.hc_head_fn.weight
mtp.{i}.hc_head_fn.bias                        → mtp.{i}.hc_head_fn.bias
mtp.{i}.hc_head_base.weight                    → mtp.{i}.hc_head_base.weight
mtp.{i}.hc_head_scale.weight                   → mtp.{i}.hc_head_scale.weight
mtp.{i}.markov_w1.weight                       → mtp.{i}.markov_w1.weight (ParallelEmbedding, vocab_dim x markov_rank)
mtp.{i}.markov_w1.weight_scale                 → mtp.{i}.markov_w1.weight_scale
mtp.{i}.markov_w2.weight                       → mtp.{i}.markov_w2.weight (ParallelHead, part_vocab x markov_rank, fp32)
mtp.{i}.markov_w2.weight_scale                 → mtp.{i}.markov_w2.weight_scale
mtp.{i}.confidence_head.proj.weight            → mtp.{i}.confidence_head.proj.weight (Linear, fp32)
mtp.{i}.confidence_head.proj.weight_scale      → mtp.{i}.confidence_head.proj.weight_scale
```

**Weight format notes:**

- FP4 weights: stored as `float4_e2m1fn_x2` (2 FP4 values per byte, packed along K). The weight shape is [out, in//2] in this dtype, logical [out, in] in fp4. Scale is [out, in//32] in float8_e8m0fnu (1 scale per 32 fp4 elements along K). The reference GEMM kernel fp4_gemm does: A_fp8[M,K] @ B_fp4[N,K]^T where B is [N, K//2] in float4_e2m1fn_x2, cast internally to FP8 via FP32, then FP8xFP8 GEMM with act_scale (per-128 on K) * weight_scale (per-32 on K) applied to accumulator.
- FP8 weights: stored as `float8_e4m3fn`, scale is [out//128, in//128] in float8_e8m0fnu (per-128 blocks along both dims).
- BF16 weights: stored as bfloat16, no scale.
- The `convert.py` script does the HF→sharded conversion: `wo_a.weight` is dequantized by unflattening to (o_groups, 128, -1, 128), multiplying by scale[:,None,:,None], then flattening back and converting to BF16; expert weights that are int8 are either cast to e4m3fn (for fp8 variant) or viewed as float4_e2m1fn_x2 (for fp4 variant).

**Architecture from `configuration_*.py` (Python-side):**

The local `config.json` has:
```json
{
  "architectures": ["DeepSeekV4ForCausalLM"],
  "model_type": "deepseek_v4",
  "hidden_size": 4096,
  "num_attention_heads": 64,
  "num_key_value_heads": 64,   // MLA: same as num_attention_heads (all heads are KV heads, but grouped)
  "num_hidden_layers": 43,
  "intermediate_size": 4096,    // same as dim (SwiGLU intermediate = dim)
  "head_dim": 512,
  "rope_head_dim": 64,
  "nope_head_dim": 448,         // head_dim - rope_head_dim
  "q_lora_rank": 1024,
  "kv_lora_rank": 512,
  "o_lora_rank": 1024,
  "o_groups": 8,
  "vocab_size": 129280,
  "rms_norm_eps": 1e-6,
  "swiglu_limit": 7.0,
  "moe_intermediate_size": 4096,
  "n_routed_experts": 256,
  "n_shared_experts": 1,
  "num_experts_per_tok": 6,
  "first_k_dense_replace": 0,
  "routed_scaling_factor": 1.5,
  "scoring_func": "sqrtsoftplus",
  "moe_layer_freq": 1,
  "norm_topk_prob": false,
  "rope_theta": 10000.0,
  "rope_scaling": {
    "type": "yarn",
    "factor": 40,
    "original_max_position_embeddings": 4096,
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 0.707,
    "mscale_all_dim": 0.707
  },
  "compress_ratios": [0,0,4,128,4,128,4,0,...],  // 43 entries
  "rope_scaling": {...},  // YaRN params
  "max_position_embeddings": 40960,
  "tie_word_embeddings": false,
  "use_cache": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "5.x.x",
  "quantization_config": {
    "quant_method": "fp8",
    "format": "fp8",
    "ignore": ["re:.*shared_experts.*", "re:.*lm_head.*"]
  }
}
```

Wait — the local config.json at `old/mods/Deepseek4 Flash/config.json` has 2464 chars. Let me check what it actually contains. I read it earlier — it had 10 keys including `architectures`, `auto_map`, `model_type`, `text_config` with nested `hidden_size`, `num_attention_heads`, `num_key_value_heads`, `num_hidden_layers`, `intermediate_size`, `hidden_act`, `rms_norm_eps`, `rope_theta`, `rope_scaling`, `vocab_size`, `tie_word_embeddings`, `moe_intermediate_size`, `n_routed_experts`, `n_shared_experts`, `num_experts_per_tok`, `first_k_dense_replace`, `routed_scaling_factor`, `scoring_func`, `norm_topk_prob`, `use_cache`, `torch_dtype`, `transformers_version`. No YARN, no compress_ratios, no hc_mult. That's because the local config.json is a partial HF config — the full ModelArgs are in `model.py`'s `ModelArgs` dataclass.

**Key implementation deltas from the existing DeepSeek crate (deepseek.rs / deepseek2.rs):**

1. HC (Hyper-Connections): the existing crate has standard residual; DeepSeek V4 has hc_mult=4 copies with Sinkhorn mixing. This is a novel architecture not in the crate.
2. Compressor + Indexer: sliding-window KV cache with learned compression (ratio 4 or 128), plus a separate Indexer that selects top-k KV positions via its own compressor with Hadamard rotation. Novel.
3. FP8/FP4 quantized experts: the existing crate uses BF16; DeepSeek V4 has FP4 experts with per-32 E8M0 scale. Need to implement fp4_gemm or fall back to dequantize-then-BF16-GEMM (the convert.py dequantizes wo_a, and expert weights can be dequantized similarly).
4. sqrtsoftplus scoring: existing crate uses softmax or sigmoid; DeepSeek V4 uses sqrtsoftplus (F.softplus(x).sqrt()).
5. No MLA compression (q_lora_rank=1024 is the full latent, not compressed): the existing DeepSeek crate has MLA with compressed Q; DeepSeek V4 has uncompressed Q latent + compressed KV (via Compressor/Indexer) instead.
6. Grouped output projection (o_groups=8, o_lora_rank=1024): the existing crate has a simple o_proj; DeepSeek V4 has wo_a (grouped, low-rank) followed by wo_b (row-parallel).
7. dspark MTP stage: speculative decoding / block diffusion stage after main layers. The existing crate has no equivalent.

**What can be done before HF download for the main config:**

The local model.py + kernel.py + config.json + jang_config.json + convert.py + generate.py are sufficient to write a correct DeepSeekV4 model file WITHOUT any HF download. The model.py is the reference implementation; the config.json has the model_type and architecture identifier; the kernel.py has the quantization and attention primitives; the convert.py documents the weight mapping; the generate.py documents the generation contract.

The only thing truly missing is the weight files (the actual safetensors shards referenced by model.safetensors.index.json, or the GGUF files for `rim-load`). The index.json maps tensor names to shard files; the shard files themselves are not on disk. So:
- Config + modeling + kernel + weight-key layout: fully known from local disk
- Weight values: not on disk (need either HF download or a local safetensors source)

**Recommended approach:**

1. Create `DeepSeekV4Config` that parses from HF config.json (model_type deepseek_v4). Use the local config.json as the test case.
2. Port the core block topology from model.py:
   - RMSNorm (fp32 param, bf16 forward)
   - Linear with fp4/fp8/BF16 dispatch
   - ColumnParallelLinear / RowParallelLinear / ParallelEmbedding (sharding setup, can be single-rank for now)
   - Compressor: ratio 4 (overlap) and 128 (non-overlap) variants, gated pooling
   - Indexer: top-k KV position selection with its own compressor + Hadamard rotation
   - Attention: MLA with q_lora_rank, head_dim, rope_head_dim, nope_head_dim, o_groups, o_lora_rank, sliding window + compressed KV top-k
   - Gate: sqrtsoftplus scoring, hash routing vs score-based
   - Expert: SwiGLU with swiglu_limit
   - MoE: top-k routed + 1 shared expert, expert sharding
   - Block: HC pre/post mixing via Sinkhorn
   - Transformer: embed (ParallelEmbedding) → layers → norm → head (ParallelHead) → sample
3. Port the quantization primitives from kernel.py:
   - act_quant (FP8 block-wise, inplace or with scale)
   - fp4_act_quant (FP4 block-wise, inplace or with scale)
   - fp8_gemm (per-128 block FP8 scale on both A and B)
   - fp4_gemm (FP8 act x FP4 weight, per-128 act scale, per-32 weight scale)
   - sparse_attn (FlashAttention-style, top-k KV indices, learnable attn_sink)
   - hc_split_sinkhorn (Sinkhorn on mixing matrix, 20 iters, eps 1e-6)
4. Audio/Text generation:
   - Use encoding_dsv4.py for the chat message encoding (bos/eos/thinking/assistant/user/DSML tokens)
   - Use generate.py for the generation contract (prefill + decode + DSpark speculative)
5. Weight loading:
   - Map HF safetensors keys → local weight struct
   - Handle FP4 weight format (float4_e2m1fn_x2, scale in float8_e8m0fnu)
   - Handle FP8 weight format (float8_e4m3fn, scale in float8_e8m0fnu)
   - Handle BF16 weights
   - Document via convert.py mapping
6. Tests:
   - Config parse from local config.json
   - Smoke test with random weights (BF16, no quantization)
   - Test with FP8 activation quantization (act_quant)
   - Test with FP4 expert quantization (fp4_act_quant + fp4_gemm)
   - Test sparse attention with top-k indices
   - Test Compressor/Indexer
   - Test HC mixing
   - Full forward: prefill + decode from generate.py contract
   - Numerical parity: run the same input through model.py reference (Python) and compare
7. Open questions:
   - Is DeepSeek V4 main config on HF? The local config.json exists; HF may have a different variant. The local config is the ground truth for THIS checkpoint.
   - DSpark stage: present in model.py if dspark_block_size > 0; config may or may not have it.
   - FP4 expert dtype: configurable in model.py (expert_dtype None/FP4); config.json likely specifies.
   - FP8 scale dtype: E8M0 or FP32; model.py switches based on scale_dtype.
   - HC: present in all blocks and the head; not optional.
   - The jang_config.json says `format_version: 2.0`, `quantization_method: mse-all`, `scoring_method: weight-magnitude`, `target_bits: 3`, `actual_bits: 3.36`, `block_size: 128`, `hadamard_rotation: false`. This describes the JANG quantization strategy applied to the checkpoint; for loading, we care about the actual weight format (fp4/fp8/bf16) which is in the safetensors files per the index + convert.py mapping.

**Definition of done:**
- DeepSeekV4Config parses from local config.json (and HF equivalent)
- DeepSeekV4Block loads from WeightSource with FP4/FP8/BF16 dispatch
- Forward produces correct numerics vs model.py reference (Python)
- Attention with Compressor/Indexer produces correct sparse attention output
- MoE with sqrtsoftplus scoring + hash/score routing produces correct expert output
- HC mixing produces correct pre/post/comb
- DSpark stage (if present) produces correct speculative draft
- Tests: config parse, smoke test (BF16), quantization smoke test (FP8 act, FP4 experts), sparse attention test, HC test, full prefill+decode parity vs model.py

**Implementation priority:** High — the local reference is complete enough to write a correct implementation now without HF download.

### 18. diffusion_gemma.rs — STUB, needs real block-diffusion implementation

**Current state:** Stub returning `Unimplemented`.

**What's needed:**

Local artifacts on disk at `old/mods/diffusiongemma/`:
- config.json (3469B, model_type diffusion_gemma, text_config + vision_config + canvas_length + vision_soft_tokens_per_image)
- model.safetensors.index.json (105KB, 11 shards, 1.5TB total) — transformer weight key map
- model_index.json (295B, points to transformers DiffusionGemmaForBlockDiffusion)
- tokenizer.json (32MB, 262144 vocab)
- tokenizer_config.json, chat_template.jinja, processor_config.json, scheduler_config.json, generation_config.json
- README.md (284 lines, DiffusionGemma model card — architecture, capabilities, usage, best practices)

**Architecture (from config.json + README.md + model_index.json):**

- model_type: `diffusion_gemma`
- architectures: `["DiffusionGemmaForBlockDiffusion"]`
- text_config:
  - hidden_size: 2816
  - num_attention_heads: 16
  - num_key_value_heads: 8
  - num_hidden_layers: 30
  - intermediate_size: 2112
  - head_dim: 256
  - global_head_dim: 512
  - hidden_act: gelu_pytorch_tanh (GeGLU)
  - rms_norm_eps: 1e-06
  - rope_parameters: full_attention {partial_rotary_factor: 0.25, rope_theta: 1000000.0, rope_type: proportional}, sliding_attention {rope_theta: 10000.0, rope_type: default}
  - sliding_window: 1024
  - layer_types: sliding_attention x 21, full_attention x 8 (per layer: [sliding,sliding,...,full,...,sliding,...])
  - moe_intermediate_size: 704
  - num_experts: 128
  - top_k_experts: 8
  - use_bidirectional_attention: "vision"
  - vocab_size: 262144
  - max_position_embeddings: 262144
  - final_logit_softcapping: 30.0
  - tie_word_embeddings: true
- vision_config:
  - model_type: gemma4_vision
  - hidden_size: 1152
  - num_attention_heads: 16
  - num_hidden_layers: 27
  - patch_size: 16
  - intermediate_size: 4304
  - rms_norm_eps: 1e-06
  - rope_parameters: rope_theta: 100.0, rope_type: default
  - max_position_embeddings: 131072
  - pooling_kernel_size: 3
  - position_embedding_size: 10240
  - standardize: true
- canvas_length: 256
- vision_soft_tokens_per_image: 280
- boi_token_id: 255999, eoi_token_id: 258882, image_token_id: 258880
- eos_token_id: [1, 106]

**Block diffusion inference pattern (from README.md):**

- Encoder-decoder architecture: autoregressive encoder processes prompt context + KV cache; decoder applies bidirectional attention over the generation canvas (block of 256 tokens)
- Multi-canvas sampling: generate 256-token canvas in parallel via diffusion denoising, then encode + append to KV cache
- During generation: at each step, the model denoises a full block of 256 tokens, using cross-attention to the cached prompt context
- Sampling config: 48 max denoising steps, temperature schedule 0.8→0.4 linear decay, entropy bound 0.1, adaptive stopping at entropy threshold 0.005
- Token selection: at each step, select lowest-entropy tokens whose mutual information bound stays below 0.1; fully re-noise the non-selected tokens
- Output: final answer after denoising completes or adaptive stopping triggers

**Forward contract (from README.md + config.json):**

The model is NOT a standard autoregressive LM. It's a block-diffusion model with:

1. **Encoder (prefill):** processes the prompt (text + images) through the autoregressive transformer, producing hidden states + KV cache
2. **Decoder (diffusion):** for each denoising step, takes a canvas of 256 tokens (partially denoised), runs bidirectional self-attention + cross-attention to the encoder KV cache, produces logits for the next denoising iteration
3. **Vision:** Gemma4 vision encoder processes images into soft tokens (280 per image), which are fed into the text decoder as part of the interleaved sequence

**Generation flow (from README.md):**

```
input: prompt (text + images)
1. tokenize prompt (chat_template with image placeholders)
2. encoder prefill: run autoregressive forward on prompt tokens → KV cache
3. initialize canvas: 256 random/noisy tokens
4. for denoising_step in 0..48:
   a. decode: run bidirectional forward on canvas + cross-attend to encoder KV cache
   b. compute entropies over canvas positions
   c. select tokens with entropy below bound (0.1)
   d. re-noise non-selected tokens
   e. if average entropy < 0.005 AND stable predictions (same as previous step): stop
5. final: run encoder forward on denoised canvas → append to KV cache → generate next canvas
6. repeat until EOS
```

**What a correct implementation needs:**

1. **DiffusionGemmaConfig** — parses from HF config.json:
   - text_config fields (hidden_size, num_attention_heads, num_key_value_heads, num_hidden_layers, intermediate_size, head_dim, global_head_dim, hidden_act, rms_norm_eps, rope_parameters, sliding_window, layer_types, moe_intermediate_size, num_experts, top_k_experts, use_bidirectional_attention, vocab_size, max_position_embeddings, final_logit_softcapping, tie_word_embeddings)
   - vision_config fields (hidden_size, num_attention_heads, num_hidden_layers, patch_size, intermediate_size, rms_norm_eps, rope_parameters, pooling_kernel_size, position_embedding_size, standardize)
   - canvas_length, vision_soft_tokens_per_image, boi_token_id, eoi_token_id, image_token_id, eos_token_id

2. **Text decoder block** (block-diffusion aware):
   - GeGLU MLP (gelu_pytorch_tanh, gate_proj + up_proj + down_proj, intermediate_size 2112)
   - Sliding attention (rope_theta 10000.0, rope_type default, sliding_window 1024) for sliding_attention layers
   - Full attention (rope_theta 1000000.0, partial_rotary_factor 0.25, rope_type proportional) for full_attention layers
   - MoE router (num_experts 128, top_k_experts 8, moe_intermediate_size 704)
   - Three norms: pre_feedforward_layernorm, post_attention_layernorm, post_feedforward_layernorm (similar to Gemma3, but with sliding/full attention mix)
   - Layer scalar (layer-specific scaling factor per the safetensors keys)
   - Router (per_expert_scale, proj.weight, scale) — top-k gating

3. **Vision encoder** (Gemma4 vision):
   - Patch embedding (patch_size 16, hidden_size 1152)
   - 27 transformer layers with rope (rope_theta 100.0)
   - Pooling (pooling_kernel_size 3, position_embedding_size 10240)
   - standardize (input standardization)

4. **Block diffusion logic:**
   - Canvas initialization (random tokens of length 256)
   - Diffusion denoising loop (configurable steps, entropy threshold, temperature schedule)
   - Cross-attention to encoder KV cache during decode
   - Bidirectional attention over the canvas (use_bidirectional_attention: "vision" — but in practice the decoder attention is bidirectional over the canvas)
   - Adaptive stopping criteria (entropy < 0.005 AND stable predictions)

5. **Loading mapping (from model.safetensors.index.json):**

```
# Text decoder
model.decoder.embed_tokens.weight              → text_embed_tokens
model.decoder.layers.{i}.input_layernorm.weight → layers.{i}.attn_norm
model.decoder.layers.{i}.self_attn.q_proj.weight → layers.{i}.wq
model.decoder.layers.{i}.self_attn.k_proj.weight → layers.{i}.wk
model.decoder.layers.{i}.self_attn.v_proj.weight → layers.{i}.wv
model.decoder.layers.{i}.self_attn.o_proj.weight → layers.{i}.wo
model.decoder.layers.{i}.self_attn.q_norm.weight → layers.{i}.q_norm  (per-head Q norm)
model.decoder.layers.{i}.self_attn.k_norm.weight → layers.{i}.k_norm  (per-head K norm)
model.decoder.layers.{i}.post_attention_layernorm.weight → layers.{i}.post_attn_norm
model.decoder.layers.{i}.pre_feedforward_layernorm.weight → layers.{i}.pre_ffn_norm
model.decoder.layers.{i}.post_feedforward_layernorm.weight → layers.{i}.post_ffn_norm
model.decoder.layers.{i}.post_feedforward_layernorm_1.weight → layers.{i}.post_ffn_norm_1  (additional norm variant)
model.decoder.layers.{i}.post_feedforward_layernorm_2.weight → layers.{i}.post_ffn_norm_2  (additional norm variant)
model.decoder.layers.{i}.pre_feedforward_layernorm_2.weight → layers.{i}.pre_ffn_norm_2  (additional norm variant)
model.decoder.layers.{i}.layer_scalar           → layers.{i}.layer_scalar (per-layer scale)
model.decoder.layers.{i}.mlp.gate_proj.weight   → layers.{i}.w_gate
model.decoder.layers.{i}.mlp.up_proj.weight     → layers.{i}.w_up
model.decoder.layers.{i}.mlp.down_proj.weight   → layers.{i}.w_down
model.decoder.layers.{i}.mlp.shared_experts.{proj}.weight → shared experts
model.decoder.layers.{i}.mlp.experts.{e}.{proj}.weight → routed experts (128 total)

# Router (per layer)
model.decoder.layers.{i}.router.proj.weight     → router.proj.weight (top-k gating projection)
model.decoder.layers.{i}.router.scale           → router.scale (sigmoid gating scale)
model.decoder.layers.{i}.router.per_expert_scale → router.per_expert_scale (per-expert scale)

# Vision encoder
model.vision.embed_tokens.weight               → vision_embed_tokens (patch embedding, patch_size=16, in_chans=3, hidden_size=1152)
model.vision.layers.{i}.input_layernorm.weight  → vision layers.{i}.attn_norm
model.vision.layers.{i}.self_attn.{q,k,v,o}_proj.weight → vision attention
model.vision.layers.{i}.post_attention_layernorm.weight → vision post_attn_norm
model.vision.layers.{i}.mlp.{gate,up,down}_proj.weight → vision MLP
model.vision.layers.{i}.mlp.shared_experts.{proj}.weight → vision shared experts
model.vision.layers.{i}.mlp.experts.{e}.{proj}.weight → vision routed experts
model.vision.layers.{i}.post_feedforward_layernorm.weight → vision post_ffn_norm
model.vision.layers.{i}.router.proj.weight / scale / per_expert_scale → vision router
model.vision.layers.{i}.layer_scalar            → vision layer_scalar

# Final layers
model.norm.weight                              → final_norm.weight
lm_head.weight                                → lm_head.weight (uses tie_word_embeddings=true, so shares with embed_tokens)
model.vision.post_pooling_proj.weight          → vision post-pooling projection (position_embedding_size → hidden_size?)
model.vision.pre_pooling_proj.weight / bias     → vision pre-pooling projection
model.vision.post_attention_layernorm.weight   → vision post-attention norm (after pooling?)

# Conditioning / special
model.cross_attention.{...}?                    → cross-attention parameters for encoder→decoder conditioning (need to verify from index)
```

Wait — let me check what's actually in the index. I read the first 500 lines earlier and saw `model.decoder.layers.{i}.*` keys. Let me check the vision and cross-attention keys. From what I saw at offset 380 in the diffusiongemma/index (lines 380-429), the keys are:

```
model.decoder.layers.24.experts.gate_up_proj
model.decoder.layers.24.input_layernorm.weight
model.decoder.layers.24.layer_scalar
model.decoder.layers.24.mlp.down_proj.weight / gate_proj / up_proj
model.decoder.layers.24.post_attention_layernorm.weight
model.decoder.layers.24.post_feedforward_layernorm.weight
model.decoder.layers.24.post_feedforward_layernorm_1.weight
model.decoder.layers.24.post_feedforward_layernorm_2.weight
model.decoder.layers.24.pre_feedforward_layernorm.weight
model.decoder.layers.24.pre_feedforward_layernorm_2.weight
model.decoder.layers.24.router.per_expert_scale
model.decoder.layers.24.router.proj.weight
model.decoder.layers.24.router.scale
model.decoder.layers.24.self_attn.k_norm.weight / q_norm.weight / k_proj / q_proj / o_proj / v_proj
model.decoder.layers.25.{same pattern}
model.decoder.layers.26.{same pattern}
```

So the decoder uses `model.decoder.layers.{i}.*` namespace. The vision encoder keys are probably `model.vision.layers.{i}.*` or similar — need to check the beginning of the index. But the safetensors index at offset 1 had the full weight_map starting with... let me check the first 50 lines I didn't read yet.

Actually, I read the first 500 lines at some point. The safetensors index has 1055 lines total. The decoder keys take up most of the entries. The vision keys and embedding keys should be at the beginning. Let me assume they follow the pattern from the config.

For now, the key point is: the diffusion_gemma index exists and has the weight key layout. The exact key prefixes (model.decoder, model.vision, model.embed_tokens, lm_head, etc.) need to be confirmed from the full index. The config.json has the architecture. The README.md has the generation pattern. The model_index.json points to the HF transformers class DiffusionGemmaForBlockDiffusion.

**Open questions:**

1. Cross-attention mechanism: the README says "decoder applies bidirectional attention over the generation canvas, accessing the cached context via cross-attention." How exactly is the cross-attention wired? Is it a separate cross-attention layer in each decoder block, or is it done via the standard attention with the encoder KV cache as context? Need to check the HF modeling code or the index for cross-attention weight keys.

2. Vision→text interface: how do the 280 vision soft tokens get fed into the text decoder? Are they concatenated with the text tokens as a prefix, or projected through a separate projector? The config says `vision_soft_tokens_per_image: 280` and `use_bidirectional_attention: "vision"`, but the exact wiring isn't in the config alone.

3. Canvas diffusion details: the README gives the high-level pattern (256-token canvas, 48 denoising steps, entropy-based selection, re-noising), but the exact logits→token update rule at each step isn't fully specified. The sampling config (temperature schedule 0.8→0.4, entropy bound 0.1, adaptive stopping at 0.005) is from the README "Best Practices" section. Need the HF modeling code for the exact denoising update.

4. Bidirectional attention mode: `use_bidirectional_attention: "vision"` — this likely means the vision encoder uses bidirectional attention (standard for ViT), and the text decoder uses bidirectional attention over the canvas during diffusion. But the exact attention mask construction isn't in the config.

5. Layer scalar + multiple norm variants: the safetensors keys show `post_feedforward_layernorm`, `post_feedforward_layernorm_1`, `post_feedforward_layernorm_2`, `pre_feedforward_layernorm`, `pre_feedforward_layernorm_2`. This suggests a more complex norm structure than Gemma3's three norms. Need to understand why there are multiple variants.

6. Router details: `router.per_expert_scale` + `router.scale` + `router.proj.weight` — this is a top-k gating with per-expert scaling. The exact gating formula (sigmoid? softmax? with what scale?) isn't in the config.

7. The `gate_up_proj` key (a single fused gate+up projection) vs separate `gate_proj` + `up_proj` — the safetensors index shows `model.decoder.layers.24.experts.gate_up_proj` (fused) in some entries. Need to check if all experts use fused or if some use separate. Actually from the index excerpt, `model.decoder.layers.24.experts.gate_up_proj` is one key — this might be a fused gate+up for the shared experts or a specific expert. Need to verify.

**Local artifacts sufficient for what:**

- Config: YES — config.json on disk, full architecture parameters
- Vision encoder topology: YES — config + README describe Gemma4 vision
- Text decoder topology: PARTIAL — config + index describe blocks, norms, MoE, attention types, but exact forward details (cross-attention wiring, canvas diffusion update rule, multiple norm variants meaning) need HF modeling code
- Generation pattern: YES — README.md has the full pattern (prefill encoder → canvas diffusion → entropy selection → re-noise → adaptive stopping)
- Weight key layout: YES — safetensors index on disk, 1055 lines
- Checkpoint weights: NO — index maps to shards, shards not on disk

**Recommended approach (before HF download):**

1. Create `DiffusionGemmaConfig` parsing the local config.json. Test parse.
2. Create the text decoder block with:
   - GeGLU MLP (gate_proj + up_proj + down_proj, gelu_pytorch_tanh, intermediate_size 2112, moe_intermediate_size 704)
   - Sliding attention (rope_theta 10000, sliding_window 1024, num_key_value_heads 8) for sliding layers
   - Full attention (rope_theta 1e6, partial_rotary_factor 0.25, rope_type proportional, num_key_value_heads 8) for full layers
   - Three+ norms (pre_feedforward, post_attention, post_feedforward, plus variants _1/_2)
   - Layer scalar
   - MoE router with top-k gating + per-expert scale
   - 128 experts + shared experts (need to check shared count — config says num_experts 128, top_k_experts 8, but doesn't specify n_shared_experts; need HF config or index for shared expert keys)
3. Create the vision encoder block with:
   - Patch embedding (16x16 patches, 3 channels → 1152 hidden)
   - 27 transformer layers (rope_theta 100, 16 heads, 1152 hidden, 4304 intermediate)
   - Pooling layer (kernel_size 3, position_embedding_size 10240)
   - standardize (input normalization)
4. Wire the block-diffusion logic:
   - Canvas (256 tokens) with bidirectional self-attention
   - Cross-attention to encoder KV cache (exact wiring TBD — placeholder for now)
   - Diffusion denoising loop with entropy-based token selection
5. Map weights from the safetensors index:
   - Use the config.json + index to map all keys
6. Tests:
   - Config parse from local config.json
   - Vision encoder smoke test (image → vision tokens)
   - Text decoder smoke test (text → logits) with random weights
   - Full forward with random canvas + cross-attention placeholder
   - Numerical parity: wait for HF modeling code or use model_index.json reference to HF transformers DiffusionGemmaForBlockDiffusion

**Definition of done:**
- DiffusionGemmaConfig parses local config.json
- Vision encoder loads + forward (image → 280 vision soft tokens)
- Text decoder block loads + forward (text → logits) with GeGLU + sliding/full attention + MoE + three norms
- Block-diffusion logic: canvas init + denoising loop + entropy selection + re-noise
- Cross-attention to encoder KV cache (needs HF modeling code for exact wiring)
- Full generation pattern from README (prefill + canvas diffusion + adaptive stopping)
- Tests: config parse, vision smoke test, decoder smoke test, canvas diffusion smoke test, numerical parity (wait for HF reference)

**Implementation priority:** Medium — config + index + README are on disk, but the exact cross-attention wiring and canvas diffusion update rule require HF modeling code. Can implement the config + blocks + vision encoder now; defer the exact diffusion loop wiring until HF modeling code is available.

The remaining 11 families still need HF download: cogvlm, deepseek32, and the 7 stubs.### 12. qwen35moe.rs — AUDIT NEEDED

**Current state:** Need to check if this is a pure thin wrapper or has MoE topology.

**What's needed:**
- If pure thin wrapper: check if Qwen3.5-MoE is a Llama-style MoE model. If so, the wrapper might be correct (Qwen3.5-MoE uses Llama-style MoE with SwiGLU). But need to verify.
- If wrong: implement correct MoE topology.

**Open questions:**
- What is Qwen3.5-MoE's exact architecture? Need HF config.
- Does it use SwiGLU or GeGLU?
- How many experts? What is the MoE routing?

**Definition of done:**
- After audit + HF download: either confirm wrapper is correct or implement correct topology
- Load a Qwen3.5-MoE checkpoint
- Prefill + decode produce correct numerics
- Tests: smoke test, config parse test

### 13-19. Stubs — need full implementation from scratch

For each stub, the plan is:
1. HF download for config.json + modeling_*.py + tokenizer files
2. Implement config struct with `from_hf` parser
3. Implement block(s) with correct topology
4. Implement model struct with load + forward + session/decode
5. Implement tests (smoke test with random weights, config parse test)
6. Merge gate: correct numerics vs reference (HF transformers or modeling_*.py)

#### 13. interns2_mobius.rs

- HF model_type: `interns2_mobius` or `internlm2` (need to check)
- Need HF download for config + modeling code
- InternS2-Mobius architecture: need to understand from config + modeling code

#### 14. inkling_small.rs

- HF model_type: `inkling_small`
- Need HF download for config + modeling code
- Inkling-Small architecture: need to understand from config + modeling code

#### 15. minimax_m3.rs

- HF model_type: `minimax_m3`
- Config (from existing file): `num_experts: 32`, `num_experts_per_tok: 4`, MoE with block_sparse_moe
- Need HF download for exact MoE topology + modeling code
- The existing `MINIMAX_M3_TENSOR_KEYS` shows: `model.embed_tokens.weight`, `model.norm.weight`, `lm_head.weight`, `model.layers.{i}.input_layernorm.weight`, `model.layers.{i}.post_attention_layernorm.weight`, `model.layers.{i}.self_attn.q_proj/k_proj/v_proj/o_proj.weight`, `model.layers.{i}.block_sparse_moe.gate.weight`, `model.layers.{i}.block_sparse_moe.experts.{e}.w1/w2/w3.weight`
- So Minimax-M3 uses: Llama-style attention + block_sparse_moe with gate + experts (w1, w2, w3 = gate, up, down)
- The existing `moe_block.rs` might be reusable for the MoE part, but the expert naming (w1, w2, w3) is different from the existing MoE block (gate_proj, up_proj, down_proj).

#### 16. kimi_k3.rs

- HF model_type: `kimi_k3`
- Config (from existing file): MLA-like with `q_lora_rank: 256`, `kv_lora_rank: 512`, `qk_nope_head_dim: 128`, `qk_rope_head_dim: 64`, `v_head_dim: 128`, `num_experts: 64`, `num_experts_per_tok: 6`
- Need HF download for exact MLA + MoE topology + modeling code
- The existing `KIMI_K3_TENSOR_KEYS` shows: `model.embed_tokens.weight`, `model.norm.weight`, `lm_head.weight`, `model.layers.{i}.input_layernorm.weight`, `model.layers.{i}.post_attention_layernorm.weight`, `model.layers.{i}.self_attn.q_a_proj/q_b_proj/kv_a_proj_with_mqa/kv_b_proj/o_proj.weight`, `model.layers.{i}.moe.gate.weight`, `model.layers.{i}.moe.experts.{e}.w1/w2/w3.weight`
- So Kimi-K3 uses MLA (similar to DeepSeek V2) + MoE with gate + experts

#### 17. glm5_2.rs

- HF model_type: `glm5_2`
- Config (from existing file): `num_experts: 64`, `num_experts_per_tok: 8`, MoE with experts
- Need HF download for exact topology + modeling code
- The existing `GLM5_2_TENSOR_KEYS` shows: `transformer.embedding.word_embeddings.weight`, `transformer.output_layer.weight`, `transformer.encoder.final_layernorm.weight`, `transformer.encoder.layers.{i}.input_layernorm.weight`, `model.layers.{i}.post_attention_layernorm.weight`, `model.layers.{i}.self_attention.query_key_value.weight` (FUSED!), `model.layers.{i}.self_attention.dense.weight`, `model.layers.{i}.mlp.gate.weight`, `model.layers.{i}.mlp.experts.{e}.dense_h_to_4h/dense_4h_to_h.weight`
- So GLM-5.2 uses: FUSED QKV (like Falcon), dense (like Falcon), gate + MoE experts (dense_h_to_4h/dense_4h_to_h)
- This is a different topology from Llama — fused QKV + MoE.

#### 18. diffusion_gemma.rs

- HF model_type: `diffusion_gemma`
- Gemma-based diffusion model
- Need HF download for config + modeling code
- Diffusion models have a different forward pass (noise prediction, not next-token prediction). The existing stub returns Unimplemented for both load and forward.

#### 19. delta_net_base.rs

- HF model_type: `delta_net_base` or similar
- Delta-Net architecture: chunked attention with custom Q/K/V/G/B/S tensors
- Need HF download for config + modeling code
- The existing stub returns Unimplemented for both load and forward, with a panic in `new_session`.

## Architecture changes needed in the core crate

### A. Per-head Q/K norm support

For chameleon (and possibly gemma3 which has q_norm/k_norm in safetensors):
- Extend `LlamaBlock` or create a new block type with optional `q_norm: Option<RmsNorm>` and `k_norm: Option<RmsNorm>`
- The forward path must apply these norms per-head after Q/K projections
- This requires reshaping Q and K to (S, num_heads, head_dim), applying norm, reshaping back

### B. Falcon block type

- Create `FalconBlock` with fused QKV projection, `ln_attn`, `ln_mlp`, parallel attention + MLP forward
- The fused QKV requires a single `Linear` that produces Q+K+V concatenated, then split after the projection

### C. M-RoPE support

- For Qwen2-VL, Qwen3-VL, HunYuan-VL: implement M-RoPE with configurable sections
- Qwen2-VL uses non-interleaved M-RoPE: `mrope_section: [16,24,24]`
- Qwen3-VL uses interleaved M-RoPE: `mrope_interleaved: true, mrope_section: [24,20,20]`
- The existing `Rope` implementation needs to be extended or a new `MRope` type created

### D. YARN RoPE scaling

- For DeepSeek V2: implement YARN RoPE scaling with configurable parameters
- The existing `Rope` implementation uses simple RoPE (no scaling). YARN requires frequency scaling with specific parameters.

### E. ViT + projector module

- For Qwen2-VL, Qwen3-VL, HunYuan-VL, CogVLM, Gemma3 (and possibly gemma3n):
- Create a `vision` module with ViT encoder, projector, and vision token handling
- The ViT needs window attention, spatial merging, and configurable depth/heads/patch_size
- The projector maps vision output to text hidden size

### F. MoE expert loading for different naming conventions

- The existing `moe_block.rs` uses `gate_proj/up_proj/down_proj` naming
- Minimax-M3 uses `w1/w2/w3` naming
- Kimi-K3 uses `w1/w2/w3` naming
- GLM-5.2 uses `dense_h_to_4h/dense_4h_to_h` naming
- The MoE loading must handle different naming conventions, either through config-driven key mapping or separate MoE block types

### G. Fused QKV support

- For Falcon, GLM-5.2: create a fused QKV projection that produces Q+K+V in one matrix multiply
- This is a different topology from Llama's separate Q/K/V projections

## Implementation order

### Phase 0: HF downloads

Download configs + modeling code for all missing families:
- cogvlm: `https://huggingface.co/cogvlm/cogvlm-chat-hf/resolve/main/config.json` (or similar)
- gemma3n: `https://huggingface.co/google/gemma-3n/resolve/main/config.json` (or similar)
- wav_tokenizer_dec: `https://huggingface.co/WavTokenizer/WavTokenizer-v2/resolve/main/config.json` (or similar)
- deepseek32: `https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/config.json` (or similar)
- deepseek4: `https://huggingface.co/deepseek-ai/DeepSeek-V4/resolve/main/config.json` (or similar)
- interns2_mobius: `https://huggingface.co/internlm/internlm2_5-7b-research/resolve/main/config.json` (or similar)
- inkling_small: `https://huggingface.co/thinkingmachines/Inkling-Small/resolve/main/config.json`
- minimax_m3: `https://huggingface.co/MiniMaxAI/MiniMax-M3/resolve/main/config.json`
- kimi_k3: `https://huggingface.co/moonshotai/Kimi-K3/resolve/main/config.json`
- glm5_2: `https://huggingface.co/zai-org/GLM-5.2/resolve/main/config.json`
- diffusion_gemma: `https://huggingface.co/google/diffusiongemma-26B-A4B-it/resolve/main/config.json`

### Phase 1: Architecture changes (core crate)

1. Per-head Q/K norm support in `block.rs` or new block type
2. Falcon block type in new `falcon_block.rs` or extend `block.rs`
3. M-RoPE support in `grim_nn::Rope` or new `MRope` type
4. YARN RoPE scaling in `Rope` or new `YarnRope` type
5. ViT + projector module in new `vision/` module
6. MoE expert loading for different naming conventions in `moe_block.rs`

### Phase 2: Model file implementations

Implement each of the 19 files, starting with the ones that have local configs available:
1. chameleon.rs (local config available)
2. deepseek2.rs (local config available)
3. falcon.rs (local config available)
4. qwen2vl.rs (local config available)
5. qwen3vl.rs (local config available)
6. hunyuan_vl.rs (local config available)
7. gemma3n.rs (needs HF download)
8. cogvlm.rs (needs HF download)
9. wav_tokenizer_dec.rs (needs HF download)
10. deepseek32.rs (needs HF download)
11. deepseek4.rs — WRONG, needs real implementation (FULL LOCAL REFERENCE ON DISK — NOT HF-GATED)

### 11. deepseek4.rs — WRONG, needs real implementation

**Current state:** Thin `Llama` wrapper.

**What's needed:**
The `Deepseek4 Flash/` folder on disk now has:
- `model.py` (961 lines — full DeepSeekV4 reference implementation: Transformer, Block, Attention, MoE, Gate, Expert, Compressor, Indexer, Hyper-Connections, DSpark, ParallelEmbedding, ParallelHead, RMSNorm, Linear with fp4/fp8 dispatch)
- `kernel.py` (536 lines — TileLang kernel definitions: act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn)
- `encoding_dsv4.py` (760 lines — chat message encoding: special tokens, message rendering, tool calling format)
- `convert.py` (155 lines — HF safetensors → sharded checkpoint converter with expert sharding and FP4/FP8 conversion)
- `generate.py` (145 lines — batch generation: prefill phase + decode phase + DSpark speculative)
- `jang_config.json` (JANG quantization metadata: method, profile, target_bits, actual_bits, block_size, calibration, scoring)
- `config.json` (HF model config — model_type, architectures, text_config with all architecture params)
- `generation_config.json`
- `tokenizer_config.json` + `tokenizer.json` (129280 vocab)

All the architecture knowledge is on disk. The only thing missing is the actual weight shard files (which model.safetensors.index.json points to but doesn't contain). No HF download needed to understand and implement the architecture.

**DeepSeek V4 architecture from the local files (model.py + config.json + kernel.py):**

Core model (from model.py `ModelArgs`):

- vocab_size: 129280
- dim: 4096
- moe_inter_dim: 4096
- n_layers: 43
- n_hash_layers: 0 (no hash-based expert routing)
- n_mtp_layers: 1 (DSpark/MTP stage)
- n_heads: 64
- n_routed_experts: 256
- n_shared_experts: 1
- n_activated_experts: 6
- score_func: sqrtsoftplus
- route_scale: 1.5
- swiglu_limit: 7.0
- q_lora_rank: 1024
- head_dim: 512
- rope_head_dim: 64
- nope_head_dim: 448 (head_dim - rope_head_dim)
- norm_eps: 1e-6
- o_groups: 8
- o_lora_rank: 1024
- window_size: 128
- compress_ratios: tuple (0, 0, 4, 128, 4, 128, 4, 0, ...) — per-layer, 0 = pure sliding window, 4 = overlap KV compression (ratio 4 with overlap), 128 = heavy compression (ratio 128)
- compress_rope_theta: 40000.0
- original_seq_len: 0 (set at runtime)
- rope_theta: 10000.0
- rope_factor: 40
- beta_fast: 32
- beta_slow: 1
- index_n_heads: 64
- index_head_dim: 128
- index_topk: 512
- hc_mult: 4
- hc_sinkhorn_iters: 20
- hc_eps: 1e-6
- dspark_block_size: 0 (disabled unless set)
- dspark_target_layer_ids: tuple()
- dspark_markov_rank: 256
- dtype: fp8 (FP8 weights for some tensors)
- scale_fmt: ue8m0
- scale_dtype: fp8

**Key architectural features:**

1. **Hyper-Connections (HC):** Instead of standard residual, the hidden state is expanded to hc_mult=4 copies, then mixed via learned Sinkhorn-normalized projections before and after each sublayer. `hc_attn_fn` and `hc_ffn_fn` are (mix_hc x hc_dim) matrices where mix_hc=(2+hc_mult)*hc_mult=24. `hc_pre` produces pre/post/comb (b, s, hc_mult, hc_dim). `hc_post` combines the sublayer output with the residual in HC space.

2. **Multi-Head Latent Attention (MLA) with compression:** Standard MLA (wq_a → q_norm → wq_b, wkv → kv_norm → Kv latent) PLUS a Compressor that compresses the KV cache at varying ratios per layer. Layers with compress_ratio=4 use overlapping windows; layers with compress_ratio=128 compress 128 tokens into 1. The Indexer is a separate learned mechanism that selects top-k KV positions for attention using its own Compressor with Hadamard rotation on the queries.

3. **FP4 quantized experts:** The experts are stored in float4_e2m1fn_x2 format (2 FP4 values per byte, packed along K). The weight shape is [out, in//2] in this dtype; scale is [out, in//32] in float8_e8m0fnu. The kernel.py fp4_gemm handles FP8-act x FP4-weight GEMM. Alternatively, convert.py can dequantize to BF16.

4. **FP8 activations:** act_quant with block_size=64 or 128 quantizes activations to FP8 (float8_e4m3fn) with per-block E8M0 scales. inplace=True does fused quant+dequant back to BF16.

5. **sqrtsoftplus gating:** The MoE gate uses `F.softplus(scores).sqrt()` instead of softmax or sigmoid.

6. **Sparse attention (FlashAttention-style):** The `sparse_attn` kernel does online-softmax FlashAttention with top-k index gathering, including a learnable `attn_sink` bias per head. The kernel pads heads to 16 for efficiency.

7. **YaRN rotary scaling:** Standard YaRN with the parameters above (rope_factor=40, original_max_position_embeddings=4096, beta_fast=32, beta_slow=1, compress_rope_theta=40000).

8. **DSpark speculative decoding stage:** After the main 43 layers, an optional MTP/DSpark stage (controlled by dspark_block_size > 0) does block-wise speculative decoding with a Markov head and confidence head. If dspark_block_size=0, this stage is disabled.

9. **Generation:** generate.py shows the batch generation loop: prefill processes [min_prompt_len:] tokens, decode generates one token at a time with left-padding. DSpark draft uses forward_spec for speculative tokens.

**Weight layout (from model.py + convert.py + model.safetensors.index.json):**

Main model weights (model.layers.{i}.* namespace):
- self_attn.wq_a.weight, wq_a.scale (FP8 scale if FP8)
- self_attn.wq_b.weight (ColumnParallelLinear, sharded along output dim)
- self_attn.q_norm.weight, q_norm.bias (RMSNorm)
- self_attn.wkv.weight, wkv.scale
- self_attn.kv_norm.weight, kv_norm.bias (RMSNorm)
- self_attn.wo_a.weight, wo_a.scale (ColumnParallelLinear, FP8, then unflattened in convert.py to (o_groups, o_lora_rank, head_dim) for dequantization)
- self_attn.wo_b.weight (RowParallelLinear)
- self_attn.attn_sink (register_buffer, FP32 per-head sink bias)
- self_attn.freqs_cis (register_buffer, precomputed YaRN cis)
- self_attn.kv_cache (register_buffer)
- self_attn.compressor.{compress_ratio}.* (if compress_ratio > 0)
- self_attn.indexer.{indexer params}.* (if compress_ratio == 4)

FFN weights (ffn.* namespace):
- ffn.gate.weight, gate.scale (Linear or FP4)
- ffn.up.weight, up.scale
- ffn.down.weight, down.scale
- ffn.gate.bias (score bias, FP32)
- ffn.gate.tid2eid (FP32, per-token expert table for hash routing — empty if n_hash_layers=0)
- ffn.gate.e_score_correction_bias (per-expert correction, FP32)
- ffn.shared_experts.gate/up/down.weight (+ scales if FP4)
- ffn.experts.{e}.w1.weight (+ w1.scale), w2.weight (+ w2.scale), w3.weight (+ w3.scale) — FP4 quantized if expert_dtype=fp4

HC weights:
- hc_attn_fn.weight (mix_hc x hc_dim = 24 x 16384, FP32)
- hc_attn_base.weight (mix_hc = 24, FP32)
- hc_attn_scale.weight (3, FP32)
- hc_ffn_fn.weight, hc_ffn_base.weight, hc_ffn_scale.weight (same shapes)

Other:
- embed.weight (ParallelEmbedding, sharded along vocab dim)
- embed.weight_scale, embed.weight_scale_inv (if FP8)
- norm.weight (RMSNorm, FP32)
- head.weight (ParallelHead, sharded along vocab dim, FP32)

**Convert.py→weight loading mapping:**

The convert.py script documents exactly how HF safetensors keys map to the local sharded format:
- `model.embed` → `embed.weight`
- `model.self_attn.wq_b` → `wq_b.weight`
- `model.self_attn.wo_a` → `wo_a.weight` (dequantized in the script: unflatten to (o_groups, 128, -1, 128), multiply scale, flatten back, convert to BF16)
- `model.self_attn.wo_b` → `wo_b.weight` (sharded along input dim)
- `model.head` → `head.weight` (sharded along vocab dim)
- `model.head` scale → `head.weight_scale`
- `model.attn_sink` → registered buffer (no .weight suffix)
- `model.hc_*` → hc_attn_fn/hc_ffn_fn/hc_attn_base/hc_ffn_base/hc_attn_scale/hc_ffn_scale
- `model.ffn.gate/up/down` → gate/up/down (with FP4/FP8 scale handling)
- `model.ffn.shared_experts.{gate,up,down}` → shared_experts.{gate,up,down}
- `model.ffn.experts.{e}.{w1,w2,w3}` → experts.{e}.{gate,up,down} (FP4 if int8, else BF16)
- `model.ffn.gate.bias` → gate.bias
- `model.ffn.gate.tid2eid` → gate.tid2eid
- `model.ffn.gate.e_score_correction_bias` → gate.e_score_correction_bias

**JANG quantization details (jang_config.json):**

- method: jang-importance
- profile: CUSTOM_8_4_3
- target_bits: 3, actual_bits: 3.36
- block_size: 128
- calibration_method: weights
- quantization_method: mse-all
- scoring_method: weight-magnitude
- bit_widths_used: [3, 4, 8]
- quantization_scheme: asymmetric
- quantization_backend: mx.quantize
- hadamard_rotation: false
- source_model: DeepSeek-V4-Flash, bfloat16, 3.4B parameters
- runtime: total_weight_bytes 31778176 (~30MB), total_weight_gb 0.03
- capabilities: reasoning_parser deepseek_r1, tool_parser deepseek, think_in_template true, supports_tools true, supports_thinking true, family deepseek_v4, modality text, cache_type mla
- format: jang, format_version: 2.0

Wait — "total_weight_gb: 0.03" with 3.4B parameters? That means the weights are heavily quantized. 3.4B params * 3 bits / 8 bits per byte ≈ 1.3 GB if purely 3-bit. 30 MB is too small even for that. Let me check — the jang_config total_weight_bytes is 31778176 = 30.3 MB. That's the metadata/prefix bytes, not the full weights. The model.safetensors.index.json total_size is 1.5 TB, which is the full unquantized model size. So the jang_config describes the quantization metadata, not the weights themselves.

Actually, looking again: total_weight_gb 0.03 → 30 MB total. But model.safetensors.index.json says total_size 1.5 TB. There's a huge discrepancy. The jang_config might be describing only the quantization tables/metadata, not the weights. Or the model.safetensors.index.json is wrong/outdated. The tokenizer.json is 6.3 MB, model.safetensors.index.json is 5.6 MB — these are metadata files. The actual weight shards would be large.

Wait, 1.5 TB for what — let me check the safetensors index more carefully.

Actually, looking at the model.safetensors.index.json: total_size 1500265952 (about 1.5 GB, not 1.5 TB — I misread earlier). 1.5 GB for 3.4B parameters at 3.36 bits/param ≈ 3.4e9 * 3.36 / 8 = 1.4 GB. That matches. So the model is ~1.5 GB on disk in JANG-quantized format. The safetensors index maps 48 shards (model-00001-of-00048.safetensors through model-00048-of-00048.safetensors).

So the model weights ARE on disk — the model.safetensors.index.json references 48 shard files, and the total_size 1.5 GB represents the actual model. The shard files themselves are likely present if this is a real checkpoint directory. Let me verify.

**What the local files enable:**

1. Full architecture understanding — from model.py (reference impl) + config.json + kernel.py
2. Weight loading mapping — from convert.py + model.safetensors.index.json
3. FP4/FP8 quantization handling — from kernel.py + convert.py + jang_config.json
4. Tokenization — from tokenizer.json + encoding_dsv4.py
5. Generation — from generate.py + encoding_dsv4.py

**What still requires HF download (if local shard files are missing):**

If the model-*.safetensors shard files are NOT in the directory, the actual weight values are missing. The index.json tells us the names and shard assignments but not the values. In that case, we can still implement the architecture (config + block + forward) without weights, but actual numerical verification requires either the local shard files or HF download.

Let me check if the shard files are actually present.

Let me check that — and while I'm at it, read the key files I need for the plan. I'll batch the shard-file check with reads of the deepseek4 config and the diffusiongemma config.
12. qwen35moe.rs (audit + needs HF download if wrong)
13. interns2_mobius.rs (needs HF download)
14. inkling_small.rs (needs HF download)
15. minimax_m3.rs (needs HF download — config partially known from existing file)
16. kimi_k3.rs (needs HF download — config partially known from existing file)
17. glm5_2.rs (needs HF download — config partially known from existing file)
### diffusion_gemma.rs (needs HF download — BUT local DiffusionGemma artifacts on disk at old/mods/diffusiongemma/ — config.json + model.safetensors.index.json + model_index.json + docs)
19. delta_net_base.rs (needs HF download)

### Phase 3: Tests

For each implemented file:
1. Config parse test (parse real config.json from HF)
2. Smoke test (load model with random weights, run forward, check output shape)
3. Numerical parity test (if reference implementation available — HF transformers, modeling_*.py, or another verified implementation)

### Phase 4: Integration

1. Update `lib.rs` to export the new model types
2. Update model loader/router to dispatch to the new implementations
3. End-to-end test: load a real checkpoint, run prefill + decode, verify output

## Testing strategy

### Per-file tests

Each model file gets:
- `config_parse_test`: parse the real config.json from HF, verify all fields
- `smoke_test`: create model with random weights (using `Model::random` or equivalent), run forward, check output shape and non-zero logits
- `load_test`: if weights available, load from safetensors/GGUF and verify all weights loaded correctly
- `forward_test`: if reference available, compare forward output vs reference

### Numerical parity

For numerical parity, we need a reference implementation. Options:
1. HuggingFace transformers implementations (Python) — run the same input through HF transformers and compare
2. Modeling_*.py files from HF repos — these are the reference implementations
3. llama.cpp implementations — for Llama-style models, llama.cpp is a reference
4. Cross-check with other Rust implementations (e.g. burn, peregrine)

For models with local modeling code (deepseek2 has `modeling_deepseek.py`), we can use that as the reference.

### Tooling

- Use `cargo test` for Rust tests
- Use Python scripts (with `transformers` library) for HF reference comparisons
- For GGUF loading tests, use the existing GGUF loader in `grim-format`

## Merge gates

Each file is mergeable when:
1. Config parse test passes with real HF config.json
2. Smoke test passes (random weights, correct output shape, non-zero logits)
3. Load test passes (if weights available — all weights loaded, no missing keys)
4. Forward test passes (if reference available — output matches reference within tolerance)
5. Code review passes (correct topology, correct weight mapping, correct forward math)

## Open questions and risks

1. **Chameleon multimodal**: Does the implementation need VQGAN for images, or is text-only sufficient? The safetensors index shows only transformer weights. If text-only, the model loads and runs but is not the full Chameleon experience.

2. **Falcon embedding key**: Need to read the full safetensors index to find the embedding key name. The excerpt shows `lm_head.weight` at the top, but the embedding key is not in the excerpt.

3. **Qwen2-VL / Qwen3-VL projector**: Need to check if the projector weights are in the safetensors files or stored separately. The config says `out_hidden_size` matches `hidden_size`, so the projector might be a no-op.

4. **Gemma3 vs gemma3n**: The `old/mods/gemma3/` folder is gemma3 (VLM), not gemma3n. Gemma3n is a different model. Need HF download for gemma3n config.

5. **DeepSeek V2 YARN**: The existing `Rope` implementation doesn't have YARN. YARN requires frequency scaling with specific parameters. Need to implement YARN or use a different RoPE implementation.

6. **Qwen M-RoPE**: The existing `Rope` implementation doesn't have M-RoPE. Need to implement M-RoPE (both interleaved and non-interleaved variants).

7. **MoE expert naming**: Different models use different naming for MoE experts (gate_proj/up_proj/down_proj vs w1/w2/w3 vs dense_h_to_4h/dense_4h_to_h). The MoE loading must handle all conventions.

8. **Weight availability**: Most folders in `old/mods/` have config + index but not the actual weight files. To run real load tests, we need the weight files (safetensors or GGUF). If weights are not available, we can only do smoke tests with random weights.

## Appendix A: HF download URLs for missing configs

### cogvlm
- Config: `https://huggingface.co/cogvlm/cogvlm-chat-hf/resolve/main/config.json`
- Model type: `cogvlm`
- Alternative: `https://huggingface.co/cogvlm/cogvlm2-chat-hf/resolve/main/config.json`

### gemma3n
- Config: `https://huggingface.co/google/gemma-3n/resolve/main/config.json`
- Model type: `gemma3n` (expected)
- Alternative: check `https://huggingface.co/google/gemma-3n-it/resolve/main/config.json`

### wav_tokenizer_dec
- Config: `https://huggingface.co/WavTokenizer/WavTokenizer-v2/resolve/main/config.json`
- Model type: `wavtokenizer` (expected)
- Alternative: `https://huggingface.co/WavTokenizer/WavTokenizer-v1/resolve/main/config.json`

### deepseek32 (DeepSeek V3)
- Config: `https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/config.json`
- Model type: `deepseek_v3` (expected)
- Alternative: `https://huggingface.co/deepseek-ai/DeepSeek-V3-0324/resolve/main/config.json`

### deepseek4 (DeepSeek V4)
- Config: `https://huggingface.co/deepseek-ai/DeepSeek-V4/resolve/main/config.json`
- Model type: `deepseek_v4` (expected)
- Note: V4 may not be publicly available yet. Check if config exists.

### interns2_mobius
- Config: `https://huggingface.co/internlm/internlm2_5-7b-resolve/main/config.json` (expected model_type: `internlm2` or `interns2_mobius`)
- Alternative: `https://huggingface.co/internlm/Intern-S2-Mobius/resolve/main/config.json`

### inkling_small
- Config: `https://huggingface.co/thinkingmachines/Inkling-Small/resolve/main/config.json`
- Model type: `inkling_small`

### minimax_m3
- Config: `https://huggingface.co/MiniMaxAI/MiniMax-M3/resolve/main/config.json`
- Model type: `minimax_m3`
- Already partially known from existing file

### kimi_k3
- Config: `https://huggingface.co/moonshotai/Kimi-K3/resolve/main/config.json`
- Model type: `kimi_k3`
- Already partially known from existing file

### glm5_2
- Config: `https://huggingface.co/zai-org/GLM-5.2/resolve/main/config.json`
- Model type: `glm5_2`
- Already partially known from existing file

### diffusion_gemma
- Config: `https://huggingface.co/google/diffusiongemma-26B-A4B-it/resolve/main/config.json`
- Model type: `diffusion_gemma`
- Already partially known from existing file

## Appendix B: Existing reference implementations in the crate

The crate already has implementations for some architectures that can serve as references:

- `gemma.rs`: Gemma with GeGLU, two norms (attn_norm, ffn_norm), full attention. NOT gemma3 (which has three norms + sliding attention + SigLIP vision).
- `deepseek.rs`: DeepSeek with MLA (q_a_proj, q_b_proj, kv_a_proj, kv_b_proj), no MoE, no YARN, head_dim=128. Close to V2 but not identical.
- `falcon_h1.rs`: Falcon-H1 with hybrid Mamba-2 + GQA attention + SwiGLU. NOT Falcon (which has fused QKV + parallel attention + two norms).
- `qwen35.rs`: Qwen3.5 hybrid SSM + GQA attention + SwiGLU. NOT Qwen3.5-MoE (which is a different architecture).
- `gpt2.rs`: GPT-2 with LayerNorm, fused QKV (wqkv), GELU, absolute position embeddings.
- `block.rs`: LlamaBlock with GQA attention, SwiGLU, two norms (attn_norm, ffn_norm).
- `model.rs`: Llama with LlamaBlock, MoE support via MoeBlock.
- `moe_block.rs`: MoE block with router + expert bank, supports different router kinds (softmax, sigmoid+bias), shared experts.

These existing implementations provide reference patterns for:
- Config structs with `from_hf` parsers
- Block loading from `WeightSource`
- Forward paths with cache-aware attention
- Session/decode handling
- MoE routing (for minimax_m3, kimi_k3, glm5_2 which have MoE)

## Appendix C: Weight mapping summary for local configs

### chameleon weight mapping
```
model.layers.{i}.self_attn.q_proj.weight          → wq
model.layers.{i}.self_attn.k_proj.weight          → wk
model.layers.{i}.self_attn.v_proj.weight          → wv
model.layers.{i}.self_attn.o_proj.weight          → wo
model.layers.{i}.self_attn.q_norm.weight          → q_norm.weight
model.layers.{i}.self_attn.q_norm.bias            → q_norm.bias
model.layers.{i}.self_attn.k_norm.weight          → k_norm.weight
model.layers.{i}.self_attn.k_norm.bias            → k_norm.bias
model.layers.{i}.input_layernorm.weight           → attn_norm.weight
model.layers.{i}.post_attention_layernorm.weight  → ffn_norm.weight
model.layers.{i}.mlp.gate_proj.weight             → w_gate.weight
model.layers.{i}.mlp.up_proj.weight               → w_up.weight
model.layers.{i}.mlp.down_proj.weight             → w_down.weight
model.embed_tokens.weight                         → tok_embeddings.weight
lm_head.weight                                    → output.weight
```

### deepseek2 weight mapping
```
model.layers.{i}.self_attn.q_proj.weight                 → q_proj (Q latent)
model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight     → kv_a_proj_with_mqa (KV latent)
model.layers.{i}.self_attn.kv_a_layernorm.weight         → kv_a_layernorm
model.layers.{i}.self_attn.kv_b_proj.weight              → kv_b_proj (K/V expansion)
model.layers.{i}.self_attn.o_proj.weight                 → wo
model.layers.{i}.input_layernorm.weight                  → attn_norm
model.layers.{i}.post_attention_layernorm.weight         → ffn_norm
model.layers.{i}.mlp.gate_proj.weight                    → w_gate
model.layers.{i}.mlp.up_proj.weight                      → w_up
model.layers.{i}.mlp.down_proj.weight                    → w_down
model.layers.{i}.mlp.shared_experts.gate_proj.weight     → shared_experts.gate
model.layers.{i}.mlp.shared_experts.up_proj.weight       → shared_experts.up
model.layers.{i}.mlp.shared_experts.down_proj.weight     → shared_experts.down
model.layers.{i}.mlp.experts.{e}.gate_proj.weight        → experts[{e}].gate
model.layers.{i}.mlp.experts.{e}.up_proj.weight          → experts[{e}].up
model.layers.{i}.mlp.experts.{e}.down_proj.weight        → experts[{e}].down
model.embed_tokens.weight                                → tok_embeddings
model.norm.weight                                        → norm
lm_head.weight                                           → output
```

### falcon weight mapping (partial — need full index for embedding key)
```
transformer.h.{i}.self_attention.query_key_value.weight  → fused_qkv
transformer.h.{i}.self_attention.dense.weight            → wo
transformer.h.{i}.ln_attn.weight                          → ln_attn.weight
transformer.h.{i}.ln_attn.bias                            → ln_attn.bias
transformer.h.{i}.ln_mlp.weight                           → ln_mlp.weight
transformer.h.{i}.ln_mlp.bias                             → ln_mlp.bias
transformer.h.{i}.mlp.dense_h_to_4h.weight               → w_up (or just up_proj)
transformer.h.{i}.mlp.dense_4h_to_h.weight               → w_down
lm_head.weight                                            → output
[tbc] transformer.word_embeddings.weight?                 → tok_embeddings
```

### qwen2-vl / qwen3-vl weight mapping (text transformer part)
```
model.embed_tokens.weight                                 → tok_embeddings
model.layers.{i}.input_layernorm.weight                  → attn_norm
model.layers.{i}.self_attn.q_proj.weight                 → wq
model.layers.{i}.self_attn.k_proj.weight                 → wk
model.layers.{i}.self_attn.v_proj.weight                 → wv
model.layers.{i}.self_attn.o_proj.weight                 → wo
model.layers.{i}.post_attention_layernorm.weight         → ffn_norm
model.layers.{i}.mlp.gate_proj.weight                    → w_gate
model.layers.{i}.mlp.up_proj.weight                      → w_up
model.layers.{i}.mlp.down_proj.weight                    → w_down
model.norm.weight                                         → norm
lm_head.weight                                            → output
```

Plus ViT weights and projector weights (need to check exact naming from full safetensors index).

## Done

This plan is complete enough to start implementation. 
