# Q7 Design Note: Two-Device Placement for Qwen3.8-Flash-Next

## 1. Problem & Memory Footprint
`Qwen38FlashNext` currently loads all 48 layers onto a single `Device`. On a 2×16 GB workstation (e.g., RX 9070 XT 16 GB + RX 9060 XT 16 GB), the 45 GB `Qwen3.8-Flash-Next-GSQ-RCO-3.5bit.gguf` checkpoint + 28 GB PLE n-gram table cannot reside on one GPU without expert offload:
- **Dense/shared trunk** (embeddings, `lm_head` `[248320, 2560]`, 36 GDN projections/states, 12 QSA projections/KV, routers, shared experts): fits comfortably across 2 GPUs with headroom for KV cache and staging slabs.
- **Routed experts** (48 layers × 512 experts × 3 projections): remains host-backed (`Qwen38MoeExperts::Offloaded` / `ExpertOffloadBanks`) with bounded per-layer device staging cache governed by `UseCappedResidentPolicy` (Q5).
- **PLE n-gram table** (28 GB): stays on host RAM (lookup + projection outside decode graph capture).

## 2. Proposed Placement Strategy (Pipeline Split, No TP2)
Mirror `qwen35.rs::plan_layer_devices` (`crates/grim-models/transformer/src/qwen35.rs:7489-7610`) with **hidden-state-only crossing** between GPU 0 and GPU 1:
1. **Reserve accounting on output GPU (last device):**
   - `tok_embeddings` (if tied or on dev0) and `lm_head` (`[248320, 2560]`).
   - `nextn_reserve_bytes` for optional MTP/NextN head (matching `qwen35.rs:7577-7582`).
   - Working scratch & `GRIM_ALLOC_POOL=1` slab headroom (Q0 showed unpooled `staged_to` F32 dequant leaks ~2.4 GiB/step without `GRIM_ALLOC_POOL=1`).
2. **Per-layer byte accounting:**
   - GDN layers (36): dense projections + `ssm_state` (`48 × 128 × 128 × 4 B = 3.0 MiB/layer`) + `conv_state` (`10240 × 3 × 4 B = 120 KiB/layer`) + bounded expert staging budget.
   - QSA layers (12): dense projections + indexer weights + KV cache (`2 × 2 × 256 × ctx × dtype`) + bounded expert staging budget.
3. **Explicit override:** support `GRIM_SPLIT_LAYERS="a,b"` (matching `qwen35.rs:7507`) for deterministic A/B and debugging.
4. **Cross-device boundary:** single P2P/host-staged transfer of `[seq_len, 2560]` (`10 KiB` at `seq_len=1` decode) at the split layer. **Do NOT reopen TP2** (`plans/PLAN-tp2-decode.md` measured and closed).

## 3. Blockers & Prerequisites Before Implementation
- Real-GGUF loader shape/quant mismatch (`qwen4exp_weight_loading_test`, `plans/TODO-open-items.md` §C2) must be resolved so real-checkpoint `Qwen38MoeExperts::Offloaded` memory sizing can be measured on hardware.
- Owner review of this design note before adding `plan_flash_next_layer_devices` code.
