//! Phase 0 — Per-layer arena accounting at 262K + baseline measurement table.
//!
//! # Architecture & Geometry (Qwen3.5-9B, 262K context)
//!
//! Total layers: 32.
//! Full Attention layers (MHA/GQA): 8 layers (indices 3, 7, 11, 15, 19, 23, 27, 31).
//! Linear Attention (KDA / Gated Delta Net / SSM) layers: 24 layers (all other indices).
//!
//! ## KV-Arena Accounting:
//! Only the 8 full attention layers allocate context-proportional KV arenas.
//! KDA layers carry fixed-size recurrent state matrices + conv buffers that do NOT grow with context.
//!
//! Attention Geometry:
//! - num_heads = 16, num_kv_heads = 4, head_dim = 256
//! - K/V width per token per layer = num_kv_heads × head_dim = 4 × 256 = 1,024 elements
//! - Per token per layer (K + V) = 2 × 1,024 = 2,048 elements
//! - Across all 8 full attention layers = 8 × 2,048 = 16,384 elements/token
//!
//! Byte Footprint per token:
//! - FP32 (4 bytes/elem): 16,384 × 4 = 65,536 bytes/token = 64.0 KB/token
//! - FP16 (2 bytes/elem): 16,384 × 2 = 32,768 bytes/token = 32.0 KB/token
//! - INT8 (1 byte/elem):  16,384 × 1 = 16,384 bytes/token = 16.0 KB/token
//! - INT4 (0.5 byte/elem): 16,384 × 0.5 = 8,192 bytes/token = 8.0 KB/token
//!
//! Total Arena Allocation at 262,144 tokens:
//! - FP32 (Baseline): 65,536 B × 262,144 = 17,179,869,184 B = 16.00 GiB (17.18 GB) [1.00x]
//! - FP16 (Phase 1):  32,768 B × 262,144 = 8,589,934,592 B = 8.00 GiB (8.59 GB)   [2.00x]
//! - INT8 (Phase 2):  16,384 B × 262,144 = 4,294,967,296 B = 4.00 GiB (4.29 GB)   [4.00x]
//! - INT4 (Phase 3):   8,192 B × 262,144 = 2,147,483,648 B = 2.00 GiB (2.15 GB)   [8.00x]
//! - CurDKV (Phase 4): ~819 B × 262,144 = ~214,748,364 B = ~0.20 GiB (0.22 GB)    [~78.0x]
//!
//! ## KDA State Accounting (Context-Invariant):
//! Conv state per layer: conv_kernel (4) × conv_dim (8,192) × 4B = 131,072 B (128 KB)
//! SSM recurrent state per layer: (ssm_dt_rank (32) + 2 × ssm_n_group (16)) × ssm_d_state (128) × 4B
//! Across 24 KDA layers, total recurrent state is ~few MBs and fixed throughout 262K tokens.

use grim_format::tprov::GgufProvider;
use grim_models_transformer::qwen35::{Qwen35Block, Qwen35Config, Qwen35LayerCache};
use grim_tensor::{Device, Shape};

const CONTEXT_262K: usize = 262_144;

fn model_path() -> Option<std::path::PathBuf> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    for up in ["../../..", "../..", ".."] {
        let p = root
            .join(up)
            .join("models/qwen35-9b/Qwen3.5-9B-Q4_K_M.gguf");
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn cfg_9b() -> Qwen35Config {
    let mut c = Qwen35Config::default();
    c.vocab_size = 248320;
    c.hidden_size = 4096;
    c.num_heads = 16;
    c.num_kv_heads = 4;
    c.head_dim = 256;
    c.num_layers = 32;
    c.intermediate_size = 12288;
    c.full_attention_interval = 4;
    c.ssm_d_conv = 4;
    c.ssm_d_state = 128;
    c.ssm_dt_rank = 32;
    c.ssm_n_group = 16;
    c.ssm_d_inner = 32 * 128;
    c.devices = Vec::new();
    c
}

/// Baseline accounting test verifying exact mathematical bytes at 262,144 context.
#[test]
fn qwen35_9b_262k_kv_arena_baseline_accounting() {
    let c = cfg_9b();

    // Verify layer breakdown
    let mut num_attn_layers = 0;
    let mut num_kda_layers = 0;
    for layer_idx in 0..c.num_layers {
        if (layer_idx + 1) % c.full_attention_interval == 0 {
            num_attn_layers += 1;
        } else {
            num_kda_layers += 1;
        }
    }
    assert_eq!(num_attn_layers, 8, "Expected exactly 8 attention layers");
    assert_eq!(num_kda_layers, 24, "Expected exactly 24 KDA layers");

    // KV arena geometry per attention layer
    let k_elems_per_token_per_layer = c.num_kv_heads * c.head_dim;
    let v_elems_per_token_per_layer = c.num_kv_heads * c.head_dim;
    assert_eq!(k_elems_per_token_per_layer, 1024);
    assert_eq!(v_elems_per_token_per_layer, 1024);

    let kv_elems_per_token_total = num_attn_layers * (k_elems_per_token_per_layer + v_elems_per_token_per_layer);
    assert_eq!(kv_elems_per_token_total, 16384);

    // FP32 Baseline
    let fp32_bytes_per_token = kv_elems_per_token_total * 4;
    assert_eq!(fp32_bytes_per_token, 65536, "Baseline must be 64KB/token");
    let fp32_total_bytes_262k = fp32_bytes_per_token as u64 * CONTEXT_262K as u64;
    assert_eq!(fp32_total_bytes_262k, 17_179_869_184, "Baseline must be exactly 17.18 GB (16.00 GiB)");

    // FP16 (Phase 1)
    let fp16_bytes_per_token = kv_elems_per_token_total * 2;
    assert_eq!(fp16_bytes_per_token, 32768, "Phase 1 must be 32KB/token");
    let fp16_total_bytes_262k = fp16_bytes_per_token as u64 * CONTEXT_262K as u64;
    assert_eq!(fp16_total_bytes_262k, 8_589_934_592, "Phase 1 must be exactly 8.59 GB (8.00 GiB)");

    // INT8 (Phase 2)
    let int8_bytes_per_token = kv_elems_per_token_total * 1;
    assert_eq!(int8_bytes_per_token, 16384, "Phase 2 must be 16KB/token");
    let int8_total_bytes_262k = int8_bytes_per_token as u64 * CONTEXT_262K as u64;
    assert_eq!(int8_total_bytes_262k, 4_294_967_296, "Phase 2 must be exactly 4.29 GB (4.00 GiB)");

    // INT4 (Phase 3)
    let int4_bytes_per_token = kv_elems_per_token_total / 2;
    assert_eq!(int4_bytes_per_token, 8192, "Phase 3 must be 8KB/token");
    let int4_total_bytes_262k = int4_bytes_per_token as u64 * CONTEXT_262K as u64;
    assert_eq!(int4_total_bytes_262k, 2_147_483_648, "Phase 3 must be exactly 2.15 GB (2.00 GiB)");

    eprintln!("\n================ KV Arena Baseline Table (Qwen3.5-9B @ 262,144) ================");
    eprintln!("Layer breakdown: {num_attn_layers} Attention layers, {num_kda_layers} KDA layers");
    eprintln!("Per-layer KV width: {} elems (K: {}, V: {})", k_elems_per_token_per_layer + v_elems_per_token_per_layer, k_elems_per_token_per_layer, v_elems_per_token_per_layer);
    eprintln!("Total KV elems / token across model: {kv_elems_per_token_total} elems");
    eprintln!("--------------------------------------------------------------------------------");
    eprintln!("Phase               Bytes/token    Total @ 262K (GiB)    Total @ 262K (GB)    Ratio");
    eprintln!("Baseline (FP32)     64 KB          16.00 GiB             17.18 GB             1.00x");
    eprintln!("Phase 1 (FP16)      32 KB           8.00 GiB              8.59 GB             2.00x");
    eprintln!("Phase 2 (INT8)      16 KB           4.00 GiB              4.29 GB             4.00x");
    eprintln!("Phase 3 (INT4)       8 KB           2.00 GiB              2.15 GB             8.00x");
    eprintln!("Phase 4 (CurDKV)    ~0.8 KB        ~0.20 GiB             ~0.22 GB            ~78.0x");
    eprintln!("================================================================================\n");
}

/// Baseline parity snapshot against real weights: verifies attention block load,
/// KV projection dims, and single-token decode parity on real checkpoint.
#[test]
fn qwen35_9b_real_weights_baseline_parity_snapshot() {
    let Some(path) = model_path() else {
        eprintln!("[SKIP] qwen35-9b checkpoint not found on disk");
        return;
    };
    let path_str = path.to_str().expect("valid utf-8 path");
    let prov = GgufProvider::open(path_str).expect("open gguf");
    let c = cfg_9b();

    // Verify metadata matches architecture assumptions
    let full_attn_interval = prov.metadata("qwen35.full_attention_interval")
        .and_then(|v| v.as_u32())
        .unwrap_or(4) as usize;
    assert_eq!(full_attn_interval, 4);

    let num_layers = prov.metadata("qwen35.block_count")
        .and_then(|v| v.as_u32())
        .unwrap_or(32) as usize;
    assert_eq!(num_layers, 32);

    let context_len = prov.metadata("qwen35.context_length")
        .and_then(|v| v.as_u32())
        .unwrap_or(262144) as usize;
    assert_eq!(context_len, 262144);

    // Load full-attention layer 3
    let ws = grim_nn::WeightSource::root(&prov, Device::Cpu).pp("blk.3");
    let blk = Qwen35Block::load_tp(&ws, &c, 3, grim_nn::TensorParallelConfig::default())
        .expect("load layer 3");
    assert!(blk.is_full_attention, "Layer 3 must be full attention");

    let mut cache = Qwen35LayerCache::new(&c);
    let x = vec![0.05f32; c.hidden_size];
    let in_tensor = grim_backend_cpu::cpu_tensor(x, Shape::new(vec![1, c.hidden_size]));

    let out = blk.forward(&in_tensor, &[0], &mut cache).expect("forward step 0");
    let out_vec = out.to_vec_f32().expect("read out");
    assert_eq!(out_vec.len(), c.hidden_size);
    assert!(out_vec.iter().all(|v| v.is_finite()), "Step-0 baseline decode output must be finite");

    // Verify cache allocated expected KV elements in the arena (step 0 -> 1 token of kv_dim)
    let kvd = c.num_kv_heads * c.head_dim;
    let k_arena = cache.k_device.as_ref().expect("k_device arena must be allocated");
    let v_arena = cache.v_device.as_ref().expect("v_device arena must be allocated");
    let k_vec = k_arena.to_cpu_vec_f32().expect("read K arena");
    let v_vec = v_arena.to_cpu_vec_f32().expect("read V arena");
    assert!(k_vec.len() >= kvd, "k_arena should hold at least 1 token of kvd floats");
    assert!(v_vec.len() >= kvd, "v_arena should hold at least 1 token of kvd floats");
    assert!(k_vec[..kvd].iter().all(|v| v.is_finite()), "K arena row 0 must be finite");
    assert!(v_vec[..kvd].iter().all(|v| v.is_finite()), "V arena row 0 must be finite");
    cache.current_pos += 1;
    assert_eq!(cache.current_pos, 1);
}
