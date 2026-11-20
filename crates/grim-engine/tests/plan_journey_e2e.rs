//! End-to-end integration and user-journey tests validating production entry points,
//! backend dispatch arms, container/quantization pipelines, and runtime state machines.

use grim_backend_cpu::cpu_tensor;
use grim_backend_rocm::decode_graph_buffers::DecodeGraphBuffers;
use grim_backend_rocm::device::moe_hybrid_exec::{
    PcieLatencyModel, compute_pcie_share, select_top5_of_6_faster_experts,
};
use grim_backend_rocm::device::roc_device::RocmDevice;
use grim_backend_rocm::kernels::comm_fuse::{COMM_FUSE_KERNEL_SOURCE, WTAB_PATCH_CAP, WtabPatch};
use grim_backend_rocm::kernels::compute_kernels::OTHER_KERNEL_SOURCE;
use grim_backend_rocm::kernels::dot_gemv::KERNEL_SOURCE as DOT_GEMV_SOURCE;
use grim_backend_rocm::kernels::fwht::KERNEL_SOURCE as FWHT_SOURCE;
use grim_backend_rocm::kernels::gla_kernels::GLA_KERNEL_SOURCE;
use grim_backend_rocm::kernels::qkv_attention::{KvArenaFormat, parse_kv_arena_env};
use grim_backend_rocm::kernels::shared_device_fns::KERNEL_SOURCE as SHARED_DEV_SOURCE;
use grim_backend_rocm::kernels::wmma_i8_gemm::WMMA_I8_KERNEL_SOURCE;
use grim_backend_rocm::kernels::wmma_iq_gemm::KERNEL_SOURCE as WMMA_IQ_SOURCE;
use grim_backend_rocm::memory::fault::{self, HsaMemoryAccessFault};
use grim_backend_rocm::memory::hugepage::{
    compute_cgroup_v2_reclaimable_bytes, compute_host_pin_budget,
    should_madvise_hugepage_from_proc, should_skip_uma_host_pin,
};
use grim_backend_rocm::memory::ledger::{self, UnownedDiagnosis};
use grim_backend_rocm::memory::pool::{VMM_GRANULARITY_FLOOR_BYTES, align_vmm_elastic_chunk};
use grim_backend_rocm::speculative::{
    TokenAcceptor, coupled_gumbel_max_draw, verify_gated_speculative_step,
};
use grim_core::sampler::{SamplingParams, load_gen_sampling_overlay};
use grim_engine::kva_prefill::{KvaSplitPlan, LayerPrefillMode};
use grim_format::format::{
    ExpertPriorEntry, GRIM_ALIGN_DIRECTIO_UNIT, GRIM_ALIGN_THP_POOL, GrimFile, GrimHeader,
    GrimTensorEntry, WaveSize, encode_expert_priors, read_expert_priors_from_file,
    write_attachments,
};
use grim_kvquant::{
    dequantize_q4_0_deterministic, quantize_k8v4_head, quantize_q4_0_deterministic_tiebreak,
};
use grim_memory::moe_budget::LruResidencyTracker;
use grim_memory::{BLOCK_SIZE, KvBlockPool, RecurrentCheckpointPool, SeqWindow};
use grim_models_transformer::Gemma4ProportionalRope;
use grim_models_transformer::qwen4exp_gdn::{
    GdnParams, KdaHeadPairing, Qwen38GdnCache, gated_delta_net_forward,
};
use grim_nn::Rope;
use grim_quant::tqk_trellis::{
    dequant_tqk_block32, encode_tqk_tail_biting_viterbi, tq_computed_pair_bits, tq_computed_quad,
};
use grim_tensor::Shape;
use std::collections::HashMap;
use std::sync::Mutex;

static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Journey 1: `.grim` container serialization with 2 MiB THP pool + 4 KiB O_DIRECT alignment
/// and embedded expert priors -> `LruResidencyTracker` per-layer quota -> PCIe/CPU hybrid split.
#[test]
fn journey_1_grim_container_directio_thp_priors_to_moe_quota_and_pcie_split() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let dir = tempfile::tempdir().expect("tempdir");
    let grim_path = dir.path().join("test_model.grim");

    unsafe { std::env::set_var("GRIM_DIRECTIO_ALIGN", "1") };
    let e0 = GrimTensorEntry {
        name: "blk.0.ffn_gate_exps.weight".to_string(),
        shape: vec![64, 64],
        base_bitwidth: 4,
        payload_size: 2048,
        ..Default::default()
    };
    let e1 = GrimTensorEntry {
        name: "blk.1.ffn_gate_exps.weight".to_string(),
        shape: vec![64, 64],
        base_bitwidth: 4,
        payload_size: 2048,
        ..Default::default()
    };
    let grim_file = GrimFile {
        header: GrimHeader::new(2, 0),
        metadata: grim_format::gguf::GrimMetadata::default(),
        tensors: vec![e0, e1],
        wave: WaveSize::W32,
        tensors_by_name: HashMap::new(),
        kv_blobs: HashMap::new(),
    };

    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .read(true)
        .write(true)
        .open(&grim_path)
        .expect("open grim file");
    let entries = grim_file.write(&mut file).expect("GrimFile::write");
    unsafe { std::env::remove_var("GRIM_DIRECTIO_ALIGN") };

    assert_eq!(entries[0].payload_offset % GRIM_ALIGN_THP_POOL, 0);
    assert_eq!(entries[1].payload_offset % GRIM_ALIGN_DIRECTIO_UNIT, 0);

    // Append expert priors trailer and read back through file API
    let priors = vec![
        ExpertPriorEntry {
            layer: 0,
            expert: 2,
            share: 0.45,
        },
        ExpertPriorEntry {
            layer: 1,
            expert: 5,
            share: 0.35,
        },
    ];
    write_attachments(
        &mut file,
        &[("expert_priors".to_string(), encode_expert_priors(&priors))],
    )
    .expect("write_attachments");
    drop(file);

    let loaded_priors = read_expert_priors_from_file(&grim_path)
        .expect("read_expert_priors_from_file")
        .expect("some priors");
    assert_eq!(loaded_priors, priors);

    // Feed 48-layer x 2-chunk prefill routing trace through production LruResidencyTracker::new
    unsafe { std::env::remove_var("GRIM_MOE_LAYER_QUOTA") };
    let mut tracker_off = LruResidencyTracker::new(192);
    unsafe { std::env::set_var("GRIM_MOE_LAYER_QUOTA", "1") };
    let mut tracker_on = LruResidencyTracker::new(192);
    unsafe { std::env::remove_var("GRIM_MOE_LAYER_QUOTA") };

    for l in 0..48 {
        for e in 0..8 {
            tracker_off.admit(l, e);
            tracker_on.admit(l, e);
        }
    }
    let mut hits_off = 0usize;
    let mut hits_on = 0usize;
    for l in 0..48 {
        for e in (0..8).rev() {
            if tracker_off.lookup(l, e).is_some() {
                hits_off += 1;
            }
            tracker_off.admit(l, e);
            if tracker_on.lookup(l, e).is_some() {
                hits_on += 1;
            }
            tracker_on.admit(l, e);
        }
    }
    assert_eq!(hits_off, 0);
    assert_eq!(hits_on, 48 * 4);

    // Verify PCIe/CPU hybrid split and top-5-of-6 rule
    let share = compute_pcie_share(28.0, 25.0);
    assert!(share > 0.40 && share < 0.60);
    let model = PcieLatencyModel {
        pcie_setup_us: 8.0,
        pcie_bw_bytes_per_us: 28_000.0,
        cpu_setup_us: 5.0,
        cpu_bw_bytes_per_us: 25_000.0,
    };
    assert_eq!(model.optimal_split_mask(&vec![2 * 1024 * 1024; 4]).count_ones(), 2);
    let routed = vec![
        (0, 0.30),
        (1, 0.25),
        (2, 0.18),
        (3, 0.12),
        (4, 0.10),
        (5, 0.05),
    ];
    assert_eq!(select_top5_of_6_faster_experts(&routed, |_| false).len(), 5);
}

/// Journey 2: Hybrid Gated DeltaNet prefill through production `gated_delta_net_forward`
/// comparing sequential execution against 16-token WY-chunked forward substitution, plus KVA split.
#[test]
fn journey_2_gdn_wy_chunk16_through_production_forward_and_kva_split() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let kva = KvaSplitPlan::from_env_str(Some("20/8"), 36).expect("kva plan");
    assert_eq!(kva.prefix_modes[19], LayerPrefillMode::Full);
    assert_eq!(kva.prefix_modes[20], LayerPrefillMode::StateOnly);
    assert_eq!(kva.tail_modes[20], LayerPrefillMode::Full);
    let prompt: Vec<u32> = (0..48).collect();
    let (prefix, tail) = kva.split_prompt(&prompt);
    assert_eq!(prefix.len(), 40);
    assert_eq!(tail.len(), 8);

    let n_k_heads = 2;
    let n_v_heads = 4;
    let head_dim = 8;
    let seq_len = 32;
    let conv_dim = 2 * n_k_heads * head_dim + n_v_heads * head_dim;

    let conv_mix: Vec<f32> = (0..(seq_len * conv_dim))
        .map(|i| ((i * 17 + 5) % 97) as f32 * 0.02 - 0.9)
        .collect();
    let alpha: Vec<f32> = (0..(seq_len * n_v_heads))
        .map(|i| ((i * 13 + 3) % 31) as f32 * 0.05 - 0.8)
        .collect();
    let beta: Vec<f32> = (0..(seq_len * n_v_heads))
        .map(|i| ((i * 7 + 1) % 29) as f32 * 0.08 - 1.1)
        .collect();
    let a = [-0.25f32, -0.40, -0.15, -0.30];
    let dt_bias = [0.10f32, -0.05, 0.20, 0.0];
    let norm = vec![1.0f32; head_dim];

    let params = GdnParams {
        conv_mix: &conv_mix,
        alpha: &alpha,
        beta: &beta,
        a: &a,
        dt_bias: &dt_bias,
        norm: &norm,
        eps: 1e-6,
        n_v_heads,
        n_k_heads,
        head_dim,
        conv_dim,
        seq_len,
        pairing: KdaHeadPairing::Grouped,
    };

    unsafe { std::env::remove_var("GRIM_GDN_WY_CHUNK") };
    let mut cache_seq = Qwen38GdnCache::new(n_v_heads, head_dim, head_dim, 4, conv_dim);
    let mut out_seq = vec![0.0f32; seq_len * n_v_heads * head_dim];
    gated_delta_net_forward(&params, &mut cache_seq, &mut out_seq).expect("sequential forward");

    unsafe { std::env::set_var("GRIM_GDN_WY_CHUNK", "1") };
    let mut cache_wy = Qwen38GdnCache::new(n_v_heads, head_dim, head_dim, 4, conv_dim);
    let mut out_wy = vec![0.0f32; seq_len * n_v_heads * head_dim];
    gated_delta_net_forward(&params, &mut cache_wy, &mut out_wy).expect("wy chunk16 forward");
    unsafe { std::env::remove_var("GRIM_GDN_WY_CHUNK") };

    for (i, (&s, &w)) in out_seq.iter().zip(out_wy.iter()).enumerate() {
        assert!(
            (s - w).abs() < 1e-4,
            "GDN output mismatch at {i}: seq={s} vs wy={w}"
        );
    }
    for (i, (&s, &w)) in cache_seq.ssm_state.iter().zip(cache_wy.ssm_state.iter()).enumerate() {
        assert!(
            (s - w).abs() < 1e-4,
            "GDN final state mismatch at {i}: seq={s} vs wy={w}"
        );
    }
}

/// Journey 3: Computed-codebook trellis Viterbi quantization, deterministic opposite-sign `q4_0`
/// tie-breaking, asymmetric `K8V4` head quantization, and 2 MiB VMM chunk alignment.
#[test]
fn journey_3_trellis_viterbi_and_kv_q4_deterministic_tiebreak_k8v4_vmm() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    for s in [0u16, 1, 42, 1024, 16383] {
        let quad = tq_computed_quad(s);
        for v in quad {
            assert!(v.is_finite() && v.abs() >= 0.0625 && v.abs() <= 2.0);
        }
    }
    assert_eq!(tq_computed_pair_bits(42) & 0x7000_7000, 0x3000_3000);

    let mut target = [0.0f32; 32];
    for (i, v) in target.iter_mut().enumerate() {
        *v = ((i as f32 * 0.73).sin() * 1.4) + ((i as f32 * 1.91).cos() * 0.6);
    }
    let (states, scale) = encode_tqk_tail_biting_viterbi(&target, 4);
    let recon = dequant_tqk_block32(&states, scale);
    let dot: f32 = target.iter().zip(recon.iter()).map(|(a, b)| a * b).sum();
    let na: f32 = target.iter().map(|a| a * a).sum::<f32>().sqrt();
    let nb: f32 = recon.iter().map(|b| b * b).sum::<f32>().sqrt();
    assert!(dot / (na * nb + 1e-12) > 0.78);

    // Opposite-sign tie-break permutation invariance
    let mut g1 = [0.0f32; 32];
    g1[0] = 3.5;
    g1[1] = -3.5;
    let mut g2 = [0.0f32; 32];
    g2[0] = -3.5;
    g2[1] = 3.5;
    let (q1, d1) = quantize_q4_0_deterministic_tiebreak(&g1);
    let (q2, d2) = quantize_q4_0_deterministic_tiebreak(&g2);
    assert_eq!(d1.to_bits(), d2.to_bits());
    let dec1 = dequantize_q4_0_deterministic(&q1, d1);
    let dec2 = dequantize_q4_0_deterministic(&q2, d2);
    assert_eq!(dec1[0].to_bits(), dec2[1].to_bits());

    let (k_i8, k_s, v_p, v_s) = quantize_k8v4_head(&g1, &g2);
    assert_eq!(k_i8.len(), 32);
    assert_eq!(v_p.len(), 16);
    assert!(k_s > 0.0 && v_s != 0.0);

    unsafe { std::env::remove_var("GRIM_KV_K8V4") };
    assert_eq!(parse_kv_arena_env("k8v4"), Some(KvArenaFormat::Int8));
    unsafe { std::env::set_var("GRIM_KV_K8V4", "1") };
    assert_eq!(parse_kv_arena_env("k8v4"), Some(KvArenaFormat::Int4));
    unsafe { std::env::remove_var("GRIM_KV_K8V4") };

    assert_eq!(
        align_vmm_elastic_chunk(512 * 1024, 64 * 1024),
        VMM_GRANULARITY_FLOOR_BYTES
    );
}

/// Journey 4: Coupled Philox-4x32 Gumbel-max drafting, unbiased rejection with residual recovery,
/// `TokenAcceptor::decide` production dispatch, and `DualRecurrentSlot` speculative state rollback.
#[test]
fn journey_4_coupled_philox_speculative_verification_and_dual_recurrent_rollback() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let probs = [0.10f32, 0.50, 0.25, 0.15];
    for pos in 0..64u64 {
        assert_eq!(
            coupled_gumbel_max_draw(&probs, 0xCAFE_BABE, pos),
            coupled_gumbel_max_draw(&probs, 0xCAFE_BABE, pos)
        );
    }

    let q_draft = [0.40f32, 0.15, 0.10, 0.35];
    let trials = 10_000u64;
    let mut counts = [0usize; 4];
    for step in 0..trials {
        let dt = coupled_gumbel_max_draw(&q_draft, 0x1337_C0DE, step);
        let out = verify_gated_speculative_step(&probs, &q_draft, dt, true, 0x1337_C0DE, step);
        counts[out] += 1;
    }
    for v in 0..4 {
        let emp = counts[v] as f32 / trials as f32;
        assert!((emp - probs[v]).abs() < 0.02);
    }

    unsafe { std::env::set_var("GRIM_COUPLED_SPEC", "1") };
    let acceptor = TokenAcceptor::new(0.5);
    let _ = acceptor.decide(&[(0.8, 0.8), (0.8, 0.6), (0.8, 0.1)], 3);
    unsafe { std::env::remove_var("GRIM_COUPLED_SPEC") };

    let mut ckpt_pool = RecurrentCheckpointPool::new(4);
    let dual = ckpt_pool.init_dual_slot(8, 4, 3);
    dual.seed_committed(&[1.0; 8]);
    dual.draft_state_mut()[0] = 99.0;
    dual.rollback_draft();
    assert_eq!(dual.draft_state()[0], 1.0);
    dual.draft_state_mut()[0] = 42.0;
    dual.commit_draft();
    assert_eq!(dual.committed_state()[0], 42.0);
}

/// Journey 5: `KvBlockPool` reference+ring window, copy-on-idle clean flag, multi-turn checkpoint
/// save/restore, GPU fault UAF/overrun attribution, liveness arena packing, and THP/UMA pin budget.
#[test]
fn journey_5_kv_pool_seq_window_conversation_checkpoint_fault_uaf_and_bufplan() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let win = SeqWindow::new(50, 32, BLOCK_SIZE);
    assert_eq!(win.max_blocks_needed(), 6);
    for pos in 0..500 {
        assert!(win.logical_block_for_pos(pos) < win.max_blocks_needed());
    }

    let mut pool = KvBlockPool::new(8, 2, 8);
    let blk = pool.alloc().expect("alloc");
    pool.mark_block_clean_on_host(blk);
    assert!(pool.is_block_clean_on_host(blk));
    pool.write_keys(blk, &vec![1.0; 16], 1);
    assert!(!pool.is_block_clean_on_host(blk));

    unsafe { std::env::set_var("GRIM_ASTRUM_CONVERSATION_CKPT", "1") };
    pool.save_conversation_turn_checkpoint(vec![10, 20, 30], vec![1, 2, 3, 4], Vec::new());
    let restored = pool
        .restore_conversation_turn_checkpoint(&[10, 20, 30, 40])
        .expect("prefix match");
    assert_eq!(restored.key.len(), 3);
    unsafe { std::env::remove_var("GRIM_ASTRUM_CONVERSATION_CKPT") };

    // Fault resolution + freed-ring UAF & overrun diagnosis
    ledger::reset();
    ledger::register(0x4000_0000, 4096, 0, false, "kv_cache_slab");
    ledger::deregister(0x4000_0000);
    let rep = fault::resolve(
        HsaMemoryAccessFault {
            node_id: 1,
            virtual_address: 0x4000_0100,
            failure: 0,
            flags: 0,
        },
        &|_| Some(0),
    );
    assert!(matches!(rep.attribution, ledger::Attribution::Unowned { .. }));
    assert!(matches!(
        ledger::diagnose_unowned(0x4000_0100),
        Some(UnownedDiagnosis::UseAfterFree { .. })
    ));

    unsafe { std::env::set_var("GRIM_ARENA_BUFPLAN", "1") };
    let (full_bytes, shrunk_bytes) =
        DecodeGraphBuffers::liveness_bufplan_summary(32, 4096).expect("bufplan summary");
    unsafe { std::env::remove_var("GRIM_ARENA_BUFPLAN") };
    assert!(shrunk_bytes < full_bytes);

    let buddy_ok = "Node 0, zone   Normal      1 2 3 4 5 6 7 8 9 100 50\n";
    assert!(should_madvise_hugepage_from_proc(
        "always [madvise] never",
        buddy_ok,
        32 * 1024 * 1024
    ));
    assert!(should_skip_uma_host_pin("gfx1151", true));
    assert_eq!(
        compute_cgroup_v2_reclaimable_bytes("inactive_file 1000\nfile_dirty 200\nfile_writeback 50\n"),
        750
    );
    let budget = compute_host_pin_budget(64 << 30, 48 << 30, None, false);
    assert!(budget.soft_budget_bytes < budget.hard_budget_bytes);
}

/// Journey 6: 3-tier `generation_config.json` / KV / `.generation.json` overlay and proportional RoPE.
#[test]
fn journey_6_three_tier_gen_sampling_overlay_and_proportional_rope() {
    let dir = tempfile::tempdir().expect("tempdir");
    let model_path = dir.path().join("qwen_test.grim");
    std::fs::write(
        dir.path().join("generation_config.json"),
        r#"{"temperature": 0.9, "top_k": 50, "top_p": 0.95}"#,
    )
    .expect("write generation_config.json");
    std::fs::write(
        dir.path().join("qwen_test.grim.generation.json"),
        r#"{"top_k": 20}"#,
    )
    .expect("write sidecar");

    let mut kv = HashMap::new();
    kv.insert("temperature", "0.7");
    let stacked = load_gen_sampling_overlay(
        SamplingParams::default(),
        Some(&model_path),
        Some(|k: &str| kv.get(k).copied()),
    );
    assert!((stacked.temperature - 0.7).abs() < 1e-6);
    assert_eq!(stacked.top_k, 20);
    assert!((stacked.top_p - 0.95).abs() < 1e-6);

    let prop = Gemma4ProportionalRope {
        base: 1_000_000.0,
        head_dim: 512,
        n_pairs: 64,
    };
    let x = cpu_tensor(vec![0.25f32; 2 * 512], Shape::new(vec![2, 512]));
    let ref_out = prop.apply(&x, 1, &[0, 7]).expect("apply").to_vec_f32().unwrap();
    let rope = Rope::from_config(prop.to_yarn_rope_config());
    let x_3d = cpu_tensor(vec![0.25f32; 2 * 512], Shape::new(vec![1, 2, 512]));
    let got_out = rope.forward(&x_3d, &[0, 7]).expect("forward").to_vec_f32().unwrap();
    for (a, b) in ref_out.iter().zip(got_out.iter()) {
        assert!((a - b).abs() < 1e-4);
    }
}

/// Journey 7: ROCm kernel source presence, ABI parameter lists, WtabPatch, and residual probe.
#[test]
fn journey_7_rocm_kernel_abi_and_device_probe_invariants() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    assert!(FWHT_SOURCE.contains("grim_fwht_tiled_q80_f32"));
    assert!(OTHER_KERNEL_SOURCE.contains("grim_qk_rope_wide_dev_base"));
    assert!(WMMA_I8_KERNEL_SOURCE.contains("grim_wmma_w4a8_iu8_gfx12"));
    assert!(COMM_FUSE_KERNEL_SOURCE.contains("grim_copy_link_16b"));
    assert!(COMM_FUSE_KERNEL_SOURCE.contains("grim_wtab_patch"));
    assert!(SHARED_DEV_SOURCE.contains("grim_sync_lds"));
    assert!(SHARED_DEV_SOURCE.contains("grim_tsum4_wave32"));
    assert!(WMMA_IQ_SOURCE.contains("grim_wmma_iq4nl_fast"));
    assert!(WMMA_IQ_SOURCE.contains("grim_wmma_iq4xs_fast"));
    assert!(GLA_KERNEL_SOURCE.contains("grim_gdn_state_update_d128_f32"));
    assert!(DOT_GEMV_SOURCE.contains("grim_dot4_q2_0_s2_fast_gemv"));

    let mut patch = WtabPatch::default();
    let mut table = [0u64; WTAB_PATCH_CAP];
    for i in 0..WTAB_PATCH_CAP {
        assert!(patch.push((&mut table[i] as *mut u64) as u64, (i as u64) + 100));
    }
    patch.apply_host_sim();
    assert_eq!(table[0], 100);
    assert_eq!(table[WTAB_PATCH_CAP - 1], (WTAB_PATCH_CAP as u64) + 99);

    let mut row = vec![0.20f32; 256];
    row[0] = 50.0;
    unsafe { std::env::set_var("GRIM_RESID_PROBE", "1") };
    let stat = RocmDevice::probe_resid_stat(&row, "journey7").expect("probe_resid_stat");
    unsafe { std::env::remove_var("GRIM_RESID_PROBE") };
    assert!((stat.median_abs - 0.20).abs() < 1e-5);
    assert!((stat.max_abs - 50.0).abs() < 1e-5);
}
