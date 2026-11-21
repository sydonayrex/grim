//! Baseline sync and stage-timing probe for `Qwen38FlashNext` (Q0).
//!
//! When `GRIM_FLASH_PROFILE` is unset (or not `"1"`), [`span`], [`record_d2h`],
//! and [`record_h2d`] are zero-cost pass-throughs that do not touch atomics or
//! clocks. When `GRIM_FLASH_PROFILE=1`, they accumulate per-stage wall time and
//! D2H/H2D transfer counts and byte totals across [`ProbeStage`].

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Flash-Next forward sub-stages tracked by [`span`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum ProbeStage {
    Router = 0,
    ExpertStage = 1,
    Gdn = 2,
    Qsa = 3,
    LmHead = 4,
}

impl ProbeStage {
    pub const ALL: [ProbeStage; 5] = [
        ProbeStage::Router,
        ProbeStage::ExpertStage,
        ProbeStage::Gdn,
        ProbeStage::Qsa,
        ProbeStage::LmHead,
    ];

    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            ProbeStage::Router => "Router",
            ProbeStage::ExpertStage => "ExpertStage",
            ProbeStage::Gdn => "Gdn",
            ProbeStage::Qsa => "Qsa",
            ProbeStage::LmHead => "LmHead",
        }
    }
}

/// Snapshot of accumulated counters for a single [`ProbeStage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageStats {
    pub stage: ProbeStage,
    pub calls: u64,
    pub total_ns: u64,
    pub d2h_count: u64,
    pub d2h_bytes: u64,
    pub h2d_count: u64,
    pub h2d_bytes: u64,
}

struct AtomicStageCounters {
    calls: AtomicU64,
    total_ns: AtomicU64,
    d2h_count: AtomicU64,
    d2h_bytes: AtomicU64,
    h2d_count: AtomicU64,
    h2d_bytes: AtomicU64,
}

impl AtomicStageCounters {
    const fn new() -> Self {
        Self {
            calls: AtomicU64::new(0),
            total_ns: AtomicU64::new(0),
            d2h_count: AtomicU64::new(0),
            d2h_bytes: AtomicU64::new(0),
            h2d_count: AtomicU64::new(0),
            h2d_bytes: AtomicU64::new(0),
        }
    }
}

static STAGE_COUNTERS: [AtomicStageCounters; 5] = [
    AtomicStageCounters::new(),
    AtomicStageCounters::new(),
    AtomicStageCounters::new(),
    AtomicStageCounters::new(),
    AtomicStageCounters::new(),
];

/// Returns `true` when `GRIM_FLASH_PROFILE=1` is active.
#[inline]
pub fn flash_profile_enabled() -> bool {
    std::env::var("GRIM_FLASH_PROFILE").ok().as_deref() == Some("1")
}

/// Execute `f`, recording wall time into `stage` iff `GRIM_FLASH_PROFILE=1`.
/// When unset, this is an inline pass-through returning `f()` unchanged.
#[inline]
pub fn span<T, F: FnOnce() -> T>(stage: ProbeStage, f: F) -> T {
    if !flash_profile_enabled() {
        return f();
    }
    let t0 = Instant::now();
    let out = f();
    let ns = t0.elapsed().as_nanos().min(u64::MAX as u128) as u64;
    let c = &STAGE_COUNTERS[stage as usize];
    c.calls.fetch_add(1, Ordering::Relaxed);
    c.total_ns.fetch_add(ns, Ordering::Relaxed);
    out
}

/// Record one device-to-host transfer in `stage` when profiling is enabled.
#[inline]
pub fn record_d2h(stage: ProbeStage, bytes: usize) {
    if !flash_profile_enabled() {
        return;
    }
    let c = &STAGE_COUNTERS[stage as usize];
    c.d2h_count.fetch_add(1, Ordering::Relaxed);
    c.d2h_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
}

/// Record one host-to-device transfer in `stage` when profiling is enabled.
#[inline]
pub fn record_h2d(stage: ProbeStage, bytes: usize) {
    if !flash_profile_enabled() {
        return;
    }
    let c = &STAGE_COUNTERS[stage as usize];
    c.h2d_count.fetch_add(1, Ordering::Relaxed);
    c.h2d_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
}

/// Reset all accumulated stage counters to zero.
pub fn reset_profile() {
    for c in &STAGE_COUNTERS {
        c.calls.store(0, Ordering::Relaxed);
        c.total_ns.store(0, Ordering::Relaxed);
        c.d2h_count.store(0, Ordering::Relaxed);
        c.d2h_bytes.store(0, Ordering::Relaxed);
        c.h2d_count.store(0, Ordering::Relaxed);
        c.h2d_bytes.store(0, Ordering::Relaxed);
    }
}

/// Capture a snapshot of all five [`ProbeStage`] counters.
pub fn snapshot_profile() -> [StageStats; 5] {
    ProbeStage::ALL.map(|stage| {
        let c = &STAGE_COUNTERS[stage as usize];
        StageStats {
            stage,
            calls: c.calls.load(Ordering::Relaxed),
            total_ns: c.total_ns.load(Ordering::Relaxed),
            d2h_count: c.d2h_count.load(Ordering::Relaxed),
            d2h_bytes: c.d2h_bytes.load(Ordering::Relaxed),
            h2d_count: c.h2d_count.load(Ordering::Relaxed),
            h2d_bytes: c.h2d_bytes.load(Ordering::Relaxed),
        }
    })
}

/// Render a human-readable table of the current profile snapshot.
pub fn format_profile_report() -> String {
    let snap = snapshot_profile();
    let mut out = String::from(
        "stage       | calls  | total_ms   | d2h_cnt | d2h_bytes   | h2d_cnt | h2d_bytes\n",
    );
    for s in snap {
        let ms = s.total_ns as f64 / 1e6;
        out.push_str(&format!(
            "{:<11} | {:>6} | {:>10.3} | {:>7} | {:>11} | {:>7} | {:>11}\n",
            s.stage.name(),
            s.calls,
            ms,
            s.d2h_count,
            s.d2h_bytes,
            s.h2d_count,
            s.h2d_bytes
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn unset_profile_is_passthrough_returning_value_unchanged() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("GRIM_FLASH_PROFILE").ok();
        unsafe {
            std::env::remove_var("GRIM_FLASH_PROFILE");
        }
        reset_profile();

        let val = span(ProbeStage::Router, || 42usize + 7);
        assert_eq!(val, 49);
        record_d2h(ProbeStage::Gdn, 1024);
        record_h2d(ProbeStage::Qsa, 2048);

        let snap = snapshot_profile();
        for s in snap {
            assert_eq!(s.calls, 0, "unset path must not record calls");
            assert_eq!(s.d2h_count, 0, "unset path must not record d2h");
            assert_eq!(s.h2d_count, 0, "unset path must not record h2d");
        }

        if let Some(v) = prev {
            unsafe {
                std::env::set_var("GRIM_FLASH_PROFILE", v);
            }
        }
    }

    #[test]
    fn enabled_profile_accumulates_time_and_transfers() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("GRIM_FLASH_PROFILE").ok();
        unsafe {
            std::env::set_var("GRIM_FLASH_PROFILE", "1");
        }
        reset_profile();

        let out = span(ProbeStage::Gdn, || 123u32);
        assert_eq!(out, 123);
        record_d2h(ProbeStage::Gdn, 4096);
        record_h2d(ProbeStage::Gdn, 2048);
        record_d2h(ProbeStage::Router, 512);

        let snap = snapshot_profile();
        let gdn = snap[ProbeStage::Gdn as usize];
        assert_eq!(gdn.calls, 1);
        assert_eq!(gdn.d2h_count, 1);
        assert_eq!(gdn.d2h_bytes, 4096);
        assert_eq!(gdn.h2d_count, 1);
        assert_eq!(gdn.h2d_bytes, 2048);

        let router = snap[ProbeStage::Router as usize];
        assert_eq!(router.d2h_count, 1);
        assert_eq!(router.d2h_bytes, 512);

        reset_profile();
        let cleared = snapshot_profile();
        assert_eq!(cleared[ProbeStage::Gdn as usize].calls, 0);

        unsafe {
            std::env::remove_var("GRIM_FLASH_PROFILE");
            if let Some(v) = prev {
                std::env::set_var("GRIM_FLASH_PROFILE", v);
            }
        }
    }

    /// Q0.3 GPU baseline measurement on Card 1 (`HIP_VISIBLE_DEVICES=1`) using exact Flash-Next
    /// dimensions (2 layers: 1 GDN + 1 QSA, 512 Q8_0 experts, top_k=10, hidden=2560,
    /// GDN 48v/16k/128d/conv=4, QSA 24h/2kv/256d/idx 4x128/top_k=2048/r=4, lm_head vocab=248320).
    #[test]
    #[ignore = "Q0.3 GPU baseline benchmark; run with HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 GRIM_REQUIRE_GPU=1 GRIM_FLASH_PROFILE=1 GRIM_KERNEL_TIMES=1"]
    fn bench_flash_next_baseline_gpu() {
        if !grim_backend_rocm::gpu_test_enabled() {
            return;
        }
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            std::env::set_var("GRIM_FLASH_PROFILE", "1");
        }

        use crate::qwen4exp_flash_next::{
            Qwen38FlashNext, Qwen38FlashNextConfig, Qwen38GdnSession,
        };
        use grim_core::model::CausalLm;
        use grim_nn::WeightSource;
        use grim_tensor::dtype::{DType, KQuantScheme, QuantProvenance, Storage};
        use grim_tensor::provider::{RawTensor, TensorMeta, TensorProvider};
        use grim_tensor::{Device, Shape};
        use std::collections::HashMap;
        use std::sync::Arc;

        fn q80_dtype() -> DType {
            DType {
                arith: grim_tensor::ArithType::F32,
                storage: Storage::KQuant(KQuantScheme::Q80),
            }
        }

        fn make_q80_bytes(rows: usize, cols: usize, seed: u32) -> Vec<u8> {
            assert_eq!(cols % 32, 0);
            let blocks_per_row = cols / 32;
            let row_bytes = blocks_per_row * 34;
            let mut out = vec![0u8; rows * row_bytes];
            // f16 scale = 0.015625 (2^-6 -> 0x2400)
            let d_le = 0x2400u16.to_le_bytes();
            let mut s = seed.wrapping_mul(747796405).wrapping_add(2891336453);
            for r in 0..rows {
                for b in 0..blocks_per_row {
                    let off = (r * blocks_per_row + b) * 34;
                    out[off] = d_le[0];
                    out[off + 1] = d_le[1];
                    for q in 0..32 {
                        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                        out[off + 2 + q] = (((s >> 24) as i8) / 4) as u8;
                    }
                }
            }
            out
        }

        fn make_f32_bytes(n: usize, val: f32) -> Vec<u8> {
            let mut out = Vec::with_capacity(n * 4);
            let b = val.to_le_bytes();
            for _ in 0..n {
                out.extend_from_slice(&b);
            }
            out
        }

        struct ExactBenchProvider {
            fixed: HashMap<String, (Vec<usize>, DType, Arc<Vec<u8>>)>,
            expert_gate_up: Arc<Vec<u8>>,
            expert_down: Arc<Vec<u8>>,
            hidden: usize,
            inter: usize,
            num_experts: usize,
        }

        impl ExactBenchProvider {
            fn resolve(&self, name: &str) -> Option<(Vec<usize>, DType, Arc<Vec<u8>>)> {
                if let Some(v) = self.fixed.get(name) {
                    return Some(v.clone());
                }
                if name.contains(".mlp.experts.") {
                    if name.ends_with(".gate_proj.weight") || name.ends_with(".up_proj.weight") {
                        return Some((
                            vec![self.inter, self.hidden],
                            q80_dtype(),
                            self.expert_gate_up.clone(),
                        ));
                    }
                    if name.ends_with(".down_proj.weight") {
                        return Some((
                            vec![self.hidden, self.inter],
                            q80_dtype(),
                            self.expert_down.clone(),
                        ));
                    }
                }
                let _ = self.num_experts;
                None
            }
        }

        impl TensorProvider for ExactBenchProvider {
            fn get(&self, name: &str) -> grim_tensor::error::Result<RawTensor> {
                let (shape, dtype, bytes) = self.resolve(name).ok_or_else(|| {
                    grim_tensor::error::Error::Backend(format!("missing {name}"))
                })?;
                Ok(RawTensor {
                    bytes: (*bytes).clone(),
                    shape,
                    dtype,
                    provenance: QuantProvenance::GrimNative,
                })
            }
            fn meta(&self, name: &str) -> grim_tensor::error::Result<TensorMeta> {
                let (shape, dtype, _) = self.resolve(name).ok_or_else(|| {
                    grim_tensor::error::Error::Backend(format!("missing meta {name}"))
                })?;
                Ok(TensorMeta {
                    dtype,
                    provenance: QuantProvenance::GrimNative,
                    shape,
                    fusion_mask: 0,
                })
            }
        }

        let mut cfg = Qwen38FlashNextConfig::default();
        cfg.num_layers = 2;
        cfg.layer_types = vec!["linear_attention".into(), "full_attention".into()];
        cfg.attention_compress_ratios = vec![0, 4];
        cfg.ngram_vocab_size = None;
        cfg.ngram_dim = None;
        cfg.vocab_size = 64; // keep synthetic tok_embeddings tiny; lm_head is swapped to real 248320 below

        let h = cfg.hidden_size; // 2560
        let inter = cfg.intermediate_size; // 640
        let q_dim = cfg.num_heads * cfg.head_dim; // 24*256 = 6144
        let kv_dim = cfg.num_kv_heads * cfg.head_dim; // 2*256 = 512
        let key_dim = cfg.ssm_n_group * cfg.ssm_d_state; // 16*128 = 2048
        let val_dim = cfg.ssm_dt_rank * cfg.ssm_d_state; // 48*128 = 6144
        let conv_dim = 2 * key_dim + val_dim; // 10240

        let mut fixed: HashMap<String, (Vec<usize>, DType, Arc<Vec<u8>>)> = HashMap::new();
        fn put_q80(
            fixed: &mut HashMap<String, (Vec<usize>, DType, Arc<Vec<u8>>)>,
            name: String,
            rows: usize,
            cols: usize,
            bytes: Vec<u8>,
        ) {
            fixed.insert(name, (vec![rows, cols], q80_dtype(), Arc::new(bytes)));
        }
        fn put_f32(
            fixed: &mut HashMap<String, (Vec<usize>, DType, Arc<Vec<u8>>)>,
            name: String,
            shape: Vec<usize>,
            bytes: Vec<u8>,
        ) {
            fixed.insert(name, (shape, DType::F32, Arc::new(bytes)));
        }

        put_f32(
            &mut fixed,
            "model.embed_tokens.weight".into(),
            vec![h, cfg.vocab_size],
            make_f32_bytes(h * cfg.vocab_size, 0.01),
        );
        put_f32(
            &mut fixed,
            "model.norm.weight".into(),
            vec![h],
            make_f32_bytes(h, 1.0),
        );
        put_q80(
            &mut fixed,
            "lm_head.weight".into(),
            248320,
            h,
            make_q80_bytes(248320, h, 99),
        );

        for i in 0..2 {
            let p = format!("model.layers.{i}");
            let s = (i as u32 + 1) * 100;
            put_f32(&mut fixed, format!("{p}.input_layernorm.weight"), vec![h], make_f32_bytes(h, 1.0));
            put_f32(&mut fixed, format!("{p}.post_attention_layernorm.weight"), vec![h], make_f32_bytes(h, 1.0));
            put_q80(&mut fixed, format!("{p}.mlp.gate.weight"), cfg.num_experts, h, make_q80_bytes(cfg.num_experts, h, s + 1));
            put_f32(&mut fixed, format!("{p}.ffn_gate_inp_shexp.weight"), vec![1, h], make_f32_bytes(h, 0.01));
            put_q80(&mut fixed, format!("{p}.ffn_gate_shexp.weight"), inter, h, make_q80_bytes(inter, h, s + 2));
            put_q80(&mut fixed, format!("{p}.ffn_up_shexp.weight"), inter, h, make_q80_bytes(inter, h, s + 3));
            put_q80(&mut fixed, format!("{p}.ffn_down_shexp.weight"), h, inter, make_q80_bytes(h, inter, s + 4));

            if i == 0 {
                put_q80(&mut fixed, format!("{p}.attn_qkv.weight"), conv_dim, h, make_q80_bytes(conv_dim, h, s + 10));
                put_q80(&mut fixed, format!("{p}.attn_gate.weight"), val_dim, h, make_q80_bytes(val_dim, h, s + 11));
                put_f32(
                    &mut fixed,
                    format!("{p}.ssm_conv1d.weight"),
                    vec![conv_dim, cfg.linear_conv_kernel_dim],
                    make_f32_bytes(conv_dim * cfg.linear_conv_kernel_dim, 0.05),
                );
                put_f32(&mut fixed, format!("{p}.ssm_dt.bias"), vec![cfg.ssm_dt_rank], make_f32_bytes(cfg.ssm_dt_rank, 0.0));
                put_f32(&mut fixed, format!("{p}.ssm_a"), vec![cfg.ssm_dt_rank], make_f32_bytes(cfg.ssm_dt_rank, -0.5));
                put_q80(&mut fixed, format!("{p}.ssm_beta.weight"), cfg.ssm_dt_rank, h, make_q80_bytes(cfg.ssm_dt_rank, h, s + 12));
                put_q80(&mut fixed, format!("{p}.ssm_alpha.weight"), cfg.ssm_dt_rank, h, make_q80_bytes(cfg.ssm_dt_rank, h, s + 13));
                put_f32(&mut fixed, format!("{p}.ssm_norm.weight"), vec![cfg.ssm_d_state], make_f32_bytes(cfg.ssm_d_state, 1.0));
                put_q80(&mut fixed, format!("{p}.ssm_out.weight"), h, val_dim, make_q80_bytes(h, val_dim, s + 14));
            } else {
                put_q80(&mut fixed, format!("{p}.self_attn.q_proj.weight"), q_dim, h, make_q80_bytes(q_dim, h, s + 20));
                put_q80(&mut fixed, format!("{p}.self_attn.k_proj.weight"), kv_dim, h, make_q80_bytes(kv_dim, h, s + 21));
                put_q80(&mut fixed, format!("{p}.self_attn.v_proj.weight"), kv_dim, h, make_q80_bytes(kv_dim, h, s + 22));
                put_q80(&mut fixed, format!("{p}.self_attn.o_proj.weight"), h, q_dim, make_q80_bytes(h, q_dim, s + 23));
                put_f32(&mut fixed, format!("{p}.attn_q_norm.weight"), vec![cfg.head_dim], make_f32_bytes(cfg.head_dim, 1.0));
                put_f32(&mut fixed, format!("{p}.attn_k_norm.weight"), vec![cfg.head_dim], make_f32_bytes(cfg.head_dim, 1.0));
                put_q80(
                    &mut fixed,
                    format!("{p}.indexer.q_proj.weight"),
                    cfg.indexer_n_heads * cfg.indexer_key_length,
                    h,
                    make_q80_bytes(cfg.indexer_n_heads * cfg.indexer_key_length, h, s + 24),
                );
                put_q80(
                    &mut fixed,
                    format!("{p}.indexer.k_proj.weight"),
                    cfg.indexer_key_length,
                    h,
                    make_q80_bytes(cfg.indexer_key_length, h, s + 25),
                );
                put_f32(
                    &mut fixed,
                    format!("{p}.indexer.q_norm.weight"),
                    vec![cfg.indexer_key_length],
                    make_f32_bytes(cfg.indexer_key_length, 1.0),
                );
                put_f32(
                    &mut fixed,
                    format!("{p}.indexer.k_norm.weight"),
                    vec![cfg.indexer_key_length],
                    make_f32_bytes(cfg.indexer_key_length, 1.0),
                );
            }
        }

        let provider = ExactBenchProvider {
            fixed,
            expert_gate_up: Arc::new(make_q80_bytes(inter, h, 777)),
            expert_down: Arc::new(make_q80_bytes(h, inter, 888)),
            hidden: h,
            inter,
            num_experts: cfg.num_experts,
        };

        let dev = Device::Rocm(0);
        let ws = WeightSource::root(&provider, dev.clone());
        let mut model = Qwen38FlashNext::load(dev.clone(), &ws, cfg.clone())
            .expect("exact Flash-Next 2-layer Q8_0 fixture must load on ROCm");
        model.output = grim_nn::Linear::load_shape(&ws.scoped("lm_head"), [h, 248320])
            .expect("real 248320 lm_head");

        // Warmup 1 step (compiles hipRTC kernels before timing).
        {
            let mut sess = model.new_session();
            let ids = grim_backend_cpu::cpu_tensor(vec![1.0], Shape::new(vec![1]));
            let pos = grim_backend_cpu::cpu_tensor(vec![0.0], Shape::new(vec![1]));
            let _ = model.forward(sess.as_mut(), &ids, &pos, &[]).expect("warmup");
        }

        let median3 = |mut v: Vec<f64>| -> f64 {
            v.sort_by(|a, b| a.total_cmp(b));
            v[v.len() / 2]
        };

        for (label, ctx_tokens) in [("decode_1k", 1024usize), ("decode_19k", 19456usize)] {
            let mut totals = Vec::new();
            let mut last_report = String::new();
            for run in 0..3 {
                let mut sess = model.new_session();
                let mut gs = Qwen38GdnSession::default();
                gs.ensure(&model.layers, &model.cfg);
                gs.qsa_keys = vec![vec![], vec![0.01f32; ctx_tokens * cfg.indexer_key_length]];
                sess.set_model_state(Box::new(gs));

                let ids = grim_backend_cpu::cpu_tensor(vec![5.0], Shape::new(vec![1]));
                let pos =
                    grim_backend_cpu::cpu_tensor(vec![ctx_tokens as f32], Shape::new(vec![1]));
                reset_profile();
                let t0 = Instant::now();
                let out = model
                    .forward(sess.as_mut(), &ids, &pos, &[])
                    .expect("decode forward");
                let _ = out.shape();
                grim_backend_rocm::RocmDevice::shared(0).synchronize();
                let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
                totals.push(elapsed_ms);
                if run == 2 {
                    last_report = format_profile_report();
                }
            }
            eprintln!(
                "[Q0-BASELINE] {label}: 3-run={totals:?} ms, median={:.3} ms\n{last_report}",
                median3(totals.clone())
            );
        }

        // Prefill measurement: 64 tokens full 2-layer forward (3 runs) + 2048-row lm_head microbench (3 runs)
        {
            let seq_len = 64usize;
            let mut totals = Vec::new();
            let mut last_report = String::new();
            let ids_v: Vec<f32> = (0..seq_len).map(|i| (i % 60) as f32).collect();
            let pos_v: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
            for run in 0..3 {
                let mut sess = model.new_session();
                let ids = grim_backend_cpu::cpu_tensor(ids_v.clone(), Shape::new(vec![seq_len]));
                let pos = grim_backend_cpu::cpu_tensor(pos_v.clone(), Shape::new(vec![seq_len]));
                reset_profile();
                let t0 = Instant::now();
                let _ = model
                    .forward(sess.as_mut(), &ids, &pos, &[])
                    .expect("prefill forward");
                grim_backend_rocm::RocmDevice::shared(0).synchronize();
                let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
                totals.push(elapsed_ms);
                if run == 2 {
                    last_report = format_profile_report();
                }
            }
            eprintln!(
                "[Q0-BASELINE] prefill_{seq_len}: 3-run={totals:?} ms, median={:.3} ms\n{last_report}",
                median3(totals.clone())
            );
        }

        {
            let seq_len = 2048usize;
            let normed_cpu =
                grim_backend_cpu::cpu_tensor(vec![0.05f32; seq_len * h], Shape::new(vec![seq_len, h]));
            let normed = grim_nn::modules::move_to_device(&normed_cpu, &dev)
                .expect("move normed_2048 to device");
            let mut totals = Vec::new();
            let mut last_report = String::new();
            for run in 0..3 {
                reset_profile();
                let t0 = Instant::now();
                let _logits = span(ProbeStage::LmHead, || model.output.forward(&normed))
                    .expect("lm_head_2048");
                grim_backend_rocm::RocmDevice::shared(0).synchronize();
                let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
                totals.push(elapsed_ms);
                if run == 2 {
                    last_report = format_profile_report();
                }
            }
            eprintln!(
                "[Q0-BASELINE] lm_head_{seq_len}: 3-run={totals:?} ms, median={:.3} ms\n{last_report}",
                median3(totals.clone())
            );
        }

        let ktimes = grim_backend_rocm::kernel_times_snapshot();
        let mut sorted_k = ktimes;
        sorted_k.sort_by(|a, b| b.2.cmp(&a.2));
        eprintln!("[Q0-KERNEL-TIMES top 15]:");
        for (name, count, ns) in sorted_k.into_iter().take(15) {
            eprintln!("  {name:<40} calls={count:<6} total_ms={:.3}", ns as f64 / 1e6);
        }

        unsafe {
            std::env::remove_var("GRIM_FLASH_PROFILE");
        }
    }
}
