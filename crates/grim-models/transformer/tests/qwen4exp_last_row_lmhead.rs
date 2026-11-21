//! TDD tests for Q1a (`GRIM_FLASH_LAST_ROW_LMHEAD=1`): last-row LM head slicing on prefill.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

use grim_core::error::Result as GrimResult;
use grim_core::CausalLm;
use grim_nn::WeightSource;
use grim_tensor::dtype::{KQuantScheme, Storage};
use grim_tensor::provider::{RawTensor, TensorMeta, TensorProvider};
use grim_tensor::{ArithType, DType, Device, QuantProvenance, Shape};

use grim_models_transformer::qwen4exp_flash_next::{
    flash_last_row_lmhead_enabled, Qwen38FlashNext, Qwen38FlashNextConfig,
};

static ENV_LOCK: Mutex<()> = Mutex::new(());

struct EnvGuard {
    key: &'static str,
    prev: Option<String>,
}

impl EnvGuard {
    fn set(key: &'static str, val: &str) -> Self {
        let prev = std::env::var(key).ok();
        unsafe { std::env::set_var(key, val) };
        Self { key, prev }
    }

    fn unset(key: &'static str) -> Self {
        let prev = std::env::var(key).ok();
        unsafe { std::env::remove_var(key) };
        Self { key, prev }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        unsafe {
            match &self.prev {
                Some(v) => std::env::set_var(self.key, v),
                None => std::env::remove_var(self.key),
            }
        }
    }
}

struct MemProvider {
    tensors: HashMap<String, RawTensor>,
}

impl TensorProvider for MemProvider {
    fn get(&self, name: &str) -> grim_tensor::error::Result<RawTensor> {
        self.tensors
            .get(name)
            .cloned()
            .ok_or_else(|| grim_tensor::error::Error::Backend(format!("missing tensor {name}")))
    }

    fn meta(&self, name: &str) -> grim_tensor::error::Result<TensorMeta> {
        let t = self.get(name)?;
        Ok(TensorMeta {
            dtype: t.dtype,
            provenance: t.provenance,
            shape: t.shape,
            fusion_mask: 0,
        })
    }

    fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }
}

fn raw_f32(seed: u32, shape: Vec<usize>) -> RawTensor {
    let n: usize = shape.iter().product();
    let mut bytes = Vec::with_capacity(n * 4);
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    for _ in 0..n {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        let v = (((s >> 8) as f32 / 8_388_608.0) - 1.0) * 0.05;
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    RawTensor {
        bytes,
        shape,
        dtype: DType::F32,
        provenance: QuantProvenance::GrimNative,
    }
}

fn raw_ones(shape: Vec<usize>) -> RawTensor {
    let n: usize = shape.iter().product();
    let mut bytes = Vec::with_capacity(n * 4);
    for _ in 0..n {
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
    }
    RawTensor {
        bytes,
        shape,
        dtype: DType::F32,
        provenance: QuantProvenance::GrimNative,
    }
}

fn small_cfg() -> Qwen38FlashNextConfig {
    let mut cfg = Qwen38FlashNextConfig::default();
    cfg.vocab_size = 64;
    cfg.hidden_size = 32;
    cfg.num_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.head_dim = 8;
    cfg.value_head_dim = 8;
    cfg.num_layers = 1;
    cfg.layer_types = vec!["linear_attention".to_string()];
    cfg.attention_compress_ratios = vec![0];
    cfg.intermediate_size = 16;
    cfg.num_experts = 4;
    cfg.num_experts_per_tok = 2;
    cfg.shared_expert_intermediate_size = Some(16);
    cfg.hc_count = 2;
    cfg.hc_lowrank = 4;
    cfg.ssm_d_state = 4;
    cfg.ssm_n_group = 2;
    cfg.ssm_d_inner = 16;
    cfg.ssm_dt_rank = 4;
    cfg.linear_conv_kernel_dim = 2;
    cfg.linear_num_key_heads = 2;
    cfg.linear_key_head_dim = 4;
    cfg.linear_value_head_dim = 8;
    cfg.linear_num_value_heads = 2;
    cfg.ngram_vocab_size = None;
    cfg.ngram_dim = None;
    cfg.max_seq_len = 64;
    cfg
}

fn build_synth_tensors(cfg: &Qwen38FlashNextConfig) -> MemProvider {
    let mut t = HashMap::new();
    let h = cfg.hidden_size;
    let hc_dim = h * cfg.hc_count;
    let hc_lr = cfg.hc_lowrank;
    let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
    let value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
    let conv_dim = key_dim * 2 + value_dim;
    let mut put = |name: String, r: RawTensor| {
        t.insert(format!("model.{name}"), r);
    };
    put("embed_tokens.weight".into(), raw_f32(1, vec![h, cfg.vocab_size]));
    put("output_norm.weight".into(), raw_ones(vec![h]));
    put(
        "hyper_connection_mixer.hc_norm.weight".into(),
        raw_ones(vec![h]),
    );
    put("output.weight".into(), raw_f32(2, vec![h, cfg.vocab_size]));

    let p = "layers.0";
    for br in ["attn", "ffn"] {
        put(format!("{p}.hc_{br}_norm.weight"), raw_ones(vec![hc_dim]));
        put(
            format!("{p}.hc_{br}_down.weight"),
            raw_f32(101, vec![hc_lr, hc_dim]),
        );
        put(
            format!("{p}.hc_{br}_up.weight"),
            raw_f32(102, vec![hc_dim, hc_lr]),
        );
        put(
            format!("{p}.hc_{br}_inject.weight"),
            raw_f32(103, vec![cfg.hc_count, hc_dim]),
        );
    }
    put(format!("{p}.input_layernorm.weight"), raw_ones(vec![h]));
    put(
        format!("{p}.post_attention_layernorm.weight"),
        raw_ones(vec![h]),
    );
    put(
        format!("{p}.ffn_gate_inp.weight"),
        raw_f32(104, vec![cfg.num_experts, h]),
    );
    put(format!("{p}.ffn_gate_inp_shexp.weight"), raw_f32(105, vec![1, h]));
    put(
        format!("{p}.ffn_gate_exps.weight"),
        raw_f32(106, vec![cfg.num_experts, cfg.intermediate_size, h]),
    );
    put(
        format!("{p}.ffn_up_exps.weight"),
        raw_f32(107, vec![cfg.num_experts, cfg.intermediate_size, h]),
    );
    put(
        format!("{p}.ffn_down_exps.weight"),
        raw_f32(108, vec![cfg.num_experts, h, cfg.intermediate_size]),
    );
    let shared = cfg.shared_expert_intermediate_size.unwrap_or(cfg.intermediate_size);
    put(format!("{p}.ffn_gate_shexp.weight"), raw_f32(109, vec![shared, h]));
    put(format!("{p}.ffn_up_shexp.weight"), raw_f32(110, vec![shared, h]));
    put(format!("{p}.ffn_down_shexp.weight"), raw_f32(111, vec![h, shared]));
    put(format!("{p}.attn_qkv.weight"), raw_f32(112, vec![conv_dim, h]));
    put(format!("{p}.attn_gate.weight"), raw_f32(113, vec![value_dim, h]));
    put(
        format!("{p}.ssm_conv1d.weight"),
        raw_f32(114, vec![conv_dim, cfg.linear_conv_kernel_dim]),
    );
    put(format!("{p}.ssm_a"), raw_f32(115, vec![cfg.ssm_dt_rank]));
    put(format!("{p}.ssm_dt.bias"), raw_f32(116, vec![cfg.ssm_dt_rank]));
    put(
        format!("{p}.ssm_alpha.weight"),
        raw_f32(117, vec![cfg.ssm_dt_rank, h]),
    );
    put(
        format!("{p}.ssm_beta.weight"),
        raw_f32(118, vec![cfg.ssm_dt_rank, h]),
    );
    put(
        format!("{p}.ssm_norm.weight"),
        raw_ones(vec![cfg.ssm_d_state]),
    );
    put(format!("{p}.ssm_out.weight"), raw_f32(119, vec![h, value_dim]));

    MemProvider { tensors: t }
}

fn load_small_cpu_model() -> GrimResult<(Qwen38FlashNext, Qwen38FlashNextConfig)> {
    let cfg = small_cfg();
    let provider = build_synth_tensors(&cfg);
    let ws = WeightSource::root(&provider, Device::Cpu);
    let model = Qwen38FlashNext::load(Device::Cpu, &ws, cfg.clone())?;
    Ok((model, cfg))
}

#[test]
fn env_flag_defaults_off_and_parses_truthy() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    {
        let _e = EnvGuard::unset("GRIM_FLASH_LAST_ROW_LMHEAD");
        assert!(!flash_last_row_lmhead_enabled());
    }
    for v in ["0", "false", "FALSE", ""] {
        let _e = EnvGuard::set("GRIM_FLASH_LAST_ROW_LMHEAD", v);
        assert!(!flash_last_row_lmhead_enabled(), "value={v}");
    }
    for v in ["1", "true", "TRUE"] {
        let _e = EnvGuard::set("GRIM_FLASH_LAST_ROW_LMHEAD", v);
        assert!(flash_last_row_lmhead_enabled(), "value={v}");
    }
}

#[test]
fn last_row_lmhead_matches_full_prefill_bitwise_and_preserves_hidden() -> GrimResult<()> {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (model, cfg) = load_small_cpu_model()?;
    let seq_len = 8usize;
    let ids = grim_backend_cpu::cpu_tensor(
        (0..seq_len).map(|i| (i % cfg.vocab_size) as f32).collect(),
        Shape::new(vec![seq_len]),
    );
    let pos = grim_backend_cpu::cpu_tensor(
        (0..seq_len).map(|i| i as f32).collect(),
        Shape::new(vec![seq_len]),
    );

    let (off_logits, off_hidden) = {
        let _e = EnvGuard::set("GRIM_FLASH_LAST_ROW_LMHEAD", "0");
        let mut sess = model.new_session();
        let logits = model.forward(sess.as_mut(), &ids, &pos, &[])?;
        assert_eq!(logits.shape().dims(), &[seq_len, cfg.vocab_size]);
        let hidden = sess
            .get_last_hidden_state()
            .expect("session must record last_hidden_state");
        assert_eq!(hidden.shape().dims(), &[seq_len, cfg.hidden_size]);
        (logits.to_vec_f32()?, hidden.to_vec_f32()?)
    };

    let (on_logits, on_hidden) = {
        let _e = EnvGuard::set("GRIM_FLASH_LAST_ROW_LMHEAD", "1");
        let mut sess = model.new_session();
        let logits = model.forward(sess.as_mut(), &ids, &pos, &[])?;
        assert_eq!(logits.shape().dims(), &[1, cfg.vocab_size]);
        let hidden = sess
            .get_last_hidden_state()
            .expect("session must keep full normed hidden_state");
        assert_eq!(hidden.shape().dims(), &[seq_len, cfg.hidden_size]);
        (logits.to_vec_f32()?, hidden.to_vec_f32()?)
    };

    let off_last = &off_logits[(seq_len - 1) * cfg.vocab_size..];
    for c in 0..cfg.vocab_size {
        assert_eq!(
            on_logits[c].to_bits(),
            off_last[c].to_bits(),
            "last-row logits mismatch at vocab col {c}: on={} off={}",
            on_logits[c],
            off_last[c]
        );
    }
    assert_eq!(
        on_hidden.len(),
        off_hidden.len(),
        "full normed hidden_state len mismatch"
    );
    for (i, (&a, &b)) in on_hidden.iter().zip(off_hidden.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "hidden mismatch at {i}");
    }
    Ok(())
}

#[test]
fn full_output_logits_flag_overrides_last_row_slice() -> GrimResult<()> {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (model, cfg) = load_small_cpu_model()?;
    let seq_len = 8usize;
    let ids = grim_backend_cpu::cpu_tensor(
        (0..seq_len).map(|i| (i % cfg.vocab_size) as f32).collect(),
        Shape::new(vec![seq_len]),
    );
    let pos = grim_backend_cpu::cpu_tensor(
        (0..seq_len).map(|i| i as f32).collect(),
        Shape::new(vec![seq_len]),
    );

    let off_logits = {
        let _e = EnvGuard::set("GRIM_FLASH_LAST_ROW_LMHEAD", "0");
        let mut sess = model.new_session();
        model.forward(sess.as_mut(), &ids, &pos, &[])?.to_vec_f32()?
    };

    let full_logits = {
        let _e = EnvGuard::set("GRIM_FLASH_LAST_ROW_LMHEAD", "1");
        let mut sess = model.new_session();
        sess.set_full_output_logits(true);
        let out = model.forward(sess.as_mut(), &ids, &pos, &[])?;
        assert_eq!(out.shape().dims(), &[seq_len, cfg.vocab_size]);
        out.to_vec_f32()?
    };

    for (i, (&a, &b)) in full_logits.iter().zip(off_logits.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "full_output_logits mismatch at {i}");
    }
    Ok(())
}

#[test]
fn skip_output_projection_returns_empty_and_resets_session_flag() -> GrimResult<()> {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _e = EnvGuard::set("GRIM_FLASH_LAST_ROW_LMHEAD", "1");
    let (model, cfg) = load_small_cpu_model()?;
    let seq_len = 8usize;
    let ids = grim_backend_cpu::cpu_tensor(
        (0..seq_len).map(|i| (i % cfg.vocab_size) as f32).collect(),
        Shape::new(vec![seq_len]),
    );
    let pos = grim_backend_cpu::cpu_tensor(
        (0..seq_len).map(|i| i as f32).collect(),
        Shape::new(vec![seq_len]),
    );
    let mut sess = model.new_session();
    sess.set_skip_output_projection(true);
    let out = model.forward(sess.as_mut(), &ids, &pos, &[])?;
    assert_eq!(out.shape().elem_count(), 0);
    assert!(
        !sess.skip_output_projection(),
        "skip_output_projection must reset to false after forward"
    );
    Ok(())
}

#[ignore = "ROCm Card 1 parity & lm_head_512/2048 benchmark"]
#[test]
fn bench_last_row_lmhead_gpu_card1() -> GrimResult<()> {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut cfg = small_cfg();
    cfg.hidden_size = 2560;
    cfg.vocab_size = 64; // tiny embed_tokens; real 248320 lm_head loaded below
    let real_vocab = 248320usize;
    cfg.num_heads = 16;
    cfg.num_kv_heads = 4;
    cfg.head_dim = 128;
    cfg.value_head_dim = 128;
    cfg.num_layers = 0;
    cfg.layer_types = vec![];
    cfg.attention_compress_ratios = vec![];

    let mut t = HashMap::new();
    let h = cfg.hidden_size;
    let q80 = DType {
        arith: ArithType::F32,
        storage: Storage::KQuant(KQuantScheme::Q80),
    };
    let d_le = 0x2400u16.to_le_bytes();
    let mut s = 919u32;
    let mut q_bytes = Vec::with_capacity((h * real_vocab) / 32 * 34);
    for _ in 0..(h * real_vocab / 32) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        q_bytes.extend_from_slice(&d_le);
        for j in 0..32u8 {
            q_bytes.push((((s >> (j % 16)) & 0x1F) as i8 - 16) as u8);
        }
    }
    t.insert(
        "model.embed_tokens.weight".to_string(),
        raw_f32(1, vec![h, cfg.vocab_size]),
    );
    t.insert("model.output_norm.weight".to_string(), raw_ones(vec![h]));
    t.insert(
        "model.hyper_connection_mixer.hc_norm.weight".to_string(),
        raw_ones(vec![h]),
    );
    t.insert(
        "lm_head.weight".to_string(),
        RawTensor {
            bytes: q_bytes,
            shape: vec![real_vocab, h],
            dtype: q80,
            provenance: QuantProvenance::GrimNative,
        },
    );
    let provider = MemProvider { tensors: t };
    let dev = Device::Rocm(0);
    let ws = WeightSource::root(&provider, dev.clone());
    let mut model = Qwen38FlashNext::load(dev.clone(), &ws, cfg.clone())?;
    model.output = grim_nn::Linear::load_shape(&ws.scoped("lm_head"), [h, real_vocab])?;

    // Warmup
    {
        let warm_cpu = grim_backend_cpu::cpu_tensor(vec![0.05f32; 8 * h], Shape::new(vec![8, h]));
        let warm_dev = grim_nn::modules::move_to_device(&warm_cpu, &dev)?;
        let mut sess = model.new_session();
        let _ = model.project_lm_head(sess.as_mut(), &warm_dev)?;
        grim_backend_rocm::RocmDevice::shared(0).synchronize();
    }

    // GPU parity check at seq_len=8 (OFF vs ON)
    {
        let cpu_h = raw_f32(42, vec![8, h]);
        let vals: Vec<f32> = cpu_h
            .bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let norm_cpu = grim_backend_cpu::cpu_tensor(vals, Shape::new(vec![8, h]));
        let norm_dev = grim_nn::modules::move_to_device(&norm_cpu, &dev)?;

        let off = {
            let _e = EnvGuard::set("GRIM_FLASH_LAST_ROW_LMHEAD", "0");
            let mut sess = model.new_session();
            let out = model.project_lm_head(sess.as_mut(), &norm_dev)?;
            grim_backend_rocm::RocmDevice::shared(0).synchronize();
            assert_eq!(out.shape().dims(), &[8, real_vocab]);
            out.to_vec_f32()?
        };
        let on = {
            let _e = EnvGuard::set("GRIM_FLASH_LAST_ROW_LMHEAD", "1");
            let mut sess = model.new_session();
            let out = model.project_lm_head(sess.as_mut(), &norm_dev)?;
            grim_backend_rocm::RocmDevice::shared(0).synchronize();
            assert_eq!(out.shape().dims(), &[1, real_vocab]);
            out.to_vec_f32()?
        };
        let off_last = &off[7 * real_vocab..];
        let mut max_diff = 0.0f32;
        let mut top1_on = (0usize, f32::NEG_INFINITY);
        let mut top1_off = (0usize, f32::NEG_INFINITY);
        for c in 0..real_vocab {
            max_diff = max_diff.max((on[c] - off_last[c]).abs());
            if on[c] > top1_on.1 {
                top1_on = (c, on[c]);
            }
            if off_last[c] > top1_off.1 {
                top1_off = (c, off_last[c]);
            }
        }
        eprintln!(
            "[Q1a-PARITY] GPU seq=8 max_abs_diff={max_diff:.6e} top1_on={} top1_off={}",
            top1_on.0, top1_off.0
        );
        assert_eq!(top1_on.0, top1_off.0, "GPU last-row top-1 mismatch");
        assert!(max_diff < 1e-2, "GPU last-row parity diff={max_diff}");
    }

    let median3 = |mut v: Vec<f64>| -> f64 {
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };

    for seq_len in [512usize, 2048usize] {
        let cpu = grim_backend_cpu::cpu_tensor(vec![0.05f32; seq_len * h], Shape::new(vec![seq_len, h]));
        let normed = grim_nn::modules::move_to_device(&cpu, &dev)?;
        for flag in ["0", "1"] {
            let _e = EnvGuard::set("GRIM_FLASH_LAST_ROW_LMHEAD", flag);
            let mut times = Vec::with_capacity(3);
            for _ in 0..3 {
                let mut sess = model.new_session();
                let t0 = Instant::now();
                let out = model.project_lm_head(sess.as_mut(), &normed)?;
                let _ = out.shape();
                grim_backend_rocm::RocmDevice::shared(0).synchronize();
                times.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            eprintln!(
                "[Q1a-BENCH] seq_len={seq_len} flag={flag}: runs={times:?} ms, median={:.3} ms",
                median3(times.clone())
            );
        }
    }
    Ok(())
}
