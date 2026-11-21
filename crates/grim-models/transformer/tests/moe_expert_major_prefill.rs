//! TDD tests for Q1b (`GRIM_MOE_EXPERT_MAJOR=1`): expert-major MoE prefill grouping.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result as GrimResult;
use grim_core::CausalLm;
use grim_nn::WeightSource;
use grim_tensor::dtype::{KQuantScheme, Storage};
use grim_tensor::provider::{RawTensor, TensorMeta, TensorProvider};
use grim_tensor::{ArithType, DType, Device, QuantProvenance, Shape, Tensor};

use grim_models_transformer::qwen4exp_flash_next::{Qwen38FlashNext, Qwen38FlashNextConfig};
use grim_models_transformer::qwen4exp_moe_device::{
    expert_major_prefill, moe_expert_major_enabled, route_row_host,
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

fn lcg_vec(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((s >> 33) as f32) / (u32::MAX as f32 * 0.5) - 1.0;
        out.push(u * 0.1);
    }
    out
}

#[test]
fn env_flag_defaults_off_and_parses_truthy() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    {
        let _e = EnvGuard::unset("GRIM_MOE_EXPERT_MAJOR");
        assert!(!moe_expert_major_enabled());
    }
    for v in ["0", "false", ""] {
        let _e = EnvGuard::set("GRIM_MOE_EXPERT_MAJOR", v);
        assert!(!moe_expert_major_enabled(), "value={v}");
    }
    for v in ["1", "true", "TRUE"] {
        let _e = EnvGuard::set("GRIM_MOE_EXPERT_MAJOR", v);
        assert!(moe_expert_major_enabled(), "value={v}");
    }
}

#[test]
fn expert_major_matches_token_major_bitwise_on_4tok_6exp_with_single_all_and_tie() -> GrimResult<()> {
    let seq_len = 4usize;
    let num_exp = 6usize;
    let top_k = 2usize;
    let hidden = 8usize;
    let scale = 1.0f32;

    let x_vec = lcg_vec(12345, seq_len * hidden);
    let x_t = cpu_tensor(x_vec.clone(), Shape::new(vec![seq_len, hidden]));

    // Craft router logits [4, 6]:
    // - Expert 0 selected by ALL 4 tokens (highest logit 5.0).
    // - Expert 5 selected ONLY by token 3 (second highest 3.0).
    // - Token 2 has an exact TIE between expert 2 and expert 4 (both 2.5 -> lower idx 2 wins).
    let logits_vec: Vec<f32> = vec![
        5.0, 2.0, 0.5, 0.1, -0.2, -1.0, // tok 0 -> [0, 1]
        5.0, 0.1, 0.2, 2.5, -0.5, -1.0, // tok 1 -> [0, 3]
        5.0, 0.0, 2.5, 0.1, 2.5, -1.0, // tok 2 -> [0, 2] (tie 2 vs 4 -> lower idx 2)
        5.0, 0.0, 0.1, 0.2, 0.3, 3.0, // tok 3 -> [0, 5] (expert 5 has 1 token)
    ];

    // Deterministic per-expert linear weights [hidden, hidden] + shared weight [hidden, hidden]
    let exp_weights: Vec<Vec<f32>> = (0..num_exp)
        .map(|e| lcg_vec(1000 + e as u64, hidden * hidden))
        .collect();
    let shared_w = lcg_vec(9999, hidden * hidden);

    let apply_linear = |w: &[f32], inp: &Tensor| -> GrimResult<Tensor> {
        let v = inp.to_vec_f32()?;
        let rows = v.len() / hidden;
        let mut out = vec![0.0f32; rows * hidden];
        for r in 0..rows {
            for c in 0..hidden {
                let mut acc = 0.0f32;
                for k in 0..hidden {
                    acc += v[r * hidden + k] * w[c * hidden + k];
                }
                out[r * hidden + c] = acc;
            }
        }
        Ok(cpu_tensor(out, Shape::new(vec![rows, hidden])))
    };

    // Token-major reference (exact loop from qwen4exp_flash_next.rs:867-923)
    let mut ref_out = vec![0.0f32; seq_len * hidden];
    for s in 0..seq_len {
        let (ids, weights) = route_row_host(&logits_vec[s * num_exp..(s + 1) * num_exp], top_k, scale);
        if s == 2 {
            assert_eq!(ids, vec![0, 2], "tie between 2 and 4 must pick lower index 2");
        }
        let tok_x = cpu_tensor(
            x_vec[s * hidden..(s + 1) * hidden].to_vec(),
            Shape::new(vec![1, hidden]),
        );
        for (slot, &exp_idx) in ids.iter().enumerate() {
            let w = weights[slot];
            let eo = apply_linear(&exp_weights[exp_idx], &tok_x)?.to_vec_f32()?;
            for d in 0..hidden {
                ref_out[s * hidden + d] += w * eo[d];
            }
        }
        let so = apply_linear(&shared_w, &tok_x)?.to_vec_f32()?;
        for d in 0..hidden {
            ref_out[s * hidden + d] += so[d];
        }
    }

    // Expert-major under test
    let mut expert_calls = 0usize;
    let em_out = expert_major_prefill(
        &x_t,
        &x_vec,
        &logits_vec,
        seq_len,
        hidden,
        num_exp,
        top_k,
        scale,
        true,
        |exp_idx, batch_x| {
            expert_calls += 1;
            apply_linear(&exp_weights[exp_idx], batch_x)
        },
        |all_x| apply_linear(&shared_w, all_x),
    )?;

    // Distinct experts used: {0, 1, 2, 3, 5} = 5 calls (instead of 4*2 = 8 calls)
    assert_eq!(expert_calls, 5, "each distinct active expert must be called once");
    assert_eq!(em_out.len(), ref_out.len());
    for (i, (&a, &b)) in em_out.iter().zip(ref_out.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "bitwise mismatch at idx {i}: expert_major={a} token_major={b}"
        );
    }
    Ok(())
}

#[ignore = "ROCm Card 1 MoE expert-major prefill parity & A/B benchmark"]
#[test]
fn bench_moe_expert_major_prefill_gpu_card1() -> GrimResult<()> {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut cfg = Qwen38FlashNextConfig::default();
    cfg.num_layers = 1;
    cfg.layer_types = vec!["linear_attention".into()];
    cfg.attention_compress_ratios = vec![0];
    cfg.ngram_vocab_size = None;
    cfg.ngram_dim = None;
    cfg.vocab_size = 64;

    let h = cfg.hidden_size; // 2560
    let inter = cfg.intermediate_size; // 640
    let key_dim = cfg.ssm_n_group * cfg.ssm_d_state;
    let val_dim = cfg.ssm_dt_rank * cfg.ssm_d_state;
    let conv_dim = 2 * key_dim + val_dim;

    let q80 = DType {
        arith: ArithType::F32,
        storage: Storage::KQuant(KQuantScheme::Q80),
    };
    let make_q80 = |rows: usize, cols: usize, seed: u32| -> Vec<u8> {
        let blocks = cols / 32;
        let mut out = vec![0u8; rows * blocks * 34];
        let d_le = 0x2400u16.to_le_bytes();
        let mut s = seed.wrapping_mul(747796405).wrapping_add(2891336453);
        for r in 0..rows {
            for b in 0..blocks {
                let off = (r * blocks + b) * 34;
                out[off] = d_le[0];
                out[off + 1] = d_le[1];
                for q in 0..32 {
                    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                    out[off + 2 + q] = (((s >> 24) as i8) / 4) as u8;
                }
            }
        }
        out
    };
    let make_f32 = |n: usize, val: f32| -> Vec<u8> {
        let mut out = Vec::with_capacity(n * 4);
        let b = val.to_le_bytes();
        for _ in 0..n {
            out.extend_from_slice(&b);
        }
        out
    };

    struct BenchProvider {
        fixed: HashMap<String, (Vec<usize>, DType, Arc<Vec<u8>>)>,
        gate_up: Arc<Vec<u8>>,
        down: Arc<Vec<u8>>,
        h: usize,
        inter: usize,
        q80: DType,
    }
    impl TensorProvider for BenchProvider {
        fn get(&self, name: &str) -> grim_tensor::error::Result<RawTensor> {
            let (shape, dtype, bytes) = if let Some(v) = self.fixed.get(name) {
                v.clone()
            } else if name.contains(".mlp.experts.") {
                if name.ends_with(".gate_proj.weight") || name.ends_with(".up_proj.weight") {
                    (vec![self.inter, self.h], self.q80.clone(), self.gate_up.clone())
                } else if name.ends_with(".down_proj.weight") {
                    (vec![self.h, self.inter], self.q80.clone(), self.down.clone())
                } else {
                    return Err(grim_tensor::error::Error::Backend(format!("missing {name}")));
                }
            } else {
                return Err(grim_tensor::error::Error::Backend(format!("missing {name}")));
            };
            Ok(RawTensor {
                bytes: (*bytes).clone(),
                shape,
                dtype,
                provenance: QuantProvenance::GrimNative,
            })
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
            self.fixed.keys().cloned().collect()
        }
    }

    let mut fixed: HashMap<String, (Vec<usize>, DType, Arc<Vec<u8>>)> = HashMap::new();
    fixed.insert(
        "model.embed_tokens.weight".into(),
        (vec![h, cfg.vocab_size], DType::F32, Arc::new(make_f32(h * cfg.vocab_size, 0.01))),
    );
    fixed.insert(
        "model.norm.weight".into(),
        (vec![h], DType::F32, Arc::new(make_f32(h, 1.0))),
    );
    let p = "model.layers.0";
    fixed.insert(format!("{p}.input_layernorm.weight"), (vec![h], DType::F32, Arc::new(make_f32(h, 1.0))));
    fixed.insert(format!("{p}.post_attention_layernorm.weight"), (vec![h], DType::F32, Arc::new(make_f32(h, 1.0))));
    fixed.insert(format!("{p}.mlp.gate.weight"), (vec![cfg.num_experts, h], q80.clone(), Arc::new(make_q80(cfg.num_experts, h, 11))));
    fixed.insert(format!("{p}.ffn_gate_inp_shexp.weight"), (vec![1, h], DType::F32, Arc::new(make_f32(h, 0.01))));
    fixed.insert(format!("{p}.ffn_gate_shexp.weight"), (vec![inter, h], q80.clone(), Arc::new(make_q80(inter, h, 12))));
    fixed.insert(format!("{p}.ffn_up_shexp.weight"), (vec![inter, h], q80.clone(), Arc::new(make_q80(inter, h, 13))));
    fixed.insert(format!("{p}.ffn_down_shexp.weight"), (vec![h, inter], q80.clone(), Arc::new(make_q80(h, inter, 14))));
    fixed.insert(format!("{p}.attn_qkv.weight"), (vec![conv_dim, h], q80.clone(), Arc::new(make_q80(conv_dim, h, 15))));
    fixed.insert(format!("{p}.attn_gate.weight"), (vec![val_dim, h], q80.clone(), Arc::new(make_q80(val_dim, h, 16))));
    fixed.insert(format!("{p}.ssm_conv1d.weight"), (vec![conv_dim, cfg.linear_conv_kernel_dim], DType::F32, Arc::new(make_f32(conv_dim * cfg.linear_conv_kernel_dim, 0.05))));
    fixed.insert(format!("{p}.ssm_a"), (vec![cfg.ssm_dt_rank], DType::F32, Arc::new(make_f32(cfg.ssm_dt_rank, -0.5))));
    fixed.insert(format!("{p}.ssm_dt.bias"), (vec![cfg.ssm_dt_rank], DType::F32, Arc::new(make_f32(cfg.ssm_dt_rank, 0.1))));
    fixed.insert(format!("{p}.ssm_alpha.weight"), (vec![cfg.ssm_dt_rank, h], q80.clone(), Arc::new(make_q80(cfg.ssm_dt_rank, h, 18))));
    fixed.insert(format!("{p}.ssm_beta.weight"), (vec![cfg.ssm_dt_rank, h], q80.clone(), Arc::new(make_q80(cfg.ssm_dt_rank, h, 19))));
    fixed.insert(format!("{p}.ssm_norm.weight"), (vec![cfg.ssm_d_state], DType::F32, Arc::new(make_f32(cfg.ssm_d_state, 1.0))));
    fixed.insert(format!("{p}.ssm_out.weight"), (vec![h, val_dim], q80.clone(), Arc::new(make_q80(h, val_dim, 17))));

    let provider = BenchProvider {
        fixed,
        gate_up: Arc::new(make_q80(inter, h, 777)),
        down: Arc::new(make_q80(h, inter, 888)),
        h,
        inter,
        q80,
    };
    let dev = Device::Rocm(0);
    let ws = WeightSource::root(&provider, dev.clone());
    let model = Qwen38FlashNext::load(dev.clone(), &ws, cfg.clone())?;

    // Parity at seq_len=8
    {
        let ids = cpu_tensor((0..8).map(|i| (i % 60) as f32).collect(), Shape::new(vec![8]));
        let pos = cpu_tensor((0..8).map(|i| i as f32).collect(), Shape::new(vec![8]));
        let off = {
            let _e = EnvGuard::set("GRIM_MOE_EXPERT_MAJOR", "0");
            let mut s = model.new_session();
            let out = model.forward(s.as_mut(), &ids, &pos, &[])?;
            grim_backend_rocm::RocmDevice::shared(0).synchronize();
            out.to_vec_f32()?
        };
        let on = {
            let _e = EnvGuard::set("GRIM_MOE_EXPERT_MAJOR", "1");
            let mut s = model.new_session();
            let out = model.forward(s.as_mut(), &ids, &pos, &[])?;
            grim_backend_rocm::RocmDevice::shared(0).synchronize();
            out.to_vec_f32()?
        };
        let max_diff = off
            .iter()
            .zip(on.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[Q1b-PARITY] GPU seq=8 max_abs_diff={max_diff:.6e}");
        assert!(max_diff < 1e-2, "GPU expert-major parity diff={max_diff}");
    }

    let median3 = |mut v: Vec<f64>| -> f64 {
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };

    // 3-run A/B at seq_len=64
    let seq_len = 64usize;
    let ids = cpu_tensor((0..seq_len).map(|i| (i % 60) as f32).collect(), Shape::new(vec![seq_len]));
    let pos = cpu_tensor((0..seq_len).map(|i| i as f32).collect(), Shape::new(vec![seq_len]));
    for flag in ["0", "1"] {
        let _e = EnvGuard::set("GRIM_MOE_EXPERT_MAJOR", flag);
        let mut times = Vec::with_capacity(3);
        for _ in 0..3 {
            let mut s = model.new_session();
            let t0 = Instant::now();
            let _ = model.forward(s.as_mut(), &ids, &pos, &[])?;
            grim_backend_rocm::RocmDevice::shared(0).synchronize();
            times.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        eprintln!(
            "[Q1b-BENCH] seq_len={seq_len} flag={flag}: runs={times:?} ms, median={:.3} ms",
            median3(times.clone())
        );
    }
    Ok(())
}
