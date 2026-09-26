//! ROCm-gated seams for the Qwen3.8-Flash-Next hybrid path.
//!
//! # What this covers that the CPU-only synthetic test cannot
//!
//! The Qwen3.8 GDN and QSA paths are host reference code: the GDN recurrence
//! and the QSA indexer both run on the CPU, then hand a result back to a
//! device-resident block. On a ROCm run that host result MUST be relocated onto
//! the model's device before `wo` and the rest of the block consume it. A CPU
//! test cannot observe that seam, because on the host the relocation is a
//! no-op and "forgot to relocate" is indistinguishable from "relocated".
//!
//! So each test here has two legs:
//!
//!   - a CPU twin that ALWAYS runs, proving the harness compiles, the synthetic
//!     checkpoint loads on both devices, and the numeric contract holds;
//!   - a GPU leg behind `#[ignore]` + `gpu_test_enabled()` + `probe_one(0)`,
//!     which asserts the device of every tensor crossing the seam.
//!
//! # Seams audited
//!
//!  1. the sparse-attention result lands on the model's device, not the host
//!  2. the GDN conv state and the QSA indexer key history stay per-session and
//!     per-layer, and a ROCm run is reproducible run-to-run
//!
//! Run with: GRIM_GPU_TEST=1 cargo test -p grim-models-transformer --test
//! qwen38_rocm_seam -- --ignored

use std::collections::HashMap;

use grim_core::model::{CausalLm, Model};
use grim_nn::WeightSource;
use grim_tensor::provider::{RawTensor, TensorMeta, TensorProvider};
use grim_tensor::{CoreTensorOps, DType, Device, QuantProvenance, Shape};

use grim_models_transformer::qwen38_flash_next::{Qwen38FlashNext, Qwen38FlashNextConfig};

// ---------------------------------------------------------------------------
// Synthetic checkpoint
// ---------------------------------------------------------------------------

fn raw_f32(seed: u32, shape: Vec<usize>) -> RawTensor {
    let n: usize = shape.iter().product();
    let mut bytes = Vec::with_capacity(n * 4);
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    for _ in 0..n {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let v = ((state >> 8) as f32 / 8_388_608.0) - 1.0;
        bytes.extend_from_slice(&(v * 0.05).to_le_bytes());
    }
    RawTensor {
        bytes,
        shape,
        dtype: DType::F32,
        provenance: QuantProvenance::GrimNative,
    }
}

fn raw_f32_ones(shape: Vec<usize>) -> RawTensor {
    let n: usize = shape.iter().product();
    let bytes: Vec<u8> = std::iter::repeat(0)
        .flat_map(|_| 1.0f32.to_le_bytes())
        .take(n * 4)
        .collect();
    RawTensor {
        bytes,
        shape,
        dtype: DType::F32,
        provenance: QuantProvenance::GrimNative,
    }
}

struct SynthProvider {
    tensors: HashMap<String, RawTensor>,
}

impl TensorProvider for SynthProvider {
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

/// Same geometry as the CPU-only synthetic test, deliberately with
/// `ssm_d_state != linear_value_head_dim` so the two are distinguishable.
fn synth_cfg(n_layers: usize) -> Qwen38FlashNextConfig {
    let mut cfg = Qwen38FlashNextConfig::default();
    cfg.vocab_size = 64;
    cfg.hidden_size = 32;
    cfg.num_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.head_dim = 8;
    cfg.value_head_dim = 8;
    cfg.num_layers = n_layers;
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
    cfg.indexer_n_heads = 2;
    cfg.indexer_key_length = 4;
    // Small enough that the selection prunes once history accumulates.
    cfg.indexer_top_k = 2;
    cfg.ngram_vocab_size = None;
    cfg.ngram_dim = None;
    cfg.max_seq_len = 64;
    cfg.layer_types = (0..n_layers)
        .map(|i| {
            if i + 1 == n_layers {
                "full_attention".to_string()
            } else {
                "linear_attention".to_string()
            }
        })
        .collect();
    cfg.attention_compress_ratios = (0..n_layers)
        .map(|i| if i + 1 == n_layers { 2 } else { 0 })
        .collect();
    cfg
}

fn synth_tensors(cfg: &Qwen38FlashNextConfig) -> SynthProvider {
    let mut t: HashMap<String, RawTensor> = HashMap::new();
    let h = cfg.hidden_size;
    let hc_dim = h * cfg.hc_count;
    let q_dim = cfg.num_heads * cfg.head_dim;
    let kv_dim = cfg.num_kv_heads * cfg.head_dim;
    let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
    let value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
    let conv_dim = key_dim * 2 + value_dim;
    let idx_d = cfg.indexer_key_length;
    let n_idx = cfg.indexer_n_heads;
    let hc_lr = cfg.hc_lowrank;
    let mut put = |name: String, r: RawTensor| {
        t.insert(format!("model.{name}"), r);
    };

    put(
        "embed_tokens.weight".into(),
        raw_f32(1, vec![h, cfg.vocab_size]),
    );
    put("output_norm.weight".into(), raw_f32_ones(vec![h]));
    put(
        "hyper_connection_mixer.hc_norm.weight".into(),
        raw_f32_ones(vec![h]),
    );
    put("output.weight".into(), raw_f32(2, vec![h, cfg.vocab_size]));

    for i in 0..cfg.num_layers {
        let p = format!("layers.{i}");
        let s = (i as u32 + 10) * 977;
        for br in ["attn", "ffn"] {
            put(
                format!("{p}.hc_{br}_norm.weight"),
                raw_f32_ones(vec![hc_dim]),
            );
            put(
                format!("{p}.hc_{br}_down.weight"),
                raw_f32(s + 1, vec![hc_lr, hc_dim]),
            );
            put(
                format!("{p}.hc_{br}_up.weight"),
                raw_f32(s + 2, vec![hc_dim, hc_lr]),
            );
            put(
                format!("{p}.hc_{br}_inject.weight"),
                raw_f32(s + 3, vec![cfg.hc_count, hc_dim]),
            );
        }
        put(format!("{p}.input_layernorm.weight"), raw_f32_ones(vec![h]));
        put(
            format!("{p}.post_attention_layernorm.weight"),
            raw_f32_ones(vec![h]),
        );
        put(
            format!("{p}.ffn_gate_inp.weight"),
            raw_f32(s + 4, vec![cfg.num_experts, h]),
        );
        put(
            format!("{p}.ffn_gate_inp_shexp.weight"),
            raw_f32(s + 5, vec![1, h]),
        );
        put(
            format!("{p}.ffn_gate_exps.weight"),
            raw_f32(s + 6, vec![cfg.num_experts, cfg.intermediate_size, h]),
        );
        put(
            format!("{p}.ffn_up_exps.weight"),
            raw_f32(s + 7, vec![cfg.num_experts, cfg.intermediate_size, h]),
        );
        put(
            format!("{p}.ffn_down_exps.weight"),
            raw_f32(s + 8, vec![cfg.num_experts, h, cfg.intermediate_size]),
        );
        let shared = cfg
            .shared_expert_intermediate_size
            .unwrap_or(cfg.intermediate_size);
        put(
            format!("{p}.ffn_gate_shexp.weight"),
            raw_f32(s + 9, vec![shared, h]),
        );
        put(
            format!("{p}.ffn_up_shexp.weight"),
            raw_f32(s + 10, vec![shared, h]),
        );
        put(
            format!("{p}.ffn_down_shexp.weight"),
            raw_f32(s + 11, vec![h, shared]),
        );

        if cfg.layer_types[i] == "full_attention" {
            put(
                format!("{p}.attn_q.weight"),
                raw_f32(s + 12, vec![q_dim, h]),
            );
            put(
                format!("{p}.attn_k.weight"),
                raw_f32(s + 13, vec![kv_dim, h]),
            );
            put(
                format!("{p}.attn_v.weight"),
                raw_f32(s + 14, vec![kv_dim, h]),
            );
            put(
                format!("{p}.attn_output.weight"),
                raw_f32(s + 15, vec![h, q_dim]),
            );
            put(
                format!("{p}.attn_q_norm.weight"),
                raw_f32_ones(vec![cfg.head_dim]),
            );
            put(
                format!("{p}.attn_k_norm.weight"),
                raw_f32_ones(vec![cfg.head_dim]),
            );
            put(
                format!("{p}.indexer.q_proj.weight"),
                raw_f32(s + 16, vec![n_idx * idx_d, h]),
            );
            put(
                format!("{p}.indexer.k_proj.weight"),
                raw_f32(s + 17, vec![idx_d, h]),
            );
            put(
                format!("{p}.indexer.q_norm.weight"),
                raw_f32_ones(vec![idx_d]),
            );
            put(
                format!("{p}.indexer.k_norm.weight"),
                raw_f32_ones(vec![idx_d]),
            );
        } else {
            put(
                format!("{p}.attn_qkv.weight"),
                raw_f32(s + 12, vec![conv_dim, h]),
            );
            put(
                format!("{p}.attn_gate.weight"),
                raw_f32(s + 13, vec![value_dim, h]),
            );
            put(
                format!("{p}.ssm_conv1d.weight"),
                raw_f32(s + 14, vec![conv_dim, cfg.linear_conv_kernel_dim]),
            );
            put(format!("{p}.ssm_a"), raw_f32(s + 15, vec![cfg.ssm_dt_rank]));
            put(
                format!("{p}.ssm_dt.bias"),
                raw_f32(s + 16, vec![cfg.ssm_dt_rank]),
            );
            put(
                format!("{p}.ssm_alpha.weight"),
                raw_f32(s + 17, vec![cfg.ssm_dt_rank, h]),
            );
            put(
                format!("{p}.ssm_beta.weight"),
                raw_f32(s + 18, vec![cfg.ssm_dt_rank, h]),
            );
            put(
                format!("{p}.ssm_norm.weight"),
                raw_f32_ones(vec![cfg.ssm_d_state]),
            );
            put(
                format!("{p}.ssm_out.weight"),
                raw_f32(s + 19, vec![h, value_dim]),
            );
        }
    }
    SynthProvider { tensors: t }
}

fn build_model(cfg: &Qwen38FlashNextConfig, device: Device) -> Qwen38FlashNext {
    let provider = synth_tensors(cfg);
    let ws = WeightSource::root(&provider, device.clone());
    Qwen38FlashNext::load(device, &ws, cfg.clone())
        .expect("the synthetic qwen4exp checkpoint must load")
}

/// Build an input tensor on `device`. The forward takes token ids and
/// positions, and a ROCm model's forward reads them through device kernels, so
/// a CPU tensor here fails with "storage is not CpuStorage" the moment the model
/// is not on the host.
fn device_f32(device: &Device, data: Vec<f32>, shape: Shape) -> grim_tensor::Tensor {
    match device {
        Device::Cpu => grim_backend_cpu::cpu_tensor(data, shape),
        Device::Rocm(ord) => {
            let dev = grim_backend_rocm::RocmDevice::shared(*ord);
            let storage = dev
                .from_cpu(&data, &shape, DType::F32)
                .expect("upload input to ROCm");
            grim_tensor::Tensor::new(
                std::sync::Arc::from(storage),
                shape,
                DType::F32,
                QuantProvenance::GrimNative,
                Device::Rocm(*ord),
            )
        }
        other => panic!("unsupported device for this test: {other:?}"),
    }
}

fn decode(
    model: &Qwen38FlashNext,
    session: &mut dyn grim_core::session::SessionT,
    tok: u32,
) -> grim_tensor::Tensor {
    let device = model.device().clone();
    let ids = device_f32(&device, vec![tok as f32], Shape::new(vec![1]));
    let pos = device_f32(&device, vec![0.0f32], Shape::new(vec![1]));
    model
        .forward(session, &ids, &pos, &[])
        .expect("forward must succeed")
}

/// Run `steps` decode steps and return the final logits.
fn decode_n(
    model: &Qwen38FlashNext,
    session: &mut dyn grim_core::session::SessionT,
    steps: u32,
) -> grim_tensor::Tensor {
    let mut out = None;
    for t in 0..steps {
        out = Some(decode(model, session, t + 1));
    }
    out.expect("at least one step")
}

// ---------------------------------------------------------------------------
// CPU twin leg: always runs, proves the harness and the contract
// ---------------------------------------------------------------------------

#[test]
fn cpu_leg_sparse_forward_is_finite_and_on_the_host() {
    let cfg = synth_cfg(3);
    let model = build_model(&cfg, Device::Cpu);
    let mut session = model.new_session();
    let logits = decode_n(&model, session.as_mut(), 8);
    assert_eq!(*logits.device(), Device::Cpu);
    let v = logits.to_vec_f32().expect("logits");
    assert_eq!(v.len(), cfg.vocab_size);
    assert!(v.iter().all(|x| x.is_finite()), "CPU logits must be finite");
}

#[test]
fn cpu_leg_sparse_and_dense_differ() {
    // The gate must be observable on any device: forcing dense has to change
    // the numbers, otherwise the indexer mask is not being applied anywhere.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let cfg = synth_cfg(3);
    let sparse = {
        let model = build_model(&cfg, Device::Cpu);
        let mut s = model.new_session();
        decode_n(&model, s.as_mut(), 8)
            .to_vec_f32()
            .expect("logits")
    };
    // SAFETY: ENV_LOCK is held for the whole window and the model does no
    // other thread creation, so no other thread can read the environment.
    unsafe { std::env::set_var("GRIM_QWEN38_QSA", "0") };
    let dense = {
        let model = build_model(&cfg, Device::Cpu);
        let mut s = model.new_session();
        decode_n(&model, s.as_mut(), 8)
            .to_vec_f32()
            .expect("logits")
    };
    unsafe { std::env::remove_var("GRIM_QWEN38_QSA") };

    assert_ne!(
        sparse, dense,
        "forcing dense must change the result; the QSA mask is not applied"
    );
}

// ---------------------------------------------------------------------------
// GPU leg: the actual seam audit
// ---------------------------------------------------------------------------

/// Both gates, matching the existing transformer GPU tests. Either alone is
/// wrong: the env flag can be set on a box with ROCm installed but no GPU, and
/// a machine with a GPU but no flag should skip in CI.
fn gpu_available() -> bool {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("skip: set GRIM_GPU_TEST=1 for the ROCm seam tests");
        return false;
    }
    if !grim_backend_rocm::RocmDevice::probe_one(0).unwrap_or(false) {
        eprintln!("skip: no ROCm ordinal 0");
        return false;
    }
    true
}

#[test]
#[ignore = "requires GRIM_GPU_TEST=1 and a ROCm device at ordinal 0"]
fn rocm_sparse_forward_keeps_every_tensor_on_the_device() {
    if !gpu_available() {
        return;
    }
    let cfg = synth_cfg(3);
    let model = build_model(&cfg, Device::Rocm(0));
    let mut session = model.new_session();
    let logits = decode_n(&model, session.as_mut(), 8);

    // THE SEAM. The QSA masked softmax and the GDN recurrence are host code;
    // their result must be relocated before the rest of the block consumes it.
    // If it stayed on the host, the block would mix devices and this returns a
    // host tensor while the model claims ROCm.
    assert_eq!(
        *logits.device(),
        Device::Rocm(0),
        "the sparse path returned a tensor on the wrong device; the host \\
         result was not relocated before the block continued"
    );
    let v = logits.to_vec_f32().expect("logits");
    assert!(
        v.iter().all(|x| x.is_finite()),
        "ROCm logits must be finite"
    );
}

#[test]
#[ignore = "requires GRIM_GPU_TEST=1 and a ROCm device at ordinal 0"]
fn rocm_gate_switches_sparse_and_dense_without_changing_device() {
    if !gpu_available() {
        return;
    }
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let cfg = synth_cfg(3);
    let sparse = {
        let model = build_model(&cfg, Device::Rocm(0));
        let mut s = model.new_session();
        decode_n(&model, s.as_mut(), 8)
    };
    // SAFETY: ENV_LOCK held; the model creates no threads.
    unsafe { std::env::set_var("GRIM_QWEN38_QSA", "0") };
    let dense = {
        let model = build_model(&cfg, Device::Rocm(0));
        let mut s = model.new_session();
        decode_n(&model, s.as_mut(), 8)
    };
    unsafe { std::env::remove_var("GRIM_QWEN38_QSA") };

    assert_eq!(*sparse.device(), Device::Rocm(0), "sparse arm device");
    assert_eq!(*dense.device(), Device::Rocm(0), "dense arm device");
    let a = sparse.to_vec_f32().expect("sparse");
    let b = dense.to_vec_f32().expect("dense");
    assert_ne!(a, b, "the QSA gate must change the ROCm result");
}

#[test]
#[ignore = "requires GRIM_GPU_TEST=1 and a ROCm device at ordinal 0"]
fn rocm_prefill_runs_the_multi_token_moe_path() {
    if !gpu_available() {
        return;
    }
    let cfg = synth_cfg(3);
    let model = build_model(&cfg, Device::Rocm(0));
    let mut session = model.new_session();

    // A 4-token step is the ONLY way to reach the multi-token branch of the
    // MoE block, where each token's slice is staged on the host and must be
    // lifted back to the device before the expert matmuls. Single-token decode
    // takes a different branch entirely, so decode-only tests cannot cover it.
    let ids = device_f32(
        &Device::Rocm(0),
        vec![1.0, 2.0, 3.0, 4.0],
        Shape::new(vec![4]),
    );
    let pos = device_f32(
        &Device::Rocm(0),
        vec![0.0, 1.0, 2.0, 3.0],
        Shape::new(vec![4]),
    );
    let logits = model
        .forward(session.as_mut(), &ids, &pos, &[])
        .expect("a 4-token ROCm prefill must succeed");
    assert_eq!(
        *logits.device(),
        Device::Rocm(0),
        "the multi-token MoE path must return a device tensor; the host \
         out_vec was not lifted back"
    );
    let v = logits.to_vec_f32().expect("logits");
    assert_eq!(v.len(), 4 * cfg.vocab_size, "one row per token");
    assert!(
        v.iter().all(|x| x.is_finite()),
        "prefill logits must be finite"
    );
}

#[test]
#[ignore = "requires GRIM_GPU_TEST=1 and a ROCm device at ordinal 0"]
fn rocm_non_gated_experts_run_on_the_device() {
    if !gpu_available() {
        return;
    }
    // The NonGated ReLU^2 experts are a separate bank from the SwiGLU one, and
    // their activation was built on the host. Build the bank directly so this
    // does not depend on which branch Qwen38FlashNext selects.
    let dev = grim_backend_rocm::RocmDevice::shared(0);
    let ord = 0usize;
    let hidden = 8usize;
    let inter = 6usize;
    let n_exp = 3usize;
    let mk = |rows: usize, cols: usize, seed: u32| -> grim_nn::Linear {
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| (((i as u32).wrapping_mul(31).wrapping_add(seed) % 17) as f32) / 40.0 - 0.2)
            .collect();
        let shape = Shape::new(vec![rows, cols]);
        let w = grim_tensor::Tensor::new(
            std::sync::Arc::from(dev.from_cpu(&data, &shape, DType::F32).expect("upload")),
            shape.clone(),
            DType::F32,
            QuantProvenance::GrimNative,
            Device::Rocm(ord),
        );
        let w_t = grim_tensor::Tensor::new(
            std::sync::Arc::from(
                dev.from_cpu(
                    &vec![0.0f32; cols * rows],
                    &Shape::new(vec![cols, rows]),
                    DType::F32,
                )
                .expect("upload"),
            ),
            Shape::new(vec![cols, rows]),
            DType::F32,
            QuantProvenance::GrimNative,
            Device::Rocm(ord),
        );
        grim_nn::Linear {
            weight: w,
            bias: None,
            w_t,
            quant_format: None,
        }
    };
    let bank = grim_nn::moe::NonGatedExpertBank::from_linears(
        (0..n_exp)
            .map(|e| mk(inter, hidden, e as u32 * 7 + 1))
            .collect(),
        (0..n_exp)
            .map(|e| mk(hidden, inter, e as u32 * 7 + 2))
            .collect(),
    );
    let x = device_f32(
        &Device::Rocm(0),
        (0..hidden).map(|i| i as f32 * 0.1).collect(),
        Shape::new(vec![1, hidden]),
    );
    let out = bank
        .expert_forward(1, &x)
        .expect("a device-resident ReLU^2 expert must run");
    assert_eq!(
        *out.device(),
        Device::Rocm(0),
        "expert output must stay on device"
    );
    let v = out.to_vec_f32().expect("expert logits");
    assert!(v.iter().all(|x| x.is_finite()));
}

#[test]
#[ignore = "requires GRIM_GPU_TEST=1 and a ROCm device at ordinal 0"]
fn rocm_recurrent_state_is_session_scoped_and_deterministic() {
    if !gpu_available() {
        return;
    }
    let cfg = synth_cfg(3);

    // Two independent sessions from the same model must not see each other's
    // GDN state or QSA key history. A shared cache would make the second
    // session's output depend on the first.
    let model = build_model(&cfg, Device::Rocm(0));
    let run = |model: &Qwen38FlashNext| {
        let mut s = model.new_session();
        decode_n(model, s.as_mut(), 8).to_vec_f32().expect("logits")
    };
    let a1 = run(&model);
    let b1 = run(&model); // fresh session, same model
    let max_diff = a1
        .iter()
        .zip(&b1)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-5,
        "a fresh session must reproduce the same logits; delta {max_diff} \\
         suggests the recurrent state leaked across sessions"
    );

    // Two models built independently must also agree: the weights are
    // deterministic, so any difference is state or routing.
    let m2 = build_model(&cfg, Device::Rocm(0));
    let a2 = run(&m2);
    let max_diff2 = a1
        .iter()
        .zip(&a2)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff2 < 1e-4,
        "an independently built model must produce the same logits; delta \\
         {max_diff2}"
    );
    assert!(a1.iter().all(|x| x.is_finite()));
}
