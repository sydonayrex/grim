//! Single-layer GPU repro for the Qwen3.8 forward-pass page fault.
//!
//! The 27B run faults with
//! `Memory access fault by GPU node-2 ... Page not present or supervisor
//! privilege` during the forward pass, AFTER loading all 65 layers and
//! tokenizing correctly. That is a device-memory access violation rather than a
//! capacity problem, and it fires before any token is generated — so a 25-minute
//! model load buys almost no diagnostic signal.
//!
//! This builds ONE recurrent layer at the real Qwen3.8 geometry and runs one
//! forward on a single GPU. Seconds instead of 25 minutes, and it either
//! reproduces the fault or rules this layer out.
//!
//! Runs on ordinal 1 (the RX 9060 XT) rather than 0, so a fault here cannot
//! destabilise the primary device.
//!
//! Gated: `GRIM_GPU_TEST=1`.

use grim_tensor::provider::{RawTensor, TensorMeta, TensorProvider};
use grim_tensor::{DType, QuantProvenance, Result as TResult, Storage};
use std::collections::HashMap;
use std::sync::Arc;

/// Minimal in-memory provider: f32 tensors named exactly as a Qwen35 recurrent
/// layer expects, at the real widths.
struct MapProvider {
    tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl TensorProvider for MapProvider {
    fn get(&self, name: &str) -> TResult<RawTensor> {
        let (data, shape) = self
            .tensors
            .get(name)
            .ok_or_else(|| grim_tensor::Error::Backend(format!("no tensor {name}")))?;
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Ok(RawTensor {
            bytes,
            shape: shape.clone(),
            dtype: DType { arith: grim_tensor::ArithType::F32, storage: Storage::Native },
            provenance: QuantProvenance::GrimNative,
        })
    }

    fn meta(&self, name: &str) -> TResult<TensorMeta> {
        let (_d, shape) = self
            .tensors
            .get(name)
            .ok_or_else(|| grim_tensor::Error::Backend(format!("no tensor {name}")))?;
        Ok(TensorMeta {
            dtype: DType { arith: grim_tensor::ArithType::F32, storage: Storage::Native },
            provenance: QuantProvenance::GrimNative,
            shape: shape.clone(),
            fusion_mask: 0,
        })
    }

    fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }
}

fn gpu1() -> Option<Arc<grim_backend_rocm::RocmDevice>> {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run");
        return None;
    }
    std::panic::catch_unwind(|| {
        Arc::new(grim_backend_rocm::RocmDevice::try_new(1).expect("RocmDevice::try_new(1)"))
    })
    .ok()
}

#[test]
fn single_recurrent_layer_forward_on_gpu() {
    let Some(dev) = gpu1() else { return };
    eprintln!("[repro] running on ordinal 1 (RX 9060 XT)");

    let mut cfg = grim_models_transformer::qwen35::Qwen35Config::default();
    cfg.vocab_size = 64;
    cfg.hidden_size = 5120;
    cfg.num_heads = 24;
    cfg.num_kv_heads = 4;
    cfg.head_dim = 256;
    cfg.num_layers = 65;
    cfg.intermediate_size = 256;
    cfg.full_attention_interval = 4;
    cfg.ssm_d_conv = 4;
    cfg.ssm_d_inner = 6144;
    cfg.ssm_d_state = 128;
    cfg.ssm_dt_rank = 48;
    cfg.ssm_n_group = 16;

    let value_dim = cfg.ssm_dt_rank * cfg.ssm_d_state;
    let key_dim = cfg.ssm_n_group * cfg.ssm_d_state;
    let ssm_qkv_dim = 2 * key_dim + value_dim;
    eprintln!("[repro] ssm_qkv_dim={ssm_qkv_dim} (expect 10240)");

    let mut tensors: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    {
        let mat = |rows: usize, cols: usize| (vec![0.1f32; rows * cols], vec![rows, cols]);
        let vec1 = |n: usize| (vec![0.1f32; n], vec![n]);
        tensors.insert("attn_norm.weight".into(), vec1(cfg.hidden_size));
        tensors.insert("post_attention_norm.weight".into(), vec1(cfg.hidden_size));
        tensors.insert("ssm_a".into(), vec1(cfg.ssm_dt_rank));
        tensors.insert("ssm_dt.bias".into(), vec1(cfg.ssm_dt_rank));
        tensors.insert("ssm_norm.weight".into(), vec1(cfg.ssm_d_state));
        tensors.insert(
            "attn_qkv.weight".into(),
            mat(cfg.hidden_size, ssm_qkv_dim),
        );
        tensors.insert(
            "ssm_alpha.weight".into(),
            mat(cfg.hidden_size, cfg.ssm_dt_rank),
        );
        tensors.insert(
            "ssm_beta.weight".into(),
            mat(cfg.hidden_size, cfg.ssm_dt_rank),
        );
        tensors.insert("ssm_out.weight".into(), mat(value_dim, cfg.hidden_size));
        tensors.insert(
            "ffn_gate.weight".into(),
            mat(cfg.intermediate_size, cfg.hidden_size),
        );
        tensors.insert(
            "ffn_up.weight".into(),
            mat(cfg.intermediate_size, cfg.hidden_size),
        );
        tensors.insert(
            "ffn_down.weight".into(),
            mat(cfg.hidden_size, cfg.intermediate_size),
        );
        tensors.insert(
            "ssm_conv1d.weight".into(),
            mat(cfg.ssm_d_conv, ssm_qkv_dim),
        );
    }

    let provider = MapProvider { tensors };
    let ws = grim_nn::WeightSource::root(&provider, grim_tensor::Device::Rocm(1));
    let blk = match grim_models_transformer::qwen35::Qwen35Block::load_tp(
        &ws,
        &cfg,
        0, // layer 0: (0+1) % 4 != 0 -> recurrent
        grim_nn::TensorParallelConfig::default(),
    ) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[repro] load_tp FAILED: {e}");
            panic!("load_tp: {e}");
        }
    };
    assert!(!blk.is_full_attention, "layer 0 must be recurrent");

    let mut cache = grim_models_transformer::qwen35::Qwen35LayerCache::new(&cfg);
    eprintln!(
        "[repro] conv_state={} ssm_state={} (expect {} and {})",
        cache.conv_state.len(),
        cache.ssm_state.len(),
        (cfg.ssm_d_conv - 1) * ssm_qkv_dim,
        48 * 128 * 128
    );

    let dev_t = grim_nn::modules::pick_device_for_storage_device(&grim_tensor::Device::Rocm(1));
    let x_storage = grim_tensor::CoreTensorOps::from_cpu(
        &dev_t,
        &vec![0.1; cfg.hidden_size],
        &grim_tensor::Shape::new(vec![1, cfg.hidden_size]),
        DType::F32,
    )
    .expect("upload x");
    let x = grim_tensor::Tensor::new(
        Arc::from(x_storage),
        grim_tensor::Shape::new(vec![1, cfg.hidden_size]),
        DType::F32,
        QuantProvenance::GrimNative,
        grim_tensor::Device::Rocm(1),
    );
    let _ = &dev;

    eprintln!("[repro] running forward...");
    let out = blk
        .forward(&x, &[0], &mut cache)
        .expect("single recurrent layer forward");
    let vals = out.to_vec_f32().expect("read output");
    eprintln!(
        "[repro] OK: {} elems, first={:.4}, all_finite={}",
        vals.len(),
        vals[0],
        vals.iter().all(|v| v.is_finite())
    );
    assert!(vals.iter().all(|v| v.is_finite()));
    assert!(
        !cache.ssm_state.iter().all(|v| *v == 0.0),
        "recurrent state must be updated"
    );
}

/// The full-attention counterpart of `single_recurrent_layer_forward_on_gpu`.
///
/// The recurrent layer is clean, so the remaining untested path in a Qwen35
/// forward is the ATTENTION layer — which is where the new fused `attn_q` split
/// (340592ad) and the `2 * q_dim` out-width live. That code has never executed
/// on a device.
///
/// At the real geometry: attn_q is [2 * q_dim, hidden] = [12288, 5120] with
/// q_dim = 24 * 256 = 6144, and attn_output/wo consumes 6144.
#[test]
fn single_attention_layer_forward_on_gpu() {
    let Some(dev) = gpu1() else { return };
    let _ = &dev;
    eprintln!("[repro-attn] running on ordinal 1 (RX 9060 XT)");

    let mut cfg = grim_models_transformer::qwen35::Qwen35Config::default();
    cfg.vocab_size = 64;
    cfg.hidden_size = 5120;
    cfg.num_heads = 24;
    cfg.num_kv_heads = 4;
    cfg.head_dim = 256;
    cfg.num_layers = 65;
    cfg.intermediate_size = 256;
    cfg.full_attention_interval = 4;
    cfg.ssm_d_conv = 4;
    cfg.ssm_d_inner = 6144;
    cfg.ssm_d_state = 128;
    cfg.ssm_dt_rank = 48;
    cfg.ssm_n_group = 16;

    let q_dim = cfg.num_heads * cfg.head_dim; // 6144
    let kv_dim = cfg.num_kv_heads * cfg.head_dim; // 1024

    let mut tensors: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    {
        let mat = |rows: usize, cols: usize| (vec![0.1f32; rows * cols], vec![rows, cols]);
        let vec1 = |n: usize| (vec![0.1f32; n], vec![n]);
        tensors.insert("attn_norm.weight".into(), vec1(cfg.hidden_size));
        tensors.insert("post_attention_norm.weight".into(), vec1(cfg.hidden_size));
        // The fused Q + gate projection, at the REAL width.
        tensors.insert("attn_q.weight".into(), mat(2 * q_dim, cfg.hidden_size));
        tensors.insert("attn_k.weight".into(), mat(kv_dim, cfg.hidden_size));
        tensors.insert("attn_v.weight".into(), mat(kv_dim, cfg.hidden_size));
        tensors.insert("attn_output.weight".into(), mat(cfg.hidden_size, q_dim));
        tensors.insert("attn_q_norm.weight".into(), vec1(cfg.head_dim));
        tensors.insert("attn_k_norm.weight".into(), vec1(cfg.head_dim));
        tensors.insert(
            "ffn_gate.weight".into(),
            mat(cfg.intermediate_size, cfg.hidden_size),
        );
        tensors.insert(
            "ffn_up.weight".into(),
            mat(cfg.intermediate_size, cfg.hidden_size),
        );
        tensors.insert(
            "ffn_down.weight".into(),
            mat(cfg.hidden_size, cfg.intermediate_size),
        );
    }

    let provider = MapProvider { tensors };
    let ws = grim_nn::WeightSource::root(&provider, grim_tensor::Device::Rocm(1));
    // layer 3: (3+1) % 4 == 0 -> full attention
    let blk = match grim_models_transformer::qwen35::Qwen35Block::load_tp(
        &ws,
        &cfg,
        3,
        grim_nn::TensorParallelConfig::default(),
    ) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[repro-attn] load_tp FAILED: {e}");
            panic!("load_tp: {e}");
        }
    };
    assert!(blk.is_full_attention, "layer 3 must be full attention");

    let mut cache = grim_models_transformer::qwen35::Qwen35LayerCache::new(&cfg);
    eprintln!("[repro-attn] running forward...");
    let dev_t = grim_nn::modules::pick_device_for_storage_device(&grim_tensor::Device::Rocm(1));
    let x_storage = grim_tensor::CoreTensorOps::from_cpu(
        &dev_t,
        &vec![0.1; cfg.hidden_size],
        &grim_tensor::Shape::new(vec![1, cfg.hidden_size]),
        DType::F32,
    )
    .expect("upload x");
    let x = grim_tensor::Tensor::new(
        Arc::from(x_storage),
        grim_tensor::Shape::new(vec![1, cfg.hidden_size]),
        DType::F32,
        QuantProvenance::GrimNative,
        grim_tensor::Device::Rocm(1),
    );

    let out = blk
        .forward(&x, &[0], &mut cache)
        .expect("single attention layer forward");
    let vals = out.to_vec_f32().expect("read output");
    eprintln!(
        "[repro-attn] OK: {} elems, first={:.4}, all_finite={}",
        vals.len(),
        vals[0],
        vals.iter().all(|v| v.is_finite())
    );
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "attention output must be finite"
    );
}
