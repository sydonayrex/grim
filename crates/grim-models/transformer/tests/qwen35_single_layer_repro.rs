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

/// One weight in whichever dtype the checkpoint would actually carry.
enum Rep {
    F32(Vec<f32>, Vec<usize>),
    Q4K(Vec<u8>, Vec<usize>),
}

impl Rep {
    fn shape(&self) -> &[usize] {
        match self {
            Rep::F32(_, s) | Rep::Q4K(_, s) => s,
        }
    }

    fn dtype(&self) -> DType {
        match self {
            Rep::F32(..) => DType {
                arith: grim_tensor::ArithType::F32,
                storage: Storage::Native,
            },
            Rep::Q4K(..) => DType {
                arith: grim_tensor::ArithType::F32,
                storage: Storage::KQuant(grim_tensor::dtype::KQuantScheme::Q4K),
            },
        }
    }
}

/// Minimal in-memory provider: tensors named exactly as a Qwen35 recurrent layer
/// expects, at the real widths.
///
/// Matrices are packed to Q4_K rather than left f32, because that is what the
/// real checkpoint carries and the difference decides whether this repro can
/// cover the model's full depth. At f32 the 27B geometry costs 563 MB per layer
/// and 65 layers ask for ~36.6 GB — more than twice a 17.1 GB card — which
/// spilled to managed memory and failed a module load. At Q4_K the same tensors
/// cost ~78 MB per layer, so all 65 layers fit in about 5.1 GB. One-dimensional
/// norms and biases stay f32, as they do in the checkpoint: they are negligible
/// in bytes, and their element counts are not multiples of the 256-element
/// super-block.
struct MapProvider {
    tensors: HashMap<String, Rep>,
}

/// Pack a weight into the dtype a real checkpoint would carry: 2-D projections
/// as Q4_K super-blocks, 1-D norms and biases left as f32.
fn rep(data: Vec<f32>, shape: Vec<usize>) -> Rep {
    if shape.len() == 2 {
        Rep::Q4K(
            grim_quant::quant_q4k(&data).expect("Q4_K super-block packing"),
            shape,
        )
    } else {
        Rep::F32(data, shape)
    }
}

impl TensorProvider for MapProvider {
    fn get(&self, name: &str) -> TResult<RawTensor> {
        let rep = self
            .tensors
            .get(name)
            .ok_or_else(|| grim_tensor::Error::Backend(format!("no tensor {name}")))?;
        let bytes: Vec<u8> = match rep {
            Rep::F32(data, _) => data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            Rep::Q4K(bytes, _) => bytes.clone(),
        };
        Ok(RawTensor {
            bytes,
            shape: rep.shape().to_vec(),
            dtype: rep.dtype(),
            provenance: QuantProvenance::GrimNative,
        })
    }

    fn meta(&self, name: &str) -> TResult<TensorMeta> {
        let rep = self
            .tensors
            .get(name)
            .ok_or_else(|| grim_tensor::Error::Backend(format!("no tensor {name}")))?;
        Ok(TensorMeta {
            dtype: rep.dtype(),
            provenance: QuantProvenance::GrimNative,
            shape: rep.shape().to_vec(),
            fusion_mask: 0,
        })
    }

    fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    /// The default shard path refuses quantized bytes outright, because a
    /// super-block cannot be split at an arbitrary byte offset. These tests run
    /// `TensorParallelConfig::default()` - world size 1 - where the shard is the
    /// whole tensor and no slicing happens, so short-circuiting is exact rather
    /// than a convenience. Multi-rank callers still get the library helper.
    fn get_packed_sharded(
        &self,
        name: &str,
        dim: usize,
        rank: usize,
        world_size: usize,
    ) -> TResult<RawTensor> {
        if world_size <= 1 {
            return self.get(name);
        }
        grim_tensor::provider::shard_raw_tensor(self.get(name)?, dim, rank, world_size)
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

    let mut tensors: HashMap<String, Rep> = HashMap::new();
    {
        let mat = |rows: usize, cols: usize| rep(vec![0.1f32; rows * cols], vec![rows, cols]);
        let vec1 = |n: usize| rep(vec![0.1f32; n], vec![n]);
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

    let mut tensors: HashMap<String, Rep> = HashMap::new();
    {
        let mat = |rows: usize, cols: usize| rep(vec![0.1f32; rows * cols], vec![rows, cols]);
        let vec1 = |n: usize| rep(vec![0.1f32; n], vec![n]);
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

/// N-layer scaling repro, single device (ordinal 1).
///
/// Both layer types are clean in isolation, so the 27B fault needs SCALE. This
/// builds a stack of N layers with the real alternating pattern ((i+1)%4==0 is
/// attention, otherwise recurrent) and runs one forward through all of them,
/// to find whether the fault appears with depth alone.
///
/// N is capped so a failure stays cheap; the 27B run has 65 layers across two
/// GPUs, and this keeps the same per-layer geometry at a fraction of the cost.
fn build_layer(
    provider: &MapProvider,
    cfg: &grim_models_transformer::qwen35::Qwen35Config,
    idx: usize,
) -> grim_models_transformer::qwen35::Qwen35Block {
    let ws = grim_nn::WeightSource::root(provider, grim_tensor::Device::Rocm(1));
    grim_models_transformer::qwen35::Qwen35Block::load_tp(
        &ws,
        cfg,
        idx,
        grim_nn::TensorParallelConfig::default(),
    )
    .unwrap_or_else(|e| panic!("load layer {idx}: {e}"))
}

#[test]
fn n_layer_stack_forward_on_gpu() {
    let Some(dev) = gpu1() else { return };
    let _ = &dev;

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

    let q_dim = cfg.num_heads * cfg.head_dim;
    let kv_dim = cfg.num_kv_heads * cfg.head_dim;
    let value_dim = cfg.ssm_dt_rank * cfg.ssm_d_state;
    let key_dim = cfg.ssm_n_group * cfg.ssm_d_state;
    let ssm_qkv_dim = 2 * key_dim + value_dim;

    // Depth is bounded by what the card can actually hold, not hardcoded to 65.
    //
    // This model was previously built in f32, which costs ~563 MB per layer at
    // the 27B geometry, so 65 layers asked for ~36.6 GB on a 17.1 GB card. That
    // over-subscription silently spilled to HIP managed memory (`storage.rs`
    // falls back to `hipMallocManaged` whenever `allocator.alloc` fails) and the
    // run died four minutes later with `hipModuleLoad 209` on whichever kernel
    // happened to be loading - `grim_silu_mul` before, `grim_rms_norm` after,
    // which is a resource symptom and not a kernel-specific fault. The message
    // settled the arch question directly: `gpu_target=gfx1200, self.ordinal=1,
    // ctx_device=1, ctx_arch=gfx1200, hsa_override=None`. The code object and
    // the context agreed; 209 was reporting exhaustion.
    //
    // The real checkpoint is Q4_K, and packing to it drops the footprint to
    // ~78 MB per layer, so full 65-layer depth now fits (~5.1 GB) and this repro
    // can finally guard the geometry the model actually has. The clamp stays as
    // a guard: a card too small for the model should say so rather than spill.
    let per_layer_bytes: u64 = {
        // 2-D projections are Q4_K: 144 bytes per 256-element super-block.
        let q4k = |n: usize, m: usize| ((n * m).div_ceil(256) * 144) as u64;
        // 1-D norms and biases stay f32, as they do in the checkpoint.
        let f32v = |n: usize| (n * 4) as u64;
        q4k(2 * q_dim, cfg.hidden_size) // attn_q
            + q4k(kv_dim, cfg.hidden_size) * 2 // attn_k, attn_v
            + q4k(cfg.hidden_size, q_dim) // attn_output
            + q4k(cfg.hidden_size, 48) * 2 // ssm_alpha, ssm_beta
            + q4k(value_dim, cfg.hidden_size) // ssm_out
            + q4k(cfg.ssm_d_conv, ssm_qkv_dim) // ssm_conv1d
            + q4k(cfg.intermediate_size, cfg.hidden_size) * 2 // ffn_gate, ffn_up
            + q4k(cfg.hidden_size, cfg.intermediate_size) // ffn_down
            + f32v(cfg.hidden_size) * 2 // attn_norm, post_attention_norm
            + f32v(cfg.head_dim) * 2 // attn_q_norm, attn_k_norm
            + f32v(48) * 2 // ssm_a, ssm_dt.bias
            + f32v(cfg.ssm_d_state) // ssm_norm
    };
    let (free, total) = grim_backend_rocm::vram_info(1);
    // Headroom for the decode-graph KV/SSM arenas and activation scratch.
    let fits = (((free as f64 * 0.80) as u64) / per_layer_bytes.max(1)) as usize;
    let n_layers = fits.min(65);
    eprintln!(
        "[repro-scale] depth {n_layers}/65 layers: per-layer {:.1} MB, free {:.1} of \
         {:.1} GB on ordinal 1{}",
        per_layer_bytes as f64 / 1e6,
        free as f64 / 1e9,
        total as f64 / 1e9,
        if n_layers < 65 {
            " -- CLAMPED, full depth exceeds this card's VRAM"
        } else {
            ""
        }
    );
    assert!(n_layers > 0, "ordinal 1 cannot hold even one layer");
    cfg.num_layers = n_layers;

    // Every tensor both layer types need, at the real widths.
    let mut tensors: HashMap<String, Rep> = HashMap::new();
    {
        let mat = |rows: usize, cols: usize| rep(vec![0.1f32; rows * cols], vec![rows, cols]);
        let vec1 = |n: usize| rep(vec![0.1f32; n], vec![n]);
        tensors.insert("attn_norm.weight".into(), vec1(cfg.hidden_size));
        tensors.insert("post_attention_norm.weight".into(), vec1(cfg.hidden_size));
        tensors.insert("attn_q.weight".into(), mat(2 * q_dim, cfg.hidden_size));
        tensors.insert("attn_k.weight".into(), mat(kv_dim, cfg.hidden_size));
        tensors.insert("attn_v.weight".into(), mat(kv_dim, cfg.hidden_size));
        tensors.insert("attn_output.weight".into(), mat(cfg.hidden_size, q_dim));
        tensors.insert("attn_q_norm.weight".into(), vec1(cfg.head_dim));
        tensors.insert("attn_k_norm.weight".into(), vec1(cfg.head_dim));
        tensors.insert("ssm_alpha.weight".into(), mat(cfg.hidden_size, cfg.ssm_dt_rank));
        tensors.insert("ssm_beta.weight".into(), mat(cfg.hidden_size, cfg.ssm_dt_rank));
        tensors.insert("ssm_a".into(), vec1(cfg.ssm_dt_rank));
        tensors.insert("ssm_dt.bias".into(), vec1(cfg.ssm_dt_rank));
        tensors.insert("ssm_norm.weight".into(), vec1(cfg.ssm_d_state));
        tensors.insert("ssm_out.weight".into(), mat(value_dim, cfg.hidden_size));
        tensors.insert("ssm_conv1d.weight".into(), mat(cfg.ssm_d_conv, ssm_qkv_dim));
        tensors.insert("ffn_gate.weight".into(), mat(cfg.intermediate_size, cfg.hidden_size));
        tensors.insert("ffn_up.weight".into(), mat(cfg.intermediate_size, cfg.hidden_size));
        tensors.insert("ffn_down.weight".into(), mat(cfg.hidden_size, cfg.intermediate_size));
    }
    let provider = MapProvider { tensors };

    let mut caches: Vec<_> = (0..n_layers)
        .map(|_| grim_models_transformer::qwen35::Qwen35LayerCache::new(&cfg))
        .collect();
    let blocks: Vec<_> = (0..n_layers)
        .map(|i| build_layer(&provider, &cfg, i))
        .collect();

    let dev_t = grim_nn::modules::pick_device_for_storage_device(&grim_tensor::Device::Rocm(1));
    let x_storage = grim_tensor::CoreTensorOps::from_cpu(
        &dev_t,
        &vec![0.1; cfg.hidden_size],
        &grim_tensor::Shape::new(vec![1, cfg.hidden_size]),
        DType::F32,
    )
    .expect("upload x");
    let mut h = grim_tensor::Tensor::new(
        Arc::from(x_storage),
        grim_tensor::Shape::new(vec![1, cfg.hidden_size]),
        DType::F32,
        QuantProvenance::GrimNative,
        grim_tensor::Device::Rocm(1),
    );

    for (i, blk) in blocks.iter().enumerate() {
        eprintln!("[repro-scale] layer {i}");
        h = blk
            .forward(&h, &[i as u32], &mut caches[i])
            .unwrap_or_else(|e| panic!("layer {i} forward: {e}"));
    }
    let vals = h.to_vec_f32().expect("read output");
    eprintln!(
        "[repro-scale] OK: {} layers, {} elems, all_finite={}",
        n_layers,
        vals.len(),
        vals.iter().all(|v| v.is_finite())
    );
    assert!(vals.iter().all(|v| v.is_finite()));
}
