// The Qwen3.5 synthetic weight provider, shared by the capture gates and
// the M7.0 A/B.
//
// It lives in a subdirectory so cargo does not compile it as a test target
// of its own, and it is shared rather than copied because a COPY of a model
// builder is a model builder that can drift from the one the gates
// validate: a second builder that names one tensor differently loads a
// DIFFERENT model, and the gate then measures something other than what its
// name says. One builder, two consumers.
//
// `include!`d (not a module) so the two consumers share the same `use`
// items without a `pub(crate)` on every helper. The `use` items live HERE,
// and the consumers must NOT re-import them — a duplicate import is a
// compile error, which is how this file's first version failed.
//
// NOTE: this file is SPLICED into a consumer, so it cannot carry `//!`
// inner doc comments: they are only legal at the top of a file. The file's
// own header comment lives in the consumers.

use std::collections::HashMap;
use std::sync::Arc;

use grim_backend_rocm::RocmDevice;
use grim_models_transformer::qwen35::Qwen35Config;
use grim_tensor::provider::{RawTensor, TensorMeta, TensorProvider};
use grim_tensor::{ArithType, CoreTensorOps, DType, Device, Shape, Tensor};

/// Same geometry as `qwen35_attn_d2d_parity.rs` — proven through the eager
/// device path, the quantizers, and the graph kernels.
pub const HEADS: usize = 4;
pub const KV_HEADS: usize = 2;
pub const HEAD_DIM: usize = 64;
pub const HIDDEN: usize = 256;
pub const INTER: usize = 192;
pub const VOCAB: usize = 64;
pub const TOKENS: [u32; 3] = [3, 7, 11];

/// Deterministic xorshift32 f32 in [-0.05, 0.05); a run is reproducible
/// without an RNG dependency.
pub fn raw_f32(seed: u32, shape: Vec<usize>) -> RawTensor {
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
        provenance: grim_tensor::QuantProvenance::GrimNative,
    }
}

/// A deterministic IQ4_NL-PACKED tensor: the format's real packing, through
/// the quantizer, not a fake byte pattern.
///
/// The coverage table's QuantFormat axis reads the weight's STORAGE, so a
/// weight that was dequantized at load would report "native" and the row
/// would stay unexercised no matter how many models carried the format. The
/// row needs a weight that stays PACKED — which is also the state the ROCm
/// fast path keeps it in.
pub fn raw_iq4nl(seed: u32, shape: Vec<usize>) -> RawTensor {
    let n: usize = shape.iter().product();
    let mut data = Vec::with_capacity(n);
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    for _ in 0..n {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let v = ((state >> 8) as f32 / 8_388_608.0) - 1.0;
        data.push(v * 0.05);
    }
    let bytes = grim_quant::quant_iq4nl(&data).expect("quant_iq4nl");
    RawTensor {
        bytes,
        shape,
        dtype: DType {
            arith: ArithType::F32,
            storage: grim_tensor::Storage::KQuant(grim_tensor::KQuantScheme::IQ4NL),
        },
        provenance: grim_tensor::QuantProvenance::GrimNative,
    }
}

pub struct SynthProvider {
    pub tensors: HashMap<String, RawTensor>,
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

pub fn synth_weights_layers(n: usize) -> SynthProvider {
    synth_weights_hybrid(n, 1)
}

/// Hybrid synthesizer: with `interval = 1` every layer is full attention
/// (as before); with `interval = k`, layer i is full attention iff
/// (i+1) % k == 0, and the other layers are KDA/recurrent and get the
/// recurrent tensor set the loader requires (qwen35.rs:470-586):
/// `attn_qkv` [kda_qkv_dim, hidden], `attn_gate` [max(q_dim, value_dim), hidden],
/// `ssm_out` [hidden, value_dim], `ssm_conv1d` [conv_dim, d_conv], plus the
/// scalar/vector KDA gates. Dims derive from `Qwen35Config::default()`'s
/// ssm fields (d_state 128, d_conv 4, dt_rank 48, n_group 16), NOT literals.
pub fn synth_weights_hybrid(n: usize, interval: usize) -> SynthProvider {
    synth_weights_hybrid_with(n, interval, false)
}

/// The same model, with the FFN's down-projection weight in IQ4_NL.
///
/// The coverage table's IQ4_NL row needs a model that genuinely carries the
/// format (see [`raw_iq4nl`]): the axis reads the weight's storage, so a
/// weight that arrives dequantized would report "native" and the row would
/// stay unexercised.
pub fn synth_weights_hybrid_with(n: usize, interval: usize, iq4nl_down: bool) -> SynthProvider {
    let q_dim = HEADS * HEAD_DIM;
    let kv_dim = KV_HEADS * HEAD_DIM;
    let d_state = 128usize;
    let d_conv = 4usize;
    let dt_rank = 48usize;
    let n_group = 16usize;
    let kda_qkv_dim = (dt_rank + 2 * n_group) * d_state; // == conv_dim
    let value_dim = dt_rank * d_state;
    let mut m = HashMap::new();
    let mut seed = 1u32;
    let mut put = |m: &mut HashMap<String, RawTensor>, name: &str, shape: Vec<usize>| {
        seed = seed.wrapping_add(17);
        m.insert(name.to_string(), raw_f32(seed, shape));
    };
    put(&mut m, "token_embd.weight", vec![VOCAB, HIDDEN]);
    for i in 0..n {
        let p = format!("blk.{i}");
        let is_attn = (i + 1) % interval.max(1) == 0;
        put(&mut m, &format!("{p}.attn_norm.weight"), vec![HIDDEN]);
        put(&mut m, &format!("{p}.post_attention_norm.weight"), vec![HIDDEN]);
        put(&mut m, &format!("{p}.ffn_norm.weight"), vec![HIDDEN]);
        put(&mut m, &format!("{p}.ffn_gate.weight"), vec![INTER, HIDDEN]);
        put(&mut m, &format!("{p}.ffn_up.weight"), vec![INTER, HIDDEN]);
        if iq4nl_down {
            // The layer index is the seed: deterministic, and distinct per
            // layer (the shared `seed` counter is borrowed by `put`).
            m.insert(
                format!("{p}.ffn_down.weight"),
                raw_iq4nl(i as u32 + 7, vec![HIDDEN, INTER]),
            );
        } else {
            put(&mut m, &format!("{p}.ffn_down.weight"), vec![HIDDEN, INTER]);
        }
        if is_attn {
            put(&mut m, &format!("{p}.attn_q.weight"), vec![2 * q_dim, HIDDEN]);
            put(&mut m, &format!("{p}.attn_k.weight"), vec![kv_dim, HIDDEN]);
            put(&mut m, &format!("{p}.attn_v.weight"), vec![kv_dim, HIDDEN]);
            put(&mut m, &format!("{p}.attn_output.weight"), vec![HIDDEN, q_dim]);
        } else {
            put(&mut m, &format!("{p}.attn_qkv.weight"), vec![kda_qkv_dim, HIDDEN]);
            put(&mut m, &format!("{p}.attn_gate.weight"), vec![q_dim.max(value_dim), HIDDEN]);
            put(&mut m, &format!("{p}.ssm_out.weight"), vec![HIDDEN, value_dim]);
            put(&mut m, &format!("{p}.ssm_conv1d.weight"), vec![kda_qkv_dim, d_conv]);
            put(&mut m, &format!("{p}.ssm_a"), vec![value_dim]);
            put(&mut m, &format!("{p}.ssm_alpha.weight"), vec![dt_rank, HIDDEN]);
            put(&mut m, &format!("{p}.ssm_beta.weight"), vec![dt_rank, HIDDEN]);
            put(&mut m, &format!("{p}.ssm_dt.bias"), vec![dt_rank]);
            put(&mut m, &format!("{p}.ssm_norm.weight"), vec![value_dim]);
        }
    }
    put(&mut m, "output_norm.weight", vec![HIDDEN]);
    put(&mut m, "output.weight", vec![VOCAB, HIDDEN]);
    SynthProvider { tensors: m }
}

pub fn synth_weights() -> SynthProvider {
    synth_weights_layers(1)
}

/// One full-attention layer (`interval = 1`): no KDA/recurrent machinery, so
/// a gate that wants the dense-KV graph path alone can ask for it.
pub fn tiny_cfg() -> Qwen35Config {
    let mut c = Qwen35Config::default();
    c.vocab_size = VOCAB;
    c.hidden_size = HIDDEN;
    c.num_heads = HEADS;
    c.num_kv_heads = KV_HEADS;
    c.head_dim = HEAD_DIM;
    c.num_layers = 1;
    c.intermediate_size = INTER;
    c.full_attention_interval = 1;
    c
}

/// The GPU probe. `RocmDevice::try_new(0)` under a `catch_unwind`, because a
/// device that cannot be created must SKIP the test, not panic it — a
/// machine without the card is not a machine with a failing gate.
pub fn gpu_device() -> Option<(Device, Arc<RocmDevice>)> {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run");
        return None;
    }
    std::panic::catch_unwind(|| {
        let dev = Arc::new(RocmDevice::try_new(0).expect("RocmDevice::try_new(0)"));
        (Device::Rocm(0), dev)
    })
    .ok()
}

#[allow(dead_code)]
pub fn dev_tensor(dev: &RocmDevice, device: &Device, data: &[f32]) -> Tensor {
    let shape = Shape::new(vec![data.len()]);
    let st = CoreTensorOps::from_cpu(dev, data, &shape, DType::F32).expect("from_cpu");
    Tensor::new(
        Arc::from(st),
        shape,
        DType::F32,
        Default::default(),
        device.clone(),
    )
}

/// Worst normalized elementwise difference (mirrors the d2d gates' metric,
/// which does not explode on near-zero outputs).
///
/// Both halves of this had to be made non-finite-aware, because a gate that
/// cannot fail on NaN is not a gate. When the 27B's graph replay returned
/// all-NaN logits, this function reported `worst = 0.0` and
/// `split_graph_replay_matches_eager` passed, which is how a NaN-producing
/// replay shipped behind a "2.4e-5" parity number.
///
/// 1. `f32::max` returns the non-NaN operand, so a NaN in the reference
///    used to drop out of `scale` entirely and shrink the denominator.
/// 2. `NaN > worst.0` is false, so any non-finite pair was skipped and the
///    divergence it represented was never recorded. Treat it as the worst
///    possible outcome instead of a pass.
pub fn worst_scaled(a: &[f32], b: &[f32]) -> (f32, usize, f32) {
    let scale = a
        .iter()
        .fold(0.0f32, |m, x| {
            if x.is_finite() {
                m.max(x.abs())
            } else {
                m
            }
        })
        .max(1e-6);
    let mut worst = (0.0f32, 0usize, 0.0f32);
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs() / scale;
        let d = if d.is_finite() { d } else { f32::INFINITY };
        if d > worst.0 {
            worst = (d, i, (x - y).abs());
        }
    }
    worst
}
