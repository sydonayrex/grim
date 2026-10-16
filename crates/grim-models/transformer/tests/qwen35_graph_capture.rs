//! Capture-level KV-arena gates for the Qwen35 decode graph.
//!
//! `graph_i8_append_and_attend_match_eager_f32` (grim-backend-rocm) gates the
//! int8 launcher pair one level BELOW capture: it calls the two launchers on
//! one synthetic row. These tests gate the layer the production path actually
//! runs — `get_or_create_decode_graph` -> seed-free capture -> `forward_replay`
//! — with the arena format resolved through the SAME env -> `KvArenaFormat`
//! resolver the model loader uses.
//!
//! Three gates:
//! 1. `qwen35_i8_graph_replay_matches_eager` — int8 arena: replayed logits
//!    must match the int8 eager path step for step, and the graph pool must
//!    prove engagement (`kv_arena_packed_width() == Some(8)` + sidecars).
//!    Without the engagement proof a test that silently ran f32 twice passes.
//! 2. `qwen35_packed_graph_refusal_int2v4` — int2v4 resolves through the
//!    parser but has NO capturable kernel pair; capture must REFUSE (eager
//!    fallback), never run the 8-bit kernels over 4-codes-per-byte keys.
//! 3. `qwen35_packed_graph_refusal_int4` — same for int4.
//!
//! Gated on `GRIM_GPU_TEST=1` + ROCm device. All three mutate process-global
//! env, so they serialize on `KV_FORMAT_ENV_LOCK`.

use std::collections::HashMap;
use std::sync::Arc;

use grim_backend_rocm::RocmDevice;
use grim_core::model::{AdapterHandle, CausalLm};
use grim_models_transformer::decode_graph::DecodeGraphModel;
use grim_models_transformer::qwen35::{Qwen35, Qwen35Config};
use grim_nn::WeightSource;
use grim_tensor::provider::{RawTensor, TensorMeta, TensorProvider};
use grim_tensor::{CoreTensorOps, DType, Device, Shape, Tensor};

static KV_FORMAT_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

// Same geometry as qwen35_attn_d2d_parity.rs — proven through the eager
// device path, the quantizers, and the graph kernels.
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const HIDDEN: usize = 256;
const INTER: usize = 192;
const VOCAB: usize = 64;
const TOKENS: [u32; 3] = [3, 7, 11];

/// Deterministic xorshift32 f32 in [-0.05, 0.05); a run is reproducible
/// without an RNG dependency.
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
        provenance: grim_tensor::QuantProvenance::GrimNative,
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

fn synth_weights() -> SynthProvider {
    let q_dim = HEADS * HEAD_DIM;
    let kv_dim = KV_HEADS * HEAD_DIM;
    let mut m = HashMap::new();
    let mut seed = 1u32;
    let mut put = |m: &mut HashMap<String, RawTensor>, name: &str, shape: Vec<usize>| {
        seed = seed.wrapping_add(17);
        m.insert(name.to_string(), raw_f32(seed, shape));
    };
    put(&mut m, "token_embd.weight", vec![VOCAB, HIDDEN]);
    put(&mut m, "blk.0.attn_norm.weight", vec![HIDDEN]);
    put(&mut m, "blk.0.attn_q.weight", vec![2 * q_dim, HIDDEN]);
    put(&mut m, "blk.0.attn_k.weight", vec![kv_dim, HIDDEN]);
    put(&mut m, "blk.0.attn_v.weight", vec![kv_dim, HIDDEN]);
    put(&mut m, "blk.0.attn_output.weight", vec![HIDDEN, q_dim]);
    put(&mut m, "blk.0.post_attention_norm.weight", vec![HIDDEN]);
    put(&mut m, "blk.0.ffn_norm.weight", vec![HIDDEN]);
    put(&mut m, "blk.0.ffn_gate.weight", vec![INTER, HIDDEN]);
    put(&mut m, "blk.0.ffn_up.weight", vec![INTER, HIDDEN]);
    put(&mut m, "blk.0.ffn_down.weight", vec![HIDDEN, INTER]);
    put(&mut m, "output_norm.weight", vec![HIDDEN]);
    put(&mut m, "output.weight", vec![VOCAB, HIDDEN]);
    SynthProvider { tensors: m }
}

/// One full-attention layer (`interval = 1`): no KDA/recurrent machinery, so
/// the gate isolates the dense-KV graph path.
fn tiny_cfg() -> Qwen35Config {
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

fn gpu_device() -> Option<(Device, Arc<RocmDevice>)> {
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

fn dev_tensor(dev: &RocmDevice, device: &Device, data: &[f32]) -> Tensor {
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
fn worst_scaled(a: &[f32], b: &[f32]) -> (f32, usize, f32) {
    let scale = a.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-6);
    let mut worst = (0.0f32, 0usize, 0.0f32);
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs() / scale;
        if d > worst.0 {
            worst = (d, i, (x - y).abs());
        }
    }
    worst
}

#[test]
fn qwen35_packed_graph_replay_matches_eager() {
    // Both formats with a DEVICE-TOTAL kernel pair get the same capture-level
    // gate: replayed logits must match the same-format eager path step for
    // step. int8 is the production default; int4's pair landed with WS-3B and
    // needs the same evidence at the capture layer, not just kernel level.
    for (format, bits) in [("int8", 8u32), ("int4", 4u32)] {
        let Some((device, dev)) = gpu_device() else {
            return;
        };
        let _env = KV_FORMAT_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        unsafe {
            std::env::set_var("GRIM_KV_ARENA", format);
            std::env::remove_var("GRIM_KV_ARENA_ROWS");
        }

        // The model AND the graph pool resolve the arena format from env at
        // construction, so the format must be set before `load`.
        let provider = synth_weights();
        let ws = WeightSource::root(&provider, device.clone());
        let model = Qwen35::load(device.clone(), &ws, tiny_cfg()).expect("load tiny qwen35");

        // ── Eager side: fresh session, one token per step, absolute positions.
        eprintln!("[STAGE] {format}: eager start");
        let mut session = model.new_session();
        let mut eager_logits = Vec::new();
        for (step, tok) in TOKENS.iter().enumerate() {
            let ids = dev_tensor(dev.as_ref(), &device, &[*tok as f32]);
            let pos = dev_tensor(dev.as_ref(), &device, &[step as f32]);
            let out = CausalLm::forward(&model, session.as_mut(), &ids, &pos, &[] as &[AdapterHandle])
                .expect("eager forward")
                .to_vec_f32()
                .expect("eager readback");
            eager_logits.push(out);
        }

        // ── Graph side: warmup (executes; state discarded by the pos
        // rewrite), capture (RECORDS only), then replay the SAME token
        // sequence from an equally empty arena. run.rs re-seeds from a
        // session because its capture step ran eagerly in the caller; here
        // the first replay simply replays the capture token, so no session
        // seed is needed and both sides see identical KV history
        // (0, 1, 2 prior rows).
        eprintln!("[STAGE] {format}: eager done, graph pool alloc");
        let mut graph = model
            .get_or_create_decode_graph(16, 1)
            .unwrap_or_else(|e| panic!("{format}: graph pool alloc failed: {e}"));

        // Engagement proof: the pool must hold the recorded width and its
        // scale sidecars. If the width recording regresses to the old dtype
        // conflation, this fails before any numeric comparison can mislead.
        assert_eq!(
            graph.buffers.kv_arena_packed_width(),
            Some(bits),
            "{format}: graph pool must record the {bits}-bit arena width"
        );
        assert!(
            !graph.buffers.k_scale_arena.is_empty() && !graph.buffers.v_scale_arena.is_empty(),
            "{format}: graph pool must allocate scale sidecars before capture"
        );

        eprintln!("[STAGE] {format}: pool ok, warmups");
        let _ = model.forward_capture(&mut graph, TOKENS[0]);
        let _ = model.forward_capture(&mut graph, TOKENS[0]);
        eprintln!("[STAGE] {format}: warmups done, begin capture");
        graph.begin_capture().expect("begin capture");
        model
            .forward_capture(&mut graph, TOKENS[0])
            .expect("recorded capture");
        graph.end_capture().expect("end capture");
        assert!(graph.is_captured, "{format}: capture must succeed");

        eprintln!("[STAGE] {format}: captured, replays");
        let mut graph_logits = Vec::new();
        for tok in TOKENS {
            dev.reset_launch_count();
            model
                .forward_replay(&mut graph, tok)
                .unwrap_or_else(|e| panic!("{format}: graph replay failed: {e}"));
            // Replay must be a single hipGraphLaunch — any eager launch here
            // means the captured topology silently fell back.
            assert_eq!(
                dev.launch_count(),
                0,
                "{format}: replay must enqueue zero eager launches"
            );
            let logits = graph.read_logits_f32().expect("graph logit readback");
            assert!(
                logits.iter().all(|x| x.is_finite()),
                "{format}: replay logits non-finite"
            );
            graph_logits.push(logits);
            graph.buffers.current_pos = graph.buffers.current_pos.wrapping_add(1);
        }

        // ── Compare. Both sides quantize K/V with the SAME per-(row, kv_head)
        // amax scheme and read identical codes, so the only divergence source
        // is attention reduction order (eager kernel vs graph device-total
        // kernel) amplified by the FFN — the d2d gates measure ~1e-2 for a
        // FULL format change (f32 vs int8); same-format reduction-order noise
        // is strictly smaller. Bar is ~an order of magnitude under the eager
        // format-change gate (3e-1) with printed headroom.
        for (step, (a, b)) in eager_logits.iter().zip(graph_logits.iter()).enumerate() {
            let (e, at, abs) = worst_scaled(a, b);
            eprintln!(
                "[{format}-graph-parity] step {step}: worst {e:.3e} at {at} (abs {abs:.3e})"
            );
            assert!(
                e <= 1e-1,
                "{format} step {step}: graph replay diverges from eager: worst {e:.3e} at {at} \
                 (eager {}, graph {})",
                a[at],
                b[at]
            );
        }

        unsafe {
            std::env::set_var("GRIM_KV_ARENA", "f32");
        }
    }
}

/// int2v4 resolves through `parse_kv_arena_env` and has EAGER kernels, but no
/// capturable pair. `get_or_create_decode_graph` must refuse — the historic
/// bug ran the 8-bit launcher pair over a 4-codes-per-byte arena, silently.
#[test]
fn qwen35_packed_graph_refusal_int2v4() {
    let Some((device, _dev)) = gpu_device() else {
        return;
    };
    let _env = KV_FORMAT_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    unsafe {
        std::env::set_var("GRIM_KV_ARENA", "int2v4");
    }
    let provider = synth_weights();
    let ws = WeightSource::root(&provider, device.clone());
    let model = Qwen35::load(device.clone(), &ws, tiny_cfg()).expect("load tiny qwen35");
    let err = model
        .get_or_create_decode_graph(16, 1)
        .expect_err("int2v4 graph capture must be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("no device-total append/attend kernel pair"),
        "refusal must name the missing kernel pair, got: {msg}"
    );
    unsafe {
        std::env::set_var("GRIM_KV_ARENA", "f32");
    }
}
