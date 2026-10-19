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

use grim_backend_rocm::{BackendStorage, RocmDevice};
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

fn synth_weights_layers(n: usize) -> SynthProvider {
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
fn synth_weights_hybrid(n: usize, interval: usize) -> SynthProvider {
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
        put(&mut m, &format!("{p}.ffn_down.weight"), vec![HIDDEN, INTER]);
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

fn synth_weights() -> SynthProvider {
    synth_weights_layers(1)
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
    // Both halves of this had to be made non-finite-aware, because a gate that
    // cannot fail on NaN is not a gate. When the 27B's graph replay returned
    // all-NaN logits, this function reported `worst = 0.0` and
    // `split_graph_replay_matches_eager` passed, which is how a NaN-producing
    // replay shipped behind a "2.4e-5" parity number.
    //
    // 1. `f32::max` returns the non-NaN operand, so a NaN in the reference
    //    used to drop out of `scale` entirely and shrink the denominator.
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
        // 2. `NaN > worst.0` is false, so any non-finite pair was skipped and
        //    the divergence it represented was never recorded. Treat it as the
        //    worst possible outcome instead of a pass.
        let d = if d.is_finite() { d } else { f32::INFINITY };
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

/// A6 GATE: a layer-split model must now CAPTURE and REPLAY, one graph per
/// device, with the boundary hidden state handed across by the ordered peer
/// copy (A4). This replaces the old refusal gate: the refusal existed because
/// a cross-device capture faults on replay (`p2p_graph_probe.rs` isolates
/// that); the per-segment capture removes the cross-device capture, not the
/// symptom.
///
/// Fixture: 4 layers at interval = 2 (see GATE FIXTURE NOTE in the plan) —
/// recurrent {0,2}, attention {1,3}, both segments hold both kinds. NO
/// seeding: both sides start from an empty arena exactly like the
/// single-device gate (seeding coverage is A5's split gate).
#[test]
fn qwen35_layer_split_model_replays_like_eager() {
    let Some((device, dev)) = gpu_device() else {
        return;
    };
    if grim_backend_rocm::peer_access::enumerate_devices().unwrap_or(0) < 2 {
        eprintln!("[SKIP] needs 2 ROCm devices");
        return;
    }
    let _lock = KV_FORMAT_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    unsafe { std::env::set_var("GRIM_KV_ARENA", "f16") };

    let mut cfg = tiny_cfg();
    cfg.num_layers = 4;
    cfg.full_attention_interval = 2;
    // The planner prefers ONE device when the model fits it, and this tiny model
    // fits anywhere — force the split so this gate tests the split path.
    unsafe { std::env::set_var("GRIM_FORCE_SPLIT", "1") };
    cfg.devices = vec![Device::Rocm(0), Device::Rocm(1)];
    let provider = synth_weights_hybrid(4, 2);
    let ws = WeightSource::root(&provider, device.clone());
    let model = Qwen35::load(device.clone(), &ws, cfg).expect("load split hybrid qwen35");
    unsafe { std::env::remove_var("GRIM_FORCE_SPLIT") };

    let ords: Vec<usize> = model
        .blocks
        .iter()
        .map(|b| match b.device {
            Device::Rocm(o) => o,
            _ => usize::MAX,
        })
        .collect();
    assert!(
        ords.iter().any(|&o| o != ords[0]),
        "model did not layer-split; this gate would be vacuous: {ords:?}"
    );
    // Both layer kinds must be present, or the gate is not testing a hybrid.
    let is_attn = |i: usize| (i + 1) % 2 == 0;
    assert!(
        (0..4).any(is_attn) && (0..4).any(|i| !is_attn(i)),
        "the gate model must contain BOTH attention and recurrent layers"
    );

    // Eager reference, one token per step, absolute positions.
    let mut session = model.new_session();
    let mut eager = Vec::new();
    for (step, tok) in TOKENS.iter().enumerate() {
        let ids = dev_tensor(dev.as_ref(), &device, &[*tok as f32]);
        let pos = dev_tensor(dev.as_ref(), &device, &[step as f32]);
        eager.push(
            CausalLm::forward(&model, session.as_mut(), &ids, &pos, &[] as &[AdapterHandle])
                .expect("eager forward")
                .to_vec_f32()
                .expect("readback"),
        );
    }

    let mut graph = model
        .get_or_create_decode_graph(16, 1)
        .expect("a layer-split model must now be capturable");
    assert_eq!(graph.segments.len(), 2, "one graph per device");
    assert_eq!(graph.segments[0].layers, 0..2);
    assert_eq!(graph.segments[1].layers, 2..4);
    // The head must land in the LAST segment, or its kernels are in no graph.
    let head_ord = match model.output.weight().device() {
        Device::Rocm(o) => *o,
        _ => usize::MAX,
    };
    assert_eq!(
        head_ord,
        graph.segments.last().unwrap().device,
        "the output head must live on the last segment's device"
    );

    let _ = model.forward_capture(&mut graph, TOKENS[0]);
    let _ = model.forward_capture(&mut graph, TOKENS[0]);
    graph.begin_capture().expect("begin capture");
    model
        .forward_capture(&mut graph, TOKENS[0])
        .expect("recorded capture");
    graph.end_capture().expect("end capture");
    assert!(graph.is_captured, "split capture must succeed");

    let mut got = Vec::new();
    for tok in TOKENS {
        model
            .forward_replay(&mut graph, tok)
            .expect("replay");
        got.push(graph.read_logits_f32().expect("readback"));
        graph.buffers.current_pos = graph.buffers.current_pos.wrapping_add(1);
    }
    for (step, (a, b)) in eager.iter().zip(got.iter()).enumerate() {
        let (e, at, abs) = worst_scaled(a, b);
        eprintln!("[split-graph-parity] step {step}: worst {e:.3e} at {at} (abs {abs:.3e})");
        assert!(
            e <= 1e-1,
            "split step {step}: replay diverges from eager: worst {e:.3e} at {at} \
             (eager {}, graph {})",
            a[at],
            b[at]
        );
    }
    unsafe {
        std::env::set_var("GRIM_KV_ARENA", "f16");
    }
}

/// A5b GATE: each segment's `pos` replica must LIVE on that segment's device and
/// must receive the replay's start position.
///
/// A6's split-parity gate catches a wrong VALUE; this catches the wrong DEVICE.
/// `pos_dev` is bumped INSIDE the graph (`launch_bump_i32_slots`), so on a split
/// placement a device-1 layer would mutate device-0 memory — peer access covers
/// the seed pairs only, not this path. The replica design puts one pos scalar on
/// each segment's device and writes the replay start to all of them before the
/// launches; this test asserts both halves of that contract directly, so a
/// replica that silently stayed on device 0 fails HERE rather than as a fault
/// (or a silent peer-read) on replay.
#[test]
fn pos_replicas_exist_per_segment_device_and_start_aligned() {
    let Some((device, _dev)) = gpu_device() else {
        return;
    };
    if grim_backend_rocm::peer_access::enumerate_devices().unwrap_or(0) < 2 {
        eprintln!("[SKIP] needs 2 ROCm devices");
        return;
    }
    let _lock = KV_FORMAT_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    unsafe { std::env::set_var("GRIM_KV_ARENA", "f16") };

    let mut cfg = tiny_cfg();
    cfg.num_layers = 4;
    cfg.full_attention_interval = 2;
    // The planner prefers ONE device when the model fits, and this tiny model
    // fits anywhere — force the split so replicas are actually built.
    unsafe { std::env::set_var("GRIM_FORCE_SPLIT", "1") };
    cfg.devices = vec![Device::Rocm(0), Device::Rocm(1)];
    let provider = synth_weights_hybrid(4, 2);
    let ws = WeightSource::root(&provider, device.clone());
    let model = Qwen35::load(device.clone(), &ws, cfg).expect("load split hybrid qwen35");
    unsafe { std::env::remove_var("GRIM_FORCE_SPLIT") };

    let mut graph = model
        .get_or_create_decode_graph(16, 1)
        .expect("a layer-split model must be capturable");
    assert_eq!(graph.segments.len(), 2, "one graph per device");

    // 1. Every segment's replica must live on that segment's device. A replica
    //    still on device 0 while segment 1's graph reads it is the A5b defect.
    for (seg, s) in graph.segments.iter().enumerate() {
        let p = graph.buffers.pos_dev_for(s.device);
        assert_eq!(
            p.device_ordinal(),
            s.device,
            "pos replica {seg} must live on its segment's device {}",
            s.device
        );
    }

    // 2. Writing the replay start position must land on EVERY segment's replica,
    //    each on its own device's stream.
    graph
        .buffers
        .write_pos_all_segments(7)
        .expect("write pos to all segments");
    for (seg, s) in graph.segments.iter().enumerate() {
        RocmDevice::shared(s.device).synchronize();
        let p = graph.buffers.pos_dev_for(s.device);
        assert_eq!(
            p.to_cpu_vec_u32().expect("read pos")[0],
            7,
            "replica {seg} (device {}) did not receive the replay start position",
            s.device
        );
    }

    unsafe { std::env::set_var("GRIM_KV_ARENA", "f16") };
}

/// A0 BASELINE (external review 2026-10-16, item 1). The captured pos scalar is
/// bumped by EVERY attention layer (decode_graph.rs:1849, per-layer, no reset),
/// so a step with TWO OR MORE attention layers makes layer 2 read pos + steps.
/// Every existing graph gate here uses num_layers = 1 / interval = 1 — exactly
/// one attention layer per step — so this has never been observed. Single
/// device, f32 arena, hybrid 4-layer model, NO seed: pure graph-vs-eager.
/// If this goes red, the single-device graph has latent pos drift and A5b's
/// per-device replica design inherits the same defect.
#[test]
fn qwen35_graph_matches_eager_with_two_attention_layers_per_step() {
    let Some((device, dev)) = gpu_device() else {
        return;
    };
    let _env = KV_FORMAT_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    unsafe { std::env::set_var("GRIM_KV_ARENA", "f32") };

    let mut cfg = tiny_cfg();
    cfg.num_layers = 4;
    cfg.full_attention_interval = 2; // attention layers 1 and 3: two bumps per step
    let provider = synth_weights_hybrid(4, 2);
    let ws = WeightSource::root(&provider, device.clone());
    let model = Qwen35::load(device.clone(), &ws, cfg).expect("load 4-layer hybrid");

    let is_attn = |i: usize| (i + 1) % 2 == 0;
    let n_attn: usize = (0..4).filter(|&i| is_attn(i)).count();
    assert_eq!(n_attn, 2, "gate model must hold exactly two attention layers");

    // Eager reference.
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

    // Graph side: warmup, capture, replay — no seed, both sides from empty arena.
    let mut graph = model
        .get_or_create_decode_graph(16, 1)
        .expect("graph pool alloc");
    let _ = model.forward_capture(&mut graph, TOKENS[0]);
    let _ = model.forward_capture(&mut graph, TOKENS[0]);
    graph.begin_capture().expect("begin capture");
    model
        .forward_capture(&mut graph, TOKENS[0])
        .expect("recorded capture");
    graph.end_capture().expect("end capture");
    assert!(graph.is_captured);

    let mut graph_logits = Vec::new();
    for tok in TOKENS {
        model
            .forward_replay(&mut graph, tok)
            .expect("graph replay");
        graph_logits.push(graph.read_logits_f32().expect("graph logit readback"));
        graph.buffers.current_pos = graph.buffers.current_pos.wrapping_add(1);
    }

    for (step, (a, b)) in eager_logits.iter().zip(graph_logits.iter()).enumerate() {
        let (e, at, abs) = worst_scaled(a, b);
        eprintln!(
            "[a0-two-attn-parity] step {step}: worst {e:.3e} at {at} (abs {abs:.3e})"
        );
        assert!(
            e <= 1e-1,
            "A0 step {step}: graph replay diverges from eager with two attention \
             layers per step: worst {e:.3e} at {at} (eager {}, graph {})",
            a[at],
            b[at]
        );
    }
    unsafe { std::env::set_var("GRIM_KV_ARENA", "f32") };
}

/// A5 gate: a split model's graph buffers must be seeded per segment — each
/// family (KV, GDL, conv, KDA) copies on the LAYER's device context. Needs
/// two live devices; the seeder code paths themselves are covered per-layer
/// by construction (layer_devices[i] == the eager layer's device).
#[test]
fn qwen35_split_model_seed_families_run_on_their_own_devices() {
    let Some((device, dev)) = gpu_device() else {
        return;
    };
    if grim_backend_rocm::peer_access::enumerate_devices().unwrap_or(0) < 2 {
        eprintln!("[SKIP] needs 2 ROCm devices");
        return;
    }
    let _lock = KV_FORMAT_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    unsafe { std::env::set_var("GRIM_KV_ARENA", "f32") };

    let mut cfg = tiny_cfg();
    cfg.num_layers = 4;
    cfg.full_attention_interval = 2;
    // The planner prefers ONE device when the model fits it, and this tiny model
    // fits anywhere — force the split so this gate tests the split path.
    unsafe { std::env::set_var("GRIM_FORCE_SPLIT", "1") };
    cfg.devices = vec![Device::Rocm(0), Device::Rocm(1)];
    let provider = synth_weights_hybrid(4, 2);
    let ws = WeightSource::root(&provider, device.clone());
    let model = Qwen35::load(device.clone(), &ws, cfg).expect("load split hybrid");
    unsafe { std::env::remove_var("GRIM_FORCE_SPLIT") };

    // Eager prefill to produce seed sources.
    let mut session = model.new_session();
    let ids = dev_tensor(dev.as_ref(), &device, &[TOKENS[0] as f32]);
    let pos = dev_tensor(dev.as_ref(), &device, &[0.0f32]);
    let _ = CausalLm::forward(&model, session.as_mut(), &ids, &pos, &[] as &[AdapterHandle])
        .expect("eager prefill");

    let srcs = model.eager_kv_seed_sources(session.as_ref(), 1).expect("kv seeds");
    let conv = model.eager_conv_device_seed_sources(session.as_ref()).expect("conv seeds");
    let kda = model.eager_kda_seed_sources(session.as_ref()).expect("kda seeds");

    // The production seed path (run.rs macros) must succeed against a split
    // placement: every family copies on the right device context now.
    let mut graph = model.get_or_create_decode_graph(16, 1).expect("graph");
    let seed_dev = grim_backend_rocm::RocmDevice::shared(0);
    graph
        .buffers
        .seed_kv_arena_from_eager(&seed_dev, &srcs)
        .expect("seed kv (split placement)");
    graph
        .buffers
        .seed_gdl_state_from_eager(&seed_dev, &srcs)
        .expect("seed gdl (split placement)");
    graph
        .buffers
        .seed_conv_rings_from_device(&seed_dev, &conv)
        .expect("seed conv (split placement)");
    graph
        .buffers
        .seed_kda_state_from_eager(&seed_dev, &kda)
        .expect("seed kda (split placement)");
}

/// D1 GATE: the decode graph must SERVE `GRIM_KV_EVICT_MODE=curdkv` instead of
/// refusing. Today the refusal fires because the host compaction pass rewrites
/// `current_pos`; the device-side replacement keeps `total_dev` on-device.
/// Fixture: 4-layer hybrid (interval 2), f32 arena, cap 8 rows (sinks 2,
/// residual 2), 12 tokens so the run crosses the eviction boundary twice.
/// Both arms run under the SAME env; the eager arm exercises the host
/// compaction, the graph arm the device-side one — replay must track eager.
#[test]
fn qwen35_graph_serves_curdvk_eviction_like_eager() {
    let Some((device, dev)) = gpu_device() else {
        return;
    };
    let _lock = KV_FORMAT_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    unsafe {
        std::env::set_var("GRIM_KV_ARENA", "f32");
        std::env::set_var("GRIM_KV_ARENA_ROWS", "8");
        std::env::set_var("GRIM_KV_SINKS", "2");
        std::env::set_var("GRIM_KV_RESIDUAL", "2");
        std::env::set_var("GRIM_KV_EVICT_MODE", "curdkv");
    }

    let mut cfg = tiny_cfg();
    cfg.num_layers = 4;
    cfg.full_attention_interval = 2;
    let provider = synth_weights_hybrid(4, 2);
    let ws = WeightSource::root(&provider, device.clone());
    let model = Qwen35::load(device.clone(), &ws, cfg).expect("load curdkv model");

    // 12 tokens: cap 8 with sinks 2 / residual 2 means the boundary is crossed
    // at pos 9 (compaction) and again later — the gate needs both arms to
    // survive at least two compactions.
    let tokens: Vec<u32> = (0..12).map(|i| (3 + 7 * i) % VOCAB as u32).collect();

    // Eager arm: host compaction at the boundary.
    let mut session = model.new_session();
    let mut eager = Vec::new();
    for (step, tok) in tokens.iter().enumerate() {
        let ids = dev_tensor(dev.as_ref(), &device, &[*tok as f32]);
        let pos = dev_tensor(dev.as_ref(), &device, &[step as f32]);
        eager.push(
            CausalLm::forward(&model, session.as_mut(), &ids, &pos, &[] as &[AdapterHandle])
                .expect("eager curdkv forward")
                .to_vec_f32()
                .expect("eager readback"),
        );
    }

    // Graph arm: warmup 2, capture 1, replay the rest. The FIRST capture must
    // succeed — that is the red part of this gate while the refusal stands.
    // Env hygiene: this gate OWNS the curdkv env, and a panic must not leak it
    // into the rest of the test process (it poisoned every later gate once).
    let graph_res = model.get_or_create_decode_graph(16, 1);
    let mut graph = match graph_res {
        Ok(g) => g,
        Err(e) => {
            unsafe {
                std::env::remove_var("GRIM_KV_ARENA_ROWS");
                std::env::remove_var("GRIM_KV_SINKS");
                std::env::remove_var("GRIM_KV_RESIDUAL");
                std::env::remove_var("GRIM_KV_EVICT_MODE");
            }
            panic!("D1: the graph must not refuse CurDKV eviction anymore: {e}");
        }
    };
    let _ = model.forward_capture(&mut graph, tokens[0]);
    let _ = model.forward_capture(&mut graph, tokens[1]);
    graph.begin_capture().expect("begin capture");
    model
        .forward_capture(&mut graph, tokens[2])
        .expect("recorded capture");
    graph.end_capture().expect("end capture");
    assert!(graph.is_captured, "curdkv capture must succeed");

    let mut got = Vec::new();
    for (step, tok) in tokens.iter().enumerate().skip(2) {
        model
            .forward_replay(&mut graph, *tok)
            .unwrap_or_else(|e| panic!("D1 replay step {step}: {e}"));
        got.push(graph.read_logits_f32().expect("graph logit readback"));
        graph.buffers.current_pos = graph.buffers.current_pos.wrapping_add(1);
    }

    for (gi, (a, b)) in eager.iter().skip(2).zip(got.iter()).enumerate() {
        let step = gi + 2;
        let (e, at, abs) = worst_scaled(a, b);
        eprintln!("[d1-curdvk-parity] step {step}: worst {e:.3e} at {at} (abs {abs:.3e})");
        assert!(
            e <= 1e-1,
            "D1 curdkv step {step}: graph diverges from eager host compaction: \\
             worst {e:.3e} at {at} (eager {}, graph {})",
            a[at],
            b[at]
        );
    }
    unsafe {
        std::env::set_var("GRIM_KV_ARENA", "f32");
        std::env::remove_var("GRIM_KV_ARENA_ROWS");
        std::env::remove_var("GRIM_KV_SINKS");
        std::env::remove_var("GRIM_KV_RESIDUAL");
        std::env::remove_var("GRIM_KV_EVICT_MODE");
    }
}

/// PM-2 gate: with `GRIM_EMIT_TASK_GRAPH=1`, capturing the 4-layer hybrid must
/// publish a task-graph IR whose structure mirrors the real walk — embedding
/// node, per-layer (RECURRENT-or-ATTENTION, FFN GEMM) pairs, final NORM, head
/// GEMM — with a derived capability set and a single sink. This is the M2
/// interpreter's input contract.
#[test]
fn qwen35_capture_emits_a_structurally_true_task_graph() {
    let Some((device, _dev)) = gpu_device() else {
        return;
    };
    let _lock = KV_FORMAT_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    unsafe {
        std::env::set_var("GRIM_KV_ARENA", "f32");
        std::env::set_var("GRIM_EMIT_TASK_GRAPH", "1");
    }

    let mut cfg = tiny_cfg();
    cfg.num_layers = 4;
    cfg.full_attention_interval = 2;
    let provider = synth_weights_hybrid(4, 2);
    let ws = WeightSource::root(&provider, device.clone());
    let model = Qwen35::load(device.clone(), &ws, cfg).expect("load emit model");

    let mut graph = model.get_or_create_decode_graph(16, 1).expect("graph");
    assert!(graph.buffers.emitted_task_graph.is_none(), "no IR before capture");

    let _ = model.forward_capture(&mut graph, TOKENS[0]);
    let _ = model.forward_capture(&mut graph, TOKENS[0]);
    graph.begin_capture().expect("begin capture");
    model
        .forward_capture(&mut graph, TOKENS[0])
        .expect("recorded capture");
    graph.end_capture().expect("end capture");

    let ir = graph
        .buffers
        .emitted_task_graph
        .as_ref()
        .expect("the capture must publish the emitted task graph");

    // Structure: 1 embed + 4 layers x (block + FFN GEMM) + 1 NORM + 1 head GEMM.
    assert_eq!(ir.nodes.len(), 1 + 4 * 2 + 2, "node count: {:#?}", {
        let ops: Vec<u32> = ir.nodes.iter().map(|n| n.opcode).collect();
        ops
    });
    use grim_backend_rocm::task_graph::op;
    let ops: Vec<u32> = ir.nodes.iter().map(|n| n.opcode).collect();
    assert_eq!(ops[0], op::ELEMENTWISE, "first node is the embedding gather");
    for (li, i) in (0..4).enumerate() {
        let block = ops[1 + li * 2];
        let want = if (i + 1) % 2 == 0 { op::ATTENTION } else { op::RECURRENT };
        assert_eq!(block, want, "layer {i} block op");
        assert_eq!(ops[2 + li * 2], op::GEMM, "layer {i} FFN op");
    }
    assert_eq!(ops[9], op::NORM, "final norm");
    assert_eq!(ops[10], op::GEMM, "head");

    // The chain is total: every node except the first depends on its
    // predecessor, so the single sink is the head.
    assert_eq!(ir.sinks, vec![(ir.nodes.len() - 1) as u32]);

    // Derived caps, single device.
    assert!(!ir.caps.needs_peer, "single-device gate model");
    assert!(ir.caps.opcodes.contains(&op::ATTENTION));
    assert!(ir.caps.opcodes.contains(&op::RECURRENT));
    assert!(ir.caps.opcodes.contains(&op::GEMM));
    assert_eq!(ir.caps.devices.len(), 1);
    // The argument table is the model's: every arg handle resolves.
    for n in &ir.nodes {
        for &a in &n.args[..n.n_args as usize] {
            assert!((a as usize) < ir.arg_table.len(), "dangling arg handle {a}");
        }
    }

    unsafe {
        std::env::remove_var("GRIM_EMIT_TASK_GRAPH");
        std::env::set_var("GRIM_KV_ARENA", "f32");
    }
}
