//! PM-3 / condition 1a: the Llama emitter proven on the REAL
//! `Llama-3.2-1B-Instruct-Q4_1.gguf` checkpoint.
//!
//! Q4_1 had no decoder in grim until this task — it loaded as `Unsupported`
//! ("has no decoder in grim"). The checkpoint now dequantizes at load through
//! `grim_quant::dequant_q4_1` (llama.cpp block layout: `[f16 d][f16 m]`
//! `[16 B nibbles]`, `w = q*d + m`), so this file exercises BOTH halves of
//! condition 1's proof vehicle: a real Q4_1 producer artifact AND the Llama
//! task-graph emitter.
//!
//! Three gates:
//! 1. The real checkpoint LOADS through the new Q4_1 decoder (real producer
//!    bytes, not a synthetic provider).
//! 2. With `GRIM_EMIT_TASK_GRAPH=1`, capture publishes a task-graph IR whose
//!    structure mirrors Llama's walk: embedding ELEMENTWISE, then per layer an
//!    ATTENTION node + an FFN node, then the final NORM and the head GEMM.
//!    Llama-3.2-1B is 16 all-attention layers with no MoE, so the count is
//!    `1 + 16*2 + 2 = 35` (the gate asserts the expression, not a literal, so
//!    a node added to the emitter moves the count rather than a constant).
//! 3. Graph replay tracks the same model's eager path per token, at the
//!    real-model worst-scaled bar.
//!
//! Run (single process, env-mutating; `--test-threads=1`; GPU 1 is the
//! measurement card — `Device::Rocm(0)` under `HIP_VISIBLE_DEVICES=1`):
//! `HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 \
//!  cargo test -p grim-cli --test llama_q41_task_graph -- --nocapture --test-threads=1`

use grim_models_transformer::Llama;
use grim_models_transformer::decode_graph::DecodeGraphModel;
use grim_tensor::{CoreTensorOps, Device, Shape, Tensor};

/// The named condition-1 proof vehicle, relative to the workspace root.
const MODEL: &str = "models/Llama-3.2-1B-Instruct-Q4_1.gguf";
/// Llama-3.2-1B is 16 all-attention layers, no MoE.
const LAYERS: usize = 16;
/// In-vocab BPE ids; the values only need to be < vocab_size.
const TOKENS: [u32; 4] = [128000, 11, 322, 262];
/// Graph context is tiny: this gate replays four tokens.
const GRAPH_CTX: usize = 64;

fn set_var(k: &str, v: &str) {
    // SAFETY: the test mutates process-global env and must run
    // `--test-threads=1`; it is the only test in this binary.
    unsafe { std::env::set_var(k, v) }
}

fn workspace_path(rel: &str) -> std::path::PathBuf {
    let p = std::path::Path::new(rel);
    if p.is_absolute() {
        return p.to_path_buf();
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    root.join(rel)
}

fn tensor_on(dev: &Device, data: Vec<f32>, shape: Shape) -> Tensor {
    let storage = match dev {
        Device::Rocm(o) => grim_backend_rocm::RocmDevice::shared(*o)
            .from_cpu(&data, &shape, grim_tensor::DType::F32)
            .expect("from_cpu"),
        other => panic!("unsupported device {other:?}"),
    };
    Tensor::new(
        std::sync::Arc::from(storage),
        shape,
        grim_tensor::DType::F32,
        grim_tensor::QuantProvenance::GrimNative,
        dev.clone(),
    )
}

/// Worst RELATIVE difference between `a` and `b` (against `a`'s own scale), plus
/// that scale and the index where it happened.
///
/// A raw abs-diff bar is meaningless across checkpoints: the same
/// reduction-order noise is 1e-3 on a sharp distribution and 1e-1 on a flat
/// one. Both halves are non-finite-aware, because a gate that cannot fail on
/// NaN is not a gate (`qwen35_graph_capture.rs:196` had to be fixed for
/// exactly that, after an all-NaN replay reported `worst = 0.0`):
/// `f32::max` returns the non-NaN operand, so a NaN reference used to drop out
/// of the scale entirely, and `NaN > worst` is false, so a non-finite pair was
/// never recorded. A non-finite pair is the worst possible outcome.
fn worst_scaled(a: &[f32], b: &[f32]) -> (f32, usize, f32) {
    let scale = a
        .iter()
        .fold(
            0.0f32,
            |m, x| if x.is_finite() { m.max(x.abs()) } else { m },
        )
        .max(1e-6);
    let mut worst = (0.0f32, 0usize, 0.0f32);
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let abs = (x - y).abs();
        let d = abs / scale;
        let d = if d.is_finite() { d } else { f32::INFINITY };
        if d > worst.0 {
            worst = (d, i, abs);
        }
    }
    worst
}

#[test]
fn llama_q41_loads_emits_and_replays_like_eager() {
    if std::env::var("GRIM_GPU_TEST")
        .map(|v| v != "1")
        .unwrap_or(true)
    {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run the Llama Q4_1 task-graph gate");
        return;
    }
    let model_path = workspace_path(MODEL);
    if !model_path.exists() {
        eprintln!("[SKIP] {} not present", model_path.display());
        return;
    }
    let model_path = model_path.to_string_lossy().into_owned();
    let device = Device::Rocm(0);
    // The arena format is resolved from env at model AND pool construction, so
    // both sides must see the same spelling before `load`.
    set_var("GRIM_KV_ARENA", "f16");
    set_var("GRIM_EMIT_TASK_GRAPH", "1");

    // ── Gate 1: the real Q4_1 checkpoint loads.
    let lm = grim_engine::model_loader::load_model_from_gguf(&model_path, device.clone())
        .unwrap_or_else(|e| panic!("load {MODEL} (Q4_1 decoder): {e}"));
    // The checkpoint has no `output.weight` (tied LM head) and `output_norm`
    // without `output.weight` is the loader's SmolLM2 signature, so resolve the
    // graph model through the same Llama-family route `run.rs` uses rather than
    // assuming the concrete type.
    let graph_model: &dyn DecodeGraphModel = match lm.as_any().downcast_ref::<Llama>() {
        Some(l) => l,
        None => grim_models_transformer::llama_wrapper_graph_model(lm.as_any())
            .expect("the Llama-family checkpoint must expose a decode-graph model"),
    };

    // ── Eager reference: fresh session, one token per step, absolute positions.
    let mut session = lm.new_session();
    let mut eager = Vec::new();
    for (step, tok) in TOKENS.iter().enumerate() {
        let ids = tensor_on(&device, vec![*tok as f32], Shape::new(vec![1, 1]));
        let pos = tensor_on(&device, vec![step as f32], Shape::new(vec![1, 1]));
        eager.push(
            lm.forward(&mut *session, &ids, &pos, &[])
                .expect("eager forward")
                .to_vec_f32()
                .expect("eager logits d2h"),
        );
    }

    // ── Graph side: warmups (execute; their arena rows are reset below),
    // capture (records only), then replay the same token sequence from an
    // equally empty arena.
    let mut graph = graph_model
        .get_or_create_decode_graph(GRAPH_CTX, 1)
        .expect("graph pool alloc");
    assert!(
        graph.buffers.emitted_task_graph.is_none(),
        "no IR before capture"
    );
    for _ in 0..2 {
        let _ = graph_model.forward_capture(&mut graph, TOKENS[0]);
    }
    graph.begin_capture().expect("begin capture");
    graph_model
        .forward_capture(&mut graph, TOKENS[0])
        .expect("recorded capture");
    eprintln!("[marker] before end_capture");
    graph.end_capture().expect("end capture");
    eprintln!("[marker] after end_capture");
    assert!(graph.is_captured, "capture must succeed");

    // ── Gate 2: the published IR.
    let ir = graph
        .buffers
        .emitted_task_graph
        .as_ref()
        .expect("capture must publish the emitted task graph");
    use grim_backend_rocm::task_graph::op;
    let ops: Vec<u32> = ir.nodes.iter().map(|n| n.opcode).collect();
    assert_eq!(
        ir.nodes.len(),
        1 + LAYERS * 2 + 2,
        "node count (1 embed + {LAYERS} x (attn + ffn) + norm + head): {ops:?}"
    );
    assert_eq!(
        ops[0],
        op::ELEMENTWISE,
        "first node is the embedding gather"
    );
    for li in 0..LAYERS {
        assert_eq!(ops[1 + li * 2], op::ATTENTION, "layer {li} block op");
        assert_eq!(ops[2 + li * 2], op::GEMM, "layer {li} FFN op");
    }
    assert_eq!(ops[1 + LAYERS * 2], op::NORM, "final norm");
    assert_eq!(ops[2 + LAYERS * 2], op::GEMM, "head");
    // The chain is total, so the single sink is the head.
    assert_eq!(ir.sinks, vec![(ir.nodes.len() - 1) as u32]);
    // Derived caps, single device, no MoE, no peer traffic.
    assert!(!ir.caps.needs_peer, "single-device model");
    assert!(
        !ir.caps.opcodes.contains(&op::MOE_DISPATCH),
        "Llama-3.2-1B has no MoE layers"
    );
    assert!(ir.caps.opcodes.contains(&op::ATTENTION));
    assert_eq!(ir.caps.devices.len(), 1);
    // The argument table is the model's: every arg handle resolves.
    for n in &ir.nodes {
        for &a in &n.args[..n.n_args as usize] {
            assert!((a as usize) < ir.arg_table.len(), "dangling arg handle {a}");
        }
    }

    // The warmups executed against this arena and bumped each layer's live-row
    // total and the position, so reset the replay state to "empty arena" — the
    // same history the eager session started from. One API, because the three
    // fields it covers have to move together.
    graph
        .buffers
        .reset_for_replay()
        .expect("reset replay state");

    eprintln!(
        "[addr-scan] token_ids_dev = 0x{:x}",
        graph.buffers.token_ids_dev.device_ptr_u64().unwrap_or(0)
    );
    eprintln!(
        "[addr-scan] pos_dev = 0x{:x}",
        graph.buffers.pos_dev.device_ptr_u64().unwrap_or(0)
    );
    eprintln!(
        "[addr-scan] head_input = 0x{:x}",
        graph.buffers.head_input.device_ptr_u64().unwrap_or(0)
    );
    eprintln!(
        "[addr-scan] head_output = 0x{:x}",
        graph.buffers.head_output.device_ptr_u64().unwrap_or(0)
    );
    for (i, b) in graph.buffers.layer_input.iter().enumerate() {
        eprintln!(
            "[addr-scan] layer_input[{i}] = 0x{:x} len={}",
            b.device_ptr_u64().unwrap_or(0),
            b.bytes()
        );
    }
    for (i, b) in graph.buffers.norm_buf.iter().enumerate() {
        eprintln!(
            "[addr-scan] norm_buf[{i}] = 0x{:x} len={}",
            b.device_ptr_u64().unwrap_or(0),
            b.bytes()
        );
    }
    for (i, b) in graph.buffers.k_arena.iter().enumerate() {
        eprintln!(
            "[addr-scan] k_arena[{i}] = 0x{:x} len={}",
            b.device_ptr_u64().unwrap_or(0),
            b.bytes()
        );
    }
    for (i, b) in graph.buffers.v_arena.iter().enumerate() {
        eprintln!(
            "[addr-scan] v_arena[{i}] = 0x{:x} len={}",
            b.device_ptr_u64().unwrap_or(0),
            b.bytes()
        );
    }
    for (i, b) in graph.buffers.act_q81_buf.iter().enumerate() {
        eprintln!(
            "[addr-scan] act_q81_buf[{i}] = 0x{:x} len={}",
            b.device_ptr_u64().unwrap_or(0),
            b.bytes()
        );
    }
    let llama_ref: Option<&Llama> = lm.as_any().downcast_ref::<Llama>().or_else(|| {
        lm.as_any()
            .downcast_ref::<grim_models_transformer::smollm2::SmolLm2>()
            .map(|s| &s.inner)
    });
    if let Some(l) = llama_ref {
        let t_s = l.tok_embeddings.weight.storage();
        let rocm_s = t_s
            .as_any()
            .downcast_ref::<grim_backend_rocm::RocmStorage>();
        eprintln!(
            "[addr-scan] tok_embeddings.weight = 0x{:x} bytes={}",
            rocm_s.and_then(|s| s.device_ptr_u64()).unwrap_or(0),
            rocm_s.map(|s| s.bytes()).unwrap_or(0)
        );
        let o_s = l.output.weight().storage();
        let rocm_o = o_s
            .as_any()
            .downcast_ref::<grim_backend_rocm::RocmStorage>();
        eprintln!(
            "[addr-scan] output.weight = 0x{:x} bytes={}",
            rocm_o.and_then(|s| s.device_ptr_u64()).unwrap_or(0),
            rocm_o.map(|s| s.bytes()).unwrap_or(0)
        );
        for (i, layer) in l.layers.iter().enumerate() {
            let pr = |name: &str, t: &grim_tensor::Tensor| {
                let s = t.storage();
                let rocm = s.as_any().downcast_ref::<grim_backend_rocm::RocmStorage>();
                if let Some(r) = rocm {
                    let ptr = r.device_ptr_u64().unwrap_or(0);
                    eprintln!(
                        "[addr-scan] L{i}.{name} = 0x{:x}..0x{:x} ({} B)",
                        ptr,
                        ptr + r.bytes() as u64,
                        r.bytes()
                    );
                }
            };
            pr("wq", layer.wq.weight());
            pr("wk", layer.wk.weight());
            pr("wv", layer.wv.weight());
            pr("wo", layer.wo.weight());
            if let Some(wg) = &layer.w_gate {
                pr("w_gate", wg.weight());
            }
            if let Some(wu) = &layer.w_up {
                pr("w_up", wu.weight());
            }
            if let Some(wd) = &layer.w_down {
                pr("w_down", wd.weight());
            }
        }
    }

    // ── Gate 3: replay tracks eager.
    let mut got = Vec::new();
    for (ri, tok) in TOKENS.iter().enumerate() {
        eprintln!("[marker] replay {ri} begin");
        graph_model
            .forward_replay(&mut graph, *tok)
            .unwrap_or_else(|e| panic!("graph replay: {e}"));
        eprintln!("[marker] replay {ri} done");
        let logits = graph.read_logits_f32().expect("graph logits readback");
        assert!(
            logits.iter().all(|x| x.is_finite()),
            "replay logits must be finite"
        );
        got.push(logits);
        graph.buffers.current_pos = graph.buffers.current_pos.wrapping_add(1);
    }
    for (step, (a, b)) in eager.iter().zip(got.iter()).enumerate() {
        assert_eq!(a.len(), b.len(), "step {step}: logits width");
        let (worst, at, abs) = worst_scaled(a, b);
        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                    if x > bv { (i, x) } else { (bi, bv) }
                })
        };
        let (ea, ev) = argmax(a);
        let (ga, gv) = argmax(b);
        // The scale is an invariant of `a`: folding it inside the filter made
        // this O(n^2) (128256^2 float ops per step) and turned a seconds-long
        // gate into a ten-minute one.
        let scale = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let big = a
            .iter()
            .zip(b.iter())
            .filter(|(x, y)| (*x - *y).abs() > 0.1 * scale)
            .count();
        eprintln!(
            "[llama-q41-parity] step {step}: worst {worst:.3e} at {at} (abs {abs:.3e}); \
             eager argmax {ea} ({ev:.3e}) graph argmax {ga} ({gv:.3e}); {big} cols over 10% of scale"
        );
        assert!(
            worst <= 1e-2,
            "step {step}: replay diverges from eager: worst rel {worst:.3e} at {at} (abs {abs:.3e})"
        );
    }

    unsafe {
        std::env::remove_var("GRIM_EMIT_TASK_GRAPH");
    }
}
