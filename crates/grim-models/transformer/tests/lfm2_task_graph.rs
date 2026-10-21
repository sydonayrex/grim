//! M5 (PM-5 gap): the LFM2's task graph.
//!
//! The LFM2 published NO task graph until this: `lfm2_graph.rs`'s capture
//! walk made no `emit_node` calls, so the coverage table had no LFM2 row and
//! the ring had nothing to bind. The plan named this as the PM-5 gap.
//!
//! Three things are gated here:
//!
//! 1. The capture publishes an IR whose structure describes the walk: the
//!    embedding gather, a branch and an FFN per layer, and the head's norm
//!    and output projection.
//! 2. The IR's QuantFormat axis carries the checkpoint's formats — the LFM2
//!    family's row, which no gate could report before.
//! 3. The walk (the launchers, in the model's order) reproduces the captured
//!    graph BIT-EXACTLY. This is the M2 oracle's job, and for the LFM2 it can
//!    only be stated once the model has an `interpret_ir` at all.
//!
//! The ShortConv layers' branch is a FALLBACK node: the ring has no arm for a
//! stateful short convolution, and the plan's fallback contract says an op
//! with no opcode yet runs through its existing launcher. The IR says so
//! rather than pretending the branch is an attention.
//!
//! Run (GPU 1 is the measurement card; GPU 0 is reserved):
//! `HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 AMD_SERIALIZE_KERNEL=3 GRIM_CONTEXT=512 \
//!  cargo test -p grim-models-transformer --test lfm2_task_graph -- --nocapture --test-threads=1`

use grim_models_transformer::DecodeGraphModel;
use grim_tensor::Device;

fn worst_scaled(a: &[f32], b: &[f32]) -> (f32, usize, f32) {
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
        if !x.is_finite() || !y.is_finite() {
            return (f32::INFINITY, i, f32::NAN);
        }
        let abs = (x - y).abs();
        let rel = abs / scale;
        if rel > worst.0 {
            worst = (rel, i, abs);
        }
    }
    worst
}

#[test]
fn the_lfm2_publishes_a_task_graph_and_its_walk_matches_the_replay() {
    if std::env::var("GRIM_GPU_TEST").map(|v| v != "1").unwrap_or(true) {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run the LFM2 task-graph gate");
        return;
    }
    unsafe {
        std::env::set_var("GRIM_CONTEXT", "512");
        std::env::set_var("GRIM_EMIT_TASK_GRAPH", "1");
    }

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .unwrap_or_default();
    let path = root.join("models/LFM2.5-350M-Q8_0.gguf");
    assert!(
        path.exists(),
        "the LFM2 checkpoint must be present: {}",
        path.display()
    );
    let device = Device::Rocm(0);
    let lm = grim_engine::model_loader::load_model_from_gguf(&path.to_string_lossy(), device.clone())
        .expect("load the LFM2 checkpoint");
    let model = lm
        .as_any()
        .downcast_ref::<grim_models_transformer::Lfm2>()
        .expect("the checkpoint must load as an Lfm2");

    // ── The captured graph, and the IR it publishes.
    let mut graph = model
        .get_or_create_decode_graph(512, 1)
        .expect("capturable");
    let _ = model.forward_capture(&mut graph, 100);
    let _ = model.forward_capture(&mut graph, 100);
    graph.begin_capture().expect("begin capture");
    model
        .forward_capture(&mut graph, 100)
        .expect("recorded capture");
    graph.end_capture().expect("end capture");
    let ir = graph
        .buffers
        .emitted_task_graph
        .clone()
        .expect("the LFM2 capture must publish a task graph");

    use grim_backend_rocm::task_graph::op;
    let mut op_counts: std::collections::BTreeMap<u32, usize> = Default::default();
    for n in &ir.nodes {
        *op_counts.entry(n.opcode).or_default() += 1;
    }
    eprintln!("[lfm2-ir] {} nodes, opcodes {op_counts:?}", ir.nodes.len());
    eprintln!("[lfm2-ir] the axis is {:?}", ir.caps.quant_modes);
    assert!(
        op_counts.contains_key(&op::ELEMENTWISE),
        "the IR must carry the embedding gather: {op_counts:?}"
    );
    assert!(
        op_counts.contains_key(&op::GEMM),
        "the IR must carry the head's output projection (and the FFNs): {op_counts:?}"
    );
    assert!(
        op_counts.contains_key(&op::ATTENTION),
        "the IR must carry the attention layers' branch: {op_counts:?}"
    );
    assert!(
        op_counts.contains_key(&op::NORM),
        "the IR must carry the head's norm: {op_counts:?}"
    );
    // The chain: every node but the first depends on its predecessor, and
    // there is exactly ONE sink (the head's GEMM).
    assert_eq!(
        ir.sinks.len(),
        1,
        "the IR must have one sink: {:?}",
        ir.sinks
    );
    for (i, n) in ir.nodes.iter().enumerate() {
        if i == 0 {
            continue;
        }
        assert!(
            n.deps[..n.n_deps as usize].contains(&((i - 1) as u32)),
            "node {i} must depend on node {}: the IR is a chain",
            i - 1
        );
    }
    // The axis must be derived from the quant-carrying nodes, and the
    // checkpoint's own format must be on it.
    assert!(
        !ir.caps.quant_modes.is_empty(),
        "the LFM2's QuantFormat axis must not be empty"
    );
    let q8_0 = grim_tensor::QuantFormat::Q8_0 as u32;
    assert!(
        ir.caps.quant_modes.contains(&q8_0),
        "the Q8_0 checkpoint's axis {:?} does not carry Q8_0 ({q8_0})",
        ir.caps.quant_modes
    );

    // ── The walk must reproduce the captured graph, bit-exactly.
    let mut gp = model
        .get_or_create_decode_graph(512, 1)
        .expect("capturable (walk)");
    let _ = model.forward_capture(&mut gp, 100);
    let _ = model.forward_capture(&mut gp, 100);
    gp.begin_capture().expect("begin capture (walk)");
    model
        .forward_capture(&mut gp, 100)
        .expect("recorded capture (walk)");
    gp.end_capture().expect("end capture (walk)");
    // BOTH arms start from an empty arena: the capture's two EAGER warm-ups
    // advanced the KV arena's layer totals, so the graph's history would
    // carry two rows the walk's graph does not have.
    graph.buffers.reset_for_replay().expect("reset (replay)");
    gp.buffers.reset_for_replay().expect("reset (walk)");

    let tokens = [100u32, 200, 300];
    let mut replay_logits = Vec::new();
    for tok in tokens {
        model.forward_replay(&mut graph, tok).expect("graph replay");
        replay_logits.push(graph.read_logits_f32().expect("replay logits"));
        graph.buffers.current_pos = graph.buffers.current_pos.wrapping_add(1);
    }
    let mut walk_logits = Vec::new();
    for tok in tokens {
        model
            .forward_interpret_ir(&mut gp, tok)
            .expect("walk interpret");
        walk_logits.push(gp.read_logits_f32().expect("walk logits"));
        gp.buffers.current_pos = gp.buffers.current_pos.wrapping_add(1);
    }
    for (step, (a, b)) in replay_logits.iter().zip(walk_logits.iter()).enumerate() {
        let (e, at, abs) = worst_scaled(a, b);
        eprintln!("[lfm2-ir] step {step}: the walk vs the graph replay, worst {e:.3e} at {at} (abs {abs:.3e})");
        assert!(
            e == 0.0,
            "the LFM2's walk diverges from the captured graph at step {step}: worst {e:.3e} at {at}"
        );
    }
    unsafe {
        std::env::remove_var("GRIM_EMIT_TASK_GRAPH");
    }
}
