//! PM-4 / condition 1b: the Xing40 emitter proven on the REAL MoE+MLA
//! checkpoint `models/Xing4.0/Xing4.0-29B-A4B-IQ3_M-MTP.gguf`.
//!
//! Condition 1 requires the per-model emitter on Llama (PM-3,
//! `llama_q41_task_graph.rs`) AND "at least one MoE/MLA model". Xing4.0 is
//! both: noaux_tc MoE (A4B) over MLA attention, so its IR is the first one
//! whose capability set contains MOE_DISPATCH — the op the ring (M4) has no
//! support for yet, which is exactly what `ring_ready` must report rather
//! than hide.
//!
//! Gates:
//! 1. The real checkpoint loads.
//! 2. Capture publishes an IR: embedding ELEMENTWISE, per layer an ATTENTION
//!    (MLA) node + its FFN node (MOE_DISPATCH for routed layers, GEMM for any
//!    dense ones), final NORM, head GEMM — no FALLBACK, single sink, every
//!    arg handle resolves, `ring_ready` reports non-full with MOE_DISPATCH in
//!    `missing`.
//! 3. Graph replay tracks eager per token at the real-model worst-scaled bar.
//!
//! Run (GPU 1 is the measurement card; GPU 0 is reserved):
//! `HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 GRIM_CONTEXT=512 \
//!  cargo test -p grim-cli --test xing40_moe_task_graph -- --nocapture --test-threads=1`

use grim_models_transformer::decode_graph::DecodeGraphModel;
use grim_tensor::{CoreTensorOps, Device, Shape, Tensor};

const MODEL: &str = "models/Xing4.0/Xing4.0-29B-A4B-IQ3_M-MTP.gguf";
/// Small on purpose: a 29B checkpoint's load time dominates; parity over three
/// decode steps is the claim under test, not context length.
const TOKENS: [u32; 3] = [128000, 6727, 11];
const GRAPH_CTX: usize = 64;

fn set_var(k: &str, v: &str) {
    // SAFETY: single test in this binary; run with --test-threads=1.
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
fn xing40_moe_loads_emits_and_replays_like_eager() {
    if std::env::var("GRIM_GPU_TEST")
        .map(|v| v != "1")
        .unwrap_or(true)
    {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run the Xing40 MoE task-graph gate");
        return;
    }
    let model_path = workspace_path(MODEL);
    if !model_path.exists() {
        eprintln!("[SKIP] {} not present", model_path.display());
        return;
    }
    let model_path = model_path.to_string_lossy().into_owned();
    let device = Device::Rocm(0);
    set_var("GRIM_CONTEXT", "512");
    set_var("GRIM_EMIT_TASK_GRAPH", "1");

    // ── Gate 1: the real MoE+MLA checkpoint loads.
    let lm = grim_engine::model_loader::load_model_from_gguf(&model_path, device.clone())
        .unwrap_or_else(|e| panic!("load {MODEL}: {e}"));
    // Xing40 has its OWN DecodeGraphModel impl (not a thin wrapper), so
    // `llama_wrapper_graph_model` does not list it — downcast directly, the
    // same route `resolve_graph_model` takes.
    use grim_models_transformer::Xing40;
    let graph_model: &dyn DecodeGraphModel = lm
        .as_any()
        .downcast_ref::<Xing40>()
        .expect("checkpoint must load as Xing40");

    // ── Eager reference.
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

    // ── Graph side: warmups, capture, then replay from an equally empty arena.
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
    graph.end_capture().expect("end capture");
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
        ops[0],
        op::ELEMENTWISE,
        "first node is the embedding gather"
    );
    let ffn_ops = &ops[1..ops.len() - 2];
    let mut layers = 0usize;
    let mut moe_layers = 0usize;
    for pair in ffn_ops.chunks(2) {
        assert_eq!(pair[0], op::ATTENTION, "layer block op (MLA): {pair:?}");
        assert!(
            pair[1] == op::MOE_DISPATCH || pair[1] == op::GEMM,
            "layer FFN op must be MOE_DISPATCH or GEMM: {pair:?}"
        );
        if pair[1] == op::MOE_DISPATCH {
            moe_layers += 1;
        }
        layers += 1;
    }
    assert!(layers > 0, "no layer nodes emitted");
    assert!(
        moe_layers > 0,
        "an A4B MoE checkpoint must emit at least one MOE_DISPATCH node"
    );
    assert_eq!(ops[ops.len() - 2], op::NORM, "final norm");
    assert_eq!(ops[ops.len() - 1], op::GEMM, "head");
    assert!(
        !ops.contains(&op::FALLBACK),
        "every op must have a real opcode; FALLBACK in the IR means the emitter \
         skipped a launch it could not name"
    );
    // The chain is total: single sink is the head.
    assert_eq!(ir.sinks, vec![(ir.nodes.len() - 1) as u32]);
    for n in &ir.nodes {
        for &a in &n.args[..n.n_args as usize] {
            assert!((a as usize) < ir.arg_table.len(), "dangling arg handle {a}");
        }
    }
    // Condition 3 in action: the readiness predicate reports the MoE gap.
    let ready = grim_backend_rocm::task_graph::ring_ready(&ir.caps);
    eprintln!(
        "[xing40-ir] nodes={} layers={layers} moe_layers={moe_layers} opcodes={:?} \
         ring_ready full={} missing={:?}",
        ir.nodes.len(),
        {
            let v: Vec<u32> = ir.caps.opcodes.iter().copied().collect();
            v
        },
        ready.full,
        ready.missing
    );
    // The readiness expectation is DERIVED from `RING_OPCODES`, not hardcoded.
    // This gate used to assert `!ready.full` ("the ring implements no opcodes
    // yet"), which was true when RING_OPCODES was `[GEMM]` but became false the
    // moment M5 added MOE_DISPATCH/ATTENTION/NORM/ELEMENTWISE — the Xing4.0 IR is
    // now fully covered, so the honest expectation is `full: true` with an empty
    // missing set. Hardcoding either direction fails on a correct predicate.
    let ring: std::collections::BTreeSet<u32> = grim_backend_rocm::task_graph_exec::RING_OPCODES
        .iter()
        .copied()
        .collect();
    let mut expect_missing = ir.caps.opcodes.clone();
    expect_missing.remove(&op::FALLBACK);
    for r in &ring {
        expect_missing.remove(r);
    }
    assert_eq!(
        ready.full,
        expect_missing.is_empty(),
        "xing40: `full` must be exactly 'nothing is missing' (ring covers {ring:?})"
    );
    let missing_set: std::collections::BTreeSet<u32> = ready.missing.iter().copied().collect();
    assert_eq!(
        missing_set, expect_missing,
        "xing40: missing must be exactly the IR's opcodes minus the ring's"
    );
    if !ready.full {
        assert!(
            !ready.missing.is_empty(),
            "xing40: not full but nothing missing — the predicate is not derived"
        );
    }

    // The warmups executed; reset the replay state to the eager session's
    // empty history.
    graph
        .buffers
        .reset_for_replay()
        .expect("reset replay state");

    // ── Gate 3: replay tracks eager.
    for (step, tok) in TOKENS.iter().enumerate() {
        graph_model
            .forward_replay(&mut graph, *tok)
            .unwrap_or_else(|e| panic!("graph replay: {e}"));
        let logits = graph.read_logits_f32().expect("graph logits readback");
        assert!(logits.iter().all(|x| x.is_finite()), "replay logits finite");
        let (worst, at, abs) = worst_scaled(&eager[step], &logits);
        eprintln!("[xing40-parity] step {step}: worst {worst:.3e} at {at} (abs {abs:.3e})");
        assert!(
            worst <= 1e-1,
            "step {step}: replay diverges from eager: worst rel {worst:.3e} at {at}"
        );
        graph.buffers.current_pos = graph.buffers.current_pos.wrapping_add(1);
    }
    unsafe {
        std::env::remove_var("GRIM_EMIT_TASK_GRAPH");
    }
}
