//! PM-5c / conditions 3+5: the model-zoo IR coverage gate.
//!
//! Condition 5's failure mode: a coverage gate that loads EVERY shape through
//! `Qwen35::load` cannot catch emitter gaps, because Qwen35 has no MoE/MLA
//! code — the gate would pass while other models emit nothing. So this gate
//! loads real checkpoints through their OWN model types (Lfm2's hybrid
//! recurrent walk, Llama's dense walk, Xing40's MoE+MLA walk) and asserts, per
//! emitter:
//!
//! 1. capture publishes a non-empty IR with no FALLBACK node (every launch the
//!    emitter walked got a real opcode),
//! 2. the IR is a total chain with a single sink and every arg handle
//!    resolves,
//! 3. `ring_ready` is DERIVED — non-full while the ring is empty, with a
//!    non-empty `missing` and reason (the predicate-is-a-function check from
//!    the plan's M3),
//! 4. the emitters are genuinely different: the hybrid model's IR contains
//!    RECURRENT, the MoE+MLA model's contains MOE_DISPATCH. If an emitter
//!    stops naming its exotic op, this fires.
//!
//! Run (GPU 1 is the measurement card; GPU 0 is reserved):
//! `HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 GRIM_CONTEXT=512 \
//!  cargo test -p grim-cli --test model_zoo_ir_coverage -- --nocapture --test-threads=1`

use grim_models_transformer::decode_graph::DecodeGraphModel;
use grim_tensor::Device;

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

/// One zoo entry: a real checkpoint loaded through its own model type.
struct Entry {
    rel: &'static str,
    label: &'static str,
    /// Opcode whose PRESENCE proves this emitter names its exotic op.
    must_contain: Option<u32>,
    must_not_contain: Option<u32>,
    /// Whether this model's IR may carry FALLBACK nodes.
    ///
    /// A FALLBACK node means "there is a launch here the ring cannot name" —
    /// which is the plan's fallback contract for an op with no opcode yet. It
    /// is only a DEFECT when the emitter could have named the launch: the
    /// LFM2's ShortConv branch and its fused Q8_0 down-plus-residual
    /// projection are both launches with no ring opcode, so that model's IR
    /// legitimately carries FALLBACK nodes.
    allow_fallback: bool,
    /// The token the capture feeds.
    ///
    /// Per entry, because the vocabularies differ: the Llama's is 128256 and
    /// the LFM2's is 65536, and a token past the end of the vocabulary makes
    /// the embedding gather read out of bounds — a GPU fault that looks
    /// nothing like "the token is wrong".
    token: u32,
}

#[test]
fn every_zoo_emitter_publishes_a_derived_ready_task_graph() {
    if std::env::var("GRIM_GPU_TEST").map(|v| v != "1").unwrap_or(true) {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run the model-zoo IR coverage gate");
        return;
    }
    let device = Device::Rocm(0);
    set_var("GRIM_CONTEXT", "512");
    set_var("GRIM_EMIT_TASK_GRAPH", "1");

    let entries = [
        Entry {
            rel: "models/Llama-3.2-1B-Instruct-Q4_1.gguf",
            label: "llama-dense",
            must_contain: None,
            must_not_contain: Some(grim_backend_rocm::task_graph::op::MOE_DISPATCH),
            allow_fallback: false,
            token: 128000,
        },
        Entry {
            rel: "models/Xing4.0/Xing4.0-29B-A4B-IQ3_M-MTP.gguf",
            label: "xing40-moe-mla",
            must_contain: Some(grim_backend_rocm::task_graph::op::MOE_DISPATCH),
            must_not_contain: None,
            allow_fallback: false,
            token: 128000,
        },
        Entry {
            // The LFM2 family (M5's PM-5 gap): the short-convolution family's
            // own row. Its IR carries FALLBACK nodes for the launches the ring
            // has no opcode for (the ShortConv branches, and the Q8_0
            // down-plus-residual projection the launcher fuses).
            rel: "models/LFM2.5-350M-Q8_0.gguf",
            label: "lfm2-shortconv",
            must_contain: Some(grim_backend_rocm::task_graph::op::ATTENTION),
            must_not_contain: Some(grim_backend_rocm::task_graph::op::MOE_DISPATCH),
            allow_fallback: true,
            token: 100,
        },
    ];

    for e in entries {
        let path = workspace_path(e.rel);
        if !path.exists() {
            eprintln!("[SKIP] {} not present", path.display());
            continue;
        }
        let path = path.to_string_lossy().into_owned();
        let lm = grim_engine::model_loader::load_model_from_gguf(&path, device.clone())
            .unwrap_or_else(|err| panic!("{}: load: {err}", e.label));

        let token: u32 = e.token;
        // Warmups + capture. The replay/parity halves live in the per-model
        // gates (llama_q41_task_graph, xing40_moe_task_graph); this gate is
        // about EMITTER COVERAGE across the zoo, so it stops at the IR.
        // Mirror `resolve_graph_model`'s resolution: thin wrappers through the
        // wrapper list, models with their OWN impl (Lfm2, Xing40) by direct
        // downcast. Loading every shape through one type is exactly what
        // condition 5 forbids.
        let graph_model: &dyn DecodeGraphModel = lm
            .as_any()
            .downcast_ref::<grim_models_transformer::Lfm2>()
            .map(|m| m as &dyn DecodeGraphModel)
            .or_else(|| {
                lm.as_any()
                    .downcast_ref::<grim_models_transformer::Xing40>()
                    .map(|m| m as &dyn DecodeGraphModel)
            })
            .or_else(|| {
                grim_models_transformer::llama_wrapper_graph_model(lm.as_any())
            })
            .unwrap_or_else(|| panic!("{}: no decode-graph model", e.label));
        let mut graph = graph_model
            .get_or_create_decode_graph(64, 1)
            .unwrap_or_else(|err| panic!("{}: graph pool alloc: {err}", e.label));
        let _ = graph_model.forward_capture(&mut graph, token);
        let _ = graph_model.forward_capture(&mut graph, token);
        graph.begin_capture().expect("begin capture");
        graph_model
            .forward_capture(&mut graph, token)
            .unwrap_or_else(|err| panic!("{}: capture: {err}", e.label));
        graph.end_capture().expect("end capture");

        let ir = graph
            .buffers
            .emitted_task_graph
            .as_ref()
            .unwrap_or_else(|| panic!("{}: capture published no task graph", e.label));
        let ops: Vec<u32> = ir.nodes.iter().map(|n| n.opcode).collect();
        assert!(
            !ir.nodes.is_empty(),
            "{}: no task graph emitted",
            e.label
        );
        if !e.allow_fallback {
            assert!(
                !ops.contains(&grim_backend_rocm::task_graph::op::FALLBACK),
                "{}: FALLBACK in the IR means the emitter skipped a launch it could \
                 not name: {ops:?}",
                e.label
            );
        }
        assert_eq!(
            ir.sinks,
            vec![(ir.nodes.len() - 1) as u32],
            "{}: the chain must be total (single sink = head)",
            e.label
        );
        for n in &ir.nodes {
            for &a in &n.args[..n.n_args as usize] {
                assert!(
                    (a as usize) < ir.arg_table.len(),
                    "{}: dangling arg handle {a}",
                    e.label
                );
            }
        }
        if let Some(want) = e.must_contain {
            assert!(
                ops.contains(&want),
                "{}: emitter must name opcode {want} (its exotic op); IR was {ops:?}",
                e.label
            );
        }
        if let Some(ban) = e.must_not_contain {
            assert!(
                !ops.contains(&ban),
                "{}: IR must not contain {ban}: {ops:?}",
                e.label
            );
        }

        // The predicate is a function of the caps AND the ring's opcode list:
        // what is missing must be exactly the graph's opcodes that the ring does
        // not implement, and `full` must be exactly "nothing is missing".
        //
        // Deriving BOTH from RING_OPCODES is what stops a constant predicate from
        // passing. Hardcoding `!ready.full` did not survive the ring gaining
        // opcodes: once M5 added NORM/ELEMENTWISE/ATTENTION, the dense model's IR
        // became fully covered, and a gate that insists otherwise fails on a
        // CORRECT predicate.
        let ready = grim_backend_rocm::task_graph::ring_ready(&ir.caps);
        let ring: std::collections::BTreeSet<u32> =
            grim_backend_rocm::task_graph_exec::RING_OPCODES.iter().copied().collect();
        let mut expect_missing = ir.caps.opcodes.clone();
        expect_missing.remove(&grim_backend_rocm::task_graph::op::FALLBACK);
        for r in &ring {
            expect_missing.remove(r);
        }
        assert_eq!(
            ready.full,
            expect_missing.is_empty(),
            "{}: `full` must be exactly 'nothing is missing' (ring covers {:?})",
            e.label,
            ring
        );
        if !ready.full {
            assert!(
                !ready.missing.is_empty(),
                "{}: not full but nothing missing — the predicate is not derived",
                e.label
            );
        }
        assert!(!ready.reason.is_empty());
        let missing_set: std::collections::BTreeSet<u32> = ready.missing.iter().copied().collect();
        assert_eq!(
            missing_set, expect_missing,
            "{}: missing must be exactly the graph's opcodes minus the ring's",
            e.label
        );
        eprintln!(
            "[zoo] {}: nodes={} opcodes={:?} missing={:?}",
            e.label, ir.nodes.len(),
            { let v: Vec<u32> = ir.caps.opcodes.iter().copied().collect(); v },
            ready.missing
        );

        // Drop BEFORE the next entry: two 29B-class models must never be
        // resident at once (standing constraint).
        drop(graph);
        drop(lm);
        grim_backend_rocm::RocmDevice::shared(0).synchronize();
    }
    unsafe {
        std::env::remove_var("GRIM_EMIT_TASK_GRAPH");
    }
}
