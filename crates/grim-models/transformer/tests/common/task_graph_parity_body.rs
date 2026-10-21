// Plan task M2 — the compatibility oracle's shared body.
//
// `include!`d by `task_graph_parity.rs` (Llama) and
// `task_graph_parity_xing40.rs` (Xing4.0), which differ ONLY in the checkpoint
// they name. It lives in a subdirectory so cargo does not compile it as a test
// target of its own.
//
// It is shared rather than copied because the copy DRIFTED: the Xing40 entry
// point still named the Llama checkpoint, so a binary whose name says "xing40"
// was asserting Llama's walk — a gate that does not reach the code it names.
// Two files, one body: the checkpoint and the token sequence are the only
// things either file is allowed to say.

use grim_models_transformer::decode_graph::DecodeGraphModel;
use grim_tensor::{CoreTensorOps, Device, Shape, Tensor};

fn set_var(k: &str, v: &str) {
    // SAFETY: single test in this binary; run with --test-threads=1.
    unsafe { std::env::set_var(k, v) }
}

fn workspace_path(rel: &str) -> std::path::PathBuf {
    let p = std::path::Path::new(rel);
    if p.is_absolute() {
        return p.to_path_buf();
    }
    // This test lives three levels below the workspace root
    // (crates/grim-models/transformer/tests).
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
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

/// NaN-aware worst relative difference (against `a`'s own scale). A gate that
/// cannot fail on NaN is not a gate (see 642a366d).
fn worst_scaled(a: &[f32], b: &[f32]) -> (f32, usize) {
    let scale = a
        .iter()
        .fold(0.0f32, |m, x| if x.is_finite() { m.max(x.abs()) } else { m })
        .max(1e-6);
    let mut worst = (0.0f32, 0usize);
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs() / scale;
        let d = if d.is_finite() { d } else { f32::INFINITY };
        if d > worst.0 {
            worst = (d, i);
        }
    }
    worst
}

/// One model's full oracle: eager reference -> capture + replay -> capture
/// again + interpret, then IR-execution-vs-replay and the QuantFormat-axis
/// assertions. `interpret_bar` is the interpret-vs-replay bar: both paths run
/// the SAME launchers, so the only permitted difference is kernel scheduling,
/// not arithmetic.
fn run_oracle(rel: &str, label: &str, tokens: &[u32], interpret_bar: f32) {
    let device = Device::Rocm(0);
    let model_path = workspace_path(rel);
    let model_path = model_path.to_string_lossy().into_owned();
    set_var("GRIM_EMIT_TASK_GRAPH", "1");

    let lm = grim_engine::model_loader::load_model_from_gguf(&model_path, device.clone())
        .unwrap_or_else(|e| panic!("{label}: load: {e}"));
    // Mirror `resolve_graph_model`: models with their OWN trait impl
    // (Xing40) by direct downcast; thin wrappers (a Q4_1 Llama-3.2 may load
    // as SmolLm2 — its tied head matches the SmolLM2 signature) through the
    // wrapper list, which covers 70+ Llama-family renames.
    let graph_model: &dyn DecodeGraphModel = lm
        .as_any()
        .downcast_ref::<grim_models_transformer::Xing40>()
        .map(|m| m as &dyn DecodeGraphModel)
        .or_else(|| grim_models_transformer::llama_wrapper_graph_model(lm.as_any()))
        .unwrap_or_else(|| panic!("{label}: no decode-graph model"));

    // Stage markers: this oracle has an open, layout-sensitive page fault in
    // the replay/interpret paths (recorded in the plan), and a fault that
    // prints nothing is evidence about nothing. Each stage names itself.
    let stage = |s: &str| eprintln!("[{label}] stage: {s}");

    // ── Eager reference.
    stage("eager");
    let mut session = lm.new_session();
    let mut eager = Vec::new();
    for (step, tok) in tokens.iter().enumerate() {
        let ids = tensor_on(&device, vec![*tok as f32], Shape::new(vec![1, 1]));
        let pos = tensor_on(&device, vec![step as f32], Shape::new(vec![1, 1]));
        eager.push(
            lm.forward(&mut *session, &ids, &pos, &[])
                .expect("eager forward")
                .to_vec_f32()
                .expect("eager d2h"),
        );
    }

    // ── Capture + REPLAY arm on the FIRST pool.
    //
    // ARM ORDER IS LOAD-BEARING and recorded as such: with the interpret arm
    // on the first pool this oracle page-faults at its third step (+4 MB past
    // a mapping), while in this order it runs to completion — a
    // layout-sensitive out-of-bounds in the interpret path that lands in
    // allocator slack here. That is the open replay/interpret fault family in
    // the plan, not a difference in what is compared.
    stage("graph pool A");
    let mut graph = graph_model
        .get_or_create_decode_graph(64, 1)
        .expect("graph pool alloc");
    for _ in 0..2 {
        let _ = graph_model.forward_capture(&mut graph, tokens[0]);
    }
    stage("capture A");
    graph.begin_capture().expect("begin capture");
    graph_model
        .forward_capture(&mut graph, tokens[0])
        .expect("recorded capture");
    graph.end_capture().expect("end capture");
    let ir = graph
        .buffers
        .emitted_task_graph
        .clone()
        .expect("capture must publish the IR");

    graph.buffers.reset_for_replay().expect("reset");
    let mut replayed = Vec::new();
    for (step, tok) in tokens.iter().enumerate() {
        stage(&format!("replay {step}"));
        graph_model
            .forward_replay(&mut graph, *tok)
            .expect("replay");
        replayed.push(graph.read_logits_f32().expect("replay logits"));
        graph.buffers.current_pos = graph.buffers.current_pos.wrapping_add(1);
    }

    // ── INTERPRET arm on a SECOND pool: the published IR executed through the
    // same launchers, node by node.
    stage("graph pool B");
    let mut graph2 = graph_model
        .get_or_create_decode_graph(64, 1)
        .expect("graph pool alloc (interpret)");
    for _ in 0..2 {
        let _ = graph_model.forward_capture(&mut graph2, tokens[0]);
    }
    stage("capture B");
    graph2
        .begin_capture()
        .expect("begin capture (interpret pool)");
    graph_model
        .forward_capture(&mut graph2, tokens[0])
        .expect("recorded capture (interpret pool)");
    graph2.end_capture().expect("end capture (interpret pool)");
    graph2.buffers.reset_for_replay().expect("reset (interpret)");
    let mut interpreted = Vec::new();
    for (step, tok) in tokens.iter().enumerate() {
        stage(&format!("interpret {step}"));
        graph_model
            .forward_interpret_ir(&mut graph2, *tok)
            .expect("interpret");
        interpreted.push(graph2.read_logits_f32().expect("interpret logits"));
        // State probe: the device position must advance by exactly one step
        // per call. An interpreter that ran the wrong number of bumps is
        // invisible in step 0's logits but not in the state. (`pos_dev` ends a
        // step at `base + n_layers`: every layer's block bumps it once — that
        // is the block's own contract, not the interpreter's.)
        if step < 2 {
            let pd: Vec<i32> = graph2
                .buffers
                .pos_dev
                .copy_to_host()
                .expect("pos_dev d2h")
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let tot: Vec<i32> = graph2
                .buffers
                .layer_totals
                .first()
                .and_then(|t| t.copy_to_host().ok())
                .map(|b| {
                    b.chunks_exact(4)
                        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect()
                })
                .unwrap_or_default();
            let am = interpreted
                .last()
                .and_then(|v| {
                    v.iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i)
                })
                .unwrap_or(0);
            eprintln!(
                "[{label}] interpret step {step}: tok {tok} pos_dev={pd:?} layer_totals0={tot:?} argmax={am} host_pos={}",
                graph2.buffers.current_pos
            );
        }
        graph2.buffers.current_pos = graph2.buffers.current_pos.wrapping_add(1);
    }

    // ── The oracle.
    for step in 0..tokens.len() {
        let (ri, rat) = worst_scaled(&interpreted[step], &replayed[step]);
        let (ei, eat) = worst_scaled(&interpreted[step], &eager[step]);
        let (er, _) = worst_scaled(&replayed[step], &eager[step]);
        eprintln!(
            "[{label}] step {step}: interpret-vs-replay {ri:.3e}@{rat} | \
             interpret-vs-eager {ei:.3e}@{eat} | replay-vs-eager {er:.3e}"
        );
        assert!(
            ri <= interpret_bar,
            "{label} step {step}: IR interpretation diverges from graph replay: \
             worst {ri:.3e} at {rat}"
        );
    }

    // ── RING arm (M4, opt-in `GRIM_TASK_RING=1`): the SAME IR walk with every
    // GEMM node routed to the persistent task ring instead of its launcher.
    //
    // Each arm gets its OWN captured pool, starting from an empty arena: the
    // interpret arm has already advanced pool B's KV arenas by one row per step,
    // and a ring arm on the same pool would read those stale rows as context — a
    // wrong comparison, not a ring defect.
    //
    // The context is built from the IR (not from a graph) and stages its
    // descriptors ONCE, so the steady-state step performs no upload.
    if std::env::var_os("GRIM_TASK_RING").is_some() {
        stage("ring: derive plan + stage");
        // `wave_launch_count()` is PROCESS-GLOBAL, and with `GRIM_TASK_RING=1`
        // every `forward_replay` routes through the ring (M5 Step 4) — so the
        // replay arm that runs BEFORE this one has already launched its own
        // waves. This gate is about THIS arm's dispatch, so measure the delta.
        let base_launches = grim_backend_rocm::kernels::task_ring::wave_launch_count();
        let rocm_dev = grim_backend_rocm::RocmDevice::shared(0);
        let ring_ctx =
            grim_backend_rocm::kernels::task_ring::TaskRingContext::from_graph(&rocm_dev, &ir)
                .expect("ring context");
        let Some(mut ring) = ring_ctx else {
            panic!(
                "{label}: GRIM_TASK_RING=1 but the IR has no executable GEMM node — \
                 the ring arm would compare nothing"
            );
        };
        assert!(
            !ring.is_empty(),
            "{label}: the ring plan must be non-empty for this arm to mean anything"
        );
        eprintln!(
            "[{label}] ring plan: {} of {} IR nodes converted to the ring",
            ring.len(),
            ir.nodes.len()
        );

        // Two passes over the same tokens, on two pools, because the KV arenas
        // advance per step and a second pass on the same pool would be a
        // different prompt. Byte-identity is the determinism gate (M5): a
        // device-side scheduler that chooses its schedule at runtime must
        // still produce the same bits for the same prompt.
        let mut pass1 = Vec::new();
        let mut pass2 = Vec::new();
        for (pass, sink) in [("ring", &mut pass1), ("ring determinism", &mut pass2)] {
            stage(&format!("{pass} pool"));
            let mut gp = graph_model
                .get_or_create_decode_graph(64, 1)
                .expect("graph pool alloc (ring)");
            for _ in 0..2 {
                let _ = graph_model.forward_capture(&mut gp, tokens[0]);
            }
            gp.begin_capture().expect("begin capture (ring)");
            graph_model
                .forward_capture(&mut gp, tokens[0])
                .expect("recorded capture (ring)");
            gp.end_capture().expect("end capture (ring)");
            gp.buffers.reset_for_replay().expect("reset (ring)");
            for (step, tok) in tokens.iter().enumerate() {
                stage(&format!("{pass} {step}"));
                graph_model
                    .forward_interpret_ir_ring(&mut gp, *tok, &mut ring)
                    .expect("ring interpret");
                sink.push(gp.read_logits_f32().expect("ring logits"));
                gp.buffers.current_pos = gp.buffers.current_pos.wrapping_add(1);
            }
            std::mem::forget(gp);
        }

        for step in 0..tokens.len() {
            assert_eq!(
                pass1[step], pass2[step],
                "{label} step {step}: the ring is not deterministic — two runs of the \
                 same prompt produced different logits"
            );
        }
        eprintln!("[{label}] ring determinism: byte-identical across two runs");

        // The ring's dispatch covers EXACTLY the converted set — one wave per
        // converted GEMM, no more (nothing extra ran) and no fewer (nothing was
        // silently dropped).
        //
        // This is deliberately NOT "one wave launch per step". A non-resident
        // wave launched on the caller's stream serializes with every host kernel
        // queued behind it, so a step whose GEMMs are interleaved with attention
        // cannot drain them all from one wave without running layer 1's GEMM
        // before layer 0's attention exists. That property is what a RESIDENT
        // wave on its own stream buys (task M6); the whole-set form
        // (`TaskRingContext::launch`) is what the standalone gate exercises,
        // where the GEMM set IS contiguous.
        let ring_launches =
            grim_backend_rocm::kernels::task_ring::wave_launch_count() - base_launches;
        // `step_len()` is the plan's WHOLE dispatch set for one step (gemms +
        // norms + adds + attns + moes), which is what the walk enqueues. Using
        // `len()` here counted GEMMs only, and the absolute counter also included
        // the replay arm's launches, so the gate failed at 144 against an
        // expected 96.
        assert_eq!(
            ring_launches as usize,
            2 * ring.plan().step_len() * tokens.len(),
            "{label}: the ring must dispatch exactly its converted node set — got \
             {ring_launches} wave launches for 2 passes x {} steps x {} ring nodes",
            tokens.len(),
            ring.plan().step_len()
        );
        for step in 0..tokens.len() {
            let (rl, at) = worst_scaled(&pass1[step], &interpreted[step]);
            eprintln!("[{label}] step {step}: ring-vs-launcher {rl:.3e}@{at}");
            assert!(
                rl <= interpret_bar,
                "{label} step {step}: the ring diverged from the launcher path: \
                 worst {rl:.3e} at {at}"
            );
        }
        std::mem::forget(ring);
    }

    // ── The coverage table's QuantFormat axis is FED: every node that
    // carries a weight format names it, and `quant_modes` is derived from
    // them. A GEMM's slot 0 is its weight's format; a MOE_DISPATCH's is its
    // expert BANKS' format (the MoE's experts are a different format from
    // the dense linears around them, and an axis that only saw the dense
    // ones claimed coverage the model does not have).
    let quant_params: Vec<u32> = ir
        .nodes
        .iter()
        .filter(|n| {
            n.opcode == grim_backend_rocm::task_graph::op::GEMM
                || n.opcode == grim_backend_rocm::task_graph::op::MOE_DISPATCH
        })
        .map(|n| n.params[0])
        .collect();
    assert!(
        !quant_params.is_empty(),
        "{label}: no GEMM nodes carry a quant param"
    );
    eprintln!("[{label}] quant params[0] (QuantFormat ids, 0 = native): {quant_params:?}");
    // Negative control on the axis itself. The head nodes used to put
    // `vocab_size` in slot 0; that reads as a plausible integer and silently
    // widened `quant_modes` with a geometry value. `QuantFormat`'s ids are
    // contiguous from 1, so `1..=last` is exactly "is a variant".
    let last_id = grim_tensor::QuantFormat::Iq1M as u32;
    for &q in &quant_params {
        assert!(
            q == 0 || (1..=last_id).contains(&q),
            "{label}: a quant-carrying node's params[0] = {q} is not a QuantFormat id \
             (0 = native, 1..={last_id} = variants) — the coverage axis has \
             been fed a geometry value"
        );
    }
    assert!(
        !ir.caps.quant_modes.is_empty(),
        "{label}: quant_modes must be derived from the quant-carrying nodes"
    );
    assert_eq!(
        ir.caps.quant_modes,
        quant_params.iter().copied().collect(),
        "{label}: quant_modes must equal the set of the quant-carrying nodes' params[0]"
    );
    // Moving `vocab_size` out of slot 0 must not DROP it. The head is the last
    // GEMM, and its output width is exactly the logits length.
    let head = ir
        .nodes
        .iter()
        .filter(|n| n.opcode == grim_backend_rocm::task_graph::op::GEMM)
        .next_back()
        .expect("head GEMM node");
    assert_eq!(
        head.params[1] as usize,
        interpreted[0].len(),
        "{label}: the head GEMM node must still record its output width in params[1]"
    );
    assert!(
        quant_params.len() > 1,
        "{label}: expected per-layer FFN GEMMs in addition to the head"
    );

    // Free EVERYTHING before the process exits: two resident models overcommit
    // the card and the allocator's managed fallback page-faults (recorded
    // failure mode). One model at a time is the standing constraint.
    // Deliberate leak: this test binary hosts ONE model, and tearing the
    // graph objects + allocator slabs down mid-process faulted (recorded
    // above) even under a device guard. The process exits right after the
    // asserts and the OS reclaims everything, so `forget` is the honest way
    // to reach that exit without exercising the flaky path.
    std::mem::forget(graph);
    std::mem::forget(graph2);
    std::mem::forget(lm);
    std::mem::forget(session);
    unsafe {
        std::env::remove_var("GRIM_EMIT_TASK_GRAPH");
    }
}
