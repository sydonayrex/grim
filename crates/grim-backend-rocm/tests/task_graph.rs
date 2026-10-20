//! M1 gates: the IR types, the derived capability set, and the argument-table
//! contract (condition 2 — nodes carry HANDLES, not inline pointers, and the
//! table is wide enough for the 7-buffer fused launchers).

use grim_backend_rocm::task_graph::{arg_role, op, TaskGraphBuilder};

#[test]
fn capability_set_is_derived_from_the_nodes() {
    let mut b = TaskGraphBuilder::new();
    let w = b.push_arg(10, 4096, arg_role::WEIGHT);
    let x = b.push_arg(20, 1024, arg_role::ACT_IN);
    let y = b.push_arg(30, 1024, arg_role::ACT_OUT);
    let a = b.push(op::GEMM, [1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], &[w, x, y], &[], 0, 0);
    let c = b.push(op::ATTENTION, [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], &[y], &[a], 0, 0);
    let g = b.finish();
    assert_eq!(g.nodes.len(), 2);
    assert_eq!(g.sinks, vec![c], "the node with no dependents is the sink");
    // DERIVED, not declared: the caps come from the nodes, so an emitter that
    // forgets an op cannot under-report what the step needs.
    assert!(g.caps.opcodes.contains(&op::GEMM));
    assert!(g.caps.opcodes.contains(&op::ATTENTION));
    assert!(!g.caps.opcodes.contains(&op::MOE_DISPATCH));
    assert_eq!(g.caps.max_fan_in, 1);
    assert_eq!(g.caps.devices, [0].into_iter().collect());
    // Condition 3 groundwork: GEMM quant modes are derived from params[0].
    assert!(g.caps.quant_modes.contains(&1), "GEMM params[0] is the quant mode");
}

#[test]
fn a_peer_node_marks_the_step_as_needing_peer_access() {
    let mut b = TaskGraphBuilder::new();
    let s = b.push_arg(1, 64, arg_role::ACT_IN);
    // The destination is the PEER's buffer, so it takes the PEER role — an
    // ACT_OUT would mean "this device's output", which is a different thing.
    let d = b.push_arg(2, 64, arg_role::PEER);
    b.push(op::PEER_COPY, [0; 16], &[s, d], &[], 0, 0);
    let w = b.push_arg(3, 128, arg_role::WEIGHT);
    b.push(op::GEMM, [0; 16], &[w], &[0], 1, 0);
    let g = b.finish();
    assert!(g.caps.needs_peer, "a peer node must set needs_peer");
    assert_eq!(g.caps.devices.len(), 2);
}

/// CONDITION 2: the argument table must hold MORE than the old [u64; 4] could —
/// a node with the 7 buffers of `fused_add_rms_norm_gate_up_silu_dot4_into`
/// must be expressible, with handles resolving to the right table entries.
#[test]
fn a_seven_buffer_fused_node_fits_the_argument_table() {
    let mut b = TaskGraphBuilder::new();
    let mut handles = Vec::new();
    for i in 0..7u64 {
        handles.push(b.push_arg(1000 * i, 256 * i, arg_role::SCRATCH));
    }
    let n = b.push(op::GEMM, [0; 16], &handles, &[], 0, 0);
    let g = b.finish();
    let node = &g.nodes[n as usize];
    assert_eq!(node.n_args, 7);
    for slot in 0..7 {
        let entry = &g.arg_table[node.args[slot] as usize];
        assert_eq!(entry.ptr, 1000 * slot as u64);
        assert_eq!(entry.bytes, 256 * slot as u64);
        assert_eq!(entry.role, arg_role::SCRATCH);
    }
}

/// Unset handle slots are u32::MAX and must never be dereferenced by an
/// interpreter — assert the sentinel convention the builders establish.
#[test]
fn unused_arg_slots_are_the_documented_sentinel() {
    let mut b = TaskGraphBuilder::new();
    let only = b.push_arg(5, 16, arg_role::ACT_IN);
    let n = b.push(op::NORM, [0; 16], &[only], &[], 0, 0);
    let g = b.finish();
    assert_eq!(g.nodes[n as usize].n_args, 1);
    assert_eq!(g.nodes[n as usize].args[0], only);
    assert_eq!(g.nodes[n as usize].args[1], u32::MAX);
}

/// M2: the walk dispatches every node exactly once, after its dependencies.
/// The order assertion is the point — a scheduler that reorders silently is
/// the failure the deterministic ready-set exists to prevent.
#[test]
fn execute_walks_in_dependency_order_and_detects_cycles() {
    use grim_backend_rocm::task_graph_exec::execute;

    // chain: 0 -> 1 -> 2, plus 3 depending on 0.
    let mut b = TaskGraphBuilder::new();
    let a = b.push(op::ELEMENTWISE, [0; 16], &[], &[], 0, 0);
    let c = b.push(op::ATTENTION, [0; 16], &[], &[a], 0, 0);
    let d = b.push(op::NORM, [0; 16], &[], &[c], 0, 0);
    let _e = b.push(op::GEMM, [0; 16], &[], &[a], 0, 0);
    let g = b.finish();

    let mut fired = Vec::new();
    execute(&g, |_idx, n| {
        fired.push(n.opcode);
        Ok(())
    })
    .expect("walk");
    assert_eq!(fired.len(), 4, "every node dispatched exactly once");
    assert_eq!(fired[0], op::ELEMENTWISE, "the root goes first");
    assert!(
        fired.iter().position(|&x| x == op::ATTENTION).unwrap()
            < fired.iter().position(|&x| x == op::NORM).unwrap(),
        "a node must run after its dependency"
    );
    let _ = d;

    // Cycle: 0 -> 1 -> 0. `finish` does not validate, so build by hand.
    let mut b = TaskGraphBuilder::new();
    let x = b.push(op::GEMM, [0; 16], &[], &[], 0, 0);
    let y = b.push(op::NORM, [0; 16], &[], &[x], 0, 0);
    let mut g = b.finish();
    g.nodes[x as usize].deps[0] = y;
    g.nodes[x as usize].n_deps = 1;
    let err = execute(&g, |_idx, _n| Ok(())).expect_err("cycle must be detected");
    let msg = format!("{err}");
    assert!(msg.contains("cycle"), "error must name the cycle: {msg}");
}

/// M3 / condition 3: readiness is a FUNCTION of the caps, not a constant.
/// Empty ring: nothing is full, and every unsupported op shows up in
/// `missing` with a reason.
#[test]
fn ring_ready_is_derived_from_the_capability_set() {
    use grim_backend_rocm::task_graph::ring_ready;

    // The expectation is DERIVED from RING_OPCODES, not hard-coded: M4 converted
    // op::GEMM, so a GEMM-only graph is now full and the old "empty ring cannot
    // be full" premise no longer holds for it. Pick an opcode the ring has NOT
    // converted so the predicate still has something to report.
    //
    // DERIVED from the opcode list, not named: this test used to name
    // op::RECURRENT, and M5 converted it — so it failed on a CORRECT predicate.
    // A hardcoded "unconverted" opcode goes stale the moment the ring converts
    // it, which is the same hand-maintained-list failure as the one it guards.
    const ALL: [u32; 11] = [
        op::GEMM,
        op::MOE_DISPATCH,
        op::MOE_COMBINE,
        op::ATTENTION,
        op::RECURRENT,
        op::NORM,
        op::ROPE,
        op::ELEMENTWISE,
        op::SAMPLE,
        op::PEER_COPY,
        op::Q8_0_DOT4,
    ];
    let ring = grim_backend_rocm::task_graph_exec::RING_OPCODES;
    let unconverted = ALL
        .iter()
        .copied()
        .find(|o| !ring.contains(o))
        .expect("every opcode is converted — this test needs one the ring lacks");
    let mut caps = grim_backend_rocm::task_graph::CapabilitySet::default();
    caps.opcodes.insert(unconverted);
    let r = ring_ready(&caps);
    assert!(!r.full, "a ring that does not implement this opcode cannot be full");
    assert_eq!(r.missing, vec![unconverted]);
    assert!(!r.reason.is_empty());

    // Adding another unsupported opcode must add to missing:
    let mut caps2 = caps.clone();
    caps2.opcodes.insert(op::ROPE);
    let r2 = ring_ready(&caps2);
    assert_eq!(r2.missing.len(), r.missing.len() + 1);

    // A graph with only ring-supported opcodes is full:
    let mut caps_supported = grim_backend_rocm::task_graph::CapabilitySet::default();
    caps_supported.opcodes.insert(op::GEMM);
    let r_supp = ring_ready(&caps_supported);
    assert!(r_supp.full);
    assert!(r_supp.missing.is_empty());

    // A graph with no opcodes at all is trivially full.
    let r3 = ring_ready(&grim_backend_rocm::task_graph::CapabilitySet::default());
    assert!(r3.full);
    assert!(r3.missing.is_empty());
}
