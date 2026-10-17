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
    let d = b.push_arg(2, 64, arg_role::ACT_OUT);
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
