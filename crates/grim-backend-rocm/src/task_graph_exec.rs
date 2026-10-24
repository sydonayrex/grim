//! Execute a [`TaskGraph`] by walking it in dependency order (plan Phase M, task M2).
//!
//! No new kernels: each node is handed to a caller-supplied dispatch closure,
//! which routes it to the same launcher the eager path already uses. That is
//! what keeps this module free of model knowledge — and what makes the
//! compatibility oracle ("IR execution == graph replay, for any model")
//! available before a single new kernel exists.
//!
//! The walk is Kahn's algorithm over the fan-in counts. The IR carries no
//! ordering field, so a mis-wired dependency must surface as a detected cycle
//! rather than a silent reordering. The ready set is a `BTreeSet`, not a
//! stack, so dispatch order is deterministic — the determinism contract the
//! device-side scheduler (M4) has to honour starts here.

use crate::task_graph::{TaskGraph, TaskNode};
use grim_core::error::{Error, Result};

/// Walk the graph in dependency order, handing each node to `dispatch`.
///
/// `dispatch` receives each node's EMITTER INDEX and the node itself, exactly
/// once, after every node it depends on. The index is what links a node back to
/// the emitter's own ordering — the walk order is derived from `deps`, so a
/// caller that has to match a node against something the emitter produced (a
/// per-node plan, a descriptor) cannot recover the index from the walk.
///
/// A cycle (or a dangling dependency index) is an error naming the node, not a
/// silent skip — a scheduler that silently reorders is the failure mode this
/// structure exists to prevent.
pub fn execute(
    tg: &TaskGraph,
    mut dispatch: impl FnMut(u32, &TaskNode) -> Result<()>,
) -> Result<()> {
    let n = tg.nodes.len();
    // Remaining fan-in per node, and the adjacency (dependents) per node.
    let mut remaining = vec![0u32; n];
    let mut dependents: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (i, node) in tg.nodes.iter().enumerate() {
        for &d in &node.deps[..node.n_deps as usize] {
            let d = d as usize;
            if d >= n {
                return Err(Error::Backend(format!(
                    "task graph node {i} depends on {d}, which does not exist"
                )));
            }
            remaining[i] += 1;
            dependents[d].push(i as u32);
        }
    }

    // Ready set: everything with no unmet dependencies, in index order so the
    // walk is deterministic for a given IR.
    let mut ready: std::collections::BTreeSet<u32> = (0..n as u32)
        .filter(|&i| remaining[i as usize] == 0)
        .collect();
    let mut done = 0usize;
    let mut order: Vec<u32> = Vec::with_capacity(n);
    // Opt-in trace: on a fault inside `dispatch` the process dies without
    // unwinding, so the only record of how far the walk got is what was
    // printed BEFORE the call. A fault that names its node is a location; a
    // fault that prints nothing is evidence about nothing.
    let trace = std::env::var_os("GRIM_IR_TRACE").is_some();
    while let Some(&i) = ready.iter().next() {
        ready.remove(&i);
        let node = &tg.nodes[i as usize];
        if trace {
            eprintln!(
                "[ir-exec] node {}/{n} opcode={} args={} params={:?}",
                done + 1,
                node.opcode,
                node.n_args,
                &node.params[..2]
            );
        }
        dispatch(i as u32, node)?;
        order.push(i);
        done += 1;
        for &dep in &dependents[i as usize] {
            remaining[dep as usize] -= 1;
            if remaining[dep as usize] == 0 {
                ready.insert(dep);
            }
        }
    }

    if done != n {
        let stuck: Vec<usize> = (0..n).filter(|&i| remaining[i] > 0).collect();
        return Err(Error::Backend(format!(
            "task graph has a dependency cycle: {stuck:?} never became ready \
             (nodes total {n}, executed {done})"
        )));
    }
    Ok(())
}

/// The opcodes the persistent ring can execute today.
///
/// M4 converts `op::GEMM`: `task_ring` derives the executable subset of an
/// emitted graph, stages its descriptors once, and drains them with the existing
/// persistent wave. The wave's GEMM arm computes in f32, so a QUANTIZED weight is
/// not executable by it — that node stays on its launcher, which is the fallback
/// contract and is why the ring never dequantizes.
///
/// The Q8_0 dot4 arm (plan condition 4's named target) LANDED on 2026-10-20:
/// `op::Q8_0_DOT4` is a separate opcode because a Q8_0 weight cannot run on
/// the f32 arm, and the wave's arm is the launcher's own
/// `grim_dot4_q80_q81_gemv` body — the integer dot, no dequantization, and a
/// bit-identical answer.
///
/// This list is the ONLY hand-maintained part of the readiness predicate:
/// everything else about [`crate::task_graph::ring_ready`] is derived from the
/// capability set, so a model that gains an op is automatically re-classified
/// and an op that is added to a model's IR without ring support shows up in
/// `missing` rather than being silently assumed. The hand-maintained part is
/// pinned against the kernel by
/// `every_advertised_ring_opcode_maps_to_an_implemented_wave_arm`.
pub const RING_OPCODES: &[crate::task_graph::OpCode] = &[
    crate::task_graph::op::GEMM,
    crate::task_graph::op::NORM,
    crate::task_graph::op::ELEMENTWISE,
    crate::task_graph::op::ATTENTION,
    crate::task_graph::op::MOE_DISPATCH,
    // M6: the cross-device handoff. Bound to the wave's `OP_PEER_BROADCAST` arm
    // and to `TaskRingPlan::peers`, so advertising it is honest.
    crate::task_graph::op::PEER_COPY,
    // M5: the qwen35 KDA/recurrent step. The wave's `OP_RECURRENT` arm casts
    // the node's WEIGHT operand to a device-resident `KdaTaskDescriptor` and
    // runs the short-conv step, the batched gated delta rule and the gated head
    // norm inline. Advertised because the arm, the binding AND the emitter's
    // descriptor staging all exist — otherwise `ring_ready` would claim coverage
    // the plan cannot bind, which is the Q8_0_DOT4 failure mode.
    crate::task_graph::op::RECURRENT,
    // PM-6 / condition 4: the Q8_0 dot4 GEMV. The wave's `OP_Q8_0_DOT4` arm
    // runs the launcher's OWN `grim_dot4_q80_q81_gemv` body, so the ring's
    // answer is bit-identical to the launcher's — and the ring never
    // dequantizes. Advertised now that the arm, the binding AND the geometry
    // checks exist; the number it was given before (8) was the wave's
    // `OP_PEER_REDUCE`, and that mistake is still pinned out by
    // `every_advertised_ring_opcode_maps_to_an_implemented_wave_arm`.
    crate::task_graph::op::Q8_0_DOT4,
    // The f32-ACTIVATION variant of the same format — a different launch (a
    // different kernel, a different activation layout), and the one the
    // models' plain Q8_0 projections take.
    crate::task_graph::op::Q8_0_DOT4_F32ACT,
];
