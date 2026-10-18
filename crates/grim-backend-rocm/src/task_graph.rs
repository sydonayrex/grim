//! Model-agnostic task graph for the decode step (plan Phase M, task M1).
//!
//! The emitter is per-model (the existing `forward_capture` path); this module is
//! not. A node is a fixed-size record so it can be written to a device-side ring
//! without allocation, and `opcode` is numeric so the device can switch on it —
//! the same convention `scythe2.rs` already uses.
//!
//! CONDITION 2 (external review 2026-10-17): node buffer references are HANDLES
//! into a model-owned argument table (`arg_table`), not inline pointers. The
//! original spec's `[u64; 4]` could not address real launches — the fused
//! `fused_add_rms_norm_gate_up_silu_dot4_into` alone takes 7 buffers. Eight
//! handles per node cover every in-tree launcher; the table carries the
//! pointer/size/role so the ring interpreter can validate what it is handed.

pub type OpCode = u32;

pub mod op {
    pub const GEMM: super::OpCode = 1;
    pub const MOE_DISPATCH: super::OpCode = 6; // matches the existing scythe opcode
    pub const MOE_COMBINE: super::OpCode = 7;
    pub const ATTENTION: super::OpCode = 8;
    pub const RECURRENT: super::OpCode = 9;
    pub const NORM: super::OpCode = 10;
    pub const ROPE: super::OpCode = 11;
    pub const ELEMENTWISE: super::OpCode = 12;
    pub const SAMPLE: super::OpCode = 13;
    pub const PEER_COPY: super::OpCode = 14;
    /// Extension point: an op with no opcode yet runs through its existing
    /// launcher (Phase M fallback contract). It still participates in the DAG,
    /// so its dependents stay correctly ordered.
    pub const FALLBACK: super::OpCode = 0xFFFF_FFFF;
}

/// Argument-table roles, for interpreter-side validation (a WEIGHT handle passed
/// where an ACT_IN is expected is an emitter bug, not a runtime surprise).
pub mod arg_role {
    pub const WEIGHT: u8 = 0;
    pub const ACT_IN: u8 = 1;
    pub const ACT_OUT: u8 = 2;
    pub const SCRATCH: u8 = 3;
    pub const SCALES: u8 = 4;
}

/// One entry in the model-owned argument table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArgEntry {
    pub ptr: u64,
    pub bytes: u64,
    pub role: u8,
}

/// Index into [`TaskGraph::arg_table`].
pub type ArgId = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TaskNode {
    pub opcode: OpCode,
    /// Opcode-specific geometry. Fixed size: no allocation on the hot path.
    /// For GEMM, `params[0]` is the quant mode (feeds `CapabilitySet::quant_modes`).
    pub params: [u32; 16],
    /// Handles into the model-owned argument table. Eight covers every in-tree
    /// launcher (the widest fused launcher takes 7 buffers).
    pub args: [ArgId; 8],
    pub n_args: u8,
    /// Fan-in, capped at 8; a node needing more must be split.
    pub deps: [u32; 8],
    pub n_deps: u8,
    pub device: u8,
    /// Memory scope for chiplet-aware placement (Fleet, arXiv:2604.15379).
    /// 0 = device-wide. A no-op on monolithic RDNA4; present so MI300/MI350
    /// (8 XCDs, private L2) can place work near its data without an IR change.
    pub scope: u8,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CapabilitySet {
    pub opcodes: std::collections::BTreeSet<OpCode>,
    /// Quant modes seen on GEMM nodes (`params[0]`), DERIVED from the nodes.
    pub quant_modes: std::collections::BTreeSet<u32>,
    pub needs_peer: bool,
    pub max_fan_in: u8,
    pub devices: std::collections::BTreeSet<u8>,
}

#[derive(Clone, Debug, Default)]
pub struct TaskGraph {
    pub nodes: Vec<TaskNode>,
    /// The model-owned argument table (condition 2). Node handles index this.
    pub arg_table: Vec<ArgEntry>,
    pub sinks: Vec<u32>,
    pub caps: CapabilitySet,
}

/// How much of a model's task graph the persistent ring (M4) can run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RingReadiness {
    /// Every opcode in the graph has ring support.
    pub full: bool,
    /// Opcodes present in the graph that the ring cannot execute; each such
    /// node falls back to its ordinary launcher (the fallback contract), so
    /// this is a coverage report, not an error.
    pub missing: Vec<OpCode>,
    /// Human-readable reason, always non-empty when `full` is false.
    pub reason: String,
}

/// Derived compatibility predicate (plan task M3, condition 3).
///
/// Ask the emitted graph what it needs; never maintain a list of models. The
/// only input is the capability set DERIVED from the nodes (`finish()` fills
/// it), so a model that gains an op is automatically re-classified and a
/// partially-supported model still reports `full: false` with the missing set
/// rather than failing. `RING_OPCODES` — the one hand-maintained input, the
/// opcodes the ring implements — lives in [`crate::task_graph_exec`].
pub fn ring_ready(caps: &CapabilitySet) -> RingReadiness {
    let ring: std::collections::BTreeSet<OpCode> =
        crate::task_graph_exec::RING_OPCODES.iter().copied().collect();
    let missing: Vec<OpCode> = caps
        .opcodes
        .iter()
        .filter(|op| !ring.contains(op) && **op != op::FALLBACK)
        .copied()
        .collect();
    if missing.is_empty() {
        RingReadiness {
            full: true,
            missing: Vec::new(),
            reason: "every opcode in the graph has ring support".into(),
        }
    } else {
        RingReadiness {
            full: false,
            reason: format!(
                "ring implements {:?}; graph also needs {missing:?}, which falls back \
                 to its ordinary launcher per node",
                ring.iter().collect::<Vec<_>>()
            ),
            missing,
        }
    }
}

#[derive(Default)]
#[derive(Debug)]
pub struct TaskGraphBuilder {
    nodes: Vec<TaskNode>,
    arg_table: Vec<ArgEntry>,
}

impl TaskGraphBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a buffer in the model-owned argument table; the returned id is
    /// what nodes reference. Duplicate registration is allowed (the same
    /// scratch passed to several nodes) — the table is a bag of handles.
    pub fn push_arg(&mut self, ptr: u64, bytes: u64, role: u8) -> ArgId {
        self.arg_table.push(ArgEntry { ptr, bytes, role });
        (self.arg_table.len() - 1) as ArgId
    }

    pub fn push(
        &mut self,
        opcode: OpCode,
        params: [u32; 16],
        args: &[ArgId],
        deps: &[u32],
        device: u8,
        scope: u8,
    ) -> u32 {
        assert!(args.len() <= 8, "more than 8 args: widen TaskNode::args");
        assert!(deps.len() <= 8, "fan-in > 8: split the node");
        let mut a = [u32::MAX; 8];
        a[..args.len()].copy_from_slice(args);
        let mut d = [u32::MAX; 8];
        d[..deps.len()].copy_from_slice(deps);
        let idx = self.nodes.len() as u32;
        self.nodes.push(TaskNode {
            opcode,
            params,
            args: a,
            n_args: args.len() as u8,
            deps: d,
            n_deps: deps.len() as u8,
            device,
            scope,
        });
        idx
    }

    /// Number of nodes pushed so far. The emitter walks are strictly
    /// sequential, so "the previously emitted node" is `len() - 1` — that is
    /// the dependency every mid-chain node carries.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn finish(self) -> TaskGraph {
        let mut caps = CapabilitySet::default();
        let mut has_dependent = vec![false; self.nodes.len()];
        for n in &self.nodes {
            caps.opcodes.insert(n.opcode);
            caps.devices.insert(n.device);
            caps.max_fan_in = caps.max_fan_in.max(n.n_deps);
            if n.opcode == op::GEMM {
                caps.quant_modes.insert(n.params[0]);
            }
            if n.opcode == op::PEER_COPY {
                caps.needs_peer = true;
            }
            for &d in &n.deps[..n.n_deps as usize] {
                has_dependent[d as usize] = true;
            }
        }
        let sinks = (0..self.nodes.len())
            .filter(|&i| !has_dependent[i])
            .map(|i| i as u32)
            .collect();
        TaskGraph {
            nodes: self.nodes,
            arg_table: self.arg_table,
            sinks,
            caps,
        }
    }
}
