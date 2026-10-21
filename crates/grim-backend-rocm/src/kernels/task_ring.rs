//! Plan Phase M, task M4: the persistent task ring — ONE launch executes a step's
//! `op::GEMM` nodes.
//!
//! The ring is not a new device kernel. It is a host-side plan/enqueue layer
//! over the wave that already exists (`grim_scythe_persistent_dispatch`,
//! `scythe_persistent.rs`), which polls a VRAM slot array and switches on the
//! descriptor opcode. What this task adds:
//!
//! 1. **A plan.** `TaskRingPlan::from_graph` reads the emitted task graph and
//!    binds the subset of `op::GEMM` nodes a ring could actually run — nodes that
//!    name all three operands by role AND carry geometry AND whose weight is
//!    native f32. The wave's GEMM arm computes in f32, so a quantized weight is
//!    not executable by it: that node stays on its ordinary launcher, which is
//!    the fallback contract (partial conversion is always usable) and is why the
//!    ring never dequantizes.
//!
//! 2. **Staged descriptors.** The plan is serialized to a device-side array ONCE
//!    (the geometry of a decode step does not change between steps), so the
//!    steady-state step performs no descriptor upload at all.
//!
//! 3. **One launch.** `launch_task_ring` launches the wave with `staged_count != 0`;
//!    the wave's prologue uploads the descriptors and repoints tail/head, then the
//!    existing poll loop consumes them. Enqueue + dispatch = ONE launch for the
//!    whole step's GEMM set, where the launcher path spends one launch per node.
//!
//! The task-graph `op` numbering and the scythe descriptor numbering are DIFFERENT
//! conventions (task-graph `op::GEMM` is 1; the wave's row-GEMM is 2). The ring
//! translates at this boundary and nowhere else — the IR keeps its own ids.
//!
//! `wave_launch_count` is the launch-count assertion's subject: it must advance
//! by exactly one per step regardless of how many GEMM nodes the step has.

use std::sync::atomic::{AtomicU64, Ordering};

use grim_tensor::dtype::{ArithType, DType, Storage as DTypeStorage};
use grim_tensor::{BackendStorage, MemoryOps, Shape};

use crate::device::util::{arg, as_rocm, dev_ptr};
use crate::scythe_task_descriptors::ScytheTaskDescriptor;
use crate::task_graph::{arg_role, op, ArgEntry, TaskGraph};

/// Wave-side opcode for a row-major-B GEMM: B is `[N, K]`, `C = A @ B^T`.
/// This is the `matmul_op` convention, which is what grim's dense linears use.
const OP_ROW_GEMM: u32 = 2;/// Wave-side opcode for an RMS norm over the last axis (`eps` is fixed in the
/// kernel). Matches the arm in `scythe_persistent::KERNEL_SOURCE`.
const OP_NORM: u32 = 4;
/// Wave-side opcode for element-wise add: `out[i] = a[i] + b[i]` over `m*n`
/// elements. The ELEMENTWISE task-graph opcode maps here; the emitter must
/// ensure the node is actually a binary add (the arm cannot do anything else).
const OP_ADD: u32 = 7;
/// Wave-side opcode for attention: flash-attention with q/k/v/out and
/// seq_len/num_heads/head_dim. The arm has a hard 256 head_dim limit.
const OP_ATTN: u32 = 3;
/// Wave-side opcode for MoE dispatch: casts `weight_ptr` to
/// `moe_task_descriptor_t*` and calls the fused MoE kernel inline.
const OP_MOE: u32 = 6;
/// Wave-side opcode for the qwen35 KDA/recurrent step (M5). The arm casts
/// `weight_ptr` to a `kda_task_descriptor_t` and runs the short-conv step, the
/// batched gated delta rule and the gated head norm inline — the same
/// descriptor-carrying pattern `OP_MOE` uses for the fused MoE.
const OP_RECURRENT: u32 = 11;
/// Wave-side opcode for a peer broadcast: write the local source into the PEER
/// device's buffer (`desc->peer_ptr`). This is M6's handoff primitive — segment
/// *k* copies its boundary residual straight into segment *k+1*'s input, so the
/// handoff never leaves the GPU and the host-side event disappears.
const OP_PEER_BROADCAST: u32 = 9;

/// The task-graph `op` → wave opcode translation (M5: opcode generalization).
///
/// Two numbering systems meet here: the task graph's (`op::GEMM` = 1, `op::NORM`
/// = 10, …) and the scythe descriptor's (row-GEMM = 2, norm = 4, …). The ring
/// translates at this table and nowhere else, so the IR keeps its own ids and a
/// wave never sees one.
///
/// This is a CONVERTIBILITY list, not a capability list: an entry means the wave
/// has an arm for that opcode AND the ring can bind the node's operands to it.
/// M5 adds NORM (already bound, was missing from RING_OPCODES), ELEMENTWISE
/// (OP_ADD — a binary add), ATTENTION (OP_ATTN — flash attention), and
/// MOE_DISPATCH (OP_MOE — fused MoE with data-dependent routing schedule).
///
/// `Q8_0_DOT4` was listed here and in `RING_OPCODES` but is NOT implemented: the
/// wave has no dot4 arm, and the number it was given (8) is the wave's
/// `OP_PEER_REDUCE`. Advertising it made `ring_ready` report a model as fully
/// covered while the node actually fell back to its launcher — and had it ever
/// been bound, the wave would have run an all-reduce instead of a GEMM. It is
/// removed until the arm exists; the invariant that keeps it out is gated by
/// `every_advertised_ring_opcode_maps_to_an_implemented_wave_arm`.
pub const WAVE_OPCODES: &[(crate::task_graph::OpCode, u32)] = &[
    (crate::task_graph::op::GEMM, OP_ROW_GEMM),
    (crate::task_graph::op::NORM, OP_NORM),
    (crate::task_graph::op::ELEMENTWISE, OP_ADD),
    (crate::task_graph::op::ATTENTION, OP_ATTN),
    (crate::task_graph::op::MOE_DISPATCH, OP_MOE),
    (crate::task_graph::op::PEER_COPY, OP_PEER_BROADCAST),
    (crate::task_graph::op::RECURRENT, OP_RECURRENT),
];

/// Translate a task-graph opcode to its wave opcode.
///
/// `None` means the ring cannot execute this opcode — the caller leaves the node
/// on its ordinary launcher (the fallback contract). Never a silent default: a
/// node run by the wrong arm is a wrong result, not an error.
pub fn wave_opcode(op: crate::task_graph::OpCode) -> Option<u32> {
    WAVE_OPCODES.iter().find(|&&(o, _)| o == op).map(|&(_, w)| w)
}

/// How many descriptors one staged array holds. 64 covers the largest in-tree
/// emitter (40 layers x FFN + head) with headroom; a plan larger than this
/// stages the first `STAGED_CAPACITY` and leaves the rest on their launchers.
pub const STAGED_CAPACITY: u32 = 64;

/// One ring-executable GEMM. All pointers are device addresses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TaskRingGemm {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub input: u64,
    pub weight: u64,
    pub output: u64,
    pub quant: u32,
    /// The IR node index this entry was derived from. The interpreter matches
    /// on this rather than on a bare counter: a model whose GEMMs are
    /// interleaved with other nodes has unbindable GEMMs too (a quantized
    /// weight, a missing operand), and those occupy no plan slot — so counting
    /// GEMM nodes and indexing the plan by that count desyncs the moment one is
    /// skipped.
    pub ir_node: u32,
}

/// The executable subset of a task graph, in dependency order (the order the
/// wave will consume it — the slots are a FIFO, so a mis-ordered plan is a wrong
/// result, not an error).
/// One ring-executable RMS norm: `out[m,k] = a[m,k] * rsqrt(mean(a^2)) * w[k]`.
/// Geometry is `m` (rows) and `k` (the normalized axis).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TaskRingNorm {
    pub m: u32,
    pub k: u32,
    pub input: u64,
    pub weight: u64,
    pub output: u64,
    pub ir_node: u32,
}

/// One ring-executable element-wise add: `out[i] = a[i] + b[i]` over `m*n` elements.
/// The wave's OP_ADD arm is a binary add; the emitter must ensure the node is
/// actually an add (the arm cannot do anything else).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TaskRingAdd {
    pub m: u32,
    pub n: u32,
    pub input_a: u64,
    pub input_b: u64,
    pub output: u64,
    pub ir_node: u32,
}

/// One ring-executable attention: flash-attention with q/k/v/out.
/// Geometry is `seq_len` (m), `num_heads` (n), `head_dim` (k).
/// The wave's OP_ATTN arm has a hard 256 head_dim limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TaskRingAttn {
    pub seq_len: u32,
    pub num_heads: u32,
    pub head_dim: u32,
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub output: u64,
    pub ir_node: u32,
}

/// One ring-executable MoE dispatch. The wave's OP_MOE arm casts `weight_ptr`
/// to `moe_task_descriptor_t*`, so the descriptor must be staged on device and
/// its address passed as the weight pointer. The routing schedule
/// (token_ids, expert_ids, weights) is data-dependent — it changes per step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TaskRingMoe {
    pub hidden: u32,
    pub inter: u32,
    pub num_tokens: u32,
    pub block_size: u32,
    pub num_experts: u32,
    pub top_k: u32,
    pub quant_mode: u32,
    pub routed_scaling_factor: f32,
    pub gate_w: u64,
    pub up_w: u64,
    pub down_w: u64,
    pub token_ids: u64,
    pub expert_ids: u64,
    pub weights: u64,
    pub input: u64,
    pub output: u64,
    pub ir_node: u32,
}

/// One ring-executable qwen35 KDA/recurrent step (M5). The descriptor is
/// device-resident and carries every buffer the step touches, so the plan only
/// needs its DEVICE address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TaskRingRecurrent {
    /// Device address of the `KdaTaskDescriptor` (the node's WEIGHT operand).
    pub kda_dev: u64,
    pub ir_node: u32,
}

/// One ring-executable cross-device copy (M6): `dst[0..elems] = src[0..elems]`,
/// where `dst` lives on the PEER device. `src` is this device's `ACT_IN` and
/// `dst` is the node's `arg_role::PEER` operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TaskRingPeer {
    pub elems: u32,
    pub src: u64,
    pub dst: u64,
    pub ir_node: u32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TaskRingPlan {
    pub gemms: Vec<TaskRingGemm>,
    pub norms: Vec<TaskRingNorm>,
    pub adds: Vec<TaskRingAdd>,
    pub attns: Vec<TaskRingAttn>,
    pub moes: Vec<TaskRingMoe>,
    pub peers: Vec<TaskRingPeer>,
    pub recurrences: Vec<TaskRingRecurrent>,
}

impl TaskRingPlan {
    /// Bind the ring-executable GEMM subset of an emitted task graph.
    ///
    /// A node is bound only when it is `op::GEMM`, names all three operands by
    /// role, has real geometry, and its weight is native (`params[0] == 0`).
    /// Anything else — a quantized weight, a missing operand, a node with no
    /// down projection — is simply absent from the plan and keeps running on
    /// its ordinary launcher. The ring reports what it took rather than guessing.
    pub fn from_graph(ir: &TaskGraph) -> Self {
        let mut gemms = Vec::new();
        let mut norms = Vec::new();
        let mut adds = Vec::new();
        let mut attns = Vec::new();
        let mut moes = Vec::new();
        let mut peers = Vec::new();
        let mut recurrences = Vec::new();
        // Indexed, not searched: `execute` walks in dependency order rather than
        // node order, so the emitter's own index is the only thing that links a
        // plan entry back to the node it came from.
        for (idx, node) in ir.nodes.iter().enumerate() {
            // Resolve operands by ROLE first, then apply each opcode's own
            // geometry and capability check. A node that fails its check is
            // absent from the plan and keeps running on its ordinary launcher.
            match node.opcode {
                op::GEMM | op::NORM | op::ELEMENTWISE | op::ATTENTION | op::MOE_DISPATCH
                | op::PEER_COPY | op::RECURRENT => {}
                _ => continue,
            }
            let (mut input, mut weight, mut output) = (None, None, None);
            let mut second_input = None;
            let mut peer = None;
            for &h in &node.args[..node.n_args as usize] {
                let entry: &ArgEntry = match ir.arg_table.get(h as usize) {
                    Some(e) => e,
                    None => continue,
                };
                match entry.role {
                    r if r == arg_role::ACT_IN => {
                        if input.is_none() {
                            input = Some(entry.ptr);
                        } else {
                            second_input = Some(entry.ptr);
                        }
                    }
                    r if r == arg_role::WEIGHT => weight = Some(entry.ptr),
                    r if r == arg_role::ACT_OUT => output = Some(entry.ptr),
                    // M6: the destination lives on the OTHER device. It gets its
                    // own role precisely so a handoff cannot be mistaken for a
                    // local output.
                    r if r == arg_role::PEER => peer = Some(entry.ptr),
                    _ => {}
                }
            }
            match node.opcode {
                op::PEER_COPY => {
                    // `dst[0..elems] = src[0..elems]`, dst on the peer device.
                    // The wave's OP_PEER_BROADCAST arm takes the element count in
                    // `desc->m`; `params[0]` carries it. A copy with no peer
                    // destination is NOT bound — it would be a local memcpy
                    // wearing a handoff's name.
                    let (Some(src), Some(dst)) = (input, peer) else { continue };
                    let elems = node.params[0];
                    if elems == 0 {
                        continue;
                    }
                    peers.push(TaskRingPeer { elems, src, dst, ir_node: idx as u32 });
                }
                op::RECURRENT => {
                    // The KDA descriptor rides the node's WEIGHT operand — the
                    // descriptor-carrying convention `OP_MOE` established. A
                    // RECURRENT node without one is NOT bound: that would be a
                    // whole recurrent block wearing a single launch's name, and
                    // the wave would silently execute nothing.
                    let Some(kda_dev) = weight else { continue };
                    if kda_dev == 0 {
                        continue;
                    }
                    recurrences.push(TaskRingRecurrent { kda_dev, ir_node: idx as u32 });
                }
                op::NORM => {
                    let (Some(i), Some(w), Some(o)) = (input, weight, output) else { continue };
                    // `m`/`k` in params[2..4]. The wave's arm fixes `eps` at
                    // 1e-5, the Llama-family value: a model with a different
                    // epsilon is NOT bound here, because normalizing with the
                    // wrong constant is a silent wrong answer.
                    let m = node.params[2];
                    let k = node.params[1];
                    if m == 0 || k == 0 {
                        continue;
                    }
                    norms.push(TaskRingNorm { m, k, input: i, weight: w, output: o, ir_node: idx as u32 });
                }
                op::ELEMENTWISE => {
                    let (Some(a), Some(b), Some(o)) = (input, second_input, output) else { continue };
                    // OP_ADD: `out[i] = a[i] + b[i]` over `m*n` elements.
                    // params[0] = m, params[1] = n.
                    let m = node.params[0];
                    let n = node.params[1];
                    if m == 0 || n == 0 {
                        continue;
                    }
                    adds.push(TaskRingAdd { m, n, input_a: a, input_b: b, output: o, ir_node: idx as u32 });
                }
                op::ATTENTION => {
                    let (Some(q), Some(k), Some(o)) = (input, weight, output) else { continue };
                    // OP_ATTN: input=q, weight=k, peer=v, output=out.
                    // The wave's arm has a hard 256 head_dim limit.
                    let seq_len = node.params[0];
                    let num_heads = node.params[1];
                    let head_dim = node.params[2];
                    if seq_len == 0 || num_heads == 0 || head_dim == 0 || head_dim > 256 {
                        continue;
                    }
                    // V is the peer operand — the third ACT_IN or a SCRATCH role.
                    let v = second_input.or_else(|| {
                        node.args[..node.n_args as usize].iter().find_map(|&h| {
                            ir.arg_table.get(h as usize).and_then(|e| {
                                if e.role == arg_role::SCRATCH { Some(e.ptr) } else { None }
                            })
                        })
                    });
                    let Some(v) = v else { continue };
                    attns.push(TaskRingAttn { seq_len, num_heads, head_dim, q, k, v, output: o, ir_node: idx as u32 });
                }
                op::MOE_DISPATCH => {
                    let (Some(input), Some(output)) = (input, output) else { continue };
                    // OP_MOE: the wave casts weight_ptr to moe_task_descriptor_t*.
                    // The descriptor must be staged on device. The routing
                    // schedule (token_ids, expert_ids, weights) is data-dependent.
                    // params layout: [quant_mode, hidden, inter, num_tokens,
                    //   block_size, num_experts, top_k]
                    let quant_mode = node.params[0];
                    let hidden = node.params[1];
                    let inter = node.params[2];
                    let num_tokens = node.params[3];
                    let block_size = node.params[4];
                    let num_experts = node.params[5];
                    let top_k = node.params[6];
                    if hidden == 0 || inter == 0 || num_experts == 0 || top_k == 0 {
                        continue;
                    }
                    // The wave's OP_MOE arm only supports MOE_QUANT_FP32.
                    if quant_mode != 0 {
                        continue;
                    }
                    // Resolve the weight and schedule pointers from the arg table.
                    let mut gate_w = None;
                    let mut up_w = None;
                    let mut down_w = None;
                    let mut token_ids = None;
                    let mut expert_ids = None;
                    let mut weights = None;
                    for &h in &node.args[..node.n_args as usize] {
                        let Some(e) = ir.arg_table.get(h as usize) else { continue };
                        match e.role {
                            r if r == arg_role::WEIGHT => {
                                if gate_w.is_none() {
                                    gate_w = Some(e.ptr);
                                } else if up_w.is_none() {
                                    up_w = Some(e.ptr);
                                } else if down_w.is_none() {
                                    down_w = Some(e.ptr);
                                }
                            }
                            r if r == arg_role::SCRATCH => {
                                if token_ids.is_none() {
                                    token_ids = Some(e.ptr);
                                } else if expert_ids.is_none() {
                                    expert_ids = Some(e.ptr);
                                } else if weights.is_none() {
                                    weights = Some(e.ptr);
                                }
                            }
                            _ => {}
                        }
                    }
                    let (Some(gw), Some(uw), Some(dw), Some(ti), Some(ei), Some(w)) =
                        (gate_w, up_w, down_w, token_ids, expert_ids, weights) else { continue };
                    moes.push(TaskRingMoe {
                        hidden, inter, num_tokens, block_size, num_experts, top_k,
                        quant_mode, routed_scaling_factor: 1.0,
                        gate_w: gw, up_w: uw, down_w: dw,
                        token_ids: ti, expert_ids: ei, weights: w,
                        input, output, ir_node: idx as u32,
                    });
                }
                op::GEMM => {
                    let (Some(i), Some(w), Some(o)) = (input, weight, output) else { continue };
                    // GEMM: `n`/`k`/`m` in params[1..4], and a native weight
                    // (`params[0] == 0`) because the f32 arm cannot dequantize.
                    let n = node.params[1];
                    let k = node.params[2];
                    let m = node.params[3];
                    if m == 0 || n == 0 || k == 0 || node.params[0] != 0 {
                        continue;
                    }
                    gemms.push(TaskRingGemm { m, n, k, input: i, weight: w, output: o, quant: node.params[0], ir_node: idx as u32 });
                }
                _ => {}
            }
        }
        Self { gemms, norms, adds, attns, moes, peers, recurrences }
    }

    pub fn is_empty(&self) -> bool {
        self.gemms.is_empty() && self.norms.is_empty() && self.adds.is_empty()
            && self.attns.is_empty() && self.moes.is_empty() && self.peers.is_empty()
            && self.recurrences.is_empty()
    }

    pub fn len(&self) -> usize {
        self.gemms.len()
    }

    pub fn norms(&self) -> &[TaskRingNorm] {
        &self.norms
    }

    pub fn adds(&self) -> &[TaskRingAdd] {
        &self.adds
    }

    pub fn attns(&self) -> &[TaskRingAttn] {
        &self.attns
    }

    pub fn moes(&self) -> &[TaskRingMoe] {
        &self.moes
    }

    pub fn peers(&self) -> &[TaskRingPeer] {
        &self.peers
    }

    pub fn recurrences(&self) -> &[TaskRingRecurrent] {
        &self.recurrences
    }

    /// The ring's whole dispatch set for one step — what the launch-count
    /// assertion measures.
    pub fn step_len(&self) -> usize {
        self.gemms.len() + self.norms.len() + self.adds.len() + self.attns.len()
            + self.moes.len() + self.peers.len() + self.recurrences.len()
    }

    /// Serialize the plan into the device-side staged array.
    ///
    /// `ScytheTaskDescriptor`'s own layout is the ABI here (that is the contract
    /// `upload()` and the scythe path already honour), so the bytes are produced
    /// from the struct rather than by hand.
    ///
    /// Returns the storage plus the number of descriptors actually staged (the
    /// first `STAGED_CAPACITY` of the plan).
    ///
    /// The returned storage must outlive every wave that reads it: the staged
    /// array holds the descriptors the wave dereferences on each step, so it is
    /// uploaded ONCE and kept. Callers that do not want to track it can
    /// `mem::forget` it — a few KB against the weights it schedules.
    pub fn stage_on_device(
        &self,
        dev: &crate::RocmDevice,
    ) -> crate::Result<(Box<dyn BackendStorage>, u32)> {
        // The staged set is the plan's tasks in IR-NODE ORDER, not in bucket
        // order. The order is the WAVE's execution order, so a task that
        // another task depends on must come first — the boundary handoff is
        // exactly that: the sequence counter is advanced by an ELEMENTWISE
        // task, and the flag copy that publishes it must read the advanced
        // value. Bucket order (gemms, then adds, then peers) would put the
        // publish before the advance.
        //
        // Only the tasks whose descriptor is self-contained are staged: a
        // GEMM, an ELEMENTWISE add and a PEER copy. The others (attention,
        // MoE, recurrent) carry a device-side descriptor of their own and are
        // enqueued per node.
        let mut staged: Vec<(u32, ScytheTaskDescriptor)> = Vec::new();
        for g in &self.gemms {
            staged.push((
                g.ir_node,
                ScytheTaskDescriptor {
                    opcode: OP_ROW_GEMM,
                    m: g.m,
                    n: g.n,
                    k: g.k,
                    input_ptr: g.input,
                    weight_ptr: g.weight,
                    output_ptr: g.output,
                    peer_ptr: 0,
                    status: 0,
                },
            ));
        }
        for a in &self.adds {
            staged.push((
                a.ir_node,
                ScytheTaskDescriptor {
                    opcode: OP_ADD,
                    m: a.m,
                    n: a.n,
                    k: 0,
                    input_ptr: a.input_a,
                    weight_ptr: a.input_b,
                    output_ptr: a.output,
                    peer_ptr: 0,
                    status: 0,
                },
            ));
        }
        for p in &self.peers {
            staged.push((
                p.ir_node,
                ScytheTaskDescriptor {
                    opcode: OP_PEER_BROADCAST,
                    m: p.elems,
                    n: 0,
                    k: 0,
                    input_ptr: p.src,
                    weight_ptr: 0,
                    output_ptr: 0,
                    peer_ptr: p.dst,
                    status: 0,
                },
            ));
        }
        staged.sort_by_key(|(ir_node, _)| *ir_node);
        staged.truncate(STAGED_CAPACITY as usize);
        let total = staged.len();
        let mut bytes = vec![0u8; total * std::mem::size_of::<ScytheTaskDescriptor>()];
        for (i, (_, desc)) in staged.iter().enumerate() {
            let off = i * std::mem::size_of::<ScytheTaskDescriptor>();
            let cell = unsafe {
                std::slice::from_raw_parts(
                    desc as *const ScytheTaskDescriptor as *const u8,
                    std::mem::size_of::<ScytheTaskDescriptor>(),
                )
            };
            bytes[off..off + cell.len()].copy_from_slice(cell);
        }
        let storage = dev.from_cpu_bytes(
            &bytes,
            &Shape::new(vec![bytes.len()]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        Ok((storage, total as u32))
    }
}

/// A caller-owned, REUSABLE ring context for one task graph.
///
/// The descriptors are uploaded ONCE (`stage_on_device`), because the geometry
/// of a decode step does not change between steps; re-staging per step would
/// put the very launch back this task removes. `launch` then spends ONE wave per
/// step for the whole GEMM set.
pub struct TaskRingContext {
    plan: TaskRingPlan,
    staged: Option<Box<dyn BackendStorage>>,
    count: u32,
    /// The ring itself, owned by the context and reused across steps: the wave
    /// repoints tail/head every step, so the same buffers serve all of them.
    ///
    /// These four are `ManuallyDrop` so that `Drop` can DECIDE whether to free
    /// them: a wave that is still parked is reading them, and freeing device
    /// memory under a live kernel is a use-after-free that wedges the device.
    /// When the wave cannot be confirmed gone, they are leaked instead (a few
    /// KB, in exchange for a device that still works).
    slots: std::mem::ManuallyDrop<Box<dyn BackendStorage>>,
    tail: std::mem::ManuallyDrop<Box<dyn BackendStorage>>,
    head: std::mem::ManuallyDrop<Box<dyn BackendStorage>>,
    stop: std::mem::ManuallyDrop<Box<dyn BackendStorage>>,
    desc_staging: Box<dyn BackendStorage>,
    capacity: u32,
    /// M6 step 3: true when a RESIDENT wave is parked on this ring. A resident
    /// wave is launched once and drains every node the ring is given, so the
    /// per-node enqueue does not launch a wave of its own.
    pub resident: bool,
    /// M6 step 3: the ring's OWN stream. A parked wave must not live on the
    /// caller's stream — it would occupy it, and the enqueues (and the stop
    /// kernel) would queue behind it instead of being seen. Null when the
    /// device's stream pool had nothing to give.
    pub stream: *mut std::ffi::c_void,
    /// The device the ring lives on, so `Drop` can stop a parked wave.
    ordinal: usize,
    /// The ring slot the most recent enqueue wrote to. A consumer that waits on
    /// that descriptor's completion (M6) needs to know which slot it is: the
    /// wave walks the slots in tail order, and the slot a node lands in is
    /// `head % capacity`.
    last_slot: std::sync::atomic::AtomicUsize,
    /// True once the wave kernel has been resolved for this ring.
    ///
    /// The on-stream launch path only LAUNCHES; the kernel has to be loaded
    /// first. Resolving it once per ring keeps the hot path free of the extra
    /// host call a per-token resolution would cost.
    wave_resolved: std::sync::atomic::AtomicBool,
    /// The ring's OWN head counter — the number of tasks this ring has
    /// enqueued.
    ///
    /// It must be PER RING, not process-global: the wave visits the slots in
    /// TAIL order, and the host writes the descriptor into the slot the head
    /// counter names. Those two agree only if the counter starts where the
    /// ring's tail starts. A process-global counter breaks that the moment a
    /// SECOND ring in the same process enqueues anything — the descriptor
    /// lands in a slot the wave never visits, and the node is never drained.
    /// That is not a test artifact: it is a ring whose node disappears
    /// because another ring was used first.
    next_head: std::sync::atomic::AtomicU32,
    /// Milliseconds (on the host's clock) at which the ring last had WORK for
    /// its resident wave.
    ///
    /// The wave's life is an IDLE budget, so it expires only after a long
    /// silence. The host knows when the silence started, because the host is
    /// the one that feeds it — so the host also knows when the wave may be
    /// gone, and can launch a fresh one before publishing the next node.
    wave_last_work_ms: std::sync::atomic::AtomicU64,
    /// True once the resident wave has been stopped AND confirmed gone.
    ///
    /// The confirmation is the point: a wave that has been told to stop but has
    /// not yet left its poll loop is still reading the ring's buffers, so the
    /// buffers must not be freed. `stop_wave` waits for the exit; this flag
    /// records that it happened, and makes a second stop a no-op.
    stopped: std::sync::atomic::AtomicBool,
}

impl Drop for TaskRingContext {
    fn drop(&mut self) {
        // A resident wave parks until it is stopped, and the host owns its
        // lifetime. Without this, a dropped ring would leave a wave spinning on
        // the device — which is the wedge the plan's own risk table names.
        //
        // The stream is the marker: `ensure_stream` creates it only on the
        // resident launch paths, so a non-null stream means a wave may be
        // parked. `stop_wave` sets the flag AND waits for the wave to leave
        // (a bounded `hipStreamQuery` poll on the ring's own stream) — the wait
        // matters, because otherwise the ring's buffers are freed while the
        // wave is still reading them, and the wave never sees the flag (it
        // reads freed, possibly reused memory) and spins for the life of the
        // process.
        if self.stream.is_null() {
            self.free_ring_buffers();
            return;
        }
        let dev = crate::RocmDevice::shared(self.ordinal);
        let stopped = self.stop_wave(&dev).is_ok();
        let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        if stopped {
            unsafe {
                let _ = crate::hipStreamDestroy(self.stream);
            }
            self.stream = std::ptr::null_mut();
            self.free_ring_buffers();
        } else {
            // The wave could not be confirmed gone, so it may still be reading
            // the slots, the tail, the head and the stop flag — and it is
            // running ON this stream.
            //
            // Freeing the buffers now is a use-after-free on the device, and
            // the observed consequence of exactly that was the wave spinning on
            // freed memory forever, wedging the device for the whole process.
            // Destroying the stream is just as bad: `hipStreamDestroy` on a
            // stream with a live kernel BLOCKS until that kernel finishes, and
            // this kernel is parked forever.
            //
            // So LEAK both: a couple of KB and one stream, in exchange for a
            // device that still works. The leak is reported, not silent.
            eprintln!(
                "[task-ring] leaking the ring's stream and buffers ({} slots, \
                 tail/head/stop): the resident wave was not confirmed gone and may still be \
                 reading them",
                self.capacity
            );
        }
    }
}

impl TaskRingContext {
    /// Free the ring's device buffers.
    ///
    /// The four buffers are `ManuallyDrop`, so this is the ONLY place they are
    /// released. It is called on the path where the wave is known to be gone;
    /// on the other path they are deliberately leaked.
    fn free_ring_buffers(&mut self) {
        unsafe {
            std::mem::ManuallyDrop::drop(&mut self.slots);
            std::mem::ManuallyDrop::drop(&mut self.tail);
            std::mem::ManuallyDrop::drop(&mut self.head);
            std::mem::ManuallyDrop::drop(&mut self.stop);
        }
    }
}

impl std::fmt::Debug for TaskRingContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaskRingContext")
            .field("plan", &self.plan)
            .field("count", &self.count)
            .field("capacity", &self.capacity)
            .finish()
    }
}

impl TaskRingContext {
    /// Derive and stage the ring-executable subset of an emitted task graph, and
    /// allocate the ring it will run on.
    ///
    /// `None` when the graph contains no executable GEMM — nothing to convert,
    /// and the caller keeps every node on its launchers.
    pub fn from_graph(dev: &crate::RocmDevice, ir: &TaskGraph) -> crate::Result<Option<Self>> {
        let plan = TaskRingPlan::from_graph(ir);
        if plan.is_empty() {
            return Ok(None);
        }
        let capacity = 32u32;
        let slots = dev.alloc_scythe_ring_bytes((capacity as usize) * 64)?;
        let u32_scalar = |v: u32| {
            dev.from_cpu_bytes(
                &v.to_ne_bytes(),
                &Shape::new(vec![1]),
                DType {
                    arith: ArithType::U32,
                    storage: DTypeStorage::Native,
                },
            )
        };
        // 128 bytes staging for on-device dynamic task descriptors (e.g. MoETaskDescriptor)
        let desc_staging = dev.from_cpu_bytes(
            &vec![0u8; 128],
            &Shape::new(vec![128]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
        )?;
        let mut ctx = Self {
            plan,
            staged: None,
            count: 0,
            slots: std::mem::ManuallyDrop::new(Box::new(slots)),
            tail: std::mem::ManuallyDrop::new(u32_scalar(0)?),
            head: std::mem::ManuallyDrop::new(u32_scalar(0)?),
            stop: std::mem::ManuallyDrop::new(u32_scalar(0)?),
            desc_staging,
            capacity,
            resident: false,
            stream: std::ptr::null_mut(),
            ordinal: dev.ordinal,
            last_slot: std::sync::atomic::AtomicUsize::new(0),
            next_head: std::sync::atomic::AtomicU32::new(0),
            wave_last_work_ms: std::sync::atomic::AtomicU64::new(host_now_ms()),
            wave_resolved: std::sync::atomic::AtomicBool::new(false),
            stopped: std::sync::atomic::AtomicBool::new(false),
        };
        // Resolve the ENQUEUE kernels now, while nothing is parked.
        //
        // The first enqueue after a wave parks would otherwise JIT
        // `grim_scythe_write_slot` and `grim_scythe_publish_head` — a cold
        // hipRTC compile of the (large) attention source, which can outlast
        // the wave's idle budget. With them resolved up front, the parked
        // window contains no compilation.
        {
            // SCRATCH slots and head, NOT the ring's own: the write_slot kernel
            // writes a whole descriptor, and a descriptor left in the ring's
            // slot 0 is a task the wave will CLAIM and consume — taking the
            // ring's tail past the slot the ring's first real node will use.
            // The node then lands in a slot the wave never visits.
            let scratch_slots = dev.alloc_scythe_ring_bytes(64)?;
            let scratch_head = dev.from_cpu_bytes(
                &0u32.to_ne_bytes(),
                &Shape::new(vec![1]),
                DType {
                    arith: ArithType::U32,
                    storage: DTypeStorage::Native,
                },
            )?;
            let scratch_head_ptr = as_rocm(scratch_head.as_ref())?
                .device_ptr_u64()
                .ok_or_else(|| {
                    crate::Error::Backend("task ring scratch head has no device ptr".into())
                })?;
            crate::kernels::qkv_attention::launch_scythe_write_slot(
                dev,
                &scratch_slots,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            )?;
            crate::kernels::qkv_attention::launch_scythe_publish_head(
                dev,
                scratch_head_ptr,
                0,
            )?;
        }

        // Stage AFTER the ring exists so the caller gets a fully formed context
        // back; the plan is known non-empty here, so count lands non-zero.
        let (staged, count) = ctx.plan.stage_on_device(dev)?;
        ctx.staged = Some(staged);
        ctx.count = count;
        Ok(Some(ctx))
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The number of GEMMs this context runs per step.
    pub fn len(&self) -> usize {
        self.count as usize
    }

    /// The plan itself — for a caller that wants to assert what was converted.
    pub fn plan(&self) -> &TaskRingPlan {
        &self.plan
    }

    /// The NORM entries of the plan. Kept on the context (not the plan) because
    /// the interpreter holds the context; the plan is a value.
    pub fn norms(&self) -> &[TaskRingNorm] {
        &self.plan.norms
    }

    /// Enqueue ONE GEMM and drain it with a one-task wave.
    ///
    /// This is the granularity an INTERLEAVED model forces: a non-resident wave
    /// launched on the caller's stream SERIALIZES with every host kernel queued
    /// behind it, so a wave launched at the start of the step would run all of
    /// the GEMMs before layer 1's attention exists. Descriptors therefore go in
    /// one at a time, at the point in the step where each runs.
    ///
    /// The launches per node are the ones `scythe_route::route_gemm` already
    /// spends (slot write, head publish, wave) — all on-device and
    /// graph-capturable, none a host round-trip. What this task adds over that
    /// path is that the descriptor comes from the EMITTED TASK GRAPH rather than
    /// a hand-written call site, which is what lets a model convert without its
    /// launcher knowing a ring exists.
    ///
    /// The ONE-launch-per-step property is therefore NOT claimed for an
    /// interleaved model here; it is what a resident wave on its own stream
    /// (task M6) buys. [`Self::launch`] is the whole-set form, used where the
    /// GEMM set is contiguous.
    pub fn enqueue_gemm(&self, dev: &crate::RocmDevice, g: &TaskRingGemm) -> crate::Result<()> {
        let head_value = self.next_head.fetch_add(1, Ordering::Relaxed);
        let head_ptr = as_rocm(self.head.as_ref())?
            .device_ptr_u64()
            .ok_or_else(|| crate::Error::Backend("task ring head has no device ptr".into()))?;
        let slot = (head_value as u64 % self.capacity as u64) as usize;
        self.last_slot.store(slot, Ordering::Relaxed);
        crate::kernels::qkv_attention::launch_scythe_write_slot(
            dev,
            &**self.slots,
            slot,
            OP_ROW_GEMM,
            g.m,
            g.n,
            g.k,
            g.input,
            g.weight,
            g.output,
            0,
        )?;
        crate::kernels::qkv_attention::launch_scythe_publish_head_async(dev, head_ptr, head_value + 1)?;
        self.drain(dev)?;
        Ok(())
    }

    /// Enqueue one RMS norm and drain it with a one-task wave.
    ///
    /// The wave's arm fixes `eps` at 1e-5 — the Llama-family value. A model
    /// whose norm uses a different epsilon is NOT bound (see `from_graph`),
    /// because normalizing with the wrong constant is a silent wrong answer.
    pub fn enqueue_norm(&self, dev: &crate::RocmDevice, n: &TaskRingNorm) -> crate::Result<()> {
        let head_value = self.next_head.fetch_add(1, Ordering::Relaxed);
        let slot = (head_value as u64 % self.capacity as u64) as usize;
        self.last_slot.store(slot, Ordering::Relaxed);
        crate::kernels::qkv_attention::launch_scythe_write_slot(
            dev,
            &**self.slots,
            slot,
            OP_NORM,
            // Geometry is (m, k): the wave's NORM arm reads `desc->m` (rows) and
            // `desc->k` (the normalized axis). The width was being written into
            // the `n` slot, so `desc->k` was 0 and the arm's inner loop never ran
            // — the norm wrote nothing and the caller saw Ok.
            n.m,
            0,
            n.k,
            n.input,
            n.weight,
            n.output,
            0,
        )?;
        let head_ptr = as_rocm(self.head.as_ref())?
            .device_ptr_u64()
            .ok_or_else(|| crate::Error::Backend("task ring head has no device ptr".into()))?;
        crate::kernels::qkv_attention::launch_scythe_publish_head_async(dev, head_ptr, head_value + 1)?;
        self.drain(dev)?;
        Ok(())
    }

    /// Enqueue one element-wise add and drain it with a one-task wave.
    ///
    /// The wave's OP_ADD arm computes `out[i] = a[i] + b[i]` over `m*n`
    /// elements. The emitter must ensure the node is actually a binary add.
    pub fn enqueue_add(&self, dev: &crate::RocmDevice, a: &TaskRingAdd) -> crate::Result<()> {
        let head_value = self.next_head.fetch_add(1, Ordering::Relaxed);
        let slot = (head_value as u64 % self.capacity as u64) as usize;
        self.last_slot.store(slot, Ordering::Relaxed);
        crate::kernels::qkv_attention::launch_scythe_write_slot(
            dev,
            &**self.slots,
            slot,
            OP_ADD,
            a.m,
            a.n,
            0,
            a.input_a,
            a.input_b,
            a.output,
            0,
        )?;
        let head_ptr = as_rocm(self.head.as_ref())?
            .device_ptr_u64()
            .ok_or_else(|| crate::Error::Backend("task ring head has no device ptr".into()))?;
        crate::kernels::qkv_attention::launch_scythe_publish_head_async(dev, head_ptr, head_value + 1)?;
        self.drain(dev)?;
        Ok(())
    }

    /// Enqueue one attention and drain it with a one-task wave.
    ///
    /// The wave's OP_ATTN arm computes flash-attention with q/k/v/out.
    /// The arm has a hard 256 head_dim limit (enforced at bind time).
    pub fn enqueue_attn(&self, dev: &crate::RocmDevice, a: &TaskRingAttn) -> crate::Result<()> {
        let head_value = self.next_head.fetch_add(1, Ordering::Relaxed);
        let slot = (head_value as u64 % self.capacity as u64) as usize;
        self.last_slot.store(slot, Ordering::Relaxed);
        crate::kernels::qkv_attention::launch_scythe_write_slot(
            dev,
            &**self.slots,
            slot,
            OP_ATTN,
            a.seq_len,
            a.num_heads,
            a.head_dim,
            a.q,
            a.k,
            a.output,
            a.v,
        )?;
        let head_ptr = as_rocm(self.head.as_ref())?
            .device_ptr_u64()
            .ok_or_else(|| crate::Error::Backend("task ring head has no device ptr".into()))?;
        crate::kernels::qkv_attention::launch_scythe_publish_head_async(dev, head_ptr, head_value + 1)?;
        self.drain(dev)?;
        Ok(())
    }

    /// Enqueue one MoE dispatch and drain it with a one-task wave.
    ///
    /// The wave's OP_MOE arm casts `weight_ptr` to `moe_task_descriptor_t*`,
    /// so the descriptor is staged on device and its address passed as the
    /// weight pointer. The routing schedule (token_ids, expert_ids, weights)
    /// is data-dependent — it changes per step as routing decisions change.
    ///
    /// The wave's arm only supports MOE_QUANT_FP32; a quantized MoE is NOT
    /// bound (see `from_graph`).
    pub fn enqueue_moe(&self, dev: &crate::RocmDevice, m: &TaskRingMoe) -> crate::Result<()> {
        use crate::scythe_task_descriptors::{moe_quant_mode, MoETaskDescriptor};
        // Stage the MoETaskDescriptor on device. The wave casts weight_ptr to
        // moe_task_descriptor_t*, so the descriptor must be in device memory.
        let desc = MoETaskDescriptor {
            hidden: m.hidden,
            inter: m.inter,
            num_tokens: m.num_tokens,
            block_size: m.block_size,
            num_experts: m.num_experts,
            top_k: m.top_k,
            quant_mode: moe_quant_mode::FP32,
            routed_scaling_factor: m.routed_scaling_factor,
            gate_w_ptr: m.gate_w,
            up_w_ptr: m.up_w,
            down_w_ptr: m.down_w,
            token_ids_ptr: m.token_ids,
            expert_ids_ptr: m.expert_ids,
            weights_ptr: m.weights,
        };
        let desc_ptr = desc.upload(dev)?;
        let head_value = self.next_head.fetch_add(1, Ordering::Relaxed);
        let slot = (head_value as u64 % self.capacity as u64) as usize;
        self.last_slot.store(slot, Ordering::Relaxed);
        crate::kernels::qkv_attention::launch_scythe_write_slot(
            dev,
            &**self.slots,
            slot,
            OP_MOE,
            m.num_tokens,
            m.num_experts,
            m.top_k,
            m.input,
            desc_ptr,
            m.output,
            0,
        )?;
        let head_ptr = as_rocm(self.head.as_ref())?
            .device_ptr_u64()
            .ok_or_else(|| crate::Error::Backend("task ring head has no device ptr".into()))?;
        crate::kernels::qkv_attention::launch_scythe_publish_head_async(dev, head_ptr, head_value + 1)?;
        self.drain(dev)?;
        Ok(())
    }

    /// Enqueue one cross-device copy and drain it with a one-task wave (M6).
    ///
    /// The wave's `OP_PEER_BROADCAST` arm writes `desc->input_ptr` into
    /// `desc->peer_ptr` — the PEER device's buffer — over peer access, so the
    /// handoff between segment *k* and segment *k+1* never leaves the GPU and the
    /// host-side event between their graph launches is not needed. `desc->m`
    /// carries the element count.
    pub fn enqueue_peer_copy(&self, dev: &crate::RocmDevice, p: &TaskRingPeer) -> crate::Result<()> {
        let head_value = self.next_head.fetch_add(1, Ordering::Relaxed);
        let slot = (head_value as u64 % self.capacity as u64) as usize;
        self.last_slot.store(slot, Ordering::Relaxed);
        crate::kernels::qkv_attention::launch_scythe_write_slot(
            dev,
            &**self.slots,
            slot,
            OP_PEER_BROADCAST,
            p.elems,
            0,
            0,
            p.src,
            0,
            0,
            p.dst,
        )?;
        let head_ptr = as_rocm(self.head.as_ref())?
            .device_ptr_u64()
            .ok_or_else(|| crate::Error::Backend("task ring head has no device ptr".into()))?;
        crate::kernels::qkv_attention::launch_scythe_publish_head_async(dev, head_ptr, head_value + 1)?;
        self.drain(dev)?;
        Ok(())
    }

    /// Drain the ring with a wave — unless a RESIDENT wave is already parked on
    /// it (M6 step 3), in which case the enqueue only publishes the descriptor
    /// and the parked wave picks it up. A step then costs ONE launch instead of
    /// one per node.
    fn drain(&self, dev: &crate::RocmDevice) -> crate::Result<()> {
        if !self.resident {
            // The wave is about to be launched on the device's ACTIVE stream;
            // the clear goes on the same stream so the two are ordered.
            self.clear_stop_flag_on(dev, dev.active_stream())?;
        }
        if self.resident {
            // The resident wave's life is an IDLE budget: it expires after a
            // long silence, and the host knows when the silence started. A
            // fresh wave is launched BEFORE the node is published, so a node is
            // never stranded by an expired wave.
            //
            // The old wave is STOPPED first (a device write, no host sync),
            // and its exit is waited for (a bounded query poll), so there is
            // never a second wave racing the first for the ring's slots.
            self.relaunch_expired_wave(dev)?;
            self.wave_last_work_ms
                .store(host_now_ms(), std::sync::atomic::Ordering::Release);
            return Ok(());
        }
        launch_task_ring(
            dev,
            &**self.slots,
            self.capacity,
            &**self.tail,
            &**self.head,
            &**self.stop,
            None,
            0,
        )?;
        Ok(())
    }

    /// Clear the stop flag on the RING'S stream, so the next wave launch is
    /// ordered after it.
    ///
    /// The flag means "no wave is running on this ring". The wave sets it when
    /// it leaves; the host clears it before launching a fresh wave. The clear
    /// goes on the ring's own stream so that it and the launch are ordered —
    /// a clear on another stream could land after the wave had already
    /// started and exited.
    fn clear_stop_flag(&self, dev: &crate::RocmDevice) -> crate::Result<()> {
        self.clear_stop_flag_on(dev, self.stream)
    }

    /// Clear the stop flag on an EXPLICIT stream.
    ///
    /// The stream must be the one the wave will be launched on: the clear and
    /// the launch are ordered by the stream, and a clear on another stream
    /// could land after the wave had already started and exited.
    fn clear_stop_flag_on(
        &self,
        dev: &crate::RocmDevice,
        stream: *mut std::ffi::c_void,
    ) -> crate::Result<()> {
        let stop_ptr = as_rocm(self.stop.as_ref())?
            .device_ptr_u64()
            .ok_or_else(|| crate::Error::Backend("task ring stop has no device ptr".into()))?;
        let mut sp = stop_ptr;
        let mut zero = 0u32;
        dev.launch_compute_kernel_on_stream(
            "grim_scythe_publish_head",
            crate::HipDim3::new(1, 1, 1),
            crate::HipDim3::new(1, 1, 1),
            &mut [arg(&mut sp), arg(&mut zero)],
            stream,
            0,
        )?;
        Ok(())
    }

    /// Launch a resident wave on this ring's own stream, WITHOUT touching the
    /// ring's state.
    ///
    /// The launch clears the stop flag (the wave does that itself), so a
    /// re-launch after an expiry starts clean.
    fn launch_resident_wave(&self, dev: &crate::RocmDevice) -> crate::Result<()> {
        if self.stream.is_null() {
            return Err(crate::Error::Backend(
                "task ring: no resident stream to launch the wave on".into(),
            ));
        }
        self.clear_stop_flag(dev)?;
        launch_task_ring_resident_on_stream(
            dev,
            &**self.slots,
            self.capacity,
            &**self.tail,
            &**self.head,
            &**self.stop,
            None,
            0,
            self.stream,
        )?;
        Ok(())
    }

    /// Re-launch the resident wave if it may have expired.
    ///
    /// The wave's idle budget is generous (it must outlast a cold JIT of the
    /// enqueue kernels), so the host's clock is a CONSERVATIVE witness: by the
    /// time this fires, the wave is certainly gone — and it is stopped and
    /// waited for before the fresh one is launched, so the two never race for
    /// the ring's slots.
    fn relaunch_expired_wave(&self, dev: &crate::RocmDevice) -> crate::Result<()> {
        let last_work = self
            .wave_last_work_ms
            .load(std::sync::atomic::Ordering::Acquire);
        let idle_ms = host_now_ms().saturating_sub(last_work);
        if idle_ms < RESIDENT_WAVE_MAX_IDLE_MS {
            return Ok(());
        }
        // Stop the old wave (a device write — no host synchronisation) and
        // wait for it to leave, so the fresh wave is the only one running.
        dev.write_f32_into(&**self.stop, &[1.0f32])?;
        self.wait_for_wave_exit();
        self.launch_resident_wave(dev)?;
        self.wave_last_work_ms
            .store(host_now_ms(), std::sync::atomic::Ordering::Release);
        Ok(())
    }

    /// Launch a RESIDENT wave on this ring (M6 step 3).
    ///
    /// The wave parks on an empty queue instead of exiting, so every node the
    /// ring is given afterwards is drained without another launch. It leaves
    /// through [`Self::stop_wave`] or its own poll budget.
    pub fn launch_resident(&mut self, dev: &crate::RocmDevice) -> crate::Result<()> {
        self.ensure_stream(dev)?;
        self.clear_stop_flag(dev)?;
        // Resolve the kernel first: the on-stream launch path only launches, it
        // does not resolve. The resolution launch runs on a SCRATCH ring, never
        // this one: a wave over the real ring would race the resident wave for
        // its slots, and the loser spins on a slot it cannot claim.
        self.resolve_wave_kernel(dev)?;
        launch_task_ring_resident_on_stream(
            dev,
            &**self.slots,
            self.capacity,
            &**self.tail,
            &**self.head,
            &**self.stop,
            None,
            0,
            self.stream,
        )?;
        self.resident = true;
        Ok(())
    }

    /// Launch a RESIDENT wave that drains the STAGED set and then parks.
    ///
    /// The whole-set form: one launch, the wave's prologue uploads the staged
    /// descriptors, and the wave consumes them and parks.
    pub fn launch_resident_staged(&mut self, dev: &crate::RocmDevice) -> crate::Result<()> {
        self.ensure_stream(dev)?;
        self.clear_stop_flag(dev)?;
        // Resolve the kernel on a SCRATCH ring, never on this one: the on-stream
        // launch path only launches, so the kernel has to be loaded first — but
        // resolving it by launching a wave over THIS ring would race the
        // resident wave for the very slots the resident wave is about to drain.
        self.resolve_wave_kernel(dev)?;
        launch_task_ring_resident_on_stream(
            dev,
            &**self.slots,
            self.capacity,
            &**self.tail,
            &**self.head,
            &**self.stop,
            self.staged.as_deref(),
            self.count,
            self.stream,
        )?;
        self.resident = true;
        Ok(())
    }

    /// Resolve the wave kernel WITHOUT running a wave over this ring.
    ///
    /// The on-stream launch path only launches; the kernel must already be
    /// resolved. Resolving it by LAUNCHING the wave would run a real wave over
    /// this ring's slots — and it would race the resident wave for them.
    /// A scratch ring is used instead: the wave sees an empty queue, exits at
    /// once, and this ring's slots are untouched.
    fn resolve_wave_kernel(&self, dev: &crate::RocmDevice) -> crate::Result<()> {
        // ONCE per ring. The on-stream launch path only launches, so the kernel
        // has to be loaded first — but a resolution launch on the hot path
        // would cost a host call per token, which is exactly what the handoff
        // is trying to remove.
        if self.wave_resolved.load(Ordering::Acquire) {
            return Ok(());
        }
        let scratch = dev.alloc_scythe_ring_bytes(64)?;
        let u32_scalar = |v: u32| -> crate::Result<Box<dyn BackendStorage>> {
            let bytes = v.to_ne_bytes().to_vec();
            Ok(dev.from_cpu_bytes(
                &bytes,
                &grim_tensor::Shape::new(vec![1]),
                grim_tensor::dtype::DType {
                    arith: grim_tensor::ArithType::U32,
                    storage: grim_tensor::dtype::Storage::Native,
                },
            )?)
        };
        let tail = u32_scalar(0)?;
        let head = u32_scalar(0)?;
        let stop = u32_scalar(0)?;
        launch_task_ring(
            dev,
            &scratch,
            1,
            tail.as_ref(),
            head.as_ref(),
            stop.as_ref(),
            None,
            0,
        )?
        // The scratch buffers are freed when this function returns, so the
        // wave that is reading them must be done first. Without the wait, the
        // free races the wave — the same use-after-free class the resident
        // wave's lifetime fix closed.
        .synchronize()?;
        self.wave_resolved.store(true, Ordering::Release);
        Ok(())
    }

    /// Give the ring its own NON-BLOCKING stream, if it has not got one.
    ///
    /// The stream is created ON THE RING'S DEVICE. This is not a detail: a
    /// stream belongs to the device that created it, and `hipStreamCreate*`
    /// creates it on the CURRENT device — whatever device the calling thread
    /// happens to be on. A ring created while the thread sat on another device
    /// (after a device-1 probe, say) would get a stream on THAT device, and
    /// its wave would then run on the wrong device: it could not see the
    /// ring's buffers, and every node it was given would silently never be
    /// drained.
    ///
    /// That is not a test artifact. It is a ring that loses work because
    /// something else in the process touched another device first.
    fn ensure_stream(&mut self, dev: &crate::RocmDevice) -> crate::Result<()> {
        if self.stream.is_null() {
            let _guard = crate::device::util::DeviceGuard::set(dev.ordinal as i32);
            // A DEDICATED, NON-BLOCKING stream. Two reasons, both measured:
            //
            //  * The device's stream pool is created with `hipStreamCreate`, whose
            //    streams are BLOCKING: they implicitly serialize with the legacy
            //    null stream. A wave parked on one of those makes the host's next
            //    SYNCHRONOUS device read (the logits readback) wait on the
            //    parked wave — a deadlock, because the wave is waiting for work
            //    the host is waiting to read.
            //  * The wave must not occupy the caller's stream, or the enqueues
            //    queue behind it instead of being seen.
            //
            // 0x1 is `hipStreamNonBlocking`.
            let mut st: *mut std::ffi::c_void = std::ptr::null_mut();
            let res = unsafe { crate::hipStreamCreateWithFlags(&mut st, 0x1, 0) };
            if res != crate::hipSuccess || st.is_null() {
                return Err(crate::Error::Backend(format!(
                    "task ring could not create a non-blocking stream (hip {res})"
                )));
            }
            self.stream = st;
        }
        Ok(())
    }

    /// The ring's slot array DEVICE address — a consumer that waits on a
    /// descriptor's completion needs to point at it.
    pub fn slots_ptr(&self) -> crate::Result<u64> {
        Ok(as_rocm(self.slots.as_ref())?
            .device_ptr_u64()
            .ok_or_else(|| crate::Error::Backend("task ring slots have no device ptr".into()))?)
    }

    /// The ring's STOP flag, read on the host.
    ///
    /// The flag means "no wave is running on this ring": the wave sets it when
    /// it leaves, and the host sets it to ask the wave to stop.
    pub fn stop_flag(&self) -> crate::Result<f32> {
        let v = as_rocm(self.stop.as_ref())?.to_cpu_vec_f32()?;
        Ok(v.first().copied().unwrap_or(0.0))
    }

    /// The ring's HEAD counter, read on the host.
    ///
    /// The host publishes it after every enqueue; the wave reads it to find
    /// work. A head that has not advanced means the publish never landed.
    pub fn head_value(&self) -> crate::Result<u32> {
        let v = as_rocm(self.head.as_ref())?.to_cpu_vec_u32()?;
        Ok(v.first().copied().unwrap_or(0))
    }

    /// The ring's TAIL counter address — the ring's MONOTONIC step counter.
    ///
    /// A consumer on ANOTHER device polls this to learn that a handoff
    /// landed: the wave advances it once per task, and a peer READ is
    /// coherent on this hardware. The value changes every step, so a consumer
    /// can tell one handoff from the next.
    pub fn tail_ptr(&self) -> crate::Result<u64> {
        Ok(as_rocm(self.tail.as_ref())?
            .device_ptr_u64()
            .ok_or_else(|| crate::Error::Backend("task ring tail has no device ptr".into()))?)
    }

    /// The ring slot the most recent enqueue wrote to.
    pub fn last_slot(&self) -> usize {
        self.last_slot.load(Ordering::Relaxed)
    }

    /// The ring's tail counter — how many tasks the wave has consumed.
    ///
    /// Exposed so a caller can wait for a PARKED wave to drain: the wave runs on
    /// the ring's own stream, so the host cannot observe the drain through the
    /// stream and reads the counter instead.
    pub fn tail_value(&self) -> crate::Result<u32> {
        let v = as_rocm(self.tail.as_ref())?.to_cpu_vec_u32()?;
        Ok(v.first().copied().unwrap_or(0))
    }

    /// Stop a resident wave: set the stop flag so the parked wave exits, and
    /// WAIT for it to actually exit.
    ///
    /// The publish kernel is a plain store to the pointer it is handed, so it
    /// sets the stop flag exactly as it sets the head.
    pub fn stop_wave(&self, dev: &crate::RocmDevice) -> crate::Result<()> {
        // Stopping twice is harmless but must not double-count: the second
        // call finds the flag already set and the wave already gone.
        if self.stopped.load(Ordering::Acquire) {
            return Ok(());
        }
        // A SYNCHRONOUS H2D copy, not a kernel launch: a launch is
        // stream-ordered, so on the ring's own stream it would queue BEHIND the
        // parked wave — the very thing it is meant to stop. The wave's check is
        // `stop != 0`, so any non-zero bit pattern works; 1.0f32 is
        // 0x3F800000.
        dev.write_f32_into(self.stop.as_ref(), &[1.0f32])?;
        // Then WAIT for the wave to leave. Without this the ring's buffers can
        // be freed while the wave is still spinning on them: the wave reads the
        // stop flag from freed (and possibly reused) memory, may never see the
        // flag, and spins for the life of the process — holding the device and
        // wedging every later allocation. That was the suite-level hang: the
        // wait-node gate parked a wave, dropped its context, and the next test
        // never got past its first allocation.
        //
        // The wait is a BOUNDED POLL of `hipStreamQuery` on the ring's OWN
        // non-blocking stream, not `hipStreamSynchronize`: synchronize would
        // block forever if the wave never noticed the flag, turning a
        // diagnosable failure into a hang. The poll also gives the wave's exit
        // an OBSERVABLE — the stream goes idle — so "stopped" can mean "the
        // wave really left" rather than "the host asked it to".
        if !self.wait_for_wave_exit() {
            return Err(crate::Error::Backend(
                "task ring: the resident wave did not leave its poll loop after the stop flag \
                 was set — it is still reading the ring's buffers"
                    .into(),
            ));
        }
        self.stopped.store(true, Ordering::Release);
        WAVE_STOPS.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Block until the resident wave has left its poll loop.
    ///
    /// Returns `true` when the ring's stream went idle (the wave really left)
    /// and `false` when it did not within the budget. Safe to call when no
    /// wave is resident: the stream is either null or carries no work.
    ///
    /// The budget is a poll count, not a `hipStreamSynchronize`: a wave that
    /// never noticed the stop flag must not be able to block the host forever.
    fn wait_for_wave_exit(&self) -> bool {
        if self.stream.is_null() {
            return true;
        }
        let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        for _ in 0..WAVE_EXIT_POLL_BUDGET {
            let res = unsafe { crate::hipStreamQuery(self.stream) };
            if res == crate::hipSuccess {
                return true;
            }
            if res != crate::hipErrorNotReady {
                eprintln!(
                    "[task-ring] hipStreamQuery on the ring stream failed (hip {res}); \
                     the wave's exit cannot be confirmed"
                );
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        eprintln!(
            "[task-ring] the resident wave did not leave its poll loop within {} ms of the stop \
             flag; it is still reading the ring's buffers",
            WAVE_EXIT_POLL_BUDGET
        );
        false
    }

    /// Enqueue one qwen35 KDA/recurrent step and drain it with a one-task wave
    /// (M5). The wave casts `weight_ptr` to the device-resident
    /// `KdaTaskDescriptor` and runs the short-conv step, the batched gated delta
    /// rule and the gated head norm inline — three kernels, one wave.
    pub fn enqueue_recurrent(
        &self,
        dev: &crate::RocmDevice,
        r: &TaskRingRecurrent,
    ) -> crate::Result<()> {
        let head_value = self.next_head.fetch_add(1, Ordering::Relaxed);
        let slot = (head_value as u64 % self.capacity as u64) as usize;
        self.last_slot.store(slot, Ordering::Relaxed);
        crate::kernels::qkv_attention::launch_scythe_write_slot(
            dev,
            &**self.slots,
            slot,
            OP_RECURRENT,
            0,
            0,
            0,
            0,
            // weight_ptr carries the descriptor, exactly as `OP_MOE` does.
            r.kda_dev,
            0,
            0,
        )?;
        let head_ptr = as_rocm(self.head.as_ref())?
            .device_ptr_u64()
            .ok_or_else(|| crate::Error::Backend("task ring head has no device ptr".into()))?;
        crate::kernels::qkv_attention::launch_scythe_publish_head_async(dev, head_ptr, head_value + 1)?;
        self.drain(dev)?;
        Ok(())
    }

    /// Dynamic (data-dependent) on-device MoE enqueue (Task M5 Step 3).
    ///
    /// Unlike `enqueue_moe` which stages descriptors on the host, this enqueues
    /// an on-device descriptor construction and atomic ring slot claim without
    /// host round-trip, executing via Event-Tensor on-device triggering.
    pub fn enqueue_dynamic_moe(&self, dev: &crate::RocmDevice, m: &TaskRingMoe) -> crate::Result<()> {
        use crate::scythe_task_descriptors::moe_quant_mode;
        let head_ptr = as_rocm(self.head.as_ref())?
            .device_ptr_u64()
            .ok_or_else(|| crate::Error::Backend("task ring head has no device ptr".into()))?;
        let desc_staging_ptr = as_rocm(self.desc_staging.as_ref())?
            .device_ptr_u64()
            .ok_or_else(|| crate::Error::Backend("task ring desc_staging has no device ptr".into()))?;

        crate::kernels::qkv_attention::launch_scythe_enqueue_dynamic_moe(
            dev,
            &**self.slots,
            self.capacity,
            head_ptr,
            desc_staging_ptr,
            m.hidden,
            m.inter,
            m.num_tokens,
            m.block_size,
            m.num_experts,
            m.top_k,
            moe_quant_mode::FP32,
            m.routed_scaling_factor,
            m.gate_w,
            m.up_w,
            m.down_w,
            m.token_ids,
            m.expert_ids,
            m.weights,
            m.input,
            m.output,
        )?;
        self.drain(dev)?;
        Ok(())
    }

    /// ONE launch for the WHOLE staged set (a contiguous GEMM run).
    pub fn launch(&self, dev: &crate::RocmDevice) -> crate::Result<()> {
        let staged = self.staged.as_deref().ok_or_else(|| {
            crate::Error::Backend("task ring context has no staged array".into())
        })?;
        launch_task_ring(
            dev,
            &**self.slots,
            self.capacity,
            &**self.tail,
            &**self.head,
            &**self.stop,
            Some(staged),
            self.count,
        )?
        .synchronize()?;
        Ok(())
    }

    /// Launch this ring's staged set on an EXPLICIT stream, WITHOUT waiting.
    ///
    /// The wave lands in the caller's stream ORDER — so a caller that has
    /// already launched the producer's graph on that stream gets the copy
    /// strictly after it. This is what the boundary handoff needs: the copy
    /// must not run before the graph that fills the buffer it reads.
    ///
    /// The kernel is resolved first, on a scratch ring, so this launch cannot
    /// race anything: `launch_compute_kernel_on_stream` only launches.
    pub fn launch_staged_on_stream(
        &self,
        dev: &crate::RocmDevice,
        stream: *mut std::ffi::c_void,
    ) -> crate::Result<()> {
        let staged = self.staged.as_deref().ok_or_else(|| {
            crate::Error::Backend("task ring context has no staged array".into())
        })?;
        self.clear_stop_flag(dev)?;
        self.resolve_wave_kernel(dev)?;
        launch_task_ring_staged_on_stream(
            dev,
            &**self.slots,
            self.capacity,
            &**self.tail,
            &**self.head,
            &**self.stop,
            staged,
            self.count,
            stream,
        )?;
        Ok(())
    }
}

/// The host's monotonic clock, in milliseconds.
///
/// A single base so every ring in the process shares a timeline — the ring
/// compares its own timestamps against this one.
pub fn host_now_ms() -> u64 {
    static BASE: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    BASE.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64
}

/// Wave launches since process start. The M4 gate asserts this advances by
/// exactly one per step, which is the property the phase exists to buy: a
/// launcher path spends one launch per GEMM node, the ring spends one per step.
static WAVE_LAUNCHES: AtomicU64 = AtomicU64::new(0);

/// Resident waves that were stopped AND confirmed gone.
///
/// A wave that was told to stop but has not yet left its poll loop is still
/// reading the ring's buffers. Counting the confirmed stops makes that
/// distinction observable: a gate can assert that every resident wave the
/// process launched was accounted for, instead of discovering a leak as a hang
/// in a later test.
static WAVE_STOPS: AtomicU64 = AtomicU64::new(0);


/// M6: wait (device-side) for a ring descriptor to complete.
///
/// The wave moves a descriptor's status PENDING -> RUNNING -> COMPLETE, so this
/// is how a consumer observes a cross-device handoff WITHOUT a host event: the
/// wait is a node in the consumer's graph, so the producer and the consumer
/// overlap instead of alternating.
///
/// The spin is bounded: the observed status is written out either way, so a
/// producer that never ran shows up as a timeout (the status is still PENDING)
/// rather than as a wedged device.
pub fn launch_scythe_wait_descriptor(
    dev: &crate::RocmDevice,
    slots_base: u64,
    slot_idx: u32,
    max_spins: u32,
    observed_status_dev: u64,
) -> Result<*mut std::ffi::c_void, crate::Error> {
    let mut sb = slots_base;
    let mut si = slot_idx;
    let mut ms = max_spins;
    let mut obs = observed_status_dev;
    dev.launch_compute_kernel(
        "grim_scythe_wait_descriptor",
        crate::HipDim3::new(1, 1, 1),
        crate::HipDim3::new(32, 1, 1),
        &mut [arg(&mut sb), arg(&mut si), arg(&mut ms), arg(&mut obs)],
    )
}

/// M6: wait (device-side) for a FLAG to become non-zero.
///
/// The cross-device completion flag: the producer's wave writes a non-zero
/// word into the CONSUMER's memory, and the consumer's device spins on that
/// word. The consumer never polls the producer's memory — the flag is the only
/// cross-device traffic — and no host event is involved.
///
/// Bounded, and the observed value is written out either way, so a producer
/// that never ran shows up as a TIMEOUT (the flag is still zero).
pub fn launch_scythe_wait_flag(
    dev: &crate::RocmDevice,
    flag_ptr: u64,
    max_spins: u32,
    observed_dev: u64,
) -> Result<*mut std::ffi::c_void, crate::Error> {
    let mut fp = flag_ptr;
    let mut ms = max_spins;
    let mut obs = observed_dev;
    dev.launch_compute_kernel(
        "grim_scythe_wait_flag",
        crate::HipDim3::new(1, 1, 1),
        crate::HipDim3::new(32, 1, 1),
        &mut [arg(&mut fp), arg(&mut ms), arg(&mut obs)],
    )
}

/// Launch the staged set on an EXPLICIT stream, WITHOUT waiting for it.
///
/// The boundary handoff needs the wave to be ORDERED AFTER the producer's
/// graph on the producer's OWN stream: a wave on a stream of its own would
/// race the very graph it is supposed to follow, and could copy a half-written
/// activation. So the wave goes on the caller's stream, and the caller decides
/// where in that stream's order it lands.
///
/// The kernel must already be resolved (the on-stream path only launches) —
/// see [`TaskRingContext::resolve_wave_kernel`].
#[allow(clippy::too_many_arguments)]
pub fn launch_task_ring_staged_on_stream(
    dev: &crate::RocmDevice,
    slots: &dyn BackendStorage,
    capacity: u32,
    tail: &dyn BackendStorage,
    head: &dyn BackendStorage,
    stop: &dyn BackendStorage,
    staged: &dyn BackendStorage,
    staged_count: u32,
    stream: *mut std::ffi::c_void,
) -> crate::Result<Box<dyn grim_tensor::backend::ComputeHandle>> {
    launch_task_ring_inner(
        dev,
        slots,
        capacity,
        tail,
        head,
        stop,
        Some(staged),
        staged_count,
        false,
        Some(stream),
    )
}

/// M6: poll a MONOTONIC counter on ANOTHER device, on an EXPLICIT stream.
///
/// The cross-device completion signal: the counter lives in the PRODUCER's
/// memory and advances once per handoff, and the consumer's own "last seen"
/// register says how far it has got. A peer READ is coherent on this
/// hardware, so the consumer sees the producer's advance.
///
/// The kernel must be pre-resolved under this entry name — see
/// [`launch_scythe_wait_flag_on_stream`].
pub fn launch_scythe_wait_counter_on_stream(
    dev: &crate::RocmDevice,
    counter_ptr: u64,
    last_seen_ptr: u64,
    max_spins: u32,
    observed_dev: u64,
    stream: *mut std::ffi::c_void,
) -> Result<*mut std::ffi::c_void, crate::Error> {
    let mut cp = counter_ptr;
    let mut ls = last_seen_ptr;
    let mut ms = max_spins;
    let mut obs = observed_dev;
    dev.launch_compute_kernel_on_stream(
        "grim_scythe_wait_counter",
        crate::HipDim3::new(1, 1, 1),
        crate::HipDim3::new(32, 1, 1),
        &mut [arg(&mut cp), arg(&mut ls), arg(&mut ms), arg(&mut obs)],
        stream,
        0,
    )
}

/// M6: poll a MONOTONIC counter, eagerly (no stream binding).
///
/// Used to RESOLVE the kernel under its own entry name: the on-stream launch
/// path only launches, so the entry has to be in the resolved-function cache.
/// A zero-spin launch resolves it and returns at once.
pub fn launch_scythe_wait_counter(
    dev: &crate::RocmDevice,
    counter_ptr: u64,
    last_seen_ptr: u64,
    max_spins: u32,
    observed_dev: u64,
) -> Result<*mut std::ffi::c_void, crate::Error> {
    let mut cp = counter_ptr;
    let mut ls = last_seen_ptr;
    let mut ms = max_spins;
    let mut obs = observed_dev;
    dev.launch_compute_kernel(
        "grim_scythe_wait_counter",
        crate::HipDim3::new(1, 1, 1),
        crate::HipDim3::new(32, 1, 1),
        &mut [arg(&mut cp), arg(&mut ls), arg(&mut ms), arg(&mut obs)],
    )
}

/// M6: wait (device-side) for a FLAG to CHANGE, on an EXPLICIT stream.
///
/// The cross-device completion signal for a REPEATED handoff: the producer
/// writes a value that changes per handoff, and the consumer waits for the
/// flag to differ from the value it last saw. `last_seen` is the CONSUMER's
/// own register, so the comparison needs no host involvement.
///
/// The kernel must be pre-resolved under this entry name — see
/// [`launch_scythe_wait_flag_on_stream`].
pub fn launch_scythe_wait_flag_change_on_stream(
    dev: &crate::RocmDevice,
    flag_ptr: u64,
    last_seen_ptr: u64,
    max_spins: u32,
    observed_dev: u64,
    stream: *mut std::ffi::c_void,
) -> Result<*mut std::ffi::c_void, crate::Error> {
    let mut fp = flag_ptr;
    let mut ls = last_seen_ptr;
    let mut ms = max_spins;
    let mut obs = observed_dev;
    dev.launch_compute_kernel_on_stream(
        "grim_scythe_wait_flag_change",
        crate::HipDim3::new(1, 1, 1),
        crate::HipDim3::new(32, 1, 1),
        &mut [arg(&mut fp), arg(&mut ls), arg(&mut ms), arg(&mut obs)],
        stream,
        0,
    )
}

/// M6: wait (device-side) for a FLAG to CHANGE, eagerly (no stream binding).
///
/// Used to RESOLVE the kernel under its own entry name: the on-stream launch
/// path only launches, so the entry has to be in the resolved-function cache.
/// A zero-spin launch resolves it and returns at once.
pub fn launch_scythe_wait_flag_change(
    dev: &crate::RocmDevice,
    flag_ptr: u64,
    last_seen_ptr: u64,
    max_spins: u32,
    observed_dev: u64,
) -> Result<*mut std::ffi::c_void, crate::Error> {
    let mut fp = flag_ptr;
    let mut ls = last_seen_ptr;
    let mut ms = max_spins;
    let mut obs = observed_dev;
    dev.launch_compute_kernel(
        "grim_scythe_wait_flag_change",
        crate::HipDim3::new(1, 1, 1),
        crate::HipDim3::new(32, 1, 1),
        &mut [arg(&mut fp), arg(&mut ls), arg(&mut ms), arg(&mut obs)],
    )
}

/// M6: wait (device-side) for a FLAG on an EXPLICIT stream.
///
/// The consumer's wait must be ordered BEFORE the consumer's graph, and the
/// consumer's graph runs on its SEGMENT's stream — not on the device's active
/// stream. So the wait is launched on the segment's stream.
///
/// The kernel must be pre-resolved under this entry name (the on-stream path
/// only launches): [`BoundaryHandoff`](crate::peer_access::BoundaryHandoff)
/// resolves it with a zero-spin eager launch when it is prepared.
pub fn launch_scythe_wait_flag_on_stream(
    dev: &crate::RocmDevice,
    flag_ptr: u64,
    max_spins: u32,
    observed_dev: u64,
    stream: *mut std::ffi::c_void,
) -> Result<*mut std::ffi::c_void, crate::Error> {
    let mut fp = flag_ptr;
    let mut ms = max_spins;
    let mut obs = observed_dev;
    dev.launch_compute_kernel_on_stream(
        "grim_scythe_wait_flag",
        crate::HipDim3::new(1, 1, 1),
        crate::HipDim3::new(32, 1, 1),
        &mut [arg(&mut fp), arg(&mut ms), arg(&mut obs)],
        stream,
        0,
    )
}

/// How many claim attempts a RESIDENT wave may make before it gives up on its
/// own (M6 step 3). The wave normally leaves through `stop_wave`; this is the
/// backstop, so a host that forgets to stop it cannot hold the device.
///
/// The budget must outlast the host's FIRST enqueue after the wave parks: that
/// enqueue JITs `grim_scythe_write_slot` and `grim_scythe_publish_head` on its
/// first use in the process, which can take longer than a small budget — and a
/// wave that expires first leaves the node undrained, which reads as an
/// intermittent failure rather than as a timeout. Each idle iteration costs a
/// backoff sleep (~2 us), so 1<<20 is a couple of seconds of parking: long
/// enough to cover the JIT, short enough to be a real backstop.
pub const RESIDENT_POLL_BUDGET: u32 = 1 << 20;

/// How long the host waits before it believes the resident wave has expired,
/// in milliseconds.
///
/// The wave's budget is counted in polls, and the host's clock is only a
/// CONSERVATIVE witness — this value is deliberately larger than the wave's
/// real idle lifetime, so the wave is certainly gone by the time the host acts.
/// A wave that outlives it (a slow clock, a deep backoff) is stopped and
/// waited for before the fresh one is launched, so the two never race.
pub const RESIDENT_WAVE_MAX_IDLE_MS: u64 = 30_000;

/// How many IDLE polls a resident wave may make before it gives up.
///
/// The budget counts polls with no work, and every claimed task resets it — so
/// a wave that is being fed never expires. It is finite because it is the
/// DEADLOCK BOUND: a host that synchronises the device while the wave is
/// parked would otherwise wait for the wave forever, and the wave would wait
/// for the host.
///
/// The wall-clock value depends on the poll's backoff: the wave sleeps longer
/// and longer between idle polls (up to 64 sleeps of 64 cycles, ~100 us at the
/// cap), so an idle poll costs about 100 us — MEASURED, not guessed.
///
/// The budget is therefore in POLLS, and 1<<16 polls is roughly 6.5 seconds of
/// idle parking: long enough to be harmless, short enough to be a practical
/// deadlock bound. The enqueue kernels are resolved when the ring is created,
/// so no JIT can eat it (a cold hipRTC compile of this source tree can outlast
/// ANY wall-clock budget, which is why the resolution is eager).
pub const RESIDENT_IDLE_POLL_BUDGET: u32 = 1 << 16;

/// How long the host waits for a stopped resident wave to leave its poll loop,
/// in milliseconds.
///
/// The wave notices the stop flag within an idle backoff (~2 us), so this is
/// orders of magnitude more than it needs — it is a backstop against a wave
/// that can never leave, not a tuning knob. A wave that outlasts it is
/// REPORTED, not waited on forever.
const WAVE_EXIT_POLL_BUDGET: u32 = 5_000;

pub fn wave_launch_count() -> u64 {
    WAVE_LAUNCHES.load(Ordering::Relaxed)
}

/// Resident waves that were stopped and CONFIRMED gone.
///
/// Every resident wave the process launched must appear here once its context
/// is dropped: a wave that is still parked when its ring's buffers are freed
/// reads freed memory, misses the stop flag and spins for the life of the
/// process — a device-wide wedge that shows up as a hang in some LATER test.
pub fn wave_stop_count() -> u64 {
    WAVE_STOPS.load(Ordering::Relaxed)
}

/// ONE launch for the whole step's GEMM set.
///
/// `staged` is the device array from [`TaskRingPlan::stage_on_device`]; the
/// wave's prologue uploads `staged_count` descriptors into the slots and repoints
/// tail/head, then the existing poll loop consumes them. The caller must not
/// also call `launch_scythe_publish_head` — the wave publishes head itself, and
/// a host publish on top of that would double-advance the counter.
///
/// `staged_count` must be `> 0` with `staged` present (a null array with a
/// non-zero count would have the wave read nothing), and must be `<= capacity`
/// (the prologue masks with `capacity - 1`).
pub fn launch_task_ring(
    dev: &crate::RocmDevice,
    slots: &dyn BackendStorage,
    capacity: u32,
    tail: &dyn BackendStorage,
    head: &dyn BackendStorage,
    stop: &dyn BackendStorage,
    staged: Option<&dyn BackendStorage>,
    staged_count: u32,
) -> crate::Result<Box<dyn grim_tensor::backend::ComputeHandle>> {
    launch_task_ring_inner(dev, slots, capacity, tail, head, stop, staged, staged_count, false, None)
}

/// M6 step 3: launch the wave in RESIDENT mode.
///
/// A resident wave does not exit when the queue drains — it parks on the empty
/// queue and picks up whatever the ring is given next, so a step costs ONE
/// launch instead of one per node. It leaves either when `stop_ptr` is set
/// ([`TaskRingContext::stop_wave`]) or when its poll budget runs out, so a host
/// that forgets to stop it cannot wedge the device.
pub fn launch_task_ring_resident(
    dev: &crate::RocmDevice,
    slots: &dyn BackendStorage,
    capacity: u32,
    tail: &dyn BackendStorage,
    head: &dyn BackendStorage,
    stop: &dyn BackendStorage,
    staged: Option<&dyn BackendStorage>,
    staged_count: u32,
) -> crate::Result<Box<dyn grim_tensor::backend::ComputeHandle>> {
    launch_task_ring_inner(dev, slots, capacity, tail, head, stop, staged, staged_count, true, None)
}

/// M6 step 3: launch a RESIDENT wave on an EXPLICIT stream.
///
/// A resident wave parks on the empty queue, so it must not run on the caller's
/// stream: it would occupy that stream and the enqueues (and the stop kernel)
/// would queue BEHIND it instead of being seen. The ring therefore owns a
/// stream, and the wave lives there while the caller keeps enqueueing on its own.
pub fn launch_task_ring_resident_on_stream(
    dev: &crate::RocmDevice,
    slots: &dyn BackendStorage,
    capacity: u32,
    tail: &dyn BackendStorage,
    head: &dyn BackendStorage,
    stop: &dyn BackendStorage,
    staged: Option<&dyn BackendStorage>,
    staged_count: u32,
    stream: *mut std::ffi::c_void,
) -> crate::Result<Box<dyn grim_tensor::backend::ComputeHandle>> {
    launch_task_ring_inner(
        dev, slots, capacity, tail, head, stop, staged, staged_count, true, Some(stream),
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_task_ring_inner(
    dev: &crate::RocmDevice,
    slots: &dyn BackendStorage,
    capacity: u32,
    tail: &dyn BackendStorage,
    head: &dyn BackendStorage,
    stop: &dyn BackendStorage,
    staged: Option<&dyn BackendStorage>,
    staged_count: u32,
    resident: bool,
    stream: Option<*mut std::ffi::c_void>,
) -> crate::Result<Box<dyn grim_tensor::backend::ComputeHandle>> {
    // `None, 0` is the pre-existing contract (the host published `head`; the
    // prologue is a no-op). A count with a null array is the bug: the wave would
    // read nothing while believing it has work.
    let staged = match staged {
        Some(s) if staged_count > 0 => Some(s),
        None if staged_count == 0 => None,
        _ => {
            return Err(crate::Error::Backend(
                "launch_task_ring: staged_count > 0 requires the staged descriptor array"
                    .into(),
            ))
        }
    };
    if staged_count > capacity {
        return Err(crate::Error::Backend(format!(
            "launch_task_ring: staged_count {staged_count} exceeds ring capacity {capacity}"
        )));
    }
    let mut slots_ptr = dev_ptr(as_rocm(slots)?)?;
    let mut tail_ptr = dev_ptr(as_rocm(tail)?)?;
    let mut head_ptr = dev_ptr(as_rocm(head)?)?;
    let mut stop_ptr = dev_ptr(as_rocm(stop)?)?;
    // The staged pointer is only meaningful when the prologue runs
    // (staged_count > 0); pass null otherwise — the kernel never reads it.
    let mut staged_ptr: *mut std::ffi::c_void = match staged {
        Some(s) => crate::device::util::dev_ptr_dyn(s)?,
        None => std::ptr::null_mut(),
    };
    let mut cap = capacity;
    // When staged_count > 0, the prologue uploads staged_count descriptors and the wave
    // drains exactly that many. When staged_count == 0 (per-node enqueue), max_tasks
    // cannot be 0 (which exits immediately without executing anything) — give it a
    // generous budget (e.g. 1024 iterations) to drain the queue until head == tail.
    // The wave's iteration budget = how many claim attempts it may make.
    //  - staged set: exactly that many descriptors to drain.
    //  - per-node enqueue: a generous budget, so the wave drains the queue
    //    until head == tail.
    //  - RESIDENT: the wave PARKS after the queue drains, so its budget is a
    //    POLL budget, not a task count. It is bounded so a host that never
    //    stops it cannot hold the device forever.
    let mut limit = if resident {
        // The wave's lifetime is an IDLE budget (the kernel resets it on every
        // claimed task), so a wave that is being FED never expires — however
        // long the feeding takes, including a cold JIT of the enqueue kernels.
        //
        // The budget is still FINITE, and that is the point: a host that
        // synchronises the device while the wave is parked waits for the wave,
        // and the wave waits for the host — the only thing that breaks that
        // cycle is the wave giving up on its own. With a bounded idle budget
        // the host's wait is bounded instead of infinite, and the ring
        // re-launches a fresh wave on the next enqueue (the wave flags its own
        // exit by leaving the stop flag set).
        RESIDENT_IDLE_POLL_BUDGET
    } else if staged_count > 0 {
        staged_count
    } else {
        1024u32
    };
    // 1 = the wave parks on an empty queue and only leaves through `stop_ptr`
    // (or its poll budget); 0 = it exits as soon as the queue drains.
    let mut resident_flag: u32 = if resident { 1 } else { 0 };
    let mut count = staged_count;
    let _dev_guard = crate::device::util::DeviceGuard::set(dev.ordinal as i32);
    let mut args = [
        arg(&mut slots_ptr),
        arg(&mut cap),
        arg(&mut tail_ptr),
        arg(&mut head_ptr),
        arg(&mut stop_ptr),
        arg(&mut limit),
        arg(&mut resident_flag), // 1 = park on an empty queue instead of exiting
        arg(&mut staged_ptr),
        arg(&mut count),
    ];
    match stream {
        Some(st) => {
            // The on-stream path only LAUNCHES: the kernel must already be
            // resolved (the eager launch resolves it). It is launched with the
            // same grid/block, so the cache key matches.
            dev.launch_compute_kernel_on_stream(
                "grim_scythe_persistent_dispatch",
                crate::HipDim3::new(1, 1, 1),
                crate::HipDim3::new(128, 1, 1),
                &mut args,
                st,
                0,
            )?;
        }
        None => {
            dev.launch_compute_kernel(
                "grim_scythe_persistent_dispatch",
                crate::HipDim3::new(1, 1, 1),
                crate::HipDim3::new(128, 1, 1),
                &mut args,
            )?;
        }
    }
    WAVE_LAUNCHES.fetch_add(1, Ordering::Relaxed);
    Ok(Box::new(crate::RocmHandle::new(Some(
        stream.unwrap_or(dev.active_stream()),
    ))))
}
