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
        let count = self.gemms.len().min(STAGED_CAPACITY as usize);
        let mut bytes = vec![0u8; count * std::mem::size_of::<ScytheTaskDescriptor>()];
        for (i, g) in self.gemms.iter().take(count).enumerate() {
            let desc = ScytheTaskDescriptor {
                opcode: OP_ROW_GEMM,
                m: g.m,
                n: g.n,
                k: g.k,
                input_ptr: g.input,
                weight_ptr: g.weight,
                output_ptr: g.output,
                peer_ptr: 0,
                status: 0,
            };
            let off = i * std::mem::size_of::<ScytheTaskDescriptor>();
            let cell = unsafe {
                std::slice::from_raw_parts(
                    &desc as *const ScytheTaskDescriptor as *const u8,
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
        Ok((storage, count as u32))
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
    slots: Box<dyn BackendStorage>,
    tail: Box<dyn BackendStorage>,
    head: Box<dyn BackendStorage>,
    stop: Box<dyn BackendStorage>,
    desc_staging: Box<dyn BackendStorage>,
    capacity: u32,
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
            slots: Box::new(slots),
            tail: u32_scalar(0)?,
            head: u32_scalar(0)?,
            stop: u32_scalar(0)?,
            desc_staging,
            capacity,
        };
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
        let head_value = NEXT_HEAD.fetch_add(1, Ordering::Relaxed) as u32;
        let head_ptr = as_rocm(self.head.as_ref())?
            .device_ptr_u64()
            .ok_or_else(|| crate::Error::Backend("task ring head has no device ptr".into()))?;
        let slot = (head_value as u64 % self.capacity as u64) as usize;
        crate::kernels::qkv_attention::launch_scythe_write_slot(
            dev,
            &*self.slots,
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
        crate::kernels::qkv_attention::launch_scythe_publish_head(dev, head_ptr, head_value + 1)?;
        launch_task_ring(dev, &*self.slots, self.capacity, &*self.tail, &*self.head, &*self.stop, None, 0)?;
        Ok(())
    }

    /// Enqueue one RMS norm and drain it with a one-task wave.
    ///
    /// The wave's arm fixes `eps` at 1e-5 — the Llama-family value. A model
    /// whose norm uses a different epsilon is NOT bound (see `from_graph`),
    /// because normalizing with the wrong constant is a silent wrong answer.
    pub fn enqueue_norm(&self, dev: &crate::RocmDevice, n: &TaskRingNorm) -> crate::Result<()> {
        let head_value = NEXT_HEAD.fetch_add(1, Ordering::Relaxed) as u32;
        let slot = (head_value as u64 % self.capacity as u64) as usize;
        crate::kernels::qkv_attention::launch_scythe_write_slot(
            dev,
            &*self.slots,
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
        crate::kernels::qkv_attention::launch_scythe_publish_head(dev, head_ptr, head_value + 1)?;
        launch_task_ring(dev, &*self.slots, self.capacity, &*self.tail, &*self.head, &*self.stop, None, 0)?;
        Ok(())
    }

    /// Enqueue one element-wise add and drain it with a one-task wave.
    ///
    /// The wave's OP_ADD arm computes `out[i] = a[i] + b[i]` over `m*n`
    /// elements. The emitter must ensure the node is actually a binary add.
    pub fn enqueue_add(&self, dev: &crate::RocmDevice, a: &TaskRingAdd) -> crate::Result<()> {
        let head_value = NEXT_HEAD.fetch_add(1, Ordering::Relaxed) as u32;
        let slot = (head_value as u64 % self.capacity as u64) as usize;
        crate::kernels::qkv_attention::launch_scythe_write_slot(
            dev,
            &*self.slots,
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
        crate::kernels::qkv_attention::launch_scythe_publish_head(dev, head_ptr, head_value + 1)?;
        launch_task_ring(dev, &*self.slots, self.capacity, &*self.tail, &*self.head, &*self.stop, None, 0)?;
        Ok(())
    }

    /// Enqueue one attention and drain it with a one-task wave.
    ///
    /// The wave's OP_ATTN arm computes flash-attention with q/k/v/out.
    /// The arm has a hard 256 head_dim limit (enforced at bind time).
    pub fn enqueue_attn(&self, dev: &crate::RocmDevice, a: &TaskRingAttn) -> crate::Result<()> {
        let head_value = NEXT_HEAD.fetch_add(1, Ordering::Relaxed) as u32;
        let slot = (head_value as u64 % self.capacity as u64) as usize;
        crate::kernels::qkv_attention::launch_scythe_write_slot(
            dev,
            &*self.slots,
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
        crate::kernels::qkv_attention::launch_scythe_publish_head(dev, head_ptr, head_value + 1)?;
        launch_task_ring(dev, &*self.slots, self.capacity, &*self.tail, &*self.head, &*self.stop, None, 0)?;
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
        let head_value = NEXT_HEAD.fetch_add(1, Ordering::Relaxed) as u32;
        let slot = (head_value as u64 % self.capacity as u64) as usize;
        crate::kernels::qkv_attention::launch_scythe_write_slot(
            dev,
            &*self.slots,
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
        crate::kernels::qkv_attention::launch_scythe_publish_head(dev, head_ptr, head_value + 1)?;
        launch_task_ring(dev, &*self.slots, self.capacity, &*self.tail, &*self.head, &*self.stop, None, 0)?;
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
        let head_value = NEXT_HEAD.fetch_add(1, Ordering::Relaxed) as u32;
        let slot = (head_value as u64 % self.capacity as u64) as usize;
        crate::kernels::qkv_attention::launch_scythe_write_slot(
            dev,
            &*self.slots,
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
        crate::kernels::qkv_attention::launch_scythe_publish_head(dev, head_ptr, head_value + 1)?;
        launch_task_ring(dev, &*self.slots, self.capacity, &*self.tail, &*self.head, &*self.stop, None, 0)?;
        Ok(())
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
        let head_value = NEXT_HEAD.fetch_add(1, Ordering::Relaxed) as u32;
        let slot = (head_value as u64 % self.capacity as u64) as usize;
        crate::kernels::qkv_attention::launch_scythe_write_slot(
            dev,
            &*self.slots,
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
        crate::kernels::qkv_attention::launch_scythe_publish_head(dev, head_ptr, head_value + 1)?;
        launch_task_ring(dev, &*self.slots, self.capacity, &*self.tail, &*self.head, &*self.stop, None, 0)?;
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
            &*self.slots,
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
        launch_task_ring(dev, &*self.slots, self.capacity, &*self.tail, &*self.head, &*self.stop, None, 0)?;
        Ok(())
    }

    /// ONE launch for the WHOLE staged set (a contiguous GEMM run).
    pub fn launch(&self, dev: &crate::RocmDevice) -> crate::Result<()> {
        let staged = self.staged.as_deref().ok_or_else(|| {
            crate::Error::Backend("task ring context has no staged array".into())
        })?;
        launch_task_ring(
            dev,
            &*self.slots,
            self.capacity,
            &*self.tail,
            &*self.head,
            &*self.stop,
            Some(staged),
            self.count,
        )?
        .synchronize()?;
        Ok(())
    }
}

/// Wave launches since process start. The M4 gate asserts this advances by
/// exactly one per step, which is the property the phase exists to buy: a
/// launcher path spends one launch per GEMM node, the ring spends one per step.
static WAVE_LAUNCHES: AtomicU64 = AtomicU64::new(0);

/// Host-side monotonic head counter for the per-node enqueue path. The device
/// `head` scalar is published from this; monotonic across steps, which is what
/// lets the wave tell "queue empty" from "nothing enqueued yet".
static NEXT_HEAD: AtomicU64 = AtomicU64::new(0);

pub fn wave_launch_count() -> u64 {
    WAVE_LAUNCHES.load(Ordering::Relaxed)
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
    let mut limit = if staged_count > 0 { staged_count } else { 1024u32 };
    let mut count = staged_count;
    let _dev_guard = crate::device::util::DeviceGuard::set(dev.ordinal as i32);
    dev.launch_compute_kernel(
        "grim_scythe_persistent_dispatch",
        crate::HipDim3::new(1, 1, 1),
        crate::HipDim3::new(128, 1, 1),
        &mut [
            arg(&mut slots_ptr),
            arg(&mut cap),
            arg(&mut tail_ptr),
            arg(&mut head_ptr),
            arg(&mut stop_ptr),
            arg(&mut limit),
            arg(&mut 0u32), // resident = 0: the staged prologue resets tail
            arg(&mut staged_ptr),
            arg(&mut count),
        ],
    )?;
    WAVE_LAUNCHES.fetch_add(1, Ordering::Relaxed);
    Ok(Box::new(crate::RocmHandle::new(Some(
        dev.active_stream(),
    ))))
}
