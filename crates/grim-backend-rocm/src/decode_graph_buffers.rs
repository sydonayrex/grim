//! i-was-dumb-graph.md Phase 1-2 + Phase 6 scalar update + P3 batch parameterization.
//!
//! Fixed device buffer pool at stable addresses + HIP graph lifecycle.
//! Decode forward is deterministic: same kernels, same topology each step,
//! only input data changes. Pre-alloc once, capture once, replay via single
//! `hipGraphLaunch`.
//!
//! No-sync rule inside capture: no D2H, no `hipDeviceSynchronize`, no host
//! alloc. All writes via async H2D or kernel launches. GEMM stays in
//! rocBLAS per Rule 0 — this module owns buffers + graph, not math.

use crate::task_graph::TaskGraph as GrimTaskGraph;
use std::ffi::c_void;
use std::sync::Arc;

use grim_tensor::BackendStorage;

use crate::DTypeStorage;
use crate::HipMemcpyKind;
use crate::device::handles::hipEventCreateWithFlags;
use crate::device::roc_device::RocmDevice;
use crate::device::util::dtype_f32;
use crate::kernels::qkv_attention::KvRingPolicy;
use crate::memory::storage::RocmStorage;
use crate::{
    Shape, arg, check_hip, hipGraphDestroy, hipGraphExecDestroy, hipGraphExecKernelNodeSetParams,
    hipGraphInstantiate, hipMemcpyAsync, hipStreamBeginCapture, hipStreamEndCapture, linear_launch,
};
use grim_tensor::error::{Error, Result};
use grim_tensor::{ArithType, DType};

// Capture-scoped padded-fp8 act scratch for the WhiteRaven blocked leg.
//
// Installed by `begin_capture` and cleared by `end_capture`/`abort_capture` so
// the pointer can never outlive the `DecodeGraphBuffers` that own it. It is
// thread-local because capture is single-threaded per device; a stray read
// outside a capture returns `None`, which the blocked leg treats as a hard
// error rather than quietly falling back to the host conversion.
thread_local! {
    static CAPTURE_FP8_PAD: std::cell::Cell<*const RocmStorage> =
        const { std::cell::Cell::new(std::ptr::null()) };
}

fn install_capture_fp8_pad(scratch: Option<&RocmStorage>) {
    CAPTURE_FP8_PAD.with(|c| c.set(scratch.map_or(std::ptr::null(), |s| s as *const _)));
}

/// The padded-fp8 act scratch published for the open capture, if any.
pub fn capture_fp8_pad_scratch() -> Option<&'static RocmStorage> {
    CAPTURE_FP8_PAD.with(|c| {
        let p = c.get();
        if p.is_null() {
            None
        } else {
            // SAFETY: installed by begin_capture and cleared by
            // end_capture/abort_capture on the same thread, so the pointee
            // (an element of the live DecodeGraphBuffers) outlives every read.
            Some(unsafe { &*p })
        }
    })
}

/// `DecodeGraphBuffers::kv_arena_bits` marker for the WhiteCrow (u4
/// group-128 OSTQuant) arena.
///
/// WhiteCrow stores 4-bit codes, but `Some(4)` is taken by the int4 family
/// (two codes per byte, `head_dim/2` byte strides, f32 scale sidecar) and the
/// capture dispatch selects kernels from this value — so WhiteCrow records
/// `67`, the BYTES of one group's block. 67 is the one width no other format
/// can claim (int8 rows are `nkv*hd`, int4 `nkv*hd/2`, int2v4 `nkv*hd/4` at
/// these geometries), and every consumer needs the block geometry anyway:
/// a slot's byte extent is `max_ctx * nkv * (head_dim/128) * 67`.
pub const KV_ARENA_BITS_WHITECROW: u32 = 67;

/// Spec §Phase 1: owns all decode intermediates at fixed addresses.
/// Allocated once at model load, never freed until unload.
/// `batch` parameterizes all per-step slots as `[batch, dim]` so the same
/// captured graph replays a whole decode batch in one `hipGraphLaunch`.
#[derive(Debug)]
pub struct DecodeGraphBuffers {
    /// Per-layer buffers (indexed by layer_idx)
    pub layer_input: Vec<RocmStorage>, // [batch, hidden_size]
    /// One handoff buffer per device transition, on the PRODUCING device
    /// (empty for a single-device placement). Task A3.
    pub boundary_out: Vec<RocmStorage>,
    pub layer_output: Vec<RocmStorage>, // [batch, hidden_size]
    /// Attention-specific
    pub q_buf: Vec<RocmStorage>, // [batch, n_q]
    pub k_buf: Vec<RocmStorage>,        // [batch, n_k]
    pub v_buf: Vec<RocmStorage>,        // [batch, n_v]
    /// The two halves of the fused attention Q|gate projection, split.
    ///
    /// `q_buf` holds `[batch, 2*q_dim]` — the fused Q|gate — because that is
    /// what the projection produces. But the per-head Q norm and the RoPE both
    /// operate on Q ALONE at `[batch, q_dim]`, and the gate is needed separately
    /// to multiply the attention output. Running either of them over the whole
    /// fused buffer is a shape mismatch (and would norm the gate half too).
    ///
    /// llama.cpp splits with a view: `gate = ggml_view_3d(Qcur_full, ...)`
    /// (qwen35.cpp:289-293) and norms/ropes `Qcur` on its own. vLLM keeps
    /// `in_proj_qkv` and `in_proj_z` as separate parameters
    /// (qwen_gdn_linear_attn.py:1033-1038). The graph has neither, so it gets
    /// two real buffers and one copy each.
    pub q_head_buf: Vec<RocmStorage>, // [batch, n_attn_out]
    /// The gate half, consumed by `sigmoid` before `wo`.
    pub q_gate_buf: Vec<RocmStorage>, // [batch, n_attn_out]
    /// Branch output before the output projection.
    ///
    /// `[batch, n_attn_out]` — and `n_attn_out` is the ATTENTION branch width
    /// (`q_dim`), NOT the maxed `n_q` that `q_buf` uses. The two are different
    /// quantities and conflating them is what broke capture:
    ///
    ///   * `q_buf` holds a PROJECTION OUTPUT, so it must fit whichever layer
    ///     type is running: `conv_dim` for recurrent (the fused qkv), `2*q_dim`
    ///     for attention (the fused Q|gate). Hence the max.
    ///   * `attn_out_buf` holds a BRANCH OUTPUT, consumed by `wo` (attention,
    ///     `q_dim` wide) or by `ssm_out` (recurrent, `value_dim` wide). It is
    ///     NOT a projection output and must never be maxed.
    ///
    /// Recurrent layers write their branch to `KdaLayerBuffers::branch` at
    /// `value_dim`; this buffer is then only read by attention layers.
    ///
    /// Both references agree and neither maxes: llama.cpp returns `q_dim` from
    /// `build_layer_attn` (`qwen35.cpp:320-334`, gated then `wo`) and
    /// `value_dim` from `build_layer_attn_linear` (`:462`); vLLM splits
    /// `[q,k,v,z]` with `z_size = value_dim` and feeds `_output_projection`
    /// `core_attn_out` of shape `[tokens, num_v_heads, head_v_dim]`
    /// (`qwen_gdn_linear_attn.py:1033-1049`).
    pub attn_out_buf: Vec<RocmStorage>, // [batch, n_q]
    /// FFN-specific
    pub gate_up_buf: Vec<RocmStorage>, // [batch, 2*intermediate_size] (reserved: fused gate+up path)
    pub gate_buf: Vec<RocmStorage>,      // [batch, intermediate_size]
    pub up_buf: Vec<RocmStorage>,        // [batch, intermediate_size]
    pub activated_buf: Vec<RocmStorage>, // [batch, intermediate_size]
    /// Per-layer [batch, hidden] staging for norm outputs and GEMM results that
    /// feed a residual add. Sequentially reused within a layer (stream order
    /// preserves dependencies); never live across layers.
    pub norm_buf: Vec<RocmStorage>, // [batch, hidden_size]
    /// Online-softmax partials for `launch_qkv_attention_dev`, kept in-pool
    /// so capture allocates nothing.
    pub attn_max_buf: Vec<RocmStorage>, // [num_heads]
    pub attn_sum_buf: Vec<RocmStorage>,  // [num_heads]
    /// KV cache arenas (pre-allocated to max context)
    pub k_arena: Vec<RocmStorage>, // [max_ctx, n_k]
    pub v_arena: Vec<RocmStorage>,       // [max_ctx, n_v]
    /// Phase-2 int8 KV scale sidecars: one f32 per (row, kv_head), laid out
    /// `[batch * max_ctx, n_kv_heads]` to mirror the arena slotting. Empty
    /// unless the arenas hold u8 codes; every int8 launcher reads these, so
    /// a codes arena without its sidecar is a loud failure, not a reinterpret.
    pub k_scale_arena: Vec<RocmStorage>,
    pub v_scale_arena: Vec<RocmStorage>,
    /// Phase-3 eviction ring. `cap_rows <= 0` (the default) means the arena
    /// grows with the context and every ring kernel is an identity; otherwise
    /// the arena is a fixed capacity with `sink_rows` pinned attention sinks
    /// and the newest tokens in a ring. Owned here so capture and replay read
    /// the same policy the capture baked into the kernels.
    pub kv_ring: KvRingPolicy,
    /// Bits per KV code for the packed arenas this pool was built with: `None`
    /// for f16/f32, `Some(8)`/`Some(4)`/`Some(2)` for int8/int4/int2v4, and
    /// [`KV_ARENA_BITS_WHITECROW`] for WhiteCrow (4-bit codes in 67-byte
    /// group-128 blocks — recorded as the block width so it cannot be
    /// confused with the int4 family). Set once
    /// at allocation from the resolved `KvArenaFormat`; the capture dispatch
    /// reads it to pick the right device-total kernel pair.
    ///
    /// Recorded rather than inferred: an 8-bit and a 4-bit arena differ only by
    /// a factor of two in the innermost shape axis (which is a head's BYTE
    /// count, not `head_dim`), so no storage-level predicate can tell them apart.
    /// [`Self::kv_arena_packed_width`] cross-checks it against the arena dtype.
    pub kv_arena_bits: Option<u32>,
    /// Latent KV cache arena for MLA models (DeepSeek, Kimi): [batch * max_ctx, latent_kv_dim]
    pub latent_kv_arena: Vec<RocmStorage>,
    /// Output projection
    pub head_input: RocmStorage, // [batch, hidden_size]
    pub head_output: Arc<RocmStorage>, // [batch, vocab_size]
    /// Host-side position mirror. Device scalar lives in `pos_dev`;
    /// kernels read pos from device so graph stays capturable.
    pub current_pos: u32,
    /// Single-u32 device buffer holding KV position. Updated before each
    /// replay via one 4-byte async H2D (spec §Scalar second approach).
    pub pos_dev: RocmStorage,
    /// A5b: one pos replica per FOREIGN device in the placement (empty for a
    /// single-device model — `pos_dev_for` falls back to `pos_dev`). Each
    /// replica is zeroed at alloc like `pos_dev`.
    pub pos_replicas: Vec<RocmStorage>,
    /// Device ordinal for each entry in `pos_replicas`.
    pub pos_replica_ords: Vec<usize>,
    /// A5b: read-only attn_dummy replica per foreign device (empty = fallback
    /// to `attn_dummy`).
    pub attn_dummy_replicas: Vec<RocmStorage>,
    pub attn_dummy_replica_ords: Vec<usize>,
    /// M1: the emitted task-graph IR, set by `end_capture` when the emitter is
    /// enabled (`GRIM_EMIT_TASK_GRAPH=1`). `None` before capture — which is
    /// what makes a missing emission a loud failure in M2 rather than an empty
    /// graph that compares equal.
    pub emitted_task_graph: Option<GrimTaskGraph>,
    /// D1-2a: per-layer KV total (live row count), i32-in-f32-slot like
    /// `pos_dev`. `pos_dev` keeps serving RoPE position and accumulates its
    /// per-layer bumps (A0-pinned); `layer_totals[i]` counts only layer i's
    /// appends, so the append row index and the attention live-row count are
    /// EXACT (today the shared scalar over-reads zeroed rows on models with
    /// >1 attention layer — masked by arena zeroing). Zeroed at alloc; the
    /// seeder writes the prefill count per layer (write_layer_totals).
    pub layer_totals: Vec<RocmStorage>,
    /// M2 (PLAN-kernel-fusion): MoE routing + output staging. Empty vecs when
    /// the model has no MoE layers. Routing triple is shared across layers —
    /// each layer writes then consumes it in stream order within its bracket.
    pub moe_gate_logits: Vec<RocmStorage>, // [batch, n_expert] per layer
    pub moe_out: Vec<RocmStorage>,      // [batch, hidden] per layer
    pub moe_route_tokens: RocmStorage,  // [batch*top_k] u32
    pub moe_route_experts: RocmStorage, // [batch*top_k] u32
    pub moe_route_weights: RocmStorage, // [batch*top_k] f32
    /// S2: ShortConv staging. Empty vecs when the model has no ShortConv
    /// layers. `sc_state[l]` is the device-resident ring `[h_dim*(kc)]`
    /// (channel-major [d, kc], the HIP kernel's in-place layout).
    pub sc_proj_buf: Vec<RocmStorage>, // [batch, 3*h_dim]
    pub sc_b: Vec<RocmStorage>,         // [batch*h_dim]
    pub sc_c: Vec<RocmStorage>,
    pub sc_x: Vec<RocmStorage>,
    pub sc_bx: Vec<RocmStorage>,
    pub sc_sum: Vec<RocmStorage>,
    pub sc_y: Vec<RocmStorage>, // [batch, h_dim]
    /// The short-conv's causal state, ZEROED at allocation: a fresh session
    /// has no history, and the caching allocator hands out recycled blocks
    /// that still hold the previous owner's values.
    pub sc_state: Vec<RocmStorage>, // [h_dim*(l_cache-1)] per layer
    /// Shared read-only dummy ([1], zeros) for unused attention inputs
    /// (o_proj weights with fuse_o=0, alibi slopes with has_alibi=0).
    pub attn_dummy: RocmStorage,
    /// Per-slot token-id staging ([batch] u32 in memory): replay writes fresh
    /// token ids here BEFORE launch; the graph's embedding-gather node reads
    /// them from device memory, so replayed logits track the input.
    pub token_ids_dev: RocmStorage,
    /// Per-layer Q8_1 activation staging (U8 bytes) for fused/quant GEMVs.
    /// Sized for max(hidden, intermediate): `(max/32)*36` bytes, reused
    /// sequentially across QKV and gate+up projections (stream-ordered).
    pub act_q81_buf: Vec<RocmStorage>,
    /// Per-layer padded fp8 activation staging for the WhiteRaven blocked
    /// WMMA GEMM. Sized `batch * 16 * max(hidden, intermediate)` u8 bytes so
    /// a one-row decode still forms a whole 16-row fragment tile. Filled by
    /// the `launch_quant_fp8` prologue at capture; reused sequentially
    /// (stream-ordered). Distinct scratch from act_q81_buf: the blocked
    /// kernel's A operand is fp8 codes, not Q8_1 blocks.
    pub act_fp8_pad_buf: Vec<RocmStorage>,
    /// Per-layer fused-QKV output staging ([n_q + 2*n_kv] F32); the three
    /// Q/K/V slices are D2D-copied into `q/k/v_buf`.
    pub fused_qkv_out: Vec<RocmStorage>,
    /// GRAVE Phase 4: per-layer GDN-2 buffers (`None` for softmax layers).
    /// Filled by [`DecodeGraphBuffers::allocate_gdl_layer`] — never inside
    /// capture (the Taylor-default H2D uploads + sync live there).
    pub gdl: Vec<Option<GdlLayerBuffers>>,
    /// KDA recurrent staging: per-layer alpha/beta/acc/state/conv_out buffers (`None` for non-KDA layers).
    /// Filled by [`DecodeGraphBuffers::allocate_kda_layer`] — never inside capture.
    pub kda: Vec<Option<KdaLayerBuffers>>,
    /// WhiteCrow token-parallel split partials, per layer (empty = split not
    /// in play: format not WhiteCrow, or the context too small to chunk).
    /// Layout per (slot=0, chunk, row, head): pm/pl = [max_chunks*steps*nh],
    /// po = [max_chunks*steps*nh*head_dim]. Allocated OUTSIDE capture (at the
    /// graph build, right after the arena-format resolution) so the pointers
    /// are stable across replays; both split kernels derive the live chunk
    /// count on device, and every live partial slot is rewritten each launch,
    /// so the buffers need no clearing between replays.
    pub wc_split_pm: Vec<RocmStorage>,
    pub wc_split_pl: Vec<RocmStorage>,
    pub wc_split_po: Vec<RocmStorage>,
    /// Tokens per chunk the partials were sized for (`None` = no split).
    pub wc_split_chunk_len: Option<u32>,
    pub num_layers: usize,
    pub max_ctx: usize,
    pub batch: usize,
    pub layer_devices: Vec<usize>,
}

/// The maximal contiguous runs of equal device ordinal in `layer_devices`,
/// in layer order. Each run becomes one captured graph, because a HIP graph
/// can only hold kernels from one device (see
/// `examples/p2p_graph_probe.rs`).
///
/// The runs tile the layer range exactly (no gaps, no overlap): segment k+1
/// starts where segment k ended. That is what lets the replay driver hand the
/// hidden state straight from one segment's last layer to the next segment's
/// first layer.
pub fn device_segments(layer_devices: &[usize]) -> Vec<(usize, std::ops::Range<usize>)> {
    let mut out: Vec<(usize, std::ops::Range<usize>)> = Vec::new();
    for (i, &ord) in layer_devices.iter().enumerate() {
        match out.last_mut() {
            Some((last_ord, range)) if *last_ord == ord => range.end = i + 1,
            _ => out.push((ord, i..i + 1)),
        }
    }
    out
}

/// KDA per-layer device state + projection buffers for HIP graph capture.
#[derive(Debug)]
pub struct KdaLayerBuffers {
    /// Recurrent state `[num_val_heads * head_dim * head_dim]` f32, zeros at alloc.
    pub state: RocmStorage,
    /// Projected alpha `[batch, num_val_heads]` f32.
    pub alpha: RocmStorage,
    /// Projected beta `[batch, num_val_heads]` f32.
    pub beta: RocmStorage,
    /// Delta-rule accumulator scratch `[batch * num_val_heads * head_dim]` f32.
    pub acc_scratch: RocmStorage,
    /// Short-conv output `[batch, conv_dim]` f32 before delta rule.
    pub conv_out: RocmStorage,
    /// The recurrent BRANCH output `[batch, num_val_heads * head_dim]` f32 —
    /// i.e. `value_dim`, what `ssm_out` consumes.
    ///
    /// Not `attn_out_buf`: that is sized `2 * q_dim` for the fused attention
    /// Q|gate projection (8192 on the 9B), while the recurrent branch is
    /// `value_dim` wide (4096). Feeding the 8192 buffer to `ssm_out`, whose
    /// weight is `[hidden, value_dim]`, aborted capture with
    ///   `act [1, 8192], weight [4096, 4096], out [1, 4096]`
    pub branch: RocmStorage,
    /// Projected z (the KDA output gate) `[batch, num_val_heads * head_dim]` f32.
    ///
    /// This CANNOT be the FFN's `gate_buf`, which is `[batch, intermediate_size]`.
    /// On the 9B those are 4096 and 12288, so projecting the KDA z gate into the
    /// FFN buffer made `linear_decode_into` refuse the shape and killed graph
    /// capture at step 1 with
    ///   `linear_decode[kda.attn_gate @ layer 0]: expected [1, 4096], got [1, 12288]`
    /// The z gate is per VALUE-STREAM element, so its width is
    /// `value_dim = num_val_heads * head_dim` — the same quantity `ssm_out`
    /// consumes. (The same confusion in `qwen35.rs` is why `attn_gate` is
    /// loaded at `q_dim.max(blk_value_dim(cfg))` there: the attention layers
    /// need q_dim, the recurrent ones need value_dim, and one tensor serves
    /// both.)
    pub gate: RocmStorage,
    pub num_val_heads: usize,
    pub num_key_heads: usize,
    pub head_dim: usize,
    pub conv_dim: usize,
}

/// GRAVE Phase 4 — per-layer GDN-2 device state + gate buffers.
///
/// One entry per LFM2 layer (`None` for non-GDL layers). The recurrent state
/// `[batch, heads, dk, dv]` replaces the ~800 MB K/V arenas at 32K with
/// 16 KB/layer; the gate buffers carry the Taylor-Calibrated operating
/// point until distillation writes learned gates (sidecar → re-upload).
#[derive(Debug)]
pub struct GdlLayerBuffers {
    /// Recurrent state `[batch*heads*dk*dv]` f32, zeros at alloc.
    pub state: RocmStorage,
    /// Decay α per key channel `[batch*heads*dk]` (init `2^(−1/dk)`).
    pub alpha: RocmStorage,
    /// Erase gate b `[batch*heads*dk]` (init 0.5 = σ(0)).
    pub b_gate: RocmStorage,
    /// Write gate w `[batch*heads*dv]` (init 0.5 = σ(0)).
    pub w_gate: RocmStorage,
    /// Gated-norm weight γ `[batch*heads*dv]` (init 1.0; learned later).
    pub norm_w: RocmStorage,
    /// Output-gate scalars `[batch*heads]` (init 0.0 → σ = 0.5, neutral).
    pub out_gate: RocmStorage,
    /// Expanded K buffer `[batch, nh*hd]` for GQA (None when nh == nkv).
    /// The GEMV writes compact `[nkv, hd]` into `k_buf`; this buffer receives
    /// the head-repeat expansion the GDL kernel consumes.
    pub k_exp_buf: Option<RocmStorage>,
    /// Expanded V buffer `[batch, nh*hd]` for GQA (None when nh == nkv).
    pub v_exp_buf: Option<RocmStorage>,
    pub heads: usize,
    pub kv_heads: usize,
    pub dk: usize,
    pub dv: usize,
    pub batch: usize,
}

/// PLAN-reduce-d2h-h2d (prefill seeding): one layer's eager device K/V source.
/// Raw device pointers — caller guarantees they outlive the seed copy.
pub struct EagerKvSource<'a> {
    /// Device pointer to eager K rows (`prefill_len × kv_stride` elements).
    pub k_dev: *const f32,
    /// Device pointer to eager V rows (`prefill_len × kv_stride` elements).
    pub v_dev: *const f32,
    /// Valid rows in each buffer.
    pub prefill_len: u32,
    /// Elements per row (`num_kv_heads × head_dim`).
    pub kv_stride: usize,
    /// Bytes per source element (1 for int8 codes, 2 for f16, 4 for f32).
    /// The seeder copies raw bytes when this matches the arena, converts
    /// f32->f16 when the graph runs f16 over f32 eager rows, and fails loud
    /// otherwise — never truncates.
    pub src_elem_bytes: usize,
    /// Phase-2 int8: f32 scale sidecar pointers, one per (row, kv_head).
    /// Set iff the eager arenas hold u8 codes; the seeder copies these into
    /// the graph's sidecars alongside the codes.
    pub k_scales_dev: Option<*const f32>,
    pub v_scales_dev: Option<*const f32>,
    /// GDL recurrent state pointer (set for GDL layers, None for dense).
    /// Points to `[batch*heads*dk*dv]` f32 device buffer.
    pub gdl_state: Option<*const f32>,
    /// Device ordinal where this eager source buffer resides (e.g. 0, 1).
    pub device_ordinal: i32,
    /// Borrow anchor so the pointers can't outlive the session caches.
    pub _anchor: std::marker::PhantomData<&'a ()>,
}

/// KDA recurrent state seeding from eager prefill.
pub struct EagerKdaSource<'a> {
    /// Points to `[num_val_heads*head_dim*head_dim]` f32 device buffer.
    pub kda_state: *const f32,
    pub _anchor: std::marker::PhantomData<&'a ()>,
}

/// Prefill seeding for a recurrent (ShortConv) layer's device ring. The host
/// mirror holds `kc` rows of `h_dim` (`host[t*h_dim + d]` = b·x from `t` steps
/// ago); the device ring is the transposed channel-major `[d, kc]` layout the
/// `grim_short_conv1d_causal_step` kernel updates in place.
/// Device conv-ring seed: the eager prefill ran the D2D conv, so the LIVE
/// ring is `conv_state_dev` (kernel layout `[channel][tap]`, same layout the
/// graph's `sc_state` uses) and the host mirror is stale. Seeding from the
/// host mirror in that case uploads ZEROS and the graph replays every
/// recurrent layer against a zeroed ring — greedy decode diverges from
/// eager after one token. D2D from the live device ring instead.
pub struct ConvDeviceSeed<'a> {
    pub dev_ring: *const f32,
    pub _anchor: std::marker::PhantomData<&'a ()>,
}

pub struct ConvRingSeed<'a> {
    pub host: &'a [f32],
    pub h_dim: usize,
    /// Ring depth (`l_cache - 1` previous b·x rows).
    pub kc: usize,
    pub _anchor: std::marker::PhantomData<&'a ()>,
}

impl DecodeGraphBuffers {
    /// Allocate full pool on `dev`. Fails fast on OOM -> caller falls back eager.
    /// `batch` parameterizes all per-token slots as `[batch, dim]`. `batch=1`
    /// preserves the original single-token shape.
    /// `num_heads` sizes the per-layer attention scratch (`attn_max/sum_buf`).
    #[allow(clippy::too_many_arguments)]
    pub fn allocate(
        dev: &RocmDevice,
        num_layers: usize,
        hidden_size: usize,
        n_q: usize,
        // Branch width for `attn_out_buf`: the ATTENTION branch (`q_dim`), which
        // is NOT the maxed `n_q`. See the field doc.
        n_attn_out: usize,
        n_k: usize,
        n_v: usize,
        intermediate_size: usize,
        max_ctx: usize,
        vocab_size: usize,
        num_heads: usize,
        batch: usize,
        n_expert: usize,
        top_k: usize,
        sc_h_dim: usize,
        sc_l_cache: usize,
    ) -> Result<Self> {
        Self::allocate_with_mla(
            dev,
            num_layers,
            hidden_size,
            n_q,
            n_attn_out,
            n_k,
            n_v,
            intermediate_size,
            max_ctx,
            vocab_size,
            num_heads,
            batch,
            n_expert,
            top_k,
            sc_h_dim,
            sc_l_cache,
            0,
        )
    }

    /// Allocate full pool on `dev`, with optional MLA latent KV arena support.
    #[allow(clippy::too_many_arguments)]
    pub fn allocate_with_mla(
        dev: &RocmDevice,
        num_layers: usize,
        hidden_size: usize,
        n_q: usize,
        n_attn_out: usize,
        n_k: usize,
        n_v: usize,
        intermediate_size: usize,
        max_ctx: usize,
        vocab_size: usize,
        num_heads: usize,
        batch: usize,
        n_expert: usize,
        top_k: usize,
        sc_h_dim: usize,
        sc_l_cache: usize,
        latent_kv_dim: usize,
    ) -> Result<Self> {
        Self::allocate_with_layer_devices(
            dev,
            None,
            num_layers,
            hidden_size,
            n_q,
            n_attn_out,
            n_k,
            n_v,
            intermediate_size,
            max_ctx,
            vocab_size,
            num_heads,
            batch,
            n_expert,
            top_k,
            sc_h_dim,
            sc_l_cache,
            latent_kv_dim,
        )
    }

    /// Multi-device P2P: allocate per-layer buffers on each layer's assigned device.
    #[allow(clippy::too_many_arguments)]
    pub fn allocate_with_layer_devices(
        dev: &RocmDevice,
        layer_devices: Option<&[usize]>,
        num_layers: usize,
        hidden_size: usize,
        n_q: usize,
        n_attn_out: usize,
        n_k: usize,
        n_v: usize,
        intermediate_size: usize,
        max_ctx: usize,
        vocab_size: usize,
        num_heads: usize,
        batch: usize,
        n_expert: usize,
        top_k: usize,
        sc_h_dim: usize,
        sc_l_cache: usize,
        latent_kv_dim: usize,
    ) -> Result<Self> {
        if num_layers == 0 || hidden_size == 0 || max_ctx == 0 || vocab_size == 0 || batch == 0 {
            return Err(Error::Backend(
                "DecodeGraphBuffers::allocate: zero dim".into(),
            ));
        }
        let dt = dtype_f32();
        let mut layer_input = Vec::with_capacity(num_layers);
        let mut layer_output = Vec::with_capacity(num_layers);
        let mut q_buf = Vec::with_capacity(num_layers);
        let mut k_buf = Vec::with_capacity(num_layers);
        let mut v_buf = Vec::with_capacity(num_layers);
        let mut attn_out_buf = Vec::with_capacity(num_layers);
        let mut q_head_buf = Vec::with_capacity(num_layers);
        let mut q_gate_buf = Vec::with_capacity(num_layers);
        let mut gate_up_buf = Vec::with_capacity(num_layers);
        let mut gate_buf = Vec::with_capacity(num_layers);
        let mut up_buf = Vec::with_capacity(num_layers);
        let mut activated_buf = Vec::with_capacity(num_layers);
        let mut norm_buf = Vec::with_capacity(num_layers);
        let mut attn_max_buf = Vec::with_capacity(num_layers);
        let mut attn_sum_buf = Vec::with_capacity(num_layers);
        let mut act_q81_buf = Vec::with_capacity(num_layers);
        let mut act_fp8_pad_buf = Vec::with_capacity(num_layers);
        // M2/S2: MoE + ShortConv staging (empty when the model lacks those).
        let moe_layers = n_expert > 0 && top_k > 0;
        let sc_layers = sc_h_dim > 0 && sc_l_cache > 1;
        let mut moe_gate_logits = Vec::with_capacity(if moe_layers { num_layers } else { 0 });
        let mut moe_out = Vec::with_capacity(if moe_layers { num_layers } else { 0 });
        let mut sc_proj_buf = Vec::with_capacity(if sc_layers { num_layers } else { 0 });
        let mut sc_b = Vec::with_capacity(if sc_layers { num_layers } else { 0 });
        let mut sc_c = Vec::with_capacity(if sc_layers { num_layers } else { 0 });
        let mut sc_x = Vec::with_capacity(if sc_layers { num_layers } else { 0 });
        let mut sc_bx = Vec::with_capacity(if sc_layers { num_layers } else { 0 });
        let mut sc_sum = Vec::with_capacity(if sc_layers { num_layers } else { 0 });
        let mut sc_y = Vec::with_capacity(if sc_layers { num_layers } else { 0 });
        let mut sc_state = Vec::with_capacity(if sc_layers { num_layers } else { 0 });
        let mut fused_qkv_out = Vec::with_capacity(num_layers);
        let mut k_arena = Vec::with_capacity(num_layers);
        let mut v_arena = Vec::with_capacity(num_layers);
        // Filled on demand by `set_int8_kv_scales` (an empty vec means the
        // arenas are f32/f16 and no sidecar exists).
        let k_scale_arena: Vec<RocmStorage> = Vec::new();
        let v_scale_arena: Vec<RocmStorage> = Vec::new();
        let mut latent_kv_arena =
            Vec::with_capacity(if latent_kv_dim > 0 { num_layers } else { 0 });
        // Q8_1 staging must cover the widest activation row (hidden vs inter),
        // times one row per batch slot (P3).
        let q81_elems = hidden_size.max(intermediate_size).max(32);
        let q81_bytes = batch * (q81_elems / 32) * 36;
        let q81_dt = DType {
            arith: ArithType::U8,
            storage: DTypeStorage::Native,
        };
        let hid = hidden_size.max(1);
        let nqk = n_q.max(1);
        let nkk = n_k.max(1);
        let nvk = n_v.max(1);
        let inter = intermediate_size.max(1);
        for layer_i in 0..num_layers {
            let l_dev = match layer_devices.and_then(|ds| ds.get(layer_i)) {
                Some(&ord) if ord != dev.ordinal => crate::RocmDevice::shared(ord),
                _ => crate::RocmDevice::shared(dev.ordinal),
            };
            layer_input.push(RocmStorage::alloc_gpu(
                &Shape::new(vec![batch, hid]),
                dt.clone(),
                &l_dev.allocator,
                l_dev.ordinal,
            )?);
            layer_output.push(RocmStorage::alloc_gpu(
                &Shape::new(vec![batch, hid]),
                dt.clone(),
                &l_dev.allocator,
                l_dev.ordinal,
            )?);
            q_buf.push(RocmStorage::alloc_gpu(
                &Shape::new(vec![batch, nqk]),
                dt.clone(),
                &l_dev.allocator,
                l_dev.ordinal,
            )?);
            k_buf.push(RocmStorage::alloc_gpu(
                &Shape::new(vec![batch, nkk]),
                dt.clone(),
                &l_dev.allocator,
                l_dev.ordinal,
            )?);
            v_buf.push(RocmStorage::alloc_gpu(
                &Shape::new(vec![batch, nvk]),
                dt.clone(),
                &l_dev.allocator,
                l_dev.ordinal,
            )?);
            q_head_buf.push(RocmStorage::alloc_gpu(
                &Shape::new(vec![batch, n_attn_out]),
                dt.clone(),
                &l_dev.allocator,
                l_dev.ordinal,
            )?);
            q_gate_buf.push(RocmStorage::alloc_gpu(
                &Shape::new(vec![batch, n_attn_out]),
                dt.clone(),
                &l_dev.allocator,
                l_dev.ordinal,
            )?);
            attn_out_buf.push(RocmStorage::alloc_gpu(
                &Shape::new(vec![batch, n_attn_out]),
                dt.clone(),
                &l_dev.allocator,
                l_dev.ordinal,
            )?);
            gate_up_buf.push(RocmStorage::alloc_gpu(
                &Shape::new(vec![batch, 2 * inter]),
                dt.clone(),
                &l_dev.allocator,
                l_dev.ordinal,
            )?);
            gate_buf.push(RocmStorage::alloc_gpu(
                &Shape::new(vec![batch, inter]),
                dt.clone(),
                &l_dev.allocator,
                l_dev.ordinal,
            )?);
            up_buf.push(RocmStorage::alloc_gpu(
                &Shape::new(vec![batch, inter]),
                dt.clone(),
                &l_dev.allocator,
                l_dev.ordinal,
            )?);
            activated_buf.push(RocmStorage::alloc_gpu(
                &Shape::new(vec![batch, inter]),
                dt.clone(),
                &l_dev.allocator,
                l_dev.ordinal,
            )?);
            norm_buf.push(RocmStorage::alloc_gpu(
                &Shape::new(vec![batch, hid]),
                dt.clone(),
                &l_dev.allocator,
                l_dev.ordinal,
            )?);
            // ZEROED at allocation, like the recurrent state. The norm buffer
            // is a scratch slot, and the FUSED paths (the LFM2's norm-fused
            // Q8_0 projections) never write it — so without this it holds
            // whatever the recycled block held, and two graphs that captured
            // at different times disagree on it. A scratch slot that a fused
            // path leaves untouched must still start at a KNOWN value, or the
            // model's answer depends on the allocator's history.
            {
                let last = norm_buf.last().expect("just pushed");
                let ptr = last
                    .device_ptr_u64()
                    .ok_or_else(|| Error::Backend("norm_buf: no device ptr".into()))?
                    as *mut std::ffi::c_void;
                let _guard = crate::device::util::DeviceGuard::set(l_dev.ordinal as i32);
                let res = unsafe {
                    crate::device::handles::hipMemsetAsync(
                        ptr,
                        0,
                        last.bytes,
                        l_dev.active_stream(),
                    )
                };
                if res != crate::hipSuccess {
                    return Err(Error::Backend(format!("norm_buf memset: {res}")));
                }
            }
            attn_max_buf.push(RocmStorage::alloc_gpu(
                &Shape::new(vec![batch * num_heads.max(1)]),
                dt.clone(),
                &l_dev.allocator,
                l_dev.ordinal,
            )?);
            attn_sum_buf.push(RocmStorage::alloc_gpu(
                &Shape::new(vec![batch * num_heads.max(1)]),
                dt.clone(),
                &l_dev.allocator,
                l_dev.ordinal,
            )?);
            act_q81_buf.push(RocmStorage::alloc_gpu_with_bytes(
                &Shape::new(vec![q81_bytes]),
                q81_dt.clone(),
                q81_bytes,
                &l_dev.allocator,
                l_dev.ordinal,
            )?);
            // Padded fp8 act scratch: max row padded to whole 16-row tiles.
            let fp8_pad_bytes = batch * 16 * q81_elems.max(intermediate_size).max(32);
            act_fp8_pad_buf.push(RocmStorage::alloc_gpu_with_bytes(
                &Shape::new(vec![fp8_pad_bytes]),
                q81_dt.clone(),
                fp8_pad_bytes,
                &l_dev.allocator,
                l_dev.ordinal,
            )?);
            fused_qkv_out.push(RocmStorage::alloc_gpu(
                &Shape::new(vec![batch, nqk + 2 * nkk]),
                dt.clone(),
                &l_dev.allocator,
                l_dev.ordinal,
            )?);
            // GRAVE Phase 1: KV arena element dtype follows GRIM_F16_KV so
            // the kv_append/attention kernels' f16 flag matches the storage.
            //
            // Only f32 and f16 are selectable here. Packed KQuant storage (q8_0 /
            // q4_K) is deliberately NOT wired into this arena: its consumer is
            // the non-quantized attention kernel, which reads these rows raw, so
            // a packed dtype here would be reinterpreted rather than dequantized.
            // Quantized KV goes through the paged path
            // (`launch_paged_attention_quant`), whose reader inlines
            // `dequant_kv_element` - see tests/paged_attention_quant.rs.
            let kv_dt = if crate::kernels::qkv_attention::kv_f16_enabled() {
                DType {
                    arith: ArithType::F16,
                    storage: grim_tensor::Storage::Native,
                }
            } else {
                dt.clone()
            };
            let k_arena_s = RocmStorage::alloc_gpu(
                &Shape::new(vec![batch * max_ctx, nkk]),
                kv_dt.clone(),
                &l_dev.allocator,
                l_dev.ordinal,
            )?;
            let v_arena_s = RocmStorage::alloc_gpu(
                &Shape::new(vec![batch * max_ctx, nvk]),
                kv_dt,
                &l_dev.allocator,
                l_dev.ordinal,
            )?;
            v_arena.push(v_arena_s);
            k_arena.push(k_arena_s);
            // Zero the KV arenas at allocation (same rationale as `pos_dev`
            // above): `alloc_gpu` recycles pool blocks, and a replay that
            // reads any V byte the fresh graph has not written inherits the
            // PREVIOUS graph's values — deterministic per allocation history
            // (bit-stable 1.3529 vs -1.3476 in the lfm2_graph_moe suite run,
            // clean in isolation where first-touch pages are driver-zeroed).
            // One memset per arena at graph construction, off the hot path.
            for arena in [k_arena.last().unwrap(), v_arena.last().unwrap()] {
                let _zg = crate::device::util::DeviceGuard::set(arena.device_ordinal() as i32);
                let zp = arena
                    .device_ptr
                    .ok_or_else(|| Error::Backend("kv arena zero: no ptr".into()))?;
                let res: crate::HipErrorT = unsafe {
                    crate::device::handles::hipMemset(zp as *mut std::ffi::c_void, 0, arena.bytes)
                };
                if res != crate::hipSuccess {
                    return Err(Error::Backend(format!("zero kv arena: {res}")));
                }
            }
            if latent_kv_dim > 0 {
                latent_kv_arena.push(RocmStorage::alloc_gpu(
                    &Shape::new(vec![batch * max_ctx, latent_kv_dim]),
                    dt.clone(),
                    &l_dev.allocator,
                    l_dev.ordinal,
                )?);
            }
        }
        let head_dev = match layer_devices.and_then(|ds| ds.last()) {
            Some(&ord) if ord != dev.ordinal => crate::RocmDevice::shared(ord),
            _ => crate::RocmDevice::shared(dev.ordinal),
        };
        let head_input = RocmStorage::alloc_gpu(
            &Shape::new(vec![batch, hid]),
            dt.clone(),
            &head_dev.allocator,
            head_dev.ordinal,
        )?;
        let head_output = Arc::new(RocmStorage::alloc_gpu(
            &Shape::new(vec![batch, vocab_size]),
            dt.clone(),
            &head_dev.allocator,
            head_dev.ordinal,
        )?);
        // One i32 position per batch slot (P3): the graph's append/bump
        // kernels index `pos_dev[slot]`.
        let pos_dev = RocmStorage::alloc_gpu(
            &Shape::new(vec![batch]),
            dt.clone(),
            &dev.allocator,
            dev.ordinal,
        )?;
        // The device mirror must start at 0 to match `current_pos` (the host
        // mirror above is 0). `alloc_gpu` does NOT zero: a recycled block
        // inherits the previous pool's bytes, and `pos_dev` is read as `int*`
        // by the append/bump kernels but written as f32 by `write_pos_async`
        // — stale float bits (e.g. 3.0f = 0x40400000) read as ~1.08e9 and the
        // first append writes at row*~69GB. The int8 pool survived this only
        // by allocating fresh (driver-zeroed) pages; the int4 pool after it
        // recycled the int8 pool and faulted the warmup append. Zeroed here,
        // outside any capture bracket (pool construction is pre-capture).
        {
            let _g = crate::device::util::DeviceGuard::set(dev.ordinal as i32);
            let p = pos_dev
                .device_ptr
                .ok_or_else(|| Error::Backend("pos_dev has no device ptr".into()))?;
            let res: crate::HipErrorT = unsafe {
                crate::device::handles::hipMemset(
                    p as *mut std::ffi::c_void,
                    0,
                    batch.max(1) * std::mem::size_of::<f32>(),
                )
            };
            if res != crate::hipSuccess {
                return Err(Error::Backend(format!("zero pos_dev: {res}")));
            }
        }
        let token_ids_dev = RocmStorage::alloc_gpu(
            &Shape::new(vec![batch.max(1)]),
            dt.clone(),
            &dev.allocator,
            dev.ordinal,
        )?;
        let attn_dummy = RocmStorage::alloc_gpu(
            &Shape::new(vec![1]),
            dt.clone(),
            &dev.allocator,
            dev.ordinal,
        )?;
        // A5b: one zeroed pos replica + attn_dummy replica per FOREIGN device
        // in the placement, so no graph kernel reads another device's memory.
        // Single-device placements produce empty vecs (fallback to the
        // originals — byte-identical behavior).
        let mut pos_replicas = Vec::new();
        let mut pos_replica_ords = Vec::new();
        let mut attn_dummy_replicas = Vec::new();
        let mut attn_dummy_replica_ords = Vec::new();
        if let Some(ds) = layer_devices {
            for &ord in ds.iter().collect::<std::collections::BTreeSet<_>>() {
                if ord == dev.ordinal {
                    continue;
                }
                let r_dev = crate::RocmDevice::shared(ord);
                let pr = RocmStorage::alloc_gpu(
                    &Shape::new(vec![batch]),
                    dt.clone(),
                    &r_dev.allocator,
                    r_dev.ordinal,
                )?;
                {
                    let _g = crate::device::util::DeviceGuard::set(r_dev.ordinal as i32);
                    let p = pr
                        .device_ptr
                        .ok_or_else(|| Error::Backend("pos replica has no device ptr".into()))?;
                    let res: crate::HipErrorT = unsafe {
                        crate::device::handles::hipMemset(
                            p as *mut std::ffi::c_void,
                            0,
                            batch.max(1) * std::mem::size_of::<f32>(),
                        )
                    };
                    if res != crate::hipSuccess {
                        return Err(Error::Backend(format!("zero pos replica: {res}")));
                    }
                }
                pos_replicas.push(pr);
                pos_replica_ords.push(r_dev.ordinal);
                attn_dummy_replicas.push(RocmStorage::alloc_gpu(
                    &Shape::new(vec![1]),
                    dt.clone(),
                    &r_dev.allocator,
                    r_dev.ordinal,
                )?);
                attn_dummy_replica_ords.push(r_dev.ordinal);
            }
        }
        // D1-2a: one zeroed total per layer, on the layer's device.
        let mut layer_totals = Vec::with_capacity(num_layers);
        for layer_i in 0..num_layers {
            let l_dev = match layer_devices.and_then(|ds| ds.get(layer_i)) {
                Some(&ord) if ord != dev.ordinal => crate::RocmDevice::shared(ord),
                _ => crate::RocmDevice::shared(dev.ordinal),
            };
            let t = RocmStorage::alloc_gpu(
                &Shape::new(vec![batch.max(1)]),
                dt.clone(),
                &l_dev.allocator,
                l_dev.ordinal,
            )?;
            {
                let _g = crate::device::util::DeviceGuard::set(l_dev.ordinal as i32);
                let p = t
                    .device_ptr
                    .ok_or_else(|| Error::Backend("layer_total has no ptr".into()))?;
                let res: crate::HipErrorT = unsafe {
                    crate::device::handles::hipMemset(
                        p as *mut std::ffi::c_void,
                        0,
                        batch.max(1) * std::mem::size_of::<f32>(),
                    )
                };
                if res != crate::hipSuccess {
                    return Err(Error::Backend(format!("zero layer_total: {res}")));
                }
            }
            layer_totals.push(t);
        }
        let push_all = |dev: &RocmDevice,
                        dt: &DType,
                        moe_gate_logits: &mut Vec<RocmStorage>,
                        moe_out: &mut Vec<RocmStorage>,
                        sc_proj_buf: &mut Vec<RocmStorage>,
                        sc_b: &mut Vec<RocmStorage>,
                        sc_c: &mut Vec<RocmStorage>,
                        sc_x: &mut Vec<RocmStorage>,
                        sc_bx: &mut Vec<RocmStorage>,
                        sc_sum: &mut Vec<RocmStorage>,
                        sc_y: &mut Vec<RocmStorage>,
                        sc_state: &mut Vec<RocmStorage>|
         -> Result<()> {
            for _ in 0..num_layers {
                if moe_layers {
                    moe_gate_logits.push(RocmStorage::alloc_gpu(
                        &Shape::new(vec![batch, n_expert]),
                        dt.clone(),
                        &dev.allocator,
                        dev.ordinal,
                    )?);
                    moe_out.push(RocmStorage::alloc_gpu(
                        &Shape::new(vec![batch, hidden_size]),
                        dt.clone(),
                        &dev.allocator,
                        dev.ordinal,
                    )?);
                }
                if sc_layers {
                    let kc = sc_l_cache - 1;
                    sc_proj_buf.push(RocmStorage::alloc_gpu(
                        &Shape::new(vec![batch, 3 * sc_h_dim]),
                        dt.clone(),
                        &dev.allocator,
                        dev.ordinal,
                    )?);
                    sc_b.push(RocmStorage::alloc_gpu(
                        &Shape::new(vec![batch * sc_h_dim]),
                        dt.clone(),
                        &dev.allocator,
                        dev.ordinal,
                    )?);
                    sc_c.push(RocmStorage::alloc_gpu(
                        &Shape::new(vec![batch * sc_h_dim]),
                        dt.clone(),
                        &dev.allocator,
                        dev.ordinal,
                    )?);
                    sc_x.push(RocmStorage::alloc_gpu(
                        &Shape::new(vec![batch * sc_h_dim]),
                        dt.clone(),
                        &dev.allocator,
                        dev.ordinal,
                    )?);
                    sc_bx.push(RocmStorage::alloc_gpu(
                        &Shape::new(vec![batch * sc_h_dim]),
                        dt.clone(),
                        &dev.allocator,
                        dev.ordinal,
                    )?);
                    sc_sum.push(RocmStorage::alloc_gpu(
                        &Shape::new(vec![batch * sc_h_dim]),
                        dt.clone(),
                        &dev.allocator,
                        dev.ordinal,
                    )?);
                    sc_y.push(RocmStorage::alloc_gpu(
                        &Shape::new(vec![batch, sc_h_dim]),
                        dt.clone(),
                        &dev.allocator,
                        dev.ordinal,
                    )?);
                    sc_state.push(RocmStorage::alloc_gpu(
                        &Shape::new(vec![sc_h_dim * kc]),
                        dt.clone(),
                        &dev.allocator,
                        dev.ordinal,
                    )?);
                    // The short-conv state is a STATE: a fresh session has no
                    // history, so it starts at zero and not at whatever the
                    // recycled block held.
                    let last = sc_state.last().expect("just pushed");
                    Self::zero_device_buffer(dev, last)?;
                }
            }
            Ok(())
        };
        push_all(
            dev,
            &dt,
            &mut moe_gate_logits,
            &mut moe_out,
            &mut sc_proj_buf,
            &mut sc_b,
            &mut sc_c,
            &mut sc_x,
            &mut sc_bx,
            &mut sc_sum,
            &mut sc_y,
            &mut sc_state,
        )?;
        let (moe_route_tokens, moe_route_experts, moe_route_weights) = if moe_layers {
            let np = batch * top_k;
            let udt = DType {
                arith: grim_tensor::ArithType::U32,
                storage: grim_tensor::Storage::Native,
            };
            (
                RocmStorage::alloc_gpu(
                    &Shape::new(vec![np]),
                    udt.clone(),
                    &dev.allocator,
                    dev.ordinal,
                )?,
                RocmStorage::alloc_gpu(&Shape::new(vec![np]), udt, &dev.allocator, dev.ordinal)?,
                RocmStorage::alloc_gpu(
                    &Shape::new(vec![np]),
                    dt.clone(),
                    &dev.allocator,
                    dev.ordinal,
                )?,
            )
        } else {
            // Unused dummy [1] f32 zeros keeps the fields non-null.
            let d = RocmStorage::alloc_gpu(
                &Shape::new(vec![1]),
                dt.clone(),
                &dev.allocator,
                dev.ordinal,
            )?;
            let d2 = RocmStorage::alloc_gpu(
                &Shape::new(vec![1]),
                dt.clone(),
                &dev.allocator,
                dev.ordinal,
            )?;
            let d3 = RocmStorage::alloc_gpu(
                &Shape::new(vec![1]),
                dt.clone(),
                &dev.allocator,
                dev.ordinal,
            )?;
            (d, d2, d3)
        };

        Ok(Self {
            layer_input,
            layer_output,
            q_buf,
            k_buf,
            v_buf,
            attn_out_buf,
            q_head_buf,
            q_gate_buf,
            gate_up_buf,
            gate_buf,
            up_buf,
            activated_buf,
            norm_buf,
            attn_max_buf,
            attn_sum_buf,
            k_arena,
            v_arena,
            k_scale_arena,
            v_scale_arena,
            kv_ring: KvRingPolicy::UNLIMITED,
            kv_arena_bits: None,
            latent_kv_arena,
            head_input,
            head_output,
            current_pos: 0,
            pos_dev,
            pos_replicas,
            pos_replica_ords,
            attn_dummy_replicas,
            attn_dummy_replica_ords,
            layer_totals,
            emitted_task_graph: None,
            token_ids_dev,
            attn_dummy,
            act_q81_buf,
            act_fp8_pad_buf,
            fused_qkv_out,
            gdl: (0..num_layers).map(|_| None).collect(),
            kda: (0..num_layers).map(|_| None).collect(),
            wc_split_pm: Vec::new(),
            wc_split_pl: Vec::new(),
            wc_split_po: Vec::new(),
            wc_split_chunk_len: None,
            moe_gate_logits,
            moe_out,
            moe_route_tokens,
            moe_route_experts,
            moe_route_weights,
            sc_proj_buf,
            sc_b,
            sc_c,
            sc_x,
            sc_bx,
            sc_sum,
            sc_y,
            sc_state,
            num_layers,
            max_ctx,
            batch,
            layer_devices: match layer_devices {
                Some(ds) => ds.to_vec(),
                None => vec![dev.ordinal; num_layers],
            },
            boundary_out: Self::build_boundary_out(layer_devices, batch, hid, &dt)?,
        })
    }

    /// One handoff buffer per device transition, on the PRODUCING device: the
    /// last layer of segment k writes here (so segment k's graph stays
    /// single-device), and the replay driver copies it into the next segment's
    /// `layer_input` before launching segment k+1.
    fn build_boundary_out(
        layer_devices: Option<&[usize]>,
        batch: usize,
        hid: usize,
        dt: &DType,
    ) -> Result<Vec<RocmStorage>> {
        let Some(ds) = layer_devices else {
            return Ok(Vec::new());
        };
        let mut out = Vec::new();
        for w in ds.windows(2) {
            if w[0] == w[1] {
                continue;
            }
            let l_dev = crate::RocmDevice::shared(w[0]);
            out.push(RocmStorage::alloc_gpu(
                &Shape::new(vec![batch, hid]),
                dt.clone(),
                &l_dev.allocator,
                l_dev.ordinal,
            )?);
        }
        Ok(out)
    }

    /// The `boundary_out` slot for the transition AFTER `layer_idx`, or `None`
    /// when layer `layer_idx + 1` is on the same device. Boundaries are
    /// numbered in layer order, matching `boundary_out`'s construction order.
    pub fn boundary_index(&self, layer_idx: usize) -> Option<usize> {
        if layer_idx + 1 >= self.layer_input.len() {
            return None;
        }
        if self.layer_input[layer_idx].device_ordinal()
            == self.layer_input[layer_idx + 1].device_ordinal()
        {
            return None;
        }
        Some(
            (0..=layer_idx)
                .filter(|&i| {
                    i + 1 < self.layer_input.len()
                        && self.layer_input[i].device_ordinal()
                            != self.layer_input[i + 1].device_ordinal()
                })
                .count()
                - 1,
        )
    }

    /// Packed KV codes per element width, from STORAGE (never env): `None` when
    /// the arenas hold f16/f32 K/V directly, `Some(8)` for int8, `Some(4)` for
    /// int4, `Some(2)` for int2v4.
    ///
    /// This replaced `kv_arena_is_int8() -> bool`, which asked `arith == U8` and
    /// therefore answered "int8" for int4 and int2v4 as well. A bool cannot carry
    /// three widths, and the dispatch that consumed it could not tell them apart
    /// — which is how `GRIM_KV_ARENA=int2v4` came to run the 8-bit kernels over
    /// 4-codes-per-byte keys.
    ///
    /// The width is RECORDED at allocation ([`Self::kv_arena_bits`]) rather than
    /// inferred from the shape: a packed row's innermost axis is a head's BYTE
    /// count, so the shapes of an 8-bit and a 4-bit arena differ only by a
    /// factor of two in that axis, and `head_dim` is a model property this
    /// struct does not hold.
    pub fn kv_arena_packed_width(&self) -> Option<u32> {
        // Storage truth: the recorded width and the arena's dtype must agree, so
        // a mis-set field cannot route a packed arena to the unpacked launcher.
        let is_u8 = self
            .k_arena
            .first()
            .map(|a| a.dtype.arith == grim_tensor::dtype::ArithType::U8)
            .unwrap_or(false);
        debug_assert_eq!(
            is_u8,
            self.kv_arena_bits.is_some(),
            "kv_arena_bits ({:?}) disagrees with the arena storage (U8: {is_u8})",
            self.kv_arena_bits
        );
        self.kv_arena_bits
    }

    /// Allocate the per-layer f32 scale sidecars for a PACKED KV format, one f32
    /// per (row, kv_head) laid out `[batch * max_ctx, nkv_heads]`. The layout is
    /// the same for every packed width: int8, int4 and int2v4 each store one
    /// scale per (row, kv head) per side. MUST run outside capture. Idempotent:
    /// a second call is a no-op.
    pub fn allocate_packed_kv_scales(
        &mut self,
        _dev: &RocmDevice,
        num_kv_heads: usize,
    ) -> Result<()> {
        if !self.k_scale_arena.is_empty() {
            return Ok(());
        }
        if num_kv_heads == 0 {
            return Err(Error::Backend(
                "allocate_packed_kv_scales: num_kv_heads is zero".into(),
            ));
        }
        let rows = self.batch.max(1) * self.max_ctx;
        let shape = Shape::new(vec![rows, num_kv_heads]);
        for (i, arena) in self.k_arena.iter().enumerate() {
            let ord = self
                .layer_devices
                .get(i)
                .copied()
                .unwrap_or(arena.device_ordinal());
            let l_dev = RocmDevice::shared(ord);
            self.k_scale_arena.push(RocmStorage::alloc_gpu(
                &shape,
                DType {
                    arith: ArithType::F32,
                    storage: DTypeStorage::Native,
                },
                &l_dev.allocator,
                l_dev.ordinal,
            )?);
            self.v_scale_arena.push(RocmStorage::alloc_gpu(
                &shape,
                DType {
                    arith: ArithType::F32,
                    storage: DTypeStorage::Native,
                },
                &l_dev.allocator,
                l_dev.ordinal,
            )?);
        }
        Ok(())
    }

    /// Zero a device buffer on the device's active stream.
    ///
    /// The caching allocator hands out RECYCLED blocks, and a recycled block
    /// still holds the previous owner's values. A buffer that must START at
    /// zero — the recurrent state, the short-conv state — therefore has to be
    /// zeroed explicitly: a fresh `hipMalloc` happens to be zero, but the
    /// second graph to ask for the same size gets the first graph's leftovers.
    ///
    /// The symptom this fixes is nasty precisely because it is small: two runs
    /// of the same model on the same weights disagreed in the 1e-5 class, and
    /// the divergence moved with the ALLOCATION HISTORY rather than with the
    /// code. A recurrent state that is not zero at the start of a session is
    /// a wrong answer, not a slow one.
    fn zero_device_buffer(dev: &RocmDevice, buf: &RocmStorage) -> Result<()> {
        let ptr = buf
            .device_ptr_u64()
            .ok_or_else(|| Error::Backend("zero_device_buffer: no device ptr".into()))?
            as *mut c_void;
        let _guard = crate::device::util::DeviceGuard::set(dev.ordinal as i32);
        let res = unsafe {
            crate::device::handles::hipMemsetAsync(ptr, 0, buf.bytes, dev.active_stream())
        };
        if res != crate::hipSuccess {
            return Err(Error::Backend(format!(
                "zero_device_buffer: hipMemsetD8Async {res}"
            )));
        }
        Ok(())
    }

    /// Allocate one layer's KDA recurrent device buffers (state + projections).
    /// MUST run outside capture.
    pub fn allocate_kda_layer(
        &mut self,
        dev: &RocmDevice,
        layer_idx: usize,
        batch: usize,
        num_val_heads: usize,
        num_key_heads: usize,
        head_dim: usize,
        conv_dim: usize,
    ) -> Result<()> {
        let _dev_guard = crate::device::util::DeviceGuard::set(dev.ordinal as i32);
        if layer_idx >= self.kda.len() {
            return Err(Error::Backend(format!(
                "allocate_kda_layer: layer {layer_idx} >= {}",
                self.kda.len()
            )));
        }
        if batch == 0 || num_val_heads == 0 || num_key_heads == 0 || head_dim == 0 || conv_dim == 0
        {
            return Err(Error::Backend("allocate_kda_layer: zero dim".into()));
        }
        let dt = dtype_f32();
        let state = RocmStorage::alloc_gpu(
            &Shape::new(vec![num_val_heads * head_dim * head_dim]),
            dt.clone(),
            &dev.allocator,
            dev.ordinal,
        )?;
        let branch = RocmStorage::alloc_gpu(
            &Shape::new(vec![batch, num_val_heads * head_dim]),
            dt.clone(),
            &dev.allocator,
            dev.ordinal,
        )?;
        let gate = RocmStorage::alloc_gpu(
            &Shape::new(vec![batch, num_val_heads * head_dim]),
            dt.clone(),
            &dev.allocator,
            dev.ordinal,
        )?;
        let alpha = RocmStorage::alloc_gpu(
            &Shape::new(vec![batch, num_val_heads]),
            dt.clone(),
            &dev.allocator,
            dev.ordinal,
        )?;
        let beta = RocmStorage::alloc_gpu(
            &Shape::new(vec![batch, num_val_heads]),
            dt.clone(),
            &dev.allocator,
            dev.ordinal,
        )?;
        let acc_scratch = RocmStorage::alloc_gpu(
            &Shape::new(vec![batch * num_val_heads * head_dim]),
            dt.clone(),
            &dev.allocator,
            dev.ordinal,
        )?;
        let conv_out = RocmStorage::alloc_gpu(
            &Shape::new(vec![batch, conv_dim]),
            dt,
            &dev.allocator,
            dev.ordinal,
        )?;
        // The recurrent state starts a SESSION, so it starts at zero — not at
        // whatever the recycled block happened to hold.
        Self::zero_device_buffer(dev, &state)?;

        self.kda[layer_idx] = Some(KdaLayerBuffers {
            state,
            alpha,
            beta,
            acc_scratch,
            conv_out,
            branch,
            gate,
            num_val_heads,
            num_key_heads,
            head_dim,
            conv_dim,
        });
        Ok(())
    }

    /// GRAVE Phase 4: allocate one layer's GDN-2 device buffers with the
    /// Taylor-Calibrated operating point (decay `2^(−1/dk)`, erase/write 0.5,
    /// norm weight 1.0, output gate 0.0). MUST run outside capture: performs
    /// H2D uploads + a synchronize (same contract as the seed methods).
    /// Learned gates from distillation re-upload through this path later
    /// (sidecar → host vecs → same H2D; no struct change).
    pub fn allocate_gdl_layer(
        &mut self,
        dev: &RocmDevice,
        layer_idx: usize,
        batch: usize,
        heads: usize,
        kv_heads: usize,
        dk: usize,
        dv: usize,
    ) -> Result<()> {
        // P1-3: the H2D copies below bind to the CALLING THREAD's current
        // device. Pin `dev` for the whole call so a drifted thread cannot stage
        // into another card's context.
        let _dev_guard = crate::device::util::DeviceGuard::set(dev.ordinal as i32);
        if layer_idx >= self.gdl.len() {
            return Err(Error::Backend(format!(
                "allocate_gdl_layer: layer {layer_idx} >= {}",
                self.gdl.len()
            )));
        }
        if batch == 0 || heads == 0 || kv_heads == 0 || dk == 0 || dv == 0 {
            return Err(Error::Backend("allocate_gdl_layer: zero dim".into()));
        }
        if heads % kv_heads != 0 {
            return Err(Error::Backend(format!(
                "allocate_gdl_layer: heads ({heads}) must be a multiple of kv_heads ({kv_heads})"
            )));
        }
        if dk != 64 || dv != 64 {
            return Err(Error::Backend(format!(
                "allocate_gdl_layer: LDS fast path needs dk==dv==64, got {dk}x{dv}"
            )));
        }
        let dt = dtype_f32();
        let n_qk = batch * heads * dk;
        let n_v = batch * heads * dv;
        let n_g = batch * heads;
        let decay = (-2.0f64.ln() / dk as f64).exp() as f32;
        let upload =
            |dev: &RocmDevice, vals: Vec<f32>, shape: grim_tensor::Shape| -> Result<RocmStorage> {
                let st = RocmStorage::alloc_gpu(&shape, dt.clone(), &dev.allocator, dev.ordinal)?;
                let dst = st
                    .device_ptr_u64()
                    .ok_or_else(|| Error::Backend("allocate_gdl_layer: no ptr".into()))?
                    as *mut c_void;
                let bytes = vals.len() * 4;
                let host_bytes =
                    unsafe { std::slice::from_raw_parts(vals.as_ptr() as *const u8, bytes) };
                let (src_ptr, _pinned) =
                    crate::memory::pinned::stage_h2d_pinned(dev.ordinal, host_bytes);
                let stream = dev.active_stream();
                let res: crate::HipErrorT = unsafe {
                    crate::hipMemcpyAsync(
                        dst,
                        src_ptr as *const c_void,
                        bytes,
                        HipMemcpyKind::HostToDevice,
                        stream,
                    )
                };
                if res != crate::hipSuccess {
                    return Err(Error::Backend(format!(
                        "allocate_gdl_layer: H2D upload failed: {res}"
                    )));
                }
                Ok(st)
            };
        let state = upload(
            dev,
            vec![0.0f32; n_qk * dv],
            grim_tensor::Shape::new(vec![batch * heads * dk * dv]),
        )?;
        let alpha = upload(dev, vec![decay; n_qk], grim_tensor::Shape::new(vec![n_qk]))?;
        let b_gate = upload(dev, vec![0.5f32; n_qk], grim_tensor::Shape::new(vec![n_qk]))?;
        let w_gate = upload(dev, vec![0.5f32; n_v], grim_tensor::Shape::new(vec![n_v]))?;
        let norm_w = upload(dev, vec![1.0f32; n_v], grim_tensor::Shape::new(vec![n_v]))?;
        let out_gate = upload(dev, vec![0.0f32; n_g], grim_tensor::Shape::new(vec![n_g]))?;

        // GQA: allocate expanded [nh, hd] K/V buffers when nh != nkv.
        // These receive the head-repeat expansion of the compact [nkv, hd]
        // GEMV output before the fused GDL kernel launch.
        let (k_exp_buf, v_exp_buf) = if heads != kv_heads {
            let n_exp = batch * heads * dk;
            let k_exp = upload(
                dev,
                vec![0.0f32; n_exp],
                grim_tensor::Shape::new(vec![n_exp]),
            )?;
            let v_exp = upload(
                dev,
                vec![0.0f32; n_exp],
                grim_tensor::Shape::new(vec![n_exp]),
            )?;
            (Some(k_exp), Some(v_exp))
        } else {
            (None, None)
        };

        dev.synchronize();
        self.gdl[layer_idx] = Some(GdlLayerBuffers {
            state,
            alpha,
            b_gate,
            w_gate,
            norm_w,
            out_gate,
            k_exp_buf,
            v_exp_buf,
            heads,
            kv_heads,
            dk,
            dv,
            batch,
        });
        Ok(())
    }

    /// Async H2D of `pos` into `pos_dev` on `stream` — broadcast to every
    /// batch slot. Must run BEFORE replay, never inside capture.
    pub fn write_pos_async(
        &mut self,
        dev: &RocmDevice,
        pos: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        self.write_pos_broadcast_async(dev, pos, stream)
    }

    /// Async H2D of `pos` broadcast into every slot of `pos_dev` on `stream`.
    /// `current_pos` host mirror is updated to `pos`.
    pub fn write_pos_broadcast_async(
        &mut self,
        dev: &RocmDevice,
        pos: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        let vals = vec![pos; self.batch.max(1)];
        self.write_pos_slots_raw(dev, &vals, stream)?;
        self.current_pos = pos;
        Ok(())
    }

    /// Reset the replay position state to "empty arena": `current_pos` and
    /// every layer's live-row total go back to 0.
    ///
    /// THREE fields drive a replay and they must move together: the host-side
    /// `current_pos` mirror (which `forward_replay` publishes to each device's
    /// `pos_dev`), and `layer_totals[i]` (the append row index AND the
    /// attention live-row count). `begin_capture` touches none of them, and
    /// every caller warms the pool up by EXECUTING `forward_capture` with
    /// `capturing == false` first — the warmups are what memoize the weight
    /// caches a capture cannot allocate — so without this the first replay
    /// inherits the warmups' bumps and its attention reads rows the warmups
    /// wrote. `write_layer_totals` covers one of the three and had no caller;
    /// a hand-rolled reset re-derives the list at each call site, which is how
    /// a reset that misses one field stays invisible. This is the one place
    /// that knows all of them.
    ///
    /// A caller that HAS a prefill should seed instead: `write_layer_totals`
    /// with the prefill length, plus the KV rows themselves.
    pub fn reset_for_replay(&mut self) -> Result<()> {
        self.write_layer_totals(0)?;
        // `write_pos_async` also updates the `current_pos` host mirror.
        self.write_pos_all_segments(0)
    }

    /// D1-2a: seed EVERY layer's KV total to `rows` (the prefill length),
    /// each on its layer's device. Call after arena seeding and BEFORE the
    /// first replay; in-graph bumps then maintain them per step, and a
    /// per-layer CurDKV compaction resets the ones it compacted.
    pub fn write_layer_totals(&self, rows: u32) -> Result<()> {
        for (i, t) in self.layer_totals.iter().enumerate() {
            let ord = t.device_ordinal();
            let dev = crate::RocmDevice::shared(ord);
            let _g = crate::device::util::DeviceGuard::set(ord as i32);
            let dst = t
                .device_ptr_u64()
                .ok_or_else(|| Error::Backend("write_layer_totals: no ptr".into()))?
                as *mut c_void;
            let vals = vec![rows; self.batch.max(1)];
            let bytes = 4 * self.batch.max(1);
            let host_bytes =
                unsafe { std::slice::from_raw_parts(vals.as_ptr() as *const u8, bytes) };
            let (src_ptr, _pinned) = crate::memory::pinned::stage_h2d_pinned(ord, host_bytes);
            let res: crate::HipErrorT = unsafe {
                hipMemcpyAsync(
                    dst,
                    src_ptr as *const c_void,
                    bytes,
                    HipMemcpyKind::HostToDevice,
                    dev.active_stream(),
                )
            };
            if res != crate::hipSuccess {
                return Err(Error::Backend(format!(
                    "write_layer_totals layer {i}: hipMemcpyAsync {res}"
                )));
            }
        }
        Ok(())
    }

    /// Async H2D of per-slot positions into `pos_dev` (P3: requests in a
    /// decode bucket may sit at different KV positions). `vals.len()` is
    /// clamped to `batch`; missing slots keep their previous device values
    /// (H2D writes them as 0 only when `vals` is shorter — callers should
    /// pass exactly `batch` values).
    pub fn write_pos_slots_async(
        &mut self,
        dev: &RocmDevice,
        vals: &[u32],
        stream: *mut c_void,
    ) -> Result<()> {
        self.write_pos_slots_raw(dev, vals, stream)?;
        if let Some(&last) = vals.last() {
            self.current_pos = last;
        }
        Ok(())
    }

    /// Seed the graph's KV arena from the eager prefill's per-layer K/V caches.
    ///
    /// The eager prefill (run before graph capture) populates device-resident K/V
    /// buffers in the session's `Lfm2LayerCache`. The graph path uses its OWN
    /// fixed arena (`k_arena`/`v_arena`) that starts empty — without seeding,
    /// the captured graph's attention only sees decode tokens and never the
    /// prompt, so greedy decode diverges from eager. This copies the prefill
    /// K/V rows into the arena and sets `current_pos = prefill_len` so the
    /// first decode step appends at the right offset and attends over the full
    /// prompt context.
    ///
    /// `per_layer` is indexed by layer_idx; `None` for non-attention layers.
    /// Each `EagerKvSource` describes the device buffer + valid row count.
    pub fn seed_kv_arena_from_eager(
        &mut self,
        dev: &RocmDevice,
        per_layer: &[Option<EagerKvSource<'_>>],
    ) -> Result<()> {
        let _guard = crate::device::util::DeviceGuard::set(dev.ordinal as i32);
        let stream = dev.active_stream();
        for (layer_idx, src) in per_layer.iter().enumerate() {
            let Some(src) = src else { continue };
            if src.prefill_len == 0 {
                continue;
            }
            if layer_idx >= self.k_arena.len() {
                return Err(Error::Backend(format!(
                    "seed_kv_arena: layer {layer_idx} >= {}",
                    self.k_arena.len()
                )));
            }
            // Phase 3: with a fixed-capacity ring, `prefill_len` is the ABSOLUTE
            // token count while the eager arena only holds the live prefix
            // [0, min(prefill_len, cap)). Both arenas use the same ring policy,
            // so the layouts already agree and seeding stays a single
            // contiguous copy — no gather, and the sinks stay at rows [0, sinks).
            let n_rows = match self.kv_ring.cap_rows {
                cap if cap > 0 => src.prefill_len.min(cap as u32),
                _ => src.prefill_len,
            };
            // The arena row extent is the FORMAT's row width, and both the
            // eager source and the graph arena are built from the same format
            // resolver — so `src.kv_stride` (the eager arena's trailing-dims
            // product) already IS that extent: `nkv*hd` for int8/f16/f32 and
            // the packed `nkv*(hd/2)` for int4. We assert it against the graph
            // arena's last dim and against the arena's real byte count so a
            // format/dtype drift fails loud instead of copying half a row.
            let arena_row_extent: usize = src.kv_stride;
            let n_elem: usize = (n_rows as usize) * arena_row_extent;
            let arena_k_cols = self.k_arena[layer_idx]
                .shape
                .dims()
                .last()
                .copied()
                .unwrap_or(0);
            let arena_v_cols = self.v_arena[layer_idx]
                .shape
                .dims()
                .last()
                .copied()
                .unwrap_or(0);
            if arena_k_cols != arena_row_extent || arena_v_cols != arena_row_extent {
                return Err(Error::Backend(format!(
                    "seed_kv_arena: layer {layer_idx} arena width k={arena_k_cols} v={arena_v_cols} != format row extent {arena_row_extent} (kv_stride {})",
                    src.kv_stride
                )));
            }
            if n_elem > self.k_arena[layer_idx].shape.elem_count() {
                return Err(Error::Backend(format!(
                    "seed_kv_arena: layer {layer_idx} prefill {n_elem} > arena {}",
                    self.k_arena[layer_idx].shape.elem_count()
                )));
            }
            // SAFETY: src.k_dev/src.v_dev are device mem owned by the session
            // caches (valid for the generation); dst is the graph arena. D2D
            // copy on the active stream, ordered vs later launches.
            // Element widths follow the SOURCE (src_elem_bytes) and the ARENA
            // (live flag) independently: equal widths take the raw copy; an
            // f32 eager source seeding an f16 arena converts elementwise
            // (grim_f32_to_f16); anything else (e.g. f16 rows into an f32
            // arena after a mid-run flag downgrade) fails loud. The old code
            // trusted the live flag for both sides and silently copied half
            // the bytes when they disagreed.
            let arena_elem_bytes = match self.k_arena[layer_idx].dtype.arith {
                grim_tensor::ArithType::U8 => 1usize,
                grim_tensor::ArithType::F16 => 2usize,
                grim_tensor::ArithType::F32 => 4usize,
                // Storage truth, like the launchers: only dense u8/f16/f32
                // arenas seed here; packed formats use their own path.
                other => {
                    return Err(Error::Backend(format!(
                        "seed_kv_arena: layer {layer_idx} arena dtype {other:?} unsupported"
                    )));
                }
            };
            let n_bytes = n_elem * arena_elem_bytes;
            debug_assert_eq!(
                n_bytes,
                (n_rows as usize) * arena_row_extent * arena_elem_bytes,
                "seed_kv_arena: layer {layer_idx} byte count must be rows x format row extent"
            );
            debug_assert!(
                n_bytes <= self.k_arena[layer_idx].bytes
                    && n_bytes <= self.v_arena[layer_idx].bytes,
                "seed_kv_arena: layer {layer_idx} seed {n_bytes} B exceeds arena bytes                  (k={}, v={})",
                self.k_arena[layer_idx].bytes,
                self.v_arena[layer_idx].bytes
            );
            let dst_k = self.k_arena[layer_idx]
                .device_ptr_u64()
                .ok_or_else(|| Error::Backend("seed: k_arena has no ptr".into()))?
                as *mut c_void;
            let dst_v = self.v_arena[layer_idx]
                .device_ptr_u64()
                .ok_or_else(|| Error::Backend("seed: v_arena has no ptr".into()))?
                as *mut c_void;
            let src_k = src.k_dev as *const c_void;
            let src_v = src.v_dev as *const c_void;
            // int8 codes arenas: byte-copy the codes, then copy the f32
            // scale sidecars. Requires int8 eager sources — the graph and
            // eager arenas come from the same format resolver, so a mismatch
            // means something reassigned the flag mid-run; fail loud rather
            // than quantize the prompt silently.
            //
            // WhiteCrow (`KV_ARENA_BITS_WHITECROW`) stores its bf16 scale +
            // u8 zero INLINE in each 67-byte block: there is no f32 sidecar on
            // either side, so its codes skip the sidecar block entirely and
            // seed through the raw 1-byte/elem copy below (which also
            // byte-compares eager vs graph).
            if arena_elem_bytes == 1 && self.kv_arena_bits != Some(KV_ARENA_BITS_WHITECROW) {
                if src.src_elem_bytes != 1 {
                    return Err(Error::Backend(format!(
                        "seed_kv_arena: layer {layer_idx} int8 graph arena but eager source is \
                         {}B/elem — both arenas must share one KV format",
                        src.src_elem_bytes
                    )));
                }
                let n_kv = self
                    .k_scale_arena
                    .get(layer_idx)
                    .map(|s| s.shape.dims().get(1).copied().unwrap_or(0))
                    .unwrap_or(0);
                let scale_n = (n_rows as usize) * n_kv;
                for (s_ptr, d_ptr, sc_src, dst_sc, tag) in [
                    (
                        src.k_dev as *const c_void,
                        dst_k,
                        src.k_scales_dev,
                        &self.k_scale_arena,
                        "K",
                    ),
                    (
                        src.v_dev as *const c_void,
                        dst_v,
                        src.v_scales_dev,
                        &self.v_scale_arena,
                        "V",
                    ),
                ] {
                    let sc = sc_src.filter(|p| !p.is_null()).ok_or_else(|| {
                        Error::Backend(format!(
                            "seed_kv_arena: layer {layer_idx} int8 arena but eager {tag} \
                                 scale sidecar is missing"
                        ))
                    })?;
                    let dst_sidecar = dst_sc
                        .get(layer_idx)
                        .filter(|s| s.shape.elem_count() >= scale_n)
                        .and_then(|s| s.device_ptr_u64())
                        .ok_or_else(|| {
                            Error::Backend(format!(
                                "seed_kv_arena: layer {layer_idx} graph {tag} scale sidecar \
                                 missing or too small ({scale_n} scales needed)"
                            ))
                        })?;
                    let src_ord = src.device_ordinal;
                    let dst_ord = self
                        .layer_devices
                        .get(layer_idx)
                        .copied()
                        .unwrap_or(dev.ordinal) as i32;
                    for (dst, bytes, what) in [
                        (d_ptr as *mut c_void, n_bytes, format!("codes {tag}")),
                        (
                            dst_sidecar as *mut c_void,
                            scale_n * 4,
                            format!("scales {tag}"),
                        ),
                    ] {
                        let src_p = if what.starts_with("codes") {
                            s_ptr
                        } else {
                            sc as *const c_void
                        };
                        if src_ord != dst_ord {
                            crate::peer_access::p2p_memcpy_async(
                                dst, dst_ord, src_p, src_ord, bytes, stream,
                            )
                            .map_err(|e| {
                                Error::Backend(format!(
                                    "seed_kv_arena: P2P copy {what} failed: {e}"
                                ))
                            })?;
                        } else {
                            let res: crate::HipErrorT = unsafe {
                                crate::hipMemcpyAsync(
                                    dst,
                                    src_p,
                                    bytes,
                                    HipMemcpyKind::DeviceToDevice,
                                    stream,
                                )
                            };
                            if res != crate::hipSuccess {
                                return Err(Error::Backend(format!(
                                    "seed_kv_arena: memcpy {what} failed: {res}"
                                )));
                            }
                        }
                    }
                }
                continue;
            }
            if src.src_elem_bytes == arena_elem_bytes {
                let dst_ord = dev.ordinal as i32;
                let src_ord = src.device_ordinal;
                if src_ord != dst_ord {
                    // Cross-device P2P copy
                    crate::peer_access::p2p_memcpy_async(
                        dst_k, dst_ord, src_k, src_ord, n_bytes, stream,
                    )
                    .map_err(|e| {
                        Error::Backend(format!("seed_kv_arena: P2P copy K failed: {e}"))
                    })?;

                    crate::peer_access::p2p_memcpy_async(
                        dst_v, dst_ord, src_v, src_ord, n_bytes, stream,
                    )
                    .map_err(|e| {
                        Error::Backend(format!("seed_kv_arena: P2P copy V failed: {e}"))
                    })?;
                } else {
                    let res: crate::HipErrorT = unsafe {
                        crate::hipMemcpyAsync(
                            dst_k,
                            src_k,
                            n_bytes,
                            HipMemcpyKind::DeviceToDevice,
                            stream,
                        )
                    };
                    if res != crate::hipSuccess {
                        return Err(Error::Backend(format!(
                            "seed_kv_arena: hipMemcpyAsync K failed: {res}"
                        )));
                    }
                    let res: crate::HipErrorT = unsafe {
                        crate::hipMemcpyAsync(
                            dst_v,
                            src_v,
                            n_bytes,
                            HipMemcpyKind::DeviceToDevice,
                            stream,
                        )
                    };
                    if res != crate::hipSuccess {
                        return Err(Error::Backend(format!(
                            "seed_kv_arena: hipMemcpyAsync V failed: {res}"
                        )));
                    }
                }
                // Zero out any remaining tail rows in the arena for layer_idx
                let total_k_bytes = self.k_arena[layer_idx].bytes;
                if total_k_bytes > n_bytes {
                    let tail_bytes = total_k_bytes - n_bytes;
                    crate::check_hip("seed_kv_arena hipMemsetAsync K tail", unsafe {
                        crate::device::handles::hipMemsetAsync(
                            (dst_k as *mut u8).add(n_bytes) as *mut std::ffi::c_void,
                            0,
                            tail_bytes,
                            stream,
                        )
                    })?;
                    crate::check_hip("seed_kv_arena hipMemsetAsync V tail", unsafe {
                        crate::device::handles::hipMemsetAsync(
                            (dst_v as *mut u8).add(n_bytes) as *mut std::ffi::c_void,
                            0,
                            tail_bytes,
                            stream,
                        )
                    })?;
                }
                eprintln!(
                    "[seed-debug] layer {layer_idx}: copied {n_bytes} bytes ({n_rows} rows), tail {} bytes zeroed. total_arena={total_k_bytes}",
                    total_k_bytes.saturating_sub(n_bytes)
                );
                // BYTE-COMPARE CHECK: Verify graph arena matches eager arena bit-for-bit on rows 0..n_rows
                crate::check_hip("seed_kv_arena hipStreamSynchronize", unsafe {
                    crate::device::handles::hipStreamSynchronize(stream)
                })?;
                let mut host_eager_k = vec![0u8; n_bytes];
                let mut host_graph_k = vec![0u8; n_bytes];
                let mut host_eager_v = vec![0u8; n_bytes];
                let mut host_graph_v = vec![0u8; n_bytes];
                check_hip("seed_kv_arena D2H eager_k", unsafe {
                    // BLOCKING D2H: a host round-trip that drains the pipeline (see `blocking_d2h_count`).
                    crate::memory::storage::count_blocking_d2h();
                    crate::hipMemcpy(
                        host_eager_k.as_mut_ptr() as *mut std::ffi::c_void,
                        src_k,
                        n_bytes,
                        HipMemcpyKind::DeviceToHost,
                    )
                })?;
                check_hip("seed_kv_arena D2H graph_k", unsafe {
                    // BLOCKING D2H: a host round-trip that drains the pipeline (see `blocking_d2h_count`).
                    crate::memory::storage::count_blocking_d2h();
                    crate::hipMemcpy(
                        host_graph_k.as_mut_ptr() as *mut std::ffi::c_void,
                        dst_k,
                        n_bytes,
                        HipMemcpyKind::DeviceToHost,
                    )
                })?;
                check_hip("seed_kv_arena D2H eager_v", unsafe {
                    // BLOCKING D2H: a host round-trip that drains the pipeline (see `blocking_d2h_count`).
                    crate::memory::storage::count_blocking_d2h();
                    crate::hipMemcpy(
                        host_eager_v.as_mut_ptr() as *mut std::ffi::c_void,
                        src_v,
                        n_bytes,
                        HipMemcpyKind::DeviceToHost,
                    )
                })?;
                check_hip("seed_kv_arena D2H graph_v", unsafe {
                    // BLOCKING D2H: a host round-trip that drains the pipeline (see `blocking_d2h_count`).
                    crate::memory::storage::count_blocking_d2h();
                    crate::hipMemcpy(
                        host_graph_v.as_mut_ptr() as *mut std::ffi::c_void,
                        dst_v,
                        n_bytes,
                        HipMemcpyKind::DeviceToHost,
                    )
                })?;
                let k_diff = host_eager_k
                    .iter()
                    .zip(host_graph_k.iter())
                    .filter(|(a, b)| a != b)
                    .count();
                let v_diff = host_eager_v
                    .iter()
                    .zip(host_graph_v.iter())
                    .filter(|(a, b)| a != b)
                    .count();
                // Check for NaNs in rows 0..n_rows
                let k_u16: &[u16] = unsafe {
                    std::slice::from_raw_parts(host_graph_k.as_ptr() as *const u16, n_bytes / 2)
                };
                let v_u16: &[u16] = unsafe {
                    std::slice::from_raw_parts(host_graph_v.as_ptr() as *const u16, n_bytes / 2)
                };
                let k_nans = k_u16
                    .iter()
                    .filter(|&&x| (x & 0x7c00) == 0x7c00 && (x & 0x03ff) != 0)
                    .count();
                let v_nans = v_u16
                    .iter()
                    .filter(|&&x| (x & 0x7c00) == 0x7c00 && (x & 0x03ff) != 0)
                    .count();
                eprintln!(
                    "[seed-verify] layer {layer_idx}: k_diff_bytes={k_diff}/{n_bytes}, v_diff_bytes={v_diff}/{n_bytes}, k_nans={k_nans}, v_nans={v_nans}"
                );
                if k_nans > 0 {
                    for (idx, &val) in k_u16.iter().enumerate() {
                        if (val & 0x7c00) == 0x7c00 && (val & 0x03ff) != 0 {
                            let row = idx / 1024;
                            let col = idx % 1024;
                            eprintln!(
                                "[seed-nan-k] layer {layer_idx}: row={row} col={col} val=0x{val:04x}"
                            );
                            if row > 0 {
                                break;
                            }
                        }
                    }
                }
                continue;
            }
            if !(src.src_elem_bytes == 4 && arena_elem_bytes == 2) {
                return Err(Error::Backend(format!(
                    "seed_kv_arena: layer {layer_idx} source width {}B vs arena {}B — \
                     only f32->f16 seeding is supported (downgrades fail loud)",
                    src.src_elem_bytes, arena_elem_bytes
                )));
            }
            // f32 eager rows -> f16 arena: elementwise convert, K then V.
            let (grid, block) = linear_launch(n_elem);
            for (s_ptr, d_ptr) in [(src_k, dst_k), (src_v, dst_v)] {
                let mut s = s_ptr as u64;
                let mut d = d_ptr as u64;
                let mut src_off = 0i32;
                let mut dst_off = 0i32;
                let mut n = n_elem as i32;
                dev.launch_compute_kernel(
                    "grim_f32_to_f16",
                    grid,
                    block,
                    &mut [
                        arg(&mut s),
                        arg(&mut d),
                        arg(&mut src_off),
                        arg(&mut dst_off),
                        arg(&mut n),
                    ],
                )?;
            }
        }
        // Set the append position to the end of the seeded prefill so the first
        // decode step appends at the right offset and attends over the prompt.
        let prefill_len = per_layer
            .iter()
            .find_map(|s| s.as_ref().map(|e| e.prefill_len))
            .unwrap_or(0);
        self.current_pos = prefill_len;
        // Seed the device position scalar to match; the graph's bump kernel
        // increments it on each replay, so it must start at prefill_len.
        self.write_pos_async(dev, prefill_len, stream)?;
        dev.synchronize();
        Ok(())
    }

    /// GDL state seeding: D2D copy each layer's eager recurrent `dev_state`
    /// into the graph's `gdl.state` buffer. Without this the graph replays
    /// GDL layers against a zeroed state after any prompt prefill — decode
    /// diverges from eager. Must run OUTSIDE a capture bracket (blocking H2D).
    /// `per_layer` is indexed by layer_idx; `None` for non-GDL layers.
    pub fn seed_gdl_state_from_eager(
        &mut self,
        dev: &RocmDevice,
        per_layer: &[Option<EagerKvSource<'_>>],
    ) -> Result<()> {
        for (layer_idx, src) in per_layer.iter().enumerate() {
            let Some(src) = src else { continue };
            let Some(state_ptr) = src.gdl_state else {
                continue;
            };
            // A5: guard + stream on the LAYER's device, not the graph's
            // primary — a split model's layer-i buffers live on
            // layer_devices[i], and issuing the copy from the primary
            // device's context is the seeder fault class this task exists
            // for (hipMemcpyAsync on foreign-device pointers).
            let ord = self
                .layer_devices
                .get(layer_idx)
                .copied()
                .unwrap_or(dev.ordinal);
            let _guard = crate::device::util::DeviceGuard::set(ord as i32);
            let stream = crate::RocmDevice::shared(ord).active_stream();
            let Some(gdl) = self.gdl.get(layer_idx).and_then(|o| o.as_ref()) else {
                return Err(Error::Backend(format!(
                    "seed_gdl_state: layer {layer_idx} has no GDL buffers"
                )));
            };
            let n_elem = gdl.state.shape().elem_count();
            if n_elem == 0 {
                return Err(Error::Backend(format!(
                    "seed_gdl_state: layer {layer_idx} GDL state buffer is empty"
                )));
            }
            let bytes = n_elem * 4; // f32
            let dst = gdl
                .state
                .device_ptr_u64()
                .ok_or_else(|| Error::Backend("seed_gdl_state: no dst ptr".into()))?
                as *mut c_void;
            let src_ptr = state_ptr as *const c_void;
            let res: crate::HipErrorT = unsafe {
                crate::hipMemcpyAsync(dst, src_ptr, bytes, HipMemcpyKind::DeviceToDevice, stream)
            };
            if res != crate::hipSuccess {
                return Err(Error::Backend(format!(
                    "seed_gdl_state: hipMemcpyAsync failed: {res}"
                )));
            }
        }
        dev.synchronize();
        Ok(())
    }

    /// Device conv-ring seeding: D2D copy each layer's live `conv_state_dev`
    /// ring into the graph's `sc_state` (identical `[channel][tap]` layout).
    /// Use when the eager prefill ran the D2D conv — the host mirror is stale
    /// in that case. Must run OUTSIDE a capture bracket.
    pub fn seed_conv_rings_from_device(
        &mut self,
        dev: &RocmDevice,
        per_layer: &[Option<ConvDeviceSeed<'_>>],
    ) -> Result<()> {
        for (layer_idx, seed) in per_layer.iter().enumerate() {
            let Some(sd) = seed else { continue };
            // A5: per-layer device context (see seed_gdl_state_from_eager).
            let ord = self
                .layer_devices
                .get(layer_idx)
                .copied()
                .unwrap_or(dev.ordinal);
            let _guard = crate::device::util::DeviceGuard::set(ord as i32);
            let stream = crate::RocmDevice::shared(ord).active_stream();
            if layer_idx >= self.sc_state.len() {
                return Err(Error::Backend(format!(
                    "seed_conv_rings_from_device: layer {layer_idx} >= {} sc rings",
                    self.sc_state.len()
                )));
            }
            let n_elem = self.sc_state[layer_idx].shape.elem_count();
            if n_elem == 0 {
                continue;
            }
            let dst = self.sc_state[layer_idx]
                .device_ptr_u64()
                .ok_or_else(|| Error::Backend("seed_conv_rings_from_device: no dst ptr".into()))?
                as *mut c_void;
            let res: crate::HipErrorT = unsafe {
                crate::hipMemcpyAsync(
                    dst,
                    sd.dev_ring as *const c_void,
                    n_elem * 4,
                    HipMemcpyKind::DeviceToDevice,
                    stream,
                )
            };
            if res != crate::hipSuccess {
                return Err(Error::Backend(format!(
                    "seed_conv_rings_from_device: hipMemcpyAsync failed: {res}"
                )));
            }
        }
        dev.synchronize();
        Ok(())
    }

    /// KDA recurrent state seeding: D2D copy each layer's eager recurrent `ssm_state_dev`
    /// into the graph's `kda.state` buffer. Must run OUTSIDE a capture bracket.
    pub fn seed_kda_state_from_eager(
        &mut self,
        dev: &RocmDevice,
        per_layer: &[Option<EagerKdaSource<'_>>],
    ) -> Result<()> {
        for (layer_idx, src) in per_layer.iter().enumerate() {
            let Some(src) = src else { continue };
            let state_ptr = src.kda_state;
            // A5: per-layer device context (see seed_gdl_state_from_eager).
            let ord = self
                .layer_devices
                .get(layer_idx)
                .copied()
                .unwrap_or(dev.ordinal);
            let _guard = crate::device::util::DeviceGuard::set(ord as i32);
            let stream = crate::RocmDevice::shared(ord).active_stream();
            let Some(kda) = self.kda.get(layer_idx).and_then(|o| o.as_ref()) else {
                continue;
            };
            let n_elem = kda.state.shape().elem_count();
            if n_elem == 0 {
                continue;
            }
            let bytes = n_elem * 4; // f32
            let dst = kda
                .state
                .device_ptr_u64()
                .ok_or_else(|| Error::Backend("seed_kda_state: no dst ptr".into()))?
                as *mut c_void;
            let src_ptr = state_ptr as *const c_void;
            let res: crate::HipErrorT = unsafe {
                crate::hipMemcpyAsync(dst, src_ptr, bytes, HipMemcpyKind::DeviceToDevice, stream)
            };
            if res != crate::hipSuccess {
                return Err(Error::Backend(format!(
                    "seed_kda_state: hipMemcpyAsync failed: {res}"
                )));
            }
        }
        dev.synchronize();
        Ok(())
    }

    /// Prefill seeding for recurrent (ShortConv) layers: upload each layer's
    /// host conv ring (transposed to the kernel's `[d, kc]` layout) into
    /// `sc_state`. Without this the graph replays conv layers against a
    /// ZEROED ring after any prompt prefill — greedy decode diverges from
    /// eager. Must run OUTSIDE a capture bracket (blocking H2D).
    /// `per_layer` is indexed by layer_idx; `None` for attention layers.
    pub fn seed_conv_rings(&mut self, per_layer: &[Option<ConvRingSeed<'_>>]) -> Result<()> {
        for (layer_idx, seed) in per_layer.iter().enumerate() {
            let Some(s) = seed else { continue };
            if layer_idx >= self.sc_state.len() {
                return Err(Error::Backend(format!(
                    "seed_conv_rings: layer {layer_idx} >= {} sc rings",
                    self.sc_state.len()
                )));
            }
            let expect = s.h_dim * s.kc;
            if s.host.len() != expect {
                return Err(Error::Backend(format!(
                    "seed_conv_rings: layer {layer_idx} host ring {} != h_dim*kc {expect}",
                    s.host.len()
                )));
            }
            if self.sc_state[layer_idx].shape.elem_count() != expect {
                return Err(Error::Backend(format!(
                    "seed_conv_rings: layer {layer_idx} sc ring {} != {expect}",
                    self.sc_state[layer_idx].shape.elem_count()
                )));
            }
            let mut col_major = vec![0.0f32; expect];
            for t in 0..s.kc {
                for d in 0..s.h_dim {
                    col_major[d * s.kc + t] = s.host[t * s.h_dim + d];
                }
            }
            self.sc_state[layer_idx]
                .write_host_f32(&col_major)
                .map_err(|e| {
                    Error::Backend(format!("seed_conv_rings: H2D layer {layer_idx}: {e}"))
                })?;
        }
        Ok(())
    }

    /// Prefill seeding for MLA compressed latent KV arenas (`latent_kv_arena`).
    /// `src.k_dev` points to the `[prefill_len, latent_kv_dim]` latent tensor.
    pub fn seed_latent_kv_arena_from_eager(
        &mut self,
        dev: &RocmDevice,
        per_layer: &[Option<EagerKvSource<'_>>],
    ) -> Result<()> {
        let _guard = crate::device::util::DeviceGuard::set(dev.ordinal as i32);
        let stream = dev.active_stream();
        for (layer_idx, src) in per_layer.iter().enumerate() {
            let Some(src) = src else { continue };
            if src.prefill_len == 0 {
                continue;
            }
            if layer_idx >= self.latent_kv_arena.len() {
                return Err(Error::Backend(format!(
                    "seed_latent_kv_arena: layer {layer_idx} >= {}",
                    self.latent_kv_arena.len()
                )));
            }
            let n_rows = src.prefill_len;
            let n_elem: usize = (n_rows as usize) * src.kv_stride;
            let arena_cols = self.latent_kv_arena[layer_idx]
                .shape
                .dims()
                .last()
                .copied()
                .unwrap_or(0);
            if arena_cols != src.kv_stride {
                return Err(Error::Backend(format!(
                    "seed_latent_kv_arena: layer {layer_idx} arena width {arena_cols} != latent stride {}",
                    src.kv_stride
                )));
            }
            if n_elem > self.latent_kv_arena[layer_idx].shape.elem_count() {
                return Err(Error::Backend(format!(
                    "seed_latent_kv_arena: layer {layer_idx} prefill {n_elem} > arena {}",
                    self.latent_kv_arena[layer_idx].shape.elem_count()
                )));
            }
            let bytes = n_elem * std::mem::size_of::<f32>();
            let dst_latent = self.latent_kv_arena[layer_idx]
                .device_ptr_u64()
                .ok_or_else(|| Error::Backend("seed: latent_kv_arena has no ptr".into()))?
                as *mut c_void;
            let src_latent = src.k_dev as *const c_void;
            let res: crate::HipErrorT = unsafe {
                crate::hipMemcpyAsync(
                    dst_latent,
                    src_latent,
                    bytes,
                    HipMemcpyKind::DeviceToDevice,
                    stream,
                )
            };
            if res != crate::hipSuccess {
                return Err(Error::Backend(format!(
                    "seed_latent_kv_arena: hipMemcpyAsync failed: {res}"
                )));
            }
        }
        let prefill_len = per_layer
            .iter()
            .find_map(|s| s.as_ref().map(|e| e.prefill_len))
            .unwrap_or(0);
        self.current_pos = prefill_len;
        self.write_pos_async(dev, prefill_len, stream)?;
        dev.synchronize();
        Ok(())
    }

    /// Raw per-slot write; does not touch the host position mirror.
    fn write_pos_slots_raw(
        &mut self,
        dev: &RocmDevice,
        vals: &[u32],
        stream: *mut c_void,
    ) -> Result<()> {
        let _guard = crate::device::util::DeviceGuard::set(dev.ordinal as i32);
        // A5b: the replica of THIS device (== pos_dev itself on the primary).
        let dst = self
            .pos_dev_for(dev.ordinal)
            .device_ptr_u64()
            .ok_or_else(|| Error::Backend("pos_dev has no device ptr".into()))?
            as *mut c_void;
        let mut buf = vals.to_vec();
        buf.resize(self.batch.max(1), 0);
        let bytes = 4 * self.batch.max(1);
        // Stage through pinned memory (see memory::pinned::stage_h2d_pinned):
        // pageable H2D copies take the driver's slow staged path and dominated
        // per-token decode time.
        let host_bytes = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, bytes) };
        let (src_ptr, _pinned) = crate::memory::pinned::stage_h2d_pinned(dev.ordinal, host_bytes);
        // SAFETY: dst is device mem owned by pos_dev (4*batch bytes); src is
        // the pinned stage (or the original buf on fallback) with exactly
        // `bytes` valid bytes, valid per stage_h2d_pinned's safety contract.
        let res: crate::HipErrorT = unsafe {
            hipMemcpyAsync(
                dst,
                src_ptr as *const c_void,
                bytes,
                HipMemcpyKind::HostToDevice,
                stream,
            )
        };
        if res != crate::hipSuccess {
            return Err(Error::Backend(format!(
                "write_pos_async hipMemcpyAsync failed: {res}"
            )));
        }
        Ok(())
    }

    /// A5b: the pos scalar of device `ord` — the original `pos_dev` on the
    /// allocating device, the matching replica elsewhere.
    pub fn pos_dev_for(&self, ord: usize) -> &RocmStorage {
        self.pos_replica_ords
            .iter()
            .position(|&o| o == ord)
            .map(|i| &self.pos_replicas[i])
            .unwrap_or(&self.pos_dev)
    }

    /// A5b: the attn_dummy of device `ord` (read-only dummy).
    pub fn attn_dummy_for(&self, ord: usize) -> &RocmStorage {
        self.attn_dummy_replica_ords
            .iter()
            .position(|&o| o == ord)
            .map(|i| &self.attn_dummy_replicas[i])
            .unwrap_or(&self.attn_dummy)
    }

    /// A5b: write `pos` into EVERY device's pos scalar, each on its own
    /// device's stream, so every segment's replay starts at the same value
    /// (the A0-green pre-write design). Must run OUTSIDE any capture bracket.
    pub fn write_pos_all_segments(&mut self, pos: u32) -> Result<()> {
        let primary = self.pos_dev.device_ordinal();
        let mut ords: Vec<usize> = vec![primary];
        ords.extend(self.pos_replica_ords.iter().copied());
        for ord in ords {
            let dev = crate::RocmDevice::shared(ord);
            let stream = dev.active_stream();
            self.write_pos_async(&dev, pos, stream)?;
        }
        Ok(())
    }
}

/// Spec §Phase 2: wraps HIP graph state + fixed buffers.
#[derive(Debug)]
pub struct DecodeGraph {
    pub graph: *mut c_void,
    pub exec: *mut c_void,
    pub stream: *mut c_void,
    pub buffers: DecodeGraphBuffers,
    pub is_captured: bool,
    /// True between `begin_capture` and `end_capture`/`abort_capture`.
    /// Capture-time callers MUST NOT enqueue H2D writes of host-owned data:
    /// an async H2D inside capture bakes the (dead) host pointer into a graph
    /// memcpy node read at replay. Seed inputs are written eagerly before
    /// `begin_capture` or per-replay from `forward_replay` (outside capture).
    pub capturing: bool,
    /// Kernel node whose scalar args change per step (KV append). Null ->
    /// use `pos_dev` + async memcpy path instead.
    pub kv_append_node: *mut c_void,
    /// One captured graph per contiguous device run. EMPTY = the legacy
    /// single-graph path (graph/exec/stream above); only `set_layer_devices`
    /// populates it, and a split placement yields one segment per run.
    /// Capture contract: `begin_capture` does today's arm work on segment 0
    /// (fp8-pad publish, open + gate); the model's `forward_capture` advances
    /// each interior boundary (`end_capture_segment(k)` ->
    /// `begin_capture_segment(k+1)`); `end_capture` closes the last open
    /// segment; `abort_capture` ends whichever segment is open;
    /// `replay()` delegates to `replay_chain()`.
    pub segments: Vec<GraphSegment>,
    /// One event PER BOUNDARY (index = producing segment ordinal), each
    /// created under a DeviceGuard on the producing device. Task A4.
    pub boundary_evs: Vec<*mut std::ffi::c_void>,
    /// M6: one handoff PER BOUNDARY (index = producing segment ordinal, the
    /// same indexing as `boundary_evs`).
    ///
    /// A handoff is the producer's ring (carrying the boundary's PEER_COPY
    /// tasks) plus the consumer's completion flag. When it is present,
    /// `replay_chain` uses it INSTEAD of A6's host-issued peer copy + event:
    /// the producer's wave does the copy and the consumer's DEVICE spins on
    /// the flag, so the host issues no boundary synchronisation at all.
    ///
    /// EMPTY means the A6 path — the default, unless `GRIM_RING_HANDOFF=1`.
    pub boundary_handoffs: Vec<Option<crate::peer_access::BoundaryHandoff>>,
    /// M1/M2: the task-graph emitter, Some only when
    /// `GRIM_EMIT_TASK_GRAPH=1`. `forward_capture` pushes nodes into it; the
    /// finished IR is published to `buffers.emitted_task_graph` at
    /// `end_capture`.
    pub emitter: Option<crate::task_graph::TaskGraphBuilder>,
    /// M4/M5: persistent task ring context built from the emitted task graph.
    pub ring_context: Option<crate::kernels::task_ring::TaskRingContext>,
    /// M5: device addresses of the per-layer `KdaTaskDescriptor` the qwen35
    /// emitter puts on each RECURRENT node. Indexed by LAYER (None for attention
    /// layers). Staged OUTSIDE capture — an H2D copy inside the capture bracket
    /// poisons the graph (measured: `hipMemcpyHostToDevice ... error code 906`).
    pub kda_desc_ptrs: Vec<Option<u64>>,
    /// WI-4: per-boundary host staging buffer owned by this decode graph.
    pub boundary_staging: Vec<Option<crate::p2p_route::HostStagingBuffer>>,
    /// WI-0: per-segment timing event pairs (start, stop) under GRIM_SEGMENT_TIMING=1.
    pub segment_timing_evs: Vec<(*mut std::ffi::c_void, *mut std::ffi::c_void)>,
    ordinal: usize,
}

/// One device's slice of a split capture: its own graph, exec, and — the part
/// that matters — its own stream, because a graph must be recorded and
/// launched on the device whose kernels it holds.
#[derive(Debug)]
pub struct GraphSegment {
    pub device: usize,
    pub layers: std::ops::Range<usize>,
    pub graph: *mut c_void,
    pub exec: *mut c_void,
    pub stream: *mut c_void,
}

// SAFETY: raw HIP handles owned by self, only touched on owning device.
unsafe impl Send for DecodeGraph {}
unsafe impl Sync for DecodeGraph {}

/// Push one node into a caller-held emitter (free fn so a model that holds a
/// `&mut graph.buffers` reborrow across its capture walk can still emit: the
/// emitter and the buffers are DISJOINT fields of `DecodeGraph`, and only a
/// free fn can borrow them separately).
pub fn emit_node_into(
    emitter: &mut Option<crate::task_graph::TaskGraphBuilder>,
    opcode: crate::task_graph::OpCode,
    params: [u32; 16],
    args: &[(u64, u64, u8)],
    deps: &[u32],
    dev_ord: u8,
) -> u32 {
    let Some(b) = emitter.as_mut() else {
        return u32::MAX;
    };
    let handles: Vec<crate::task_graph::ArgId> = args
        .iter()
        .map(|&(ptr, bytes, role)| b.push_arg(ptr, bytes, role))
        .collect();
    b.push(opcode, params, &handles, deps, dev_ord, 0)
}

impl DecodeGraph {
    pub fn new(dev: &RocmDevice, buffers: DecodeGraphBuffers, stream: *mut c_void) -> Self {
        Self {
            graph: std::ptr::null_mut(),
            exec: std::ptr::null_mut(),
            stream,
            buffers,
            is_captured: false,
            capturing: false,
            kv_append_node: std::ptr::null_mut(),
            segments: Vec::new(),
            boundary_evs: Vec::new(),
            emitter: if std::env::var("GRIM_EMIT_TASK_GRAPH").is_ok() {
                Some(crate::task_graph::TaskGraphBuilder::new())
            } else {
                None
            },
            ring_context: None,
            kda_desc_ptrs: Vec::new(),
            boundary_handoffs: Vec::new(),
            boundary_staging: Vec::new(),
            segment_timing_evs: Vec::new(),
            ordinal: dev.ordinal,
        }
    }

    /// Recompute the per-device segments from the layer placement. Every
    /// segment gets its OWN device's pooled stream — a graph must be captured
    /// and launched on its device's stream, and mixing them is the fault this
    /// whole feature exists to avoid. Call BEFORE `begin_capture`; until this
    /// runs the graph stays on the legacy single-graph path.
    pub fn set_layer_devices(&mut self, ords: &[usize]) {
        self.segments = device_segments(ords)
            .into_iter()
            .map(|(device, layers)| {
                let d = crate::device::roc_device::RocmDevice::shared(device);
                let stream = d
                    .get_stream_from_pool(0)
                    .unwrap_or_else(|| d.active_stream());
                GraphSegment {
                    device,
                    layers,
                    graph: std::ptr::null_mut(),
                    exec: std::ptr::null_mut(),
                    stream,
                }
            })
            .collect();
        // One event per boundary, created under a DeviceGuard on the PRODUCING
        // device (an event is a device-side object; hipStreamWaitEvent from
        // another device's stream must reference one from the producing side).
        let disable_timing = matches!(
            std::env::var("GRIM_BOUNDARY_DISABLE_TIMING").as_deref(),
            Ok("1" | "true" | "on")
        ) || std::env::var_os("GRIM_BOUNDARY_OWNED_STAGING").is_some();
        self.boundary_evs = self.segments[..self.segments.len().saturating_sub(1)]
            .iter()
            .map(|s| {
                let _guard = crate::device::util::DeviceGuard::set(s.device as i32);
                let mut ev: *mut std::ffi::c_void = std::ptr::null_mut();
                let res: crate::HipErrorT = if disable_timing {
                    unsafe { hipEventCreateWithFlags(&mut ev, 0x2) }
                } else {
                    unsafe { crate::hipEventCreate(&mut ev) }
                };
                if res != crate::hipSuccess {
                    eprintln!("[decode-graph] hipEventCreate/WithFlags failed: {res}");
                    return std::ptr::null_mut();
                }
                ev
            })
            .collect();

        // WI-4: allocate graph-owned host staging buffers for each boundary.
        // Enabled by default unless GRIM_BOUNDARY_OWNED_STAGING=0 is set, or if GRIM_BOUNDARY_OWNED_STAGING=1.
        let use_owned_staging = match std::env::var("GRIM_BOUNDARY_OWNED_STAGING").as_deref() {
            Ok("0" | "false" | "off") => false,
            _ => true,
        };
        let n_boundaries = self.segments.len().saturating_sub(1);
        self.boundary_staging = Vec::with_capacity(n_boundaries);
        if use_owned_staging {
            for k in 0..n_boundaries {
                let bytes = self
                    .buffers
                    .boundary_out
                    .get(k)
                    .map(|b| b.bytes())
                    .unwrap_or(0);
                if bytes > 0 {
                    let _guard =
                        crate::device::util::DeviceGuard::set(self.segments[k].device as i32);
                    match crate::p2p_route::HostStagingBuffer::with_flags(bytes, 0x1) {
                        Ok(buf) => self.boundary_staging.push(Some(buf)),
                        Err(e) => {
                            eprintln!(
                                "[decode-graph] boundary {k}: failed to allocate owned staging buffer ({e}); falling back to global"
                            );
                            self.boundary_staging.push(None);
                        }
                    }
                } else {
                    self.boundary_staging.push(None);
                }
            }
        } else {
            self.boundary_staging.resize_with(n_boundaries, || None);
        }

        // Clean up any previous timing events
        for (start, stop) in self.segment_timing_evs.drain(..) {
            if !start.is_null() {
                unsafe {
                    let _ = crate::hipEventDestroy(start);
                }
            }
            if !stop.is_null() {
                unsafe {
                    let _ = crate::hipEventDestroy(stop);
                }
            }
        }
        if matches!(
            std::env::var("GRIM_SEGMENT_TIMING").as_deref(),
            Ok("1" | "true" | "on")
        ) {
            for s in &self.segments {
                let _guard = crate::device::util::DeviceGuard::set(s.device as i32);
                let mut start: *mut std::ffi::c_void = std::ptr::null_mut();
                let mut stop: *mut std::ffi::c_void = std::ptr::null_mut();
                unsafe {
                    let _ = crate::hipEventCreate(&mut start);
                    let _ = crate::hipEventCreate(&mut stop);
                }
                self.segment_timing_evs.push((start, stop));
            }
        }

        self.build_boundary_handoffs();
    }

    /// M6: build one handoff per device boundary, when the handoff is enabled.
    ///
    /// The handoff replaces A6's host-issued peer copy + cross-stream event
    /// with a wave launch (the producer's copy) and a device-side wait (the
    /// consumer's flag). It needs peer access granted between the two
    /// devices — the wave's peer arm writes the consumer's memory DIRECTLY,
    /// which is a different mechanism from the copy engine's
    /// `hipMemcpyPeerAsync` — so the grant is taken here, once, at setup.
    ///
    /// OFF unless `GRIM_RING_HANDOFF=1`: the grant changes the process's
    /// device topology, and its interaction with the rest of the ring is not
    /// yet understood (see the plan's M6 note).
    fn build_boundary_handoffs(&mut self) {
        self.boundary_handoffs = Vec::new();
        if !crate::peer_access::ring_handoff_enabled() || self.segments.len() < 2 {
            return;
        }
        let n_boundaries = self.segments.len() - 1;
        // The handoff's element count: the boundary buffer's floats, capped by
        // the destination's — the same min A6's `copy_boundary_ordered` takes.
        for k in 0..n_boundaries {
            let Some(src) = self.buffers.boundary_out.get(k) else {
                return;
            };
            let consumer_layers = &self.segments[k + 1].layers;
            let Some(dst) = self.buffers.layer_input.get(consumer_layers.start) else {
                return;
            };
            let elems = (src.bytes().min(dst.bytes()) / 4) as u32;
            if elems == 0 {
                return;
            }
            let producer_ord = self.segments[k].device;
            let consumer_ord = self.segments[k + 1].device;
            if producer_ord == consumer_ord {
                return;
            }
            // BOTH grants, and each is load-bearing:
            //
            //  * producer -> consumer: the wave's peer arm writes the
            //    activation (and the flag) into the CONSUMER's memory. A direct
            //    peer write needs the grant; the copy engine's
            //    `hipMemcpyPeerAsync` does not, which is why A6's boundary
            //    never asked for it.
            //  * consumer -> producer: the consumer's DEVICE polls the
            //    producer's tail over peer access, and a peer READ needs the
            //    grant in the other direction.
            //
            // Missing the second one is a page fault on the consumer, not a
            // wrong answer.
            for (a, b) in [
                (producer_ord as i32, consumer_ord as i32),
                (consumer_ord as i32, producer_ord as i32),
            ] {
                if let Err(e) = crate::peer_access::enable_peer_access(a, b) {
                    eprintln!(
                        "[decode-graph] boundary {k}: peer access {a}->{b} unavailable ({e}); \
                         keeping the A6 boundary"
                    );
                    return;
                }
            }
            let producer = crate::RocmDevice::shared(producer_ord);
            let consumer = crate::RocmDevice::shared(consumer_ord);
            match crate::peer_access::BoundaryHandoff::prepare(
                &producer, &consumer, src, dst, elems,
            ) {
                Ok(h) => self.boundary_handoffs.push(Some(h)),
                Err(e) => {
                    eprintln!(
                        "[decode-graph] boundary {k}: handoff preparation failed ({e}); \
                         keeping the A6 boundary"
                    );
                    return;
                }
            }
        }
    }

    /// True when every boundary has a handoff, so `replay_chain` can use them.
    fn use_boundary_handoffs(&self) -> bool {
        !self.boundary_handoffs.is_empty()
            && self.boundary_handoffs.len() == self.segments.len().saturating_sub(1)
            && self.boundary_handoffs.iter().all(|h| h.is_some())
    }

    fn split_capture(&self) -> bool {
        self.segments.len() > 1
    }

    /// M1/M2: push one node into the emitter when emission is enabled.
    /// No-op otherwise — production paths pay an `Option` check, nothing more.
    /// `dev` is the ordinal the node's kernels run on.
    pub fn emit_node(
        &mut self,
        opcode: crate::task_graph::OpCode,
        params: [u32; 16],
        args: &[(u64, u64, u8)],
        deps: &[u32],
        dev_ord: u8,
    ) -> u32 {
        emit_node_into(&mut self.emitter, opcode, params, args, deps, dev_ord)
    }

    /// Publish the finished IR (called from `end_capture`).
    fn publish_task_graph(&mut self) {
        if let Some(b) = self.emitter.take() {
            let ir = b.finish();
            // The ring is an EXECUTION choice, not an emission side effect.
            // `GRIM_EMIT_TASK_GRAPH=1` only asks capture to publish the IR for
            // inspection; building the ring here unconditionally made that debug
            // flag silently reroute `forward_replay` through the ring, because
            // Step 4 keys on `ring_context.is_some()`. The ring is built only
            // under its own flag, so turning emission on cannot change what runs.
            if std::env::var_os("GRIM_TASK_RING").is_some() {
                let dev = crate::device::roc_device::RocmDevice::shared(self.ordinal);
                if let Ok(ctx) = crate::kernels::task_ring::TaskRingContext::from_graph(&dev, &ir) {
                    self.ring_context = ctx;
                }
            }
            self.buffers.emitted_task_graph = Some(ir);
        }
    }

    /// Open segment `seg`'s capture bracket: begin capture on ITS stream under
    /// ITS device's guard, and raise THIS segment device's capture gate
    /// pointing at THIS segment's stream — `active_stream()` on that device
    /// then returns the capturing stream, so plain launches route into the
    /// graph. Raising it on a foreign device is what routed foreign kernels
    /// into a graph and faulted on replay.
    pub fn begin_capture_segment(&mut self, seg: usize) -> Result<()> {
        let s = &self.segments[seg];
        let _guard = crate::device::util::DeviceGuard::set(s.device as i32);
        // SAFETY: stream owned by that device's pool, valid for capture duration.
        let res: crate::HipErrorT = unsafe { hipStreamBeginCapture(s.stream, 2) };
        if res != crate::hipSuccess {
            return Err(Error::Backend(format!(
                "hipStreamBeginCapture(seg {seg}) failed: {res}"
            )));
        }
        self.capturing = true;
        crate::device::roc_device::RocmDevice::shared(s.device)
            .set_capture_gate(true, Some(s.stream));
        Ok(())
    }

    /// Close segment `seg`'s bracket: end capture, lower its gate, instantiate
    /// and upload ITS graph on ITS stream.
    pub fn end_capture_segment(&mut self, seg: usize) -> Result<()> {
        {
            let s = &self.segments[seg];
            let _guard = crate::device::util::DeviceGuard::set(s.device as i32);
            let mut graph: *mut c_void = std::ptr::null_mut();
            // SAFETY: stream is under capture; graph out-ptr valid.
            let res: crate::HipErrorT = unsafe { hipStreamEndCapture(s.stream, &mut graph) };
            crate::device::roc_device::RocmDevice::shared(s.device).set_capture_gate(false, None);
            if res != crate::hipSuccess {
                self.capturing = false;
                return Err(Error::Backend(format!(
                    "hipStreamEndCapture(seg {seg}) failed: {res}"
                )));
            }
            let mut exec: *mut c_void = std::ptr::null_mut();
            // SAFETY: graph just captured, exec out-ptr valid.
            let inst: crate::HipErrorT = unsafe {
                hipGraphInstantiate(
                    &mut exec,
                    graph,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    0,
                )
            };
            if inst != crate::hipSuccess {
                self.capturing = false;
                unsafe {
                    let _ = hipGraphDestroy(graph);
                }
                return Err(Error::Backend(format!(
                    "hipGraphInstantiate(seg {seg}) failed: {inst}"
                )));
            }
            let up = unsafe { crate::hipGraphUpload(exec, s.stream) };
            if up != crate::hipSuccess {
                self.capturing = false;
                unsafe {
                    let _ = hipGraphExecDestroy(exec);
                    let _ = hipGraphDestroy(graph);
                }
                return Err(Error::Backend(format!(
                    "hipGraphUpload(seg {seg}) failed: {up}"
                )));
            }
            let s = &mut self.segments[seg];
            s.graph = graph;
            s.exec = exec;
        }
        // Only the LAST segment's close ends the global capturing window;
        // interior boundaries immediately open the next bracket.
        if seg + 1 >= self.segments.len() {
            self.capturing = false;
        }
        Ok(())
    }

    /// Launch segment `seg`'s graph on its own device + stream.
    pub fn replay_segment(&self, seg: usize) -> Result<()> {
        let s = &self.segments[seg];
        if s.exec.is_null() {
            return Err(Error::Backend(format!(
                "replay_segment({seg}): not captured"
            )));
        }
        let _guard = crate::device::util::DeviceGuard::set(s.device as i32);
        // SAFETY: exec/stream owned by the segment's device, valid post-capture.
        check_hip("hipGraphLaunch", unsafe {
            crate::hipGraphLaunch(s.exec, s.stream)
        })?;
        Ok(())
    }

    /// Allocate buffers + wrap. Single entry used by `get_or_create_decode_graph`.
    #[allow(clippy::too_many_arguments)]
    pub fn allocate(
        dev: &RocmDevice,
        stream: *mut c_void,
        num_layers: usize,
        hidden_size: usize,
        n_q: usize,
        n_k: usize,
        n_v: usize,
        intermediate_size: usize,
        max_ctx: usize,
        vocab_size: usize,
        num_heads: usize,
        batch: usize,
    ) -> Result<Self> {
        let buffers = DecodeGraphBuffers::allocate(
            dev,
            num_layers,
            hidden_size,
            n_q,
            // The unit test exercises the attention path, so the branch width is
            // the attention one.
            n_q,
            n_k,
            n_v,
            intermediate_size,
            max_ctx,
            vocab_size,
            num_heads,
            batch,
            0,
            0,
            0,
            0, // no MoE / ShortConv layers in this simplified constructor
        )?;
        Ok(Self::new(dev, buffers, stream))
    }

    pub fn begin_capture(&mut self) -> Result<()> {
        if !decode_graph_enabled() {
            return Err(Error::Backend("decode graph disabled by env".into()));
        }
        if self.is_captured {
            return Err(Error::Backend("begin_capture: already captured".into()));
        }
        // Split path: publish the fp8 pad, then open segment 0's bracket. The
        // interior boundaries are advanced by the model's forward_capture;
        // end_capture closes the last open segment.
        if self.split_capture() {
            // Publish the padded-fp8 act scratch for the duration of the
            // capture (WhiteRaven blocked leg needs it and cannot allocate
            // inside a capture).
            install_capture_fp8_pad(self.buffers.act_fp8_pad_buf.first());
            return self.begin_capture_segment(0);
        }
        let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        // SAFETY: stream owned by device pool, valid for capture duration.
        let res: crate::HipErrorT = unsafe { hipStreamBeginCapture(self.stream, 2) };
        if res != crate::hipSuccess {
            return Err(Error::Backend(format!(
                "hipStreamBeginCapture failed: {res}"
            )));
        }
        self.capturing = true;
        // Raise the device capture gate on all model devices: routes that are not
        // capture-safe (GsqRco3p5 u4-lane GEMV) must refuse HERE, or their kernels launch
        // on the default stream, never enter this graph, and every replay
        // reads an uninitialized output (all-NaN logits).
        let mut ords = self.buffers.layer_devices.clone();
        if !ords.contains(&self.ordinal) {
            ords.push(self.ordinal);
        }
        for &ord in &ords {
            crate::device::roc_device::RocmDevice::shared(ord)
                .set_capture_gate(true, Some(self.stream));
        }
        // Publish the padded-fp8 act scratch for the duration of the capture.
        // The WhiteRaven blocked leg needs it (its A operand is fp8 codes,
        // padded to whole 16-row tiles) and cannot allocate one here.
        install_capture_fp8_pad(self.buffers.act_fp8_pad_buf.first());
        Ok(())
    }

    /// Abort an open capture after a recording failure (e.g. a layer bailed
    /// to eager fallback mid-capture). Ends the capture, destroys the partial
    /// graph, leaves `is_captured=false`. Without this the stream stays in
    /// capture mode and later D2H/H2D calls fail (hipMemcpyDtoH 906).
    pub fn abort_capture(&mut self) -> Result<()> {
        if self.split_capture() {
            // Per open segment: close under ITS device's guard, destroy the
            // partial graph directly — NO instantiate/upload of a graph we are
            // about to destroy. Then tear down any already-finalized segment.
            for seg in 0..self.segments.len() {
                if self.segments[seg].exec.is_null() && self.capturing {
                    let s = &self.segments[seg];
                    let _guard = crate::device::util::DeviceGuard::set(s.device as i32);
                    let mut graph: *mut c_void = std::ptr::null_mut();
                    // SAFETY: stream is under capture; result ignored beyond
                    // cleanup: the partial graph is always discarded.
                    let _ = unsafe { hipStreamEndCapture(s.stream, &mut graph) };
                    crate::device::roc_device::RocmDevice::shared(s.device)
                        .set_capture_gate(false, None);
                    if !graph.is_null() {
                        unsafe {
                            let _ = hipGraphDestroy(graph);
                        }
                    }
                }
                let (graph, exec) = (self.segments[seg].graph, self.segments[seg].exec);
                if !graph.is_null() || !exec.is_null() {
                    let _guard =
                        crate::device::util::DeviceGuard::set(self.segments[seg].device as i32);
                    unsafe {
                        if !exec.is_null() {
                            let _ = hipGraphExecDestroy(exec);
                        }
                        if !graph.is_null() {
                            let _ = hipGraphDestroy(graph);
                        }
                    }
                    self.segments[seg].graph = std::ptr::null_mut();
                    self.segments[seg].exec = std::ptr::null_mut();
                }
            }
            self.is_captured = false;
            self.capturing = false;
            // Clear the thread-local fp8 pad, exactly like the legacy path.
            install_capture_fp8_pad(None);
            return Ok(());
        }
        let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let mut graph: *mut c_void = std::ptr::null_mut();
        // SAFETY: stream is under capture; graph out-ptr valid. Result ignored
        // beyond cleanup: the partial graph is always discarded.
        let _ = unsafe { hipStreamEndCapture(self.stream, &mut graph) };
        self.capturing = false;
        let mut ords = self.buffers.layer_devices.clone();
        if !ords.contains(&self.ordinal) {
            ords.push(self.ordinal);
        }
        for &ord in &ords {
            crate::device::roc_device::RocmDevice::shared(ord).set_capture_gate(false, None);
        }
        install_capture_fp8_pad(None);
        if !graph.is_null() {
            unsafe {
                let _ = hipGraphDestroy(graph);
            }
        }
        Ok(())
    }

    pub fn end_capture(&mut self) -> Result<()> {
        // Split path: close the last open segment (forward boundaries were
        // closed by forward_capture itself), validate every segment captured,
        // then clear the fp8 pad.
        if self.split_capture() {
            if !self.capturing {
                return Err(Error::Backend("end_capture without begin_capture".into()));
            }
            let last = self.segments.len() - 1;
            if self.segments[last].exec.is_null() {
                self.end_capture_segment(last)?;
            }
            if let Some(missing) = self.segments.iter().position(|s| s.exec.is_null()) {
                return Err(Error::Backend(format!(
                    "end_capture: segment {missing} was never captured"
                )));
            }
            self.is_captured = true;
            self.capturing = false;
            install_capture_fp8_pad(None);
            self.publish_task_graph();
            return Ok(());
        }
        let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let mut graph: *mut c_void = std::ptr::null_mut();
        // SAFETY: stream is under capture; graph out-ptr valid.
        let res: crate::HipErrorT = unsafe { hipStreamEndCapture(self.stream, &mut graph) };
        install_capture_fp8_pad(None);
        let mut ords = self.buffers.layer_devices.clone();
        if !ords.contains(&self.ordinal) {
            ords.push(self.ordinal);
        }
        if res != crate::hipSuccess {
            self.capturing = false;
            for &ord in &ords {
                crate::device::roc_device::RocmDevice::shared(ord).set_capture_gate(false, None);
            }
            return Err(Error::Backend(format!("hipStreamEndCapture failed: {res}")));
        }
        let mut exec: *mut c_void = std::ptr::null_mut();
        // SAFETY: graph just captured, exec out-ptr valid.
        let inst: crate::HipErrorT = unsafe {
            hipGraphInstantiate(
                &mut exec,
                graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            )
        };
        if inst != crate::hipSuccess {
            self.capturing = false;
            for &ord in &ords {
                crate::device::roc_device::RocmDevice::shared(ord).set_capture_gate(false, None);
            }
            unsafe {
                let _ = hipGraphDestroy(graph);
            }
            return Err(Error::Backend(format!(
                "hipGraphInstantiate failed: {inst}"
            )));
        }
        let upload = unsafe { crate::hipGraphUpload(exec, self.stream) };
        if upload != crate::hipSuccess {
            self.capturing = false;
            for &ord in &ords {
                crate::device::roc_device::RocmDevice::shared(ord).set_capture_gate(false, None);
            }
            unsafe {
                let _ = hipGraphExecDestroy(exec);
                let _ = hipGraphDestroy(graph);
            }
            return Err(Error::Backend(format!("hipGraphUpload failed: {upload}")));
        }
        self.graph = graph;
        self.exec = exec;
        self.is_captured = true;
        self.capturing = false;
        for &ord in &ords {
            crate::device::roc_device::RocmDevice::shared(ord).set_capture_gate(false, None);
        }
        self.publish_task_graph();
        Ok(())
    }

    /// Replay entire graph in one launch. Caller must `write_pos_async`
    /// + H2D input first, D2H logits after.
    ///
    /// Split path: delegates to `replay_chain` (task A4) — a bare segment
    /// loop would read the previous segment's stale boundary state, so until
    /// A4 lands a split graph refuses to replay rather than silently
    /// producing wrong output.
    pub fn replay(&self) -> Result<()> {
        if self.split_capture() {
            return self.replay_chain();
        }
        if !self.is_captured || self.exec.is_null() {
            return Err(Error::Backend("replay: graph not captured".into()));
        }
        let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        // SAFETY: exec/stream owned, valid post-capture.
        check_hip("hipGraphLaunch", unsafe {
            crate::hipGraphLaunch(self.exec, self.stream)
        })?;
        Ok(())
    }

    /// Replay every segment in order, handing the hidden state across each
    /// device boundary. The handoff is the only per-token cross-device
    /// traffic: segment k's graph writes `boundary_out[k]`, the ordered peer
    /// copy moves it into segment k+1's first `layer_input`, and segment k+1's
    /// launch waits on the event before starting.
    pub fn replay_chain(&self) -> Result<()> {
        if !self.split_capture() {
            return self.replay();
        }
        if self.segments.iter().any(|s| s.exec.is_null()) {
            return Err(Error::Backend(
                "replay_chain: a segment is not captured".into(),
            ));
        }
        let handoffs = self.use_boundary_handoffs();
        for seg in 0..self.segments.len() {
            if handoffs && seg > 0 {
                // M6: the consumer's DEVICE waits for the previous boundary's
                // copy to land — no host event. The wait is on the consumer's
                // OWN segment stream, so the consumer's graph (launched next,
                // on that same stream) starts only after the handoff landed.
                let consumer = crate::RocmDevice::shared(self.segments[seg].device);
                let h = self.boundary_handoffs[seg - 1]
                    .as_ref()
                    .expect("handoff presence checked");
                h.launch_wait(&consumer, self.segments[seg].stream)?;
            }
            if let Some(&(start_ev, _)) = self.segment_timing_evs.get(seg) {
                if !start_ev.is_null() {
                    let _guard =
                        crate::device::util::DeviceGuard::set(self.segments[seg].device as i32);
                    unsafe {
                        let _ = crate::hipEventRecord(start_ev, self.segments[seg].stream);
                    }
                }
            }
            let host_launch_start = if std::env::var_os("GRIM_SEGMENT_TIMING").is_some() {
                Some(std::time::Instant::now())
            } else {
                None
            };
            self.replay_segment(seg)?;
            if let Some(t0) = host_launch_start {
                let launch_us = t0.elapsed().as_micros();
                if seg == 0 {
                    eprintln!(
                        "[segment-launch] seg{seg}_host_launch={launch_us} us ({:.3} ms)",
                        launch_us as f64 / 1000.0
                    );
                }
            }
            if let Some(&(_, stop_ev)) = self.segment_timing_evs.get(seg) {
                if !stop_ev.is_null() {
                    let _guard =
                        crate::device::util::DeviceGuard::set(self.segments[seg].device as i32);
                    unsafe {
                        let _ = crate::hipEventRecord(stop_ev, self.segments[seg].stream);
                    }
                }
            }
            if seg + 1 < self.segments.len() {
                let s = &self.segments[seg];
                let n = &self.segments[seg + 1];
                if handoffs {
                    // M6: the producer's wave does the copy, on the producer's
                    // own stream — so it is ordered AFTER the producer's graph,
                    // which was just launched on that same stream.
                    let producer = crate::RocmDevice::shared(s.device);
                    let h = self.boundary_handoffs[seg]
                        .as_ref()
                        .expect("handoff presence checked");
                    h.launch_copy(&producer, s.stream)?;
                } else {
                    let ev = self
                        .boundary_evs
                        .get(seg)
                        .copied()
                        .unwrap_or(std::ptr::null_mut());
                    if ev.is_null() {
                        return Err(Error::Backend(format!(
                            "replay_chain: no boundary event for segment {seg}"
                        )));
                    }
                    let src = &self.buffers.boundary_out[seg];
                    let dst = &self.buffers.layer_input[n.layers.start];
                    let staging_ptr = self
                        .boundary_staging
                        .get(seg)
                        .and_then(|s| s.as_ref())
                        .map(|s| s.as_device_ptr());
                    crate::peer_access::copy_boundary_ordered_with_staging(
                        src,
                        dst,
                        s.stream,
                        n.stream,
                        ev,
                        staging_ptr,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Read per-segment elapsed milliseconds from recorded timing events.
    /// Returns None if timing is disabled or events were not recorded/synchronized.
    pub fn read_segment_timings(&self) -> Option<Vec<f32>> {
        if self.segment_timing_evs.is_empty() {
            return None;
        }
        let mut times = Vec::with_capacity(self.segment_timing_evs.len());
        for (i, &(start, stop)) in self.segment_timing_evs.iter().enumerate() {
            if start.is_null() || stop.is_null() {
                return None;
            }
            let seg_dev = self
                .segments
                .get(i)
                .map(|s| s.device)
                .unwrap_or(self.ordinal);
            let _guard = crate::device::util::DeviceGuard::set(seg_dev as i32);
            let mut elapsed_ms = 0.0f32;
            let rc = unsafe {
                crate::device::handles::hipEventElapsedTime(&mut elapsed_ms, start, stop)
            };
            if rc != crate::hipSuccess {
                return None;
            }
            times.push(elapsed_ms);
        }
        Some(times)
    }

    /// Spec §Scalar: update KV-position kernel args before replay.
    /// Calls real `hipGraphExecKernelNodeSetParams` when node known;
    /// else caller uses `write_pos_async` device-buffer path.
    pub fn update_kv_pos_params(&self, node_params: *const c_void) -> Result<()> {
        if self.kv_append_node.is_null() {
            return Err(Error::Unimplemented(
                "no kv_append_node; use write_pos_async".into(),
            ));
        }
        if node_params.is_null() {
            return Err(Error::Backend("update_kv_pos_params: null params".into()));
        }
        let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        // SAFETY: exec + node from capture; params point to valid node struct.
        let res: crate::HipErrorT =
            unsafe { hipGraphExecKernelNodeSetParams(self.exec, self.kv_append_node, node_params) };
        if res != crate::hipSuccess {
            return Err(Error::Backend(format!(
                "hipGraphExecKernelNodeSetParams failed: {res}"
            )));
        }
        Ok(())
    }

    /// Async 4-byte H2D input write (token id as f32 bits) into layer 0.
    /// Runs BEFORE replay, never inside capture. Ordered vs replay on same stream.
    pub fn write_input_async(&self, dev: &RocmDevice, token_id: u32) -> Result<()> {
        let dst = self
            .buffers
            .layer_input
            .first()
            .ok_or_else(|| Error::Backend("write_input: no layers".into()))?;
        let bits = f32::from_bits(token_id);
        dev.write_f32_into_async(dst, &[bits])?;
        Ok(())
    }

    /// Write a batch of token IDs into layer 0 input buffer (P3 batch decode).
    /// `token_ids` must have length == `buffers.batch`.
    pub fn write_input_batch_async(&self, dev: &RocmDevice, token_ids: &[u32]) -> Result<()> {
        let dst = self
            .buffers
            .layer_input
            .first()
            .ok_or_else(|| Error::Backend("write_input_batch: no layers".into()))?;
        let n = token_ids.len().min(self.buffers.batch);
        let bits: Vec<f32> = token_ids[..n].iter().map(|&t| f32::from_bits(t)).collect();
        dev.write_f32_into_async(dst, &bits)?;
        Ok(())
    }

    /// Single D2H sync point AFTER replay (spec §Phase 5).
    /// Reads `buffers.head_output` to host. Never call inside capture.
    pub fn read_logits_f32(&self) -> Result<Vec<f32>> {
        use grim_tensor::backend::BackendStorage;
        self.buffers.head_output.to_cpu_vec_f32()
    }

    /// Access live device storage for logits (P0 GPU sampler path).
    pub fn logits_device_storage(&self) -> &RocmStorage {
        &self.buffers.head_output
    }

    /// Access live device-resident logits Tensor (zero-copy, zero D2H).
    pub fn logits_tensor(&self) -> Result<grim_tensor::Tensor> {
        use grim_tensor::backend::BackendStorage;
        let shape = self.buffers.head_output.shape().clone();
        let dtype = self.buffers.head_output.dtype();
        let prov = self.buffers.head_output.provenance();
        let dev = grim_tensor::Device::Rocm(self.ordinal);
        let storage: Arc<dyn BackendStorage> = self.buffers.head_output.clone();
        Ok(grim_tensor::Tensor::new(storage, shape, dtype, prov, dev))
    }
}

impl Drop for DecodeGraph {
    fn drop(&mut self) {
        // HIP seam: graph-object destruction is device-context-sensitive, and
        // a dropped graph at teardown runs on whatever ordinal the thread
        // last touched. Pin the graph's own device for every destroy call.
        let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        unsafe {
            if !self.exec.is_null() {
                let _ = hipGraphExecDestroy(self.exec);
                self.exec = std::ptr::null_mut();
            }
            if !self.graph.is_null() {
                let _ = hipGraphDestroy(self.graph);
                self.graph = std::ptr::null_mut();
            }
            // Per-segment graphs + the boundary events (tasks A2/A4).
            for s in self.segments.iter_mut() {
                let _guard = crate::device::util::DeviceGuard::set(s.device as i32);
                if !s.exec.is_null() {
                    let _ = hipGraphExecDestroy(s.exec);
                    s.exec = std::ptr::null_mut();
                }
                if !s.graph.is_null() {
                    let _ = hipGraphDestroy(s.graph);
                    s.graph = std::ptr::null_mut();
                }
            }
            for (seg_idx, ev) in self.boundary_evs.drain(..).enumerate() {
                let dev_ord = self
                    .segments
                    .get(seg_idx)
                    .map(|s| s.device)
                    .unwrap_or(self.ordinal);
                let _guard = crate::device::util::DeviceGuard::set(dev_ord as i32);
                if !ev.is_null() {
                    let _ = crate::hipEventDestroy(ev);
                }
            }
            for (seg_idx, (start, stop)) in self.segment_timing_evs.drain(..).enumerate() {
                let dev_ord = self
                    .segments
                    .get(seg_idx)
                    .map(|s| s.device)
                    .unwrap_or(self.ordinal);
                let _guard = crate::device::util::DeviceGuard::set(dev_ord as i32);
                if !start.is_null() {
                    let _ = crate::hipEventDestroy(start);
                }
                if !stop.is_null() {
                    let _ = crate::hipEventDestroy(stop);
                }
            }
            // Sync segment streams before dropping owned staging buffers
            for s in &self.segments {
                if !s.stream.is_null() {
                    let _guard = crate::device::util::DeviceGuard::set(s.device as i32);
                    let _ = crate::device::handles::hipStreamSynchronize(s.stream);
                }
            }
            self.boundary_staging.clear();
        }
    }
}

/// Spec §Env: both flags disable. `=0/false/off` (case-insensitive `false/off`) = eager.
pub fn decode_graph_enabled() -> bool {
    for key in ["GRIM_DECODE_GRAPH", "GRIM_CAPTURE_GRAPH"] {
        if let Ok(v) = std::env::var(key) {
            if matches!(v.as_str(), "0" | "false" | "off" | "False" | "OFF") {
                return false;
            }
        }
    }
    true
}

/// Spec §Phase 6 helpers. GEMM stays in rocBLAS (Rule 0); these validate
/// stable-address topology for capture. Real KV/attention launches live in
/// `kernels::qkv_attention` (`launch_kv_append`, `launch_qkv_attention_dev`).
pub fn launch_qkv_gemv(k_arena: &RocmStorage, pos: u32, max_ctx: usize) -> Result<()> {
    launch_qkv_gemv_ring(k_arena, pos, max_ctx, KvRingPolicy::UNLIMITED)
}

/// Ring-aware [`launch_qkv_gemv`].
///
/// `pos` is the ABSOLUTE token index, which under the Phase-3 eviction ring may
/// legitimately exceed `max_ctx` (the arena's capacity): the token lands at
/// `ring.row(pos)`. Validating the raw `pos` against `max_ctx` would reject
/// every capture taken once the ring wraps, silently forcing the eager path —
/// exactly when eviction is supposed to be carrying the load. So bounds-check
/// the row that will actually be written. With `UNLIMITED` this is exactly the
/// old `pos < max_ctx` check.
pub fn launch_qkv_gemv_ring(
    k_arena: &RocmStorage,
    pos: u32,
    max_ctx: usize,
    ring: KvRingPolicy,
) -> Result<()> {
    let ptr = k_arena
        .device_ptr_u64()
        .ok_or_else(|| Error::Backend("launch_qkv_gemv: k_arena has no device ptr".into()))?;
    if ptr == 0 {
        return Err(Error::Backend("launch_qkv_gemv: null arena ptr".into()));
    }
    let row = ring.row(pos as usize);
    if row >= max_ctx {
        return Err(Error::Backend(format!(
            "launch_qkv_gemv: pos {pos} maps to arena row {row} >= max_ctx {max_ctx} \
             (ring cap {} sinks {})",
            ring.cap_rows, ring.sink_rows
        )));
    }
    Ok(())
}

/// Validate attention replay topology (stable pointers, in-range pos).
pub fn launch_attention(k_arena: &RocmStorage, pos: u32, max_ctx: usize) -> Result<()> {
    launch_qkv_gemv_ring(k_arena, pos, max_ctx, KvRingPolicy::UNLIMITED)
}

/// Ring-aware [`launch_attention`].
pub fn launch_attention_ring(
    k_arena: &RocmStorage,
    pos: u32,
    max_ctx: usize,
    ring: KvRingPolicy,
) -> Result<()> {
    launch_qkv_gemv_ring(k_arena, pos, max_ctx, ring)
}

/// Async H2D of token embedding id into fixed input buffer (4 bytes).
/// Thin wrapper so `lfm2_graph` + `run.rs` share one call site.
/// Ordered vs later launches on the active stream — no host sync.
/// Convention: `(dev, dst, token_id)` — matches `write_embeddings_to_buffer_batch`.
pub fn write_embedding_to_buffer(dev: &RocmDevice, dst: &RocmStorage, token_id: u32) -> Result<()> {
    let bits = f32::from_bits(token_id);
    dev.write_f32_into_async(dst, &[bits])
}

/// P3 batch version: write per-slot token IDs into `token_ids_dev`
/// (shape [batch]) — token id bits per slot, one u32 per slot. For a
/// 2-D `[batch, hidden]` destination, ids land at the start of each slot
/// row (legacy stub layout). `token_ids.len()` must be <= batch.
pub fn write_embeddings_to_buffer_batch(
    dev: &RocmDevice,
    dst: &RocmStorage,
    token_ids: &[u32],
) -> Result<()> {
    let dims = dst.shape.dims();
    let (batch, hidden) = match dims.len() {
        2 => (dims[0].max(1), dims[1].max(1)),
        // [batch] token-id device buffer: contiguous, one slot per id.
        1 => (dims[0].max(1), 1),
        _ => {
            return Err(Error::Backend(format!(
                "write_embeddings_to_buffer_batch: expected [batch, hidden] buffer, got {dims:?}"
            )));
        }
    };
    if token_ids.len() > batch {
        return Err(Error::Backend(format!(
            "write_embeddings_to_buffer_batch: {} token ids > batch {}",
            token_ids.len(),
            batch
        )));
    }
    let mut buf = vec![0.0f32; batch * hidden];
    for (slot, &t) in token_ids.iter().enumerate() {
        buf[slot * hidden] = f32::from_bits(t);
    }
    dev.write_f32_into_async(dst, &buf)
}

/// Shape guard for per-layer graph recording (no alloc, no sync).
pub fn check_layer_topology(buffers: &DecodeGraphBuffers, layer_idx: usize) -> Result<()> {
    if layer_idx >= buffers.num_layers {
        return Err(Error::Backend(format!(
            "forward_graph: layer {layer_idx} >= {}",
            buffers.num_layers
        )));
    }
    for (name, v) in [
        ("layer_input", &buffers.layer_input),
        ("layer_output", &buffers.layer_output),
        ("q_buf", &buffers.q_buf),
        ("k_buf", &buffers.k_buf),
        ("v_buf", &buffers.v_buf),
        ("attn_out_buf", &buffers.attn_out_buf),
        ("gate_buf", &buffers.gate_buf),
        ("up_buf", &buffers.up_buf),
        ("activated_buf", &buffers.activated_buf),
        ("norm_buf", &buffers.norm_buf),
        ("attn_max_buf", &buffers.attn_max_buf),
        ("attn_sum_buf", &buffers.attn_sum_buf),
        ("act_q81_buf", &buffers.act_q81_buf),
        ("fused_qkv_out", &buffers.fused_qkv_out),
        ("k_arena", &buffers.k_arena),
        ("v_arena", &buffers.v_arena),
    ] {
        let s = v
            .get(layer_idx)
            .ok_or_else(|| Error::Backend(format!("{name}[{layer_idx}] missing")))?;
        s.device_ptr_checked()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_segments_are_maximal_contiguous_runs() {
        assert_eq!(device_segments(&[0, 0, 1, 1]), vec![(0, 0..2), (1, 2..4)]);
        // Single device: one segment covering everything.
        assert_eq!(device_segments(&[0, 0, 0]), vec![(0, 0..3)]);
        // Non-contiguous placement must NOT be silently merged — it yields more
        // segments, which is exactly what the caller must know about.
        assert_eq!(
            device_segments(&[0, 1, 0]),
            vec![(0, 0..1), (1, 1..2), (0, 2..3)]
        );
        assert!(device_segments(&[]).is_empty());
    }

    #[test]
    fn device_segments_rejects_a_two_segment_split_at_layer_zero() {
        // A run that starts at 0 is well-formed; the invariant the caller relies on
        // is that segment k+1 starts where segment k ended (no gaps, no overlap).
        let segs = device_segments(&[0, 0, 1, 1, 1]);
        for w in segs.windows(2) {
            assert_eq!(
                w[0].1.end, w[1].1.start,
                "segments must tile the layer range"
            );
        }
        assert_eq!(segs.first().unwrap().1.start, 0);
        assert_eq!(segs.last().unwrap().1.end, 5);
    }

    /// A2: one captured graph per contiguous device run, each on ITS device's
    /// pooled stream. Needs two live ROCm devices; skips otherwise.
    #[test]
    fn decode_graph_builds_one_segment_per_device_run() {
        if !crate::gpu_test_enabled() {
            return;
        }
        let _lock = crate::device::util::gpu_test_lock();
        let Ok(dev) = crate::RocmDevice::try_new(0) else {
            eprintln!("[SKIP] device 0 unavailable");
            return;
        };
        if crate::RocmDevice::try_new(1).is_err() {
            eprintln!("[SKIP] needs 2 ROCm devices");
            return;
        }
        // 2 layers on device 0, 1 on device 1 -> 2 segments.
        let mut g = DecodeGraph::allocate(
            &dev,
            dev.active_stream(),
            3,
            64,
            32,
            8,
            8,
            128,
            16,
            64,
            4,
            1,
        )
        .expect("allocate");
        assert!(
            g.segments.is_empty(),
            "no set_layer_devices call yet -> legacy path"
        );
        g.set_layer_devices(&[0, 0, 1]);
        assert_eq!(g.segments.len(), 2, "two device runs must yield two graphs");
        assert_eq!(g.segments[0].layers, 0..2);
        assert_eq!(g.segments[1].layers, 2..3);
        assert_eq!(g.segments[0].device, 0);
        assert_eq!(g.segments[1].device, 1);
        // Each segment's stream must belong to ITS device.
        assert_eq!(
            g.segments[0].stream,
            crate::device::roc_device::RocmDevice::shared(0).active_stream()
        );
        assert_eq!(
            g.segments[1].stream,
            crate::device::roc_device::RocmDevice::shared(1).active_stream()
        );
    }

    /// A3: one handoff buffer per device transition, allocated on the
    /// PRODUCING device, so each segment's graph stays single-device.
    #[test]
    fn boundary_residual_targets_the_producing_device() {
        if !crate::gpu_test_enabled() {
            return;
        }
        let _lock = crate::device::util::gpu_test_lock();
        let Ok(dev) = crate::RocmDevice::try_new(0) else {
            eprintln!("[SKIP] device 0 unavailable");
            return;
        };
        let buffers = DecodeGraphBuffers::allocate_with_layer_devices(
            // (dev, layer_devices, num_layers, hidden, n_q, n_attn_out, n_k, n_v,
            //  inter, max_ctx, vocab, num_heads, batch, n_expert, top_k, sc_h_dim,
            //  sc_l_cache, latent_kv_dim) -- 18 params.
            &dev,
            Some(&[0, 0, 1]),
            3,
            64,
            32,
            32,
            8,
            8,
            128,
            16,
            64,
            4,
            1,
            0,
            0,
            0,
            0,
            0,
        )
        .expect("allocate");
        // Exactly one boundary (between layer 1 and layer 2).
        assert_eq!(buffers.boundary_out.len(), 1);
        // The producer is segment 0's device, NOT the consumer's (layer 2's device).
        assert_eq!(
            buffers.boundary_out[0].device_ordinal(),
            0,
            "the boundary buffer must live on the device whose graph writes it"
        );
        // The consumer buffer is still on the next layer's device.
        assert_eq!(buffers.layer_input[2].device_ordinal(), 1);
        // Same-device placement must produce no boundary buffers.
        let same = DecodeGraphBuffers::allocate_with_layer_devices(
            &dev,
            Some(&[0, 0, 0]),
            3,
            64,
            32,
            32,
            8,
            8,
            128,
            16,
            64,
            4,
            1,
            0,
            0,
            0,
            0,
            0,
        )
        .expect("allocate same-device");
        assert!(same.boundary_out.is_empty());
        assert_eq!(same.boundary_index(1), None);
        assert_eq!(same.boundary_index(2), None);
        // Split: layer 1 is the producer of the only boundary.
        assert_eq!(buffers.boundary_index(1), Some(0));
        assert_eq!(buffers.boundary_index(0), None);
    }

    /// A4: the boundary handoff is ordered (producer graph -> peer copy ->
    /// consumer graph) and byte-exact. Needs a P2P-capable pair.
    #[test]
    fn boundary_copy_is_ordered_and_byte_exact() {
        if !crate::gpu_test_enabled() {
            return;
        }
        let _lock = crate::device::util::gpu_test_lock();
        let Ok(d0) = crate::RocmDevice::try_new(0) else {
            eprintln!("[SKIP] device 0 unavailable");
            return;
        };
        if crate::RocmDevice::try_new(1).is_err() {
            eprintln!("[SKIP] needs 2 ROCm devices");
            return;
        }
        use grim_tensor::{CoreTensorOps, MemoryOps};
        let _ = crate::peer_access::enable_peer_access(0, 1);
        let n = 4096usize;
        let dt = dtype_f32();
        let src =
            RocmStorage::alloc_gpu(&Shape::new(vec![n]), dt.clone(), &d0.allocator, 0).unwrap();
        let d1 = crate::device::roc_device::RocmDevice::shared(1);
        let dst =
            RocmStorage::alloc_gpu(&Shape::new(vec![n]), dt.clone(), &d1.allocator, 1).unwrap();
        let want: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let host = d0
            .from_cpu(&want, &Shape::new(vec![n]), dt.clone())
            .expect("from_cpu");
        d0.copy_slice_into(&src, host.as_ref(), 0, n).unwrap();
        // `hipEventCreate` is the only event constructor grim declares.
        let mut ev: *mut std::ffi::c_void = std::ptr::null_mut();
        let _ev_guard = crate::device::util::DeviceGuard::set(0);
        assert_eq!(
            unsafe { crate::hipEventCreate(&mut ev) },
            crate::hipSuccess,
            "hipEventCreate"
        );
        crate::peer_access::copy_boundary_ordered(
            &src,
            &dst,
            d0.active_stream(),
            d1.active_stream(),
            ev,
        )
        .expect("copy");
        d0.synchronize();
        d1.synchronize();
        let got = dst.to_cpu_vec_f32().expect("readback");
        assert_eq!(got, want, "boundary copy must be byte-exact");
        unsafe {
            let _ = crate::hipEventDestroy(ev);
        }
    }

    #[test]
    fn decode_graph_disabled_by_either_flag() {
        temp_env::with_var("GRIM_DECODE_GRAPH", Some("0"), || {
            assert!(!decode_graph_enabled());
        });
        temp_env::with_var("GRIM_CAPTURE_GRAPH", Some("0"), || {
            assert!(!decode_graph_enabled());
        });
        temp_env::with_vars(
            [
                ("GRIM_DECODE_GRAPH", None::<&str>),
                ("GRIM_CAPTURE_GRAPH", None::<&str>),
            ],
            || assert!(decode_graph_enabled()),
        );
    }

    #[test]
    fn decode_graph_buffers_allocate_rejects_zero_batch() {
        // Zero-dim guard fires before any HIP alloc; no GPU needed.
        assert!(
            DecodeGraphBuffers::allocate(
                &crate::device::roc_device::RocmDevice::shared(0),
                // num_layers, hidden, n_q, n_attn_out, n_k, n_v, inter,
                // max_ctx, vocab, then the zero the guard is meant to catch.
                1,
                64,
                64,
                64,
                64,
                256,
                8,
                100,
                8,
                0,
                0,
                0,
                0,
                0,
                0
            )
            .is_err()
        );
    }

    #[test]
    fn launch_qkv_gemv_rejects_oob_pos() {
        // No GPU needed: null-ptr path errors before any HIP call.
        let alloc = std::sync::Arc::new(crate::memory::allocator::RocmCachingAllocator::new(
            0,
            1 << 20,
        ));
        let st = RocmStorage::alloc_gpu_with_bytes(
            &Shape::new(vec![8, 8]),
            dtype_f32(),
            8 * 8 * 4,
            &alloc,
            0,
        );
        // On CPU-only CI alloc fails -> fallback eager is correct behavior.
        if let Ok(s) = st {
            assert!(launch_qkv_gemv(&s, 99999, 8).is_err());
        }
    }
}
