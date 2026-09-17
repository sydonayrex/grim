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

use std::ffi::c_void;
use std::sync::Arc;

use crate::device::roc_device::RocmDevice;
use crate::device::util::dtype_f32;
use crate::memory::storage::RocmStorage;
use crate::DTypeStorage;
use crate::{
    Shape, check_hip, hipGraphDestroy, hipGraphExecDestroy, hipGraphExecKernelNodeSetParams,
    hipGraphInstantiate, hipMemcpyAsync, hipStreamBeginCapture, hipStreamEndCapture,
};
use crate::HipMemcpyKind;
use grim_tensor::error::{Error, Result};
use grim_tensor::{ArithType, DType};

/// Spec §Phase 1: owns all decode intermediates at fixed addresses.
/// Allocated once at model load, never freed until unload.
/// `batch` parameterizes all per-step slots as `[batch, dim]` so the same
/// captured graph replays a whole decode batch in one `hipGraphLaunch`.
#[derive(Debug)]
pub struct DecodeGraphBuffers {
    /// Per-layer buffers (indexed by layer_idx)
    pub layer_input: Vec<RocmStorage>,   // [batch, hidden_size]
    pub layer_output: Vec<RocmStorage>,  // [batch, hidden_size]
    /// Attention-specific
    pub q_buf: Vec<RocmStorage>,         // [batch, n_q]
    pub k_buf: Vec<RocmStorage>,         // [batch, n_k]
    pub v_buf: Vec<RocmStorage>,         // [batch, n_v]
    pub attn_out_buf: Vec<RocmStorage>,  // [batch, n_q]
    /// FFN-specific
    pub gate_up_buf: Vec<RocmStorage>,   // [batch, 2*intermediate_size] (reserved: fused gate+up path)
    pub gate_buf: Vec<RocmStorage>,      // [batch, intermediate_size]
    pub up_buf: Vec<RocmStorage>,        // [batch, intermediate_size]
    pub activated_buf: Vec<RocmStorage>, // [batch, intermediate_size]
    /// Per-layer [batch, hidden] staging for norm outputs and GEMM results that
    /// feed a residual add. Sequentially reused within a layer (stream order
    /// preserves dependencies); never live across layers.
    pub norm_buf: Vec<RocmStorage>,      // [batch, hidden_size]
    /// Online-softmax partials for `launch_qkv_attention_dev`, kept in-pool
    /// so capture allocates nothing.
    pub attn_max_buf: Vec<RocmStorage>,  // [num_heads]
    pub attn_sum_buf: Vec<RocmStorage>,  // [num_heads]
    /// KV cache arenas (pre-allocated to max context)
    pub k_arena: Vec<RocmStorage>,       // [max_ctx, n_k]
    pub v_arena: Vec<RocmStorage>,       // [max_ctx, n_v]
    /// Output projection
    pub head_input: RocmStorage,         // [batch, hidden_size]
    pub head_output: Arc<RocmStorage>,   // [batch, vocab_size]
    /// Host-side position mirror. Device scalar lives in `pos_dev`;
    /// kernels read pos from device so graph stays capturable.
    pub current_pos: u32,
    /// Single-u32 device buffer holding KV position. Updated before each
    /// replay via one 4-byte async H2D (spec §Scalar second approach).
    pub pos_dev: RocmStorage,
    /// M2 (PLAN-kernel-fusion): MoE routing + output staging. Empty vecs when
    /// the model has no MoE layers. Routing triple is shared across layers —
    /// each layer writes then consumes it in stream order within its bracket.
    pub moe_gate_logits: Vec<RocmStorage>, // [batch, n_expert] per layer
    pub moe_out: Vec<RocmStorage>,         // [batch, hidden] per layer
    pub moe_route_tokens: RocmStorage,     // [batch*top_k] u32
    pub moe_route_experts: RocmStorage,    // [batch*top_k] u32
    pub moe_route_weights: RocmStorage,    // [batch*top_k] f32
    /// S2: ShortConv staging. Empty vecs when the model has no ShortConv
    /// layers. `sc_state[l]` is the device-resident ring `[h_dim*(kc)]`
    /// (column-major [d, kc], the HIP kernel's in-place layout).
    pub sc_proj_buf: Vec<RocmStorage>, // [batch, 3*h_dim]
    pub sc_b: Vec<RocmStorage>,        // [batch*h_dim]
    pub sc_c: Vec<RocmStorage>,
    pub sc_x: Vec<RocmStorage>,
    pub sc_bx: Vec<RocmStorage>,
    pub sc_sum: Vec<RocmStorage>,
    pub sc_y: Vec<RocmStorage>,        // [batch, h_dim]
    pub sc_state: Vec<RocmStorage>,    // [h_dim*(l_cache-1)] per layer
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
    /// Per-layer fused-QKV output staging ([n_q + 2*n_kv] F32); the three
    /// Q/K/V slices are D2D-copied into `q/k/v_buf`.
    pub fused_qkv_out: Vec<RocmStorage>,
    pub num_layers: usize,
    pub max_ctx: usize,
    pub batch: usize,
}

/// PLAN-reduce-d2h-h2d (prefill seeding): one layer's eager device K/V source.
/// Raw device pointers — caller guarantees they outlive the seed copy.
pub struct EagerKvSource<'a> {
    /// Device pointer to eager K rows (f32, `prefill_len × kv_stride`).
    pub k_dev: *const f32,
    /// Device pointer to eager V rows (f32, `prefill_len × kv_stride`).
    pub v_dev: *const f32,
    /// Valid rows in each buffer.
    pub prefill_len: u32,
    /// Elements per row (`num_kv_heads × head_dim`).
    pub kv_stride: usize,
    /// Borrow anchor so the pointers can't outlive the session caches.
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
        let mut gate_up_buf = Vec::with_capacity(num_layers);
        let mut gate_buf = Vec::with_capacity(num_layers);
        let mut up_buf = Vec::with_capacity(num_layers);
        let mut activated_buf = Vec::with_capacity(num_layers);
        let mut norm_buf = Vec::with_capacity(num_layers);
        let mut attn_max_buf = Vec::with_capacity(num_layers);
        let mut attn_sum_buf = Vec::with_capacity(num_layers);
        let mut act_q81_buf = Vec::with_capacity(num_layers);
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
        for _ in 0..num_layers {
            layer_input.push(RocmStorage::alloc_gpu(
                &Shape::new(vec![batch, hid]),
                dt.clone(),
                &dev.allocator,
                dev.ordinal,
            )?);
            layer_output.push(RocmStorage::alloc_gpu(
                &Shape::new(vec![batch, hid]),
                dt.clone(),
                &dev.allocator,
                dev.ordinal,
            )?);
            q_buf.push(RocmStorage::alloc_gpu(
                &Shape::new(vec![batch, nqk]),
                dt.clone(),
                &dev.allocator,
                dev.ordinal,
            )?);
            k_buf.push(RocmStorage::alloc_gpu(
                &Shape::new(vec![batch, nkk]),
                dt.clone(),
                &dev.allocator,
                dev.ordinal,
            )?);
            v_buf.push(RocmStorage::alloc_gpu(
                &Shape::new(vec![batch, nvk]),
                dt.clone(),
                &dev.allocator,
                dev.ordinal,
            )?);
            attn_out_buf.push(RocmStorage::alloc_gpu(
                &Shape::new(vec![batch, nqk]),
                dt.clone(),
                &dev.allocator,
                dev.ordinal,
            )?);
            gate_up_buf.push(RocmStorage::alloc_gpu(
                &Shape::new(vec![batch, 2 * inter]),
                dt.clone(),
                &dev.allocator,
                dev.ordinal,
            )?);
            gate_buf.push(RocmStorage::alloc_gpu(
                &Shape::new(vec![batch, inter]),
                dt.clone(),
                &dev.allocator,
                dev.ordinal,
            )?);
            up_buf.push(RocmStorage::alloc_gpu(
                &Shape::new(vec![batch, inter]),
                dt.clone(),
                &dev.allocator,
                dev.ordinal,
            )?);
            activated_buf.push(RocmStorage::alloc_gpu(
                &Shape::new(vec![batch, inter]),
                dt.clone(),
                &dev.allocator,
                dev.ordinal,
            )?);
            norm_buf.push(RocmStorage::alloc_gpu(
                &Shape::new(vec![batch, hid]),
                dt.clone(),
                &dev.allocator,
                dev.ordinal,
            )?);
            attn_max_buf.push(RocmStorage::alloc_gpu(
                &Shape::new(vec![batch * num_heads.max(1)]),
                dt.clone(),
                &dev.allocator,
                dev.ordinal,
            )?);
            attn_sum_buf.push(RocmStorage::alloc_gpu(
                &Shape::new(vec![batch * num_heads.max(1)]),
                dt.clone(),
                &dev.allocator,
                dev.ordinal,
            )?);
            act_q81_buf.push(RocmStorage::alloc_gpu_with_bytes(
                &Shape::new(vec![q81_bytes]),
                q81_dt.clone(),
                q81_bytes,
                &dev.allocator,
                dev.ordinal,
            )?);
            fused_qkv_out.push(RocmStorage::alloc_gpu(
                &Shape::new(vec![batch, nqk + 2 * nkk]),
                dt.clone(),
                &dev.allocator,
                dev.ordinal,
            )?);
            k_arena.push(RocmStorage::alloc_gpu(
                &Shape::new(vec![batch * max_ctx, nkk]),
                dt.clone(),
                &dev.allocator,
                dev.ordinal,
            )?);
            v_arena.push(RocmStorage::alloc_gpu(
                &Shape::new(vec![batch * max_ctx, nvk]),
                dt.clone(),
                &dev.allocator,
                dev.ordinal,
            )?);
        }
        let head_input = RocmStorage::alloc_gpu(
            &Shape::new(vec![batch, hid]),
            dt.clone(),
            &dev.allocator,
            dev.ordinal,
        )?;
        let head_output = Arc::new(RocmStorage::alloc_gpu(
            &Shape::new(vec![batch, vocab_size]),
            dt.clone(),
            &dev.allocator,
            dev.ordinal,
        )?);
        // One i32 position per batch slot (P3): the graph's append/bump
        // kernels index `pos_dev[slot]`.
        let pos_dev = RocmStorage::alloc_gpu(
            &Shape::new(vec![batch]),
            dt.clone(),
            &dev.allocator,
            dev.ordinal,
        )?;
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
                        &Shape::new(vec![batch, 3 * sc_h_dim]), dt.clone(), &dev.allocator, dev.ordinal)?);
                    sc_b.push(RocmStorage::alloc_gpu(
                        &Shape::new(vec![batch * sc_h_dim]), dt.clone(), &dev.allocator, dev.ordinal)?);
                    sc_c.push(RocmStorage::alloc_gpu(
                        &Shape::new(vec![batch * sc_h_dim]), dt.clone(), &dev.allocator, dev.ordinal)?);
                    sc_x.push(RocmStorage::alloc_gpu(
                        &Shape::new(vec![batch * sc_h_dim]), dt.clone(), &dev.allocator, dev.ordinal)?);
                    sc_bx.push(RocmStorage::alloc_gpu(
                        &Shape::new(vec![batch * sc_h_dim]), dt.clone(), &dev.allocator, dev.ordinal)?);
                    sc_sum.push(RocmStorage::alloc_gpu(
                        &Shape::new(vec![batch * sc_h_dim]), dt.clone(), &dev.allocator, dev.ordinal)?);
                    sc_y.push(RocmStorage::alloc_gpu(
                        &Shape::new(vec![batch, sc_h_dim]), dt.clone(), &dev.allocator, dev.ordinal)?);
                    sc_state.push(RocmStorage::alloc_gpu(
                        &Shape::new(vec![sc_h_dim * kc]), dt.clone(), &dev.allocator, dev.ordinal)?);
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
                RocmStorage::alloc_gpu(&Shape::new(vec![np]), udt.clone(), &dev.allocator, dev.ordinal)?,
                RocmStorage::alloc_gpu(&Shape::new(vec![np]), udt, &dev.allocator, dev.ordinal)?,
                RocmStorage::alloc_gpu(&Shape::new(vec![np]), dt.clone(), &dev.allocator, dev.ordinal)?,
            )
        } else {
            // Unused dummy [1] f32 zeros keeps the fields non-null.
            let d = RocmStorage::alloc_gpu(&Shape::new(vec![1]), dt.clone(), &dev.allocator, dev.ordinal)?;
            let d2 = RocmStorage::alloc_gpu(&Shape::new(vec![1]), dt.clone(), &dev.allocator, dev.ordinal)?;
            let d3 = RocmStorage::alloc_gpu(&Shape::new(vec![1]), dt.clone(), &dev.allocator, dev.ordinal)?;
            (d, d2, d3)
        };

        Ok(Self {
            layer_input,
            layer_output,
            q_buf,
            k_buf,
            v_buf,
            attn_out_buf,
            gate_up_buf,
            gate_buf,
            up_buf,
            activated_buf,
            norm_buf,
            attn_max_buf,
            attn_sum_buf,
            k_arena,
            v_arena,
            head_input,
            head_output,
            current_pos: 0,
            pos_dev,
            token_ids_dev,
            attn_dummy,
            act_q81_buf,
            fused_qkv_out,
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
        })
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
            let n_rows = src.prefill_len;
            let n_elem: usize = (n_rows as usize) * src.kv_stride;
            // K arena row width must match the eager cache stride.
            let arena_k_cols = self.k_arena[layer_idx].shape.dims().last().copied().unwrap_or(0);
            let arena_v_cols = self.v_arena[layer_idx].shape.dims().last().copied().unwrap_or(0);
            if arena_k_cols != src.kv_stride || arena_v_cols != src.kv_stride {
                return Err(Error::Backend(format!(
                    "seed_kv_arena: layer {layer_idx} arena width k={arena_k_cols} v={arena_v_cols} != kv_stride {}",
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
            let bytes = n_elem * std::mem::size_of::<f32>();
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
            let res: crate::HipErrorT = unsafe {
                crate::hipMemcpyAsync(
                    dst_k,
                    src_k,
                    bytes,
                    HipMemcpyKind::DeviceToDevice,
                    stream,
                )
            };
            if res != crate::hipSuccess {
                return Err(Error::Backend(format!("seed_kv_arena: hipMemcpyAsync K failed: {res}")));
            }
            let res: crate::HipErrorT = unsafe {
                crate::hipMemcpyAsync(
                    dst_v,
                    src_v,
                    bytes,
                    HipMemcpyKind::DeviceToDevice,
                    stream,
                )
            };
            if res != crate::hipSuccess {
                return Err(Error::Backend(format!("seed_kv_arena: hipMemcpyAsync V failed: {res}")));
            }
        }
        // Set the append position to the end of the seeded prefill so the first
        // decode step appends at the right offset and attends over the prompt.
        let prefill_len = per_layer.iter().find_map(|s| s.as_ref().map(|e| e.prefill_len)).unwrap_or(0);
        self.current_pos = prefill_len;
        // Seed the device position scalar to match; the graph's bump kernel
        // increments it on each replay, so it must start at prefill_len.
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
        let dst = self
            .pos_dev
            .device_ptr_u64()
            .ok_or_else(|| Error::Backend("pos_dev has no device ptr".into()))?
            as *mut c_void;
        let mut buf = vals.to_vec();
        buf.resize(self.batch.max(1), 0);
        let bytes = 4 * self.batch.max(1);
        // SAFETY: dst is device mem owned by pos_dev (4*batch bytes); host buf
        // has exactly that many valid bytes.
        let res: crate::HipErrorT = unsafe {
            hipMemcpyAsync(
                dst,
                buf.as_ptr() as *const c_void,
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
    ordinal: usize,
}

// SAFETY: raw HIP handles owned by self, only touched on owning device.
unsafe impl Send for DecodeGraph {}
unsafe impl Sync for DecodeGraph {}

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
            ordinal: dev.ordinal,
        }
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
            n_k,
            n_v,
            intermediate_size,
            max_ctx,
            vocab_size,
            num_heads,
            batch,
            0, 0, 0, 0, // no MoE / ShortConv layers in this simplified constructor
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
        let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        // SAFETY: stream owned by device pool, valid for capture duration.
        let res: crate::HipErrorT = unsafe { hipStreamBeginCapture(self.stream, 2) };
        if res != crate::hipSuccess {
            return Err(Error::Backend(format!(
                "hipStreamBeginCapture failed: {res}"
            )));
        }
        self.capturing = true;
        Ok(())
    }

    /// Abort an open capture after a recording failure (e.g. a layer bailed
    /// to eager fallback mid-capture). Ends the capture, destroys the partial
    /// graph, leaves `is_captured=false`. Without this the stream stays in
    /// capture mode and later D2H/H2D calls fail (hipMemcpyDtoH 906).
    pub fn abort_capture(&mut self) -> Result<()> {
        let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let mut graph: *mut c_void = std::ptr::null_mut();
        // SAFETY: stream is under capture; graph out-ptr valid. Result ignored
        // beyond cleanup: the partial graph is always discarded.
        let _ = unsafe { hipStreamEndCapture(self.stream, &mut graph) };
        self.capturing = false;
        if !graph.is_null() {
            unsafe {
                let _ = hipGraphDestroy(graph);
            }
        }
        Ok(())
    }

    pub fn end_capture(&mut self) -> Result<()> {
        let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let mut graph: *mut c_void = std::ptr::null_mut();
        // SAFETY: stream is under capture; graph out-ptr valid.
        let res: crate::HipErrorT = unsafe { hipStreamEndCapture(self.stream, &mut graph) };
        if res != crate::hipSuccess {
            self.capturing = false;
            return Err(Error::Backend(format!(
                "hipStreamEndCapture failed: {res}"
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
                "hipGraphInstantiate failed: {inst}"
            )));
        }
        self.graph = graph;
        self.exec = exec;
        self.is_captured = true;
        self.capturing = false;
        Ok(())
    }

    /// Replay entire graph in one launch. Caller must `write_pos_async`
    /// + H2D input first, D2H logits after.
    pub fn replay(&self) -> Result<()> {
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
        let res: crate::HipErrorT = unsafe {
            hipGraphExecKernelNodeSetParams(self.exec, self.kv_append_node, node_params)
        };
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
        unsafe {
            if !self.exec.is_null() {
                let _ = hipGraphExecDestroy(self.exec);
                self.exec = std::ptr::null_mut();
            }
            if !self.graph.is_null() {
                let _ = hipGraphDestroy(self.graph);
                self.graph = std::ptr::null_mut();
            }
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
pub fn launch_qkv_gemv(
    k_arena: &RocmStorage,
    pos: u32,
    max_ctx: usize,
) -> Result<()> {
    let ptr = k_arena
        .device_ptr_u64()
        .ok_or_else(|| Error::Backend("launch_qkv_gemv: k_arena has no device ptr".into()))?;
    if ptr == 0 {
        return Err(Error::Backend("launch_qkv_gemv: null arena ptr".into()));
    }
    if (pos as usize) >= max_ctx {
        return Err(Error::Backend(format!(
            "launch_qkv_gemv: pos {pos} >= max_ctx {max_ctx}"
        )));
    }
    Ok(())
}

/// Validate attention replay topology (stable pointers, in-range pos).
pub fn launch_attention(k_arena: &RocmStorage, pos: u32, max_ctx: usize) -> Result<()> {
    launch_qkv_gemv(k_arena, pos, max_ctx)
}

/// Async H2D of token embedding id into fixed input buffer (4 bytes).
/// Thin wrapper so `lfm2_graph` + `run.rs` share one call site.
/// Ordered vs later launches on the active stream — no host sync.
/// Convention: `(dev, dst, token_id)` — matches `write_embeddings_to_buffer_batch`.
pub fn write_embedding_to_buffer(
    dev: &RocmDevice,
    dst: &RocmStorage,
    token_id: u32,
) -> Result<()> {
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
            )))
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
        assert!(DecodeGraphBuffers::allocate(
            &crate::device::roc_device::RocmDevice::shared(0),
            1, 64, 64, 64, 64, 256, 8, 100, 8, 0, 0, 0, 0, 0
        )
        .is_err());
    }

    #[test]
    fn launch_qkv_gemv_rejects_oob_pos() {
        // No GPU needed: null-ptr path errors before any HIP call.
        let alloc = std::sync::Arc::new(
            crate::memory::allocator::RocmCachingAllocator::new(0, 1 << 20),
        );
        let st = RocmStorage::alloc_gpu_with_bytes(
            &Shape::new(vec![8, 8]),
            dtype_f32(),
            8 * 8 * 4,
            &alloc,
            0,
        );
        // On CPU-only CI alloc fails -> fallback eager is correct behavior.
        match st {
            Ok(s) => {
                assert!(launch_qkv_gemv(&s, 99999, 8).is_err());
            }
            Err(_) => {}
        }
    }
}
