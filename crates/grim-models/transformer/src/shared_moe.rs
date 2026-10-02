//! Shared Mixture-of-Experts dispatch (Phase 3a + 3c).
//!
//! Consolidates the per-expert loop pattern used by deepseek2/32/4, kimi_k3,
//! mellum into a single `fused_moe_dispatch` entry point.
//!
//! On ROCm the dispatch can route through Charon's grouped fused-kernel
//! (`grim_moe_fused_grouped` via `RocmDevice::moe_fused_grouped_dispatch_resident`),
//! which evaluates every routed (token, expert) pair in a single token-sorted
//! launch with the SiLU combine fused in-register. To keep the crate's
//! zero-host-round-trip-per-decode budget, the expert weights are stacked into
//! contiguous `[num_experts, ...]` f32 device buffers **once** and cached in a
//! [`CharonCache`] held by the model; every subsequent decode step reuses the
//! resident buffers with no H2D/D2H traffic. This mirrors the proven
//! `grim_nn::moe::RocmResidentWeights` design.
//!
//! Falls back to the stock per-expert loop on CPU, non-ROCm, or when the
//! backend lacks the needed primitives (or `GRIM_MOE_CHARON=0`).

use std::sync::{Arc, Mutex};

use grim_core::error::Result;
use grim_nn::Linear;
use grim_nn::modules::silu_mul_on_device;
use grim_tensor::backend::BackendDevice;
use grim_tensor::{CoreTensorOps, DType, Device, MemoryOps, Shape, Tensor};

/// One expert's SwiGLU FFN as three Linear layers (gate/up projection + down
/// projection), matching the `w1`/`w3`/`w2` naming used across the MoE models.
#[derive(Clone)]
pub struct MoeExpert {
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}

/// Routing result for a single token: (expert_index, combine_weight) pairs.
pub type TokenRouting = Vec<(usize, f32)>;

/// Resident expert-weight stack for the Charon grouped dispatch (Phase 3a).
///
/// Built once from the per-expert gate/up/down `Linear` weights and reused
/// across forward calls, so the decode hot path performs no host round-trips.
/// The stack is rebuilt only when the structural key `(num_experts, hidden,
/// inter)` changes. It also holds the device-resident routing scratch buffers
/// (tokens/experts/weights) written by the device-side routing kernel, so the
/// top-k selection and the routing table never round-trip through the host.
pub struct CharonCache {
    resident: Mutex<Option<ResidentWeights>>,
    /// Device-resident routing scratch: (tokens, experts, weights), resized to
    /// `seq_len * top_k` on demand, keyed by `(seq_len, top_k)`.
    routing: Mutex<Option<(usize, usize, RoutingBuffers)>>,
    /// Resident W8A8-int8 packed expert stacks (WI-gpu-native-moe Phase 2).
    /// Built once from packed per-expert blobs, keyed like `resident`.
    w8a8: Mutex<Option<ResidentWeights>>,
    /// Resident W8A8-fp8 packed expert stacks (same discipline as `w8a8`).
    w8a8fp8: Mutex<Option<ResidentWeights>>,
    /// Resident AWQ packed expert stacks. Fingerprint tag carries
    /// bits/group (see [`stack_tag_awq`]) so a config change rebuilds.
    awq: Mutex<Option<ResidentWeights>>,
    /// Resident MXFP4 codes + shared-exponent stacks (separate buffers —
    /// the kernel takes 6 weight pointers, not 3 packed blobs).
    mxfp4: Mutex<Option<Mxfp4Resident>>,
    /// Resident WhiteCrow u4-group128 expert blobs (gate/up/down, expert-
    /// strided, one blob per projection). Built on demand behind
    /// `GRIM_MOE_NATIVE_WHITECROW`; ~0.5 B/param vs 4 B/param for the f32
    /// stacks, which is what lets the decode graph's MoE leg fit in VRAM.
    whitecrow: Mutex<Option<WhiteCrowResident>>,
    /// Native K-quant dispatch handles (per-expert pointer arrays over the
    /// already-resident all-IQ3_S packed banks). Zero extra weight VRAM, so
    /// this arm needs no budget gate.
    kq_native: Mutex<Option<KqNativeResident>>,
    /// Set when a build was refused (budget/scheme); refused builds stay
    /// refused for this cache's lifetime so per-call retries stay cheap.
    whitecrow_refused: Mutex<bool>,
    /// Which dispatch arm the last `fused_moe_dispatch_from_logits` call took.
    /// Tests assert this to prove the NATIVE quantized arm ran (a numeric
    /// match alone cannot distinguish it from the dequant fallback — both
    /// compute the same math by design).
    last_dispatch: Mutex<DispatchKind>,
}

/// Which numeric path served a `fused_moe_dispatch_from_logits` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchKind {
    /// f32 expert stacks (native f32, or any quantized format dequantized
    /// to f32 at stack-build time).
    F32Dequant,
    /// Native W8A8-int8 dispatch (`grim_moe_fused_dispatch_w8a8_int8`)
    /// straight from packed int8 blobs.
    W8a8Native,
    /// V_DOT4 variant of the int8 arm (`grim_moe_fused_dispatch_w8a8_int8_
    /// dot4`): same numeric path (int32 Q8_1 dot products), different
    /// contraction. Recorded distinctly so tests/benches prove which ran.
    W8a8NativeDot4,
    /// Native K-quant dispatch (`grim_moe_fused_dispatch_kq_native`): the
    /// resident IQ3_S/Q4_K per-expert banks decoded in-register through
    /// device pointer arrays. Zero extra weight VRAM.
    KqNative,
    /// Native W8A8-fp8 dispatch (`grim_moe_fused_dispatch_w8a8_fp8`)
    /// straight from packed fp8 blobs.
    W8a8Fp8Native,
    /// Native AWQ dispatch (`grim_moe_fused_dispatch_awq`).
    AwqNative,
    /// Native MXFP4 dispatch (`grim_moe_fused_dispatch_mxfp4`). The f32
    /// dequant arm cannot serve MXFP4 (no device dequant exists), so MXFP4
    /// is native-or-error, never silent fallback.
    Mxfp4Native,
    /// WhiteCrow u4-group128 grouped dispatch
    /// (`grim_moe_fused_dispatch_whitecrow`): K-quant expert banks requantized
    /// once to OSTQuant u4 group-128 blobs, decoded in-register with f32
    /// activations. ~0.5 B/param resident vs the f32 stacks' 4 B/param.
    WhiteCrowNative,
}

/// Resident WhiteCrow u4-group128 expert blobs. One blob per projection;
/// expert `e` occupies `e * stride` bytes. Blob layout per expert mirrors
/// `ostquant_segment_offsets`: `[u64 qw_len][qweight u32 words N*K/8]
/// [u64 sc_len][scales bf16 N*K/128][u64 zr_len][zeros u8 N*K/128]`.
/// Native K-quant grouped dispatch handles: device pointer arrays (one u64
/// base pointer per expert, per projection) over the per-expert packed banks
/// that are ALREADY resident as the model's own weights. Allocates only 3 x
/// num_experts x 8 bytes. The arrays borrow the banks' device addresses; the
/// banks live in the owning layer and outlive this cache, so the addresses
/// are stable.
pub(crate) struct KqNativeResident {
    pub gate_ptrs: Arc<dyn grim_tensor::backend::BackendStorage>,
    pub up_ptrs: Arc<dyn grim_tensor::backend::BackendStorage>,
    pub down_ptrs: Arc<dyn grim_tensor::backend::BackendStorage>,
    pub num_experts: usize,
    /// PER-EXPERT packed bytes, per projection — the launcher derives the
    /// per-row stride the kernel addresses with (host-measured, never
    /// assumed).
    pub gate_bytes: u64,
    pub down_bytes: u64,
    /// Down bank scheme per layer: 1 = Q4_K (144 B per 256), 0 = IQ3_S.
    /// The checkpoint's down banks are NOT uniform across layers (blk.2's is
    /// Q4_K, others IQ3_S), so the kernel's down leg switches on this.
    pub down_q4k: bool,
}

pub(crate) struct WhiteCrowResident {
    pub gate: Arc<dyn grim_tensor::backend::BackendStorage>,
    pub up: Arc<dyn grim_tensor::backend::BackendStorage>,
    pub down: Arc<dyn grim_tensor::backend::BackendStorage>,
    pub gate_stride: u64,
    pub down_stride: u64,
}

struct ResidentWeights {
    gate: Arc<dyn grim_tensor::BackendStorage>,
    up: Arc<dyn grim_tensor::BackendStorage>,
    down: Arc<dyn grim_tensor::BackendStorage>,
    /// `(num_experts, hidden, inter, format_tag)`. The tag discriminates
    /// stacked formats sharing one slot family (`0` = f32, `1` = w8a8-int8,
    /// `2` = w8a8-fp8, `3` = awq with bits/group folded in — see
    /// [`stack_tag_awq`]); a tag mismatch rebuilds instead of aliasing.
    fingerprint: (usize, usize, usize, u64),
}

/// Resident MXFP4 stacks: E2M1 code bytes and E8M0 shared-exponent bytes,
/// each concatenated per expert (gate/up/down), no length prefixes.
struct Mxfp4Resident {
    codes_gate: Arc<dyn grim_tensor::BackendStorage>,
    codes_up: Arc<dyn grim_tensor::BackendStorage>,
    codes_down: Arc<dyn grim_tensor::BackendStorage>,
    exps_gate: Arc<dyn grim_tensor::BackendStorage>,
    exps_up: Arc<dyn grim_tensor::BackendStorage>,
    exps_down: Arc<dyn grim_tensor::BackendStorage>,
    fingerprint: (usize, usize, usize, u64),
}

/// Format tag for [`ResidentWeights::fingerprint`]. AWQ folds bits/group in
/// so heterogeneous AWQ configs never alias one stack.
fn stack_tag_awq(bits: u8, group_size: usize) -> u64 {
    3 | ((bits as u64) << 32) | ((group_size as u64) << 40)
}

/// Stack tags for [`ResidentWeights::fingerprint`].
const TAG_F32: u64 = 0;
const TAG_W8A8_INT8: u64 = 1;
const TAG_W8A8_FP8: u64 = 2;
/// GELU path stores a dummy up projection — it must never alias the f32
/// stack of the same dims (which carries a real up).
const TAG_GELU: u64 = 5;

/// Device-resident routing triple: sortless (token, expert, weight) buffers.
struct RoutingBuffers {
    tokens: Arc<dyn grim_tensor::BackendStorage>,
    experts: Arc<dyn grim_tensor::BackendStorage>,
    weights: Arc<dyn grim_tensor::BackendStorage>,
}

impl Default for CharonCache {
    fn default() -> Self {
        Self::new()
    }
}

impl CharonCache {
    pub fn new() -> Self {
        Self {
            resident: Mutex::new(None),
            routing: Mutex::new(None),
            w8a8: Mutex::new(None),
            w8a8fp8: Mutex::new(None),
            awq: Mutex::new(None),
            mxfp4: Mutex::new(None),
            whitecrow: Mutex::new(None),
            kq_native: Mutex::new(None),
            whitecrow_refused: Mutex::new(false),
            last_dispatch: Mutex::new(DispatchKind::F32Dequant),
        }
    }

    /// Clear any cached resident weights (used when a model's expert set is
    /// reloaded or replaced).
    pub fn invalidate(&self) {
        *self.resident.lock().unwrap_or_else(|e| e.into_inner()) = None;
        *self.whitecrow.lock().unwrap_or_else(|e| e.into_inner()) = None;
        *self.kq_native.lock().unwrap_or_else(|e| e.into_inner()) = None;
        *self.whitecrow_refused.lock().unwrap_or_else(|e| e.into_inner()) = false;
        *self.routing.lock().unwrap_or_else(|e| e.into_inner()) = None;
        *self.w8a8.lock().unwrap_or_else(|e| e.into_inner()) = None;
        *self.w8a8fp8.lock().unwrap_or_else(|e| e.into_inner()) = None;
        *self.awq.lock().unwrap_or_else(|e| e.into_inner()) = None;
        *self.mxfp4.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }

    /// Engagement sentinel (WI-gpu-native-moe Phase 0.3): true once the
    /// device-resident routing scratch has been populated by a successful
    /// `fused_moe_dispatch_from_logits` call. Parity tests assert this to
    /// fail loudly on silent `Ok(None)` fallback (vacuous pass guard).
    pub fn is_routing_engaged(&self) -> bool {
        self.routing
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
    }

    /// Arm sentinel (WI-gpu-native-moe Phase 2): which numeric path the last
    /// dispatch took. Quantized-engagement tests assert `W8a8Native` for
    /// packed-int8 experts to prove the native arm ran.
    pub fn last_dispatch_kind(&self) -> DispatchKind {
        *self.last_dispatch.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn record_dispatch(&self, kind: DispatchKind) {
        *self.last_dispatch.lock().unwrap_or_else(|e| e.into_inner()) = kind;
    }
}

/// Environment gate for the Charon grouped-dispatch path (Phase 3a). Set
/// `GRIM_MOE_CHARON=0` to force the per-expert loop reference path for parity
/// debugging.
fn charon_enabled() -> bool {
    std::env::var("GRIM_MOE_CHARON").as_deref() != Ok("0")
}

/// Fused MoE dispatch: routes `x` through the selected experts and returns the
/// weighted sum (plus optional shared-expert contribution).
///
/// * `experts` - per-expert FFN weights.
/// * `shared_expert` - optional always-on shared expert applied to all tokens.
/// * `routings` - per-token selected-expert indices + combine weights (top-k).
/// * `routed_scaling_factor` - architecture-specific scale (DeepSeek dedup gating).
/// * `cache` - Charon resident-weight cache; the stacked weight buffers are
///   built here on first use and reused thereafter (no per-decode round-trips).
///
/// On ROCm with `GRIM_MOE_CHARON` left enabled, the dispatch stacks the expert
/// weights (once, into `cache`) and runs Charon's grouped kernel
/// (`grim_moe_fused_grouped`) — one token-sorted launch, all active experts
/// evaluated in-register with SiLU fused, output accumulated by atomicAdd.
/// That is the exact math of the per-expert loop, cross-checked by
/// `tests/golden_charon_moe_gpu.rs` (≤1e-3 max-abs-diff). On any other backend
/// (or when the kernel is unavailable) it falls back to the per-expert loop.
///
/// M2 (PLAN-kernel-fusion): resolve (lazily allocate, once per shape key) the
/// device routing scratch triple + resident stacked expert weights that the
/// Charon grouped dispatch consumes. Used by `fused_moe_dispatch_from_logits`
/// and by the capture-safe graph path (`Lfm2Block::moe_forward_graph`), which
/// must hit the already-allocated buffers inside a capture bracket — callers
/// warm up once before `begin_capture` so every later call is a cache hit.
pub type CharonScratchBuffers = (
    Arc<dyn grim_tensor::BackendStorage>,
    Arc<dyn grim_tensor::BackendStorage>,
    Arc<dyn grim_tensor::BackendStorage>,
    Arc<dyn grim_tensor::BackendStorage>,
    Arc<dyn grim_tensor::BackendStorage>,
    Arc<dyn grim_tensor::BackendStorage>,
);

/// Build (once) the WhiteCrow u4-group128 resident expert blobs behind
/// `GRIM_MOE_NATIVE_WHITECROW`. Each expert weight (any K-quant scheme the
/// requantizer covers: Q4K/Q5K/Q6K/Q3K/Q2K/IQ4NL) is converted through the
/// proven `requant_kquant_to_whitecrow` path — D2H, host dequant, OSTQuant
/// group-128 re-encode, disk cache — then the three per-expert device
/// buffers are read back and concatenated into expert-strided blobs.
///
/// The conversion D2Hs, so this MUST run outside any graph-capture bracket:
/// call it from the eager dispatch or at decode-graph pool-build time; a
/// capture-time build poisons the capture.
/// Build (once) the native K-quant dispatch handles: per-projection device
/// pointer arrays over the per-expert packed banks. Only h2d-uploads 3 x
/// num_experts x 8 bytes — the weights themselves are the model's own. The
/// upload is tiny but still an H2D of host-owned data, so the capture path
/// must call this from pool-build time (or peek via [`peek_kq_native`]),
/// never from inside a capture bracket.
fn pick_expert_bytes(
    e: &MoeExpert,
    pick: fn(&MoeExpert) -> &grim_tensor::Tensor,
) -> u64 {
    pick(e)
        .storage()
        .as_ref()
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .map(|st| st.bytes() as u64)
        .unwrap_or(0)
}

pub(crate) fn ensure_kq_native(
    ordinal: usize,
    experts: &[MoeExpert],
    cache: &CharonCache,
) -> Result<KqNativeResident> {
    // GRIM_MOE_KQ_FRESH=1: rebuild the pointer arrays on every call —
    // staleness probe for the page-fault chase (are baked bank pointers
    // going dangling between build and dispatch?).
    let fresh = matches!(
        std::env::var("GRIM_MOE_KQ_FRESH").as_deref(),
        Ok("1" | "true")
    );
    if !fresh {
        let guard = cache.kq_native.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(r) = guard.as_ref() {
            return Ok(KqNativeResident {
                gate_ptrs: Arc::clone(&r.gate_ptrs),
                up_ptrs: Arc::clone(&r.up_ptrs),
                down_ptrs: Arc::clone(&r.down_ptrs),
                num_experts: r.num_experts,
                gate_bytes: r.gate_bytes,
                down_bytes: r.down_bytes,
                down_q4k: r.down_q4k,
            });
        }
    }
    let num_experts = experts.len();
    if num_experts == 0 {
        return Err(grim_core::error::Error::Backend(
            "kq native: no experts".into(),
        ));
    }
    let rocm = grim_backend_rocm::RocmDevice::shared(ordinal);
    // Scheme contract: gate/up decode as IQ3_S, down as Q4_K (xing4.0's
    // banks). Any other combination refuses - a wrong per-format decode is
    // exactly the class of bug that size checks cannot see.
    let scheme_of = |w: &grim_tensor::Tensor| -> Result<grim_tensor::KQuantScheme> {
        match &w.dtype().storage {
            grim_tensor::Storage::KQuant(sch) => Ok(*sch),
            other => Err(grim_core::error::Error::Backend(format!(
                "kq native: expert weight storage {other:?} is not packed K-quant"
            ))),
        }
    };
    for e in experts {
        if scheme_of(&e.gate.weight)? != grim_tensor::KQuantScheme::IQ3S
            || scheme_of(&e.up.weight)? != grim_tensor::KQuantScheme::IQ3S
            || !matches!(
                scheme_of(&e.down.weight)?,
                grim_tensor::KQuantScheme::Q4K | grim_tensor::KQuantScheme::IQ3S
            )
        {
            let (g, u, d) = (
                scheme_of(&e.gate.weight)?,
                scheme_of(&e.up.weight)?,
                scheme_of(&e.down.weight)?,
            );
            let gd = e.gate.weight.shape().dims();
            let dd = e.down.weight.shape().dims();
            let gb2 = pick_expert_bytes(e, |x| &x.gate.weight);
            let db2 = pick_expert_bytes(e, |x| &x.down.weight);
            return Err(grim_core::error::Error::Backend(format!(
                "kq native: banks are not the IQ3S gate/up + Q4K down set this kernel decodes: gate {g:?} up {u:?} down {d:?}, gate dims {gd:?} bytes {gb2}, down dims {dd:?} bytes {db2}"
            )));
        }
    }
    let ptr_array = |pick: fn(&MoeExpert) -> &grim_tensor::Tensor| -> Result<
        Arc<dyn grim_tensor::backend::BackendStorage>,
    > {
        let mut addrs: Vec<u64> = Vec::with_capacity(num_experts);
        for (ei, e) in experts.iter().enumerate() {
            let st = pick(e)
                .storage()
                .as_ref()
                .as_any()
                .downcast_ref::<grim_backend_rocm::RocmStorage>()
                .ok_or_else(|| {
                    grim_core::error::Error::Backend("kq native: weight not RocmStorage".into())
                })?;
            if !st.device_ptr_is_valid() {
                return Err(grim_core::error::Error::Backend(format!(
                    "kq native: expert {ei} bank has an invalid device pointer (not uploaded?)"
                )));
            }
            let addr = st.device_ptr_u64().ok_or_else(|| {
                grim_core::error::Error::Backend("kq native: bank has no device ptr".into())
            })?;
            if std::env::var_os("GRIM_MOE_KQ_DEBUG").is_some() {
                eprintln!("[kq-debug] expert {ei}: ptr {addr:#x} bytes {}", st.bytes());
            }
            addrs.push(addr);
        }
        // u64 addresses upload as raw little-endian bytes (never through
        // f32); U32 words, 2 per address.
        let bytes: Vec<u8> = addrs.iter().flat_map(|a| a.to_le_bytes()).collect();
        let shape = grim_tensor::Shape::new(vec![addrs.len() * 2]);
        let st = rocm.from_cpu_bytes(
            &bytes,
            &shape,
            grim_tensor::DType {
                arith: grim_tensor::ArithType::U32,
                storage: grim_tensor::Storage::Native,
            },
        )?;
        Ok(Arc::from(st))
    };
    let gate_ptrs = ptr_array(|e| &e.gate.weight)?;
    let up_ptrs = ptr_array(|e| &e.up.weight)?;
    let down_ptrs = ptr_array(|e| &e.down.weight)?;
    // PER-EXPERT byte counts — the launcher derives the per-row stride as
    // bytes / rows-of-one-expert; a total across all experts would inflate
    // the stride by num_experts and read 64x past the bank (the page fault
    // the phase-mask bisection chased).
    let gate_bytes = pick_expert_bytes(&experts[0], |e| &e.gate.weight);
    let down_bytes = pick_expert_bytes(&experts[0], |e| &e.down.weight);
    let down_q4k = scheme_of(&experts[0].down.weight)? == grim_tensor::KQuantScheme::Q4K;
    let resident = KqNativeResident {
        gate_ptrs,
        up_ptrs,
        down_ptrs,
        num_experts,
        gate_bytes,
        down_bytes,
        down_q4k,
    };
    *cache.kq_native.lock().unwrap_or_else(|e| e.into_inner()) =
        Some(KqNativeResident {
            gate_ptrs: Arc::clone(&resident.gate_ptrs),
            up_ptrs: Arc::clone(&resident.up_ptrs),
            down_ptrs: Arc::clone(&resident.down_ptrs),
            num_experts: resident.num_experts,
            gate_bytes: resident.gate_bytes,
            down_bytes: resident.down_bytes,
            down_q4k: resident.down_q4k,
        });
    Ok(resident)
}

/// Cache-only peek for capture brackets (see `peek_whitecrow_stacks`).
///
/// Not called yet: the kq-native capture bracket that needs it is still
/// unwired, and `-D dead_code` is on. Kept (rather than deleted) because the
/// capture step must never call the `ensure_` variant -- a miss there D2Hs
/// inside the bracket and poisons capture.
#[allow(dead_code)]
pub(crate) fn peek_kq_native(cache: &CharonCache) -> Option<KqNativeResident> {
    let guard = cache.kq_native.lock().unwrap_or_else(|e| e.into_inner());
    guard.as_ref().map(|r| KqNativeResident {
        gate_ptrs: Arc::clone(&r.gate_ptrs),
        up_ptrs: Arc::clone(&r.up_ptrs),
        down_ptrs: Arc::clone(&r.down_ptrs),
        num_experts: r.num_experts,
        gate_bytes: r.gate_bytes,
        down_bytes: r.down_bytes,
        down_q4k: r.down_q4k,
    })
}

/// Cache-only peek: returns the resident WhiteCrow stacks if built, WITHOUT
/// attempting a conversion. The capture step must use this (never `ensure_`),
/// because a miss would D2H inside the capture bracket and poison it.
pub(crate) fn peek_whitecrow_stacks(cache: &CharonCache) -> Option<WhiteCrowResident> {
    let guard = cache.whitecrow.lock().unwrap_or_else(|e| e.into_inner());
    guard.as_ref().map(|r| WhiteCrowResident {
        gate: Arc::clone(&r.gate),
        up: Arc::clone(&r.up),
        down: Arc::clone(&r.down),
        gate_stride: r.gate_stride,
        down_stride: r.down_stride,
    })
}

pub(crate) fn ensure_whitecrow_scratch(
    ordinal: usize,
    experts: &[MoeExpert],
    cache: &CharonCache,
) -> Result<WhiteCrowResident> {
    {
        let guard = cache.whitecrow.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(r) = guard.as_ref() {
            return Ok(WhiteCrowResident {
                gate: Arc::clone(&r.gate),
                up: Arc::clone(&r.up),
                down: Arc::clone(&r.down),
                gate_stride: r.gate_stride,
                down_stride: r.down_stride,
            });
        }
    }
    if !matches!(
        std::env::var("GRIM_MOE_NATIVE_WHITECROW").as_deref(),
        Ok("1" | "true" | "on")
    ) {
        return Err(grim_core::error::Error::Backend(
            "whitecrow stacks: GRIM_MOE_NATIVE_WHITECROW not enabled".into(),
        ));
    }
    let num_experts = experts.len();
    if num_experts == 0 {
        return Err(grim_core::error::Error::Backend(
            "whitecrow stacks: no experts".into(),
        ));
    }
    let hidden = experts[0].gate.weight.shape().dim(1).unwrap_or(0);
    let inter = experts[0].gate.weight.shape().dim(0).unwrap_or(0);
    let down_rows = experts[0].down.weight.shape().dim(0).unwrap_or(0);
    if hidden == 0 || inter == 0 || down_rows != hidden || hidden % 128 != 0 || inter % 128 != 0 {
        return Err(grim_core::error::Error::Backend(format!(
            "whitecrow stacks: geometry hidden={hidden} inter={inter} down_rows={down_rows}              not group-128 aligned"
        )));
    }
    {
        let refused = cache
            .whitecrow_refused
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if *refused {
            return Err(grim_core::error::Error::Backend(
                "whitecrow stacks: previously refused (budget/scheme)".into(),
            ));
        }
    }
    let rocm = grim_backend_rocm::RocmDevice::shared(ordinal);
    let dev = grim_backend_rocm::RocmDevice::shared(ordinal);

    // VRAM budget: the stacks are PER LAYER and every layer's set must stay
    // resident for as long as its cache lives (a decode-graph replay touches
    // all layers in one pass). Building layer by layer until the allocator
    // dies produced a GPU page fault, so the total committed bytes are
    // bounded here: GRIM_MOE_WHITECROW_BUDGET_MB (default 1024 MiB process-
    // wide). A refusal is latched per cache and degrades to the arms below.
    let per_layer_bytes =
        ((inter * hidden / 2 + inter * (hidden / 128) * 3 + 24)
            + (inter * hidden / 2 + inter * (hidden / 128) * 3 + 24)
            + (hidden * inter / 2 + hidden * (inter / 128) * 3 + 24))
            * num_experts;
    static WC_COMMITTED: std::sync::atomic::AtomicU64 =
        std::sync::atomic::AtomicU64::new(0);
    // Default 0 = the arm stays OFF unless the operator sized the budget for
    // the WHOLE model (layers x per-layer bytes). A partial build — some
    // layers on WhiteCrow stacks, some on the fallback — strands the built
    // bytes and mixes numeric paths mid-model, so engagement is all-or-
    // nothing by construction: the budget must cover every MoE layer's
    // stacks, and `ensure` has no global layer count, so the operator (or a
    // model-level prewarm that measured one layer) sets it.
    let budget_bytes: u64 = std::env::var("GRIM_MOE_WHITECROW_BUDGET_MB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(|m| m.saturating_mul(1048576))
        .unwrap_or(0);
    if WC_COMMITTED.load(std::sync::atomic::Ordering::Relaxed) as u64 + per_layer_bytes as u64
        > budget_bytes
    {
        *cache
            .whitecrow_refused
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = true;
        return Err(grim_core::error::Error::Backend(format!(
            "whitecrow stacks: per-layer {per_layer_bytes} B exceeds the committed budget              (GRIM_MOE_WHITECROW_BUDGET_MB, {budget_bytes} B); resident stacks do not fit"
        )));
    }

    // Requantize one packed K-quant weight tensor into host blob bytes.
    let to_blob = |w: &grim_tensor::Tensor,
                   n: usize,
                   k: usize,
                   scheme: grim_tensor::KQuantScheme|
     -> Result<Vec<u8>> {
        let rocm_s = w
            .storage()
            .as_ref()
            .as_any()
            .downcast_ref::<grim_backend_rocm::RocmStorage>()
            .ok_or_else(|| {
                grim_core::error::Error::Backend("whitecrow: weight not RocmStorage".into())
            })?;
        let conv = rocm.requant_kquant_to_whitecrow(rocm_s, n, k, scheme)?;
        let qw_s = conv.qweight_rocm()?;
        let sc_s = conv.scales_rocm()?;
        let zr_s = conv.zeros_rocm()?;
        let qw = qw_s.copy_to_host()?;
        let sc = sc_s.copy_to_host()?;
        let zr = zr_s.copy_to_host()?;
        let mut blob = Vec::with_capacity(qw.len() + sc.len() + zr.len() + 24);
        blob.extend_from_slice(&(qw.len() as u64).to_le_bytes());
        blob.extend_from_slice(&qw);
        blob.extend_from_slice(&(sc.len() as u64).to_le_bytes());
        blob.extend_from_slice(&sc);
        blob.extend_from_slice(&(zr.len() as u64).to_le_bytes());
        blob.extend_from_slice(&zr);
        Ok(blob)
    };

    // Per-projection scheme, read off the packed storage dtype.
    let scheme_of = |w: &grim_tensor::Tensor| -> Result<grim_tensor::KQuantScheme> {
        match &w.dtype().storage {
            grim_tensor::Storage::KQuant(sch) => Ok(*sch),
            other => Err(grim_core::error::Error::Backend(format!(
                "whitecrow stacks: unsupported expert weight storage {other:?}"
            ))),
        }
    };

    let stack = |pick: fn(&MoeExpert) -> &grim_tensor::Tensor,
                 n: usize,
                 k: usize|
     -> Result<(std::sync::Arc<dyn grim_tensor::backend::BackendStorage>, u64)> {
        let mut blob: Vec<u8> = Vec::new();
        for e in experts {
            let w = pick(e);
            blob.extend_from_slice(&to_blob(w, n, k, scheme_of(w)?)?);
        }
        let stride = (blob.len() / num_experts) as u64;
        let shape = grim_tensor::Shape::new(vec![blob.len()]);
        let st = grim_backend_rocm::RocmDevice::shared(ordinal).from_cpu_bytes(
            &blob,
            &shape,
            grim_tensor::DType {
                arith: grim_tensor::ArithType::U8,
                storage: grim_tensor::Storage::Native,
            },
        )?;
        Ok((std::sync::Arc::from(st), stride))
    };

    let (gate, gate_stride) = stack(|e| &e.gate.weight, inter, hidden)?;
    let (up, up_stride) = stack(|e| &e.up.weight, inter, hidden)?;
    if up_stride != gate_stride {
        return Err(grim_core::error::Error::Backend(
            "whitecrow stacks: gate/up stride mismatch".into(),
        ));
    }
    let (down, down_stride) = stack(|e| &e.down.weight, hidden, inter)?;

    let f32_mib = (inter * hidden * 2 + hidden * inter) * num_experts * 4;
    eprintln!(
        "[charon] whitecrow stacks: {num_experts} experts, gate/up blob {:.1} MiB, down blob {:.1} MiB (f32 stacks would be {:.1} MiB)",
        (gate_stride * num_experts as u64) as f64 / 1048576.0,
        (down_stride * num_experts as u64) as f64 / 1048576.0,
        f32_mib as f64 / 1048576.0,
    );

    let resident = WhiteCrowResident {
        gate,
        up,
        down,
        gate_stride,
        down_stride,
    };
    WC_COMMITTED.fetch_add(
        per_layer_bytes as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    *cache.whitecrow.lock().unwrap_or_else(|e| e.into_inner()) = Some(WhiteCrowResident {
        gate: Arc::clone(&resident.gate),
        up: Arc::clone(&resident.up),
        down: Arc::clone(&resident.down),
        gate_stride: resident.gate_stride,
        down_stride: resident.down_stride,
    });
    let _ = dev;
    Ok(resident)
}

pub fn ensure_charon_scratch(
    ordinal: usize,
    seq_len: usize,
    top_k: usize,
    experts: &[MoeExpert],
    cache: &CharonCache,
) -> Result<CharonScratchBuffers> {
    let rocm = Arc::new(grim_backend_rocm::RocmDevice::shared(ordinal));
    let num_experts = experts.len();
    let hidden = experts[0].gate.weight.shape().dim(1).unwrap_or(0);
    let inter = experts[0].gate.weight.shape().dim(0).unwrap_or(0);
    if num_experts == 0 || hidden == 0 || inter == 0 {
        return Err(grim_core::error::Error::Backend(
            "ensure_charon_scratch: degenerate expert dims".into(),
        ));
    }

    let (tokens_buf, experts_buf, weights_buf) = {
        let mut guard = cache.routing.lock().unwrap_or_else(|e| e.into_inner());
        match guard.as_ref() {
            Some((sl, tk, bufs)) if *sl == seq_len && *tk == top_k => (
                Arc::clone(&bufs.tokens),
                Arc::clone(&bufs.experts),
                Arc::clone(&bufs.weights),
            ),
            _ => {
                let num_pairs = seq_len * top_k;
                let tokens = Arc::from(rocm.zeros(
                    &Shape::new(vec![num_pairs]),
                    DType {
                        arith: grim_tensor::ArithType::U32,
                        storage: grim_tensor::Storage::Native,
                    },
                )?);
                let experts_b = Arc::from(rocm.zeros(
                    &Shape::new(vec![num_pairs]),
                    DType {
                        arith: grim_tensor::ArithType::U32,
                        storage: grim_tensor::Storage::Native,
                    },
                )?);
                let weights = Arc::from(rocm.zeros(&Shape::new(vec![num_pairs]), DType::F32)?);
                *guard = Some((
                    seq_len,
                    top_k,
                    RoutingBuffers {
                        tokens: Arc::clone(&tokens),
                        experts: Arc::clone(&experts_b),
                        weights: Arc::clone(&weights),
                    },
                ));
                (tokens, experts_b, weights)
            }
        }
    };

    let (gate_buf, up_buf, down_buf) = {
        let mut guard = cache.resident.lock().unwrap_or_else(|e| e.into_inner());
        let key = (num_experts, hidden, inter, TAG_F32);
        match guard.as_ref() {
            Some(r) if r.fingerprint == key => {
                (Arc::clone(&r.gate), Arc::clone(&r.up), Arc::clone(&r.down))
            }
            _ => {
                let (gate_flat, up_flat, down_flat) =
                    stack_expert_weights(experts, num_experts, hidden, inter)?;
                let gate = Arc::from(rocm.from_cpu(
                    &gate_flat,
                    &Shape::new(vec![gate_flat.len()]),
                    DType::F32,
                )?);
                let up = Arc::from(rocm.from_cpu(
                    &up_flat,
                    &Shape::new(vec![up_flat.len()]),
                    DType::F32,
                )?);
                let down = Arc::from(rocm.from_cpu(
                    &down_flat,
                    &Shape::new(vec![down_flat.len()]),
                    DType::F32,
                )?);
                *guard = Some(ResidentWeights {
                    gate: Arc::clone(&gate),
                    up: Arc::clone(&up),
                    down: Arc::clone(&down),
                    fingerprint: key,
                });
                (gate, up, down)
            }
        }
    };

    Ok((
        tokens_buf,
        experts_buf,
        weights_buf,
        gate_buf,
        up_buf,
        down_buf,
    ))
}

pub fn fused_moe_dispatch(
    dev: &dyn BackendDevice,
    x: &Tensor,
    experts: &[MoeExpert],
    shared_expert: Option<&MoeExpert>,
    routings: &[TokenRouting],
    routed_scaling_factor: f32,
    cache: &CharonCache,
) -> Result<Tensor> {
    if charon_enabled() {
        if let Some(out) = charon_grouped_dispatch(
            dev,
            x,
            experts,
            shared_expert,
            routings,
            routed_scaling_factor,
            cache,
        )? {
            return Ok(out);
        }
    }
    per_expert_loop(
        dev,
        x,
        experts,
        shared_expert,
        routings,
        routed_scaling_factor,
    )
}

/// Opt-in gate for native quantized arms that trail the f32 dequant path
/// on latency (measured gfx1201: fp8 1.8x, awq 1.7x slower at decode
/// shapes — per-element decode under an occupancy-starved launch).
/// `GRIM_MOE_NATIVE_FP8=1` / `GRIM_MOE_NATIVE_AWQ=1` engage them; default
/// is the proven dequant arm. int8 (latency-neutral) and MXFP4 (no dequant
/// alternative exists) are always on and ignore this gate.
fn native_quant_allowed(marker: &str) -> bool {
    let var = match marker {
        "fp8" => "GRIM_MOE_NATIVE_FP8",
        "awq" => "GRIM_MOE_NATIVE_AWQ",
        _ => return false,
    };
    matches!(
        std::env::var(var).as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
}

/// Upload per-token activation scales of 1.0 (v1 policy shared by the
/// native quantized arms: quantization error lives in the weight codes,
/// exactly like the dequant path, so native dispatch is accuracy-neutral
/// vs today's path and only saves weight traffic).
fn upload_ones_ascale(
    rocm: &grim_backend_rocm::RocmDevice,
    seq_len: usize,
) -> Result<Arc<dyn grim_tensor::backend::BackendStorage>> {
    let ones = vec![1.0f32; seq_len.max(1)];
    Ok(Arc::from(rocm.from_cpu(
        &ones,
        &Shape::new(vec![ones.len()]),
        DType::F32,
    )?))
}

/// Downcast a resident buffer to `RocmStorage` for a kernel launch.
fn as_rocm_storage<'a>(
    buf: &'a Arc<dyn grim_tensor::backend::BackendStorage>,
    label: &str,
) -> Result<&'a grim_backend_rocm::RocmStorage> {
    buf.as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .ok_or_else(|| grim_core::error::Error::Backend(format!("{label} not RocmStorage")))
}

/// Zero-output early return shared by the native quantized arms when
/// `num_pairs == 0` (no routed pairs: output is zeros + shared expert).
fn zero_moe_output(
    dev: &dyn BackendDevice,
    x: &Tensor,
    shared_expert: Option<&MoeExpert>,
) -> Result<Tensor> {
    let out = dev.zeros(x.shape(), DType::F32)?;
    let out_t = Tensor::new(
        Arc::from(out),
        x.shape().clone(),
        DType::F32,
        x.provenance().clone(),
        x.device().clone(),
    );
    shared_expert_tail(dev, out_t, x, shared_expert)
}

/// Fully device-resident (D2D) MoE dispatch: routing is computed **on-device**
/// from `logits` via `grim_moe_route_topk`, and the expert evaluation is launched
/// from device-resident routing buffers via `moe_fused_dispatch_resident_routing`.
///
/// This eliminates the two host round-trips of the legacy path: the gate logits
/// no longer D2H, and the routing table is no longer H2D-uploaded per launch.
/// The only host interaction is the (unchanged) 4-byte sampled-token readback at
/// the sampler, which is inherent to token streaming.
///
/// * `logits` - device-resident gate projection `[seq_len, num_experts]`.
/// * `top_k` - experts-per-token selection width.
/// * `route_mode` - gating transform: 0 = softmax (global denom, HF Qwen),
///   1 = sqrt-softplus (DeepSeek-V4), 2 = sigmoid+bias (DeepSeek-V2/V3 dedup),
///   3 = softmax renormalized over top-k only (GLM/Qwen `normalize_weights`).
///
/// Returns `Ok(None)` when the path cannot run (non-ROCm, unavailable kernel, or
/// `GRIM_MOE_CHARON=0`), so the caller falls back to the host-routing path.
#[allow(clippy::too_many_arguments)]
pub fn fused_moe_dispatch_from_logits(
    dev: &dyn BackendDevice,
    x: &Tensor,
    logits: &Tensor,
    experts: &[MoeExpert],
    shared_expert: Option<&MoeExpert>,
    top_k: usize,
    routed_scaling_factor: f32,
    route_mode: i32,
    cache: &CharonCache,
) -> Result<Option<Tensor>> {
    fused_moe_dispatch_from_logits_with_bias(
        dev,
        x,
        logits,
        None,
        experts,
        shared_expert,
        top_k,
        routed_scaling_factor,
        route_mode,
        false, // no bias path: caller decides normalization
        cache,
    )
}

/// [`fused_moe_dispatch_from_logits`] with the per-expert `e_score_correction_bias`.
///
/// Only meaningful for `route_mode == 2` (sigmoid+bias, DeepSeek-V2/V3 and
/// Xing4.0's `noaux_tc`), where the kernel uses `sigmoid(logit) + bias[i]` to
/// *select* experts while the combine weight stays the raw `sigmoid(logit)`.
/// The bias must be a device-resident `[num_experts]` F32 tensor matching
/// `logits`' device; it is read on the device, so supplying it does not
/// reintroduce a host round-trip.
#[allow(clippy::too_many_arguments)]
pub fn fused_moe_dispatch_from_logits_with_bias(
    dev: &dyn BackendDevice,
    x: &Tensor,
    logits: &Tensor,
    bias: Option<&Tensor>,
    experts: &[MoeExpert],
    shared_expert: Option<&MoeExpert>,
    top_k: usize,
    routed_scaling_factor: f32,
    route_mode: i32,
    // Reference `norm_w`: normalize the top-k combine weights to sum 1
    // (llama.cpp build_moe_ffn). Xing4.0 declares expert_weights_norm = true.
    norm_weights: bool,
    cache: &CharonCache,
) -> Result<Option<Tensor>> {
    if !charon_enabled() {
        return Ok(None);
    }
    let dims = x.shape().dims();
    if dims.len() != 2 {
        return Ok(None);
    }
    let (seq_len, hidden) = (dims[0], dims[1]);
    if seq_len == 0 || hidden == 0 || experts.is_empty() {
        return Ok(None);
    }
    let ordinal = match x.device() {
        Device::Rocm(o) => *o,
        _ => return Ok(None),
    };
    let num_experts = experts.len();
    let inter = experts[0].gate.weight.shape().dim(0).unwrap_or(0);
    if inter == 0 {
        return Ok(None);
    }
    let down_rows = experts[0].down.weight.shape().dim(0).unwrap_or(0);
    if down_rows != hidden {
        return Ok(None);
    }

    let rocm = Arc::new(grim_backend_rocm::RocmDevice::shared(ordinal));
    let logits_rocm = logits
        .storage()
        .as_ref()
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .ok_or_else(|| grim_tensor::Error::Backend("logits is not RocmStorage".into()))?;
    let x_rocm = x
        .storage()
        .as_ref()
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .ok_or_else(|| grim_tensor::Error::Backend("x is not RocmStorage".into()))?;

    // Resolve (lazily allocate) the device routing scratch buffers.
    let num_pairs = seq_len * top_k;
    let (tokens_buf, experts_buf, weights_buf) = {
        let mut guard = cache.routing.lock().unwrap_or_else(|e| e.into_inner());
        match guard.as_ref() {
            Some((sl, tk, bufs)) if *sl == seq_len && *tk == top_k => (
                Arc::clone(&bufs.tokens),
                Arc::clone(&bufs.experts),
                Arc::clone(&bufs.weights),
            ),
            _ => {
                let tokens = Arc::from(rocm.zeros(
                    &Shape::new(vec![num_pairs]),
                    DType {
                        arith: grim_tensor::ArithType::U32,
                        storage: grim_tensor::Storage::Native,
                    },
                )?);
                let experts = Arc::from(rocm.zeros(
                    &Shape::new(vec![num_pairs]),
                    DType {
                        arith: grim_tensor::ArithType::U32,
                        storage: grim_tensor::Storage::Native,
                    },
                )?);
                let weights = Arc::from(rocm.zeros(&Shape::new(vec![num_pairs]), DType::F32)?);
                *guard = Some((
                    seq_len,
                    top_k,
                    RoutingBuffers {
                        tokens: Arc::clone(&tokens),
                        experts: Arc::clone(&experts),
                        weights: Arc::clone(&weights),
                    },
                ));
                (tokens, experts, weights)
            }
        }
    };

    let tokens_rocm = tokens_buf
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .ok_or_else(|| grim_tensor::Error::Backend("tokens not RocmStorage".into()))?;
    let experts_rocm = experts_buf
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .ok_or_else(|| grim_tensor::Error::Backend("experts not RocmStorage".into()))?;
    let weights_rocm = weights_buf
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .ok_or_else(|| grim_tensor::Error::Backend("weights not RocmStorage".into()))?;

    // 1. Device-side routing (D2D): write sortless routing triples.
    // `bias` is consumed on the device by route_mode 2; a non-ROCm or
    // wrong-width bias degrades to the unbiased selection rather than failing.
    let bias_rocm: Option<&grim_backend_rocm::RocmStorage> = bias.and_then(|b| {
        if b.shape().elem_count() != experts.len() || b.device() != logits.device() {
            return None;
        }
        b.storage()
            .as_any()
            .downcast_ref::<grim_backend_rocm::RocmStorage>()
    });
    rocm.moe_route_topk_on_device(
        logits_rocm,
        bias_rocm,
        tokens_rocm,
        experts_rocm,
        weights_rocm,
        seq_len,
        num_experts,
        top_k,
        route_mode,
        norm_weights,
    )?;

    // Native K-quant arm (default for all-IQ3_S banks, e.g. xing4.0):
    // decodes the ALREADY-RESIDENT per-expert banks in-register via
    // device pointer arrays — zero extra weight VRAM, no budget gate. GRIM_
    // MOE_KQ_NATIVE=0 disables. The build only uploads 3 pointer arrays, but
    // that is still an H2D of host-owned data, so the capture path prewarms
    // at pool-build and peeks (same discipline as the WhiteCrow arm).
    if !matches!(
        std::env::var("GRIM_MOE_KQ_NATIVE").as_deref(),
        Ok("0" | "false" | "off")
    ) {
        if let Ok(kq) = ensure_kq_native(ordinal, experts, cache) {
            cache.record_dispatch(DispatchKind::KqNative);
            let out_shape = Shape::new(vec![seq_len, hidden]);
            let out_storage_b = rocm.zeros(&out_shape, DType::F32)?;
            let out_storage = out_storage_b
                .as_any()
                .downcast_ref::<grim_backend_rocm::RocmStorage>()
                .ok_or_else(|| grim_tensor::Error::Backend("kq out not RocmStorage".into()))?;
            if num_pairs > 0 {
                let g_ptrs = kq
                    .gate_ptrs
                    .as_any()
                    .downcast_ref::<grim_backend_rocm::RocmStorage>()
                    .ok_or_else(|| grim_tensor::Error::Backend("kq gate ptrs not RocmStorage".into()))?;
                let u_ptrs = kq
                    .up_ptrs
                    .as_any()
                    .downcast_ref::<grim_backend_rocm::RocmStorage>()
                    .ok_or_else(|| grim_tensor::Error::Backend("kq up ptrs not RocmStorage".into()))?;
                let d_ptrs = kq
                    .down_ptrs
                    .as_any()
                    .downcast_ref::<grim_backend_rocm::RocmStorage>()
                    .ok_or_else(|| grim_tensor::Error::Backend("kq down ptrs not RocmStorage".into()))?;
                rocm.moe_fused_dispatch_kq_native_into(
                    x_rocm,
                    g_ptrs,
                    u_ptrs,
                    d_ptrs,
                    tokens_rocm,
                    experts_rocm,
                    weights_rocm,
                    num_pairs,
                    out_storage,
                    hidden,
                    inter,
                    routed_scaling_factor,
                    // PER-ROW strides: gate/up are [inter, hidden] (rows =
                    // inter), down is [hidden, inter] (rows = hidden). The
                    // kernel derives blocks-per-row from these; passing the
                    // per-expert TOTAL inflated g_sb 1024x and read megabytes
                    // past each row (page fault -> sync hang).
                    kq.gate_bytes / inter as u64,
                    kq.down_bytes / hidden as u64,
                    i32::from(kq.down_q4k),
                )?;
            }
            let out_t = Tensor::new(
                Arc::from(out_storage_b),
                out_shape,
                DType::F32,
                x.provenance().clone(),
                x.device().clone(),
            );
            return shared_expert_tail(dev, out_t, x, shared_expert).map(Some);
        }
    }

    // WhiteCrow u4-group128 arm (GRIM_MOE_NATIVE_WHITECROW=1): K-quant
    // expert banks requantized once to OSTQuant blobs (~0.5 B/param vs the
    // f32 stacks' 4 B/param) and dispatched by the grouped in-register-decode
    // kernel. Env-gated because the requant is lossy relative to the
    // checkpoint's own quantization; a build failure here degrades to the
    // arms below (printed, not silent — the Q6K lm_head lesson).
    if let Ok(wc) = ensure_whitecrow_scratch(ordinal, experts, cache) {
        cache.record_dispatch(DispatchKind::WhiteCrowNative);
        let out_shape = Shape::new(vec![seq_len, hidden]);
        let out_storage_b = rocm.zeros(&out_shape, DType::F32)?;
        let out_storage = out_storage_b
            .as_any()
            .downcast_ref::<grim_backend_rocm::RocmStorage>()
            .ok_or_else(|| grim_tensor::Error::Backend("wc out not RocmStorage".into()))?;
        if num_pairs > 0 {
            rocm.moe_fused_dispatch_whitecrow_grouped_into(
                x_rocm,
                wc.gate
                    .as_any()
                    .downcast_ref::<grim_backend_rocm::RocmStorage>()
                    .ok_or_else(|| grim_tensor::Error::Backend("wc gate not RocmStorage".into()))?,
                wc.up
                    .as_any()
                    .downcast_ref::<grim_backend_rocm::RocmStorage>()
                    .ok_or_else(|| grim_tensor::Error::Backend("wc up not RocmStorage".into()))?,
                wc.down
                    .as_any()
                    .downcast_ref::<grim_backend_rocm::RocmStorage>()
                    .ok_or_else(|| grim_tensor::Error::Backend("wc down not RocmStorage".into()))?,
                tokens_rocm,
                experts_rocm,
                weights_rocm,
                num_pairs,
                &out_storage,
                hidden,
                inter,
                routed_scaling_factor,
                wc.gate_stride,
                wc.down_stride,
            )?;
        }
        let out_t = Tensor::new(
            Arc::from(out_storage_b),
            out_shape,
            DType::F32,
            x.provenance().clone(),
            x.device().clone(),
        );
        return shared_expert_tail(dev, out_t, x, shared_expert).map(Some);
    }

    // WI-gpu-native-moe Phase 2: native W8A8-int8 arm. When every expert
    // projection is a ROCm-resident CompressedTensorsW8A8Int8 packed blob,
    // dispatch straight from the packed stacks (no dequant, no fallback).
    // Anything else rides the f32 arm below (native f32, or any quantized
    // format dequantized at stack-build time).
    if experts_use_w8a8_native(experts) {
        let (gate_buf, up_buf, down_buf) = {
            let mut guard = cache.w8a8.lock().unwrap_or_else(|e| e.into_inner());
            let key = (num_experts, hidden, inter, TAG_W8A8_INT8);
            match guard.as_ref() {
                Some(r) if r.fingerprint == key => {
                    (Arc::clone(&r.gate), Arc::clone(&r.up), Arc::clone(&r.down))
                }
                _ => {
                    let (gate_flat, up_flat, down_flat) =
                        stack_w8a8_blobs(experts, num_experts, hidden, inter)?;
                    let pack_dtype = || DType {
                        arith: grim_tensor::ArithType::F32,
                        storage: grim_tensor::Storage::CompressedTensorsW8A8Int8,
                    };
                    let gate = Arc::from(rocm.from_cpu_bytes(
                        &gate_flat,
                        &Shape::new(vec![gate_flat.len()]),
                        pack_dtype(),
                    )?);
                    let up = Arc::from(rocm.from_cpu_bytes(
                        &up_flat,
                        &Shape::new(vec![up_flat.len()]),
                        pack_dtype(),
                    )?);
                    let down = Arc::from(rocm.from_cpu_bytes(
                        &down_flat,
                        &Shape::new(vec![down_flat.len()]),
                        pack_dtype(),
                    )?);
                    *guard = Some(ResidentWeights {
                        gate: Arc::clone(&gate),
                        up: Arc::clone(&up),
                        down: Arc::clone(&down),
                        fingerprint: key,
                    });
                    (gate, up, down)
                }
            }
        };

        // Per-token activation scale. v1 policy: 1.0 (quantization error
        // lives in the int8 codes, exactly like the dequant path — so the
        // native arm is accuracy-neutral vs today's path and only saves
        // weight traffic). Dynamic per-token max-scaling is a follow-up.
        let ones = vec![1.0f32; seq_len.max(1)];
        let ascale_buf: Arc<dyn grim_tensor::backend::BackendStorage> =
            Arc::from(rocm.from_cpu(&ones, &Shape::new(vec![ones.len()]), DType::F32)?);
        let ascale_rocm = ascale_buf
            .as_any()
            .downcast_ref::<grim_backend_rocm::RocmStorage>()
            .ok_or_else(|| grim_tensor::Error::Backend("ascale not RocmStorage".into()))?;

        cache.record_dispatch(DispatchKind::W8a8Native);
        if num_pairs == 0 {
            let out = dev.zeros(x.shape(), DType::F32)?;
            let out_t = Tensor::new(
                Arc::from(out),
                x.shape().clone(),
                DType::F32,
                x.provenance().clone(),
                x.device().clone(),
            );
            return shared_expert_tail(dev, out_t, x, shared_expert).map(Some);
        }

        let out_shape = Shape::new(vec![seq_len, hidden]);
        // WI-gpu-native-moe #2: prefer the V_DOT4 contraction when the
        // shape (hidden % 32) and the architecture allow it; the scalar
        // sortless kernel is the exact-math fallback. The variant is a
        // perf detail inside one numeric path (both recorded distinctly
        // so tests/benches can tell them apart).
        let use_dot4 =
            hidden % 32 == 0 && grim_backend_rocm::kernels::charon::dot4_supported(rocm.gcn_arch());
        let (out_storage, _handle) = if use_dot4 {
            cache.record_dispatch(DispatchKind::W8a8NativeDot4);
            rocm.moe_fused_dispatch_resident_routing_w8a8_int8_dot4(
                x_rocm,
                &*gate_buf,
                &*up_buf,
                &*down_buf,
                ascale_rocm,
                tokens_rocm,
                experts_rocm,
                weights_rocm,
                num_pairs,
                &out_shape,
                hidden,
                inter,
                routed_scaling_factor,
            )?
        } else {
            rocm.moe_fused_dispatch_resident_routing_w8a8_int8(
                x_rocm,
                &*gate_buf,
                &*up_buf,
                &*down_buf,
                ascale_rocm,
                tokens_rocm,
                experts_rocm,
                weights_rocm,
                num_pairs,
                &out_shape,
                hidden,
                inter,
                routed_scaling_factor,
            )?
        };

        let out_t = Tensor::new(
            Arc::from(out_storage),
            out_shape,
            DType::F32,
            x.provenance().clone(),
            x.device().clone(),
        );

        return shared_expert_tail(dev, out_t, x, shared_expert).map(Some);
    }

    // WI-gpu-native-moe Phase 2: native W8A8-fp8 arm (mirrors the int8 arm;
    // per-expert blobs carry ONE f32 scale — see `w8a8_fp8_strides`).
    // Gated by GRIM_MOE_NATIVE_FP8 (bench: 1.8x slower than dequant —
    // per-element fp8 decode under an occupancy-starved launch).
    if experts_use_w8a8_fp8_native(experts) && native_quant_allowed("fp8") {
        let (gate_buf, up_buf, down_buf) = {
            let mut guard = cache.w8a8fp8.lock().unwrap_or_else(|e| e.into_inner());
            let key = (num_experts, hidden, inter, TAG_W8A8_FP8);
            match guard.as_ref() {
                Some(r) if r.fingerprint == key => {
                    (Arc::clone(&r.gate), Arc::clone(&r.up), Arc::clone(&r.down))
                }
                _ => {
                    let (gate_flat, up_flat, down_flat) =
                        stack_w8a8_fp8_blobs(experts, num_experts, hidden, inter)?;
                    let pack_dtype = || DType {
                        arith: grim_tensor::ArithType::F32,
                        storage: grim_tensor::Storage::CompressedTensorsW8A8Fp8,
                    };
                    let gate = Arc::from(rocm.from_cpu_bytes(
                        &gate_flat,
                        &Shape::new(vec![gate_flat.len()]),
                        pack_dtype(),
                    )?);
                    let up = Arc::from(rocm.from_cpu_bytes(
                        &up_flat,
                        &Shape::new(vec![up_flat.len()]),
                        pack_dtype(),
                    )?);
                    let down = Arc::from(rocm.from_cpu_bytes(
                        &down_flat,
                        &Shape::new(vec![down_flat.len()]),
                        pack_dtype(),
                    )?);
                    *guard = Some(ResidentWeights {
                        gate: Arc::clone(&gate),
                        up: Arc::clone(&up),
                        down: Arc::clone(&down),
                        fingerprint: key,
                    });
                    (gate, up, down)
                }
            }
        };

        let ascale_buf = upload_ones_ascale(&rocm, seq_len)?;
        let ascale_rocm = as_rocm_storage(&ascale_buf, "ascale")?;

        cache.record_dispatch(DispatchKind::W8a8Fp8Native);
        if num_pairs == 0 {
            return zero_moe_output(dev, x, shared_expert).map(Some);
        }

        let out_shape = Shape::new(vec![seq_len, hidden]);
        let (out_storage, _handle) = rocm.moe_fused_dispatch_resident_routing_w8a8_fp8(
            x_rocm,
            &*gate_buf,
            &*up_buf,
            &*down_buf,
            ascale_rocm,
            tokens_rocm,
            experts_rocm,
            weights_rocm,
            num_pairs,
            &out_shape,
            hidden,
            inter,
            routed_scaling_factor,
        )?;

        let out_t = Tensor::new(
            Arc::from(out_storage),
            out_shape,
            DType::F32,
            x.provenance().clone(),
            x.device().clone(),
        );

        return shared_expert_tail(dev, out_t, x, shared_expert).map(Some);
    }

    // WI-gpu-native-moe Phase 2: native AWQ arm. Uniform bits/group across
    // the whole bank is enforced by `awq_uniform_config`; heterogeneous
    // banks fall through to the f32 dequant arm (which handles per-tensor
    // configs). Gated by GRIM_MOE_NATIVE_AWQ (bench: 1.7x slower than
    // dequant — per-element unpack under an occupancy-starved launch).
    if let Some((bits, group_size)) =
        awq_uniform_config(experts).filter(|_| native_quant_allowed("awq"))
    {
        let (gate_buf, up_buf, down_buf) = {
            let mut guard = cache.awq.lock().unwrap_or_else(|e| e.into_inner());
            let key = (num_experts, hidden, inter, stack_tag_awq(bits, group_size));
            match guard.as_ref() {
                Some(r) if r.fingerprint == key => {
                    (Arc::clone(&r.gate), Arc::clone(&r.up), Arc::clone(&r.down))
                }
                _ => {
                    let (gate_flat, up_flat, down_flat) =
                        stack_awq_blobs(experts, num_experts, hidden, inter, bits, group_size)?;
                    let pack_dtype = || DType {
                        arith: grim_tensor::ArithType::F32,
                        storage: grim_tensor::Storage::Awq(grim_tensor::dtype::AwqStorageConfig {
                            bits,
                            group_size,
                        }),
                    };
                    let gate = Arc::from(rocm.from_cpu_bytes(
                        &gate_flat,
                        &Shape::new(vec![gate_flat.len()]),
                        pack_dtype(),
                    )?);
                    let up = Arc::from(rocm.from_cpu_bytes(
                        &up_flat,
                        &Shape::new(vec![up_flat.len()]),
                        pack_dtype(),
                    )?);
                    let down = Arc::from(rocm.from_cpu_bytes(
                        &down_flat,
                        &Shape::new(vec![down_flat.len()]),
                        pack_dtype(),
                    )?);
                    *guard = Some(ResidentWeights {
                        gate: Arc::clone(&gate),
                        up: Arc::clone(&up),
                        down: Arc::clone(&down),
                        fingerprint: key,
                    });
                    (gate, up, down)
                }
            }
        };

        let ascale_buf = upload_ones_ascale(&rocm, seq_len)?;
        let ascale_rocm = as_rocm_storage(&ascale_buf, "ascale")?;

        cache.record_dispatch(DispatchKind::AwqNative);
        if num_pairs == 0 {
            return zero_moe_output(dev, x, shared_expert).map(Some);
        }

        let out_shape = Shape::new(vec![seq_len, hidden]);
        let (out_storage, _handle) = rocm.moe_fused_dispatch_resident_routing_awq(
            x_rocm,
            &*gate_buf,
            &*up_buf,
            &*down_buf,
            ascale_rocm,
            tokens_rocm,
            experts_rocm,
            weights_rocm,
            num_pairs,
            &out_shape,
            hidden,
            inter,
            bits,
            group_size,
            routed_scaling_factor,
        )?;

        let out_t = Tensor::new(
            Arc::from(out_storage),
            out_shape,
            DType::F32,
            x.provenance().clone(),
            x.device().clone(),
        );

        return shared_expert_tail(dev, out_t, x, shared_expert).map(Some);
    }

    // WI-gpu-native-moe Phase 2: native MXFP4 arm. The f32 dequant arm
    // CANNOT serve MXFP4 (no device dequant exists), so MXFP4 is
    // native-or-error — never silent fallback, never silent zeros.
    if experts_use_mxfp4_native(experts, hidden, inter) {
        let (cg, cu, cd, eg, eu, ed) = {
            let mut guard = cache.mxfp4.lock().unwrap_or_else(|e| e.into_inner());
            let key = (num_experts, hidden, inter, 4u64);
            match guard.as_ref() {
                Some(r) if r.fingerprint == key => (
                    Arc::clone(&r.codes_gate),
                    Arc::clone(&r.codes_up),
                    Arc::clone(&r.codes_down),
                    Arc::clone(&r.exps_gate),
                    Arc::clone(&r.exps_up),
                    Arc::clone(&r.exps_down),
                ),
                _ => {
                    let (cg_v, cu_v, cd_v, eg_v, eu_v, ed_v) =
                        stack_mxfp4(experts, num_experts, hidden, inter)?;
                    let pack_dtype = || DType {
                        arith: grim_tensor::ArithType::F32,
                        storage: grim_tensor::Storage::FloatPack(
                            grim_tensor::FloatPackScheme::MxFp4,
                        ),
                    };
                    // from_cpu_bytes failures propagate as dispatch errors
                    // (loud) — never unwrap into a panic inside serving.
                    let build =
                        |v: &Vec<u8>| -> Result<Arc<dyn grim_tensor::backend::BackendStorage>> {
                            Ok(Arc::from(rocm.from_cpu_bytes(
                                v,
                                &Shape::new(vec![v.len()]),
                                pack_dtype(),
                            )?))
                        };
                    let cg_b = build(&cg_v)?;
                    let cu_b = build(&cu_v)?;
                    let cd_b = build(&cd_v)?;
                    let eg_b = build(&eg_v)?;
                    let eu_b = build(&eu_v)?;
                    let ed_b = build(&ed_v)?;
                    *guard = Some(Mxfp4Resident {
                        codes_gate: Arc::clone(&cg_b),
                        codes_up: Arc::clone(&cu_b),
                        codes_down: Arc::clone(&cd_b),
                        exps_gate: Arc::clone(&eg_b),
                        exps_up: Arc::clone(&eu_b),
                        exps_down: Arc::clone(&ed_b),
                        fingerprint: key,
                    });
                    (cg_b, cu_b, cd_b, eg_b, eu_b, ed_b)
                }
            }
        };

        let ascale_buf = upload_ones_ascale(&rocm, seq_len)?;
        let ascale_rocm = as_rocm_storage(&ascale_buf, "ascale")?;

        cache.record_dispatch(DispatchKind::Mxfp4Native);
        if num_pairs == 0 {
            return zero_moe_output(dev, x, shared_expert).map(Some);
        }

        let out_shape = Shape::new(vec![seq_len, hidden]);
        let (out_storage, _handle) = rocm.moe_fused_dispatch_resident_routing_mxfp4(
            x_rocm,
            &*cg,
            &*cu,
            &*cd,
            &*eg,
            &*eu,
            &*ed,
            ascale_rocm,
            tokens_rocm,
            experts_rocm,
            weights_rocm,
            num_pairs,
            &out_shape,
            hidden,
            inter,
            routed_scaling_factor,
        )?;

        let out_t = Tensor::new(
            Arc::from(out_storage),
            out_shape,
            DType::F32,
            x.provenance().clone(),
            x.device().clone(),
        );

        return shared_expert_tail(dev, out_t, x, shared_expert).map(Some);
    }

    // Resolve the resident weight stack (built once).
    let (gate_buf, up_buf, down_buf) = {
        let mut guard = cache.resident.lock().unwrap_or_else(|e| e.into_inner());
        let key = (num_experts, hidden, inter, TAG_F32);
        match guard.as_ref() {
            Some(r) if r.fingerprint == key => {
                (Arc::clone(&r.gate), Arc::clone(&r.up), Arc::clone(&r.down))
            }
            _ => {
                let (gate_flat, up_flat, down_flat) =
                    match stack_expert_weights(experts, num_experts, hidden, inter) {
                        Ok(v) => v,
                        Err(e) => {
                            // Cannot materialize the f32 expert stacks (e.g. Xing4.0-29B
                            // needs 2.8 GiB on a 17.1 GB card). Degrade to the
                            // per-expert reference path instead of faulting inside
                            // the kernel.
                            if std::env::var_os("GRIM_MOE_TRACE").is_some() {
                                eprintln!("[moe] grouped dispatch declined: {e}");
                            }
                            return Ok(None);
                        }
                    };
                let gate = Arc::from(rocm.from_cpu(
                    &gate_flat,
                    &Shape::new(vec![gate_flat.len()]),
                    DType::F32,
                )?);
                let up = Arc::from(rocm.from_cpu(
                    &up_flat,
                    &Shape::new(vec![up_flat.len()]),
                    DType::F32,
                )?);
                let down = Arc::from(rocm.from_cpu(
                    &down_flat,
                    &Shape::new(vec![down_flat.len()]),
                    DType::F32,
                )?);
                *guard = Some(ResidentWeights {
                    gate: Arc::clone(&gate),
                    up: Arc::clone(&up),
                    down: Arc::clone(&down),
                    fingerprint: key,
                });
                (gate, up, down)
            }
        }
    };

    // 2. Sortless fused dispatch from device-resident routing (no H2D).
    if num_pairs == 0 {
        cache.record_dispatch(DispatchKind::F32Dequant);
        let out = dev.zeros(x.shape(), DType::F32)?;
        let out_t = Tensor::new(
            Arc::from(out),
            x.shape().clone(),
            DType::F32,
            x.provenance().clone(),
            x.device().clone(),
        );
        return shared_expert_tail(dev, out_t, x, shared_expert).map(Some);
    }

    let out_shape = Shape::new(vec![seq_len, hidden]);
    let (out_storage, _handle) = rocm.moe_fused_dispatch_resident_routing(
        x_rocm,
        &*gate_buf,
        &*up_buf,
        &*down_buf,
        tokens_rocm,
        experts_rocm,
        weights_rocm,
        num_pairs,
        &out_shape,
        hidden,
        inter,
        routed_scaling_factor,
    )?;
    cache.record_dispatch(DispatchKind::F32Dequant);

    let out_t = Tensor::new(
        Arc::from(out_storage),
        out_shape,
        DType::F32,
        x.provenance().clone(),
        x.device().clone(),
    );

    shared_expert_tail(dev, out_t, x, shared_expert).map(Some)
}

/// Phase 3a: Charon grouped fused-dispatch path.
///
/// Stacks `experts[*].{gate,up,down}` weights into three contiguous f32 device
/// buffers (`[num_experts, inter*hidden]` gate/up, `[num_experts, hidden*inter]`
/// down), materializing each weight to row-major f32 (Native weights ride the
/// existing dequant path; KQuant/GroupInt/etc. dequantize to f32 so the scalar
/// Charon kernel consumes raw floats). The stack is built **once** into `cache`
/// and reused, keeping the decode path round-trip-free.
///
/// Returns `Ok(None)` when the path cannot run (non-ROCm input, no device
/// pointer, no experts/routing) so the caller falls back to the per-expert loop.
fn charon_grouped_dispatch(
    dev: &dyn BackendDevice,
    x: &Tensor,
    experts: &[MoeExpert],
    shared_expert: Option<&MoeExpert>,
    routings: &[TokenRouting],
    routed_scaling_factor: f32,
    cache: &CharonCache,
) -> Result<Option<Tensor>> {
    // Only the scalar Charon kernel is safe to route here; it consumes f32
    // weights, so we need shapes independent of the weight quant format.
    let dims = x.shape().dims();
    if dims.len() != 2 {
        return Ok(None);
    }
    let (seq_len, hidden) = (dims[0], dims[1]);
    if seq_len == 0 || hidden == 0 || experts.is_empty() {
        return Ok(None);
    }

    let ordinal = match x.device() {
        Device::Rocm(o) => *o,
        _ => return Ok(None),
    };

    // Inter dimension = the gate weight's output rows (dim 0 of the Linear).
    let inter = experts[0].gate.weight.shape().dim(0).unwrap_or(0);
    if inter == 0 {
        return Ok(None);
    }

    // Down projection restores `hidden`: down.weight has rows == hidden.
    let down_rows = experts[0].down.weight.shape().dim(0).unwrap_or(0);
    if down_rows != hidden {
        return Ok(None);
    }
    let num_experts = experts.len();

    // Build the sortless RoutingAssignment from the per-token routings, then
    // token-sort it into the grouped layout the kernel consumes.
    let indices: Vec<Vec<usize>> = routings
        .iter()
        .map(|r| r.iter().map(|(e, _)| *e).collect())
        .collect();
    let weights: Vec<Vec<f32>> = routings
        .iter()
        .map(|r| r.iter().map(|(_, w)| *w).collect())
        .collect();
    let assignment =
        grim_backend_rocm::kernels::charon::RoutingAssignment::from_route(&indices, &weights)?;
    if assignment.num_pairs() == 0 {
        // No routed pairs: output belongs to the shared expert alone (or zeros).
        let out = dev.zeros(x.shape(), DType::F32)?;
        let out_t = Tensor::new(
            Arc::from(out),
            x.shape().clone(),
            DType::F32,
            x.provenance().clone(),
            x.device().clone(),
        );
        return shared_expert_tail(dev, out_t, x, shared_expert).map(Some);
    }

    let sorted = grim_backend_rocm::kernels::charon::moe_align_block_size(
        &assignment,
        64.max(rocm_wavefront_size(ordinal)),
        num_experts,
    );

    // Resolve (and lazily build) the resident weight stack. Built once; the
    // fingerprint prevents rebuilds across decode steps.
    let rocm = Arc::new(grim_backend_rocm::RocmDevice::shared(ordinal));
    let (gate_buf, up_buf, down_buf) = {
        let mut guard = cache.resident.lock().unwrap_or_else(|e| e.into_inner());
        let key = (num_experts, hidden, inter, TAG_F32);
        match guard.as_ref() {
            Some(r) if r.fingerprint == key => {
                (Arc::clone(&r.gate), Arc::clone(&r.up), Arc::clone(&r.down))
            }
            _ => {
                let (gate_flat, up_flat, down_flat) =
                    match stack_expert_weights(experts, num_experts, hidden, inter) {
                        Ok(v) => v,
                        Err(e) => {
                            // Cannot materialize the f32 expert stacks (e.g. Xing4.0-29B
                            // needs 2.8 GiB on a 17.1 GB card). Degrade to the
                            // per-expert reference path instead of faulting inside
                            // the kernel.
                            if std::env::var_os("GRIM_MOE_TRACE").is_some() {
                                eprintln!("[moe] grouped dispatch declined: {e}");
                            }
                            return Ok(None);
                        }
                    };
                let gate = Arc::from(rocm.from_cpu(
                    &gate_flat,
                    &Shape::new(vec![gate_flat.len()]),
                    DType::F32,
                )?);
                let up = Arc::from(rocm.from_cpu(
                    &up_flat,
                    &Shape::new(vec![up_flat.len()]),
                    DType::F32,
                )?);
                let down = Arc::from(rocm.from_cpu(
                    &down_flat,
                    &Shape::new(vec![down_flat.len()]),
                    DType::F32,
                )?);
                *guard = Some(ResidentWeights {
                    gate: Arc::clone(&gate),
                    up: Arc::clone(&up),
                    down: Arc::clone(&down),
                    fingerprint: key,
                });
                (gate, up, down)
            }
        }
    };

    let x_rocm = x
        .storage()
        .as_ref()
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .ok_or_else(|| grim_tensor::Error::Backend("x is not RocmStorage".into()))?;

    let out_shape = Shape::new(vec![seq_len, hidden]);
    let (out_storage, _handle) = rocm.moe_fused_grouped_dispatch_resident(
        x_rocm,
        &*gate_buf,
        &*up_buf,
        &*down_buf,
        &sorted,
        &out_shape,
        hidden,
        inter,
        num_experts,
        routed_scaling_factor,
    )?;

    let out_t = Tensor::new(
        Arc::from(out_storage),
        out_shape,
        DType::F32,
        x.provenance().clone(),
        x.device().clone(),
    );

    shared_expert_tail(dev, out_t, x, shared_expert).map(Some)
}

/// Charon grouped dispatch for GELU-activation MoE experts (GLM-5.2 style).
///
/// GLM-5.2 experts have a single projection (`dense_h_to_4h`, stored as `gate`)
/// followed by GELU and a down projection (`dense_4h_to_h`, stored as `down`).
/// There is no `up` projection. The GELU kernel (`grim_moe_fused_grouped_gelu`)
/// accepts `up_ptr = 0` and is pre-wired in `moe_fused_grouped_dispatch_gelu_resident`.
///
/// `gate_weights` — `[num_experts]` refs to expert projection weight tensors.
/// `down_weights` — `[num_experts]` refs to expert down projection weight tensors.
/// `routings`     — per-token (expert_idx, weight) routing result from the CPU gate.
///
/// Returns `Ok(None)` when the GELU path cannot run (non-ROCm, etc.) so the
/// caller falls back to the per-expert CPU loop.
pub fn gelu_charon_dispatch(
    dev: &dyn BackendDevice,
    x: &Tensor,
    gate_weights: &[&grim_tensor::Tensor],
    down_weights: &[&grim_tensor::Tensor],
    routings: &[TokenRouting],
    routed_scaling_factor: f32,
    cache: &CharonCache,
) -> Result<Option<Tensor>> {
    if !charon_enabled() {
        return Ok(None);
    }
    let dims = x.shape().dims();
    if dims.len() != 2 {
        return Ok(None);
    }
    let (seq_len, hidden) = (dims[0], dims[1]);
    let num_experts = gate_weights.len();
    if seq_len == 0 || hidden == 0 || num_experts == 0 || num_experts != down_weights.len() {
        return Ok(None);
    }
    let ordinal = match x.device() {
        Device::Rocm(o) => *o,
        _ => return Ok(None),
    };

    // gate weight dim(0) == inter (intermediate), dim(1) == hidden.
    let inter = gate_weights[0].shape().dim(0).unwrap_or(0);
    if inter == 0 {
        return Ok(None);
    }

    // Build sortless routing assignment from host-computed routings.
    let indices: Vec<Vec<usize>> = routings
        .iter()
        .map(|r| r.iter().map(|(e, _)| *e).collect())
        .collect();
    let weights: Vec<Vec<f32>> = routings
        .iter()
        .map(|r| r.iter().map(|(_, w)| *w).collect())
        .collect();
    let assignment =
        grim_backend_rocm::kernels::charon::RoutingAssignment::from_route(&indices, &weights)?;
    if assignment.num_pairs() == 0 {
        let out = dev.zeros(x.shape(), DType::F32)?;
        let out_t = Tensor::new(
            Arc::from(out),
            x.shape().clone(),
            DType::F32,
            x.provenance().clone(),
            x.device().clone(),
        );
        return Ok(Some(out_t));
    }
    let sorted = grim_backend_rocm::kernels::charon::moe_align_block_size(
        &assignment,
        64.max(rocm_wavefront_size(ordinal)),
        num_experts,
    );

    let rocm = Arc::new(grim_backend_rocm::RocmDevice::shared(ordinal));

    // Lazily build + cache gate and down resident stacks (no up).
    let (gate_buf, down_buf) = {
        let mut guard = cache.resident.lock().unwrap_or_else(|e| e.into_inner());
        let key = (num_experts, hidden, inter, TAG_GELU);
        match guard.as_ref() {
            Some(r) if r.fingerprint == key => (Arc::clone(&r.gate), Arc::clone(&r.down)),
            _ => {
                let mut gate_flat = Vec::with_capacity(num_experts * inter * hidden);
                let mut down_flat = Vec::with_capacity(num_experts * hidden * inter);
                for (gw, dw) in gate_weights.iter().zip(down_weights.iter()) {
                    let gv = match gw.device() {
                        Device::Rocm(ord) => grim_nn::moe::rocm_dequant_expert_weight(gw, *ord)
                            .map_err(grim_core::error::Error::Tensor)?,
                        _ => gw.to_vec_f32()?,
                    };
                    gate_flat.extend_from_slice(&gv);
                    let dv = match dw.device() {
                        Device::Rocm(ord) => grim_nn::moe::rocm_dequant_expert_weight(dw, *ord)
                            .map_err(grim_core::error::Error::Tensor)?,
                        _ => dw.to_vec_f32()?,
                    };
                    down_flat.extend_from_slice(&dv);
                }
                let gate = Arc::from(rocm.from_cpu(
                    &gate_flat,
                    &Shape::new(vec![gate_flat.len()]),
                    DType::F32,
                )?);
                // Dummy zero up buffer (kernel ignores it via up_ptr=0, but
                // ResidentWeights.up slot must be filled).
                let up_dummy = Arc::from(rocm.zeros(&Shape::new(vec![1]), DType::F32)?);
                let down = Arc::from(rocm.from_cpu(
                    &down_flat,
                    &Shape::new(vec![down_flat.len()]),
                    DType::F32,
                )?);
                *guard = Some(ResidentWeights {
                    gate: Arc::clone(&gate),
                    up: Arc::clone(&up_dummy),
                    down: Arc::clone(&down),
                    fingerprint: key,
                });
                (gate, down)
            }
        }
    };

    let x_rocm = x
        .storage()
        .as_ref()
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .ok_or_else(|| grim_tensor::Error::Backend("x is not RocmStorage".into()))?;

    let out_shape = Shape::new(vec![seq_len, hidden]);
    let (out_storage, _handle) = rocm.moe_fused_grouped_dispatch_gelu_resident(
        x_rocm,
        &*gate_buf,
        &*down_buf,
        &sorted,
        &out_shape,
        hidden,
        inter,
        num_experts,
        routed_scaling_factor,
    )?;

    let out_t = Tensor::new(
        Arc::from(out_storage),
        out_shape,
        DType::F32,
        x.provenance().clone(),
        x.device().clone(),
    );
    Ok(Some(out_t))
}

/// Wavefront size of the shared ROCm device (32 for RDNA, 64 for CDNA), with a
/// fallback for the non-ROCm path (never reached here but keeps the call safe).
fn rocm_wavefront_size(ordinal: usize) -> usize {
    grim_backend_rocm::RocmDevice::shared(ordinal).wavefront_size() as usize
}

/// W8A8-int8 packed-blob strides, in bytes: `[u64 prefix | int8 codes |
/// f32 per-row scales]`. MUST match `grim_moe_fused_dispatch_w8a8_int8`
/// (`gate_stride` / `down_stride`).
fn w8a8_strides(hidden: usize, inter: usize) -> (usize, usize) {
    (
        8 + inter * hidden + inter * 4,
        8 + hidden * inter + hidden * 4,
    )
}

/// Native-arm predicate (WI-gpu-native-moe Phase 2): true only when every
/// projection of every expert is a ROCm-resident CompressedTensorsW8A8Int8
/// packed blob. Mixed/quantized-other formats ride the f32 dequant arm.
fn experts_use_w8a8_native(experts: &[MoeExpert]) -> bool {
    experts.iter().all(|e| {
        [&e.gate, &e.up, &e.down].iter().all(|l| {
            matches!(l.weight.device(), Device::Rocm(_))
                && matches!(
                    l.weight.dtype().storage,
                    grim_tensor::Storage::CompressedTensorsW8A8Int8
                )
        })
    })
}

/// Concatenate per-expert W8A8-int8 packed blobs into three contiguous
/// device-upload-ready stacks (gate/up share `gate_stride`, down uses
/// `down_stride`). Each blob is validated byte-exact: a short/long blob is
/// a loader bug and must fail loudly, never silently misalign the kernel's
/// per-expert pointer arithmetic.
fn stack_w8a8_blobs(
    experts: &[MoeExpert],
    num_experts: usize,
    hidden: usize,
    inter: usize,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let (gate_stride, down_stride) = w8a8_strides(hidden, inter);
    let mut gate_flat = Vec::with_capacity(num_experts * gate_stride);
    let mut up_flat = Vec::with_capacity(num_experts * gate_stride);
    let mut down_flat = Vec::with_capacity(num_experts * down_stride);
    for e in experts.iter().take(num_experts) {
        for (dst, lin, stride) in [
            (&mut gate_flat, &e.gate, gate_stride),
            (&mut up_flat, &e.up, gate_stride),
            (&mut down_flat, &e.down, down_stride),
        ] {
            let storage = lin
                .weight
                .storage()
                .as_ref()
                .as_any()
                .downcast_ref::<grim_backend_rocm::RocmStorage>()
                .ok_or_else(|| {
                    grim_tensor::Error::Backend("stack_w8a8_blobs: expert not RocmStorage".into())
                })?;
            let bytes = storage.copy_to_host()?;
            if bytes.len() != stride {
                return Err(grim_core::error::Error::Shape(format!(
                    "stack_w8a8_blobs: expert blob len {} != stride {} (hidden={hidden} inter={inter})",
                    bytes.len(),
                    stride,
                )));
            }
            dst.extend_from_slice(&bytes);
        }
    }
    Ok((gate_flat, up_flat, down_flat))
}

/// W8A8-fp8 packed-blob strides, in bytes: `[u64 prefix | fp8 codes |
/// ONE f32 scale]`. MUST match `grim_moe_fused_dispatch_w8a8_fp8`.
fn w8a8_fp8_strides(hidden: usize, inter: usize) -> (usize, usize) {
    (8 + inter * hidden + 4, 8 + hidden * inter + 4)
}

/// Native-arm predicate for W8A8-fp8 (mirrors [`experts_use_w8a8_native`]).
fn experts_use_w8a8_fp8_native(experts: &[MoeExpert]) -> bool {
    experts.iter().all(|e| {
        [&e.gate, &e.up, &e.down].iter().all(|l| {
            matches!(l.weight.device(), Device::Rocm(_))
                && matches!(
                    l.weight.dtype().storage,
                    grim_tensor::Storage::CompressedTensorsW8A8Fp8
                )
        })
    })
}

/// Concatenate per-expert W8A8-fp8 packed blobs (byte-exact validation,
/// same discipline as [`stack_w8a8_blobs`]).
fn stack_w8a8_fp8_blobs(
    experts: &[MoeExpert],
    num_experts: usize,
    hidden: usize,
    inter: usize,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let (gate_stride, down_stride) = w8a8_fp8_strides(hidden, inter);
    let mut gate_flat = Vec::with_capacity(num_experts * gate_stride);
    let mut up_flat = Vec::with_capacity(num_experts * gate_stride);
    let mut down_flat = Vec::with_capacity(num_experts * down_stride);
    for e in experts.iter().take(num_experts) {
        for (dst, lin, stride) in [
            (&mut gate_flat, &e.gate, gate_stride),
            (&mut up_flat, &e.up, gate_stride),
            (&mut down_flat, &e.down, down_stride),
        ] {
            let storage = lin
                .weight
                .storage()
                .as_ref()
                .as_any()
                .downcast_ref::<grim_backend_rocm::RocmStorage>()
                .ok_or_else(|| {
                    grim_tensor::Error::Backend(
                        "stack_w8a8_fp8_blobs: expert not RocmStorage".into(),
                    )
                })?;
            let bytes = storage.copy_to_host()?;
            if bytes.len() != stride {
                return Err(grim_core::error::Error::Shape(format!(
                    "stack_w8a8_fp8_blobs: expert blob len {} != stride {} (hidden={hidden} inter={inter})",
                    bytes.len(),
                    stride,
                )));
            }
            dst.extend_from_slice(&bytes);
        }
    }
    Ok((gate_flat, up_flat, down_flat))
}

/// AWQ values-per-u32-word for a bit width. MUST match `awq_split_bank`
/// and `grim_moe_fused_dispatch_awq` (`vpw`).
fn awq_vpw(bits: u8) -> Option<usize> {
    match bits {
        4 => Some(8),
        2 => Some(16),
        8 => Some(1),
        _ => None,
    }
}

/// AWQ per-expert blob stride for a [rows=out, cols=k] projection.
/// MUST match `awq_split_bank`'s per-expert layout AND the launcher's
/// segment math: `[u64 qw_len | qw | u64 qz_len | qzeros | u64 sc_len |
/// f16 scales]`.
fn awq_blob_stride(out: usize, k: usize, bits: u8, group_size: usize) -> Option<usize> {
    let vpw = awq_vpw(bits)?;
    if group_size == 0 {
        return None;
    }
    let qw_len = k.div_ceil(vpw) * out * 4;
    let groups = k.div_ceil(group_size);
    let qz_len = groups * out.div_ceil(vpw) * 4;
    let sc_len = groups * out * 2;
    Some(8 + qw_len + 8 + qz_len + 8 + sc_len)
}

/// Native-arm config for AWQ: `Some((bits, group_size))` only when EVERY
/// projection of EVERY expert is ROCm-resident AWQ with IDENTICAL
/// bits/group. Heterogeneous banks ride the f32 dequant arm (which handles
/// per-tensor configs); silently picking one config would misdecode the rest.
fn awq_uniform_config(experts: &[MoeExpert]) -> Option<(u8, usize)> {
    let mut cfg: Option<(u8, usize)> = None;
    for e in experts {
        for l in [&e.gate, &e.up, &e.down] {
            if !matches!(l.weight.device(), Device::Rocm(_)) {
                return None;
            }
            let this = match l.weight.dtype().storage {
                grim_tensor::Storage::Awq(c) => (c.bits, c.group_size),
                _ => return None,
            };
            match cfg {
                None => cfg = Some(this),
                Some(c) if c == this => {}
                _ => return None,
            }
        }
    }
    let (bits, group) = cfg?;
    awq_vpw(bits)?;
    if group == 0 {
        return None;
    }
    Some((bits, group))
}

/// Concatenate per-expert AWQ packed blobs (byte-exact validation against
/// [`awq_blob_stride`]).
fn stack_awq_blobs(
    experts: &[MoeExpert],
    num_experts: usize,
    hidden: usize,
    inter: usize,
    bits: u8,
    group_size: usize,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let gate_stride = awq_blob_stride(inter, hidden, bits, group_size).ok_or_else(|| {
        grim_core::error::Error::Backend("stack_awq_blobs: bad gate layout".into())
    })?;
    let down_stride = awq_blob_stride(hidden, inter, bits, group_size).ok_or_else(|| {
        grim_core::error::Error::Backend("stack_awq_blobs: bad down layout".into())
    })?;
    let mut gate_flat = Vec::with_capacity(num_experts * gate_stride);
    let mut up_flat = Vec::with_capacity(num_experts * gate_stride);
    let mut down_flat = Vec::with_capacity(num_experts * down_stride);
    for e in experts.iter().take(num_experts) {
        for (dst, lin, stride) in [
            (&mut gate_flat, &e.gate, gate_stride),
            (&mut up_flat, &e.up, gate_stride),
            (&mut down_flat, &e.down, down_stride),
        ] {
            let storage = lin
                .weight
                .storage()
                .as_ref()
                .as_any()
                .downcast_ref::<grim_backend_rocm::RocmStorage>()
                .ok_or_else(|| {
                    grim_tensor::Error::Backend("stack_awq_blobs: expert not RocmStorage".into())
                })?;
            let bytes = storage.copy_to_host()?;
            if bytes.len() != stride {
                return Err(grim_core::error::Error::Shape(format!(
                    "stack_awq_blobs: expert blob len {} != stride {} (hidden={hidden} inter={inter} bits={bits} group={group_size})",
                    bytes.len(),
                    stride,
                )));
            }
            dst.extend_from_slice(&bytes);
        }
    }
    Ok((gate_flat, up_flat, down_flat))
}

/// Native-arm predicate for MXFP4: every projection ROCm-resident
/// FloatPack(MxFp4) AND `(hidden*inter) % 32 == 0` (whole 32-groups —
/// the kernel cannot address partial groups; misaligned shapes stay out
/// rather than read OOB).
fn experts_use_mxfp4_native(experts: &[MoeExpert], hidden: usize, inter: usize) -> bool {
    (hidden * inter) % 32 == 0
        && experts.iter().all(|e| {
            [&e.gate, &e.up, &e.down].iter().all(|l| {
                matches!(l.weight.device(), Device::Rocm(_))
                    && matches!(
                        l.weight.dtype().storage,
                        grim_tensor::Storage::FloatPack(grim_tensor::FloatPackScheme::MxFp4)
                    )
            })
        })
}

/// Split one framed MXFP4 per-expert blob (`[u64 clen | codes | u64 xlen |
/// exps]`, as produced by the bank loader) into its `(codes, exps)` parts
/// with exact length validation.
fn split_mxfp4_blob(
    bytes: &[u8],
    rows: usize,
    k: usize,
    label: &str,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let need_codes = rows * k / 2;
    let need_exps = (rows * k).div_ceil(32);
    if bytes.len() < 8 {
        return Err(grim_core::error::Error::Shape(format!(
            "stack_mxfp4: {label} blob truncated (len={})",
            bytes.len()
        )));
    }
    let clen = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    if clen != need_codes || bytes.len() < 8 + clen + 8 {
        return Err(grim_core::error::Error::Shape(format!(
            "stack_mxfp4: {label} codes len {clen} != {need_codes}"
        )));
    }
    let xlen_off = 8 + clen;
    let xlen = u64::from_le_bytes(bytes[xlen_off..xlen_off + 8].try_into().unwrap()) as usize;
    if xlen != need_exps || bytes.len() < 8 + clen + 8 + xlen {
        return Err(grim_core::error::Error::Shape(format!(
            "stack_mxfp4: {label} exps len {xlen} != {need_exps}"
        )));
    }
    Ok((
        bytes[8..8 + clen].to_vec(),
        bytes[xlen_off + 8..xlen_off + 8 + xlen].to_vec(),
    ))
}

/// Build MXFP4 code + shared-exponent stacks (gate/up share shapes, down
/// differs). Returns `(codes_gate, codes_up, codes_down, exps_gate,
/// exps_up, exps_down)`.
type Mxfp4Stacks = (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>);
fn stack_mxfp4(
    experts: &[MoeExpert],
    num_experts: usize,
    hidden: usize,
    inter: usize,
) -> Result<Mxfp4Stacks> {
    let codes_per_gate = inter * hidden / 2;
    let exps_per_gate = (inter * hidden).div_ceil(32);
    let codes_per_down = hidden * inter / 2;
    let exps_per_down = (hidden * inter).div_ceil(32);
    let mut cg = Vec::with_capacity(num_experts * codes_per_gate);
    let mut cu = Vec::with_capacity(num_experts * codes_per_gate);
    let mut cd = Vec::with_capacity(num_experts * codes_per_down);
    let mut eg = Vec::with_capacity(num_experts * exps_per_gate);
    let mut eu = Vec::with_capacity(num_experts * exps_per_gate);
    let mut ed = Vec::with_capacity(num_experts * exps_per_down);
    for e in experts.iter().take(num_experts) {
        for (codes_dst, exps_dst, lin, rows, k, label) in [
            (&mut cg, &mut eg, &e.gate, inter, hidden, "gate"),
            (&mut cu, &mut eu, &e.up, inter, hidden, "up"),
            (&mut cd, &mut ed, &e.down, hidden, inter, "down"),
        ] {
            let storage = lin
                .weight
                .storage()
                .as_ref()
                .as_any()
                .downcast_ref::<grim_backend_rocm::RocmStorage>()
                .ok_or_else(|| {
                    grim_tensor::Error::Backend("stack_mxfp4: expert not RocmStorage".into())
                })?;
            let bytes = storage.copy_to_host()?;
            let (codes, exps) = split_mxfp4_blob(&bytes, rows, k, label)?;
            codes_dst.extend_from_slice(&codes);
            exps_dst.extend_from_slice(&exps);
        }
    }
    Ok((cg, cu, cd, eg, eu, ed))
}
fn stack_expert_weights(
    experts: &[MoeExpert],
    num_experts: usize,
    hidden: usize,
    inter: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    // The f32 stacks are 3 x num_experts x inter x hidden x 4 B: 2.8 GiB for
    // Xing4.0-29B (64 experts, 1024 x 3584) on top of a 12.9 GiB packed model.
    // On a 17.1 GB card that overcommits and the dispatch dies in the kernel
    // with "Page not present" instead of falling back. Refuse here so the
    // caller takes the per-expert reference path, which is correct (slow).
    let need = (3 * num_experts * inter * hidden * 4) as u64;
    if let Some(ord) = experts.first().map(|e| match e.gate.weight.device() {
        Device::Rocm(o) => Some(*o),
        _ => None,
    }) {
        if let Some(o) = ord {
            let (free, _total) = grim_backend_rocm::vram_info(o);
            let headroom = (free as f64) * 0.85;
            if need as f64 > headroom {
                return Err(grim_core::error::Error::Backend(format!(
                    "expert f32 stacks need {:.2} GiB, only {:.2} GiB free on ordinal {o}; \
                     falling back to the per-expert reference path",
                    need as f64 / (1u64 << 30) as f64,
                    free as f64 / (1u64 << 30) as f64
                )));
            }
        }
    }
    let mut gate_flat = Vec::with_capacity(num_experts * inter * hidden);
    let mut up_flat = Vec::with_capacity(num_experts * inter * hidden);
    let mut down_flat = Vec::with_capacity(num_experts * hidden * inter);
    for e in experts.iter().take(num_experts) {
        for (dst, lin) in [
            (&mut gate_flat, &e.gate),
            (&mut up_flat, &e.up),
            (&mut down_flat, &e.down),
        ] {
            let w = match lin.weight.device() {
                Device::Rocm(ord) => grim_nn::moe::rocm_dequant_expert_weight(&lin.weight, *ord)
                    .map_err(grim_core::error::Error::Tensor)?,
                _ => lin.weight.to_vec_f32()?,
            };
            if w.len() != lin.weight.shape().elem_count() {
                return Err(grim_core::error::Error::Shape(format!(
                    "stack_expert_weights: expert weight len {} != elem_count {}",
                    w.len(),
                    lin.weight.shape().elem_count()
                )));
            }
            dst.extend_from_slice(&w);
        }
    }
    Ok((gate_flat, up_flat, down_flat))
}

/// Apply the shared (always-on) expert on top of the routed output.
fn shared_expert_tail(
    _dev: &dyn BackendDevice,
    out: Tensor,
    x: &Tensor,
    shared_expert: Option<&MoeExpert>,
) -> Result<Tensor> {
    match shared_expert {
        Some(shared) => {
            let sh_out = expert_forward(shared, x)?;
            grim_nn::modules::add_on_device(&out, &sh_out).map_err(grim_core::error::Error::from)
        }
        None => Ok(out),
    }
}

fn per_expert_loop(
    dev: &dyn BackendDevice,
    x: &Tensor,
    experts: &[MoeExpert],
    shared_expert: Option<&MoeExpert>,
    routings: &[TokenRouting],
    routed_scaling_factor: f32,
) -> Result<Tensor> {
    let dims = x.shape().dims();
    let seq_len = dims[0];
    let hidden_dim = dims[1];

    let out_st = dev.zeros(x.shape(), DType::F32)?;

    for (s, routing) in routings.iter().enumerate() {
        if routing.is_empty() {
            continue;
        }
        let token_x = if seq_len == 1 {
            x.clone()
        } else {
            let tok_shape = Shape::new(vec![1, hidden_dim]);
            let token_st = dev.alloc_storage(&tok_shape, DType::F32)?;
            dev.copy_slice_range(
                token_st.as_ref(),
                0,
                x.storage().as_ref(),
                s * hidden_dim,
                hidden_dim,
            )?;
            Tensor::new(
                Arc::from(token_st),
                tok_shape,
                DType::F32,
                x.provenance().clone(),
                x.device().clone(),
            )
        };

        let mut acc: Option<Tensor> = None;
        for (exp_idx, weight) in routing {
            let expert = &experts[*exp_idx];
            let exp_out = expert_forward(expert, &token_x)?;
            let w = *weight * routed_scaling_factor;
            let (scaled_st, _handle) =
                dev.mul_scalar(exp_out.storage().as_ref(), w, exp_out.shape())?;
            let scaled = Tensor::new(
                Arc::from(scaled_st),
                exp_out.shape().clone(),
                DType::F32,
                exp_out.provenance().clone(),
                exp_out.device().clone(),
            );
            acc = Some(match acc {
                Some(a) => grim_nn::modules::add_on_device(&a, &scaled)?,
                None => scaled,
            });
        }
        if let Some(acc) = acc {
            dev.copy_slice_into(
                out_st.as_ref(),
                acc.storage().as_ref(),
                s * hidden_dim,
                hidden_dim,
            )?;
        }
    }

    let out_t = Tensor::new(
        Arc::from(out_st),
        x.shape().clone(),
        DType::F32,
        x.provenance().clone(),
        x.device().clone(),
    );

    shared_expert_tail(dev, out_t, x, shared_expert)
}

/// Compute per-token top-k routing from flat gate logits. `logits_v` has layout
/// `[seq_len, num_experts]`. Returns one `TokenRouting` per token (sorted by raw
/// logit, descending) with softmax-normalized combine weights (no architecture
/// scaling applied — multiply by `routed_scaling_factor` at dispatch time).
pub fn route_topk(logits_v: &[f32], num_experts: usize, top_k: usize) -> Result<Vec<TokenRouting>> {
    let seq_len = logits_v.len().checked_div(num_experts).unwrap_or(0);
    let mut out = Vec::with_capacity(seq_len);
    for s in 0..seq_len {
        let row = &logits_v[s * num_experts..(s + 1) * num_experts];
        let mut indexed: Vec<(usize, f32)> = row.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let k = top_k.min(num_experts);
        let topk = &indexed[..k];
        let mut entry: Vec<(usize, f32)> = topk.iter().map(|(idx, _)| (*idx, 0.0)).collect();
        let weights = normalize_weights(topk);
        for (j, (idx, _)) in topk.iter().enumerate() {
            entry[j] = (*idx, weights[j]);
        }
        out.push(entry);
    }
    Ok(out)
}

/// Softmax-normalize the top-k combine weights.
pub fn normalize_weights(topk: &[(usize, f32)]) -> Vec<f32> {
    if topk.is_empty() {
        return Vec::new();
    }
    let max_l = topk
        .iter()
        .map(|(_, l)| *l)
        .fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = topk.iter().map(|(_, l)| (l - max_l).exp()).collect();
    let sum_e: f32 = exps.iter().sum();
    exps.iter().map(|e| e / (sum_e + 1e-12)).collect()
}

fn expert_forward(expert: &MoeExpert, x: &Tensor) -> Result<Tensor> {
    let lift = |r: std::result::Result<Tensor, grim_tensor::Error>| {
        r.map_err(grim_core::error::Error::from)
    };
    let gate = lift(expert.gate.forward(x))?;
    let up = lift(expert.up.forward(x))?;
    let swiglu = silu_mul_on_device(&gate, &up).map_err(grim_core::error::Error::from)?;
    lift(expert.down.forward(&swiglu))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_route_topk_ranking_and_normalization() {
        let logits = vec![1.0, 5.0, 2.0, 4.0];
        let routings = route_topk(&logits, 4, 2).expect("route_topk succeeds");
        assert_eq!(routings.len(), 1);
        let topk = &routings[0];
        assert_eq!(topk.len(), 2);
        assert_eq!(topk[0].0, 1); // expert 1 has logit 5.0
        assert_eq!(topk[1].0, 3); // expert 3 has logit 4.0
        let sum_w: f32 = topk.iter().map(|(_, w)| *w).sum();
        assert!((sum_w - 1.0).abs() < 1e-4);
    }

    /// Unit: `normalize_weights` matches the route_mode-3 device kernel
    /// semantics (softmax renormalized over top-k only). Guards the
    /// Phase-1 renorm rollout's numeric contract on CPU (no GPU needed).
    #[test]
    fn test_normalize_weights_renorm_contract() {
        // Top-k subset (logits 5.0, 4.0) renormalized: exp(0)/(exp(0)+exp(-1)).
        let topk = vec![(1usize, 5.0f32), (3usize, 4.0f32)];
        let ws = normalize_weights(&topk);
        assert_eq!(ws.len(), 2);
        let e0 = 1.0f32;
        let e1 = (-1.0f32).exp();
        let sum = e0 + e1;
        assert!((ws[0] - e0 / (sum + 1e-12)).abs() < 1e-6);
        assert!((ws[1] - e1 / (sum + 1e-12)).abs() < 1e-6);
        let total: f32 = ws.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "renorm weights must sum to 1");
    }

    /// Unit: `route_topk` output equals `normalize_weights` applied to the
    /// raw-logit top-k (i.e. host path == route_mode 3, not global softmax).
    /// Multi-token + edge cases (top_k == num_experts, top_k == 1).
    #[test]
    fn test_route_topk_matches_normalize_weights_multi_token() {
        let logits = vec![
            1.0, 5.0, 2.0, 4.0, // token 0
            -3.0, 0.5, 2.5, 2.0, // token 1
        ];
        for top_k in [1usize, 2, 4] {
            let routings = route_topk(&logits, 4, top_k).expect("route_topk succeeds");
            assert_eq!(routings.len(), 2);
            for (s, routing) in routings.iter().enumerate() {
                assert_eq!(routing.len(), top_k.min(4));
                // Re-derive expected top-k from raw logits.
                let row = &logits[s * 4..(s + 1) * 4];
                let mut indexed: Vec<(usize, f32)> = row.iter().cloned().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let k = top_k.min(4);
                let expected_w = normalize_weights(&indexed[..k]);
                for (j, (idx, w)) in routing.iter().enumerate() {
                    assert_eq!(*idx, indexed[j].0, "rank mismatch s={s} k={top_k}");
                    assert!(
                        (*w - expected_w[j]).abs() < 1e-6,
                        "weight mismatch s={s} k={top_k}: {w} vs {}",
                        expected_w[j]
                    );
                }
                let total: f32 = routing.iter().map(|(_, w)| *w).sum();
                assert!((total - 1.0).abs() < 1e-5);
            }
        }
    }

    /// Unit: engagement sentinel starts disengaged and `invalidate` resets it.
    /// (Engaged state itself is set only by the GPU D2D path; asserted in the
    /// GPU parity tests via `is_routing_engaged`.)
    #[test]
    fn test_charon_cache_engagement_sentinel_lifecycle() {
        let cache = CharonCache::new();
        assert!(
            !cache.is_routing_engaged(),
            "fresh cache must not report engaged routing"
        );
        cache.invalidate();
        assert!(
            !cache.is_routing_engaged(),
            "invalidate must leave routing disengaged"
        );
    }
}

#[cfg(test)]
mod kq_native_integration_tests {
    use super::*;

    /// MODEL-SHAPED integration test — the "how it should work" oracle for
    /// the exact production path: load blk.2's expert bank through
    /// ExpertBank::load on the ROCm device (the storages the model actually
    /// runs with), dispatch through the SAME `ensure_kq_native` +
    /// `moe_fused_dispatch_kq_native_into` pair the eager arm uses, and
    /// compare against host dequant of the same storages' bytes. If the e2e
    /// page fault reproduces here, it is debuggable in a harness; if this
    /// passes while the e2e faults, the difference isolates to the model's
    /// invocation context.
    #[test]
    #[ignore = "needs XING_GGUF + a ROCm device; GRIM_RUN_GPU_TESTS=1"]
    fn kq_native_over_model_bank_via_production_launcher() {
        if std::env::var("GRIM_RUN_GPU_TESTS").as_deref() != Ok("1") {
            return;
        }
        let Ok(path) = std::env::var("XING_GGUF") else { return };
        let hidden = 3584usize;
        let inter = 1024usize;
        let num_experts = 64usize;

        use grim_format::tprov::GgufProvider;
        let prov = GgufProvider::open(&path).expect("open gguf");
        let ws = grim_nn::WeightSource::root(&prov, grim_tensor::Device::Rocm(0))
            .pp("blk")
            .pp("2");
        let bank = grim_nn::moe::ExpertBank::load(&ws, num_experts, hidden, inter, false)
            .expect("load bank on Rocm (model path)");

        fn st_of(t: &grim_tensor::Tensor) -> &grim_backend_rocm::RocmStorage {
            t.storage()
                .as_ref()
                .as_any()
                .downcast_ref::<grim_backend_rocm::RocmStorage>()
                .unwrap()
        }
        eprintln!(
            "[kq-model] gate[0] {:?} bytes {} | down[0] {:?} bytes {}",
            bank.gate[0].weight.dtype(),
            st_of(&bank.gate[0].weight).bytes(),
            bank.down[0].weight.dtype(),
            st_of(&bank.down[0].weight).bytes()
        );

        let dev = grim_backend_rocm::RocmDevice::shared(0);
        let experts: Vec<MoeExpert> = (0..num_experts)
            .map(|i| MoeExpert {
                gate: bank.gate[i].clone(),
                up: bank.up[i].clone(),
                down: bank.down[i].clone(),
            })
            .collect();
        let cache = CharonCache::new();
        let kq = ensure_kq_native(0, &experts, &cache)
            .expect("ensure_kq_native over the model's own storages");

        // Model-shaped routing: 1 token, ONE pair (the host oracle below
        // dequantizes in a debug build — 4 pairs re-dequantized the same
        // expert 4x and took minutes; the routing WEIGHT path is already
        // covered by the production geometry parity test).
        let tokens = vec![0u32];
        let expert_ids = vec![7u32];
        let weights = vec![0.4f32];
        let num_pairs = tokens.len();
        let rsf = 1.0f32;

        let act: Vec<f32> = (0..hidden).map(|i| ((i % 13) as f32 - 6.0) * 0.07).collect();
        let up = |v: Vec<u8>| -> grim_tensor::Tensor {
            grim_tensor::Tensor::new(
                std::sync::Arc::from(
                    dev.from_cpu_bytes(
                        &v,
                        &grim_tensor::Shape::new(vec![v.len()]),
                        grim_tensor::DType {
                            arith: grim_tensor::ArithType::U8,
                            storage: grim_tensor::Storage::Native,
                        },
                    )
                    .expect("upload"),
                ),
                grim_tensor::Shape::new(vec![v.len()]),
                grim_tensor::DType::F32,
                grim_tensor::QuantProvenance::default(),
                grim_tensor::Device::Rocm(0),
            )
        };
        let u32_bytes = |v: &[u32]| -> Vec<u8> {
            v.iter().flat_map(|x| x.to_le_bytes()).collect()
        };
        let f32_bytes = |v: &[f32]| -> Vec<u8> {
            v.iter().flat_map(|x| x.to_le_bytes()).collect()
        };
        let a_t = up(f32_bytes(&act));
        let t_t = up(u32_bytes(&tokens));
        let e_t = up(u32_bytes(&expert_ids));
        let w_t = up(f32_bytes(&weights));
        let out_rocm = grim_backend_rocm::RocmStorage::alloc_gpu(
            &grim_tensor::Shape::new(vec![1, hidden]),
            grim_tensor::DType::F32,
            &dev.allocator_handle(),
            0,
        )
        .expect("out");

        fn arc_of(a: &Arc<dyn grim_tensor::backend::BackendStorage>) -> &grim_backend_rocm::RocmStorage {
            a.as_any()
                .downcast_ref::<grim_backend_rocm::RocmStorage>()
                .unwrap()
        }
        dev.moe_fused_dispatch_kq_native_into(
            st_of(&a_t),
            arc_of(&kq.gate_ptrs),
            arc_of(&kq.up_ptrs),
            arc_of(&kq.down_ptrs),
            st_of(&t_t),
            st_of(&e_t),
            st_of(&w_t),
            num_pairs,
            &out_rocm,
            hidden,
            inter,
            rsf,
            kq.gate_bytes / inter as u64,
            kq.down_bytes / hidden as u64,
            i32::from(kq.down_q4k),
        )
        .expect("kq dispatch over the model's storages");
        eprintln!("[kq-model] dispatch returned (kernel + memset enqueued)");
        dev.synchronize();
        eprintln!("[kq-model] sync done (kernel completed)");
        let got_host = out_rocm.copy_to_host().expect("read out");
        let got: Vec<f32> = got_host
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Host reference over the RUNTIME storages' bytes.
        let mut want = vec![0f32; hidden];
        for (p, &e) in expert_ids.iter().enumerate() {
            let gw = st_of(&bank.gate[e as usize].weight).copy_to_host().unwrap();
            let uw = st_of(&bank.up[e as usize].weight).copy_to_host().unwrap();
            let dwb = st_of(&bank.down[e as usize].weight).copy_to_host().unwrap();
            // gate/up are IQ3_S (110 B per 256); the down bank is Q4_K
            // (144 B per 256) — decoding it with the IQ3_S stride produces
            // finite-but-wrong reference values.
            let deq_iq3s = |blob: &[u8], rows: usize, k: usize| -> Vec<f32> {
                let mut flat = Vec::with_capacity(rows * k);
                for r in 0..rows {
                    let row = &blob[r * (k / 256) * 110..(r + 1) * (k / 256) * 110];
                    flat.extend_from_slice(&grim_quant::dequant_iq3s(row, k).unwrap());
                }
                flat
            };
            let deq_q4k = |blob: &[u8], rows: usize, k: usize| -> Vec<f32> {
                let mut flat = Vec::with_capacity(rows * k);
                for r in 0..rows {
                    let row = &blob[r * (k / 256) * 144..(r + 1) * (k / 256) * 144];
                    flat.extend_from_slice(&grim_quant::dequant_q4k(row, k).unwrap());
                }
                flat
            };
            let g_flat = deq_iq3s(&gw, inter, hidden);
            let u_flat = deq_iq3s(&uw, inter, hidden);
            let mut actv = vec![0f32; inter];
            for j in 0..inter {
                let mut g = 0f32;
                let mut u = 0f32;
                for i in 0..hidden {
                    let av = act[i];
                    g += av * g_flat[j * hidden + i];
                    u += av * u_flat[j * hidden + i];
                }
                actv[j] = g / (1.0 + (-g).exp()) * u;
            }
            let d_flat = deq_q4k(&dwb, hidden, inter);
            for h in 0..hidden {
                let mut acc = 0f32;
                for j in 0..inter {
                    acc += actv[j] * d_flat[h * inter + j];
                }
                want[h] += rsf * weights[p] * acc;
            }
        }
        let mut max_abs = 0f32;
        let mut scale = 1e-6f32;
        for (g, wv) in got.iter().zip(&want) {
            max_abs = max_abs.max((g - wv).abs());
            scale = scale.max(g.abs()).max(wv.abs());
        }
        let rel = max_abs / scale;
        eprintln!(
            "[kq-model] kernel vs host on runtime storages: max_abs {max_abs:.3e} rel {rel:.3e}"
        );
        assert!(rel < 1e-3, "model-shaped integration diverges (rel {rel:.3e})");
    }
}
