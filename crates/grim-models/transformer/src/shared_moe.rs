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
use grim_nn::modules::silu_mul_on_device;
use grim_nn::Linear;
use grim_tensor::backend::BackendDevice;
use grim_tensor::{CoreTensorOps, DType, Device, Shape, Tensor};

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
}

struct ResidentWeights {
    gate: Arc<dyn grim_tensor::BackendStorage>,
    up: Arc<dyn grim_tensor::BackendStorage>,
    down: Arc<dyn grim_tensor::BackendStorage>,
    fingerprint: (usize, usize, usize),
}

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
        }
    }

    /// Clear any cached resident weights (used when a model's expert set is
    /// reloaded or replaced).
    pub fn invalidate(&self) {
        *self.resident.lock().unwrap_or_else(|e| e.into_inner()) = None;
        *self.routing.lock().unwrap_or_else(|e| e.into_inner()) = None;
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
    per_expert_loop(dev, x, experts, shared_expert, routings, routed_scaling_factor)
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
/// * `route_mode` - gating transform: 0 = softmax, 1 = sqrt-softplus (DeepSeek-V4).
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
                let weights = Arc::from(rocm.zeros(
                    &Shape::new(vec![num_pairs]),
                    DType::F32,
                )?);
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
    rocm.moe_route_topk_on_device(
        logits_rocm,
        None,
        tokens_rocm,
        experts_rocm,
        weights_rocm,
        seq_len,
        num_experts,
        top_k,
        route_mode,
    )?;

    // Resolve the resident weight stack (built once).
    let (gate_buf, up_buf, down_buf) = {
        let mut guard = cache.resident.lock().unwrap_or_else(|e| e.into_inner());
        let key = (num_experts, hidden, inter);
        match guard.as_ref() {
            Some(r) if r.fingerprint == key => (
                Arc::clone(&r.gate),
                Arc::clone(&r.up),
                Arc::clone(&r.down),
            ),
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

    // 2. Sortless fused dispatch from device-resident routing (no H2D).
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
    let assignment = grim_backend_rocm::kernels::charon::RoutingAssignment::from_route(
        &indices, &weights,
    )?;
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
        let key = (num_experts, hidden, inter);
        match guard.as_ref() {
            Some(r) if r.fingerprint == key => (
                Arc::clone(&r.gate),
                Arc::clone(&r.up),
                Arc::clone(&r.down),
            ),
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

/// Wavefront size of the shared ROCm device (32 for RDNA, 64 for CDNA), with a
/// fallback for the non-ROCm path (never reached here but keeps the call safe).
fn rocm_wavefront_size(ordinal: usize) -> usize {
    grim_backend_rocm::RocmDevice::shared(ordinal).wavefront_size() as usize
}

/// Stack `experts[*].{gate,up,down}` into three contiguous row-major f32 vecs.
fn stack_expert_weights(
    experts: &[MoeExpert],
    num_experts: usize,
    hidden: usize,
    inter: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let mut gate_flat = Vec::with_capacity(num_experts * inter * hidden);
    let mut up_flat = Vec::with_capacity(num_experts * inter * hidden);
    let mut down_flat = Vec::with_capacity(num_experts * hidden * inter);
    for e in experts.iter().take(num_experts) {
        for (dst, lin) in [
            (&mut gate_flat, &e.gate),
            (&mut up_flat, &e.up),
            (&mut down_flat, &e.down),
        ] {
            let w = lin.weight.to_vec_f32()?;
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

    for s in 0..seq_len {
        let routing = &routings[s];
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
    let seq_len = if num_experts == 0 {
        0
    } else {
        logits_v.len() / num_experts
    };
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
    let lift = |r: std::result::Result<Tensor, grim_tensor::Error>| r.map_err(grim_core::error::Error::from);
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
}