//! Segment-wise activation replay for real gradient checkpointing (WI-X13).
//!
//! [`crate::tape::Tape::free_intermediate_activations`] drops intra-segment
//! intermediates after the forward pass. During backward, a freed segment is
//! reconstructed by [`replay_segment`]: entries with `segment_idx == seg`
//! are re-executed in forward order against the retained inputs (segment
//! boundaries, parameter tensors, cross-segment inputs), and the recomputed
//! activations are placed into an overlay map owned by the backward pass.
//!
//! # Semantics source (per op)
//!
//! Every replayed forward mirrors the EXACT semantics of its production
//! twin / backward counterpart:
//!
//! - `MatMul`: `output = eff(A) @ eff(B)` honoring `transpose_a/transpose_b`.
//!   Linear layers record `transpose_b = true` (`y = x @ W^T`; see
//!   `grim-engine/src/streaming_forward.rs` and `matmul_backward` in ops.rs).
//! - `Add` / `Scale`: trivial routes matching `add_backward` /
//!   `scale_backward` in ops.rs. Add supports elementwise and row-broadcast
//!   (the shapes production call sites use).
//! - `LoRAApply`: `base + scale * (x @ A^T) @ B^T` with
//!   `scale = alpha / rank`, byte-for-byte the math of
//!   `BackendDevice::lora_accumulate` (grim-tensor) as invoked by
//!   `apply_and_record_lora` (ops.rs). A is `[rank, in]`, B is `[out, rank]`.
//!   Note: RSLoRA's `alpha/sqrt(rank)` scaling is not representable in
//!   `TapeMetadata::LoRAApply`, so — exactly like `lora_backward` — replay
//!   assumes the standard `alpha/rank`.
//! - `SiluMul`: `silu(gate) * up` with `silu(v) = v / (1 + exp(-v))`
//!   (`CpuDevice::silu_mul`; matches the CPU path of `silu_mul_backward`).
//! - `RmsNorm`: row-wise over the last dim,
//!   `(x / sqrt(mean(x^2) + eps)) * weight` (`CpuDevice::rms_norm`; same rms
//!   definition as `rmsnorm_backward`).
//! - `Rope`: half-split pair rotation `(x_i, x_{i+h})` by `(cos_i, sin_i)`:
//!   `y_i = x_i c - x_{i+h} s`, `y_{i+h} = x_i s + x_{i+h} c`. This is the
//!   exact inverse of `rope_backward` (ops.rs), which reads the recorded
//!   cos/sin tensors — replayed activations therefore match what backward
//!   expects. (The production `CpuDevice::rope` kernel uses interleaved
//!   adjacent pairs and derives its tables internally; no call site records
//!   Rope entries today, so backward consistency wins here.)
//! - `Softmax`: numerically stable max-subtracted softmax along the last
//!   dim (matches what `softmax_backward` expects: sum-to-one rows).
//! - `Embedding`: `output[i] = weight[token_ids[i]]` row gather
//!   (`CpuDevice::embedding`; matches `embedding_backward`'s scatter-add).
//!
//! Output shapes come from metadata plus resolved input shapes (`MatMul`
//! from `m,k,n`, `Embedding` from `[token_ids.len(), hidden_dim]`,
//! everything else from its primary input's shape) because freed outputs no
//! longer have a live tensor to copy the shape from.
//!
//! # Device migration
//!
//! Replay executes on f32 CPU storages regardless of where the original
//! forward ran. Values are what matter for gradient parity; a GPU-trained
//! activation being replayed on CPU is acceptable by design. Reconstructed
//! tensors are plain F32 CPU tensors with default provenance (same
//! construction pattern as the crate's test helpers).

use crate::tape::{Tape, TapeEntry, TapeKind, TapeMetadata, TensorId};
use grim_tensor::error::{Error, Result};
use grim_tensor::{Shape, Tensor};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Number of segment replays performed process-wide (monotonic counter).
pub static REPLAY_COUNT: AtomicUsize = AtomicUsize::new(0);

/// How many segment replays have been executed since process start.
pub fn replay_count() -> usize {
    REPLAY_COUNT.load(Ordering::Relaxed)
}

/// Reset the replay counter (test convenience).
pub fn reset_replay_count() {
    REPLAY_COUNT.store(0, Ordering::Relaxed);
}

/// Reconstruct every dropped intermediate produced by checkpoint segment
/// `seg`.
///
/// Entries with `segment_idx == seg` are replayed in forward order. Each
/// input is resolved from `overlay` first, then from `tape` (retained
/// boundary/cross-segment/parameter tensors); an input present in neither is
/// a hard error. Outputs are inserted into `overlay`. Entries whose output is
/// still materialized (live in the tape or already overlaid) are skipped, so
/// repeated calls for an already-replayed segment are cheap no-ops.
pub fn replay_segment(
    tape: &Tape,
    seg: usize,
    overlay: &mut HashMap<TensorId, Tensor>,
) -> Result<()> {
    let mut replayed_any = false;
    for entry in tape.entries().iter().filter(|e| e.segment_idx == seg) {
        if overlay.contains_key(&entry.output) || tape.get(entry.output).is_some() {
            continue; // still materialized — nothing to recompute
        }
        let out = replay_entry(tape, entry, overlay)?;
        overlay.insert(entry.output, out);
        replayed_any = true;
    }
    if replayed_any {
        REPLAY_COUNT.fetch_add(1, Ordering::Relaxed);
    }
    Ok(())
}

fn resolve_input<'t>(
    tape: &'t Tape,
    overlay: &'t HashMap<TensorId, Tensor>,
    id: TensorId,
    ctx: &str,
) -> Result<&'t Tensor> {
    overlay.get(&id).or_else(|| tape.get(id)).ok_or_else(|| {
        Error::Backend(format!(
            "replay_segment({ctx}): missing input tensor {id:?}"
        ))
    })
}

fn meta_mismatch(kind: TapeKind, other: &TapeMetadata) -> Error {
    Error::Backend(format!(
        "replay_segment: metadata {other:?} does not match kind {kind:?}"
    ))
}

/// Build an F32 CPU tensor with default provenance (test-helper pattern).
fn cpu_f32(data: Vec<f32>, shape: Shape) -> Tensor {
    grim_backend_cpu::cpu_tensor(data, shape)
}

fn replay_entry(
    tape: &Tape,
    entry: &TapeEntry,
    overlay: &HashMap<TensorId, Tensor>,
) -> Result<Tensor> {
    let r = |i: usize| resolve_input(tape, overlay, entry.inputs[i], "entry");

    match entry.kind {
        TapeKind::MatMul => {
            let (transpose_a, transpose_b, m, k, n) = match &entry.metadata {
                TapeMetadata::MatMul {
                    transpose_a,
                    transpose_b,
                    m,
                    k,
                    n,
                } => (*transpose_a, *transpose_b, *m, *k, *n),
                other => return Err(meta_mismatch(TapeKind::MatMul, other)),
            };
            let a = r(0)?;
            let b = r(1)?;
            let av = a.to_vec_f32()?;
            let bv = b.to_vec_f32()?;
            // Effective (post-transpose) operand extents.
            let (ar, ac) = if transpose_a { (k, m) } else { (m, k) };
            let (br, bc) = if transpose_b { (n, k) } else { (k, n) };
            if av.len() != ar * ac || bv.len() != br * bc || ac != br {
                return Err(Error::Backend(format!(
                    "replay matmul: operands {}x{} ({} els) and {}x{} ({} els) disagree with m={m} k={k} n={n}",
                    ar,
                    ac,
                    av.len(),
                    br,
                    bc,
                    bv.len()
                )));
            }
            let mut out = vec![0.0f32; m * n];
            for i in 0..m {
                let obase = i * n;
                for l in 0..k {
                    let a_il = if transpose_a {
                        av[l * m + i]
                    } else {
                        av[i * k + l]
                    };
                    if a_il == 0.0 {
                        continue;
                    }
                    if transpose_b {
                        // B stored [n,k]: column l of B^T is row-strided by k.
                        for j in 0..n {
                            out[obase + j] += a_il * bv[j * k + l];
                        }
                    } else {
                        // B stored [k,n]: row l contiguous.
                        let b_base = l * n;
                        for j in 0..n {
                            out[obase + j] += a_il * bv[b_base + j];
                        }
                    }
                }
            }
            Ok(cpu_f32(out, Shape::new(vec![m, n])))
        }
        TapeKind::Add => {
            let lhs = r(0)?;
            let rhs = r(1)?;
            let lv = lhs.to_vec_f32()?;
            let rv = rhs.to_vec_f32()?;
            let shape = lhs.shape().clone();
            let n = shape.elem_count();
            let mut out = vec![0.0f32; n];
            if rv.len() == n {
                for (o, (&l, &rr)) in out.iter_mut().zip(lv.iter().zip(rv.iter())) {
                    *o = l + rr;
                }
            } else if !rv.is_empty() && n % rv.len() == 0 {
                // Row-broadcast rhs (single row repeated across leading dims).
                for (i, o) in out.iter_mut().enumerate() {
                    *o = lv[i] + rv[i % rv.len()];
                }
            } else {
                return Err(Error::Backend(format!(
                    "replay add: lhs {} els vs rhs {} els not broadcastable",
                    lv.len(),
                    rv.len()
                )));
            }
            Ok(cpu_f32(out, shape))
        }
        TapeKind::Scale => {
            let factor = match &entry.metadata {
                TapeMetadata::Scale { factor } => *factor,
                other => return Err(meta_mismatch(TapeKind::Scale, other)),
            };
            let inp = r(0)?;
            let out = inp.to_vec_f32()?.into_iter().map(|v| v * factor).collect();
            Ok(cpu_f32(out, inp.shape().clone()))
        }
        TapeKind::LoRAApply => {
            let (alpha, rank, meta_a, meta_b) = match &entry.metadata {
                TapeMetadata::LoRAApply { alpha, rank, a, b } => (*alpha, *rank, *a, *b),
                other => return Err(meta_mismatch(TapeKind::LoRAApply, other)),
            };
            let base = r(0)?;
            let x = r(1)?;
            // A/B live as registered param tensors; fall back to the
            // param-id -> tensor map if the input ids were unavailable.
            let a = resolve_input(tape, overlay, entry.inputs[2], "lora a").or_else(|_| {
                tape.param_tensor(meta_a)
                    .ok_or_else(|| Error::Backend("replay lora: missing a tensor".into()))
                    .and_then(|tid| {
                        tape.get(tid)
                            .ok_or_else(|| Error::Backend("replay lora: missing a tensor".into()))
                    })
            })?;
            let b = resolve_input(tape, overlay, entry.inputs[3], "lora b").or_else(|_| {
                tape.param_tensor(meta_b)
                    .ok_or_else(|| Error::Backend("replay lora: missing b tensor".into()))
                    .and_then(|tid| {
                        tape.get(tid)
                            .ok_or_else(|| Error::Backend("replay lora: missing b tensor".into()))
                    })
            })?;

            let xv = x.to_vec_f32()?;
            let av = a.to_vec_f32()?;
            let bv = b.to_vec_f32()?;
            let basev = base.to_vec_f32()?;

            let x_dims = x.shape().dims();
            let (batch, in_features) = match x_dims.len() {
                1 => (1usize, x_dims[0]),
                _ => (
                    x_dims[..x_dims.len() - 1].iter().product::<usize>(),
                    x_dims[x_dims.len() - 1],
                ),
            };
            let a_dims = a.shape().dims();
            let b_dims = b.shape().dims();
            if a_dims.len() != 2 || b_dims.len() != 2 {
                return Err(Error::Backend(format!(
                    "replay lora: A/B must be 2-D, got {:?} / {:?}",
                    a_dims, b_dims
                )));
            }
            let (a_rank, a_in) = (a_dims[0], a_dims[1]);
            let (b_out, b_rank) = (b_dims[0], b_dims[1]);
            if a_in != in_features || a_rank != b_rank {
                return Err(Error::Backend(format!(
                    "replay lora: shape mismatch x [{batch},{in_features}], A [{a_rank},{a_in}], B [{b_out},{b_rank}]"
                )));
            }

            // h = x @ A^T : [batch, a_rank]
            let mut h = vec![0.0f32; batch * a_rank];
            for bi in 0..batch {
                let x_row = bi * in_features;
                let h_row = bi * a_rank;
                for l in 0..in_features {
                    let x_l = xv[x_row + l];
                    if x_l == 0.0 {
                        continue;
                    }
                    for rr in 0..a_rank {
                        h[h_row + rr] += x_l * av[rr * a_in + l];
                    }
                }
            }
            // out = base + scale * (h @ B^T) : [batch, b_out]
            let mut out = basev;
            if out.len() != batch * b_out {
                return Err(Error::Backend(format!(
                    "replay lora: base {} els vs expected {}",
                    out.len(),
                    batch * b_out
                )));
            }
            let scale = alpha / rank as f32;
            for bi in 0..batch {
                let h_row = bi * a_rank;
                let o_row = bi * b_out;
                for rr in 0..a_rank {
                    let hv = h[h_row + rr] * scale;
                    if hv == 0.0 {
                        continue;
                    }
                    for oo in 0..b_out {
                        out[o_row + oo] += hv * bv[oo * a_rank + rr];
                    }
                }
            }
            Ok(cpu_f32(out, base.shape().clone()))
        }
        TapeKind::SiluMul => {
            let gate = r(0)?;
            let up = r(1)?;
            let gv = gate.to_vec_f32()?;
            let uv = up.to_vec_f32()?;
            if gv.len() != uv.len() {
                return Err(Error::Backend(format!(
                    "replay silu_mul: gate/up size mismatch {} vs {}",
                    gv.len(),
                    uv.len()
                )));
            }
            let out: Vec<f32> = gv
                .iter()
                .zip(uv.iter())
                .map(|(&g, &u)| {
                    let silu = g / (1.0f32 + (-g).exp());
                    silu * u
                })
                .collect();
            Ok(cpu_f32(out, gate.shape().clone()))
        }
        TapeKind::RmsNorm => {
            let eps = match &entry.metadata {
                TapeMetadata::RmsNorm { eps, .. } => *eps,
                other => return Err(meta_mismatch(TapeKind::RmsNorm, other)),
            };
            let x = r(0)?;
            let w = r(1)?;
            let xv = x.to_vec_f32()?;
            let wv = w.to_vec_f32()?;
            let dim = wv.len();
            if dim == 0 || xv.len() % dim != 0 {
                return Err(Error::Backend(format!(
                    "replay rmsnorm: invalid hidden_dim {dim} for input size {}",
                    xv.len()
                )));
            }
            let rows = xv.len() / dim;
            let mut out = vec![0.0f32; xv.len()];
            for rw in 0..rows {
                let off = rw * dim;
                let mean_sq: f32 =
                    xv[off..off + dim].iter().map(|&v| v * v).sum::<f32>() / dim as f32;
                let inv = 1.0f32 / (mean_sq + eps).sqrt();
                for c in 0..dim {
                    out[off + c] = xv[off + c] * inv * wv[c];
                }
            }
            Ok(cpu_f32(out, x.shape().clone()))
        }
        TapeKind::Rope => {
            let x = r(0)?;
            let cos = r(1)?;
            let sin = r(2)?;
            let xv = x.to_vec_f32()?;
            let cv = cos.to_vec_f32()?;
            let sv = sin.to_vec_f32()?;
            let half = cv.len().min(sv.len());
            if half == 0 || xv.len() % (half * 2) != 0 {
                return Err(Error::Backend(format!(
                    "replay rope: cos/sin len {half} incompatible with input size {}",
                    xv.len()
                )));
            }
            let head = half * 2;
            let mut out = xv.clone();
            for t in 0..xv.len() / head {
                let off = t * head;
                for i in 0..half {
                    let x0 = xv[off + i];
                    let x1 = xv[off + half + i];
                    let c = cv[i];
                    let s = sv[i];
                    out[off + i] = x0 * c - x1 * s;
                    out[off + half + i] = x0 * s + x1 * c;
                }
            }
            Ok(cpu_f32(out, x.shape().clone()))
        }
        TapeKind::Softmax => {
            let x = r(0)?;
            let xv = x.to_vec_f32()?;
            let last = x.shape().dims().last().copied().unwrap_or(0);
            if last == 0 || xv.len() % last != 0 {
                return Err(Error::Backend(format!(
                    "replay softmax: last dim {last} invalid for input size {}",
                    xv.len()
                )));
            }
            let mut out = vec![0.0f32; xv.len()];
            for rw in 0..xv.len() / last {
                let off = rw * last;
                let row = &xv[off..off + last];
                let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for (i, &v) in row.iter().enumerate() {
                    let e = (v - mx).exp();
                    out[off + i] = e;
                    sum += e;
                }
                let inv = 1.0f32 / sum;
                for i in 0..last {
                    out[off + i] *= inv;
                }
            }
            Ok(cpu_f32(out, x.shape().clone()))
        }
        TapeKind::Embedding => {
            let (token_ids, vocab_size, hidden_dim) = match &entry.metadata {
                TapeMetadata::Embedding {
                    token_ids,
                    vocab_size,
                    hidden_dim,
                    ..
                } => (token_ids, *vocab_size, *hidden_dim),
                other => return Err(meta_mismatch(TapeKind::Embedding, other)),
            };
            let w = r(0)?;
            let wv = w.to_vec_f32()?;
            if wv.len() != vocab_size * hidden_dim {
                return Err(Error::Backend(format!(
                    "replay embedding: weight {} els vs vocab*hidden {}",
                    wv.len(),
                    vocab_size * hidden_dim
                )));
            }
            let mut out = vec![0.0f32; token_ids.len() * hidden_dim];
            for (i, &tok) in token_ids.iter().enumerate() {
                let t = tok as usize;
                if t >= vocab_size {
                    return Err(Error::Backend(format!(
                        "replay embedding: token {t} >= vocab {vocab_size}"
                    )));
                }
                out[i * hidden_dim..(i + 1) * hidden_dim]
                    .copy_from_slice(&wv[t * hidden_dim..(t + 1) * hidden_dim]);
            }
            Ok(cpu_f32(out, Shape::new(vec![token_ids.len(), hidden_dim])))
        }
    }
}
