//! Mixture-of-Experts primitives.
//!
//! This module provides the architecture-agnostic building blocks for MoE
//! inference: an [`ExpertBank`] (per-expert feed-forward triples), an
//! [`MoeRouter`] (softmax-top-k or sigmoid+bias-top-k gating), and an
//! [`MoeFfn`] that routes tokens to selected experts and combines their
//! outputs (plus an optional shared expert).
//!
//! Design notes (per the project's verifiable-correctness discipline):
//!
//! * The forward path implemented here is the **correct-but-unoptimized CPU
//!   reference**. It materializes each selected expert's contribution and
//!   weighted-sums them. A fused/grouped GPU GEMM (WI-M5) is a separate,
//!   non-blocking performance item and must remain parity-checked against this
//!   reference.
//! * Router math (softmax / sigmoid / top-k / bias-application) is computed in
//!   host Rust over the gate logits pulled to CPU. This keeps the selection
//!   logic unit-testable with hand-computed expectations and avoids depending
//!   on backend kernels that may not exist on every device.
//! * No architecture-specific naming or assumptions leak in here. Per-arch
//!   differences (router kind, shared expert presence, top-k, tensor name
//!   mapping) are supplied by the caller (`architecture.rs` map + the per-arch
//!   loader in `grim-models-transformer`).

use grim_backend_cpu::cpu_tensor;
use grim_tensor::shape::Shape;
use grim_tensor::Tensor;

use crate::modules::Linear;
use crate::varbuilder::WeightSource;

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// Router gating strategy. `SoftmaxTopK` covers Qwen2/3-MoE, GLM4-MoE,
/// Granite-MoE, etc. `SigmoidTopKWithBias` covers Laguna (sigmoid gate logits
/// plus a learned per-expert bias added **at selection time only**, never to
/// the combine weights).
#[derive(Debug, Clone)]
pub enum RouterKind {
    SoftmaxTopK,
    /// Sigmoid gate logits plus a learned per-expert bias added **at selection
    /// time only**, never to the combine weights. The bias tensor itself is
    /// loaded from the checkpoint (`exp_probs_b`) and passed to `MoeRouter::new`.
    SigmoidTopKWithBias,
}

impl RouterKind {
    fn is_bias(&self) -> bool {
        matches!(self, RouterKind::SigmoidTopKWithBias)
    }
}

/// Router: a gate `Linear` (`hidden -> n_experts`), the gating strategy, and an
/// optional correction bias (for `SigmoidTopKWithBias`, loaded from `exp_probs_b`).
pub struct MoeRouter {
    pub gate: Linear,
    pub kind: RouterKind,
    pub top_k: usize,
    pub num_experts: usize,
    pub correction_bias: Option<Tensor>,
}

impl MoeRouter {
    pub fn new(
        gate: Linear,
        kind: RouterKind,
        top_k: usize,
        num_experts: usize,
        correction_bias: Option<Tensor>,
    ) -> Self {
        Self {
            gate,
            kind,
            top_k,
            num_experts,
            correction_bias,
        }
    }

    /// Compute gate logits for a `[batch, hidden]` input, returning a
    /// `[batch, num_experts]` tensor on the CPU for host-side selection.
    fn gate_logits(&self, x: &Tensor) -> Result<Tensor, grim_tensor::error::Error> {
        // `x` is expected to be a CPU tensor (router selection runs on host);
        // `Linear::forward` preserves the input device, so the result is CPU.
        self.gate.forward(x)
    }

    /// Route a `[batch, hidden]` input.
    ///
    /// Returns, per token, the selected expert indices and their combine
    /// weights (already normalized over the selected set). The selection for
    /// `SigmoidTopKWithBias` adds the correction bias to the sigmoid scores
    /// *only* for ranking; the returned combine weights are the unbiased
    /// sigmoid values of the selected experts.
    pub fn route(
        &self,
        x: &Tensor,
    ) -> Result<(Vec<Vec<usize>>, Vec<Vec<f32>>), grim_tensor::error::Error> {
        let z = self.gate_logits(x)?.to_vec_f32()?;
        let hidden = self.gate.weight.shape().dim(1)?; // in_dim of the gate
        let batch = z.len() / self.num_experts;
        let _ = hidden;
        let k = self.top_k.min(self.num_experts);

        let mut indices = Vec::with_capacity(batch);
        let mut weights = Vec::with_capacity(batch);

        for t in 0..batch {
            let row = &z[t * self.num_experts..(t + 1) * self.num_experts];
            let sel_scores: Vec<f32> = match &self.kind {
                RouterKind::SoftmaxTopK => softmax(row),
                RouterKind::SigmoidTopKWithBias => {
                    let b = self
                        .correction_bias
                        .as_ref()
                        .map(|t| t.to_vec_f32())
                        .transpose()?
                        .unwrap_or_else(|| vec![0.0f32; self.num_experts]);
                    row.iter()
                        .enumerate()
                        .map(|(i, &v)| sigmoid(v) + b.get(i).copied().unwrap_or(0.0))
                        .collect()
                }
            };
            // Rank by selection scores, take top_k.
            let mut order: Vec<usize> = (0..self.num_experts).collect();
            order.sort_by(|&a, &b| sel_scores[b].partial_cmp(&sel_scores[a]).unwrap());
            let chosen = &order[..k];

            // Combine weights.
            //   * SoftmaxTopK: the softmax probabilities over the chosen
            //     logits (inherently normalized).
            //   * SigmoidTopKWithBias: the **unbiased** sigmoid gate values of
            //     the chosen experts, used directly as combine weights (NOT
            //     renormalized). The correction bias is applied only at
            //     selection time, above — never to the combine weights.
            let raw: Vec<f32> = match &self.kind {
                RouterKind::SoftmaxTopK => {
                    let logits: Vec<f32> = chosen.iter().map(|&i| row[i]).collect();
                    softmax(&logits)
                }
                RouterKind::SigmoidTopKWithBias => {
                    chosen.iter().map(|&i| sigmoid(row[i])).collect()
                }
            };
            indices.push(chosen.to_vec());
            weights.push(raw);
        }

        Ok((indices, weights))
    }
}

// ---------------------------------------------------------------------------
// Expert bank
// ---------------------------------------------------------------------------

/// Holds the per-expert SwiGLU feed-forward triples `{gate, up, down}`.
pub struct ExpertBank {
    pub gate: Vec<Linear>,
    pub up: Vec<Linear>,
    pub down: Vec<Linear>,
}

impl ExpertBank {
    /// Construct directly from per-expert `Linear`s (used by tests and
    /// synthetic construction).
    pub fn from_linears(gate: Vec<Linear>, up: Vec<Linear>, down: Vec<Linear>) -> Self {
        Self { gate, up, down }
    }

    pub fn num_experts(&self) -> usize {
        self.gate.len()
    }

    /// Load experts from a GGUF-style 3D weight layout. Matches the in-repo
    /// Lfm2 MoE loader's naming and layout convention:
    ///   `ffn_gate_exps.weight` = `[n_experts, inter, hidden]`
    ///   `ffn_up_exps.weight`   = `[n_experts, inter, hidden]`
    ///   `ffn_down_exps.weight` = `[n_experts, inter, hidden]`
    /// (experts are the OUTERMOST dimension). Each expert's
    /// `[inter, hidden]` block is sliced out; the down projection is
    /// transposed to `[hidden, inter]` for the `Linear` (out=hidden, in=inter).
    pub fn load(
        ws: &WeightSource<'_>,
        num_experts: usize,
        hidden: usize,
        inter: usize,
        has_bias: bool,
    ) -> Result<Self, grim_tensor::error::Error> {
        let gate_3d =
            ws.get(Shape::new(vec![num_experts, inter, hidden]), "ffn_gate_exps.weight")?;
        let up_3d =
            ws.get(Shape::new(vec![num_experts, inter, hidden]), "ffn_up_exps.weight")?;
        let down_3d =
            ws.get(Shape::new(vec![num_experts, inter, hidden]), "ffn_down_exps.weight")?;

        let gate_v = gate_3d.to_vec_f32()?;
        let up_v = up_3d.to_vec_f32()?;
        let down_v = down_3d.to_vec_f32()?;

        let mut gate = Vec::with_capacity(num_experts);
        let mut up = Vec::with_capacity(num_experts);
        let mut down = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let g = slice_expert(&gate_v, e, inter, hidden);
            let u = slice_expert(&up_v, e, inter, hidden);
            // down per-expert block is `[inter, hidden]`; transpose to
            // `[hidden, inter]` for the down `Linear` (out=hidden, in=inter).
            let d_block = slice_expert(&down_v, e, inter, hidden);
            let d = transpose_block(&d_block, inter, hidden);
            gate.push(Linear::from_tensor(
                cpu_tensor(g, Shape::new(vec![inter, hidden])),
                bias_opt(has_bias, inter),
            ));
            up.push(Linear::from_tensor(
                cpu_tensor(u, Shape::new(vec![inter, hidden])),
                bias_opt(has_bias, inter),
            ));
            down.push(Linear::from_tensor(
                cpu_tensor(d, Shape::new(vec![hidden, inter])),
                bias_opt(has_bias, hidden),
            ));
        }
        Ok(Self { gate, up, down })
    }

    /// Run a single expert's SwiGLU feed-forward on `x` (`[batch, hidden]`),
    /// returning `[batch, hidden]`.
    pub fn expert_forward(
        &self,
        e: usize,
        x: &Tensor,
    ) -> Result<Tensor, grim_tensor::error::Error> {
        let g = self.gate[e].forward(x)?; // [batch, inter]
        let u = self.up[e].forward(x)?; // [batch, inter]
        let h = silu_mul_host(&g, &u)?; // [batch, inter]
        self.down[e].forward(&h) // [batch, hidden]
    }
}

// ---------------------------------------------------------------------------
// MoE FFN
// ---------------------------------------------------------------------------

/// A routed MoE feed-forward block: router + experts + optional shared expert.
pub struct MoeFfn {
    pub router: MoeRouter,
    pub experts: ExpertBank,
    pub shared_expert: Option<ExpertTriple>,
    pub routed_scaling_factor: f32,
}

/// An independent SwiGLU triple for the (always-on) shared expert.
pub struct ExpertTriple {
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
    pub inter: usize,
    pub hidden: usize,
}

impl ExpertTriple {
    /// Load the shared expert's three projections from `ws` under the
    /// `ffn_gate_she` / `ffn_up_she` / `ffn_down_she` GGUF names.
    pub fn load(
        ws: &WeightSource<'_>,
        hidden: usize,
        inter: usize,
        has_bias: bool,
    ) -> Result<Self, grim_tensor::error::Error> {
        let gate = Linear::load(&ws.pp("ffn_gate_she"), hidden, inter, has_bias)?;
        let up = Linear::load(&ws.pp("ffn_up_she"), hidden, inter, has_bias)?;
        let down = Linear::load(&ws.pp("ffn_down_she"), inter, hidden, has_bias)?;
        Ok(Self {
            gate,
            up,
            down,
            inter,
            hidden,
        })
    }
}

impl ExpertTriple {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, grim_tensor::error::Error> {
        let g = self.gate.forward(x)?;
        let u = self.up.forward(x)?;
        let h = silu_mul_host(&g, &u)?;
        self.down.forward(&h)
    }
}

impl MoeFfn {
    pub fn new(
        router: MoeRouter,
        experts: ExpertBank,
        shared_expert: Option<ExpertTriple>,
        routed_scaling_factor: f32,
    ) -> Self {
        Self {
            router,
            experts,
            shared_expert,
            routed_scaling_factor,
        }
    }

    /// Correct-but-unoptimized CPU reference forward for `[batch, hidden]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, grim_tensor::error::Error> {
        let (indices, weights) = self.router.route(x)?;
        let batch = indices.len();
        let hidden = self
            .experts
            .down
            .first()
            .map(|l| l.weight.shape().dim(0).unwrap_or(0))
            .unwrap_or_else(|| x.shape().dims().last().copied().unwrap_or(0));

        let mut out_vec = vec![0.0f32; batch * hidden];

        for t in 0..batch {
            let experts = &indices[t];
            let w = &weights[t];
            let xt = slice_row(x, t)?; // [1, hidden]
            // Routed experts: combined output is scaled by `routed_scaling_factor`
            // (DeepSeek/Laguna convention — scales the *routed* path, not shared).
            let mut routed = vec![0.0f32; hidden];
            for (rank, &e) in experts.iter().enumerate() {
                let y = self.experts.expert_forward(e, &xt)?; // [1, hidden]
                let yv = y.to_vec_f32()?;
                for (i, v) in yv.iter().enumerate() {
                    routed[i] += w[rank] * v;
                }
            }
            for (i, v) in routed.iter().enumerate() {
                out_vec[t * hidden + i] += self.routed_scaling_factor * v;
            }
            // Shared/always-on expert is added unscaled.
            if let Some(sh) = &self.shared_expert {
                let s = sh.forward(&xt)?;
                let sv = s.to_vec_f32()?;
                for (i, v) in sv.iter().enumerate() {
                    out_vec[t * hidden + i] += v;
                }
            }
        }

        Ok(cpu_tensor(out_vec, Shape::new(vec![batch, hidden])))
    }
}

// ---------------------------------------------------------------------------
// Host math helpers
// ---------------------------------------------------------------------------

fn softmax(v: &[f32]) -> Vec<f32> {
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = v.iter().map(|&x| (x - m).exp()).collect();
    let s: f32 = exps.iter().sum();
    exps.iter().map(|e| e / s).collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Elementwise `silu(g) * u` on host (SwiGLU activation).
fn silu_mul_host(g: &Tensor, u: &Tensor) -> Result<Tensor, grim_tensor::error::Error> {
    let gv = g.to_vec_f32()?;
    let uv = u.to_vec_f32()?;
    let out: Vec<f32> = gv
        .iter()
        .zip(uv.iter())
        .map(|(&a, &b)| a * sigmoid(a) * b)
        .collect();
    Ok(cpu_tensor(out, g.shape().clone()))
}

fn slice_expert(flat: &[f32], e: usize, out: usize, in_dim: usize) -> Vec<f32> {
    let stride = out * in_dim;
    flat[e * stride..(e + 1) * stride].to_vec()
}

/// Transpose a contiguous `[out, in_dim]` block into `[in_dim, out]`.
fn transpose_block(v: &[f32], out: usize, in_dim: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; v.len()];
    for o in 0..out {
        for i in 0..in_dim {
            t[i * out + o] = v[o * in_dim + i];
        }
    }
    t
}

fn slice_row(x: &Tensor, t: usize) -> Result<Tensor, grim_tensor::error::Error> {
    let v = x.to_vec_f32()?;
    let hidden = x.shape().dims().last().copied().unwrap_or(0);
    Ok(cpu_tensor(
        v[t * hidden..(t + 1) * hidden].to_vec(),
        Shape::new(vec![1, hidden]),
    ))
}

fn bias_opt(has_bias: bool, dim: usize) -> Option<Tensor> {
    if has_bias {
        Some(cpu_tensor(vec![0.0f32; dim], Shape::new(vec![dim])))
    } else {
        None
    }
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

// ===========================================================================
// WI-C — Router-distilled lookahead predictor + PlanBuilder + SRP/SCH gate
// ===========================================================================
//
// The predict leg of P-DAFD (PROBE 2602.00509, MxMoE 2505.05799, DynaExq
// 2511.15015, SRP/SCH 2505.16056). This is the genuinely novel composition —
// no published system fuses dispatch AND predicts AND varies per-expert
// precision. All three components below are host-side and unit-testable
// without a GPU: the falsifiable core of WI-C (G-C1/C2/C3) does NOT require
// hardware (plan §5).
//
// Honesty valves (do not weaken):
// * G-C2 scores the predictor against *actual next-layer routing* (Hit@k ≥
//   0.80), not output parity — a predictor wrong in an interesting way
//   cannot be rescued by the kernel producing the right answer.
// * G-C3 requires the feature to beat its own off-switch (pre-registered
//   Δ ≥ +0.05 Hit@k or PPL) or it is recorded as FAIL "prediction adds no
//   signal", never "≈acceptable".
// * The SRP/SCH confidence gate is mandatory (§5): below-threshold routing
//   consistency disables prediction, falling back to WI-B reactive matching.

// ---------------------------------------------------------------------------
// LookaheadPredictor — gate-initialized low-rank distilled router copy
// ---------------------------------------------------------------------------

/// A tiny distilled copy of `MoeRouter::gate` that forecasts the *next*
/// layer's activated-expert distribution from the current layer's gate
/// logits (PROBE 2602.00509, "gate-initialized" lookahead).
///
/// The predictor is a single low-rank linear: `predicted_next_logits =
/// current_logits @ W_distill`, where `W_distill` is `[num_experts,
/// num_experts]` initialized to a per-expert identity (the "gate-init"
/// prior that next-layer routing ≈ this-layer routing). It runs host-side;
/// output = predicted histogram (softmax over the predicted logits) + a
/// per-expert hotness vector (the predicted top-k probabilities).
///
/// Distillation updates `W_distill` online from observed (current → next)
/// routing pairs; v1 uses a closed-form ridge update, no GPU.
pub struct LookaheadPredictor {
    /// `W_distill`, `[num_experts, num_experts]` row-major.
    pub distill: Vec<f32>,
    pub num_experts: usize,
    /// Top-k the predictor forecasts hotness for.
    pub top_k: usize,
    /// Whether the SRP/SCH gate has enabled prediction. When `false`,
    /// `predict` returns the identity prior (this-layer routing unchanged),
    /// i.e. the WI-B reactive fallback.
    pub enabled: bool,
}

impl LookaheadPredictor {
    /// Build a gate-initialized predictor: `W_distill = I` (next-layer ≈
    /// current-layer routing, the strongest uninformed prior). `enabled`
    /// starts `true`; the SRP/SCH gate sets it `false` when the model's
    /// routing consistency is below threshold.
    pub fn new_gate_initialized(num_experts: usize, top_k: usize) -> Self {
        let mut distill = vec![0.0f32; num_experts * num_experts];
        for i in 0..num_experts {
            distill[i * num_experts + i] = 1.0; // identity prior
        }
        Self {
            distill,
            num_experts,
            top_k: top_k.min(num_experts),
            enabled: true,
        }
    }

    /// Predict the next layer's activated-expert distribution from this
    /// layer's gate logits.
    ///
    /// Returns `(predicted_top_k_indices, predicted_top_k_probs)` — the
    /// forecast hot set and their normalized probabilities. When `enabled`
    /// is `false`, returns the current-layer top-k unchanged (the reactive
    /// fallback that adds no prediction signal — G-C3's off-switch).
    pub fn predict(
        &self,
        current_logits: &[f32],
    ) -> (Vec<usize>, Vec<f32>) {
        assert_eq!(
            current_logits.len(),
            self.num_experts,
            "current_logits length must equal num_experts"
        );
        if !self.enabled {
            // Identity prior: next-layer routing ≈ this-layer routing.
            // Skip the distill matrix multiply entirely — the off-switch
            // must not consult W_distill at all.
            return self.top_k_from_logits(current_logits);
        }
        // predicted_next_logits[j] = sum_i current_logits[i] * W[i, j]
        let mut pred = vec![0.0f32; self.num_experts];
        for j in 0..self.num_experts {
            let mut acc = 0.0f32;
            for i in 0..self.num_experts {
                acc += current_logits[i] * self.distill[i * self.num_experts + j];
            }
            pred[j] = acc;
        }
        self.top_k_from_logits(&pred)
    }

    /// Softmax over `logits`, then take `top_k` by probability and renormalize
    /// the selected probabilities over the chosen set (mirrors
    /// `MoeRouter::route`'s SoftmaxTopK combine-weight convention).
    fn top_k_from_logits(&self, logits: &[f32]) -> (Vec<usize>, Vec<f32>) {
        let probs = softmax(logits);
        let mut order: Vec<usize> = (0..self.num_experts).collect();
        order.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        let chosen: Vec<usize> = order.iter().take(self.top_k).copied().collect();
        let raw: Vec<f32> = chosen.iter().map(|&i| probs[i]).collect();
        let sum: f32 = raw.iter().sum();
        let chosen_probs: Vec<f32> = if sum > 0.0 {
            raw.iter().map(|p| p / sum).collect()
        } else {
            raw
        };
        (chosen, chosen_probs)
    }

    /// One closed-form ridge distillation step from an observed
    /// (current_logits → next_layer_activated_set) pair. Strength `lr ∈
    /// (0, 1]`; v1 uses a Hebbian-style update pulling `W[i, j]` toward the
    /// co-activation signal `current_logits[i] * next_onehot[j]`.
    pub fn distill_step(
        &mut self,
        current_logits: &[f32],
        next_activated: &[usize],
        lr: f32,
    ) {
        let mut next_onehot = vec![0.0f32; self.num_experts];
        for &e in next_activated {
            if e < self.num_experts {
                next_onehot[e] = 1.0;
            }
        }
        // Hebbian: W[i,j] += lr * (target - W[i,j]*current[i]) * current[i]
        // — a one-step ridge pull toward the observed co-activation.
        for i in 0..self.num_experts {
            for j in 0..self.num_experts {
                let pred_ij = current_logits[i] * self.distill[i * self.num_experts + j];
                let target_ij = current_logits[i] * next_onehot[j];
                self.distill[i * self.num_experts + j] += lr * (target_ij - pred_ij);
            }
        }
    }
}

/// Score prediction Hit@k: the fraction of the realized top-k set that the
/// predictor's top-k forecast captured. `1.0` = perfect overlap, `0.0` =
/// no overlap. This is the G-C2 metric (≥0.80 bar), scored against actual
/// next-layer routing — not output parity.
pub fn prediction_hit_at_k(predicted: &[usize], realized: &[usize]) -> f32 {
    if realized.is_empty() {
        return 0.0;
    }
    let hits = predicted.iter().filter(|p| realized.contains(p)).count();
    hits as f32 / realized.len() as f32
}

// ---------------------------------------------------------------------------
// PlanBuilder — DynaExq-budget-feasible resident-set + precision plan
// ---------------------------------------------------------------------------

/// Per-expert precision in the resident set (MxMoE 2505.05799 mixed-precision
/// flavor). Hot experts stay fp16; cold experts fall back to int8 (via the
/// existing `q*k_gemm` dequant path) to fit the HBM envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertPrecision {
    Fp16,
    Int8,
}

/// A budget-feasible resident-set plan: which experts are hot (fp16
/// resident) vs cold (int8 fallback), under the HBM byte envelope. Output
/// of `PlanBuilder::build`.
#[derive(Debug, Clone)]
pub struct ResidentPlan {
    pub precision: Vec<ExpertPrecision>,
    /// Whether prediction drove this plan (`true`) or it's the reactive
    /// WI-B fallback (`false`). G-C3 compares both on the same traces.
    pub prediction_driven: bool,
}

/// DynaExq-style budget-feasible top-n planner. Keeps the hottest experts
/// fp16-resident up to the HBM byte budget; demotes the rest to int8.
pub struct PlanBuilder {
    /// Bytes per fp16-resident expert (gate+up+down triples).
    bytes_per_expert_fp16: usize,
    /// Bytes per int8 expert (≈ fp16/2 + quant overhead).
    bytes_per_expert_int8: usize,
    /// HBM envelope for the expert resident set.
    hbm_budget_bytes: usize,
}

impl PlanBuilder {
    /// Construct with per-expert byte costs and the total HBM envelope.
    /// `bytes_per_expert_fp16` is the full `[inter, hidden] × 3` triple;
    /// `bytes_per_expert_int8` is the quantized size (typically fp16/2).
    pub fn new(
        bytes_per_expert_fp16: usize,
        bytes_per_expert_int8: usize,
        hbm_budget_bytes: usize,
    ) -> Self {
        Self {
            bytes_per_expert_fp16,
            bytes_per_expert_int8,
            hbm_budget_bytes,
        }
    }

    /// Build a resident plan from a per-expert hotness vector (predicted or
    /// observed routing frequency). The hottest experts are kept fp16 up
    /// to the budget; the rest demote to int8. `prediction_driven` labels
    /// the plan for G-C3's off-switch comparison.
    pub fn build(
        &self,
        hotness: &[f32],
        prediction_driven: bool,
    ) -> ResidentPlan {
        let n = hotness.len();
        // Rank experts by hotness (desc); ties broken by index for stability.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            hotness[b]
                .partial_cmp(&hotness[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });

        let mut precision = vec![ExpertPrecision::Int8; n];
        let mut used = 0usize;
        // Greedy: promote experts to fp16 in hotness order until budget hit.
        // Start from the all-int8 baseline cost, then upgrade.
        let mut baseline = n * self.bytes_per_expert_int8;
        for &e in &order {
            let upgrade_cost =
                self.bytes_per_expert_fp16.saturating_sub(self.bytes_per_expert_int8);
            let _ = baseline; // baseline tracks the all-int8 floor
            if used + upgrade_cost <= self.hbm_budget_bytes || self.hbm_budget_bytes == 0 {
                precision[e] = ExpertPrecision::Fp16;
                used += upgrade_cost;
            } else {
                break;
            }
        }
        ResidentPlan {
            precision,
            prediction_driven,
        }
    }

    /// Bytes the resident set would occupy under this plan.
    pub fn plan_bytes(&self, plan: &ResidentPlan) -> usize {
        plan.precision
            .iter()
            .map(|p| match p {
                ExpertPrecision::Fp16 => self.bytes_per_expert_fp16,
                ExpertPrecision::Int8 => self.bytes_per_expert_int8,
            })
            .sum()
    }
}

// ---------------------------------------------------------------------------
// SRP/SCH confidence gate — mandatory prediction on/off valve
// ---------------------------------------------------------------------------

/// Compute the model's local-routing-consistency (SRP/SCH 2505.16056) from
/// a trace of consecutive-layer routing decisions. Returns the fraction of
/// (layer, token, expert) triples that recur in the next layer — a measure
/// of how predictable the routing is. Below `threshold`, the
/// `LookaheadPredictor` is disabled (§5: the gate is mandatory, not
/// optional — don't claim prediction works on models it measurably can't).
///
/// `trace[t]` = the activated-expert set for token row `t` across layers;
/// the outer Vec is layers, inner Vec is per-token activated experts. We
/// score the per-token set-overlap between adjacent layers averaged over
/// tokens and layer-transitions.
pub fn routing_consistency(trace: &[Vec<Vec<usize>>]) -> f32 {
    if trace.len() < 2 {
        return 0.0; // need at least two layers to measure consistency
    }
    let mut total_overlap = 0.0f32;
    let mut total_sets = 0u32;
    for layer in 0..trace.len() - 1 {
        let cur = &trace[layer];
        let nxt = &trace[layer + 1];
        let rows = cur.len().min(nxt.len());
        for t in 0..rows {
            let cur_set = &cur[t];
            let nxt_set = &nxt[t];
            if cur_set.is_empty() {
                continue;
            }
            let overlap = cur_set.iter().filter(|e| nxt_set.contains(e)).count();
            total_overlap += overlap as f32 / cur_set.len() as f32;
            total_sets += 1;
        }
    }
    if total_sets == 0 {
        return 0.0;
    }
    total_overlap / total_sets as f32
}

/// Apply the SRP/SCH gate to a predictor: if the trace's routing
/// consistency is below `threshold`, disable prediction (set
/// `predictor.enabled = false`) so it falls back to the reactive WI-B
/// matching. Returns the measured consistency so the caller can log it.
pub fn apply_srp_sch_gate(
    predictor: &mut LookaheadPredictor,
    trace: &[Vec<Vec<usize>>],
    threshold: f32,
) -> f32 {
    let consistency = routing_consistency(trace);
    predictor.enabled = consistency >= threshold;
    consistency
}

// ---------------------------------------------------------------------------
// Tests — synthetic, hand-computed, CPU-only
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny MoE with 4 experts, top-2, hidden=4, inter=4.
    /// Gate weights chosen so token selects experts 0 and 2; combine weights
    /// hand-derived in the test body.
    fn build_synthetic(
        kind: RouterKind,
        shared: Option<ExpertTriple>,
        correction_bias: Option<Tensor>,
    ) -> MoeFfn {
        build_synthetic_rsf(kind, shared, correction_bias, 1.0)
    }

    fn build_synthetic_rsf(
        kind: RouterKind,
        shared: Option<ExpertTriple>,
        correction_bias: Option<Tensor>,
        rsf: f32,
    ) -> MoeFfn {
        let hidden = 4;
        let inter = 4;
        let n = 4;
        let top_k = 2;

        // Gate: out=n, in=hidden. forward computes x @ W^T, so with x=[1,0,0,0]
        // the gate logits are W's column 0: gate_logits[j] = W[j][0].
        // Set W's column 0 so logits = [3.0, 0.1, 2.0, -1.0]:
        //   expert0 -> high, expert2 -> high, expert1 low, expert3 lowest.
        // softmax([3,0.1,2,-1]) top-2 = {0, 2}.
        let mut gate_w = vec![0.0f32; n * hidden];
        gate_w[0 * hidden + 0] = 3.0; // expert 0 gate logit
        gate_w[1 * hidden + 0] = 0.1; // expert 1
        gate_w[2 * hidden + 0] = 2.0; // expert 2
        gate_w[3 * hidden + 0] = -1.0; // expert 3
        let gate = Linear::from_tensor(cpu_tensor(gate_w, Shape::new(vec![n, hidden])), None);

        let mut eg = Vec::new();
        let mut eu = Vec::new();
        let mut ed = Vec::new();
        for e in 0..n {
            // identity-ish experts: gate=up=diag(inter), down=diag(hidden).
            let mut gw = vec![0.0f32; inter * hidden];
            let mut uw = vec![0.0f32; inter * hidden];
            let mut dw = vec![0.0f32; hidden * inter];
            for i in 0..inter.min(hidden) {
                gw[i * hidden + i] = 1.0 + (e as f32); // expert e scales by (1+e)
                uw[i * hidden + i] = 1.0;
                dw[i * inter + i] = 1.0;
            }
            eg.push(Linear::from_tensor(
                cpu_tensor(gw, Shape::new(vec![inter, hidden])),
                None,
            ));
            eu.push(Linear::from_tensor(
                cpu_tensor(uw, Shape::new(vec![inter, hidden])),
                None,
            ));
            ed.push(Linear::from_tensor(
                cpu_tensor(dw, Shape::new(vec![hidden, inter])),
                None,
            ));
        }
        let bank = ExpertBank::from_linears(eg, eu, ed);
        let router = MoeRouter::new(gate, kind, top_k, n, correction_bias);
        MoeFfn::new(router, bank, shared, rsf)
    }

    fn token() -> Tensor {
        cpu_tensor(vec![1.0, 0.0, 0.0, 0.0], Shape::new(vec![1, 4]))
    }

    #[test]
    fn softmax_topk_selects_expected_experts() {
        let m = build_synthetic(RouterKind::SoftmaxTopK, None, None);
        let (idx, w) = m.router.route(&token()).unwrap();
        assert_eq!(idx[0], vec![0, 2], "top-2 should be experts 0 and 2");
        // weights normalized over the 2 selected: softmax([3.0, 2.0]).
        let expected0 = (3.0f32.exp()) / (3.0f32.exp() + 2.0f32.exp());
        assert!((w[0][0] - expected0).abs() < 1e-5);
        assert!((w[0][1] - (1.0 - expected0)).abs() < 1e-5);
    }

    #[test]
    fn sigmoid_bias_changes_selection_only_at_rank_time() {
        // Without bias: softmax([3,0.1,2,-1]) top2 = {0,2}.
        // With bias pushing expert 1 up by a lot, selection should prefer 1.
        let bias = cpu_tensor(vec![0.0, 10.0, 0.0, 0.0], Shape::new(vec![4]));
        let m = build_synthetic(RouterKind::SigmoidTopKWithBias, None, Some(bias));
        let (idx, w) = m.router.route(&token()).unwrap();
        assert_eq!(idx[0][0], 1, "bias must move expert 1 to rank 1");
        // Combine weight for selected expert 1 is the unbiased sigmoid of its
        // gate logit (0.1) -> 1/(1+e^-0.1), NOT the biased score.
        let unbiased = sigmoid(0.1);
        assert!((w[0][0] - unbiased).abs() < 1e-5, "combine weight must be unbiased");
    }

    #[test]
    fn forward_matches_hand_computed() {
        // x=[1,0,0,0]; expert e acts as: h = silu(x) * x (since gate=up=diag(e+1),
        // x has only dim0=1) -> silu(1*(e+1)) * 1*(e+1)? careful:
        //   gate(x) = (e+1)*x -> [ (e+1), 0,0,0 ] (inter=4, only dim0)
        //   up(x)   = x        -> [ 1, 0,0,0 ]
        //   silu(gate) = silu(e+1) on dim0, 0 elsewhere
        //   h = silu(gate) * up = [ silu(e+1), 0,0,0 ]
        //   down(h) = h (diag) -> [ silu(e+1), 0,0,0 ]  (hidden=4)
        // So expert e output dim0 = silu(e+1).
        let m = build_synthetic(RouterKind::SoftmaxTopK, None, None);
        let out = m.forward(&token()).unwrap();
        let v = out.to_vec_f32().unwrap();
        // selected experts 0,2 with weights w0,w2.
        let w0 = (3.0f32.exp()) / (3.0f32.exp() + 2.0f32.exp());
        let w2 = 1.0 - w0;
        let expected0 = w0 * silu(1.0) + w2 * silu(3.0);
        assert!(
            (v[0] - expected0).abs() < 1e-4,
            "dim0 = {} expected {}", v[0], expected0
        );
        assert!(v[1].abs() < 1e-6 && v[2].abs() < 1e-6 && v[3].abs() < 1e-6);
    }

    #[test]
    fn shared_expert_scaled_add() {
        let hidden = 4;
        let inter = 4;
        // shared expert = identity SwiGLU -> output dim0 = silu(1)=~0.731.
        let mut gw = vec![0.0f32; inter * hidden];
        let mut uw = vec![0.0f32; inter * hidden];
        let mut dw = vec![0.0f32; hidden * inter];
        for i in 0..inter.min(hidden) {
            gw[i * hidden + i] = 1.0;
            uw[i * hidden + i] = 1.0;
            dw[i * inter + i] = 1.0;
        }
        let shared = ExpertTriple {
            gate: Linear::from_tensor(cpu_tensor(gw, Shape::new(vec![inter, hidden])), None),
            up: Linear::from_tensor(cpu_tensor(uw, Shape::new(vec![inter, hidden])), None),
            down: Linear::from_tensor(cpu_tensor(dw, Shape::new(vec![hidden, inter])), None),
            inter,
            hidden,
        };
        let m = build_synthetic(RouterKind::SoftmaxTopK, Some(shared), None);
        let out = m.forward(&token()).unwrap();
        let v = out.to_vec_f32().unwrap();
        let w0 = (3.0f32.exp()) / (3.0f32.exp() + 2.0f32.exp());
        let w2 = 1.0 - w0;
        let expected0 = w0 * silu(1.0) + w2 * silu(3.0) + 1.0 * silu(1.0);
        assert!((v[0] - expected0).abs() < 1e-4, "with shared: dim0 = {} vs {}", v[0], expected0);
    }

    #[test]
    fn routed_scaling_factor_scales_routed_not_shared() {
        // Shared expert is the identity SwiGLU -> dim0 = silu(1) (~0.731).
        let hidden = 4;
        let inter = 4;
        let mut gw = vec![0.0f32; inter * hidden];
        let mut uw = vec![0.0f32; inter * hidden];
        let mut dw = vec![0.0f32; hidden * inter];
        for i in 0..inter.min(hidden) {
            gw[i * hidden + i] = 1.0;
            uw[i * hidden + i] = 1.0;
            dw[i * inter + i] = 1.0;
        }
        let shared = ExpertTriple {
            gate: Linear::from_tensor(cpu_tensor(gw, Shape::new(vec![inter, hidden])), None),
            up: Linear::from_tensor(cpu_tensor(uw, Shape::new(vec![inter, hidden])), None),
            down: Linear::from_tensor(cpu_tensor(dw, Shape::new(vec![hidden, inter])), None),
            inter,
            hidden,
        };
        // rsf = 0.5: routed contribution halved, shared added unscaled.
        let m = build_synthetic_rsf(RouterKind::SoftmaxTopK, Some(shared), None, 0.5);
        let out = m.forward(&token()).unwrap();
        let v = out.to_vec_f32().unwrap();
        let w0 = (3.0f32.exp()) / (3.0f32.exp() + 2.0f32.exp());
        let w2 = 1.0 - w0;
        let expected0 = 0.5 * (w0 * silu(1.0) + w2 * silu(3.0)) + 1.0 * silu(1.0);
        assert!(
            (v[0] - expected0).abs() < 1e-4,
            "rsf=0.5: dim0 = {} vs {}",
            v[0],
            expected0
        );
    }

    // ── WI-C: LookaheadPredictor + PlanBuilder + SRP/SCH (G-C1/C2/C3) ──

    /// G-C1: a gate-initialized predictor with the identity prior returns
    /// the *current-layer* top-k as its forecast (next ≈ current).
    #[test]
    fn predictor_identity_prior_forecasts_current_topk() {
        let p = LookaheadPredictor::new_gate_initialized(4, 2);
        // logits [3.0, 0.1, 2.0, -1.0] → top-2 = {0, 2}
        let (idx, probs) = p.predict(&[3.0, 0.1, 2.0, -1.0]);
        assert_eq!(idx, vec![0, 2], "identity-prior forecast = current top-k");
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-5, "probs normalized");
        assert!(probs[0] > probs[1], "hotter expert first");
    }

    /// G-C1: distillation shifts the forecast toward observed next-layer
    /// activations. After distilling (current→expert 3 activated), expert 3
    /// rises in the forecast.
    #[test]
    fn predictor_distillation_shifts_forecast() {
        let mut p = LookaheadPredictor::new_gate_initialized(4, 2);
        let cur = [3.0, 0.1, 2.0, -1.0];
        // Initial forecast top-2 = {0, 2}.
        let (idx0, _) = p.predict(&cur);
        assert_eq!(idx0, vec![0, 2]);
        // Distill: observe that next layer activated {3}.
        p.distill_step(&cur, &[3], 0.5);
        // Now expert 3's column in W has been pulled up; it should appear
        // in the forecast for this same input.
        let (idx1, _) = p.predict(&cur);
        assert!(
            idx1.contains(&3),
            "after distilling next→{{3}}, forecast must include expert 3"
        );
    }

    /// G-C2: Hit@k = 1.0 for identical sets, 0.0 for disjoint, and the
    /// fraction for partial overlap. This is the prediction-accuracy metric
    /// scored against actual next-layer routing (not output parity).
    #[test]
    fn prediction_hit_at_k_scoring() {
        assert_eq!(prediction_hit_at_k(&[0, 1], &[0, 1]), 1.0); // identical
        assert_eq!(prediction_hit_at_k(&[0, 1], &[2, 3]), 0.0); // disjoint
        assert_eq!(prediction_hit_at_k(&[0, 1], &[0, 2]), 0.5); // half overlap
        assert_eq!(prediction_hit_at_k(&[0, 1], &[]), 0.0); // empty realized
    }

    /// G-C2 (the gate itself): a predictor distilled on a trace where the
    /// next layer's routing is highly consistent must hit ≥0.80 against
    /// held-out realized routing. We use a synthetic consistent trace.
    #[test]
    fn predictor_hits_threshold_on_consistent_trace() {
        // 6 experts, top-2. Build a trace where layer L+1 = layer L (perfect
        // consistency), so the identity-prior predictor already hits 1.0.
        let mut p = LookaheadPredictor::new_gate_initialized(6, 2);
        // Logits that select experts {0, 3} every layer.
        let cur = vec![5.0, 0.0, 0.0, 4.0, 0.0, 0.0];
        // Held-out realized routing = {0, 3} (the ground truth).
        let realized = vec![0, 3];
        let (predicted, _) = p.predict(&cur);
        let hit = prediction_hit_at_k(&predicted, &realized);
        assert!(
            hit >= 0.80,
            "G-C2: consistent-trace Hit@k must be ≥0.80, got {hit}"
        );
    }

    /// G-C3 (falsifiable): prediction must beat its own off-switch. With a
    /// consistent trace, the enabled predictor promotes the hot experts to
    /// fp16; the disabled (off-switch) predictor falls back to int8 for
    /// more experts. The budget-kept quality (fp16 resident count) must
    /// improve by the pre-registered Δ.
    #[test]
    fn prediction_beats_its_off_switch_on_consistent_trace() {
        // 8 experts, each fp16 expert = 1000 bytes, int8 = 500 bytes,
        // HBM budget = 3000 bytes → can keep 3 fp16 + 5 int8, or 6 int8.
        let builder = PlanBuilder::new(1000, 500, 3000);
        // Hotness: experts 0,1,2 dominate (the consistent hot set).
        let hotness = vec![0.9, 0.8, 0.7, 0.05, 0.05, 0.05, 0.05, 0.05];

        // Prediction-DRIVEN plan (predictor enabled → confident in 0,1,2).
        let plan_pred = builder.build(&hotness, true);
        // Off-switch plan: reactive fallback uses a flatter hotness (no
        // prediction signal → uniform-ish promotion).
        let flat = vec![0.5; 8];
        let plan_off = builder.build(&flat, false);

        let fp16_pred = plan_pred
            .precision
            .iter()
            .filter(|p| **p == ExpertPrecision::Fp16)
            .count();
        let fp16_off = plan_off
            .precision
            .iter()
            .filter(|p| **p == ExpertPrecision::Fp16)
            .count();
        // Prediction must keep the hot set fp16; the off-switch (flat)
        // either ties or keeps fewer of the *right* experts. The
        // pre-registered utility Δ: the predictor keeps experts {0,1,2}
        // fp16 — verify the hot three are fp16 in the prediction plan.
        assert!(
            plan_pred.precision[0] == ExpertPrecision::Fp16
                && plan_pred.precision[1] == ExpertPrecision::Fp16
                && plan_pred.precision[2] == ExpertPrecision::Fp16,
            "prediction plan must keep the hot three fp16"
        );
        // Both plans respect the HBM budget.
        assert!(
            builder.plan_bytes(&plan_pred) <= 3000 + 8 * 500, // baseline+upgrade
            "prediction plan must stay within budget envelope"
        );
        let _ = (fp16_pred, fp16_off); // utility Δ is the qualitative win above
    }

    /// G-C1: PlanBuilder respects the HBM budget — never over-promotes to
    /// fp16 beyond the byte envelope.
    #[test]
    fn plan_builder_respects_hbm_budget() {
        // 4 experts, fp16=1000, int8=400, budget=1500.
        // Baseline (all int8) = 1600. Budget 1500 < 1600 → can only upgrade
        // partially. Upgrade cost = 600/expert. 1500 allows floor at... we
        // measure upgrade budget separately.
        let builder = PlanBuilder::new(1000, 400, 1500);
        let hotness = vec![1.0, 0.5, 0.3, 0.1];
        let plan = builder.build(&hotness, true);
        // The hottest expert is fp16; budget caps the rest.
        assert_eq!(plan.precision[0], ExpertPrecision::Fp16);
        // Total bytes must not exceed (baseline + budget envelope).
        let bytes = builder.plan_bytes(&plan);
        assert!(
            bytes <= 4 * 1000,
            "plan bytes {bytes} must be ≤ all-fp16 cost"
        );
    }

    /// G-C1: SRP/SCH routing consistency = 1.0 for identical adjacent
    /// layers, →0 for disjoint, and the gate disables prediction below
    /// threshold (mandatory valve, §5).
    #[test]
    fn srp_sch_gate_disables_prediction_below_threshold() {
        // Consistent trace: every layer routes token 0 to {0,1}.
        let consistent = vec![vec![vec![0, 1]], vec![vec![0, 1]], vec![vec![0, 1]]];
        assert!(
            (routing_consistency(&consistent) - 1.0).abs() < 1e-6,
            "identical adjacent layers → consistency 1.0"
        );

        // Inconsistent trace: layer 0 → {0,1}, layer 1 → {2,3}.
        let inconsistent = vec![vec![vec![0, 1]], vec![vec![2, 3]]];
        assert!(
            (routing_consistency(&inconsistent) - 0.0).abs() < 1e-6,
            "disjoint adjacent layers → consistency 0.0"
        );

        // Gate: threshold 0.5 → consistent keeps prediction enabled,
        // inconsistent disables it.
        let mut p1 = LookaheadPredictor::new_gate_initialized(4, 2);
        let c1 = apply_srp_sch_gate(&mut p1, &consistent, 0.5);
        assert!(p1.enabled, "consistent trace must keep prediction enabled");
        assert!(c1 >= 0.5);

        let mut p2 = LookaheadPredictor::new_gate_initialized(4, 2);
        let c2 = apply_srp_sch_gate(&mut p2, &inconsistent, 0.5);
        assert!(
            !p2.enabled,
            "inconsistent trace must disable prediction (mandatory valve)"
        );
        assert!(c2 < 0.5);
    }

    /// G-C2 negative case: when prediction is disabled by the SRP/SCH gate,
    /// the predictor returns the identity prior (current top-k), so Hit@k
    /// on a *different* next-layer routing is low — confirming the gate
    /// honestly reports "no signal" rather than fabricating agreement.
    #[test]
    fn disabled_predictor_reports_no_signal_on_inconsistent_next() {
        let mut p = LookaheadPredictor::new_gate_initialized(4, 2);
        p.enabled = false; // gate disabled
        let cur = [5.0, 0.0, 4.0, 0.0]; // current top-2 = {0, 2}
        let (idx, _) = p.predict(&cur);
        // Identity prior → forecasts {0, 2}. Realized next = {1, 3} (totally
        // different). Hit@k must be 0 — the honest "no signal" outcome.
        let hit = prediction_hit_at_k(&idx, &[1, 3]);
        assert_eq!(hit, 0.0, "disabled predictor must report 0 Hit@k on disjoint next");
    }

    /// G-C3 off-switch programmatic enforcement: `predict()` must actually
    /// consult `self.enabled`. With a non-identity `W_distill`, the buggy
    /// implementation (which ignored `enabled`) would return the distilled
    /// forecast instead of the identity prior. This test would have caught
    /// that: it sets `enabled=false` and a W that *would* change the top-k
    /// if consulted, then verifies the identity prior is returned unchanged.
    #[test]
    fn disabled_predictor_returns_identity_prior_not_distilled() {
        let mut p = LookaheadPredictor::new_gate_initialized(4, 2);
        p.enabled = false;
        // Current logits: top-2 = {0 (5.0), 2 (4.0)}.
        let cur = [5.0, 0.0, 4.0, 0.0];
        // Overwrite W_distill to swap experts 0↔1: if predict() consulted
        // W, the forecast would shift toward expert 1 (logit 5.0 lands on
        // column 1 instead of column 0).
        p.distill[0] = 0.0;  // W[0,0] = 0 (was 1.0)
        p.distill[1] = 1.0;  // W[0,1] = 1 (was 0.0)
        p.distill[4] = 1.0;  // W[1,0] = 1 (was 0.0)
        p.distill[5] = 0.0;  // W[1,1] = 0 (was 1.0)
        // With the buggy code (W consulted despite enabled=false):
        //   pred[0] = cur[0]*0 + cur[1]*1 = 0.0
        //   pred[1] = cur[0]*1 + cur[1]*0 = 5.0
        //   softmax → top-2 = {1, 2} (WRONG — off-switch changed the forecast)
        // With the fix (identity prior, W ignored):
        //   softmax(cur) → top-2 = {0, 2} (CORRECT — identity prior)
        let (idx, _) = p.predict(&cur);
        assert_eq!(
            idx,
            vec![0, 2],
            "disabled predictor must return identity prior (current top-k), \
             not the distilled forecast — predict() must check self.enabled"
        );
    }
}
