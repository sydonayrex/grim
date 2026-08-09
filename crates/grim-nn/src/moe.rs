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

    /// Load experts from a GGUF-style 3D weight layout
    /// (`ffn_gate_exps` = `[n_experts, inter, hidden]`, etc.). Slices each
    /// expert out of the 3D tensor into a 2D `Linear`.
    pub fn load(
        ws: &WeightSource<'_>,
        num_experts: usize,
        hidden: usize,
        inter: usize,
        has_bias: bool,
    ) -> Result<Self, grim_tensor::error::Error> {
        let gate_3d = ws.get(Shape::new(vec![num_experts, inter, hidden]), "ffn_gate_exps")?;
        let up_3d = ws.get(Shape::new(vec![num_experts, inter, hidden]), "ffn_up_exps")?;
        let down_3d = ws.get(Shape::new(vec![num_experts, hidden, inter]), "ffn_down_exps")?;

        let gate_v = gate_3d.to_vec_f32()?;
        let up_v = up_3d.to_vec_f32()?;
        let down_v = down_3d.to_vec_f32()?;

        let mut gate = Vec::with_capacity(num_experts);
        let mut up = Vec::with_capacity(num_experts);
        let mut down = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let g = slice_expert(&gate_v, e, inter, hidden);
            let u = slice_expert(&up_v, e, inter, hidden);
            let d = slice_expert(&down_v, e, hidden, inter);
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
            for (rank, &e) in experts.iter().enumerate() {
                let y = self.experts.expert_forward(e, &xt)?; // [1, hidden]
                let yv = y.to_vec_f32()?;
                for (i, v) in yv.iter().enumerate() {
                    out_vec[t * hidden + i] += w[rank] * v;
                }
            }
            if let Some(sh) = &self.shared_expert {
                let s = sh.forward(&xt)?;
                let sv = s.to_vec_f32()?;
                for (i, v) in sv.iter().enumerate() {
                    out_vec[t * hidden + i] += self.routed_scaling_factor * v;
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
        MoeFfn::new(router, bank, shared, 1.0)
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
}
