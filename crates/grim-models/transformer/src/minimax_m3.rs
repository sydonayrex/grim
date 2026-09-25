//! Compatibility loader and native implementation for `MiniMaxAI/MiniMax-M3`.
//! # Architecture Details - **Block Sparse MoE**: Top-4 routing across 32 sparse experts using softmax.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_nn::{Linear, RmsNorm, Rope, WeightSource};
use grim_tensor::{ArithType, Device, Shape, Tensor};

// Config

/// Configuration for MiniMax-M3 architecture.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MiniMaxM3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
}

impl Default for MiniMaxM3Config {
    fn default() -> Self {
        Self {
            vocab_size: 128000,
            hidden_size: 3072,
            num_attention_heads: 24,
            num_key_value_heads: 8,
            head_dim: 128,
            num_hidden_layers: 36,
            intermediate_size: 8192,
            num_experts: 32,
            num_experts_per_tok: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 100000.0,
            max_position_embeddings: 32768,
        }
    }
}

impl ModelConfig for MiniMaxM3Config {
    fn name(&self) -> &str {
        "minimax_m3"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl MiniMaxM3Config {
    pub fn from_hf(value: &serde_json::Value) -> Self {
        let u = |k: &str| value.get(k).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let f = |k: &str| value.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        MiniMaxM3Config {
            vocab_size: u("vocab_size"),
            hidden_size: u("hidden_size"),
            num_attention_heads: u("num_attention_heads"),
            num_key_value_heads: u("num_key_value_heads"),
            head_dim: u("head_dim"),
            num_hidden_layers: u("num_hidden_layers"),
            intermediate_size: u("intermediate_size"),
            num_experts: u("num_experts"),
            num_experts_per_tok: u("num_experts_per_tok"),
            rms_norm_eps: f("rms_norm_eps"),
            rope_theta: f("rope_theta"),
            max_position_embeddings: u("max_position_embeddings"),
        }
    }
}

// Block Sparse MoE

pub struct MiniMaxM3Expert {
    pub w1: Linear,
    pub w3: Linear,
    pub w2: Linear,
}

impl MiniMaxM3Expert {
    pub fn load(
        ws: &WeightSource<'_>,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Self> {
        let w1 = Linear::load_shape(&ws.scoped("w1"), [hidden_size, intermediate_size])?;
        let w3 = Linear::load_shape(&ws.scoped("w3"), [hidden_size, intermediate_size])?;
        let w2 = Linear::load_shape(&ws.scoped("w2"), [intermediate_size, hidden_size])?;
        Ok(Self { w1, w3, w2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.w1.forward(x)?;
        let up = self.w3.forward(x)?;
        let swiglu = grim_nn::modules::silu_mul_on_device(&gate, &up)
            .map_err(grim_core::error::Error::from)?;
        Ok(self.w2.forward(&swiglu)?)
    }
}

pub struct MiniMaxM3BlockSparseMoe {
    pub gate: Linear,
    pub experts: Vec<MiniMaxM3Expert>,
    pub num_experts_per_tok: usize,
    /// Device-resident routing scratch + resident stacked expert weights for
    /// the D2D Charon dispatch (WI-gpu-native-moe Phase 1). Built once,
    /// reused every decode step (no per-step H2D/D2H traffic).
    pub charon_cache: crate::shared_moe::CharonCache,
}

impl MiniMaxM3BlockSparseMoe {
    pub fn load(ws: &WeightSource<'_>, cfg: &MiniMaxM3Config) -> Result<Self> {
        let gate = Linear::load_shape(&ws.scoped("gate"), [cfg.hidden_size, cfg.num_experts])?;

        let mut experts = Vec::with_capacity(cfg.num_experts);
        let exp_ws = ws.scoped("experts");
        for e in 0..cfg.num_experts {
            let exp = MiniMaxM3Expert::load(
                &exp_ws.scoped(&e.to_string()),
                cfg.hidden_size,
                cfg.intermediate_size,
            )?;
            experts.push(exp);
        }

        Ok(Self {
            gate,
            experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            charon_cache: crate::shared_moe::CharonCache::new(),
        })
    }

    /// Tri-modal forward (DeepSeek2 template, WI-gpu-native-moe Phase 1):
    /// D2D device dispatch first (no gate-logits round-trip), host path as
    /// the documented reference fallback (CPU device or non-ROCm backend).
    /// The host path math is unchanged — it is the parity reference.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if matches!(x.device(), Device::Rocm(_)) {
            let logits = self.gate.forward(x)?;
            if let Some(out) = self.forward_moe_device_d2d(x, &logits)? {
                return Ok(out);
            }
        }
        self.forward_moe_host(x)
    }

    /// Device-resident MoE (D2D): routing computed on-device from the gate
    /// logits (route_mode 3 = softmax renormalized over top-k, matching the
    /// host loop's `normalize_weights` semantics) and expert evaluation
    /// launched from device-resident routing buffers. Returns `Ok(None)`
    /// when the D2D path is unavailable; caller falls back to host routing.
    fn forward_moe_device_d2d(&self, x: &Tensor, logits: &Tensor) -> Result<Option<Tensor>> {
        let dev = grim_nn::modules::pick_device_for_storage_device(x.device());
        let experts: Vec<crate::shared_moe::MoeExpert> = self
            .experts
            .iter()
            .map(|e| crate::shared_moe::MoeExpert {
                gate: e.w1.clone(),
                up: e.w3.clone(),
                down: e.w2.clone(),
            })
            .collect();
        crate::shared_moe::fused_moe_dispatch_from_logits(
            dev.as_ref(),
            x,
            logits,
            &experts,
            None,
            self.num_experts_per_tok,
            1.0,
            3, // route_mode: renorm over top-k (matches host loop below)
            &self.charon_cache,
        )
    }

    /// Host routing reference path — the documented FALLBACK (CPU device, or
    /// GPU backends missing a needed primitive). Identical math to the device
    /// path: per-token top-k routing on the gate logits with renorm-over-
    /// top-k combine weights, expert forward, weighted accumulation.
    /// Host routing stays by design on this path: gate logits
    /// (steps×n_expert) are tiny and top-k selection is host logic.
    /// The per-token input rows are pulled once (hoisted out of the loop - was re-downloading.
    pub fn forward_moe_host(&self, x: &Tensor) -> Result<Tensor> {
        let seq_len = x.shape().dims()[0];
        let hidden_dim = x.shape().dims()[1];
        let logits = self.gate.forward(x)?;
        let logits_v = logits.to_vec_f32()?;
        let num_exp = self.experts.len();

        let xv = x.to_vec_f32()?;
        let mut out = vec![0.0f32; seq_len * hidden_dim];

        for s in 0..seq_len {
            let row = &logits_v[s * num_exp..(s + 1) * num_exp];
            let mut indexed: Vec<(usize, f32)> = row.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let topk = &indexed[..self.num_experts_per_tok.min(num_exp)];

            let max_l = topk
                .iter()
                .map(|(_, l)| *l)
                .fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = topk.iter().map(|(_, l)| (l - max_l).exp()).collect();
            let sum_e: f32 = exps.iter().sum();
            let weights: Vec<f32> = exps.iter().map(|e| e / (sum_e + 1e-12)).collect();

            let token_x = cpu_tensor(
                xv[s * hidden_dim..(s + 1) * hidden_dim].to_vec(),
                Shape::new(vec![1, hidden_dim]),
            );

            for (i, (exp_idx, _)) in topk.iter().enumerate() {
                let w = weights[i];
                let exp_out = self.experts[*exp_idx].forward(&token_x)?.to_vec_f32()?;
                for d in 0..hidden_dim {
                    out[s * hidden_dim + d] += w * exp_out[d];
                }
            }
        }

        Ok(cpu_tensor(out, x.shape().clone()))
    }
}

// Block

pub struct MiniMaxM3Block {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
    pub block_sparse_moe: MiniMaxM3BlockSparseMoe,
    pub rope: Rope,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl MiniMaxM3Block {
    pub fn load(ws: &WeightSource<'_>, cfg: &MiniMaxM3Config) -> Result<Self> {
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        let attn_ws = ws.scoped("self_attn");
        let wq = Linear::load_shape(&attn_ws.scoped("q_proj"), [cfg.hidden_size, q_dim])?;
        let wk = Linear::load_shape(&attn_ws.scoped("k_proj"), [cfg.hidden_size, kv_dim])?;
        let wv = Linear::load_shape(&attn_ws.scoped("v_proj"), [cfg.hidden_size, kv_dim])?;
        let wo = Linear::load_shape(&attn_ws.scoped("o_proj"), [q_dim, cfg.hidden_size])?;

        let input_layernorm = RmsNorm::load(
            &ws.scoped("input_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;
        let post_attention_layernorm = RmsNorm::load(
            &ws.scoped("post_attention_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;

        let block_sparse_moe = MiniMaxM3BlockSparseMoe::load(&ws.scoped("block_sparse_moe"), cfg)?;
        let rope = Rope::new(cfg.head_dim, cfg.rope_theta);

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            input_layernorm,
            post_attention_layernorm,
            block_sparse_moe,
            rope,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        })
    }

    /// GPU-first forward: Q/K RoPE, KV-cache concat, attention and the residual adds run on the tensor's device.
    /// Host paths are only reached through the fused-kernel fallback guards and the (host-side) MoE routing.
    pub fn forward(
        &self,
        x: &Tensor,
        positions: &[u32],
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let seq_len = x.shape().dims()[0];
        let normed_attn = self.input_layernorm.forward(x)?;

        let q = self.wq.forward(&normed_attn)?;
        let k = self.wk.forward(&normed_attn)?;
        let v = self.wv.forward(&normed_attn)?;

        let q =
            crate::shared_attention::rope_2d_on_device(&self.rope, &q, self.num_heads, positions)?;
        let k = crate::shared_attention::rope_2d_on_device(
            &self.rope,
            &k,
            self.num_kv_heads,
            positions,
        )?;

        // Device-side history: prev rows stay resident, only the new rows
        // are appended (D2D arena copy when the backend supports it).
        let (k_all, v_all) = if let Some((prev_k, prev_v)) = kv_cache {
            let full_k = crate::shared_attention::concat_rows_on_device(prev_k, &k)?;
            let full_v = crate::shared_attention::concat_rows_on_device(prev_v, &v)?;
            *kv_cache = Some((full_k.clone(), full_v.clone()));
            (full_k, full_v)
        } else {
            *kv_cache = Some((k.clone(), v.clone()));
            (k.clone(), v.clone())
        };
        let kv_len = k_all.shape().dims()[0];

        // Shared helper applies the causal mask at cache_offset + s (fixes
        // future-token leakage during multi-token prefill).
        let attn_tensor = crate::shared_attention::fused_attention_tensors(
            &q,
            &k_all,
            &v_all,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            seq_len,
            kv_len,
            None,
        )?;
        let attn_proj = self.wo.forward(&attn_tensor)?;

        let res1 = grim_nn::modules::add_on_device(x, &attn_proj)?;
        let normed_ffn = self.post_attention_layernorm.forward(&res1)?;
        let mlp_out = self.block_sparse_moe.forward(&normed_ffn)?;
        // Routing stays host-side, so the MoE output lands on the host; stage
        // it back next to `res1` before the residual add.
        let mlp_out = grim_nn::modules::move_to_device(&mlp_out, x.device())?;

        grim_nn::modules::add_on_device(&res1, &mlp_out).map_err(grim_core::error::Error::from)
    }
}

// Model & Session

pub struct MiniMaxM3 {
    pub cfg: MiniMaxM3Config,
    pub device: Device,
    pub tok_embeddings: Linear,
    pub layers: Vec<MiniMaxM3Block>,
    pub norm: RmsNorm,
    pub output: Linear,
}

impl MiniMaxM3 {
    pub fn load(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: MiniMaxM3Config,
    ) -> Result<Self> {
        Self::load_tp(device, ws, cfg)
    }

    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: MiniMaxM3Config,
    ) -> Result<Self> {
        let root = ws.scoped("model");

        let tok_embeddings = Linear::load_shape(
            &root.scoped("embed_tokens"),
            [cfg.vocab_size, cfg.hidden_size],
        )?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let layer_ws = root.scoped("layers").scoped(&i.to_string());
            let block = MiniMaxM3Block::load(&layer_ws, &cfg)?;
            layers.push(block);
        }

        let norm = RmsNorm::load(&root.scoped("norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let output = Linear::load_shape(&ws.scoped("lm_head"), [cfg.hidden_size, cfg.vocab_size])
            .unwrap_or_else(|_| tok_embeddings.clone());

        Ok(Self {
            cfg,
            device,
            tok_embeddings,
            layers,
            norm,
            output,
        })
    }
}

impl Model for MiniMaxM3 {
    fn config(&self) -> &dyn ModelConfig {
        &self.cfg
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn param_arith(&self) -> ArithType {
        ArithType::F32
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl CausalLm for MiniMaxM3 {
    fn new_session(&self) -> Box<dyn SessionT> {
        Box::new(grim_core::session::Session::new(self.device.clone()))
    }

    fn forward(
        &self,
        session: &mut dyn SessionT,
        input_ids: &Tensor,
        positions: &Tensor,
        _adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
        let ids_f32 = input_ids.to_vec_f32()?;
        let seq_len = ids_f32.len();
        let ids: Vec<u32> = ids_f32.iter().map(|&t| t as u32).collect();
        let pos_v: Vec<u32> = positions
            .to_vec_f32()
            .map(|v| v.into_iter().map(|p| p as u32).collect())
            .unwrap_or_else(|_| (0..seq_len as u32).collect());

        // GPU-first embedding gather: rows land on the weight's device; the
        // vocab×hidden table never crosses to host.
        let mut x = grim_nn::embedding_gather_on_device(
            &self.tok_embeddings.weight,
            &ids,
            seq_len,
            self.cfg.hidden_size,
        )?;

        if session.model_state().is_none() {
            let fresh: Vec<Option<(Tensor, Tensor)>> = vec![None; self.layers.len()];
            session.set_model_state(Box::new(fresh));
        }

        let kv_caches = session
            .model_state_mut()
            .and_then(|s| s.downcast_mut::<Vec<Option<(Tensor, Tensor)>>>())
            .ok_or_else(|| {
                grim_core::error::Error::Backend(
                    "MiniMaxM3::forward: model_state must be Vec<Option<(Tensor, Tensor)>>".into(),
                )
            })?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, &pos_v, &mut kv_caches[layer_idx])?;
        }

        let normed = self.norm.forward(&x)?;
        let logits = self.output.forward(&normed)?;
        session.advance_pos(seq_len);
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_backend_cpu::cpu_tensor;
    use grim_core::architecture::ModelArchitecture;

    const MINIMAX_M3_CONFIG: &str = r#"{
        "architectures": ["MiniMaxM3ForCausalLM"],
        "hidden_size": 3072,
        "num_hidden_layers": 36,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 8192,
        "num_experts": 32,
        "num_experts_per_tok": 4,
        "rms_norm_eps": 1e-05,
        "rope_theta": 100000.0,
        "vocab_size": 128000
    }"#;

    #[test]
    fn parses_minimax_m3_config() {
        let v: serde_json::Value = serde_json::from_str(MINIMAX_M3_CONFIG).unwrap();
        let cfg = MiniMaxM3Config::from_hf(&v);
        assert_eq!(cfg.hidden_size, 3072);
        assert_eq!(cfg.num_hidden_layers, 36);
        assert_eq!(cfg.num_experts, 32);
        assert_eq!(cfg.name(), "minimax_m3");
    }

    #[test]
    fn dispatches_minimax_m3_architecture() {
        assert_eq!(
            ModelArchitecture::from_str("minimax_m3"),
            ModelArchitecture::MiniMaxM3
        );
    }

    fn det_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.4
            })
            .collect()
    }

    type SyntheticMoe = (
        MiniMaxM3BlockSparseMoe,
        Vec<Vec<f32>>,
        Vec<Vec<f32>>,
        Vec<Vec<f32>>,
        Vec<f32>,
    );

    fn synthetic_moe(
        hidden: usize,
        inter: usize,
        num_experts: usize,
        top_k: usize,
    ) -> SyntheticMoe {
        let gate_w = det_vec(num_experts * hidden, 11);
        let gate = Linear::from_tensor(
            cpu_tensor(gate_w.clone(), Shape::new(vec![num_experts, hidden])),
            None,
        );
        let mut experts = Vec::with_capacity(num_experts);
        let mut w1_all = Vec::with_capacity(num_experts);
        let mut w3_all = Vec::with_capacity(num_experts);
        let mut w2_all = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let w1 = det_vec(inter * hidden, 100 + e as u64 * 3);
            let w3 = det_vec(inter * hidden, 200 + e as u64 * 3);
            let w2 = det_vec(hidden * inter, 300 + e as u64 * 3);
            experts.push(MiniMaxM3Expert {
                w1: Linear::from_tensor(
                    cpu_tensor(w1.clone(), Shape::new(vec![inter, hidden])),
                    None,
                ),
                w3: Linear::from_tensor(
                    cpu_tensor(w3.clone(), Shape::new(vec![inter, hidden])),
                    None,
                ),
                w2: Linear::from_tensor(
                    cpu_tensor(w2.clone(), Shape::new(vec![hidden, inter])),
                    None,
                ),
            });
            w1_all.push(w1);
            w3_all.push(w3);
            w2_all.push(w2);
        }
        (
            MiniMaxM3BlockSparseMoe {
                gate,
                experts,
                num_experts_per_tok: top_k,
                charon_cache: crate::shared_moe::CharonCache::new(),
            },
            w1_all,
            w3_all,
            w2_all,
            gate_w,
        )
    }

    /// Independent renorm-over-top-k SwiGLU oracle (mirrors the host loop's
    /// documented semantics; written separately so the test is not a copy of
    /// the implementation).
    #[allow(clippy::too_many_arguments)]
    fn renorm_oracle(
        x_data: &[f32],
        gate_w: &[f32],
        w1_all: &[Vec<f32>],
        w3_all: &[Vec<f32>],
        w2_all: &[Vec<f32>],
        seq: usize,
        hidden: usize,
        inter: usize,
        num_experts: usize,
        top_k: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; seq * hidden];
        for s in 0..seq {
            let xt = &x_data[s * hidden..(s + 1) * hidden];
            // gate logits: [num_experts, hidden] @ xt
            let mut logits = vec![0.0f32; num_experts];
            for e in 0..num_experts {
                for c in 0..hidden {
                    logits[e] += gate_w[e * hidden + c] * xt[c];
                }
            }
            let mut idx: Vec<usize> = (0..num_experts).collect();
            idx.sort_by(|&a, &b| {
                logits[b]
                    .partial_cmp(&logits[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let k = top_k.min(num_experts);
            let max_l = idx[..k]
                .iter()
                .map(|&e| logits[e])
                .fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = idx[..k]
                .iter()
                .map(|&e| (logits[e] - max_l).exp())
                .collect();
            let sum: f32 = exps.iter().sum();
            for (rank, &e) in idx[..k].iter().enumerate() {
                let w = exps[rank] / (sum + 1e-12);
                let mut act = vec![0.0f32; inter];
                for j in 0..inter {
                    let mut g = 0.0f32;
                    let mut u = 0.0f32;
                    for c in 0..hidden {
                        g += w1_all[e][j * hidden + c] * xt[c];
                        u += w3_all[e][j * hidden + c] * xt[c];
                    }
                    act[j] = g / (1.0 + (-g).exp()) * u;
                }
                for h in 0..hidden {
                    let mut v = 0.0f32;
                    for j in 0..inter {
                        v += w2_all[e][h * inter + j] * act[j];
                    }
                    out[s * hidden + h] += w * v;
                }
            }
        }
        out
    }

    /// Unit (numeric): CPU `forward` (host reference path) matches the
    /// independent renorm oracle within 1e-5 and is deterministic.
    #[test]
    fn minimax_m3_host_matches_renorm_oracle() {
        let (hidden, inter, num_experts, top_k, seq) = (8, 16, 4, 2, 3);
        let (moe, w1_all, w3_all, w2_all, gate_w) =
            synthetic_moe(hidden, inter, num_experts, top_k);
        let x_data = det_vec(seq * hidden, 7);
        let x = cpu_tensor(x_data.clone(), Shape::new(vec![seq, hidden]));

        let got = moe.forward(&x).unwrap().to_vec_f32().unwrap();
        let want = renorm_oracle(
            &x_data,
            &gate_w,
            &w1_all,
            &w3_all,
            &w2_all,
            seq,
            hidden,
            inter,
            num_experts,
            top_k,
        );
        assert_eq!(got.len(), want.len());
        let max_diff = got
            .iter()
            .zip(want.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "minimax_m3 host vs oracle max diff {max_diff:.6} exceeds 1e-5"
        );

        // Determinism: second forward is bitwise identical.
        let again = moe.forward(&x).unwrap().to_vec_f32().unwrap();
        assert_eq!(got, again, "host forward must be deterministic");
        assert!(
            !moe.charon_cache.is_routing_engaged(),
            "CPU path must not engage device routing"
        );
    }

    /// Unit (edge): `top_k > num_experts` clamps without panic/NaN.
    #[test]
    fn minimax_m3_topk_clamp_edge() {
        let (hidden, inter, num_experts, seq) = (8, 16, 3, 2);
        let (moe, ..) = synthetic_moe(hidden, inter, num_experts, 8);
        let x = cpu_tensor(det_vec(seq * hidden, 77), Shape::new(vec![seq, hidden]));
        let out = moe.forward(&x).unwrap().to_vec_f32().unwrap();
        assert_eq!(out.len(), seq * hidden);
        assert!(out.iter().all(|v| v.is_finite()), "clamped output finite");
    }
}
