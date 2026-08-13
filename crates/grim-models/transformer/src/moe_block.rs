//! Shared Mixture-of-Experts block (WI-M2/M3).
//!
//! A `MoeBlock` replaces the dense SwiGLU FFN in a transformer layer: the
//! attention output is RMS-normalized, then routed through a `grim_nn::moe`
//! router + expert bank (+ optional shared expert). Every MoE architecture
//! (Qwen2/3-MoE, Laguna, GLM4, Granite-MoE, Phi, DBRX, OLMoE, BailingMoE,
//! Nemotron-hMoE, ...) funnels through this single implementation and
//! differs only by its `MoESpec` (expert counts, router kind, shared expert).

use grim_core::error::Result;
use grim_nn::moe::{ExpertBank, ExpertTriple, MoeFfn, MoeRouter, RouterKind};
use grim_nn::{
    Linear, RmsNorm, TensorParallelConfig, WeightSource,
};
use grim_tensor::{Shape, Tensor};

use crate::model::LlamaConfig;

/// Per-architecture routing configuration that distinguishes MoE families.
#[derive(Debug, Clone)]
pub struct MoESpec {
    /// Total number of experts (`expert_count`).
    pub num_experts: usize,
    /// Experts activated per token (`expert_used_count` / top-k).
    pub top_k: usize,
    /// Router scoring convention: softmax (Qwen/GLM/Granite/Phi/...) or
    /// sigmoid+bias (Laguna, DeepSeek-V2/V3 dedup gating).
    pub router_kind: RouterKind,
    /// Scaling applied to the (routed) expert output before adding the
    /// shared-expert contribution (`routed_scaling_factor`).
    pub routed_scaling_factor: f32,
    /// Whether this architecture carries an always-on shared expert
    /// (`ffn_gate_she` / `ffn_up_she` / `ffn_down_she`).
    pub has_shared_expert: bool,
    /// Per-routed-expert FFN width override (e.g., 1024 for Laguna-S-2.1).
    pub moe_intermediate_size: Option<usize>,
    /// Shared-expert FFN width override (e.g., 1024 for Laguna-S-2.1).
    pub shared_expert_intermediate_size: Option<usize>,
}

/// A single MoE transformer layer's feed-forward (routing) block.
pub struct MoeBlock {
    pub ffn_norm: RmsNorm,
    pub moe: MoeFfn,
    pub tp_config: TensorParallelConfig,
}

impl MoeBlock {
    /// Load a MoE block from a `WeightSource` positioned at the layer root
    /// (i.e. the caller has already done `ws.pp("layers").pp(&i.to_string())`).
    pub fn load(
        ws: &WeightSource<'_>,
        cfg: &LlamaConfig,
        spec: &MoESpec,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        let ffn_norm = RmsNorm::load(&ws.pp("ffn_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;

        // Router gate. llama.cpp stores the expert router as
        // `ffn_gate_inp.weight` = [hidden, num_experts] (consistent with the
        // in-repo Lfm2 MoE loader). `Linear::load` is TP-aware via `ws`'s
        // tensor-parallel config.
        let gate = Linear::load(
            &ws.pp("ffn_gate_inp"),
            cfg.hidden_size,
            spec.num_experts,
            /*has_bias=*/ false,
        )?;

        // Optional dedup/noisy-router correction bias (Laguna / DeepSeek).
        // Stored as `ffn_exp_probs_b.bias` in GGUF (per Lfm2 loader).
        let correction_bias = match &spec.router_kind {
            RouterKind::SigmoidTopKWithBias => {
                let b = ws.get(Shape::new(vec![spec.num_experts]), "ffn_exp_probs_b.bias")?;
                Some(b)
            }
            RouterKind::SoftmaxTopK => None,
        };

        let router = MoeRouter::new(
            gate,
            spec.router_kind.clone(),
            spec.top_k,
            spec.num_experts,
            correction_bias,
        );

        let moe_inter = spec.moe_intermediate_size.unwrap_or(cfg.intermediate_size);
        let shared_inter = spec.shared_expert_intermediate_size.unwrap_or(cfg.intermediate_size);

        // Per-expert SwiGLU triples from the 3D GGUF layout
        // (`ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps`).
        let experts = ExpertBank::load(
            ws,
            spec.num_experts,
            cfg.hidden_size,
            moe_inter,
            /*has_bias=*/ false,
        )?;

        // Optional always-on shared expert.
        let shared_expert: Option<ExpertTriple> = if spec.has_shared_expert {
            Some(ExpertTriple::load(
                ws,
                cfg.hidden_size,
                shared_inter,
                /*has_bias=*/ false,
            )?)
        } else {
            None
        };


        let moe = MoeFfn::new(router, experts, shared_expert, spec.routed_scaling_factor);

        Ok(Self {
            ffn_norm,
            moe,
            tp_config: tp,
        })
    }

    /// Forward: RMS-norm the attention output, then route through the MoE.
    /// `x` is `[batch, hidden]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let normed = self.ffn_norm.forward(x)?;
        let routed = self.moe.forward(&normed)?;
        Ok(routed)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use grim_backend_cpu::cpu_tensor;
    use grim_core::error::Result;
    use grim_nn::moe::RouterKind;
    use grim_nn::{TensorParallelConfig, WeightSource};
    use grim_tensor::dtype::{DType, Device, QuantProvenance};
    use grim_tensor::provider::{RawTensor, TensorMeta, TensorProvider};
    use grim_tensor::shape::Shape;
    use grim_tensor::Tensor;

    use crate::model::LlamaConfig;
    use crate::moe_block::{MoESpec, MoeBlock};

    /// In-memory `TensorProvider` so the MoE load path can be exercised
    /// without a real GGUF file (WI-M6). Mirrors the `FullProvider` used in
    /// `block.rs`'s load-path tests.
    #[derive(Clone)]
    struct FullProvider {
        tensors: HashMap<String, RawTensor>,
    }

    impl TensorProvider for FullProvider {
        fn get(&self, name: &str) -> grim_tensor::error::Result<RawTensor> {
            self.tensors.get(name).cloned().ok_or_else(|| {
                grim_tensor::error::Error::Backend(format!("tensor '{name}' not found"))
            })
        }
        fn meta(&self, _name: &str) -> grim_tensor::error::Result<TensorMeta> {
            Ok(TensorMeta {
                dtype: DType::F32,
                provenance: QuantProvenance::GrimNative,
                shape: vec![],
                fusion_mask: 0,
            })
        }
    }

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn cfg(hidden: usize, inter: usize, num_experts: usize) -> LlamaConfig {
        LlamaConfig {
            vocab_size: 100,
            hidden_size: hidden,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: hidden / 2,
            num_layers: 1,
            intermediate_size: inter,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 512,
        }
    }

    /// Build a synthetic MoE layer's weights and load a `MoeBlock` from them,
    /// then run a forward pass and assert the output shape + sanity.
    #[test]
    fn moe_block_load_and_forward() -> Result<()> {
        let hidden = 8usize;
        let inter = 8usize;
        let num_experts = 4usize;
        let top_k = 2usize;

        let mut tensors = HashMap::new();
        // RMS norm scale (all ones -> identity-ish norm).
        tensors.insert(
            "ffn_norm.weight".to_string(),
            RawTensor {
                bytes: f32_bytes(&vec![1.0f32; hidden]),
                shape: vec![hidden],
                dtype: DType::F32,
                provenance: QuantProvenance::GrimNative,
            },
        );
        // Router gate (`MoeBlock::load` queries `ffn_gate_inp.weight`, matching
        // the in-repo Lfm2 MoE loader). `Linear::load(hidden, num_experts)`
        // expects the stored weight in [out, in] = [num_experts, hidden].
        let gate_w: Vec<f32> = (0..num_experts * hidden)
            .map(|i| (i as f32 * 0.3 - 1.0))
            .collect();
        tensors.insert(
            "ffn_gate_inp.weight".to_string(),
            RawTensor {
                bytes: f32_bytes(&gate_w),
                shape: vec![num_experts, hidden],
                dtype: DType::F32,
                provenance: QuantProvenance::GrimNative,
            },
        );
        // 3D expert tensors [num_experts, inter, hidden] (experts outermost,
        // matching llama.cpp / Lfm2 GGUF convention).
        let exp_gate: Vec<f32> = (0..num_experts * inter * hidden)
            .map(|i| (i as f32 * 0.1 - 0.5))
            .collect();
        tensors.insert(
            "ffn_gate_exps.weight".to_string(),
            RawTensor {
                bytes: f32_bytes(&exp_gate),
                shape: vec![num_experts, inter, hidden],
                dtype: DType::F32,
                provenance: QuantProvenance::GrimNative,
            },
        );
        let exp_up = exp_gate.clone();
        tensors.insert(
            "ffn_up_exps.weight".to_string(),
            RawTensor {
                bytes: f32_bytes(&exp_up),
                shape: vec![num_experts, inter, hidden],
                dtype: DType::F32,
                provenance: QuantProvenance::GrimNative,
            },
        );
        let exp_down: Vec<f32> = (0..num_experts * inter * hidden)
            .map(|i| (i as f32 * 0.1 - 0.5))
            .collect();
        tensors.insert(
            "ffn_down_exps.weight".to_string(),
            RawTensor {
                bytes: f32_bytes(&exp_down),
                shape: vec![num_experts, inter, hidden],
                dtype: DType::F32,
                provenance: QuantProvenance::GrimNative,
            },
        );

        let provider = FullProvider { tensors };
        let ws = WeightSource::root(&provider, Device::Cpu);
        let tp = TensorParallelConfig {
            rank: 0,
            world_size: 1,
        };
        let spec = MoESpec {
            num_experts,
            top_k,
            router_kind: RouterKind::SoftmaxTopK,
            routed_scaling_factor: 1.0,
            has_shared_expert: false,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
        };


        let block = MoeBlock::load(&ws, &cfg(hidden, inter, num_experts), &spec, tp)?;

        let input = cpu_tensor(vec![0.5f32; hidden], Shape::new(vec![1, hidden]));
        let out = block.forward(&input)?;
        assert_eq!(out.shape().dims(), &[1usize, hidden], "MoE output shape");

        let out_v = out.to_vec_f32()?;
        assert_eq!(out_v.len(), hidden);
        assert!(
            out_v.iter().all(|x| x.is_finite()),
            "MoE output must be finite (no NaN/Inf)"
        );
        Ok(())
    }
}
