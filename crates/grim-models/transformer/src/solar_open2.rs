//! Solar-Open2-250B hybrid-attention MoE model architecture implementation.
//!
//! Features 48 layers interleaving 1 GQA (softmax attention) layer with 3 KDA
//! (linear attention) layers. MoE block utilizes 320 routed experts + 1 shared expert.

use crate::{DeltaNetBase, DeltaNetBaseConfig};
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint};
use grim_core::session::{Inner, SessionT};
use grim_core::{Model, ModelConfig};
use grim_nn::{Embedding, Linear, RmsNorm, TensorParallelConfig, WeightSource};
use grim_tensor::{Device, Shape, Tensor};

use crate::block::LlamaLayerCache;
use crate::moe_block::MoeBlock;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolarLayerType {
    Gqa,
    Kda,
}

#[derive(Debug, Clone)]
pub struct SolarOpen2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub num_routed_experts: usize,
    pub num_shared_experts: usize,
    pub top_k: usize,
    pub max_seq_len: usize,
}

impl Default for SolarOpen2Config {
    fn default() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 4096,
            num_heads: 64,
            num_kv_heads: 8,
            head_dim: 128,
            num_layers: 48,
            intermediate_size: 11008,
            rms_norm_eps: 1e-5,
            num_routed_experts: 320,
            num_shared_experts: 1,
            top_k: 8,
            max_seq_len: 131072,
        }
    }
}

impl SolarOpen2Config {
    pub fn from_hf(value: &serde_json::Value) -> Self {
        let u = |k: &str, def: usize| {
            value
                .get(k)
                .and_then(|v| v.as_u64())
                .map(|x| x as usize)
                .unwrap_or(def)
        };
        let f = |k: &str, def: f32| {
            value
                .get(k)
                .and_then(|v| v.as_f64())
                .map(|x| x as f32)
                .unwrap_or(def)
        };

        let hidden_size = u("hidden_size", 4096);
        let num_heads = u("num_attention_heads", 64);
        let raw_head_dim = u("head_dim", 0);
        let head_dim = if raw_head_dim > 0 {
            raw_head_dim
        } else {
            hidden_size.checked_div(num_heads).unwrap_or(128)
        };

        SolarOpen2Config {
            vocab_size: u("vocab_size", 151936),
            hidden_size,
            num_heads,
            num_kv_heads: u("num_key_value_heads", 8),
            head_dim,
            num_layers: u("num_hidden_layers", 48),
            intermediate_size: u("intermediate_size", 11008),
            rms_norm_eps: f("rms_norm_eps", 1e-5),
            num_routed_experts: u("num_routed_experts", 320),
            num_shared_experts: u("num_shared_experts", 1),
            top_k: u("num_experts_per_tok", 8),
            max_seq_len: u("max_position_embeddings", 131072),
        }
    }

    pub fn layer_type(&self, layer_idx: usize) -> SolarLayerType {
        if layer_idx % 4 == 0 {
            SolarLayerType::Gqa
        } else {
            SolarLayerType::Kda
        }
    }
}

impl ModelConfig for SolarOpen2Config {
    fn name(&self) -> &str {
        "solar_open2"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub struct SolarOpen2Block {
    pub layer_type: SolarLayerType,
    pub attn_norm: RmsNorm,
    pub llama_block: Option<crate::block::LlamaBlock>,
    pub delta_net: Option<DeltaNetBase>,
    pub ffn_norm: RmsNorm,
    pub moe: MoeBlock,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl SolarOpen2Block {
    pub fn load_tp(
        ws: &WeightSource<'_>,
        cfg: &SolarOpen2Config,
        layer_idx: usize,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        let layer_type = cfg.layer_type(layer_idx);
        let attn_norm = RmsNorm::load(&ws.pp("attn_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;

        let (llama_block, delta_net) = match layer_type {
            SolarLayerType::Gqa => {
                let llama_cfg = crate::model::LlamaConfig {
                    vocab_size: cfg.vocab_size,
                    hidden_size: cfg.hidden_size,
                    num_heads: cfg.num_heads,
                    num_kv_heads: cfg.num_kv_heads,
                    head_dim: cfg.head_dim,
                    num_layers: cfg.num_layers,
                    intermediate_size: cfg.intermediate_size,
                    rms_norm_eps: cfg.rms_norm_eps,
                    rope_theta: 10000.0,
                    max_seq_len: cfg.max_seq_len,
                    partial_rotary_factor: 1.0,
                    yarn: None,
                };
                let mut block = crate::block::LlamaBlock::load_tp(ws, &llama_cfg, tp)?;
                block.ffn_disabled = true;
                (Some(block), None)
            }
            SolarLayerType::Kda => {
                let dnet = DeltaNetBase::load(
                    ws.device(),
                    ws,
                    DeltaNetBaseConfig {
                        vocab_size: cfg.vocab_size,
                        hidden_size: cfg.hidden_size,
                        num_heads: cfg.num_heads,
                        head_dim: cfg.head_dim,
                        num_layers: cfg.num_layers,
                        intermediate_size: cfg.intermediate_size,
                        chunk_size: 64,
                        rms_norm_eps: 1e-5,
                        max_seq_len: cfg.max_seq_len,
                    },
                )?;
                (None, Some(dnet))
            }
        };

        let ffn_norm = RmsNorm::load(&ws.pp("ffn_norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let llama_cfg = crate::model::LlamaConfig {
            vocab_size: cfg.vocab_size,
            hidden_size: cfg.hidden_size,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            num_layers: cfg.num_layers,
            intermediate_size: cfg.intermediate_size,
            rms_norm_eps: cfg.rms_norm_eps,
            rope_theta: 10000.0,
            max_seq_len: cfg.max_seq_len,
            partial_rotary_factor: 1.0,
            yarn: None,
        };
        let spec = crate::moe_block::MoESpec {
            num_experts: cfg.num_routed_experts,
            top_k: cfg.top_k,
            router_kind: grim_nn::moe::RouterKind::SoftmaxTopK,
            routed_scaling_factor: 1.0,
            has_shared_expert: cfg.num_shared_experts > 0,
            moe_intermediate_size: Some(cfg.intermediate_size),
            shared_expert_intermediate_size: None,
            transposed_expert_layout: false,
        };
        let moe = MoeBlock::load(ws, &llama_cfg, &spec, tp)?;

        Ok(Self {
            layer_type,
            attn_norm,
            llama_block,
            delta_net,
            ffn_norm,
            moe,
            hidden_size: cfg.hidden_size,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        positions: &[u32],
        sess: Option<&mut dyn SessionT>,
        caches: Option<&mut [Option<LlamaLayerCache>]>,
        layer: usize,
    ) -> Result<Tensor> {
        let residual = x;
        let normed_attn = self.attn_norm.forward(x)?;

        let attn_out = match self.layer_type {
            SolarLayerType::Gqa => {
                let (out, _, _) = self.llama_block.as_ref().unwrap().forward_with_kv_paged(
                    &normed_attn,
                    positions,
                    sess,
                    caches.and_then(|c| c.get_mut(0).and_then(|x| x.as_mut())),
                    layer,
                )?;
                out
            }
            SolarLayerType::Kda => {
                let pos_tensor = grim_backend_cpu::cpu_tensor(
                    positions.iter().map(|&p| p as f32).collect(),
                    Shape::new(vec![positions.len()]),
                );
                let mut dummy_sess =
                    grim_core::session::Inner::new(self.delta_net.as_ref().unwrap().device.clone());
                self.delta_net.as_ref().unwrap().forward(
                    &mut dummy_sess,
                    &normed_attn,
                    &pos_tensor,
                    &[],
                )?
            }
        };

        let h = grim_nn::modules::add_on_device(residual, &attn_out)?;
        let normed_ffn = self.ffn_norm.forward(&h)?;
        let ffn_out = self.moe.forward(&normed_ffn)?;
        Ok(grim_nn::modules::add_on_device(&h, &ffn_out)?)
    }
}

pub struct SolarOpen2 {
    pub cfg: SolarOpen2Config,
    pub device: Device,
    pub tok_embeddings: Embedding,
    pub layers: Vec<SolarOpen2Block>,
    pub norm: RmsNorm,
    pub output: Linear,
}

impl SolarOpen2 {
    pub fn load_tp(ws: &WeightSource<'_>, cfg: SolarOpen2Config) -> Result<Self> {
        let tp = ws.tp_config();
        let tok_embeddings =
            Embedding::load(&ws.pp("embed_tokens"), cfg.vocab_size, cfg.hidden_size)?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for idx in 0..cfg.num_layers {
            let layer_ws = ws.pp(&format!("layers.{idx}"));
            layers.push(SolarOpen2Block::load_tp(&layer_ws, &cfg, idx, tp)?);
        }
        let norm = RmsNorm::load(&ws.pp("norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let output = Linear::load_column_parallel(
            &ws.pp("lm_head"),
            cfg.hidden_size,
            cfg.vocab_size,
            false,
            tp,
        )?;

        Ok(Self {
            cfg,
            device: ws.device().clone(),
            tok_embeddings,
            layers,
            norm,
            output,
        })
    }
}

impl Model for SolarOpen2 {
    fn config(&self) -> &dyn ModelConfig {
        &self.cfg
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn param_arith(&self) -> grim_tensor::ArithType {
        grim_tensor::ArithType::F32
    }
    fn as_any(&self) -> &(dyn std::any::Any + 'static) {
        self
    }
}

impl CausalLm for SolarOpen2 {
    fn forward(
        &self,
        session: &mut dyn SessionT,
        input_tokens: &Tensor,
        positions: &Tensor,
        _adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
        let pos_vec = positions.to_vec_f32()?;
        let pos_u32: Vec<u32> = pos_vec.iter().map(|&x| x as u32).collect();
        let seq_len = pos_u32.len();
        let ids = input_tokens
            .to_vec_f32()?
            .iter()
            .map(|&x| x as u32)
            .collect::<Vec<_>>();

        let mut h = self
            .tok_embeddings
            .forward(&ids, seq_len, self.cfg.hidden_size)?;

        if session.model_state().is_none() {
            let mut init_caches: Vec<Option<LlamaLayerCache>> =
                Vec::with_capacity(self.layers.len());
            for _ in 0..self.layers.len() {
                init_caches.push(None);
            }
            session.set_model_state(Box::new(init_caches));
        }
        let caches = session
            .model_state_mut()
            .and_then(|s| s.downcast_mut::<Vec<Option<LlamaLayerCache>>>())
            .expect(
                "SolarOpen2::forward: session.model_state must be Vec<Option<LlamaLayerCache>>",
            );

        for (idx, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &pos_u32, None, Some(&mut caches[idx..idx + 1]), idx)?;
        }
        let h_norm = self.norm.forward(&h)?;
        let logits = self.output.forward(&h_norm)?;
        session.advance_pos(seq_len);
        Ok(logits)
    }

    fn new_session(&self) -> Box<dyn SessionT> {
        Box::new(Inner::new(self.device.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_solar_open2_config() {
        let json = serde_json::json!({
            "vocab_size": 151936,
            "hidden_size": 4096,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "num_hidden_layers": 48,
            "num_routed_experts": 320,
            "num_experts_per_tok": 8
        });
        let cfg = SolarOpen2Config::from_hf(&json);
        assert_eq!(cfg.vocab_size, 151936);
        assert_eq!(cfg.num_layers, 48);
        assert_eq!(cfg.layer_type(0), SolarLayerType::Gqa);
        assert_eq!(cfg.layer_type(1), SolarLayerType::Kda);
        assert_eq!(cfg.layer_type(2), SolarLayerType::Kda);
        assert_eq!(cfg.layer_type(3), SolarLayerType::Kda);
        assert_eq!(cfg.layer_type(4), SolarLayerType::Gqa);
    }
}
