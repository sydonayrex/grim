//! Llama/Mistral-style dense transformer — `CausalLm` implementation.

use grim_backend_cpu::CpuDevice;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint};
use grim_core::session::{Inner, SessionT};
use grim_core::{Model, ModelConfig};
use grim_nn::{Embedding, Linear, RmsNorm, Rope};
use grim_tensor::{ArithType, Device, DType, Shape, Tensor};

use crate::block::{LlamaBlock, LlamaConfigRefs};
use crate::rng::SimpleRng;

#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_seq_len: usize,
}

impl ModelConfig for LlamaConfig {
    fn name(&self) -> &str {
        "llama"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub struct Llama {
    pub cfg: LlamaConfig,
    pub device: Device,
    pub tok_embeddings: Embedding,
    pub layers: Vec<LlamaBlock>,
    pub norm: RmsNorm,
    pub output: Linear,
    #[allow(dead_code)]
    rope: Rope,
}

impl Llama {
    pub fn load(ws: &grim_nn::WeightSource<'_>, cfg: LlamaConfig) -> Result<Self> {
        let tok_embeddings = Embedding::load(
            &ws.pp("tok_embeddings"),
            cfg.vocab_size,
            cfg.hidden_size,
        )?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(LlamaBlock::load(&ws.pp("layers").pp(&i.to_string()), &cfg)?);
        }
        let norm = RmsNorm::load(&ws.pp("norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let output = Linear::load(
            &ws.pp("output"),
            cfg.hidden_size,
            cfg.vocab_size,
            false,
        )?;
        let rope = Rope::new(cfg.head_dim, cfg.rope_theta);
        Ok(Self {
            cfg,
            device: Device::Cpu,
            tok_embeddings,
            layers,
            norm,
            output,
            rope,
        })
    }

    pub fn random(cfg: LlamaConfig) -> Self {
        use grim_backend_cpu::cpu_tensor;
        let dev = CpuDevice::new();
        let mut rng = SimpleRng::new(0xDEAD_BEEF_CAFE_F00Du64);

        let embed_data: Vec<f32> = (0..cfg.vocab_size * cfg.hidden_size)
            .map(|_| (rng.next_f32() - 0.5) * 0.02)
            .collect();
        let tok_embeddings = Embedding {
            weight: cpu_tensor(embed_data, Shape::new(vec![cfg.vocab_size, cfg.hidden_size])),
        };

        let mut linear = |out: usize, inp: usize| {
            let data: Vec<f32> = (0..out * inp)
                .map(|_| (rng.next_f32() - 0.5) * 0.02)
                .collect();
            Linear {
                weight: cpu_tensor(data, Shape::new(vec![out, inp])),
                bias: None,
            }
        };
        let rms = |dim: usize| RmsNorm {
            weight: cpu_tensor(vec![1.0; dim], Shape::new(vec![dim])),
            eps: cfg.rms_norm_eps,
        };

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for _ in 0..cfg.num_layers {
            layers.push(LlamaBlock {
                attn_norm: rms(cfg.hidden_size),
                wq: linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size),
                wk: linear(cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size),
                wv: linear(cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size),
                wo: linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim),
                ffn_norm: rms(cfg.hidden_size),
                w_gate: linear(cfg.intermediate_size, cfg.hidden_size),
                w_up:   linear(cfg.intermediate_size, cfg.hidden_size),
                w_down: linear(cfg.hidden_size, cfg.intermediate_size),
                _dev: dev.clone(),
                _cfg: LlamaConfigRefs {
                    hidden_size: cfg.hidden_size,
                    num_heads: cfg.num_heads,
                    num_kv_heads: cfg.num_kv_heads,
                    head_dim: cfg.head_dim,
                    intermediate_size: cfg.intermediate_size,
                },
            });
        }

        let norm = rms(cfg.hidden_size);
        let output = linear(cfg.vocab_size, cfg.hidden_size);
        let rope = Rope::new(cfg.head_dim, cfg.rope_theta);
        Self {
            cfg: cfg.clone(),
            device: Device::Cpu,
            tok_embeddings,
            layers,
            norm,
            output,
            rope,
        }
    }

    pub fn embed_token(&self, token: u32) -> Result<Tensor> {
        Ok(self.tok_embeddings.forward(&[token], 1, self.cfg.hidden_size)?)
    }

    pub fn decode(&self, hidden: &Tensor, _positions: &[u32]) -> Result<Tensor> {
        let mut h = hidden.clone();
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }
        let h = self.norm.forward(&h)?;
        let logits = self.output.forward(&h)?;
        Ok(logits)
    }
}

impl Model for Llama {
    fn config(&self) -> &dyn ModelConfig {
        &self.cfg
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn param_arith(&self) -> ArithType {
        ArithType::F32
    }
}

impl CausalLm for Llama {
    fn new_session(&self) -> Box<dyn SessionT> {
        Box::new(Inner::new(self.device.clone()))
    }

    fn forward(
        &self,
        session: &mut dyn SessionT,
        input_ids: &Tensor,
        _positions: &Tensor,
        adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
        let ids: Vec<u32> = match input_ids.dtype() {
            d if d == DType::F32 => {
                let v = input_ids.to_vec_f32()?;
                v.into_iter().map(|x| x as u32).collect()
            }
            _ => {
                return Err(grim_tensor::Error::Unimplemented(
                    "non-F32 input_ids not yet supported".into(),
                )
                .into());
            }
        };
        let seq_len = ids.len();
        let hidden: Vec<f32> = self
            .tok_embeddings
            .forward(&ids, seq_len, self.cfg.hidden_size)?
            .to_vec_f32()?;
        let hidden_t = grim_backend_cpu::cpu_tensor(
            hidden,
            Shape::new(vec![1, seq_len, self.cfg.hidden_size]),
        );
        let positions: Vec<u32> = (0..seq_len).map(|i| i as u32).collect();
        let logits = self.decode(&hidden_t, &positions)?;
        let logits = if adapters.is_empty() {
            logits
        } else {
            // §4.5: fuse every active adapter's (α·x·A·B) bias into the
            // output projection along the vocab dim. We apply it post-hoc
            // to the final logits — a structural placeholder for the
            // per-layer Punica-style fused matmul that ROCm fills in
            // phase 4. Until then the correct mathematical operation
            // (rank-r LoRA bias) still runs, just not fused with the
            // base matmul.
            crate::lora::apply_adapters_to_logits(&logits, adapters, self.cfg.hidden_size)?
        };
        session.advance_pos(seq_len);
        Ok(logits)
    }
}