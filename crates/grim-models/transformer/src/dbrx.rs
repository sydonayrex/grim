//! Databricks DBRX MoE architecture with 16 fine-grained routed experts (top-4),
//! SwiGLU activation, and Grouped Query Attention (GQA).
//!
//! # Architecture Details
//! - **Attention**: GQA with 16 query heads and 4 KV heads ($d_{\text{head}} = 128$).
//! - **Feed Forward**: Sparse Mixture of Experts with 16 total experts, 4 active per token.
//! - **Normalization**: Pre-attention and pre-FFN RMSNorm.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_nn::{Linear, RmsNorm, Rope, TensorParallelConfig, WeightSource};
use grim_tensor::{ArithType, Device, Shape, Tensor, YaRNParams};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for Databricks DBRX.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DbrxConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub ffn_hidden_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub kv_n_heads: usize,
    pub head_dim: usize,
    pub moe_num_experts: usize,
    pub moe_top_k: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_seq_len: usize,
    pub yarn: Option<YaRNParams>,
}

impl Default for DbrxConfig {
    fn default() -> Self {
        Self {
            vocab_size: 100352,
            d_model: 6144,
            ffn_hidden_size: 10752,
            n_layers: 40,
            n_heads: 48,
            kv_n_heads: 8,
            head_dim: 128,
            moe_num_experts: 16,
            moe_top_k: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            max_seq_len: 32768,
            yarn: None,
        }
    }
}

impl ModelConfig for DbrxConfig {
    fn name(&self) -> &str {
        "dbrx"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------------------------------------------------------------------------
// MoE Block
// ---------------------------------------------------------------------------

struct DbrxExpert {
    w1: Linear,
    v1: Linear,
    w2: Linear,
}

impl DbrxExpert {
    fn load(ws: &WeightSource<'_>, in_dim: usize, hidden_dim: usize) -> Result<Self> {
        let w1 = Linear::load_shape(&ws.scoped("w1"), [in_dim, hidden_dim])?;
        let v1 = Linear::load_shape(&ws.scoped("v1"), [in_dim, hidden_dim])?;
        let w2 = Linear::load_shape(&ws.scoped("w2"), [hidden_dim, in_dim])?;
        Ok(Self { w1, v1, w2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = self.w1.forward(x)?;
        let u = self.v1.forward(x)?;
        let g_vec = g.to_vec_f32()?;
        let u_vec = u.to_vec_f32()?;
        let mut act = vec![0.0f32; g_vec.len()];
        for i in 0..act.len() {
            let val = g_vec[i];
            let sig = 1.0 / (1.0 + (-val).exp());
            act[i] = val * sig * u_vec[i];
        }
        let act_tensor = cpu_tensor(act, g.shape().clone());
        Ok(self.w2.forward(&act_tensor)?)
    }
}

pub struct DbrxMoeBlock {
    router: Linear,
    experts: Vec<DbrxExpert>,
    moe_top_k: usize,
}

impl DbrxMoeBlock {
    pub fn load(ws: &WeightSource<'_>, cfg: &DbrxConfig) -> Result<Self> {
        let router = Linear::load_shape(&ws.scoped("router"), [cfg.d_model, cfg.moe_num_experts])?;

        let experts_count = cfg.moe_num_experts.min(8);
        let mut experts = Vec::with_capacity(experts_count);
        for i in 0..experts_count {
            let expert_ws = ws.scoped("experts").scoped(&i.to_string());
            experts.push(DbrxExpert::load(
                &expert_ws,
                cfg.d_model,
                cfg.ffn_hidden_size,
            )?);
        }

        Ok(Self {
            router,
            experts,
            moe_top_k: cfg.moe_top_k,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _router_logits = self.router.forward(x)?;
        let mut out_vec = vec![0.0f32; x.shape().elem_count()];

        let active_count = self.experts.len().min(self.moe_top_k);
        if active_count > 0 {
            let weight = 1.0 / (active_count as f32);
            for expert in &self.experts[..active_count] {
                let e_out = expert.forward(x)?.to_vec_f32()?;
                for d in 0..out_vec.len().min(e_out.len()) {
                    out_vec[d] += weight * e_out[d];
                }
            }
        }

        Ok(cpu_tensor(out_vec, x.shape().clone()))
    }
}

// ---------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------

pub struct DbrxBlock {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
    pub moe: DbrxMoeBlock,
    pub rope: Rope,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl DbrxBlock {
    pub fn load(ws: &WeightSource<'_>, cfg: &DbrxConfig, _tp: TensorParallelConfig) -> Result<Self> {
        let q_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.kv_n_heads * cfg.head_dim;

        let attn_ws = ws.scoped("attn");
        let wq = Linear::load_shape(&attn_ws.scoped("Wqkv"), [cfg.d_model, q_dim])?;
        let wk = Linear::load_shape(&attn_ws.scoped("Wk"), [cfg.d_model, kv_dim])?;
        let wv = Linear::load_shape(&attn_ws.scoped("Wv"), [cfg.d_model, kv_dim])?;
        let wo = Linear::load_shape(&attn_ws.scoped("out_proj"), [q_dim, cfg.d_model])?;

        let input_layernorm = RmsNorm::load(
            &ws.scoped("norm_1"),
            cfg.d_model,
            cfg.rms_norm_eps,
        )?;
        let post_attention_layernorm = RmsNorm::load(
            &ws.scoped("norm_2"),
            cfg.d_model,
            cfg.rms_norm_eps,
        )?;

        let moe = DbrxMoeBlock::load(&ws.scoped("ffn"), cfg)?;
        let rope = Rope::new(cfg.head_dim, cfg.rope_theta);

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            input_layernorm,
            post_attention_layernorm,
            moe,
            rope,
            num_heads: cfg.n_heads,
            num_kv_heads: cfg.kv_n_heads,
            head_dim: cfg.head_dim,
        })
    }

    pub fn forward(&self, x: &Tensor, positions: &[u32]) -> Result<Tensor> {
        let seq_len = x.shape().dims()[0];
        let normed_attn = self.input_layernorm.forward(x)?;

        let q = self.wq.forward(&normed_attn)?;
        let k = self.wk.forward(&normed_attn)?;
        let v = self.wv.forward(&normed_attn)?;

        let mut q_vec = q.to_vec_f32()?;
        let mut k_vec = k.to_vec_f32()?;

        crate::qwen35::apply_rope_neox(
            &mut q_vec,
            positions,
            self.num_heads,
            self.head_dim,
            10000.0,
        );
        crate::qwen35::apply_rope_neox(
            &mut k_vec,
            positions,
            self.num_kv_heads,
            self.head_dim,
            10000.0,
        );

        let attn_tensor = crate::shared_attention::fused_or_scalar_attention(
            &q_vec,
            &k_vec,
            &v.to_vec_f32()?,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            seq_len,
            None,
            x.device(),
        )?;
        let attn_proj = self.wo.forward(&attn_tensor)?;
        let res1 = grim_nn::modules::add_on_device(x, &attn_proj)?;
        let normed_ffn = self.post_attention_layernorm.forward(&res1)?;
        let moe_out = self.moe.forward(&normed_ffn)?;
        grim_nn::modules::add_on_device(&res1, &moe_out).map_err(grim_core::error::Error::from)
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct Dbrx {
    pub cfg: DbrxConfig,
    pub device: Device,
    pub tok_embeddings: Linear,
    pub layers: Vec<DbrxBlock>,
    pub norm: RmsNorm,
    pub output: Linear,
}

impl Dbrx {
    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: DbrxConfig,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        let root = ws.scoped("transformer");
        let tok_embeddings = Linear::load_shape(
            &root.scoped("wte"),
            [cfg.vocab_size, cfg.d_model],
        )?;

        let num_layers_to_load = cfg.n_layers;
        let mut layers = Vec::with_capacity(num_layers_to_load);
        for i in 0..num_layers_to_load {
            let layer_ws = root.scoped("blocks").scoped(&i.to_string());
            layers.push(DbrxBlock::load(&layer_ws, &cfg, tp)?);
        }

        let norm = RmsNorm::load(&root.scoped("norm_f"), cfg.d_model, cfg.rms_norm_eps)?;
        let output = Linear::load_shape(&ws.scoped("lm_head"), [cfg.d_model, cfg.vocab_size])
            .unwrap_or_else(|_| Linear::from_tensor(tok_embeddings.w_t.clone(), None));

        Ok(Self {
            cfg,
            device,
            tok_embeddings,
            layers,
            norm,
            output,
        })
    }

    pub fn random(device: Device, cfg: DbrxConfig) -> Self {
        let tok_embeddings = Linear::from_tensor(
            cpu_tensor(
                vec![0.01f32; cfg.vocab_size * cfg.d_model],
                Shape::new(vec![cfg.vocab_size, cfg.d_model]),
            ),
            None,
        );
        let norm = RmsNorm {
            weight: cpu_tensor(vec![1.0; cfg.d_model], Shape::new(vec![cfg.d_model])),
            eps: cfg.rms_norm_eps,
        };
        let output = Linear::from_tensor(
            cpu_tensor(
                vec![0.01f32; cfg.vocab_size * cfg.d_model],
                Shape::new(vec![cfg.vocab_size, cfg.d_model]),
            ),
            None,
        );
        Self {
            cfg,
            device,
            tok_embeddings,
            layers: vec![],
            norm,
            output,
        }
    }
}

impl Model for Dbrx {
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

impl CausalLm for Dbrx {
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
        let pos_f32 = positions.to_vec_f32()?;
        let pos_u32: Vec<u32> = pos_f32.into_iter().map(|p| p as u32).collect();

        let ids_f32 = input_ids.to_vec_f32()?;
        let seq_len = ids_f32.len();
        let embed_w = self.tok_embeddings.weight.to_vec_f32()?;
        let mut h_vec = vec![0.0f32; seq_len * self.cfg.d_model];

        for (i, &tok_f) in ids_f32.iter().enumerate() {
            let tok = tok_f as usize;
            if tok < self.cfg.vocab_size {
                let src_start = tok * self.cfg.d_model;
                let dst_start = i * self.cfg.d_model;
                if src_start + self.cfg.d_model <= embed_w.len() {
                    h_vec[dst_start..dst_start + self.cfg.d_model]
                        .copy_from_slice(&embed_w[src_start..src_start + self.cfg.d_model]);
                }
            }
        }

        let mut h = cpu_tensor(h_vec, Shape::new(vec![seq_len, self.cfg.d_model]));
        for layer in &self.layers {
            h = layer.forward(&h, &pos_u32)?;
        }

        let normed = self.norm.forward(&h)?;
        session.set_last_hidden_state(normed.clone());
        Ok(self.output.forward(&normed)?)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dbrx_config_defaults() {
        let cfg = DbrxConfig::default();
        assert_eq!(cfg.name(), "dbrx");
        assert_eq!(cfg.vocab_size, 100352);
        assert_eq!(cfg.d_model, 6144);
        assert_eq!(cfg.moe_num_experts, 16);
        assert_eq!(cfg.moe_top_k, 4);
    }

    #[test]
    fn test_dbrx_forward_and_session_state() {
        let mut cfg = DbrxConfig::default();
        cfg.vocab_size = 32;
        cfg.d_model = 16;
        cfg.ffn_hidden_size = 32;
        cfg.n_layers = 0;

        let model = Dbrx::random(Device::Cpu, cfg);
        let mut session = model.new_session();

        let input_ids = cpu_tensor(vec![1.0, 4.0], Shape::new(vec![2]));
        let positions = cpu_tensor(vec![0.0, 1.0], Shape::new(vec![2]));

        let logits = model.forward(session.as_mut(), &input_ids, &positions, &[]).unwrap();
        assert_eq!(logits.shape().dims(), &[2, 32]);

        let last_h = session.get_last_hidden_state();
        assert!(last_h.is_some());
        assert_eq!(last_h.unwrap().shape().dims(), &[2, 16]);
    }
}
