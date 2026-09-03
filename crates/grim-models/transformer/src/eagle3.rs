//! EAGLE-3 draft model architecture with multi-layer target feature fusion.
//!
//! Real EAGLE-3 fuses intermediate layer representations (typically 3 layers: low, mid, high)
//! from the target base model via a linear projection `fc: 3 * D_target -> D_draft`,
//! and decodes autoregressively using a lightweight decoder layer with concatenated
//! token embedding + hidden state attention projection [E_t, H_{t-1}].

use grim_backend_cpu::cpu_tensor;
use grim_core::error::{Error, Result};
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::{Inner, SessionT};
use grim_nn::{
    ColumnParallelLinear, Embedding, Linear, RmsNorm, Rope, RowParallelLinear,
    TensorParallelConfig, WeightSource,
};
use grim_tensor::{ArithType, DType, Device, Shape, Tensor};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Eagle3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub target_hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_seq_len: usize,
    pub num_target_fusion_layers: usize,
}

impl Default for Eagle3Config {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            target_hidden_size: 4096,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            num_layers: 1,
            intermediate_size: 11008,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 2048,
            num_target_fusion_layers: 3,
        }
    }
}

impl ModelConfig for Eagle3Config {
    fn name(&self) -> &str {
        "eagle3"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------------------------------------------------------------------------
// EAGLE-3 Decoder Layer (concatenated [E_t, H_{t-1}] input attention)
// ---------------------------------------------------------------------------

pub struct Eagle3DecoderLayer {
    pub hidden_norm: RmsNorm,
    pub input_layernorm: RmsNorm,
    pub wq: ColumnParallelLinear,
    pub wk: ColumnParallelLinear,
    pub wv: ColumnParallelLinear,
    pub wo: RowParallelLinear,
    pub post_attn_norm: RmsNorm,
    pub w_gate: ColumnParallelLinear,
    pub w_up: ColumnParallelLinear,
    pub w_down: RowParallelLinear,
    pub rope: Rope,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
}

impl Eagle3DecoderLayer {
    pub fn load_tp(
        ws: &WeightSource<'_>,
        cfg: &Eagle3Config,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let concat_dim = hidden_size * 2;
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;

        let hidden_norm = RmsNorm::load(&ws.pp("hidden_norm"), hidden_size, cfg.rms_norm_eps)?;
        let input_layernorm =
            RmsNorm::load(&ws.pp("input_layernorm"), hidden_size, cfg.rms_norm_eps)?;

        let wq = ColumnParallelLinear::new(
            Linear::load(&ws.pp("self_attn").pp("q_proj"), concat_dim, q_dim, false)?,
            tp,
        );
        let wk = ColumnParallelLinear::new(
            Linear::load(&ws.pp("self_attn").pp("k_proj"), concat_dim, kv_dim, false)?,
            tp,
        );
        let wv = ColumnParallelLinear::new(
            Linear::load(&ws.pp("self_attn").pp("v_proj"), concat_dim, kv_dim, false)?,
            tp,
        );
        let wo = RowParallelLinear::new(
            Linear::load(&ws.pp("self_attn").pp("o_proj"), q_dim, hidden_size, false)?,
            tp,
        );

        let post_attn_norm = RmsNorm::load(
            &ws.pp("post_attention_layernorm"),
            hidden_size,
            cfg.rms_norm_eps,
        )?;

        let w_gate = ColumnParallelLinear::new(
            Linear::load(
                &ws.pp("mlp").pp("gate_proj"),
                hidden_size,
                cfg.intermediate_size,
                false,
            )?,
            tp,
        );
        let w_up = ColumnParallelLinear::new(
            Linear::load(
                &ws.pp("mlp").pp("up_proj"),
                hidden_size,
                cfg.intermediate_size,
                false,
            )?,
            tp,
        );
        let w_down = RowParallelLinear::new(
            Linear::load(
                &ws.pp("mlp").pp("down_proj"),
                cfg.intermediate_size,
                hidden_size,
                false,
            )?,
            tp,
        );

        let rope = Rope::from_config(grim_tensor::RopeConfig {
            dim: cfg.head_dim,
            base: cfg.rope_theta,
            rotary_dim: cfg.head_dim,
            yarn: None,
            interleaved: true,
        });

        Ok(Self {
            hidden_norm,
            input_layernorm,
            wq,
            wk,
            wv,
            wo,
            post_attn_norm,
            w_gate,
            w_up,
            w_down,
            rope,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            hidden_size,
        })
    }

    pub fn random(cfg: &Eagle3Config, rng_seed: u64) -> Self {
        let mut rng = SimpleEagleRng::new(rng_seed);
        let hidden_size = cfg.hidden_size;
        let concat_dim = hidden_size * 2;
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let tp = TensorParallelConfig::default();

        let lin = |out_dim: usize, in_dim: usize, rng: &mut SimpleEagleRng| {
            let data: Vec<f32> = (0..out_dim * in_dim)
                .map(|_| (rng.next_f32() - 0.5) * 0.02)
                .collect();
            Linear::from_tensor(cpu_tensor(data, Shape::new(vec![out_dim, in_dim])), None)
        };
        let rms = |dim: usize| RmsNorm {
            weight: cpu_tensor(vec![1.0; dim], Shape::new(vec![dim])),
            eps: cfg.rms_norm_eps,
        };

        Self {
            hidden_norm: rms(hidden_size),
            input_layernorm: rms(hidden_size),
            wq: ColumnParallelLinear::new(lin(q_dim, concat_dim, &mut rng), tp),
            wk: ColumnParallelLinear::new(lin(kv_dim, concat_dim, &mut rng), tp),
            wv: ColumnParallelLinear::new(lin(kv_dim, concat_dim, &mut rng), tp),
            wo: RowParallelLinear::new(lin(hidden_size, q_dim, &mut rng), tp),
            post_attn_norm: rms(hidden_size),
            w_gate: ColumnParallelLinear::new(
                lin(cfg.intermediate_size, hidden_size, &mut rng),
                tp,
            ),
            w_up: ColumnParallelLinear::new(lin(cfg.intermediate_size, hidden_size, &mut rng), tp),
            w_down: RowParallelLinear::new(lin(hidden_size, cfg.intermediate_size, &mut rng), tp),
            rope: Rope::from_config(grim_tensor::RopeConfig {
                dim: cfg.head_dim,
                base: cfg.rope_theta,
                rotary_dim: cfg.head_dim,
                yarn: None,
                interleaved: true,
            }),
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            hidden_size,
        }
    }

    /// Forward pass with token embedding `input_emb` and previous hidden states `hidden_states`.
    pub fn forward(
        &self,
        input_emb: &Tensor,
        hidden_states: &Tensor,
        positions: &[u32],
    ) -> Result<Tensor> {
        let norm_h = self.hidden_norm.forward(hidden_states)?;
        let norm_e = self.input_layernorm.forward(input_emb)?;

        let h_vec = norm_h.to_vec_f32()?;
        let e_vec = norm_e.to_vec_f32()?;
        let seq_len = positions.len().max(1);
        let hidden_dim = self.hidden_size;

        // Concatenate along hidden dimension: [E_t, H_{t-1}] -> [seq_len, 2 * hidden_dim]
        let mut concat_vec = Vec::with_capacity(seq_len * hidden_dim * 2);
        for t in 0..seq_len {
            let e_slice = &e_vec[t * hidden_dim..(t + 1) * hidden_dim];
            let h_slice = &h_vec[t * hidden_dim..(t + 1) * hidden_dim];
            concat_vec.extend_from_slice(e_slice);
            concat_vec.extend_from_slice(h_slice);
        }

        let concat_t = cpu_tensor(concat_vec, Shape::new(vec![seq_len, hidden_dim * 2]));

        // Q, K, V projections from concatenated input
        let q = self.wq.forward(&concat_t)?;
        let k = self.wk.forward(&concat_t)?;
        let v = self.wv.forward(&concat_t)?;

        // RoPE stays on the projection's device: relabel to the per-head row
        // layout the Rope module's kernel contract expects.
        let mut q_ext_pos = Vec::with_capacity(seq_len * self.num_heads);
        for &pos in positions {
            for _ in 0..self.num_heads {
                q_ext_pos.push(pos);
            }
        }
        let mut k_ext_pos = Vec::with_capacity(seq_len * self.num_kv_heads);
        for &pos in positions {
            for _ in 0..self.num_kv_heads {
                k_ext_pos.push(pos);
            }
        }
        let q_3d = crate::block::reshaped_view(
            &q,
            &Shape::new(vec![1, seq_len * self.num_heads, self.head_dim]),
        )?;
        let q_rope = self.rope.forward(&q_3d, &q_ext_pos)?;

        let k_3d = crate::block::reshaped_view(
            &k,
            &Shape::new(vec![1, seq_len * self.num_kv_heads, self.head_dim]),
        )?;
        let k_rope = self.rope.forward(&k_3d, &k_ext_pos)?;

        // Dense causal self-attention on device (host scalar loop only via
        // the fused-kernel fallback guard).
        let q_flat = crate::block::reshaped_view(
            &q_rope,
            &Shape::new(vec![seq_len, self.num_heads * self.head_dim]),
        )?;
        let k_flat = crate::block::reshaped_view(
            &k_rope,
            &Shape::new(vec![seq_len, self.num_kv_heads * self.head_dim]),
        )?;
        let v_flat = crate::block::reshaped_view(
            &v,
            &Shape::new(vec![seq_len, self.num_kv_heads * self.head_dim]),
        )?;
        let attn_t = crate::shared_attention::fused_attention_tensors(
            &q_flat,
            &k_flat,
            &v_flat,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            seq_len,
            seq_len,
            None,
        )?;
        let o_proj = self.wo.forward(&attn_t)?;

        // Residual add with hidden_states
        let attn_res = grim_nn::modules::add_on_device(hidden_states, &o_proj)?;

        // SwiGLU MLP
        let norm_mlp = self.post_attn_norm.forward(&attn_res)?;
        let gate = self.w_gate.forward(&norm_mlp)?;
        let up = self.w_up.forward(&norm_mlp)?;
        let act = grim_nn::modules::silu_mul_on_device(&gate, &up)?;
        let down = self.w_down.forward(&act)?;

        Ok(grim_nn::modules::add_on_device(&attn_res, &down)?)
    }
}

// ---------------------------------------------------------------------------
// EAGLE-3 Model
// ---------------------------------------------------------------------------

pub struct Eagle3 {
    pub cfg: Eagle3Config,
    pub device: Device,
    /// Multi-layer target feature fusion projection: 3 * D_target -> D_draft.
    pub fc: Linear,
    pub tok_embeddings: Embedding,
    pub layers: Vec<Eagle3DecoderLayer>,
    pub norm: RmsNorm,
    pub output: Linear,
}

impl Eagle3 {
    pub fn load(device: Device, ws: &WeightSource<'_>, cfg: Eagle3Config) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    pub fn load_tp(
        device: Device,
        ws: &WeightSource<'_>,
        cfg: Eagle3Config,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        let num_fusion = cfg.num_target_fusion_layers.max(1);
        let fusion_in_dim = cfg.target_hidden_size * num_fusion;

        let fc = Linear::load(&ws.pp("fc"), fusion_in_dim, cfg.hidden_size, false)?;
        let tok_embeddings =
            Embedding::load(&ws.pp("tok_embeddings"), cfg.vocab_size, cfg.hidden_size)?;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(Eagle3DecoderLayer::load_tp(
                &ws.pp("layers").pp(&i.to_string()),
                &cfg,
                tp,
            )?);
        }

        let norm = RmsNorm::load(&ws.pp("norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let output = match Linear::load_column_parallel(
            &ws.pp("output"),
            cfg.hidden_size,
            cfg.vocab_size,
            false,
            tp,
        ) {
            Ok(o) => o,
            Err(_) => {
                let ws_unsharded = ws.with_tp_config(TensorParallelConfig::default());
                Linear::load(
                    &ws_unsharded.pp("output"),
                    cfg.hidden_size,
                    cfg.vocab_size,
                    false,
                )?
            }
        };

        Ok(Self {
            cfg,
            device,
            fc,
            tok_embeddings,
            layers,
            norm,
            output,
        })
    }

    pub fn random(device: Device, cfg: Eagle3Config) -> Self {
        let mut rng = SimpleEagleRng::new(0xEAE6_3000_1337_F00Du64);
        let num_fusion = cfg.num_target_fusion_layers.max(1);
        let fusion_in_dim = cfg.target_hidden_size * num_fusion;

        let lin = |out_dim: usize, in_dim: usize, rng: &mut SimpleEagleRng| {
            let data: Vec<f32> = (0..out_dim * in_dim)
                .map(|_| (rng.next_f32() - 0.5) * 0.02)
                .collect();
            Linear::from_tensor(cpu_tensor(data, Shape::new(vec![out_dim, in_dim])), None)
        };
        let rms = |dim: usize| RmsNorm {
            weight: cpu_tensor(vec![1.0; dim], Shape::new(vec![dim])),
            eps: cfg.rms_norm_eps,
        };

        let fc = lin(cfg.hidden_size, fusion_in_dim, &mut rng);

        let embed_data: Vec<f32> = (0..cfg.vocab_size * cfg.hidden_size)
            .map(|_| (rng.next_f32() - 0.5) * 0.02)
            .collect();
        let tok_embeddings = Embedding {
            weight: cpu_tensor(
                embed_data,
                Shape::new(vec![cfg.vocab_size, cfg.hidden_size]),
            ),
        };

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(Eagle3DecoderLayer::random(&cfg, 0x1000 + i as u64));
        }

        let output = lin(cfg.vocab_size, cfg.hidden_size, &mut rng);

        Self {
            fc,
            tok_embeddings,
            layers,
            norm: rms(cfg.hidden_size),
            output,
            device,
            cfg,
        }
    }

    /// Fuse multi-layer target feature activations into initial draft hidden state H_0.
    ///
    /// Concatenates `target_layer_hiddens` (e.g. 3 layers) along the feature channel and
    /// projects through `self.fc: 3 * D_target -> D_draft`.
    pub fn fuse_target_layers(&self, target_layer_hiddens: &[&Tensor]) -> Result<Tensor> {
        if target_layer_hiddens.is_empty() {
            return Err(Error::Config(
                "Eagle3: target_layer_hiddens cannot be empty".into(),
            ));
        }

        let seq_len = target_layer_hiddens[0]
            .shape()
            .dims()
            .first()
            .copied()
            .unwrap_or(1);
        let num_fusion = self.cfg.num_target_fusion_layers.max(1);

        // Pull each target hidden ONCE — the old loop re-downloaded every
        // layer tensor `seq_len` times from inside the per-token loop.
        let layer_rows: Vec<Vec<f32>> = (0..num_fusion)
            .map(|layer_idx| {
                let layer_tensor = if layer_idx < target_layer_hiddens.len() {
                    target_layer_hiddens[layer_idx]
                } else {
                    target_layer_hiddens[target_layer_hiddens.len() - 1]
                };
                layer_tensor.to_vec_f32().map_err(grim_core::error::Error::from)
            })
            .collect::<Result<_>>()?;

        let mut flattened_fusion =
            Vec::with_capacity(seq_len * self.cfg.target_hidden_size * num_fusion);
        for t in 0..seq_len {
            for vec in &layer_rows {
                let start = t * self.cfg.target_hidden_size;
                let end = (start + self.cfg.target_hidden_size).min(vec.len());
                flattened_fusion.extend_from_slice(&vec[start..end]);
            }
        }

        let fusion_shape = Shape::new(vec![seq_len, self.cfg.target_hidden_size * num_fusion]);
        let fusion_t = cpu_tensor(flattened_fusion, fusion_shape);
        Ok(self.fc.forward(&fusion_t)?)
    }

    /// Perform a single draft step given current token embedding `input_emb` and hidden `hidden_states`.
    pub fn decode_step(
        &self,
        input_emb: &Tensor,
        hidden_states: &Tensor,
        positions: &[u32],
    ) -> Result<(Tensor, Tensor)> {
        let mut cur_h = hidden_states.clone();
        for layer in &self.layers {
            cur_h = layer.forward(input_emb, &cur_h, positions)?;
        }
        let normed = self.norm.forward(&cur_h)?;
        let logits = self.output.forward(&normed)?;
        Ok((logits, cur_h))
    }
}

impl Model for Eagle3 {
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

impl CausalLm for Eagle3 {
    fn new_session(&self) -> Box<dyn SessionT> {
        Box::new(Inner::new(self.device.clone()))
    }

    fn forward(
        &self,
        _session: &mut dyn SessionT,
        input_ids: &Tensor,
        positions: &Tensor,
        _adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
        let ids: Vec<u32> = match input_ids.dtype() {
            d if d == DType::F32 => input_ids
                .to_vec_f32()?
                .into_iter()
                .map(|x| x as u32)
                .collect(),
            _ => {
                return Err(Error::Config(
                    "non-F32 input_ids not supported in Eagle3".into(),
                ));
            }
        };
        let seq_len = ids.len();
        let embed = self
            .tok_embeddings
            .forward(&ids, seq_len, self.cfg.hidden_size)?;

        let pos_vec: Vec<u32> = if positions.shape().dims().iter().product::<usize>() == seq_len {
            positions
                .to_vec_f32()?
                .into_iter()
                .map(|x| x as u32)
                .collect()
        } else {
            (0..seq_len as u32).collect()
        };

        // For standalone forward, initialize hidden representation from embedding
        let (logits, _) = self.decode_step(&embed, &embed, &pos_vec)?;
        Ok(logits)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct SimpleEagleRng {
    state: u64,
}

impl SimpleEagleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }
    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state >> 32) as u32
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eagle3_fusion_and_decode_step() {
        let cfg = Eagle3Config {
            vocab_size: 100,
            hidden_size: 32,
            target_hidden_size: 64,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 16,
            num_layers: 1,
            intermediate_size: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 128,
            num_target_fusion_layers: 3,
        };

        let model = Eagle3::random(Device::Cpu, cfg);

        // 3 target layer hidden states of dim 64
        let h1 = cpu_tensor(vec![1.0; 64], Shape::new(vec![1, 64]));
        let h2 = cpu_tensor(vec![2.0; 64], Shape::new(vec![1, 64]));
        let h3 = cpu_tensor(vec![3.0; 64], Shape::new(vec![1, 64]));

        // Target feature fusion
        let fused_h0 = model.fuse_target_layers(&[&h1, &h2, &h3]).unwrap();
        assert_eq!(fused_h0.shape().dims(), &[1, 32]);

        // Token embedding for token 42
        let embed = model.tok_embeddings.forward(&[42], 1, 32).unwrap();
        assert_eq!(embed.shape().dims(), &[1, 32]);

        // Autoregressive draft step
        let (logits, next_h) = model.decode_step(&embed, &fused_h0, &[0]).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 100]);
        assert_eq!(next_h.shape().dims(), &[1, 32]);
    }
}
