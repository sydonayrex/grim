//! Llama/Mistral-style dense transformer — `CausalLm` implementation.

use std::sync::Arc;

use grim_backend_cpu::CpuDevice;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint};
use grim_core::session::{Inner, SessionT};
use grim_core::{Model, ModelConfig};
use grim_nn::{
    ColumnParallelLinear, Embedding, Linear, RowParallelLinear, TensorParallelConfig,
};
use grim_nn::RmsNorm;
use grim_nn::Rope;
use grim_nn::pick_device_for_storage_device;
use grim_tensor::{ArithType, DType, Device, Shape, Tensor};

use crate::block::{LlamaBlock, LlamaConfigRefs};
use grim_core::rng::SimpleRng;

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
}

impl Llama {
    /// Load a `Llama` model with TP config taken from the `WeightSource`.
    ///
    /// The `WeightSource` is the single source of truth for TP — it carries
    /// the `(rank, world_size)` set by `model_loader` (which derives it from
    /// `GRIM_TP_*` env once and threads it through `with_tp_config`). Re-reading
    /// the env here would split the contract: the loader's `get_sharded` would
    /// slice by `ws.tp_config().rank` while `load_tp` would shard by a
    /// freshly-parsed env rank. During a transient env mutation those can
    /// disagree, so this entry uses `ws.tp_config()` and never calls
    /// `from_env()`.
    pub fn load(device: Device, ws: &grim_nn::WeightSource<'_>, cfg: LlamaConfig) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    /// Load a `Llama` model with an explicit `TensorParallelConfig`.
    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: LlamaConfig,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        let tok_embeddings =
            Embedding::load(&ws.pp("tok_embeddings"), cfg.vocab_size, cfg.hidden_size)?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(LlamaBlock::load_tp(
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
            /*has_bias=*/ false,
            tp,
        ) {
            Ok(o) => o,
            Err(_) => {
                let ws_unsharded = ws.with_tp_config(TensorParallelConfig::default());
                Linear::load(&ws_unsharded.pp("output"), cfg.hidden_size, cfg.vocab_size, false)?
            }
        };
        Ok(Self {
            cfg,
            device: device.clone(),
            tok_embeddings,
            layers,
            norm,
            output,
        })
    }

    pub fn random(device: Device, cfg: LlamaConfig) -> Self {
        use grim_backend_cpu::cpu_tensor;
        let dev = CpuDevice::new();
        let mut rng = SimpleRng::new(0xDEAD_BEEF_CAFE_F00Du64);

        let embed_data: Vec<f32> = (0..cfg.vocab_size * cfg.hidden_size)
            .map(|_| (rng.next_f32() - 0.5) * 0.02)
            .collect();
        let tok_embeddings = Embedding {
            weight: cpu_tensor(
                embed_data,
                Shape::new(vec![cfg.vocab_size, cfg.hidden_size]),
            ),
        };

        let mut linear = |out: usize, inp: usize| {
            let data: Vec<f32> = (0..out * inp)
                .map(|_| (rng.next_f32() - 0.5) * 0.02)
                .collect();
            Linear::from_tensor(cpu_tensor(data, Shape::new(vec![out, inp])), None)
        };
        let rms = |dim: usize| RmsNorm {
            weight: cpu_tensor(vec![1.0; dim], Shape::new(vec![dim])),
            eps: cfg.rms_norm_eps,
        };

        let tp = TensorParallelConfig::default();
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for _ in 0..cfg.num_layers {
            layers.push(LlamaBlock {
                attn_norm: rms(cfg.hidden_size),
                wq: ColumnParallelLinear::new(
                    linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim),
                    tp,
                ),
                wk: ColumnParallelLinear::new(
                    linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim),
                    tp,
                ),
                wv: ColumnParallelLinear::new(
                    linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim),
                    tp,
                ),
                wo: RowParallelLinear::new(
                    linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size),
                    tp,
                ),
                ffn_norm: rms(cfg.hidden_size),
                w_gate: ColumnParallelLinear::new(
                    linear(cfg.hidden_size, cfg.intermediate_size),
                    tp,
                ),
                w_up: ColumnParallelLinear::new(
                    linear(cfg.hidden_size, cfg.intermediate_size),
                    tp,
                ),
                w_down: RowParallelLinear::new(
                    linear(cfg.intermediate_size, cfg.hidden_size),
                    tp,
                ),
                rope: Rope::new(cfg.head_dim, cfg.rope_theta),
                tp_config: tp,
                _dev: Device::Cpu,
                _cfg: LlamaConfigRefs {
                    hidden_size: cfg.hidden_size,
                    num_heads: cfg.num_heads,
                    num_kv_heads: cfg.num_kv_heads,
                    head_dim: cfg.head_dim,
                    intermediate_size: cfg.intermediate_size,
                    tp_world_size: 1,
                    local_num_heads: cfg.num_heads,
                    local_num_kv_heads: cfg.num_kv_heads,
                    kv_head_replica_factor: 1,
                },
            });
        }

        let norm = rms(cfg.hidden_size);
        let output = linear(cfg.vocab_size, cfg.hidden_size);
        Self {
            cfg: cfg.clone(),
            device,
            tok_embeddings,
            layers,
            norm,
            output,
        }
    }

    pub fn embed_token(&self, token: u32) -> Result<Tensor> {
        Ok(self
            .tok_embeddings
            .forward(&[token], 1, self.cfg.hidden_size)?)
    }

    pub fn decode(
        &self,
        hidden: &Tensor,
        positions: &[u32],
    ) -> Result<(Tensor, Tensor, Vec<(Tensor, Tensor)>)> {
        let mut h = hidden.clone();
        let mut kv_pairs = Vec::new();
        for layer in &self.layers {
            let (out, k, v) = layer.forward_with_kv(&h, positions)?;
            kv_pairs.push((k, v));
            h = out;
        }
        let h = self.norm.forward(&h)?;
        let logits = self.output.forward(&h)?;
        Ok((logits, h, kv_pairs))
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
    fn as_any(&self) -> &dyn std::any::Any {
        self
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
        positions: &Tensor,
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
        let hidden_shape = Shape::new(vec![1, seq_len, self.cfg.hidden_size]);
        let dev = pick_device_for_storage_device(&self.device);
        let hidden_storage = dev.from_cpu(&hidden, &hidden_shape, DType::F32)?;
        let hidden_t = Tensor::new(
            Arc::from(hidden_storage),
            hidden_shape.clone(),
            DType::F32,
            self.tok_embeddings.weight.provenance().clone(),
            self.device.clone(),
        );
        // MAJ-3: use the positions tensor passed by the engine instead of
        // hardcoding 0..seq_len. During decode the engine passes the actual
        // current_pos so RoPE sees the correct absolute position.
        let pos_vec: Vec<u32> = if positions.shape().dims().iter().product::<usize>() == seq_len {
            positions
                .to_vec_f32()?
                .into_iter()
                .map(|x| x as u32)
                .collect()
        } else {
            (0..seq_len).map(|i| i as u32).collect()
        };
        let (logits, hidden_state, kv_pairs) = self.decode(&hidden_t, &pos_vec)?;
        // MAJ-1: populate the KV cache with K/V from each layer so the
        // cache infrastructure is no longer dead code.
        for (k, v) in &kv_pairs {
            session.append_kv(k, v)?;
        }
        session.set_last_hidden_state(hidden_state);
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
