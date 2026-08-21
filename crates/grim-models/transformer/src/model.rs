//! Llama/Mistral-style dense transformer — `CausalLm` implementation.

use std::sync::Arc;

use grim_backend_cpu::CpuDevice;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint};
use grim_core::session::{Inner, SessionT};
use grim_core::{Model, ModelConfig};
use grim_nn::RmsNorm;
use grim_nn::Rope;
use grim_nn::pick_device_for_storage_device;
use grim_nn::{ColumnParallelLinear, Embedding, Linear, RowParallelLinear, TensorParallelConfig};
use grim_tensor::{ArithType, DType, Device, Shape, Tensor};

use crate::block::{LlamaBlock, LlamaConfigRefs};
use crate::moe_block::{MoESpec, MoeBlock};
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
    /// Fraction of `head_dim` that participates in RoPE (e.g. 0.5 → rotate
    /// half the channels). Defaults to 1.0 (full rotary); set < 1.0 for
    /// partial-rotary models like Qwen3.5-MoE.
    pub partial_rotary_factor: f32,
    /// YaRN RoPE scaling parameters. `None` ⇒ plain RoPE. When `Some`, every
    /// attention layer applies the YaRN frequency ramp + magnitude correction
    /// on top of the partial-rotary base frequencies.
    pub yarn: Option<grim_tensor::YaRNParams>,
}

impl LlamaConfig {
    /// Derived rotary dim: `round(head_dim * partial_rotary_factor)`,
    /// clamped to `head_dim`. This is what each layer's `RopeConfig.rotary_dim`
    /// is populated with (mirrors the Laguna/Maple convention).
    pub fn rotary_dim(&self) -> usize {
        let r = (self.head_dim as f32 * self.partial_rotary_factor).round() as usize;
        r.min(self.head_dim)
    }
}

impl ModelConfig for LlamaConfig {
    fn name(&self) -> &str {
        "llama"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    /// Context window in tokens. For LLaMA-family models this equals
    /// `max_seq_len` (populated from GGUF `llama.context_length` /
    /// `<arch>.context_length` during loading).
    fn context_length(&self) -> u64 {
        self.max_seq_len as u64
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
    /// Per-layer optional MoE routing block. `Some` for MoE layers (the
    /// corresponding `LlamaBlock.ffn_disabled` is set, so the dense FFN is
    /// skipped and this router+expert bank runs instead). `None` for dense
    /// layers and for the dense fallback when `load_tp` is used.
    pub moe_blocks: Vec<Option<MoeBlock>>,
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
                Linear::load(
                    &ws_unsharded.pp("output"),
                    cfg.hidden_size,
                    cfg.vocab_size,
                    false,
                )?
            }
        };
        let num_layers = cfg.num_layers;

        // Weight sanity check: models that loaded with zeroed weights should fail
        // at load time rather than silently returning Unimplemented on first forward.
        // [P1-36 fix: fail loudly on zeroed weights.]
        let check_not_zeroed = |name: &str, tensor: &grim_tensor::Tensor| {
            let data = tensor.to_vec_f32();
            if let Ok(data) = data {
                let all_zero = data.iter().all(|&v| v.abs() < 1e-10);
                let all_same = data.windows(2).all(|w| (w[1] - w[0]).abs() < 1e-10);
                if all_zero || all_same {
                    return Err(grim_tensor::Error::Backend(format!(
                        "{name}: weights appear to be zeroed or constant — \
                         model may be structurally broken"
                    )));
                }
            }
            Ok(())
        };
        check_not_zeroed("tok_embeddings", &tok_embeddings.weight)?;
        check_not_zeroed("norm", &norm.weight)?;
        check_not_zeroed("output", &output.weight())?;
        for (i, layer) in layers.iter().enumerate() {
            check_not_zeroed(&format!("layer.{i}.attn_norm"), &layer.attn_norm.weight)?;
            check_not_zeroed(&format!("layer.{i}.ffn_norm"), &layer.ffn_norm.weight)?;
            check_not_zeroed(&format!("layer.{i}.wq"), &layer.wq.weight())?;
            check_not_zeroed(&format!("layer.{i}.wk"), &layer.wk.weight())?;
            check_not_zeroed(&format!("layer.{i}.wv"), &layer.wv.weight())?;
            check_not_zeroed(&format!("layer.{i}.wo"), &layer.wo.weight())?;
            if let Some(ref g) = layer.w_gate {
                check_not_zeroed(&format!("layer.{i}.w_gate"), &g.weight())?;
            }
            if let Some(ref d) = layer.w_down {
                check_not_zeroed(&format!("layer.{i}.w_down"), &d.weight())?;
            }
        }

        Ok(Self {
            cfg,
            device: device.clone(),
            tok_embeddings,
            layers,
            moe_blocks: (0..num_layers).map(|_| None).collect(),
            norm,
            output,
        })
    }

    /// Load a `Llama` model that mixes dense and MoE layers.
    ///
    /// `moe_spec` is `Some(spec)` for layers that should route through a
    /// `MoeBlock` (their dense FFN is disabled), `None` for plain dense
    /// layers. The attention towers are always loaded per layer.
    pub fn load_tp_moe(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: LlamaConfig,
        moe_spec: &[Option<MoESpec>],
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        let attn_specs: Vec<crate::block::LayerAttentionSpec> = (0..cfg.num_layers)
            .map(|_| {
                crate::block::LayerAttentionSpec::full_with_rope(
                    cfg.num_heads,
                    cfg.num_kv_heads,
                    cfg.head_dim,
                    cfg.rope_theta,
                    cfg.rotary_dim(),
                    cfg.yarn,
                )
            })
            .collect();
        Self::load_tp_moe_specs(device, ws, cfg, moe_spec, &attn_specs, tp)
    }

    /// Load a `Llama` model with per-layer `MoESpec` and per-layer `LayerAttentionSpec`.
    pub fn load_tp_moe_specs(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: LlamaConfig,
        moe_spec: &[Option<MoESpec>],
        attn_specs: &[crate::block::LayerAttentionSpec],
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        if moe_spec.len() != cfg.num_layers {
            return Err(grim_core::error::Error::Config(format!(
                "load_tp_moe_specs: moe_spec len {} != num_layers {}",
                moe_spec.len(),
                cfg.num_layers
            )));
        }
        if attn_specs.len() != cfg.num_layers {
            return Err(grim_core::error::Error::Config(format!(
                "load_tp_moe_specs: attn_specs len {} != num_layers {}",
                attn_specs.len(),
                cfg.num_layers
            )));
        }
        let tok_embeddings =
            Embedding::load(&ws.pp("tok_embeddings"), cfg.vocab_size, cfg.hidden_size)?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        let mut moe_blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let lws = ws.pp("layers").pp(&i.to_string());
            let is_moe = moe_spec[i].is_some();
            let block = LlamaBlock::load_tp_spec_with_ffn(&lws, &cfg, &attn_specs[i], tp, !is_moe)?;
            if let Some(spec) = &moe_spec[i] {
                let moe = MoeBlock::load(&lws, &cfg, spec, tp)?;
                moe_blocks.push(Some(moe));
            } else {
                moe_blocks.push(None);
            }
            layers.push(block);
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
            device: device.clone(),
            tok_embeddings,
            layers,
            moe_blocks,
            norm,
            output,
        })
    }

    pub fn random(device: Device, cfg: LlamaConfig) -> Self {
        use grim_backend_cpu::cpu_tensor;
        let num_layers = cfg.num_layers;
        let _dev = CpuDevice::new();
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

        let mut linear = |out: usize, in_: usize| {
            let data: Vec<f32> = (0..out * in_)
                .map(|_| (rng.next_f32() - 0.5) * 0.02)
                .collect();
            Linear::from_tensor(cpu_tensor(data, Shape::new(vec![out, in_])), None)
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
                    linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size),
                    tp,
                ),
                wk: ColumnParallelLinear::new(
                    linear(cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size),
                    tp,
                ),
                wv: ColumnParallelLinear::new(
                    linear(cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size),
                    tp,
                ),
                wo: RowParallelLinear::new(
                    linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim),
                    tp,
                ),
                g_proj: None,
                q_norm: None,
                k_norm: None,
                ffn_norm: rms(cfg.hidden_size),
                w_gate: Some(ColumnParallelLinear::new(
                    linear(cfg.intermediate_size, cfg.hidden_size),
                    tp,
                )),
                w_up: Some(ColumnParallelLinear::new(
                    linear(cfg.intermediate_size, cfg.hidden_size),
                    tp,
                )),
                w_down: Some(RowParallelLinear::new(
                    linear(cfg.hidden_size, cfg.intermediate_size),
                    tp,
                )),
                rope: Rope::from_config(grim_tensor::RopeConfig {
                    dim: cfg.head_dim,
                    base: cfg.rope_theta,
                    rotary_dim: cfg.rotary_dim(),
                    yarn: cfg.yarn,
                }),
                tp_config: tp,
                ffn_disabled: false,
                _dev: Device::Cpu,
                _cfg: LlamaConfigRefs {
                    hidden_size: cfg.hidden_size,
                    num_heads: cfg.num_heads,
                    num_kv_heads: cfg.num_kv_heads,
                    head_dim: cfg.head_dim,
                    intermediate_size: cfg.intermediate_size,
                    sliding_window: None,
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
            moe_blocks: (0..num_layers).map(|_| None).collect(),
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
        let mut throwaway = Inner::new(self.device.clone());
        self.decode_paged(hidden, positions, &mut throwaway, None, 0)
    }

    pub fn decode_paged(
        &self,
        hidden: &Tensor,
        positions: &[u32],
        session: &mut dyn SessionT,
        mut caches: Option<&mut [Option<crate::block::LlamaLayerCache>]>,
        _layer: usize,
    ) -> Result<(Tensor, Tensor, Vec<(Tensor, Tensor)>)> {
        let mut h = hidden.clone();
        let mut kv_pairs = Vec::new();
        for (i, block) in self.layers.iter().enumerate() {
            let cache = caches.as_deref_mut().and_then(|c| c[i].as_mut());
            let (attn_out, k, v) =
                block.forward_with_kv_paged(&h, positions, Some(&mut *session), cache, i)?;
            kv_pairs.push((k, v));
            // MoE layers: `attn_out` is the post-attention residual (dense FFN
            // was skipped inside the block). Route it through the experts and
            // ADD the residual back — `MoeBlock::forward` returns only the
            // normed expert output, so skipping the add drops the residual
            // stream entirely (matches bailingmoe3 / solar_open2 convention).
            let out = if let Some(moe) = &self.moe_blocks[i] {
                let routed = moe.forward(&attn_out)?;
                grim_nn::modules::add_on_device(&attn_out, &routed)?
            } else {
                attn_out
            };
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
        // 2-D [seq_len, hidden] (batch=1): block layers feed this straight
        // into backend matmuls, which require storage rank 2.
        let hidden_shape = Shape::new(vec![seq_len, self.cfg.hidden_size]);
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
        // Two execution paths:
        //  * paged  — when the session carries a `PagedKvCache` (engine
        //    serving), K/V is written into physical page tensors and attention
        //    runs through the paged kernel via the logical block table.
        //  * classic — otherwise (single-shot tests, no KV session) we keep the
        //    per-layer `LlamaLayerCache` in `model_state` and use dense causal
        //    attention.
        let use_paged = session.has_kv();
        let (logits, hidden_state, _kv_pairs) = if use_paged {
            self.decode_paged(&hidden_t, &pos_vec, &mut *session, None, 0)?
        } else {
            // Initialize per-layer KV cache in model_state if not present.
            if session.model_state().is_none() {
                session.set_model_state(Box::new(
                    (0..self.layers.len())
                        .map(|_| Some(crate::block::LlamaLayerCache::new()))
                        .collect::<Vec<_>>(),
                ));
            }
            let caches = session
                .model_state_mut()
                .and_then(|s| s.downcast_mut::<Vec<Option<crate::block::LlamaLayerCache>>>())
                .ok_or_else(|| {
                    grim_core::error::Error::Session(
                        "Llama::forward: session.model_state has wrong type for this operation"
                            .into(),
                    )
                })?;
            let mut throwaway = Inner::new(self.device.clone());
            self.decode_paged(&hidden_t, &pos_vec, &mut throwaway, Some(caches), 0)?
        };
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

#[cfg(test)]
mod tests {
    use super::*;

    fn base_cfg(head_dim: usize, prf: f32) -> LlamaConfig {
        LlamaConfig {
            vocab_size: 100,
            hidden_size: 32,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim,
            num_layers: 1,
            intermediate_size: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_seq_len: 512,
            partial_rotary_factor: prf,
            yarn: None,
        }
    }

    /// `rotary_dim` derives `round(head_dim * partial_rotary_factor)` and clamps to
    /// `head_dim`. Qwen3.5-MoE uses a 0.5 partial factor on head_dim=128 ⇒ 64.
    #[test]
    fn llama_config_rotary_dim_derives_and_clamps() {
        assert_eq!(base_cfg(128, 1.0).rotary_dim(), 128, "full rotary");
        assert_eq!(
            base_cfg(128, 0.5).rotary_dim(),
            64,
            "qwen3.5-moe-style 0.5 partial rotary"
        );
        assert_eq!(
            base_cfg(32, 0.25).rotary_dim(),
            8,
            "0.25 partial rotary on head_dim 32"
        );
        // round-half-to-even on .5 boundary: round(0.5*33) = round(16.5) = 16 or 17
        let r = base_cfg(33, 0.5).rotary_dim();
        assert!(
            r == 16 || r == 17,
            "midpoint rounding must be near 16/17, got {r}"
        );
        // Clamp: prf > 1.0 cannot exceed head_dim.
        assert_eq!(base_cfg(16, 2.0).rotary_dim(), 16, "clamped to head_dim");
    }
}
