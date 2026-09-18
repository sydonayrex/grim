//! Compatibility loader and native model for `zai-org/GLM-5.2` (HuggingFace `model_type = "glm5_2"`).
//! # Architecture Details - **Fused QKV**: Input is projected to query, key, and value vectors.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_nn::{Linear, RmsNorm, Rope, TensorParallelConfig, WeightSource};
use grim_tensor::{ArithType, Device, Shape, Tensor};

/// Native mirror of `Glm52Config` (HuggingFace `glm5_2`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Glm52Config {
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

impl Default for Glm52Config {
    fn default() -> Self {
        Self {
            vocab_size: 151552,
            hidden_size: 4096,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            num_hidden_layers: 40,
            intermediate_size: 13824,
            num_experts: 64,
            num_experts_per_tok: 8,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_position_embeddings: 8192,
        }
    }
}

impl ModelConfig for Glm52Config {
    fn name(&self) -> &str {
        "glm5_2"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Glm52Config {
    pub fn from_hf(value: &serde_json::Value) -> Self {
        let u = |k: &str| value.get(k).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let f = |k: &str| value.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        Glm52Config {
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

/// A single expert MLP inside the GLM-5.2 MoE layer.
pub struct Glm52Expert {
    pub dense_h_to_4h: Linear,
    pub dense_4h_to_h: Linear,
}

impl Glm52Expert {
    pub fn load(
        ws: &WeightSource<'_>,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Self> {
        let dense_h_to_4h = Linear::load_shape(
            &ws.scoped("dense_h_to_4h"),
            [hidden_size, intermediate_size],
        )?;
        let dense_4h_to_h = Linear::load_shape(
            &ws.scoped("dense_4h_to_h"),
            [intermediate_size, hidden_size],
        )?;
        Ok(Self {
            dense_h_to_4h,
            dense_4h_to_h,
        })
    }

    /// Kernel gap: GELU (tanh approximation) has no device kernel, so this
    /// activation stays host-side.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.dense_h_to_4h.forward(x)?;
        let hv = h.to_vec_f32()?;
        let gelu_v: Vec<f32> = hv
            .iter()
            .map(|&v| 0.5 * v * (1.0 + (0.797_884_6 * (v + 0.044715 * v.powi(3))).tanh()))
            .collect();
        let gelu_t = cpu_tensor(gelu_v, h.shape().clone());
        Ok(self.dense_4h_to_h.forward(&gelu_t)?)
    }
}

/// MoE layer with top-k gating over a bank of GLM-5.2 experts.
pub struct Glm52Moe {
    pub gate: Linear,
    pub experts: Vec<Glm52Expert>,
    pub num_experts_per_tok: usize,
    /// Resident GPU weight cache for the GELU Charon dispatch path.
    charon_cache: crate::shared_moe::CharonCache,
}

impl Glm52Moe {
    pub fn load(ws: &WeightSource<'_>, cfg: &Glm52Config) -> Result<Self> {
        let gate = Linear::load_shape(&ws.scoped("gate"), [cfg.hidden_size, cfg.num_experts])?;
        let exp_ws = ws.scoped("experts");
        let mut experts = Vec::with_capacity(cfg.num_experts);
        for e in 0..cfg.num_experts {
            let expert = Glm52Expert::load(
                &exp_ws.scoped(&e.to_string()),
                cfg.hidden_size,
                cfg.intermediate_size,
            )?;
            experts.push(expert);
        }
        Ok(Self {
            gate,
            experts,
            num_experts_per_tok: cfg.num_experts_per_tok.max(1),
            charon_cache: crate::shared_moe::CharonCache::new(),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let seq_len = x.shape().dims()[0];
        let hidden_dim = x.shape().dims()[1];
        let logits = self.gate.forward(x)?;
        let logits_v = logits.to_vec_f32()?;
        let num_exp = self.experts.len();

        // Build per-token top-k routings (used by both GPU and CPU paths).
        let mut routings: Vec<crate::shared_moe::TokenRouting> =
            Vec::with_capacity(seq_len);
        for s in 0..seq_len {
            let row_logits = &logits_v[s * num_exp..(s + 1) * num_exp];
            let mut indexed: Vec<(usize, f32)> =
                row_logits.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let topk = &indexed[..self.num_experts_per_tok.min(num_exp)];
            let max_logit = topk.iter().map(|(_, l)| *l).fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = topk.iter().map(|(_, l)| (l - max_logit).exp()).collect();
            let sum_exp: f32 = exps.iter().sum();
            let weights: Vec<f32> = exps.iter().map(|e| e / (sum_exp + 1e-12)).collect();
            routings.push(topk.iter().zip(weights.iter()).map(|((ei, _), w)| (*ei, *w)).collect());
        }

        // GPU path: Charon GELU grouped dispatch (ROCm only).
        if matches!(x.device(), grim_tensor::Device::Rocm(_)) {
            let gate_refs: Vec<&grim_tensor::Tensor> =
                self.experts.iter().map(|e| &e.dense_h_to_4h.weight).collect();
            let down_refs: Vec<&grim_tensor::Tensor> =
                self.experts.iter().map(|e| &e.dense_4h_to_h.weight).collect();
            let dev = grim_backend_rocm::RocmDevice::shared(match x.device() {
                grim_tensor::Device::Rocm(o) => *o,
                _ => unreachable!(),
            });
            if let Some(out) = crate::shared_moe::gelu_charon_dispatch(
                &*dev,
                x,
                &gate_refs,
                &down_refs,
                &routings,
                1.0,
                &self.charon_cache,
            )? {
                return Ok(out);
            }
        }

        // CPU fallback: per-expert GELU loop.
        let xv = x.to_vec_f32()?;
        let mut out = vec![0.0f32; seq_len * hidden_dim];
        for (s, routing) in routings.iter().enumerate() {
            let token_x = cpu_tensor(
                xv[s * hidden_dim..(s + 1) * hidden_dim].to_vec(),
                Shape::new(vec![1, hidden_dim]),
            );
            for &(expert_idx, w) in routing {
                let exp_out = self.experts[expert_idx].forward(&token_x)?;
                let exp_out_v = exp_out.to_vec_f32()?;
                for d in 0..hidden_dim {
                    out[s * hidden_dim + d] += w * exp_out_v[d];
                }
            }
        }
        Ok(cpu_tensor(out, x.shape().clone()))
    }
}

/// A transformer block in GLM-5.2 with fused QKV attention and MoE routing.
pub struct Glm52Block {
    pub fused_qkv: Linear,
    pub dense: Linear,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
    pub mlp: Glm52Moe,
    pub rope: Rope,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl Glm52Block {
    pub fn load(ws: &WeightSource<'_>, cfg: &Glm52Config) -> Result<Self> {
        let qkv_out_dim = (cfg.num_attention_heads + 2 * cfg.num_key_value_heads) * cfg.head_dim;
        let fused_qkv = Linear::load_shape(
            &ws.scoped("self_attention").scoped("query_key_value"),
            [cfg.hidden_size, qkv_out_dim],
        )?;
        let dense = Linear::load_shape(
            &ws.scoped("self_attention").scoped("dense"),
            [cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size],
        )?;

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

        let mlp = Glm52Moe::load(&ws.scoped("mlp"), cfg)?;
        let rope = Rope::new(cfg.head_dim, cfg.rope_theta);

        Ok(Self {
            fused_qkv,
            dense,
            input_layernorm,
            post_attention_layernorm,
            mlp,
            rope,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        })
    }

    /// GPU-first forward for the fused-QKV attention path.
    /// Kernel gap: the fused `query_key_value` projection has no device column-split primitive, so the Q/K/V split.
    pub fn forward(
        &self,
        x: &Tensor,
        positions: &[u32],
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let seq_len = x.shape().dims()[0];
        let attn_normed = self.input_layernorm.forward(x)?;

        let qkv = self.fused_qkv.forward(&attn_normed)?;
        let qkv_vec = qkv.to_vec_f32()?;

        let q_dim = self.num_heads * self.head_dim;
        let k_dim = self.num_kv_heads * self.head_dim;
        let v_dim = self.num_kv_heads * self.head_dim;
        let total_qkv = q_dim + k_dim + v_dim;

        let mut q_data = vec![0.0f32; seq_len * q_dim];
        let mut k_data = vec![0.0f32; seq_len * k_dim];
        let mut v_data = vec![0.0f32; seq_len * v_dim];

        for s in 0..seq_len {
            let row_offset = s * total_qkv;
            q_data[s * q_dim..(s + 1) * q_dim]
                .copy_from_slice(&qkv_vec[row_offset..row_offset + q_dim]);
            k_data[s * k_dim..(s + 1) * k_dim]
                .copy_from_slice(&qkv_vec[row_offset + q_dim..row_offset + q_dim + k_dim]);
            v_data[s * v_dim..(s + 1) * v_dim]
                .copy_from_slice(&qkv_vec[row_offset + q_dim + k_dim..row_offset + total_qkv]);
        }

        let q_rot = cpu_tensor(q_data, Shape::new(vec![seq_len, q_dim]));
        let k_rot = cpu_tensor(k_data, Shape::new(vec![seq_len, k_dim]));
        let v_tensor = cpu_tensor(v_data, Shape::new(vec![seq_len, v_dim]));

        let q_rot = crate::shared_attention::rope_2d_on_device(
            &self.rope,
            &q_rot,
            self.num_heads,
            positions,
        )?;
        let k_rot = crate::shared_attention::rope_2d_on_device(
            &self.rope,
            &k_rot,
            self.num_kv_heads,
            positions,
        )?;

        // Device-side history: prev rows stay resident, only the new rows
        // are appended (D2D arena copy when the backend supports it).
        let (k_all, v_all) = if let Some((prev_k, prev_v)) = kv_cache {
            let full_k = crate::shared_attention::concat_rows_on_device(prev_k, &k_rot)?;
            let full_v = crate::shared_attention::concat_rows_on_device(prev_v, &v_tensor)?;
            *kv_cache = Some((full_k.clone(), full_v.clone()));
            (full_k, full_v)
        } else {
            *kv_cache = Some((k_rot.clone(), v_tensor.clone()));
            (k_rot, v_tensor)
        };
        let kv_len = k_all.shape().dims()[0];

        // Shared helper applies the causal mask at cache_offset + s (fixes
        // future-token leakage during multi-token prefill).
        let attn_tensor = crate::shared_attention::fused_attention_tensors(
            &q_rot,
            &k_all,
            &v_all,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            seq_len,
            kv_len,
            None,
        )?;
        let attn_proj = self.dense.forward(&attn_tensor)?;

        let res1 = grim_nn::modules::add_on_device(x, &attn_proj)?;
        let normed_ffn = self.post_attention_layernorm.forward(&res1)?;
        let mlp_out = self.mlp.forward(&normed_ffn)?;
        // Routing stays host-side, so the MoE output lands on the host; stage
        // it back next to `res1` before the residual add.
        let mlp_out = grim_nn::modules::move_to_device(&mlp_out, x.device())?;

        grim_nn::modules::add_on_device(&res1, &mlp_out).map_err(grim_core::error::Error::from)
    }
}

// Model & Session

pub struct Glm52 {
    pub cfg: Glm52Config,
    pub device: Device,
    pub word_embeddings: Linear,
    pub layers: Vec<Glm52Block>,
    pub final_layernorm: RmsNorm,
    pub output_layer: Linear,
}

impl Glm52 {
    pub fn load(device: Device, ws: &grim_nn::WeightSource<'_>, cfg: Glm52Config) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: Glm52Config,
        _tp: TensorParallelConfig,
    ) -> Result<Self> {
        let root = if ws.has_tensor("transformer.layers.0.input_layernorm.weight") {
            ws.scoped("transformer")
        } else {
            ws.scoped("model")
        };

        let word_embeddings = Linear::load_shape(
            &root.scoped("word_embeddings"),
            [cfg.vocab_size, cfg.hidden_size],
        )
        .or_else(|_| {
            Linear::load_shape(
                &root.scoped("embed_tokens"),
                [cfg.vocab_size, cfg.hidden_size],
            )
        })?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let layer_ws = root.scoped("layers").scoped(&i.to_string());
            let block = Glm52Block::load(&layer_ws, &cfg)?;
            layers.push(block);
        }

        let final_layernorm = RmsNorm::load(
            &root.scoped("final_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )
        .or_else(|_| RmsNorm::load(&root.scoped("norm"), cfg.hidden_size, cfg.rms_norm_eps))?;

        let output_layer = Linear::load_shape(
            &root.scoped("output_layer"),
            [cfg.hidden_size, cfg.vocab_size],
        )
        .or_else(|_| Linear::load_shape(&ws.scoped("lm_head"), [cfg.hidden_size, cfg.vocab_size]))
        .unwrap_or_else(|_| word_embeddings.clone());

        Ok(Self {
            cfg,
            device,
            word_embeddings,
            layers,
            final_layernorm,
            output_layer,
        })
    }
}

impl Model for Glm52 {
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

impl CausalLm for Glm52 {
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
            &self.word_embeddings.weight,
            &ids,
            seq_len,
            self.cfg.hidden_size,
        )?;

        let mut kv_caches = vec![None; self.layers.len()];

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, &pos_v, &mut kv_caches[layer_idx])?;
        }

        let normed = self.final_layernorm.forward(&x)?;
        let logits = self.output_layer.forward(&normed)?;
        session.advance_pos(seq_len);
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_core::architecture::ModelArchitecture;

    const GLM5_2_CONFIG: &str = r#"{
        "architectures": ["Glm52ForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": 40,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 13824,
        "num_experts": 64,
        "num_experts_per_tok": 8,
        "rms_norm_eps": 1e-05,
        "rope_theta": 10000.0,
        "vocab_size": 151552
    }"#;

    #[test]
    fn parses_glm5_2_config() {
        let v: serde_json::Value = serde_json::from_str(GLM5_2_CONFIG).unwrap();
        let cfg = Glm52Config::from_hf(&v);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 40);
        assert_eq!(cfg.num_experts, 64);
        assert_eq!(cfg.name(), "glm5_2");
    }

    #[test]
    fn dispatches_glm5_2_architecture() {
        assert_eq!(
            ModelArchitecture::from_str("glm5_2"),
            ModelArchitecture::Glm52
        );
    }
}

// ===========================================================================
// GELU Charon dispatch parity (WI-gpu-native-moe Phase 3): the device grouped
// GELU kernel (null expert_up_w, tanh-approx GELU) must match the host
// reference loop. Regression: the launch was rejected by the null-check on
// expert_up_w before `allow_null_up` was added to validate_grouped_inputs.
// =========================================================================

#[cfg(test)]
mod gelu_dispatch_parity_tests {
    use super::*;
    use grim_backend_rocm::RocmDevice;
    use grim_tensor::CoreTensorOps;

    fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.4
            })
            .collect()
    }

    fn rocm_tensor(dev: &RocmDevice, data: Vec<f32>, shape: Shape) -> Tensor {
        let storage = dev.from_cpu(&data, &shape, grim_tensor::dtype::DType::F32).unwrap();
        Tensor::new(
            std::sync::Arc::from(storage),
            shape,
            grim_tensor::dtype::DType::F32,
            grim_tensor::QuantProvenance::GrimNative,
            Device::Rocm(0),
        )
    }

    fn make_moe(dev: Option<&RocmDevice>, hidden: usize, inter: usize, n_exp: usize) -> Glm52Moe {
        let lin = |data: Vec<f32>, out: usize, inp: usize| -> Linear {
            let t = match dev {
                Some(d) => rocm_tensor(d, data, Shape::new(vec![out, inp])),
                None => cpu_tensor(data, Shape::new(vec![out, inp])),
            };
            Linear::from_tensor(t, None)
        };
        let experts = (0..n_exp)
            .map(|e| {
                let s = (e as u64 + 1) * 613;
                Glm52Expert {
                    dense_h_to_4h: lin(rand_vec(inter * hidden, s + 1), inter, hidden),
                    dense_4h_to_h: lin(rand_vec(inter * hidden, s + 2), hidden, inter),
                }
            })
            .collect();
        Glm52Moe {
            gate: lin(rand_vec(n_exp * hidden, 42), n_exp, hidden),
            experts,
            num_experts_per_tok: 2,
            charon_cache: crate::shared_moe::CharonCache::new(),
        }
    }

    #[test]
    fn glm52_gelu_charon_dispatch_matches_host_reference() {
        if !grim_backend_rocm::device::util::gpu_test_enabled() {
            eprintln!("skip: set GRIM_GPU_TEST=1 for GPU graph test");
            return;
        }
        let _gpu = grim_backend_rocm::device::util::gpu_test_lock();
        let dev = RocmDevice::shared(0);
        let hidden = 32usize;
        let inter = 64usize;
        let n_exp = 8usize;
        let seq = 3usize;

        let block_gpu = make_moe(Some(&dev), hidden, inter, n_exp);
        let x_data = rand_vec(seq * hidden, 7);
        let x = rocm_tensor(&dev, x_data.clone(), Shape::new(vec![seq, hidden]));
        let out_gpu = block_gpu.forward(&x).unwrap().to_vec_f32().unwrap();

        // Host reference: identical weights on CPU device (host loop path).
        let block_cpu = make_moe(None, hidden, inter, n_exp);
        let out_ref = block_cpu
            .forward(&cpu_tensor(x_data, Shape::new(vec![seq, hidden])))
            .unwrap()
            .to_vec_f32()
            .unwrap();

        assert_eq!(out_gpu.len(), out_ref.len());
        let max_diff = out_gpu
            .iter()
            .zip(out_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 2e-3,
            "GELU Charon dispatch diverged from host reference: max_diff={max_diff}"
        );
    }
}
