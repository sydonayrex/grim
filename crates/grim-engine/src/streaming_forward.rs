//! Streaming block-wise forward execution with gradient checkpointing (WI-T2).
//!
//! Provides `StreamingBlockForward` that reads quantized transformer weights
//! lazily block-by-block from a `TensorProvider`, runs fused forward operations,
//! and manages activation recomputation buffers (`GradientCheckpointBuffer`).

use grim_core::error::{Error, Result};
use grim_models_transformer::{LlamaBlock, LlamaConfig};
use grim_nn::WeightSource;
use grim_tensor::{Device, Tensor, TensorProvider};
use std::collections::HashMap;

/// Saved activation checkpoint for a transformer block.
#[derive(Debug, Clone)]
pub struct LayerActivationCheckpoint {
    pub layer_idx: usize,
    pub input_x: Tensor,
}

/// Gradient checkpointing buffer enforcing bounded peak memory by retaining only block inputs.
#[derive(Debug, Default)]
pub struct GradientCheckpointBuffer {
    checkpoints: HashMap<usize, LayerActivationCheckpoint>,
}

impl GradientCheckpointBuffer {
    /// Create a new empty checkpoint buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Save input activation checkpoint for layer `layer_idx`.
    pub fn save(&mut self, layer_idx: usize, input_x: Tensor) {
        self.checkpoints.insert(layer_idx, LayerActivationCheckpoint {
            layer_idx,
            input_x,
        });
    }

    /// Retrieve input activation checkpoint for layer `layer_idx`.
    pub fn get(&self, layer_idx: usize) -> Option<&Tensor> {
        self.checkpoints.get(&layer_idx).map(|c| &c.input_x)
    }

    /// Clear stored checkpoints after backward pass completion.
    pub fn clear(&mut self) {
        self.checkpoints.clear();
    }
}

/// Block-wise streaming forward executor for memory-bounded QLoRA fine-tuning.
pub struct StreamingBlockForward {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub checkpoint_buffer: GradientCheckpointBuffer,
}

impl StreamingBlockForward {
    /// Create a new `StreamingBlockForward` instance.
    pub fn new(num_layers: usize, hidden_size: usize) -> Self {
        Self {
            num_layers,
            hidden_size,
            checkpoint_buffer: GradientCheckpointBuffer::new(),
        }
    }

    /// Run streaming block-wise forward pass for `layer_idx`.
    ///
    /// Reads layer input `x`, records activation checkpoint in `checkpoint_buffer`,
    /// then loads block weights lazily from `provider` and runs real
    /// transformer block math (RMSNorm → GQA attention → residual →
    /// RMSNorm → SwiGLU FFN → residual) via `LlamaBlock`.
    pub fn forward_block(
        &mut self,
        provider: &dyn TensorProvider,
        cfg: &LlamaConfig,
        layer_idx: usize,
        x: &Tensor,
    ) -> Result<Tensor> {
        if layer_idx >= self.num_layers {
            return Err(Error::Config(format!(
                "layer_idx {} out of bounds for num_layers {}",
                layer_idx, self.num_layers
            )));
        }

        // Save input checkpoint for recomputation during backward pass
        self.checkpoint_buffer.save(layer_idx, x.clone());

        // Load block weights lazily from provider on target tensor device, run real forward
        let ws = WeightSource::root(provider, x.device().clone());
        let block_ws = ws.pp("layers").pp(&layer_idx.to_string());
        let block = LlamaBlock::load(&block_ws, cfg)?;
        block.forward(x)
    }

    /// Recompute block forward pass from saved input checkpoint during backward traversal.
    /// Reloads block weights from `provider` and re-runs the real forward.
    pub fn recompute_block(
        &self,
        provider: &dyn TensorProvider,
        cfg: &LlamaConfig,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let input_x = self
            .checkpoint_buffer
            .get(layer_idx)
            .ok_or_else(|| Error::Config(format!("missing activation checkpoint for layer {}", layer_idx)))?;

        // Reload block weights from provider on target tensor device, run real forward from saved input
        let ws = WeightSource::root(provider, input_x.device().clone());
        let block_ws = ws.pp("layers").pp(&layer_idx.to_string());
        let block = LlamaBlock::load(&block_ws, cfg)?;
        block.forward(input_x)
    }

    /// Run streaming block-wise forward pass for `layer_idx` with autograd tape recording.
    ///
    /// Loads block weights lazily from `provider`, executes pre-norm attention and SwiGLU FFN,
    /// applies enabled LoRA adapters via `apply_and_record_lora`, and records `LoRAApply` and `Add`
    /// entries on `tape`. Returns `(output_tensor_id, output_tensor)`.
    pub fn forward_block_with_autograd(
        &mut self,
        provider: &dyn TensorProvider,
        cfg: &LlamaConfig,
        autograd_reg: &grim_autograd::AutogradRegistry,
        tape: &mut grim_autograd::Tape,
        layer_idx: usize,
        x: &Tensor,
        x_id: grim_autograd::TensorId,
    ) -> Result<(grim_autograd::TensorId, Tensor)> {
        if layer_idx >= self.num_layers {
            return Err(Error::Config(format!(
                "layer_idx {} out of bounds for num_layers {}",
                layer_idx, self.num_layers
            )));
        }

        self.checkpoint_buffer.save(layer_idx, x.clone());

        let ws = WeightSource::root(provider, x.device().clone());
        let block_ws = ws.pp("layers").pp(&layer_idx.to_string());
        let block = LlamaBlock::load(&block_ws, cfg)?;

        // Pre-attention norm & Q/K/V projections
        let x_norm = block.attn_norm.forward(x)?;
        let x_norm_base_id = tape.register(x_norm.clone());

        let q_base = block.wq.forward(&x_norm)?;
        let q_base_id = tape.register(q_base.clone());
        let (_q_id, q) = grim_autograd::apply_and_record_lora(
            autograd_reg,
            tape,
            layer_idx,
            grim_autograd::LoRAInjectionPoint::QProj,
            q_base,
            q_base_id,
            x_norm.clone(),
            x_norm_base_id,
        )?;

        let k_base = block.wk.forward(&x_norm)?;
        let k_base_id = tape.register(k_base.clone());
        let (_k_id, k) = grim_autograd::apply_and_record_lora(
            autograd_reg,
            tape,
            layer_idx,
            grim_autograd::LoRAInjectionPoint::KProj,
            k_base,
            k_base_id,
            x_norm.clone(),
            x_norm_base_id,
        )?;

        let v_base = block.wv.forward(&x_norm)?;
        let v_base_id = tape.register(v_base.clone());
        let (_v_id, v) = grim_autograd::apply_and_record_lora(
            autograd_reg,
            tape,
            layer_idx,
            grim_autograd::LoRAInjectionPoint::VProj,
            v_base,
            v_base_id,
            x_norm.clone(),
            x_norm_base_id,
        )?;

        // Attention output projection
        let qd = q.to_vec_f32()?;
        let kd = k.to_vec_f32()?;
        let vd = v.to_vec_f32()?;
        let num_head_dims = cfg.num_heads * cfg.head_dim;
        let total_tokens = qd.len() / num_head_dims;
        let scale = 1.0 / (cfg.head_dim as f32).sqrt();
        let mut attn_raw_data = vec![0.0f32; total_tokens * num_head_dims];
        let kv_stride = cfg.num_kv_heads * cfg.head_dim;

        for h in 0..cfg.num_heads {
            let kvh = (h * cfg.num_kv_heads) / cfg.num_heads;
            for t in 0..total_tokens {
                let mut scores = vec![0.0f32; total_tokens];
                for t2 in 0..total_tokens {
                    if t2 > t {
                        scores[t2] = f32::NEG_INFINITY;
                    } else {
                        let mut dot = 0.0f32;
                        for d in 0..cfg.head_dim {
                            dot += qd[t * num_head_dims + h * cfg.head_dim + d]
                                * kd[t2 * kv_stride + kvh * cfg.head_dim + d];
                        }
                        scores[t2] = dot * scale;
                    }
                }
                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                for s in &mut scores {
                    *s /= sum;
                }
                for d in 0..cfg.head_dim {
                    let mut acc = 0.0f32;
                    for t2 in 0..total_tokens {
                        acc += scores[t2] * vd[t2 * kv_stride + kvh * cfg.head_dim + d];
                    }
                    attn_raw_data[t * num_head_dims + h * cfg.head_dim + d] = acc;
                }
            }
        }
        let attn_raw = grim_backend_cpu::cpu_tensor(attn_raw_data, grim_tensor::Shape::new(vec![total_tokens, num_head_dims]));
        let attn_raw_id = tape.register(attn_raw.clone());

        let wo_base = block.wo.forward(&attn_raw)?;
        let wo_base_id = tape.register(wo_base.clone());
        let (wo_id, wo_out) = grim_autograd::apply_and_record_lora(
            autograd_reg,
            tape,
            layer_idx,
            grim_autograd::LoRAInjectionPoint::OProj,
            wo_base,
            wo_base_id,
            attn_raw,
            attn_raw_id,
        )?;

        // Residual addition 1
        let dev = grim_autograd::pick_device_for_tensor(&x);
        let (res1_storage, _) = dev.add(x.storage().as_ref(), wo_out.storage().as_ref(), x.shape())?;
        let res1 = Tensor::new(
            std::sync::Arc::from(res1_storage),
            x.shape().clone(),
            grim_tensor::DType::F32,
            x.provenance().clone(),
            x.device().clone(),
        );
        let res1_id = tape.record_add(x_id, wo_id, res1.clone(), None);

        // FFN pre-norm & Gate/Up/Down projections
        let ffn_norm_out = block.ffn_norm.forward(&res1)?;
        let ffn_norm_id = tape.register(ffn_norm_out.clone());

        let gate_base = block.w_gate.forward(&ffn_norm_out)?;
        let gate_base_id = tape.register(gate_base.clone());
        let (_gate_id, gate) = grim_autograd::apply_and_record_lora(
            autograd_reg,
            tape,
            layer_idx,
            grim_autograd::LoRAInjectionPoint::GateProj,
            gate_base,
            gate_base_id,
            ffn_norm_out.clone(),
            ffn_norm_id,
        )?;

        let up_base = block.w_up.forward(&ffn_norm_out)?;
        let up_base_id = tape.register(up_base.clone());
        let (_up_id, up) = grim_autograd::apply_and_record_lora(
            autograd_reg,
            tape,
            layer_idx,
            grim_autograd::LoRAInjectionPoint::UpProj,
            up_base,
            up_base_id,
            ffn_norm_out.clone(),
            ffn_norm_id,
        )?;

        let (silu_storage, _) = dev.silu_mul(gate.storage().as_ref(), up.storage().as_ref(), gate.shape())?;
        let silu_tensor = Tensor::new(
            std::sync::Arc::from(silu_storage),
            gate.shape().clone(),
            grim_tensor::DType::F32,
            gate.provenance().clone(),
            gate.device().clone(),
        );
        let silu_id = tape.register(silu_tensor.clone());

        let down_base = block.w_down.forward(&silu_tensor)?;
        let down_base_id = tape.register(down_base.clone());
        let (down_id, down_out) = grim_autograd::apply_and_record_lora(
            autograd_reg,
            tape,
            layer_idx,
            grim_autograd::LoRAInjectionPoint::DownProj,
            down_base,
            down_base_id,
            silu_tensor,
            silu_id,
        )?;

        // Residual addition 2
        let (res2_storage, _) = dev.add(res1.storage().as_ref(), down_out.storage().as_ref(), x.shape())?;
        let res2 = Tensor::new(
            std::sync::Arc::from(res2_storage),
            x.shape().clone(),
            grim_tensor::DType::F32,
            x.provenance().clone(),
            x.device().clone(),
        );
        let res2_id = tape.record_add(res1_id, down_id, res2.clone(), None);

        Ok((res2_id, res2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_backend_cpu::cpu_tensor;
    use grim_tensor::dtype::{DType, QuantProvenance};
    use grim_tensor::{RawTensor, Shape, TensorProvider};

    struct StubProvider {
        cfg: LlamaConfig,
    }

    impl StubProvider {
        fn new() -> Self {
            Self {
                cfg: LlamaConfig {
                    vocab_size: 256,
                    hidden_size: 32,
                    num_heads: 2,
                    num_kv_heads: 1,
                    head_dim: 16,
                    num_layers: 4,
                    intermediate_size: 64,
                    rms_norm_eps: 1e-5,
                    rope_theta: 10000.0,
                    max_seq_len: 64,
                },
            }
        }
    }

    impl TensorProvider for StubProvider {
        fn get(&self, name: &str) -> grim_tensor::error::Result<RawTensor> {
            let c = &self.cfg;
            let (n, shape) = if name.contains("attn_norm") || name.contains("ffn_norm") {
                (c.hidden_size, vec![c.hidden_size])
            } else if name.contains("wq") || name.contains("wo") {
                let rows = if name.contains("wq") { c.num_heads * c.head_dim } else { c.hidden_size };
                let cols = if name.contains("wq") { c.hidden_size } else { c.num_heads * c.head_dim };
                (rows * cols, vec![rows, cols])
            } else if name.contains("wk") || name.contains("wv") {
                (c.num_kv_heads * c.head_dim * c.hidden_size,
                 vec![c.num_kv_heads * c.head_dim, c.hidden_size])
            } else if name.contains("w_gate") || name.contains("w_up") {
                (c.intermediate_size * c.hidden_size,
                 vec![c.intermediate_size, c.hidden_size])
            } else if name.contains("w_down") {
                (c.hidden_size * c.intermediate_size,
                 vec![c.hidden_size, c.intermediate_size])
            } else {
                return Err(grim_tensor::Error::Backend(format!("stub: unknown tensor {name}")));
            };
            Ok(RawTensor {
                bytes: vec![0u8; n * 4],
                shape,
                dtype: DType::F32,
                provenance: QuantProvenance::GrimNative,
            })
        }

        fn meta(&self, _name: &str) -> grim_tensor::error::Result<grim_tensor::TensorMeta> {
            Ok(grim_tensor::TensorMeta {
                dtype: DType::F32,
                provenance: QuantProvenance::GrimNative,
                shape: vec![],
                fusion_mask: 0,
            })
        }
    }

    #[test]
    fn gradient_checkpoint_buffer_saves_and_retrieves() {
        let mut buf = GradientCheckpointBuffer::new();
        let t = cpu_tensor(vec![1.0, 2.0], Shape::new(vec![1, 2]));
        buf.save(0, t.clone());

        let retrieved = buf.get(0).unwrap();
        assert_eq!(retrieved.to_vec_f32().unwrap(), vec![1.0, 2.0]);
    }

    #[test]
    fn streaming_block_forward_runs_real_llama_block() {
        let provider = StubProvider::new();
        let cfg = provider.cfg.clone();
        let mut forward = StreamingBlockForward::new(4, cfg.hidden_size);
        let x = cpu_tensor(vec![0.5; cfg.hidden_size], Shape::new(vec![1, cfg.hidden_size]));

        let out = forward.forward_block(&provider, &cfg, 0, &x).unwrap();

        // Output must have same shape as input (real block forward ran without error)
        assert_eq!(out.shape().dims(), x.shape().dims());
    }

    #[test]
    fn streaming_block_recompute_matches_forward() {
        let provider = StubProvider::new();
        let cfg = provider.cfg.clone();
        let mut forward = StreamingBlockForward::new(4, cfg.hidden_size);
        let x = cpu_tensor(vec![0.5; cfg.hidden_size], Shape::new(vec![1, cfg.hidden_size]));

        let out = forward.forward_block(&provider, &cfg, 0, &x).unwrap();
        let recomputed = forward.recompute_block(&provider, &cfg, 0).unwrap();

        // Recomputed output must match the original forward output
        let out_vals = out.to_vec_f32().unwrap();
        let rec_vals = recomputed.to_vec_f32().unwrap();
        assert_eq!(out_vals.len(), rec_vals.len());
        for (a, b) in out_vals.iter().zip(rec_vals.iter()) {
            assert!((a - b).abs() < 1e-6, "recompute mismatch: {a} vs {b}");
        }
    }
}
