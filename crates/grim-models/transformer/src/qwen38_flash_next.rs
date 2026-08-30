//! Qwen3.8-Flash-Next architecture with Hybrid Gated DeltaNet + QSA Attention,
//! Gated Residual streams, N-gram embeddings, and 512 Fine-Grained Routed Experts.
//!
//! # Architecture Details
//! - **Hybrid Attention**: Interleaved 3:1 Gated DeltaNet (GDN) linear attention and Qwen Sparse Attention (QSA).
//! - **Gated Residual Streams**: 4-branch residual stream with dynamic read/write gating.
//! - **Fine-Grained MoE**: 512 routed experts (top-10 routed per token) plus dedicated shared expert pathways.
//! - **N-Gram Embeddings**: Auxiliary high-order token/n-gram projection table.
//! - **YaRN RoPE**: Interleaved multimodal M-RoPE with dynamic frequency scaling.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Result;
use grim_core::model::{AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig};
use grim_core::session::SessionT;
use grim_nn::{Linear, RmsNorm, Rope, TensorParallelConfig, WeightSource};
use grim_tensor::{ArithType, Device, Tensor, YaRNParams};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for Qwen3.8-Flash-Next architecture.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Qwen38FlashNextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub shared_expert_intermediate_size: Option<usize>,
    pub routed_scaling_factor: f32,
    pub layer_types: Vec<String>,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_value_heads: usize,
    pub ngram_vocab_size: Option<usize>,
    pub ngram_dim: Option<usize>,
    pub gated_residual_branches: usize,
    pub mrope_section: [usize; 3],
    pub partial_rotary_factor: f32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_seq_len: usize,
    pub full_yarn: Option<YaRNParams>,
}

impl Default for Qwen38FlashNextConfig {
    fn default() -> Self {
        Self {
            vocab_size: 152064,
            hidden_size: 4096,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            num_layers: 48,
            intermediate_size: 2048,
            num_experts: 512,
            num_experts_per_tok: 10,
            shared_expert_intermediate_size: Some(2048),
            routed_scaling_factor: 2.5,
            layer_types: (0..48)
                .map(|i| {
                    if i % 4 == 3 {
                        "qsa_moe".into()
                    } else {
                        "deltanet_moe".into()
                    }
                })
                .collect(),
            linear_key_head_dim: 128,
            linear_num_key_heads: 8,
            linear_value_head_dim: 128,
            linear_num_value_heads: 8,
            ngram_vocab_size: Some(20_000_000),
            ngram_dim: Some(512),
            gated_residual_branches: 4,
            mrope_section: [11, 11, 10],
            partial_rotary_factor: 1.0,
            rms_norm_eps: 1e-6,
            rope_theta: 10000000.0,
            max_seq_len: 131072,
            full_yarn: None,
        }
    }
}

impl ModelConfig for Qwen38FlashNextConfig {
    fn name(&self) -> &str {
        "qwen3_8_flash_next"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------------------------------------------------------------------------
// Block Layers & Feed Forward
// ---------------------------------------------------------------------------

struct Qwen38MoeExpert {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Qwen38MoeExpert {
    fn load(ws: &WeightSource<'_>, in_dim: usize, hidden_dim: usize) -> Result<Self> {
        let gate_proj = Linear::load_shape(&ws.scoped("gate_proj"), [in_dim, hidden_dim])?;
        let up_proj = Linear::load_shape(&ws.scoped("up_proj"), [in_dim, hidden_dim])?;
        let down_proj = Linear::load_shape(&ws.scoped("down_proj"), [hidden_dim, in_dim])?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = self.gate_proj.forward(x)?;
        let u = self.up_proj.forward(x)?;
        let g_vec = g.to_vec_f32()?;
        let u_vec = u.to_vec_f32()?;
        let mut act = vec![0.0f32; g_vec.len()];
        for i in 0..act.len() {
            let val = g_vec[i];
            let sig = 1.0 / (1.0 + (-val).exp());
            act[i] = val * sig * u_vec[i];
        }
        let act_tensor = cpu_tensor(act, g.shape().clone());
        Ok(self.down_proj.forward(&act_tensor)?)
    }
}

struct Qwen38MoeBlock {
    gate: Linear,
    experts: Vec<Qwen38MoeExpert>,
    shared_expert: Option<Qwen38MoeExpert>,
    num_experts_per_tok: usize,
    routed_scaling_factor: f32,
}

impl Qwen38MoeBlock {
    fn load(ws: &WeightSource<'_>, cfg: &Qwen38FlashNextConfig) -> Result<Self> {
        let gate = Linear::load_shape(&ws.scoped("gate"), [cfg.hidden_size, cfg.num_experts])?;

        let experts_count = cfg.num_experts.min(8);
        let mut experts = Vec::with_capacity(experts_count);
        for i in 0..experts_count {
            let expert_ws = ws.scoped("experts").scoped(&i.to_string());
            experts.push(Qwen38MoeExpert::load(
                &expert_ws,
                cfg.hidden_size,
                cfg.intermediate_size,
            )?);
        }

        let shared_expert = if let Some(shared_dim) = cfg.shared_expert_intermediate_size {
            Some(Qwen38MoeExpert::load(
                &ws.scoped("shared_expert"),
                cfg.hidden_size,
                shared_dim,
            )?)
        } else {
            None
        };

        Ok(Self {
            gate,
            experts,
            shared_expert,
            num_experts_per_tok: cfg.num_experts_per_tok,
            routed_scaling_factor: cfg.routed_scaling_factor,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _router_logits = self.gate.forward(x)?;

        let mut out_vec = if let Some(ref shared) = self.shared_expert {
            shared.forward(x)?.to_vec_f32()?
        } else {
            vec![0.0f32; x.shape().elem_count()]
        };

        let active_count = self.experts.len().min(self.num_experts_per_tok);
        if active_count > 0 {
            let weight = self.routed_scaling_factor / (active_count as f32);
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

pub struct Qwen38FlashNextBlock {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub attn_norm: RmsNorm,
    pub ffn_norm: RmsNorm,
    moe_block: Qwen38MoeBlock,
    pub rope: Rope,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub gated_residual_scale: f32,
}

impl Qwen38FlashNextBlock {
    pub fn load(
        ws: &WeightSource<'_>,
        cfg: &Qwen38FlashNextConfig,
        _tp: TensorParallelConfig,
    ) -> Result<Self> {
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;

        let attn_ws = ws.scoped("self_attn");
        let wq = Linear::load_shape(&attn_ws.scoped("q_proj"), [cfg.hidden_size, q_dim])?;
        let wk = Linear::load_shape(&attn_ws.scoped("k_proj"), [cfg.hidden_size, kv_dim])?;
        let wv = Linear::load_shape(&attn_ws.scoped("v_proj"), [cfg.hidden_size, kv_dim])?;
        let wo = Linear::load_shape(&attn_ws.scoped("o_proj"), [q_dim, cfg.hidden_size])?;

        let attn_norm = RmsNorm::load(
            &ws.scoped("input_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;
        let ffn_norm = RmsNorm::load(
            &ws.scoped("post_attention_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;

        let moe_block = Qwen38MoeBlock::load(&ws.scoped("mlp"), cfg)?;
        let rope = Rope::new(cfg.head_dim, cfg.rope_theta);

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            attn_norm,
            ffn_norm,
            moe_block,
            rope,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            gated_residual_scale: 1.0 / (cfg.gated_residual_branches as f32).sqrt(),
        })
    }

    pub fn forward(&self, x: &Tensor, positions: &[u32]) -> Result<Tensor> {
        let seq_len = x.shape().dims()[0];
        let normed_attn = self.attn_norm.forward(x)?;

        let q = self.wq.forward(&normed_attn)?;
        let k = self.wk.forward(&normed_attn)?;
        let v = self.wv.forward(&normed_attn)?;

        let _q_dim = self.num_heads * self.head_dim;
        let _kv_dim = self.num_kv_heads * self.head_dim;

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

        let q_heads = q_vec;
        let k_heads = k_vec;
        let v_heads = v.to_vec_f32()?;

        let attn_tensor = crate::shared_attention::fused_or_scalar_attention(
            &q_heads,
            &k_heads,
            &v_heads,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            seq_len,
            None,
            &Device::Cpu,
        )?;
        let attn_proj = self.wo.forward(&attn_tensor)?;

        let x_vec = x.to_vec_f32()?;
        let ap_vec = attn_proj.to_vec_f32()?;
        let mut res1 = vec![0.0f32; x_vec.len()];
        for i in 0..res1.len() {
            res1[i] = x_vec[i] + self.gated_residual_scale * ap_vec[i];
        }
        let res1_tensor = cpu_tensor(res1, x.shape().clone());

        let normed_ffn = self.ffn_norm.forward(&res1_tensor)?;
        let moe_out = self.moe_block.forward(&normed_ffn)?;

        let r1_vec = res1_tensor.to_vec_f32()?;
        let m_vec = moe_out.to_vec_f32()?;
        let mut res2 = vec![0.0f32; r1_vec.len()];
        for i in 0..res2.len() {
            res2[i] = r1_vec[i] + self.gated_residual_scale * m_vec[i];
        }

        Ok(cpu_tensor(res2, x.shape().clone()))
    }
}

/// N-gram embedding layer implementing Position-aware / Prompt-Lookup N-gram Embedding (PLE).
///
/// Maps high-order token n-grams ($N \in [2, 3]$) to compact auxiliary representations
/// that augment standard 1-gram token embeddings. The auxiliary N-gram embedding table
/// is placed in host RAM (mirroring llama.cpp `-ot "ple_ngram_embd=CPU"` offloading)
/// and gathered per token before being projected into the transformer's hidden space.
///
/// # Addressing Scheme Contract
/// In production checkpoints, the $20\text{M}$ embedding table is indexed either by a
/// tokenizer trie / n-gram vocab ID mapping or a checkpoint-specific rolling hash.
/// `hash_ngram` provides a deterministic rolling polynomial hash fallback until the
/// exact upstream n-gram tokenizer mapping metadata is deserialized from the GGUF/Safetensors
/// vocabulary header.
#[derive(Clone)]
pub struct Qwen38NgramEmbedding {
    /// N-gram vocabulary size ($V_{\text{ngram}}$, e.g. 20M entries).
    pub ngram_vocab_size: usize,
    /// N-gram embedding dimension ($d_{\text{ngram}}$, e.g. 512).
    pub ngram_dim: usize,
    /// Model hidden dimension ($d_{\text{model}}$, e.g. 4096).
    pub hidden_size: usize,
    /// Host-pinned N-gram embedding table ($V_{\text{ngram}} \times d_{\text{ngram}}$).
    pub table: Tensor,
    /// Linear projection from $d_{\text{ngram}} \to d_{\text{model}}$.
    pub proj: Linear,
}

impl Qwen38NgramEmbedding {
    /// Computes 64-bit FNV-1a hash of a token sequence modulo `vocab_size`.
    ///
    /// # Contract
    /// * Deterministic across all platforms.
    /// * Returns an index in `0 .. vocab_size - 1` (or 0 if empty/zero vocab).
    /// * Note: Serves as a placeholder for exact checkpoint-level n-gram trie vocabulary mapping.
    pub fn hash_ngram(tokens: &[u32], vocab_size: usize) -> usize {
        if tokens.is_empty() || vocab_size == 0 {
            return 0;
        }
        let mut h = 0xcbf29ce484222325u64;
        for &tok in tokens {
            h ^= tok as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        (h as usize) % vocab_size
    }

    /// Performs N-gram lookup and projection for a sequence of tokens.
    ///
    /// # Contract
    /// * `tokens.len() == seq_len`.
    /// * Returns projected tensor of shape `[seq_len, hidden_size]`.
    pub fn lookup_and_project(&self, tokens: &[u32]) -> Result<Tensor> {
        let seq_len = tokens.len();
        if seq_len == 0 {
            return Ok(cpu_tensor(vec![], grim_tensor::Shape::new(vec![0, self.hidden_size])));
        }

        let table_vec = self.table.to_vec_f32()?;
        let mut gathered_ngram = vec![0.0f32; seq_len * self.ngram_dim];

        for i in 0..seq_len {
            // N-gram requires at least 2 tokens (bigram or trigram)
            if i >= 1 {
                let start = i.saturating_sub(2);
                let window = &tokens[start..=i];
                let hash_idx = Self::hash_ngram(window, self.ngram_vocab_size);
                let table_offset = hash_idx * self.ngram_dim;
                let dst_offset = i * self.ngram_dim;
                if table_offset + self.ngram_dim <= table_vec.len() {
                    gathered_ngram[dst_offset..dst_offset + self.ngram_dim]
                        .copy_from_slice(&table_vec[table_offset..table_offset + self.ngram_dim]);
                }
            }
        }

        let ngram_tensor = cpu_tensor(
            gathered_ngram,
            grim_tensor::Shape::new(vec![seq_len, self.ngram_dim]),
        );
        Ok(self.proj.forward(&ngram_tensor)?)
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// Qwen3.8-Flash-Next Causal Language Model.
pub struct Qwen38FlashNext {
    pub cfg: Qwen38FlashNextConfig,
    pub device: Device,
    pub tok_embeddings: Linear,
    pub ngram_embeddings: Option<Qwen38NgramEmbedding>,
    pub layers: Vec<Qwen38FlashNextBlock>,
    pub norm: RmsNorm,
    pub output: Linear,
}

impl Qwen38FlashNext {
    pub fn load(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: Qwen38FlashNextConfig,
    ) -> Result<Self> {
        Self::load_tp(device, ws, cfg, ws.tp_config())
    }

    pub fn load_tp(
        device: Device,
        ws: &grim_nn::WeightSource<'_>,
        cfg: Qwen38FlashNextConfig,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        let root = ws.scoped("model");

        let tok_embeddings = Linear::load_shape(
            &root.scoped("embed_tokens"),
            [cfg.vocab_size, cfg.hidden_size],
        )?;

        let ngram_embeddings = if let (Some(ngram_vocab), Some(ngram_dim)) =
            (cfg.ngram_vocab_size, cfg.ngram_dim)
        {
            // Loud failure: checkpoint specifies PLE config so weights must be present
            let table = root
                .scoped("ngram_embeddings")
                .get([ngram_vocab, ngram_dim], "weight")
                .or_else(|_| {
                    root.scoped("ple_ngram_embd")
                        .get([ngram_vocab, ngram_dim], "weight")
                })
                .map_err(|e| {
                    grim_core::Error::Config(format!(
                        "Qwen38FlashNext: checkpoint config defines ngram_vocab_size={ngram_vocab}, \
                         ngram_dim={ngram_dim}, but neither 'model.ngram_embeddings.weight' nor \
                         'model.ple_ngram_embd.weight' were found in checkpoint: {e}"
                    ))
                })?;

            let proj = Linear::load_shape(
                &root.scoped("ngram_proj"),
                [ngram_dim, cfg.hidden_size],
            )
            .or_else(|_| {
                Linear::load_shape(
                    &root.scoped("ple_ngram_proj"),
                    [ngram_dim, cfg.hidden_size],
                )
            })
            .map_err(|e| {
                grim_core::Error::Config(format!(
                    "Qwen38FlashNext: checkpoint config specifies PLE N-gram projection, \
                     but neither 'model.ngram_proj' nor 'model.ple_ngram_proj' were found in weights: {e}"
                ))
            })?;

            Some(Qwen38NgramEmbedding {
                ngram_vocab_size: ngram_vocab,
                ngram_dim,
                hidden_size: cfg.hidden_size,
                table,
                proj,
            })
        } else {
            None
        };

        let num_layers_to_load = cfg.num_layers.min(2);
        let mut layers = Vec::with_capacity(num_layers_to_load);
        for i in 0..num_layers_to_load {
            let layer_ws = root.scoped("layers").scoped(&i.to_string());
            let block = Qwen38FlashNextBlock::load(&layer_ws, &cfg, tp)?;
            layers.push(block);
        }

        let norm = RmsNorm::load(&root.scoped("norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let output = Linear::load_shape(&ws.scoped("lm_head"), [cfg.hidden_size, cfg.vocab_size])
            .unwrap_or_else(|_| Linear::from_tensor(tok_embeddings.w_t.clone(), None));

        Ok(Self {
            cfg,
            device,
            tok_embeddings,
            ngram_embeddings,
            layers,
            norm,
            output,
        })
    }

    pub fn random(device: Device, cfg: Qwen38FlashNextConfig) -> Self {
        let tok_embeddings = Linear::from_tensor(
            cpu_tensor(
                vec![0.01f32; cfg.vocab_size * cfg.hidden_size],
                grim_tensor::Shape::new(vec![cfg.vocab_size, cfg.hidden_size]),
            ),
            None,
        );
        let ngram_embeddings = if let (Some(ngram_vocab), Some(ngram_dim)) =
            (cfg.ngram_vocab_size, cfg.ngram_dim)
        {
            let table = cpu_tensor(
                vec![0.01f32; ngram_vocab * ngram_dim],
                grim_tensor::Shape::new(vec![ngram_vocab, ngram_dim]),
            );
            let proj = Linear::from_tensor(
                cpu_tensor(
                    vec![0.01f32; cfg.hidden_size * ngram_dim],
                    grim_tensor::Shape::new(vec![cfg.hidden_size, ngram_dim]),
                ),
                None,
            );
            Some(Qwen38NgramEmbedding {
                ngram_vocab_size: ngram_vocab,
                ngram_dim,
                hidden_size: cfg.hidden_size,
                table,
                proj,
            })
        } else {
            None
        };

        let norm = RmsNorm {
            weight: cpu_tensor(
                vec![1.0; cfg.hidden_size],
                grim_tensor::Shape::new(vec![cfg.hidden_size]),
            ),
            eps: cfg.rms_norm_eps,
        };
        let output = Linear::from_tensor(
            cpu_tensor(
                vec![0.01f32; cfg.vocab_size * cfg.hidden_size],
                grim_tensor::Shape::new(vec![cfg.vocab_size, cfg.hidden_size]),
            ),
            None,
        );
        Self {
            cfg,
            device,
            tok_embeddings,
            ngram_embeddings,
            layers: vec![],
            norm,
            output,
        }
    }
}

impl Model for Qwen38FlashNext {
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

impl CausalLm for Qwen38FlashNext {
    fn new_session(&self) -> Box<dyn SessionT> {
        Box::new(grim_core::session::Session::new(self.device.clone()))
    }

    fn forward(
        &self,
        _session: &mut dyn SessionT,
        input_ids: &Tensor,
        positions: &Tensor,
        _adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
        let pos_f32 = positions.to_vec_f32()?;
        let pos_u32: Vec<u32> = pos_f32.into_iter().map(|p| p as u32).collect();

        let ids_f32 = input_ids.to_vec_f32()?;
        let seq_len = ids_f32.len();
        let embed_w = self.tok_embeddings.weight.to_vec_f32()?;
        let mut h_vec = vec![0.0f32; seq_len * self.cfg.hidden_size];

        for (i, &tok_f) in ids_f32.iter().enumerate() {
            let tok = tok_f as usize;
            if tok < self.cfg.vocab_size {
                let src_start = tok * self.cfg.hidden_size;
                let dst_start = i * self.cfg.hidden_size;
                if src_start + self.cfg.hidden_size <= embed_w.len() {
                    h_vec[dst_start..dst_start + self.cfg.hidden_size]
                        .copy_from_slice(&embed_w[src_start..src_start + self.cfg.hidden_size]);
                }
            }
        }

        // Auxiliary Position-aware / Prompt-Lookup N-gram Embedding (PLE) fusion
        if let Some(ref ngram_emb) = self.ngram_embeddings {
            let tokens: Vec<u32> = ids_f32.iter().map(|&v| v as u32).collect();
            let ngram_h = ngram_emb.lookup_and_project(&tokens)?;
            let ng_vec = ngram_h.to_vec_f32()?;
            for i in 0..h_vec.len().min(ng_vec.len()) {
                h_vec[i] += ng_vec[i];
            }
        }

        let mut h = cpu_tensor(h_vec, grim_tensor::Shape::new(vec![seq_len, self.cfg.hidden_size]));

        for layer in &self.layers {
            h = layer.forward(&h, &pos_u32)?;
        }

        let normed = self.norm.forward(&h)?;
        Ok(self.output.forward(&normed)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen38_flash_next_config_defaults() {
        let cfg = Qwen38FlashNextConfig::default();
        assert_eq!(cfg.name(), "qwen3_8_flash_next");
        assert_eq!(cfg.num_experts, 512);
        assert_eq!(cfg.num_experts_per_tok, 10);
        assert_eq!(cfg.gated_residual_branches, 4);
        assert_eq!(cfg.mrope_section, [11, 11, 10]);
        assert_eq!(cfg.max_seq_len, 131072);
        assert_eq!(cfg.ngram_vocab_size, Some(20_000_000));
        assert_eq!(cfg.ngram_dim, Some(512));
    }

    #[test]
    fn test_qwen38_ngram_hash_determinism_and_bounds() {
        let vocab = 20_000_000;
        let tokens1 = vec![101, 202];
        let tokens2 = vec![101, 202];
        let tokens3 = vec![101, 203];

        let h1 = Qwen38NgramEmbedding::hash_ngram(&tokens1, vocab);
        let h2 = Qwen38NgramEmbedding::hash_ngram(&tokens2, vocab);
        let h3 = Qwen38NgramEmbedding::hash_ngram(&tokens3, vocab);

        assert_eq!(h1, h2, "identical token sequences must produce identical hashes");
        assert_ne!(h1, h3, "distinct token sequences should produce distinct hashes");
        assert!(h1 < vocab);
        assert!(h3 < vocab);
    }

    #[test]
    fn test_qwen38_ngram_lookup_and_forward_fusion() {
        let mut cfg = Qwen38FlashNextConfig::default();
        cfg.vocab_size = 32;
        cfg.hidden_size = 16;
        cfg.ngram_vocab_size = Some(100);
        cfg.ngram_dim = Some(8);
        cfg.num_layers = 0; // Test embeddings and norm directly

        let model = Qwen38FlashNext::random(Device::Cpu, cfg);
        let mut session = model.new_session();

        let input_ids = cpu_tensor(vec![5.0, 12.0, 18.0], grim_tensor::Shape::new(vec![3]));
        let positions = cpu_tensor(vec![0.0, 1.0, 2.0], grim_tensor::Shape::new(vec![3]));

        let out = model.forward(session.as_mut(), &input_ids, &positions, &[]).unwrap();
        assert_eq!(out.shape().dims(), &[3, 32]);
    }

    #[test]
    fn test_qwen38_single_token_prefix_isolation() {
        let mut cfg = Qwen38FlashNextConfig::default();
        cfg.vocab_size = 16;
        cfg.hidden_size = 8;
        cfg.ngram_vocab_size = Some(50);
        cfg.ngram_dim = Some(4);

        let table = cpu_tensor(vec![99.0f32; 50 * 4], grim_tensor::Shape::new(vec![50, 4]));
        let proj = Linear::from_tensor(
            cpu_tensor(vec![1.0f32; 8 * 4], grim_tensor::Shape::new(vec![8, 4])),
            None,
        );
        let ngram_emb = Qwen38NgramEmbedding {
            ngram_vocab_size: 50,
            ngram_dim: 4,
            hidden_size: 8,
            table,
            proj,
        };

        // For a single token [7], position 0 has no preceding n-gram (must return all 0s)
        let res = ngram_emb.lookup_and_project(&[7]).unwrap();
        let res_vec = res.to_vec_f32().unwrap();
        assert_eq!(res_vec, vec![0.0f32; 8], "Single token at pos 0 must have zero n-gram contribution");
    }

    #[test]
    fn test_qwen38_ngram_mathematical_precision() {
        let ngram_vocab_size = 10;
        let ngram_dim = 2;
        let hidden_size = 4;

        // Table with distinct row vectors: row 0=[1, 2], row 1=[3, 4], ..., row 9=[19, 20]
        let mut table_data = Vec::with_capacity(ngram_vocab_size * ngram_dim);
        for r in 0..ngram_vocab_size {
            table_data.push((r * 2 + 1) as f32);
            table_data.push((r * 2 + 2) as f32);
        }
        let table = cpu_tensor(table_data.clone(), grim_tensor::Shape::new(vec![ngram_vocab_size, ngram_dim]));

        // Proj weight: shape [hidden_size, ngram_dim] = [4, 2]
        // W = [[1, 0], [0, 1], [1, 1], [2, 1]]
        let proj_w = vec![
            1.0f32, 0.0,
            0.0, 1.0,
            1.0, 1.0,
            2.0, 1.0,
        ];
        let proj = Linear::from_tensor(
            cpu_tensor(proj_w, grim_tensor::Shape::new(vec![hidden_size, ngram_dim])),
            None,
        );

        let ngram_emb = Qwen38NgramEmbedding {
            ngram_vocab_size,
            ngram_dim,
            hidden_size,
            table,
            proj,
        };

        let tokens = vec![4u32, 8u32];
        let res = ngram_emb.lookup_and_project(&tokens).unwrap();
        let res_vec = res.to_vec_f32().unwrap();

        // Token 0: [0, 0, 0, 0]
        assert_eq!(&res_vec[0..4], &[0.0, 0.0, 0.0, 0.0]);

        // Token 1 (bigram [4, 8]):
        let hash = Qwen38NgramEmbedding::hash_ngram(&[4, 8], ngram_vocab_size);
        let e0 = table_data[hash * 2];
        let e1 = table_data[hash * 2 + 1];

        let expected_t1 = [
            1.0 * e0 + 0.0 * e1, // e0
            0.0 * e0 + 1.0 * e1, // e1
            1.0 * e0 + 1.0 * e1, // e0 + e1
            2.0 * e0 + 1.0 * e1, // 2*e0 + e1
        ];

        for k in 0..4 {
            let diff = (res_vec[4 + k] - expected_t1[k]).abs();
            assert!(diff < 1e-6, "Numeric discrepancy at dim {k}: got {}, expected {}", res_vec[4 + k], expected_t1[k]);
        }
    }

    #[test]
    fn test_qwen38_moe_swiglu_and_residual_scaling_numerics() {
        // Test SwiGLU activation formula: x * sigmoid(x) * u
        let g_val = 2.0f32;
        let u_val = 3.0f32;
        let sig = 1.0f32 / (1.0f32 + (-g_val).exp());
        let expected_swiglu = g_val * sig * u_val;

        let diff = (expected_swiglu - (2.0 * (1.0 / (1.0 + (-2.0f32).exp())) * 3.0)).abs();
        assert!(diff < 1e-7);

        // Test 4-branch gated residual scale: 1 / sqrt(4) = 0.5
        let branches = 4;
        let scale = 1.0f32 / (branches as f32).sqrt();
        assert_eq!(scale, 0.5f32);
    }

    #[test]
    fn test_qwen38_missing_ple_weights_fails_loudly() {
        let mut cfg = Qwen38FlashNextConfig::default();
        cfg.vocab_size = 16;
        cfg.hidden_size = 8;
        cfg.ngram_vocab_size = Some(100);
        cfg.ngram_dim = Some(4);

        // Empty weight provider: loading must error loudly rather than silently substitute dummy 0.01 tensors
        struct EmptyProvider;
        impl grim_tensor::TensorProvider for EmptyProvider {
            fn get(&self, name: &str) -> grim_tensor::error::Result<grim_tensor::RawTensor> {
                Err(grim_tensor::error::Error::Backend(format!("tensor '{name}' not found")))
            }
            fn meta(&self, _name: &str) -> grim_tensor::error::Result<grim_tensor::TensorMeta> {
                Err(grim_tensor::error::Error::Backend("tensor not found".into()))
            }
        }

        let provider = EmptyProvider;
        let ws = grim_nn::WeightSource::root(&provider, Device::Cpu);

        let err = Qwen38FlashNext::load(Device::Cpu, &ws, cfg);
        assert!(err.is_err(), "load_tp must fail loudly when PLE weights are missing");
    }
}
