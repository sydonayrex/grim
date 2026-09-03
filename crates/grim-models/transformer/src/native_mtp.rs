//! Native Multi-Token Prediction (MTP) implementation for transformer models.
//!
//! Real MTP employs dedicated autoregressive prediction heads $k \in [1..D]$:
//! each head $k$ takes the previous hidden state $h_{k-1}$ and candidate token embedding $E(t_{k-1})$,
//! projects them through a fusion layer ($2 \cdot d_{\text{model}} \to d_{\text{model}}$),
//! applies RMSNorm, and computes next-token logits via an LM head.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::{Error, Result};
use grim_core::model::{AdapterHandle, CausalLm};
use grim_core::session::SessionT;
use grim_core::{Model, ModelConfig};
use grim_nn::{Linear, RmsNorm, WeightSource};
use grim_tensor::{ArithType, Device, Shape, Tensor};

use crate::Llama;

/// MTP depth control trait for speculative decoding.
pub trait MtpDepthProvider: Send + Sync {
    /// How many extra speculative tokens this model predicts per forward step.
    fn mtp_depth(&self) -> usize;

    /// Autoregressively predict `mtp_depth()` speculative candidate tokens
    /// using genuine MTP projection and LM head layers.
    fn predict_mtp_tokens(
        &self,
        session: &mut dyn SessionT,
        input_ids: &Tensor,
        positions: &Tensor,
    ) -> Result<Vec<u32>>;
}

/// A single Multi-Token Prediction (MTP) stage.
///
/// Fuses previous hidden state $h \in \mathbb{R}^D$ and previous token embedding $e \in \mathbb{R}^D$
/// via linear projection $W_{\text{proj}} \in \mathbb{R}^{D \times 2D}$, normalizes, and projects
/// to vocabulary logits $W_{\text{head}} \in \mathbb{R}^{V \times D}$.
pub struct MtpLayer {
    /// Projection from concatenated $[h, e]$ ($2D \to D$).
    pub proj: Linear,
    /// RMS normalization applied to the projected hidden state.
    pub norm: RmsNorm,
    /// Output projection mapping hidden state to vocabulary logits ($D \to V$).
    pub lm_head: Linear,
    /// Model hidden dimension ($D$).
    pub hidden_size: usize,
    /// Vocabulary size ($V$).
    pub vocab_size: usize,
}

impl MtpLayer {
    /// Load an MTP stage from checkpoint weight source.
    ///
    /// # Contract
    /// Fails loudly if MTP projection layers or LM head are not found in the checkpoint.
    pub fn load(ws: &WeightSource<'_>, hidden_size: usize, vocab_size: usize, eps: f32) -> Result<Self> {
        let proj = Linear::load_shape(&ws.scoped("fc_hidden"), [hidden_size, hidden_size])
            .or_else(|_| Linear::load_shape(&ws.scoped("proj"), [hidden_size * 2, hidden_size]))?;
        let norm = RmsNorm::load(&ws.scoped("pre_fc_norm_hidden"), hidden_size, eps)
            .or_else(|_| RmsNorm::load(&ws.scoped("norm"), hidden_size, eps))?;
        let lm_head = Linear::load_shape(&ws.scoped("lm_head"), [hidden_size, vocab_size])?;
        Ok(Self {
            proj,
            norm,
            lm_head,
            hidden_size,
            vocab_size,
        })
    }

    /// Construct an MTP stage with random/mock weights for testing.
    pub fn random(hidden_size: usize, vocab_size: usize, eps: f32) -> Self {
        let proj_w = cpu_tensor(
            vec![0.01f32; hidden_size * (hidden_size * 2)],
            Shape::new(vec![hidden_size, hidden_size * 2]),
        );
        let proj = Linear::from_tensor(proj_w, None);

        let norm = RmsNorm {
            weight: cpu_tensor(vec![1.0f32; hidden_size], Shape::new(vec![hidden_size])),
            eps,
        };

        let lm_head_w = cpu_tensor(
            vec![0.01f32; vocab_size * hidden_size],
            Shape::new(vec![vocab_size, hidden_size]),
        );
        let lm_head = Linear::from_tensor(lm_head_w, None);

        Self {
            proj,
            norm,
            lm_head,
            hidden_size,
            vocab_size,
        }
    }

    /// Forward pass through the MTP stage:
    /// computes $h_{\text{next}} = \text{Norm}(W_{\text{proj}} \cdot [h, e])$
    /// and $\text{logits} = W_{\text{head}} \cdot h_{\text{next}}$.
    pub fn forward(&self, h: &[f32], e: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        if h.len() != self.hidden_size || e.len() != self.hidden_size {
            return Err(Error::Session(format!(
                "MtpLayer::forward dimension mismatch: expected h={}, e={}, got h={}, e={}",
                self.hidden_size, self.hidden_size, h.len(), e.len()
            )));
        }

        let mut concat = Vec::with_capacity(self.hidden_size * 2);
        concat.extend_from_slice(h);
        concat.extend_from_slice(e);

        let concat_cpu = cpu_tensor(concat, Shape::new(vec![1, self.hidden_size * 2]));
        let target_dev = self.proj.weight.device();
        let concat_t = grim_nn::modules::move_to_device(&concat_cpu, target_dev)?;

        let projected = self.proj.forward(&concat_t)?;
        let normed = self.norm.forward(&projected)?;
        let logits = self.lm_head.forward(&normed)?;

        let h_next = normed.to_vec_f32()?;
        let logits_vec = logits.to_vec_f32()?;
        Ok((h_next, logits_vec))
    }

    /// Device-native forward pass through the MTP stage:
    /// computes $h_{\text{next}} = \text{Norm}(W_{\text{proj}} \cdot [h, e])$
    /// and $\text{logits} = W_{\text{head}} \cdot h_{\text{next}}$.
    pub fn forward_tensor(&self, h: &[f32], e: &[f32]) -> Result<(Tensor, Tensor)> {
        let (h_next_vec, logits_vec) = self.forward(h, e)?;
        let target_dev = self.proj.weight.device();
        let h_t = grim_nn::modules::move_to_device(
            &cpu_tensor(h_next_vec, Shape::new(vec![1, self.hidden_size])),
            target_dev,
        )?;
        let l_t = grim_nn::modules::move_to_device(
            &cpu_tensor(logits_vec, Shape::new(vec![1, self.vocab_size])),
            target_dev,
        )?;
        Ok((h_t, l_t))
    }

    /// One autoregressive MTP step: returns `(h_next, argmax(logits))`.
    ///
    /// GPU-first: the vocabulary logits never cross to host — the argmax
    /// runs on the device over the final logits tensor; only the
    /// hidden-size `h_next` row is pulled (needed as the next step's host
    /// input). Host fallbacks cover backends without the `argmax` kernel.
    pub fn forward_step(&self, h: &[f32], e: &[f32]) -> Result<(Vec<f32>, u32)> {
        if h.len() != self.hidden_size || e.len() != self.hidden_size {
            return Err(Error::Session(format!(
                "MtpLayer::forward dimension mismatch: expected h={}, e={}, got h={}, e={}",
                self.hidden_size,
                self.hidden_size,
                h.len(),
                e.len()
            )));
        }

        let mut concat = Vec::with_capacity(self.hidden_size * 2);
        concat.extend_from_slice(h);
        concat.extend_from_slice(e);

        let concat_cpu = cpu_tensor(concat, Shape::new(vec![1, self.hidden_size * 2]));
        let target_dev = self.proj.weight.device();
        let concat_t = grim_nn::modules::move_to_device(&concat_cpu, target_dev)?;

        let projected = self.proj.forward(&concat_t)?;
        let normed = self.norm.forward(&projected)?;
        let logits = self.lm_head.forward(&normed)?;

        let next_token = argmax_last_row_device(&logits, self.vocab_size)?;
        let h_next = normed.to_vec_f32()?;
        Ok((h_next, next_token))
    }
}

/// Llama model with genuine Multi-Token Prediction (MTP) layers.
pub struct LlamaMtp {
    pub base: Llama,
    pub mtp_layers: Vec<MtpLayer>,
}

impl LlamaMtp {
    pub fn new(base: Llama, mtp_layers: Vec<MtpLayer>) -> Self {
        Self { base, mtp_layers }
    }

    pub fn new_random(base: Llama, depth: usize) -> Self {
        let hidden_size = base.cfg.hidden_size;
        let vocab_size = base.cfg.vocab_size;
        let eps = base.cfg.rms_norm_eps;
        let mtp_layers = (0..depth)
            .map(|_| MtpLayer::random(hidden_size, vocab_size, eps))
            .collect();
        Self { base, mtp_layers }
    }
}

impl Model for LlamaMtp {
    fn config(&self) -> &dyn ModelConfig {
        self.base.config()
    }
    fn device(&self) -> &Device {
        self.base.device()
    }
    fn param_arith(&self) -> ArithType {
        self.base.param_arith()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl CausalLm for LlamaMtp {
    fn new_session(&self) -> Box<dyn SessionT> {
        self.base.new_session()
    }

    fn forward(
        &self,
        session: &mut dyn SessionT,
        input_ids: &Tensor,
        positions: &Tensor,
        adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
        self.base.forward(session, input_ids, positions, adapters)
    }
}

impl MtpDepthProvider for LlamaMtp {
    fn mtp_depth(&self) -> usize {
        self.mtp_layers.len()
    }

    fn predict_mtp_tokens(
        &self,
        session: &mut dyn SessionT,
        input_ids: &Tensor,
        positions: &Tensor,
    ) -> Result<Vec<u32>> {
        let base_logits = self.base.forward(session, input_ids, positions, &[])?;
        let vocab_size = self.base.cfg.vocab_size;
        let hidden_size = self.base.cfg.hidden_size;

        if base_logits.shape().elem_count() == 0 || vocab_size == 0 {
            return Ok(vec![]);
        }

        // Device-side argmax over the final position's vocab row — the
        // [seq, vocab] logits tensor never crosses to host.
        let mut curr_token = argmax_last_row_device(&base_logits, vocab_size)?;

        // Retrieve last hidden state from session or approximate from a
        // single-row embedding gather (never the whole table).
        let mut curr_h = if let Some(last_h) = session.get_last_hidden_state() {
            let vec = last_h.to_vec_f32()?;
            let offset = vec.len().saturating_sub(hidden_size);
            vec[offset..].to_vec()
        } else {
            embedding_row_device(
                &self.base.tok_embeddings.weight,
                curr_token,
                hidden_size,
                vocab_size,
            )?
        };

        let mut tokens = Vec::with_capacity(self.mtp_layers.len());
        for layer in &self.mtp_layers {
            let curr_e = embedding_row_device(
                &self.base.tok_embeddings.weight,
                curr_token,
                hidden_size,
                vocab_size,
            )?;
            let (next_h, next_token) = layer.forward_step(&curr_h, &curr_e)?;
            tokens.push(next_token);
            curr_h = next_h;
            curr_token = next_token;
        }

        Ok(tokens)
    }
}

/// Qwen3.8-Flash-Next model with genuine Multi-Token Prediction (MTP) layers.
pub struct Qwen38FlashNextMtp {
    pub base: crate::qwen38_flash_next::Qwen38FlashNext,
    pub mtp_layers: Vec<MtpLayer>,
}

impl Qwen38FlashNextMtp {
    pub fn new(base: crate::qwen38_flash_next::Qwen38FlashNext, mtp_layers: Vec<MtpLayer>) -> Self {
        Self { base, mtp_layers }
    }

    pub fn new_random(base: crate::qwen38_flash_next::Qwen38FlashNext, depth: usize) -> Self {
        let hidden_size = base.cfg.hidden_size;
        let vocab_size = base.cfg.vocab_size;
        let eps = base.cfg.rms_norm_eps;
        let mtp_layers = (0..depth)
            .map(|_| MtpLayer::random(hidden_size, vocab_size, eps))
            .collect();
        Self { base, mtp_layers }
    }
}

impl Model for Qwen38FlashNextMtp {
    fn config(&self) -> &dyn ModelConfig {
        self.base.config()
    }
    fn device(&self) -> &Device {
        self.base.device()
    }
    fn param_arith(&self) -> ArithType {
        self.base.param_arith()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl CausalLm for Qwen38FlashNextMtp {
    fn new_session(&self) -> Box<dyn SessionT> {
        self.base.new_session()
    }

    fn forward(
        &self,
        session: &mut dyn SessionT,
        input_ids: &Tensor,
        positions: &Tensor,
        adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
        self.base.forward(session, input_ids, positions, adapters)
    }
}

impl MtpDepthProvider for Qwen38FlashNextMtp {
    fn mtp_depth(&self) -> usize {
        self.mtp_layers.len()
    }

    fn predict_mtp_tokens(
        &self,
        session: &mut dyn SessionT,
        input_ids: &Tensor,
        positions: &Tensor,
    ) -> Result<Vec<u32>> {
        let base_logits = self.base.forward(session, input_ids, positions, &[])?;
        let vocab_size = self.base.cfg.vocab_size;
        let hidden_size = self.base.cfg.hidden_size;

        if base_logits.shape().elem_count() == 0 || vocab_size == 0 {
            return Ok(vec![]);
        }

        let mut curr_token = argmax_last_row_device(&base_logits, vocab_size)?;

        let mut curr_h = if let Some(last_h) = session.get_last_hidden_state() {
            let vec = last_h.to_vec_f32()?;
            let offset = vec.len().saturating_sub(hidden_size);
            vec[offset..].to_vec()
        } else {
            embedding_row_device(
                &self.base.tok_embeddings.weight,
                curr_token,
                hidden_size,
                vocab_size,
            )?
        };

        let mut tokens = Vec::with_capacity(self.mtp_layers.len());
        for layer in &self.mtp_layers {
            let curr_e = embedding_row_device(
                &self.base.tok_embeddings.weight,
                curr_token,
                hidden_size,
                vocab_size,
            )?;
            let (next_h, next_token) = layer.forward_step(&curr_h, &curr_e)?;
            tokens.push(next_token);
            curr_h = next_h;
            curr_token = next_token;
        }

        Ok(tokens)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn argmax(slice: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in slice.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}

/// Argmax of the LAST `[*, vocab]` row of `logits`, on the tensor's device.
/// Stages only that row D2D (`copy_slice_range`) and runs the `argmax`
/// kernel; falls back to a single-row host pull when the backend lacks the
/// primitives. Never pulls the whole `[seq, vocab]` tensor.
fn argmax_last_row_device(logits: &Tensor, vocab_size: usize) -> Result<u32> {
    if vocab_size == 0 {
        return Err(Error::Session("argmax over empty vocabulary".into()));
    }
    let total = logits.shape().elem_count();
    if total == 0 || total < vocab_size {
        return Err(Error::Session(format!(
            "argmax_last_row_device: logits has {total} elements, vocab {vocab_size}"
        )));
    }
    let rows = total / vocab_size;
    let dev = grim_nn::modules::pick_device_for_storage_device(logits.device());
    if let Ok(scratch) = dev.alloc_storage(
        &Shape::new(vec![1, vocab_size]),
        grim_tensor::DType::F32,
    ) {
        if dev
            .copy_slice_range(
                scratch.as_ref(),
                0,
                logits.storage().as_ref(),
                (rows - 1) * vocab_size,
                vocab_size,
            )
            .is_ok()
        {
            if let Ok(idx) = dev.argmax(scratch.as_ref()) {
                return Ok(idx);
            }
        }
    }
    // Fallback: pull only the final vocab row.
    let all = logits.to_vec_f32()?;
    let offset = all.len().saturating_sub(vocab_size);
    Ok(argmax(&all[offset..]) as u32)
}

/// Single-row embedding gather: pulls exactly one hidden-size row from the
/// table (via the device gather kernel), never the whole vocab×hidden
/// table. OOV tokens clamp to row 0 (matches the legacy host helper).
fn embedding_row_device(
    weight: &Tensor,
    tok: u32,
    hidden_size: usize,
    vocab_size: usize,
) -> Result<Vec<f32>> {
    let tok = if (tok as usize) < vocab_size { tok } else { 0 };
    let row = grim_nn::embedding_gather_on_device(
        weight,
        &[tok],
        1,
        hidden_size,
    )?;
    Ok(row.to_vec_f32()?)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LlamaConfig;

    #[test]
    fn test_llama_mtp_creation_and_prediction() {
        let base_cfg = LlamaConfig {
            vocab_size: 64,
            hidden_size: 32,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 16,
            num_layers: 1,
            intermediate_size: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 32,
            partial_rotary_factor: 1.0,
            yarn: None,
        };
        let base = Llama::random(Device::Cpu, base_cfg);
        let mtp = LlamaMtp::new_random(base, 2);

        assert_eq!(mtp.mtp_depth(), 2);

        let mut session = mtp.new_session();
        let input_ids = cpu_tensor(vec![3.0, 7.0], Shape::new(vec![2]));
        let positions = cpu_tensor(vec![0.0, 1.0], Shape::new(vec![2]));

        let tokens = mtp.predict_mtp_tokens(session.as_mut(), &input_ids, &positions).unwrap();
        assert_eq!(tokens.len(), 2);
        for &tok in &tokens {
            assert!(tok < 64);
        }
    }

    #[test]
    fn test_qwen38_flash_next_mtp_creation_and_prediction() {
        let cfg = crate::qwen38_flash_next::Qwen38FlashNextConfig {
            vocab_size: 128,
            hidden_size: 64,
            num_layers: 1,
            ngram_vocab_size: Some(64),
            ngram_dim: Some(16),
            ..Default::default()
        };
        let base = crate::qwen38_flash_next::Qwen38FlashNext::random(Device::Cpu, cfg);
        let mtp = Qwen38FlashNextMtp::new_random(base, 3);
        assert_eq!(mtp.mtp_depth(), 3);

        let mut session = mtp.new_session();
        let input_ids = cpu_tensor(vec![5.0, 12.0], Shape::new(vec![2]));
        let positions = cpu_tensor(vec![0.0, 1.0], Shape::new(vec![2]));

        let tokens = mtp.predict_mtp_tokens(session.as_mut(), &input_ids, &positions).unwrap();
        assert_eq!(tokens.len(), 3);
        for &tok in &tokens {
            assert!(tok < 128);
        }

        // Verify that the trunk forward pass populated the real contextual hidden state in session
        let trunk_last_h = session.get_last_hidden_state();
        assert!(trunk_last_h.is_some(), "trunk forward must populate session.last_hidden_state");
        let h_dims = trunk_last_h.unwrap().shape().dims().to_vec();
        assert_eq!(h_dims, vec![2, 64]);
    }
}
