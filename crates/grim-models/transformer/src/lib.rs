//! `grim-models-transformer` — Llama / Mistral / Qwen-style dense
//! transformer CausalLm implementation. Phase 1 deliverable: a real
//! (small) model that loads from a TensorProvider and runs forward
//! on the CPU backend.

pub mod block;
pub mod model;
pub mod rng;

pub use block::{LlamaBlock, LlamaConfigRefs};
pub use model::{Llama, LlamaConfig};

#[cfg(test)]
mod tests {
    use crate::{Llama, LlamaConfig};

    #[test]
    fn smoke_tiny_llama_logits() {
        let cfg = LlamaConfig {
            vocab_size: 32000,
            hidden_size: 128,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 32,
            num_layers: 2,
            intermediate_size: 384,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 256,
        };
        let model = Llama::random(cfg);
        let tok = grim_backend_cpu::cpu_tensor(vec![1.0f32], grim_tensor::Shape::new(vec![1]));
        use grim_core::CausalLm;
        use grim_core::session::Inner;
        let mut sess = Inner::new(model.device.clone());
        let logits = CausalLm::forward(&model, &mut sess, &tok, &tok).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 32000]);
        let v = logits.to_vec_f32().unwrap();
        assert!(v.iter().any(|x| x.is_finite()));
        assert!(!v.iter().all(|x| *x == 0.0));
    }
}