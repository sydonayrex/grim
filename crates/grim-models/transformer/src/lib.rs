//! `grim-models-transformer` — Llama / Mistral / Qwen-style dense
//! transformer CausalLm implementation. Phase 1 deliverable: a real
//! (small) model that loads from a TensorProvider and runs forward
//! on the CPU backend.

pub mod block;
pub mod lora;
pub mod model;
pub mod native_mtp;
pub mod rng;
pub mod lfm2;
pub mod gpt2;
pub mod gemma;
pub mod deepseek;
pub mod t5;

pub use block::{LlamaBlock, LlamaConfigRefs};
pub use lora::apply_adapters_to_logits;
pub use model::{Llama, LlamaConfig};
pub use native_mtp::{LlamaMtp, MtpDepthProvider};
pub use lfm2::{Lfm2, Lfm2Config};
pub use gpt2::{Gpt2, Gpt2Config};
pub use gemma::{Gemma, GemmaConfig};
pub use deepseek::{DeepSeek, DeepSeekConfig};
pub use t5::{T5, T5Config};

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
        let logits = CausalLm::forward(&model, &mut sess, &tok, &tok, &[]).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 32000]);
        let v = logits.to_vec_f32().unwrap();
        assert!(v.iter().any(|x| x.is_finite()));
        assert!(!v.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn smoke_llama_with_empty_adapters_matches_baseline() {
        // Running with zero adapters must produce the same numerics as
        // the no-adapter sweep — guards against the fused-LoRA path
        // accidentally perturbing the base distribution.
        let cfg = LlamaConfig {
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
        };
        let model = Llama::random(cfg);
        let tok = grim_backend_cpu::cpu_tensor(vec![1.0f32, 2.0f32], grim_tensor::Shape::new(vec![2]));
        let mut sess_a = grim_core::session::Inner::new(model.device.clone());
        let mut sess_b = grim_core::session::Inner::new(model.device.clone());
        let base = grim_core::CausalLm::forward(&model, &mut sess_a, &tok, &tok, &[]).unwrap();
        let with_zero_adapters =
            grim_core::CausalLm::forward(&model, &mut sess_b, &tok, &tok, &[]).unwrap();
        let base_v = base.to_vec_f32().unwrap();
        let same = with_zero_adapters.to_vec_f32().unwrap();
        assert_eq!(base_v, same);
    }

    #[test]
    fn lora_apply_with_one_adapter_perturbs_logit_distribution() {
        // A single non-zero LoRA must measurably shift the logits
        // (preserving the architectural §4.5 contract that adapters
        // change the per-token distribution).
        use crate::lora::apply_adapters_to_logits;
        use grim_core::model::AdapterHandle;
        let logits = grim_backend_cpu::cpu_tensor(
            (0..32).map(|i| (i as f32 + 1.0) * 0.01).collect(),
            grim_tensor::Shape::new(vec![1, 32]),
        );
        let r = 4usize;
        let hidden = 32usize;
        let adapter = AdapterHandle {
            id: 1,
            a: grim_backend_cpu::cpu_tensor(
                (0..r * hidden).map(|i| ((i as f32) - (r * hidden) as f32 / 2.0) * 0.01).collect(),
                grim_tensor::Shape::new(vec![r, hidden]),
            ),
            b: grim_backend_cpu::cpu_tensor(
                (0..32 * r).map(|i| ((i as f32) - (32 * r) as f32 / 2.0) * 0.01).collect(),
                grim_tensor::Shape::new(vec![32, r]),
            ),
            alpha: 1.0,
        };
        let new_logits = apply_adapters_to_logits(&logits, &[adapter], hidden).unwrap();
        let v = new_logits.to_vec_f32().unwrap();
        assert!(v.iter().any(|x| *x != 0.0), "adapters must perturb the zero baseline");
    }
}