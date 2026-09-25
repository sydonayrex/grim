use grim_core::CausalLm;
use grim_models_transformer::{Llama, LlamaConfig};
use grim_speculative::{SpeculativeCausalLm, Strategy};
use grim_tensor::Device;

#[test]
fn test_speculative_inner_target_downcast() {
    let llama_cfg = LlamaConfig {
        vocab_size: 100,
        hidden_size: 64,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 16,
        num_layers: 2,
        intermediate_size: 128,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_seq_len: 256,
        partial_rotary_factor: 1.0,
        yarn: None,
    };
    let target = Box::new(Llama::random(Device::Cpu, llama_cfg));
    let spec_model = SpeculativeCausalLm::plain(target);

    assert_eq!(spec_model.strategy(), Strategy::Plain);

    // Test inner_target() accessor returns valid &dyn CausalLm
    let inner: &dyn CausalLm = spec_model.inner_target();
    let downcasted = inner.as_any().downcast_ref::<Llama>();
    assert!(downcasted.is_some(), "inner_target() must downcast to target model type");

    // Also verify target() returns the same target
    let target_ref: &dyn CausalLm = spec_model.target();
    assert!(target_ref.as_any().downcast_ref::<Llama>().is_some());
}
