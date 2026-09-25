use grim_core::CausalLm;
use grim_models_transformer::{Llama, LlamaConfig};
use grim_speculative::{SpeculativeCausalLm, Strategy};
use grim_tensor::Device;

#[test]
fn test_speculative_as_any_and_strategy_guard() {
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

    let model_ref: &dyn CausalLm = &spec_model;

    // 1. as_any() on dyn CausalLm downcasts to SpeculativeCausalLm
    let spec_downcast = model_ref.as_any().downcast_ref::<SpeculativeCausalLm>();
    assert!(spec_downcast.is_some(), "model_ref.as_any() must downcast to SpeculativeCausalLm");
    let spec = spec_downcast.unwrap();

    // 2. Under Strategy::Plain, strategy() check passes
    assert_eq!(spec.strategy(), Strategy::Plain);

    // 3. Inner target downcasts to Llama
    let inner_downcast = spec.inner_target().as_any().downcast_ref::<Llama>();
    assert!(inner_downcast.is_some(), "inner_target must downcast to Llama");

    // 4. Fail-closed check: Invalid downcast does not panic, returns None
    assert!(model_ref.as_any().downcast_ref::<Llama>().is_none());
    assert!(spec.inner_target().as_any().downcast_ref::<grim_models_transformer::Lfm2>().is_none());
}
