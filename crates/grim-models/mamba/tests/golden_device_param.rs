use grim_core::model::Model;
use grim_tensor::Device;

// Minimal config for test-size Mamba model.
fn mamba_config() -> grim_models_mamba::MambaConfig {
    grim_models_mamba::MambaConfig {
        vocab_size: 64,
        hidden_size: 16,
        d_state: 4,
        d_conv: 2,
        d_inner: 32,
        num_layers: 1,
        conv_kernel: 4,
        rms_norm_eps: 1e-5,
    }
}

/// Mamba::random(Device::Cpu, ...) returns a model on Device::Cpu.
#[test]
fn golden_device_random_ctor_takes_device() {
    let model = grim_models_mamba::Mamba::random(Device::Cpu, mamba_config());
    assert_eq!(*model.device(), Device::Cpu);
}
