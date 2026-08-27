use grim_autograd::{
    AdamW, AdamWConfig, AutogradRegistry, InjectionConfig, LoRAInjectionPoint,
    LoRAInjectionRegistry, Tape, apply_and_record_lora, backward, cross_entropy_loss,
};
use grim_tensor::{Device, Shape, backend::BackendDevice};

/// Parameterized property test verifying loss decrease parity across devices.
fn run_overfit_test_for_device(device: Device) -> (f32, f32) {
    let vocab = 16usize;
    let hidden = 8usize;
    let rank = 4usize;
    let scale = 8.0f32;

    let inj_cfg = InjectionConfig {
        hidden_size: hidden,
        num_heads: 2,
        num_kv_heads: 2,
        head_dim: 4,
        intermediate_size: 16,
        vocab_size: vocab,
    };

    let mut inj_reg = LoRAInjectionRegistry::new();
    inj_reg.add(grim_autograd::LoRAInjectionConfig::new(
        LoRAInjectionPoint::Logits,
        0,
        1,
        rank,
        scale,
    ));

    let mut autograd_reg = AutogradRegistry::new(inj_cfg, inj_reg).unwrap();
    let mut optimizer = AdamW::new(AdamWConfig {
        lr: 0.1,
        ..AdamWConfig::default()
    });

    let input_ids = [0u32, 1, 2, 3];
    let targets = vec![1usize, 2, 3, 4];
    let batch = input_ids.len();

    // Create base synthetic activation tensors on target device
    let dev: Box<dyn BackendDevice> = match device {
        Device::Cpu => Box::new(grim_backend_cpu::CpuDevice::new()),
        #[cfg(feature = "rocm-mem")]
        Device::Rocm(_) => {
            if let Ok(d) = grim_backend_rocm::RocmDevice::try_new(0) {
                Box::new(d)
            } else {
                Box::new(grim_backend_cpu::CpuDevice::new())
            }
        }
        _ => Box::new(grim_backend_cpu::CpuDevice::new()),
    };

    let h_data = vec![0.5f32; batch * hidden];
    let base_logits_data = vec![0.1f32; batch * vocab];

    let mut initial_loss = 0.0f32;
    let mut final_loss = 0.0f32;

    for step in 0..10 {
        autograd_reg.zero_grads().unwrap();
        let mut tape = Tape::new();

        let h_norm_st = dev
            .from_cpu(
                &h_data,
                &Shape::new(vec![batch, hidden]),
                grim_tensor::DType::F32,
            )
            .unwrap();
        let logits_base_st = dev
            .from_cpu(
                &base_logits_data,
                &Shape::new(vec![batch, vocab]),
                grim_tensor::DType::F32,
            )
            .unwrap();

        let h_norm = grim_tensor::Tensor::new(
            std::sync::Arc::from(h_norm_st),
            Shape::new(vec![batch, hidden]),
            grim_tensor::DType::F32,
            grim_tensor::QuantProvenance::default(),
            device.clone(),
        );

        let logits_base = grim_tensor::Tensor::new(
            std::sync::Arc::from(logits_base_st),
            Shape::new(vec![batch, vocab]),
            grim_tensor::DType::F32,
            grim_tensor::QuantProvenance::default(),
            device.clone(),
        );

        let h_norm_id = tape.register(h_norm.clone());
        let logits_base_id = tape.register(logits_base.clone());

        let (logits_id, logits_out) = apply_and_record_lora(
            &autograd_reg,
            &mut tape,
            0,
            LoRAInjectionPoint::Logits,
            logits_base,
            logits_base_id,
            h_norm,
            h_norm_id,
        )
        .unwrap();

        let (loss_val, loss_grad) = cross_entropy_loss(&logits_out, &targets).unwrap();
        if step == 0 {
            initial_loss = loss_val;
        }
        final_loss = loss_val;

        backward(&tape, loss_grad, logits_id, &mut autograd_reg.params).unwrap();
        optimizer.step(&mut autograd_reg.params).unwrap();
    }

    (initial_loss, final_loss)
}

#[test]
fn test_l0_parity_cpu_overfit_loss_decreases() {
    let (initial_loss, final_loss) = run_overfit_test_for_device(Device::Cpu);
    assert!(initial_loss > 0.0, "CPU initial loss must be positive");
    assert!(
        final_loss < initial_loss,
        "CPU final loss ({final_loss}) must be strictly lower than initial loss ({initial_loss})"
    );
}

#[test]
fn test_l0_parity_rocm_device_gated() {
    if !grim_backend_rocm::gpu_test_enabled() {
        return;
    }
    let (cpu_init, cpu_final) = run_overfit_test_for_device(Device::Cpu);
    let (rocm_init, rocm_final) = run_overfit_test_for_device(Device::Rocm(0));

    assert!(
        (cpu_init - rocm_init).abs() < 1e-3,
        "Initial loss mismatch CPU vs ROCm: {cpu_init} vs {rocm_init}"
    );
    assert!(
        (cpu_final - rocm_final).abs() < 1e-3,
        "Final loss mismatch CPU vs ROCm: {cpu_final} vs {rocm_final}"
    );
}
