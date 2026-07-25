use grim_autograd::{
    injection::{InjectionConfig, LoRAInjectionConfig, LoRAInjectionPoint, LoRAInjectionRegistry},
    ops::apply_and_record_lora,
    param::TrainableParam,
    registry::AutogradRegistry,
    tape::{Tape, TapeKind},
};
use grim_backend_cpu::cpu_tensor;
use grim_tensor::Shape;

#[test]
fn test_apply_and_record_lora_golden_mutation_resistant() {
    let model_config = InjectionConfig {
        hidden_size: 4,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: 4,
        intermediate_size: 4,
        vocab_size: 10,
    };
    let mut injection_reg = LoRAInjectionRegistry::new();
    let lora_cfg = LoRAInjectionConfig::new(LoRAInjectionPoint::QProj, 0, 0, 4, 4.0);
    injection_reg.add(lora_cfg.clone());

    let mut autograd_reg = AutogradRegistry::new(model_config, injection_reg)
        .expect("AutogradRegistry creation failed");

    // A is rank x in_features = 4 x 4
    let a_data = vec![
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    // B is out_features x rank = 4 x 4
    let b_data = vec![
        2.0, 0.0, 0.0, 0.0,
        0.0, 2.0, 0.0, 0.0,
        0.0, 0.0, 2.0, 0.0,
        0.0, 0.0, 0.0, 2.0,
    ];
    let a_param = TrainableParam::new(lora_cfg.param_id_a(), cpu_tensor(a_data, Shape::new(vec![4, 4]))).unwrap();
    let b_param = TrainableParam::new(lora_cfg.param_id_b(), cpu_tensor(b_data, Shape::new(vec![4, 4]))).unwrap();
    autograd_reg.params.insert(a_param);
    autograd_reg.params.insert(b_param);

    let mut tape = Tape::new();

    let x = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![1, 4]));
    let base = cpu_tensor(vec![0.5, 0.5, 0.5, 0.5], Shape::new(vec![1, 4]));

    let x_id = tape.register(x.clone());
    let base_id = tape.register(base.clone());

    let (out_id, out_tensor) = apply_and_record_lora(
        &autograd_reg,
        &mut tape,
        0,
        LoRAInjectionPoint::QProj,
        base.clone(),
        base_id,
        x.clone(),
        x_id,
    )
    .expect("apply_and_record_lora failed");

    // Output delta = scale (alpha/rank = 4.0/4.0 = 1.0) * (x @ A^T) @ B^T
    // x @ A^T = [1.0, 2.0, 3.0, 4.0]
    // (x @ A^T) @ B^T = [2.0, 4.0, 6.0, 8.0]
    // out = [0.5, 0.5, 0.5, 0.5] + [2.0, 4.0, 6.0, 8.0] = [2.5, 4.5, 6.5, 8.5]
    let out_vec = out_tensor.to_vec_f32().unwrap();
    let expected = vec![2.5f32, 4.5, 6.5, 8.5];
    assert_eq!(out_vec.len(), expected.len());
    for (i, (&actual, &exp)) in out_vec.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - exp).abs() < 1e-5,
            "Mismatch at element {i}: got {actual}, expected {exp}"
        );
    }

    // Verify tape recording
    assert_eq!(tape.len(), 1);
    let entry = &tape.entries()[0];
    assert_eq!(entry.kind, TapeKind::LoRAApply);
    assert_eq!(entry.output, out_id);
}
