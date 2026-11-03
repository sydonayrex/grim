//! Integration tests for PreferenceTrainer across DPO, KTO, SimPO, ORPO, and GRPO.

use grim_autograd::{PreferenceKind, PreferenceStepConfig, PreferenceTrainer};

#[test]
fn test_preference_trainer_all_algorithms() {
    let trainer = PreferenceTrainer::new(PreferenceStepConfig {
        beta: 0.1,
        orpo_lambda: 0.1,
        simpo_gamma: 0.5,
        kto_desirable_weight: 1.0,
        kto_undesirable_weight: 1.0,
        grpo_epsilon: 0.2,
    });

    let chosen_logps = vec![-1.2, -0.8];
    let rejected_logps = vec![-2.5, -3.1];
    let ref_chosen = vec![-1.5, -1.0];
    let ref_rejected = vec![-2.0, -2.2];
    let chosen_lens = vec![10, 12];
    let rejected_lens = vec![14, 15];
    let rewards = vec![1.5, 2.0];

    // 1. DPO
    let (dpo_loss, dpo_cw, dpo_rw) = trainer
        .compute_preference_step(
            PreferenceKind::Dpo,
            &chosen_logps,
            &rejected_logps,
            &ref_chosen,
            &ref_rejected,
            &chosen_lens,
            &rejected_lens,
            None,
        )
        .unwrap();
    assert!(dpo_loss > 0.0);
    assert_eq!(dpo_cw.len(), 2);
    assert!(dpo_cw[0] < 0.0);
    assert!(dpo_rw[0] > 0.0);

    // 2. ORPO
    let (orpo_loss, orpo_cw, _orpo_rw) = trainer
        .compute_preference_step(
            PreferenceKind::Orpo,
            &chosen_logps,
            &rejected_logps,
            &ref_chosen,
            &ref_rejected,
            &chosen_lens,
            &rejected_lens,
            None,
        )
        .unwrap();
    assert!(orpo_loss > 0.0);
    assert_eq!(orpo_cw.len(), 2);

    // 3. SimPO
    let (simpo_loss, simpo_cw, _simpo_rw) = trainer
        .compute_preference_step(
            PreferenceKind::Simpo,
            &chosen_logps,
            &rejected_logps,
            &ref_chosen,
            &ref_rejected,
            &chosen_lens,
            &rejected_lens,
            None,
        )
        .unwrap();
    assert!(simpo_loss > 0.0);
    assert_eq!(simpo_cw.len(), 2);

    // 4. KTO
    let (kto_loss, kto_cw, _kto_rw) = trainer
        .compute_preference_step(
            PreferenceKind::Kto,
            &chosen_logps,
            &rejected_logps,
            &ref_chosen,
            &ref_rejected,
            &chosen_lens,
            &rejected_lens,
            None,
        )
        .unwrap();
    assert!(kto_loss > 0.0);
    assert_eq!(kto_cw.len(), 2);

    // 5. GRPO
    let (grpo_loss, grpo_cw, _grpo_rw) = trainer
        .compute_preference_step(
            PreferenceKind::Grpo,
            &chosen_logps,
            &rejected_logps,
            &ref_chosen,
            &ref_rejected,
            &chosen_lens,
            &rejected_lens,
            Some(&rewards),
        )
        .unwrap();
    assert_eq!(grpo_cw.len(), 2);
    let _ = grpo_loss;
}

#[test]
fn test_log_softmax_vjp_cpu_vs_reference() {
    let seq_len = 16;
    let vocab_size = 128;
    let ignore_index = 0u32;
    let logp_grad = -0.5f32;

    let mut logits_vec = vec![0.0f32; seq_len * vocab_size];
    for i in 0..logits_vec.len() {
        logits_vec[i] = ((i as f32 * 0.05).sin() * 2.0).clamp(-10.0, 10.0);
    }
    let targets: Vec<u32> = (0..seq_len as u32).map(|i| if i % 4 == 0 { ignore_index } else { (i * 7) % vocab_size as u32 }).collect();

    let grad_cpu = PreferenceTrainer::compute_log_softmax_vjp(
        &logits_vec,
        &targets,
        vocab_size,
        logp_grad,
        ignore_index,
    );

    let logits_tensor = grim_backend_cpu::cpu_tensor(
        logits_vec,
        grim_tensor::Shape::new(vec![seq_len, vocab_size]),
    );

    let grad_tensor = PreferenceTrainer::compute_log_softmax_vjp_tensor(
        &logits_tensor,
        &targets,
        vocab_size,
        logp_grad,
        ignore_index,
    ).expect("vjp tensor");

    let grad_tensor_vec = grad_tensor.to_vec_f32().expect("to_vec_f32");
    assert_eq!(grad_cpu.len(), grad_tensor_vec.len());

    let mut max_diff: f32 = 0.0;
    for (c, t) in grad_cpu.iter().zip(grad_tensor_vec.iter()) {
        let diff = (c - t).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    assert!(max_diff < 1e-6, "max_diff {max_diff} exceeds 1e-6");
}
