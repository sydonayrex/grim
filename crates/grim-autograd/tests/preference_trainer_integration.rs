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
