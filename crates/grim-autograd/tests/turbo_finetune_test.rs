//! Integration tests for `grim-autograd::turbo_finetune`.
use grim_autograd::turbo_finetune::{
    PrecisionKind, TurboFinetuneConfig, TurboFinetuneScheduler, TurboStage,
};

#[test]
fn integration_turbo_finetune_scheduler_end_to_end() {
    let config = TurboFinetuneConfig {
        stages: vec![
            TurboStage {
                name: "adapter".into(),
                target_layers: vec![0, 1],
                precision: PrecisionKind::Fp16,
                duration_steps: 10,
            },
            TurboStage {
                name: "full".into(),
                target_layers: vec![0, 1, 2, 3],
                precision: PrecisionKind::Bf16,
                duration_steps: 20,
            },
            TurboStage {
                name: "adapter".into(),
                target_layers: vec![3],
                precision: PrecisionKind::Fp8,
                duration_steps: 5,
            },
        ],
        lisa_interval: 50,
        lisa_k: 0.5,
    };

    let mut scheduler = TurboFinetuneScheduler::new(config);
    let mut reported_precisions = Vec::new();
    for step in 0..40 {
        let p = scheduler.advance(40).unwrap();
        reported_precisions.push(p);
        if (step + 1) % 10 == 0 {
            scheduler.update_lisa(step, 4);
        }
    }

    assert_eq!(reported_precisions[0..10], [PrecisionKind::Fp16; 10]);
    assert_eq!(reported_precisions[10..30], [PrecisionKind::Bf16; 20]);
    assert_eq!(reported_precisions[30..35], [PrecisionKind::Fp8; 5]);
    assert_eq!(reported_precisions[35..40], [PrecisionKind::Fp8; 5]);
}
