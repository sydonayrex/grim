//! TURBO-FINETUNE: stage-gated precision switching for parameter-efficient
//! fine-tuning. This module lives in `grim-autograd` because it owns the
//! precision abstraction used during scheduler-driven stage transitions.
use serde::{Deserialize, Serialize};
use std::fmt;

/// Training mode enumeration used by stage-transition hooks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingMode {
    /// Parameter-efficient LoRA fine-tuning.
    Lora,
    /// QLoRA with quantized base weights.
    QLoRA,
    /// Full-parameter finetuning in BF16.
    Bf16Full,
    /// Full-parameter finetuning in FP16.
    Fp16Full,
}

/// Precision kind selected by a turbo-finance stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PrecisionKind {
    Fp16,
    #[default]
    Bf16,
    Fp8,
    Fp4,
}

impl fmt::Display for PrecisionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PrecisionKind::Fp16 => write!(f, "fp16"),
            PrecisionKind::Bf16 => write!(f, "bf16"),
            PrecisionKind::Fp8 => write!(f, "fp8"),
            PrecisionKind::Fp4 => write!(f, "fp4"),
        }
    }
}

/// One stage within a turbo-finance schedule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurboStage {
    pub name: String,
    pub target_layers: Vec<usize>,
    pub precision: PrecisionKind,
    pub duration_steps: usize,
}

/// Top-level turbo-finance configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TurboFinetuneConfig {
    pub stages: Vec<TurboStage>,
    pub lisa_interval: usize,
    pub lisa_k: f32,
}

impl Default for TurboFinetuneConfig {
    fn default() -> Self {
        Self {
            stages: Vec::new(),
            lisa_interval: 50,
            lisa_k: 0.5,
        }
    }
}

/// Scheduler that walks the stage list and reports the target precision
/// for each optimizer step.
#[derive(Debug, Clone, PartialEq)]
pub struct TurboFinetuneScheduler {
    pub config: TurboFinetuneConfig,
    pub current_stage_idx: usize,
    pub step_in_stage: usize,
    pub lisa_active_layers: Vec<usize>,
}

impl TurboFinetuneScheduler {
    pub fn new(config: TurboFinetuneConfig) -> Self {
        Self {
            config,
            current_stage_idx: 0,
            step_in_stage: 0,
            lisa_active_layers: Vec::new(),
        }
    }

    /// Advance the scheduler by one optimizer step. Returns the target
    /// precision for the step just advanced into.
    pub fn advance(&mut self, _total_steps: usize) -> Option<PrecisionKind> {
        if self.config.stages.is_empty() {
            return None;
        }

        let stage = &self.config.stages[self.current_stage_idx];
        self.step_in_stage += 1;
        let precision = stage.precision;

        if self.step_in_stage >= stage.duration_steps {
            self.current_stage_idx += 1;
            self.step_in_stage = 0;
            if self.current_stage_idx >= self.config.stages.len() {
                self.current_stage_idx = self.config.stages.len() - 1;
            }
        }

        Some(precision)
    }

    /// LISA layer selection stub: round-robin activation of two layers per
    /// update interval. The two selected indices are derived from
    /// `current_step` and `lisa_k` so they move as training progresses.
    pub fn update_lisa(&mut self, current_step: usize, total_layers: usize) -> Vec<usize> {
        if total_layers == 0 {
            return Vec::new();
        }
        let a = current_step % total_layers;
        let b = ((current_step as f32 + self.config.lisa_k * total_layers as f32) as usize)
            % total_layers;
        let selected = if a == b { vec![a] } else { vec![a, b] };
        self.lisa_active_layers = selected.clone();
        selected
    }

    /// Host-side stage-transition hook. Mutates the wrapped job config
    /// before the next step based on the current stage.
    pub fn stage_transition_hook<T: StageTransitionTarget>(
        &self,
        job: &mut T,
    ) -> Result<(), String> {
        if let Some(stage) = self.config.stages.get(self.current_stage_idx) {
            match stage.name.as_str() {
                "adapter" => {
                    job.set_training_mode(TrainingMode::QLoRA);
                    Ok(())
                }
                "full" => {
                    job.set_training_mode(TrainingMode::Bf16Full);
                    Ok(())
                }
                _ => Ok(()),
            }
        } else {
            Ok(())
        }
    }
}

/// Abstraction over the job fields mutated by a stage-transition hook.
///
/// Implemented by `grim-garage::jobs::TrainingJob` so `grim-autograd`
/// stays independent of the garage crate.
pub trait StageTransitionTarget {
    fn training_mode(&self) -> TrainingMode;
    fn set_training_mode(&mut self, mode: TrainingMode);
    fn optimizer(&self) -> crate::OptimizerKind;
    fn set_optimizer(&mut self, optimizer: crate::OptimizerKind);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal test double for `StageTransitionTarget`.
    struct FakeJob {
        mode: TrainingMode,
        optimizer: crate::OptimizerKind,
    }

    impl StageTransitionTarget for FakeJob {
        fn training_mode(&self) -> TrainingMode {
            self.mode
        }
        fn set_training_mode(&mut self, mode: TrainingMode) {
            self.mode = mode;
        }
        fn optimizer(&self) -> crate::OptimizerKind {
            self.optimizer
        }
        fn set_optimizer(&mut self, optimizer: crate::OptimizerKind) {
            self.optimizer = optimizer;
        }
    }

    #[test]
    fn test_turbo_stage_transition() {
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
            ],
            lisa_interval: 50,
            lisa_k: 0.5,
        };

        let mut scheduler = TurboFinetuneScheduler::new(config);
        for _ in 0..10 {
            let p = scheduler.advance(30);
            assert_eq!(p, Some(PrecisionKind::Fp16));
        }
        let p = scheduler.advance(30);
        assert_eq!(p, Some(PrecisionKind::Bf16));
    }

    #[test]
    fn test_turbo_precision_switching() {
        let config = TurboFinetuneConfig {
            stages: vec![
                TurboStage {
                    name: "adapter".into(),
                    target_layers: vec![0],
                    precision: PrecisionKind::Fp8,
                    duration_steps: 5,
                },
                TurboStage {
                    name: "full".into(),
                    target_layers: vec![0, 1],
                    precision: PrecisionKind::Fp4,
                    duration_steps: 5,
                },
            ],
            lisa_interval: 50,
            lisa_k: 0.5,
        };

        let mut scheduler = TurboFinetuneScheduler::new(config);
        for step in 0..12 {
            let p = scheduler.advance(12);
            if step < 5 {
                assert_eq!(p, Some(PrecisionKind::Fp8));
            } else {
                assert_eq!(p, Some(PrecisionKind::Fp4));
            }
        }
    }

    #[test]
    fn test_turbo_lisa_selection() {
        let config = TurboFinetuneConfig {
            stages: vec![TurboStage {
                name: "adapter".into(),
                target_layers: vec![0, 1, 2],
                precision: PrecisionKind::Fp16,
                duration_steps: 100,
            }],
            lisa_interval: 50,
            lisa_k: 0.5,
        };

        let mut scheduler = TurboFinetuneScheduler::new(config);
        let layers = scheduler.update_lisa(7, 8);
        assert_eq!(layers, vec![7, (7 + (0.5 * 8.0) as usize) % 8]);
    }

    #[test]
    fn test_turbo_stage_transition_hook_adapter() {
        let config = TurboFinetuneConfig {
            stages: vec![TurboStage {
                name: "adapter".into(),
                target_layers: vec![0],
                precision: PrecisionKind::Fp16,
                duration_steps: 100,
            }],
            lisa_interval: 50,
            lisa_k: 0.5,
        };

        let scheduler = TurboFinetuneScheduler::new(config);
        let mut job = FakeJob {
            mode: TrainingMode::Lora,
            optimizer: crate::OptimizerKind::AdamW,
        };
        scheduler.stage_transition_hook(&mut job).unwrap();
        assert_eq!(job.mode, TrainingMode::QLoRA);
    }

    #[test]
    fn test_turbo_stage_transition_hook_full() {
        let config = TurboFinetuneConfig {
            stages: vec![TurboStage {
                name: "full".into(),
                target_layers: vec![0, 1, 2],
                precision: PrecisionKind::Bf16,
                duration_steps: 100,
            }],
            lisa_interval: 50,
            lisa_k: 0.5,
        };

        let scheduler = TurboFinetuneScheduler::new(config);
        let mut job = FakeJob {
            mode: TrainingMode::Lora,
            optimizer: crate::OptimizerKind::AdamW,
        };
        scheduler.stage_transition_hook(&mut job).unwrap();
        assert_eq!(job.mode, TrainingMode::Bf16Full);
    }
}
