//! Cosine-with-warmup learning rate schedule.

/// Cosine learning rate scheduler with linear warmup.
///
/// Ramp LR linearly from 0 to `base_lr` over `warmup_steps`,
/// then decay via cosine curve to `min_lr` across remaining steps up to `total_steps`.
#[derive(Debug, Clone, Copy)]
pub struct CosineWarmupSchedule {
    pub total_steps: usize,
    pub warmup_steps: usize,
    pub base_lr: f32,
    pub min_lr: f32,
}

impl CosineWarmupSchedule {
    /// Create a new cosine warmup schedule.
    pub fn new(total_steps: usize, warmup_steps: usize, base_lr: f32, min_lr: f32) -> Self {
        Self {
            total_steps,
            warmup_steps,
            base_lr,
            min_lr,
        }
    }

    /// Calculate learning rate at step `step` (0-indexed).
    pub fn lr_at_step(&self, step: usize) -> f32 {
        if self.warmup_steps > 0 && step <= self.warmup_steps {
            self.base_lr * (step as f32 / self.warmup_steps as f32)
        } else if step >= self.total_steps {
            self.min_lr
        } else {
            let decay_steps = (self.total_steps - self.warmup_steps).max(1) as f32;
            let current_decay_step = (step - self.warmup_steps) as f32;
            let cosine = 0.5 * (1.0 + (std::f32::consts::PI * current_decay_step / decay_steps).cos());
            self.min_lr + (self.base_lr - self.min_lr) * cosine
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warmup_reaches_base_lr_at_warmup_step() {
        let sched = CosineWarmupSchedule::new(100, 10, 1e-4, 1e-6);
        assert!((sched.lr_at_step(10) - 1e-4).abs() < 1e-7);
    }

    #[test]
    fn lr_at_final_step_is_min_lr() {
        let sched = CosineWarmupSchedule::new(100, 10, 1e-4, 1e-6);
        assert!((sched.lr_at_step(100) - 1e-6).abs() < 1e-7);
    }

    #[test]
    fn lr_is_monotone_decreasing_after_warmup() {
        let sched = CosineWarmupSchedule::new(100, 10, 1e-4, 1e-6);
        for step in 10..100 {
            assert!(sched.lr_at_step(step) >= sched.lr_at_step(step + 1));
        }
    }
}
