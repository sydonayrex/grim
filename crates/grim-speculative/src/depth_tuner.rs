//! Dynamic Speculative Speculation-Depth PID Controller.
//!
//! Automatically adjusts the speculative rollout depth $K \in [1, 5]$ online
//! based on empirical token acceptance rates to maximize generation throughput.

/// Configuration for the speculative rollout depth PID controller.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeculativeDepthPidConfig {
    /// Target acceptance rate $\alpha^*$ (default 0.75).
    pub target_acceptance_rate: f32,
    /// Proportional gain $K_p$ (default 2.0).
    pub kp: f32,
    /// Integral gain $K_i$ (default 0.2).
    pub ki: f32,
    /// Derivative gain $K_d$ (default 0.1).
    pub kd: f32,
    /// Exponential moving average smoothing factor for acceptance rate (default 0.9).
    pub ema_decay: f32,
    /// Minimum rollout depth (default 1).
    pub min_depth: usize,
    /// Maximum rollout depth (default 5).
    pub max_depth: usize,
}

impl Default for SpeculativeDepthPidConfig {
    fn default() -> Self {
        Self {
            target_acceptance_rate: 0.75,
            kp: 2.0,
            ki: 0.2,
            kd: 0.1,
            ema_decay: 0.9,
            min_depth: 1,
            max_depth: 5,
        }
    }
}

/// Online PID controller that tunes speculative decoding depth $K$.
#[derive(Debug, Clone)]
pub struct SpeculativeDepthPidController {
    config: SpeculativeDepthPidConfig,
    current_depth: usize,
    ema_acceptance: f32,
    integral_error: f32,
    prev_error: f32,
    initialized: bool,
}

impl SpeculativeDepthPidController {
    /// Create a new PID controller with configuration.
    pub fn new(config: SpeculativeDepthPidConfig) -> Self {
        let initial_depth = (config.min_depth + config.max_depth) / 2;
        Self {
            config,
            current_depth: initial_depth.max(1),
            ema_acceptance: 0.75,
            integral_error: 0.0,
            prev_error: 0.0,
            initialized: false,
        }
    }

    /// Default PID controller.
    pub fn with_default_config() -> Self {
        Self::new(SpeculativeDepthPidConfig::default())
    }

    /// Current speculation depth $K$.
    pub fn current_depth(&self) -> usize {
        self.current_depth
    }

    /// Current smoothed acceptance rate estimate.
    pub fn acceptance_rate(&self) -> f32 {
        self.ema_acceptance
    }

    /// Update controller with step observation (`accepted_tokens`, `proposed_tokens`)
    /// and return the adjusted speculation depth $K_{t+1}$.
    pub fn update(&mut self, accepted_tokens: usize, proposed_tokens: usize) -> usize {
        if proposed_tokens == 0 {
            return self.current_depth;
        }

        let step_alpha = (accepted_tokens as f32) / (proposed_tokens as f32);

        if !self.initialized {
            self.ema_acceptance = step_alpha;
            self.initialized = true;
        } else {
            self.ema_acceptance = self.config.ema_decay * self.ema_acceptance
                + (1.0 - self.config.ema_decay) * step_alpha;
        }

        // Error: positive if acceptance is above target (can speculate more),
        // negative if acceptance is below target (should speculate less).
        let error = self.ema_acceptance - self.config.target_acceptance_rate;
        self.integral_error = (self.integral_error + error).clamp(-10.0, 10.0);
        let derivative_error = error - self.prev_error;
        self.prev_error = error;

        let delta = self.config.kp * error
            + self.config.ki * self.integral_error
            + self.config.kd * derivative_error;

        let target_float = (self.current_depth as f32) + delta;
        let new_depth = target_float.round() as isize;

        let clamped_depth = new_depth.clamp(
            self.config.min_depth as isize,
            self.config.max_depth as isize,
        ) as usize;

        self.current_depth = clamped_depth;
        self.current_depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speculative_depth_pid_scaling() {
        let mut controller = SpeculativeDepthPidController::with_default_config();
        assert_eq!(controller.current_depth(), 3);

        // High acceptance (100% accepted) -> depth should scale up to max (5)
        for _ in 0..10 {
            controller.update(3, 3);
        }
        assert_eq!(controller.current_depth(), 5);

        // Low acceptance (0% accepted) -> depth should scale down to min (1)
        for _ in 0..20 {
            controller.update(0, 5);
        }
        assert_eq!(controller.current_depth(), 1);
    }
}
