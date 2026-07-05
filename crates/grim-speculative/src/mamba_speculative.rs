//! Speculative decoding loop tailored for Mamba sequence/SSM architectures.
//!
//! Handles saving, restoring, and rolling back internal Mamba/SSM state vectors
//! (SSM states and convolution state FIFOs) upon draft token rejection.

use grim_core::error::Result;

/// Cache containing the historical SSM and convolution states for rollback.
#[derive(Clone, Debug)]
pub struct MambaStepState {
    pub step: usize,
    pub ssm_state: Vec<f32>,
    pub conv_state: Vec<f32>,
}

pub struct MambaSpeculativeEngine {
    state_history: Vec<MambaStepState>,
    d_model: usize,
    d_state: usize,
    d_conv: usize,
}

impl MambaSpeculativeEngine {
    pub fn new(d_model: usize, d_state: usize, d_conv: usize) -> Self {
        Self {
            state_history: Vec::new(),
            d_model,
            d_state,
            d_conv,
        }
    }

    /// Record the state before running step `step`.
    pub fn record_state(&mut self, step: usize, ssm_state: &[f32], conv_state: &[f32]) {
        self.state_history.push(MambaStepState {
            step,
            ssm_state: ssm_state.to_vec(),
            conv_state: conv_state.to_vec(),
        });
    }

    /// Reset/rollback the active state back to the recorded state at step `target_step`.
    pub fn rollback_to(&mut self, target_step: usize) -> Result<MambaStepState> {
        while let Some(last) = self.state_history.last() {
            if last.step > target_step {
                self.state_history.pop();
            } else {
                break;
            }
        }
        self.state_history
            .last()
            .cloned()
            .ok_or_else(|| grim_core::Error::Session(format!(
                "No Mamba state recorded for step <= {}",
                target_step
            )))
    }

    /// Clear all state history.
    pub fn clear(&mut self) {
        self.state_history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mamba_speculative_rollback() {
        let mut engine = MambaSpeculativeEngine::new(16, 8, 4);
        
        // Record states for steps 0, 1, 2
        engine.record_state(0, &[0.0; 128], &[0.0; 64]);
        engine.record_state(1, &[1.0; 128], &[1.0; 64]);
        engine.record_state(2, &[2.0; 128], &[2.0; 64]);

        // Rollback to step 1
        let state = engine.rollback_to(1).unwrap();
        assert_eq!(state.step, 1);
        assert_eq!(state.ssm_state[0], 1.0);

        // State for step 2 should be discarded
        assert_eq!(engine.state_history.len(), 2);
    }
}
