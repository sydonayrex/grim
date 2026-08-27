use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct OmniGradConfig {
    pub per_layer_lr: Vec<f32>,
    pub noise_threshold: f32,
    pub phase_gate_threshold: f32,
}

impl Default for OmniGradConfig {
    fn default() -> Self {
        Self {
            per_layer_lr: Vec::new(),
            noise_threshold: 1.5,
            phase_gate_threshold: 0.5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OmniGradRouter {
    pub config: OmniGradConfig,
    pub modality_metadata: HashMap<usize, String>,
    pub phase: f32,
}

impl OmniGradRouter {
    pub fn new(config: OmniGradConfig, modality_tags: HashMap<usize, String>) -> Self {
        Self {
            config,
            modality_metadata: modality_tags,
            phase: 0.0,
        }
    }

    pub fn advance_phase(&mut self, total_steps: usize, current_step: usize) {
        if total_steps == 0 {
            self.phase = 0.0;
            return;
        }
        self.phase = current_step as f32 / total_steps as f32;
    }

    pub fn route_gradients(&self, layer_idx: usize, gradient: &mut [f32], _modality: &str) {
        if self.config.per_layer_lr.is_empty() {
            return;
        }

        let lr = self
            .config
            .per_layer_lr
            .get(layer_idx)
            .copied()
            .unwrap_or(1.0);
        for g in gradient.iter_mut() {
            *g *= lr;
        }

        let norm = gradient.iter().map(|g| g * g).sum::<f32>().sqrt();
        if norm > self.config.noise_threshold && norm > 0.0 {
            let scale = self.config.noise_threshold / norm;
            for g in gradient.iter_mut() {
                *g *= scale;
            }
        }

        if self.phase < self.config.phase_gate_threshold {
            for g in gradient.iter_mut() {
                *g = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_omnigrad_route_scales_grad() {
        let config = OmniGradConfig {
            per_layer_lr: vec![0.5, 2.0],
            noise_threshold: 100.0, // disable clipping for this test
            ..OmniGradConfig::default()
        };
        let mut router = OmniGradRouter::new(config.clone(), HashMap::new());
        router.advance_phase(10, 10);

        let mut grad = vec![1.0, -2.0, 3.0];
        router.route_gradients(1, &mut grad, "text");
        assert_eq!(grad, vec![2.0, -4.0, 6.0]);
    }

    #[test]
    fn test_omnigrad_noise_clip() {
        let config = OmniGradConfig {
            per_layer_lr: vec![1.0],
            noise_threshold: 1.0,
            phase_gate_threshold: 0.5,
        };
        let mut router = OmniGradRouter::new(config, HashMap::new());
        router.advance_phase(10, 10);

        let mut grad = vec![10.0, 10.0];
        router.route_gradients(0, &mut grad, "text");
        let norm = grad.iter().map(|g| g * g).sum::<f32>().sqrt();
        assert!(norm <= 1.0 + 1e-5);
    }

    #[test]
    fn test_omnigrad_phase_gate() {
        // Use a noise_threshold high enough so clipping doesn't interfere.
        let config = OmniGradConfig {
            per_layer_lr: vec![1.0],
            noise_threshold: 100.0,
            phase_gate_threshold: 0.5,
        };
        let mut router = OmniGradRouter::new(config, HashMap::new());
        router.advance_phase(10, 2); // phase = 2/10 = 0.2 < 0.5 → gate zeroes grads

        let mut grad = vec![1.0, -1.0];
        router.route_gradients(0, &mut grad, "text");
        assert_eq!(grad, vec![0.0, 0.0]);
    }
}
