use std::collections::HashMap;

use grim_autograd::omnigrad::{OmniGradConfig, OmniGradRouter};

#[test]
fn omnigrad_routes_synthetic_3_layer_network() {
    let config = OmniGradConfig {
        per_layer_lr: vec![0.1, 0.2, 0.3],
        noise_threshold: 2.0,
        phase_gate_threshold: 0.5,
    };
    let mut tags = HashMap::new();
    tags.insert(0, "text".into());
    tags.insert(1, "audio".into());
    tags.insert(2, "visual".into());

    let mut router = OmniGradRouter::new(config, tags);
    router.advance_phase(10, 10);

    let mut grad = vec![1.0; 4];
    router.route_gradients(1, &mut grad, "audio");
    assert_eq!(grad, vec![0.2; 4]);

    let mut noisy = vec![10.0; 4];
    router.route_gradients(0, &mut noisy, "text");
    // lr=0.1 → [1.0; 4], norm=sqrt(4)=2.0 == threshold → no clipping (uses strict >)
    let norm = noisy.iter().map(|g| g * g).sum::<f32>().sqrt();
    assert!(norm <= 2.0 + 1e-5, "norm {norm} exceeded threshold 2.0");
    for g in &noisy {
        assert!((*g - 1.0).abs() < 1e-5, "expected 1.0, got {g}");
    }

    router.advance_phase(10, 2);
    let mut early = vec![0.25, -0.25, 0.5, -0.5];
    router.route_gradients(2, &mut early, "visual");
    assert_eq!(early, vec![0.0; 4]);
}
