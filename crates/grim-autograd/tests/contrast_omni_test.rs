use grim_autograd::contrast_omni::{ContrastOmniConfig, ContrastOmniLoss};
use std::collections::HashMap;

#[test]
fn integration_contrast_omni_diagonal_frechet_matches_manual() {
    let mean1 = [0.0f32, 2.0];
    let mean2 = [3.0, 2.0];
    let cov1 = [1.0, 4.0];
    let cov2 = [4.0, 1.0];
    let dist = ContrastOmniLoss::compute_frechet_distance(&mean1, &mean2, &cov1, &cov2, 2);
    let expected =
        9.0 + ((1.0f32.sqrt() - 4.0f32.sqrt()).powi(2)) + ((4.0f32.sqrt() - 1.0f32.sqrt()).powi(2));
    assert!(
        (dist - expected).abs() < 1e-5,
        "integration frechet mismatch: got {dist}, expected {expected}"
    );
}

#[test]
fn integration_contrast_omni_hierarchical_total_non_negative() {
    let cfg = ContrastOmniConfig::default();
    let loss = ContrastOmniLoss::new(cfg);
    let features = vec![
        0.1, 0.2, 0.3, 0.15, 0.25, 0.35, // modality 0
        0.9, 0.8, 0.7, 0.85, 0.75, 0.65, // modality 1
    ];
    let modality_ids = vec![0, 0, 1, 1];
    let modality_names = vec![
        String::from("a"),
        String::from("a"),
        String::from("b"),
        String::from("b"),
    ];
    let labels = vec![0, 0, 1, 1];
    let total = loss.hierarchical_contrastive(&features, &modality_ids, &modality_names, &labels);
    assert!(
        total >= 0.0,
        "hierarchical loss must be non-negative, got {total}"
    );
    assert!(
        total.is_finite(),
        "hierarchical loss must be finite, got {total}"
    );
}

#[test]
fn integration_contrast_omni_utility_weighting_scales_output() {
    let mut weights = HashMap::new();
    weights.insert(String::from("text"), 2.0);
    let cfg = ContrastOmniConfig {
        temperature: 0.07,
        modality_weights: weights,
        hierarchy_levels: 2,
    };
    let loss = ContrastOmniLoss::new(cfg);
    let scores = vec![1.0, 2.0, 3.0];
    let tags = vec![
        String::from("text"),
        String::from("text"),
        String::from("text"),
    ];
    let base = loss.utility_weighted_contrastive(&scores, &tags, 1.0);
    let doubled = loss.utility_weighted_contrastive(&scores, &tags, 2.0);
    assert!(
        (doubled - 2.0 * base).abs() < 1e-4,
        "utility scaling failed: base={base}, doubled={doubled}"
    );
}
