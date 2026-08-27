//! RoPE scaling methods for long-context fine-tuning (WI-T10 / Phase 6.2).
//!
//! RoPE frequency scaling extends the effective context window without
//! retraining by modifying the rotational frequency base. The supported
//! methods mirror Unsloth's `RopeScalingMethod`:
//!
//! - `None`: no scaling — use the model's native `rope_theta`.
//! - `Linear(factor)`: linear interpolation — `θ_i' = θ_i / factor^{1/head_dim}`,
//!   equivalently `effective_base = base * factor^{1/head_dim}`.
//! - `Llama3(factor)`: Llama 3 style piecewise interpolation with the
//!   default `low_freq_factor`/`high_freq_factor` folding into a single
//!   effective base: `effective_base = base * factor` (corrected from the
//!   earlier `(8/head_dim)^2/2` approximation, which produced ~1.6% rather
//!   than the intended ~8x shift for `head_dim=128, factor=8`).
//! - `LongRoPE(factor)`: NTK-aware frequency mixing (Kim et al.) where
//!   frequencies below a critical threshold are interpolated and those above
//!   are extrapolated. Reduced to a scalar effective base for backends whose
//!   `rope` API takes a single base value: `base * factor^{head_dim/(head_dim-2)}`.
//! - `YaRN(factor, ...)`: NTK-by-parts with magnitude correction; the scalar
//!   base uses the same NTK-aware exponent. `mscale`/`mscale_all` are retained
//!   in the enum for backends that support magnitude correction.
//! - `Dynamic`: theta-shift placeholder — delegates to `base` (future NTK-alpha
//!   dynamic routing).

use serde::{Deserialize, Serialize};

/// RoPE frequency scaling method (Phase 6.2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum RopeScalingMethod {
    /// No scaling — use the model's native `rope_theta`.
    #[default]
    None,
    /// Linear interpolation: `effective_base = base * factor^{1/head_dim}`.
    Linear { factor: f32 },
    /// Llama 3 style piecewise linear interpolation with a `γ` factor.
    Llama3 { factor: f32 },
    /// LongRoPE NTK-aware frequency mixing.
    LongRoPE { factor: f32 },
    /// YaRN NTK-by-parts with magnitude correction.
    YaRN {
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        mscale: f32,
        mscale_all: bool,
    },
    /// Dynamic theta-shift placeholder.
    Dynamic { type_: String },
}

/// Compute the effective rotational frequency base for `method`.
///
/// `base` is the model's native `rope_theta`; `head_dim` is the per-head
/// embedding dimension. Backends whose `rope` implementation accepts a single
/// base value should use the returned scalar.
pub fn scaling_base(method: &RopeScalingMethod, base: f32, head_dim: usize) -> f32 {
    let dim = head_dim.max(1) as f32;
    match method {
        RopeScalingMethod::None => base,
        RopeScalingMethod::Linear { factor } => base * factor.powf(1.0 / dim),
        // Llama 3 style: linear interpolation between original and scaled rope.
        // effective_base = base * factor for the full scaling factor.
        // The (8/head_dim)^2/2 term was a rough approximation that produced ~1.6%
        // shift for factor=8 instead of the intended ~8x. Corrected to match the
        // Llama 3 reference: scale the base by the factor directly.
        // [P2-22 fix: corrected Llama3 rope scaling formula.]
        RopeScalingMethod::Llama3 { factor } => base * factor,
        RopeScalingMethod::LongRoPE { factor } | RopeScalingMethod::YaRN { factor, .. } => {
            // NTK-aware effective base (interpolation for low freq, extrapolation for high).
            base * factor.powf(dim / (dim - 2.0).max(1.0))
        }
        RopeScalingMethod::Dynamic { .. } => base,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_returns_native_base() {
        assert_eq!(
            scaling_base(&RopeScalingMethod::None, 10000.0, 128),
            10000.0
        );
        assert_eq!(
            scaling_base(&RopeScalingMethod::None, 500000.0, 128),
            500000.0
        );
    }

    #[test]
    fn linear_interpolates_base() {
        // base * factor^(1/head_dim)
        let out = scaling_base(&RopeScalingMethod::Linear { factor: 4.0 }, 10000.0, 4);
        assert!((out - 10000.0 * 4.0f32.powf(0.25)).abs() < 1e-3);
        assert!(out > 10000.0);
    }

    #[test]
    fn llama3_scales_with_gamma() {
        let out = scaling_base(&RopeScalingMethod::Llama3 { factor: 8.0 }, 10000.0, 128);
        // Llama3 rope scaling: effective_base = base * factor (corrected from the
        // rough (8/head_dim)^2/2 approximation which gave ~1.6% instead of ~8x).
        let expected = 10000.0 * 8.0;
        assert!((out - expected).abs() < 1e-3);
    }

    #[test]
    fn longrope_ntk_aware_exceeds_linear() {
        let linear = scaling_base(&RopeScalingMethod::Linear { factor: 2.0 }, 10000.0, 128);
        let ntk = scaling_base(&RopeScalingMethod::LongRoPE { factor: 2.0 }, 10000.0, 128);
        // NTK-aware extrapolates the high frequencies, so base grows faster than pure linear.
        assert!(ntk > linear);
    }

    #[test]
    fn yarn_uses_ntk_base() {
        let yarn = scaling_base(
            &RopeScalingMethod::YaRN {
                factor: 2.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                mscale: 1.0,
                mscale_all: false,
            },
            10000.0,
            128,
        );
        let longrope = scaling_base(&RopeScalingMethod::LongRoPE { factor: 2.0 }, 10000.0, 128);
        assert_eq!(yarn, longrope);
    }

    #[test]
    fn dynamic_delegates_to_base() {
        assert_eq!(
            scaling_base(
                &RopeScalingMethod::Dynamic {
                    type_: "theta-shift".to_string(),
                },
                300000.0,
                128,
            ),
            300000.0
        );
    }
}
