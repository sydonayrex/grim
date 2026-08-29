//! `grim-backend-tests` — Centralized Known-Answer Tests (KAT) and backend parity verification.
//!
//! §WI-E9: Multi-format numerical parity tests across CPU, ROCm, and CUDA backends.

use grim_quant::QuantFormat;

/// List of standard quantized formats tested in the parity matrix.
/// Variant names match `grim_tensor::dtype::QuantFormat` exactly.
pub const TEST_QUANT_FORMATS: &[QuantFormat] = &[
    QuantFormat::Q8_0,
    QuantFormat::Q4K,
    QuantFormat::Q5K,
    QuantFormat::Q6K,
    QuantFormat::Iq4Nl,
];

/// List of K dimensions tested (includes single-block, multi-block, and large LLM head/hidden sizes).
pub const TEST_K_DIMS: &[usize] = &[256, 512, 1536, 2048, 4096];
