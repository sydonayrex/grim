//! SpQR (Sparse Quantized Representation) salient-weight identification.

use grim_tensor::error::{Error, Result};

/// Container for salient weights extracted during SpQR quantization.
///
/// Salient weights are the ~1% of weights with the highest Hessian curvature sensitivity.
/// They are stored as sparse (index, value) pairs in FP16 to preserve precision while
/// allowing the remaining 99% of weights to be aggressively quantized (e.g. Crow Q4K / Jay MXFP4).
#[derive(Debug, Clone, PartialEq)]
pub struct SpqrSalientResidual {
    /// 0-indexed flat element indices of salient weights in the tensor.
    pub indices: Vec<u32>,
    /// Original unquantized values of salient weights (f32).
    pub values: Vec<f32>,
}

/// Identify salient weights whose Hessian curvature exceeds `threshold_multiplier * mean_curvature`.
///
/// Returns an `SpqrSalientResidual` containing the indices and values of the salient entries.
pub fn spqr_identify_salient(
    weights: &[f32],
    curvature: &[f32],
    threshold_multiplier: f32,
) -> Result<SpqrSalientResidual> {
    if weights.len() != curvature.len() {
        return Err(Error::Backend(format!(
            "spqr_identify_salient: weights len ({}) != curvature len ({})",
            weights.len(),
            curvature.len()
        )));
    }

    if weights.is_empty() {
        return Ok(SpqrSalientResidual {
            indices: Vec::new(),
            values: Vec::new(),
        });
    }

    let mean_curv = curvature.iter().map(|c| c.abs()).sum::<f32>() / curvature.len() as f32;
    let cutoff = mean_curv * threshold_multiplier;

    let mut indices = Vec::new();
    let mut values = Vec::new();

    for (i, (&w, &c)) in weights.iter().zip(curvature.iter()).enumerate() {
        if c.abs() > cutoff {
            indices.push(i as u32);
            values.push(w);
        }
    }

    Ok(SpqrSalientResidual { indices, values })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spqr_selects_high_curvature_weights_as_salient() {
        let weights = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
        let curvature = vec![0.01f32, 0.01, 0.01, 0.01, 100.0];
        let res = spqr_identify_salient(&weights, &curvature, 2.0).unwrap();
        assert_eq!(res.indices, vec![4]);
        assert_eq!(res.values, vec![0.5]);
    }

    #[test]
    fn spqr_handles_empty_inputs() {
        let res = spqr_identify_salient(&[], &[], 2.0).unwrap();
        assert!(res.indices.is_empty());
        assert!(res.values.is_empty());
    }

    #[test]
    fn spqr_errors_on_mismatched_lengths() {
        let res = spqr_identify_salient(&[1.0], &[1.0, 2.0], 1.0);
        assert!(res.is_err());
    }
}
