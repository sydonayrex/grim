//! Disaggregated attention/FFN placement advisor (DisagMoE / R5 idea, analytical).
//!
//! DisagMoE (Zeng et al. arXiv:2605.11005) splits attention and FFN onto disjoint
//! GPU groups and balances their bandwidth with a compute-communication roofline
//! model. The paper targets 16-node datacenter clusters — mismatched to grim's
//! single-node consumer target — so we do NOT implement cross-node execution here.
//!
//! What IS portable and useful: the roofline analysis itself. Given a model and a
//! set of GPU capabilities, `advise_placement` reports whether attention and FFN
//! arithmetic intensity favor co-location (one fast GPU) or separation (different
//! GPUs), and the per-group bandwidth allocation that balances them. grim's
//! C2plrController can consume these hints when placing MoE layers across the
//! two GPUs common in consumer multi-GPU boxes.

use crate::error::{Error, Result};
use crate::hyperparams::ArchHyperparameters;

/// Simplified GPU capability descriptor for placement advice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuCapability {
    /// Peak FP16/BF16 compute in TFLOPS.
    pub tflops: f64,
    /// Inter-GPU / inter-device bandwidth in GiB/s.
    pub bandwidth_gib_s: f64,
}

/// Roofline-based placement recommendation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlacementAdvice {
    /// Attention and FFN should run on the same device (co-located).
    /// Typical when both are compute-bound at the target batch/seq.
    CoLocated,
    /// Attention and FFN favor different devices (disaggregated).
    /// Typical when FFN is communication-bound and attention compute-bound.
    Disaggregated {
        /// Fraction of inter-device bandwidth to allocate to the attention
        /// group (0..=1); the FFN group gets the remainder.
        attn_bandwidth_fraction: u8,
    },
}

/// Advise whether attention and FFN should be co-located or split across GPUs
/// for `hparams` at the given sequence length and batch size, given the
/// available `gpus`.
///
/// The model (DisagMoE §4.3): per-GPU network bandwidth differs between groups
/// (NIC split); both groups collectively occupy half the system's total NICs.
/// Effective turning points are ÎA = (2m/(m+n))·Î and ÎF = (2n/(m+n))·Î where
/// Î = PeakBandwidth/PeakFLOPs. Attention (quadratic in seq) reaches
/// compute-bound sooner than FFN (linear), which disaggregation exploits.
///
/// Here we collapse to the single-node question: is the FFN's arithmetic
/// intensity low enough (communication-bound) that it should move to a second
/// GPU while attention stays compute-bound on the first?
pub fn advise_placement(
    hparams: &ArchHyperparameters,
    seq_len: usize,
    batch_size: usize,
    gpus: &[GpuCapability],
) -> Result<PlacementAdvice> {
    if gpus.len() < 2 {
        return Ok(PlacementAdvice::CoLocated);
    }
    if seq_len == 0 || batch_size == 0 {
        return Ok(PlacementAdvice::CoLocated);
    }

    let ffn_flops = ffn_flops(hparams, seq_len, batch_size);
    let attn_flops = attn_flops(hparams, seq_len, batch_size);

    // Use the two most-capable GPUs (sorted descending by TFLOPS).
    let mut gpus_sorted = gpus.to_vec();
    gpus_sorted.sort_by(|a, b| b.tflops.partial_cmp(&a.tflops).unwrap_or(std::cmp::Ordering::Equal));
    let g_slow = &gpus_sorted[1];

    // Turning point (FLOPs/Byte) of the slower GPU — above this, a kernel is
    // compute-bound on that device; below, it is communication-bound.
    let gib = 1024.0 * 1024.0 * 1024.0;
    let turning_point =
        if g_slow.bandwidth_gib_s > 0.0 && g_slow.tflops > 0.0 {
            g_slow.tflops * 1e12 / (g_slow.bandwidth_gib_s * gib)
        } else {
            f64::INFINITY
        };

    // FFN arithmetic intensity ≈ FLOPs / bytes moved. FFN moves its activation
    // once per layer (2 * batch * hidden bytes for the residual exchange).
    let hidden = hparams.hidden_size as f64;
    // Full activation tensor exchanged between the attention group and
    // the FFN group, per microbatch, across all layers (FP32).
    let ffn_bytes = batch_size as f64
        * seq_len as f64
        * hidden
        * hparams.num_layers as f64
        * 4.0;
    let ffn_intensity = if ffn_bytes > 0.0 { ffn_flops / ffn_bytes } else { f64::INFINITY };

    if ffn_intensity >= turning_point {
        // FFN is compute-bound even on the slower GPU → co-locate, no benefit.
        return Ok(PlacementAdvice::CoLocated);
    }

    // FFN is communication-bound: it benefits from being on a separate device.
    // Attention should keep most bandwidth (compute-bound side). Allocate ~60%
    // to attention, the rest to FFN (DisagMoE's MILP solves this; we approximate).
    let attn_intensity = {
        let attn_bytes = batch_size as f64 * seq_len as f64 * hidden * hparams.num_layers as f64 * 4.0;
        if attn_bytes > 0.0 { attn_flops / attn_bytes } else { f64::INFINITY }
    };
    let total_intensity = attn_intensity + ffn_intensity;
    let frac = if total_intensity > 0.0 {
        (attn_intensity / total_intensity).clamp(0.5, 0.8)
    } else {
        0.6
    };
    let attn_bandwidth_fraction = (frac * 100.0).round() as u8;

    Ok(PlacementAdvice::Disaggregated {
        attn_bandwidth_fraction,
    })
}

/// Estimated FFN FLOPs for the full model at (seq, batch): each expert FFN is
/// a SwiGLU (gate+up+down, each 2·hidden·inter per token) over active experts.
fn ffn_flops(hparams: &ArchHyperparameters, seq_len: usize, batch_size: usize) -> f64 {
    let tokens = (seq_len * batch_size) as f64;
    let per_tok_per_expert = 2.0 * 3.0 * hparams.hidden_size as f64 * hparams.intermediate_size as f64;
    if let Some(n_experts) = hparams.expert_count {
        let active = hparams.expert_used_count.unwrap_or(2).min(n_experts);
        tokens * active as f64 * per_tok_per_expert * hparams.num_layers as f64
    } else {
        tokens * per_tok_per_expert * hparams.num_layers as f64
    }
}

/// Estimated attention FLOPs: 2 * num_heads * seq^2 * head_dim per token batch,
/// across layers (the quadratic term that makes attention hit compute-bound first).
fn attn_flops(hparams: &ArchHyperparameters, seq_len: usize, batch_size: usize) -> f64 {
    let s = seq_len as f64;
    let q_per_tok = 2.0 * hparams.num_heads as f64 * s * hparams.head_dim as f64;
    let tokens = (seq_len * batch_size) as f64;
    tokens * q_per_tok * hparams.num_layers as f64
}

/// Bandwidth allocation split between attention and FFN groups, derived from the
/// placement advice. Returns `(attn_fraction, ffn_fraction)` each in [0,1].
pub fn bandwidth_split(advice: &PlacementAdvice) -> (f64, f64) {
    match advice {
        PlacementAdvice::CoLocated => (1.0, 0.0),
        PlacementAdvice::Disaggregated {
            attn_bandwidth_fraction,
        } => {
            let a = (*attn_bandwidth_fraction as f64) / 100.0;
            (a.clamp(0.0, 1.0), 1.0 - a.clamp(0.0, 1.0))
        }
    }
}

/// Validate that a placement advice is internally consistent.
pub fn validate(advice: &PlacementAdvice) -> Result<()> {
    match advice {
        PlacementAdvice::CoLocated => Ok(()),
        PlacementAdvice::Disaggregated {
            attn_bandwidth_fraction,
        } => {
            if *attn_bandwidth_fraction > 100 {
                Err(Error::Config(format!(
                    "disagg_placement: attn_bandwidth_fraction {attn_bandwidth_fraction} exceeds 100"
                )))
            } else {
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hparams(experts: Option<usize>) -> ArchHyperparameters {
        let mut hp = ArchHyperparameters::default();
        hp.hidden_size = 2048;
        hp.intermediate_size = 5632;
        hp.num_layers = 48;
        hp.num_heads = 16;
        hp.num_kv_heads = 16;
        hp.head_dim = 128;
        hp.expert_count = experts;
        hp.expert_used_count = experts.map(|_| 8);
        hp
    }

    #[test]
    fn single_gpu_is_colocated() {
        let advice = advise_placement(&hparams(None), 4096, 1, &[GpuCapability {
            tflops: 30.0,
            bandwidth_gib_s: 50.0,
        }])
        .unwrap();
        assert_eq!(advice, PlacementAdvice::CoLocated);
    }

    #[test]
    fn zero_seq_is_colocated() {
        let advice = advise_placement(
            &hparams(None),
            0,
            1,
            &[
                GpuCapability { tflops: 30.0, bandwidth_gib_s: 50.0 },
                GpuCapability { tflops: 20.0, bandwidth_gib_s: 50.0 },
            ],
        )
        .unwrap();
        assert_eq!(advice, PlacementAdvice::CoLocated);
    }

    #[test]
    fn heavy_single_request_moe_is_colocated() {
        // Realistic consumer MoE (wide FFN, 8 active experts) at batch=1: FFN is
        // strongly compute-bound even on a slow second GPU, so disaggregation
        // does not help -> co-locate. This is the correct single-node outcome and
        // matches the synthesis note that R5 targets multi-node datacenter scale.
        let advice = advise_placement(
            &hparams(Some(128)),
            2048,
            1,
            &[
                GpuCapability { tflops: 60.0, bandwidth_gib_s: 800.0 },
                GpuCapability { tflops: 30.0, bandwidth_gib_s: 400.0 },
            ],
        )
        .unwrap();
        assert_eq!(advice, PlacementAdvice::CoLocated);
    }

    #[test]
    fn consumer_configs_are_colocated() {
        // DisagMoE's disaggregation benefit requires many pipelined microbatches
        // at datacenter scale. For every realistic consumer config (single-node,
        // any batch/seq that fits), the FFN matmul is strongly compute-bound and
        // the advisor correctly returns CoLocated. This is the honest outcome and
        // matches the synthesis note that R5 is datacenter-scoped, not consumer.
        for &(seq, batch) in &[(128usize, 1usize), (512, 1), (2048, 1), (512, 32), (2048, 4)] {
            let advice = advise_placement(
                &hparams(Some(128)),
                seq,
                batch,
                &[
                    GpuCapability { tflops: 60.0, bandwidth_gib_s: 800.0 },
                    GpuCapability { tflops: 30.0, bandwidth_gib_s: 400.0 },
                ],
            )
            .unwrap();
            assert_eq!(
                advice,
                PlacementAdvice::CoLocated,
                "consumer config seq={seq} batch={batch} should be co-located"
            );
        }
    }

    #[test]
    fn disaggregation_reachable_only_at_extreme_bandwidth_deficit() {
        // Extreme synthetic case: near-zero inter-device bandwidth makes even a
        // modest FFN communication-bound. Documents the branch exists; NOT a
        // realistic consumer scenario (real links are 100+ GiB/s).
        let advice = advise_placement(
            &hparams(None), // dense, single "expert"
            4096,
            64,
            &[
                GpuCapability { tflops: 0.5, bandwidth_gib_s: 0.01 },
                GpuCapability { tflops: 0.5, bandwidth_gib_s: 0.01 },
            ],
        )
        .unwrap();
        assert!(
            matches!(advice, PlacementAdvice::Disaggregated { .. }),
            "extreme bandwidth deficit should disaggregate, got {advice:?}"
        );
    }
    #[test]
    fn bandwidth_split_sums_to_one() {
        let (a, f) = bandwidth_split(&PlacementAdvice::Disaggregated {
            attn_bandwidth_fraction: 60,
        });
        assert!((a + f - 1.0).abs() < 1e-9);
        assert!((a - 0.6).abs() < 1e-9);
    }

    #[test]
    fn validate_rejects_over_100() {
        let bad = PlacementAdvice::Disaggregated {
            attn_bandwidth_fraction: 150,
        };
        assert!(validate(&bad).is_err());
        let good = PlacementAdvice::Disaggregated {
            attn_bandwidth_fraction: 60,
        };
        assert!(validate(&good).is_ok());
    }
}
