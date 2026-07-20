//! `Sampler` trait — token selection from logits.
//!
//! Concrete samplers (greedy, top-k, nucleus, mirostat, ...) implement this
//! trait; plugins (§6) provide extensions via either the dylib or WASM path.

use grim_tensor::error::Result;
use grim_tensor::Tensor;

/// History-aware token sampler. The `history` argument carries the most
/// recently emitted tokens (typically the last 64 tokens) for samplers
/// that need repetition context (DRY, mirostat variants, etc.).
pub trait Sampler: Send + Sync {
    /// Sample one token from the logits distribution.
    fn sample(&self, logits: &Tensor, history: &[u32]) -> Result<u32>;

    /// Human-readable name for logs / sampler registry.
    fn name(&self) -> &str;
}

/// Sampling parameters parsed from an OpenAI/Ollama request.
///
/// `temperature == 0.0` is the canonical "greedy / deterministic" signal and
/// must produce argmax output (see `GreedySampler`); any positive temperature
/// enables stochastic sampling. `top_p` (nucleus) clips the cumulative
/// probability mass; `top_k` bounds the candidate set before the top-p pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        // OpenAI-compatible defaults: greedy off, mild nucleus, no hard top-k.
        SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
        }
    }
}

impl SamplingParams {
    /// Resolve an explicit greedy sampler when temperature is zero, otherwise
    /// a stochastic top-p sampler seeded from `seed`. Both implement `Sampler`
    /// so callers hold a single trait object regardless of mode.
    pub fn into_sampler(self, seed: u64) -> Box<dyn Sampler> {
        if self.temperature <= 0.0 {
            Box::new(GreedySampler)
        } else {
            Box::new(TopPSampler::new(self, seed))
        }
    }
}

/// Greedy (argmax) sampler — used when `temperature == 0`.
///
/// Deterministic: always returns the highest-logit token. This preserves the
/// historical `chat_completions` behavior for callers that request
/// deterministic output, and is the test oracle for the stochastic path.
pub struct GreedySampler;

impl Sampler for GreedySampler {
    fn sample(&self, logits: &Tensor, _history: &[u32]) -> Result<u32> {
        let v = logits.to_vec_f32()?;
        Ok(argmax(&v))
    }

    fn name(&self) -> &str {
        "greedy"
    }
}

/// Stochastic top-p (nucleus) sampler with temperature scaling.
///
/// Owns its RNG state so sampling is reproducible for a given seed + call
/// order. No external RNG dependency is pulled in for this — a xorshift64
/// state is sufficient for token selection and keeps the crate portable.
pub struct TopPSampler {
    params: SamplingParams,
    rng_state: std::sync::Mutex<u64>,
}

impl TopPSampler {
    pub fn new(params: SamplingParams, seed: u64) -> Self {
        // Avoid a zero state, which would stick xorshift at 0.
        TopPSampler {
            params,
            rng_state: std::sync::Mutex::new(if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed }),
        }
    }

    fn next_u32(&self) -> u32 {
        let mut state = self.rng_state.lock().unwrap();
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state >> 32) as u32
    }
}

impl Sampler for TopPSampler {
    fn sample(&self, logits: &Tensor, _history: &[u32]) -> Result<u32> {
        let v = logits.to_vec_f32()?;
        let token = sample_logits(&v, self.params.temperature, self.params.top_p, self.params.top_k, &mut || self.next_u32());
        Ok(token)
    }

    fn name(&self) -> &str {
        "top-p"
    }
}

/// Index of the maximum element. Ties resolve to the first occurrence, which
/// matches the prior inline `max_by` behavior and keeps greedy output stable.
fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Pure, dependency-free token sampler.
///
/// Pipeline: temperature-scale the logits → optional top-k truncation →
/// softmax → top-p (nucleus) cumulative-mass cutoff → weighted choice driven
/// by `rng`. `temperature <= 0` short-circuits to argmax so this single
/// function covers both the greedy and stochastic paths (the caller chooses
/// which via `SamplingParams::into_sampler`, but the function itself stays
/// branch-free on the greedy case for testability).
///
/// `rng` is only invoked when sampling is stochastic; callers that want
/// argmax behavior should pass `temperature <= 0` and may supply a no-op rng.
pub fn sample_logits<F>(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    top_k: u32,
    rng: &mut F,
) -> u32
where
    F: FnMut() -> u32,
{
    if temperature <= 0.0 {
        return argmax(logits);
    }

    // Temperature scaling. Guard against non-finite input so a single NaN logit
    // cannot poison the whole distribution (treat as -inf → never selected).
    let scaled: Vec<f32> = logits
        .iter()
        .map(|&x| {
            // NaN poisons the distribution; +INF is a valid (dominant) logit
            // and must survive temperature scaling.
            if x.is_nan() {
                f32::NEG_INFINITY
            } else {
                x / temperature
            }
        })
        .collect();

    // Optional top-k pre-truncation: keep only the k highest logits, mask the
    // rest to -inf. top_k == 0 means "no truncation".
    let masked = if top_k > 0 && (top_k as usize) < scaled.len() {
        let mut order: Vec<usize> = (0..scaled.len()).collect();
        order.sort_by(|&a, &b| scaled[b].partial_cmp(&scaled[a]).unwrap_or(std::cmp::Ordering::Equal));
        let mut m = scaled.clone();
        for &idx in order.iter().skip(top_k as usize) {
            m[idx] = f32::NEG_INFINITY;
        }
        m
    } else {
        scaled
    };

    // Numerically stable softmax over the (possibly masked) logits.
    let max = masked.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = masked
        .iter()
        .map(|&x| if x == f32::NEG_INFINITY { 0.0 } else { (x - max).exp() })
        .collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 || !max.is_finite() {
        // `INF - INF` is NaN, so a +INF-dominant logit makes `sum` NaN; in that
        // case the softmax is a one-hot on the max-logit token(s). Delegate to
        // argmax over the (already NaN-masked) scaled logits.
        return argmax(&masked);
    }
    let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

    // Top-p (nucleus): sort indices by descending probability, accumulate
    // mass, and cut once we cross `top_p`. A `top_p >= 1.0` keeps everything.
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));

    let cutoff = if top_p >= 1.0 {
        probs.len()
    } else {
        let mut mass = 0.0f32;
        let mut n = 0usize;
        for &idx in &order {
            mass += probs[idx];
            n += 1;
            if mass >= top_p {
                break;
            }
        }
        n.max(1)
    };

    // Weighted choice over the nucleus set using the supplied rng.
    let mut cumulative = 0.0f32;
    let mut cdf: Vec<(usize, f32)> = Vec::with_capacity(cutoff);
    for &idx in order.iter().take(cutoff) {
        cumulative += probs[idx];
        cdf.push((idx, cumulative));
    }
    if cumulative <= 0.0 {
        return argmax(logits);
    }

    let draw = (rng() as f64 / (u32::MAX as f64)) * cumulative as f64;
    for (idx, c) in &cdf {
        if draw <= *c as f64 {
            return *idx as u32;
        }
    }
    cdf.last().map(|(idx, _)| *idx as u32).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_max_logit() {
        let logits = vec![0.1, 2.0, 0.5, -1.0];
        assert_eq!(sample_logits(&logits, 0.0, 1.0, 0, &mut || 0), 1);
    }

    #[test]
    fn temperature_zero_is_argmax_regardless_of_rng() {
        // A non-zero rng must not perturb a greedy draw.
        let logits = vec![0.1, 2.0, 0.5, -1.0];
        let chosen = sample_logits(&logits, 0.0, 1.0, 0, &mut || 0xFFFF_FFFF);
        assert_eq!(chosen, 1);
    }

    #[test]
    fn stochastic_draw_respects_distribution() {
        // With a sharply peaked distribution the sampled token is the max almost
        // always; over many draws it must never leave the support and must hit
        // the dominant token the vast majority of the time.
        let logits = vec![0.0, 10.0, 0.0, 0.0];
        let mut seed: u64 = 0x1234_5678;
        let mut rng = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 32) as u32
        };
        let mut dominant = 0usize;
        for _ in 0..1000 {
            if sample_logits(&logits, 1.0, 0.95, 0, &mut rng) == 1 {
                dominant += 1;
            }
        }
        assert!(dominant > 990, "dominant token should win ~always, got {dominant}");
    }

    #[test]
    fn top_p_excludes_low_probability_tokens() {
        // A clearly dominant logit: softmax of [4,0,0,0] ≈ [0.95, 0.017, ...],
        // so with top_p=0.5 the nucleus contains only token 0.
        let logits = vec![4.0, 0.0, 0.0, 0.0];
        let mut seed: u64 = 7;
        let mut rng = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 32) as u32
        };
        for _ in 0..200 {
            assert_eq!(sample_logits(&logits, 1.0, 0.5, 0, &mut rng), 0);
        }
    }

    #[test]
    fn non_finite_logits_do_not_poison_sample() {
        let logits = vec![f32::NAN, 2.0, f32::INFINITY, -1.0];
        // INFINITY dominates softmax → token 2; NaN is masked to -inf.
        let chosen = sample_logits(&logits, 1.0, 1.0, 0, &mut || 0);
        assert_eq!(chosen, 2);
    }

    #[test]
    fn params_resolve_to_greedy_when_temperature_zero() {
        let sampler = SamplingParams { temperature: 0.0, top_p: 0.9, top_k: 40 }.into_sampler(42);
        assert_eq!(sampler.name(), "greedy");
    }
}
