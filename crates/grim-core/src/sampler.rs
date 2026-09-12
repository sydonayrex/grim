//! `Sampler` trait - token selection from logits.
//! Concrete samplers (greedy, top-k, nucleus, mirostat, ...) implement this trait; plugins (§6) provide extensions via.

use grim_tensor::Tensor;
use grim_tensor::error::Result;

/// History-aware token sampler. The `history` argument carries the most recently emitted tokens (typically
/// the last 64 tokens) for samplers that need repetition context (DRY, mirostat variants, etc.).
pub trait Sampler: Send + Sync {
    /// Sample one token from the logits distribution.
    fn sample(&self, logits: &Tensor, history: &[u32]) -> Result<u32>;

    /// Human-readable name for logs / sampler registry.
    fn name(&self) -> &str;
}

/// Model thinking / reasoning effort level control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThinkingLevel {
    /// Explicitly disable reasoning/thinking tokens (e.g. `reasoning_effort: "none"` or `0`).
    Off,
    /// Default model thinking mode (enabled for models supporting on/off toggle).
    #[default]
    Default,
    /// Low reasoning effort (e.g. `reasoning_effort: "low"` / ~1024 token thinking budget).
    Low,
    /// Medium reasoning effort (e.g. `reasoning_effort: "medium"` / ~4096 token thinking budget).
    Medium,
    /// High reasoning effort (e.g. `reasoning_effort: "high"` / ~16384 token thinking budget).
    High,
    /// Custom extended effort level above high (e.g. `reasoning_effort: "max"`, `"ultra"`, or explicit max token budget).
    Custom(u32),
}

impl ThinkingLevel {
    /// Parse from string (e.g. "off", "none", "0", "default", "on", "low", "medium", "high", "max", "ultra").
    pub fn parse(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "off" | "none" | "0" | "false" | "disabled" => ThinkingLevel::Off,
            "default" | "on" | "true" | "enabled" | "auto" => ThinkingLevel::Default,
            "low" | "minimal" => ThinkingLevel::Low,
            "medium" | "med" | "moderate" => ThinkingLevel::Medium,
            "high" | "max" => ThinkingLevel::High,
            other => {
                if let Ok(budget) = other.parse::<u32>() {
                    match budget {
                        0 => ThinkingLevel::Off,
                        1..=2048 => ThinkingLevel::Low,
                        2049..=8192 => ThinkingLevel::Medium,
                        8193..=32768 => ThinkingLevel::High,
                        custom => ThinkingLevel::Custom(custom),
                    }
                } else {
                    // For custom named levels above high (e.g. "ultra", "extreme")
                    ThinkingLevel::Custom(65536)
                }
            }
        }
    }

    /// Returns the recommended thinking token budget (maximum reasoning tokens).
    pub fn max_thinking_tokens(&self) -> Option<u32> {
        match self {
            ThinkingLevel::Off => Some(0),
            ThinkingLevel::Default => None,
            ThinkingLevel::Low => Some(1024),
            ThinkingLevel::Medium => Some(4096),
            ThinkingLevel::High => Some(16384),
            ThinkingLevel::Custom(budget) => Some(*budget),
        }
    }
}

/// Sampling parameters parsed from an OpenAI/Ollama request.
/// `temperature == 0.0` is the canonical "greedy / deterministic" signal and must produce argmax output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    /// Repetition penalty (CTRL paper, Keskar 2019): logits for tokens already present in `history` are divided by this value before temperature scaling.
    /// 1.0 = disabled.
    pub repeat_penalty: f32,
    /// Controls model reasoning / thinking effort (Off, Default, Low, Medium, High, Custom).
    pub thinking_level: ThinkingLevel,
    /// Minimum tokens to generate before EOS is allowed. Prevents premature stopping
    /// on models that may emit EOS-biased logits early in the sequence. 0 = no minimum.
    pub min_tokens: u32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        // OpenAI-compatible defaults: greedy off, mild nucleus, no hard top-k.
        // repeat_penalty defaults to a mild 1.10 — disables cleanly at 1.0.
        SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repeat_penalty: 1.0,
            thinking_level: ThinkingLevel::Default,
            min_tokens: 0,
        }
    }
}

impl SamplingParams {
    /// Resolve an explicit greedy sampler when temperature is zero, otherwise a stochastic top-p sampler seeded from `seed`.
    /// Both implement `Sampler` so callers hold a single trait object regardless of mode.
    pub fn into_sampler(self, seed: u64) -> Box<dyn Sampler> {
        if self.temperature <= 0.0 {
            Box::new(GreedySampler::new(self.repeat_penalty))
        } else {
            Box::new(TopPSampler::new(self, seed))
        }
    }
}

/// Pre-allocated sampling buffers to avoid per-token heap allocations.
///
/// Owns reusable working memory for the sampling pipeline: temperature-scaled
/// logits, softmax exponentials, sort indices, and the top-p CDF. Reuse one
/// `SamplerState` across all decode steps in a session.
///
/// Typical per-token allocation without `SamplerState`: ~5-7 heap allocations
/// of vocab-size arrays (vocab = 128K → ~3.8 MB allocated per token). With
/// `SamplerState`, zero heap allocations per token after construction.
pub struct SamplerState {
    /// Reused for temperature-scaled logits (and repeat-penalty output).
    pub logits_buf: Vec<f32>,
    /// Reused for softmax exponentials.
    pub exp_buf: Vec<f32>,
    /// Reused for sort indices (top-k and top-p).
    pub idx_buf: Vec<usize>,
    /// Reused for top-p CDF: (token_index, cumulative_probability).
    pub cdf_buf: Vec<(usize, f32)>,
}

impl SamplerState {
    /// Allocate a fresh `SamplerState` with capacity for `vocab_size` tokens.
    pub fn with_capacity(vocab_size: usize) -> Self {
        Self {
            logits_buf: Vec::with_capacity(vocab_size),
            exp_buf: Vec::with_capacity(vocab_size),
            idx_buf: (0..vocab_size).collect(),
            cdf_buf: Vec::with_capacity(vocab_size.min(4096)),
        }
    }

    /// Reset internal buffers to length 0 (capacity preserved).
    pub fn clear(&mut self) {
        self.logits_buf.clear();
        self.exp_buf.clear();
        self.cdf_buf.clear();
        // idx_buf is always 0..n; only the length prefix is valid per call.
    }
}

/// Histogram-based partial top-k selection — O(n) instead of O(n log n).
///
/// Finds the `k` highest-logit tokens using a 128-bucket histogram over the
/// logit range [-10, 10]. Tokens outside this range are clamped to the
/// nearest bucket. Returns the index into `idx_buf` where the top-k tokens
/// begin (tokens are sorted descending by logit within the result).
///
/// This matches the approach in llama.cpp's `llama_token_data_array_partial_sort`.
fn histogram_top_k(
    logits: &[f32],
    idx_buf: &mut [usize],
    k: usize,
) -> usize {
    const NBUCKETS: usize = 128;
    const BUCKET_LOW: f32 = -10.0;
    const BUCKET_HIGH: f32 = 10.0;
    const BUCKET_SCALE: f32 = NBUCKETS as f32 / (BUCKET_HIGH - BUCKET_LOW);
    const BUCKET_INTER: f32 = -BUCKET_LOW * BUCKET_SCALE;

    // Build histogram: count tokens per bucket.
    let mut histo = [0u32; NBUCKETS];
    for &logit in logits.iter() {
        let mut ib = (BUCKET_SCALE * logit + BUCKET_INTER) as i32;
        ib = ib.clamp(0, NBUCKETS as i32 - 1);
        histo[ib as usize] += 1;
    }

    // Find the highest bucket that contains the k-th token.
    let mut nhave = 0usize;
    let mut ib = NBUCKETS;
    for b in (0..NBUCKETS).rev() {
        nhave += histo[b] as usize;
        if nhave >= k {
            ib = b;
            break;
        }
    }
    if ib == NBUCKETS {
        // All tokens fit in fewer than k buckets — return everything.
        return 0;
    }

    // Collect tokens from buckets >= ib into idx_buf, sorted descending.
    // First pass: copy indices from buckets above ib (already sorted by bucket).
    let mut pos = 0usize;
    for b in ((ib + 1)..NBUCKETS).rev() {
        for (i, &logit) in logits.iter().enumerate() {
            let mut bucket = (BUCKET_SCALE * logit + BUCKET_INTER) as i32;
            bucket = bucket.clamp(0, NBUCKETS as i32 - 1);
            if bucket as usize == b {
                idx_buf[pos] = i;
                pos += 1;
            }
        }
    }

    // Second pass: tokens in bucket ib — sort by descending logit.
    let bucket_start = pos;
    for (i, &logit) in logits.iter().enumerate() {
        let mut bucket = (BUCKET_SCALE * logit + BUCKET_INTER) as i32;
        bucket = bucket.clamp(0, NBUCKETS as i32 - 1);
        if bucket as usize == ib {
            idx_buf[pos] = i;
            pos += 1;
        }
    }
    // Sort the tokens within bucket ib by descending logit.
    idx_buf[bucket_start..pos].sort_by(|&a, &b| {
        logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Return the start index of the top-k region.
    // Tokens before `bucket_start` are from higher buckets (already sorted).
    // Tokens from `bucket_start..pos` are from bucket ib (just sorted).
    // The top-k is the first k tokens in this combined list.
    if pos <= k {
        0
    } else {
        // We may have collected more than k tokens from bucket ib.
        // Return the start such that idx_buf[start..start+k] is the top-k.
        pos.saturating_sub(k)
    }
}

/// Optimized sampling using pre-allocated `SamplerState` buffers.
///
/// Pipeline: temperature-scale → histogram top-k → fused softmax + top-p CDF → draw.
/// Zero heap allocations per token (all buffers reused from `state`).
///
/// Falls back to the scalar `sample_logits` for edge cases (greedy, non-finite).
pub fn sample_logits_with_state<F>(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    top_k: u32,
    repeat_penalty: f32,
    history: &[u32],
    rng: &mut F,
    state: &mut SamplerState,
) -> u32
where
    F: FnMut() -> u32,
{
    if temperature <= 0.0 {
        let scaled = apply_repeat_penalty(logits, repeat_penalty, history);
        return argmax_first(&scaled);
    }

    let n = logits.len();
    state.clear();

    // Apply repetition penalty + temperature scaling into logits_buf.
    state.logits_buf.extend(logits.iter().map(|&x| {
        if x.is_nan() {
            f32::NEG_INFINITY
        } else {
            x / temperature
        }
    }));
    // Apply repeat penalty in-place.
    if repeat_penalty > 1.0 && !history.is_empty() {
        let mut seen = std::collections::HashSet::with_capacity(history.len().min(1024));
        for &tok in history {
            if !seen.insert(tok) {
                continue;
            }
            let i = tok as usize;
            if i < state.logits_buf.len() {
                if state.logits_buf[i] < 0.0 {
                    state.logits_buf[i] *= repeat_penalty;
                } else {
                    state.logits_buf[i] /= repeat_penalty;
                }
            }
        }
    }

    // Histogram top-k: find the top-k tokens into idx_buf.
    let top_k = if top_k > 0 && (top_k as usize) < n {
        top_k as usize
    } else {
        n
    };
    let topk_start = if top_k < n {
        // Ensure idx_buf has the full index range.
        if state.idx_buf.len() < n {
            state.idx_buf = (0..n).collect();
        }
        histogram_top_k(&state.logits_buf, &mut state.idx_buf, top_k)
    } else {
        // No top-k truncation: use all tokens.
        if state.idx_buf.len() != n {
            state.idx_buf = (0..n).collect();
        }
        0
    };
    let topk_end = (topk_start + top_k).min(n);

    // Fused softmax + top-p CDF build over the top-k tokens.
    // First: find max logit among top-k for numerical stability.
    let mut max_logit = f32::NEG_INFINITY;
    for &idx in &state.idx_buf[topk_start..topk_end] {
        if state.logits_buf[idx] > max_logit {
            max_logit = state.logits_buf[idx];
        }
    }

    // Compute exps and sum in one pass.
    state.exp_buf.clear();
    let mut sum_exps = 0.0f32;
    for &idx in &state.idx_buf[topk_start..topk_end] {
        let x = state.logits_buf[idx];
        let exp_val = if x == f32::NEG_INFINITY || !x.is_finite() {
            0.0
        } else {
            (x - max_logit).exp()
        };
        state.exp_buf.push(exp_val);
        sum_exps += exp_val;
    }

    if sum_exps <= 0.0 || !max_logit.is_finite() {
        return argmax_first(&state.logits_buf);
    }

    // Build CDF over top-k tokens, applying top-p cutoff.
    state.cdf_buf.clear();
    let mut cumulative = 0.0f32;
    let mut cutoff = topk_end - topk_start;
    if top_p < 1.0 {
        for (j, &exp_val) in state.exp_buf.iter().enumerate() {
            cumulative += exp_val / sum_exps;
            let idx = state.idx_buf[topk_start + j];
            state.cdf_buf.push((idx, cumulative));
            if cumulative >= top_p {
                cutoff = j + 1;
                break;
            }
        }
        if cutoff == 0 {
            cutoff = 1;
        }
    } else {
        for (j, &exp_val) in state.exp_buf.iter().enumerate() {
            cumulative += exp_val / sum_exps;
            let idx = state.idx_buf[topk_start + j];
            state.cdf_buf.push((idx, cumulative));
        }
    }

    if cumulative <= 0.0 {
        return argmax_first(logits);
    }

    // Draw from CDF.
    let draw = (rng() as f64 / (u32::MAX as f64)) * cumulative as f64;
    for &(idx, c) in state.cdf_buf.iter().take(cutoff) {
        if draw <= c as f64 {
            return idx as u32;
        }
    }
    state.cdf_buf[cutoff - 1].0 as u32
}

/// Greedy (argmax) sampler - used when `temperature == 0`.
/// Deterministic: always returns the highest-logit token.
pub struct GreedySampler {
    /// Repetition penalty to apply before argmax.
    /// Forwarded via [`SamplingParams::repeat_penalty`] - without it, greedy decoding gets stuck emitting the same token forever.
    pub repeat_penalty: Option<f32>,
}

impl GreedySampler {
    pub fn new(repeat_penalty: f32) -> Self {
        Self {
            repeat_penalty: if repeat_penalty <= 1.0 {
                None
            } else {
                Some(repeat_penalty)
            },
        }
    }
}

impl Sampler for GreedySampler {
    fn sample(&self, logits: &Tensor, history: &[u32]) -> Result<u32> {
        let v = logits.to_vec_f32()?;
        let adjusted = match self.repeat_penalty {
            Some(rp) => apply_repeat_penalty(&v, rp, history),
            None => v,
        };
        Ok(argmax_first(&adjusted))
    }

    fn name(&self) -> &str {
        "greedy"
    }
}

/// Stochastic top-p (nucleus) sampler with temperature scaling.
/// Owns its RNG state so sampling is reproducible for a given seed + call order.
pub struct TopPSampler {
    params: SamplingParams,
    rng_state: std::sync::Mutex<u64>,
}

impl TopPSampler {
    pub fn new(params: SamplingParams, seed: u64) -> Self {
        // Avoid a zero state, which would stick xorshift at 0.
        TopPSampler {
            params,
            rng_state: std::sync::Mutex::new(if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            }),
        }
    }

    fn next_u32(&self) -> u32 {
        let mut state = self.rng_state.lock().unwrap_or_else(|e| e.into_inner());
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state >> 32) as u32
    }
}

impl Sampler for TopPSampler {
    fn sample(&self, logits: &Tensor, history: &[u32]) -> Result<u32> {
        let v = logits.to_vec_f32()?;
        let token = sample_logits(
            &v,
            self.params.temperature,
            self.params.top_p,
            self.params.top_k,
            self.params.repeat_penalty,
            history,
            &mut || self.next_u32(),
        );
        Ok(token)
    }

    fn name(&self) -> &str {
        "top-p"
    }
}

/// Index of the maximum element with first-occurrence tie-breaking.
/// Uses `max_by` + reverse enumeration so the first max in forward order wins.
fn argmax_first(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .rev()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Apply repetition penalty to logits. For every token
/// id present in `history`, divide its logit by `repeat_penalty`.
fn apply_repeat_penalty(logits: &[f32], repeat_penalty: f32, history: &[u32]) -> Vec<f32> {
    if repeat_penalty <= 1.0 || history.is_empty() {
        return logits.to_vec();
    }
    let mut out = logits.to_vec();
    let mut seen = std::collections::HashSet::with_capacity(history.len().min(1024));
    for &tok in history {
        if !seen.insert(tok) {
            continue;
        }
        let i = tok as usize;
        if i < out.len() {
            if out[i] < 0.0 {
                out[i] *= repeat_penalty;
            } else {
                out[i] /= repeat_penalty;
            }
        }
    }
    out
}

/// Pure, dependency-free token sampler. Pipeline: temperature-scale the logits → optional top-k truncation
/// → softmax → top-p (nucleus) cumulative-mass cutoff → weighted choice driven by `rng`.
pub fn sample_logits<F>(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    top_k: u32,
    repeat_penalty: f32,
    history: &[u32],
    rng: &mut F,
) -> u32
where
    F: FnMut() -> u32,
{
    if temperature <= 0.0 {
        // Greedy mode still benefits from repetition penalty — apply it before
        // argmax so a token emitted once isn't immediately re-emitted forever.
        let scaled = apply_repeat_penalty(logits, repeat_penalty, history);
        return argmax_first(&scaled);
    }

    // Apply repetition penalty first, before temperature scaling — CTRL paper.
    let pen_log: Vec<f32> = apply_repeat_penalty(logits, repeat_penalty, history);

    // Temperature scaling. Guard against non-finite input so a single NaN logit
    // cannot poison the whole distribution (treat as -inf → never selected).
    let scaled: Vec<f32> = pen_log
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
        order.sort_by(|&a, &b| {
            scaled[b]
                .partial_cmp(&scaled[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
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
        .map(|&x| {
            if x == f32::NEG_INFINITY {
                0.0
            } else {
                (x - max).exp()
            }
        })
        .collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 || !max.is_finite() {
        // `INF - INF` is NaN, so a +INF-dominant logit makes `sum` NaN; in that case the softmax is a one-hot on the max-logit token(s).
        // Delegate to argmax over the (already NaN-masked) scaled logits.
        return argmax_first(&masked);
    }
    let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

    // Top-p (nucleus): sort indices by descending probability, accumulate
    // mass, and cut once we cross `top_p`. A `top_p >= 1.0` keeps everything.
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

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
        return argmax_first(logits);
    }

    let draw = (rng() as f64 / (u32::MAX as f64)) * cumulative as f64;
    for (idx, c) in &cdf {
        if draw <= *c as f64 {
            return *idx as u32;
        }
    }
    cdf.last().map(|(idx, _)| *idx as u32).unwrap_or(0)
}

/// Returns true if the given token is an end-of-sequence token for the provided tokenizer.
/// Checks the model's native `eos_token_id` plus common chat-format stop tokens.
pub fn is_eos_token(token: u32, eos_token_id: Option<u32>, stop_token_ids: &[u32]) -> bool {
    eos_token_id.map_or(false, |id| token == id) || stop_token_ids.contains(&token)
}

/// Determines whether generation should stop given the current state.
///
/// Returns `true` if `next_token` is an EOS token AND either:
/// - `min_tokens` is 0 (no minimum), OR
/// - `generated` >= `min_tokens` (minimum met)
///
/// This prevents premature EOS on models that emit EOS-biased logits early
/// in the sequence (e.g. LFM2 with certain prompt patterns).
pub fn should_stop(
    next_token: u32,
    generated: u32,
    min_tokens: u32,
    eos_token_id: Option<u32>,
    stop_token_ids: &[u32],
) -> bool {
    if !is_eos_token(next_token, eos_token_id, stop_token_ids) {
        return false;
    }
    // EOS generated — only stop if we've met the minimum token budget.
    min_tokens == 0 || generated >= min_tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_max_logit() {
        let logits = vec![0.1, 2.0, 0.5, -1.0];
        assert_eq!(sample_logits(&logits, 0.0, 1.0, 0, 1.0, &[], &mut || 0), 1);
    }

    #[test]
    fn temperature_zero_is_argmax_regardless_of_rng() {
        // A non-zero rng must not perturb a greedy draw.
        let logits = vec![0.1, 2.0, 0.5, -1.0];
        let chosen = sample_logits(&logits, 0.0, 1.0, 0, 1.0, &[], &mut || 0xFFFF_FFFF);
        assert_eq!(chosen, 1);
    }

    #[test]
    fn stochastic_draw_respects_distribution() {
        // With a sharply peaked distribution the sampled token is the max almost always; over many draws it
        // must never leave the support and must hit the dominant token the vast majority of the time.
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
            if sample_logits(&logits, 1.0, 0.95, 0, 1.0, &[], &mut rng) == 1 {
                dominant += 1;
            }
        }
        assert!(
            dominant > 990,
            "dominant token should win ~always, got {dominant}"
        );
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
            assert_eq!(sample_logits(&logits, 1.0, 0.5, 0, 1.0, &[], &mut rng), 0);
        }
    }

    #[test]
    fn non_finite_logits_do_not_poison_sample() {
        let logits = vec![f32::NAN, 2.0, f32::INFINITY, -1.0];
        // INFINITY dominates softmax → token 2; NaN is masked to -inf.
        let chosen = sample_logits(&logits, 1.0, 1.0, 0, 1.0, &[], &mut || 0);
        assert_eq!(chosen, 2);
    }

    #[test]
    fn params_resolve_to_greedy_when_temperature_zero() {
        let sampler = SamplingParams {
            temperature: 0.0,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.0,
            thinking_level: ThinkingLevel::Default,
            min_tokens: 0,
        }
        .into_sampler(42);
        assert_eq!(sampler.name(), "greedy");
    }

    #[test]
    fn should_stop_respects_min_tokens() {
        let eos = Some(7u32);
        let stops: &[u32] = &[];

        // EOS token 7, min_tokens=0 → always stop
        assert!(should_stop(7, 0, 0, eos, stops));
        assert!(should_stop(7, 5, 0, eos, stops));

        // EOS token 7, min_tokens=100, generated=5 → don't stop yet
        assert!(!should_stop(7, 5, 100, eos, stops));

        // EOS token 7, min_tokens=100, generated=100 → stop
        assert!(should_stop(7, 100, 100, eos, stops));

        // EOS token 7, min_tokens=100, generated=150 → stop
        assert!(should_stop(7, 150, 100, eos, stops));

        // Non-EOS token → never stop regardless of min_tokens
        assert!(!should_stop(42, 5, 100, eos, stops));
        assert!(!should_stop(42, 0, 0, eos, stops));
    }

    #[test]
    fn should_stop_with_custom_stop_tokens() {
        let eos = None;
        let stops = &[7u32, 100, 200];

        assert!(should_stop(7, 0, 0, eos, stops));
        assert!(should_stop(100, 0, 0, eos, stops));
        assert!(should_stop(200, 0, 0, eos, stops));
        assert!(!should_stop(42, 0, 0, eos, stops));
    }

    #[test]
    fn is_eos_token_checks_all_sources() {
        assert!(is_eos_token(7, Some(7), &[]));
        assert!(is_eos_token(7, None, &[7, 100]));
        assert!(is_eos_token(100, Some(7), &[100]));
        assert!(!is_eos_token(42, Some(7), &[100]));
        assert!(!is_eos_token(0, None, &[]));
    }

    #[test]
    fn test_thinking_level_parsing_and_token_budget() {
        assert_eq!(ThinkingLevel::parse("off"), ThinkingLevel::Off);
        assert_eq!(ThinkingLevel::parse("0"), ThinkingLevel::Off);
        assert_eq!(ThinkingLevel::parse("none"), ThinkingLevel::Off);

        assert_eq!(ThinkingLevel::parse("default"), ThinkingLevel::Default);
        assert_eq!(ThinkingLevel::parse("on"), ThinkingLevel::Default);

        assert_eq!(ThinkingLevel::parse("low"), ThinkingLevel::Low);
        assert_eq!(ThinkingLevel::parse("medium"), ThinkingLevel::Medium);
        assert_eq!(ThinkingLevel::parse("high"), ThinkingLevel::High);

        // Custom levels above high or numeric budgets
        assert_eq!(ThinkingLevel::parse("50000"), ThinkingLevel::Custom(50000));
        assert_eq!(ThinkingLevel::parse("ultra"), ThinkingLevel::Custom(65536));

        assert_eq!(ThinkingLevel::Off.max_thinking_tokens(), Some(0));
        assert_eq!(ThinkingLevel::Default.max_thinking_tokens(), None);
        assert_eq!(ThinkingLevel::Low.max_thinking_tokens(), Some(1024));
        assert_eq!(ThinkingLevel::Medium.max_thinking_tokens(), Some(4096));
        assert_eq!(ThinkingLevel::High.max_thinking_tokens(), Some(16384));
        assert_eq!(
            ThinkingLevel::Custom(50000).max_thinking_tokens(),
            Some(50000)
        );
    }

    // --- P1-F: SamplerState tests ---

    #[test]
    fn sampler_state_allocates_with_capacity() {
        let state = SamplerState::with_capacity(128);
        assert_eq!(state.logits_buf.capacity(), 128);
        assert_eq!(state.exp_buf.capacity(), 128);
        assert_eq!(state.idx_buf.len(), 128);
        assert_eq!(state.idx_buf[0], 0);
        assert_eq!(state.idx_buf[127], 127);
    }

    #[test]
    fn sampler_state_clear_preserves_capacity() {
        let mut state = SamplerState::with_capacity(64);
        state.logits_buf.extend_from_slice(&[1.0, 2.0, 3.0]);
        state.exp_buf.extend_from_slice(&[0.5, 0.5]);
        state.cdf_buf.push((0, 1.0));
        state.clear();
        assert!(state.logits_buf.is_empty());
        assert!(state.exp_buf.is_empty());
        assert!(state.cdf_buf.is_empty());
        assert_eq!(state.logits_buf.capacity(), 64);
        assert_eq!(state.exp_buf.capacity(), 64);
    }

    // --- P1-F: histogram_top_k tests ---

    #[test]
    fn histogram_top_k_selects_highest_logits() {
        let logits = vec![0.1, 5.0, 3.0, 1.0, 4.0, 2.0];
        let mut idx_buf = (0..logits.len()).collect::<Vec<_>>();
        let start = histogram_top_k(&logits, &mut idx_buf, 3);
        // The top-3 indices (by logit) should be 1 (5.0), 4 (4.0), 2 (3.0).
        let top3: Vec<usize> = idx_buf[start..start + 3].to_vec();
        assert!(top3.contains(&1));
        assert!(top3.contains(&4));
        assert!(top3.contains(&2));
        // And they should be sorted descending.
        assert_eq!(top3, vec![1, 4, 2]);
    }

    #[test]
    fn histogram_top_k_handles_k_equal_n() {
        let logits = vec![1.0, 2.0, 3.0];
        let mut idx_buf = (0..logits.len()).collect::<Vec<_>>();
        let start = histogram_top_k(&logits, &mut idx_buf, 3);
        assert_eq!(start, 0);
        assert_eq!(idx_buf[..3], vec![2, 1, 0]);
    }

    // --- P1-F: sample_logits_with_state tests ---

    #[test]
    fn state_based_sample_matches_scalar_greedy() {
        // With temperature 0, both paths should return argmax.
        let logits = vec![0.1, 2.0, 0.5, -1.0];
        let mut seed: u64 = 0x1234;
        let mut rng = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 32) as u32
        };
        let mut state = SamplerState::with_capacity(logits.len());
        let result = sample_logits_with_state(&logits, 0.0, 1.0, 0, 1.0, &[], &mut rng, &mut state);
        assert_eq!(result, 1);
    }

    #[test]
    fn state_based_sample_respects_distribution() {
        // With a sharply peaked distribution, the sampled token is the max almost always.
        let logits = vec![0.0, 10.0, 0.0, 0.0];
        let mut seed: u64 = 0x1234;
        let mut rng = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 32) as u32
        };
        let mut state = SamplerState::with_capacity(logits.len());
        let mut dominant = 0usize;
        for _ in 0..1000 {
            if sample_logits_with_state(&logits, 1.0, 0.95, 0, 1.0, &[], &mut rng, &mut state) == 1 {
                dominant += 1;
            }
        }
        assert!(dominant > 990, "dominant token should win ~always, got {dominant}");
    }

    #[test]
    fn state_based_sample_excludes_low_prob_tokens() {
        let logits = vec![4.0, 0.0, 0.0, 0.0];
        let mut seed: u64 = 7;
        let mut rng = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 32) as u32
        };
        let mut state = SamplerState::with_capacity(logits.len());
        for _ in 0..200 {
            assert_eq!(
                sample_logits_with_state(&logits, 1.0, 0.5, 0, 1.0, &[], &mut rng, &mut state),
                0
            );
        }
    }

    #[test]
    fn state_based_sample_with_top_k() {
        let logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        let mut seed: u64 = 42;
        let mut rng = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 32) as u32
        };
        let mut state = SamplerState::with_capacity(logits.len());
        // With top_k=2, only tokens 1 (5.0) and 4 (4.0) should be selected.
        for _ in 0..500 {
            let tok =
                sample_logits_with_state(&logits, 1.0, 1.0, 2, 1.0, &[], &mut rng, &mut state);
            assert!(tok == 1 || tok == 4, "top_k=2 should only select tokens 1 or 4, got {tok}");
        }
    }
}
