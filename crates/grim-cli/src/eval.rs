//! Held-out evaluation and token-weighted perplexity loop.

pub struct EvalReport {
    pub step: usize,
    pub loss: f64,
    pub ppl: f64,
    pub tokens: usize,
}

/// Token-weighted perplexity across the held-out set.
/// loss = sum(token_loss) / sum(tokens); ppl = exp(loss).
pub fn perplexity<F, E>(dataset: &[Vec<u32>], mut forward_loss: F) -> Result<EvalReport, E>
where
    F: FnMut(&[u32]) -> Result<f64, E>,
    E: From<String>,
{
    let mut total_loss = 0.0f64;
    let mut total_tokens = 0usize;
    for seq in dataset {
        let n = seq.len().max(1);
        let l = forward_loss(seq)?;
        total_loss += l * n as f64;
        total_tokens += n;
    }
    if total_tokens == 0 {
        return Err(E::from("eval: empty dataset".into()));
    }
    let avg = total_loss / total_tokens as f64;
    Ok(EvalReport {
        step: 0,
        loss: avg,
        ppl: avg.exp(),
        tokens: total_tokens,
    })
}

/// Helper to load an evaluation dataset from path and return raw token vectors.
pub fn load_eval_dataset(
    path: &str,
    tokenizer: &grim_format::GgufTokenizer,
    max_seq_len: usize,
) -> grim_core::error::Result<Vec<Vec<u32>>> {
    let examples = crate::train::load_dataset(path, tokenizer, max_seq_len)?;
    Ok(examples.into_iter().map(|(toks, _)| toks).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perplexity_finite_on_synthetic_corpus() {
        let dataset = vec![vec![1, 2, 3, 4], vec![5, 6]];
        // Synthetic forward returning constant cross-entropy loss = ln(2) ~= 0.693147
        let report = perplexity::<_, String>(&dataset, |_seq| Ok(2.0f64.ln())).unwrap();
        assert_eq!(report.tokens, 6);
        assert!((report.ppl - 2.0).abs() < 1e-5, "expected ppl = 2.0, got {}", report.ppl);
    }
}
