//! KVA Late-Layer Prefill Skip for Hybrid Models.
//!
//! Splits a prompt of length `prompt_len` across `n_layers` into:
//! 1. **Prefix region** `0..(prompt_len - tail_len)`:
//!    - Early layers `0..early_layers` execute full forward (`LayerPrefillMode::Full`).
//!    - Late layers `early_layers..n_layers` execute state-only (`LayerPrefillMode::StateOnly`:
//!      updating KV cache / GDN / Mamba recurrent state while skipping attention output & FFN/MLP).
//! 2. **Exact tail region** `(prompt_len - tail_len)..prompt_len` (`tail_len >= 1` enforced):
//!    - All layers `0..n_layers` execute full forward (`LayerPrefillMode::Full`) so the final-token
//!      logits used for generation are computed through all layers over complete KV/recurrent state.
//!
//! Gated behind `GRIM_ASTRUM_KVA_SPLIT=<early_layers>/<tail_len>` (default OFF).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerPrefillMode {
    /// Run full attention/recurrent mixer + output projection + MLP/MoE.
    Full,
    /// Update KV page / recurrent GDN/conv state only; skip attention output + MLP/MoE.
    StateOnly,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvaSplitPlan {
    pub n_layers: usize,
    pub early_layers: usize,
    pub tail_len: usize,
    pub prefix_modes: Vec<LayerPrefillMode>,
    pub tail_modes: Vec<LayerPrefillMode>,
}

impl KvaSplitPlan {
    /// Build a KVA prefill split plan covering all `n_layers` layers and guaranteeing `tail_len >= 1`.
    pub fn plan(n_layers: usize, early_layers: usize, tail_len: usize) -> Self {
        let clamped_early = early_layers.min(n_layers);
        let safe_tail = tail_len.max(1);
        let mut prefix_modes = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            if l < clamped_early {
                prefix_modes.push(LayerPrefillMode::Full);
            } else {
                prefix_modes.push(LayerPrefillMode::StateOnly);
            }
        }
        let tail_modes = vec![LayerPrefillMode::Full; n_layers];
        Self {
            n_layers,
            early_layers: clamped_early,
            tail_len: safe_tail,
            prefix_modes,
            tail_modes,
        }
    }

    /// Parse `GRIM_ASTRUM_KVA_SPLIT="<early>/<tail>"` (e.g. `"24/64"`). Returns `None` when unset or invalid.
    pub fn from_env_str(val: Option<&str>, n_layers: usize) -> Option<Self> {
        let raw = val?.trim();
        let (early_s, tail_s) = raw.split_once('/')?;
        let early = early_s.trim().parse::<usize>().ok()?;
        let tail = tail_s.trim().parse::<usize>().ok()?;
        if n_layers == 0 || early >= n_layers {
            return None;
        }
        Some(Self::plan(n_layers, early, tail.max(1)))
    }

    pub fn from_env(n_layers: usize) -> Option<Self> {
        let v = std::env::var("GRIM_ASTRUM_KVA_SPLIT").ok();
        Self::from_env_str(v.as_deref(), n_layers)
    }

    /// Split `tokens` into `(prefix_slice, exact_tail_slice)` for KVA two-phase prefill.
    /// If `tokens.len() <= self.tail_len`, `prefix_slice` is empty and the entire prompt runs as exact tail.
    pub fn split_prompt<'a, T>(&self, tokens: &'a [T]) -> (&'a [T], &'a [T]) {
        if tokens.len() <= self.tail_len {
            (&tokens[..0], tokens)
        } else {
            let split_idx = tokens.len() - self.tail_len;
            (&tokens[..split_idx], &tokens[split_idx..])
        }
    }
}
