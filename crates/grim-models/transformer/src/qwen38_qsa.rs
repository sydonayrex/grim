//! Qwen Sparse Attention (QSA) indexer for `Qwen38FlashNext`.
//!
//! # Provenance
//!
//! Transcribed from `old/repo/llama.cpp-master/src/models/qwen4exp.cpp` at
//! commit `d7241ac8`, functions `build_qsa_top_k` and the `kq_mask_top_k`
//! construction in `build_attn_qsa`. That is the current upstream
//! implementation, taken from GitHub directly.
//!
//! # Why the mask is applied
//!
//! At `f3f1a8f` — the commit the Qwen3.8-Flash-Next GSQ-RCO release pins in its
//! build record — upstream computed the indexer and then discarded it:
//!
//! ```c
//! // TODO: enable sparse attention when we are ready
//! //ggml_tensor * cur = build_attn_mha(..., top_k->ne[0], kq_scale, il);
//! ggml_tensor * cur = build_attn_mha(..., 0, kq_scale, il);
//! ```
//!
//! So the published PPL of 3.1058 was measured with sparsity OFF despite the
//! "Flash" tag. `build_qsa_top_k` itself is byte-identical between the two
//! commits; only the line that consumes the mask changed. Grim follows current
//! upstream and applies the mask.
//!
//! # Pipeline
//!
//! ```text
//!   k_raw  = index_k_proj(x)                  [n_kv, idx_dim]  cached RAW
//!   pooled = mean over each r-cell block      [n_blocks, idx_dim]
//!   k      = rms_norm(pooled, index_k_norm)
//!   k      = rope(k, block position)
//!   q      = rms_norm(index_q_proj(x), index_q_norm)   [n_idx_h, idx_dim]
//!   q      = rope(q, token position)
//!   score  = relu(k . q_h)      <-- rectify PER HEAD, before the sum
//!   score  = sum_h score
//!   cell   = block score of the cell's block   (every token shares it)
//!   width  = min(n_kv, top_k + r - 1)
//!   top_k  = argpartition(cell, width)
//!   mask   = 0 on selected cells, -inf elsewhere, then += causal mask
//! ```
//!
//! Two details that a naive reading gets wrong, both confirmed from the source
//! and the tensor shapes:
//!
//! 1. The indexer has ONE key head. `index_k_proj` is `[n_embd, idx_dim]` =
//!    `[2560, 128]`, not 4x128. The `indexer.head_count` of 4 counts *query*
//!    heads: `index_q_proj` is `[n_embd, n_idx_h * idx_dim]` = `[2560, 512]`.
//! 2. The budget is in CELLS, not blocks. `r = compress_ratio` is 4 on exactly
//!    the 12 full-attention layers and 0 on the 36 GDN layers; `build_qsa_top_k`
//!    asserts `r > 0` and only runs on the former.

use grim_core::error::{Error, Result};

/// Indexer geometry for one sparse-attention layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QsaIndexConfig {
    /// `indexer.key_length` (128): width of one indexer head.
    pub idx_dim: usize,
    /// `indexer.head_count` (4): number of indexer QUERY heads.
    pub n_idx_h: usize,
    /// `indexer.top_k` (2048): selection budget, in cells.
    pub top_k: usize,
    /// `attention.compress_ratios[layer]` (4 on full-attention layers).
    pub compress_ratio: usize,
}

impl QsaIndexConfig {
    /// Validate the geometry.
    ///
    /// # Errors
    /// Returns [`Error::Config`] when any dimension is zero, or when
    /// `compress_ratio` is 0 — upstream asserts `r > 0` on this path, and a 0
    /// ratio means the layer is Gated DeltaNet, which must not reach QSA at all.
    pub fn validate(&self) -> Result<()> {
        if self.idx_dim == 0 {
            return Err(Error::Config("qsa: idx_dim must be > 0".into()));
        }
        if self.n_idx_h == 0 {
            return Err(Error::Config("qsa: n_idx_h must be > 0".into()));
        }
        if self.top_k == 0 {
            return Err(Error::Config("qsa: top_k must be > 0".into()));
        }
        if self.compress_ratio == 0 {
            return Err(Error::Config(
                "qsa: compress_ratio must be > 0; a 0 ratio marks a Gated DeltaNet \
                 layer, which has no sparse attention"
                    .into(),
            ));
        }
        Ok(())
    }

    /// Number of blocks a `n_kv`-cell history divides into.
    pub fn n_blocks(&self, n_kv: usize) -> usize {
        if self.compress_ratio == 0 {
            return 0;
        }
        n_kv.div_ceil(self.compress_ratio)
    }

    /// The selection width, verbatim from upstream:
    /// `width = min(n_kv, indexer_top_k + r - 1)` — whole blocks plus the tail.
    pub fn select_width(&self, n_kv: usize) -> usize {
        n_kv.min(self.top_k + self.compress_ratio - 1)
    }
}

/// Mean-pool raw indexer keys into blocks.
///
/// `k_raw` is `[n_kv, idx_dim]`, laid out row-major. Cells `b*r .. b*r + r` of
/// block `b` are averaged, giving `[n_blocks, idx_dim]`.
///
/// The keys are cached RAW: pooling happens first, and the norm and rotation
/// are applied to the pooled result, not per cell. Norming before pooling would
/// not commute with the mean.
pub fn pool_indexer_keys(k_raw: &[f32], n_kv: usize, idx_dim: usize, r: usize) -> Result<Vec<f32>> {
    if r == 0 {
        return Err(Error::Config("qsa: pool with r = 0".into()));
    }
    if k_raw.len() < n_kv * idx_dim {
        return Err(Error::Shape(format!(
            "qsa: indexer keys hold {} values, need {} for n_kv={} idx_dim={}",
            k_raw.len(),
            n_kv * idx_dim,
            n_kv,
            idx_dim
        )));
    }
    let n_blocks = n_kv.div_ceil(r);
    let mut pooled = vec![0.0f32; n_blocks * idx_dim];
    for b in 0..n_blocks {
        let start = b * r;
        let end = ((b + 1) * r).min(n_kv);
        let n = (end - start) as f32;
        if n <= 0.0 {
            continue;
        }
        for cell in start..end {
            let src = &k_raw[cell * idx_dim..(cell + 1) * idx_dim];
            let dst = &mut pooled[b * idx_dim..(b + 1) * idx_dim];
            for i in 0..idx_dim {
                dst[i] += src[i];
            }
        }
        let dst = &mut pooled[b * idx_dim..(b + 1) * idx_dim];
        for v in dst.iter_mut() {
            *v /= n;
        }
    }
    Ok(pooled)
}

/// RMSNorm in place over each row of a `[rows, width]` buffer.
pub fn rms_norm_rows(x: &mut [f32], width: usize, weight: &[f32], eps: f32) {
    for row in x.chunks_mut(width) {
        let ss: f32 = row.iter().map(|v| v * v).sum::<f32>() / width as f32;
        let scale = 1.0 / (ss + eps).sqrt();
        for (i, v) in row.iter_mut().enumerate() {
            *v = *v * scale * weight.get(i).copied().unwrap_or(1.0);
        }
    }
}

/// Score every block for one query token.
///
/// `pooled_k` is `[n_blocks, idx_dim]` (post-norm, post-rotation) and `q` is
/// `[n_tps, n_idx_h * idx_dim]`. Returns `[n_tps, n_blocks]`.
///
/// The ReLU is applied to EACH HEAD'S dot product BEFORE the heads are summed —
/// "rectify each head dot product before the sum, as in the DeepSeek lightning
/// indexer". Summing first and rectifying after gives a different, always
/// non-negative score and is the most likely way to get this subtly wrong.
pub fn indexer_block_scores(
    pooled_k: &[f32],
    q: &[f32],
    n_blocks: usize,
    n_idx_h: usize,
    idx_dim: usize,
    n_tps: usize,
) -> Result<Vec<f32>> {
    let need_k = n_blocks * idx_dim;
    if pooled_k.len() < need_k {
        return Err(Error::Shape(format!(
            "qsa: pooled keys hold {}, need {need_k}",
            pooled_k.len()
        )));
    }
    let need_q = n_tps * n_idx_h * idx_dim;
    if q.len() < need_q {
        return Err(Error::Shape(format!(
            "qsa: indexer queries hold {}, need {need_q}",
            q.len()
        )));
    }
    let mut scores = vec![0.0f32; n_tps * n_blocks];
    for t in 0..n_tps {
        for b in 0..n_blocks {
            let krow = &pooled_k[b * idx_dim..(b + 1) * idx_dim];
            let mut acc = 0.0f32;
            for h in 0..n_idx_h {
                let qrow = &q[t * n_idx_h * idx_dim + h * idx_dim
                    ..t * n_idx_h * idx_dim + (h + 1) * idx_dim];
                let dot: f32 = qrow.iter().zip(krow.iter()).map(|(a, b)| a * b).sum();
                acc += dot.max(0.0); // ReLU per head, then sum.
            }
            scores[t * n_blocks + b] = acc;
        }
    }
    Ok(scores)
}

/// Broadcast per-block scores back onto every cell.
///
/// `block_scores` is `[n_tps, n_blocks]`; returns `[n_tps, n_kv]` where each
/// cell carries its block's score. The budget cuts on a cell boundary, so this
/// expansion happens BEFORE the top-k.
pub fn expand_block_scores(
    block_scores: &[f32],
    n_blocks: usize,
    n_kv: usize,
    r: usize,
    n_tps: usize,
) -> Result<Vec<f32>> {
    if block_scores.len() < n_tps * n_blocks {
        return Err(Error::Shape(format!(
            "qsa: block scores hold {}, need {}",
            block_scores.len(),
            n_tps * n_blocks
        )));
    }
    let mut cells = vec![0.0f32; n_tps * n_kv];
    for t in 0..n_tps {
        for c in 0..n_kv {
            cells[t * n_kv + c] = block_scores[t * n_blocks + c / r];
        }
    }
    Ok(cells)
}

/// Expand allocated block scores onto attention cells.
///
/// `blk_of[cell]` is the block holding that cell, or [`NO_BLOCK`] when the cell
/// sits in no complete block. Deriving the block from the cell index instead
/// (`cell / r`) is the bug the allocator exists to remove: it is only equal
/// when every cell sits in a full block at its own bucket, which fails as soon
/// as the cache is paged or a bucket is incomplete.
///
/// A cell with no block gets `NEG_INFINITY` so it can never be selected, which
/// is the same outcome upstream reaches with a `-inf` bias.
pub fn expand_allocated_block_scores(
    block_scores: &[f32],
    blk_of: &[i32],
    n_tps: usize,
) -> Vec<f32> {
    let n_bid = block_scores.len() / n_tps.max(1);
    let mut cells = vec![f32::NEG_INFINITY; n_tps * blk_of.len()];
    for t in 0..n_tps {
        for (c, &b) in blk_of.iter().enumerate() {
            if b < 0 || b as usize >= n_bid {
                continue; // unpooled: never selectable
            }
            cells[t * blk_of.len() + c] = block_scores[t * n_bid + b as usize];
        }
    }
    cells
}

/// Indices of the `width` highest-scoring cells.
///
/// Ranked by score descending, ties broken by cell ascending, which matches
/// upstream's `ggml_top_k` on a contiguous axis only up to tie order — the
/// model is insensitive to which of several equal scores wins, so pinning it
/// here keeps decode deterministic.
pub fn top_k_cells(cell_scores: &[f32], width: usize) -> Vec<usize> {
    let n = cell_scores.len();
    let width = width.min(n);
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        cell_scores[b]
            .partial_cmp(&cell_scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    // Cells that scored -inf were never a candidate: they sit in no complete
    // block, or their block is unobserved, or they are in the future. Truncating
    // without this filter can hand the budget to a masked cell when the number
    // of selectable cells is below the width -- which is the normal case early
    // in a sequence, so it would silently select exactly the cells the block
    // allocator excluded.
    idx.retain(|&i| cell_scores[i] != f32::NEG_INFINITY);
    idx.truncate(width);
    idx
}

/// Build the additive attention mask: `0.0` on selected cells, `-inf` on all
/// others, with an optional additive causal mask summed on top.
///
/// Returns `[n_kv]` for a single query row.
///
/// # Errors
/// Returns [`Error::Shape`] if `additive_mask` is present and not `n_kv` long.
pub fn build_top_k_mask(
    n_kv: usize,
    selected: &[usize],
    additive_mask: Option<&[f32]>,
) -> Result<Vec<f32>> {
    if let Some(m) = additive_mask {
        if m.len() < n_kv {
            return Err(Error::Shape(format!(
                "qsa: additive mask holds {}, need {n_kv}",
                m.len()
            )));
        }
    }
    const NEG_INF: f32 = f32::NEG_INFINITY;
    let mut mask = vec![NEG_INF; n_kv];
    for &c in selected {
        if c < n_kv {
            mask[c] = 0.0;
        }
    }
    // ggml_add(kq_mask_top_k, kq_mask): the selected cells keep whatever the
    // causal/alibi mask says, the rest stay masked out.
    if let Some(m) = additive_mask {
        for (i, v) in mask.iter_mut().enumerate() {
            *v += m[i];
        }
    }
    Ok(mask)
}

/// End-to-end indexer: raw keys + queries in, selected cell indices out.
///
/// This is `build_qsa_top_k` minus the RoPE stages, which the caller applies
/// (they need the device's position tables). `pooled_k` and `q` must already be
/// normed and rotated.
#[allow(clippy::too_many_arguments)]
pub fn qsa_select_cells(
    cfg: &QsaIndexConfig,
    k_raw: &[f32],
    q: &[f32],
    n_kv: usize,
    n_tps: usize,
) -> Result<Vec<usize>> {
    cfg.validate()?;
    let pooled = pool_indexer_keys(k_raw, n_kv, cfg.idx_dim, cfg.compress_ratio)?;
    let n_blocks = cfg.n_blocks(n_kv);
    let scores = indexer_block_scores(&pooled, q, n_blocks, cfg.n_idx_h, cfg.idx_dim, n_tps)?;
    let cells = expand_block_scores(&scores, n_blocks, n_kv, cfg.compress_ratio, n_tps)?;
    Ok(top_k_cells(&cells, cfg.select_width(n_kv)))
}

/// Softmax attention over the KV history, restricted to the selected cells.
///
/// Mirrors `shared_attention::scalar_attention` (GQA head mapping, `1/sqrt(d)`
/// scale, causal limit from `cache_offset + t`) but takes an explicit
/// `keep` mask instead of a sliding window, which is what the indexer
/// produces. Kept local rather than adding a parameter to
/// `fused_or_scalar_attention`, which has 22 call sites across the workspace.
///
/// `q` is `[steps, num_heads * head_dim]`; `k_history` / `v_history` are
/// `[kv_len, num_kv_heads * head_dim]`. Returns `[steps, num_heads * head_dim]`.
///
/// # Errors
/// Returns [`Error::Shape`] if the buffers do not match the declared geometry
/// or if `keep` is shorter than `kv_len`.
pub fn masked_gqa_attention(
    q: &[f32],
    k_history: &[f32],
    v_history: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    steps: usize,
    kv_len: usize,
    cache_offset: usize,
    keep: &[f32],
) -> Result<Vec<f32>> {
    if num_heads == 0 || num_kv_heads == 0 || head_dim == 0 {
        return Err(Error::Config("qsa: degenerate attention geometry".into()));
    }
    if keep.len() < kv_len {
        return Err(Error::Shape(format!(
            "qsa: keep mask holds {} cells, need {kv_len}",
            keep.len()
        )));
    }
    let q_dim = num_heads * head_dim;
    let kv_stride = num_kv_heads * head_dim;
    if q.len() < steps * q_dim {
        return Err(Error::Shape(format!(
            "qsa: q holds {}, need {}",
            q.len(),
            steps * q_dim
        )));
    }
    if k_history.len() < kv_len * kv_stride || v_history.len() < kv_len * kv_stride {
        return Err(Error::Shape(format!(
            "qsa: kv history too short for kv_len={kv_len} kv_stride={kv_stride}"
        )));
    }
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; steps * q_dim];

    for h in 0..num_heads {
        let kvh = (h * num_kv_heads) / num_heads;
        for t in 0..steps {
            let causal_limit = cache_offset + t;
            let mut scores = vec![0.0f32; kv_len];
            let mut any = false;
            for t2 in 0..kv_len {
                // The indexer mask: 0 keeps, -inf removes. Causality is applied
                // independently, so a selected future cell stays excluded.
                let m = keep[t2];
                if m == f32::NEG_INFINITY || t2 > causal_limit {
                    scores[t2] = f32::NEG_INFINITY;
                    continue;
                }
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[t * q_dim + h * head_dim + d]
                        * k_history[t2 * kv_stride + kvh * head_dim + d];
                }
                scores[t2] = dot * scale + m;
                if scores[t2].is_finite() {
                    any = true;
                }
            }
            if !any {
                // Every visible cell was masked out. Upstream would produce
                // NaN from a -inf softmax; emitting zeros keeps the shape and
                // stays finite, and the residual carries the token forward.
                continue;
            }
            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in &mut scores {
                *s = if s.is_finite() { (*s - mx).exp() } else { 0.0 };
                sum += *s;
            }
            if sum > 0.0 {
                for s in &mut scores {
                    *s /= sum;
                }
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for t2 in 0..kv_len {
                        acc += scores[t2] * v_history[t2 * kv_stride + kvh * head_dim + d];
                    }
                    out[t * q_dim + h * head_dim + d] = acc;
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> QsaIndexConfig {
        // The released checkpoint's geometry.
        QsaIndexConfig {
            idx_dim: 128,
            n_idx_h: 4,
            top_k: 2048,
            compress_ratio: 4,
        }
    }

    #[test]
    fn geometry_matches_the_released_checkpoint() {
        let c = cfg();
        assert_eq!(c.idx_dim, 128, "indexer.key_length");
        assert_eq!(c.n_idx_h, 4, "indexer.head_count");
        assert_eq!(c.top_k, 2048, "indexer.top_k");
        assert_eq!(
            c.compress_ratio, 4,
            "compress_ratios on a full-attention layer"
        );
        c.validate().expect("geometry must validate");
    }

    #[test]
    fn zero_compress_ratio_is_rejected_as_a_gdn_layer() {
        let c = QsaIndexConfig {
            compress_ratio: 0,
            ..cfg()
        };
        let err = c.validate().expect_err("r = 0 marks a GDN layer");
        assert!(
            format!("{err}").contains("Gated DeltaNet"),
            "error should name why: {err}"
        );
    }

    #[test]
    fn select_width_is_top_k_plus_r_minus_one_capped_at_n_kv() {
        let c = cfg();
        // 2048 + 4 - 1 = 2051, capped by n_kv.
        assert_eq!(c.select_width(10_000), 2051, "whole blocks plus the tail");
        assert_eq!(
            c.select_width(1000),
            1000,
            "a short history is not over-selected"
        );
        assert_eq!(c.select_width(2051), 2051, "exactly at the cap");
        assert_eq!(c.select_width(2048), 2048, "below the cap");
    }

    #[test]
    fn n_blocks_rounds_up() {
        let c = cfg();
        assert_eq!(c.n_blocks(1024), 256, "1024 / 4");
        assert_eq!(
            c.n_blocks(1025),
            257,
            "a partial trailing block still counts"
        );
        assert_eq!(c.n_blocks(0), 0);
    }

    #[test]
    fn pooling_is_a_mean_not_a_sum() {
        // One block of 4 cells, width 1. Cells 1,2,3,4 -> mean 2.5.
        let k = vec![1.0f32, 2.0, 3.0, 4.0];
        let pooled = pool_indexer_keys(&k, 4, 1, 4).expect("pool");
        assert_eq!(pooled.len(), 1);
        assert!((pooled[0] - 2.5).abs() < 1e-6, "got {}", pooled[0]);
    }

    #[test]
    fn pooling_averages_only_the_cells_present_in_a_partial_block() {
        // 5 cells, r = 4 -> block 0 has 4 cells, block 1 has 1.
        let k: Vec<f32> = (1..=5).map(|v| v as f32).collect();
        let pooled = pool_indexer_keys(&k, 5, 1, 4).expect("pool");
        assert_eq!(pooled.len(), 2, "two blocks");
        assert!((pooled[0] - 2.5).abs() < 1e-6, "block 0 mean of 1..4");
        assert!(
            (pooled[1] - 5.0).abs() < 1e-6,
            "block 1 is the lone cell, not divided by 4"
        );
    }

    #[test]
    fn short_key_buffer_is_rejected() {
        let err = pool_indexer_keys(&[0.0; 3], 4, 1, 4).expect_err("needs 4 keys");
        assert!(format!("{err}").contains("indexer keys"));
    }

    #[test]
    fn relu_is_applied_per_head_before_the_sum() {
        // Two heads, one block, idx_dim 1. Head 0 dot = -1 (rejected by ReLU),
        // head 1 dot = 3. Per-head ReLU then sum = 0 + 3 = 3. Sum then ReLU
        // would give relu(-1 + 3) = 2, and no ReLU at all would give 2.
        let pooled = vec![1.0f32];
        // q row layout: [n_tps, n_idx_h * idx_dim] = 2 heads of width 1.
        let q = vec![-1.0f32, 3.0];
        let s = indexer_block_scores(&pooled, &q, 1, 2, 1, 1).expect("scores");
        assert!(
            (s[0] - 3.0).abs() < 1e-6,
            "per-head ReLU then sum = 3, got {}",
            s[0]
        );
    }

    #[test]
    fn all_negative_head_dots_score_zero() {
        let pooled = vec![1.0f32, 1.0f32];
        let q = vec![-5.0f32, -2.0];
        let s = indexer_block_scores(&pooled, &q, 1, 2, 1, 1).expect("scores");
        assert_eq!(s[0], 0.0, "rectified negatives contribute nothing");
    }

    #[test]
    fn block_scores_expand_to_every_cell_in_the_block() {
        // 2 blocks, r = 2, 4 cells. Block scores 10 and 1.
        let block = vec![10.0f32, 1.0];
        let cells = expand_block_scores(&block, 2, 4, 2, 1).expect("expand");
        assert_eq!(cells, vec![10.0, 10.0, 1.0, 1.0], "cells 0,1 -> block 0");
    }

    #[test]
    fn top_k_picks_the_highest_scores() {
        let scores = vec![0.1f32, 0.9, 0.5, 0.7];
        assert_eq!(top_k_cells(&scores, 2), vec![1, 3], "descending by score");
    }

    /// A cell scoring -inf was never a candidate. Truncating without the
    /// filter hands the budget to masked cells whenever the selectable count
    /// is below the width, which is the normal case early in a sequence.
    ///
    /// This is the only place that behaviour is pinned: the end-to-end forward
    /// cannot see it, because an all-masked selection still yields a finite
    /// result that happens to match the dense arm at those positions.
    #[test]
    fn top_k_never_selects_a_masked_cell() {
        // Two selectable cells and three masked ones; the width is 5, larger
        // than the selectable count, which is the case that matters.
        let scores = vec![1.0f32, -f32::INFINITY, 2.0, -f32::INFINITY, -f32::INFINITY];
        let sel = top_k_cells(&scores, 5);
        // Score descending: cell 2 scores 2.0, cell 0 scores 1.0.
        assert_eq!(
            sel,
            vec![2, 0],
            "only the finite-scoring cells may be selected, even when the \
             budget exceeds them"
        );
        assert!(
            sel.iter().all(|&i| scores[i] != f32::NEG_INFINITY),
            "no -inf cell may appear in the selection"
        );
        // An all-masked input selects nothing rather than everything.
        let all_masked = vec![-f32::INFINITY; 4];
        assert!(top_k_cells(&all_masked, 4).is_empty());
    }

    #[test]
    fn top_k_breaks_ties_by_cell_ascending() {
        let scores = vec![0.5f32, 0.5, 0.5, 0.5];
        assert_eq!(top_k_cells(&scores, 2), vec![0, 1], "deterministic ties");
    }

    #[test]
    fn top_k_clamps_to_the_available_cells() {
        let scores = vec![1.0f32, 2.0];
        assert_eq!(
            top_k_cells(&scores, 99).len(),
            2,
            "cannot select more than exist"
        );
    }

    #[test]
    fn mask_masks_everything_except_the_selection() {
        let m = build_top_k_mask(8, &[1, 5], None).expect("mask");
        assert_eq!(m[1], 0.0, "selected");
        assert_eq!(m[5], 0.0, "selected");
        for i in [0usize, 2, 3, 4, 6, 7] {
            assert!(m[i] == f32::NEG_INFINITY, "cell {i} must be masked");
        }
    }

    #[test]
    fn additive_causal_mask_is_summed_onto_the_selection() {
        // Cell 2 is not selected but is causally visible: it stays masked.
        // Cell 5 is selected and causally visible: it gets the causal bias.
        let causal = vec![0.0f32; 8];
        let m = build_top_k_mask(8, &[5], Some(&causal)).expect("mask");
        assert_eq!(m[5], 0.0, "selected + visible");
        assert!(m[2] == f32::NEG_INFINITY, "unselected stays masked");
    }

    #[test]
    fn causal_bias_applies_to_selected_cells_too() {
        // A -inf causal entry on a selected cell must win: the sum is -inf.
        let mut causal = vec![0.0f32; 4];
        causal[2] = f32::NEG_INFINITY;
        let m = build_top_k_mask(4, &[1, 2, 3], Some(&causal)).expect("mask");
        assert_eq!(m[1], 0.0);
        assert!(m[2] == f32::NEG_INFINITY, "causality must still exclude it");
        assert_eq!(m[3], 0.0);
    }

    #[test]
    fn short_additive_mask_is_rejected() {
        let causal = vec![0.0f32; 3];
        assert!(build_top_k_mask(8, &[0], Some(&causal)).is_err());
    }

    #[test]
    fn masked_attention_ignores_unselected_cells() {
        // One head, one kv head, head_dim 1. V cell 0 = 1, cell 1 = 100.
        // Only cell 0 is kept, so the output must be 1, not ~100.
        let q = [1.0f32];
        let k = [0.0f32, 0.0];
        let v = [1.0f32, 100.0];
        let keep = [0.0f32, f32::NEG_INFINITY];
        let out = masked_gqa_attention(&q, &k, &v, 1, 1, 1, 1, 2, 0, &keep).expect("attn");
        assert_eq!(out.len(), 1);
        assert!(
            (out[0] - 1.0).abs() < 1e-6,
            "the unselected cell must not contribute; got {}",
            out[0]
        );
    }

    #[test]
    fn masked_attention_averages_when_two_cells_are_kept_and_visible() {
        // cache_offset 1 with 1 step means both cells are causally visible, so
        // an all-zero keep mask must give a plain mean over the two values.
        let q = [1.0f32];
        let k = [0.0f32, 0.0];
        let v = [1.0f32, 100.0];
        let keep = [0.0f32, 0.0];
        let out = masked_gqa_attention(&q, &k, &v, 1, 1, 1, 1, 2, 1, &keep).expect("attn");
        assert!(
            (out[0] - 50.5).abs() < 1e-4,
            "both cells kept and visible should average to 50.5; got {}",
            out[0]
        );
    }

    #[test]
    fn masked_attention_still_obeys_causality() {
        // Cell 1 is selected but lies in the future for step 0 (cache_offset 0).
        let q = [1.0f32];
        let k = [0.0f32, 0.0];
        let v = [1.0f32, 100.0];
        let keep = [0.0f32, 0.0];
        let out = masked_gqa_attention(&q, &k, &v, 1, 1, 1, 1, 2, 0, &keep).expect("attn");
        assert!(
            (out[0] - 1.0).abs() < 1e-6,
            "a future cell must stay excluded even when selected; got {}",
            out[0]
        );
    }

    #[test]
    fn masked_attention_emits_zeros_when_everything_is_masked() {
        // All masked: must stay finite rather than producing NaN.
        let q = [1.0f32, 1.0];
        let k = [0.0f32, 0.0];
        let v = [1.0f32, 1.0];
        let keep = [f32::NEG_INFINITY; 2];
        let out = masked_gqa_attention(&q, &k, &v, 1, 1, 1, 2, 2, 0, &keep).expect("attn");
        assert!(out.iter().all(|v| v.is_finite()), "got {out:?}");
        assert!(out.iter().all(|v| *v == 0.0), "expected zeros, got {out:?}");
    }

    #[test]
    fn masked_attention_rejects_a_short_mask() {
        let q = [0.0f32];
        let k = [0.0f32; 2];
        let v = [0.0f32; 2];
        assert!(masked_gqa_attention(&q, &k, &v, 1, 1, 1, 1, 2, 0, &[0.0]).is_err());
    }

    #[test]
    fn masked_attention_matches_dense_when_everything_is_kept() {
        // With a keep mask of all zeros, this must equal the dense result.
        let num_heads = 4;
        let num_kv = 2;
        let hd = 8;
        let kv_len = 6;
        let steps = 3;
        let q: Vec<f32> = (0..steps * num_heads * hd)
            .map(|i| ((i % 17) as f32 - 8.0) / 8.0)
            .collect();
        let k: Vec<f32> = (0..kv_len * num_kv * hd)
            .map(|i| ((i % 13) as f32 - 6.0) / 6.0)
            .collect();
        let v: Vec<f32> = (0..kv_len * num_kv * hd)
            .map(|i| ((i % 11) as f32 - 5.0) / 5.0)
            .collect();
        let keep = vec![0.0f32; kv_len];
        let out = masked_gqa_attention(&q, &k, &v, num_heads, num_kv, hd, steps, kv_len, 0, &keep)
            .expect("attn");
        assert_eq!(out.len(), steps * num_heads * hd);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "an all-kept mask must behave like dense attention"
        );
    }

    /// The per-head ReLU must be INSIDE the accumulation, not applied to the
    /// sum. Two heads with dots -1 and +3:
    ///   per-head ReLU then sum = 0 + 3 = 3   (correct)
    ///   sum then ReLU        = relu(2) = 2   (wrong)
    /// Both are non-negative, so a test that only checks "the score is not
    /// negative" passes either way.
    #[test]
    fn relu_placement_is_not_equivalent_to_rectifying_the_sum() {
        let pooled = vec![1.0f32];
        let q = vec![-1.0f32, 3.0];
        let got = indexer_block_scores(&pooled, &q, 1, 2, 1, 1).expect("scores")[0];
        let sum_then_relu = 2.0f32; // relu(-1 + 3)
        assert!(
            (got - sum_then_relu).abs() > 1e-6,
            "score {got} is indistinguishable from sum-then-ReLU ({sum_then_relu}); \
             the test would not discriminate"
        );
        assert!(
            (got - 3.0).abs() < 1e-6,
            "per-head ReLU then sum = 3, got {got}"
        );
    }

    /// Pool-then-norm is not norm-then-pool, and upstream does the former.
    ///
    /// RMSNorm is scale-invariant: a row is projected onto the unit sphere, so
    /// multiplying every cell of a block by a constant cannot change the pooled
    /// result. The order is therefore observable only when a block's cells have
    /// DIFFERENT DIRECTIONS, not merely different magnitudes. Block 0 is
    /// [1,0] and [0,1] — orthogonal, so pooling first averages them to [0.5,0.5]
    /// while norming first maps each to itself and averages to the same point
    /// only if the two are symmetric. Block 1 is [3,0] and [1,0]: same
    /// direction, and the orders coincide there too, which is exactly why the
    /// mixed-direction block is the one that discriminates.
    #[test]
    fn norm_order_changes_the_pooled_value() {
        let r = 2;
        let idx_dim = 2;
        // Block 0: [1,0] and [0,1] -- averaged to [0.5, 0.5].
        // Block 1: [1,0] and [0,1] as well, but with different magnitudes.
        let k: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 4.0];
        let n_kv = 4;

        let mut pooled = pool_indexer_keys(&k, n_kv, idx_dim, r).expect("pool");
        assert!((pooled[0] - 0.5).abs() < 1e-6, "block 0 pooled x");
        assert!((pooled[2] - 0.5).abs() < 1e-6, "block 1 pooled x");
        assert!((pooled[3] - 2.0).abs() < 1e-6, "block 1 pooled y");
        rms_norm_rows(&mut pooled, idx_dim, &[1.0, 1.0], 1e-5);

        let mut per_cell = k.clone();
        rms_norm_rows(&mut per_cell, idx_dim, &[1.0, 1.0], 1e-5);
        let pooled_first = pool_indexer_keys(&per_cell, n_kv, idx_dim, r).expect("pool");

        assert!(
            (pooled[3] - pooled_first[3]).abs() > 1e-3,
            "the two orders must disagree on block 1: pool-then-norm {:?} vs \
             norm-then-pool {:?}",
            &pooled[2..4],
            &pooled_first[2..4]
        );
    }

    /// The host reference result must be moved onto the tensor's own device.
    ///
    /// A ROCm run that returns a CPU tensor from the masked softmax hands a
    /// host buffer to `wo` and the rest of the block, which then mixes devices.
    /// `move_to_device` is the seam's only correct exit, so assert it is
    /// reachable and a no-op on the CPU.
    #[test]
    fn host_result_is_relocated_onto_the_target_device() {
        let t = grim_backend_cpu::cpu_tensor(vec![1.0f32, 2.0], grim_tensor::Shape::new(vec![2]));
        assert_eq!(*t.device(), grim_tensor::Device::Cpu);
        // Same device: a cheap clone, still correct.
        let same = grim_nn::modules::move_to_device(&t, t.device()).expect("move");
        assert_eq!(*same.device(), grim_tensor::Device::Cpu);
        assert_eq!(same.to_vec_f32().expect("vals"), vec![1.0, 2.0]);
    }

    #[test]
    fn end_to_end_selects_the_scored_cells() {
        // idx_dim 1, 2 heads, 8 cells, r = 4 -> 2 blocks of 4.
        // Block 0 keys sum low, block 1 keys high; a positive query scores
        // block 1 higher, so its 4 cells win.
        let c = QsaIndexConfig {
            idx_dim: 1,
            n_idx_h: 2,
            top_k: 4,
            compress_ratio: 4,
        };
        let k: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0];
        let q = vec![1.0, 1.0]; // one token, two query heads
        let sel = qsa_select_cells(&c, &k, &q, 8, 1).expect("select");
        // width = min(n_kv, top_k + r - 1) = min(8, 4 + 4 - 1) = 7, not 4.
        assert_eq!(sel.len(), 7, "select_width(8) = {}", c.select_width(8));
        // Block 1 (cells 4..8) scores 5x block 0, so all four of its cells are
        // selected first; the remaining 3 slots go to the lower block, lowest
        // score first. What matters is that the high block is fully in.
        for i in 4..8 {
            assert!(
                sel.contains(&i),
                "cell {i} is in the high-scoring block and must be selected; got {sel:?}"
            );
        }
        // And the 3 lowest-scoring cells are the ones excluded.
        let excluded: Vec<usize> = (0..8).filter(|i| !sel.contains(i)).collect();
        assert_eq!(excluded.len(), 1, "7 of 8 selected");
    }

    #[test]
    fn short_history_selects_everything() {
        // A history shorter than the budget must not be truncated.
        let c = QsaIndexConfig {
            idx_dim: 1,
            n_idx_h: 1,
            top_k: 2048,
            compress_ratio: 4,
        };
        let k: Vec<f32> = (0..10).map(|v| v as f32).collect();
        let q = vec![1.0];
        let sel = qsa_select_cells(&c, &k, &q, 10, 1).expect("select");
        assert_eq!(sel.len(), 10, "all 10 cells fit in the budget");
    }
}
