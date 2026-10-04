//! Vision M-RoPE for the Qwen3-VL clip tower (plan Task 2.2, WI-3).
//!
//! Implements `ggml_rope_multi` in `GGML_ROPE_TYPE_VISION` as the reference
//! defines it (`old/repo/llama.cpp-master/ggml/src/ggml-cpu/ops.cpp`). Three
//! properties distinguish it from an ordinary RoPE, and each is a silent-wrong
//! trap because a RoPE is the identity at position 0:
//!
//! 1. **Half-split pairing.** The VISION dispatch is
//!    `rotate_pairs<T>(ne0, n_dims, cache, src, dst)` (`ops.cpp:6214-6216`) with
//!    `n_dims = d_head / 2` (`qwen3vl.cpp:106`). So the offset is `ne0/2` and dim
//!    `i` pairs with dim `i + head_dim/2` - the NeoX half-split, not the
//!    interleaved GPT-J pairing. `is_vision` also skips the pass-through tail
//!    (`ops.cpp:6221`), so the entire head is rotated.
//!
//! 2. **Theta resets per section.** `ggml_mrope_cache_init` is called with
//!    `indep_sects = is_vision = true` (`ops.cpp:6150`, `:6194-6196`), so
//!    `theta_t/h/w/e` are reset to their bases at each section boundary
//!    (`ops.cpp:6009-6024`). Without this, each section would inherit the
//!    previous section's decayed theta and drift to the wrong base frequency.
//!
//! 3. **Four position ids per token**, read as t, h, w, e
//!    (`ops.cpp:6190-6193`), with the sections ordered t, h, w, e over
//!    `sections[]` (`ops.cpp:6038-6046`).
//!
//! Frequency: `theta *= freq_base^(-2/n_dims)` per pair (`ops.cpp:5988`), with
//! `n_dims` the number of PAIRS. No YaRN scaling is applied here: the reference
//! passes `ext_factor = 0` and `mscale = 1`, so `rope_yarn` reduces to a plain
//! `cosf(theta) / sinf(theta)` (`ops.cpp:5962-5972`).

/// Vision rotary parameters, derived from the checkpoint's `head_dim`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MropeParams {
    /// Width of one attention head. The rotation offset is half of this
    /// (half-split), and the whole head is rotated.
    pub head_dim: usize,
    /// How many equal-width sections the head is divided into. Qwen3-VL uses 4,
    /// one each for t, h, w and e (`qwen3vl.cpp:14`).
    pub num_sections: usize,
    /// `freq_base`; the reference passes 10000 for the vision tower
    /// (`qwen3vl.cpp:106-107`).
    pub freq_base: f32,
}

impl MropeParams {
    /// Pairs per head: the cache holds one cos/sin pair per pair of dims.
    fn n_pairs(&self) -> usize {
        self.head_dim / 2
    }

    /// Pairs per section, from `qwen3vl.cpp:14`'s
    /// `mrope_sections = {d_head/4, d_head/4, d_head/4, d_head/4}`.
    ///
    /// `ops.cpp:6008` indexes the section by `(i0/2) % sect_dims` where
    /// `sect_dims` is the total pair count, so a section is a contiguous run of
    /// this many pairs.
    ///
    /// A head must be wide enough to divide into `num_sections` non-empty
    /// sections. The real checkpoint's head_dim 72 gives 36 pairs, 9 per section.
    fn pairs_per_section(&self) -> usize {
        let per = self.n_pairs() / self.num_sections;
        debug_assert!(
            per > 0,
            "head_dim {} yields {} pairs, which cannot fill {} sections",
            self.head_dim,
            self.n_pairs(),
            self.num_sections
        );
        per
    }

    /// Whether this geometry can be roped at all.
    ///
    /// A head narrower than `2 * num_sections` cannot be split into non-empty
    /// sections, and the reference's `ggml_mrope_cache_init` would divide by an
    /// empty section width. This is a fixture-scale concern rather than a
    /// checkpoint one, but it must be an explicit refusal and never a panic.
    pub fn is_valid(&self) -> bool {
        self.head_dim >= 2 * self.num_sections
    }

    /// The `[cos, sin]` cache for one token's four position ids, mirroring
    /// `ggml_mrope_cache_init` with `indep_sects = true`.
    ///
    /// `ids` is `[t, h, w, e]`. Returns `n_pairs() * 2` floats.
    pub fn cache(&self, ids: &[i64]) -> Vec<f32> {
        let n = self.n_pairs();
        let per = self.pairs_per_section();
        let theta_scale = self.freq_base.powf(-2.0 / n as f32);

        // ops.cpp:5996-5999 - each axis has its own running theta, and with
        // indep_sects each is RESET to its base when its section begins
        // (ops.cpp:6009-6024).
        let mut thetas = [ids[0] as f32, ids[1] as f32, ids[2] as f32, ids[3] as f32];
        let mut cache = vec![0.0f32; n * 2];

        for pair in 0..n {
            let sector = pair % n; // (i0/2) % sect_dims, with sect_dims == n
                                   // Section index and whether this pair starts a new section.
            let sec = sector / per;
            let first_of_section = sector % per == 0;
            if first_of_section {
                // ops.cpp:6012-6023: reset the entering axis' theta to its base.
                // The bases are the position ids themselves (theta_base_* are
                // the p_t/p_h/p_w/p_e read at ops.cpp:6190-6193).
                thetas[sec] = ids[sec] as f32;
            }
            // Section order is t, h, w, e (ops.cpp:6038-6046), so sec indexes
            // straight into `thetas`.
            let theta = thetas[sec];
            cache[pair * 2] = theta.cos();
            cache[pair * 2 + 1] = theta.sin();
            // Advance only the axis this section belongs to (ops.cpp:5988).
            thetas[sec] *= theta_scale;
        }
        cache
    }

    /// Rotate one head of width `head_dim` at the given position ids.
    ///
    /// `x` is the head's dims; `ids` is `[t, h, w, e]`. Returns a new vector.
    ///
    /// The pairing is half-split: pair `p` rotates dims `p` and `p + head_dim/2`
    /// (`ops.cpp:6215` with `n_dims = head_dim/2`).
    pub fn apply(&self, x: &[f32], ids: &[i64]) -> Vec<f32> {
        let half = self.head_dim / 2;
        let cache = self.cache(ids);
        let mut out = x.to_vec();
        for pair in 0..half {
            let (cos, sin) = (cache[pair * 2], cache[pair * 2 + 1]);
            // ops.cpp:6070-6077: src = src_data + ic with ic = i0/2, and the
            // partner is src[n_offset] with n_offset = n_dims = head_dim/2.
            let (x0, x1) = (x[pair], x[pair + half]);
            out[pair] = x0 * cos - x1 * sin;
            out[pair + half] = x0 * sin + x1 * cos;
        }
        out
    }

    /// Rotate `heads` heads laid out contiguously as `[rows, heads, head_dim]`.
    ///
    /// `position_ids` is one `[t, h, w, e]` per row, so row `r`'s head `hd` uses
    /// `position_ids[r]`. Returns a new `[rows, heads, head_dim]` buffer.
    pub fn apply_batch(
        &self,
        x: &[f32],
        rows: usize,
        heads: usize,
        position_ids: &[[i64; 4]],
    ) -> Vec<f32> {
        let hd = self.head_dim;
        let stride = heads * hd;
        let mut out = vec![0.0f32; rows * stride];
        for r in 0..rows {
            let ids = position_ids[r];
            for h in 0..heads {
                let base = r * stride + h * hd;
                let head = &x[base..base + hd];
                let rotated = self.apply(head, &ids);
                out[base..base + hd].copy_from_slice(&rotated);
            }
        }
        out
    }
}

/// Position-id construction for the vision tower.
pub struct MropeCache;

impl MropeCache {
    /// The `[t, h, w, e]` position id per merged token, matching what
    /// `qwen3vl.cpp` feeds the rope.
    ///
    /// The reference's loop (`clip.cpp:4812-4829`) walks merged positions in
    /// row-major order - outer `y`, inner `x`, both stepping by `merge_ratio` -
    /// and for each writes four `(y + dy, x + dx)` corners into the t, h, w and e
    /// slots. All four corners belong to the SAME merged token, so the rope sees
    /// one id per token: the block's top-left corner.
    ///
    /// The reference fills four slots per token (`clip.cpp:4827-4830`):
    /// slot 0 = `y + dy`, slot 1 = `x + dx`, slot 2 = `y + dy`, slot 3 =
    /// `x + dx`. The rope then reads them as t, h, w, e (`ops.cpp:6190-6193`),
    /// so **t == w** and **h == e**. That is the opposite pairing to a naive
    /// reading, and it is why the third section rotates by the row position and
    /// the fourth by the column.
    ///
    /// Returns `px * py / merge^2` entries.
    pub fn position_ids(px: usize, py: usize, merge: usize) -> Vec<[i64; 4]> {
        debug_assert_eq!(px % merge, 0, "patch columns must be whole merge blocks");
        debug_assert_eq!(py % merge, 0, "patch rows must be whole merge blocks");
        // One id per PATCH, not per merged block. The reference's rope runs on the
        // pre-merge rows (the merger reshape happens after the transformer stack,
        // at qwen3vl.cpp:174-181), so the table must cover px*py entries even
        // though the four corners of a block share a column position.
        let mut ids = Vec::with_capacity(px * py);
        for y in 0..py {
            for x in 0..px {
                // clip.cpp:4827-4830 writes the four slots as
                // (y+dy, x+dx, y+dy, x+dx), and the rope reads them t, h, w, e.
                ids.push([y as i64, x as i64, y as i64, x as i64]);
            }
        }
        debug_assert_eq!(ids.len(), px * py);
        let _ = merge;
        ids
    }
}
