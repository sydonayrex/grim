//! Qwen Sparse Attention block allocator.
//!
//! Transcribed from `old/repo/llama.cpp-master/src/llama-memory-hybrid-idx.cpp`
//! at commit `d7241ac8`, `llama_memory_hybrid_idx::set_input_qsa`.
//!
//! # Why this exists
//!
//! A QSA block is not `cell / r`. Upstream keys blocks on
//! `(sequence set, index bucket)`:
//!
//! > "a block is keyed on (sequence set, index bucket): a unified cache counts
//! > every sequence from zero, so the bucket alone would pool two sequences
//! > into one block"
//!
//! A paged cache makes that mandatory: two sequences can hold cells at
//! positions 0 and 4, and `pos / r` would pool them into one block, letting
//! one sequence attend to the other's content. The allocator therefore groups
//! cells by bucket AND by their sequence set, only promotes a group to a block
//! once all `r` slots are filled, and parks every unpooled cell in a single
//! spare "dead" block.
//!
//! # What this replaces
//!
//! The earlier implementation pooled `cell / r` over a dense contiguous cache.
//! That is correct only for one sequence with no holes, which is exactly what
//! the synthetic test builds, so it passed every test while being wrong for
//! any paged or multi-sequence cache.
//!
//! # Scope
//!
//! The mrope / 2D-position re-ranking path is modelled explicitly but not
//! reimplemented: it exists to break ties when `mrope` repeats a position
//! across an image frame, which a text-only run never hits. [`BlockLayout::new`]
//! takes the cell positions and their sequence sets, which is the whole input
//! the text path needs.

use grim_core::error::{Error, Result};

/// The sentinel upstream uses for "this cell belongs to no poolable block".
///
/// Upstream parks unpooled cells in a single spare block rather than leaving
/// them unmapped, so every cell has a `cell_blk` entry. We record the spare
/// block id explicitly instead, which is equivalent and keeps the bias logic
/// readable.
pub const NO_BLOCK: i32 = -1;

/// Cell metadata the allocator needs. One entry per cache cell.
#[derive(Clone, Debug, PartialEq)]
pub struct CellInfo {
    /// Position of this cell in its sequence, or `None` if the cell is empty.
    pub pos: Option<u32>,
    /// Which sequences may read this cell. Two cells with different sets must
    /// never be pooled together. Empty for an empty cell.
    pub seqs: Vec<u32>,
}

impl CellInfo {
    /// An empty cache cell.
    pub fn empty() -> Self {
        Self {
            pos: None,
            seqs: Vec::new(),
        }
    }

    /// A filled cell at `pos`, readable by the single sequence `seq`.
    pub fn new(pos: u32, seq: u32) -> Self {
        Self {
            pos: Some(pos),
            seqs: vec![seq],
        }
    }

    /// A filled cell readable by several sequences.
    pub fn with_seqs(pos: u32, seqs: Vec<u32>) -> Self {
        Self {
            pos: Some(pos),
            seqs,
        }
    }
}

/// The result of allocating blocks over a cache window.
#[derive(Clone, Debug, PartialEq)]
pub struct BlockLayout {
    /// Number of poolable blocks, `n_bid`. Each holds exactly `r` cells.
    pub n_bid: usize,
    /// `blk_of[j]`: the block holding cell `j`, or [`NO_BLOCK`].
    pub blk_of: Vec<i32>,
    /// `blk_cells[b * r + slot]`: the cell in block `b` at slot, or -1 if the
    /// slot is empty. Only full blocks are emitted, so every slot is filled.
    pub blk_cells: Vec<i32>,
    /// `blk_pos[b]`: the start position of block `b`, i.e. `bucket * r`.
    pub blk_pos: Vec<u32>,
    /// The spare block id unpooled cells are parked in, if any exists.
    ///
    /// Upstream: "a spare block exists only when some cell is unpooled:
    /// `n_bid == n_blocks` means every cell sits in a full block."
    pub dead_bid: Option<usize>,
    /// True when a cell's bucket is at or past `n_blocks`, which upstream
    /// treats as an out-of-range condition it asserts against in block-bias
    /// mode.
    pub out_of_range: bool,
}

impl BlockLayout {
    /// Pooled key for block `b`, averaged over the block's `r` cells.
    ///
    /// `k_raw` is the raw per-cell indexer key history, `n_kv` cells of
    /// `idx_dim` values. A block with no members yields zeros, matching
    /// upstream's zero-filled `pooled` before the norm.
    pub fn pooled(&self, k_raw: &[f32], n_kv: usize, idx_dim: usize, r: usize) -> Result<Vec<f32>> {
        if r == 0 {
            return Err(Error::Config("qsa: block ratio r must be positive".into()));
        }
        if k_raw.len() < n_kv * idx_dim {
            return Err(Error::Shape(format!(
                "qsa: indexer keys hold {} values, need {} for n_kv={n_kv} idx_dim={idx_dim}",
                k_raw.len(),
                n_kv * idx_dim
            )));
        }
        let mut pooled = vec![0.0f32; self.n_bid * idx_dim];
        for b in 0..self.n_bid {
            let mut n = 0usize;
            for slot in 0..r {
                let cell = self.blk_cells[b * r + slot];
                if cell < 0 || cell as usize >= n_kv {
                    continue;
                }
                let cell = cell as usize;
                let src = &k_raw[cell * idx_dim..(cell + 1) * idx_dim];
                let dst = &mut pooled[b * idx_dim..(b + 1) * idx_dim];
                for (d, s) in dst.iter_mut().zip(src.iter()) {
                    *d += *s;
                }
                n += 1;
            }
            if n > 0 {
                let inv = 1.0f32 / n as f32;
                for d in &mut pooled[b * idx_dim..(b + 1) * idx_dim] {
                    *d *= inv;
                }
            }
        }
        Ok(pooled)
    }

    /// Per-block bias for a query at absolute position `q`, in block-bias mode.
    ///
    /// Port of the `blk_bias` branch:
    ///
    /// - a block with no cells, or one belonging to another sequence, is
    ///   `-inf`
    /// - the tail (an incomplete block at or past `tail_start`) gets `+1e9`, so
    ///   it always wins the top-k: "the tail is an incomplete block and is
    ///   always visible, as in the reference"
    /// - every other visible block gets `0.0`
    ///
    /// All values stay finite except the masked ones, because a `-inf` row
    /// alone produces a NaN in the subsequent softmax. Upstream is explicit
    /// about this: "finite, so it can never meet a -inf and produce a nan".
    ///
    /// `q` is the rank of the query among visible cells; `visible` says whether
    /// each cell belongs to the query's sequence and is readable.
    pub fn block_bias(&self, q_pos: u32, r: usize, visible: &dyn Fn(usize) -> bool) -> Vec<f32> {
        let tail_start = ((q_pos as usize + 1) / r) * r;
        let mut bias = vec![f32::NEG_INFINITY; self.n_bid];
        for b in 0..self.n_bid {
            if !self.block_visible(b, visible) {
                continue;
            }
            bias[b] = if self.blk_pos[b] as usize >= tail_start {
                1e9f32
            } else {
                0.0f32
            };
        }
        bias
    }

    /// Whether any member cell of block `b` is visible to the query.
    fn block_visible(&self, b: usize, visible: &dyn Fn(usize) -> bool) -> bool {
        let r = (self.blk_cells.len() / self.n_bid.max(1)).max(1);
        for slot in 0..r {
            let cell = self.blk_cells[b * r + slot];
            if cell >= 0 && visible(cell as usize) {
                return true;
            }
        }
        false
    }

    /// Whether the query at `q_pos` can see the whole of block `b`.
    ///
    /// Eq. (15): a block is scored only once every one of its `r` tokens has
    /// been observed, i.e. `p_b + r - 1 <= q`.
    pub fn block_is_observed(&self, b: usize, q_pos: u32, r: usize) -> bool {
        let p_b = self.blk_pos.get(b).copied().unwrap_or(0) as usize;
        p_b + r - 1 <= q_pos as usize
    }
}

/// Allocate QSA blocks over a cache window.
///
/// `n_blocks` is the number of buckets, i.e. how many blocks the window could
/// hold if every bucket were full. Upstream derives it from the block-position
/// tensor's width divided by the 4 mrope sections; a text-only run passes the
/// number of buckets directly.
///
/// `r` is the compression ratio. Upstream asserts `r <= 64` because the slot
/// occupancy set is a `u64` bitmask.
///
/// # Errors
/// Returns [`Error::Config`] if `r` is zero or greater than 64, matching
/// upstream's `GGML_ASSERT(r > 0)` and `GGML_ASSERT(r <= 64)`.
pub fn allocate_blocks(cells: &[CellInfo], n_blocks: usize, r: usize) -> Result<BlockLayout> {
    if r == 0 {
        return Err(Error::Config("qsa: block ratio r must be positive".into()));
    }
    if r > 64 {
        return Err(Error::Config(format!(
            "qsa: block ratio {r} exceeds the 64-slot occupancy limit"
        )));
    }
    let n_kv = cells.len();
    let slots_full: u64 = if r == 64 { u64::MAX } else { (1u64 << r) - 1 };

    // Group cells by (bucket, sequence set). `group_head[bucket]` is the head
    // of a singly linked list of groups in that bucket; `group_next` chains
    // them. Upstream uses the same intrusive-list shape.
    let mut group_head = vec![-1i32; n_blocks];
    let mut group_next: Vec<i32> = Vec::new();
    let mut group_first: Vec<i32> = Vec::new();
    let mut group_slot0: Vec<i32> = Vec::new();
    let mut group_slots: Vec<u64> = Vec::new();
    // Block id a group was promoted to, or NO_BLOCK. Recorded here so the
    // cell -> block map is a direct lookup: searching `bid_cell` for a match
    // can alias two distinct groups that happen to share a first cell.
    let mut group_bid: Vec<i32> = Vec::new();

    let mut blk_of = vec![NO_BLOCK; n_kv];
    let mut cell_grp = vec![-1i32; n_kv];
    let mut out_of_range = false;

    for j in 0..n_kv {
        let Some(idx) = cells[j].pos else {
            continue; // empty cell
        };
        let bucket = (idx as usize) / r;
        if bucket >= n_blocks {
            out_of_range = true;
            continue;
        }
        // Find an existing group in this bucket with the same sequence set.
        //
        // `one_seq` is "the cache as a whole holds at most one sequence", NOT
        // "every cell names at most one sequence": the latter is true for a
        // cache holding three different sequences and would pool them together,
        // which is the exact bug this module exists to prevent. Upstream
        // computes it by counting sequences present and comparing to 1.
        let one_seq = {
            let mut distinct: Vec<&Vec<u32>> = Vec::new();
            for c in cells.iter() {
                if !c.seqs.is_empty() && !distinct.iter().any(|d| **d == c.seqs) {
                    distinct.push(&c.seqs);
                }
            }
            distinct.len() <= 1
        };
        let mut g: i32 = -1;
        let mut cur = group_head[bucket];
        while cur >= 0 {
            let c = cur as usize;
            let same = one_seq
                || cells[group_first[c] as usize]
                    .seqs
                    .iter()
                    .eq(cells[j].seqs.iter());
            if same {
                g = cur;
                break;
            }
            cur = group_next[c];
        }
        if g < 0 {
            g = group_first.len() as i32;
            group_next.push(group_head[bucket]);
            group_first.push(j as i32);
            group_slot0.push(-1);
            group_slots.push(0);
            group_bid.push(NO_BLOCK);
            group_head[bucket] = g;
        }
        let g = g as usize;
        let bit = 1u64 << ((idx as usize) % r);
        group_slots[g] |= bit;
        cell_grp[j] = g as i32;
        if (idx as usize) % r == 0 {
            group_slot0[g] = j as i32;
        }
    }

    // Promote only FULL groups to blocks. An incomplete group cannot be
    // pooled: its mean would be over fewer than r cells, which changes the
    // scale the norm then sees.
    let mut n_bid = 0usize;
    let mut bid_idx: Vec<u32> = Vec::new();
    for pb in 0..n_blocks {
        let mut g = group_head[pb];
        while g >= 0 {
            let gi = g as usize;
            if group_slots[gi] == slots_full {
                group_bid[gi] = n_bid as i32;
                bid_idx.push((pb * r) as u32);
                n_bid += 1;
            }
            g = group_next[gi];
        }
    }
    debug_assert!(n_bid <= n_blocks);

    // Assign blocks back to cells and record each block's members.
    let mut blk_cells = vec![-1i32; n_bid * r];
    for j in 0..n_kv {
        let g = cell_grp[j];
        if g < 0 {
            continue;
        }
        let b = group_bid[g as usize];
        if b < 0 {
            continue; // incomplete group, never promoted
        }
        blk_of[j] = b;
        let idx = cells[j].pos.unwrap() as usize;
        blk_cells[b as usize * r + (idx % r)] = j as i32;
    }

    // A spare block exists only when some cell is unpooled.
    let dead_bid = if n_bid < n_blocks { Some(n_bid) } else { None };

    Ok(BlockLayout {
        n_bid,
        blk_of,
        blk_cells,
        blk_pos: bid_idx,
        dead_bid,
        out_of_range,
    })
}
