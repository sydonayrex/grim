//! Liveness interval-coloring buffer planner and decode step-level arena shrinking,
//! ported from Radiance `core/build/rad_bufplan.cpp` and `core/runtime/arena.h`.

/// Every buffer in the packed arena starts on a 256-byte boundary (Wave64 segment / RDNA4 `dwordx4`).
pub const BUF_ALIGN_BYTES: u64 = 256;

#[inline]
pub const fn align_up_buf(bytes: u64) -> u64 {
    bytes.div_ceil(BUF_ALIGN_BYTES) * BUF_ALIGN_BYTES
}

/// Lifetime class of a declared buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufKind {
    /// Scoped to `[first_def, last_use]` within a single step; shares storage with non-overlapping transients.
    Transient,
    /// Survives across steps (or written before step start); placed in the pinned bottom region `[0, fixed_bytes)`.
    Persist,
}

/// Declaration of one activation or persistent buffer in the program.
#[derive(Debug, Clone)]
pub struct BufDecl {
    pub name: String,
    pub kind: BufKind,
    pub bytes: u64,
    pub first_def: i32,
    pub last_use: i32,
}

/// Packed arena layout produced by [`plan_buffers`] or [`plan_level`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufPlan {
    /// Arena byte offset for each buffer (indexed parallel to `decls`).
    pub offsets: Vec<u64>,
    /// Pinned `[0, fixed_bytes)` region holding all [`BufKind::Persist`] buffers.
    pub fixed_bytes: u64,
    /// Packed transient extent above `fixed_bytes`.
    pub transient_bytes: u64,
    /// Total arena high-water mark (`fixed_bytes + transient_bytes`).
    pub arena_bytes: u64,
    /// Maximum bytes simultaneously live at any op index (lower bound for a perfect packer).
    pub peak_live: u64,
    /// Unshared sum of all transient buffer sizes.
    pub sum_transient: u64,
    /// Count of transient buffers with undeclared liveness (`first_def < 0 || last_use < first_def`).
    pub unknown_live: usize,
}

#[derive(Clone)]
struct Item {
    idx: usize,
    bytes: u64,
    lo: i32,
    hi: i32,
    off: Option<u64>,
}

#[inline]
fn overlaps(a: &Item, b: &Item) -> bool {
    a.lo <= b.hi && b.lo <= a.hi
}

fn first_fit(want: &Item, placed: &[Item]) -> u64 {
    let mut occ: Vec<(u64, u64)> = placed
        .iter()
        .filter_map(|p| {
            let off = p.off?;
            overlaps(p, want).then_some((off, off + p.bytes))
        })
        .collect();
    occ.sort_unstable();

    let mut off = 0u64;
    for (s, e) in occ {
        if s >= off + want.bytes {
            break;
        }
        off = off.max(align_up_buf(e));
    }
    off
}

/// Greedy interval coloring: largest buffer first, ties by `lo` then declaration `idx` (`rad_bufplan.cpp:58-70`).
fn pack_items(items: &mut [Item]) -> u64 {
    items.sort_by(|a, b| {
        b.bytes
            .cmp(&a.bytes)
            .then_with(|| a.lo.cmp(&b.lo))
            .then_with(|| a.idx.cmp(&b.idx))
    });
    let mut end = 0u64;
    for i in 0..items.len() {
        let want = items[i].clone();
        let off = first_fit(&want, items);
        items[i].off = Some(off);
        end = end.max(off + want.bytes);
    }
    end
}

/// Compute a deterministic single-allocation arena plan (`Persist` at `[0, fixed_bytes)`,
/// `Transient` interval-colored above `fixed_bytes`).
pub fn plan_buffers(decls: &[BufDecl], total_ops: i32) -> BufPlan {
    let last_op = total_ops.max(0);
    let mut offsets = vec![0u64; decls.len()];
    let mut transient = Vec::new();
    let mut persist = Vec::new();
    let mut unknown_live = 0usize;

    for (idx, d) in decls.iter().enumerate() {
        let aligned = align_up_buf(d.bytes.max(1));
        let (lo, hi) = if d.first_def < 0 || d.last_use < d.first_def {
            if d.kind == BufKind::Transient {
                unknown_live += 1;
            }
            (0, last_op)
        } else {
            (d.first_def, d.last_use)
        };
        let it = Item {
            idx,
            bytes: aligned,
            lo,
            hi,
            off: None,
        };
        match d.kind {
            BufKind::Persist => persist.push(it),
            BufKind::Transient => transient.push(it),
        }
    }

    let sum_transient: u64 = transient.iter().map(|t| t.bytes).sum();
    let mut peak_live = 0u64;
    for t in 0..=last_op {
        let live: u64 = transient
            .iter()
            .filter(|it| it.lo <= t && t <= it.hi)
            .map(|it| it.bytes)
            .sum();
        peak_live = peak_live.max(live);
    }

    let mut off = 0u64;
    for it in &persist {
        offsets[it.idx] = off;
        off = align_up_buf(off + it.bytes);
    }
    let fixed_bytes = align_up_buf(off);

    let transient_bytes = pack_items(&mut transient);
    for it in &transient {
        offsets[it.idx] = fixed_bytes + it.off.unwrap_or(0);
    }

    BufPlan {
        offsets,
        fixed_bytes,
        transient_bytes,
        arena_bytes: fixed_bytes + transient_bytes,
        peak_live,
        sum_transient,
        unknown_live,
    }
}

/// Compute a shrunk step-level arena layout (`rad_bufplan.cpp:151-213`):
/// keeps all `Persist` buffers at their exact `full_plan` offsets and repacks `Transient` buffers
/// at `shrunk_bytes` so `(level.arena_bytes .. full_plan.arena_bytes)` can be lent to the MoE slab.
pub fn plan_level(
    full_decls: &[BufDecl],
    full_plan: &BufPlan,
    shrunk_bytes: &[u64],
    total_ops: i32,
) -> Result<BufPlan, String> {
    if full_decls.len() != shrunk_bytes.len() {
        return Err(format!(
            "plan_level: decl count {} != shrunk_bytes count {}",
            full_decls.len(),
            shrunk_bytes.len()
        ));
    }
    let last_op = total_ops.max(0);
    let mut offsets = full_plan.offsets.clone();
    let mut transient = Vec::new();

    for (idx, (d, &sb)) in full_decls.iter().zip(shrunk_bytes.iter()).enumerate() {
        if d.kind == BufKind::Persist {
            continue;
        }
        if sb > d.bytes {
            return Err(format!(
                "buffer '{}' is larger in level probe ({} B > full {} B)",
                d.name, sb, d.bytes
            ));
        }
        let (lo, hi) = if d.first_def < 0 || d.last_use < d.first_def {
            (0, last_op)
        } else {
            (d.first_def, d.last_use)
        };
        transient.push(Item {
            idx,
            bytes: align_up_buf(sb.max(1)),
            lo,
            hi,
            off: None,
        });
    }

    let sum_transient: u64 = transient.iter().map(|t| t.bytes).sum();
    let mut peak_live = 0u64;
    for t in 0..=last_op {
        let live: u64 = transient
            .iter()
            .filter(|it| it.lo <= t && t <= it.hi)
            .map(|it| it.bytes)
            .sum();
        peak_live = peak_live.max(live);
    }

    let transient_bytes = pack_items(&mut transient);
    for it in &transient {
        offsets[it.idx] = full_plan.fixed_bytes + it.off.unwrap_or(0);
    }

    Ok(BufPlan {
        offsets,
        fixed_bytes: full_plan.fixed_bytes,
        transient_bytes,
        arena_bytes: full_plan.fixed_bytes + transient_bytes,
        peak_live,
        sum_transient,
        unknown_live: 0,
    })
}
