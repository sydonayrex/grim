//! Tree drafting from one-pass block logits (CAST-style top-product tree).
//!
//! Pure CPU data structure + greedy verification oracle. Production tree
//! attention is a backend concern; this module owns only the candidate tree
//! and the lossless-for-greedy acceptance rule. Fixed fanout schedules are
//! deliberately not implemented (CAST reports they lose to a chain).

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A drafted candidate node. `parent` indexes into the owning tree's `nodes`.
#[derive(Debug, Clone, PartialEq)]
pub struct DraftTreeNode {
    pub token: u32,
    pub parent: Option<usize>,
    pub depth: usize,
    /// Cumulative log product of the proposer's marginals along this path.
    pub log_mass: f64,
}

/// Build configuration for a best-first prefix-closed tree.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TreeBuildConfig {
    /// Per-position rank cap retained from the proposer's marginals.
    pub rank_cap: usize,
    /// Non-root node budget.
    pub max_nodes: usize,
    /// Maximum path depth.
    pub max_depth: usize,
}

impl Default for TreeBuildConfig {
    fn default() -> Self {
        Self {
            rank_cap: 8,
            max_nodes: 32,
            max_depth: 8,
        }
    }
}

/// A prefix-closed candidate tree in BFS insertion order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DraftTree {
    pub nodes: Vec<DraftTreeNode>,
}

#[derive(Clone, Copy)]
struct HeapItem {
    score: f64,
    parent: Option<usize>,
    token: u32,
    /// Next token depth to emit; -1 is the virtual root.
    depth: isize,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.score.total_cmp(&other.score) == Ordering::Equal
    }
}
impl Eq for HeapItem {}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.total_cmp(&other.score)
    }
}

impl DraftTree {
    /// Grow a tree from a `[positions, vocab]` block of proposer logits.
    ///
    /// Ties are resolved deterministically by smaller token id (the heap's
    /// insertion order), so the same logits always produce the same tree.
    pub fn from_block_logits(
        logits: &[f32],
        vocab: usize,
        config: TreeBuildConfig,
    ) -> Self {
        if vocab == 0 || logits.len() < vocab {
            return Self::default();
        }
        let positions = logits.len() / vocab;
        let rank_cap = config.rank_cap.max(1);
        let max_depth = config.max_depth.min(positions);
        let mut heap: BinaryHeap<HeapItem> = BinaryHeap::new();
        heap.push(HeapItem {
            score: 0.0,
            parent: None,
            token: 0,
            depth: -1,
        });
        let mut nodes: Vec<DraftTreeNode> = Vec::with_capacity(config.max_nodes);

        while let Some(item) = heap.pop() {
            let child_depth = item.depth + 1;
            if child_depth >= max_depth as isize {
                continue;
            }
            // The virtual root has no node; every other item becomes a node
            // before its children are enqueued, keeping the tree prefix-closed.
            let parent = if item.depth < 0 {
                None
            } else {
                if nodes.len() >= config.max_nodes {
                    break;
                }
                let idx = nodes.len();
                nodes.push(DraftTreeNode {
                    token: item.token,
                    parent: item.parent,
                    depth: item.depth as usize,
                    log_mass: item.score,
                });
                Some(idx)
            };
            // Proposer row `child_depth` predicts a token at that depth.
            let row_start = child_depth as usize * vocab;
            let row = &logits[row_start..row_start + vocab];
            // Collect the rank_cap largest logits; tie-break by token id.
            let mut top: Vec<(f64, u32)> = row
                .iter()
                .enumerate()
                .map(|(v, &l)| (l as f64, v as u32))
                .collect();
            top.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            for (logit, token) in top.into_iter().take(rank_cap) {
                // Score is a product of marginals; unnormalised logits still
                // give the same ranking, which is all the builder needs.
                heap.push(HeapItem {
                    score: item.score + logit,
                    parent,
                    token,
                    depth: child_depth,
                });
            }
        }
        Self { nodes }
    }

    /// Packed token ids in node order (for a tree-attention query tensor).
    pub fn tokens(&self) -> Vec<u32> {
        self.nodes.iter().map(|n| n.token).collect()
    }

    /// Parent indices in node order (`None` for depth-0 candidates).
    pub fn parents(&self) -> Vec<Option<usize>> {
        self.nodes.iter().map(|n| n.parent).collect()
    }

    /// Token path from the root to `node` inclusive.
    pub fn path_to(&self, node: usize) -> Vec<u32> {
        let mut out = Vec::new();
        let mut cur = Some(node);
        while let Some(i) = cur {
            out.push(self.nodes[i].token);
            cur = self.nodes[i].parent;
        }
        out.reverse();
        out
    }

    pub fn is_prefix_closed(&self) -> bool {
        self.nodes
            .iter()
            .enumerate()
            .all(|(i, n)| match n.parent {
                None => n.depth == 0,
                Some(p) => p < i && self.nodes[p].depth + 1 == n.depth,
            })
    }
}

/// Result of a greedy tree verification pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeVerification {
    /// Deepest accepted node index (`None` when no candidate matched).
    pub best_node: Option<usize>,
    /// Accepted path tokens followed by the target's bonus token.
    pub emitted_tokens: Vec<u32>,
}

fn argmax_row(row: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    best as u32
}

/// Greedy tree verification oracle. `target_logits` has one `[vocab]` row per
/// tree node in node order; `base_row` supplies the root's bonus distribution
/// when no candidate is accepted. A rejected node invalidates its whole
/// subtree, which is the ancestor-mask rule tree attention must enforce.
pub fn verify_tree_greedy(
    tree: &DraftTree,
    target_logits: &[f32],
    base_row: &[f32],
    vocab: usize,
) -> TreeVerification {
    if vocab == 0 || base_row.is_empty() {
        return TreeVerification {
            best_node: None,
            emitted_tokens: vec![0],
        };
    }
    let mut accepted = vec![false; tree.nodes.len()];
    let mut best: Option<usize> = None;
    for (i, node) in tree.nodes.iter().enumerate() {
        let ancestor_ok = match node.parent {
            None => true,
            Some(p) => accepted.get(p).copied().unwrap_or(false),
        };
        let start = i * vocab;
        let row_ok = target_logits.get(start..start + vocab)
            .map(|row| argmax_row(row) == node.token)
            .unwrap_or(false);
        accepted[i] = ancestor_ok && row_ok;
        if accepted[i]
            && best.map_or(true, |b| node.depth > tree.nodes[b].depth)
        {
            best = Some(i);
        }
    }
    match best {
        Some(i) => {
            let mut emitted = tree.path_to(i);
            let start = i * vocab;
            let row = target_logits
                .get(start..start + vocab)
                .unwrap_or(base_row);
            emitted.push(argmax_row(row));
            TreeVerification {
                best_node: Some(i),
                emitted_tokens: emitted,
            }
        }
        None => TreeVerification {
            best_node: None,
            emitted_tokens: vec![argmax_row(base_row)],
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits(rows: &[Vec<f32>], _vocab: usize) -> Vec<f32> {
        rows.iter().flat_map(|r| r.iter().copied()).collect()
    }

    #[test]
    fn tree_is_prefix_closed_and_bounded() {
        let mut row = vec![0.0f32; 8];
        row[3] = 4.0;
        row[5] = 3.0;
        let flat: Vec<f32> = (0..4).flat_map(|_| row.iter().copied()).collect();
        let tree = DraftTree::from_block_logits(
            &flat,
            8,
            TreeBuildConfig {
                rank_cap: 2,
                max_nodes: 6,
                max_depth: 4,
            },
        );
        assert!(!tree.nodes.is_empty());
        assert!(tree.nodes.len() <= 6);
        assert!(tree.is_prefix_closed());
        assert!(tree.nodes.iter().all(|n| n.depth <= 4));
    }

    #[test]
    fn rejected_node_invalidates_descendants() {
        let tree = DraftTree {
            nodes: vec![
                DraftTreeNode { token: 1, parent: None, depth: 0, log_mass: 0.0 },
                DraftTreeNode { token: 2, parent: Some(0), depth: 1, log_mass: 0.0 },
            ],
        };
        let base = [0.0f32, 9.0, 0.0, 0.0];
        // Node 0 accepts (argmax 1), node 1 also accepts.
        let good = logits(&[vec![0.0, 9.0, 0.0, 0.0], vec![0.0, 0.0, 9.0, 0.0]], 4);
        let r = verify_tree_greedy(&tree, &good, &base, 4);
        assert_eq!(r.best_node, Some(1));
        assert_eq!(r.emitted_tokens, vec![1, 2, 2]);
        // Node 0 rejects: the descendant must not be considered.
        let bad = logits(&[vec![9.0, 0.0, 0.0, 0.0], vec![0.0, 0.0, 9.0, 0.0]], 4);
        let r = verify_tree_greedy(&tree, &bad, &base, 4);
        assert_eq!(r.best_node, None);
        assert_eq!(r.emitted_tokens, vec![1]); // argmax of base row
    }
}
