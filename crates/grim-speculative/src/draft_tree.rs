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
            if item.depth >= max_depth as isize {
                continue;
            }
            let child_depth = item.depth + 1;
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
            if child_depth >= max_depth as isize {
                // Node was inserted above; it is a leaf.
                continue;
            }
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

    /// Parent array in node order for the tree-attention kernel: depth-0
    /// candidates use 0 as their own parent (the kernel's root sentinel).
    pub fn parents_u32(&self) -> Vec<u32> {
        self.nodes
            .iter()
            .map(|n| n.parent.unwrap_or(0) as u32)
            .collect()
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

fn softmax_vec(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum.is_finite() && sum > 0.0 {
        exps.into_iter().map(|v| v / sum).collect()
    } else {
        vec![1.0 / logits.len() as f32; logits.len()]
    }
}

/// Proposal probability of one tree path, computed from the one-pass
/// per-position draft marginals. Used by stochastic tree verification;
/// `DraftTreeNode::log_mass` is raw-logit ranking only and must not be used
/// as a probability.
pub fn path_q(
    tree: &DraftTree,
    node: usize,
    draft_logits: &[f32],
    vocab: usize,
) -> f64 {
    let path = tree.path_to(node);
    let mut q = 1.0f64;
    for (depth, &tok) in path.iter().enumerate() {
        let start = depth * vocab;
        let Some(row) = draft_logits.get(start..start + vocab) else {
            return 0.0;
        };
        let p = softmax_vec(row);
        q *= p.get(tok as usize).copied().unwrap_or(0.0) as f64;
    }
    q
}

/// Lossless stochastic tree verification using per-step modified rejection.
///
/// `target_logits` has one row per node in BFS order; `draft_logits` has one
/// row per tree depth. At a rejected node the residual `max(0, p - q)` is
/// sampled and every descendant is invalidated. When the whole path survives,
/// the bonus token is drawn from the target row of the deepest node.
pub fn verify_tree_stochastic(
    tree: &DraftTree,
    target_logits: &[f32],
    base_row: &[f32],
    draft_logits: &[f32],
    vocab: usize,
    uniforms: &[f32],
) -> TreeVerification {
    if vocab == 0 || tree.nodes.is_empty() {
        return TreeVerification {
            best_node: None,
            emitted_tokens: vec![0],
        };
    }
    let mut accepted = vec![false; tree.nodes.len()];
    let mut best: Option<usize> = None;
    let mut rejected_at: Option<usize> = None;
    for (i, node) in tree.nodes.iter().enumerate() {
        let ancestor_ok = match node.parent {
            None => true,
            Some(p) => accepted.get(p).copied().unwrap_or(false),
        };
        if !ancestor_ok || rejected_at.is_some() {
            continue;
        }
        let Some(target_row) = target_logits.get(i * vocab..i * vocab + vocab) else {
            rejected_at = Some(i);
            break;
        };
        let p = softmax_vec(target_row);
        let q_row = draft_logits
            .get(node.depth * vocab..node.depth * vocab + vocab)
            .map(softmax_vec)
            .unwrap_or_default();
        let q = q_row.get(node.token as usize).copied().unwrap_or(0.0);
        let p_tok = p.get(node.token as usize).copied().unwrap_or(0.0);
        let accept_prob = if q > 0.0 {
            (p_tok / q).min(1.0)
        } else {
            1.0
        };
        let u = uniforms.get(i).copied().unwrap_or(1.0);
        if u <= accept_prob {
            accepted[i] = true;
            if best.map_or(true, |b| node.depth > tree.nodes[b].depth) {
                best = Some(i);
            }
            continue;
        }
        rejected_at = Some(i);
        break;
    }

    if let Some(i) = rejected_at {
        let node = &tree.nodes[i];
        let target_row = target_logits
            .get(i * vocab..i * vocab + vocab)
            .unwrap_or(base_row);
        let p = softmax_vec(target_row);
        let q = draft_logits
            .get(node.depth * vocab..node.depth * vocab + vocab)
            .map(softmax_vec)
            .unwrap_or_default();
        let residual_total: f32 = p
            .iter()
            .zip(q.iter())
            .map(|(&pv, &qv)| (pv - qv).max(0.0))
            .sum();
        let threshold = uniforms.get(tree.nodes.len()).copied().unwrap_or(0.5)
            * residual_total;
        let mut cum = 0.0f32;
        let mut sampled = 0usize;
        for v in 0..vocab {
            let pv = p.get(v).copied().unwrap_or(0.0);
            let qv = q.get(v).copied().unwrap_or(0.0);
            cum += (pv - qv).max(0.0);
            if cum >= threshold {
                sampled = v;
                break;
            }
            sampled = v;
        }
        let mut emitted = match best {
            Some(b) => tree.path_to(b),
            None => Vec::new(),
        };
        emitted.push(sampled as u32);
        return TreeVerification {
            best_node: best,
            emitted_tokens: emitted,
        };
    }

    let mut emitted = match best {
        Some(b) => tree.path_to(b),
        None => Vec::new(),
    };
    let bonus_row = best
        .and_then(|b| target_logits.get(b * vocab..b * vocab + vocab))
        .unwrap_or(base_row);
    let p = softmax_vec(bonus_row);
    let threshold = uniforms
        .get(tree.nodes.len())
        .copied()
        .unwrap_or(0.5);
    let mut cum = 0.0f32;
    let mut bonus = p.len().saturating_sub(1);
    for (v, &pv) in p.iter().enumerate() {
        cum += pv;
        if cum >= threshold {
            bonus = v;
            break;
        }
    }
    emitted.push(bonus as u32);
    TreeVerification {
        best_node: best,
        emitted_tokens: emitted,
    }
}

/// Per-node recurrent state for a tree, keyed by parent. This is the host-side
/// representation a true packed-tree trunk needs for KDA/short-conv branching;
/// the model-side branch that consumes it is still a follow-up.
#[derive(Debug, Clone)]
pub struct TreeState<T> {
    states: Vec<T>,
    parents: Vec<Option<usize>>,
}

impl<T: Clone> TreeState<T> {
    pub fn new(parents: Vec<Option<usize>>, root: T) -> Self {
        let states = vec![root; parents.len().max(1)];
        Self { states, parents }
    }

    pub fn parent(&self, node: usize) -> Option<usize> {
        self.parents.get(node).copied().flatten()
    }

    pub fn set(&mut self, node: usize, state: T) {
        if let Some(slot) = self.states.get_mut(node) {
            *slot = state;
        }
    }

    pub fn get(&self, node: usize) -> Option<&T> {
        self.states.get(node)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits(rows: &[Vec<f32>], _vocab: usize) -> Vec<f32> {
        rows.iter().flat_map(|r| r.iter().copied()).collect()
    }

    #[test]
    fn tree_state_tracks_parent_lineage() {
        let mut state = TreeState::new(vec![None, Some(0), Some(0)], 0u32);
        assert_eq!(state.parent(0), None);
        assert_eq!(state.parent(1), Some(0));
        state.set(1, 11);
        state.set(2, *state.get(1).unwrap());
        assert_eq!(state.get(2), Some(&11));
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

    #[test]
    fn parents_u32_uses_root_sentinel() {
        let tree = DraftTree {
            nodes: vec![
                DraftTreeNode { token: 1, parent: None, depth: 0, log_mass: 0.0 },
                DraftTreeNode { token: 2, parent: Some(0), depth: 1, log_mass: 0.0 },
            ],
        };
        assert_eq!(tree.parents_u32(), vec![0, 0]);
    }

    #[test]
    fn path_q_uses_normalised_draft_marginals() {
        let tree = DraftTree {
            nodes: vec![DraftTreeNode {
                token: 1, parent: None, depth: 0, log_mass: 999.0,
            }],
        };
        let draft = logits(&[vec![0.0, 2.0, 0.0, 0.0]], 4);
        let q = path_q(&tree, 0, &draft, 4);
        let expected = 2.0f64.exp() / (1.0 + 2.0f64.exp() + 1.0 + 1.0);
        assert!((q - expected).abs() < 1e-6, "q={q}, expected={expected}");
    }

    #[test]
    fn stochastic_tree_residual_and_bonus() {
        let tree = DraftTree {
            nodes: vec![
                DraftTreeNode { token: 1, parent: None, depth: 0, log_mass: 0.0 },
                DraftTreeNode { token: 2, parent: Some(0), depth: 1, log_mass: 0.0 },
            ],
        };
        let base = [0.0f32, 9.0, 0.0, 0.0];
        // Draft marginals: depth 0 mostly token 1, depth 1 mostly token 2.
        let draft = logits(&[vec![0.0, 4.0, 0.0, 0.0], vec![0.0, 0.0, 4.0, 0.0]], 4);
        // Target rows strongly favour both drafted tokens; u=0 accepts both.
        let target = logits(&[vec![0.0, 9.0, 0.0, 0.0], vec![0.0, 0.0, 9.0, 0.0]], 4);
        let r = verify_tree_stochastic(&tree, &target, &base, &draft, 4, &[0.0, 0.0, 0.0]);
        assert_eq!(r.best_node, Some(1));
        assert_eq!(r.emitted_tokens.len(), 3);
        // A target row favouring another token at node 0 rejects; residual
        // draw picks from max(0, p - q).
        let reject_target =
            logits(&[vec![9.0, 0.0, 0.0, 0.0], vec![0.0, 0.0, 9.0, 0.0]], 4);
        let r = verify_tree_stochastic(
            &tree, &reject_target, &base, &draft, 4, &[0.999, 0.0, 0.0],
        );
        assert_eq!(r.best_node, None);
        assert_eq!(r.emitted_tokens.len(), 1);
    }
}
