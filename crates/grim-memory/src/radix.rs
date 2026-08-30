//! Block-granular radix tree for prefix (RadixAttention-style) KV sharing.
//!
//! One [`RadixNode`] corresponds to exactly one physical KV [`BlockId`]
//! (matching [`crate::BLOCK_SIZE`]), so the existing block allocator is
//! untouched: sharing is always whole-block, which is the only unit the KV
//! cache can reuse.
//!
//! `children` is keyed by a content hash of a block's tokens (the leading
//! token is insufficient — two distinct blocks could otherwise collide on a
//! shared first token while differing later in the block). Because each node
//! is one atomic block, branching happens at block boundaries; there is no
//! partial-block split (a half-block of KV can never be shared).

use std::collections::HashMap;
use std::time::Instant;

/// Content hash used to key a child node: FNV-1a over one block of tokens.
pub type TokenKey = u64;

/// A single tree node — one physical KV block.
#[derive(Debug)]
pub struct RadixNode {
    /// Physical block this node owns.
    pub block_id: usize,
    /// Token range this block covers (e.g. `[0,16)`, `[16,32)`).
    pub token_span: std::ops::Range<usize>,
    /// Next-block content hash → child node index.
    pub children: HashMap<TokenKey, usize>,
    /// Parent node index (root has `None`).
    pub parent: Option<usize>,
    /// Number of sequences whose prefix traverses this node.
    pub ref_count: u32,
    /// Last time this node (or a descendant) was matched/inserted.
    pub last_access: Instant,
    /// Checkpoint ID of attached recurrent/hybrid layer state at this block boundary (if any).
    pub recurrent_state_id: Option<usize>,
}

/// Block-granular radix tree over request token sequences.
pub struct RadixTree {
    nodes: Vec<RadixNode>,
    root: usize,
    block_size: usize,
    /// Reverse map so `remove` can locate a node from its block id.
    block_to_node: HashMap<usize, usize>,
}

impl RadixTree {
    /// Build an empty tree. `block_size` must match the pool's
    /// [`crate::BLOCK_SIZE`].
    pub fn new(block_size: usize) -> Self {
        let root = RadixNode {
            block_id: usize::MAX,
            token_span: 0..0,
            children: HashMap::new(),
            parent: None,
            ref_count: 0,
            last_access: Instant::now(),
            recurrent_state_id: None,
        };
        Self {
            nodes: vec![root],
            root: 0,
            block_size,
            block_to_node: HashMap::new(),
        }
    }

    /// Hash the tokens of one block starting at `offset`.
    fn block_key(tokens: &[u32], offset: usize, block_size: usize) -> TokenKey {
        let mut h: TokenKey = 0xcbf2_9ce4_8422_2325;
        let end = (offset + block_size).min(tokens.len());
        for &token in &tokens[offset..end] {
            h ^= token as TokenKey;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    /// Walk from the root, returning the physical blocks whose content
    /// matches the leading tokens of `tokens`, plus the number of matched
    /// tokens. Shared prefixes stop at the first non-matching block.
    pub fn match_prefix(&self, tokens: &[u32]) -> (Vec<usize>, usize) {
        let (matched, offset, _) = self.match_prefix_with_anchor(tokens);
        (matched, offset)
    }

    /// Walk from the root, returning matched blocks, full token count, and whether blending is available.
    pub fn match_prefix_blending(&self, tokens: &[u32]) -> (Vec<usize>, usize, bool) {
        let (matched, full_tokens) = self.match_prefix(tokens);
        let blended = full_tokens < tokens.len() && !matched.is_empty();
        (matched, full_tokens, blended)
    }

    /// Walk from the root, returning the matched physical blocks, token count,
    /// and the deepest valid `recurrent_state_id` anchored along the matched path.
    pub fn match_prefix_with_anchor(&self, tokens: &[u32]) -> (Vec<usize>, usize, Option<usize>) {
        let mut matched = Vec::new();
        let mut node = self.root;
        let mut offset = 0;
        let mut deepest_state_id = None;

        while offset + self.block_size <= tokens.len() {
            let key = Self::block_key(tokens, offset, self.block_size);
            match self.nodes[node].children.get(&key) {
                Some(&child) => {
                    matched.push(self.nodes[child].block_id);
                    if let Some(st_id) = self.nodes[child].recurrent_state_id {
                        deepest_state_id = Some(st_id);
                    }
                    offset += self.block_size;
                    node = child;
                }
                None => break,
            }
        }
        (matched, offset, deepest_state_id)
    }

    /// Attach a recurrent-state checkpoint ID to the node corresponding to `block_id`.
    pub fn attach_recurrent_state(&mut self, block_id: usize, state_id: usize) {
        if let Some(&node_idx) = self.block_to_node.get(&block_id) {
            self.nodes[node_idx].recurrent_state_id = Some(state_id);
        }
    }

    /// Touch `last_access` along the matched path so eviction prefers
    /// genuinely cold leaves.
    pub fn touch(&mut self, tokens: &[u32]) {
        let mut node = self.root;
        let mut offset = 0;
        while offset + self.block_size <= tokens.len() {
            let key = Self::block_key(tokens, offset, self.block_size);
            match self.nodes[node].children.get(&key) {
                Some(&child) => {
                    self.nodes[child].last_access = Instant::now();
                    offset += self.block_size;
                    node = child;
                }
                None => break,
            }
        }
    }

    /// Register newly computed `blocks` for `tokens`. Shared prefix nodes
    /// have their refcount incremented; diverging blocks become new nodes.
    pub fn insert(&mut self, tokens: &[u32], blocks: &[usize]) {
        let mut node = self.root;
        let mut offset = 0;
        for &bid in blocks {
            let key = Self::block_key(tokens, offset, self.block_size);
            if let Some(&child) = self.nodes[node].children.get(&key) {
                // Shared prefix: reuse the existing node, bump refcount.
                self.nodes[child].ref_count += 1;
                node = child;
                offset += self.block_size;
                continue;
            }
            let child_idx = self.nodes.len();
            self.nodes.push(RadixNode {
                block_id: bid,
                token_span: offset..(offset + self.block_size),
                children: HashMap::new(),
                parent: Some(node),
                ref_count: 1,
                last_access: Instant::now(),
                recurrent_state_id: None,
            });
            self.nodes[node].children.insert(key, child_idx);
            self.block_to_node.insert(bid, child_idx);
            node = child_idx;
            offset += self.block_size;
        }
    }

    /// Drop one sequence's reference to `blocks`. Refcounts are decremented
    /// but nodes are NOT pruned here — an unreferenced prefix stays cached
    /// (refcount 0) for future reuse until [`RadixTree::evict_coldest_leaf`]
    /// reclaims it under pressure. This matches RadixAttention semantics:
    /// prefixes are cached until evicted, not deleted the moment a sequence
    /// ends.
    pub fn remove(&mut self, blocks: &[usize]) {
        for &bid in blocks {
            if let Some(&idx) = self.block_to_node.get(&bid) {
                let rc = &mut self.nodes[idx].ref_count;
                *rc = rc.saturating_sub(1);
            }
        }
    }

    /// Evict the coldest childless leaf with `ref_count == 0`, returning its
    /// block id. After detaching the leaf, walks up pruning any parent that
    /// has become childless and unreferenced, so eviction never reclaims a
    /// block another request's partial prefix still depends on. Returns
    /// `None` if nothing is evictable.
    pub fn evict_coldest_leaf(&mut self) -> Option<usize> {
        let mut coldest: Option<(usize, Instant)> = None;
        for (idx, node) in self.nodes.iter().enumerate() {
            if idx == self.root {
                continue;
            }
            if !node.children.is_empty() || node.ref_count > 0 {
                continue;
            }
            match coldest {
                None => coldest = Some((idx, node.last_access)),
                Some((_, cold_time)) if node.last_access < cold_time => {
                    coldest = Some((idx, node.last_access))
                }
                _ => {}
            }
        }
        let (idx, _) = coldest?;
        let bid = self.nodes[idx].block_id;
        // Walk up pruning childless, unreferenced parents.
        let mut cur = Some(idx);
        while let Some(n) = cur {
            let (bid_n, has_children, parent) = {
                let node = &self.nodes[n];
                (node.block_id, !node.children.is_empty(), node.parent)
            };
            if n != self.root && !has_children {
                if let Some(p) = parent {
                    let key = self.nodes[p]
                        .children
                        .iter()
                        .find(|(_, v)| **v == n)
                        .map(|(k, _)| *k);
                    if let Some(k) = key {
                        self.nodes[p].children.remove(&k);
                    }
                    self.block_to_node.remove(&bid_n);
                    cur = Some(p);
                } else {
                    cur = None;
                }
            } else {
                cur = None;
            }
        }
        Some(bid)
    }

    /// Number of leaf/branch nodes (excluding root) — a rough tree-size probe.
    pub fn node_count(&self) -> usize {
        self.nodes.len().saturating_sub(1)
    }

    /// Return the block id of the coldest childless leaf with `ref_count ==
    /// 0` **without removing it** from the tree. Used by pressure demotion
    /// (Phase 2.1), which keeps the cached prefix entry so a future request
    /// can still match and promote it back. Returns `None` if there is no
    /// cold leaf to demote.
    pub fn coldest_leaf(&self) -> Option<usize> {
        let mut coldest: Option<(usize, Instant)> = None;
        for (idx, node) in self.nodes.iter().enumerate() {
            if idx == self.root {
                continue;
            }
            if !node.children.is_empty() || node.ref_count > 0 {
                continue;
            }
            match coldest {
                None => coldest = Some((idx, node.last_access)),
                Some((_, cold_time)) if node.last_access < cold_time => {
                    coldest = Some((idx, node.last_access))
                }
                _ => {}
            }
        }
        coldest.map(|(idx, _)| self.nodes[idx].block_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn match_then_insert_full_prefix_is_idempotent() {
        let mut tree = RadixTree::new(16);
        let tokens: Vec<u32> = (0..48).collect(); // three blocks
        // No match initially.
        assert_eq!(tree.match_prefix(&tokens), (vec![], 0));

        // Insert three blocks.
        tree.insert(&tokens, &[10, 11, 12]);
        let (matched, n) = tree.match_prefix(&tokens);
        assert_eq!(matched, vec![10, 11, 12]);
        assert_eq!(n, 48);

        // Re-inserting the same sequence must reuse nodes (refcount bump),
        // not allocate new ones.
        tree.insert(&tokens, &[10, 11, 12]);
        let (matched2, _) = tree.match_prefix(&tokens);
        assert_eq!(matched2, vec![10, 11, 12]);
    }

    #[test]
    fn partial_prefix_sharing_branches_after_divergence() {
        let mut tree = RadixTree::new(16);
        let base: Vec<u32> = (0..32).collect(); // two shared blocks
        let mut a = base.clone();
        a.extend_from_slice(&[
            100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
        ]); // +1 block
        let mut b = base.clone();
        b.extend_from_slice(&[
            200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216,
        ]); // +1 block (diverges)

        tree.insert(&a, &[10, 11, 12]);
        // b shares the first two blocks (same content) but diverges at block 3.
        let (matched_b, n) = tree.match_prefix(&b);
        assert_eq!(matched_b, vec![10, 11]);
        assert_eq!(n, 32);

        tree.insert(&b, &[10, 11, 13]);
        // After insert, b's full prefix is present.
        let (matched_b2, n2) = tree.match_prefix(&b);
        assert_eq!(matched_b2, vec![10, 11, 13]);
        assert_eq!(n2, 48);

        // a is unaffected by b's insert.
        let (matched_a, _) = tree.match_prefix(&a);
        assert_eq!(matched_a, vec![10, 11, 12]);
    }

    #[test]
    fn remove_keeps_unreferenced_prefix_cached() {
        // remove() only decrements refcounts; an unreferenced prefix stays
        // cached (refcount 0) for future reuse until evicted.
        let mut tree = RadixTree::new(16);
        let a: Vec<u32> = (0..48).collect();
        tree.insert(&a, &[10, 11, 12]);
        // Two sequences share the prefix; insert b which diverges at block 3.
        let mut b = a[..32].to_vec();
        b.extend(std::iter::repeat_n(999u32, 16));
        tree.insert(&b, &[10, 11, 20]);

        // Remove sequence a. Its prefix is now unreferenced but still cached.
        tree.remove(&[10, 11, 12]);
        assert_eq!(tree.match_prefix(&a), (vec![10, 11, 12], 48));
        assert_eq!(tree.match_prefix(&b), (vec![10, 11, 20], 48));
    }

    #[test]
    fn evict_coldest_leaf_prunes_unshared_tail_and_walks_up() {
        let mut tree = RadixTree::new(16);
        let a: Vec<u32> = (0..48).collect();
        tree.insert(&a, &[10, 11, 12]);
        // b shares the first two blocks, diverges at block 3.
        let mut b = a[..32].to_vec();
        b.extend(std::iter::repeat_n(777u32, 16));
        tree.insert(&b, &[10, 11, 20]);

        // Remove sequence a → its unique tail block 12 is now unreferenced.
        tree.remove(&[10, 11, 12]);

        // Eviction reclaims the coldest unreferenced leaf (block 12).
        let evicted = tree.evict_coldest_leaf();
        assert_eq!(evicted, Some(12));

        // a's prefix is now truncated at the divergence point.
        assert_eq!(tree.match_prefix(&a), (vec![10, 11], 32));
        // b's prefix (including shared 10,11 and its own tail 20) is intact.
        assert_eq!(tree.match_prefix(&b), (vec![10, 11, 20], 48));
    }
}
