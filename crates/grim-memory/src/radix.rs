//! Block-granular radix tree for prefix (RadixAttention-style) KV sharing.
//! One [`RadixNode`] corresponds to exactly one physical KV [`BlockId`] (matching [`crate::BLOCK_SIZE`]), so the existing block.

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
    /// WI-HYBRID Layer 2: while `Some(t)` and `t > now`, eviction
    /// (`evict_coldest_leaf`/`coldest_leaf`) skips this node — session-tagged
    /// blocks are pinned until N seconds idle.
    pub pinned_until: Option<Instant>,
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

impl RadixNode {
    fn is_pinned(&self) -> bool {
        self.pinned_until.is_some_and(|t| t > Instant::now())
    }
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
            pinned_until: None,
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

    /// Walk from the root, returning the physical blocks whose content matches the leading tokens of `tokens`, plus the number of matched tokens.
    /// Shared prefixes stop at the first non-matching block.
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
                pinned_until: None,
            });
            self.nodes[node].children.insert(key, child_idx);
            self.block_to_node.insert(bid, child_idx);
            node = child_idx;
            offset += self.block_size;
        }
    }

    /// Incremental registration: `blocks[..skip_blocks]` were already claimed
    /// or registered by this request — descend through them without touching
    /// their refcounts, then insert the tail normally. Token keys are always
    /// computed with absolute offsets into the FULL sequence, so keys match
    /// the whole-sequence hashing used by `match_prefix`.
    pub fn insert_at(&mut self, tokens: &[u32], blocks: &[usize], skip_blocks: usize) {
        let mut node = self.root;
        let mut offset = 0;
        for (i, &bid) in blocks.iter().enumerate() {
            let key = Self::block_key(tokens, offset, self.block_size);
            if i < skip_blocks {
                if let Some(&child) = self.nodes[node].children.get(&key) {
                    node = child;
                    offset += self.block_size;
                    continue;
                }
                // The covering claim was evicted mid-flight; fall through and
                // recreate the node so the tail stays reachable.
            }
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
                pinned_until: None,
            });
            self.nodes[node].children.insert(key, child_idx);
            self.block_to_node.insert(bid, child_idx);
            node = child_idx;
            offset += self.block_size;
        }
    }
    /// Drop one sequence's reference to `blocks`.
    /// Refcounts are decremented but nodes are NOT pruned here - an unreferenced prefix stays cached.
    pub fn remove(&mut self, blocks: &[usize]) {
        for &bid in blocks {
            if let Some(&idx) = self.block_to_node.get(&bid) {
                let rc = &mut self.nodes[idx].ref_count;
                *rc = rc.saturating_sub(1);
            }
        }
    }

    /// True when `bid` is still mapped by any tree node (i.e. it is cached
    /// prefix content, referenced or not). The pool consults this in
    /// `free_with_tier` so it never zeroes/free-lists a block the tree can
    /// still match against.
    pub fn contains_block(&self, bid: usize) -> bool {
        self.block_to_node.contains_key(&bid)
    }

    /// WI-HYBRID Layer 2 retention: pin the nodes mapped to `blocks` until
    /// `secs` seconds after NOW (idle-time pin — refreshed by the next
    /// session-tagged turn). Unmapped bids are ignored; expired pins are
    /// simply stale values the time comparison ignores.
    pub fn pin_blocks(&mut self, blocks: &[usize], secs: u64) {
        let until = Instant::now() + std::time::Duration::from_secs(secs);
        for &bid in blocks {
            if let Some(&idx) = self.block_to_node.get(&bid) {
                let node = &mut self.nodes[idx];
                node.pinned_until = Some(match node.pinned_until {
                    Some(t) if t > until => t,
                    _ => until,
                });
            }
        }
    }

    /// Drop expired pin entries (housekeeping — expired pins are inert by
    /// time comparison, but the map/node fields would grow without bound).
    pub fn sweep_expired_pins(&mut self) {
        let now = Instant::now();
        for node in &mut self.nodes {
            if node.pinned_until.is_some_and(|t| t <= now) {
                node.pinned_until = None;
            }
        }
    }

    /// True while the node mapped to `bid` is pin-protected from LRU eviction.
    pub fn is_pinned(&self, bid: usize) -> bool {
        self.block_to_node
            .get(&bid)
            .is_some_and(|&idx| self.nodes[idx].pinned_until.is_some_and(|t| t > Instant::now()))
    }

    /// WI-HYBRID Layer 2 (admission accounting): attached, refcount-0,
    /// unpinned nodes — all of these are reclaimable via leaf eviction plus
    /// the cascade prune (child leaves go first, then the childless parent
    /// is pruned and its page freed).
    pub fn evictable_leaves(&self) -> usize {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(idx, n)| {
                *idx != self.root
                    && n.ref_count == 0
                    && !n.is_pinned()
                    && self.block_to_node.get(&n.block_id) == Some(idx)
            })
            .count()
    }

    /// Drop the `n` oldest (earliest-expiring) pins, returning how many were
    /// actually dropped. Admission uses this as the rescue when a request's
    /// block demand can't be satisfied otherwise (spec: "pin sweep drops
    /// oldest pins when admission can't be satisfied").
    pub fn drop_oldest_pins(&mut self, n: usize) -> usize {
        let now = Instant::now();
        let mut pinned: Vec<(usize, Instant)> = self
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(idx, node)| {
                node.pinned_until
                    .filter(|t| *t > now)
                    .map(|t| (idx, t))
            })
            .collect();
        pinned.sort_by_key(|(_, t)| *t);
        let mut dropped = 0;
        for (idx, _) in pinned.into_iter().take(n) {
            self.nodes[idx].pinned_until = None;
            dropped += 1;
        }
        dropped
    }

    /// Evict the coldest childless leaf with `ref_count == 0`, returning its block id.
    /// After detaching the leaf, walks up pruning any parent that has become childless and unreferenced,.
    pub fn evict_coldest_leaf(&mut self) -> Option<usize> {
        self.evict_coldest_chain().first().copied()
    }

    /// Detach the coldest childless, unreferenced, unpinned leaf and
    /// cascade-prune childless unreferenced ancestors, returning EVERY
    /// detached block id (leaf first, then pruned ancestors). The caller owns
    /// all these pages — the pruned ancestors' mappings are gone, so their
    /// contents are garbage and their pages must return to the free list
    /// (the old leaf-only return stranded one parent page per eviction).
    pub fn evict_coldest_chain(&mut self) -> Vec<usize> {
        let mut coldest: Option<(usize, Instant)> = None;
        for (idx, node) in self.nodes.iter().enumerate() {
            if idx == self.root {
                continue;
            }
            // Detached zombie from a prior eviction: the walk-up removed it
            // from its parent and from block_to_node, but the node object
            // stays in `nodes`. Returning it would demote/reclaim stale data.
            if self.block_to_node.get(&node.block_id) != Some(&idx) {
                continue;
            }
            if !node.children.is_empty() || node.ref_count > 0 || node.is_pinned() {
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
        let Some((idx, _)) = coldest else {
            return Vec::new();
        };
        let mut detached = vec![self.nodes[idx].block_id];
        // Walk up pruning childless, unreferenced parents — their pages are
        // collected for the caller to free.
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
                    if n != idx {
                        detached.push(bid_n);
                    }
                    cur = Some(p);
                } else {
                    cur = None;
                }
            } else {
                cur = None;
            }
        }
        detached
    }

    /// Number of leaf/branch nodes (excluding root) — a rough tree-size probe.
    pub fn node_count(&self) -> usize {
        self.nodes.len().saturating_sub(1)
    }

    /// Return the block id of the coldest childless leaf with `ref_count == 0` **without removing it** from the tree.
    /// Used by pressure demotion (Phase 2.1), which keeps the cached prefix entry so a future.
    pub fn coldest_leaf(&self) -> Option<usize> {
        let mut coldest: Option<(usize, Instant)> = None;
        for (idx, node) in self.nodes.iter().enumerate() {
            if idx == self.root {
                continue;
            }
            // Detached zombie from a prior eviction: the walk-up removed it
            // from its parent and from block_to_node, but the node object
            // stays in `nodes`. Returning it would demote/reclaim stale data.
            if self.block_to_node.get(&node.block_id) != Some(&idx) {
                continue;
            }
            if !node.children.is_empty() || node.ref_count > 0 || node.is_pinned() {
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

    // ===========================================================================
    // WI-HYBRID Layer 2 (session identity): pin/retention unit tests
    // =========================================================================

    /// Two leaves; A inserted first (colder last_access). Pin A: eviction and
    /// cold-prefix selection must skip it and return B instead.
    #[test]
    fn pinned_block_is_skipped_by_eviction_and_cold_prefix() {
        let mut tree = RadixTree::new(16);
        let tokens_a: Vec<u32> = (0..16).collect();
        let tokens_b: Vec<u32> = (16..32).collect();
        tree.insert(&tokens_a, &[0]);
        tree.insert(&tokens_b, &[1]);
        // Simulate finish_request: refcounts drop to 0, mappings survive —
        // that is the state in which LRU eviction actually competes.
        tree.remove(&[0]);
        tree.remove(&[1]);
        // Sanity: without pins the colder leaf (A, inserted first) is evicted.
        assert_eq!(tree.coldest_leaf(), Some(0));

        tree.pin_blocks(&[0], 300);
        assert!(tree.is_pinned(0));
        assert!(!tree.is_pinned(1));

        // Eviction must skip the pinned coldest leaf and take B.
        assert_eq!(tree.evict_coldest_leaf(), Some(1));
        // coldest_leaf (demote_cold_prefix source) must also skip A.
        assert_eq!(tree.coldest_leaf(), None, "A is pinned; nothing else left");
        // A must survive: still mapped, still matchable.
        assert!(tree.contains_block(0));
        assert_eq!(tree.match_prefix(&tokens_a).1, 16);
    }

    /// A pin with `secs == 0` is immediately expired: the block returns to
    /// normal LRU eligibility.
    #[test]
    fn expired_pin_no_longer_protects() {
        let mut tree = RadixTree::new(16);
        tree.insert(&(0..16).collect::<Vec<u32>>(), &[7]);
        tree.remove(&[7]); // finish_request: refcount 0, mapping survives
        tree.pin_blocks(&[7], 0);
        assert!(!tree.is_pinned(7), "zero-second pin is already expired");
        assert_eq!(tree.coldest_leaf(), Some(7));
    }

    /// Re-pinning extends protection: pin for 0 (expired), then re-pin for a
    /// real duration — protection must come back (idle-time refresh).
    #[test]
    fn repin_refreshes_protection() {
        let mut tree = RadixTree::new(16);
        tree.insert(&(0..16).collect::<Vec<u32>>(), &[3]);
        tree.remove(&[3]); // finish_request simulation
        tree.pin_blocks(&[3], 0);
        assert!(!tree.is_pinned(3));
        tree.pin_blocks(&[3], 300);
        assert!(tree.is_pinned(3));
        assert_eq!(tree.coldest_leaf(), None, "re-pinned leaf must be skipped");
        assert_eq!(
            tree.evict_coldest_leaf(),
            None,
            "eviction must skip the re-pinned leaf"
        );
    }

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
