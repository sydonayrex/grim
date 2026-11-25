//! Trigram suffix (prompt-lookup) drafter (astrum P6).
//!
//! Indexes a recent token ring by trigram and drafts by finding the longest
//! suffix of the current tail that also occurred earlier, emitting the tokens
//! that followed that occurrence. O(1) append; bounded memory (ring + capped
//! position lists). Default-off integration; usable standalone.

use crate::draft_backbone::DraftBlock;
use grim_backend_cpu::CpuStorage;
use grim_tensor::{DType, Device, QuantProvenance, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

/// Trigram-indexed prompt-lookup drafter over a recent token stream.
pub struct SuffixDrafter {
    capacity: usize,
    ring: Vec<u32>,
    total: usize, // total tokens ever appended (ring head == total % capacity when full)
    // trigram -> ring positions (absolute indices in [total-len, total)); capped.
    index: HashMap<(u32, u32, u32), Vec<usize>>,
    // Bounded suffix trie over the same ring, rebuilt lazily on eviction.
    trie: Vec<TrieNode>,
    trie_base: usize,
}

#[derive(Default, Clone)]
struct TrieNode {
    children: Vec<(u32, usize)>,
    /// Absolute end positions of occurrences whose prefix ends at this node.
    occurrences: Vec<usize>,
}

const MAX_POS_PER_TRIGRAM: usize = 4;
const MAX_TRIE_DEPTH: usize = 64;
const MAX_TRIE_NODES: usize = 16_384;

impl SuffixDrafter {
    /// `capacity_tokens`: ring size (recent tokens retained). Default use: 8192.
    pub fn new(capacity_tokens: usize) -> Self {
        let cap = capacity_tokens.max(4);
        Self {
            capacity: cap,
            ring: Vec::with_capacity(cap.min(8192)),
            total: 0,
            index: HashMap::new(),
            trie: vec![TrieNode::default()],
            trie_base: 0,
        }
    }

    fn ring_len(&self) -> usize {
        self.ring.len()
    }

    fn base_abs(&self) -> usize {
        self.total.saturating_sub(self.ring_len())
    }

    fn get_abs(&self, abs: usize) -> Option<u32> {
        let base = self.base_abs();
        if abs < base || abs >= self.total {
            return None;
        }
        Some(self.ring[abs - base])
    }

    fn rebuild_trie(&mut self) {
        self.trie.clear();
        self.trie.push(TrieNode::default());
        self.trie_base = self.base_abs();
        let base = self.trie_base;
        for start in base..self.total {
            if self.trie.len() >= MAX_TRIE_NODES {
                break;
            }
            self.insert_suffix(start);
        }
    }

    fn insert_suffix(&mut self, start: usize) {
        let mut node = 0usize;
        for depth in 1..=MAX_TRIE_DEPTH {
            let Some(tok) = self.get_abs(start + depth - 1) else {
                break;
            };
            let child = if let Some(pos) = self.trie[node]
                .children
                .iter()
                .position(|(t, _)| *t == tok)
            {
                self.trie[node].children[pos].1
            } else {
                if self.trie.len() >= MAX_TRIE_NODES {
                    return;
                }
                let idx = self.trie.len();
                self.trie.push(TrieNode::default());
                self.trie[node].children.push((tok, idx));
                idx
            };
            let occ = &mut self.trie[child].occurrences;
            let end = start + depth - 1;
            if !occ.contains(&end) {
                if occ.len() >= MAX_POS_PER_TRIGRAM {
                    occ.remove(0);
                }
                occ.push(end);
            }
            node = child;
        }
    }

    fn push_token(&mut self, tok: u32) {
        if self.trie_base != self.base_abs() {
            self.rebuild_trie();
        }
        if self.ring.len() < self.capacity {
            self.ring.push(tok);
        } else {
            let slot = self.total % self.capacity;
            self.ring[slot] = tok;
        }
        let abs_pos = self.total; // position of this token
        self.total += 1;

        // index trigram ending at abs_pos (needs 3 tokens)
        if abs_pos >= 2 {
            if let (Some(a), Some(b), Some(c)) = (
                self.get_abs(abs_pos - 2),
                self.get_abs(abs_pos - 1),
                self.get_abs(abs_pos),
            ) {
                let key = (a, b, c);
                let v = self.index.entry(key).or_default();
                // cap per trigram, keep newest
                if v.len() >= MAX_POS_PER_TRIGRAM {
                    v.remove(0);
                }
                v.push(abs_pos);
            }
        }
        // Incremental trie insertion for the new suffix. Rebuild above keeps
        // occurrences inside the live ring after eviction.
        // Every suffix ending at this token grows by one node. Inserting all
        // active starts keeps the trie incremental without replaying the ring.
        let earliest = self
            .base_abs()
            .max(abs_pos.saturating_sub(MAX_TRIE_DEPTH - 1));
        for start in earliest..=abs_pos {
            self.insert_suffix(start);
        }
    }

    /// Append tokens, updating the trigram index and suffix trie.
    pub fn append(&mut self, tokens: &[u32]) {
        for &t in tokens {
            self.push_token(t);
        }
    }

    /// Draft up to `max_len` tokens by longest-suffix match (min match `min_match`).
    /// Returns an empty `DraftBlock` when no match qualifies.
    pub fn draft(&self, min_match: usize, max_len: usize) -> DraftBlock {
        let tokens = self.draft_tokens_trie(min_match, max_len);
        let tokens = if tokens.is_empty() {
            self.draft_tokens(min_match, max_len)
        } else {
            tokens
        };
        make_block(tokens)
    }

    /// Longest-match query over the bounded suffix trie. Walking the tail
    /// forward through the trie is O(match length) rather than the trigram
    /// index's O(match length × candidates) retry loop.
    fn draft_tokens_trie(&self, min_match: usize, max_len: usize) -> Vec<u32> {
        let min_m = min_match.max(3);
        let tail_len = (min_m + max_len).min(self.ring_len());
        if tail_len < min_m || self.trie.len() <= 1 {
            return Vec::new();
        }
        let tail_start = self.total - tail_len;
        let tail: Vec<u32> = (tail_start..self.total)
            .filter_map(|p| self.get_abs(p))
            .collect();
        // The trie stores suffixes in forward order. For each match length m,
        // walk the suffix `tail[len-m..]`; stop at the first missing edge.
        for m in 1..=tail.len().min(MAX_TRIE_DEPTH) {
            let suffix = &tail[tail.len() - m..];
            let mut node = 0usize;
            let mut found = true;
            for &tok in suffix {
                let Some((_, child)) = self.trie[node]
                    .children
                    .iter()
                    .find(|(t, _)| *t == tok)
                else {
                    found = false;
                    break;
                };
                node = *child;
            }
            if !found {
                break;
            }
            if m < min_m {
                continue;
            }
            // The candidate must precede the current suffix of length m.
            let limit = self.total - m;
            for &cand_end in self.trie[node].occurrences.iter().rev() {
                if cand_end < self.base_abs() || cand_end >= limit {
                    continue; // evicted or overlapping the current suffix
                }
                let cand_start = cand_end + 1 - m;
                if cand_start < self.base_abs() {
                    continue;
                }
                let mut out = Vec::new();
                let mut src = cand_start + m;
                while out.len() < max_len && src < limit {
                    match self.get_abs(src) {
                        Some(t) => out.push(t),
                        None => break,
                    }
                    src += 1;
                }
                if !out.is_empty() {
                    return out;
                }
            }
        }
        Vec::new()
    }

    fn draft_tokens(&self, min_match: usize, max_len: usize) -> Vec<u32> {
        let len = self.ring_len();
        let min_m = min_match.max(3);
        if len < min_m {
            return Vec::new();
        }
        let base = self.base_abs();
        let end = self.total; // exclusive, current tail end
        // try longest match lengths first (capped so continuation can exist)
        let max_match = len.saturating_sub(1).min(min_m.saturating_add(max_len));
        for m in (min_m..=max_match).rev() {
            let suffix_start = end - m;
            // trigram key from first 3 of suffix
            let Some(a) = self.get_abs(suffix_start) else { continue };
            let Some(b) = self.get_abs(suffix_start + 1) else { continue };
            let Some(c) = self.get_abs(suffix_start + 2) else { continue };
            let Some(cands) = self.index.get(&(a, b, c)) else { continue };
            // check candidates newest-first, skip any that overlap the suffix or are stale
            for &cand_end_abs in cands.iter().rev() {
                // cand_end_abs is the abs pos of the 3rd trigram token (suffix_start+2 match)
                let cand_start = match cand_end_abs.checked_sub(2) {
                    Some(v) => v,
                    None => continue,
                };
                if cand_start < base {
                    continue; // evicted
                }
                // ensure candidate occurrence + continuation doesn't overlap current suffix
                if cand_start + m >= suffix_start {
                    continue;
                }
                // verify full match of length m
                if !self.matches(cand_start, suffix_start, m) {
                    continue;
                }
                // emit tokens following the matched occurrence
                let mut out = Vec::new();
                for j in 0..max_len {
                    let src = cand_start + m + j;
                    if src >= suffix_start {
                        break; // don't read into/overlap the suffix itself
                    }
                    if let Some(tok) = self.get_abs(src) {
                        out.push(tok);
                    } else {
                        break;
                    }
                }
                if !out.is_empty() {
                    return out;
                }
            }
        }
        Vec::new()
    }

    fn matches(&self, a_start: usize, b_start: usize, m: usize) -> bool {
        for i in 0..m {
            match (self.get_abs(a_start + i), self.get_abs(b_start + i)) {
                (Some(x), Some(y)) if x == y => {}
                _ => return false,
            }
        }
        true
    }
}

fn make_block(tokens: Vec<u32>) -> DraftBlock {
    let n = tokens.len().max(1);
    let base_logits = Tensor::new(
        Arc::new(CpuStorage::new(
            vec![0.0f32; n],
            Shape::new(vec![n, 1]),
            DType::F32,
        )) as Arc<dyn grim_tensor::BackendStorage>,
        Shape::new(vec![n, 1]),
        DType::F32,
        QuantProvenance::GrimNative,
        Device::Cpu,
    );
    let confidence = vec![1.0f32; tokens.len()];
    DraftBlock {
        tokens,
        base_logits,
        confidence,
    }
}
