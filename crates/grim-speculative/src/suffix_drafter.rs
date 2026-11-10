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
}

const MAX_POS_PER_TRIGRAM: usize = 4;

impl SuffixDrafter {
    /// `capacity_tokens`: ring size (recent tokens retained). Default use: 8192.
    pub fn new(capacity_tokens: usize) -> Self {
        let cap = capacity_tokens.max(4);
        Self {
            capacity: cap,
            ring: Vec::with_capacity(cap.min(8192)),
            total: 0,
            index: HashMap::new(),
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

    fn push_token(&mut self, tok: u32) {
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
    }

    /// Append tokens, updating the trigram index.
    pub fn append(&mut self, tokens: &[u32]) {
        for &t in tokens {
            self.push_token(t);
        }
    }

    /// Draft up to `max_len` tokens by longest-suffix match (min match `min_match`).
    /// Returns an empty `DraftBlock` when no match qualifies.
    pub fn draft(&self, min_match: usize, max_len: usize) -> DraftBlock {
        let tokens = self.draft_tokens(min_match, max_len);
        make_block(tokens)
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
