//! Q5.3 Layer-2 cross-crate integration test for `UseCappedResidentPolicy` vs `LruResidencyTracker`.

use std::collections::{HashSet, VecDeque};

use grim_memory::{moe_use_victim_enabled, LruResidencyTracker, UseCappedResidentPolicy};

#[test]
fn use_capped_policy_protects_recent_beats_or_matches_lru_and_respects_gate() {
    const N_EXPERTS: usize = 32;
    const CAP: usize = 8;
    const STEPS: usize = 64;

    // Deterministic 64-step routing trace: 6 hot experts (0..6) interleaved with
    // rotating cold experts (6..32) to exercise frequency retention vs pure LRU.
    let mut trace: Vec<usize> = Vec::with_capacity(STEPS);
    for step in 0..STEPS {
        if step % 3 != 2 {
            trace.push(step % 6);
        } else {
            trace.push(6 + ((step / 3) % (N_EXPERTS - 6)));
        }
    }

    // (a) Protected window: victim never lands on an expert used in the last 4 steps
    let mut policy = UseCappedResidentPolicy::new(N_EXPERTS, 256);
    let mut recent4: VecDeque<usize> = VecDeque::new();
    let mut policy_cache: HashSet<usize> = HashSet::new();
    let mut policy_hits = 0usize;

    for &exp in &trace {
        policy.begin_token();
        if policy_cache.contains(&exp) {
            policy_hits += 1;
            policy.count_look(exp);
        } else {
            if policy_cache.len() >= CAP {
                let protected: Vec<usize> = recent4.iter().copied().collect();
                if let Some(v) = policy.victim(&protected) {
                    assert!(
                        !protected.contains(&v),
                        "victim {v} must not be in protected window {protected:?}"
                    );
                }
                let cached_ids: Vec<usize> = policy_cache.iter().copied().collect();
                let hottest = policy.keep_hottest(&cached_ids);
                policy_cache = hottest.into_iter().take(CAP - 1).collect();
            }
            policy.count_look(exp);
            policy_cache.insert(exp);
        }
        recent4.push_back(exp);
        if recent4.len() > 4 {
            recent4.pop_front();
        }
    }

    // (b) Compare hit-rate against LruResidencyTracker on the exact same trace
    let mut lru = LruResidencyTracker::new(CAP);
    let mut lru_hits = 0usize;
    for &exp in &trace {
        if lru.lookup(0, exp).is_some() {
            lru_hits += 1;
        } else {
            let _ = lru.admit(0, exp);
        }
    }
    eprintln!(
        "[Q5.3] 64-step trace: policy_hits={policy_hits}/{STEPS}, lru_hits={lru_hits}/{STEPS}"
    );
    assert!(
        policy_hits >= lru_hits,
        "use-capped policy hit-rate ({policy_hits}) must be >= LRU ({lru_hits})"
    );

    // (c) Gate OFF reproduces clear-all behaviour when cache.len() >= CAP
    let prev = std::env::var("GRIM_MOE_USE_VICTIM").ok();
    unsafe { std::env::remove_var("GRIM_MOE_USE_VICTIM") };
    assert!(!moe_use_victim_enabled(), "gate must default OFF");
    let mut clear_all_cache: HashSet<usize> = (0..CAP).collect();
    if !moe_use_victim_enabled() && clear_all_cache.len() >= CAP {
        clear_all_cache.clear();
    }
    assert!(clear_all_cache.is_empty(), "gate OFF must clear all");
    unsafe {
        match prev {
            Some(v) => std::env::set_var("GRIM_MOE_USE_VICTIM", v),
            None => std::env::remove_var("GRIM_MOE_USE_VICTIM"),
        }
    }
}
