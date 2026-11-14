use grim_backend_rocm::device::probe::parse_kfd_mem_bank_is_private_vram;
use grim_backend_rocm::memory::ledger::{
    FREED_RING_CAP, UnownedDiagnosis, attribute, deregister, diagnose_unowned, register, reset,
};

#[test]
fn freed_allocation_ring_identifies_use_after_free_and_buffer_overrun() {
    reset();
    register(0x10_0000, 4096, 0, false, "layer 7 kv_k");
    register(0x20_0000, 2048, 0, false, "layer 7 attn_q");

    // Free kv_k; a fault inside [0x10_0000, 0x10_1000) is now Unowned in live table
    // but resolves to UseAfterFree in diagnose_unowned.
    deregister(0x10_0000);
    let _ = attribute(0x10_0080, 0);
    match diagnose_unowned(0x10_0080) {
        Some(UnownedDiagnosis::UseAfterFree { record, offset, .. }) => {
            assert_eq!(record.owner, "layer 7 kv_k");
            assert_eq!(offset, 0x80);
        }
        other => panic!("expected UseAfterFree, got {other:?}"),
    }

    // A fault 64 bytes past live `layer 7 attn_q` [0x20_0000, 0x20_0800) resolves to BufferOverrun.
    match diagnose_unowned(0x20_0800 + 64) {
        Some(UnownedDiagnosis::BufferOverrun {
            record,
            bytes_past_end,
        }) => {
            assert_eq!(record.owner, "layer 7 attn_q");
            assert_eq!(bytes_past_end, 64);
        }
        other => panic!("expected BufferOverrun, got {other:?}"),
    }
}

/// Ported from Radiance `aql.cpp:760-828`:
/// 1. `FREED_RING_CAP = 1024` FIFO eviction: freeing 1025 allocations evicts #0 while #1..=1024 resolve.
/// 2. Re-leased address range: when a freed address is re-used and freed again, the newest owner wins.
/// 3. Closest-preceding live allocation tie-break for buffer overruns within `OVERRUN_WINDOW_BYTES`.
#[test]
fn radiance_oracle_freed_ring_1024_fifo_wrap_and_closest_overrun_tiebreak() {
    reset();

    // 1. Register and free 1025 allocations (`0..=1024`); entry 0 must be evicted.
    for i in 0..=(FREED_RING_CAP as u64) {
        let base = 0x1000_0000 + i * 0x2000;
        register(base, 4096, 0, false, &format!("freed_{i}"));
        deregister(base);
    }
    assert!(
        diagnose_unowned(0x1000_0000 + 16).is_none(),
        "entry 0 must be evicted after 1025 frees"
    );
    match diagnose_unowned(0x1000_0000 + 0x2000 + 32) {
        Some(UnownedDiagnosis::UseAfterFree { record, offset, .. }) => {
            assert_eq!(record.owner, "freed_1");
            assert_eq!(offset, 32);
        }
        other => panic!("expected freed_1, got {other:?}"),
    }

    // 2. Re-allocating and freeing the same VA returns the most recent owner (`rev()` scan).
    let reused_va = 0x5000_0000;
    register(reused_va, 4096, 0, false, "old_owner");
    deregister(reused_va);
    register(reused_va, 4096, 0, false, "new_owner");
    deregister(reused_va);
    match diagnose_unowned(reused_va + 8) {
        Some(UnownedDiagnosis::UseAfterFree { record, .. }) => {
            assert_eq!(record.owner, "new_owner");
        }
        other => panic!("expected new_owner, got {other:?}"),
    }

    // 3. Two adjacent live buffers within 64 KiB: fault picks the narrowest positive overrun delta.
    register(0x8000_0000, 4096, 0, false, "buf_far");
    register(0x8000_2000, 4096, 0, false, "buf_near");
    match diagnose_unowned(0x8000_3000 + 128) {
        Some(UnownedDiagnosis::BufferOverrun {
            record,
            bytes_past_end,
        }) => {
            assert_eq!(record.owner, "buf_near");
            assert_eq!(bytes_past_end, 128);
        }
        other => panic!("expected buf_near overrun, got {other:?}"),
    }
}

/// Ported from Radiance `aql.cpp:295-319` + `tests/diag_test.cpp:70-100`:
/// Only `heap_type 2` (`HSA_HEAPTYPE_FRAME_BUFFER_PRIVATE`) with `size_in_bytes > 0`
/// flags disabled Resizable BAR; empty (`size_in_bytes 0`) or public (`heap_type 1`) banks pass.
#[test]
fn kfd_mem_bank_parser_detects_non_rebar_private_framebuffer() {
    let rebar_on = "heap_type 1\nsize_in_bytes 25753026560\nflags 0\n";
    let rebar_off = "heap_type 2\nsize_in_bytes 25501368320\nflags 0\n";
    let empty_private_bank = "heap_type 2\nsize_in_bytes 0\nflags 0\n";
    assert!(!parse_kfd_mem_bank_is_private_vram(rebar_on));
    assert!(parse_kfd_mem_bank_is_private_vram(rebar_off));
    assert!(!parse_kfd_mem_bank_is_private_vram(empty_private_bank));
}
