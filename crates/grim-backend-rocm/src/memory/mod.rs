//! Device-side memory subsystem (Phase-3 §3.1 of the QKV spec). [see: `memory/`, `lib.rs`, `pool`, `DeviceScratchPool`]

pub mod allocator;
pub mod budget;
pub mod hugepage;
pub mod pinned;
pub mod pool;
pub mod storage;
