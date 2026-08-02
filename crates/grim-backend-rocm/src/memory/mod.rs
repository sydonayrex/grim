//! Device-side memory subsystem (Phase-3 §3.1 of the QKV spec). [see: `memory/`, `lib.rs`, `pool`, `DeviceScratchPool`]

pub mod allocator;
pub mod pinned;
pub mod pool;
pub mod storage;
