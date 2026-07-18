//! RCCL/ROCm lib-dir discovery, exposed for unit tests.
//!
//! The implementation lives in `build_rocm_detect.rs` (a `include!`-shared
//! source so `build.rs` and this module cannot drift). See that file for the
//! priority order and the candidate `librccl.so*` names.

include!("../build_rocm_detect.rs");
