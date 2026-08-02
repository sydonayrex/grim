//! ROCm device module — the `[device]` grouping the spec's anti-pattern [see: `gemm_tuning`, `lookup_gemm_config`]

pub mod accel_features;
pub mod accel_ffi;
/// SCYTHE-2 WI-2: live GPU capability profiler.
pub mod capability_profiler;
pub mod cubecl;
pub mod gemm_tuning;
pub mod handles;
pub mod helpers;
pub mod layout;
pub mod probe;
pub mod roc_device;
pub mod rocblas;
pub mod util;
