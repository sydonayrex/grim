//! ROCm device module — the `[device]` grouping the spec's anti-pattern [see: `gemm_tuning`, `lookup_gemm_config`]

pub mod accel_features;
pub(crate) mod accel_ffi;
pub mod batch_orchestrator;
/// SCYTHE-2 WI-2: live GPU capability profiler.
pub mod capability_profiler;
pub mod cubecl;
pub mod eplb;
pub mod gemm_tuning;
pub mod handles;
pub mod hardware_spec;
pub mod helpers;
pub mod jit_cache;
pub mod layout;
pub mod moe_hybrid_exec;
pub mod probe;
pub mod roc_device;
pub mod rocblas;
pub mod scythe_route;
pub mod util;

/// Integration-test shims for the GPTQ fused dequant-GEMM path. These are
/// thin public forwardings to `pub(crate)` launchers so `tests/` binaries can
/// exercise real device kernels without widening the production API.
#[cfg(feature = "gpu-test-shims")]
pub mod gptq_test_shim {
    use crate::memory::storage::RocmStorage;
    use crate::{Result, RocmDevice};

    /// Compute GroupInt segment offsets (see
    /// [`RocmDevice::gptq_segment_offsets`]).
    #[allow(clippy::type_complexity)]
    pub fn gptq_offsets_for_test(
        bits: u8,
        group_size: usize,
        k: usize,
        n: usize,
        blob_bytes: usize,
    ) -> Result<(i64, i64, i64, i64, bool)> {
        // Reuse the private helper through a zero-sized device handle: it is
        // an associated function with no device state.
        let offsets = RocmDevice::gptq_segment_offsets(bits, group_size, k, n, blob_bytes)?;
        Ok(offsets)
    }

    /// Launch the GPTQ fused dequant-GEMM (forward).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_gptq_dequant_gemm_for_test(
        dev: &RocmDevice,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        bits: u8,
        group_size: usize,
        has_g_idx: bool,
        qw_off: i64,
        qz_off: i64,
        sc_off: i64,
        gi_off: i64,
    ) -> Result<*mut std::ffi::c_void> {
        dev.launch_gptq_dequant_gemm(
            a, b, out, m, n, k, bits, group_size, has_g_idx, qw_off, qz_off, sc_off, gi_off,
        )
    }

    /// Launch the GPTQ fused dequant-GEMM (backward).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_gptq_dequant_backward_gemm_for_test(
        dev: &RocmDevice,
        dy: &RocmStorage,
        b: &RocmStorage,
        dx: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        bits: u8,
        group_size: usize,
        has_g_idx: bool,
        qw_off: i64,
        qz_off: i64,
        sc_off: i64,
        gi_off: i64,
    ) -> Result<*mut std::ffi::c_void> {
        dev.launch_gptq_dequant_backward_gemm(
            dy, b, dx, m, n, k, bits, group_size, has_g_idx, qw_off, qz_off, sc_off, gi_off,
        )
    }

    /// Compute AWQ segment offsets (see [`RocmDevice::awq_segment_offsets`]).
    pub fn awq_offsets_for_test(
        bits: u8,
        group_size: usize,
        k: usize,
        n: usize,
        blob_bytes: usize,
    ) -> Result<(i64, i64, i64)> {
        RocmDevice::awq_segment_offsets(bits, group_size, k, n, blob_bytes)
    }

    /// Launch the AWQ fused dequant-GEMM (forward).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_awq_dequant_gemm_for_test(
        dev: &RocmDevice,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        bits: u8,
        group_size: usize,
        qw_off: i64,
        qz_off: i64,
        sc_off: i64,
    ) -> Result<*mut std::ffi::c_void> {
        dev.launch_awq_dequant_gemm(
            a, b, out, m, n, k, bits, group_size, qw_off, qz_off, sc_off,
        )
    }

    /// Launch the AWQ fused dequant-GEMM (backward).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_awq_dequant_backward_gemm_for_test(
        dev: &RocmDevice,
        dy: &RocmStorage,
        b: &RocmStorage,
        dx: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        bits: u8,
        group_size: usize,
        qw_off: i64,
        qz_off: i64,
        sc_off: i64,
    ) -> Result<*mut std::ffi::c_void> {
        dev.launch_awq_dequant_backward_gemm(
            dy, b, dx, m, n, k, bits, group_size, qw_off, qz_off, sc_off,
        )
    }

    /// Launch Charon W8A8 INT8 grouped MoE kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_charon_grouped_dispatch_w8a8_int8_for_test(
        dev: &RocmDevice,
        act: &RocmStorage,
        egate_w_ptr: u64,
        eup_w_ptr: u64,
        edown_w_ptr: u64,
        a_scale_ptr: u64,
        sorted: &crate::kernels::charon::SortedRouting,
        out: &RocmStorage,
        hidden: usize,
        inter: usize,
        num_experts: usize,
        rsf: f32,
    ) -> Result<*mut std::ffi::c_void> {
        dev.launch_charon_grouped_dispatch_w8a8_int8(
            act, egate_w_ptr, eup_w_ptr, edown_w_ptr, a_scale_ptr, sorted, out, hidden, inter, num_experts, rsf,
        )
    }

    /// Launch Charon W8A8 FP8 grouped MoE kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_charon_grouped_dispatch_w8a8_fp8_for_test(
        dev: &RocmDevice,
        act: &RocmStorage,
        egate_w_ptr: u64,
        eup_w_ptr: u64,
        edown_w_ptr: u64,
        a_scale_ptr: u64,
        sorted: &crate::kernels::charon::SortedRouting,
        out: &RocmStorage,
        hidden: usize,
        inter: usize,
        num_experts: usize,
        rsf: f32,
    ) -> Result<*mut std::ffi::c_void> {
        dev.launch_charon_grouped_dispatch_w8a8_fp8(
            act, egate_w_ptr, eup_w_ptr, edown_w_ptr, a_scale_ptr, sorted, out, hidden, inter, num_experts, rsf,
        )
    }

    /// Launch Charon AWQ grouped MoE kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_charon_grouped_dispatch_awq_for_test(
        dev: &RocmDevice,
        act: &RocmStorage,
        egate_w_ptr: u64,
        eup_w_ptr: u64,
        edown_w_ptr: u64,
        a_scale_ptr: u64,
        sorted: &crate::kernels::charon::SortedRouting,
        out: &RocmStorage,
        hidden: usize,
        inter: usize,
        num_experts: usize,
        bits: u8,
        group_size: usize,
        gate_qw_off: i64,
        gate_qz_off: i64,
        gate_sc_off: i64,
        gate_stride: u64,
        down_qw_off: i64,
        down_qz_off: i64,
        down_sc_off: i64,
        down_stride: u64,
        rsf: f32,
    ) -> Result<*mut std::ffi::c_void> {
        dev.launch_charon_grouped_dispatch_awq(
            act, egate_w_ptr, eup_w_ptr, edown_w_ptr, a_scale_ptr, sorted, out, hidden, inter, num_experts,
            bits, group_size, gate_qw_off, gate_qz_off, gate_sc_off, gate_stride, down_qw_off, down_qz_off, down_sc_off, down_stride, rsf,
        )
    }
}

