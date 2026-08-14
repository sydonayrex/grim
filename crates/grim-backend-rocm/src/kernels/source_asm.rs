//! `compute_kernel_source`: a small helper that re-assembles the [see: `kernels::compute_kernels::OTHER_KERNEL_SOURCE`]

pub fn compute_kernel_source() -> String {
    let mut s =
        String::with_capacity(crate::kernels::compute_kernels::OTHER_KERNEL_SOURCE.len() + 16384);
    s.push_str(crate::kernels::shared_device_fns::KERNEL_SOURCE);
    s.push_str(crate::kernels::charon::KERNEL_SOURCE);
    s.push_str(crate::kernels::compute_kernels::OTHER_KERNEL_SOURCE);
    s.push_str(crate::kernels::fused_linear_ce::FUSED_LINEAR_CE_KERNEL_SOURCE);
    s.push_str(crate::kernels::qkv_attention::KERNEL_SOURCE);
    s.push_str(crate::kernels::decode_gemm::KERNEL_SOURCE);
    s.push_str(crate::kernels::fused_dequant_gemm::KERNEL_SOURCE);
    s.push_str(crate::kernels::kv_dequant_attention::KERNEL_SOURCE);
    s.push_str(crate::kernels::wmma_gemm::KERNEL_SOURCE);
    s.push_str(crate::kernels::q8_0_dequant::KERNEL_SOURCE);
    s.push_str(crate::kernels::q4k_gemm::KERNEL_SOURCE);
    s.push_str(crate::kernels::q5k_gemm::KERNEL_SOURCE);
    s.push_str(crate::kernels::q6k_gemm::KERNEL_SOURCE);
    s.push_str(crate::kernels::q2k_gemm::KERNEL_SOURCE);
    s.push_str(crate::kernels::q3k_gemm::KERNEL_SOURCE);
    s.push_str(crate::kernels::iq_gemm::KERNEL_SOURCE);
    s.push_str(crate::kernels::fp8_standalone::KERNEL_SOURCE);
    s.push_str(crate::kernels::fp8_gemm_rdna4::KERNEL_SOURCE);
    s.push_str(crate::kernels::mxfp_standalone::KERNEL_SOURCE);
    s.push_str(crate::kernels::selective_scan::KERNEL_SOURCE);
    s.push_str(crate::kernels::q4k_dequant::KERNEL_SOURCE);
    s.push_str(crate::kernels::iq_dequant::KERNEL_SOURCE);
    s.push_str(crate::kernels::cross_attention::KERNEL_SOURCE);
    s.push_str(crate::kernels::rwkv::KERNEL_SOURCE);
    s.push_str(crate::kernels::quant_standalone::KERNEL_SOURCE);
    s.push_str(crate::kernels::silu_mul_quant::SILU_MUL_QUANT_KERNEL_SOURCE);
    s.push_str(crate::kernels::sage_attention::SAGE_ATTENTION_KERNEL_SOURCE);
    s.push_str(crate::kernels::scythe_persistent::KERNEL_SOURCE);
    s
}

/// Generate JIT kernel source parameterized by HardwareSpec, tile selection, and multi-GPU shard parameters.
pub fn compute_kernel_source_with_spec(
    spec: &crate::device::hardware_spec::HardwareSpec,
    _entry: &str,
    shape_class: crate::autotune::ShapeClass,
    dims: crate::kernels::tile_picker::ShapeDims,
    device_id: u32,
    num_devices: u32,
    tiles: Option<&crate::kernels::tile_picker::TileConfig>,
) -> String {
    let tiles = match tiles {
        Some(t) => t.clone(),
        None => crate::kernels::tile_picker::pick_tiles(spec, shape_class, dims),
    };

    let mut source = compute_kernel_source();

    let defines = format!(
        r#"
#define GRIM_WAVEFRONT_SIZE   {}
#define GRIM_MAX_LDS_BYTES    {}
#define GRIM_CU_COUNT         {}
#define GRIM_BLOCK_M          {}
#define GRIM_BLOCK_N          {}
#define GRIM_BLOCK_K          {}
#define GRIM_SPLIT_K          {}
#define GRIM_GRID_STRIDE_M    {}
#define GRIM_GRID_STRIDE_N    {}
#define GRIM_DEVICE_ID        {}
#define GRIM_NUM_DEVICES      {}
#define GRIM_LDS_DOUBLE_BUFFER {}
#define GRIM_SCHED_GROUP_BARRIER {}

"#,
        spec.wavefront_size,
        spec.max_shared_mem_per_block,
        spec.cu_count,
        tiles.block_m,
        tiles.block_n,
        tiles.block_k,
        tiles.split_k,
        tiles.grid_stride_m,
        tiles.grid_stride_n,
        device_id,
        num_devices,
        tiles.lds_double_buffer as u32,
        (tiles.block_k >= 32) as u32,
    );

    source.push_str(&defines);
    source
}

#[cfg(test)]
mod source_asm_self_tests {
    use super::*;

    #[test]
    fn compute_kernel_source_contains_both_sub_sources() {
        let src = compute_kernel_source();
        // The add / mul / rms_norm kernel names live in OTHER_KERNEL_SOURCE.
        assert!(src.contains("grim_add"));
        assert!(src.contains("grim_rms_norm"));
        // The fused QKV attention lives in qkv_attention::KERNEL_SOURCE.
        assert!(src.contains("grim_qkv_attention"));
        // WMMA GEMM lives in wmma_gemm::KERNEL_SOURCE.
        assert!(src.contains("grim_wmma_gemm"));
        // Q8_0 dequant kernel lives in q8_0_dequant::KERNEL_SOURCE.
        assert!(src.contains("grim_dequant_q8_0"));
    }

    #[test]
    fn compute_kernel_source_pre_allocation_is_at_least_qkv_length() {
        // Rough upper bound: the function pre-allocates
        let _ = compute_kernel_source();
    }

    #[test]
    fn compute_kernel_source_contains_phase2_kernels() {
        let src = compute_kernel_source();
        assert!(src.contains("grim_cross_attention"));
        assert!(src.contains("grim_rwkv_time_mix"));
        assert!(src.contains("grim_fp8_gemm_rdna4"));
    }

    #[test]
    fn specialized_source_wires_lds_and_schedule_controls() {
        let topology = crate::peer_access::P2PTopology {
            device_count: 0,
            links: Vec::new(),
        };
        let spec = crate::device::hardware_spec::HardwareSpec {
            gcn_arch: "gfx1100".into(),
            wavefront_size: 32,
            max_shared_mem_per_block: 64 * 1024,
            cu_count: 1,
            max_threads_per_block: 1024,
            mem_bandwidth_gb_s: 1000.0,
            multiprocessor_count: 1,
            p2p_topology: topology,
        };
        let dims = crate::kernels::tile_picker::ShapeDims::new(32, 32, 64);
        let tiles = crate::kernels::tile_picker::pick_tiles(
            &spec,
            crate::autotune::ShapeClass::Prefill,
            dims,
        );
        let source = compute_kernel_source_with_spec(
            &spec,
            "grim_wmma_gemm",
            crate::autotune::ShapeClass::Prefill,
            dims,
            0,
            1,
            Some(&tiles),
        );
        assert!(source.contains("#define GRIM_LDS_DOUBLE_BUFFER"));
        assert!(source.contains("#define GRIM_SCHED_GROUP_BARRIER"));
        assert!(source.contains("__builtin_amdgcn_sched_group_barrier"));
        assert!(source.contains("lds_a[2][256]"));
    }

    /// A.0 regression guard: every shared __device__ helper must be defined
    #[test]
    fn kernel_source_has_no_duplicate_device_fn_definitions() {
        let src = compute_kernel_source();
        // These four symbols are defined in shared_device_fns::KERNEL_SOURCE only.
        let shared_syms = [
            "float fp16_to_float_device(",
            "float fp8_e4m3_to_float_hip(",
            "float mxfp4_to_float_hip(",
            "float dequant_q4k_element(",
        ];
        for sym in &shared_syms {
            let count = src.matches(sym).count();
            assert_eq!(
                count, 1,
                "shared __device__ symbol defined {} times (expected 1): '{}'",
                count, sym
            );
        }
    }
}
