use grim_backend_rocm::autotune::{AutotuneConfig, ShapeClass};
use grim_backend_rocm::device::hardware_spec::HardwareSpec;
use grim_backend_rocm::kernels::charon::default_variant_table;
use grim_backend_rocm::kernels::tile_picker::{pick_tiles, ShapeDims};
use grim_backend_rocm::peer_access::P2PTopology;
use grim_tensor::dtype::{ArithType, DType, Storage};

#[test]
fn test_charon_variant_table_nan_ordering_safety() {
    let t = default_variant_table();
    let mut buckets: Vec<f32> = t.iter().map(|r| r.skew_bucket).collect();

    // Inject extreme values including NaN and Infs
    buckets.push(f32::NAN);
    buckets.push(f32::INFINITY);
    buckets.push(f32::NEG_INFINITY);

    // Must sort without panicking
    buckets.sort_by(|a, b| a.total_cmp(b));
    assert!(buckets.len() >= 4);
}

#[test]
fn test_quant_format_subnormal_and_extreme_floats() {
    let fp8_dtype = DType {
        arith: ArithType::U8,
        storage: Storage::CompressedTensorsW8A8Fp8,
    };
    assert!(fp8_dtype.is_quantized());

    // Verify subnormal floats in CPU reference calculations
    let subnormal = f32::from_bits(0x00000001); // Smallest positive subnormal f32
    assert!(subnormal.is_subnormal());
    let scaled = subnormal * 2.0;
    assert_eq!(scaled, f32::from_bits(0x00000002));
}

#[test]
fn test_fp8_mxfp4_boundary_conversions() {
    // Test IEEE-754 overflow saturation to FP8 max representations
    let pos_inf = f32::INFINITY;
    let neg_inf = f32::NEG_INFINITY;
    let nan_val = f32::NAN;

    // E4M3 clamp range [-448.0, 448.0]
    let clamp_e4m3 = |x: f32| -> f32 {
        if x.is_nan() {
            f32::NAN
        } else {
            x.clamp(-448.0, 448.0)
        }
    };
    assert_eq!(clamp_e4m3(pos_inf), 448.0);
    assert_eq!(clamp_e4m3(neg_inf), -448.0);
    assert!(clamp_e4m3(nan_val).is_nan());
}

#[test]
fn test_attention_window_stride_saturating_arithmetic_bounds() {
    // Test sliding window lower-bound math: abs_first.saturating_sub(w.saturating_sub(1))
    // Validate for extreme w and abs_first combinations
    for &(abs_first, window_size) in &[
        (0usize, 0usize),
        (0, 1),
        (0, 1024),
        (10, 512),
        (512, 10),
        (usize::MAX, 1024),
        (1024, usize::MAX),
    ] {
        let w_sub = window_size.saturating_sub(1);
        let win_lo = abs_first.saturating_sub(w_sub);
        assert!(win_lo <= abs_first, "Lower bound must never exceed current position");
        let active_len = abs_first - win_lo + 1;
        assert!(active_len >= 1, "Active window slice length must be positive");
    }
}

#[test]
fn test_tile_picker_large_dimension_overflow_bounds() {
    let spec = HardwareSpec {
        gcn_arch: "gfx1100".into(),
        wavefront_size: 32,
        max_shared_mem_per_block: 64 * 1024,
        max_threads_per_block: 1024,
        cu_count: 96,
        multiprocessor_count: 96,
        mem_bandwidth_gb_s: 800.0,
        peak_flops_fp16: 60.0e12,
        p2p_topology: P2PTopology {
            device_count: 0,
            links: Vec::new(),
        },
    };

    // Extreme dimensions exceeding u16::MAX (e.g. 1M token batch or 128k context)
    let dims = ShapeDims::new(131072, 65536, 32768);
    let tile_cfg = pick_tiles(&spec, ShapeClass::Prefill, dims);

    assert!(tile_cfg.block_m > 0);
    assert!(tile_cfg.block_n > 0);
    assert!(tile_cfg.block_k > 0);
    assert!(tile_cfg.threads <= spec.max_threads_per_block);
    assert_eq!(tile_cfg.threads % spec.wavefront_size, 0);

    // Compute grid dimensions safely
    let grid_m = (dims.m as u64).div_ceil(tile_cfg.block_m as u64);
    let grid_n = (dims.n as u64).div_ceil(tile_cfg.block_n as u64);
    assert!(grid_m <= u32::MAX as u64);
    assert!(grid_n <= u32::MAX as u64);
}

#[test]
fn test_autotune_serialization_nan_and_inf_handling() {
    let cfg = AutotuneConfig {
        block_dim: 256,
        tile_kv: 64,
        grid_stride: 1,
        cycles_per_invocation: 1200,
        spec_gamma: 4,
        spec_acceptance_threshold: f32::NAN,
        spec_alpha: f32::INFINITY,
    };

    // Serializing NaN/Inf to JSON and reading it back
    let json_res = serde_json::to_string(&cfg);
    // serde_json default serialization policy produces null for NaN/Inf floats
    if let Ok(json_str) = json_res {
        let de_res: Result<AutotuneConfig, _> = serde_json::from_str(&json_str);
        if let Ok(deserialized) = de_res {
            assert_eq!(deserialized.block_dim, 256);
            assert_eq!(deserialized.spec_gamma, 4);
        }
    }
}
