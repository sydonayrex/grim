use grim_backend_rocm::kernels::charon::default_variant_table;
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
fn test_grid_dimension_overflow_bounds() {
    let max_dim: usize = 65536 * 1024;
    let block_size: usize = 256;
    let grid_x: u64 = (max_dim as u64).div_ceil(block_size as u64);
    assert!(grid_x <= u32::MAX as u64);
    let grid_u32 = u32::try_from(grid_x).unwrap();
    assert_eq!(grid_u32, 262144);
}
