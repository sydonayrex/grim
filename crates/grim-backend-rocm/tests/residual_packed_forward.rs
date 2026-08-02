//! Forward ResidualPacked dispatch coverage.
//!
//! Run with:
//! `GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm --test residual_packed_forward -- --ignored --nocapture`

use grim_backend_rocm::{FUSED_FORWARD_DISPATCH_STATS, RocmDevice};
use grim_tensor::dtype::{ArithType, DType, QuantProvenance, Storage};
use grim_tensor::{BackendDevice, Shape};

const GPU_TEST_ENV: &str = "GRIM_RUN_GPU_TESTS";

fn pack_bpw2(codes: [u8; 4]) -> u8 {
    (codes[0] << 6) | (codes[1] << 4) | (codes[2] << 2) | codes[3]
}

#[test]
#[ignore = "requires real ROCm device; run with GRIM_RUN_GPU_TESTS=1 and -- --ignored"]
fn residual_packed_forward_passes_backup2_and_merges_it() {
    if std::env::var(GPU_TEST_ENV).is_err() {
        return;
    }
    let devices = match RocmDevice::probe() {
        Ok(devices) if !devices.is_empty() => devices,
        _ => return,
    };
    let dev = RocmDevice::try_new(devices[0].ordinal()).expect("ROCm device");
    dev.set_fused_dequant_gemm_enabled(true);

    FUSED_FORWARD_DISPATCH_STATS
        .attempts
        .store(0, std::sync::atomic::Ordering::Relaxed);
    FUSED_FORWARD_DISPATCH_STATS
        .kernel_calls
        .store(0, std::sync::atomic::Ordering::Relaxed);
    FUSED_FORWARD_DISPATCH_STATS
        .fallback_calls
        .store(0, std::sync::atomic::Ordering::Relaxed);

    let k = 4usize;
    let n = 1usize;
    let row_bytes = 256usize;
    let backup2_codes_offset = row_bytes;
    let backup2_scale_offset = backup2_codes_offset + row_bytes;
    let mut packed = vec![0u8; backup2_scale_offset + n];
    packed[0] = pack_bpw2([1, 1, 1, 1]); // primary = -1/3 per element
    packed[backup2_codes_offset] = pack_bpw2([3, 3, 3, 3]); // backup2 = +1
    packed[backup2_scale_offset] = 255;

    let b_dtype = DType {
        arith: ArithType::U8,
        storage: Storage::ResidualPacked(
            grim_tensor::dtype::ResidualPackedConfig { bpw: 2 },
        ),
    };
    let mut b = dev
        .from_cpu_bytes(&packed, &Shape::from_slice(&[packed.len()]), b_dtype)
        .expect("upload packed weights");
    b.set_provenance(QuantProvenance::WithResiduals {
        outlier_count: 0,
        outlier_indices_offset: 0,
        outlier_values_offset: 0,
        outlier_indices: Vec::new(),
        outlier_values_bits: Vec::new(),
        primary_scale_offset: 0,
        primary_scale_size: 1,
        primary_row_scale_dtype: 0,
        primary_scale_bytes: vec![255],
        backup1_bpw: 0,
        backup1_codes_offset: 0,
        backup1_scale_offset: 0,
        backup2_bpw: 2,
        backup2_codes_offset,
        backup2_scale_offset,
    });
    let a = dev
        .from_cpu(&[1.0f32; 4], &Shape::from_slice(&[1, k]), DType::F32)
        .expect("upload activation");

    let (out, handle) = dev
        .quantized_matmul(a.as_ref(), b.as_ref(), &[], &Shape::from_slice(&[1, n]))
        .expect("forward ResidualPacked matmul");
    handle.synchronize().expect("synchronize forward matmul");

    assert_eq!(
        FUSED_FORWARD_DISPATCH_STATS
            .kernel_calls
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
    assert_eq!(
        FUSED_FORWARD_DISPATCH_STATS
            .last_backup2_bpw
            .load(std::sync::atomic::Ordering::Relaxed),
        2
    );
    assert_eq!(
        FUSED_FORWARD_DISPATCH_STATS
            .last_backup2_codes_offset
            .load(std::sync::atomic::Ordering::Relaxed),
        backup2_codes_offset
    );
    assert_eq!(
        FUSED_FORWARD_DISPATCH_STATS
            .last_backup2_scale_offset
            .load(std::sync::atomic::Ordering::Relaxed),
        backup2_scale_offset
    );

    let got = out.to_cpu_vec_f32().expect("read forward output")[0];
    let expected = 4.0f32 * (-1.0 / 3.0 + 1.0);
    assert!((got - expected).abs() < 0.05, "got {got}, expected {expected}");
}

#[test]
#[ignore = "requires real ROCm device; run with GRIM_RUN_GPU_TESTS=1 and -- --ignored"]
fn residual_packed_forward_applies_outlier_correction_in_fused_path() {
    if std::env::var(GPU_TEST_ENV).is_err() { return; }
    let devices = match RocmDevice::probe() { Ok(devices) if !devices.is_empty() => devices, _ => return };
    let dev = RocmDevice::try_new(devices[0].ordinal()).expect("ROCm device");
    dev.set_fused_dequant_gemm_enabled(true);
    FUSED_FORWARD_DISPATCH_STATS.attempts.store(0, std::sync::atomic::Ordering::Relaxed);
    FUSED_FORWARD_DISPATCH_STATS.kernel_calls.store(0, std::sync::atomic::Ordering::Relaxed);
    FUSED_FORWARD_DISPATCH_STATS.fallback_calls.store(0, std::sync::atomic::Ordering::Relaxed);

    let packed = vec![pack_bpw2([1, 1, 1, 1])];
    let b_dtype = DType { arith: ArithType::U8, storage: Storage::ResidualPacked(
        grim_tensor::dtype::ResidualPackedConfig { bpw: 2 }) };
    let mut b = dev.from_cpu_bytes(&packed, &Shape::from_slice(&[packed.len()]), b_dtype)
        .expect("upload packed weights");
    b.set_provenance(QuantProvenance::WithResiduals {
        outlier_count: 1,
        outlier_indices_offset: 0,
        outlier_values_offset: 0,
        outlier_indices: vec![0],
        outlier_values_bits: vec![2.0f32.to_bits()],
        primary_scale_offset: 0,
        primary_scale_size: 0,
        primary_row_scale_dtype: 0,
        primary_scale_bytes: Vec::new(),
        backup1_bpw: 0,
        backup1_codes_offset: 0,
        backup1_scale_offset: 0,
        backup2_bpw: 0,
        backup2_codes_offset: 0,
        backup2_scale_offset: 0,
    });
    let a = dev.from_cpu(&[1.0f32; 4], &Shape::from_slice(&[1, 4]), DType::F32)
        .expect("upload activation");
    let (out, handle) = dev.quantized_matmul(a.as_ref(), b.as_ref(), &[], &Shape::from_slice(&[1, 1]))
        .expect("forward ResidualPacked matmul");
    handle.synchronize().expect("synchronize forward matmul");
    assert_eq!(FUSED_FORWARD_DISPATCH_STATS.kernel_calls.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(FUSED_FORWARD_DISPATCH_STATS.fallback_calls.load(std::sync::atomic::Ordering::Relaxed), 0);
    let got = out.to_cpu_vec_f32().expect("read forward output")[0];
    let expected = 2.0f32 + 3.0f32 * (-1.0 / 3.0);
    assert!((got - expected).abs() < 0.05, "got {got}, expected {expected}");
}
