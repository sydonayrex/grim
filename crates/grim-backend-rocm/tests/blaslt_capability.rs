//! Runtime capability evidence for the optional hipBLASLt/rocBLASLt path.

use grim_backend_rocm::{
    as_rocm, gpu_test_enabled, hipStreamSynchronize, launch_col_major_to_row_major,
    matmul_col_major_f32, probe_blaslt, select_blaslt_candidate, BlasLtSelection, RocmDevice,
};
use grim_tensor::{CoreTensorOps, DType, Shape};

#[test]
#[ignore]
fn gpu_probe_reports_blaslt_runtime_without_routing_decode() {
    if !gpu_test_enabled() {
        eprintln!("skipping: set GRIM_GPU_TEST=1");
        return;
    }
    if !RocmDevice::probe_one(0).unwrap_or(false) {
        eprintln!("skipping: no ROCm ordinal 0");
        return;
    }

    let probe = probe_blaslt();
    eprintln!("[blaslt-audit] {}", probe.describe());
    if !probe.available() {
        eprintln!("skipping BLASLt selection assertion: runtime unavailable");
        return;
    }
    assert!(
        probe.version.unwrap_or_default() > 0,
        "available BLASLt must report a positive version"
    );
    assert_eq!(
        select_blaslt_candidate(&probe, 1, 1024, 1024, true),
        BlasLtSelection::NotMeasured,
        "decode must not route to BLASLt"
    );
    assert_eq!(
        select_blaslt_candidate(&probe, 32, 4096, 4096, false),
        BlasLtSelection::NotMeasured,
        "unmeasured prefill must not route to BLASLt"
    );
}

#[test]
#[ignore]
fn gpu_blaslt_identity_layout_probe() {
    if !gpu_test_enabled() {
        return;
    }
    if !RocmDevice::probe_one(0).unwrap_or(false) || !probe_blaslt().available() {
        return;
    }
    let dev = RocmDevice::shared(0);
    let m = 2usize;
    let n = 4usize;
    let k = 3usize;
    let a_data = vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0];
    let b_data = vec![
        1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    ];
    let a = CoreTensorOps::from_cpu(&dev, &a_data, &Shape::new(vec![m, k]), DType::F32).unwrap();
    let b = CoreTensorOps::from_cpu(&dev, &b_data, &Shape::new(vec![k, n]), DType::F32).unwrap();
    let out = CoreTensorOps::from_cpu(&dev, &vec![0.0; m * n], &Shape::new(vec![m, n]), DType::F32)
        .unwrap();
    let stream = dev.get_stream_from_pool(0).unwrap();
    matmul_col_major_f32(
        stream,
        as_rocm(a.as_ref()).unwrap().device_ptr_checked().unwrap() as *const _,
        as_rocm(b.as_ref()).unwrap().device_ptr_checked().unwrap() as *const _,
        as_rocm(out.as_ref()).unwrap().device_ptr_checked().unwrap() as *mut _,
        m,
        n,
        k,
    )
    .unwrap();
    assert_eq!(
        unsafe { hipStreamSynchronize(stream) },
        grim_backend_rocm::hipSuccess
    );
    assert_eq!(
        out.to_cpu_vec_f32().unwrap(),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 0.0, 0.0]
    );
}

#[test]
#[ignore]
fn gpu_blaslt_prefill_microbench_real_arguments() {
    if !gpu_test_enabled() || !RocmDevice::probe_one(0).unwrap_or(false) {
        return;
    }
    if !probe_blaslt().available() {
        return;
    }
    let dev = RocmDevice::shared(0);
    let m = 32usize;
    let n = 4096usize;
    let k = 4096usize;
    let a_col: Vec<f32> = (0..m * k)
        .map(|i| ((i % 31) as f32 - 15.0) / 100.0)
        .collect();
    let b_col: Vec<f32> = (0..n * k)
        .map(|i| ((i % 29) as f32 - 14.0) / 100.0)
        .collect();
    let a = CoreTensorOps::from_cpu(&dev, &a_col, &Shape::new(vec![m, k]), DType::F32).unwrap();
    let b = CoreTensorOps::from_cpu(&dev, &b_col, &Shape::new(vec![k, n]), DType::F32).unwrap();
    let out_col = CoreTensorOps::from_cpu(
        &dev,
        &vec![0.0f32; m * n],
        &Shape::new(vec![m, n]),
        DType::F32,
    )
    .unwrap();
    let out = CoreTensorOps::from_cpu(
        &dev,
        &vec![0.0f32; m * n],
        &Shape::new(vec![m, n]),
        DType::F32,
    )
    .unwrap();
    let stream = dev.get_stream_from_pool(0).unwrap();
    let a_ptr = as_rocm(a.as_ref()).unwrap().device_ptr_checked().unwrap() as *const _;
    let b_ptr = as_rocm(b.as_ref()).unwrap().device_ptr_checked().unwrap() as *const _;
    let out_col_ptr = as_rocm(out_col.as_ref())
        .unwrap()
        .device_ptr_checked()
        .unwrap() as *mut _;
    for _ in 0..2 {
        matmul_col_major_f32(stream, a_ptr, b_ptr, out_col_ptr, m, n, k).unwrap();
        launch_col_major_to_row_major(
            &dev,
            as_rocm(out_col.as_ref()).unwrap(),
            as_rocm(out.as_ref()).unwrap(),
            m,
            n,
            m,
        )
        .unwrap();
    }
    dev.synchronize();
    let start = std::time::Instant::now();
    let iters = 5usize;
    for _ in 0..iters {
        matmul_col_major_f32(stream, a_ptr, b_ptr, out_col_ptr, m, n, k).unwrap();
        launch_col_major_to_row_major(
            &dev,
            as_rocm(out_col.as_ref()).unwrap(),
            as_rocm(out.as_ref()).unwrap(),
            m,
            n,
            m,
        )
        .unwrap();
    }
    dev.synchronize();
    eprintln!(
        "[blaslt-microbench] shape=32x4096x4096 per_call_us={:.3}",
        start.elapsed().as_secs_f64() * 1e6 / iters as f64
    );
    assert!(out.to_cpu_vec_f32().unwrap().iter().all(|v| v.is_finite()));
}

#[test]
#[ignore]
fn gpu_blaslt_prefill_matmul_matches_cpu() {
    if !gpu_test_enabled() {
        eprintln!("skipping: set GRIM_GPU_TEST=1");
        return;
    }
    if !RocmDevice::probe_one(0).unwrap_or(false) {
        eprintln!("skipping: no ROCm ordinal 0");
        return;
    }
    let probe = probe_blaslt();
    if !probe.available() {
        eprintln!("skipping: BLASLt runtime unavailable: {}", probe.describe());
        return;
    }

    let dev = RocmDevice::shared(0);
    let m = 32usize;
    let n = 256usize;
    let k = 256usize;
    let a_data: Vec<f32> = (0..m * k)
        .map(|i| ((i % 31) as f32 - 15.0) / 100.0)
        .collect();
    let b_data: Vec<f32> = (0..n * k)
        .map(|i| ((i % 29) as f32 - 14.0) / 100.0)
        .collect();
    let mut a_col = Vec::with_capacity(m * k);
    for col in 0..k {
        for row in 0..m {
            a_col.push(a_data[row * k + col]);
        }
    }
    let mut b_col = Vec::with_capacity(n * k);
    for col in 0..n {
        for row in 0..k {
            b_col.push(b_data[col * k + row]);
        }
    }
    let a = CoreTensorOps::from_cpu(&dev, &a_col, &Shape::new(vec![m, k]), DType::F32)
        .expect("upload A");
    let b = CoreTensorOps::from_cpu(&dev, &b_col, &Shape::new(vec![k, n]), DType::F32)
        .expect("upload B");
    let out_col = CoreTensorOps::from_cpu(
        &dev,
        &vec![0.0f32; m * n],
        &Shape::new(vec![m, n]),
        DType::F32,
    )
    .expect("upload output");
    let out = CoreTensorOps::from_cpu(
        &dev,
        &vec![0.0f32; m * n],
        &Shape::new(vec![m, n]),
        DType::F32,
    )
    .expect("upload row-major output");
    let stream = dev.get_stream_from_pool(0).expect("ROCm stream");
    matmul_col_major_f32(
        stream,
        as_rocm(a.as_ref()).unwrap().device_ptr_checked().unwrap() as *const _,
        as_rocm(b.as_ref()).unwrap().device_ptr_checked().unwrap() as *const _,
        as_rocm(out_col.as_ref())
            .unwrap()
            .device_ptr_checked()
            .unwrap() as *mut _,
        m,
        n,
        k,
    )
    .expect("BLASLt matmul");
    launch_col_major_to_row_major(
        &dev,
        as_rocm(out_col.as_ref()).unwrap(),
        as_rocm(out.as_ref()).unwrap(),
        m,
        n,
        m,
    )
    .expect("BLASLt output layout conversion");
    assert_eq!(
        unsafe { hipStreamSynchronize(stream) },
        grim_backend_rocm::hipSuccess
    );

    let got = out.to_cpu_vec_f32().expect("read output");
    let mut want = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            want[row * n + col] = (0..k)
                .map(|idx| a_data[row * k + idx] * b_data[col * k + idx])
                .sum();
        }
    }
    let max_abs = got
        .iter()
        .zip(&want)
        .map(|(got, want)| (got - want).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs < 1e-3, "BLASLt parity max abs error {max_abs}");
}
