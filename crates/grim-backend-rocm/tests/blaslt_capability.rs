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

#[test]
#[ignore]
fn gpu_blaslt_dispatch_matmul_matches_cpu_and_reuses_scratch() {
    if !gpu_test_enabled() || !RocmDevice::probe_one(0).unwrap_or(false) {
        return;
    }
    if !probe_blaslt().available() {
        return;
    }

    let previous = std::env::var("GRIM_BLASLT_PREFILL").ok();
    unsafe { std::env::set_var("GRIM_BLASLT_PREFILL", "1") };

    let dev = RocmDevice::shared(0);
    let (m, n, k) = (32usize, 256usize, 256usize);
    let make_input = |seed: usize, rows: usize, cols: usize| -> Vec<f32> {
        (0..rows * cols)
            .map(|i| (((i * 17 + seed * 13) % 37) as f32 - 18.0) / 100.0)
            .collect()
    };
    let a1_data = make_input(1, m, k);
    let b1_data = make_input(2, n, k);
    let a2_data = make_input(3, m, k);
    let b2_data = make_input(4, n, k);
    let a1 = CoreTensorOps::from_cpu(&dev, &a1_data, &Shape::new(vec![m, k]), DType::F32)
        .expect("upload A1");
    let b1 = CoreTensorOps::from_cpu(&dev, &b1_data, &Shape::new(vec![n, k]), DType::F32)
        .expect("upload B1");
    let a2 = CoreTensorOps::from_cpu(&dev, &a2_data, &Shape::new(vec![m, k]), DType::F32)
        .expect("upload A2");
    let b2 = CoreTensorOps::from_cpu(&dev, &b2_data, &Shape::new(vec![n, k]), DType::F32)
        .expect("upload B2");
    let out1 = CoreTensorOps::from_cpu(
        &dev,
        &vec![0.0f32; m * n],
        &Shape::new(vec![m, n]),
        DType::F32,
    )
    .expect("allocate output 1");
    let out2 = CoreTensorOps::from_cpu(
        &dev,
        &vec![0.0f32; m * n],
        &Shape::new(vec![m, n]),
        DType::F32,
    )
    .expect("allocate output 2");

    // Do not synchronize the first handle. The second call must wait on the
    // persistent completion event before reusing the same A/D scratch.
    dev.matmul_op_into(
        a1.as_ref(),
        b1.as_ref(),
        as_rocm(out1.as_ref()).unwrap(),
        grim_backend_rocm::autotune::GemmOp::Other,
    )
    .expect("BLASLt dispatch 1");
    let second = dev
        .matmul_op_into(
            a2.as_ref(),
            b2.as_ref(),
            as_rocm(out2.as_ref()).unwrap(),
            grim_backend_rocm::autotune::GemmOp::Other,
        )
        .expect("BLASLt dispatch 2");
    second.synchronize().expect("synchronize second dispatch");

    let got1 = out1.to_cpu_vec_f32().expect("read output 1");
    let got2 = out2.to_cpu_vec_f32().expect("read output 2");
    let reference = |a: &[f32], b: &[f32]| -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                c[row * n + col] = (0..k).map(|idx| a[row * k + idx] * b[col * k + idx]).sum();
            }
        }
        c
    };
    let want1 = reference(&a1_data, &b1_data);
    let want2 = reference(&a2_data, &b2_data);
    let max_error = |got: &[f32], want: &[f32]| {
        got.iter()
            .zip(want)
            .map(|(got, want)| (got - want).abs())
            .fold(0.0f32, f32::max)
    };
    let error1 = max_error(&got1, &want1);
    let error2 = max_error(&got2, &want2);

    match previous {
        Some(value) => unsafe { std::env::set_var("GRIM_BLASLT_PREFILL", value) },
        None => unsafe { std::env::remove_var("GRIM_BLASLT_PREFILL") },
    }
    assert!(error1 < 1e-3, "first dispatch max abs error {error1}");
    assert!(error2 < 1e-3, "reused dispatch max abs error {error2}");
}
