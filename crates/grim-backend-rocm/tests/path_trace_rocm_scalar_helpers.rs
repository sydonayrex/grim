//! Path trace and numerical parity verification for ROCm GPU scalar arithmetic helpers.

use grim_backend_rocm::RocmDevice;
use grim_tensor::dtype::DType;
use grim_tensor::{BackendDevice, Shape};

#[test]
fn test_rocm_path_trace_all_scalar_helpers() {
    let dev: Box<dyn BackendDevice> = Box::new(RocmDevice::new(0));
    let shape = Shape::new(vec![4]);

    let x_data = vec![100.0f32, 200.0, 300.0, 400.0];
    let x = dev.from_cpu(&x_data, &shape, DType::F32).expect("x");

    // Path 1: BackendDevice::mul_scalar on ROCm GPU
    let (out_mul, h_mul) = dev
        .mul_scalar(x.as_ref(), 0.5, &shape)
        .expect("mul_scalar path");
    h_mul.synchronize().expect("sync");
    assert_eq!(
        out_mul.to_cpu_vec_f32().unwrap(),
        vec![50.0, 100.0, 150.0, 200.0]
    );

    // Path 2: BackendDevice::add_scalar on ROCm GPU
    let (out_add, h_add) = dev
        .add_scalar(x.as_ref(), 10.0, &shape)
        .expect("add_scalar path");
    h_add.synchronize().expect("sync");
    assert_eq!(
        out_add.to_cpu_vec_f32().unwrap(),
        vec![110.0, 210.0, 310.0, 410.0]
    );

    // Path 3: BackendDevice::sub_scalar on ROCm GPU
    let (out_sub, h_sub) = dev
        .sub_scalar(x.as_ref(), 50.0, &shape)
        .expect("sub_scalar path");
    h_sub.synchronize().expect("sync");
    assert_eq!(
        out_sub.to_cpu_vec_f32().unwrap(),
        vec![50.0, 150.0, 250.0, 350.0]
    );

    // Path 4: BackendDevice::div_scalar on ROCm GPU
    let (out_div, h_div) = dev
        .div_scalar(x.as_ref(), 4.0, &shape)
        .expect("div_scalar path");
    h_div.synchronize().expect("sync");
    assert_eq!(
        out_div.to_cpu_vec_f32().unwrap(),
        vec![25.0, 50.0, 75.0, 100.0]
    );
}
