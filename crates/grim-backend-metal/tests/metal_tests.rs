//! Parity tests for the Metal backend dispatch methods.
//!
//! On Apple these exercise the GPU fast-path via MSL kernels.
//! On non-Apple platforms the CPU fallback is exercised, which keeps
//! the suite green in headless CI environments without an Apple GPU.

use grim_backend_metal::MetalDevice;
use grim_tensor::dtype::DType;
use grim_tensor::{BackendDevice, BackendStorage, Shape};
use grim_tensor::{ScytheLink, ScythePlacement};

#[test]
fn test_metal_all_reduce_parity() {
    let dev = MetalDevice::new(0).unwrap();

    let shape = Shape::new(vec![8]);
    let inputs_data = vec![
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![0.5f32, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
    ];
    let storages: Vec<Box<dyn BackendStorage>> = inputs_data
        .iter()
        .map(|v| dev.from_cpu(v, &shape, DType::F32).unwrap())
        .collect();
    let refs: Vec<&dyn BackendStorage> = storages.iter().map(|s| s.as_ref()).collect();

    let (out, handle) = dev.all_reduce(&refs, "sum").unwrap();
    handle.synchronize().unwrap();
    let result = out.to_cpu_vec_f32().unwrap();

    let expected: Vec<f32> = (0..8)
        .map(|i| inputs_data.iter().map(|v| v[i]).sum::<f32>())
        .collect();
    assert_eq!(result.len(), expected.len());
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!(
            (r - e).abs() < 1e-5,
            "all_reduce mismatch: {} != {}",
            r,
            e
        );
    }
}

#[test]
fn test_metal_all_reduce_single_input_parity() {
    let dev = MetalDevice::new(0).unwrap();

    let shape = Shape::new(vec![4]);
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let storage = dev.from_cpu(&data, &shape, DType::F32).unwrap();
    let refs: Vec<&dyn BackendStorage> = vec![storage.as_ref()];

    let (out, _) = dev.all_reduce(&refs, "sum").unwrap();
    let result = out.to_cpu_vec_f32().unwrap();

    for (r, e) in result.iter().zip(data.iter()) {
        assert!(
            (r - e).abs() < 1e-5,
            "all_reduce single mismatch: {} != {}",
            r,
            e
        );
    }
}

#[test]
fn test_metal_comm_fuse_reduce_parity() {
    let dev = MetalDevice::new(0).unwrap();

    let m = 2usize;
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0]; // [2, 2]
    let b_data = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0]; // [2, 3]
    let shape_a = Shape::new(vec![m, 2]);
    let shape_b = Shape::new(vec![m, 3]);

    let a = dev.from_cpu(&a_data, &shape_a, DType::F32).unwrap();
    let b = dev.from_cpu(&b_data, &shape_b, DType::F32).unwrap();

    let placement = ScythePlacement {
        ranks: vec![0, 1],
        partition: vec![0.5, 0.5],
        routes: vec![ScytheLink::Host; 4],
    };
    let partials: Vec<(&dyn BackendStorage, &ScythePlacement)> =
        vec![(a.as_ref(), &placement), (b.as_ref(), &placement)];

    let out = dev.comm_fuse_reduce(&partials).unwrap();
    let result = out.to_cpu_vec_f32().unwrap();

    // Column-concat: [[1, 2, 10, 20, 30], [3, 4, 40, 50, 60]]
    let expected = vec![1.0f32, 2.0, 10.0, 20.0, 30.0, 3.0, 4.0, 40.0, 50.0, 60.0];
    assert_eq!(result.len(), expected.len());
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!(
            (r - e).abs() < 1e-5,
            "comm_fuse mismatch: {} != {}",
            r,
            e
        );
    }
}
