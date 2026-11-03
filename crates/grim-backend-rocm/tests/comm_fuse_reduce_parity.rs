//! Task 10.8 gate: the production `BackendDevice::comm_fuse_reduce` path must
//! produce column-concat assembly identical to the CPU `comm_fuse_fan_in`
//! oracle now that the F32 arm assembles with the fused
//! `grim_comm_fuse_p2p_epilogue` kernel (one launch per partial) instead of
//! per-row synchronous D2D memcpys.

use grim_backend_rocm::device::roc_device::RocmDevice;
use grim_tensor::backend::{BackendStorage, CollectiveOps};
use grim_tensor::{CoreTensorOps, DType, Shape};
use std::panic;

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::new should succeed")).ok()
}

#[test]
#[ignore]
fn comm_fuse_reduce_f32_matches_fan_in_oracle() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };

    let m = 2usize;
    // Two column shards, deliberately overlapping values so an accumulate
    // bug (double-add) or a missed zero of the assembly buffer shows up.
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0]; // [2, 2]
    let b_data = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0]; // [2, 3]
    let a = CoreTensorOps::from_cpu(&dev, &a_data, &Shape::new(vec![m, 2]), DType::F32)?;
    let b = CoreTensorOps::from_cpu(&dev, &b_data, &Shape::new(vec![m, 3]), DType::F32)?;

    let placement = grim_tensor::backend::ScythePlacement {
        ranks: vec![0, 1],
        partition: vec![0.5, 0.5],
        routes: vec![grim_tensor::backend::ScytheLink::Host; 4],
    };
    let partials: Vec<(&dyn BackendStorage, &grim_tensor::backend::ScythePlacement)> =
        vec![(a.as_ref(), &placement), (b.as_ref(), &placement)];

    let out = dev.comm_fuse_reduce(&partials)?;
    let result = out.to_cpu_vec_f32()?;

    // CPU oracle: comm_fuse_fan_in over the same shards.
    let expected = grim_backend_rocm::kernels::comm_fuse::comm_fuse_fan_in(
        &[(a_data.as_slice(), 2usize), (b_data.as_slice(), 3usize)],
        m,
        5,
        &placement,
    )?;
    let expected_flat: Vec<f32> = expected.data;

    assert_eq!(result.len(), expected_flat.len());
    for (i, (r, e)) in result.iter().zip(expected_flat.iter()).enumerate() {
        assert!(
            (r - e).abs() < 1e-5,
            "comm_fuse_reduce mismatch at {i}: got={r}, oracle={e}"
        );
    }
    println!("comm_fuse_reduce F32 (p2p_epilogue path) == comm_fuse_fan_in oracle ✓");
    Ok(())
}
