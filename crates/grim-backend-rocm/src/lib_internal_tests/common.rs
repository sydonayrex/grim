//! Shared test-support helpers for the split lib_internal_tests modules.

#![allow(dead_code)] // helpers kept local to the test tree

use crate::RocmDevice;
use grim_tensor::CoreTensorOps;
use grim_tensor::{DType, Shape};

pub(super) fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

pub(super) fn run_matmul_on_dev(
    dev: &RocmDevice,
    a: &[f32],
    a_dims: &[usize],
    b: &[f32],
    b_dims: &[usize],
    out_dims: &[usize],
) -> Vec<f32> {
    let a_s = dev
        .from_cpu(a, &Shape::from_slice(a_dims), DType::F32)
        .unwrap();
    let b_s = dev
        .from_cpu(b, &Shape::from_slice(b_dims), DType::F32)
        .unwrap();
    let (out, _h) = dev
        .matmul(a_s.as_ref(), b_s.as_ref(), &Shape::from_slice(out_dims))
        .unwrap();
    out.to_cpu_vec_f32().unwrap()
}
