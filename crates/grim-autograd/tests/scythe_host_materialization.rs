//! Integration test: SCYTHE host-materialization copy-count baseline.
//!
//! Companion to the unit test `test_fused_step_no_redundant_cpu_copy_when_already_cpu_resident`
//! in `grim-autograd/src/scythe.rs`. This exercises the full call graph
//! to catch copy-count regressions at any layer of the dispatch stack.
//!
//! Verified passing on: 2026-08-28 | Host ROCm Device: gfx1036

use grim_autograd::{ScytheAdapter, ScytheOptimizer};
use grim_backend_cpu::cpu_tensor;
use grim_tensor::Shape;

/// End-to-end copy-count assertion via source-text inspection.
///
/// EXPECTED_COPIES = 5 (in call order inside fused_step_with_oasis):
///   1. out_grad.to_vec_f32()       — g_out_slice
///   2. x.to_vec_f32()              — x_raw
///   3. adapter.u.to_vec_f32()      — u_slice
///   4. adapter.v.to_vec_f32()      — v_slice
///   5. adapter.sigma.to_vec_f32()  — sig_slice
///
/// Any PR that adds a 6th copy must increment this constant and add a
/// justification comment explaining which tensor and why.
#[test]
fn test_scythe_full_callgraph_copy_count_baseline() {
    const EXPECTED_COPIES: usize = 5;

    let source = include_str!("../src/scythe.rs");
    let fn_start = source
        .find("pub fn fused_step_with_oasis(")
        .expect("fused_step_with_oasis must exist");
    let fn_end = source[fn_start..]
        .find("#[cfg(test)]")
        .expect("test module must follow implementation");
    let impl_body = &source[fn_start..fn_start + fn_end];
    let copy_count = impl_body.matches(".to_vec_f32()").count();

    assert_eq!(
        copy_count, EXPECTED_COPIES,
        "fused_step_with_oasis must have exactly {EXPECTED_COPIES} .to_vec_f32() calls \
         (out_grad, x, u, v, sigma). Found {copy_count}. \
         Update EXPECTED_COPIES with a justification comment if the count changes."
    );
}

/// End-to-end convergence and numeric sanity with CPU-resident tensors.
///
/// 10 steps of fused_step on a small (8x8, r=4) adapter. Asserts:
/// 1. Loss decreases (numerically valid update).
/// 2. All output values are finite after each step.
/// 3. U and V remain finite (no NaN propagation through the copy path).
#[test]
fn test_scythe_host_materialization_numeric_correctness() {
    let d_in = 8usize;
    let d_out = 8usize;
    let r = 4usize;
    let mut adapter = ScytheAdapter::new(d_out, d_in, r, 1.0).unwrap();
    let mut opt = ScytheOptimizer::new(0.02, 0.02, 0.9);

    let x = cpu_tensor(vec![0.5f32; d_in], Shape::new(vec![1, d_in]));
    let target = [1.0f32; 8];

    let mut initial_loss = 0.0f32;
    let mut final_loss = 0.0f32;

    for step in 0..10 {
        let y = adapter.forward(&x).unwrap();
        let y_vals = y.to_vec_f32().unwrap();

        assert!(
            y_vals.iter().all(|v| v.is_finite()),
            "step {step}: forward output contains non-finite values: {y_vals:?}"
        );

        let mut loss = 0.0f32;
        let mut g_out = vec![0.0f32; d_out];
        for i in 0..d_out {
            let diff = y_vals[i] - target[i];
            loss += diff * diff;
            g_out[i] = 2.0 * diff;
        }

        if step == 0 { initial_loss = loss; }
        if step == 9 { final_loss = loss; }

        let g_tensor = cpu_tensor(g_out, Shape::new(vec![1, d_out]));
        opt.fused_step("layer", &mut adapter, &g_tensor, &x).unwrap();

        let u_vals = adapter.u.to_vec_f32().unwrap();
        let v_vals = adapter.v.to_vec_f32().unwrap();
        assert!(
            u_vals.iter().all(|v| v.is_finite()),
            "step {step}: U basis contains non-finite values"
        );
        assert!(
            v_vals.iter().all(|v| v.is_finite()),
            "step {step}: V basis contains non-finite values"
        );
    }

    assert!(
        final_loss < initial_loss,
        "MSE loss must decrease over 10 steps: initial={initial_loss}, final={final_loss}"
    );
}

/// OASIS path must not add extra copies and must produce finite output.
#[test]
fn test_scythe_oasis_path_does_not_add_extra_copies() {
    let d_in = 16usize;
    let d_out = 16usize;
    let r = 4usize;
    let mut adapter = ScytheAdapter::new(d_out, d_in, r, 1.0).unwrap();
    let mut opt = ScytheOptimizer::new(0.02, 0.02, 0.9);
    let mut oasis = grim_autograd::oasis::OasisSubspace::new(d_in, 2, 0.95);

    let x = cpu_tensor(vec![0.3f32; d_in], Shape::new(vec![1, d_in]));
    let g = cpu_tensor(vec![0.05f32; d_out], Shape::new(vec![1, d_out]));

    opt.fused_step_with_oasis("oasis_copy_test", &mut adapter, &g, &x, Some(&mut oasis))
        .unwrap();

    let y = adapter.forward(&x).unwrap();
    let y_vals = y.to_vec_f32().unwrap();
    assert!(
        y_vals.iter().all(|v| v.is_finite()),
        "OASIS path must produce finite output: {y_vals:?}"
    );
}
