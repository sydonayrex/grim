use grim_autograd::ops::selective_scan_backward;
use grim_backend_cpu::cpu_tensor;
use grim_tensor::Shape;

#[test]
fn test_selective_scan_backward_analytical_vs_numeric() {
    let d_inner = 4;
    let d_state = 3;
    let batch = 1;

    let x_vals = vec![1.2f32, -0.5, 0.8, 2.1];
    let a_vals = vec![
        0.1f32, 0.2, 0.3,
        -0.1, 0.05, 0.2,
        0.4, -0.3, 0.1,
        0.2, 0.1, -0.2,
    ];
    let b_vals = vec![
        0.5f32, -0.2, 0.1,
        0.3, 0.4, -0.1,
        -0.2, 0.1, 0.3,
        0.1, -0.3, 0.2,
    ];
    let d_vals = vec![0.7f32, 1.1, -0.4, 0.9];
    let dy_vals = vec![1.0f32, 0.5, -1.0, 2.0];

    let x = cpu_tensor(x_vals.clone(), Shape::new(vec![batch, d_inner]));
    let a_log = cpu_tensor(a_vals.clone(), Shape::new(vec![d_inner, d_state]));
    let b = cpu_tensor(b_vals.clone(), Shape::new(vec![d_inner, d_state]));
    let d = cpu_tensor(d_vals.clone(), Shape::new(vec![d_inner]));
    let dy = cpu_tensor(dy_vals.clone(), Shape::new(vec![batch, d_inner]));

    let (dx, dh_prev) = selective_scan_backward(&dy, &a_log, &b, &d, &x, 0, d_inner as i32, d_state as i32)
        .expect("backward pass");

    let dx_vec = dx.to_vec_f32().expect("dx to vec");
    let dh_vec = dh_prev.to_vec_f32().expect("dh to vec");

    assert_eq!(dx_vec.len(), d_inner);
    assert_eq!(dh_vec.len(), d_inner * d_state);

    // Analytical validation:
    // d_x[n] = D[n] * dy[n] + sum_s(b[n, s]) * dy[n]
    for n in 0..d_inner {
        let b_sum: f32 = (0..d_state).map(|s| b_vals[n * d_state + s]).sum();
        let expected_dx = (d_vals[n] + b_sum) * dy_vals[n];
        assert!(
            (dx_vec[n] - expected_dx).abs() < 1e-5,
            "n={n}: dx {} vs expected {}",
            dx_vec[n],
            expected_dx
        );

        for s in 0..d_state {
            let expected_dh = a_vals[n * d_state + s] * dy_vals[n];
            let actual_dh = dh_vec[n * d_state + s];
            assert!(
                (actual_dh - expected_dh).abs() < 1e-5,
                "n={n}, s={s}: dh {} vs expected {}",
                actual_dh,
                expected_dh
            );
        }
    }
}
