use grim_backend_cpu::gemv::cpu_gemv;

#[test]
fn test_cpu_gemv_matches_reference() {
    let a = vec![1.0f32, 2.0, 3.0, 4.0]; // 2x2 matrix: row 0 = [1, 2], row 1 = [3, 4]
    let x = vec![1.0f32, 1.0]; // vector: [1, 1]
    let y = cpu_gemv(&a, &x, 2, 2).expect("cpu_gemv should succeed");

    // Row 0: 1*1 + 2*1 = 3.0
    // Row 1: 3*1 + 4*1 = 7.0
    assert_eq!(y, vec![3.0, 7.0]);
}

#[test]
fn test_cpu_gemv_dimension_mismatch() {
    let a = vec![1.0f32; 4];
    let x = vec![1.0f32; 3]; // Mismatch (expected 2)
    assert!(cpu_gemv(&a, &x, 2, 2).is_err());
}
