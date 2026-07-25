use grim_backend_cpu::cpu_tensor;
use grim_tensor::{backend::BackendDevice, Shape};

#[test]
fn test_lora_accumulate_golden_mutation_resistant() {
    let dev = grim_backend_cpu::CpuDevice::new();

    // Base: 1 x 4 [0.5, 0.5, 0.5, 0.5]
    // x: 1 x 4 [1.0, 2.0, 3.0, 4.0]
    // A: rank 4 x in_features 4 (identity)
    // B: out_features 4 x rank 4 (2 * identity)
    // Scale: 0.5

    let base = cpu_tensor(vec![0.5, 0.5, 0.5, 0.5], Shape::new(vec![1, 4]));
    let x = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![1, 4]));
    let a = cpu_tensor(
        vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
        Shape::new(vec![4, 4]),
    );
    let b = cpu_tensor(
        vec![
            2.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 0.0, 0.0,
            0.0, 0.0, 2.0, 0.0,
            0.0, 0.0, 0.0, 2.0,
        ],
        Shape::new(vec![4, 4]),
    );

    // delta = (x @ A^T) @ B^T = [1.0, 2.0, 3.0, 4.0] @ [2, 4, 6, 8] = [2.0, 4.0, 6.0, 8.0]
    // scaled_delta = 0.5 * delta = [1.0, 2.0, 3.0, 4.0]
    // out = base + scaled_delta = [0.5, 0.5, 0.5, 0.5] + [1.0, 2.0, 3.0, 4.0] = [1.5, 2.5, 3.5, 4.5]

    let (out_st, _) = dev
        .lora_accumulate(
            base.storage().as_ref(),
            x.storage().as_ref(),
            a.storage().as_ref(),
            b.storage().as_ref(),
            0.5,
            base.shape(),
        )
        .expect("lora_accumulate");

    let out_data = out_st.to_cpu_vec_f32().expect("to_cpu_vec_f32");
    let expected = vec![1.5f32, 2.5, 3.5, 4.5];

    assert_eq!(out_data.len(), expected.len());
    for (i, (&act, &exp)) in out_data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (act - exp).abs() < 1e-4,
            "Mismatch at index {i}: got {act}, expected {exp}"
        );
    }
}
