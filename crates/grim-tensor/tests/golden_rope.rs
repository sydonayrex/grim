use grim_backend_cpu::cpu_tensor;
use grim_tensor::{AttentionOps, Shape};

#[test]
fn test_rope_golden_mutation_resistant() {
    let dev = grim_backend_cpu::CpuDevice::new();

    // B=1, S=2, D=4, base=10000.0
    // d=4, half=2.
    // inv_freq[0] = 1.0 / 10000.0^(0/4) = 1.0
    // inv_freq[1] = 1.0 / 10000.0^(2/4) = 1.0 / 100.0 = 0.01

    // Position 0 (pos=0.0):
    // cos = [1.0, 1.0], sin = [0.0, 0.0] -> input unchanged.

    // Position 1 (pos=1.0):
    // i=0: a = 1.0 * 1.0 = 1.0 rad. cos(1.0) ≈ 0.5403023, sin(1.0) ≈ 0.84147098
    // i=1: a = 1.0 * 0.01 = 0.01 rad. cos(0.01) ≈ 0.99995, sin(0.01) ≈ 0.00999983

    let x_data = vec![
        // Batch 0, Pos 0
        1.0, 2.0, 3.0, 4.0, // Batch 0, Pos 1
        1.0, 2.0, 3.0, 4.0,
    ];

    let x = cpu_tensor(x_data, Shape::new(vec![1, 2, 4]));
    let positions = vec![0u32, 1u32];

    let rope_cfg = grim_tensor::RopeConfig::new(4, 10000.0);
    let (out_st, _) = dev
        .rope(x.storage().as_ref(), &positions, &rope_cfg, x.shape())
        .expect("rope");
    let out_data = out_st.to_cpu_vec_f32().expect("to_cpu_vec_f32");

    // Pos 0 output must match input [1.0, 2.0, 3.0, 4.0]
    assert!((out_data[0] - 1.0).abs() < 1e-4);
    assert!((out_data[1] - 2.0).abs() < 1e-4);
    assert!((out_data[2] - 3.0).abs() < 1e-4);
    assert!((out_data[3] - 4.0).abs() < 1e-4);

    // Pos 1 output calculations:
    // x1 = 1.0 (at offset 4), x2 = 2.0 (at offset 5, half offset = 1 for adjacent pairs)
    // out[4] = x1 * cos(1.0) - x2 * sin(1.0) = 1.0 * cos(1.0) - 2.0 * sin(1.0) = -1.1426396
    // out[5] = x1 * sin(1.0) + x2 * cos(1.0) = 1.0 * sin(1.0) + 2.0 * cos(1.0) = 1.9220756
    let expected_pos1_i0_low = 1.0 * (1.0f32).cos() - 2.0 * (1.0f32).sin();
    let expected_pos1_i0_high = 1.0 * (1.0f32).sin() + 2.0 * (1.0f32).cos();

    assert!(
        (out_data[4] - expected_pos1_i0_low).abs() < 1e-4,
        "Pos 1 low dim match: got {}, expected {}",
        out_data[4],
        expected_pos1_i0_low
    );
    assert!(
        (out_data[5] - expected_pos1_i0_high).abs() < 1e-4,
        "Pos 1 high dim match: got {}, expected {}",
        out_data[5],
        expected_pos1_i0_high
    );
}
