use grim_autograd::ops::{matmul_backward, MatMulArgs};
use grim_backend_cpu::cpu_tensor;
use grim_quant::{dequant_q4k, quant_q4k};
use grim_tensor::{
    dtype::{ArithType, DType, KQuantScheme, Storage},
    Tensor,
};

#[test]
fn test_q4k_matmul_backward_grad_a_golden_mutation_resistant() {
    // M=2, K=256, N=256
    let (m, k, n) = (2, 256, 256);
    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05).sin()).collect();
    let b_orig: Vec<f32> = (0..k * n).map(|i| 1.0 + (i as f32 * 0.015).cos().abs() * 8.0).collect();
    let out_grad_data: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.02).cos()).collect();

    let a_tensor = cpu_tensor(a_data, grim_tensor::Shape::new(vec![m, k]));
    let out_grad_tensor = cpu_tensor(out_grad_data.clone(), grim_tensor::Shape::new(vec![m, n]));

    let b_packed = quant_q4k(&b_orig).expect("quant_q4k");
    let b_dequant = dequant_q4k(&b_packed, b_orig.len()).expect("dequant_q4k");

    // True dequantized reference gradient for grad_a: dA = out_grad @ B_dequant^T
    let mut expected_grad_a = vec![0.0f32; m * k];
    for i in 0..m {
        for j in 0..k {
            let mut sum = 0.0f32;
            for l in 0..n {
                sum += out_grad_data[i * n + l] * b_dequant[j * n + l];
            }
            expected_grad_a[i * k + j] = sum;
        }
    }

    let q4k_dtype = DType {
        storage: Storage::KQuant(KQuantScheme::Q4K),
        arith: ArithType::F32,
    };
    let b_storage = grim_backend_cpu::cpu_tensor(b_dequant, grim_tensor::Shape::new(vec![k, n]));
    let b_tensor = Tensor::new(
        b_storage.storage().clone(),
        grim_tensor::Shape::new(vec![k, n]),
        q4k_dtype,
        grim_tensor::QuantProvenance::default(),
        grim_tensor::Device::Cpu,
    );

    let args = MatMulArgs {
        a: a_tensor,
        b: b_tensor,
        out_grad: out_grad_tensor,
        transpose_a: false,
        transpose_b: false,
    };

    let (grad_a, _grad_b) = matmul_backward(&args).expect("matmul_backward");
    let actual_grad_a = grad_a.to_vec_f32().expect("to_vec_f32");

    assert_eq!(actual_grad_a.len(), expected_grad_a.len());
    let mut max_err: f32 = 0.0;
    for (act, exp) in actual_grad_a.iter().zip(expected_grad_a.iter()) {
        let err = (act - exp).abs();
        if err > max_err {
            max_err = err;
        }
    }

    // Assert high accuracy CPU reference match
    assert!(
        max_err < 1e-4,
        "Q4_K matmul_backward grad_a max absolute error {max_err} exceeds 1e-4"
    );
}
