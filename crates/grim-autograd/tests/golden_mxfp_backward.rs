use grim_autograd::ops::{MatMulArgs, matmul_backward};
use grim_backend_cpu::cpu_tensor;
use grim_quant::{dequant_mxfp4, dequant_mxfp8};
use grim_tensor::{
    Tensor,
    dtype::{ArithType, DType, FloatPackScheme, Storage},
};

fn frame_segment(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + data.len());
    out.extend_from_slice(&(data.len() as u64).to_le_bytes());
    out.extend_from_slice(data);
    out
}

#[test]
fn test_mxfp4_matmul_backward_grad_a_golden() {
    let (m, k, n) = (4, 64, 128); // K=64 (multiple of 32), N=128

    // Generate random / deterministic codes and exponents
    let num_elements = k * n;
    let num_groups = num_elements / 32;

    // Packed 4-bit codes: 2 per byte
    let codes: Vec<u8> = (0..num_elements / 2)
        .map(|i| ((i % 15) as u8) | (((15 - (i % 15)) as u8) << 4))
        .collect();
    // Exponents around 127 bias
    let exps: Vec<u8> = (0..num_groups)
        .map(|i| (125 + (i % 5)) as u8)
        .collect();

    let mut framed = Vec::new();
    framed.extend(frame_segment(&codes));
    framed.extend(frame_segment(&exps));

    let b_dequant = dequant_mxfp4(&framed, num_elements).expect("dequant_mxfp4");

    let out_grad_data: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.03).sin()).collect();
    let out_grad_tensor = cpu_tensor(out_grad_data.clone(), grim_tensor::Shape::new(vec![m, n]));

    // True dequantized reference gradient for grad_a:
    // C = A @ B^T where A is [M, K], B is [K, N] in CPU matmul convention
    // dA = dY @ B^T where dY is [M, N], B is [K, N], dA is [M, K]
    // dA[i, j] = sum_l dY[i, l] * B[j, l]
    let mut expected_grad_a = vec![0.0f32; m * k];
    for row in 0..m {
        for col_k in 0..k {
            let mut sum = 0.0f32;
            for col_n in 0..n {
                sum += out_grad_data[row * n + col_n] * b_dequant[col_k * n + col_n];
            }
            expected_grad_a[row * k + col_k] = sum;
        }
    }

    let mxfp4_dtype = DType {
        storage: Storage::FloatPack(FloatPackScheme::MxFp4),
        arith: ArithType::F32,
    };
    let b_storage = grim_backend_cpu::cpu_tensor(b_dequant, grim_tensor::Shape::new(vec![k, n]));
    let b_tensor = Tensor::new(
        b_storage.storage().clone(),
        grim_tensor::Shape::new(vec![k, n]),
        mxfp4_dtype,
        grim_tensor::QuantProvenance::default(),
        grim_tensor::Device::Cpu,
    );

    let a_dummy = cpu_tensor(vec![0.0f32; m * k], grim_tensor::Shape::new(vec![m, k]));
    let args = MatMulArgs {
        a: a_dummy,
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

    assert!(
        max_err < 1e-4,
        "MXFP4 matmul_backward grad_a max absolute error {max_err} exceeds 1e-4"
    );
}

#[test]
fn test_mxfp8_matmul_backward_grad_a_golden() {
    let (m, k, n) = (4, 64, 128); // K=64, N=128

    let num_elements = k * n;
    let num_groups = num_elements / 32;

    let codes: Vec<u8> = (0..num_elements).map(|i| (i % 250) as u8).collect();
    let exps: Vec<u8> = (0..num_groups).map(|i| (126 + (i % 3)) as u8).collect();

    let mut framed = Vec::new();
    framed.extend(frame_segment(&codes));
    framed.extend(frame_segment(&exps));

    let b_dequant = dequant_mxfp8(&framed, num_elements).expect("dequant_mxfp8");

    let out_grad_data: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.04).cos()).collect();
    let out_grad_tensor = cpu_tensor(out_grad_data.clone(), grim_tensor::Shape::new(vec![m, n]));

    let mut expected_grad_a = vec![0.0f32; m * k];
    for row in 0..m {
        for col_k in 0..k {
            let mut sum = 0.0f32;
            for col_n in 0..n {
                sum += out_grad_data[row * n + col_n] * b_dequant[col_k * n + col_n];
            }
            expected_grad_a[row * k + col_k] = sum;
        }
    }

    let mxfp8_dtype = DType {
        storage: Storage::FloatPack(FloatPackScheme::MxFp8),
        arith: ArithType::F32,
    };
    let b_storage = grim_backend_cpu::cpu_tensor(b_dequant, grim_tensor::Shape::new(vec![k, n]));
    let b_tensor = Tensor::new(
        b_storage.storage().clone(),
        grim_tensor::Shape::new(vec![k, n]),
        mxfp8_dtype,
        grim_tensor::QuantProvenance::default(),
        grim_tensor::Device::Cpu,
    );

    let a_dummy = cpu_tensor(vec![0.0f32; m * k], grim_tensor::Shape::new(vec![m, k]));
    let args = MatMulArgs {
        a: a_dummy,
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

    assert!(
        max_err < 1e-4,
        "MXFP8 matmul_backward grad_a max absolute error {max_err} exceeds 1e-4"
    );
}
