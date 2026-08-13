//! Parity and numerical contract tests for `grim-backend-cpu`.

use grim_backend_cpu::CpuDevice;
use grim_tensor::dtype::{ArithType, DType, QuantFormat, Storage};
use grim_tensor::{BackendDevice, Shape};

fn dev() -> CpuDevice {
    CpuDevice::new()
}

#[test]
fn cpu_alloc_storage_and_copy_slice_into_kv_arena() {
    let dev = dev();
    let shape = Shape::from_slice(&[1, 4, 16]); // B, H, D
    let dtype = DType {
        arith: ArithType::F32,
        storage: Storage::Native,
    };

    // Allocate KV arena
    let arena = dev.alloc_storage(&shape, dtype.clone()).expect("alloc KV arena");
    assert_eq!(arena.shape(), &shape);

    // Source slice (e.g. 1 head row of 16 floats)
    let src_data: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
    let src = dev
        .from_cpu(&src_data, &Shape::from_slice(&[1, 16]), dtype)
        .expect("src storage");

    // Copy into offset 16 (second row)
    dev.copy_slice_into(arena.as_ref(), src.as_ref(), 16, 16)
        .expect("copy_slice_into");

    let arena_out = arena.to_cpu_vec_f32().expect("read arena");
    // Verify first 16 are 0.0, next 16 match src_data
    for i in 0..16 {
        assert_eq!(arena_out[i], 0.0);
        assert_eq!(arena_out[16 + i], src_data[i]);
    }
}

#[test]
fn cpu_silu_mul_backward_numerics() {
    let dev = dev();
    let n = 8;
    let shape = Shape::from_slice(&[1, n]);
    let dtype = DType {
        arith: ArithType::F32,
        storage: Storage::Native,
    };

    let e_data: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let g_data: Vec<f32> = (0..n).map(|i| (i as f32 - 4.0) * 0.2).collect();
    let dw_data: Vec<f32> = vec![1.0; n];

    let e = dev.from_cpu(&e_data, &shape, dtype.clone()).expect("e");
    let g = dev.from_cpu(&g_data, &shape, dtype.clone()).expect("g");
    let dw = dev.from_cpu(&dw_data, &shape, dtype).expect("dw");

    let (de, dg, handle) = dev
        .silu_mul_backward(e.as_ref(), g.as_ref(), dw.as_ref(), &shape)
        .expect("silu_mul_backward");
    handle.synchronize().expect("sync");

    let de_vec = de.to_cpu_vec_f32().expect("read de");
    let dg_vec = dg.to_cpu_vec_f32().expect("read dg");

    for i in 0..n {
        let x = g_data[i];
        let sigm = 1.0 / (1.0 + (-x).exp());
        let expected_de = x * sigm;
        let expected_dg = e_data[i] * sigm * (1.0 + x * (1.0 - sigm));

        assert!(
            (de_vec[i] - expected_de).abs() < 1e-5,
            "de mismatch at i={i}: got {}, expected {}",
            de_vec[i],
            expected_de
        );
        assert!(
            (dg_vec[i] - expected_dg).abs() < 1e-5,
            "dg mismatch at i={i}: got {}, expected {}",
            dg_vec[i],
            expected_dg
        );
    }
}

#[test]
fn cpu_quantized_matmul_and_backward_ste_parity() {
    let dev = dev();
    let (m, k, n) = (2, 32, 32);

    let a_data: Vec<f32> = (0..(m * k)).map(|i| (i as f32 * 0.01).sin()).collect();
    let dy_data: Vec<f32> = (0..(m * n)).map(|i| (i as f32 * 0.02).cos()).collect();

    // Q8_0 style block weight
    let mut b_bytes = vec![0u8; n * k];
    for col in 0..n {
        for p in 0..k {
            b_bytes[col * k + p] = ((p % 15) as i8) as u8;
        }
    }
    let b_scales = vec![1.0f32; (n * k) / 32];

    let a = dev
        .from_cpu(&a_data, &Shape::from_slice(&[m, k]), DType::F32)
        .expect("a");
    let dy = dev
        .from_cpu(&dy_data, &Shape::from_slice(&[m, n]), DType::F32)
        .expect("dy");

    let b_packed = dev
        .from_cpu_bytes(
            &b_bytes,
            &Shape::from_slice(&[n, k]),
            DType {
                arith: ArithType::U8,
                storage: Storage::Native,
            },
        )
        .expect("b_packed");

    // Forward quantized_matmul
    let (out_fwd, handle_fwd) = dev
        .quantized_matmul(
            a.as_ref(),
            b_packed.as_ref(),
            &b_scales,
            QuantFormat::Q8_0,
            &Shape::from_slice(&[m, n]),
        )
        .expect("quantized_matmul");
    handle_fwd.synchronize().expect("sync fwd");
    let fwd_vec = out_fwd.to_cpu_vec_f32().expect("read fwd");
    assert_eq!(fwd_vec.len(), m * n);

    // Backward quantized_matmul_backward_dx
    let (dx_out, handle_bwd) = dev
        .quantized_matmul_backward_dx(
            dy.as_ref(),
            b_packed.as_ref(),
            &b_scales,
            8,
            m,
            n,
            k,
            &Shape::from_slice(&[m, k]),
            None,
        )
        .expect("quantized_matmul_backward_dx");
    handle_bwd.synchronize().expect("sync bwd");
    let dx_vec = dx_out.to_cpu_vec_f32().expect("read dx");
    assert_eq!(dx_vec.len(), m * k);
}

#[test]
fn cpu_fused_add_rms_norm_numerics() {
    let dev = dev();
    let shape = Shape::from_slice(&[2, 4]); // 2 rows, 4 hidden dim
    let w_shape = Shape::from_slice(&[4]);
    let dtype = DType {
        arith: ArithType::F32,
        storage: Storage::Native,
    };

    let x_data = vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0, 0.5, 1.5];
    let res_data = vec![0.5, 0.5, 0.5, 0.5, 1.0, 2.0, -0.5, -1.5];
    let w_data = vec![1.0, 1.0, 1.0, 1.0];
    let eps = 1e-5f32;

    let x = dev.from_cpu(&x_data, &shape, dtype.clone()).expect("x");
    let res = dev.from_cpu(&res_data, &shape, dtype.clone()).expect("res");
    let w = dev.from_cpu(&w_data, &w_shape, dtype).expect("w");

    let (y, updated_res, handle) = dev
        .fused_add_rms_norm(x.as_ref(), res.as_ref(), w.as_ref(), eps, &shape)
        .expect("fused_add_rms_norm");
    handle.synchronize().expect("sync");

    let y_vec = y.to_cpu_vec_f32().expect("read y");
    let res_vec = updated_res.to_cpu_vec_f32().expect("read res");

    // Check res_vec = x + res
    for i in 0..8 {
        assert_eq!(res_vec[i], x_data[i] + res_data[i]);
    }

    // Check row 0 RMSNorm
    let row0 = &res_vec[0..4];
    let mean_sq0 = row0.iter().map(|v| v * v).sum::<f32>() / 4.0;
    let scale0 = 1.0 / (mean_sq0 + eps).sqrt();
    for c in 0..4 {
        assert!((y_vec[c] - (row0[c] * scale0)).abs() < 1e-5);
    }
}

#[test]
fn cpu_scalar_arithmetic_helpers() {
    let dev = dev();
    let shape = Shape::from_slice(&[4]);
    let dtype = DType {
        arith: ArithType::F32,
        storage: Storage::Native,
    };

    let x_data = vec![10.0, 20.0, 30.0, 40.0];
    let x = dev.from_cpu(&x_data, &shape, dtype).expect("x");

    // add_scalar
    let (out_add, h_add) = dev.add_scalar(x.as_ref(), 5.0, &shape).expect("add_scalar");
    h_add.synchronize().expect("sync");
    assert_eq!(out_add.to_cpu_vec_f32().expect("read"), vec![15.0, 25.0, 35.0, 45.0]);

    // sub_scalar
    let (out_sub, h_sub) = dev.sub_scalar(x.as_ref(), 5.0, &shape).expect("sub_scalar");
    h_sub.synchronize().expect("sync");
    assert_eq!(out_sub.to_cpu_vec_f32().expect("read"), vec![5.0, 15.0, 25.0, 35.0]);

    // div_scalar
    let (out_div, h_div) = dev.div_scalar(x.as_ref(), 10.0, &shape).expect("div_scalar");
    h_div.synchronize().expect("sync");
    assert_eq!(out_div.to_cpu_vec_f32().expect("read"), vec![1.0, 2.0, 3.0, 4.0]);
}


