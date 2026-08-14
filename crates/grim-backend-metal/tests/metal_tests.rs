//! Parity tests for the Metal backend dispatch methods.
//!
//! On Apple these exercise the GPU fast-path via MSL kernels.
//! On non-Apple platforms the CPU fallback is exercised, which keeps
//! the suite green in headless CI environments without an Apple GPU.

use grim_backend_metal::MetalDevice;
use grim_tensor::dtype::DType;
use grim_tensor::{BackendDevice, BackendStorage, Shape};
use grim_tensor::{ScytheLink, ScythePlacement};

#[test]
fn test_metal_all_reduce_parity() {
    let dev = MetalDevice::new(0).unwrap();

    let shape = Shape::new(vec![8]);
    let inputs_data = vec![
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![0.5f32, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
    ];
    let storages: Vec<Box<dyn BackendStorage>> = inputs_data
        .iter()
        .map(|v| dev.from_cpu(v, &shape, DType::F32).unwrap())
        .collect();
    let refs: Vec<&dyn BackendStorage> = storages.iter().map(|s| s.as_ref()).collect();

    let (out, handle) = dev.all_reduce(&refs, "sum").unwrap();
    handle.synchronize().unwrap();
    let result = out.to_cpu_vec_f32().unwrap();

    let expected: Vec<f32> = (0..8)
        .map(|i| inputs_data.iter().map(|v| v[i]).sum::<f32>())
        .collect();
    assert_eq!(result.len(), expected.len());
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-5, "all_reduce mismatch: {} != {}", r, e);
    }
}

#[test]
fn test_metal_all_reduce_single_input_parity() {
    let dev = MetalDevice::new(0).unwrap();

    let shape = Shape::new(vec![4]);
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let storage = dev.from_cpu(&data, &shape, DType::F32).unwrap();
    let refs: Vec<&dyn BackendStorage> = vec![storage.as_ref()];

    let (out, _) = dev.all_reduce(&refs, "sum").unwrap();
    let result = out.to_cpu_vec_f32().unwrap();

    for (r, e) in result.iter().zip(data.iter()) {
        assert!(
            (r - e).abs() < 1e-5,
            "all_reduce single mismatch: {} != {}",
            r,
            e
        );
    }
}

#[test]
fn test_metal_comm_fuse_reduce_parity() {
    let dev = MetalDevice::new(0).unwrap();

    let m = 2usize;
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0]; // [2, 2]
    let b_data = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0]; // [2, 3]
    let shape_a = Shape::new(vec![m, 2]);
    let shape_b = Shape::new(vec![m, 3]);

    let a = dev.from_cpu(&a_data, &shape_a, DType::F32).unwrap();
    let b = dev.from_cpu(&b_data, &shape_b, DType::F32).unwrap();

    let placement = ScythePlacement {
        ranks: vec![0, 1],
        partition: vec![0.5, 0.5],
        routes: vec![ScytheLink::Host; 4],
    };
    let partials: Vec<(&dyn BackendStorage, &ScythePlacement)> =
        vec![(a.as_ref(), &placement), (b.as_ref(), &placement)];

    let out = dev.comm_fuse_reduce(&partials).unwrap();
    let result = out.to_cpu_vec_f32().unwrap();

    // Column-concat: [[1, 2, 10, 20, 30], [3, 4, 40, 50, 60]]
    let expected = vec![1.0f32, 2.0, 10.0, 20.0, 30.0, 3.0, 4.0, 40.0, 50.0, 60.0];
    assert_eq!(result.len(), expected.len());
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-5, "comm_fuse mismatch: {} != {}", r, e);
    }
}

/// Fused grouped MoE dispatch parity (WI-M5).
///
/// 2 experts, 2 tokens, top-1 routing, hidden=4, inter=3, rsf=0.5. The
/// router arrays are f32-backed (Metal has no integer storage in this crate)
/// and the shader casts them back to `int`. On Apple this exercises the MSL
/// kernel; elsewhere it exercises the verified CPU fallback that the GPU path
/// must match exactly.
#[test]
fn test_metal_moe_fused_dispatch_parity() {
    let dev = MetalDevice::new(0).unwrap();

    let hidden: u32 = 4;
    let inter: u32 = 3;
    let num_experts: u32 = 2;
    let batch: u32 = 2;
    let rsf: f32 = 0.5;

    // Token hidden states.
    let x_data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, // token 0
        5.0, 6.0, 7.0, 8.0, // token 1
    ];

    // Expert gate/up/down weights: [num_experts, inter*hidden] / [hidden*inter].
    let mut gate_flat = Vec::new();
    let mut up_flat = Vec::new();
    let mut down_flat = Vec::new();
    let mut w = 0.1f32;
    for _ in 0..(num_experts as usize * inter as usize * hidden as usize) {
        gate_flat.push(w);
        up_flat.push(w + 0.05);
        w += 0.01;
    }
    w = 0.2f32;
    for _ in 0..(num_experts as usize * hidden as usize * inter as usize) {
        down_flat.push(w);
        w += 0.02;
    }

    // Routing: token 0 -> expert 1, token 1 -> expert 0 (top-1). f32-backed.
    let rtok: Vec<f32> = vec![0.0, 1.0];
    let rexp: Vec<f32> = vec![1.0, 0.0];
    let rw: Vec<f32> = vec![1.0, 1.0];

    let x = BackendDevice::from_cpu(
        &dev,
        &x_data,
        &Shape::new(vec![batch as usize * hidden as usize]),
        DType::F32,
    )
    .unwrap();
    let gate = BackendDevice::from_cpu(
        &dev,
        &gate_flat,
        &Shape::new(vec![
            num_experts as usize * inter as usize * hidden as usize,
        ]),
        DType::F32,
    )
    .unwrap();
    let up = BackendDevice::from_cpu(
        &dev,
        &up_flat,
        &Shape::new(vec![
            num_experts as usize * inter as usize * hidden as usize,
        ]),
        DType::F32,
    )
    .unwrap();
    let down = BackendDevice::from_cpu(
        &dev,
        &down_flat,
        &Shape::new(vec![
            num_experts as usize * hidden as usize * inter as usize,
        ]),
        DType::F32,
    )
    .unwrap();
    let tok = BackendDevice::from_cpu(&dev, &rtok, &Shape::new(vec![2]), DType::F32).unwrap();
    let exp = BackendDevice::from_cpu(&dev, &rexp, &Shape::new(vec![2]), DType::F32).unwrap();
    let rw_s = BackendDevice::from_cpu(&dev, &rw, &Shape::new(vec![2]), DType::F32).unwrap();

    let out_shape = Shape::new(vec![batch as usize, hidden as usize]);
    let (out, handle) = dev
        .moe_fused_dispatch(
            x.as_ref(),
            gate.as_ref(),
            up.as_ref(),
            down.as_ref(),
            tok.as_ref(),
            exp.as_ref(),
            rw_s.as_ref(),
            &out_shape,
            hidden,
            inter,
            num_experts,
            batch,
            rsf,
        )
        .unwrap();
    handle.synchronize().unwrap();
    let result = out.to_cpu_vec_f32().unwrap();

    // Reference (matches the MSL kernel / CPU fallback exactly).
    let hidden_us = hidden as usize;
    let inter_us = inter as usize;
    let batch_us = batch as usize;
    let xv = &x_data;
    let mut expected = vec![0.0f32; batch_us * hidden_us];
    for tok_i in 0..batch_us {
        let x_base = tok_i * hidden_us;
        for p in 0..2 {
            if rtok[p] as usize != tok_i {
                continue;
            }
            let exp_id = rexp[p] as usize;
            let weight = rw[p];
            let gw_base = exp_id * inter_us * hidden_us;
            let uw_base = exp_id * inter_us * hidden_us;
            let dw_base = exp_id * hidden_us * inter_us;
            for h in 0..hidden_us {
                let mut dn = 0.0f32;
                for i in 0..inter_us {
                    let mut g = 0.0f32;
                    let mut u = 0.0f32;
                    for j in 0..hidden_us {
                        let xvj = xv[x_base + j];
                        g += gate_flat[gw_base + i * hidden_us + j] * xvj;
                        u += up_flat[uw_base + i * hidden_us + j] * xvj;
                    }
                    let a = (g / (1.0f32 + (-g).exp())) * u;
                    dn += down_flat[dw_base + h * inter_us + i] * a;
                }
                expected[tok_i * hidden_us + h] += rsf * weight * dn;
            }
        }
    }

    assert_eq!(result.len(), expected.len());
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-3, "moe mismatch: {} != {}", r, e);
    }
}
