//! grim-autograd audit A3 gate: `matmul_backward`'s quantized-B GPU fast
//! path must produce the SAME gradients as the CPU reference.
//!
//! The pre-fix fast path computed grad_b as `dev.matmul(A, G, b.shape())` —
//! under the row-major matmul convention that is `A @ G`, not the required
//! `Aᵀ @ G`: dimensionally invalid for m ≠ k (hard error) and silently wrong
//! for square m == k. Device-gated: `GRIM_GPU_TEST=1`.

use grim_autograd::ops::{MatMulArgs, matmul_backward};
use grim_backend_rocm::RocmDevice;
use grim_tensor::{CoreTensorOps, DType, Device, MemoryOps, Shape, Tensor};
use std::sync::Arc;

fn tensor_from_storage(
    st: Box<dyn grim_tensor::BackendStorage>,
    shape: Shape,
    device: Device,
) -> Tensor {
    // Preserve the storage's real dtype — the quantized-B tensor must stay
    // KQuant or matmul_backward's fast-path gate won't engage (that exact
    // mistake hid the defect in the first version of this gate).
    let dtype = st.dtype();
    Tensor::new(
        Arc::from(st),
        shape,
        dtype,
        grim_tensor::QuantProvenance::default(),
        device,
    )
}

#[test]
fn quantized_b_gpu_backward_matches_cpu_reference() {
    let dev = match RocmDevice::try_new(0) {
        Ok(d) => d,
        Err(_) => return,
    };
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[skipped: GRIM_GPU_TEST not set]");
        return;
    }

    // Deliberately NON-square (m=2, k=4, n=3): the pre-fix GPU path's
    // A@G operand order is dimensionally invalid here.
    let (m, k, n) = (2usize, 4usize, 3usize);
    let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32) * 0.3 - 1.0).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 5) as f32) * 0.25 - 0.5).collect();
    let g_data: Vec<f32> = (0..m * n).map(|i| ((i % 9) as f32) * 0.2 - 0.8).collect();

    // GPU tensors; B packed as Q4K on host, uploaded as KQuant storage
    // (mirrors the golden q4k tests — quantize_on_device only supports
    // Q8_0/Fp8 and the Q8_0 kernel is mid-refactor by the parallel
    // compressed-tensors workstream).
    let a_shape = Shape::new(vec![m, k]);
    let b_shape = Shape::new(vec![k, n]);
    let g_shape = Shape::new(vec![m, n]);
    let a_gpu = dev.from_cpu(&a_data, &a_shape, DType::F32).unwrap();
    let g_gpu = dev.from_cpu(&g_data, &g_shape, DType::F32).unwrap();
    let b_packed = grim_quant::quant_q4k(&b_data).expect("quant_q4k");
    let b_deq = grim_quant::dequant_q4k(&b_packed, b_data.len()).expect("dequant_q4k");
    let q4k_dtype = DType {
        arith: grim_tensor::ArithType::F32,
        storage: grim_tensor::Storage::KQuant(grim_tensor::KQuantScheme::Q4K),
    };
    let b_quant = dev.from_cpu_bytes(&b_packed, &b_shape, q4k_dtype).unwrap();

    let a_cpu = grim_backend_cpu::CpuDevice::new();
    let a_ref = a_cpu.from_cpu(&a_data, &a_shape, DType::F32).unwrap();
    let b_ref = a_cpu.from_cpu(&b_deq, &b_shape, DType::F32).unwrap();
    let g_ref = a_cpu.from_cpu(&g_data, &g_shape, DType::F32).unwrap();

    let (ref_ga, ref_gb) = matmul_backward(&MatMulArgs {
        a: tensor_from_storage(a_ref, a_shape.clone(), Device::Cpu),
        b: tensor_from_storage(b_ref, b_shape.clone(), Device::Cpu),
        out_grad: tensor_from_storage(g_ref, g_shape.clone(), Device::Cpu),
        transpose_a: false,
        transpose_b: false,
    })
    .expect("CPU reference backward");

    let (gpu_ga, gpu_gb) = matmul_backward(&MatMulArgs {
        a: tensor_from_storage(a_gpu, a_shape.clone(), Device::Rocm(0)),
        b: tensor_from_storage(b_quant, b_shape.clone(), Device::Rocm(0)),
        out_grad: tensor_from_storage(g_gpu, g_shape.clone(), Device::Rocm(0)),
        transpose_a: false,
        transpose_b: false,
    })
    .unwrap_or_else(|e| panic!("quantized GPU matmul_backward failed: {e}"));

    for (name, got, want) in [
        (
            "grad_a",
            gpu_ga.to_vec_f32().unwrap(),
            ref_ga.to_vec_f32().unwrap(),
        ),
        (
            "grad_b",
            gpu_gb.to_vec_f32().unwrap(),
            ref_gb.to_vec_f32().unwrap(),
        ),
    ] {
        assert_eq!(got.len(), want.len(), "{name} length");
        let md = got
            .iter()
            .zip(want.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            md < 5e-2,
            "{name}: quantized GPU backward diverged from CPU reference (max diff {md:.4})\n\
             gpu={got:?}\nref={want:?}"
        );
    }
}
