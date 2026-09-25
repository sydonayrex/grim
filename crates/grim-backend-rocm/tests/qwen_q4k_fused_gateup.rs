//! Q4_K fused GateUp projection parity for the Qwen hybrid model path.

use grim_backend_rocm::{as_rocm, RocmDevice};
use grim_tensor::{
    ArithType, CoreTensorOps, DType, KQuantScheme, MemoryOps, QuantFormat, QuantOps, Shape, Storage,
};

#[test]
#[ignore]
fn q4k_fused_gateup_matches_separate_q4k_projections() {
    if !grim_backend_rocm::gpu_test_enabled() || !RocmDevice::probe_one(0).unwrap_or(false) {
        return;
    }
    let dev = RocmDevice::shared(0);
    let (m, n, k) = (1usize, 128usize, 256usize);
    let a_data: Vec<f32> = (0..m * k)
        .map(|i| ((i % 29) as f32 - 14.0) / 100.0)
        .collect();
    let gate_data: Vec<f32> = (0..n * k)
        .map(|i| ((i % 31) as f32 - 15.0) / 100.0)
        .collect();
    let up_data: Vec<f32> = (0..n * k)
        .map(|i| ((i % 37) as f32 - 18.0) / 100.0)
        .collect();
    let a = dev
        .from_cpu(&a_data, &Shape::new(vec![m, k]), DType::F32)
        .unwrap();
    let gate_q4_bytes = grim_quant::quant_q4k(&gate_data).unwrap();
    let up_q4_bytes = grim_quant::quant_q4k(&up_data).unwrap();
    let q4_dtype = DType {
        arith: ArithType::F32,
        storage: Storage::KQuant(KQuantScheme::Q4K),
    };
    let gate_q4 = dev
        .from_cpu_bytes(
            &gate_q4_bytes,
            &Shape::new(vec![n, k]),
            q4_dtype.clone(),
        )
        .unwrap();
    let up_q4 = dev
        .from_cpu_bytes(&up_q4_bytes, &Shape::new(vec![n, k]), q4_dtype)
        .unwrap();

    let fused = dev
        .build_fused_gate_up_q4k(gate_q4.as_ref(), up_q4.as_ref())
        .unwrap();
    let fused_out = dev
        .launch_fused_gate_up_q4k(as_rocm(a.as_ref()).unwrap(), &fused)
        .unwrap()
        .to_cpu_vec_f32()
        .unwrap();

    let (gate_out, gate_handle) = dev
        .quantized_matmul(
            a.as_ref(),
            gate_q4.as_ref(),
            &[],
            QuantFormat::Q4K,
            &Shape::new(vec![m, n]),
        )
        .unwrap();
    gate_handle.synchronize().unwrap();
    let (up_out, up_handle) = dev
        .quantized_matmul(
            a.as_ref(),
            up_q4.as_ref(),
            &[],
            QuantFormat::Q4K,
            &Shape::new(vec![m, n]),
        )
        .unwrap();
    up_handle.synchronize().unwrap();
    let gate = gate_out.to_cpu_vec_f32().unwrap();
    let up = up_out.to_cpu_vec_f32().unwrap();
    let mut split = Vec::with_capacity(2 * n);
    split.extend_from_slice(&gate);
    split.extend_from_slice(&up);
    let max_abs = fused_out
        .iter()
        .zip(&split)
        .map(|(fused, split)| (fused - split).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs < 2e-3, "Q4_K fused GateUp max abs error {max_abs}");
}
