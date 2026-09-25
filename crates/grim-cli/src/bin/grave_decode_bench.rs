//! G4a decode benchmark (GRAVE plan §5 gate 6): tok/s for the trained
//! LFM2.5-350M-Q8_0 + grave-2 sidecar (per-token GdlGate projections active).
//! Times repeated single-token decode forward passes (the fused device path);
//! correctness is covered by gla_fused_parity.rs and the audit_tests, this is
//! purely throughput. Usage:
//!   grave_decode_bench <gguf> <sidecar.grave.json> [device_ordinal=0]

use grim_backend_cpu as _;
use grim_backend_rocm as _;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: grave_decode_bench <gguf> <sidecar.grave.json> [device_ordinal=0]");
        std::process::exit(2);
    }
    let gguf = &args[1];
    let sidecar = &args[2];
    let ordinal: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(0);

    unsafe {
        std::env::set_var("GRIM_LFM2_ATTENTION_MODE", "gdl");
        std::env::set_var("GRIM_GRAVE_SIDECAR", sidecar);
        std::env::remove_var("GRIM_GRAVE");
        std::env::remove_var("GRIM_GRAVE_GATES");
    }

    let device = grim_tensor::Device::Rocm(ordinal);
    eprintln!("[grave_decode_bench] loading {gguf} (ordinal {ordinal}) sidecar={sidecar}");
    let model = grim_engine::model_loader::load_model_from_gguf(gguf, device).unwrap_or_else(|e| {
        eprintln!("load failed: {e}");
        std::process::exit(1);
    });
    let model: &dyn grim_core::model::CausalLm = &*model;

    // Synthesize a prompt of `ctx_len` arbitrary token ids, prefill once, then
    // step a fixed decode token repeatedly and time it.
    let decode_token: u32 = 42;
    let lens = [32usize, 64, 128, 256, 512, 1024, 2048];
    let warmup = 20usize;
    let iters = 200usize;

    for ctx in lens {
        let prompt: Vec<u32> = (0..ctx as u32).map(|i| (i % 100) + 1).collect();
        let (tps, ms) = bench(model, &prompt, decode_token, warmup, iters);
        println!("ctx={ctx:>5} tok/s={tps:>7.1} latency={ms:>6.2} ms/step");
    }
    println!("[grave_decode_bench] done");
}

fn bench(
    model: &dyn grim_core::model::CausalLm,
    prompt: &[u32],
    tok: u32,
    warmup: usize,
    iters: usize,
) -> (f64, f64) {
    let dev = model.device().clone();

    // Prefill the prompt to seed caches.
    let mut sess = model.new_session();
    let ids: Vec<f32> = prompt.iter().map(|&t| t as f32).collect();
    let in_t = tot(&ids, &[1, prompt.len()], dev.clone());
    let pos: Vec<f32> = (0..prompt.len()).map(|p| p as f32).collect();
    let pos_t = tot(&pos, &[1, prompt.len()], dev.clone());
    let _ = model
        .forward(sess.as_mut(), &in_t, &pos_t, &[])
        .expect("prefill");
    let base = prompt.len();

    // warmup
    for i in 0..warmup {
        let in_t = tot(&[tok as f32], &[1, 1], dev.clone());
        let pos_t = tot(&[(base + i) as f32], &[1, 1], dev.clone());
        let _ = model.forward(sess.as_mut(), &in_t, &pos_t, &[]);
    }
    // measure
    let start = Instant::now();
    for i in 0..iters {
        let in_t = tot(&[tok as f32], &[1, 1], dev.clone());
        let pos_t = tot(&[(base + warmup + i) as f32], &[1, 1], dev.clone());
        let _ = model.forward(sess.as_mut(), &in_t, &pos_t, &[]);
    }
    let us = start.elapsed().as_micros() as u64;
    let secs = us as f64 / 1e6;
    (iters as f64 / secs, secs * 1000.0 / iters as f64)
}

use grim_tensor::{CoreTensorOps, DType, Device};

fn tot(data: &[f32], dims: &[usize], device: Device) -> grim_tensor::Tensor {
    let shape = grim_tensor::Shape::new(dims.to_vec());
    let storage = match &device {
        Device::Cpu => grim_backend_cpu::CpuDevice::new()
            .from_cpu(data, &shape, DType::F32)
            .unwrap(),
        Device::Rocm(o) => grim_backend_rocm::RocmDevice::shared(*o)
            .from_cpu(data, &shape, DType::F32)
            .unwrap(),
        _ => panic!("unsupported"),
    };
    grim_tensor::Tensor::new(
        Arc::from(storage),
        shape,
        DType::F32,
        grim_tensor::QuantProvenance::GrimNative,
        device,
    )
}
