//! Which D2D decode guard actually fires, on the REAL checkpoint, per token?
//!
//! `gated_delta_net_forward_d2d` and `attention_layer_d2d` decline through
//! `Ok(None)` for a dozen different reasons and, before `GRIM_D2D_TRACE`, said
//! nothing about which. That made "the host fallback is being taken every
//! token" an inference rather than a measurement, and it made every fix a
//! guess. This test settles it: it loads the real GGUF, runs real single-token
//! decode steps, and attributes each decline to the guard that produced it.
//!
//! Run it as the Phase-1 evidence, not as a correctness gate:
//!
//! ```text
//! GRIM_GPU_TEST=1 GRIM_D2D_TRACE=1 \
//!   cargo test -p grim-engine --test qwen35_d2d_decline_trace \
//!     -- --nocapture --ignored
//! ```
//!
//! `GRIM_CHECKPOINT` selects the model (default: the 27B Q4_K_M in ~/.grim).
//! `GRIM_TRACE_TOKENS` sets the decode-step count (default 3 — long enough to
//! separate prefill from steady state, short enough to finish).

use grim_backend_rocm::RocmDevice;
use grim_core::model::CausalLm;
use grim_tensor::{CoreTensorOps, DType, Device, Shape, Tensor};

/// Decode steps to run. The first is a prefill-shaped step in most drivers;
/// the rest are steady-state, which is the only regime the plan cares about.
fn trace_tokens() -> usize {
    std::env::var("GRIM_TRACE_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3)
}

fn checkpoint() -> String {
    std::env::var("GRIM_CHECKPOINT").unwrap_or_else(|_| {
        let p = std::env::var("HOME").unwrap_or_default();
        format!("{p}/.grim/models/Qwen3.8-27B-Q4_K_M.gguf")
    })
}

fn main_ignored() {
    let path = checkpoint();
    if !std::path::Path::new(&path).exists() {
        eprintln!("[SKIP] no checkpoint at {path}; set GRIM_CHECKPOINT");
        return;
    }
    // The harness used to hardcode ordinal 0 for both the device and the input
    // tensors. With `GRIM_GPUS=1` the model loads onto ordinal 1, so the input
    // tensors were built on ordinal 0 and the very first D2H in forward crossed
    // devices — `hipMemcpyDtoH failed: 1` on every token, which reads exactly
    // like a model bug and is not one. Everything (device, inputs, model) must
    // be on the same ordinal.
    let ordinal: usize = std::env::var("GRIM_GPU_ORDINAL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let dev = std::panic::catch_unwind(|| {
        RocmDevice::try_new(ordinal)
            .unwrap_or_else(|e| panic!("RocmDevice::try_new({ordinal}): {e}"))
    })
    .expect("ROCm device unavailable");
    let device = Device::Rocm(ordinal);

    // Load through the production loader path, not a hand-built config: the
    // whole question is whether the *shipped* loader produces tensors the D2D
    // guards accept.
    let model = load_model(&path, device.clone());
    let mut session = model.new_session();

    let n = trace_tokens();

    println!("== D2D decline trace ==");
    println!("checkpoint: {path}");
    println!("device:     {device:?}");
    println!("decode steps: {n}");
    println!();

    for step in 0..n {
        // One real token, position = step, so the first call is a prefill of
        // length 1 and every later call is ordinary steady-state decode.
        let tok = (1000 + step) as f32;
        let input = dev_tensor(&dev, device.clone(), &[tok], Shape::new(vec![1, 1]));
        let pos = dev_tensor(&dev, device.clone(), &[step as f32], Shape::new(vec![1, 1]));

        eprintln!("--- token {step} begin ---");
        let out = model.forward(&mut *session, &input, &pos, &[]);
        eprintln!("--- token {step} end ---");

        match out {
            Ok(t) => {
                // A real forward returning Ok proves the host fallback path is
                // still able to produce a number; it does not prove D2D ran.
                let dev0 = t.device().clone();
                println!("token {step}: ok, logits device {dev0:?}");
            }
            Err(e) => println!("token {step}: ERR {e}"),
        }
    }

    println!();
    println!("Read the [d2d-decline] lines above: any guard that fires on every");
    println!("token is an unconditional fallback to the host reference.");
}

/// Load the real checkpoint through the engine's production loader.
fn load_model(path: &str, device: Device) -> Box<dyn CausalLm> {
    // The engine owns GGUF -> config -> Qwen35::load_tp, including the exact
    // metadata reads this bug depends on. Re-deriving a config here would test
    // my guesses instead of the shipping path.
    grim_engine::model_loader::load_model_from_gguf(path, device)
        .unwrap_or_else(|e| panic!("load {path}: {e}"))
}

fn dev_tensor(dev: &RocmDevice, device: Device, data: &[f32], shape: Shape) -> Tensor {
    let storage = CoreTensorOps::from_cpu(dev, data, &shape, DType::F32).expect("from_cpu");
    Tensor::new(
        std::sync::Arc::from(storage),
        shape,
        DType::F32,
        Default::default(),
        device,
    )
}

#[test]
#[ignore = "loads a real 17 GB checkpoint and a real GPU; run explicitly with GRIM_GPU_TEST=1"]
fn qwen35_d2d_decline_trace_on_real_checkpoint() {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run");
        return;
    }
    main_ignored();
}
