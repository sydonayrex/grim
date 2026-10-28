use grim_core::model::CausalLm;
use grim_core::session::Inner as SessionInner;
use grim_engine::model_loader::load_model_from_gguf;
use grim_tensor::{CoreTensorOps, Device, Shape, Tensor, dtype::DType, dtype::QuantProvenance};
use std::sync::Arc;

fn build_t(
    data: &[f32],
    shape: &Shape,
    device: &Device,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let dev = grim_backend_rocm::RocmDevice::shared(0);
    let storage: Arc<dyn grim_tensor::BackendStorage> =
        Arc::from(dev.from_cpu(data, shape, DType::F32)?);
    Ok(Tensor::new(
        storage,
        shape.clone(),
        DType::F32,
        QuantProvenance::default(),
        device.clone(),
    ))
}

fn run_sweep(
    tag: &str,
    model: &Box<dyn CausalLm>,
    device: &Device,
    full_tokens: &[u32],
) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("\n=== Running Determinism Sweep [{tag}] ===");
    for seq_len in [64, 128, 1024] {
        let prompt_tokens: Vec<u32> = full_tokens[..seq_len].to_vec();
        let in_ids_f32: Vec<f32> = prompt_tokens.iter().map(|&t| t as f32).collect();
        let pos_f32: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();

        let in_t1 = build_t(&in_ids_f32, &Shape::new(vec![seq_len]), &device)?;
        let pos_t1 = build_t(&pos_f32, &Shape::new(vec![seq_len]), &device)?;
        let in_t2 = build_t(&in_ids_f32, &Shape::new(vec![seq_len]), &device)?;
        let pos_t2 = build_t(&pos_f32, &Shape::new(vec![seq_len]), &device)?;

        let mut sess1 = SessionInner::new(model.device().clone());
        let out1 = CausalLm::forward(&**model, &mut sess1, &in_t1, &pos_t1, &[])?.to_vec_f32()?;

        let mut sess2 = SessionInner::new(model.device().clone());
        let out2 = CausalLm::forward(&**model, &mut sess2, &in_t2, &pos_t2, &[])?.to_vec_f32()?;

        let mut max_diff = 0.0f32;
        let mut sum_diff = 0.0f64;
        for i in 0..out1.len() {
            let d = (out1[i] - out2[i]).abs();
            if d > max_diff {
                max_diff = d;
            }
            sum_diff += d as f64;
        }
        let mean_diff = sum_diff / out1.len() as f64;
        eprintln!(
            "[{tag}] [seq_len={:4}] max_diff={:.6e}, mean_diff={:.6e}",
            seq_len, max_diff, mean_diff
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "/drive/bigfast/grim/models/qwen38-27b/Qwen3.8-27B-Q4_K_M.gguf";
    let device = Device::Rocm(0);
    let model = load_model_from_gguf(model_path, device.clone())?;

    let provider = grim_format::GgufProvider::open(model_path)?;
    let tokenizer = provider.tokenizer()?;
    let base_text = "In physics and mathematics, the phase space of a dynamical system is a space in which all possible states of a system are represented, with each possible state corresponding to one unique point in the phase space. For mechanical systems, the phase space usually consists of all possible values of position and momentum variables. The concept of phase space was developed in the late 19th century by Ludwig Boltzmann, Henri Poincaré, and Josiah Willard Gibbs. In a phase space, every degree of freedom or parameter of the system is represented as an axis of a multidimensional space; a one-dimensional system is called a phase line, while a two-dimensional system is called a phase plane. For every possible state of the system, or allowed combination of values of the system's parameters, a point is included in the multidimensional space. The system's evolving state over time traces a path (a phase space trajectory for the system) through the high-dimensional space. The phase space represents all that the system can be, and its shape can easily elucidate qualities of the system that might not be obvious otherwise. ";

    let mut full_tokens = tokenizer.encode(base_text);
    while full_tokens.len() < 2200 {
        full_tokens.extend(tokenizer.encode(base_text));
    }

    let dev0 = grim_backend_rocm::RocmDevice::shared(0);
    let h0 = dev0.get_rocblas_handle()?;
    let mut mode: grim_backend_rocm::RocblasAtomicsMode = 0;
    unsafe {
        grim_backend_rocm::rocblas_get_atomics_mode(h0, &mut mode);
    }
    eprintln!(
        "[atomics_probe] dev0 initial rocblas atomics mode: {mode} (0=not_allowed, 1=allowed)"
    );

    // Test with atomics allowed
    unsafe {
        grim_backend_rocm::rocblas_set_atomics_mode(h0, grim_backend_rocm::rocblas_atomics_allowed);
    }
    run_sweep("atomics_allowed", &model, &device, &full_tokens)?;

    // Test with atomics not allowed
    unsafe {
        grim_backend_rocm::rocblas_set_atomics_mode(
            h0,
            grim_backend_rocm::rocblas_atomics_not_allowed,
        );
    }
    run_sweep("atomics_not_allowed", &model, &device, &full_tokens)?;

    Ok(())
}
