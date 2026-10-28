use grim_core::model::CausalLm;
use grim_core::session::{Inner as SessionInner, SessionT};
use grim_engine::model_loader::load_model_from_gguf;
use grim_tensor::{CoreTensorOps, Device, Shape, Tensor, dtype::DType, dtype::QuantProvenance};
use std::sync::Arc;
use std::time::Instant;

fn build_t(
    data: &[f32],
    shape: &Shape,
    device: &Device,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let dtype = DType::F32;
    let dev = grim_backend_rocm::RocmDevice::shared(0);
    let storage: Arc<dyn grim_tensor::BackendStorage> =
        Arc::from(dev.from_cpu(data, shape, dtype.clone())?);
    Ok(Tensor::new(
        storage,
        shape.clone(),
        dtype,
        QuantProvenance::default(),
        device.clone(),
    ))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "/drive/bigfast/grim/models/qwen38-27b/Qwen3.8-27B-Q4_K_M.gguf";
    eprintln!("[bench] Loading model...");
    let device = Device::Rocm(0);
    let model = load_model_from_gguf(model_path, device.clone())?;

    let provider = grim_format::GgufProvider::open(model_path)?;
    let tokenizer = provider.tokenizer()?;
    let base_text = "In physics and mathematics, the phase space of a dynamical system is a space in which all possible states of a system are represented, with each possible state corresponding to one unique point in the phase space. For mechanical systems, the phase space usually consists of all possible values of position and momentum variables. The concept of phase space was developed in the late 19th century by Ludwig Boltzmann, Henri Poincaré, and Josiah Willard Gibbs. In a phase space, every degree of freedom or parameter of the system is represented as an axis of a multidimensional space; a one-dimensional system is called a phase line, while a two-dimensional system is called a phase plane. For every possible state of the system, or allowed combination of values of the system's parameters, a point is included in the multidimensional space. The system's evolving state over time traces a path (a phase space trajectory for the system) through the high-dimensional space. The phase space represents all that the system can be, and its shape can easily elucidate qualities of the system that might not be obvious otherwise. ";

    let mut full_tokens = tokenizer.encode(base_text);
    while full_tokens.len() < 20000 {
        full_tokens.extend(tokenizer.encode(base_text));
    }

    let chunk_size = 1024;
    let prompt_lengths = vec![512, 2048, 4096, 8192, 16384];

    eprintln!(
        "\n| Prompt Len | Chunks | Total Time (ms) | ms/token | Per-Chunk Latencies (ms) | Dev0 Min Free (MB) | Dev1 Min Free (MB) |"
    );
    eprintln!("|---|---|---|---|---|---|---|");

    for &seq_len in &prompt_lengths {
        let (f0_start, _) = grim_backend_rocm::vram_info(0);
        let (f1_start, _) = grim_backend_rocm::vram_info(1);
        let mut min_f0 = f0_start;
        let mut min_f1 = f1_start;

        let prompt_tokens: Vec<u32> = full_tokens[..seq_len].to_vec();
        let mut sess = SessionInner::new(model.device().clone());

        let num_chunks = (seq_len + chunk_size - 1) / chunk_size;
        let base_chunk_len = seq_len / num_chunks;
        let remainder = seq_len % num_chunks;

        let mut chunk_start = 0;
        let mut chunk_times = Vec::new();

        let t_total_start = Instant::now();

        for chunk_idx in 0..num_chunks {
            let cur_chunk_len = base_chunk_len + if chunk_idx < remainder { 1 } else { 0 };
            let chunk_end = chunk_start + cur_chunk_len;
            let is_last = chunk_end == seq_len;
            sess.set_skip_output_projection(!is_last);

            let sub_ids: Vec<f32> = prompt_tokens[chunk_start..chunk_end]
                .iter()
                .map(|&t| t as f32)
                .collect();
            let sub_pos: Vec<f32> = (chunk_start..chunk_end).map(|i| i as f32).collect();
            let sub_len = sub_ids.len();

            let sub_in_t = build_t(&sub_ids, &Shape::new(vec![sub_len]), &device)?;
            let sub_pos_t = build_t(&sub_pos, &Shape::new(vec![sub_len]), &device)?;

            let t_chunk = Instant::now();
            let _ = CausalLm::forward(&*model, &mut sess, &sub_in_t, &sub_pos_t, &[])?;
            let chunk_ms = t_chunk.elapsed().as_secs_f64() * 1000.0;
            chunk_times.push(chunk_ms);

            let (f0, _) = grim_backend_rocm::vram_info(0);
            let (f1, _) = grim_backend_rocm::vram_info(1);
            if f0 < min_f0 {
                min_f0 = f0;
            }
            if f1 < min_f1 {
                min_f1 = f1;
            }

            chunk_start = chunk_end;
        }

        let total_ms = t_total_start.elapsed().as_secs_f64() * 1000.0;
        let ms_per_token = total_ms / (seq_len as f64);
        let chunk_str = chunk_times
            .iter()
            .map(|t| format!("{:.1}", t))
            .collect::<Vec<_>>()
            .join(", ");
        eprintln!(
            "| {:5} | {:2} | {:10.2} | {:8.3} | [{}] | {:6.0} | {:6.0} |",
            seq_len,
            num_chunks,
            total_ms,
            ms_per_token,
            chunk_str,
            min_f0 as f64 / 1e6,
            min_f1 as f64 / 1e6
        );
    }

    Ok(())
}
