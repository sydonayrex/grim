use grim_core::model::CausalLm;
use grim_core::session::{Inner as SessionInner, SessionT};
use grim_engine::model_loader::load_model_from_gguf;
use grim_tensor::{CoreTensorOps, Device, Shape, Tensor, dtype::DType, dtype::QuantProvenance};
use std::sync::Arc;

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

fn compute_kl_and_metrics(
    u: &[f32],
    c: &[f32],
    vocab: usize,
) -> (f32, f32, f64, Vec<(usize, f32, f32)>) {
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    for i in 0..vocab {
        let diff = (u[i] - c[i]).abs();
        if diff > max_abs {
            max_abs = diff;
        }
        sum_abs += diff as f64;
    }
    let mean_abs = sum_abs / (vocab as f64);

    // Softmax and KL divergence: KL(P_u || P_c) = sum P_u * (log P_u - log P_c)
    let max_u = u.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let max_c = c.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp_u: f64 = u.iter().map(|&x| ((x - max_u) as f64).exp()).sum();
    let sum_exp_c: f64 = c.iter().map(|&x| ((x - max_c) as f64).exp()).sum();
    let lse_u = max_u as f64 + sum_exp_u.ln();
    let lse_c = max_c as f64 + sum_exp_c.ln();

    let mut kl = 0.0f64;
    for i in 0..vocab {
        let log_p_u = u[i] as f64 - lse_u;
        let log_p_c = c[i] as f64 - lse_c;
        let p_u = log_p_u.exp();
        if p_u > 1e-12 {
            kl += p_u * (log_p_u - log_p_c);
        }
    }

    // Top-5 tokens of unchunked reference
    let mut indices: Vec<usize> = (0..vocab).collect();
    indices.sort_by(|&a, &b| u[b].partial_cmp(&u[a]).unwrap());
    let top5: Vec<(usize, f32, f32)> = indices
        .iter()
        .take(5)
        .map(|&idx| (idx, u[idx], c[idx]))
        .collect();

    (max_abs, mean_abs as f32, kl, top5)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "/drive/bigfast/grim/models/qwen38-27b/Qwen3.8-27B-Q4_K_M.gguf";
    eprintln!("[diagnose] Loading model...");
    let device = Device::Rocm(0);
    let model = load_model_from_gguf(model_path, device.clone())?;

    let provider = grim_format::GgufProvider::open(model_path)?;
    let tokenizer = provider.tokenizer()?;
    let base_text = "In physics and mathematics, the phase space of a dynamical system is a space in which all possible states of a system are represented, with each possible state corresponding to one unique point in the phase space. For mechanical systems, the phase space usually consists of all possible values of position and momentum variables. The concept of phase space was developed in the late 19th century by Ludwig Boltzmann, Henri Poincaré, and Josiah Willard Gibbs. In a phase space, every degree of freedom or parameter of the system is represented as an axis of a multidimensional space; a one-dimensional system is called a phase line, while a two-dimensional system is called a phase plane. For every possible state of the system, or allowed combination of values of the system's parameters, a point is included in the multidimensional space. The system's evolving state over time traces a path (a phase space trajectory for the system) through the high-dimensional space. The phase space represents all that the system can be, and its shape can easily elucidate qualities of the system that might not be obvious otherwise. ";

    let mut full_tokens = tokenizer.encode(base_text);
    while full_tokens.len() < 2200 {
        full_tokens.extend(tokenizer.encode(base_text));
    }

    let test_configs = vec![
        (
            "Control: 1024 tokens chunked with chunk_size=1024 (1 chunk)",
            1024,
            1024,
            false,
        ),
        ("2048 tokens chunked 1024+1024", 2048, 1024, false),
        ("1025 tokens greedy chunked 1024+1", 1025, 1024, false),
        ("1025 tokens balanced chunked 513+512", 1025, 1024, true),
        ("1027 tokens greedy chunked 1024+3", 1027, 1024, false),
        ("1027 tokens balanced chunked 514+513", 1027, 1024, true),
    ];

    for (name, seq_len, chunk_size, balanced) in test_configs {
        eprintln!("\n=======================================================");
        eprintln!(
            "[diagnose] Config: {name} (seq_len={seq_len}, chunk_size={chunk_size}, balanced={balanced})"
        );
        eprintln!("=======================================================");

        let prompt_tokens: Vec<u32> = full_tokens[..seq_len].to_vec();

        // 1. Unchunked
        let mut sess_unchunked = SessionInner::new(model.device().clone());
        let in_ids_f32: Vec<f32> = prompt_tokens.iter().map(|&t| t as f32).collect();
        let pos_f32: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
        let in_t = build_t(&in_ids_f32, &Shape::new(vec![seq_len]), &device)?;
        let pos_t = build_t(&pos_f32, &Shape::new(vec![seq_len]), &device)?;

        let unchunked_logits = CausalLm::forward(&*model, &mut sess_unchunked, &in_t, &pos_t, &[])?;
        let unchunked_vec = unchunked_logits.to_vec_f32()?;
        let vocab = unchunked_vec.len();

        // 2. Chunked
        let mut sess_chunked = SessionInner::new(model.device().clone());
        let mut chunk_start = 0;
        let mut chunked_logits = None;

        let num_chunks = (seq_len + chunk_size - 1) / chunk_size;
        let base_chunk_len = seq_len / num_chunks;
        let remainder = seq_len % num_chunks;

        for chunk_idx in 0..num_chunks {
            let cur_chunk_len = if balanced {
                base_chunk_len + if chunk_idx < remainder { 1 } else { 0 }
            } else {
                chunk_size.min(seq_len - chunk_start)
            };
            let chunk_end = chunk_start + cur_chunk_len;
            let is_last = chunk_end == seq_len;
            sess_chunked.set_skip_output_projection(!is_last);

            let sub_ids: Vec<f32> = prompt_tokens[chunk_start..chunk_end]
                .iter()
                .map(|&t| t as f32)
                .collect();
            let sub_pos: Vec<f32> = (chunk_start..chunk_end).map(|i| i as f32).collect();
            let sub_len = sub_ids.len();

            let sub_in_t = build_t(&sub_ids, &Shape::new(vec![sub_len]), &device)?;
            let sub_pos_t = build_t(&sub_pos, &Shape::new(vec![sub_len]), &device)?;

            let out = CausalLm::forward(&*model, &mut sess_chunked, &sub_in_t, &sub_pos_t, &[])?;
            if is_last {
                chunked_logits = Some(out);
            }
            chunk_start = chunk_end;
        }
        sess_chunked.set_skip_output_projection(false);
        let chunked_out = chunked_logits.expect("logits produced");
        let chunked_vec = chunked_out.to_vec_f32()?;

        let (max_abs, mean_abs, kl, top5) =
            compute_kl_and_metrics(&unchunked_vec, &chunked_vec, vocab);
        eprintln!(
            "[diagnose] Results: max_abs_diff={max_abs:.6}, mean_abs_diff={mean_abs:.6}, KL(softmax)={kl:.6e}"
        );
        eprintln!("[diagnose] Top-5 reference logits vs chunked:");
        for (rank, (idx, u_val, c_val)) in top5.iter().enumerate() {
            let diff = c_val - u_val;
            eprintln!(
                "   Rank {}: Token {idx} | Ref={u_val:.4} | Chunked={c_val:.4} (diff={diff:+.4})",
                rank + 1
            );
        }
    }

    Ok(())
}
