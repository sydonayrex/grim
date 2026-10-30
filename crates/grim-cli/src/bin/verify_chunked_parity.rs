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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "/drive/bigfast/grim/models/qwen38-27b/Qwen3.8-27B-Q4_K_M.gguf";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Model path not found: {model_path}");
        return Ok(());
    }

    eprintln!("[parity-gate] Loading model...");
    let device = Device::Rocm(0);
    let model = load_model_from_gguf(model_path, device.clone())?;

    let provider = grim_format::GgufProvider::open(model_path)?;
    let tokenizer = provider.tokenizer()?;
    let base_text = "In physics and mathematics, the phase space of a dynamical system is a space in which all possible states of a system are represented, with each possible state corresponding to one unique point in the phase space. For mechanical systems, the phase space usually consists of all possible values of position and momentum variables. The concept of phase space was developed in the late 19th century by Ludwig Boltzmann, Henri Poincaré, and Josiah Willard Gibbs. In a phase space, every degree of freedom or parameter of the system is represented as an axis of a multidimensional space; a one-dimensional system is called a phase line, while a two-dimensional system is called a phase plane. For every possible state of the system, or allowed combination of values of the system's parameters, a point is included in the multidimensional space. The system's evolving state over time traces a path (a phase space trajectory for the system) through the high-dimensional space. The phase space represents all that the system can be, and its shape can easily elucidate qualities of the system that might not be obvious otherwise. ";

    let mut full_tokens = tokenizer.encode(base_text);
    while full_tokens.len() < 2200 {
        full_tokens.extend(tokenizer.encode(base_text));
    }

    // We test three prompt sequence lengths:
    // 1) 2048 tokens (standard 2k context parity: chunk 1024 + 1024)
    // 2) 1025 tokens (chunk_size + 1 awkward boundary: chunk 1024 + 1)
    // 3) 1027 tokens (final chunk of 3 tokens: chunk 1024 + 3)
    let test_lengths = vec![2048, 1025, 1027];
    let chunk_size = 1024;

    for &seq_len in &test_lengths {
        eprintln!("\n=======================================================");
        eprintln!("[parity-gate] Testing sequence length {seq_len} (chunk_size {chunk_size})");
        eprintln!("=======================================================");

        let prompt_tokens: Vec<u32> = full_tokens[..seq_len].to_vec();

        // ----------------------------------------------------
        // Pass A: Unchunked Prefill
        // ----------------------------------------------------
        eprintln!("[parity-gate] Running unchunked prefill pass...");
        let mut sess_unchunked = SessionInner::new(model.device().clone());
        let in_ids_f32: Vec<f32> = prompt_tokens.iter().map(|&t| t as f32).collect();
        let pos_f32: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();

        let in_t = build_t(&in_ids_f32, &Shape::new(vec![seq_len]), &device)?;
        let pos_t = build_t(&pos_f32, &Shape::new(vec![seq_len]), &device)?;

        let unchunked_logits = CausalLm::forward(&*model, &mut sess_unchunked, &in_t, &pos_t, &[])?;
        let unchunked_vec = unchunked_logits.to_vec_f32()?;
        let vocab = unchunked_vec.len();
        eprintln!("[parity-gate] Unchunked prefill complete. Logits shape len: {vocab}");

        // ----------------------------------------------------
        // Pass B: Chunked Prefill
        // ----------------------------------------------------
        eprintln!("[parity-gate] Running chunked prefill pass...");
        let mut sess_chunked = SessionInner::new(model.device().clone());
        let mut chunk_start = 0;
        let mut chunked_logits = None;

        while chunk_start < seq_len {
            let chunk_end = (chunk_start + chunk_size).min(seq_len);
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

        let chunked_out = chunked_logits.expect("final chunk produced logits");
        let chunked_vec = chunked_out.to_vec_f32()?;

        // ----------------------------------------------------
        // Parity Verification
        // ----------------------------------------------------
        assert_eq!(
            unchunked_vec.len(),
            chunked_vec.len(),
            "Vocab size mismatch"
        );

        let mut max_abs_diff = 0.0f32;
        let mut max_abs_tok = 0u32;
        let mut sum_abs_diff = 0.0f64;
        for i in 0..vocab {
            let u = unchunked_vec[i];
            let c = chunked_vec[i];
            let abs_diff = (u - c).abs();
            sum_abs_diff += abs_diff as f64;
            if abs_diff > max_abs_diff {
                max_abs_diff = abs_diff;
                max_abs_tok = i as u32;
            }
        }
        let mean_abs_diff = sum_abs_diff / (vocab as f64);

        // Top 5 agreement
        let mut unchunked_top5: Vec<(u32, f32)> = (0..vocab as u32)
            .map(|i| (i, unchunked_vec[i as usize]))
            .collect();
        unchunked_top5.sort_by(|a, b| b.1.total_cmp(&a.1));
        unchunked_top5.truncate(5);

        let mut chunked_top5: Vec<(u32, f32)> = (0..vocab as u32)
            .map(|i| (i, chunked_vec[i as usize]))
            .collect();
        chunked_top5.sort_by(|a, b| b.1.total_cmp(&a.1));
        chunked_top5.truncate(5);

        let top1_abs_diff = (unchunked_top5[0].1 - chunked_top5[0].1).abs();
        let top1_rel_diff = top1_abs_diff / (unchunked_top5[0].1.abs().max(1e-5));

        eprintln!("[parity-gate] Mean abs diff across vocab: {mean_abs_diff:.6}");
        eprintln!(
            "[parity-gate] Max abs diff: {max_abs_diff:.6} (at token {max_abs_tok}: u={:.4}, c={:.4})",
            unchunked_vec[max_abs_tok as usize], chunked_vec[max_abs_tok as usize]
        );
        eprintln!("[parity-gate] Top-1 abs diff: {top1_abs_diff:.6}, rel diff: {top1_rel_diff:.6}");
        eprintln!("[parity-gate] Unchunked top 5: {:?}", unchunked_top5);
        eprintln!("[parity-gate] Chunked top 5:   {:?}", chunked_top5);

        let top1_match = unchunked_top5[0].0 == chunked_top5[0].0;
        let top5_overlap_count = unchunked_top5
            .iter()
            .filter(|(u_tok, _)| chunked_top5.iter().any(|(c_tok, _)| c_tok == u_tok))
            .count();
        eprintln!(
            "[parity-gate] Top-1 exact match: {top1_match}, Top-5 overlap count: {top5_overlap_count}/5"
        );

        if !top1_match || top5_overlap_count < 4 {
            return Err("Parity failed: Top-1 mismatch or Top-5 overlap < 4".into());
        }

        // Teacher-forced 32 generated tokens
        eprintln!("[parity-gate] Verifying 32 teacher-forced decode steps...");
        let mut test_curr = unchunked_top5[0].0;
        let mut decode_diffs = 0;
        for step in 0..32 {
            let next_pos = seq_len + step;
            let in_tok = build_t(&[test_curr as f32], &Shape::new(vec![1]), &device)?;
            let p_tok = build_t(&[next_pos as f32], &Shape::new(vec![1]), &device)?;

            let out_u = CausalLm::forward(&*model, &mut sess_unchunked, &in_tok, &p_tok, &[])?
                .to_vec_f32()?;
            let out_c = CausalLm::forward(&*model, &mut sess_chunked, &in_tok, &p_tok, &[])?
                .to_vec_f32()?;

            let mut best_u = 0u32;
            let mut best_u_val = f32::NEG_INFINITY;
            let mut best_c = 0u32;
            let mut best_c_val = f32::NEG_INFINITY;
            for i in 0..vocab {
                if out_u[i] > best_u_val {
                    best_u_val = out_u[i];
                    best_u = i as u32;
                }
                if out_c[i] > best_c_val {
                    best_c_val = out_c[i];
                    best_c = i as u32;
                }
            }

            if best_u != best_c {
                eprintln!(
                    "[parity-gate] Disagreement at decode step {step}: unchunked={best_u}, chunked={best_c}"
                );
                decode_diffs += 1;
            }
            test_curr = best_u;
        }

        eprintln!("[parity-gate] Teacher-forced 32 tokens decode mismatches: {decode_diffs}/32");
        if decode_diffs > 4 {
            return Err(
                "Teacher-forced decode mismatches > 4 between chunked and unchunked prefill".into(),
            );
        }
    }

    eprintln!("\n[parity-gate] ALL PARITY GATES PASSED SUCCESSFULLY!");
    Ok(())
}
