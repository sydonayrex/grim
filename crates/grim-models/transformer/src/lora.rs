//! LoRA adapter application - fused-LoRA path for `CausalLm`.
//! §4.5: the architecture commits to batched LoRA serving as a `CausalLm` capability.

use grim_backend_cpu::cpu_tensor;
use grim_core::error::Error;
use grim_core::error::Result;
use grim_core::model::AdapterHandle;
use grim_tensor::Shape;
use grim_tensor::Tensor;

/// Apply each active adapter as a low-rank bias added to the logits row.
/// `hidden_size` is the model's hidden dimension (rank of A's second axis).
pub fn apply_adapters_to_logits(
    logits: &Tensor,
    adapters: &[AdapterHandle],
    hidden_size: usize,
) -> Result<Tensor> {
    // Filter to logits-targeted adapters (or legacy adapters with no target specified)
    let logits_adapters: Vec<&AdapterHandle> = adapters
        .iter()
        .filter(|a| matches!(a.target, None | Some(grim_core::model::TargetModule::Logits)))
        .collect();

    if logits_adapters.is_empty() {
        return Ok(logits.clone());
    }
    let shape_dims = logits.shape().dims().to_vec();
    // Flatten non-2D logits (e.g. [batch, seq, vocab]) to 2D [batch*seq, vocab]
    // so LoRA applies uniformly. Reshape back after.
    let original_dims = shape_dims.clone();
    let needs_reshape = shape_dims.len() != 2;
    let logits_2d = if needs_reshape {
        let flat_len: usize = shape_dims.iter().product();
        let vocab = shape_dims.last().copied().unwrap_or(1);
        let seq_len = flat_len / vocab;
        let data = logits.to_vec_f32()?;
        let dev = grim_nn::modules::pick_device_for_tensor(logits);
        let shape = Shape::new(vec![seq_len, vocab]);
        Tensor::new(
            std::sync::Arc::from(dev.from_cpu(&data, &shape, grim_tensor::dtype::DType::F32)?),
            shape,
            grim_tensor::dtype::DType::F32,
            logits.provenance().clone(),
            logits.device().clone(),
        )
    } else {
        logits.clone()
    };
    let (seq_len, vocab) = (logits_2d.shape().dims()[0], logits_2d.shape().dims()[1]);

    let dev = grim_nn::modules::pick_device_for_tensor(&logits_2d);
    let is_cpu = matches!(logits_2d.device(), grim_tensor::Device::Cpu);

    if !is_cpu {
        // GPU path: the fused `lora_accumulate` kernel computes `out = base + scale * (x @ A^T) @ B^T` entirely on-device - no transposes, no eager syncs, no host roundtrips.
        // Backends without the kernel degrade to the staged matmul chain below.
        let mut running_logits = logits_2d.clone();
        for adapter in logits_adapters {
            let rank = adapter
                .a
                .shape()
                .dim(0)
                .map_err(|e| Error::Shape(e.to_string()))?;
            let in_dim = adapter
                .a
                .shape()
                .dim(1)
                .map_err(|e| Error::Shape(e.to_string()))?;
            if in_dim != hidden_size {
                return Err(Error::Shape(format!(
                    "LoRA A in_dim {in_dim} != model hidden_size {hidden_size}"
                )));
            }
            let out_dim = adapter
                .b
                .shape()
                .dim(0)
                .map_err(|e| Error::Shape(e.to_string()))?;
            if out_dim != vocab {
                return Err(Error::Shape(format!(
                    "LoRA B out_dim {out_dim} != vocab {vocab}"
                )));
            }

            // scale is alpha / rank
            let scale = adapter.alpha / rank as f32;

            let fused = dev.lora_accumulate(
                running_logits.storage().as_ref(),
                logits_2d.storage().as_ref(),
                adapter.a.storage().as_ref(),
                adapter.b.storage().as_ref(),
                scale,
                logits_2d.shape(),
            );
            match fused {
                Ok((s, _handle)) => {
                    running_logits = Tensor::new(
                        std::sync::Arc::from(s),
                        logits_2d.shape().clone(),
                        grim_tensor::dtype::DType::F32,
                        logits_2d.provenance().clone(),
                        logits_2d.device().clone(),
                    );
                    continue;
                }
                Err(e) if !grim_nn::is_kernel_unimplemented(&e) => {
                    return Err(Error::from(e));
                }
                Err(_) => {} // kernel missing — staged fallback below
            }

            // Staged fallback: matmul → matmul → scale → add, all on-device.
            // No eager `synchronize()` calls: intermediates are consumed by further device ops and sync lazily on.
            let a_t = transpose_last_two(&adapter.a)?;
            let (temp_s, _h1) = dev.matmul(
                running_logits.storage().as_ref(),
                a_t.storage().as_ref(),
                &Shape::new(vec![seq_len, rank]),
            )?;
            let temp_tensor = Tensor::new(
                std::sync::Arc::from(temp_s),
                Shape::new(vec![seq_len, rank]),
                grim_tensor::dtype::DType::F32,
                logits_2d.provenance().clone(),
                logits_2d.device().clone(),
            );

            let b_t = transpose_last_two(&adapter.b)?;
            let (delta_s, _h2) = dev.matmul(
                temp_tensor.storage().as_ref(),
                b_t.storage().as_ref(),
                &Shape::new(vec![seq_len, vocab]),
            )?;
            let delta_tensor = Tensor::new(
                std::sync::Arc::from(delta_s),
                Shape::new(vec![seq_len, vocab]),
                grim_tensor::dtype::DType::F32,
                logits_2d.provenance().clone(),
                logits_2d.device().clone(),
            );

            // Scale on-device; host roundtrip only if the backend lacks the
            // scalar kernel.
            let scaled_delta_tensor = match dev.mul_scalar(
                delta_tensor.storage().as_ref(),
                scale,
                delta_tensor.shape(),
            ) {
                Ok((s, _h)) => Tensor::new(
                    std::sync::Arc::from(s),
                    delta_tensor.shape().clone(),
                    grim_tensor::dtype::DType::F32,
                    logits_2d.provenance().clone(),
                    logits_2d.device().clone(),
                ),
                Err(e) if !grim_nn::is_kernel_unimplemented(&e) => {
                    return Err(Error::from(e));
                }
                Err(_) => {
                    let mut delta_vec = delta_tensor.to_vec_f32()?;
                    for val in &mut delta_vec {
                        *val *= scale;
                    }
                    let scaled_delta_s = dev.from_cpu(
                        &delta_vec,
                        delta_tensor.shape(),
                        grim_tensor::dtype::DType::F32,
                    )?;
                    Tensor::new(
                        std::sync::Arc::from(scaled_delta_s),
                        delta_tensor.shape().clone(),
                        grim_tensor::dtype::DType::F32,
                        logits_2d.provenance().clone(),
                        logits_2d.device().clone(),
                    )
                }
            };

            let (added_s, _h3) = dev.add(
                running_logits.storage().as_ref(),
                scaled_delta_tensor.storage().as_ref(),
                logits_2d.shape(),
            )?;
            running_logits = Tensor::new(
                std::sync::Arc::from(added_s),
                logits_2d.shape().clone(),
                grim_tensor::dtype::DType::F32,
                logits_2d.provenance().clone(),
                logits_2d.device().clone(),
            );
        }
        // Reshape back to original shape if we flattened — stays on-device
        // via a zero-copy/D2D relabel instead of a host roundtrip.
        if needs_reshape {
            return crate::block::reshaped_view(&running_logits, &Shape::new(original_dims));
        }
        return Ok(running_logits);
    }

    // CPU fallback path:
    let mut acc = vec![0.0f32; seq_len * vocab];
    for adapter in logits_adapters {
        let rank = adapter
            .a
            .shape()
            .dim(0)
            .map_err(|e| Error::Shape(e.to_string()))?;
        let in_dim = adapter
            .a
            .shape()
            .dim(1)
            .map_err(|e| Error::Shape(e.to_string()))?;
        if in_dim != hidden_size {
            return Err(Error::Shape(format!(
                "LoRA A in_dim {in_dim} != model hidden_size {hidden_size}"
            )));
        }
        let out_dim = adapter
            .b
            .shape()
            .dim(0)
            .map_err(|e| Error::Shape(e.to_string()))?;
        if out_dim != vocab {
            return Err(Error::Shape(format!(
                "LoRA B out_dim {out_dim} != vocab {vocab}"
            )));
        }
        // MED-5: The CPU LoRA path only has access to `logits` (shape [seq_len, vocab]), not the hidden state (shape [seq_len, hidden_size]).
        // When hidden_size != vocab, using logits as a surrogate silently reads the wrong element.
        if in_dim != vocab {
            return Err(Error::Shape(format!(
                "CPU LoRA path requires hidden_size ({hidden_size}) == vocab ({vocab}) because it \
                 uses the final logits tensor as a hidden-state surrogate. \
                 hidden_size ({}) and vocab ({}) differ.",
                in_dim, vocab
            )));
        }
        let scale = adapter.alpha / rank as f32;
        let a_data = adapter.a.to_vec_f32()?;
        let b_data = adapter.b.to_vec_f32()?;
        let in_dim = adapter
            .a
            .shape()
            .dim(1)
            .map_err(|e| Error::Shape(e.to_string()))?;
        let logits_data = logits_2d.to_vec_f32()?;
        for token in 0..seq_len {
            for vocab_j in 0..vocab {
                let mut total = 0.0f32;
                for r in 0..rank {
                    let mut inner = 0.0f32;
                    for h in 0..in_dim {
                        inner +=
                            a_data[r * in_dim + h] * logits_data[token * vocab + h.min(vocab - 1)];
                    }
                    total += b_data[vocab_j * rank + r] * inner;
                }
                acc[token * vocab + vocab_j] += scale * total;
            }
        }
    }
    let mut base = logits_2d.to_vec_f32()?;
    for i in 0..base.len() {
        base[i] += acc[i];
    }
    if needs_reshape {
        Ok(cpu_tensor(base, Shape::new(original_dims)))
    } else {
        Ok(cpu_tensor(base, Shape::new(shape_dims)))
    }
}

/// Apply active adapters matching `(layer, target)` to the linear projection activation `act`
/// and accumulate the delta onto `out`:
/// `out += (alpha / r) * (act @ A^T) @ B^T`
pub fn apply_adapters_to_projection(
    act: &Tensor,
    out: &Tensor,
    adapters: &[AdapterHandle],
    layer: usize,
    target: grim_core::model::TargetModule,
) -> Result<Tensor> {
    let matching: Vec<&AdapterHandle> = adapters
        .iter()
        .filter(|a| a.layer_idx == Some(layer) && a.target == Some(target))
        .collect();

    if matching.is_empty() {
        return Ok(out.clone());
    }

    let dev = grim_nn::modules::pick_device_for_tensor(out);
    let mut current_out = out.clone();

    for adapter in matching {
        let rank = adapter
            .a
            .shape()
            .dim(0)
            .map_err(|e| Error::Shape(e.to_string()))?;
        let in_dim = adapter
            .a
            .shape()
            .dim(1)
            .map_err(|e| Error::Shape(e.to_string()))?;
        let out_dim = adapter
            .b
            .shape()
            .dim(0)
            .map_err(|e| Error::Shape(e.to_string()))?;

        let scale = adapter.alpha / rank as f32;

        if matches!(out.device(), grim_tensor::Device::Cpu) {
            let act_dims = act.shape().dims();
            let act_2d = if act_dims.len() == 2 {
                act.clone()
            } else {
                let rows: usize = act_dims[..act_dims.len() - 1].iter().product();
                let cols = *act_dims.last().unwrap_or(&1);
                crate::block::reshaped_view(act, &Shape::new(vec![rows, cols]))?
            };
            let (tokens, feat) = (act_2d.shape().dims()[0], act_2d.shape().dims()[1]);
            if feat != in_dim {
                return Err(Error::Shape(format!(
                    "LoRA A in_dim {in_dim} != act last dim {feat}"
                )));
            }

            let act_vec = act_2d.to_vec_f32()?;
            let a_vec = adapter.a.to_vec_f32()?;
            let b_vec = adapter.b.to_vec_f32()?;
            let mut out_vec = current_out.to_vec_f32()?;

            // act: [tokens, in_dim]
            // A: [rank, in_dim] -> A^T: [in_dim, rank]
            // B: [out_dim, rank] -> B^T: [rank, out_dim]
            for t in 0..tokens {
                for o in 0..out_dim {
                    let mut delta = 0.0f32;
                    for r in 0..rank {
                        let mut inter = 0.0f32;
                        for i in 0..in_dim {
                            inter += act_vec[t * in_dim + i] * a_vec[r * in_dim + i];
                        }
                        delta += b_vec[o * rank + r] * inter;
                    }
                    out_vec[t * out_dim + o] += scale * delta;
                }
            }

            current_out = cpu_tensor(out_vec, current_out.shape().clone());
        } else {
            // GPU path: staged matmul or lora_accumulate
            let act_dims = act.shape().dims();
            let act_2d = if act_dims.len() == 2 {
                act.clone()
            } else {
                let rows: usize = act_dims[..act_dims.len() - 1].iter().product();
                let cols = *act_dims.last().unwrap_or(&1);
                crate::block::reshaped_view(act, &Shape::new(vec![rows, cols]))?
            };
            let tokens = act_2d.shape().dims()[0];

            let a_t = transpose_last_two(&adapter.a)?;
            let (temp_s, _) = dev.matmul(
                act_2d.storage().as_ref(),
                a_t.storage().as_ref(),
                &Shape::new(vec![tokens, rank]),
            )?;
            let temp_tensor = Tensor::new(
                std::sync::Arc::from(temp_s),
                Shape::new(vec![tokens, rank]),
                grim_tensor::dtype::DType::F32,
                act.provenance().clone(),
                act.device().clone(),
            );

            let b_t = transpose_last_two(&adapter.b)?;
            let (delta_s, _) = dev.matmul(
                temp_tensor.storage().as_ref(),
                b_t.storage().as_ref(),
                &Shape::new(vec![tokens, out_dim]),
            )?;
            let delta_tensor = Tensor::new(
                std::sync::Arc::from(delta_s),
                Shape::new(vec![tokens, out_dim]),
                grim_tensor::dtype::DType::F32,
                act.provenance().clone(),
                act.device().clone(),
            );

            let scaled_delta = match dev.mul_scalar(
                delta_tensor.storage().as_ref(),
                scale,
                delta_tensor.shape(),
            ) {
                Ok((s, _)) => Tensor::new(
                    std::sync::Arc::from(s),
                    delta_tensor.shape().clone(),
                    grim_tensor::dtype::DType::F32,
                    act.provenance().clone(),
                    act.device().clone(),
                ),
                Err(e) if !grim_nn::is_kernel_unimplemented(&e) => return Err(Error::from(e)),
                Err(_) => {
                    let mut d_vec = delta_tensor.to_vec_f32()?;
                    for v in &mut d_vec {
                        *v *= scale;
                    }
                    let s = dev.from_cpu(&d_vec, delta_tensor.shape(), grim_tensor::dtype::DType::F32)?;
                    Tensor::new(
                        std::sync::Arc::from(s),
                        delta_tensor.shape().clone(),
                        grim_tensor::dtype::DType::F32,
                        act.provenance().clone(),
                        act.device().clone(),
                    )
                }
            };

            let (added_s, _) = dev.add(
                current_out.storage().as_ref(),
                scaled_delta.storage().as_ref(),
                current_out.shape(),
            )?;
            current_out = Tensor::new(
                std::sync::Arc::from(added_s),
                current_out.shape().clone(),
                grim_tensor::dtype::DType::F32,
                current_out.provenance().clone(),
                current_out.device().clone(),
            );
        }
    }

    Ok(current_out)
}

/// Helper to transpose the last two dimensions of a 2D tensor.
fn transpose_last_two(tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Shape("Transpose expects a 2D tensor".into()));
    }
    let rows = dims[0];
    let cols = dims[1];
    let data = tensor.to_vec_f32()?;
    let mut transposed = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            transposed[c * rows + r] = data[r * cols + c];
        }
    }
    let dev = grim_nn::modules::pick_device_for_tensor(tensor);
    let shape = Shape::new(vec![cols, rows]);
    let storage = dev.from_cpu(&transposed, &shape, grim_tensor::dtype::DType::F32)?;
    Ok(Tensor::new(
        std::sync::Arc::from(storage),
        shape,
        grim_tensor::dtype::DType::F32,
        tensor.provenance().clone(),
        tensor.device().clone(),
    ))
}
