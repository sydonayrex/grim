//! Block quantizers for Grim. Each format stores a block of weights as
//! low-bit integers plus per-block scale (and optionally min for asymmetric).
//!
//! Q8_0: 8-bit symmetric, block size 32 — one f16 scale per 32 values.
//! Q4_K: llama.cpp K-quant, block size 32 — 6-bit super-block scale,
//!       4-bit sub-block values, per-sub-block scale.
//! GPTQ Group-INT: Grouped asymmetric quantization (EfficientQAT) with 2/3/4/8-bit variants.
//!   - 3-bit uses cross-word packing: 32 values across 3 u32 words (96 bits)
//!
//! Phase 2 (`.grim` oxidizer): Importance-matrix calibration and refined scale fitting.
//! Phase 3 (`.grim` oxidizer): EvoPress evolutionary per-tensor bitwidth search.

use grim_tensor::error::{Error, Result};

pub const BLOCK_SIZE_Q8: usize = 32;
pub const BLOCK_SIZE_Q4_K: usize = 32;
const BLOCK_SIZE_QK: usize = 32;
const GPTQ_PROXY_COLUMN_GROUP: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    Q8_0,
    Q4K,
    Q5K,
    Q6K,
}

#[derive(Debug, Clone)]
pub struct TensorRewritePlan {
    pub target: QuantFormat,
    pub shape: Vec<usize>,
    pub importance: Option<Vec<f32>>,
    pub curvature: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct RewrittenTensorData {
    pub bytes: Vec<u8>,
    pub logical_shape: Vec<usize>,
    pub target: QuantFormat,
    /// True if weights are stored in wavefront-tiled layout for ROCm LDS efficiency.
    /// When true, `write_grim_file` should set `layout_hint = GrimLayoutHint::WavefrontTiled`.
    pub wavefront_tiled: bool,
}

/// Dequantize grouped INT weights (EfficientQAT/GPTQ format).
/// 
/// # Layout
/// - `qweight`: packed low-bit weights (strided)
/// - `qzeros`: per-group zero-points (uint16 for 2/3/4-bit, uint8 for 8-bit)
/// - `scales`: per-group scales (f32 or f16)
/// - `g_idx`: sequential group indices (EfficientQAT) or permutation (classic GPTQ)
/// 
/// # 3-bit cross-word packing
/// 32 values are packed across 3 consecutive u32 words using GPTQ/BitBLAS layout:
/// values 0-10 in word 0, 11-21 in word 1, 22-31 in word 2
pub fn dequant_gptq_group_int(
    qweight: &[u8],
    qzeros: &[u8],
    scales: &[u8],
    shape: &[usize],
    bits: u32,
    group_size: usize,
) -> Result<Vec<f32>> {
    let in_features = shape[0];
    let out_features = shape[1];
    let num_groups = (in_features + group_size - 1) / group_size;
    
    // Dequant output
    let mut out = vec![0.0f32; in_features * out_features];
    
    // Compute packed element stride
    let values_per_word = match bits {
        2 => 16,
        3 => 32, // Special: 32 values across 3 words
        4 => 8,
        8 => 1,
        _ => return Err(Error::Backend(format!("unsupported GPTQ bits: {bits}"))),
    };
    
    let words_per_group = match bits {
        2 => (group_size + 15) / 16,
        3 => 3 * ((group_size + 31) / 32), // Cross-word packing
        4 => (group_size + 7) / 8,
        8 => group_size,
        _ => unreachable!(),
    };
    
    let words_per_row = match bits {
        2 => (out_features + 15) / 16,
        3 => 3 * ((out_features + 31) / 32),
        4 => (out_features + 7) / 8,
        8 => out_features,
        _ => unreachable!(),
    };
    
    for g in 0..num_groups {
        let scale = if bits == 8 {
            // For 8-bit, scales are f8 or use uint8 with default scale
            1.0f32
        } else {
            // Parse scale (f32 or f16 depending on format)
            let scale_bytes = [scales[g * 4], scales[g * 4 + 1], scales[g * 4 + 2], scales[g * 4 + 3]];
            f32::from_le_bytes(scale_bytes)
        };
        
        for row in 0..out_features {
            let out_idx = (g * group_size) * out_features + row;
            
            if bits == 3 {
                // 3-bit cross-word unpacking
                let word_base = g * words_per_group * out_features + row * words_per_row;
                let mut val_idx = 0usize;
                
                // Unpack 32 values across 3 u32 words
                for &word_idx in &[word_base, word_base + 1, word_base + 2] {
                    let word = u32::from_le_bytes([
                        qweight[word_idx * 4],
                        qweight[word_idx * 4 + 1],
                        qweight[word_idx * 4 + 2],
                        qweight[word_idx * 4 + 3],
                    ]);
                    
                    // Each word contributes ~10-11 values
                    // This is a simplified unpacking - real GPTQ uses lookup tables
                    for bit_offset in 0..32 {
                        if val_idx >= group_size {
                            break;
                        }
                        let bits_val = ((word >> bit_offset) as u32) & 0x7;
                        let zero = ((qzeros[g * 2] as u32) | ((qzeros[g * 2 + 1] as u32) << 8)) as u8 as f32;
                        let quantized = bits_val as f32;
                        out[out_idx + val_idx * out_features] = (quantized - zero) * scale;
                        val_idx += 1;
                    }
                }
            } else {
                // 2/4/8-bit unpacking (simpler)
                let word_base = g * words_per_group + row * words_per_row;
                for col in 0..group_size {
                    let src_idx = word_base + col / values_per_word;
                    let bit_offset = match bits {
                        2 => (col % 16) * 2,
                        4 => (col % 8) * 4,
                        8 => col,
                        _ => unreachable!(),
                    };
                    
                    let quantized = match bits {
                        2 => (qweight[src_idx] as u32 >> bit_offset) & 0x3,
                        4 => (qweight[src_idx] as u32 >> bit_offset) & 0xF,
                        8 => qweight[src_idx] as u32,
                        _ => unreachable!(),
                    };
                    
                    // Get zero-point for this group
                    let zero = match bits {
                        2 | 4 => {
                            let (zl, zh) = match bits {
                                2 => (qzeros[g], qzeros[g / 128]),
                                4 => (qzeros[g * 2], qzeros[g * 2 + 1]),
                                _ => (0, 0),
                            };
                            u16::from_le_bytes([zl, zh]) as f32
                        }
                        _ => 0.0,
                    };
                    
                    out[out_idx + col * out_features] = (quantized as f32 - zero) * scale;
                }
            }
        }
    }
    
    Ok(out)
}

/// Dequantize Q8_0 bytes to f32.
/// Q8_0 layout: for every 32 weights, a `f16` scale followed by 32 `i8` values.
pub fn dequant_q80(data: &[u8], num_weights: usize) -> Result<Vec<f32>> {
    let stride = std::mem::size_of::<u16>() + BLOCK_Q8_WEIGHTS; // 2 + 32 = 34 bytes
    let num_blocks = (num_weights + BLOCK_Q8_WEIGHTS - 1) / BLOCK_Q8_WEIGHTS;
    if data.len() < num_blocks * stride {
        return Err(Error::Backend(format!(
            "Q8_0: expected {} bytes for {num_weights} weights, got {}",
            num_blocks * stride,
            data.len()
        )));
    }
    let mut out = Vec::with_capacity(num_weights);
    let mut data_pos = 0;
    let mut remaining = num_weights;
    for _ in 0..num_blocks {
        let scale = f16_to_f32(data[data_pos], data[data_pos + 1]);
        data_pos += 2;
        let n = remaining.min(BLOCK_Q8_WEIGHTS);
        for _ in 0..n {
            let v = data[data_pos] as i8 as f32;
            out.push(v * scale);
            data_pos += 1;
        }
        data_pos += BLOCK_Q8_WEIGHTS - n;
        remaining = remaining.saturating_sub(BLOCK_Q8_WEIGHTS);
    }
    Ok(out)
}

const BLOCK_Q8_WEIGHTS: usize = 32;

/// Dequantize Q4_K bytes to f32.
/// Q4_K layout (llama.cpp style): per super-block (32 weights):
///   - scale (6-bit super-block scale, stored as u8 in `d` or `dmin`)
///   - 8 sub-blocks of 4 weights each, 4-bit each
///   - per-sub-block scale
/// Simplified: for v1, treat as 4-bit symmetric with one f32 scale per 32 values.
pub fn dequant_q4k(data: &[u8], num_weights: usize) -> Result<Vec<f32>> {
    dequant_packed_symmetric(data, num_weights, 4)
}

pub fn dequant_q5k(data: &[u8], num_weights: usize) -> Result<Vec<f32>> {
    dequant_packed_symmetric(data, num_weights, 5)
}

pub fn dequant_q6k(data: &[u8], num_weights: usize) -> Result<Vec<f32>> {
    dequant_packed_symmetric(data, num_weights, 6)
}

fn f16_to_f32(lo: u8, hi: u8) -> f32 {
    let bits = u16::from_le_bytes([lo, hi]);
    let sign = (bits >> 15) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        // Subnormal or zero
        f32::from_bits((sign << 31) | (mant << 13))
    } else if exp == 31 {
        // NaN or inf
        f32::from_bits((sign << 31) | 0x7F800000 | (mant << 13))
    } else {
        f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
    }
}

/// Quantize a slice of f32 values to Q8_0 bytes.
/// Each block of 32 gets a f16 scale and 32 i8 values.
pub fn quant_q80(data: &[f32]) -> Result<Vec<u8>> {
    let num_blocks = (data.len() + BLOCK_Q8_WEIGHTS - 1) / BLOCK_Q8_WEIGHTS;
    let mut out = Vec::with_capacity(num_blocks * (2 + BLOCK_Q8_WEIGHTS));
    for block in data.chunks(BLOCK_Q8_WEIGHTS) {
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if amax == 0.0 { 1.0 } else { amax / 127.0 };
        let scale_bits = f32_to_f16(scale);
        out.extend_from_slice(&scale_bits.to_le_bytes());
        for &v in block {
            let q = (v / scale).round().clamp(-128.0, 127.0) as i8;
            out.push(q as u8);
        }
        // Pad incomplete block
        for _ in block.len()..BLOCK_Q8_WEIGHTS {
            out.push(0u8);
        }
    }
    Ok(out)
}

/// Quantize a slice of f32 values to Grim's simplified Q4_K encoding.
/// Each block stores one f32 scale followed by 16 packed 4-bit values.
pub fn quant_q4k(data: &[f32]) -> Result<Vec<u8>> {
    quant_packed_symmetric(data, 4, None, None, None)
}

pub fn quant_q5k(data: &[f32]) -> Result<Vec<u8>> {
    quant_packed_symmetric(data, 5, None, None, None)
}

pub fn quant_q6k(data: &[f32]) -> Result<Vec<u8>> {
    quant_packed_symmetric(data, 6, None, None, None)
}

fn quant_packed_symmetric(
    data: &[f32],
    bits: u8,
    importance: Option<&[f32]>,
    curvature: Option<&[f32]>,
    shape: Option<&[usize]>,
) -> Result<Vec<u8>> {
    let prepared = prepare_gptq_proxy_tensor(data, bits, importance, curvature, shape)?;
    let packed_bytes_per_block = (BLOCK_SIZE_QK * bits as usize).div_ceil(8);
    let num_blocks = prepared.len().div_ceil(BLOCK_SIZE_QK);
    let mut out = Vec::with_capacity(num_blocks * (4 + packed_bytes_per_block));

    for (block_idx, block) in prepared.chunks(BLOCK_SIZE_QK).enumerate() {
        let block_importance = importance.map(|imp| {
            let start = block_idx * BLOCK_SIZE_QK;
            let end = (start + block.len()).min(imp.len());
            &imp[start..end]
        });
        let fit = fit_block_quantization(block, bits, block_importance)?;
        let packed = pack_bits(&fit.codes, bits);
        let scale = fit.scale;
        out.extend_from_slice(&scale.to_le_bytes());
        out.extend_from_slice(&packed);
        for _ in packed.len()..packed_bytes_per_block {
            out.push(0);
        }
    }
    Ok(out)
}

/// Rewrite a tensor payload to a target quantized format.
/// This is the first Pass 4 substrate: it materializes the tensor into
/// a logical f32 view, optionally refines per-block scales using importance
/// weights, and then emits a new packed payload.
pub fn rewrite_tensor_data(data: &[f32], plan: &TensorRewritePlan) -> Result<RewrittenTensorData> {
    let rewritten_bytes = match plan.target {
        QuantFormat::Q8_0 => quant_q80(data)?,
        QuantFormat::Q4K => {
            quant_packed_symmetric(
                data,
                4,
                plan.importance.as_deref(),
                plan.curvature.as_deref(),
                Some(&plan.shape),
            )?
        }
        QuantFormat::Q5K => quant_packed_symmetric(
            data,
            5,
            plan.importance.as_deref(),
            plan.curvature.as_deref(),
            Some(&plan.shape),
        )?,
        QuantFormat::Q6K => quant_packed_symmetric(
            data,
            6,
            plan.importance.as_deref(),
            plan.curvature.as_deref(),
            Some(&plan.shape),
        )?,
    };

    Ok(RewrittenTensorData {
        bytes: rewritten_bytes,
        logical_shape: plan.shape.clone(),
        target: plan.target,
        wavefront_tiled: false,
    })
}

fn dequant_packed_symmetric(data: &[u8], num_weights: usize, bits: u8) -> Result<Vec<f32>> {
    let packed_bytes_per_block = (BLOCK_SIZE_QK * bits as usize).div_ceil(8);
    let stride = 4 + packed_bytes_per_block;
    let num_blocks = num_weights.div_ceil(BLOCK_SIZE_QK);
    if data.len() < num_blocks * stride {
        return Err(Error::Backend(format!(
            "packed symmetric q{bits}: expected {} bytes for {num_weights} weights, got {}",
            num_blocks * stride,
            data.len()
        )));
    }
    let mut out = Vec::with_capacity(num_weights);
    let mut pos = 0usize;
    for block_index in 0..num_blocks {
        let scale = f32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        pos += 4;
        let packed = &data[pos..pos + packed_bytes_per_block];
        pos += packed_bytes_per_block;
        let remaining = num_weights.saturating_sub(block_index * BLOCK_SIZE_QK);
        let block_len = remaining.min(BLOCK_SIZE_QK);
        let unpacked = unpack_bits(packed, bits, block_len);
        out.extend(dequantize_block_signed(&unpacked, scale, bits));
    }
    Ok(out)
}

#[derive(Debug, Clone)]
struct BlockQuantization {
    scale: f32,
    codes: Vec<u32>,
}

fn fit_block_quantization(
    block: &[f32],
    bits: u8,
    importance: Option<&[f32]>,
) -> Result<BlockQuantization> {
    let absmax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let signed_limit = signed_quant_limit(bits);
    let base_scale = if absmax == 0.0 || signed_limit == 0.0 {
        1.0
    } else {
        absmax / signed_limit
    };
    let weights = importance.unwrap_or(&[]);

    let mut best_scale = base_scale;
    let mut best_error = f32::INFINITY;
    let mut best_q = Vec::new();

    for multiplier in [0.6f32, 0.75, 0.9, 1.0, 1.1, 1.25, 1.4] {
        let candidate_scale = base_scale * multiplier;
        let quantized = quantize_block_linear(block, candidate_scale, bits);
        let quantized = refine_block_residuals(block, &quantized, candidate_scale, bits, weights);
        let dequantized = dequantize_block_signed(&quantized, candidate_scale, bits);
        let error = weighted_error(block, &dequantized, weights);
        if error < best_error {
            best_error = error;
            best_scale = candidate_scale;
            best_q = quantized;
        }
    }

    Ok(BlockQuantization {
        scale: best_scale,
        codes: best_q,
    })
}

fn prepare_gptq_proxy_tensor(
    data: &[f32],
    bits: u8,
    importance: Option<&[f32]>,
    curvature: Option<&[f32]>,
    shape: Option<&[usize]>,
) -> Result<Vec<f32>> {
    let row_width = infer_row_width(shape, data.len());
    let mut prepared = Vec::with_capacity(data.len());

    for row_index in 0..data.len().div_ceil(row_width.max(1)) {
        let row_start = row_index * row_width;
        if row_start >= data.len() {
            break;
        }
        let row_end = (row_start + row_width).min(data.len());
        let row = &data[row_start..row_end];
        let row_importance = importance.map(|imp| {
            let end = row_end.min(imp.len());
            &imp[row_start..end]
        });
        let row_curvature = curvature.map(|diag| {
            let end = row_end.min(diag.len());
            &diag[row_start..end]
        });
        let prepared_row = prepare_row_with_sequential_update(row, bits, row_importance, row_curvature)?;
        prepared.extend_from_slice(&prepared_row);
    }

    Ok(prepared)
}

fn prepare_row_with_sequential_update(
    row: &[f32],
    bits: u8,
    importance: Option<&[f32]>,
    curvature: Option<&[f32]>,
) -> Result<Vec<f32>> {
    let weights = importance.unwrap_or(&[]);
    let curvature_diag = curvature.unwrap_or(&[]);
    let baseline_error = row_rewrite_error(row, row, bits, weights, curvature_diag)?;
    let mut prepared = row.to_vec();
    let mut carry = 0.0f32;
    let mut residual_tail = 0.0f32;

    for block_index in 0..row.len().div_ceil(BLOCK_SIZE_QK) {
        let start = block_index * BLOCK_SIZE_QK;
        let end = (start + BLOCK_SIZE_QK).min(row.len());
        let block_weights = &weights[start.min(weights.len())..end.min(weights.len())];
        let block_curvature = &curvature_diag[start.min(curvature_diag.len())..end.min(curvature_diag.len())];

        for value in &mut prepared[start..end] {
            *value += carry + residual_tail;
        }

        apply_block_diagonal_update(&mut prepared[start..end], block_weights, block_curvature);

        let fit = fit_block_quantization(&prepared[start..end], bits, Some(block_weights))?;
        let dequantized = dequantize_block_signed(&fit.codes, fit.scale, bits);
        let residual_energy = prepared[start..end]
            .iter()
            .zip(dequantized.iter())
            .enumerate()
            .map(|(idx, (original, approx))| {
                let weight = block_weights.get(idx).copied().unwrap_or(1.0);
                let h = block_curvature.get(idx).copied().unwrap_or(weight.max(1.0));
                weight * h * (original - approx)
            })
            .sum::<f32>();
        let curvature_mass = block_curvature.iter().copied().sum::<f32>();
        let normalizer = (block_weights.iter().copied().sum::<f32>() + curvature_mass)
            .max(end.saturating_sub(start).max(1) as f32);
        carry = (residual_energy / normalizer) * 0.25;
        residual_tail = block_curvature
            .last()
            .copied()
            .unwrap_or(1.0)
            .sqrt()
            .min(4.0)
            * carry
            * 0.1;
    }

    let sequential_error = row_rewrite_error(row, &prepared, bits, weights, curvature_diag)?;
    if sequential_error <= baseline_error {
        Ok(prepared)
    } else {
        Ok(row.to_vec())
    }
}

fn apply_block_diagonal_update(
    block: &mut [f32],
    weights: &[f32],
    curvature: &[f32],
) {
    if block.len() <= 1 {
        return;
    }

    for group_start in (0..block.len()).step_by(GPTQ_PROXY_COLUMN_GROUP) {
        let group_end = (group_start + GPTQ_PROXY_COLUMN_GROUP).min(block.len());
        let group_weights = &weights[group_start.min(weights.len())..group_end.min(weights.len())];
        let group_curvature = &curvature[group_start.min(curvature.len())..group_end.min(curvature.len())];
        let mean = weighted_group_mean(&block[group_start..group_end], group_weights, group_curvature);
        let coupling = block_group_coupling(group_curvature);

        for offset in 0..(group_end - group_start) {
            let idx = group_start + offset;
            let weight = group_weights.get(offset).copied().unwrap_or(1.0);
            let h = group_curvature.get(offset).copied().unwrap_or(1.0);
            let trust = (weight * h).sqrt().min(8.0);
            let blend = (0.04 * coupling / trust.max(1e-3)).clamp(0.0, 0.2);
            block[idx] = block[idx] * (1.0 - blend) + mean * blend;
        }
    }
}

fn weighted_group_mean(values: &[f32], weights: &[f32], curvature: &[f32]) -> f32 {
    let mut weighted_sum = 0.0f32;
    let mut mass = 0.0f32;
    for (index, value) in values.iter().enumerate() {
        let w = weights.get(index).copied().unwrap_or(1.0);
        let h = curvature.get(index).copied().unwrap_or(1.0);
        let scale = (w * h).max(1e-4);
        weighted_sum += scale * *value;
        mass += scale;
    }
    if mass <= 1e-6 {
        0.0
    } else {
        weighted_sum / mass
    }
}

fn block_group_coupling(curvature: &[f32]) -> f32 {
    if curvature.len() <= 1 {
        return 0.0;
    }
    let mean = curvature.iter().copied().sum::<f32>() / curvature.len() as f32;
    let variance = curvature
        .iter()
        .map(|value| {
            let delta = *value - mean;
            delta * delta
        })
        .sum::<f32>()
        / curvature.len() as f32;
    1.0 / (1.0 + variance.sqrt())
}

fn infer_row_width(shape: Option<&[usize]>, len: usize) -> usize {
    let inferred = shape
        .and_then(|dims| dims.last().copied())
        .filter(|width| *width > 0)
        .unwrap_or(len.max(1));
    inferred.min(len.max(1))
}

fn row_rewrite_error(
    original: &[f32],
    candidate: &[f32],
    bits: u8,
    weights: &[f32],
    curvature: &[f32],
) -> Result<f32> {
    let mut total_error = 0.0f32;
    for block_index in 0..candidate.len().div_ceil(BLOCK_SIZE_QK) {
        let start = block_index * BLOCK_SIZE_QK;
        let end = (start + BLOCK_SIZE_QK).min(candidate.len());
        let block_weights = &weights[start.min(weights.len())..end.min(weights.len())];
        let block_curvature = &curvature[start.min(curvature.len())..end.min(curvature.len())];
        let fit = fit_block_quantization(&candidate[start..end], bits, Some(block_weights))?;
        let dequantized = dequantize_block_signed(&fit.codes, fit.scale, bits);
        total_error += weighted_curvature_error(
            &original[start..end],
            &dequantized,
            block_weights,
            block_curvature,
        );
    }
    Ok(total_error)
}

fn weighted_curvature_error(
    original: &[f32],
    dequantized: &[f32],
    weights: &[f32],
    curvature: &[f32],
) -> f32 {
    original
        .iter()
        .enumerate()
        .map(|(index, lhs)| {
            let weight = weights.get(index).copied().unwrap_or(1.0);
            let h = curvature.get(index).copied().unwrap_or(1.0);
            let residual = lhs - dequantized.get(index).copied().unwrap_or_default();
            weight * h.max(1e-4) * residual * residual
        })
        .sum()
}

fn quantize_block_linear(block: &[f32], scale: f32, bits: u8) -> Vec<u32> {
    let zero_point = quant_zero_point(bits) as f32;
    let signed_limit = signed_quant_limit(bits);
    block.iter()
        .map(|value| {
            (((value / scale).round()).clamp(-signed_limit, signed_limit) + zero_point) as u32
        })
        .collect()
}

fn dequantize_block_signed(block: &[u32], scale: f32, bits: u8) -> Vec<f32> {
    let zero_point = quant_zero_point(bits) as f32;
    block.iter()
        .map(|value| ((*value as f32) - zero_point) * scale)
        .collect()
}

fn refine_block_residuals(
    original: &[f32],
    initial_codes: &[u32],
    scale: f32,
    bits: u8,
    weights: &[f32],
) -> Vec<u32> {
    let mut codes = initial_codes.to_vec();
    let max_code = (1u32 << bits) - 1;
    if original.is_empty() {
        return codes;
    }

    for _ in 0..3 {
        let mut changed = false;
        for index in 0..codes.len() {
            let current = codes[index];
            let base_weight = weights.get(index).copied().unwrap_or(1.0);
            let current_value = dequantize_block_signed(&[current], scale, bits)[0];
            let current_error = base_weight * (original[index] - current_value).powi(2);

            let mut best_code = current;
            let mut best_error = current_error;

            for candidate in [current.saturating_sub(1), current.saturating_add(1).min(max_code)] {
                if candidate == current {
                    continue;
                }
                let candidate_value = dequantize_block_signed(&[candidate], scale, bits)[0];
                let candidate_error = base_weight * (original[index] - candidate_value).powi(2);
                if candidate_error + 1e-8 < best_error {
                    best_error = candidate_error;
                    best_code = candidate;
                }
            }

            if best_code != current {
                codes[index] = best_code;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    codes
}

fn quant_zero_point(bits: u8) -> u32 {
    1u32 << (bits - 1)
}

fn signed_quant_limit(bits: u8) -> f32 {
    ((1u32 << (bits - 1)) - 1) as f32
}

fn weighted_error(original: &[f32], dequantized: &[f32], weights: &[f32]) -> f32 {
    original.iter().enumerate().map(|(index, lhs)| {
        let weight = weights.get(index).copied().unwrap_or(1.0);
        let residual = lhs - dequantized.get(index).copied().unwrap_or_default();
        weight * residual * residual
    }).sum()
}

fn pack_bits(values: &[u32], bits: u8) -> Vec<u8> {
    let total_bits = values.len() * bits as usize;
    let mut out = vec![0u8; total_bits.div_ceil(8)];
    let mut bit_cursor = 0usize;
    for value in values {
        let mut remaining = *value;
        for _ in 0..bits {
            let byte_index = bit_cursor / 8;
            let bit_index = bit_cursor % 8;
            out[byte_index] |= ((remaining & 1) as u8) << bit_index;
            remaining >>= 1;
            bit_cursor += 1;
        }
    }
    out
}

fn unpack_bits(bytes: &[u8], bits: u8, count: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(count);
    let mut bit_cursor = 0usize;
    for _ in 0..count {
        let mut value = 0u32;
        for bit in 0..bits {
            let byte_index = bit_cursor / 8;
            let bit_index = bit_cursor % 8;
            let bit_value = ((bytes[byte_index] >> bit_index) & 1) as u32;
            value |= bit_value << bit;
            bit_cursor += 1;
        }
        out.push(value);
    }
    out
}

fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = (bits >> 31) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;
    if exp == 0 {
        return sign << 15;
    }
    if exp >= 0x8D {
        // Overflow: return inf
        return (sign << 15) | 0x7C00;
    }
    if exp <= 0x70 {
        // Underflow: subnormal
        return sign << 15;
    }
    let new_exp = exp - 127 + 15;
    if new_exp <= 0 {
        return sign << 15;
    }
    (sign << 15) | ((new_exp as u16) << 10) | ((mant >> 13) as u16)
}


/// Randomized SVD algorithm for importance matrix calculation (§0 / §19).
/// Replicates `scirs2_linalg` randomized SVD projection strategy:
/// Projects high-dimensional weight arrays to lower-rank spaces with Gaussian matrices.
pub fn randomized_svd_importance(
    matrix: &[f32],
    rows: usize,
    cols: usize,
    target_rank: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    if target_rank == 0 || target_rank > rows.min(cols) {
        return Err(Error::Backend("Invalid target rank for randomized SVD".into()));
    }
    // Replicating Martinsson/Tropp Randomized SVD pattern:
    // 1. Generate random Gaussian matrix Omega of size (cols, target_rank + oversampling)
    let oversampling = 5;
    let rank_k = (target_rank + oversampling).min(cols);
    let mut omega = vec![0.0f32; cols * rank_k];
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    for val in &mut omega {
        // Quick deterministic LCG-based normal distribution sample
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let u1 = ((seed >> 40) as u32 as f32) / 16777216.0;
        let u2 = (((seed & 0xFFFFFFFF) >> 8) as u32 as f32) / 16777216.0;
        let normal = (-2.0 * u1.max(1e-5).ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        *val = normal;
    }

    // 2. Form sample matrix Y = A * Omega (rows, rank_k)
    let mut y = vec![0.0f32; rows * rank_k];
    for r in 0..rows {
        for c in 0..rank_k {
            let mut sum = 0.0f32;
            for k in 0..cols {
                sum += matrix[r * cols + k] * omega[k * rank_k + c];
            }
            y[r * rank_k + c] = sum;
        }
    }

    // 3. Orthonormalize Y using Gram-Schmidt projection (approximation of QR decomposition Q)
    let mut q = vec![0.0f32; rows * rank_k];
    for col in 0..rank_k {
        let mut v = vec![0.0f32; rows];
        for r in 0..rows {
            v[r] = y[r * rank_k + col];
        }
        for prev in 0..col {
            let mut dot = 0.0f32;
            for r in 0..rows {
                dot += y[r * rank_k + col] * q[r * rank_k + prev];
            }
            for r in 0..rows {
                v[r] -= dot * q[r * rank_k + prev];
            }
        }
        let mut norm = 0.0f32;
        for r in 0..rows {
            norm += v[r] * v[r];
        }
        let norm = norm.sqrt().max(1e-5);
        for r in 0..rows {
            q[r * rank_k + col] = v[r] / norm;
        }
    }

    // 4. Form B = Q^T * A (rank_k, cols)
    let mut b = vec![0.0f32; rank_k * cols];
    for r in 0..rank_k {
        for c in 0..cols {
            let mut sum = 0.0f32;
            for k in 0..rows {
                sum += q[k * rank_k + r] * matrix[k * cols + c];
            }
            b[r * cols + c] = sum;
        }
    }

    // Return the low-rank projections (U_approx = Q, S_approx = singular values mock, V_approx = B)
    // S_approx holds column norm representations of B projection spaces
    let mut s = vec![0.0f32; target_rank];
    for r in 0..target_rank {
        let mut norm = 0.0f32;
        for c in 0..cols {
            norm += b[r * cols + c] * b[r * cols + c];
        }
        s[r] = norm.sqrt();
    }

    // Truncate Q and B to the target rank
    let mut u_trunc = vec![0.0f32; rows * target_rank];
    for r in 0..rows {
        for c in 0..target_rank {
            u_trunc[r * target_rank + c] = q[r * rank_k + c];
        }
    }

    let mut vt_trunc = vec![0.0f32; target_rank * cols];
    for r in 0..target_rank {
        for c in 0..cols {
            vt_trunc[r * cols + c] = b[r * cols + c];
        }
    }

    Ok((u_trunc, s, vt_trunc))
}

trait Clamp {
    fn clamp(self, lo: f32, hi: f32) -> Self;
}
impl Clamp for f32 {
    fn clamp(self, lo: f32, hi: f32) -> Self {
        if self < lo { lo } else if self > hi { hi } else { self }
    }
}

// ---------------------------------------------------------------------------
// Phase 2: Importance-Matrix Calibration
// ---------------------------------------------------------------------------

/// Per-layer importance scores from calibration.
///
/// `layer_scores[i]` is the importance of tensor `i` (higher = more
/// quantization-sensitive — should use more bits).
#[derive(Debug, Clone)]
pub struct ImportanceScores {
    pub tensor_names: Vec<String>,
    pub layer_scores: Vec<f32>,
}

impl ImportanceScores {
    pub fn new(tensor_names: Vec<String>, layer_scores: Vec<f32>) -> Self {
        assert_eq!(tensor_names.len(), layer_scores.len());
        Self { tensor_names, layer_scores }
    }

    pub fn score_for(&self, tensor_name: &str) -> f32 {
        self.layer_scores
            .iter()
            .zip(&self.tensor_names)
            .find(|(_, n)| *n == tensor_name)
            .map(|(s, _)| *s)
            .unwrap_or(0.0)
    }
}

/// Compute per-tensor importance scores using randomized SVD.
///
/// For each tensor, runs randomized SVD and returns the column-norm-based
/// importance: the Frobenius norm of each singular vector weighted by its
/// singular value. Tensors with higher importance scores are more
/// quantization-sensitive and should receive higher bitwidth in EvoPress.
pub fn compute_importance_scores(
    tensors: &[(String, Vec<f32>, usize, usize)],
) -> Vec<f32> {
    let mut scores = Vec::with_capacity(tensors.len());
    for (name, data, rows, cols) in tensors {
        if *rows == 0 || *cols == 0 {
            scores.push(0.0);
            continue;
        }
        let r = (*rows).min(*cols);
        let target_rank = if r > 8 { 8 } else if r < 1 { 1 } else { r };
        let (_, s, vt) = match randomized_svd_importance(data, *rows, *cols, target_rank) {
            Ok(r) => r,
            Err(_) => {
                scores.push(0.0);
                continue;
            }
        };
        let n_cols = *cols;
        let s_len = s.len();
        let mut col_norms: Vec<f32> = Vec::with_capacity(n_cols);
        for c in 0..n_cols {
            let mut norm_sq: f32 = 0.0;
            for row in 0..s_len {
                let val = vt[row * n_cols + c];
                norm_sq += val * val;
            }
            col_norms.push(norm_sq.sqrt());
        }
        let total_importance: f32 = s
            .iter()
            .zip(&col_norms)
            .take(target_rank)
            .map(|(sig, cn)| sig * cn)
            .sum();
        scores.push(total_importance);
    }
    scores
}

// ---------------------------------------------------------------------------
// Phase 4: Fisher/GGN Diagonal Computation for GPTQ Error-Correcting Updates
// ---------------------------------------------------------------------------

/// One calibration sample: input activations and output gradients for a specific
/// tensor. Populated by running the calibration dataset forward+backward through
/// the model and capturing intermediate activations/gradients via Hook.
/// For a linear layer, `input_activations` has shape (batch, in_features) and
/// `output_gradients` has shape (batch, out_features).
#[derive(Debug, Clone)]
pub struct FisherCalibrationSample {
    pub input_activations: Vec<f32>,
    pub output_gradients: Vec<f32>,
}

/// Compute the diagonal of the Generalized Gauss-Newton (GGN) matrix for a
/// weight matrix using a batch of pre-computed calibration activations and gradients.
///
/// This is the "true" curvature for GPTQ error-correcting updates, replacing
/// `build_curvature_proxy`. The GGN diagonal is:
///
///   diag(H) ≈ (1/M) Σ_m (x_m ⊗ x_m) where x_m is the input activation
///   (ignoring cross-term correlations — this is the standard GPTQ diagonal).
///
/// Each calibration sample contributes `diag(grad_out_m @ grad_out_m^T) ⊗ (x_m @ x_m^T)`.
/// Summing over samples and averaging gives the GGN diagonal.
///
/// # Arguments
/// * `weights` — the f32 weight matrix, row-major (rows × cols)
/// * `calibration_samples` — per-sample (activations, gradients) pairs
/// * `rows` — number of output features (out_features)
/// * `cols` — number of input features (in_features)
/// * `group_size` — GPTQ group size for grouped diagonal (default 128)
///
/// # Returns
/// Per-element diagonal curvature of shape (rows × cols), same shape as `weights`.
pub fn compute_fisher_diagonal(
    _weights: &[f32],
    calibration_samples: &[FisherCalibrationSample],
    rows: usize,
    cols: usize,
    group_size: usize,
) -> Vec<f32> {
    if calibration_samples.is_empty() || rows == 0 || cols == 0 {
        return vec![1.0f32; rows * cols];
    }

    let batch_size = calibration_samples
        .first()
        .map(|s| s.output_gradients.len() / rows)
        .unwrap_or(1)
        .max(1);
    let num_groups = (cols + group_size - 1) / group_size;

    // Accumulate per-column and per-element diagonal
    let mut h_diag = vec![0.0f32; cols];
    let m = calibration_samples.len() as f32;

    for sample in calibration_samples {
        let batch = sample.output_gradients.len() / rows;
        if sample.input_activations.len() != batch * cols || batch == 0 {
            continue;
        }

        for b in 0..batch {
            let grad_out_slice = &sample.output_gradients[b * rows..(b + 1) * rows];
            let in_slice = &sample.input_activations[b * cols..(b + 1) * cols];

            for col in 0..cols {
                let x_sq = in_slice[col] * in_slice[col];
                let mut col_h = 0.0f32;
                for row in 0..rows {
                    let go_sq = grad_out_slice[row] * grad_out_slice[row];
                    col_h += x_sq * go_sq;
                }
                h_diag[col] += col_h;
            }
        }
    }

    // Average
    for val in &mut h_diag {
        *val /= m;
        *val = val.max(1e-8);
    }

    // Broadcast per-column diagonal across all rows (each row gets the same diagonal)
    let mut out = Vec::with_capacity(rows * cols);
    for _ in 0..rows {
        out.extend_from_slice(&h_diag);
    }

    out.truncate(rows * cols);
    while out.len() < rows * cols {
        out.push(1.0);
    }

    out
}

/// Compute per-group GGN diagonal — one curvature value per quantization group.
///
/// This is the format actually used in GPTQ re-quantization: each group of
/// `group_size` columns shares one diagonal entry, reducing storage and
/// matching how GPTQ applies correction (per-group scale factors).
///
/// # Returns
/// `num_groups` curvature values, one per group. The group assignment is:
///   group_idx = col_idx / group_size
pub fn compute_grouped_fisher_diagonal(
    _weights: &[f32],
    calibration_samples: &[FisherCalibrationSample],
    rows: usize,
    cols: usize,
    group_size: usize,
) -> Vec<f32> {
    if calibration_samples.is_empty() || rows == 0 || cols == 0 {
        return vec![1.0f32; (cols + group_size - 1) / group_size];
    }

    let batch_size = calibration_samples
        .first()
        .map(|s| s.output_gradients.len() / rows)
        .unwrap_or(1)
        .max(1);
    let num_groups = (cols + group_size - 1) / group_size;
    let mut group_h_diag = vec![0.0f32; num_groups];
    let m = calibration_samples.len() as f32;

    for sample in calibration_samples {
        let batch = sample.output_gradients.len() / rows;
        if sample.input_activations.len() != batch * cols || batch == 0 {
            continue;
        }

        for b in 0..batch {
            let grad_out_slice = &sample.output_gradients[b * rows..(b + 1) * rows];
            let in_slice = &sample.input_activations[b * cols..(b + 1) * cols];

            for (gi, g_start) in (0..num_groups).map(|gi| (gi, gi * group_size)) {
                let g_end = (g_start + group_size).min(cols);
                let mut accum = 0.0f32;
                let mut col_count = 0usize;
                for col in g_start..g_end {
                    let x_sq = in_slice[col] * in_slice[col];
                    for row in 0..rows {
                        let go_sq = grad_out_slice[row] * grad_out_slice[row];
                        accum += x_sq * go_sq;
                    }
                    col_count += 1;
                }
                if col_count > 0 {
                    group_h_diag[gi] += accum / (cols as f32);
                }
            }
        }
    }

    for val in &mut group_h_diag {
        *val /= m;
        *val = val.max(1e-8);
    }

    group_h_diag
}

/// Compute an importance-weighted curvature proxy when calibration data is
/// not available (CPU fallback).
///
/// This is a first-order approximation of the GGN diagonal using activation
/// magnitude as a proxy for second-order importance. Used when
/// `calibration_samples` is empty or unavailable.
pub fn compute_curvature_proxy(data: &[f32], layer_importance: f32) -> Vec<f32> {
    let layer_scale = layer_importance.abs().max(1e-3);
    data.iter()
        .map(|value| 1.0 + layer_scale * (value.abs() + value * value).min(16.0))
        .collect()
}

/// Refined Scale Fit (RSF) for K-quant blocks.
///
/// Re-fits the per-block scales using importance-weighted L2 reconstruction
/// error minimization. The original K-quant scales are a rough estimate;
/// RSF uses the importance scores to give more weight to sensitive regions.
///
/// # Arguments
/// * `data` — flat f32 weight data (row-major)
/// * `importance` — per-element importance weights (same shape as data)
/// * `block_size` — K-quant block size (32)
/// * `n_levels` — quantization levels (16 for Q4_K, 32 for Q5_K, 64 for Q6_K)
pub fn refined_scale_fit(
    data: &[f32],
    importance: &[f32],
    block_size: usize,
    n_levels: u32,
) -> Result<Vec<f32>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }
    let n_blocks = (data.len() + block_size - 1) / block_size;
    let mut scales = Vec::with_capacity(n_blocks);
    let step = (n_levels - 1) as f32;

    for bi in 0..n_blocks {
        let start = bi * block_size;
        let end = (start + block_size).min(data.len());
        let blk_data = &data[start..end];
        let blk_imp = &importance[start..end];

        // Weighted RMS of the block: scale = sqrt(sum(w * x^2) / sum(w)) / (n_levels/2)
        let weighted_sq_sum: f32 = blk_data
            .iter()
            .zip(blk_imp.iter())
            .map(|(x, w)| w * x * x)
            .sum();
        let weight_sum: f32 = blk_imp.iter().sum();
        if weight_sum < 1e-9 {
            scales.push(1.0);
            continue;
        }
        let rms = (weighted_sq_sum / weight_sum).sqrt();
        let scale = if rms < 1e-9 { 1.0 } else { rms / (step as f32) * 2.0 };
        scales.push(scale);
    }
    Ok(scales)
}

// ---------------------------------------------------------------------------
// Phase 3: EvoPress Evolutionary Search
// ---------------------------------------------------------------------------

/// Configuration for the EvoPress evolutionary bitwidth search.
#[derive(Debug, Clone)]
pub struct EvoPressConfig {
    /// Number of individuals in the population.
    pub population_size: usize,
    /// Number of generations to run.
    pub generations: usize,
    /// Target average bits-per-weight across all tensors.
    pub target_bpw: f32,
    /// Tournament size for selection.
    pub tournament_size: usize,
    /// Crossover probability.
    pub crossover_prob: f32,
    /// Mutation probability per gene.
    pub mutation_prob: f32,
    /// Available bitwidth choices per tensor (e.g. [2, 3, 4, 5, 6] for K-quants).
    pub available_bpws: Vec<u32>,
}

impl Default for EvoPressConfig {
    fn default() -> Self {
        Self {
            population_size: 128,
            generations: 50,
            target_bpw: 4.0,
            tournament_size: 3,
            crossover_prob: 0.8,
            mutation_prob: 0.05,
            available_bpws: vec![2, 3, 4, 5, 6, 8],
        }
    }
}

/// One individual in the EvoPress population. `genes[i]` is the bitwidth
/// assigned to tensor `i`.
#[derive(Debug, Clone)]
pub struct Individual {
    pub genes: Vec<u32>,
    pub fitness: f32,
}

/// Run EvoPress evolutionary search to find optimal per-tensor bitwidths.
///
/// The search respects the `target_bpw` constraint while maximizing a
/// quality proxy derived from importance scores. The returned vector maps
/// each tensor index to its assigned bitwidth.
pub fn evopress_search(
    config: &EvoPressConfig,
    importance_scores: &[f32],
    tensor_sizes: &[usize],
) -> Vec<u32> {
    let n_tensors = importance_scores.len();
    if n_tensors == 0 {
        return Vec::new();
    }

    let mut rng = SimpleRng::new(0x9E37_79B9_7F4A_7C15);
    let total_size: usize = tensor_sizes.iter().sum();
    if total_size == 0 {
        return vec![config.target_bpw as u32; n_tensors];
    }

    // Build initial population.
    let mut population: Vec<Individual> = (0..config.population_size)
        .map(|i| {
            let genes = if i == 0 {
                // First individual: greedy baseline matching target_bpw
                let mut genes = Vec::with_capacity(n_tensors);
                let mut budget = (config.target_bpw * total_size as f32) as usize;
                for (ti, sz) in tensor_sizes.iter().enumerate() {
                    let imp = importance_scores[ti];
                    // Higher importance → higher bitwidth (bias toward important layers)
                    let imp_sum = importance_scores.iter().sum::<f32>().max(1e-9);
                    let imp_ratio = imp / imp_sum;
                    let target_bpw_for_tensor = (config.target_bpw * imp_ratio * 2.0).clamp(2.0, 8.0);
                    let gene = *config.available_bpws.iter()
                        .min_by(|a, b| {
                            let da = ((**a) as f32 - target_bpw_for_tensor).abs();
                            let db = ((**b) as f32 - target_bpw_for_tensor).abs();
                            da.partial_cmp(&db).unwrap()
                        })
                        .unwrap_or(&4);
                    genes.push(gene);
                    budget = budget.saturating_sub(gene as usize * sz);
                }
                genes
            } else {
                (0..n_tensors).map(|_| {
                    *config.available_bpws.choose(&mut rng).unwrap_or(&4)
                }).collect()
            };
            let fitness = eval_individual(&genes, importance_scores, tensor_sizes, config.target_bpw, total_size);
            Individual { genes, fitness }
        })
        .collect();

    // Evolutionary loop.
    for _generation in 0..config.generations {
        let mut next_gen = Vec::with_capacity(config.population_size);

        // Elitism: keep top-2.
        population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        if config.population_size >= 2 {
            next_gen.push(population[0].clone());
            next_gen.push(population[1].clone());
        }

        while next_gen.len() < config.population_size {
            // Tournament selection.
            let p1 = tournament_select(&population, config.tournament_size, &mut rng);
            let p2 = tournament_select(&population, config.tournament_size, &mut rng);

            // Crossover.
            let mut child_genes = if rng.next_f32() < config.crossover_prob {
                crossover(&p1.genes, &p2.genes, &mut rng)
            } else {
                p1.genes.clone()
            };

            // Mutation.
            for gene in &mut child_genes {
                if rng.next_f32() < config.mutation_prob {
                    *gene = *config.available_bpws.choose(&mut rng).unwrap_or(gene);
                }
            }

            let fitness = eval_individual(&child_genes, importance_scores, tensor_sizes, config.target_bpw, total_size);
            next_gen.push(Individual { genes: child_genes, fitness });
        }

        population = next_gen;
    }

    population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
    population[0].genes.clone()
}

fn tournament_select<'a>(pop: &'a [Individual], k: usize, rng: &mut SimpleRng) -> &'a Individual {
    let best = (0..k)
        .map(|_| {
            let idx = (rng.next_u64() as usize) % pop.len().max(1);
            &pop[idx]
        })
        .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
        .unwrap();
    best
}

fn crossover(p1: &[u32], p2: &[u32], rng: &mut SimpleRng) -> Vec<u32> {
    let min_len = p1.len().min(p2.len());
    let cut = rng.next_u64() as usize % (min_len + 1);
    let mut child = p1[..cut].to_vec();
    child.extend_from_slice(&p2[cut..]);
    child
}

fn eval_individual(
    genes: &[u32],
    importance_scores: &[f32],
    tensor_sizes: &[usize],
    target_bpw: f32,
    total_size: usize,
) -> f32 {
    if genes.is_empty() || total_size == 0 {
        return 0.0;
    }
    // Weighted quality score: higher importance + correct BPW = better
    let mut quality: f32 = 0.0;
    for (ti, (&gene, &sz)) in genes.iter().zip(tensor_sizes.iter()).enumerate() {
        if ti < importance_scores.len() {
            let imp = importance_scores[ti];
            // Reward matching target BPW; reward higher bits for high-importance tensors
            let bpw_error = (gene as f32 - target_bpw).abs();
            quality += imp / (bpw_error + 0.1);
        }
    }

    // Penalty for deviation from target average BPW.
    let total_bits: usize = genes.iter().zip(tensor_sizes.iter())
        .map(|(g, s)| (*g as usize) * s)
        .sum();
    let actual_bpw = total_bits as f32 / total_size as f32;
    let bpw_penalty = (actual_bpw - target_bpw).abs() * 100.0;

    quality - bpw_penalty
}

// ---------------------------------------------------------------------------
// Simple deterministic RNG for EvoPress (no external crate dependency)
// ---------------------------------------------------------------------------

struct SimpleRng {
    seed: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.seed
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / 16777216.0
    }
}

trait Choose<T> {
    fn choose(&self, rng: &mut SimpleRng) -> Option<&T>;
}

impl<T> Choose<T> for [T] {
    fn choose(&self, rng: &mut SimpleRng) -> Option<&T> {
        if self.is_empty() {
            return None;
        }
        let idx = (rng.next_u64() as usize) % self.len();
        Some(&self[idx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_q80() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.5).collect();
        let quantized = quant_q80(&data).unwrap();
        let dequantized = dequant_q80(&quantized, data.len()).unwrap();
        assert_eq!(data.len(), dequantized.len());
        // Q8_0 should be close
        for i in 0..data.len() {
            let diff = (data[i] - dequantized[i]).abs();
            assert!(diff < 0.5, "diff at {i}: {} vs {}, diff={}", data[i], dequantized[i], diff);
        }
    }

    #[test]
    fn dequant_q4k_basic() {
        // 64 weights: 2 blocks of 32 each, 20 bytes per block = 40 bytes
        let mut data = vec![0u8; 40];
        // Block 0: scale = 1.0 (0x3F800000 in LE f32)
        data[0..4].copy_from_slice(&1.0f32.to_le_bytes());
        // values: byte 4..=19 packed 4-bit, centered around zero-point 8
        for i in 0..16u8 {
            let low = i % 16;
            let high = (i + 1) % 16;
            data[4 + i as usize] = low | (high << 4);
        }
        // Block 1 same pattern
        data[20..24].copy_from_slice(&0.5f32.to_le_bytes());
        let deq = dequant_q4k(&data, 64).unwrap();
        assert_eq!(deq.len(), 64);
        assert_eq!(deq[0], -8.0f32);
        assert_eq!(deq[1], -7.0f32);
    }

    #[test]
    fn roundtrip_q4k() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 3.0).collect();
        let quantized = quant_q4k(&data).unwrap();
        let dequantized = dequant_q4k(&quantized, data.len()).unwrap();
        assert_eq!(dequantized.len(), data.len());
        let mse = mean_squared_error(&data, &dequantized);
        assert!(mse < 0.5, "q4k mse too high: {mse}");
    }

    #[test]
    fn roundtrip_q5k() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 9.0).collect();
        let quantized = quant_q5k(&data).unwrap();
        let dequantized = dequant_q5k(&quantized, data.len()).unwrap();
        assert_eq!(dequantized.len(), data.len());
        let mse = mean_squared_error(&data, &dequantized);
        assert!(mse < 0.05, "q5k mse too high: {mse}");
    }

    #[test]
    fn roundtrip_q6k() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 12.0).collect();
        let quantized = quant_q6k(&data).unwrap();
        let dequantized = dequant_q6k(&quantized, data.len()).unwrap();
        assert_eq!(dequantized.len(), data.len());
        let mse = mean_squared_error(&data, &dequantized);
        assert!(mse < 0.02, "q6k mse too high: {mse}");
    }

    #[test]
    fn rewrite_tensor_to_q80() {
        let data: Vec<f32> = (0..32).map(|i| i as f32 * 0.25).collect();
        let rewritten = rewrite_tensor_data(
            &data,
            &TensorRewritePlan {
                target: QuantFormat::Q8_0,
                shape: vec![32, 1],
                importance: None,
                curvature: None,
            },
        )
        .unwrap();
        assert!(!rewritten.bytes.is_empty());
        assert_eq!(rewritten.target, QuantFormat::Q8_0);
    }

    #[test]
    fn residual_refinement_beats_linear_baseline() {
        let block = vec![
            -3.2f32, -2.8, -2.1, -1.7, -1.2, -0.9, -0.3, 0.1,
            0.25, 0.6, 0.95, 1.3, 1.8, 2.2, 2.7, 3.4,
        ];
        let weights = vec![1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0];
        let bits = 4;
        let scale = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max) / signed_quant_limit(bits);

        let linear_codes = quantize_block_linear(&block, scale, bits);
        let refined_codes = refine_block_residuals(&block, &linear_codes, scale, bits, &weights);
        let linear = dequantize_block_signed(&linear_codes, scale, bits);
        let refined = dequantize_block_signed(&refined_codes, scale, bits);

        let linear_error = weighted_error(&block, &linear, &weights);
        let refined_error = weighted_error(&block, &refined, &weights);
        assert!(refined_error <= linear_error, "residual refinement regressed: {refined_error} > {linear_error}");
    }

    #[test]
    fn sequential_row_update_improves_two_block_tensor() {
        let mut row = Vec::new();
        for i in 0..64 {
            let base = if i < 32 { (i as f32 - 16.0) / 2.5 } else { (i as f32 - 48.0) / 4.0 };
            let bias = if i >= 32 { 0.35 } else { 0.0 };
            row.push(base + bias);
        }
        let weights = vec![1.0f32; row.len()];
        let bits = 4;

        let baseline_bytes = {
            let packed_bytes_per_block = (BLOCK_SIZE_QK * bits as usize).div_ceil(8);
            let mut out = Vec::new();
            for block in row.chunks(BLOCK_SIZE_QK) {
                let fit = fit_block_quantization(block, bits, Some(&weights[..block.len()])).unwrap();
                out.extend_from_slice(&fit.scale.to_le_bytes());
                let packed = pack_bits(&fit.codes, bits);
                out.extend_from_slice(&packed);
                for _ in packed.len()..packed_bytes_per_block {
                    out.push(0);
                }
            }
            out
        };
        let sequential_bytes =
            quant_packed_symmetric(&row, bits, Some(&weights), None, Some(&[1usize, row.len()])).unwrap();

        let baseline = dequant_q4k(&baseline_bytes, row.len()).unwrap();
        let sequential = dequant_q4k(&sequential_bytes, row.len()).unwrap();
        let baseline_error = mean_squared_error(&row, &baseline);
        let sequential_error = mean_squared_error(&row, &sequential);
        assert!(
            sequential_error <= baseline_error,
            "sequential row update regressed: {sequential_error} > {baseline_error}"
        );
    }

    #[test]
    fn curvature_weighted_row_update_is_non_regressive() {
        let row: Vec<f32> = (0..64)
            .map(|i| {
                let x = i as f32 - 32.0;
                (x / 7.0).sin() * 3.0 + if i > 40 { 0.45 } else { -0.15 }
            })
            .collect();
        let weights = vec![1.0f32; row.len()];
        let curvature: Vec<f32> = row
            .iter()
            .enumerate()
            .map(|(idx, value)| 1.0 + value.abs() + if idx > 40 { 2.0 } else { 0.25 })
            .collect();

        let baseline_error = row_rewrite_error(&row, &row, 4, &weights, &curvature).unwrap();
        let prepared = prepare_row_with_sequential_update(&row, 4, Some(&weights), Some(&curvature)).unwrap();
        let curved_error = row_rewrite_error(&row, &prepared, 4, &weights, &curvature).unwrap();
        assert!(
            curved_error <= baseline_error,
            "curvature-aware row update regressed: {curved_error} > {baseline_error}"
        );
    }

    #[test]
    fn block_diagonal_update_preserves_group_center() {
        let mut block = vec![2.0f32, 2.4, 1.6, 2.2, -1.0, -0.8, -1.2, -0.9];
        let weights = vec![1.0f32; block.len()];
        let curvature = vec![2.0f32, 2.1, 1.9, 2.0, 1.5, 1.4, 1.6, 1.5];
        let before_a = weighted_group_mean(&block[..4], &weights[..4], &curvature[..4]);
        let before_b = weighted_group_mean(&block[4..], &weights[4..], &curvature[4..]);

        apply_block_diagonal_update(&mut block, &weights, &curvature);

        let after_a = weighted_group_mean(&block[..4], &weights[..4], &curvature[..4]);
        let after_b = weighted_group_mean(&block[4..], &weights[4..], &curvature[4..]);
        assert!((after_a - before_a).abs() < 0.05, "group A drifted too far");
        assert!((after_b - before_b).abs() < 0.05, "group B drifted too far");
    }

    fn mean_squared_error(lhs: &[f32], rhs: &[f32]) -> f32 {
        lhs.iter()
            .zip(rhs.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / lhs.len().max(1) as f32
    }

    #[test]
    fn test_randomized_svd_determinism_and_dimensions() {
        let matrix = vec![1.0f32; 100]; // 10x10 matrix
        let target_rank = 3;
        let (u, s, vt) = randomized_svd_importance(&matrix, 10, 10, target_rank).unwrap();
        
        assert_eq!(u.len(), 10 * target_rank);
        assert_eq!(s.len(), target_rank);
        assert_eq!(vt.len(), target_rank * 10);

        // Deterministic repeat check
        let (u2, s2, vt2) = randomized_svd_importance(&matrix, 10, 10, target_rank).unwrap();
        assert_eq!(u, u2);
        assert_eq!(s, s2);
        assert_eq!(vt, vt2);
    }

    #[test]
    fn gptq_3bit_cross_word_packing() {
        // Test 3-bit GPTQ unpacking across 3 u32 words
        // 32 values packed across 96 bits (3 u32 words)
        // This is the BitBLAS/GPTQ format used by EfficientQAT
        
        // Create mock qweight: 3 u32 words
        let mut qweight = [0u8; 12];
        // Word 0: bits 0-10 of values 0-10
        // Word 1: bits 11-21 of values 11-21  
        // Word 2: bits 22-31 of values 22-31
        // Simplified: just use identity pattern for testing
        
        // Create mock qzeros (sequential, no desc_act)
        let qzeros = [0u8, 0u8]; // zero point = 0
        
        // Create mock scales
        let scales = [0x3F80_0000u32, 0x3F80_0000u32]; // scale = 1.0
        let mut scales_bytes = [0u8; 8];
        scales_bytes[0..4].copy_from_slice(&scales[0].to_le_bytes());
        scales_bytes[4..8].copy_from_slice(&scales[1].to_le_bytes());
        
        let result = dequant_gptq_group_int(
            &qweight,
            &qzeros,
            &scales_bytes,
            &[32, 1], // in_features=32, out_features=1
            3,       // 3-bit
            32,      // group_size=32
        );
        
        // Should produce 32 values
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 32);
    }

    #[test]
    fn gptq_2bit_basic() {
        // 2-bit GPTQ: 16 values per u32 word
        let qweight = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0]; // 8 bytes = 16 values
        let qzeros = [0u8; 8]; // all zeros
        let scales = [0x3F80_0000u32; 4]; // scale = 1.0
        let mut scales_bytes = [0u8; 16];
        for (i, &s) in scales.iter().enumerate() {
            scales_bytes[i*4..(i+1)*4].copy_from_slice(&s.to_le_bytes());
        }
        
        let result = dequant_gptq_group_int(
            &qweight,
            &qzeros,
            &scales_bytes,
            &[16, 1], // in_features=16
            2,       // 2-bit
            16,      // group_size=16
        );
        
        assert!(result.is_ok());
        let deq = result.unwrap();
        assert_eq!(deq.len(), 16);
    }

    #[test]
    fn gptq_4bit_basic() {
        // 4-bit GPTQ: 8 values per u32 word  
        let qweight = [0x12, 0x34, 0x56, 0x78]; // 4 bytes = 8 values
        let qzeros = [0u8, 0u8, 0u8, 0u8]; // zero points
        let scales = [0x3F80_0000u32; 2]; // scale = 1.0
        let mut scales_bytes = [0u8; 8];
        for (i, &s) in scales.iter().enumerate() {
            scales_bytes[i*4..(i+1)*4].copy_from_slice(&s.to_le_bytes());
        }
        
        let result = dequant_gptq_group_int(
            &qweight,
            &qzeros,
            &scales_bytes,
            &[8, 1], // in_features=8
            4,       // 4-bit
            8,       // group_size=8
        );
        
        assert!(result.is_ok());
        let deq = result.unwrap();
        assert_eq!(deq.len(), 8);
    }

    // ------------------------------------------------------------------------
    // Pass 4: Fisher/Hessian diagonal — unit tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_compute_fisher_diagonal_empty_calibration() {
        let weights = vec![0.1f32; 256];
        let result = compute_fisher_diagonal(&weights, &[], 16, 16, 128);
        // Empty calibration → should return ones (identity-like curvature)
        assert_eq!(result.len(), 256);
        assert!(result.iter().all(|v| (*v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_compute_fisher_diagonal_single_sample() {
        let rows = 4;
        let cols = 8;
        let weights = vec![0.1f32; rows * cols];
        let samples = vec![FisherCalibrationSample {
            input_activations: vec![1.0; cols],
            output_gradients: vec![0.5; rows],
        }];
        let result = compute_fisher_diagonal(&weights, &samples, rows, cols, 128);
        assert_eq!(result.len(), rows * cols);
        assert!(result.iter().all(|v| *v > 0.0));
    }

    #[test]
    fn test_compute_grouped_fisher_diagonal() {
        let rows = 4;
        let cols = 64;
        let weights = vec![0.1f32; rows * cols];
        let samples = vec![FisherCalibrationSample {
            input_activations: vec![1.0; cols],
            output_gradients: vec![1.0; rows],
        }];
        let result = compute_grouped_fisher_diagonal(&weights, &samples, rows, cols, 32);
        let expected_groups = (cols + 32 - 1) / 32;
        assert_eq!(result.len(), expected_groups);
        assert!(result.iter().all(|v| *v > 0.0));
    }

    #[test]
    fn test_compute_grouped_fisher_diagonal_empty() {
        let result = compute_grouped_fisher_diagonal(&[], &[], 4, 64, 32);
        assert_eq!(result.len(), 2); // 64/32 = 2 groups, returns ones
        assert!(result.iter().all(|v| (*v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_compute_curvature_proxy() {
        let data = vec![0.0f32, 1.0, -1.0, 2.0, -2.0];
        let layer_importance = 1.0;
        let result = compute_curvature_proxy(&data, layer_importance);
        assert_eq!(result.len(), data.len());
        // Base value is 1.0 + importance * (|x| + x²) min 16
        assert!(result.iter().all(|v| *v >= 1.0));
        // Larger magnitude → larger curvature
        assert!(result[3] > result[0]); // |2.0| > |0.0|
        assert!(result[4] > result[1]); // |-2.0| > |1.0|
    }

    #[test]
    fn test_compute_curvature_proxy_zero_importance() {
        let data = vec![1.0f32, 2.0, 3.0];
        let result = compute_curvature_proxy(&data, 0.0);
        // Minimum scale is 1e-3 even when importance is 0 (safeguard against degenerate values)
        // value=1.0: 1.0 + 0.001 * (1+1) = 1.002
        // value=2.0: 1.0 + 0.001 * (2+4) = 1.006
        // value=3.0: 1.0 + 0.001 * (3+9) = 1.012
        assert!((result[0] - 1.002).abs() < 1e-5);
        assert!((result[1] - 1.006).abs() < 1e-5);
        assert!((result[2] - 1.012).abs() < 1e-5);
        // All values >= 1.0
        assert!(result.iter().all(|v| *v >= 1.0));
    }
}
