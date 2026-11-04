//! LoRA support for diffusion models (Flux 2 MM-DiT and UNet 2D).
//!
//! Supports:
//! - Loading LoRA weights from safetensors (Kohya and Diffusers formats).
//! - Normalizing keys to canonical model linear projection names.
//! - Merge-at-load: `W += (scale * alpha / rank) * (B @ A)`.
//! - Runtime dispatched LoRA forward pass.

use std::collections::HashMap;
use std::path::Path;
use grim_backend_cpu::cpu_tensor;
use grim_core::error::{Error, Result};
use grim_format::safetensors::read_safetensors_header;
use grim_tensor::{Shape, Tensor};

/// A single low-rank LoRA adapter module (A and B matrices).
#[derive(Debug, Clone)]
pub struct LoraModule {
    pub a: Tensor,
    pub b: Tensor,
    pub rank: usize,
    pub in_features: usize,
    pub out_features: usize,
    pub alpha: f32,
    pub scale: f32,
}

impl LoraModule {
    pub fn new(a: Tensor, b: Tensor, alpha: f32, scale: f32) -> Result<Self> {
        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();
        if a_dims.len() < 2 || b_dims.len() < 2 {
            return Err(Error::Shape("LoRA matrices must have at least 2 dimensions".into()));
        }
        let rank = a_dims[0];
        let in_features = a_dims[1];
        let out_features = b_dims[0];
        let b_rank = b_dims[1];

        if rank != b_rank {
            return Err(Error::Shape(format!(
                "LoRA rank mismatch: A rank {rank} != B rank {b_rank}"
            )));
        }

        Ok(Self {
            a,
            b,
            rank,
            in_features,
            out_features,
            alpha,
            scale,
        })
    }

    /// Compute delta matrix: `delta = (scale * alpha / rank) * (B @ A)`
    pub fn compute_delta(&self) -> Result<Vec<f32>> {
        let eff_scale = self.scale * (self.alpha / self.rank as f32);
        let a_vec = self.a.to_vec_f32()?;
        let b_vec = self.b.to_vec_f32()?;
        let mut delta = vec![0.0f32; self.out_features * self.in_features];

        // B: [out_features, rank], A: [rank, in_features]
        for o in 0..self.out_features {
            for i in 0..self.in_features {
                let mut sum = 0.0f32;
                for r in 0..self.rank {
                    sum += b_vec[o * self.rank + r] * a_vec[r * self.in_features + i];
                }
                delta[o * self.in_features + i] = eff_scale * sum;
            }
        }
        Ok(delta)
    }

    /// Forward pass through LoRA: `delta_y = (scale * alpha / rank) * (x @ A^T) @ B^T`
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let eff_scale = self.scale * (self.alpha / self.rank as f32);
        let x_dims = x.shape().dims();
        let rows: usize = x_dims[..x_dims.len() - 1].iter().product();
        let cols = *x_dims.last().unwrap_or(&1);

        if cols != self.in_features {
            return Err(Error::Shape(format!(
                "LoraModule::forward: input feat {cols} != in_features {}",
                self.in_features
            )));
        }

        let x_vec = x.to_vec_f32()?;
        let a_vec = self.a.to_vec_f32()?;
        let b_vec = self.b.to_vec_f32()?;
        let mut out_vec = vec![0.0f32; rows * self.out_features];

        // x: [rows, in_features]
        // A: [rank, in_features] -> (x @ A^T): [rows, rank]
        // B: [out_features, rank] -> (inter @ B^T): [rows, out_features]
        for t in 0..rows {
            for o in 0..self.out_features {
                let mut delta = 0.0f32;
                for r in 0..self.rank {
                    let mut inter = 0.0f32;
                    for i in 0..self.in_features {
                        inter += x_vec[t * self.in_features + i] * a_vec[r * self.in_features + i];
                    }
                    delta += b_vec[o * self.rank + r] * inter;
                }
                out_vec[t * self.out_features + o] = eff_scale * delta;
            }
        }

        let mut out_shape_dims = x_dims.to_vec();
        *out_shape_dims.last_mut().unwrap() = self.out_features;
        Ok(cpu_tensor(out_vec, Shape::new(out_shape_dims)))
    }
}

/// A parsed Diffusion LoRA adapter containing modules targeted at canonical model projections.
#[derive(Debug, Clone)]
pub struct DiffusionLoraAdapter {
    pub modules: HashMap<String, LoraModule>,
    pub scale: f32,
}

impl DiffusionLoraAdapter {
    /// Load a diffusion LoRA adapter from a safetensors file.
    pub fn from_safetensors(path: &Path, scale: f32) -> Result<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| Error::Backend(format!("Failed to open safetensors file {:?}: {e}", path)))?;
        let mut reader = std::io::BufReader::new(file);
        let (tensor_infos, _meta, _header_len) = read_safetensors_header(&mut reader)
            .map_err(|e| Error::Backend(format!("Failed to read safetensors header: {e}")))?;

        // Map tensor names to raw f32 data
        let mut raw_tensors: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
        use std::io::{Read, Seek, SeekFrom};

        for (name, info) in tensor_infos {
            reader.seek(SeekFrom::Start(info.data_start))
                .map_err(|e| Error::Backend(format!("Seek failed: {e}")))?;
            let size = (info.data_end - info.data_start) as usize;
            let mut buf = vec![0u8; size];
            reader.read_exact(&mut buf)
                .map_err(|e| Error::Backend(format!("Read failed for tensor {name}: {e}")))?;

            let vals: Vec<f32> = match info.dtype_tag.as_str() {
                "F32" => {
                    buf.chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                        .collect()
                }
                "F16" => {
                    buf.chunks_exact(2)
                        .map(|c| half::f16::from_le_bytes(c.try_into().unwrap()).to_f32())
                        .collect()
                }
                "BF16" => {
                    buf.chunks_exact(2)
                        .map(|c| half::bf16::from_le_bytes(c.try_into().unwrap()).to_f32())
                        .collect()
                }
                other => {
                    return Err(Error::Backend(format!("Unsupported dtype {other} in LoRA safetensors")));
                }
            };
            raw_tensors.insert(name, (info.dims, vals));
        }

        Self::from_raw_tensors(&raw_tensors, scale)
    }

    /// Construct adapter from in-memory parsed tensors.
    pub fn from_raw_tensors(
        tensors: &HashMap<String, (Vec<usize>, Vec<f32>)>,
        scale: f32,
    ) -> Result<Self> {
        // Group into (down/A, up/B, alpha) pairs by normalized module name
        let mut grouped: HashMap<String, (Option<Tensor>, Option<Tensor>, Option<f32>)> = HashMap::new();

        for (key, (dims, vals)) in tensors {
            let (target_module, kind) = normalize_diffusion_lora_key(key)?;
            let entry = grouped.entry(target_module).or_insert((None, None, None));
            match kind {
                LoraKeyKind::Down => {
                    let shape = Shape::new(dims.clone());
                    entry.0 = Some(cpu_tensor(vals.clone(), shape));
                }
                LoraKeyKind::Up => {
                    let shape = Shape::new(dims.clone());
                    entry.1 = Some(cpu_tensor(vals.clone(), shape));
                }
                LoraKeyKind::Alpha => {
                    if let Some(&a) = vals.first() {
                        entry.2 = Some(a);
                    }
                }
            }
        }

        let mut modules = HashMap::new();
        for (name, (a_opt, b_opt, alpha_opt)) in grouped {
            if let (Some(a), Some(b)) = (a_opt, b_opt) {
                let rank = a.shape().dims()[0];
                let alpha = alpha_opt.unwrap_or(rank as f32);
                let module = LoraModule::new(a, b, alpha, scale)?;
                modules.insert(name, module);
            }
        }

        Ok(Self { modules, scale })
    }

    /// Apply merged weights into base weights map:
    /// `W += (scale * alpha / rank) * (B @ A)`
    /// Returns the number of tensors merged.
    pub fn apply_merged(
        &self,
        weights: &mut HashMap<String, Tensor>,
        scale_override: Option<f32>,
    ) -> Result<usize> {
        let mut merged_count = 0;

        for (name, module) in &self.modules {
            if let Some(base_tensor) = weights.get_mut(name) {
                let mut mod_copy = module.clone();
                if let Some(s) = scale_override {
                    mod_copy.scale = s;
                }
                let delta = mod_copy.compute_delta()?;
                let mut base_vec = base_tensor.to_vec_f32()?;
                if base_vec.len() != delta.len() {
                    return Err(Error::Shape(format!(
                        "apply_merged: weight {name} len {} != delta len {}",
                        base_vec.len(),
                        delta.len()
                    )));
                }

                for i in 0..base_vec.len() {
                    base_vec[i] += delta[i];
                }

                *base_tensor = cpu_tensor(base_vec, base_tensor.shape().clone());
                merged_count += 1;
            }
        }

        Ok(merged_count)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum LoraKeyKind {
    Down,
    Up,
    Alpha,
}

/// Normalize Kohya and Diffusers LoRA keys into canonical linear layer names.
///
/// Examples:
/// - Kohya UNet: `lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight`
///   -> (`down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q`, LoraKeyKind::Down)
/// - Diffusers Flux: `transformer.joint_blocks.0.img_attn.qkv.lora_A.weight`
///   -> (`joint_blocks.0.img_attn.qkv`, LoraKeyKind::Down)
/// - Diffusers UNet: `unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_B.weight`
///   -> (`down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q`, LoraKeyKind::Up)
pub fn normalize_diffusion_lora_key(key: &str) -> Result<(String, LoraKeyKind)> {
    let (stripped, kind) = if let Some(s) = key.strip_suffix(".lora_down.weight") {
        (s, LoraKeyKind::Down)
    } else if let Some(s) = key.strip_suffix(".lora_down.bias") {
        (s, LoraKeyKind::Down)
    } else if let Some(s) = key.strip_suffix(".lora_A.weight") {
        (s, LoraKeyKind::Down)
    } else if let Some(s) = key.strip_suffix(".lora_up.weight") {
        (s, LoraKeyKind::Up)
    } else if let Some(s) = key.strip_suffix(".lora_up.bias") {
        (s, LoraKeyKind::Up)
    } else if let Some(s) = key.strip_suffix(".lora_B.weight") {
        (s, LoraKeyKind::Up)
    } else if let Some(s) = key.strip_suffix(".alpha") {
        (s, LoraKeyKind::Alpha)
    } else {
        return Err(Error::Backend(format!("Unknown LoRA key format: {key}")));
    };

    let canonical = if let Some(rest) = stripped.strip_prefix("lora_unet_") {
        // Kohya UNet: e.g. down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q
        // replaces `_` around numbers and structure markers to match diffusers/canonical hierarchy
        convert_kohya_unet_key(rest)
    } else if let Some(rest) = stripped.strip_prefix("lora_te_") {
        format!("text_encoder.{}", rest)
    } else if let Some(rest) = stripped.strip_prefix("transformer.") {
        rest.to_string()
    } else if let Some(rest) = stripped.strip_prefix("unet.") {
        rest.to_string()
    } else {
        stripped.to_string()
    };

    Ok((canonical, kind))
}

fn convert_kohya_unet_key(key: &str) -> String {
    // Kohya UNet naming uses underscores where Diffusers uses dots, e.g.:
    // down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q -> down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q
    let mut s = key.to_string();
    // Replace known token boundaries
    for token in &["down_blocks", "mid_block", "up_blocks", "attentions", "resnets", "transformer_blocks"] {
        s = s.replace(&format!("{token}_"), &format!("{token}."));
    }
    // Replace digit boundaries like `.0_` -> `.0.`
    let mut res = String::new();
    let chars: Vec<char> = s.chars().collect();
    let len = chars.len();
    let mut i = 0;
    while i < len {
        if chars[i] == '_' {
            let prev_digit = i > 0 && chars[i - 1].is_ascii_digit();
            let next_digit = i + 1 < len && chars[i + 1].is_ascii_digit();
            let next_is_marker = s[i + 1..].starts_with("attentions")
                || s[i + 1..].starts_with("resnets")
                || s[i + 1..].starts_with("transformer_blocks")
                || s[i + 1..].starts_with("attn1")
                || s[i + 1..].starts_with("attn2")
                || s[i + 1..].starts_with("to_")
                || s[i + 1..].starts_with("proj");
            if prev_digit || next_digit || next_is_marker {
                res.push('.');
            } else {
                res.push('_');
            }
        } else {
            res.push(chars[i]);
        }
        i += 1;
    }
    res
}
