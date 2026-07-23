//! `grim oxidizer` — ROCm-optimized GGUF conversion tool.

use std::collections::HashMap;
use std::fs;
use std::io::{BufReader, BufWriter, Read, Seek, Write};
use std::path::Path;

use grim_format::gguf::{
    read_gguf, read_tensor_bytes, GgufDType, GgufFile, GgufTensorInfo, GgufValue, GrimFusionOp,
    GrimLayoutHint, GrimMetadata, GrimRocmlProfile, GrimTrainQuantMode,
};
use grim_format::GgufProvider;
use grim_quant::{
    compute_fisher_diagonal, compute_importance_scores,
    dequant_q4k, dequant_q80, evopress_search, rewrite_tensor_data, EvoPressConfig,
    FisherCalibrationSample, ImportanceScores, QuantFormat, RewrittenTensorData,
    TensorRewritePlan,
};
use grim_backend_rocm::{
    enforce_attention_precision, is_attention_projection, resolve_weight_layout,
    WeightLayout,
};
use grim_tensor::provider::TensorProvider;
use grim_tensor_graph::build_transformer_ir;

const OXIDIZER_VERSION: u32 = 1;

fn open_provider(path: &str) -> Result<(Box<dyn TensorProvider>, Vec<String>, Vec<usize>, GrimMetadata), String> {
    let lower = path.to_ascii_lowercase();
    if lower.ends_with(".safetensors") || lower.ends_with(".bin") {
        let provider = grim_format::tprov::SafetensorsProvider::open(path).map_err(|e| e.to_string())?;
        let names: Vec<String> = provider.tensors().keys().cloned().collect();
        let sizes = names.iter().map(|n| provider.tensors().get(n).map(|i| i.shape().iter().product()).unwrap_or(0)).collect();
        Ok((Box::new(provider), names, sizes, GrimMetadata::default()))
    } else {
        let provider = GgufProvider::open(path).map_err(|e| e.to_string())?;
        let names: Vec<String> = provider.tensors().keys().cloned().collect();
        let sizes = names.iter().map(|n| provider.tensors().get(n).map(|i| i.shape().iter().product()).unwrap_or(0)).collect();
        let meta = provider.grim_metadata().clone();
        Ok((Box::new(provider), names, sizes, meta))
    }
}

pub fn cmd_oxidizer_info(path: &str) -> Result<(), String> {
    let lower = path.to_ascii_lowercase();
    if lower.ends_with(".safetensors") || lower.ends_with(".bin") {
        let provider = grim_format::tprov::SafetensorsProvider::open(path).map_err(|e| e.to_string())?;
        println!("File: {path}");
        println!("Format: safetensors");
        println!("Tensors: {} entries", provider.tensors().len());
        return Ok(());
    }
    let provider = GgufProvider::open(path).map_err(|e| e.to_string())?;
    let grim = provider.grim_metadata();

    println!("File: {path}");
    println!("Format: {}", if grim.is_grim() { ".grim (ROCm-optimized)" } else { "plain GGUF" });
    if let Some(magic) = &grim.magic {
        println!("grim.magic: {magic}");
    }
    if let Some(v) = grim.quant_version {
        println!("grim.quant_version: {v}");
    }
    println!("grim.rocml.profile: {:?}", grim.rocml_profile);
    if grim.wavefront_size > 0 {
        println!("grim.rocml.wavefront_size: {}", grim.wavefront_size);
    }
    if let Some(ref gcn) = grim.target_gcn {
        println!("grim.rocml.target_gcn: {gcn}");
    }
    if let Some(lds) = grim.lds_size {
        println!("grim.rocml.lds_size: {lds}");
    }
    if let Some(xnack) = grim.xnack_enabled {
        println!("grim.rocml.xnack_enabled: {xnack}");
    }
    if let Some(kv) = grim.kv_layout_optimized {
        println!("grim.rocml.kv_layout_optimized: {kv}");
    }
    if !grim.rocm_fusion_ops.is_empty() {
        println!(
            "grim.rocm.fusion_ops: {}",
            grim.rocm_fusion_ops
                .iter()
                .map(|op| op.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    if let Some(mode) = grim.train_quant_mode {
        println!("grim.train.quant_mode: {}", mode.as_str());
    }
    if !grim.train_fusion_ops.is_empty() {
        println!(
            "grim.train.fusion_ops: {}",
            grim.train_fusion_ops
                .iter()
                .map(|op| op.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    println!("grim.quant_overrides: {} entries", grim.quant_overrides.len());
    Ok(())
}

// ---------------------------------------------------------------------------
// Calibration batch for Fisher/Hessian diagonal computation
// ---------------------------------------------------------------------------

/// Holds a batch of calibration samples collected during model forward+backward
/// passes. Each sample contains input activations and output gradients for one
/// or more weight tensors.
///
/// When `samples` is empty, `build_curvature` falls back to the CPU heuristic
/// (`build_curvature_proxy`) so there's no regression for users without a
/// calibrated model.
#[allow(dead_code)] // benchmark helper
#[derive(Debug, Clone, Default)]
pub struct CalibrationBatch {
    pub samples: Vec<FisherCalibrationSample>,
    pub group_size: usize,
}

#[allow(dead_code)]
impl CalibrationBatch {
    pub fn new(group_size: usize) -> Self {
        Self { samples: Vec::new(), group_size }
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn add_sample(&mut self, sample: FisherCalibrationSample) {
        self.samples.push(sample);
    }
}

pub fn cmd_oxidizer_calibrate(
    model_path: &str,
    output_path: &str,
    calibration_dataset: Option<&str>,
) -> Result<ImportanceScores, String> {
    if let Some(ds) = calibration_dataset {
        eprintln!("[oxidizer] calibrate: using dataset '{ds}'");
    } else {
        eprintln!("[oxidizer] calibrate: no calibration dataset provided (CPU heuristic)");
    }
    let (provider, names, _sizes, _meta) = open_provider(model_path)?;
    let mut tensor_data: Vec<(String, Vec<f32>, usize, usize)> = Vec::new();

    for name in &names {
        let meta = provider.meta(name).map_err(|e| e.to_string())?;
        let shape = meta.shape;
        if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
            continue;
        }
        let Ok(tensor) = provider.get(name) else { continue };
        if tensor.bytes.len() < shape[0] * shape[1] * 4 {
            continue;
        }
        let flat = tensor
            .bytes
            .chunks_exact(4)
            .take(shape[0] * shape[1])
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<_>>();
        tensor_data.push((name.clone(), flat, shape[0], shape[1]));
    }

    let scores = compute_importance_scores(&tensor_data);
    let names_collected = tensor_data.iter().map(|(n, _, _, _)| n.clone()).collect();
    let result = ImportanceScores::new(names_collected, scores);
    let out_json = serde_json::json!({
        "version": OXIDIZER_VERSION,
        "model_path": model_path,
        "calibration_dataset": calibration_dataset,
        "tensors": result.tensor_names.iter().zip(result.layer_scores.iter()).map(|(n, s)| {
            serde_json::json!({ "name": n, "importance_score": s })
        }).collect::<Vec<_>>(),
    });
    let json_path = format!("{}.importance.json", output_path);
    fs::write(&json_path, serde_json::to_string_pretty(&out_json).unwrap())
        .map_err(|e| format!("failed to write importance scores: {e}"))?;
    Ok(result)
}

pub fn cmd_oxidizer_search(
    importance_scores: &ImportanceScores,
    tensor_sizes: &[usize],
    target_bpw: f32,
    generations: usize,
) -> Vec<u32> {
    evopress_search(
        &EvoPressConfig {
            target_bpw,
            generations,
            ..Default::default()
        },
        &importance_scores.layer_scores,
        tensor_sizes,
    )
}

pub fn cmd_oxidizer_convert(
    model_path: &str,
    output_path: &str,
    target_bpw: f32,
    generations: usize,
    rocml_profile: Option<&str>,
    calibration_dataset: Option<String>,
) -> Result<(), String> {
    let (provider, names, sizes, mut grim_meta) = open_provider(model_path)?;
    let importance_scores = if Path::new(&format!("{}.importance.json", model_path)).exists() {
        load_importance_scores(&format!("{}.importance.json", model_path))?
    } else {
        cmd_oxidizer_calibrate(model_path, output_path, calibration_dataset.as_deref())?
    };

    let tensor_names = importance_scores.tensor_names.clone();
    let tensor_sizes = tensor_names
        .iter()
        .map(|name| {
            if let Some(idx) = names.iter().position(|n| n == name) {
                sizes[idx]
            } else {
                0
            }
        })
        .collect::<Vec<usize>>();
    let bitwidths = cmd_oxidizer_search(&importance_scores, &tensor_sizes, target_bpw, generations);

    // Create full bitwidths array for ALL tensors in the model
    // Tensors with importance scores get their EvoPress bitwidth, others get target_bpw
    let default_bw = target_bpw.round() as u32;
    let full_bitwidths: Vec<u32> = names.iter().map(|name| {
        if let Some(idx) = tensor_names.iter().position(|n| n == name) {
            bitwidths[idx]
        } else {
            default_bw
        }
    }).collect();

    grim_meta.magic = Some("grim-v1".into());
    grim_meta.quant_version = Some(OXIDIZER_VERSION);
    grim_meta.rocml_profile = rocml_profile
        .map(GrimRocmlProfile::from_str)
        .unwrap_or(grim_meta.rocml_profile);
    grim_meta.wavefront_size = grim_meta.rocml_profile.wavefront_size();
    grim_meta.lds_size = Some(grim_meta.rocml_profile.lds_size());
    grim_meta.quant_method = Some("evopress-gptq-sequential".into());
    grim_meta.calibration_dataset = calibration_dataset.clone();
    grim_meta.quant_overrides = names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let bw = full_bitwidths[i];
            let effective_bpw = if is_attention_projection(name) {
                enforce_attention_precision(bw)
            } else {
                bw
            };
            let layout_hint = if is_attention_projection(name) {
                Some(GrimLayoutHint::WavefrontTiled)
            } else {
                None
            };
            grim_format::gguf::GrimQuantOverride {
                tensor_name: name.clone(),
                effective_bpw,
                override_dtype: bitwidth_to_dtype(effective_bpw),
                importance_score: importance_scores.layer_scores.get(i).copied().unwrap_or(0.0),
                layout_hint,
            }
        })
        .collect();

    let resolved_gcn = rocml_profile.unwrap_or("gfx1100");
    grim_format::convert_to_grim(
        model_path,
        output_path,
        resolved_gcn,
        target_bpw,
        generations,
        calibration_dataset.as_deref(),
        None,
        Some(full_bitwidths),
        // Pass the calibrated metadata through — `convert_to_grim` will
        // preserve `quant_overrides` / `ext_entries` / etc. that carry
        // per-tensor importance scores and layout hints, and stamp the
        // grim-v1 identity + ROCm target/gcn fields on top.
        Some(grim_meta),
    ).map_err(|e| e.to_string())?;
    Ok(())
}

pub fn cmd_oxidizer_prepare(
    input_path: &str,
    output_path: &str,
    train: bool,
    format: &str,
    profile: Option<&str>,
    dataset: Option<String>,
) -> Result<(), String> {
    let (_provider, names, _sizes, mut grim) = open_provider(input_path)?;
    grim.magic = Some("grim-v1".into());
    grim.quant_version = Some(OXIDIZER_VERSION);
    if let Some(profile) = profile {
        grim.rocml_profile = GrimRocmlProfile::from_str(profile);
        grim.wavefront_size = grim.rocml_profile.wavefront_size();
        grim.lds_size = Some(grim.rocml_profile.lds_size());
    }
    grim.calibration_dataset = dataset;
    if train {
        grim.train_quant_mode = GrimTrainQuantMode::from_str(format);
        grim.train_fusion_ops = inferred_fusion_ops(&names);
        grim.quant_method.get_or_insert_with(|| "train-prepare".into());
    }
    write_grim_file(input_path, output_path, &grim, &HashMap::new())
}

pub fn cmd_oxidizer_fuse(
    input_path: &str,
    output_path: &str,
    profile: Option<&str>,
    rocm: bool,
) -> Result<(), String> {
    let (_provider, names, _sizes, mut grim) = open_provider(input_path)?;
    grim.magic = Some("grim-v1".into());
    grim.quant_version = Some(OXIDIZER_VERSION);
    if let Some(profile) = profile {
        grim.rocml_profile = GrimRocmlProfile::from_str(profile);
    }
    grim.wavefront_size = grim.rocml_profile.wavefront_size();
    grim.lds_size = Some(grim.rocml_profile.lds_size());
    grim.rocm_fusion_ops = inferred_fusion_ops(&names);
    grim.kv_layout_optimized = Some(rocm);
    grim.xnack_enabled = Some(false);
    grim.quant_method.get_or_insert_with(|| "rocm-fuse".into());
    write_grim_file(input_path, output_path, &grim, &HashMap::new())
}

fn inferred_fusion_ops(names: &[String]) -> Vec<GrimFusionOp> {
    let ir = build_transformer_ir(names.iter().map(String::as_str));
    ir.recommended_fusion_ops()
}

fn load_importance_scores(path: &str) -> Result<ImportanceScores, String> {
    let content = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let v: serde_json::Value = serde_json::from_str(&content).map_err(|e| e.to_string())?;
    let tensors = v["tensors"].as_array().ok_or("invalid cached importance format")?;
    let names = tensors
        .iter()
        .map(|t| t["name"].as_str().unwrap_or_default().to_string())
        .collect();
    let scores = tensors
        .iter()
        .map(|t| t["importance_score"].as_f64().unwrap_or_default() as f32)
        .collect();
    Ok(ImportanceScores::new(names, scores))
}

fn bitwidth_to_dtype(bw: u32) -> GgufDType {
    match bw {
        0..=2 => GgufDType::Q2K,
        3 => GgufDType::Q3K,
        4 => GgufDType::Q4K,
        5 => GgufDType::Q5K,
        _ => GgufDType::Q6K,
    }
}

#[allow(dead_code)] // benchmark helper
fn build_rewritten_tensors(
    provider: &GgufProvider,
    importance_scores: &ImportanceScores,
    bitwidths: &[u32],
    calibration_batch: &CalibrationBatch,
    grim_meta: Option<&GrimMetadata>,
) -> Result<HashMap<String, RewrittenTensorData>, String> {
    let mut rewritten = HashMap::new();
    for (index, name) in importance_scores.tensor_names.iter().enumerate() {
        let raw = match provider.get(name) {
            Ok(raw) => raw,
            Err(_) => continue,
        };
        if raw.provenance.is_external_qat() {
            continue;
        }
        if raw.shape.len() != 2 || raw.shape[0] == 0 || raw.shape[1] == 0 {
            continue;
        }
        let rows = raw.shape[0];
        let cols = raw.shape[1];

        let suggested_bw = bitwidths.get(index).copied().unwrap_or(4);
        let effective_bw = if is_attention_projection(name) {
            enforce_attention_precision(suggested_bw)
        } else {
            suggested_bw
        };

        let Some(target) = quant_format_for_bitwidth(effective_bw) else { continue };

        let data = match materialize_f32(&raw.bytes, &raw.shape, provider.tensors().get(name).map(|t| t.dtype)) {
            Ok(data) => data,
            Err(_) => continue,
        };

        let layer_importance = importance_scores.layer_scores.get(index).copied().unwrap_or(1.0);
        let importance = vec![layer_importance; data.len()];

        // Fisher diagonal if calibration_batch present, otherwise heuristic proxy
        let curvature = build_curvature(&data, layer_importance, rows, cols, calibration_batch);

        let plan = TensorRewritePlan {
            target,
            shape: raw.shape.clone(),
            importance: Some(importance),
            curvature: Some(curvature),
        };

        let mut rewritten_tensor = match rewrite_tensor_data(&data, &plan) {
            Ok(rt) => rt,
            Err(_) => continue,
        };

        // Wavefront-tiled layout for attention projections on ROCm
        if is_attention_projection(name) {
            let layout = resolve_weight_layout(name, grim_meta, grim_backend_rocm::WavefrontSize::W64);
            if matches!(layout, WeightLayout::WavefrontTiled { .. }) {
                let wf_layout = grim_backend_rocm::WavefrontTiledLayout::new(rows, cols, 64);
                let tiled = wf_layout.tile(&data, rows, cols);
                let (nwf, cpad, wf) = wf_layout.output_shape();
                let tiled_shape = vec![nwf, cpad, wf];
                // Re-quantize from tiled f32
                let tiled_plan = TensorRewritePlan {
                    target,
                    shape: tiled_shape.clone(),
                    importance: Some(vec![layer_importance; tiled.len()]),
                    curvature: Some(build_curvature(&tiled, layer_importance, nwf * wf, cpad, calibration_batch)),
                };
                if let Ok(retiled) = rewrite_tensor_data(&tiled, &tiled_plan) {
                    rewritten_tensor = RewrittenTensorData {
                        bytes: retiled.bytes,
                        logical_shape: tiled_shape,
                        target,
                        wavefront_tiled: true,
                    };
                }
            }
        }

        rewritten.insert(name.clone(), rewritten_tensor);
    }
    Ok(rewritten)
}

/// Compute per-element curvature for GPTQ re-quantization.
///
/// Uses true Fisher/GGN diagonal when `calibration_batch` has samples;
/// otherwise falls back to the heuristic `build_curvature_proxy`.
#[allow(dead_code)] // benchmark helper
fn build_curvature(
    data: &[f32],
    layer_importance: f32,
    rows: usize,
    cols: usize,
    calibration_batch: &CalibrationBatch,
) -> Vec<f32> {
    if calibration_batch.is_empty() {
        return build_curvature_proxy(data, layer_importance);
    }
    compute_fisher_diagonal(data, &calibration_batch.samples, rows, cols, calibration_batch.group_size)
}

/// Fallback: heuristic curvature proxy using activation magnitude as importance proxy.
/// Used when no calibration data is available.
#[allow(dead_code)] // benchmark helper
fn build_curvature_proxy(data: &[f32], layer_importance: f32) -> Vec<f32> {
    let layer_scale = layer_importance.abs().max(1e-3);
    data.iter()
        .map(|value| 1.0 + layer_scale * (value.abs() + value * value).min(16.0))
        .collect()
}

fn write_grim_file(
    src_path: &str,
    dst_path: &str,
    grim_meta: &GrimMetadata,
    rewritten_tensors: &HashMap<String, RewrittenTensorData>,
) -> Result<(), String> {
    let src = fs::File::open(src_path).map_err(|e| e.to_string())?;
    let mut src_reader = BufReader::new(src);
    let gguf = read_gguf(BufReader::new(fs::File::open(src_path).map_err(|e| e.to_string())?))
        .map_err(|e| e.to_string())?;

    let mut metadata = gguf.metadata.clone();
    metadata.extend(grim_meta.to_gguf_metadata());

    let dst = fs::File::create(dst_path).map_err(|e| e.to_string())?;
    let mut writer = BufWriter::new(dst);
    write_gguf(&mut writer, &gguf, &metadata, rewritten_tensors, &mut src_reader)?;
    writer.flush().map_err(|e| e.to_string())
}

fn write_gguf<W: Write, R: Read + Seek>(
    writer: &mut W,
    gguf: &GgufFile,
    metadata: &HashMap<String, GgufValue>,
    rewritten_tensors: &HashMap<String, RewrittenTensorData>,
    src_reader: &mut R,
) -> Result<(), String> {
    let tensor_meta_size = gguf
        .tensors
        .iter()
        .map(estimate_tensor_info_size)
        .sum::<u64>();
    let metadata_size = metadata
        .iter()
        .map(|(key, value)| estimate_string_size(key) + estimate_value_size(value))
        .sum::<u64>();
    let header_size = 4 + 4 + 8 + 8;
    let unaligned_data_start = header_size + metadata_size + tensor_meta_size;
    let data_start = align32(unaligned_data_start);

    let mut current_offset = 0u64;
    let rewritten_infos = gguf
        .tensors
        .iter()
        .map(|info| {
            let rewritten = rewritten_tensors.get(&info.name);
            let updated = GgufTensorInfo {
                name: info.name.clone(),
                dims: info.dims.clone(),
                offset: current_offset,
                size_bytes: rewritten.map(|r| r.bytes.len() as u64).unwrap_or(info.size_bytes),
                dtype: rewritten.map(|r| gguf_dtype_for_quant_format(r.target)).unwrap_or(info.dtype),
            };
            current_offset += info.size_bytes;
            if let Some(rewritten) = rewritten {
                current_offset = updated.offset + rewritten.bytes.len() as u64;
            }
            updated
        })
        .collect::<Vec<_>>();

    writer.write_all(&0x4655_4747u32.to_le_bytes()).map_err(|e| e.to_string())?;
    writer.write_all(&gguf.version.to_le_bytes()).map_err(|e| e.to_string())?;
    writer.write_all(&(rewritten_infos.len() as u64).to_le_bytes()).map_err(|e| e.to_string())?;
    writer.write_all(&(metadata.len() as u64).to_le_bytes()).map_err(|e| e.to_string())?;

    let mut entries = metadata.iter().collect::<Vec<_>>();
    entries.sort_by(|(a, _), (b, _)| a.cmp(b));
    for (key, value) in entries {
        write_string(writer, key)?;
        write_value(writer, value)?;
    }
    for tensor in &rewritten_infos {
        write_tensor_info(writer, tensor)?;
    }

    let bytes_written = header_size
        + metadata_size
        + tensor_meta_size;
    if data_start > bytes_written {
        writer
            .write_all(&vec![0u8; (data_start - bytes_written) as usize])
            .map_err(|e| e.to_string())?;
    }

    for (src_info, dst_info) in gguf.tensors.iter().zip(rewritten_infos.iter()) {
        let bytes = if let Some(rewritten) = rewritten_tensors.get(&src_info.name) {
            rewritten.bytes.clone()
        } else {
            read_tensor_bytes(src_reader, gguf, src_info).map_err(|e| e.to_string())?
        };
        writer.write_all(&bytes).map_err(|e| e.to_string())?;
        let expected_end = data_start + dst_info.offset + dst_info.size_bytes;
        let actual_end = data_start + dst_info.offset + bytes.len() as u64;
        if actual_end != expected_end {
            return Err(format!("tensor '{}' size mismatch while writing .grim", dst_info.name));
        }
    }
    Ok(())
}

#[allow(dead_code)] // benchmark helper
fn quant_format_for_bitwidth(bw: u32) -> Option<QuantFormat> {
    match bw {
        8 => Some(QuantFormat::Q8_0),
        4 => Some(QuantFormat::Q4K),
        5 => Some(QuantFormat::Q5K),
        6 => Some(QuantFormat::Q6K),
        _ => None,
    }
}

fn gguf_dtype_for_quant_format(format: QuantFormat) -> GgufDType {
    match format {
        QuantFormat::Q8_0 => GgufDType::Q8_0,
        QuantFormat::Q4K => GgufDType::Q4K,
        QuantFormat::Q5K => GgufDType::Q5K,
        QuantFormat::Q6K => GgufDType::Q6K,
        QuantFormat::Fp4 | QuantFormat::Nf4 | QuantFormat::Fp8 | QuantFormat::Fp4Block16 | QuantFormat::Fp8Block16 => unimplemented!("fp4/nf4/fp8 quantization not implemented in CLI"),
    }
}


#[allow(dead_code)] // benchmark helper
fn materialize_f32(bytes: &[u8], shape: &[usize], source_dtype: Option<GgufDType>) -> Result<Vec<f32>, String> {
    let elem_count = shape.iter().product::<usize>();
    match source_dtype.unwrap_or(GgufDType::F32) {
        GgufDType::F32 => Ok(bytes
            .chunks_exact(4)
            .take(elem_count)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        GgufDType::Q8_0 => dequant_q80(bytes, elem_count).map_err(|e| e.to_string()),
        GgufDType::Q4K | GgufDType::Q4_0 | GgufDType::Q4_1 | GgufDType::Q4_2 => {
            dequant_q4k(bytes, elem_count).map_err(|e| e.to_string())
        }
        _ => Err(format!("unsupported source dtype for Pass 4 materialization: {:?}", source_dtype)),
    }
}

fn write_tensor_info<W: Write>(writer: &mut W, tensor: &GgufTensorInfo) -> Result<(), String> {
    write_string(writer, &tensor.name)?;
    writer
        .write_all(&(tensor.dims.len() as u32).to_le_bytes())
        .map_err(|e| e.to_string())?;
    for dim in &tensor.dims {
        writer.write_all(&dim.to_le_bytes()).map_err(|e| e.to_string())?;
    }
    writer
        .write_all(&(tensor.dtype as u32).to_le_bytes())
        .map_err(|e| e.to_string())?;
    writer.write_all(&tensor.offset.to_le_bytes()).map_err(|e| e.to_string())
}

fn write_string<W: Write>(writer: &mut W, value: &str) -> Result<(), String> {
    writer
        .write_all(&(value.len() as u64).to_le_bytes())
        .and_then(|_| writer.write_all(value.as_bytes()))
        .map_err(|e| e.to_string())
}

fn write_value_raw<W: Write>(writer: &mut W, value: &GgufValue) -> Result<(), String> {
    match value {
        GgufValue::Uint8(v) => writer.write_all(&[*v]).map_err(|e| e.to_string()),
        GgufValue::Int8(v) => writer.write_all(&v.to_le_bytes()).map_err(|e| e.to_string()),
        GgufValue::Uint16(v) => writer.write_all(&v.to_le_bytes()).map_err(|e| e.to_string()),
        GgufValue::Int16(v) => writer.write_all(&v.to_le_bytes()).map_err(|e| e.to_string()),
        GgufValue::Uint32(v) => writer.write_all(&v.to_le_bytes()).map_err(|e| e.to_string()),
        GgufValue::Int32(v) => writer.write_all(&v.to_le_bytes()).map_err(|e| e.to_string()),
        GgufValue::Float32(v) => writer.write_all(&v.to_le_bytes()).map_err(|e| e.to_string()),
        GgufValue::Bool(v) => writer.write_all(&[*v as u8]).map_err(|e| e.to_string()),
        GgufValue::String(v) => write_string(writer, v),
        GgufValue::Array(values) => {
            let type_tag = values.first().map(value_type_tag).unwrap_or(8);
            writer.write_all(&type_tag.to_le_bytes()).map_err(|e| e.to_string())?;
            writer
                .write_all(&(values.len() as u64).to_le_bytes())
                .map_err(|e| e.to_string())?;
            for item in values {
                write_value_raw(writer, item)?;
            }
            Ok(())
        }
        GgufValue::Uint64(v) => writer.write_all(&v.to_le_bytes()).map_err(|e| e.to_string()),
        GgufValue::Int64(v) => writer.write_all(&v.to_le_bytes()).map_err(|e| e.to_string()),
        GgufValue::Float64(v) => writer.write_all(&v.to_le_bytes()).map_err(|e| e.to_string()),
    }
}

fn write_value<W: Write>(writer: &mut W, value: &GgufValue) -> Result<(), String> {
    let tag = value_type_tag(value);
    writer.write_all(&tag.to_le_bytes()).map_err(|e| e.to_string())?;
    write_value_raw(writer, value)
}

fn estimate_tensor_info_size(info: &GgufTensorInfo) -> u64 {
    estimate_string_size(&info.name) + 4 + (info.dims.len() as u64 * 8) + 4 + 8
}

fn estimate_string_size(value: &str) -> u64 {
    8 + value.len() as u64
}

fn estimate_value_raw_size(value: &GgufValue) -> u64 {
    match value {
        GgufValue::Uint8(_) | GgufValue::Int8(_) | GgufValue::Bool(_) => 1,
        GgufValue::Uint16(_) | GgufValue::Int16(_) => 2,
        GgufValue::Uint32(_) | GgufValue::Int32(_) | GgufValue::Float32(_) => 4,
        GgufValue::Uint64(_) | GgufValue::Int64(_) | GgufValue::Float64(_) => 8,
        GgufValue::String(v) => estimate_string_size(v),
        GgufValue::Array(values) => {
            let _type_tag = values.first().map(value_type_tag).unwrap_or(8);
            4 + 8 + values.iter().map(estimate_value_raw_size).sum::<u64>()
        }
    }
}

fn estimate_value_size(value: &GgufValue) -> u64 {
    4 + estimate_value_raw_size(value)
}

fn value_type_tag(value: &GgufValue) -> u32 {
    match value {
        GgufValue::Uint8(_) => 0,
        GgufValue::Int8(_) => 1,
        GgufValue::Uint16(_) => 2,
        GgufValue::Int16(_) => 3,
        GgufValue::Uint32(_) => 4,
        GgufValue::Int32(_) => 5,
        GgufValue::Float32(_) => 6,
        GgufValue::Bool(_) => 7,
        GgufValue::String(_) => 8,
        GgufValue::Array(_) => 9,
        GgufValue::Uint64(_) => 10,
        GgufValue::Int64(_) => 11,
        GgufValue::Float64(_) => 12,
    }
}

fn align32(value: u64) -> u64 {
    (value + 31) & !31
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn bitwidth_maps_to_expected_dtype() {
        assert_eq!(bitwidth_to_dtype(2), GgufDType::Q2K);
        assert_eq!(bitwidth_to_dtype(4), GgufDType::Q4K);
        assert_eq!(bitwidth_to_dtype(7), GgufDType::Q6K);
    }

    #[test]
    fn prepare_round_trips_grim_metadata() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("tiny.gguf");
        let output = dir.path().join("tiny.grim");
        write_minimal_gguf(&input).unwrap();

        cmd_oxidizer_prepare(
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            true,
            "bf16",
            Some("cdna3"),
            Some("alpaca".into()),
        )
        .unwrap();

        let provider = GgufProvider::open(output.to_str().unwrap()).unwrap();
        let grim = provider.grim_metadata();
        assert!(grim.is_grim());
        assert_eq!(grim.train_quant_mode, Some(GrimTrainQuantMode::Bf16));
        assert_eq!(grim.calibration_dataset.as_deref(), Some("alpaca"));
        assert_eq!(grim.rocml_profile, GrimRocmlProfile::Cdna3);
    }

    #[test]
    fn fuse_bakes_rocm_fusion_ops_into_output_metadata() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("tiny.gguf");
        let output = dir.path().join("fused.grim");
        write_minimal_gguf(&input).unwrap();

        cmd_oxidizer_fuse(
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            Some("cdna2"),
            true,
        )
        .unwrap();

        let provider = GgufProvider::open(output.to_str().unwrap()).unwrap();
        let grim = provider.grim_metadata();
        assert!(grim.is_grim());
        assert_eq!(grim.rocml_profile, GrimRocmlProfile::Cdna2);
        // tiny.gguf fixture contains `blk.0.attention.wq.weight` -> QKV fusion is inferred
        assert_eq!(grim.quant_method.as_deref(), Some("rocm-fuse"));
        assert_eq!(grim.kv_layout_optimized, Some(true));
    }

    #[test]
    fn write_gguf_rewrites_dtype_and_payload() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("tiny.gguf");
        let output = dir.path().join("rewritten.grim");
        write_minimal_gguf(&input).unwrap();

        let src = fs::File::open(&input).unwrap();
        let mut src_reader = BufReader::new(src);
        let gguf = read_gguf(BufReader::new(fs::File::open(&input).unwrap())).unwrap();

        let mut rewritten_tensors = HashMap::new();
        let rewritten = rewrite_tensor_data(
            &vec![1.0f32; 32],
            &TensorRewritePlan {
                target: QuantFormat::Q8_0,
                shape: vec![32, 1],
                importance: None,
                curvature: None,
            },
        )
        .unwrap();
        rewritten_tensors.insert("blk.0.attention.wq.weight".into(), rewritten.clone());

        let file = fs::File::create(&output).unwrap();
        let mut writer = BufWriter::new(file);
        write_gguf(&mut writer, &gguf, &gguf.metadata, &rewritten_tensors, &mut src_reader).unwrap();
        writer.flush().unwrap();

        let rewritten_file = read_gguf(BufReader::new(fs::File::open(&output).unwrap())).unwrap();
        assert_eq!(rewritten_file.tensors[0].dtype, GgufDType::Q8_0);

        let mut rewritten_reader = BufReader::new(fs::File::open(&output).unwrap());
        let rewritten_bytes = read_tensor_bytes(&mut rewritten_reader, &rewritten_file, &rewritten_file.tensors[0]).unwrap();
        assert_eq!(rewritten_bytes, rewritten.bytes);
    }

    fn write_minimal_gguf(path: &Path) -> Result<(), String> {
        let tensor = GgufTensorInfo {
            name: "blk.0.attention.wq.weight".into(),
            dims: vec![32, 1],
            offset: 0,
            size_bytes: 128,
            dtype: GgufDType::F32,
        };
        let gguf = GgufFile {
            version: 3,
            tensor_count: 1,
            metadata: HashMap::from([(
                "general.architecture".into(),
                GgufValue::String("llama".into()),
            )]),
            tensors: vec![tensor],
            data_start: 0,
        };
        let mut src = BufReader::new(std::io::Cursor::new(vec![0u8; 128]));
        let file = fs::File::create(path).map_err(|e| e.to_string())?;
        let mut writer = BufWriter::new(file);
        write_gguf(&mut writer, &gguf, &gguf.metadata, &HashMap::new(), &mut src)?;
        writer.flush().map_err(|e| e.to_string())
    }
}
