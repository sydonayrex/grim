//! `grim oxidizer` — ROCm-optimized GGUF conversion tool.

use std::collections::HashMap;
use std::fs;
use std::io::{BufReader, BufWriter, Read, Seek, Write};
use std::path::Path;

use grim_backend_rocm::{
    WeightLayout, enforce_attention_precision, is_attention_projection, resolve_weight_layout,
};
use grim_format::GgufProvider;
use grim_format::fusion::build_transformer_ir;
use grim_format::gguf::{
    GgufDType, GgufFile, GgufTensorInfo, GgufValue, GrimFusionOp, GrimMetadata,
    GrimRocmlProfile, GrimTrainQuantMode, read_gguf, read_tensor_bytes,
};
use grim_quant::{
    FisherCalibrationSample, ImportanceScores, QuantFormat, RcoConfig, RewrittenTensorData,
    TensorRewritePlan, compute_fisher_diagonal, compute_importance_scores, dequant_q4k,
    dequant_q80, rco_search, rewrite_tensor_data,
};
use grim_tensor::provider::TensorProvider;

const OXIDIZER_VERSION: u32 = 1;

fn open_provider(
    path: &str,
) -> Result<
    (
        Box<dyn TensorProvider>,
        Vec<String>,
        Vec<usize>,
        GrimMetadata,
    ),
    String,
> {
    let lower = path.to_ascii_lowercase();
    if lower.ends_with(".safetensors") || lower.ends_with(".bin") {
        let provider =
            grim_format::tprov::SafetensorsProvider::open(path).map_err(|e| e.to_string())?;
        let names: Vec<String> = provider.tensors().keys().cloned().collect();
        let sizes = names
            .iter()
            .map(|n| {
                provider
                    .tensors()
                    .get(n)
                    .map(|i| i.shape().iter().product())
                    .unwrap_or(0)
            })
            .collect();
        let mut meta = GrimMetadata::default();
        meta.train_fusion_ops = inferred_fusion_ops(&names);
        meta.rocm_fusion_ops = inferred_fusion_ops(&names);
        Ok((Box::new(provider), names, sizes, meta))
    } else {
        let provider = GgufProvider::open(path).map_err(|e| e.to_string())?;
        let names: Vec<String> = provider.tensors().keys().cloned().collect();
        let sizes = names
            .iter()
            .map(|n| {
                provider
                    .tensors()
                    .get(n)
                    .map(|i| i.shape().iter().product())
                    .unwrap_or(0)
            })
            .collect();
        let meta = provider.grim_metadata().clone();
        Ok((Box::new(provider), names, sizes, meta))
    }
}

pub fn cmd_oxidizer_info(path: &str) -> Result<(), String> {
    let lower = path.to_ascii_lowercase();
    if lower.ends_with(".safetensors") || lower.ends_with(".bin") {
        let provider =
            grim_format::tprov::SafetensorsProvider::open(path).map_err(|e| e.to_string())?;
        println!("File: {path}");
        println!("Format: safetensors");
        println!("Tensors: {} entries", provider.tensors().len());
        return Ok(());
    }
    let provider = GgufProvider::open(path).map_err(|e| e.to_string())?;
    let grim = provider.grim_metadata();

    println!("File: {path}");
    println!(
        "Format: {}",
        if grim.is_grim() {
            ".grim (ROCm-optimized)"
        } else {
            "plain GGUF"
        }
    );
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
    println!(
        "grim.quant_overrides: {} entries",
        grim.quant_overrides.len()
    );
    if let Some(alloc) = &grim.rco_allocation {
        println!(
            "grim.rco.allocation: {} entries (provenance: {})",
            alloc.len(),
            grim.rco_allocation_provenance.as_deref().unwrap_or("unknown")
        );
    }
    if let Some(scores) = &grim.calibration_scores {
        println!("grim.calibration.scores: {} tensors", scores.len());
    }
    match grim_format::format::attachment_index_from_file(path) {
        Ok(idx) if idx.is_empty() => println!("embedded sidecars: none"),
        Ok(idx) => {
            println!(
                "embedded sidecars: {}",
                idx.iter()
                    .map(|a| format!(
                        "{} ({} bytes, sha256:{})",
                        a.kind,
                        a.size,
                        a.sha256.iter().take(8).map(|b| format!("{b:02x}")).collect::<String>()
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        Err(e) => println!("embedded sidecars: index unreadable ({e})"),
    }
    Ok(())
}

// Calibration batch for Fisher/Hessian diagonal computation

/// Holds a batch of calibration samples. Each sample has input activations and
/// output gradients for weight tensors. Empty samples fall back to CPU heuristic.
#[derive(Debug, Clone, Default)]
pub struct CalibrationBatch {
    pub samples: Vec<FisherCalibrationSample>,
    pub group_size: usize,
}

impl CalibrationBatch {
    pub fn new(group_size: usize) -> Self {
        Self {
            samples: Vec::new(),
            group_size,
        }
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
    progress: &mut Option<&mut (dyn FnMut(&str, usize, usize) + Send + Sync)>,
) -> Result<ImportanceScores, String> {
    if let Some(ds) = calibration_dataset {
        eprintln!("[oxidizer] calibrate: using dataset '{ds}'");
    } else {
        eprintln!("[oxidizer] calibrate: no calibration dataset provided (CPU heuristic)");
    }
    let (provider, names, _sizes, _meta) = open_provider(model_path)?;
    let mut batch = CalibrationBatch::new(128);
    batch.add_sample(FisherCalibrationSample {
        input_activations: vec![1.0],
        output_gradients: vec![0.1],
    });
    let mut tensor_data: Vec<(String, Vec<f32>, usize, usize)> = Vec::new();
    let total = names.len();

    for (i, name) in names.iter().enumerate() {
        if let Some(cb) = progress.as_deref_mut() {
            cb("calibrate", i + 1, total);
        }
        let meta = provider.meta(name).map_err(|e| e.to_string())?;
        let shape = meta.shape;
        if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
            continue;
        }
        let Ok(tensor) = provider.get(name) else {
            continue;
        };
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
    let json_text = serde_json::to_string_pretty(&out_json)
        .map_err(|e| format!("failed to serialize importance scores: {e}"))?;
    fs::write(&json_path, json_text)
        .map_err(|e| format!("failed to write importance scores: {e}"))?;
    Ok(result)
}

pub fn cmd_oxidizer_search(
    importance_scores: &ImportanceScores,
    tensor_sizes: &[usize],
    target_bpw: f32,
    generations: usize,
    progress: Option<&mut dyn FnMut(usize, usize)>,
) -> Vec<u32> {
    rco_search(
        &RcoConfig {
            target_bpw,
            steps: generations.max(20),
            // TIER STRUCTURE: 2 = GSQ-RCO (tag 81, the one packed format
            // with per-block scales and a matching reader), 32 = F32
            // verbatim passthrough. The flat 3/4/8-bit tiers are NOT offered:
            // pack_row_bpw_for_wave is scale-less and clamps to [-1, 1], so
            // a "4-bit" assignment both destroyed accuracy and had no reader
            // mapping (dtype_from_bitwidth decodes the bytes as Q4_K/MXFP4 —
            // a different layout). Re-offer intermediate tiers only when a
            // scaled packer + reader pair exists for them.
            available_bpws: vec![2, 32],
            ..Default::default()
        },
        &importance_scores.layer_scores,
        tensor_sizes,
        progress,
    )
}

/// The three embedded sidecar payloads a conversion can carry. Every field
/// is an explicit path (hard error if missing) or `None` (auto-detect a
/// sibling GGUF next to the input). The payloads ride an end-of-file trailer
/// inside the single output `.grim` — no separate sidecar files ship.
#[derive(Debug, Default, Clone)]
pub struct EmbedSidecars {
    /// Multimodal projector GGUF (required for multimodal models, optional
    /// on text-only).
    pub mmproj: Option<String>,
    /// Importance-matrix GGUF (only meaningful for imatrix-derived formats).
    pub imatrix: Option<String>,
    /// MTP / speculative-draft GGUF (MTP tensors beyond the main checkpoint).
    pub mtp: Option<String>,
}

/// Sibling file names auto-detected next to the input model, in probe order.
fn sibling_candidates(kind: &str, model_path: &str) -> Vec<String> {
    let p = Path::new(model_path);
    let dir = p.parent().unwrap_or_else(|| Path::new("."));
    let stem = p
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_else(|| "model");
    let pats: &[&str] = match kind {
        "mmproj" => &[
            "{s}-mmproj.gguf",
            "mmproj-{s}.gguf",
            "{s}.mmproj.gguf",
            "mmproj.gguf",
        ],
        "imatrix" => &["{s}-imatrix.gguf", "{s}.imatrix.gguf", "imatrix-{s}.gguf"],
        "mtp" => &["{s}-mtp.gguf", "{s}.mtp.gguf", "{s}-mtp-draft.gguf", "{s}-draft.gguf"],
        _ => &[],
    };
    pats.iter()
        .map(|pat| dir.join(pat.replace("{s}", stem)).to_string_lossy().into_owned())
        .collect()
}

/// Where a consumable imatrix GGUF lives, in priority order: the explicit
/// `--embed-imatrix` path, an auto-detected sibling next to the input, or —
/// when re-converting a `.grim` — the in-file `imatrix` sidecar extracted to
/// a content-addressed cache path. `None` means nothing to consume.
fn imatrix_source_path(
    model_path: &str,
    embed: &EmbedSidecars,
    embed_attachments: &[(String, Vec<u8>)],
) -> Option<String> {
    if let Some(p) = &embed.imatrix {
        return Some(p.clone()); // resolve() already validated existence
    }
    if let Some(sibling) = sibling_candidates("imatrix", model_path)
        .into_iter()
        .find(|c| Path::new(c).exists())
    {
        return Some(sibling);
    }
    if model_path.ends_with(".grim") {
        let att = grim_format::format::attachment_index_from_file(model_path)
            .ok()?
            .into_iter()
            .find(|a| a.kind == "imatrix")?;
        let sha16: String = att
            .sha256
            .iter()
            .take(8)
            .map(|b| format!("{b:02x}"))
            .collect();
        let dest = std::env::temp_dir()
            .join("grim-sidecars")
            .join(format!("{sha16}-imatrix.gguf"));
        return grim_format::format::extract_attachment_from_file(model_path, "imatrix", &dest)
            .ok()
            .flatten()
            .map(|p| p.to_string_lossy().into_owned());
    }
    let _ = embed_attachments; // bytes already validated by resolve(); path form is what the parser needs
    None
}

impl EmbedSidecars {
    /// Resolve every kind to concrete bytes: an explicit path must exist
    /// (fail loud), `None` auto-detects a sibling and stays silent when the
    /// model simply has no such sidecar. Returns (kind, bytes) pairs in
    /// trailer order mmproj, imatrix, mtp.
    pub fn resolve(&self, model_path: &str) -> Result<Vec<(String, Vec<u8>)>, String> {
        let mut out = Vec::new();
        for (kind, explicit) in [
            ("mmproj", &self.mmproj),
            ("imatrix", &self.imatrix),
            ("mtp", &self.mtp),
        ] {
            let path: String = match explicit {
                Some(p) => {
                    if !Path::new(p).exists() {
                        return Err(format!(
                            "--embed-{kind}: file not found: {p}"
                        ));
                    }
                    p.clone()
                }
                None => match sibling_candidates(kind, model_path)
                    .into_iter()
                    .find(|c| Path::new(c).exists())
                {
                    Some(c) => c,
                    None => continue,
                },
            };
            let bytes = fs::read(&path)
                .map_err(|e| format!("--embed-{kind}: cannot read {path}: {e}"))?;
            eprintln!(
                "[grim convert] embedding {kind} sidecar: {} ({} bytes)",
                path,
                bytes.len()
            );
            out.push((kind.to_string(), bytes));
        }
        Ok(out)
    }
}

pub fn cmd_oxidizer_convert(
    model_path: &str,
    output_path: &str,
    target_bpw: f32,
    generations: usize,
    rocml_profile: Option<&str>,
    calibration_dataset: Option<String>,
    wave_override: Option<grim_format::WaveSize>,
    use_gpu: bool,
    mut progress: Option<&mut (dyn FnMut(&str, usize, usize) + Send + Sync)>,
    target_format: Option<&str>,
    embed: &EmbedSidecars,
) -> Result<(), String> {
    // GRIM-FORMAT DEFAULT (user directive): a .grim conversion with no
    // --format names GSQ-RCO 3.5-bit (tag 81). The .grim container is
    // grim-native, so there is no portability constraint like WhiteRaven's;
    // the GSQ paper's codebook is the source of truth for the format.
    let target_format = target_format.or(Some("gsq_rco_3p5"));
    let (_provider, names, sizes, mut grim_meta) = open_provider(model_path)?;
    let embed_attachments = embed.resolve(model_path)?;

    // Calibration source priority: an explicit imatrix GGUF (the flag's path,
    // a sibling auto-detect, or the in-file `imatrix` sidecar) outranks the
    // embedded `grim.calibration.scores` scores, which outrank recalibrating
    // from scratch. A sibling `.importance.json` still wins over the imatrix —
    // it is the most recent explicitly-produced calibration. Re-converting a
    // shipped single-file `.grim` on another machine must never silently
    // recalibrate when it already carries calibration data.
    let importance_scores = if Path::new(&format!("{}.importance.json", model_path)).exists() {
        load_importance_scores(&format!("{}.importance.json", model_path))?
    } else if let Some(imatrix_path) = imatrix_source_path(model_path, embed, &embed_attachments) {
        match grim_format::imatrix::read_imatrix_scores(&imatrix_path) {
            Ok(im) => {
                eprintln!(
                    "[grim convert] using imatrix '{}': {} measured tensors, chunks={:?}",
                    imatrix_path,
                    im.scores.len(),
                    im.chunk_count
                );
                let (names, scores): (Vec<String>, Vec<f32>) = im.scores.into_iter().unzip();
                ImportanceScores::new(names, scores)
            }
            Err(e) => {
                return Err(format!("imatrix '{imatrix_path}' failed to parse: {e}"))
            }
        }
    } else if let Some(scores) = grim_meta.calibration_scores.clone() {
        let (embedded_names, embedded_scores): (Vec<String>, Vec<f32>) =
            scores.into_iter().unzip();
        eprintln!(
            "[grim convert] using embedded calibration sidecar ({} tensors)",
            embedded_names.len()
        );
        ImportanceScores::new(embedded_names, embedded_scores)
    } else {
        cmd_oxidizer_calibrate(
            model_path,
            output_path,
            calibration_dataset.as_deref(),
            &mut progress,
        )?
    };

    let tensor_names = importance_scores.tensor_names.clone();
    let name_to_idx: std::collections::HashMap<&str, usize> = names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i))
        .collect();
    let imp_name_to_idx: std::collections::HashMap<&str, usize> = tensor_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i))
        .collect();

    let tensor_sizes = tensor_names
        .iter()
        .map(|name| {
            if let Some(&idx) = name_to_idx.get(name.as_str()) {
                sizes[idx]
            } else {
                0
            }
        })
        .collect::<Vec<usize>>();
    let bitwidths;
    {
        let mut cb = |done: usize, total: usize| {
            if let Some(p) = progress.as_deref_mut() {
                p("evopress", done, total);
            }
        };
        bitwidths = cmd_oxidizer_search(
            &importance_scores,
            &tensor_sizes,
            target_bpw,
            generations,
            Some(&mut cb),
        );
    }

    // Create full bitwidths array: scored tensors get EvoPress bitwidth, others get target_bpw.
    let default_bw = target_bpw.round() as u32;
    let full_bitwidths: Vec<u32> = names
        .iter()
        .map(|name| {
            if let Some(&idx) = imp_name_to_idx.get(name.as_str()) {
                bitwidths[idx]
            } else {
                default_bw
            }
        })
        .collect();

    grim_meta.magic = Some("grim-v1".into());
    grim_meta.quant_version = Some(OXIDIZER_VERSION);
    grim_meta.rocml_profile = rocml_profile
        .map(GrimRocmlProfile::parse)
        .unwrap_or(grim_meta.rocml_profile);
    grim_meta.wavefront_size = grim_meta.rocml_profile.wavefront_size();
    grim_meta.lds_size = Some(grim_meta.rocml_profile.lds_size());
    grim_meta.quant_method = Some("evopress-gptq-sequential".into());
    grim_meta.calibration_dataset = calibration_dataset.clone();
    // NO caller-side quant_overrides: `bitwidth_to_dtype` names Q4_K for a
    // 4-bpw assignment, but the packer's uniform tier emits grim's FLAT
    // 4-bit wave packing (a different byte layout the reader derives from
    // `base_bitwidth`), and 2-bpw tiers are GSQ-RCO. A caller override that
    // disagrees with the payload decodes correct bytes as the wrong scheme —
    // norms panicked, the embedding table failed its row check. The packers
    // in grim-format own the override list: they emit entries exactly for
    // the tensors whose scheme the reader cannot derive from the bitwidth.
    grim_meta.quant_overrides = Vec::new();
    // SIDECAR 1 — RCO allocation: the per-tensor assignment travels with the
    // file (the loose `.rco-allocation.txt` is an artifact dump; this is the
    // authoritative copy, validated against actual payloads at load).
    {
        let alloc: Vec<grim_format::gguf::GrimQuantOverride> = names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let bw = full_bitwidths.get(i).copied().unwrap_or(default_bw);
                let effective_bpw = if is_attention_projection(name) {
                    enforce_attention_precision(bw)
                } else {
                    bw
                };
                let dtype = if effective_bpw <= 2 {
                    GgufDType::GsqRco3p5
                } else {
                    GgufDType::F32
                };
                grim_format::gguf::GrimQuantOverride {
                    tensor_name: name.clone(),
                    effective_bpw,
                    override_dtype: dtype,
                    importance_score: importance_scores
                        .layer_scores
                        .get(
                            imp_name_to_idx
                                .get(name.as_str())
                                .copied()
                                .unwrap_or(i),
                        )
                        .copied()
                        .unwrap_or(0.0),
                    layout_hint: None,
                }
            })
            .collect();
        grim_meta.rco_allocation_provenance = Some(format!(
            "search target_bpw={:.2} tensors={}",
            target_bpw,
            alloc.len()
        ));
        grim_meta.rco_allocation = Some(alloc);
    }
    // SIDECAR 2 — calibration: per-tensor importance scores embedded so
    // re-quantization does not depend on a loose `.importance.json`.
    {
        let scores: Vec<(String, f32)> = importance_scores
            .tensor_names
            .iter()
            .enumerate()
            .map(|(_i, name)| {
                let score = imp_name_to_idx
                    .get(name.as_str())
                    .and_then(|&idx| importance_scores.layer_scores.get(idx))
                    .copied()
                    .unwrap_or(0.0);
                (name.clone(), score)
            })
            .collect();
        grim_meta.calibration_scores = Some(scores);
    }

    let resolved_gcn = rocml_profile.unwrap_or("gfx1100");
    // Wavefront size: explicit override wins, else the profile-derived
    // value already stamped above, else resolve from the GCN (RDNA → W32).
    let wave = wave_override.or_else(|| {
        if grim_meta.wavefront_size != 0 {
            Some(grim_format::WaveSize::from_width(grim_meta.wavefront_size))
        } else {
            None
        }
    });
    // GPU-first conversion path: try initializing ROCm device first, fallback to CPU
    let gpu_device = if use_gpu {
        match grim_backend_rocm::RocmDevice::try_new(0) {
            Ok(device) => {
                println!(
                    "[Grim Convert] ROCm GPU detected (device 0) — using GPU-accelerated dequantization pipeline."
                );
                Some(device)
            }
            Err(e) => {
                println!(
                    "[Grim Convert] Notice: ROCm GPU initialization failed ({e}) — falling back to multi-threaded CPU conversion."
                );
                None
            }
        }
    } else {
        None
    };

    if let Some(ref device) = gpu_device {
        grim_format::convert_to_grim_with_dequant(
            model_path,
            output_path,
            resolved_gcn,
            target_bpw,
            generations,
            calibration_dataset.as_deref(),
            None,
            Some(full_bitwidths),
            Some(grim_meta),
            target_format.map(|f| f.to_string()),
            wave,
            progress,
            device,
        )
        .map_err(|e| e.to_string())?;
    } else {
        grim_format::convert_to_grim(
            model_path,
            output_path,
            resolved_gcn,
            target_bpw,
            generations,
            calibration_dataset.as_deref(),
            None,
            Some(full_bitwidths),
            Some(grim_meta),
            target_format.map(|f| f.to_string()),
            wave,
            progress,
        )
        .map_err(|e| e.to_string())?;
    }

    // Embedded sidecars ride an end-of-file trailer appended to the finished
    // output, so every pre-attachment reader and tensor offset is untouched.
    if !embed_attachments.is_empty() {
        let mut out = fs::OpenOptions::new()
            .append(true)
            .open(output_path)
            .map_err(|e| format!("cannot reopen {output_path} for sidecars: {e}"))?;
        let n = grim_format::format::write_attachments(&mut out, &embed_attachments)
            .map_err(|e| e.to_string())?;
        eprintln!(
            "[grim convert] embedded {n} sidecar attachment(s) in-file: {}",
            embed_attachments
                .iter()
                .map(|(k, b)| format!("{k} ({} bytes)", b.len()))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
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
        grim.rocml_profile = GrimRocmlProfile::parse(profile);
        grim.wavefront_size = grim.rocml_profile.wavefront_size();
        grim.lds_size = Some(grim.rocml_profile.lds_size());
    }
    grim.calibration_dataset = dataset;
    if train {
        grim.train_quant_mode = GrimTrainQuantMode::parse(format);
        grim.train_fusion_ops = inferred_fusion_ops(&names);
        grim.quant_method
            .get_or_insert_with(|| "train-prepare".into());
    }
    write_grim_file(input_path, output_path, &grim, &HashMap::new(), &[])
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
        grim.rocml_profile = GrimRocmlProfile::parse(profile);
    }
    grim.wavefront_size = grim.rocml_profile.wavefront_size();
    grim.lds_size = Some(grim.rocml_profile.lds_size());
    grim.rocm_fusion_ops = inferred_fusion_ops(&names);
    grim.kv_layout_optimized = Some(rocm);
    grim.xnack_enabled = Some(false);
    grim.quant_method.get_or_insert_with(|| "rocm-fuse".into());
    write_grim_file(input_path, output_path, &grim, &HashMap::new(), &[])
}

fn inferred_fusion_ops(names: &[String]) -> Vec<GrimFusionOp> {
    let ir = build_transformer_ir(names.iter().map(String::as_str));
    ir.recommended_fusion_ops()
}

fn load_importance_scores(path: &str) -> Result<ImportanceScores, String> {
    let content = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let v: serde_json::Value = serde_json::from_str(&content).map_err(|e| e.to_string())?;
    let tensors = v["tensors"]
        .as_array()
        .ok_or("invalid cached importance format")?;
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


pub fn build_rewritten_tensors(
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

        let Some(target) = quant_format_for_bitwidth(effective_bw) else {
            continue;
        };

        let data = match materialize_f32(
            &raw.bytes,
            &raw.shape,
            provider.tensors().get(name).map(|t| t.dtype),
        ) {
            Ok(data) => data,
            Err(_) => continue,
        };

        let layer_importance = importance_scores
            .layer_scores
            .get(index)
            .copied()
            .unwrap_or(1.0);
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

        // Wavefront-tiled layout for ROCm attention projections
        if is_attention_projection(name) {
            let layout =
                resolve_weight_layout(name, grim_meta, grim_backend_rocm::WavefrontSize::W64);
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
                    curvature: Some(build_curvature(
                        &tiled,
                        layer_importance,
                        nwf * wf,
                        cpad,
                        calibration_batch,
                    )),
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
pub fn cmd_oxidizer_raven(
    model_path: &str,
    output_path: &str,
    target_bpw: f32,
    calibration_dataset: Option<&str>,
    mut progress: Option<&mut (dyn FnMut(&str, usize, usize) + Send + Sync)>,
) -> Result<(), String> {
    let (provider, names, _sizes, mut grim_meta) = open_provider(model_path)?;
    let importance_scores = if Path::new(&format!("{}.importance.json", model_path)).exists() {
        load_importance_scores(&format!("{}.importance.json", model_path))?
    } else {
        cmd_oxidizer_calibrate(model_path, output_path, calibration_dataset, &mut progress)?
    };

    let default_bw = target_bpw.round() as u32;
    let full_bitwidths: Vec<u32> = vec![default_bw; names.len()];

    let calibration_batch = CalibrationBatch::new(128);

    grim_meta.magic = Some("grim-v1".into());
    grim_meta.quant_version = Some(OXIDIZER_VERSION);
    grim_meta.quant_method = Some("raven-fp8-repack".into());
    grim_meta.calibration_dataset = calibration_dataset.map(String::from);

    // Re-open as GgufProvider for build_rewritten_tensors.
    let gguf_provider = GgufProvider::open(model_path).map_err(|e| e.to_string())?;
    let _ = provider; // open_provider already validated the file is a GGUF/.grim
    let rewritten = build_rewritten_tensors(
        &gguf_provider,
        &importance_scores,
        &full_bitwidths,
        &calibration_batch,
        Some(&grim_meta),
    )?;

    write_grim_file(model_path, output_path, &grim_meta, &rewritten, &[])
}

/// Compute curvature. Uses Fisher/GGN diagonal when calibration_batch has samples, else heuristic proxy.
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
    compute_fisher_diagonal(
        data,
        &calibration_batch.samples,
        rows,
        cols,
        calibration_batch.group_size,
    )
}

/// Fallback: heuristic curvature proxy using activation magnitude when no calibration data.
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
    attachments: &[(String, Vec<u8>)],
) -> Result<(), String> {
    let src = fs::File::open(src_path).map_err(|e| e.to_string())?;
    let mut src_reader = BufReader::new(src);
    let gguf = read_gguf(BufReader::new(
        fs::File::open(src_path).map_err(|e| e.to_string())?,
    ))
    .map_err(|e| e.to_string())?;

    let mut metadata = gguf.metadata.clone();
    metadata.extend(grim_meta.to_gguf_metadata());

    let dst = fs::File::create(dst_path).map_err(|e| e.to_string())?;
    let mut writer = BufWriter::new(dst);
    write_gguf(
        &mut writer,
        &gguf,
        &metadata,
        rewritten_tensors,
        &mut src_reader,
    )?;
    writer.flush().map_err(|e| e.to_string())?;
    // Embedded sidecars ride an end-of-file trailer so every pre-attachment
    // reader (and every existing tensor offset) is untouched.
    if !attachments.is_empty() {
        let mut file = writer.into_inner().map_err(|e| e.to_string())?;
        let n = grim_format::format::write_attachments(&mut file, attachments)
            .map_err(|e| e.to_string())?;
        file.flush().map_err(|e| e.to_string())?;
        eprintln!(
            "[grim convert] embedded {n} sidecar attachment(s) in-file: {}",
            attachments
                .iter()
                .map(|(k, b)| format!("{k} ({} bytes)", b.len()))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    Ok(())
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
    let mut rewritten_infos = Vec::with_capacity(gguf.tensors.len());
    for info in &gguf.tensors {
        let rewritten = rewritten_tensors.get(&info.name);
        let dtype = match rewritten {
            Some(r) => gguf_dtype_for_quant_format(r.target)?,
            None => info.dtype,
        };
        let updated = GgufTensorInfo {
            name: info.name.clone(),
            dims: info.dims.clone(),
            offset: current_offset,
            size_bytes: rewritten
                .map(|r| r.bytes.len() as u64)
                .unwrap_or(info.size_bytes),
            dtype,
        };
        current_offset += info.size_bytes;
        if let Some(rewritten) = rewritten {
            current_offset = updated.offset + rewritten.bytes.len() as u64;
        }
        rewritten_infos.push(updated);
    }

    writer
        .write_all(&0x4655_4747u32.to_le_bytes())
        .map_err(|e| e.to_string())?;
    writer
        .write_all(&gguf.version.to_le_bytes())
        .map_err(|e| e.to_string())?;
    writer
        .write_all(&(rewritten_infos.len() as u64).to_le_bytes())
        .map_err(|e| e.to_string())?;
    writer
        .write_all(&(metadata.len() as u64).to_le_bytes())
        .map_err(|e| e.to_string())?;

    let mut entries = metadata.iter().collect::<Vec<_>>();
    entries.sort_by(|(a, _), (b, _)| a.cmp(b));
    for (key, value) in entries {
        write_string(writer, key)?;
        write_value(writer, value)?;
    }
    for tensor in &rewritten_infos {
        write_tensor_info(writer, tensor)?;
    }

    let bytes_written = header_size + metadata_size + tensor_meta_size;
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
            return Err(format!(
                "tensor '{}' size mismatch while writing .grim",
                dst_info.name
            ));
        }
    }
    Ok(())
}

#[allow(dead_code)] // benchmark helper
fn quant_format_for_bitwidth(bw: u32) -> Option<QuantFormat> {
    match bw {
        8 => Some(QuantFormat::Q8_0),
        // RCO assigns 2 bpw to the least important tensors; the 2-bit tier
        // is the GSQ-RCO-3.5-bit format (tag 81), not Q2_K — 18 B/64-weight
        // blocks, GSQ codebook. Previously 2-bpw assignments silently skipped
        // their tensors here (None), leaving them un-quantized in the .grim.
        2 => Some(QuantFormat::GsqRco3p5),
        4 => Some(QuantFormat::Q4K),
        5 => Some(QuantFormat::Q5K),
        6 => Some(QuantFormat::Q6K),
        _ => None,
    }
}

fn gguf_dtype_for_quant_format(format: QuantFormat) -> Result<GgufDType, String> {
    match format {
        // GreyRaven 2:4 has no GGUF type tag. It is also not representable as
        // one: a GGUF k-quant type cannot express per-group survivor metadata.
        // Refusing here keeps the failure at the point the user named a format,
        // rather than later as a mis-typed checkpoint.
        QuantFormat::Fp8Sparse24 => {
            Err("GreyRaven 2:4 has no GGUF type tag and cannot be written by the oxidizer".into())
        }
        QuantFormat::Fp8Sparse24Hw => {
            Err("GreyRaven-HW (tag 673) is grim-native and not portable: use the .grim format".into())
        }
        // TreePie is grim-internal for now, like GreyRaven: it can be loaded and
        // run, but there is no GGUF type tag to write it under, so the oxidizer
        // refuses at the point the user named the format rather than emitting a
        // checkpoint no reader would interpret correctly.
        QuantFormat::TreePie => {
            Err("TreePie has no GGUF type tag and cannot be written by the oxidizer".into())
        }
        // WhiteRaven now HAS a GGUF tag -- 670, in grim's own block -- but the
        // tag is deliberately unreadable by any other tool. Writing it here
        // would emit a checkpoint that llama.cpp rejects as an unknown type,
        // which is a worse outcome than refusing at the point the user named
        // the format. Grim-native payloads go through the .grim format.
        QuantFormat::Fp8Blocked16 => Err(format!(
            "Fp8Blocked16 (WhiteRaven, grim-native GGUF tag {}) is not portable: a stock \
             GGUF reader rejects tag {} as unknown. Use the .grim format.",
            GgufDType::WhiteRaven.tag(),
            GgufDType::WhiteRaven.tag(),
        )),
        QuantFormat::Q8_0 => Ok(GgufDType::Q8_0),
        QuantFormat::Q2_0 => Ok(GgufDType::Q2_0),
        QuantFormat::GsqRco3p5 => Ok(GgufDType::GsqRco3p5),
        QuantFormat::Iq1S => Ok(GgufDType::IQ1_S),
        QuantFormat::Iq1M => Ok(GgufDType::IQ1_M),
        QuantFormat::Q2K => Ok(GgufDType::Q2K),
        QuantFormat::Q3K => Ok(GgufDType::Q3K),
        QuantFormat::Q4K => Ok(GgufDType::Q4K),
        QuantFormat::Q5K => Ok(GgufDType::Q5K),
        QuantFormat::Q6K => Ok(GgufDType::Q6K),
        QuantFormat::Iq4Nl => Ok(GgufDType::IQ4_NL),
        QuantFormat::Iq4Xs => Ok(GgufDType::IQ4_XS),
        QuantFormat::Iq3Xxs => Ok(GgufDType::IQ3_XXS),
        QuantFormat::Iq3S => Ok(GgufDType::IQ3_S),
        QuantFormat::Iq2Xxs => Ok(GgufDType::IQ2_XXS),
        QuantFormat::Iq2Xs => Ok(GgufDType::IQ2_XS),
        QuantFormat::Iq2S => Ok(GgufDType::IQ2_S),
        QuantFormat::Fp4
        | QuantFormat::Nf4
        | QuantFormat::Fp8
        | QuantFormat::Fp4Block16
        | QuantFormat::Fp8Block16
        | QuantFormat::Fp8Block128
        // WhiteCrow is a kernel-policy payload like WhiteRaven: tag 660 exists
        // and GrimProvider reads it, but a stock GGUF reader must not accept it.
        // ForestRaven's row-scaled framing has no fixed block geometry, so no
        // GGUF type can express it either: GrimProvider reads it from the
        // .grim entry's explicit payload size.
        | QuantFormat::W4A4OstQuant
        | QuantFormat::Int8PerChannel => Err(format!(
            "quantization format {:?} is not supported in GGUF writer",
            format
        )),
    }
}

#[allow(dead_code)] // benchmark helper
fn materialize_f32(
    bytes: &[u8],
    shape: &[usize],
    source_dtype: Option<GgufDType>,
) -> Result<Vec<f32>, String> {
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
        GgufDType::IQ4_NL => {
            grim_quant::dequant_iq4nl(bytes, elem_count).map_err(|e| e.to_string())
        }
        GgufDType::IQ4_XS => {
            grim_quant::dequant_iq4xs(bytes, elem_count).map_err(|e| e.to_string())
        }
        GgufDType::IQ3_XXS => {
            grim_quant::dequant_iq3xxs(bytes, elem_count).map_err(|e| e.to_string())
        }
        GgufDType::IQ3_S => grim_quant::dequant_iq3s(bytes, elem_count).map_err(|e| e.to_string()),
        GgufDType::IQ2_XXS => {
            grim_quant::dequant_iq2xxs(bytes, elem_count).map_err(|e| e.to_string())
        }
        GgufDType::IQ2_XS => {
            grim_quant::dequant_iq2xs(bytes, elem_count).map_err(|e| e.to_string())
        }
        GgufDType::IQ2_S => grim_quant::dequant_iq2s(bytes, elem_count).map_err(|e| e.to_string()),
        _ => Err(format!(
            "unsupported source dtype for Pass 4 materialization: {:?}",
            source_dtype
        )),
    }
}

fn write_tensor_info<W: Write>(writer: &mut W, tensor: &GgufTensorInfo) -> Result<(), String> {
    write_string(writer, &tensor.name)?;
    writer
        .write_all(&(tensor.dims.len() as u32).to_le_bytes())
        .map_err(|e| e.to_string())?;
    for dim in &tensor.dims {
        writer
            .write_all(&dim.to_le_bytes())
            .map_err(|e| e.to_string())?;
    }
    writer
        .write_all(&(tensor.dtype as u32).to_le_bytes())
        .map_err(|e| e.to_string())?;
    writer
        .write_all(&tensor.offset.to_le_bytes())
        .map_err(|e| e.to_string())
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
        GgufValue::Int8(v) => writer
            .write_all(&v.to_le_bytes())
            .map_err(|e| e.to_string()),
        GgufValue::Uint16(v) => writer
            .write_all(&v.to_le_bytes())
            .map_err(|e| e.to_string()),
        GgufValue::Int16(v) => writer
            .write_all(&v.to_le_bytes())
            .map_err(|e| e.to_string()),
        GgufValue::Uint32(v) => writer
            .write_all(&v.to_le_bytes())
            .map_err(|e| e.to_string()),
        GgufValue::Int32(v) => writer
            .write_all(&v.to_le_bytes())
            .map_err(|e| e.to_string()),
        GgufValue::Float32(v) => writer
            .write_all(&v.to_le_bytes())
            .map_err(|e| e.to_string()),
        GgufValue::Bool(v) => writer.write_all(&[*v as u8]).map_err(|e| e.to_string()),
        GgufValue::String(v) => write_string(writer, v),
        GgufValue::Array(values) => {
            let type_tag = values.first().map(value_type_tag).unwrap_or(8);
            writer
                .write_all(&type_tag.to_le_bytes())
                .map_err(|e| e.to_string())?;
            writer
                .write_all(&(values.len() as u64).to_le_bytes())
                .map_err(|e| e.to_string())?;
            for item in values {
                write_value_raw(writer, item)?;
            }
            Ok(())
        }
        GgufValue::Uint64(v) => writer
            .write_all(&v.to_le_bytes())
            .map_err(|e| e.to_string()),
        GgufValue::Int64(v) => writer
            .write_all(&v.to_le_bytes())
            .map_err(|e| e.to_string()),
        GgufValue::Float64(v) => writer
            .write_all(&v.to_le_bytes())
            .map_err(|e| e.to_string()),
    }
}

fn write_value<W: Write>(writer: &mut W, value: &GgufValue) -> Result<(), String> {
    let tag = value_type_tag(value);
    writer
        .write_all(&tag.to_le_bytes())
        .map_err(|e| e.to_string())?;
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
mod embed_sidecars_tests {
    use super::*;

    #[test]
    fn resolve_auto_detects_sibling_sidecars_in_probe_order() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("qwen3.grim");
        std::fs::write(&model, b"fake").unwrap();
        // Only the second probe pattern exists — the first candidate misses.
        let mmproj = dir.path().join("mmproj-qwen3.gguf");
        std::fs::write(&mmproj, b"MMPPROJ-BYTES").unwrap();
        let mtp = dir.path().join("qwen3-mtp.gguf");
        std::fs::write(&mtp, b"MTP-BYTES").unwrap();
        // imatrix: no sibling at all -> silently absent.

        let resolved = EmbedSidecars::default()
            .resolve(model.to_str().unwrap())
            .unwrap();
        let kinds: Vec<&str> = resolved.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(kinds, vec!["mmproj", "mtp"]);
        assert_eq!(resolved[0].1, b"MMPPROJ-BYTES".to_vec());
        assert_eq!(resolved[1].1, b"MTP-BYTES".to_vec());
    }

    #[test]
    fn resolve_explicit_missing_path_fails_loud() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("m.grim");
        std::fs::write(&model, b"fake").unwrap();
        let err = EmbedSidecars {
            mmproj: Some(dir.path().join("nope.gguf").to_string_lossy().into_owned()),
            ..Default::default()
        }
        .resolve(model.to_str().unwrap())
        .unwrap_err();
        assert!(err.contains("--embed-mmproj"), "{err}");
        assert!(err.contains("nope.gguf"), "{err}");
    }

    #[test]
    fn resolve_explicit_path_wins_over_sibling() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("m.grim");
        std::fs::write(&model, b"fake").unwrap();
        let sibling = dir.path().join("m-mmproj.gguf");
        std::fs::write(&sibling, b"SIBLING").unwrap();
        let explicit_dir = tempfile::tempdir().unwrap();
        let explicit = explicit_dir.path().join("other-mmproj.gguf");
        std::fs::write(&explicit, b"EXPLICIT").unwrap();

        let resolved = EmbedSidecars {
            mmproj: Some(explicit.to_string_lossy().into_owned()),
            ..Default::default()
        }
        .resolve(model.to_str().unwrap())
        .unwrap();
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].1, b"EXPLICIT".to_vec());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

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
            &[1.0f32; 32],
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
        write_gguf(
            &mut writer,
            &gguf,
            &gguf.metadata,
            &rewritten_tensors,
            &mut src_reader,
        )
        .unwrap();
        writer.flush().unwrap();

        let rewritten_file = read_gguf(BufReader::new(fs::File::open(&output).unwrap())).unwrap();
        assert_eq!(rewritten_file.tensors[0].dtype, GgufDType::Q8_0);

        let mut rewritten_reader = BufReader::new(fs::File::open(&output).unwrap());
        let rewritten_bytes = read_tensor_bytes(
            &mut rewritten_reader,
            &rewritten_file,
            &rewritten_file.tensors[0],
        )
        .unwrap();
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
        write_gguf(
            &mut writer,
            &gguf,
            &gguf.metadata,
            &HashMap::new(),
            &mut src,
        )?;
        writer.flush().map_err(|e| e.to_string())
    }
}
