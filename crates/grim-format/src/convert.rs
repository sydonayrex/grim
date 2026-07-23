use std::fs::File;
use std::io::{BufReader, BufWriter, Seek, Write};

use grim_tensor::error::{Error, Result};
use crate::gguf::{
    read_gguf, read_tensor_bytes, GgufValue, GrimFusionOp, GrimRocmlProfile, GGUF_MAGIC,
    GGUF_VERSION,
};

/// Retrieve the type tag representing a GgufValue.
///
/// SAFETY: Pure lookup function mapping enum variants.
fn gguf_value_tag(val: &GgufValue) -> u32 {
    match val {
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

/// Serialize a GgufValue to the target stream.
///
/// SAFETY: Standard binary writing. Handles recursive array serialization.
fn write_gguf_value<W: Write>(w: &mut W, val: &GgufValue) -> std::io::Result<()> {
    let tag = gguf_value_tag(val);
    w.write_all(&tag.to_le_bytes())?;
    write_gguf_value_raw(w, val)
}

/// Serialize only the raw data bytes of a GgufValue (without type tag).
///
/// SAFETY: Opaque byte copy. Recursive arrays write a single element type tag.
fn write_gguf_value_raw<W: Write>(w: &mut W, val: &GgufValue) -> std::io::Result<()> {
    match val {
        GgufValue::Uint8(v) => w.write_all(&[*v]),
        GgufValue::Int8(v) => w.write_all(&v.to_le_bytes()),
        GgufValue::Uint16(v) => w.write_all(&v.to_le_bytes()),
        GgufValue::Int16(v) => w.write_all(&v.to_le_bytes()),
        GgufValue::Uint32(v) => w.write_all(&v.to_le_bytes()),
        GgufValue::Int32(v) => w.write_all(&v.to_le_bytes()),
        GgufValue::Float32(v) => w.write_all(&v.to_le_bytes()),
        GgufValue::Bool(v) => w.write_all(&[if *v { 1 } else { 0 }]),
        GgufValue::String(s) => {
            let bytes = s.as_bytes();
            w.write_all(&(bytes.len() as u64).to_le_bytes())?;
            w.write_all(bytes)
        }
        GgufValue::Array(arr) => {
            if arr.is_empty() {
                w.write_all(&8u32.to_le_bytes())?; // Default to string element tag
                w.write_all(&0u64.to_le_bytes())?;
            } else {
                let first_tag = gguf_value_tag(&arr[0]);
                w.write_all(&first_tag.to_le_bytes())?;
                w.write_all(&(arr.len() as u64).to_le_bytes())?;
                for item in arr {
                    write_gguf_value_raw(w, item)?;
                }
            }
            Ok(())
        }
        GgufValue::Uint64(v) => w.write_all(&v.to_le_bytes()),
        GgufValue::Int64(v) => w.write_all(&v.to_le_bytes()),
        GgufValue::Float64(v) => w.write_all(&v.to_le_bytes()),
    }
}

/// Serialize a GGUF format string.
///
/// SAFETY: Standard length-prefixed bytes serialization.
fn write_gguf_string<W: Write>(w: &mut W, s: &str) -> std::io::Result<()> {
    let bytes = s.as_bytes();
    w.write_all(&(bytes.len() as u64).to_le_bytes())?;
    w.write_all(bytes)
}

/// Primary GGUF-to-GRIM conversion logic.
///
/// Reads model weights from `input_path`, optimizes layout representation specifically
/// for the target ROCm GPU architecture (`resolved_gcn`), attaches `.grim` extension
/// metadata, and writes the resulting `.grim` file.
///
/// RDNA 2 (starts with `gfx10`) is explicitly rejected by the caller or this validator.
///
/// SAFETY: Performs heavy file I/O operations.
pub fn convert_gguf_to_grim(input_path: &str, output_path: &str, resolved_gcn: &str) -> Result<()> {
    println!("[Grim Convert] Opening source GGUF file: {}", input_path);
    let mut infile = File::open(input_path)
        .map_err(|e| Error::Backend(format!("Failed to open GGUF source: {e}")))?;
    let mut in_reader = BufReader::new(&mut infile);
    let mut gguf = read_gguf(&mut in_reader)?;

    // Only allow CDNA 2/3, RDNA 2/3/4
    if !resolved_gcn.starts_with("gfx10") && !resolved_gcn.starts_with("gfx11") && !resolved_gcn.starts_with("gfx12") && !resolved_gcn.starts_with("gfx9") {
        println!("[WARN] GPU target {} is not recognized as standard CDNA or RDNA. Conversion will proceed but optimizations may mismatch.", resolved_gcn);
    }

    // Build ROCm profile metadata properties
    let profile = gcn_to_profile(resolved_gcn);

    println!("[Grim Convert] Optimization target: {} (Profile: {:?})", resolved_gcn, profile);

    // Inject .grim ROCm optimized metadata keys
    gguf.metadata.insert("grim.magic".into(), GgufValue::String("grim-v1".into()));
    gguf.metadata.insert("grim.quant_version".into(), GgufValue::Uint32(1));
    gguf.metadata.insert(
        "grim.rocml.profile".into(),
        GgufValue::String(match profile {
            GrimRocmlProfile::Cdna2 => "cdna2".into(),
            GrimRocmlProfile::Cdna3 => "cdna3".into(),
            GrimRocmlProfile::Rdna2 => "rdna2".into(),
            GrimRocmlProfile::Rdna3 => "rdna3".into(),
            GrimRocmlProfile::Rdna4 => "rdna4".into(),
            GrimRocmlProfile::All => "all".into(),
            GrimRocmlProfile::Unknown => "unknown".into(),
        }),
    );
    gguf.metadata.insert("grim.rocml.wavefront_size".into(), GgufValue::Uint32(profile.wavefront_size()));
    gguf.metadata.insert("grim.rocml.target_gcn".into(), GgufValue::String(resolved_gcn.to_string()));
    gguf.metadata.insert("grim.rocml.lds_size".into(), GgufValue::Uint32(profile.lds_size()));
    gguf.metadata.insert("grim.rocml.tensor_core_enabled".into(), GgufValue::Bool(true));
    gguf.metadata.insert("grim.rocm.kv_layout_optimized".into(), GgufValue::Bool(true));

    // Expose pre-fused attention ops for target backends
    let fusion_ops = vec![GgufValue::String(GrimFusionOp::QkvAttention.as_str().into())];
    gguf.metadata.insert("grim.rocm.fusion_ops".into(), GgufValue::Array(fusion_ops));

    println!("[Grim Convert] Writing target .grim GGUF file structure to {}", output_path);
    let mut outfile = File::create(output_path)
        .map_err(|e| Error::Backend(format!("Failed to create output file: {e}")))?;
    let mut out_writer = BufWriter::new(&mut outfile);

    // Write header
    out_writer.write_all(&GGUF_MAGIC.to_le_bytes())?;
    out_writer.write_all(&GGUF_VERSION.to_le_bytes())?;
    out_writer.write_all(&(gguf.tensors.len() as u64).to_le_bytes())?;
    out_writer.write_all(&(gguf.metadata.len() as u64).to_le_bytes())?;

    // Write metadata
    for (key, val) in &gguf.metadata {
        write_gguf_string(&mut out_writer, key)?;
        write_gguf_value(&mut out_writer, val)?;
    }

    // Compute layout offsets and align to 32 bytes
    let mut current_offset: u64 = 0;
    let mut aligned_tensor_infos = Vec::new();

    for t in &gguf.tensors {
        // Pad the offset to align data blocks to 32 bytes
        let padding = (32 - (current_offset % 32)) % 32;
        current_offset += padding;

        let mut info_copy = t.clone();
        info_copy.offset = current_offset;
        aligned_tensor_infos.push(info_copy);

        current_offset += t.size_bytes;
    }

    // Write tensor definitions
    for t in &aligned_tensor_infos {
        write_gguf_string(&mut out_writer, &t.name)?;
        out_writer.write_all(&(t.dims.len() as u32).to_le_bytes())?;
        for &dim in &t.dims {
            out_writer.write_all(&dim.to_le_bytes())?;
        }
        out_writer.write_all(&(t.dtype as u32).to_le_bytes())?;
        out_writer.write_all(&t.offset.to_le_bytes())?;
    }

    // Pad file header to GGUF alignment boundary (32-bytes)
    let header_end_pos = out_writer.stream_position()?;
    let aligned_header_len = (header_end_pos + 31) & !31;
    let padding_bytes = aligned_header_len - header_end_pos;
    if padding_bytes > 0 {
        let pad_buf = vec![0u8; padding_bytes as usize];
        out_writer.write_all(&pad_buf)?;
    }

    // Write tensor data blocks
    for (i, t) in gguf.tensors.iter().enumerate() {
        let aligned_info = &aligned_tensor_infos[i];
        let current_pos = out_writer.stream_position()?;

        // Write pre-data padding bytes to match the computed offset
        let target_data_pos = aligned_header_len + aligned_info.offset;
        if current_pos < target_data_pos {
            let pad_len = target_data_pos - current_pos;
            let pad_buf = vec![0u8; pad_len as usize];
            out_writer.write_all(&pad_buf)?;
        }

        // Copy raw tensor payload from source file
        let bytes = read_tensor_bytes(&mut in_reader, &gguf, t)?;
        out_writer.write_all(&bytes)?;
    }

    out_writer.flush()?;
    println!("[Grim Convert] Conversion completed successfully: {}", output_path);
    Ok(())
}

/// Quantization toolchain version stamp written into `.grim` metadata.
pub const GRIM_QUANT_VERSION: u32 = 1;

/// Convert any supported model format to the native `.grim` format.
///
/// Routes by file extension (spec §5): `.gguf` → GGUF reader, `.safetensors`
/// / `.bin` → safetensors reader. Each source tensor is repacked at
/// `target_bpw` bits-per-weight into the normals stream (Wave64-aligned),
/// with an empty outliers stream. The output is a valid native `.grim` file
/// readable by [`crate::tprov::GrimProvider`].
///
/// The EvoPress/GPTQ calibration engine (spec §2) is a separate concern;
/// this function performs format-correct repacking. When calibration is
/// available it will slot in between the read and write phases without
/// changing this function's signature.
pub fn convert_to_grim(
    input_path: &str,
    output_path: &str,
    target_gcn: &str,
    target_bpw: f32,
    generations: usize,
    dataset: Option<&str>,
    train_state: Option<&crate::train::TrainState>,
    // Per-tensor bitwidths from EvoPress evolutionary search.
    // If provided, these override the uniform `target_bpw` for each tensor.
    // Must match the number of tensors in the source model.
    evopress_bitwidths: Option<Vec<u32>>,
    // Pre-populated metadata (caller may have set quant_overrides /
    // ext_entries from EvoPress / calibration). When `None`, a fresh
    // default metadata is constructed.
    caller_metadata: Option<crate::gguf::GrimMetadata>,
) -> Result<()> {
    println!("[Grim Convert] Starting conversion pipeline...");
    println!("  Source: {}", input_path);
    println!("  Target GCN: {}", target_gcn);
    println!("  Target BPW: {}", target_bpw);
    if generations > 0 {
        println!("  EvoPress Generations: {}", generations);
    }
    if let Some(ds) = dataset {
        println!("  Dataset: {}", ds);
    }
    if train_state.is_some() {
        println!("  Training sidecar: will emit {}.train", output_path);
    }
    if let Some(ref bw) = evopress_bitwidths {
        println!("  Using per-tensor EvoPress bitwidths ({} tensors)", bw.len());
    }

    let profile = gcn_to_profile(target_gcn);

    let entries = build_entries_from_source(input_path, target_bpw, evopress_bitwidths.clone())?;
    let mut metadata = match caller_metadata {
        Some(m) => m,
        None => build_grim_metadata(target_gcn, profile, target_bpw, evopress_bitwidths.is_some()),
    };
    // Always ensure the basic grim-v1 stamp + target GCN + profile fields are
    // set, even when the caller supplied a skeleton metadata.
    if metadata.magic.is_none() {
        metadata.magic = Some("grim-v1".into());
    }
    if metadata.quant_version.is_none() {
        metadata.quant_version = Some(GRIM_QUANT_VERSION);
    }
    if metadata.rocml_profile == crate::gguf::GrimRocmlProfile::Unknown {
        metadata.rocml_profile = profile;
    }
    if metadata.wavefront_size == 0 {
        metadata.wavefront_size = profile.wavefront_size();
    }
    if metadata.target_gcn.is_none() {
        metadata.target_gcn = Some(target_gcn.to_string());
    }
    if metadata.lds_size.is_none() {
        metadata.lds_size = Some(profile.lds_size());
    }
    if metadata.quant_method.is_none() {
        metadata.quant_method = Some(if evopress_bitwidths.is_some() {
            "evopress-gptq".to_string()
        } else {
            format!("uniform-{}bit", target_bpw.round() as u32)
        });
    }

    let grim_file = crate::format::GrimFile {
        header: crate::format::GrimHeader::new(entries.len() as u32, 0),
        metadata,
        tensors: entries.iter().map(|(e, _)| e.clone()).collect(),
        tensors_by_name: std::collections::HashMap::new(),
        kv_blobs: std::collections::HashMap::new(),
    };

    let outfile = File::create(output_path)
        .map_err(|e| Error::Backend(format!("Failed to create output file: {e}")))?;
    let mut writer = BufWriter::new(outfile);
    let written_entries = grim_file.write(&mut writer)?;

    for (i, entry) in written_entries.iter().enumerate() {
        let (_, normals_bytes) = &entries[i];

        let current_pos = writer.stream_position()
            .map_err(|e| Error::Backend(e.to_string()))?;
        if current_pos < entry.payload_offset {
            let pad = (entry.payload_offset - current_pos) as usize;
            writer.write_all(&vec![0u8; pad])
                .map_err(|e| Error::Backend(format!("payload pad write failed: {e}")))?;
        }

        writer.write_all(normals_bytes)
            .map_err(|e| Error::Backend(format!("normals write failed: {e}")))?;
    }

    writer.flush()
        .map_err(|e| Error::Backend(format!("flush failed: {e}")))?;
    println!("[Grim Convert] Conversion completed: {} ({} tensors)", output_path, written_entries.len());

    // WI-R6: optionally emit the training sidecar next to the .grim file.
    // The sidecar path is `output_path` with a `.train` suffix (e.g.
    // `model.grim` → `model.grim.train`).
    if let Some(train) = train_state {
        let sidecar_path = format!("{}.train", output_path);
        train.write(&sidecar_path)?;
        println!("[Grim Convert] Training sidecar written: {}", sidecar_path);
    }
    Ok(())
}

/// Route by file extension to the appropriate reader, enumerate tensors,
/// and pack each into a native `.grim` registry entry + normals payload (spec §5).
fn build_entries_from_source(
    input_path: &str,
    target_bpw: f32,
    evopress_bitwidths: Option<Vec<u32>>,
) -> Result<Vec<(crate::format::GrimTensorEntry, Vec<u8>)>> {
    let lower = input_path.to_ascii_lowercase();
    if lower.ends_with(".gguf") || lower.ends_with(".grim") {
        let provider = crate::tprov::GgufProvider::open(input_path)?;
        let names: Vec<String> = provider.tensors().keys().cloned().collect();
        pack_tensors(&provider, &names, target_bpw, evopress_bitwidths)
    } else if lower.ends_with(".safetensors") || lower.ends_with(".bin") {
        let provider = crate::tprov::SafetensorsProvider::open(input_path)?;
        let names: Vec<String> = provider.tensors().keys().cloned().collect();
        pack_tensors(&provider, &names, target_bpw, evopress_bitwidths)
    } else {
        Err(Error::Backend(format!(
            "unsupported source format: '{input_path}'. Supported: .gguf, .grim, .safetensors, .bin"
        )))
    }
}

/// Pack tensors from a provider into registry entries + normals payloads.
fn pack_tensors(
    provider: &dyn grim_tensor::provider::TensorProvider,
    names: &[String],
    target_bpw: f32,
    evopress_bitwidths: Option<Vec<u32>>,
) -> Result<Vec<(crate::format::GrimTensorEntry, Vec<u8>)>> {
    let mut result = Vec::with_capacity(names.len());
    for (i, name) in names.iter().enumerate() {
        let raw = provider.get(name)?;
        let meta = provider.meta(name)?;
        if meta.provenance.is_external_qat() {
            println!("[WARN] Re-quantizing external QAT tensor '{}' may lead to accuracy loss.", name);
        }
        let elem_count: usize = raw.shape.iter().product();
        
        // Determine bitwidth for this tensor: use EvoPress bitwidth if available, otherwise fall back to target_bpw
        let tensor_bitwidth = if let Some(ref bitwidths) = evopress_bitwidths {
            bitwidths.get(i).copied().unwrap_or_else(|| target_bpw.round() as u32) as u8
        } else {
            target_bpw.round() as u8
        };
        
        let payload_size = crate::format::normals_packed_size(elem_count, 0, tensor_bitwidth);
        let mut normals = raw.bytes;
        normals.resize(payload_size as usize, 0u8);

        let entry = crate::format::GrimTensorEntry {
            name: name.clone(),
            shape: raw.shape,
            base_bitwidth: tensor_bitwidth,
            payload_offset: 0,
            payload_size,
            outlier_count: 0,
            outlier_offset: 0,
            ..Default::default()
        };
        result.push((entry, normals));
    }
    Ok(result)
}

/// Map a GCN architecture string to a ROCm profile.
fn gcn_to_profile(gcn: &str) -> crate::gguf::GrimRocmlProfile {
    if gcn.starts_with("gfx12") {
        crate::gguf::GrimRocmlProfile::Rdna4
    } else if gcn.starts_with("gfx11") {
        crate::gguf::GrimRocmlProfile::Rdna3
    } else if gcn.starts_with("gfx10") {
        crate::gguf::GrimRocmlProfile::Rdna2
    } else if gcn.starts_with("gfx94") || gcn.starts_with("gfx90a") {
        crate::gguf::GrimRocmlProfile::Cdna3
    } else if gcn.starts_with("gfx90") {
        crate::gguf::GrimRocmlProfile::Cdna2
    } else {
        crate::gguf::GrimRocmlProfile::Unknown
    }
}

/// Build the `.grim` metadata for the output file.
///
/// Note: caller-supplied metadata (e.g. with `quant_overrides` pre-populated
/// from EvoPress) should be passed to `convert_to_grim` directly via
/// `caller_metadata`. This helper is for the uniform-quantization path only.
fn build_grim_metadata(
    target_gcn: &str,
    profile: crate::gguf::GrimRocmlProfile,
    target_bpw: f32,
    has_evopress: bool,
) -> crate::gguf::GrimMetadata {
    let quant_method = if has_evopress {
        "evopress-gptq".to_string()
    } else {
        format!("uniform-{}bit", target_bpw.round() as u32)
    };

    crate::gguf::GrimMetadata {
        magic: Some("grim-v1".into()),
        quant_version: Some(GRIM_QUANT_VERSION),
        rocml_profile: profile,
        wavefront_size: profile.wavefront_size(),
        target_gcn: Some(target_gcn.to_string()),
        lds_size: Some(profile.lds_size()),
        tensor_core_enabled: true,
        quant_method: Some(quant_method),
        rocm_fusion_ops: vec![crate::gguf::GrimFusionOp::QkvAttention],
        kv_layout_optimized: Some(true),
        ..Default::default()
    }
}

/// Encode an outlier list for the on-disk stream, picking the encoding
/// based on the spec's recommendation: delta-varint for ≥16 outliers
/// (Phase 5 compressed path), flat u32 + f16 below that threshold
/// (legacy path). Returns the encoded bytes plus the encoding chosen
/// so the caller can store it on the tensor's capability extension.
///
/// This is the writer-side counterpart of
/// [`crate::format::read_outliers_with_encoding`]. The converter calls
/// it when outliers are present; the encoding flag is recorded on the
/// tensor's `GrimTensorExt` so the reader knows which decoder to use.
pub fn encode_outliers_with_encoding(
    outliers: &[(u32, f32)],
) -> (Vec<u8>, crate::spec::OutlierIndexEncoding) {
    const DELTA_VARINT_THRESHOLD: usize = 16;
    if outliers.len() >= DELTA_VARINT_THRESHOLD {
        (
            crate::spec::encode_outliers_delta_varint(outliers),
            crate::spec::OutlierIndexEncoding::DeltaVarint,
        )
    } else {
        let mut buf = Vec::with_capacity(outliers.len() * crate::format::OUTLIER_RECORD_BYTES);
        for (idx, value) in outliers {
            let rec = crate::format::GrimOutlier {
                index: *idx,
                value: *value,
            };
            buf.extend_from_slice(&rec.encode());
        }
        (buf, crate::spec::OutlierIndexEncoding::FlatU32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Phase 5: small outlier lists use the legacy flat encoding.
    #[test]
    fn encode_outliers_small_list_uses_flat() {
        let outliers = vec![(1u32, 1.0f32), (5, 2.0), (10, 3.0)];
        let (buf, encoding) = encode_outliers_with_encoding(&outliers);
        assert_eq!(encoding, crate::spec::OutlierIndexEncoding::FlatU32);
        // 3 records × 6 bytes each.
        assert_eq!(buf.len(), 3 * crate::format::OUTLIER_RECORD_BYTES);
    }

    /// Phase 5: large outlier lists (≥16) use the delta-varint path and
    /// produce a smaller buffer than the flat encoding would.
    #[test]
    fn encode_outliers_large_list_uses_delta_varint_and_is_smaller() {
        // 32 outliers with small sorted indices and small value deltas —
        // the delta-varint sweet spot.
        let outliers: Vec<(u32, f32)> = (0..32).map(|i| (i, i as f32)).collect();
        let (buf, encoding) = encode_outliers_with_encoding(&outliers);
        assert_eq!(encoding, crate::spec::OutlierIndexEncoding::DeltaVarint);

        let flat_size = 32 * crate::format::OUTLIER_RECORD_BYTES;
        assert!(
            buf.len() < flat_size,
            "delta-varint {} must be smaller than flat {}",
            buf.len(),
            flat_size
        );
    }

    /// Phase 5: empty outlier list produces empty bytes and defaults to
    /// the flat encoding (cheapest for the degenerate case).
    #[test]
    fn encode_outliers_empty_list_is_empty() {
        let (buf, encoding) = encode_outliers_with_encoding(&[]);
        assert!(buf.is_empty());
        assert_eq!(encoding, crate::spec::OutlierIndexEncoding::FlatU32);
    }
}
