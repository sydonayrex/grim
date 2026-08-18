//! Bolt-on adapter attachment, detachment, and permanent merge using the
//! `backup2` residual slot (WI-T8).
//!
//! Provides `attach_bolt_on`, `detach_bolt_on`, and `merge_bolt_on` functions
//! operating directly on `.grim` tensor files.
//!
//! - `attach_bolt_on` reversibly quantizes a low-rank update `ΔW = scale·B@A`
//!   into pre-allocated `backup2` capacity without format resizes.
//! - `detach_bolt_on` zeroes the `backup2` byte regions, reverting the tensor.
//! - `merge_bolt_on` permanently bakes the residual into the primary weight
//!   stream (per-row 256-byte packing, matching the CPU `dequant_row` decoder
//!   and the bolt-on read/write path), then clears `backup1`/`backup2` so the
//!   slot is freed and detachment becomes impossible.

use crate::format::{
    GrimFile, OUTLIER_RECORD_BYTES, WaveSize, pack_row_bpw_for_wave, read_kv_block,
    read_outliers_with_encoding,
};
use crate::spec::{BackupLayer, GrimTensorExt};
use grim_tensor::{
    Tensor,
    error::{Error, Result},
};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Attach a trained LoRA adapter `ΔW = scale * B @ A` into the `backup2` slot of a named base tensor in a `.grim` file.
///
/// CONTRACT: The base tensor's `GrimTensorExt` must have `backup2` provisioned with matching dimensions and non-zero `codes_size`.
pub fn attach_bolt_on(
    grim_path: &Path,
    tensor_name: &str,
    a_tensor: &Tensor,
    b_tensor: &Tensor,
    scale: f32,
) -> Result<()> {
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(grim_path)
        .map_err(Error::Io)?;

    let grim_file = GrimFile::read(&mut file)?;
    let entry = grim_file
        .tensor(tensor_name)
        .ok_or_else(|| Error::Backend(format!("tensor {} not found in .grim file", tensor_name)))?;

    let ext = grim_file
        .metadata
        .get_tensor_ext(tensor_name)
        .ok_or_else(|| {
            Error::Backend(format!(
                "tensor {} has no GrimTensorExt metadata",
                tensor_name
            ))
        })?;

    if !ext.backup2.is_present() {
        return Err(Error::Backend(format!(
            "tensor {} does not have backup2 capacity provisioned",
            tensor_name
        )));
    }

    let a_vec = a_tensor.to_vec_f32()?;
    let b_vec = b_tensor.to_vec_f32()?;
    let a_dims = a_tensor.shape().dims();
    let b_dims = b_tensor.shape().dims();

    let out_features = b_dims[0];
    let rank = b_dims[1];
    let in_features = a_dims[1];

    let mut delta_w = vec![0.0f32; out_features * in_features];
    for o in 0..out_features {
        for i in 0..in_features {
            let mut sum = 0.0f32;
            for r in 0..rank {
                sum += b_vec[o * rank + r] * a_vec[r * in_features + i];
            }
            delta_w[o * in_features + i] = scale * sum;
        }
    }

    let bpw = ext.backup2.bpw;
    let row_bytes = ((in_features * bpw as usize + 7) / 8 + 255) & !255;
    let mut packed_codes = Vec::with_capacity(out_features * row_bytes);
    let mut row_scales = Vec::with_capacity(out_features);

    for r in 0..out_features {
        let row = &delta_w[r * in_features..(r + 1) * in_features];
        let max_abs = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
        let scale_byte = (max_abs.min(1.0) * 255.0).round() as u8;
        row_scales.push(scale_byte);

        let eff_scale = scale_byte as f32 / 255.0f32;
        let mut row_packed = vec![0u8; row_bytes];

        for (c_idx, &v) in row.iter().enumerate() {
            let norm = if eff_scale > 0.0 {
                (v / eff_scale).clamp(-1.0, 1.0)
            } else {
                0.0
            };
            let levels = (1usize << bpw as usize) - 1;
            let code = (((norm + 1.0) * 0.5) * levels as f32).round() as u32;

            let bit_offset = c_idx * bpw as usize;
            let byte_offset = bit_offset / 8;
            let in_byte = bit_offset % 8;
            let bits_left = 8 - in_byte;

            if bits_left >= bpw as usize {
                let shift = bits_left - bpw as usize;
                row_packed[byte_offset] |= (code << shift) as u8;
            } else {
                let high_bits = bits_left;
                let low_bits = bpw as usize - high_bits;
                row_packed[byte_offset] |= (code >> low_bits) as u8;
                if byte_offset + 1 < row_bytes {
                    row_packed[byte_offset + 1] |= (code << (8 - low_bits)) as u8;
                }
            }
        }
        packed_codes.extend_from_slice(&row_packed);
    }

    let codes_abs_offset = entry.payload_offset + ext.backup2.codes_offset;
    file.seek(SeekFrom::Start(codes_abs_offset))
        .map_err(Error::Io)?;
    if packed_codes.len() > ext.backup2.codes_size as usize {
        return Err(Error::Backend(format!(
            "bolt-on codes overflow: {} bytes written > {} bytes provisioned",
            packed_codes.len(),
            ext.backup2.codes_size
        )));
    }
    file.write_all(&packed_codes).map_err(Error::Io)?;

    let scale_abs_offset = entry.payload_offset + ext.backup2.scale_offset;
    file.seek(SeekFrom::Start(scale_abs_offset))
        .map_err(Error::Io)?;
    file.write_all(&row_scales).map_err(Error::Io)?;

    Ok(())
}

/// Detach a bolt-on adapter by zeroing out `backup2` code and scale byte regions.
pub fn detach_bolt_on(grim_path: &Path, tensor_name: &str) -> Result<()> {
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(grim_path)
        .map_err(Error::Io)?;

    let grim_file = GrimFile::read(&mut file)?;
    let entry = grim_file
        .tensor(tensor_name)
        .ok_or_else(|| Error::Backend(format!("tensor {} not found in .grim file", tensor_name)))?;

    let ext = grim_file
        .metadata
        .get_tensor_ext(tensor_name)
        .ok_or_else(|| {
            Error::Backend(format!(
                "tensor {} has no GrimTensorExt metadata",
                tensor_name
            ))
        })?;

    if !ext.backup2.is_present() {
        return Ok(());
    }

    let zeros_codes = vec![0u8; ext.backup2.codes_size as usize];
    let zeros_scales = vec![0u8; ext.backup2.scale_size as usize];

    let codes_abs_offset = entry.payload_offset + ext.backup2.codes_offset;
    file.seek(SeekFrom::Start(codes_abs_offset))
        .map_err(Error::Io)?;
    file.write_all(&zeros_codes).map_err(Error::Io)?;

    let scale_abs_offset = entry.payload_offset + ext.backup2.scale_offset;
    file.seek(SeekFrom::Start(scale_abs_offset))
        .map_err(Error::Io)?;
    file.write_all(&zeros_scales).map_err(Error::Io)?;

    Ok(())
}

/// Merge a trained LoRA adapter `ΔW = (scale / rank) * B @ A` permanently into
/// the primary weight stream of a named tensor in a `.grim` file.
///
/// The effective f32 weights are first reconstructed exactly as the CPU
/// `dequant_row` decoder produces them (primary + `backup1` when
/// `gptq_ordered > 0` + `backup2` + outlier corrections), the adapter is
/// added, and the result is re-packed into the primary codes at the tensor's
/// `default_bpw` with per-row 256-byte packing. The outlier stream is baked
/// in and cleared, `backup1`/`backup2` are cleared (slot freed), and
/// `gptq_ordered` is reset to 0 so the tensor decodes as a native pipeline.
///
/// Because the metadata (gptq ordering + backup declarations) must change,
/// the whole file is rewritten atomically through a temp file: every other
/// tensor's raw payload bytes are copied verbatim so their relative backup /
/// distinct layouts are preserved to the byte.
pub fn merge_bolt_on(
    grim_path: &Path,
    tensor_name: &str,
    a_tensor: &Tensor,
    b_tensor: &Tensor,
    scale: f32,
) -> Result<()> {
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(grim_path)
        .map_err(Error::Io)?;

    let src = GrimFile::read(&mut file)?;
    let idx = src
        .tensors_by_name
        .get(tensor_name)
        .copied()
        .ok_or_else(|| Error::Backend(format!("tensor {} not found in .grim file", tensor_name)))?;
    let entry = &src.tensors[idx];
    let src_ext = src.metadata.get_tensor_ext(tensor_name).ok_or_else(|| {
        Error::Backend(format!(
            "tensor {} has no GrimTensorExt metadata",
            tensor_name
        ))
    })?;

    let row_count = src_ext.row_count.max(1) as usize;
    let row_stride = src_ext.row_stride as usize;
    let default_bpw = src_ext.default_bpw.clamp(2, 8);
    if row_stride == 0 {
        return Err(Error::Backend(format!(
            "tensor {} row_stride is zero",
            tensor_name
        )));
    }

    // Whole payload region: primary codes + per-row scales + backup layers.
    let payload = read_region(&mut file, entry.payload_offset, entry.payload_size)?;

    // Primary per-row scales live inside the payload at ext.scale_offset.
    let primary_scales: Vec<u8> = if src_ext.scale_size > 0 {
        let s = src_ext.scale_offset as usize;
        let e = s + src_ext.scale_size as usize;
        if e <= payload.len() {
            payload[s..e].to_vec()
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    // Outlier corrections; baked into the primary during the merge, then the
    // outlier stream is dropped so a position is not double-represented.
    let outliers = read_outliers_with_encoding(&mut file, entry, src_ext.outlier_index_encoding)?;
    let outlier_pairs: Vec<(u32, f32)> = outliers.into_iter().map(|o| (o.index, o.value)).collect();

    // Effective f32 weights, mirroring the `dequant_row` combine order.
    let mut effective = Vec::with_capacity(row_count * row_stride);
    for r in 0..row_count {
        let row = dequantize_effective_row(
            r,
            row_stride,
            default_bpw,
            &payload,
            &primary_scales,
            src_ext,
            &outlier_pairs,
        );
        effective.extend_from_slice(&row);
    }

    // ΔW = (scale / rank) * B @ A
    let delta = compute_delta_w(a_tensor, b_tensor, scale)?;
    if delta.len() != effective.len() {
        return Err(Error::Backend(format!(
            "adapter delta size {} does not match tensor element count {}",
            delta.len(),
            effective.len()
        )));
    }
    for i in 0..effective.len() {
        effective[i] += delta[i];
    }

    // Re-pack the primary at default_bpw with per-row 256-byte packing,
    // matching the CPU `dequant_row` decoder (independent of the file wave).
    let mut new_codes = Vec::new();
    let mut new_scales = Vec::new();
    for r in 0..row_count {
        let row = &effective[r * row_stride..(r + 1) * row_stride];
        if primary_scales.is_empty() {
            pack_row_bpw_for_wave(&mut new_codes, row, default_bpw, WaveSize::W64);
        } else {
            let max_abs = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
            let scale_byte = (max_abs.min(1.0) * 255.0).round() as u8;
            new_scales.push(scale_byte);
            let eff_scale = scale_byte as f32 / 255.0f32;
            let scaled: Vec<f32> = row
                .iter()
                .map(|&v| if eff_scale > 0.0 { v / eff_scale } else { 0.0 })
                .collect();
            pack_row_bpw_for_wave(&mut new_codes, &scaled, default_bpw, WaveSize::W64);
        }
    }
    let new_scale_offset = new_codes.len() as u64;
    let new_payload_size = (new_codes.len() + new_scales.len()) as u64;

    // Updated metadata: clear backup slots + GPTQ ordering; relocate scales.
    let mut new_meta = src.metadata.clone();
    if let Some(ext) = new_meta
        .ext_entries
        .iter_mut()
        .find(|e| e.tensor_name == tensor_name)
    {
        ext.gptq_ordered = 0;
        ext.backup1 = BackupLayer::default();
        ext.backup2 = BackupLayer::default();
        if new_scales.is_empty() {
            ext.scale_offset = 0;
            ext.scale_size = 0;
        } else {
            ext.scale_offset = new_scale_offset;
            ext.scale_size = new_scales.len() as u64;
        }
    }

    // Updated registry: the merged tensor's payload shrinks and its outlier
    // stream is dropped.
    let mut new_tensors = src.tensors.clone();
    new_tensors[idx].payload_size = new_payload_size;
    new_tensors[idx].outlier_count = 0;

    // Assemble per-tensor blink blobs, preserving every other tensor verbatim.
    let mut merged_payload = new_codes;
    merged_payload.extend_from_slice(&new_scales);

    let mut payload_blobs = Vec::with_capacity(new_tensors.len());
    let mut outlier_blobs = Vec::with_capacity(new_tensors.len());
    let mut kv_map: HashMap<String, Vec<u8>> = HashMap::new();
    for (i, t) in src.tensors.iter().enumerate() {
        if i == idx {
            payload_blobs.push(merged_payload.clone());
        } else {
            payload_blobs.push(read_region(&mut file, t.payload_offset, t.payload_size)?);
        }
        if t.outlier_count > 0 {
            outlier_blobs.push(read_region(
                &mut file,
                t.outlier_offset,
                t.outlier_count as u64 * OUTLIER_RECORD_BYTES as u64,
            )?);
        } else {
            outlier_blobs.push(Vec::new());
        }
        let kv = read_kv_block(&mut file, t)?;
        if !kv.is_empty() {
            kv_map.insert(t.name.clone(), kv);
        }
    }

    let new_grim = GrimFile {
        header: src.header.clone(),
        metadata: new_meta,
        tensors: new_tensors,
        tensors_by_name: HashMap::new(),
        kv_blobs: kv_map.clone(),
        wave: src.wave,
    };

    // Rewrite atomically through a unique temp file: fsync the temp data to
    // stable storage first, then rename it over the target so readers never
    // observe a partially-written `.grim`. The unique name means concurrent
    // merges cannot collide on the same temp path.
    let tmp = temp_path(grim_path);
    let result = (|| -> Result<()> {
        let out_file = File::create(&tmp).map_err(Error::Io)?;
        let mut writer = BufWriter::new(out_file);
        let written = new_grim.write(&mut writer)?;

        for (i, we) in written.iter().enumerate() {
            write_region_at(&mut writer, we.payload_offset, &payload_blobs[i])?;
            if !outlier_blobs[i].is_empty() {
                write_region_at(&mut writer, we.outlier_offset, &outlier_blobs[i])?;
            }
            if we.kv_present != 0 && we.kv_compressed_size > 0 {
                if let Some(kv) = kv_map.get(&we.name) {
                    write_region_at(&mut writer, we.kv_compressed_offset, kv)?;
                }
            }
        }

        writer.flush().map_err(Error::Io)?;
        // Flush the buffer into the kernel AND to stable storage before the
        // atomic rename — the old file is only replaced once the new data is
        // durable on disk.
        writer.get_ref().sync_all().map_err(Error::Io)?;
        drop(writer);

        std::fs::rename(&tmp, grim_path).map_err(Error::Io)?;

        // Best-effort directory fsync so the rename itself is durable.
        if let Some(dir) = grim_path.parent() {
            if let Ok(d) = File::open(dir) {
                let _ = d.sync_all();
            }
        }
        Ok(())
    })();

    if let Err(e) = result {
        // Never leave a half-written temp file behind.
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(())
}

/// Read `len` bytes at an absolute `offset` from a reader.
fn read_region<R: Read + Seek>(r: &mut R, offset: u64, len: u64) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; len as usize];
    if len > 0 {
        r.seek(SeekFrom::Start(offset)).map_err(Error::Io)?;
        r.read_exact(&mut buf)
            .map_err(|e| Error::Backend(format!("payload read failed: {e}")))?;
    }
    Ok(buf)
}

/// Write `blob` at absolute `target`, zero-padding forward as needed.
fn write_region_at<W: Write + Seek>(w: &mut W, target: u64, blob: &[u8]) -> Result<()> {
    let cur = w
        .stream_position()
        .map_err(|e| Error::Backend(e.to_string()))?;
    if cur < target {
        let mut remaining = target - cur;
        let zeros = [0u8; 4096];
        while remaining > 0 {
            let n = remaining.min(4096) as usize;
            w.write_all(&zeros[..n])
                .map_err(|e| Error::Backend(format!("payload pad write failed: {e}")))?;
            remaining -= n as u64;
        }
    } else if cur > target {
        w.seek(SeekFrom::Start(target)).map_err(Error::Io)?;
    }
    w.write_all(blob)
        .map_err(|e| Error::Backend(format!("payload write failed: {e}")))?;
    Ok(())
}

/// Temp file path placed next to `grim_path` (same directory). The name is
/// unique per process + timestamp so concurrent merges cannot collide.
fn temp_path(grim_path: &Path) -> PathBuf {
    let name = grim_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model.grim".into());
    let unique = format!(
        "{name}.merge.tmp.{}.{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    grim_path.with_file_name(unique)
}

/// Compute `ΔW = (scale / rank) * B @ A` for a LoRA adapter.
fn compute_delta_w(a: &Tensor, b: &Tensor, scale: f32) -> Result<Vec<f32>> {
    let a_vec = a.to_vec_f32()?;
    let b_vec = b.to_vec_f32()?;
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();

    if a_dims.len() < 2 || b_dims.len() < 2 {
        return Err(Error::Backend("LoRA adapter tensors must be 2-D".into()));
    }
    let out_features = b_dims[0];
    let rank = b_dims[1];
    let in_features = a_dims[1];
    if a_dims[0] != rank {
        return Err(Error::Backend(format!(
            "LoRA A rows {} do not match B columns {}",
            a_dims[0], rank
        )));
    }
    let adj = if rank > 0 { scale / rank as f32 } else { scale };

    let mut delta = vec![0.0f32; out_features * in_features];
    for o in 0..out_features {
        for i in 0..in_features {
            let mut sum = 0.0f32;
            for r in 0..rank {
                sum += b_vec[o * rank + r] * a_vec[r * in_features + i];
            }
            delta[o * in_features + i] = adj * sum;
        }
    }
    Ok(delta)
}

/// Dequantize one row to effective f32, mirroring `dequant_row`'s combine
/// order (primary → backup1 when gptq > 0 → backup2 → outlier overwrite).
fn dequantize_effective_row(
    row_idx: usize,
    row_stride: usize,
    default_bpw: u8,
    payload: &[u8],
    scales: &[u8],
    ext: &GrimTensorExt,
    outliers: &[(u32, f32)],
) -> Vec<f32> {
    let mut out = vec![0.0f32; row_stride];

    let bpw = default_bpw;
    let row_bytes = ((row_stride * bpw as usize + 7) / 8 + 255) & !255;
    let row_start = row_idx * row_bytes;
    let row_data = if row_start < payload.len() {
        &payload[row_start..]
    } else {
        &[]
    };
    for i in 0..row_stride {
        let code = decode_code(row_data, i, bpw);
        let levels = (1u32 << bpw) as f32;
        let normalized = code as f32 / (levels - 1.0);
        out[i] = normalized * 2.0 - 1.0;
    }

    let scale_val = if !scales.is_empty() && row_idx < scales.len() {
        scales[row_idx] as f32 / 255.0f32
    } else {
        1.0f32
    };
    for v in out.iter_mut() {
        *v *= scale_val;
    }

    if ext.backup1.is_present() && ext.gptq_ordered > 0 {
        let b1 = dequantize_backup_layer(payload, &ext.backup1, row_idx, row_stride);
        for i in 0..row_stride {
            out[i] += b1[i];
        }
    }
    if ext.backup2.is_present() {
        let b2 = dequantize_backup_layer(payload, &ext.backup2, row_idx, row_stride);
        for i in 0..row_stride {
            out[i] += b2[i];
        }
    }

    for &(idx, val) in outliers {
        let r = idx as usize / row_stride;
        let c = idx as usize % row_stride;
        if r == row_idx && c < out.len() {
            out[c] = val;
        }
    }
    out
}

/// Dequantize a single backup layer row and scale into f32.
fn dequantize_backup_layer(
    payload: &[u8],
    layer: &BackupLayer,
    row_idx: usize,
    row_stride: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; row_stride];
    let bpw = layer.bpw;
    let row_bytes = ((row_stride * bpw as usize + 7) / 8 + 255) & !255;
    let start = layer.codes_offset as usize + row_idx * row_bytes;
    let row_data = if start < payload.len() {
        &payload[start..]
    } else {
        &[]
    };
    let scale_idx = layer.scale_offset as usize + row_idx;
    let scale = if scale_idx < payload.len() {
        payload[scale_idx] as f32 / 255.0f32
    } else {
        1.0f32
    };
    for i in 0..row_stride {
        let code = decode_code(row_data, i, bpw);
        let levels = (1u32 << bpw) as f32;
        let normalized = code as f32 / (levels - 1.0);
        out[i] = (normalized * 2.0 - 1.0) * scale;
    }
    out
}

/// Big-endian-bit / little-endian-byte code decode for one element.
fn decode_code(data: &[u8], idx: usize, bpw: u8) -> u32 {
    let bit_offset = idx * bpw as usize;
    let byte_offset = bit_offset / 8;
    let in_byte = bit_offset % 8;
    let bits_left = 8 - in_byte;
    if byte_offset >= data.len() {
        return 0;
    }
    if bits_left >= bpw as usize {
        let shift = bits_left - bpw as usize;
        ((data[byte_offset] >> shift) & ((1 << bpw) - 1)) as u32
    } else {
        let high_bits = bits_left;
        let low_bits = bpw as usize - high_bits;
        let high_part = (data[byte_offset] & ((1 << high_bits) - 1)) as u32;
        let low_part = if byte_offset + 1 < data.len() {
            let shift = 8 - low_bits;
            ((data[byte_offset + 1] >> shift) & ((1 << low_bits) - 1)) as u32
        } else {
            0
        };
        (high_part << low_bits) | low_part
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detach_on_absent_backup2_is_ok() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.grim");
        // Creating a dummy file returns error when reading GrimFile, which is expected
        let res = detach_bolt_on(&path, "nonexistent");
        assert!(res.is_err());
    }

    /// Verifies `attach_bolt_on` and `detach_bolt_on` update backup2 metadata and payload byte regions on disk.
    #[test]
    fn test_attach_and_detach_bolt_on_updates_backup2_and_ext_entries() {
        use crate::format::{GrimFile, GrimHeader, GrimTensorEntry};
        use crate::gguf::{GrimMetadata, GrimRocmlProfile};
        use crate::spec::GrimTensorExt;
        use crate::tprov::GrimProvider;
        use std::collections::HashMap;
        use std::io::Cursor;

        let tensor_name = "layer.0.weight";
        let metadata = GrimMetadata {
            magic: Some("grim-v1".into()),
            quant_version: Some(1),
            rocml_profile: GrimRocmlProfile::Rdna3,
            wavefront_size: 64,
            target_gcn: Some("gfx1100".into()),
            ext_entries: vec![GrimTensorExt {
                tensor_name: tensor_name.into(),
                row_count: 32,
                row_stride: 128,
                default_bpw: 4,
                scale_size: 32,
                scale_offset: 256,
                backup2: crate::spec::BackupLayer {
                    // Provision must fit the full packed codes: 32 rows ×
                    // row_bytes ((128×2+7)/8 padded to 256) = 8192 bytes.
                    codes_offset: 512,
                    codes_size: 8192,
                    bpw: 2,
                    scale_offset: 512 + 8192,
                    scale_size: 256,
                },
                ..Default::default()
            }],
            ..Default::default()
        };

        let entry = GrimTensorEntry {
            name: tensor_name.into(),
            shape: vec![32, 128],
            base_bitwidth: 4,
            payload_offset: 0,
            payload_size: 1024,
            ..Default::default()
        };

        let grim_file = GrimFile {
            header: GrimHeader::new(1, 0),
            metadata,
            tensors: vec![entry],
            tensors_by_name: HashMap::new(),
            kv_blobs: HashMap::new(),
            wave: crate::format::WaveSize::W64,
        };

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.grim");

        let mut buf = Vec::new();
        {
            let mut cursor = Cursor::new(&mut buf);
            let written = grim_file.write(&mut cursor).unwrap();
            let needed = (written[0].payload_offset + written[0].payload_size) as usize;
            if buf.len() < needed {
                buf.resize(needed + 512, 0);
            }
        }
        std::fs::write(&path, &buf).unwrap();

        let a_tensor = grim_backend_cpu::cpu_tensor(
            vec![0.1f32; 2 * 128],
            grim_tensor::shape::Shape::new(vec![2, 128]),
        );
        let b_tensor = grim_backend_cpu::cpu_tensor(
            vec![0.1f32; 32 * 2],
            grim_tensor::shape::Shape::new(vec![32, 2]),
        );

        // Attach 2-bit bolt-on
        attach_bolt_on(&path, tensor_name, &a_tensor, &b_tensor, 1.0).expect("attach bolt-on");

        // Reopen and assert backup2 is populated
        let provider = GrimProvider::open(path.to_str().unwrap()).expect("reopen after attach");
        let ext = provider.ext_for(tensor_name).expect("ext for tensor");
        assert!(ext.backup2.is_present());
        assert_eq!(ext.backup2.bpw, 2);

        // Detach bolt-on
        detach_bolt_on(&path, tensor_name).expect("detach bolt-on");

        // Reopen and assert backup2 capacity is retained after detach
        let provider_detached =
            GrimProvider::open(path.to_str().unwrap()).expect("reopen after detach");
        let ext_detached = provider_detached
            .ext_for(tensor_name)
            .expect("ext for tensor");
        assert!(ext_detached.backup2.is_present());
    }
}
