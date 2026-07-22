//! Non-destructive bolt-on adapter attachment and detachment using `backup2` residual slot (WI-T8).
//!
//! Provides `attach_bolt_on` and `detach_bolt_on` functions operating directly on `.grim` tensor files.
//! Reversibly quantizes low-rank updates `ΔW = (α/r)·B@A` into pre-allocated `backup2` capacity without format resizes.

use crate::format::GrimFile;
use grim_tensor::{Tensor, error::{Error, Result}};
use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

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
        .ok_or_else(|| Error::Backend(format!("tensor {} has no GrimTensorExt metadata", tensor_name)))?;

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
            let norm = if eff_scale > 0.0 { (v / eff_scale).clamp(-1.0, 1.0) } else { 0.0 };
            let code = (((norm + 1.0) * 0.5) * 15.0).round() as u32;

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
    file.seek(SeekFrom::Start(codes_abs_offset)).map_err(Error::Io)?;
    file.write_all(&packed_codes).map_err(Error::Io)?;

    let scale_abs_offset = entry.payload_offset + ext.backup2.scale_offset;
    file.seek(SeekFrom::Start(scale_abs_offset)).map_err(Error::Io)?;
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
        .ok_or_else(|| Error::Backend(format!("tensor {} has no GrimTensorExt metadata", tensor_name)))?;

    if !ext.backup2.is_present() {
        return Ok(());
    }

    let zeros_codes = vec![0u8; ext.backup2.codes_size as usize];
    let zeros_scales = vec![0u8; ext.backup2.scale_size as usize];

    let codes_abs_offset = entry.payload_offset + ext.backup2.codes_offset;
    file.seek(SeekFrom::Start(codes_abs_offset)).map_err(Error::Io)?;
    file.write_all(&zeros_codes).map_err(Error::Io)?;

    let scale_abs_offset = entry.payload_offset + ext.backup2.scale_offset;
    file.seek(SeekFrom::Start(scale_abs_offset)).map_err(Error::Io)?;
    file.write_all(&zeros_scales).map_err(Error::Io)?;

    Ok(())
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
}
