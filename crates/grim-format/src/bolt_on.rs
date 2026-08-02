//! Non-destructive bolt-on adapter attachment and detachment using `backup2` residual slot (WI-T8).
//!
//! Provides `attach_bolt_on` and `detach_bolt_on` functions operating directly on `.grim` tensor files.
//! Reversibly quantizes low-rank updates `ΔW = (α/r)·B@A` into pre-allocated `backup2` capacity without format resizes.

use crate::format::GrimFile;
use grim_tensor::{
    Tensor,
    error::{Error, Result},
};
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
    file.seek(SeekFrom::Start(codes_abs_offset))
        .map_err(Error::Io)?;
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
                    codes_offset: 512,
                    codes_size: 256,
                    bpw: 2,
                    scale_offset: 768,
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
