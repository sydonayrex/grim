//! OSTQuant (W4A4) ingestion.
//! Checkpoints store `*.qweight` (u32), `*.scales` (bf16), and `*.zeros` (u8) tensors.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

use crate::safetensors::read_safetensors_header;
use grim_tensor::dtype::{ArithType, DType, OstQuantConfig, QuantProvenance, Storage};
use grim_tensor::error::{Error, Result};
use grim_tensor::provider::{RawTensor, TensorMeta, TensorProvider};

/// OSTQuant configuration parsed from config.json.
#[derive(Debug, Clone, Copy)]
pub struct OstQuantMetadata {
    pub group_size: usize,
}

impl OstQuantMetadata {
    pub fn from_json(path: &str) -> Result<Self> {
        let parent = std::path::Path::new(path)
            .parent()
            .unwrap_or(std::path::Path::new(""));
        let cfg_path = parent.join("config.json");
        let content = std::fs::read_to_string(&cfg_path).map_err(|e| {
            Error::Backend(format!("OSTQuant: cannot read {}: {e}", cfg_path.display()))
        })?;
        let val: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| Error::Backend(format!("OSTQuant: invalid config.json: {e}")))?;

        let ost_obj = val.get("ostquant_int4_packed").ok_or_else(|| {
            Error::Backend("OSTQuant: missing 'ostquant_int4_packed' in config.json".into())
        })?;

        let group_size = ost_obj
            .get("groupsize")
            .or_else(|| ost_obj.get("group_size"))
            .and_then(|v| v.as_u64())
            .unwrap_or(128) as usize;

        Ok(Self { group_size })
    }
}

/// Metadata for one quantized OSTQuant layer.
#[derive(Debug, Clone)]
pub struct OstQuantTensorInfo {
    pub name: String,
    /// Shape: [out_features, in_features]
    pub shape: Vec<usize>,
    pub group_size: usize,
    pub qweight_offset: u64,
    pub qweight_size: u64,
    pub scales_offset: u64,
    pub scales_size: u64,
    pub zeros_offset: u64,
    pub zeros_size: u64,
}

pub struct OstQuantProvider {
    pub tensors: HashMap<String, OstQuantTensorInfo>,
    reader: std::sync::Mutex<BufReader<File>>,
    data_region_start: u64,
    pub config: OstQuantMetadata,
}

impl OstQuantProvider {
    pub fn open(path: &str) -> Result<Self> {
        let resolved =
            std::fs::canonicalize(path).unwrap_or_else(|_| std::path::PathBuf::from(path));
        let config = OstQuantMetadata::from_json(&resolved.to_string_lossy())?;

        let (info, _metadata, data_region_start) =
            read_safetensors_header(BufReader::new(File::open(&resolved).map_err(|e| {
                Error::Backend(format!(
                    "cannot open OSTQuant file '{}': {e}",
                    resolved.display()
                ))
            })?))?;

        let mut tensors = HashMap::new();
        for (name, tensor_info) in &info {
            if !name.ends_with(".qweight") {
                continue;
            }
            let base_name = name.strip_suffix(".qweight").unwrap();

            let scales_name = format!("{base_name}.scales");
            let zeros_name = format!("{base_name}.zeros");
            let (Some(sc), Some(zr)) = (info.get(&scales_name), info.get(&zeros_name)) else {
                continue;
            };

            // OSTQuant qweight shape in safetensors: [N, K / 8]
            let qw_shape = tensor_info.shape();
            if qw_shape.len() != 2 {
                continue;
            }
            let out_features = qw_shape[0];
            let in_features = qw_shape[1] * 8;
            let shape = vec![out_features, in_features];

            let t_info = OstQuantTensorInfo {
                name: base_name.to_string(),
                shape,
                group_size: config.group_size,
                qweight_offset: tensor_info.data_start,
                qweight_size: tensor_info.data_end - tensor_info.data_start,
                scales_offset: sc.data_start,
                scales_size: sc.data_end - sc.data_start,
                zeros_offset: zr.data_start,
                zeros_size: zr.data_end - zr.data_start,
            };
            tensors.insert(base_name.to_string(), t_info.clone());
            tensors.insert(format!("{base_name}.weight"), t_info);
        }

        if tensors.is_empty() {
            return Err(Error::Backend(format!(
                "OSTQuant: no .qweight tensors found in '{}'",
                resolved.display()
            )));
        }

        let file = File::open(&resolved).map_err(|e| {
            Error::Backend(format!(
                "cannot reopen OSTQuant file '{}': {e}",
                resolved.display()
            ))
        })?;

        Ok(Self {
            tensors,
            reader: std::sync::Mutex::new(BufReader::new(file)),
            data_region_start,
            config,
        })
    }

    fn read_segment(&self, offset: u64, size: u64) -> Result<Vec<u8>> {
        let mut reader = self
            .reader
            .lock()
            .map_err(|_| Error::Backend("OSTQuant reader mutex poisoned".into()))?;
        let start = self.data_region_start + offset;
        reader.seek(SeekFrom::Start(start))?;
        let mut buf = vec![0u8; size as usize];
        reader.read_exact(&mut buf)?;
        Ok(buf)
    }
}

/// Packs qweight, scales, zeros into canonical blob:
/// `[u64 qw_len][qweight][u64 sc_len][scales][u64 zr_len][zeros]`
pub fn pack_ostquant_native(
    info: &OstQuantTensorInfo,
    qweight: &[u8],
    scales: &[u8],
    zeros: &[u8],
) -> Result<Vec<u8>> {
    let out_features = *info
        .shape
        .first()
        .ok_or_else(|| Error::Backend("OSTQuant: missing out_features".into()))?;
    let in_features = *info
        .shape
        .get(1)
        .ok_or_else(|| Error::Backend("OSTQuant: missing in_features".into()))?;

    let qw_expected = out_features * (in_features / 8) * 4;
    let n_groups = in_features.div_ceil(info.group_size);
    let sc_expected = out_features * n_groups * 2;
    let zr_expected = out_features * n_groups;

    if qweight.len() < qw_expected {
        return Err(Error::Backend(format!(
            "OSTQuant {}: qweight truncated ({} bytes, need {})",
            info.name,
            qweight.len(),
            qw_expected
        )));
    }
    if scales.len() < sc_expected {
        return Err(Error::Backend(format!(
            "OSTQuant {}: scales truncated ({} bytes, need {})",
            info.name,
            scales.len(),
            sc_expected
        )));
    }
    if zeros.len() < zr_expected {
        return Err(Error::Backend(format!(
            "OSTQuant {}: zeros truncated ({} bytes, need {})",
            info.name,
            zeros.len(),
            zr_expected
        )));
    }

    let mut out = Vec::with_capacity(24 + qw_expected + sc_expected + zr_expected);
    out.extend_from_slice(&(qw_expected as u64).to_le_bytes());
    out.extend_from_slice(&qweight[..qw_expected]);
    out.extend_from_slice(&(sc_expected as u64).to_le_bytes());
    out.extend_from_slice(&scales[..sc_expected]);
    out.extend_from_slice(&(zr_expected as u64).to_le_bytes());
    out.extend_from_slice(&zeros[..zr_expected]);
    Ok(out)
}

impl TensorProvider for OstQuantProvider {
    fn get(&self, name: &str) -> Result<RawTensor> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::Backend(format!("tensor '{name}' not found in OSTQuant file")))?;
        let qweight = self.read_segment(info.qweight_offset, info.qweight_size)?;
        let scales = self.read_segment(info.scales_offset, info.scales_size)?;
        let zeros = self.read_segment(info.zeros_offset, info.zeros_size)?;

        let f32s = grim_quant::dequant_ostquant_w4a4(
            &qweight,
            &scales,
            &zeros,
            &info.shape,
            info.group_size,
        )?;
        let mut bytes = Vec::with_capacity(f32s.len() * 4);
        for f in f32s {
            bytes.extend_from_slice(&f.to_le_bytes());
        }

        Ok(RawTensor {
            bytes,
            shape: info.shape.clone(),
            dtype: DType::F32,
            provenance: QuantProvenance::ExternalQat {
                bits: 4,
                group_size: info.group_size,
                scheme: grim_tensor::dtype::GroupQuantScheme::Asymmetric,
                desc_act: false,
            },
        })
    }

    fn get_packed(&self, name: &str) -> Result<RawTensor> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::Backend(format!("tensor '{name}' not found in OSTQuant file")))?;
        let qweight = self.read_segment(info.qweight_offset, info.qweight_size)?;
        let scales = self.read_segment(info.scales_offset, info.scales_size)?;
        let zeros = self.read_segment(info.zeros_offset, info.zeros_size)?;
        let bytes = pack_ostquant_native(info, &qweight, &scales, &zeros)?;
        Ok(RawTensor {
            bytes,
            shape: info.shape.clone(),
            dtype: DType {
                arith: ArithType::BF16,
                storage: Storage::W4A4OstQuant(OstQuantConfig {
                    group_size: info.group_size,
                }),
            },
            provenance: QuantProvenance::ExternalQat {
                bits: 4,
                group_size: info.group_size,
                scheme: grim_tensor::dtype::GroupQuantScheme::Asymmetric,
                desc_act: false,
            },
        })
    }

    fn meta(&self, name: &str) -> Result<TensorMeta> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::Backend(format!("tensor '{name}' not found in OSTQuant file")))?;
        Ok(TensorMeta {
            dtype: DType {
                arith: ArithType::BF16,
                storage: Storage::W4A4OstQuant(OstQuantConfig {
                    group_size: info.group_size,
                }),
            },
            provenance: QuantProvenance::ExternalQat {
                bits: 4,
                group_size: info.group_size,
                scheme: grim_tensor::dtype::GroupQuantScheme::Asymmetric,
                desc_act: false,
            },
            shape: info.shape.clone(),
            fusion_mask: 0,
        })
    }
}
