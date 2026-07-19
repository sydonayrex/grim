//! `TensorProvider` implementations for GGUF, native `.grim`, and safetensors files.
//!
//! Each implements `TensorProvider` so `WeightSource` can walk checkpoints
//! without caring whether they came from GGUF, native `.grim`, or safetensors.

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;

use grim_tensor::dtype::{DType, KQuantScheme, QuantProvenance, Storage};
use grim_tensor::error::{Error, Result};
use grim_tensor::provider::{RawTensor, TensorMeta, TensorProvider};

use crate::gguf::{
    read_gguf, read_tensor_bytes, GgufDType, GgufFile, GgufTensorInfo, GrimFusionOp, GrimMetadata,
    GrimQuantOverride, GrimTrainQuantMode,
};
use crate::format::{GrimFile, GrimTensorEntry, read_normals, read_outliers};
use crate::safetensors::{read_safetensor_bytes, read_safetensors_header, SafetensorInfo};

/// GGUF-backed `TensorProvider`. Holds the parsed file index and wraps a
/// `BufReader<File>` for lazy tensor reads.
pub struct GgufProvider {
    file: GgufFile,
    reader: std::sync::Mutex<BufReader<File>>,
    tensors: HashMap<String, GgufTensorInfo>,
    grim: GrimMetadata,
    overrides: HashMap<String, GrimQuantOverride>,
}

impl GgufProvider {
    pub fn open(path: &str) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| Error::Backend(format!("cannot open GGUF file '{path}': {e}")))?;
        let reader = BufReader::new(file);
        let gguf = read_gguf(reader)?;
        let mut tensors = HashMap::new();
        for t in &gguf.tensors {
            tensors.insert(t.name.clone(), t.clone());
        }

        // Parse `.grim` ROCm extension metadata.
        let grim = GrimMetadata::from_gguf_metadata(&gguf.metadata);
        let overrides: HashMap<String, GrimQuantOverride> = grim
            .quant_overrides
            .iter()
            .map(|o| (o.tensor_name.clone(), o.clone()))
            .collect();

        // §5.3 companion draft bundle config loading
        let companion_path = format!("{}.json", path);
        if std::path::Path::new(&companion_path).exists() {
            if let Ok(content) = std::fs::read_to_string(&companion_path) {
                println!(
                    "[GgufProvider] Loaded draft companion file: {} (content: {})",
                    companion_path, content
                );
            }
        }

        let file = File::open(path)
            .map_err(|e| Error::Backend(format!("cannot reopen GGUF file '{path}': {e}")))?;
        let reader = std::sync::Mutex::new(BufReader::new(file));
        Ok(Self {
            file: gguf,
            reader,
            tensors,
            grim,
            overrides,
        })
    }

    pub fn metadata(&self, key: &str) -> Option<&crate::gguf::GgufValue> {
        self.file.metadata.get(key)
    }

    pub fn architecture(&self) -> Option<&str> {
        self.metadata("general.architecture")?.as_str()
    }

    /// Returns the parsed `.grim` metadata for this file (empty if plain GGUF).
    pub fn grim_metadata(&self) -> &GrimMetadata {
        &self.grim
    }

    /// Construct a `GgufTokenizer` using this provider's parsed metadata keys.
    pub fn tokenizer(&self) -> Result<crate::tokenizer::GgufTokenizer> {
        crate::tokenizer::GgufTokenizer::from_metadata(&self.file.metadata)
    }

    /// Resolve the architecture string. Mirrors `architecture()` but returns an owned
    /// copy for callers that need to keep the value past the provider's lifetime.
    pub fn architecture_owned(&self) -> Option<String> {
        self.architecture().map(|s| s.to_string())
    }

    /// Access the tensor index (name → info mapping).
    pub fn tensors(&self) -> &HashMap<String, GgufTensorInfo> {
        &self.tensors
    }

    /// Training quantization mode declared by `.grim` metadata, if any.
    pub fn train_quant_mode(&self) -> Option<GrimTrainQuantMode> {
        self.grim.train_quant_mode
    }

    /// Target GPU GCN architecture (e.g. "gfx1100") declared by `.grim` metadata, if any.
    pub fn target_gcn(&self) -> Option<&str> {
        self.grim.target_gcn.as_deref()
    }

    /// Target execution wavefront/warp size (32 or 64) declared by `.grim` metadata.
    pub fn wavefront_size(&self) -> u32 {
        self.grim.wavefront_size
    }

    /// Target GPU LDS (local data share) memory size in bytes, if any.
    pub fn lds_size(&self) -> Option<u32> {
        self.grim.lds_size
    }

    /// ROCm fusion operations declared by `.grim` metadata, if any.
    pub fn rocm_fusion_ops(&self) -> &[GrimFusionOp] {
        &self.grim.rocm_fusion_ops
    }

    /// `true` if RMSNorm+MatMul fusion is requested either via train or ROCm metadata.
    pub fn has_rmsnorm_matmul_fusion(&self) -> bool {
        self.grim.train_fusion_ops.contains(&GrimFusionOp::RmsNormMatMul)
            || self.grim.rocm_fusion_ops.contains(&GrimFusionOp::RmsNormMatMul)
    }

    /// `true` if QKV+Attention fusion is requested either via train or ROCm metadata.
    pub fn has_qkv_attention_fusion(&self) -> bool {
        self.grim.train_fusion_ops.contains(&GrimFusionOp::QkvAttention)
            || self.grim.rocm_fusion_ops.contains(&GrimFusionOp::QkvAttention)
    }
}

/// Maps a `GgufDType` to a grim `DType` using the canonical mapping.
///
/// Delegates to [`crate::gguf::map_gguf_dtype_to_storage`] so there is a
/// single source of truth for GGUF→DType conversion.
fn dtype_from_gguf(gguf_dtype: GgufDType) -> DType {
    crate::gguf::map_gguf_dtype_to_storage(gguf_dtype)
}

/// Resolves the effective dtype for a tensor, applying any `.grim` per-tensor
/// override if present. Falls back to the GGUF dtype from the tensor info.
fn effective_dtype(info: &GgufTensorInfo, overrides: &HashMap<String, GrimQuantOverride>) -> DType {
    if let Some(ov) = overrides.get(&info.name) {
        return dtype_from_gguf(ov.override_dtype);
    }
    dtype_from_gguf(info.dtype)
}

impl TensorProvider for GgufProvider {
    fn get(&self, name: &str) -> Result<RawTensor> {
        let info = self.tensors.get(name).ok_or_else(|| {
            Error::Backend(format!("tensor '{name}' not found in GGUF file"))
        })?;
        let mut reader = self.reader.lock().unwrap();
        let bytes = read_tensor_bytes(&mut *reader, &self.file, info)?;
        let dtype = effective_dtype(info, &self.overrides);
        Ok(RawTensor {
            bytes,
            shape: info.shape(),
            dtype,
            provenance: QuantProvenance::GrimNative,
        })
    }

    fn meta(&self, name: &str) -> Result<TensorMeta> {
        let info = self.tensors.get(name).ok_or_else(|| {
            Error::Backend(format!("tensor '{name}' not found in GGUF file"))
        })?;
        let dtype = effective_dtype(info, &self.overrides);
        Ok(TensorMeta {
            dtype,
            provenance: QuantProvenance::GrimNative,
            shape: info.shape(),
            fusion_mask: 0,
        })
    }
}

/// Safetensors-backed `TensorProvider`.
pub struct SafetensorsProvider {
    info: std::collections::HashMap<String, SafetensorInfo>,
    tensors: std::collections::HashMap<String, SafetensorInfo>,
    reader: std::sync::Mutex<BufReader<File>>,
    data_region_start: u64,
    gptq: Option<crate::gptq::GptqProvider>,
}

impl SafetensorsProvider {
    pub fn open(path: &str) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| Error::Backend(format!("cannot open safetensors file '{path}': {e}")))?;
        let reader = BufReader::new(file);
        let (info, _metadata, data_region_start) = read_safetensors_header(reader)?;

        // §5.3 companion draft bundle config loading
        let companion_path = format!("{}.json", path);
        if std::path::Path::new(&companion_path).exists() {
            if let Ok(content) = std::fs::read_to_string(&companion_path) {
                println!("[SafetensorsProvider] Loaded draft companion file: {} (content: {})", companion_path, content);
            }
        }

        let has_gptq = info.keys().any(|k| k.ends_with(".qweight"));
        let gptq = if has_gptq {
            Some(crate::gptq::GptqProvider::open(path)?)
        } else {
            None
        };

        let mut tensors = info.clone();
        if let Some(ref g) = gptq {
            tensors.retain(|k, _| {
                !k.ends_with(".qweight") && !k.ends_with(".qzeros") && !k.ends_with(".scales") && !k.ends_with(".g_idx")
            });
            for (base_name, gptq_info) in &g.tensors {
                let qweight_name = format!("{}.qweight", base_name);
                if let Some(qw_info) = info.get(&qweight_name) {
                    tensors.insert(base_name.clone(), SafetensorInfo {
                        name: base_name.clone(),
                        dims: gptq_info.shape.clone(),
                        dtype_tag: qw_info.dtype_tag.clone(),
                        data_start: qw_info.data_start,
                        data_end: qw_info.data_end,
                    });
                }
            }
        }

        let file = File::open(path)
            .map_err(|e| Error::Backend(format!("cannot reopen safetensors file '{path}': {e}")))?;
        let reader = std::sync::Mutex::new(BufReader::new(file));
        Ok(Self {
            info,
            tensors,
            reader,
            data_region_start,
            gptq,
        })
    }

    /// Access the logical tensor index (filtered for quantized GPTQ models).
    pub fn tensors(&self) -> &std::collections::HashMap<String, SafetensorInfo> {
        &self.tensors
    }
}

impl TensorProvider for SafetensorsProvider {
    fn get(&self, name: &str) -> Result<RawTensor> {
        if let Some(ref gptq) = self.gptq {
            if gptq.tensors.contains_key(name) {
                return gptq.get(name);
            }
        }
        let info = self.info.get(name).ok_or_else(|| {
            Error::Backend(format!("tensor '{name}' not found in safetensors file"))
        })?;
        let mut reader = self.reader.lock().unwrap();
        let bytes = read_safetensor_bytes(&mut *reader, info, self.data_region_start)?;
        Ok(RawTensor {
            bytes,
            shape: info.shape(),
            dtype: info.grim_dtype()?,
            provenance: QuantProvenance::GrimNative,
        })
    }

    fn meta(&self, name: &str) -> Result<TensorMeta> {
        if let Some(ref gptq) = self.gptq {
            if gptq.tensors.contains_key(name) {
                return gptq.meta(name);
            }
        }
        let info = self.info.get(name).ok_or_else(|| {
            Error::Backend(format!("tensor '{name}' not found in safetensors file"))
        })?;
        Ok(TensorMeta {
            dtype: info.grim_dtype()?,
            provenance: QuantProvenance::GrimNative,
            shape: info.shape(),
            fusion_mask: 0,
        })
    }
}

/// Native `.grim`-backed `TensorProvider`.
///
/// Opens a file with the `GRIM\x01` magic header, parses the JSON metadata
/// layer and tensor registry, and lazily reads normals + outliers streams
/// on `get()`. The returned `RawTensor.bytes` contains the raw packed
/// normals; callers that need outlier correction read them separately via
/// the [`crate::format`] helpers.
pub struct GrimProvider {
    file: GrimFile,
    reader: std::sync::Mutex<BufReader<File>>,
}

impl GrimProvider {
    /// Open a native `.grim` file for reading.
    pub fn open(path: &str) -> Result<Self> {
        let f = File::open(path)
            .map_err(|e| Error::Backend(format!("cannot open .grim file '{path}': {e}")))?;
        let mut reader = BufReader::new(f);
        let file = GrimFile::read(&mut reader)?;

        // Reopen for lazy reads — the BufReader above was consumed by the parse.
        let f = File::open(path)
            .map_err(|e| Error::Backend(format!("cannot reopen .grim file '{path}': {e}")))?;
        let reader = std::sync::Mutex::new(BufReader::new(f));
        Ok(Self { file, reader })
    }

    /// Access the parsed `.grim` metadata.
    pub fn grim_metadata(&self) -> &GrimMetadata {
        &self.file.metadata
    }

    /// Look up the per-tensor capability extension for `name`, if any.
    ///
    /// Returns `None` for plain version-1 tensors that carry no extension
    /// declaration. Callers can use `GrimTensorExt::is_legacy()` to detect
    /// the default (zeroed) extension.
    pub fn ext_for(&self, name: &str) -> Option<&crate::spec::GrimTensorExt> {
        self.file.metadata.ext_entries.iter().find(|e| e.tensor_name == name)
    }

    /// Access the tensor registry.
    pub fn tensors(&self) -> &[GrimTensorEntry] {
        &self.file.tensors
    }

    /// Read the outliers stream for a tensor (lazily, from disk).
    pub fn outliers(&self, name: &str) -> Result<Vec<crate::format::GrimOutlier>> {
        let entry = self.file.tensor(name).ok_or_else(|| {
            Error::Backend(format!("tensor '{name}' not found in .grim file"))
        })?;
        let mut reader = self.reader.lock().unwrap();
        read_outliers(&mut *reader, entry)
    }
}

impl TensorProvider for GrimProvider {
    fn get(&self, name: &str) -> Result<RawTensor> {
        let entry = self.file.tensor(name).ok_or_else(|| {
            Error::Backend(format!("tensor '{name}' not found in .grim file"))
        })?;
        let mut reader = self.reader.lock().unwrap();
        let bytes = read_normals(&mut *reader, entry)?;
        Ok(RawTensor {
            bytes,
            shape: entry.shape.clone(),
            dtype: dtype_from_bitwidth(entry.base_bitwidth),
            provenance: QuantProvenance::GrimNative,
        })
    }

    fn meta(&self, name: &str) -> Result<TensorMeta> {
        let entry = self.file.tensor(name).ok_or_else(|| {
            Error::Backend(format!("tensor '{name}' not found in .grim file"))
        })?;
        let fusion_mask = self
            .ext_for(name)
            .map(|ext| ext.fusion_mask)
            .unwrap_or(0);
        Ok(TensorMeta {
            dtype: dtype_from_bitwidth(entry.base_bitwidth),
            provenance: QuantProvenance::GrimNative,
            shape: entry.shape.clone(),
            fusion_mask,
        })
    }
}

/// Map a `.grim` base bitwidth to the arithmetic `DType` used for dequant.
///
/// The packed normals are stored at `base_bitwidth` bits per weight; the
/// arithmetic type is always F32 (dequantized). The storage layer carries
/// the bitwidth so dequant kernels know how to unpack.
fn dtype_from_bitwidth(base_bitwidth: u8) -> DType {
    match base_bitwidth {
        16 => DType::F16,
        8 => DType {
            arith: grim_tensor::ArithType::F32,
            storage: Storage::KQuant(KQuantScheme::Q80),
        },
        4 => DType {
            arith: grim_tensor::ArithType::F32,
            storage: Storage::Block(grim_tensor::dtype::BlockDtype::Fp4),
        },
        3 | 2 => DType {
            arith: grim_tensor::ArithType::F32,
            storage: Storage::KQuant(KQuantScheme::Q2K),
        },
        _ => DType::F32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::io::{Cursor, Write};

    use crate::gguf::{GgufValue, GGUF_MAGIC, GGUF_VERSION};

    /// Build a minimal GGUF byte stream with the given metadata KV pairs and zero tensors.
    /// Used by tprov accessor tests to exercise `GgufProvider::open` against real serialized bytes.
    ///
    /// Values supported:
    /// - `GgufValue::String(s)` — written as a GGUF string
    /// - `GgufValue::Array(items)` — written as a GGUF array, each string element is `&str`
    fn write_minimal_gguf_bytes(metadata: &HashMap<&str, GgufValue>) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_all(&GGUF_MAGIC.to_le_bytes()).expect("write magic");
        buf.write_all(&GGUF_VERSION.to_le_bytes()).expect("write version");
        buf.write_all(&0u64.to_le_bytes()).expect("write tensor count");
        buf.write_all(&(metadata.len() as u64).to_le_bytes())
            .expect("write metadata kv count");

        for (key, value) in metadata {
            let key_bytes = key.as_bytes();
            buf.write_all(&(key_bytes.len() as u64).to_le_bytes()).expect("write key len");
            buf.write_all(key_bytes).expect("write key bytes");

            match value {
                GgufValue::String(s) => {
                    buf.write_all(&8u32.to_le_bytes()).expect("write string tag");
                    let s_bytes = s.as_bytes();
                    buf.write_all(&(s_bytes.len() as u64).to_le_bytes()).expect("write string len");
                    buf.write_all(s_bytes).expect("write string bytes");
                }
                GgufValue::Array(items) => {
                    // GGUF array: tag=9, element_tag=8 (string), count=items.len(), then each string.
                    // Note: each array element re-emits its own tag (matches `read_gguf_value`).
                    buf.write_all(&9u32.to_le_bytes()).expect("write array tag");
                    buf.write_all(&8u32.to_le_bytes()).expect("write array elem string tag");
                    buf.write_all(&(items.len() as u64).to_le_bytes())
                        .expect("write array count");
                    for item in items {
                        if let GgufValue::String(s) = item {
                            let s_bytes = s.as_bytes();
                            buf.write_all(&(s_bytes.len() as u64).to_le_bytes())
                                .expect("write elem string len");
                            buf.write_all(s_bytes).expect("write elem string bytes");
                        } else {
                            panic!("test helper only supports string array elements, got {item:?}");
                        }
                    }
                }
                other => panic!("test helper currently supports only string/array values, got {other:?}"),
            }
        }

        buf
    }

    /// Round-trip: write the byte stream to a temp file and open via `GgufProvider::open`.
    fn open_provider_from_metadata(metadata: HashMap<&str, GgufValue>) -> GgufProvider {
        let bytes = write_minimal_gguf_bytes(&metadata);
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("tiny.gguf");
        std::fs::write(&path, &bytes).expect("write gguf file");
        let provider = GgufProvider::open(path.to_str().unwrap()).expect("open gguf");
        // Keep the tempdir alive for the lifetime of the test by leaking it.
        std::mem::forget(dir);
        provider
    }

    #[test]
    fn train_quant_mode_returns_none_for_plain_gguf() {
        let provider = open_provider_from_metadata(HashMap::new());
        assert_eq!(provider.train_quant_mode(), None);
    }

    #[test]
    fn train_quant_mode_parses_bf16() {
        let mut meta = HashMap::new();
        meta.insert("grim.train.quant_mode", GgufValue::String("bf16".into()));
        let provider = open_provider_from_metadata(meta);
        assert_eq!(provider.train_quant_mode(), Some(GrimTrainQuantMode::Bf16));
    }

    #[test]
    fn rocm_fusion_ops_empty_for_plain_gguf() {
        let provider = open_provider_from_metadata(HashMap::new());
        assert!(provider.rocm_fusion_ops().is_empty());
    }

    #[test]
    fn has_rmsnorm_matmul_fusion_via_rocm_ops() {
        let mut meta = HashMap::new();
        meta.insert(
            "grim.rocm.fusion_ops",
            GgufValue::Array(vec![GgufValue::String("rmsnorm_matmul".into())]),
        );
        let provider = open_provider_from_metadata(meta);
        assert!(provider.has_rmsnorm_matmul_fusion());
        assert!(!provider.has_qkv_attention_fusion());
    }

    #[test]
    fn has_qkv_attention_fusion_via_train_ops() {
        let mut meta = HashMap::new();
        meta.insert(
            "grim.train.fusion_ops",
            GgufValue::Array(vec![GgufValue::String("qkv_attention".into())]),
        );
        let provider = open_provider_from_metadata(meta);
        assert!(provider.has_qkv_attention_fusion());
        assert!(!provider.has_rmsnorm_matmul_fusion());
    }

    #[test]
    fn rocm_fusion_ops_returns_empty_slice_for_unknown_string() {
        let mut meta = HashMap::new();
        meta.insert(
            "grim.rocm.fusion_ops",
            GgufValue::Array(vec![GgufValue::String("not_a_real_op".into())]),
        );
        let provider = open_provider_from_metadata(meta);
        assert!(provider.rocm_fusion_ops().is_empty());
    }

    #[test]
    fn test_dtype_from_gguf_block_mappings() {
        use crate::gguf::GgufDType;
        use grim_tensor::dtype::{BlockDtype, Storage, KQuantScheme};
        
        let d_q4k = super::dtype_from_gguf(GgufDType::Q4K);
        assert_eq!(d_q4k.storage, Storage::Block(BlockDtype::Fp4));
        
        let d_q5k = super::dtype_from_gguf(GgufDType::Q5K);
        assert_eq!(d_q5k.storage, Storage::Block(BlockDtype::Nf4));

        let d_q6k = super::dtype_from_gguf(GgufDType::Q6K);
        assert_eq!(d_q6k.storage, Storage::Block(BlockDtype::Fp8));

        let d_q80 = super::dtype_from_gguf(GgufDType::Q8_0);
        assert_eq!(d_q80.storage, Storage::KQuant(KQuantScheme::Q80));
    }

    /// Write a minimal native `.grim` file to a temp path, then open it
    /// with `GrimProvider` and verify `meta`/`get` round-trip the registry.
    fn write_minimal_grim_file(path: &std::path::Path) {
        use crate::format::{GrimFile, GrimHeader, GrimTensorEntry, OUTLIER_RECORD_BYTES};
        use crate::gguf::{GrimMetadata, GrimRocmlProfile};
        use std::collections::HashMap;
        use std::io::Cursor;

        let metadata = GrimMetadata {
            magic: Some("grim-v1".into()),
            quant_version: Some(1),
            rocml_profile: GrimRocmlProfile::Rdna3,
            wavefront_size: 64,
            target_gcn: Some("gfx1100".into()),
            ..Default::default()
        };

        let tensor = GrimTensorEntry {
            name: "layer.0.weight".into(),
            shape: vec![4, 4],
            base_bitwidth: 4,
            payload_offset: 0,
            payload_size: 16, // 16 elements * 4 bits / 8 = 8 bytes, padded
            outlier_count: 1,
            outlier_offset: 0,
            ..Default::default()
        };

        let grim_file = GrimFile {
            header: GrimHeader::new(1, 0),
            metadata,
            tensors: vec![tensor.clone()],
            tensors_by_name: HashMap::new(),
            kv_blobs: HashMap::new(),
        };

        let mut buf = Vec::new();
        let mut cursor = Cursor::new(&mut buf);
        let written = grim_file.write(&mut cursor).unwrap();
        let entry = &written[0];
        let normals_bytes = entry.payload_size as usize;

        // Pad from buf end to payload_offset, append normals, then the single outlier.
        let current_len = buf.len();
        if (current_len as u64) < entry.payload_offset {
            buf.resize(entry.payload_offset as usize, 0u8);
        }
        buf.resize(buf.len() + normals_bytes, 0u8);
        let outlier = crate::format::GrimOutlier { index: 0, value: 1.0 };
        buf.extend_from_slice(&outlier.encode());

        std::fs::write(path, &buf).unwrap();
    }

    #[test]
    fn grim_provider_opens_native_file_and_reads_meta() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.grim");
        write_minimal_grim_file(&path);

        let provider = GrimProvider::open(path.to_str().unwrap()).unwrap();
        assert_eq!(provider.grim_metadata().magic.as_deref(), Some("grim-v1"));
        assert_eq!(provider.grim_metadata().target_gcn.as_deref(), Some("gfx1100"));

        let meta = provider.meta("layer.0.weight").unwrap();
        assert_eq!(meta.shape, vec![4, 4]);
    }

    #[test]
    fn grim_provider_reads_normals_and_outliers() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.grim");
        write_minimal_grim_file(&path);

        let provider = GrimProvider::open(path.to_str().unwrap()).unwrap();
        let tensor = provider.get("layer.0.weight").unwrap();
        assert_eq!(tensor.shape, vec![4, 4]);
        assert!(!tensor.bytes.is_empty());

        let outliers = provider.outliers("layer.0.weight").unwrap();
        assert_eq!(outliers.len(), 1);
        assert_eq!(outliers[0].index, 0);
        assert!((outliers[0].value - 1.0).abs() < 1e-2);
    }

    #[test]
    fn grim_provider_rejects_unknown_tensor() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.grim");
        write_minimal_grim_file(&path);

        let provider = GrimProvider::open(path.to_str().unwrap()).unwrap();
        assert!(provider.get("nonexistent").is_err());
        assert!(provider.meta("nonexistent").is_err());
    }
}