//! Phase E: NVMe disk persistence for cross-process hybrid prefix checkpoints
//! (KV blocks + GDN/ShortConv recurrent state snapshots).
//!
//! File layout (`<model_fp_hex>_<token_hash_hex>.grimkv`):
//!   - 8 B  magic: `b"GRIMKV01"`
//!   - 4 B  version: `u32` LE (`1`)
//!   - 8 B  model_fingerprint: `u64` LE
//!   - 4 B  kv_dtype_tag: `u32` LE (`1=F32, 2=F16, 3=Q8_1, 4=INT8`)
//!   - 8 B  prefix_len: `u64` LE
//!   - 4 B  num_attn_layers: `u32` LE
//!   - 4 B  num_gdn_layers: `u32` LE
//!   - 8 B  kv_bytes_len: `u64` LE
//!   - 8 B  recurrent_bytes_len: `u64` LE
//!   - 4 B  crc32: `u32` LE (IEEE CRC32 over `kv_bytes || recurrent_bytes`)
//!   - [kv_bytes]
//!   - [recurrent_bytes]
//!
//! Writes are atomic (`tmp -> fsync -> rename`). Loads validate magic, version,
//! `model_fingerprint`, `kv_dtype_tag`, and CRC32; any corruption unlinks the
//! bad file and returns `Ok(None)` so the caller fails closed to cold prefill.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

const MAGIC: &[u8; 8] = b"GRIMKV01";
const FORMAT_VERSION: u32 = 1;
const HEADER_BYTES: usize = 60;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum PrefixKvDtype {
    F32 = 1,
    F16 = 2,
    Q8_1 = 3,
    Int8 = 4,
}

impl PrefixKvDtype {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            1 => Some(Self::F32),
            2 => Some(Self::F16),
            3 => Some(Self::Q8_1),
            4 => Some(Self::Int8),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixCheckpointMeta {
    pub model_fingerprint: u64,
    pub kv_dtype: PrefixKvDtype,
    pub prefix_len: usize,
    pub num_attn_layers: u32,
    pub num_gdn_layers: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixCheckpoint {
    pub meta: PrefixCheckpointMeta,
    pub kv_bytes: Vec<u8>,
    pub recurrent_bytes: Vec<u8>,
}

/// Simple deterministic IEEE CRC32 (no external crate needed).
fn crc32_ieee(a: &[u8], b: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in a.iter().chain(b.iter()) {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// FNV-1a 64-bit token sequence hash for checkpoint filenames.
pub fn hash_tokens_u64(tokens: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &t in tokens {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    h
}

pub struct DiskPrefixStore {
    dir: PathBuf,
    budget_bytes: u64,
}

impl DiskPrefixStore {
    pub fn new(dir: impl AsRef<Path>, budget_bytes: u64) -> std::io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        Ok(Self { dir, budget_bytes })
    }

    pub fn checkpoint_path(&self, model_fingerprint: u64, token_hash: u64) -> PathBuf {
        self.dir
            .join(format!("{model_fingerprint:016x}_{token_hash:016x}.grimkv"))
    }

    /// Atomically write checkpoint (`tmp -> fsync -> rename`) and enforce LRU disk budget.
    pub fn save(
        &self,
        token_hash: u64,
        ckpt: &PrefixCheckpoint,
    ) -> std::io::Result<PathBuf> {
        let final_path = self.checkpoint_path(ckpt.meta.model_fingerprint, token_hash);
        let tmp_path = final_path.with_extension("grimkv.tmp");
        let crc = crc32_ieee(&ckpt.kv_bytes, &ckpt.recurrent_bytes);

        let mut hdr = Vec::with_capacity(HEADER_BYTES);
        hdr.extend_from_slice(MAGIC);
        hdr.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        hdr.extend_from_slice(&ckpt.meta.model_fingerprint.to_le_bytes());
        hdr.extend_from_slice(&(ckpt.meta.kv_dtype as u32).to_le_bytes());
        hdr.extend_from_slice(&(ckpt.meta.prefix_len as u64).to_le_bytes());
        hdr.extend_from_slice(&ckpt.meta.num_attn_layers.to_le_bytes());
        hdr.extend_from_slice(&ckpt.meta.num_gdn_layers.to_le_bytes());
        hdr.extend_from_slice(&(ckpt.kv_bytes.len() as u64).to_le_bytes());
        hdr.extend_from_slice(&(ckpt.recurrent_bytes.len() as u64).to_le_bytes());
        hdr.extend_from_slice(&crc.to_le_bytes());
        debug_assert_eq!(hdr.len(), HEADER_BYTES);

        {
            let mut f = File::create(&tmp_path)?;
            f.write_all(&hdr)?;
            f.write_all(&ckpt.kv_bytes)?;
            f.write_all(&ckpt.recurrent_bytes)?;
            f.sync_all()?;
        }
        fs::rename(&tmp_path, &final_path)?;
        self.enforce_budget(&final_path)?;
        Ok(final_path)
    }

    /// Load and validate a checkpoint. Returns `Ok(None)` on miss, fingerprint/dtype
    /// mismatch, or CRC32 corruption (unlinking corrupted files).
    pub fn load(
        &self,
        expected_model_fingerprint: u64,
        expected_kv_dtype: PrefixKvDtype,
        token_hash: u64,
    ) -> std::io::Result<Option<PrefixCheckpoint>> {
        let path = self.checkpoint_path(expected_model_fingerprint, token_hash);
        if !path.exists() {
            return Ok(None);
        }
        let mut f = File::open(&path)?;
        let mut hdr = [0u8; HEADER_BYTES];
        if f.read_exact(&mut hdr).is_err() {
            let _ = fs::remove_file(&path);
            return Ok(None);
        }
        if &hdr[0..8] != MAGIC {
            let _ = fs::remove_file(&path);
            return Ok(None);
        }
        let version = u32::from_le_bytes(hdr[8..12].try_into().unwrap());
        let model_fp = u64::from_le_bytes(hdr[12..20].try_into().unwrap());
        let dtype_raw = u32::from_le_bytes(hdr[20..24].try_into().unwrap());
        let prefix_len = u64::from_le_bytes(hdr[24..32].try_into().unwrap()) as usize;
        let num_attn = u32::from_le_bytes(hdr[32..36].try_into().unwrap());
        let num_gdn = u32::from_le_bytes(hdr[36..40].try_into().unwrap());
        let kv_len = u64::from_le_bytes(hdr[40..48].try_into().unwrap()) as usize;
        let rec_len = u64::from_le_bytes(hdr[48..56].try_into().unwrap()) as usize;
        let stored_crc = u32::from_le_bytes(hdr[56..60].try_into().unwrap());

        if version != FORMAT_VERSION
            || model_fp != expected_model_fingerprint
            || PrefixKvDtype::from_u32(dtype_raw) != Some(expected_kv_dtype)
        {
            return Ok(None);
        }

        let mut kv_bytes = vec![0u8; kv_len];
        let mut recurrent_bytes = vec![0u8; rec_len];
        if f.read_exact(&mut kv_bytes).is_err() || f.read_exact(&mut recurrent_bytes).is_err() {
            let _ = fs::remove_file(&path);
            return Ok(None);
        }
        if crc32_ieee(&kv_bytes, &recurrent_bytes) != stored_crc {
            let _ = fs::remove_file(&path);
            return Ok(None);
        }

        Ok(Some(PrefixCheckpoint {
            meta: PrefixCheckpointMeta {
                model_fingerprint: model_fp,
                kv_dtype: expected_kv_dtype,
                prefix_len,
                num_attn_layers: num_attn,
                num_gdn_layers: num_gdn,
            },
            kv_bytes,
            recurrent_bytes,
        }))
    }

    fn enforce_budget(&self, keep: &Path) -> std::io::Result<()> {
        if self.budget_bytes == 0 {
            return Ok(());
        }
        let mut entries: Vec<(PathBuf, u64, SystemTime)> = Vec::new();
        let mut total: u64 = 0;
        for ent in fs::read_dir(&self.dir)? {
            let ent = ent?;
            let p = ent.path();
            if p.extension().and_then(|e| e.to_str()) != Some("grimkv") {
                continue;
            }
            if let Ok(md) = ent.metadata() {
                let sz = md.len();
                total = total.saturating_add(sz);
                let mtime = md.modified().unwrap_or(SystemTime::UNIX_EPOCH);
                entries.push((p, sz, mtime));
            }
        }
        if total <= self.budget_bytes {
            return Ok(());
        }
        entries.sort_by_key(|(_, _, t)| *t);
        for (p, sz, _) in entries {
            if total <= self.budget_bytes {
                break;
            }
            if p == keep {
                continue;
            }
            if fs::remove_file(&p).is_ok() {
                total = total.saturating_sub(sz);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ckpt(fp: u64, dtype: PrefixKvDtype) -> PrefixCheckpoint {
        PrefixCheckpoint {
            meta: PrefixCheckpointMeta {
                model_fingerprint: fp,
                kv_dtype: dtype,
                prefix_len: 28_000,
                num_attn_layers: 16,
                num_gdn_layers: 48,
            },
            kv_bytes: vec![0xAB; 256],
            recurrent_bytes: vec![0xCD; 128],
        }
    }

    #[test]
    fn roundtrip_and_negative_gates() {
        let dir = std::env::temp_dir().join(format!("grim_prefix_store_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let store = DiskPrefixStore::new(&dir, 4096).unwrap();
        let th = hash_tokens_u64(&[1, 2, 3, 4]);
        let ckpt = sample_ckpt(0x1111, PrefixKvDtype::Q8_1);

        // 1. Atomic save + exact load
        let path = store.save(th, &ckpt).unwrap();
        let loaded = store.load(0x1111, PrefixKvDtype::Q8_1, th).unwrap().unwrap();
        assert_eq!(loaded, ckpt);

        // 2. Negative gate: model fingerprint mismatch -> fail-closed None
        assert!(store.load(0x2222, PrefixKvDtype::Q8_1, th).unwrap().is_none());

        // 3. Negative gate: KV dtype mismatch -> fail-closed None
        assert!(store.load(0x1111, PrefixKvDtype::F16, th).unwrap().is_none());

        // 4. Negative gate: CRC32 corruption -> unlinks & returns None
        let mut raw = fs::read(&path).unwrap();
        let last = raw.len() - 1;
        raw[last] ^= 0xFF;
        fs::write(&path, &raw).unwrap();
        assert!(store.load(0x1111, PrefixKvDtype::Q8_1, th).unwrap().is_none());
        assert!(!path.exists(), "corrupted checkpoint must be unlinked");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn lru_budget_evicts_oldest_file() {
        let dir = std::env::temp_dir().join(format!("grim_prefix_store_lru_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        // Budget fits 1 checkpoint (~444 B) but not 2 (~888 B)
        let store = DiskPrefixStore::new(&dir, 600).unwrap();
        let p1 = store.save(hash_tokens_u64(&[1]), &sample_ckpt(0x1, PrefixKvDtype::Q8_1)).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        let p2 = store.save(hash_tokens_u64(&[2]), &sample_ckpt(0x1, PrefixKvDtype::Q8_1)).unwrap();
        assert!(!p1.exists(), "oldest checkpoint should be evicted under disk budget");
        assert!(p2.exists(), "newest checkpoint must be retained");
        let _ = fs::remove_dir_all(&dir);
    }
}
