//! GDS / hipFile Direct NVMe Tier with Host Fallback.
//!
//! Provides direct DMA transfers between GPU VRAM and local NVMe storage,
//! bypassing host memory round-trips when supported by the hardware/driver,
//! and gracefully falling back to standard scratch file I/O otherwise.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::collections::HashMap;

use grim_core::error::{Error, Result};
use crate::gds_ffi::{HipFileHandle, HipFileLib};
use crate::BlockId;

/// Direct DMA GDS storage tier with host-bounce fallback.
pub enum GdsTier {
    /// Direct GDS DMA path through libhipfile.
    Direct {
        lib: HipFileLib,
        handle: HipFileHandle,
        file_path: PathBuf,
        block_offsets: Mutex<HashMap<BlockId, u64>>,
        fallback_dir: PathBuf,
        fallback_files: Mutex<HashMap<BlockId, PathBuf>>,
    },
    /// Host fallback using scratch files.
    Fallback {
        scratch_dir: PathBuf,
        files: Mutex<HashMap<BlockId, PathBuf>>,
    },
}

impl GdsTier {
    /// Create a new GdsTier at `scratch_path`.
    ///
    /// Automatically probes for GDS/hipFile availability:
    /// - If `libhipfile.so` is present and functional, initializes `GdsTier::Direct`.
    /// - Otherwise initializes `GdsTier::Fallback`.
    pub fn new<P: AsRef<Path>>(scratch_path: P) -> Result<Self> {
        let p = scratch_path.as_ref();
        let dir = if p.is_dir() {
            p.to_path_buf()
        } else {
            p.parent().unwrap_or(Path::new(".")).to_path_buf()
        };
        fs::create_dir_all(&dir).map_err(|e| Error::KvCache(format!("GDS fallback dir failed: {e}")))?;

        if let Some(lib) = HipFileLib::load() {
            let file_path = dir.join("gds_kv_pool.bin");

            // Create file if it doesn't exist
            let _ = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&file_path)
                .map_err(|e| Error::KvCache(format!("GDS open file failed: {e}")))?;

            if let Some(handle) = lib.register_file(file_path.to_str().unwrap_or(""), 0) {
                return Ok(Self::Direct {
                    lib,
                    handle,
                    file_path,
                    block_offsets: Mutex::new(HashMap::new()),
                    fallback_dir: dir,
                    fallback_files: Mutex::new(HashMap::new()),
                });
            }
        }

        Ok(Self::Fallback {
            scratch_dir: dir,
            files: Mutex::new(HashMap::new()),
        })
    }

    /// Whether this tier is operating in Direct GDS mode.
    pub fn is_direct(&self) -> bool {
        matches!(self, Self::Direct { .. })
    }

    /// Demote a KV block slice from memory to NVMe tier.
    pub fn demote_block(&self, block_id: BlockId, data: &[f32]) -> Result<()> {
        let byte_len = data.len() * std::mem::size_of::<f32>();
        match self {
            Self::Direct {
                lib,
                handle,
                block_offsets,
                fallback_dir,
                fallback_files,
                ..
            } => {
                let mut offsets = block_offsets.lock().unwrap();
                let offset = (block_id as u64) * (byte_len as u64);

                let written = lib.write_direct(
                    *handle,
                    data.as_ptr() as *const _,
                    byte_len,
                    offset as i64,
                    0,
                );
                if written > 0 && written as usize == byte_len {
                    offsets.insert(block_id, offset);
                    return Ok(());
                }

                // If direct DMA is rejected (e.g. host memory buffer or unregistered pointer), use fallback scratch file
                let file_path = fallback_dir.join(format!("block_{block_id}.kv"));
                let mut file = File::create(&file_path).map_err(|e| Error::KvCache(format!("GDS create file failed: {e}")))?;
                let bytes = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len)
                };
                file.write_all(bytes).map_err(|e| Error::KvCache(format!("GDS write file failed: {e}")))?;
                fallback_files.lock().unwrap().insert(block_id, file_path);
                Ok(())
            }
            Self::Fallback { scratch_dir, files } => {
                let file_path = scratch_dir.join(format!("block_{block_id}.kv"));
                let mut file = File::create(&file_path).map_err(|e| Error::KvCache(format!("GDS create file failed: {e}")))?;
                let bytes = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len)
                };
                file.write_all(bytes).map_err(|e| Error::KvCache(format!("GDS write file failed: {e}")))?;
                files.lock().unwrap().insert(block_id, file_path);
                Ok(())
            }
        }
    }

    /// Promote a KV block from NVMe tier back into output slice.
    pub fn promote_block(&self, block_id: BlockId, out_buf: &mut [f32]) -> Result<()> {
        let byte_len = out_buf.len() * std::mem::size_of::<f32>();
        match self {
            Self::Direct {
                lib,
                handle,
                block_offsets,
                fallback_files,
                ..
            } => {
                let offsets = block_offsets.lock().unwrap();
                if let Some(&offset) = offsets.get(&block_id) {
                    let read_bytes = lib.read_direct(
                        *handle,
                        out_buf.as_mut_ptr() as *mut _,
                        byte_len,
                        offset as i64,
                        0,
                    );
                    if read_bytes > 0 && read_bytes as usize == byte_len {
                        return Ok(());
                    }
                }

                // Fallback scratch file check
                let path = {
                    let map = fallback_files.lock().unwrap();
                    map.get(&block_id).cloned().ok_or_else(|| {
                        Error::KvCache(format!("GDS direct/fallback promote: block {block_id} not found"))
                    })?
                };

                let mut file = File::open(path).map_err(|e| Error::KvCache(format!("GDS open fallback failed: {e}")))?;
                file.seek(SeekFrom::Start(0)).map_err(|e| Error::KvCache(format!("GDS seek fallback failed: {e}")))?;
                let bytes = unsafe {
                    std::slice::from_raw_parts_mut(out_buf.as_mut_ptr() as *mut u8, byte_len)
                };
                file.read_exact(bytes).map_err(|e| Error::KvCache(format!("GDS read fallback failed: {e}")))?;
                Ok(())
            }
            Self::Fallback { files, .. } => {
                let path = {
                    let map = files.lock().unwrap();
                    map.get(&block_id).cloned().ok_or_else(|| {
                        Error::KvCache(format!("GDS fallback promote: block {block_id} not found"))
                    })?
                };

                let mut file = File::open(path).map_err(|e| Error::KvCache(format!("GDS open fallback failed: {e}")))?;
                file.seek(SeekFrom::Start(0)).map_err(|e| Error::KvCache(format!("GDS seek fallback failed: {e}")))?;
                let bytes = unsafe {
                    std::slice::from_raw_parts_mut(out_buf.as_mut_ptr() as *mut u8, byte_len)
                };
                file.read_exact(bytes).map_err(|e| Error::KvCache(format!("GDS read fallback failed: {e}")))?;
                Ok(())
            }
        }
    }
}

impl Drop for GdsTier {
    fn drop(&mut self) {
        if let Self::Direct { lib, handle, .. } = self {
            lib.deregister_file(*handle);
        }
    }
}
