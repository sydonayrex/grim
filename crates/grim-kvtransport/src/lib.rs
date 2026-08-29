//! `grim-kvtransport` — tiered KV cache local transport and spillage.
//!
//! Handles moving KV block contents between GPU, Host RAM, and local scratch NVMe files.
//! Sits inside the paged KV pool's eviction policy to support demote-before-drop.

use parking_lot::RwLock;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::Mutex;

use grim_core::error::{Error, Result};
use grim_tensor::backend::BackendDevice;

pub mod bitmask_index;
pub use bitmask_index::{BitmaskChunkIndex, ChunkEntry, TierMask};

pub mod pin_lease;
pub use pin_lease::{LeaseStatus, PinLeaseMonitor, PinnedLease, SharedPinLeaseMonitor};

pub type BlockId = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheTier {
    Gpu,
    HostRam,
    NvMe,
    /// An NVMe weight-streaming layer used when weight tensors exceed VRAM/DRAM.
    NvMeWeightStream,
}

/// Applies OS-level `madvise` to the given slice/pointer range under Linux/macOS.
pub fn grimvise_advise(data: &[f32], advice: grim_tensor::MemAdvice) -> Result<()> {
    #[cfg(target_os = "linux")]
    {
        use std::os::raw::c_void;
        let ptr = data.as_ptr() as *mut c_void;
        let len = std::mem::size_of_val(data);

        let raw_advice = match advice {
            grim_tensor::MemAdvice::Sequential => libc::MADV_SEQUENTIAL,
            grim_tensor::MemAdvice::Random => libc::MADV_RANDOM,
            grim_tensor::MemAdvice::WillNeed => libc::MADV_WILLNEED,
            grim_tensor::MemAdvice::DontNeed => libc::MADV_DONTNEED,
            _ => return Ok(()), // GPU advice ignored on CPU host pages
        };

        let res = unsafe { libc::madvise(ptr, len, raw_advice) };
        if res != 0 {
            return Err(Error::KvCache(format!(
                "madvise failed with system error code {}",
                std::io::Error::last_os_error()
            )));
        }
    }

    #[cfg(target_os = "macos")]
    {
        use std::os::raw::c_void;
        let ptr = data.as_ptr() as *mut c_void;
        let len = data.len() * std::mem::size_of::<f32>();

        let raw_advice = match advice {
            grim_tensor::MemAdvice::Sequential => libc::MADV_SEQUENTIAL,
            grim_tensor::MemAdvice::Random => libc::MADV_RANDOM,
            grim_tensor::MemAdvice::WillNeed => libc::MADV_WILLNEED,
            grim_tensor::MemAdvice::DontNeed => libc::MADV_DONTNEED,
            _ => return Ok(()), // GPU advice ignored on CPU host pages
        };

        let res = unsafe { libc::madvise(ptr, len, raw_advice) };
        if res != 0 {
            return Err(Error::KvCache(format!(
                "madvise failed on macOS with system error code {}",
                std::io::Error::last_os_error()
            )));
        }
    }

    // Windows / other OS: advisory hint is a no-op
    let _ = data;
    let _ = advice;
    Ok(())
}

/// Memory pinning and zero-copy transfer registration attributes for RDMA / RoCE interconnects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RdmaPinnedRegion {
    pub virtual_addr: u64,
    pub length_bytes: usize,
    pub lkey: u32,
    pub rkey: u32,
}

impl RdmaPinnedRegion {
    /// Create a simulated pinned memory region descriptor for zero-copy DMA writes.
    pub fn new_pinned(virtual_addr: u64, length_bytes: usize, lkey: u32, rkey: u32) -> Self {
        Self {
            virtual_addr,
            length_bytes,
            lkey,
            rkey,
        }
    }

    /// Check if an offset and length fall within this registered region.
    pub fn bounds_check(&self, offset: usize, len: usize) -> bool {
        offset.saturating_add(len) <= self.length_bytes
    }
}

/// Manages tiered storage of KV blocks.
pub struct LocalSpillManager {
    /// Directory where NVMe spill files are cached.
    scratch_dir: PathBuf,
    /// Maps each block to its current storage tier.
    block_tiers: HashMap<BlockId, CacheTier>,
    /// In-memory cache for Host RAM tier.
    host_ram_cache: HashMap<BlockId, (Vec<f32>, Vec<f32>)>,
    /// File path tracking for NVMe disk tier.
    nvme_cache: HashMap<BlockId, PathBuf>,
    /// Size of each block in floats.
    block_elems: usize,
}

impl LocalSpillManager {
    /// Creates a new manager. NVMe temporary files will be stored under the given scratch directory.
    pub fn new(scratch_dir: PathBuf, block_elems: usize) -> Result<Self> {
        if !scratch_dir.exists() {
            fs::create_dir_all(&scratch_dir).map_err(|e| Error::KvCache(e.to_string()))?;
        }
        Ok(Self {
            scratch_dir,
            block_tiers: HashMap::new(),
            host_ram_cache: HashMap::new(),
            nvme_cache: HashMap::new(),
            block_elems,
        })
    }

    /// Demotes a block from GPU memory to Host RAM.
    pub fn demote_to_host(&mut self, block_id: BlockId, k: Vec<f32>, v: Vec<f32>) -> Result<()> {
        if k.len() != self.block_elems || v.len() != self.block_elems {
            return Err(Error::KvCache(format!(
                "block element length mismatch: expected {}, got k={}, v={}",
                self.block_elems,
                k.len(),
                v.len()
            )));
        }
        self.host_ram_cache.insert(block_id, (k, v));
        self.block_tiers.insert(block_id, CacheTier::HostRam);
        Ok(())
    }

    /// Demotes a block from Host RAM to NVMe disk cache, freeing RAM space.
    pub fn demote_to_nvme(&mut self, block_id: BlockId) -> Result<()> {
        if let Some((k, v)) = self.host_ram_cache.remove(&block_id) {
            let temp_path = self.scratch_dir.join(format!("tmp_kv_block_{}.bin", block_id));
            let file_path = self.scratch_dir.join(format!("kv_block_{}.bin", block_id));

            let write_res = (|| -> Result<()> {
                let mut file = File::create(&temp_path).map_err(|e| Error::KvCache(e.to_string()))?;
                let k_bytes: &[u8] =
                    unsafe { std::slice::from_raw_parts(k.as_ptr() as *const u8, k.len() * 4) };
                let v_bytes: &[u8] =
                    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };

                file.write_all(k_bytes)
                    .map_err(|e| Error::KvCache(e.to_string()))?;
                file.write_all(v_bytes)
                    .map_err(|e| Error::KvCache(e.to_string()))?;
                file.sync_all().map_err(|e| Error::KvCache(e.to_string()))?;
                Ok(())
            })();

            if let Err(e) = write_res {
                let _ = fs::remove_file(&temp_path);
                // Put back in host cache if write failed
                self.host_ram_cache.insert(block_id, (k, v));
                return Err(e);
            }

            if let Err(e) = fs::rename(&temp_path, &file_path) {
                let _ = fs::remove_file(&temp_path);
                self.host_ram_cache.insert(block_id, (k, v));
                return Err(Error::KvCache(format!("atomic rename failed: {e}")));
            }

            self.nvme_cache.insert(block_id, file_path);
            self.block_tiers.insert(block_id, CacheTier::NvMe);
        }
        Ok(())
    }

    /// Retrieves a block's contents from whichever tier it currently resides in.
    /// Returns `None` if the block is on GPU or unmanaged.
    pub fn retrieve(&mut self, block_id: BlockId) -> Result<Option<(Vec<f32>, Vec<f32>)>> {
        let tier = match self.block_tiers.get(&block_id) {
            Some(t) => *t,
            None => return Ok(None),
        };

        match tier {
            CacheTier::Gpu => Ok(None),
            CacheTier::HostRam => Ok(self.host_ram_cache.get(&block_id).cloned()),
            CacheTier::NvMe => {
                if let Some(path) = self.nvme_cache.remove(&block_id) {
                    let read_res = (|| -> Result<(Vec<f32>, Vec<f32>)> {
                        let mut file = File::open(&path).map_err(|e| Error::KvCache(e.to_string()))?;
                        let mut k = vec![0.0f32; self.block_elems];
                        let mut v = vec![0.0f32; self.block_elems];

                        let k_bytes: &mut [u8] = unsafe {
                            std::slice::from_raw_parts_mut(k.as_mut_ptr() as *mut u8, k.len() * 4)
                        };
                        let v_bytes: &mut [u8] = unsafe {
                            std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, v.len() * 4)
                        };

                        file.read_exact(k_bytes)
                            .map_err(|e| Error::KvCache(e.to_string()))?;
                        file.read_exact(v_bytes)
                            .map_err(|e| Error::KvCache(e.to_string()))?;
                        Ok((k, v))
                    })();

                    match read_res {
                        Ok((k, v)) => {
                            // Remove disk spill file on promotion to Host RAM
                            let _ = fs::remove_file(&path);
                            self.host_ram_cache.insert(block_id, (k.clone(), v.clone()));
                            self.block_tiers.insert(block_id, CacheTier::HostRam);
                            Ok(Some((k, v)))
                        }
                        Err(e) => {
                            // Put back the path if read failed
                            self.nvme_cache.insert(block_id, path);
                            Err(e)
                        }
                    }
                } else {
                    Err(Error::KvCache("NVMe block path missing".into()))
                }
            }
            CacheTier::NvMeWeightStream => {
                // Weight streaming blocks are not managed as standard KV pairs in the local spill retrieve.
                Ok(None)
            }
        }
    }

    /// Evicts / deletes a block entirely from tiered caches.
    pub fn evict(&mut self, block_id: BlockId) {
        self.block_tiers.remove(&block_id);
        self.host_ram_cache.remove(&block_id);
        if let Some(path) = self.nvme_cache.remove(&block_id) {
            let _ = fs::remove_file(path);
        }
    }

    /// Gets the current storage tier of a block.
    pub fn get_tier(&self, block_id: BlockId) -> Option<CacheTier> {
        self.block_tiers.get(&block_id).copied()
    }

    /// Retargets the rotary position embedding of a cached block in Host RAM from
    /// `old_start_pos` to `new_start_pos` using CPU Re-RoPE without re-prefill.
    pub fn retarget_block_positions(
        &mut self,
        block_id: BlockId,
        old_start_pos: usize,
        new_start_pos: usize,
        tokens_per_block: usize,
        head_dim: usize,
        num_heads: usize,
        base_freq: f32,
    ) -> Result<()> {
        let (k_data, v_data) = self.retrieve(block_id)?.ok_or_else(|| {
            Error::KvCache(format!("block {} not found in cache for retargeting", block_id))
        })?;

        let expected_elems = tokens_per_block * num_heads * head_dim;
        if k_data.len() != expected_elems {
            return Err(Error::KvCache(format!(
                "retarget_block_positions: block elements {} does not match expected {}",
                k_data.len(), expected_elems
            )));
        }

        let dev = grim_backend_cpu::CpuDevice::new();
        let k_storage = grim_backend_cpu::CpuStorage::new(
            k_data,
            grim_tensor::shape::Shape::new(vec![num_heads, tokens_per_block, head_dim]),
            grim_tensor::dtype::DType::F32,
        );

        let old_positions: Vec<u32> = (0..tokens_per_block)
            .map(|i| (old_start_pos + i) as u32)
            .collect();
        let new_positions: Vec<u32> = (0..tokens_per_block)
            .map(|i| (new_start_pos + i) as u32)
            .collect();

        let cfg = grim_tensor::RopeConfig::new(head_dim, base_freq);

        let (retargeted_k, _) = dev.rerope(
            &k_storage,
            &old_positions,
            &new_positions,
            &cfg,
            &grim_tensor::shape::Shape::new(vec![num_heads, tokens_per_block, head_dim]),
        ).map_err(|e| Error::KvCache(e.to_string()))?;

        let new_k_vec = retargeted_k.to_cpu_vec_f32().map_err(|e| Error::KvCache(e.to_string()))?;
        self.host_ram_cache.insert(block_id, (new_k_vec, v_data));
        self.block_tiers.insert(block_id, CacheTier::HostRam);
        Ok(())
    }
}

impl Drop for LocalSpillManager {
    fn drop(&mut self) {
        // Clean up all temporary files on exit
        for path in self.nvme_cache.values() {
            let _ = fs::remove_file(path);
        }
    }
}

/// Shared wrapper for multi-threaded access.
pub struct SharedSpillManager {
    inner: RwLock<LocalSpillManager>,
}

impl SharedSpillManager {
    pub fn new(scratch_dir: PathBuf, block_elems: usize) -> Result<Self> {
        Ok(Self {
            inner: RwLock::new(LocalSpillManager::new(scratch_dir, block_elems)?),
        })
    }

    pub fn demote_to_host(&self, block_id: BlockId, k: Vec<f32>, v: Vec<f32>) -> Result<()> {
        self.inner.write().demote_to_host(block_id, k, v)
    }

    pub fn demote_to_nvme(&self, block_id: BlockId) -> Result<()> {
        self.inner.write().demote_to_nvme(block_id)
    }

    pub fn retrieve(&self, block_id: BlockId) -> Result<Option<(Vec<f32>, Vec<f32>)>> {
        self.inner.write().retrieve(block_id)
    }

    pub fn evict(&self, block_id: BlockId) {
        self.inner.write().evict(block_id);
    }

    pub fn get_tier(&self, block_id: BlockId) -> Option<CacheTier> {
        self.inner.read().get_tier(block_id)
    }
}

// ── Network KV transport wire protocol ────────────────────────────────────────

/// Wire-protocol magic for the V2 header. 0x4B56434B = "KVCK" in ASCII.
const KV_MAGIC: u32 = 0x4B56434B;

/// Current on-wire protocol version.
const KV_PROTOCOL_VERSION: u32 = 2;

/// Fixed-size header (28 bytes) prepended to every KV block transfer.
///
/// Layout (all little-endian):
/// | magic (u32) | version (u32) | block_id (u64) |
/// | layer_idx (u32) | num_elements (u32) | checksum (u32) |
#[derive(Debug, Clone, Copy)]
pub struct KvBlockHeader {
    pub magic: u32,
    pub version: u32,
    pub block_id: u64,
    pub layer_idx: u32,
    pub num_elements: u32,
    pub checksum: u32,
}

impl KvBlockHeader {
    pub const SIZE: usize = 28; // 4+4+8+4+4+4

    /// Serialise the header to a 28-byte little-endian buffer.
    pub fn serialize(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..4].copy_from_slice(&self.magic.to_le_bytes());
        buf[4..8].copy_from_slice(&self.version.to_le_bytes());
        buf[8..16].copy_from_slice(&self.block_id.to_le_bytes());
        buf[16..20].copy_from_slice(&self.layer_idx.to_le_bytes());
        buf[20..24].copy_from_slice(&self.num_elements.to_le_bytes());
        buf[24..28].copy_from_slice(&self.checksum.to_le_bytes());
        buf
    }

    /// Deserialise a header from a byte slice.  Returns `None` if the slice
    /// is too short.
    pub fn deserialize(buf: &[u8]) -> Option<Self> {
        if buf.len() < Self::SIZE {
            return None;
        }
        let magic = u32::from_le_bytes(buf[0..4].try_into().ok()?);
        let version = u32::from_le_bytes(buf[4..8].try_into().ok()?);
        let block_id = u64::from_le_bytes(buf[8..16].try_into().ok()?);
        let layer_idx = u32::from_le_bytes(buf[16..20].try_into().ok()?);
        let num_elements = u32::from_le_bytes(buf[20..24].try_into().ok()?);
        let checksum = u32::from_le_bytes(buf[24..28].try_into().ok()?);
        Some(Self {
            magic,
            version,
            block_id,
            layer_idx,
            num_elements,
            checksum,
        })
    }

    /// Verify the magic number and protocol version.
    pub fn verify(&self) -> bool {
        self.magic == KV_MAGIC && self.version == KV_PROTOCOL_VERSION
    }
}

/// FNV-1a 32-bit checksum over raw bytes.  A simple non-cryptographic
/// checksum sufficient to detect truncation or bit-corruption on the wire.
pub fn compute_checksum_bytes(bytes: &[u8]) -> u32 {
    let mut hash: u32 = 0x811c9275; // FNV offset basis
    for &b in bytes {
        hash ^= b as u32;
        hash = hash.wrapping_mul(0x01000193); // FNV prime
    }
    hash
}

/// FNV-1a 32-bit checksum over the raw bytes of the key and value float slices,
/// preserving exact bit representations for all IEEE-754 payloads including NaNs.
pub fn compute_checksum(k: &[f32], v: &[f32]) -> u32 {
    let k_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(k.as_ptr() as *const u8, k.len() * 4) };
    let v_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
    let mut hash: u32 = 0x811c9275;
    for &b in k_bytes.iter().chain(v_bytes.iter()) {
        hash ^= b as u32;
        hash = hash.wrapping_mul(0x01000193);
    }
    hash
}

/// Trait abstracting the operations a network KV receiver needs from a block
/// store.  Defined here (in grim-kvtransport) to avoid a circular dependency:
/// grim-memory depends on grim-kvtransport, so it can implement this trait
/// for `KvBlockPool`, but grim-kvtransport cannot depend on grim-memory.
pub trait KvBlockStore: Send + Sync {
    /// Total number of physical blocks in the pool.
    fn num_blocks(&self) -> usize;
    /// Number of f32 elements per token (num_heads * head_dim).
    fn block_elem_per_token(&self) -> usize;
    /// Maximum number of tokens one block can hold (BLOCK_SIZE).
    fn block_size(&self) -> usize;
    /// Write key data into `id`'s block.  `num_tokens` is the number of tokens
    /// written (capped at `block_size()` internally by the pool).
    fn write_keys(&mut self, id: BlockId, keys: &[f32], num_tokens: usize);
    /// Write value data into `id`'s block.  Uses the `num_tokens` previously
    /// set by `write_keys`.
    fn write_values(&mut self, id: BlockId, values: &[f32]);
    /// Whether block `id` has received real KV data (via `write_keys`,
    /// `store_kv`, or network ingestion).  Replaces the fragile non-zero
    /// content sniff: a genuinely all-zero KV block is valid data, not
    /// "not yet arrived."
    fn block_is_received(&self, id: BlockId) -> bool;

    /// Read key data for layer 0 of block `id`.  Returns `None` when the
    /// block is out of range — pull-mode fetch (F8/F10) turns that into a
    /// "not available" reply instead of deadlocking or fabricating data.
    fn read_keys(&self, id: BlockId) -> Option<Vec<f32>>;
    /// Read value data for layer 0 of block `id`.  Returns `None` when the
    /// block is out of range.
    fn read_values(&self, id: BlockId) -> Option<Vec<f32>>;

    /// Read key data for a specific layer of block `id`.  Defaults to the
    /// layer-0 behavior, mirroring the write-side default.
    fn read_layer_keys(&self, id: BlockId, layer_idx: u32) -> Option<Vec<f32>> {
        if layer_idx == 0 {
            self.read_keys(id)
        } else {
            None
        }
    }

    /// Read value data for a specific layer of block `id`.  Defaults to the
    /// layer-0 behavior, mirroring the write-side default.
    fn read_layer_values(&self, id: BlockId, layer_idx: u32) -> Option<Vec<f32>> {
        if layer_idx == 0 {
            self.read_values(id)
        } else {
            None
        }
    }

    /// Write key data into `id`'s block for a specific layer.
    fn write_layer_keys(&mut self, id: BlockId, layer_idx: u32, keys: &[f32], num_tokens: usize) {
        if layer_idx == 0 {
            self.write_keys(id, keys, num_tokens);
        }
    }

    /// Write value data into `id`'s block for a specific layer.
    fn write_layer_values(&mut self, id: BlockId, layer_idx: u32, values: &[f32]) {
        if layer_idx == 0 {
            self.write_values(id, values);
        }
    }
}

/// Network transport layer for network-based (RDMA/TCP) KV handoffs.
pub struct NetworkKvClient {
    pub local_ip: String,
}

impl NetworkKvClient {
    /// Creates a new network KV transport client bound to the specified local IP interface address.
    pub fn new(local_ip: String) -> Self {
        Self { local_ip }
    }

    /// Resolve a target specifier into a `host:port` string.
    fn resolve_addr(target_ip: &str) -> String {
        if target_ip.contains(':') {
            target_ip.to_string()
        } else {
            format!("{target_ip}:9190")
        }
    }

    /// Dispatches a KV block key/value payload buffer to a target remote IP
    /// endpoint over a TCP/network stream using the V2 wire protocol.
    ///
    /// The protocol sends a 28-byte header (magic, version, block_id,
    /// layer_idx, num_elements, checksum) followed by the raw f32 bytes of
    /// the key slice and then the value slice.
    pub fn send_block_remote(
        &self,
        block_id: BlockId,
        layer_idx: u32,
        k: &[f32],
        v: &[f32],
        target_ip: &str,
    ) -> Result<()> {
        if k.len() != v.len() {
            return Err(Error::KvCache(
                "Key and Value slice lengths must match for block transport".into(),
            ));
        }
        if k.is_empty() {
            return Err(Error::KvCache("Cannot send an empty KV block".into()));
        }
        let addr = Self::resolve_addr(target_ip);
        let checksum = compute_checksum(k, v);
        let header = KvBlockHeader {
            magic: KV_MAGIC,
            version: KV_PROTOCOL_VERSION,
            block_id: block_id as u64,
            layer_idx,
            num_elements: k.len() as u32,
            checksum,
        };

        let mut buf = Vec::with_capacity(KvBlockHeader::SIZE + k.len() * 8);
        buf.extend_from_slice(&header.serialize());
        for &val in k.iter().chain(v.iter()) {
            buf.extend_from_slice(&val.to_le_bytes());
        }

        let socket_addr = addr
            .parse()
            .map_err(|e| Error::KvCache(format!("Invalid target IP address '{target_ip}': {e}")))?;

        let mut stream = std::net::TcpStream::connect_timeout(
            &socket_addr,
            std::time::Duration::from_millis(500),
        )
        .map_err(|e| Error::KvCache(format!("TCP send block connection failed to {addr}: {e}")))?;

        stream
            .write_all(&buf)
            .map_err(|e| Error::KvCache(format!("TCP send block error: {e}")))?;
        Ok(())
    }

    /// Fetches a key/value payload block from a remote IP endpoint over a TCP stream.
    ///
    /// Sends a V2 fetch request (header only) and receives a V2 response
    /// containing the key and value data.  Returns an error if the remote
    /// endpoint is unreachable or the response fails validation — never
    /// fabricates data.
    pub fn fetch_block_remote(
        &self,
        block_id: BlockId,
        layer_idx: u32,
        target_ip: &str,
        block_elems: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let addr = Self::resolve_addr(target_ip);
        let socket_addr = addr
            .parse()
            .map_err(|e| Error::KvCache(format!("Invalid target IP address '{target_ip}': {e}")))?;

        // Build a fetch-request header: same format but with a zero checksum
        // and a special request flag in layer_idx (bit 31 set). Mask bit 31
        // out of the caller's layer so a stray high bit can't smuggle a
        // second flag into the request.
        let req_header = KvBlockHeader {
            magic: KV_MAGIC,
            version: KV_PROTOCOL_VERSION,
            block_id: block_id as u64,
            layer_idx: (layer_idx & !FETCH_REQUEST_FLAG) | FETCH_REQUEST_FLAG,
            num_elements: block_elems as u32,
            checksum: 0,
        };

        let mut stream = std::net::TcpStream::connect_timeout(
            &socket_addr,
            std::time::Duration::from_millis(500),
        )
        .map_err(|e| Error::KvCache(format!("TCP fetch connection failed to {addr}: {e}")))?;
        // Deadline the whole exchange, not just the connect: a server that
        // never answers (protocol mismatch, wedged peer) must fail fast
        // instead of hanging the caller's read_exact forever (F8 follow-up).
        const IO_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
        stream
            .set_read_timeout(Some(IO_TIMEOUT))
            .map_err(|e| Error::KvCache(format!("TCP fetch: set_read_timeout failed: {e}")))?;
        stream
            .set_write_timeout(Some(IO_TIMEOUT))
            .map_err(|e| Error::KvCache(format!("TCP fetch: set_write_timeout failed: {e}")))?;

        stream
            .write_all(&req_header.serialize())
            .map_err(|e| Error::KvCache(format!("TCP fetch request error: {e}")))?;

        // Read the response header
        let mut hdr = [0u8; KvBlockHeader::SIZE];
        stream
            .read_exact(&mut hdr)
            .map_err(|e| Error::KvCache(format!("TCP fetch header read error: {e}")))?;
        let header = KvBlockHeader::deserialize(&hdr)
            .ok_or_else(|| Error::KvCache("TCP fetch: invalid header size".into()))?;
        if !header.verify() {
            return Err(Error::KvCache(format!(
                "TCP fetch: protocol mismatch magic={:#x} version={}",
                header.magic, header.version
            )));
        }
        // The receiver answers "block/layer not available here" with an
        // empty payload rather than silence — translate that into an error
        // so no caller mistakes it for a legitimate zero-length block.
        if header.num_elements == 0 {
            return Err(Error::KvCache(format!(
                "TCP fetch: remote {addr} reports block {block_id} layer {} not available",
                layer_idx & !FETCH_REQUEST_FLAG
            )));
        }

        // Read the response payload
        // KVT-1: Cap allocation to prevent DoS from malicious num_elements.
        const MAX_PAYLOAD_BYTES: usize = 512 * 1024 * 1024; // 512 MiB
        let total_bytes = (header.num_elements as usize)
            .checked_mul(8)
            .filter(|&b| b <= MAX_PAYLOAD_BYTES)
            .ok_or_else(|| {
                Error::KvCache(format!(
                    "TCP fetch: num_elements={} overflows or exceeds cap",
                    header.num_elements
                ))
            })?;
        let mut payload = vec![0u8; total_bytes];
        stream
            .read_exact(&mut payload)
            .map_err(|e| Error::KvCache(format!("TCP fetch read error: {e}")))?;

        // Verify checksum
        let k_vec = parse_f32_slice(&payload[..header.num_elements as usize * 4]);
        let v_vec = parse_f32_slice(&payload[header.num_elements as usize * 4..]);
        let expected = compute_checksum(&k_vec, &v_vec);
        if expected != header.checksum {
            return Err(Error::KvCache(format!(
                "TCP fetch: checksum mismatch (expected {expected:#x}, got {:#x})",
                header.checksum
            )));
        }

        Ok((k_vec, v_vec))
    }

    /// Send a prompt-token control message to a remote node (push model).
    ///
    /// This is the real control channel for disaggregated handoff: the
    /// prefill→decode (or decode→prefill) side learns WHICH tokens a request
    /// carries without smuggling them through a fake KV block. Payload is the
    /// raw token IDs as little-endian u32; `request_id` rides in `block_id`.
    pub fn send_prompt_tokens(
        &self,
        request_id: u64,
        tokens: &[u32],
        target_ip: &str,
    ) -> Result<()> {
        if tokens.is_empty() {
            return Err(Error::KvCache(
                "send_prompt_tokens: token list cannot be empty".into(),
            ));
        }
        let addr = Self::resolve_addr(target_ip);
        let mut payload = Vec::with_capacity(tokens.len() * 4);
        for &t in tokens {
            payload.extend_from_slice(&t.to_le_bytes());
        }
        let header = KvBlockHeader {
            magic: KV_MAGIC,
            version: KV_PROTOCOL_VERSION,
            block_id: request_id,
            layer_idx: PROMPT_FLAG,
            num_elements: tokens.len() as u32,
            checksum: compute_checksum_bytes(&payload),
        };
        let mut buf = header.serialize().to_vec();
        buf.extend_from_slice(&payload);

        let socket_addr = addr
            .parse()
            .map_err(|e| Error::KvCache(format!("Invalid target IP address '{target_ip}': {e}")))?;
        let mut stream = std::net::TcpStream::connect_timeout(
            &socket_addr,
            std::time::Duration::from_millis(500),
        )
        .map_err(|e| Error::KvCache(format!("TCP prompt send connection failed to {addr}: {e}")))?;
        stream
            .write_all(&buf)
            .map_err(|e| Error::KvCache(format!("TCP prompt send error: {e}")))?;
        Ok(())
    }
}

/// Bit flag embedded in `layer_idx` of a fetch-request header to signal
/// "this is a fetch request, reply with data" rather than a push transfer.
const FETCH_REQUEST_FLAG: u32 = 0x8000_0000;

/// Bit flag embedded in `layer_idx` marking a prompt-token control message:
/// the payload is raw u32 token IDs, `block_id` carries the request id, and
/// the receiver stores it in the shared [`PromptChannel`] instead of the KV
/// block store.
const PROMPT_FLAG: u32 = 0x4000_0000;

/// Shared store for prompt-token control messages received over the wire
/// (see [`NetworkKvClient::send_prompt_tokens`]). Cloneable handle; `take`
/// consumes the stored prompt for a request id.
#[derive(Default, Clone)]
pub struct PromptChannel {
    inner: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<u64, Vec<u32>>>>,
}

impl PromptChannel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Store (or overwrite) the prompt tokens for a request id.
    pub fn store(&self, request_id: u64, tokens: Vec<u32>) {
        self.inner
            .lock()
            .expect("prompt channel mutex poisoned")
            .insert(request_id, tokens);
    }

    /// Consume the stored prompt tokens for a request id, if any.
    pub fn take(&self, request_id: u64) -> Option<Vec<u32>> {
        self.inner
            .lock()
            .expect("prompt channel mutex poisoned")
            .remove(&request_id)
    }

    /// Whether a prompt is waiting for `request_id` (without consuming).
    pub fn contains(&self, request_id: u64) -> bool {
        self.inner
            .lock()
            .expect("prompt channel mutex poisoned")
            .contains_key(&request_id)
    }
}

/// Reinterpret a byte slice as f32 values (little-endian). Safe parse — no
/// unsafe pointer casts, avoids alignment UB on Vec<u8> buffers.
fn parse_f32_slice(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            let arr: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
            f32::from_le_bytes(arr)
        })
        .collect()
}

/// Maximum tokens allowed in a single prompt-token control message (1M tokens).
pub const MAX_PROMPT_TOKENS: usize = 1_000_000;

/// Spawns a background TCP server that listens for incoming
/// `NetworkKvClient::send_block_remote` streams and writes them into a shared
/// KV block store.
///
/// The server runs in a background OS thread, accepting one connection per
/// transferred block.  Each connection is expected to send a 28-byte
/// [`KvBlockHeader`] followed by the key slice and value slice as raw
/// little-endian f32 bytes.  The magic number and checksum are verified
/// before the data is committed to the store.
///
/// Accepts any type implementing [`KvBlockStore`] behind a
/// `std::sync::Mutex` — this avoids a circular dependency on grim-memory
/// (grim-memory already depends on grim-kvtransport; KvBlockPool implements
/// the trait in grim-memory).
pub fn start_kv_receiver_server<T>(
    listen_addr: &str,
    pool: std::sync::Arc<std::sync::Mutex<T>>,
) -> Result<std::thread::JoinHandle<()>>
where
    T: KvBlockStore + 'static,
{
    start_kv_receiver_server_with_prompts(listen_addr, pool, PromptChannel::new())
}

/// Like [`start_kv_receiver_server`], but prompt-token control messages
/// ([`NetworkKvClient::send_prompt_tokens`]) are stored into the supplied
/// [`PromptChannel`] instead of being dropped.
pub fn start_kv_receiver_server_with_prompts<T>(
    listen_addr: &str,
    pool: std::sync::Arc<std::sync::Mutex<T>>,
    prompts: PromptChannel,
) -> Result<std::thread::JoinHandle<()>>
where
    T: KvBlockStore + 'static,
{
    use std::net::TcpListener;

    let listener = TcpListener::bind(listen_addr).map_err(|e| {
        Error::KvCache(format!(
            "start_kv_receiver_server: bind failed on {listen_addr}: {e}"
        ))
    })?;
    listener
        .set_nonblocking(true)
        .map_err(|e| Error::KvCache(format!("set_nonblocking failed: {e}")))?;

    let addr_str = listen_addr.to_string();
    let handle = std::thread::spawn(move || {
        eprintln!("[grim-kvtransport] KV receiver listening on {addr_str}");
        loop {
            match listener.accept() {
                Ok((mut stream, _peer)) => {
                    // Read the fixed-size header.
                    let mut hdr = [0u8; KvBlockHeader::SIZE];
                    if stream.read_exact(&mut hdr).is_err() {
                        continue;
                    }
                    let header = match KvBlockHeader::deserialize(&hdr) {
                        Some(h) => h,
                        None => continue,
                    };

                    // Reject protocol mismatches immediately.
                    if !header.verify() {
                        eprintln!(
                            "[grim-kvtransport] KV receiver: rejecting connection \
                             — bad magic={:#x} version={}",
                            header.magic, header.version
                        );
                        continue;
                    }

                    // F8/F10: a fetch request (FETCH_REQUEST_FLAG set in
                    // layer_idx) asks the server to REPLY with the block's
                    // data instead of pushing a payload. The old loop fell
                    // through to the push path unconditionally and blocked
                    // reading a payload the fetcher never sends — a
                    // deadlock between both sides on the first pull-mode
                    // fetch. Answer from the store's read side; an empty
                    // payload means "not available here".
                    if header.layer_idx & FETCH_REQUEST_FLAG != 0 {
                        let layer_idx = header.layer_idx & !FETCH_REQUEST_FLAG;
                        let block_id = header.block_id as usize;
                        let (k_data, v_data) = {
                            let guard = pool.lock().unwrap_or_else(|e| e.into_inner());
                            if block_id < guard.num_blocks() && guard.block_is_received(block_id) {
                                match (
                                    guard.read_layer_keys(block_id, layer_idx),
                                    guard.read_layer_values(block_id, layer_idx),
                                ) {
                                    (Some(k), Some(v)) if k.len() == v.len() && !k.is_empty() => {
                                        (k, v)
                                    }
                                    _ => (Vec::new(), Vec::new()),
                                }
                            } else {
                                (Vec::new(), Vec::new())
                            }
                        };
                        let resp = KvBlockHeader {
                            magic: KV_MAGIC,
                            version: KV_PROTOCOL_VERSION,
                            block_id: header.block_id,
                            layer_idx,
                            num_elements: k_data.len() as u32,
                            checksum: compute_checksum(&k_data, &v_data),
                        };
                        let mut buf = resp.serialize().to_vec();
                        for &val in k_data.iter().chain(v_data.iter()) {
                            buf.extend_from_slice(&val.to_le_bytes());
                        }
                        if let Err(e) = stream.write_all(&buf) {
                            eprintln!(
                                "[grim-kvtransport] KV receiver: fetch reply for block {} \
                                 layer {layer_idx} failed: {e}",
                                header.block_id
                            );
                        }
                        continue;
                    }

                    // Prompt-token control message: payload is raw u32 token
                    // IDs (request id rides in block_id). Store into the
                    // prompt channel — never into the KV block store.
                    if header.layer_idx & PROMPT_FLAG != 0 {
                        let num_tokens = header.num_elements as usize;
                        if num_tokens > MAX_PROMPT_TOKENS {
                            eprintln!(
                                "[grim-kvtransport] KV receiver: prompt num_tokens {num_tokens} \
                                 exceeds safety cap {MAX_PROMPT_TOKENS}"
                            );
                            continue;
                        }
                        let mut payload = vec![0u8; num_tokens.saturating_mul(4)];
                        if stream.read_exact(&mut payload).is_err() {
                            eprintln!(
                                "[grim-kvtransport] KV receiver: short read on prompt message \
                                 for request {}",
                                header.block_id
                            );
                            continue;
                        }
                        if compute_checksum_bytes(&payload) != header.checksum {
                            eprintln!(
                                "[grim-kvtransport] KV receiver: checksum mismatch on prompt \
                                 message for request {}",
                                header.block_id
                            );
                            continue;
                        }
                        let tokens: Vec<u32> = payload
                            .chunks_exact(4)
                            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();
                        prompts.store(header.block_id, tokens);
                        continue;
                    }

                    let num_elems = header.num_elements as usize;
                    if num_elems > 100_000_000 {
                        eprintln!(
                            "[grim-kvtransport] KV receiver: num_elements {num_elems} exceeds safety cap"
                        );
                        continue;
                    }
                    let total_bytes = match num_elems.checked_mul(8) {
                        Some(b) => b,
                        None => {
                            eprintln!(
                                "[grim-kvtransport] KV receiver: num_elements {num_elems} multiplied by 8 overflowed usize"
                            );
                            continue;
                        }
                    };
                    let mut payload = vec![0u8; total_bytes];
                    if stream.read_exact(&mut payload).is_err() {
                        eprintln!(
                            "[grim-kvtransport] KV receiver: short read on \
                             block {}",
                            header.block_id
                        );
                        continue;
                    }

                    // Split payload into key/value float slices and verify checksum.
                    let k_bytes = &payload[..num_elems * 4];
                    let v_bytes = &payload[num_elems * 4..];
                    let k_data = parse_f32_slice(k_bytes);
                    let v_data = parse_f32_slice(v_bytes);
                    let computed = compute_checksum(&k_data, &v_data);
                    if computed != header.checksum {
                        eprintln!(
                            "[grim-kvtransport] KV receiver: checksum mismatch \
                             for block {} (expected {:#x}, got {:#x})",
                            header.block_id, header.checksum, computed
                        );
                        continue;
                    }

                    // Write into the pool.
                    let mut guard = pool.lock().unwrap_or_else(|e| e.into_inner());
                    if header.block_id < guard.num_blocks() as u64 {
                        let block_id = header.block_id as usize;
                        let elem_per_token = guard.block_elem_per_token();
                        // The sender computed `num_elems` using its OWN pool's
                        // `block_elem_per_token()`. If that differs from this
                        // receiver's value, the token count derived below is
                        // wrong. Detect the mismatch (a sender/receiver
                        // `elem_per_token` disagreement makes `num_elems`
                        // non-divisible by the receiver's value) and warn, but
                        // keep sizing off the receiver's value so the write
                        // never overflows the block.
                        if elem_per_token == 0 {
                            eprintln!(
                                "[grim-kvtransport] KV receiver: block_elem_per_token is zero; \
                                 cannot size block {}",
                                header.block_id
                            );
                        } else if num_elems % elem_per_token != 0 {
                            eprintln!(
                                "[grim-kvtransport] KV receiver: num_elements {num_elems} not \
                                 divisible by block_elem_per_token {elem_per_token} for block {} \
                                 — possible sender/receiver elem_per_token mismatch; using \
                                 receiver value",
                                header.block_id
                            );
                        }
                        let num_tokens = num_elems
                            .checked_div(elem_per_token)
                            .unwrap_or(num_elems)
                            .min(guard.block_size());
                        guard.write_layer_keys(block_id, header.layer_idx, &k_data, num_tokens);
                        guard.write_layer_values(block_id, header.layer_idx, &v_data);
                    } else {
                        eprintln!(
                            "[grim-kvtransport] KV receiver: block_id {} out of range",
                            header.block_id
                        );
                    }
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    // No incoming connection — spin briefly.
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(_) => continue,
            }
        }
    });

    Ok(handle)
}

/// Reads one layer's weights from the configured NVMe weights file.
///
/// The file is treated as a flat sequence of `f32` values sectioned by
/// `LAYER_ELEMS` per layer (1024 floats = 4096 bytes per layer). Reads the
/// layer's slice via `pread` so we don't mutate a shared file offset.
///
/// Returns an explicit `KvCache` error if the file is missing, too short for
/// the requested layer, or the I/O call fails — never substitutes mock data
/// (sims.md issue #3).
fn read_layer_weights(
    weights_path: &std::path::Path,
    layer_id: usize,
    layer_elems: usize,
    layer_bytes: usize,
) -> Result<Vec<f32>> {
    use std::io::{Read, Seek, SeekFrom};

    if !weights_path.exists() {
        return Err(Error::KvCache(format!(
            "NVMe weights file not found at {:?}; cannot prefetch layer {}",
            weights_path, layer_id
        )));
    }

    let mut file = std::fs::File::open(weights_path).map_err(|e| {
        Error::KvCache(format!(
            "failed to open NVMe weights {:?}: {}",
            weights_path, e
        ))
    })?;

    let offset = (layer_id as u64)
        .checked_mul(layer_bytes as u64)
        .ok_or_else(|| {
            Error::KvCache(format!(
                "offset overflow calculating layer {layer_id} * {layer_bytes}"
            ))
        })?;
    let metadata = file
        .metadata()
        .map_err(|e| Error::KvCache(format!("failed to stat NVMe weights: {}", e)))?;
    let required_len = offset.checked_add(layer_bytes as u64).ok_or_else(|| {
        Error::KvCache(format!(
            "offset + layer_bytes overflow calculating layer {layer_id}"
        ))
    })?;
    if metadata.len() < required_len {
        return Err(Error::KvCache(format!(
            "NVMe weights file {:?} too short for layer {}: have {} bytes, need {} at offset {}",
            weights_path,
            layer_id,
            metadata.len(),
            layer_bytes,
            offset
        )));
    }

    file.seek(SeekFrom::Start(offset))
        .map_err(|e| Error::KvCache(format!("seek failed on NVMe weights: {}", e)))?;

    let mut bytes = vec![0u8; layer_bytes];
    file.read_exact(&mut bytes)
        .map_err(|e| Error::KvCache(format!("read failed on NVMe weights: {}", e)))?;

    // Reinterpret the bytes as a little-endian f32 array.
    let mut weights = vec![0.0f32; layer_elems];
    for (i, w) in weights.iter_mut().enumerate() {
        let start = i * std::mem::size_of::<f32>();
        *w = f32::from_le_bytes([
            bytes[start],
            bytes[start + 1],
            bytes[start + 2],
            bytes[start + 3],
        ]);
    }
    Ok(weights)
}

/// Double-buffered weight prefetch engine for NVMe layer/unit streaming.
///
/// Originally framed as a "layer" streamer with a fixed 1024-element per-layer
/// assumption; now generalised so callers specify `unit_elems` at construction
/// time. An "embedding unit" maps naturally to this framing: `unit_id` is
/// "which row-block of the vocabulary" and `unit_elems` is "floats per row-block"
/// (e.g. 4096 vocab-rows × 128 hidden-dim = 524 288 floats per unit).
///
/// When a unit is evicted from the host-RAM LRU to NVMe, its tier transitions
/// to `CacheTier::NvMeWeightStream` in `unit_tier_map`. Call [`NvmeWeightStreamer::get_unit_tier`]
/// to query placement, mirroring `LocalSpillManager::get_tier` for KV blocks.
pub struct NvmeWeightStreamer {
    /// LRU unit cache capacity (number of units held in host RAM simultaneously).
    pub lru_capacity_layers: usize,
    /// NVMe file path for model weights / embedding table.
    pub weights_path: PathBuf,
    /// Number of f32 elements per unit (replaces the former `const LAYER_ELEMS = 1024`).
    /// Set once at construction; never changes after that.
    pub unit_elems: usize,
    /// Host RAM LRU weight cache: unit_id → f32 data.
    host_weight_cache: Mutex<HashMap<usize, Vec<f32>>>,
    /// LRU access order (front = oldest, back = most-recently-used).
    lru_order: Mutex<Vec<usize>>,
    /// Double buffers for async weight prefetching (active / transfer).
    double_buffers: Mutex<(Vec<f32>, Vec<f32>)>,
    /// Simulated io_uring submission status flag.
    uring_submitting: Mutex<bool>,
    /// Current transfer bandwidth usage (bytes/sec) for PCIe backpressure.
    bandwidth_usage: Mutex<f64>,
    /// Tier tracking per unit: populated when a unit is evicted to NVMe so
    /// `grim-scheduler` can inspect embedding-table placement the same way
    /// it inspects KV-block placement via `LocalSpillManager::get_tier`.
    unit_tier_map: Mutex<HashMap<usize, CacheTier>>,
}

impl NvmeWeightStreamer {
    /// Create a new streamer.
    ///
    /// # Parameters
    /// - `weights_path`: Path to the flat f32 weight file (row-major, concatenated units).
    /// - `lru_capacity_layers`: How many units to keep in host RAM simultaneously.
    /// - `unit_elems`: Number of f32 elements per unit. Replaces the former hardcoded
    ///   `LAYER_ELEMS = 1024` constant. Pass `1024` to reproduce the old behaviour for
    ///   tests and callers that have not yet migrated to a larger granularity.
    pub fn new(weights_path: PathBuf, lru_capacity_layers: usize, unit_elems: usize) -> Self {
        assert!(unit_elems > 0, "NvmeWeightStreamer: unit_elems must be greater than 0");
        Self {
            weights_path,
            lru_capacity_layers,
            unit_elems,
            host_weight_cache: Mutex::new(HashMap::new()),
            lru_order: Mutex::new(Vec::new()),
            double_buffers: Mutex::new((vec![], vec![])),
            uring_submitting: Mutex::new(false),
            bandwidth_usage: Mutex::new(0.0),
            unit_tier_map: Mutex::new(HashMap::new()),
        }
    }

    /// Prefetch a target unit's weights asynchronously into pinned CPU RAM.
    ///
    /// In a production environment under Linux, this leverages `io_uring` and
    /// `O_DIRECT`; here we synchronously `pread` the weights file, since the
    /// network/io_uring backend is not yet wired (sims.md issue #3). The
    /// previous implementation inserted hardcoded `vec![0.5f32; 1024]` weights
    /// instead of reading from disk. We now read the real bytes from `weights_path`:
    ///
    /// - If the file is missing or the read fails, we surface an explicit
    ///   `KvCache` error rather than substituting mock data.
    /// - The on-disk layout is a flat stream of `f32` weights sectioned by
    ///   `self.unit_elems` floats per unit. When `unit_elems` is 1024 this
    ///   matches the previous fixed behaviour exactly.
    ///
    /// When a unit is evicted from the host-RAM LRU, its tier is set to
    /// `CacheTier::NvMeWeightStream` in `unit_tier_map` so `grim-scheduler`
    /// can query embedding placement via `get_unit_tier`.
    pub fn prefetch_layer_async(&self, layer_id: usize) -> Result<()> {
        // Bandwidth Admission and Backpressure check:
        // If bandwidth usage exceeds 12.0 GB/s (~PCIe Gen4 x8 saturation),
        // defer the prefetch instead of saturating the link.
        let cur_bandwidth = *self.bandwidth_usage.lock().unwrap();
        if cur_bandwidth > 12.0 * 1024.0 * 1024.0 * 1024.0 {
            return Err(Error::KvCache(
                "PCIe transfer bandwidth limit backpressure triggered".into(),
            ));
        }

        // Use the caller-configured unit size rather than a hardcoded constant.
        let unit_elems = self.unit_elems;
        let unit_bytes = unit_elems * std::mem::size_of::<f32>();

        // Read the unit's weights from the configured NVMe path up-front
        // (before acquiring cache locks) so I/O errors fail loudly instead of
        // leaving the cache half-mutated.
        let weights = read_layer_weights(&self.weights_path, layer_id, unit_elems, unit_bytes)?;

        *self.uring_submitting.lock().unwrap() = true;

        // Populate LRU cache.
        let mut cache = self.host_weight_cache.lock().unwrap();
        let mut order = self.lru_order.lock().unwrap();
        let mut tier_map = self.unit_tier_map.lock().unwrap();

        if !cache.contains_key(&layer_id) {
            // Evict LRU if capacity exceeded; record the evicted unit's tier
            // as NvMeWeightStream so grim-scheduler can see it moved to disk.
            if cache.len() >= self.lru_capacity_layers && !order.is_empty() {
                let evicted = order.remove(0);
                cache.remove(&evicted);
                tier_map.insert(evicted, CacheTier::NvMeWeightStream);
            }

            cache.insert(layer_id, weights.clone());
            order.push(layer_id);
            // Unit now resident in host RAM.
            tier_map.insert(layer_id, CacheTier::HostRam);

            // Populate double buffers (async swap preparation).
            let mut buffers = self.double_buffers.lock().unwrap();
            buffers.1 = weights; // Load into transfer buffer
        } else {
            // Move unit to end of access order (most-recently-used).
            if let Some(pos) = order.iter().position(|&x| x == layer_id) {
                order.remove(pos);
            }
            order.push(layer_id);
            // Ensure tier reflects current HostRam residency.
            tier_map.insert(layer_id, CacheTier::HostRam);
        }

        *self.uring_submitting.lock().unwrap() = false;
        Ok(())
    }

    /// Query the current storage tier of a weight unit.
    ///
    /// Returns `Some(CacheTier::HostRam)` if the unit is in the LRU cache,
    /// `Some(CacheTier::NvMeWeightStream)` if it was evicted to disk, or
    /// `None` if the unit has never been prefetched. This mirrors
    /// `LocalSpillManager::get_tier` so `grim-scheduler` can reason about
    /// embedding-table placement the same way it reasons about KV-block placement.
    pub fn get_unit_tier(&self, unit_id: usize) -> Option<CacheTier> {
        self.unit_tier_map.lock().unwrap().get(&unit_id).copied()
    }

    /// Retrieve the cached weight data for a unit, if present in host RAM.
    ///
    /// Returns `Some(data)` when the unit is cached, `None` when it has been
    /// evicted or never loaded. Use `prefetch_layer_async` first to ensure the
    /// unit is cached before calling this.
    pub fn retrieve_unit(&self, unit_id: usize) -> Option<Vec<f32>> {
        self.host_weight_cache
            .lock()
            .unwrap()
            .get(&unit_id)
            .cloned()
    }

    /// Swaps the target double-buffers to update GPU memory.
    pub fn commit_and_swap(&self, _current_layer: usize, _next_layer: usize) -> Result<()> {
        let mut buffers = self.double_buffers.lock().unwrap();
        // Double-buffered swap: Active buffer becomes transfer buffer and vice versa.
        let (buf0, buf1) = &mut *buffers;
        std::mem::swap(buf0, buf1);
        Ok(())
    }

    /// Update the tracked transfer bandwidth usage (bytes/sec).
    pub fn set_bandwidth_usage(&self, bytes_per_sec: f64) {
        *self.bandwidth_usage.lock().unwrap() = bytes_per_sec;
    }
}

/// Tiered embedding table spill manager.
///
/// Wraps `NvmeWeightStreamer` around a flat embedding weight tensor
/// (shape `[vocab_size, hidden_dim]` row-major) sharded into row-blocks
/// ("units") of `rows_per_unit` vocabulary rows each.
///
/// # Tier policy: Gpu → HostRam → NvMe
/// - The full table is never assumed to be GPU-resident here; this manager
///   operates at the HostRam/NvMe boundary. GPU-side promotion is up to the
///   calling inference stack.
/// - When a unit is evicted from the host-RAM LRU, its tier flips to
///   `CacheTier::NvMeWeightStream` and is queryable via `get_unit_tier`.
///
/// # Wire to grim-scheduler
/// `EmbeddingSpillManager::get_unit_tier` mirrors `LocalSpillManager::get_tier`
/// so the scheduler can reason about embedding placement alongside KV-block
/// placement using the same `CacheTier` enum.
pub struct EmbeddingSpillManager {
    streamer: NvmeWeightStreamer,
    /// Number of vocabulary rows in one streaming unit.
    pub rows_per_unit: usize,
    /// Embedding hidden dimension (floats per vocabulary row).
    pub hidden_dim: usize,
}

impl EmbeddingSpillManager {
    /// Create a new manager backed by the given NVMe file.
    ///
    /// # Parameters
    /// - `weights_path`: Flat row-major f32 file containing the full embedding
    ///   table in `[vocab_size, hidden_dim]` layout (concatenated row-wise).
    /// - `lru_capacity_units`: Number of row-block units to keep in host RAM.
    /// - `rows_per_unit`: How many vocabulary rows per streaming unit. A
    ///   reasonable starting point is 4096 rows; tune to match available RAM.
    /// - `hidden_dim`: Embedding hidden dimension (floats per row).
    pub fn new(
        weights_path: std::path::PathBuf,
        lru_capacity_units: usize,
        rows_per_unit: usize,
        hidden_dim: usize,
    ) -> Self {
        assert!(rows_per_unit > 0, "EmbeddingSpillManager: rows_per_unit must be > 0");
        assert!(hidden_dim > 0, "EmbeddingSpillManager: hidden_dim must be > 0");
        let unit_elems = rows_per_unit
            .checked_mul(hidden_dim)
            .expect("EmbeddingSpillManager: unit_elems (rows_per_unit * hidden_dim) overflowed usize");
        Self {
            streamer: NvmeWeightStreamer::new(weights_path, lru_capacity_units, unit_elems),
            rows_per_unit,
            hidden_dim,
        }
    }

    /// Look up the embedding row for `token_id`.
    ///
    /// Computes which unit the token's row lives in, prefetches the unit if
    /// not already cached, and extracts the `hidden_dim`-length row.
    ///
    /// Returns an explicit error if the unit cannot be loaded (file missing,
    /// too short, or bandwidth-saturated). Never fabricates a zero row.
    pub fn lookup(&self, token_id: u32) -> Result<Vec<f32>> {
        let token = token_id as usize;
        let unit_id = token / self.rows_per_unit;
        let row_within_unit = token % self.rows_per_unit;

        // Prefetch ensures the unit is in host RAM; this is a no-op if already cached.
        self.streamer.prefetch_layer_async(unit_id)?;

        let unit_data = self.streamer.retrieve_unit(unit_id).ok_or_else(|| {
            Error::KvCache(format!(
                "EmbeddingSpillManager: unit {unit_id} missing from cache after prefetch"
            ))
        })?;

        let start = row_within_unit * self.hidden_dim;
        let end = start + self.hidden_dim;
        if end > unit_data.len() {
            return Err(Error::KvCache(format!(
                "EmbeddingSpillManager: token {token_id} row [{start}..{end}] \
                 out of bounds for unit {unit_id} len {}",
                unit_data.len()
            )));
        }
        Ok(unit_data[start..end].to_vec())
    }

    /// Query the current storage tier for the unit containing `token_id`.
    ///
    /// Returns `None` if the containing unit has never been prefetched,
    /// `Some(CacheTier::HostRam)` if cached, or
    /// `Some(CacheTier::NvMeWeightStream)` if evicted to disk.
    pub fn get_unit_tier_for_token(&self, token_id: u32) -> Option<CacheTier> {
        let unit_id = token_id as usize / self.rows_per_unit;
        self.streamer.get_unit_tier(unit_id)
    }

    /// Query the tier for an explicit unit id (equivalent to streamer's `get_unit_tier`).
    pub fn get_unit_tier(&self, unit_id: usize) -> Option<CacheTier> {
        self.streamer.get_unit_tier(unit_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// Find a free TCP port by binding to port 0 and reading the assigned port.
    fn find_free_port() -> u16 {
        let listener =
            std::net::TcpListener::bind("127.0.0.1:0").expect("must bind to find free port");
        let port = listener.local_addr().expect("must get local addr").port();
        drop(listener);
        port
    }

    #[test]
    fn test_tiered_spillage_and_retrieval() {
        let dir = tempdir().unwrap();
        let manager = SharedSpillManager::new(dir.path().to_path_buf(), 8).unwrap();

        let k = vec![1.0f32; 8];
        let v = vec![2.0f32; 8];

        // 1. Demote to Host RAM
        manager.demote_to_host(42, k.clone(), v.clone()).unwrap();
        assert_eq!(manager.get_tier(42), Some(CacheTier::HostRam));

        // 2. Demote to NVMe
        manager.demote_to_nvme(42).unwrap();
        assert_eq!(manager.get_tier(42), Some(CacheTier::NvMe));

        // 3. Retrieve (promotes back to Host RAM)
        let (ret_k, ret_v) = manager.retrieve(42).unwrap().unwrap();
        assert_eq!(ret_k, k);
        assert_eq!(ret_v, v);
        assert_eq!(manager.get_tier(42), Some(CacheTier::HostRam));

        // 4. Evict
        manager.evict(42);
        assert_eq!(manager.get_tier(42), None);
    }

    #[test]
    fn test_network_kv_client() {
        // Real TCP loopback: start a receiver server, send a block, verify it
        // arrives intact (sims.md issue #2 — transport is now implemented).
        use crate::start_kv_receiver_server;

        /// Minimal KvBlockStore stand-in that stores all received data verbatim.
        struct TestStore {
            blocks: std::collections::HashMap<BlockId, (Vec<f32>, Vec<f32>)>,
            received: std::collections::HashSet<BlockId>,
        }

        impl crate::KvBlockStore for TestStore {
            fn num_blocks(&self) -> usize {
                128
            }
            fn block_elem_per_token(&self) -> usize {
                8
            }
            fn block_size(&self) -> usize {
                16
            }
            fn write_keys(&mut self, id: BlockId, keys: &[f32], _num_tokens: usize) {
                self.blocks.insert(id, (keys.to_vec(), Vec::new()));
                self.received.insert(id);
            }
            fn write_values(&mut self, id: BlockId, values: &[f32]) {
                if let Some((_, v)) = self.blocks.get_mut(&id) {
                    *v = values.to_vec();
                }
            }
            fn block_is_received(&self, id: BlockId) -> bool {
                self.received.contains(&id)
            }
            fn read_keys(&self, id: BlockId) -> Option<Vec<f32>> {
                self.blocks.get(&id).map(|(k, _)| k.clone())
            }
            fn read_values(&self, id: BlockId) -> Option<Vec<f32>> {
                self.blocks.get(&id).map(|(_, v)| v.clone())
            }
        }

        impl TestStore {
            fn new() -> Self {
                Self {
                    blocks: std::collections::HashMap::new(),
                    received: std::collections::HashSet::new(),
                }
            }
        }

        let port = crate::tests::find_free_port();
        let addr = format!("127.0.0.1:{port}");
        let store = std::sync::Arc::new(std::sync::Mutex::new(TestStore::new()));
        let _handle = start_kv_receiver_server(&addr, store.clone()).unwrap();

        let client = NetworkKvClient::new("127.0.0.1".to_string());
        let k = vec![1.0f32; 64];
        let v = vec![2.0f32; 64];
        client
            .send_block_remote(100, 0, &k, &v, &addr)
            .expect("send must succeed against live receiver");

        // Give the receiver thread a moment to write.
        std::thread::sleep(std::time::Duration::from_millis(200));
        let guard = store.lock().unwrap();
        let stored = guard
            .blocks
            .get(&100)
            .expect("block 100 must have been written");
        assert_eq!(stored.0, k, "keys must match exactly");
        assert_eq!(stored.1, v, "values must match exactly");
    }

    #[test]
    fn test_network_kv_client_various_sizes() {
        // Verify real loopback roundtrips for various block sizes.
        use crate::start_kv_receiver_server;

        struct TestStore {
            blocks: std::collections::HashMap<BlockId, (Vec<f32>, Vec<f32>)>,
            received: std::collections::HashSet<BlockId>,
        }

        impl crate::KvBlockStore for TestStore {
            fn num_blocks(&self) -> usize {
                1024
            }
            fn block_elem_per_token(&self) -> usize {
                4
            }
            fn block_size(&self) -> usize {
                64
            }
            fn write_keys(&mut self, id: BlockId, keys: &[f32], _num_tokens: usize) {
                self.blocks.insert(id, (keys.to_vec(), Vec::new()));
                self.received.insert(id);
            }
            fn write_values(&mut self, id: BlockId, values: &[f32]) {
                if let Some((_, v)) = self.blocks.get_mut(&id) {
                    *v = values.to_vec();
                }
            }
            fn block_is_received(&self, id: BlockId) -> bool {
                self.received.contains(&id)
            }
            fn read_keys(&self, id: BlockId) -> Option<Vec<f32>> {
                self.blocks.get(&id).map(|(k, _)| k.clone())
            }
            fn read_values(&self, id: BlockId) -> Option<Vec<f32>> {
                self.blocks.get(&id).map(|(_, v)| v.clone())
            }
        }

        let port = crate::tests::find_free_port();
        let addr = format!("127.0.0.1:{port}");
        let store = std::sync::Arc::new(std::sync::Mutex::new(TestStore {
            blocks: std::collections::HashMap::new(),
            received: std::collections::HashSet::new(),
        }));
        let _handle = start_kv_receiver_server(&addr, store.clone()).unwrap();

        let client = NetworkKvClient::new("127.0.0.1".to_string());

        for &size in &[1usize, 16, 64, 256] {
            let k = vec![0.5f32; size];
            let v = vec![0.25f32; size];
            let block_id = 42 + size;
            client
                .send_block_remote(block_id, 0, &k, &v, &addr)
                .unwrap_or_else(|e| panic!("send_block_remote(size={size}) failed: {e}"));

            std::thread::sleep(std::time::Duration::from_millis(50));
            let guard = store.lock().unwrap();
            let stored = guard.blocks.get(&block_id).unwrap_or_else(|| {
                panic!("block {block_id} (size={size}) must have been written");
            });
            assert_eq!(stored.0, k, "keys must match for size {size}");
            assert_eq!(stored.1, v, "values must match for size {size}");
            drop(guard);
        }
    }

    #[test]
    fn test_kv_block_header_roundtrip() {
        let header = KvBlockHeader {
            magic: KV_MAGIC,
            version: KV_PROTOCOL_VERSION,
            block_id: 0xDEAD_BEEF,
            layer_idx: 3,
            num_elements: 256,
            checksum: 0xCAFEBABE,
        };
        let bytes = header.serialize();
        assert_eq!(bytes.len(), KvBlockHeader::SIZE);
        let decoded = KvBlockHeader::deserialize(&bytes).unwrap();
        assert_eq!(decoded.magic, KV_MAGIC);
        assert_eq!(decoded.version, KV_PROTOCOL_VERSION);
        assert_eq!(decoded.block_id, 0xDEAD_BEEF);
        assert_eq!(decoded.layer_idx, 3);
        assert_eq!(decoded.num_elements, 256);
        assert_eq!(decoded.checksum, 0xCAFEBABE);
        assert!(decoded.verify());
    }

    #[test]
    fn test_checksum_detects_corruption() {
        let k = vec![1.0f32; 8];
        let v = vec![2.0f32; 8];
        let cs_ok = compute_checksum(&k, &v);

        // Flip a bit in the value → checksum changes.
        let mut bad_v = v.clone();
        bad_v[0] = 999.0f32;
        let cs_bad = compute_checksum(&k, &bad_v);
        assert_ne!(cs_ok, cs_bad, "checksum must detect data corruption");
    }

    #[test]
    fn test_fetch_block_remote_returns_error_on_unreachable() {
        // sims.md issue #2: must NOT fabricate data on connection failure.
        let client = NetworkKvClient::new("127.0.0.1".to_string());
        let res = client.fetch_block_remote(100, 0, "127.0.0.1:1", 8);
        assert!(
            res.is_err(),
            "fetch_block_remote must return Err on unreachable endpoint"
        );
        let msg = res.unwrap_err().to_string();
        assert!(
            !msg.contains("fabricated"),
            "error must not reference fabricated data: {msg}"
        );
    }

    /// F8/F10: the receiver server must ANSWER fetch requests from the
    /// store's read side instead of deadlocking on a payload the fetcher
    /// never sends. Push a block, then pull it back and require exact
    /// round-trip equality.
    #[test]
    fn test_fetch_block_remote_roundtrip_against_live_server() {
        struct TestStore {
            blocks: std::collections::HashMap<BlockId, (Vec<f32>, Vec<f32>)>,
            received: std::collections::HashSet<BlockId>,
        }

        impl crate::KvBlockStore for TestStore {
            fn num_blocks(&self) -> usize {
                128
            }
            fn block_elem_per_token(&self) -> usize {
                8
            }
            fn block_size(&self) -> usize {
                16
            }
            fn write_keys(&mut self, id: BlockId, keys: &[f32], _num_tokens: usize) {
                self.blocks.insert(id, (keys.to_vec(), Vec::new()));
                self.received.insert(id);
            }
            fn write_values(&mut self, id: BlockId, values: &[f32]) {
                if let Some((_, v)) = self.blocks.get_mut(&id) {
                    *v = values.to_vec();
                }
            }
            fn block_is_received(&self, id: BlockId) -> bool {
                self.received.contains(&id)
            }
            fn read_keys(&self, id: BlockId) -> Option<Vec<f32>> {
                self.blocks.get(&id).map(|(k, _)| k.clone())
            }
            fn read_values(&self, id: BlockId) -> Option<Vec<f32>> {
                self.blocks.get(&id).map(|(_, v)| v.clone())
            }
        }

        let port = crate::tests::find_free_port();
        let addr = format!("127.0.0.1:{port}");
        let store = std::sync::Arc::new(std::sync::Mutex::new(TestStore {
            blocks: std::collections::HashMap::new(),
            received: std::collections::HashSet::new(),
        }));
        let _handle = start_kv_receiver_server(&addr, store.clone()).unwrap();

        let client = NetworkKvClient::new("127.0.0.1".to_string());
        let k: Vec<f32> = (0..64).map(|i| i as f32 * 0.25).collect();
        let v: Vec<f32> = (0..64).map(|i| (i as f32 * -0.5) - 1.0).collect();
        client
            .send_block_remote(7, 0, &k, &v, &addr)
            .expect("push must succeed");

        // Give the receiver thread a moment to commit the write.
        std::thread::sleep(std::time::Duration::from_millis(200));

        let (got_k, got_v) = client
            .fetch_block_remote(7, 0, &addr, 64)
            .expect("fetch must round-trip against the live server");
        assert_eq!(got_k, k, "fetched keys must match pushed keys");
        assert_eq!(got_v, v, "fetched values must match pushed values");
    }

    /// F8/F10: fetching a block the server does not hold must produce a
    /// prompt "not available" error — never silence, never fabricated data,
    /// never a hang (the pre-fix server wedged here).
    #[test]
    fn test_fetch_block_remote_missing_block_errors_not_hangs() {
        struct EmptyStore;
        impl crate::KvBlockStore for EmptyStore {
            fn num_blocks(&self) -> usize {
                128
            }
            fn block_elem_per_token(&self) -> usize {
                8
            }
            fn block_size(&self) -> usize {
                16
            }
            fn write_keys(&mut self, _id: BlockId, _keys: &[f32], _num_tokens: usize) {}
            fn write_values(&mut self, _id: BlockId, _values: &[f32]) {}
            fn block_is_received(&self, _id: BlockId) -> bool {
                false
            }
            fn read_keys(&self, _id: BlockId) -> Option<Vec<f32>> {
                None
            }
            fn read_values(&self, _id: BlockId) -> Option<Vec<f32>> {
                None
            }
        }

        let port = crate::tests::find_free_port();
        let addr = format!("127.0.0.1:{port}");
        let store = std::sync::Arc::new(std::sync::Mutex::new(EmptyStore));
        let _handle = start_kv_receiver_server(&addr, store).unwrap();

        let client = NetworkKvClient::new("127.0.0.1".to_string());
        let res = client.fetch_block_remote(55, 0, &addr, 64);
        let err = res.expect_err("fetching a block the server lacks must error");
        assert!(
            err.to_string().contains("not available"),
            "error should say the block is not available: {err}"
        );
    }

    #[test]
    fn test_nvme_weight_streamer_reads_real_file() {
        // sims.md issue #3: prefetch_layer_async must read real weights from
        // the file rather than substituting mock 0.5f32 values.
        let dir = tempdir().unwrap();
        let weights_path = dir.path().join("layer_weights.bin");

        // Write 2 layers of 1024 f32 each (4096 bytes per layer).
        let layer0: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let layer1: Vec<f32> = (0..1024).map(|i| (i as f32) * 2.0).collect();
        let mut buf = Vec::new();
        for w in layer0.iter().chain(layer1.iter()) {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        std::fs::write(&weights_path, &buf).unwrap();

        // Pass unit_elems=1024 to preserve existing 1024-float-per-layer behaviour.
        let streamer = NvmeWeightStreamer::new(weights_path.clone(), 4, 1024);

        // Prefetch layer 0 and verify the cached weights match what we wrote.
        streamer
            .prefetch_layer_async(0)
            .expect("layer 0 should prefetch");
        let cache = streamer.host_weight_cache.lock().unwrap();
        let got = cache.get(&0).expect("layer 0 should be cached");
        assert_eq!(
            got, &layer0,
            "layer 0 weights must be the real file contents, not mock 0.5"
        );
        drop(cache);

        // Prefetch layer 1 and verify.
        streamer
            .prefetch_layer_async(1)
            .expect("layer 1 should prefetch");
        let cache = streamer.host_weight_cache.lock().unwrap();
        let got1 = cache.get(&1).expect("layer 1 should be cached");
        assert_eq!(
            got1, &layer1,
            "layer 1 weights must be the real file contents"
        );
    }

    #[test]
    fn test_nvme_weight_streamer_missing_file_errors() {
        // sims.md issue #3: a missing weights file must produce an explicit
        // error, not silently insert mock 0.5f32 data.
        let dir = tempdir().unwrap();
        let weights_path = dir.path().join("does_not_exist.bin");
        // unit_elems=1024: same element count as the former hardcoded constant.
        let streamer = NvmeWeightStreamer::new(weights_path, 4, 1024);

        let res = streamer.prefetch_layer_async(0);
        assert!(
            res.is_err(),
            "missing weights file must error, not silently use mocks"
        );
        let msg = res.unwrap_err().to_string();
        assert!(
            msg.contains("not found"),
            "error should mention missing file: {}",
            msg
        );
    }

    #[test]
    fn test_nvme_weight_streamer_short_file_errors() {
        // sims.md issue #3: a file too short for the requested layer must
        // error rather than silently serving mock data.
        let dir = tempdir().unwrap();
        let weights_path = dir.path().join("short.bin");
        // Only 100 bytes — far too short for layer 0 (4096 bytes).
        std::fs::write(&weights_path, [0u8; 100]).unwrap();
        // unit_elems=1024: same element count as the former hardcoded constant.
        let streamer = NvmeWeightStreamer::new(weights_path, 4, 1024);

        let res = streamer.prefetch_layer_async(0);
        assert!(res.is_err(), "short file must error for layer 0");
        let msg = res.unwrap_err().to_string();
        assert!(
            msg.contains("too short"),
            "error should mention short file: {}",
            msg
        );
    }

    // ── New tests for generalised NvmeWeightStreamer (unit_elems param) ──────

    /// Verify NvmeWeightStreamer works with non-1024 unit_elems (embedding-table granularity).
    ///
    /// Uses rows_per_unit=4, hidden_dim=8 → unit_elems=32 for a fast synthetic test.
    /// Asserts exact round-trip values and that the tier map reflects HostRam after load.
    #[test]
    fn test_nvme_weight_streamer_configurable_unit_elems() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("embed_weights.bin");

        // Two units of 32 f32 each (rows_per_unit=4, hidden_dim=8).
        let unit_elems = 32usize;
        let unit0: Vec<f32> = (0..unit_elems).map(|i| i as f32 * 0.1).collect();
        let unit1: Vec<f32> = (0..unit_elems).map(|i| -(i as f32) * 0.2).collect();
        let mut buf = Vec::with_capacity((unit0.len() + unit1.len()) * 4);
        for f in unit0.iter().chain(unit1.iter()) {
            buf.extend_from_slice(&f.to_le_bytes());
        }
        std::fs::write(&path, &buf).unwrap();

        let streamer = NvmeWeightStreamer::new(path, 4, unit_elems);

        // Load unit 0 and assert exact values.
        streamer.prefetch_layer_async(0).expect("unit 0 must prefetch");
        let got0 = streamer.retrieve_unit(0).expect("unit 0 must be cached");
        assert_eq!(got0, unit0, "unit 0 round-trip must be exact");
        assert_eq!(
            streamer.get_unit_tier(0),
            Some(CacheTier::HostRam),
            "unit 0 tier must be HostRam after prefetch"
        );

        // Load unit 1.
        streamer.prefetch_layer_async(1).expect("unit 1 must prefetch");
        let got1 = streamer.retrieve_unit(1).expect("unit 1 must be cached");
        assert_eq!(got1, unit1, "unit 1 round-trip must be exact");
        assert_eq!(
            streamer.get_unit_tier(1),
            Some(CacheTier::HostRam),
            "unit 1 tier must be HostRam after prefetch"
        );
    }

    /// Verify that LRU eviction records the evicted unit's tier as NvMeWeightStream.
    ///
    /// Capacity = 1 unit → loading unit 1 evicts unit 0.
    /// Unit 0's tier must flip to NvMeWeightStream; unit 1's tier must be HostRam.
    #[test]
    fn test_nvme_weight_streamer_lru_eviction_updates_tier() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("evict_weights.bin");

        let unit_elems = 8usize;
        // Three units of 8 f32.
        let mut buf = Vec::with_capacity(3 * unit_elems * 4);
        for unit in 0u32..3 {
            for elem in 0u32..unit_elems as u32 {
                let val = (unit * 100 + elem) as f32;
                buf.extend_from_slice(&val.to_le_bytes());
            }
        }
        std::fs::write(&path, &buf).unwrap();

        // LRU capacity = 1: only one unit fits in host RAM at a time.
        let streamer = NvmeWeightStreamer::new(path, 1, unit_elems);

        // Load unit 0 → tier HostRam.
        streamer.prefetch_layer_async(0).unwrap();
        assert_eq!(streamer.get_unit_tier(0), Some(CacheTier::HostRam));

        // Load unit 1 → evicts unit 0.
        streamer.prefetch_layer_async(1).unwrap();
        assert_eq!(
            streamer.get_unit_tier(0),
            Some(CacheTier::NvMeWeightStream),
            "evicted unit 0 must have tier NvMeWeightStream"
        );
        assert_eq!(
            streamer.get_unit_tier(1),
            Some(CacheTier::HostRam),
            "current unit 1 must have tier HostRam"
        );

        // Load unit 2 → evicts unit 1.
        streamer.prefetch_layer_async(2).unwrap();
        assert_eq!(
            streamer.get_unit_tier(1),
            Some(CacheTier::NvMeWeightStream),
            "evicted unit 1 must have tier NvMeWeightStream"
        );
        assert_eq!(
            streamer.get_unit_tier(2),
            Some(CacheTier::HostRam),
            "current unit 2 must have tier HostRam"
        );
    }

    // ── EmbeddingSpillManager tests ───────────────────────────────────────────

    /// Unit test: construct an EmbeddingSpillManager with a small synthetic embedding
    /// table, lookup several token IDs, assert exact row values.
    ///
    /// vocab=8 tokens, hidden_dim=4, rows_per_unit=4 → 2 units.
    #[test]
    fn test_embedding_spill_manager_lookup_exact_values() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("emb.bin");

        let vocab = 8usize;
        let hidden = 4usize;
        // Build a [vocab, hidden] table where row i = [i*hidden + 0, i*h+1, ..., i*h+hidden-1].
        let mut table = Vec::with_capacity(vocab * hidden);
        for i in 0..vocab {
            for j in 0..hidden {
                table.push((i * hidden + j) as f32);
            }
        }
        let mut buf = Vec::with_capacity(table.len() * 4);
        for f in &table {
            buf.extend_from_slice(&f.to_le_bytes());
        }
        std::fs::write(&path, &buf).unwrap();

        // rows_per_unit=4: tokens 0-3 → unit 0, tokens 4-7 → unit 1.
        let mgr = EmbeddingSpillManager::new(path, 4, 4, hidden);

        for token in 0u32..vocab as u32 {
            let row = mgr.lookup(token).unwrap_or_else(|e| {
                panic!("lookup({token}) must succeed: {e}")
            });
            assert_eq!(row.len(), hidden, "row length must be hidden_dim={hidden}");
            let expected: Vec<f32> = (0..hidden).map(|j| (token as usize * hidden + j) as f32).collect();
            assert_eq!(
                row, expected,
                "token {token} row must match exact table values"
            );
        }
    }

    /// Integration test: eviction under small LRU capacity, then re-request returns identical data.
    ///
    /// This mirrors LocalSpillManager's existing demote_to_nvme/retrieve test pattern
    /// (plan Issue 2 criterion §2). Uses a 3-unit table with capacity=1 so the first
    /// unit is evicted when the second is loaded. Re-requesting the first unit must
    /// round-trip the same values (via re-prefetch from disk).
    #[test]
    fn test_embedding_spill_manager_eviction_and_reread_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("emb_evict.bin");

        let vocab = 12usize; // 3 units of 4 rows each
        let hidden = 8usize;
        let rows_per_unit = 4usize;

        let mut table = Vec::with_capacity(vocab * hidden);
        for i in 0..vocab {
            for j in 0..hidden {
                // Distinct values per cell to catch any partial-unit read errors.
                table.push((i as f32) * 1000.0 + j as f32);
            }
        }
        let mut buf = Vec::with_capacity(table.len() * 4);
        for f in &table {
            buf.extend_from_slice(&f.to_le_bytes());
        }
        std::fs::write(&path, &buf).unwrap();

        // LRU capacity = 1 → each new unit evicts the previous one.
        let mgr = EmbeddingSpillManager::new(path, 1, rows_per_unit, hidden);

        // First lookup: unit 0 loaded, HostRam.
        let row0_first = mgr.lookup(0).expect("initial lookup of token 0 must succeed");
        assert_eq!(mgr.get_unit_tier(0), Some(CacheTier::HostRam));

        // Second lookup: unit 1 loaded, unit 0 evicted to NvMeWeightStream.
        let _ = mgr.lookup(4).expect("lookup of token 4 (unit 1) must succeed");
        assert_eq!(
            mgr.get_unit_tier(0),
            Some(CacheTier::NvMeWeightStream),
            "unit 0 must be NvMeWeightStream after eviction"
        );

        // Re-request token 0: triggers re-prefetch from disk.
        let row0_second = mgr.lookup(0).expect("re-lookup of token 0 must succeed after eviction");
        assert_eq!(
            row0_first, row0_second,
            "re-read from NvMe must produce bit-identical values to first read"
        );
    }

    /// Verify that get_unit_tier mirrors LocalSpillManager::get_tier for the embedding case
    /// (plan Issue 2 criterion §3: scheduler-queryable placement).
    #[test]
    fn test_embedding_spill_manager_tier_query_mirrors_kv_pattern() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("emb_tier.bin");

        let vocab = 8usize;
        let hidden = 4usize;
        let rows_per_unit = 4usize;

        let table: Vec<f32> = (0..vocab * hidden).map(|i| i as f32).collect();
        let buf: Vec<u8> = table.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(&path, &buf).unwrap();

        let mgr = EmbeddingSpillManager::new(path, 4, rows_per_unit, hidden);

        // Before any lookup: tier is None (not yet prefetched).
        assert_eq!(
            mgr.get_unit_tier(0),
            None,
            "tier must be None before first prefetch"
        );
        assert_eq!(
            mgr.get_unit_tier_for_token(0),
            None,
            "tier-by-token must also be None before first prefetch"
        );

        // After lookup of token 0: unit 0 is HostRam.
        mgr.lookup(0).unwrap();
        assert_eq!(mgr.get_unit_tier(0), Some(CacheTier::HostRam));
        assert_eq!(mgr.get_unit_tier_for_token(0), Some(CacheTier::HostRam));
        assert_eq!(mgr.get_unit_tier_for_token(3), Some(CacheTier::HostRam),
            "all tokens in unit 0 (rows 0-3) must report the same tier");

        // After lookup of token 4 (unit 1): unit 0 is still HostRam (capacity=4, not evicted yet).
        mgr.lookup(4).unwrap();
        assert_eq!(mgr.get_unit_tier(0), Some(CacheTier::HostRam));
        assert_eq!(mgr.get_unit_tier(1), Some(CacheTier::HostRam));
    }
}
