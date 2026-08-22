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
            let file_path = self.scratch_dir.join(format!("kv_block_{}.bin", block_id));
            let mut file = File::create(&file_path).map_err(|e| Error::KvCache(e.to_string()))?;

            // Write keys and values as raw bytes
            let k_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(k.as_ptr() as *const u8, k.len() * 4) };
            let v_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };

            file.write_all(k_bytes)
                .map_err(|e| Error::KvCache(e.to_string()))?;
            file.write_all(v_bytes)
                .map_err(|e| Error::KvCache(e.to_string()))?;

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
                if let Some(path) = self.nvme_cache.get(&block_id) {
                    let mut file = File::open(path).map_err(|e| Error::KvCache(e.to_string()))?;
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

                    // Bring back to Host RAM (cache promotion)
                    self.host_ram_cache.insert(block_id, (k.clone(), v.clone()));
                    self.block_tiers.insert(block_id, CacheTier::HostRam);

                    Ok(Some((k, v)))
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
        Some(Self {
            magic: u32::from_le_bytes(buf[0..4].try_into().unwrap()),
            version: u32::from_le_bytes(buf[4..8].try_into().unwrap()),
            block_id: u64::from_le_bytes(buf[8..16].try_into().unwrap()),
            layer_idx: u32::from_le_bytes(buf[16..20].try_into().unwrap()),
            num_elements: u32::from_le_bytes(buf[20..24].try_into().unwrap()),
            checksum: u32::from_le_bytes(buf[24..28].try_into().unwrap()),
        })
    }

    /// Verify the magic number and protocol version.
    pub fn verify(&self) -> bool {
        self.magic == KV_MAGIC && self.version == KV_PROTOCOL_VERSION
    }
}

/// FNV-1a 32-bit checksum over the raw bytes of the key and value float slices.
/// A simple non-cryptographic checksum sufficient to detect truncation or
/// bit-corruption on the wire.
fn compute_checksum(k: &[f32], v: &[f32]) -> u32 {
    let mut hash: u32 = 0x811c9275; // FNV offset basis
    for f in k.iter().chain(v.iter()) {
        for &b in f.to_le_bytes().iter() {
            hash ^= b as u32;
            hash = hash.wrapping_mul(0x01000193); // FNV prime
        }
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
        // and a special request flag in layer_idx (bit 31 set).
        let req_header = KvBlockHeader {
            magic: KV_MAGIC,
            version: KV_PROTOCOL_VERSION,
            block_id: block_id as u64,
            layer_idx: layer_idx | FETCH_REQUEST_FLAG,
            num_elements: block_elems as u32,
            checksum: 0,
        };

        let mut stream = std::net::TcpStream::connect_timeout(
            &socket_addr,
            std::time::Duration::from_millis(500),
        )
        .map_err(|e| Error::KvCache(format!("TCP fetch connection failed to {addr}: {e}")))?;

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
}

/// Bit flag embedded in `layer_idx` of a fetch-request header to signal
/// "this is a fetch request, reply with data" rather than a push transfer.
const FETCH_REQUEST_FLAG: u32 = 0x8000_0000;

/// Reinterpret a byte slice as f32 values (little-endian).  Safe parse — no
/// unsafe pointer casts, avoids alignment UB on Vec<u8> buffers.
fn parse_f32_slice(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

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
                    let mut guard = pool.lock().unwrap();
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
                        let num_tokens = if elem_per_token > 0 {
                            (num_elems / elem_per_token).min(guard.block_size())
                        } else {
                            num_elems
                        };
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

    let offset = layer_id as u64 * layer_bytes as u64;
    let metadata = file
        .metadata()
        .map_err(|e| Error::KvCache(format!("failed to stat NVMe weights: {}", e)))?;
    if metadata.len() < offset + layer_bytes as u64 {
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

/// Double-buffered weight prefetch engine for NVMe layer streaming.
pub struct NvmeWeightStreamer {
    /// LRU layer cache capacity
    pub lru_capacity_layers: usize,
    /// NVMe file path for model weights
    pub weights_path: PathBuf,
    /// Host RAM LRU weight cache
    host_weight_cache: Mutex<HashMap<usize, Vec<f32>>>,
    /// Track layer access order for LRU eviction
    lru_order: Mutex<Vec<usize>>,
    /// Double buffers for async weight prefetching
    double_buffers: Mutex<(Vec<f32>, Vec<f32>)>,
    /// Simulated io_uring submission/completion queue status
    uring_submitting: Mutex<bool>,
    /// Current transfer bandwidth usage (bytes/sec)
    bandwidth_usage: Mutex<f64>,
}

impl NvmeWeightStreamer {
    pub fn new(weights_path: PathBuf, lru_capacity_layers: usize) -> Self {
        Self {
            weights_path,
            lru_capacity_layers,
            host_weight_cache: Mutex::new(HashMap::new()),
            lru_order: Mutex::new(Vec::new()),
            double_buffers: Mutex::new((vec![], vec![])),
            uring_submitting: Mutex::new(false),
            bandwidth_usage: Mutex::new(0.0),
        }
    }

    /// Prefetch a target layer's weights asynchronously into pinned CPU RAM.
    ///
    /// In a production environment under Linux, this leverages `io_uring` and
    /// `O_DIRECT`; here we synchronously `pread` the weights file, since the
    /// network/io_uring backend is not yet wired (sims.md issue #3). The
    /// previous implementation inserted hardcoded `vec![0.5f32; 1024]` weights
    /// instead of reading from disk, silently corrupting any computation that
    /// consumed the cache. We now read the real bytes from `weights_path`:
    ///
    /// - If the file is missing or the read fails, we surface an explicit
    ///   `KvCache` error rather than substituting mock data.
    /// - The on-disk layout is treated as a flat stream of `f32` weights
    ///   sectioned per layer by `LAYER_ELEMS` (1024 floats / 4096 bytes per
    ///   layer), matching the previous mock weight length so callers that
    ///   depend on a 1024-element layer keep working when real data is
    ///   available.
    ///
    /// The LRU admission + bandwidth backpressure logic is preserved — only
    /// the data source changes from mock to real.
    pub fn prefetch_layer_async(&self, layer_id: usize) -> Result<()> {
        // Bandwidth Admission and Backpressure check (real logic preserved):
        // If bandwidth usage exceeds 12.0 GB/s (~PCIe Gen4 x8 saturation),
        // defer the prefetch instead of saturating the link.
        let cur_bandwidth = *self.bandwidth_usage.lock().unwrap();
        if cur_bandwidth > 12.0 * 1024.0 * 1024.0 * 1024.0 {
            return Err(Error::KvCache(
                "PCIe transfer bandwidth limit backpressure triggered".into(),
            ));
        }

        // Per-layer size in f32 elements (matches the previous mock length so
        // existing callers preserve their shape when backed by a real file).
        const LAYER_ELEMS: usize = 1024;
        let layer_bytes = LAYER_ELEMS * std::mem::size_of::<f32>();

        // Read the layer's weights from the configured NVMe path. We do this
        // up-front (before acquiring the cache locks) so an I/O error fails
        // loudly instead of leaving the cache half-mutated.
        let weights = read_layer_weights(&self.weights_path, layer_id, LAYER_ELEMS, layer_bytes)?;

        *self.uring_submitting.lock().unwrap() = true;

        // Populate LRU cache
        let mut cache = self.host_weight_cache.lock().unwrap();
        let mut order = self.lru_order.lock().unwrap();

        if !cache.contains_key(&layer_id) {
            // Evict LRU if capacity exceeded
            if cache.len() >= self.lru_capacity_layers {
                if !order.is_empty() {
                    let evicted = order.remove(0);
                    cache.remove(&evicted);
                }
            }

            cache.insert(layer_id, weights.clone());
            order.push(layer_id);

            // Populate double buffers (async swap preparation)
            let mut buffers = self.double_buffers.lock().unwrap();
            buffers.1 = weights; // Load into transfer buffer
        } else {
            // Move layer to end of access order
            if let Some(pos) = order.iter().position(|&x| x == layer_id) {
                order.remove(pos);
            }
            order.push(layer_id);
        }

        *self.uring_submitting.lock().unwrap() = false;
        Ok(())
    }

    /// Swaps the target double-buffers to update GPU memory.
    pub fn commit_and_swap(&self, _current_layer: usize, _next_layer: usize) -> Result<()> {
        let mut buffers = self.double_buffers.lock().unwrap();
        // Double-buffered swap: Active buffer becomes transfer buffer and vice versa.
        // The previous implementation logged the layer ids; the swap itself is
        // the only observable effect, so the println was just noise in logs.
        let (buf0, buf1) = &mut *buffers;
        std::mem::swap(buf0, buf1);
        Ok(())
    }

    /// Update the tracked transfer bandwidth usage (bytes/sec).
    pub fn set_bandwidth_usage(&self, bytes_per_sec: f64) {
        *self.bandwidth_usage.lock().unwrap() = bytes_per_sec;
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
                .send_block_remote(block_id as usize, 0, &k, &v, &addr)
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

        let streamer = NvmeWeightStreamer::new(weights_path.clone(), 4);

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
        let streamer = NvmeWeightStreamer::new(weights_path, 4);

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
        std::fs::write(&weights_path, &[0u8; 100]).unwrap();
        let streamer = NvmeWeightStreamer::new(weights_path, 4);

        let res = streamer.prefetch_layer_async(0);
        assert!(res.is_err(), "short file must error for layer 0");
        let msg = res.unwrap_err().to_string();
        assert!(
            msg.contains("too short"),
            "error should mention short file: {}",
            msg
        );
    }
}
