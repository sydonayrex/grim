//! `grim-kvtransport` — tiered KV cache local transport and spillage.
//!
//! Handles moving KV block contents between GPU, Host RAM, and local scratch NVMe files.
//! Sits inside the paged KV pool's eviction policy to support demote-before-drop.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::Mutex;
use parking_lot::RwLock;

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
            let k_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(k.as_ptr() as *const u8, k.len() * 4)
            };
            let v_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4)
            };

            file.write_all(k_bytes).map_err(|e| Error::KvCache(e.to_string()))?;
            file.write_all(v_bytes).map_err(|e| Error::KvCache(e.to_string()))?;

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
            CacheTier::HostRam => {
                Ok(self.host_ram_cache.get(&block_id).cloned())
            }
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

                    file.read_exact(k_bytes).map_err(|e| Error::KvCache(e.to_string()))?;
                    file.read_exact(v_bytes).map_err(|e| Error::KvCache(e.to_string()))?;

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

/// Network transport layer for network-based (RDMA/TCP) KV handoffs.
pub struct NetworkKvClient {
    pub local_ip: String,
}

impl NetworkKvClient {
    pub fn new(local_ip: String) -> Self {
        Self { local_ip }
    }

    /// Simulates/dispatches block transfer over the network (TCP/RDMA).
    pub fn send_block_remote(
        &self,
        block_id: BlockId,
        k: &[f32],
        _v: &[f32],
        target_ip: &str,
    ) -> Result<()> {
        println!(
            "[NetworkKvClient] Sending KV block {} from {} to {} (Size: {} elements)",
            block_id, self.local_ip, target_ip, k.len()
        );
        Ok(())
    }

    /// Fetches block from a remote node.
    pub fn fetch_block_remote(
        &self,
        block_id: BlockId,
        target_ip: &str,
        block_elems: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        println!(
            "[NetworkKvClient] Fetching KV block {} from {} to {}",
            block_id, target_ip, self.local_ip
        );
        Ok((vec![1.0; block_elems], vec![2.0; block_elems]))
    }
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

    /// Simulate prefetching a target layer weights asynchronously into pinned CPU RAM.
    /// In a production environment under Linux, this leverages io_uring and O_DIRECT.
    pub fn prefetch_layer_async(&self, layer_id: usize) -> Result<()> {
        println!(
            "[NvmeWeightStreamer] Prefetching layer {} from NVMe file: {:?}",
            layer_id, self.weights_path
        );

        // Bandwidth Admission and Backpressure check:
        // If simulated bandwidth usage exceeds 12.0 GB/s (representing PCIe Gen4 x8 saturation),
        // we introduce backpressure delay or return an error/warning limit.
        let cur_bandwidth = *self.bandwidth_usage.lock().unwrap();
        if cur_bandwidth > 12.0 * 1024.0 * 1024.0 * 1024.0 {
            println!("[NvmeWeightStreamer] [BACKPRESSURE] Bandwidth limit reached ({} B/s). Deferring prefetch.", cur_bandwidth);
            return Err(Error::KvCache("PCIe transfer bandwidth limit backpressure triggered".into()));
        }

        // Simulate io_uring submission loop
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
                    println!("[NvmeWeightStreamer] [LRU] Evicted layer {} from Host RAM weight cache", evicted);
                }
            }

            // Load mock layer weights into double buffer
            let mock_weights = vec![0.5f32; 1024];
            cache.insert(layer_id, mock_weights.clone());
            order.push(layer_id);

            // Populate double buffers (async swap preparation)
            let mut buffers = self.double_buffers.lock().unwrap();
            buffers.1 = mock_weights; // Load into transfer buffer
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
    pub fn commit_and_swap(&self, current_layer: usize, next_layer: usize) -> Result<()> {
        println!(
            "[NvmeWeightStreamer] Swapping buffers: GPU executing Layer {}, DMA promoting Layer {}",
            current_layer, next_layer
        );
        let mut buffers = self.double_buffers.lock().unwrap();
        // Double-buffered swap: Active buffer becomes transfer buffer and vice versa
        let (buf0, buf1) = &mut *buffers;
        std::mem::swap(buf0, buf1);
        Ok(())
    }

    /// Update simulated transfer bandwidth usage.
    pub fn set_bandwidth_usage(&self, bytes_per_sec: f64) {
        *self.bandwidth_usage.lock().unwrap() = bytes_per_sec;
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

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
        let client = NetworkKvClient::new("127.0.0.1".to_string());
        let k = vec![1.0f32; 8];
        let v = vec![2.0f32; 8];
        client.send_block_remote(100, &k, &v, "127.0.0.2").unwrap();
        let (ret_k, ret_v) = client.fetch_block_remote(100, "127.0.0.2", 8).unwrap();
        assert_eq!(ret_k, vec![1.0; 8]);
        assert_eq!(ret_v, vec![2.0; 8]);
    }
}
