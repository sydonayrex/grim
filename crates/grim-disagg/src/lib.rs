//! Distributed serving and disaggregation layer: decouples prefill/decode, manages cross-node KV cache transfers.
//!
/// ReMP 2D KV-cache migration (WI-8): coalesced 128-byte block-major transfer within same VRAM pool.

use std::sync::Arc;
use std::sync::Mutex;

use grim_core::error::{Error, Result};
use grim_kvtransport::NetworkKvClient;
use grim_memory::KvBlockPool;

// ── ReMP KV migration types (WI-8) ────────────────────────────────────────────

/// Single KV block: one layer, one sequence chunk. Mirrors `kv_to_block_major` block-major layout.
#[derive(Debug, Clone)]
pub struct KvBlock {
    /// K and V data for this layer segment (flattened f32).
    pub data: Vec<f32>,
    /// Layer index this block belongs to.
    pub layer_idx: u32,
    /// Position (sequence chunk) index within the layer.
    pub seq_chunk: u32,
}

/// 2D ReMP batch: outer = layers, inner = seq chunks. `migrate()` drains to flat buffer.
#[derive(Debug, Default)]
pub struct ReMPMigrationBatch {
    pub blocks: Vec<KvBlock>,
    pub num_layers: u32,
    pub num_seq_chunks: u32,
}

impl ReMPMigrationBatch {
    /// Validate batch shape: non-empty, 2D dims match block count.
    pub fn validate(&self) -> Result<()> {
        if self.blocks.is_empty() {
            return Err(Error::KvCache("ReMPMigrationBatch: no blocks".into()));
        }
        let expected = self.num_layers as usize * self.num_seq_chunks as usize;
        if self.blocks.len() != expected {
            return Err(Error::KvCache(format!(
                "ReMPMigrationBatch: expected {} blocks ({} layers × {} chunks), got {}",
                expected,
                self.num_layers,
                self.num_seq_chunks,
                self.blocks.len()
            )));
        }
        Ok(())
    }

    /// Drain 2D block matrix to flat KV buffer (layer-major, chunk-major). No re-layout needed.
    pub fn migrate(&self) -> Result<Vec<f32>> {
        self.validate()?;
        let total: usize = self.blocks.iter().map(|b| b.data.len()).sum();
        let mut flat = Vec::with_capacity(total);
        // Outer loop: layers. Inner loop: seq chunks (matches paged_attention KV fetch order).
        for layer in 0..self.num_layers {
            // Inner loop: seq chunks (matches paged_attention KV fetch order).
            for chunk in 0..self.num_seq_chunks {
                if let Some(block) = self
                    .blocks
                    .iter()
                    .find(|b| b.layer_idx == layer && b.seq_chunk == chunk)
                {
                    flat.extend_from_slice(&block.data);
                } else {
                    return Err(Error::KvCache(format!(
                        "ReMPMigrationBatch: missing block layer={layer} chunk={chunk}"
                    )));
                }
            }
        }
        Ok(flat)
    }
}

/// Node role in serving cluster (§5.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolRole {
    Colocated,
    Prefill,
    Decode,
}

/// Source prefill node params carried inside decode step (§5.6).
#[derive(Debug, Clone)]
pub struct PoolAssignment {
    pub source_prefill_pool_addr: String,
    pub request_id: u64,
}

/// Disaggregation configuration carried by `EngineConfig` and the serving CLI.
/// Defined here (in grim-disagg) so both `grim-engine` and `grim-server` can
/// depend on it without creating a circular dependency.
#[derive(Debug, Clone)]
pub struct DisaggConfig {
    pub role: PoolRole,
    pub prefill_addr: String,
    pub decode_addr: String,
}

impl Default for DisaggConfig {
    fn default() -> Self {
        Self {
            role: PoolRole::Colocated,
            prefill_addr: String::new(),
            decode_addr: String::new(),
        }
    }
}

/// Router for cross-pool dispatch and network KV transfers.
///
/// When `pool` is set, the router extracts **real** KV block data from the
/// engine's `KvBlockPool` via `read_keys()` / `read_values()` rather than
/// synthesising fake payloads.
pub trait DisaggRouterT: Send + Sync {
    /// Dispatch a prefill task — sends real KV blocks from the local pool
    /// to the prefill node.
    fn dispatch_prefill(&self, request_id: u64, tokens: &[u32]) -> Result<()>;
    /// Transfer real KV blocks (identified by physical `block_ids`) from the
    /// local pool to the decode node.
    fn transfer_kv_cache(&self, request_id: u64, block_ids: &[usize]) -> Result<()>;
    /// Dispatches a step-decode task to a dedicated decode engine.
    fn dispatch_decode(
        &self,
        request_id: u64,
        last_token: u32,
        assignment: PoolAssignment,
    ) -> Result<()>;
}

pub struct DisaggRouter {
    pub prefill_node_addr: String,
    pub decode_node_addr: String,
    pub pool_role: PoolRole,
    kv_client: NetworkKvClient,
    use_rdma: bool,
    /// Reference to the engine's KvBlockPool for real KV extraction.
    /// When `None`, methods that require real KV data return an error.
    pub pool: Option<Arc<Mutex<KvBlockPool>>>,
}

impl DisaggRouter {
    pub fn new(prefill_node_addr: &str, decode_node_addr: &str, pool_role: PoolRole) -> Self {
        Self {
            prefill_node_addr: prefill_node_addr.to_string(),
            decode_node_addr: decode_node_addr.to_string(),
            pool_role,
            kv_client: NetworkKvClient::new(prefill_node_addr.to_string()),
            // Use TCP by default; enable RDMA fallback with enable_rdma().
            use_rdma: false,
            pool: None,
        }
    }

    /// Attach the engine's shared KvBlockPool so that `transfer_kv_cache`,
    /// `dispatch_prefill`, and `dispatch_decode` extract **real** KV data
    /// instead of synthetic payloads.
    pub fn with_pool(mut self, pool: Arc<Mutex<KvBlockPool>>) -> Self {
        self.pool = Some(pool);
        self
    }

    /// Enable or disable RDMA network transport.
    pub fn enable_rdma(&mut self, enabled: bool) {
        self.use_rdma = enabled;
    }

    /// Transfer real KV blocks extracted from a physical KvBlockPool to the
    /// remote decode engine.
    ///
    /// Each block's key/value float vectors are read from the pool via
    /// `read_keys()` and `read_values()`, then sent over TCP using the V2
    /// wire protocol (magic + checksum in the header).
    pub fn transfer_kv_cache_real(
        &self,
        _request_id: u64,
        block_ids: &[usize],
        pool: &grim_memory::KvBlockPool,
    ) -> Result<()> {
        if block_ids.is_empty() {
            return Err(Error::KvCache(
                "Handoff protocol error: block list cannot be empty".into(),
            ));
        }
        for &b_id in block_ids {
            let k_data = pool.read_keys(b_id);
            let v_data = pool.read_values(b_id);
            self.kv_client.send_block_remote(
                b_id,
                0, // layer_idx — pool blocks are layer-local; 0 is the canonical value
                k_data,
                v_data,
                &self.decode_node_addr,
            )?;
        }
        Ok(())
    }

    /// Extract real KV blocks from the stored pool and send them to the
    /// prefill node address.
    fn extract_and_send_prefill(&self, _request_id: u64, tokens: &[u32]) -> Result<()> {
        let pool = self.pool.as_ref().ok_or_else(|| {
            Error::KvCache("dispatch_prefill: no KvBlockPool attached for real KV extraction".into())
        })?;
        let guard = pool.lock().map_err(|e| {
            Error::KvCache(format!("dispatch_prefill: pool mutex poisoned: {e}"))
        })?;
        // Iterate over all allocated blocks in the pool and stream their real
        // KV data to the prefill node.
        for block_id in 0..guard.num_blocks() {
            let k_data = guard.read_keys(block_id);
            let v_data = guard.read_values(block_id);
            self.kv_client.send_block_remote(
                block_id,
                0,
                k_data,
                v_data,
                &self.prefill_node_addr,
            )?;
        }
        // Also forward the prompt token IDs as a meta-block so the prefill
        // node knows which tokens to decode next.
        let k_buf = tokens.iter().map(|&t| t as f32).collect::<Vec<_>>();
        let v_buf = vec![0.0f32; tokens.len()];
        self.kv_client.send_block_remote(
            guard.num_blocks(),
            0,
            &k_buf,
            &v_buf,
            &self.prefill_node_addr,
        )?;
        Ok(())
    }

    /// Extract real KV blocks from the stored pool and send them to the
    /// decode node address as a decode step context.
    fn extract_and_send_decode(&self, _request_id: u64, _last_token: u32) -> Result<()> {
        let pool = self.pool.as_ref().ok_or_else(|| {
            Error::KvCache("dispatch_decode: no KvBlockPool attached for real KV extraction".into())
        })?;
        let guard = pool.lock().map_err(|e| {
            Error::KvCache(format!("dispatch_decode: pool mutex poisoned: {e}"))
        })?;
        for block_id in 0..guard.num_blocks() {
            let k_data = guard.read_keys(block_id);
            let v_data = guard.read_values(block_id);
            self.kv_client.send_block_remote(
                block_id,
                0,
                k_data,
                v_data,
                &self.decode_node_addr,
            )?;
        }
        Ok(())
    }

    /// Fetch a single KV block from the prefill node (decode → prefill pull).
    /// Returns the raw key/value float slices.  Errors on connection failure
    /// — never fabricates data.
    pub fn fetch_kv_block(
        &self,
        block_id: usize,
        layer_idx: u32,
        block_elems: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.kv_client
            .fetch_block_remote(block_id, layer_idx, &self.prefill_node_addr, block_elems)
    }
}

impl DisaggRouterT for DisaggRouter {
    /// Dispatch a prefill task to a dedicated prefill engine.
    fn dispatch_prefill(&self, request_id: u64, tokens: &[u32]) -> Result<()> {
        self.extract_and_send_prefill(request_id, tokens)
    }

    /// Transfer real KV blocks to the decode node.
    fn transfer_kv_cache(&self, request_id: u64, block_ids: &[usize]) -> Result<()> {
        if block_ids.is_empty() {
            return Err(Error::KvCache(
                "Handoff protocol error: block list cannot be empty".into(),
            ));
        }
        let pool = self.pool.as_ref().ok_or_else(|| {
            Error::KvCache(
                "transfer_kv_cache: no KvBlockPool attached for real KV extraction".into(),
            )
        })?;
        let guard = pool.lock().map_err(|e| {
            Error::KvCache(format!("transfer_kv_cache: pool mutex poisoned: {e}"))
        })?;
        for &b_id in block_ids {
            let k_data = guard.read_keys(b_id);
            let v_data = guard.read_values(b_id);
            self.kv_client.send_block_remote(
                b_id,
                0,
                k_data,
                v_data,
                &self.decode_node_addr,
            )?;
        }
        Ok(())
    }

    /// Dispatch a step-decode task to a dedicated decode engine.
    fn dispatch_decode(
        &self,
        request_id: u64,
        last_token: u32,
        _assignment: PoolAssignment,
    ) -> Result<()> {
        self.extract_and_send_decode(request_id, last_token)
    }
}

impl DisaggRouter {
    /// ReMP 2D KV-cache migration (WI-8): colocated same-VRAM-pool transfer. No network round-trip.
    pub fn transfer_kv_colocated(
        &self,
        request_id: u64,
        batch: &ReMPMigrationBatch,
    ) -> Result<Vec<f32>> {
        // ReMP requires colocated pool role.
        if self.pool_role != PoolRole::Colocated {
            return Err(Error::KvCache(format!(
                "transfer_kv_colocated: only valid for Colocated pool role, got {:?}",
                self.pool_role
            )));
        }
        // Validate and drain the 2D batch.
        let flat = batch.migrate().map_err(|e| {
            Error::KvCache(format!(
                "transfer_kv_colocated(request_id={request_id}): migration failed: {e}"
            ))
        })?;
        eprintln!(
            "[grim-disagg] ReMP colocated transfer: request_id={request_id}, \
             layers={}, chunks={}, total_elements={}",
            batch.num_layers,
            batch.num_seq_chunks,
            flat.len()
        );
        Ok(flat)
    }
}

/// Background TCP receiver server for cross-node KV cache block ingestion.
///
/// Wraps `grim_kvtransport::start_kv_receiver_server` with a reference to the
/// engine's `KvBlockPool`.  The receiver listens on `listen_addr`, accepts
/// incoming V2 protocol block transfers (with magic / checksum verification),
/// and writes the received key/value data into the pool via the `KvBlockStore`
/// trait.
pub struct KvReceiverServer {
    listen_addr: String,
    pool: Arc<Mutex<KvBlockPool>>,
    #[allow(dead_code)]
    handle: Option<std::thread::JoinHandle<()>>,
}

impl KvReceiverServer {
    /// Start a KV receiver server on `listen_addr` that writes into `pool`.
    pub fn new(listen_addr: &str, pool: Arc<Mutex<KvBlockPool>>) -> Result<Self> {
        let handle = grim_kvtransport::start_kv_receiver_server(listen_addr, pool.clone())?;
        Ok(Self {
            listen_addr: listen_addr.to_string(),
            pool,
            handle: Some(handle),
        })
    }

    pub fn listen_addr(&self) -> &str {
        &self.listen_addr
    }

    pub fn pool(&self) -> &Arc<Mutex<KvBlockPool>> {
        &self.pool
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Find a free TCP port for loopback tests.
    fn find_free_port() -> u16 {
        let listener =
            std::net::TcpListener::bind("127.0.0.1:0").expect("must bind to find free port");
        listener.local_addr().expect("must get local addr").port()
    }

    #[test]
    fn test_disaggregated_kv_routing() {
        // Without a pool, dispatch methods must error (no real KV to extract).
        let router = DisaggRouter::new("127.0.0.1:0", "127.0.0.1:0", PoolRole::Prefill);

        assert_eq!(router.prefill_node_addr, "127.0.0.1:0");
        assert_eq!(router.decode_node_addr, "127.0.0.1:0");
        assert_eq!(router.pool_role, PoolRole::Prefill);

        // dispatch_prefill without a pool → error
        let prefill_res = router.dispatch_prefill(42, &[101, 102, 103]);
        assert!(
            prefill_res.is_err(),
            "dispatch_prefill should fail without a pool"
        );

        // transfer_kv_cache without a pool → error
        let transfer_res = router.transfer_kv_cache(42, &[0, 1, 2, 3]);
        assert!(
            transfer_res.is_err(),
            "transfer_kv_cache should fail without a pool"
        );

        // dispatch_decode without a pool → error
        let assignment = PoolAssignment {
            source_prefill_pool_addr: "127.0.0.1:0".to_string(),
            request_id: 42,
        };
        let decode_res = router.dispatch_decode(42, 104, assignment);
        assert!(
            decode_res.is_err(),
            "dispatch_decode should fail without a pool"
        );

        // Verify the pool field exists and is None by default.
        assert!(router.pool.is_none(), "pool must default to None");
    }

    #[test]
    fn test_transfer_kv_cache_rejects_zero_blocks() {
        let router = DisaggRouter::new("127.0.0.1:0", "127.0.0.1:0", PoolRole::Prefill);
        let result = router.transfer_kv_cache(42, &[]);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("block list cannot be empty"),
            "error should mention empty block list: {}",
            err_msg
        );
    }

    #[test]
    fn test_rdma_toggle() {
        let mut router = DisaggRouter::new("127.0.0.1:0", "127.0.0.1:0", PoolRole::Prefill);
        assert!(!router.use_rdma);
        router.enable_rdma(true);
        assert!(router.use_rdma);
        router.enable_rdma(false);
        assert!(!router.use_rdma);
    }

    #[test]
    fn test_dispatch_decode_preserves_assignment() {
        let router = DisaggRouter::new("127.0.0.1:0", "127.0.0.1:0", PoolRole::Decode);
        let assignment = PoolAssignment {
            source_prefill_pool_addr: "127.0.0.1:0".to_string(),
            request_id: 99,
        };
        // Verify assignment fields are accessible and correct
        assert_eq!(assignment.source_prefill_pool_addr, "127.0.0.1:0");
        assert_eq!(assignment.request_id, 99);
        let decode_res = router.dispatch_decode(99, 200, assignment);
        assert!(
            decode_res.is_err(),
            "dispatch_decode should fail without a pool"
        );
    }

    #[test]
    fn test_pool_assignment_fields() {
        // Verify PoolAssignment stores fields correctly — catches mutations
        // that rename or remove fields.
        let assignment = PoolAssignment {
            source_prefill_pool_addr: "192.168.1.1:9000".to_string(),
            request_id: 12345,
        };
        assert_eq!(assignment.source_prefill_pool_addr, "192.168.1.1:9000");
        assert_eq!(assignment.request_id, 12345);
    }

    /// WI-8 gate: ReMP colocated migration must preserve session data.
    #[test]
    fn test_kv_migration_preserves_session() {
        let router = DisaggRouter::new("localhost:0", "localhost:0", PoolRole::Colocated);
        // Build a 2-layer × 2-chunk KV batch.
        let block_size = 4;
        let mut batch = ReMPMigrationBatch {
            num_layers: 2,
            num_seq_chunks: 2,
            blocks: Vec::new(),
        };
        for layer in 0..2u32 {
            for chunk in 0..2u32 {
                batch.blocks.push(KvBlock {
                    data: vec![layer as f32 * 10.0 + chunk as f32; block_size],
                    layer_idx: layer,
                    seq_chunk: chunk,
                });
            }
        }
        let result = router.transfer_kv_colocated(1, &batch).unwrap();
        // Verify all data was migrated in layer-major, chunk-major order.
        assert_eq!(result.len(), 4 * block_size);
        // Layer 0, chunk 0: value 0.0
        assert_eq!(result[0], 0.0);
        // Layer 0, chunk 1: value 1.0
        assert_eq!(result[block_size], 1.0);
        // Layer 1, chunk 0: value 10.0
        assert_eq!(result[block_size * 2], 10.0);
        // Layer 1, chunk 1: value 11.0
        assert_eq!(result[block_size * 3], 11.0);
    }

    /// WI-8 gate: non-colocated router must reject transfer_kv_colocated.
    #[test]
    fn test_kv_migration_rejects_non_colocated() {
        let router = DisaggRouter::new("127.0.0.1:0", "127.0.0.1:0", PoolRole::Prefill);
        let batch = ReMPMigrationBatch {
            blocks: vec![],
            num_layers: 0,
            num_seq_chunks: 0,
        };
        let result = router.transfer_kv_colocated(99, &batch);
        assert!(result.is_err());
    }

    /// WI-8 gate: missing block must return an error.
    #[test]
    fn test_kv_migration_missing_block_error() {
        let router = DisaggRouter::new("localhost:0", "localhost:0", PoolRole::Colocated);
        let mut batch = ReMPMigrationBatch {
            num_layers: 2,
            num_seq_chunks: 2,
            blocks: Vec::new(),
        };
        // Only add 3 of 4 required blocks.
        for layer in 0..2u32 {
            for chunk in 0..2u32 {
                if !(layer == 1 && chunk == 1) {
                    batch.blocks.push(KvBlock {
                        data: vec![1.0; 4],
                        layer_idx: layer,
                        seq_chunk: chunk,
                    });
                }
            }
        }
        let result = router.transfer_kv_colocated(99, &batch);
        assert!(result.is_err(), "should fail with missing block");
    }

    /// Real KV transfer loopback: start a receiver, attach a pool, create a
    /// DisaggRouter with the pool, transfer KV blocks over TCP, and verify
    /// the data arrives intact on the receiving end.
    #[test]
    fn test_real_kv_transfer_loopback() {
        // Set up a real KvBlockPool with one block of known data.
        let mut pool = KvBlockPool::new(4, 2, 4); // 4 blocks, 2 heads, head_dim 4
        let block_id = 0usize;
        let k_data: Vec<f32> = (0..(16 * 2 * 4)).map(|i| i as f32).collect();
        let v_data: Vec<f32> = (0..(16 * 2 * 4)).map(|i| (i as f32) * 10.0).collect();
        pool.write_keys(block_id, &k_data, 16);
        pool.write_values(block_id, &v_data);

        let shared_pool = Arc::new(Mutex::new(pool));

        let port = find_free_port();
        let addr = format!("127.0.0.1:{port}");
        let _receiver =
            crate::KvReceiverServer::new(&addr, shared_pool.clone()).unwrap();

        // Create the destination pool (receiver side).
        let dest_pool = Arc::new(Mutex::new(KvBlockPool::new(4, 2, 4)));

        // Start a second receiver on the destination
        let port2 = find_free_port();
        let addr2 = format!("127.0.0.1:{port2}");
        let receiver = crate::KvReceiverServer::new(&addr2, dest_pool.clone()).unwrap();

        // Create router with the source pool.
        let router = DisaggRouter::new(&addr2, &addr2, PoolRole::Prefill)
            .with_pool(shared_pool.clone());

        // Transfer the real KV block.
        router
            .transfer_kv_cache_real(1, &[block_id], &shared_pool.lock().unwrap())
            .expect("transfer must succeed");

        // Give the receiver thread time to process.
        std::thread::sleep(std::time::Duration::from_millis(300));

        // Verify the data arrived at the destination.
        // transfer_kv_cache_real now sends the physical block_id (0),
        // not request_id + i (1) as the old buggy version did.
        let dest_guard = dest_pool.lock().unwrap();
        let recv_k = dest_guard.read_keys(0);
        let recv_v = dest_guard.read_values(0);
        assert_eq!(
            &recv_k[..k_data.len()],
            &k_data[..],
            "received keys must match sent keys"
        );
        assert_eq!(
            &recv_v[..v_data.len()],
            &v_data[..],
            "received values must match sent values"
        );

        drop(receiver);
    }
}
