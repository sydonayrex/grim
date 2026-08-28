//! Distributed serving and disaggregation layer: decouples prefill/decode, manages cross-node KV cache transfers.
//!
/// ReMP 2D KV-cache migration (WI-8): coalesced 128-byte block-major transfer within same VRAM pool.
use std::sync::Arc;
use std::sync::Mutex;

use grim_core::error::{Error, Result};
use grim_kvtransport::{NetworkKvClient, PromptChannel};
use grim_memory::KvBlockPool;

/// Bounded exponential-backoff retry policy for cross-node KV transfers.
///
/// Transfers are one-shot TCP; a node that is briefly busy (receiver thread
/// saturated, restart in progress) previously failed the whole handoff on the
/// first refused connection. Transient connection-level failures are retried
/// with exponential backoff; protocol-level failures (checksum, "not
/// available", bad address) fail fast — retrying those can never succeed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    /// Total attempts including the first (default 3).
    pub max_attempts: u32,
    /// Backoff before the second attempt, in ms (default 50).
    pub initial_backoff_ms: u64,
    /// Backoff ceiling, in ms (default 800).
    pub max_backoff_ms: u64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_backoff_ms: 50,
            max_backoff_ms: 800,
        }
    }
}

/// Whether a transfer error is worth retrying: connection-level failures
/// only. Server-authored answers ("not available"), protocol mismatches,
/// checksum errors, and caller bugs (empty payload, bad address) are final.
fn is_transient_transfer_error(e: &Error) -> bool {
    let msg = e.to_string();
    // io::Error display strings for connect/read/write failures:
    // "Connection refused (os error 111)", "Connection reset by peer",
    // "timed out", plus the crate's own "connection failed"/"read error"
    // wrappers around them.
    msg.contains("connection failed")
        || msg.contains("Connection refused")
        || msg.contains("reset by peer")
        || msg.contains("timed out")
        || msg.contains("read error")
        || msg.contains("write error")
}

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

/// Layer-pipelined asynchronous KV transfer manager.
///
/// Overlaps compute with communication by transmitting KV blocks layer-by-layer
/// as soon as each transformer layer completes prompt prefill.
pub struct LayerPipelinedKvStreamer {
    decode_node_addr: String,
    kv_client: NetworkKvClient,
    retry: RetryPolicy,
}

impl LayerPipelinedKvStreamer {
    pub fn new(decode_node_addr: String) -> Self {
        Self {
            decode_node_addr: decode_node_addr.clone(),
            kv_client: NetworkKvClient::new(decode_node_addr),
            retry: RetryPolicy::default(),
        }
    }

    /// Stream a single layer's KV block slice asynchronously across the wire.
    /// Transient connection failures retry per the default [`RetryPolicy`].
    pub fn stream_layer_block(
        &self,
        block_id: usize,
        layer_idx: u32,
        k: &[f32],
        v: &[f32],
    ) -> Result<()> {
        let mut attempt: u32 = 1;
        let mut backoff_ms = self.retry.initial_backoff_ms;
        loop {
            match self.kv_client.send_block_remote(
                block_id,
                layer_idx,
                k,
                v,
                &self.decode_node_addr,
            ) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    if attempt >= self.retry.max_attempts.max(1) || !is_transient_transfer_error(&e)
                    {
                        return Err(e);
                    }
                    std::thread::sleep(std::time::Duration::from_millis(backoff_ms));
                    backoff_ms = (backoff_ms.saturating_mul(2)).min(self.retry.max_backoff_ms);
                    attempt += 1;
                }
            }
        }
    }
}

/// Disaggregated serving cluster orchestrator managing prefill and decode worker roles,
/// worker heartbeats, and dynamic failover policies.
#[derive(Debug, Clone)]
pub struct DisaggOrchestrator {
    pub config: DisaggConfig,
    prefill_healthy: bool,
    decode_healthy: bool,
    last_prefill_heartbeat_ms: u64,
    last_decode_heartbeat_ms: u64,
}

impl DisaggOrchestrator {
    pub fn new(config: DisaggConfig) -> Self {
        Self {
            config,
            prefill_healthy: true,
            decode_healthy: true,
            last_prefill_heartbeat_ms: 0,
            last_decode_heartbeat_ms: 0,
        }
    }

    /// Record a heartbeat timestamp for a node role.
    pub fn record_heartbeat(&mut self, role: PoolRole, now_ms: u64) {
        match role {
            PoolRole::Prefill => {
                self.prefill_healthy = true;
                self.last_prefill_heartbeat_ms = now_ms;
            }
            PoolRole::Decode => {
                self.decode_healthy = true;
                self.last_decode_heartbeat_ms = now_ms;
            }
            PoolRole::Colocated => {
                self.prefill_healthy = true;
                self.decode_healthy = true;
            }
        }
    }

    /// Check health against a timeout window; fails over to colocated fallback if dead.
    pub fn evaluate_failover(&mut self, now_ms: u64, timeout_ms: u64) -> PoolRole {
        if self.config.role == PoolRole::Decode
            && now_ms.saturating_sub(self.last_prefill_heartbeat_ms) > timeout_ms
            && self.last_prefill_heartbeat_ms > 0
        {
            self.prefill_healthy = false;
            // Fallback to local colocated execution if prefill remote is unreachable
            return PoolRole::Colocated;
        } else if self.config.role == PoolRole::Prefill
            && now_ms.saturating_sub(self.last_decode_heartbeat_ms) > timeout_ms
            && self.last_decode_heartbeat_ms > 0
        {
            self.decode_healthy = false;
            return PoolRole::Colocated;
        }
        self.config.role
    }

    /// Whether this node instance should execute compute-heavy prefill passes.
    #[inline]
    pub fn handles_prefill(&self) -> bool {
        matches!(self.config.role, PoolRole::Colocated | PoolRole::Prefill)
    }

    /// Whether this node instance should execute bandwidth-heavy autoregressive decode passes.
    #[inline]
    pub fn handles_decode(&self) -> bool {
        matches!(self.config.role, PoolRole::Colocated | PoolRole::Decode)
    }
}

/// Router for cross-pool dispatch and network KV transfers.
///
/// When `pool` is set, the router extracts **real** KV block data from the
/// engine's `KvBlockPool` via `read_keys()` / `read_values()` rather than
/// synthesising fake payloads.
pub trait DisaggRouterT: Send + Sync {
    /// Dispatch a prefill task — sends the request's real KV blocks (its
    /// logical→physical block table) from the local pool to the prefill node.
    fn dispatch_prefill(&self, request_id: u64, tokens: &[u32], block_ids: &[usize]) -> Result<()>;
    /// Transfer real KV blocks (identified by physical `block_ids`) from the
    /// local pool to the decode node.
    fn transfer_kv_cache(&self, request_id: u64, block_ids: &[usize]) -> Result<()>;
    /// Dispatches a step-decode task to a dedicated decode engine.
    fn dispatch_decode(
        &self,
        request_id: u64,
        last_token: u32,
        assignment: PoolAssignment,
        block_ids: &[usize],
    ) -> Result<()>;
}

pub struct DisaggRouter {
    pub prefill_node_addr: String,
    pub decode_node_addr: String,
    pub pool_role: PoolRole,
    kv_client: NetworkKvClient,
    retry: RetryPolicy,
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
            retry: RetryPolicy::default(),
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

    /// Override the transfer retry policy (default: 3 attempts, 50 ms→800 ms
    /// exponential backoff on transient connection failures).
    pub fn with_retry_policy(mut self, retry: RetryPolicy) -> Self {
        self.retry = retry;
        self
    }

    /// Run `f` with the router's bounded exponential-backoff retry. Only
    /// transient connection-level failures retry; everything else fails on
    /// the first attempt.
    fn retrying<T>(&self, op: &str, mut f: impl FnMut() -> Result<T>) -> Result<T> {
        let mut attempt: u32 = 1;
        let mut backoff_ms = self.retry.initial_backoff_ms;
        loop {
            match f() {
                Ok(v) => return Ok(v),
                Err(e) => {
                    if attempt >= self.retry.max_attempts.max(1) || !is_transient_transfer_error(&e)
                    {
                        return Err(e);
                    }
                    eprintln!(
                        "[grim-disagg] {op}: transient failure (attempt {attempt}/{}), \
                         retrying in {backoff_ms}ms: {e}",
                        self.retry.max_attempts
                    );
                    std::thread::sleep(std::time::Duration::from_millis(backoff_ms));
                    backoff_ms = (backoff_ms.saturating_mul(2)).min(self.retry.max_backoff_ms);
                    attempt += 1;
                }
            }
        }
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
            let num_layers = pool.num_layers(b_id);
            for layer in 0..num_layers {
                if let (Some(k_data), Some(v_data)) = (
                    pool.read_layer_keys(b_id, layer),
                    pool.read_layer_values(b_id, layer),
                ) {
                    if !k_data.is_empty() && !v_data.is_empty() {
                        self.retrying("transfer_kv_cache_real", || {
                            self.kv_client.send_block_remote(
                                b_id,
                                layer as u32,
                                k_data,
                                v_data,
                                &self.decode_node_addr,
                            )
                        })?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Zero-copy Peer-to-Peer direct memory KV migration across GPUs within the same node.
    ///
    /// Copies KV block allocations directly across device pools without intermediate
    /// host allocations or network serialization.
    pub fn transfer_kv_p2p_direct(
        &self,
        block_ids: &[usize],
        src_pool: &mut grim_memory::KvBlockPool,
        dst_pool: &mut grim_memory::KvBlockPool,
    ) -> Result<()> {
        if block_ids.is_empty() {
            return Err(Error::KvCache(
                "P2P direct transfer error: block list cannot be empty".into(),
            ));
        }

        for &b_id in block_ids {
            let num_layers = src_pool.num_layers(b_id);
            for layer in 0..num_layers {
                if let (Some(k_data), Some(v_data)) = (
                    src_pool.read_layer_keys(b_id, layer),
                    src_pool.read_layer_values(b_id, layer),
                ) {
                    let num_tokens = k_data.len();
                    dst_pool.write_layer_keys(b_id, layer, k_data, num_tokens);
                    dst_pool.write_layer_values(b_id, layer, v_data);
                }
            }
        }
        Ok(())
    }

    /// Transfer real multi-layer KV blocks from a `PagedKvCache` across all layers.
    pub fn transfer_paged_cache_real(
        &self,
        _request_id: u64,
        block_ids: &[usize],
        cache: &grim_memory::PagedKvCache,
    ) -> Result<()> {
        if block_ids.is_empty() {
            return Err(Error::KvCache(
                "Handoff protocol error: block list cannot be empty".into(),
            ));
        }
        let num_layers = cache.num_layers();
        for layer in 0..num_layers {
            for &b_id in block_ids {
                if let Some((k_slice, v_slice)) = cache.layer_block_slice(layer, b_id) {
                    self.retrying("transfer_paged_cache_real", || {
                        self.kv_client.send_block_remote(
                            b_id,
                            layer as u32,
                            k_slice,
                            v_slice,
                            &self.decode_node_addr,
                        )
                    })?;
                }
            }
        }
        Ok(())
    }

    /// Extract real KV blocks for `block_ids` from the stored pool and send
    /// them to the prefill node address.
    ///
    /// Only the request's own blocks are transferred — the pool is shared
    /// across concurrent requests, so a full-pool scan would leak other
    /// requests' KV cache over the wire.
    fn extract_and_send_prefill(
        &self,
        request_id: u64,
        tokens: &[u32],
        block_ids: &[usize],
    ) -> Result<()> {
        let pool = self.pool.as_ref().ok_or_else(|| {
            Error::KvCache(
                "dispatch_prefill: no KvBlockPool attached for real KV extraction".into(),
            )
        })?;
        let guard = pool
            .lock()
            .map_err(|e| Error::KvCache(format!("dispatch_prefill: pool mutex poisoned: {e}")))?;
        // Stream this request's real KV blocks to the prefill node.
        for &block_id in block_ids {
            let k_data = guard.read_keys(block_id);
            let v_data = guard.read_values(block_id);
            self.retrying("extract_and_send_prefill", || {
                self.kv_client.send_block_remote(
                    block_id,
                    0,
                    k_data,
                    v_data,
                    &self.prefill_node_addr,
                )
            })?;
        }
        // Forward the prompt token IDs over the real control channel
        // (PROMPT_FLAG protocol message stored in the receiver's
        // PromptChannel). The previous mechanism smuggled them through as a
        // fake KV "meta-block" at id = pool.num_blocks() that no receiver
        // ever decoded.
        self.retrying("extract_and_send_prefill(prompt)", || {
            self.kv_client
                .send_prompt_tokens(request_id, tokens, &self.prefill_node_addr)
        })
    }

    /// Extract real KV blocks for `block_ids` from the stored pool and send
    /// them to the decode node address as a decode step context.
    fn extract_and_send_decode(
        &self,
        _request_id: u64,
        _last_token: u32,
        block_ids: &[usize],
    ) -> Result<()> {
        let pool = self.pool.as_ref().ok_or_else(|| {
            Error::KvCache("dispatch_decode: no KvBlockPool attached for real KV extraction".into())
        })?;
        let guard = pool
            .lock()
            .map_err(|e| Error::KvCache(format!("dispatch_decode: pool mutex poisoned: {e}")))?;
        for &block_id in block_ids {
            let k_data = guard.read_keys(block_id);
            let v_data = guard.read_values(block_id);
            self.retrying("extract_and_send_decode", || {
                self.kv_client.send_block_remote(
                    block_id,
                    0,
                    k_data,
                    v_data,
                    &self.decode_node_addr,
                )
            })?;
        }
        Ok(())
    }

    /// Fetch a single KV block from the prefill node (decode → prefill pull).
    /// Returns the raw key/value float slices. Transient connection failures
    /// retry per the router's [`RetryPolicy`]; a server "not available"
    /// answer is final. Errors otherwise — never fabricates data.
    pub fn fetch_kv_block(
        &self,
        block_id: usize,
        layer_idx: u32,
        block_elems: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.retrying("fetch_kv_block", || {
            self.kv_client.fetch_block_remote(
                block_id,
                layer_idx,
                &self.prefill_node_addr,
                block_elems,
            )
        })
    }

    /// Send a prompt-token control message to the prefill node (push model,
    /// retry-backed). The receiver stores it in its `PromptChannel`.
    pub fn send_prompt_tokens(&self, request_id: u64, tokens: &[u32]) -> Result<()> {
        self.retrying("send_prompt_tokens", || {
            self.kv_client
                .send_prompt_tokens(request_id, tokens, &self.prefill_node_addr)
        })
    }

    /// Dispatches a single layer's KV block key/value slice to the decode node address.
    pub fn send_layer_block_remote(
        &self,
        block_id: usize,
        layer_idx: u32,
        k: &[f32],
        v: &[f32],
    ) -> Result<()> {
        self.retrying("send_layer_block_remote", || {
            self.kv_client
                .send_block_remote(block_id, layer_idx, k, v, &self.decode_node_addr)
        })
    }
}

impl DisaggRouterT for DisaggRouter {
    /// Dispatch a prefill task to a dedicated prefill engine.
    fn dispatch_prefill(&self, request_id: u64, tokens: &[u32], block_ids: &[usize]) -> Result<()> {
        self.extract_and_send_prefill(request_id, tokens, block_ids)
    }

    /// Transfer real KV blocks to the decode node.
    fn transfer_kv_cache(&self, _request_id: u64, block_ids: &[usize]) -> Result<()> {
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
        let guard = pool
            .lock()
            .map_err(|e| Error::KvCache(format!("transfer_kv_cache: pool mutex poisoned: {e}")))?;
        self.transfer_kv_cache_real(_request_id, block_ids, &guard)
    }

    /// Dispatch a step-decode task to a dedicated decode engine.
    fn dispatch_decode(
        &self,
        request_id: u64,
        last_token: u32,
        _assignment: PoolAssignment,
        block_ids: &[usize],
    ) -> Result<()> {
        self.extract_and_send_decode(request_id, last_token, block_ids)
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
    /// Store for prompt-token control messages arriving over the wire
    /// (`NetworkKvClient::send_prompt_tokens` → `PROMPT_FLAG` protocol).
    prompts: PromptChannel,
    #[allow(dead_code)]
    handle: Option<std::thread::JoinHandle<()>>,
}

impl KvReceiverServer {
    /// Start a KV receiver server on `listen_addr` that writes into `pool`.
    /// Prompt-token control messages are collected in an internal
    /// [`PromptChannel`] — read them via [`KvReceiverServer::prompt_channel`].
    pub fn new(listen_addr: &str, pool: Arc<Mutex<KvBlockPool>>) -> Result<Self> {
        let prompts = PromptChannel::new();
        let handle = grim_kvtransport::start_kv_receiver_server_with_prompts(
            listen_addr,
            pool.clone(),
            prompts.clone(),
        )?;
        Ok(Self {
            listen_addr: listen_addr.to_string(),
            pool,
            prompts,
            handle: Some(handle),
        })
    }

    pub fn listen_addr(&self) -> &str {
        &self.listen_addr
    }

    pub fn pool(&self) -> &Arc<Mutex<KvBlockPool>> {
        &self.pool
    }

    /// Consume prompt tokens received for `request_id` over the control
    /// channel. `None` when no prompt message has arrived (yet).
    pub fn take_prompt_tokens(&self, request_id: u64) -> Option<Vec<u32>> {
        self.prompts.take(request_id)
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
        let prefill_res = router.dispatch_prefill(42, &[101, 102, 103], &[0, 1, 2, 3]);
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
        let decode_res = router.dispatch_decode(42, 104, assignment, &[0, 1, 2, 3]);
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

    /// Retry policy: transient connection failures retry with backoff and
    /// eventually succeed once the receiver comes up. A listener binds only
    /// after a delay, so the first attempts are refused; with enough attempts
    /// the transfer lands and the data round-trips.
    #[test]
    fn test_retry_recovers_from_refused_connection() {
        let port = find_free_port();
        let addr = format!("127.0.0.1:{port}");

        // Destination pool + receiver that only starts after a delay —
        // early attempts hit "Connection refused".
        let dest_pool = Arc::new(Mutex::new(KvBlockPool::new(4, 2, 4)));
        let late_addr = addr.clone();
        let late_pool = dest_pool.clone();
        let listener_thread = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(150));
            crate::KvReceiverServer::new(&late_addr, late_pool).unwrap()
        });

        // Source pool with one known block.
        let mut pool = KvBlockPool::new(4, 2, 4);
        let k_data: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let v_data: Vec<f32> = (0..128).map(|i| -i as f32).collect();
        pool.write_keys(0, &k_data, 16);
        pool.write_values(0, &v_data);
        let shared_pool = Arc::new(Mutex::new(pool));

        // Generous retry budget so the delayed listener is reached.
        let router = DisaggRouter::new(&addr, &addr, PoolRole::Prefill)
            .with_retry_policy(RetryPolicy {
                max_attempts: 40,
                initial_backoff_ms: 20,
                max_backoff_ms: 100,
            })
            .with_pool(shared_pool.clone());

        router
            .transfer_kv_cache_real(1, &[0], &shared_pool.lock().unwrap())
            .expect("transfer must succeed after retries");

        std::thread::sleep(std::time::Duration::from_millis(300));
        let dest_guard = dest_pool.lock().unwrap();
        assert_eq!(
            &dest_guard.read_keys(0)[..k_data.len()],
            &k_data[..],
            "data must round-trip once the receiver is up"
        );
        listener_thread.join().unwrap();
    }

    /// A server-authored "not available" answer is final: the fetch must
    /// fail fast (one attempt), not burn the full retry budget.
    #[test]
    fn test_fetch_not_available_fails_fast_without_retry() {
        let port = find_free_port();
        let addr = format!("127.0.0.1:{port}");
        let _receiver =
            crate::KvReceiverServer::new(&addr, Arc::new(Mutex::new(KvBlockPool::new(4, 2, 4))))
                .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(150));

        // max_attempts huge + huge backoff: if the router retried "not
        // available" the test would take ≥1.5s. Fail-fast finishes in ms.
        let router =
            DisaggRouter::new(&addr, &addr, PoolRole::Decode).with_retry_policy(RetryPolicy {
                max_attempts: 100,
                initial_backoff_ms: 500,
                max_backoff_ms: 1_000,
            });
        let start = std::time::Instant::now();
        let res = router.fetch_kv_block(3, 0, 128);
        let elapsed = start.elapsed();
        let err = res.expect_err("unwritten block must error");
        assert!(err.to_string().contains("not available"), "{err}");
        assert!(
            elapsed < std::time::Duration::from_millis(400),
            "'not available' must not be retried (took {elapsed:?})"
        );
    }

    /// Real control channel: prompt tokens ride the PROMPT_FLAG protocol
    /// message and are consumable at the receiver via `take_prompt_tokens`.
    #[test]
    fn test_prompt_control_channel_roundtrip() {
        let port = find_free_port();
        let addr = format!("127.0.0.1:{port}");
        let receiver =
            crate::KvReceiverServer::new(&addr, Arc::new(Mutex::new(KvBlockPool::new(4, 2, 4))))
                .unwrap();

        let sender = DisaggRouter::new(&addr, &addr, PoolRole::Prefill);
        sender
            .send_prompt_tokens(77, &[11, 22, 33, 44])
            .expect("prompt control send must succeed");

        // Receiver thread commits asynchronously; poll the channel.
        let mut got = None;
        for _ in 0..50 {
            if let Some(t) = receiver.take_prompt_tokens(77) {
                got = Some(t);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert_eq!(got, Some(vec![11, 22, 33, 44]));
        assert!(
            receiver.take_prompt_tokens(77).is_none(),
            "take must consume the prompt"
        );
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
        let decode_res = router.dispatch_decode(99, 200, assignment, &[0]);
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
        let _receiver = crate::KvReceiverServer::new(&addr, shared_pool.clone()).unwrap();

        // Create the destination pool (receiver side).
        let dest_pool = Arc::new(Mutex::new(KvBlockPool::new(4, 2, 4)));

        // Start a second receiver on the destination
        let port2 = find_free_port();
        let addr2 = format!("127.0.0.1:{port2}");
        let receiver = crate::KvReceiverServer::new(&addr2, dest_pool.clone()).unwrap();

        // Create router with the source pool.
        let router =
            DisaggRouter::new(&addr2, &addr2, PoolRole::Prefill).with_pool(shared_pool.clone());

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

    /// F8/F10 integration gate: the decode→prefill PULL path
    /// (`DisaggRouter::fetch_kv_block`) against a live receiver backed by a
    /// real `KvBlockPool`. Before the fix, the server never answered fetch
    /// requests at all — both sides deadlocked on the first pull. Populates
    /// BOTH layers via the existing network PUSH path, then pulls them back
    /// and requires exact round-trip equality.
    #[test]
    fn test_fetch_kv_block_pull_roundtrip_real_pool() {
        // Pool geometry: 4 blocks, 2 heads, head_dim 4 → elem_per_token 8,
        // block_elems 16*8 = 128. Both layers arrive over the wire: layer 0
        // through `write_layer_keys(0, …)` (which mirrors into `key_data`),
        // layer 1 through the per-layer store — this is the shape a real
        // multi-layer handoff produces.
        let pool = KvBlockPool::new(4, 2, 4);
        let block_id = 1usize;
        let k0: Vec<f32> = (0..128).map(|i| i as f32 * 0.5).collect();
        let v0: Vec<f32> = (0..128).map(|i| (i as f32 * -0.25) - 2.0).collect();
        let k1: Vec<f32> = vec![7.0f32; 128];
        let v1: Vec<f32> = vec![-9.0f32; 128];

        let shared_pool = Arc::new(Mutex::new(pool));
        let port = find_free_port();
        let addr = format!("127.0.0.1:{port}");
        let _receiver = crate::KvReceiverServer::new(&addr, shared_pool).unwrap();

        // Push both layers through the router's own send path.
        let router = DisaggRouter::new(&addr, &addr, PoolRole::Prefill);
        router
            .send_layer_block_remote(block_id, 0, &k0, &v0)
            .expect("layer-0 push must succeed");
        router
            .send_layer_block_remote(block_id, 1, &k1, &v1)
            .expect("layer-1 push must succeed");

        // Let the receiver thread commit both writes before pulling.
        std::thread::sleep(std::time::Duration::from_millis(300));

        let (got_k0, got_v0) = router
            .fetch_kv_block(block_id, 0, 128)
            .expect("layer-0 fetch must round-trip");
        assert_eq!(got_k0, k0, "layer-0 keys must round-trip exactly");
        assert_eq!(got_v0, v0, "layer-0 values must round-trip exactly");

        let (got_k1, got_v1) = router
            .fetch_kv_block(block_id, 1, 128)
            .expect("layer-1 fetch must round-trip");
        assert_eq!(got_k1, k1, "layer-1 keys must round-trip exactly");
        assert_eq!(got_v1, v1, "layer-1 values must round-trip exactly");

        // A block the pool holds but never received data for must produce a
        // prompt "not available" error, not a hang.
        let res = router.fetch_kv_block(3, 0, 128);
        let err = res.expect_err("fetching an unwritten block must error");
        assert!(
            err.to_string().contains("not available"),
            "error should say the block is not available: {err}"
        );
    }
}
