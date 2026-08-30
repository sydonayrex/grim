//! Distributed serving and disaggregation layer: decouples prefill/decode, manages cross-node KV cache transfers.
//!

pub mod bloom;
pub mod lookup;
pub mod coherence;

pub use bloom::BloomFilter;
pub use lookup::LookupClient;
pub use coherence::{CacheCoherenceManager, InvalidationMsg};

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
/// only. Server-authored answers ("not available", ACK rejections),
/// protocol mismatches, checksum errors, and caller bugs (empty payload,
/// bad address) are final.
fn is_transient_transfer_error(e: &Error) -> bool {
    let msg = e.to_string();
    // io::Error display strings for connect/read/write failures:
    // "Connection refused (os error 111)", "Connection reset by peer",
    // "Broken pipe (os error 32)", "timed out", plus the crate's own
    // "connection failed"/"read error"/"write error" wrappers around them.
    // (grim-core carries no typed error kinds, so classification is by
    // display string — keep this in sync with every failure string the
    // transport can emit for a connection-level fault.)
    msg.contains("connection failed")
        || msg.contains("Connection refused")
        || msg.contains("reset by peer")
        || msg.contains("Broken pipe")
        || msg.contains("connection aborted")
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
    /// Validate batch shape: non-empty, 2D dims match block count, and all
    /// blocks carry the same non-zero element count (consumers slice the
    /// drained buffer with a uniform stride, so mixed lengths would
    /// silently misalign it).
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
        let first_len = self.blocks[0].data.len();
        if first_len == 0 {
            return Err(Error::KvCache(
                "ReMPMigrationBatch: blocks must carry non-empty data".into(),
            ));
        }
        if let Some(bad) = self
            .blocks
            .iter()
            .find(|b| b.data.len() != first_len)
            .map(|b| (b.layer_idx, b.seq_chunk, b.data.len()))
        {
            return Err(Error::KvCache(format!(
                "ReMPMigrationBatch: inconsistent block sizes (first block has {first_len} \
                 elements, block layer={} chunk={} has {})",
                bad.0, bad.1, bad.2
            )));
        }
        Ok(())
    }

    /// Drain 2D block matrix to flat KV buffer (layer-major, chunk-major). No re-layout needed.
    pub fn migrate(&self) -> Result<Vec<f32>> {
        self.validate()?;
        // Index blocks once: (layer, chunk) → slot in the flat output.
        // The naive per-slot `blocks.iter().find(...)` scans the whole batch
        // L·C times — quadratic in blocks, which is 4×10⁸ comparisons for an
        // 80-layer × 256-chunk model.
        let mut slots: Vec<Option<&KvBlock>> =
            vec![None; self.num_layers as usize * self.num_seq_chunks as usize];
        for block in &self.blocks {
            if block.layer_idx >= self.num_layers || block.seq_chunk >= self.num_seq_chunks {
                return Err(Error::KvCache(format!(
                    "ReMPMigrationBatch: block layer={} chunk={} outside batch bounds \
                     ({} layers × {} chunks)",
                    block.layer_idx, block.seq_chunk, self.num_layers, self.num_seq_chunks
                )));
            }
            let slot =
                block.layer_idx as usize * self.num_seq_chunks as usize + block.seq_chunk as usize;
            if slots[slot].is_some() {
                return Err(Error::KvCache(format!(
                    "ReMPMigrationBatch: duplicate block layer={} chunk={}",
                    block.layer_idx, block.seq_chunk
                )));
            }
            slots[slot] = Some(block);
        }
        let block_elems = self.blocks[0].data.len();
        let mut flat = Vec::with_capacity(slots.len() * block_elems);
        for (slot, block) in slots.iter().enumerate() {
            match block {
                Some(block) => flat.extend_from_slice(&block.data),
                None => {
                    let layer = slot / self.num_seq_chunks as usize;
                    let chunk = slot % self.num_seq_chunks as usize;
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

/// Run `f` under `retry`: transient connection-level failures are retried
/// with exponential backoff (capped at `max_backoff_ms`); everything else
/// fails on the first attempt. Shared by [`DisaggRouter`] and
/// [`LayerPipelinedKvStreamer`] so retry semantics cannot drift apart.
fn retry_with_policy<T>(
    retry: &RetryPolicy,
    op: &str,
    mut f: impl FnMut() -> Result<T>,
) -> Result<T> {
    let mut attempt: u32 = 1;
    let mut backoff_ms = retry.initial_backoff_ms;
    loop {
        match f() {
            Ok(v) => return Ok(v),
            Err(e) => {
                if attempt >= retry.max_attempts.max(1) || !is_transient_transfer_error(&e) {
                    return Err(e);
                }
                eprintln!(
                    "[grim-disagg] {op}: transient failure (attempt {attempt}/{}), \
                     retrying in {backoff_ms}ms: {e}",
                    retry.max_attempts
                );
                std::thread::sleep(std::time::Duration::from_millis(backoff_ms));
                backoff_ms = (backoff_ms.saturating_mul(2)).min(retry.max_backoff_ms);
                attempt += 1;
            }
        }
    }
}

/// Layer-pipelined KV transfer manager.
///
/// Overlaps compute with communication by transmitting KV blocks layer-by-layer
/// as soon as each transformer layer completes prompt prefill. Calls are
/// synchronous — each blocks the caller until the receiver ACKs the block —
/// so pipeline by invoking from a dedicated sender thread's layer loop.
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

    /// Set the transport wire protocol (TCP, RDMA/RoCE, or UCX Direct).
    pub fn with_protocol(mut self, protocol: grim_kvtransport::TransportProtocol) -> Self {
        self.kv_client.protocol = protocol;
        self
    }

    /// Stream a single layer's KV block slice across the wire. Transient
    /// connection failures retry per the default [`RetryPolicy`];
    /// `num_tokens` is the block's valid token count (carried end-to-end so
    /// the receiver does not derive it from the zero-padded payload).
    pub fn stream_layer_block(
        &self,
        block_id: usize,
        layer_idx: u32,
        k: &[f32],
        v: &[f32],
        num_tokens: usize,
    ) -> Result<()> {
        retry_with_policy(&self.retry, "stream_layer_block", || {
            self.kv_client.send_block_remote(
                block_id,
                layer_idx,
                k,
                v,
                num_tokens,
                &self.decode_node_addr,
            )
        })
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
    /// The role failover actually resolved to; may differ from
    /// `config.role` after a failover. `handles_prefill`/`handles_decode`
    /// consult this so they stay truthful post-failover.
    effective_role: PoolRole,
    /// Clock baseline for peers that have never heartbeated: the first
    /// `evaluate_failover` call. Without it, a node whose peer never sent
    /// a single heartbeat would trust that peer forever (the old
    /// `last_heartbeat_ms > 0` guard made the timeout unreachable).
    first_eval_ms: Option<u64>,
}

impl DisaggOrchestrator {
    pub fn new(config: DisaggConfig) -> Self {
        let effective_role = config.role;
        Self {
            config,
            prefill_healthy: true,
            decode_healthy: true,
            last_prefill_heartbeat_ms: 0,
            last_decode_heartbeat_ms: 0,
            effective_role,
            first_eval_ms: None,
        }
    }

    /// Record a heartbeat timestamp for a node role. A Colocated heartbeat
    /// means this node is alive in BOTH roles, so both freshness
    /// timestamps advance.
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
                self.last_prefill_heartbeat_ms = now_ms;
                self.last_decode_heartbeat_ms = now_ms;
            }
        }
    }

    /// Check peer freshness against a timeout window; fail over to
    /// colocated execution when the remote peer is presumed dead. The
    /// resolved role is retained — [`Self::effective_role`],
    /// [`Self::handles_prefill`], and [`Self::handles_decode`] all report
    /// it until the next evaluation. A peer that was failed over recovers
    /// automatically once its heartbeats resume.
    pub fn evaluate_failover(&mut self, now_ms: u64, timeout_ms: u64) -> PoolRole {
        let first_eval = *self.first_eval_ms.get_or_insert(now_ms);
        // Freshness baseline: the peer's last heartbeat, or — for a peer
        // that has NEVER sent one — the first evaluation (startup grace =
        // one timeout window from when failover checking began).
        let prefill_baseline = if self.last_prefill_heartbeat_ms > 0 {
            self.last_prefill_heartbeat_ms
        } else {
            first_eval
        };
        let decode_baseline = if self.last_decode_heartbeat_ms > 0 {
            self.last_decode_heartbeat_ms
        } else {
            first_eval
        };
        let prefill_fresh = now_ms.saturating_sub(prefill_baseline) <= timeout_ms;
        let decode_fresh = now_ms.saturating_sub(decode_baseline) <= timeout_ms;
        self.effective_role = match self.config.role {
            PoolRole::Decode if !prefill_fresh => {
                self.prefill_healthy = false;
                // Fallback to local colocated execution: prefill remote is unreachable.
                PoolRole::Colocated
            }
            PoolRole::Prefill if !decode_fresh => {
                self.decode_healthy = false;
                PoolRole::Colocated
            }
            _ => self.config.role,
        };
        self.effective_role
    }

    /// The role failover last resolved to: `Colocated` means the remote
    /// peer is presumed dead and remote handoff should be gated off.
    pub fn effective_role(&self) -> PoolRole {
        self.effective_role
    }

    /// Whether this node instance should execute compute-heavy prefill passes.
    #[inline]
    pub fn handles_prefill(&self) -> bool {
        matches!(self.effective_role, PoolRole::Colocated | PoolRole::Prefill)
    }

    /// Whether this node instance should execute bandwidth-heavy autoregressive decode passes.
    #[inline]
    pub fn handles_decode(&self) -> bool {
        matches!(self.effective_role, PoolRole::Colocated | PoolRole::Decode)
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

/// One (block, layer) KV payload snapshotted from a local pool, ready to
/// send. `num_tokens` is the source block's valid token count — carried
/// end-to-end so the receiver stores the real fill state instead of
/// deriving a full-block count from the zero-padded payload length.
struct BlockLayerPayload {
    block_id: usize,
    layer_idx: u32,
    k: Vec<f32>,
    v: Vec<f32>,
    num_tokens: usize,
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

    /// Set the transport wire protocol (TCP, RDMA/RoCE, or UCX Direct).
    pub fn with_protocol(mut self, protocol: grim_kvtransport::TransportProtocol) -> Self {
        self.kv_client.protocol = protocol;
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
    fn retrying<T>(&self, op: &str, f: impl FnMut() -> Result<T>) -> Result<T> {
        retry_with_policy(&self.retry, op, f)
    }

    /// Copy one request's KV blocks (all layers, with their valid token
    /// counts) out of `pool`. Bounds-checked and fill-checked: an
    /// out-of-range or never-populated block id is an error, not a panic
    /// or a silent zero-block handoff.
    ///
    /// Callers must release the pool lock before sending the returned
    /// payloads over the network — the pool must never be held across
    /// connect timeouts, retry backoff sleeps, or stalled sockets.
    fn snapshot_block_layers(
        pool: &grim_memory::KvBlockPool,
        block_ids: &[usize],
        op: &str,
    ) -> Result<Vec<BlockLayerPayload>> {
        let mut payloads = Vec::new();
        for &block_id in block_ids {
            if block_id >= pool.num_blocks() {
                return Err(Error::KvCache(format!(
                    "{op}: block_id {block_id} out of range (pool holds {} blocks)",
                    pool.num_blocks()
                )));
            }
            if !pool.block_is_received(block_id) {
                return Err(Error::KvCache(format!(
                    "{op}: block {block_id} has no KV data to transfer"
                )));
            }
            let num_tokens = pool.block_num_tokens(block_id).unwrap_or(0);
            for layer in 0..pool.num_layers(block_id) {
                if let (Some(k), Some(v)) = (
                    pool.read_layer_keys(block_id, layer),
                    pool.read_layer_values(block_id, layer),
                ) {
                    if !k.is_empty() && !v.is_empty() {
                        payloads.push(BlockLayerPayload {
                            block_id,
                            layer_idx: layer as u32,
                            k: k.to_vec(),
                            v: v.to_vec(),
                            num_tokens,
                        });
                    }
                }
            }
        }
        Ok(payloads)
    }

    /// Send snapshot payloads to `target` over the wire: pipelined through
    /// one connection per chunk of messages, each message ACKed by the
    /// receiver, each chunk retried on transient connection failures.
    fn send_block_layers(
        &self,
        payloads: Vec<BlockLayerPayload>,
        target: &str,
        op: &str,
    ) -> Result<()> {
        const MAX_MESSAGES_PER_CONNECTION: usize = 256;
        for chunk in payloads.chunks(MAX_MESSAGES_PER_CONNECTION) {
            let transfers: Vec<grim_kvtransport::KvBlockTransfer<'_>> = chunk
                .iter()
                .map(|p| grim_kvtransport::KvBlockTransfer {
                    block_id: p.block_id,
                    layer_idx: p.layer_idx,
                    k: &p.k,
                    v: &p.v,
                    num_tokens: p.num_tokens,
                })
                .collect();
            self.retrying(op, || {
                self.kv_client.send_blocks_batch_remote(&transfers, target)
            })?;
        }
        Ok(())
    }

    /// Transfer real KV blocks extracted from a physical KvBlockPool to the
    /// remote decode engine.
    ///
    /// Each block's per-layer key/value data (and its valid token count) is
    /// snapshotted from the pool, then sent over TCP using the V3 wire
    /// protocol (magic + checksum + num_tokens in the header, per-message
    /// ACK from the receiver).
    ///
    /// Note: `pool` is only read here. Entry points that hold the pool
    /// mutex themselves (`transfer_kv_cache`, `dispatch_prefill`,
    /// `dispatch_decode`) snapshot first and release the lock before any
    /// network I/O happens.
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
        let payloads = Self::snapshot_block_layers(pool, block_ids, "transfer_kv_cache_real")?;
        self.send_block_layers(payloads, &self.decode_node_addr, "transfer_kv_cache_real")
    }

    /// Copy KV blocks between two LOCAL pools (same node), all layers,
    /// preserving each block's valid token count. This is a host-memory
    /// memcpy path — no network, no serialization, and also not VRAM-to-VRAM
    /// DMA despite the historical "zero-copy" name.
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
            if b_id >= src_pool.num_blocks() || b_id >= dst_pool.num_blocks() {
                return Err(Error::KvCache(format!(
                    "transfer_kv_p2p_direct: block_id {b_id} out of range"
                )));
            }
            if !src_pool.block_is_received(b_id) {
                return Err(Error::KvCache(format!(
                    "transfer_kv_p2p_direct: source block {b_id} has no KV data"
                )));
            }
            // The source block's true valid token count. Passing the
            // ELEMENT count here instead would mark every destination block
            // as fully valid (element counts always exceed BLOCK_SIZE).
            let num_tokens = src_pool.block_num_tokens(b_id).unwrap_or(0);
            let layers: Vec<(usize, Vec<f32>, Vec<f32>)> = (0..src_pool.num_layers(b_id))
                .filter_map(|layer| {
                    let k = src_pool.read_layer_keys(b_id, layer)?;
                    let v = src_pool.read_layer_values(b_id, layer)?;
                    Some((layer, k.to_vec(), v.to_vec()))
                })
                .collect();
            for (layer, k_data, v_data) in layers {
                dst_pool.write_layer_keys(b_id, layer, &k_data, num_tokens);
                dst_pool.write_layer_values(b_id, layer, &v_data);
            }
        }
        Ok(())
    }

    /// Transfer real multi-layer KV blocks from a `PagedKvCache` across all
    /// layers, carrying each block's valid token count.
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
        let mut payloads = Vec::new();
        for layer in 0..num_layers {
            for &b_id in block_ids {
                if let Some((k_slice, v_slice)) = cache.layer_block_slice(layer, b_id) {
                    if k_slice.is_empty() || v_slice.is_empty() {
                        continue;
                    }
                    let num_tokens = cache
                        .block_num_tokens(b_id)
                        .ok_or_else(|| {
                            Error::KvCache(format!(
                                "transfer_paged_cache_real: block {b_id} outside the cache's block table"
                            ))
                        })?
                        .min(k_slice.len());
                    payloads.push(BlockLayerPayload {
                        block_id: b_id,
                        layer_idx: layer as u32,
                        k: k_slice.to_vec(),
                        v: v_slice.to_vec(),
                        num_tokens,
                    });
                }
            }
        }
        self.send_block_layers(payloads, &self.decode_node_addr, "transfer_paged_cache_real")
    }

    /// Extract real KV blocks for `block_ids` from the stored pool and send
    /// them to the prefill node address.
    ///
    /// Only the request's own blocks are transferred — the pool is shared
    /// across concurrent requests, so a full-pool scan would leak other
    /// requests' KV cache over the wire. The pool lock is held only for the
    /// snapshot; all network I/O happens after it is released.
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
        // Snapshot every layer of this request's real KV blocks, then drop
        // the lock before touching the network.
        let payloads = {
            let guard = pool.lock().map_err(|e| {
                Error::KvCache(format!("dispatch_prefill: pool mutex poisoned: {e}"))
            })?;
            Self::snapshot_block_layers(&guard, block_ids, "dispatch_prefill")?
        };
        self.send_block_layers(payloads, &self.prefill_node_addr, "dispatch_prefill")?;
        // Forward the prompt token IDs over the real control channel
        // (PROMPT_FLAG protocol message stored in the receiver's
        // PromptChannel). The previous mechanism smuggled them through as a
        // fake KV "meta-block" at id = pool.num_blocks() that no receiver
        // ever decoded.
        self.retrying("dispatch_prefill(prompt)", || {
            self.kv_client
                .send_prompt_tokens(request_id, tokens, &self.prefill_node_addr)
        })
    }

    /// Extract real KV blocks for `block_ids` from the stored pool and send
    /// them to the decode node address as a decode step context. Snapshot
    /// under the lock, send after releasing it (all layers, like every
    /// other handoff path).
    fn extract_and_send_decode(
        &self,
        _request_id: u64,
        _last_token: u32,
        block_ids: &[usize],
    ) -> Result<()> {
        let pool = self.pool.as_ref().ok_or_else(|| {
            Error::KvCache("dispatch_decode: no KvBlockPool attached for real KV extraction".into())
        })?;
        let payloads = {
            let guard = pool.lock().map_err(|e| {
                Error::KvCache(format!("dispatch_decode: pool mutex poisoned: {e}"))
            })?;
            Self::snapshot_block_layers(&guard, block_ids, "dispatch_decode")?
        };
        self.send_block_layers(payloads, &self.decode_node_addr, "dispatch_decode")
    }

    /// Fetch a single KV block from the prefill node (decode → prefill pull).
    /// Returns the key/value float data plus the block's stored valid token
    /// count. Transient connection failures retry per the router's
    /// [`RetryPolicy`]; a server "not available" answer is final. Errors
    /// otherwise — never fabricates data.
    pub fn fetch_kv_block(
        &self,
        block_id: usize,
        layer_idx: u32,
        block_elems: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, usize)> {
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

    /// Dispatches a single layer's KV block key/value slice to the decode
    /// node address. `num_tokens` is the block's valid token count.
    pub fn send_layer_block_remote(
        &self,
        block_id: usize,
        layer_idx: u32,
        k: &[f32],
        v: &[f32],
        num_tokens: usize,
    ) -> Result<()> {
        self.retrying("send_layer_block_remote", || {
            self.kv_client.send_block_remote(
                block_id,
                layer_idx,
                k,
                v,
                num_tokens,
                &self.decode_node_addr,
            )
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
        // Snapshot under the lock, then release it before any network I/O:
        // the engine's shared pool must never be held across connect
        // timeouts, retry backoff sleeps, or stalled sockets.
        let payloads = {
            let guard = pool.lock().map_err(|e| {
                Error::KvCache(format!("transfer_kv_cache: pool mutex poisoned: {e}"))
            })?;
            Self::snapshot_block_layers(&guard, block_ids, "transfer_kv_cache")?
        };
        self.send_block_layers(payloads, &self.decode_node_addr, "transfer_kv_cache")
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
/// incoming V3 protocol block transfers (magic / checksum / num_tokens
/// verification, per-message ACK), and writes the received key/value data
/// into the pool via the `KvBlockStore` trait.
///
/// Dropping the server sets the receiver's stop flag; the accept loop exits
/// within its poll interval (the listener thread is not joined on drop).
pub struct KvReceiverServer {
    listen_addr: String,
    pool: Arc<Mutex<KvBlockPool>>,
    /// Store for prompt-token control messages arriving over the wire
    /// (`NetworkKvClient::send_prompt_tokens` → `PROMPT_FLAG` protocol).
    prompts: PromptChannel,
    stop: Option<Arc<std::sync::atomic::AtomicBool>>,
    /// Stop flag for the shared-memory inbox poller (`SharedMemP2p`
    /// handoffs). `None` when the listen address had no explicit port.
    shm_stop: Option<Arc<std::sync::atomic::AtomicBool>>,
}

impl KvReceiverServer {
    /// Start a KV receiver server on `listen_addr` that writes into `pool`.
    /// Prompt-token control messages are collected in an internal
    /// [`PromptChannel`] — read them via [`KvReceiverServer::take_prompt_tokens`].
    ///
    /// Also starts the same-host shared-memory inbox poller when the listen
    /// address carries an explicit port, so `SharedMemP2p` senders can hand
    /// off without touching a socket; TCP senders are unaffected.
    pub fn new(listen_addr: &str, pool: Arc<Mutex<KvBlockPool>>) -> Result<Self> {
        use std::sync::atomic::AtomicBool;
        let prompts = PromptChannel::new();
        // The receiver thread is detached: dropping the server signals the
        // stop flag (see Drop), and the accept loop exits on its next poll.
        let (_handle, stop) = grim_kvtransport::start_kv_receiver_server_stoppable(
            listen_addr,
            pool.clone(),
            prompts.clone(),
        )?;

        let shm_stop =
            match listen_addr.rsplit(':').next().and_then(|p| p.parse::<u16>().ok()) {
                Some(port) if port != 0 => {
                    let flag = Arc::new(AtomicBool::new(false));
                    match grim_kvtransport::start_shm_inbox_poller(
                        listen_addr,
                        pool.clone(),
                        flag.clone(),
                    ) {
                        Ok(_handle) => Some(flag),
                        Err(e) => {
                            eprintln!(
                                "[grim-disagg] shm inbox poller disabled for {listen_addr}: {e}"
                            );
                            None
                        }
                    }
                }
                _ => None,
            };

        Ok(Self {
            listen_addr: listen_addr.to_string(),
            pool,
            prompts,
            stop: Some(stop),
            shm_stop,
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

impl Drop for KvReceiverServer {
    fn drop(&mut self) {
        if let Some(stop) = &self.stop {
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        if let Some(stop) = &self.shm_stop {
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
        }
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
    /// and requires exact round-trip equality (data AND token counts).
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

        // Push both layers through the router's own send path. Pushes ACK
        // on commit, so the data is in the pool by the time they return.
        let router = DisaggRouter::new(&addr, &addr, PoolRole::Prefill);
        router
            .send_layer_block_remote(block_id, 0, &k0, &v0, 16)
            .expect("layer-0 push must succeed");
        router
            .send_layer_block_remote(block_id, 1, &k1, &v1, 16)
            .expect("layer-1 push must succeed");

        let (got_k0, got_v0, got_tokens0) = router
            .fetch_kv_block(block_id, 0, 128)
            .expect("layer-0 fetch must round-trip");
        assert_eq!(got_k0, k0, "layer-0 keys must round-trip exactly");
        assert_eq!(got_v0, v0, "layer-0 values must round-trip exactly");
        assert_eq!(got_tokens0, 16, "layer-0 token count must round-trip");

        let (got_k1, got_v1, got_tokens1) = router
            .fetch_kv_block(block_id, 1, 128)
            .expect("layer-1 fetch must round-trip");
        assert_eq!(got_k1, k1, "layer-1 keys must round-trip exactly");
        assert_eq!(got_v1, v1, "layer-1 values must round-trip exactly");
        assert_eq!(got_tokens1, 16, "layer-1 token count must round-trip");

        // A block the pool holds but never received data for must produce a
        // prompt "not available" error, not a hang.
        let res = router.fetch_kv_block(3, 0, 128);
        let err = res.expect_err("fetching an unwritten block must error");
        assert!(
            err.to_string().contains("not available"),
            "error should say the block is not available: {err}"
        );
    }

    /// G4 gate: float payloads survive the wire bit-exactly — including
    /// NaN payloads, which f32 equality cannot check (NaN != NaN), so the
    /// assertion compares bit patterns.
    #[test]
    fn test_nan_float_bit_exact_roundtrip() {
        let nan_a = f32::from_bits(0x7fc0_0001);
        let nan_b = f32::from_bits(0x7ff0_0000);
        let k: Vec<f32> = vec![nan_a, -0.0, f32::INFINITY, f32::NEG_INFINITY, 1.5e-38];
        let v: Vec<f32> = vec![nan_b, f32::MIN, f32::MAX, 0.0, nan_a];
        let block_elems = 128;
        let mut k_full = k.clone();
        let mut v_full = v.clone();
        k_full.resize(block_elems, 0.0);
        v_full.resize(block_elems, 0.0);

        let src_shared = Arc::new(Mutex::new(KvBlockPool::new(4, 2, 4)));
        {
            let mut src = src_shared.lock().unwrap();
            src.write_layer_keys(0, 0, &k_full, 16);
            src.write_layer_values(0, 0, &v_full);
        }
        let port = find_free_port();
        let addr = format!("127.0.0.1:{port}");
        let _receiver = crate::KvReceiverServer::new(&addr, src_shared.clone()).unwrap();

        let router = DisaggRouter::new(&addr, &addr, PoolRole::Prefill).with_pool(src_shared);
        router.send_layer_block_remote(0, 0, &k_full, &v_full, 16).unwrap();

        let (got_k, got_v, _) = router.fetch_kv_block(0, 0, block_elems).unwrap();
        let k_bits: Vec<u32> = k_full.iter().map(|f| f.to_bits()).collect();
        let v_bits: Vec<u32> = v_full.iter().map(|f| f.to_bits()).collect();
        let got_k_bits: Vec<u32> = got_k.iter().map(|f| f.to_bits()).collect();
        let got_v_bits: Vec<u32> = got_v.iter().map(|f| f.to_bits()).collect();
        assert_eq!(k_bits, got_k_bits, "K bits (incl. NaN payloads) must round-trip");
        assert_eq!(v_bits, got_v_bits, "V bits (incl. NaN payloads) must round-trip");
        // And the edge values specifically: negative zero must not become positive zero.
        assert!((got_k[1].to_bits() >> 31) & 1 == 1, "-0.0 must stay -0.0");
    }

    /// L1 regression (wire push): a partially-filled block must arrive with
    /// its REAL valid token count, not inflated to BLOCK_SIZE. Before the
    /// num_tokens wire field, the receiver derived the count from the
    /// zero-padded payload length and marked every block full.
    #[test]
    fn test_wire_push_preserves_num_tokens() {
        // Geometry: 4 blocks, 2 heads, head_dim 4 → elem_per_token 8,
        // BLOCK_SIZE 16. Source block carries 5 valid tokens (40 elements).
        let mut src = KvBlockPool::new(4, 2, 4);
        let k: Vec<f32> = (0..40).map(|i| i as f32).collect();
        let v: Vec<f32> = vec![9.0f32; 40];
        src.write_keys(0, &k, 5);
        src.write_values(0, &v);
        assert_eq!(src.block_num_tokens(0), Some(5));

        let src_shared = Arc::new(Mutex::new(src));
        let dest_shared = Arc::new(Mutex::new(KvBlockPool::new(4, 2, 4)));

        let port = find_free_port();
        let addr = format!("127.0.0.1:{port}");
        let _receiver = crate::KvReceiverServer::new(&addr, dest_shared.clone()).unwrap();

        let router = DisaggRouter::new(&addr, &addr, PoolRole::Prefill).with_pool(src_shared);
        router
            .transfer_kv_cache(1, &[0])
            .expect("wire push must succeed (ACK = committed)");

        let dest_guard = dest_shared.lock().unwrap();
        assert_eq!(
            dest_guard.block_num_tokens(0),
            Some(5),
            "destination must carry the source's valid token count, not BLOCK_SIZE"
        );
        assert_eq!(&dest_guard.read_keys(0)[..40], &k[..]);
    }

    /// L1 regression (P2P): the direct pool-to-pool path must copy the
    /// source block's valid token count — not `k_data.len()`, which is an
    /// ELEMENT count and always inflates to BLOCK_SIZE after capping.
    #[test]
    fn test_p2p_preserves_num_tokens() {
        let mut src = KvBlockPool::new(4, 2, 4);
        let mut dst = KvBlockPool::new(4, 2, 4);
        let k: Vec<f32> = vec![3.5f32; 56]; // 7 tokens × 8 elems
        let v: Vec<f32> = vec![2.5f32; 56];
        src.write_layer_keys(0, 0, &k, 7);
        src.write_layer_values(0, 0, &v);

        let router = DisaggRouter::new("127.0.0.1:1", "127.0.0.1:1", PoolRole::Colocated);
        router
            .transfer_kv_p2p_direct(&[0], &mut src, &mut dst)
            .expect("p2p must succeed");

        assert_eq!(
            dst.block_num_tokens(0),
            Some(7),
            "p2p destination must carry the source's valid token count"
        );
        assert_eq!(&dst.read_layer_keys(0, 0).unwrap()[..56], &k[..]);
    }

    /// B2 regression: a receiver that cannot store a block (id out of
    /// range on the destination pool) must surface a FINAL error on the
    /// sender, not report Ok while the data silently vanished.
    #[test]
    fn test_receiver_rejection_surfaces_as_error() {
        // Source pool has 8 blocks; destination receiver only 1 — block 1
        // is out of range on the receiving end.
        let mut src = KvBlockPool::new(8, 2, 4);
        let k: Vec<f32> = vec![1.0f32; 128];
        let v: Vec<f32> = vec![2.0f32; 128];
        src.write_keys(1, &k, 16);
        src.write_values(1, &v);

        let src_shared = Arc::new(Mutex::new(src));
        let port = find_free_port();
        let addr = format!("127.0.0.1:{port}");
        let _receiver =
            crate::KvReceiverServer::new(&addr, Arc::new(Mutex::new(KvBlockPool::new(1, 2, 4))))
                .unwrap();

        let router = DisaggRouter::new(&addr, &addr, PoolRole::Prefill).with_pool(src_shared);
        let res = router.transfer_kv_cache(1, &[1]);
        let err = res.expect_err("receiver-side rejection must not report success");
        assert!(
            err.to_string().contains("rejected"),
            "error should say the receiver rejected the block: {err}"
        );
    }

    /// L2 regression: dispatch_prefill must transfer ALL layers of each
    /// block, not just the layer-0 mirror. Verified by pulling each layer
    /// back from the prefill node and comparing exactly.
    #[test]
    fn test_dispatch_prefill_transfers_all_layers() {
        let mut src = KvBlockPool::new(4, 2, 4);
        let k0: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let v0: Vec<f32> = (0..128).map(|i| -(i as f32)).collect();
        let k1: Vec<f32> = (0..128).map(|i| i as f32 + 0.25).collect();
        let v1: Vec<f32> = (0..128).map(|i| -(i as f32) - 0.25).collect();
        src.write_layer_keys(0, 0, &k0, 16);
        src.write_layer_values(0, 0, &v0);
        src.write_layer_keys(0, 1, &k1, 16);
        src.write_layer_values(0, 1, &v1);

        let src_shared = Arc::new(Mutex::new(src));
        let prefill_shared = Arc::new(Mutex::new(KvBlockPool::new(4, 2, 4)));
        let port = find_free_port();
        let addr = format!("127.0.0.1:{port}");
        let _prefill_receiver = crate::KvReceiverServer::new(&addr, prefill_shared).unwrap();

        let router = DisaggRouter::new(&addr, &addr, PoolRole::Decode).with_pool(src_shared);
        router
            .dispatch_prefill(7, &[101, 102], &[0])
            .expect("dispatch_prefill must succeed");

        // Pull both layers back from the prefill node — the prefill node
        // must hold the full multi-layer context, not just layer 0.
        let (got_k0, got_v0, _) = router
            .fetch_kv_block(0, 0, 128)
            .expect("layer-0 must round-trip through the prefill node");
        assert_eq!(got_k0, k0);
        assert_eq!(got_v0, v0);
        let (got_k1, got_v1, _) = router
            .fetch_kv_block(0, 1, 128)
            .expect("layer-1 must round-trip through the prefill node (L2: all layers)");
        assert_eq!(got_k1, k1);
        assert_eq!(got_v1, v1);
    }

    /// B5 regression: a peer that has NEVER heartbeated must still be
    /// failed over after one timeout window from the first evaluation —
    /// the old `last_heartbeat > 0` guard trusted a silent peer forever.
    #[test]
    fn test_failover_without_any_heartbeat() {
        let cfg = DisaggConfig {
            role: PoolRole::Decode,
            prefill_addr: "127.0.0.1:9001".into(),
            decode_addr: "127.0.0.1:9002".into(),
        };
        let mut orch = DisaggOrchestrator::new(cfg);
        // First evaluation sets the freshness baseline; within the window
        // the configured role stands (startup grace).
        assert_eq!(orch.evaluate_failover(1_000, 500), PoolRole::Decode);
        // One window of total silence later: fail over.
        assert_eq!(orch.evaluate_failover(1_501, 500), PoolRole::Colocated);
    }

    /// Failover must recover when the peer's heartbeats resume.
    #[test]
    fn test_failover_recovers_after_heartbeat_resumes() {
        let cfg = DisaggConfig {
            role: PoolRole::Decode,
            prefill_addr: "127.0.0.1:9001".into(),
            decode_addr: "127.0.0.1:9002".into(),
        };
        let mut orch = DisaggOrchestrator::new(cfg);
        orch.record_heartbeat(PoolRole::Prefill, 1_000);
        assert_eq!(orch.evaluate_failover(2_000, 500), PoolRole::Colocated);
        assert_eq!(orch.effective_role(), PoolRole::Colocated);
        // A failed-over node executes BOTH roles locally — the fallback's
        // whole point — but no longer reports the pure Decode role.
        assert!(orch.handles_decode());
        assert!(orch.handles_prefill());
        orch.record_heartbeat(PoolRole::Prefill, 2_100);
        assert_eq!(orch.evaluate_failover(2_200, 500), PoolRole::Decode);
        assert_eq!(orch.effective_role(), PoolRole::Decode);
        assert!(orch.handles_decode());
        assert!(!orch.handles_prefill(), "recovered peer restores pure decode role");
    }

    /// B6 regression: a Colocated heartbeat must advance BOTH freshness
    /// timestamps, not just the health flags.
    #[test]
    fn test_colocated_heartbeat_refreshes_timestamps() {
        let cfg = DisaggConfig {
            role: PoolRole::Prefill,
            prefill_addr: "127.0.0.1:9001".into(),
            decode_addr: "127.0.0.1:9002".into(),
        };
        let mut orch = DisaggOrchestrator::new(cfg);
        orch.record_heartbeat(PoolRole::Colocated, 1_000);
        // Prefill role checks the DECODE heartbeat timestamp; a colocated
        // heartbeat must have refreshed it or this fails over spuriously.
        assert_eq!(orch.evaluate_failover(1_200, 500), PoolRole::Prefill);
    }

    /// P2 regression: mixed block sizes must be rejected — the flat buffer
    /// would silently misalign any consumer slicing with a uniform stride.
    #[test]
    fn test_migration_rejects_inconsistent_block_sizes() {
        let router = DisaggRouter::new("localhost:0", "localhost:0", PoolRole::Colocated);
        let mut batch = ReMPMigrationBatch {
            num_layers: 2,
            num_seq_chunks: 2,
            blocks: Vec::new(),
        };
        for layer in 0..2u32 {
            for chunk in 0..2u32 {
                let len = if layer == 1 && chunk == 0 { 8 } else { 4 };
                batch.blocks.push(KvBlock {
                    data: vec![1.0; len],
                    layer_idx: layer,
                    seq_chunk: chunk,
                });
            }
        }
        let res = router.transfer_kv_colocated(1, &batch);
        assert!(res.is_err(), "mixed block sizes must be rejected");
        assert!(res.unwrap_err().to_string().contains("inconsistent"));
    }

    /// P1 regression: a duplicate (layer, chunk) must be rejected (it would
    /// otherwise silently leave the duplicate's slot's counterpart missing).
    #[test]
    fn test_migration_rejects_duplicate_blocks() {
        let batch = ReMPMigrationBatch {
            num_layers: 1,
            num_seq_chunks: 2,
            blocks: vec![
                KvBlock {
                    data: vec![1.0; 4],
                    layer_idx: 0,
                    seq_chunk: 0,
                },
                KvBlock {
                    data: vec![2.0; 4],
                    layer_idx: 0,
                    seq_chunk: 0,
                },
            ],
        };
        let err = batch.migrate().expect_err("duplicate block must error");
        assert!(err.to_string().contains("duplicate"), "{err}");
    }
}
