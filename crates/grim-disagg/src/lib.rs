//! Distributed serving and disaggregation layer: decouples prefill/decode, manages cross-node KV cache transfers.
//!

/// ReMP 2D KV-cache migration (WI-8): coalesced 128-byte block-major transfer within same VRAM pool.
use std::sync::Arc;
use std::sync::Mutex;

use grim_core::error::{Error, Result};
use grim_kvtransport::{NetworkKvClient, PromptChannel};
use grim_memory::KvBlockPool;

/// Bounded exponential-backoff retry policy for cross-node KV transfers.
/// Transfers are one-shot TCP; a node that is briefly busy (receiver thread saturated, restart in.
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

/// Whether a transfer error is worth retrying: connection-level failures only.
/// Server-authored answers ("not available", ACK rejections), protocol mismatches, checksum errors, and caller bugs (empty payload,.
fn is_transient_transfer_error(e: &Error) -> bool {
    let msg = e.to_string();
    // io::Error display strings for connect/read/write failures: "Connection refused (os error 111)", "Connection reset by peer", "Broken pipe (os error 32)", "timed out", plus the crate's own "connection failed"/"read error"/"write error" wrappers around them.
    // (grim-core carries no typed error kinds, so classification is by display string - keep this.
    msg.contains("connection failed")
        || msg.contains("Connection refused")
        || msg.contains("reset by peer")
        || msg.contains("Broken pipe")
        || msg.contains("connection aborted")
        || msg.contains("timed out")
        || msg.contains("read error")
        || msg.contains("write error")
}

/// Node role in serving cluster (§5.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolRole {
    Colocated,
    Prefill,
    Decode,
}

/// Disaggregation configuration carried by `EngineConfig` and the serving CLI.
/// Defined here (in grim-disagg) so both `grim-engine` and `grim-server` can depend on it without creating.
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

/// Run `f` under `retry`: transient connection-level failures are retried with exponential backoff (capped at `max_backoff_ms`); everything else fails on the first attempt.
/// Shared by [`DisaggRouter`] and [`LayerPipelinedKvStreamer`] so retry semantics cannot drift apart.
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

/// Disaggregated serving cluster orchestrator managing prefill and decode worker roles,
/// worker heartbeats, and dynamic failover policies.
#[derive(Debug, Clone)]
pub struct DisaggOrchestrator {
    pub config: DisaggConfig,
    prefill_healthy: bool,
    decode_healthy: bool,
    last_prefill_heartbeat_ms: u64,
    last_decode_heartbeat_ms: u64,
    /// The role failover actually resolved to; may differ from `config.role` after a failover.
    /// `handles_prefill`/`handles_decode` consult this so they stay truthful post-failover.
    effective_role: PoolRole,
    /// Clock baseline for peers that have never heartbeated: the first `evaluate_failover` call.
    /// Without it, a node whose peer never sent a single heartbeat would trust that peer.
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

    /// Record a heartbeat timestamp for a node role.
    /// A Colocated heartbeat means this node is alive in BOTH roles, so both freshness timestamps.
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

    /// Check peer freshness against a timeout window; fail over to colocated execution when the remote peer is presumed dead.
    /// The resolved role is retained - [`Self::effective_role`], [`Self::handles_prefill`], and [`Self::handles_decode`] all report it until the.
    pub fn evaluate_failover(&mut self, now_ms: u64, timeout_ms: u64) -> PoolRole {
        let first_eval = *self.first_eval_ms.get_or_insert(now_ms);
        // Freshness baseline: the peer's last heartbeat, or - for a peer that has NEVER sent
        // one - the first evaluation (startup grace = one timeout window from when failover checking began).
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

/// One (block, layer) KV payload snapshotted from a local pool, ready to send.
/// `num_tokens` is the source block's valid token count - carried end-to-end so the receiver stores.
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

    /// Set the transport wire protocol (TCP, RDMA/RoCE, or UCX Direct).
    pub fn with_protocol(mut self, protocol: grim_kvtransport::TransportProtocol) -> Self {
        self.kv_client.protocol = protocol;
        self
    }

    /// Run `f` with the router's bounded exponential-backoff retry.
    /// Only transient connection-level failures retry; everything else fails on the first attempt.
    fn retrying<T>(&self, op: &str, f: impl FnMut() -> Result<T>) -> Result<T> {
        retry_with_policy(&self.retry, op, f)
    }

    /// Copy one request's KV blocks (all layers, with their valid token counts) out of `pool`.
    /// Bounds-checked and fill-checked: an out-of-range or never-populated block id is an error, not a panic.
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

    /// Send snapshot payloads to `target` over the wire: pipelined through one connection per chunk
    /// of messages, each message ACKed by the receiver, each chunk retried on transient connection failures.
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

    /// Transfer real KV blocks extracted from a physical KvBlockPool to the remote decode engine.
    /// Each block's per-layer key/value data (and its valid token count) is snapshotted from the pool,.
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

    /// Fetch a single KV block from the prefill node (decode → prefill pull).
    /// Returns the key/value float data plus the block's stored valid token count.
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

/// Background TCP receiver server for cross-node KV cache block ingestion.
/// Wraps `grim_kvtransport::start_kv_receiver_server` with a reference to the engine's `KvBlockPool`.
pub struct KvReceiverServer {
    listen_addr: String,
    pool: Arc<Mutex<KvBlockPool>>,
    /// Store for prompt-token control messages arriving over the wire
    /// (`NetworkKvClient::send_prompt_tokens` → `PROMPT_FLAG` protocol).
    stop: Option<Arc<std::sync::atomic::AtomicBool>>,
    /// Stop flag for the shared-memory inbox poller (`SharedMemP2p`
    /// handoffs). `None` when the listen address had no explicit port.
    shm_stop: Option<Arc<std::sync::atomic::AtomicBool>>,
}

impl KvReceiverServer {
    /// Start a KV receiver server on `listen_addr` that writes into `pool`.
    /// Prompt-token control messages are collected in an internal [`PromptChannel`] consumed by the receiver thread.
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

        let shm_stop = match listen_addr
            .rsplit(':')
            .next()
            .and_then(|p| p.parse::<u16>().ok())
        {
            Some(port) if port != 0 => {
                let flag = Arc::new(AtomicBool::new(false));
                match grim_kvtransport::start_shm_inbox_poller(
                    listen_addr,
                    pool.clone(),
                    flag.clone(),
                ) {
                    Ok(_handle) => Some(flag),
                    Err(e) => {
                        eprintln!("[grim-disagg] shm inbox poller disabled for {listen_addr}: {e}");
                        None
                    }
                }
            }
            _ => None,
        };

        Ok(Self {
            listen_addr: listen_addr.to_string(),
            pool,
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
        // A failed-over node executes BOTH roles locally — the fallback's
        // whole point — but no longer reports the pure Decode role.
        assert!(orch.handles_decode());
        assert!(orch.handles_prefill());
        orch.record_heartbeat(PoolRole::Prefill, 2_100);
        assert_eq!(orch.evaluate_failover(2_200, 500), PoolRole::Decode);
        assert!(orch.handles_decode());
        assert!(
            !orch.handles_prefill(),
            "recovered peer restores pure decode role"
        );
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

    /// F8/F10 integration gate: the decode→prefill PULL path (`DisaggRouter::fetch_kv_block`) against a live receiver backed by a real `KvBlockPool`.
    #[test]
    fn test_fetch_kv_block_pull_roundtrip_real_pool() {
        // Pool geometry: 4 blocks, 2 heads, head_dim 4 → elem_per_token 8, block_elems 16*8 = 128.
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
        assert_eq!(got_k1, k1);
        assert_eq!(got_v1, v1);
        assert_eq!(got_tokens1, 16);

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
