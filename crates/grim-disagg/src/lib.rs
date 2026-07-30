//! Distributed serving and disaggregation layer for Grim.
//!
//! Exposes the `DisaggRouter` trait and concrete implementations to decouple
//! prefill execution from decode execution and manage cross-node KV cache transfers.
//!
//! ## SCYTHE-2 WI-8: ReMP 2D KV-cache migration
//! The ReMP (Re-Migration Protocol) transfers KV blocks between sessions in the
//! same VRAM pool without a PCIe round-trip. The outer loop is over the KV
//! "block rows" (attention heads × seq chunks); the inner loop is over
//! transformer layers. This 2-D pattern keeps the memory-bus access pattern
//! coalesced (128-byte cache-line alignment per block) and matches the block-
//! major KV layout emitted by `kv_to_block_major` (`device/layout.rs:73`).

use grim_core::error::{Error, Result};
use grim_kvtransport::NetworkKvClient;

// ── ReMP KV migration types (WI-8) ────────────────────────────────────────────

/// Represents a single KV block (one transformer layer, one sequence chunk).
/// Block layout mirrors `kv_to_block_major` (`device/layout.rs`): all K then all V,
/// each of size `num_heads × block_size × head_dim × dtype_bytes`.
#[derive(Debug, Clone)]
pub struct KvBlock {
    /// Flattened f32 data for K and V of one layer segment.
    pub data: Vec<f32>,
    /// Layer index this block belongs to.
    pub layer_idx: u32,
    /// Position (sequence chunk) index within the layer.
    pub seq_chunk: u32,
}

/// A 2D KV migration batch (ReMP inner state).
///
/// Outer dim = num_layers, inner dim = num_seq_chunks (block rows).
/// Calling `migrate()` drains the matrix into the destination session.
#[derive(Debug, Default)]
pub struct ReMPMigrationBatch {
    pub blocks: Vec<KvBlock>,
    pub num_layers: u32,
    pub num_seq_chunks: u32,
}

impl ReMPMigrationBatch {
    /// Validate that the batch is non-empty and has the expected 2D shape.
    pub fn validate(&self) -> Result<()> {
        if self.blocks.is_empty() {
            return Err(Error::KvCache("ReMPMigrationBatch: no blocks".into()));
        }
        let expected = self.num_layers as usize * self.num_seq_chunks as usize;
        if self.blocks.len() != expected {
            return Err(Error::KvCache(format!(
                "ReMPMigrationBatch: expected {} blocks ({} layers × {} chunks), got {}",
                expected, self.num_layers, self.num_seq_chunks, self.blocks.len()
            )));
        }
        Ok(())
    }

    /// Drain the 2D block matrix into a flat KV buffer (layer-major, then chunk-major).
    ///
    /// Returns `(flat_data, num_elements)` suitable for passing to a session's
    /// `restore_kv` method. The 2D iteration order matches the access pattern
    /// of `paged_attention` so no re-layout is needed.
    pub fn migrate(&self) -> Result<Vec<f32>> {
        self.validate()?;
        let total: usize = self.blocks.iter().map(|b| b.data.len()).sum();
        let mut flat = Vec::with_capacity(total);
        // Outer loop: layers (matches KvBlockPool::block_for_layer access pattern).
        for layer in 0..self.num_layers {
            // Inner loop: seq chunks (matches paged_attention KV fetch order).
            for chunk in 0..self.num_seq_chunks {
                if let Some(block) = self.blocks.iter().find(
                    |b| b.layer_idx == layer && b.seq_chunk == chunk
                ) {
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

/// The role that a given node or pool plays in the serving cluster (§5.6)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolRole {
    Colocated,
    Prefill,
    Decode,
}

/// Carry context containing the source prefill node parameters inside the decode step (§5.6)
#[derive(Debug, Clone)]
pub struct PoolAssignment {
    pub source_prefill_pool_addr: String,
    pub request_id: u64,
}

/// The router interface managing cross-pool dispatch and network transfers.
pub trait DisaggRouterT: Send + Sync {
    fn dispatch_prefill(&self, request_id: u64, tokens: &[u32]) -> Result<()>;
    fn transfer_kv_cache(&self, request_id: u64, num_blocks: usize) -> Result<()>;
    fn dispatch_decode(&self, request_id: u64, last_token: u32, assignment: PoolAssignment) -> Result<()>;
}

pub struct DisaggRouter {
    pub prefill_node_addr: String,
    pub decode_node_addr: String,
    pub pool_role: PoolRole,
    kv_client: NetworkKvClient,
    use_rdma: bool,
}

impl DisaggRouter {
    pub fn new(prefill_node_addr: &str, decode_node_addr: &str, pool_role: PoolRole) -> Self {
        Self {
            prefill_node_addr: prefill_node_addr.to_string(),
            decode_node_addr: decode_node_addr.to_string(),
            pool_role,
            kv_client: NetworkKvClient::new(prefill_node_addr.to_string()),
            use_rdma: false, // Default to TCP transport, fallback to RDMA if flag enabled
        }
    }

    /// Enable RDMA fallback network layer
    pub fn enable_rdma(&mut self, enabled: bool) {
        self.use_rdma = enabled;
    }
}

impl DisaggRouterT for DisaggRouter {
    /// Dispatches a prefill task to a dedicated prefill execution engine.
    ///
    /// NOTE: The remote dispatch path is not yet wired to a real execution
    /// engine — the network transport is a stub (see `NetworkKvClient`).
    /// Returning `Ok(())` here previously masked the missing implementation;
    /// callers now receive an explicit `Unimplemented` error so the gap can
    /// never silently succeed in production.
    fn dispatch_prefill(&self, request_id: u64, _tokens: &[u32]) -> Result<()> {
        Err(Error::Unimplemented(format!(
            "dispatch_prefill(request_id={request_id}) -> prefill node {} (pool role {:?}): \
             remote prefill dispatch is not yet implemented; the NetworkKvClient is a stub.",
            self.prefill_node_addr, self.pool_role
        )))
    }

    /// Performs the KV-transfer step from the prefill engine to the decode engine
    /// utilizing the remote network-transport KV client.
    ///
    /// NOTE: Real KV block transfer is blocked on the `NetworkKvClient`
    /// stub implementation (it returns hardcoded data instead of round-tripping
    /// the actual KV tensors). Rather than shipping mock `0.5f32` values
    /// and pretending the handoff succeeded, we surface an explicit error and
    /// only validate the protocol-level precondition (`num_blocks > 0`).
    fn transfer_kv_cache(&self, request_id: u64, num_blocks: usize) -> Result<()> {
        // Handoff protocol handshake validation — this part is real.
        if num_blocks == 0 {
            return Err(Error::KvCache("Handoff protocol error: block count cannot be zero".into()));
        }

        Err(Error::Unimplemented(format!(
            "transfer_kv_cache(request_id={request_id}, num_blocks={num_blocks}) over {}: \
             NetworkKvClient is a stub that returns hardcoded 1.0/2.0 values instead of the \
             transferred KV tensors; refusing to fake the handoff. not yet implemented.",
            if self.use_rdma { "RDMA" } else { "TCP" }
        )))
    }

    /// Dispatches a step-decode task to a dedicated decode execution engine.
    ///
    /// NOTE: Like `dispatch_prefill`, the remote decode dispatch is not yet
    /// wired to a real engine. Returning `Ok(())` previously masked the gap.
    fn dispatch_decode(&self, request_id: u64, _last_token: u32, assignment: PoolAssignment) -> Result<()> {
        Err(Error::Unimplemented(format!(
            "dispatch_decode(request_id={request_id}, prefill pool src {}) -> decode node {}: \
             remote decode dispatch is not yet implemented; the NetworkKvClient is a stub.",
            assignment.source_prefill_pool_addr, self.decode_node_addr
        )))
    }
}

impl DisaggRouter {
    /// SCYTHE-2 WI-8: ReMP 2D KV-cache migration for colocated (same-node) transfers.
    ///
    /// Transfers KV blocks within the same VRAM pool without a network round-trip.
    /// The 2D iteration order (outer = layers, inner = seq chunks) matches the
    /// block-major KV layout from `kv_to_block_major` (`device/layout.rs:73`)
    /// so no re-layout is needed. Returns the flat migrated buffer on success.
    ///
    /// ## Contract
    /// - `batch.num_layers × batch.num_seq_chunks` must equal `batch.blocks.len()`.
    /// - All blocks must be present (no sparse batch).
    /// - This method does NOT route over the network; it is only valid when
    ///   `pool_role == PoolRole::Colocated` or when the prefill and decode engines
    ///   share address space.
    ///
    /// ## Staleness safety (scythe2.md §3.5 mode C)
    /// If the session ID changes between prefill and decode (e.g., a preemption),
    /// the migrated buffer is still valid — the caller is responsible for binding
    /// it to the new session ID. The migration itself is idempotent (pure copy).
    pub fn transfer_kv_colocated(
        &self,
        request_id: u64,
        batch: &ReMPMigrationBatch,
    ) -> Result<Vec<f32>> {
        // Protocol validation: ReMP requires colocated role.
        if self.pool_role != PoolRole::Colocated {
            return Err(Error::KvCache(format!(
                "transfer_kv_colocated: only valid for Colocated pool role, got {:?}",
                self.pool_role
            )));
        }
        // Validate and drain the 2D batch.
        let flat = batch.migrate().map_err(|e| Error::KvCache(format!(
            "transfer_kv_colocated(request_id={request_id}): migration failed: {e}"
        )))?;
        eprintln!(
            "[grim-disagg] ReMP colocated transfer: request_id={request_id}, \
             layers={}, chunks={}, total_elements={}",
            batch.num_layers, batch.num_seq_chunks, flat.len()
        );
        Ok(flat)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disaggregated_kv_routing() {
        let router = DisaggRouter::new("10.0.0.1:8000", "10.0.0.2:8000", PoolRole::Prefill);

        // Verify router fields are correctly stored
        assert_eq!(router.prefill_node_addr, "10.0.0.1:8000");
        assert_eq!(router.decode_node_addr, "10.0.0.2:8000");
        assert_eq!(router.pool_role, PoolRole::Prefill);

        // Dispatch prefill — not yet implemented; must surface an explicit
        // error rather than silently succeeding (sims.md issue #1).
        let prefill_res = router.dispatch_prefill(42, &[101, 102, 103]);
        assert!(prefill_res.is_err(), "dispatch_prefill should fail loudly, not silently Ok");
        assert!(
            prefill_res.unwrap_err().to_string().contains("not yet implemented"),
            "dispatch_prefill error should mention not-implemented"
        );

        // Transfer 4 KV blocks — also not implemented; must not silently ship mock data.
        let transfer_res = router.transfer_kv_cache(42, 4);
        assert!(transfer_res.is_err(), "transfer_kv_cache should fail loudly, not silently Ok");
        assert!(
            transfer_res.unwrap_err().to_string().contains("not yet implemented"),
            "transfer_kv_cache error should mention not-implemented"
        );

        // Dispatch decode carrying PoolAssignment context
        let assignment = PoolAssignment {
            source_prefill_pool_addr: "10.0.0.1:8000".to_string(),
            request_id: 42,
        };
        assert_eq!(assignment.request_id, 42);
        assert_eq!(assignment.source_prefill_pool_addr, "10.0.0.1:8000");
        let decode_res = router.dispatch_decode(42, 104, assignment);
        assert!(decode_res.is_err(), "dispatch_decode should fail loudly, not silently Ok");
        assert!(
            decode_res.unwrap_err().to_string().contains("not yet implemented"),
            "dispatch_decode error should mention not-implemented"
        );
    }

    #[test]
    fn test_transfer_kv_cache_rejects_zero_blocks() {
        let router = DisaggRouter::new("10.0.0.1:8000", "10.0.0.2:8000", PoolRole::Prefill);
        let result = router.transfer_kv_cache(42, 0);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("block count cannot be zero"),
            "error should mention zero block count: {}",
            err_msg
        );
    }

    #[test]
    fn test_rdma_toggle() {
        let mut router = DisaggRouter::new("10.0.0.1:8000", "10.0.0.2:8000", PoolRole::Prefill);
        assert!(!router.use_rdma);
        router.enable_rdma(true);
        assert!(router.use_rdma);
        router.enable_rdma(false);
        assert!(!router.use_rdma);
    }

    #[test]
    fn test_dispatch_decode_preserves_assignment() {
        let router = DisaggRouter::new("10.0.0.1:8000", "10.0.0.2:8000", PoolRole::Decode);
        let assignment = PoolAssignment {
            source_prefill_pool_addr: "10.0.0.1:8000".to_string(),
            request_id: 99,
        };
        // Verify assignment fields are accessible and correct
        assert_eq!(assignment.source_prefill_pool_addr, "10.0.0.1:8000");
        assert_eq!(assignment.request_id, 99);
        // Dispatch is a stub — should surface Unimplemented, not silently Ok.
        let decode_res = router.dispatch_decode(99, 200, assignment);
        assert!(decode_res.is_err());
        assert!(
            decode_res.unwrap_err().to_string().contains("not yet implemented"),
            "dispatch_decode should surface the not-implemented gap"
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
        let router = DisaggRouter::new("10.0.0.1:8000", "10.0.0.2:8000", PoolRole::Prefill);
        let batch = ReMPMigrationBatch { blocks: vec![], num_layers: 0, num_seq_chunks: 0 };
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
                    batch.blocks.push(KvBlock { data: vec![1.0; 4], layer_idx: layer, seq_chunk: chunk });
                }
            }
        }
        let result = router.transfer_kv_colocated(99, &batch);
        assert!(result.is_err(), "should fail with missing block");
    }
}

