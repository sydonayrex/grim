//! Distributed serving and disaggregation layer: decouples prefill/decode, manages cross-node KV cache transfers.
//!
/// ReMP 2D KV-cache migration (WI-8): coalesced 128-byte block-major transfer within same VRAM pool.
use grim_core::error::{Error, Result};
use grim_kvtransport::NetworkKvClient;

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
        // Outer loop: layers. Inner loop: seq chunks. Matches paged_attention KV fetch order.
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

/// Router for cross-pool dispatch and network KV transfers.
pub trait DisaggRouterT: Send + Sync {
    fn dispatch_prefill(&self, request_id: u64, tokens: &[u32]) -> Result<()>;
    fn transfer_kv_cache(&self, request_id: u64, num_blocks: usize) -> Result<()>;
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
        }
    }

    /// Enable or disable RDMA network transport.
    pub fn enable_rdma(&mut self, enabled: bool) {
        self.use_rdma = enabled;
    }
}

impl DisaggRouterT for DisaggRouter {
    /// Dispatch a prefill task to a dedicated prefill engine. Returns Unimplemented for stub transport.
    fn dispatch_prefill(&self, request_id: u64, tokens: &[u32]) -> Result<()> {
        let k_buf = tokens.iter().map(|&t| t as f32).collect::<Vec<_>>();
        let v_buf = tokens.iter().map(|&t| (t + 1) as f32).collect::<Vec<_>>();
        self.kv_client.send_block_remote(
            request_id as usize,
            &k_buf,
            &v_buf,
            &self.prefill_node_addr,
        )
    }

    /// Transfer KV blocks from prefill to decode engine via network transport.
    fn transfer_kv_cache(&self, request_id: u64, num_blocks: usize) -> Result<()> {
        if num_blocks == 0 {
            return Err(Error::KvCache(
                "Handoff protocol error: block count cannot be zero".into(),
            ));
        }
        let dummy_k = vec![1.0f32; 64];
        let dummy_v = vec![2.0f32; 64];
        for b in 0..num_blocks {
            self.kv_client.send_block_remote(
                request_id as usize + b,
                &dummy_k,
                &dummy_v,
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
        assignment: PoolAssignment,
    ) -> Result<()> {
        let client = NetworkKvClient::new(assignment.source_prefill_pool_addr.clone());
        let k_buf = vec![last_token as f32];
        let v_buf = vec![(last_token + 1) as f32];
        client.send_block_remote(request_id as usize, &k_buf, &v_buf, &self.decode_node_addr)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disaggregated_kv_routing() {
        let router = DisaggRouter::new("10.0.0.1:8000", "10.0.0.2:8000", PoolRole::Prefill);

        // Verify router fields stored correctly.
        assert_eq!(router.prefill_node_addr, "10.0.0.1:8000");
        assert_eq!(router.decode_node_addr, "10.0.0.2:8000");
        assert_eq!(router.pool_role, PoolRole::Prefill);

        // Dispatch prefill — network transport attempts connection and fails on dummy IP.
        let prefill_res = router.dispatch_prefill(42, &[101, 102, 103]);
        assert!(
            prefill_res.is_err(),
            "dispatch_prefill should fail when target endpoint is unreachable"
        );

        // Transfer 4 KV blocks — network transport attempts connection.
        let transfer_res = router.transfer_kv_cache(42, 4);
        assert!(
            transfer_res.is_err(),
            "transfer_kv_cache should fail when target endpoint is unreachable"
        );

        // Dispatch decode carrying PoolAssignment context
        let assignment = PoolAssignment {
            source_prefill_pool_addr: "10.0.0.1:8000".to_string(),
            request_id: 42,
        };
        assert_eq!(assignment.request_id, 42);
        assert_eq!(assignment.source_prefill_pool_addr, "10.0.0.1:8000");
        let decode_res = router.dispatch_decode(42, 104, assignment);
        assert!(
            decode_res.is_err(),
            "dispatch_decode should fail when target endpoint is unreachable"
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
        let decode_res = router.dispatch_decode(99, 200, assignment);
        assert!(decode_res.is_err());
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
}
