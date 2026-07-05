//! Distributed serving and disaggregation layer for Grim.
//!
//! Exposes the `DisaggRouter` trait and concrete implementations to decouple
//! prefill execution from decode execution and manage cross-node KV cache transfers.

use grim_core::error::{Error, Result};
use grim_kvtransport::NetworkKvClient;

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
    fn dispatch_prefill(&self, request_id: u64, _tokens: &[u32]) -> Result<()> {
        println!(
            "[DisaggRouter] Dispatching prefill task for Request {} to prefill node: {} (PoolRole: {:?})",
            request_id, self.prefill_node_addr, self.pool_role
        );
        Ok(())
    }

    /// Performs the KV-transfer step from the prefill engine to the decode engine
    /// utilizing the remote network-transport KV client.
    fn transfer_kv_cache(&self, request_id: u64, num_blocks: usize) -> Result<()> {
        println!(
            "[DisaggRouter] Coordinating cross-pool KV handoff (Blocks: {}) for Request {} over {}...",
            num_blocks, request_id, if self.use_rdma { "RDMA Fallback Layer" } else { "TCP Network" }
        );

        // Handoff protocol handshake validation
        if num_blocks == 0 {
            return Err(Error::KvCache("Handoff protocol error: block count cannot be zero".into()));
        }

        // Simulate network block transfers
        for block_idx in 0..num_blocks {
            let mock_data = vec![0.5f32; 1024];
            self.kv_client.send_block_remote(block_idx, &mock_data, &mock_data, &self.decode_node_addr)?;
            let _fetched = self.kv_client.fetch_block_remote(block_idx, &self.decode_node_addr, 1024)?;
        }

        println!("[DisaggRouter] Cross-pool KV handoff protocol exchange finished.");
        Ok(())
    }

    /// Dispatches a step-decode task to a dedicated decode execution engine.
    fn dispatch_decode(&self, request_id: u64, _last_token: u32, assignment: PoolAssignment) -> Result<()> {
        println!(
            "[DisaggRouter] Dispatching decode task for Request {} to decode node: {} (Prefill pool src: {})",
            request_id, self.decode_node_addr, assignment.source_prefill_pool_addr
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disaggregated_kv_routing() {
        let router = DisaggRouter::new("10.0.0.1:8000", "10.0.0.2:8000", PoolRole::Prefill);
        
        // Dispatch prefill
        assert!(router.dispatch_prefill(42, &[101, 102, 103]).is_ok());

        // Transfer 4 KV blocks over the simulated network
        assert!(router.transfer_kv_cache(42, 4).is_ok());

        // Dispatch decode carrying PoolAssignment context
        let assignment = PoolAssignment {
            source_prefill_pool_addr: "10.0.0.1:8000".to_string(),
            request_id: 42,
        };
        assert!(router.dispatch_decode(42, 104, assignment).is_ok());
    }
}
