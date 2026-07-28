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
}
