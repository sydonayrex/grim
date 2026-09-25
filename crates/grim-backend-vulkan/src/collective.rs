//! Multi-GPU communicator for Vulkan backends.
//! `VkCommunicator` provides the structural scaffolding for cross-GPU collectives (all-reduce, reduce-scatter, all-gather).

use grim_tensor::error::{Error, Result};

/// Multi-GPU communicator for Vulkan backends.
#[derive(Clone, Debug)]
/// Holds the rank/world_size topology. When `world_size == 1`, collectives degenerate to local accumulation.
pub struct VkCommunicator {
    pub world_size: usize,
    pub rank: usize,
}

/// Peer capability info for P2P transport (Task 2, Step 1).
#[derive(Clone, Debug)]
pub struct PeerInfo {
    pub local_idx: usize,
    pub remote_idx: usize,
    pub can_access: bool,
    pub can_coherent: bool,
}

impl VkCommunicator {
    pub fn new(world_size: usize, rank: usize) -> Result<Self> {
        if world_size == 0 {
            return Err(Error::Backend("world_size must be >= 1".into()));
        }
        if rank >= world_size {
            return Err(Error::Backend(format!(
                "rank ({}) must be < world_size ({})",
                rank, world_size
            )));
        }
        Ok(Self { world_size, rank })
    }

    /// Real P2P capability probe — structural scaffold (Task 2, Step 1).
    /// Returns peer pairs; `can_access` set false until `vkGetPhysicalDevicePeerMemoryFeatures` wired.
    pub fn probe_peer_capabilities(&self) -> Vec<PeerInfo> {
        let mut peers = Vec::new();
        for i in 0..self.world_size {
            for j in 0..self.world_size {
                if i == j {
                    continue;
                }
                peers.push(PeerInfo {
                    local_idx: i,
                    remote_idx: j,
                    can_access: false,
                    can_coherent: false,
                });
            }
        }
        peers
    }

    /// P2P copy method — real error propagation, no silent stub (Task 2, Step 2-4).
    /// Loud failure: returns `Err` with exact message naming what's missing (P2P transport / peer access).
    /// This matches the seeding-trap rule: fail loud, not silent zero-state.
    pub fn p2p_copy(
        &self,
        src_idx: usize,
        dst_idx: usize,
        _src_addr: u64,
        _dst_addr: u64,
        _size: usize,
    ) -> Result<()> {
        if src_idx >= self.world_size || dst_idx >= self.world_size {
            return Err(Error::Backend("P2P index out of range".into()));
        }
        if src_idx == dst_idx {
            return Err(Error::Backend("P2P src==dst".into()));
        }
        // P2P transport requires actual Vulkan device handles + external memory extensions.
        // Per plan: needs vkGetPhysicalDevicePeerMemoryFeatures + VK_KHR_external_memory_capabilities.
        // Loud error — never silent default; user must wire transport before multi-GPU works.
        Err(Error::Backend(
            "all_reduce_multi_gpu: P2P transport not yet wired (needs vkGetPhysicalDevicePeerMemoryFeatures + VK_KHR_external_memory_capabilities / VK_EXT_external_memory_host for host bounce fallback; see plan Task 2 Steps 1-4)".into(),
        ))
    }

    /// Accumulate inputs via summation. For `world_size == 1`, this is a local accumulation.
    pub fn all_reduce_sum(&self, inputs: &[Vec<f32>]) -> Vec<f32> {
        let n = inputs[0].len();
        let mut out = vec![0.0f32; n];
        for input in inputs {
            for i in 0..n {
                out[i] += input[i];
            }
        }
        out
    }
}
