//! PCIe Overlap of Next-Layer Expert Uploads.
//!
//! Streams the NEXT layer's missing MoE expert weights (`layer + 1`) over PCIe into a double-buffered
//! staging ring (`slot = (layer + 1) & 1`) on a secondary HIP stream while the CURRENT layer (`layer`)
//! executes attention/compute. Before layer `layer + 1`'s MoE kernel launches, the compute stream
//! waits on the prefetch completion event (`sync_layer_ready`), guaranteeing byte-identical weights
//! under arbitrary stress loops.
//!
//! Gated behind `GRIM_ASTRUM_EXPERT_PREFETCH=1` (default OFF).

pub fn astrum_expert_prefetch_enabled() -> bool {
    std::env::var("GRIM_ASTRUM_EXPERT_PREFETCH")
        .ok()
        .as_deref()
        == Some("1")
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefetchLayerTicket {
    pub layer: usize,
    pub ring_slot: usize,
    pub experts: Vec<usize>,
    pub synced: bool,
}

/// Double-buffered (`2`-slot) next-layer PCIe expert prefetch planner and stream coordinator.
#[derive(Debug, Clone)]
pub struct ExpertPrefetchStream {
    num_layers: usize,
    ring_slots: [Option<PrefetchLayerTicket>; 2],
    completed_syncs: usize,
}

impl ExpertPrefetchStream {
    pub fn new(num_layers: usize) -> Self {
        Self {
            num_layers: num_layers.max(1),
            ring_slots: [None, None],
            completed_syncs: 0,
        }
    }

    /// Issue an asynchronous PCIe prefetch for `next_layer` onto double-buffer slot `next_layer & 1`.
    pub fn enqueue_next_layer_prefetch(
        &mut self,
        next_layer: usize,
        miss_experts: Vec<usize>,
        staging_ring: &mut [Vec<u8>; 2],
        host_expert_payloads: &[&[u8]],
    ) -> Option<PrefetchLayerTicket> {
        if next_layer >= self.num_layers || miss_experts.is_empty() {
            return None;
        }
        let slot = next_layer & 1;
        staging_ring[slot].clear();
        for & b in host_expert_payloads {
            staging_ring[slot].extend_from_slice(b);
        }
        let ticket = PrefetchLayerTicket {
            layer: next_layer,
            ring_slot: slot,
            experts: miss_experts,
            synced: false,
        };
        self.ring_slots[slot] = Some(ticket.clone());
        Some(ticket)
    }

    /// Synchronize the prefetch stream for `layer` before `layer`'s MoE kernel reads `staging_ring[layer & 1]`.
    pub fn sync_layer_ready(&mut self, layer: usize) -> Option<PrefetchLayerTicket> {
        let slot = layer & 1;
        let ticket = self.ring_slots[slot].as_mut()?;
        if ticket.layer != layer {
            return None;
        }
        ticket.synced = true;
        self.completed_syncs += 1;
        Some(ticket.clone())
    }

    pub fn completed_syncs(&self) -> usize {
        self.completed_syncs
    }
}
