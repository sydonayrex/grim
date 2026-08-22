//! Full-layer double-buffered MoE prefill transfer pipelining.
//!
//! During prompt prefill, thousands of tokens per layer activate almost the complete expert set.
//! On-demand loading serializes expert transfers against GPU compute, exposing large PCIe latency stalls.
//!
//! This pipeline maintains two full-layer weight buffers in GPU cache (`BufferA` and `BufferB`).
//! While the GPU compute stream evaluates layer $l$ from one buffer, a dedicated asynchronous
//! transfer stream prefetches the complete expert set of layer $l+1$ into the alternate buffer.
//! Once both finish, buffers swap roles, making prefill execution bandwidth-bound rather than latency-bound.

use grim_tensor::error::Result;

/// State of a double-buffer slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferRole {
    /// Actively being read by GPU compute kernel for layer $l$.
    ComputeActive,
    /// Actively receiving PCIe DMA transfer for layer $l+1$.
    TransferActive,
    /// Idle / ready for next stage.
    Ready,
}

/// Double-buffered prefill pipeline coordinator.
#[derive(Debug)]
pub struct MoePrefillPipeline {
    /// Number of transformer layers in the model.
    pub total_layers: usize,
    /// Total routed experts per MoE layer.
    pub num_experts: usize,
    /// Size in bytes of one complete layer's expert weights.
    pub layer_bytes: usize,
    /// Current layer index undergoing computation.
    pub current_compute_layer: usize,
    /// Role state of Buffer A.
    pub buffer_a_role: BufferRole,
    /// Role state of Buffer B.
    pub buffer_b_role: BufferRole,
    /// Whether double-buffering is active or fell back to single-buffer on-demand mode.
    pub double_buffering_enabled: bool,
}

impl MoePrefillPipeline {
    /// Create a new prefill pipeline for a model.
    ///
    /// # Contract
    /// `total_layers` and `num_experts` must be > 0.
    /// If `available_vram_bytes < 2 * layer_bytes`, falls back to single-buffer mode to prevent OOM.
    pub fn new(
        total_layers: usize,
        num_experts: usize,
        layer_bytes: usize,
        available_vram_bytes: usize,
    ) -> Self {
        assert!(total_layers > 0, "total_layers must be > 0");
        assert!(num_experts > 0, "num_experts must be > 0");

        let can_double_buffer = available_vram_bytes >= 2 * layer_bytes;

        Self {
            total_layers,
            num_experts,
            layer_bytes,
            current_compute_layer: 0,
            buffer_a_role: BufferRole::Ready,
            buffer_b_role: BufferRole::Ready,
            double_buffering_enabled: can_double_buffer,
        }
    }

    /// Prime the pipeline by initiating prefetch of Layer 0 (and Layer 1 if double-buffering).
    ///
    /// # Contract
    /// Returns the target layer index to kick off on the transfer stream.
    pub fn prime(&mut self) -> Result<Vec<usize>> {
        self.current_compute_layer = 0;
        if self.total_layers == 0 {
            return Ok(Vec::new());
        }

        if self.double_buffering_enabled && self.total_layers > 1 {
            self.buffer_a_role = BufferRole::ComputeActive; // Layer 0
            self.buffer_b_role = BufferRole::TransferActive; // Layer 1
            Ok(vec![0, 1])
        } else {
            self.buffer_a_role = BufferRole::ComputeActive; // Layer 0
            Ok(vec![0])
        }
    }

    /// Advance the pipeline after completing computation for `current_compute_layer`.
    ///
    /// # Contract
    /// Swaps buffer roles and returns `Some(next_layer_to_prefetch)` if more layers remain,
    /// or `None` when prefetch is complete.
    pub fn step_and_swap(&mut self) -> Result<Option<usize>> {
        if self.current_compute_layer >= self.total_layers {
            return Ok(None);
        }

        self.current_compute_layer += 1;

        if !self.double_buffering_enabled {
            // Single-buffer fallback: next layer transfers on-demand
            if self.current_compute_layer < self.total_layers {
                return Ok(Some(self.current_compute_layer));
            }
            return Ok(None);
        }

        // Swap buffer roles
        std::mem::swap(&mut self.buffer_a_role, &mut self.buffer_b_role);

        // Layer to prefetch in background is current_compute + 1
        let prefetch_layer = self.current_compute_layer + 1;
        if prefetch_layer < self.total_layers {
            Ok(Some(prefetch_layer))
        } else {
            Ok(None)
        }
    }

    /// Execute a full-model prefill pass with pipelined DMA transfers overlapping GPU compute.
    ///
    /// # Contract
    /// `compute_fn(layer_idx, buffer_idx)` runs GPU kernel on the compute stream.
    /// `dma_fn(layer_idx, buffer_idx)` initiates asynchronous DMA on the transfer stream.
    pub fn execute_pipelined<C, D>(&mut self, mut compute_fn: C, mut dma_fn: D) -> Result<()>
    where
        C: FnMut(usize, usize) -> Result<()>,
        D: FnMut(usize, usize) -> Result<()>,
    {
        let primed = self.prime()?;
        if primed.is_empty() {
            return Ok(());
        }

        // Initial transfers
        dma_fn(primed[0], 0)?;
        if primed.len() > 1 {
            dma_fn(primed[1], 1)?;
        }

        for l in 0..self.total_layers {
            let active_buf = if l % 2 == 0 { 0 } else { 1 };
            let alt_buf = 1 - active_buf;

            // Trigger background transfer for next layer if available
            if let Some(next_l) = self.step_and_swap()? {
                dma_fn(next_l, alt_buf)?;
            }

            // Execute compute on current layer buffer
            compute_fn(l, active_buf)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefill_pipeline_double_buffering_lifecycle() {
        let total_layers = 4;
        let num_experts = 64;
        let layer_bytes = 100_000_000;
        let vram_bytes = 300_000_000; // Fits 2 full layers

        let mut pipeline =
            MoePrefillPipeline::new(total_layers, num_experts, layer_bytes, vram_bytes);
        assert!(pipeline.double_buffering_enabled);

        // Prime: loads Layer 0 & Layer 1
        let primed = pipeline.prime().unwrap();
        assert_eq!(primed, vec![0, 1]);
        assert_eq!(pipeline.buffer_a_role, BufferRole::ComputeActive);
        assert_eq!(pipeline.buffer_b_role, BufferRole::TransferActive);

        // Step 0 -> Layer 1 computes, Layer 2 prefetches
        let next = pipeline.step_and_swap().unwrap();
        assert_eq!(next, Some(2));
        assert_eq!(pipeline.buffer_a_role, BufferRole::TransferActive);
        assert_eq!(pipeline.buffer_b_role, BufferRole::ComputeActive);

        // Step 1 -> Layer 2 computes, Layer 3 prefetches
        let next = pipeline.step_and_swap().unwrap();
        assert_eq!(next, Some(3));

        // Step 2 -> Layer 3 computes, no more layers to prefetch
        let next = pipeline.step_and_swap().unwrap();
        assert_eq!(next, None);
    }

    #[test]
    fn test_prefill_pipeline_low_vram_fallback() {
        let total_layers = 4;
        let num_experts = 64;
        let layer_bytes = 100_000_000;
        let vram_bytes = 150_000_000; // Cannot fit 2 full layers

        let mut pipeline =
            MoePrefillPipeline::new(total_layers, num_experts, layer_bytes, vram_bytes);
        assert!(!pipeline.double_buffering_enabled);

        let primed = pipeline.prime().unwrap();
        assert_eq!(primed, vec![0]);

        let next = pipeline.step_and_swap().unwrap();
        assert_eq!(next, Some(1));
    }
}
