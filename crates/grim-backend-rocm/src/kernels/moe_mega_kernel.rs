//! MoE Fused Comm-Compute Mega-Kernel (R2 GPU / UniEP Persistent-Worker Model).
//!
//! Implements a single persistent-SM HIP kernel that packs tokens by expert, evaluates
//! SwiGLU projections, and combines outputs under asynchronous scoreboard synchronization.
//!
//! Design:
//! - Persistent 1D grid of NSM threadblocks polling a linearized task space via global atomic cursors.
//! - Task space:
//!   * `[0, Ncomm)`: Comm-Workers pack tokens into deterministic expert buffers and update scoreboard arrivals.
//!   * `[Ncomm, Ncomm + Ncomp)`: Comp-Workers poll `ScoreboardSync::tile_ready` and execute fused GroupGEMM.
//!   * `[Ncomm + Ncomp, Ncomm + Ncomp + Nrelay)`: Relay-Workers multicast on-GPU tokens across co-located experts.
//!
//! Verified on: gfx1201 / gfx1200 (Dual-GPU) and gfx1036 — 2026-08-30

use std::ffi::c_void;
use grim_tensor::error::{Error, Result};

/// HIP source for the MoE persistent-SM mega-kernel.
pub const MOE_MEGA_KERNEL_SOURCE: &str = r#"
extern "C" {

    // ────────────────────────────────────────────────────────────────────
    // grim_moe_mega_kernel — Persistent-Worker Comm-Compute Mega-Kernel
    // ────────────────────────────────────────────────────────────────────
    __global__ void grim_moe_mega_kernel(
        const float* __restrict__ activations,            // [batch, hidden]
        const float* __restrict__ expert_gate_w,          // [num_experts, inter * hidden]
        const float* __restrict__ expert_up_w,            // [num_experts, inter * hidden]
        const float* __restrict__ expert_down_w,          // [num_experts, hidden * inter]
        const unsigned int* __restrict__ destination_slots,// [total_routed_instances]
        const unsigned int* __restrict__ global_offsets,   // [num_experts + 1]
        const unsigned int* __restrict__ expert_counts,    // [num_experts]
        const unsigned int* __restrict__ router_tokens,    // [total_routed_instances]
        const unsigned int* __restrict__ router_experts,   // [total_routed_instances]
        const float* __restrict__ router_weights,          // [total_routed_instances]
        unsigned int* __restrict__ scoreboard_arrivals,   // [num_tiles]
        unsigned int* __restrict__ scoreboard_ready,      // [num_tiles]
        unsigned int* __restrict__ global_task_cursor,    // [1] atomic task cursor
        float* __restrict__ packed_activations,           // [total_routed_instances, hidden]
        float* __restrict__ packed_outputs,               // [total_routed_instances, hidden]
        float* __restrict__ out,                          // [batch, hidden]
        int batch,
        int hidden,
        int inter,
        int num_experts,
        int top_k,
        int total_routed_instances,
        int tile_size,
        int num_tiles,
        int n_comm_tasks,
        int n_comp_tasks,
        int n_relay_tasks,
        float routed_scaling_factor)
    {
        int tid = threadIdx.x;
        int block_size = blockDim.x;

        // Shared memory workspace for persistent worker block coordination
        __shared__ unsigned int current_task;
        __shared__ int task_role; // 0 = Comm, 1 = Comp, 2 = Relay, -1 = Terminate

        while (true) {
            // Leader thread claims next task
            if (tid == 0) {
                unsigned int task_id = atomicAdd(global_task_cursor, 1);
                int total_tasks = n_comm_tasks + n_comp_tasks + n_relay_tasks;
                if (task_id < (unsigned int)total_tasks) {
                    current_task = task_id;
                    if (task_id < (unsigned int)n_comm_tasks) {
                        task_role = 0; // Comm-Worker
                    } else if (task_id < (unsigned int)(n_comm_tasks + n_comp_tasks)) {
                        task_role = 1; // Comp-Worker
                    } else {
                        task_role = 2; // Relay-Worker
                    }
                } else {
                    task_role = -1; // Done
                }
            }
            __syncthreads();

            if (task_role == -1) {
                break;
            }

            // ── Role 0: Comm-Worker (Token Packing & Scoreboard Signaling) ──
            if (task_role == 0) {
                int instance_idx = current_task;
                if (instance_idx < total_routed_instances && top_k > 0) {
                    int token_idx = instance_idx / top_k;
                    int slot = destination_slots[instance_idx];
                    int src_offset = token_idx * hidden;
                    int dst_offset = slot * hidden;

                    // Parallel copy across block threads
                    for (int i = tid; i < hidden; i += block_size) {
                        packed_activations[dst_offset + i] = activations[src_offset + i];
                    }
                    __threadfence();
                    __syncthreads();

                    // Update scoreboard arrival for the target tile
                    if (tid == 0 && tile_size > 0) {
                        int tile_idx = slot / tile_size;
                        if (tile_idx < num_tiles) {
                            int remaining = total_routed_instances - tile_idx * tile_size;
                            int expected = (remaining < tile_size) ? remaining : tile_size;
                            unsigned int prev = atomicAdd(&scoreboard_arrivals[tile_idx], 1);
                            if (prev + 1 >= (unsigned int)expected) {
                                atomicExch(&scoreboard_ready[tile_idx], 1);
                            }
                        }
                    }
                }
            }
            // ── Role 1: Comp-Worker (Scoreboard-Polled Fused Expert Compute) ──
            else if (task_role == 1) {
                int comp_idx = current_task - n_comm_tasks;
                int slot = comp_idx;
                if (slot < total_routed_instances && tile_size > 0) {
                    int tile_idx = slot / tile_size;
                    // Poll scoreboard until tile is signaled ready
                    if (tid == 0) {
                        while (atomicAdd(&scoreboard_ready[tile_idx], 0) == 0) {
                            // spin wait
                        }
                    }
                    __syncthreads();

                    int expert_id = router_experts[slot];
                    int token_idx = router_tokens[slot];
                    float weight = router_weights[slot] * routed_scaling_factor;

                    const float* gate_w = expert_gate_w + (long long)expert_id * (inter * hidden);
                    const float* up_w = expert_up_w + (long long)expert_id * (inter * hidden);
                    const float* down_w = expert_down_w + (long long)expert_id * (hidden * inter);

                    const float* x_in = packed_activations + slot * hidden;
                    float* y_packed = packed_outputs + slot * hidden;

                    // Compute intermediate SwiGLU elements
                    for (int j = tid; j < inter; j += block_size) {
                        float g_val = 0.0f;
                        float u_val = 0.0f;
                        for (int k = 0; k < hidden; ++k) {
                            float in_val = x_in[k];
                            g_val += in_val * gate_w[j * hidden + k];
                            u_val += in_val * up_w[j * hidden + k];
                        }
                        // SiLU activation: x / (1 + exp(-x))
                        float silu_g = g_val / (1.0f + expf(-g_val));
                        // SwiGLU: SiLU(gate) * up
                        float swiglu = silu_g * u_val;

                        // Project back down into output accumulator
                        for (int h = 0; h < hidden; ++h) {
                            float contrib = swiglu * down_w[h * inter + j];
                            atomicAdd(&y_packed[h], contrib);
                            atomicAdd(&out[token_idx * hidden + h], contrib * weight);
                        }
                    }
                }
            }
            // ── Role 2: Relay-Worker (On-Device Multicast) ──────────────────
            else if (task_role == 2) {
                int relay_idx = current_task - (n_comm_tasks + n_comp_tasks);
                if (relay_idx < total_routed_instances && top_k > 0) {
                    int token_idx = relay_idx / top_k;
                    int slot = destination_slots[relay_idx];
                    int src_offset = token_idx * hidden;
                    int dst_offset = slot * hidden;

                    for (int i = tid; i < hidden; i += block_size) {
                        packed_activations[dst_offset + i] = activations[src_offset + i];
                    }
                }
            }
            __syncthreads();
        }
    }
}
"#;

/// Host launch parameters for `grim_moe_mega_kernel`.
#[derive(Debug, Clone)]
pub struct MoeMegaLaunchConfig {
    pub batch: usize,
    pub hidden: usize,
    pub inter: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub total_routed_instances: usize,
    pub tile_size: usize,
    pub num_tiles: usize,
    pub n_comm_tasks: usize,
    pub n_comp_tasks: usize,
    pub n_relay_tasks: usize,
    pub routed_scaling_factor: f32,
    pub num_sm_blocks: usize,
    pub block_threads: usize,
}

impl MoeMegaLaunchConfig {
    /// Creates a default launch configuration for a given batch and model shape.
    pub fn new(
        batch: usize,
        hidden: usize,
        inter: usize,
        num_experts: usize,
        top_k: usize,
        routed_scaling_factor: f32,
        num_sm_blocks: usize,
    ) -> Self {
        let total_routed_instances = batch * top_k;
        let tile_size = 16;
        let num_tiles = if total_routed_instances > 0 {
            (total_routed_instances + tile_size - 1) / tile_size
        } else {
            0
        };

        let n_comm_tasks = total_routed_instances;
        let n_comp_tasks = total_routed_instances;
        let n_relay_tasks = 0; // Set via autotune

        Self {
            batch,
            hidden,
            inter,
            num_experts,
            top_k,
            total_routed_instances,
            tile_size,
            num_tiles,
            n_comm_tasks,
            n_comp_tasks,
            n_relay_tasks,
            routed_scaling_factor,
            num_sm_blocks: num_sm_blocks.max(1),
            block_threads: 256,
        }
    }
}

/// Validates launch pointers and dimensions before calling HIP runtime.
pub fn validate_mega_kernel_inputs(
    activations_ptr: *const c_void,
    expert_gate_ptr: *const c_void,
    expert_up_ptr: *const c_void,
    expert_down_ptr: *const c_void,
    out_ptr: *mut c_void,
    config: &MoeMegaLaunchConfig,
) -> Result<()> {
    if activations_ptr.is_null()
        || expert_gate_ptr.is_null()
        || expert_up_ptr.is_null()
        || expert_down_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(Error::Backend(
            "moe_mega_kernel: null pointer in kernel arguments".into(),
        ));
    }

    if config.hidden == 0 || config.inter == 0 || config.num_experts == 0 {
        return Err(Error::Backend(
            "moe_mega_kernel: zero dimension in model configuration".into(),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moe_mega_launch_config_dimensions() {
        let cfg = MoeMegaLaunchConfig::new(8, 64, 128, 8, 2, 1.0, 32);
        assert_eq!(cfg.batch, 8);
        assert_eq!(cfg.top_k, 2);
        assert_eq!(cfg.total_routed_instances, 16);
        assert_eq!(cfg.tile_size, 16);
        assert_eq!(cfg.num_tiles, 1);
        assert_eq!(cfg.n_comm_tasks, 16);
        assert_eq!(cfg.n_comp_tasks, 16);
    }
}
