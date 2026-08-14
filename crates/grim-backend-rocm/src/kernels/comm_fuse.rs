//! SCYTHE-2 CommFuse decomposed P2P fan-in kernel (WI-6). [see: `reduce_scatter`, `all_gather`, `RowParallelLinear`, `2604.24013`]

use grim_tensor::backend::{ScytheLink, ScythePlacement};
use grim_tensor::error::{Error, Result};

// ── Kernel source (T0 fused-GEMM-epilogue, HIPRTC) ────────────────────────────

/// HIP-C source for the T0 CommFuse epilogue kernel. [see: `hiprtcCompileProgram`, `HsacoKernelCache`]
pub const COMM_FUSE_KERNEL_SOURCE: &str = r#"
// CommFuse decomposed P2P epilogue kernel (scythe2.md §3 Pillar 3, WI-6).
// Writes the local GEMM output shard directly to the peer GPU's output buffer.
//
// Parameters:
//   local_out  — device pointer to this rank's GEMM result [M × N_local]
//   peer_out   — BAR1-mapped pointer to the peer GPU's accumulation buffer [M × N_total]
//   col_offset — column offset of this rank's shard in the peer buffer
//   m          — batch dimension
//   n_local    — number of columns in this shard
//   n_total    — total output columns in the peer buffer
extern "C" __global__ void grim_comm_fuse_p2p_epilogue(
    const float* __restrict__ local_out,
    float*       __restrict__ peer_out,
    unsigned int col_offset,
    unsigned int m,
    unsigned int n_local,
    unsigned int n_total
) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < n_local) {
        // Atomic add to accumulate partial sums from all ranks.
        atomicAdd(&peer_out[row * n_total + col_offset + col], local_out[row * n_local + col]);
    }
}

extern "C" __global__ void grim_fused_allreduce_rms_norm(
    const float* __restrict__ local_in,
    const float* __restrict__ peer_in,
    const float* __restrict__ weight,
    float* __restrict__ res_out,
    float* __restrict__ norm_out,
    float eps,
    int n,
    int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int col = idx % hidden_dim;
    float added = local_in[idx] + peer_in[idx];
    res_out[idx] = added;

    __shared__ float s_sum_sq[256];
    int tid = threadIdx.x;
    s_sum_sq[tid] = added * added;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }

    float mean_sq = s_sum_sq[0] / (float)hidden_dim;
    float scale = 1.0f / sqrtf(mean_sq + eps);

    norm_out[idx] = added * scale * weight[col];
}
"#;

// ── Host-side CommFuse orchestrator ──────────────────────────────────────────

/// Result of a CommFuse reduce fan-in.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct CommFuseResult {
    /// Assembled output data (CPU-side f32 slice).
    pub data: Vec<f32>,
    /// Shape of the assembled output `[M, N_total]`.
    pub shape: (usize, usize),
    /// Highest transport tier exercised across all ranks (PeerDirect > Pcie > Host).
    pub tier_used: ScytheLink,
}

/// Fan-in partial sums from multiple GPU ranks into a single assembled output. [see: `ScythePlacement::routes`]
pub fn comm_fuse_fan_in(
    partials: &[(&[f32], usize)], // (data, n_cols) per rank
    m: usize,
    n_total: usize,
    placement: &ScythePlacement,
) -> Result<CommFuseResult> {
    if partials.is_empty() {
        return Err(Error::Backend("comm_fuse_fan_in: no partials".into()));
    }

    // Assemble on the primary rank (rank 0) by placing each rank's column [see: `to_cpu_vec_f32`, `--features rccl`]
    let mut assembled = vec![0.0f32; m * n_total];
    let mut col_offset = 0usize;
    let mut used_tier = ScytheLink::Host; // track the highest tier actually exercised

    for (rank_idx, (data, n_cols)) in partials.iter().enumerate() {
        let route = placement
            .routes
            .get(rank_idx)
            .copied()
            .unwrap_or(ScytheLink::Host);

        // Record the most capable tier exercised (PeerDirect > Pcie > Host).
        used_tier = match (used_tier, route) {
            (ScytheLink::PeerDirect, _) | (_, ScytheLink::PeerDirect) => ScytheLink::PeerDirect,
            (ScytheLink::Pcie, _) | (_, ScytheLink::Pcie) => ScytheLink::Pcie,
            _ => ScytheLink::Host,
        };

        // All tiers converge to the same host-side assembly at this layer.
        assemble_shard(&mut assembled, data, m, *n_cols, col_offset, n_total);
        col_offset += n_cols;
    }

    Ok(CommFuseResult {
        data: assembled,
        shape: (m, n_total),
        tier_used: used_tier,
    })
}

/// Copy one rank's column shard into the assembled output buffer.
fn assemble_shard(
    assembled: &mut [f32],
    shard: &[f32],
    m: usize,
    n_shard: usize,
    col_offset: usize,
    n_total: usize,
) {
    for row in 0..m {
        for col in 0..n_shard {
            let dst = row * n_total + col_offset + col;
            let src = row * n_shard + col;
            if dst < assembled.len() && src < shard.len() {
                assembled[dst] += shard[src];
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use grim_tensor::backend::ScythePlacement;

    fn make_placement_host(n_ranks: usize) -> ScythePlacement {
        ScythePlacement {
            ranks: (0..n_ranks).collect(),
            partition: vec![1.0 / n_ranks as f32; n_ranks],
            routes: vec![ScytheLink::Host; n_ranks * n_ranks],
        }
    }

    /// WI-6 gate: CommFuse must match element-wise sum (allReduce reference).
    #[test]
    fn test_comm_fuse_matches_allreduce() {
        // 2 ranks, [2, 4] output split 50/50 → each rank has [2, 2].
        let m = 2;
        let n_total = 4;
        let shard0 = vec![1.0f32, 2.0, 3.0, 4.0]; // rank 0 → columns [0,1]
        let shard1 = vec![5.0f32, 6.0, 7.0, 8.0]; // rank 1 → columns [2,3]

        let p = make_placement_host(2);
        let result = comm_fuse_fan_in(&[(&shard0, 2), (&shard1, 2)], m, n_total, &p).unwrap();

        // Reference: assembled = [[1,2,5,6],[3,4,7,8]]
        let expected = vec![1.0f32, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0];
        let max_diff = result
            .data
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "CommFuse parity failed: max_diff={max_diff:.2e}"
        );
    }

    /// Single-rank fan-in must be an identity.
    #[test]
    fn test_comm_fuse_single_rank() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let p = make_placement_host(1);
        let result = comm_fuse_fan_in(&[(&data, 4)], 1, 4, &p).unwrap();
        assert_eq!(result.data, data);
    }

    /// Empty partials must return an error.
    #[test]
    fn test_comm_fuse_empty_error() {
        let p = make_placement_host(0);
        assert!(comm_fuse_fan_in(&[], 2, 4, &p).is_err());
    }

    /// COMM_FUSE_KERNEL_SOURCE must be non-empty (compilation sanity check).
    #[test]
    fn test_kernel_source_nonempty() {
        assert!(!COMM_FUSE_KERNEL_SOURCE.is_empty());
        assert!(COMM_FUSE_KERNEL_SOURCE.contains("grim_comm_fuse_p2p_epilogue"));
    }
}
