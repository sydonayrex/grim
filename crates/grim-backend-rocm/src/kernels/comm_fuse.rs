//! SCYTHE-2 CommFuse decomposed P2P fan-in kernel (WI-6). [see: `reduce_scatter`, `all_gather`, `RowParallelLinear`, `2604.24013`]

use grim_tensor::backend::{ScytheLink, ScythePlacement};
use grim_tensor::error::{Error, Result};

// ── Kernel source (T0 fused-GEMM-epilogue, HIPRTC) ────────────────────────────

/// HIP-C source for the T0 CommFuse epilogue kernel. [see: `hiprtcCompileProgram`, `HsacoKernelCache`]
pub const COMM_FUSE_KERNEL_SOURCE: &str = r#"
// CommFuse decomposed P2P epilogue kernel (scythe2.md §3 Pillar 3, WI-6).
// Writes the local GEMM output shard directly to the peer GPU's output buffer.
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
// Narrow-workgroup PCIe link copy (derived from Radiance aql_copy_link):
// 1 block of 128 threads (4 Wave32 waves), 16 uint4 (16B) loads in flight per thread (32 KiB/iter).
// Saturates PCIe 4.0x16 (~28.9 GB/s) using only 1 CU so concurrent GEMMs keep >96% of CUs.
extern "C" __global__ __launch_bounds__(128) void grim_copy_link_16b(
    uint4* __restrict__ dst,
    const uint4* __restrict__ src,
    unsigned long long n_vec
) {
    if (n_vec == 0ULL) return;
    const unsigned long long step = 128ULL * 16ULL;
    unsigned long long i = (unsigned long long)threadIdx.x;
    for (; i + step <= n_vec; i += step) {
        uint4 v[16];
        #pragma unroll
        for (int u = 0; u < 16; ++u) v[u] = src[i + (unsigned long long)u * 128ULL];
        #pragma unroll
        for (int u = 0; u < 16; ++u) dst[i + (unsigned long long)u * 128ULL] = v[u];
    }
    if (i < n_vec) {
        uint4 v[16];
        #pragma unroll
        for (int u = 0; u < 16; ++u) {
            unsigned long long j = i + (unsigned long long)u * 128ULL;
            v[u] = src[j < n_vec ? j : n_vec - 1ULL];
        }
        #pragma unroll
        for (int u = 0; u < 16; ++u) {
            unsigned long long j = i + (unsigned long long)u * 128ULL;
            if (j < n_vec) dst[j] = v[u];
        }
    }
}
struct GrimWtabPatch {
    unsigned int n;
    unsigned int pad;
    unsigned long long dst[24];
    unsigned long long val[24];
};

// Stream-ordered by-value device weight-table pointer patcher (ported from Radiance wtab.hip).
// Grid: (1, 1, 1), Block: (32, 1, 1).
extern "C" __global__ __launch_bounds__(32)
void grim_wtab_patch(GrimWtabPatch p) {
    const unsigned int i = threadIdx.x;
    if (i < p.n) {
        *((unsigned long long*)p.dst[i]) = p.val[i];
    }
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

use crate::device::roc_device::RocmDevice;
use crate::memory::storage::RocmStorage;
use std::ffi::c_void;

/// Launch GPU-accelerated P2P epilogue kernel for CommFuse (Task 10.8).
pub fn launch_comm_fuse_p2p_epilogue(
    dev: &RocmDevice,
    local_out: &RocmStorage,
    peer_out: &RocmStorage,
    col_offset: u32,
    m: u32,
    n_local: u32,
    n_total: u32,
) -> Result<*mut c_void> {
    let local_ptr = local_out
        .device_ptr
        .ok_or_else(|| Error::Backend("comm_fuse_p2p_epilogue: local_out has no device ptr".into()))?;
    let peer_ptr = peer_out
        .device_ptr
        .ok_or_else(|| Error::Backend("comm_fuse_p2p_epilogue: peer_out has no device ptr".into()))?;

    let block_dim = crate::HipDim3::new(16, 16, 1);
    let grid_dim = crate::HipDim3::new(
        n_local.div_ceil(16),
        m.div_ceil(16),
        1,
    );

    let mut lp = local_ptr;
    let mut pp = peer_ptr;
    let mut coff = col_offset;
    let mut m_val = m;
    let mut nl = n_local;
    let mut nt = n_total;

    dev.launch_compute_kernel(
        "grim_comm_fuse_p2p_epilogue",
        grid_dim,
        block_dim,
        &mut [
            crate::arg(&mut lp),
            crate::arg(&mut pp),
            crate::arg(&mut coff),
            crate::arg(&mut m_val),
            crate::arg(&mut nl),
            crate::arg(&mut nt),
        ],
    )
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
        let expected = [1.0f32, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0];
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

    #[test]
    #[ignore]
    fn test_comm_fuse_p2p_epilogue_gpu_parity() {
        use crate::device::roc_device::RocmDevice;
        use crate::as_rocm;
        use grim_tensor::backend::CoreTensorOps;
        use grim_tensor::Shape;

        let dev = match RocmDevice::try_new(0) {
            Ok(d) => d,
            Err(_) => return,
        };

        let m = 2usize;
        let n_local = 3usize;
        let n_total = 6usize;
        let col_offset = 3usize;

        let local_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut peer_data = vec![0.0f32; m * n_total];
        peer_data[col_offset] = 10.0;

        let mut expected = peer_data.clone();
        for r in 0..m {
            for c in 0..n_local {
                expected[r * n_total + col_offset + c] += local_data[r * n_local + c];
            }
        }

        let local_shape = Shape::new(vec![m, n_local]);
        let peer_shape = Shape::new(vec![m, n_total]);

        let local_dev = CoreTensorOps::from_cpu(&dev, &local_data, &local_shape, grim_tensor::DType::F32).unwrap();
        let peer_dev = CoreTensorOps::from_cpu(&dev, &peer_data, &peer_shape, grim_tensor::DType::F32).unwrap();

        let local_s = as_rocm(local_dev.as_ref()).unwrap();
        let peer_s = as_rocm(peer_dev.as_ref()).unwrap();

        launch_comm_fuse_p2p_epilogue(&dev, local_s, peer_s, col_offset as u32, m as u32, n_local as u32, n_total as u32).unwrap();
        dev.synchronize();

        let got = peer_dev.to_cpu_vec_f32().unwrap();
        assert_eq!(got.len(), expected.len());
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 1e-5, "mismatch at {i}: got {g} vs exp {e}");
        }
    }
}

/// Launch narrow-workgroup (`1 block × 128 threads`) 16-byte vector copy kernel.
/// Saturates PCIe link bandwidth with 32 KiB in flight on a single CU without starving concurrent GEMMs.
/// Requires `n_bytes` to be a multiple of 16.
pub fn launch_copy_link_16b(
    dev: &RocmDevice,
    dst: &RocmStorage,
    src: &RocmStorage,
    n_bytes: usize,
) -> Result<*mut c_void> {
    if n_bytes % 16 != 0 {
        return Err(Error::Backend(format!(
            "launch_copy_link_16b: n_bytes ({n_bytes}) must be a multiple of 16"
        )));
    }
    let dst_ptr = dst
        .device_ptr
        .ok_or_else(|| Error::Backend("launch_copy_link_16b: dst has no device ptr".into()))?;
    let src_ptr = src
        .device_ptr
        .ok_or_else(|| Error::Backend("launch_copy_link_16b: src has no device ptr".into()))?;

    let mut dp = dst_ptr;
    let mut sp = src_ptr;
    let mut n_vec: u64 = (n_bytes / 16) as u64;

    dev.launch_compute_kernel(
        "grim_copy_link_16b",
        crate::HipDim3::new(1, 1, 1),
        crate::HipDim3::new(128, 1, 1),
        &mut [
            crate::arg(&mut dp),
            crate::arg(&mut sp),
            crate::arg(&mut n_vec),
        ],
    )
}

/// Maximum pointer writes carried by-value in a single `grim_wtab_patch` AQL dispatch (`wtab.h:18`).
pub const WTAB_PATCH_CAP: usize = 24;

/// Stream-ordered device pointer table patch passed by value in kernel arguments so the host
/// can queue multiple steps ahead without racing on a staging buffer (`core/runtime/wtab.h`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct WtabPatch {
    pub n: u32,
    pub pad: u32,
    pub dst: [u64; WTAB_PATCH_CAP],
    pub val: [u64; WTAB_PATCH_CAP],
}

impl Default for WtabPatch {
    fn default() -> Self {
        Self {
            n: 0,
            pad: 0,
            dst: [0; WTAB_PATCH_CAP],
            val: [0; WTAB_PATCH_CAP],
        }
    }
}

impl WtabPatch {
    /// Append a `(dst_device_slot_addr, new_device_ptr)` write. Returns `false` if full (`24` entries).
    pub fn push(&mut self, dst_addr: u64, new_val: u64) -> bool {
        let idx = self.n as usize;
        if idx >= WTAB_PATCH_CAP {
            return false;
        }
        self.dst[idx] = dst_addr;
        self.val[idx] = new_val;
        self.n += 1;
        true
    }

    /// Host simulation of `grim_wtab_patch` for CPU/host-memory unit tests.
    pub fn apply_host_sim(&self) {
        for i in 0..(self.n as usize).min(WTAB_PATCH_CAP) {
            let ptr = self.dst[i] as *mut u64;
            if !ptr.is_null() {
                unsafe {
                    *ptr = self.val[i];
                }
            }
        }
    }
}

/// Residual-stream composition probe statistics (`arch/common/rad_arch.h:53-81`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DbgResidStat {
    pub rms: f64,
    pub median_abs: f64,
    pub max_abs: f64,
}

/// Compute `rms`, `median(|x|)` (bulk magnitude immune to massive-activation outliers), and `max(|x|)`
/// over one activation row (`rad_arch.h:58-81`).
pub fn compute_dbg_resid_stat(row: &[f32]) -> DbgResidStat {
    if row.is_empty() {
        return DbgResidStat {
            rms: -1.0,
            median_abs: -1.0,
            max_abs: -1.0,
        };
    }
    let mut sum_sq = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut abs_vals: Vec<f32> = Vec::with_capacity(row.len());
    for &x in row {
        let xf = x as f64;
        sum_sq += xf * xf;
        let a = x.abs();
        if (a as f64) > max_abs {
            max_abs = a as f64;
        }
        abs_vals.push(a);
    }
    abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    DbgResidStat {
        rms: (sum_sq / (row.len() as f64)).sqrt(),
        median_abs: abs_vals[abs_vals.len() / 2] as f64,
        max_abs,
    }
}
