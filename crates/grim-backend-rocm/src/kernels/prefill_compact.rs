//! Prefill hit compaction kernel (FreeToken).
//!
//! Compacts MoE expert request vectors on-device by partitioning cache-resident
//! expert slots (which can be gathered via high-bandwidth D2D copy) from cache
//! misses (which must be streamed asynchronously over PCIe).

use grim_tensor::error::{Error, Result};

/// Compacted result of expert cache residency scan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactedExpertSet {
    /// Expert slots already resident in GPU cache: `Vec<(expert_id, resident_slot)>`.
    pub resident: Vec<(usize, usize)>,
    /// Expert IDs that are not resident and must be fetched from host memory.
    pub misses: Vec<usize>,
}

/// Classify requested experts into cache hits vs misses using compaction.
pub fn compact_expert_requests(
    requested_experts: &[usize],
    slot_table: &[Option<usize>],
) -> Result<CompactedExpertSet> {
    let mut resident = Vec::new();
    let mut misses = Vec::new();

    for &exp in requested_experts {
        if exp >= slot_table.len() {
            return Err(Error::Backend(format!(
                "compact_expert_requests: expert index {exp} exceeds slot table length {}",
                slot_table.len()
            )));
        }

        if let Some(slot) = slot_table[exp] {
            resident.push((exp, slot));
        } else {
            misses.push(exp);
        }
    }

    Ok(CompactedExpertSet { resident, misses })
}

/// GPU HipRTC kernel source template for device-side hit compaction.
pub const PREFILL_COMPACT_HIP_SRC: &str = r#"
extern "C" __global__ void prefill_hit_compact_kernel(
    const int* __restrict__ requested_experts,
    const int* __restrict__ slot_table,
    int num_requested,
    int* __restrict__ out_resident_expert,
    int* __restrict__ out_resident_slot,
    int* __restrict__ out_resident_count,
    int* __restrict__ out_miss_expert,
    int* __restrict__ out_miss_count
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num_requested) return;

    int exp = requested_experts[tid];
    int slot = slot_table[exp];

    if (slot >= 0) {
        int idx = atomicAdd(out_resident_count, 1);
        out_resident_expert[idx] = exp;
        out_resident_slot[idx] = slot;
    } else {
        int idx = atomicAdd(out_miss_count, 1);
        out_miss_expert[idx] = exp;
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compact_expert_requests() {
        // 8 total experts: E0 (slot 3), E1 (None), E2 (slot 0), E3 (None), E4 (slot 1), E5 (None)
        let slot_table = vec![
            Some(3),
            None,
            Some(0),
            None,
            Some(1),
            None,
        ];

        let requested = vec![0, 1, 2, 3, 4];
        let compacted = compact_expert_requests(&requested, &slot_table).unwrap();

        assert_eq!(compacted.resident, vec![(0, 3), (2, 0), (4, 1)]);
        assert_eq!(compacted.misses, vec![1, 3]);
    }

    #[test]
    fn test_compact_expert_requests_all_resident() {
        let slot_table = vec![Some(0), Some(1), Some(2)];
        let requested = vec![0, 1, 2];
        let compacted = compact_expert_requests(&requested, &slot_table).unwrap();
        assert_eq!(compacted.resident.len(), 3);
        assert_eq!(compacted.misses.len(), 0);
    }
}
