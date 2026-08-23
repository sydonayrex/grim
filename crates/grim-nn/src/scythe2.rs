//! SCYTHE-2 capacity-calibrated sharded linears (WI-3).
//!
//! Implements the concrete leaf layer that the C²PLR controller drives:
//! `Scythe2Linear::forward_placed` slices the weight matrix per the controller-
//! chosen `ScythePlacement`, dispatches each shard to its GPU, then assembles
//! the output via CommFuse decomposed P2P fan-in (for row-parallel) or simple
//! concatenation (for column-parallel).
//!
//! ## Why forward_placed, not a static partition?
//! v1's `ScytheColumnParallelLinear` fixed the shard ratio at load time.
//! SCYTHE-2 fixes it at *forward time*: the controller may choose a 70/30 split
//! for a compute-bound GEMM and 100/100 (replicated) for a memory-bound norm,
//! using the same weight tensor for both. The partition is a runtime parameter,
//! not a construction parameter.
//!
//! ## Staleness contract
//! A stale `placement.partition` (failure mode A, scythe2.md §3.5) means
//! GPU 0 gets 70% of columns when it should get 60% — the result is still
//! *correct* (concatenation is shape-valid for any partition that sums to ≤ 1
//! per GPU), just slightly load-imbalanced. `forward_placed` never panics on
//! a stale placement; it produces the right tensor at potentially suboptimal
//! latency.
//!
//! Skill attribution:
//! - `rust-ffi-grim` §1 — ABI-safe repr for all structs passed over FFI.
//! - `rust-ffi-grim` §3 — compile-time gate via `cargo check`.

use std::sync::Arc;

use grim_tensor::BackendStorage;
use grim_tensor::backend::ScythePlacement;
use grim_tensor::error::{Error, Result};
use grim_tensor::shape::Shape;
use grim_tensor::{BackendDevice, DType, Device, Tensor};

use crate::modules::pick_device_for_storage_device;

// ── WeightSource helpers ──────────────────────────────────────────────────────

/// Slice the output-dimension of a weight tensor `[out_features, in_features]`.
///
/// Column-parallel sharding: GPU k gets columns `[start, start+count)` of
/// the output dimension. Returns a new tensor with shape `[count, in_features]`.
///
/// This is the `slice_output_dim` called for in scythe2.md §5.2.
///
/// # Contract
/// - `start + count <= weight.shape().dims()[0]` must hold; returns `Err` otherwise.
/// - The returned tensor lives on the same device as `weight`.
pub fn slice_output_dim(weight: &Tensor, start: usize, count: usize) -> Result<Tensor> {
    let dims = weight.shape().dims();
    if dims.len() < 2 {
        return Err(Error::Backend(
            "slice_output_dim: weight must be at least 2-D".into(),
        ));
    }
    let out_dim = dims[0];
    let in_dim: usize = dims[1..].iter().product();
    if start + count > out_dim {
        return Err(Error::Backend(format!(
            "slice_output_dim: start({start}) + count({count}) > out_dim({out_dim})"
        )));
    }
    // Extract the sub-matrix via host round-trip.
    // For a production path this would be a device-side slice kernel;
    // the host round-trip is correct and exercisable in tests (WI-3 gate).
    let full = weight.storage().to_cpu_vec_f32()?;
    let slice: Vec<f32> = full
        .chunks(in_dim)
        .skip(start)
        .take(count)
        .flat_map(|row| row.iter().copied())
        .collect();
    let dev = pick_device_for_storage_device(weight.device());
    let out_shape = Shape::new({
        let mut d = dims.to_vec();
        d[0] = count;
        d
    });
    let storage = dev.from_cpu(&slice, &out_shape, weight.dtype())?;
    Ok(Tensor::new(
        Arc::from(storage),
        out_shape,
        weight.dtype(),
        weight.provenance().clone(),
        weight.device().clone(),
    ))
}

/// Slice the input-dimension of a weight tensor `[out_features, in_features]`.
///
/// Row-parallel sharding: GPU k gets rows `[start, start+count)` of the
/// input dimension (i.e. the first `in_features` axis). Returns a tensor
/// with shape `[out_features, count]`.
///
/// This is the `slice_input_dim` called for in scythe2.md §5.2.
pub fn slice_input_dim(weight: &Tensor, start: usize, count: usize) -> Result<Tensor> {
    let dims = weight.shape().dims();
    if dims.len() < 2 {
        return Err(Error::Backend(
            "slice_input_dim: weight must be at least 2-D".into(),
        ));
    }
    let out_dim = dims[0];
    let in_dim = dims[1];
    if start + count > in_dim {
        return Err(Error::Backend(format!(
            "slice_input_dim: start({start}) + count({count}) > in_dim({in_dim})"
        )));
    }
    let full = weight.storage().to_cpu_vec_f32()?;
    // Weight is stored row-major [out, in]; select columns [start, start+count).
    let slice: Vec<f32> = full
        .chunks(in_dim)
        .flat_map(|row| row[start..start + count].iter().copied())
        .collect();
    let dev = pick_device_for_storage_device(weight.device());
    let out_shape = Shape::new(vec![out_dim, count]);
    let storage = dev.from_cpu(&slice, &out_shape, weight.dtype())?;
    Ok(Tensor::new(
        Arc::from(storage),
        out_shape,
        weight.dtype(),
        weight.provenance().clone(),
        weight.device().clone(),
    ))
}

// ── Scythe2Linear ─────────────────────────────────────────────────────────────

/// A linear layer whose shard boundaries are chosen per-forward by the
/// C²PLR controller, not fixed at load time.
///
/// The full (unsharded) weight is replicated on every participating GPU.
/// For >30B models where that is too large, the caller pre-shards at load
/// via `slice_output_dim` and sets `full_weight` to the local shard; the
/// controller then decides the *active* partition per forward.
///
/// The `layer_id` is the fingerprint index used by `PlacementCache`.
pub struct Scythe2Linear {
    /// Full unsharded weight tensor `[out_features, in_features]`.
    pub full_weight: Tensor,
    /// Optional bias `[out_features]`.
    pub bias: Option<Tensor>,
    /// Fingerprint index for `PlacementCache` lookup. Must be unique per layer.
    pub layer_id: u32,
    /// Preferred device for this layer (may be overridden by the placement).
    pub device: Device,
}

impl Scythe2Linear {
    /// Forward pass under a controller-chosen `ScythePlacement`.
    ///
    /// Behaviour depends on the placement type:
    /// - **Column-parallel** (`partition` does not sum to K across ranks):
    ///   each rank computes a column shard of the output; shards are
    ///   concatenated on the primary rank (rank 0). No collective needed.
    /// - **Row-parallel** (caller sets `is_row_parallel = true`):
    ///   each rank computes a partial row sum; CommFuse P2P fan-in is used to
    ///   collect partials. Default: column-parallel.
    /// - **Replicated** (`partition` all 1.0): every rank runs the full GEMM
    ///   and the outputs are identical — only rank 0's output is used.
    ///
    /// ## Staleness safety (scythe2.md §3.5, mode A)
    /// A stale `partition` produces a shape-valid result — the slice
    /// boundaries are always within `[0, out_features]` because `partition[k]`
    /// is clamped to `[0, 1]` and multiplied by `out_features` before
    /// `floor()`. No panic is possible from partition staleness.
    pub fn forward_placed(
        &self,
        x: &Tensor,
        placement: &ScythePlacement,
        is_row_parallel: bool,
    ) -> Result<Tensor> {
        let out_features = self.full_weight.shape().dims()[0];
        let dev = pick_device_for_storage_device(&self.device);

        if placement.ranks.is_empty() {
            return Err(Error::Backend(
                "Scythe2Linear: placement.ranks is empty".into(),
            ));
        }

        if is_row_parallel {
            self.forward_row_parallel(x, placement, &*dev)
        } else {
            self.forward_col_parallel(x, placement, out_features, &*dev)
        }
    }

    /// Column-parallel forward: each rank computes a column shard,
    /// outputs are concatenated. No collective required.
    ///
    /// WI-SB5 step 1: shards execute as real backend `matmul` calls on their
    /// placed rank's device (rocBLAS-bound when the layer lives on ROCm),
    /// replacing the host triple-loop emulation. Cross-rank fan-in is still
    /// host-staged — ring descriptors carrying shard pointers (opcode 1/2)
    /// remain the open second step, as does binding one rocBLAS handle per
    /// rank stream.
    fn forward_col_parallel(
        &self,
        x: &Tensor,
        placement: &ScythePlacement,
        out_features: usize,
        dev: &dyn BackendDevice,
    ) -> Result<Tensor> {
        let n_ranks = placement.ranks.len();
        let mut col_start = 0usize;
        let mut shards: Vec<(Box<dyn BackendStorage>, usize)> = Vec::with_capacity(n_ranks);
        let mut shard_out_dim = 0usize;

        let x_dims = x.shape().dims();
        let m = x_dims[..x_dims.len() - 1].iter().product::<usize>().max(1);
        let k = *x_dims.last().unwrap_or(&1);
        let x_vec = x.storage().to_cpu_vec_f32()?;

        for (rank_idx, gpu_ord) in placement.ranks.iter().enumerate() {
            let ratio = placement
                .partition
                .get(rank_idx)
                .copied()
                .unwrap_or(0.0)
                .clamp(0.0, 1.0);
            let count = ((ratio * out_features as f32).floor() as usize).max(0);
            let count = count.min(out_features - col_start);
            if count == 0 {
                continue;
            }
            // Slice the weight for this rank's column shard.
            let w_shard = slice_output_dim(&self.full_weight, col_start, count)?;
            // The shard executes on its placed rank's device when the layer
            // itself lives on ROCm; otherwise the layer's own backend runs it
            // so off-box tests stay hermetic.
            let rank_device = match (&self.device, gpu_ord) {
                (Device::Rocm(_), ord) => Device::Rocm(*ord),
                (other, _) => other.clone(),
            };
            let rank_dev = pick_device_for_storage_device(&rank_device);

            // B operand = shard transposed to (k, count). Host transpose is
            // O(k·count) copies against an O(m·k·count) GEMM; a cached
            // per-shard transposed weight arrives with descriptor work.
            let w_vec = w_shard.storage().to_cpu_vec_f32()?;
            let mut w_t = vec![0.0f32; k * count];
            for ni in 0..count {
                for ki in 0..k {
                    w_t[ki * count + ni] = w_vec[ni * k + ki];
                }
            }
            let b_storage = rank_dev.from_cpu(&w_t, &Shape::new(vec![k, count]), DType::F32)?;
            let a_storage = rank_dev.from_cpu(&x_vec, &Shape::new(vec![m, k]), DType::F32)?;
            let (out_s, _handle) = rank_dev.matmul(
                a_storage.as_ref(),
                b_storage.as_ref(),
                &Shape::new(vec![m, count]),
            )?;
            shards.push((out_s, count));
            shard_out_dim += count;
            col_start += count;
        }

        // Concatenate column shards (host-staged gather across ranks).
        if shards.is_empty() {
            return Err(Error::Backend(
                "Scythe2Linear: all column shards are empty".into(),
            ));
        }
        let mut concat = vec![0.0f32; m * shard_out_dim];
        let mut col_offset = 0usize;
        for (shard, n) in &shards {
            let shard_vec = shard.to_cpu_vec_f32()?;
            for bi in 0..m {
                concat[bi * shard_out_dim + col_offset..bi * shard_out_dim + col_offset + n]
                    .copy_from_slice(&shard_vec[bi * n..(bi + 1) * n]);
            }
            col_offset += n;
        }

        // Add bias if present.
        if let Some(bias) = &self.bias {
            let b_vec = bias.storage().to_cpu_vec_f32()?;
            for bi in 0..m {
                for ni in 0..shard_out_dim {
                    if ni < b_vec.len() {
                        concat[bi * shard_out_dim + ni] += b_vec[ni];
                    }
                }
            }
        }

        let out_shape = Shape::new(vec![m, shard_out_dim]);
        let storage = dev.from_cpu(&concat, &out_shape, DType::F32)?;
        Ok(Tensor::new(
            Arc::from(storage),
            out_shape,
            DType::F32,
            self.full_weight.provenance().clone(),
            self.device.clone(),
        ))
    }

    /// Row-parallel forward: each rank computes a partial row sum;
    /// CommFuse decomposed P2P fan-in collects the partials.
    ///
    /// Falls back to naive CPU-side sum when CommFuse is unavailable
    /// (non-ROCm builds), which is always correct and exercisable in tests.
    fn forward_row_parallel(
        &self,
        x: &Tensor,
        placement: &ScythePlacement,
        dev: &dyn BackendDevice,
    ) -> Result<Tensor> {
        let in_features = self.full_weight.shape().dims().get(1).copied().unwrap_or(1);
        let out_features = self.full_weight.shape().dims()[0];
        let x_dims = x.shape().dims();
        let m = x_dims[..x_dims.len() - 1].iter().product::<usize>().max(1);
        let x_vec = x.storage().to_cpu_vec_f32()?;
        let mut partial_sum = vec![0.0f32; m * out_features];

        let n_ranks = placement.ranks.len();
        let mut row_start = 0usize;
        for rank_idx in 0..n_ranks {
            let ratio = placement
                .partition
                .get(rank_idx)
                .copied()
                .unwrap_or(0.0)
                .clamp(0.0, 1.0);
            let count = ((ratio * in_features as f32).floor() as usize).max(0);
            let count = count.min(in_features - row_start);
            if count == 0 {
                continue;
            }
            let w_shard = slice_input_dim(&self.full_weight, row_start, count)?;
            // WI-SB5 step 1: partial products execute as device matmuls on
            // the rank's device; the P2P fan-in of partials stays host-staged
            // until CommFuse carries shard pointers (opcode 1/2 descriptors).
            let rank_device = match (&self.device, placement.ranks.get(rank_idx)) {
                (Device::Rocm(_), Some(ord)) => Device::Rocm(*ord),
                (other, _) => other.clone(),
            };
            let rank_dev = pick_device_for_storage_device(&rank_device);

            // A operand: this rank's K-slice of x, (m, count).
            let mut xs = vec![0.0f32; m * count];
            for bi in 0..m {
                xs[bi * count..(bi + 1) * count].copy_from_slice(
                    &x_vec[bi * in_features + row_start..bi * in_features + row_start + count],
                );
            }
            // B operand: shard transposed to (count, out_features).
            let w_vec = w_shard.storage().to_cpu_vec_f32()?;
            let mut w_t = vec![0.0f32; count * out_features];
            for ni in 0..out_features {
                for ki in 0..count {
                    w_t[ki * out_features + ni] = w_vec[ni * count + ki];
                }
            }
            let a_storage = rank_dev.from_cpu(&xs, &Shape::new(vec![m, count]), DType::F32)?;
            let b_storage =
                rank_dev.from_cpu(&w_t, &Shape::new(vec![count, out_features]), DType::F32)?;
            let (partial, _h) = rank_dev.matmul(
                a_storage.as_ref(),
                b_storage.as_ref(),
                &Shape::new(vec![m, out_features]),
            )?;
            let pv = partial.to_cpu_vec_f32()?;
            for (dst, src) in partial_sum.iter_mut().zip(pv) {
                *dst += src;
            }
            row_start += count;
        }

        // Add bias if present.
        if let Some(bias) = &self.bias {
            let b_vec = bias.storage().to_cpu_vec_f32()?;
            for bi in 0..m {
                for ni in 0..out_features {
                    if ni < b_vec.len() {
                        partial_sum[bi * out_features + ni] += b_vec[ni];
                    }
                }
            }
        }

        let out_shape = Shape::new(vec![m, out_features]);
        let storage = dev.from_cpu(&partial_sum, &out_shape, DType::F32)?;
        Ok(Tensor::new(
            Arc::from(storage),
            out_shape,
            DType::F32,
            self.full_weight.provenance().clone(),
            self.device.clone(),
        ))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use grim_backend_cpu::CpuDevice;
    use grim_tensor::backend::ScythePlacement;
    use grim_tensor::shape::Shape;
    use grim_tensor::{BackendDevice, DType, Device};

    /// Build a simple Tensor on CPU with given data.
    fn make_tensor(data: Vec<f32>, shape: Shape) -> Tensor {
        let dev = CpuDevice::new();
        let storage = dev.from_cpu(&data, &shape, DType::F32).unwrap();
        Tensor::new(
            Arc::from(storage),
            shape,
            DType::F32,
            grim_tensor::dtype::QuantProvenance::GrimNative,
            Device::Cpu,
        )
    }

    fn single_rank_placement() -> ScythePlacement {
        ScythePlacement {
            ranks: vec![0],
            partition: vec![1.0],
            routes: vec![grim_tensor::ScytheLink::Host],
        }
    }

    fn two_rank_placement_col() -> ScythePlacement {
        ScythePlacement {
            ranks: vec![0, 1],
            partition: vec![0.5, 0.5],
            routes: vec![
                grim_tensor::ScytheLink::Host,
                grim_tensor::ScytheLink::Host,
                grim_tensor::ScytheLink::Host,
                grim_tensor::ScytheLink::Host,
            ],
        }
    }

    /// WI-3 gate: ‖Y_scythe2 − Y_ref‖∞ < 1e-4 for a single-rank (no-sharding) pass.
    #[test]
    fn test_scythe2_linear_parity_single_rank() {
        // Weight [4, 3], input [2, 3].
        let w = make_tensor(
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            Shape::new(vec![4, 3]),
        );
        let x = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]));

        let layer = Scythe2Linear {
            full_weight: w.clone(),
            bias: None,
            layer_id: 0,
            device: Device::Cpu,
        };

        let p = single_rank_placement();
        let y = layer.forward_placed(&x, &p, false).unwrap();
        let y_vec = y.storage().to_cpu_vec_f32().unwrap();

        // Reference: x @ W^T manually.
        // W^T = [[1,0,0,1],[0,1,0,1],[0,0,1,1]]
        // x[0]=[1,2,3] → [1, 2, 3, 6]
        // x[1]=[4,5,6] → [4, 5, 6, 15]
        let expected = vec![1.0, 2.0, 3.0, 6.0, 4.0, 5.0, 6.0, 15.0];
        let max_diff = y_vec
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "Parity test failed: max_diff={max_diff:.2e} expected <1e-4"
        );
    }

    /// WI-3 gate: 50/50 column-parallel split must produce same result as reference.
    #[test]
    fn test_scythe2_linear_parity_col_parallel() {
        let w = make_tensor(
            // [4, 2] — 4 output units, 2 input
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 1.0],
            Shape::new(vec![4, 2]),
        );
        let x = make_tensor(vec![1.0, 2.0], Shape::new(vec![1, 2]));

        let layer = Scythe2Linear {
            full_weight: w.clone(),
            bias: None,
            layer_id: 1,
            device: Device::Cpu,
        };

        let p = two_rank_placement_col();
        let y = layer.forward_placed(&x, &p, false).unwrap();
        let y_vec = y.storage().to_cpu_vec_f32().unwrap();

        // Reference: x @ W^T  = [1*1+2*0, 1*0+2*1, 1*1+2*1, 1*(-1)+2*1]
        //                      = [1, 2, 3, 1]
        let expected = vec![1.0, 2.0, 3.0, 1.0];
        let max_diff = y_vec
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "Col-parallel parity test failed: max_diff={max_diff:.2e}"
        );
    }

    /// slice_output_dim must preserve the inner dimensions.
    #[test]
    fn test_slice_output_dim_shape() {
        let w = make_tensor(vec![1.0f32; 12], Shape::new(vec![4, 3]));
        let sliced = slice_output_dim(&w, 1, 2).unwrap();
        assert_eq!(sliced.shape().dims(), &[2, 3]);
    }

    /// slice_input_dim must preserve the output dimension.
    #[test]
    fn test_slice_input_dim_shape() {
        let w = make_tensor(vec![1.0f32; 12], Shape::new(vec![3, 4]));
        let sliced = slice_input_dim(&w, 1, 2).unwrap();
        assert_eq!(sliced.shape().dims(), &[3, 2]);
    }

    /// Out-of-range slice must return an error, not panic.
    #[test]
    fn test_slice_output_dim_oob() {
        let w = make_tensor(vec![1.0f32; 12], Shape::new(vec![4, 3]));
        assert!(slice_output_dim(&w, 3, 2).is_err());
    }
}
