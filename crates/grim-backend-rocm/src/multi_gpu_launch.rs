//! Multi-GPU parallel kernel execution launcher splitting problem dimensions across devices.

use std::ffi::c_void;

use grim_tensor::error::{Error, Result};

use crate::autotune::ShapeClass;
use crate::device::hardware_spec::HardwareSpec;
use crate::device::roc_device::RocmDevice;
use crate::kernels::source_asm::compute_kernel_source_with_spec;
use crate::kernels::tile_picker::{ShapeDims, pick_tiles};
use crate::rccl::RcclAllReduce;

/// Launch a kernel across N GPUs with shard split on the M dimension.
///
/// Each device `i` computes shard `[i * M/N, (i+1) * M/N)` of the output into
/// its own output pointer (per-rank shard). After all device kernels complete,
/// an optional RCCL all-reduce combines the shards using each rank's own
/// communicator. `out_ptrs` must contain one device pointer per rank, each
/// pointing to the rank's shard of the output buffer (shard_m * full_dims.n
/// elements at offset `rank * shard_m * full_dims.n`).
///
/// [P0-18/P0-19 fix: previously used shared output pointer + wrong count +
/// rank-0 communicator only.]
pub fn launch_multi_gpu_kernel(
    devices: &[&RocmDevice],
    comm: Option<&RcclAllReduce>,
    entry: &str,
    shape_class: ShapeClass,
    full_dims: ShapeDims,
    hardware_specs: &[HardwareSpec],
    out_ptrs: &[*mut c_void],
) -> Result<()> {
    let n = devices.len();
    if n != hardware_specs.len() {
        return Err(Error::Backend(format!(
            "device count {} != spec count {}",
            n,
            hardware_specs.len()
        )));
    }
    if n < 2 {
        return Err(Error::Backend(format!(
            "multi-GPU launch requires at least 2 devices, got {}",
            n
        )));
    }
    if out_ptrs.len() != n {
        return Err(Error::Backend(format!(
            "out_ptrs len {} != device count {}",
            out_ptrs.len(),
            n
        )));
    }

    for (i, (device, spec)) in devices.iter().zip(hardware_specs.iter()).enumerate() {
        let shard_start = (i as u32 * full_dims.m) / (n as u32);
        let shard_end = ((i as u32 + 1) * full_dims.m) / (n as u32);
        let shard_m = shard_end - shard_start;

        let shard_dims = ShapeDims::new(shard_m, full_dims.n, full_dims.k);
        let tiles = pick_tiles(spec, shape_class, shard_dims);

        let source = compute_kernel_source_with_spec(
            spec,
            entry,
            shape_class,
            full_dims,
            i as u32,
            n as u32,
            Some(&tiles),
        );

        let (_hsaco, _lowered) = device.jit_compile_or_cache(&source, entry, Some(spec))?;

        let grid_m = (shard_m + tiles.grid_stride_m - 1) / tiles.grid_stride_m;
        let grid_n = (full_dims.n + tiles.grid_stride_n - 1) / tiles.grid_stride_n;

        let mut args = out_ptrs[i..i + 1].to_vec();
        let grid = crate::HipDim3::new(grid_m, grid_n, 1);
        let block = crate::HipDim3::new(tiles.threads, 1, 1);

        let _ = device.launch_compute_kernel_with_solution(entry, grid, block, &mut args, None, 0)?;
    }

    if let Some(rccl) = comm {
        for (i, &out_ptr) in out_ptrs.iter().enumerate() {
            if !out_ptr.is_null() {
                let shard_count = ((i as u32 + 1) * full_dims.m / (n as u32)
                    - i as u32 * full_dims.m / (n as u32))
                    * full_dims.n as u32 as usize;
                let ptr_val = out_ptr as u64;
                let _ = rccl.sum_gradients_device(ptr_val, ptr_val, shard_count, 0, i);
            }
        }
    }

    Ok(())
}
