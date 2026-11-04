//! FWHT (Prism/Bonsai Hadamard) launcher for `RocmDevice`.
//!
//! One kernel launch per call: rows x FWHT-1024 in-place-out-of-place
//! transform, optional ±1 signs applied at load with the n_blk modulo
//! semantics of fwht.cu:100 (`signs + (r % n_blk) * N`). Decode shapes
//! (few rows) use the 256-thread block variant; prefill shapes use the LDS
//! variant. See `kernels::fwht`.

use std::ffi::c_void;

use grim_tensor::dtype::DType;
use grim_tensor::error::{Error, Result};

use crate::device::roc_device::RocmDevice;
use crate::memory::storage::RocmStorage;
use crate::{HipDim3, arg};

/// Rows at or below this run the 256-thread in-register block variant
/// (`grim_fwht_block_*`, decode); above it, the LDS smem variant
/// (`grim_fwht_smem_*`, prefill). Mirrors Prism's decode/prefill split
/// (fwht.cu:132-135).
const FWHT_BLOCK_VARIANT_MAX_ROWS: usize = 64;

/// Launch `grim_fwht_signed`-style FWHT over `rows` rows of width `n`.
///
/// - `n` must be 1024 in this task (extend the dispatch when a model needs
///   another width).
/// - `src` f32 or f16; `dst` f32, same element count.
/// - `signs`: optional f32 tensor of n_blk * n elements (n_blk >= 1); row r
///   uses `signs[(r % n_blk) * n ...]`. `None` passes a null pointer.
pub fn launch_fwht_signed(
    dev: &RocmDevice,
    src: &RocmStorage,
    dst: &RocmStorage,
    signs: Option<&RocmStorage>,
    n: u32,
    rows: usize,
) -> Result<*mut c_void> {
    if n != 1024 {
        return Err(Error::Backend(format!(
            "launch_fwht_signed: unsupported width {n} (only 1024 is wired in this task)"
        )));
    }
    if rows == 0 {
        return Err(Error::Backend(
            "launch_fwht_signed: rows must be non-zero".into(),
        ));
    }
    let src_ptr = src
        .device_ptr
        .ok_or_else(|| Error::Backend("fwht: src has no device ptr".into()))?;
    let dst_ptr = dst
        .device_ptr
        .ok_or_else(|| Error::Backend("fwht: dst has no device ptr".into()))?;
    if dst.dtype != DType::F32 {
        return Err(Error::Backend(format!(
            "fwht: dst must be f32, got {:?}",
            dst.dtype
        )));
    }
    let f16 = if src.dtype == DType::F32 {
        false
    } else if src.dtype == DType::F16 {
        true
    } else {
        return Err(Error::Backend(format!(
            "fwht: src must be f32 or f16, got {:?}",
            src.dtype
        )));
    };
    let (signs_ptr, n_blk) = match signs {
        Some(sg) => {
            if sg.dtype != DType::F32 {
                return Err(Error::Backend(format!(
                    "fwht: signs must be f32, got {:?}",
                    sg.dtype
                )));
            }
            let elems = sg.shape.elem_count();
            if elems == 0 || elems % (n as usize) != 0 {
                return Err(Error::Backend(format!(
                    "fwht: signs element count {elems} must be a multiple of n={n}"
                )));
            }
            let ptr = sg
                .device_ptr
                .ok_or_else(|| Error::Backend("fwht: signs has no device ptr".into()))?;
            (ptr, (elems / n as usize) as i32)
        }
        None => (0u64, 1),
    };

    let (kernel, grid_dim) = if rows <= FWHT_BLOCK_VARIANT_MAX_ROWS {
        (
            if f16 { "grim_fwht_block_f16" } else { "grim_fwht_block_f32" },
            HipDim3::new(rows as u32, 1, 1),
        )
    } else {
        (
            if f16 { "grim_fwht_smem_f16" } else { "grim_fwht_smem_f32" },
            HipDim3::new(rows as u32, 1, 1),
        )
    };
    let block_dim = HipDim3::new(256, 1, 1);

    let mut sptr = src_ptr;
    let mut dptr = dst_ptr;
    let mut rows64 = rows as i64;
    let mut scale = 1.0f32 / (n as f32).sqrt();
    let mut sptr_signs = signs_ptr;
    let mut n_blk_i32 = n_blk;
    dev.launch_compute_kernel(
        kernel,
        grid_dim,
        block_dim,
        &mut [
            arg(&mut sptr),
            arg(&mut dptr),
            arg(&mut rows64),
            arg(&mut scale),
            arg(&mut sptr_signs),
            arg(&mut n_blk_i32),
        ],
    )
}
