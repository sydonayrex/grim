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

// ---------------------------------------------------------------------------
// Tiled multi-block prologue (Task 4 graph wiring, block width 1024).
//
// A folded linear whose input dim is a multiple of 1024 (e.g. 5120 =
// 5 blocks of 1024) applies block-diagonal D·H to the full row: block b
// uses signs[b*1024 .. (b+1)*1024] and its own FWHT. The existing single-
// launch `launch_fwht_signed` handles ONE 1024-wide row per grid row with
// `signs[(r % n_blk) * N]` where r is the activation row — correct only
// when each activation row is 1024 wide. For 5120-wide inputs the host
// calls this tiled wrapper `n_blk` times, one per block:
//
//   For b in 0..n_blk:
//     launch one kernel over `rows` activation rows where block b of
//     every row is transformed as an INDEPENDENT 1024-row.
//
// The offset kernel pointers are computed inside the same HIP kernel
// interface: each call passes `src + b*1024` as a contiguous base for
// the elements of block b across the whole activation. But `launch_-
// fwht_signed` assumes rows are contiguous 1024-wide in the whole
// buffer — which fails for any input whose row stride > 1024.
//
// TILED INVARIANT (checked here, not assumed):
//   * n is exactly 1024 (the kernel width)
//   * Buffer holds `rows` activation rows of `n * n_blk` elements,
//     row-major. The caller's sign buffer holds `n_blk * n` elements
//     (full width = signs vector for the whole width).
//   * Each call passes a *slice* of the signs buffer for one block
//     as a separate `signs` RocmStorage view (offset base), so
//     `n_blk == 1` and every activation row's block b gets
//     `signs_row[b*1024..]` — exactly the folded D·H order.
//   * src/dst buffers hold `rows * n * n_blk` elements. The call
//     for block b reads/writes only cells c where
//     c % (n*n_blk) in [b*n, (b+1)*n) — enforced by EXPLICIT
//     STRIDED copy: this function issues one `launch_fwht_signed`
//     per block against a *contiguous* staging buffer — never a
//     kernel that reads across the row stride.
//
// The strided copy D2D staging (device→device copy of a strided
// input block into a page-aligned contiguous `rows × 1024` page,
// kernel, then D2D copy back) is a measure-zero design compromise:
// decode batch ≤ 64 rows × 1024 × 4 B = 256 KB max staging per
// linear, well within the existing activation pool VRAM budget.
//
// API mirrors the single-block form but takes the full-width shapes:
//   src/dst: rows × (n * n_blk) elements, f32
//   signs:   width = n * n_blk elements, f32, must have elem_count
//            == n * n_blk (checked)
//   n_blk > 0, n == 1024, rows > 0
// Returns Ok(()) after the ptr-slice staging launch sequence.
// The kernel variant switch (block vs smem) is per block: rows is
// the activation row count, so a decode batch uses the block
// variant — matching the single-block launcher rule.
//
// CERTEFICATE — what this wrapper guarantees (and what it does not):
//   * Each block's 1024 outputs == blockFwht(x_row[b*1024:(b+1)*1024] * signs[b*1024:(b+1)*1024]).
//   * No element outside block b's columns is read or written.
//   * Out-of-range full-width signs → hard Backend error (the
//     fork's `has no sign vector for width %u` throw, not a silent
//     skip — enforced by the caller via `HadamardPlan::signs_for`
//     BEFORE this function is reached).
pub fn launch_fwht_signed_tiled(
    dev: &RocmDevice,
    src: &RocmStorage,
    dst: &RocmStorage,
    signs: Option<&RocmStorage>,
    n: u32,
    n_blk: u32,
    rows: usize,
) -> Result<()> {
    if n != 1024 {
        return Err(Error::Backend(format!(
            "launch_fwht_signed_tiled: width {n} must be 1024"
        )));
    }
    if n_blk == 0 {
        return Err(Error::Backend("launch_fwht_signed_tiled: n_blk must be > 0".into()));
    }
    if rows == 0 {
        return Err(Error::Backend("launch_fwht_signed_tiled: rows must be > 0".into()));
    }
    let full_width = n as usize * n_blk as usize;
    if let Some(sg) = signs {
        if sg.dtype != DType::F32 {
            return Err(Error::Backend(format!("fwht: signs must be f32, got {:?}", sg.dtype)));
        }
        if sg.shape.elem_count() != full_width {
            return Err(Error::Backend(format!(
                "fwht: signs element count {} != full width {}",
                sg.shape.elem_count(),
                full_width
            )));
        }
    }
    if src.shape.elem_count() != rows * full_width || dst.shape.elem_count() < rows * full_width {
        return Err(Error::Backend(format!(
            "fwht tiled: src/dst must hold rows*width = {} elements",
            rows * full_width
        )));
    }
    if dst.dtype != DType::F32 {
        return Err(Error::Backend(format!("fwht: dst must be f32, got {:?}", dst.dtype)));
    }
    let f16 = if src.dtype == DType::F32 {
        false
    } else if src.dtype == DType::F16 {
        true
    } else {
        return Err(Error::Backend(format!(
            "fwht tiled: src must be f32 or f16, got {:?}",
            src.dtype
        )));
    };
    let src_ptr = src
        .device_ptr
        .ok_or_else(|| Error::Backend("fwht tiled: src has no device ptr".into()))?;
    let dst_ptr = dst
        .device_ptr
        .ok_or_else(|| Error::Backend("fwht tiled: dst has no device ptr".into()))?;
    let signs_ptr = match signs {
        Some(sg) => sg
            .device_ptr
            .ok_or_else(|| Error::Backend("fwht tiled: signs has no device ptr".into()))?,
        None => 0u64,
    };

    // 2D grid: blockIdx.x = block index b, blockIdx.y = activation row r.
    // ONE launch for the whole n_blk * rows grid — a folded linear costs one
    // graph node, not n_blk gather/kernel/scatter round-trips.
    let kernel = if f16 {
        "grim_fwht_tiled_f16"
    } else {
        "grim_fwht_tiled_f32"
    };
    let grid_dim = HipDim3::new(n_blk, rows as u32, 1);
    let block_dim = HipDim3::new(256, 1, 1);

    let mut sptr = src_ptr;
    let mut dptr = dst_ptr;
    let mut rows64 = rows as i64;
    let mut stride64 = full_width as i64;
    let mut scale = 1.0f32 / (n as f32).sqrt();
    let mut sptr_signs = signs_ptr;
    dev.launch_compute_kernel(
        kernel,
        grid_dim,
        block_dim,
        &mut [
            arg(&mut sptr),
            arg(&mut dptr),
            arg(&mut rows64),
            arg(&mut stride64),
            arg(&mut scale),
            arg(&mut sptr_signs),
        ],
    )?;
    Ok(())
}

/// GDN ssm_out activation permutation (llama-graph.cpp:1696-1704):
/// reshape [hd, nk, rep] -> [hd, rep, nk] before signs + FWHT.
pub fn launch_gdn_perm(
    dev: &RocmDevice,
    src: &RocmStorage,
    dst: &RocmStorage,
    rows: usize,
    perm_hd: usize,
    perm_nk: usize,
    perm_rep: usize,
) -> Result<()> {
    let n_in = perm_hd * perm_nk * perm_rep;
    let total = rows * n_in;
    if src.shape.elem_count() < total || dst.shape.elem_count() < total {
        return Err(Error::Backend(format!(
            "gdn_perm: src/dst must hold at least {} elements",
            total
        )));
    }
    let mut sptr = src
        .device_ptr
        .ok_or_else(|| Error::Backend("gdn_perm: src has no device ptr".into()))?;
    let mut dptr = dst
        .device_ptr
        .ok_or_else(|| Error::Backend("gdn_perm: dst has no device ptr".into()))?;
    let mut total_i = total as i64;
    let mut hd = perm_hd as i32;
    let mut nk = perm_nk as i32;
    let mut rep = perm_rep as i32;
    let grid = HipDim3::new((total as u32).div_ceil(256), 1, 1);
    let block = HipDim3::new(256, 1, 1);
    dev.launch_compute_kernel(
        "grim_gdn_perm_f32",
        grid,
        block,
        &mut [
            arg(&mut sptr),
            arg(&mut dptr),
            arg(&mut total_i),
            arg(&mut hd),
            arg(&mut nk),
            arg(&mut rep),
        ],
    )?;
    Ok(())
}

/// Fused Tiled SignFlip + FWHT-1024 + Q8_0 quantization (34 bytes per 32 elements).
/// Eliminates the intermediate f32 HBM write/read and separate quantize launch.
pub fn launch_fwht_signed_tiled_q80(
    dev: &RocmDevice,
    src: &RocmStorage,
    dst_q80: &RocmStorage,
    signs: Option<&RocmStorage>,
    n: u32,
    n_blk: u32,
    rows: usize,
) -> Result<()> {
    let _dev_guard = crate::device::util::DeviceGuard::set(dev.ordinal as i32);
    if n != 1024 || n_blk == 0 || rows == 0 {
        return Err(Error::Backend(format!(
            "launch_fwht_signed_tiled_q80: invalid (n={n}, n_blk={n_blk}, rows={rows})"
        )));
    }
    let full_width = (n as usize) * (n_blk as usize);
    if src.dtype != DType::F32 || src.shape.elem_count() != rows * full_width {
        return Err(Error::Backend(
            "launch_fwht_signed_tiled_q80: src must be f32 of rows*full_width".into(),
        ));
    }
    let needed_bytes = rows * (n_blk as usize) * 32 * 34;
    if dst_q80.shape.elem_count() < needed_bytes {
        return Err(Error::Backend(format!(
            "launch_fwht_signed_tiled_q80: dst_q80 needs {needed_bytes} bytes, got {}",
            dst_q80.shape.elem_count()
        )));
    }
    let mut sptr = src
        .device_ptr
        .ok_or_else(|| Error::Backend("fwht_q80: src has no device ptr".into()))?;
    let mut dptr = dst_q80
        .device_ptr
        .ok_or_else(|| Error::Backend("fwht_q80: dst_q80 has no device ptr".into()))?;
    let mut sptr_signs = match signs {
        Some(sg) => {
            if sg.dtype != DType::F32 || sg.shape.elem_count() != full_width {
                return Err(Error::Backend(
                    "fwht_q80: signs must be f32 matching full_width".into(),
                ));
            }
            sg.device_ptr
                .ok_or_else(|| Error::Backend("fwht_q80: signs has no device ptr".into()))?
        }
        None => 0u64,
    };
    let mut rows64 = rows as i64;
    let mut stride64 = full_width as i64;
    let mut scale = 1.0f32 / (n as f32).sqrt();
    let grid_dim = HipDim3::new(n_blk, rows as u32, 1);
    let block_dim = HipDim3::new(256, 1, 1);
    dev.launch_compute_kernel(
        "grim_fwht_tiled_q80_f32",
        grid_dim,
        block_dim,
        &mut [
            arg(&mut sptr),
            arg(&mut dptr),
            arg(&mut rows64),
            arg(&mut stride64),
            arg(&mut scale),
            arg(&mut sptr_signs),
        ],
    )?;
    Ok(())
}
