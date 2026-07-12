//! Wavefront-aware tensor layout utilities.
//!
//! AMD CDNA/RDNA GPUs process workitems in wavefronts (32 or 64 threads
//! executing lockstep). GEMM kernels achieve peak LDS bandwidth when the
//! weight matrix columns are a multiple of the wavefront size — every lane
//! accesses a distinct column, achieving fully-coalesced loads.
//!
//! This module provides the dimension-computation half of wavefront-tiled
//! alignment. The actual data tiling/untiling is in the ROCm backend's
//! `WavefrontTiledLayout::tile/untile`.

/// Compute the wavefront-tiled padded dimensions for a row-major weight matrix.
///
/// Rows and columns are each rounded up to the next multiple of
/// `wavefront_size` (64 for CDNA2/3, 32 for RDNA2/3). Shapes that are
/// already aligned pass through unchanged.
///
/// # Arguments
/// * `rows` — original row count (M dimension)
/// * `cols` — original column count (K dimension)
/// * `wavefront_size` — wavefront size (64 for CDNA, 32 for RDNA)
///
/// # Returns
/// `(padded_rows, padded_cols)`
///
/// # Example
/// ```
/// use grim_tensor::wavefront::padded_dims;
///
/// // 70×60 with wavefront=64 → 128×64
/// assert_eq!(padded_dims(70, 60, 64), (128, 64));
///
/// // 64×64 is already aligned → unchanged
/// assert_eq!(padded_dims(64, 64, 64), (64, 64));
/// ```
pub const fn padded_dims(rows: usize, cols: usize, wavefront_size: u32) -> (usize, usize) {
    let wf = wavefront_size as usize;
    let rows_padded = (rows + wf - 1) & !(wf - 1);
    let cols_padded = (cols + wf - 1) & !(wf - 1);
    (rows_padded, cols_padded)
}