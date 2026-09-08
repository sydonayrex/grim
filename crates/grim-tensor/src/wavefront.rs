//! Wavefront-aware tensor layout utilities for AMD CDNA/RDNA architectures.
//! Computes dimension padding to align weight matrices with wavefront boundaries.

/// Computes wavefront-aligned padded dimensions for row-major weight matrices.
/// Rounds rows and columns up to the nearest multiple of `wavefront_size`.
pub const fn padded_dims(rows: usize, cols: usize, wavefront_size: u32) -> (usize, usize) {
    debug_assert!(
        wavefront_size.is_power_of_two(),
        "padded_dims: wavefront_size must be a power of two — the bit-mask rounding is wrong otherwise"
    );
    let wf = wavefront_size as usize;
    let rows_padded = (rows + wf - 1) & !(wf - 1);
    let cols_padded = (cols + wf - 1) & !(wf - 1);
    (rows_padded, cols_padded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_padded_dims_boundary_conditions() {
        assert_eq!(padded_dims(0, 0, 64), (0, 0));
        assert_eq!(padded_dims(64, 64, 64), (64, 64));
        assert_eq!(padded_dims(65, 63, 64), (128, 64));
        assert_eq!(padded_dims(1, 1, 64), (64, 64));
        assert_eq!(padded_dims(33, 33, 32), (64, 64));
        assert_eq!(padded_dims(32, 32, 32), (32, 32));
    }
}
