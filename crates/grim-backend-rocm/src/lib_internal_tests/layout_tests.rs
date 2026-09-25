//! layout tests — split from the original monolithic lib_internal_tests.rs.

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn test_wavefront_tiled_layout_tile_untile_roundtrip() {
        let wf = WavefrontTiledLayout::new(128, 64, 64);
        assert_eq!(wf.num_wavefronts, 2);
        assert_eq!(wf.cols_padded, 64);

        let src: Vec<f32> = (0..128 * 64).map(|i| i as f32).collect();
        let tiled = wf.tile(&src, 128, 64);
        let (nwf, cpad, wfs) = wf.output_shape();
        assert_eq!(nwf, 2);
        assert_eq!(cpad, 64);
        assert_eq!(wfs, 64);
        assert_eq!(tiled.len(), 2 * 64 * 64);

        let recovered = wf.untile(&tiled, 128, 64);
        assert_eq!(recovered.len(), src.len());
        for (a, b) in src.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_packed_quant_layout_roundtrip() {
        use crate::device::layout::PackedQuantLayout;
        for bits in [2, 3, 4] {
            let layout = PackedQuantLayout::new(4, 32, bits, 64);
            let src: Vec<f32> = (0..4 * 32)
                .map(|i| (i as f32 / 128.0) * 2.0 - 1.0)
                .collect();
            let packed = layout.pack(&src);
            let unpacked = layout.unpack(&packed);
            assert_eq!(unpacked.len(), src.len());
            // Quantization will introduce some error, but check that it's within quantization bucket size:
            let max_allowed_error = 1.0 / ((1 << bits) - 1) as f32 + 1e-5;
            for (a, b) in src.iter().zip(unpacked.iter()) {
                assert!(
                    (a - b).abs() <= max_allowed_error,
                    "Error too large for {} bits: got a={}, b={}, diff={}",
                    bits,
                    a,
                    b,
                    (a - b).abs()
                );
            }
        }
    }

    #[test]
    fn test_wavefront_tiled_layout_with_padding() {
        let wf = WavefrontTiledLayout::new(70, 50, 64);
        assert_eq!(wf.num_wavefronts, 2);
        assert_eq!(wf.cols_padded, 64);

        let src: Vec<f32> = (0..70 * 50).map(|i| i as f32).collect();
        let tiled = wf.tile(&src, 70, 50);
        assert_eq!(tiled.len(), 2 * 64 * 64);

        let recovered = wf.untile(&tiled, 70, 50);
        assert_eq!(recovered.len(), 70 * 50);
        for (a, b) in src.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-6, "untiled value differs at some index");
        }
    }

    #[test]
    fn test_wavefront_tiled_layout_35x40_roundtrip() {
        let wf = WavefrontTiledLayout::new(35, 40, 64);
        assert_eq!(wf.num_wavefronts, 1);
        assert_eq!(wf.cols_padded, 64);

        let src: Vec<f32> = (0..35 * 40).map(|i| i as f32 * 0.5).collect();
        let tiled = wf.tile(&src, 35, 40);
        assert_eq!(tiled.len(), 64 * 64);

        let recovered = wf.untile(&tiled, 35, 40);
        assert_eq!(recovered.len(), 35 * 40);
        for (a, b) in src.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-6, "35x40 round-trip value mismatch");
        }
    }

    #[test]
    fn test_wavefront_tiled_layout_32_wavefront_rdna_roundtrip() {
        let wf = WavefrontTiledLayout::new(40, 50, 32);
        assert_eq!(wf.num_wavefronts, 2);
        assert_eq!(wf.cols_padded, 64);

        let src: Vec<f32> = (0..40 * 50).map(|i| (i as f32) * 1.5).collect();
        let tiled = wf.tile(&src, 40, 50);
        assert_eq!(tiled.len(), 2 * 32 * 64);

        let recovered = wf.untile(&tiled, 40, 50);
        assert_eq!(recovered.len(), 40 * 50);
        for (i, (a, b)) in src.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "RDNA 32-thread round-trip mismatch at index {i}: got {b} want {a}"
            );
        }
    }

    #[test]
    fn test_resolve_weight_layout_attention_defaults_to_wavefront_tiled() {
        let layout = resolve_weight_layout("blk.48.attn_q.weight", None, WavefrontSize::W64);
        match layout {
            WeightLayout::WavefrontTiled { wavefront_size } => assert_eq!(wavefront_size, 64),
            other => panic!("expected WavefrontTiled, got {other:?}"),
        }
    }

    #[test]
    fn test_resolve_weight_layout_non_attention_defaults_to_row_major() {
        let layout = resolve_weight_layout(
            "model.layers.0.mlp.gate_proj.weight",
            None,
            WavefrontSize::W64,
        );
        match layout {
            WeightLayout::RowMajor => {}
            other => panic!("expected RowMajor, got {other:?}"),
        }
    }

    #[test]
    fn test_wavefront_size_for_gcn_w64() {
        // CDNA2 (gfx90a) is Wave64.
        let wf = wavefront_size_for_gcn("gfx90a");
        assert_eq!(wf, 64);
    }

    #[test]
    fn test_wavefront_size_for_gcn_w32() {
        // RDNA2/3 (gfx1100, gfx1036) is Wave32.
        let wf = wavefront_size_for_gcn("gfx1100");
        assert_eq!(wf, 32);
        let wf = wavefront_size_for_gcn("gfx1036");
        assert_eq!(wf, 32);
    }

    #[test]
    fn test_wavefront_size_for_gcn_unknown_returns_32() {
        // Unknown GCN returns safe default of 32 (RDNA-first project).
        let wf = wavefront_size_for_gcn("gfx_unknown");
        assert_eq!(wf, 32);
    }

    #[test]
    fn test_wavefront_size_for_gcn_cdna2_returns_64() {
        // CDNA2 (gfx90a) returns 64 — the only W64 case in the table.
        let wf = wavefront_size_for_gcn("gfx90a");
        assert_eq!(wf, 64);
    }

    #[test]
    fn test_wavefront_size_detection_initializes() {
        let dev = RocmDevice::new(0);
        // Ensure wavefront size has a valid enum variant populated
        let size = dev.props.wavefront_size;
        assert!(size == WavefrontSize::W32 || size == WavefrontSize::W64);
    }

    #[test]
    fn test_align_tensor_pads_rows_to_wavefront() {
        // 70 rows with W64 should pad to 128
        let data: Vec<f32> = (0..70 * 60).map(|i| i as f32).collect();
        let (padded, new_rows, new_cols) = align_tensor_for_rocm_gemm(&data, 70, 60, 64);
        assert_eq!(new_rows, 128); // Padded to next multiple of 64
        assert_eq!(new_cols, 60); // Not padded
        assert_eq!(padded.len(), 128 * 60);
        // First 70*60 elements should be preserved
        assert_eq!(padded[0], 0.0);
        // Row 1, col 0 -> padded[60]
        assert_eq!(padded[60], 60.0, "row 1, col 0 should be data[60]=60.0");
    }

    #[test]
    fn test_align_tensor_32_wavefront() {
        // 35 rows with W32 should pad to 64
        let data: Vec<f32> = (0..35 * 40).map(|i| i as f32).collect();
        let (padded, new_rows, new_cols) = align_tensor_for_rocm_gemm(&data, 35, 40, 32);
        assert_eq!(new_rows, 64);
        assert_eq!(new_cols, 40);
        // Padded values should be zero
        for row in 35..64 {
            for col in 0..40 {
                assert_eq!(
                    padded[row * 40 + col],
                    0.0,
                    "padding should be zero at row {row}, col {col}"
                );
            }
        }
    }

    #[test]
    fn test_align_tensor_preserves_data() {
        // Already aligned data should be unchanged
        let data: Vec<f32> = (0..64 * 64).map(|i| i as f32).collect();
        let (padded, new_rows, new_cols) = align_tensor_for_rocm_gemm(&data, 64, 64, 64);
        assert_eq!(new_rows, 64);
        assert_eq!(new_cols, 64);
        assert_eq!(padded.len(), 64 * 64);
        for (i, &val) in data.iter().enumerate() {
            assert_eq!(padded[i], val, "data at {i} should be preserved");
        }
    }

    #[test]
    fn test_align_quantized_tensor_basic() {
        // 128x256 tensor with 4-bit quantization
        let data: Vec<u8> = vec![0xAB; 128 * 256 / 2]; // 4-bit = 2 values per byte
        let shape = vec![128, 256];
        let (padded, new_shape) = align_quantized_tensor_for_rocm_gemm(&data, &shape, 4, 64);

        assert_eq!(new_shape, vec![128, 256]); // Already aligned
        assert_eq!(padded.len(), data.len());
    }

    #[test]
    fn test_align_quantized_tensor_pads_rows() {
        // 70x60 tensor with 4-bit quantization - 70 not multiple of 64
        let orig_rows = 70;
        let orig_cols = 60;
        let data: Vec<u8> = vec![0xAB; orig_rows * orig_cols / 2];
        let shape = vec![orig_rows, orig_cols];
        let (_padded, new_shape) = align_quantized_tensor_for_rocm_gemm(&data, &shape, 4, 64);

        // Rows should be padded to 128
        assert_eq!(new_shape[0], 128);
        assert_eq!(new_shape[1], orig_cols);
    }
}
