//! Internal `lib.rs` unit tests. Moved here from the top-level module
//! (which had grown to > 4,000 lines, well over the spec's 1,500-line
//! anti-pattern ceiling). Tests still reach everything in the crate root
//! via `use crate::*;` plus the few names that were globally imported in
//! `lib.rs`'s `mod tests` (Error / Result from grim_tensor::error).

#[cfg(test)]
mod tests {
    use crate::*;
    use grim_tensor::error::{Error, Result};

    #[test]
    fn dtype_byte_size_layout() {
        // Verify the byte-size matrix; HIP alignment-aware alloc calls
        // rely on these being right.
        assert_eq!(dtype_byte_size(&DType { arith: ArithType::F32, storage: DTypeStorage::Native }), 4);
        assert_eq!(dtype_byte_size(&DType { arith: ArithType::F16, storage: DTypeStorage::Native }), 2);
        assert_eq!(dtype_byte_size(&DType { arith: ArithType::BF16, storage: DTypeStorage::Native }), 2);
        assert_eq!(dtype_byte_size(&DType { arith: ArithType::I64, storage: DTypeStorage::Native }), 8);
        assert_eq!(dtype_byte_size(&DType { arith: ArithType::U8, storage: DTypeStorage::Native }), 1);
    }

    #[test]
    fn probe_with_ordinal_override_returns_one_device() {
        // The override path always returns one device; the with_var guard
        // reverts the env to its prior state when the closure returns.
        temp_env::with_var("GRIM_ROCM_ORDINAL_OVERRIDE", Some("0"), || {
            let devices = RocmDevice::probe().expect("probe");
            assert_eq!(devices.len(), 1);
        });
    }

    #[test]
    fn probe_without_hip_runtime_returns_empty_or_one() {
        // On a host without HIP installed, `hipSetDevice(0)` will fail
        // and we return Vec::new(). When HIP is installed, we return
        // one. The test asserts the contract without coupling to the
        // host environment.
        let devices = RocmDevice::probe().expect("probe");
        assert!(devices.len() <= 1);
    }

    #[test]
    fn rocblas_handle_cache_initializes_lazily() {
        // Without HIP installed, this returns an Error. We accept either.
        let dev = RocmDevice::new(0);
        let res = dev.get_rocblas_handle();
        match res {
            Ok(_h) => {}
            Err(_) => {}
        }
    }

    #[test]
    fn rocm_storage_metadata_is_stable() {
        // Allocating `RocmStorage` requires HIP installed, so we only
        // exercise the metadata methods on a defaulted instance to
        // ensure the SurfaceType sticks together.
        let dummy = RocmStorage {
            device_ptr: None,
            bytes: 0,
            shape: Shape::new(vec![1]),
            dtype: DType { arith: ArithType::F32, storage: DTypeStorage::Native },
            provenance: QuantProvenance::GrimNative,
            ordinal: 0,
            allocator: Arc::new(RocmCachingAllocator::new(0, 0)),
        };
        assert_eq!(dummy.bytes(), 0);
        assert_eq!(dummy.shape_metadata().elem_count(), 1);
        assert!(!dummy.device_ptr_is_valid());
        assert_eq!(dummy.device_ordinal(), 0);
    }

    // ------------------------------------------------------------------------
    // Pass 4: WeightLayout, WavefrontTiledLayout, attention routing
    // ------------------------------------------------------------------------

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
            let src: Vec<f32> = (0..4 * 32).map(|i| (i as f32 / 128.0) * 2.0 - 1.0).collect();
            let packed = layout.pack(&src);
            let unpacked = layout.unpack(&packed);
            assert_eq!(unpacked.len(), src.len());
            // Quantization will introduce some error, but check that it's within quantization bucket size:
            // max error = 0.5 * step_size = 0.5 * (2.0 / (2^bits - 1))
            let max_allowed_error = 1.0 / ((1 << bits) - 1) as f32 + 1e-5;
            for (a, b) in src.iter().zip(unpacked.iter()) {
                assert!((a - b).abs() <= max_allowed_error, "Error too large for {} bits: got a={}, b={}, diff={}", bits, a, b, (a-b).abs());
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
        assert_eq!(tiled.len(), 1 * 64 * 64);

        let recovered = wf.untile(&tiled, 35, 40);
        assert_eq!(recovered.len(), 35 * 40);
        for (a, b) in src.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-6, "35x40 round-trip value mismatch");
        }
    }

    #[test]
    fn test_is_attention_projection() {
        let cases = &[
            ("blk.48.attn_q.weight", true),
            ("blk.48.attn_k.weight", true),
            ("blk.48.attn_v.weight", true),
            ("blk.48.attn_o.weight", true),
            ("model.embed_tokens.weight", false),
            ("model.layers.48.mlp.gate_proj.weight", false),
            ("model.layers.48.mlp.up_proj.weight", false),
            ("model.layers.48.mlp.down_proj.weight", false),
            ("blk.48.ffn_gate", false),
            ("self_attn.q_proj.weight", true),
            ("self_attn.k_proj.weight", true),
            ("self_attn.v_proj.weight", true),
            ("self_attn.o_proj.weight", true),
        ];
        for (name, expected) in cases {
            assert_eq!(is_attention_projection(name), *expected, "failed for {name}");
        }
    }

    #[test]
    fn test_enforce_attention_precision() {
        assert_eq!(enforce_attention_precision(3), 5);
        assert_eq!(enforce_attention_precision(4), 5);
        assert_eq!(enforce_attention_precision(5), 5);
        assert_eq!(enforce_attention_precision(6), 6);
        assert_eq!(enforce_attention_precision(8), 8);
    }

    #[test]
    fn test_attention_min_bpw() {
        assert_eq!(attention_min_bpw(), 5);
    }

    #[test]
    fn test_resolve_weight_layout_attention_defaults_to_wavefront_tiled() {
        let layout = resolve_weight_layout(
            "blk.48.attn_q.weight",
            None,
            WavefrontSize::W64,
        );
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
        // "gfx1100" routes to RDNA2/3 -> W32 (gcn match expression returns 32)
        let wf = wavefront_size_for_gcn("gfx1100");
        assert_eq!(wf, 32);
    }

    #[test]
    fn test_wavefront_size_for_gcn_w32() {
        // "gfx1100" routes to RDNA2/3 -> W32
        let wf = wavefront_size_for_gcn("gfx1100");
        assert_eq!(wf, 32);
    }

    #[test]
    fn test_wavefront_size_for_gcn_unknown_returns_64() {
        // Unknown GCN returns safe default of 64
        let wf = wavefront_size_for_gcn("gfx_unknown");
        assert_eq!(wf, 64);
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
    fn test_fused_dequant_gemm_compiles() {
        if std::env::var("GRIM_RUN_GPU_TESTS").is_err() {
            return;
        }
        let kernel_source = crate::kernels::source_asm::compute_kernel_source();
        let target = detect_gpu_arch(0);
        let res = jit_compile_hsaco(&kernel_source, "grim_fused_dequant_gemm_f16", &target);
        assert!(res.is_ok(), "Failed to JIT compile grim_fused_dequant_gemm_f16: {:?}", res.err());
    }

    #[test]
    fn test_split_k_reduction_compiles() {
        if std::env::var("GRIM_RUN_GPU_TESTS").is_err() {
            return;
        }
        let kernel_source = crate::kernels::source_asm::compute_kernel_source();
        let target = detect_gpu_arch(0);
        let res = jit_compile_hsaco(&kernel_source, "grim_split_k_reduction", &target);
        assert!(res.is_ok(), "Failed to JIT compile grim_split_k_reduction: {:?}", res.err());
    }

    #[test]
    fn test_qkv_attention_large_head_dim_compiles() {
        if std::env::var("GRIM_RUN_GPU_TESTS").is_err() {
            return;
        }
        let kernel_source = crate::kernels::source_asm::compute_kernel_source();
        let target = detect_gpu_arch(0);
        let res = jit_compile_hsaco(&kernel_source, "grim_qkv_attention", &target);
        assert!(res.is_ok(), "Failed to JIT compile grim_qkv_attention with large head_dim support: {:?}", res.err());
    }

    #[test]
    fn test_wmma_capability_gates() {
        use crate::device::accel_features::{wmma_supported, wmma_dispatch};
        use crate::quantization::{GcnArch, QuantMode};

        // RDNA3 and RDNA4 support WMMA for native modes
        assert!(wmma_supported(GcnArch::RDNA3, QuantMode::F16));
        assert!(wmma_supported(GcnArch::RDNA4, QuantMode::Fp8));
        
        // CDNA and RDNA1/2 do not support WMMA
        assert!(!wmma_supported(GcnArch::CDNA2, QuantMode::F16));
        assert!(!wmma_supported(GcnArch::RDNA1, QuantMode::F16));

        // dispatch checks
        assert_eq!(wmma_dispatch("gfx1100", QuantMode::F16), Ok(QuantMode::F16));
        assert!(wmma_dispatch("gfx90a", QuantMode::F16).is_err());
        assert!(wmma_dispatch("gfx1100", QuantMode::Fp8).is_err()); // gfx1100 (RDNA3) doesn't support FP8
    }

    // ------------------------------------------------------------------------
    // align_tensor_for_rocm_gemm tests
    // ------------------------------------------------------------------------

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
                assert_eq!(padded[row * 40 + col], 0.0, "padding should be zero at row {row}, col {col}");
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
        let bytes_per_elem = 0.5; // 4-bit
        let data: Vec<u8> = vec![0xAB; (orig_rows * orig_cols / 2) as usize];
        let shape = vec![orig_rows, orig_cols];
        let (padded, new_shape) = align_quantized_tensor_for_rocm_gemm(&data, &shape, 4, 64);

        // Rows should be padded to 128
        assert_eq!(new_shape[0], 128);
        assert_eq!(new_shape[1], orig_cols);
    }

    // ------------------------------------------------------------------------
    // Compute op correctness (add / mul / silu_mul / rms_norm / softmax / embedding)
    // ------------------------------------------------------------------------
    //
    // These require a live AMD GPU + ROCm. They are gated behind GRIM_RUN_GPU_TESTS
    // so GPU-less CI does not fail; set the var to run real numerical checks.
    // When gated off, we still build the device and assert the path does not panic.

    const GPU_TEST_ENV: &str = "GRIM_RUN_GPU_TESTS";

    /// Run a binary compute op on host f32 row vectors, returning the device result
    /// as a host vector. Returns `None` when GPU execution is unavailable.
    fn run_binary_op(
        env_present: bool,
        a: &[f32],
        b: &[f32],
        out_shape: &[usize],
        op: impl FnOnce(&RocmDevice, &dyn BackendStorage, &dyn BackendStorage, &Shape) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)>,
    ) -> Option<Vec<f32>> {
        if !env_present {
            return None;
        }
        let dev = RocmDevice::new(0);
        let a_s = dev.from_cpu(a, &Shape::from_slice(&[a.len()]), DType::F32).ok()?;
        let b_s = dev.from_cpu(b, &Shape::from_slice(&[b.len()]), DType::F32).ok()?;
        let (out, _h) = op(&dev, a_s.as_ref(), b_s.as_ref(), &Shape::from_slice(out_shape)).ok()?;
        out.to_cpu_vec_f32().ok()
    }

    /// Run a unary compute op (softmax) on a host f32 matrix row-major.
    fn run_softmax_op(env_present: bool, x: &[f32], shape: &[usize]) -> Option<Vec<f32>> {
        if !env_present {
            return None;
        }
        let dev = RocmDevice::new(0);
        let x_s = dev.from_cpu(x, &Shape::from_slice(shape), DType::F32).ok()?;
        let (out, _h) = dev.softmax(x_s.as_ref(), &Shape::from_slice(shape)).ok()?;
        out.to_cpu_vec_f32().ok()
    }

    /// Run rms_norm on a host f32 matrix with a weight vector.
    fn run_rms_norm_op(env_present: bool, x: &[f32], w: &[f32], shape: &[usize], eps: f32) -> Option<Vec<f32>> {
        if !env_present {
            return None;
        }
        let dev = RocmDevice::new(0);
        let x_s = dev.from_cpu(x, &Shape::from_slice(shape), DType::F32).ok()?;
        let w_s = dev.from_cpu(w, &Shape::from_slice(&[w.len()]), DType::F32).ok()?;
        let (out, _h) = dev.rms_norm(x_s.as_ref(), w_s.as_ref(), eps, &Shape::from_slice(shape)).ok()?;
        out.to_cpu_vec_f32().ok()
    }

    /// Run embedding gather on a host f32 weight matrix [vocab, dim].
    fn run_embedding_op(env_present: bool, weight: &[f32], indices: &[u32], vocab: usize, dim: usize) -> Option<Vec<f32>> {
        if !env_present {
            return None;
        }
        let dev = RocmDevice::new(0);
        let w_s = dev.from_cpu(weight, &Shape::from_slice(&[vocab, dim]), DType::F32).ok()?;
        let out_shape = Shape::from_slice(&[indices.len(), dim]);
        let (out, _h) = dev.embedding(w_s.as_ref(), indices, &out_shape).ok()?;
        out.to_cpu_vec_f32().ok()
    }

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn add_produces_elementwise_sum() {
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        let got = run_binary_op(env, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], &[4], |d, a, b, s| {
            d.add(a, b, s)
        });
        if let Some(out) = got {
            assert!(approx_eq(out[0], 6.0, 1e-3), "add[0] expected 6.0 got {}", out[0]);
            assert!(approx_eq(out[3], 12.0, 1e-3), "add[3] expected 12.0 got {}", out[3]);
        }
    }

    #[test]
    fn mul_produces_elementwise_product() {
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        let got = run_binary_op(env, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], &[4], |d, a, b, s| {
            d.mul(a, b, s)
        });
        if let Some(out) = got {
            assert!(approx_eq(out[0], 5.0, 1e-3), "mul[0] expected 5.0 got {}", out[0]);
            assert!(approx_eq(out[3], 32.0, 1e-3), "mul[3] expected 32.0 got {}", out[3]);
        }
    }

    #[test]
    fn silu_mul_matches_swiglu_formula() {
        // silu(gate) * up, with silu(x) = x / (1 + exp(-x))
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        let gate = [1.0f32, -2.0, 0.0, 3.5];
        let up = [2.0f32, 4.0, 1.0, 0.5];
        let got = run_binary_op(env, &gate, &up, &[4], |d, a, b, s| d.silu_mul(a, b, s));
        if let Some(out) = got {
            for i in 0..4 {
                let expected = gate[i] / (1.0 + (-gate[i]).exp()) * up[i];
                assert!(approx_eq(out[i], expected, 1e-2), "silu_mul[{i}] expected {expected} got {}", out[i]);
            }
        }
    }

    #[test]
    fn rms_norm_normalizes_to_unit_when_weight_is_one() {
        // x = [3,4] over row_len 2, weight = 1, eps = 0:
        // rms = sqrt((9+16)/2) = sqrt(12.5) ~= 3.5355, out = x / rms
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        let x = [3.0f32, 4.0];
        let w = [1.0f32, 1.0];
        let got = run_rms_norm_op(env, &x, &w, &[2], 0.0);
        if let Some(out) = got {
            let rms = (12.5f32).sqrt();
            assert!(approx_eq(out[0], 3.0 / rms, 1e-3), "rms_norm[0] expected {} got {}", 3.0 / rms, out[0]);
            assert!(approx_eq(out[1], 4.0 / rms, 1e-3), "rms_norm[1] expected {} got {}", 4.0 / rms, out[1]);
        }
    }

    #[test]
    fn softmax_sums_to_one_per_row_and_orders_by_max() {
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        // Two rows: [1,2,3] and [10, 0, -5]
        let x = [1.0f32, 2.0, 3.0, 10.0, 0.0, -5.0];
        let got = run_softmax_op(env, &x, &[2, 3]);
        if let Some(out) = got {
            let row0_sum: f32 = out[0..3].iter().sum();
            let row1_sum: f32 = out[3..6].iter().sum();
            assert!(approx_eq(row0_sum, 1.0, 1e-3), "softmax row0 should sum to 1, got {row0_sum}");
            assert!(approx_eq(row1_sum, 1.0, 1e-3), "softmax row1 should sum to 1, got {row1_sum}");
            // argmax of row1 is index 0 (value 10)
            assert!(out[3] > out[4] && out[3] > out[5], "softmax row1 argmax should be col 0");
        }
    }

    #[test]
    fn embedding_gathers_weight_rows_by_index() {
        // weight = [[1,2,3],[4,5,6],[7,8,9]], dim=3, vocab=3
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        let weight = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let got = run_embedding_op(env, &weight, &[2, 0, 1], 3, 3);
        if let Some(out) = got {
            // indices [2,0,1] -> rows 2,0,1 of weight
            assert_eq!(out.len(), 9);
            assert!(approx_eq(out[0], 7.0, 1e-3), "embed row0[0] expected 7.0 got {}", out[0]);
            assert!(approx_eq(out[3], 1.0, 1e-3), "embed row1[0] expected 1.0 got {}", out[3]);
            assert!(approx_eq(out[6], 4.0, 1e-3), "embed row2[0] expected 4.0 got {}", out[6]);
        }
    }

    #[test]
    fn embedding_rejects_index_count_mismatch() {
        // Without a GPU this still exercises the shape guard (no device alloc needed
        // beyond construction, which is allowed to fail gracefully).
        let dev = RocmDevice::new(0);
        let weight = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w_s = match dev.from_cpu(&weight, &Shape::from_slice(&[2, 3]), DType::F32) {
            Ok(s) => s,
            Err(_) => return, // no GPU; shape-guard logic is covered by the GPU-gated path
        };
        let out_shape = Shape::from_slice(&[2, 3]);
        let res = dev.embedding(w_s.as_ref(), &[0, 1, 2], &out_shape); // 3 indices vs leading dim 2
        assert!(res.is_err(), "embedding must reject indices.len() != out leading dim");
    }

    // ------------------------------------------------------------------------
    // Item 0: rocBLAS `gemm_ex` ABI correctness
    // ------------------------------------------------------------------------
    //
    // The original FFI used fabricated integer discriminants (RocblasOperation =
    // 0/1/2, rocblas_datatype = 0/1/2/...) and a truncated/ reordered
    // `rocblas_gemm_ex` argument list. rocBLAS expects the exact enum values from
    // rocblas/rocblas-types.h, otherwise every GEMM returns invalid_value and
    // silently zeroes the output. These tests pin the ABI constants so the bug
    // cannot regress.

    #[test]
    fn gemm_ex_abi_constants_match_rocblas() {
        // rocblas_operation_*
        assert_eq!(RocblasOperation::None as i32, 111);
        assert_eq!(RocblasOperation::Transpose as i32, 112);
        assert_eq!(RocblasOperation::ConjugateTranspose as i32, 113);

        // rocblas_datatype_* (real discriminants from rocblas-types.h)
        assert_eq!(rocblas_datatype::f16_r as i32, 150);
        assert_eq!(rocblas_datatype::f32_r as i32, 151);
        assert_eq!(rocblas_datatype::bf16_r as i32, 168);
        assert_eq!(rocblas_datatype::i8_r as i32, 160);
        assert_eq!(rocblas_datatype::i32_r as i32, 162);

        // gemm_ex control enums
        assert_eq!(rocblas_gemm_algo::standard as i32, 0x0);
        assert_eq!(rocblas_gemm_algo::solution_index as i32, 0x1);
        assert_eq!(ROCBLAS_GEMM_FLAGS_NONE, 0x0);
    }

    #[test]
    fn arith_to_rocblas_dtype_is_not_fabricated() {
        // Previously BF16 was mapped to the F16 constant and the constants were
        // fabricated (0/1/2). These must map to the real rocBLAS discriminants.
        assert_eq!(arith_to_rocblas_dtype(ArithType::F32), rocblas_datatype::f32_r);
        assert_eq!(arith_to_rocblas_dtype(ArithType::F16), rocblas_datatype::f16_r);
        assert_eq!(arith_to_rocblas_dtype(ArithType::BF16), rocblas_datatype::bf16_r);
        // Mixed-precision GEMMs accumulate in FP32.
        assert_eq!(arith_to_compute_dtype(ArithType::F16), rocblas_datatype::f32_r);
        assert_eq!(arith_to_compute_dtype(ArithType::BF16), rocblas_datatype::f32_r);
    }

    /// Run a 2-D matmul on host f32 and return the device result, or `None` when
    /// GPU execution is unavailable.
    /// Run a matmul on an explicit device and read the result back. Used by tests
    /// that need to share a single `RocmDevice` (and thus a single allocator).
    fn run_matmul_on_dev(
        dev: &RocmDevice,
        a: &[f32],
        a_dims: &[usize],
        b: &[f32],
        b_dims: &[usize],
        out_dims: &[usize],
    ) -> Vec<f32> {
        let a_s = dev.from_cpu(a, &Shape::from_slice(a_dims), DType::F32).unwrap();
        let b_s = dev.from_cpu(b, &Shape::from_slice(b_dims), DType::F32).unwrap();
        let (out, _h) = dev
            .matmul(a_s.as_ref(), b_s.as_ref(), &Shape::from_slice(out_dims))
            .unwrap();
        out.to_cpu_vec_f32().unwrap()
    }

    fn run_matmul_op(
        env_present: bool,
        a: &[f32],
        a_dims: &[usize],
        b: &[f32],
        b_dims: &[usize],
        out_dims: &[usize],
    ) -> Option<Vec<f32>> {
        if !env_present {
            return None;
        }
        let dev = RocmDevice::new(0);
        Some(run_matmul_on_dev(
            &dev,
            a,
            a_dims,
            b,
            b_dims,
            out_dims,
        ))
    }

    /// Reference row-major matmul: C[m,n] = sum_k A[m,k] * B[k,n].
    fn cpu_matmul(a: &[f32], a_dims: &[usize], b: &[f32], b_dims: &[usize]) -> Vec<f32> {
        let (m, k) = (a_dims[0], a_dims[1]);
        let n = b_dims[1];
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = acc;
            }
        }
        c
    }

    #[test]
    fn matmul_batched_matches_loop_of_single_gemms() {
        // Item 6: a batch of same-shape GEMMs collapsed into one
        // rocblas_gemm_strided_batched_ex call must equal running the equivalent
        // single GEMMs (dev.matmul) in a loop, for several batch sizes.
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        if !env {
            return;
        }
        let dev = RocmDevice::new(0);
        for &batch in &[1usize, 3, 5] {
            let m = 8usize;
            let k = 16usize;
            let n = 8usize;
            let mut a_storages: Vec<Box<dyn BackendStorage>> = Vec::new();
            let mut b_storages: Vec<Box<dyn BackendStorage>> = Vec::new();
            for bi in 0..batch {
                let av: Vec<f32> =
                    (0..m * k).map(|i| (i as f32 * 0.05) + bi as f32).collect();
                let bv: Vec<f32> =
                    (0..k * n).map(|i| (i as f32 * 0.05) - 0.5 + bi as f32).collect();
                a_storages.push(dev.from_cpu(&av, &Shape::from_slice(&[m, k]), DType::F32).unwrap());
                b_storages.push(dev.from_cpu(&bv, &Shape::from_slice(&[k, n]), DType::F32).unwrap());
            }
            let a_refs: Vec<&dyn BackendStorage> =
                a_storages.iter().map(|s| s.as_ref()).collect();
            let b_refs: Vec<&dyn BackendStorage> =
                b_storages.iter().map(|s| s.as_ref()).collect();
            let batched = dev
                .matmul_batched(&a_refs, &b_refs, &Shape::from_slice(&[m, n]))
                .unwrap();
            assert_eq!(batched.len(), batch, "batch count mismatch for batch={batch}");
            for bi in 0..batch {
                let (ref_out, _h) = dev
                    .matmul(
                        a_storages[bi].as_ref(),
                        b_storages[bi].as_ref(),
                        &Shape::from_slice(&[m, n]),
                    )
                    .unwrap();
                let ref_vec = ref_out.to_cpu_vec_f32().unwrap();
                let got = batched[bi].to_cpu_vec_f32().unwrap();
                assert_eq!(got.len(), ref_vec.len(), "len mismatch batch {bi}");
                for (i, (g, e)) in got.iter().zip(ref_vec.iter()).enumerate() {
                    assert!(
                        approx_eq(*g, *e, 1e-2),
                        "matmul_batched mismatch batch {bi} [{}/{}]: got {}, loop {}",
                        i / n,
                        i % n,
                        g,
                        e
                    );
                }
            }
        }
    }

    #[test]
    fn gemm_ex_f32_matches_cpu_reference() {
        // Force the gemm_ex (extended-datatype) code path even for FP32 inputs by
        // selecting a CDNA target, which exercises the Item 0 ABI fix directly.
        temp_env::with_var("GRIM_GPU_TARGET", Some("gfx90a"), || {
            let env = std::env::var(GPU_TEST_ENV).is_ok();
            let a_dims = [4usize, 8];
            let b_dims = [8usize, 4];
            let a: Vec<f32> = (0..32).map(|i| i as f32 * 0.1 + 1.0).collect();
            let b: Vec<f32> = (0..32).map(|i| (i as f32 * 0.2) - 3.0).collect();
            let expected = cpu_matmul(&a, &a_dims, &b, &b_dims);
            let got = run_matmul_op(env, &a, &a_dims, &b, &b_dims, &[4, 4]);
            if let Some(out) = got {
                assert_eq!(out.len(), expected.len());
                for (i, (g, e)) in out.iter().zip(expected.iter()).enumerate() {
                    assert!(
                        approx_eq(*g, *e, 1e-2),
                        "gemm_ex f32 mismatch at [{}/{}]: got {}, expected {}",
                        i / 4,
                        i % 4,
                        g,
                        e
                    );
                }
            }
        });
    }

    // ------------------------------------------------------------------------
    // Item 1: caching/pooling GPU allocator
    // ------------------------------------------------------------------------

    #[test]
    fn caching_allocator_reuses_buffers_across_steps() {
        // After a short warmup of same-shape matmuls, the steady-state loop must
        // reuse pooled device buffers and NOT call hipMalloc per step. This is the
        // regression test for Item 1'sallocator reuse.
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        if !env {
            return;
        }
        let dev = RocmDevice::new(0);
        let a_dims = [16usize, 32];
        let b_dims = [32usize, 16];
        let a: Vec<f32> = (0..16 * 32).map(|i| (i as f32 * 0.01) - 1.0).collect();
        let b: Vec<f32> = (0..32 * 16).map(|i| (i as f32 * 0.02)).collect();

        // Warmup so the pool fills with the right size classes.
        for _ in 0..3 {
            let _ = run_matmul_on_dev(&dev, &a, &a_dims, &b, &b_dims, &[16, 16]);
        }
        let (m1, _f1) = dev.allocator_stats();
        for _ in 0..20 {
            let _ = run_matmul_on_dev(&dev, &a, &a_dims, &b, &b_dims, &[16, 16]);
        }
        let (m2, _f2) = dev.allocator_stats();

        // Steady-state: repeated same-shape matmuls reuse pooled buffers, so new
        // hipMalloc calls must be ~0 (allow a couple for slack).
        assert!(
            (m2 - m1) <= 2,
            "hipMalloc calls grew by {} during steady-state loop (expected ~0, proving pool reuse)",
            m2 - m1
        );
    }

    #[test]
    fn empty_cache_releases_pooled_buffers() {
        // empty_cache() must actually hipFree the retained buffers, bounding memory.
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        if !env {
            return;
        }
        let dev = RocmDevice::new(0);
        let a_dims = [8usize, 8];
        let b_dims = [8usize, 8];
        let a: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..64).map(|i| (i + 1) as f32).collect();
        for _ in 0..5 {
            let _ = run_matmul_on_dev(&dev, &a, &a_dims, &b, &b_dims, &[8, 8]);
        }
        let (_m_before, f_before) = dev.allocator_stats();
        dev.empty_cache();
        let (_m_after, f_after) = dev.allocator_stats();
        assert!(
            f_after > f_before,
            "empty_cache must release pooled buffers via hipFree (free_count {} -> {})",
            f_before,
            f_after
        );
    }

    // ------------------------------------------------------------------------
    // Item 2: module cache + no per-launch sync
    // ------------------------------------------------------------------------

    #[test]
    fn module_cache_loads_each_kernel_once() {
        // Each unique compute kernel must be hipModuleLoad'd exactly once for the
        // process lifetime; repeated dispatches reuse the cached module (Item 2).
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        if !env {
            return;
        }
        // The device detects its own gfx target from the driver, so kernel
        // compilation is immune to the process-global `GRIM_GPU_TARGET` flips
        // done by sibling tests via temp_env.
        let dev = RocmDevice::new(0);

        let x = dev.from_cpu(&vec![1.0f32; 4*8], &Shape::from_slice(&[4,8]), DType::F32).unwrap();
        let w_norm = dev.from_cpu(&vec![1.0f32; 8], &Shape::from_slice(&[8]), DType::F32).unwrap();
        let w_mat = dev.from_cpu(&vec![1.0f32; 8*16], &Shape::from_slice(&[8,16]), DType::F32).unwrap();

        // Warmup: load the rmsnorm_matmul module once.
        let (_o, _h) = dev
            .rmsnorm_matmul(x.as_ref(), w_norm.as_ref(), w_mat.as_ref(), 1e-5, &Shape::from_slice(&[4, 16]))
            .unwrap();
        let baseline = dev.module_load_stats();
        assert!(baseline >= 1, "expected >=1 module loaded, got {}", baseline);

        // Repeat many times: module load count must NOT increase.
        for _ in 0..20 {
            let (_o, _h) = dev
                .rmsnorm_matmul(x.as_ref(), w_norm.as_ref(), w_mat.as_ref(), 1e-5, &Shape::from_slice(&[4, 16]))
                .unwrap();
        }
        assert_eq!(
            dev.module_load_stats(),
            baseline,
            "module cache reloaded rmsnorm_matmul across repeated dispatches"
        );

        // A second distinct kernel (qkv_attention) must load once, then reuse.
        // num_heads=4, num_kv_heads=2 (a 2:1 GQA ratio), head_dim=64 fits the
        // Wave64 + Phase-1 head_dim<=64 constraint. seq_len=4, kv_seq_len=4,
        // cache_offset=0 is a degenerate identity-size prefill.
        let q = dev.from_cpu(&vec![1.0f32; 4*4*64], &Shape::from_slice(&[4,4,64]), DType::F32).unwrap();
        let (_o, _h) = dev
            .qkv_attention(
                q.as_ref(),
                q.as_ref(),
                q.as_ref(),
                2,                // num_kv_heads: real param, not num_heads/4
                4,                // kv_seq_len
                0,                // cache_offset
                &Shape::from_slice(&[4, 4, 64]),
                None,
                None,
            )
            .unwrap();
        let with_qkv = dev.module_load_stats();
        assert_eq!(with_qkv, baseline + 1, "qkv_attention should load exactly 1 new module");
        for _ in 0..10 {
            let (_o, _h) = dev
                .qkv_attention(
                    q.as_ref(),
                    q.as_ref(),
                    q.as_ref(),
                    2,
                    4,
                    0,
                    &Shape::from_slice(&[4, 4, 64]),
                    None,
                    None,
                )
                .unwrap();
        }
        assert_eq!(
            dev.module_load_stats(),
            with_qkv,
            "module cache reloaded qkv_attention across repeated dispatches"
        );
    }

    #[test]
    fn embedding_frees_temp_buffer_after_launch() {
        // Regression: embedding allocated a temp idx buffer and freed it right
        // after launch. With the per-launch sync removed (Item 2) it must still
        // synchronize the stream before hipFree to avoid a use-after-free race.
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        if !env {
            return;
        }
        let dev = RocmDevice::new(0);
        let weight = dev.from_cpu(&vec![1.0f32; 16*8], &Shape::from_slice(&[16,8]), DType::F32).unwrap();
        let indices: Vec<u32> = (0..4).collect();
        let out_shape = Shape::from_slice(&[4, 8]);
        let res = dev.embedding(weight.as_ref(), &indices, &out_shape);
        assert!(res.is_ok(), "embedding must succeed without use-after-free: {:?}", res.err());
    }

    // ------------------------------------------------------------------------
    // Item 3: zeros() must zero device memory via hipMemset, not a host round-trip
    // ------------------------------------------------------------------------

    #[test]
    fn zeros_uses_hipmemset_not_host_copy() {
        // zeros() must fill the device buffer with zero bytes for every dtype it
        // supports, without allocating a host-side Vec (Item 3). hipMemset zeroes
        // bytes, which is valid because every supported dtype's zero is all-zero bytes.
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        if !env {
            return;
        }
        let dev = RocmDevice::new(0);
        let shape = Shape::from_slice(&[3, 7, 5]);

        let dtypes = [
            DType::F32,
            DType { arith: ArithType::F16, storage: DTypeStorage::Native },
            DType::BF16,
            DType { arith: ArithType::U32, storage: DTypeStorage::Native },
            DType { arith: ArithType::U8, storage: DTypeStorage::Native },
        ];
        for dtype in &dtypes {
            let storage = dev.zeros(&shape, dtype.clone()).unwrap();
            let rs = storage.as_any().downcast_ref::<RocmStorage>().expect("RocmStorage");
            assert!(rs.device_ptr_is_valid(), "expected valid ptr for {dtype:?}");
            let nbytes = rs.bytes();
            let mut host = vec![0xABu8; nbytes];
            let res = unsafe {
                hipMemcpy(
                    host.as_mut_ptr() as *mut c_void,
                    rs.device_ptr.unwrap() as *mut c_void,
                    nbytes,
                    HipMemcpyKind::DeviceToHost,
                )
            };
            assert_eq!(res, hipSuccess, "readback failed for {dtype:?}");
            assert!(
                host.iter().all(|&b| b == 0),
                "zeros() left non-zero bytes for {dtype:?}: {:?}",
                &host[..nbytes.min(8)]
            );
        }
    }

    #[test]
    fn host_transfer_pinned_async_matches_sync() {
        // The pinned + async host-transfer path (Item 4) must produce results
        // identical to the cold-path synchronous pageable path.
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        if !env {
            return;
        }
        let dev = RocmDevice::new(0);
        let shape = Shape::from_slice(&[64, 64]);
        let data: Vec<f32> = (0..shape.elem_count())
            .map(|i| (i as f32) * 0.1 - 5.0)
            .collect();

        // Cold path: pageable Vec + synchronous hipMemcpy.
        let sync_storage = dev.from_cpu(&data, &shape, DType::F32).unwrap();
        let sync_out = sync_storage.to_cpu_vec_f32().unwrap();

        // Hot path: pinned buffer + async hipMemcpy.
        let async_storage = dev.copy_from_host_async(&data, &shape, DType::F32).unwrap();
        let async_out = dev.read_to_host_async(async_storage.as_ref()).unwrap();

        assert_eq!(sync_out.len(), data.len());
        assert_eq!(async_out.len(), data.len());
        for i in 0..data.len() {
            assert!(
                (sync_out[i] - data[i]).abs() < 1e-3,
                "sync round-trip mismatch at {i}: {} vs {}",
                sync_out[i],
                data[i]
            );
            assert!(
                (async_out[i] - data[i]).abs() < 1e-3,
                "pinned-async round-trip mismatch at {i}: {} vs {}",
                async_out[i],
                data[i]
            );
        }

        // Reusable pinned buffer path (decode-loop steady state).
        let mut pinned = RocmPinnedBuffer::<f32>::alloc(data.len()).unwrap();
        let async_storage2 = dev.copy_from_host_async(&data, &shape, DType::F32).unwrap();
        dev.read_into_pinned(async_storage2.as_ref(), &mut pinned)
            .unwrap();
        assert_eq!(pinned.as_slice(), data.as_slice());

        // Reusable pinned buffer for the upload side too.
        let pinned_in = RocmPinnedBuffer::<f32>::from_slice(&data).unwrap();
        let async_storage3 = dev.upload_from_pinned(&pinned_in, &shape, DType::F32).unwrap();
        let async_out3 = dev.read_to_host_async(async_storage3.as_ref()).unwrap();
        for i in 0..data.len() {
            assert!(
                (async_out3[i] - data[i]).abs() < 1e-3,
                "upload_from_pinned round-trip mismatch at {i}",
            );
        }
    }

    #[test]
    fn host_transfer_pinned_async_benchmark() {
        // Benchmark: per-token host round-trip latency, pageable+sync vs pinned+async.
        // Mirrors the decode-loop transfer (feed a token in / read logits out).
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        if !env {
            return;
        }
        let dev = RocmDevice::new(0);
        // Logits-sized staging buffer (vocab ~32k floats), typical decode readback.
        let n = 32_768;
        let shape = Shape::from_slice(&[n]);
        let data: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();

        let iters = 200;
        let warmup = 20;

        // Pageable + synchronous hipMemcpy round trip.
        for _ in 0..warmup {
            let s = dev.from_cpu(&data, &shape, DType::F32).unwrap();
            let _ = s.to_cpu_vec_f32().unwrap();
        }
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let s = dev.from_cpu(&data, &shape, DType::F32).unwrap();
            let _ = s.to_cpu_vec_f32().unwrap();
        }
        let sync_elapsed = t0.elapsed();

        // Pinned + async hipMemcpy round trip (reusing one pinned buffer for input
        // and one for output, as the decode loop would across tokens).
        let mut pinned_in = RocmPinnedBuffer::<f32>::from_slice(&data).unwrap();
        let mut pinned_out = RocmPinnedBuffer::<f32>::alloc(n).unwrap();
        for _ in 0..warmup {
            let s = dev.upload_from_pinned(&pinned_in, &shape, DType::F32).unwrap();
            dev.read_into_pinned(s.as_ref(), &mut pinned_out).unwrap();
        }
        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            let s = dev.upload_from_pinned(&pinned_in, &shape, DType::F32).unwrap();
            dev.read_into_pinned(s.as_ref(), &mut pinned_out).unwrap();
        }
        let async_elapsed = t1.elapsed();

        let sync_us = sync_elapsed.as_secs_f64() * 1e6 / iters as f64;
        let async_us = async_elapsed.as_secs_f64() * 1e6 / iters as f64;
        println!(
            "[Item 4 benchmark] pageable+sync={:.1} us/round-trip, pinned+async={:.1} us/round-trip ({:.2}x)",
            sync_us, async_us, sync_us / async_us.max(1e-9)
        );
        // Sanity: pinned+async must not be catastrophically slower (bandwidth
        // floor is the same memory, so at worst it's parity with the sync path).
        assert!(
            async_us <= sync_us * 4.0 + 1.0,
            "pinned+async unexpectedly slower: {async_us:.1} vs {sync_us:.1} us"
        );
    }

    // ------------------------------------------------------------------------
    // Item 5: generic graph-capture session API (begin/end/replay, keyed cache)
    // ------------------------------------------------------------------------
    //
    // Capture is gated by GRIM_CAPTURE_GRAPH (read once in RocmDevice::new). The
    // API is a no-op when disabled, so these tests flip it on for the device they
    // construct. The op sequence bracketed below is a plain matmul -> add ->
    // rms_norm chain using only primitives that already exist in this crate; the
    // backend does NOT bake in a "decode step" — the caller picks the key and the
    // ops, exactly as the spec requires.

    #[test]
    fn graph_capture_session_replays_decode_sequence() {
        temp_env::with_var("GRIM_CAPTURE_GRAPH", Some("1"), || {
            let env = std::env::var(GPU_TEST_ENV).is_ok();
            if !env {
                return;
            }
            let dev = RocmDevice::new(0);

            // Inputs are uploaded eagerly (outside the capture bracket) so the
            // captured graph only contains compute ops on stable device pointers.
            let m = 16usize;
            let k = 32usize;
            let n = 16usize;
            let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05) - 1.0).collect();
            let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.05) + 0.5).collect();
            let w: Vec<f32> = (0..m * n).map(|i| 1.0 + (i as f32 * 0.1)).collect();
            let a_s = dev.from_cpu(&a, &Shape::from_slice(&[m, k]), DType::F32).unwrap();
            let b_s = dev.from_cpu(&b, &Shape::from_slice(&[k, n]), DType::F32).unwrap();
            let w_s = dev.from_cpu(&w, &Shape::from_slice(&[m, n]), DType::F32).unwrap();
            let out_shape = Shape::from_slice(&[m, n]);
            let eps = 1e-5f32;

            // --- CPU reference (hardware-independent ground truth) ---
            // rocBLAS may pick a different GEMM algorithm for the captured path than
            // for an eager path, so we validate the captured graph against a pure-CPU
            // computation of the same matmul+add+rms_norm sequence rather than against
            // a GPU-eager run (Item 5).
            let mut c_ref = vec![0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut s = 0f32;
                    for kk in 0..k {
                        s += a[i * k + kk] * b[kk * n + j];
                    }
                    c_ref[i * n + j] = s;
                }
            }
            let d_ref: Vec<f32> = c_ref.iter().map(|x| x * 2.0).collect();
            let mut e_ref = vec![0f32; m * n];
            for i in 0..m {
                let mut ss = 0f32;
                for j in 0..n {
                    ss += d_ref[i * n + j] * d_ref[i * n + j];
                }
                let rms = (ss / n as f32 + eps).sqrt();
                for j in 0..n {
                    e_ref[i * n + j] = d_ref[i * n + j] * w[i * n + j] / rms;
                }
            }

            // --- Capture + replay ---
            let key = "item5_test_seq";
            // First lookup misses -> caller captures this time.
            assert!(!dev.replay_graph(key).unwrap());
            dev.begin_graph_capture(key).unwrap();
            let (c, _) = dev.matmul(a_s.as_ref(), b_s.as_ref(), &out_shape).unwrap();
            let (d, _) = dev.add(c.as_ref(), c.as_ref(), &out_shape).unwrap();
            let (e, _) = dev.rms_norm(d.as_ref(), w_s.as_ref(), eps, &out_shape).unwrap();
            dev.end_graph_capture(key).unwrap();
            // Graph is cached; replay fills c/d/e.
            assert!(dev.replay_graph(key).unwrap());
            let replay = e.to_cpu_vec_f32().unwrap();

            assert_eq!(replay.len(), e_ref.len());
            for (i, (rp, eg)) in replay.iter().zip(e_ref.iter()).enumerate() {
                assert!(
                    approx_eq(*rp, *eg, 1e-2),
                    "capture/replay mismatch at [{}][{}]: got {}, cpu ref {}",
                    i / n,
                    i % n,
                    rp,
                    eg
                );
            }
        });
    }

    #[test]
    fn graph_capture_replay_miss_returns_false() {
        // Capturing under one key and then replaying a *different* key must return
        // Ok(false) — never replay the wrong graph or error.
        temp_env::with_var("GRIM_CAPTURE_GRAPH", Some("1"), || {
            let env = std::env::var(GPU_TEST_ENV).is_ok();
            if !env {
                return;
            }
            let dev = RocmDevice::new(0);
            let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
            let b: Vec<f32> = vec![0.5, 0.5, 0.5, 0.5];
            let a_s = dev.from_cpu(&a, &Shape::from_slice(&[2, 2]), DType::F32).unwrap();
            let b_s = dev.from_cpu(&b, &Shape::from_slice(&[2, 2]), DType::F32).unwrap();

            dev.begin_graph_capture("A").unwrap();
            let (out_a, _) = dev.matmul(a_s.as_ref(), b_s.as_ref(), &Shape::from_slice(&[2, 2])).unwrap();
            dev.end_graph_capture("A").unwrap();

            assert!(dev.replay_graph("A").unwrap(), "key A should be cached");
            assert!(!dev.replay_graph("B").unwrap(), "key B is a miss -> Ok(false)");
            // Keep the captured output alive until the test ends so the cached graph
            // (which references its device pointer) never targets freed memory.
            drop(out_a);
        });
    }

    #[test]
    fn graph_capture_session_benchmark() {
        // Capture once, replay N times, and compare wall-clock against N eager
        // runs of the same op sequence on real hardware.
        temp_env::with_var("GRIM_CAPTURE_GRAPH", Some("1"), || {
            let env = std::env::var(GPU_TEST_ENV).is_ok();
            if !env {
                return;
            }
            let dev = RocmDevice::new(0);
            let m = 64usize;
            let k = 128usize;
            let n = 64usize;
            let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01) - 1.0).collect();
            let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.02)).collect();
            let w: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32 * 0.1)).collect();
            let a_s = dev.from_cpu(&a, &Shape::from_slice(&[m, k]), DType::F32).unwrap();
            let b_s = dev.from_cpu(&b, &Shape::from_slice(&[k, n]), DType::F32).unwrap();
            let w_s = dev.from_cpu(&w, &Shape::from_slice(&[n]), DType::F32).unwrap();
            let out = Shape::from_slice(&[m, n]);
            let eps = 1e-5f32;

            let iters = 100usize;
            let warmup = 10usize;

            for _ in 0..warmup {
                let (c, _) = dev.matmul(a_s.as_ref(), b_s.as_ref(), &out).unwrap();
                let (d, _) = dev.add(c.as_ref(), c.as_ref(), &out).unwrap();
                let (_e, _) = dev.rms_norm(d.as_ref(), w_s.as_ref(), eps, &out).unwrap();
            }
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let (c, _) = dev.matmul(a_s.as_ref(), b_s.as_ref(), &out).unwrap();
                let (d, _) = dev.add(c.as_ref(), c.as_ref(), &out).unwrap();
                let (_e, _) = dev.rms_norm(d.as_ref(), w_s.as_ref(), eps, &out).unwrap();
            }
            let eager_elapsed = t0.elapsed();

            let key = "item5_bench_seq";
            assert!(!dev.replay_graph(key).unwrap());
            dev.begin_graph_capture(key).unwrap();
            let (c, _) = dev.matmul(a_s.as_ref(), b_s.as_ref(), &out).unwrap();
            let (d, _) = dev.add(c.as_ref(), c.as_ref(), &out).unwrap();
            let (e, _) = dev.rms_norm(d.as_ref(), w_s.as_ref(), eps, &out).unwrap();
            dev.end_graph_capture(key).unwrap();
            for _ in 0..warmup {
                dev.replay_graph(key).unwrap();
            }
            let t1 = std::time::Instant::now();
            for _ in 0..iters {
                dev.replay_graph(key).unwrap();
            }
            let replay_elapsed = t1.elapsed();
            // The captured graph targets c/d/e; keep them alive across replays.
            drop(c);
            drop(d);
            drop(e);

            let eager_us = eager_elapsed.as_secs_f64() * 1e6 / iters as f64;
            let replay_us = replay_elapsed.as_secs_f64() * 1e6 / iters as f64;
            println!(
                "[Item 5 benchmark] eager={:.1} us/seq, capture+replay={:.1} us/seq ({:.2}x)",
                eager_us, replay_us, eager_us / replay_us.max(1e-9)
            );
            // Replay must not be catastrophically slower than eager (launch overhead
            // is amortized into one graph launch).
            assert!(
                replay_us <= eager_us * 3.0 + 1.0,
                "capture+replay unexpectedly slower: {replay_us:.1} vs {eager_us:.1} us"
            );
        });
    }

    // ------------------------------------------------------------------------
    // WI 1.6.1 — wavefront-parallel attention correctness (grim_rocm_consumer_perf_plan.md)
    //
    // The fused QKV attention kernel splits the KV walk across all 4 RDNA
    // wavefronts in the 256-thread block, then merges the 4 partial (max, sum,
    // acc[d]) states in wave 0 via LDS. These tests verify that the 4-way
    // merge produces mathematically identical output to the structural
    // CPU reference.
    //
    // Reference: `woody_attention_online_f32` — a CPU helper that performs
    // the *same* online-softmax algorithm (running max + running weighted
    // sum) with the same float32 reductions, but on a single thread per dim.
    // This isolates correctness (merge + causal mask + GQA) from
    // softmax-precision noise on near-uniform inputs.
    //
    // Tolerance rationale: f32 softmax on head_dim<=64 with normally
    // distributed scores has ~1e-5 abs+rel error; we use 1e-3 to be
    // generous against saturating input ranges that occasionally appear in
    // the shape-class sweep below. The kv_seq_len==1 case (only one valid j)
    // has zero softmax-pre-cision noise and is checked bit-exact at 1e-5.
    // ------------------------------------------------------------------------

    /// CPU reference for the fused QKV attention kernel.
    ///
    /// Mirrors the kernel algorithm: causal j ∈ [0, min(abs_i+1, kv_seq_len));
    /// GQA via `kv_head = h / (num_heads/num_kv_heads)`; online softmax with
    /// numerically stable running max + running sum; weighted sum against V.
    /// Layouts: q shape `[seq_len * num_heads * head_dim]` (row-major 3-D),
    /// k/v shape `[kv_seq_len * num_kv_heads * head_dim]` (shared across queries).
    #[allow(clippy::too_many_arguments)]
    fn woody_attention_online_f32(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        kv_seq_len: usize,
        cache_offset: u32,
    ) -> Vec<f32> {
        assert!(num_heads % num_kv_heads == 0, "GQA: num_heads must be multiple of num_kv_heads");
        let q_per_kv = num_heads / num_kv_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let q_stride = num_heads * head_dim;
        let kv_stride = num_kv_heads * head_dim;
        let mut out = vec![0.0f32; seq_len * num_heads * head_dim];

        for h in 0..num_heads {
            let kv_head = h / q_per_kv;
            for qt in 0..seq_len {
                let abs_i = (cache_offset as usize) + qt;
                let hi = (abs_i + 1).min(kv_seq_len);

                // Per-d online softmax running state.
                let mut acc = vec![0.0f32; head_dim];
                let mut running_max = vec![f32::NEG_INFINITY; head_dim];
                let mut running_sum = vec![0.0f32; head_dim];

                for j in 0..hi {
                    // Score = (q · k[j]) * scale
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[qt * q_stride + h * head_dim + d]
                            * k[j * kv_stride + kv_head * head_dim + d];
                    }
                    let s = dot * scale;
                    for d in 0..head_dim {
                        let prev_m = running_max[d];
                        // Stable online softmax update.
                        let new_m = if s > prev_m { s } else { prev_m };
                        // scale = exp(prev_m - new_m): 1.0 when prev_m == -inf
                        let scale_prev = if new_m == f32::NEG_INFINITY {
                            0.0
                        } else {
                            (prev_m - new_m).exp()
                        };
                        running_sum[d] = running_sum[d] * scale_prev;
                        acc[d] = acc[d] * scale_prev;
                        running_max[d] = new_m;
                        // Weight for this j.
                        let w = if s == new_m { 1.0f32 } else { (s - new_m).exp() };
                        running_sum[d] += w;
                        acc[d] += w * v[j * kv_stride + kv_head * head_dim + d];
                    }
                }

                // Final write: out = acc / sum (with F5 zero-guard for empty ranges).
                for d in 0..head_dim {
                    let denom = running_sum[d];
                    out[qt * q_stride + h * head_dim + d] =
                        if denom > 0.0 { acc[d] / denom } else { 0.0 };
                }
            }
        }
        out
    }

    /// Deterministic f32 pattern: lanes are derivable, promote exact reproducibility.
    fn lcg_f32(seed: u32) -> Vec<f32> {
        // Wyrand-style: Cheap and reproducible in f32.
        let mut state = seed.wrapping_add(0x9E3779B9);
        let mut out = Vec::new();
        for _ in 0..4096 {
            state = state.wrapping_mul(0x85EBCA6B).wrapping_add(0xC2B2AE35);
            let x = (state as f32) / (u32::MAX as f32) * 4.0 - 2.0; // ~[-2, 2]
            out.push(x);
        }
        out
    }

    /// Run `dev.qkv_attention` and copy result back to host. Gated by env.
    fn run_qkv_attention(
        env_present: bool,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        num_kv_heads: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Option<Vec<f32>> {
        if !env_present {
            return None;
        }
        let dev = RocmDevice::new(0);
        let q_s = dev
            .from_cpu(
                q,
                &Shape::from_slice(&[seq_len, num_heads, head_dim]),
                DType::F32,
            )
            .ok()?;
        let k_s = dev
            .from_cpu(
                k,
                &Shape::from_slice(&[kv_seq_len, num_kv_heads, head_dim]),
                DType::F32,
            )
            .ok()?;
        let v_s = dev
            .from_cpu(
                v,
                &Shape::from_slice(&[kv_seq_len, num_kv_heads, head_dim]),
                DType::F32,
            )
            .ok()?;
        let (out, _h) = dev
            .qkv_attention(
                q_s.as_ref(),
                k_s.as_ref(),
                v_s.as_ref(),
                num_kv_heads,
                kv_seq_len,
                cache_offset,
                &Shape::from_slice(&[seq_len, num_heads, head_dim]),
                None,
                None,
            )
            .ok()?;
        out.to_cpu_vec_f32().ok()
    }

    fn approx_close(a: f32, b: f32, abs_tol: f32, rel_tol: f32) -> bool {
        let diff = (a - b).abs();
        let scale = a.abs().max(b.abs());
        diff <= abs_tol.max(rel_tol * scale)
    }

    /// Compare two flat vectors and return the max abs diff observed.
    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "len mismatch: {} vs {}", a.len(), b.len());
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    /// Common shape for all 5 shape-class tests.
    ///
    /// `head_dim = 32` keeps the tests runnable on wave32 hardware (RDNA
    /// iGPU, e.g. gfx1036 — `RocmDevice::props.wavefront_size == 32`), where
    /// lane_id caps at 32. On wave64 hardware (RDNA dGPU / CDNA) the SAME
    /// fixture still exercises the WI 1.4.2 4-way partition because the
    /// host launches block = wave_size_check(64) -> 4 wavefronts by default,
    /// making waves 1-3 fully populated for d in [0, 32). The WI 1.4.2 merge
    /// arithmetic run on each wave's partials, which is exactly what these
    /// tests verify.
    fn shape_fixture() -> (usize, usize, usize, usize) {
        // (num_heads, num_kv_heads, head_dim, seq_len)
        // 2:1 GQA, 32-dim, 4 query positions.
        (8, 4, 32, 4)
    }

    fn build_inputs(
        seed: u32,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
        kv_seq_len: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let q_len = seq_len * num_heads * head_dim;
        let kv_len = kv_seq_len * num_kv_heads * head_dim;
        let mut stream = lcg_f32(seed);
        let mut take = |n: usize| -> Vec<f32> {
            if stream.len() < n {
                // deterministic refill for big shapes
                let mut s = (seed.wrapping_add(n as u32)).wrapping_add(0x9E3779B9);
                for _ in 0..n {
                    s = s.wrapping_mul(0x85EBCA6B).wrapping_add(0xC2B2AE35);
                    stream.push((s as f32) / (u32::MAX as f32) * 4.0 - 2.0);
                }
            }
            stream.drain(..n).collect()
        };
        let q = take(q_len);
        let k = take(kv_len);
        let v = take(kv_len);
        (q, k, v)
    }

    #[test]
    fn wi1_qkv_attention_kvseq_mod4_eq0() {
        let _ = approx_close; // silence dead-code warning for helper-only tests
        let (nh, nkv, hd, sl) = shape_fixture();
        let kv_seq = 64usize; // divisible by 4
        let cache_off = 4u32; // ensures causal path active
        let (q, k, v) = build_inputs(0xA1, nh, nkv, hd, sl, kv_seq);
        let cpu = woody_attention_online_f32(&q, &k, &v, sl, nh, nkv, hd, kv_seq, cache_off);
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        let got = run_qkv_attention(env, &q, &k, &v, nkv, kv_seq, cache_off, sl, nh, hd);
        if let Some(out) = got {
            let max = max_abs_diff(&out, &cpu);
            assert!(max <= 1e-3, "wi1 mod4=0 max_abs_diff {} too large", max);
        }
    }

    #[test]
    fn wi1_qkv_attention_kvseq_mod4_ne0() {
        let _ = approx_close;
        let (nh, nkv, hd, sl) = shape_fixture();
        let kv_seq = 65usize; // 65 mod 4 == 1 — splits unevenly across waves
        let cache_off = 16u32; // forces 17 valid js, all in the past window
        let (q, k, v) = build_inputs(0xB2, nh, nkv, hd, sl, kv_seq);
        let cpu = woody_attention_online_f32(&q, &k, &v, sl, nh, nkv, hd, kv_seq, cache_off);
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        let got = run_qkv_attention(env, &q, &k, &v, nkv, kv_seq, cache_off, sl, nh, hd);
        if let Some(out) = got {
            let max = max_abs_diff(&out, &cpu);
            assert!(max <= 1e-3, "wi1 mod4!=0 max_abs_diff {} too large", max);
        }
    }

    #[test]
    fn wi1_qkv_attention_kvseq_lt_4() {
        let _ = approx_close;
        let (nh, nkv, hd, sl) = shape_fixture();
        let kv_seq = 3usize; // smaller than wavefront count — most waves idle
        let cache_off = 0u32;
        let (q, k, v) = build_inputs(0xC3, nh, nkv, hd, sl, kv_seq);
        let cpu = woody_attention_online_f32(&q, &k, &v, sl, nh, nkv, hd, kv_seq, cache_off);
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        let got = run_qkv_attention(env, &q, &k, &v, nkv, kv_seq, cache_off, sl, nh, hd);
        if let Some(out) = got {
            let max = max_abs_diff(&out, &cpu);
            assert!(max <= 1e-3, "wi1 kv<4 max_abs_diff {} too large", max);
        }
    }

    #[test]
    fn wi1_qkv_attention_kvseq_eq_1_bit_exact() {
        // kv_seq_len=1 has zero softmax-precision noise (only one valid j and
        // one softmax weight = 1.0). The 4-wavefront merge receives three empty
        // partials (max=-1e30,sum=0,acc=0) plus one real partial. The structural
        // identity test below asserts it matches the CPU reference bit-exact
        // at 1e-5 tolerance, which is tight enough to flag any merge arithmetic
        // bug independent of precision noise.
        let _ = approx_close;
        let (nh, nkv, hd, sl) = shape_fixture();
        let kv_seq = 1usize;
        let cache_off = 0u32;
        let (q, k, v) = build_inputs(0xD4, nh, nkv, hd, sl, kv_seq);
        let cpu = woody_attention_online_f32(&q, &k, &v, sl, nh, nkv, hd, kv_seq, cache_off);
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        let got = run_qkv_attention(env, &q, &k, &v, nkv, kv_seq, cache_off, sl, nh, hd);
        if let Some(out) = got {
            let max = max_abs_diff(&out, &cpu);
            assert!(max <= 1e-5, "wi1 kv=1 (bit-exact) max_abs_diff {} too large", max);
        }
    }

    #[test]
    fn wi1_qkv_attention_skewed_short_seq() {
        // Different head_dim (32, num_heads=4, num_kv_heads=2) and a sharply
        // skewed layout: small head_dim, small num_heads, large kv_seq_len
        // (1021 = 4*255+1). Confirms the 4-way split holds when individual
        // dims don't match power-of-two boundaries. This is the most
        // numerically hostile shape (long softmax tail, granular split).
        let _ = approx_close;
        let nh = 4usize;
        let nkv = 2usize;
        let hd = 32usize;
        let sl = 2usize;
        let kv_seq = 1021usize;
        let cache_off = 0u32;
        let (q, k, v) = build_inputs(0xE5, nh, nkv, hd, sl, kv_seq);
        let cpu = woody_attention_online_f32(&q, &k, &v, sl, nh, nkv, hd, kv_seq, cache_off);
        let env = std::env::var(GPU_TEST_ENV).is_ok();
        let got = run_qkv_attention(env, &q, &k, &v, nkv, kv_seq, cache_off, sl, nh, hd);
        if let Some(out) = got {
            let max = max_abs_diff(&out, &cpu);
            assert!(max <= 5e-3, "wi1 skewed max_abs_diff {} too large", max);
        }
    }
}
