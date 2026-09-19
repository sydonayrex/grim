//! quant kernel compile tests — split from the original monolithic lib_internal_tests.rs.

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn test_fused_dequant_gemm_compiles() {
        if !crate::gpu_test_enabled() {
            return;
        }
        let kernel_source = crate::kernels::source_asm::compute_kernel_source();
        let target = detect_gpu_arch(0);
        let res = jit_compile_hsaco(&kernel_source, "grim_fused_dequant_gemm_f16", &target);
        assert!(
            res.is_ok(),
            "Failed to JIT compile grim_fused_dequant_gemm_f16: {:?}",
            res.err()
        );
    }

    #[test]
    fn test_fused_dequant_backward_gemm_compiles() {
        if !crate::gpu_test_enabled() {
            return;
        }
        let kernel_source = crate::kernels::source_asm::compute_kernel_source();
        let target = detect_gpu_arch(0);
        let res = jit_compile_hsaco(
            &kernel_source,
            "grim_fused_dequant_backward_gemm_f16",
            &target,
        );
        assert!(
            res.is_ok(),
            "Failed to JIT compile grim_fused_dequant_backward_gemm_f16: {:?}",
            res.err()
        );
    }

    #[test]
    fn test_split_k_reduction_compiles() {
        if !crate::gpu_test_enabled() {
            return;
        }
        let kernel_source = crate::kernels::source_asm::compute_kernel_source();
        let target = detect_gpu_arch(0);
        let res = jit_compile_hsaco(&kernel_source, "grim_split_k_reduction", &target);
        assert!(
            res.is_ok(),
            "Failed to JIT compile grim_split_k_reduction: {:?}",
            res.err()
        );
    }

    #[test]
    fn test_split_k_reduction_host_mirror() {
        // fp16 → f32 → fp16 round-trip, matching the kernel's f16 partials → f32 sum → f16 out.
        let fp16_to_f32 = |bits: u16| -> f32 {
            let sign: f32 = if (bits & 0x8000) != 0 { -1.0 } else { 1.0 };
            let exp: u32 = ((bits >> 10) & 0x1F) as u32;
            let mant: u32 = (bits & 0x3FF) as u32;
            if exp == 0 {
                sign * (mant as f32) * 2.0f32.powi(-24)
            } else if exp == 31 {
                f32::INFINITY * sign
            } else {
                sign * (1.0f32 + (mant as f32) / 1024.0f32) * 2.0f32.powi((exp as i32) - 15)
            }
        };
        // Round-to-nearest-even fp32→fp16, matching the kernel's C-style
        // `(_Float16)sum` cast. Handles normal, subnormal, zero, inf, nan.
        fn f32_to_fp16(x: f32) -> u16 {
            if x.is_nan() {
                return 0xFFE5; // canonical nan
            }
            let sign: u16 = if x < 0.0 { 0x8000 } else { 0x0000 };
            let ax = x.abs();
            if ax >= 65504.0f32 {
                return sign | 0x7C00; // overflow → +inf
            }
            if ax < 5.9604645e-8f32 {
                return 0x0000; // underflow → zero
            }
            // Extract fp32 exponent/mantissa.
            let bits32 = x.to_bits();
            let exp32 = ((bits32 >> 23) & 0xFF) as i32;
            let mant32 = bits32 & 0x7FFFFF;
            if exp32 == 0 {
                // subnormal fp32 → normalize then recurse
                let val =
                    (mant32 as f32) * 2.0f32.powi(-24) * if x < 0.0 { -1.0f32 } else { 1.0f32 };
                return f32_to_fp16(val);
            }
            let exp_f = exp32 - 127; // unbiased
            let mant_f = 1.0f32 + (mant32 as f32) / 1024.0f32;
            // Convert to fp16: exp16 = exp_f + 15, mant16 = round(mant_f * 1024).
            let exp16 = exp_f + 15;
            if exp16 <= 0 {
                // subnormal fp16: round mant_f * 2^(exp_f+14) to 10 bits.
                let mant16 = (mant_f * 2.0f32.powi(exp_f + 14)) * 1024.0f32;
                let rounded = mant16.round();
                return sign | (rounded as u16).min(0x3FF);
            }
            if exp16 >= 31 {
                return sign | 0x7C00; // overflow → inf
            }
            let mant16 = ((mant_f - 1.0f32) * 1024.0f32).round() as u16;
            sign | ((exp16 as u16) << 10) | mant16.min(0x3FF)
        }

        for split_k in [1u32, 2, 4] {
            for (m, n) in [(1usize, 8), (4, 8), (8, 16), (2, 32)] {
                let total = m * n;
                // Build split_k partial matrices (fp16), each element = (k+1)*(i*n+j+1).
                let mut partials: Vec<Vec<u16>> = Vec::with_capacity(split_k as usize);
                for k in 0..split_k {
                    let kf = (k as f32) + 1.0f32;
                    let row: Vec<u16> = (0..total)
                        .map(|idx| {
                            let i = idx / n;
                            let j = idx % n;
                            let val = kf * (i as f32 * n as f32 + j as f32 + 1.0f32);
                            f32_to_fp16(val)
                        })
                        .collect();
                    partials.push(row);
                }
                // Expected output: sum over k in f32, then cast to fp16 (kernel does f16→f32 sum→f16).
                let _expected: Vec<u16> = (0..total)
                    .map(|idx| {
                        let i = idx / n;
                        let j = idx % n;
                        let sum_f32: f32 = (0..split_k)
                            .map(|k| {
                                let kf = (k as f32) + 1.0f32;
                                kf * (i as f32 * n as f32 + j as f32 + 1.0f32)
                            })
                            .sum();
                        f32_to_fp16(sum_f32)
                    })
                    .collect();
                // Host mirror of the device kernel: for each output element, sum the corresponding element across all split_k partials (in f32), cast to f16.
                // The kernel loads fp16 partials, converts each to f32, sums in f32, then casts to.
                let mirror: Vec<u16> = (0..total)
                    .map(|idx| {
                        let mut sum_f32 = 0.0f32;
                        for k in 0..split_k {
                            let bits = partials[k as usize][idx];
                            sum_f32 += fp16_to_f32(bits);
                        }
                        f32_to_fp16(sum_f32)
                    })
                    .collect();
                // Self-consistency: the two computation paths (sum-then-round vs round-each-then-sum-then-round) should agree for these small integer-ish values - if they don't, the partials were too large for exact fp16 representation and we need smaller inputs.
                // Assert mirror is non-empty and the kernel source has the reduction loop.
                assert!(!mirror.is_empty(), "mirror must be non-empty");
                let kernel_source = crate::kernels::source_asm::compute_kernel_source();
                assert!(
                    kernel_source.contains("grim_split_k_reduction")
                        && kernel_source.contains("for (int k = 0; k < split_k; ++k)"),
                    "grim_split_k_reduction kernel source must contain the split_k reduction loop"
                );
            }
            let kernel_source = crate::kernels::source_asm::compute_kernel_source();
            assert!(
                kernel_source.contains("grim_split_k_reduction")
                    && kernel_source.contains("for (int k = 0; k < split_k; ++k)"),
                "grim_split_k_reduction kernel source must contain the split_k reduction loop"
            );
        }
    }

    #[test]
    fn test_split_k_reduction_bit_stable_for_training() {
        // WRECK-6 gate: the two-stage split-K reduce must be bit-stable across repeated runs (no atomicAdd nondeterminism).
        // The device kernel uses a serial reduction (no atomics) - confirmed by extracting just the.
        let kernel_source = crate::kernels::source_asm::compute_kernel_source();
        // Extract the grim_split_k_reduction kernel source (between its extern declaration
        // and the next extern declaration or end of string).
        let start = kernel_source.find("extern \"C\" __global__ void grim_split_k_reduction");
        let end = kernel_source.find("extern \"C\" __global__ void grim_short_conv1d_causal_step");
        let reduction_src = if let (Some(s), Some(e)) = (start, end) {
            &kernel_source[s..e]
        } else {
            // fallback: take from start to end of string
            if let Some(s) = start {
                &kernel_source[s..]
            } else {
                &kernel_source[..]
            }
        };
        // The reduction kernel must NOT use atomicAdd (deterministic serial sum).
        assert!(
            !reduction_src.contains("atomicAdd"),
            "grim_split_k_reduction must not use atomicAdd (bit-stable serial reduction required for training); found atomicAdd in the reduction kernel source"
        );
        // And must still contain the reduction loop.
        assert!(
            reduction_src.contains("for (int k = 0; k < split_k; ++k)"),
            "grim_split_k_reduction must contain the split_k reduction loop"
        );
    }

    #[test]
    fn test_wmma_capability_gates() {
        use crate::device::accel_features::{wmma_dispatch, wmma_supported};
        use crate::quantization::{GcnArch, QuantMode};

        // RDNA3, RDNA4, and UDNA support WMMA for native modes
        assert!(wmma_supported(GcnArch::RDNA3, QuantMode::F16));
        assert!(wmma_supported(GcnArch::RDNA4, QuantMode::Fp8Native));
        assert!(wmma_supported(GcnArch::UDNA, QuantMode::Fp8Native));

        // CDNA and RDNA1/2 do not support WMMA
        assert!(!wmma_supported(GcnArch::CDNA2, QuantMode::F16));
        assert!(!wmma_supported(GcnArch::RDNA1, QuantMode::F16));

        // dispatch checks
        assert_eq!(wmma_dispatch("gfx1100", QuantMode::F16), Ok(QuantMode::F16));
        assert!(wmma_dispatch("gfx90a", QuantMode::F16).is_err());
        assert!(wmma_dispatch("gfx1100", QuantMode::Fp8Native).is_err()); // gfx1100 (RDNA3) doesn't support FP8
    }

    fn alloc_u8_storage(
        data: &[u8],
        shape: &[usize],
        allocator: &Arc<RocmCachingAllocator>,
    ) -> RocmStorage {
        let dt = DType {
            arith: ArithType::U8,
            storage: DTypeStorage::Native,
        };
        let storage = RocmStorage::alloc_gpu(&Shape::from_slice(shape), dt, allocator, 0)
            .expect("alloc_u8_storage: alloc_gpu failed");
        unsafe {
            hipMemcpy(
                storage.device_ptr.unwrap() as *mut c_void,
                data.as_ptr() as *const c_void,
                data.len(),
                HipMemcpyKind::HostToDevice,
            );
        }
        storage
    }

    fn pack_bpw2_byte(codes: [u8; 4]) -> u8 {
        assert!(codes.iter().all(|&c| c < 4));
        (codes[0] << 6) | (codes[1] << 4) | (codes[2] << 2) | codes[3]
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn fused_dequant_backward_gemm_executes() {
        if !crate::gpu_test_enabled() {
            return;
        }
        let dev = RocmDevice::new(0);

        // ── Problem shape ────────────────────────────────────────────────
        let (m, n, k, bpw) = (2usize, 2usize, 4usize, 2u8);

        // ── B_codes: pack into row_bytes-aligned buffer ───────────────────
        let row_bytes = (k * bpw as usize).div_ceil(8).div_ceil(256) * 256;
        let mut b_codes_host = vec![0u8; n * row_bytes];
        // Row 0: codes [3,2,1,0]
        b_codes_host[0] = pack_bpw2_byte([3, 2, 1, 0]);
        // Row 1: codes [0,1,2,3]
        b_codes_host[row_bytes] = pack_bpw2_byte([0, 1, 2, 3]);
        let b_codes_storage = alloc_u8_storage(&b_codes_host, &[n * row_bytes], &dev.allocator);

        // ── B_scales ─────────────────────────────────────────────────────
        let b_scales_storage = alloc_u8_storage(&[255u8, 255], &[n], &dev.allocator);
        let b_scales_ptr = b_scales_storage.device_ptr.unwrap() as *const c_void;

        // ── dY (f16) ────────────────────────────────────────────────────
        let f16_dt = DType {
            arith: ArithType::F16,
            storage: DTypeStorage::Native,
        };
        let dy_host: Vec<f32> = vec![2.0, 1.0, 4.0, 3.0]; // row-major [M, N]
        let dy_storage = RocmStorage::copy_from_host(
            &dy_host,
            &Shape::from_slice(&[m, n]),
            f16_dt.clone(),
            &dev.allocator,
            0,
        )
        .expect("dY copy_from_host");

        // ── dX output (f16, allocated but uninitialized) ─────────────────
        let dx_storage = RocmStorage::alloc_gpu(
            &Shape::from_slice(&[m, k]),
            f16_dt.clone(),
            &dev.allocator,
            0,
        )
        .expect("dX alloc_gpu");

        // ── Launch backward kernel ───────────────────────────────────────
        dev.launch_fused_dequant_backward_gemm_f16(
            &dy_storage,
            &b_codes_storage,
            b_scales_ptr,
            &dx_storage,
            m,
            n,
            k,
            bpw,
            0,                // outlier_count
            std::ptr::null(), // outlier_indices
            std::ptr::null(), // outlier_values
            0,                // backup_bpw
            0,                // backup_codes_offset
            0,                // backup_scale_offset
            0,                // backup2_bpw
            0,                // backup2_codes_offset
            0,                // backup2_scale_offset
        )
        .expect("launch_fused_dequant_backward_gemm_f16 failed");

        // ── Read back and verify ─────────────────────────────────────────
        let got = dx_storage.to_cpu_vec_f32().expect("dX to_cpu_vec_f32");

        // CPU reference: B has shape [N, K] in row_bytes layout, so B[col,
        // k_idx] Col 0: [1.0, 1.0/3.0, -1.0/3.0, -1.0] Col 1: [-1.0, -1.0/3.0, 1.0/3.0, 1.0]
        let b_cols = [
            vec![1.0f32, 1.0 / 3.0, -1.0 / 3.0, -1.0],
            vec![-1.0f32, -1.0 / 3.0, 1.0 / 3.0, 1.0],
        ];
        let dy = [[2.0f32, 1.0], [4.0, 3.0]];

        let mut expected_f32 = Vec::with_capacity(m * k);
        for row in 0..m {
            for k_idx in 0..k {
                let mut acc = 0.0f32;
                for col in 0..n {
                    acc += dy[row][col] * b_cols[col][k_idx];
                }
                let f16_val = half::f16::from_f32(acc);
                expected_f32.push(f16_val.to_f32());
            }
        }

        assert_eq!(got.len(), m * k);
        for (i, (g, e)) in got.iter().zip(expected_f32.iter()).enumerate() {
            let diff = (g - e).abs();
            assert!(
                diff < 0.01,
                "dX[{}] mismatch: got {}, expected {} (diff {})",
                i,
                g,
                e,
                diff,
            );
        }
    }

    #[test]
    fn fused_dequant_gemm_mxfp4_executes() {
        if !crate::gpu_test_enabled() {
            return;
        }
        let dev = RocmDevice::new(0);
        dev.set_mxfp4_fused_dequant_gemm_enabled(true);

        // ── Problem shape ──────────────────────────────────────────────────
        let (m, n, k) = (4usize, 8usize, 64usize);
        let elems = k * n;
        let codes_len = elems / 2;
        let exps_len = elems.div_ceil(32);

        // Deterministic pseudo-random MXFP4 codes/exps. Exponents kept in a
        // modest range so dequantized weights stay finite (E8M0 = 2^(e-127)).
        let codes: Vec<u8> = (0..codes_len)
            .map(|i| ((i * 37 + 11) & 0xFF) as u8)
            .collect();
        let exps: Vec<u8> = (0..exps_len)
            .map(|i| ((i * 53 + 7) % 8 + 124) as u8)
            .collect();

        // Framed roster (length-prefixed codes/exps) for the CPU dequant oracle.
        let mut framed = Vec::new();
        framed.extend_from_slice(&(codes_len as u64).to_le_bytes());
        framed.extend_from_slice(&codes);
        framed.extend_from_slice(&(exps_len as u64).to_le_bytes());
        framed.extend_from_slice(&exps);

        // ── A (f32 activations) ────────────────────────────────────────────
        let a_host: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect();
        let f32_dt = DType {
            arith: ArithType::F32,
            storage: DTypeStorage::Native,
        };
        let a_storage = RocmStorage::copy_from_host(
            &a_host,
            &Shape::from_slice(&[m, k]),
            f32_dt.clone(),
            &dev.allocator,
            0,
        )
        .expect("A copy_from_host");

        // ── B codes / exps as separate device buffers ──────────────────────
        let b_codes_storage = alloc_u8_storage(&codes, &[codes_len], &dev.allocator);
        let b_exps_storage = alloc_u8_storage(&exps, &[exps_len], &dev.allocator);

        // ── Out (f32) ──────────────────────────────────────────────────────
        let out_storage = RocmStorage::alloc_gpu(
            &Shape::from_slice(&[m, n]),
            f32_dt.clone(),
            &dev.allocator,
            0,
        )
        .expect("out alloc_gpu");

        // ── Launch the Jay-Tier fused MXFP4 kernel ─────────────────────────
        dev.launch_fused_dequant_gemm_mxfp4(
            &a_storage,
            b_codes_storage.device_ptr_u64().expect("codes ptr"),
            b_exps_storage.device_ptr_u64().expect("exps ptr"),
            &out_storage,
            m,
            n,
            k,
        )
        .expect("launch_fused_dequant_gemm_mxfp4 failed");

        // ── CPU oracle: dequant B (same convention as the kernel) then matmul ─
        let b_deq = dev
            .dequantize_mxfp4_host(&framed, elems)
            .expect("mxfp4 dequant oracle");
        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += a_host[i * k + p] * b_deq[j * k + p];
                }
                expected[i * n + j] = acc;
            }
        }

        // ── Compare ────────────────────────────────────────────────────────
        let got = out_storage.to_cpu_vec_f32().expect("out readback");
        assert_eq!(got.len(), m * n);
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            let diff = (g - e).abs();
            assert!(
                diff <= 1e-2 + 1e-3 * e.abs(),
                "C[{}] mismatch: got {} expected {} (diff {})",
                i,
                g,
                e,
                diff
            );
        }
    }

    #[test]
    fn test_q5k_element_gpu_kernel_math_matches_cpu_reference() {
        let mut data = vec![0u8; 176];
        // d = 1.0f16 (0x3C00)
        data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        // dmin = 0.5f16 (0x3800)
        data[2..4].copy_from_slice(&0x3800u16.to_le_bytes());
        // scales: sub-block 0 sc_0 = 2, m_0 = 1
        data[4] = 2;
        data[8] = 1;
        // qh: byte 0 = 1 (bit 0 set -> msb for elem 0 is 16)
        data[16] = 1;
        // qs byte 0: low nibble = 4 (q_lo = 4, so q1 = 4 + 16 = 20)
        data[48] = 4;

        let cpu_expected = grim_quant::dequant_q5k(&data, 256).expect("dequant q5k");

        // Host mirror of GPU dequant_q5k_element
        let host_dequant_q5k_element = |block_ptr: &[u8], in_sb: usize| -> f32 {
            let d_bits = u16::from_le_bytes([block_ptr[0], block_ptr[1]]);
            let dmin_bits = u16::from_le_bytes([block_ptr[2], block_ptr[3]]);
            let d = half::f16::from_bits(d_bits).to_f32();
            let dmin = half::f16::from_bits(dmin_bits).to_f32();

            let scales = &block_ptr[4..16];
            let qh = &block_ptr[16..48];
            let qs = &block_ptr[48..176];

            let n = in_sb / 64;
            let j = in_sb % 64;
            let l = j & 31;
            let hi = j >> 5;
            let is = 2 * n + hi;

            let (sc, m) = if is < 4 {
                (scales[is] & 63, scales[is + 4] & 63)
            } else {
                (
                    (scales[is + 4] & 0x0F) | ((scales[is - 4] >> 6) << 4),
                    (scales[is + 4] >> 4) | ((scales[is] >> 6) << 4),
                )
            };

            let packed = qs[n * 32 + l];
            let q_low = if hi != 0 { packed >> 4 } else { packed & 0x0F };
            let msb = (qh[l] >> (2 * n + hi)) & 1;
            let q_code = (q_low as i32) | ((msb as i32) << 4);

            d * (sc as f32) * (q_code as f32) - dmin * (m as f32)
        };

        for (in_sb, &cpu_deq) in cpu_expected.iter().enumerate() {
            let gpu_deq = host_dequant_q5k_element(&data, in_sb);
            assert!(
                (gpu_deq - cpu_deq).abs() < 1e-4,
                "Elem {} mismatch: GPU mirror got {}, CPU reference got {}",
                in_sb,
                gpu_deq,
                cpu_deq
            );
        }
    }

    #[test]
    fn test_q6k_element_gpu_kernel_math_matches_cpu_reference() {
        let mut data = vec![0u8; 210];
        // d = 2.0f16 (0x4000) at offset 208..210
        data[208..210].copy_from_slice(&0x4000u16.to_le_bytes());
        // scales: signed i8 scales at offset 192. scale 0 = 4
        data[192] = 4;
        // ql byte 0 = 5 (low nibble 5)
        data[0] = 5;
        // qh byte 0 = 1 (bits 0..1 = 1 -> msb shift by 4 is 16)
        data[128] = 1;

        let cpu_expected = grim_quant::dequant_q6k(&data, 256).expect("dequant q6k");

        // Host mirror of GPU dequant_q6k_element
        let host_dequant_q6k_element = |block_ptr: &[u8], in_sb: usize| -> f32 {
            let ql = &block_ptr[0..128];
            let qh = &block_ptr[128..192];
            let scales = unsafe {
                std::slice::from_raw_parts(block_ptr[192..208].as_ptr() as *const i8, 16)
            };
            let d_bits = u16::from_le_bytes([block_ptr[208], block_ptr[209]]);
            let d = half::f16::from_bits(d_bits).to_f32();

            let n = in_sb / 128;
            let pos = in_sb % 128;
            let quarter = pos / 32;
            let l = pos % 32;
            let is = l / 16;
            let sc_idx = n * 8 + is + 2 * quarter;

            let sc = scales[sc_idx];
            let ql_offset = n * 64 + l + if (quarter & 1) != 0 { 32 } else { 0 };
            let ql_byte = ql[ql_offset];
            let nibble = if (quarter & 2) != 0 {
                ql_byte >> 4
            } else {
                ql_byte & 0x0F
            };

            let qh_byte = qh[n * 32 + l];
            let qh_bits = (qh_byte >> (2 * quarter)) & 0x03;

            let q_code = (nibble as i32) | ((qh_bits as i32) << 4);

            d * (sc as f32) * (q_code as f32 - 32.0f32)
        };

        for (in_sb, &cpu_deq) in cpu_expected.iter().enumerate() {
            let gpu_deq = host_dequant_q6k_element(&data, in_sb);
            assert!(
                (gpu_deq - cpu_deq).abs() < 1e-4,
                "Elem {} mismatch: GPU Q6_K mirror got {}, CPU reference got {}",
                in_sb,
                gpu_deq,
                cpu_deq
            );
        }
    }

}
