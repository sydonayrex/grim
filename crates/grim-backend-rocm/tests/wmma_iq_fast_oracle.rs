use grim_backend_rocm::kernels::wmma_iq_gemm::KERNEL_SOURCE;

#[test]
fn oracle_ieee754_magic_bias_i32_to_f32_is_exact() {
    const MAGIC_I32: i32 = 0x4B40_0000;
    const MAGIC_F32: f32 = 12_582_912.0;
    for x in [-4_194_304i32, -1_000_000, -32768, -1, 0, 1, 42, 32767, 1_000_000, 4_194_303] {
        let bits = (x.wrapping_add(MAGIC_I32)) as u32;
        let fast = f32::from_bits(bits) - MAGIC_F32;
        let exact = x as f32;
        assert_eq!(fast.to_bits(), exact.to_bits());
    }
}

#[test]
fn oracle_iq_sign_bytes_and_two_complement_negate_exact() {
    let iq_sign_bytes = |s4: u32| -> u32 {
        let b = (s4 & 1) | ((s4 & 2) << 7) | ((s4 & 4) << 14) | ((s4 & 8) << 21);
        b.wrapping_mul(0xFF)
    };
    let iq_negate = |grid: u32, m: u32| -> u32 {
        (grid ^ m).wrapping_add(m & 0x0101_0101u32)
    };

    let grid_bytes: [i8; 4] = [3, 17, 45, 113];
    let grid_u32 = u32::from_le_bytes(grid_bytes.map(|b| b as u8));
    for mask4 in 0..16u32 {
        let m = iq_sign_bytes(mask4);
        let neg_u32 = iq_negate(grid_u32, m);
        let out_bytes = neg_u32.to_le_bytes().map(|b| b as i8);
        for lane in 0..4 {
            let want = if (mask4 & (1 << lane)) != 0 {
                -grid_bytes[lane]
            } else {
                grid_bytes[lane]
            };
            assert_eq!(out_bytes[lane], want);
        }
    }
}

#[test]
fn wmma_iq_source_contains_strata_fast_no_scratch_kernels() {
    assert!(KERNEL_SOURCE.contains("grim_wmma_iq4nl_fast"));
    assert!(KERNEL_SOURCE.contains("grim_wmma_iq4xs_fast"));
}
