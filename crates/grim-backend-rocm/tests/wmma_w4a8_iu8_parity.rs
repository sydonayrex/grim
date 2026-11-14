use grim_backend_rocm::kernels::wmma_i8_gemm::WMMA_I8_KERNEL_SOURCE;

/// Reference unpack of 8 unsigned 4-bit nibbles into two 32-bit words where each byte holds 16 * nibble.
fn unpack_u4x8_times_16(packed_u32: u32) -> (u32, u32) {
    let lo = (packed_u32 << 4) & 0xF0F0_F0F0u32; // nibbles 0, 2, 4, 6 * 16
    let hi = packed_u32 & 0xF0F0_F0F0u32;        // nibbles 1, 3, 5, 7 * 16
    (lo, hi)
}

/// Ported from Radiance `tests/relayout_parity_test.cpp:108-125`:
/// Pack canonical [N=16, K=16] unsigned 4-bit weights (0..15) into the exact 32-lane RDNA4 WMMA
/// fragment order: lane = n % 16 + ((k % 16) / 8) * 16; byte j (0..4) holds k+j low and k+j+4 high.
fn pack_w4_16x16_for_wmma(w_nk: &[u8; 256]) -> [u32; 32] {
    let mut packed = [0u32; 32];
    for n in 0..16usize {
        for half in 0..2usize {
            let lane = n + half * 16;
            let k_base = half * 8;
            let mut word = 0u32;
            for j in 0..4usize {
                let lo_nibble = (w_nk[n * 16 + k_base + j] & 0xF) as u32;
                let hi_nibble = (w_nk[n * 16 + k_base + j + 4] & 0xF) as u32;
                let byte = lo_nibble | (hi_nibble << 4);
                word |= byte << (j * 8);
            }
            packed[lane] = word;
        }
    }
    packed
}

#[test]
fn wmma_i8_source_has_direct_amdgcn_w4a8_kernel() {
    assert!(
        WMMA_I8_KERNEL_SOURCE.contains("grim_wmma_w4a8_iu8_gfx12"),
        "missing grim_wmma_w4a8_iu8_gfx12 in WMMA_I8_KERNEL_SOURCE"
    );
    assert!(
        WMMA_I8_KERNEL_SOURCE.contains("__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12"),
        "must call direct AMDGCN iu8 intrinsic to allow int8 x uint8"
    );
}

/// Ported from Radiance `relayout_parity_test.cpp:108-125` & `ref_test.cpp:2989-3025`:
/// Verifies every single nibble's WMMA fragment address + 3-op unpack + `>> 4` de-scale
/// against a canonical 16x16x16 signed-int8 x unsigned-u4 matrix multiply.
#[test]
fn radiance_oracle_wmma_16x16x16_fragment_unpack_and_dot_parity() {
    let mut a_mk = [0i8; 256];
    let mut w_nk = [0u8; 256];
    for i in 0..256usize {
        a_mk[i] = ((i as i32 * 13 - 97) % 127) as i8;
        w_nk[i] = ((i * 7 + 3) & 0xF) as u8;
    }

    let b_u4 = pack_w4_16x16_for_wmma(&w_nk);

    // Simulate the 32-lane RDNA4 wave executing the 3-op unpack + iu8 WMMA + `>> 4` shift
    for m in 0..16usize {
        for n in 0..16usize {
            let mut acc_sim: i32 = 0;
            for half in 0..2usize {
                let lane_b = n + half * 16;
                let (lo_u32, hi_u32) = unpack_u4x8_times_16(b_u4[lane_b]);
                let lo_bytes = lo_u32.to_le_bytes();
                let hi_bytes = hi_u32.to_le_bytes();
                let k_base = half * 8;
                for j in 0..4usize {
                    acc_sim += (a_mk[m * 16 + k_base + j] as i32) * (lo_bytes[j] as i32);
                    acc_sim += (a_mk[m * 16 + k_base + j + 4] as i32) * (hi_bytes[j] as i32);
                }
            }
            let got = acc_sim >> 4;
            let want: i32 = (0..16)
                .map(|k| (a_mk[m * 16 + k] as i32) * (w_nk[n * 16 + k] as i32))
                .sum();
            assert_eq!(got, want, "WMMA 16x16x16 mismatch at (m={m}, n={n})");
        }
    }
}
