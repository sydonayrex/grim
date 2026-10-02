//! Legacy Q4_0 must never be typed as K-quant again, and the GreyCrow g32
//! repack of a Q4_0 payload must be bit-exact.
//!
//! The bug this guards: `map_gguf_dtype_to_storage` typed `Q4_0` as
//! `KQuant(Q4K)`, so every consumer (byte slicing, host dequant, expected
//! size) used 144-byte super-blocks on an 18-byte-per-32-weights payload.
//! That yields finite, plausible, wrong weights -- the failure mode that
//! motivates the fix.

use grim_format::gguf::{map_gguf_dtype_to_storage, GgufDType};
use grim_tensor::dtype::{BlockDtype, Storage, UnsupportedFormat};

#[test]
fn q4_0_is_block_32_with_18_bytes_not_a_kquant_superblock() {
    let d = map_gguf_dtype_to_storage(GgufDType::Q4_0);
    assert_eq!(
        d.storage,
        Storage::Block(BlockDtype::Q4_0),
        "Q4_0 must be its own block storage"
    );
    assert_ne!(
        d.storage,
        Storage::KQuant(grim_tensor::dtype::KQuantScheme::Q4K),
        "Q4_0 is 18 B / 32 weights, never Q4_K's 144 B / 256"
    );
    // 18 bytes per 32 weights, and NOT 144 per 256 (which is 4.5x denser per
    // weight -- the exact mismatch that corrupted Q4_0 slices).
    assert_eq!(d.expected_bytes(32), 18);
    assert_eq!(d.expected_bytes(256), 144);
    assert_eq!(d.expected_bytes(160), 18 * 5, "ngram_dim=160 rows");
}

#[test]
fn q4_siblings_refuse_with_correct_geometry_instead_of_posing_as_kquant() {
    for (tag, bytes) in [
        (GgufDType::Q4_1, 20usize),
        (GgufDType::Q4_2, 20),
        (GgufDType::Q5_0, 22),
        (GgufDType::Q5_1, 24),
    ] {
        let d = map_gguf_dtype_to_storage(tag);
        match d.storage {
            Storage::Unsupported(UnsupportedFormat {
                block_size,
                bytes_per_block,
                ..
            }) => {
                assert_eq!(block_size, Some(32), "{tag:?} block size");
                assert_eq!(bytes_per_block, Some(bytes), "{tag:?} bytes per block");
            }
            other => panic!("{tag:?} must refuse, got {other:?}"),
        }
        assert_eq!(d.expected_bytes(32), bytes);
    }
}

#[test]
fn q5_k_still_maps_to_kquant() {
    // Guard the neighbouring arm: only the legacy block quants moved.
    assert_eq!(
        map_gguf_dtype_to_storage(GgufDType::Q5K).storage,
        Storage::KQuant(grim_tensor::dtype::KQuantScheme::Q5K)
    );
    assert_eq!(
        map_gguf_dtype_to_storage(GgufDType::Q4K).storage,
        Storage::KQuant(grim_tensor::dtype::KQuantScheme::Q4K)
    );
}

/// Round-to-nearest-even f32 -> f16 bits, so the fixture does not need a
/// Q4_0 *packer* (grim has no quant_q4_0; only the decoder exists).
fn f32_to_f16_bits(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let mut exp = ((x >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = x & 0x007F_FFFF;
    if exp <= 0 {
        return sign;
    }
    if exp >= 0x1F {
        return sign | 0x7C00;
    }
    let round = 0x0FFF + ((mant >> 12) & 1);
    let mut m = mant + round;
    if m > 0x007F_FFFF {
        m = 0;
        exp += 1;
        if exp >= 0x1F {
            return sign | 0x7C00;
        }
    }
    sign | ((exp as u16) << 10) | ((m >> 13) as u16)
}

#[test]
fn greycrow_g32_repack_of_a_real_q4_0_payload_is_bit_exact() {
    // Column-major Q4_0 payload, one 18-byte block per 32 weights per column.
    let (n, k) = (8usize, 64usize);
    let blocks = k / 32;
    let mut q40 = Vec::with_capacity(n * blocks * 18);
    for col in 0..n {
        for g in 0..blocks {
            let scale = 0.01 + ((col + g) % 7) as f32 * 0.017;
            let mut blk = [0u8; 18];
            blk[0..2].copy_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
            for i in 0..32 {
                let nib = ((col * 5 + g * 3 + i * 7) % 16) as u8;
                if i % 2 == 0 {
                    blk[2 + i / 2] = nib;
                } else {
                    blk[2 + i / 2] |= nib << 4;
                }
            }
            q40.extend_from_slice(&blk);
        }
    }
    assert_eq!(q40.len(), n * blocks * 18);

    let (qw, sc, zr) = grim_quant::repack_q40_to_greycrow_g32(&q40, n, k).unwrap();
    assert_eq!(qw.len(), n * (k / 8) * 4);
    assert_eq!(sc.len(), n * blocks * 2);
    assert_eq!(zr.len(), n * blocks);
    assert!(zr.iter().all(|&z| z == 8));

    let via_greycrow = grim_quant::dequant_greycrow_g32(&qw, &sc, &zr, n, k).unwrap();
    let via_q40 = grim_quant::dequant_q4_0(&q40, n * k).unwrap();
    for (i, (a, b)) in via_greycrow.iter().zip(via_q40.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "GreyCrow must equal Q4_0 exactly, diverged at {i}: {a} vs {b}"
        );
    }
}
