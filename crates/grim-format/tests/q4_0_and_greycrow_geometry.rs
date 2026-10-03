//! Legacy Q4_0 must never be typed as K-quant again, and the GreyCrow g32
//! repack of a Q4_0 payload must be bit-exact.
//!
//! The bug this guards: `map_gguf_dtype_to_storage` typed `Q4_0` as
//! `KQuant(Q4K)`, so every consumer (byte slicing, host dequant, expected
//! size) used 144-byte super-blocks on an 18-byte-per-32-weights payload.
//! That yields finite, plausible, wrong weights -- the failure mode that
//! motivates the fix.

use grim_format::gguf::{map_gguf_dtype_to_storage, GgufDType};
use grim_format::tprov::GgufProvider;
use grim_tensor::dtype::{BlockDtype, KQuantScheme, Storage, UnsupportedFormat};
use grim_tensor::provider::TensorProvider;

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

/// Upstream `Q2_0` (tag 42) and Prism GSQRCO (tag 81) share 64-elem / 18-byte
/// geometry but DIFFER by one codebook level, so they must stay separate
/// schemes.
///
/// The bug this guards: `Q2_0` had no geometry at all
/// (`type_size_per_block() == 0`), which made `read_gguf` reject the whole
/// Qwen3.8-Flash-Next GSQ-RCO-3.5bit checkpoint with "dtype Q2_0 has no
/// implemented block size". The tempting fix — alias tag 42 to the existing
/// `GsqRco3p5` scheme, whose arithmetic happened to match — would be wrong in
/// general: upstream decodes `y = (q - 1) * d` over `{-1, 0, +1, +2}`, GSQRCO
/// decodes `y = (q - 2) * d` over `{-2, -1, 0, +1}`. Every expert weight would
/// be off by exactly one scale, which is finite and plausible, not obviously
/// broken.
#[test]
fn q2_0_and_gsqrco_share_geometry_but_not_a_scheme() {
    // Geometry: identical, and matching upstream `block_q2_0`
    // (ggml-common.h:187 = ggml_half d + qs[QK2_0/4], QK2_0 = 64).
    assert_eq!(GgufDType::Q2_0.tag(), 42);
    assert_eq!(GgufDType::GsqRco3p5.tag(), 81);
    for tag in [GgufDType::Q2_0, GgufDType::GsqRco3p5] {
        assert_eq!(tag.block_size(), 64, "{tag:?} block size");
        assert_eq!(tag.type_size_per_block(), 18, "{tag:?} bytes per block");
    }
    // 64 weights in 18 bytes is 2.25 bpw, which matches the build record's
    // `Q2_0: payload_bpw 2.25` for the real checkpoint.
    assert_eq!(
        map_gguf_dtype_to_storage(GgufDType::Q2_0).expected_bytes(64),
        18
    );

    // Routing: two distinct schemes, never the same one.
    let q2 = map_gguf_dtype_to_storage(GgufDType::Q2_0).storage;
    let gsq = map_gguf_dtype_to_storage(GgufDType::GsqRco3p5).storage;
    assert_eq!(q2, Storage::KQuant(KQuantScheme::Q2_0));
    assert_eq!(gsq, Storage::KQuant(KQuantScheme::GsqRco3p5));
    assert_ne!(q2, gsq);
}

/// The codebook is the whole point of keeping them apart: one non-degenerate
/// block (all-zero input dequantizes correctly under ANY layout, so it cannot
/// discriminate), decoded under both.
#[test]
fn q2_0_and_gsqrco_codebooks_are_one_level_apart() {
    // d = 0.5 is exactly representable in fp16: 0x3800.
    const D_HALF_F16: u16 = 0x3800;
    let mut block = vec![0u8; 18];
    block[0] = D_HALF_F16 as u8;
    block[1] = (D_HALF_F16 >> 8) as u8;
    // Byte 0 holds codes for elements 0..3; set them to 0,1,2,3 in order.
    block[2] = 0b11_10_01_00;

    let q2 = grim_quant::dequant_q2_0(&block, 64).unwrap();
    let gsq = grim_quant::dequant_gsq_rco_3p5(&block, 64).unwrap();

    // Upstream: (q - 1) * d -> {-1, 0, +1, +2} * 0.5
    assert_eq!(&q2[..4], &[-0.5, 0.0, 0.5, 1.0]);
    // GSQRCO: (q - 2) * d -> {-2, -1, 0, +1} * 0.5
    assert_eq!(&gsq[..4], &[-1.0, -0.5, 0.0, 0.5]);

    // Every element differs by exactly one scale.
    for (a, b) in q2.iter().zip(gsq.iter()) {
        assert!(
            (a - b - 0.5).abs() < 1e-6,
            "Q2_0 and GSQRCO must differ by exactly one d: {a} vs {b}"
        );
    }
}

/// End-to-end on the real checkpoint: the tag-42 expert banks parse, report
/// the upstream 64-elem/18-byte geometry, and dequantize to finite values.
///
/// Before the fix, `read_gguf` rejected the whole file with "dtype Q2_0 has no
/// implemented block size", so nothing in it was reachable. Note this reads one
/// bank by byte range rather than materializing it: the tensor is 838 M params
/// (~14.6 GB packed), which does not belong in a unit test.
#[test]
fn real_checkpoint_q2_0_expert_banks_parse_and_dequantize() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../models/QWen38-Flash/Qwen3.8-Flash-Next-GSQ-RCO-3.5bit.gguf");
    if !path.exists() {
        return;
    }
    let prov = GgufProvider::open(path.to_str().unwrap()).expect("real checkpoint opens");
    for name in ["blk.0.ffn_down_exps.weight", "blk.0.ffn_up_exps.weight"] {
        let meta = prov.meta(name).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            meta.dtype.storage,
            Storage::KQuant(KQuantScheme::Q2_0),
            "{name} must route to the upstream Q2_0 scheme"
        );
        let elems: usize = meta.shape.iter().product();
        assert_eq!(elems % 64, 0, "{name} must be block-aligned");
        assert_eq!(
            meta.dtype.expected_bytes(elems),
            (elems / 64) * 18,
            "{name} byte size must follow 18 B / 64 elems"
        );
        // 2.25 bpw, matching the build record's Q2_0 payload_bpw.
        let bpw = (meta.dtype.expected_bytes(elems) * 8) as f64 / elems as f64;
        assert!(
            (bpw - 2.25).abs() < 1e-9,
            "{name} bpw {bpw} != 2.25"
        );
    }
}
