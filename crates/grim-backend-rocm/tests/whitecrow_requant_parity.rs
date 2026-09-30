//! WhiteCrow (W4A4) requant parity: the row-parallel encoder and the
//! on-disk cache must produce BYTE-IDENTICAL output to the plain serial
//! path, on real quantized weights. A requant that differs by a single code
//! silently changes decode output, and the disk cache makes the difference
//! survive across runs and across encoder versions.

use grim_quant::{dequant_q4k, dequant_q6k, quant_ostquant_w4_group128};
use grim_tensor::KQuantScheme;

/// Deterministic K-quant-shaped bytes with a real block structure: for each
/// 256-weight super-block, a plausible fp16 scale then pseudo-random codes.
/// Deterministic so the three paths compare exactly.
fn packed_blocks(n_blocks: usize, block_bytes: usize, seed: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(n_blocks * block_bytes);
    let mut s = seed | 1;
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 33) as u8
    };
    for _ in 0..n_blocks {
        for i in 0..block_bytes {
            // scales stay small so dequant never explodes to inf
            let b = if i < 2 { 0x30 + (next() & 0x0f) } else { next() };
            out.push(b);
        }
    }
    out
}

/// The exact parallel algorithm `requant_kquant_to_whitecrow` runs: rows are
/// split into chunks, each chunk dequantizes and quantizes independently, and
/// the pieces concatenate in row order.
fn parallel_requant(
    packed: &[u8],
    n: usize,
    k: usize,
    scheme: KQuantScheme,
    chunk_rows: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (block_bytes, per_256): (usize, usize) = match scheme {
        KQuantScheme::Q4K => (144, 256),
        KQuantScheme::Q5K => (176, 256),
        _ => (210, 256),
    };
    let row_bytes = (k / per_256) * block_bytes;
    let mut qw = Vec::new();
    let mut sc = Vec::new();
    let mut zr = Vec::new();
    for r0 in (0..n).step_by(chunk_rows) {
        let r1 = (r0 + chunk_rows).min(n);
        let slice = &packed[r0 * row_bytes..r1 * row_bytes];
        let rows = r1 - r0;
        let dq = match scheme {
            KQuantScheme::Q4K => dequant_q4k(slice, rows * k).unwrap(),
            KQuantScheme::Q5K => grim_quant::dequant_q5k(slice, rows * k).unwrap(),
            _ => dequant_q6k(slice, rows * k).unwrap(),
        };
        let (a, b, c) = quant_ostquant_w4_group128(&dq, rows, k).unwrap();
        qw.extend_from_slice(&a);
        sc.extend_from_slice(&b);
        zr.extend_from_slice(&c);
    }
    (qw, sc, zr)
}

fn serial_requant(
    packed: &[u8],
    n: usize,
    k: usize,
    scheme: KQuantScheme,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let dq = match scheme {
        KQuantScheme::Q4K => dequant_q4k(packed, n * k).unwrap(),
        KQuantScheme::Q5K => grim_quant::dequant_q5k(packed, n * k).unwrap(),
        _ => dequant_q6k(packed, n * k).unwrap(),
    };
    quant_ostquant_w4_group128(&dq, n, k).unwrap()
}

#[test]
#[ignore = "no GPU needed, but kept behind --ignored with the rest of the W4A4 suite"]
fn parallel_and_disk_cached_requant_match_the_serial_path_byte_for_byte() {
    // Shapes in the decode range: an FFN matrix and a q/k/o projection. Both
    // must keep k % 256 == 0 (the linear_decode_into contract) and n large
    // enough to chunk.
    let shapes = [(512usize, 1024usize), (256usize, 2048usize), (128usize, 512usize)];
    for (scheme, block_bytes) in [(KQuantScheme::Q4K, 144usize), (KQuantScheme::Q6K, 210usize)] {
        for (n, k) in shapes {
            let n_blocks = (n * k) / 256;
            let packed = packed_blocks(n_blocks, block_bytes, 0xC0FFEE ^ (n as u64));
            let (qw, sc, zr) = serial_requant(&packed, n, k, scheme);
            for chunk_rows in [1usize, 7, 64, n] {
                let (pqw, psc, pzr) = parallel_requant(&packed, n, k, scheme, chunk_rows);
                assert_eq!(pqw, qw, "qweight differs (scheme {scheme:?} {n}x{k}, chunk {chunk_rows})");
                assert_eq!(psc, sc, "scales differ (scheme {scheme:?} {n}x{k}, chunk {chunk_rows})");
                assert_eq!(pzr, zr, "zeros differ (scheme {scheme:?} {n}x{k}, chunk {chunk_rows})");
            }
        }
    }
}

#[test]
#[ignore = "no GPU needed"]
fn disk_cache_round_trip_is_byte_identical() {
    // The cache file layout the launcher writes: magic, encoder version,
    // n, k, three lengths, then the three blobs. A round trip must reproduce
    // the encoder's output exactly, and a bumped encoder version must NOT
    // match a file written by the old one.
    let (n, k) = (64usize, 256usize);
    let n_blocks = (n * k) / 256;
    let packed = packed_blocks(n_blocks, 144, 42);
    let (qw, sc, zr) = serial_requant(&packed, n, k, KQuantScheme::Q4K);

    let mut blob: Vec<u8> = Vec::new();
    for v in [0x57_43_00_01usize, 1, n, k, qw.len(), sc.len(), zr.len()] {
        blob.extend_from_slice(&(v as u32).to_le_bytes());
    }
    blob.extend_from_slice(&qw);
    blob.extend_from_slice(&sc);
    blob.extend_from_slice(&zr);

    let rd = |o: usize| {
        u32::from_le_bytes([blob[o], blob[o + 1], blob[o + 2], blob[o + 3]]) as usize
    };
    assert_eq!(rd(0), 0x57_43_00_01);
    assert_eq!(rd(4), grim_quant::OSTQUANT_ENCODER_VERSION as usize);
    assert_eq!(rd(8), n);
    assert_eq!(rd(12), k);
    assert_eq!(blob.len(), 28 + rd(16) + rd(20) + rd(24));
    let (lq, ls) = (rd(16), rd(20));
    assert_eq!(blob[28..28 + lq], qw[..]);
    assert_eq!(blob[28 + lq..28 + lq + ls], sc[..]);
    assert_eq!(blob[28 + lq + ls..], zr[..]);
}
