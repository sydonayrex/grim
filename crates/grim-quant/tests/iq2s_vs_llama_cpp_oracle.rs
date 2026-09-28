//! `dequant_iq2s` vs llama.cpp's OWN `dequantize_row_iq2_s`, bit-exact.
//!
//! The oracle is not a transcription. `tools/gen_iq2s_oracle.c` assembles a C
//! program from llama.cpp's actual sources — the function body, the
//! `iq2s_grid` and `kmask_iq2xs` tables, and `ggml_compute_fp16_to_fp32` — all
//! extracted from `old/repo/llama.cpp-master/ggml/src/`, and it is compiled and
//! run as a subprocess. Nothing in the expected value was written by hand, so
//! this cannot pass by agreeing with a misreading of the spec.
//!
//! The previous decoder differed from that function in three ways (scale
//! grouping and factor, arithmetic grid vs the `iq2s_grid` table, 1-bit vs
//! 4-bit packed signs) and read 8 bytes past the block for the sign index. A
//! second Rust transcription would share any misreading; this cannot.
//!
//! Bit-exact, not tolerance-based: both evaluate the same f16 scale, the same
//! 4-bit sub-scale, the same table lookup and the same sign mask, so any
//! disagreement is a defect rather than accumulated rounding.
//!
//! Skips LOUDLY if the oracle binary is absent — never silently.

use std::io::Write;
use std::process::{Command, Stdio};

const ORACLE: &str = "/tmp/iq2s_oracle/oracle";

/// Deterministic packed bytes. The f16 scale in each block's first two bytes is
/// kept to a sane exponent so the products stay finite and comparable; every
/// other byte is free to exercise the grid index, the high bits of `qh` and the
/// packed sign mask.
fn packed(n_blocks: usize) -> Vec<u8> {
    let mut s: u32 = 0x9E37_79B9 ^ (n_blocks as u32);
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    let mut b = vec![0u8; n_blocks * 82];
    for blk in 0..n_blocks {
        // exponent 0b01110 (14) -> ~2^-1; mantissa varied
        let f16 = 0x3800u16 | ((next() & 0x03ff) as u16);
        b[blk * 82] = (f16 & 0xff) as u8;
        b[blk * 82 + 1] = (f16 >> 8) as u8;
        for k in 2..82 {
            b[blk * 82 + k] = (next() & 0xff) as u8;
        }
    }
    b
}

#[test]
fn dequant_iq2s_is_bit_exact_vs_llama_cpp() {
    if !std::path::Path::new(ORACLE).exists() {
        panic!(
            "oracle missing at {ORACLE}. It is llama.cpp's own dequantize_row_iq2_s, \
             compiled from old/repo/. Build it with tools/gen_iq2s_oracle.py; this gate \
             must not pass without it."
        );
    }

    for n_blocks in [1usize, 3, 8, 17] {
        let bytes = packed(n_blocks);
        let n = n_blocks * 256;

        let mut child = Command::new(ORACLE)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("spawn oracle");
        child
            .stdin
            .as_mut()
            .expect("oracle stdin")
            .write_all(&bytes)
            .expect("write to oracle");
        let out = child.wait_with_output().expect("run oracle");
        assert!(out.status.success(), "oracle exited {:?}", out.status);
        let want_f32: Vec<f32> = out
            .stdout
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(want_f32.len(), n, "oracle returned the wrong length");

        let got = grim_quant::dequant_iq2s(&bytes, n).expect("dequant_iq2s");
        assert_eq!(got.len(), n, "dequant_iq2s returned the wrong length");

        let mut worst = 0.0f32;
        let mut at = 0usize;
        for i in 0..n {
            if got[i].to_bits() != want_f32[i].to_bits() {
                let d = (got[i] - want_f32[i]).abs();
                if d > worst {
                    worst = d;
                    at = i;
                }
            }
        }
        eprintln!(
            "[iq2s-oracle] {n_blocks} blocks ({n} weights): bit-exact={} worst |diff| {worst:.3e} \
             at {at} (block {}, offset {}) — want {:+.9e}, got {:+.9e}",
            worst == 0.0,
            at / 256,
            at % 256,
            want_f32[at],
            got[at]
        );
        assert_eq!(
            worst, 0.0,
            "dequant_iq2s disagrees with llama.cpp at block {} offset {}: \
             want {:+.9e} (0x{:08x}), got {:+.9e} (0x{:08x})",
            at / 256,
            at % 256,
            want_f32[at],
            want_f32[at].to_bits(),
            got[at],
            got[at].to_bits(),
        );
    }
    eprintln!("[iq2s-oracle] PASS bit-exact vs llama.cpp");
}

/// `quant_iq2s` vs llama.cpp's OWN `quantize_row_iq2_s_ref`, bit-exact on the
/// packed bytes. The oracle is assembled from llama.cpp's sources by
/// `tools/gen_iq2s_oracle.py` (including `iq2xs_init_impl`, which builds the
/// code->grid map and neighbour lists the quantizer searches); no expected
/// value was written by hand. The previous in-repo quantizer stored per-element
/// 2-bit codes with all-zero sub-scales, which this gate exists to catch.
#[test]
fn quant_iq2s_is_bit_exact_vs_llama_cpp() {
    const QUANT_ORACLE: &str = "/tmp/iq2s_oracle/quant_oracle";
    if !std::path::Path::new(QUANT_ORACLE).exists() {
        panic!(
            "quant oracle missing at {QUANT_ORACLE}. It is llama.cpp's own \
             quantize_row_iq2_s_ref, compiled from old/repo/. Build both oracles with \
             crates/grim-quant/tools/gen_iq2s_oracle.py; this gate must not pass without it."
        );
    }

    for n_blocks in [1usize, 3, 8, 17] {
        let n = n_blocks * 256;
        let mut s: u32 = 0x1234_5679 ^ (n as u32);
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        // Magnitudes in [0, 2^-k) exercise the scale fit; a later group of
        // near-zero values exercises the GROUP_MAX_EPS_IQ2_S skip.
        let xs: Vec<f32> = (0..n)
            .map(|i| {
                let v = (next() & 0xffff) as f32 / 65535.0 - 0.5;
                if (i / 256) % 2 == 1 {
                    v * 1e-10
                } else {
                    v
                }
            })
            .collect();

        let mut child = Command::new(QUANT_ORACLE)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("spawn quant oracle");
        child
            .stdin
            .as_mut()
            .expect("oracle stdin")
            .write_all(&bytemuck_bytes(&xs))
            .expect("write to oracle");
        let out = child.wait_with_output().expect("run oracle");
        assert!(out.status.success(), "oracle exited {:?}", out.status);
        let want = out.stdout;
        assert_eq!(want.len(), n_blocks * 82, "oracle byte count");

        let got = grim_quant::quant_iq2s(&xs).expect("quant_iq2s");
        assert_eq!(got.len(), want.len(), "quant_iq2s byte count");

        let mut first_diff: Option<(usize, u8, u8)> = None;
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            if g != w {
                first_diff = Some((i, *g, *w));
                break;
            }
        }
        if let Some((i, g, w)) = first_diff {
            panic!(
                "quant_iq2s disagrees with llama.cpp at byte {i} (block {}, offset {}): \
                 ours {g:#04x}, reference {w:#04x}",
                i / 82,
                i % 82
            );
        }
    }
    eprintln!("[iq2s-oracle] PASS quant_iq2s bit-exact vs llama.cpp");
}

/// f32 slice as little-endian bytes without pulling in a bytemuck dependency.
fn bytemuck_bytes(xs: &[f32]) -> Vec<u8> {
    xs.iter().flat_map(|x| x.to_le_bytes()).collect()
}
