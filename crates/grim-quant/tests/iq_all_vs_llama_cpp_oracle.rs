//! Bit-exact verification of all standardized IQ formats against compiled llama.cpp C oracles.
//!
//! Evaluates IQ4_XS, IQ3_XXS, IQ3_S, IQ2_XXS, and IQ2_XS against standalone binaries
//! compiled directly from `old/repo/llama.cpp-master/ggml/src/`.

use std::io::Write;
use std::process::{Command, Stdio};

fn generate_packed_bytes(n_blocks: usize, block_size: usize) -> Vec<u8> {
    let mut s: u32 = 0x9E37_79B9 ^ ((n_blocks * block_size) as u32);
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    let mut b = vec![0u8; n_blocks * block_size];
    for blk in 0..n_blocks {
        // sane exponent for f16: ~2^-1
        let f16 = 0x3800u16 | ((next() & 0x03ff) as u16);
        b[blk * block_size] = (f16 & 0xff) as u8;
        b[blk * block_size + 1] = (f16 >> 8) as u8;
        for k in 2..block_size {
            b[blk * block_size + k] = (next() & 0xff) as u8;
        }
    }
    b
}

fn run_oracle(oracle_path: &str, bytes: &[u8], n_weights: usize) -> Vec<f32> {
    if !std::path::Path::new(oracle_path).exists() {
        panic!("oracle binary not found at {oracle_path}. Build it via tools/gen_all_iq_oracles.py");
    }

    let mut child = Command::new(oracle_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn oracle");

    child
        .stdin
        .as_mut()
        .expect("oracle stdin")
        .write_all(bytes)
        .expect("write to oracle");

    let out = child.wait_with_output().expect("run oracle");
    assert!(out.status.success(), "oracle exited {:?}", out.status);
    let want: Vec<f32> = out
        .stdout
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(want.len(), n_weights);
    want
}

fn assert_bit_exact(got: &[f32], want: &[f32], label: &str) {
    assert_eq!(got.len(), want.len());
    let mut mismatches = 0;
    let mut worst = 0.0f32;
    for i in 0..got.len() {
        if got[i].to_bits() != want[i].to_bits() {
            mismatches += 1;
            let diff = (got[i] - want[i]).abs();
            if diff > worst {
                worst = diff;
            }
        }
    }
    assert_eq!(
        mismatches, 0,
        "[{label}] {mismatches}/{} mismatches (worst |diff|: {worst:.3e})",
        got.len()
    );
}

#[test]
fn test_iq4_xs_bit_exact_vs_llama_cpp() {
    const BLOCK_SIZE: usize = 136;
    for n_blocks in [1usize, 3, 8] {
        let n_weights = n_blocks * 256;
        let bytes = generate_packed_bytes(n_blocks, BLOCK_SIZE);
        let want = run_oracle("/tmp/iq_oracles/iq4_xs_oracle", &bytes, n_weights);
        let got = grim_quant::dequant_iq4xs(&bytes, n_weights).expect("dequant_iq4xs");
        assert_bit_exact(&got, &want, "IQ4_XS");
    }
}

#[test]
fn test_iq3_xxs_bit_exact_vs_llama_cpp() {
    const BLOCK_SIZE: usize = 98;
    for n_blocks in [1usize, 3, 8] {
        let n_weights = n_blocks * 256;
        let bytes = generate_packed_bytes(n_blocks, BLOCK_SIZE);
        let want = run_oracle("/tmp/iq_oracles/iq3_xxs_oracle", &bytes, n_weights);
        let got = grim_quant::dequant_iq3xxs(&bytes, n_weights).expect("dequant_iq3xxs");
        assert_bit_exact(&got, &want, "IQ3_XXS");
    }
}

#[test]
fn test_iq3_s_bit_exact_vs_llama_cpp() {
    const BLOCK_SIZE: usize = 110;
    for n_blocks in [1usize, 3, 8] {
        let n_weights = n_blocks * 256;
        let bytes = generate_packed_bytes(n_blocks, BLOCK_SIZE);
        let want = run_oracle("/tmp/iq_oracles/iq3_s_oracle", &bytes, n_weights);
        let got = grim_quant::dequant_iq3s(&bytes, n_weights).expect("dequant_iq3s");
        assert_bit_exact(&got, &want, "IQ3_S");
    }
}

#[test]
fn test_iq2_xxs_bit_exact_vs_llama_cpp() {
    const BLOCK_SIZE: usize = 66;
    for n_blocks in [1usize, 3, 8] {
        let n_weights = n_blocks * 256;
        let bytes = generate_packed_bytes(n_blocks, BLOCK_SIZE);
        let want = run_oracle("/tmp/iq_oracles/iq2_xxs_oracle", &bytes, n_weights);
        let got = grim_quant::dequant_iq2xxs(&bytes, n_weights).expect("dequant_iq2xxs");
        assert_bit_exact(&got, &want, "IQ2_XXS");
    }
}

#[test]
fn test_iq2_xs_bit_exact_vs_llama_cpp() {
    const BLOCK_SIZE: usize = 74;
    for n_blocks in [1usize, 3, 8] {
        let n_weights = n_blocks * 256;
        let bytes = generate_packed_bytes(n_blocks, BLOCK_SIZE);
        let want = run_oracle("/tmp/iq_oracles/iq2_xs_oracle", &bytes, n_weights);
        let got = grim_quant::dequant_iq2xs(&bytes, n_weights).expect("dequant_iq2xs");
        assert_bit_exact(&got, &want, "IQ2_XS");
    }
}
