// crates/grim-quant/src/hadamard.rs
//! Prism rotated-domain host math (Task reference: plans/PLAN-prism-hadamard-bonsai.md Task 1).
//! H = normalized Sylvester Walsh-Hadamard; entry sign is the parity of `row & col`
//! (bitwise-AND, then XOR-fold all bits — llama-model.cpp:2338-2341).
//! `fwht_f32` is the llama.cpp CPU butterfly (ops.cpp:12301) ported verbatim:
//! scale by 1/sqrt(n) first, then len-doubling passes, low = u+v, high = u−v.

/// One H entry without materializing the matrix: `(parity(row & col) ? -1 : 1) / sqrt(n)`.
pub fn hadamard_entry(row: u32, col: u32, n: u32) -> f32 {
    let scale = 1.0 / (n as f32).sqrt();
    let mut parity = row & col;
    parity ^= parity >> 16;
    parity ^= parity >> 8;
    parity ^= parity >> 4;
    parity ^= parity >> 2;
    parity ^= parity >> 1;
    if parity & 1 == 1 {
        -scale
    } else {
        scale
    }
}

/// Row-major n×n H. n must be a power of two.
pub fn build_hadamard_matrix(n: u32) -> Vec<f32> {
    assert!(n.is_power_of_two(), "hadamard n must be a power of two");
    (0..n * n)
        .map(|idx| hadamard_entry(idx / n, idx % n, n))
        .collect()
}

/// In-place normalized fast Walsh-Hadamard transform. `data.len()` must be a power of two.
pub fn fwht_f32(data: &mut [f32]) {
    let n = data.len();
    assert!(n.is_power_of_two(), "fwht len must be a power of two");
    let scale = 1.0 / (n as f32).sqrt();
    for v in data.iter_mut() {
        *v *= scale;
    }
    let mut len = 1usize;
    while len < n {
        for i in (0..n).step_by(2 * len) {
            for j in 0..len {
                let u = data[i + j];
                let v = data[i + len + j];
                data[i + j] = u + v;
                data[i + len + j] = u - v;
            }
        }
        len <<= 1;
    }
}

/// The runtime fold order for a Hadamard-aware linear: signs D applied to the
/// input, then H. Equivalent to `W_f · (H·(D·x))` against a folded weight.
pub fn fwht_signed_f32(data: &mut [f32], signs: &[i32]) {
    assert_eq!(data.len(), signs.len(), "sign vector length mismatch");
    for (v, s) in data.iter_mut().zip(signs.iter()) {
        *v *= *s as f32;
    }
    fwht_f32(data);
}
