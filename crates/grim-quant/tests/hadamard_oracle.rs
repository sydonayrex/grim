// crates/grim-quant/tests/hadamard_oracle.rs
use grim_quant::hadamard::{build_hadamard_matrix, fwht_f32, fwht_signed_f32, hadamard_entry};

fn approx(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol * a.abs().max(1.0)
}

#[test]
fn h_entry_is_parity_of_row_and_col_over_sqrt_n() {
    // Reference: llama-model.cpp:2338-2341 — sign = parity of (r & c) XOR-folded.
    let n = 8.0f32;
    let s = 1.0 / n.sqrt();
    assert_eq!(hadamard_entry(0, 0, 8), s);
    assert_eq!(hadamard_entry(1, 1, 8), -s); // 1 & 1 = 1 -> odd parity
    assert_eq!(hadamard_entry(1, 2, 8), s);  // 1 & 2 = 0 -> even
    assert_eq!(hadamard_entry(3, 1, 8), -s); // 3 & 1 = 1 -> odd
    let _ = n;
}

#[test]
fn h_h_equals_identity_for_1024() {
    let n = 1024usize;
    let h = build_hadamard_matrix(n as u32);
    // (H·H)[i][j] = delta_ij since H is orthonormal (H^2 = I).
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f32;
            for k in 0..n {
                acc += h[i * n + k] * h[k * n + j];
            }
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(approx(acc, want, 1e-4), "H·H[{i}][{j}]={acc}");
        }
    }
}

#[test]
fn fwht_matches_matrix_product_on_random_rows() {
    // The butterfly must equal the explicit matrix product H·x.
    let n = 1024usize;
    let h = build_hadamard_matrix(n as u32);
    let mut state = 0x243F6A8885A308D3u64;
    let mut rand = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f32 / (1u64 << 53) as f32 * 4.0 - 2.0
    };
    for _row in 0..4 {
        let x: Vec<f32> = (0..n).map(|_| rand()).collect();
        let mut via_butterfly = x.clone();
        fwht_f32(&mut via_butterfly);
        for i in 0..n {
            let mut acc = 0.0f32;
            for k in 0..n {
                acc += h[i * n + k] * x[k];
            }
            assert!(approx(via_butterfly[i], acc, 1e-4));
        }
    }
}

#[test]
fn fwht_signed_applies_signs_then_h() {
    // D then H: fwht_signed(x, d)[i] == sum_k H[i][k] * d[k] * x[k]
    let n = 64usize;
    let h = build_hadamard_matrix(n as u32);
    let signs: Vec<i32> = (0..n).map(|i| if i % 3 == 0 { -1 } else { 1 }).collect();
    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25 - 4.0).collect();
    let mut got = x.clone();
    fwht_signed_f32(&mut got, &signs);
    for i in 0..n {
        let mut acc = 0.0f32;
        for k in 0..n {
            acc += h[i * n + k] * (signs[k] as f32) * x[k];
        }
        assert!(approx(got[i], acc, 1e-4));
    }
}
