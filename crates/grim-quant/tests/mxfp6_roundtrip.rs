use grim_quant::{f32_to_mxfp4_e2m1, mxfp4_e2m1_to_f32};

#[test]
fn mxfp_shared_exponent_roundtrip_is_stable() {
    let input_vals = vec![0.0f32, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0];
    let shared_exp = 127u8; // scale = 2^0 = 1.0

    for &v in &input_vals {
        let code = f32_to_mxfp4_e2m1(v, shared_exp);
        let recon = mxfp4_e2m1_to_f32(code, shared_exp);
        let diff = (v - recon).abs();
        assert!(diff <= 0.5, "mxfp roundtrip error too high for {v}: got {recon}, diff {diff}");
    }
}
