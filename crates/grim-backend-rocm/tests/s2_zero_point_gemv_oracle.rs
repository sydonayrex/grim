use grim_backend_rocm::kernels::dot_gemv::{
    KERNEL_SOURCE, s2_zero_point_gemv_enabled,
};

#[test]
fn s2_zero_point_gemv_kernel_present_and_defaults_off() {
    assert!(KERNEL_SOURCE.contains("strata_s2_zero_point_gemv_kernel"));
    assert!(KERNEL_SOURCE.contains("0x01010101u"));

    let prev = std::env::var("GRIM_STRATA_S2_ZP_GEMV").ok();
    unsafe { std::env::remove_var("GRIM_STRATA_S2_ZP_GEMV") };
    assert!(!s2_zero_point_gemv_enabled());
    if let Some(v) = prev {
        unsafe { std::env::set_var("GRIM_STRATA_S2_ZP_GEMV", v) };
    }
}

#[test]
fn s2_zero_point_bitmask_oracle_exact() {
    let w_bytes: [u8; 4] = [12, 5, 250, 0];
    let w_packed = u32::from_le_bytes(w_bytes);
    let zp: u8 = 8;
    let z4 = (zp as u32).wrapping_mul(0x0101_0101u32);
    let scale = 0.25f32;
    let x = [1.0f32, -2.0, 0.5, 4.0];

    let mut acc_fast = 0.0f32;
    for b in 0..4 {
        let q = ((w_packed >> (b * 8)) & 0xFF) as i32 - ((z4 >> (b * 8)) & 0xFF) as i32;
        acc_fast += (q as f32) * scale * x[b];
    }

    let mut acc_ref = 0.0f32;
    for b in 0..4 {
        let q = (w_bytes[b] as i32) - (zp as i32);
        acc_ref += (q as f32) * scale * x[b];
    }

    assert!((acc_fast - acc_ref).abs() < 1e-6);
}
