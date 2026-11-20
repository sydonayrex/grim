use grim_quant::tqk_trellis::{
    TQK_2BIT_CODEBOOK, dequant_tqk_2bit, dequant_tqk_4bit, pack_tqk_2bit, pack_tqk_4bit,
    tqk_trellis_enabled,
};

#[test]
fn tqk_trellis_2bit_roundtrip_oracle() {
    let idx: [u8; 16] = [0, 1, 2, 3, 3, 2, 1, 0, 0, 2, 1, 3, 1, 0, 3, 2];
    let packed = pack_tqk_2bit(&idx);
    let scale = 0.5f32;
    let out = dequant_tqk_2bit(packed, scale);
    for i in 0..16 {
        let want = TQK_2BIT_CODEBOOK[idx[i] as usize] * scale;
        assert!((out[i] - want).abs() < 1e-6);
    }
}

#[test]
fn tqk_trellis_4bit_roundtrip_oracle() {
    let idx: [u8; 8] = [0, 4, 8, 12, 15, 7, 1, 9];
    let packed = pack_tqk_4bit(&idx);
    let scale = 0.125f32;
    let out = dequant_tqk_4bit(packed, scale);
    for i in 0..8 {
        let want = ((idx[i] as i32) - 8) as f32 * scale;
        assert!((out[i] - want).abs() < 1e-6);
    }
}

#[test]
fn tqk_trellis_defaults_off() {
    let prev = std::env::var("GRIM_STRATA_TQK_TRELLIS").ok();
    unsafe { std::env::remove_var("GRIM_STRATA_TQK_TRELLIS") };
    assert!(!tqk_trellis_enabled());
    if let Some(v) = prev {
        unsafe { std::env::set_var("GRIM_STRATA_TQK_TRELLIS", v) };
    }
}
