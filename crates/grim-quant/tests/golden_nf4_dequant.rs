use grim_quant::dequant_nf4;

fn close(got: f32, want: f32, ctx: &str) {
    let abs = (got - want).abs();
    let denom = want.abs().max(1e-7);
    assert!(got.is_finite(), "{ctx}: non-finite {got:?} (want {want:?})");
    assert!(
        abs == 0.0 || (abs / denom) < 1e-5,
        "{ctx}: got {got:?} want {want:?} (abs={abs})",
    );
}

use grim_quant::NF4_LUT;

#[test]
fn nf4_golden_hand_constructed_buffer() {
    let scale = 2.0f32;
    let mut buf = Vec::new();
    buf.extend_from_slice(&scale.to_le_bytes());
    // byte 0 = 0xF0: hi=0xF (15→1.0), lo=0x0 (0→-1.0)
    // byte 1 = 0xA5: hi=0xA (10→0.23828125), lo=0x5 (5→-0.23828125)
    buf.push(0xF0);
    buf.push(0xA5);

    let out = dequant_nf4(&buf, 4).expect("nf4 dequant");
    assert_eq!(out.len(), 4);

    close(out[0], NF4_LUT[0xF] * scale, "nf4[0] hi=F");
    close(out[1], NF4_LUT[0x0] * scale, "nf4[1] lo=0");
    close(out[2], NF4_LUT[0xA] * scale, "nf4[2] hi=A");
    close(out[3], NF4_LUT[0x5] * scale, "nf4[3] lo=5");
}

#[test]
fn nf4_golden_spans_multiple_bytes() {
    let scale = 1.0f32;
    let mut buf = Vec::new();
    buf.extend_from_slice(&scale.to_le_bytes());
    // byte 0 = 0xE1: hi=0xE (14→0.69921875), lo=0x1 (1→-0.69921875)
    // byte 1 = 0x87: hi=0x8 (8→0.10009765625), lo=0x7 (7→-0.10009765625)
    buf.push(0xE1);
    buf.push(0x87);

    let out = dequant_nf4(&buf, 4).expect("nf4 multi-byte");
    assert_eq!(out.len(), 4);

    close(out[0], NF4_LUT[0xE], "nf4[0] hi=E");
    close(out[1], NF4_LUT[0x1], "nf4[1] lo=1");
    close(out[2], NF4_LUT[0x8], "nf4[2] hi=8");
    close(out[3], NF4_LUT[0x7], "nf4[3] lo=7");
}

#[test]
#[should_panic(expected = "out of range")]
fn nf4_golden_rejects_empty_buffer() {
    let _ = dequant_nf4(&[], 1);
}

#[test]
fn nf4_golden_scale_only_no_codes_returns_empty() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&3.0f32.to_le_bytes());
    let out = dequant_nf4(&buf, 0).expect("nf4 zero values");
    assert!(out.is_empty());
}
