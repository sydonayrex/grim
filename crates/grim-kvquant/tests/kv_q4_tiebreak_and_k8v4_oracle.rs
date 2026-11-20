use grim_kvquant::{
    quantize_k8_symmetric, kv_k8v4_enabled, kv_q4_tiebreak_round,
};

#[test]
fn kv_q4_tiebreak_half_away_from_zero_oracle() {
    assert_eq!(kv_q4_tiebreak_round(0.5), 1);
    assert_eq!(kv_q4_tiebreak_round(1.5), 2);
    assert_eq!(kv_q4_tiebreak_round(2.5), 3);
    assert_eq!(kv_q4_tiebreak_round(-0.5), -1);
    assert_eq!(kv_q4_tiebreak_round(-1.5), -2);
    assert_eq!(kv_q4_tiebreak_round(-2.5), -3);
}

#[test]
fn kv_k8_symmetric_roundtrip_oracle() {
    let k = [0.0f32, 0.5, -0.5, 1.0, -1.0, 0.25];
    let (q, scale) = quantize_k8_symmetric(&k);
    for (&orig, &qi) in k.iter().zip(q.iter()) {
        let deq = (qi as f32) * scale;
        assert!((orig - deq).abs() < 0.01);
    }
}

#[test]
fn kv_k8v4_defaults_off() {
    let prev = std::env::var("GRIM_KV_K8V4").ok();
    unsafe { std::env::remove_var("GRIM_KV_K8V4") };
    assert!(!kv_k8v4_enabled());
    if let Some(v) = prev {
        unsafe { std::env::set_var("GRIM_KV_K8V4", v) };
    }
}
