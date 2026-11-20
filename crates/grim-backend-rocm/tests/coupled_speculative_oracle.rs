use grim_backend_rocm::speculative::{
    coupled_speculative_accept_count, coupled_speculative_enabled,
};

#[test]
fn coupled_speculative_accept_oracle() {
    let p_target = [0.8f32, 0.6, 0.2, 0.9];
    let p_draft = [0.8f32, 0.8, 0.5, 0.9];
    let u = [0.5f32, 0.7, 0.6, 0.1];
    // i=0: 0.5 <= 1.0 -> accept
    // i=1: 0.7 <= 0.75 -> accept
    // i=2: 0.6 <= 0.4 -> reject -> stop at 2, resample=true
    let (accepted, resample) = coupled_speculative_accept_count(&p_target, &p_draft, &u);
    assert_eq!(accepted, 2);
    assert!(resample);
}

#[test]
fn coupled_speculative_defaults_off() {
    let prev = std::env::var("GRIM_COUPLED_SPEC").ok();
    unsafe { std::env::remove_var("GRIM_COUPLED_SPEC") };
    assert!(!coupled_speculative_enabled());
    if let Some(v) = prev {
        unsafe { std::env::set_var("GRIM_COUPLED_SPEC", v) };
    }
}
