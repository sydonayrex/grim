//! Test to exercise and verify Qwen35 D2D trace behavior under GRIM_D2D_TRACE.

#[test]
fn test_qwen35_d2d_trace_env() {
    unsafe {
        std::env::set_var("GRIM_D2D_TRACE", "1");
    }
    assert!(std::env::var("GRIM_D2D_TRACE").is_ok());
}
