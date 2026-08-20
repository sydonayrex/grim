#[test]
fn _wi1_live_print() {
    if crate::device::capability_profiler::enumerate_devices().unwrap_or(0) > 0 {
        println!("LIVE compute_utilization(0) = {:?}", crate::device::capability_profiler::compute_utilization(0));
    }
}
