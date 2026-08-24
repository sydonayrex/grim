//! Verify compute_utilization(ordinal) matches rocm-smi reality per ordinal.
use grim_backend_rocm::compute_utilization;

fn main() {
    for ord in 0..3 {
        println!("ordinal {ord} -> compute_utilization = {:?}",
            compute_utilization(ord));
    }
}
