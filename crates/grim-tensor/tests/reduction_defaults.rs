//! G2 gate: the host-fallback reduction defaults (reduce_sum / reduce_max /
//! argmax) compute correct values through the CPU backend, which does not
//! override them — this pins the default path every backend inherits.

use grim_backend_cpu::cpu_tensor;
use grim_tensor::{ElementwiseOps, Shape};

#[test]
fn host_fallback_reductions_compute_correct_values() {
    let dev = grim_backend_cpu::CpuDevice::new();
    // cpu_tensor yields F32 storage on the CPU device.
    let x = cpu_tensor(vec![1.0, -3.0, 3.5, 2.0], Shape::new(vec![4]));

    assert_eq!(dev.reduce_sum(x.storage().as_ref()).unwrap(), 3.5);
    assert_eq!(dev.reduce_max(x.storage().as_ref()).unwrap(), 3.5);
    assert_eq!(dev.argmax(x.storage().as_ref()).unwrap(), 2, "argmax is 2 (value 3.5)");

    // Last index wins ties (`Iterator::max_by` semantics, matching the
    // greedy sampling path).
    let tied = cpu_tensor(vec![5.0, 5.0], Shape::new(vec![2]));
    assert_eq!(dev.argmax(tied.storage().as_ref()).unwrap(), 1);

    // Empty tensors error instead of fabricating a value.
    let empty = cpu_tensor(Vec::<f32>::new(), Shape::new(vec![0]));
    assert!(dev.reduce_sum(empty.storage().as_ref()).is_err());
    assert!(dev.reduce_max(empty.storage().as_ref()).is_err());
    assert!(dev.argmax(empty.storage().as_ref()).is_err());
}
