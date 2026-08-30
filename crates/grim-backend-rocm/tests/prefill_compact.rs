use grim_backend_rocm::kernels::prefill_compact::compact_expert_requests;

#[test]
fn test_prefill_compact_partitioning() {
    // 16 experts total in layer
    let mut slot_table = vec![None; 16];
    slot_table[0] = Some(10);
    slot_table[2] = Some(11);
    slot_table[5] = Some(12);
    slot_table[8] = Some(13);

    let requested = vec![0, 1, 2, 5, 7, 8, 9];
    let compacted = compact_expert_requests(&requested, &slot_table).unwrap();

    // Resident hits
    assert_eq!(compacted.resident, vec![(0, 10), (2, 11), (5, 12), (8, 13)]);
    // PCIe misses
    assert_eq!(compacted.misses, vec![1, 7, 9]);
}

#[test]
fn test_prefill_compact_device_gate() {
    // Gate on ROCm device probe
    let device_visible = match grim_backend_rocm::device::roc_device::RocmDevice::probe_one(0) {
        Ok(true) => true,
        _ => false,
    };
    println!("ROCm device visible for prefill compact test: {}", device_visible);
}
