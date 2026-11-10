//! CPU-only tests for the astrum measured BLASLt/rocBLAS solution table (P1).
//! No GPU required — must not be #[ignore].
use grim_backend_rocm::device::gemm_tuning_measured::{
    load_measured_table, lookup_measured_solution, table_version_guard_str, MeasuredEntry,
    MeasuredTable,
};
use grim_tensor::ArithType;

#[test]
fn table_roundtrip_and_lookup() {
    let table = MeasuredTable {
        arch: "gfx110x".into(),
        library: "blaslt".into(),
        blaslt_version: "12345".into(),
        entries: vec![MeasuredEntry {
            m: 4096,
            n: 4096,
            k: 4096,
            arith: "f32".to_string(),
            solution: 73,
            speedup: 1.8,
        }],
    };
    let path = std::env::temp_dir().join("astrum_blaslt_table_test.json");
    std::fs::write(&path, serde_json::to_string(&table).unwrap()).unwrap();
    let loaded = load_measured_table(path.to_str().unwrap()).unwrap();
    assert_eq!(loaded.arch, "gfx110x");
    // exact hit (M bucketed to pow2-ceil >= 8)
    assert_eq!(
        lookup_measured_solution(&loaded, 4096, 4096, 4096, "gfx110x", ArithType::F32),
        Some(73)
    );
    // M-bucket miss -> None (fall through to hand-seeded table)
    assert_eq!(
        lookup_measured_solution(&loaded, 7, 4096, 4096, "gfx110x", ArithType::F32),
        None
    );
    // arch mismatch -> None
    assert_eq!(
        lookup_measured_solution(&loaded, 4096, 4096, 4096, "gfx1030", ArithType::F32),
        None
    );
}

#[test]
fn version_guard_rejects_mismatch() {
    let table = MeasuredTable {
        arch: "gfx110x".into(),
        library: "blaslt".into(),
        blaslt_version: "v9".into(),
        entries: vec![],
    };
    assert!(table_version_guard_str(&table, "v9"));
    assert!(!table_version_guard_str(&table, "v8"));
}
