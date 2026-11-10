//! CPU-only tests for astrum P2 GEMV variant dispatch table.
use grim_backend_rocm::device::compute::gemv_dispatch_table::{
    current_variant, lookup_variant, GemvKey, GemvTable, GemvVariant,
};

#[test]
fn table_lookup_and_default_off() {
    let t = GemvTable::from_entries(
        "gfx110x",
        vec![(
            GemvKey {
                quant: "q4k".to_string(),
                arch: "gfx110x".to_string(),
                tokens: 2,
                rows_class: 1,
            },
            GemvVariant::RowsR2,
            1.70,
        )],
    );
    assert_eq!(
        lookup_variant(
            &t,
            &GemvKey {
                quant: "q4k".to_string(),
                arch: "gfx110x".to_string(),
                tokens: 2,
                rows_class: 1,
            }
        ),
        GemvVariant::RowsR2
    );
    // any field mismatch -> Plain (bit-identical default)
    assert_eq!(
        lookup_variant(
            &t,
            &GemvKey {
                quant: "q6k".to_string(),
                arch: "gfx110x".to_string(),
                tokens: 2,
                rows_class: 1,
            }
        ),
        GemvVariant::Plain
    );
    assert_eq!(
        lookup_variant(
            &t,
            &GemvKey {
                quant: "q4k".to_string(),
                arch: "gfx1030".to_string(),
                tokens: 2,
                rows_class: 1,
            }
        ),
        GemvVariant::Plain
    );
    // table absent (env unset in this test process) -> Plain
    // SAFETY: integration test binary is single-purpose here; no concurrent env mutation.
    unsafe { std::env::remove_var("GRIM_ASTRUM_GEMV_TABLE") };
    assert_eq!(
        current_variant(&GemvKey {
            quant: "q4k".to_string(),
            arch: "gfx110x".to_string(),
            tokens: 2,
            rows_class: 1,
        }),
        GemvVariant::Plain
    );
}
