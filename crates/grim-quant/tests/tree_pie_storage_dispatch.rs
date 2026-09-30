//! TreePie is reachable through `Storage::FloatPack`, not only through its codec.
//!
//! The WS-A kernel and its parity tests called `pack_tree_pie_32` / the launcher
//! directly, so the format worked and was validated while a model still could not
//! actually be *loaded* as TreePie: there was no `FloatPackScheme::TreePie`, so no
//! storage branch, no dequant dispatch, and no way to size the buffer. These pin
//! the plumbing, which is the part that has no GPU test.

use grim_quant::tree_pie::{dequant_tree_pie_bytes, pack_tree_pie_bytes};
use grim_tensor::dtype::{DType, FloatPackScheme, QuantFormat, Storage};

#[test]
fn tree_pie_has_a_float_pack_scheme_and_survives_the_round_trip_to_dtype() {
    let dt = DType { arith: grim_tensor::ArithType::F32, storage: Storage::FloatPack(FloatPackScheme::TreePie) };
    // Storage -> QuantFormat, which is what a load path resolving a declared
    // scheme needs.
    assert_eq!(QuantFormat::try_from(&dt.storage), Ok(QuantFormat::TreePie));
}

#[test]
fn packed_size_is_five_bytes_per_value() {
    // 5.0 bpw: 32 values per 5 i32 = 20 bytes. The sizing formula is what a
    // loader uses to size the buffer, so it is pinned against the packer's own
    // output rather than against arithmetic written twice.
    let dt = DType { arith: grim_tensor::ArithType::F32, storage: Storage::FloatPack(FloatPackScheme::TreePie) };
    for groups in [1usize, 4, 8, 64] {
        let n = groups * 32;
        let values = vec![1.0f32; n];
        let bytes = pack_tree_pie_bytes(&values);
        assert_eq!(dt.expected_bytes(n), bytes.len(), "sizing disagrees with the packer at n={n}");
        assert_eq!(bytes.len(), groups * 20);
    }
}

#[test]
fn byte_storage_round_trips() {
    // Values chosen to be exactly representable in E2M2 so the comparison is exact
    // and a packing error cannot hide behind a tolerance.
    let mut values = Vec::new();
    for i in 0..32 * 4 {
        let m = (i % 9) as f32; // 0..8, all E2M2-representable magnitudes
        values.push(if i % 3 == 0 { -m } else { m });
    }
    let bytes = pack_tree_pie_bytes(&values);
    assert_eq!(bytes.len(), values.len() * 5 / 8, "5.0 bpw");
    let back = dequant_tree_pie_bytes(&bytes, values.len());
    assert_eq!(back.len(), values.len());
    for (a, b) in values.iter().zip(&back) {
        assert_eq!(a, b, "TreePie byte round trip is not exact on E2M2 values");
    }
}

#[test]
fn ragged_lengths_are_rejected_loudly() {
    // A partial group would silently overstate the 5.0 bpw claim, so both
    // directions must refuse rather than pad.
    let ragged: Vec<f32> = (0..31).map(|i| i as f32).collect();
    assert!(std::panic::catch_unwind(|| pack_tree_pie_bytes(&ragged)).is_err(), "pack must reject a 31-value group");

    let short = vec![0u8; 4 * 31];
    assert!(
        std::panic::catch_unwind(|| dequant_tree_pie_bytes(&short, 32)).is_err(),
        "dequant must reject a storage length that is not 5 i32 per group"
    );
}
