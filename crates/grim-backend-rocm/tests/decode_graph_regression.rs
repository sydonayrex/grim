//! CPU-only regression tests for the decode-graph feature.
//!
//! These run in CI without a GPU. They verify the parts of the wiring that do
//! NOT require a live ROCm device: env guards, topology validation, bucket
//! mapping, pool struct layout, and the public API surface. GPU-required
//! behaviour (capture/replay parity, determinism, stress) lives in
//! `grim-models-transformer::tests::lfm2_graph_{capture,p4}`.

use grim_backend_rocm::decode_graph_buffers::{
    check_layer_topology, decode_graph_enabled, launch_attention, launch_qkv_gemv,
    write_embedding_to_buffer, DecodeGraphBuffers,
};
use grim_backend_rocm::decode_graph_buffers::DecodeGraph;
use grim_backend_rocm::{DecodeBatchBucket, FullDecodeGraph};

// ===========================================================================
// 1. Env guard: decode_graph_enabled honours both flags, all spellings
// ===========================================================================

#[test]
fn env_disabled_by_decode_graph_zero() {
    temp_env::with_var("GRIM_DECODE_GRAPH", Some("0"), || {
        assert!(!decode_graph_enabled());
    });
}

#[test]
fn env_disabled_by_capture_graph_zero() {
    temp_env::with_var("GRIM_CAPTURE_GRAPH", Some("0"), || {
        assert!(!decode_graph_enabled());
    });
}

#[test]
fn env_disabled_by_false_off_spellings() {
    // Matches the exact spellings decode_graph_enabled() recognises.
    for v in ["false", "False", "off", "OFF"] {
        temp_env::with_vars(
            [("GRIM_DECODE_GRAPH", None::<&str>), ("GRIM_CAPTURE_GRAPH", None::<&str>)],
            || {
                temp_env::with_var("GRIM_CAPTURE_GRAPH", Some(v), || {
                    assert!(!decode_graph_enabled(), "expected {v} to disable");
                });
            },
        );
    }
}

#[test]
fn env_case_sensitive_exact_match() {
    // Values NOT in the match set are treated as enabled (non-"0"/"false"/"off").
    temp_env::with_var("GRIM_CAPTURE_GRAPH", Some("FALSE"), || {
        assert!(decode_graph_enabled(), "FALSE (all-caps) is not a disable token");
    });
    temp_env::with_var("GRIM_CAPTURE_GRAPH", Some("no"), || {
        assert!(decode_graph_enabled(), "\"no\" is not a disable token");
    });
}

#[test]
fn env_enabled_when_unset() {
    temp_env::with_vars(
        [
            ("GRIM_DECODE_GRAPH", None::<&str>),
            ("GRIM_CAPTURE_GRAPH", None::<&str>),
        ],
        || assert!(decode_graph_enabled()),
    );
}

#[test]
fn env_one_enabled_one_disabled() {
    // DECODE=1 but CAPTURE=0 -> still disabled (either flag disables).
    temp_env::with_vars(
        [
            ("GRIM_DECODE_GRAPH", Some("1")),
            ("GRIM_CAPTURE_GRAPH", Some("0")),
        ],
        || assert!(!decode_graph_enabled()),
    );
}

// ===========================================================================
// 2. Topology validation (no GPU: null-ptr path errors before HIP call)
// ===========================================================================

// ===========================================================================
// 3. Bucket mapping: exhaustive + out-of-range
// ===========================================================================

#[test]
fn bucket_exact_matches() {
    assert_eq!(DecodeBatchBucket::from_batch_size(1), Some(DecodeBatchBucket::B1));
    assert_eq!(DecodeBatchBucket::from_batch_size(2), Some(DecodeBatchBucket::B2));
    assert_eq!(DecodeBatchBucket::from_batch_size(4), Some(DecodeBatchBucket::B4));
    assert_eq!(DecodeBatchBucket::from_batch_size(8), Some(DecodeBatchBucket::B8));
    assert_eq!(DecodeBatchBucket::from_batch_size(16), Some(DecodeBatchBucket::B16));
    assert_eq!(DecodeBatchBucket::from_batch_size(32), Some(DecodeBatchBucket::B32));
}

#[test]
fn bucket_ceiling_matches() {
    assert_eq!(DecodeBatchBucket::from_batch_size(3), Some(DecodeBatchBucket::B4));
    assert_eq!(DecodeBatchBucket::from_batch_size(5), Some(DecodeBatchBucket::B8));
    assert_eq!(DecodeBatchBucket::from_batch_size(9), Some(DecodeBatchBucket::B16));
    assert_eq!(DecodeBatchBucket::from_batch_size(17), Some(DecodeBatchBucket::B32));
    assert_eq!(DecodeBatchBucket::from_batch_size(31), Some(DecodeBatchBucket::B32));
}

#[test]
fn bucket_out_of_range() {
    assert_eq!(DecodeBatchBucket::from_batch_size(0), None);
    assert_eq!(DecodeBatchBucket::from_batch_size(33), None);
    assert_eq!(DecodeBatchBucket::from_batch_size(1000), None);
}

#[test]
fn bucket_all_buckets_ordered() {
    let all = DecodeBatchBucket::all_buckets();
    assert_eq!(all.len(), 6);
    for (i, b) in all.iter().enumerate() {
        assert_eq!(b.batch_size(), 1 << i);
    }
    assert_eq!(all[0], DecodeBatchBucket::B1);
    assert_eq!(all[5], DecodeBatchBucket::B32);
}

// ===========================================================================
// 4. API surface: types are exported and constructible shapes are sound
// ===========================================================================

#[test]
fn api_full_decode_graph_is_decode_graph() {
    // FullDecodeGraph is a type alias for DecodeGraph — assert they have the
    // same size so the alias never silently diverges.
    use std::mem::size_of;
    assert_eq!(size_of::<FullDecodeGraph>(), size_of::<DecodeGraph>());
}

#[test]
fn api_launch_qkv_helpers_are_exported() {
    // These would fail to compile if the functions were not exported.
    let _ = launch_qkv_gemv;
    let _ = launch_attention;
    let _ = check_layer_topology;
    let _ = write_embedding_to_buffer;
    let _ = decode_graph_enabled;
}

// ===========================================================================
// 5. Topology + pool guards are covered in decode_graph_buffers::tests (lib
//    unit tests) which run on CPU-only CI. The env guards above are duplicated
//    here as an integration-level smoke test so a broken re-export is caught
//    at the crate's own test boundary.
// ===========================================================================

#[test]
fn topology_guard_is_exported() {
    // If check_layer_topology's signature ever drifts, this fails to compile.
    let fn_ptr: fn(&DecodeGraphBuffers, usize) -> Result<(), grim_tensor::error::Error> =
        check_layer_topology;
    let _ = fn_ptr;
}
