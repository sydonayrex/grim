use grim_format::format::{
    ExpertPriorEntry, GRIM_ALIGN_DIRECTIO_UNIT, GRIM_ALIGN_THP_POOL, WAVE32_SEGMENT_BYTES,
    WAVE64_SEGMENT_BYTES, align_to_directio_unit, align_to_segment, align_to_thp_pool,
    decode_expert_priors, encode_expert_priors, read_attachment_bytes, read_attachment_index,
    write_attachments,
};
use std::io::Cursor;

/// Ported from Radiance `tests/format_test.cpp:70-120`:
/// Enforces the three-tier alignment hierarchy (`SUBPLANE 128/256 B` -> `UNIT 4096 B` -> `POOL 2 MiB`):
/// every 2 MiB THP pool boundary is a valid 4 KiB `O_DIRECT` movement unit, and every 4 KiB unit
/// is a valid Wave32 (128 B) and Wave64 (256 B) segment boundary.
#[test]
fn radiance_oracle_three_tier_alignment_hierarchy_invariants() {
    assert_eq!(align_to_directio_unit(0), 0);
    assert_eq!(align_to_directio_unit(1), GRIM_ALIGN_DIRECTIO_UNIT);
    assert_eq!(align_to_directio_unit(4096), 4096);
    assert_eq!(align_to_directio_unit(4097), 8192);

    assert_eq!(align_to_thp_pool(0), 0);
    assert_eq!(align_to_thp_pool(128), GRIM_ALIGN_THP_POOL);
    assert_eq!(align_to_thp_pool(GRIM_ALIGN_THP_POOL), GRIM_ALIGN_THP_POOL);

    for raw_offset in [1u64, 17, 127, 129, 255, 4095, 4097, 65537, 2_097_151, 5_000_003] {
        let sub32 = align_to_segment(raw_offset, WAVE32_SEGMENT_BYTES);
        let sub64 = align_to_segment(raw_offset, WAVE64_SEGMENT_BYTES);
        let unit = align_to_directio_unit(sub64);
        let pool = align_to_thp_pool(unit);

        assert_eq!(sub32 % (WAVE32_SEGMENT_BYTES as u64), 0);
        assert_eq!(sub64 % (WAVE64_SEGMENT_BYTES as u64), 0);
        assert_eq!(unit % GRIM_ALIGN_DIRECTIO_UNIT, 0);
        assert_eq!(unit % (WAVE64_SEGMENT_BYTES as u64), 0);
        assert_eq!(pool % GRIM_ALIGN_THP_POOL, 0);
        assert_eq!(pool % GRIM_ALIGN_DIRECTIO_UNIT, 0);
        assert!(sub32 <= sub64 && sub64 <= unit && unit <= pool);
    }
}

#[test]
fn expert_priors_roundtrip_and_corrupt_payload_refusal() {
    let priors = vec![
        ExpertPriorEntry {
            layer: 0,
            expert: 3,
            share: 0.42,
        },
        ExpertPriorEntry {
            layer: 0,
            expert: 7,
            share: 0.19,
        },
        ExpertPriorEntry {
            layer: 1,
            expert: 2,
            share: 0.31,
        },
    ];
    let blob = encode_expert_priors(&priors);

    let mut file_buf = Cursor::new(vec![0u8; 4096]);
    let written = write_attachments(&mut file_buf, &[("expert_priors".to_string(), blob)])
        .expect("write_attachments");
    assert_eq!(written, 1);

    let index = read_attachment_index(&mut file_buf).expect("read_attachment_index");
    let att = index
        .iter()
        .find(|a| a.kind == "expert_priors")
        .expect("expert_priors attachment");
    let raw = read_attachment_bytes(&mut file_buf, att).expect("read_attachment_bytes");
    let decoded = decode_expert_priors(&raw).expect("decode_expert_priors");
    assert_eq!(decoded, priors);

    // Untrusted input validation (`format_test.cpp`): truncated / non-multiple-of-12 byte payloads fail cleanly.
    for bad_len in [1usize, 11, 13, 23, 25] {
        assert!(
            decode_expert_priors(&vec![0u8; bad_len]).is_err(),
            "expected error on truncated expert_priors len={bad_len}"
        );
    }
}
