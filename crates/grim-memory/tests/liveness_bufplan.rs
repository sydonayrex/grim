use grim_memory::bufplan::{BUF_ALIGN_BYTES, BufDecl, BufKind, plan_buffers, plan_level};

#[test]
fn radiance_oracle_greedy_interval_coloring_packs_non_overlapping_transients() {
    // 4-layer transformer loop sharing residual `x` (Persist) while per-layer `qkv` and `mlp`
    // transients have non-overlapping lifetimes [1..2], [3..4], [5..6], [7..8].
    let mut decls = vec![BufDecl {
        name: "x_resid".into(),
        kind: BufKind::Persist,
        bytes: 4096,
        first_def: 0,
        last_use: 8,
    }];
    for l in 0..4 {
        let op_base = (l * 2 + 1) as i32;
        decls.push(BufDecl {
            name: format!("blk.{l}.qkv"),
            kind: BufKind::Transient,
            bytes: 8192,
            first_def: op_base,
            last_use: op_base,
        });
        decls.push(BufDecl {
            name: format!("blk.{l}.mlp"),
            kind: BufKind::Transient,
            bytes: 16384,
            first_def: op_base + 1,
            last_use: op_base + 1,
        });
    }

    let plan = plan_buffers(&decls, 8);
    // Persist region sits at [0, 4096)
    assert_eq!(plan.offsets[0], 0);
    assert_eq!(plan.fixed_bytes, 4096);

    // Unshared transient sum is 4 * (8192 + 16384) = 98_304 B, but peak_live is only 16_384 B!
    assert_eq!(plan.sum_transient, 98_304);
    assert_eq!(plan.peak_live, 16_384);
    // Greedy interval coloring packs all 8 transient buffers into exactly 16_384 B at offset 4096!
    assert_eq!(plan.transient_bytes, 16_384);
    assert_eq!(plan.arena_bytes, 4096 + 16_384);
    for off in &plan.offsets[1..] {
        assert_eq!(*off, 4096);
    }
}

#[test]
fn radiance_oracle_plan_level_pins_persist_and_lends_shrunk_transient_top() {
    let decls = vec![
        BufDecl {
            name: "persist_state".into(),
            kind: BufKind::Persist,
            bytes: 8192,
            first_def: 0,
            last_use: 10,
        },
        BufDecl {
            name: "act_a".into(),
            kind: BufKind::Transient,
            bytes: 1024 * 1024, // 1 MiB at prefill max_tok=4096
            first_def: 1,
            last_use: 5,
        },
        BufDecl {
            name: "act_b".into(),
            kind: BufKind::Transient,
            bytes: 1024 * 1024, // overlaps with act_a at op 3..7
            first_def: 3,
            last_use: 7,
        },
    ];
    let full = plan_buffers(&decls, 10);
    assert_eq!(full.fixed_bytes, 8192);
    assert_eq!(full.transient_bytes, 2 * 1024 * 1024);

    // At decode (rows=16), transients shrink 256x to 4096 B each; Persist stays 8192 B.
    let shrunk = vec![8192, 4096, 4096];
    let level = plan_level(&decls, &full, &shrunk, 10).expect("plan_level");
    // Persist buffer MUST NOT move
    assert_eq!(level.offsets[0], full.offsets[0]);
    assert_eq!(level.fixed_bytes, full.fixed_bytes);
    // Shrunk transients pack into 8192 B above fixed_bytes, freeing > 2 MiB at the top for MoE loan!
    assert_eq!(level.arena_bytes, 8192 + 8192);
    let lendable_bytes = full.arena_bytes - level.arena_bytes;
    assert_eq!(lendable_bytes, 2 * 1024 * 1024 - 8192);
    assert_eq!(level.arena_bytes % BUF_ALIGN_BYTES, 0);

    // Refuses a sizing probe where a transient grew larger than the full plan
    assert!(plan_level(&decls, &full, &[8192, 2 * 1024 * 1024, 4096], 10).is_err());
}
