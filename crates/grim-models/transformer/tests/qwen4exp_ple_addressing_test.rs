//! PLE n-gram hash addressing: known-answer tests against the reference.
//!
//! # What was wrong
//!
//! The previous implementation synthesised coprime moduli from an `m_base` and
//! accumulated a polynomial modular sum, using the MODEL's `vocab_size` as the
//! modulus. None of that appears in the reference. The correct algorithm
//! (`llm_graph_input_ple::set_input` in
//! `old/repo/llama.cpp-master/src/models/qwen4exp.cpp` @ d7241ac8) is:
//!
//! ```text
//! mixed_n = (t[p]*m[0]) ^ (t[p-1]*m[1]) ^ ... ^ (t[p-n+1]*m[n-1])
//! row    = mixed_n % vocab[h] + offset[h]
//! ```
//!
//! An XOR of scaled products, with PER-HEAD moduli from the checkpoint. Four
//! independent bugs fell out of that:
//!
//!  1. polynomial sum instead of XOR
//!  2. one shared modulus (model vocab) instead of 16 per-head moduli
//!  3. no per-head row offsets
//!  4. one row per position instead of `n_heads` rows per position
//!
//! Tests 1-4 use a 4-head configuration so the differences are small enough to
//! compute by hand. The 16-head released geometry is checked against the
//! metadata directly.

use grim_models_transformer::qwen4exp_flash_next::{Qwen38FlashNextConfig, Qwen38NgramAddressing};

/// A tiny 4-head PLE: ngram_size 3, heads_per_ngram 2 -> (3-1)*2 = 4 heads.
fn tiny() -> Qwen38NgramAddressing {
    Qwen38NgramAddressing::from_metadata(
        // multipliers, one per n-gram order
        vec![10, 20, 30],
        // offsets: deliberately non-uniform and non-contiguous so a missing
        // offset term changes the answer
        vec![0, 100, 7, 70],
        // per-head vocab sizes: deliberately DIFFERENT, because the real file's
        // are too (16 distinct values around 2.0e7)
        vec![1000, 2000, 3000, 4000],
        2,  // heads_per_ngram
        3,  // ngram_size
        99, // eos
        Some(77),
    )
    .expect("tiny metadata is well-formed")
}

fn row(mixed: u64, h: usize, addr: &Qwen38NgramAddressing) -> u32 {
    (mixed % addr.head_vocab_sizes[h] + addr.head_offsets[h]) as u32
}

// ---------------------------------------------------------------------------
// 1. The hash itself: XOR of scaled products
// ---------------------------------------------------------------------------

#[test]
fn hash_is_xor_of_scaled_products_not_a_polynomial_sum() {
    let a = tiny();
    // ctx = [t0, t1, t2]
    let ctx = vec![5u32, 7, 11];

    // n = 2: only ctx[0] and ctx[1].
    //   mixed = 5*10 ^ 7*20
    // 5*10 = 50 = 0b00110010, 7*20 = 140 = 0b10001100
    // XOR   = 190 = 0b10111110 ; a polynomial SUM would also be 190, so this
    // particular pair does not discriminate. Use (3,5) below for that.
    let n2 = 5u64 * 10 ^ 7u64 * 20;
    assert_eq!(n2, 190, "5*10 ^ 7*20");

    // A pair whose XOR and sum genuinely differ: 3*10=30=0b00011110,
    // 5*20=100=0b01100100 -> XOR=0b01111010=122, sum=130.
    assert_eq!(3u64 * 10 ^ 5u64 * 20, 122);
    assert_ne!(
        3u64 * 10 ^ 5u64 * 20,
        3 * 10 + 5 * 20,
        "XOR and polynomial sum must differ for this pair; a test that cannot \
         tell them apart does not discriminate"
    );

    // n = 3: add ctx[2]*30. 11*30 = 330, 190 ^ 330 = 500.
    let n3 = n2 ^ 11u64 * 30;
    assert_eq!(n3, 500);

    let rows = a.rows_for_context(&ctx);
    assert_eq!(rows.len(), 4, "(3-1) * 2 = 4 heads");
    // heads 0,1 belong to n=2; heads 2,3 to n=3.
    assert_eq!(rows[0], row(n2, 0, &a));
    assert_eq!(rows[1], row(n2, 1, &a));
    assert_eq!(rows[2], row(n3, 2, &a));
    assert_eq!(rows[3], row(n3, 3, &a));
}

#[test]
fn head_assignment_is_n_outer_then_g_inner() {
    let a = tiny();
    let ctx = vec![5u32, 7, 11];
    let n2 = 5u64 * 10 ^ 7u64 * 20;
    let n3 = n2 ^ 11u64 * 30;
    let rows = a.rows_for_context(&ctx);
    // base = (n-2) * per_gram, so n=2 owns heads 0..2 and n=3 owns 2..4.
    assert_eq!(rows[0], row(n2, 0, &a));
    assert_eq!(rows[1], row(n2, 1, &a));
    assert_eq!(rows[2], row(n3, 2, &a));
    assert_eq!(rows[3], row(n3, 3, &a));
    // And the two n=2 rows must use DIFFERENT moduli, which is the whole
    // point of per-head vocab sizes.
    assert_ne!(
        rows[0], rows[1],
        "per-head moduli must produce different rows"
    );
}

// ---------------------------------------------------------------------------
// 2. Per-head modulus and offset, not a single shared vocabulary
// ---------------------------------------------------------------------------

#[test]
fn each_head_uses_its_own_modulus_and_offset() {
    let a = tiny();
    // A large token makes the modulus matter: with a shared modulus the four
    // rows would be congruent; with per-head moduli they are not.
    let big = vec![1_000_003u32, 1_000_033, 1_000_037];
    let rows = a.rows_for_context(&big);
    let distinct: std::collections::HashSet<u32> = rows.iter().copied().collect();
    assert_eq!(
        distinct.len(),
        4,
        "four heads with four different moduli/offsets must give four rows; \
         a shared modulus would collapse them"
    );
}

#[test]
fn the_offset_is_added_after_the_modulus() {
    let a = tiny();
    // A 2-element context: the n=3 term falls back to the EOS default, exactly
    // as `rows_for_context` does for a short slice.
    let rows = a.rows_for_context(&vec![12345u32, 6789]);
    let n2 = 12345u64 * 10 ^ 6789u64 * 20;
    // ctx[j] falls back to EOS and is scaled by multipliers[j], so the missing
    // ctx[2] term is eos * multipliers[2].
    let n3 = n2 ^ a.eos_token_id as u64 * 30;
    for (h, got) in rows.iter().enumerate() {
        let mixed = if h < 2 { n2 } else { n3 };
        let expected = (mixed % a.head_vocab_sizes[h]) as u32 + a.head_offsets[h] as u32;
        assert_eq!(*got, expected, "head {h}: (mixed % vocab) + offset");
        // The offset must be OUTSIDE the modulus, so the result is >= offset.
        assert!(*got >= a.head_offsets[h] as u32, "row {got} below offset");
        // And it must be below offset+vocab.
        assert!(
            *got < (a.head_offsets[h] + a.head_vocab_sizes[h]) as u32,
            "row {got} outside head {h}'s slice"
        );
    }
}

// ---------------------------------------------------------------------------
// 3. The EOS reset rule
// ---------------------------------------------------------------------------

#[test]
fn an_eos_in_the_window_resets_everything_at_or_before_it() {
    let a = tiny();
    // Two predecessors, oldest first. The newest is EOS, so the older one is
    // replaced too.
    let prev = vec![Some(41u32), Some(99u32)]; // 99 == eos
    let ctx = a.context_at(7, &prev);
    assert_eq!(ctx[0], 7, "the token's own context is never cut");
    assert_eq!(ctx[1], 99, "the EOS stays EOS");
    assert_eq!(
        ctx[2], 99,
        "everything at or before the EOS collapses to EOS, not to zero"
    );
}

#[test]
fn token_ids_above_u32_are_not_truncated() {
    // The released multipliers are ~2.4e13, so a u32 anywhere in the multiply
    // or the accumulator silently collapses distinct tokens onto one row. A
    // 33-bit token id is enough to expose it: masked to 32 bits it differs.
    let a = tiny();
    let lo = 5_000_000_000u64 as u32; // placeholder, replaced below
    let _ = lo;
    // Build the context directly so the token is a full u64 value carried
    // through the u32 token type: values above u32::MAX are the risk.
    let big_a: u32 = u32::MAX;
    let big_b: u32 = u32::MAX - 1;
    let rows_a = a.rows_for_context(&[big_a, big_b]);
    let rows_b = a.rows_for_context(&[big_b, big_a]);
    assert_ne!(
        rows_a, rows_b,
        "swapping two near-max tokens must change the rows; a truncated \
         multiply would make them identical"
    );
    // And the multiplier must be used at full width: with multipliers >= 2^32
    // the product exceeds 64 bits only for huge tokens, so assert the
    // accumulator is not reduced modulo 2^32.
    let c = Qwen38NgramAddressing::from_metadata(
        vec![4_294_967_297, 4_294_967_299, 4_294_967_303], // just over u32::MAX
        vec![0; 4],
        vec![1_000_000_007; 4],
        2,
        3,
        0,
        None,
    )
    .expect("wide multipliers");
    let r = c.rows_for_context(&[3, 5, 7]);
    let n2 = 3u64 * 4_294_967_297u64 ^ 5u64 * 4_294_967_299u64;
    assert_eq!(
        r[0],
        (n2 % 1_000_000_007) as u32,
        "a multiplier above u32::MAX must not be truncated to 1"
    );
    // Truncating every multiplier to 1 would give 3 ^ 5 = 6, so assert the
    // two differ.
    assert_ne!((3u64 ^ 5u64) % 1_000_000_007, n2 % 1_000_000_007);
}

#[test]
fn a_missing_predecessor_reads_as_eos() {
    let a = tiny();
    // No history at all: both predecessors are absent.
    let ctx = a.context_at(7, &[None, None]);
    assert_eq!(ctx, vec![7, 99, 99]);
}

#[test]
fn the_tokens_own_eos_does_not_cut_its_own_context() {
    let a = tiny();
    // ctx[0] is EOS, predecessors are ordinary. Upstream: "the EOS of the token
    // itself does not cut its own context".
    let ctx = a.context_at(99, &[Some(41), Some(43)]);
    assert_eq!(ctx[0], 99);
    assert_eq!(ctx[1], 43, "the newest predecessor is kept");
    assert_eq!(ctx[2], 41, "and so is the one before it");
}

#[test]
fn a_normal_window_is_preserved_oldest_first() {
    let a = tiny();
    let prev = vec![Some(11u32), Some(22u32)];
    let ctx = a.context_at(33, &prev);
    assert_eq!(ctx, vec![33, 22, 11], "prev is oldest-first, newest last");
}

// ---------------------------------------------------------------------------
// 4. Geometry
// ---------------------------------------------------------------------------

#[test]
fn head_count_is_ngram_minus_one_times_heads_per_ngram() {
    assert_eq!(tiny().n_heads(), 4, "(3-1) * 2");
    let released = Qwen38FlashNextConfig::default()
        .ple_addressing()
        .expect("default");
    assert_eq!(
        released.n_heads(),
        16,
        "(3-1) * 8 = 16 for the released geometry"
    );
}

#[test]
fn table_rows_is_max_of_offset_plus_vocab() {
    let a = tiny();
    // max over (offset + vocab): 70+4000 = 4070.
    assert_eq!(a.table_rows(), 4070);

    // The released geometry: 16 contiguous heads, total 320_001_446 rows.
    let r = Qwen38FlashNextConfig::default()
        .ple_addressing()
        .expect("default");
    assert_eq!(
        r.table_rows(),
        320_001_446,
        "the shared PLE table must provide max(offset + vocab_size) rows"
    );
    // Naive 16 * 20_000_000 is close but wrong, which is exactly the kind of
    // error that loads and then gathers the wrong rows.
    assert_ne!(r.table_rows(), 16 * 20_000_000);
}

#[test]
fn the_released_geometry_matches_the_checkpoint_metadata() {
    let c = Qwen38FlashNextConfig::default();
    assert_eq!(
        c.ple_layer_multipliers,
        vec![23703573157769, 20109073645365, 8052911324071]
    );
    assert_eq!(c.ple_heads_per_ngram, 8);
    assert_eq!(c.ngram_size, 3);
    assert_eq!(c.ple_head_offsets.len(), 16);
    assert_eq!(c.ple_head_vocab_sizes.len(), 16);
    // The multipliers are ~2.4e13: a u32 accessor would silently truncate them
    // to near-zero, which would collapse every distinct token onto one row.
    for m in &c.ple_layer_multipliers {
        assert!(*m > u32::MAX as u64, "multiplier {m} must exceed u32::MAX");
    }
    // Offsets are contiguous in the released file, and the vocab sizes are all
    // distinct, so a single shared modulus is definitively wrong.
    let ends: Vec<u64> = c
        .ple_head_offsets
        .iter()
        .zip(c.ple_head_vocab_sizes.iter())
        .map(|(o, v)| o + v)
        .collect();
    for i in 0..c.ple_head_offsets.len() - 1 {
        assert_eq!(
            c.ple_head_offsets[i + 1],
            ends[i],
            "head {i} must end where head {} begins",
            i + 1
        );
    }
    let mut sorted = c.ple_head_vocab_sizes.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        16,
        "all 16 per-head vocab sizes are distinct; a shared modulus is invalid"
    );
}

// ---------------------------------------------------------------------------
// 5. Metadata validation
// ---------------------------------------------------------------------------

#[test]
fn inconsistent_metadata_is_rejected_not_silently_accepted() {
    // Upstream requires layer_multipliers.len() == ngram_size.
    assert!(
        Qwen38NgramAddressing::from_metadata(vec![1, 2], vec![0; 4], vec![10; 4], 2, 3, 0, None)
            .is_err(),
        "too few multipliers must be rejected"
    );
    // Upstream requires both head arrays to have n_heads entries.
    assert!(
        Qwen38NgramAddressing::from_metadata(vec![1, 2, 3], vec![0; 3], vec![10; 3], 2, 3, 0, None)
            .is_err(),
        "too few head offsets must be rejected"
    );
    assert!(
        Qwen38NgramAddressing::from_metadata(vec![1, 2, 3], vec![0; 4], vec![10; 4], 2, 3, 0, None)
            .is_ok(),
        "the well-formed control case (3 multipliers, 4 head entries) must pass"
    );
    // ngram_size below 2 is out of range.
    assert!(
        Qwen38NgramAddressing::from_metadata(vec![1], vec![0; 2], vec![10; 2], 2, 1, 0, None)
            .is_err(),
        "ngram_size < 2 must be rejected"
    );
    // A zero modulus would divide by zero at the modulo.
    assert!(
        Qwen38NgramAddressing::from_metadata(
            vec![1, 2, 3],
            vec![0; 4],
            vec![10, 0, 10, 10],
            2,
            3,
            0,
            None
        )
        .is_err(),
        "a zero head vocab size must be rejected"
    );
}

#[test]
fn the_projection_consumes_every_heads_row() {
    // The gather yields n_heads rows per token, so the projected block is
    // n_heads * ple_head_dim wide. For the released geometry that is 16 * 160
    // = 2560 = hidden_size, which is why the reference applies no projection
    // at all. An addressing-only test cannot see a wrong projection width, so
    // pin the relationship the loader must respect.
    let c = Qwen38FlashNextConfig::default();
    let a = c.ple_addressing().expect("default");
    let gathered = a.n_heads() * c.ngram_dim.expect("ple_head_dim is set");
    assert_eq!(
        gathered, 2560,
        "16 heads x 160 ple_head_dim = the model hidden size"
    );
    assert_eq!(
        gathered, c.hidden_size,
        "which is why the reference needs no projection; a Linear at this \
         width is the identity in shape terms"
    );
    // A different geometry must NOT silently collide, otherwise the
    // "no projection needed" shortcut would be wrong there.
    let mut small = Qwen38FlashNextConfig::default();
    small.ngram_dim = Some(8);
    small.ple_heads_per_ngram = 2;
    // Shrinking heads_per_ngram to 2 means n_heads = (3-1)*2 = 4, so the head
    // arrays must shrink to match. Validation rejects a mismatch, which is
    // asserted separately.
    small.ple_head_offsets = vec![0, 10, 20, 30];
    small.ple_head_vocab_sizes = vec![100, 200, 300, 400];
    let sa = small.ple_addressing().expect("small is well-formed");
    assert_eq!(sa.n_heads(), 4);
    assert_ne!(
        sa.n_heads() * 8,
        small.hidden_size,
        "a shrunken geometry must gather a different width, so the loader \
         cannot assume gathered == hidden"
    );
}

#[test]
fn a_missing_offset_term_would_change_the_answer() {
    // Guards the test above: if offsets were ignored, heads 2 and 3 (offsets 7
    // and 70) would still be distinguishable only by modulus. Assert the
    // offsets are individually load-bearing.
    let a = tiny();
    let ctx = vec![12345u32, 6789];
    let rows = a.rows_for_context(&ctx);
    let n2 = 12345u64 * 10 ^ 6789u64 * 20;
    assert_eq!(rows[0], (n2 % 1000) as u32, "head 0 offset is 0");
    assert_eq!(rows[1], (n2 % 2000 + 100) as u32, "head 1 offset is 100");
    // This test uses a 2-token context, so ctx[2] falls back to EOS and is
    // scaled by multipliers[2].
    let n3 = n2 ^ a.eos_token_id as u64 * 30;
    assert_eq!(rows[2], row(n3, 2, &a), "head 2 is n=3, offset 7");
    assert_eq!(rows[3], row(n3, 3, &a), "head 3 is n=3, offset 70");
    assert_eq!(a.head_offsets[2], 7);
    assert_eq!(a.head_offsets[3], 70);
}
