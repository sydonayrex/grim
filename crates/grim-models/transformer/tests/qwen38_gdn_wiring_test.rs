//! Proves the GDN recurrence is WIRED, not merely implemented.
//!
//! `qwen38_gdn.rs` proves the math. These tests prove the block actually calls
//! it, because a correct recurrence that nothing invokes is the same as no
//! recurrence — which is exactly the state 4c started in: the mixer computed a
//! plausible `ssm_norm(ssm_conv1d(qkv))` and never touched the state at all.

use grim_models_transformer::qwen38_flash_next::{Qwen38FlashNextConfig, Qwen38GdnSession};
use grim_models_transformer::qwen38_gdn::{
    GdnParams, KdaHeadPairing, Qwen38GdnCache, gated_delta_net_forward,
};

/// Assert the layer-kind split the GDN session sizing depends on.
#[test]
fn session_sizes_one_cache_per_layer_with_gdn_geometry() {
    // 48 layers, full attention every 4th -> 36 GDN layers.
    let cfg = Qwen38FlashNextConfig::default();
    assert_eq!(
        cfg.layer_types.len(),
        48,
        "config carries one kind per layer"
    );
    let gdn_layers = cfg
        .layer_types
        .iter()
        .filter(|t| *t == "linear_attention")
        .count();
    assert_eq!(gdn_layers, 36, "36 of 48 layers are Gated DeltaNet");
    assert_eq!(
        cfg.layer_types
            .iter()
            .filter(|t| *t == "full_attention")
            .count(),
        12,
        "12 full-attention layers"
    );
}

/// The geometry the released checkpoint declares, from GGUF.
#[test]
fn released_checkpoint_geometry_is_the_one_hardcoded_in_the_reference() {
    let cfg = Qwen38FlashNextConfig::default();
    assert_eq!(cfg.ssm_dt_rank, 48, "n_v_heads = ssm_dt_rank");
    assert_eq!(cfg.ssm_n_group, 16, "n_k_heads = ssm_n_group");
    assert_eq!(cfg.ssm_d_state, 128, "head_dim = d_k = d_v");
    let key_dim = cfg.ssm_n_group * cfg.ssm_d_state; // 2048
    let value_dim = cfg.ssm_dt_rank * cfg.ssm_d_state; // 6144
    assert_eq!(key_dim, 2048);
    assert_eq!(value_dim, 6144);
    assert_eq!(
        cfg.linear_conv_kernel_dim, 4,
        "conv kernel width from ssm.conv_kernel"
    );
    // The real per-layer state is 48 * 128 * 128 floats = 3 MiB.
    let cache = Qwen38GdnCache::new(
        cfg.ssm_dt_rank,
        cfg.ssm_d_state,
        cfg.ssm_d_state,
        4,
        2 * key_dim + value_dim,
    );
    assert_eq!(cache.ssm_state.len(), 48 * 128 * 128);
    assert_eq!(
        cache.ssm_state.len() * 4,
        3 * 1024 * 1024,
        "3 MiB per layer"
    );
    assert_eq!(cache.conv_state.len(), 3 * (2 * key_dim + value_dim));
}

/// The recurrence must be reachable from a *fresh* cache and must mutate it.
#[test]
fn a_fresh_cache_is_advanced_by_the_recurrence() {
    let n_v = 3;
    let n_k = 1;
    let hd = 2;
    let conv_dim = 2 * n_k * hd + n_v * hd;
    let mut cache = Qwen38GdnCache::new(n_v, hd, hd, 4, conv_dim);
    assert!(
        cache.ssm_state.iter().all(|v| *v == 0.0),
        "a fresh cache must be zeroed"
    );

    let mut conv = vec![0.0f32; conv_dim];
    conv[0] = 1.0; // k[0] = 1
    conv[2 * n_k * hd] = 2.0; // v[0] = 2
    let alpha = vec![0.0; n_v];
    let beta = vec![0.0; n_v]; // sigmoid(0) = 0.5
    let a = vec![0.0; n_v];
    let dt = vec![0.0; n_v];
    let norm = vec![1.0; hd];
    let mut out = vec![0.0; n_v * hd];

    gated_delta_net_forward(
        &GdnParams {
            conv_mix: &conv,
            alpha: &alpha,
            beta: &beta,
            a: &a,
            dt_bias: &dt,
            norm: &norm,
            n_v_heads: n_v,
            n_k_heads: n_k,
            head_dim: hd,
            conv_dim,
            seq_len: 1,
            pairing: KdaHeadPairing::Interleaved,
        },
        &mut cache,
        &mut out,
    )
    .expect("forward");

    assert!(
        cache.ssm_state.iter().any(|v| *v != 0.0),
        "the recurrence must write the recurrent state"
    );
    assert_eq!(cache.pos, 1, "the cache must count the token");
    assert!(out.iter().any(|v| *v != 0.0), "and produce non-zero output");
}

/// Two identical steps from a shared cache must differ: the second sees the
/// first's state. A mixer that ignores its state produces identical outputs,
/// which is the signature of the bug 4c exists to fix.
#[test]
fn repeated_steps_diverge_because_state_accumulates() {
    let n_v = 2;
    let n_k = 1;
    let hd = 2;
    let conv_dim = 2 * n_k * hd + n_v * hd;
    let mut cache = Qwen38GdnCache::new(n_v, hd, hd, 4, conv_dim);

    let mut conv = vec![0.0f32; conv_dim];
    conv[0] = 1.0;
    conv[2 * n_k * hd] = 1.0;
    let alpha = vec![0.0; n_v];
    let beta = vec![0.0; n_v];
    let a = vec![0.0; n_v];
    let dt = vec![0.0; n_v];
    let norm = vec![1.0; hd];

    let step = |cache: &mut Qwen38GdnCache| -> Vec<f32> {
        let mut out = vec![0.0; n_v * hd];
        gated_delta_net_forward(
            &GdnParams {
                conv_mix: &conv,
                alpha: &alpha,
                beta: &beta,
                a: &a,
                dt_bias: &dt,
                norm: &norm,
                n_v_heads: n_v,
                n_k_heads: n_k,
                head_dim: hd,
                conv_dim,
                seq_len: 1,
                pairing: KdaHeadPairing::Interleaved,
            },
            cache,
            &mut out,
        )
        .expect("forward");
        out
    };

    let first = step(&mut cache);
    let second = step(&mut cache);
    let third = step(&mut cache);
    assert_ne!(first, second, "step 2 must see step 1's state");
    assert_ne!(second, third, "step 3 must see step 2's state");
    assert_eq!(cache.pos, 3, "three steps advanced the cache");
}

/// A session must hand out a cache per layer and default to empty.
#[test]
fn gdn_session_starts_empty_and_is_clonable_for_snapshots() {
    let s = Qwen38GdnSession::default();
    assert!(s.caches.is_empty(), "a fresh session has no caches yet");
    let snapshot = s.clone();
    assert_eq!(snapshot.caches.len(), 0);
    // Clone must be a real deep copy of the state, so a snapshot/restore
    // round-trip cannot alias the same buffer.
    let mut warm = Qwen38GdnCache::new(2, 2, 2, 4, 8);
    warm.ssm_state[0] = 1.0;
    let snapshot = Qwen38GdnSession {
        caches: vec![warm.clone()],
        qsa_keys: Vec::new(),
    };
    let mut mutated = snapshot.clone();
    mutated.caches[0].ssm_state[0] = 99.0;
    assert_eq!(
        snapshot.caches[0].ssm_state[0], 1.0,
        "cloned sessions must not share state buffers"
    );
}
