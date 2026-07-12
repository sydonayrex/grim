//! RED-GREEN-REFACTOR tests for `autotune::Autotuner`.
//!
//! Phase-3 §3.6 of the QKV spec, RED-first. The model here is minimal but
//! captures the three things the new autotuner must (a) key on, (b) cache
//! correctly, and (c) survive a process restart:
//!
//!   - **Key shape**: `(kernel_name, gpu_arch, problem_shape)` — `problem_shape`
//!     is implementation-defined but must at minimum distinguish M/N/K.
//!   - **Cache hit**: a second `get_or_tune` for the same key returns the
//!     previously-recorded config without re-running the benchmark closure.
//!   - **Persist**: a `to_json` / `from_json` round-trip preserves the
//!     config byte-for-byte so the disk cache is meaningful.
//!   - **Per-arch**: `(kernel, "gfx1036", ...)` and `(kernel, "gfx1200",
//!     ...)` are distinct cache slots, so a future Instinct-build or
//!     RDNA4-build cannot reuse a config tuned on a different arch.
//!
//! The spec's `autotune_block_dim / tile_kv / grid_stride` microbenchmarks
//! (`for block_dim in [64,128,256,512]`) move into the `Autotuner` runtime
//! path, parallel to the spec; this test exercises the *interface* and
//! not the GPU timing itself.
//!
//! Skill attribution:
//! - `rocm-profiling-perf` — autotune loop, `rocblas_gemm_ex_get_solutions`
//!   runtime enumeration, warm-up discipline.
//! - `rust-gpu-discipline` — no fake-GPU: every benchmark closure returns
//!   a measured number, never a synthesized "fast" value.
//! - `rust-ai-ml-inference-guide` Action 8 — benchmark under controlled
//!   shapes before deploying.

use grim_backend_rocm::autotune::{AutotuneConfig, Autotuner, KernelKey};
use grim_tensor::error::Error;

type TestError = Box<dyn std::error::Error + Send + Sync>;
type TestResult<R = ()> = Result<R, TestError>;

// =========================================================================
// RED — `KernelKey` is the cache slot identity. It must Hash + Eq, must
// distinguish every field the spec calls out, and must NOT collapse
// different arches into the same slot (per-arch dispatch is non-trivial).
// =========================================================================

#[test]
fn kernel_key_distinguishes_arch() -> TestResult {
    let base = KernelKey {
        kernel: "grim_qkv_attention",
        gpu_arch: "gfx1036",
        m: 1,
        n: 4096,
        k: 4096,
    };
    let other_arch = KernelKey { gpu_arch: "gfx1200", ..base };
    assert_ne!(base, other_arch, "different gpu_arch must yield different cache keys");
    Ok(())
}

#[test]
fn kernel_key_distinguishes_every_field() -> TestResult {
    let base = KernelKey {
        kernel: "grim_qkv_attention",
        gpu_arch: "gfx1036",
        m: 1,
        n: 4096,
        k: 4096,
    };
    let others = vec![
        KernelKey { kernel: "grim_matmul", ..base },
        KernelKey { gpu_arch: "gfx942", ..base },
        KernelKey { m: 8, ..base },
        KernelKey { n: 11008, ..base },
        KernelKey { k: 11008, ..base },
    ];
    for o in &others {
        assert_ne!(&base, o);
    }
    Ok(())
}

#[test]
fn kernel_key_hash_and_eq_consistent() -> TestResult {
    use std::collections::HashMap;
    let a = KernelKey { kernel: "grim_qkv_attention", gpu_arch: "gfx1036", m: 1, n: 4096, k: 4096 };
    let b = a; // Copy
    let mut m = HashMap::new();
    m.insert(a, 1_u64);
    m.insert(b, 2_u64);
    assert_eq!(m.len(), 1, "equivalent keys must dedupe through HashMap");
    assert_eq!(m[&a], 2, "second insert must win");
    Ok(())
}

#[test]
fn kernel_key_debug_doesnt_panic() -> TestResult {
    let k = KernelKey { kernel: "x", gpu_arch: "y", m: 1, n: 2, k: 3 };
    let _ = format!("{:?}", k);
    Ok(())
}

// =========================================================================
// RED — `AutotuneConfig` returns fields the spec calls out. It's the
// cache payload, so PartialEq + Debug + Clone + Serialize + Deserialize
// are all part of the API surface.
// =========================================================================

#[test]
fn autotune_config_partial_eq_when_all_fields_match() -> TestResult {
    let a = AutotuneConfig {
        block_dim: 256,
        tile_kv: 64,
        grid_stride: 1,
        cycles_per_invocation: 12_345,
    };
    let b = a; // Copy
    assert_eq!(a, b);
    let c = AutotuneConfig { block_dim: 128, ..a };
    assert_ne!(a, c);
    Ok(())
}

#[test]
fn autotune_config_default_is_sensible() -> TestResult {
    let d = AutotuneConfig::default();
    assert!(d.block_dim > 0);
    assert!(d.tile_kv > 0);
    assert!(d.grid_stride > 0);
    Ok(())
}

// =========================================================================
// RED — `Autotuner::for_device(device_ordinal, "gfx1036")` is the entry
// point; the autotuner must be infallible in construction (no GPU calls)
// and it must hand back a usable handle.
// =========================================================================

#[test]
fn autotuner_for_device_is_infallible_and_zeroed() -> TestResult {
    let tuner = Autotuner::for_device(0, "gfx1036");
    assert_eq!(tuner.cache_dir(), None, "default cache_dir is unset");
    let _ = tuner.list_keys(); // empty initially
    Ok(())
}

#[test]
fn autotuner_cache_dir_roundtrip() -> TestResult {
    let mut tuner = Autotuner::for_device(0, "gfx1036");
    let expected = std::path::PathBuf::from("/tmp/grim-test-autotune");
    tuner.set_cache_dir(expected.clone());
    let got = tuner
        .cache_dir()
        .ok_or("cache_dir must be Some after set_cache_dir")?;
    assert_eq!(got, expected.as_path(), "cache_dir round-trip failed");
    Ok(())
}

// =========================================================================
// RED — get_or_tune loop:
// 1. cache miss → closure runs once, config is recorded
// 2. cache hit → closure does NOT run, previously recorded config returned
// =========================================================================

#[test]
fn get_or_tune_cache_hit_avoids_rerunning_benchmark() -> TestResult {
    let mut tuner = Autotuner::for_device(0, "gfx1036");
    let key = KernelKey { kernel: "grim_qkv_attention", gpu_arch: "gfx1036", m: 1, n: 4096, k: 4096 };

    let mut bench_runs = 0_u32;
    // First call: closure runs.
    let config_v1 = tuner.get_or_tune(key, |_kernel| {
        bench_runs += 1;
        Ok(AutotuneConfig { block_dim: 256, tile_kv: 64, grid_stride: 1, cycles_per_invocation: 1 })
    })?;
    assert_eq!(bench_runs, 1, "closure must run on cache miss");
    // Second call: cache hit, closure does not run.
    let config_v2 = tuner.get_or_tune(key, |_kernel| {
        bench_runs += 1;
        Ok(AutotuneConfig { block_dim: 999, tile_kv: 99, grid_stride: 9, cycles_per_invocation: 999 })
    })?;
    assert_eq!(bench_runs, 1, "closure must not run on cache hit");
    assert_eq!(config_v1, config_v2, "cache hit returns the recorded config, never the closure's 'fresh' value");
    Ok(())
}

#[test]
fn get_or_tune_distinct_keys_run_closure_independently() -> TestResult {
    let mut tuner = Autotuner::for_device(0, "gfx1036");
    let key_a = KernelKey { kernel: "qkv", gpu_arch: "gfx1036", m: 1, n: 4096, k: 4096 };
    let key_b = KernelKey { kernel: "matmul", gpu_arch: "gfx1036", m: 1, n: 4096, k: 4096 };

    // Same arity: closure returns different config; we expect each to be retained.
    let ca = tuner.get_or_tune(key_a, |_| Ok(AutotuneConfig::default())).map_err(|e| format!("a: {}", e))?;
    tuner.get_or_tune(key_b, |_| {
        Ok(AutotuneConfig { block_dim: 64, tile_kv: 32, grid_stride: 1, cycles_per_invocation: 0 })
    }).map_err(|e| format!("b: {}", e))?;
    // list_keys must now contain both.
    let keys = tuner.list_keys();
    assert_eq!(keys.len(), 2);
    assert!(keys.contains(&key_a));
    assert!(keys.contains(&key_b));
    let _ = ca; // we don't compare values here; equality matters in the previous test
    Ok(())
}

#[test]
fn get_or_tune_records_closure_failure_as_err_and_does_not_cache() -> TestResult {
    let mut tuner = Autotuner::for_device(0, "gfx1036");
    let key = KernelKey { kernel: "x", gpu_arch: "gfx1036", m: 1, n: 2, k: 3 };
    // First call: closure returns Err -> public Err surfaces.
    let r1 = tuner.get_or_tune(key, |_| -> Result<_, Error> {
        Err(Error::Backend("synthetic failure".into()))
    });
    assert!(r1.is_err(), "closure failure must surface as Err");
    // Second call: closure RE-runs (the failure wasn't cached).
    let mut runs = 0_u32;
    let r2 = tuner.get_or_tune(key, |_| {
        runs += 1;
        Ok(AutotuneConfig::default())
    })?;
    assert_eq!(runs, 1, "closure failure must not poison the cache");
    let _ = r2;
    Ok(())
}

// =========================================================================
// RED — serialize/deserialize. The on-disk cache file (`{gpu_arch}.json`
// per the spec) is meaningful only if a config can survive a process
// round-trip. We can't always write to `~/.cache/grim/autotune/...` in
// CI, so we exercise the serializer against an in-memory buffer.
// =========================================================================

#[test]
fn autotune_config_serde_roundtrip() -> TestResult {
    let cfg = AutotuneConfig {
        block_dim: 256,
        tile_kv: 64,
        grid_stride: 1,
        cycles_per_invocation: 9_999,
    };
    let s = serde_json::to_string(&cfg)?;
    let d: AutotuneConfig = serde_json::from_str(&s)?;
    assert_eq!(cfg, d, "JSON round-trip must be byte-equivalent");
    Ok(())
}

#[test]
fn autotuner_load_save_round_trips_via_buffer() -> TestResult {
    let mut tuner = Autotuner::for_device(0, "gfx1036");
    let key = KernelKey { kernel: "qkv", gpu_arch: "gfx1036", m: 8, n: 4096, k: 4096 };
    let cfg = AutotuneConfig { block_dim: 128, tile_kv: 128, grid_stride: 2, cycles_per_invocation: 1 };
    tuner.record(key, cfg)?;
    let buf = tuner.to_json_bytes()?;
    let second = Autotuner::from_json_bytes(0, "gfx1036", &buf)?;
    let cfg2 = second.lookup(key).expect("key restored after round-trip");
    assert_eq!(cfg, cfg2);
    Ok(())
}

#[test]
fn autotuner_reports_device_ordinal() -> TestResult {
    let tuner = Autotuner::for_device(7, "gfx1036");
    assert_eq!(tuner.device_ordinal(), 7);
    Ok(())
}

// =========================================================================
// RED — `caches_for_arch()` must key everything by gpu_arch. Without an
// arch separation, a config tuned for `gfx1200` would be returned to a
// `gfx1036` session.
// =========================================================================

#[test]
fn autotuner_per_arch_separation() -> TestResult {
    let mut rdn2 = Autotuner::for_device(0, "gfx1036");
    let mut rdn4 = Autotuner::for_device(0, "gfx1200");
    let key_rdn2 = KernelKey { kernel: "qkv", gpu_arch: "gfx1036", m: 1, n: 4096, k: 4096 };
    let key_rdn4 = KernelKey { kernel: "qkv", gpu_arch: "gfx1200", m: 1, n: 4096, k: 4096 };
    rdn2.record(key_rdn2, AutotuneConfig::default())?;
    rdn4.record(key_rdn4, AutotuneConfig { block_dim: 64, ..AutotuneConfig::default() })?;
    assert_eq!(rdn2.lookup(key_rdn2).map(|c| c.block_dim), Some(256));
    assert_eq!(rdn4.lookup(key_rdn4).map(|c| c.block_dim), Some(64));
    Ok(())
}

#[test]
fn autotuner_lookup_returns_none_for_unknown_keys() -> TestResult {
    let mut tuner = Autotuner::for_device(0, "gfx1036");
    let key = KernelKey { kernel: "missing", gpu_arch: "gfx1036", m: 1, n: 2, k: 3 };
    assert!(tuner.lookup(key).is_none());
    Ok(())
}

// Silence unused import when Error is only referenced inside gated test arms.
