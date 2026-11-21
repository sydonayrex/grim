//! TDD tests for Q9 (`GRIM_SPEC_MAY_VERIFY=1`): Basalt `MayVerifyGate` with
//! mandatory probe exception (`kProbeChance = 0.85`) and low->high recovery test.

use std::sync::Mutex;

use grim_speculative::may_verify::{spec_may_verify_enabled, MayVerifyGate};

static ENV_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn env_flag_defaults_off_and_parses_truthy() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let prev = std::env::var("GRIM_SPEC_MAY_VERIFY").ok();
    unsafe { std::env::remove_var("GRIM_SPEC_MAY_VERIFY") };
    assert!(!spec_may_verify_enabled());
    unsafe { std::env::set_var("GRIM_SPEC_MAY_VERIFY", "1") };
    assert!(spec_may_verify_enabled());
    unsafe {
        match prev {
            Some(v) => std::env::set_var("GRIM_SPEC_MAY_VERIFY", v),
            None => std::env::remove_var("GRIM_SPEC_MAY_VERIFY"),
        }
    }
}

#[test]
fn low_acceptance_disables_then_probe_recovers_on_high_acceptance() {
    let mut gate = MayVerifyGate::new(3, 0.35, 0.2, 0.85, 42);
    assert!(gate.should_verify(), "initial healthy EMA must allow verify");

    // 1) Low-acceptance regime: 0/3 accepted for 20 steps -> EMA drops below min_chance (0.35)
    for _ in 0..20 {
        gate.observe(0, 3);
    }
    assert!(
        gate.chance(0) < 0.35,
        "EMA must drop below min_chance after 20 zero-accept steps, got {}",
        gate.chance(0)
    );

    // 2) Confirm gate skips most steps when below threshold, but probe exception
    //    (probe_chance = 0.85 -> 15% probe rate) still fires periodically so
    //    speculation is NEVER permanently disabled.
    let mut skips = 0usize;
    let mut probes = 0usize;
    for _ in 0..200 {
        if gate.should_verify() {
            probes += 1;
            // High-acceptance regime begins: every probed step accepts 3/3
            gate.observe(3, 3);
            if gate.chance(0) >= 0.35 {
                break;
            }
        } else {
            skips += 1;
        }
    }
    assert!(skips > 0, "gate must skip steps during low-acceptance regime");
    assert!(probes > 0, "probe exception must fire during low-acceptance regime");
    assert!(
        gate.chance(0) >= 0.35,
        "EMA must recover >= min_chance via probes, got {}",
        gate.chance(0)
    );
    // 3) Once recovered, continuous verification is re-enabled without skipping.
    for _ in 0..10 {
        assert!(
            gate.should_verify(),
            "recovered gate must verify continuously"
        );
    }
}
