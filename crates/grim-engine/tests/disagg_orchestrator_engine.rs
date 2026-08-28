//! §5.6 disagg failover wiring gate: the engine auto-arms a
//! `DisaggOrchestrator` from `disagg_config`, evaluates peer heartbeats, and
//! exposes the effective role that gates the remote KV handoff.

use grim_disagg::{DisaggConfig, PoolRole};
use grim_engine::{Engine, EngineConfig};

fn decode_engine(timeout_ms: u64) -> Engine {
    Engine::new(EngineConfig {
        disagg_config: Some(DisaggConfig {
            role: PoolRole::Decode,
            prefill_addr: "127.0.0.1:0".into(),
            decode_addr: "127.0.0.1:0".into(),
        }),
        disagg_heartbeat_timeout_ms: timeout_ms,
        ..Default::default()
    })
}

#[test]
fn orchestrator_auto_arms_from_disagg_config() {
    let engine = decode_engine(5_000);
    // Auto-armed: configured role until failover evaluation says otherwise.
    assert_eq!(engine.disagg_effective_role(), PoolRole::Decode);
}

#[test]
fn orchestrator_fails_over_after_heartbeat_timeout() {
    let engine = decode_engine(1_000);

    // Peer (prefill) proves alive at t=1_000.
    engine.disagg_record_peer_heartbeat(PoolRole::Prefill, 1_000);
    assert_eq!(
        engine.disagg_evaluate_failover(1_500),
        PoolRole::Decode,
        "fresh heartbeat keeps the configured role"
    );

    // Peer silent past the timeout → colocated fallback.
    assert_eq!(
        engine.disagg_evaluate_failover(1_000 + 1_001),
        PoolRole::Colocated,
        "silent peer must fail the node over to colocated"
    );
    assert_eq!(engine.disagg_effective_role(), PoolRole::Colocated);

    // A fresh heartbeat restores the configured role.
    engine.disagg_record_peer_heartbeat(PoolRole::Prefill, 1_000 + 1_002);
    assert_eq!(
        engine.disagg_evaluate_failover(1_000 + 1_003),
        PoolRole::Decode,
        "recovered peer restores remote execution"
    );
}

#[test]
fn orchestrator_without_heartbeat_never_fails_over() {
    // A peer that has never been seen is not declared dead by the
    // orchestrator's conservative contract (no fabricated failover).
    let engine = decode_engine(1);
    assert_eq!(
        engine.disagg_evaluate_failover(10_000),
        PoolRole::Decode,
        "never-seen peer must not flip the role"
    );
}

#[test]
fn orchestrator_external_instance_is_used() {
    // A caller-provided orchestrator is adopted, not replaced.
    let orch = std::sync::Arc::new(std::sync::Mutex::new(grim_disagg::DisaggOrchestrator::new(
        DisaggConfig {
            role: PoolRole::Decode,
            prefill_addr: "127.0.0.1:0".into(),
            decode_addr: "127.0.0.1:0".into(),
        },
    )));
    let engine = Engine::new(EngineConfig {
        disagg_config: Some(DisaggConfig {
            role: PoolRole::Decode,
            prefill_addr: "127.0.0.1:0".into(),
            decode_addr: "127.0.0.1:0".into(),
        }),
        disagg_orchestrator: Some(orch.clone()),
        disagg_heartbeat_timeout_ms: 1_000,
        ..Default::default()
    });
    // State flows through the SHARED instance both ways.
    orch.lock()
        .unwrap()
        .record_heartbeat(PoolRole::Prefill, 500);
    assert_eq!(engine.disagg_evaluate_failover(600), PoolRole::Decode);
    assert_eq!(
        engine.disagg_evaluate_failover(500 + 1_001),
        PoolRole::Colocated
    );
}
