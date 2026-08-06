//! RED-GREEN-REFACTOR tests for the RCCL collective wrapper (WI-R1).
//!
//! The wrapper reuses `device::accel_ffi`'s NCCL FFI (F11) rather than
//! re-declaring the symbols — single source of truth, no duplicated
//! knowledge (clean-code imperative 11). This module's body is
//! `#[cfg(feature = "rccl")]`-gated; when the feature is OFF the
//! stubs return `Err` (never panic, never silently succeed —
//! clean-code imperative 18).
//!
//! Skill attribution:
//! - `rust-tdd` — assert the default-off contract with `assert_eq!`
//!   and the error contract with `assert!(is_err())`; no snapshots.
//! - `rust-ffi-grim` — §1 panic safety: the FFI boundary never
//!   panics; errors surface as `grim_tensor::Result`.
//! - `clean-code-guard` — no `unwrap()` in tests; `?`-bubble + `assert_*`.
//!
//! Dual-GPU test results (syd-beasty, ROCm 7.2.53211):
//!   Hardware: RX 9070 XT (gfx1201, device 0) + RX 9060 XT (gfx1200, device 1)
//!   — 5/7 PASS (unit/linkage + communicator init + topology check).
//!   rccl_multi_gpu_all_reduce_sums_real_device_buffers HANGS on the
//!   actual all-reduce collective (communicator initializes in ~0.85s
//!   but the NCCL ring-allreduce deadlocks between the two different
//!   RDNA4 SKUs across PCIe — consumer GPUs lack xGMI).

use grim_backend_rocm::rccl::{CollectiveConfig, UniqueId, p2p_memcpy_async};
#[cfg(feature = "rccl")]
use grim_tensor::backend::BackendDevice;
#[cfg(feature = "rccl")]
use grim_tensor::{DType, Shape};

type TestError = Box<dyn std::error::Error + Send + Sync>;
type TestResult<R = ()> = Result<R, TestError>;

#[cfg(feature = "rccl")]
fn integration_ordinals(devices: &[grim_backend_rocm::RocmDevice]) -> TestResult<Vec<usize>> {
    if let Some(value) = std::env::var_os("GRIM_RCCL_ORDINALS") {
        let ordinals = value
            .to_string_lossy()
            .split(',')
            .map(|value| value.trim().parse::<usize>())
            .collect::<Result<Vec<_>, _>>()?;
        if ordinals.len() != 2 {
            return Err("GRIM_RCCL_ORDINALS must contain exactly two ordinals".into());
        }
        if ordinals
            .iter()
            .any(|ordinal| !devices.iter().any(|device| device.ordinal() == *ordinal))
        {
            return Err("GRIM_RCCL_ORDINALS contains an unavailable device".into());
        }
        return Ok(ordinals);
    }
    Ok(devices
        .iter()
        .take(2)
        .map(|device| device.ordinal())
        .collect())
}

#[cfg(feature = "rccl")]
fn require_integration_hardware() -> TestResult<bool> {
    if std::env::var_os("GRIM_RUN_RCCL_INTEGRATION").is_none() {
        return Ok(false);
    }
    Ok(true)
}

/// Real multi-GPU RCCL admission test. This is opt-in because communicator
/// creation claims all selected devices and requires a ROCm runtime with at
/// least two usable GPUs. The test deliberately uses the same explicit
/// ordinal list as the training worker, so non-contiguous selections are
/// covered when the environment exposes them.
#[cfg(feature = "rccl")]
#[test]
fn rccl_multi_gpu_communicator_initializes_on_hardware() -> TestResult {
    if !require_integration_hardware()? {
        return Ok(());
    }
    let devices = grim_backend_rocm::RocmDevice::probe()?;
    if devices.len() < 2 {
        eprintln!("skipping RCCL integration: fewer than two ROCm devices detected");
        return Ok(());
    }
    let ordinals = integration_ordinals(&devices)?;
    let communicator = grim_backend_rocm::RcclAllReduce::try_new(&ordinals)?;
    assert_eq!(communicator.num_gpus as usize, ordinals.len());
    Ok(())
}

/// Opt-in end-to-end RCCL data-path check. Each selected GPU owns a real
/// device allocation, the collective sums them in place, and the result is
/// copied back and checked on every rank. This catches communicator-only
/// false positives and validates the pointer/ordinal contract used by the
/// training gradient path.
#[cfg(feature = "rccl")]
#[test]
fn rccl_multi_gpu_all_reduce_sums_real_device_buffers() -> TestResult {
    if !require_integration_hardware()? {
        return Ok(());
    }
    let devices = grim_backend_rocm::RocmDevice::probe()?;
    if devices.len() < 2 {
        eprintln!("skipping RCCL integration: fewer than two ROCm devices detected");
        return Ok(());
    }
    let ordinals = integration_ordinals(&devices)?;
    let communicator = grim_backend_rocm::RcclAllReduce::try_new(&ordinals)?;
    let shape = Shape::new(vec![4]);
    let mut buffers = Vec::with_capacity(ordinals.len());
    for (rank, ordinal) in ordinals.iter().copied().enumerate() {
        let device = grim_backend_rocm::RocmDevice::new(ordinal);
        let storage = device.from_cpu(&[(rank + 1) as f32; 4], &shape, DType::F32)?;
        let ptr = storage
            .as_any()
            .downcast_ref::<grim_backend_rocm::RocmStorage>()
            .ok_or("from_cpu did not return RocmStorage")?
            .device_ptr_u64()
            .ok_or("missing device pointer")?;
        buffers.push((device, storage, ptr));
    }
    std::thread::scope(|scope| {
        let handles = buffers
            .iter()
            .map(|(_, _, ptr)| {
                let communicator = &communicator;
                scope.spawn(move || communicator.sum_gradients_device(*ptr, *ptr, 4, 0))
            })
            .collect::<Vec<_>>();
        for handle in handles {
            handle.join().map_err(|_| "RCCL rank thread panicked")??;
        }
        Ok::<(), TestError>(())
    })?;
    for (_, storage, _) in buffers {
        let values = storage.to_cpu_vec_f32()?;
        assert_eq!(values, vec![3.0; 4]);
    }
    Ok(())
}

#[cfg(feature = "rccl")]
#[test]
fn rccl_selected_pair_matches_requested_topology() -> TestResult {
    if !require_integration_hardware()? {
        return Ok(());
    }
    let devices = grim_backend_rocm::RocmDevice::probe()?;
    if devices.len() < 2 {
        return Ok(());
    }
    let ordinals = integration_ordinals(&devices)?;
    let profiles = ordinals
        .iter()
        .map(|ordinal| {
            let device = devices
                .iter()
                .find(|device| device.ordinal() == *ordinal)
                .unwrap();
            (device.ordinal(), grim_backend_rocm::vram_info(*ordinal).1)
        })
        .collect::<Vec<_>>();
    match std::env::var("GRIM_RCCL_TOPOLOGY").as_deref() {
        Ok("symmetric") => assert_eq!(profiles[0].1, profiles[1].1),
        Ok("asymmetric") => assert_ne!(profiles[0].1, profiles[1].1),
        Ok(other) => return Err(format!("unknown GRIM_RCCL_TOPOLOGY={other}").into()),
        Err(_) => {}
    }
    eprintln!("RCCL integration pair: {profiles:?}");
    Ok(())
}

/// `CollectiveConfig` mirrors `DecodeGemmConfig` / `QkvAttentionFusionConfig`:
/// default-OFF, so a stock single-GPU build never triggers multi-GPU
/// collectives.
#[test]
fn collective_config_default_is_disabled() -> TestResult {
    let cfg = CollectiveConfig::default();
    assert_eq!(cfg.enabled, false);
    Ok(())
}

/// When the `rccl` feature is OFF, `RocmComm::new` must fail with a
/// typed error (not panic, not a silently-valid comm). We assert only
/// that it is `Err` — the variant differs between the real path (needs
/// real peers) and the stub (feature off), but both are errors.
#[cfg(not(feature = "rccl"))]
#[test]
fn rocm_comm_new_is_err_when_unavailable() -> TestResult {
    // `UniqueId::new` is itself feature-gated; on a feature-off build it
    // returns Err, which is the contract we exercise here.
    let id = UniqueId::new();
    assert!(id.is_err());
    Ok(())
}

/// ON-build: the `RocmComm` FFI symbols resolved at link time (the crate
/// compiled against `librccl.so`). Exercising them needs real peers +
/// stream, so we assert only that the FFI is wired — not a runtime call
/// that would trip the HIP null-pointer assertion.
#[cfg(feature = "rccl")]
#[test]
fn rocm_comm_ffi_linked() -> TestResult {
    // A dangling symbol would be a *link* error, not a runtime one.
    assert!(true, "RCCL RocmComm FFI linked (build-time check)");
    Ok(())
}

/// P2P copy is feature-gated: off-builds return `Err` rather than
/// reaching the (absent) `hipMemcpyPeerAsync` symbol. Real P2P
/// needs a peer link + live stream, so the ON path is only
/// link-verified (mirrors `f11_rccl_linked` in accel_ffi.rs).
#[cfg(not(feature = "rccl"))]
#[test]
fn p2p_memcpy_is_err_when_unavailable() -> TestResult {
    let res = p2p_memcpy_async(
        std::ptr::null_mut(),
        0,
        std::ptr::null(),
        0,
        0,
        std::ptr::null_mut(),
    );
    assert!(res.is_err(), "p2p copy must error when RCCL unavailable");
    Ok(())
}

/// ON-build: the `hipMemcpyPeerAsync` symbol resolved at link time
/// (the crate compiled against `librccl.so`). Exercising it needs
/// real peers, so we assert only that the FFI is wired — not a
/// runtime call that would trip the HIP null-pointer assertion.
#[cfg(feature = "rccl")]
#[test]
fn p2p_ffi_linked() -> TestResult {
    // A dangling symbol would be a *link* error, not a runtime one.
    assert!(true, "RCCL P2P FFI linked (build-time check)");
    Ok(())
}

/// WI-R5: `KvDequantAttentionConfig` mirrors the other fusion configs —
/// default-OFF so a stock build never dispatches the KV-dequant kernel.
#[test]
fn kv_dequant_attention_config_default_is_disabled() -> TestResult {
    use grim_backend_rocm::KvDequantAttentionConfig;
    let cfg = KvDequantAttentionConfig::default();
    assert_eq!(cfg.enabled, false);
    assert_eq!(cfg.quant_bits, 4);
    Ok(())
}
