//! MoE routing and Charon kernel dispatchers for `RocmDevice`.

//! Module root: re-exports the routing submodules and keeps the small
//! EPLB/reorder/Scythe-persistent `impl RocmDevice` block.

use std::ffi::c_void;

use grim_tensor::backend::{BackendStorage, ComputeHandle};
use grim_tensor::error::Result;

use crate::device::roc_device::RocmDevice;
use crate::{ arg, as_rocm, dev_ptr, RocmHandle };

mod charon_fused;
mod charon_grouped_dispatch;
mod charon_roundtrip;
mod moe_dispatch;

#[allow(unused_imports)] // flat public API surface: `device::routing::<method>`
pub use charon_fused::*;
#[allow(unused_imports)]
pub use charon_grouped_dispatch::*;
#[allow(unused_imports)]
pub use charon_roundtrip::*;
#[allow(unused_imports)]
pub use moe_dispatch::*;

impl RocmDevice {/// Compute dynamic Expert Parallel Load Balancing (EPLB) greedy LPT placement.
    pub fn eplb_balance_experts(
        &self, expert_frequencies: &[f32], num_ranks: usize, replication_slots: usize, ) -> crate::device::eplb::EplbPackingPlan {
        crate::device::eplb::EplbRouter::balance_experts(
            expert_frequencies, num_ranks, replication_slots, )
    }

    /// Plan continuous batch reordering into [Decode : Extend : Prefill] partitions.
    pub fn reorder_batch(
        &self, sequences: &[crate::device::batch_orchestrator::SequenceMeta], ) -> crate::device::batch_orchestrator::ReorderedBatch {
        crate::device::batch_orchestrator::BatchReorderer::plan(sequences)}

    /// Launch one bounded Scythe persistent worker.
    /// The worker is intentionally launched as a single 128-thread block: the callable Charon device function.
    pub fn launch_scythe_persistent_dispatch(
        &self,
        slots: &dyn BackendStorage,
        capacity: u32,
        tail: &dyn BackendStorage,
        head: &dyn BackendStorage,
        stop: &dyn BackendStorage,
        max_tasks: u32,
        resident: u32,
    ) -> Result<Box<dyn ComputeHandle>> {
        let mut slots_ptr = dev_ptr(as_rocm(slots)?)?;
        let mut tail_ptr = dev_ptr(as_rocm(tail)?)?;
        let mut head_ptr = dev_ptr(as_rocm(head)?)?;
        let mut stop_ptr = dev_ptr(as_rocm(stop)?)?;
        let mut cap = capacity;
        let mut limit = max_tasks;
        let mut res = resident;
        if std::env::var_os("GRIM_RING_DIAG").is_some() {
            eprintln!(
                "[launch-diag] persistent wave: cap={cap} max_tasks={max_tasks} resident={res} stream_nonnull=false"
            );
        }
        self.launch_compute_kernel(
            "grim_scythe_persistent_dispatch",
            crate::HipDim3::new(1, 1, 1),
            crate::HipDim3::new(128, 1, 1),
            &mut [
                arg(&mut slots_ptr),
                arg(&mut cap),
                arg(&mut tail_ptr),
                arg(&mut head_ptr),
                arg(&mut stop_ptr),
                arg(&mut limit),
                arg(&mut res),
            ],
        )?;
        Ok(Box::new(RocmHandle::new(Some(self.active_stream()))))
    }

    /// WI-SB6: launch the persistent worker on an EXPLICIT non-blocking stream.
    /// The batch-mode wrapper above uses the device active stream; resident mode must own its stream.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_scythe_persistent_dispatch_on(
        &self,
        slots: &dyn BackendStorage,
        capacity: u32,
        tail: &dyn BackendStorage,
        head: &dyn BackendStorage,
        stop: &dyn BackendStorage,
        max_tasks: u32,
        resident: u32,
        stream: *mut c_void,
    ) -> Result<Box<dyn ComputeHandle>> {
        let mut slots_ptr = dev_ptr(as_rocm(slots)?)?;
        let mut tail_ptr = dev_ptr(as_rocm(tail)?)?;
        let mut head_ptr = dev_ptr(as_rocm(head)?)?;
        let mut stop_ptr = dev_ptr(as_rocm(stop)?)?;
        let mut cap = capacity;
        let mut limit = max_tasks;
        let mut res = resident;
        if std::env::var_os("GRIM_RING_DIAG").is_some() {
            eprintln!(
                "[launch-diag] persistent wave: cap={cap} max_tasks={max_tasks} resident={res} stream_nonnull={}",
                !stream.is_null()
            );
        }
        self.launch_compute_kernel(
            "grim_scythe_persistent_dispatch",
            crate::HipDim3::new(1, 1, 1),
            crate::HipDim3::new(128, 1, 1),
            &mut [
                arg(&mut slots_ptr),
                arg(&mut cap),
                arg(&mut tail_ptr),
                arg(&mut head_ptr),
                arg(&mut stop_ptr),
                arg(&mut limit),
                arg(&mut res),
            ],
        )?;
        Ok(Box::new(RocmHandle::new(Some(stream))))
    }
}

#[cfg(test)]
mod routing_dispatch_selection_tests {
    use super::*;

    const BATCH: usize = 2;
    const HIDDEN: usize = 32;
    const INTER: usize = 64;
    const NUM_EXPERTS: usize = 2;
    const RSF: f32 = 1.0;

    fn gpu_device() -> Option<RocmDevice> {
        if !crate::gpu_test_enabled() {
            eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
            return None;
        }
        std::panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new")).ok()
    }

    fn assignment() -> crate::kernels::charon::RoutingAssignment {
        crate::kernels::charon::RoutingAssignment {
            tokens: vec![0, 0, 1],
            experts: vec![0, 1, 0],
            weights: vec![0.7, 0.3, 0.5],
        }
    }

    fn activations() -> Vec<f32> {
        (0..BATCH * HIDDEN)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.11)
            .collect()
    }

    fn weights_f32() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mk = |n: usize, salt: f32| -> Vec<f32> {
            (0..NUM_EXPERTS * n)
                .map(|i| ((i % 11) as f32 - 5.0) * 0.07 * salt)
                .collect()
        };
        (mk(HIDDEN * INTER, 1.0), mk(HIDDEN * INTER, 1.3), mk(INTER * HIDDEN, 0.8))
    }

    fn assert_finite_shape(out: Vec<f32>, what: &str) {
        assert_eq!(out.len(), BATCH * HIDDEN, "{what} output shape");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "{what} produced non-finite output: {out:?}"
        );
    }

    /// GGUF Q8_0: per 32 weights, f16 scale + 32 i8 codes.
    fn quant_q80(w: &[f32]) -> Vec<u8> {
        let mut out = vec![0u8; w.len() / 32 * 34];
        for (blk, chunk) in w.chunks(32).enumerate() {
            let amax = chunk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let d = (amax / 127.0).max(1e-30);
            let bits = half::f16::from_f32(d).to_bits();
            let o = &mut out[blk * 34..blk * 34 + 34];
            o[0] = (bits & 0xFF) as u8;
            o[1] = ((bits >> 8) & 0xFF) as u8;
            for (j, &v) in chunk.iter().enumerate() {
                o[2 + j] = ((v / d).round().clamp(-127.0, 127.0) as i8) as u8;
            }
        }
        out
    }

    /// E2M1 nibble codes (values 0..±6), e8m0 scale byte = 127 (2^0).
    fn quant_mxfp4(w: &[f32]) -> Vec<u8> {
        let e2m1 = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let nib = |v: f32| -> u8 {
            let s = (v < 0.0) as u8;
            let a = v.abs();
            let m = e2m1
                .iter()
                .enumerate()
                .min_by(|a_kv, b_kv| {
                    let x = a_kv.1;
                    let y = b_kv.1;
                    (x - a).abs().total_cmp(&(y - a).abs())
                })
                .map(|(i, _)| i as u8)
                .unwrap_or(0);
            (s << 3) | m
        };
        let codes = w.len() / 2;
        let mut out = vec![0u8; codes];
        for (i, pair) in w.chunks(2).enumerate() {
            out[i] = nib(pair[0]) | (nib(*pair.get(1).unwrap_or(&0.0)) << 4);
        }
        out
    }

    /// E4M3 codes for weights snapped to {0, ±0.5, ±1, ±2}; e8m0 scale 127.
    fn quant_e4m3(w: &[f32]) -> Vec<u8> {
        w.iter()
            .map(|&v| match v {
                v if v < 0.25 && v > -0.25 => 0x00u8,
                v if (0.25..0.75).contains(&v) => 0x30,
                v if (-0.75..=-0.25).contains(&v) => 0xB0,
                v if (0.75..1.5).contains(&v) => 0x38,
                v if (-1.5..=-0.75).contains(&v) => 0xB8,
                v if v >= 1.5 => 0x40,
                _ => 0xC0,
            })
            .collect()
    }

    #[test]
    fn grouped_dispatch_f32_reachable() {
        let Some(dev) = gpu_device() else { return };
        let (gw, uw, dw) = weights_f32();
        let out = dev
            .charon_grouped_dispatch_roundtrip(
                &activations(),
                &gw,
                &uw,
                &dw,
                &assignment(),
                BATCH,
                HIDDEN,
                INTER,
                RSF,
            )
            .expect("f32 grouped roundtrip");
        assert_finite_shape(out, "f32 grouped");
    }

    #[test]
    fn grouped_dispatch_wmma_reachable() {
        let Some(dev) = gpu_device() else { return };
        let (gw, uw, dw) = weights_f32();
        let out = dev
            .charon_grouped_dispatch_wmma_roundtrip(
                &activations(),
                &gw,
                &uw,
                &dw,
                &assignment(),
                BATCH,
                HIDDEN,
                INTER,
                RSF,
            )
            .expect("wmma grouped roundtrip");
        assert_finite_shape(out, "wmma grouped");
    }

    #[test]
    fn grouped_dispatch_q80_reachable() {
        let Some(dev) = gpu_device() else { return };
        let (gw, uw, dw) = weights_f32();
        let out = dev
            .charon_grouped_dispatch_roundtrip_q80(
                &activations(),
                &quant_q80(&gw),
                &quant_q80(&uw),
                &quant_q80(&dw),
                &[1.0; BATCH],
                &assignment(),
                BATCH,
                HIDDEN,
                INTER,
                RSF,
            )
            .expect("q80 grouped roundtrip");
        assert_finite_shape(out, "q80 grouped");
    }

    #[test]
    fn grouped_dispatch_iqk_q4k_reachable() {
        let Some(dev) = gpu_device() else { return };
        let (gw, uw, dw) = weights_f32();
        let q4k = |w: &[f32]| -> Vec<u8> {
            assert_eq!(w.len() % 256, 0);
            let mut out = vec![0u8; w.len() / 256 * 144];
            for (sb, superchunk) in w.chunks(256).enumerate() {
                let o = &mut out[sb * 144..sb * 144 + 144];
                let bits = half::f16::from_f32(1.0).to_bits();
                o[0] = (bits & 0xFF) as u8;
                o[1] = ((bits >> 8) & 0xFF) as u8;
                o[2] = (bits & 0xFF) as u8;
                o[3] = ((bits >> 8) & 0xFF) as u8;
                let mut sc_m = [(0u8, 0u8); 8];
                for (is, chunk) in superchunk.chunks(32).enumerate() {
                    let wmin = chunk.iter().copied().fold(f32::INFINITY, f32::min);
                    let wmax = chunk.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let sc = (((wmax - wmin) / 15.0).ceil() as i32).clamp(1, 63) as u8;
                    let m = ((-wmin / sc as f32).round() as i32).clamp(0, 63) as u8;
                    sc_m[is] = (sc, m);
                }
                for is in 0..8 {
                    let (sc, m) = sc_m[is];
                    if is < 4 {
                        o[4 + is] = sc & 63;
                        o[4 + is + 4] = m & 63;
                    } else {
                        o[4 + is + 4] = (sc & 0x0F) | ((m & 0x0F) << 4);
                        o[4 + is - 4] = (o[4 + is - 4] & 0x3F) | ((sc >> 4) << 6);
                        o[4 + is] = (o[4 + is] & 0x3F) | ((m >> 4) << 6);
                    }
                }
                for (is, chunk) in superchunk.chunks(32).enumerate() {
                    let (_, m) = sc_m[is];
                    let group = is / 2;
                    let half_idx = is % 2;
                    for (j, &v) in chunk.iter().enumerate() {
                        let q = ((v + m as f32) / sc_m[is].0 as f32)
                            .round()
                            .clamp(0.0, 15.0) as u8;
                        let idx = 16 + group * 32 + j;
                        if half_idx == 0 {
                            o[idx] |= q & 0x0F;
                        } else {
                            o[idx] |= q << 4;
                        }
                    }
                }
            }
            out
        };
        let out = dev
            .charon_grouped_dispatch_roundtrip_iqk(
                &activations(),
                &q4k(&gw),
                &q4k(&uw),
                &q4k(&dw),
                &[1.0; BATCH],
                &assignment(),
                BATCH,
                HIDDEN,
                INTER,
                7,    // scalar IQK q4k format (golden-tested in golden_charon_moe_gpu)
                144,  // q4k block bytes
                RSF,
            )
            .expect("iqk q4k grouped roundtrip");
        assert_finite_shape(out, "iqk q4k");
    }

    #[test]
    fn grouped_dispatch_mxfp4_reachable() {
        let Some(dev) = gpu_device() else { return };
        let (gw, uw, dw) = weights_f32();
        let e8m0 = || vec![127u8; NUM_EXPERTS * (HIDDEN * INTER / 32)];
        let out = dev
            .charon_grouped_dispatch_roundtrip_mxfp4(
                &activations(),
                &quant_mxfp4(&gw),
                &quant_mxfp4(&uw),
                &quant_mxfp4(&dw),
                &e8m0(),
                &e8m0(),
                &e8m0(),
                &[1.0; BATCH],
                &assignment(),
                BATCH,
                HIDDEN,
                INTER,
                RSF,
            )
            .expect("mxfp4 grouped roundtrip");
        assert_finite_shape(out, "mxfp4");
    }

    #[test]
    fn grouped_dispatch_mxfp8_reachable() {
        let Some(dev) = gpu_device() else { return };
        let (gw, uw, dw) = weights_f32();
        let e8m0 = || vec![127u8; NUM_EXPERTS * (HIDDEN * INTER / 32)];
        let out = dev
            .charon_grouped_dispatch_roundtrip_mxfp8(
                &activations(),
                &quant_e4m3(&gw),
                &quant_e4m3(&uw),
                &quant_e4m3(&dw),
                &e8m0(),
                &e8m0(),
                &e8m0(),
                &[1.0; BATCH],
                &assignment(),
                BATCH,
                HIDDEN,
                INTER,
                RSF,
            )
            .expect("mxfp8 grouped roundtrip");
        assert_finite_shape(out, "mxfp8");
    }

    #[test]
    fn grouped_dispatch_fp8_reachable() {
        let Some(dev) = gpu_device() else { return };
        let (gw, uw, dw) = weights_f32();
        let scales = || vec![1.0f32; NUM_EXPERTS * INTER * HIDDEN.div_ceil(16)];
        let out = dev
            .charon_grouped_dispatch_roundtrip_fp8(
                &activations(),
                &quant_e4m3(&gw),
                &quant_e4m3(&uw),
                &quant_e4m3(&dw),
                &scales(),
                &scales(),
                &scales(),
                &[1.0; BATCH],
                &assignment(),
                BATCH,
                HIDDEN,
                INTER,
                RSF,
            )
            .expect("fp8 grouped roundtrip");
        assert_finite_shape(out, "fp8");
    }

    #[test]
    fn grouped_dispatch_dot4_q80_reachable() {
        let Some(dev) = gpu_device() else { return };
        use crate::kernels::charon::{dot4_entry_for, CharonDot4Quant};
        let (gw, uw, dw) = weights_f32();
        let out = dev
            .charon_grouped_dispatch_roundtrip_dot4(
                dot4_entry_for(CharonDot4Quant::Q8_0, dev.gcn_arch(), false)
                    .expect("dot4 entry for arch"),
                &activations(),
                &quant_q80(&gw),
                &quant_q80(&uw),
                &quant_q80(&dw),
                &[1.0; BATCH],
                &assignment(),
                BATCH,
                HIDDEN,
                INTER,
                NUM_EXPERTS,
                RSF,
            )
            .expect("dot4 q80 grouped roundtrip");
        assert_finite_shape(out, "dot4 q80");
    }

    #[test]
    fn grouped_dispatch_dot4_rejects_unaligned_hidden() {
        let Some(dev) = gpu_device() else { return };
        use crate::kernels::charon::{dot4_entry_for, CharonDot4Quant};
        let (gw, uw, dw) = weights_f32();
        let err = dev.charon_grouped_dispatch_roundtrip_dot4(
            dot4_entry_for(CharonDot4Quant::Q8_0, dev.gcn_arch(), false)
                .expect("dot4 entry for arch"),
            &activations(),
            &quant_q80(&gw),
            &quant_q80(&uw),
            &quant_q80(&dw),
            &[1.0; BATCH],
            &assignment(),
            BATCH,
            8, // hidden not a multiple of 32 -> must be rejected, not mis-computed
            INTER,
            NUM_EXPERTS,
            RSF,
        );
        assert!(err.is_err(), "unaligned hidden must be rejected");
    }
}



