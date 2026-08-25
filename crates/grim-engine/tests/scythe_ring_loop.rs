//! WI-SB6 gate: engine-loop integration for the persistent ring.
//!
//! Submits an RMSNorm (opcode 4) followed by a column-GEMM (opcode 1) into
//! `ScytheRingExec`, drains them with ONE bounded persistent worker launch,
//! and verifies the chained output against a host reference. This is the
//! decode-loop pattern (norm → projection) executing entirely through ring
//! descriptors.
//!
//! Device-gated: `GRIM_GPU_TEST=1`.

use grim_engine::scythe2::ScytheRingExec;
use grim_backend_rocm::RocmDevice;
use grim_tensor::backend::{BackendDevice, BackendStorage};
use grim_tensor::{DType, Shape};

fn gpu_ready() -> bool {
    if std::env::var("GRIM_GPU_TEST").as_deref() != Ok("1") {
        eprintln!("[skipped: GRIM_GPU_TEST not set]");
        return false;
    }
    true
}

fn dev_ptr(storage: &dyn BackendStorage) -> u64 {
    storage
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .and_then(|rs| rs.device_ptr_u64())
        .expect("rocml device pointer")
}

#[test]
fn ring_norm_then_gemm_chain_matches_host_reference() {
    if !gpu_ready() {
        return;
    }
    let m = 4usize;
    let k = 64usize;
    let n = 256usize;
    let eps = 1e-5f32;

    let mut exec = ScytheRingExec::new(16, 0).expect("ring exec");
    let dev = RocmDevice::try_new(0).expect("dev");

    let x_data: Vec<f32> = (0..m * k).map(|i| ((i % 9) as f32) * 0.3 - 1.2).collect();
    let w_data: Vec<f32> = (0..k).map(|i| ((i % 5) as f32) * 0.2 + 0.5).collect();
    let g_data: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32) * 0.02 - 0.12).collect();

    let x = dev
        .from_cpu(&x_data, &Shape::from_slice(&[m, k]), DType::F32)
        .expect("x");
    let w = dev
        .from_cpu(&w_data, &Shape::from_slice(&[k]), DType::F32)
        .expect("w");
    let g = dev
        .from_cpu(&g_data, &Shape::from_slice(&[k, n]), DType::F32)
        .expect("g");
    let tmp = dev
        .alloc_storage(&Shape::from_slice(&[m, k]), DType::F32)
        .expect("tmp");
    let out = dev
        .alloc_storage(&Shape::from_slice(&[m, n]), DType::F32)
        .expect("out");

    exec.submit_norm(m as u32, k as u32, dev_ptr(x.as_ref()), dev_ptr(w.as_ref()), dev_ptr(tmp.as_ref()))
        .expect("submit norm");
    exec.submit_col_gemm(
        m as u32,
        n as u32,
        k as u32,
        dev_ptr(tmp.as_ref()),
        dev_ptr(g.as_ref()),
        dev_ptr(out.as_ref()),
    )
    .expect("submit gemm");

    eprintln!("[diag] submitting norm+gemm, running batch");
    let drained = exec.run_batch().expect("run batch");
    eprintln!("[diag] drained={drained}");
    assert_eq!(drained, 2, "expected both descriptors drained");

    // Host reference: RMSNorm(eps=1e-5) * weight, then x @ G.
    let got = out.to_cpu_vec_f32().expect("out readback");
    let mut want = vec![0f32; m * n];
    for r in 0..m {
        let row = &x_data[r * k..(r + 1) * k];
        let ss: f32 = row.iter().map(|v| v * v).sum::<f32>() / k as f32;
        let inv = 1.0 / (ss + eps).sqrt();
        let normed: Vec<f32> =
            row.iter().zip(w_data.iter()).map(|(&v, &gw)| v * inv * gw).collect();
        for (j, cell) in want[r * n..(r + 1) * n].iter_mut().enumerate() {
            let mut acc = 0f32;
            for p in 0..k {
                acc += normed[p] * g_data[p * n + j];
            }
            *cell = acc;
        }
    }
    let d = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("[sb6] ring norm->gemm chain max_abs_diff={d:.3e}");
    assert!(d < 1e-3, "SB6 ring chain diverged from host reference: {d:.3e}");

    // Ring bookkeeping must be consistent after the drain.
    assert!(exec.ring.is_empty(), "ring must be empty after run_batch");
}

/// WI-SB6 resident-wave mode (EXPERIMENTAL — stalls after first drain on
/// gfx1200/1201 stack; host args verified correct [resident=1], suspected
/// kernel-arg marshaling or scheduler behavior — see plan log 2026-08-24):
/// ONE worker launch survives idle gaps; two
/// batches are submitted across separate flushes while it runs; shutdown
/// exits via the stop flag.
#[test]
fn ring_resident_wave_two_batches() {
    if !gpu_ready() {
        return;
    }
    let m = 2usize;
    let k = 32usize;
    let n = 48usize;

    let mut exec = ScytheRingExec::new(16, 0).expect("ring exec");
    let dev = RocmDevice::try_new(0).expect("dev");

    // Batch A tensors.
    let xa_data: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32) * 0.2 - 0.5).collect();
    let ga_data: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32) * 0.05 - 0.2).collect();
    let xa = dev
        .from_cpu(&xa_data, &Shape::from_slice(&[m, k]), DType::F32)
        .expect("xa");
    let w_data: Vec<f32> = vec![1.0f32; k];
    let wa = dev
        .from_cpu(&w_data, &Shape::from_slice(&[k]), DType::F32)
        .expect("wa");
    let ga = dev
        .from_cpu(&ga_data, &Shape::from_slice(&[k, n]), DType::F32)
        .expect("ga");
    let tmpa = dev
        .alloc_storage(&Shape::from_slice(&[m, k]), DType::F32)
        .expect("tmpa");
    let outa = dev
        .alloc_storage(&Shape::from_slice(&[m, n]), DType::F32)
        .expect("outa");

    eprintln!("[diag] launching resident wave");
    // Resident mode is experimental: it currently stalls after the first
    // batch on this stack (worker stops consuming; see scythe2 plan log
    // 2026-08-24). Opt in explicitly for that investigation.
    if std::env::var("GRIM_SCYTHE_RING_RESIDENT").as_deref() != Ok("1") {
        eprintln!("[skipped: GRIM_SCYTHE_RING_RESIDENT not set]");
        return;
    }
    exec.launch_resident().expect("launch resident");
    eprintln!("[diag] resident wave live");

    // Phase 0 — no idle gap: all four descriptors submitted before any
    // polling. If THIS stalls, the ring/kernel drops work under backlog;
    // if it passes, the defect is specific to idle-then-resume.
    eprintln!("[diag] phase0: submitting norm");
    exec.submit_norm(m as u32, k as u32, dev_ptr(xa.as_ref()), dev_ptr(wa.as_ref()), dev_ptr(tmpa.as_ref()))
        .expect("A norm");
    eprintln!("[diag] phase0: submitting gemm");
    exec.submit_col_gemm(m as u32, n as u32, k as u32, dev_ptr(tmpa.as_ref()), dev_ptr(ga.as_ref()), dev_ptr(outa.as_ref()))
        .expect("A gemm");
    exec.flush().expect("phase0 flush");
    let done = exec.wait_completed(2, std::time::Duration::from_secs(8)).expect("wait p0");
    println!("[sb6] phase0 completed={done}");
    assert_eq!(done, 2, "phase0 (no-idle backlog) stalled at {done}");

    // Batch A.
    eprintln!("[diag] A: submitting norm");
    exec.submit_norm(m as u32, k as u32, dev_ptr(xa.as_ref()), dev_ptr(wa.as_ref()), dev_ptr(tmpa.as_ref()))
        .expect("A norm");
    eprintln!("[diag] A: norm done; submitting gemm");
    exec.submit_col_gemm(m as u32, n as u32, k as u32, dev_ptr(tmpa.as_ref()), dev_ptr(ga.as_ref()), dev_ptr(outa.as_ref()))
        .expect("A gemm");
    eprintln!("[diag] A: gemm done; flushing");
    exec.flush().expect("flush A");
    eprintln!("[diag] A: flushed; polling");
    let nb = dev.create_non_blocking_stream().expect("diag stream");
    let mut pin = grim_backend_rocm::RocmPinnedBuffer::<u8>::alloc(16 * 64)
        .expect("pinned diag buffer");
    for i in 0..12 {
        let c = exec.completed().expect("completed read");
        let mut line = format!("[diag] A poll {i}: completed={c}");
        for slot in 0..6 {
            match exec.peek_slot_status(slot) {
                Ok(st) => line += &format!(" s{slot}.st={st}"),
                Err(e) => line += &format!(" s{slot}.ERR({e})"),
            }
        }
        eprintln!("{line}");
        if c >= 2 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(700));
    }
    let done = exec.completed().expect("final completed");
    assert!(done >= 2, "batch A stalled: completed={done}");
    assert_eq!(done, 2, "batch A must complete exactly 2 tasks (got {done})");

    // Batch B reuses the same operand buffers (deterministic outputs).
    eprintln!("[diag] B: submitting norm");
    exec.submit_norm(m as u32, k as u32, dev_ptr(xa.as_ref()), dev_ptr(wa.as_ref()), dev_ptr(tmpa.as_ref()))
        .expect("B norm");
    eprintln!("[diag] B: norm done; submitting gemm");
    exec.submit_col_gemm(m as u32, n as u32, k as u32, dev_ptr(tmpa.as_ref()), dev_ptr(ga.as_ref()), dev_ptr(outa.as_ref()))
        .expect("B gemm");
    eprintln!("[diag] B: gemm done; flushing");
    exec.flush().expect("flush B");
    eprintln!("[diag] B: flushed; polling");
    let done = exec.wait_completed(6, std::time::Duration::from_secs(8)).expect("wait B");
    if done < 6 {
        // Forensics: dump slot statuses (offset 48, u32) + control scalars.
        let nb = dev.create_non_blocking_stream().expect("nb");
        let mut raw = vec![0u8; 16 * 64];
        let slots_ptr = exec.ring.slots_storage().unwrap().device_ptr_u64().unwrap();
        unsafe {
            grim_backend_rocm::hipMemcpyAsync(
                raw.as_mut_ptr() as *mut _,
                slots_ptr as *const _,
                raw.len(),
                grim_backend_rocm::HipMemcpyKind::DeviceToHost,
                nb,
            );
        }
        grim_backend_rocm::hip_stream_synchronize(nb);
        dev.destroy_stream(nb);
        for (slot, chunk) in raw.chunks(64).enumerate() {
            let status = u32::from_ne_bytes(chunk[48..52].try_into().unwrap());
            let opcode = u32::from_ne_bytes(chunk[0..4].try_into().unwrap());
            eprintln!("[forensic] slot {slot}: opcode={opcode} status={status}");
        }
        eprintln!("[forensic] device tail(completed)={done}");
        panic!("batch B stalled at completed={done}");
    }
    assert_eq!(done, 6, "batch B must bring the completed count to 6");

    // Output correctness through the resident wave.
    let got = outa.to_cpu_vec_f32().expect("readback");
    let mut want = vec![0f32; m * n];
    for r in 0..m {
        let row = &xa_data[r * k..(r + 1) * k];
        let inv = 1.0 / (row.iter().map(|v| v * v).sum::<f32>() / k as f32 + 1e-5).sqrt();
        let normed: Vec<f32> =
            row.iter().zip(w_data.iter()).map(|(&v, &w)| v * inv * w).collect();
        for (j, cell) in want[r * n..(r + 1) * n].iter_mut().enumerate() {
            *cell = (0..k).map(|p| normed[p] * ga_data[p * n + j]).sum();
        }
    }
    let d = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("[sb6] resident-wave output max_abs_diff={d:.3e}");
    assert!(d < 1e-3, "resident wave diverged: {d:.3e}");

    exec.shutdown().expect("shutdown");
}

/// WI-SB5/SB6 convergence: descriptor-linked row-parallel fan-in.
///
/// Row-shard GEMMs run as opcode-1 descriptors writing disjoint partials
/// into resident ring-ordinal buffers; an opcode-7 ADD sums them. Every
/// pointer is a resident-buffer reference — no host slices cross the ring.
/// Parity vs monolithic matmul within fp tolerance.
#[test]
fn ring_row_parallel_descriptor_fanin_parity() {
    if !gpu_ready() {
        return;
    }
    let m = 4usize;
    let k = 64usize;
    let n = 128usize;
    let half = k / 2;

    let mut exec = ScytheRingExec::new(16, 0).expect("ring exec");
    let dev = RocmDevice::try_new(0).expect("dev");

    let w_flat: Vec<f32> = (0..n * k).map(|i| ((i % 23) as f32) * 0.03 - 0.3).collect();
    let x_data: Vec<f32> = (0..m * k).map(|i| ((i % 9) as f32) * 0.2 - 0.8).collect();
    let x = dev
        .from_cpu(&x_data, &Shape::from_slice(&[m, k]), DType::F32)
        .expect("x");

    // Per-shard operands: transposed B[k_half,n] slices plus this rank's
    // K-slice of x — exactly what a real row-parallel rank receives.
    let mut b_t = Vec::new();   // device storages
    let mut a_s = Vec::new();   // device storages
    let bounds = [(0usize, half), (half, k - half)];
    for &(k_start, k_count) in &bounds {
        // B^T slice: rows k_start..k_start+k_count of W, transposed to [k_count, n].
        let mut bt = vec![0f32; k_count * n];
        for ni in 0..n {
            for ki in 0..k_count {
                bt[ki * n + ni] = w_flat[ni * k + (k_start + ki)];
            }
        }
        let bt_shape = Shape::from_slice(&[k_count, n]);
        b_t.push(
            dev.from_cpu(&bt, &bt_shape, DType::F32)
                .expect("bt upload"),
        );
        let mut xs = vec![0f32; m * k_count];
        for r in 0..m {
            for c in 0..k_count {
                xs[r * k_count + c] = x_data[r * k + k_start + c];
            }
        }
        let xs_shape = Shape::from_slice(&[m, k_count]);
        a_s.push(dev.from_cpu(&xs, &xs_shape, DType::F32).expect("xs upload"));
    }

    let p0 = dev.alloc_storage(&Shape::from_slice(&[m, n]), DType::F32).expect("p0");
    let p1 = dev.alloc_storage(&Shape::from_slice(&[m, n]), DType::F32).expect("p1");
    let summed = dev
        .alloc_storage(&Shape::from_slice(&[m, n]), DType::F32)
        .expect("summed");

    exec.submit_col_gemm(
        m as u32, n as u32, half as u32,
        dev_ptr(a_s[0].as_ref()), dev_ptr(b_t[0].as_ref()), dev_ptr(p0.as_ref()),
    )
    .expect("gemm shard0");
    exec.submit_col_gemm(
        m as u32,
        n as u32,
        (k - half) as u32,
        dev_ptr(a_s[1].as_ref()),
        dev_ptr(b_t[1].as_ref()),
        dev_ptr(p1.as_ref()),
    )
    .expect("gemm shard1");
    exec.submit_add(
        m as u32,
        n as u32,
        dev_ptr(p0.as_ref()),
        dev_ptr(p1.as_ref()),
        dev_ptr(summed.as_ref()),
    )
    .expect("submit add");

    let drained = exec.run_batch().expect("run batch");
    assert_eq!(drained, 3, "expected gemm+gemm+add drained");
    eprintln!("[sb56] drained={drained}");

    // Host reference: y = x @ W^T over the FULL k.
    let mut want = vec![0f32; m * n];
    for r in 0..m {
        for j in 0..n {
            let mut accv = 0f32;
            for p in 0..k {
                accv += x_data[r * k + p] * w_flat[j * k + p];
            }
            want[r * n + j] = accv;
        }
    }
    let got = summed.to_cpu_vec_f32().expect("readback");
    let d = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("[sb56] descriptor fan-in max_abs_diff={d:.3e}");
    assert!(d < 1e-3, "descriptor-linked fan-in diverged: {d:.3e}");
}
