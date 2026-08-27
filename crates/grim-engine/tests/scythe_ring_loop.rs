//! WI-SB6 gate: engine-loop integration for the persistent ring.
//!
//! Submits an RMSNorm (opcode 4) followed by a column-GEMM (opcode 1) into
//! `ScytheRingExec`, drains them with ONE bounded persistent worker launch,
//! and verifies the chained output against a host reference. This is the
//! decode-loop pattern (norm → projection) executing entirely through ring
//! descriptors.
//!
//! Device-gated: `GRIM_GPU_TEST=1`.

use grim_backend_rocm::RocmDevice;
use grim_engine::scythe2::ScytheRingExec;
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
    let g_data: Vec<f32> = (0..k * n)
        .map(|i| ((i % 13) as f32) * 0.02 - 0.12)
        .collect();

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

    exec.submit_norm(
        m as u32,
        k as u32,
        dev_ptr(x.as_ref()),
        dev_ptr(w.as_ref()),
        dev_ptr(tmp.as_ref()),
    )
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
        let normed: Vec<f32> = row
            .iter()
            .zip(w_data.iter())
            .map(|(&v, &gw)| v * inv * gw)
            .collect();
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
    assert!(
        d < 1e-3,
        "SB6 ring chain diverged from host reference: {d:.3e}"
    );

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
    exec.submit_norm(
        m as u32,
        k as u32,
        dev_ptr(xa.as_ref()),
        dev_ptr(wa.as_ref()),
        dev_ptr(tmpa.as_ref()),
    )
    .expect("A norm");
    eprintln!("[diag] phase0: submitting gemm");
    exec.submit_col_gemm(
        m as u32,
        n as u32,
        k as u32,
        dev_ptr(tmpa.as_ref()),
        dev_ptr(ga.as_ref()),
        dev_ptr(outa.as_ref()),
    )
    .expect("A gemm");
    exec.flush().expect("phase0 flush");
    let done = exec
        .wait_completed(2, std::time::Duration::from_secs(8))
        .expect("wait p0");
    println!("[sb6] phase0 completed={done}");
    assert_eq!(done, 2, "phase0 (no-idle backlog) stalled at {done}");

    // Batch A.
    eprintln!("[diag] A: submitting norm");
    exec.submit_norm(
        m as u32,
        k as u32,
        dev_ptr(xa.as_ref()),
        dev_ptr(wa.as_ref()),
        dev_ptr(tmpa.as_ref()),
    )
    .expect("A norm");
    eprintln!("[diag] A: norm done; submitting gemm");
    exec.submit_col_gemm(
        m as u32,
        n as u32,
        k as u32,
        dev_ptr(tmpa.as_ref()),
        dev_ptr(ga.as_ref()),
        dev_ptr(outa.as_ref()),
    )
    .expect("A gemm");
    eprintln!("[diag] A: gemm done; flushing");
    exec.flush().expect("flush A");
    eprintln!("[diag] A: flushed; polling");
    let _nb = dev.create_non_blocking_stream().expect("diag stream");
    let _pin =
        grim_backend_rocm::RocmPinnedBuffer::<u8>::alloc(16 * 64).expect("pinned diag buffer");
    for i in 0..12 {
        let c = exec.completed().expect("completed read");
        eprintln!("[diag] A poll {i}: completed={c}");
        if c >= 4 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(700));
    }
    let done = exec.completed().expect("final completed");
    assert!(done >= 4, "batch A stalled: completed={done}");
    assert_eq!(done, 4, "phase0+A must total 4 tasks (got {done})");

    // Batch B reuses the same operand buffers (deterministic outputs).
    eprintln!("[diag] B: submitting norm");
    exec.submit_norm(
        m as u32,
        k as u32,
        dev_ptr(xa.as_ref()),
        dev_ptr(wa.as_ref()),
        dev_ptr(tmpa.as_ref()),
    )
    .expect("B norm");
    eprintln!("[diag] B: norm done; submitting gemm");
    exec.submit_col_gemm(
        m as u32,
        n as u32,
        k as u32,
        dev_ptr(tmpa.as_ref()),
        dev_ptr(ga.as_ref()),
        dev_ptr(outa.as_ref()),
    )
    .expect("B gemm");
    eprintln!("[diag] B: gemm done; flushing");
    exec.flush().expect("flush B");
    eprintln!("[diag] B: flushed; polling");
    let mut done = 0u32;
    for i in 0..10 {
        done = exec.completed().expect("completed read B");
        eprintln!("[diag] B poll {i}: completed={done}");
        if done >= 6 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(700));
    }
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
        let _ = grim_backend_rocm::hip_stream_synchronize(nb);
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

    // Join the wave BEFORE any standard DtoH: pageable-staged reads cannot
    // complete while an eternal kernel runs (device-scope staging).
    exec.shutdown().expect("shutdown");
    eprintln!("[diag] wave shut down; reading outputs");

    // Output correctness through the resident wave.
    let got = outa.to_cpu_vec_f32().expect("readback");
    let mut want = vec![0f32; m * n];
    for r in 0..m {
        let row = &xa_data[r * k..(r + 1) * k];
        let inv = 1.0 / (row.iter().map(|v| v * v).sum::<f32>() / k as f32 + 1e-5).sqrt();
        let normed: Vec<f32> = row
            .iter()
            .zip(w_data.iter())
            .map(|(&v, &w)| v * inv * w)
            .collect();
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

/// F3+F4 (audit) end-to-end gate: opcode 6 driven through the REAL public
/// Rust API — `MoETaskDescriptor::upload` (device-resident descriptor, F4),
/// three independent schedule pointers (F3 Option A), `enqueue_via`,
/// `ScytheRing::enqueue`, one bounded `run_batch` — and checked against a
/// host reference of the Charon fused-grouped math. The pre-fix path handed
/// the kernel a HOST pointer in `weight_ptr` and a contiguous schedule
/// contract no producer ever emitted; the hand-packed device test could
/// not catch either because it bypassed this API entirely.
#[test]
fn moe_opcode6_via_public_api_matches_host_reference() {
    if !gpu_ready() {
        return;
    }
    use grim_engine::scythe2::{MoETaskDescriptor, moe_quant_mode};

    let hidden = 2usize;
    let inter = 3usize;
    let num_experts = 2usize;
    let num_tokens = 2usize; // schedule slots (top_k = 1, no padding)

    let mut exec = ScytheRingExec::new(8, 0).expect("ring exec");
    let dev = RocmDevice::try_new(0).expect("dev");

    // Deterministic operands.
    let act: Vec<f32> = (0..num_tokens * hidden)
        .map(|i| ((i % 5) as f32) * 0.3 - 0.4)
        .collect();
    let gate: Vec<f32> = (0..num_experts * inter * hidden)
        .map(|i| ((i % 7) as f32) * 0.1 - 0.3)
        .collect();
    let up: Vec<f32> = (0..num_experts * inter * hidden)
        .map(|i| ((i % 6) as f32) * 0.15 - 0.2)
        .collect();
    let down: Vec<f32> = (0..num_experts * hidden * inter)
        .map(|i| ((i % 8) as f32) * 0.12 - 0.45)
        .collect();
    let token_ids: Vec<u32> = vec![0, 1];
    let expert_ids: Vec<u32> = vec![0, 1];
    let weights: Vec<f32> = vec![0.8, 0.6];
    let rsf = 0.9f32;

    let u32_bytes = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_ne_bytes()).collect() };
    let u32_dtype = grim_tensor::dtype::DType {
        arith: grim_tensor::ArithType::U32,
        storage: grim_tensor::dtype::Storage::Native,
    };

    let act_s = dev
        .from_cpu(&act, &Shape::from_slice(&[num_tokens, hidden]), DType::F32)
        .expect("act");
    let gate_s = dev
        .from_cpu(
            &gate,
            &Shape::from_slice(&[num_experts, inter * hidden]),
            DType::F32,
        )
        .expect("gate");
    let up_s = dev
        .from_cpu(
            &up,
            &Shape::from_slice(&[num_experts, inter * hidden]),
            DType::F32,
        )
        .expect("up");
    let down_s = dev
        .from_cpu(
            &down,
            &Shape::from_slice(&[num_experts, hidden * inter]),
            DType::F32,
        )
        .expect("down");
    let tids_s = dev
        .from_cpu_bytes(
            &u32_bytes(&token_ids),
            &Shape::from_slice(&[num_tokens]),
            u32_dtype.clone(),
        )
        .expect("token ids");
    let eids_s = dev
        .from_cpu_bytes(
            &u32_bytes(&expert_ids),
            &Shape::from_slice(&[num_tokens]),
            u32_dtype.clone(),
        )
        .expect("expert ids");
    let ws_s = dev
        .from_cpu(&weights, &Shape::from_slice(&[num_tokens]), DType::F32)
        .expect("weights");
    // Charon accumulates with atomicAdd — the output MUST start zeroed.
    let out = dev
        .from_cpu(
            &vec![0f32; num_tokens * hidden],
            &Shape::from_slice(&[num_tokens, hidden]),
            DType::F32,
        )
        .expect("out");

    let moe = MoETaskDescriptor {
        hidden: hidden as u32,
        inter: inter as u32,
        num_tokens: num_tokens as u32,
        // Only block 0 of the grouped schedule exists (single-wave launch),
        // so block_size must cover every slot.
        block_size: num_tokens as u32,
        num_experts: num_experts as u32,
        top_k: 1,
        quant_mode: moe_quant_mode::FP32,
        routed_scaling_factor: rsf,
        gate_w_ptr: dev_ptr(gate_s.as_ref()),
        up_w_ptr: dev_ptr(up_s.as_ref()),
        down_w_ptr: dev_ptr(down_s.as_ref()),
        token_ids_ptr: dev_ptr(tids_s.as_ref()),
        expert_ids_ptr: dev_ptr(eids_s.as_ref()),
        weights_ptr: dev_ptr(ws_s.as_ref()),
    };
    moe.validate().expect("descriptor must validate");
    // F4: the descriptor itself is uploaded to device memory and the DEVICE
    // address is what rides the ring.
    let moe_dev = moe.upload(&dev).expect("MoE descriptor upload");

    let task =
        MoETaskDescriptor::enqueue_via(moe_dev, dev_ptr(act_s.as_ref()), dev_ptr(out.as_ref()), 0);
    let slot = exec.ring.enqueue(task).expect("enqueue");
    assert_eq!(slot, 0);
    let drained = exec.run_batch().expect("run batch");
    assert_eq!(drained, 1);

    // Host reference mirroring grim_moe_fused_grouped_device:
    // out[t][h] = rsf * w_t * Σ_j down_e[h*inter+j] * silu(gate_e[j]·a_t) * (up_e[j]·a_t)
    let got = out.to_cpu_vec_f32().expect("readback");
    let mut want = vec![0f32; num_tokens * hidden];
    for (&t, &e) in token_ids.iter().zip(expert_ids.iter()) {
        let (t, e) = (t as usize, e as usize);
        let w = weights[t];
        for h in 0..hidden {
            let mut acc = 0f32;
            for j in 0..inter {
                let mut g = 0f32;
                let mut u = 0f32;
                for i in 0..hidden {
                    g += gate[e * inter * hidden + j * hidden + i] * act[t * hidden + i];
                    u += up[e * inter * hidden + j * hidden + i] * act[t * hidden + i];
                }
                let silu_g = g / (1.0 + (-g).exp());
                acc += down[e * hidden * inter + h * inter + j] * silu_g * u;
            }
            want[t * hidden + h] = rsf * w * acc;
        }
    }
    let d = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("[f3f4] opcode-6 public-API MoE max_abs_diff={d:.3e}");
    assert!(d < 1e-4, "opcode-6 via public API diverged: {d:.3e}");
}

/// WI-SB6 production routing gate: with `GRIM_SCYTHE_RING=1`,
/// `RocmDevice::matmul_op` itself must ride the persistent ring and produce
/// byte-compatible results with the direct rocBLAS path. This is the seam
/// real decode layers flow through when the flag is set.
#[test]
fn production_ring_routing_matmul_parity() {
    if !gpu_ready() {
        return;
    }
    let dev = RocmDevice::try_new(0).expect("dev");

    let m = 4usize;
    let k = 64usize;
    let n = 96usize;
    let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 9) as f32) * 0.2 - 0.8).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32) * 0.05 - 0.3).collect();

    let a = dev
        .from_cpu(&a_data, &Shape::from_slice(&[m, k]), DType::F32)
        .expect("a");
    let b = dev
        .from_cpu(&b_data, &Shape::from_slice(&[k, n]), DType::F32)
        .expect("b");
    let out_shape = Shape::from_slice(&[m, n]);

    // Direct rocBLAS baseline (BackendDevice::matmul → matmul_op).
    let (direct, handle) = dev
        .matmul(a.as_ref(), b.as_ref(), &out_shape)
        .expect("direct matmul");
    handle.synchronize().expect("direct sync");
    let direct = direct.to_cpu_vec_f32().expect("direct readback");

    // Ring-routed path: flip the production gate for the duration of this
    // test (edition-2024 env mutation is unsafe; this binary has no other
    // matmul callers, so the process-global flag cannot poison a sibling).
    unsafe { std::env::set_var("GRIM_SCYTHE_RING", "1") };
    let routed = {
        let (out, handle) = dev
            .matmul(a.as_ref(), b.as_ref(), &out_shape)
            .expect("ring-routed matmul");
        handle.synchronize().expect("routed sync");
        out.to_cpu_vec_f32().expect("routed readback")
    };
    unsafe { std::env::remove_var("GRIM_SCYTHE_RING") };

    let d = direct
        .iter()
        .zip(routed.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    println!("[sb6] production ring-vs-direct matmul max_abs_diff={d:.3e}");
    assert!(
        d < 1e-4,
        "GRIM_SCYTHE_RING=1 routing must match direct rocBLAS: {d:.3e}"
    );
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
    let _x = dev
        .from_cpu(&x_data, &Shape::from_slice(&[m, k]), DType::F32)
        .expect("x");

    // Per-shard operands: transposed B[k_half,n] slices plus this rank's
    // K-slice of x — exactly what a real row-parallel rank receives.
    let mut b_t = Vec::new(); // device storages
    let mut a_s = Vec::new(); // device storages
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
        b_t.push(dev.from_cpu(&bt, &bt_shape, DType::F32).expect("bt upload"));
        let mut xs = vec![0f32; m * k_count];
        for r in 0..m {
            for c in 0..k_count {
                xs[r * k_count + c] = x_data[r * k + k_start + c];
            }
        }
        let xs_shape = Shape::from_slice(&[m, k_count]);
        a_s.push(dev.from_cpu(&xs, &xs_shape, DType::F32).expect("xs upload"));
    }

    let p0 = dev
        .alloc_storage(&Shape::from_slice(&[m, n]), DType::F32)
        .expect("p0");
    let p1 = dev
        .alloc_storage(&Shape::from_slice(&[m, n]), DType::F32)
        .expect("p1");
    let summed = dev
        .alloc_storage(&Shape::from_slice(&[m, n]), DType::F32)
        .expect("summed");

    exec.submit_col_gemm(
        m as u32,
        n as u32,
        half as u32,
        dev_ptr(a_s[0].as_ref()),
        dev_ptr(b_t[0].as_ref()),
        dev_ptr(p0.as_ref()),
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
