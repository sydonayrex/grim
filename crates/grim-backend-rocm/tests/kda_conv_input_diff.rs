//! THE ARBITRATION (qwen35 D2D decode divergence): run BOTH conv kernels on
//! the EXACT bytes the production runs fed them (dumps from
//! GRIM_KDA_STEP_PROBE at layer 0, decode step 1), and diff each against the
//! production outputs. Whichever kernel disagrees with its own production
//! output is the wrong one; the weight indexing it reads with is wrong for
//! this layout.
//!
//! Dumps (f32 LE): conv_w  [8192, 4] (channels x taps, d2d == host bitwise),
//! conv_x [8192], pre-step ring [8192 x 3] (both runs' POST-step rings are
//! byte-identical, so PRE-step rings are identical too; the pre-step ring is
//! derived by inverting the shift), conv_mix (silu applied by the caller).
//!
//! RUN: GRIM_RUN_GPU_TEST=1 HIP_VISIBLE_DEVICES=1 cargo test -p grim-backend-rocm \
//!      --test kda_conv_input_diff -- --ignored --nocapture

use grim_backend_rocm::RecurrentOps;
use grim_backend_rocm::RocmDevice;
use grim_tensor::{CoreTensorOps, DType, Shape, Storage};

fn load_f32(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).expect(path);
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

#[test]
#[ignore = "device-gated: run with GRIM_RUN_GPU_TEST=1; needs the probe dumps in /tmp"]
fn conv_kernels_arbitrated_on_production_bytes() {
    let ch = 8192usize;
    let ks = 4usize;
    let w = load_f32("/tmp/kda_probe_d2d_conv_w.bin");
    let x = load_f32("/tmp/kda_probe_d2d_conv_x.bin");
    let mix_d2d = load_f32("/tmp/kda_probe_d2d_conv_mix.bin");
    let mix_host = load_f32("/tmp/kda_probe_host_conv_mix.bin");
    let ring_post_d2d = load_f32("/tmp/kda_probe_d2d_conv_ring.bin");
    // Invert the shift to get the PRE-step ring: post[c*3 + 0..2] == pre[c*3 + 1..3];
    // post[c*3 + 2] == x[c] (the sample just inserted). So pre = [?, post0, post1].
    // The oldest slot is unknown from post alone — recover it from the HOST
    // post ring's relation... both post rings identical, so reconstruct the
    // pre-ring from the HOST cpu-side knowledge instead: run the host formula
    // backward. Simpler: pre-ring slot 1,2 = post 0,1; slot 0 = unknown u.
    // The conv output sum = x*w3 + pre[0]*w0 + pre[1]*w1 + pre[2]*w2. Solve
    // for the output using only the known slots plus u as a parameter — the
    // arbitration works by MATCHING each kernel against its own production
    // output, and u is the same for both, so pass u=0 to both and let the
    // matching decide.
    let mut ring_pre = vec![0.0f32; ch * (ks - 1)];
    for c in 0..ch {
        ring_pre[c * (ks - 1) + 0] = 0.0; // unknown slot, same for both arms
        ring_pre[c * (ks - 1) + 1] = ring_post_d2d[c * (ks - 1) + 0];
        ring_pre[c * (ks - 1) + 2] = ring_post_d2d[c * (ks - 1) + 1];
    }

    let dev = RocmDevice::try_new(0).expect("RocmDevice::try_new(0)");

    // ── CPU oracle on the same bytes (weight indexing [c*ks + k]) ──
    let mut cpu = vec![0.0f32; ch];
    for c in 0..ch {
        let mut sum = x[c] * w[c * ks + ks - 1];
        for k in 0..ks - 1 {
            sum += ring_pre[c * (ks - 1) + k] * w[c * ks + k];
        }
        cpu[c] = silu(sum);
    }

    // ── arm 1: the D2D step kernel (grim_short_conv1d_causal_step) ──
    let w_s = CoreTensorOps::from_cpu(&dev, &w, &Shape::new(vec![ch, ks]), DType::F32).unwrap();
    let x_s = CoreTensorOps::from_cpu(&dev, &x, &Shape::new(vec![ch]), DType::F32).unwrap();
    let ring_s = CoreTensorOps::from_cpu(
        &dev,
        &ring_pre,
        &Shape::new(vec![ch * (ks - 1)]),
        DType::F32,
    )
    .unwrap();
    let (out1, _h) = dev
        .short_conv1d_causal_step(
            x_s.as_ref(),
            w_s.as_ref(),
            None,
            ring_s.as_ref(),
            &Shape::new(vec![ch]),
        )
        .expect("step kernel");
    let arm1 = out1.to_cpu_vec_f32().unwrap();

    // ── arm 2: modules::short_conv1d CPU fallback (the host path's math,
    //    which reads w[di*k_size + ki] — same [c*ks+k] indexing, but it walks
    //    the SEQUENCE and the state differently: prev_idx = si - (k-1-ki),
    //    i.e. w index ki=0 pairs with the OLDEST input) ──
    // Host-path replication, exact: for s=1, prev_idx = -(k-1-ki); taps with
    // prev_idx<0 read the state at (state_k + prev_idx), i.e. the ring in
    // REVERSE order relative to arm 1.
    let mut arm2 = vec![0.0f32; ch];
    for c in 0..ch {
        let state_k = ks - 1;
        let mut sum = 0.0f32;
        for ki in 0..ks {
            let prev_idx = 0isize - (ks - 1 - ki) as isize;
            let val = if prev_idx >= 0 {
                x[c]
            } else {
                let state_idx = (state_k as isize + prev_idx) as usize;
                ring_pre[c * state_k + state_idx]
            };
            sum += val * w[c * ks + ki];
        }
        arm2[c] = silu(sum);
    }

    let d = |a: &[f32], b: &[f32]| -> f32 {
        a.iter()
            .zip(b)
            .map(|(g, w)| (g - w).abs())
            .fold(0.0f32, f32::max)
    };
    eprintln!("[arb] arm1 (step kernel) vs d2d production: {:.6}", d(&arm1, &mix_d2d));
    eprintln!("[arb] arm1 (step kernel) vs host production: {:.6}", d(&arm1, &mix_host));
    eprintln!("[arb] arm2 (host math)   vs d2d production: {:.6}", d(&arm2, &mix_d2d));
    eprintln!("[arb] arm2 (host math)   vs host production: {:.6}", d(&arm2, &mix_host));
    eprintln!("[arb] cpu oracle         vs arm1: {:.6}   vs arm2: {:.6}", d(&cpu, &arm1), d(&cpu, &arm2));
    eprintln!("[arb] samples c=0..3: cpu={:?} arm1={:?} arm2={:?}", &cpu[..4], &arm1[..4], &arm2[..4]);
    eprintln!("[arb] production: d2d_mix[0..4]={:?} host_mix[0..4]={:?}", &mix_d2d[..4], &mix_host[..4]);
}

/// Extract the step kernel's ACTUAL coefficient vector per input position:
/// set x=0, ring=[0,0,0], then fire each unit input separately and read the
/// output. Whatever weights it multiplies by will fall out directly.
#[test]
#[ignore = "device-gated: run with GRIM_RUN_GPU_TEST=1"]
fn step_kernel_coefficient_extraction() {
    let ch = 256usize;
    let ks = 4usize;
    let dev = RocmDevice::try_new(0).expect("RocmDevice::try_new(0)");
    let mut seed: u64 = 7;
    let mut rand = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (((seed >> 33) & 0xffffffff) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let w: Vec<f32> = (0..ch * ks).map(|_| rand()).collect();
    let w_s = CoreTensorOps::from_cpu(&dev, &w, &Shape::new(vec![ch, ks]), DType::F32).unwrap();

    let run = |x: &[f32], ring: &[f32]| -> Vec<f32> {
        let x_s = CoreTensorOps::from_cpu(&dev, x, &Shape::new(vec![ch]), DType::F32).unwrap();
        let r_s = CoreTensorOps::from_cpu(&dev, ring, &Shape::new(vec![ch * (ks - 1)]), DType::F32)
            .unwrap();
        let (out, _h) = dev
            .short_conv1d_causal_step(
                x_s.as_ref(),
                w_s.as_ref(),
                None,
                r_s.as_ref(),
                &Shape::new(vec![ch]),
            )
            .expect("step");
        out.to_cpu_vec_f32().unwrap()
    };
    let zeros = vec![0.0f32; ch];
    let zero_ring = vec![0.0f32; ch * (ks - 1)];

    // x = 1, ring = 0 -> expect w3 per channel (times silu; read pre-silu via
    // known small values: use x = 0.001 to stay in silu's linear regime).
    let x1: Vec<f32> = vec![0.001; ch];
    let out_x = run(&x1, &zero_ring);
    // silu linear regime: out ≈ 0.001 * sigmoid(0) * ... ≈ v/2 for tiny v? silu(v)=v/2 as v->0.
    // So pre-silu ≈ out*2.
    let pre_x: Vec<f32> = out_x.iter().map(|&v| v * 2.0).collect();
    eprintln!(
        "[coeff] x-tap: expect w3[c], got pre/0.001: c0={:.5} w3={:.5} | c1={:.5} w3={:.5}",
        pre_x[0] / 0.001,
        w[0 * ks + 3],
        pre_x[1] / 0.001,
        w[1 * ks + 3]
    );

    // ring slot 0 = 1 (x = 0) -> expect w0.
    let mut r0 = zero_ring.clone();
    for c in 0..ch {
        r0[c * (ks - 1)] = 0.001;
    }
    let out_r0 = run(&zeros, &r0);
    let pre_r0: Vec<f32> = out_r0.iter().map(|&v| v * 2.0).collect();
    eprintln!(
        "[coeff] ring0-tap: expect w0[c], got pre/0.001: c0={:.5} w0={:.5} | c1={:.5} w0={:.5}",
        pre_r0[0] / 0.001,
        w[0 * ks + 0],
        pre_r0[1] / 0.001,
        w[1 * ks + 0]
    );

    // ring slot 2 = 1 -> expect w2.
    let mut r2 = zero_ring.clone();
    for c in 0..ch {
        r2[c * (ks - 1) + 2] = 0.001;
    }
    let out_r2 = run(&zeros, &r2);
    let pre_r2: Vec<f32> = out_r2.iter().map(|&v| v * 2.0).collect();
    eprintln!(
        "[coeff] ring2-tap: expect w2[c], got pre/0.001: c0={:.5} w2={:.5} | c1={:.5} w2={:.5}",
        pre_r2[0] / 0.001,
        w[0 * ks + 2],
        pre_r2[1] / 0.001,
        w[1 * ks + 2]
    );
}

/// Gather arbitration: run grim's own device packed gather on the EXACT
/// device table + the production token id 11751, and compare against the
/// file row. If the gather is wrong at seq_len 1, this reproduces the
/// production interleave signature ([A,B,A,B] stride-2) seen in the x dumps.
#[test]
#[ignore = "device-gated: run with GRIM_RUN_GPU_TEST=1"]
fn embedding_gather_arbitration_row_11751() {
    let dev = RocmDevice::try_new(0).expect("RocmDevice::try_new(0)");
    // Real device table: reuse the probe dump (2304 B row 11751) embedded in a
    // [248320, 4096]-shaped storage is impossible cheaply; instead build a
    // minimal [vocab=3, 4096] table of rows (11750, 11751, 11752) from the
    // file and gather index 1 with an offset map… but the production kernel
    // derives the row offset as index * row_bytes internally, so the minimal
    // table must place row 11751's bytes at offset 11751 * 2304. Allocate
    // 11752 * 2304 + 2304 bytes (sparse, but fine) and write the real row at
    // its production offset; zero elsewhere.
    let row_bytes = 2304usize;
    let file_row = std::fs::read("/tmp/kda_probe_file_row11751.bin").expect("file row dump");
    let vocab_rows = 11753usize;
    let mut table = vec![0u8; vocab_rows * row_bytes];
    table[11751 * row_bytes..11752 * row_bytes].copy_from_slice(&file_row);
    let table_s = grim_tensor::MemoryOps::from_cpu_bytes(
        &dev,
        &table,
        &Shape::new(vec![table.len()]),
        DType {
            arith: grim_tensor::ArithType::F32,
            storage: Storage::KQuant(grim_tensor::KQuantScheme::Q4K),
        },
    )
    .expect("upload table");
    let out_s = CoreTensorOps::from_cpu(
        &dev,
        &vec![0.0f32; 4096],
        &Shape::new(vec![4096]),
        DType::F32,
    )
    .unwrap();
    let _ = out_s;
    // Gather through the production entry: embedding_packed.
    let (out, _h) = grim_tensor::CoreTensorOps::embedding_packed(
        &dev,
        table_s.as_ref(),
        &[11751],
        &Shape::new(vec![1, 4096]),
        4096,
    )
    .expect("embedding_packed");
    // read back
    let r = grim_backend_rocm::as_rocm(out.as_ref()).expect("rocm out");
    let bytes = r.copy_to_host().expect("read back gather");
    let got: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let file_row_deq = grim_quant::dequant_q4k(&file_row, 4096).unwrap();
    let d = got
        .iter()
        .zip(&file_row_deq)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "[gather-arb] gathered row vs file row: max|diff| {d:.6}; got[0..4]={:?}",
        &got[..4]
    );
    eprintln!(
        "[gather-arb] production x (block input) was interleaved [A,B,A,B]; compare shape here"
    );
}
