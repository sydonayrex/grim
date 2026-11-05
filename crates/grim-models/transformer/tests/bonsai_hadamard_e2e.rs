use grim_format::prism_hadamard::parse_prism_hadamard;
use grim_models_transformer::decode_graph::HadamardPlan;
use grim_tensor::provider::TensorProvider;
use std::collections::HashSet;
use std::path::Path;

const MODEL: &str = "/drive/bigfast/grim/models/Bonsai-27B/Ternary-Bonsai-2-27B-PQ2_0.gguf";

#[test]
fn gate_a_hadamard_plan_accepts_bonsai() {
    if !Path::new(MODEL).exists() {
        eprintln!("skipping: no Bonsai file");
        return;
    }
    let provider = grim_format::tprov::GgufProvider::open(MODEL).unwrap();
    let tensor_names: HashSet<String> = provider.tensor_names().into_iter().collect();
    let has = |name: &str| tensor_names.contains(name);
    let meta = parse_prism_hadamard(provider.metadata_map(), &has).unwrap().expect("bonsai must carry hadamard");
    let plan = HadamardPlan::from_meta(&meta, &tensor_names, 48, 16).unwrap();
    assert_eq!(plan.folded.len(), 401);
    assert_eq!(plan.ssm_perm, Some((128, 16, 3)));
    assert!(plan.inverse.contains("token_embd.weight"));

    // Reject an unknown foldable kind.
    let mut bad = meta.clone();
    if !bad.weight_names.is_empty() {
        bad.weight_names[0] = "blk.0.unknown.weight".to_string();
    }
    assert!(HadamardPlan::from_meta(&bad, &tensor_names, 48, 16).is_err());
}



#[test]
#[ignore]
fn gate_d_bonsai_first_32_token_parity() -> TestResult {
    if !Path::new(MODEL).exists() {
        return Ok(());
    }
    let dev = match gpu_device() {
        Some(d) => d,
        None => return Ok(()),
    };
    let device = grim_tensor::Device::Rocm(dev.ordinal());
    let model = grim_engine::model_loader::load_model_from_gguf(MODEL, device.clone())
        .map_err(|e| format!("load_model_from_gguf failed: {e}"))?;
    let provider = grim_format::GgufProvider::open(MODEL).map_err(|e| format!("GgufProvider::open failed: {e}"))?;
    let _tokenizer = provider.tokenizer().map_err(|e| format!("tokenizer failed: {e}"));
    // The exact chat-template token ids the REFERENCE (llama-tokenize) encodes
    // for this prompt. grim-cli's own template encoding differs: it maps
    // "user"->1156 / "assistant"->17313 (plain-text pieces) where the
    // reference maps "user"->846 / "assistant"->74455 (template role pieces).
    // Same decoded text, different ids -> different embeddings -> diverged
    // prefill state. Parity requires byte-identical token ids.
    let input_ids: Vec<u32> = vec![
        248045, 846, 198, 814, 20139, 1204, 264, 94562, 4138, 303, 1330, 41228,
        25, 248046, 198, 248045, 74455, 198, 248068, 198,
    ];
    let prompt_len = input_ids.len();
    let ids: Vec<f32> = input_ids.iter().map(|&x| x as f32).collect();
    let dev_box: Box<dyn grim_tensor::backend::BackendDevice> = Box::new(dev);
    let mk = |vals: &[f32]| -> grim_tensor::Tensor {
        let shape = Shape::new(vec![1, vals.len()]);
        let st = dev_box.from_cpu(vals, &shape, DType::F32).expect("from_cpu");
        grim_tensor::Tensor::new(
            std::sync::Arc::from(st),
            shape,
            DType::F32,
            grim_tensor::QuantProvenance::default(),
            device.clone(),
        )
    };
    let pos: Vec<f32> = (0..prompt_len).map(|i| i as f32).collect();
    let input = mk(&ids);
    let pos_t = mk(&pos);
    let mut session = model.new_session();
    let logits = model.forward(session.as_mut(), &input, &pos_t, &[]).map_err(|e| format!("prefill failed: {e}"))?;
    let vocab = 248320usize;
    let flat = logits.to_vec_f32().map_err(|e| format!("logits f32: {e}"))?;
    let row = &flat[flat.len() - vocab..];
    let mut generated = Vec::new();
    // (argmax, top1, top2, logit-of-REF-token) per step — the ref-token logit
    // quantifies whether a divergence is a near-tie flip or a real reorder.
    let mut gaps: Vec<(u32, f32, f32, f32)> = Vec::new();
    let ref_tokens: Vec<u32> = std::fs::read_to_string("/tmp/bonsai_ref_tokens.txt")
        .unwrap_or_default()
        .split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect();
    let snap = |v: &[f32], step_i: u32| -> (u32, f32, f32, f32) {
        let mut idx: Vec<u32> = (0..v.len() as u32).collect();
        idx.sort_by(|&a, &b| v[b as usize].total_cmp(&v[a as usize]));
        let ref_logit = ref_tokens.get(step_i as usize)
            .map(|&rt| v[rt as usize])
            .unwrap_or(f32::NAN);
        (idx[0], v[idx[0] as usize], v[idx[1] as usize], ref_logit)
    };
    let mut g = snap(row, 0);
    generated.push(g.0);
    gaps.push(g);
    let mut current = g.0;
    for i in 1..32 {
        let step_pos = prompt_len + i - 1;
        let st_in = mk(&[current as f32]);
        let st_pos = mk(&[step_pos as f32]);
        let step_logits = model.forward(session.as_mut(), &st_in, &st_pos, &[]).map_err(|e| format!("decode failed: {e}"))?;
        let flat = step_logits.to_vec_f32().map_err(|e| format!("step logits: {e}"))?;
        let row = &flat[flat.len() - vocab..];
        g = snap(row, i as u32);
        current = g.0;
        generated.push(g.0);
        gaps.push(g);
    }
    let ref_path = std::path::Path::new("/tmp/bonsai_ref_tokens.txt");
    let txt = std::fs::read_to_string(ref_path).map_err(|e| format!("read ref: {e}"))?;
    let want: Vec<u32> = txt.split_whitespace().map(|t| t.parse().expect("token id")).collect();
    if generated[..32] != want[..32] {
        // Plan rule (Task 6 Step 2): on divergence print both logit gaps and
        // STOP — do not weaken to "similar text". Report the greedy top-2 gap
        // at the FIRST divergent position for both sides.
        let div = (0..32).find(|&i| generated[i] != want[i]).unwrap();
        let (_, l1, l2, lref) = gaps[div];
        eprintln!(
            "[gate-d] first divergence at token {div}: grim={} ref={}\n  grim[0..8]={:?}\n  ref [0..8]={:?}\n  grim logits at divergence: top1={l1:.4} top2={l2:.4} gap={:.4} | ref-token logit in grim: {lref:.4} (rank delta {:.4}){}",
            generated[div], want[div], &generated[..8], &want[..8],
            l1 - l2,
            l1 - lref,
            if (l1 - lref).abs() < 0.1 { "  <- ref token within float noise of grim top1" } else { "" }
        );
        eprintln!(
            "[gate-d] STOP per plan: parity NOT met (matched {div}/32 tokens)"
        );
        return Err(format!("Gate D parity: diverged at token {div} (grim {} vs ref {})", generated[div], want[div]).into());
    }
    println!("[gate-d] first 32 tokens identical to reference");
    let _ = dev_box;
    Ok(())
}

use grim_backend_rocm::device::roc_device::RocmDevice;
use grim_backend_rocm::{CoreTensorOps, MemoryOps, Shape};
use grim_tensor::DType;
use grim_quant::hadamard::fwht_signed_f32;

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    grim_backend_rocm::require_gpu_device()
}

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127;
    let man = bits & 0x7f_ffff;
    if exp < -14 { return sign; }
    if exp > 15 { return sign | 0x7c00; }
    let e = (exp + 15) as u16;
    let m = ((man >> 13) & 0x3ff) as u16;
    sign | (e << 10) | m
}

fn pack_pq2_0_block(delta: f32, codes: &[u8; 128]) -> Vec<u8> {
    let mut blk = f32_to_f16_bits(delta).to_le_bytes().to_vec();
    for b in 0..32 {
        let mut byte = 0u8;
        for i in 0..4 {
            byte |= (codes[b * 4 + i] & 0x3) << (i * 2);
        }
        blk.push(byte);
    }
    blk
}

fn pack_matrix_pq2_0(w: &[f32], n: usize, k: usize) -> Vec<u8> {
    let mut packed = Vec::with_capacity(n * (k / 128) * 34);
    for ni in 0..n {
        let row = &w[ni * k..(ni + 1) * k];
        let amax = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let delta = (amax / 2.0).max(1e-8);
        for blk in 0..k / 128 {
            let mut codes = [0u8; 128];
            for i in 0..128 {
                let v = row[blk * 128 + i];
                let c = (v / delta).round() + 1.0;
                codes[i] = c.clamp(0.0, 3.0) as u8;
            }
            packed.extend(pack_pq2_0_block(delta, &codes));
        }
    }
    packed
}

fn quantize_q8_0(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() / 32 * 34);
    for chunk in vals.chunks(32) {
        let amax = chunk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let d = (amax / 127.0).max(1e-8);
        out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        for v in chunk {
            let q = (v / d).round().clamp(-127.0, 127.0) as i8;
            out.push(q as u8);
        }
    }
    out
}

fn dequant_q8_0(bytes: &[u8], k: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(k);
    for blk in 0..k / 32 {
        let base = blk * 34;
        let d = f32_to_f16_roundtrip(u16::from_le_bytes([bytes[base], bytes[base + 1]]));
        for i in 0..32 {
            out.push(bytes[base + 2 + i] as i8 as f32 * d);
        }
    }
    out
}

fn f32_to_f16_roundtrip(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let man = (h & 0x3ff) as f32 / 1024.0;
    if exp == 0 {
        sign * man * (2f32).powi(-14)
    } else {
        sign * (1.0 + man) * (2f32).powi(exp - 15)
    }
}

fn gemv_pq2_0(dev: &RocmDevice, packed: &[u8], a_q8: &[u8], m: usize, n: usize, k: usize) -> Vec<f32> {
    let b_dev = MemoryOps::from_cpu_bytes(dev, packed, &Shape::new(vec![packed.len()]), DType::U8).unwrap();
    let a_dev = MemoryOps::from_cpu_bytes(dev, a_q8, &Shape::new(vec![a_q8.len()]), DType::U8).unwrap();
    let out_dev = CoreTensorOps::zeros(dev, &Shape::new(vec![m, n]), DType::F32).unwrap();
    let b_s = grim_backend_rocm::device::util::as_rocm(b_dev.as_ref()).unwrap();
    let a_s = grim_backend_rocm::device::util::as_rocm(a_dev.as_ref()).unwrap();
    let out_s = grim_backend_rocm::device::util::as_rocm(out_dev.as_ref()).unwrap();
    dev.launch_dot4_pq2_0_q80_gemv(a_s, b_s, out_s, m, n, k).unwrap();
    dev.synchronize();
    out_dev.to_cpu_vec_f32().unwrap()
}

#[test]
#[ignore]
fn gate_b_folded_linear_transform_matches_oracle() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()); };
    let n = 8usize;
    let k = 1024usize;
    let m = 4usize;

    let mut state = 0x243F6A8885A308D3u64;
    let mut rand = move || {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        (state >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0
    };

    let signs: Vec<i32> = (0..k).map(|_| if rand() < 0.5 { -1 } else { 1 }).collect();
    let mut w = vec![0.0f32; n * k];
    for v in &mut w { *v = rand(); }
    // W_f = W · D · H (host, Task-1 math)
    let mut w_f = vec![0.0f32; n * k];
    for ni in 0..n {
        let mut row: Vec<f32> = w[ni * k..(ni + 1) * k].to_vec();
        fwht_signed_f32(&mut row, &signs);
        w_f[ni * k..(ni + 1) * k].copy_from_slice(&row);
    }

    let mut x = vec![0.0f32; m * k];
    for v in &mut x { *v = rand(); }
    // transformed x
    let mut x_t = x.clone();
    for r in 0..m {
        fwht_signed_f32(&mut x_t[r * k..(r + 1) * k], &signs);
    }

    let packed_f = pack_matrix_pq2_0(&w_f, n, k);
    let a_q8_t = quantize_q8_0(&x_t);
    let out_gpu = gemv_pq2_0(&dev, &packed_f, &a_q8_t, m, n, k);

    // oracle: dequant(W_f) dot dequant(q8 transformed x)
    let w_f_dq = grim_quant::dequant_pq2_0(&packed_f, n * k).unwrap();
    let x_t_dq = dequant_q8_0(&a_q8_t, m * k);
    let mut oracle = vec![0.0f32; m * n];
    for r in 0..m {
        for ni in 0..n {
            let mut acc = 0.0f32;
            for ki in 0..k {
                acc += w_f_dq[ni * k + ki] * x_t_dq[r * k + ki];
            }
            oracle[r * n + ni] = acc;
        }
    }
    for (i, (&g, &o)) in out_gpu.iter().zip(oracle.iter()).enumerate() {
        let rel = (g - o).abs() / o.abs().max(1e-6);
        assert!(rel < 2e-2, "gate B mismatch {i}: gpu={g} oracle={o} rel={rel}");
    }
    println!("gate B folded transform+gemv ok");
    Ok(())
}

#[test]
#[ignore]
fn gate_c_primal_identity() -> TestResult {
    let _dev = match gpu_device() { Some(d) => d, None => return Ok(()) };
    let n = 8usize;
    let k = 1024usize;
    let m = 64usize;

    let mut state = 0x243F6A8885A308D3u64;
    let mut rand = move || {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        (state >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0
    };

    let signs: Vec<i32> = (0..k).map(|_| if rand() < 0.5 { -1 } else { 1 }).collect();
    let mut w = vec![0.0f32; n * k];
    for v in &mut w { *v = rand(); }
    // W_f = W · D · H
    let mut w_f = vec![0.0f32; n * k];
    for ni in 0..n {
        let mut row: Vec<f32> = w[ni * k..(ni + 1) * k].to_vec();
        fwht_signed_f32(&mut row, &signs);
        w_f[ni * k..(ni + 1) * k].copy_from_slice(&row);
    }

    let mut x = vec![0.0f32; m * k];
    for v in &mut x { *v = rand(); }
    let mut x_t = x.clone();
    for r in 0..m {
        fwht_signed_f32(&mut x_t[r * k..(r + 1) * k], &signs);
    }

    // (a) f32 identity pack envelope
    for r in 0..m {
        for ni in 0..n {
            let mut a = 0.0f32;
            let mut b = 0.0f32;
            for ki in 0..k {
                a += w[ni * k + ki] * x[r * k + ki];
                b += w_f[ni * k + ki] * x_t[r * k + ki];
            }
            assert!((a - b).abs() < 5e-3, "gate C f32 mismatch r={r} ni={ni}: {a} vs {b}");
        }
    }
    println!("gate C primal-vs-folded identity ok (f32 envelope)");
    Ok(())
}

#[test]
#[ignore]
fn gate_tiled_fwht_matches_blockwise_oracle() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()); };
    for &w in &[5120usize, 6144usize] {
        let n_blk = w / 1024;
        let rows = 4usize;
        let mut state = 0x243F6A8885A308D3u64;
        let mut rand = move || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; (state >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0 };
        let signs: Vec<f32> = (0..w).map(|_| if rand() < 0.0 { -1.0 } else { 1.0 }).collect();
        let mut x = vec![0.0f32; rows * w];
        for v in &mut x { *v = rand(); }
        // host oracle: block-diagonal signs+FWHT
        let mut oracle = x.clone();
        for r in 0..rows {
            for b in 0..n_blk {
                let off = r * w + b * 1024;
                let mut slice = oracle[off..off+1024].to_vec();
                let s = &signs[b*1024..(b+1)*1024];
                let si: Vec<i32> = s.iter().map(|&v| if v < 0.0 { -1 } else { 1 }).collect();
                grim_quant::hadamard::fwht_signed_f32(&mut slice, &si);
                oracle[off..off+1024].copy_from_slice(&slice);
            }
        }
        let src = MemoryOps::from_cpu_bytes(&dev, &bytemuck_f32(&x), &Shape::new(vec![rows*w]), DType::F32).unwrap();
        let signs_dev = MemoryOps::from_cpu_bytes(&dev, &bytemuck_f32(&signs), &Shape::new(vec![w]), DType::F32).unwrap();
        let dst = CoreTensorOps::zeros(&dev, &Shape::new(vec![rows*w]), DType::F32).unwrap();
        let src_s = grim_backend_rocm::device::util::as_rocm(src.as_ref()).unwrap();
        let dst_s = grim_backend_rocm::device::util::as_rocm(dst.as_ref()).unwrap();
        let signs_s = grim_backend_rocm::device::util::as_rocm(signs_dev.as_ref()).unwrap();
        grim_backend_rocm::device::hadamard_launch::launch_fwht_signed_tiled(&dev, src_s, dst_s, Some(signs_s), 1024, n_blk as u32, rows).unwrap();
        dev.synchronize();
        let got = dst.to_cpu_vec_f32().unwrap();
        let mut maxerr = 0.0f32;
        for (i,(a,b)) in got.iter().zip(oracle.iter()).enumerate() {
            maxerr = maxerr.max((a-b).abs());
            if (a-b).abs() > 1e-4 { panic!("mismatch {} got {} oracle {}", i, a, b); }
        }
        println!("tiled fwht width={} maxerr={}", w, maxerr);
    }
    Ok(())
}

fn bytemuck_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

#[test]
#[ignore]
fn gate_eager_helper_matches_host() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()); };
    // need plan active
    let model = MODEL;
    let provider = grim_format::tprov::GgufProvider::open(model).unwrap();
    let tensor_names: HashSet<String> = provider.tensor_names().into_iter().collect();
    let meta = grim_format::prism_hadamard::parse_prism_hadamard(provider.metadata_map(), &|name: &str| tensor_names.contains(name)).unwrap().unwrap();
    let plan = HadamardPlan::from_meta(&meta, &tensor_names, 48, 16).unwrap();
    grim_models_transformer::decode_graph::set_current_hadamard_plan(Some(&plan));
    // transform a hidden-width vector
    let w = 5120usize;
    let mut state = 1u64;
    let mut rand = move || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; (state >> 11) as f32 / (1u64 << 53) as f32 };
    let mut x = vec![0.0f32; w];
    for v in &mut x { *v = rand(); }
    let _x_cpu = grim_backend_cpu::cpu_tensor(x.clone(), Shape::new(vec![1,w]));
    let x_gpu_st = MemoryOps::from_cpu_bytes(&dev, &x.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>(), &Shape::new(vec![1,w]), DType::F32).unwrap();
    let x_gpu = grim_tensor::Tensor::new(
        std::sync::Arc::from(x_gpu_st),
        Shape::new(vec![1, w]),
        DType::F32,
        grim_tensor::QuantProvenance::GrimNative,
        grim_tensor::Device::Rocm(dev.ordinal()),
    );
    let got_gpu = grim_models_transformer::decode_graph::hadamard_transform_tensor(&x_gpu, 0, "attn_q").unwrap();
    let got_gpu_v = got_gpu.to_vec_f32().unwrap();
    // host oracle: blockwise signs then FWHT
    let signs = plan.signs.get(&w).unwrap();
    let mut oracle = x.clone();
    for blk in 0..w/1024 {
        let off = blk*1024;
        let mut sl = oracle[off..off+1024].to_vec();
        let si: Vec<i32> = signs[off..off+1024].to_vec();
        grim_quant::hadamard::fwht_signed_f32(&mut sl, &si);
        oracle[off..off+1024].copy_from_slice(&sl);
    }
    let mut maxerr = 0.0f32;
    for (a,b) in got_gpu_v.iter().zip(oracle.iter()) { maxerr = maxerr.max((a-b).abs()); }
    assert!(maxerr < 1e-4, "maxerr {}", maxerr);
    println!("helper maxerr {}", maxerr);
    Ok(())
}
