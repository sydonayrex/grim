//! Native-arm D2D dispatch gate tests (WI-gpu-native-moe Phase 2).
//! Split out of moe_all_models_parity_gpu.rs: these tests leave deferred
//! async GPU work that faults unrelated tests when co-hosted in one binary.

use std::sync::Arc;

use grim_backend_rocm::RocmDevice;
use grim_nn::moe::ExpertBank;
fn o1_vec(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + 1.0) * 0.37 + seed).sin() + ((i as f32 + 1.0) * 0.11 + seed * 2.0).cos() * 0.5)
        .collect()
}

use grim_nn::Linear;
use grim_tensor::shape::Shape;
use grim_tensor::{CoreTensorOps, DType, Device, Tensor};

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    std::panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new")).ok()
}

fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    grim_backend_rocm::device::util::gpu_test_lock()
}

fn rocm_tensor(dev: &RocmDevice, data: Vec<f32>, shape: Shape) -> Tensor {
    let storage = dev
        .from_cpu(&data, &shape, grim_tensor::dtype::DType::F32)
        .unwrap();
    Tensor::new(
        std::sync::Arc::from(storage),
        shape,
        grim_tensor::dtype::DType::F32,
        grim_tensor::dtype::QuantProvenance::default(),
        Device::Rocm(0),
    )
}

use std::collections::HashMap;

type RawTensorEntry = (Vec<u8>, Vec<usize>, DType, grim_tensor::QuantProvenance);

#[derive(Clone)]
struct MemProvider {
    tensors: Arc<HashMap<String, RawTensorEntry>>,
}

impl grim_tensor::provider::TensorProvider for MemProvider {
    fn get(
        &self,
        name: &str,
    ) -> Result<grim_tensor::provider::RawTensor, grim_tensor::error::Error> {
        let (bytes, shape, dtype, provenance) = self.tensors.get(name).cloned().ok_or_else(|| {
            grim_tensor::error::Error::Backend(format!("missing tensor '{name}'"))
        })?;
        Ok(grim_tensor::provider::RawTensor {
            bytes,
            shape,
            dtype,
            provenance,
        })
    }
    fn meta(
        &self,
        name: &str,
    ) -> Result<grim_tensor::provider::TensorMeta, grim_tensor::error::Error> {
        let (_, shape, dtype, provenance) = self.tensors.get(name).cloned().ok_or_else(|| {
            grim_tensor::error::Error::Backend(format!("missing tensor '{name}'"))
        })?;
        Ok(grim_tensor::provider::TensorMeta {
            dtype,
            provenance,
            shape,
            fusion_mask: 0,
        })
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Process-scoped env override for the opt-in native-arm gates
/// (`GRIM_MOE_NATIVE_FP8/AWQ`). All GPU tests in this binary hold
/// `gpu_lock()`, so no two tests race on the process environment.
// SAFETY: construction/destruction only happens under `gpu_lock()`.
struct NativeArmEnv {
    key: &'static str,
}

impl NativeArmEnv {
    fn enable(key: &'static str) -> Self {
        unsafe {
            std::env::set_var(key, "1");
        }
        Self { key }
    }
}

impl Drop for NativeArmEnv {
    fn drop(&mut self) {
        unsafe {
            std::env::remove_var(self.key);
        }
    }
}

/// Per-row symmetric W4A16 quantization with EXACT dequant round-trip:
/// reference weights are rewritten to `(code - 8) * scale` so parity is not
/// limited by quantization error.
fn w4a16_quantize_rowmajor(
    w: &mut [f32],
    rows: usize,
    k: usize,
    group_size: usize,
) -> Vec<u8> {
    let words_per_row = k / 8;
    let groups_per_row = k.div_ceil(group_size);
    let mut codes = vec![0u32; rows * words_per_row];
    let mut scales = vec![0.0f32; rows * groups_per_row];
    for row in 0..rows {
        for g in 0..groups_per_row {
            let lo = g * group_size;
            let hi = (lo + group_size).min(k);
            let max_abs = (lo..hi)
                .map(|c| w[row * k + c].abs())
                .fold(0.0f32, f32::max);
            let scale = if max_abs == 0.0 { 1e-12 } else { max_abs / 7.0 };
            scales[row * groups_per_row + g] = scale;
            for c in lo..hi {
                let q = ((w[row * k + c] / scale).round() as i32).clamp(-8, 7);
                codes[row * words_per_row + c / 8] |= ((q + 8) as u32) << ((c % 8) * 4);
                w[row * k + c] = q as f32 * scale;
            }
        }
    }
    let mut blob = Vec::new();
    for x in &codes {
        blob.extend_from_slice(&x.to_le_bytes());
    }
    for x in &scales {
        blob.extend_from_slice(&x.to_le_bytes());
    }
    blob
}

#[test]
fn test_quantized_w4a16_d2d_engages_and_matches_native() {
    use grim_models_transformer::shared_moe::{CharonCache, MoeExpert, fused_moe_dispatch_from_logits};

    let _guard = gpu_lock();
    let Some(dev) = gpu_device() else {
        eprintln!("[skip: set GRIM_RUN_GPU_TESTS=1 for quantized D2D test]");
        return;
    };

    const E: usize = 2;
    const HIDDEN: usize = 8;
    const INTER: usize = 16;
    const GROUP: usize = 8;
    const TOP_K: usize = 2; // = E: routing identical regardless of fp noise

    let mut seed = 0xC0FFEEu64;
    let mut rand = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };

    // Packed fixture + exact-dequant native twin.
    let mut packed_map: HashMap<String, (Vec<u8>, Vec<usize>, DType, grim_tensor::QuantProvenance)> =
        HashMap::new();
    let mut native_map: HashMap<String, (Vec<u8>, Vec<usize>, DType, grim_tensor::QuantProvenance)> =
        HashMap::new();
    for (name, out, k) in [
        ("ffn_gate_exps.weight", INTER, HIDDEN),
        ("ffn_up_exps.weight", INTER, HIDDEN),
        ("ffn_down_exps.weight", HIDDEN, INTER),
    ] {
        let mut w: Vec<f32> = (0..E * out * k).map(|_| rand()).collect();
        let blob = w4a16_quantize_rowmajor(&mut w, E * out, k, GROUP);
        packed_map.insert(
            name.to_string(),
            (
                blob,
                vec![E, out, k],
                DType {
                    arith: grim_tensor::ArithType::F32,
                    storage: grim_tensor::Storage::W4A16(
                        grim_tensor::dtype::W4A16Config { group_size: GROUP },
                    ),
                },
                grim_tensor::QuantProvenance::GrimNative,
            ),
        );
        native_map.insert(
            name.to_string(),
            (
                f32_bytes(&w),
                vec![E, out, k],
                DType {
                    arith: grim_tensor::ArithType::F32,
                    storage: grim_tensor::Storage::Native,
                },
                grim_tensor::QuantProvenance::GrimNative,
            ),
        );
    }
    let router: Vec<f32> = vec![
        3.0, 0.1, 0.2, -3.0, 2.5, 0.05, -2.5, 0.3, -2.0, 0.4, 2.8, 0.15, -0.5, 3.2, 0.25, 1.0,
    ];
    for map in [&mut packed_map, &mut native_map] {
        map.insert(
            "ffn_gate_inp.weight".to_string(),
            (
                f32_bytes(&router),
                vec![E, HIDDEN],
                DType {
                    arith: grim_tensor::ArithType::F32,
                    storage: grim_tensor::Storage::Native,
                },
                grim_tensor::QuantProvenance::GrimNative,
            ),
        );
    }

    let run_d2d = |provider: &MemProvider| -> Vec<f32> {
        let ws = grim_nn::WeightSource::root(provider, Device::Rocm(0));
        let bank = ExpertBank::load(&ws, E, HIDDEN, INTER, false).expect("expert bank load");
        let experts: Vec<MoeExpert> = (0..E)
            .map(|e| MoeExpert {
                gate: bank.gate[e].clone(),
                up: bank.up[e].clone(),
                down: bank.down[e].clone(),
            })
            .collect();
        // Gate logits stay on device: project x through the packed bank's
        // router (router is native f32 in both fixtures).
        let gate_w = ws
            .get(Shape::new(vec![E, HIDDEN]), "ffn_gate_inp.weight")
            .unwrap();
        let x_data: Vec<f32> = (0..HIDDEN).map(|i| ((i % 7) as f32 * 0.4) - 1.2).collect();
        let x_gpu = rocm_tensor(&dev, x_data, Shape::new(vec![1, HIDDEN]));
        let router_lin = Linear::from_tensor(gate_w, None);
        let logits_gpu = router_lin.forward(&x_gpu).unwrap();
        let cache = CharonCache::new();
        let out = fused_moe_dispatch_from_logits(
            &dev, &x_gpu, &logits_gpu, &experts, None, TOP_K, 1.0, 0, &cache,
        )
        .unwrap()
        .expect("quantized D2D dispatch returned Ok(None) — silent fallback, FAIL");
        assert!(
            cache.is_routing_engaged(),
            "quantized D2D did not populate routing scratch — engagement FAIL"
        );
        out.to_vec_f32().unwrap()
    };

    let packed_provider = MemProvider {
        tensors: Arc::new(packed_map),
    };
    let native_provider = MemProvider {
        tensors: Arc::new(native_map),
    };
    let got = run_d2d(&packed_provider);
    let want = run_d2d(&native_provider);

    assert_eq!(got.len(), want.len());
    let max_diff = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 5e-3,
        "W4A16 D2D vs exact-dequant native max diff {max_diff:.5} exceeds 5e-3"
    );
    eprintln!("W4A16 D2D engagement parity OK  max_diff={max_diff:.2e}");
}

// ── WI-gpu-native-moe Phase 2: native W8A8-int8 arm ───────────────────────
// Packed-int8 experts must take the NATIVE arm (`DispatchKind::W8a8Native`,
// `grim_moe_fused_dispatch_w8a8_int8` straight from packed blobs) — not the
// dequant fallback — and match the exact-dequant native reference, which
// must itself take `F32Dequant` (proves arm discrimination both ways).

/// Exact-roundtrip int8 row-quantize: `w` rewritten to `code * scale`.
/// Returns packed per-expert blob `[u64 len | codes | f32 scales]`.
fn w8a8_int8_pack_rows(w: &mut [f32], rows: usize, k: usize) -> Vec<u8> {
    let mut codes = vec![0u8; rows * k];
    let mut scales = vec![0.0f32; rows];
    for r in 0..rows {
        let max_abs = (0..k).map(|c| w[r * k + c].abs()).fold(0.0f32, f32::max);
        let scale = if max_abs == 0.0 { 1e-12 } else { max_abs / 127.0 };
        scales[r] = scale;
        for c in 0..k {
            let q = ((w[r * k + c] / scale).round() as i32).clamp(-128, 127);
            codes[r * k + c] = q as i8 as u8;
            w[r * k + c] = q as f32 * scale;
        }
    }
    let mut blob = Vec::with_capacity(8 + codes.len() + scales.len() * 4);
    blob.extend_from_slice(&(codes.len() as u64).to_le_bytes());
    blob.extend_from_slice(&codes);
    for s in &scales {
        blob.extend_from_slice(&s.to_le_bytes());
    }
    blob
}

#[test]
fn test_w8a8_native_d2d_engages_and_matches_dequant() {
    use grim_models_transformer::shared_moe::{
        CharonCache, DispatchKind, MoeExpert, fused_moe_dispatch_from_logits,
    };

    let _guard = gpu_lock();
    let Some(dev) = gpu_device() else {
        eprintln!("[skip: set GRIM_RUN_GPU_TESTS=1 for W8A8 native D2D test]");
        return;
    };

    const E: usize = 2;
    const HIDDEN: usize = 8;
    const INTER: usize = 16;
    const TOP_K: usize = 2; // = E: routing identical regardless of fp noise

    let mut seed = 0xC0FFEEu64;
    let mut rand = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };

    let pack_dtype = || DType {
        arith: grim_tensor::ArithType::F32,
        storage: grim_tensor::Storage::CompressedTensorsW8A8Int8,
    };
    let native_dtype = || DType {
        arith: grim_tensor::ArithType::F32,
        storage: grim_tensor::Storage::Native,
    };

    // O(1) magnitudes so a wrong arm or bad scales exceed tolerance.
    let mut packed_map: HashMap<String, (Vec<u8>, Vec<usize>, DType, grim_tensor::QuantProvenance)> =
        HashMap::new();
    let mut native_map: HashMap<String, (Vec<u8>, Vec<usize>, DType, grim_tensor::QuantProvenance)> =
        HashMap::new();
    for (name, out, k) in [
        ("ffn_gate_exps.weight", INTER, HIDDEN),
        ("ffn_up_exps.weight", INTER, HIDDEN),
        ("ffn_down_exps.weight", HIDDEN, INTER),
    ] {
        let mut w: Vec<f32> = (0..E * out * k).map(|_| rand()).collect();
        // Per-expert-concat bank layout (split fn passes it through verbatim).
        let mut bank = Vec::new();
        let mut flat = Vec::with_capacity(E * out * k);
        for e in 0..E {
            let mut slab = w[e * out * k..(e + 1) * out * k].to_vec();
            bank.extend_from_slice(&w8a8_int8_pack_rows(&mut slab, out, k));
            flat.extend_from_slice(&slab);
        }
        // Rewrite source to exact-dequant values (shared by the twin below).
        for e in 0..E {
            w[e * out * k..(e + 1) * out * k]
                .copy_from_slice(&flat[e * out * k..(e + 1) * out * k]);
        }
        packed_map.insert(name.to_string(), (bank, vec![E, out, k], pack_dtype(), grim_tensor::QuantProvenance::GrimNative));
        native_map.insert(
            name.to_string(),
            (
                f32_bytes(&flat),
                vec![E, out, k],
                native_dtype(),
                grim_tensor::QuantProvenance::GrimNative,
            ),
        );
    }
    let router: Vec<f32> = vec![
        3.0, 0.1, 0.2, -3.0, 2.5, 0.05, -2.5, 0.3, -2.0, 0.4, 2.8, 0.15, -0.5, 3.2, 0.25, 1.0,
    ];
    for map in [&mut packed_map, &mut native_map] {
        map.insert(
            "ffn_gate_inp.weight".to_string(),
            (
                f32_bytes(&router),
                vec![E, HIDDEN],
                native_dtype(),
                grim_tensor::QuantProvenance::GrimNative,
            ),
        );
    }

    let run_d2d = |provider: &MemProvider| -> (Vec<f32>, DispatchKind) {
        let ws = grim_nn::WeightSource::root(provider, Device::Rocm(0));
        let bank = ExpertBank::load(&ws, E, HIDDEN, INTER, false).expect("expert bank load");
        let experts: Vec<MoeExpert> = (0..E)
            .map(|e| MoeExpert {
                gate: bank.gate[e].clone(),
                up: bank.up[e].clone(),
                down: bank.down[e].clone(),
            })
            .collect();
        let gate_w = ws
            .get(Shape::new(vec![E, HIDDEN]), "ffn_gate_inp.weight")
            .unwrap();
        let x_data: Vec<f32> = (0..HIDDEN).map(|i| ((i % 7) as f32 * 0.4) - 1.2).collect();
        let x_gpu = rocm_tensor(&dev, x_data, Shape::new(vec![1, HIDDEN]));
        let router_lin = Linear::from_tensor(gate_w, None);
        let logits_gpu = router_lin.forward(&x_gpu).unwrap();
        let cache = CharonCache::new();
        let out = fused_moe_dispatch_from_logits(
            &dev, &x_gpu, &logits_gpu, &experts, None, TOP_K, 1.0, 0, &cache,
        )
        .unwrap()
        .expect("W8A8 D2D dispatch returned Ok(None) — silent fallback, FAIL");
        assert!(
            cache.is_routing_engaged(),
            "W8A8 D2D did not populate routing scratch — engagement FAIL"
        );
        (out.to_vec_f32().unwrap(), cache.last_dispatch_kind())
    };

    let (got, got_kind) = run_d2d(&MemProvider {
        tensors: Arc::new(packed_map),
    });
    assert_eq!(
        got_kind,
        DispatchKind::W8a8Native,
        "packed-int8 experts must take the NATIVE arm, got {got_kind:?}"
    );
    let (want, want_kind) = run_d2d(&MemProvider {
        tensors: Arc::new(native_map),
    });
    assert_eq!(
        want_kind,
        DispatchKind::F32Dequant,
        "native-f32 experts must take the F32 arm, got {want_kind:?}"
    );

    assert_eq!(got.len(), want.len());
    let max_diff = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 5e-3,
        "W8A8 native vs dequant-arm max diff {max_diff:.5} exceeds 5e-3"
    );
    eprintln!("W8A8 native-arm parity OK  max_diff={max_diff:.2e}");
    dev.synchronize(); // drain deferred async work so faults surface here, not in the next test
    // Also drain the SHARED device singleton: the charon dispatch path
    // launches through Dev::shared(ordinal), whose streams are distinct from
    // this test's device instance.
    grim_backend_rocm::RocmDevice::shared(0).synchronize();
    // The charon d2d path can enqueue on a drifted foreign context (node-1);
    // drain the shared singletons so deferred faults surface inside this test.
    if let Ok(d1) = grim_backend_rocm::RocmDevice::try_new(1) {
        d1.synchronize();
    }
}

// ── WI-gpu-native-moe Phase 2: fp8 / awq / mxfp4 native arms ───────────────
// Same gate as the int8 test: packed experts must take their NATIVE arm
// (asserted via `DispatchKind`), the exact-dequant twin must take
// `F32Dequant`, and the two must agree.

/// Exact fp8-E4M3 pack (nearest valid code; exp-15 NaN range excluded —
/// see backend harness for why). Returns per-expert blob.
fn fp8_pack_rows(w: &mut [f32]) -> Vec<u8> {
    let max_abs = w.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let scale = if max_abs == 0.0 { 1e-12 } else { max_abs / 240.0 };
    let mut codes = vec![0u8; w.len()];
    for (i, v) in w.iter_mut().enumerate() {
        let target = *v / scale;
        let mut best = 0u8;
        let mut best_d = f32::INFINITY;
        for c in 0..256u16 {
            if (c & 0x7F) >= 0x78 {
                continue;
            }
            let d = (grim_quant::fp8_e4m3_to_f32(c as u8) - target).abs();
            if d < best_d {
                best_d = d;
                best = c as u8;
            }
        }
        codes[i] = best;
        *v = grim_quant::fp8_e4m3_to_f32(best) * scale;
    }
    let mut blob = Vec::with_capacity(8 + codes.len() + 4);
    blob.extend_from_slice(&(codes.len() as u64).to_le_bytes());
    blob.extend_from_slice(&codes);
    blob.extend_from_slice(&scale.to_le_bytes());
    blob
}

#[test]
fn test_w8a8_fp8_native_d2d_engages_and_matches_dequant() {
    use grim_models_transformer::shared_moe::{
        CharonCache, DispatchKind, MoeExpert, fused_moe_dispatch_from_logits,
    };

    let _guard = gpu_lock();
    let Some(dev) = gpu_device() else {
        eprintln!("[skip: set GRIM_RUN_GPU_TESTS=1 for fp8 native D2D test]");
        return;
    };
    // Opt-in: the fp8 native arm trails dequant on latency (gated by default).
    let _native = NativeArmEnv::enable("GRIM_MOE_NATIVE_FP8");

    const E: usize = 2;
    const HIDDEN: usize = 8;
    const INTER: usize = 16;
    const TOP_K: usize = 2;

    let mut s = 0x5EEDu64;
    let mut rand = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((s >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };

    let pack_dtype = || DType {
        arith: grim_tensor::ArithType::F32,
        storage: grim_tensor::Storage::CompressedTensorsW8A8Fp8,
    };
    let native_dtype = || DType {
        arith: grim_tensor::ArithType::F32,
        storage: grim_tensor::Storage::Native,
    };

    let mut packed_map: HashMap<String, (Vec<u8>, Vec<usize>, DType, grim_tensor::QuantProvenance)> =
        HashMap::new();
    let mut native_map: HashMap<String, (Vec<u8>, Vec<usize>, DType, grim_tensor::QuantProvenance)> =
        HashMap::new();
    for (name, out, k) in [
        ("ffn_gate_exps.weight", INTER, HIDDEN),
        ("ffn_up_exps.weight", INTER, HIDDEN),
        ("ffn_down_exps.weight", HIDDEN, INTER),
    ] {
        let w: Vec<f32> = (0..E * out * k).map(|_| rand()).collect();
        let mut bank = Vec::new();
        let mut flat = Vec::with_capacity(E * out * k);
        for e in 0..E {
            let mut slab = w[e * out * k..(e + 1) * out * k].to_vec();
            bank.extend_from_slice(&fp8_pack_rows(&mut slab));
            flat.extend_from_slice(&slab);
        }
        packed_map.insert(name.to_string(), (bank, vec![E, out, k], pack_dtype(), grim_tensor::QuantProvenance::GrimNative));
        native_map.insert(
            name.to_string(),
            (f32_bytes(&flat), vec![E, out, k], native_dtype(), grim_tensor::QuantProvenance::GrimNative),
        );
    }
    let router: Vec<f32> = vec![
        3.0, 0.1, 0.2, -3.0, 2.5, 0.05, -2.5, 0.3, -2.0, 0.4, 2.8, 0.15, -0.5, 3.2, 0.25, 1.0,
    ];
    for map in [&mut packed_map, &mut native_map] {
        map.insert(
            "ffn_gate_inp.weight".to_string(),
            (f32_bytes(&router), vec![E, HIDDEN], native_dtype(), grim_tensor::QuantProvenance::GrimNative),
        );
    }

    let run_d2d = |provider: &MemProvider| -> (Vec<f32>, DispatchKind) {
        let ws = grim_nn::WeightSource::root(provider, Device::Rocm(0));
        let bank = ExpertBank::load(&ws, E, HIDDEN, INTER, false).expect("expert bank load");
        let experts: Vec<MoeExpert> = (0..E)
            .map(|e| MoeExpert {
                gate: bank.gate[e].clone(),
                up: bank.up[e].clone(),
                down: bank.down[e].clone(),
            })
            .collect();
        let gate_w = ws.get(Shape::new(vec![E, HIDDEN]), "ffn_gate_inp.weight").unwrap();
        let x_data: Vec<f32> = (0..HIDDEN).map(|i| ((i % 7) as f32 * 0.4) - 1.2).collect();
        let x_gpu = rocm_tensor(&dev, x_data, Shape::new(vec![1, HIDDEN]));
        let logits_gpu = Linear::from_tensor(gate_w, None).forward(&x_gpu).unwrap();
        let cache = CharonCache::new();
        let out = fused_moe_dispatch_from_logits(
            &dev, &x_gpu, &logits_gpu, &experts, None, TOP_K, 1.0, 0, &cache,
        )
        .unwrap()
        .expect("fp8 D2D dispatch returned Ok(None) — silent fallback, FAIL");
        assert!(cache.is_routing_engaged(), "fp8 D2D routing not engaged — FAIL");
        (out.to_vec_f32().unwrap(), cache.last_dispatch_kind())
    };

    let (got, got_kind) = run_d2d(&MemProvider { tensors: Arc::new(packed_map) });
    assert_eq!(got_kind, DispatchKind::W8a8Fp8Native, "packed-fp8 must take NATIVE arm, got {got_kind:?}");
    let (want, want_kind) = run_d2d(&MemProvider { tensors: Arc::new(native_map) });
    assert_eq!(want_kind, DispatchKind::F32Dequant, "native-f32 must take F32 arm, got {want_kind:?}");

    let max_diff = got.iter().zip(want.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_diff < 5e-3, "W8A8 native vs dequant-arm max diff {max_diff:.5} exceeds 5e-3");
    eprintln!("W8A8 native-arm parity OK  max_diff={max_diff:.2e}");
}

#[test]
fn test_w8a8_dot4_selected_at_aligned_shapes() {
    use grim_models_transformer::shared_moe::{
        CharonCache, DispatchKind, MoeExpert, fused_moe_dispatch_from_logits,
    };

    let _guard = gpu_lock();
    let Some(dev) = gpu_device() else {
        eprintln!("[skip: set GRIM_RUN_GPU_TESTS=1 for dot4 selection test]");
        return;
    };
    if !grim_backend_rocm::kernels::charon::dot4_supported(
        grim_backend_rocm::RocmDevice::shared(0).gcn_arch(),
    ) {
        eprintln!("[skip: no dot4 on this arch]");
        return;
    }

    // hidden % 32 == 0 → the V_DOT4 contraction must be selected.
    const E: usize = 2;
    const HIDDEN: usize = 64;
    const INTER: usize = 64;
    const TOP_K: usize = 2;

    let mut seed = 0xD074u64;
    let mut rand = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    // O(1) magnitudes (accumulating to O(1e4) outputs at these widths):
    // the twin gate below is RELATIVE because dot4 carries Q8_1
    // activation-quant noise proportional to magnitude.
    let pack_dtype = || DType {
        arith: grim_tensor::ArithType::F32,
        storage: grim_tensor::Storage::CompressedTensorsW8A8Int8,
    };
    let native_dtype = || DType {
        arith: grim_tensor::ArithType::F32,
        storage: grim_tensor::Storage::Native,
    };
    let mut packed_map: HashMap<String, (Vec<u8>, Vec<usize>, DType, grim_tensor::QuantProvenance)> =
        HashMap::new();
    let mut native_map: HashMap<String, (Vec<u8>, Vec<usize>, DType, grim_tensor::QuantProvenance)> =
        HashMap::new();
    for (name, out, k) in [
        ("ffn_gate_exps.weight", INTER, HIDDEN),
        ("ffn_up_exps.weight", INTER, HIDDEN),
        ("ffn_down_exps.weight", HIDDEN, INTER),
    ] {
        let w: Vec<f32> = (0..E * out * k).map(|_| rand()).collect();
        let mut bank = Vec::new();
        let mut flat = Vec::with_capacity(E * out * k);
        for e in 0..E {
            let mut slab = w[e * out * k..(e + 1) * out * k].to_vec();
            bank.extend_from_slice(&w8a8_int8_pack_rows(&mut slab, out, k));
            flat.extend_from_slice(&slab);
        }
        packed_map.insert(name.to_string(), (bank, vec![E, out, k], pack_dtype(), grim_tensor::QuantProvenance::GrimNative));
        native_map.insert(
            name.to_string(),
            (f32_bytes(&flat), vec![E, out, k], native_dtype(), grim_tensor::QuantProvenance::GrimNative),
        );
    }
    let router: Vec<f32> = (0..E * HIDDEN).map(|_| rand()).collect();
    for map in [&mut packed_map, &mut native_map] {
        map.insert(
            "ffn_gate_inp.weight".to_string(),
            (f32_bytes(&router), vec![E, HIDDEN], native_dtype(), grim_tensor::QuantProvenance::GrimNative),
        );
    }
    // One shared input: both arms must see identical activations or the
    // twin comparison is meaningless.
    let x_data: Vec<f32> = (0..HIDDEN).map(|_| rand()).collect();

    let run_d2d = |provider: &MemProvider| -> (Vec<f32>, DispatchKind) {
        let ws = grim_nn::WeightSource::root(provider, Device::Rocm(0));
        let bank = ExpertBank::load(&ws, E, HIDDEN, INTER, false).expect("expert bank load");
        let experts: Vec<MoeExpert> = (0..E)
            .map(|e| MoeExpert {
                gate: bank.gate[e].clone(),
                up: bank.up[e].clone(),
                down: bank.down[e].clone(),
            })
            .collect();
        let gate_w = ws.get(Shape::new(vec![E, HIDDEN]), "ffn_gate_inp.weight").unwrap();
        let x_gpu = rocm_tensor(&dev, x_data.clone(), Shape::new(vec![1, HIDDEN]));
        let logits_gpu = Linear::from_tensor(gate_w, None).forward(&x_gpu).unwrap();
        let cache = CharonCache::new();
        let out = fused_moe_dispatch_from_logits(
            &dev, &x_gpu, &logits_gpu, &experts, None, TOP_K, 1.0, 0, &cache,
        )
        .unwrap()
        .expect("dot4 D2D dispatch returned Ok(None) — silent fallback, FAIL");
        (out.to_vec_f32().unwrap(), cache.last_dispatch_kind())
    };

    let (got, got_kind) = run_d2d(&MemProvider { tensors: Arc::new(packed_map) });
    assert_eq!(got_kind, DispatchKind::W8a8NativeDot4, "aligned int8 must select DOT4, got {got_kind:?}");
    let (want, _) = run_d2d(&MemProvider { tensors: Arc::new(native_map) });
    // RELATIVE tolerance: the dot4 contraction quantizes activations to
    // int8 (Q8_1, ~1/127 per element), so absolute error scales with output
    // magnitude (here O(1e4) from 64-wide O(1) accumulation). Misalignment
    // or decode bugs show up as O(100%) relative errors; measured Q8_1
    // noise is ~0.1%.
    let max_rel = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
        .fold(0.0f32, f32::max);
    let max_diff = got.iter().zip(want.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("dot4-selection parity OK  max_abs_diff={max_diff:.2e} max_rel={max_rel:.2e}");
    assert!(max_rel < 1e-2, "dot4 vs dequant twin blew past Q8_1 noise: rel {max_rel:.5}");
}

#[test]
fn test_fp8_native_arm_gated_off_by_default() {
    use grim_models_transformer::shared_moe::{
        CharonCache, DispatchKind, MoeExpert, fused_moe_dispatch_from_logits,
    };

    let _guard = gpu_lock();
    let Some(dev) = gpu_device() else {
        eprintln!("[skip: set GRIM_RUN_GPU_TESTS=1 for fp8 gate test]");
        return;
    };
    // Ensure a leaked env from another test cannot fake this result.
    unsafe {
        std::env::remove_var("GRIM_MOE_NATIVE_FP8");
    }

    const E: usize = 2;
    const HIDDEN: usize = 8;
    const INTER: usize = 16;
    let fp8_dtype = || DType {
        arith: grim_tensor::ArithType::F32,
        storage: grim_tensor::Storage::CompressedTensorsW8A8Fp8,
    };
    let mut map: HashMap<String, (Vec<u8>, Vec<usize>, DType, grim_tensor::QuantProvenance)> =
        HashMap::new();
    for (name, out, k, seed) in [
        ("ffn_gate_exps.weight", INTER, HIDDEN, 1u64),
        ("ffn_up_exps.weight", INTER, HIDDEN, 2u64),
        ("ffn_down_exps.weight", HIDDEN, INTER, 3u64),
    ] {
        let mut bank = Vec::new();
        for e in 0..E {
            let mut slab: Vec<f32> = (0..out * k)
                .map(|i| (((i + e * 131) as f32 + seed as f32) * 0.31).sin())
                .collect();
            bank.extend_from_slice(&fp8_pack_rows(&mut slab));
        }
        map.insert(
            name.to_string(),
            (bank, vec![E, out, k], fp8_dtype(), grim_tensor::QuantProvenance::GrimNative),
        );
    }
    let router: Vec<f32> = (0..E * HIDDEN).map(|i| ((i as f32 + 1.0) * 0.5).sin()).collect();
    map.insert(
        "ffn_gate_inp.weight".to_string(),
        (
            f32_bytes(&router),
            vec![E, HIDDEN],
            DType { arith: grim_tensor::ArithType::F32, storage: grim_tensor::Storage::Native },
            grim_tensor::QuantProvenance::GrimNative,
        ),
    );
    let provider = MemProvider { tensors: Arc::new(map) };
    let ws = grim_nn::WeightSource::root(&provider, Device::Rocm(0));
    let bank = ExpertBank::load(&ws, E, HIDDEN, INTER, false).expect("expert bank load");
    let experts: Vec<MoeExpert> = (0..E)
        .map(|e| MoeExpert {
            gate: bank.gate[e].clone(),
            up: bank.up[e].clone(),
            down: bank.down[e].clone(),
        })
        .collect();
    let gate_w = ws.get(Shape::new(vec![E, HIDDEN]), "ffn_gate_inp.weight").unwrap();
    let x_gpu = rocm_tensor(&dev, vec![0.1f32; HIDDEN], Shape::new(vec![1, HIDDEN]));
    let logits_gpu = Linear::from_tensor(gate_w, None).forward(&x_gpu).unwrap();
    let cache = CharonCache::new();
    let out = fused_moe_dispatch_from_logits(
        &dev, &x_gpu, &logits_gpu, &experts, None, 2, 1.0, 0, &cache,
    )
    .unwrap()
    .expect("fp8-gated dispatch must still engage (via F32 arm)");
    assert_eq!(
        cache.last_dispatch_kind(),
        DispatchKind::F32Dequant,
        "packed-fp8 WITHOUT opt-in must take the F32 arm"
    );
    assert_eq!(out.shape().dims(), &[1, HIDDEN]);
}

/// f32 ↔ f16 bit conversions (AWQ scales are f16).
fn f32_to_f16_bits(v: f32) -> u16 {
    if v.is_nan() {
        return 0x7E00;
    }
    let sign = if v.is_sign_negative() { 0x8000u32 } else { 0 };
    let a = v.abs();
    if !a.is_finite() {
        return (sign | 0x7BFF) as u16;
    }
    if a < 5.960_464_5e-8 {
        return sign as u16;
    }
    let b = (a * 4096.0 + 0.5) as u32;
    let exp = ((b >> 23) as i32) - 112;
    if exp >= 31 {
        return (sign | 0x7BFF) as u16;
    }
    if exp <= 0 {
        return (sign | (b >> 14)) as u16;
    }
    (sign | ((exp as u32) << 10) | ((b >> 13) & 0x3FF)) as u16
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut e = 127 - 14;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            (sign << 31) | ((e as u32) << 23) | (m << 13)
        }
    } else if exp == 31 {
        (sign << 31) | (0xFF << 23) | (mant << 13)
    } else {
        (sign << 31) | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// Exact AWQ-4bit pack of one [rows, k] slab (zero=8, f16 scales, group 8);
/// slab rewritten to decoded values. Returns per-expert blob.
fn awq4_pack_rows(w: &mut [f32], rows: usize, k: usize) -> Vec<u8> {
    const GROUP: usize = 8;
    const VPW: usize = 8;
    let groups = k.div_ceil(GROUP);
    let qw_len = k.div_ceil(VPW) * rows * 4;
    let qz_len = groups * rows.div_ceil(VPW) * 4;
    let mut qw = vec![0u32; qw_len / 4];
    let mut sc = vec![0u16; groups * rows];
    for r in 0..rows {
        for g in 0..groups {
            let lo = g * GROUP;
            let hi = (lo + GROUP).min(k);
            let max_abs = (lo..hi).map(|c| w[r * k + c].abs()).fold(0.0f32, f32::max);
            let fscale = f32_to_f16_bits(if max_abs == 0.0 { 1e-12 } else { max_abs / 7.0 });
            sc[g * rows + r] = fscale;
            let fsc = f16_bits_to_f32(fscale);
            for c in lo..hi {
                let q = ((w[r * k + c] / fsc).round().clamp(-8.0, 7.0) as i32 + 8).clamp(0, 15) as u32;
                qw[(c / VPW) * rows + r] |= q << ((c % VPW) * 4);
                w[r * k + c] = (q as f32 - 8.0) * fsc;
            }
        }
    }
    let mut qz = vec![0u32; qz_len / 4];
    for g in 0..groups {
        for c in 0..rows {
            qz[g * rows.div_ceil(VPW) + c / VPW] |= 8 << ((c % VPW) * 4);
        }
    }
    let mut blob = Vec::new();
    blob.extend_from_slice(&(qw_len as u64).to_le_bytes());
    for x in &qw {
        blob.extend_from_slice(&x.to_le_bytes());
    }
    blob.extend_from_slice(&(qz_len as u64).to_le_bytes());
    for x in &qz {
        blob.extend_from_slice(&x.to_le_bytes());
    }
    blob.extend_from_slice(&((groups * rows * 2) as u64).to_le_bytes());
    for x in &sc {
        blob.extend_from_slice(&x.to_le_bytes());
    }
    blob
}

#[test]
fn test_awq_native_d2d_engages_and_matches_dequant() {
    use grim_models_transformer::shared_moe::{
        CharonCache, DispatchKind, MoeExpert, fused_moe_dispatch_from_logits,
    };

    let _guard = gpu_lock();
    let Some(dev) = gpu_device() else {
        eprintln!("[skip: set GRIM_RUN_GPU_TESTS=1 for AWQ native D2D test]");
        return;
    };
    // Opt-in: the AWQ native arm trails dequant on latency (gated by default).
    let _native = NativeArmEnv::enable("GRIM_MOE_NATIVE_AWQ");

    const E: usize = 2;
    const HIDDEN: usize = 8;
    const INTER: usize = 16;
    const TOP_K: usize = 2;
    const BITS: u8 = 4;
    const GROUP: usize = 8;

    let mut s = 0xA9u64;
    let mut rand = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((s >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };

    let awq_dtype = || DType {
        arith: grim_tensor::ArithType::F32,
        storage: grim_tensor::Storage::Awq(grim_tensor::dtype::AwqStorageConfig {
            bits: BITS,
            group_size: GROUP,
        }),
    };
    let native_dtype = || DType {
        arith: grim_tensor::ArithType::F32,
        storage: grim_tensor::Storage::Native,
    };

    let mut packed_map: HashMap<String, (Vec<u8>, Vec<usize>, DType, grim_tensor::QuantProvenance)> =
        HashMap::new();
    let mut native_map: HashMap<String, (Vec<u8>, Vec<usize>, DType, grim_tensor::QuantProvenance)> =
        HashMap::new();
    for (name, out, k) in [
        ("ffn_gate_exps.weight", INTER, HIDDEN),
        ("ffn_up_exps.weight", INTER, HIDDEN),
        ("ffn_down_exps.weight", HIDDEN, INTER),
    ] {
        let w: Vec<f32> = (0..E * out * k).map(|_| rand()).collect();
        let mut bank = Vec::new();
        let mut flat = Vec::with_capacity(E * out * k);
        for e in 0..E {
            let mut slab = w[e * out * k..(e + 1) * out * k].to_vec();
            bank.extend_from_slice(&awq4_pack_rows(&mut slab, out, k));
            flat.extend_from_slice(&slab);
        }
        packed_map.insert(name.to_string(), (bank, vec![E, out, k], awq_dtype(), grim_tensor::QuantProvenance::GrimNative));
        native_map.insert(
            name.to_string(),
            (f32_bytes(&flat), vec![E, out, k], native_dtype(), grim_tensor::QuantProvenance::GrimNative),
        );
    }
    let router: Vec<f32> = vec![
        3.0, 0.1, 0.2, -3.0, 2.5, 0.05, -2.5, 0.3, -2.0, 0.4, 2.8, 0.15, -0.5, 3.2, 0.25, 1.0,
    ];
    for map in [&mut packed_map, &mut native_map] {
        map.insert(
            "ffn_gate_inp.weight".to_string(),
            (f32_bytes(&router), vec![E, HIDDEN], native_dtype(), grim_tensor::QuantProvenance::GrimNative),
        );
    }
    // One shared input: both arms must see identical activations or the
    // twin comparison is meaningless.
    let x_data: Vec<f32> = (0..HIDDEN).map(|_| rand()).collect();

    let run_d2d = |provider: &MemProvider| -> (Vec<f32>, DispatchKind) {
        let ws = grim_nn::WeightSource::root(provider, Device::Rocm(0));
        let bank = ExpertBank::load(&ws, E, HIDDEN, INTER, false).expect("expert bank load");
        let experts: Vec<MoeExpert> = (0..E)
            .map(|e| MoeExpert {
                gate: bank.gate[e].clone(),
                up: bank.up[e].clone(),
                down: bank.down[e].clone(),
            })
            .collect();
        let gate_w = ws.get(Shape::new(vec![E, HIDDEN]), "ffn_gate_inp.weight").unwrap();
        let x_gpu = rocm_tensor(&dev, x_data.clone(), Shape::new(vec![1, HIDDEN]));
        let logits_gpu = Linear::from_tensor(gate_w, None).forward(&x_gpu).unwrap();
        let cache = CharonCache::new();
        let out = fused_moe_dispatch_from_logits(
            &dev, &x_gpu, &logits_gpu, &experts, None, TOP_K, 1.0, 0, &cache,
        )
        .unwrap()
        .expect("AWQ D2D dispatch returned Ok(None) — silent fallback, FAIL");
        (out.to_vec_f32().unwrap(), cache.last_dispatch_kind())
    };

    let (got, got_kind) = run_d2d(&MemProvider { tensors: Arc::new(packed_map) });
    assert_eq!(got_kind, DispatchKind::AwqNative, "packed-AWQ must take NATIVE arm, got {got_kind:?}");
    let (want, want_kind) = run_d2d(&MemProvider { tensors: Arc::new(native_map) });
    assert_eq!(want_kind, DispatchKind::F32Dequant, "native-f32 must take F32 arm, got {want_kind:?}");
    let max_diff = got.iter().zip(want.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("AWQ native-arm parity OK  max_diff={max_diff:.2e}");
    dev.synchronize(); // drain deferred async work so faults surface here, not in the next test
    // Also drain the SHARED device singleton: the charon dispatch path
    // launches through Dev::shared(ordinal), whose streams are distinct from
    // this test's device instance.
    grim_backend_rocm::RocmDevice::shared(0).synchronize();
    // The charon d2d path can enqueue on a drifted foreign context (node-1);
    // drain the shared singletons so deferred faults surface inside this test.
    if let Ok(d1) = grim_backend_rocm::RocmDevice::try_new(1) {
        d1.synchronize();
    }
    assert!(max_diff < 5e-3, "AWQ native vs dequant twin diverged: {max_diff:.5}");
}

#[test]
fn test_awq_native_arm_gated_off_by_default() {
    use grim_models_transformer::shared_moe::{
        CharonCache, DispatchKind, MoeExpert, fused_moe_dispatch_from_logits,
    };

    let _guard = gpu_lock();
    let Some(dev) = gpu_device() else {
        eprintln!("[skip: set GRIM_RUN_GPU_TESTS=1 for AWQ gate test]");
        return;
    };
    unsafe {
        std::env::remove_var("GRIM_MOE_NATIVE_AWQ");
    }

    const E: usize = 2;
    const HIDDEN: usize = 8;
    const INTER: usize = 16;
    const BITS: u8 = 4;
    const GROUP: usize = 8;
    let awq_dtype = || DType {
        arith: grim_tensor::ArithType::F32,
        storage: grim_tensor::Storage::Awq(grim_tensor::dtype::AwqStorageConfig {
            bits: BITS,
            group_size: GROUP,
        }),
    };
    let mut map: HashMap<String, (Vec<u8>, Vec<usize>, DType, grim_tensor::QuantProvenance)> =
        HashMap::new();
    for (name, out, k, seed) in [
        ("ffn_gate_exps.weight", INTER, HIDDEN, 7u64),
        ("ffn_up_exps.weight", INTER, HIDDEN, 8u64),
        ("ffn_down_exps.weight", HIDDEN, INTER, 9u64),
    ] {
        let mut bank = Vec::new();
        for e in 0..E {
            let mut slab: Vec<f32> = (0..out * k)
                .map(|i| (((i + e * 131) as f32 + seed as f32) * 0.29).sin())
                .collect();
            bank.extend_from_slice(&awq4_pack_rows(&mut slab, out, k));
        }
        map.insert(
            name.to_string(),
            (bank, vec![E, out, k], awq_dtype(), grim_tensor::QuantProvenance::GrimNative),
        );
    }
    let router: Vec<f32> = (0..E * HIDDEN).map(|i| ((i as f32 + 1.0) * 0.5).sin()).collect();
    map.insert(
        "ffn_gate_inp.weight".to_string(),
        (
            f32_bytes(&router),
            vec![E, HIDDEN],
            DType { arith: grim_tensor::ArithType::F32, storage: grim_tensor::Storage::Native },
            grim_tensor::QuantProvenance::GrimNative,
        ),
    );
    let provider = MemProvider { tensors: Arc::new(map) };
    let ws = grim_nn::WeightSource::root(&provider, Device::Rocm(0));
    let bank = ExpertBank::load(&ws, E, HIDDEN, INTER, false).expect("expert bank load");
    let experts: Vec<MoeExpert> = (0..E)
        .map(|e| MoeExpert {
            gate: bank.gate[e].clone(),
            up: bank.up[e].clone(),
            down: bank.down[e].clone(),
        })
        .collect();
    let gate_w = ws.get(Shape::new(vec![E, HIDDEN]), "ffn_gate_inp.weight").unwrap();
    let x_gpu = rocm_tensor(&dev, vec![0.1f32; HIDDEN], Shape::new(vec![1, HIDDEN]));
    let logits_gpu = Linear::from_tensor(gate_w, None).forward(&x_gpu).unwrap();
    let cache = CharonCache::new();
    let out = fused_moe_dispatch_from_logits(
        &dev, &x_gpu, &logits_gpu, &experts, None, 2, 1.0, 0, &cache,
    )
    .unwrap()
    .expect("AWQ-gated dispatch must still engage (via F32 arm)");
    assert_eq!(
        cache.last_dispatch_kind(),
        DispatchKind::F32Dequant,
        "packed-AWQ WITHOUT opt-in must take the F32 arm"
    );
    assert_eq!(out.shape().dims(), &[1, HIDDEN]);
}

/// Exact MXFP4 pack of one [rows, k] slab (shared exp 127, nearest E2M1
/// code; slab rewritten to decoded values). Returns framed per-expert blob
/// `[u64 clen | codes | u64 xlen | exps]` (bank-loader format).
fn mxfp4_pack_rows(w: &mut [f32]) -> Vec<u8> {
    let groups = w.len().div_ceil(32);
    let mut codes = vec![0u8; w.len().div_ceil(2)];
    let exps = vec![127u8; groups];
    for (i, v) in w.iter_mut().enumerate() {
        let mut best = 0u8;
        let mut best_d = f32::INFINITY;
        for c in 0..16u8 {
            let d = (grim_quant::mxfp4_e2m1_to_f32(c, 127) - *v).abs();
            if d < best_d {
                best_d = d;
                best = c;
            }
        }
        if i % 2 == 0 {
            codes[i / 2] = (codes[i / 2] & 0xF0) | best;
        } else {
            codes[i / 2] = (codes[i / 2] & 0x0F) | (best << 4);
        }
        *v = grim_quant::mxfp4_e2m1_to_f32(best, 127);
    }
    let mut blob = Vec::with_capacity(16 + codes.len() + exps.len());
    blob.extend_from_slice(&(codes.len() as u64).to_le_bytes());
    blob.extend_from_slice(&codes);
    blob.extend_from_slice(&(exps.len() as u64).to_le_bytes());
    blob.extend_from_slice(&exps);
    blob
}

#[test]
fn test_mxfp4_native_d2d_engages_and_matches_dequant() {
    use grim_models_transformer::shared_moe::{
        CharonCache, DispatchKind, MoeExpert, fused_moe_dispatch_from_logits,
    };

    let _guard = gpu_lock();
    let Some(dev) = gpu_device() else {
        eprintln!("[skip: set GRIM_RUN_GPU_TESTS=1 for MXFP4 native D2D test]");
        return;
    };

    const E: usize = 2;
    const HIDDEN: usize = 8;
    const INTER: usize = 32;
    const TOP_K: usize = 2;

    let mut seed = 0x4B1Fu64;
    let mut rand = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };

    let mxfp4_dtype = || DType {
        arith: grim_tensor::ArithType::F32,
        storage: grim_tensor::Storage::FloatPack(grim_tensor::FloatPackScheme::MxFp4),
    };
    let native_dtype = || DType {
        arith: grim_tensor::ArithType::F32,
        storage: grim_tensor::Storage::Native,
    };

    // Bank-level framed layout: [u64 clen | codes(all experts) | u64 xlen |
    // exps(all experts)] per projection (loader splits per expert).
    let mut packed_map: HashMap<String, (Vec<u8>, Vec<usize>, DType, grim_tensor::QuantProvenance)> =
        HashMap::new();
    let mut native_map: HashMap<String, (Vec<u8>, Vec<usize>, DType, grim_tensor::QuantProvenance)> =
        HashMap::new();
    for (name, out, k) in [
        ("ffn_gate_exps.weight", INTER, HIDDEN),
        ("ffn_up_exps.weight", INTER, HIDDEN),
        ("ffn_down_exps.weight", HIDDEN, INTER),
    ] {
        assert!((out * k) % 32 == 0, "fixture dims must give whole 32-groups");
        let mut codes_all = Vec::new();
        let mut exps_all = Vec::new();
        let mut flat = Vec::with_capacity(E * out * k);
        for e in 0..E {
            let mut w: Vec<f32> = (0..out * k).map(|_| rand()).collect();
            // Clamp to E2M1 range so the nearest-code search is meaningful.
            for v in w.iter_mut() {
                *v = v.clamp(-6.0, 6.0);
            }
            let blob = mxfp4_pack_rows(&mut w);
            let clen = u64::from_le_bytes(blob[0..8].try_into().unwrap()) as usize;
            let xlen_off = 8 + clen;
            let xlen = u64::from_le_bytes(blob[xlen_off..xlen_off + 8].try_into().unwrap()) as usize;
            codes_all.extend_from_slice(&blob[8..8 + clen]);
            exps_all.extend_from_slice(&blob[xlen_off + 8..xlen_off + 8 + xlen]);
            flat.extend_from_slice(&w);
            let _ = e;
        }
        let mut bank = Vec::new();
        bank.extend_from_slice(&(codes_all.len() as u64).to_le_bytes());
        bank.extend_from_slice(&codes_all);
        bank.extend_from_slice(&(exps_all.len() as u64).to_le_bytes());
        bank.extend_from_slice(&exps_all);
        packed_map.insert(name.to_string(), (bank, vec![E, out, k], mxfp4_dtype(), grim_tensor::QuantProvenance::GrimNative));
        native_map.insert(
            name.to_string(),
            (f32_bytes(&flat), vec![E, out, k], native_dtype(), grim_tensor::QuantProvenance::GrimNative),
        );
    }
    let router: Vec<f32> = vec![
        3.0, 0.1, 0.2, -3.0, 2.5, 0.05, -2.5, 0.3, -2.0, 0.4, 2.8, 0.15, -0.5, 3.2, 0.25, 1.0,
    ];
    for map in [&mut packed_map, &mut native_map] {
        map.insert(
            "ffn_gate_inp.weight".to_string(),
            (f32_bytes(&router), vec![E, HIDDEN], native_dtype(), grim_tensor::QuantProvenance::GrimNative),
        );
    }

    let run_d2d = |provider: &MemProvider| -> (Vec<f32>, DispatchKind) {
        let ws = grim_nn::WeightSource::root(provider, Device::Rocm(0));
        let bank = ExpertBank::load(&ws, E, HIDDEN, INTER, false).expect("expert bank load");
        let experts: Vec<MoeExpert> = (0..E)
            .map(|e| MoeExpert {
                gate: bank.gate[e].clone(),
                up: bank.up[e].clone(),
                down: bank.down[e].clone(),
            })
            .collect();
        let gate_w = ws.get(Shape::new(vec![E, HIDDEN]), "ffn_gate_inp.weight").unwrap();
        let x_data: Vec<f32> = (0..HIDDEN).map(|i| ((i % 7) as f32 * 0.4) - 1.2).collect();
        let x_gpu = rocm_tensor(&dev, x_data, Shape::new(vec![1, HIDDEN]));
        let logits_gpu = Linear::from_tensor(gate_w, None).forward(&x_gpu).unwrap();
        let cache = CharonCache::new();
        let out = fused_moe_dispatch_from_logits(
            &dev, &x_gpu, &logits_gpu, &experts, None, TOP_K, 1.0, 0, &cache,
        )
        .unwrap()
        .expect("MXFP4 D2D dispatch returned Ok(None) — silent fallback, FAIL");
        assert!(cache.is_routing_engaged(), "MXFP4 D2D routing not engaged — FAIL");
        (out.to_vec_f32().unwrap(), cache.last_dispatch_kind())
    };

    let (got, got_kind) = run_d2d(&MemProvider { tensors: Arc::new(packed_map) });
    assert_eq!(got_kind, DispatchKind::Mxfp4Native, "packed-MXFP4 must take NATIVE arm, got {got_kind:?}");
    let (want, want_kind) = run_d2d(&MemProvider { tensors: Arc::new(native_map) });
    assert_eq!(want_kind, DispatchKind::F32Dequant, "native-f32 must take F32 arm, got {want_kind:?}");

    let max_diff = got.iter().zip(want.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_diff < 5e-3, "MXFP4 native vs dequant-arm max diff {max_diff:.5} exceeds 5e-3");
    eprintln!("MXFP4 native-arm parity OK  max_diff={max_diff:.2e}");
    dev.synchronize(); // drain deferred async work so faults surface here, not in the next test
    // Also drain the SHARED device singleton: the charon dispatch path
    // launches through Dev::shared(ordinal), whose streams are distinct from
    // this test's device instance.
    grim_backend_rocm::RocmDevice::shared(0).synchronize();
    // The charon d2d path can enqueue on a drifted foreign context (node-1);
    // drain the shared singletons so deferred faults surface inside this test.
    if let Ok(d1) = grim_backend_rocm::RocmDevice::try_new(1) {
        d1.synchronize();
    }
}

// ── WI-gpu-native-moe #1: D2D-vs-host latency gate (≥64 experts) ───────────
// Device-gated + ignored (timing-sensitive; run explicitly on gfx1201).
// Compares, at serving-like MoE shapes:
//   f32 D2D (route on device + resident dispatch) vs f32 host-routed
//     (gate matmul + D2H logits + host top-k + H2D routing upload + grouped
//     dispatch) — the apples-to-apples "eager host-dispatch" predecessor;
//   int8 native D2D vs int8-via-dequant D2D.
// Prints best-of-N ms; soft-gates only against CATASTROPHIC regression
// (D2D > 2x host). Rationale: at decode shapes both sides are dominated by
// the same occupancy-starved kernels, so eager timing cannot resolve small
// deltas — D2D's structural wins (no per-step D2H/H2D, graph-capture
// compatibility) don't show up here. Numbers go to the WI perf gate.
#[ignore = "timing gate: GRIM_RUN_GPU_TESTS=1 cargo test --test moe_all_models_parity_gpu -- --ignored d2d_vs_host_latency_gate"]
#[test]
fn d2d_vs_host_latency_gate() {
    use grim_models_transformer::shared_moe::{
        CharonCache, MoeExpert, fused_moe_dispatch, fused_moe_dispatch_from_logits, route_topk,
    };

    let _guard = gpu_lock();
    let Some(dev) = gpu_device() else {
        eprintln!("[skip: set GRIM_RUN_GPU_TESTS=1 for latency gate]");
        return;
    };

    const E: usize = 64;
    // Narrow widths (expert COUNT is the gating criterion per the WI) so the
    // gate fits shared-box VRAM: f32 stacks = 64*1024*256*4B*3 ≈ 200MB.
    // Full-width serving shapes are covered by the micro-benches; the
    // structural comparison (round-trips eliminated) is width-independent.
    const HIDDEN: usize = 256;
    const INTER: usize = 1024;
    const TOPK: usize = 6;
    const ITERS: usize = 6;

    // f32 experts on device (O(1) magnitudes; bench only, no oracle).
    let lin = |data: Vec<f32>, rows: usize, cols: usize| {
        Linear::from_tensor(rocm_tensor(&dev, data, Shape::new(vec![rows, cols])), None)
    };
    let o1 = |n: usize, s: f32| o1_vec(n, s);
    let experts: Vec<MoeExpert> = (0..E)
        .map(|e| {
            let s = e as f32 + 1.0;
            MoeExpert {
                gate: lin(o1(INTER * HIDDEN, s), INTER, HIDDEN),
                up: lin(o1(INTER * HIDDEN, s + 1000.0), INTER, HIDDEN),
                down: lin(o1(HIDDEN * INTER, s + 2000.0), HIDDEN, INTER),
            }
        })
        .collect();
    let gate_w = o1(E * HIDDEN, 7.0);
    let gate = lin(gate_w, E, HIDDEN);

    for seq in [1usize, 8usize] {
        for route_mode in [0i32, 3i32] {
            let x_data = o1(seq * HIDDEN, 42.0);
            let x_gpu = rocm_tensor(&dev, x_data, Shape::new(vec![seq, HIDDEN]));
            let logits_gpu = gate.forward(&x_gpu).unwrap();
            let cache = CharonCache::new();
            let backend = grim_nn::modules::pick_device_for_storage_device(x_gpu.device());

            // Warmup (stack builds + JIT) outside timing.
            eprintln!("[gate] warmup start e={E} h={HIDDEN} i={INTER} topk={TOPK} seq={seq} mode={route_mode}");
            let _ = fused_moe_dispatch_from_logits(
                backend.as_ref(), &x_gpu, &logits_gpu, &experts, None, TOPK, 1.0, route_mode, &cache,
            )
            .unwrap()
            .unwrap()
            .to_vec_f32()
            .unwrap();
            eprintln!("[gate] warmup done seq={seq} mode={route_mode}");

            // a) D2D.
            let mut best_d2d = f64::INFINITY;
            for _ in 0..ITERS {
                let start = std::time::Instant::now();
                let out = fused_moe_dispatch_from_logits(
                    backend.as_ref(), &x_gpu, &logits_gpu, &experts, None, TOPK, 1.0, route_mode,
                    &cache,
                )
                .unwrap()
                .unwrap();
                let _ = out.to_vec_f32().unwrap();
                best_d2d = best_d2d.min(start.elapsed().as_secs_f64() * 1e3);
            }
            eprintln!("[gate] D2D done seq={seq} mode={route_mode}: {best_d2d:.3}ms");

            // b) Host-routed predecessor (D2H logits + host top-k + upload + grouped).
            let mut best_host = f64::INFINITY;
            for _ in 0..ITERS {
                let start = std::time::Instant::now();
                let logits_v = logits_gpu.to_vec_f32().unwrap();
                let routings = route_topk(&logits_v, E, TOPK).unwrap();
                let out = fused_moe_dispatch(
                    backend.as_ref(), &x_gpu, &experts, None, &routings, 1.0, &cache,
                )
                .unwrap();
                let _ = out.to_vec_f32().unwrap();
                best_host = best_host.min(start.elapsed().as_secs_f64() * 1e3);
            }
            eprintln!("[gate] host done seq={seq} mode={route_mode}: {best_host:.3}ms");

            eprintln!(
                "[gate] f32 e={E} h={HIDDEN} i={INTER} topk={TOPK} seq={seq} mode={route_mode}: D2D={best_d2d:.3}ms host={best_host:.3}ms ratio={:.3}",
                best_d2d / best_host.max(1e-9),
            );
            assert!(
                best_d2d <= 2.0 * best_host.max(1e-9),
                "D2D catastrophically slower than host path (seq={seq} mode={route_mode})"
            );
        }
    }
}

// ── Decode Graph Capture & Replay parity for non-Llama standalone MoEs ─────

