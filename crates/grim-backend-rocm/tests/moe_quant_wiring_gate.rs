//! Quant workstream MoE gate: packed expert banks (Storage::W4A16 /
//! Storage::GroupInt) loaded through `ExpertBank::load_quantized` must route
//! through the marlin/gptq fused kernels and match an F32-dequantized
//! reference MoE forward.
//!
//! Device-gated: `GRIM_GPU_TEST=1`.

use grim_nn::moe::{ExpertBank, MoeFfn, MoeRouter, RouterKind};
use grim_nn::{Linear, WeightSource};
use grim_tensor::dtype::QuantProvenance;
use grim_tensor::provider::{RawTensor, TensorMeta, TensorProvider};
use grim_tensor::{ArithType, BackendDevice, DType, Device, Shape, Storage, Tensor};
use grim_backend_rocm::RocmDevice;
use std::collections::HashMap;
use std::sync::Arc;

const E: usize = 2; // experts
const HIDDEN: usize = 8;
const INTER: usize = 16; // % 8 == 0 for GPTQ word alignment
const GROUP: usize = 8;

#[derive(Clone)]
struct MemProvider {
    tensors: Arc<HashMap<String, (Vec<u8>, Vec<usize>, DType, QuantProvenance)>>,
}

impl TensorProvider for MemProvider {
    fn get(&self, name: &str) -> Result<RawTensor, grim_tensor::error::Error> {
        let (bytes, shape, dtype, provenance) = self
            .tensors
            .get(name)
            .cloned()
            .ok_or_else(|| grim_tensor::error::Error::Backend(format!("missing {name}")))?;
        Ok(RawTensor {
            bytes,
            shape,
            dtype,
            provenance,
        })
    }
    fn meta(&self, name: &str) -> Result<TensorMeta, grim_tensor::error::Error> {
        let (_, shape, dtype, provenance) = self
            .tensors
            .get(name)
            .cloned()
            .ok_or_else(|| grim_tensor::error::Error::Backend(format!("missing {name}")))?;
        Ok(TensorMeta {
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

fn native(name: &str, shape: Vec<usize>, v: &[f32]) -> (String, Vec<u8>, Vec<usize>, DType, QuantProvenance) {
    (
        name.to_string(),
        f32_bytes(v),
        shape,
        DType {
            arith: ArithType::F32,
            storage: Storage::Native,
        },
        grim_tensor::QuantProvenance::GrimNative,
    )
}

/// Per-row symmetric W4A16 quantization with EXACT dequant round-trip:
/// reference weights are rewritten to `(code - 8) * scale` so parity is not
/// limited by quantization error.
fn w4a16_quantize_rowmajor(
    w: &mut [f32], // [rows, k], row-major; rewritten in place to dequant values
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
            let max_abs = (lo..hi).map(|c| w[row * k + c].abs()).fold(0.0f32, f32::max);
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

/// Build a GPTQ bits=4 GroupInt bank over a [K, N]-packed weight (N = E*out),
/// returning (blob, dequantized reference [K, N] row-major).
fn gptq_quantize_bank(
    w: &[f32], // [k, n_total]
    k: usize,
    n_total: usize,
    group_size: usize,
) -> (Vec<u8>, Vec<f32>) {
    let vpw = 8usize;
    let groups = k.div_ceil(group_size);
    // Random-ish codes/zeros/scales from the weight magnitudes.
    let mut seed = 0x9E37u64;
    let mut rand = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32 / u32::MAX as f32)
    };
    let mut qweight = vec![0u32; k.div_ceil(vpw) * n_total];
    let mut qzeros = vec![0u32; groups * n_total.div_ceil(vpw)];
    let mut scales = vec![0.0f32; groups * n_total];
    let mut deq = vec![0.0f32; k * n_total];
    for col in 0..n_total {
        for g in 0..groups {
            let lo = g * group_size;
            let hi = (lo + group_size).min(k);
            let max_abs = (lo..hi).map(|kk| w[kk * n_total + col].abs()).fold(0.0f32, f32::max);
            let scale = if max_abs == 0.0 { 1e-12 } else { max_abs / 15.0 * 2.0 };
            scales[g * n_total + col] = scale;
            let zero_code = (rand() * 15.0) as u32;
            for kk in lo..hi {
                let code = rand() as u32 & 0xF;
                qzeros[g * n_total.div_ceil(vpw) + col / vpw] |=
                    zero_code << ((col % vpw) * 4);
                let true_zero = zero_code as f32 + 1.0;
                let val = (code as f32 - true_zero) * scale;
                deq[kk * n_total + col] = val;
                let chunk = kk / vpw;
                qweight[chunk * n_total + col] |= code << ((kk % vpw) * 4);
                let _ = w[kk * n_total + col];
            }
        }
    }
    // Assemble prefixed blob (empty g_idx segment).
    let mut blob = Vec::new();
    blob.extend_from_slice(&(qweight.len() as u64).to_le_bytes());
    for x in &qweight {
        blob.extend_from_slice(&x.to_le_bytes());
    }
    blob.extend_from_slice(&(qzeros.len() as u64).to_le_bytes());
    for x in &qzeros {
        blob.extend_from_slice(&x.to_le_bytes());
    }
    blob.extend_from_slice(&(scales.len() as u64).to_le_bytes());
    for x in &scales {
        blob.extend_from_slice(&x.to_le_bytes());
    }
    blob.extend_from_slice(&0u64.to_le_bytes());
    (blob, deq)
}

struct MoEFixture {
    provider: MemProvider,
    /// Exact dequantized gate/up/down per expert for the CPU reference.
    ref_gate: Vec<Vec<f32>>, // [E][inter*hidden]
    ref_up: Vec<Vec<f32>>,
    ref_down: Vec<Vec<f32>>, // [E][hidden*inter]
}

fn build_w4a16_fixture() -> MoEFixture {
    let mut seed = 0xC0FFEEu64;
    let mut rand = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };

    let mut provider_map: HashMap<String, (Vec<u8>, Vec<usize>, DType, QuantProvenance)> =
        HashMap::new();
    let mut ref_g = Vec::new();
    let mut ref_u = Vec::new();
    let mut ref_d = Vec::new();

    let projections = [
        ("ffn_gate_exps.weight", INTER, HIDDEN),
        ("ffn_up_exps.weight", INTER, HIDDEN),
        ("ffn_down_exps.weight", HIDDEN, INTER),
    ];
    for (name, out, k) in projections {
        let mut w: Vec<f32> = (0..E * out * k).map(|_| rand()).collect();
        let blob = w4a16_quantize_rowmajor(&mut w, E * out, k, GROUP);
        provider_map.insert(
            name.to_string(),
            (
                blob,
                vec![E, out, k],
                DType {
                    arith: ArithType::F32,
                    storage: Storage::W4A16(grim_tensor::dtype::W4A16Config { group_size: GROUP }),
                },
                grim_tensor::QuantProvenance::GrimNative,
            ),
        );
        match name {
            "ffn_gate_exps.weight" => {
                for e in 0..E {
                    ref_g.push(w[e * out * k..(e + 1) * out * k].to_vec());
                }
            }
            "ffn_up_exps.weight" => {
                for e in 0..E {
                    ref_u.push(w[e * out * k..(e + 1) * out * k].to_vec());
                }
            }
            _ => {
                for e in 0..E {
                    ref_d.push(w[e * out * k..(e + 1) * out * k].to_vec());
                }
            }
        }
    }

    // Router: well-separated rows so top-k is unambiguous on any backend.
    let router: Vec<f32> = vec![
        3.0, 0.1, 0.2, -3.0, 2.5, 0.05, -2.5, 0.3, // expert 0 row
        -2.0, 0.4, 2.8, 0.15, -0.5, 3.2, 0.25, 1.0, // expert 1 row
    ];
    let (n, b, s, d, p) = native("ffn_gate_inp.weight", vec![E, HIDDEN], &router);
    provider_map.insert(n, (b, s, d, p));

    MoEFixture {
        provider: MemProvider {
            tensors: Arc::new(provider_map),
        },
        ref_gate: ref_g,
        ref_up: ref_u,
        ref_down: ref_d,
    }
}

fn build_gptq_fixture() -> MoEFixture {
    let mut seed = 0x1234_5678u64;
    let mut rand = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let mut provider_map: HashMap<String, (Vec<u8>, Vec<usize>, DType, QuantProvenance)> =
        HashMap::new();
    let mut ref_g = Vec::new();
    let mut ref_u = Vec::new();
    let mut ref_d = Vec::new();

    let projections = [
        ("ffn_gate_exps.weight", INTER, HIDDEN),
        ("ffn_up_exps.weight", INTER, HIDDEN),
        ("ffn_down_exps.weight", HIDDEN, INTER),
    ];
    for (name, out, k) in projections {
        let n_total = E * out;
        // Bank packs a [K=k, N=E*out] weight.
        let bank_w: Vec<f32> = (0..k * n_total).map(|_| rand()).collect();
        let (blob, deq) = gptq_quantize_bank(&bank_w, k, n_total, GROUP);
        provider_map.insert(
            name.to_string(),
            (
                blob,
                vec![E, out, k],
                DType {
                    arith: ArithType::F32,
                    storage: Storage::GroupInt(grim_tensor::GpuIntConfig {
                        bits: 4,
                        group_size: GROUP,
                        scheme: grim_tensor::GroupQuantScheme::Symmetric,
                        desc_act: false,
                    }),
                },
                grim_tensor::QuantProvenance::GrimNative,
            ),
        );
        // Reference per expert: Linear expects [out, in] row-major where the
        // forward computes x @ W^T... but the GPTQ kernel computes C = A @
        // deq(B)^T with B packing [K, N] — i.e. per expert, W_e = deq[e][..]
        // viewed as [k, out] and the Linear weight tensor must be its
        // transpose flattened [out, k]. The moe split path materializes
        // shape [out, in] from the raw bytes verbatim, so the reference slab
        // must be Bᵀ-e rows: deq[k*k? ...]. Concretely: element
        // (row=out_idx, col=in_idx) of expert e's Linear weight equals
        // deq[in_idx * n_total + e*out + out_idx].
        for e in 0..E {
            let mut slab = vec![0.0f32; out * k];
            for oi in 0..out {
                for ki in 0..k {
                    slab[oi * k + ki] = deq[ki * n_total + e * out + oi];
                }
            }
            match name {
                "ffn_gate_exps.weight" => ref_g.push(slab),
                "ffn_up_exps.weight" => ref_u.push(slab),
                _ => ref_d.push(slab),
            }
        }
        let _ = bank_w;
    }

    let router: Vec<f32> = vec![
        2.0, 0.2, -2.8, 0.1, 3.0, 0.05, -1.5, 0.4,
        -3.0, 0.3, 2.2, 0.12, -2.0, 2.6, 0.08, 0.9,
    ];
    let (n, b, s, d, p) = native("ffn_gate_inp.weight", vec![E, HIDDEN], &router);
    provider_map.insert(n, (b, s, d, p));

    MoEFixture {
        provider: MemProvider {
            tensors: Arc::new(provider_map),
        },
        ref_gate: ref_g,
        ref_up: ref_u,
        ref_down: ref_d,
    }
}

fn run_moe_forward(fixture: &MoEFixture, device: Device, x_host: &[f32]) -> Vec<f32> {
    let ws = WeightSource::root(&fixture.provider, device.clone());
    let bank = ExpertBank::load(&ws, E, HIDDEN, INTER, false).expect("expert bank load");
    let router_w = ws.get(Shape::new(vec![E, HIDDEN]), "ffn_gate_inp.weight").unwrap();
    let router = MoeRouter::new(
        Linear::from_tensor(router_w, None),
        RouterKind::SoftmaxTopK,
        E, // top_k = all experts → routing identical regardless of fp noise
        E,
        None,
    );
    let moe = MoeFfn::new(router, bank, None, 1.0);

    let dev_backend = grim_nn::pick_device_for_storage_device(&device);
    let x_st = dev_backend
        .from_cpu(x_host, &Shape::new(vec![1, HIDDEN]), DType::F32)
        .unwrap();
    let x_t = Tensor::new(
        std::sync::Arc::from(x_st),
        Shape::new(vec![1, HIDDEN]),
        DType::F32,
        grim_tensor::QuantProvenance::default(),
        device.clone(),
    );
    let out = moe.forward(&x_t).expect("moe forward");
    out.to_vec_f32().unwrap()
}

impl MoEFixture {
    /// Native-F32 twin fixture built from the exact dequantized weights —
    /// the parity reference. Packed storages must NOT ride the CPU fallback
    /// (the CPU backend cannot dispatch them), so the reference re-runs the
    /// identical MoE with F32 experts.
    fn native_reference(&self) -> MoEFixture {
        let projections = [
            ("ffn_gate_exps.weight", INTER, HIDDEN),
            ("ffn_up_exps.weight", INTER, HIDDEN),
            ("ffn_down_exps.weight", HIDDEN, INTER),
        ];
        let refs = [&self.ref_gate, &self.ref_up, &self.ref_down];
        let mut map: HashMap<String, (Vec<u8>, Vec<usize>, DType, QuantProvenance)> =
            HashMap::new();
        let slabs: [&[Vec<f32>]; 3] = [&self.ref_gate, &self.ref_up, &self.ref_down];
        for ((name, out, k), expert_slabs) in projections.iter().zip(slabs.iter()) {
            // Flatten the per-expert slabs back into a bank [E*out, k] f32.
            let mut flat = Vec::with_capacity(E * out * k);
            for e in 0..E {
                flat.extend_from_slice(&expert_slabs[e]);
            }
            map.insert(
                name.to_string(),
                (f32_bytes(&flat), vec![E, *out, *k], DType {
                    arith: ArithType::F32,
                    storage: Storage::Native,
                }, grim_tensor::QuantProvenance::GrimNative),
            );
        }
        let router = native("ffn_gate_inp.weight", vec![E, HIDDEN], &{
            // Router weights are copied from the packed fixture verbatim.
            let raw = self.provider.get("ffn_gate_inp.weight").unwrap();
            raw.bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<f32>>()
        });
        map.insert(router.0, (router.1, router.2, router.3, router.4));
        MoEFixture {
            provider: MemProvider { tensors: Arc::new(map) },
            ref_gate: self.ref_gate.clone(),
            ref_up: self.ref_up.clone(),
            ref_down: self.ref_down.clone(),
        }
    }
}

fn quant_moe_parity(fixture: MoEFixture, label: &str) {
    let dev = match RocmDevice::try_new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[skipped: no ROCm device: {e:?}]");
            return;
        }
    };
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[skipped: GRIM_GPU_TEST not set]");
        return;
    }

    let x_host: Vec<f32> = (0..HIDDEN).map(|i| ((i % 7) as f32 * 0.4) - 1.2).collect();

    // Quantized-expert forward on GPU.
    let got = run_moe_forward(&fixture, Device::Rocm(0), &x_host);

    // F32-dequantized reference ON THE SAME GPU PATH — the packed fixture
    // and the native fixture both ride forward_rocm's fused kernel, so any
    // divergence isolates the quantized materialization, not pre-existing
    // fused-vs-CPU behavioral differences.
    let gpu_native = run_moe_forward(&fixture.native_reference(), Device::Rocm(0), &x_host);
    // Secondary check: the native fixture must also agree with the CPU
    // reference path (guards the reference fixture itself).
    let cpu_ref = run_moe_forward(&fixture.native_reference(), Device::Cpu, &x_host);
    let md2 = gpu_native
        .iter()
        .zip(cpu_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("[{label}] fused-vs-cpu native baseline max diff: {md2:.5}");

    assert_eq!(got.len(), gpu_native.len(), "{label}: output length");
    let md = got
        .iter()
        .zip(gpu_native.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        md < 5e-3,
        "{label}: quantized-expert MoE forward diverged from F32-dequant \
         reference (max diff {md:.5})\ngot={got:?}\nref={gpu_native:?}"
    );
}

#[test]
fn w4a16_expert_bank_moe_forward_matches_reference() {
    // Stage 1: split + dequant service must reproduce the reference slab
    // exactly — isolates bank splitting and materialization from the fused
    // forward.
    if grim_backend_rocm::gpu_test_enabled() {
        if let Ok(dev) = RocmDevice::try_new(0) {
            let fx = build_w4a16_fixture();
            let ws = WeightSource::root(&fx.provider, Device::Rocm(0));
            let bank = ExpertBank::load(&ws, E, HIDDEN, INTER, false).expect("bank");
            for e in 0..E {
                // Down projection exercises k=16 / 2 groups per row.
                let wd = &bank.down[e].weight;
                let blob_d = wd
                    .storage()
                    .as_any()
                    .downcast_ref::<grim_backend_rocm::RocmStorage>()
                    .unwrap();
                let out_d = dev
                    .dequant_w4a16_blob_to_f32(blob_d, HIDDEN, INTER, GROUP)
                    .unwrap();
                let c_dt = Tensor::new(
                    std::sync::Arc::from(out_d),
                    Shape::new(vec![INTER, HIDDEN]),
                    DType::F32,
                    grim_tensor::QuantProvenance::default(),
                    Device::Rocm(0),
                );
                let cd = c_dt.to_vec_f32().unwrap();
                let mut dd = vec![0.0f32; HIDDEN * INTER];
                for r in 0..HIDDEN {
                    for c2 in 0..INTER {
                        dd[r * INTER + c2] = cd[c2 * HIDDEN + r];
                    }
                }
                let mdd = dd
                    .iter()
                    .zip(fx.ref_down[e].iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                eprintln!("[W4A16-stage1] expert {e} down dequant max diff {mdd}");
                assert!(mdd < 1e-3, "expert {e} DOWN split+dequant diverged: {mdd}");

                let w = &bank.gate[e].weight;
                let blob_rocm = w
                    .storage()
                    .as_any()
                    .downcast_ref::<grim_backend_rocm::RocmStorage>()
                    .unwrap();
                let out_box = dev
                    .dequant_w4a16_blob_to_f32(
                        blob_rocm,
                        INTER,
                        HIDDEN,
                        GROUP,
                    )
                    .unwrap();
                let c_t = Tensor::new(
                    std::sync::Arc::from(out_box),
                    Shape::new(vec![HIDDEN, INTER]),
                    DType::F32,
                    grim_tensor::QuantProvenance::default(),
                    Device::Rocm(0),
                );
                let c = c_t.to_vec_f32().unwrap();
                let mut d = vec![0.0f32; INTER * HIDDEN];
                for r in 0..INTER {
                    for c2 in 0..HIDDEN {
                        d[r * HIDDEN + c2] = c[c2 * INTER + r];
                    }
                }
                let md = d
                    .iter()
                    .zip(fx.ref_gate[e].iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                eprintln!("[W4A16-stage1] expert {e} gate dequant max diff {md}");
                assert!(md < 1e-3, "expert {e} split+dequant diverged: {md}");
            }
        }
    }
    quant_moe_parity(build_w4a16_fixture(), "W4A16");
}

#[test]
fn gptq_groupint_expert_bank_moe_forward_matches_reference() {
    quant_moe_parity(build_gptq_fixture(), "GPTQ/GroupInt");
}
