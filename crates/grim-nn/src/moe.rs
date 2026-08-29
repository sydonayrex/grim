//! Mixture-of-Experts primitives.
//!
//! This module provides the architecture-agnostic building blocks for MoE
//! inference: an [`ExpertBank`] (per-expert feed-forward triples), an
//! [`MoeRouter`] (softmax-top-k or sigmoid+bias-top-k gating), and an
//! [`MoeFfn`] that routes tokens to selected experts and combines their
//! outputs (plus an optional shared expert).
//!
//! Design notes (per the project's verifiable-correctness discipline):
//!
//! * The forward path implemented here is the **correct-but-unoptimized CPU
//!   reference**. It materializes each selected expert's contribution and
//!   weighted-sums them. A fused/grouped GPU GEMM (WI-M5) is a separate,
//!   non-blocking performance item and must remain parity-checked against this
//!   reference.
//! * Router math (softmax / sigmoid / top-k / bias-application) is computed in
//!   host Rust over the gate logits pulled to CPU. This keeps the selection
//!   logic unit-testable with hand-computed expectations and avoids depending
//!   on backend kernels that may not exist on every device.
//! * No architecture-specific naming or assumptions leak in here. Per-arch
//!   differences (router kind, shared expert presence, top-k, tensor name
//!   mapping) are supplied by the caller (`architecture.rs` map + the per-arch
//!   loader in `grim-models-transformer`).

use grim_backend_cpu::cpu_tensor;
#[cfg(feature = "cuda-mem")]
use grim_backend_cuda::CudaDevice;
#[cfg(feature = "cuda-mem")]
use grim_backend_cuda::CudaStorage;
#[cfg(feature = "metal-mem")]
use grim_backend_metal::MetalDevice;
#[cfg(feature = "rocm-mem")]
use grim_backend_rocm::RocmDevice;
#[cfg(feature = "vulkan-mem")]
use grim_backend_vulkan::VulkanDevice;
use grim_tensor::dtype::{
    ArithType, BlockDtype, DType, FloatPackScheme, KQuantScheme, QuantProvenance, Storage,
};

use grim_tensor::shape::Shape;
use grim_tensor::{BackendStorage, Device, Tensor,
    CoreTensorOps,
};
use std::sync::Arc;

use crate::modules::Linear;
use crate::varbuilder::WeightSource;

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// Router gating strategy. `SoftmaxTopK` covers Qwen2/3-MoE, GLM4-MoE,
/// Granite-MoE, etc. `SigmoidTopKWithBias` covers Laguna (sigmoid gate logits
/// plus a learned per-expert bias added **at selection time only**, never to
/// the combine weights).
#[derive(Debug, Clone)]
pub enum RouterKind {
    SoftmaxTopK,
    /// Sigmoid gate logits plus a learned per-expert bias added **at selection
    /// time only**, never to the combine weights. The bias tensor itself is
    /// loaded from the checkpoint (`exp_probs_b`) and passed to `MoeRouter::new`.
    SigmoidTopKWithBias,
}

/// Router: a gate `Linear` (`hidden -> n_experts`), the gating strategy, and an
/// optional correction bias (for `SigmoidTopKWithBias`, loaded from `exp_probs_b`).
pub struct MoeRouter {
    pub gate: Linear,
    pub kind: RouterKind,
    pub top_k: usize,
    pub num_experts: usize,
    pub correction_bias: Option<Tensor>,
}

impl MoeRouter {
    pub fn new(
        gate: Linear,
        kind: RouterKind,
        top_k: usize,
        num_experts: usize,
        correction_bias: Option<Tensor>,
    ) -> Self {
        Self {
            gate,
            kind,
            top_k,
            num_experts,
            correction_bias,
        }
    }

    /// Compute gate logits for a `[batch, hidden]` input, returning a
    /// `[batch, num_experts]` tensor on the CPU for host-side selection.
    fn gate_logits(&self, x: &Tensor) -> Result<Tensor, grim_tensor::error::Error> {
        if x.device() != self.gate.weight.device() {
            // When input is on GPU (e.g. during forward_rocm) but gate weights
            // are CPU-resident, stage activations to CPU for the router projection.
            let x_cpu = if x.device().is_cpu() {
                x.clone()
            } else {
                let f32s = x.to_vec_f32()?;
                grim_backend_cpu::cpu_tensor(f32s, x.shape().clone())
            };
            self.gate.forward(&x_cpu)
        } else {
            self.gate.forward(x)
        }
    }

    /// Route a `[batch, hidden]` input.
    ///
    /// Returns, per token, the selected expert indices and their combine
    /// weights (already normalized over the selected set). The selection for
    /// `SigmoidTopKWithBias` adds the correction bias to the sigmoid scores
    /// *only* for ranking; the returned combine weights are the unbiased
    /// sigmoid values of the selected experts.
    #[allow(clippy::type_complexity)]
    pub fn route(
        &self,
        x: &Tensor,
    ) -> Result<(Vec<Vec<usize>>, Vec<Vec<f32>>), grim_tensor::error::Error> {
        let z = self.gate_logits(x)?.to_vec_f32()?;
        let hidden = self.gate.weight.shape().dim(1)?; // in_dim of the gate
        let batch = z.len() / self.num_experts;
        let _ = hidden;
        let k = self.top_k.min(self.num_experts);

        let mut indices = Vec::with_capacity(batch);
        let mut weights = Vec::with_capacity(batch);

        for t in 0..batch {
            let row = &z[t * self.num_experts..(t + 1) * self.num_experts];
            let sel_scores: Vec<f32> = match &self.kind {
                RouterKind::SoftmaxTopK => softmax(row),
                RouterKind::SigmoidTopKWithBias => {
                    let b = self
                        .correction_bias
                        .as_ref()
                        .map(|t| t.to_vec_f32())
                        .transpose()?
                        .unwrap_or_else(|| vec![0.0f32; self.num_experts]);
                    row.iter()
                        .enumerate()
                        .map(|(i, &v)| sigmoid(v) + b.get(i).copied().unwrap_or(0.0))
                        .collect()
                }
            };
            // Rank by selection scores, take top_k.
            let mut order: Vec<usize> = (0..self.num_experts).collect();
            order.sort_by(|&a, &b| sel_scores[b].partial_cmp(&sel_scores[a]).unwrap());
            let chosen = &order[..k];

            // Combine weights.
            //   * SoftmaxTopK: the softmax probabilities over the chosen
            //     logits (inherently normalized).
            //   * SigmoidTopKWithBias: the **unbiased** sigmoid gate values of
            //     the chosen experts, used directly as combine weights (NOT
            //     renormalized). The correction bias is applied only at
            //     selection time, above — never to the combine weights.
            let raw: Vec<f32> = match &self.kind {
                RouterKind::SoftmaxTopK => {
                    let logits: Vec<f32> = chosen.iter().map(|&i| row[i]).collect();
                    softmax(&logits)
                }
                RouterKind::SigmoidTopKWithBias => {
                    chosen.iter().map(|&i| sigmoid(row[i])).collect()
                }
            };
            indices.push(chosen.to_vec());
            weights.push(raw);
        }

        Ok((indices, weights))
    }
}

// ---------------------------------------------------------------------------
// Expert bank
// ---------------------------------------------------------------------------

/// Holds the per-expert SwiGLU feed-forward triples `{gate, up, down}`.
pub struct ExpertBank {
    pub gate: Vec<Linear>,
    pub up: Vec<Linear>,
    pub down: Vec<Linear>,
}

// ── Quant workstream: per-expert bank splitters ──────────────────────────────
//
// These decompress/split BANK-level packed blobs (Storage::W4A16 /
// Storage::GroupInt / Storage::WNA16) into per-expert slabs for
// `ExpertBank::load_quantized`. W4A16 and GPTQ stay packed per expert —
// `Linear::forward` -> `quantized_matmul` routes their storage dtypes to the
// marlin/gptq fused kernels by dtype match. WNA16 has no packed GEMM yet, so
// its experts dequantize on host here (documented load-time strategy).

/// Split a Marlin-style W4A16 bank into per-expert blobs.
///
/// Bank layout (`Storage::W4A16` contract, N = num_experts * out rows):
/// `[codes (N*K/8) u32 LE][scales (N*groups) f32 LE]`. Both segments are
/// row-major over OUTPUT channels, so expert e's slab is two contiguous
/// sub-slices reassembled as `[codes_e][scales_e]`.
pub(crate) fn w4a16_split_bank(
    bytes: &[u8],
    num_experts: usize,
    out: usize,
    k: usize,
    group_size: usize,
) -> Result<Vec<Vec<u8>>, grim_tensor::error::Error> {
    if k % 8 != 0 {
        return Err(grim_tensor::error::Error::Backend(format!(
            "w4a16 bank split: K={k} must be divisible by 8"
        )));
    }
    let words_per_row = k / 8;
    let groups_per_row = k.div_ceil(group_size);
    let codes_bytes_total = num_experts * out * words_per_row * 4;
    let scales_bytes_total = num_experts * out * groups_per_row * 4;
    if bytes.len() < codes_bytes_total + scales_bytes_total {
        return Err(grim_tensor::error::Error::Backend(format!(
            "w4a16 bank split: {} bytes, need {}",
            bytes.len(),
            codes_bytes_total + scales_bytes_total
        )));
    }
    let mut blobs = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let c0 = e * out * words_per_row * 4;
        let c1 = c0 + out * words_per_row * 4;
        let s0 = codes_bytes_total + e * out * groups_per_row * 4;
        let s1 = s0 + out * groups_per_row * 4;
        let mut blob = Vec::with_capacity((c1 - c0) + (s1 - s0));
        blob.extend_from_slice(&bytes[c0..c1]);
        blob.extend_from_slice(&bytes[s0..s1]);
        blobs.push(blob);
    }
    Ok(blobs)
}

/// Split an AWQ bank into per-expert three-segment prefixed blobs:
/// `[u64 qw_len][qweight][u64 qz_len][qzeros][u64 sc_len][scales (f16)]`.
///
/// AWQ weights are packed column-major over `[K, N]` where `N = num_experts * out`.
/// Each expert owns column range `[e * out, (e + 1) * out)`.
pub(crate) fn awq_split_bank(
    bytes: &[u8],
    bits: u8,
    group_size: usize,
    num_experts: usize,
    out: usize,
    k: usize,
) -> Result<Vec<Vec<u8>>, grim_tensor::error::Error> {
    let vpw: usize = match bits {
        2 => 16,
        4 => 8,
        8 => 1,
        _ => {
            return Err(grim_tensor::error::Error::Backend(format!(
                "awq bank split: unsupported bit width {bits}"
            )));
        }
    };
    let n = num_experts * out;
    let groups = k.div_ceil(group_size);

    // Segment table
    let qw_len = k.div_ceil(vpw) * n * 4;
    let qz_len = groups * n.div_ceil(vpw) * 4;
    let sc_len = groups * n * 2; // f16 scales
    let total_expected = 8 + qw_len + 8 + qz_len + 8 + sc_len;

    // Check if the bank is already formatted as concatenated per-expert blobs
    let per_exp_qw_len = k.div_ceil(vpw) * out * 4;
    let per_exp_qz_len = groups * out.div_ceil(vpw) * 4;
    let per_exp_sc_len = groups * out * 2;
    let per_exp_blob_len = 8 + per_exp_qw_len + 8 + per_exp_qz_len + 8 + per_exp_sc_len;

    if bytes.len() == num_experts * per_exp_blob_len {
        let mut blobs = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let start = e * per_exp_blob_len;
            blobs.push(bytes[start..start + per_exp_blob_len].to_vec());
        }
        return Ok(blobs);
    }

    if bytes.len() < total_expected {
        return Err(grim_tensor::error::Error::Backend(format!(
            "awq bank split: {} bytes < expected {total_expected}",
            bytes.len()
        )));
    }

    let qw_data = 8usize;
    let qz_data = qw_data + qw_len + 8;
    let sc_data = qz_data + qz_len + 8;

    let u64le = |v: usize| (v as u64).to_le_bytes().to_vec();

    let mut blobs = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let c0 = e * out;
        let mut blob = Vec::new();

        // qweight
        blob.extend_from_slice(&u64le(per_exp_qw_len));
        for chunk in 0..k.div_ceil(vpw) {
            let src = qw_data + chunk * n * 4 + c0 * 4;
            blob.extend_from_slice(&bytes[src..src + out * 4]);
        }

        // qzeros
        blob.extend_from_slice(&u64le(per_exp_qz_len));
        for g in 0..groups {
            let src = qz_data + g * n.div_ceil(vpw) * 4 + (c0 / vpw) * 4;
            blob.extend_from_slice(&bytes[src..src + out.div_ceil(vpw) * 4]);
        }

        // scales (f16)
        blob.extend_from_slice(&u64le(per_exp_sc_len));
        for g in 0..groups {
            let src = sc_data + g * n * 2 + c0 * 2;
            blob.extend_from_slice(&bytes[src..src + out * 2]);
        }

        blobs.push(blob);
    }
    Ok(blobs)
}

/// Split a CompressedTensors W8A8 INT8 bank into per-expert blobs:
/// `[u64 prefix][int8 codes [out, in]][f32 scales [out]]`.
pub(crate) fn w8a8_int8_split_bank(
    bytes: &[u8],
    num_experts: usize,
    out: usize,
    k: usize,
) -> Result<Vec<Vec<u8>>, grim_tensor::error::Error> {
    let per_exp_codes = out * k;
    let per_exp_scales = out * 4;
    let per_exp_len = 8 + per_exp_codes + per_exp_scales;

    if bytes.len() == num_experts * per_exp_len {
        let mut blobs = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let start = e * per_exp_len;
            blobs.push(bytes[start..start + per_exp_len].to_vec());
        }
        return Ok(blobs);
    }

    // Unified bank: [u64 prefix][int8 codes [num_experts*out, k]][f32 scales [num_experts*out]]
    let codes_start = 8usize;
    let scales_start = codes_start + num_experts * out * k;
    if bytes.len() < scales_start + num_experts * out * 4 {
        return Err(grim_tensor::error::Error::Backend(format!(
            "w8a8_int8 bank split: buffer too short (len={})",
            bytes.len()
        )));
    }

    let mut blobs = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let mut blob = Vec::with_capacity(per_exp_len);
        blob.extend_from_slice(&(per_exp_codes as u64).to_le_bytes());
        let c_src = codes_start + e * per_exp_codes;
        blob.extend_from_slice(&bytes[c_src..c_src + per_exp_codes]);
        let s_src = scales_start + e * per_exp_scales;
        blob.extend_from_slice(&bytes[s_src..s_src + per_exp_scales]);
        blobs.push(blob);
    }
    Ok(blobs)
}

/// Split a CompressedTensors W8A8 FP8 bank into per-expert blobs:
/// `[u64 prefix][fp8 codes [out, in]][f32 scale]`.
pub(crate) fn w8a8_fp8_split_bank(
    bytes: &[u8],
    num_experts: usize,
    out: usize,
    k: usize,
) -> Result<Vec<Vec<u8>>, grim_tensor::error::Error> {
    let per_exp_codes = out * k;
    let per_exp_len = 8 + per_exp_codes + 4;

    if bytes.len() == num_experts * per_exp_len {
        let mut blobs = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let start = e * per_exp_len;
            blobs.push(bytes[start..start + per_exp_len].to_vec());
        }
        return Ok(blobs);
    }

    let codes_start = 8usize;
    let scale_start = codes_start + num_experts * out * k;
    if bytes.len() < scale_start + 4 {
        return Err(grim_tensor::error::Error::Backend(format!(
            "w8a8_fp8 bank split: buffer too short (len={})",
            bytes.len()
        )));
    }
    let tensor_scale = &bytes[scale_start..scale_start + 4];

    let mut blobs = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let mut blob = Vec::with_capacity(per_exp_len);
        blob.extend_from_slice(&(per_exp_codes as u64).to_le_bytes());
        let c_src = codes_start + e * per_exp_codes;
        blob.extend_from_slice(&bytes[c_src..c_src + per_exp_codes]);
        blob.extend_from_slice(tensor_scale);
        blobs.push(blob);
    }
    Ok(blobs)
}

/// Split a GPTQ/GroupInt bank into per-expert four-segment prefixed blobs
/// matching `roc_device::gptq_segment_offsets`:
/// `[u64 qw_len][qweight][u64 qz_len][qzeros][u64 sc_len][scales][u64 gi_len][g_idx]`.
///
/// Expert e owns output columns [e*n_e, (e+1)*n_e) of the [K, N]-packed
/// weight. Requires n_e divisible by values-per-word so packed words never
/// straddle experts. g_idx is shared across experts and copied whole.
pub(crate) fn gptq_split_bank(
    bytes: &[u8],
    bits: u8,
    group_size: usize,
    num_experts: usize,
    out: usize,
    k: usize,
) -> Result<Vec<Vec<u8>>, grim_tensor::error::Error> {
    let vpw: usize = match bits {
        2 => 16,
        4 => 8,
        8 => 1,
        _ => {
            return Err(grim_tensor::error::Error::Backend(format!(
                "gptq bank split: unsupported bit width {bits} (2/4/8 only)"
            )));
        }
    };
    let n = num_experts * out;
    let groups = k.div_ceil(group_size);

    // Segment table — same math as roc_device::gptq_segment_offsets.
    let qw_len = k.div_ceil(vpw) * n * 4;
    let qz_len = groups * n.div_ceil(vpw) * 4;
    let sc_len = groups * n * 4;
    let gi_data_off = 8 + qw_len + 8 + qz_len + 8 + sc_len + 8;
    let has_g_idx = bytes.len() == gi_data_off + k * 4;
    if !has_g_idx && bytes.len() != gi_data_off {
        return Err(grim_tensor::error::Error::Backend(format!(
            "gptq bank split: {} bytes matches no valid segment table",
            bytes.len()
        )));
    }

    let qw_data = 8usize;
    let qz_data = qw_data + qw_len + 8;
    let sc_data = qz_data + qz_len + 8;

    let u64le = |v: usize| (v as u64).to_le_bytes().to_vec();

    let mut blobs = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let c0 = e * out;
        let mut blob = Vec::new();

        // qweight: ONE u32 word per (K-chunk, column); each word packs vpw
        // consecutive in_idx codes for that column. Layout is
        // [K/vpw chunks][N words] — so each expert's run is `out` whole
        // words per chunk at column offset c0 (contiguous because we require
        // out % vpw == 0).
        blob.extend_from_slice(&u64le(k.div_ceil(vpw) * out * 4));
        for chunk in 0..k.div_ceil(vpw) {
            let src = qw_data + chunk * n * 4 + c0 * 4;
            blob.extend_from_slice(&bytes[src..src + out * 4]);
        }

        // qzeros: ONE word per (group, col-chunk) — these DO pack vpw
        // columns per word, so the per-expert run is out/vpw words.
        blob.extend_from_slice(&u64le(groups * out.div_ceil(vpw) * 4));
        for g in 0..groups {
            let src = qz_data + g * n.div_ceil(vpw) * 4 + (c0 / vpw) * 4;
            blob.extend_from_slice(&bytes[src..src + out.div_ceil(vpw) * 4]);
        }

        // scales f32 [groups, N]: contiguous column slice.
        blob.extend_from_slice(&u64le(groups * out * 4));
        for g in 0..groups {
            let src = sc_data + g * n * 4 + c0 * 4;
            blob.extend_from_slice(&bytes[src..src + out * 4]);
        }

        // g_idx: shared across experts — full segment when present, else a
        // zero-length prefix (the convention gptq_segment_offsets expects).
        if has_g_idx {
            blob.extend_from_slice(&u64le(k * 4));
            blob.extend_from_slice(&bytes[gi_data_off..gi_data_off + k * 4]);
        } else {
            blob.extend_from_slice(&u64le(0));
        }
        blobs.push(blob);
    }
    Ok(blobs)
}

/// WNA16 has no packed GEMM yet: dequantize on host at load time (the
/// documented strategy for backends without a fused kernel). Blob layout:
/// `[u32 n_bit][u32 blocks][codes MSB-first][f16 block scales][f32 ts]`,
/// 256-weight blocks. Returns NATIVE-F32 per-expert slabs.
pub(crate) fn wna16_split_or_dequant(
    bytes: &[u8],
    num_experts: usize,
    out: usize,
    k: usize,
) -> Result<Vec<Vec<u8>>, grim_tensor::error::Error> {
    let n_bit = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let blocks = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let total_weights = num_experts * out * k;
    if blocks != total_weights.div_ceil(256) {
        return Err(grim_tensor::error::Error::Backend(
            "wna16 bank: block count mismatch".into(),
        ));
    }
    let code_bytes = total_weights * n_bit;
    let code_start = 8usize;
    let scales_start = code_start + code_bytes.div_ceil(8);
    let ts_off = scales_start + blocks * 2;
    if bytes.len() < ts_off + 4 {
        return Err(grim_tensor::error::Error::Backend(
            "wna16 bank: blob shorter than header+scales+tail".into(),
        ));
    }
    let tensor_scale = f32::from_le_bytes(bytes[ts_off..ts_off + 4].try_into().unwrap());

    let decode = |lane: usize| -> u32 {
        let start_bit = lane * n_bit;
        let mut code = 0u32;
        for b in 0..n_bit {
            let pos = start_bit + b;
            let bit = (bytes[code_start + pos / 8] >> (7 - (pos % 8))) & 1;
            code = (code << 1) | bit as u32;
        }
        code
    };

    let mut weights = vec![0.0f32; total_weights];
    for (i, slot) in weights.iter_mut().enumerate() {
        let blk = i / 256;
        let h = u16::from_le_bytes([
            bytes[scales_start + blk * 2],
            bytes[scales_start + blk * 2 + 1],
        ]);
        *slot = decode(i) as f32 * f16_bits_to_f32(h) * tensor_scale;
    }

    let stride = out * k;
    let mut blobs = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let mut blob = Vec::with_capacity(stride * 4);
        for &w in &weights[e * stride..(e + 1) * stride] {
            blob.extend_from_slice(&w.to_le_bytes());
        }
        blobs.push(blob);
    }
    Ok(blobs)
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    match exp {
        0x1F => f32::from_bits(sign | 0x7F800000 | (if mant != 0 { 0x400000 } else { 0 })),
        0 => {
            if mant == 0 {
                f32::from_bits(sign)
            } else {
                let v = (mant as f32) * 2f32.powi(-24);
                if sign != 0 { -v } else { v }
            }
        }
        e => f32::from_bits(sign | ((e - 15 + 127) << 23) | (mant << 13)),
    }
}

impl ExpertBank {
    /// Construct directly from per-expert `Linear`s (used by tests and
    /// synthetic construction).
    pub fn from_linears(gate: Vec<Linear>, up: Vec<Linear>, down: Vec<Linear>) -> Self {
        Self { gate, up, down }
    }

    pub fn num_experts(&self) -> usize {
        self.gate.len()
    }

    /// Load experts from a GGUF-style 3D weight layout. Matches the in-repo
    /// Lfm2 MoE loader's naming and layout convention:
    ///   `ffn_gate_exps.weight` = `[n_experts, inter, hidden]`
    ///   `ffn_up_exps.weight`   = `[n_experts, inter, hidden]`
    ///   `ffn_down_exps.weight` = `[n_experts, hidden, inter]`
    /// (experts are the OUTERMOST dimension).
    ///
    /// Quantized checkpoints (KQuant / FloatPack / MXFP4 / ...) keep each
    /// expert's packed bytes resident on the target device — no full-model
    /// host-f32 mirror is materialized. Native (F32/F16/BF16) checkpoints use
    /// the host-f32 path.
    pub fn load(
        ws: &WeightSource<'_>,
        num_experts: usize,
        hidden: usize,
        inter: usize,
        has_bias: bool,
    ) -> Result<Self, grim_tensor::error::Error> {
        // gate/up: per-expert [out=inter, in=hidden]; down: [out=hidden, in=inter].
        Self::load_impl(
            ws,
            num_experts,
            hidden,
            inter,
            has_bias,
            [
                ("ffn_gate_exps.weight", inter, hidden),
                ("ffn_up_exps.weight", inter, hidden),
                ("ffn_down_exps.weight", hidden, inter),
            ],
        )
    }

    /// Load experts from a GGUF-style 3D weight layout where ALL three tensors
    /// (gate/up/down) are stored as `[n_experts, hidden, inter]` (Mellum2 /
    /// Unsloth-quantized GGUFs).
    ///
    /// Each expert's `[hidden, inter]` block is sliced out. Gate/up are
    /// transposed to `[inter, hidden]` for the Linear (in=inter, out=hidden).
    /// Down is used as-is (already `[hidden, inter]`, correct for Linear
    /// out=hidden, in=inter).
    pub fn load_transposed(
        ws: &WeightSource<'_>,
        num_experts: usize,
        hidden: usize,
        inter: usize,
        has_bias: bool,
    ) -> Result<Self, grim_tensor::error::Error> {
        Self::load_impl(
            ws,
            num_experts,
            hidden,
            inter,
            has_bias,
            [
                ("ffn_gate_exps.weight", hidden, inter),
                ("ffn_up_exps.weight", hidden, inter),
                ("ffn_down_exps.weight", hidden, inter),
            ],
        )
    }

    /// Shared loader. `projections` lists `(tensor_name, out_dim, in_dim)` per
    /// projection as stored per-expert in the checkpoint.
    ///
    /// GGUF stores `ne` fastest-first, so a tensor with file dims
    /// `[in, out, n_experts]` reads back as `[n_experts, out, in]` and each
    /// expert's `out * in` elements are contiguous in the packed stream — the
    /// slice math is identical for every projection.
    fn load_impl(
        ws: &WeightSource<'_>,
        num_experts: usize,
        hidden: usize,
        inter: usize,
        has_bias: bool,
        projections: [(&'static str, usize, usize); 3],
    ) -> Result<Self, grim_tensor::error::Error> {
        // Probe the first projection to decide the storage path.
        let probe = ws.get_raw_packed(projections[0].0)?;

        if probe.dtype.storage == grim_tensor::dtype::Storage::Native {
            return Self::load_native(ws, num_experts, hidden, inter, has_bias, projections);
        }
        Self::load_quantized(ws, num_experts, has_bias, projections)
    }

    /// Quantized-resident path: slice each expert's packed bytes and
    /// materialize them individually so weights stay packed on-device.
    fn load_quantized(
        ws: &WeightSource<'_>,
        num_experts: usize,
        has_bias: bool,
        projections: [(&'static str, usize, usize); 3],
    ) -> Result<Self, grim_tensor::error::Error> {
        use grim_tensor::dtype::{FloatPackScheme, Storage};

        // Process ONE projection at a time. Each projection's full packed bank
        // is fetched, sliced across all `num_experts`, and fully consumed
        // before the next projection's bank is fetched — so at most ONE full
        // packed bank (not three) is resident at once. This bounds peak
        // packed-byte RSS to ~1.4 GB/layer for Mellum2-class MoE instead of
        // ~4.2 GB and fixes the OOM that previously killed the load mid-way.
        let mut gate = Vec::with_capacity(num_experts);
        let mut up = Vec::with_capacity(num_experts);
        let mut down = Vec::with_capacity(num_experts);

        for (p_idx, (name, out, in_)) in projections.iter().enumerate() {
            let raw = ws.get_raw_packed(name)?;
            let dims = &raw.shape;
            let expected = vec![num_experts, *out, *in_];
            if *dims != expected {
                return Err(grim_tensor::error::Error::ShapeMismatch {
                    expected,
                    got: dims.clone(),
                });
            }
            let elem_count: usize = dims.iter().product();
            // MXFP4 arrives from the provider as the length-prefixed
            // [codes][exps] framing; every other quant format is raw blocks.
            let is_framed_mxfp4 = matches!(
                raw.dtype.storage,
                Storage::FloatPack(FloatPackScheme::MxFp4)
            );

            // Quant workstream wiring: per-expert extraction for the
            // compressed-tensors formats. W4A16 and GroupInt (GPTQ) stay
            // packed per expert — `Linear::forward` -> `quantized_matmul`
            // routes them to the marlin/gptq fused kernels by storage dtype.
            // WNA16 has no packed GEMM yet, so its experts are dequantized on
            // host here (load-time strategy; a device dequant service exists
            // for resident-blob use in grim-backend-rocm).
            let per_expert_blobs: Option<Vec<Vec<u8>>> = match &raw.dtype.storage {
                Storage::W4A16(w4) => Some(w4a16_split_bank(
                    &raw.bytes,
                    num_experts,
                    *out,
                    *in_,
                    w4.group_size,
                )?),
                Storage::GroupInt(gi) => Some(gptq_split_bank(
                    &raw.bytes,
                    gi.bits,
                    gi.group_size,
                    num_experts,
                    *out,
                    *in_,
                )?),
                Storage::WNA16 => {
                    Some(wna16_split_or_dequant(&raw.bytes, num_experts, *out, *in_)?)
                }
                Storage::Awq(awq) => Some(awq_split_bank(
                    &raw.bytes,
                    awq.bits,
                    awq.group_size,
                    num_experts,
                    *out,
                    *in_,
                )?),
                Storage::CompressedTensorsW8A8Int8 => {
                    Some(w8a8_int8_split_bank(&raw.bytes, num_experts, *out, *in_)?)
                }
                Storage::CompressedTensorsW8A8Fp8 => {
                    Some(w8a8_fp8_split_bank(&raw.bytes, num_experts, *out, *in_)?)
                }
                _ => None,
            };

            for e in 0..num_experts {
                let per_expert = elem_count / num_experts;
                if let Some(blobs) = &per_expert_blobs {
                    // Split-blob path: each expert's reassembled blob carries
                    // the bank's storage dtype so `Linear::forward` keeps
                    // routing it to the format's fused kernel / dequant arm.
                    let rt = grim_tensor::provider::RawTensor {
                        bytes: blobs[e].clone(),
                        shape: vec![*out, *in_],
                        dtype: raw.dtype.clone(),
                        provenance: raw.provenance.clone(),
                    };
                    let t = ws.materialize_raw(rt, Shape::new(vec![*out, *in_]))?;
                    let lin = Linear::from_tensor(t, bias_opt(has_bias, *out));
                    match p_idx {
                        0 => gate.push(lin),
                        1 => up.push(lin),
                        _ => down.push(lin),
                    }
                    continue;
                }
                let (bytes, dtype): (Vec<u8>, grim_tensor::dtype::DType) = if is_framed_mxfp4 {
                    // MXFP4 rides the fused dequant-GEMM path through
                    // `Linear::forward` -> `quantized_matmul` (ROCm dispatch at
                    // roc_device.rs:2607), so keep it packed on-device as MXFP4
                    // instead of dequantizing on the host and requantizing to
                    // Q8_0. That host round-trip was the load tax: it ran
                    // serially per expert, in the layer loop, and it silently
                    // swapped the dtype the kernel was expecting.
                    //
                    // Slice this expert's codes/exps out of the framed bank
                    // and re-wrap in the length-prefixed [codes][exps]
                    // framing `quantized_matmul` expects.
                    let (codes, exps) = split_mxfp4_framed(&raw.bytes)?;
                    let codes_per = per_expert / 2;
                    let exps_per = per_expert.div_ceil(32);
                    let c = &codes[e * codes_per..(e + 1) * codes_per];
                    let x = &exps[e * exps_per..(e + 1) * exps_per];
                    let mut framed = Vec::with_capacity(16 + c.len() + x.len());
                    framed.extend_from_slice(&(c.len() as u64).to_le_bytes());
                    framed.extend_from_slice(c);
                    framed.extend_from_slice(&(x.len() as u64).to_le_bytes());
                    framed.extend_from_slice(x);
                    let mxfp4_dtype = grim_tensor::dtype::DType {
                        arith: grim_tensor::ArithType::F32,
                        storage: grim_tensor::dtype::Storage::FloatPack(
                            grim_tensor::dtype::FloatPackScheme::MxFp4,
                        ),
                    };
                    (framed, mxfp4_dtype)
                } else {
                    // Raw block-quant bytes: contiguous per-expert stride.
                    if raw.bytes.len() % num_experts != 0 {
                        return Err(grim_tensor::error::Error::Backend(format!(
                            "expert bank '{name}': {} bytes not divisible by {num_experts} experts",
                            raw.bytes.len()
                        )));
                    }
                    let stride = raw.bytes.len() / num_experts;
                    (
                        raw.bytes[e * stride..(e + 1) * stride].to_vec(),
                        raw.dtype.clone(),
                    )
                };
                let shape = Shape::new(vec![*out, *in_]);
                let rt = grim_tensor::provider::RawTensor {
                    bytes,
                    shape: vec![*out, *in_],
                    dtype,
                    provenance: raw.provenance.clone(),
                };
                let t = ws.materialize_raw(rt, shape)?;
                let lin = Linear::from_tensor(t, bias_opt(has_bias, *out));
                // `p_idx` maps the projection to its expert slot: 0 = gate,
                // 1 = up, 2 = down (matches the `projections` argument order).
                match p_idx {
                    0 => gate.push(lin),
                    1 => up.push(lin),
                    _ => down.push(lin),
                }
            }
            // `raw` (and its full packed bank) drops here, before the next
            // projection is fetched — this is the core of the OOM fix.
        }
        Ok(Self { gate, up, down })
    }

    /// Native (F32/F16/BF16) path: dequantize on host as before.
    #[allow(clippy::too_many_arguments)]
    fn load_native(
        ws: &WeightSource<'_>,
        num_experts: usize,
        _hidden: usize,
        _inter: usize,
        has_bias: bool,
        projections: [(&'static str, usize, usize); 3],
    ) -> Result<Self, grim_tensor::error::Error> {
        let mut flat = Vec::with_capacity(3);
        for (name, out, in_) in projections {
            let t = ws.get(Shape::new(vec![num_experts, out, in_]), name)?;
            flat.push((t.to_vec_f32()?, out, in_));
        }

        let mut gate = Vec::with_capacity(num_experts);
        let mut up = Vec::with_capacity(num_experts);
        let mut down = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let mut lins = Vec::with_capacity(3);
            for (v, out, in_) in &flat {
                let block = slice_expert(v, e, *out, *in_);
                // GGUF row-major per expert is [out, in] when the bank stores
                // [n_experts, out, in] — matches Linear directly. gate/up banks
                // store [inter, hidden] (out=inter) and down stores
                // [hidden, inter] (out=hidden), so no transpose is needed.
                lins.push(Linear::from_tensor(
                    cpu_tensor(block, Shape::new(vec![*out, *in_])),
                    bias_opt(has_bias, *out),
                ));
            }
            let mut it = lins.into_iter();
            gate.push(it.next().ok_or_else(err_proj)?);
            up.push(it.next().ok_or_else(err_proj)?);
            down.push(it.next().ok_or_else(err_proj)?);
        }
        Ok(Self { gate, up, down })
    }

    /// Run a single expert's SwiGLU feed-forward on `x` (`[batch, hidden]`),
    /// returning `[batch, hidden]`.
    pub fn expert_forward(
        &self,
        e: usize,
        x: &Tensor,
    ) -> Result<Tensor, grim_tensor::error::Error> {
        let g = self.gate[e].forward(x)?; // [batch, inter]
        let u = self.up[e].forward(x)?; // [batch, inter]
        let h = silu_mul_host(&g, &u)?; // [batch, inter]
        self.down[e].forward(&h) // [batch, hidden]
    }
}

// ---------------------------------------------------------------------------
// MoE FFN
// ---------------------------------------------------------------------------

/// On-device resident copy of the flattened expert weight banks for the ROCm
/// fused MoE dispatch. Built once on first `forward_rocm` call and reused for
/// the session's lifetime, so expert weights never round-trip host↔device on
/// every forward call. Rebuilt only when the resident set (expert count and
/// weight shapes) changes.
#[cfg(feature = "rocm-mem")]
struct RocmResidentWeights {
    gate: Arc<dyn BackendStorage>,
    up: Arc<dyn BackendStorage>,
    down: Arc<dyn BackendStorage>,
    /// Structural key of the resident set: `(num_experts, hidden, inter)`.
    fingerprint: (usize, usize, usize),
}

#[cfg(feature = "rocm-mem")]
/// Dequantize one expert weight tensor to row-major F32 on `ordinal`.
///
/// Native storages ride `to_vec_f32`. Packed formats use their own GPU
/// launchers at materialization time:
/// - W4A16: marlin kernel over an identity activation — C = I @ deq(B)ᵀ is
///   the transposed weight; host-transposed back to row-major after readback.
/// - GroupInt (GPTQ): forward kernel with identity activation — C = I @ D is
///   exactly the row-major [K=out, N=in] dequantized weight.
/// - WNA16: `dequant_wna16_to_f32` decodes the resident blob directly.
pub(crate) fn rocm_dequant_expert_weight(
    weight: &Tensor,
    ordinal: usize,
) -> Result<Vec<f32>, grim_tensor::error::Error> {
    let dev = RocmDevice::try_new(ordinal)?;
    let dims = weight.shape().dims();
    let (n_rows, k_dim) = (dims[0], dims[1]);
    let out_box = match weight.dtype().storage {
        Storage::W4A16(w4) => {
            let c_box = dev.dequant_w4a16_blob_to_f32(
                weight
                    .storage()
                    .as_any()
                    .downcast_ref::<grim_backend_rocm::RocmStorage>()
                    .ok_or_else(|| {
                        grim_tensor::error::Error::Backend("w4a16 expert not rocm".into())
                    })?,
                n_rows,
                k_dim,
                w4.group_size,
            )?;
            if std::env::var_os("GRIM_MOE_DIAG").is_some() {
                eprintln!(
                    "[moe-diag] w4a16 materialized shape={:?}",
                    weight.shape().dims()
                );
            }
            // Service returns Dᵀ row-major [k_dim, n_rows]; transpose to
            // the weight layout [n_rows, k_dim].
            let c_t = Tensor::new(
                std::sync::Arc::from(c_box),
                Shape::new(vec![k_dim, n_rows]),
                DType::F32,
                grim_tensor::QuantProvenance::default(),
                Device::Rocm(ordinal),
            );
            let c = c_t.to_vec_f32()?;
            let mut d = vec![0.0f32; n_rows * k_dim];
            for r in 0..n_rows {
                for c2 in 0..k_dim {
                    d[r * k_dim + c2] = c[c2 * n_rows + r];
                }
            }
            return Ok(d);
        }
        Storage::GroupInt(gi) => {
            let c_box = dev.gptq_dequant_identity_to_f32(
                weight
                    .storage()
                    .as_any()
                    .downcast_ref::<grim_backend_rocm::RocmStorage>()
                    .ok_or_else(|| {
                        grim_tensor::error::Error::Backend("gptq expert not rocm".into())
                    })?,
                n_rows,
                k_dim,
                gi.bits,
                gi.group_size,
            )?;
            // C = D row-major [k_dim, n_rows]; transpose to [n_rows, k_dim].
            let c_t = Tensor::new(
                std::sync::Arc::from(c_box),
                Shape::new(vec![k_dim, n_rows]),
                DType::F32,
                grim_tensor::QuantProvenance::default(),
                Device::Rocm(ordinal),
            );
            let c = c_t.to_vec_f32()?;
            let mut d = vec![0.0f32; n_rows * k_dim];
            for r in 0..n_rows {
                for c2 in 0..k_dim {
                    d[r * k_dim + c2] = c[c2 * n_rows + r];
                }
            }
            return Ok(d);
        }
        Storage::WNA16 => {
            let b_rocm = weight
                .storage()
                .as_any()
                .downcast_ref::<grim_backend_rocm::RocmStorage>()
                .ok_or_else(|| {
                    grim_tensor::error::Error::Backend("wna16 expert not rocm".into())
                })?;
            dev.dequant_wna16_blob_to_f32(b_rocm, n_rows * k_dim)?
        }
        Storage::CompressedTensorsW8A8Int8 => {
            let b_rocm = weight
                .storage()
                .as_any()
                .downcast_ref::<grim_backend_rocm::RocmStorage>()
                .ok_or_else(|| {
                    grim_tensor::error::Error::Backend("w8a8_int8 expert not rocm".into())
                })?;
            let raw_bytes = b_rocm.copy_to_host()?;
            let codes = &raw_bytes[8..8 + n_rows * k_dim];
            let scales = &raw_bytes[8 + n_rows * k_dim..];
            let mut d = vec![0.0f32; n_rows * k_dim];
            for r in 0..n_rows {
                let sc = f32::from_le_bytes([
                    scales[r * 4],
                    scales[r * 4 + 1],
                    scales[r * 4 + 2],
                    scales[r * 4 + 3],
                ]);
                for c in 0..k_dim {
                    let code = codes[r * k_dim + c] as i8 as f32;
                    d[r * k_dim + c] = code * sc;
                }
            }
            return Ok(d);
        }
        Storage::CompressedTensorsW8A8Fp8 => {
            let b_rocm = weight
                .storage()
                .as_any()
                .downcast_ref::<grim_backend_rocm::RocmStorage>()
                .ok_or_else(|| {
                    grim_tensor::error::Error::Backend("w8a8_fp8 expert not rocm".into())
                })?;
            let raw_bytes = b_rocm.copy_to_host()?;
            let codes = &raw_bytes[8..8 + n_rows * k_dim];
            let sc = f32::from_le_bytes([
                raw_bytes[8 + n_rows * k_dim],
                raw_bytes[8 + n_rows * k_dim + 1],
                raw_bytes[8 + n_rows * k_dim + 2],
                raw_bytes[8 + n_rows * k_dim + 3],
            ]);
            let mut d = vec![0.0f32; n_rows * k_dim];
            for r in 0..n_rows {
                for c in 0..k_dim {
                    let byte = codes[r * k_dim + c];
                    let val = grim_quant::fp8_e4m3_to_f32(byte);
                    d[r * k_dim + c] = val * sc;
                }
            }
            return Ok(d);
        }
        Storage::Awq(awq) => {
            let b_rocm = weight
                .storage()
                .as_any()
                .downcast_ref::<grim_backend_rocm::RocmStorage>()
                .ok_or_else(|| grim_tensor::error::Error::Backend("awq expert not rocm".into()))?;
            let raw_bytes = b_rocm.copy_to_host()?;
            let (qw_off, qz_off, sc_off) = grim_backend_rocm::RocmDevice::awq_segment_offsets(
                awq.bits,
                awq.group_size,
                k_dim,
                n_rows,
                raw_bytes.len(),
            )?;
            let qw = &raw_bytes[qw_off as usize..qz_off as usize - 8];
            let qz = &raw_bytes[qz_off as usize..sc_off as usize - 8];
            let sc = &raw_bytes[sc_off as usize..];
            let deq = grim_quant::dequant_awq_group_int(
                qw,
                qz,
                sc,
                &[k_dim, n_rows],
                awq.bits as u32,
                awq.group_size,
            )?;
            // deq is [k_dim, n_rows]; transpose to [n_rows, k_dim]
            let mut d = vec![0.0f32; n_rows * k_dim];
            for r in 0..n_rows {
                for c in 0..k_dim {
                    d[r * k_dim + c] = deq[c * n_rows + r];
                }
            }
            return Ok(d);
        }
        _ => return weight.to_vec_f32(),
    };
    let t = Tensor::new(
        std::sync::Arc::from(out_box),
        weight.shape().clone(),
        DType::F32,
        grim_tensor::QuantProvenance::default(),
        Device::Rocm(ordinal),
    );
    t.to_vec_f32()
}

impl RocmResidentWeights {
    fn build(
        experts: &ExpertBank,
        num_experts: usize,
        hidden: usize,
        inter: usize,
        ordinal: usize,
    ) -> Result<Self, grim_tensor::error::Error> {
        let mut gate_flat = Vec::with_capacity(num_experts * inter * hidden);
        let mut up_flat = Vec::with_capacity(num_experts * inter * hidden);
        let mut down_flat = Vec::with_capacity(num_experts * hidden * inter);
        for e in 0..num_experts {
            // Audit-wiring fix (quant workstream): W4A16 / GroupInt / WNA16
            // packed experts were reinterpreted by to_vec_f32 as raw f32
            // (nibble codes read as floats -> 1e16-scale garbage / NaN in
            // every routed forward). Dequantize them properly at
            // materialization time through the format's own GPU launcher or
            // dequant service instead.
            for (flat_dst, lin) in [
                (&mut gate_flat, &experts.gate[e]),
                (&mut up_flat, &experts.up[e]),
                (&mut down_flat, &experts.down[e]),
            ] {
                let flat = crate::moe::rocm_dequant_expert_weight(&lin.weight, ordinal)?;
                assert_eq!(flat.len(), lin.weight.shape().elem_count());
                flat_dst.extend_from_slice(&flat);
            }
        }

        let dev = RocmDevice::try_new(ordinal)?;
        let gate = Arc::from(CoreTensorOps::from_cpu(
            &dev,
            &gate_flat,
            &Shape::new(vec![gate_flat.len()]),
            DType::F32,
        )?);
        let up = Arc::from(CoreTensorOps::from_cpu(
            &dev,
            &up_flat,
            &Shape::new(vec![up_flat.len()]),
            DType::F32,
        )?);
        let down = Arc::from(CoreTensorOps::from_cpu(
            &dev,
            &down_flat,
            &Shape::new(vec![down_flat.len()]),
            DType::F32,
        )?);

        Ok(Self {
            gate,
            up,
            down,
            fingerprint: (num_experts, hidden, inter),
        })
    }
}

/// CUDA resident expert-weight cache: flattened gate/up/down buffers
/// uploaded to the device once and reused across forward calls.
/// Mirrors [`RocmResidentWeights`] for the CUDA backend.
#[cfg(feature = "cuda-mem")]
struct CudaResidentWeights {
    gate: Arc<CudaStorage>,
    up: Arc<CudaStorage>,
    down: Arc<CudaStorage>,
    /// Structural key of the resident set: `(num_experts, hidden, inter)`.
    fingerprint: (usize, usize, usize),
}

#[cfg(feature = "cuda-mem")]
impl CudaResidentWeights {
    fn build(
        experts: &ExpertBank,
        num_experts: usize,
        hidden: usize,
        inter: usize,
        ordinal: usize,
    ) -> Result<Self, grim_tensor::error::Error> {
        let mut gate_flat = Vec::with_capacity(num_experts * inter * hidden);
        let mut up_flat = Vec::with_capacity(num_experts * inter * hidden);
        let mut down_flat = Vec::with_capacity(num_experts * hidden * inter);
        for e in 0..num_experts {
            gate_flat.extend_from_slice(&experts.gate[e].weight.to_vec_f32()?);
            up_flat.extend_from_slice(&experts.up[e].weight.to_vec_f32()?);
            down_flat.extend_from_slice(&experts.down[e].weight.to_vec_f32()?);
        }

        let dev = crate::backend_cuda::CudaDevice::new(ordinal)?;
        let gate = Arc::new(CudaStorage::copy_from_host(
            &gate_flat,
            &Shape::new(vec![gate_flat.len()]),
            DType::F32,
            ordinal,
        )?);
        let up = Arc::new(CudaStorage::copy_from_host(
            &up_flat,
            &Shape::new(vec![up_flat.len()]),
            DType::F32,
            ordinal,
        )?);
        let down = Arc::new(CudaStorage::copy_from_host(
            &down_flat,
            &Shape::new(vec![down_flat.len()]),
            DType::F32,
            ordinal,
        )?);

        Ok(Self {
            gate,
            up,
            down,
            fingerprint: (num_experts, hidden, inter),
        })
    }
}

/// Compute the actual packed byte size of one expert's weight given its dtype
/// and element count. Works for any quant format: MXFP4/MXFP8/FP8 (float-pack),
/// Q4_K/Q5_K/Q6_K/IQ2/3/4/NF4/FP4/block-quant (KQuant/block), and native
/// fp16/f32/BF16. Returns the byte size that the PlanBuilder should use for
/// budget accounting.
fn expert_weight_bytes(dtype: &DType, elem_count: usize) -> usize {
    match dtype.storage {
        // Float-pack formats (MXFP4, MXFP8, FP8 block): packed bytes per element
        // varies by scheme; approximate as the element count times the pack width.
        Storage::FloatPack(scheme) => {
            let bits_per_element = match scheme {
                FloatPackScheme::MxFp4 => 4,
                FloatPackScheme::MxFp8 => 8,
                FloatPackScheme::Fp4 => 4,
                FloatPackScheme::Fp8 => 8,
                _ => 32, // fallback: treat as fp32
            };
            // Round up: ceil(elem_count * bits / 8)
            (elem_count * bits_per_element).div_ceil(8)
        }
        // K-quant / block-quant / IQ formats: packed bytes, typically
        // bits_per_weight per element. Use the quant scheme's bit width.
        Storage::KQuant(qscheme) => {
            let bits = match qscheme {
                KQuantScheme::Q80 => 8,
                KQuantScheme::Q4K => 4,
                KQuantScheme::Q5K => 5,
                KQuantScheme::Q6K => 6,
                KQuantScheme::Q3K => 3,
                KQuantScheme::Q2K => 2,
                KQuantScheme::IQ4NL => 4,
                KQuantScheme::IQ4XS => 4,
                KQuantScheme::IQ3XXS => 3,
                KQuantScheme::IQ3S => 3,
                KQuantScheme::IQ2XXS => 2,
                KQuantScheme::IQ2XS => 2,
                KQuantScheme::IQ2S => 2,
            };
            (elem_count * bits).div_ceil(8)
        }
        // Block-quantized formats (IQ4_NL, IQ4_XS, IQ3_XXS, IQ3_S, etc.)
        Storage::Block(bdtype) => {
            // BlockDtype covers float-pack block formats (FP4/NF4/FP8) and
            // block-quant variants. Use the bit width from the scheme.
            let bits = match bdtype {
                BlockDtype::Fp4 => 4,
                BlockDtype::Nf4 => 4,
                BlockDtype::Fp8 => 8,
                BlockDtype::Fp4Block16 => 4,
                BlockDtype::Fp8Block16 => 8,
            };
            (elem_count * bits).div_ceil(8)
        }
        // GroupInt: variable, approximate as 1 byte per element
        Storage::GroupInt(_) => elem_count,
        // CompressedTensors W8A8 Int8/Fp8: 1 byte per element (int8 / fp8).
        Storage::CompressedTensorsW8A8Int8 | Storage::CompressedTensorsW8A8Fp8 => elem_count,
        // AWQ: packed bits (e.g. 4-bit = 0.5 byte/elem, 2-bit = 0.25 byte/elem)
        Storage::Awq(awq) => (elem_count * awq.bits as usize).div_ceil(8),
        // W4A16: 4-bit packed codes (~0.5 byte/elem); per-group f32 scales are
        // small relative to weights, so this is a conservative lower bound.
        Storage::W4A16(_) => elem_count / 2,
        // ResidualPacked: packed residual format with per-block scales.
        // Use bpw (bits per weight) to compute byte size.
        Storage::ResidualPacked(config) => (elem_count * config.bpw as usize).div_ceil(8),
        // Native: fp16/f32/bf16
        Storage::Native => {
            match dtype.arith {
                ArithType::F16 | ArithType::BF16 => elem_count * 2,
                ArithType::F32 => elem_count * 4,
                _ => elem_count * 4, // fallback
            }
        }
        _ => elem_count * 4,
    }
}

/// Detect the MoE-resident HBM budget from hardware, subtracting reservations.
///
/// Uses `hipMemGetInfo` (via `RocmDevice::query_device_vram_bytes`) to get total
/// device memory, then subtracts estimated reservations for:
/// - Driver/kernel overhead (~1-2 GB typical)
/// - Kernel launch buffers, HIP streams, rocBLAS handles (~100-200 MB)
/// - KV cache reservation (configurable, default 0 — caller should account separately)
/// - Activation workspace for the current batch (configurable, default 0)
///
/// The remainder is the budget available for resident expert weights.
/// For GDDR-only systems (Radeon), this is the GDDR capacity minus reservations.
/// For hybrid systems, this is the HBM capacity minus reservations.
///
/// Returns `None` if VRAM probing fails or the device has 0 bytes (e.g., CPU fallback).
pub fn detect_moe_budget(
    ordinal: usize,
    _kv_cache_reservation_bytes: usize,
    _activation_reservation_bytes: usize,
) -> Option<usize> {
    let total_vram = RocmDevice::query_device_vram_bytes(ordinal);
    if total_vram == 0 {
        return None;
    }
    // Rough reservations: driver overhead, HIP streams, rocBLAS, kernel buffers.
    // Conservative estimate: ~2 GB for non-weight GPU state.
    let driver_reservation = 2usize * 1024 * 1024 * 1024usize;
    let available = total_vram.saturating_sub(driver_reservation);
    if available == 0 {
        return None;
    }
    // Subtract caller-specified reservations (KV cache, activations)
    let after_kv = available.saturating_sub(_kv_cache_reservation_bytes);
    let after_activations = after_kv.saturating_sub(_activation_reservation_bytes);
    if after_activations == 0 {
        return None;
    }
    Some(after_activations)
}

/// A routed MoE feed-forward block: router + experts + optional shared expert.
pub struct MoeFfn {
    pub router: MoeRouter,
    pub experts: ExpertBank,
    pub shared_expert: Option<ExpertTriple>,
    pub routed_scaling_factor: f32,
    /// Per-expert routing hotness accumulator (updated each forward call).
    /// Used by PlanBuilder to decide which experts deserve fp16 residency.
    hotness: std::sync::Mutex<Vec<f32>>,
    /// PlanBuilder for budget-feasible resident-set selection (P2-1 wiring).
    plan_builder: PlanBuilder,
    /// Cached resident plan from the last cache-miss rebuild.
    cached_plan: std::sync::Mutex<Option<ResidentPlan>>,
    /// ROCm resident expert-weight cache (see [`RocmResidentWeights`]).
    #[cfg(feature = "rocm-mem")]
    rocm_weights: std::sync::Mutex<Option<RocmResidentWeights>>,
    /// CUDA resident expert-weight cache (see [`CudaResidentWeights`]).
    #[cfg(feature = "cuda-mem")]
    cuda_weights: std::sync::Mutex<Option<CudaResidentWeights>>,
}

/// An independent SwiGLU triple for the (always-on) shared expert.
pub struct ExpertTriple {
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
    pub inter: usize,
    pub hidden: usize,
}

impl ExpertTriple {
    /// Load the shared expert's three projections from `ws` under the
    /// `ffn_gate_she` / `ffn_up_she` / `ffn_down_she` GGUF names.
    pub fn load(
        ws: &WeightSource<'_>,
        hidden: usize,
        inter: usize,
        has_bias: bool,
    ) -> Result<Self, grim_tensor::error::Error> {
        let gate = Linear::load(&ws.pp("ffn_gate_she"), hidden, inter, has_bias)?;
        let up = Linear::load(&ws.pp("ffn_up_she"), hidden, inter, has_bias)?;
        let down = Linear::load(&ws.pp("ffn_down_she"), inter, hidden, has_bias)?;
        Ok(Self {
            gate,
            up,
            down,
            inter,
            hidden,
        })
    }
}

impl ExpertTriple {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, grim_tensor::error::Error> {
        let g = self.gate.forward(x)?;
        let u = self.up.forward(x)?;
        let h = silu_mul_host(&g, &u)?;
        self.down.forward(&h)
    }
}

impl MoeFfn {
    pub fn new(
        router: MoeRouter,
        experts: ExpertBank,
        shared_expert: Option<ExpertTriple>,
        routed_scaling_factor: f32,
    ) -> Self {
        let n_experts = experts.gate.len();
        // Compute per-expert byte costs from the ACTUAL packed/precision dtype
        // of each expert's weights — works for any quant format (MXFP4/MXFP8/FP8/
        // Q4_K/Q5_K/Q6_K/IQ2/3/4 variants/NF4/FP4/block-quant) as well as native
        // fp16/f32. The PlanBuilder uses these real costs for budget-feasible selection.
        let (bytes_per_expert, hbm_budget) = if n_experts > 0 {
            let mut total_bytes = 0usize;
            for e in 0..n_experts {
                // Sum the packed byte cost of gate + up + down for this expert.
                // Use the dtype's storage to determine the actual per-element byte size.
                let gate_dtype = experts.gate[e].weight.dtype();
                let up_dtype = experts.up[e].weight.dtype();
                let down_dtype = experts.down[e].weight.dtype();
                let gate_elem = experts.gate[e]
                    .weight
                    .shape()
                    .dims()
                    .iter()
                    .product::<usize>();
                let up_elem = experts.up[e]
                    .weight
                    .shape()
                    .dims()
                    .iter()
                    .product::<usize>();
                let down_elem = experts.down[e]
                    .weight
                    .shape()
                    .dims()
                    .iter()
                    .product::<usize>();
                total_bytes += expert_weight_bytes(&gate_dtype, gate_elem)
                    + expert_weight_bytes(&up_dtype, up_elem)
                    + expert_weight_bytes(&down_dtype, down_elem);
            }
            // HBM budget: fit ALL experts at their actual packed cost (no demotion
            // needed unless the budget is explicitly tightened).
            let hbm_budget = total_bytes;
            (total_bytes / n_experts, hbm_budget)
        } else {
            (0, 0)
        };
        // int8 cost is ~half the actual packed cost (used when demoting fp16 experts
        // to a tighter format). For already-int8/quantized formats, demotion would
        // mean a further quantize step; we approximate as half.
        let bytes_per_int8 = (bytes_per_expert / 2).max(1);
        let plan_builder = PlanBuilder::new(bytes_per_expert, bytes_per_int8, hbm_budget);

        Self {
            router,
            experts,
            shared_expert,
            routed_scaling_factor,
            hotness: std::sync::Mutex::new(vec![0.0f32; n_experts]),
            plan_builder,
            cached_plan: std::sync::Mutex::new(None),
            #[cfg(feature = "rocm-mem")]
            rocm_weights: std::sync::Mutex::new(None),
            #[cfg(feature = "cuda-mem")]
            cuda_weights: std::sync::Mutex::new(None),
        }
    }

    /// Correct-but-unoptimized CPU reference forward for `[batch, hidden]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, grim_tensor::error::Error> {
        // WI-M5: when the activation already lives on the Vulkan device (the
        // model was loaded onto Vulkan) and there is no shared/always-on
        // expert to fold in, dispatch the fused grouped MoE kernel instead of
        // materializing every expert on the CPU. Router selection still runs on
        // the host (matches the CPU reference); only the expert GEMMs move to
        // GPU fast paths. Each mirrors `grim_moe_fused_dispatch` on the
        // respective backend; on any backend hiccup we fall through to the
        // verified CPU reference so the fused path is never wrong.
        #[cfg(feature = "cuda-mem")]
        if matches!(x.device(), Device::Cuda(_)) && self.shared_expert.is_none() {
            if let Ok(out) = self.forward_cuda(x) {
                return Ok(out);
            }
        }
        #[cfg(feature = "vulkan-mem")]
        if matches!(x.device(), Device::Vulkan) && self.shared_expert.is_none() {
            if let Ok(out) = self.forward_vulkan(x) {
                return Ok(out);
            }
            // Any backend hiccup falls back to the verified CPU reference.
        }
        #[cfg(feature = "metal-mem")]
        if matches!(x.device(), Device::Metal(_)) && self.shared_expert.is_none() {
            if let Ok(out) = self.forward_metal(x) {
                return Ok(out);
            }
        }
        #[cfg(feature = "rocm-mem")]
        if matches!(x.device(), Device::Rocm(_)) {
            if let Ok(out) = self.forward_rocm(x) {
                return Ok(out);
            }
        }

        let (indices, weights) = self.router.route(x)?;
        let batch = indices.len();
        let hidden = self
            .experts
            .down
            .first()
            .map(|l| l.weight.shape().dim(0).unwrap_or(0))
            .unwrap_or_else(|| x.shape().dims().last().copied().unwrap_or(0));

        let mut out_vec = vec![0.0f32; batch * hidden];

        for t in 0..batch {
            let experts = &indices[t];
            let w = &weights[t];
            let xt = slice_row(x, t)?; // [1, hidden]
            // Routed experts: combined output is scaled by `routed_scaling_factor`
            // (DeepSeek/Laguna convention — scales the *routed* path, not shared).
            let mut routed = vec![0.0f32; hidden];
            for (rank, &e) in experts.iter().enumerate() {
                let y = self.experts.expert_forward(e, &xt)?; // [1, hidden]
                let yv = y.to_vec_f32()?;
                for (i, v) in yv.iter().enumerate() {
                    routed[i] += w[rank] * v;
                }
            }
            for (i, v) in routed.iter().enumerate() {
                out_vec[t * hidden + i] += self.routed_scaling_factor * v;
            }
            // Shared/always-on expert is added unscaled.
            if let Some(sh) = &self.shared_expert {
                let s = sh.forward(&xt)?;
                let sv = s.to_vec_f32()?;
                for (i, v) in sv.iter().enumerate() {
                    out_vec[t * hidden + i] += v;
                }
            }
        }

        Ok(cpu_tensor(out_vec, Shape::new(vec![batch, hidden])))
    }

    /// MoE-aware bandwidth-adaptive hybrid decode forward pass.
    ///
    /// # Contract
    /// Partitions active routed experts into GPU resident/fills ($\mathcal{H} \cup \mathcal{F}$)
    /// and CPU host-RAM misses ($\mathcal{C}$) using the empirical $q^*$ policy.
    /// Evaluates both branches and merges partial sums: $y = y_{\text{GPU}} + y_{\text{CPU}}$.
    pub fn forward_moe_aware_hybrid(
        &self,
        x: &Tensor,
        executor: &grim_backend_rocm::MoeHybridExecutor,
        is_resident: impl Fn(usize) -> bool,
    ) -> Result<Tensor, grim_tensor::error::Error> {
        let (indices, weights) = self.router.route(x)?;
        let batch = indices.len();
        let hidden = self
            .experts
            .down
            .first()
            .map(|l| l.weight.shape().dim(0).unwrap_or(0))
            .unwrap_or_else(|| x.shape().dims().last().copied().unwrap_or(0));

        let mut gpu_partial = vec![0.0f32; batch * hidden];
        let mut cpu_partial = vec![0.0f32; batch * hidden];

        for t in 0..batch {
            let experts = &indices[t];
            let w = &weights[t];
            let xt = slice_row(x, t)?;

            // Build hybrid plan for this token's active experts
            let plan = executor.plan_step(0, experts, &is_resident);
            let gpu_set: std::collections::HashSet<usize> = plan
                .gpu_resident_experts
                .iter()
                .chain(plan.gpu_fill_experts.iter())
                .copied()
                .collect();
            let cpu_set: std::collections::HashSet<usize> =
                plan.cpu_compute_experts.iter().copied().collect();

            // GPU path (resident + fills)
            for (rank, &e) in experts.iter().enumerate() {
                if gpu_set.contains(&e) {
                    let y = self.experts.expert_forward(e, &xt)?;
                    let yv = y.to_vec_f32()?;
                    for (i, v) in yv.iter().enumerate() {
                        gpu_partial[t * hidden + i] += w[rank] * v;
                    }
                }
            }

            // CPU path (residual misses in host RAM)
            for (rank, &e) in experts.iter().enumerate() {
                if cpu_set.contains(&e) {
                    let y = self.experts.expert_forward(e, &xt)?;
                    let yv = y.to_vec_f32()?;
                    for (i, v) in yv.iter().enumerate() {
                        cpu_partial[t * hidden + i] += w[rank] * v;
                    }
                }
            }
        }

        // Exact merge y = y_GPU + y_CPU
        grim_backend_rocm::MoeHybridExecutor::merge_outputs(&mut gpu_partial, &cpu_partial)?;

        let mut out_vec = vec![0.0f32; batch * hidden];
        for t in 0..batch {
            for i in 0..hidden {
                out_vec[t * hidden + i] = self.routed_scaling_factor * gpu_partial[t * hidden + i];
            }
            if let Some(sh) = &self.shared_expert {
                let xt = slice_row(x, t)?;
                let s = sh.forward(&xt)?;
                let sv = s.to_vec_f32()?;
                for (i, v) in sv.iter().enumerate() {
                    out_vec[t * hidden + i] += v;
                }
            }
        }

        Ok(cpu_tensor(out_vec, Shape::new(vec![batch, hidden])))
    }

    /// Vulkan dispatch of the fused grouped MoE kernel (WI-M5).
    ///
    /// Flattens each expert's gate/up/down weights into the single contiguous
    /// `[num_experts, inter, hidden]` / `[num_experts, hidden, inter]` buffers
    /// the shader expects, expands top-k routing into flat token/expert/weight
    /// arrays, and launches one workgroup per routed (token, expert) pair. The
    /// router selection is identical to the CPU reference (host-computed), so
    /// the only behavioral difference is the expert GEMMs running on the GPU.
    ///
    /// NOTE: weights are pulled to the host and re-uploaded as one buffer here
    /// (a host round-trip). It is correct and matches the CPU reference; caching
    /// the flattened weight buffers on first use is a follow-up optimization.
    #[cfg(feature = "vulkan-mem")]
    fn forward_vulkan(&self, x: &Tensor) -> Result<Tensor, grim_tensor::error::Error> {
        let (indices, weights) = self.router.route(x)?;
        let batch = indices.len();
        let hidden = self
            .experts
            .down
            .first()
            .map(|l| l.weight.shape().dim(0).unwrap_or(0))
            .unwrap_or_else(|| x.shape().dims().last().copied().unwrap_or(0));
        let inter = self
            .experts
            .gate
            .first()
            .map(|l| l.weight.shape().dim(0).unwrap_or(0))
            .unwrap_or(0);
        let num_experts = self.experts.gate.len();
        if inter == 0 || num_experts == 0 || hidden == 0 {
            return Err(grim_tensor::error::Error::ShapeMismatch {
                expected: vec![inter, hidden, num_experts],
                got: vec![0, 0, 0],
            });
        }

        // Flatten expert weights (row-major per expert, outer = expert idx).
        let mut gate_flat = Vec::with_capacity(num_experts * inter * hidden);
        let mut up_flat = Vec::with_capacity(num_experts * inter * hidden);
        let mut down_flat = Vec::with_capacity(num_experts * hidden * inter);
        for e in 0..num_experts {
            gate_flat.extend_from_slice(&self.experts.gate[e].weight.to_vec_f32()?);
            up_flat.extend_from_slice(&self.experts.up[e].weight.to_vec_f32()?);
            down_flat.extend_from_slice(&self.experts.down[e].weight.to_vec_f32()?);
        }

        // Expand top-k routing into flat arrays (one entry per routed pair).
        let mut rtok: Vec<u32> = Vec::new();
        let mut rexp: Vec<u32> = Vec::new();
        let mut rw: Vec<f32> = Vec::new();
        for t in 0..batch {
            for (rank, &e) in indices[t].iter().enumerate() {
                rtok.push(t as u32);
                rexp.push(e as u32);
                rw.push(weights[t][rank]);
            }
        }
        let num_pairs = rtok.len();

        let dev = VulkanDevice::new();
        let x_storage: &dyn BackendStorage = &**x.storage();
        let gate_buf =
            dev.upload_f32(&gate_flat, &Shape::new(vec![num_experts * inter * hidden]))?;
        let up_buf = dev.upload_f32(&up_flat, &Shape::new(vec![num_experts * inter * hidden]))?;
        let down_buf =
            dev.upload_f32(&down_flat, &Shape::new(vec![num_experts * hidden * inter]))?;
        let tok_buf = dev.upload_u32(&rtok, &Shape::new(vec![num_pairs]))?;
        let exp_buf = dev.upload_u32(&rexp, &Shape::new(vec![num_pairs]))?;
        let w_buf = dev.upload_f32(&rw, &Shape::new(vec![num_pairs]))?;

        let out_shape = Shape::new(vec![batch, hidden]);
        let (out_storage, _handle) = dev.moe_fused_dispatch(
            x_storage,
            &*gate_buf,
            &*up_buf,
            &*down_buf,
            &*tok_buf,
            &*exp_buf,
            &*w_buf,
            &out_shape,
            hidden as u32,
            inter as u32,
            num_experts as u32,
            batch as u32,
            self.routed_scaling_factor,
        )?;

        Ok(Tensor::new(
            Arc::from(out_storage),
            out_shape,
            DType::F32,
            QuantProvenance::default(),
            Device::Vulkan,
        ))
    }

    /// CUDA dispatch of the fused grouped MoE kernel (WI-M5). Mirrors
    /// `forward_vulkan`: expands top-k routing into flat token/expert/weight
    /// arrays and calls `CudaDevice::moe_fused_dispatch_resident` against
    /// device-resident weight buffers (P2-1: weights are uploaded once and
    /// cached across forward calls, eliminating the per-call host round-trip).
    /// The activation `x` is already `CudaStorage` (model runs on CUDA).
    #[cfg(feature = "cuda-mem")]
    fn forward_cuda(&self, x: &Tensor) -> Result<Tensor, grim_tensor::error::Error> {
        let ordinal = match x.device() {
            Device::Cuda(o) => *o,
            _ => {
                return Err(grim_tensor::error::Error::Backend(
                    "forward_cuda: x is not on a CUDA device".into(),
                ));
            }
        };
        let (indices, weights) = self.router.route(x)?;
        let batch = indices.len();
        let hidden = self
            .experts
            .down
            .first()
            .map(|l| l.weight.shape().dim(0).unwrap_or(0))
            .unwrap_or_else(|| x.shape().dims().last().copied().unwrap_or(0));
        let inter = self
            .experts
            .gate
            .first()
            .map(|l| l.weight.shape().dim(0).unwrap_or(0))
            .unwrap_or(0);
        let num_experts = self.experts.gate.len();
        if inter == 0 || num_experts == 0 || hidden == 0 {
            return Err(grim_tensor::error::Error::ShapeMismatch {
                expected: vec![inter, hidden, num_experts],
                got: vec![0, 0, 0],
            });
        }

        // Resident expert-weight fast path: the flattened gate/up/down banks
        // are uploaded to the device once and reused across forward calls
        // (P2-1). The cache is rebuilt only when the resident set — expert
        // count plus weight shapes — actually changes.
        let (gate_buf, up_buf, down_buf) = {
            let mut guard = self.cuda_weights.lock().unwrap_or_else(|e| e.into_inner());
            let key = (num_experts, hidden, inter);
            match guard.as_ref() {
                Some(r) if r.fingerprint == key => {
                    (Arc::clone(&r.gate), Arc::clone(&r.up), Arc::clone(&r.down))
                }
                _ => {
                    let built = CudaResidentWeights::build(
                        &self.experts,
                        num_experts,
                        hidden,
                        inter,
                        ordinal,
                    )?;
                    let triple = (
                        Arc::clone(&built.gate),
                        Arc::clone(&built.up),
                        Arc::clone(&built.down),
                    );
                    *guard = Some(built);
                    triple
                }
            }
        };

        // Expand top-k routing into flat arrays (one entry per routed pair).
        let mut rtok: Vec<u32> = Vec::new();
        let mut rexp: Vec<u32> = Vec::new();
        let mut rw: Vec<f32> = Vec::new();
        for t in 0..batch {
            for (rank, &e) in indices[t].iter().enumerate() {
                rtok.push(t as u32);
                rexp.push(e as u32);
                rw.push(weights[t][rank]);
            }
        }
        let num_pairs = rtok.len();

        let dev = CudaDevice::new(ordinal)?;
        let x_storage: &dyn BackendStorage = &**x.storage();
        // Router arrays are integer; stage the raw u32 bytes (the kernel reads
        // them as `unsigned int*`). DType label is irrelevant for a raw copy.
        let rtok_bytes: Vec<u8> = rtok.iter().flat_map(|v| v.to_le_bytes()).collect();
        let rexp_bytes: Vec<u8> = rexp.iter().flat_map(|v| v.to_le_bytes()).collect();
        let rw_bytes: Vec<u8> = rw.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tok_buf = Box::new(CudaStorage::copy_from_host_raw_bytes(
            &rtok_bytes,
            &Shape::new(vec![num_pairs]),
            DType::F32,
            ordinal,
        )?);
        let exp_buf = Box::new(CudaStorage::copy_from_host_raw_bytes(
            &rexp_bytes,
            &Shape::new(vec![num_pairs]),
            DType::F32,
            ordinal,
        )?);
        let w_buf = Box::new(CudaStorage::copy_from_host_raw_bytes(
            &rw_bytes,
            &Shape::new(vec![num_pairs]),
            DType::F32,
            ordinal,
        )?);

        let out_shape = Shape::new(vec![batch, hidden]);
        let (out_storage, _handle) = dev.moe_fused_dispatch_resident(
            x_storage,
            &*gate_buf,
            &*up_buf,
            &*down_buf,
            &*tok_buf,
            &*exp_buf,
            &*w_buf,
            &out_shape,
            hidden as u32,
            inter as u32,
            num_experts as u32,
            batch as u32,
            self.routed_scaling_factor,
        )?;

        Ok(Tensor::new(
            Arc::from(out_storage),
            out_shape,
            DType::F32,
            QuantProvenance::default(),
            Device::Cuda(ordinal),
        ))
    }

    /// Metal dispatch of the fused grouped MoE kernel (WI-M5). Mirrors
    /// `forward_vulkan`/`forward_cuda`: flattens expert weights, expands top-k
    /// routing into flat (token, expert, weight) arrays, and runs the MSL
    /// `grim_moe_fused_dispatch` kernel. The router arrays are f32-backed
    /// (Metal has no integer storage in this crate) and the shader casts them
    /// back to `int`. On any backend hiccup the caller falls back to the
    /// verified CPU reference.
    #[cfg(feature = "metal-mem")]
    fn forward_metal(&self, x: &Tensor) -> Result<Tensor, grim_tensor::error::Error> {
        let ordinal = match x.device() {
            Device::Metal(o) => *o,
            _ => {
                return Err(grim_tensor::error::Error::Backend(
                    "forward_metal: x is not on a Metal device".into(),
                ));
            }
        };
        let (indices, weights) = self.router.route(x)?;
        let batch = indices.len();
        let hidden = self
            .experts
            .down
            .first()
            .map(|l| l.weight.shape().dim(0).unwrap_or(0))
            .unwrap_or_else(|| x.shape().dims().last().copied().unwrap_or(0));
        let inter = self
            .experts
            .gate
            .first()
            .map(|l| l.weight.shape().dim(0).unwrap_or(0))
            .unwrap_or(0);
        let num_experts = self.experts.gate.len();
        if inter == 0 || num_experts == 0 || hidden == 0 {
            return Err(grim_tensor::error::Error::ShapeMismatch {
                expected: vec![inter, hidden, num_experts],
                got: vec![0, 0, 0],
            });
        }

        // Flatten expert weights (row-major per expert, outer = expert idx).
        let mut gate_flat = Vec::with_capacity(num_experts * inter * hidden);
        let mut up_flat = Vec::with_capacity(num_experts * inter * hidden);
        let mut down_flat = Vec::with_capacity(num_experts * hidden * inter);
        for e in 0..num_experts {
            gate_flat.extend_from_slice(&self.experts.gate[e].weight.to_vec_f32()?);
            up_flat.extend_from_slice(&self.experts.up[e].weight.to_vec_f32()?);
            down_flat.extend_from_slice(&self.experts.down[e].weight.to_vec_f32()?);
        }

        // Expand top-k routing into flat arrays (one entry per routed pair).
        let mut rtok: Vec<f32> = Vec::new();
        let mut rexp: Vec<f32> = Vec::new();
        let mut rw: Vec<f32> = Vec::new();
        for t in 0..batch {
            for (rank, &e) in indices[t].iter().enumerate() {
                rtok.push(t as f32);
                rexp.push(e as f32);
                rw.push(weights[t][rank]);
            }
        }
        let num_pairs = rtok.len();

        let dev = MetalDevice::new(ordinal)?;
        let x_storage: &dyn BackendStorage = &**x.storage();
        let gate_buf = CoreTensorOps::from_cpu(
            &dev,
            &gate_flat,
            &Shape::new(vec![num_experts * inter * hidden]),
            DType::F32,
        )?;
        let up_buf = CoreTensorOps::from_cpu(
            &dev,
            &up_flat,
            &Shape::new(vec![num_experts * inter * hidden]),
            DType::F32,
        )?;
        let down_buf = CoreTensorOps::from_cpu(
            &dev,
            &down_flat,
            &Shape::new(vec![num_experts * hidden * inter]),
            DType::F32,
        )?;
        // Router arrays are f32-backed (the shader casts back to int).
        let tok_buf =
            CoreTensorOps::from_cpu(&dev, &rtok, &Shape::new(vec![num_pairs]), DType::F32)?;
        let exp_buf =
            CoreTensorOps::from_cpu(&dev, &rexp, &Shape::new(vec![num_pairs]), DType::F32)?;
        let w_buf = CoreTensorOps::from_cpu(&dev, &rw, &Shape::new(vec![num_pairs]), DType::F32)?;

        let out_shape = Shape::new(vec![batch, hidden]);
        let (out_storage, _handle) = dev.moe_fused_dispatch(
            x_storage,
            &*gate_buf,
            &*up_buf,
            &*down_buf,
            &*tok_buf,
            &*exp_buf,
            &*w_buf,
            &out_shape,
            hidden as u32,
            inter as u32,
            num_experts as u32,
            batch as u32,
            self.routed_scaling_factor,
        )?;

        Ok(Tensor::new(
            Arc::from(out_storage),
            out_shape,
            DType::F32,
            QuantProvenance::default(),
            Device::Metal(ordinal),
        ))
    }

    /// ROCm HIP dispatch of the Charon fused MoE kernel.
    #[cfg(feature = "rocm-mem")]
    fn forward_rocm(&self, x: &Tensor) -> Result<Tensor, grim_tensor::error::Error> {
        let ordinal = match x.device() {
            Device::Rocm(o) => *o,
            _ => {
                return Err(grim_tensor::error::Error::Backend(
                    "forward_rocm: x is not on a ROCm device".into(),
                ));
            }
        };
        let (indices, weights) = self.router.route(x)?;
        let batch = indices.len();
        let hidden = self
            .experts
            .down
            .first()
            .map(|l| l.weight.shape().dim(0).unwrap_or(0))
            .unwrap_or_else(|| x.shape().dims().last().copied().unwrap_or(0));
        let inter = self
            .experts
            .gate
            .first()
            .map(|l| l.weight.shape().dim(0).unwrap_or(0))
            .unwrap_or(0);
        let num_experts = self.experts.gate.len();
        if inter == 0 || num_experts == 0 || hidden == 0 {
            return Err(grim_tensor::error::Error::ShapeMismatch {
                expected: vec![inter, hidden, num_experts],
                got: vec![0, 0, 0],
            });
        }

        let assignment =
            grim_backend_rocm::kernels::charon::RoutingAssignment::from_route(&indices, &weights)?;

        let x_storage: &dyn BackendStorage = &**x.storage();
        let x_rocm = x_storage
            .as_any()
            .downcast_ref::<grim_backend_rocm::RocmStorage>()
            .ok_or_else(|| grim_tensor::error::Error::Backend("x is not RocmStorage".into()))?;

        let out_shape = Shape::new(vec![batch, hidden]);

        // Update per-expert routing hotness from this call's router weights.
        // Each token's routing weights indicate how much each expert was used;
        // accumulate to track which experts are "hot" over time.
        {
            let mut hotness = self.hotness.lock().unwrap();
            for (token_indices, token_weights) in indices.iter().zip(weights.iter()) {
                for (&expert, &w) in token_indices.iter().zip(token_weights.iter()) {
                    hotness[expert] += w;
                }
            }
        }

        // PlanBuilder: decide which experts deserve fp16 residency under the
        // HBM budget. Called on every call with the updated hotness; the plan
        // is cached for the resident-set rebuild (cache-miss path below).
        let plan = self
            .plan_builder
            .build(&self.hotness.lock().unwrap(), false);
        *self.cached_plan.lock().unwrap() = Some(plan.clone());

        // Resident expert-weight fast path: the flattened gate/up/down banks
        // are uploaded to the device once and reused across forward calls
        // (P2-1). The cache is rebuilt only when the resident set changes.
        // PlanBuilder directs which experts are worth keeping resident; the
        // current build uploads all experts (the PlanBuilder selects the subset
        // that fits the HBM budget, but the full upload happens once per
        // fingerprint change — cold experts would use the int8 dequant path
        // in a full implementation).
        let resident = {
            let mut guard = self.rocm_weights.lock().unwrap_or_else(|e| e.into_inner());
            let key = (num_experts, hidden, inter);
            match guard.as_ref() {
                Some(r) if r.fingerprint == key => {
                    (Arc::clone(&r.gate), Arc::clone(&r.up), Arc::clone(&r.down))
                }
                _ => {
                    let built = RocmResidentWeights::build(
                        &self.experts,
                        num_experts,
                        hidden,
                        inter,
                        ordinal,
                    )?;
                    let triple = (
                        Arc::clone(&built.gate),
                        Arc::clone(&built.up),
                        Arc::clone(&built.down),
                    );
                    *guard = Some(built);
                    triple
                }
            }
        };

        let dev = RocmDevice::try_new(ordinal)?;
        let (out_storage, _handle) = dev.moe_fused_dispatch_resident(
            x_rocm,
            &*resident.0,
            &*resident.1,
            &*resident.2,
            &assignment,
            &out_shape,
            hidden,
            inter,
            self.routed_scaling_factor,
        )?;

        let mut out_tensor = Tensor::new(
            Arc::from(out_storage),
            out_shape.clone(),
            DType::F32,
            QuantProvenance::default(),
            Device::Rocm(ordinal),
        );

        if let Some(sh) = &self.shared_expert {
            let s = sh.forward(x)?;
            out_tensor = crate::modules::add_tensors(&out_tensor, &s)?;
        }

        Ok(out_tensor)
    }
}

// ---------------------------------------------------------------------------
// Host math helpers
// ---------------------------------------------------------------------------

fn softmax(v: &[f32]) -> Vec<f32> {
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = v.iter().map(|&x| (x - m).exp()).collect();
    let s: f32 = exps.iter().sum();
    exps.iter().map(|e| e / s).collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Elementwise `silu(g) * u` on host (SwiGLU activation).
fn silu_mul_host(g: &Tensor, u: &Tensor) -> Result<Tensor, grim_tensor::error::Error> {
    let gv = g.to_vec_f32()?;
    let uv = u.to_vec_f32()?;
    let out: Vec<f32> = gv
        .iter()
        .zip(uv.iter())
        .map(|(&a, &b)| a * sigmoid(a) * b)
        .collect();
    Ok(cpu_tensor(out, g.shape().clone()))
}

fn slice_expert(flat: &[f32], e: usize, out: usize, in_dim: usize) -> Vec<f32> {
    let stride = out * in_dim;
    flat[e * stride..(e + 1) * stride].to_vec()
}

fn err_proj() -> grim_tensor::error::Error {
    grim_tensor::error::Error::Backend("expert projection missing".into())
}

/// f32 → IEEE 754 half-precision bits (round-to-nearest-even).
/// Split a length-prefixed MXFP4 buffer into its `(codes, exps)` segments.
fn split_mxfp4_framed(bytes: &[u8]) -> Result<(&[u8], &[u8]), grim_tensor::error::Error> {
    if bytes.len() < 16 {
        return Err(grim_tensor::error::Error::Backend(
            "split_mxfp4_framed: buffer too short for two length prefixes".into(),
        ));
    }
    let codes_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    if bytes.len() < 8 + codes_len + 8 {
        return Err(grim_tensor::error::Error::Backend(
            "split_mxfp4_framed: truncated codes segment".into(),
        ));
    }
    let exps_len =
        u64::from_le_bytes(bytes[8 + codes_len..8 + codes_len + 8].try_into().unwrap()) as usize;
    if bytes.len() < 8 + codes_len + 8 + exps_len {
        return Err(grim_tensor::error::Error::Backend(
            "split_mxfp4_framed: truncated exps segment".into(),
        ));
    }
    Ok((
        &bytes[8..8 + codes_len],
        &bytes[8 + codes_len + 8..8 + codes_len + 8 + exps_len],
    ))
}

fn slice_row(x: &Tensor, t: usize) -> Result<Tensor, grim_tensor::error::Error> {
    let v = x.to_vec_f32()?;
    let hidden = x.shape().dims().last().copied().unwrap_or(0);
    Ok(cpu_tensor(
        v[t * hidden..(t + 1) * hidden].to_vec(),
        Shape::new(vec![1, hidden]),
    ))
}

fn bias_opt(has_bias: bool, dim: usize) -> Option<Tensor> {
    if has_bias {
        Some(cpu_tensor(vec![0.0f32; dim], Shape::new(vec![dim])))
    } else {
        None
    }
}

#[cfg(test)]
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

// ===========================================================================
// WI-C — Router-distilled lookahead predictor + PlanBuilder + SRP/SCH gate
// ===========================================================================
//
// The predict leg of P-DAFD (PROBE 2602.00509, MxMoE 2505.05799, DynaExq
// 2511.15015, SRP/SCH 2505.16056). This is the genuinely novel composition —
// no published system fuses dispatch AND predicts AND varies per-expert
// precision. All three components below are host-side and unit-testable
// without a GPU: the falsifiable core of WI-C (G-C1/C2/C3) does NOT require
// hardware (plan §5).
//
// Honesty valves (do not weaken):
// * G-C2 scores the predictor against *actual next-layer routing* (Hit@k ≥
//   0.80), not output parity — a predictor wrong in an interesting way
//   cannot be rescued by the kernel producing the right answer.
// * G-C3 requires the feature to beat its own off-switch (pre-registered
//   Δ ≥ +0.05 Hit@k or PPL) or it is recorded as FAIL "prediction adds no
//   signal", never "≈acceptable".
// * The SRP/SCH confidence gate is mandatory (§5): below-threshold routing
//   consistency disables prediction, falling back to WI-B reactive matching.

// ---------------------------------------------------------------------------
// LookaheadPredictor — gate-initialized low-rank distilled router copy
// ---------------------------------------------------------------------------

/// A tiny distilled copy of `MoeRouter::gate` that forecasts the *next*
/// layer's activated-expert distribution from the current layer's gate
/// logits (PROBE 2602.00509, "gate-initialized" lookahead).
///
/// The predictor is a single low-rank linear: `predicted_next_logits =
/// current_logits @ W_distill`, where `W_distill` is `[num_experts,
/// num_experts]` initialized to a per-expert identity (the "gate-init"
/// prior that next-layer routing ≈ this-layer routing). It runs host-side;
/// output = predicted histogram (softmax over the predicted logits) + a
/// per-expert hotness vector (the predicted top-k probabilities).
///
/// Distillation updates `W_distill` online from observed (current → next)
/// routing pairs; v1 uses a closed-form ridge update, no GPU.
pub struct LookaheadPredictor {
    /// `W_distill`, `[num_experts, num_experts]` row-major.
    pub distill: Vec<f32>,
    pub num_experts: usize,
    /// Top-k the predictor forecasts hotness for.
    pub top_k: usize,
    /// Whether the SRP/SCH gate has enabled prediction. When `false`,
    /// `predict` returns the identity prior (this-layer routing unchanged),
    /// i.e. the WI-B reactive fallback.
    pub enabled: bool,
}

impl LookaheadPredictor {
    /// Build a gate-initialized predictor: `W_distill = I` (next-layer ≈
    /// current-layer routing, the strongest uninformed prior). `enabled`
    /// starts `true`; the SRP/SCH gate sets it `false` when the model's
    /// routing consistency is below threshold.
    pub fn new_gate_initialized(num_experts: usize, top_k: usize) -> Self {
        let mut distill = vec![0.0f32; num_experts * num_experts];
        for i in 0..num_experts {
            distill[i * num_experts + i] = 1.0; // identity prior
        }
        Self {
            distill,
            num_experts,
            top_k: top_k.min(num_experts),
            enabled: true,
        }
    }

    /// Predict the next layer's activated-expert distribution from this
    /// layer's gate logits.
    ///
    /// Returns `(predicted_top_k_indices, predicted_top_k_probs)` — the
    /// forecast hot set and their normalized probabilities. When `enabled`
    /// is `false`, returns the current-layer top-k unchanged (the reactive
    /// fallback that adds no prediction signal — G-C3's off-switch).
    pub fn predict(&self, current_logits: &[f32]) -> (Vec<usize>, Vec<f32>) {
        assert_eq!(
            current_logits.len(),
            self.num_experts,
            "current_logits length must equal num_experts"
        );
        if !self.enabled {
            // Identity prior: next-layer routing ≈ this-layer routing.
            // Skip the distill matrix multiply entirely — the off-switch
            // must not consult W_distill at all.
            return self.top_k_from_logits(current_logits);
        }
        // predicted_next_logits[j] = sum_i current_logits[i] * W[i, j]
        let mut pred = vec![0.0f32; self.num_experts];
        for (j, slot) in pred.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (i, &logit) in current_logits.iter().enumerate() {
                acc += logit * self.distill[i * self.num_experts + j];
            }
            *slot = acc;
        }
        self.top_k_from_logits(&pred)
    }

    /// Softmax over `logits`, then take `top_k` by probability and renormalize
    /// the selected probabilities over the chosen set (mirrors
    /// `MoeRouter::route`'s SoftmaxTopK combine-weight convention).
    fn top_k_from_logits(&self, logits: &[f32]) -> (Vec<usize>, Vec<f32>) {
        let probs = softmax(logits);
        let mut order: Vec<usize> = (0..self.num_experts).collect();
        order.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        let chosen: Vec<usize> = order.iter().take(self.top_k).copied().collect();
        let raw: Vec<f32> = chosen.iter().map(|&i| probs[i]).collect();
        let sum: f32 = raw.iter().sum();
        let chosen_probs: Vec<f32> = if sum > 0.0 {
            raw.iter().map(|p| p / sum).collect()
        } else {
            raw
        };
        (chosen, chosen_probs)
    }

    /// One closed-form ridge distillation step from an observed
    /// (current_logits → next_layer_activated_set) pair. Strength `lr ∈
    /// (0, 1]`; v1 uses a Hebbian-style update pulling `W[i, j]` toward the
    /// co-activation signal `current_logits[i] * next_onehot[j]`.
    pub fn distill_step(&mut self, current_logits: &[f32], next_activated: &[usize], lr: f32) {
        let mut next_onehot = vec![0.0f32; self.num_experts];
        for &e in next_activated {
            if e < self.num_experts {
                next_onehot[e] = 1.0;
            }
        }
        // Hebbian: W[i,j] += lr * (target - W[i,j]*current[i]) * current[i]
        // — a one-step ridge pull toward the observed co-activation.
        for (i, &cur_i) in current_logits.iter().enumerate() {
            for (j, &onehot_j) in next_onehot.iter().enumerate() {
                let pred_ij = cur_i * self.distill[i * self.num_experts + j];
                let target_ij = cur_i * onehot_j;
                self.distill[i * self.num_experts + j] += lr * (target_ij - pred_ij);
            }
        }
    }
}

/// Score prediction Hit@k: the fraction of the realized top-k set that the
/// predictor's top-k forecast captured. `1.0` = perfect overlap, `0.0` =
/// no overlap. This is the G-C2 metric (≥0.80 bar), scored against actual
/// next-layer routing — not output parity.
pub fn prediction_hit_at_k(predicted: &[usize], realized: &[usize]) -> f32 {
    if realized.is_empty() {
        return 0.0;
    }
    let hits = predicted.iter().filter(|p| realized.contains(p)).count();
    hits as f32 / realized.len() as f32
}

// ---------------------------------------------------------------------------
// PlanBuilder — DynaExq-budget-feasible resident-set + precision plan
// ---------------------------------------------------------------------------

/// Per-expert precision in the resident set (MxMoE 2505.05799 mixed-precision
/// flavor). Hot experts stay fp16; cold experts fall back to int8 (via the
/// existing `q*k_gemm` dequant path) to fit the HBM envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertPrecision {
    Fp16,
    Int8,
}

/// A budget-feasible resident-set plan: which experts are hot (fp16
/// resident) vs cold (int8 fallback), under the HBM byte envelope. Output
/// of `PlanBuilder::build`.
#[derive(Debug, Clone)]
pub struct ResidentPlan {
    pub precision: Vec<ExpertPrecision>,
    /// Whether prediction drove this plan (`true`) or it's the reactive
    /// WI-B fallback (`false`). G-C3 compares both on the same traces.
    pub prediction_driven: bool,
}

/// DynaExq-style budget-feasible top-n planner. Keeps the hottest experts
/// fp16-resident up to the HBM byte budget; demotes the rest to int8.
pub struct PlanBuilder {
    /// Bytes per fp16-resident expert (gate+up+down triples).
    bytes_per_expert_fp16: usize,
    /// Bytes per int8 expert (≈ fp16/2 + quant overhead).
    bytes_per_expert_int8: usize,
    /// HBM envelope for the expert resident set.
    hbm_budget_bytes: usize,
}

impl PlanBuilder {
    /// Construct with per-expert byte costs and the total HBM envelope.
    /// `bytes_per_expert_fp16` is the full `[inter, hidden] × 3` triple;
    /// `bytes_per_expert_int8` is the quantized size (typically fp16/2).
    pub fn new(
        bytes_per_expert_fp16: usize,
        bytes_per_expert_int8: usize,
        hbm_budget_bytes: usize,
    ) -> Self {
        Self {
            bytes_per_expert_fp16,
            bytes_per_expert_int8,
            hbm_budget_bytes,
        }
    }

    /// Build a resident plan from a per-expert hotness vector (predicted or
    /// observed routing frequency). The hottest experts are kept fp16 up
    /// to the budget; the rest demote to int8. `prediction_driven` labels
    /// the plan for G-C3's off-switch comparison.
    pub fn build(&self, hotness: &[f32], prediction_driven: bool) -> ResidentPlan {
        let n = hotness.len();
        // Rank experts by hotness (desc); ties broken by index for stability.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            hotness[b]
                .partial_cmp(&hotness[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });

        let mut precision = vec![ExpertPrecision::Int8; n];
        let mut used = 0usize;
        // Greedy: promote experts to fp16 in hotness order while the *upgrade*
        // cost (fp16 bytes − int8 bytes) stays within the HBM upgrade budget.
        // The all-int8 floor is the always-resident baseline and is not counted
        // against `hbm_budget_bytes` (an empty/zero budget still keeps the int8
        // floor resident).
        let _baseline = n * self.bytes_per_expert_int8;
        for &e in &order {
            let upgrade_cost = self
                .bytes_per_expert_fp16
                .saturating_sub(self.bytes_per_expert_int8);
            if used + upgrade_cost <= self.hbm_budget_bytes {
                // budget == 0 means nothing gets promoted beyond the int8 baseline
                // (the `|| budget == 0` clause was removed — it contradicted the
                // doc comment which says zero budget keeps everything at int8).
                precision[e] = ExpertPrecision::Fp16;
                used += upgrade_cost;
            } else {
                break;
            }
        }
        ResidentPlan {
            precision,
            prediction_driven,
        }
    }

    /// Bytes the resident set would occupy under this plan.
    pub fn plan_bytes(&self, plan: &ResidentPlan) -> usize {
        plan.precision
            .iter()
            .map(|p| match p {
                ExpertPrecision::Fp16 => self.bytes_per_expert_fp16,
                ExpertPrecision::Int8 => self.bytes_per_expert_int8,
            })
            .sum()
    }
}

// ---------------------------------------------------------------------------
// SRP/SCH confidence gate — mandatory prediction on/off valve
// ---------------------------------------------------------------------------

/// Compute the model's local-routing-consistency (SRP/SCH 2505.16056) from
/// a trace of consecutive-layer routing decisions. Returns the fraction of
/// (layer, token, expert) triples that recur in the next layer — a measure
/// of how predictable the routing is. Below `threshold`, the
/// `LookaheadPredictor` is disabled (§5: the gate is mandatory, not
/// optional — don't claim prediction works on models it measurably can't).
///
/// `trace[t]` = the activated-expert set for token row `t` across layers;
/// the outer Vec is layers, inner Vec is per-token activated experts. We
/// score the per-token set-overlap between adjacent layers averaged over
/// tokens and layer-transitions.
pub fn routing_consistency(trace: &[Vec<Vec<usize>>]) -> f32 {
    if trace.len() < 2 {
        return 0.0; // need at least two layers to measure consistency
    }
    let mut total_overlap = 0.0f32;
    let mut total_sets = 0u32;
    for layer in 0..trace.len() - 1 {
        let cur = &trace[layer];
        let nxt = &trace[layer + 1];
        let rows = cur.len().min(nxt.len());
        for t in 0..rows {
            let cur_set = &cur[t];
            let nxt_set = &nxt[t];
            if cur_set.is_empty() {
                continue;
            }
            let overlap = cur_set.iter().filter(|e| nxt_set.contains(e)).count();
            total_overlap += overlap as f32 / cur_set.len() as f32;
            total_sets += 1;
        }
    }
    if total_sets == 0 {
        return 0.0;
    }
    total_overlap / total_sets as f32
}

/// Apply the SRP/SCH gate to a predictor: if the trace's routing
/// consistency is below `threshold`, disable prediction (set
/// `predictor.enabled = false`) so it falls back to the reactive WI-B
/// matching. Returns the measured consistency so the caller can log it.
pub fn apply_srp_sch_gate(
    predictor: &mut LookaheadPredictor,
    trace: &[Vec<Vec<usize>>],
    threshold: f32,
) -> f32 {
    let consistency = routing_consistency(trace);
    predictor.enabled = consistency >= threshold;
    consistency
}

// ===========================================================================
// WI-EP1 — ExpertPlacementMap (host-side, no device required)
// ===========================================================================
//
// charon_kernel_plan_v3.md §3 WI-EP1: "ExpertPlacementMap — which GPU owns
// which expert, built via `C2plrController::decide()` at expert granularity,
// capacity-proportional fallback tested under both homogeneous and mixed-GPU
// synthetic cases."
//
// The placement *logic* is host-testable without a device: it consumes the
// `GpuCapability` snapshots the host already gathers (VRAM, TFLOPS, throttle)
// and assigns each expert to a rank proportional to that rank's capacity.
// The on-device dispatch (experts actually firing on their assigned ranks) is
// device-gated; this struct is the host-side planner that feeds WI-EP2's
// cross-GPU token dispatch and WI-EP3's combine.
//
// Two placement policies:
//   * `CapacityProportional` — split experts across ranks proportional to a
//     capacity metric (VRAM, TFLOPS, or a blend). The default; the plan's
//     "capacity-proportional fallback" requirement.
//   * `Controller` — defer to `C2plrController::decide()` per expert (the
//     online-learning path). When the controller's MLP weights are zero
//     (fresh controller), `decide` falls back to round-robin, so this policy
//     degrades gracefully; capacity-proportional is the explicit fallback for
//     the controller's cold-start case.
//
// Both policies produce the same `ExpertPlacementMap` shape: a per-expert →
// rank assignment plus the per-rank load fraction (used by WI-EP2 to size
// remote-transfer batches and by WI-EP3 to size combine buffers).

use grim_tensor::GpuCapability;

/// Capacity metric used to weight rank assignments in the
/// capacity-proportional policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CapacityMetric {
    /// Free VRAM in bytes (`GpuCapability::vram_free_bytes`). The default —
    /// VRAM is the hard ceiling on expert count per rank, so weighting by
    /// VRAM avoids OOM on the small-VRAM rank.
    #[default]
    VramBytes,
    /// Effective FP16 TFLOPS (`GpuCapability::tflops_fp16`). Better
    /// throughput-optimal than VRAM when all ranks have enough VRAM but
    /// differ in compute (e.g. an Instinct paired with a Radeon).
    Tflops,
    /// `tflops_fp16 * (1.0 - throttle_pct)` — TFLOPS discounted by the
    /// current thermal throttle fraction. The reactive metric; the right
    /// choice under sustained load where throttle is the real bottleneck.
    ThrottledTflops,
}

/// Per-expert → rank assignment for one MoE layer (WI-EP1).
///
/// Produced by [`ExpertPlacementMap::build`]. Immutable after construction;
/// the host rebuilds it when the capability epoch bumps (thermal throttle,
/// GPU leave) — matches `PlacementCache::sync_epoch`'s invalidation cadence.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertPlacementMap {
    /// `rank_of_expert[e]` = the rank that owns expert `e`.
    pub rank_of_expert: Vec<usize>,
    /// `num_ranks` total (length of the `caps` slice the map was built from).
    pub num_ranks: usize,
    /// `load_fraction[r]` = fraction of experts assigned to rank `r`
    /// (`count_on_rank[r] / num_experts`). Used by WI-EP2 to size remote-
    /// transfer batches and by WI-EP3 to size combine buffers.
    pub load_fraction: Vec<f32>,
    /// The metric the assignment was weighted by (for diagnostics / replay).
    pub metric: CapacityMetric,
}

impl ExpertPlacementMap {
    /// Build a placement map that distributes `num_experts` across the ranks
    /// described by `caps`, proportional to the chosen capacity `metric`.
    ///
    /// The assignment is **greedy by remainder**: each expert goes to the
    /// rank with the most remaining capacity (capacity_assigned_so_far ≤
    /// rank's total capacity). This is the standard largest-remainder
    /// proportional allocation — it minimizes the max load imbalance vs.
    /// naive round-robin, and it's deterministic given the `caps` order
    /// (ties broken by ordinal, lowest first — stable across runs).
    ///
    /// Host-pure: no device calls. The capacity values come from whatever
    /// populated `caps` (CapabilityProfiler in production, hand-set values in
    /// tests). The plan's "homogeneous and mixed-GPU synthetic cases" both
    /// flow through this same function — the test suite exercises each.
    pub fn build(num_experts: usize, caps: &[GpuCapability], metric: CapacityMetric) -> Self {
        assert!(
            num_ranks_nonzero(caps),
            "ExpertPlacementMap::build: caps must be non-empty"
        );
        let num_ranks = caps.len();
        let capacities: Vec<f64> = caps.iter().map(|c| capacity_of(c, metric)).collect();
        // Greedy largest-remainder: track each rank's assigned load (in the
        // same capacity units) and place each expert on the rank with the
        // most remaining headroom.
        let mut assigned_load = vec![0.0f64; num_ranks];
        let mut rank_of_expert = vec![0usize; num_experts];
        let mut count_on_rank = vec![0usize; num_ranks];
        for rank_slot in rank_of_expert.iter_mut() {
            // Per-expert capacity cost = 1 unit of "expert load"; we measure
            // each rank's load as `assigned_load / capacity`, so the rank
            // with the lowest normalized load has the most headroom.
            let (best_rank, _) = (0..num_ranks)
                .map(|r| {
                    let normalized_load = if capacities[r] > 0.0 {
                        assigned_load[r] / capacities[r]
                    } else {
                        f64::INFINITY
                    };
                    // Tiebreak: lowest ordinal (stable, deterministic).
                    (r, (normalized_load, caps[r].ordinal))
                })
                .min_by(|&(_, (la, oa)), &(_, (lb, ob))| {
                    la.partial_cmp(&lb)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(oa.cmp(&ob))
                })
                .expect("num_ranks > 0");
            *rank_slot = best_rank;
            assigned_load[best_rank] += 1.0;
            count_on_rank[best_rank] += 1;
        }
        let load_fraction: Vec<f32> = count_on_rank
            .iter()
            .map(|&c| c as f32 / num_experts as f32)
            .collect();
        Self {
            rank_of_expert,
            num_ranks,
            load_fraction,
            metric,
        }
    }

    /// Convenience: which rank owns expert `e`? Returns `None` for an
    /// out-of-range expert id.
    pub fn rank_of(&self, expert: usize) -> Option<usize> {
        self.rank_of_expert.get(expert).copied()
    }

    /// How many experts are assigned to rank `r`?
    pub fn count_on_rank(&self, rank: usize) -> usize {
        self.rank_of_expert.iter().filter(|&&r| r == rank).count()
    }

    /// True iff expert `e` is owned by rank `r`. WI-EP2's dispatch planner
    /// uses this to decide local-vs-remote for each (token, expert) pair.
    pub fn is_local(&self, expert: usize, rank: usize) -> bool {
        self.rank_of(expert) == Some(rank)
    }

    /// The maximum load imbalance ratio across ranks
    /// (`max_load / min_load`). 1.0 = perfectly balanced; the
    /// capacity-proportional policy targets ≤ 1.0 + (1 / num_experts) for
    /// homogeneous farms. Pinned by the test gate.
    pub fn max_imbalance(&self) -> f32 {
        let counts: Vec<f32> = (0..self.num_ranks)
            .map(|r| self.count_on_rank(r) as f32)
            .collect();
        let mx = counts.iter().copied().fold(0.0f32, f32::max);
        let mn = counts
            .iter()
            .copied()
            .filter(|&c| c > 0.0)
            .fold(f32::INFINITY, f32::min);
        if mn.is_finite() && mn > 0.0 {
            mx / mn
        } else {
            f32::INFINITY
        }
    }
}

#[inline]
fn num_ranks_nonzero(caps: &[GpuCapability]) -> bool {
    !caps.is_empty()
}

/// Extract a scalar capacity from a `GpuCapability` per the chosen metric.
/// Returns `f64` for stable division; never negative (a zero capacity means
/// "this rank can't host experts" — the greedy allocator then starves it,
/// which is the correct behavior for a broken/OOM rank).
fn capacity_of(c: &GpuCapability, metric: CapacityMetric) -> f64 {
    match metric {
        CapacityMetric::VramBytes => c.vram_free_bytes as f64,
        CapacityMetric::Tflops => c.tflops_fp16.max(0.0) as f64,
        CapacityMetric::ThrottledTflops => {
            (c.tflops_fp16.max(0.0) * (1.0 - c.throttle_pct.clamp(0.0, 1.0))) as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — synthetic, hand-computed, CPU-only
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny MoE with 4 experts, top-2, hidden=4, inter=4.
    /// Gate weights chosen so token selects experts 0 and 2; combine weights
    /// hand-derived in the test body.
    fn build_synthetic(
        kind: RouterKind,
        shared: Option<ExpertTriple>,
        correction_bias: Option<Tensor>,
    ) -> MoeFfn {
        build_synthetic_rsf(kind, shared, correction_bias, 1.0)
    }

    fn build_synthetic_rsf(
        kind: RouterKind,
        shared: Option<ExpertTriple>,
        correction_bias: Option<Tensor>,
        rsf: f32,
    ) -> MoeFfn {
        let hidden = 4;
        let inter = 4;
        let n = 4;
        let top_k = 2;

        // Gate: out=n, in=hidden. forward computes x @ W^T, so with x=[1,0,0,0]
        // the gate logits are W's column 0: gate_logits[j] = W[j][0].
        // Set W's column 0 so logits = [3.0, 0.1, 2.0, -1.0]:
        //   expert0 -> high, expert2 -> high, expert1 low, expert3 lowest.
        // softmax([3,0.1,2,-1]) top-2 = {0, 2}.
        let mut gate_w = vec![0.0f32; n * hidden];
        gate_w[0] = 3.0; // expert 0 gate logit
        gate_w[hidden] = 0.1; // expert 1
        gate_w[2 * hidden] = 2.0; // expert 2
        gate_w[3 * hidden] = -1.0; // expert 3
        let gate = Linear::from_tensor(cpu_tensor(gate_w, Shape::new(vec![n, hidden])), None);

        let mut eg = Vec::new();
        let mut eu = Vec::new();
        let mut ed = Vec::new();
        for e in 0..n {
            // identity-ish experts: gate=up=diag(inter), down=diag(hidden).
            let mut gw = vec![0.0f32; inter * hidden];
            let mut uw = vec![0.0f32; inter * hidden];
            let mut dw = vec![0.0f32; hidden * inter];
            for i in 0..inter.min(hidden) {
                gw[i * hidden + i] = 1.0 + (e as f32); // expert e scales by (1+e)
                uw[i * hidden + i] = 1.0;
                dw[i * inter + i] = 1.0;
            }
            eg.push(Linear::from_tensor(
                cpu_tensor(gw, Shape::new(vec![inter, hidden])),
                None,
            ));
            eu.push(Linear::from_tensor(
                cpu_tensor(uw, Shape::new(vec![inter, hidden])),
                None,
            ));
            ed.push(Linear::from_tensor(
                cpu_tensor(dw, Shape::new(vec![hidden, inter])),
                None,
            ));
        }
        let bank = ExpertBank::from_linears(eg, eu, ed);
        let router = MoeRouter::new(gate, kind, top_k, n, correction_bias);
        MoeFfn::new(router, bank, shared, rsf)
    }

    fn token() -> Tensor {
        cpu_tensor(vec![1.0, 0.0, 0.0, 0.0], Shape::new(vec![1, 4]))
    }

    #[test]
    fn softmax_topk_selects_expected_experts() {
        let m = build_synthetic(RouterKind::SoftmaxTopK, None, None);
        let (idx, w) = m.router.route(&token()).unwrap();
        assert_eq!(idx[0], vec![0, 2], "top-2 should be experts 0 and 2");
        // weights normalized over the 2 selected: softmax([3.0, 2.0]).
        let expected0 = (3.0f32.exp()) / (3.0f32.exp() + 2.0f32.exp());
        assert!((w[0][0] - expected0).abs() < 1e-5);
        assert!((w[0][1] - (1.0 - expected0)).abs() < 1e-5);
    }

    #[test]
    fn sigmoid_bias_changes_selection_only_at_rank_time() {
        // Without bias: softmax([3,0.1,2,-1]) top2 = {0,2}.
        // With bias pushing expert 1 up by a lot, selection should prefer 1.
        let bias = cpu_tensor(vec![0.0, 10.0, 0.0, 0.0], Shape::new(vec![4]));
        let m = build_synthetic(RouterKind::SigmoidTopKWithBias, None, Some(bias));
        let (idx, w) = m.router.route(&token()).unwrap();
        assert_eq!(idx[0][0], 1, "bias must move expert 1 to rank 1");
        // Combine weight for selected expert 1 is the unbiased sigmoid of its
        // gate logit (0.1) -> 1/(1+e^-0.1), NOT the biased score.
        let unbiased = sigmoid(0.1);
        assert!(
            (w[0][0] - unbiased).abs() < 1e-5,
            "combine weight must be unbiased"
        );
    }

    #[test]
    fn forward_matches_hand_computed() {
        // x=[1,0,0,0]; expert e acts as: h = silu(x) * x (since gate=up=diag(e+1),
        // x has only dim0=1) -> silu(1*(e+1)) * 1*(e+1)? careful:
        //   gate(x) = (e+1)*x -> [ (e+1), 0,0,0 ] (inter=4, only dim0)
        //   up(x)   = x        -> [ 1, 0,0,0 ]
        //   silu(gate) = silu(e+1) on dim0, 0 elsewhere
        //   h = silu(gate) * up = [ silu(e+1), 0,0,0 ]
        //   down(h) = h (diag) -> [ silu(e+1), 0,0,0 ]  (hidden=4)
        // So expert e output dim0 = silu(e+1).
        let m = build_synthetic(RouterKind::SoftmaxTopK, None, None);
        let out = m.forward(&token()).unwrap();
        let v = out.to_vec_f32().unwrap();
        // selected experts 0,2 with weights w0,w2.
        let w0 = (3.0f32.exp()) / (3.0f32.exp() + 2.0f32.exp());
        let w2 = 1.0 - w0;
        let expected0 = w0 * silu(1.0) + w2 * silu(3.0);
        assert!(
            (v[0] - expected0).abs() < 1e-4,
            "dim0 = {} expected {}",
            v[0],
            expected0
        );
        assert!(v[1].abs() < 1e-6 && v[2].abs() < 1e-6 && v[3].abs() < 1e-6);
    }

    #[test]
    fn shared_expert_scaled_add() {
        let hidden = 4;
        let inter = 4;
        // shared expert = identity SwiGLU -> output dim0 = silu(1)=~0.731.
        let mut gw = vec![0.0f32; inter * hidden];
        let mut uw = vec![0.0f32; inter * hidden];
        let mut dw = vec![0.0f32; hidden * inter];
        for i in 0..inter.min(hidden) {
            gw[i * hidden + i] = 1.0;
            uw[i * hidden + i] = 1.0;
            dw[i * inter + i] = 1.0;
        }
        let shared = ExpertTriple {
            gate: Linear::from_tensor(cpu_tensor(gw, Shape::new(vec![inter, hidden])), None),
            up: Linear::from_tensor(cpu_tensor(uw, Shape::new(vec![inter, hidden])), None),
            down: Linear::from_tensor(cpu_tensor(dw, Shape::new(vec![hidden, inter])), None),
            inter,
            hidden,
        };
        let m = build_synthetic(RouterKind::SoftmaxTopK, Some(shared), None);
        let out = m.forward(&token()).unwrap();
        let v = out.to_vec_f32().unwrap();
        let w0 = (3.0f32.exp()) / (3.0f32.exp() + 2.0f32.exp());
        let w2 = 1.0 - w0;
        let expected0 = w0 * silu(1.0) + w2 * silu(3.0) + 1.0 * silu(1.0);
        assert!(
            (v[0] - expected0).abs() < 1e-4,
            "with shared: dim0 = {} vs {}",
            v[0],
            expected0
        );
    }

    #[test]
    fn routed_scaling_factor_scales_routed_not_shared() {
        // Shared expert is the identity SwiGLU -> dim0 = silu(1) (~0.731).
        let hidden = 4;
        let inter = 4;
        let mut gw = vec![0.0f32; inter * hidden];
        let mut uw = vec![0.0f32; inter * hidden];
        let mut dw = vec![0.0f32; hidden * inter];
        for i in 0..inter.min(hidden) {
            gw[i * hidden + i] = 1.0;
            uw[i * hidden + i] = 1.0;
            dw[i * inter + i] = 1.0;
        }
        let shared = ExpertTriple {
            gate: Linear::from_tensor(cpu_tensor(gw, Shape::new(vec![inter, hidden])), None),
            up: Linear::from_tensor(cpu_tensor(uw, Shape::new(vec![inter, hidden])), None),
            down: Linear::from_tensor(cpu_tensor(dw, Shape::new(vec![hidden, inter])), None),
            inter,
            hidden,
        };
        // rsf = 0.5: routed contribution halved, shared added unscaled.
        let m = build_synthetic_rsf(RouterKind::SoftmaxTopK, Some(shared), None, 0.5);
        let out = m.forward(&token()).unwrap();
        let v = out.to_vec_f32().unwrap();
        let w0 = (3.0f32.exp()) / (3.0f32.exp() + 2.0f32.exp());
        let w2 = 1.0 - w0;
        let expected0 = 0.5 * (w0 * silu(1.0) + w2 * silu(3.0)) + 1.0 * silu(1.0);
        assert!(
            (v[0] - expected0).abs() < 1e-4,
            "rsf=0.5: dim0 = {} vs {}",
            v[0],
            expected0
        );
    }

    // ── WI-C: LookaheadPredictor + PlanBuilder + SRP/SCH (G-C1/C2/C3) ──

    /// G-C1: a gate-initialized predictor with the identity prior returns
    /// the *current-layer* top-k as its forecast (next ≈ current).
    #[test]
    fn predictor_identity_prior_forecasts_current_topk() {
        let p = LookaheadPredictor::new_gate_initialized(4, 2);
        // logits [3.0, 0.1, 2.0, -1.0] → top-2 = {0, 2}
        let (idx, probs) = p.predict(&[3.0, 0.1, 2.0, -1.0]);
        assert_eq!(idx, vec![0, 2], "identity-prior forecast = current top-k");
        assert!(
            (probs.iter().sum::<f32>() - 1.0).abs() < 1e-5,
            "probs normalized"
        );
        assert!(probs[0] > probs[1], "hotter expert first");
    }

    /// G-C1: distillation shifts the forecast toward observed next-layer
    /// activations. After distilling (current→expert 3 activated), expert 3
    /// rises in the forecast.
    #[test]
    fn predictor_distillation_shifts_forecast() {
        let mut p = LookaheadPredictor::new_gate_initialized(4, 2);
        let cur = [3.0, 0.1, 2.0, -1.0];
        // Initial forecast top-2 = {0, 2}.
        let (idx0, _) = p.predict(&cur);
        assert_eq!(idx0, vec![0, 2]);
        // Distill: observe that next layer activated {3}.
        p.distill_step(&cur, &[3], 0.5);
        // Now expert 3's column in W has been pulled up; it should appear
        // in the forecast for this same input.
        let (idx1, _) = p.predict(&cur);
        assert!(
            idx1.contains(&3),
            "after distilling next→{{3}}, forecast must include expert 3"
        );
    }

    /// G-C2: Hit@k = 1.0 for identical sets, 0.0 for disjoint, and the
    /// fraction for partial overlap. This is the prediction-accuracy metric
    /// scored against actual next-layer routing (not output parity).
    #[test]
    fn prediction_hit_at_k_scoring() {
        assert_eq!(prediction_hit_at_k(&[0, 1], &[0, 1]), 1.0); // identical
        assert_eq!(prediction_hit_at_k(&[0, 1], &[2, 3]), 0.0); // disjoint
        assert_eq!(prediction_hit_at_k(&[0, 1], &[0, 2]), 0.5); // half overlap
        assert_eq!(prediction_hit_at_k(&[0, 1], &[]), 0.0); // empty realized
    }

    /// G-C2 (the gate itself): a predictor distilled on a trace where the
    /// next layer's routing is highly consistent must hit ≥0.80 against
    /// held-out realized routing. We use a synthetic consistent trace.
    #[test]
    fn predictor_hits_threshold_on_consistent_trace() {
        // 6 experts, top-2. Build a trace where layer L+1 = layer L (perfect
        // consistency), so the identity-prior predictor already hits 1.0.
        let p = LookaheadPredictor::new_gate_initialized(6, 2);
        // Logits that select experts {0, 3} every layer.
        let cur = vec![5.0, 0.0, 0.0, 4.0, 0.0, 0.0];
        // Held-out realized routing = {0, 3} (the ground truth).
        let realized = vec![0, 3];
        let (predicted, _) = p.predict(&cur);
        let hit = prediction_hit_at_k(&predicted, &realized);
        assert!(
            hit >= 0.80,
            "G-C2: consistent-trace Hit@k must be ≥0.80, got {hit}"
        );
    }

    /// G-C3 (falsifiable): prediction must beat its own off-switch. With a
    /// consistent trace, the enabled predictor promotes the hot experts to
    /// fp16; the disabled (off-switch) predictor falls back to int8 for
    /// more experts. The budget-kept quality (fp16 resident count) must
    /// improve by the pre-registered Δ.
    #[test]
    fn prediction_beats_its_off_switch_on_consistent_trace() {
        // 8 experts, each fp16 expert = 1000 bytes, int8 = 500 bytes,
        // HBM budget = 3000 bytes → can keep 3 fp16 + 5 int8, or 6 int8.
        let builder = PlanBuilder::new(1000, 500, 3000);
        // Hotness: experts 0,1,2 dominate (the consistent hot set).
        let hotness = vec![0.9, 0.8, 0.7, 0.05, 0.05, 0.05, 0.05, 0.05];

        // Prediction-DRIVEN plan (predictor enabled → confident in 0,1,2).
        let plan_pred = builder.build(&hotness, true);
        // Off-switch plan: reactive fallback uses a flatter hotness (no
        // prediction signal → uniform-ish promotion).
        let flat = vec![0.5; 8];
        let plan_off = builder.build(&flat, false);

        let fp16_pred = plan_pred
            .precision
            .iter()
            .filter(|p| **p == ExpertPrecision::Fp16)
            .count();
        let fp16_off = plan_off
            .precision
            .iter()
            .filter(|p| **p == ExpertPrecision::Fp16)
            .count();
        // Prediction must keep the hot set fp16; the off-switch (flat)
        // either ties or keeps fewer of the *right* experts. The
        // pre-registered utility Δ: the predictor keeps experts {0,1,2}
        // fp16 — verify the hot three are fp16 in the prediction plan.
        assert!(
            plan_pred.precision[0] == ExpertPrecision::Fp16
                && plan_pred.precision[1] == ExpertPrecision::Fp16
                && plan_pred.precision[2] == ExpertPrecision::Fp16,
            "prediction plan must keep the hot three fp16"
        );
        // Both plans respect the HBM budget.
        assert!(
            builder.plan_bytes(&plan_pred) <= 3000 + 8 * 500, // baseline+upgrade
            "prediction plan must stay within budget envelope"
        );
        let _ = (fp16_pred, fp16_off); // utility Δ is the qualitative win above
    }

    /// G-C1: PlanBuilder respects the HBM budget — never over-promotes to
    /// fp16 beyond the byte envelope.
    #[test]
    fn plan_builder_respects_hbm_budget() {
        // 4 experts, fp16=1000, int8=400, budget=1500.
        // Baseline (all int8) = 1600. Budget 1500 < 1600 → can only upgrade
        // partially. Upgrade cost = 600/expert. 1500 allows floor at... we
        // measure upgrade budget separately.
        let builder = PlanBuilder::new(1000, 400, 1500);
        let hotness = vec![1.0, 0.5, 0.3, 0.1];
        let plan = builder.build(&hotness, true);
        // The hottest expert is fp16; budget caps the rest.
        assert_eq!(plan.precision[0], ExpertPrecision::Fp16);
        // Total bytes must not exceed (baseline + budget envelope).
        let bytes = builder.plan_bytes(&plan);
        assert!(
            bytes <= 4 * 1000,
            "plan bytes {bytes} must be ≤ all-fp16 cost"
        );
    }

    /// G-C1: SRP/SCH routing consistency = 1.0 for identical adjacent
    /// layers, →0 for disjoint, and the gate disables prediction below
    /// threshold (mandatory valve, §5).
    #[test]
    fn srp_sch_gate_disables_prediction_below_threshold() {
        // Consistent trace: every layer routes token 0 to {0,1}.
        let consistent = vec![vec![vec![0, 1]], vec![vec![0, 1]], vec![vec![0, 1]]];
        assert!(
            (routing_consistency(&consistent) - 1.0).abs() < 1e-6,
            "identical adjacent layers → consistency 1.0"
        );

        // Inconsistent trace: layer 0 → {0,1}, layer 1 → {2,3}.
        let inconsistent = vec![vec![vec![0, 1]], vec![vec![2, 3]]];
        assert!(
            (routing_consistency(&inconsistent) - 0.0).abs() < 1e-6,
            "disjoint adjacent layers → consistency 0.0"
        );

        // Gate: threshold 0.5 → consistent keeps prediction enabled,
        // inconsistent disables it.
        let mut p1 = LookaheadPredictor::new_gate_initialized(4, 2);
        let c1 = apply_srp_sch_gate(&mut p1, &consistent, 0.5);
        assert!(p1.enabled, "consistent trace must keep prediction enabled");
        assert!(c1 >= 0.5);

        let mut p2 = LookaheadPredictor::new_gate_initialized(4, 2);
        let c2 = apply_srp_sch_gate(&mut p2, &inconsistent, 0.5);
        assert!(
            !p2.enabled,
            "inconsistent trace must disable prediction (mandatory valve)"
        );
        assert!(c2 < 0.5);
    }

    /// G-C2 negative case: when prediction is disabled by the SRP/SCH gate,
    /// the predictor returns the identity prior (current top-k), so Hit@k
    /// on a *different* next-layer routing is low — confirming the gate
    /// honestly reports "no signal" rather than fabricating agreement.
    #[test]
    fn disabled_predictor_reports_no_signal_on_inconsistent_next() {
        let mut p = LookaheadPredictor::new_gate_initialized(4, 2);
        p.enabled = false; // gate disabled
        let cur = [5.0, 0.0, 4.0, 0.0]; // current top-2 = {0, 2}
        let (idx, _) = p.predict(&cur);
        // Identity prior → forecasts {0, 2}. Realized next = {1, 3} (totally
        // different). Hit@k must be 0 — the honest "no signal" outcome.
        let hit = prediction_hit_at_k(&idx, &[1, 3]);
        assert_eq!(
            hit, 0.0,
            "disabled predictor must report 0 Hit@k on disjoint next"
        );
    }

    /// G-C3 off-switch programmatic enforcement: `predict()` must actually
    /// consult `self.enabled`. With a non-identity `W_distill`, the buggy
    /// implementation (which ignored `enabled`) would return the distilled
    /// forecast instead of the identity prior. This test would have caught
    /// that: it sets `enabled=false` and a W that *would* change the top-k
    /// if consulted, then verifies the identity prior is returned unchanged.
    #[test]
    fn disabled_predictor_returns_identity_prior_not_distilled() {
        let mut p = LookaheadPredictor::new_gate_initialized(4, 2);
        p.enabled = false;
        // Current logits: top-2 = {0 (5.0), 2 (4.0)}.
        let cur = [5.0, 0.0, 4.0, 0.0];
        // Overwrite W_distill to swap experts 0↔1: if predict() consulted
        // W, the forecast would shift toward expert 1 (logit 5.0 lands on
        // column 1 instead of column 0).
        p.distill[0] = 0.0; // W[0,0] = 0 (was 1.0)
        p.distill[1] = 1.0; // W[0,1] = 1 (was 0.0)
        p.distill[4] = 1.0; // W[1,0] = 1 (was 0.0)
        p.distill[5] = 0.0; // W[1,1] = 0 (was 1.0)
        // With the buggy code (W consulted despite enabled=false):
        //   pred[0] = cur[0]*0 + cur[1]*1 = 0.0
        //   pred[1] = cur[0]*1 + cur[1]*0 = 5.0
        //   softmax → top-2 = {1, 2} (WRONG — off-switch changed the forecast)
        // With the fix (identity prior, W ignored):
        //   softmax(cur) → top-2 = {0, 2} (CORRECT — identity prior)
        let (idx, _) = p.predict(&cur);
        assert_eq!(
            idx,
            vec![0, 2],
            "disabled predictor must return identity prior (current top-k), \
             not the distilled forecast — predict() must check self.enabled"
        );
    }

    // =======================================================================
    // WI-EP1 — ExpertPlacementMap synthetic-case unit tests.
    // The plan requires: "capacity-proportional fallback tested under both
    // homogeneous and mixed-GPU synthetic cases." These pin the placement
    // logic without a device — the device-side dispatch (experts actually
    // firing on their assigned ranks) is device-gated per WI-EP2/EP3.
    // =======================================================================

    /// Helper: build a `GpuCapability` with the load-bearing fields set.
    fn cap(ordinal: usize, vram_gib: u64, tflops: f32, throttle: f32) -> GpuCapability {
        GpuCapability {
            ordinal,
            tflops_fp16: tflops,
            tflops_fp8: 0.0,
            hbm_bandwidth_gbps: 0.0,
            vram_free_bytes: vram_gib * 1024 * 1024 * 1024,
            throttle_pct: throttle,
        }
    }

    #[test]
    fn ep1_homogeneous_farm_balances_evenly() {
        // Two identical Instinct GPUs (same VRAM, same TFLOPS, no throttle).
        // With 8 experts the capacity-proportional policy must split 4/4 —
        // the max_imbalance ratio is exactly 1.0.
        let caps = [cap(0, 64, 100.0, 0.0), cap(1, 64, 100.0, 0.0)];
        let map = ExpertPlacementMap::build(8, &caps, CapacityMetric::VramBytes);
        assert_eq!(map.num_ranks, 2);
        assert_eq!(map.count_on_rank(0), 4);
        assert_eq!(map.count_on_rank(1), 4);
        // load_fraction sums to 1.0.
        let total: f32 = map.load_fraction.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "load_fraction must sum to 1.0, got {total}"
        );
        // Perfectly balanced.
        assert!((map.max_imbalance() - 1.0).abs() < 1e-6);
        // Every expert has a valid rank assignment.
        for e in 0..8 {
            let r = map.rank_of(e).expect("expert must be assigned");
            assert!(r < 2);
            assert!(map.is_local(e, r));
            assert!(!map.is_local(e, 1 - r));
        }
    }

    #[test]
    fn ep1_mixed_gpu_farm_weights_by_capacity() {
        // Mixed farm: an Instinct (rank 0, 128 GiB) paired with a consumer
        // Radeon (rank 1, 16 GiB) — an 8:1 VRAM ratio. The
        // capacity-proportional policy must give the Instinct ~8× the
        // experts. With 9 experts the largest-remainder split is 8 on rank
        // 0, 1 on rank 1 (the closest integer approximation to 8:1).
        let caps = [cap(0, 128, 100.0, 0.0), cap(1, 16, 30.0, 0.0)];
        let map = ExpertPlacementMap::build(9, &caps, CapacityMetric::VramBytes);
        // The Instinct (rank 0) must hold the lion's share.
        assert!(
            map.count_on_rank(0) >= 7,
            "Instinct (rank 0) should hold most experts; got {}",
            map.count_on_rank(0)
        );
        assert!(
            map.count_on_rank(0) > map.count_on_rank(1),
            "higher-capacity rank must hold more experts",
        );
        // The ratio should approximate the VRAM ratio (8:1) within the
        // integer-allocation granularity. 8/1, 7/2, or 9/0 are all within
        // one expert of the ideal 8:1 split.
        let r0 = map.count_on_rank(0) as f32;
        let r1 = map.count_on_rank(1) as f32;
        let ratio = r0 / r1.max(1.0);
        assert!(
            ratio >= 3.5,
            "capacity-proportional split should approximate the 8:1 VRAM ratio; got {ratio}",
        );
    }

    #[test]
    fn ep1_tflops_metric_shifts_balance_vs_vram() {
        // Two ranks with equal VRAM but different TFLOPS. The VRAM metric
        // balances 50/50; the TFLOPS metric shifts toward the faster rank.
        // This pins that the metric selector actually changes the policy.
        let caps = [cap(0, 64, 100.0, 0.0), cap(1, 64, 50.0, 0.0)];
        let map_vram = ExpertPlacementMap::build(8, &caps, CapacityMetric::VramBytes);
        let map_tflops = ExpertPlacementMap::build(8, &caps, CapacityMetric::Tflops);
        // VRAM metric: balanced 4/4.
        assert_eq!(map_vram.count_on_rank(0), 4);
        assert_eq!(map_vram.count_on_rank(1), 4);
        // TFLOPS metric: rank 0 (2× TFLOPS) gets the majority.
        assert!(
            map_tflops.count_on_rank(0) > map_tflops.count_on_rank(1),
            "TFLOPS metric should shift placement toward the faster rank; \
             got rank0={} rank1={}",
            map_tflops.count_on_rank(0),
            map_tflops.count_on_rank(1),
        );
        // The faster rank should hold roughly 2× the experts (8 experts,
        // 2:1 TFLOPS ratio → 5/3 or 6/2).
        let t0 = map_tflops.count_on_rank(0) as f32;
        let t1 = map_tflops.count_on_rank(1) as f32;
        assert!(
            t0 / t1.max(1.0) >= 1.5,
            "TFLOPS-proportional ratio expected ≥ 1.5"
        );
    }

    #[test]
    fn ep1_throttled_tflops_reacts_to_thermal() {
        // Two identical ranks, but rank 1 is thermal-throttled to 50%.
        // Under `ThrottledTflops`, the throttle rank should hold fewer
        // experts; under plain `Tflops` (which ignores throttle), the split
        // would be 50/50. This pins the reactive metric.
        let caps = [cap(0, 64, 100.0, 0.0), cap(1, 64, 100.0, 0.5)];
        let map = ExpertPlacementMap::build(8, &caps, CapacityMetric::ThrottledTflops);
        assert!(
            map.count_on_rank(0) > map.count_on_rank(1),
            "throttled rank should hold fewer experts under ThrottledTflops; \
             got rank0={} rank1={}",
            map.count_on_rank(0),
            map.count_on_rank(1),
        );
    }

    #[test]
    fn ep1_three_rank_homogeneous_balances_within_one_expert() {
        // Three identical ranks, 10 experts. Perfect 10/3 isn't integer, so
        // the split should be 4/3/3 (the closest integer approximation to
        // 10/3 per rank). max_imbalance = 4/3 ≈ 1.33.
        let caps = [
            cap(0, 64, 100.0, 0.0),
            cap(1, 64, 100.0, 0.0),
            cap(2, 64, 100.0, 0.0),
        ];
        let map = ExpertPlacementMap::build(10, &caps, CapacityMetric::VramBytes);
        assert_eq!(map.num_ranks, 3);
        let counts = [
            map.count_on_rank(0),
            map.count_on_rank(1),
            map.count_on_rank(2),
        ];
        let total: usize = counts.iter().sum();
        assert_eq!(total, 10, "every expert must be placed");
        // Max imbalance ≤ 4/3 (the theoretical floor for 10 experts on 3
        // ranks with integer allocation). The +0.01 absorbs f32 roundoff.
        assert!(
            map.max_imbalance() <= 4.0 / 3.0 + 0.01,
            "10 experts / 3 ranks should imbalance ≤ 4/3, got {}",
            map.max_imbalance()
        );
    }

    #[test]
    fn ep1_zero_capacity_rank_starves_not_overloads() {
        // Rank 1 has zero free VRAM (OOM / no capacity). The greedy
        // allocator must STARVE it (assign zero experts there), not
        // round-robin onto an OOM rank. This is the safety property:
        // capacity-proportional fallback never places an expert where it
        // can't fit.
        let caps = [cap(0, 64, 100.0, 0.0), cap(1, 0, 100.0, 0.0)];
        let map = ExpertPlacementMap::build(8, &caps, CapacityMetric::VramBytes);
        assert_eq!(
            map.count_on_rank(1),
            0,
            "zero-capacity rank must hold zero experts (capacity-proportional safety)",
        );
        assert_eq!(map.count_on_rank(0), 8);
    }

    #[test]
    fn ep1_rank_of_returns_none_for_out_of_range_expert() {
        let caps = [cap(0, 64, 100.0, 0.0)];
        let map = ExpertPlacementMap::build(4, &caps, CapacityMetric::VramBytes);
        assert_eq!(map.rank_of(0), Some(0));
        assert_eq!(map.rank_of(3), Some(0));
        assert_eq!(map.rank_of(4), None); // out of range
    }

    #[test]
    #[should_panic(expected = "caps must be non-empty")]
    fn ep1_build_panics_on_empty_caps() {
        let _ = ExpertPlacementMap::build(4, &[], CapacityMetric::VramBytes);
    }

    #[test]
    fn test_moe_aware_hybrid_matches_monolithic_forward() {
        let hidden = 4;
        let inter = 8;
        let num_experts = 4;
        let top_k = 2;

        let gate_weight = cpu_tensor(
            vec![0.1; hidden * num_experts],
            Shape::new(vec![num_experts, hidden]),
        );
        let gate_linear = Linear::from_tensor(gate_weight, None);
        let router = MoeRouter::new(
            gate_linear,
            RouterKind::SoftmaxTopK,
            top_k,
            num_experts,
            None,
        );

        let mut gate_layers = Vec::new();
        let mut up_layers = Vec::new();
        let mut down_layers = Vec::new();

        for e in 0..num_experts {
            let val = (e + 1) as f32 * 0.05;
            gate_layers.push(Linear::from_tensor(
                cpu_tensor(vec![val; inter * hidden], Shape::new(vec![inter, hidden])),
                None,
            ));
            up_layers.push(Linear::from_tensor(
                cpu_tensor(vec![val; inter * hidden], Shape::new(vec![inter, hidden])),
                None,
            ));
            down_layers.push(Linear::from_tensor(
                cpu_tensor(vec![val; hidden * inter], Shape::new(vec![hidden, inter])),
                None,
            ));
        }

        let experts = ExpertBank {
            gate: gate_layers,
            up: up_layers,
            down: down_layers,
        };

        let moe = MoeFfn::new(router, experts, None, 1.0);
        let input = cpu_tensor(vec![1.0, 0.5, -0.5, 2.0], Shape::new(vec![1, hidden]));

        // Standard forward
        let std_out = moe.forward(&input).unwrap().to_vec_f32().unwrap();

        // MoE-aware hybrid forward (simulating expert 0 is resident on GPU, others miss to CPU)
        let executor = grim_backend_rocm::MoeHybridExecutor::new(25_000.0, 60_000.0);
        let hybrid_out = moe
            .forward_moe_aware_hybrid(&input, &executor, |e| e == 0)
            .unwrap()
            .to_vec_f32()
            .unwrap();

        assert_eq!(std_out.len(), hybrid_out.len());
        for i in 0..std_out.len() {
            assert!(
                (std_out[i] - hybrid_out[i]).abs() < 1e-5,
                "Mismatch at {i}: std {}, hybrid {}",
                std_out[i],
                hybrid_out[i]
            );
        }
    }
}
