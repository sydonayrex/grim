//! RWKV RNN family — Time-Mix & Channel-Mix recurrent layers.

use crate::cpu_tensor;
use grim_backend_cpu::add_tensors;
use grim_core::error::{Error, Result};
use grim_core::model::{
    AdapterHandle, CausalLm, ModalityHint, Model, ModelConfig, SsmState, StatefulSequence,
};
use grim_nn::{Linear, RmsNorm};
use grim_tensor::{ArithType, Device, Shape, Tensor};
use std::any::Any;

/// Validate that a loaded weight tensor has the expected shape, returning a
/// descriptive `Error::Shape` when it does not.
fn assert_weight_shape(tensor: &Tensor, expected: &[usize], name: &str) -> Result<()> {
    let dims = tensor.shape().dims();
    if dims != expected {
        return Err(Error::Shape(format!(
            "RWKV weight '{}' has shape {:?}, expected {:?}",
            name, dims, expected
        )));
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct RwkvConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub rms_norm_eps: f64,
}

impl Default for RwkvConfig {
    fn default() -> Self {
        Self {
            vocab_size: 0,
            hidden_size: 0,
            num_layers: 0,
            rms_norm_eps: 1e-5,
        }
    }
}

impl ModelConfig for RwkvConfig {
    fn name(&self) -> &str {
        "rwkv"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Per-session RWKV recurrence state (batch = 1).
///
/// Layout: `data[layer][slot][channel]` flattened, slot order
/// `[attn_xx, attn_aa, attn_bb, attn_pp, ffn_xx]` — the RWKV-4 five-buffer
/// state: token-shift carries (`*_xx`) and the WKV numerator/denominator/max
/// triples (`aa`/`bb`/`pp`).
#[derive(Clone, Debug)]
pub struct RwkvState {
    pub data: Vec<f32>,
    pub num_layers: usize,
    pub hidden: usize,
}

/// Slot stride per layer.
const RWKV_SLOTS_PER_LAYER: usize = 5;

impl RwkvState {
    fn layer_offset(&self, layer: usize) -> usize {
        layer * RWKV_SLOTS_PER_LAYER * self.hidden
    }

    /// Mutable split of one layer's five slots (bounds-checked).
    fn layer_slots_mut(&mut self, layer: usize) -> Result<[&mut [f32]; 5]> {
        let hidden = self.hidden;
        let base = self.layer_offset(layer);
        let end = base + RWKV_SLOTS_PER_LAYER * hidden;
        if end > self.data.len() {
            return Err(Error::Session(format!(
                "RWKV state too small: {} elements, need {}",
                self.data.len(),
                end
            )));
        }
        let s = &mut self.data[base..end];
        // Disjoint split (split_at_mut chains avoid aliasing borrows):
        // slots are [xx_attn | aa | bb | pp | xx_ffn].
        let (s01, rest) = s.split_at_mut(2 * hidden);
        let (s0, s1) = s01.split_at_mut(hidden);
        let (s23, s4) = rest.split_at_mut(2 * hidden);
        let (s2, s3) = s23.split_at_mut(hidden);
        Ok([s0, s1, s2, s3, s4])
    }
}

impl SsmState for RwkvState {
    fn clone_snapshot(&self) -> Result<Box<dyn SsmState>> {
        Ok(Box::new(self.clone()))
    }
    fn restore_snapshot(&mut self, snap: &dyn SsmState) -> Result<()> {
        let other = snap
            .as_any()
            .downcast_ref::<RwkvState>()
            .ok_or_else(|| Error::Session("downcast failed".into()))?;
        if other.data.len() != self.data.len() {
            return Err(Error::Session("RWKV state size mismatch".into()));
        }
        self.data.copy_from_slice(&other.data);
        Ok(())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

pub struct RwkvBlock {
    pub norm: RmsNorm,
    /// Second block layernorm (v4 `ln_2`, applied before the channel mix).
    /// Falls back to a unit norm when the checkpoint lacks it.
    pub norm2: RmsNorm,
    pub time_mix_key: Linear,
    pub time_mix_value: Linear,
    pub time_mix_receptance: Linear,
    pub time_mix_output: Linear,
    pub channel_mix_key: Linear,
    pub channel_mix_receptance: Linear,
    pub channel_mix_value: Linear,
    pub device: Device,
    /// RWKV-4 recurrence parameters, `[hidden]` each. Loaded from real
    /// checkpoints (`att.time_mix_k/r`, `att.time_decay`, `att.time_first`,
    /// `ffn.time_mix_k/r`); synthetic/test models get neutral defaults so
    /// the recurrence still runs (documented in the audit).
    pub tm_mix_k: Vec<f32>,
    pub tm_mix_v: Vec<f32>,
    pub tm_mix_r: Vec<f32>,
    pub time_decay: Vec<f32>,
    pub time_first: Vec<f32>,
    pub ffn_tm_mix_k: Vec<f32>,
    pub ffn_tm_mix_r: Vec<f32>,
}

impl RwkvBlock {
    pub fn load(ws: &grim_nn::WeightSource<'_>, cfg: &RwkvConfig, device: Device) -> Result<Self> {
        let norm = RmsNorm::load(&ws.pp("ln_x"), cfg.hidden_size, cfg.rms_norm_eps as f32)?;
        let time_mix_key =
            Linear::load(&ws.pp("att.key"), cfg.hidden_size, cfg.hidden_size, false)?;
        let time_mix_value =
            Linear::load(&ws.pp("att.value"), cfg.hidden_size, cfg.hidden_size, false)?;
        let time_mix_receptance = Linear::load(
            &ws.pp("att.receptance"),
            cfg.hidden_size,
            cfg.hidden_size,
            false,
        )?;
        let time_mix_output = Linear::load(
            &ws.pp("att.output"),
            cfg.hidden_size,
            cfg.hidden_size,
            false,
        )?;

        let channel_mix_key =
            Linear::load(&ws.pp("ffn.key"), cfg.hidden_size, cfg.hidden_size, false)?;
        let channel_mix_receptance = Linear::load(
            &ws.pp("ffn.receptance"),
            cfg.hidden_size,
            cfg.hidden_size,
            false,
        )?;
        let channel_mix_value =
            Linear::load(&ws.pp("ffn.value"), cfg.hidden_size, cfg.hidden_size, false)?;

        assert_weight_shape(&norm.weight, &[cfg.hidden_size], "ln_x")?;
        assert_weight_shape(
            &time_mix_key.weight,
            &[cfg.hidden_size, cfg.hidden_size],
            "att.key",
        )?;
        assert_weight_shape(
            &time_mix_value.weight,
            &[cfg.hidden_size, cfg.hidden_size],
            "att.value",
        )?;
        assert_weight_shape(
            &time_mix_receptance.weight,
            &[cfg.hidden_size, cfg.hidden_size],
            "att.receptance",
        )?;
        assert_weight_shape(
            &time_mix_output.weight,
            &[cfg.hidden_size, cfg.hidden_size],
            "att.output",
        )?;
        assert_weight_shape(
            &channel_mix_key.weight,
            &[cfg.hidden_size, cfg.hidden_size],
            "ffn.key",
        )?;
        assert_weight_shape(
            &channel_mix_receptance.weight,
            &[cfg.hidden_size, cfg.hidden_size],
            "ffn.receptance",
        )?;
        assert_weight_shape(
            &channel_mix_value.weight,
            &[cfg.hidden_size, cfg.hidden_size],
            "ffn.value",
        )?;

        // RWKV-4 recurrence parameters. Real checkpoints carry them; absent
        // tensors (synthetic models) get neutral defaults — mixes of 0.5 and
        // a mild decay/first pair — so the state threading still exercises.
        let vec_param = |name: &str, default: f32| -> Vec<f32> {
            ws.get(Shape::new(vec![cfg.hidden_size]), name)
                .map(|t| {
                    t.to_vec_f32()
                        .unwrap_or_else(|_| vec![default; cfg.hidden_size])
                })
                .unwrap_or_else(|_| vec![default; cfg.hidden_size])
        };
        let tm_mix_k = vec_param("att.time_mix_k", 0.5);
        let tm_mix_v = vec_param("att.time_mix_v", 0.5);
        let tm_mix_r = vec_param("att.time_mix_r", 0.5);
        // Both buffers are stored RAW in log space by v4 checkpoints
        // (decay is ADDED to the running max between tokens; first is added
        // to the current key inside the step). Defaults are mild synthetic
        // values, not transforms of real ones.
        let time_decay = vec_param("att.time_decay", -0.6);
        let time_first = vec_param("att.time_first", 3.0);
        let ffn_tm_mix_k = vec_param("ffn.time_mix_k", 0.5);
        let ffn_tm_mix_r = vec_param("ffn.time_mix_r", 0.5);
        let norm2 = RmsNorm::load(&ws.pp("ln_2"), cfg.hidden_size, cfg.rms_norm_eps as f32)
            .unwrap_or(RmsNorm {
                weight: cpu_tensor(
                    vec![1.0f32; cfg.hidden_size],
                    Shape::new(vec![cfg.hidden_size]),
                ),
                eps: cfg.rms_norm_eps as f32,
            });

        Ok(Self {
            norm,
            norm2,
            time_mix_key,
            time_mix_value,
            time_mix_receptance,
            time_mix_output,
            channel_mix_key,
            channel_mix_receptance,
            channel_mix_value,
            device,
            tm_mix_k,
            tm_mix_v,
            tm_mix_r,
            time_decay,
            time_first,
            ffn_tm_mix_k,
            ffn_tm_mix_r,
        })
    }

    /// Forward ONE token through this block, threading the RWKV-4 recurrence
    /// state.
    ///
    /// Audit fix (grim-models): the previous implementation ignored its state
    /// parameter entirely — the time-mix was a memoryless elementwise
    /// `sigmoid(r)·k·v`, so served RWKV had no context after prefill. This is
    /// the canonical v4 single-token recurrence:
    ///
    /// * attention token-shift: k/v/r project `tm·x + (1-tm)·xx_prev`
    ///   (xx_prev = previous token's post-norm hidden);
    /// * WKV one-token update per channel:
    ///   `ww = u + k; p = max(pp, k); e1 = exp(pp-p); e2 = exp(ww-p);
    ///    y = (e1·aa + e2·v)/(e1·bb + e2); aa' = num; bb' = den;
    ///    pp' = p + w` (w = time_decay, log-space);
    /// * channel-mix token-shift with ReLU on the mixed key.
    ///
    /// The ROCm `rwkv_time_mix` kernel dispatch was REMOVED, not just
    /// bypassed: its signature has no state I/O, so it can never implement
    /// this recurrence — using it produced silently context-free output. The
    /// kernels remain in the backend for future state-aware wiring.
    pub fn step(&self, x: &Tensor, layer_idx: usize, state: &mut RwkvState) -> Result<Tensor> {
        let dim = self.cfg_hidden();
        let x_vec = x.to_vec_f32()?;
        if x_vec.len() != dim {
            return Err(Error::Shape(format!(
                "RWKV block step expects [{dim}] hidden, got {}",
                x_vec.len()
            )));
        }
        let [attn_xx, aa, bb, pp, ffn_xx] = state.layer_slots_mut(layer_idx)?;

        // ── Attention (time-mix) ─────────────────────────────────────────
        // Token shift against the stored PREVIOUS post-norm hidden.
        let mut shifted = vec![0.0f32; 3 * dim];
        for i in 0..dim {
            let mix_k = self.tm_mix_k[i];
            let mix_v = self.tm_mix_v[i];
            let mix_r = self.tm_mix_r[i];
            shifted[i] = mix_k * x_vec[i] + (1.0 - mix_k) * attn_xx[i];
            shifted[dim + i] = mix_v * x_vec[i] + (1.0 - mix_v) * attn_xx[i];
            shifted[2 * dim + i] = mix_r * x_vec[i] + (1.0 - mix_r) * attn_xx[i];
        }
        // Post-norm current hidden becomes the next call's shift carry.
        let norm_x = self.norm.forward(x)?.to_vec_f32()?;
        attn_xx.copy_from_slice(&norm_x);

        let mixed = cpu_tensor(shifted.clone(), Shape::new(vec![3, dim]));
        // Audit fix (grim-models, found by the WKV numeric reference test):
        // `flat` documents `off` as a ROW of the [3, dim] mixed tensor, but
        // sliced by ELEMENT offset — so the value projection read elements
        // 1..dim+1 and the receptance projection 2..dim+2, mixing channels
        // across the k/v/r boundaries (silently wrong attention output for
        // every RWKV checkpoint). Slice by row: off * dim.
        let flat = |t: &Tensor, row: usize| -> Result<Vec<f32>> {
            Ok(t.to_vec_f32()?[row * dim..(row + 1) * dim].to_vec())
        };
        let k_t = self
            .time_mix_key
            .forward(&cpu_tensor(flat(&mixed, 0)?, Shape::new(vec![1, dim])))?;
        let v_t = self
            .time_mix_value
            .forward(&cpu_tensor(flat(&mixed, 1)?, Shape::new(vec![1, dim])))?;
        let r_t = self
            .time_mix_receptance
            .forward(&cpu_tensor(flat(&mixed, 2)?, Shape::new(vec![1, dim])))?;

        // One-token WKV update (see doc comment) + sigmoid(r) gate.
        let k_vec = k_t.to_vec_f32()?;
        let v_vec = v_t.to_vec_f32()?;
        let r_vec = r_t.to_vec_f32()?;
        let mut attn_y = vec![0.0f32; dim];
        for i in 0..dim {
            let k_ch = k_vec[i];
            let ww = self.time_first[i] + k_ch;
            let p = pp[i].max(k_ch);
            let e11 = (pp[i] - p).exp();
            let e22 = (ww - p).exp();
            let num = e11 * aa[i] + e22 * v_vec[i];
            let den = e11 * bb[i] + e22;
            attn_y[i] = {
                let sig_r = 1.0 / (1.0 + (-r_vec[i]).exp());
                sig_r * if den != 0.0 { num / den } else { 0.0 }
            };
            aa[i] = num;
            bb[i] = den;
            pp[i] = p + self.time_decay[i];
        }

        let att_out = self
            .time_mix_output
            .forward(&cpu_tensor(attn_y, Shape::new(vec![1, dim])))?;
        let x_res1 = add_tensors(x, &att_out).map_err(grim_core::Error::Tensor)?;

        // ── Channel mix (FFN) ────────────────────────────────────────────
        // v4: token shift against the stored PREVIOUS post-LN2 residual;
        // current residual goes through ln2; k passes ReLU after mixing, r
        // gates with sigmoid.
        let x_res1_normed = self.norm2.forward(&x_res1)?.to_vec_f32()?;
        let mut ffn_k_in = vec![0.0f32; dim];
        let mut ffn_r_in = vec![0.0f32; dim];
        for i in 0..dim {
            let mix_k = self.ffn_tm_mix_k[i];
            let mix_r = self.ffn_tm_mix_r[i];
            ffn_k_in[i] = mix_k * x_res1_normed[i] + (1.0 - mix_k) * ffn_xx[i];
            ffn_r_in[i] = mix_r * x_res1_normed[i] + (1.0 - mix_r) * ffn_xx[i];
            ffn_xx[i] = x_res1_normed[i];
        }

        let ffn_r = self
            .channel_mix_receptance
            .forward(&cpu_tensor(ffn_r_in, Shape::new(vec![1, dim])))?;
        let ffn_k = self
            .channel_mix_key
            .forward(&cpu_tensor(ffn_k_in, Shape::new(vec![1, dim])))?;
        let ffn_r_vec = ffn_r.to_vec_f32()?;
        let ffn_k_vec = ffn_k.to_vec_f32()?;

        // ReLU on the projected key, sigmoid(r) gate, value projection.
        let mut gated = vec![0.0f32; dim];
        for i in 0..dim {
            let sig_r = 1.0 / (1.0 + (-ffn_r_vec[i]).exp());
            gated[i] = sig_r * ffn_k_vec[i].max(0.0);
        }
        let ffn_v = self
            .channel_mix_value
            .forward(&cpu_tensor(gated, Shape::new(vec![1, dim])))?;

        // Residual: x_res1 + ffn_out.
        let out = add_tensors(&x_res1, &ffn_v).map_err(grim_core::Error::Tensor)?;
        Ok(out)
    }
}

impl RwkvBlock {
    fn cfg_hidden(&self) -> usize {
        self.time_mix_key.weight.shape().dims()[1]
    }
}

pub struct Rwkv {
    pub cfg: RwkvConfig,
    pub device: Device,
    /// Embedding table: [vocab_size, hidden_size] stored as flat Vec.
    /// Used as a gather (token_id -> row), NOT a Linear matrix multiply.
    pub emb: Vec<f32>,
    pub emb_shape: (usize, usize), // (vocab_size, hidden_size)
    pub layers: Vec<RwkvBlock>,
    pub ln_out: RmsNorm,
    pub head: Linear,
}

impl Rwkv {
    pub fn load(ws: &grim_nn::WeightSource<'_>, cfg: RwkvConfig, device: Device) -> Result<Self> {
        Self::load_tp(ws, cfg, device, ws.tp_config())
    }

    /// Tensor-parallel load entry for RWKV. RWKV is a recurrent (time-mix)
    /// model: like Mamba, the recurrent path has no row-parallel all-reduce
    /// semantics, so column/row sharding of the time-/channel-mix matrices
    /// would change the recurrence rather than parallelise a matmul. A safe
    /// `load_tp` needs a bespoke RWKV sharding plan. Refuses `world_size > 1`
    /// until then.
    pub fn load_tp(
        ws: &grim_nn::WeightSource<'_>,
        cfg: RwkvConfig,
        device: Device,
        tp: grim_nn::TensorParallelConfig,
    ) -> Result<Self> {
        grim_nn::require_single_device(
            tp,
            "RWKV",
            "the recurrent time/channel-mix path has no row-parallel all-reduce semantics; \
             sharding needs a bespoke plan",
        )
        .map_err(grim_core::Error::Unimplemented)?;
        let emb_weight = ws
            .pp("emb")
            .get(Shape::new(vec![cfg.vocab_size, cfg.hidden_size]), "weight")?
            .to_vec_f32()?;
        // RWKV embedding is [vocab_size, hidden_size] — use as a gather table,
        // NOT a Linear matrix multiply.
        // [P1-32 fix: emb is an embedding table, not a Linear.]
        if emb_weight.len() != cfg.vocab_size * cfg.hidden_size {
            return Err(Error::Shape(format!(
                "RWKV emb: expected {} elements, got {}",
                cfg.vocab_size * cfg.hidden_size,
                emb_weight.len()
            )));
        }
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(RwkvBlock::load(
                &ws.pp("blocks").pp(&i.to_string()),
                &cfg,
                device.clone(),
            )?);
        }
        let ln_out = RmsNorm::load(&ws.pp("ln_out"), cfg.hidden_size, cfg.rms_norm_eps as f32)?;
        let head = Linear::load(&ws.pp("head"), cfg.hidden_size, cfg.vocab_size, false)?;

        let vocab_size = cfg.vocab_size;
        let hidden_size = cfg.hidden_size;
        assert_weight_shape(&ln_out.weight, &[hidden_size], "ln_out")?;
        assert_weight_shape(&head.weight, &[vocab_size, hidden_size], "head")?;

        Ok(Self {
            cfg,
            device,
            emb: emb_weight,
            emb_shape: (vocab_size, hidden_size),
            layers,
            ln_out,
            head,
        })
    }
}

impl Model for Rwkv {
    fn config(&self) -> &dyn ModelConfig {
        &self.cfg
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn param_arith(&self) -> ArithType {
        ArithType::F32
    }
}

impl StatefulSequence for Rwkv {
    fn init_state(&self, batch: usize) -> Box<dyn SsmState> {
        assert_eq!(batch, 1, "RWKV state threading supports batch 1");
        let hidden = self.cfg.hidden_size;
        let mut data = vec![0.0f32; self.cfg.num_layers * 5 * hidden];
        for l in 0..self.cfg.num_layers {
            // slot 3 per layer is pp (WKV running max): start at -inf so the
            // first token's e1 vanishes and time_first dominates (v4 init).
            let base = l * 5 * hidden + 3 * hidden;
            for v in data[base..base + hidden].iter_mut() {
                *v = -1e38f32;
            }
        }
        Box::new(RwkvState {
            data,
            num_layers: self.cfg.num_layers,
            hidden,
        })
    }

    /// Step the whole model over `input`'s tokens IN ORDER, threading the
    /// recurrence across tokens AND calls. Returns logits for the LAST
    /// position.
    fn step(&self, state: &mut dyn SsmState, input: &Tensor) -> Result<Tensor> {
        let s = state
            .as_any_mut()
            .downcast_mut::<RwkvState>()
            .ok_or_else(|| Error::Session("RWKV state downcast failed".into()))?;
        if s.num_layers != self.cfg.num_layers || s.hidden != self.cfg.hidden_size {
            return Err(Error::Session(
                "RWKV state was initialized for a different model".into(),
            ));
        }
        // Embedding gather: token ID -> table row. NOT a Linear matmul.
        let input_ids = input.to_vec_f32()?;
        if input_ids.is_empty() {
            return Err(Error::Shape("empty RWKV input".into()));
        }
        let mut last_logits = None;
        for &id in &input_ids {
            let idx = id as usize * self.emb_shape.1;
            if idx + self.emb_shape.1 > self.emb.len() {
                return Err(Error::Shape(format!(
                    "RWKV token id {} out of range for vocab {}",
                    id as usize, self.emb_shape.0
                )));
            }
            let mut h = cpu_tensor(
                self.emb[idx..idx + self.emb_shape.1].to_vec(),
                Shape::new(vec![1, self.emb_shape.1]),
            );
            for (l, layer) in self.layers.iter().enumerate() {
                h = layer.step(&h, l, s)?;
            }
            let h = self.ln_out.forward(&h)?;
            last_logits = Some(self.head.forward(&h)?);
        }
        Ok(last_logits.expect("non-empty input guarantees one forward"))
    }
}

impl CausalLm for Rwkv {
    fn new_session(&self) -> Box<dyn grim_core::session::SessionT> {
        Box::new(grim_core::session::Inner::new(self.device.clone()))
    }

    fn forward(
        &self,
        session: &mut dyn grim_core::session::SessionT,
        input_ids: &Tensor,
        _positions: &Tensor,
        _adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
        // Audit fix (grim-models): the recurrence state now lives on the
        // SESSION and advances across calls — the pre-fix code created a
        // fresh state per call (and the recurrence math itself ignored
        // state), so decode was context-free after prefill.
        if session.model_state().is_none() {
            session.set_model_state(Box::new(self.init_state(1)));
        }
        let cell = session
            .model_state_mut()
            .ok_or_else(|| Error::Session("rwkv: session model_state vanished".into()))?;
        let boxed_state = cell.downcast_mut::<Box<dyn SsmState>>().ok_or_else(|| {
            Error::Session("rwkv: session model_state holds another model's state".into())
        })?;
        let logits = self.step(boxed_state.as_mut(), input_ids)?;
        let seq_len = input_ids.shape().elem_count();
        session.advance_pos(seq_len);
        Ok(logits)
    }
}

#[cfg(test)]
mod audit_tests {
    use super::*;
    use grim_core::model::CausalLm;
    use grim_core::session::{Inner, SessionT};

    fn tiny_rwkv() -> Rwkv {
        // Weights are built directly (no GGUF): random-ish deterministic
        // tables so the recurrence has signal.
        let cfg = RwkvConfig {
            vocab_size: 64,
            hidden_size: 16,
            num_layers: 2,
            rms_norm_eps: 1e-5,
        };
        fn lcg(seed: &mut u64) -> f32 {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((*seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
        let mut lin_seed = 0x5EED_C0DEu64;
        let mut mk_lin = |n: usize| {
            Linear::from_tensor(
                cpu_tensor(
                    (0..n * cfg.hidden_size)
                        .map(|_| lcg(&mut lin_seed) * 0.1)
                        .collect::<Vec<f32>>(),
                    Shape::new(vec![n, cfg.hidden_size]),
                ),
                None,
            )
        };
        let norm = |eps: f32| RmsNorm {
            weight: cpu_tensor(
                vec![1.0f32; cfg.hidden_size],
                Shape::new(vec![cfg.hidden_size]),
            ),
            eps,
        };
        let layers = (0..cfg.num_layers)
            .map(|_| RwkvBlock {
                norm: norm(cfg.rms_norm_eps as f32),
                norm2: norm(cfg.rms_norm_eps as f32),
                time_mix_key: mk_lin(cfg.hidden_size),
                time_mix_value: mk_lin(cfg.hidden_size),
                time_mix_receptance: mk_lin(cfg.hidden_size),
                time_mix_output: mk_lin(cfg.hidden_size),
                channel_mix_key: mk_lin(cfg.hidden_size),
                channel_mix_receptance: mk_lin(cfg.hidden_size),
                channel_mix_value: mk_lin(cfg.hidden_size),
                device: Device::Cpu,
                tm_mix_k: vec![0.4; cfg.hidden_size],
                tm_mix_v: vec![0.5; cfg.hidden_size],
                tm_mix_r: vec![0.6; cfg.hidden_size],
                time_decay: vec![-0.7; cfg.hidden_size],
                time_first: vec![3.0; cfg.hidden_size],
                ffn_tm_mix_k: vec![0.45; cfg.hidden_size],
                ffn_tm_mix_r: vec![0.55; cfg.hidden_size],
            })
            .collect();
        let ln_out = RmsNorm {
            weight: cpu_tensor(
                vec![1.0f32; cfg.hidden_size],
                Shape::new(vec![cfg.hidden_size]),
            ),
            eps: cfg.rms_norm_eps as f32,
        };
        let head_data: Vec<f32> = (0..cfg.vocab_size * cfg.hidden_size)
            .map(|i| ((i % 29) as f32 * 0.02) - 0.25)
            .collect();
        let head = Linear::from_tensor(
            cpu_tensor(head_data, Shape::new(vec![cfg.vocab_size, cfg.hidden_size])),
            None,
        );
        Rwkv {
            cfg: cfg.clone(),
            device: Device::Cpu,
            emb: {
                let mut emb_seed = 0xBEEF_F00Du64;
                (0..cfg.vocab_size * cfg.hidden_size)
                    .map(|_| lcg(&mut emb_seed) * 0.2 - 0.1)
                    .collect()
            },
            emb_shape: (cfg.vocab_size, cfg.hidden_size),
            layers,
            ln_out,
            head,
        }
    }

    fn tok(v: f32) -> Tensor {
        cpu_tensor(vec![v], Shape::new(vec![1]))
    }

    /// Audit gate: the recurrence state must persist on the session across
    /// CausalLm::forward calls — the second call's logits through one
    /// session must equal explicit init→step→step threading — and the
    /// session position must advance.
    #[test]
    fn rwkv_forward_keeps_state_across_calls() {
        let model = tiny_rwkv();
        let mut sess = Inner::new(model.device.clone());
        let _a = CausalLm::forward(&model, &mut sess, &tok(1.0), &tok(0.0), &[]).unwrap();
        let b_session = CausalLm::forward(&model, &mut sess, &tok(2.0), &tok(0.0), &[])
            .unwrap()
            .to_vec_f32()
            .unwrap();
        assert_eq!(sess.current_pos(), 2, "pos advances per call");

        // Explicit state-threading reference.
        let mut st = model.init_state(1);
        let _ = model.step(st.as_mut(), &tok(1.0)).unwrap();
        let b_ref = model
            .step(st.as_mut(), &tok(2.0))
            .unwrap()
            .to_vec_f32()
            .unwrap();
        assert_eq!(
            b_session, b_ref,
            "CausalLm::forward must thread RWKV recurrence state across calls"
        );
    }

    /// Audit gate: the recurrence must actually MATTER — the same final
    /// token after different histories produces different logits. (The
    /// pre-fix model was memoryless by construction and failed this.)
    #[test]
    fn rwkv_recurrence_changes_output_with_history() {
        let model = tiny_rwkv();
        let mut s1 = Inner::new(model.device.clone());
        let mut s2 = Inner::new(model.device.clone());
        let adapters: [AdapterHandle; 0] = [];
        {
            let t = 1.0f32;
            let _ = CausalLm::forward(&model, &mut s1, &tok(t), &tok(0.0), &adapters).unwrap();
            let _ =
                CausalLm::forward(&model, &mut s2, &tok(t + 40.0), &tok(0.0), &adapters).unwrap();
        }
        let l1 = CausalLm::forward(&model, &mut s1, &tok(7.0), &tok(0.0), &adapters)
            .unwrap()
            .to_vec_f32()
            .unwrap();
        let l2 = CausalLm::forward(&model, &mut s2, &tok(7.0), &tok(0.0), &adapters)
            .unwrap()
            .to_vec_f32()
            .unwrap();
        let diff: f32 = l1.iter().zip(l2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 1e-3,
            "identical token after different histories must differ (recurrence dead): {diff}"
        );
    }
}

// ---------------------------------------------------------------------------
// Numeric reference test (audit follow-up): the WKV recurrence was previously
// tested only for state threading, never for value correctness against an
// independent recomputation of the v4 update.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod wkv_numeric_reference_tests {
    use super::*;
    use grim_backend_cpu::cpu_tensor;
    use grim_nn::Linear;

    /// Build a hand-constructed single-layer RWKV block with known weights:
    /// identity-ish projections (diagonal weights) so the reference math
    /// stays inspectable, plus nontrivial mix/decay parameters.
    fn test_block() -> RwkvBlock {
        let hidden = 4usize;
        let diag = |scale: f32| {
            let mut w = vec![0.0f32; hidden * hidden];
            for i in 0..hidden {
                w[i * hidden + i] = scale;
            }
            Linear::from_tensor(cpu_tensor(w, Shape::new(vec![hidden, hidden])), None)
        };
        let mut b = RwkvBlock {
            norm: RmsNorm {
                weight: cpu_tensor(vec![1.0; hidden], Shape::new(vec![hidden])),
                eps: 1e-5,
            },
            norm2: RmsNorm {
                weight: cpu_tensor(vec![1.0; hidden], Shape::new(vec![hidden])),
                eps: 1e-5,
            },
            time_mix_key: diag(1.0),
            time_mix_value: diag(1.0),
            time_mix_receptance: diag(1.0),
            time_mix_output: diag(1.0),
            channel_mix_key: diag(1.0),
            channel_mix_receptance: diag(1.0),
            channel_mix_value: diag(1.0),
            device: Device::Cpu,
            tm_mix_k: vec![0.5; hidden],
            tm_mix_v: vec![0.5; hidden],
            tm_mix_r: vec![0.5; hidden],
            time_decay: vec![-0.5; hidden],
            time_first: vec![1.0; hidden],
            ffn_tm_mix_k: vec![0.5; hidden],
            ffn_tm_mix_r: vec![0.5; hidden],
        };
        b.time_mix_output = diag(2.0); // distinguishable projection scale
        b
    }

    fn fresh_state() -> RwkvState {
        // Mirrors Rwkv::init_state for one layer of hidden=4: zero carries,
        // pp (slot 3) at -1e38 so time_first dominates on token 0.
        let hidden = 4usize;
        let mut data = vec![0.0f32; 5 * hidden];
        for v in data[3 * hidden..4 * hidden].iter_mut() {
            *v = -1e38;
        }
        RwkvState {
            data,
            num_layers: 1,
            hidden,
        }
    }

    /// One block step vs an independent f64 recomputation of the documented
    /// v4 time-mix + WKV + channel-mix math, run for TWO tokens so the
    /// state carry (aa/bb/pp and the token-shift slots) is value-checked.
    #[test]
    fn rwkv_wkv_two_steps_match_f64_reference() {
        let b = test_block();
        let hidden = 4usize;
        let mut state = fresh_state();

        let inputs = [vec![0.2f32, -0.1, 0.5, 0.0], vec![0.9, 0.3, -0.4, 0.7]];
        let mut got: Vec<Vec<f32>> = Vec::new();
        for x in &inputs {
            // NOTE: 2-D [1, hidden] like every real caller — `step` accepts a
            // 1-D input by its length check, but the residual add then
            // crashes in the CPU broadcast path (rank-1 vs rank-2). That
            // latent backend bug is recorded in the audit report.
            let xt = cpu_tensor(x.clone(), Shape::new(vec![1, hidden]));
            got.push(b.step(&xt, 0, &mut state).unwrap().to_vec_f32().unwrap());
        }

        // f64 reference state slots.
        let (mut xx_attn, mut aa) = (vec![0.0f64; hidden], vec![0.0f64; hidden]);
        let (mut bb, mut pp) = (vec![0.0f64; hidden], vec![-1e38f64; hidden]);
        let mut xx_ffn = vec![0.0f64; hidden];
        let tm_k = |i| b.tm_mix_k[i] as f64;
        let w_at = |i: usize, arr: &Linear| arr.weight.to_vec_f32().unwrap()[i * hidden + i] as f64;

        for (step_idx, x) in inputs.iter().enumerate() {
            let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
            // Token shift + projections (diagonal weights).
            let mut k = vec![0.0f64; hidden];
            let mut v = vec![0.0f64; hidden];
            let mut r = vec![0.0f64; hidden];
            for i in 0..hidden {
                k[i] = w_at(i, &b.time_mix_key)
                    * (tm_k(i) * x64[i] + (1.0 - tm_k(i)) * xx_attn[i]);
                v[i] = w_at(i, &b.time_mix_value)
                    * (b.tm_mix_v[i] as f64 * x64[i]
                        + (1.0 - b.tm_mix_v[i] as f64) * xx_attn[i]);
                r[i] = w_at(i, &b.time_mix_receptance)
                    * (b.tm_mix_r[i] as f64 * x64[i]
                        + (1.0 - b.tm_mix_r[i] as f64) * xx_attn[i]);
            }
            // Post-norm current hidden becomes next call's shift carry.
            let mean_sq = x64.iter().map(|v| v * v).sum::<f64>() / hidden as f64;
            let inv = 1.0 / (mean_sq + 1e-5).sqrt();
            let norm_x: Vec<f64> = x64.iter().map(|v| v * inv).collect();
            xx_attn.copy_from_slice(&norm_x);

            // WKV update + sigmoid gate.
            let mut attn_y = vec![0.0f64; hidden];
            for i in 0..hidden {
                let ww = b.time_first[i] as f64 + k[i];
                let p = pp[i].max(k[i]);
                let e1 = (pp[i] - p).exp();
                let e2 = (ww - p).exp();
                let num = e1 * aa[i] + e2 * v[i];
                let den = e1 * bb[i] + e2;
                let sig = 1.0 / (1.0 + (-r[i]).exp());
                attn_y[i] = sig * if den != 0.0 { num / den } else { 0.0 };
                aa[i] = num;
                bb[i] = den;
                pp[i] = p + b.time_decay[i] as f64;
            }
            for i in 0..hidden {
                attn_y[i] *= w_at(i, &b.time_mix_output);
            }
            // Channel mix (diagonal projections).
            let r1: Vec<f64> =
                x64.iter().zip(&attn_y).map(|(&a, &o)| a + o).collect();
            let mean_sq = r1.iter().map(|v| v * v).sum::<f64>() / hidden as f64;
            let inv = 1.0 / (mean_sq + 1e-5).sqrt();
            let n2: Vec<f64> = r1.iter().map(|v| v * inv).collect();
            let mut out = vec![0.0f64; hidden];
            for i in 0..hidden {
                let kin = b.ffn_tm_mix_k[i] as f64 * n2[i]
                    + (1.0 - b.ffn_tm_mix_k[i] as f64) * xx_ffn[i];
                let rin = b.ffn_tm_mix_r[i] as f64 * n2[i]
                    + (1.0 - b.ffn_tm_mix_r[i] as f64) * xx_ffn[i];
                let sig = 1.0 / (1.0 + (-rin).exp());
                let gated = sig * kin.max(0.0) * w_at(i, &b.channel_mix_key);
                out[i] = r1[i] + gated * w_at(i, &b.channel_mix_value);
            }
            xx_ffn.copy_from_slice(&n2);
            for (o, g) in out.iter().zip(&got[step_idx]) {
                assert!((o - *g as f64).abs() < 1e-4, "token {step_idx}: reference {o} vs impl {g}");
            }
        }
    }

}
