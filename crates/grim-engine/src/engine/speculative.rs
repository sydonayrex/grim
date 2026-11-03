//! Speculative decoding model registration and telemetry.

use crate::*;

impl Engine {
    /// Runtime speculative decoding telemetry for a specific model, or the first loaded model if `model_id` is None.
    /// Returns `None` if no model is loaded.
    pub fn speculative_telemetry(
        &self,
        model_id: Option<&str>,
    ) -> Option<grim_speculative::SpeculativeTelemetry> {
        let model = match model_id {
            Some(id) => self.models.get(id)?,
            None => self.models.values().next()?,
        };
        Some(model.model.telemetry())
    }

    /// Total count of speculative draft tokens accepted since engine startup.
    pub fn accepted_tokens_total(&self) -> u64 {
        self.accepted_tokens_total
    }

    /// WI-E2: speculative-decoding acceptance rate — accepted / generated.
    /// Returns None when no tokens have been generated yet.
    pub fn acceptance_rate(&self) -> Option<f64> {
        if self.total_tokens_generated == 0 {
            return None;
        }
        Some(self.accepted_tokens_total as f64 / self.total_tokens_generated as f64)
    }

    pub(crate) fn register_speculative(
        &mut self,
        id: &str,
        model: Box<dyn CausalLm>,
        draft: Option<Arc<dyn DraftBackbone>>,
        markov: Option<Arc<dyn MarkovHead>>,
        confidence: Option<Arc<dyn ConfidenceHead>>,
    ) {
        // By default we check if weight streaming is active and what VRAM remains.
        // During registration we check the environment or fallback parameters.
        let is_weight_streaming_active = std::env::var("GRIM_WEIGHT_STREAMING").is_ok();
        let available_vram = std::env::var("GRIM_AVAILABLE_VRAM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());

        let dev = model.device().clone();
        // WI-INF2: one SCYTHE-2 controller per loaded model, sized by the model's real transformer depth
        // (the `Engine::new` value was a placeholder - no model is known at construction time).
        if let Some(num_layers) = model.num_layers_hint() {
            if let Some(ctrl) = self.scythe_ctrl.as_mut() {
                let num_gpus = ctrl.num_gpus();
                *ctrl = crate::scythe2::C2plrController::new(num_layers, num_gpus, ctrl.budget_ms);
            }
        }
        // Preserve the model's own modality hint (audio enc-dec, TTS, VC, vocoder, diffusion…) so serving-layer routing sees the truth.
        // Hardcoding TextInTextOut misreported every non-text model that registered through this path, including the audio models.
        let modality = model.config().modality();
        // R4 - capture the model's hyperparameters (when reportable) before `model` is moved into the
        // speculative wrapper, so the admission gate can re-certify each request's footprint against *current* free memory.
        let arch_hyperparams = model.arch_hyperparams();
        let mut wrapped = SpeculativeCausalLm::auto(
            model,
            draft,
            markov,
            confidence,
            is_weight_streaming_active,
            available_vram,
        );
        // Task 10.6: attach the GPU rejection-sampling verifier — the
        // wrapper falls back to the CPU loop for non-ROCm targets.
        wrapped.set_gpu_verifier(std::sync::Arc::new(RocmSpeculativeVerifier));
        let config: Box<dyn ModelConfig> = Box::new(grim_core::config::GenericModelConfig {
            name: id.to_string(),
            modality,
        });

        self.models.insert(
            id.to_string(),
            LoadedModel {
                model: Box::new(wrapped),
                config,
                device: dev,
                tp_config: self.tp_config(),
                arch_hyperparams,
            },
        );
    }

    /// Register a native multi-token prediction model wrapped in `LlamaMtpAdapter`.
    pub fn register_native_mtp_model(
        &mut self,
        id: &str,
        model: Arc<grim_models_transformer::LlamaMtp>,
    ) {
        let dev = grim_core::Model::device(model.as_ref()).clone();
        let modality = grim_core::Model::config(model.as_ref()).modality();
        // R4 — capture hyperparams before `model` moves into the adapters.
        let arch_hyperparams = model.arch_hyperparams();
        let adapter = grim_speculative::LlamaMtpAdapter::new(model.clone());
        let mtp_arc: Arc<dyn grim_speculative::NativeMtp> = Arc::new(adapter);
        let mut wrapped = SpeculativeCausalLm::with_native_mtp(
            Box::new(grim_speculative::LlamaMtpAdapter::new(model)),
            mtp_arc,
        );
        // Task 10.6: GPU rejection-sampling verifier (CPU fallback for non-ROCm).
        wrapped.set_gpu_verifier(std::sync::Arc::new(RocmSpeculativeVerifier));
        let config: Box<dyn ModelConfig> = Box::new(grim_core::config::GenericModelConfig {
            name: id.to_string(),
            modality,
        });
        self.models.insert(
            id.to_string(),
            LoadedModel {
                model: Box::new(wrapped),
                config,
                device: dev,
                tp_config: self.tp_config(),
                arch_hyperparams,
            },
        );
    }

    /// Register an EAGLE3 speculative drafter model coupled with a base model.
    pub fn register_eagle3_model(
        &mut self,
        id: &str,
        base_model: Box<dyn CausalLm>,
        eagle3_model: Arc<grim_models_transformer::Eagle3>,
    ) {
        let drafter = Arc::new(grim_speculative::Eagle3Drafter::new(eagle3_model));
        self.register_speculative(id, base_model, Some(drafter), None, None);
    }

    /// Load base model and optional draft model / lookahead, registering them into the engine.
    pub fn load_and_register_speculative(
        &mut self,
        id: &str,
        base_path: &str,
        draft_path: Option<&str>,
        _lookahead: bool,
    ) -> Result<()> {
        let base_model = crate::model_loader::load_from_path(base_path)?;
        if let Some(d_path) = draft_path {
            let dev = base_model.device().clone();
            if let Ok(eagle3) = crate::model_loader::load_eagle3_from_path(d_path, dev.clone()) {
                self.register_eagle3_model(id, base_model, eagle3);
            } else {
                // Generic draft file: the draft backbone is a TinyDraftBackbone
                // sized from the BASE model's hyperparameters (the draft file's
                // own weights are not 1:1 reusable as a backbone yet).
                let _ = crate::model_loader::load_from_path(d_path)?;
                let hyper = base_model.arch_hyperparams();
                let dims = hyper
                    .as_ref()
                    .map(|h| (h.vocab_size, h.hidden_size))
                    .unwrap_or((128256, 2048));
                let drafter = Arc::new(grim_speculative::TinyDraftBackbone::new(
                    dims.0, dims.1, 4, 42,
                ));
                let modality = base_model.config().modality();
                let wrapped = Self::build_dspark_model(base_model, drafter);
                self.models.insert(
                    id.to_string(),
                    LoadedModel {
                        model: wrapped,
                        config: Box::new(grim_core::config::GenericModelConfig {
                            name: id.to_string(),
                            modality,
                        }),
                        device: dev,
                        tp_config: self.tp_config(),
                        arch_hyperparams: hyper,
                    },
                );
            }
        } else {
            self.register_model(id, base_model);
        }
        Ok(())
    }
}

// ===== Task 10.6: GPU rejection-sampling verifier (ROCm) =====
//
// Implements `grim_speculative::SpeculativeVerifier` with the device-side
// `grim_speculative_rejection_sample` kernel: row slices are narrowed D2D,
// softmaxed on device, and only the tiny accepted_lens/accepted_tokens words
// cross back to host — no vocab-sized D2H copy in the verify path.

pub struct RocmSpeculativeVerifier;

impl grim_speculative::speculative_wrapper::SpeculativeVerifier for RocmSpeculativeVerifier {
    fn verify(
        &self,
        target_logits: &grim_tensor::Tensor,
        draft_logits: &grim_tensor::Tensor,
        context_len: usize,
        vocab_size: usize,
        draft_tokens: &[u32],
        uniform_rands: &[f32],
    ) -> Result<usize> {
        use grim_tensor::backend::{CoreTensorOps, ElementwiseOps, MemoryOps};
        use grim_tensor::{DType, Shape};

        let ordinal = match target_logits.device() {
            grim_tensor::Device::Rocm(o) => *o as usize,
            other => {
                return Err(grim_core::error::Error::Backend(format!(
                    "spec verify: non-ROCm target {other:?}"
                )));
            }
        };
        let dev = grim_backend_rocm::device::roc_device::RocmDevice::try_new(ordinal)?;
        let k = draft_tokens.len();
        if k == 0 {
            return Ok(0);
        }

        // Kernel ABI: target_probs [K+1, vocab] (row K = bonus row, unused by
        // the accepted-count contract — left zero), draft_probs [K, vocab],
        // draft_tokens int32 bytes, uniforms [K+1], accepted out [K+1]/[1].
        let t_rows = dev
            .narrow_rows(
                target_logits.storage().as_ref(),
                context_len,
                k,
                vocab_size,
                &Shape::new(vec![k, vocab_size]),
            )?
            .0;
        let t_probs = dev.softmax(t_rows.as_ref(), &Shape::new(vec![k, vocab_size]))?.0;
        let mut t_buf = dev.alloc_storage(
            &Shape::new(vec![k + 1, vocab_size]),
            DType::F32,
        )?;
        dev.write_rows(t_buf.as_mut(), 0, t_probs.as_ref(), k, vocab_size)?;

        let d_rows = dev
            .narrow_rows(
                draft_logits.storage().as_ref(),
                0,
                k,
                vocab_size,
                &Shape::new(vec![k, vocab_size]),
            )?
            .0;
        let d_probs = dev.softmax(d_rows.as_ref(), &Shape::new(vec![k, vocab_size]))?.0;

        // draft_tokens as raw i32 bytes (kernel dereferences `const int*` —
        // f32 bit patterns would be garbage).
        let dt_bytes: Vec<u8> = draft_tokens
            .iter()
            .flat_map(|&t| (t as i32).to_le_bytes())
            .collect();
        let dt = grim_backend_rocm::MemoryOps::from_cpu_bytes(
            &dev,
            &dt_bytes,
            &Shape::new(vec![k]),
            DType::U8,
        )?;
        let ur = <grim_backend_rocm::device::roc_device::RocmDevice as CoreTensorOps>::from_cpu(
            &dev,
            uniform_rands,
            &Shape::new(vec![uniform_rands.len()]),
            DType::F32,
        )?;
        let at = dev.alloc_storage(&Shape::new(vec![k + 1]), DType::F32)?;
        let al = dev.alloc_storage(&Shape::new(vec![1]), DType::F32)?;

        let tp_s = grim_backend_rocm::device::util::as_rocm(t_buf.as_ref())?;
        let dp_s = grim_backend_rocm::device::util::as_rocm(d_probs.as_ref())?;
        let dt_s = grim_backend_rocm::device::util::as_rocm(dt.as_ref())?;
        let ur_s = grim_backend_rocm::device::util::as_rocm(ur.as_ref())?;
        let at_s = grim_backend_rocm::device::util::as_rocm(at.as_ref())?;
        let al_s = grim_backend_rocm::device::util::as_rocm(al.as_ref())?;

        dev.launch_speculative_rejection_sample(
            tp_s, dp_s, dt_s, ur_s, at_s, al_s, 1, k, vocab_size,
        )?;
        dev.synchronize();

        // accepted_lens holds the kernel's count: all-accepted = K drafts +
        // bonus (K+1); rejected at j = j drafts + 1 residual (j+1). The
        // wrapper's contract counts DRAFT acceptances only, so subtract the
        // trailing sampled token.
        let lens = al.to_cpu_vec_f32()?;
        let accepted_len = lens
            .first()
            .map(|v| v.to_bits() as usize)
            .unwrap_or(0);
        Ok(accepted_len.saturating_sub(1).min(k))
    }
}
