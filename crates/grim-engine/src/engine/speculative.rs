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
        let wrapped = SpeculativeCausalLm::auto(
            model,
            draft,
            markov,
            confidence,
            is_weight_streaming_active,
            available_vram,
        );
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
        let wrapped = SpeculativeCausalLm::with_native_mtp(
            Box::new(grim_speculative::LlamaMtpAdapter::new(model)),
            mtp_arc,
        );
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
                self.models.insert(id.to_string(), LoadedModel {
                    model: wrapped,
                    config: Box::new(grim_core::config::GenericModelConfig {
                        name: id.to_string(),
                        modality,
                    }),
                    device: dev,
                    tp_config: self.tp_config(),
                    arch_hyperparams: hyper,
                });
            }
        } else {
            self.register_model(id, base_model);
        }
        Ok(())
    }
}
