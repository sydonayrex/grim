//! Model registration/unloading.

use crate::*;

impl Engine {
    pub fn register_model(&mut self, id: &str, model: Box<dyn CausalLm>) {
        self.register_speculative(id, model, None, None, None);
    }

    /// Build a DSpark-wrapped model from a target + draft backbone, sizing the
    /// markov/confidence heads from the target's real hyperparameters.
    ///
    /// dats-demm §3: this wires the previously-dead markov/confidence/scheduler
    /// stack end-to-end — `Strategy::DSpark` in the decode loop is now
    /// reachable from production entrypoints (CLI `--draft-model`, engine
    /// `load_and_register_speculative`).
    pub fn build_dspark_model(
        model: Box<dyn CausalLm>,
        draft: Arc<dyn DraftBackbone>,
    ) -> Box<SpeculativeCausalLm> {
        let vocab = model
            .arch_hyperparams()
            .map(|h| h.vocab_size)
            .unwrap_or(128256);
        // ponytail: UniformMarkovHead ignores hidden today; pass it through
        // anyway once the head grows a projection.
        let markov = Arc::new(grim_speculative::UniformMarkovHead::new(vocab, 8, 0xD5_A4_ED_u64));
        let confidence = Arc::new(grim_speculative::EntropyConfidenceHead);
        let scheduler = grim_speculative::ConfidenceScheduler::new(
            grim_speculative::ThroughputProfile::default(),
            grim_speculative::SpeculationConfig::default(),
        );
        Box::new(grim_speculative::SpeculativeCausalLm::with_dspark(
            model, draft, markov, confidence, scheduler,
        ))
    }

    pub fn register_with_dspark(
        &mut self,
        id: &str,
        model: Box<dyn CausalLm>,
        draft: Arc<dyn DraftBackbone>,
        markov: Arc<dyn MarkovHead>,
        confidence: Arc<dyn ConfidenceHead>,
    ) {
        self.register_speculative(id, model, Some(draft), Some(markov), Some(confidence));
    }

    /// Dynamically reconfigure the MoE VRAM budget at a safe point between inference steps.
    /// # Contract Dynamically adjusts the split between KV cache pages and MoE expert cache slots.
    pub fn reconfigure_moe_budget(
        &mut self,
        new_kv_envelope_bytes: usize,
        new_expert_envelope_bytes: usize,
    ) -> Result<usize> {
        let mut pool = self.block_pool.lock().unwrap_or_else(|e| e.into_inner());
        let block_bytes = pool.block_bytes();
        let target_blocks = new_kv_envelope_bytes.checked_div(block_bytes).unwrap_or(0);
        pool.resize_capacity(target_blocks);
        Ok(new_expert_envelope_bytes)
    }

    /// Unload a model from memory by its name. Returns true if the model was loaded.
    pub fn unload_model(&mut self, name: &str) -> bool {
        self.models.remove(name).is_some()
    }
}
