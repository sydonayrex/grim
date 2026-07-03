//! Grim engine runtime — wires scheduler + memory + model registry.
//!
//! The `Engine` is the top-level orchestrator. It owns:
//! - A `Scheduler` for batching and admission control.
//! - A model registry (type-erased, keyed by model id).
//! - Holds an optional KV block pool.
//!
//! Downstream crates (`grim-server`) call `Engine::tick()` once per
//! iteration to run the scheduler and execute the batch.

use std::collections::HashMap;
use std::sync::Arc;

use grim_core::error::Result;
use grim_core::model::{CausalLm, ModelConfig};
use grim_core::session::SessionT;
use grim_memory::KvBlockPool;

type DynModelPtr = Box<dyn CausalLm>;

/// A loaded model with its config and an instantiated CausalLm impl.
pub struct LoadedModel {
    pub model: DynModelPtr,
    pub config: Box<dyn ModelConfig>,
}

/// Engine configuration.
pub struct EngineConfig {
    pub max_batched_tokens: usize,
    pub max_num_seqs: usize,
    pub block_pool_capacity: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub target_ttft_ms: u64,
    pub target_itl_ms: u64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_batched_tokens: 4096,
            max_num_seqs: 8,
            block_pool_capacity: 1024,
            num_kv_heads: 4,
            head_dim: 128,
            target_ttft_ms: 2000,
            target_itl_ms: 100,
        }
    }
}

/// The core engine. Call `tick()` to advance one iteration.
pub struct Engine {
    pub config: EngineConfig,
    pub scheduler: grim_scheduler::Scheduler,
    pub block_pool: Arc<std::sync::Mutex<KvBlockPool>>,
    models: HashMap<String, LoadedModel>,
    sessions: HashMap<u64, Box<dyn SessionT>>,
}

impl Engine {
    pub fn new(config: EngineConfig) -> Self {
        let block_pool = Arc::new(std::sync::Mutex::new(KvBlockPool::new(
            config.block_pool_capacity,
            config.num_kv_heads,
            config.head_dim,
        )));
        let admission = grim_scheduler::AdmissionController::new(config.target_ttft_ms, config.target_itl_ms);
        let scheduler = grim_scheduler::Scheduler::new(
            config.max_batched_tokens,
            config.max_num_seqs,
            admission,
        );
        Self {
            config,
            scheduler,
            block_pool,
            models: HashMap::new(),
            sessions: HashMap::new(),
        }
    }

    pub fn register_model(&mut self, id: &str, model: Box<dyn CausalLm>) {
        let config: Box<dyn ModelConfig> = Box::new(grim_core::config::GenericModelConfig {
            name: id.to_string(),
            modality: grim_core::model::ModalityHint::TextInTextOut,
        });
        self.models.insert(id.to_string(), LoadedModel { model, config });
    }

    pub fn tick(&mut self) -> Result<grim_scheduler::SchedulerOutput> {
        let output = self.scheduler.schedule();
        // v1: scheduler decides; engine just reports what would run.
        // The actual model forward pass is added once the full tick
        // execution path is wired through.
        Ok(output)
    }

    pub fn enqueue_request(&mut self, request: grim_scheduler::Request) {
        let session = Box::new(grim_core::session::Inner::new(grim_tensor::Device::Cpu));
        self.sessions.insert(request.id, session);
        self.scheduler.enqueue(request);
    }

    pub fn finish_request(&mut self, id: u64) {
        self.scheduler.finish(id);
        self.sessions.remove(&id);
    }

    pub fn model(&self, id: &str) -> Option<&LoadedModel> {
        self.models.get(id)
    }
}

/// Re-export key types at the grim-engine crate root.
pub use grim_memory::PagedKvCache;
pub use grim_scheduler::{AdmissionController, Request, Scheduler, SchedulerOutput};