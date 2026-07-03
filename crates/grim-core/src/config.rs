//! Concrete `ModelConfig` impls live in `grim-models/<family>`.
//! Grim-core does not ship any built-in configs — only the trait surface.

use crate::model::ModalityHint;

/// Re-export so downstream crates have a single import path for both
/// the trait and the constructors they implement.
pub use crate::model::ModelConfig as ModelConfigTrait;

/// Helper for crates that need to construct configs inline (mostly tests).
pub struct InlineConfig {
    pub name: String,
    pub modality: ModalityHint,
    pub cfg: Box<dyn std::any::Any + Send + Sync>,
}

/// A minimal `ModelConfig` implementation for dynamic registration.
pub struct GenericModelConfig {
    pub name: String,
    pub modality: ModalityHint,
}

impl crate::model::ModelConfig for GenericModelConfig {
    fn name(&self) -> &str {
        &self.name
    }
    fn modality(&self) -> ModalityHint {
        self.modality
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
