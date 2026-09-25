//! LoRA adapter lifecycle and batched-LoRA application.

use crate::*;

impl Engine {
    /// Register a multi-LoRA adapter against a base model.
    /// The adapter is keyed by its [`AdapterHandle::id`] and dispatched into the forward pass when callers.
    pub fn register_adapter(
        &mut self,
        base_model_id: &str,
        name: impl Into<String>,
        handle: AdapterHandle,
    ) {
        self.adapters.insert(
            handle.id,
            LoadedAdapter {
                name: name.into(),
                handle,
                base_model_id: base_model_id.to_string(),
            },
        );
    }

    /// Resolve a set of adapter ids into concrete [`AdapterHandle`]s.
    /// Returns `None` if any id is unknown - the caller should drop the affected request.
    pub fn resolve_adapters(&self, ids: &[u32]) -> Option<Vec<AdapterHandle>> {
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            {
                let a = self.adapters.get(id)?;
                out.push(a.handle.clone())
            }
        }
        Some(out)
    }

    /// Drop an adapter from the registry. Its id is freed for reuse.
    pub fn drop_adapter(&mut self, id: u32) -> bool {
        self.adapters.remove(&id).is_some()
    }

    /// Number of currently-loaded adapters.
    pub fn adapter_count(&self) -> usize {
        self.adapters.len()
    }

    /// Look up an adapter handle by its human-readable name.
    /// Used by the HTTP server to validate names from request body `"adapters"` arrays before opening.
    pub fn get_adapter_by_name(&self, name: &str) -> Option<&LoadedAdapter> {
        self.adapters.values().find(|a| a.name == name)
    }

    /// Apply fused batched multi-LoRA (S-LoRA / Punica style) deltas to a stacked logits matrix, in place.
    /// Each maximal run of equal adapter ids in `row_adapters` becomes one segment; adapter 0 marks.
    pub fn apply_batched_lora_to_rows(
        &self,
        stacked: &mut [f32],
        row_adapters: &[u32],
        dim: usize,
        device: Option<&grim_tensor::dtype::Device>,
    ) -> Result<()> {
        if dim == 0 {
            return Err(Error::Config(
                "apply_batched_lora_to_rows: logits width must be > 0".into(),
            ));
        }
        let rows = stacked.len() / dim;
        if rows * dim != stacked.len() || row_adapters.len() != rows {
            return Err(Error::Config(format!(
                "apply_batched_lora_to_rows: stacked len {} not rows*dim \
                 (rows from adapters: {}) * dim {dim}",
                stacked.len(),
                row_adapters.len()
            )));
        }

        // Resolve every non-base adapter to its weights up front so shape failures surface before any device work.
        // We build TWO views of the same data: * `segments` + `weight_bank` - contiguous adapter.
        let mut segments: Vec<(
            grim_backend_rocm::kernels::batched_lora::BatchedLoraSegment,
            usize, // index into weight_bank
        )> = Vec::new();
        let mut weight_bank: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
        let mut id_to_idx: std::collections::HashMap<u32, (usize, usize)> =
            std::collections::HashMap::new();
        let mut dispatched_adapters: Vec<
            grim_backend_rocm::kernels::batched_lora::DispatchedLoraAdapter,
        > = Vec::new();
        for seg in grim_scheduler::LoraRowSegment::plan_for_rows(row_adapters) {
            if seg.adapter_id == 0 || seg.row_count == 0 {
                continue;
            }
            let Some(loaded) = self.adapters.get(&seg.adapter_id) else {
                return Err(Error::Config(format!(
                    "apply_batched_lora_to_rows: unknown adapter id {}",
                    seg.adapter_id
                )));
            };
            let a_vec = loaded.handle.a.to_vec_f32()?;
            let b_vec = loaded.handle.b.to_vec_f32()?;
            let rank = loaded.handle.a.shape().dim(0)?;
            let in_dim = loaded.handle.a.shape().dim(1)?;
            let out_dim = loaded.handle.b.shape().dim(0)?;
            if in_dim != dim || out_dim != dim {
                return Err(Error::Config(format!(
                    "apply_batched_lora_to_rows: adapter {} A[{rank},{in_dim}] B[{out_dim},{rank}] \
                     is incompatible with logits width {dim} (surrogate contract: in_dim == out_dim == dim)",
                    seg.adapter_id
                )));
            }
            let scaling = loaded.handle.alpha / (rank as f32).max(1.0);

            // Stable dense index per adapter id: an adapter that spans multiple
            // contiguous segments reuses the same index (and the same weights).
            let (dense, bank_idx) = *id_to_idx.entry(seg.adapter_id).or_insert_with(|| {
                let bank_idx = weight_bank.len();
                let dense = dispatched_adapters.len();
                weight_bank.push((a_vec.clone(), b_vec.clone()));
                dispatched_adapters.push(
                    grim_backend_rocm::kernels::batched_lora::DispatchedLoraAdapter {
                        a_weights: a_vec,
                        b_weights: b_vec,
                        rank,
                        scaling,
                    },
                );
                (dense, bank_idx)
            });
            let seg_kernel = grim_backend_rocm::kernels::batched_lora::BatchedLoraSegment {
                adapter_id: dense as u32,
                token_start: seg.row_start,
                token_count: seg.row_count,
                rank,
                scaling,
            };
            segments.push((seg_kernel, bank_idx));
        }
        if segments.is_empty() {
            return Ok(());
        }

        // Per-row adapter indirection table for the dispatched path: one dense
        // index (or u32::MAX for the base model) per row.
        let token_adapter_idx: Vec<u32> = row_adapters
            .iter()
            .map(|&id| {
                if id == 0 {
                    u32::MAX
                } else {
                    id_to_idx
                        .get(&id)
                        .copied()
                        .expect("adapter validated above")
                        .0 as u32
                }
            })
            .collect();

        // GPU DISPATCHED path (preferred): two kernel launches total for any
        // number of adapters. Falls back to the CPU reference.
        if let Some(grim_tensor::dtype::Device::Rocm(ordinal)) = device {
            if grim_backend_rocm::device::roc_device::RocmDevice::probe_one(*ordinal)
                .unwrap_or(false)
            {
                let device = grim_backend_rocm::device::roc_device::RocmDevice::new(*ordinal);
                let x = stacked.to_vec();
                match grim_backend_rocm::kernels::batched_lora::batched_lora_dispatched_device(
                    &device,
                    &x,
                    stacked,
                    dim,
                    dim,
                    &token_adapter_idx,
                    &dispatched_adapters,
                ) {
                    Ok(()) => return Ok(()),
                    Err(e) => {
                        log::warn!(
                            "[grim-engine] batched LoRA dispatched dispatch failed (ordinal \
                             {ordinal}), falling back to CPU reference: {e}"
                        );
                    }
                }
            }
        }

        // CPU portable reference (also the source of truth the GPU kernels are
        // tested against): per-adapter-segment accumulation.
        let x = stacked.to_vec();
        for (seg, bank_idx) in &segments {
            let (a, b) = &weight_bank[*bank_idx];
            grim_backend_rocm::kernels::batched_lora::batched_lora_accumulate_cpu(
                &x, stacked, dim, dim, seg, a, b,
            )?;
        }
        Ok(())
    }

    /// Rebuild a logits tensor with `data` contents on the same device, dtype and provenance
    /// as `like`, so downstream device-resident sampling (WI-X3) keeps working after the batched LoRA pass.
    pub(crate) fn logits_tensor_like(
        data: Vec<f32>,
        shape: grim_tensor::Shape,
        like: &grim_tensor::Tensor,
    ) -> Result<grim_tensor::Tensor> {
        if like.device().is_cpu() {
            let storage = Arc::new(grim_backend_cpu::storage::CpuStorage::new(
                data,
                shape.clone(),
                grim_tensor::DType::F32,
            ));
            return Ok(grim_tensor::Tensor::new(
                storage,
                shape,
                grim_tensor::DType::F32,
                like.provenance().clone(),
                like.device().clone(),
            ));
        }
        let dev = grim_nn::modules::pick_device_for_tensor(like);
        let storage = dev.from_cpu(&data, &shape, grim_tensor::dtype::DType::F32)?;
        Ok(grim_tensor::Tensor::new(
            std::sync::Arc::from(storage),
            shape,
            grim_tensor::dtype::DType::F32,
            like.provenance().clone(),
            like.device().clone(),
        ))
    }

    /// Fresh adapter id: max existing + 1. (`adapter_count() + 1` would reuse
    /// ids freed by `drop_adapter`, silently aliasing stale references.)
    pub fn next_adapter_id(&self) -> u32 {
        self.adapters.keys().copied().max().unwrap_or(0) + 1
    }
}

#[cfg(test)]
mod adapter_lifecycle_tests {
    use super::*;
    use crate::EngineConfig;

    fn test_handle(id: u32) -> AdapterHandle {
        AdapterHandle {
            id,
            a: grim_backend_cpu::cpu_tensor(vec![1.0], grim_tensor::Shape::new(vec![1, 1])),
            b: grim_backend_cpu::cpu_tensor(vec![1.0], grim_tensor::Shape::new(vec![1, 1])),
            alpha: 1.0,
        }
    }

    /// Dropped ids must not resolve (no stale alias); re-registration serves
    /// the new handle; `next_adapter_id` never reuses a live-or-higher id.
    #[test]
    fn drop_frees_id_without_stale_alias() {
        let mut engine = Engine::new(EngineConfig::default());
        engine.register_adapter("base", "first", test_handle(7));
        assert!(engine.resolve_adapters(&[7]).is_some());
        assert!(engine.drop_adapter(7));
        assert!(
            engine.resolve_adapters(&[7]).is_none(),
            "dropped id must not resolve"
        );
        assert!(!engine.drop_adapter(7), "double drop must report false");
        engine.register_adapter("base", "second", test_handle(7));
        let resolved = engine.resolve_adapters(&[7]).expect("re-registered");
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].id, 7);
        assert_eq!(engine.get_adapter_by_name("second").unwrap().handle.id, 7);
        assert!(engine.get_adapter_by_name("first").is_none());
    }
}
