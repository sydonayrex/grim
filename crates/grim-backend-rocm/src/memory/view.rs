//! Zero-copy sub-range views over an existing ROCm allocation.
//!
//! `RocmStorageView` lets the decode path slice a single device buffer (the
//! fused QKV GEMV output, `[1, n_q + 2·n_kv]` f32) into per-head `q`/`k`/`v`
//! tensors without an extra D2D copy or device allocation. The view borrows
//! its bytes from a parent allocation and adds a byte offset to the parent's
//! device pointer.
//!
//! # Why a distinct struct (not a second `RocmStorage`)
//!
//! `RocmStorage::drop` (memory/storage.rs:433) *frees* `device_ptr` — managed
//! branch calls `hipFree`, pooled branch calls `allocator.free`. A bare
//! `RocmStorage` whose pointer is `parent_ptr + offset` would, on drop, free
//! the mid-allocation address and corrupt the caching allocator's free-list.
//! An `Arc` keep-alive does NOT help: `Drop` fires per instance regardless of
//! shared ownership. So the view is a distinct type with **no `Drop` impl and
//! no free**; lifetime is guaranteed by `_parent: Arc<dyn BackendStorage>`.
//!
//! # Kernel-surface impact
//!
//! Device ops that consume views (`rms_norm`, `copy_slice_range`) were migrated
//! from `as_rocm()` (which downcasts to `&RocmStorage`) to the `BackendStorage`
//! trait methods `device_ptr()` / `shape()` / `device_ordinal()`, so they accept
//! either a `RocmStorage` or a `RocmStorageView` with zero extra cost. No JIT
//! kernel source changes.

use std::ffi::c_void;
use std::sync::Arc;

use grim_tensor::backend::BackendStorage;
use grim_tensor::dtype::QuantProvenance;
use grim_tensor::error::{Error, Result};

use crate::memory::storage::dequant_cpu;
use crate::{check_hip, hipMemcpy, DType, DTypeStorage, HipMemcpyKind, RocmStorage, Shape};

/// A byte-offset view over a parent ROCm allocation. `Send + Sync` (all fields
/// are), but **never frees** — see module docs.
pub struct RocmStorageView {
    /// Parent device pointer + byte_offset. Stored as u64 to match `RocmStorage`.
    device_ptr: u64,
    bytes: usize,
    shape: Shape,
    dtype: DType,
    provenance: QuantProvenance,
    ordinal: usize,
    /// Keeps the backing allocation alive. MUST outlive every view.
    _parent: Arc<dyn BackendStorage>,
}

// All fields are Send + Sync (Arc<dyn BackendStorage> + primitives); the view
// itself performs no work on drop, so sharing across threads is sound.
unsafe impl Send for RocmStorageView {}
unsafe impl Sync for RocmStorageView {}

impl RocmStorageView {
    /// Build a view over `[byte_offset, byte_offset + bytes)` of `parent`.
    ///
    /// `shape` is the logical shape of the *view* (e.g. `[n_q]`), not the parent.
    /// The caller must ensure `byte_offset + bytes` does not exceed the parent's
    /// allocation and that the parent outlives the view (it does, via `_parent`).
    pub fn from_offset(
        parent: Arc<dyn BackendStorage>,
        byte_offset: usize,
        bytes: usize,
        shape: Shape,
    ) -> Result<Self> {
        let base = parent.device_ptr().ok_or_else(|| {
            Error::Backend(
                "RocmStorageView: parent has no device pointer (CPU-resident?)".into(),
            )
        })?;
        if bytes == 0 {
            return Err(Error::Backend(
                "RocmStorageView: zero-byte view".into(),
            ));
        }
        Ok(Self {
            device_ptr: base + byte_offset as u64,
            bytes,
            shape,
            dtype: parent.dtype(),
            provenance: parent.provenance(),
            ordinal: parent.device_ordinal() as usize,
            _parent: parent,
        })
    }

    /// Convenience: view over a contiguous run of `row_count` Q8_0 rows, each
    /// `(elems_per_row / 32) * 34` bytes. Mirrors the fused-QKV weight layout.
    pub fn q80_rows(
        parent: Arc<dyn BackendStorage>,
        row_idx: usize,
        row_count: usize,
        elems_per_row: usize,
    ) -> Result<Self> {
        if elems_per_row == 0 || elems_per_row % 32 != 0 {
            return Err(Error::Backend(format!(
                "RocmStorageView::q80_rows: elems_per_row must be a multiple of 32, got {elems_per_row}"
            )));
        }
        let row_bytes = (elems_per_row / 32) * 34;
        let byte_offset = row_idx * row_bytes;
        let bytes = row_count * row_bytes;
        let shape = Shape::new(vec![row_count, elems_per_row]);
        Self::from_offset(parent, byte_offset, bytes, shape)
    }

    /// The device pointer for the start of this view (parent base + offset).
    pub fn device_ptr_u64(&self) -> u64 {
        self.device_ptr
    }

    /// Read the view's raw device bytes back to host (D2H). Used by tests and
    /// the host-fallback path; not on the hot decode path.
    pub fn copy_to_host(&self) -> Result<Vec<u8>> {
        if self.bytes == 0 {
            return Ok(Vec::new());
        }
        let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let mut raw = vec![0u8; self.bytes];
        check_hip(
            "RocmStorageView copy_to_host",
            unsafe {
                hipMemcpy(
                    raw.as_mut_ptr() as *mut c_void,
                    self.device_ptr as *const c_void,
                    self.bytes,
                    HipMemcpyKind::DeviceToHost,
                )
            },
        )?;
        Ok(raw)
    }
}

impl BackendStorage for RocmStorageView {
    fn dtype(&self) -> DType {
        self.dtype.clone()
    }

    fn provenance(&self) -> QuantProvenance {
        self.provenance.clone()
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn device_ptr(&self) -> Option<u64> {
        Some(self.device_ptr)
    }

    fn device_ordinal(&self) -> u32 {
        self.ordinal as u32
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn to_cpu_vec_f32(&self) -> Result<Vec<f32>> {
        let raw = self.copy_to_host()?;
        // f32/native: raw bytes ARE the data.
        if self.dtype.arith == grim_tensor::ArithType::F32
            && matches!(self.dtype.storage, DTypeStorage::Native)
        {
            let elems = raw.len() / 4;
            let mut out = Vec::with_capacity(elems);
            for chunk in raw.chunks_exact(4) {
                let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                out.push(f32::from_bits(bits));
            }
            return Ok(out);
        }
        // Quantized / packed: dequantize the raw bytes with the view's dtype.
        dequant_cpu(&raw, self.shape.elem_count(), &self.dtype)
    }
}

/// Allocate a `RocmStorage` and wrap it so it can be the parent of views. Helper
/// for tests and for the fused-QKV output buffer.
pub fn alloc_parent(shape: &Shape, ordinal: usize) -> Result<RocmStorage> {
    RocmStorage::alloc_gpu(
        shape,
        DType {
            arith: grim_tensor::ArithType::F32,
            storage: DTypeStorage::Native,
        },
        // The view only needs the allocator to build a real RocmStorage; the
        // shared device's allocator is the right lifetime anchor.
        &crate::RocmDevice::shared(ordinal).allocator,
        ordinal,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_tensor::CoreTensorOps;

    #[test]
    fn rocm_storage_sub_view_matches_owned_copy() {
        // RED/GREEN for Item 1a: a view over a sub-range of a device buffer must
        // read back bytes identical to an independently-copied owned buffer at the
        // same offset, AND dropping every view must not corrupt the parent.
        if std::env::var("GRIM_GPU_TEST").as_deref() != Ok("1")
            && std::env::var("GRIM_RUN_GPU_TESTS").as_deref() != Ok("1")
        {
            return;
        }
        let ordinal = 0usize;
        let dev = crate::RocmDevice::shared(ordinal);

        // Parent: 256 f32 elements, values 0.0..255.0.
        let n = 256usize;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let parent = dev
            .from_cpu(&data, &Shape::new(vec![n]), DType::F32)
            .expect("upload parent");
        let parent_arc: Arc<dyn BackendStorage> = Arc::from(parent);

        // Sanity: parent reads back intact.
        let parent_bytes = parent_arc
            .as_ref()
            .as_any()
            .downcast_ref::<RocmStorage>()
            .expect("parent is RocmStorage")
            .copy_to_host()
            .expect("parent D2H");
        assert_eq!(parent_bytes.len(), n * 4);

        // View over rows 100..164 (64 elements, f32 = 256 bytes).
        let row_a = 100usize;
        let view_elems = 64usize;
        let byte_offset = row_a * 4;
        let view_bytes = view_elems * 4;
        let view = RocmStorageView::from_offset(
            parent_arc.clone(),
            byte_offset,
            view_bytes,
            Shape::new(vec![view_elems]),
        )
        .expect("build view");

        // (i) aliased data is bit-identical to the expected sub-range.
        let view_raw = view.copy_to_host().expect("view D2H");
        assert_eq!(view_raw.len(), view_bytes);
        assert_eq!(&view_raw[..], &parent_bytes[byte_offset..byte_offset + view_bytes]);

        // Also via to_cpu_vec_f32: must decode to 100.0..164.0.
        let view_vec = view.to_cpu_vec_f32().expect("view to_cpu_vec_f32");
        assert_eq!(view_vec.len(), view_elems);
        for i in 0..view_elems {
            assert!(
                (view_vec[i] - ((row_a + i) as f32)).abs() < 1e-6,
                "view element {}: expected {}, got {}",
                i,
                row_a + i,
                view_vec[i]
            );
        }

        // (ii) dropping all views must NOT corrupt the parent allocation.
        drop(view);
        let parent_after = parent_arc
            .as_ref()
            .as_any()
            .downcast_ref::<RocmStorage>()
            .expect("parent still RocmStorage")
            .copy_to_host()
            .expect("parent D2H after view drop");
        assert_eq!(
            parent_after, parent_bytes,
            "parent bytes changed after view drop — allocator corruption"
        );
    }
}
