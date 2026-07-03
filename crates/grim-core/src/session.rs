//! `Session` — per-request mutable execution state.
//!
//! A trait object so libraries can box user-supplied sessions, and a
//! concrete `Inner` impl that holds a KV cache (when present) plus a
//! monotonically-increasing `current_pos` cursor.

use grim_tensor::{Device, Tensor};

use crate::error::Result;
use crate::kv_cache::KvCache;

/// Object-safe session interface used by `Model` traits (`CausalLm`,
/// `EncoderDecoderLm`). The simplest implementation is the `Inner`
/// concrete value returned from `Session::new_storage`.
pub trait SessionT: Send {
    fn device(&self) -> &Device;
    fn current_pos(&self) -> usize;
    fn advance_pos(&mut self, by: usize);
    fn has_kv(&self) -> bool;
    fn append_kv(&mut self, _k: &Tensor, _v: &Tensor) -> Result<()>;
}

/// Public trait-object alias used in `Model` trait DSL.
pub type DynSession = dyn SessionT;

/// A convenient concrete session. Holds an optional `KvCache` and tracks
/// positional advancement for RoPE / attention masks during decode.
pub struct Inner {
    pub device: Device,
    pub kv: Option<Box<dyn KvCache>>,
    pub current_pos: usize,
}

impl Inner {
    pub fn new(device: Device) -> Self {
        Self { device, kv: None, current_pos: 0 }
    }
    pub fn with_kv(device: Device, kv: Box<dyn KvCache>) -> Self {
        Self { device, kv: Some(kv), current_pos: 0 }
    }
}

impl SessionT for Inner {
    fn device(&self) -> &Device {
        &self.device
    }
    fn current_pos(&self) -> usize {
        self.current_pos
    }
    fn advance_pos(&mut self, by: usize) {
        self.current_pos += by;
    }
    fn has_kv(&self) -> bool {
        self.kv.is_some()
    }
    fn append_kv(&mut self, _k: &Tensor, _v: &Tensor) -> Result<()> {
        if let Some(kv) = self.kv.as_deref_mut() {
            kv.append_slot()?;
        }
        Ok(())
    }
}

impl Inner {
    /// Concrete-only escape hatch — call directly on `Inner` rather than
    /// through trait dispatch when you need a `&mut dyn KvCache`.
    pub fn with_kv_mut<R>(&mut self, f: &mut dyn FnMut(&mut dyn KvCache) -> Result<R>) -> Result<Option<R>> {
        if let Some(kv) = self.kv.as_deref_mut() {
            Ok(Some(f(kv)?))
        } else {
            Ok(None)
        }
    }
}

/// Public alias used everywhere `Session` is named as a concrete type.
pub struct Session;

impl Session {
    pub fn new(device: Device) -> Inner {
        Inner::new(device)
    }
    pub fn with_kv(device: Device, kv: Box<dyn KvCache>) -> Inner {
        Inner::with_kv(device, kv)
    }
}
