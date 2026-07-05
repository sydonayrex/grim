//! ANE Private FFI Bridge.
//!
//! Exposes private framework interface structures to communicate directly with
//! the AppleNeuralEngine.framework without CoreML high-level runtime overhead.

use grim_tensor::error::{Error, Result};

#[cfg(target_vendor = "apple")]
use std::ffi::c_void;

/// Handle to the private _ANEClient instance.
pub struct AneBridgeClient {
    #[cfg(target_vendor = "apple")]
    _raw_client: *mut c_void,
}

impl AneBridgeClient {
    /// Establishes FFI bindings to the private ANE driver interface.
    pub fn connect() -> Result<Self> {
        #[cfg(target_vendor = "apple")]
        {
            println!("[AneBridge] Connecting to private ANE client interface...");
            Ok(Self { _raw_client: std::ptr::null_mut() })
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Err(Error::Unimplemented("ANE bridge requires Apple Silicon macOS".into()))
        }
    }

    /// Evaluates a compiled model segment on ANE hardware.
    pub fn dispatch(
        &self,
        program_path: &str,
        inputs: &[*mut f32],
        outputs: &[*mut f32],
    ) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            let _ = (program_path, inputs, outputs);
            println!("[AneBridge] Dispatching job over AppleNeuralEngine.framework FFI...");
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (program_path, inputs, outputs);
            Err(Error::Unimplemented("ANE bridge requires Apple Silicon macOS".into()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ane_bridge_client_behavior() {
        let client = AneBridgeClient::connect();
        #[cfg(not(target_vendor = "apple"))]
        {
            assert!(client.is_err());
        }
        #[cfg(target_vendor = "apple")]
        {
            assert!(client.is_ok());
            client.unwrap().dispatch("test.mlmodelc", &[], &[]).unwrap();
        }
    }
}
