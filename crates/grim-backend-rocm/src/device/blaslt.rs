//! Capability-gated hipBLASLt/rocBLASLt probing.
//!
//! The decode path keeps its measured dot4/rocBLAS route. This module only
//! probes an installed BLASLt runtime and exposes an explicit selection
//! boundary for future prefill experiments; it never silently replaces GEMM.

use std::ffi::c_void;

use libloading::Library;

/// Result of probing the optional BLASLt runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlasLtProbe {
    pub library: Option<String>,
    pub version: Option<i32>,
    pub error: Option<String>,
}

impl BlasLtProbe {
    pub fn available(&self) -> bool {
        self.library.is_some() && self.version.is_some()
    }

    pub fn describe(&self) -> String {
        match (&self.library, self.version, &self.error) {
            (Some(library), Some(version), _) => format!("{library} version={version}"),
            (Some(library), None, Some(error)) => {
                format!("{library} loaded but capability query failed: {error}")
            }
            (Some(library), None, None) => format!("{library} loaded without a version"),
            (None, _, Some(error)) => format!("unavailable: {error}"),
            (None, _, None) => "unavailable".to_string(),
        }
    }
}

/// Conservative BLASLt routing decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlasLtSelection {
    /// The runtime is absent or its version could not be queried.
    Unavailable,
    /// The runtime exists, but no real matmul benchmark has been registered.
    NotMeasured,
    /// A future caller supplied a measured, capability-approved candidate.
    Eligible,
}

type CreateFn = unsafe extern "C" fn(*mut *mut c_void) -> i32;
type VersionFn = unsafe extern "C" fn(*mut c_void, *mut i32) -> i32;
type DestroyFn = unsafe extern "C" fn(*mut c_void) -> i32;

/// Probe the installed runtime without creating a permanent library handle.
///
/// The library is loaded only for the lifetime of this function. The returned
/// value is plain data, so it is safe to cache and inspect from model code.
pub fn probe_blaslt() -> BlasLtProbe {
    const LIBRARIES: [&str; 4] = [
        "libhipblaslt.so",
        "libhipblaslt.so.1",
        "librocblaslt.so",
        "librocblaslt.so.1",
    ];

    let mut load_errors = Vec::new();
    for library_name in LIBRARIES {
        let library = match unsafe { Library::new(library_name) } {
            Ok(library) => library,
            Err(error) => {
                load_errors.push(format!("{library_name}: {error}"));
                continue;
            }
        };
        let result = unsafe { probe_loaded_library(library_name, &library) };
        match result {
            Ok(version) => {
                return BlasLtProbe {
                    library: Some(library_name.to_string()),
                    version: Some(version),
                    error: None,
                };
            }
            Err(error) => load_errors.push(format!("{library_name}: {error}")),
        }
    }

    BlasLtProbe {
        library: None,
        version: None,
        error: (!load_errors.is_empty()).then(|| load_errors.join("; ")),
    }
}

unsafe fn probe_loaded_library(
    library_name: &'static str,
    library: &Library,
) -> Result<i32, String> {
    unsafe {
        let create = *library
            .get::<CreateFn>(b"hipblasLtCreate\0")
            .map_err(|error| error.to_string())?;
        let get_version = *library
            .get::<VersionFn>(b"hipblasLtGetVersion\0")
            .map_err(|error| error.to_string())?;
        let destroy = *library
            .get::<DestroyFn>(b"hipblasLtDestroy\0")
            .map_err(|error| error.to_string())?;

        let mut handle: *mut c_void = std::ptr::null_mut();
        let create_status = create(&mut handle);
        if create_status != 0 || handle.is_null() {
            return Err(format!(
                "{library_name}: hipblasLtCreate failed with status {create_status}"
            ));
        }
        let mut version: i32 = 0;
        let version_status = get_version(handle, &mut version);
        let destroy_status = destroy(handle);
        if version_status != 0 {
            return Err(format!(
                "{library_name}: hipblasLtGetVersion failed with status {version_status}"
            ));
        }
        if destroy_status != 0 {
            return Err(format!(
                "{library_name}: hipblasLtDestroy failed with status {destroy_status}"
            ));
        }
        Ok(version)
    }
}

/// Return whether a shape is eligible for a future measured BLASLt experiment.
///
/// Decode and quantized shapes deliberately remain on the measured dot4 path.
/// `measurement_verified` is an explicit caller assertion that a real shape
/// benchmark and parity check have been completed; it is not inferred from
/// library presence.
pub fn select_blaslt_candidate(
    probe: &BlasLtProbe,
    m: usize,
    n: usize,
    k: usize,
    measurement_verified: bool,
) -> BlasLtSelection {
    if !probe.available() {
        return BlasLtSelection::Unavailable;
    }
    if !measurement_verified {
        return BlasLtSelection::NotMeasured;
    }
    if m <= 8 || n < 256 || k < 256 {
        return BlasLtSelection::NotMeasured;
    }
    BlasLtSelection::Eligible
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_result_is_describable_without_a_gpu() {
        let probe = probe_blaslt();
        assert!(!probe.describe().is_empty());
    }

    #[test]
    fn absent_runtime_is_unavailable() {
        let probe = BlasLtProbe {
            library: None,
            version: None,
            error: Some("test".into()),
        };
        assert_eq!(
            select_blaslt_candidate(&probe, 32, 4096, 4096, true),
            BlasLtSelection::Unavailable
        );
    }

    #[test]
    fn unmeasured_prefill_is_not_eligible() {
        let probe = BlasLtProbe {
            library: Some("test".into()),
            version: Some(1),
            error: None,
        };
        assert_eq!(
            select_blaslt_candidate(&probe, 32, 4096, 4096, false),
            BlasLtSelection::NotMeasured
        );
    }

    #[test]
    fn decode_shape_stays_out_of_blaslt_even_after_measurement() {
        let probe = BlasLtProbe {
            library: Some("test".into()),
            version: Some(1),
            error: None,
        };
        assert_eq!(
            select_blaslt_candidate(&probe, 1, 4096, 4096, true),
            BlasLtSelection::NotMeasured
        );
    }

    #[test]
    fn measured_large_prefill_is_eligible() {
        let probe = BlasLtProbe {
            library: Some("test".into()),
            version: Some(1),
            error: None,
        };
        assert_eq!(
            select_blaslt_candidate(&probe, 32, 4096, 4096, true),
            BlasLtSelection::Eligible
        );
    }
}
