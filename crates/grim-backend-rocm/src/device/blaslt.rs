//! Capability-gated hipBLASLt/rocBLASLt probing.
//!
//! The decode path keeps its measured dot4/rocBLAS route. This module only
//! probes an installed BLASLt runtime and exposes an explicit selection
//! boundary for future prefill experiments; it never silently replaces GEMM.

use std::ffi::c_void;

use libloading::Library;

use crate::device::roc_device::RocmDevice;
use crate::memory::storage::RocmStorage;
use crate::{arg, linear_launch};

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

type MatrixLayoutCreateFn = unsafe extern "C" fn(*mut *mut c_void, i32, u64, u64, i64) -> i32;
type MatrixLayoutDestroyFn = unsafe extern "C" fn(*mut c_void) -> i32;
type MatmulDescCreateFn = unsafe extern "C" fn(*mut *mut c_void, i32, i32) -> i32;
type MatmulDescDestroyFn = unsafe extern "C" fn(*mut c_void) -> i32;
type MatmulDescSetAttributeFn = unsafe extern "C" fn(*mut c_void, i32, *const c_void, usize) -> i32;
type MatmulFn = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *const c_void,
    *const c_void,
    *mut c_void,
    *const c_void,
    *mut c_void,
    *const c_void,
    *const c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *const c_void,
    *mut c_void,
    usize,
    *mut c_void,
) -> i32;

type MatmulPreferenceCreateFn = unsafe extern "C" fn(*mut *mut c_void) -> i32;
type MatmulPreferenceDestroyFn = unsafe extern "C" fn(*mut c_void) -> i32;
type MatmulPreferenceSetAttributeFn =
    unsafe extern "C" fn(*mut c_void, i32, *const c_void, usize) -> i32;
type MatmulAlgoGetHeuristicFn = unsafe extern "C" fn(
    *mut c_void,  // handle
    *mut c_void,  // matmulDesc
    *mut c_void,  // Adesc
    *mut c_void,  // Bdesc
    *mut c_void,  // Cdesc
    *mut c_void,  // Ddesc
    *mut c_void,  // pref
    i32,          // requestedAlgoCount
    *mut HipblasLtMatmulHeuristicResult,
    *mut i32,     // returnAlgoCount
) -> i32;

/// Raw hipblasLtMatmulAlgo_t (opaque, trivially copyable/serializable).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct HipblasLtMatmulAlgo {
    pub data: [u8; 16],
    pub max_workspace_bytes: usize,
}

/// Raw hipblasLtMatmulHeuristicResult_t.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct HipblasLtMatmulHeuristicResult {
    pub algo: HipblasLtMatmulAlgo,
    pub workspace_size: usize,
    pub state: i32,
    pub waves_count: f32,
    pub reserved: [i32; 4],
}

/// A heuristic algo candidate returned by hipBLASLt.
#[derive(Debug, Clone, Copy)]
pub struct BlasLtAlgoCandidate {
    pub algo: HipblasLtMatmulAlgo,
    pub workspace_size: usize,
    pub waves_count: f32,
}

struct MatmulApi {
    _library: Library,
    create: CreateFn,
    destroy: DestroyFn,
    layout_create: MatrixLayoutCreateFn,
    layout_destroy: MatrixLayoutDestroyFn,
    desc_create: MatmulDescCreateFn,
    desc_destroy: MatmulDescDestroyFn,
    desc_set_attribute: MatmulDescSetAttributeFn,
    matmul: MatmulFn,
    pref_create: MatmulPreferenceCreateFn,
    pref_destroy: MatmulPreferenceDestroyFn,
    pref_set_attribute: MatmulPreferenceSetAttributeFn,
    algo_get_heuristic: MatmulAlgoGetHeuristicFn,
}

impl MatmulApi {
    unsafe fn open() -> Result<Self, String> {
        let mut load_errors = Vec::new();
        for library_name in ["libhipblaslt.so", "libhipblaslt.so.1", "librocblaslt.so"] {
            let library = match unsafe { Library::new(library_name) } {
                Ok(library) => library,
                Err(error) => {
                    load_errors.push(format!("{library_name}: {error}"));
                    continue;
                }
            };
            let result = (|| unsafe {
                Ok(Self {
                    create: *library
                        .get::<CreateFn>(b"hipblasLtCreate\0")
                        .map_err(|error| error.to_string())?,
                    destroy: *library
                        .get::<DestroyFn>(b"hipblasLtDestroy\0")
                        .map_err(|error| error.to_string())?,
                    layout_create: *library
                        .get::<MatrixLayoutCreateFn>(b"hipblasLtMatrixLayoutCreate\0")
                        .map_err(|error| error.to_string())?,
                    layout_destroy: *library
                        .get::<MatrixLayoutDestroyFn>(b"hipblasLtMatrixLayoutDestroy\0")
                        .map_err(|error| error.to_string())?,
                    desc_create: *library
                        .get::<MatmulDescCreateFn>(b"hipblasLtMatmulDescCreate\0")
                        .map_err(|error| error.to_string())?,
                    desc_destroy: *library
                        .get::<MatmulDescDestroyFn>(b"hipblasLtMatmulDescDestroy\0")
                        .map_err(|error| error.to_string())?,
                    desc_set_attribute: *library
                        .get::<MatmulDescSetAttributeFn>(b"hipblasLtMatmulDescSetAttribute\0")
                        .map_err(|error| error.to_string())?,
                    matmul: *library
                        .get::<MatmulFn>(b"hipblasLtMatmul\0")
                        .map_err(|error| error.to_string())?,
                    pref_create: *library
                        .get::<MatmulPreferenceCreateFn>(b"hipblasLtMatmulPreferenceCreate\0")
                        .map_err(|error| error.to_string())?,
                    pref_destroy: *library
                        .get::<MatmulPreferenceDestroyFn>(b"hipblasLtMatmulPreferenceDestroy\0")
                        .map_err(|error| error.to_string())?,
                    pref_set_attribute: *library
                        .get::<MatmulPreferenceSetAttributeFn>(b"hipblasLtMatmulPreferenceSetAttribute\0")
                        .map_err(|error| error.to_string())?,
                    algo_get_heuristic: *library
                        .get::<MatmulAlgoGetHeuristicFn>(b"hipblasLtMatmulAlgoGetHeuristic\0")
                        .map_err(|error| error.to_string())?,
                    _library: library,
                })
            })();
            if result.is_ok() {
                return result;
            }
            if let Err(error) = result {
                load_errors.push(format!("{library_name}: {error}"));
            }
        }
        Err(load_errors.join("; "))
    }

    unsafe fn set_desc_transpose(
        &self,
        desc: *mut c_void,
        attribute: i32,
        operation: i32,
    ) -> Result<(), String> {
        let status = unsafe {
            (self.desc_set_attribute)(
                desc,
                attribute,
                &operation as *const i32 as *const c_void,
                std::mem::size_of::<i32>(),
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(format!(
                "hipblasLtMatmulDescSetAttribute(transpose) failed: {status}"
            ))
        }
    }
}

/// Copy a column-major `[rows, cols]` matrix into a row-major destination.
pub fn launch_col_major_to_row_major(
    dev: &RocmDevice,
    src: &RocmStorage,
    dst: &RocmStorage,
    rows: usize,
    cols: usize,
    ld: usize,
) -> Result<(), String> {
    let mut src_ptr = src
        .device_ptr_checked()
        .map_err(|error| error.to_string())? as *mut c_void;
    let mut dst_ptr = dst
        .device_ptr_checked()
        .map_err(|error| error.to_string())? as *mut c_void;
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| "col-major output size overflow".to_string())?;
    let (grid, block) = linear_launch(total);
    let mut rows_i = rows as i32;
    let mut cols_i = cols as i32;
    let mut ld_i = ld as i32;
    dev.launch_compute_kernel(
        "grim_col_major_to_row_major_f32",
        grid,
        block,
        &mut [
            arg(&mut src_ptr),
            arg(&mut dst_ptr),
            arg(&mut rows_i),
            arg(&mut cols_i),
            arg(&mut ld_i),
        ],
    )
    .map(|_| ())
    .map_err(|error| error.to_string())
}

/// Execute a real canonical column-major FP32 `C = A @ B` through hipBLASLt.
///
/// All three buffers use canonical column-major physical layout. The caller
/// stages row-major model tensors into this layout before the call.
pub fn matmul_col_major_f32(
    stream: *mut c_void,
    a: *const c_void,
    b: *const c_void,
    d: *mut c_void,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), String> {
    matmul_col_major_f32_with_algo(stream, a, b, d, m, n, k, None, std::ptr::null_mut(), 0)
}

/// Like `matmul_col_major_f32` but allows selecting a specific heuristic algo
/// and workspace (astrum P1 tuning). `algo` is a pointer to a
/// `hipblasLtMatmulAlgo_t` (or null for default).
pub fn matmul_col_major_f32_with_algo(
    stream: *mut c_void,
    a: *const c_void,
    b: *const c_void,
    d: *mut c_void,
    m: usize,
    n: usize,
    k: usize,
    algo: Option<&HipblasLtMatmulAlgo>,
    workspace: *mut c_void,
    workspace_size: usize,
) -> Result<(), String> {
    if stream.is_null() || a.is_null() || b.is_null() || d.is_null() {
        return Err("matmul_col_major_f32 received a null pointer".into());
    }
    if m == 0 || n == 0 || k == 0 {
        return Err("matmul_col_major_f32 received an empty shape".into());
    }
    let api = unsafe { MatmulApi::open()? };
    let mut handle: *mut c_void = std::ptr::null_mut();
    let mut desc: *mut c_void = std::ptr::null_mut();
    let mut a_layout: *mut c_void = std::ptr::null_mut();
    let mut b_layout: *mut c_void = std::ptr::null_mut();
    let mut d_layout: *mut c_void = std::ptr::null_mut();
    let alpha = 1.0f32;
    let beta = 0.0f32;
    let result = unsafe {
        let create_status = (api.create)(&mut handle);
        if create_status != 0 || handle.is_null() {
            return Err(format!("hipblasLtCreate failed: {create_status}"));
        }
        let desc_status = (api.desc_create)(&mut desc, 2, 0); // COMPUTE_32F, F32 scale
        if desc_status != 0 || desc.is_null() {
            let _ = (api.destroy)(handle);
            return Err(format!("hipblasLtMatmulDescCreate failed: {desc_status}"));
        }
        if let Err(error) = api.set_desc_transpose(desc, 0, 111) {
            let _ = (api.desc_destroy)(desc);
            let _ = (api.destroy)(handle);
            return Err(error);
        }
        if let Err(error) = api.set_desc_transpose(desc, 1, 111) {
            let _ = (api.desc_destroy)(desc);
            let _ = (api.destroy)(handle);
            return Err(error);
        }
        let a_status = (api.layout_create)(&mut a_layout, 0, m as u64, k as u64, m as i64);
        let b_status = (api.layout_create)(&mut b_layout, 0, k as u64, n as u64, k as i64);
        let d_status = (api.layout_create)(&mut d_layout, 0, m as u64, n as u64, m as i64);
        if a_status != 0 || b_status != 0 || d_status != 0 {
            if !a_layout.is_null() {
                let _ = (api.layout_destroy)(a_layout);
            }
            if !b_layout.is_null() {
                let _ = (api.layout_destroy)(b_layout);
            }
            if !d_layout.is_null() {
                let _ = (api.layout_destroy)(d_layout);
            }
            let _ = (api.desc_destroy)(desc);
            let _ = (api.destroy)(handle);
            return Err(format!(
                "hipblasLtMatrixLayoutCreate failed: A={a_status} B={b_status} D={d_status}"
            ));
        }
        let algo_ptr = match algo {
            Some(a) => a as *const HipblasLtMatmulAlgo as *const c_void,
            None => std::ptr::null(),
        };
        let matmul_status = (api.matmul)(
            handle,
            desc,
            &alpha as *const f32 as *const c_void,
            a,
            a_layout,
            b,
            b_layout,
            &beta as *const f32 as *const c_void,
            d as *const c_void,
            d_layout,
            d,
            d_layout,
            algo_ptr,
            workspace,
            workspace_size,
            stream,
        );
        let _ = (api.layout_destroy)(a_layout);
        let _ = (api.layout_destroy)(b_layout);
        let _ = (api.layout_destroy)(d_layout);
        let _ = (api.desc_destroy)(desc);
        let _ = (api.destroy)(handle);
        if matmul_status == 0 {
            Ok(())
        } else {
            Err(format!("hipblasLtMatmul failed: {matmul_status}"))
        }
    };
    result
}

/// Query hipBLASLt heuristic algorithm candidates for a canonical column-major
/// FP32 matmul of shape (m,n,k) (C[M,N] = A[M,K] @ B[K,N]). Returns candidates
/// in heuristic-preferred order. `max_workspace_bytes` sets the preference
/// (0 = no workspace). astrum P1 / Step 6.
pub fn get_heuristic_algos_f32(
    m: usize,
    n: usize,
    k: usize,
    requested: i32,
    max_workspace_bytes: usize,
) -> Result<Vec<BlasLtAlgoCandidate>, String> {
    if m == 0 || n == 0 || k == 0 || requested <= 0 {
        return Ok(Vec::new());
    }
    let api = unsafe { MatmulApi::open()? };
    unsafe {
        let mut handle: *mut c_void = std::ptr::null_mut();
        let mut desc: *mut c_void = std::ptr::null_mut();
        let mut a_layout: *mut c_void = std::ptr::null_mut();
        let mut b_layout: *mut c_void = std::ptr::null_mut();
        let mut c_layout: *mut c_void = std::ptr::null_mut();
        let mut d_layout: *mut c_void = std::ptr::null_mut();
        let mut pref: *mut c_void = std::ptr::null_mut();

        let cleanup = |handle: *mut c_void,
                       desc: *mut c_void,
                       a_layout: *mut c_void,
                       b_layout: *mut c_void,
                       c_layout: *mut c_void,
                       d_layout: *mut c_void,
                       pref: *mut c_void,
                       api: &MatmulApi| {
            if !pref.is_null() {
                let _ = (api.pref_destroy)(pref);
            }
            if !d_layout.is_null() {
                let _ = (api.layout_destroy)(d_layout);
            }
            if !c_layout.is_null() {
                let _ = (api.layout_destroy)(c_layout);
            }
            if !b_layout.is_null() {
                let _ = (api.layout_destroy)(b_layout);
            }
            if !a_layout.is_null() {
                let _ = (api.layout_destroy)(a_layout);
            }
            if !desc.is_null() {
                let _ = (api.desc_destroy)(desc);
            }
            if !handle.is_null() {
                let _ = (api.destroy)(handle);
            }
        };

        let st = (api.create)(&mut handle);
        if st != 0 || handle.is_null() {
            cleanup(handle, desc, a_layout, b_layout, c_layout, d_layout, pref, &api);
            return Err(format!("hipblasLtCreate failed: {st}"));
        }
        let st = (api.desc_create)(&mut desc, 2, 0); // COMPUTE_32F, scale F32
        if st != 0 || desc.is_null() {
            cleanup(handle, desc, a_layout, b_layout, c_layout, d_layout, pref, &api);
            return Err(format!("hipblasLtMatmulDescCreate failed: {st}"));
        }
        if let Err(e) = api.set_desc_transpose(desc, 0, 111) {
            cleanup(handle, desc, a_layout, b_layout, c_layout, d_layout, pref, &api);
            return Err(e);
        }
        if let Err(e) = api.set_desc_transpose(desc, 1, 111) {
            cleanup(handle, desc, a_layout, b_layout, c_layout, d_layout, pref, &api);
            return Err(e);
        }
        // layouts: R_32F (0), column-major leading dims m,k,n,m
        let st_a = (api.layout_create)(&mut a_layout, 0, m as u64, k as u64, m as i64);
        let st_b = (api.layout_create)(&mut b_layout, 0, k as u64, n as u64, k as i64);
        let st_c = (api.layout_create)(&mut c_layout, 0, m as u64, n as u64, m as i64);
        let st_d = (api.layout_create)(&mut d_layout, 0, m as u64, n as u64, m as i64);
        if st_a != 0 || st_b != 0 || st_c != 0 || st_d != 0 || a_layout.is_null() || b_layout.is_null() || c_layout.is_null() || d_layout.is_null() {
            cleanup(handle, desc, a_layout, b_layout, c_layout, d_layout, pref, &api);
            return Err(format!("hipblasLtMatrixLayoutCreate failed: A={st_a} B={st_b} C={st_c} D={st_d}"));
        }
        let st = (api.pref_create)(&mut pref);
        if st != 0 || pref.is_null() {
            cleanup(handle, desc, a_layout, b_layout, c_layout, d_layout, pref, &api);
            return Err(format!("hipblasLtMatmulPreferenceCreate failed: {st}"));
        }
        // HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1
        let ws = max_workspace_bytes as u64;
        let st = (api.pref_set_attribute)(
            pref,
            1,
            &ws as *const u64 as *const c_void,
            std::mem::size_of::<u64>(),
        );
        if st != 0 {
            cleanup(handle, desc, a_layout, b_layout, c_layout, d_layout, pref, &api);
            return Err(format!("hipblasLtMatmulPreferenceSetAttribute(MAX_WORKSPACE) failed: {st}"));
        }

        let mut results: Vec<HipblasLtMatmulHeuristicResult> = vec![
            HipblasLtMatmulHeuristicResult {
                algo: HipblasLtMatmulAlgo { data: [0u8; 16], max_workspace_bytes: 0 },
                workspace_size: 0,
                state: 0,
                waves_count: 0.0,
                reserved: [0i32; 4],
            };
            requested as usize
        ];
        let mut returned: i32 = 0;
        let st = (api.algo_get_heuristic)(
            handle,
            desc,
            a_layout,
            b_layout,
            c_layout,
            d_layout,
            pref,
            requested,
            results.as_mut_ptr(),
            &mut returned,
        );
        cleanup(handle, desc, a_layout, b_layout, c_layout, d_layout, pref, &api);
        if st != 0 {
            return Err(format!("hipblasLtMatmulAlgoGetHeuristic failed: {st}"));
        }
        let nret = (returned.max(0) as usize).min(results.len());
        let mut out = Vec::with_capacity(nret);
        for r in results.into_iter().take(nret) {
            if r.state == 0 { // HIPBLAS_STATUS_SUCCESS
                out.push(BlasLtAlgoCandidate {
                    algo: r.algo,
                    workspace_size: r.workspace_size,
                    waves_count: r.waves_count,
                });
            }
        }
        Ok(out)
    }
}

/// Return whether the explicit model-prefill experiment is enabled.
///
/// This is intentionally opt-in. The default LFM2/Q8 decode path remains on
/// its measured dot4/rocBLAS routes; only native-F32 multi-token matmuls can
/// enter the BLASLt candidate below.
pub fn blaslt_prefill_enabled() -> bool {
    matches!(
        std::env::var("GRIM_BLASLT_PREFILL").as_deref(),
        Ok("1" | "true" | "yes" | "on")
    )
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
