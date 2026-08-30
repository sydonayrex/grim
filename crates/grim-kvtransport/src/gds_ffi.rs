//! FFI bindings for GDS / hipFile (Direct NVMe-GPU I/O).
//!
//! Provides dynamic loading of `libhipfile.so` (or `libcufile.so`) with
//! symbol verification, `#[repr(C)]` boundaries, and `catch_unwind` safety guards.

use libloading::{Library, Symbol};
use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};
use std::panic::catch_unwind;
use std::sync::atomic::{AtomicBool, Ordering};

/// Handle representing an opened hipFile registration.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HipFileHandle(pub *mut c_void);

unsafe impl Send for HipFileHandle {}
unsafe impl Sync for HipFileHandle {}

/// Driver registration parameters.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HipFileDriverStatus {
    pub is_available: bool,
    pub major_version: u32,
    pub minor_version: u32,
    pub flags: u32,
}

/// Dynamic function signatures for libhipfile.
type FnDriverOpen = unsafe extern "C" fn() -> c_int;
type FnDriverClose = unsafe extern "C" fn() -> c_int;
type FnFileHandleRegister = unsafe extern "C" fn(*const c_char, c_int) -> HipFileHandle;
type FnFileHandleDeregister = unsafe extern "C" fn(HipFileHandle) -> c_int;
type FnFileRead = unsafe extern "C" fn(HipFileHandle, *mut c_void, usize, i64, i64) -> isize;
type FnFileWrite = unsafe extern "C" fn(HipFileHandle, *const c_void, usize, i64, i64) -> isize;
type FnBufRegister = unsafe extern "C" fn(*const c_void, usize, c_int) -> c_int;
type FnBufDeregister = unsafe extern "C" fn(*const c_void) -> c_int;

/// Dynamically loaded hipFile runtime interface.
pub struct HipFileLib {
    _lib: Library,
    fn_driver_open: FnDriverOpen,
    fn_driver_close: FnDriverClose,
    fn_handle_reg: FnFileHandleRegister,
    fn_handle_dereg: FnFileHandleDeregister,
    fn_read: FnFileRead,
    fn_write: FnFileWrite,
    fn_buf_reg: FnBufRegister,
    fn_buf_dereg: FnBufDeregister,
}

static GDS_PROBED: AtomicBool = AtomicBool::new(false);
static GDS_AVAILABLE: AtomicBool = AtomicBool::new(false);

impl HipFileLib {
    /// Probe if `libhipfile.so` or `libcufile.so` is available on the system.
    pub fn probe_available() -> bool {
        if GDS_PROBED.load(Ordering::Relaxed) {
            return GDS_AVAILABLE.load(Ordering::Relaxed);
        }

        let avail = Self::load().is_some();
        GDS_AVAILABLE.store(avail, Ordering::Relaxed);
        GDS_PROBED.store(true, Ordering::Relaxed);
        avail
    }

    /// Dynamically load the library and resolve all required symbols with null-checks.
    pub fn load() -> Option<Self> {
        let candidate_libs = ["libhipfile.so", "libcufile.so", "libhipfile.so.1"];
        for name in candidate_libs {
            if let Ok(lib) = unsafe { Library::new(name) } {
                unsafe {
                    let fn_driver_open: Result<Symbol<FnDriverOpen>, _> = lib.get(b"cuFileDriverOpen\0");
                    let fn_driver_close: Result<Symbol<FnDriverClose>, _> = lib.get(b"cuFileDriverClose\0");
                    let fn_handle_reg: Result<Symbol<FnFileHandleRegister>, _> = lib.get(b"cuFileHandleRegister\0");
                    let fn_handle_dereg: Result<Symbol<FnFileHandleDeregister>, _> = lib.get(b"cuFileHandleDeregister\0");
                    let fn_read: Result<Symbol<FnFileRead>, _> = lib.get(b"cuFileRead\0");
                    let fn_write: Result<Symbol<FnFileWrite>, _> = lib.get(b"cuFileWrite\0");
                    let fn_buf_reg: Result<Symbol<FnBufRegister>, _> = lib.get(b"cuFileBufRegister\0");
                    let fn_buf_dereg: Result<Symbol<FnBufDeregister>, _> = lib.get(b"cuFileBufDeregister\0");

                    if let (
                        Ok(f_dopen),
                        Ok(f_dclose),
                        Ok(f_hreg),
                        Ok(f_hdereg),
                        Ok(f_rd),
                        Ok(f_wr),
                        Ok(f_breg),
                        Ok(f_bdereg),
                    ) = (
                        fn_driver_open,
                        fn_driver_close,
                        fn_handle_reg,
                        fn_handle_dereg,
                        fn_read,
                        fn_write,
                        fn_buf_reg,
                        fn_buf_dereg,
                    ) {
                        let f_dopen = *f_dopen;
                        let f_dclose = *f_dclose;
                        let f_hreg = *f_hreg;
                        let f_hdereg = *f_hdereg;
                        let f_rd = *f_rd;
                        let f_wr = *f_wr;
                        let f_breg = *f_breg;
                        let f_bdereg = *f_bdereg;

                        return Some(Self {
                            _lib: lib,
                            fn_driver_open: f_dopen,
                            fn_driver_close: f_dclose,
                            fn_handle_reg: f_hreg,
                            fn_handle_dereg: f_hdereg,
                            fn_read: f_rd,
                            fn_write: f_wr,
                            fn_buf_reg: f_breg,
                            fn_buf_dereg: f_bdereg,
                        });
                    }
                }
            }
        }
        None
    }

    /// Open the driver runtime interface.
    pub fn driver_open(&self) -> bool {
        let res = catch_unwind(|| unsafe { (self.fn_driver_open)() });
        res.map(|code| code == 0).unwrap_or(false)
    }

    /// Close the driver runtime interface.
    pub fn driver_close(&self) -> bool {
        let res = catch_unwind(|| unsafe { (self.fn_driver_close)() });
        res.map(|code| code == 0).unwrap_or(false)
    }

    /// Register a host/device buffer for direct DMA.
    pub fn register_buffer(&self, dev_ptr: *const c_void, size: usize, flags: i32) -> bool {
        let res = catch_unwind(|| unsafe { (self.fn_buf_reg)(dev_ptr, size, flags as c_int) });
        res.map(|code| code == 0).unwrap_or(false)
    }

    /// Deregister a buffer.
    pub fn deregister_buffer(&self, dev_ptr: *const c_void) -> bool {
        let res = catch_unwind(|| unsafe { (self.fn_buf_dereg)(dev_ptr) });
        res.map(|code| code == 0).unwrap_or(false)
    }

    /// Register a file path for direct DMA.
    pub fn register_file(&self, path: &str, flags: i32) -> Option<HipFileHandle> {
        let c_path = CString::new(path).ok()?;
        let res = catch_unwind(|| unsafe {
            (self.fn_handle_reg)(c_path.as_ptr(), flags as c_int)
        });
        match res {
            Ok(handle) if !handle.0.is_null() => Some(handle),
            _ => None,
        }
    }

    /// Deregister a previously registered file handle.
    pub fn deregister_file(&self, handle: HipFileHandle) -> bool {
        let res = catch_unwind(|| unsafe {
            (self.fn_handle_dereg)(handle)
        });
        res.map(|code| code == 0).unwrap_or(false)
    }

    /// Direct DMA read from file handle into device memory buffer.
    pub fn read_direct(
        &self,
        handle: HipFileHandle,
        dev_ptr: *mut c_void,
        size: usize,
        file_offset: i64,
        dev_offset: i64,
    ) -> isize {
        let res = catch_unwind(|| unsafe {
            (self.fn_read)(handle, dev_ptr, size, file_offset, dev_offset)
        });
        res.unwrap_or(-1)
    }

    /// Direct DMA write from device memory buffer into file handle.
    pub fn write_direct(
        &self,
        handle: HipFileHandle,
        dev_ptr: *const c_void,
        size: usize,
        file_offset: i64,
        dev_offset: i64,
    ) -> isize {
        let res = catch_unwind(|| unsafe {
            (self.fn_write)(handle, dev_ptr, size, file_offset, dev_offset)
        });
        res.unwrap_or(-1)
    }
}
