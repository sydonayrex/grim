//! Implementation of `RocmCachingAllocator` — a size-bucketed free-list [see: `hipMalloc`, `hipFree`, `RocmStorage::drop`, `Send + Sync`]

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use grim_tensor::error::Result;

use crate::{check_hip, hipFree, hipMalloc};

/// Size-bucketed caching allocator for device memory. [see: `hipMalloc`, `hipFree`, `Drop for RocmStorage`, `Arc`]
#[derive(Debug)]
pub struct RocmCachingAllocator {
    /// Unique instance id (audit diagnosis: instance address reuse lies).
    id: usize,
    /// Free-list: size class -> available device pointers (stored as `u64` so the [see: `Send + Sync`]
    /// P0-1: each entry carries an event recorded on the stream that was
    /// current at free() time; a later `alloc` reusing the block waits on
    /// that event first, so an in-flight consumer on another stream can
    /// never race the new owner's writes.
    pool: Mutex<HashMap<usize, Vec<PoolEntry>>>,
    /// Total bytes currently held in `pool` (not returned to the driver).
    cached_bytes: Mutex<usize>,
    /// Soft cap on `cached_bytes`. Once exceeded, freed buffers are actually [see: `hipFree`]
    cap_bytes: usize,
    /// Device ordinal this allocator serves. Pinned around every driver call so a thread whose
    /// HIP context drifted to another device cannot `hipMalloc`/`hipFree` against the wrong one (WI-M1 context discipline).
    ordinal: usize,
    /// Count of real `hipMalloc` calls (misses). Always incremented.
    malloc_count: AtomicUsize,
    /// Count of real `hipFree` calls (evictions / cap overflow). Always incremented.
    free_count: AtomicUsize,
    /// GRIM_ALLOC_AUDIT=1: live-set integrity audit. Maps every handed-out
    /// pointer to its size class; a pointer handed out twice while live, or
    /// freed twice, is the deterministic aliasing that produced the d2d
    /// decode gibberish.
    live: Mutex<std::collections::HashMap<u64, usize>>,
    /// GRIM_ALLOC_AUDIT=1: next live-bytes total (GiB) at which to log.
    next_live_report: AtomicUsize,
    /// Pool-disabled small-allocation slabs (see `alloc`).
    slabs: Mutex<Vec<Slab>>,
    /// Recycled slab slots: slot class -> freed device addresses.
    slab_free: Mutex<HashMap<usize, Vec<u64>>>,
}

/// A 128 MiB slab bump-allocated for small blocks (pool-disabled mode).
#[derive(Debug)]
struct Slab {
    base: u64,
    size: usize,
    bump: usize,
}

/// A pooled device block plus the fence event recorded when it was freed.
#[derive(Debug)]
struct PoolEntry {
    ptr: u64,
    event: *mut c_void,
}

unsafe impl Send for PoolEntry {}

static NEXT_ALLOCATOR_ID: AtomicUsize = AtomicUsize::new(1);

impl RocmCachingAllocator {
    pub fn new(ordinal: usize, cap_bytes: usize) -> Self {
        let id = NEXT_ALLOCATOR_ID.fetch_add(1, Ordering::Relaxed);
        Self {
            id,

            pool: Mutex::new(HashMap::new()),
            cached_bytes: Mutex::new(0),
            cap_bytes,
            ordinal,
            malloc_count: AtomicUsize::new(0),
            free_count: AtomicUsize::new(0),
            live: Mutex::new(std::collections::HashMap::new()),
            slabs: Mutex::new(Vec::new()),
            slab_free: Mutex::new(HashMap::new()),
            next_live_report: AtomicUsize::new(1 << 30),
        }
    }

    /// Round a byte size up to the next power of two. Class 0 is treated as 1 to [see: `hipMalloc`]
    pub fn size_class_of(bytes: usize) -> usize { Self::size_class(bytes) }

    /// Slab sub-allocation bounds (pool-disabled mode, small blocks only).
    const SLAB_SIZE: usize = 128 << 20;
    const SMALL_MAX: usize = 4 << 20;
    const SLOT_GRAN: usize = 4096;

    fn pooling_enabled() -> bool {
        std::env::var("GRIM_ALLOC_POOL").as_deref() == Ok("1")
    }

    fn size_class(bytes: usize) -> usize {
        if bytes <= 1 {
            1
        } else if std::env::var("GRIM_ALLOC_POOL").as_deref() == Ok("1") {
            bytes.next_power_of_two()
        } else {
            // Pool disabled (the default since a6900918): power-of-two binning
            // serves reuse only, and with a real free per alloc it is pure
            // waste — it inflated the 12.92 GiB Xing4.0 checkpoint to ~15 GiB
            // of blocks and exhausted a 17.1 GB card. Exact size, 256-B aligned.
            bytes.div_ceil(256) * 256
        }
    }

    /// Allocate a device buffer of at least `bytes` usable bytes, reusing a pooled
    pub fn alloc(&self, bytes: usize) -> Result<*mut c_void> {
        let cls = Self::size_class(bytes);
        // Pool disabled (the default): serve SMALL allocations from 128 MiB
        // slabs. gfx1200's hipMalloc commit granularity costs ~0.5 MB per
        // block; a 7.7k-block checkpoint load (many KB-scale norm/bias/
        // scale tensors) wasted 3.5 GiB of VRAM to that overhead alone and
        // exhausted the card (probe: 11.39 GiB requested -> 15.40 GiB
        // resident). Slab slots are 4 KiB-granular; freed slots recycle.
        if !Self::pooling_enabled() && bytes <= Self::SMALL_MAX {
            let slot_cls = bytes.div_ceil(Self::SLOT_GRAN) * Self::SLOT_GRAN;
            if let Ok(mut frees) = self.slab_free.lock() {
                if let Some(ptr) = frees.get_mut(&slot_cls).and_then(|v| v.pop()) {
                    self.audit_trace("SPOP ", ptr, slot_cls);
                    self.audit_insert(ptr, slot_cls);
                    return Ok(ptr as *mut c_void);
                }
            }
            if let Ok(mut slabs) = self.slabs.lock() {
                let need = slot_cls;
                // Bump from the last slab if it fits; else hipMalloc a new one.
                let bump_from = slabs
                    .last_mut()
                    .filter(|s| s.bump + need <= s.size)
                    .map(|s| (s.base + s.bump as u64, s.size));
                let (ptr, fresh_slab) = match bump_from {
                    Some((ptr, _size)) => (ptr, None),
                    None => {
                        let _guard =
                            crate::device::util::DeviceGuard::set(self.ordinal as i32);
                        let mut p: *mut c_void = std::ptr::null_mut();
                        check_hip(
                            "hipMalloc(slab)",
                            unsafe { hipMalloc(&mut p, Self::SLAB_SIZE) },
                        )?;
                        drop(_guard);
                        self.malloc_count.fetch_add(1, Ordering::Relaxed);
                        (p as u64, Some(Self::SLAB_SIZE))
                    }
                };
                if let Some(size) = fresh_slab {
                    slabs.push(Slab { base: ptr, size, bump: 0 });
                }
                if let Some(s) = slabs.last_mut() {
                    s.bump += need;
                }
                self.audit_trace("SALLOC", ptr, slot_cls);
                self.audit_insert(ptr, slot_cls);
                return Ok(ptr as *mut c_void);
            }
        }
        let reused = {
            let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
            pool.get_mut(&cls).and_then(|v| v.pop())
        };
        if let Some(entry) = reused {
            // Buffer leaves the pool: adjust cached accounting, and order the
            // new owner's writes after EVERY in-flight consumer of this block.
            //
            // The recorded event only fences the stream that was current at
            // `free()` time. The d2d decode path runs kernels on capture
            // streams that are not current when their output buffers are
            // dropped, so a stream-ordered wait let the new owner overwrite a
            // block a consumer on another stream was still reading/writing —
            // the Qwen3.5-9B d2d decode gibberish (h_normed [1024..4096)
            // garbage, GRIM_ALLOC_NO_POOL=1 clean). A device-wide
            // synchronize is correct for any stream; at decode's ~600
            // ms/token it is unmeasurable.
            self.audit_trace("POP ", entry.ptr, cls);
            self.audit_insert(entry.ptr, cls);
            if let Ok(mut cached) = self.cached_bytes.lock() {
                *cached = cached.saturating_sub(cls);
            }
            {
                let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
                unsafe {
                    let _ = crate::hipDeviceSynchronize();
                }
            }
            if !entry.event.is_null() {
                // The event fence is superseded by the synchronize above.
                unsafe {
                    let _ = crate::hipEventDestroy(entry.event);
                }
            }
            return Ok(entry.ptr as *mut c_void);
        }

        // WI-M1: `hipMalloc` allocates in the calling thread's current device context.
        // Pin this allocator's ordinal - a pool miss from a drifted thread must not materialise.
        let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let mut dev_ptr_void: *mut c_void = std::ptr::null_mut();
        let res = check_hip("hipMalloc", unsafe { hipMalloc(&mut dev_ptr_void, cls) });
        if res.is_err() {
            if std::env::var_os("GRIM_ALLOC_TRACE").is_some() {
                eprintln!(
                    "[alloc] hipMalloc({cls}) FAILED — empty_cache + one retry (managed fallback next if this fails)"
                );
            }
            self.empty_cache();
            check_hip("hipMalloc", unsafe { hipMalloc(&mut dev_ptr_void, cls) })?;
        }
        drop(_guard);
        self.malloc_count.fetch_add(1, Ordering::Relaxed);
        self.audit_trace("MALLOC", dev_ptr_void as u64, cls);
        self.audit_insert(dev_ptr_void as u64, cls);
        Ok(dev_ptr_void)
    }

    fn audit_on(&self) -> bool {
        std::env::var("GRIM_ALLOC_AUDIT").as_deref() == Ok("1")
    }

    fn audit_trace(&self, what: &str, ptr: u64, cls: usize) {
        if !self.audit_on() || cls != 16384 { return; }
        eprintln!("[alloc-trace] {} {ptr:#x} c{cls} id{}", what, self.id);
    }

    fn audit_insert(&self, ptr: u64, cls: usize) {
        // `live` is an INVARIANT, not a debug aid: `empty_cache` refuses to unmap
        // a slab or pool buffer that a live entry still points into, and it can
        // only do that if this map is always accurate. Gating the maintenance on
        // `audit_on()` left it empty in production, which made the guard a no-op
        // and let a slab be unmapped under a live RocmStorage.
        let mut live = self.live.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(old) = live.insert(ptr, cls) {
            if self.audit_on() {
                eprintln!("[alloc-audit] DOUBLE-BOOKING: {ptr:#x} id{} handed out while live (class {old}, now {cls})", self.id);
            }
        }
        if self.audit_on() {
            let live_bytes: usize = live.values().sum();
            if live_bytes >= self.next_live_report.load(Ordering::Relaxed) {
                eprintln!("[alloc-audit] live total: {:.2} GiB across {} blocks", live_bytes as f64 / (1 << 30) as f64, live.len());
                self.next_live_report.fetch_add(1 << 30, Ordering::Relaxed);
            }
        }
    }

    fn audit_remove(&self, ptr: u64, cls: usize) {
        // Same invariant reasoning as `audit_insert`: the removal must always
        // happen, the diagnostics around it are what `audit_on()` gates.
        let mut live = self.live.lock().unwrap_or_else(|e| e.into_inner());
        match live.remove(&ptr) {
            None => {
                if !self.audit_on() {
                    return;
                }
                eprintln!("[alloc-audit] FREE OF UNKNOWN: {ptr:#x} class {cls} not live");
                // One-shot backtrace names the duplicate-free caller.
                static ONCE: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!("{:?}", std::backtrace::Backtrace::force_capture());
                }
            }
            Some(c) if c != cls => {
                if self.audit_on() {
                    eprintln!("[alloc-audit] CLASS MISMATCH on free: {ptr:#x} live {c}, freed as {cls} (id{}))", self.id);
                }
            }
            _ => {}
        }
    }

    /// Return a buffer to the pool (or actually free it if over cap).
    ///
    /// DEFAULT IS NO POOL. The size-classed free list has a broken ownership
    /// invariant: a block can be pooled while a duplicate owner still holds
    /// it, and the driver can then hand the same address to a fresh
    /// `hipMalloc` — two live owners, deterministic memory corruption. This
    /// was the Qwen3.5-9B d2d decode gibberish (arbitration: pool on ->
    /// h_normed[1024..4096) garbage and step-1 logprobs "&lt" -1.93; pool
    /// off -> byte-equal to the host reference). GRIM_ALLOC_AUDIT=1 traces
    /// the violations ([alloc-audit]/[alloc-trace]); GRIM_ALLOC_POOL=1 opts
    /// back into reuse for perf work once the duplicate-owner site is fixed.
    pub fn free(&self, ptr: *mut c_void, bytes: usize) {
        // Audit BEFORE any branch: the no-pool path returns below and the
        // live-set must still see the release, or every later handout of a
        // legitimately recycled driver address prints as phantom
        // double-booking.
        // Slot class must match what alloc() recorded for slab-served small
        // blocks (4 KiB granularity), or the audit prints phantom CLASS
        // MISMATCH on every small free.
        let cls = if !Self::pooling_enabled()
            && bytes <= Self::SMALL_MAX
            && !ptr.is_null()
            && self
                .slabs
                .lock()
                .map(|slabs| {
                    let p = ptr as u64;
                    slabs.iter()
                        .any(|s| p >= s.base && p + (bytes.div_ceil(Self::SLOT_GRAN) * Self::SLOT_GRAN) as u64 <= s.base + s.size as u64)
                })
                .unwrap_or(false)
        {
            bytes.div_ceil(Self::SLOT_GRAN) * Self::SLOT_GRAN
        } else {
            Self::size_class(bytes)
        };
        self.audit_trace("FREE ", ptr as u64, cls);
        self.audit_remove(ptr as u64, cls);
        if std::env::var("GRIM_ALLOC_POOL").as_deref() != Ok("1") {
            // Small slab slot: recycle in place (no driver work). The slot
            // class is recoverable from the request size only if the caller
            // frees what it allocated — true for every RocmStorage — and the
            // range check guards against foreign pointers.
            if bytes <= Self::SMALL_MAX {
                let slot_cls = bytes.div_ceil(Self::SLOT_GRAN) * Self::SLOT_GRAN;
                if let Ok(slabs) = self.slabs.lock() {
                    let p = ptr as u64;
                    if slabs
                        .iter()
                        .any(|s| p >= s.base && p + slot_cls as u64 <= s.base + s.size as u64)
                    {
                        if let Ok(mut frees) = self.slab_free.lock() {
                            frees.entry(slot_cls).or_default().push(p);
                        }
                        return;
                    }
                }
            }
            // Correct-by-default path: device-wide synchronize (any stream's
            // in-flight consumer is done) then a real driver release.
            // WI-M1: pin the owning ordinal for the real release.
            let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
            unsafe {
                let _ = crate::hipDeviceSynchronize();
                let _ = hipFree(ptr);
            }
            self.free_count.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let over_cap = {
            let cached = self.cached_bytes.lock().unwrap_or_else(|e| e.into_inner());
            *cached + cls > self.cap_bytes
        };
        if over_cap || ptr.is_null() {
            // P0-1 (PLAN-improve-grim-perf): a real release must be ordered
            // after EVERY in-flight consumer, not just null-stream work. The
            // previous `hipFreeAsync(ptr, null stream)` could unmap a page
            // while a compute-stream kernel still read it — the "Page not
            // present" UAF (fused Q8_0 QKV fault, GPU 1 speed test). Eviction
            // is rare (only over cap), so a device-wide synchronize before
            // the free costs nothing measurable and is correct for any stream.
            let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
            unsafe {
                let _ = crate::hipDeviceSynchronize();
                let _ = hipFree(ptr);
            }
            self.free_count.fetch_add(1, Ordering::Relaxed);
            return;
        }
        // P0-1: fence the block against in-flight consumers. Record an event
        // on the stream that was current at free() time — any future reuse
        // waits on it, so a queued kernel reading this buffer can never race
        // the new owner's writes, regardless of which stream either ran on.
        let fence = crate::device::roc_device::RocmDevice::shared(self.ordinal).active_stream();
        let event: *mut c_void = if fence.is_null() {
            std::ptr::null_mut()
        } else {
            let mut ev: *mut c_void = std::ptr::null_mut();
            // SAFETY: fresh event handle; stream is the device's live stream.
            if unsafe { crate::hipEventCreate(&mut ev) } == 0 && !fence.is_null() {
                // SAFETY: as above.
                unsafe {
                    let _ = crate::hipEventRecord(ev, fence);
                }
                ev
            } else {
                std::ptr::null_mut()
            }
        };
        {
            let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
            pool.entry(cls).or_default().push(PoolEntry {
                ptr: ptr as u64,
                event,
            });
            let mut cached = self.cached_bytes.lock().unwrap_or_else(|e| e.into_inner());
            *cached += cls;
        }
    }

    /// Release every pooled buffer back to the driver. Mirrors `torch.cuda.empty_cache()`.
    pub fn empty_cache(&self) {
        // Pin the device (P1-7 discipline): hipDeviceSynchronize targets the calling thread's current device, which may
        // differ from `self.ordinal` on a multi-GPU host where another device's teardown ran on this thread.
        let _guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        unsafe {
            let _ = crate::hipDeviceSynchronize();
        }
        // Release slab sub-allocations too.
        //
        // A slab may only be unmapped once NOTHING live points into it. `live`
        // maps every handed-out pointer to its size class, and a slab-served
        // pointer is an interior offset into its slab, so a live entry inside
        // [base, base+size) means some RocmStorage still reads and writes
        // through this slab.
        //
        // Freeing it regardless is a use-after-free, and a silent one: the
        // storage keeps an `Arc` to this allocator, so the allocator outlives
        // the slab and nothing downstream notices until a DtoH reports the
        // pointer as unregistered (`hipPointerGetAttributes` -> device -2).
        // That is what `MoeFfn::forward_rocm` hit -- it allocates its output
        // from a LOCAL `RocmDevice`, returns the tensor, and drops the device
        // on the way out, and `Drop` calls here.
        //
        // The same guard covers pool buffers: a live entry there is a handed-out
        // block, and unmapping it is the identical defect.
        //
        // Copied out rather than held as a guard so no lock is taken while
        // another is acquired.
        let live: std::collections::HashSet<u64> = self
            .live
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .keys()
            .copied()
            .collect();
        {
            if let Ok(mut frees) = self.slab_free.lock() {
                frees.clear();
            }
            if let Ok(mut slabs) = self.slabs.lock() {
                let mut keep = Vec::with_capacity(slabs.len());
                for s in slabs.drain(..) {
                    let lo = s.base;
                    let hi = s.base + s.size as u64;
                    if live.iter().any(|p| *p >= lo && *p < hi) {
                        // Still referenced: leave mapped. It becomes freeable on
                        // a later empty_cache once its last slot is released.
                        keep.push(s);
                    } else {
                        unsafe {
                            let _ = hipFree(s.base as *mut c_void);
                        }
                    }
                }
                *slabs = keep;
            }
        }
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        for (_cls, bufs) in pool.drain() {
            for p in bufs {
                if live.contains(&p.ptr) {
                    continue;
                }
                unsafe {
                    let _ = hipFree(p.ptr as *mut c_void);
                    if !p.event.is_null() {
                        let _ = crate::hipEventDestroy(p.event);
                    }
                }
                self.free_count.fetch_add(1, Ordering::Relaxed);
            }
        }
        *self.cached_bytes.lock().unwrap_or_else(|e| e.into_inner()) = 0;
    }

    /// `(malloc_count, free_count)` — real driver allocation calls since start.
    pub fn stats(&self) -> (usize, usize) {
        (
            self.malloc_count.load(Ordering::Relaxed),
            self.free_count.load(Ordering::Relaxed),
        )
    }
}
