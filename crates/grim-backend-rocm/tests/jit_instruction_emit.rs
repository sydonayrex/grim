//! T1 integration: assert grim's **JIT-compiled** aggregate actually emits the
//! native FP8 dot instruction on gfx12.
//!
//! # Why this is not a source-level test
//!
//! `builtin_name_oracle_tests` proves clang *knows* the builtin and that
//! `dot_gemv.rs` *names* it. Neither proves the compiled kernel uses it: the
//! `__has_builtin` guard is evaluated per translation-unit inside hiprtc, and a
//! guard that resolves false in the JIT's compile options produces a kernel that
//! is correct but ~an order of magnitude slower, with no error anywhere. The
//! only way to close that gap is to read the instruction stream hiprtc emits.
//!
//! This is the test that would have caught the `fdot4` typo at the point it
//! mattered, rather than a compile-probe on an unrelated TU.
//!
//! Requires a GPU and is `#[ignore]`d per `tests/README.md`:
//! ```sh
//! GRIM_RUN_GPU_TESTS=1 GRIM_GPU_TEST=1 \
//!   cargo test -p grim-backend-rocm --test jit_instruction_emit -- --ignored --nocapture
//! ```

use grim_backend_rocm::device::helpers::jit_compile_hsaco;
use grim_backend_rocm::device::util::{gpu_test_enabled, gpu_test_lock};

/// Must be a kernel symbol present in the aggregate: `jit_compile_hsaco`
/// passes this to `hiprtcAddNameExpression`, not as a program name.
const ENTRY: &str = "grim_dot4_fp8_gemv";

/// The instruction the fixed `grim_fdot4_fp8` must lower to.
const WANTED: &str = "v_dot4_f32_fp8_fp8";

#[test]
#[ignore = "needs a gfx12 device + hiprtc; see module doc"]
fn jit_aggregate_emits_native_fp8_dot() {
    if !gpu_test_enabled() {
        eprintln!("skip: GRIM_GPU_TEST not set");
        return;
    }
    let _guard = gpu_test_lock();

    // Build the same aggregate the runtime JIT builds, so the guard is resolved
    // under the real hiprtc option set (`hiprtc_options_for_arch`).
    let src = grim_backend_rocm::kernels::source_asm::compute_kernel_source();
    assert!(
        src.contains("__builtin_amdgcn_dot4_f32_fp8_fp8"),
        "aggregate lost the dot4 builtin before reaching hiprtc"
    );

    let arch = "gfx1201";
    // A JIT failure is a real failure, not a skip: this aggregate is what ships.
    // (jit_compile_hsaco returns the hiprtc log only on success.)
    let (code, _hiprtc_log) = jit_compile_hsaco(&src, ENTRY, arch)
        .unwrap_or_else(|e| panic!("hiprtc failed for {arch}: {e}"));

    assert!(
        !code.is_empty(),
        "hiprtc returned an empty code object for {arch}"
    );

    // Dump the code object for llvm-objdump. hipCodeObject is a raw ELF-ish
    // bundle; objdump reads it directly.
    let path = std::env::temp_dir().join(format!("grim-agg-{arch}-{}.hsaco", std::process::id()));
    std::fs::write(&path, &code).expect("write code object");
    eprintln!("code object: {} ({} bytes)", path.display(), code.len());

    let objdump = [
        "/opt/rocm/lib/llvm/bin/llvm-objdump",
        "/usr/bin/llvm-objdump",
    ]
    .iter()
    .find(|p| std::path::Path::new(p).exists())
    .map(|p| p.to_string());

    let Some(objdump) = objdump else {
        eprintln!("skip: no llvm-objdump; code object left at {}", path.display());
        return;
    };

    let out = std::process::Command::new(&objdump)
        .args(["-d", &format!("--mcpu={arch}"), path.to_str().unwrap()])
        .output()
        .expect("run llvm-objdump");
    let disasm = String::from_utf8_lossy(&out.stdout).into_owned();

    let hits = disasm.matches(WANTED).count();
    eprintln!("{WANTED} occurrences: {hits}");

    assert!(
        hits > 0,
        "JIT-compiled aggregate contains no {WANTED}. The __has_builtin guard in \
         grim_fdot4_fp8 resolved false under hiprtc, so grim_dot4_fp8_gemv is \
         running its scalar powf-based decode instead of the hardware \
         instruction. Disassembly at {}",
        path.display()
    );
}
