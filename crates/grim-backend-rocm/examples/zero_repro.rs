//! WI root-cause hunt: first-JIT-launch zeroing (validation log 2026-08-23e).
//!
//! For each trial: JIT-compile a trivial kernel (unique source -> cold disk
//! cache unless --warm), load the module, launch it writing `in + 7.0` into
//! an output pre-filled with 41.0, synchronize, read back.
//!
//!   48.0  -> kernel executed            (OK)
//!   41.0  -> kernel did NOT execute     (the zeroing class: buffer keeps
//!                                          its pre-fill / stays zeroed)
//!
//! Variants: --warm reuses one fixed source across trials (compile once,
//! then repeated module loads); --stream own|null chooses the launch stream;
//! --calib runs the SB0 rocBLAS calibration before every trial (profiler
//! interference probe).

use grim_backend_rocm::{
    device::handles::{
        HipMemcpyKind, hipDeviceSynchronize, hipFree, hipMalloc, hipMemcpy, hipModuleGetFunction,
        hipModuleLaunchKernel, hipModuleLoad, hipModuleUnload,
    },
    device::util::DeviceGuard,
    jit_compile_hsaco,
};
use std::ffi::{CString, c_void};

const SRC_TMPL: &str = r#"
// grim_zero_repro probe {TAG}
extern "C" __global__ void grim_zero_repro(float* out, const float* in, int n) {{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) out[i] = in[i] + 7.0f;
}}
"#;

const N: usize = 64;

fn run_trial(
    device_ordinal: i32,
    gcn: &str,
    tag: &str,
    warm_src: Option<&str>,
    own_stream: bool,
) -> bool {
    let source = match warm_src {
        Some(s) => s.to_string(),
        None => SRC_TMPL.replace("{TAG}", tag),
    };
    let entry = "grim_zero_repro";
    let (code, lowered) = match jit_compile_hsaco(&source, entry, gcn) {
        Ok(p) => p,
        Err(e) => {
            println!("  trial {tag}: COMPILE FAIL {e}");
            return false;
        }
    };
    let hsaco_path = std::env::temp_dir().join(format!("zero_repro_{tag}.hsaco"));
    std::fs::write(&hsaco_path, &code).expect("write hsaco");

    unsafe {
        let _guard = DeviceGuard::set(device_ordinal);

        // in = 1.0 everywhere; out pre-filled with 41.0.
        let mut d_in: *mut c_void = std::ptr::null_mut();
        let mut d_out: *mut c_void = std::ptr::null_mut();
        assert_eq!(hipMalloc(&mut d_in, N * 4), 0);
        assert_eq!(hipMalloc(&mut d_out, N * 4), 0);
        let ones = vec![1.0f32; N];
        let fill = vec![41.0f32; N];
        hipMemcpy(d_in, ones.as_ptr() as _, N * 4, HipMemcpyKind::HostToDevice);
        hipMemcpy(
            d_out,
            fill.as_ptr() as _,
            N * 4,
            HipMemcpyKind::HostToDevice,
        );
        hipDeviceSynchronize();

        let path_c = CString::new(hsaco_path.to_str().unwrap()).unwrap();
        let entry_c = CString::new(lowered.as_str()).unwrap();
        let mut module: *mut c_void = std::ptr::null_mut();
        if hipModuleLoad(&mut module, path_c.as_ptr()) != 0 {
            println!("  trial {tag}: MODULE LOAD FAIL");
            return false;
        }
        let mut func: *mut c_void = std::ptr::null_mut();
        if hipModuleGetFunction(&mut func, module, entry_c.as_ptr()) != 0 {
            println!("  trial {tag}: GETFUNCTION FAIL");
            return false;
        }

        let mut stream: *mut c_void = std::ptr::null_mut();
        if own_stream {
            assert_eq!(
                grim_backend_rocm::device::handles::hipStreamCreate(&mut stream),
                0
            );
        }
        let target = if own_stream {
            stream
        } else {
            std::ptr::null_mut()
        };

        let (mut out_p, mut in_p) = (d_out, d_in);
        let mut n_i = N as i32;
        let mut args: [*mut c_void; 3] = [
            &mut out_p as *mut _ as *mut c_void,
            &mut in_p as *mut _ as *mut c_void,
            &mut n_i as *mut _ as *mut c_void,
        ];
        let rc = hipModuleLaunchKernel(
            func,
            1,
            1,
            1,
            64,
            1,
            1,
            0,
            target,
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        hipDeviceSynchronize();

        let mut got = vec![0.0f32; N];
        hipMemcpy(
            got.as_mut_ptr() as _,
            d_out,
            N * 4,
            HipMemcpyKind::DeviceToHost,
        );
        hipDeviceSynchronize();

        // SECOND launch with the same func: does execution recover?
        let rc2 = hipModuleLaunchKernel(
            func,
            1,
            1,
            1,
            64,
            1,
            1,
            0,
            target,
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        hipDeviceSynchronize();
        let mut got2 = vec![0.0f32; N];
        hipMemcpy(
            got2.as_mut_ptr() as _,
            d_out,
            N * 4,
            HipMemcpyKind::DeviceToHost,
        );

        hipFree(d_in);
        hipFree(d_out);
        hipModuleUnload(module);
        if own_stream && !stream.is_null() {
            grim_backend_rocm::device::handles::hipStreamDestroy(stream);
        }

        let ok1 = rc == 0 && got[0] == 8.0;
        let ok2 = rc2 == 0 && got2[0] == 8.0;
        println!(
            "  trial {tag}: launch_rc={rc} out[0]={} | relaunch out[0]={} {}",
            got[0],
            got2[0],
            if ok1 {
                "OK"
            } else if ok2 {
                "FIRST-LAUNCH LOST"
            } else {
                "BOTH DEAD"
            }
        );
        ok1
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let ordinal: i32 = args.next().and_then(|a| a.parse().ok()).unwrap_or(0);
    let gcn = args.next().unwrap_or_else(|| "gfx1201".into());
    let trials: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(6);
    let warm = std::env::var("REPRO_WARM").as_deref() == Ok("1");
    let own_stream = std::env::var("REPRO_OWN_STREAM").as_deref() == Ok("1");

    println!(
        "== zero_repro on {gcn} (ordinal {ordinal}, trials={trials}, warm={warm}, own_stream={own_stream})"
    );

    if std::env::var("REPRO_SAMPLER").as_deref() == Ok("1") {
        // Production sampler kernel with known-value logits: tokens must be
        // < vocab and reproducible for identical seeds; a broken/zeroing
        // launch shows up as token==0 every time or a fault.
        use grim_backend_rocm::{
            BackendDevice as _, memory::storage::RocmStorage, sample_logits_on_device_at,
        };
        let vocab = 65536usize;
        let dev = std::sync::Arc::new(
            grim_backend_rocm::RocmDevice::try_new(ordinal as usize).expect("try_new"),
        );
        let data: Vec<f32> = (0..vocab).map(|i| ((i % 997) as f32) * 0.001).collect();
        for t in 0..trials {
            let st = dev
                .from_cpu(
                    &data,
                    &grim_tensor::Shape::from_slice(&[vocab]),
                    grim_tensor::DType::F32,
                )
                .expect("logits upload");
            let st = st
                .as_any()
                .downcast_ref::<RocmStorage>()
                .expect("rocml storage");
            let tok = sample_logits_on_device_at(
                &dev,
                st,
                vocab,
                0.7,
                0,
                1.0,
                (t as u64) << 32 | 0x1234_5678,
                t as u32,
            );
            match tok {
                Ok(Some(id)) => println!(
                    "  trial {t}: sampled token {id} {}",
                    if id < 997 {
                        "(in weighted support)"
                    } else {
                        "(OUTSIDE support — suspect)"
                    }
                ),
                Ok(None) => println!("  trial {t}: sampler declined (invalid input)"),
                Err(e) => println!("  trial {t}: SAMPLE FAIL {e}"),
            }
        }
        return;
    }

    let fixed = SRC_TMPL.replace("{TAG}", "fixedsrc");
    let calib = std::env::var("REPRO_CALIB").as_deref() == Ok("1");
    let mut ok = 0;
    for t in 0..trials {
        if calib {
            // Interference probe: the engine constructs a profiler (whose
            // WI-SB0 calibration runs rocBLAS + copies on every device)
            // before kernels launch — mirror that here.
            let prof = grim_backend_rocm::CapabilityProfiler::new();
            std::hint::black_box(&prof);
        }
        let tag = format!("{t}");
        if run_trial(
            ordinal,
            gcn.as_str(),
            &tag,
            if warm { Some(fixed.as_str()) } else { None },
            own_stream,
        ) {
            ok += 1;
        }
    }
    println!("== RESULT: {ok}/{trials} first-launches executed");
}
